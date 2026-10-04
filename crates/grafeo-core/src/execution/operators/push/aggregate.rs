//! Push-based aggregate operator (pipeline breaker).

use crate::execution::chunk::{ColumnTypes, DataChunk};
use crate::execution::operators::OperatorError;
use crate::execution::operators::accumulator::{AggregateExpr, AggregateFunction, AggregateState};
use crate::execution::pipeline::{ChunkSizeHint, PushOperator, Sink};
#[cfg(feature = "spill")]
use crate::execution::spill::{PartitionedState, SpillManager};
use crate::execution::vector::ValueVector;
use grafeo_common::types::Value;
use std::collections::HashMap;
#[cfg(feature = "spill")]
use std::io::{Read, Write};
#[cfg(feature = "spill")]
use std::sync::Arc;

/// Creates a new [`AggregateState`] from an [`AggregateExpr`].
fn state_for_expr(expr: &AggregateExpr) -> AggregateState {
    AggregateState::new(
        expr.function,
        expr.distinct,
        expr.percentile,
        expr.separator.as_deref(),
    )
}

/// Updates a single accumulator from a data chunk row, handling bivariate
/// functions, `CountNonNull` null-skipping, and `COUNT(*)`.
fn update_accumulator(
    acc: &mut AggregateState,
    expr: &AggregateExpr,
    chunk: &DataChunk,
    row: usize,
) {
    // Bivariate set functions (COVAR, CORR, REGR_*) need two column values
    if expr.column2.is_some() {
        let y_val = expr
            .column
            .and_then(|col| chunk.column(col).and_then(|c| c.get_value(row)));
        let x_val = expr
            .column2
            .and_then(|col| chunk.column(col).and_then(|c| c.get_value(row)));
        acc.update_bivariate(y_val, x_val);
        return;
    }

    if let Some(col) = expr.column {
        let val = chunk.column(col).and_then(|c| c.get_value(row));
        // CountNonNull must skip null values
        if expr.function == AggregateFunction::CountNonNull
            && matches!(val, None | Some(Value::Null))
        {
            return;
        }
        acc.update(val);
    } else {
        // COUNT(*)
        acc.update(None);
    }
}

/// Hash key for grouping.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct GroupKey(Vec<u64>);

impl GroupKey {
    fn from_row(chunk: &DataChunk, row: usize, group_by: &[usize]) -> Self {
        let hashes: Vec<u64> = group_by
            .iter()
            .map(|&col| {
                chunk
                    .column(col)
                    .and_then(|c| c.get_value(row))
                    .map_or(0, |v| hash_value(&v))
            })
            .collect();
        Self(hashes)
    }
}

fn hash_value(value: &Value) -> u64 {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    let mut hasher = DefaultHasher::new();
    // Discriminant tag prevents cross-type collisions (e.g. Null vs unknown)
    match value {
        Value::Null => 0u8.hash(&mut hasher),
        Value::Bool(b) => {
            1u8.hash(&mut hasher);
            b.hash(&mut hasher);
        }
        Value::Int64(i) => {
            2u8.hash(&mut hasher);
            i.hash(&mut hasher);
        }
        Value::Float64(f) => {
            3u8.hash(&mut hasher);
            f.to_bits().hash(&mut hasher);
        }
        Value::String(s) => {
            4u8.hash(&mut hasher);
            s.hash(&mut hasher);
        }
        Value::Bytes(b) => {
            5u8.hash(&mut hasher);
            b.hash(&mut hasher);
        }
        Value::Timestamp(t) => {
            6u8.hash(&mut hasher);
            t.hash(&mut hasher);
        }
        Value::Date(d) => {
            7u8.hash(&mut hasher);
            d.hash(&mut hasher);
        }
        Value::Time(t) => {
            8u8.hash(&mut hasher);
            t.hash(&mut hasher);
        }
        Value::Duration(d) => {
            9u8.hash(&mut hasher);
            d.hash(&mut hasher);
        }
        Value::ZonedDatetime(zdt) => {
            10u8.hash(&mut hasher);
            zdt.hash(&mut hasher);
        }
        Value::List(list) => {
            11u8.hash(&mut hasher);
            list.len().hash(&mut hasher);
            for elem in list.iter() {
                hash_value(elem).hash(&mut hasher);
            }
        }
        Value::Map(map) => {
            12u8.hash(&mut hasher);
            map.len().hash(&mut hasher);
            // BTreeMap iterates in key order, so hashing is deterministic
            for (k, v) in map.as_ref() {
                k.as_str().hash(&mut hasher);
                hash_value(v).hash(&mut hasher);
            }
        }
        Value::Vector(vec) => {
            13u8.hash(&mut hasher);
            vec.len().hash(&mut hasher);
            for f in vec.iter() {
                f.to_bits().hash(&mut hasher);
            }
        }
        Value::Path { nodes, edges } => {
            14u8.hash(&mut hasher);
            nodes.len().hash(&mut hasher);
            for n in nodes.iter() {
                hash_value(n).hash(&mut hasher);
            }
            for e in edges.iter() {
                hash_value(e).hash(&mut hasher);
            }
        }
        Value::GCounter(map) => {
            15u8.hash(&mut hasher);
            let mut entries: Vec<_> = map.iter().collect();
            entries.sort_by_key(|(k, _)| *k);
            for (k, v) in entries {
                k.hash(&mut hasher);
                v.hash(&mut hasher);
            }
        }
        Value::OnCounter { pos, neg } => {
            16u8.hash(&mut hasher);
            let mut pos_entries: Vec<_> = pos.iter().collect();
            pos_entries.sort_by_key(|(k, _)| *k);
            for (k, v) in pos_entries {
                k.hash(&mut hasher);
                v.hash(&mut hasher);
            }
            let mut neg_entries: Vec<_> = neg.iter().collect();
            neg_entries.sort_by_key(|(k, _)| *k);
            for (k, v) in neg_entries {
                k.hash(&mut hasher);
                v.hash(&mut hasher);
            }
        }
        other => {
            255u8.hash(&mut hasher);
            std::mem::discriminant(other).hash(&mut hasher);
        }
    }
    hasher.finish()
}

/// The output columns: the group keys in their input columns' types (a node
/// or edge stays one, see `ColumnTypes`), then one per aggregate.
fn output_columns(
    group_by: &[usize],
    aggregates: usize,
    input_types: &ColumnTypes,
) -> Vec<ValueVector> {
    group_by
        .iter()
        .map(|&column| match input_types.types().get(column) {
            Some(column_type) => ValueVector::with_capacity(column_type.clone(), 0),
            None => ValueVector::new(),
        })
        .chain((0..aggregates).map(|_| ValueVector::new()))
        .collect()
}

/// Group state with key values and accumulators.
#[derive(Clone)]
struct GroupState {
    key_values: Vec<Value>,
    accumulators: Vec<AggregateState>,
}

/// Push-based aggregate operator.
///
/// This is a pipeline breaker that accumulates all input, groups by key,
/// and produces aggregated output in the finalize phase.
pub struct AggregatePushOperator {
    /// Columns to group by.
    group_by: Vec<usize>,
    /// Aggregate expressions.
    aggregates: Vec<AggregateExpr>,
    /// Group states by hash key.
    groups: HashMap<GroupKey, GroupState>,
    /// Global accumulator (for no GROUP BY).
    global_state: Option<Vec<AggregateState>>,
    /// The column types of the input chunks.
    input_types: ColumnTypes,
}

impl AggregatePushOperator {
    /// Create a new aggregate operator.
    pub fn new(group_by: Vec<usize>, aggregates: Vec<AggregateExpr>) -> Self {
        let global_state = if group_by.is_empty() {
            Some(aggregates.iter().map(state_for_expr).collect())
        } else {
            None
        };

        Self {
            group_by,
            aggregates,
            groups: HashMap::new(),
            global_state,
            input_types: ColumnTypes::default(),
        }
    }

    /// Create a simple global aggregate (no GROUP BY).
    pub fn global(aggregates: Vec<AggregateExpr>) -> Self {
        Self::new(Vec::new(), aggregates)
    }
}

impl PushOperator for AggregatePushOperator {
    fn push(&mut self, chunk: DataChunk, _sink: &mut dyn Sink) -> Result<bool, OperatorError> {
        if chunk.is_empty() {
            return Ok(true);
        }
        self.input_types.add(&chunk);

        for row in chunk.selected_indices() {
            if self.group_by.is_empty() {
                // Global aggregation
                if let Some(ref mut accumulators) = self.global_state {
                    for (acc, expr) in accumulators.iter_mut().zip(&self.aggregates) {
                        update_accumulator(acc, expr, &chunk, row);
                    }
                }
            } else {
                // Group by aggregation
                let key = GroupKey::from_row(&chunk, row, &self.group_by);

                let state = self.groups.entry(key).or_insert_with(|| {
                    let key_values: Vec<Value> = self
                        .group_by
                        .iter()
                        .map(|&col| {
                            chunk
                                .column(col)
                                .and_then(|c| c.get_value(row))
                                .unwrap_or(Value::Null)
                        })
                        .collect();

                    GroupState {
                        key_values,
                        accumulators: self.aggregates.iter().map(state_for_expr).collect(),
                    }
                });

                for (acc, expr) in state.accumulators.iter_mut().zip(&self.aggregates) {
                    update_accumulator(acc, expr, &chunk, row);
                }
            }
        }

        Ok(true)
    }

    fn finalize(&mut self, sink: &mut dyn Sink) -> Result<(), OperatorError> {
        let mut columns = output_columns(&self.group_by, self.aggregates.len(), &self.input_types);

        if self.group_by.is_empty() {
            // Global aggregation - single row output
            if let Some(ref accumulators) = self.global_state {
                for (i, acc) in accumulators.iter().enumerate() {
                    columns[i].push(acc.finalize());
                }
            }
        } else {
            // Group by - one row per group
            for state in self.groups.values() {
                // Output group key columns
                for (i, val) in state.key_values.iter().enumerate() {
                    columns[i].push(val.clone());
                }

                // Output aggregate results
                for (i, acc) in state.accumulators.iter().enumerate() {
                    columns[self.group_by.len() + i].push(acc.finalize());
                }
            }
        }

        if !columns.is_empty() && !columns[0].is_empty() {
            let chunk = DataChunk::new(columns);
            sink.consume(chunk)?;
        }

        Ok(())
    }

    fn preferred_chunk_size(&self) -> ChunkSizeHint {
        ChunkSizeHint::Default
    }

    fn name(&self) -> &'static str {
        "AggregatePush"
    }
}

/// Default spill threshold for aggregates (number of groups).
#[cfg(feature = "spill")]
pub const DEFAULT_AGGREGATE_SPILL_THRESHOLD: usize = 50_000;

/// Minimum number of groups before memory-pressure spilling can trigger.
///
/// Prevents "noisy neighbor" scenarios where a tiny aggregate buffer gets
/// spilled because unrelated subsystems consumed memory.
#[cfg(feature = "spill")]
const AGGREGATE_MIN_BUFFER_GROUPS: usize = 500;

/// Tag bytes for aggregate state variants used during spill serialization.
///
/// Each tag identifies both the aggregate function AND how to reconstruct
/// the accumulator state so it can continue receiving updates after reload.
#[cfg(feature = "spill")]
mod spill_tag {
    pub const COUNT: u8 = 0;
    pub const SUM_INT: u8 = 1;
    pub const SUM_FLOAT: u8 = 2;
    pub const AVG: u8 = 3;
    pub const MIN: u8 = 4;
    pub const MAX: u8 = 5;
    pub const FIRST: u8 = 6;
    pub const LAST: u8 = 7;
    pub const COLLECT: u8 = 8;
    /// Fallback: stores finalized value only, cannot resume accumulation.
    pub const FINALIZED: u8 = 255;
}

/// Serializes a `GroupState` to bytes.
///
/// Each accumulator is serialized with a tag byte indicating the state variant
/// followed by the internal fields needed to reconstruct a resumable state.
/// For complex variants (StdDev, percentiles, bivariate, etc.) the finalized
/// value is stored instead, since those are rare in spill scenarios.
#[cfg(feature = "spill")]
fn serialize_group_state(state: &GroupState, w: &mut dyn Write) -> std::io::Result<()> {
    use crate::execution::spill::serialize_value;

    // Write key values
    w.write_all(&(state.key_values.len() as u64).to_le_bytes())?;
    for val in &state.key_values {
        serialize_value(val, w)?;
    }

    // Write accumulators with tag bytes
    w.write_all(&(state.accumulators.len() as u64).to_le_bytes())?;
    for acc in &state.accumulators {
        match acc {
            AggregateState::Count(n) => {
                w.write_all(&[spill_tag::COUNT])?;
                w.write_all(&n.to_le_bytes())?;
            }
            AggregateState::SumInt(sum, count) => {
                w.write_all(&[spill_tag::SUM_INT])?;
                w.write_all(&sum.to_le_bytes())?;
                w.write_all(&count.to_le_bytes())?;
            }
            AggregateState::SumFloat(sum, _comp, count) => {
                w.write_all(&[spill_tag::SUM_FLOAT])?;
                w.write_all(&sum.to_le_bytes())?;
                w.write_all(&count.to_le_bytes())?;
            }
            AggregateState::Avg(sum, count) => {
                w.write_all(&[spill_tag::AVG])?;
                w.write_all(&sum.to_le_bytes())?;
                w.write_all(&count.to_le_bytes())?;
            }
            // DISTINCT variants track a HashSet that can't be serialized compactly.
            // Serialize as finalized to avoid dropping distinct semantics.
            AggregateState::CountDistinct(..)
            | AggregateState::SumIntDistinct(..)
            | AggregateState::SumFloatDistinct(..)
            | AggregateState::AvgDistinct(..)
            | AggregateState::CollectDistinct(..)
            | AggregateState::GroupConcatDistinct(..) => {
                w.write_all(&[spill_tag::FINALIZED])?;
                serialize_value(&acc.finalize(), w)?;
            }
            AggregateState::Min(val) => {
                w.write_all(&[spill_tag::MIN])?;
                serialize_value(&val.clone().unwrap_or(Value::Null), w)?;
            }
            AggregateState::Max(val) => {
                w.write_all(&[spill_tag::MAX])?;
                serialize_value(&val.clone().unwrap_or(Value::Null), w)?;
            }
            AggregateState::First(val) => {
                w.write_all(&[spill_tag::FIRST])?;
                serialize_value(&val.clone().unwrap_or(Value::Null), w)?;
            }
            AggregateState::Last(val) => {
                w.write_all(&[spill_tag::LAST])?;
                serialize_value(&val.clone().unwrap_or(Value::Null), w)?;
            }
            AggregateState::Collect(list) => {
                w.write_all(&[spill_tag::COLLECT])?;
                w.write_all(&(list.len() as u64).to_le_bytes())?;
                for val in list {
                    serialize_value(val, w)?;
                }
            }
            // Complex states: serialize finalized value as fallback
            _ => {
                w.write_all(&[spill_tag::FINALIZED])?;
                serialize_value(&acc.finalize(), w)?;
            }
        }
    }

    Ok(())
}

/// Deserializes a `GroupState` from bytes.
///
/// Reconstructs the correct `AggregateState` variant from the tag byte so that
/// reloaded groups can continue accumulating rows. Common variants (Count,
/// SumInt, SumFloat, Avg, Min, Max, First, Last, Collect) are fully resumable.
/// Rare/complex variants fall back to `Frozen(val)`.
#[cfg(feature = "spill")]
fn deserialize_group_state(r: &mut dyn Read) -> std::io::Result<GroupState> {
    use crate::execution::spill::deserialize_value;

    // Read key values
    let mut len_buf = [0u8; 8];
    r.read_exact(&mut len_buf)?;
    // reason: deserialized counts are bounded by available data
    #[allow(clippy::cast_possible_truncation)]
    let num_keys = u64::from_le_bytes(len_buf) as usize;

    let mut key_values = Vec::with_capacity(num_keys);
    for _ in 0..num_keys {
        key_values.push(deserialize_value(r)?);
    }

    // Read accumulators with tag-based reconstruction
    r.read_exact(&mut len_buf)?;
    // reason: deserialized counts are bounded by available data
    #[allow(clippy::cast_possible_truncation)]
    let num_accumulators = u64::from_le_bytes(len_buf) as usize;

    let mut accumulators = Vec::with_capacity(num_accumulators);
    for _ in 0..num_accumulators {
        let mut tag = [0u8; 1];
        r.read_exact(&mut tag)?;

        let state = match tag[0] {
            spill_tag::COUNT => {
                let mut buf = [0u8; 8];
                r.read_exact(&mut buf)?;
                AggregateState::Count(i64::from_le_bytes(buf))
            }
            spill_tag::SUM_INT => {
                let mut buf = [0u8; 8];
                r.read_exact(&mut buf)?;
                let sum = i64::from_le_bytes(buf);
                r.read_exact(&mut buf)?;
                let count = i64::from_le_bytes(buf);
                AggregateState::SumInt(sum, count)
            }
            spill_tag::SUM_FLOAT => {
                let mut buf = [0u8; 8];
                r.read_exact(&mut buf)?;
                let sum = f64::from_le_bytes(buf);
                r.read_exact(&mut buf)?;
                let count = i64::from_le_bytes(buf);
                // Reset Kahan compensation to zero; minor precision loss is acceptable
                AggregateState::SumFloat(sum, 0.0, count)
            }
            spill_tag::AVG => {
                let mut buf = [0u8; 8];
                r.read_exact(&mut buf)?;
                let sum = f64::from_le_bytes(buf);
                r.read_exact(&mut buf)?;
                let count = i64::from_le_bytes(buf);
                AggregateState::Avg(sum, count)
            }
            spill_tag::MIN => {
                let val = deserialize_value(r)?;
                let opt = if matches!(val, Value::Null) {
                    None
                } else {
                    Some(val)
                };
                AggregateState::Min(opt)
            }
            spill_tag::MAX => {
                let val = deserialize_value(r)?;
                let opt = if matches!(val, Value::Null) {
                    None
                } else {
                    Some(val)
                };
                AggregateState::Max(opt)
            }
            spill_tag::FIRST => {
                let val = deserialize_value(r)?;
                let opt = if matches!(val, Value::Null) {
                    None
                } else {
                    Some(val)
                };
                AggregateState::First(opt)
            }
            spill_tag::LAST => {
                let val = deserialize_value(r)?;
                let opt = if matches!(val, Value::Null) {
                    None
                } else {
                    Some(val)
                };
                AggregateState::Last(opt)
            }
            spill_tag::COLLECT => {
                let mut buf = [0u8; 8];
                r.read_exact(&mut buf)?;
                // reason: deserialized lengths are bounded by available data
                #[allow(clippy::cast_possible_truncation)]
                let len = u64::from_le_bytes(buf) as usize;
                let mut list = Vec::with_capacity(len);
                for _ in 0..len {
                    list.push(deserialize_value(r)?);
                }
                AggregateState::Collect(list)
            }
            _ => {
                let val = deserialize_value(r)?;
                AggregateState::Frozen(val)
            }
        };

        accumulators.push(state);
    }

    Ok(GroupState {
        key_values,
        accumulators,
    })
}

/// Push-based aggregate operator with spilling support.
///
/// Uses partitioned hash table that can spill cold partitions to disk
/// when memory pressure is high.
///
/// Two spill modes are supported:
///
/// 1. **Memory-aware** (when constructed with `with_memory_context`): registers
///    as a `MemoryConsumer` with the `BufferManager` and spills when system
///    pressure is High/Critical or when eviction is explicitly requested.
///
/// 2. **Row-count fallback** (when constructed with `new` or `with_spilling`):
///    spills when `groups.len() >= spill_threshold` (default 50K groups).
#[cfg(feature = "spill")]
pub struct SpillableAggregatePushOperator {
    /// Columns to group by.
    group_by: Vec<usize>,
    /// Aggregate expressions.
    aggregates: Vec<AggregateExpr>,
    /// Spill manager (None = no spilling, used by row-count fallback mode).
    spill_manager: Option<Arc<SpillManager>>,
    /// Partitioned groups (used when spilling is enabled).
    partitioned_groups: Option<PartitionedState<GroupState>>,
    /// Non-partitioned groups (used when spilling is disabled).
    groups: HashMap<GroupKey, GroupState>,
    /// Global accumulator (for no GROUP BY).
    global_state: Option<Vec<AggregateState>>,
    /// Spill threshold (number of groups, used by fallback mode).
    spill_threshold: usize,
    /// Whether we've switched to partitioned mode.
    using_partitioned: bool,
    /// Memory context for pressure-aware spilling.
    memory_ctx: Option<crate::execution::memory::OperatorMemoryContext>,
    /// Shared state with the registered MemoryConsumer adapter.
    spill_state: Option<std::sync::Arc<super::spill_state::OperatorSpillState>>,
    /// Running total of estimated group memory in bytes (incremental tracking).
    estimated_bytes: usize,
    /// The column types of the input chunks.
    input_types: ColumnTypes,
}

#[cfg(feature = "spill")]
impl SpillableAggregatePushOperator {
    /// Create a new spillable aggregate operator (row-count fallback mode).
    pub fn new(group_by: Vec<usize>, aggregates: Vec<AggregateExpr>) -> Self {
        let global_state = if group_by.is_empty() {
            Some(aggregates.iter().map(state_for_expr).collect())
        } else {
            None
        };

        Self {
            group_by,
            aggregates,
            spill_manager: None,
            partitioned_groups: None,
            groups: HashMap::new(),
            global_state,
            spill_threshold: DEFAULT_AGGREGATE_SPILL_THRESHOLD,
            using_partitioned: false,
            memory_ctx: None,
            spill_state: None,
            estimated_bytes: 0,
            input_types: ColumnTypes::default(),
        }
    }

    /// Create a spillable aggregate operator with spilling enabled (row-count mode).
    pub fn with_spilling(
        group_by: Vec<usize>,
        aggregates: Vec<AggregateExpr>,
        manager: Arc<SpillManager>,
        threshold: usize,
    ) -> Self {
        let global_state = if group_by.is_empty() {
            Some(aggregates.iter().map(state_for_expr).collect())
        } else {
            None
        };

        let partitioned = PartitionedState::new(
            Arc::clone(&manager),
            256, // Number of partitions
            serialize_group_state,
            deserialize_group_state,
        );

        Self {
            group_by,
            aggregates,
            spill_manager: Some(manager),
            partitioned_groups: Some(partitioned),
            groups: HashMap::new(),
            global_state,
            spill_threshold: threshold,
            using_partitioned: true,
            memory_ctx: None,
            spill_state: None,
            estimated_bytes: 0,
            input_types: ColumnTypes::default(),
        }
    }

    /// Create a spillable aggregate operator with memory-aware spilling.
    ///
    /// Registers as a `MemoryConsumer` with the `BufferManager` and spills
    /// based on system memory pressure rather than group count thresholds.
    pub fn with_memory_context(
        group_by: Vec<usize>,
        aggregates: Vec<AggregateExpr>,
        ctx: crate::execution::memory::OperatorMemoryContext,
    ) -> Self {
        use super::spill_state::{OperatorConsumerAdapter, OperatorSpillState};

        let global_state = if group_by.is_empty() {
            Some(aggregates.iter().map(state_for_expr).collect())
        } else {
            None
        };

        let state = std::sync::Arc::new(OperatorSpillState::new(
            "SpillableAggregatePush".to_string(),
        ));
        let adapter =
            std::sync::Arc::new(OperatorConsumerAdapter::new(std::sync::Arc::clone(&state)));
        ctx.register_consumer(adapter);

        // Pre-create partitioned state using the spill manager from memory context
        let partitioned = PartitionedState::new(
            std::sync::Arc::clone(ctx.spill_manager()),
            256,
            serialize_group_state,
            deserialize_group_state,
        );

        Self {
            group_by,
            aggregates,
            spill_manager: None,
            partitioned_groups: Some(partitioned),
            groups: HashMap::new(),
            global_state,
            spill_threshold: DEFAULT_AGGREGATE_SPILL_THRESHOLD,
            using_partitioned: true,
            memory_ctx: Some(ctx),
            spill_state: Some(state),
            estimated_bytes: 0,
            input_types: ColumnTypes::default(),
        }
    }

    /// Create a simple global aggregate (no GROUP BY).
    pub fn global(aggregates: Vec<AggregateExpr>) -> Self {
        Self::new(Vec::new(), aggregates)
    }

    /// Sets the spill threshold (row-count fallback mode).
    pub fn with_threshold(mut self, threshold: usize) -> Self {
        self.spill_threshold = threshold;
        self
    }

    /// Checks whether spilling should occur and performs it if needed.
    fn maybe_spill(&mut self) -> Result<(), OperatorError> {
        if self.global_state.is_some() {
            // Global aggregation doesn't need spilling
            return Ok(());
        }

        if self.spill_state.is_some() {
            // Memory-aware mode
            self.maybe_spill_memory_aware()
        } else {
            // Row-count fallback mode
            self.maybe_spill_row_count()
        }
    }

    /// Memory-aware spill decision: check eviction flag and system pressure.
    fn maybe_spill_memory_aware(&mut self) -> Result<(), OperatorError> {
        let should_spill = if let Some(ref state) = self.spill_state {
            let eviction = state.take_eviction_request().is_some();
            let pressure = self.memory_ctx.as_ref().map_or(false, |c| c.should_spill());

            // Determine current group count for minimum buffer guard
            let group_count = if let Some(ref partitioned) = self.partitioned_groups {
                partitioned.total_size()
            } else {
                self.groups.len()
            };
            let above_minimum = group_count >= AGGREGATE_MIN_BUFFER_GROUPS;

            (eviction || pressure) && above_minimum
        } else {
            false
        };

        if should_spill && let Some(ref mut partitioned) = self.partitioned_groups {
            partitioned
                .spill_largest()
                .map_err(|e| OperatorError::Execution(e.to_string()))?;
        }

        Ok(())
    }

    /// Row-count fallback spill decision.
    fn maybe_spill_row_count(&mut self) -> Result<(), OperatorError> {
        // If using partitioned state, check if we need to spill
        if let Some(ref mut partitioned) = self.partitioned_groups {
            if partitioned.total_size() >= self.spill_threshold {
                partitioned
                    .spill_largest()
                    .map_err(|e| OperatorError::Execution(e.to_string()))?;
            }
        } else if self.groups.len() >= self.spill_threshold {
            // Not using partitioned state yet, but reached threshold
            // If spilling is configured, switch to partitioned mode
            if let Some(ref manager) = self.spill_manager {
                let mut partitioned = PartitionedState::new(
                    Arc::clone(manager),
                    256,
                    serialize_group_state,
                    deserialize_group_state,
                );

                // Move existing groups to partitioned state
                for (_key, state) in self.groups.drain() {
                    partitioned
                        .insert(state.key_values.clone(), state)
                        .map_err(|e| OperatorError::Execution(e.to_string()))?;
                }

                self.partitioned_groups = Some(partitioned);
                self.using_partitioned = true;
            }
        }

        Ok(())
    }

    /// Unregisters this operator's consumer from the BufferManager.
    fn unregister_consumer(&self) {
        if let (Some(ctx), Some(state)) = (&self.memory_ctx, &self.spill_state) {
            ctx.unregister_consumer(state.name());
        }
    }
}

#[cfg(feature = "spill")]
impl PushOperator for SpillableAggregatePushOperator {
    fn push(&mut self, chunk: DataChunk, _sink: &mut dyn Sink) -> Result<bool, OperatorError> {
        if chunk.is_empty() {
            return Ok(true);
        }
        self.input_types.add(&chunk);

        for row in chunk.selected_indices() {
            if self.group_by.is_empty() {
                // Global aggregation - same as non-spillable
                if let Some(ref mut accumulators) = self.global_state {
                    for (acc, expr) in accumulators.iter_mut().zip(&self.aggregates) {
                        update_accumulator(acc, expr, &chunk, row);
                    }
                }
            } else if self.using_partitioned {
                // Use partitioned state
                if let Some(ref mut partitioned) = self.partitioned_groups {
                    let key_values: Vec<Value> = self
                        .group_by
                        .iter()
                        .map(|&col| {
                            chunk
                                .column(col)
                                .and_then(|c| c.get_value(row))
                                .unwrap_or(Value::Null)
                        })
                        .collect();

                    let aggregates = &self.aggregates;
                    let state = partitioned
                        .get_or_insert_with(key_values.clone(), || GroupState {
                            key_values: key_values.clone(),
                            accumulators: aggregates.iter().map(state_for_expr).collect(),
                        })
                        .map_err(|e| OperatorError::Execution(e.to_string()))?;

                    for (acc, expr) in state.accumulators.iter_mut().zip(&self.aggregates) {
                        update_accumulator(acc, expr, &chunk, row);
                    }
                }
            } else {
                // Use regular hash map
                let key = GroupKey::from_row(&chunk, row, &self.group_by);

                let state = self.groups.entry(key).or_insert_with(|| {
                    let key_values: Vec<Value> = self
                        .group_by
                        .iter()
                        .map(|&col| {
                            chunk
                                .column(col)
                                .and_then(|c| c.get_value(row))
                                .unwrap_or(Value::Null)
                        })
                        .collect();

                    GroupState {
                        key_values,
                        accumulators: self.aggregates.iter().map(state_for_expr).collect(),
                    }
                });

                for (acc, expr) in state.accumulators.iter_mut().zip(&self.aggregates) {
                    update_accumulator(acc, expr, &chunk, row);
                }
            }
        }

        // Update memory consumer usage estimate
        if let Some(ref spill_state) = self.spill_state {
            // Estimate: each group has key_values + accumulators
            // Rough sizing: key columns * Value size + num_aggregates * 64 bytes per accumulator
            let group_count = if self.using_partitioned {
                self.partitioned_groups
                    .as_ref()
                    .map_or(0, |p| p.total_size())
            } else {
                self.groups.len()
            };
            let key_size = self.group_by.len() * std::mem::size_of::<Value>();
            let acc_size = self.aggregates.len() * 64; // rough accumulator size
            self.estimated_bytes = group_count * (key_size + acc_size + 48);
            spill_state.set_usage(self.estimated_bytes);
        }

        // Check if we need to spill
        self.maybe_spill()?;

        Ok(true)
    }

    fn finalize(&mut self, sink: &mut dyn Sink) -> Result<(), OperatorError> {
        let mut columns = output_columns(&self.group_by, self.aggregates.len(), &self.input_types);

        if self.group_by.is_empty() {
            // Global aggregation - single row output
            if let Some(ref accumulators) = self.global_state {
                for (i, acc) in accumulators.iter().enumerate() {
                    columns[i].push(acc.finalize());
                }
            }
        } else if self.using_partitioned {
            // Drain partitioned state
            if let Some(ref mut partitioned) = self.partitioned_groups {
                let groups = partitioned
                    .drain_all()
                    .map_err(|e| OperatorError::Execution(e.to_string()))?;

                for (_key, state) in groups {
                    // Output group key columns
                    for (i, val) in state.key_values.iter().enumerate() {
                        columns[i].push(val.clone());
                    }

                    // Output aggregate results
                    for (i, acc) in state.accumulators.iter().enumerate() {
                        columns[self.group_by.len() + i].push(acc.finalize());
                    }
                }
            }
        } else {
            // Group by using regular hash map - one row per group
            for state in self.groups.values() {
                // Output group key columns
                for (i, val) in state.key_values.iter().enumerate() {
                    columns[i].push(val.clone());
                }

                // Output aggregate results
                for (i, acc) in state.accumulators.iter().enumerate() {
                    columns[self.group_by.len() + i].push(acc.finalize());
                }
            }
        }

        // Unregister consumer before emitting results
        self.unregister_consumer();

        if !columns.is_empty() && !columns[0].is_empty() {
            let chunk = DataChunk::new(columns);
            sink.consume(chunk)?;
        }

        Ok(())
    }

    fn preferred_chunk_size(&self) -> ChunkSizeHint {
        ChunkSizeHint::Default
    }

    fn name(&self) -> &'static str {
        "SpillableAggregatePush"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::execution::operators::accumulator::AggregateFunction;
    use crate::execution::sink::CollectorSink;

    fn create_test_chunk(values: &[i64]) -> DataChunk {
        let v: Vec<Value> = values.iter().map(|&i| Value::Int64(i)).collect();
        let vector = ValueVector::from_values(&v);
        DataChunk::new(vec![vector])
    }

    fn create_two_column_chunk(col1: &[i64], col2: &[i64]) -> DataChunk {
        let v1: Vec<Value> = col1.iter().map(|&i| Value::Int64(i)).collect();
        let v2: Vec<Value> = col2.iter().map(|&i| Value::Int64(i)).collect();
        DataChunk::new(vec![
            ValueVector::from_values(&v1),
            ValueVector::from_values(&v2),
        ])
    }

    /// A node group key keeps its column type.
    #[test]
    fn group_keys_keep_their_column_types() {
        use crate::execution::chunk::DataChunkBuilder;
        use grafeo_common::types::{LogicalType, NodeId};

        let mut builder = DataChunkBuilder::new(&[LogicalType::Node]);
        for node in [101, 101, 102] {
            builder
                .column_mut(0)
                .unwrap()
                .push_node_id(NodeId::new(node));
            builder.advance_row();
        }
        let mut agg = AggregatePushOperator::new(vec![0], vec![AggregateExpr::count_star()]);
        let mut sink = CollectorSink::new();
        agg.push(builder.finish(), &mut sink).unwrap();
        agg.finalize(&mut sink).unwrap();

        let chunks = sink.into_chunks();
        assert_eq!(chunks.len(), 1);
        let chunk = &chunks[0];
        assert_eq!(chunk.column_types()[0], LogicalType::Node);
        let mut groups: Vec<(u64, Option<Value>)> = chunk
            .selected_indices()
            .map(|row| {
                (
                    chunk.column(0).unwrap().get_node_id(row).unwrap().as_u64(),
                    chunk.column(1).unwrap().get_value(row),
                )
            })
            .collect();
        groups.sort_by_key(|(node, _)| *node);
        assert_eq!(
            groups,
            [(101, Some(Value::Int64(2))), (102, Some(Value::Int64(1)))]
        );
    }

    #[test]
    fn test_global_count() {
        let mut agg = AggregatePushOperator::global(vec![AggregateExpr::count_star()]);
        let mut sink = CollectorSink::new();

        agg.push(create_test_chunk(&[1, 2, 3, 4, 5]), &mut sink)
            .unwrap();
        agg.finalize(&mut sink).unwrap();

        let chunks = sink.into_chunks();
        assert_eq!(chunks.len(), 1);
        assert_eq!(
            chunks[0].column(0).unwrap().get_value(0),
            Some(Value::Int64(5))
        );
    }

    #[test]
    fn test_global_sum() {
        let mut agg = AggregatePushOperator::global(vec![AggregateExpr::sum(0)]);
        let mut sink = CollectorSink::new();

        agg.push(create_test_chunk(&[1, 2, 3, 4, 5]), &mut sink)
            .unwrap();
        agg.finalize(&mut sink).unwrap();

        let chunks = sink.into_chunks();
        // AggregateState preserves integer type for SUM of integers
        assert_eq!(
            chunks[0].column(0).unwrap().get_value(0),
            Some(Value::Int64(15))
        );
    }

    #[test]
    fn test_global_min_max() {
        let mut agg =
            AggregatePushOperator::global(vec![AggregateExpr::min(0), AggregateExpr::max(0)]);
        let mut sink = CollectorSink::new();

        agg.push(create_test_chunk(&[3, 1, 4, 1, 5, 9, 2, 6]), &mut sink)
            .unwrap();
        agg.finalize(&mut sink).unwrap();

        let chunks = sink.into_chunks();
        assert_eq!(
            chunks[0].column(0).unwrap().get_value(0),
            Some(Value::Int64(1))
        );
        assert_eq!(
            chunks[0].column(1).unwrap().get_value(0),
            Some(Value::Int64(9))
        );
    }

    #[test]
    fn test_group_by_sum() {
        // Group by column 0, sum column 1
        let mut agg = AggregatePushOperator::new(vec![0], vec![AggregateExpr::sum(1)]);
        let mut sink = CollectorSink::new();

        // Group 1: 10, 20 (sum=30), Group 2: 30, 40 (sum=70)
        agg.push(
            create_two_column_chunk(&[1, 1, 2, 2], &[10, 20, 30, 40]),
            &mut sink,
        )
        .unwrap();
        agg.finalize(&mut sink).unwrap();

        let chunks = sink.into_chunks();
        assert_eq!(chunks[0].len(), 2); // 2 groups
    }

    #[test]
    #[cfg(feature = "spill")]
    fn test_spillable_aggregate_no_spill() {
        // When threshold is not reached, should work like normal aggregate
        let mut agg = SpillableAggregatePushOperator::new(vec![0], vec![AggregateExpr::sum(1)])
            .with_threshold(100);
        let mut sink = CollectorSink::new();

        agg.push(
            create_two_column_chunk(&[1, 1, 2, 2], &[10, 20, 30, 40]),
            &mut sink,
        )
        .unwrap();
        agg.finalize(&mut sink).unwrap();

        let chunks = sink.into_chunks();
        assert_eq!(chunks[0].len(), 2); // 2 groups
    }

    #[test]
    #[cfg(feature = "spill")]
    fn test_spillable_aggregate_with_spilling() {
        use tempfile::TempDir;

        let temp_dir = TempDir::new().unwrap();
        let manager = Arc::new(SpillManager::new(temp_dir.path()).unwrap());

        // Set very low threshold to force spilling
        let mut agg = SpillableAggregatePushOperator::with_spilling(
            vec![0],
            vec![AggregateExpr::sum(1)],
            manager,
            3, // Spill after 3 groups
        );
        let mut sink = CollectorSink::new();

        // Create 10 different groups
        for i in 0..10 {
            let chunk = create_two_column_chunk(&[i], &[i * 10]);
            agg.push(chunk, &mut sink).unwrap();
        }
        agg.finalize(&mut sink).unwrap();

        let chunks = sink.into_chunks();
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].len(), 10); // 10 groups

        // Verify sums are correct (AggregateState preserves Int64 for integer sums)
        let mut sums: Vec<i64> = Vec::new();
        for i in 0..chunks[0].len() {
            if let Some(Value::Int64(sum)) = chunks[0].column(1).unwrap().get_value(i) {
                sums.push(sum);
            }
        }
        sums.sort_unstable();
        assert_eq!(sums, vec![0, 10, 20, 30, 40, 50, 60, 70, 80, 90]);
    }

    #[test]
    #[cfg(feature = "spill")]
    fn test_spillable_aggregate_global() {
        // Global aggregation shouldn't be affected by spilling
        let mut agg = SpillableAggregatePushOperator::global(vec![AggregateExpr::count_star()]);
        let mut sink = CollectorSink::new();

        agg.push(create_test_chunk(&[1, 2, 3, 4, 5]), &mut sink)
            .unwrap();
        agg.finalize(&mut sink).unwrap();

        let chunks = sink.into_chunks();
        assert_eq!(chunks.len(), 1);
        assert_eq!(
            chunks[0].column(0).unwrap().get_value(0),
            Some(Value::Int64(5))
        );
    }

    #[test]
    #[cfg(feature = "spill")]
    fn test_spillable_aggregate_many_groups() {
        use tempfile::TempDir;

        let temp_dir = TempDir::new().unwrap();
        let manager = Arc::new(SpillManager::new(temp_dir.path()).unwrap());

        let mut agg = SpillableAggregatePushOperator::with_spilling(
            vec![0],
            vec![AggregateExpr::count_star()],
            manager,
            10, // Very low threshold
        );
        let mut sink = CollectorSink::new();

        // Create 100 different groups
        for i in 0..100 {
            let chunk = create_test_chunk(&[i]);
            agg.push(chunk, &mut sink).unwrap();
        }
        agg.finalize(&mut sink).unwrap();

        let chunks = sink.into_chunks();
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].len(), 100); // 100 groups

        // Each group should have count = 1
        for i in 0..100 {
            if let Some(Value::Int64(count)) = chunks[0].column(1).unwrap().get_value(i) {
                assert_eq!(count, 1);
            }
        }
    }

    // ---------------------------------------------------------------
    // hash_value coverage for all Value variants
    // ---------------------------------------------------------------

    #[test]
    fn hash_value_null() {
        let h = hash_value(&Value::Null);
        assert_ne!(h, 0); // hasher produces non-zero for Null discriminant
    }

    #[test]
    fn hash_value_bool() {
        let t = hash_value(&Value::Bool(true));
        let f = hash_value(&Value::Bool(false));
        assert_ne!(t, f);
    }

    #[test]
    fn hash_value_int64() {
        let a = hash_value(&Value::Int64(42));
        let b = hash_value(&Value::Int64(43));
        assert_ne!(a, b);
    }

    #[test]
    fn hash_value_float64() {
        let a = hash_value(&Value::Float64(19.88));
        let b = hash_value(&Value::Float64(3.19));
        assert_ne!(a, b);
    }

    #[test]
    fn hash_value_string() {
        let a = hash_value(&Value::String("hello".into()));
        let b = hash_value(&Value::String("world".into()));
        assert_ne!(a, b);
    }

    #[test]
    fn hash_value_bytes() {
        let a = hash_value(&Value::Bytes(vec![1, 2, 3].into()));
        let b = hash_value(&Value::Bytes(vec![4, 5, 6].into()));
        assert_ne!(a, b);
    }

    #[test]
    fn hash_value_list() {
        let a = hash_value(&Value::List(vec![Value::Int64(1), Value::Int64(2)].into()));
        let b = hash_value(&Value::List(vec![Value::Int64(3)].into()));
        assert_ne!(a, b);
    }

    #[test]
    fn hash_value_map() {
        use grafeo_common::types::PropertyKey;
        use std::collections::BTreeMap;
        use std::sync::Arc;
        let mut map = BTreeMap::new();
        map.insert(PropertyKey::new("key"), Value::Int64(42));
        let h = hash_value(&Value::Map(Arc::new(map)));
        assert_ne!(h, 0);
    }

    #[test]
    fn hash_value_vector() {
        let h = hash_value(&Value::Vector(vec![1.0, 2.0, 3.0].into()));
        assert_ne!(h, 0);
    }

    #[test]
    fn hash_value_path() {
        let h = hash_value(&Value::Path {
            nodes: vec![Value::Int64(1), Value::Int64(2)].into(),
            edges: vec![Value::Int64(10)].into(),
        });
        assert_ne!(h, 0);
    }

    #[test]
    fn hash_value_gcounter() {
        use std::sync::Arc;
        let mut map = std::collections::HashMap::new();
        map.insert("replica1".to_string(), 10u64);
        let h = hash_value(&Value::GCounter(Arc::new(map)));
        assert_ne!(h, 0);
    }

    #[test]
    fn hash_value_on_counter() {
        use std::sync::Arc;
        let mut pos = std::collections::HashMap::new();
        pos.insert("replica1".to_string(), 10u64);
        let neg = std::collections::HashMap::new();
        let h = hash_value(&Value::OnCounter {
            pos: Arc::new(pos),
            neg: Arc::new(neg),
        });
        assert_ne!(h, 0);
    }

    #[test]
    fn hash_value_timestamp() {
        use grafeo_common::types::Timestamp;
        let h = hash_value(&Value::Timestamp(Timestamp::from_micros(1_700_000_000_000)));
        assert_ne!(h, 0);
    }

    #[test]
    fn hash_value_date() {
        use grafeo_common::types::Date;
        let h = hash_value(&Value::Date(Date::from_days(19000)));
        assert_ne!(h, 0);
    }

    #[test]
    fn hash_value_time() {
        use grafeo_common::types::Time;
        let h = hash_value(&Value::Time(Time::from_hms(12, 0, 0).unwrap()));
        assert_ne!(h, 0);
    }

    #[test]
    fn hash_value_duration() {
        use grafeo_common::types::Duration;
        let h = hash_value(&Value::Duration(Duration::from_days(1)));
        assert_ne!(h, 0);
    }

    #[test]
    fn hash_value_zoned_datetime() {
        use grafeo_common::types::{Timestamp, ZonedDatetime};
        let zdt =
            ZonedDatetime::from_timestamp_offset(Timestamp::from_micros(1_700_000_000_000), 3600);
        let h = hash_value(&Value::ZonedDatetime(zdt));
        assert_ne!(h, 0);
    }

    // ---------------------------------------------------------------
    // AggregateState in push context: advanced functions now work
    // ---------------------------------------------------------------

    #[test]
    fn aggregate_state_last_returns_last_value() {
        let mut state = AggregateState::new(AggregateFunction::Last, false, None, None);
        state.update(Some(Value::Int64(10)));
        state.update(Some(Value::Int64(20)));
        assert_eq!(state.finalize(), Value::Int64(20));
    }

    #[test]
    fn aggregate_state_collect_returns_list() {
        let mut state = AggregateState::new(AggregateFunction::Collect, false, None, None);
        state.update(Some(Value::Int64(1)));
        state.update(Some(Value::Int64(2)));
        assert_eq!(
            state.finalize(),
            Value::List(vec![Value::Int64(1), Value::Int64(2)].into())
        );
    }

    #[test]
    fn aggregate_state_stdev_returns_value() {
        let mut state = AggregateState::new(AggregateFunction::StdDev, false, None, None);
        state.update(Some(Value::Float64(2.0)));
        state.update(Some(Value::Float64(4.0)));
        state.update(Some(Value::Float64(6.0)));
        let result = state.finalize();
        assert!(matches!(result, Value::Float64(_)));
    }

    #[test]
    fn aggregate_state_first_returns_first_value() {
        let mut state = AggregateState::new(AggregateFunction::First, false, None, None);
        state.update(Some(Value::Int64(10)));
        state.update(Some(Value::Int64(20)));
        assert_eq!(state.finalize(), Value::Int64(10));
    }

    #[test]
    fn aggregate_state_avg_empty_returns_null() {
        let state = AggregateState::new(AggregateFunction::Avg, false, None, None);
        assert_eq!(state.finalize(), Value::Null);
    }

    #[test]
    fn aggregate_state_sum_empty_returns_null() {
        let state = AggregateState::new(AggregateFunction::Sum, false, None, None);
        assert_eq!(state.finalize(), Value::Null);
    }

    #[test]
    fn aggregate_state_min_max_empty_returns_null() {
        let min = AggregateState::new(AggregateFunction::Min, false, None, None);
        let max = AggregateState::new(AggregateFunction::Max, false, None, None);
        assert_eq!(min.finalize(), Value::Null);
        assert_eq!(max.finalize(), Value::Null);
    }

    #[test]
    fn aggregate_state_count_non_null_skips_nulls() {
        // CountNonNull maps to the Count(0) state variant, which increments
        // unconditionally. Callers (both push and pull operators) must filter
        // null values before calling update. This test verifies the expected
        // contract: only non-null values are fed to the accumulator.
        let mut state = AggregateState::new(AggregateFunction::CountNonNull, false, None, None);
        // Simulate what the operator should do: skip nulls, update only non-nulls
        // (Value::Null is skipped, Value::Int64(5) is the only non-null)
        state.update(Some(Value::Int64(5)));
        assert_eq!(state.finalize(), Value::Int64(1));
    }

    #[test]
    fn test_empty_chunk_returns_ok() {
        let mut agg = AggregatePushOperator::global(vec![AggregateExpr::count_star()]);
        let mut sink = CollectorSink::new();
        let empty = DataChunk::new(vec![ValueVector::new()]);
        let result = agg.push(empty, &mut sink).unwrap();
        assert!(result);
    }

    // ---------------------------------------------------------------
    // Spill serialization round-trip tests
    // ---------------------------------------------------------------

    #[test]
    #[cfg(feature = "spill")]
    fn spill_roundtrip_count() {
        let state = GroupState {
            key_values: vec![Value::String("grp".into())],
            accumulators: vec![AggregateState::Count(42)],
        };
        let mut buf = Vec::new();
        serialize_group_state(&state, &mut buf).unwrap();
        let restored = deserialize_group_state(&mut &buf[..]).unwrap();
        assert_eq!(restored.key_values, vec![Value::String("grp".into())]);
        assert_eq!(restored.accumulators[0].finalize(), Value::Int64(42));
    }

    #[test]
    #[cfg(feature = "spill")]
    fn spill_roundtrip_sum_int() {
        let state = GroupState {
            key_values: vec![Value::Int64(1)],
            accumulators: vec![AggregateState::SumInt(100, 5)],
        };
        let mut buf = Vec::new();
        serialize_group_state(&state, &mut buf).unwrap();
        let restored = deserialize_group_state(&mut &buf[..]).unwrap();
        assert_eq!(restored.accumulators[0].finalize(), Value::Int64(100));
    }

    #[test]
    #[cfg(feature = "spill")]
    fn spill_roundtrip_sum_float() {
        let state = GroupState {
            key_values: vec![Value::Int64(1)],
            accumulators: vec![AggregateState::SumFloat(3.125, 0.0, 2)],
        };
        let mut buf = Vec::new();
        serialize_group_state(&state, &mut buf).unwrap();
        let restored = deserialize_group_state(&mut &buf[..]).unwrap();
        assert_eq!(restored.accumulators[0].finalize(), Value::Float64(3.125));
    }

    #[test]
    #[cfg(feature = "spill")]
    fn spill_roundtrip_avg() {
        let state = GroupState {
            key_values: vec![Value::Int64(1)],
            accumulators: vec![AggregateState::Avg(30.0, 3)],
        };
        let mut buf = Vec::new();
        serialize_group_state(&state, &mut buf).unwrap();
        let restored = deserialize_group_state(&mut &buf[..]).unwrap();
        assert_eq!(restored.accumulators[0].finalize(), Value::Float64(10.0));
    }

    #[test]
    #[cfg(feature = "spill")]
    fn spill_roundtrip_min() {
        let state = GroupState {
            key_values: vec![Value::Int64(1)],
            accumulators: vec![AggregateState::Min(Some(Value::Int64(7)))],
        };
        let mut buf = Vec::new();
        serialize_group_state(&state, &mut buf).unwrap();
        let restored = deserialize_group_state(&mut &buf[..]).unwrap();
        assert_eq!(restored.accumulators[0].finalize(), Value::Int64(7));
    }

    #[test]
    #[cfg(feature = "spill")]
    fn spill_roundtrip_min_none() {
        let state = GroupState {
            key_values: vec![Value::Int64(1)],
            accumulators: vec![AggregateState::Min(None)],
        };
        let mut buf = Vec::new();
        serialize_group_state(&state, &mut buf).unwrap();
        let restored = deserialize_group_state(&mut &buf[..]).unwrap();
        assert_eq!(restored.accumulators[0].finalize(), Value::Null);
    }

    #[test]
    #[cfg(feature = "spill")]
    fn spill_roundtrip_max() {
        let state = GroupState {
            key_values: vec![Value::Int64(1)],
            accumulators: vec![AggregateState::Max(Some(Value::Int64(99)))],
        };
        let mut buf = Vec::new();
        serialize_group_state(&state, &mut buf).unwrap();
        let restored = deserialize_group_state(&mut &buf[..]).unwrap();
        assert_eq!(restored.accumulators[0].finalize(), Value::Int64(99));
    }

    #[test]
    #[cfg(feature = "spill")]
    fn spill_roundtrip_first() {
        let state = GroupState {
            key_values: vec![Value::Int64(1)],
            accumulators: vec![AggregateState::First(Some(Value::String("hello".into())))],
        };
        let mut buf = Vec::new();
        serialize_group_state(&state, &mut buf).unwrap();
        let restored = deserialize_group_state(&mut &buf[..]).unwrap();
        assert_eq!(
            restored.accumulators[0].finalize(),
            Value::String("hello".into())
        );
    }

    #[test]
    #[cfg(feature = "spill")]
    fn spill_roundtrip_last() {
        let state = GroupState {
            key_values: vec![Value::Int64(1)],
            accumulators: vec![AggregateState::Last(Some(Value::Float64(2.75)))],
        };
        let mut buf = Vec::new();
        serialize_group_state(&state, &mut buf).unwrap();
        let restored = deserialize_group_state(&mut &buf[..]).unwrap();
        assert_eq!(restored.accumulators[0].finalize(), Value::Float64(2.75));
    }

    #[test]
    #[cfg(feature = "spill")]
    fn spill_roundtrip_collect() {
        let state = GroupState {
            key_values: vec![Value::Int64(1)],
            accumulators: vec![AggregateState::Collect(vec![
                Value::Int64(10),
                Value::Int64(20),
                Value::Int64(30),
            ])],
        };
        let mut buf = Vec::new();
        serialize_group_state(&state, &mut buf).unwrap();
        let restored = deserialize_group_state(&mut &buf[..]).unwrap();
        assert_eq!(
            restored.accumulators[0].finalize(),
            Value::List(vec![Value::Int64(10), Value::Int64(20), Value::Int64(30)].into())
        );
    }

    #[test]
    #[cfg(feature = "spill")]
    fn spill_roundtrip_all_variants_combined() {
        // A single GroupState with every common accumulator type
        let state = GroupState {
            key_values: vec![Value::String("combined".into()), Value::Int64(42)],
            accumulators: vec![
                AggregateState::Count(10),
                AggregateState::SumInt(50, 5),
                AggregateState::SumFloat(7.5, 0.0, 3),
                AggregateState::Avg(20.0, 4),
                AggregateState::Min(Some(Value::Int64(1))),
                AggregateState::Max(Some(Value::Int64(99))),
                AggregateState::First(Some(Value::String("first".into()))),
                AggregateState::Last(Some(Value::String("last".into()))),
                AggregateState::Collect(vec![Value::Int64(1), Value::Int64(2)]),
            ],
        };
        let mut buf = Vec::new();
        serialize_group_state(&state, &mut buf).unwrap();
        let restored = deserialize_group_state(&mut &buf[..]).unwrap();

        assert_eq!(restored.key_values.len(), 2);
        assert_eq!(restored.key_values[0], Value::String("combined".into()));
        assert_eq!(restored.key_values[1], Value::Int64(42));
        assert_eq!(restored.accumulators.len(), 9);

        assert_eq!(restored.accumulators[0].finalize(), Value::Int64(10));
        assert_eq!(restored.accumulators[1].finalize(), Value::Int64(50));
        assert_eq!(restored.accumulators[2].finalize(), Value::Float64(7.5));
        assert_eq!(restored.accumulators[3].finalize(), Value::Float64(5.0));
        assert_eq!(restored.accumulators[4].finalize(), Value::Int64(1));
        assert_eq!(restored.accumulators[5].finalize(), Value::Int64(99));
        assert_eq!(
            restored.accumulators[6].finalize(),
            Value::String("first".into())
        );
        assert_eq!(
            restored.accumulators[7].finalize(),
            Value::String("last".into())
        );
        assert_eq!(
            restored.accumulators[8].finalize(),
            Value::List(vec![Value::Int64(1), Value::Int64(2)].into())
        );
    }

    // ---------------------------------------------------------------
    // DISTINCT variants serialize as FINALIZED
    // ---------------------------------------------------------------

    #[test]
    #[cfg(feature = "spill")]
    fn spill_roundtrip_count_distinct() {
        use crate::execution::operators::accumulator::HashableValue;
        use std::collections::HashSet;

        let mut seen = HashSet::new();
        seen.insert(HashableValue::from(Value::Int64(1)));
        seen.insert(HashableValue::from(Value::Int64(2)));
        seen.insert(HashableValue::from(Value::Int64(3)));
        let state = GroupState {
            key_values: vec![Value::Int64(1)],
            accumulators: vec![AggregateState::CountDistinct(3, seen)],
        };
        let mut buf = Vec::new();
        serialize_group_state(&state, &mut buf).unwrap();
        let restored = deserialize_group_state(&mut &buf[..]).unwrap();
        // DISTINCT serializes as FINALIZED, deserialized as Frozen(val)
        assert_eq!(restored.accumulators[0].finalize(), Value::Int64(3));
    }

    #[test]
    #[cfg(feature = "spill")]
    fn spill_roundtrip_avg_distinct() {
        use crate::execution::operators::accumulator::HashableValue;
        use std::collections::HashSet;

        let mut seen = HashSet::new();
        seen.insert(HashableValue::from(Value::Float64(2.0)));
        seen.insert(HashableValue::from(Value::Float64(4.0)));
        let state = GroupState {
            key_values: vec![Value::Int64(1)],
            accumulators: vec![AggregateState::AvgDistinct(6.0, 2, seen)],
        };
        let mut buf = Vec::new();
        serialize_group_state(&state, &mut buf).unwrap();
        let restored = deserialize_group_state(&mut &buf[..]).unwrap();
        assert_eq!(restored.accumulators[0].finalize(), Value::Float64(3.0));
    }

    #[test]
    #[cfg(feature = "spill")]
    fn spill_roundtrip_collect_distinct() {
        use crate::execution::operators::accumulator::HashableValue;
        use std::collections::HashSet;

        let mut seen = HashSet::new();
        seen.insert(HashableValue::from(Value::Int64(10)));
        seen.insert(HashableValue::from(Value::Int64(20)));
        let state = GroupState {
            key_values: vec![Value::Int64(1)],
            accumulators: vec![AggregateState::CollectDistinct(
                vec![Value::Int64(10), Value::Int64(20)],
                seen,
            )],
        };
        let mut buf = Vec::new();
        serialize_group_state(&state, &mut buf).unwrap();
        let restored = deserialize_group_state(&mut &buf[..]).unwrap();
        // CollectDistinct finalizes to a List, deserialized via FINALIZED fallback
        let result = restored.accumulators[0].finalize();
        assert!(matches!(result, Value::List(_)));
    }

    // ---------------------------------------------------------------
    // Complex variants (FINALIZED fallback)
    // ---------------------------------------------------------------

    #[test]
    #[cfg(feature = "spill")]
    fn spill_roundtrip_stddev() {
        // Build a StdDev state by feeding values
        let mut acc = AggregateState::new(AggregateFunction::StdDev, false, None, None);
        acc.update(Some(Value::Float64(2.0)));
        acc.update(Some(Value::Float64(4.0)));
        acc.update(Some(Value::Float64(6.0)));
        let expected = acc.finalize();

        let state = GroupState {
            key_values: vec![Value::Int64(1)],
            accumulators: vec![acc],
        };
        let mut buf = Vec::new();
        serialize_group_state(&state, &mut buf).unwrap();
        let restored = deserialize_group_state(&mut &buf[..]).unwrap();
        // Complex variant stored as FINALIZED, restored as Frozen(val)
        assert_eq!(restored.accumulators[0].finalize(), expected);
    }

    #[test]
    #[cfg(feature = "spill")]
    fn spill_roundtrip_percentile_disc() {
        let state = GroupState {
            key_values: vec![Value::Int64(1)],
            accumulators: vec![AggregateState::PercentileDisc {
                values: vec![1.0, 2.0, 3.0, 4.0, 5.0],
                percentile: 0.5,
            }],
        };
        let expected = state.accumulators[0].finalize();
        let mut buf = Vec::new();
        serialize_group_state(&state, &mut buf).unwrap();
        let restored = deserialize_group_state(&mut &buf[..]).unwrap();
        assert_eq!(restored.accumulators[0].finalize(), expected);
    }

    #[test]
    #[cfg(feature = "spill")]
    fn spill_roundtrip_group_concat() {
        let state = GroupState {
            key_values: vec![Value::Int64(1)],
            accumulators: vec![AggregateState::GroupConcat(
                vec!["alix".to_string(), "gus".to_string(), "vincent".to_string()],
                ", ".to_string(),
            )],
        };
        let expected = state.accumulators[0].finalize();
        let mut buf = Vec::new();
        serialize_group_state(&state, &mut buf).unwrap();
        let restored = deserialize_group_state(&mut &buf[..]).unwrap();
        assert_eq!(restored.accumulators[0].finalize(), expected);
    }

    // ---------------------------------------------------------------
    // SpillableAggregatePushOperator with Collect
    // ---------------------------------------------------------------

    #[test]
    #[cfg(feature = "spill")]
    fn test_spillable_aggregate_collect() {
        use tempfile::TempDir;

        let temp_dir = TempDir::new().unwrap();
        let manager = Arc::new(SpillManager::new(temp_dir.path()).unwrap());

        let mut agg = SpillableAggregatePushOperator::with_spilling(
            vec![0],
            vec![AggregateExpr::collect(1)],
            manager,
            3, // Spill after 3 groups
        );
        let mut sink = CollectorSink::new();

        // Create groups: group 1 collects [10, 20], group 2 collects [30, 40]
        agg.push(
            create_two_column_chunk(&[1, 2, 1, 2], &[10, 30, 20, 40]),
            &mut sink,
        )
        .unwrap();
        // Add more groups to trigger spilling
        for i in 3..10 {
            agg.push(create_two_column_chunk(&[i], &[i * 10]), &mut sink)
                .unwrap();
        }
        agg.finalize(&mut sink).unwrap();

        let chunks = sink.into_chunks();
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].len(), 9); // 9 groups

        // Find group 1 and verify its collected list
        let mut found_group1 = false;
        for row in 0..chunks[0].len() {
            if let Some(Value::Int64(1)) = chunks[0].column(0).unwrap().get_value(row) {
                let collected = chunks[0].column(1).unwrap().get_value(row).unwrap();
                if let Value::List(list) = collected {
                    assert_eq!(list.len(), 2);
                    assert!(list.contains(&Value::Int64(10)));
                    assert!(list.contains(&Value::Int64(20)));
                    found_group1 = true;
                }
            }
        }
        assert!(found_group1, "Group 1 with collected values not found");
    }

    // ---------------------------------------------------------------
    // SpillableAggregatePushOperator with Min/Max
    // ---------------------------------------------------------------

    #[test]
    #[cfg(feature = "spill")]
    fn test_spillable_aggregate_min_max() {
        use tempfile::TempDir;

        let temp_dir = TempDir::new().unwrap();
        let manager = Arc::new(SpillManager::new(temp_dir.path()).unwrap());

        let mut agg = SpillableAggregatePushOperator::with_spilling(
            vec![0],
            vec![AggregateExpr::min(1), AggregateExpr::max(1)],
            manager,
            3, // Spill after 3 groups
        );
        let mut sink = CollectorSink::new();

        // Group 1: values 50, 10, 30 => min=10, max=50
        // Group 2: values 20, 40 => min=20, max=40
        agg.push(
            create_two_column_chunk(&[1, 2, 1, 2, 1], &[50, 20, 10, 40, 30]),
            &mut sink,
        )
        .unwrap();

        // Add more groups to trigger spilling
        for i in 3..10 {
            agg.push(create_two_column_chunk(&[i], &[i * 10]), &mut sink)
                .unwrap();
        }
        agg.finalize(&mut sink).unwrap();

        let chunks = sink.into_chunks();
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].len(), 9); // 9 groups

        // Verify group 1: min=10, max=50
        let mut found_group1 = false;
        for row in 0..chunks[0].len() {
            if let Some(Value::Int64(1)) = chunks[0].column(0).unwrap().get_value(row) {
                assert_eq!(
                    chunks[0].column(1).unwrap().get_value(row),
                    Some(Value::Int64(10))
                );
                assert_eq!(
                    chunks[0].column(2).unwrap().get_value(row),
                    Some(Value::Int64(50))
                );
                found_group1 = true;
            }
        }
        assert!(found_group1, "Group 1 with min/max not found");

        // Verify group 2: min=20, max=40
        let mut found_group2 = false;
        for row in 0..chunks[0].len() {
            if let Some(Value::Int64(2)) = chunks[0].column(0).unwrap().get_value(row) {
                assert_eq!(
                    chunks[0].column(1).unwrap().get_value(row),
                    Some(Value::Int64(20))
                );
                assert_eq!(
                    chunks[0].column(2).unwrap().get_value(row),
                    Some(Value::Int64(40))
                );
                found_group2 = true;
            }
        }
        assert!(found_group2, "Group 2 with min/max not found");
    }

    // ---------------------------------------------------------------
    // Additional aggregate push operator coverage
    // ---------------------------------------------------------------

    #[test]
    fn test_aggregate_count_non_null() {
        // COUNT(column) with CountNonNull skips null values
        let expr = AggregateExpr::count(0);
        let mut agg = AggregatePushOperator::global(vec![expr]);
        let mut sink = CollectorSink::new();

        // Create a chunk with mixed values and nulls
        let mut col = ValueVector::new();
        col.push(Value::Int64(10)); // Alix's score
        col.push(Value::Null);
        col.push(Value::Int64(30)); // Gus's score
        col.push(Value::Null);
        col.push(Value::Int64(50)); // Vincent's score
        let chunk = DataChunk::new(vec![col]);

        agg.push(chunk, &mut sink).unwrap();
        agg.finalize(&mut sink).unwrap();

        let chunks = sink.into_chunks();
        assert_eq!(chunks.len(), 1);
        // Only 3 non-null values should be counted
        assert_eq!(
            chunks[0].column(0).unwrap().get_value(0),
            Some(Value::Int64(3))
        );
    }

    #[test]
    fn test_grouped_aggregate_empty_groups() {
        // Grouped aggregate with empty input produces no output
        let mut agg = AggregatePushOperator::new(vec![0], vec![AggregateExpr::sum(1)]);
        let mut sink = CollectorSink::new();

        // Push an empty chunk
        let empty = DataChunk::new(vec![ValueVector::new(), ValueVector::new()]);
        agg.push(empty, &mut sink).unwrap();
        agg.finalize(&mut sink).unwrap();

        let chunks = sink.into_chunks();
        // No groups produced, so no output chunk
        assert!(chunks.is_empty());
    }

    #[test]
    #[cfg(feature = "spill")]
    fn test_spillable_aggregate_threshold_transition() {
        // Test the transition from non-partitioned to partitioned mode
        // when the spill_manager is set but threshold is reached without
        // using with_spilling (tests the maybe_spill fallback path)
        use tempfile::TempDir;

        let temp_dir = TempDir::new().unwrap();
        let manager = Arc::new(SpillManager::new(temp_dir.path()).unwrap());

        // Use with_spilling to trigger the partitioned spill path
        let mut agg = SpillableAggregatePushOperator::with_spilling(
            vec![0],
            vec![AggregateExpr::count_star()],
            manager,
            2, // Very low threshold
        );
        let mut sink = CollectorSink::new();

        // Create 5 groups to force spilling
        for i in 0..5 {
            agg.push(create_test_chunk(&[i]), &mut sink).unwrap();
        }
        agg.finalize(&mut sink).unwrap();

        let chunks = sink.into_chunks();
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].len(), 5);
    }

    #[test]
    #[cfg(feature = "spill")]
    fn spill_finalized_frozen_ignores_further_updates() {
        let mut acc = AggregateState::new(AggregateFunction::StdDev, false, None, None);
        acc.update(Some(Value::Float64(2.0)));
        acc.update(Some(Value::Float64(4.0)));
        acc.update(Some(Value::Float64(6.0)));
        let expected = acc.finalize();

        let state = GroupState {
            key_values: vec![Value::Int64(1)],
            accumulators: vec![acc],
        };
        let mut buf = Vec::new();
        serialize_group_state(&state, &mut buf).unwrap();
        let mut restored = deserialize_group_state(&mut &buf[..]).unwrap();

        assert!(matches!(
            restored.accumulators[0],
            AggregateState::Frozen(_)
        ));

        restored.accumulators[0].update(Some(Value::Float64(100.0)));
        restored.accumulators[0].update(Some(Value::Float64(200.0)));

        assert_eq!(restored.accumulators[0].finalize(), expected);
    }

    // ---------------------------------------------------------------
    // Serialization roundtrip: end-to-end through push operator
    // ---------------------------------------------------------------

    #[test]
    #[cfg(feature = "spill")]
    fn test_serialize_deserialize_sum_state() {
        // Sum with float values to exercise SumFloat serialization path
        let state = GroupState {
            key_values: vec![Value::String("Alix".into())],
            accumulators: vec![
                AggregateState::SumInt(42, 3),
                AggregateState::SumFloat(2.72, 0.001, 2),
            ],
        };
        let mut buf = Vec::new();
        serialize_group_state(&state, &mut buf).unwrap();
        let restored = deserialize_group_state(&mut &buf[..]).unwrap();

        assert_eq!(restored.key_values, vec![Value::String("Alix".into())]);
        assert_eq!(restored.accumulators[0].finalize(), Value::Int64(42));
        assert_eq!(restored.accumulators[1].finalize(), Value::Float64(2.72));
    }

    #[test]
    #[cfg(feature = "spill")]
    fn test_serialize_deserialize_avg_state() {
        // Avg with sum=30.0, count=6 => finalize should produce 5.0
        let state = GroupState {
            key_values: vec![Value::String("Gus".into())],
            accumulators: vec![AggregateState::Avg(30.0, 6)],
        };
        let mut buf = Vec::new();
        serialize_group_state(&state, &mut buf).unwrap();
        let restored = deserialize_group_state(&mut &buf[..]).unwrap();

        assert_eq!(restored.key_values, vec![Value::String("Gus".into())]);
        assert_eq!(restored.accumulators[0].finalize(), Value::Float64(5.0));
    }

    #[test]
    #[cfg(feature = "spill")]
    fn test_serialize_deserialize_count_state() {
        let state = GroupState {
            key_values: vec![Value::String("Vincent".into())],
            accumulators: vec![AggregateState::Count(17)],
        };
        let mut buf = Vec::new();
        serialize_group_state(&state, &mut buf).unwrap();
        let restored = deserialize_group_state(&mut &buf[..]).unwrap();

        assert_eq!(restored.key_values, vec![Value::String("Vincent".into())]);
        assert_eq!(restored.accumulators[0].finalize(), Value::Int64(17));
    }

    #[test]
    #[cfg(feature = "spill")]
    fn test_serialize_deserialize_min_max_state() {
        // Test with String values (not just Int64) to cover different value types
        let state = GroupState {
            key_values: vec![Value::String("Jules".into())],
            accumulators: vec![
                AggregateState::Min(Some(Value::String("Amsterdam".into()))),
                AggregateState::Max(Some(Value::Float64(99.9))),
                AggregateState::Min(None),
                AggregateState::Max(None),
            ],
        };
        let mut buf = Vec::new();
        serialize_group_state(&state, &mut buf).unwrap();
        let restored = deserialize_group_state(&mut &buf[..]).unwrap();

        assert_eq!(
            restored.accumulators[0].finalize(),
            Value::String("Amsterdam".into())
        );
        assert_eq!(restored.accumulators[1].finalize(), Value::Float64(99.9));
        // None values serialize as Null and deserialize as Min(None)
        assert_eq!(restored.accumulators[2].finalize(), Value::Null);
        assert_eq!(restored.accumulators[3].finalize(), Value::Null);
    }

    #[test]
    #[cfg(feature = "spill")]
    fn test_serialize_deserialize_collect_state() {
        // Collect with mixed value types
        let state = GroupState {
            key_values: vec![Value::String("Mia".into())],
            accumulators: vec![AggregateState::Collect(vec![
                Value::Int64(1),
                Value::String("Berlin".into()),
                Value::Float64(2.5),
                Value::Bool(true),
            ])],
        };
        let mut buf = Vec::new();
        serialize_group_state(&state, &mut buf).unwrap();
        let restored = deserialize_group_state(&mut &buf[..]).unwrap();

        let result = restored.accumulators[0].finalize();
        if let Value::List(list) = result {
            assert_eq!(list.len(), 4);
            assert_eq!(list[0], Value::Int64(1));
            assert_eq!(list[1], Value::String("Berlin".into()));
            assert_eq!(list[2], Value::Float64(2.5));
            assert_eq!(list[3], Value::Bool(true));
        } else {
            panic!("expected List, got {result:?}");
        }
    }

    #[test]
    #[cfg(feature = "spill")]
    fn test_serialize_deserialize_count_distinct() {
        use crate::execution::operators::accumulator::HashableValue;
        use std::collections::HashSet;

        let mut seen = HashSet::new();
        seen.insert(HashableValue::from(Value::String("Paris".into())));
        seen.insert(HashableValue::from(Value::String("Prague".into())));
        seen.insert(HashableValue::from(Value::String("Barcelona".into())));
        let state = GroupState {
            key_values: vec![Value::String("Butch".into())],
            accumulators: vec![AggregateState::CountDistinct(3, seen)],
        };
        let mut buf = Vec::new();
        serialize_group_state(&state, &mut buf).unwrap();
        let restored = deserialize_group_state(&mut &buf[..]).unwrap();

        // CountDistinct serializes as FINALIZED, so deserialized as Frozen(Int64(3))
        assert_eq!(restored.accumulators[0].finalize(), Value::Int64(3));
        assert!(
            matches!(restored.accumulators[0], AggregateState::Frozen(_)),
            "DISTINCT should be deserialized as Frozen"
        );
    }

    // ---------------------------------------------------------------
    // Push operator with empty chunks and empty input
    // ---------------------------------------------------------------

    #[test]
    fn test_global_aggregate_empty_input() {
        // Global aggregate with no input at all should produce correct defaults
        let mut agg = AggregatePushOperator::global(vec![
            AggregateExpr::count_star(),
            AggregateExpr::sum(0),
            AggregateExpr::min(0),
            AggregateExpr::max(0),
        ]);
        let mut sink = CollectorSink::new();

        // No push calls at all, directly finalize
        agg.finalize(&mut sink).unwrap();

        let chunks = sink.into_chunks();
        assert_eq!(chunks.len(), 1);
        // COUNT(*) with no input should be 0
        assert_eq!(
            chunks[0].column(0).unwrap().get_value(0),
            Some(Value::Int64(0))
        );
        // SUM with no input should be Null
        assert_eq!(chunks[0].column(1).unwrap().get_value(0), Some(Value::Null));
        // MIN with no input should be Null
        assert_eq!(chunks[0].column(2).unwrap().get_value(0), Some(Value::Null));
        // MAX with no input should be Null
        assert_eq!(chunks[0].column(3).unwrap().get_value(0), Some(Value::Null));
    }

    // ---------------------------------------------------------------
    // Spillable aggregate: memory pressure triggers spilling
    // ---------------------------------------------------------------

    #[test]
    #[cfg(feature = "spill")]
    fn test_spillable_aggregate_memory_pressure() {
        use tempfile::TempDir;

        let temp_dir = TempDir::new().unwrap();
        let manager = Arc::new(SpillManager::new(temp_dir.path()).unwrap());

        // Extremely low threshold of 2 to force spilling quickly
        let mut agg = SpillableAggregatePushOperator::with_spilling(
            vec![0],
            vec![AggregateExpr::sum(1)],
            Arc::clone(&manager),
            2,
        );
        let mut sink = CollectorSink::new();

        // Push many distinct groups to trigger memory pressure and spilling
        for i in 0..20 {
            let chunk = create_two_column_chunk(&[i], &[i * 5]);
            agg.push(chunk, &mut sink).unwrap();
        }

        // Verify spilling happened (manager should have active spill files)
        assert!(
            manager.active_file_count() > 0,
            "expected spill files to be created under memory pressure"
        );

        agg.finalize(&mut sink).unwrap();

        let chunks = sink.into_chunks();
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].len(), 20);

        // Verify all sums are correct
        let mut sums: Vec<i64> = Vec::new();
        for i in 0..chunks[0].len() {
            if let Some(Value::Int64(sum)) = chunks[0].column(1).unwrap().get_value(i) {
                sums.push(sum);
            }
        }
        sums.sort_unstable();
        let expected: Vec<i64> = (0..20).map(|i| i * 5).collect();
        assert_eq!(sums, expected);
    }
}
