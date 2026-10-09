//! Push-based aggregate operator (pipeline breaker).

use crate::execution::chunk::{ColumnTypes, DataChunk};
use crate::execution::operators::OperatorError;
use crate::execution::operators::accumulator::{AggregateExpr, AggregateState, RowKey};
use crate::execution::pipeline::{ChunkSizeHint, PushOperator, Sink};
#[cfg(feature = "spill")]
use crate::execution::spill::{PartitionedState, SpillManager};
use crate::execution::vector::ValueVector;
use grafeo_common::types::{HashableValue as ValueKey, Value};
#[cfg(feature = "spill")]
use spill_codec::{deserialize_group_state, serialize_group_state};
use std::collections::HashMap;
#[cfg(feature = "spill")]
use std::sync::Arc;

/// Updates a single accumulator from a data chunk row, handling bivariate
/// functions and `COUNT(*)`. The accumulator skips a null operand.
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
        // An operand: a missing value is a null too, not a row to count.
        let val = chunk.column(col).and_then(|c| c.get_value(row));
        acc.update(Some(val.unwrap_or(Value::Null)));
    } else {
        // COUNT(*)
        acc.update(None);
    }
}

/// Hash key for grouping.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct GroupKey(Vec<GroupKeyPart>);

// Cache the existing hash, but retain the full value for collision checks.
#[derive(Debug, Clone, PartialEq, Eq)]
struct GroupKeyPart {
    hash: u64,
    value: ValueKey,
}

impl std::hash::Hash for GroupKeyPart {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        std::hash::Hash::hash(&self.hash, state);
    }
}

impl GroupKey {
    fn from_row(chunk: &DataChunk, row: usize, group_by: &[usize]) -> Self {
        let parts = group_by
            .iter()
            .map(|&col| {
                let value = chunk
                    .column(col)
                    .and_then(|c| c.get_value(row))
                    .unwrap_or(Value::Null);
                GroupKeyPart {
                    hash: hash_value(&value),
                    value: value.into(),
                }
            })
            .collect();
        Self(parts)
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
#[cfg(feature = "spill")]
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
    /// Group states by their key: keys whose values are the same values
    /// (`3` and `3.0`) are one key, see [`RowKey`].
    groups: HashMap<RowKey, GroupState>,
    /// Global accumulator (for no GROUP BY).
    global_state: Option<Vec<AggregateState>>,
    /// The column types of the input chunks.
    input_types: ColumnTypes,
}

impl AggregatePushOperator {
    /// Create a new aggregate operator.
    pub fn new(group_by: Vec<usize>, aggregates: Vec<AggregateExpr>) -> Self {
        let global_state = if group_by.is_empty() {
            Some(aggregates.iter().map(AggregateState::for_expr).collect())
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
                // Group by aggregation: a new group keeps the key values of
                // its first row.
                let group_by = &self.group_by;
                let aggregates = &self.aggregates;
                let state = self
                    .groups
                    .entry(RowKey::from_row(&chunk, row, group_by))
                    .or_insert_with(|| GroupState {
                        key_values: RowKey::values_of(&chunk, row, group_by),
                        accumulators: aggregates.iter().map(AggregateState::for_expr).collect(),
                    });

                for (acc, expr) in accumulators.iter_mut().zip(&self.aggregates) {
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
            for (key, accumulators) in &self.groups {
                // The retained equality key also owns the output values.
                for (i, part) in key.0.iter().enumerate() {
                    columns[i].push(part.value.0.clone());
                }

                // Output aggregate results
                for (i, acc) in accumulators.iter().enumerate() {
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

#[cfg(feature = "spill")]
mod spill_codec;

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
    groups: HashMap<RowKey, GroupState>,
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
            Some(aggregates.iter().map(AggregateState::for_expr).collect())
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
            Some(aggregates.iter().map(AggregateState::for_expr).collect())
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
            Some(aggregates.iter().map(AggregateState::for_expr).collect())
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

                // Move existing groups to partitioned state, filed under
                // their keys' representatives (see `push`)
                for (key, state) in self.groups.drain() {
                    partitioned
                        .insert(key.representatives(), state)
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
                    // A partition files a group under the representatives of
                    // its key values, which are equal (and serialize to the
                    // same bytes) for the same values, `3` and `3.0` too; the
                    // group keeps the key values of its first row.
                    let group_by = &self.group_by;
                    let aggregates = &self.aggregates;
                    let key = RowKey::from_row(&chunk, row, group_by).representatives();
                    let state = partitioned
                        .get_or_insert_with(key, || GroupState {
                            key_values: RowKey::values_of(&chunk, row, group_by),
                            accumulators: aggregates.iter().map(AggregateState::for_expr).collect(),
                        })
                        .map_err(|e| OperatorError::Execution(e.to_string()))?;

                    for (acc, expr) in state.accumulators.iter_mut().zip(&self.aggregates) {
                        update_accumulator(acc, expr, &chunk, row);
                    }
                }
            } else {
                // Use regular hash map
                let group_by = &self.group_by;
                let aggregates = &self.aggregates;
                let state = self
                    .groups
                    .entry(RowKey::from_row(&chunk, row, group_by))
                    .or_insert_with(|| GroupState {
                        key_values: RowKey::values_of(&chunk, row, group_by),
                        accumulators: aggregates.iter().map(AggregateState::for_expr).collect(),
                    });

                for (acc, expr) in accumulators.iter_mut().zip(&self.aggregates) {
                    update_accumulator(acc, expr, &chunk, row);
                }
            }
        }

        // Update memory consumer usage estimate
        if let Some(ref spill_state) = self.spill_state {
            // Rough sizing: retained key values and accumulator states.
            // In-memory groups reuse the equality key for output.
            let group_count = if self.using_partitioned {
                self.partitioned_groups
                    .as_ref()
                    .map_or(0, |p| p.total_size())
            } else {
                self.groups.len()
            };
            let key_part_size = if self.using_partitioned {
                std::mem::size_of::<Value>()
            } else {
                std::mem::size_of::<GroupKeyPart>()
            };
            let key_size = self.group_by.len() * key_part_size;
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
            for (key, accumulators) in &self.groups {
                // The retained equality key also owns the output values.
                for (i, part) in key.0.iter().enumerate() {
                    columns[i].push(part.value.0.clone());
                }

                // Output aggregate results
                for (i, acc) in accumulators.iter().enumerate() {
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

    fn check_counter_group_identity(mut aggregate: impl PushOperator) {
        use grafeo_common::types::HashableValue as ValueKey;
        use std::sync::Arc;

        let empty = Arc::new(HashMap::new());
        let actor = Arc::new(HashMap::from([("actor".to_owned(), 1_u64)]));
        let positive = Value::OnCounter {
            pos: Arc::clone(&actor),
            neg: Arc::clone(&empty),
        };
        let negative = Value::OnCounter {
            pos: empty,
            neg: actor,
        };
        let mut sink = CollectorSink::new();
        for values in [
            vec![positive.clone(), negative.clone()],
            vec![positive.clone()],
        ] {
            aggregate
                .push(
                    DataChunk::new(vec![ValueVector::from_values(&values)]),
                    &mut sink,
                )
                .unwrap();
        }
        aggregate.finalize(&mut sink).unwrap();
        let chunks = sink.into_chunks();
        let actual: HashMap<ValueKey, i64> = chunks
            .iter()
            .flat_map(|chunk| {
                chunk.selected_indices().map(|row| {
                    (
                        chunk.column(0).unwrap().get_value(row).unwrap().into(),
                        chunk
                            .column(1)
                            .unwrap()
                            .get_value(row)
                            .unwrap()
                            .as_int64()
                            .unwrap(),
                    )
                })
            })
            .collect();
        assert_eq!(chunks.iter().map(DataChunk::row_count).sum::<usize>(), 2);
        assert_eq!(
            actual,
            HashMap::from([(positive.into(), 2), (negative.into(), 1)])
        );
    }

    #[test]
    fn group_identity_keeps_opposite_counter_states() {
        check_counter_group_identity(AggregatePushOperator::new(
            vec![0],
            vec![AggregateExpr::count_star()],
        ));
    }

    #[test]
    #[cfg(feature = "spill")]
    fn group_identity_keeps_counters_in_spillable_fallback() {
        check_counter_group_identity(SpillableAggregatePushOperator::new(
            vec![0],
            vec![AggregateExpr::count_star()],
        ));
    }

    #[test]
    #[cfg(feature = "spill")]
    fn group_identity_keeps_counters_in_partitioned_execution() {
        let dir = tempfile::tempdir().unwrap();
        let manager = Arc::new(SpillManager::new(dir.path()).unwrap());
        check_counter_group_identity(SpillableAggregatePushOperator::with_spilling(
            vec![0],
            vec![AggregateExpr::count_star()],
            manager,
            1,
        ));
    }

    #[test]
    #[cfg(feature = "spill")]
    fn group_identity_survives_fallback_to_partitioned_execution() {
        let dir = tempfile::tempdir().unwrap();
        let manager = Arc::new(SpillManager::new(dir.path()).unwrap());
        let mut aggregate =
            SpillableAggregatePushOperator::new(vec![0], vec![AggregateExpr::count_star()])
                .with_threshold(2);
        // Start in the fallback map, then move its existing groups when the
        // threshold is reached. with_spilling starts partitioned immediately.
        aggregate.spill_manager = Some(manager);
        check_counter_group_identity(aggregate);
    }

    fn check_repeated_grouped_finalize(mut aggregate: impl PushOperator) {
        let integer_list = Value::List(std::sync::Arc::from([Value::Int64(0)]));
        let float_list = Value::List(std::sync::Arc::from([Value::Float64(0.0)]));
        let values = [
            integer_list.clone(),
            float_list.clone(),
            integer_list.clone(),
            Value::Float64(-0.0),
            Value::Float64(0.0),
        ];
        let mut input_sink = CollectorSink::new();
        aggregate
            .push(
                DataChunk::new(vec![ValueVector::from_values(&values)]),
                &mut input_sink,
            )
            .unwrap();
        let expected = HashMap::from([
            (ValueKey::from(integer_list), 2),
            (ValueKey::from(float_list), 1),
            (ValueKey::from(Value::Float64(-0.0)), 1),
            (ValueKey::from(Value::Float64(0.0)), 1),
        ]);
        for _ in 0..2 {
            let mut sink = CollectorSink::new();
            aggregate.finalize(&mut sink).unwrap();
            let chunks = sink.into_chunks();
            assert_eq!(chunks.iter().map(DataChunk::row_count).sum::<usize>(), 4);
            let actual: HashMap<ValueKey, i64> = chunks
                .iter()
                .flat_map(|chunk| {
                    chunk.selected_indices().map(|row| {
                        (
                            chunk.column(0).unwrap().get_value(row).unwrap().into(),
                            chunk
                                .column(1)
                                .unwrap()
                                .get_value(row)
                                .unwrap()
                                .as_int64()
                                .unwrap(),
                        )
                    })
                })
                .collect();
            assert_eq!(actual, expected);
        }
    }

    #[test]
    fn grouped_finalize_preserves_typed_keys_between_calls() {
        check_repeated_grouped_finalize(AggregatePushOperator::new(
            vec![0],
            vec![AggregateExpr::count_star()],
        ));
    }

    #[test]
    #[cfg(feature = "spill")]
    fn fallback_finalize_preserves_typed_keys_between_calls() {
        check_repeated_grouped_finalize(SpillableAggregatePushOperator::new(
            vec![0],
            vec![AggregateExpr::count_star()],
        ));
    }

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

    /// Grouping keys that are the same value: `3` and `3.0`, `-0.0` and
    /// `0.0`, two NaN with different bits, a path and its copy.
    fn equivalent_keys() -> Vec<DataChunk> {
        let path = |middle: i64| Value::Path {
            nodes: vec![Value::Int64(1), Value::Int64(middle)].into(),
            edges: vec![Value::Int64(10 + middle)].into(),
        };
        [
            Value::Int64(3),
            Value::Float64(3.0),
            Value::Float64(-0.0),
            Value::Float64(0.0),
            Value::Float64(f64::NAN),
            Value::Float64(f64::from_bits(f64::NAN.to_bits() | 1)),
            path(2),
            path(2),
            path(3),
        ]
        .into_iter()
        .map(|key| DataChunk::new(vec![ValueVector::from_values(&[key])]))
        .collect()
    }

    /// The `(key, count)` rows of an aggregate's output, sorted by count.
    fn key_counts(chunks: &[DataChunk]) -> Vec<(Value, i64)> {
        let mut rows: Vec<(Value, i64)> = chunks
            .iter()
            .flat_map(|chunk| {
                chunk.selected_indices().map(|row| {
                    let Some(Value::Int64(count)) = chunk.column(1).unwrap().get_value(row) else {
                        panic!("a count")
                    };
                    (chunk.column(0).unwrap().get_value(row).unwrap(), count)
                })
            })
            .collect();
        rows.sort_by_key(|(key, count)| (*count, format!("{key:?}")));
        rows
    }

    /// Checks the groups of [`equivalent_keys`]: four groups of two (each
    /// keeping the value of its first row) and one of one.
    fn assert_equivalent_key_groups(rows: &[(Value, i64)]) {
        assert_eq!(rows.len(), 5, "{rows:?}");
        assert_eq!(rows[0].1, 1, "{rows:?}");
        assert!(rows[1..].iter().all(|(_, count)| *count == 2), "{rows:?}");
        let keys: Vec<&Value> = rows.iter().map(|(key, _)| key).collect();
        assert!(keys.contains(&&Value::Int64(3)), "{rows:?}");
        assert!(
            keys.iter()
                .any(|key| matches!(key, Value::Float64(z) if z.is_sign_negative() && *z == 0.0)),
            "{rows:?}"
        );
        assert!(
            keys.iter()
                .any(|key| matches!(key, Value::Float64(n) if n.is_nan())),
            "{rows:?}"
        );
    }

    /// Keys that are the same value are one group (they were two whenever
    /// their types or bits differed), which keeps the value of its first row.
    #[test]
    fn group_keys_are_equivalent_values() {
        let mut agg = AggregatePushOperator::new(vec![0], vec![AggregateExpr::count_star()]);
        let mut sink = CollectorSink::new();
        for chunk in equivalent_keys() {
            agg.push(chunk, &mut sink).unwrap();
        }
        agg.finalize(&mut sink).unwrap();
        assert_equivalent_key_groups(&key_counts(&sink.into_chunks()));
    }

    /// A spilling aggregate files its groups under the representatives of
    /// their keys, so the same values meet in one group across spills too.
    #[test]
    #[cfg(feature = "spill")]
    fn spilled_group_keys_are_equivalent_values() {
        let temp_dir = tempfile::TempDir::new().unwrap();
        let manager = Arc::new(SpillManager::new(temp_dir.path()).unwrap());
        for threshold in [1, 3, 1000] {
            let mut agg = SpillableAggregatePushOperator::with_spilling(
                vec![0],
                vec![AggregateExpr::count_star()],
                Arc::clone(&manager),
                threshold,
            );
            let mut sink = CollectorSink::new();
            for chunk in equivalent_keys() {
                agg.push(chunk, &mut sink).unwrap();
            }
            agg.finalize(&mut sink).unwrap();
            assert_equivalent_key_groups(&key_counts(&sink.into_chunks()));
        }
    }

    /// A DISTINCT identity survives a spill: two paths of one length stay
    /// two values after the reload (identities were text, the same for both).
    #[test]
    #[cfg(feature = "spill")]
    fn spilled_distinct_identities_keep_paths_apart() {
        let path = |middle: i64| Value::Path {
            nodes: vec![Value::Int64(1), Value::Int64(middle)].into(),
            edges: vec![Value::Int64(10 + middle)].into(),
        };
        let mut count = AggregateState::new(AggregateFunction::Count, true, None, None);
        count.update(Some(path(2)));
        let state = GroupState {
            key_values: vec![Value::Int64(1)],
            accumulators: vec![count],
        };
        let mut buf = Vec::new();
        serialize_group_state(&state, &mut buf).unwrap();
        let mut restored = deserialize_group_state(&mut &buf[..]).unwrap();
        restored.accumulators[0].update(Some(path(3)));
        restored.accumulators[0].update(Some(path(2)));
        assert_eq!(restored.accumulators[0].finalize(), Value::Int64(2));
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

    /// `COUNT(x)` counts the non-null operands; `None` is a `COUNT(*)` row.
    #[test]
    fn aggregate_state_count_non_null_skips_nulls() {
        let mut state = AggregateState::new(AggregateFunction::CountNonNull, false, None, None);
        state.update(Some(Value::Null));
        state.update(Some(Value::Int64(5)));
        state.update(Some(Value::Null));
        assert_eq!(state.finalize(), Value::Int64(1));
        state.update(None);
        assert_eq!(state.finalize(), Value::Int64(2));
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
    // DISTINCT variants retain their identities and current result.
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
        let result = restored.accumulators[0].finalize();
        assert!(matches!(result, Value::List(_)));
    }

    // ---------------------------------------------------------------
    // Statistical and collected states retain their current result.
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

    /// A group whose first operand is null aggregates its later operands:
    /// `min`, `max` and `first` kept that null, and `collect` listed it.
    #[test]
    fn grouped_aggregates_skip_a_leading_null() {
        let mut keys = ValueVector::new();
        let mut operands = ValueVector::new();
        for (key, operand) in [
            (1, Value::Null),
            (1, Value::Int64(19)),
            (1, Value::Int64(3)),
            (2, Value::Null),
        ] {
            keys.push(Value::Int64(key));
            operands.push(operand);
        }
        let mut agg = AggregatePushOperator::new(
            vec![0],
            vec![
                AggregateExpr::min(1),
                AggregateExpr::max(1),
                AggregateExpr::first(1),
                AggregateExpr::collect(1),
                AggregateExpr::count(1),
                AggregateExpr::count_star(),
            ],
        );
        let mut sink = CollectorSink::new();
        agg.push(DataChunk::new(vec![keys, operands]), &mut sink)
            .unwrap();
        agg.finalize(&mut sink).unwrap();

        let chunks = sink.into_chunks();
        assert_eq!(chunks.len(), 1);
        let mut rows: Vec<Vec<Value>> = chunks[0]
            .selected_indices()
            .map(|row| {
                (0..7)
                    .map(|column| chunks[0].column(column).unwrap().get_value(row).unwrap())
                    .collect()
            })
            .collect();
        rows.sort_by_key(|row| row[0].as_int64());
        let list = |values: &[i64]| {
            Value::List(
                values
                    .iter()
                    .map(|&value| Value::Int64(value))
                    .collect::<Vec<_>>()
                    .into(),
            )
        };
        assert_eq!(
            rows,
            [
                vec![
                    Value::Int64(1),
                    Value::Int64(3),
                    Value::Int64(19),
                    Value::Int64(19),
                    list(&[19, 3]),
                    Value::Int64(2),
                    Value::Int64(3),
                ],
                vec![
                    Value::Int64(2),
                    Value::Null,
                    Value::Null,
                    Value::Null,
                    list(&[]),
                    Value::Int64(0),
                    Value::Int64(1),
                ],
            ]
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
        let mut restored = deserialize_group_state(&mut &buf[..]).unwrap();

        assert_eq!(restored.accumulators[0].finalize(), Value::Int64(3));
        assert!(
            matches!(restored.accumulators[0], AggregateState::CountDistinct(..)),
            "DISTINCT must retain its live accumulator"
        );
        restored.accumulators[0].update(Some(Value::String("Paris".into())));
        restored.accumulators[0].update(Some(Value::String("Berlin".into())));
        assert_eq!(restored.accumulators[0].finalize(), Value::Int64(4));
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

    #[cfg(feature = "spill")]
    fn restore_spilled_accumulator(state: &AggregateState) -> AggregateState {
        let group = GroupState {
            key_values: vec![Value::String("continuation".into())],
            accumulators: vec![state.clone()],
        };
        let mut bytes = Vec::new();
        serialize_group_state(&group, &mut bytes).unwrap();
        let mut restored = deserialize_group_state(&mut bytes.as_slice()).unwrap();
        assert_eq!(restored.key_values, group.key_values);
        restored.accumulators.remove(0)
    }

    #[test]
    #[cfg(feature = "spill")]
    fn spill_every_live_variant_resumes() {
        let functions = [
            AggregateFunction::Count,
            AggregateFunction::CountNonNull,
            AggregateFunction::Sum,
            AggregateFunction::Avg,
            AggregateFunction::Min,
            AggregateFunction::Max,
            AggregateFunction::First,
            AggregateFunction::Last,
            AggregateFunction::Collect,
            AggregateFunction::StdDev,
            AggregateFunction::StdDevPop,
            AggregateFunction::Variance,
            AggregateFunction::VariancePop,
            AggregateFunction::PercentileDisc,
            AggregateFunction::PercentileCont,
            AggregateFunction::GroupConcat,
            AggregateFunction::Sample,
            AggregateFunction::CovarSamp,
            AggregateFunction::CovarPop,
            AggregateFunction::Corr,
            AggregateFunction::RegrSlope,
            AggregateFunction::RegrIntercept,
            AggregateFunction::RegrR2,
            AggregateFunction::RegrCount,
            AggregateFunction::RegrSxx,
            AggregateFunction::RegrSyy,
            AggregateFunction::RegrSxy,
            AggregateFunction::RegrAvgx,
            AggregateFunction::RegrAvgy,
        ];
        let update = |state: &mut AggregateState, y, x, floats| {
            let operand = |n: i32| {
                if floats {
                    Value::Float64(f64::from(n))
                } else {
                    Value::Int64(i64::from(n))
                }
            };
            if state.is_bivariate() {
                state.update_bivariate(Some(operand(y)), Some(operand(x)));
            } else {
                state.update(Some(operand(y)));
            }
        };
        let mut failures = Vec::new();
        for function in functions {
            for distinct in [false, true] {
                for floats in [false, true] {
                    let mut uninterrupted =
                        AggregateState::new(function, distinct, Some(0.25), Some("|"));
                    update(&mut uninterrupted, 1, 2, floats);
                    update(&mut uninterrupted, 3, 4, floats);
                    let mut restored = restore_spilled_accumulator(&uninterrupted);
                    if std::mem::discriminant(&restored) != std::mem::discriminant(&uninterrupted) {
                        failures.push(format!(
                            "{function:?}, distinct={distinct}, floats={floats}: lost live state"
                        ));
                    }
                    // A duplicate tests retained DISTINCT identities; new values
                    // require numerical/list states to continue accumulating.
                    for (y, x) in [(3, 4), (5, 8), (9, 16)] {
                        update(&mut uninterrupted, y, x, floats);
                        update(&mut restored, y, x, floats);
                        if restored.finalize() != uninterrupted.finalize() {
                            failures.push(format!(
                                "{function:?}, distinct={distinct}, floats={floats}, y={y}: \
                                 {:?} != {:?}",
                                restored.finalize(),
                                uninterrupted.finalize()
                            ));
                        }
                    }
                }
            }
        }
        assert!(failures.is_empty(), "{}", failures.join("\n"));
    }

    /// A DISTINCT statistical state keeps the values it has seen across a
    /// spill, so a copy after the reload is still dropped.
    #[test]
    #[cfg(feature = "spill")]
    fn spill_distinct_statistical_state_resumes() {
        let mut state = AggregateState::new(AggregateFunction::StdDevPop, true, None, None);
        for x in [3, 19, 3] {
            state.update(Some(Value::Int64(x)));
        }
        let mut restored = restore_spilled_accumulator(&state);
        for x in [19, 88, 3] {
            restored.update(Some(Value::Int64(x)));
        }
        let mut once = AggregateState::new(AggregateFunction::StdDevPop, false, None, None);
        for x in [3, 19, 88] {
            once.update(Some(Value::Int64(x)));
        }
        assert_eq!(restored.finalize(), once.finalize());
    }

    /// Each kind of DISTINCT identity survives a spill; a null is no operand,
    /// before the spill or after it.
    #[test]
    #[cfg(feature = "spill")]
    fn spill_distinct_identity_variants_resume() {
        let other = Value::List(vec![Value::Int64(1)].into());
        let identities = vec![
            Value::Bool(true),
            Value::Int64(1),
            Value::Float64(1.5),
            Value::String("seen".into()),
            Value::String(format!("{other:?}").into()),
            other,
        ];
        let operands: Vec<Value> = std::iter::once(Value::Null)
            .chain(identities.iter().cloned())
            .collect();
        let mut count = AggregateState::new(AggregateFunction::Count, true, None, None);
        let mut collect = AggregateState::new(AggregateFunction::Collect, true, None, None);
        for value in &operands {
            count.update(Some(value.clone()));
            collect.update(Some(value.clone()));
        }
        assert_eq!(count.finalize(), Value::Int64(6));
        let mut restored_count = restore_spilled_accumulator(&count);
        let mut restored_collect = restore_spilled_accumulator(&collect);
        for value in operands.iter().chain(std::iter::once(&Value::Int64(2))) {
            restored_count.update(Some(value.clone()));
            restored_collect.update(Some(value.clone()));
        }
        assert_eq!(restored_count.finalize(), Value::Int64(7));
        let mut expected = identities;
        expected.push(Value::Int64(2));
        assert_eq!(restored_collect.finalize(), Value::List(expected.into()));
    }

    #[test]
    #[cfg(feature = "spill")]
    fn spill_float_compensation_resumes() {
        use crate::execution::operators::accumulator::HashableValue;
        use std::collections::HashSet;

        // A nonzero compensation is retained accumulator state, independently
        // of the numerical policy used to produce it from input rows.
        let sum = 9_007_199_254_740_992.0;
        let compensation = -1.0;
        let seen: HashSet<_> = [Value::Float64(sum), Value::Float64(1.0)]
            .iter()
            .map(HashableValue::from)
            .collect();
        for (mut state, next) in [
            (AggregateState::SumFloat(sum, compensation, 2), 1.0),
            (
                AggregateState::SumFloatDistinct(sum, compensation, 2, seen),
                2.0,
            ),
        ] {
            let mut restored = restore_spilled_accumulator(&state);
            let restored_compensation = match &restored {
                AggregateState::SumFloat(_, value, _)
                | AggregateState::SumFloatDistinct(_, value, _, _) => *value,
                _ => panic!("floating sum restored as a different state"),
            };
            assert_eq!(restored_compensation.to_bits(), compensation.to_bits());
            state.update(Some(Value::Float64(next)));
            restored.update(Some(Value::Float64(next)));
            assert_eq!(restored.finalize(), state.finalize());
        }
    }

    /// An optional state that saw no operand, or only nulls, spills empty; the
    /// reloaded state takes its next operand as its first and skips the nulls
    /// after it.
    #[test]
    #[cfg(feature = "spill")]
    fn spill_empty_optional_states_resume() {
        for (function, expected) in [
            (AggregateFunction::Min, 3),
            (AggregateFunction::Max, 19),
            (AggregateFunction::First, 19),
            (AggregateFunction::Last, 3),
            (AggregateFunction::Sample, 19),
        ] {
            for null_seen in [false, true] {
                let mut state = AggregateState::new(function, false, None, None);
                if null_seen {
                    state.update(Some(Value::Null));
                }
                let mut restored = restore_spilled_accumulator(&state);
                assert_eq!(restored.finalize(), Value::Null, "{function:?}");
                for value in [
                    Value::Null,
                    Value::Int64(19),
                    Value::Null,
                    Value::Int64(3),
                    Value::Null,
                ] {
                    state.update(Some(value.clone()));
                    restored.update(Some(value));
                    assert_eq!(restored.finalize(), state.finalize(), "{function:?}");
                }
                assert_eq!(restored.finalize(), Value::Int64(expected), "{function:?}");
            }
        }
    }

    #[test]
    #[cfg(feature = "spill")]
    fn spill_percentile_and_separator_configuration_resumes() {
        for (function, distinct, expected) in [
            (
                AggregateFunction::PercentileDisc,
                false,
                Value::Float64(2.0),
            ),
            (
                AggregateFunction::PercentileCont,
                false,
                Value::Float64(3.5),
            ),
            (
                AggregateFunction::GroupConcat,
                false,
                Value::String("2|4|4|8".into()),
            ),
            (
                AggregateFunction::GroupConcat,
                true,
                Value::String("2|4|8".into()),
            ),
        ] {
            let mut state = AggregateState::new(function, distinct, Some(0.25), Some("|"));
            for value in [2, 4] {
                state.update(Some(Value::Int64(value)));
            }
            let mut restored = restore_spilled_accumulator(&state);
            for value in [4, 8] {
                restored.update(Some(Value::Int64(value)));
            }
            assert_eq!(restored.finalize(), expected, "{function:?}");
        }
    }

    #[test]
    #[cfg(feature = "spill")]
    fn spill_partition_count_distinct_resumes() {
        use tempfile::TempDir;

        let directory = TempDir::new().unwrap();
        let manager = Arc::new(SpillManager::new(directory.path()).unwrap());
        let mut aggregate = SpillableAggregatePushOperator::with_spilling(
            vec![0],
            vec![AggregateExpr::count(1).with_distinct()],
            Arc::clone(&manager),
            usize::MAX,
        );
        let mut sink = CollectorSink::new();
        aggregate
            .push(create_two_column_chunk(&[7, 7], &[10, 10]), &mut sink)
            .unwrap();
        let partitioned = aggregate.partitioned_groups.as_mut().unwrap();
        let partition = partitioned.partition_for(&[Value::Int64(7)]);
        partitioned.spill_partition(partition).unwrap();
        assert!(!partitioned.is_in_memory(partition));
        assert!(manager.spilled_bytes() > 0);
        assert_eq!(std::fs::read_dir(directory.path()).unwrap().count(), 1);
        // The ordinary push updater reloads the same partition, ignores old
        // and new duplicates, and admits the new distinct operand once.
        aggregate
            .push(
                create_two_column_chunk(&[7, 7, 7], &[10, 20, 20]),
                &mut sink,
            )
            .unwrap();
        assert!(
            aggregate
                .partitioned_groups
                .as_ref()
                .unwrap()
                .is_in_memory(partition)
        );
        assert_eq!(manager.spilled_bytes(), 0);
        assert_eq!(std::fs::read_dir(directory.path()).unwrap().count(), 0);
        aggregate.finalize(&mut sink).unwrap();
        let chunks = sink.into_chunks();
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].len(), 1);
        assert_eq!(
            chunks[0].column(0).unwrap().get_value(0),
            Some(Value::Int64(7))
        );
        assert_eq!(
            chunks[0].column(1).unwrap().get_value(0),
            Some(Value::Int64(2))
        );
        assert_eq!(manager.spilled_bytes(), 0);
        assert_eq!(std::fs::read_dir(directory.path()).unwrap().count(), 0);
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
