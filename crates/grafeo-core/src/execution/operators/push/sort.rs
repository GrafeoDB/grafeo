//! Push-based sort operator (pipeline breaker).

use crate::execution::chunk::{ColumnTypes, DataChunk, DataChunkBuilder};
use crate::execution::operators::OperatorError;
use crate::execution::operators::value_utils::order_by_under;
use crate::execution::pipeline::{ChunkSizeHint, PushOperator, Sink};
#[cfg(feature = "spill")]
use crate::execution::spill::{RowOrder, SortedRuns, SpillManager};
use grafeo_common::types::{LogicalType, TemporalComparison, Value};
use std::cmp::Ordering;
#[cfg(feature = "spill")]
use std::sync::Arc;

/// Sort direction.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum SortDirection {
    /// Ascending order.
    Ascending,
    /// Descending order.
    Descending,
}

/// Where nulls go, in either sort direction.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum NullOrder {
    /// NULLs come first.
    First,
    /// NULLs come last.
    Last,
}

/// Sort key specification.
#[derive(Debug, Clone)]
pub struct SortKey {
    /// Column index to sort by.
    pub column: usize,
    /// Sort direction.
    pub direction: SortDirection,
    /// Null handling.
    pub null_order: NullOrder,
}

impl SortKey {
    /// Create a new ascending sort key.
    pub fn ascending(column: usize) -> Self {
        Self {
            column,
            direction: SortDirection::Ascending,
            null_order: NullOrder::Last,
        }
    }

    /// Create a new descending sort key.
    pub fn descending(column: usize) -> Self {
        Self {
            column,
            direction: SortDirection::Descending,
            null_order: NullOrder::First,
        }
    }
}

/// Push-based sort operator.
///
/// This is a pipeline breaker that must buffer all input before producing
/// sorted output in the finalize phase.
pub struct SortPushOperator {
    /// Sort keys.
    keys: Vec<SortKey>,
    /// Buffered rows as (row_values...).
    buffer: Vec<Vec<Value>>,
    /// Number of columns per row.
    num_columns: Option<usize>,
    /// How many leading columns each output row keeps (all when `None`).
    output_width: Option<usize>,
    /// The input's column types: a node or edge column stays one.
    column_types: ColumnTypes,
    /// How the sort keys read a local datetime against a zoned one (see
    /// [`TemporalComparison::normalize`]); `TypesApart` reads every value
    /// as itself.
    temporal: TemporalComparison,
    /// What it holds under the statement's memory, captured at construction
    /// (see [`capture`](crate::execution::memory::capture)); it counts nothing
    /// outside a budgeted statement. It does not spill: past the query memory
    /// limit or the memory limit it fails with `GRAFEO-S001`.
    memory: crate::execution::memory::OperatorMemory,
}

impl SortPushOperator {
    /// Create a new sort operator with the given sort keys.
    pub fn new(keys: Vec<SortKey>) -> Self {
        Self {
            keys,
            buffer: Vec::new(),
            num_columns: None,
            output_width: None,
            column_types: ColumnTypes::default(),
            temporal: TemporalComparison::TypesApart,
            memory: crate::execution::memory::capture(),
        }
    }

    /// Create a sort operator with a single ascending key.
    pub fn ascending(column: usize) -> Self {
        Self::new(vec![SortKey::ascending(column)])
    }

    /// Create a sort operator with a single descending key.
    pub fn descending(column: usize) -> Self {
        Self::new(vec![SortKey::descending(column)])
    }

    /// Returns only the first `width` columns of each row: the columns after
    /// them hold sort keys the planner added to sort by (see
    /// [`SortOperator::with_output_width`](crate::execution::operators::SortOperator::with_output_width)).
    #[must_use]
    pub fn with_output_width(mut self, width: usize) -> Self {
        self.output_width = Some(width);
        self
    }

    /// The operator with `temporal` as the reading of its sort keys (the
    /// statement's `<`). The rows keep their values.
    #[must_use]
    pub fn with_temporal(mut self, temporal: TemporalComparison) -> Self {
        self.temporal = temporal;
        self
    }
}

/// Compare two rows by sort keys, their values read as `temporal` reads them.
fn compare_rows(
    a: &[Value],
    b: &[Value],
    keys: &[SortKey],
    temporal: TemporalComparison,
) -> Ordering {
    for key in keys {
        let ordering = order_by_under(
            a.get(key.column),
            b.get(key.column),
            key.direction == SortDirection::Descending,
            key.null_order == NullOrder::First,
            temporal,
        );
        if ordering != Ordering::Equal {
            return ordering;
        }
    }

    Ordering::Equal
}

/// How many columns the output rows have: `width` when the sort drops
/// trailing sort-key columns, otherwise all `num_cols`.
fn output_width(num_cols: usize, width: Option<usize>) -> usize {
    width.map_or(num_cols, |width| width.min(num_cols))
}

/// The types of the first `width` output columns: the input's (a node or edge
/// column stays one, see `ColumnTypes`).
fn output_types(column_types: &ColumnTypes, width: usize) -> Vec<LogicalType> {
    (0..width)
        .map(|column| {
            column_types
                .types()
                .get(column)
                .cloned()
                .unwrap_or(LogicalType::Any)
        })
        .collect()
}

/// One chunk with the first `width` values of each sorted row, moved out of
/// `rows`: each row is freed once its values are in the chunk, so the sort
/// holds about one copy of its rows while it builds the output, not two.
fn output_chunk_taking(
    rows: Vec<Vec<Value>>,
    column_types: &ColumnTypes,
    width: usize,
) -> DataChunk {
    let types = output_types(column_types, width);
    let mut builder = DataChunkBuilder::with_capacity(&types, rows.len());
    for row in rows {
        let mut values = row.into_iter();
        for column in 0..width {
            let value = values.next().unwrap_or(Value::Null);
            if let Some(out) = builder.column_mut(column) {
                out.push_value(value);
            }
        }
        builder.advance_row();
    }
    builder.finish()
}

impl PushOperator for SortPushOperator {
    fn push(&mut self, chunk: DataChunk, _sink: &mut dyn Sink) -> Result<bool, OperatorError> {
        if chunk.is_empty() {
            return Ok(true);
        }

        // Initialize column count
        if self.num_columns.is_none() {
            self.num_columns = Some(chunk.column_count());
        }
        self.column_types.add(&chunk);

        let num_cols = chunk.column_count();

        // Buffer the selected rows, and count what they hold as the spillable
        // sort does: a row's values with their heap, and the row's own list.
        let counts = self.memory.counts();
        let mut bytes = 0usize;
        for i in chunk.selected_indices() {
            let mut row = Vec::with_capacity(num_cols);
            for col_idx in 0..num_cols {
                let val = chunk
                    .column(col_idx)
                    .and_then(|c| c.get_value(i))
                    .unwrap_or(Value::Null);
                if counts {
                    bytes = bytes.saturating_add(val.estimated_size_bytes());
                }
                row.push(val);
            }
            self.buffer.push(row);
        }
        // Against the statement's memory: a chunk at a time, a compare within
        // the grant.
        if counts {
            let rows = chunk.selected_indices().count();
            let per_row =
                num_cols * std::mem::size_of::<Value>() + std::mem::size_of::<Vec<Value>>();
            self.memory
                .add(bytes.saturating_add(rows.saturating_mul(per_row)))?;
        }

        Ok(true)
    }

    fn finalize(&mut self, sink: &mut dyn Sink) -> Result<(), OperatorError> {
        if self.buffer.is_empty() {
            return Ok(());
        }

        // Sort the buffer - borrow keys separately to avoid borrow conflict
        let keys = &self.keys;
        let temporal = self.temporal;
        self.buffer
            .sort_by(|a, b| compare_rows(a, b, keys, temporal));

        let num_cols = self.num_columns.unwrap_or(0);
        if num_cols == 0 {
            return Ok(());
        }
        let chunk = output_chunk_taking(
            std::mem::take(&mut self.buffer),
            &self.column_types,
            output_width(num_cols, self.output_width),
        );
        // Its rows are in the output chunk now, which is downstream's to count.
        self.memory.release();
        sink.consume(chunk)?;

        Ok(())
    }

    fn preferred_chunk_size(&self) -> ChunkSizeHint {
        // Sort is a breaker, chunk size doesn't matter much
        ChunkSizeHint::Default
    }

    fn name(&self) -> &'static str {
        "SortPush"
    }
}

/// Default spill threshold (number of rows before spilling).
#[cfg(feature = "spill")]
pub const DEFAULT_SPILL_THRESHOLD: usize = 100_000;

/// How a spilled GQL sort's runs order: a row's key is its sort keys'
/// values, in key order (cloned, they share the row's heap values).
#[cfg(feature = "spill")]
struct GqlRowOrder {
    /// The row's columns the keys read, in key order.
    columns: Vec<usize>,
    /// The keys over a key's values: key `i` reads value `i`.
    keys: Vec<SortKey>,
    temporal: TemporalComparison,
}

#[cfg(feature = "spill")]
impl GqlRowOrder {
    fn new(keys: &[SortKey], temporal: TemporalComparison) -> Self {
        Self {
            columns: keys.iter().map(|key| key.column).collect(),
            keys: keys
                .iter()
                .enumerate()
                .map(|(column, key)| SortKey {
                    column,
                    direction: key.direction,
                    null_order: key.null_order,
                })
                .collect(),
            temporal,
        }
    }
}

#[cfg(feature = "spill")]
impl RowOrder for GqlRowOrder {
    type Key = Vec<Value>;

    fn key(&mut self, row: &[Value]) -> Result<Vec<Value>, OperatorError> {
        Ok(self
            .columns
            .iter()
            .map(|&column| row.get(column).cloned().unwrap_or(Value::Null))
            .collect())
    }

    fn compare(&self, a: &Self::Key, b: &Self::Key) -> Ordering {
        compare_rows(a, b, &self.keys, self.temporal)
    }

    fn key_bytes(&self, key: &Self::Key) -> usize {
        key.len() * std::mem::size_of::<Value>()
    }
}

/// Minimum buffer size (rows) before memory-pressure spilling can trigger.
///
/// Prevents "noisy neighbor" scenarios where a tiny sort buffer gets spilled
/// because unrelated subsystems consumed memory.
#[cfg(feature = "spill")]
const SORT_MIN_BUFFER_ROWS: usize = 1000;

/// Push-based sort operator with spilling support.
///
/// This is a pipeline breaker that buffers input and spills to disk
/// when memory pressure is high. Its spilled runs merge at the finish a
/// chunk at a time, within the statement's memory (see
/// [`RunMerge`](crate::execution::spill::RunMerge)).
///
/// Two spill modes are supported:
///
/// 1. **Memory-aware** (when constructed with `with_memory_context`): registers
///    as a `MemoryConsumer` with the `BufferManager` and spills when system
///    pressure is High/Critical or when eviction is explicitly requested.
///
/// 2. **Row-count fallback** (when constructed with `new` or `with_spilling`):
///    spills when `buffer.len() >= spill_threshold` (default 100K rows).
#[cfg(feature = "spill")]
pub struct SpillableSortPushOperator {
    /// Sort keys.
    keys: Vec<SortKey>,
    /// Buffered rows as (row_values...).
    buffer: Vec<Vec<Value>>,
    /// Number of columns per row.
    num_columns: Option<usize>,
    /// How many leading columns each output row keeps (all when `None`).
    output_width: Option<usize>,
    /// The input's column types: a node or edge column stays one.
    column_types: ColumnTypes,
    /// How the sort keys read a local datetime against a zoned one (see
    /// [`TemporalComparison::normalize`]); `TypesApart` reads every value
    /// as itself.
    temporal: TemporalComparison,
    /// Spill manager for file creation (used by row-count fallback mode).
    spill_manager: Option<Arc<SpillManager>>,
    /// The sorted runs, in input order (from the first spill).
    runs: Option<SortedRuns<GqlRowOrder>>,
    /// Threshold to trigger spill (row count, used by fallback mode).
    spill_threshold: usize,
    /// Memory context for pressure-aware spilling.
    memory_ctx: Option<crate::execution::memory::OperatorMemoryContext>,
    /// Shared state with the registered MemoryConsumer adapter.
    spill_state: Option<std::sync::Arc<super::spill_state::OperatorSpillState>>,
    /// Running total of estimated buffer memory in bytes (incremental tracking).
    estimated_bytes: usize,
    /// The grant of the buffer's bytes, grown per chunk against the statement's
    /// memory (its query memory limit and the memory limit).
    grant: crate::execution::memory::OperatorGrant,
    /// Its memory consumer's registration, unregistered when it drops (the
    /// sort ended or failed): only its own consumer.
    registration: Option<crate::execution::memory::ConsumerRegistration>,
}

#[cfg(feature = "spill")]
impl SpillableSortPushOperator {
    /// Create a new spillable sort operator with the given sort keys.
    ///
    /// Uses row-count fallback mode with default threshold (100K rows).
    pub fn new(keys: Vec<SortKey>) -> Self {
        Self {
            keys,
            buffer: Vec::new(),
            num_columns: None,
            output_width: None,
            column_types: ColumnTypes::default(),
            temporal: TemporalComparison::TypesApart,
            spill_manager: None,
            runs: None,
            spill_threshold: DEFAULT_SPILL_THRESHOLD,
            memory_ctx: None,
            grant: crate::execution::memory::OperatorGrant::default(),
            registration: None,
            spill_state: None,
            estimated_bytes: 0,
        }
    }

    /// Create a new spillable sort operator with spilling enabled (row-count mode).
    pub fn with_spilling(keys: Vec<SortKey>, manager: Arc<SpillManager>, threshold: usize) -> Self {
        Self {
            keys,
            buffer: Vec::new(),
            num_columns: None,
            output_width: None,
            column_types: ColumnTypes::default(),
            temporal: TemporalComparison::TypesApart,
            spill_manager: Some(manager),
            runs: None,
            spill_threshold: threshold,
            memory_ctx: None,
            grant: crate::execution::memory::OperatorGrant::default(),
            registration: None,
            spill_state: None,
            estimated_bytes: 0,
        }
    }

    /// Create a spillable sort operator with memory-aware spilling.
    ///
    /// Registers as a `MemoryConsumer` with the `BufferManager` and spills
    /// based on system memory pressure rather than row count thresholds.
    pub fn with_memory_context(
        keys: Vec<SortKey>,
        ctx: crate::execution::memory::OperatorMemoryContext,
    ) -> Self {
        use super::spill_state::{OperatorConsumerAdapter, OperatorSpillState};

        // A name of its own: one sort's end unregisters only its own consumer.
        let state = std::sync::Arc::new(OperatorSpillState::new(
            crate::execution::memory::OperatorMemoryContext::consumer_name("SpillableSortPush"),
        ));
        let adapter =
            std::sync::Arc::new(OperatorConsumerAdapter::new(std::sync::Arc::clone(&state)));
        let registration = ctx.register_consumer(adapter);

        Self {
            keys,
            buffer: Vec::new(),
            num_columns: None,
            output_width: None,
            column_types: ColumnTypes::default(),
            temporal: TemporalComparison::TypesApart,
            spill_manager: None,
            runs: None,
            spill_threshold: DEFAULT_SPILL_THRESHOLD,
            memory_ctx: Some(ctx),
            grant: crate::execution::memory::OperatorGrant::default(),
            registration: Some(registration),
            spill_state: Some(state),
            estimated_bytes: 0,
        }
    }

    /// Create a sort operator with a single ascending key and spilling.
    pub fn ascending_with_spilling(
        column: usize,
        manager: Arc<SpillManager>,
        threshold: usize,
    ) -> Self {
        Self::with_spilling(vec![SortKey::ascending(column)], manager, threshold)
    }

    /// Create a sort operator with a single descending key and spilling.
    pub fn descending_with_spilling(
        column: usize,
        manager: Arc<SpillManager>,
        threshold: usize,
    ) -> Self {
        Self::with_spilling(vec![SortKey::descending(column)], manager, threshold)
    }

    /// Sets the spill threshold (row-count fallback mode).
    pub fn with_threshold(mut self, threshold: usize) -> Self {
        self.spill_threshold = threshold;
        self
    }

    /// Returns only the first `width` columns of each row: the columns after
    /// them hold sort keys the planner added to sort by (see
    /// [`SortOperator::with_output_width`](crate::execution::operators::SortOperator::with_output_width)).
    #[must_use]
    pub fn with_output_width(mut self, width: usize) -> Self {
        self.output_width = Some(width);
        self
    }

    /// The operator with `temporal` as the reading of its sort keys (the
    /// statement's `<`). The rows keep their values.
    #[must_use]
    pub fn with_temporal(mut self, temporal: TemporalComparison) -> Self {
        self.temporal = temporal;
        self
    }

    /// Checks whether spilling should occur and performs it if needed.
    fn maybe_spill(&mut self) -> Result<(), OperatorError> {
        let should_spill = if let Some(ref state) = self.spill_state {
            // Memory-aware: eviction requested OR system pressure is High/Critical
            let eviction = state.take_eviction_request().is_some();
            let pressure = self.memory_ctx.as_ref().map_or(false, |c| c.should_spill());
            // Minimum buffer guard: don't spill tiny buffers from noisy neighbors
            let above_minimum = self.buffer.len() >= SORT_MIN_BUFFER_ROWS;
            (eviction || pressure) && above_minimum
        } else {
            // Row-count fallback (no memory context)
            self.buffer.len() >= self.spill_threshold
        };

        if should_spill {
            self.spill_buffer()?;
        }
        Ok(())
    }

    /// Spills the buffer as one sorted run, when a spill manager is configured.
    ///
    /// Without one it does nothing (the caller's limit check then refuses).
    fn spill_buffer(&mut self) -> Result<(), OperatorError> {
        // Get SpillManager: prefer memory_ctx, fall back to self.spill_manager
        let manager = self
            .memory_ctx
            .as_ref()
            .and_then(|c| c.spill_manager().cloned())
            .or_else(|| self.spill_manager.clone());

        let Some(manager) = manager else {
            return Ok(()); // No spilling configured
        };

        // A stable sort: ties keep input order within the run.
        let keys = &self.keys;
        let temporal = self.temporal;
        self.buffer
            .sort_by(|a, b| compare_rows(a, b, keys, temporal));
        let columns = self.num_columns.unwrap_or(0);
        self.runs
            .get_or_insert_with(|| {
                SortedRuns::new(manager, columns, GqlRowOrder::new(keys, temporal))
            })
            .spill(std::mem::take(&mut self.buffer))?;

        // Reset memory tracking after spill; the grant goes with the buffer.
        self.estimated_bytes = 0;
        if let Some(ctx) = &self.memory_ctx {
            ctx.release(&mut self.grant);
        }
        if let Some(ref state) = self.spill_state {
            state.set_usage(0);
        }

        Ok(())
    }

    /// Unregisters this operator's consumer and gives back its grant.
    fn unregister_consumer(&mut self) {
        self.registration = None;
        if let Some(ctx) = &self.memory_ctx {
            ctx.release(&mut self.grant);
        }
    }
}

#[cfg(feature = "spill")]
impl PushOperator for SpillableSortPushOperator {
    fn push(&mut self, chunk: DataChunk, _sink: &mut dyn Sink) -> Result<bool, OperatorError> {
        if chunk.is_empty() {
            return Ok(true);
        }

        // Initialize column count
        if self.num_columns.is_none() {
            self.num_columns = Some(chunk.column_count());
        }
        self.column_types.add(&chunk);

        let num_cols = chunk.column_count();

        // Buffer all rows with incremental memory tracking
        for i in chunk.selected_indices() {
            let mut row = Vec::with_capacity(num_cols);
            for col_idx in 0..num_cols {
                let val = chunk
                    .column(col_idx)
                    .and_then(|c| c.get_value(i))
                    .unwrap_or(Value::Null);
                self.estimated_bytes += val.estimated_size_bytes();
                row.push(val);
            }
            // Account for Vec<Value> overhead per row
            self.estimated_bytes += num_cols * std::mem::size_of::<Value>();
            self.buffer.push(row);
        }

        // Update memory consumer usage
        if let Some(ref state) = self.spill_state {
            state.set_usage(self.estimated_bytes);
        }

        // The statement's memory bounds the buffer: grow the grant per chunk
        // (a compare within it). Over the query memory limit or the memory
        // limit, spill first (the spill resets the estimate and the grant goes
        // with it); what cannot spill refuses, naming the bound.
        let over_limit = match &self.memory_ctx {
            Some(ctx) if self.estimated_bytes > self.grant.size() => {
                ctx.grow(&mut self.grant, self.estimated_bytes).err()
            }
            _ => None,
        };
        if let Some(error) = over_limit {
            // Over the query limit the buffer spills whatever it holds
            // (the minimum-buffer guard is only for pressure noise); what
            // cannot spill refuses, naming the query limit.
            let can_spill = self
                .memory_ctx
                .as_ref()
                .and_then(|c| c.spill_manager().cloned())
                .or_else(|| self.spill_manager.clone())
                .is_some();
            if can_spill {
                self.spill_buffer()?;
            }
            if self.estimated_bytes != 0 {
                return Err(OperatorError::from(error));
            }
        }

        // Check if we should spill
        self.maybe_spill()?;

        Ok(true)
    }

    fn finalize(&mut self, sink: &mut dyn Sink) -> Result<(), OperatorError> {
        let num_cols = self.num_columns.unwrap_or(0);
        if num_cols == 0 && self.buffer.is_empty() {
            self.unregister_consumer();
            return Ok(());
        }

        let width = output_width(num_cols, self.output_width);
        if self.runs.is_some() {
            // The held rows are the last run: they came after every other.
            self.spill_buffer()?;
        }
        if let Some(runs) = self.runs.take() {
            // The buffer's grant goes back before the merge counts its own.
            self.unregister_consumer();
            let memory = self.memory_ctx.clone().map_or_else(
                crate::execution::memory::OperatorMemory::unbudgeted,
                crate::execution::memory::OperatorMemory::new,
            );
            let mut merge = runs.merge(memory, crate::execution::chunk::DEFAULT_CHUNK_SIZE)?;
            while let Some(batch) = merge.next_batch()? {
                if !sink.consume(output_chunk_taking(batch, &self.column_types, width))? {
                    break;
                }
            }
            return Ok(());
        }

        let keys = &self.keys;
        let temporal = self.temporal;
        self.buffer
            .sort_by(|a, b| compare_rows(a, b, keys, temporal));
        let sorted_rows = std::mem::take(&mut self.buffer);

        // Unregister consumer before emitting results (we no longer hold buffer memory)
        self.unregister_consumer();

        if sorted_rows.is_empty() {
            return Ok(());
        }
        sink.consume(output_chunk_taking(sorted_rows, &self.column_types, width))?;

        Ok(())
    }

    fn preferred_chunk_size(&self) -> ChunkSizeHint {
        // Sort is a breaker, chunk size doesn't matter much
        ChunkSizeHint::Default
    }

    fn name(&self) -> &'static str {
        "SpillableSortPush"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::execution::sink::CollectorSink;
    use crate::execution::vector::ValueVector;

    fn create_test_chunk(values: &[i64]) -> DataChunk {
        let v: Vec<Value> = values.iter().map(|&i| Value::Int64(i)).collect();
        let vector = ValueVector::from_values(&v);
        DataChunk::new(vec![vector])
    }

    #[test]
    fn test_sort_ascending() {
        let mut sort = SortPushOperator::ascending(0);
        let mut sink = CollectorSink::new();

        sort.push(create_test_chunk(&[3, 1, 4, 1, 5, 9, 2, 6]), &mut sink)
            .unwrap();
        sort.finalize(&mut sink).unwrap();

        let chunks = sink.into_chunks();
        assert_eq!(chunks.len(), 1);

        let col = chunks[0].column(0).unwrap();
        assert_eq!(col.get_value(0), Some(Value::Int64(1)));
        assert_eq!(col.get_value(1), Some(Value::Int64(1)));
        assert_eq!(col.get_value(2), Some(Value::Int64(2)));
        assert_eq!(col.get_value(3), Some(Value::Int64(3)));
    }

    #[test]
    fn test_sort_descending() {
        let mut sort = SortPushOperator::descending(0);
        let mut sink = CollectorSink::new();

        sort.push(create_test_chunk(&[3, 1, 4, 1, 5]), &mut sink)
            .unwrap();
        sort.finalize(&mut sink).unwrap();

        let chunks = sink.into_chunks();
        let col = chunks[0].column(0).unwrap();
        assert_eq!(col.get_value(0), Some(Value::Int64(5)));
        assert_eq!(col.get_value(1), Some(Value::Int64(4)));
        assert_eq!(col.get_value(2), Some(Value::Int64(3)));
    }

    /// Two chunks of mixed values with nulls.
    fn mixed_chunks() -> Vec<DataChunk> {
        let first = [
            Value::Int64(3),
            Value::String("a".into()),
            Value::Null,
            Value::Float64(2.5),
        ];
        let second = [
            Value::Bool(true),
            Value::Null,
            Value::Int64(1),
            Value::List(vec![Value::Int64(1)].into()),
        ];
        vec![
            DataChunk::new(vec![ValueVector::from_values(&first)]),
            DataChunk::new(vec![ValueVector::from_values(&second)]),
        ]
    }

    /// `ORDER BY x DESC NULLS LAST` over [`mixed_chunks`].
    const MIXED_DESC_NULLS_LAST: [&str; 8] =
        ["3", "2.5", "1", "true", "\"a\"", "[1]", "NULL", "NULL"];

    fn first_column_texts(sink: CollectorSink) -> Vec<String> {
        sink.into_chunks()
            .iter()
            .flat_map(|chunk| {
                let column = chunk.column(0).unwrap();
                (0..chunk.len())
                    .map(|row| column.get_value(row).unwrap().to_string())
                    .collect::<Vec<_>>()
            })
            .collect()
    }

    /// Values of different types sort in one order, and `NULLS LAST` holds
    /// when descending.
    #[test]
    fn mixed_values_sort_in_one_order_with_nulls_last_descending() {
        let key = SortKey {
            column: 0,
            direction: SortDirection::Descending,
            null_order: NullOrder::Last,
        };
        let mut sort = SortPushOperator::new(vec![key]);
        let mut sink = CollectorSink::new();
        for chunk in mixed_chunks() {
            sort.push(chunk, &mut sink).unwrap();
        }
        sort.finalize(&mut sink).unwrap();

        assert_eq!(first_column_texts(sink), MIXED_DESC_NULLS_LAST);
    }

    /// Nodes 7, 8 and 9 with the sort keys 3, 1 and 2, in two chunks.
    fn nodes_with_keys() -> Vec<DataChunk> {
        use crate::execution::chunk::DataChunkBuilder;
        use grafeo_common::types::{LogicalType, NodeId};

        [vec![(7_u64, 3_i64), (8, 1)], vec![(9, 2)]]
            .into_iter()
            .map(|rows| {
                let mut builder = DataChunkBuilder::new(&[LogicalType::Node, LogicalType::Int64]);
                for (id, key) in rows {
                    builder.column_mut(0).unwrap().push_node_id(NodeId::new(id));
                    builder.column_mut(1).unwrap().push_int64(key);
                    builder.advance_row();
                }
                builder.finish()
            })
            .collect()
    }

    /// The node IDs in `sink`, after checking that each chunk holds one
    /// column and that it is a node column.
    fn node_ids(sink: CollectorSink) -> Vec<u64> {
        use grafeo_common::types::LogicalType;

        let mut ids = Vec::new();
        for chunk in sink.into_chunks() {
            assert_eq!(chunk.column_count(), 1);
            let nodes = chunk.column(0).unwrap();
            assert_eq!(nodes.data_type(), &LogicalType::Node);
            ids.extend(
                chunk
                    .selected_indices()
                    .map(|row| nodes.get_node_id(row).unwrap().as_u64()),
            );
        }
        ids
    }

    /// With an output width the rows keep only their leading columns, with
    /// the types they came in with: the columns after them were sort keys.
    #[test]
    fn output_width_drops_the_sort_key_columns() {
        let mut sort = SortPushOperator::new(vec![SortKey::ascending(1)]).with_output_width(1);
        let mut sink = CollectorSink::new();
        for chunk in nodes_with_keys() {
            sort.push(chunk, &mut sink).unwrap();
        }
        sort.finalize(&mut sink).unwrap();

        assert_eq!(node_ids(sink), [8, 9, 7]);
    }

    /// The same holds for the spillable sort, in memory and after its spilled
    /// runs are merged.
    #[test]
    #[cfg(feature = "spill")]
    fn spillable_sort_keeps_the_output_width_and_types() {
        use tempfile::TempDir;

        let mut in_memory =
            SpillableSortPushOperator::new(vec![SortKey::ascending(1)]).with_output_width(1);
        let mut sink = CollectorSink::new();
        for chunk in nodes_with_keys() {
            in_memory.push(chunk, &mut sink).unwrap();
        }
        in_memory.finalize(&mut sink).unwrap();
        assert_eq!(node_ids(sink), [8, 9, 7]);

        let temp_dir = TempDir::new().unwrap();
        let manager = Arc::new(SpillManager::new(temp_dir.path()).unwrap());
        // A threshold of one row spills every chunk.
        let mut spilled =
            SpillableSortPushOperator::with_spilling(vec![SortKey::ascending(1)], manager, 1)
                .with_output_width(1);
        let mut sink = CollectorSink::new();
        for chunk in nodes_with_keys() {
            spilled.push(chunk, &mut sink).unwrap();
        }
        assert!(spilled.runs.is_some(), "the rows were spilled");
        spilled.finalize(&mut sink).unwrap();
        assert_eq!(node_ids(sink), [8, 9, 7]);
    }

    /// Spilled runs merge in the same order as an in-memory sort.
    #[test]
    #[cfg(feature = "spill")]
    fn spilled_runs_merge_mixed_values_in_the_same_order() {
        use tempfile::TempDir;

        let temp_dir = TempDir::new().unwrap();
        let manager = Arc::new(SpillManager::new(temp_dir.path()).unwrap());
        let key = SortKey {
            column: 0,
            direction: SortDirection::Descending,
            null_order: NullOrder::Last,
        };
        let mut sort = SpillableSortPushOperator::with_spilling(vec![key], manager, 3);
        let mut sink = CollectorSink::new();
        for chunk in mixed_chunks() {
            sort.push(chunk, &mut sink).unwrap();
        }
        sort.finalize(&mut sink).unwrap();

        assert_eq!(first_column_texts(sink), MIXED_DESC_NULLS_LAST);
    }

    #[test]
    fn test_sort_multiple_chunks() {
        let mut sort = SortPushOperator::ascending(0);
        let mut sink = CollectorSink::new();

        sort.push(create_test_chunk(&[5, 3, 1]), &mut sink).unwrap();
        sort.push(create_test_chunk(&[4, 2, 6]), &mut sink).unwrap();
        sort.finalize(&mut sink).unwrap();

        let chunks = sink.into_chunks();
        assert_eq!(chunks[0].len(), 6);

        let col = chunks[0].column(0).unwrap();
        assert_eq!(col.get_value(0), Some(Value::Int64(1)));
        assert_eq!(col.get_value(5), Some(Value::Int64(6)));
    }

    #[test]
    #[cfg(feature = "spill")]
    fn test_spillable_sort_no_spill() {
        // When threshold is not reached, should work like normal sort
        let mut sort =
            SpillableSortPushOperator::new(vec![SortKey::ascending(0)]).with_threshold(100);
        let mut sink = CollectorSink::new();

        sort.push(create_test_chunk(&[3, 1, 4, 1, 5, 9, 2, 6]), &mut sink)
            .unwrap();
        sort.finalize(&mut sink).unwrap();

        let chunks = sink.into_chunks();
        assert_eq!(chunks.len(), 1);

        let col = chunks[0].column(0).unwrap();
        assert_eq!(col.get_value(0), Some(Value::Int64(1)));
        assert_eq!(col.get_value(1), Some(Value::Int64(1)));
        assert_eq!(col.get_value(2), Some(Value::Int64(2)));
        assert_eq!(col.get_value(3), Some(Value::Int64(3)));
    }

    #[test]
    #[cfg(feature = "spill")]
    // reason: test values 1..=10 fit i64
    #[allow(clippy::cast_possible_wrap)]
    fn test_spillable_sort_with_spilling() {
        use tempfile::TempDir;

        let temp_dir = TempDir::new().unwrap();
        let manager = Arc::new(SpillManager::new(temp_dir.path()).unwrap());

        // Set very low threshold to force spilling
        let mut sort = SpillableSortPushOperator::ascending_with_spilling(0, manager, 5);
        let mut sink = CollectorSink::new();

        // Push more than threshold
        sort.push(create_test_chunk(&[10, 8, 6, 4, 2]), &mut sink)
            .unwrap();
        sort.push(create_test_chunk(&[9, 7, 5, 3, 1]), &mut sink)
            .unwrap();
        sort.finalize(&mut sink).unwrap();

        let chunks = sink.into_chunks();
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].len(), 10);

        // Verify sorted order
        let col = chunks[0].column(0).unwrap();
        for i in 0..10 {
            assert_eq!(col.get_value(i), Some(Value::Int64((i + 1) as i64)));
        }
    }

    #[test]
    #[cfg(feature = "spill")]
    // reason: test values 1..=15 fit i64
    #[allow(clippy::cast_possible_wrap)]
    fn test_spillable_sort_many_runs() {
        use tempfile::TempDir;

        let temp_dir = TempDir::new().unwrap();
        let manager = Arc::new(SpillManager::new(temp_dir.path()).unwrap());

        // Set very low threshold to force multiple spills
        let mut sort = SpillableSortPushOperator::ascending_with_spilling(0, manager, 3);
        let mut sink = CollectorSink::new();

        // Push data in multiple chunks
        for i in 0..5 {
            sort.push(
                create_test_chunk(&[i * 3 + 3, i * 3 + 2, i * 3 + 1]),
                &mut sink,
            )
            .unwrap();
        }
        sort.finalize(&mut sink).unwrap();

        let chunks = sink.into_chunks();
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].len(), 15);

        // Verify sorted order
        let col = chunks[0].column(0).unwrap();
        for i in 0..15 {
            assert_eq!(col.get_value(i), Some(Value::Int64((i + 1) as i64)));
        }
    }

    #[test]
    #[cfg(feature = "spill")]
    // reason: test values 1..=6 fit i64
    #[allow(clippy::cast_possible_wrap)]
    fn test_spillable_sort_descending_with_spilling() {
        use tempfile::TempDir;

        let temp_dir = TempDir::new().unwrap();
        let manager = Arc::new(SpillManager::new(temp_dir.path()).unwrap());

        let mut sort = SpillableSortPushOperator::descending_with_spilling(0, manager, 3);
        let mut sink = CollectorSink::new();

        sort.push(create_test_chunk(&[1, 3, 5]), &mut sink).unwrap();
        sort.push(create_test_chunk(&[2, 4, 6]), &mut sink).unwrap();
        sort.finalize(&mut sink).unwrap();

        let chunks = sink.into_chunks();
        let col = chunks[0].column(0).unwrap();

        // Should be descending: 6, 5, 4, 3, 2, 1
        for i in 0..6 {
            assert_eq!(col.get_value(i), Some(Value::Int64((6 - i) as i64)));
        }
    }

    /// The spilled sort's finish: its runs merge a batch at a time within the
    /// statement's memory, into the rows and order of the sort in memory.
    #[cfg(feature = "spill")]
    mod spilled {
        use super::*;
        use crate::execution::chunk::DEFAULT_CHUNK_SIZE;
        use crate::execution::memory::{OperatorMemoryContext, QueryMemory};
        use crate::execution::spill::{MERGE_FAN_IN, MERGE_MIN_READ_BUFFER};
        use grafeo_common::memory::buffer::BufferManager;
        use grafeo_common::types::{PropertyKey, Timestamp, ZonedDatetime};

        /// `rows` as text: NaN equals itself and -0.0 differs from 0.0 there,
        /// as the order has them.
        fn shown(rows: &[Vec<Value>]) -> String {
            format!("{rows:?}")
        }

        /// A spill directory and its manager.
        fn spill_dir() -> (tempfile::TempDir, Arc<SpillManager>) {
            let dir = tempfile::tempdir().unwrap();
            let manager = Arc::new(SpillManager::new(dir.path().join("spill")).unwrap());
            (dir, manager)
        }

        fn files_left(dir: &tempfile::TempDir) -> usize {
            std::fs::read_dir(dir.path().join("spill")).map_or(0, |entries| entries.count())
        }

        /// A statement's context with a query memory limit and a spill path.
        fn context(limit: usize, manager: &Arc<SpillManager>) -> OperatorMemoryContext {
            OperatorMemoryContext::new(BufferManager::with_budget(256 << 20))
                .with_spill_manager(Some(Arc::clone(manager)))
                .with_query_limit(limit)
        }

        /// `rows` rows in chunks of `chunk_rows`: a key of seven values and
        /// nulls (most rows tie), a column of mixed types, a datetime that is
        /// local or zoned, and the row's place in the input, which only the
        /// tie rule orders.
        fn chunks(rows: usize, chunk_rows: usize) -> Vec<DataChunk> {
            let row = |place: usize| -> [Value; 4] {
                let place = i64::try_from(place).unwrap();
                let key = match place * 7919 % 8 {
                    7 => Value::Null,
                    k => Value::Int64(k - 3),
                };
                let mixed = match place % 10 {
                    0 => Value::Int64(place % 3),
                    1 => Value::Float64(1.5),
                    2 => Value::from(format!("s{}", place % 4)),
                    3 => Value::Null,
                    4 => Value::Bool(place % 2 == 0),
                    5 => Value::Float64(f64::NAN),
                    6 => Value::Float64(if place % 4 == 2 { -0.0 } else { 0.0 }),
                    7 => Value::List(vec![Value::Int64(place % 2), Value::Null].into()),
                    8 => Value::Map(Arc::new(
                        [(PropertyKey::new("k"), Value::Int64(place % 3))]
                            .into_iter()
                            .collect(),
                    )),
                    _ => Value::Float64(f64::NEG_INFINITY),
                };
                // A local 10:00 or 11:00, or that instant in UTC shown at
                // +01:00: a local 11:00 read at +01:00 is the zoned 10:00 UTC.
                let at = Timestamp::from_secs(36_000 + (place % 2) * 3600);
                let when = if place % 3 == 0 {
                    Value::Timestamp(at)
                } else {
                    Value::ZonedDatetime(ZonedDatetime::from_timestamp_offset(at, 3600))
                };
                [key, mixed, when, Value::Int64(place)]
            };
            (0..rows)
                .collect::<Vec<_>>()
                .chunks(chunk_rows)
                .map(|places| {
                    let rows: Vec<[Value; 4]> = places.iter().map(|&place| row(place)).collect();
                    DataChunk::new(
                        (0..4)
                            .map(|column| {
                                let values: Vec<Value> =
                                    rows.iter().map(|row| row[column].clone()).collect();
                                ValueVector::from_values(&values)
                            })
                            .collect(),
                    )
                })
                .collect()
        }

        /// The rows `sort` gives for `input`, chunk by chunk.
        fn run(mut sort: SpillableSortPushOperator, input: Vec<DataChunk>) -> Vec<Vec<Vec<Value>>> {
            let mut sink = CollectorSink::new();
            for chunk in input {
                sort.push(chunk, &mut sink).unwrap();
            }
            sort.finalize(&mut sink).unwrap();
            sink.into_chunks().iter().map(rows_of).collect()
        }

        fn rows_of(chunk: &DataChunk) -> Vec<Vec<Value>> {
            chunk
                .selected_indices()
                .map(|row| {
                    (0..chunk.column_count())
                        .map(|column| chunk.column(column).unwrap().get_value(row).unwrap())
                        .collect()
                })
                .collect()
        }

        fn key(column: usize, descending: bool, nulls_first: bool) -> SortKey {
            SortKey {
                column,
                direction: if descending {
                    SortDirection::Descending
                } else {
                    SortDirection::Ascending
                },
                null_order: if nulls_first {
                    NullOrder::First
                } else {
                    NullOrder::Last
                },
            }
        }

        /// The edge cases of the order: one key, two keys, mixed types first,
        /// nulls first and last, both directions, a datetime key under both
        /// readings of local against zoned; at a run per chunk past the
        /// fan-in (passes), and at runs of several chunks. The oracle is the
        /// same sort in memory: the same rows, ties in input order.
        #[test]
        fn a_spilled_sort_gives_the_in_memory_order_for_every_order() {
            let orders = [
                (vec![key(0, false, false)], TemporalComparison::TypesApart),
                (
                    vec![key(0, true, true), key(1, false, false)],
                    TemporalComparison::TypesApart,
                ),
                (
                    vec![key(1, true, false), key(0, false, true)],
                    TemporalComparison::TypesApart,
                ),
                (
                    vec![key(2, false, false), key(0, true, false)],
                    TemporalComparison::LocalAt(3600),
                ),
                (
                    vec![key(2, true, true), key(0, false, false)],
                    TemporalComparison::TypesApart,
                ),
            ];
            for (keys, temporal) in orders {
                let want: Vec<Vec<Value>> = run(
                    SpillableSortPushOperator::new(keys.clone()).with_temporal(temporal),
                    chunks(4000, 100),
                )
                .concat();
                assert_eq!(want.len(), 4000);
                // 40 runs of a chunk each (passes), then 4 runs of ten.
                for threshold in [1, 1000] {
                    let (dir, manager) = spill_dir();
                    let got = run(
                        SpillableSortPushOperator::with_spilling(keys.clone(), manager, threshold)
                            .with_temporal(temporal),
                        chunks(4000, 100),
                    );
                    assert!(got.len() > 1, "{keys:?}: one chunk of every row");
                    assert_eq!(
                        shown(&got.concat()),
                        shown(&want),
                        "{keys:?} at {threshold}"
                    );
                    assert_eq!(files_left(&dir), 0, "{keys:?} at {threshold}");
                }
            }
        }

        /// What the sink sees: each chunk's rows, and the statement's memory
        /// in use at each chunk.
        struct Watching {
            query: Arc<QueryMemory>,
            limit: usize,
            chunks: Vec<Vec<Vec<Value>>>,
            least_used: usize,
        }

        impl Sink for Watching {
            fn consume(&mut self, chunk: DataChunk) -> Result<bool, OperatorError> {
                let used = self.query.used();
                assert!(used <= self.limit, "{used} of {} in use", self.limit);
                self.least_used = self.least_used.min(used);
                self.chunks.push(rows_of(&chunk));
                Ok(true)
            }

            fn finalize(&mut self) -> Result<(), OperatorError> {
                Ok(())
            }

            fn name(&self) -> &'static str {
                "Watching"
            }

            fn into_any(self: Box<Self>) -> Box<dyn std::any::Any> {
                self
            }
        }

        /// Past a limit far below its input, the sort spills more runs than
        /// one pass merges, and its finish hands the rows out a chunk at a
        /// time, none past the batch size, while the merge's buffers are
        /// counted (the statement's memory is in use at every chunk; that it
        /// stays within the limit is the statement memory's own refusal).
        /// The rows are the in-memory sort's, and no file is left.
        #[test]
        fn a_spilled_sort_finishes_a_chunk_at_a_time_within_the_limit() {
            let keys = vec![key(0, true, false)];
            let want: Vec<Vec<Value>> = run(
                SpillableSortPushOperator::new(keys.clone()),
                chunks(64 * 2048, 2048),
            )
            .concat();
            let (dir, manager) = spill_dir();
            let limit = 256 << 10;
            let ctx = context(limit, &manager);
            let mut sink = Watching {
                query: Arc::clone(ctx.query()),
                limit,
                chunks: Vec::new(),
                least_used: usize::MAX,
            };
            let mut sort = SpillableSortPushOperator::with_memory_context(keys, ctx);
            for chunk in chunks(64 * 2048, 2048) {
                sort.push(chunk, &mut sink).unwrap();
            }
            assert!(sort.runs.as_ref().unwrap().run_count() > MERGE_FAN_IN);
            sort.finalize(&mut sink).unwrap();
            assert!(sink.chunks.len() > 1, "one chunk of every row");
            assert!(
                sink.chunks
                    .iter()
                    .all(|chunk| chunk.len() <= DEFAULT_CHUNK_SIZE)
            );
            assert!(sink.least_used > 0, "the merge's buffers are counted");
            assert_eq!(shown(&sink.chunks.concat()), shown(&want));
            assert_eq!(files_left(&dir), 0);
        }

        /// Below what the merge needs at its narrowest (two runs' smallest
        /// read buffers and a row each), the finish refuses naming the query
        /// memory limit, and leaves no file.
        #[test]
        fn a_spilled_sort_below_the_merge_floor_refuses_and_leaves_no_file() {
            let (dir, manager) = spill_dir();
            let limit = 2 * 2 * MERGE_MIN_READ_BUFFER - 1;
            let mut sort = SpillableSortPushOperator::with_memory_context(
                vec![key(0, false, false)],
                context(limit, &manager),
            );
            let mut sink = CollectorSink::new();
            for chunk in chunks(400, 100) {
                sort.push(chunk, &mut sink).unwrap();
            }
            assert!(sort.runs.as_ref().unwrap().run_count() > 1);
            let error = sort.finalize(&mut sink).expect_err("refused");
            assert!(error.to_string().contains("query memory limit"), "{error}");
            assert!(sink.into_chunks().is_empty());
            assert_eq!(files_left(&dir), 0);
        }

        /// A sink that stops after its first chunk (`Ok(false)`), or fails at
        /// its second.
        struct Stopping {
            fail: bool,
            chunks: usize,
        }

        impl Sink for Stopping {
            fn consume(&mut self, _chunk: DataChunk) -> Result<bool, OperatorError> {
                self.chunks += 1;
                match (self.fail, self.chunks) {
                    (false, _) => Ok(false),
                    (true, 1) => Ok(true),
                    (true, _) => Err(OperatorError::Execution("downstream".into())),
                }
            }

            fn finalize(&mut self) -> Result<(), OperatorError> {
                Ok(())
            }

            fn name(&self) -> &'static str {
                "Stopping"
            }

            fn into_any(self: Box<Self>) -> Box<dyn std::any::Any> {
                self
            }
        }

        /// A sink that stops early ends the merge there, and one that fails
        /// fails the finish; either way no file is left once the finish
        /// returns, with the operator still alive.
        #[test]
        fn a_sink_that_stops_or_fails_ends_the_merge_and_its_files() {
            for fail in [false, true] {
                let (dir, manager) = spill_dir();
                let mut sort = SpillableSortPushOperator::with_spilling(
                    vec![key(0, false, false)],
                    manager,
                    1,
                );
                let mut sink = Stopping { fail, chunks: 0 };
                for chunk in chunks(8 * 2048, 2048) {
                    sort.push(chunk, &mut sink).unwrap();
                }
                assert!(files_left(&dir) > 0);
                let finished = sort.finalize(&mut sink);
                assert_eq!(finished.is_err(), fail);
                assert_eq!(sink.chunks, if fail { 2 } else { 1 });
                assert_eq!(files_left(&dir), 0, "fail: {fail}");
                drop(sort);
            }
        }

        /// A pipeline source of given chunks.
        struct Given(std::vec::IntoIter<DataChunk>);

        impl crate::execution::pipeline::Source for Given {
            fn next_chunk(&mut self, _size: usize) -> Result<Option<DataChunk>, OperatorError> {
                Ok(self.0.next())
            }

            fn reset(&mut self) {}

            fn name(&self) -> &'static str {
                "Given"
            }
        }

        /// The sort, counting the rows its finish hands on.
        struct Counted {
            sort: SpillableSortPushOperator,
            rows: Arc<std::sync::atomic::AtomicUsize>,
        }

        struct Counting<'a> {
            sink: &'a mut dyn Sink,
            rows: &'a std::sync::atomic::AtomicUsize,
        }

        impl Sink for Counting<'_> {
            fn consume(&mut self, chunk: DataChunk) -> Result<bool, OperatorError> {
                self.rows
                    .fetch_add(chunk.len(), std::sync::atomic::Ordering::Relaxed);
                self.sink.consume(chunk)
            }

            fn finalize(&mut self) -> Result<(), OperatorError> {
                self.sink.finalize()
            }

            fn name(&self) -> &'static str {
                "Counting"
            }

            fn into_any(self: Box<Self>) -> Box<dyn std::any::Any> {
                unreachable!("a borrowing sink is never unboxed")
            }
        }

        impl PushOperator for Counted {
            fn push(
                &mut self,
                chunk: DataChunk,
                sink: &mut dyn Sink,
            ) -> Result<bool, OperatorError> {
                self.sort.push(chunk, sink)
            }

            fn finalize(&mut self, sink: &mut dyn Sink) -> Result<(), OperatorError> {
                let mut counting = Counting {
                    sink,
                    rows: &self.rows,
                };
                self.sort.finalize(&mut counting)
            }

            fn name(&self) -> &'static str {
                "Counted"
            }
        }

        /// In a pipeline, a limit after a spilled sort stops its merge: the
        /// sort hands on one batch of its 16,384 rows, the rows are the
        /// in-memory sort's first ten, and no file is left. Eight runs, one
        /// pass: every row read back is one the merge handed on.
        #[test]
        fn a_limit_after_a_spilled_sort_in_a_pipeline_stops_its_merge() {
            use crate::execution::operators::LimitPushOperator;
            use crate::execution::pipeline::Pipeline;

            let keys = vec![key(0, true, false), key(3, false, false)];
            let want: Vec<Vec<Value>> = run(
                SpillableSortPushOperator::new(keys.clone()),
                chunks(8 * 2048, 2048),
            )
            .concat()
            .into_iter()
            .take(10)
            .collect();
            let (dir, manager) = spill_dir();
            let handed_on = Arc::new(std::sync::atomic::AtomicUsize::new(0));
            let sort = Counted {
                sort: SpillableSortPushOperator::with_spilling(keys, manager, 2048),
                rows: Arc::clone(&handed_on),
            };
            let mut pipeline = Pipeline::new(
                Box::new(Given(chunks(8 * 2048, 2048).into_iter())),
                vec![Box::new(sort), Box::new(LimitPushOperator::new(10))],
                Box::new(CollectorSink::new()),
            );
            pipeline.execute().unwrap();
            let handed_on = handed_on.load(std::sync::atomic::Ordering::Relaxed);
            assert!(
                handed_on <= DEFAULT_CHUNK_SIZE,
                "the merge handed on {handed_on} of {} rows after the limit had its ten",
                8 * 2048
            );
            let sink = pipeline
                .into_sink()
                .into_any()
                .downcast::<CollectorSink>()
                .unwrap();
            let got: Vec<Vec<Value>> = sink.into_chunks().iter().flat_map(rows_of).collect();
            assert_eq!(shown(&got), shown(&want));
            assert_eq!(files_left(&dir), 0);
        }

        /// Fails at its `fail_at`-th chunk, or at its finalize when `None`.
        struct Fails {
            fail_at: Option<usize>,
            seen: usize,
        }

        impl PushOperator for Fails {
            fn push(
                &mut self,
                chunk: DataChunk,
                sink: &mut dyn Sink,
            ) -> Result<bool, OperatorError> {
                self.seen += 1;
                if self.fail_at == Some(self.seen) {
                    return Err(OperatorError::Execution("downstream".into()));
                }
                sink.consume(chunk)
            }

            fn finalize(&mut self, _sink: &mut dyn Sink) -> Result<(), OperatorError> {
                match self.fail_at {
                    None => Err(OperatorError::Execution("at the finalize".into())),
                    Some(_) => Ok(()),
                }
            }

            fn name(&self) -> &'static str {
                "Fails"
            }
        }

        /// In a pipeline, a spilled sort leaves no file when an operator
        /// after it fails while the merge streams (at its second chunk), or
        /// fails at its own finalize after a limit stopped the merge.
        #[test]
        fn a_spilled_sort_in_a_pipeline_leaves_no_file_after_an_error() {
            use crate::execution::operators::LimitPushOperator;
            use crate::execution::pipeline::Pipeline;

            let streaming: fn() -> Vec<Box<dyn PushOperator>> = || {
                vec![Box::new(Fails {
                    fail_at: Some(2),
                    seen: 0,
                })]
            };
            let after_a_stop: fn() -> Vec<Box<dyn PushOperator>> = || {
                vec![
                    Box::new(LimitPushOperator::new(10)),
                    Box::new(Fails {
                        fail_at: None,
                        seen: 0,
                    }),
                ]
            };
            for after in [streaming, after_a_stop] {
                let (dir, manager) = spill_dir();
                let mut operators: Vec<Box<dyn PushOperator>> =
                    vec![Box::new(SpillableSortPushOperator::with_spilling(
                        vec![key(0, false, false)],
                        manager,
                        2048,
                    ))];
                operators.extend(after());
                let mut pipeline = Pipeline::new(
                    Box::new(Given(chunks(8 * 2048, 2048).into_iter())),
                    operators,
                    Box::new(CollectorSink::new()),
                );
                let error = pipeline.execute().expect_err("an operator fails");
                assert!(
                    error.to_string().contains("downstream")
                        || error.to_string().contains("finalize"),
                    "{error}"
                );
                assert_eq!(files_left(&dir), 0, "{error}");
                drop(pipeline);
            }
        }

        /// Rows whose place (column 3) is a multiple of 100.
        struct EveryHundredth;

        impl crate::execution::operators::push::FilterPredicate for EveryHundredth {
            fn evaluate(&self, chunk: &DataChunk, row: usize) -> Result<bool, OperatorError> {
                Ok(matches!(
                    chunk.column(3).and_then(|column| column.get_value(row)),
                    Some(Value::Int64(place)) if place % 100 == 0
                ))
            }
        }

        /// The rows a pipeline of `operators` gives for 16,384 rows in
        /// chunks of 2,048.
        fn piped(operators: Vec<Box<dyn PushOperator>>) -> Vec<Vec<Value>> {
            let mut pipeline = crate::execution::pipeline::Pipeline::new(
                Box::new(Given(chunks(8 * 2048, 2048).into_iter())),
                operators,
                Box::new(CollectorSink::new()),
            );
            pipeline.execute().unwrap();
            let sink = pipeline
                .into_sink()
                .into_any()
                .downcast::<CollectorSink>()
                .unwrap();
            sink.into_chunks().iter().flat_map(rows_of).collect()
        }

        /// A spilled sort followed by a distinct, an aggregate or a filter,
        /// and an aggregate followed by a spilled sort (a breaker after a
        /// breaker), give the rows, in order, of the same pipeline with the
        /// sort in memory; no file is left.
        #[test]
        fn breakers_and_filters_around_a_spilled_sort_give_the_in_memory_rows() {
            use crate::execution::operators::AggregateExpr;
            use crate::execution::operators::push::{
                AggregatePushOperator, DistinctPushOperator, FilterPushOperator,
            };

            let after: [fn() -> Box<dyn PushOperator>; 3] = [
                || Box::new(DistinctPushOperator::on_columns(vec![0, 1])),
                || {
                    Box::new(AggregatePushOperator::new(
                        vec![0],
                        vec![AggregateExpr::count_star()],
                    ))
                },
                || Box::new(FilterPushOperator::new(Box::new(EveryHundredth))),
            ];
            let keys = vec![key(1, false, false), key(3, true, false)];
            for (shape, then) in after.iter().enumerate() {
                let want = piped(vec![
                    Box::new(SpillableSortPushOperator::new(keys.clone())),
                    then(),
                ]);
                let (dir, manager) = spill_dir();
                let got = piped(vec![
                    Box::new(SpillableSortPushOperator::with_spilling(
                        keys.clone(),
                        manager,
                        2048,
                    )),
                    then(),
                ]);
                assert!(!want.is_empty(), "shape {shape}");
                assert_eq!(shown(&got), shown(&want), "shape {shape}");
                assert_eq!(files_left(&dir), 0, "shape {shape}");
            }

            // A group per row (its place), then a sort by the count and the
            // place, descending: the sort gets every group and spills.
            let group = || -> Box<dyn PushOperator> {
                Box::new(AggregatePushOperator::new(
                    vec![3],
                    vec![AggregateExpr::count_star()],
                ))
            };
            let keys = vec![key(1, true, false), key(0, true, false)];
            let want = piped(vec![
                group(),
                Box::new(SpillableSortPushOperator::new(keys.clone())),
            ]);
            let (dir, manager) = spill_dir();
            let got = piped(vec![
                group(),
                Box::new(SpillableSortPushOperator::with_spilling(
                    keys, manager, 2048,
                )),
            ]);
            assert_eq!(want.len(), 8 * 2048);
            assert_eq!(shown(&got), shown(&want), "a sort after an aggregate");
            assert_eq!(files_left(&dir), 0);
        }

        /// An operator that keeps every row it gets without spilling: on its
        /// first chunk it takes all the statement's memory has left, and keeps
        /// its rows within that.
        struct Keeper {
            memory: crate::execution::memory::OperatorMemory,
            query: Arc<QueryMemory>,
            limit: usize,
            rows: usize,
        }

        impl PushOperator for Keeper {
            fn push(
                &mut self,
                chunk: DataChunk,
                _sink: &mut dyn Sink,
            ) -> Result<bool, OperatorError> {
                if self.rows == 0 {
                    self.memory.set(self.limit - self.query.used())?;
                }
                self.rows += chunk.len();
                Ok(true)
            }

            fn finalize(&mut self, _sink: &mut dyn Sink) -> Result<(), OperatorError> {
                Ok(())
            }

            fn name(&self) -> &'static str {
                "Keeper"
            }
        }

        /// Eight chunks of 16 rows, each a key and 6,000 bytes of text: a row
        /// bigger than a grant step (4 KiB under a 256 KiB cap), so the merge
        /// grows its grant again for each batch after giving the room back.
        fn wide_chunks() -> Vec<DataChunk> {
            (0..8_i64)
                .map(|chunk| {
                    let keys: Vec<Value> = (0..16_i64)
                        .map(|row| Value::Int64((chunk * 16 + row) * 7919 % 128))
                        .collect();
                    let texts: Vec<Value> = (0..16_i64)
                        .map(|row| Value::from("x".repeat(6000) + &(chunk * 16 + row).to_string()))
                        .collect();
                    DataChunk::new(vec![
                        ValueVector::from_values(&keys),
                        ValueVector::from_values(&texts),
                    ])
                })
                .collect()
        }

        /// After the merge gave a batch's room back, the operator after it
        /// took that room and keeps it: once not a row more fits, the merge
        /// refuses, `GRAFEO-S001` naming the merge, and hands on no row past
        /// its grant; no file is left.
        #[test]
        fn a_merge_that_cannot_take_back_its_room_refuses_naming_itself() {
            let (dir, manager) = spill_dir();
            let limit = 256 << 10;
            let ctx = context(limit, &manager);
            let keeper = Keeper {
                memory: crate::execution::memory::OperatorMemory::new(ctx.clone()),
                query: Arc::clone(ctx.query()),
                limit,
                rows: 0,
            };
            let mut pipeline = crate::execution::pipeline::Pipeline::new(
                Box::new(Given(wide_chunks().into_iter())),
                vec![
                    Box::new(SpillableSortPushOperator::with_memory_context(
                        vec![key(0, false, false)],
                        ctx.clone(),
                    )),
                    Box::new(keeper),
                ],
                Box::new(CollectorSink::new()),
            );
            let error = pipeline.execute().expect_err("refused");
            let message = error.to_string();
            let OperatorError::Wrapped(inner) = &error else {
                panic!("not a refusal: {message}");
            };
            assert_eq!(inner.error_code().as_str(), "GRAFEO-S001", "{message}");
            assert!(message.contains("merge"), "names the merge: {message}");
            assert!(message.contains("query memory limit"), "{message}");
            drop(pipeline);
            assert_eq!(files_left(&dir), 0);
        }
    }
}
