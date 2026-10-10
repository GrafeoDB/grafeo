//! Push-based sort operator (pipeline breaker).

use crate::execution::chunk::{ColumnTypes, DataChunk, DataChunkBuilder};
use crate::execution::operators::OperatorError;
use crate::execution::operators::value_utils::order_by;
use crate::execution::pipeline::{ChunkSizeHint, PushOperator, Sink};
#[cfg(feature = "spill")]
use crate::execution::spill::{ExternalSort, SpillManager};
use grafeo_common::types::{LogicalType, Value};
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
}

/// Compare two rows by sort keys.
fn compare_rows(a: &[Value], b: &[Value], keys: &[SortKey]) -> Ordering {
    for key in keys {
        let ordering = order_by(
            a.get(key.column),
            b.get(key.column),
            key.direction == SortDirection::Descending,
            key.null_order == NullOrder::First,
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

/// One chunk with the first `width` values of each sorted row, in columns of
/// the input's types (a node or edge column stays one, see `ColumnTypes`).
fn output_chunk(rows: &[Vec<Value>], column_types: &ColumnTypes, width: usize) -> DataChunk {
    let types: Vec<LogicalType> = (0..width)
        .map(|column| {
            column_types
                .types()
                .get(column)
                .cloned()
                .unwrap_or(LogicalType::Any)
        })
        .collect();
    let mut builder = DataChunkBuilder::with_capacity(&types, rows.len());
    for row in rows {
        for column in 0..width {
            if let Some(out) = builder.column_mut(column) {
                out.push_value(row.get(column).cloned().unwrap_or(Value::Null));
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

        // Buffer all rows
        for i in chunk.selected_indices() {
            let mut row = Vec::with_capacity(num_cols);
            for col_idx in 0..num_cols {
                let val = chunk
                    .column(col_idx)
                    .and_then(|c| c.get_value(i))
                    .unwrap_or(Value::Null);
                row.push(val);
            }
            self.buffer.push(row);
        }

        Ok(true)
    }

    fn finalize(&mut self, sink: &mut dyn Sink) -> Result<(), OperatorError> {
        if self.buffer.is_empty() {
            return Ok(());
        }

        // Sort the buffer - borrow keys separately to avoid borrow conflict
        let keys = &self.keys;
        self.buffer.sort_by(|a, b| compare_rows(a, b, keys));

        let num_cols = self.num_columns.unwrap_or(0);
        if num_cols == 0 {
            return Ok(());
        }
        let chunk = output_chunk(
            &self.buffer,
            &self.column_types,
            output_width(num_cols, self.output_width),
        );
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

/// Minimum buffer size (rows) before memory-pressure spilling can trigger.
///
/// Prevents "noisy neighbor" scenarios where a tiny sort buffer gets spilled
/// because unrelated subsystems consumed memory.
#[cfg(feature = "spill")]
const SORT_MIN_BUFFER_ROWS: usize = 1000;

/// Push-based sort operator with spilling support.
///
/// This is a pipeline breaker that buffers input and spills to disk
/// when memory pressure is high. It uses external merge sort for
/// out-of-core sorting.
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
    /// Spill manager for file creation (used by row-count fallback mode).
    spill_manager: Option<Arc<SpillManager>>,
    /// External sort state (created when first spill occurs).
    external_sort: Option<ExternalSort>,
    /// Threshold to trigger spill (row count, used by fallback mode).
    spill_threshold: usize,
    /// Memory context for pressure-aware spilling.
    memory_ctx: Option<crate::execution::memory::OperatorMemoryContext>,
    /// Shared state with the registered MemoryConsumer adapter.
    spill_state: Option<std::sync::Arc<super::spill_state::OperatorSpillState>>,
    /// Running total of estimated buffer memory in bytes (incremental tracking).
    estimated_bytes: usize,
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
            spill_manager: None,
            external_sort: None,
            spill_threshold: DEFAULT_SPILL_THRESHOLD,
            memory_ctx: None,
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
            spill_manager: Some(manager),
            external_sort: None,
            spill_threshold: threshold,
            memory_ctx: None,
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

        let state = std::sync::Arc::new(OperatorSpillState::new("SpillableSortPush".to_string()));
        let adapter =
            std::sync::Arc::new(OperatorConsumerAdapter::new(std::sync::Arc::clone(&state)));
        ctx.register_consumer(adapter);

        Self {
            keys,
            buffer: Vec::new(),
            num_columns: None,
            output_width: None,
            column_types: ColumnTypes::default(),
            spill_manager: None,
            external_sort: None,
            spill_threshold: DEFAULT_SPILL_THRESHOLD,
            memory_ctx: Some(ctx),
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

        if !should_spill {
            return Ok(());
        }

        // Get SpillManager: prefer memory_ctx, fall back to self.spill_manager
        let manager = self
            .memory_ctx
            .as_ref()
            .map(|c| std::sync::Arc::clone(c.spill_manager()))
            .or_else(|| self.spill_manager.clone());

        let Some(manager) = manager else {
            return Ok(()); // No spilling configured
        };

        // Sort current buffer
        let keys = &self.keys;
        self.buffer.sort_by(|a, b| compare_rows(a, b, keys));

        // Initialize external sort if needed
        if self.external_sort.is_none() {
            let num_cols = self.num_columns.unwrap_or(0);
            let spill_keys = self
                .keys
                .iter()
                .map(|k| crate::execution::spill::SortKey {
                    column: k.column,
                    direction: match k.direction {
                        SortDirection::Ascending => {
                            crate::execution::spill::SortDirection::Ascending
                        }
                        SortDirection::Descending => {
                            crate::execution::spill::SortDirection::Descending
                        }
                    },
                    null_order: match k.null_order {
                        NullOrder::First => crate::execution::spill::NullOrder::First,
                        NullOrder::Last => crate::execution::spill::NullOrder::Last,
                    },
                })
                .collect();

            self.external_sort = Some(ExternalSort::new(manager, num_cols, spill_keys));
        }

        // Spill as sorted run
        let buffer = std::mem::take(&mut self.buffer);
        if let Some(ref mut ext) = self.external_sort {
            ext.spill_sorted_run(buffer)
                .map_err(|e| OperatorError::from(grafeo_common::utils::error::Error::Io(e)))?;
        }

        // Reset memory tracking after spill
        self.estimated_bytes = 0;
        if let Some(ref state) = self.spill_state {
            state.set_usage(0);
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

        // Get sorted rows: either from external merge or in-memory sort
        let sorted_rows = if let Some(ref mut ext) = self.external_sort {
            // Merge all runs with remaining buffer
            let buffer = std::mem::take(&mut self.buffer);
            ext.merge_all(buffer)
                .map_err(|e| OperatorError::from(grafeo_common::utils::error::Error::Io(e)))?
        } else {
            // No spilling occurred: just sort in memory
            let keys = &self.keys;
            self.buffer.sort_by(|a, b| compare_rows(a, b, keys));
            std::mem::take(&mut self.buffer)
        };

        // Unregister consumer before emitting results (we no longer hold buffer memory)
        self.unregister_consumer();

        if sorted_rows.is_empty() {
            return Ok(());
        }
        let chunk = output_chunk(
            &sorted_rows,
            &self.column_types,
            output_width(num_cols, self.output_width),
        );
        sink.consume(chunk)?;

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
        assert!(spilled.external_sort.is_some(), "the rows were spilled");
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
}
