//! Push-based sort operator (pipeline breaker).

use crate::execution::chunk::DataChunk;
use crate::execution::operators::OperatorError;
use crate::execution::pipeline::{ChunkSizeHint, PushOperator, Sink};
use crate::execution::spill::{ExternalSort, SpillManager};
use crate::execution::vector::ValueVector;
use grafeo_common::types::Value;
use std::cmp::Ordering;
use std::sync::Arc;

/// Sort direction.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SortDirection {
    /// Ascending order.
    Ascending,
    /// Descending order.
    Descending,
}

/// Null handling in sort.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
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
}

impl SortPushOperator {
    /// Create a new sort operator with the given sort keys.
    pub fn new(keys: Vec<SortKey>) -> Self {
        Self {
            keys,
            buffer: Vec::new(),
            num_columns: None,
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
}

/// Compare two rows by sort keys.
fn compare_rows(a: &[Value], b: &[Value], keys: &[SortKey]) -> Ordering {
    for key in keys {
        let a_val = a.get(key.column);
        let b_val = b.get(key.column);

        let ordering = match (a_val, b_val) {
            (Some(Value::Null), Some(Value::Null)) => Ordering::Equal,
            (Some(Value::Null), _) => match key.null_order {
                NullOrder::First => Ordering::Less,
                NullOrder::Last => Ordering::Greater,
            },
            (_, Some(Value::Null)) => match key.null_order {
                NullOrder::First => Ordering::Greater,
                NullOrder::Last => Ordering::Less,
            },
            (Some(a), Some(b)) => compare_values(a, b),
            _ => Ordering::Equal,
        };

        let ordering = match key.direction {
            SortDirection::Ascending => ordering,
            SortDirection::Descending => ordering.reverse(),
        };

        if ordering != Ordering::Equal {
            return ordering;
        }
    }

    Ordering::Equal
}

/// Compare two values.
fn compare_values(a: &Value, b: &Value) -> Ordering {
    match (a, b) {
        (Value::Bool(a), Value::Bool(b)) => a.cmp(b),
        (Value::Int64(a), Value::Int64(b)) => a.cmp(b),
        (Value::Float64(a), Value::Float64(b)) => a.partial_cmp(b).unwrap_or(Ordering::Equal),
        (Value::String(a), Value::String(b)) => a.cmp(b),
        _ => Ordering::Equal,
    }
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

        // Sort the buffer - clone keys to avoid borrow issues
        let keys = self.keys.clone();
        self.buffer.sort_by(|a, b| compare_rows(a, b, &keys));

        // Emit sorted rows in chunks
        let num_cols = self.num_columns.unwrap_or(0);
        if num_cols == 0 {
            return Ok(());
        }

        // Build output chunk from sorted rows
        let mut columns: Vec<ValueVector> = (0..num_cols).map(|_| ValueVector::new()).collect();

        for row in &self.buffer {
            for (col_idx, col) in columns.iter_mut().enumerate() {
                let val = row.get(col_idx).cloned().unwrap_or(Value::Null);
                col.push(val);
            }
        }

        let chunk = DataChunk::new(columns);
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
pub const DEFAULT_SPILL_THRESHOLD: usize = 100_000;

/// Push-based sort operator with spilling support.
///
/// This is a pipeline breaker that buffers input and spills to disk
/// when memory pressure is high. It uses external merge sort for
/// out-of-core sorting.
pub struct SpillableSortPushOperator {
    /// Sort keys.
    keys: Vec<SortKey>,
    /// Buffered rows as (row_values...).
    buffer: Vec<Vec<Value>>,
    /// Number of columns per row.
    num_columns: Option<usize>,
    /// Spill manager for file creation.
    spill_manager: Option<Arc<SpillManager>>,
    /// External sort state (created when first spill occurs).
    external_sort: Option<ExternalSort>,
    /// Threshold to trigger spill (row count).
    spill_threshold: usize,
}

impl SpillableSortPushOperator {
    /// Create a new spillable sort operator with the given sort keys.
    pub fn new(keys: Vec<SortKey>) -> Self {
        Self {
            keys,
            buffer: Vec::new(),
            num_columns: None,
            spill_manager: None,
            external_sort: None,
            spill_threshold: DEFAULT_SPILL_THRESHOLD,
        }
    }

    /// Create a new spillable sort operator with spilling enabled.
    pub fn with_spilling(keys: Vec<SortKey>, manager: Arc<SpillManager>, threshold: usize) -> Self {
        Self {
            keys,
            buffer: Vec::new(),
            num_columns: None,
            spill_manager: Some(manager),
            external_sort: None,
            spill_threshold: threshold,
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

    /// Sets the spill threshold.
    pub fn with_threshold(mut self, threshold: usize) -> Self {
        self.spill_threshold = threshold;
        self
    }

    /// Spills the current buffer as a sorted run.
    fn maybe_spill(&mut self) -> Result<(), OperatorError> {
        if self.buffer.len() < self.spill_threshold {
            return Ok(());
        }

        let Some(manager) = &self.spill_manager else {
            return Ok(()); // No spilling configured
        };

        // Sort current buffer
        let keys = self.keys.clone();
        self.buffer.sort_by(|a, b| compare_rows(a, b, &keys));

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

            self.external_sort = Some(ExternalSort::new(Arc::clone(manager), num_cols, spill_keys));
        }

        // Spill as sorted run
        let buffer = std::mem::take(&mut self.buffer);
        if let Some(ref mut ext) = self.external_sort {
            ext.spill_sorted_run(buffer)
                .map_err(|e| OperatorError::Execution(e.to_string()))?;
        }

        Ok(())
    }
}

impl PushOperator for SpillableSortPushOperator {
    fn push(&mut self, chunk: DataChunk, _sink: &mut dyn Sink) -> Result<bool, OperatorError> {
        if chunk.is_empty() {
            return Ok(true);
        }

        // Initialize column count
        if self.num_columns.is_none() {
            self.num_columns = Some(chunk.column_count());
        }

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

        // Check if we should spill
        self.maybe_spill()?;

        Ok(true)
    }

    fn finalize(&mut self, sink: &mut dyn Sink) -> Result<(), OperatorError> {
        let num_cols = self.num_columns.unwrap_or(0);
        if num_cols == 0 && self.buffer.is_empty() {
            return Ok(());
        }

        // Get sorted rows - either from external merge or in-memory sort
        let sorted_rows = if let Some(ref mut ext) = self.external_sort {
            // Merge all runs with remaining buffer
            let buffer = std::mem::take(&mut self.buffer);
            ext.merge_all(buffer)
                .map_err(|e| OperatorError::Execution(e.to_string()))?
        } else {
            // No spilling occurred - just sort in memory
            let keys = self.keys.clone();
            self.buffer.sort_by(|a, b| compare_rows(a, b, &keys));
            std::mem::take(&mut self.buffer)
        };

        if sorted_rows.is_empty() {
            return Ok(());
        }

        // Build output chunk from sorted rows
        let mut columns: Vec<ValueVector> = (0..num_cols).map(|_| ValueVector::new()).collect();

        for row in &sorted_rows {
            for (col_idx, col) in columns.iter_mut().enumerate() {
                let val = row.get(col_idx).cloned().unwrap_or(Value::Null);
                col.push(val);
            }
        }

        let chunk = DataChunk::new(columns);
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
