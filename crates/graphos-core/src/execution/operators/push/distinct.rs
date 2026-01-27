//! Push-based distinct operator.

use crate::execution::chunk::DataChunk;
use crate::execution::operators::OperatorError;
use crate::execution::pipeline::{ChunkSizeHint, PushOperator, Sink};
use crate::execution::selection::SelectionVector;
use crate::execution::vector::ValueVector;
use graphos_common::types::Value;
use std::collections::HashSet;

/// Hash key for distinct tracking.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct RowKey(Vec<u64>);

impl RowKey {
    fn from_row(chunk: &DataChunk, row: usize, columns: &[usize]) -> Self {
        let hashes: Vec<u64> = columns
            .iter()
            .map(|&col| {
                chunk
                    .column(col)
                    .and_then(|c| c.get_value(row))
                    .map(|v| hash_value(&v))
                    .unwrap_or(0)
            })
            .collect();
        Self(hashes)
    }

    fn from_all_columns(chunk: &DataChunk, row: usize) -> Self {
        let hashes: Vec<u64> = (0..chunk.column_count())
            .map(|col| {
                chunk
                    .column(col)
                    .and_then(|c| c.get_value(row))
                    .map(|v| hash_value(&v))
                    .unwrap_or(0)
            })
            .collect();
        Self(hashes)
    }
}

fn hash_value(value: &Value) -> u64 {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    let mut hasher = DefaultHasher::new();
    match value {
        Value::Null => 0u8.hash(&mut hasher),
        Value::Bool(b) => b.hash(&mut hasher),
        Value::Int64(i) => i.hash(&mut hasher),
        Value::Float64(f) => f.to_bits().hash(&mut hasher),
        Value::String(s) => s.hash(&mut hasher),
        _ => 0u8.hash(&mut hasher),
    }
    hasher.finish()
}

/// Push-based distinct operator.
///
/// Filters out duplicate rows based on all columns or specified columns.
/// This operator maintains state (seen values) but can produce output
/// incrementally as new unique rows arrive.
pub struct DistinctPushOperator {
    /// Columns to check for distinctness (None = all columns).
    columns: Option<Vec<usize>>,
    /// Set of seen row hashes.
    seen: HashSet<RowKey>,
}

impl DistinctPushOperator {
    /// Create a distinct operator on all columns.
    pub fn new() -> Self {
        Self {
            columns: None,
            seen: HashSet::new(),
        }
    }

    /// Create a distinct operator on specific columns.
    pub fn on_columns(columns: Vec<usize>) -> Self {
        Self {
            columns: Some(columns),
            seen: HashSet::new(),
        }
    }

    /// Get the number of unique rows seen.
    pub fn unique_count(&self) -> usize {
        self.seen.len()
    }
}

impl Default for DistinctPushOperator {
    fn default() -> Self {
        Self::new()
    }
}

impl PushOperator for DistinctPushOperator {
    fn push(&mut self, chunk: DataChunk, sink: &mut dyn Sink) -> Result<bool, OperatorError> {
        if chunk.is_empty() {
            return Ok(true);
        }

        // Find rows that are new (not seen before)
        let mut new_indices = Vec::new();

        for row in chunk.selected_indices() {
            let key = match &self.columns {
                Some(cols) => RowKey::from_row(&chunk, row, cols),
                None => RowKey::from_all_columns(&chunk, row),
            };

            if self.seen.insert(key) {
                new_indices.push(row);
            }
        }

        if new_indices.is_empty() {
            return Ok(true);
        }

        // Create filtered chunk with only new rows
        let selection = SelectionVector::from_predicate(chunk.len(), |i| new_indices.contains(&i));
        let filtered = chunk.filter(&selection);

        sink.consume(filtered)
    }

    fn finalize(&mut self, _sink: &mut dyn Sink) -> Result<(), OperatorError> {
        // Nothing to finalize - all output was produced incrementally
        Ok(())
    }

    fn preferred_chunk_size(&self) -> ChunkSizeHint {
        ChunkSizeHint::Default
    }

    fn name(&self) -> &'static str {
        "DistinctPush"
    }
}

/// Push-based distinct operator that materializes all input first.
///
/// This is a true pipeline breaker that buffers all rows and produces
/// distinct output in the finalize phase. Use this when you need
/// deterministic ordering of output.
pub struct DistinctMaterializingOperator {
    /// Columns to check for distinctness.
    columns: Option<Vec<usize>>,
    /// Buffered unique rows.
    rows: Vec<Vec<Value>>,
    /// Set of seen row hashes.
    seen: HashSet<RowKey>,
    /// Number of columns.
    num_columns: Option<usize>,
}

impl DistinctMaterializingOperator {
    /// Create a distinct operator on all columns.
    pub fn new() -> Self {
        Self {
            columns: None,
            rows: Vec::new(),
            seen: HashSet::new(),
            num_columns: None,
        }
    }

    /// Create a distinct operator on specific columns.
    pub fn on_columns(columns: Vec<usize>) -> Self {
        Self {
            columns: Some(columns),
            rows: Vec::new(),
            seen: HashSet::new(),
            num_columns: None,
        }
    }
}

impl Default for DistinctMaterializingOperator {
    fn default() -> Self {
        Self::new()
    }
}

impl PushOperator for DistinctMaterializingOperator {
    fn push(&mut self, chunk: DataChunk, _sink: &mut dyn Sink) -> Result<bool, OperatorError> {
        if chunk.is_empty() {
            return Ok(true);
        }

        if self.num_columns.is_none() {
            self.num_columns = Some(chunk.column_count());
        }

        let num_cols = chunk.column_count();

        for row in chunk.selected_indices() {
            let key = match &self.columns {
                Some(cols) => RowKey::from_row(&chunk, row, cols),
                None => RowKey::from_all_columns(&chunk, row),
            };

            if self.seen.insert(key) {
                // Store the full row
                let row_values: Vec<Value> = (0..num_cols)
                    .map(|col| {
                        chunk
                            .column(col)
                            .and_then(|c| c.get_value(row))
                            .unwrap_or(Value::Null)
                    })
                    .collect();
                self.rows.push(row_values);
            }
        }

        Ok(true)
    }

    fn finalize(&mut self, sink: &mut dyn Sink) -> Result<(), OperatorError> {
        if self.rows.is_empty() {
            return Ok(());
        }

        let num_cols = self.num_columns.unwrap_or(0);
        let mut columns: Vec<ValueVector> = (0..num_cols).map(|_| ValueVector::new()).collect();

        for row in &self.rows {
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
        ChunkSizeHint::Default
    }

    fn name(&self) -> &'static str {
        "DistinctMaterializing"
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
    fn test_distinct_all_unique() {
        let mut distinct = DistinctPushOperator::new();
        let mut sink = CollectorSink::new();

        distinct
            .push(create_test_chunk(&[1, 2, 3, 4, 5]), &mut sink)
            .unwrap();
        distinct.finalize(&mut sink).unwrap();

        assert_eq!(sink.row_count(), 5);
        assert_eq!(distinct.unique_count(), 5);
    }

    #[test]
    fn test_distinct_with_duplicates() {
        let mut distinct = DistinctPushOperator::new();
        let mut sink = CollectorSink::new();

        distinct
            .push(create_test_chunk(&[1, 2, 1, 3, 2, 1, 4]), &mut sink)
            .unwrap();
        distinct.finalize(&mut sink).unwrap();

        assert_eq!(sink.row_count(), 4); // 1, 2, 3, 4
        assert_eq!(distinct.unique_count(), 4);
    }

    #[test]
    fn test_distinct_all_same() {
        let mut distinct = DistinctPushOperator::new();
        let mut sink = CollectorSink::new();

        distinct
            .push(create_test_chunk(&[5, 5, 5, 5, 5]), &mut sink)
            .unwrap();
        distinct.finalize(&mut sink).unwrap();

        assert_eq!(sink.row_count(), 1);
        assert_eq!(distinct.unique_count(), 1);
    }

    #[test]
    fn test_distinct_multiple_chunks() {
        let mut distinct = DistinctPushOperator::new();
        let mut sink = CollectorSink::new();

        distinct.push(create_test_chunk(&[1, 2, 3]), &mut sink).unwrap();
        distinct.push(create_test_chunk(&[2, 3, 4]), &mut sink).unwrap();
        distinct.push(create_test_chunk(&[3, 4, 5]), &mut sink).unwrap();
        distinct.finalize(&mut sink).unwrap();

        assert_eq!(sink.row_count(), 5); // 1, 2, 3, 4, 5
    }

    #[test]
    fn test_distinct_materializing() {
        let mut distinct = DistinctMaterializingOperator::new();
        let mut sink = CollectorSink::new();

        distinct
            .push(create_test_chunk(&[3, 1, 4, 1, 5, 9, 2, 6]), &mut sink)
            .unwrap();
        distinct.finalize(&mut sink).unwrap();

        // All output comes in finalize
        let chunks = sink.into_chunks();
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].len(), 7); // 7 unique values
    }
}
