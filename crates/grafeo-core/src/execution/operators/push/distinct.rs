//! Push-based distinct operator.

use crate::execution::chunk::DataChunk;
use crate::execution::operators::OperatorError;
use crate::execution::operators::accumulator::RowKey;
use crate::execution::pipeline::{ChunkSizeHint, PushOperator, Sink};
use crate::execution::selection::SelectionVector;
use crate::execution::vector::ValueVector;
use grafeo_common::types::Value;
use std::collections::HashSet;

/// Push-based distinct operator.
///
/// Filters out duplicate rows based on all columns or specified columns.
/// This operator maintains state (seen values) but can produce output
/// incrementally as new unique rows arrive.
pub struct DistinctPushOperator {
    /// Columns to check for distinctness (None = all columns).
    columns: Option<Vec<usize>>,
    /// The keys of the rows seen (see `RowKey`).
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
    /// The keys of the rows seen (see `RowKey`).
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

        distinct
            .push(create_test_chunk(&[1, 2, 3]), &mut sink)
            .unwrap();
        distinct
            .push(create_test_chunk(&[2, 3, 4]), &mut sink)
            .unwrap();
        distinct
            .push(create_test_chunk(&[3, 4, 5]), &mut sink)
            .unwrap();
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

    fn create_mixed_chunk(values: &[Value]) -> DataChunk {
        let vector = ValueVector::from_values(values);
        DataChunk::new(vec![vector])
    }

    #[test]
    fn test_distinct_null_values() {
        let mut distinct = DistinctPushOperator::new();
        let mut sink = CollectorSink::new();

        let chunk = create_mixed_chunk(&[Value::Null, Value::Null, Value::Int64(1)]);
        distinct.push(chunk, &mut sink).unwrap();
        distinct.finalize(&mut sink).unwrap();
        assert_eq!(distinct.unique_count(), 2); // Null + 1
    }

    #[test]
    fn test_distinct_bool_values() {
        let mut distinct = DistinctPushOperator::new();
        let mut sink = CollectorSink::new();

        let chunk = create_mixed_chunk(&[Value::Bool(true), Value::Bool(false), Value::Bool(true)]);
        distinct.push(chunk, &mut sink).unwrap();
        distinct.finalize(&mut sink).unwrap();
        assert_eq!(distinct.unique_count(), 2);
    }

    #[test]
    fn test_distinct_float_values() {
        let mut distinct = DistinctPushOperator::new();
        let mut sink = CollectorSink::new();

        let chunk = create_mixed_chunk(&[
            Value::Float64(1.0),
            Value::Float64(2.0),
            Value::Float64(1.0),
            Value::Float64(f64::NAN),
        ]);
        distinct.push(chunk, &mut sink).unwrap();
        distinct.finalize(&mut sink).unwrap();
        assert_eq!(distinct.unique_count(), 3); // 1.0, 2.0, NaN
    }

    #[test]
    fn test_distinct_string_values() {
        let mut distinct = DistinctPushOperator::new();
        let mut sink = CollectorSink::new();

        let chunk =
            create_mixed_chunk(&[Value::from("Alix"), Value::from("Gus"), Value::from("Alix")]);
        distinct.push(chunk, &mut sink).unwrap();
        distinct.finalize(&mut sink).unwrap();
        assert_eq!(distinct.unique_count(), 2);
    }

    #[test]
    fn test_distinct_bytes_values() {
        let mut distinct = DistinctPushOperator::new();
        let mut sink = CollectorSink::new();

        let chunk = create_mixed_chunk(&[
            Value::Bytes(vec![1u8, 2, 3].into()),
            Value::Bytes(vec![4u8, 5, 6].into()),
            Value::Bytes(vec![1u8, 2, 3].into()),
        ]);
        distinct.push(chunk, &mut sink).unwrap();
        distinct.finalize(&mut sink).unwrap();
        assert_eq!(distinct.unique_count(), 2);
    }

    #[test]
    fn test_distinct_list_values() {
        let mut distinct = DistinctPushOperator::new();
        let mut sink = CollectorSink::new();

        let chunk = create_mixed_chunk(&[
            Value::List(vec![Value::Int64(1), Value::Int64(2)].into()),
            Value::List(vec![Value::Int64(3), Value::Int64(4)].into()),
            Value::List(vec![Value::Int64(1), Value::Int64(2)].into()),
        ]);
        distinct.push(chunk, &mut sink).unwrap();
        distinct.finalize(&mut sink).unwrap();
        assert_eq!(distinct.unique_count(), 2);
    }

    #[test]
    fn test_distinct_map_values() {
        use std::collections::BTreeMap;

        let mut map1 = BTreeMap::new();
        map1.insert("a".into(), Value::Int64(1));
        let mut map2 = BTreeMap::new();
        map2.insert("b".into(), Value::Int64(2));

        let mut distinct = DistinctPushOperator::new();
        let mut sink = CollectorSink::new();

        let chunk = create_mixed_chunk(&[
            Value::Map(map1.clone().into()),
            Value::Map(map2.into()),
            Value::Map(map1.into()),
        ]);
        distinct.push(chunk, &mut sink).unwrap();
        distinct.finalize(&mut sink).unwrap();
        assert_eq!(distinct.unique_count(), 2);
    }

    #[test]
    fn test_distinct_vector_values() {
        let mut distinct = DistinctPushOperator::new();
        let mut sink = CollectorSink::new();

        let chunk = create_mixed_chunk(&[
            Value::Vector(vec![1.0_f32, 2.0].into()),
            Value::Vector(vec![3.0_f32, 4.0].into()),
            Value::Vector(vec![1.0_f32, 2.0].into()),
        ]);
        distinct.push(chunk, &mut sink).unwrap();
        distinct.finalize(&mut sink).unwrap();
        assert_eq!(distinct.unique_count(), 2);
    }

    #[test]
    fn test_distinct_path_values() {
        let mut distinct = DistinctPushOperator::new();
        let mut sink = CollectorSink::new();

        let path1 = Value::Path {
            nodes: vec![Value::Int64(1), Value::Int64(2)].into(),
            edges: vec![Value::Int64(10)].into(),
        };
        let path2 = Value::Path {
            nodes: vec![Value::Int64(3), Value::Int64(4)].into(),
            edges: vec![Value::Int64(20)].into(),
        };

        let chunk = create_mixed_chunk(&[path1.clone(), path2, path1]);
        distinct.push(chunk, &mut sink).unwrap();
        distinct.finalize(&mut sink).unwrap();
        assert_eq!(distinct.unique_count(), 2);
    }

    /// Values of different types are distinct, but numbers compare by their
    /// value (openCypher equivalence): `1` and `1.0` are one value.
    #[test]
    fn values_of_different_types_are_distinct_but_equal_numbers_are_one() {
        let mut distinct = DistinctPushOperator::new();
        let mut sink = CollectorSink::new();

        let chunk = create_mixed_chunk(&[
            Value::Int64(1),
            Value::Float64(1.0),
            Value::from("1"),
            Value::Bool(true),
        ]);
        distinct.push(chunk, &mut sink).unwrap();
        distinct.finalize(&mut sink).unwrap();
        assert_eq!(distinct.unique_count(), 3);
    }

    /// `-0.0` is `0.0`, and every NaN is one value, whatever its bits.
    #[test]
    fn negative_zero_is_zero_and_nan_is_nan() {
        let mut distinct = DistinctPushOperator::new();
        let mut sink = CollectorSink::new();

        let chunk = create_mixed_chunk(&[
            Value::Float64(0.0),
            Value::Float64(-0.0),
            Value::Float64(f64::NAN),
            Value::Float64(f64::from_bits(f64::NAN.to_bits() | 1)),
        ]);
        distinct.push(chunk, &mut sink).unwrap();
        distinct.finalize(&mut sink).unwrap();
        assert_eq!(distinct.unique_count(), 2);
    }
}
