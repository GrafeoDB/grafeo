//! Distinct operator for removing duplicate rows.
//!
//! This module provides:
//! - `DistinctOperator`: Removes duplicate rows based on all or specified columns

use std::collections::HashSet;

use grafeo_common::types::{HashableValue, Value};

use super::{Operator, OperatorResult};
use crate::execution::DataChunk;
use crate::execution::chunk::DataChunkBuilder;

/// A row key for duplicate detection.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct RowKey(Vec<HashableValue>);

impl RowKey {
    /// Creates a row key from specified columns.
    fn from_row(chunk: &DataChunk, row: usize, columns: &[usize]) -> Self {
        let parts = columns
            .iter()
            .map(|&col_idx| {
                chunk
                    .column(col_idx)
                    .and_then(|col| col.get_value(row))
                    .unwrap_or(Value::Null)
                    .into()
            })
            .collect();
        RowKey(parts)
    }

    /// Creates a row key from all columns.
    fn from_all_columns(chunk: &DataChunk, row: usize) -> Self {
        let columns: Vec<usize> = (0..chunk.column_count()).collect();
        Self::from_row(chunk, row, &columns)
    }
}

/// Distinct operator.
///
/// Removes duplicate rows from the input. Can operate on all columns or a
/// subset. The rows it keeps keep their columns' types and values.
pub struct DistinctOperator {
    /// Child operator.
    child: Box<dyn Operator>,
    /// Columns to consider for uniqueness (None = all columns).
    distinct_columns: Option<Vec<usize>>,
    /// Set of seen row keys.
    seen: HashSet<RowKey>,
}

impl DistinctOperator {
    /// Creates a new distinct operator that considers all columns.
    pub fn new(child: Box<dyn Operator>) -> Self {
        Self {
            child,
            distinct_columns: None,
            seen: HashSet::new(),
        }
    }

    /// Decomposes this operator for push-based conversion.
    pub fn into_parts(self) -> (Box<dyn Operator>, Option<Vec<usize>>) {
        (self.child, self.distinct_columns)
    }

    /// Creates a distinct operator that considers only specified columns.
    pub fn on_columns(child: Box<dyn Operator>, columns: Vec<usize>) -> Self {
        Self {
            child,
            distinct_columns: Some(columns),
            seen: HashSet::new(),
        }
    }
}

impl Operator for DistinctOperator {
    fn next(&mut self) -> OperatorResult {
        loop {
            let Some(chunk) = self.child.next()? else {
                return Ok(None);
            };

            let mut builder = DataChunkBuilder::with_capacity(&chunk.column_types(), 2048);

            for row in chunk.selected_indices() {
                let key = match &self.distinct_columns {
                    Some(cols) => RowKey::from_row(&chunk, row, cols),
                    None => RowKey::from_all_columns(&chunk, row),
                };

                if self.seen.insert(key) {
                    // New unique row - copy it
                    for col_idx in 0..chunk.column_count() {
                        if let (Some(src_col), Some(dst_col)) =
                            (chunk.column(col_idx), builder.column_mut(col_idx))
                        {
                            if let Some(value) = src_col.get_value(row) {
                                dst_col.push_value(value);
                            } else {
                                dst_col.push_value(Value::Null);
                            }
                        }
                    }
                    builder.advance_row();

                    if builder.is_full() {
                        return Ok(Some(builder.finish()));
                    }
                }
            }

            if builder.row_count() > 0 {
                return Ok(Some(builder.finish()));
            }
            // If no unique rows in this chunk, continue to next
        }
    }

    fn reset(&mut self) {
        self.child.reset();
        self.seen.clear();
    }

    fn name(&self) -> &'static str {
        "Distinct"
    }

    fn into_any(self: Box<Self>) -> Box<dyn std::any::Any + Send> {
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::execution::chunk::DataChunkBuilder;
    use crate::execution::vector::ValueVector;
    use grafeo_common::types::{EdgeId, LogicalType, NodeId};

    struct MockOperator {
        chunks: Vec<DataChunk>,
        position: usize,
    }

    impl MockOperator {
        fn new(chunks: Vec<DataChunk>) -> Self {
            Self {
                chunks,
                position: 0,
            }
        }
    }

    impl Operator for MockOperator {
        fn next(&mut self) -> OperatorResult {
            if self.position < self.chunks.len() {
                // Keep the fixture available so reset can replay the same input.
                let chunk = self.chunks[self.position].clone();
                self.position += 1;
                Ok(Some(chunk))
            } else {
                Ok(None)
            }
        }

        fn reset(&mut self) {
            self.position = 0;
        }

        fn name(&self) -> &'static str {
            "Mock"
        }

        fn into_any(self: Box<Self>) -> Box<dyn std::any::Any + Send> {
            self
        }
    }

    fn create_chunk_with_duplicates() -> DataChunk {
        let mut builder = DataChunkBuilder::new(&[LogicalType::Int64, LogicalType::String]);

        let data = [
            (1i64, "a"),
            (2, "b"),
            (1, "a"), // Duplicate
            (3, "c"),
            (2, "b"), // Duplicate
            (1, "a"), // Duplicate
        ];

        for (num, text) in data {
            builder.column_mut(0).unwrap().push_int64(num);
            builder.column_mut(1).unwrap().push_string(text);
            builder.advance_row();
        }

        builder.finish()
    }

    fn value_rows_chunk(rows: &[Vec<Value>]) -> DataChunk {
        let columns = (0..rows[0].len())
            .map(|column| {
                ValueVector::from_values(
                    &rows
                        .iter()
                        .map(|row| row[column].clone())
                        .collect::<Vec<_>>(),
                )
            })
            .collect();
        DataChunk::new(columns)
    }

    fn collect_value_rows(operator: &mut DistinctOperator) -> Vec<Vec<Value>> {
        let mut rows = Vec::new();
        while let Some(chunk) = operator.next().unwrap() {
            for row in chunk.selected_indices() {
                rows.push(
                    chunk
                        .columns()
                        .iter()
                        .map(|column| column.get_value(row).unwrap())
                        .collect(),
                );
            }
        }
        rows
    }

    #[test]
    fn test_distinct_preserves_scalar_type_identity() {
        let expected = vec![vec![Value::Int64(0)], vec![Value::Float64(0.0)]];
        let input = MockOperator::new(vec![
            value_rows_chunk(&[expected[0].clone(), expected[1].clone()]),
            value_rows_chunk(&[expected[1].clone(), expected[0].clone()]),
        ]);
        let mut distinct = DistinctOperator::new(Box::new(input));

        assert_eq!(collect_value_rows(&mut distinct), expected);
    }

    #[test]
    fn test_distinct_preserves_nested_value_identity() {
        let nested = |value| Value::List(vec![Value::List(vec![value].into())].into());
        let expected = vec![
            vec![nested(Value::Int64(0))],
            vec![nested(Value::Float64(0.0))],
            vec![nested(Value::Bytes(vec![7, 1, 2].into()))],
            vec![nested(Value::Bytes(vec![7, 3, 4].into()))],
        ];
        let input = MockOperator::new(vec![
            value_rows_chunk(&expected),
            value_rows_chunk(&[expected[3].clone(), expected[0].clone()]),
        ]);
        let mut distinct = DistinctOperator::new(Box::new(input));

        assert_eq!(collect_value_rows(&mut distinct), expected);
    }

    #[test]
    fn test_distinct_selected_columns_preserve_full_rows() {
        let expected = vec![
            vec![Value::Int64(0), Value::from("integer")],
            vec![Value::Float64(0.0), Value::from("float")],
            vec![Value::Null, Value::from("null")],
        ];
        let input = MockOperator::new(vec![
            value_rows_chunk(&expected),
            value_rows_chunk(&[
                vec![Value::Int64(0), Value::from("ignored integer payload")],
                vec![Value::Float64(0.0), Value::from("ignored float payload")],
                vec![Value::Null, Value::from("ignored null payload")],
            ]),
        ]);
        let mut distinct = DistinctOperator::on_columns(Box::new(input), vec![0]);

        assert_eq!(collect_value_rows(&mut distinct), expected);
    }

    #[test]
    fn test_distinct_reset_replays_typed_rows() {
        let expected = vec![
            vec![Value::Int64(0)],
            vec![Value::Float64(0.0)],
            vec![Value::Null],
        ];
        let input = MockOperator::new(vec![value_rows_chunk(&expected)]);
        let mut distinct = DistinctOperator::new(Box::new(input));

        assert_eq!(collect_value_rows(&mut distinct), expected);
        distinct.reset();
        assert_eq!(collect_value_rows(&mut distinct), expected);
    }

    #[test]
    fn test_distinct_preserves_raw_float_bits() {
        let expected_bits = [
            0,
            (-0.0_f64).to_bits(),
            0x7ff8_0000_0000_0001,
            0x7ff8_0000_0000_0002,
        ];
        let expected: Vec<_> = expected_bits
            .iter()
            .map(|&bits| vec![Value::Float64(f64::from_bits(bits))])
            .collect();
        let input = MockOperator::new(vec![
            value_rows_chunk(&expected),
            value_rows_chunk(&expected),
        ]);
        let mut distinct = DistinctOperator::new(Box::new(input));
        let actual: Vec<_> = collect_value_rows(&mut distinct)
            .iter()
            .map(|row| match row.as_slice() {
                [Value::Float64(value)] => value.to_bits(),
                other => panic!("expected one unchanged float, got {other:?}"),
            })
            .collect();

        assert_eq!(actual, expected_bits);
    }

    #[test]
    fn test_distinct_preserves_node_edge_columns() {
        let mut builder = DataChunkBuilder::new(&[LogicalType::Node, LogicalType::Edge]);
        for id in [1, 1, 2] {
            builder.column_mut(0).unwrap().push_node_id(NodeId::new(id));
            builder.column_mut(1).unwrap().push_edge_id(EdgeId::new(id));
            builder.advance_row();
        }
        let input = MockOperator::new(vec![builder.finish()]);
        let mut distinct = DistinctOperator::new(Box::new(input));
        let chunk = distinct.next().unwrap().unwrap();

        assert_eq!(chunk.column_types(), [LogicalType::Node, LogicalType::Edge]);
        assert_eq!(chunk.row_count(), 2);
        for (row, id) in [1, 2].into_iter().enumerate() {
            assert_eq!(
                chunk.column(0).unwrap().get_node_id(row),
                Some(NodeId::new(id))
            );
            assert_eq!(
                chunk.column(1).unwrap().get_edge_id(row),
                Some(EdgeId::new(id))
            );
        }
        assert!(distinct.next().unwrap().is_none());
    }

    #[test]
    fn test_distinct_all_columns() {
        let mock = MockOperator::new(vec![create_chunk_with_duplicates()]);

        let mut distinct = DistinctOperator::new(Box::new(mock));

        let mut results = Vec::new();
        while let Some(chunk) = distinct.next().unwrap() {
            for row in chunk.selected_indices() {
                let num = chunk.column(0).unwrap().get_int64(row).unwrap();
                let text = chunk
                    .column(1)
                    .unwrap()
                    .get_string(row)
                    .unwrap()
                    .to_string();
                results.push((num, text));
            }
        }

        // Should have 3 unique rows
        assert_eq!(results.len(), 3);

        // Sort for consistent comparison
        results.sort();
        assert_eq!(
            results,
            vec![
                (1, "a".to_string()),
                (2, "b".to_string()),
                (3, "c".to_string()),
            ]
        );
    }

    #[test]
    fn test_distinct_single_column() {
        let mock = MockOperator::new(vec![create_chunk_with_duplicates()]);

        let mut distinct = DistinctOperator::on_columns(Box::new(mock), vec![0]);

        let mut results = Vec::new();
        while let Some(chunk) = distinct.next().unwrap() {
            for row in chunk.selected_indices() {
                let num = chunk.column(0).unwrap().get_int64(row).unwrap();
                results.push(num);
            }
        }

        // Should have 3 unique values in column 0
        results.sort_unstable();
        assert_eq!(results, vec![1, 2, 3]);
    }

    #[test]
    fn test_distinct_across_chunks() {
        // Create two chunks with overlapping values
        let mut builder1 = DataChunkBuilder::new(&[LogicalType::Int64]);
        for i in [1, 2, 3] {
            builder1.column_mut(0).unwrap().push_int64(i);
            builder1.advance_row();
        }

        let mut builder2 = DataChunkBuilder::new(&[LogicalType::Int64]);
        for i in [2, 3, 4] {
            builder2.column_mut(0).unwrap().push_int64(i);
            builder2.advance_row();
        }

        let mock = MockOperator::new(vec![builder1.finish(), builder2.finish()]);

        let mut distinct = DistinctOperator::new(Box::new(mock));

        let mut results = Vec::new();
        while let Some(chunk) = distinct.next().unwrap() {
            for row in chunk.selected_indices() {
                let num = chunk.column(0).unwrap().get_int64(row).unwrap();
                results.push(num);
            }
        }

        // Should have 4 unique values: 1, 2, 3, 4
        results.sort_unstable();
        assert_eq!(results, vec![1, 2, 3, 4]);
    }

    #[test]
    fn test_distinct_into_any() {
        let mock = MockOperator::new(vec![]);
        let op = DistinctOperator::new(Box::new(mock));
        let any = Box::new(op).into_any();
        assert!(any.downcast::<DistinctOperator>().is_ok());
    }

    #[test]
    fn test_distinct_into_parts() {
        let mock = MockOperator::new(vec![]);
        let op = DistinctOperator::on_columns(Box::new(mock), vec![0, 2]);
        let (mut child, distinct_columns) = op.into_parts();
        assert_eq!(distinct_columns, Some(vec![0, 2]));
        assert!(child.next().unwrap().is_none());
    }

    #[test]
    fn test_distinct_into_parts_all_columns() {
        let mock = MockOperator::new(vec![]);
        let op = DistinctOperator::new(Box::new(mock));
        let (_child, distinct_columns) = op.into_parts();
        assert!(distinct_columns.is_none());
    }
}
