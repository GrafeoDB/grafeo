//! Distinct operator for removing duplicate rows.
//!
//! This module provides:
//! - `DistinctOperator`: Removes duplicate rows based on all or specified columns

use std::collections::HashSet;

use graphos_common::types::{LogicalType, Value};

use super::{Operator, OperatorResult};
use crate::execution::chunk::DataChunkBuilder;
use crate::execution::DataChunk;

/// A row key for duplicate detection.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct RowKey(Vec<KeyPart>);

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
enum KeyPart {
    Null,
    Bool(bool),
    Int64(i64),
    String(String),
}

impl RowKey {
    /// Creates a row key from specified columns.
    fn from_row(chunk: &DataChunk, row: usize, columns: &[usize]) -> Self {
        let parts: Vec<KeyPart> = columns
            .iter()
            .map(|&col_idx| {
                chunk
                    .column(col_idx)
                    .and_then(|col| col.get_value(row))
                    .map(|v| match v {
                        Value::Null => KeyPart::Null,
                        Value::Bool(b) => KeyPart::Bool(b),
                        Value::Int64(i) => KeyPart::Int64(i),
                        Value::Float64(f) => KeyPart::Int64(f.to_bits() as i64),
                        Value::String(s) => KeyPart::String(s.to_string()),
                        _ => KeyPart::String(format!("{v:?}")),
                    })
                    .unwrap_or(KeyPart::Null)
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
/// Removes duplicate rows from the input. Can operate on all columns or a subset.
pub struct DistinctOperator {
    /// Child operator.
    child: Box<dyn Operator>,
    /// Columns to consider for uniqueness (None = all columns).
    distinct_columns: Option<Vec<usize>>,
    /// Output schema.
    output_schema: Vec<LogicalType>,
    /// Set of seen row keys.
    seen: HashSet<RowKey>,
}

impl DistinctOperator {
    /// Creates a new distinct operator that considers all columns.
    pub fn new(child: Box<dyn Operator>, output_schema: Vec<LogicalType>) -> Self {
        Self {
            child,
            distinct_columns: None,
            output_schema,
            seen: HashSet::new(),
        }
    }

    /// Creates a distinct operator that considers only specified columns.
    pub fn on_columns(
        child: Box<dyn Operator>,
        columns: Vec<usize>,
        output_schema: Vec<LogicalType>,
    ) -> Self {
        Self {
            child,
            distinct_columns: Some(columns),
            output_schema,
            seen: HashSet::new(),
        }
    }
}

impl Operator for DistinctOperator {
    fn next(&mut self) -> OperatorResult {
        loop {
            let chunk = match self.child.next()? {
                Some(c) => c,
                None => return Ok(None),
            };

            let mut builder = DataChunkBuilder::with_capacity(&self.output_schema, 2048);

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
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::execution::chunk::DataChunkBuilder;

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
                let chunk =
                    std::mem::replace(&mut self.chunks[self.position], DataChunk::empty());
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

    #[test]
    fn test_distinct_all_columns() {
        let mock = MockOperator::new(vec![create_chunk_with_duplicates()]);

        let mut distinct = DistinctOperator::new(
            Box::new(mock),
            vec![LogicalType::Int64, LogicalType::String],
        );

        let mut results = Vec::new();
        while let Some(chunk) = distinct.next().unwrap() {
            for row in chunk.selected_indices() {
                let num = chunk.column(0).unwrap().get_int64(row).unwrap();
                let text = chunk.column(1).unwrap().get_string(row).unwrap().to_string();
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

        let mut distinct = DistinctOperator::on_columns(
            Box::new(mock),
            vec![0], // Only consider first column
            vec![LogicalType::Int64, LogicalType::String],
        );

        let mut results = Vec::new();
        while let Some(chunk) = distinct.next().unwrap() {
            for row in chunk.selected_indices() {
                let num = chunk.column(0).unwrap().get_int64(row).unwrap();
                results.push(num);
            }
        }

        // Should have 3 unique values in column 0
        results.sort();
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

        let mut distinct = DistinctOperator::new(Box::new(mock), vec![LogicalType::Int64]);

        let mut results = Vec::new();
        while let Some(chunk) = distinct.next().unwrap() {
            for row in chunk.selected_indices() {
                let num = chunk.column(0).unwrap().get_int64(row).unwrap();
                results.push(num);
            }
        }

        // Should have 4 unique values: 1, 2, 3, 4
        results.sort();
        assert_eq!(results, vec![1, 2, 3, 4]);
    }
}
