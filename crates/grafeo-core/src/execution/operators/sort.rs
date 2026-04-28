//! Sort operator for ordering results.
//!
//! This module provides:
//! - `SortOperator`: Orders results by one or more columns

use std::cmp::Ordering;

use grafeo_common::types::{LogicalType, Value};

use super::value_utils::compare_values_with_nulls;
use super::{Operator, OperatorError, OperatorResult};
use crate::execution::DataChunk;
use crate::execution::chunk::DataChunkBuilder;

/// Sort direction.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum SortDirection {
    /// Ascending order (smallest first).
    Ascending,
    /// Descending order (largest first).
    Descending,
}

/// Null ordering.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum NullOrder {
    /// Nulls come first.
    NullsFirst,
    /// Nulls come last.
    NullsLast,
}

/// A sort key specification.
#[derive(Debug, Clone)]
pub struct SortKey {
    /// Column index to sort by.
    pub column: usize,
    /// Sort direction.
    pub direction: SortDirection,
    /// Null ordering.
    pub null_order: NullOrder,
}

impl SortKey {
    /// Creates a new sort key with ascending order.
    pub fn ascending(column: usize) -> Self {
        Self {
            column,
            direction: SortDirection::Ascending,
            null_order: NullOrder::NullsLast,
        }
    }

    /// Creates a new sort key with descending order.
    pub fn descending(column: usize) -> Self {
        Self {
            column,
            direction: SortDirection::Descending,
            null_order: NullOrder::NullsLast,
        }
    }

    /// Sets the null ordering.
    pub fn with_null_order(mut self, null_order: NullOrder) -> Self {
        self.null_order = null_order;
        self
    }
}

/// A row reference for sorting.
#[derive(Debug, Clone)]
struct SortRow {
    /// Index of the chunk this row belongs to.
    chunk_index: usize,
    /// Row index within the chunk.
    row_index: usize,
}

/// Sort operator.
///
/// Materializes all input and sorts by the specified keys.
pub struct SortOperator {
    /// Child operator.
    child: Box<dyn Operator>,
    /// Sort keys.
    sort_keys: Vec<SortKey>,
    /// Output schema.
    output_schema: Vec<LogicalType>,
    /// Materialized chunks.
    chunks: Vec<DataChunk>,
    /// Sorted row references.
    sorted_rows: Vec<SortRow>,
    /// Whether sorting is complete.
    sort_complete: bool,
    /// Current position in output.
    output_position: usize,
}

impl SortOperator {
    /// Creates a new sort operator.
    pub fn new(
        child: Box<dyn Operator>,
        sort_keys: Vec<SortKey>,
        output_schema: Vec<LogicalType>,
    ) -> Self {
        Self {
            child,
            sort_keys,
            output_schema,
            chunks: Vec::new(),
            sorted_rows: Vec::new(),
            sort_complete: false,
            output_position: 0,
        }
    }

    /// Decomposes this operator into its child and sort keys for push-based conversion.
    pub fn into_parts(self) -> (Box<dyn Operator>, Vec<SortKey>) {
        (self.child, self.sort_keys)
    }

    /// Materializes and sorts the input.
    fn sort(&mut self) -> Result<(), OperatorError> {
        // Materialize all input
        while let Some(chunk) = self.child.next()? {
            let chunk_idx = self.chunks.len();
            for row_idx in chunk.selected_indices() {
                self.sorted_rows.push(SortRow {
                    chunk_index: chunk_idx,
                    row_index: row_idx,
                });
            }
            self.chunks.push(chunk);
        }

        // Sort the row references
        let chunks = &self.chunks;
        let sort_keys = &self.sort_keys;

        self.sorted_rows.sort_by(|a, b| {
            for key in sort_keys {
                let chunk_a = &chunks[a.chunk_index];
                let chunk_b = &chunks[b.chunk_index];

                let val_a = chunk_a
                    .column(key.column)
                    .and_then(|c| c.get_value(a.row_index));
                let val_b = chunk_b
                    .column(key.column)
                    .and_then(|c| c.get_value(b.row_index));

                let cmp = compare_values_with_nulls(&val_a, &val_b, key.null_order);

                let cmp = match key.direction {
                    SortDirection::Ascending => cmp,
                    SortDirection::Descending => cmp.reverse(),
                };

                if cmp != Ordering::Equal {
                    return cmp;
                }
            }
            Ordering::Equal
        });

        self.sort_complete = true;
        Ok(())
    }
}


impl Operator for SortOperator {
    fn next(&mut self) -> OperatorResult {
        if !self.sort_complete {
            self.sort()?;
        }

        if self.output_position >= self.sorted_rows.len() {
            return Ok(None);
        }

        let mut builder = DataChunkBuilder::with_capacity(&self.output_schema, 2048);

        while self.output_position < self.sorted_rows.len() && !builder.is_full() {
            let row_ref = &self.sorted_rows[self.output_position];
            let source_chunk = &self.chunks[row_ref.chunk_index];

            // Copy all columns
            for col_idx in 0..source_chunk.column_count() {
                if let (Some(src_col), Some(dst_col)) =
                    (source_chunk.column(col_idx), builder.column_mut(col_idx))
                {
                    if let Some(value) = src_col.get_value(row_ref.row_index) {
                        dst_col.push_value(value);
                    } else {
                        dst_col.push_value(Value::Null);
                    }
                }
            }

            builder.advance_row();
            self.output_position += 1;
        }

        if builder.row_count() > 0 {
            Ok(Some(builder.finish()))
        } else {
            Ok(None)
        }
    }

    fn reset(&mut self) {
        self.child.reset();
        self.chunks.clear();
        self.sorted_rows.clear();
        self.sort_complete = false;
        self.output_position = 0;
    }

    fn name(&self) -> &'static str {
        "Sort"
    }

    fn into_any(self: Box<Self>) -> Box<dyn std::any::Any + Send> {
        self
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
                let chunk = std::mem::replace(&mut self.chunks[self.position], DataChunk::empty());
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

    fn create_unsorted_chunk() -> DataChunk {
        let mut builder = DataChunkBuilder::new(&[LogicalType::Int64, LogicalType::String]);

        let data = [(3i64, "cherry"), (1, "apple"), (4, "date"), (2, "banana")];

        for (num, text) in data {
            builder.column_mut(0).unwrap().push_int64(num);
            builder.column_mut(1).unwrap().push_string(text);
            builder.advance_row();
        }

        builder.finish()
    }

    #[test]
    fn test_sort_ascending() {
        let mock = MockOperator::new(vec![create_unsorted_chunk()]);

        let mut sort = SortOperator::new(
            Box::new(mock),
            vec![SortKey::ascending(0)],
            vec![LogicalType::Int64, LogicalType::String],
        );

        let mut results = Vec::new();
        while let Some(chunk) = sort.next().unwrap() {
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

        assert_eq!(
            results,
            vec![
                (1, "apple".to_string()),
                (2, "banana".to_string()),
                (3, "cherry".to_string()),
                (4, "date".to_string()),
            ]
        );
    }

    #[test]
    fn test_sort_descending() {
        let mock = MockOperator::new(vec![create_unsorted_chunk()]);

        let mut sort = SortOperator::new(
            Box::new(mock),
            vec![SortKey::descending(0)],
            vec![LogicalType::Int64, LogicalType::String],
        );

        let mut results = Vec::new();
        while let Some(chunk) = sort.next().unwrap() {
            for row in chunk.selected_indices() {
                let num = chunk.column(0).unwrap().get_int64(row).unwrap();
                results.push(num);
            }
        }

        assert_eq!(results, vec![4, 3, 2, 1]);
    }

    #[test]
    fn test_sort_by_string() {
        let mock = MockOperator::new(vec![create_unsorted_chunk()]);

        let mut sort = SortOperator::new(
            Box::new(mock),
            vec![SortKey::ascending(1)], // Sort by string column
            vec![LogicalType::Int64, LogicalType::String],
        );

        let mut results = Vec::new();
        while let Some(chunk) = sort.next().unwrap() {
            for row in chunk.selected_indices() {
                let text = chunk
                    .column(1)
                    .unwrap()
                    .get_string(row)
                    .unwrap()
                    .to_string();
                results.push(text);
            }
        }

        assert_eq!(
            results,
            vec![
                "apple".to_string(),
                "banana".to_string(),
                "cherry".to_string(),
                "date".to_string(),
            ]
        );
    }

    #[test]
    fn test_sort_empty_input() {
        let mock = MockOperator::new(vec![]);

        let mut sort = SortOperator::new(
            Box::new(mock),
            vec![SortKey::ascending(0)],
            vec![LogicalType::Int64],
        );

        assert!(sort.next().unwrap().is_none());
    }

    #[test]
    fn test_sort_already_sorted() {
        let mut builder = DataChunkBuilder::new(&[LogicalType::Int64]);
        for v in [1i64, 2, 3, 4] {
            builder.column_mut(0).unwrap().push_int64(v);
            builder.advance_row();
        }
        let chunk = builder.finish();

        let mock = MockOperator::new(vec![chunk]);
        let mut sort = SortOperator::new(
            Box::new(mock),
            vec![SortKey::ascending(0)],
            vec![LogicalType::Int64],
        );

        let mut results = Vec::new();
        while let Some(chunk) = sort.next().unwrap() {
            for row in chunk.selected_indices() {
                results.push(chunk.column(0).unwrap().get_int64(row).unwrap());
            }
        }

        assert_eq!(results, vec![1, 2, 3, 4]);
    }

    #[test]
    fn test_sort_duplicate_values() {
        let mut builder = DataChunkBuilder::new(&[LogicalType::Int64]);
        for v in [3i64, 1, 3, 2, 1] {
            builder.column_mut(0).unwrap().push_int64(v);
            builder.advance_row();
        }
        let chunk = builder.finish();

        let mock = MockOperator::new(vec![chunk]);
        let mut sort = SortOperator::new(
            Box::new(mock),
            vec![SortKey::ascending(0)],
            vec![LogicalType::Int64],
        );

        let mut results = Vec::new();
        while let Some(chunk) = sort.next().unwrap() {
            for row in chunk.selected_indices() {
                results.push(chunk.column(0).unwrap().get_int64(row).unwrap());
            }
        }

        assert_eq!(results, vec![1, 1, 2, 3, 3]);
    }

    #[test]
    fn test_sort_multi_key() {
        let mut builder = DataChunkBuilder::new(&[LogicalType::String, LogicalType::Int64]);
        let data = [("b", 2i64), ("a", 3), ("b", 1), ("a", 1)];
        for (s, n) in data {
            builder.column_mut(0).unwrap().push_string(s);
            builder.column_mut(1).unwrap().push_int64(n);
            builder.advance_row();
        }
        let chunk = builder.finish();

        let mock = MockOperator::new(vec![chunk]);
        let mut sort = SortOperator::new(
            Box::new(mock),
            vec![SortKey::ascending(0), SortKey::ascending(1)],
            vec![LogicalType::String, LogicalType::Int64],
        );

        let mut results = Vec::new();
        while let Some(chunk) = sort.next().unwrap() {
            for row in chunk.selected_indices() {
                let s = chunk
                    .column(0)
                    .unwrap()
                    .get_string(row)
                    .unwrap()
                    .to_string();
                let n = chunk.column(1).unwrap().get_int64(row).unwrap();
                results.push((s, n));
            }
        }

        assert_eq!(
            results,
            vec![
                ("a".to_string(), 1),
                ("a".to_string(), 3),
                ("b".to_string(), 1),
                ("b".to_string(), 2),
            ]
        );
    }

    #[test]
    fn test_sort_multiple_chunks() {
        // Two separate chunks that get merged during sort
        let mut b1 = DataChunkBuilder::new(&[LogicalType::Int64]);
        b1.column_mut(0).unwrap().push_int64(5);
        b1.advance_row();
        b1.column_mut(0).unwrap().push_int64(1);
        b1.advance_row();
        let chunk1 = b1.finish();

        let mut b2 = DataChunkBuilder::new(&[LogicalType::Int64]);
        b2.column_mut(0).unwrap().push_int64(3);
        b2.advance_row();
        b2.column_mut(0).unwrap().push_int64(2);
        b2.advance_row();
        let chunk2 = b2.finish();

        let mock = MockOperator::new(vec![chunk1, chunk2]);
        let mut sort = SortOperator::new(
            Box::new(mock),
            vec![SortKey::ascending(0)],
            vec![LogicalType::Int64],
        );

        let mut results = Vec::new();
        while let Some(chunk) = sort.next().unwrap() {
            for row in chunk.selected_indices() {
                results.push(chunk.column(0).unwrap().get_int64(row).unwrap());
            }
        }

        assert_eq!(results, vec![1, 2, 3, 5]);
    }

    #[test]
    fn test_sort_reverse_sorted() {
        let mut builder = DataChunkBuilder::new(&[LogicalType::Int64]);
        for v in [4i64, 3, 2, 1] {
            builder.column_mut(0).unwrap().push_int64(v);
            builder.advance_row();
        }
        let chunk = builder.finish();

        let mock = MockOperator::new(vec![chunk]);
        let mut sort = SortOperator::new(
            Box::new(mock),
            vec![SortKey::ascending(0)],
            vec![LogicalType::Int64],
        );

        let mut results = Vec::new();
        while let Some(chunk) = sort.next().unwrap() {
            for row in chunk.selected_indices() {
                results.push(chunk.column(0).unwrap().get_int64(row).unwrap());
            }
        }

        assert_eq!(results, vec![1, 2, 3, 4]);
    }

    #[test]
    fn test_sort_single_row() {
        let mut builder = DataChunkBuilder::new(&[LogicalType::Int64]);
        builder.column_mut(0).unwrap().push_int64(42);
        builder.advance_row();
        let chunk = builder.finish();

        let mock = MockOperator::new(vec![chunk]);
        let mut sort = SortOperator::new(
            Box::new(mock),
            vec![SortKey::ascending(0)],
            vec![LogicalType::Int64],
        );

        let mut count = 0;
        while let Some(chunk) = sort.next().unwrap() {
            for row in chunk.selected_indices() {
                assert_eq!(chunk.column(0).unwrap().get_int64(row), Some(42));
                count += 1;
            }
        }
        assert_eq!(count, 1);
    }

    #[test]
    fn test_sort_name() {
        let mock = MockOperator::new(vec![]);
        let sort = SortOperator::new(
            Box::new(mock),
            vec![SortKey::ascending(0)],
            vec![LogicalType::Int64],
        );
        assert_eq!(sort.name(), "Sort");
    }

    #[test]
    fn test_sort_into_any() {
        let mock = MockOperator::new(vec![]);
        let op = SortOperator::new(
            Box::new(mock),
            vec![SortKey::ascending(0)],
            vec![LogicalType::Int64],
        );
        let any = Box::new(op).into_any();
        assert!(any.downcast::<SortOperator>().is_ok());
    }

    #[test]
    fn test_sort_into_parts() {
        let mock = MockOperator::new(vec![]);
        let op = SortOperator::new(
            Box::new(mock),
            vec![SortKey::ascending(0), SortKey::descending(1)],
            vec![LogicalType::Int64, LogicalType::String],
        );
        let (mut child, sort_keys) = op.into_parts();
        assert_eq!(sort_keys.len(), 2);
        assert_eq!(sort_keys[0].column, 0);
        assert_eq!(sort_keys[1].column, 1);
        assert!(child.next().unwrap().is_none());
    }
}
