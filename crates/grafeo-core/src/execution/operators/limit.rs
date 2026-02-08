//! Limit and Skip operators for result pagination.
//!
//! This module provides:
//! - `LimitOperator`: Limits the number of output rows
//! - `SkipOperator`: Skips a number of input rows
//! - `LimitSkipOperator`: Combined LIMIT and OFFSET/SKIP

use grafeo_common::types::{LogicalType, Value};

use super::{Operator, OperatorResult};
use crate::execution::chunk::DataChunkBuilder;

/// Limit operator.
///
/// Returns at most `limit` rows from the input.
pub struct LimitOperator {
    /// Child operator.
    child: Box<dyn Operator>,
    /// Maximum number of rows to return.
    limit: usize,
    /// Output schema.
    output_schema: Vec<LogicalType>,
    /// Number of rows returned so far.
    returned: usize,
}

impl LimitOperator {
    /// Creates a new limit operator.
    pub fn new(child: Box<dyn Operator>, limit: usize, output_schema: Vec<LogicalType>) -> Self {
        Self {
            child,
            limit,
            output_schema,
            returned: 0,
        }
    }
}

impl Operator for LimitOperator {
    fn next(&mut self) -> OperatorResult {
        if self.returned >= self.limit {
            return Ok(None);
        }

        let remaining = self.limit - self.returned;

        loop {
            let Some(chunk) = self.child.next()? else {
                return Ok(None);
            };

            let row_count = chunk.row_count();
            if row_count == 0 {
                continue;
            }

            if row_count <= remaining {
                // Return entire chunk
                self.returned += row_count;
                return Ok(Some(chunk));
            }

            // Return partial chunk
            let mut builder = DataChunkBuilder::with_capacity(&self.output_schema, remaining);

            let mut count = 0;
            for row in chunk.selected_indices() {
                if count >= remaining {
                    break;
                }

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
                count += 1;
            }

            self.returned += count;
            return Ok(Some(builder.finish()));
        }
    }

    fn reset(&mut self) {
        self.child.reset();
        self.returned = 0;
    }

    fn name(&self) -> &'static str {
        "Limit"
    }
}

/// Skip operator.
///
/// Skips the first `skip` rows from the input.
pub struct SkipOperator {
    /// Child operator.
    child: Box<dyn Operator>,
    /// Number of rows to skip.
    skip: usize,
    /// Output schema.
    output_schema: Vec<LogicalType>,
    /// Number of rows skipped so far.
    skipped: usize,
}

impl SkipOperator {
    /// Creates a new skip operator.
    pub fn new(child: Box<dyn Operator>, skip: usize, output_schema: Vec<LogicalType>) -> Self {
        Self {
            child,
            skip,
            output_schema,
            skipped: 0,
        }
    }
}

impl Operator for SkipOperator {
    fn next(&mut self) -> OperatorResult {
        // Skip rows until we've skipped enough
        while self.skipped < self.skip {
            let Some(chunk) = self.child.next()? else {
                return Ok(None);
            };

            let row_count = chunk.row_count();
            let to_skip = (self.skip - self.skipped).min(row_count);

            if to_skip >= row_count {
                // Skip entire chunk
                self.skipped += row_count;
                continue;
            }

            // Skip partial chunk
            self.skipped = self.skip;

            let mut builder =
                DataChunkBuilder::with_capacity(&self.output_schema, row_count - to_skip);

            let rows: Vec<usize> = chunk.selected_indices().collect();
            for &row in rows.iter().skip(to_skip) {
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
            }

            return Ok(Some(builder.finish()));
        }

        // After skipping, just pass through
        self.child.next()
    }

    fn reset(&mut self) {
        self.child.reset();
        self.skipped = 0;
    }

    fn name(&self) -> &'static str {
        "Skip"
    }
}

/// Combined Limit and Skip operator.
///
/// Equivalent to OFFSET skip LIMIT limit.
pub struct LimitSkipOperator {
    /// Child operator.
    child: Box<dyn Operator>,
    /// Number of rows to skip.
    skip: usize,
    /// Maximum number of rows to return.
    limit: usize,
    /// Output schema.
    output_schema: Vec<LogicalType>,
    /// Number of rows skipped so far.
    skipped: usize,
    /// Number of rows returned so far.
    returned: usize,
}

impl LimitSkipOperator {
    /// Creates a new limit/skip operator.
    pub fn new(
        child: Box<dyn Operator>,
        skip: usize,
        limit: usize,
        output_schema: Vec<LogicalType>,
    ) -> Self {
        Self {
            child,
            skip,
            limit,
            output_schema,
            skipped: 0,
            returned: 0,
        }
    }
}

impl Operator for LimitSkipOperator {
    fn next(&mut self) -> OperatorResult {
        // Check if we've returned enough
        if self.returned >= self.limit {
            return Ok(None);
        }

        loop {
            let Some(chunk) = self.child.next()? else {
                return Ok(None);
            };

            let row_count = chunk.row_count();
            if row_count == 0 {
                continue;
            }

            let rows: Vec<usize> = chunk.selected_indices().collect();
            let mut start_idx = 0;

            // Skip rows if needed
            if self.skipped < self.skip {
                let to_skip = (self.skip - self.skipped).min(row_count);
                if to_skip >= row_count {
                    self.skipped += row_count;
                    continue;
                }
                self.skipped = self.skip;
                start_idx = to_skip;
            }

            // Calculate how many rows to return
            let remaining_in_chunk = row_count - start_idx;
            let remaining_to_return = self.limit - self.returned;
            let to_return = remaining_in_chunk.min(remaining_to_return);

            if to_return == 0 {
                return Ok(None);
            }

            let mut builder = DataChunkBuilder::with_capacity(&self.output_schema, to_return);

            for &row in rows.iter().skip(start_idx).take(to_return) {
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
            }

            self.returned += to_return;
            return Ok(Some(builder.finish()));
        }
    }

    fn reset(&mut self) {
        self.child.reset();
        self.skipped = 0;
        self.returned = 0;
    }

    fn name(&self) -> &'static str {
        "LimitSkip"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::execution::DataChunk;
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
    }

    fn create_numbered_chunk(values: &[i64]) -> DataChunk {
        let mut builder = DataChunkBuilder::new(&[LogicalType::Int64]);
        for &v in values {
            builder.column_mut(0).unwrap().push_int64(v);
            builder.advance_row();
        }
        builder.finish()
    }

    #[test]
    fn test_limit() {
        let mock = MockOperator::new(vec![create_numbered_chunk(&[1, 2, 3, 4, 5])]);

        let mut limit = LimitOperator::new(Box::new(mock), 3, vec![LogicalType::Int64]);

        let mut results = Vec::new();
        while let Some(chunk) = limit.next().unwrap() {
            for row in chunk.selected_indices() {
                let val = chunk.column(0).unwrap().get_int64(row).unwrap();
                results.push(val);
            }
        }

        assert_eq!(results, vec![1, 2, 3]);
    }

    #[test]
    fn test_limit_larger_than_input() {
        let mock = MockOperator::new(vec![create_numbered_chunk(&[1, 2, 3])]);

        let mut limit = LimitOperator::new(Box::new(mock), 10, vec![LogicalType::Int64]);

        let mut results = Vec::new();
        while let Some(chunk) = limit.next().unwrap() {
            for row in chunk.selected_indices() {
                let val = chunk.column(0).unwrap().get_int64(row).unwrap();
                results.push(val);
            }
        }

        assert_eq!(results, vec![1, 2, 3]);
    }

    #[test]
    fn test_skip() {
        let mock = MockOperator::new(vec![create_numbered_chunk(&[1, 2, 3, 4, 5])]);

        let mut skip = SkipOperator::new(Box::new(mock), 2, vec![LogicalType::Int64]);

        let mut results = Vec::new();
        while let Some(chunk) = skip.next().unwrap() {
            for row in chunk.selected_indices() {
                let val = chunk.column(0).unwrap().get_int64(row).unwrap();
                results.push(val);
            }
        }

        assert_eq!(results, vec![3, 4, 5]);
    }

    #[test]
    fn test_skip_all() {
        let mock = MockOperator::new(vec![create_numbered_chunk(&[1, 2, 3])]);

        let mut skip = SkipOperator::new(Box::new(mock), 5, vec![LogicalType::Int64]);

        let result = skip.next().unwrap();
        assert!(result.is_none());
    }

    #[test]
    fn test_limit_skip_combined() {
        let mock = MockOperator::new(vec![create_numbered_chunk(&[
            1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
        ])]);

        let mut op = LimitSkipOperator::new(
            Box::new(mock),
            3, // Skip first 3
            4, // Take next 4
            vec![LogicalType::Int64],
        );

        let mut results = Vec::new();
        while let Some(chunk) = op.next().unwrap() {
            for row in chunk.selected_indices() {
                let val = chunk.column(0).unwrap().get_int64(row).unwrap();
                results.push(val);
            }
        }

        assert_eq!(results, vec![4, 5, 6, 7]);
    }

    #[test]
    fn test_limit_across_chunks() {
        let mock = MockOperator::new(vec![
            create_numbered_chunk(&[1, 2]),
            create_numbered_chunk(&[3, 4]),
            create_numbered_chunk(&[5, 6]),
        ]);

        let mut limit = LimitOperator::new(Box::new(mock), 5, vec![LogicalType::Int64]);

        let mut results = Vec::new();
        while let Some(chunk) = limit.next().unwrap() {
            for row in chunk.selected_indices() {
                let val = chunk.column(0).unwrap().get_int64(row).unwrap();
                results.push(val);
            }
        }

        assert_eq!(results, vec![1, 2, 3, 4, 5]);
    }

    #[test]
    fn test_skip_across_chunks() {
        let mock = MockOperator::new(vec![
            create_numbered_chunk(&[1, 2]),
            create_numbered_chunk(&[3, 4]),
            create_numbered_chunk(&[5, 6]),
        ]);

        let mut skip = SkipOperator::new(Box::new(mock), 3, vec![LogicalType::Int64]);

        let mut results = Vec::new();
        while let Some(chunk) = skip.next().unwrap() {
            for row in chunk.selected_indices() {
                let val = chunk.column(0).unwrap().get_int64(row).unwrap();
                results.push(val);
            }
        }

        assert_eq!(results, vec![4, 5, 6]);
    }
}
