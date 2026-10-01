//! Limit and Skip operators for result pagination.
//!
//! This module provides:
//! - `LimitOperator`: Limits the number of output rows
//! - `SkipOperator`: Skips a number of input rows
//! - `LimitSkipOperator`: Combined LIMIT and OFFSET/SKIP

use super::{Operator, OperatorResult};

/// Limit operator.
///
/// Returns at most `limit` rows from the input. A chunk it cuts short keeps
/// its columns and values as they are.
pub struct LimitOperator {
    /// Child operator.
    child: Box<dyn Operator>,
    /// Maximum number of rows to return.
    limit: usize,
    /// Number of rows returned so far.
    returned: usize,
}

impl LimitOperator {
    /// Creates a new limit operator.
    pub fn new(child: Box<dyn Operator>, limit: usize) -> Self {
        Self {
            child,
            limit,
            returned: 0,
        }
    }

    /// Decomposes this operator for push-based conversion.
    pub fn into_parts(self) -> (Box<dyn Operator>, usize) {
        (self.child, self.limit)
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

            // The first rows of the chunk, copied as they are (a column
            // rebuilt by a declared type would turn values of another type
            // into that type's default).
            self.returned += remaining;
            return Ok(Some(chunk.slice(0, remaining)));
        }
    }

    fn reset(&mut self) {
        self.child.reset();
        self.returned = 0;
    }

    fn name(&self) -> &'static str {
        "Limit"
    }

    fn into_any(self: Box<Self>) -> Box<dyn std::any::Any + Send> {
        self
    }
}

/// Skip operator.
///
/// Skips the first `skip` rows from the input. A chunk it cuts short keeps
/// its columns and values as they are.
pub struct SkipOperator {
    /// Child operator.
    child: Box<dyn Operator>,
    /// Number of rows to skip.
    skip: usize,
    /// Number of rows skipped so far.
    skipped: usize,
}

impl SkipOperator {
    /// Creates a new skip operator.
    pub fn new(child: Box<dyn Operator>, skip: usize) -> Self {
        Self {
            child,
            skip,
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

            return Ok(Some(chunk.slice(to_skip, row_count - to_skip)));
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

    fn into_any(self: Box<Self>) -> Box<dyn std::any::Any + Send> {
        self
    }
}

/// Combined Limit and Skip operator.
///
/// Equivalent to OFFSET skip LIMIT limit. A chunk it cuts short keeps its
/// columns and values as they are.
pub struct LimitSkipOperator {
    /// Child operator.
    child: Box<dyn Operator>,
    /// Number of rows to skip.
    skip: usize,
    /// Maximum number of rows to return.
    limit: usize,
    /// Number of rows skipped so far.
    skipped: usize,
    /// Number of rows returned so far.
    returned: usize,
}

impl LimitSkipOperator {
    /// Creates a new limit/skip operator.
    pub fn new(child: Box<dyn Operator>, skip: usize, limit: usize) -> Self {
        Self {
            child,
            skip,
            limit,
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

            self.returned += to_return;
            if start_idx == 0 && to_return == row_count {
                return Ok(Some(chunk));
            }
            return Ok(Some(chunk.slice(start_idx, to_return)));
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

    fn into_any(self: Box<Self>) -> Box<dyn std::any::Any + Send> {
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::execution::DataChunk;
    use crate::execution::chunk::DataChunkBuilder;
    use grafeo_common::types::{LogicalType, Value};

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

        let mut limit = LimitOperator::new(Box::new(mock), 3);

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

        let mut limit = LimitOperator::new(Box::new(mock), 10);

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

        let mut skip = SkipOperator::new(Box::new(mock), 2);

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

        let mut skip = SkipOperator::new(Box::new(mock), 5);

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
            4,
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

        let mut limit = LimitOperator::new(Box::new(mock), 5);

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

        let mut skip = SkipOperator::new(Box::new(mock), 3);

        let mut results = Vec::new();
        while let Some(chunk) = skip.next().unwrap() {
            for row in chunk.selected_indices() {
                let val = chunk.column(0).unwrap().get_int64(row).unwrap();
                results.push(val);
            }
        }

        assert_eq!(results, vec![4, 5, 6]);
    }

    #[test]
    fn test_limit_into_parts() {
        let child = Box::new(MockOperator::new(vec![]));
        let limit = LimitOperator::new(child, 42);
        let (_, limit_value) = limit.into_parts();
        assert_eq!(limit_value, 42);
    }

    #[test]
    fn test_limit_into_any() {
        let child = Box::new(MockOperator::new(vec![]));
        let limit: Box<dyn Operator> = Box::new(LimitOperator::new(child, 10));
        let any = limit.into_any();
        assert!(any.downcast::<LimitOperator>().is_ok());
    }

    /// A chunk of `values` in an untyped column, with its first row filtered
    /// out: the operators must also respect the selection.
    fn mixed_chunk(values: &[Value]) -> DataChunk {
        let mut chunk = DataChunk::new(vec![crate::execution::ValueVector::from_values(values)]);
        chunk.set_selection(crate::execution::SelectionVector::from_predicate(
            values.len(),
            |row| row > 0,
        ));
        chunk
    }

    /// The values the operator returns, in order, and the type of the column
    /// of each chunk it returns.
    fn output_of(mut op: impl Operator) -> (Vec<Value>, Vec<LogicalType>) {
        let (mut values, mut types) = (Vec::new(), Vec::new());
        while let Some(chunk) = op.next().unwrap() {
            let column = chunk.column(0).unwrap();
            types.push(column.data_type().clone());
            values.extend(
                chunk
                    .selected_indices()
                    .map(|row| column.get_value(row).unwrap()),
            );
        }
        (values, types)
    }

    /// #482: cutting a chunk short keeps its values as they are, a mix of
    /// types and a null in an untyped column included. Before, the cut rows
    /// were rebuilt in columns of a declared type, which turned every value
    /// of another type into that type's default.
    #[test]
    fn a_partial_chunk_keeps_its_values() {
        let values = [
            Value::Int64(0),
            Value::Int64(7),
            Value::from("eight"),
            Value::Null,
            Value::Int64(9),
        ];
        let child = || Box::new(MockOperator::new(vec![mixed_chunk(&values)]));

        assert_eq!(output_of(LimitOperator::new(child(), 3)).0, values[1..4]);
        assert_eq!(output_of(SkipOperator::new(child(), 2)).0, values[3..]);
        assert_eq!(
            output_of(LimitSkipOperator::new(child(), 1, 2)).0,
            values[2..4]
        );
    }

    /// A cut chunk keeps its column's type: here node IDs stay node IDs.
    #[test]
    fn a_partial_chunk_keeps_its_column_type() {
        let child = || {
            let mut builder = DataChunkBuilder::new(&[LogicalType::Node]);
            for id in 1..=4 {
                builder
                    .column_mut(0)
                    .unwrap()
                    .push_node_id(grafeo_common::types::NodeId::new(id));
                builder.advance_row();
            }
            Box::new(MockOperator::new(vec![builder.finish()]))
        };

        let (values, types) = output_of(LimitOperator::new(child(), 2));
        assert_eq!(values, [Value::Int64(1), Value::Int64(2)]);
        assert_eq!(types, [LogicalType::Node]);
        let (values, types) = output_of(SkipOperator::new(child(), 3));
        assert_eq!(values, [Value::Int64(4)]);
        assert_eq!(types, [LogicalType::Node]);
        let (values, types) = output_of(LimitSkipOperator::new(child(), 1, 2));
        assert_eq!(values, [Value::Int64(2), Value::Int64(3)]);
        assert_eq!(types, [LogicalType::Node]);
    }
}
