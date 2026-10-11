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
/// its columns and values as they are. Once the limit is reached it reads no
/// more of its input, unless the input writes
/// ([`running_its_input_to_the_end`](Self::running_its_input_to_the_end)).
pub struct LimitOperator {
    /// Child operator.
    child: Box<dyn Operator>,
    /// Maximum number of rows to return.
    limit: usize,
    /// Number of rows returned so far.
    returned: usize,
    /// Whether the child runs to its end whatever rows the limit lets through.
    runs_input_to_the_end: bool,
    /// Whether the child has returned its last chunk.
    input_done: bool,
}

impl LimitOperator {
    /// Creates a new limit operator.
    pub fn new(child: Box<dyn Operator>, limit: usize) -> Self {
        Self {
            child,
            limit,
            returned: 0,
            runs_input_to_the_end: false,
            input_done: false,
        }
    }

    /// Runs the input to its end however many rows the limit lets through,
    /// for an input that writes: the limit cuts the rows, not the write. A
    /// `LIMIT 0` (or a statement without a result) then still runs the write,
    /// as soon as its first row is asked for, and a limit reached reads the
    /// rest of the input before its last row comes out.
    #[must_use]
    pub fn running_its_input_to_the_end(mut self) -> Self {
        self.runs_input_to_the_end = true;
        self
    }

    /// Whether this limit runs its input to its end (see
    /// [`running_its_input_to_the_end`](Self::running_its_input_to_the_end)).
    #[must_use]
    pub fn runs_input_to_the_end(&self) -> bool {
        self.runs_input_to_the_end
    }

    /// Decomposes this operator for push-based conversion.
    pub fn into_parts(self) -> (Box<dyn Operator>, usize) {
        (self.child, self.limit)
    }

    /// Reads the rest of the input once the limit is reached, when it runs
    /// to its end.
    fn finish_input(&mut self) -> Result<(), super::OperatorError> {
        if self.runs_input_to_the_end && !self.input_done {
            while self.child.next()?.is_some() {}
            self.input_done = true;
        }
        Ok(())
    }
}

impl Operator for LimitOperator {
    fn next(&mut self) -> OperatorResult {
        if self.returned >= self.limit {
            self.finish_input()?;
            return Ok(None);
        }

        let remaining = self.limit - self.returned;

        loop {
            let Some(chunk) = self.child.next()? else {
                self.input_done = true;
                return Ok(None);
            };

            let row_count = chunk.row_count();
            if row_count == 0 {
                continue;
            }

            if row_count < remaining {
                // Return entire chunk
                self.returned += row_count;
                return Ok(Some(chunk));
            }

            // The limit is reached: the chunk, or its first rows copied as
            // they are (a column rebuilt by a declared type would turn values
            // of another type into that type's default).
            self.returned += remaining;
            self.finish_input()?;
            return Ok(Some(if row_count == remaining {
                chunk
            } else {
                chunk.slice(0, remaining)
            }));
        }
    }

    fn reset(&mut self) {
        self.child.reset();
        self.returned = 0;
        self.input_done = false;
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

    /// Three chunks of one row each (1, 2 and 3), built again after a reset:
    /// an input that writes one row per chunk it returns.
    struct ThreeRows {
        position: i64,
    }

    impl Operator for ThreeRows {
        fn next(&mut self) -> OperatorResult {
            if self.position == 3 {
                return Ok(None);
            }
            self.position += 1;
            Ok(Some(create_numbered_chunk(&[self.position])))
        }

        fn reset(&mut self) {
            self.position = 0;
        }

        fn name(&self) -> &'static str {
            "ThreeRows"
        }

        fn into_any(self: Box<Self>) -> Box<dyn std::any::Any + Send> {
            self
        }
    }

    /// The values `operator` returns until it ends.
    fn values_of(operator: &mut LimitOperator) -> Vec<i64> {
        let mut values = Vec::new();
        while let Some(chunk) = operator.next().unwrap() {
            for row in chunk.selected_indices() {
                values.push(chunk.column(0).unwrap().get_int64(row).unwrap());
            }
        }
        values
    }

    /// How many rows of its input `operator` read.
    fn rows_read(operator: LimitOperator) -> i64 {
        let (child, _) = operator.into_parts();
        child
            .into_any()
            .downcast::<ThreeRows>()
            .expect("the child is ThreeRows")
            .position
    }

    /// The values a limit returns, and how many rows of its input it read.
    fn limited(limit: usize, runs_to_the_end: bool) -> (Vec<i64>, i64) {
        let mut operator = LimitOperator::new(Box::new(ThreeRows { position: 0 }), limit);
        if runs_to_the_end {
            operator = operator.running_its_input_to_the_end();
        }
        let values = values_of(&mut operator);
        (values, rows_read(operator))
    }

    /// A limit that runs its input to the end reads all of it, a `LIMIT 0`
    /// included, and returns the same rows as one that does not.
    #[test]
    fn a_limit_running_its_input_to_the_end_reads_all_of_it() {
        assert_eq!(limited(0, false), (vec![], 0), "a LIMIT 0 reads nothing");
        assert_eq!(limited(0, true), (vec![], 3));
        assert_eq!(limited(1, false), (vec![1], 1));
        assert_eq!(limited(1, true), (vec![1], 3));
        assert_eq!(limited(3, true), (vec![1, 2, 3], 3));
        assert_eq!(limited(19, true), (vec![1, 2, 3], 3));
    }

    /// The input is read to its end before the last row comes out, so a
    /// reader that stops at the limit leaves no write undone; after a reset
    /// it is read to its end again.
    #[test]
    fn a_limit_running_its_input_to_the_end_finishes_it_with_its_last_row() {
        let mut operator = LimitOperator::new(Box::new(ThreeRows { position: 0 }), 1)
            .running_its_input_to_the_end();
        assert!(operator.runs_input_to_the_end());
        assert_eq!(operator.next().unwrap().map(|c| c.row_count()), Some(1));
        operator.reset();
        assert_eq!(values_of(&mut operator), [1]);
        assert_eq!(
            rows_read(operator),
            3,
            "read to the end again after the reset"
        );
    }
}
