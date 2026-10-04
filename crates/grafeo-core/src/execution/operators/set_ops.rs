//! Set operations: EXCEPT, INTERSECT, and OTHERWISE.
//!
//! These operators implement the GQL composite query operations for
//! combining result sets with set semantics.

use std::collections::HashSet;

use grafeo_common::types::{HashableValue, LogicalType, Value};

use super::{DataChunk, Operator, OperatorError, OperatorResult};
use crate::execution::chunk::{ColumnTypes, DataChunkBuilder};

/// A hashable row key: one `HashableValue` per column.
type RowKey = Vec<HashableValue>;

/// Extracts a hashable row key from a `DataChunk`.
fn row_key(chunk: &DataChunk, row: usize) -> RowKey {
    let mut key = Vec::with_capacity(chunk.num_columns());
    for col_idx in 0..chunk.num_columns() {
        let val = chunk
            .column(col_idx)
            .and_then(|col| col.get_value(row))
            .unwrap_or(Value::Null);
        key.push(HashableValue(val));
    }
    key
}

/// Extracts the plain Values from a row key (for chunk reconstruction).
fn row_values(key: &RowKey) -> Vec<Value> {
    key.iter().map(|hv| hv.0.clone()).collect()
}

/// Materializes all rows from an operator into a vector of row keys, with
/// the column types of its chunks.
fn materialize(op: &mut dyn Operator) -> Result<(Vec<RowKey>, Vec<LogicalType>), OperatorError> {
    let mut rows = Vec::new();
    let mut column_types = ColumnTypes::default();
    while let Some(chunk) = op.next()? {
        column_types.add(&chunk);
        for row in chunk.selected_indices() {
            rows.push(row_key(&chunk, row));
        }
    }
    Ok((rows, column_types.types().to_vec()))
}

/// Rebuilds a `DataChunk` from a set of row keys, in columns of the types
/// the rows came in.
fn rows_to_chunk(rows: &[RowKey], schema: &[LogicalType]) -> DataChunk {
    if rows.is_empty() {
        return DataChunk::empty();
    }
    let mut builder = DataChunkBuilder::new(schema);
    for row in rows {
        let values = row_values(row);
        for (col_idx, val) in values.into_iter().enumerate() {
            if let Some(col) = builder.column_mut(col_idx) {
                col.push_value(val);
            }
        }
        builder.advance_row();
    }
    builder.finish()
}

/// EXCEPT operator: rows in left that are not in right.
pub struct ExceptOperator {
    left: Box<dyn Operator>,
    right: Box<dyn Operator>,
    all: bool,
    /// The column types of the left input, which the result rows come from.
    column_types: Vec<LogicalType>,
    result: Option<Vec<RowKey>>,
    position: usize,
}

impl ExceptOperator {
    /// Creates a new EXCEPT operator.
    pub fn new(left: Box<dyn Operator>, right: Box<dyn Operator>, all: bool) -> Self {
        Self {
            left,
            right,
            all,
            column_types: Vec::new(),
            result: None,
            position: 0,
        }
    }

    fn compute(&mut self) -> Result<(), OperatorError> {
        let (left_rows, column_types) = materialize(self.left.as_mut())?;
        let (right_rows, _) = materialize(self.right.as_mut())?;
        self.column_types = column_types;

        if self.all {
            // EXCEPT ALL: for each right row, remove one matching left row
            let mut result = left_rows;
            for right_row in &right_rows {
                if let Some(pos) = result.iter().position(|r| r == right_row) {
                    result.remove(pos);
                }
            }
            self.result = Some(result);
        } else {
            // EXCEPT DISTINCT: remove all matching rows
            let right_set: HashSet<RowKey> = right_rows.into_iter().collect();
            let mut seen = HashSet::new();
            let result: Vec<RowKey> = left_rows
                .into_iter()
                .filter(|row| !right_set.contains(row) && seen.insert(row.clone()))
                .collect();
            self.result = Some(result);
        }
        Ok(())
    }
}

impl Operator for ExceptOperator {
    fn next(&mut self) -> OperatorResult {
        if self.result.is_none() {
            self.compute()?;
        }
        let rows = self
            .result
            .as_ref()
            .expect("result is Some: compute() called above");
        if self.position >= rows.len() {
            return Ok(None);
        }
        // Emit up to 1024 rows per chunk
        let end = (self.position + 1024).min(rows.len());
        let batch = &rows[self.position..end];
        self.position = end;
        if batch.is_empty() {
            Ok(None)
        } else {
            Ok(Some(rows_to_chunk(batch, &self.column_types)))
        }
    }

    fn reset(&mut self) {
        self.left.reset();
        self.right.reset();
        self.result = None;
        self.position = 0;
    }

    fn name(&self) -> &'static str {
        "Except"
    }

    fn into_any(self: Box<Self>) -> Box<dyn std::any::Any + Send> {
        self
    }
}

/// INTERSECT operator: rows common to both inputs.
pub struct IntersectOperator {
    left: Box<dyn Operator>,
    right: Box<dyn Operator>,
    all: bool,
    /// The column types of the left input, which the result rows come from.
    column_types: Vec<LogicalType>,
    result: Option<Vec<RowKey>>,
    position: usize,
}

impl IntersectOperator {
    /// Creates a new INTERSECT operator.
    pub fn new(left: Box<dyn Operator>, right: Box<dyn Operator>, all: bool) -> Self {
        Self {
            left,
            right,
            all,
            column_types: Vec::new(),
            result: None,
            position: 0,
        }
    }

    fn compute(&mut self) -> Result<(), OperatorError> {
        let (left_rows, column_types) = materialize(self.left.as_mut())?;
        let (right_rows, _) = materialize(self.right.as_mut())?;
        self.column_types = column_types;

        if self.all {
            // INTERSECT ALL: each right row matches at most one left row
            let mut remaining_right = right_rows;
            let mut result = Vec::new();
            for left_row in &left_rows {
                if let Some(pos) = remaining_right.iter().position(|r| r == left_row) {
                    result.push(left_row.clone());
                    remaining_right.remove(pos);
                }
            }
            self.result = Some(result);
        } else {
            // INTERSECT DISTINCT: rows present in both, deduplicated
            let right_set: HashSet<RowKey> = right_rows.into_iter().collect();
            let mut seen = HashSet::new();
            let result: Vec<RowKey> = left_rows
                .into_iter()
                .filter(|row| right_set.contains(row) && seen.insert(row.clone()))
                .collect();
            self.result = Some(result);
        }
        Ok(())
    }
}

impl Operator for IntersectOperator {
    fn next(&mut self) -> OperatorResult {
        if self.result.is_none() {
            self.compute()?;
        }
        let rows = self
            .result
            .as_ref()
            .expect("result is Some: compute() called above");
        if self.position >= rows.len() {
            return Ok(None);
        }
        let end = (self.position + 1024).min(rows.len());
        let batch = &rows[self.position..end];
        self.position = end;
        if batch.is_empty() {
            Ok(None)
        } else {
            Ok(Some(rows_to_chunk(batch, &self.column_types)))
        }
    }

    fn reset(&mut self) {
        self.left.reset();
        self.right.reset();
        self.result = None;
        self.position = 0;
    }

    fn name(&self) -> &'static str {
        "Intersect"
    }

    fn into_any(self: Box<Self>) -> Box<dyn std::any::Any + Send> {
        self
    }
}

/// OTHERWISE operator: use left result if non-empty, otherwise use right.
pub struct OtherwiseOperator {
    left: Box<dyn Operator>,
    right: Box<dyn Operator>,
    /// Which input we are currently streaming from.
    state: OtherwiseState,
}

enum OtherwiseState {
    /// Haven't started yet, need to probe left.
    Init,
    /// Left produced rows: buffer first chunk, then stream rest of left.
    StreamingLeft(Option<DataChunk>),
    /// Left was empty: stream right.
    StreamingRight,
    /// Done.
    Done,
}

impl OtherwiseOperator {
    /// Creates a new OTHERWISE operator.
    pub fn new(left: Box<dyn Operator>, right: Box<dyn Operator>) -> Self {
        Self {
            left,
            right,
            state: OtherwiseState::Init,
        }
    }
}

impl Operator for OtherwiseOperator {
    fn next(&mut self) -> OperatorResult {
        loop {
            match &mut self.state {
                OtherwiseState::Init => {
                    // Probe left for first chunk
                    if let Some(chunk) = self.left.next()? {
                        self.state = OtherwiseState::StreamingLeft(Some(chunk));
                    } else {
                        // Left is empty, switch to right
                        self.state = OtherwiseState::StreamingRight;
                    }
                }
                OtherwiseState::StreamingLeft(buffered) => {
                    if let Some(chunk) = buffered.take() {
                        return Ok(Some(chunk));
                    }
                    // Continue streaming from left
                    match self.left.next()? {
                        Some(chunk) => return Ok(Some(chunk)),
                        None => {
                            self.state = OtherwiseState::Done;
                            return Ok(None);
                        }
                    }
                }
                OtherwiseState::StreamingRight => match self.right.next()? {
                    Some(chunk) => return Ok(Some(chunk)),
                    None => {
                        self.state = OtherwiseState::Done;
                        return Ok(None);
                    }
                },
                OtherwiseState::Done => return Ok(None),
            }
        }
    }

    fn reset(&mut self) {
        self.left.reset();
        self.right.reset();
        self.state = OtherwiseState::Init;
    }

    fn name(&self) -> &'static str {
        "Otherwise"
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

    fn create_int_chunk(values: &[i64]) -> DataChunk {
        let mut builder = DataChunkBuilder::new(&[LogicalType::Int64]);
        for &v in values {
            builder.column_mut(0).unwrap().push_int64(v);
            builder.advance_row();
        }
        builder.finish()
    }

    fn collect_ints(op: &mut dyn Operator) -> Vec<i64> {
        let mut result = Vec::new();
        while let Some(chunk) = op.next().unwrap() {
            for row in chunk.selected_indices() {
                if let Some(v) = chunk.column(0).and_then(|c| c.get_int64(row)) {
                    result.push(v);
                }
            }
        }
        result
    }

    #[test]
    fn test_except_distinct() {
        let left = MockOperator::new(vec![create_int_chunk(&[1, 2, 3, 2])]);
        let right = MockOperator::new(vec![create_int_chunk(&[2, 4])]);
        let mut op = ExceptOperator::new(Box::new(left), Box::new(right), false);

        let mut result = collect_ints(&mut op);
        result.sort_unstable();
        assert_eq!(result, vec![1, 3]);
    }

    #[test]
    fn test_except_all() {
        let left = MockOperator::new(vec![create_int_chunk(&[1, 2, 2, 3])]);
        let right = MockOperator::new(vec![create_int_chunk(&[2])]);
        let mut op = ExceptOperator::new(Box::new(left), Box::new(right), true);

        let mut result = collect_ints(&mut op);
        result.sort_unstable();
        // EXCEPT ALL removes one occurrence of 2
        assert_eq!(result, vec![1, 2, 3]);
    }

    #[test]
    fn test_except_empty_right() {
        let left = MockOperator::new(vec![create_int_chunk(&[1, 2])]);
        let right = MockOperator::new(vec![]);
        let mut op = ExceptOperator::new(Box::new(left), Box::new(right), false);

        let mut result = collect_ints(&mut op);
        result.sort_unstable();
        assert_eq!(result, vec![1, 2]);
    }

    #[test]
    fn test_intersect_distinct() {
        let left = MockOperator::new(vec![create_int_chunk(&[1, 2, 3, 2])]);
        let right = MockOperator::new(vec![create_int_chunk(&[2, 3, 4])]);
        let mut op = IntersectOperator::new(Box::new(left), Box::new(right), false);

        let mut result = collect_ints(&mut op);
        result.sort_unstable();
        assert_eq!(result, vec![2, 3]);
    }

    #[test]
    fn test_intersect_all() {
        let left = MockOperator::new(vec![create_int_chunk(&[1, 2, 2, 3])]);
        let right = MockOperator::new(vec![create_int_chunk(&[2, 2, 4])]);
        let mut op = IntersectOperator::new(Box::new(left), Box::new(right), true);

        let mut result = collect_ints(&mut op);
        result.sort_unstable();
        assert_eq!(result, vec![2, 2]);
    }

    #[test]
    fn test_intersect_no_overlap() {
        let left = MockOperator::new(vec![create_int_chunk(&[1, 2])]);
        let right = MockOperator::new(vec![create_int_chunk(&[3, 4])]);
        let mut op = IntersectOperator::new(Box::new(left), Box::new(right), false);

        let result = collect_ints(&mut op);
        assert!(result.is_empty(), "{result:?}");
    }

    #[test]
    fn test_otherwise_left_nonempty() {
        let left = MockOperator::new(vec![create_int_chunk(&[1, 2])]);
        let right = MockOperator::new(vec![create_int_chunk(&[10, 20])]);
        let mut op = OtherwiseOperator::new(Box::new(left), Box::new(right));

        let result = collect_ints(&mut op);
        assert_eq!(result, vec![1, 2]);
    }

    #[test]
    fn test_otherwise_left_empty() {
        let left = MockOperator::new(vec![]);
        let right = MockOperator::new(vec![create_int_chunk(&[10, 20])]);
        let mut op = OtherwiseOperator::new(Box::new(left), Box::new(right));

        let result = collect_ints(&mut op);
        assert_eq!(result, vec![10, 20]);
    }

    #[test]
    fn test_otherwise_both_empty() {
        let left = MockOperator::new(vec![]);
        let right = MockOperator::new(vec![]);
        let mut op = OtherwiseOperator::new(Box::new(left), Box::new(right));

        let result = collect_ints(&mut op);
        assert!(result.is_empty(), "{result:?}");
    }

    #[test]
    fn test_operator_names() {
        let empty = || MockOperator::new(vec![]);

        let op = ExceptOperator::new(Box::new(empty()), Box::new(empty()), false);
        assert_eq!(op.name(), "Except");

        let op = IntersectOperator::new(Box::new(empty()), Box::new(empty()), false);
        assert_eq!(op.name(), "Intersect");

        let op = OtherwiseOperator::new(Box::new(empty()), Box::new(empty()));
        assert_eq!(op.name(), "Otherwise");
    }

    #[test]
    fn test_into_any() {
        let empty = || MockOperator::new(vec![]);

        let op: Box<dyn Operator> = Box::new(ExceptOperator::new(
            Box::new(empty()),
            Box::new(empty()),
            false,
        ));
        assert!(op.into_any().downcast::<ExceptOperator>().is_ok());

        let op: Box<dyn Operator> = Box::new(IntersectOperator::new(
            Box::new(empty()),
            Box::new(empty()),
            false,
        ));
        assert!(op.into_any().downcast::<IntersectOperator>().is_ok());

        let op: Box<dyn Operator> =
            Box::new(OtherwiseOperator::new(Box::new(empty()), Box::new(empty())));
        assert!(op.into_any().downcast::<OtherwiseOperator>().is_ok());
    }
}
