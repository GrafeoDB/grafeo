//! Filter operator for applying predicates.

use super::{Operator, OperatorError, OperatorResult};
use crate::execution::{DataChunk, SelectionVector};
use graphos_common::types::Value;

/// A predicate for filtering rows.
pub trait Predicate: Send + Sync {
    /// Evaluates the predicate for a row.
    fn evaluate(&self, chunk: &DataChunk, row: usize) -> bool;
}

/// A comparison operator.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompareOp {
    /// Equal.
    Eq,
    /// Not equal.
    Ne,
    /// Less than.
    Lt,
    /// Less than or equal.
    Le,
    /// Greater than.
    Gt,
    /// Greater than or equal.
    Ge,
}

/// A simple comparison predicate.
pub struct ComparisonPredicate {
    /// Column index to compare.
    column: usize,
    /// Comparison operator.
    op: CompareOp,
    /// Value to compare against.
    value: Value,
}

impl ComparisonPredicate {
    /// Creates a new comparison predicate.
    pub fn new(column: usize, op: CompareOp, value: Value) -> Self {
        Self { column, op, value }
    }
}

impl Predicate for ComparisonPredicate {
    fn evaluate(&self, chunk: &DataChunk, row: usize) -> bool {
        let col = match chunk.column(self.column) {
            Some(c) => c,
            None => return false,
        };

        let cell_value = match col.get_value(row) {
            Some(v) => v,
            None => return false,
        };

        match (&cell_value, &self.value) {
            (Value::Int64(a), Value::Int64(b)) => match self.op {
                CompareOp::Eq => a == b,
                CompareOp::Ne => a != b,
                CompareOp::Lt => a < b,
                CompareOp::Le => a <= b,
                CompareOp::Gt => a > b,
                CompareOp::Ge => a >= b,
            },
            (Value::Float64(a), Value::Float64(b)) => match self.op {
                CompareOp::Eq => (a - b).abs() < f64::EPSILON,
                CompareOp::Ne => (a - b).abs() >= f64::EPSILON,
                CompareOp::Lt => a < b,
                CompareOp::Le => a <= b,
                CompareOp::Gt => a > b,
                CompareOp::Ge => a >= b,
            },
            (Value::String(a), Value::String(b)) => match self.op {
                CompareOp::Eq => a == b,
                CompareOp::Ne => a != b,
                CompareOp::Lt => a < b,
                CompareOp::Le => a <= b,
                CompareOp::Gt => a > b,
                CompareOp::Ge => a >= b,
            },
            (Value::Bool(a), Value::Bool(b)) => match self.op {
                CompareOp::Eq => a == b,
                CompareOp::Ne => a != b,
                _ => false, // Ordering on booleans doesn't make sense
            },
            _ => false, // Type mismatch
        }
    }
}

/// A filter operator that applies a predicate to filter rows.
pub struct FilterOperator {
    /// Child operator to read from.
    child: Box<dyn Operator>,
    /// Predicate to apply.
    predicate: Box<dyn Predicate>,
}

impl FilterOperator {
    /// Creates a new filter operator.
    pub fn new(child: Box<dyn Operator>, predicate: Box<dyn Predicate>) -> Self {
        Self { child, predicate }
    }
}

impl Operator for FilterOperator {
    fn next(&mut self) -> OperatorResult {
        // Get next chunk from child
        let mut chunk = match self.child.next()? {
            Some(c) => c,
            None => return Ok(None),
        };

        // Apply predicate to create selection vector
        let count = chunk.total_row_count();
        let selection = SelectionVector::from_predicate(count, |row| {
            self.predicate.evaluate(&chunk, row)
        });

        // If nothing passes, skip to next chunk
        if selection.is_empty() {
            return self.next();
        }

        chunk.set_selection(selection);
        Ok(Some(chunk))
    }

    fn reset(&mut self) {
        self.child.reset();
    }

    fn name(&self) -> &'static str {
        "Filter"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::execution::chunk::DataChunkBuilder;
    use graphos_common::types::LogicalType;

    struct MockScanOperator {
        chunks: Vec<DataChunk>,
        position: usize,
    }

    impl Operator for MockScanOperator {
        fn next(&mut self) -> OperatorResult {
            if self.position < self.chunks.len() {
                let chunk = std::mem::replace(
                    &mut self.chunks[self.position],
                    DataChunk::new(&[]),
                );
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
            "MockScan"
        }
    }

    #[test]
    fn test_filter_comparison() {
        // Create a chunk with values [10, 20, 30, 40, 50]
        let mut builder = DataChunkBuilder::new(&[LogicalType::Int64]);
        for i in 1..=5 {
            builder.column_mut(0).unwrap().push_int64(i * 10);
            builder.advance_row();
        }
        let chunk = builder.finish();

        let mock_scan = MockScanOperator {
            chunks: vec![chunk],
            position: 0,
        };

        // Filter for values > 25
        let predicate = ComparisonPredicate::new(0, CompareOp::Gt, Value::Int64(25));
        let mut filter = FilterOperator::new(Box::new(mock_scan), Box::new(predicate));

        let result = filter.next().unwrap().unwrap();
        // Should have 30, 40, 50 (3 values)
        assert_eq!(result.row_count(), 3);
    }
}
