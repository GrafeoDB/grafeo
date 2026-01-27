//! Push-based filter operator.

use crate::execution::chunk::DataChunk;
use crate::execution::operators::OperatorError;
use crate::execution::pipeline::{ChunkSizeHint, PushOperator, Sink};
use crate::execution::selection::SelectionVector;
use graphos_common::types::Value;

/// Predicate for filtering rows.
pub trait FilterPredicate: Send + Sync {
    /// Evaluate the predicate for a row, returning true if it passes.
    fn evaluate(&self, chunk: &DataChunk, row: usize) -> bool;

    /// Evaluate predicate for all rows, returning a selection vector.
    fn evaluate_batch(&self, chunk: &DataChunk) -> SelectionVector {
        SelectionVector::from_predicate(chunk.len(), |i| self.evaluate(chunk, i))
    }
}

/// Column comparison predicate.
pub struct ColumnPredicate {
    /// Column index to check.
    pub column: usize,
    /// Comparison operator.
    pub op: CompareOp,
    /// Value to compare against.
    pub value: Value,
}

/// Comparison operator.
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

impl FilterPredicate for ColumnPredicate {
    fn evaluate(&self, chunk: &DataChunk, row: usize) -> bool {
        let Some(col) = chunk.column(self.column) else {
            return false;
        };

        let Some(val) = col.get_value(row) else {
            return false;
        };

        match self.op {
            CompareOp::Eq => val == self.value,
            CompareOp::Ne => val != self.value,
            CompareOp::Lt => compare_values(&val, &self.value) == Some(std::cmp::Ordering::Less),
            CompareOp::Le => {
                matches!(
                    compare_values(&val, &self.value),
                    Some(std::cmp::Ordering::Less | std::cmp::Ordering::Equal)
                )
            }
            CompareOp::Gt => {
                compare_values(&val, &self.value) == Some(std::cmp::Ordering::Greater)
            }
            CompareOp::Ge => {
                matches!(
                    compare_values(&val, &self.value),
                    Some(std::cmp::Ordering::Greater | std::cmp::Ordering::Equal)
                )
            }
        }
    }
}

/// Helper to compare two values.
fn compare_values(a: &Value, b: &Value) -> Option<std::cmp::Ordering> {
    match (a, b) {
        (Value::Int64(a), Value::Int64(b)) => Some(a.cmp(b)),
        (Value::Float64(a), Value::Float64(b)) => a.partial_cmp(b),
        (Value::String(a), Value::String(b)) => Some(a.cmp(b)),
        (Value::Bool(a), Value::Bool(b)) => Some(a.cmp(b)),
        _ => None,
    }
}

/// Logical AND of two predicates.
pub struct AndPredicate<L, R> {
    left: L,
    right: R,
}

impl<L, R> AndPredicate<L, R> {
    /// Create a new AND predicate.
    pub fn new(left: L, right: R) -> Self {
        Self { left, right }
    }
}

impl<L: FilterPredicate, R: FilterPredicate> FilterPredicate for AndPredicate<L, R> {
    fn evaluate(&self, chunk: &DataChunk, row: usize) -> bool {
        self.left.evaluate(chunk, row) && self.right.evaluate(chunk, row)
    }
}

/// Logical OR of two predicates.
pub struct OrPredicate<L, R> {
    left: L,
    right: R,
}

impl<L, R> OrPredicate<L, R> {
    /// Create a new OR predicate.
    pub fn new(left: L, right: R) -> Self {
        Self { left, right }
    }
}

impl<L: FilterPredicate, R: FilterPredicate> FilterPredicate for OrPredicate<L, R> {
    fn evaluate(&self, chunk: &DataChunk, row: usize) -> bool {
        self.left.evaluate(chunk, row) || self.right.evaluate(chunk, row)
    }
}

/// Not null predicate.
pub struct NotNullPredicate {
    column: usize,
}

impl NotNullPredicate {
    /// Create a new NOT NULL predicate.
    pub fn new(column: usize) -> Self {
        Self { column }
    }
}

impl FilterPredicate for NotNullPredicate {
    fn evaluate(&self, chunk: &DataChunk, row: usize) -> bool {
        chunk
            .column(self.column)
            .and_then(|c| c.get_value(row))
            .map(|v| !matches!(v, Value::Null))
            .unwrap_or(false)
    }
}

/// Push-based filter operator.
///
/// Evaluates a predicate on incoming chunks and forwards only matching rows.
pub struct FilterPushOperator {
    predicate: Box<dyn FilterPredicate>,
}

impl FilterPushOperator {
    /// Create a new filter operator with the given predicate.
    pub fn new(predicate: Box<dyn FilterPredicate>) -> Self {
        Self { predicate }
    }

    /// Create a filter with a column comparison predicate.
    pub fn column_compare(column: usize, op: CompareOp, value: Value) -> Self {
        Self::new(Box::new(ColumnPredicate { column, op, value }))
    }
}

impl PushOperator for FilterPushOperator {
    fn push(&mut self, chunk: DataChunk, sink: &mut dyn Sink) -> Result<bool, OperatorError> {
        if chunk.is_empty() {
            return Ok(true);
        }

        // Evaluate predicate on all rows
        let selection = self.predicate.evaluate_batch(&chunk);

        if selection.is_empty() {
            // No rows match, nothing to forward
            return Ok(true);
        }

        // Create filtered chunk
        let filtered = chunk.filter(&selection);

        // Forward to sink
        sink.consume(filtered)
    }

    fn finalize(&mut self, _sink: &mut dyn Sink) -> Result<(), OperatorError> {
        // Filter is stateless, nothing to finalize
        Ok(())
    }

    fn preferred_chunk_size(&self) -> ChunkSizeHint {
        // Default chunk size works well for filtering
        ChunkSizeHint::Default
    }

    fn name(&self) -> &'static str {
        "FilterPush"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::execution::sink::CollectorSink;
    use crate::execution::vector::ValueVector;

    fn create_test_chunk(values: &[i64]) -> DataChunk {
        let v: Vec<Value> = values.iter().map(|&i| Value::Int64(i)).collect();
        let vector = ValueVector::from_values(&v);
        DataChunk::new(vec![vector])
    }

    #[test]
    fn test_filter_greater_than() {
        let mut filter = FilterPushOperator::column_compare(0, CompareOp::Gt, Value::Int64(5));
        let mut sink = CollectorSink::new();

        let chunk = create_test_chunk(&[1, 5, 10, 3, 8, 2]);
        filter.push(chunk, &mut sink).unwrap();
        filter.finalize(&mut sink).unwrap();

        assert_eq!(sink.row_count(), 2); // 10 and 8
    }

    #[test]
    fn test_filter_equals() {
        let mut filter = FilterPushOperator::column_compare(0, CompareOp::Eq, Value::Int64(5));
        let mut sink = CollectorSink::new();

        let chunk = create_test_chunk(&[1, 5, 10, 5, 8, 5]);
        filter.push(chunk, &mut sink).unwrap();
        filter.finalize(&mut sink).unwrap();

        assert_eq!(sink.row_count(), 3); // Three 5s
    }

    #[test]
    fn test_filter_no_matches() {
        let mut filter = FilterPushOperator::column_compare(0, CompareOp::Gt, Value::Int64(100));
        let mut sink = CollectorSink::new();

        let chunk = create_test_chunk(&[1, 5, 10, 3, 8, 2]);
        filter.push(chunk, &mut sink).unwrap();
        filter.finalize(&mut sink).unwrap();

        assert_eq!(sink.row_count(), 0);
    }

    #[test]
    fn test_filter_all_match() {
        let mut filter = FilterPushOperator::column_compare(0, CompareOp::Gt, Value::Int64(0));
        let mut sink = CollectorSink::new();

        let chunk = create_test_chunk(&[1, 5, 10, 3, 8, 2]);
        filter.push(chunk, &mut sink).unwrap();
        filter.finalize(&mut sink).unwrap();

        assert_eq!(sink.row_count(), 6);
    }
}
