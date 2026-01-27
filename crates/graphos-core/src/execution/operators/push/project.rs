//! Push-based project operator.

use crate::execution::chunk::DataChunk;
use crate::execution::operators::OperatorError;
use crate::execution::pipeline::{ChunkSizeHint, PushOperator, Sink};
use crate::execution::vector::ValueVector;
use graphos_common::types::Value;

/// Expression that can be evaluated to produce a value.
pub trait ProjectExpression: Send + Sync {
    /// Evaluate the expression for a row.
    fn evaluate(&self, chunk: &DataChunk, row: usize) -> Value;

    /// Evaluate the expression for all rows, returning a vector.
    fn evaluate_batch(&self, chunk: &DataChunk) -> ValueVector {
        let mut result = ValueVector::new();
        for i in chunk.selected_indices() {
            result.push(self.evaluate(chunk, i));
        }
        result
    }
}

/// Expression that returns a column value.
pub struct ColumnExpr {
    column: usize,
}

impl ColumnExpr {
    /// Create a new column expression.
    pub fn new(column: usize) -> Self {
        Self { column }
    }
}

impl ProjectExpression for ColumnExpr {
    fn evaluate(&self, chunk: &DataChunk, row: usize) -> Value {
        chunk
            .column(self.column)
            .and_then(|c| c.get_value(row))
            .unwrap_or(Value::Null)
    }
}

/// Expression that returns a constant value.
pub struct ConstantExpr {
    value: Value,
}

impl ConstantExpr {
    /// Create a new constant expression.
    pub fn new(value: Value) -> Self {
        Self { value }
    }
}

impl ProjectExpression for ConstantExpr {
    fn evaluate(&self, _chunk: &DataChunk, _row: usize) -> Value {
        self.value.clone()
    }
}

/// Arithmetic operations.
#[derive(Debug, Clone, Copy)]
pub enum ArithOp {
    /// Addition.
    Add,
    /// Subtraction.
    Sub,
    /// Multiplication.
    Mul,
    /// Division.
    Div,
    /// Modulo.
    Mod,
}

/// Binary arithmetic expression.
pub struct BinaryExpr {
    left: Box<dyn ProjectExpression>,
    right: Box<dyn ProjectExpression>,
    op: ArithOp,
}

impl BinaryExpr {
    /// Create a new binary expression.
    pub fn new(
        left: Box<dyn ProjectExpression>,
        right: Box<dyn ProjectExpression>,
        op: ArithOp,
    ) -> Self {
        Self { left, right, op }
    }
}

impl ProjectExpression for BinaryExpr {
    fn evaluate(&self, chunk: &DataChunk, row: usize) -> Value {
        let left_val = self.left.evaluate(chunk, row);
        let right_val = self.right.evaluate(chunk, row);

        match (&left_val, &right_val) {
            (Value::Int64(l), Value::Int64(r)) => match self.op {
                ArithOp::Add => Value::Int64(l.wrapping_add(*r)),
                ArithOp::Sub => Value::Int64(l.wrapping_sub(*r)),
                ArithOp::Mul => Value::Int64(l.wrapping_mul(*r)),
                ArithOp::Div => {
                    if *r == 0 {
                        Value::Null
                    } else {
                        Value::Int64(l / r)
                    }
                }
                ArithOp::Mod => {
                    if *r == 0 {
                        Value::Null
                    } else {
                        Value::Int64(l % r)
                    }
                }
            },
            (Value::Float64(l), Value::Float64(r)) => match self.op {
                ArithOp::Add => Value::Float64(l + r),
                ArithOp::Sub => Value::Float64(l - r),
                ArithOp::Mul => Value::Float64(l * r),
                ArithOp::Div => Value::Float64(l / r),
                ArithOp::Mod => Value::Float64(l % r),
            },
            _ => Value::Null,
        }
    }
}

/// Push-based project operator.
///
/// Evaluates expressions on incoming chunks to produce projected output.
pub struct ProjectPushOperator {
    expressions: Vec<Box<dyn ProjectExpression>>,
}

impl ProjectPushOperator {
    /// Create a new project operator with the given expressions.
    pub fn new(expressions: Vec<Box<dyn ProjectExpression>>) -> Self {
        Self { expressions }
    }

    /// Create a project operator that selects specific columns.
    pub fn select_columns(columns: &[usize]) -> Self {
        let expressions: Vec<Box<dyn ProjectExpression>> = columns
            .iter()
            .map(|&c| Box::new(ColumnExpr::new(c)) as Box<dyn ProjectExpression>)
            .collect();
        Self { expressions }
    }
}

impl PushOperator for ProjectPushOperator {
    fn push(&mut self, chunk: DataChunk, sink: &mut dyn Sink) -> Result<bool, OperatorError> {
        if chunk.is_empty() {
            return Ok(true);
        }

        // Evaluate all expressions to produce output columns
        let columns: Vec<ValueVector> = self
            .expressions
            .iter()
            .map(|expr| expr.evaluate_batch(&chunk))
            .collect();

        let projected = DataChunk::new(columns);

        // Forward to sink
        sink.consume(projected)
    }

    fn finalize(&mut self, _sink: &mut dyn Sink) -> Result<(), OperatorError> {
        // Project is stateless, nothing to finalize
        Ok(())
    }

    fn preferred_chunk_size(&self) -> ChunkSizeHint {
        ChunkSizeHint::Default
    }

    fn name(&self) -> &'static str {
        "ProjectPush"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::execution::sink::CollectorSink;

    fn create_test_chunk(col1: &[i64], col2: &[i64]) -> DataChunk {
        let v1: Vec<Value> = col1.iter().map(|&i| Value::Int64(i)).collect();
        let v2: Vec<Value> = col2.iter().map(|&i| Value::Int64(i)).collect();
        let vec1 = ValueVector::from_values(&v1);
        let vec2 = ValueVector::from_values(&v2);
        DataChunk::new(vec![vec1, vec2])
    }

    #[test]
    fn test_project_select_columns() {
        let mut project = ProjectPushOperator::select_columns(&[1, 0]); // Swap columns
        let mut sink = CollectorSink::new();

        let chunk = create_test_chunk(&[1, 2, 3], &[10, 20, 30]);
        project.push(chunk, &mut sink).unwrap();
        project.finalize(&mut sink).unwrap();

        assert_eq!(sink.row_count(), 3);
        let chunks = sink.into_chunks();
        assert_eq!(chunks.len(), 1);

        // First column should now be the original second column
        let col = chunks[0].column(0).unwrap();
        assert_eq!(col.get_value(0), Some(Value::Int64(10)));
    }

    #[test]
    fn test_project_constant() {
        let expressions: Vec<Box<dyn ProjectExpression>> =
            vec![Box::new(ConstantExpr::new(Value::Int64(42)))];
        let mut project = ProjectPushOperator::new(expressions);
        let mut sink = CollectorSink::new();

        let chunk = create_test_chunk(&[1, 2, 3], &[10, 20, 30]);
        project.push(chunk, &mut sink).unwrap();
        project.finalize(&mut sink).unwrap();

        assert_eq!(sink.row_count(), 3);
        let chunks = sink.into_chunks();
        let col = chunks[0].column(0).unwrap();
        assert_eq!(col.get_value(0), Some(Value::Int64(42)));
        assert_eq!(col.get_value(1), Some(Value::Int64(42)));
        assert_eq!(col.get_value(2), Some(Value::Int64(42)));
    }

    #[test]
    fn test_project_arithmetic() {
        let expressions: Vec<Box<dyn ProjectExpression>> = vec![Box::new(BinaryExpr::new(
            Box::new(ColumnExpr::new(0)),
            Box::new(ColumnExpr::new(1)),
            ArithOp::Add,
        ))];
        let mut project = ProjectPushOperator::new(expressions);
        let mut sink = CollectorSink::new();

        let chunk = create_test_chunk(&[1, 2, 3], &[10, 20, 30]);
        project.push(chunk, &mut sink).unwrap();
        project.finalize(&mut sink).unwrap();

        let chunks = sink.into_chunks();
        let col = chunks[0].column(0).unwrap();
        assert_eq!(col.get_value(0), Some(Value::Int64(11))); // 1 + 10
        assert_eq!(col.get_value(1), Some(Value::Int64(22))); // 2 + 20
        assert_eq!(col.get_value(2), Some(Value::Int64(33))); // 3 + 30
    }
}
