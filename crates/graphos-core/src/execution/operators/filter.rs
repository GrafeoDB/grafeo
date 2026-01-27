//! Filter operator for applying predicates.

use super::{Operator, OperatorResult};
use crate::execution::{DataChunk, SelectionVector};
use crate::graph::lpg::LpgStore;
use graphos_common::types::Value;
use std::collections::HashMap;
use std::sync::Arc;

/// A predicate for filtering rows.
pub trait Predicate: Send + Sync {
    /// Evaluates the predicate for a row.
    fn evaluate(&self, chunk: &DataChunk, row: usize) -> bool;
}

/// A comparison operator.
#[allow(dead_code)]
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
#[allow(dead_code)]
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
    #[allow(dead_code)]
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

/// An expression-based predicate that evaluates logical expressions.
///
/// This predicate can evaluate complex expressions involving variables,
/// properties, and operators.
pub struct ExpressionPredicate {
    /// The expression to evaluate.
    expression: FilterExpression,
    /// Map from variable name to column index.
    variable_columns: HashMap<String, usize>,
    /// The graph store for property lookups.
    store: Arc<LpgStore>,
}

/// A filter expression that can be evaluated.
#[derive(Debug, Clone)]
pub enum FilterExpression {
    /// A literal value.
    Literal(Value),
    /// A variable reference (column index).
    Variable(String),
    /// Property access on a variable.
    Property {
        /// The variable name.
        variable: String,
        /// The property name.
        property: String,
    },
    /// Binary operation.
    Binary {
        /// Left operand.
        left: Box<FilterExpression>,
        /// Operator.
        op: BinaryFilterOp,
        /// Right operand.
        right: Box<FilterExpression>,
    },
    /// Unary operation.
    Unary {
        /// Operator.
        op: UnaryFilterOp,
        /// Operand.
        operand: Box<FilterExpression>,
    },
}

/// Binary operators for filter expressions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BinaryFilterOp {
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
    /// Logical AND.
    And,
    /// Logical OR.
    Or,
}

/// Unary operators for filter expressions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UnaryFilterOp {
    /// Logical NOT.
    Not,
    /// IS NULL.
    IsNull,
    /// IS NOT NULL.
    IsNotNull,
}

impl ExpressionPredicate {
    /// Creates a new expression predicate.
    pub fn new(
        expression: FilterExpression,
        variable_columns: HashMap<String, usize>,
        store: Arc<LpgStore>,
    ) -> Self {
        Self {
            expression,
            variable_columns,
            store,
        }
    }

    /// Evaluates the expression for a row, returning the result value.
    fn eval(&self, chunk: &DataChunk, row: usize) -> Option<Value> {
        self.eval_expr(&self.expression, chunk, row)
    }

    fn eval_expr(&self, expr: &FilterExpression, chunk: &DataChunk, row: usize) -> Option<Value> {
        match expr {
            FilterExpression::Literal(v) => Some(v.clone()),
            FilterExpression::Variable(name) => {
                let col_idx = *self.variable_columns.get(name)?;
                chunk.column(col_idx)?.get_value(row)
            }
            FilterExpression::Property { variable, property } => {
                let col_idx = *self.variable_columns.get(variable)?;
                let col = chunk.column(col_idx)?;
                let node_id = col.get_node_id(row)?;
                let node = self.store.get_node(node_id)?;
                node.get_property(property).cloned()
            }
            FilterExpression::Binary { left, op, right } => {
                let left_val = self.eval_expr(left, chunk, row)?;
                let right_val = self.eval_expr(right, chunk, row)?;
                self.eval_binary_op(&left_val, *op, &right_val)
            }
            FilterExpression::Unary { op, operand } => {
                let val = self.eval_expr(operand, chunk, row);
                self.eval_unary_op(*op, val)
            }
        }
    }

    fn eval_binary_op(&self, left: &Value, op: BinaryFilterOp, right: &Value) -> Option<Value> {
        match op {
            BinaryFilterOp::And => {
                let l = left.as_bool()?;
                let r = right.as_bool()?;
                Some(Value::Bool(l && r))
            }
            BinaryFilterOp::Or => {
                let l = left.as_bool()?;
                let r = right.as_bool()?;
                Some(Value::Bool(l || r))
            }
            BinaryFilterOp::Eq => Some(Value::Bool(self.values_equal(left, right))),
            BinaryFilterOp::Ne => Some(Value::Bool(!self.values_equal(left, right))),
            BinaryFilterOp::Lt => self.compare_values(left, right).map(|c| Value::Bool(c < 0)),
            BinaryFilterOp::Le => self.compare_values(left, right).map(|c| Value::Bool(c <= 0)),
            BinaryFilterOp::Gt => self.compare_values(left, right).map(|c| Value::Bool(c > 0)),
            BinaryFilterOp::Ge => self.compare_values(left, right).map(|c| Value::Bool(c >= 0)),
        }
    }

    fn eval_unary_op(&self, op: UnaryFilterOp, val: Option<Value>) -> Option<Value> {
        match op {
            UnaryFilterOp::Not => {
                let v = val?.as_bool()?;
                Some(Value::Bool(!v))
            }
            UnaryFilterOp::IsNull => Some(Value::Bool(val.is_none() || matches!(val, Some(Value::Null)))),
            UnaryFilterOp::IsNotNull => Some(Value::Bool(val.is_some() && !matches!(val, Some(Value::Null)))),
        }
    }

    fn values_equal(&self, left: &Value, right: &Value) -> bool {
        match (left, right) {
            (Value::Null, Value::Null) => true,
            (Value::Bool(a), Value::Bool(b)) => a == b,
            (Value::Int64(a), Value::Int64(b)) => a == b,
            (Value::Float64(a), Value::Float64(b)) => (a - b).abs() < f64::EPSILON,
            (Value::String(a), Value::String(b)) => a == b,
            (Value::Int64(a), Value::Float64(b)) | (Value::Float64(b), Value::Int64(a)) => {
                (*a as f64 - b).abs() < f64::EPSILON
            }
            _ => false,
        }
    }

    fn compare_values(&self, left: &Value, right: &Value) -> Option<i32> {
        match (left, right) {
            (Value::Int64(a), Value::Int64(b)) => Some(a.cmp(b) as i32),
            (Value::Float64(a), Value::Float64(b)) => {
                if a < b {
                    Some(-1)
                } else if a > b {
                    Some(1)
                } else {
                    Some(0)
                }
            }
            (Value::String(a), Value::String(b)) => Some(a.cmp(b) as i32),
            (Value::Int64(a), Value::Float64(b)) => {
                let af = *a as f64;
                if af < *b {
                    Some(-1)
                } else if af > *b {
                    Some(1)
                } else {
                    Some(0)
                }
            }
            (Value::Float64(a), Value::Int64(b)) => {
                let bf = *b as f64;
                if *a < bf {
                    Some(-1)
                } else if *a > bf {
                    Some(1)
                } else {
                    Some(0)
                }
            }
            _ => None,
        }
    }
}

impl Predicate for ExpressionPredicate {
    fn evaluate(&self, chunk: &DataChunk, row: usize) -> bool {
        match self.eval(chunk, row) {
            Some(Value::Bool(b)) => b,
            _ => false,
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
        let selection =
            SelectionVector::from_predicate(count, |row| self.predicate.evaluate(&chunk, row));

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
