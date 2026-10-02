//! Parameter scan operator for correlated subqueries.
//!
//! Provides a single-row DataChunk of values injected from an outer scope
//! (e.g., the Apply operator). Used as the leaf of inner plans in CALL
//! {subquery} and pattern comprehensions.

use std::sync::Arc;

use grafeo_common::types::LogicalType;
use parking_lot::Mutex;

use super::{DataChunk, Operator, OperatorResult};
use crate::execution::vector::ValueVector;
use grafeo_common::types::Value;

/// Shared state between [`ApplyOperator`](super::ApplyOperator) and [`ParameterScanOperator`].
///
/// The Apply operator writes the current outer row values here before
/// executing the inner plan. The ParameterScan reads them as its output.
#[derive(Debug)]
pub struct ParameterState {
    /// Column names for the injected parameters.
    pub columns: Vec<String>,
    /// Current row values and the types of the columns they come from (set by
    /// Apply before each inner execution; no types: any).
    values: Mutex<Option<(Vec<Value>, Vec<LogicalType>)>>,
}

impl ParameterState {
    /// Creates a new parameter state for the given column names.
    #[must_use]
    pub fn new(columns: Vec<String>) -> Self {
        Self {
            columns,
            values: Mutex::new(None),
        }
    }

    /// Sets the current parameter values (called by the Apply operator), in
    /// columns of any type.
    pub fn set_values(&self, values: Vec<Value>) {
        *self.values.lock() = Some((values, Vec::new()));
    }

    /// Sets the current parameter values with the types of the columns they
    /// come from, so a node or edge of the outer row stays one in the inner
    /// plan (in a column of any type an ID reads whichever entity has it).
    pub fn set_typed_values(&self, values: Vec<Value>, types: Vec<LogicalType>) {
        *self.values.lock() = Some((values, types));
    }

    /// Clears the current parameter values.
    pub fn clear(&self) {
        *self.values.lock() = None;
    }

    /// Takes the current parameter values and their column types.
    fn take_values(&self) -> Option<(Vec<Value>, Vec<LogicalType>)> {
        self.values.lock().take()
    }
}

/// Operator that emits a single row from externally injected parameter values.
///
/// This is the leaf operator for inner plans in correlated subqueries.
/// The [`ApplyOperator`](super::ApplyOperator) sets parameter values via the shared [`ParameterState`]
/// before each inner plan execution.
pub struct ParameterScanOperator {
    state: Arc<ParameterState>,
    emitted: bool,
}

impl ParameterScanOperator {
    /// Creates a new parameter scan operator.
    #[must_use]
    pub fn new(state: Arc<ParameterState>) -> Self {
        Self {
            state,
            emitted: false,
        }
    }

    /// Returns the shared parameter state (for wiring with Apply).
    #[must_use]
    pub fn state(&self) -> &Arc<ParameterState> {
        &self.state
    }
}

impl Operator for ParameterScanOperator {
    fn next(&mut self) -> OperatorResult {
        if self.emitted {
            return Ok(None);
        }
        self.emitted = true;

        let Some((values, types)) = self.state.take_values() else {
            return Ok(None);
        };

        // Build a single-row DataChunk with one column per parameter, in the
        // type of the column it comes from.
        let columns: Vec<ValueVector> = values
            .into_iter()
            .enumerate()
            .map(|(i, val)| {
                let column_type = types.get(i).cloned().unwrap_or(LogicalType::Any);
                let mut col = ValueVector::with_capacity(column_type, 1);
                col.push_value(val);
                col
            })
            .collect();

        if columns.is_empty() {
            return Ok(None);
        }

        Ok(Some(DataChunk::new(columns)))
    }

    fn reset(&mut self) {
        self.emitted = false;
    }

    fn name(&self) -> &'static str {
        "ParameterScan"
    }

    fn into_any(self: Box<Self>) -> Box<dyn std::any::Any + Send> {
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parameter_scan_emits_single_row() {
        let state = Arc::new(ParameterState::new(vec!["x".to_string(), "y".to_string()]));
        let mut op = ParameterScanOperator::new(Arc::clone(&state));

        // Set values
        state.set_values(vec![Value::Int64(42), Value::String("hello".into())]);

        // First call: should emit the row
        let chunk = op.next().unwrap().expect("should emit a chunk");
        assert_eq!(chunk.row_count(), 1);
        assert_eq!(chunk.num_columns(), 2);
        assert_eq!(
            chunk.column(0).unwrap().get_value(0),
            Some(Value::Int64(42))
        );

        // Second call: should be exhausted
        assert!(op.next().unwrap().is_none());
    }

    #[test]
    fn test_parameter_scan_reset() {
        let state = Arc::new(ParameterState::new(vec!["x".to_string()]));
        let mut op = ParameterScanOperator::new(Arc::clone(&state));

        state.set_values(vec![Value::Int64(1)]);
        let _ = op.next().unwrap();
        assert!(op.next().unwrap().is_none());

        // Reset and set new values
        op.reset();
        state.set_values(vec![Value::Int64(2)]);
        let chunk = op.next().unwrap().expect("should emit after reset");
        assert_eq!(chunk.column(0).unwrap().get_value(0), Some(Value::Int64(2)));
    }

    /// Typed values come out in the types of the columns they came from.
    #[test]
    fn typed_values_keep_their_column_types() {
        let state = Arc::new(ParameterState::new(vec!["a".to_string(), "r".to_string()]));
        let mut op = ParameterScanOperator::new(Arc::clone(&state));

        state.set_typed_values(
            vec![Value::Int64(3), Value::Int64(3)],
            vec![LogicalType::Node, LogicalType::Edge],
        );
        let chunk = op.next().unwrap().expect("should emit a chunk");
        assert_eq!(chunk.column_types(), [LogicalType::Node, LogicalType::Edge]);
        assert_eq!(chunk.column(0).unwrap().get_node_id(0).unwrap().as_u64(), 3);
        assert!(chunk.column(0).unwrap().get_edge_id(0).is_none());
        assert_eq!(chunk.column(1).unwrap().get_edge_id(0).unwrap().as_u64(), 3);
        assert!(chunk.column(1).unwrap().get_node_id(0).is_none());
    }

    #[test]
    fn test_parameter_scan_no_values() {
        let state = Arc::new(ParameterState::new(vec!["x".to_string()]));
        let mut op = ParameterScanOperator::new(Arc::clone(&state));

        // No values set: should return None
        assert!(op.next().unwrap().is_none());
    }

    #[test]
    fn test_parameter_scan_into_any() {
        let state = Arc::new(ParameterState::new(vec!["x".to_string()]));
        let op = ParameterScanOperator::new(state);
        let any = Box::new(op).into_any();
        assert!(any.downcast::<ParameterScanOperator>().is_ok());
    }
}
