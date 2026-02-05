//! Query executor.
//!
//! Executes physical plans and produces results.

use crate::config::AdaptiveConfig;
use crate::database::QueryResult;
use grafeo_common::types::{LogicalType, Value};
use grafeo_common::utils::error::{Error, Result};
use grafeo_core::execution::operators::{Operator, OperatorError};
use grafeo_core::execution::{
    AdaptiveContext, AdaptiveSummary, CardinalityTrackingWrapper, DataChunk, SharedAdaptiveContext,
};

/// Executes a physical operator tree and collects results.
pub struct Executor {
    /// Column names for the result.
    columns: Vec<String>,
    /// Column types for the result.
    column_types: Vec<LogicalType>,
}

impl Executor {
    /// Creates a new executor.
    #[must_use]
    pub fn new() -> Self {
        Self {
            columns: Vec::new(),
            column_types: Vec::new(),
        }
    }

    /// Creates an executor with specified column names.
    #[must_use]
    pub fn with_columns(columns: Vec<String>) -> Self {
        let len = columns.len();
        Self {
            columns,
            column_types: vec![LogicalType::Any; len],
        }
    }

    /// Creates an executor with specified column names and types.
    #[must_use]
    pub fn with_columns_and_types(columns: Vec<String>, column_types: Vec<LogicalType>) -> Self {
        Self {
            columns,
            column_types,
        }
    }

    /// Executes a physical operator and collects all results.
    ///
    /// # Errors
    ///
    /// Returns an error if operator execution fails.
    pub fn execute(&self, operator: &mut dyn Operator) -> Result<QueryResult> {
        let mut result = QueryResult::with_types(self.columns.clone(), self.column_types.clone());
        let mut types_captured = !result.column_types.iter().all(|t| *t == LogicalType::Any);

        loop {
            match operator.next() {
                Ok(Some(chunk)) => {
                    // Capture column types from first non-empty chunk
                    if !types_captured && chunk.column_count() > 0 {
                        self.capture_column_types(&chunk, &mut result);
                        types_captured = true;
                    }
                    self.collect_chunk(&chunk, &mut result)?;
                }
                Ok(None) => break,
                Err(err) => return Err(convert_operator_error(err)),
            }
        }

        Ok(result)
    }

    /// Executes and returns at most `limit` rows.
    ///
    /// # Errors
    ///
    /// Returns an error if operator execution fails.
    pub fn execute_with_limit(
        &self,
        operator: &mut dyn Operator,
        limit: usize,
    ) -> Result<QueryResult> {
        let mut result = QueryResult::with_types(self.columns.clone(), self.column_types.clone());
        let mut collected = 0;
        let mut types_captured = !result.column_types.iter().all(|t| *t == LogicalType::Any);

        loop {
            if collected >= limit {
                break;
            }

            match operator.next() {
                Ok(Some(chunk)) => {
                    // Capture column types from first non-empty chunk
                    if !types_captured && chunk.column_count() > 0 {
                        self.capture_column_types(&chunk, &mut result);
                        types_captured = true;
                    }
                    let remaining = limit - collected;
                    collected += self.collect_chunk_limited(&chunk, &mut result, remaining)?;
                }
                Ok(None) => break,
                Err(err) => return Err(convert_operator_error(err)),
            }
        }

        Ok(result)
    }

    /// Captures column types from a DataChunk.
    fn capture_column_types(&self, chunk: &DataChunk, result: &mut QueryResult) {
        let col_count = chunk.column_count();
        result.column_types = Vec::with_capacity(col_count);
        for col_idx in 0..col_count {
            let col_type = chunk
                .column(col_idx)
                .map(|col| col.data_type().clone())
                .unwrap_or(LogicalType::Any);
            result.column_types.push(col_type);
        }
    }

    /// Collects all rows from a DataChunk into the result.
    ///
    /// Uses `selected_indices()` to correctly handle chunks with selection vectors
    /// (e.g., after filtering operations).
    fn collect_chunk(&self, chunk: &DataChunk, result: &mut QueryResult) -> Result<usize> {
        let col_count = chunk.column_count();
        let mut collected = 0;

        for row_idx in chunk.selected_indices() {
            let mut row = Vec::with_capacity(col_count);
            for col_idx in 0..col_count {
                let value = chunk
                    .column(col_idx)
                    .and_then(|col| col.get_value(row_idx))
                    .unwrap_or(Value::Null);
                row.push(value);
            }
            result.rows.push(row);
            collected += 1;
        }

        Ok(collected)
    }

    /// Collects up to `limit` rows from a DataChunk.
    ///
    /// Uses `selected_indices()` to correctly handle chunks with selection vectors
    /// (e.g., after filtering operations).
    fn collect_chunk_limited(
        &self,
        chunk: &DataChunk,
        result: &mut QueryResult,
        limit: usize,
    ) -> Result<usize> {
        let col_count = chunk.column_count();
        let mut collected = 0;

        for row_idx in chunk.selected_indices() {
            if collected >= limit {
                break;
            }
            let mut row = Vec::with_capacity(col_count);
            for col_idx in 0..col_count {
                let value = chunk
                    .column(col_idx)
                    .and_then(|col| col.get_value(row_idx))
                    .unwrap_or(Value::Null);
                row.push(value);
            }
            result.rows.push(row);
            collected += 1;
        }

        Ok(collected)
    }

    /// Executes a physical operator with adaptive cardinality tracking.
    ///
    /// This wraps the operator in a cardinality tracking layer and monitors
    /// deviation from estimates during execution. The adaptive summary is
    /// returned alongside the query result.
    ///
    /// # Arguments
    ///
    /// * `operator` - The root physical operator to execute
    /// * `adaptive_context` - Context with cardinality estimates from planning
    /// * `config` - Adaptive execution configuration
    ///
    /// # Errors
    ///
    /// Returns an error if operator execution fails.
    pub fn execute_adaptive(
        &self,
        operator: Box<dyn Operator>,
        adaptive_context: Option<AdaptiveContext>,
        config: &AdaptiveConfig,
    ) -> Result<(QueryResult, Option<AdaptiveSummary>)> {
        // If adaptive is disabled or no context, fall back to normal execution
        if !config.enabled {
            let mut op = operator;
            let result = self.execute(op.as_mut())?;
            return Ok((result, None));
        }

        let ctx = match adaptive_context {
            Some(ctx) => ctx,
            None => {
                let mut op = operator;
                let result = self.execute(op.as_mut())?;
                return Ok((result, None));
            }
        };

        // Create shared context for tracking
        let shared_ctx = SharedAdaptiveContext::from_context(AdaptiveContext::with_thresholds(
            config.threshold,
            config.min_rows,
        ));

        // Copy estimates from the planning context to the shared tracking context
        for (op_id, checkpoint) in ctx.all_checkpoints() {
            if let Some(mut inner) = shared_ctx.snapshot() {
                inner.set_estimate(op_id, checkpoint.estimated);
            }
        }

        // Wrap operator with tracking
        let mut wrapped = CardinalityTrackingWrapper::new(operator, "root", shared_ctx.clone());

        // Execute with tracking
        let mut result = QueryResult::with_types(self.columns.clone(), self.column_types.clone());
        let mut types_captured = !result.column_types.iter().all(|t| *t == LogicalType::Any);
        let mut total_rows: u64 = 0;
        let check_interval = config.min_rows;

        loop {
            match wrapped.next() {
                Ok(Some(chunk)) => {
                    let chunk_rows = chunk.row_count();
                    total_rows += chunk_rows as u64;

                    // Capture column types from first non-empty chunk
                    if !types_captured && chunk.column_count() > 0 {
                        self.capture_column_types(&chunk, &mut result);
                        types_captured = true;
                    }
                    self.collect_chunk(&chunk, &mut result)?;

                    // Periodically check for significant deviation
                    if total_rows >= check_interval && total_rows.is_multiple_of(check_interval) {
                        if shared_ctx.should_reoptimize() {
                            // For now, just log/note that re-optimization would trigger
                            // Full re-optimization would require plan regeneration
                            // which is a more invasive change
                        }
                    }
                }
                Ok(None) => break,
                Err(err) => return Err(convert_operator_error(err)),
            }
        }

        // Get final summary
        let summary = shared_ctx.snapshot().map(|ctx| ctx.summary());

        Ok((result, summary))
    }
}

impl Default for Executor {
    fn default() -> Self {
        Self::new()
    }
}

/// Converts an operator error to a common error.
fn convert_operator_error(err: OperatorError) -> Error {
    match err {
        OperatorError::TypeMismatch { expected, found } => Error::TypeMismatch { expected, found },
        OperatorError::ColumnNotFound(name) => {
            Error::InvalidValue(format!("Column not found: {name}"))
        }
        OperatorError::Execution(msg) => Error::Internal(msg),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use grafeo_common::types::LogicalType;
    use grafeo_core::execution::DataChunk;

    /// A mock operator that generates chunks with integer data on demand.
    struct MockIntOperator {
        values: Vec<i64>,
        position: usize,
        chunk_size: usize,
    }

    impl MockIntOperator {
        fn new(values: Vec<i64>, chunk_size: usize) -> Self {
            Self {
                values,
                position: 0,
                chunk_size,
            }
        }
    }

    impl Operator for MockIntOperator {
        fn next(&mut self) -> grafeo_core::execution::operators::OperatorResult {
            if self.position >= self.values.len() {
                return Ok(None);
            }

            let end = (self.position + self.chunk_size).min(self.values.len());
            let mut chunk = DataChunk::with_capacity(&[LogicalType::Int64], self.chunk_size);

            {
                let col = chunk.column_mut(0).unwrap();
                for i in self.position..end {
                    col.push_int64(self.values[i]);
                }
            }
            chunk.set_count(end - self.position);
            self.position = end;

            Ok(Some(chunk))
        }

        fn reset(&mut self) {
            self.position = 0;
        }

        fn name(&self) -> &'static str {
            "MockInt"
        }
    }

    /// Empty mock operator for testing empty results.
    struct EmptyOperator;

    impl Operator for EmptyOperator {
        fn next(&mut self) -> grafeo_core::execution::operators::OperatorResult {
            Ok(None)
        }

        fn reset(&mut self) {}

        fn name(&self) -> &'static str {
            "Empty"
        }
    }

    #[test]
    fn test_executor_empty() {
        let executor = Executor::with_columns(vec!["a".to_string()]);
        let mut op = EmptyOperator;

        let result = executor.execute(&mut op).unwrap();
        assert!(result.is_empty());
        assert_eq!(result.column_count(), 1);
    }

    #[test]
    fn test_executor_single_chunk() {
        let executor = Executor::with_columns(vec!["value".to_string()]);
        let mut op = MockIntOperator::new(vec![1, 2, 3], 10);

        let result = executor.execute(&mut op).unwrap();
        assert_eq!(result.row_count(), 3);
        assert_eq!(result.rows[0][0], Value::Int64(1));
        assert_eq!(result.rows[1][0], Value::Int64(2));
        assert_eq!(result.rows[2][0], Value::Int64(3));
    }

    #[test]
    fn test_executor_with_limit() {
        let executor = Executor::with_columns(vec!["value".to_string()]);
        let mut op = MockIntOperator::new((0..10).collect(), 100);

        let result = executor.execute_with_limit(&mut op, 5).unwrap();
        assert_eq!(result.row_count(), 5);
    }
}
