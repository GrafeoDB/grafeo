//! Query executor.
//!
//! Executes physical plans and produces results.

use crate::database::QueryResult;
use graphos_common::types::Value;
use graphos_common::utils::error::{Error, Result};
use graphos_core::execution::operators::{Operator, OperatorError};
use graphos_core::execution::DataChunk;

/// Executes a physical operator tree and collects results.
pub struct Executor {
    /// Column names for the result.
    columns: Vec<String>,
}

impl Executor {
    /// Creates a new executor.
    #[must_use]
    pub fn new() -> Self {
        Self {
            columns: Vec::new(),
        }
    }

    /// Creates an executor with specified column names.
    #[must_use]
    pub fn with_columns(columns: Vec<String>) -> Self {
        Self { columns }
    }

    /// Executes a physical operator and collects all results.
    ///
    /// # Errors
    ///
    /// Returns an error if operator execution fails.
    pub fn execute(&self, operator: &mut dyn Operator) -> Result<QueryResult> {
        let mut result = QueryResult::new(self.columns.clone());

        loop {
            match operator.next() {
                Ok(Some(chunk)) => {
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
        let mut result = QueryResult::new(self.columns.clone());
        let mut collected = 0;

        loop {
            if collected >= limit {
                break;
            }

            match operator.next() {
                Ok(Some(chunk)) => {
                    let remaining = limit - collected;
                    collected += self.collect_chunk_limited(&chunk, &mut result, remaining)?;
                }
                Ok(None) => break,
                Err(err) => return Err(convert_operator_error(err)),
            }
        }

        Ok(result)
    }

    /// Collects all rows from a DataChunk into the result.
    fn collect_chunk(&self, chunk: &DataChunk, result: &mut QueryResult) -> Result<usize> {
        let row_count = chunk.row_count();
        let col_count = chunk.column_count();

        for row_idx in 0..row_count {
            let mut row = Vec::with_capacity(col_count);
            for col_idx in 0..col_count {
                let value = chunk
                    .column(col_idx)
                    .and_then(|col| col.get_value(row_idx))
                    .unwrap_or(Value::Null);
                row.push(value);
            }
            result.rows.push(row);
        }

        Ok(row_count)
    }

    /// Collects up to `limit` rows from a DataChunk.
    fn collect_chunk_limited(
        &self,
        chunk: &DataChunk,
        result: &mut QueryResult,
        limit: usize,
    ) -> Result<usize> {
        let row_count = chunk.row_count().min(limit);
        let col_count = chunk.column_count();

        for row_idx in 0..row_count {
            let mut row = Vec::with_capacity(col_count);
            for col_idx in 0..col_count {
                let value = chunk
                    .column(col_idx)
                    .and_then(|col| col.get_value(row_idx))
                    .unwrap_or(Value::Null);
                row.push(value);
            }
            result.rows.push(row);
        }

        Ok(row_count)
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
    use graphos_common::types::LogicalType;
    use graphos_core::execution::DataChunk;

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
        fn next(&mut self) -> graphos_core::execution::operators::OperatorResult {
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
        fn next(&mut self) -> graphos_core::execution::operators::OperatorResult {
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
