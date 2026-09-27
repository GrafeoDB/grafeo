//! Query executor.
//!
//! Executes physical plans and produces results.

#[cfg(feature = "algos")]
pub mod procedure_call;
#[cfg(all(feature = "gql", feature = "lpg"))]
pub mod stream;
#[cfg(all(feature = "algos", feature = "gql"))]
pub mod user_procedure;

use std::time::{Duration, Instant};

use crate::config::AdaptiveConfig;
use crate::database::QueryResult;
use grafeo_common::grafeo_debug_span;
use grafeo_common::types::{LogicalType, Value};
use grafeo_common::utils::error::{Error, QueryError, Result};
use grafeo_core::execution::operators::{Operator, OperatorError};
use grafeo_core::execution::{
    AdaptiveContext, AdaptiveSummary, CardinalityTrackingWrapper, DataChunk, Pipeline,
    SharedAdaptiveContext,
};

/// Executes a physical operator tree and collects results.
pub struct Executor {
    /// Column names for the result.
    columns: Vec<String>,
    /// Column types for the result.
    column_types: Vec<LogicalType>,
    /// Wall-clock deadline after which execution is aborted.
    deadline: Option<Instant>,
    /// The configured timeout duration (for error messages).
    query_timeout: Option<Duration>,
}

impl Executor {
    /// Creates a new executor.
    #[must_use]
    pub fn new() -> Self {
        Self {
            columns: Vec::new(),
            column_types: Vec::new(),
            deadline: None,
            query_timeout: None,
        }
    }

    /// Creates an executor with specified column names.
    #[must_use]
    pub fn with_columns(columns: Vec<String>) -> Self {
        let len = columns.len();
        Self {
            columns,
            column_types: vec![LogicalType::Any; len],
            deadline: None,
            query_timeout: None,
        }
    }

    /// Creates an executor with specified column names and types.
    #[must_use]
    pub fn with_columns_and_types(columns: Vec<String>, column_types: Vec<LogicalType>) -> Self {
        Self {
            columns,
            column_types,
            deadline: None,
            query_timeout: None,
        }
    }

    /// Sets a wall-clock deadline for query execution.
    #[must_use]
    pub fn with_deadline(mut self, deadline: Option<Instant>) -> Self {
        self.deadline = deadline;
        self
    }

    /// Sets the original timeout duration (used for error messages).
    #[must_use]
    pub fn with_timeout_duration(mut self, timeout: Option<Duration>) -> Self {
        self.query_timeout = timeout;
        self
    }

    /// Checks whether the deadline has been exceeded.
    fn check_deadline(&self) -> Result<()> {
        #[cfg(not(target_arch = "wasm32"))]
        if let Some(deadline) = self.deadline
            && Instant::now() >= deadline
        {
            return Err(Error::Query(match self.query_timeout {
                Some(d) => QueryError::timeout_with_limit(d),
                None => QueryError::timeout(),
            }));
        }
        Ok(())
    }

    /// Executes a physical operator and collects all results.
    ///
    /// # Errors
    ///
    /// Returns an error if operator execution fails or the query timeout is exceeded.
    pub fn execute(&self, operator: &mut dyn Operator) -> Result<QueryResult> {
        let _span = grafeo_debug_span!("grafeo::query::execute");
        let mut result = QueryResult::with_types(self.columns.clone(), self.column_types.clone())?;
        let mut types_captured = !result.column_types.iter().all(|t| *t == LogicalType::Any);

        loop {
            self.check_deadline()?;

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

    /// Executes a push-based pipeline.
    ///
    /// The source operator is wrapped in `OperatorSource`, push operators form
    /// the pipeline body, and a `ChunkCollector` gathers results.
    ///
    /// # Panics
    ///
    /// Panics if the internal sink downcast fails (should never happen since we
    /// create the `ChunkCollector` ourselves).
    ///
    /// # Errors
    ///
    /// Returns an error if pipeline execution fails or the query timeout is exceeded.
    pub fn execute_pipeline(
        &self,
        source: Box<dyn Operator>,
        push_ops: Vec<Box<dyn grafeo_core::execution::pipeline::PushOperator>>,
    ) -> Result<QueryResult> {
        use grafeo_core::execution::{ChunkCollector, OperatorSource};

        let _span = grafeo_debug_span!("grafeo::query::execute_pipeline");

        let source = Box::new(OperatorSource::new(source));
        let collector = ChunkCollector::new();

        // Build and execute the pipeline with deadline enforcement
        let mut pipeline = Pipeline::new(source, push_ops, Box::new(collector));
        pipeline.set_deadline(self.deadline);
        pipeline.execute().map_err(convert_operator_error)?;

        // Extract the sink (ChunkCollector) and get the chunks
        // Safety: we know the sink is a ChunkCollector because we just created it
        let sink_box = pipeline.into_sink();
        let any_sink: Box<dyn std::any::Any> = sink_box.into_any();
        let collector = any_sink
            .downcast::<ChunkCollector>()
            .expect("sink should be ChunkCollector");
        let chunks = collector.into_chunks();

        let mut result = QueryResult::with_types(self.columns.clone(), self.column_types.clone())?;
        let mut types_captured = !result.column_types.iter().all(|t| *t == LogicalType::Any);

        for chunk in &chunks {
            if !types_captured && chunk.column_count() > 0 {
                self.capture_column_types(chunk, &mut result);
                types_captured = true;
            }
            self.collect_chunk(chunk, &mut result)?;
        }

        Ok(result)
    }

    /// Executes and returns at most `limit` rows.
    ///
    /// # Errors
    ///
    /// Returns an error if operator execution fails or the query timeout is exceeded.
    pub fn execute_with_limit(
        &self,
        operator: &mut dyn Operator,
        limit: usize,
    ) -> Result<QueryResult> {
        let mut result = QueryResult::with_types(self.columns.clone(), self.column_types.clone())?;
        let mut collected = 0;
        let mut types_captured = !result.column_types.iter().all(|t| *t == LogicalType::Any);

        loop {
            if collected >= limit {
                break;
            }

            self.check_deadline()?;

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
                .map_or(LogicalType::Any, |col| col.data_type().clone());
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

        let Some(ctx) = adaptive_context else {
            let mut op = operator;
            let result = self.execute(op.as_mut())?;
            return Ok((result, None));
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
        let mut result = QueryResult::with_types(self.columns.clone(), self.column_types.clone())?;
        let mut types_captured = !result.column_types.iter().all(|t| *t == LogicalType::Any);
        let mut total_rows: u64 = 0;
        let check_interval = config.min_rows;

        loop {
            self.check_deadline()?;

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
                    if total_rows >= check_interval
                        && total_rows.is_multiple_of(check_interval)
                        && shared_ctx.should_reoptimize()
                    {
                        // For now, just log/note that re-optimization would trigger
                        // Full re-optimization would require plan regeneration
                        // which is a more invasive change
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
pub(crate) fn convert_operator_error(err: OperatorError) -> Error {
    match err {
        OperatorError::TypeMismatch { expected, found } => Error::TypeMismatch { expected, found },
        OperatorError::ColumnNotFound(name) => {
            Error::InvalidValue(format!("Column not found: {name}"))
        }
        OperatorError::Execution(msg) => Error::Internal(msg),
        OperatorError::ConstraintViolation(msg) => {
            Error::InvalidValue(format!("Constraint violation: {msg}"))
        }
        OperatorError::WriteConflict(msg) => {
            Error::Transaction(grafeo_common::utils::error::TransactionError::WriteConflict(msg))
        }
        _ => Error::Internal(format!("{err}")),
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

        fn into_any(self: Box<Self>) -> Box<dyn std::any::Any + Send> {
            self
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

        fn into_any(self: Box<Self>) -> Box<dyn std::any::Any + Send> {
            self
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

    #[test]
    fn test_executor_timeout_expired() {
        use std::time::{Duration, Instant};

        // Set a deadline that has already passed
        let executor = Executor::with_columns(vec!["value".to_string()]).with_deadline(Some(
            Instant::now().checked_sub(Duration::from_secs(1)).unwrap(),
        ));
        let mut op = MockIntOperator::new(vec![1, 2, 3], 10);

        let result = executor.execute(&mut op);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(
            err.to_string().contains("Query exceeded timeout"),
            "Expected timeout error, got: {err}"
        );
    }

    #[test]
    fn test_executor_no_timeout() {
        // No deadline set - should execute normally
        let executor = Executor::with_columns(vec!["value".to_string()]).with_deadline(None);
        let mut op = MockIntOperator::new(vec![1, 2, 3], 10);

        let result = executor.execute(&mut op).unwrap();
        assert_eq!(result.row_count(), 3);
    }

    #[test]
    fn test_executor_type_capture_from_first_chunk() {
        // When column_types are all Any, types should be captured from the first
        // non-empty chunk.
        let executor = Executor::with_columns(vec!["value".to_string()]);
        // column_types starts as [Any] from with_columns
        let mut op = MockIntOperator::new(vec![42, 99], 10);

        let result = executor.execute(&mut op).unwrap();
        assert_eq!(result.row_count(), 2);
        // After execution, column types should be captured as Int64
        assert_eq!(result.column_types, vec![LogicalType::Int64]);
    }

    #[test]
    fn test_executor_type_capture_with_explicit_types() {
        // When column_types are explicitly set (not all Any), types should NOT be
        // overwritten from chunks.
        let executor =
            Executor::with_columns_and_types(vec!["value".to_string()], vec![LogicalType::String]);
        let mut op = MockIntOperator::new(vec![1], 10);

        let result = executor.execute(&mut op).unwrap();
        assert_eq!(result.row_count(), 1);
        // Types should remain as explicitly set (String), not changed to Int64
        assert_eq!(result.column_types, vec![LogicalType::String]);
    }

    #[test]
    fn test_execute_pipeline_basic() {
        let source = Box::new(MockIntOperator::new(vec![10, 20, 30], 10));
        let executor = Executor::with_columns(vec!["value".to_string()]);

        let result = executor.execute_pipeline(source, vec![]).unwrap();
        assert_eq!(result.row_count(), 3);
        assert_eq!(result.rows[0][0], Value::Int64(10));
        assert_eq!(result.rows[1][0], Value::Int64(20));
        assert_eq!(result.rows[2][0], Value::Int64(30));
    }

    #[test]
    fn test_execute_pipeline_empty_source() {
        let source = Box::new(EmptyOperator);
        let executor = Executor::with_columns(vec!["value".to_string()]);

        let result = executor.execute_pipeline(source, vec![]).unwrap();
        assert!(result.is_empty());
    }

    #[test]
    fn test_execute_pipeline_type_capture() {
        // Pipeline should capture column types from first non-empty chunk when
        // column_types are all Any.
        let source = Box::new(MockIntOperator::new(vec![1, 2], 10));
        let executor = Executor::with_columns(vec!["value".to_string()]);

        let result = executor.execute_pipeline(source, vec![]).unwrap();
        assert_eq!(result.column_types, vec![LogicalType::Int64]);
    }

    #[test]
    fn test_execute_pipeline_explicit_types_preserved() {
        // Pipeline should preserve explicitly set column types.
        let source = Box::new(MockIntOperator::new(vec![1], 10));
        let executor =
            Executor::with_columns_and_types(vec!["value".to_string()], vec![LogicalType::String]);

        let result = executor.execute_pipeline(source, vec![]).unwrap();
        // Explicit types should not be overwritten
        assert_eq!(result.column_types, vec![LogicalType::String]);
    }

    #[test]
    fn test_execute_with_limit_type_capture() {
        // execute_with_limit should also capture types from first chunk
        let executor = Executor::with_columns(vec!["value".to_string()]);
        let mut op = MockIntOperator::new(vec![1, 2, 3, 4, 5], 2);

        let result = executor.execute_with_limit(&mut op, 3).unwrap();
        assert_eq!(result.row_count(), 3);
        assert_eq!(result.column_types, vec![LogicalType::Int64]);
    }

    #[test]
    fn test_execute_with_limit_timeout_expired() {
        use std::time::{Duration, Instant};

        let expired = Instant::now().checked_sub(Duration::from_secs(1)).unwrap();
        let executor =
            Executor::with_columns(vec!["value".to_string()]).with_deadline(Some(expired));
        let mut op = MockIntOperator::new(vec![1, 2, 3], 10);

        let result = executor.execute_with_limit(&mut op, 10);
        assert!(result.is_err());
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("Query exceeded timeout")
        );
    }

    #[test]
    fn test_convert_operator_error_variants() {
        // Test all OperatorError conversion branches
        let err = convert_operator_error(OperatorError::TypeMismatch {
            expected: "Int64".to_string(),
            found: "String".to_string(),
        });
        assert!(matches!(err, Error::TypeMismatch { .. }));

        let err = convert_operator_error(OperatorError::ColumnNotFound("col_x".to_string()));
        assert!(matches!(err, Error::InvalidValue(_)));
        assert!(err.to_string().contains("col_x"));

        let err = convert_operator_error(OperatorError::Execution("internal issue".to_string()));
        assert!(matches!(err, Error::Internal(_)));

        let err = convert_operator_error(OperatorError::ConstraintViolation("unique".to_string()));
        assert!(matches!(err, Error::InvalidValue(_)));
        assert!(err.to_string().contains("unique"));

        let err =
            convert_operator_error(OperatorError::WriteConflict("concurrent write".to_string()));
        assert!(matches!(err, Error::Transaction(_)));
    }

    #[test]
    fn test_execute_pipeline_timeout_expired() {
        use std::time::{Duration, Instant};

        use grafeo_core::execution::pipeline::{Sink as PipelineSink, Source as PipelineSource};

        struct PipelineTestSource {
            remaining: usize,
        }

        impl PipelineSource for PipelineTestSource {
            fn next_chunk(
                &mut self,
                _chunk_size: usize,
            ) -> std::result::Result<Option<DataChunk>, OperatorError> {
                if self.remaining == 0 {
                    return Ok(None);
                }
                self.remaining -= 1;
                Ok(Some(DataChunk::empty()))
            }
            fn reset(&mut self) {}
            fn name(&self) -> &'static str {
                "PipelineTestSource"
            }
        }

        struct PipelineTestSink;

        impl PipelineSink for PipelineTestSink {
            fn consume(&mut self, _chunk: DataChunk) -> std::result::Result<bool, OperatorError> {
                Ok(true)
            }
            fn finalize(&mut self) -> std::result::Result<(), OperatorError> {
                Ok(())
            }
            fn name(&self) -> &'static str {
                "PipelineTestSink"
            }
            fn into_any(self: Box<Self>) -> Box<dyn std::any::Any> {
                self
            }
        }

        let expired = Instant::now().checked_sub(Duration::from_secs(1)).unwrap();
        let mut pipeline = Pipeline::simple(
            Box::new(PipelineTestSource { remaining: 10 }),
            Box::new(PipelineTestSink),
        )
        .with_deadline(Some(expired));

        let result = pipeline.execute().map_err(convert_operator_error);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(
            err.to_string().contains("Query exceeded timeout"),
            "Expected timeout error, got: {err}"
        );
    }
}
