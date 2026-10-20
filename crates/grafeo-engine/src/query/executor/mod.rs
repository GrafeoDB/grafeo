//! Query executor.
//!
//! Executes physical plans and produces results.

#[cfg(feature = "algos")]
pub mod procedure_call;
#[cfg(all(feature = "gql", feature = "lpg"))]
pub mod stream;
#[cfg(all(feature = "algos", feature = "gql"))]
pub mod user_procedure;

use std::sync::Arc;
use std::time::{Duration, Instant};

use crate::database::QueryResult;
use grafeo_common::grafeo_debug_span;
use grafeo_common::types::{LogicalType, Value};
use grafeo_common::utils::error::{Error, QueryError, Result};
use grafeo_core::execution::operators::{Operator, OperatorError, ReadCounter, WriteCounter};
use grafeo_core::execution::{DataChunk, Pipeline};

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
    /// Counts the statement's writes, for the result's counters.
    write_counter: Option<Arc<WriteCounter>>,
    /// Counts the rows the statement reads, for the result's `rows_scanned`.
    read_counter: Option<Arc<ReadCounter>>,
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
            write_counter: None,
            read_counter: None,
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
            write_counter: None,
            read_counter: None,
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
            write_counter: None,
            read_counter: None,
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

    /// Reports the writes `counter` counts in the result's counters.
    #[must_use]
    pub fn with_write_counter(mut self, counter: Arc<WriteCounter>) -> Self {
        self.write_counter = Some(counter);
        self
    }

    /// Reports the rows `counter` counts as the result's `rows_scanned`.
    #[must_use]
    pub fn with_read_counter(mut self, counter: Arc<ReadCounter>) -> Self {
        self.read_counter = Some(counter);
        self
    }

    /// The result, with the counters of the writes it made and the rows it
    /// read.
    fn with_counters(&self, mut result: QueryResult) -> QueryResult {
        if let Some(counter) = &self.write_counter {
            result.counters = counter.counters();
        }
        if let Some(counter) = &self.read_counter {
            result.rows_scanned = Some(counter.get());
        }
        result
    }

    /// Checks whether the deadline has been exceeded.
    fn check_deadline(&self) -> Result<()> {
        #[cfg(not(target_arch = "wasm32"))]
        if let Some(deadline) = self.deadline
            && Instant::now() >= deadline
        {
            return Err(self.timeout_error());
        }
        Ok(())
    }

    /// The error of a query that ran past its deadline, naming the limit
    /// when the executor knows it.
    fn timeout_error(&self) -> Error {
        Error::Query(match self.query_timeout {
            Some(d) => QueryError::timeout_with_limit(d),
            None => QueryError::timeout(),
        })
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
                    #[cfg(feature = "spill")]
                    self.collect_accounted_chunk(
                        chunk,
                        &mut result,
                        &mut types_captured,
                        usize::MAX,
                    )?;
                    #[cfg(not(feature = "spill"))]
                    {
                        // Capture column types from the first chunk with columns.
                        if !types_captured && chunk.column_count() > 0 {
                            self.capture_column_types(&chunk, &mut result);
                            types_captured = true;
                        }
                        self.collect_chunk(&chunk, &mut result)?;
                    }
                }
                Ok(None) => break,
                Err(err) => return Err(convert_operator_error(err)),
            }
        }

        Ok(self.with_counters(result))
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

        // Reject a duplicate result schema before running: the pipeline may
        // contain mutations, which must not happen for a query that errors.
        let mut result = QueryResult::with_types(self.columns.clone(), self.column_types.clone())?;

        let source = Box::new(OperatorSource::new(source));
        let collector = ChunkCollector::new();

        // Build and execute the pipeline with deadline enforcement
        let mut pipeline = Pipeline::new(source, push_ops, Box::new(collector));
        pipeline.set_deadline(self.deadline);
        pipeline.execute().map_err(|error| match error {
            OperatorError::Timeout => self.timeout_error(),
            other => convert_operator_error(other),
        })?;

        // Extract the sink (ChunkCollector) and get the chunks
        // Safety: we know the sink is a ChunkCollector because we just created it
        let sink_box = pipeline.into_sink();
        let any_sink: Box<dyn std::any::Any> = sink_box.into_any();
        let collector = any_sink
            .downcast::<ChunkCollector>()
            .expect("sink should be ChunkCollector");
        #[cfg(feature = "spill")]
        let (chunks, collector_owner) = collector.into_accounted_chunks();
        #[cfg(not(feature = "spill"))]
        let chunks = collector.into_chunks();

        let mut types_captured = !result.column_types.iter().all(|t| *t == LogicalType::Any);

        #[cfg(feature = "spill")]
        {
            // Keep the iterator before its owner in one tuple: on success or
            // error, remaining chunks and the collector Vec allocation drop
            // before that container's grant is released.
            let mut chunks = (chunks.into_iter(), collector_owner);
            for chunk in chunks.0.by_ref() {
                self.check_deadline()?;
                self.collect_accounted_chunk(chunk, &mut result, &mut types_captured, usize::MAX)?;
            }
            drop(chunks);
        }
        #[cfg(not(feature = "spill"))]
        for chunk in chunks {
            self.check_deadline()?;
            if !types_captured && chunk.column_count() > 0 {
                self.capture_column_types(&chunk, &mut result);
                types_captured = true;
            }
            self.collect_chunk(&chunk, &mut result)?;
        }

        Ok(self.with_counters(result))
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
                    let remaining = limit - collected;
                    #[cfg(feature = "spill")]
                    {
                        collected += self.collect_accounted_chunk(
                            chunk,
                            &mut result,
                            &mut types_captured,
                            remaining,
                        )?;
                    }
                    #[cfg(not(feature = "spill"))]
                    {
                        // Capture column types from the first chunk with columns.
                        if !types_captured && chunk.column_count() > 0 {
                            self.capture_column_types(&chunk, &mut result);
                            types_captured = true;
                        }
                        collected += self.collect_chunk_limited(&chunk, &mut result, remaining)?;
                    }
                }
                Ok(None) => break,
                Err(err) => return Err(convert_operator_error(err)),
            }
        }

        Ok(self.with_counters(result))
    }

    /// Transfers selected rows and their admission, including any captured schema.
    #[cfg(feature = "spill")]
    fn collect_accounted_chunk(
        &self,
        mut chunk: DataChunk,
        result: &mut QueryResult,
        types_captured: &mut bool,
        limit: usize,
    ) -> Result<usize> {
        let count = chunk.row_count().min(limit);
        let capture_types = !*types_captured && chunk.column_count() > 0;
        if chunk.owner_mut().is_none() {
            // Legacy chunks keep the direct collection path without a second
            // outer row container or a new executor reservation.
            if capture_types {
                self.capture_column_types(&chunk, result);
                *types_captured = true;
            }
            return self.collect_chunk_limited(&chunk, result, limit);
        }
        let rows = chunk
            .try_into_accounted_rows_with_types_limited(capture_types, limit)
            .map_err(|error| match error {
                OperatorError::Timeout => self.timeout_error(),
                other => convert_operator_error(other),
            })?;
        result.extend_accounted(rows)?;
        *types_captured |= capture_types;
        Ok(count)
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
    #[cfg(not(feature = "spill"))]
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
}

impl Default for Executor {
    fn default() -> Self {
        Self::new()
    }
}

/// Converts an operator error to a common error.
///
/// A statement that fails while it runs is a query execution error
/// (`GRAFEO-Q006`); only a broken invariant ([`OperatorError::Internal`]) is
/// an internal error, and an error raised below the operator keeps its own.
pub(crate) fn convert_operator_error(err: OperatorError) -> Error {
    match err {
        OperatorError::TypeMismatch { expected, found } => Error::TypeMismatch { expected, found },
        // A column an operator expects and its input lacks: the planner
        // and the operator disagree, a bug.
        OperatorError::ColumnNotFound(name) => Error::Internal(format!("Column not found: {name}")),
        OperatorError::Execution(msg) => Error::Query(QueryError::new(
            grafeo_common::utils::error::QueryErrorKind::Execution,
            msg,
        )),
        OperatorError::Internal(msg) => Error::Internal(msg),
        OperatorError::Wrapped(error) => *error,
        OperatorError::ConstraintViolation(msg) => {
            Error::InvalidValue(format!("Constraint violation: {msg}"))
        }
        OperatorError::InvalidValue(msg) => Error::InvalidValue(msg),
        OperatorError::WriteConflict(msg) => {
            Error::Transaction(grafeo_common::utils::error::TransactionError::WriteConflict(msg))
        }
        OperatorError::LimitExceeded(msg) => Error::Query(QueryError::new(
            grafeo_common::utils::error::QueryErrorKind::Execution,
            msg,
        )),
        OperatorError::Timeout => Error::Query(QueryError::timeout()),
        _ => Error::Internal(format!("{err}")),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use grafeo_common::types::LogicalType;
    use grafeo_common::utils::error::ErrorCode;
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

    /// Source that counts how often it is pulled, so tests can prove a query
    /// was rejected before any operator (possibly a mutation) ran.
    struct CountingOperator {
        pulls: std::sync::Arc<std::sync::atomic::AtomicUsize>,
    }

    impl Operator for CountingOperator {
        fn next(&mut self) -> grafeo_core::execution::operators::OperatorResult {
            self.pulls.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            Ok(None)
        }

        fn reset(&mut self) {}

        fn name(&self) -> &'static str {
            "Counting"
        }

        fn into_any(self: Box<Self>) -> Box<dyn std::any::Any + Send> {
            self
        }
    }

    #[test]
    fn test_duplicate_columns_rejected_before_any_operator_runs() {
        use std::sync::Arc;
        use std::sync::atomic::{AtomicUsize, Ordering};

        let duplicate = vec!["a".to_string(), "a".to_string()];
        let executor = Executor::with_columns(duplicate);

        let pulls = Arc::new(AtomicUsize::new(0));
        let source = Box::new(CountingOperator {
            pulls: Arc::clone(&pulls),
        });
        let err = executor.execute_pipeline(source, vec![]).unwrap_err();
        assert!(
            err.to_string().contains("duplicate column name 'a'"),
            "{err}"
        );
        assert_eq!(pulls.load(Ordering::SeqCst), 0, "pipeline must not run");

        let mut op = CountingOperator {
            pulls: Arc::clone(&pulls),
        };
        assert!(executor.execute(&mut op).is_err());
        assert!(executor.execute_with_limit(&mut op, 10).is_err());
        assert_eq!(pulls.load(Ordering::SeqCst), 0, "pull path must not run");
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
    #[cfg(feature = "spill")]
    fn accounted_executor_retains_nested_result_schema() {
        use grafeo_common::memory::buffer::{BufferManager, BufferManagerConfig};
        use grafeo_core::execution::memory::GroupedMemoryOwner;
        use std::sync::Arc;

        struct AdmittedChunk(Option<DataChunk>);
        impl Operator for AdmittedChunk {
            fn next(&mut self) -> grafeo_core::execution::operators::OperatorResult {
                Ok(self.0.take())
            }
            fn reset(&mut self) {}
            fn name(&self) -> &'static str {
                "AdmittedSchemaChunk"
            }
            fn into_any(self: Box<Self>) -> Box<dyn std::any::Any + Send> {
                self
            }
        }

        let manager = BufferManager::new(BufferManagerConfig {
            hard_limit_fraction: 1.0,
            ..BufferManagerConfig::with_budget(1024 * 1024)
        });
        let executor = Executor::with_columns(vec!["value".into()]);
        let execute = |schema: &LogicalType| {
            let owner = GroupedMemoryOwner::try_new(Arc::clone(&manager), 0).unwrap();
            let mut chunk =
                DataChunk::try_with_owner(std::slice::from_ref(schema), 1, owner).unwrap();
            // Null has no retained payload, so the two runs differ only in
            // schema heap, not in result values or row/container capacities.
            chunk.try_push_owned_value(0, Value::Null).unwrap();
            chunk.set_count(1);
            executor
                .execute_pipeline(Box::new(AdmittedChunk(Some(chunk))), vec![])
                .unwrap()
        };

        let scalar = execute(&LogicalType::Int64);
        assert_eq!(scalar.rows(), &[vec![Value::Null]]);
        let scalar_bytes = manager.allocated();
        assert!(scalar_bytes > 0);
        drop(scalar);
        assert_eq!(manager.allocated(), 0);

        let field_name = "f".repeat(8192);
        // Two Box<LogicalType> allocations, one Struct field array, and its
        // String. Cloning these nonempty Vec/String values uses their lengths.
        let schema_heap = 2 * std::mem::size_of::<LogicalType>()
            + std::mem::size_of::<(String, LogicalType)>()
            + field_name.len();
        let schema = LogicalType::List(Box::new(LogicalType::Struct(vec![(
            field_name,
            LogicalType::List(Box::new(LogicalType::Int64)),
        )])));
        let mut result = execute(&schema);
        assert_eq!(result.rows(), &[vec![Value::Null]]);
        assert_eq!(result.column_types, vec![schema]);
        assert_eq!(
            manager.allocated(),
            scalar_bytes + schema_heap,
            "capturing a nested schema must retain its additional heap admission"
        );
        // Exercise the real result lifetime after every row buffer is gone.
        // Reservation shrink on public mutation is not part of this contract;
        // the existing row grant and the schema grant stay until result drop.
        result.rows.clear();
        result.rows.shrink_to_fit();
        assert_eq!(manager.allocated(), scalar_bytes + schema_heap);
        assert_eq!(result.column_types.len(), 1);
        drop(result);
        assert_eq!(manager.allocated(), 0);
    }

    #[cfg(feature = "spill")]
    mod accounted_pull {
        use super::*;
        use grafeo_common::memory::buffer::{BufferManager, BufferManagerConfig};
        use grafeo_core::execution::memory::GroupedMemoryOwner;
        use grafeo_core::execution::selection::SelectionVector;
        use std::collections::VecDeque;

        struct Source {
            chunks: VecDeque<DataChunk>,
            manager: Arc<BufferManager>,
            failure: Option<OperatorError>,
            pulls: usize,
        }

        impl Operator for Source {
            fn next(&mut self) -> grafeo_core::execution::operators::OperatorResult {
                self.pulls += 1;
                if let Some(chunk) = self.chunks.pop_front() {
                    return Ok(Some(chunk));
                }
                if let Some(error) = self.failure.take() {
                    assert!(
                        self.manager.allocated() > 0,
                        "the first batch must still be admitted when the source fails"
                    );
                    return Err(error);
                }
                Ok(None)
            }

            fn reset(&mut self) {}
            fn name(&self) -> &'static str {
                "AdmittedPull"
            }
            fn into_any(self: Box<Self>) -> Box<dyn std::any::Any + Send> {
                self
            }
        }

        fn manager() -> Arc<BufferManager> {
            BufferManager::new(BufferManagerConfig {
                hard_limit_fraction: 1.0,
                ..BufferManagerConfig::with_budget(1024 * 1024)
            })
        }

        fn chunk(
            manager: &Arc<BufferManager>,
            schema: &LogicalType,
            values: &[Value],
        ) -> DataChunk {
            let owner = GroupedMemoryOwner::try_new(Arc::clone(manager), 0).unwrap();
            let mut chunk =
                DataChunk::try_with_owner(std::slice::from_ref(schema), values.len(), owner)
                    .unwrap();
            for value in values {
                chunk.try_push_owned_value(0, value.clone()).unwrap();
            }
            chunk.set_count(values.len());
            chunk
        }

        fn source(manager: &Arc<BufferManager>, chunks: Vec<DataChunk>) -> Source {
            Source {
                chunks: chunks.into(),
                manager: Arc::clone(manager),
                failure: None,
                pulls: 0,
            }
        }

        fn execute(source: &mut Source, limited: bool) -> Result<QueryResult> {
            let executor = Executor::with_columns(vec!["value".into()]);
            if limited {
                executor.execute_with_limit(source, usize::MAX)
            } else {
                executor.execute(source)
            }
        }

        #[test]
        fn retained_rows_keep_admission_until_drop_or_transfer() {
            for limited in [false, true] {
                for transfer in [false, true] {
                    let manager = manager();
                    let mut source = source(
                        &manager,
                        vec![chunk(&manager, &LogicalType::Int64, &[Value::Int64(7)])],
                    );
                    let result = execute(&mut source, limited).unwrap();
                    let retained = manager.allocated();
                    assert!(retained > 0, "the result owns its row admission");
                    assert_eq!(result.rows(), &[vec![Value::Int64(7)]]);
                    assert_eq!(result.iter().count(), 1);
                    assert_eq!(manager.allocated(), retained, "borrowing retains admission");
                    if transfer {
                        let rows = result.into_rows();
                        assert_eq!(manager.allocated(), 0);
                        assert_eq!(rows, vec![vec![Value::Int64(7)]]);
                    } else {
                        drop(result);
                        assert_eq!(manager.allocated(), 0);
                    }
                }
            }
        }

        #[test]
        fn empty_nested_schema_keeps_admission_until_result_drop() {
            for limited in [false, true] {
                let manager = manager();
                let field = "nested".repeat(1024);
                let schema = LogicalType::List(Box::new(LogicalType::Struct(vec![(
                    field.clone(),
                    LogicalType::Int64,
                )])));
                let mut source = source(&manager, vec![chunk(&manager, &schema, &[])]);
                let result = execute(&mut source, limited).unwrap();
                assert!(result.is_empty());
                assert_eq!(result.column_types, vec![schema]);
                assert!(
                    manager.allocated() >= field.len(),
                    "a schema-only result must retain its nested metadata admission"
                );
                drop(result);
                assert_eq!(manager.allocated(), 0);
            }
        }

        #[test]
        fn empty_then_nonempty_chunks_preserve_first_schema_and_values() {
            for limited in [false, true] {
                let manager = manager();
                let schema = LogicalType::List(Box::new(LogicalType::Int64));
                let mut source = source(
                    &manager,
                    vec![
                        chunk(&manager, &schema, &[]),
                        chunk(&manager, &schema, &[Value::Null]),
                    ],
                );
                let result = execute(&mut source, limited).unwrap();
                assert_eq!(result.column_types, vec![schema]);
                assert_eq!(result.rows(), &[vec![Value::Null]]);
                assert!(manager.allocated() > 0);
                drop(result);
                assert_eq!(manager.allocated(), 0);
            }
        }

        #[test]
        fn partial_limit_preserves_selection_and_stops_pulling() {
            let manager = manager();
            let mut input = chunk(
                &manager,
                &LogicalType::Int64,
                &[10, 20, 30, 40, 50].map(Value::Int64),
            );
            input.set_selection(SelectionVector::from_predicate(5, |row| {
                matches!(row, 1 | 3 | 4)
            }));
            let mut source = source(&manager, vec![input]);
            source.failure = Some(OperatorError::Execution("must not pull again".into()));
            let result = Executor::with_columns(vec!["value".into()])
                .execute_with_limit(&mut source, 2)
                .unwrap();
            assert_eq!(source.pulls, 1);
            assert_eq!(result.column_types, vec![LogicalType::Int64]);
            assert_eq!(
                result.rows(),
                &[vec![Value::Int64(20)], vec![Value::Int64(40)]]
            );
            assert!(manager.allocated() > 0);
            drop(result);
            assert_eq!(manager.allocated(), 0);
        }

        #[test]
        fn zero_limit_leaves_unpulled_input_with_its_caller() {
            let manager = manager();
            let mut source = source(
                &manager,
                vec![chunk(&manager, &LogicalType::Int64, &[Value::Int64(7)])],
            );
            let input_bytes = manager.allocated();
            let result = Executor::with_columns(vec!["value".into()])
                .execute_with_limit(&mut source, 0)
                .unwrap();
            assert!(result.is_empty());
            assert_eq!(source.pulls, 0);
            assert_eq!(manager.allocated(), input_bytes);
            drop(result);
            assert_eq!(manager.allocated(), input_bytes);
            drop(source);
            assert_eq!(manager.allocated(), 0);
        }

        #[test]
        fn later_source_failure_releases_already_collected_rows() {
            for limited in [false, true] {
                for timeout in [false, true] {
                    let manager = manager();
                    let mut source = source(
                        &manager,
                        vec![chunk(&manager, &LogicalType::Int64, &[Value::Int64(7)])],
                    );
                    source.failure = Some(if timeout {
                        OperatorError::Timeout
                    } else {
                        OperatorError::Execution("failure after admitted output".into())
                    });
                    let error = execute(&mut source, limited).unwrap_err();
                    assert_eq!(
                        error.error_code(),
                        if timeout {
                            ErrorCode::QueryTimeout
                        } else {
                            ErrorCode::QueryExecution
                        }
                    );
                    assert_eq!(source.pulls, 2);
                    assert_eq!(
                        manager.allocated(),
                        0,
                        "the failed result releases its rows"
                    );
                }
            }
        }
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
        assert!(matches!(err, Error::Internal(_)));
        assert!(err.to_string().contains("col_x"));

        let err = convert_operator_error(OperatorError::Execution("Alix left".to_string()));
        assert_eq!(err.error_code(), ErrorCode::QueryExecution, "{err}");

        let err = convert_operator_error(OperatorError::Internal("internal issue".to_string()));
        assert!(matches!(err, Error::Internal(_)));

        let err = convert_operator_error(OperatorError::from(Error::NodeNotFound(
            grafeo_common::types::NodeId::new(3),
        )));
        assert!(
            matches!(err, Error::NodeNotFound(_)),
            "kept as it is: {err}"
        );

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
        // A timeout in a push pipeline is the timeout a pulled plan reports:
        // retryable, never an internal error.
        assert_eq!(err.error_code(), ErrorCode::QueryTimeout, "{err}");
        assert!(err.error_code().is_retryable());
    }
}
