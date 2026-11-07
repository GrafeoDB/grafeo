//! Lazy, cursor-based query result streams.
//!
//! Today `Session::execute` drains the entire operator pipeline into a
//! `QueryResult { rows: Vec<Vec<Value>> }` before returning. For large result
//! sets this either exhausts memory or forces the caller to wait until the
//! final row has been produced before they can see the first one.
//!
//! `ResultStream` exposes the pipeline lazily: the consumer pulls one
//! `DataChunk` (up to 2048 rows) at a time from the root operator. Dropping
//! the stream releases the operator tree and decrements the owning session's
//! `active_streams` counter.
//!
//! # Stability: Experimental
//!
//! This module is new in 0.5.40. It may change or be removed in any release,
//! with a CHANGELOG entry: see <https://grafeo.dev/stability/>. Use it from
//! embedded callers that want first-row latency or bounded memory; use
//! `Session::execute` when you want a fully materialized result.

use grafeo_common::clock::Instant;
use grafeo_common::types::{LogicalType, Value};
use grafeo_common::utils::error::{Error, QueryError, Result};
use grafeo_core::execution::operators::Operator;
use grafeo_core::execution::{DataChunk, deadline};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

use std::sync::Arc;

use grafeo_common::utils::warning::{QueryWarning, WarningSink};

use crate::database::QueryResult;
use crate::transaction::ReadPin;

/// What a stream runs: the pipeline `Session::build_streaming_plan` built, with its columns,
/// its deadline, its snapshot pin and its warnings.
pub(crate) struct StreamPlan {
    pub(crate) operator: Box<dyn Operator>,
    pub(crate) columns: Vec<String>,
    pub(crate) deadline: Option<Instant>,
    pub(crate) pin: Option<ReadPin>,
    /// Where the statement raises its warnings: its translation's, there when the stream
    /// opens, and its run's, as the rows that raise them are pulled.
    pub(crate) warnings: Arc<WarningSink>,
}

/// RAII guard that increments/decrements a session's active-stream counter.
///
/// The counter prevents `commit()` / `rollback()` from racing with in-flight
/// streams that still hold references to the session's read snapshot.
pub(crate) struct StreamGuard<'s> {
    counter: &'s AtomicUsize,
}

impl<'s> StreamGuard<'s> {
    pub(crate) fn new(counter: &'s AtomicUsize) -> Self {
        counter.fetch_add(1, Ordering::AcqRel);
        Self { counter }
    }
}

impl Drop for StreamGuard<'_> {
    fn drop(&mut self) {
        self.counter.fetch_sub(1, Ordering::AcqRel);
    }
}

/// [`StreamGuard`] for a stream that outlives the borrow of its session (an
/// [`OwnedResultStream`]): it shares the session's counter, so a commit or a
/// rollback of the session's transaction waits for this stream too.
pub(crate) struct OwnedStreamGuard {
    counter: Arc<AtomicUsize>,
    /// Set when the session ended (dropped) and rolled its transaction back
    /// under the stream: the stream reads nothing more.
    session_ended: Arc<AtomicBool>,
}

impl OwnedStreamGuard {
    pub(crate) fn new(counter: &Arc<AtomicUsize>, session_ended: &Arc<AtomicBool>) -> Self {
        counter.fetch_add(1, Ordering::AcqRel);
        Self {
            counter: Arc::clone(counter),
            session_ended: Arc::clone(session_ended),
        }
    }

    /// Whether the session that opened the stream ended.
    fn session_ended(&self) -> bool {
        self.session_ended.load(Ordering::Acquire)
    }
}

impl Drop for OwnedStreamGuard {
    fn drop(&mut self) {
        self.counter.fetch_sub(1, Ordering::AcqRel);
    }
}

/// Lazy, chunk-based result stream bound to a session's lifetime.
///
/// Created by [`Session::execute_streaming`](crate::Session::execute_streaming).
/// Iterate via [`next_chunk`](Self::next_chunk) for chunk granularity or
/// [`into_row_iter`](Self::into_row_iter) for a row iterator.
///
/// # Stability: Experimental
pub struct ResultStream<'session> {
    operator: Box<dyn Operator>,
    columns: Vec<String>,
    column_types: Vec<LogicalType>,
    deadline: Option<Instant>,
    exhausted: bool,
    _guard: StreamGuard<'session>,
    /// The stream's committed snapshot pin, including a historical override.
    _pin: Option<ReadPin>,
    warnings: Arc<WarningSink>,
}

impl<'s> ResultStream<'s> {
    /// Opens a streaming result.
    ///
    /// # Errors
    ///
    /// Returns [`QueryErrorKind::Semantic`](grafeo_common::utils::error::QueryErrorKind)
    /// if `columns` contains a repeated column name.
    pub(crate) fn new(plan: StreamPlan, guard: StreamGuard<'s>) -> Result<Self> {
        QueryResult::validate_unique_columns(&plan.columns)?;
        let len = plan.columns.len();
        Ok(Self {
            operator: plan.operator,
            columns: plan.columns,
            column_types: vec![LogicalType::Any; len],
            deadline: plan.deadline,
            exhausted: false,
            _guard: guard,
            _pin: plan.pin,
            warnings: plan.warnings,
        })
    }

    /// The warnings the statement raised so far, each once, ordered by code:
    /// its translation's from the start, its run's as the rows that raise
    /// them are pulled, all of them once the stream is exhausted. After an
    /// error it keeps what was raised before it.
    #[must_use]
    pub fn warnings(&self) -> Vec<QueryWarning> {
        self.warnings.raised().to_query_warnings()
    }
    /// Where the statement raises its warnings, for a caller that reads them
    /// while another thread pulls rows: reading it never waits for the
    /// stream.
    #[must_use]
    pub fn warning_sink(&self) -> Arc<WarningSink> {
        Arc::clone(&self.warnings)
    }

    /// Column names in the order they appear in each row.
    #[must_use]
    pub fn columns(&self) -> &[String] {
        &self.columns
    }

    /// Column types. Initially `Any`; refined after the first non-empty chunk.
    #[must_use]
    pub fn column_types(&self) -> &[LogicalType] {
        &self.column_types
    }

    /// Pulls the next chunk from the pipeline.
    ///
    /// Returns `Ok(None)` when the stream is exhausted.
    ///
    /// # Errors
    ///
    /// Propagates operator errors and returns `Error::from(QueryError::Timeout)`
    /// if the session's query deadline has passed.
    pub fn next_chunk(&mut self) -> Result<Option<DataChunk>> {
        if self.exhausted {
            return Ok(None);
        }
        check_deadline(self.deadline)?;
        // The operators check the deadline in their own loops too.
        let _deadline = deadline::enter(self.deadline);
        match self.operator.next() {
            Ok(Some(chunk)) => {
                refine_column_types(&chunk, &mut self.column_types);
                Ok(Some(chunk))
            }
            Ok(None) => {
                self.exhausted = true;
                Ok(None)
            }
            Err(err) => Err(super::convert_operator_error(err)),
        }
    }

    /// Converts to a row-level iterator that buffers one chunk internally.
    #[must_use]
    pub fn into_row_iter(self) -> RowIterator<'s> {
        RowIterator {
            stream: self,
            current: None,
            cursor: 0,
        }
    }

    /// Drains the stream into a fully materialized [`QueryResult`].
    ///
    /// Useful as an escape hatch when a caller requested streaming but then
    /// decides to collect everything (e.g., `stream.collect()?` in tests).
    ///
    /// # Errors
    ///
    /// Propagates operator errors and deadline timeouts.
    pub fn collect(mut self) -> Result<QueryResult> {
        let mut result = QueryResult::with_types(self.columns.clone(), self.column_types.clone())?;
        while let Some(chunk) = self.next_chunk()? {
            append_chunk(&chunk, &mut result);
        }
        result.column_types = self.column_types;
        result.warnings = self.warnings.raised().to_query_warnings();
        Ok(result)
    }
}

/// Row-level iterator adapter over a [`ResultStream`].
///
/// # Stability: Experimental
pub struct RowIterator<'s> {
    stream: ResultStream<'s>,
    // Indices are materialized once per chunk; re-collecting on every `next`
    // would turn per-chunk iteration from O(n) into O(n^2).
    current: Option<(DataChunk, Vec<usize>)>,
    cursor: usize,
}

impl RowIterator<'_> {
    /// Column names from the source stream.
    #[must_use]
    pub fn columns(&self) -> &[String] {
        self.stream.columns()
    }

    /// The warnings the statement raised so far (see [`ResultStream::warnings`]).
    #[must_use]
    pub fn warnings(&self) -> Vec<QueryWarning> {
        self.stream.warnings()
    }

    /// Where the statement raises its warnings (see
    /// [`ResultStream::warning_sink`]).
    #[must_use]
    pub fn warning_sink(&self) -> Arc<WarningSink> {
        self.stream.warning_sink()
    }
}

impl Iterator for RowIterator<'_> {
    type Item = Result<Vec<Value>>;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            if let Some((chunk, indices)) = &self.current {
                if self.cursor < indices.len() {
                    let row_idx = indices[self.cursor];
                    self.cursor += 1;
                    return Some(Ok(extract_row(chunk, row_idx)));
                }
                self.current = None;
                self.cursor = 0;
            }
            match self.stream.next_chunk() {
                Ok(Some(chunk)) => {
                    if chunk.row_count() == 0 {
                        continue;
                    }
                    let indices: Vec<usize> = chunk.selected_indices().collect();
                    self.current = Some((chunk, indices));
                    self.cursor = 0;
                }
                Ok(None) => return None,
                Err(err) => return Some(Err(err)),
            }
        }
    }
}

/// Binding-friendly result stream with no lifetime parameter.
///
/// Used by language bindings (Python, Node.js, WASM) where Rust lifetimes
/// cannot be expressed at the FFI boundary. The operator tree is `'static`
/// because operators hold `Arc<dyn GraphStore>` rather than borrows; the
/// stores remain alive as long as the stream does.
///
/// Callers that need to tie the stream's lifetime to something else (e.g.
/// a wrapping `Arc<RwLock<GrafeoDB>>` in a binding) should carry that
/// keepalive in their own wrapper alongside the stream.
///
/// # Stability: Experimental
pub struct OwnedResultStream {
    operator: Box<dyn Operator>,
    columns: Vec<String>,
    column_types: Vec<LogicalType>,
    deadline: Option<Instant>,
    exhausted: bool,
    /// The stream's committed snapshot pin, including a historical override.
    _pin: Option<ReadPin>,
    warnings: Arc<WarningSink>,
    /// The session's active-stream count while the stream is open, when a
    /// session's transaction may be reading through it: let go of when the
    /// stream is read to the end or dropped.
    _guard: Option<OwnedStreamGuard>,
}

impl std::fmt::Debug for OwnedResultStream {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OwnedResultStream")
            .field("columns", &self.columns)
            .field("column_types", &self.column_types)
            .field("deadline", &self.deadline)
            .field("exhausted", &self.exhausted)
            .finish_non_exhaustive()
    }
}

impl OwnedResultStream {
    /// Opens an owned streaming result, failing closed if the schema is not
    /// unique (see [`ResultStream::new`]). `guard` counts the stream among its
    /// session's active streams, `None` for a session nothing else uses.
    ///
    /// # Errors
    ///
    /// Returns [`QueryErrorKind::Semantic`](grafeo_common::utils::error::QueryErrorKind)
    /// if `columns` contains a repeated column name.
    pub(crate) fn new(plan: StreamPlan, guard: Option<OwnedStreamGuard>) -> Result<Self> {
        QueryResult::validate_unique_columns(&plan.columns)?;
        let len = plan.columns.len();
        Ok(Self {
            operator: plan.operator,
            columns: plan.columns,
            column_types: vec![LogicalType::Any; len],
            deadline: plan.deadline,
            exhausted: false,
            _pin: plan.pin,
            warnings: plan.warnings,
            _guard: guard,
        })
    }

    /// The warnings the statement raised so far (see [`ResultStream::warnings`]).
    #[must_use]
    pub fn warnings(&self) -> Vec<QueryWarning> {
        self.warnings.raised().to_query_warnings()
    }
    /// Where the statement raises its warnings, for a caller that reads them
    /// while another thread pulls rows: reading it never waits for the
    /// stream.
    #[must_use]
    pub fn warning_sink(&self) -> Arc<WarningSink> {
        Arc::clone(&self.warnings)
    }

    /// Column names in the order they appear in each row.
    #[must_use]
    pub fn columns(&self) -> &[String] {
        &self.columns
    }

    /// Column types. Initially `Any`; refined after the first non-empty chunk.
    #[must_use]
    pub fn column_types(&self) -> &[LogicalType] {
        &self.column_types
    }

    /// Pulls the next chunk. See [`ResultStream::next_chunk`].
    ///
    /// # Errors
    ///
    /// Propagates operator errors and deadline timeouts.
    pub fn next_chunk(&mut self) -> Result<Option<DataChunk>> {
        if self.exhausted {
            return Ok(None);
        }
        if self
            ._guard
            .as_ref()
            .is_some_and(OwnedStreamGuard::session_ended)
        {
            self.exhausted = true;
            self._guard = None;
            return Err(Error::Transaction(
                grafeo_common::utils::error::TransactionError::InvalidState(
                    "the session that opened this stream ended, and its transaction was rolled \
                     back: the stream reads no more"
                        .to_string(),
                ),
            ));
        }
        check_deadline(self.deadline)?;
        // The operators check the deadline in their own loops too.
        let _deadline = deadline::enter(self.deadline);
        match self.operator.next() {
            Ok(Some(chunk)) => {
                refine_column_types(&chunk, &mut self.column_types);
                Ok(Some(chunk))
            }
            Ok(None) => {
                self.exhausted = true;
                // Read to the end: it reads the transaction no more.
                self._guard = None;
                Ok(None)
            }
            Err(err) => Err(super::convert_operator_error(err)),
        }
    }

    /// Converts to a row iterator that buffers one chunk internally.
    #[must_use]
    pub fn into_row_iter(self) -> OwnedRowIterator {
        OwnedRowIterator {
            stream: self,
            current: None,
            cursor: 0,
        }
    }

    /// Drains into a [`QueryResult`].
    ///
    /// # Errors
    ///
    /// Propagates operator errors and deadline timeouts.
    pub fn collect(mut self) -> Result<QueryResult> {
        let mut result = QueryResult::with_types(self.columns.clone(), self.column_types.clone())?;
        while let Some(chunk) = self.next_chunk()? {
            append_chunk(&chunk, &mut result);
        }
        result.column_types = self.column_types;
        result.warnings = self.warnings.raised().to_query_warnings();
        Ok(result)
    }
}

/// Row-level iterator over an [`OwnedResultStream`].
///
/// # Stability: Experimental
pub struct OwnedRowIterator {
    stream: OwnedResultStream,
    current: Option<(DataChunk, Vec<usize>)>,
    cursor: usize,
}

impl OwnedRowIterator {
    /// Column names from the source stream.
    #[must_use]
    pub fn columns(&self) -> &[String] {
        self.stream.columns()
    }

    /// The warnings the statement raised so far (see [`ResultStream::warnings`]).
    #[must_use]
    pub fn warnings(&self) -> Vec<QueryWarning> {
        self.stream.warnings()
    }

    /// Where the statement raises its warnings (see
    /// [`ResultStream::warning_sink`]).
    #[must_use]
    pub fn warning_sink(&self) -> Arc<WarningSink> {
        self.stream.warning_sink()
    }
}

impl Iterator for OwnedRowIterator {
    type Item = Result<Vec<Value>>;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            if let Some((chunk, indices)) = &self.current {
                if self.cursor < indices.len() {
                    let row_idx = indices[self.cursor];
                    self.cursor += 1;
                    return Some(Ok(extract_row(chunk, row_idx)));
                }
                self.current = None;
                self.cursor = 0;
            }
            match self.stream.next_chunk() {
                Ok(Some(chunk)) => {
                    if chunk.row_count() == 0 {
                        continue;
                    }
                    let indices: Vec<usize> = chunk.selected_indices().collect();
                    self.current = Some((chunk, indices));
                    self.cursor = 0;
                }
                Ok(None) => return None,
                Err(err) => return Some(Err(err)),
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

fn check_deadline(deadline: Option<Instant>) -> Result<()> {
    if let Some(d) = deadline
        && Instant::now() >= d
    {
        return Err(Error::from(QueryError::timeout()));
    }
    Ok(())
}

fn refine_column_types(chunk: &DataChunk, types: &mut Vec<LogicalType>) {
    let col_count = chunk.column_count();
    if col_count == 0 {
        return;
    }
    if types.len() != col_count {
        types.resize(col_count, LogicalType::Any);
    }
    for (col_idx, slot) in types.iter_mut().enumerate().take(col_count) {
        if matches!(slot, LogicalType::Any)
            && let Some(col) = chunk.column(col_idx)
        {
            *slot = col.data_type().clone();
        }
    }
}

fn extract_row(chunk: &DataChunk, row_idx: usize) -> Vec<Value> {
    let col_count = chunk.column_count();
    let mut row = Vec::with_capacity(col_count);
    for col_idx in 0..col_count {
        let value = chunk
            .column(col_idx)
            .and_then(|col| col.get_value(row_idx))
            .unwrap_or(Value::Null);
        row.push(value);
    }
    row
}

fn append_chunk(chunk: &DataChunk, result: &mut QueryResult) {
    for row_idx in chunk.selected_indices() {
        result.rows.push(extract_row(chunk, row_idx));
    }
}
