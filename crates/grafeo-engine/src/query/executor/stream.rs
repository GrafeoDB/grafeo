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
//! This module is new in 0.5.40. Signatures may change before being promoted
//! to Beta. Use from embedded callers that want first-row latency or bounded
//! memory; use `Session::execute` when you want a fully materialized result.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

use grafeo_common::types::{LogicalType, Value};
use grafeo_common::utils::error::{Error, QueryError, Result};
use grafeo_core::execution::DataChunk;
use grafeo_core::execution::operators::Operator;

use crate::database::QueryResult;

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
}

impl<'s> ResultStream<'s> {
    /// Opens a streaming result, failing closed if the schema is not unique.
    ///
    /// The streaming FFI surfaces (PyO3 `PyResultStream`, C `grafeo_stream_open`)
    /// build each row keyed by column name, so a duplicate name would silently
    /// drop data per row. Validating once here — at stream-open, before the first
    /// row — keeps the result-column uniqueness invariant on the streaming path
    /// as well as the eager path.
    ///
    /// # Errors
    ///
    /// Returns [`QueryErrorKind::Semantic`](grafeo_common::utils::error::QueryErrorKind)
    /// if `columns` contains a repeated column name.
    pub(crate) fn new(
        operator: Box<dyn Operator>,
        columns: Vec<String>,
        deadline: Option<Instant>,
        guard: StreamGuard<'s>,
    ) -> Result<Self> {
        QueryResult::validate_unique_columns(&columns)?;
        let len = columns.len();
        Ok(Self {
            operator,
            columns,
            column_types: vec![LogicalType::Any; len],
            deadline,
            exhausted: false,
            _guard: guard,
        })
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
    /// Propagates operator errors and returns `Error::Query(QueryError::Timeout)`
    /// if the session's query deadline has passed.
    pub fn next_chunk(&mut self) -> Result<Option<DataChunk>> {
        if self.exhausted {
            return Ok(None);
        }
        check_deadline(self.deadline)?;
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
/// because operators hold `Arc<dyn GraphStoreSearch>` rather than borrows; the
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
    /// unique (see [`ResultStream::new`]).
    ///
    /// # Errors
    ///
    /// Returns [`QueryErrorKind::Semantic`](grafeo_common::utils::error::QueryErrorKind)
    /// if `columns` contains a repeated column name.
    pub(crate) fn new(
        operator: Box<dyn Operator>,
        columns: Vec<String>,
        deadline: Option<Instant>,
    ) -> Result<Self> {
        QueryResult::validate_unique_columns(&columns)?;
        let len = columns.len();
        Ok(Self {
            operator,
            columns,
            column_types: vec![LogicalType::Any; len],
            deadline,
            exhausted: false,
        })
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
        check_deadline(self.deadline)?;
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
    #[cfg(not(target_arch = "wasm32"))]
    if let Some(d) = deadline
        && Instant::now() >= d
    {
        return Err(Error::Query(QueryError::timeout()));
    }
    #[cfg(target_arch = "wasm32")]
    let _ = deadline;
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
