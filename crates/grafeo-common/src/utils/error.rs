//! Error types for Grafeo operations.
//!
//! [`Error`] is the main error type you'll encounter. For query-specific errors,
//! [`QueryError`] includes source location and hints to help users fix issues.
//!
//! Every error carries a machine-readable [`ErrorCode`] (e.g. `GRAFEO-Q001`)
//! for programmatic handling across the ecosystem (core, server, web, bindings).

use std::fmt;

/// Machine-readable error code for programmatic error handling.
///
/// Error codes follow the pattern `GRAFEO-{category}{number}`:
/// - **Q**: Query errors (parse, semantic, timeout)
/// - **T**: Transaction errors (conflict, timeout, state)
/// - **S**: Storage errors (full, corruption)
/// - **V**: Validation errors (not found, type mismatch, invalid input)
/// - **X**: Internal errors (should not happen)
///
/// Clients can match on these codes without parsing error messages.
///
/// # Examples
///
/// ```
/// use grafeo_common::utils::error::{Error, ErrorCode};
///
/// let err = Error::Internal("something broke".into());
/// assert_eq!(err.error_code().as_str(), "GRAFEO-X001");
/// assert!(!err.error_code().is_retryable());
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum ErrorCode {
    // Query errors (Q)
    /// Query failed to parse.
    QuerySyntax,
    /// Query parsed but is invalid (unknown label, type mismatch, etc.).
    QuerySemantic,
    /// Query exceeded timeout.
    QueryTimeout,
    /// Feature not supported for this query language.
    QueryUnsupported,
    /// Query optimization failed.
    QueryOptimization,
    /// Query execution failed.
    QueryExecution,

    // Transaction errors (T)
    /// Write-write conflict (retry possible).
    TransactionConflict,
    /// Transaction exceeded TTL.
    TransactionTimeout,
    /// Transaction is read-only but attempted a write.
    TransactionReadOnly,
    /// Invalid transaction state.
    TransactionInvalidState,
    /// Serialization failure (SSI violation).
    TransactionSerialization,
    /// Deadlock detected.
    TransactionDeadlock,
    /// The database is closed: it takes no more writes.
    DatabaseClosed,
    /// A commit did not complete: nothing commits, checkpoints or saves until
    /// the database is reopened.
    IncompleteCommit,

    // Storage errors (S)
    /// Memory or disk limit reached.
    StorageFull,
    /// WAL or data corruption detected.
    StorageCorrupted,
    /// Recovery from WAL failed.
    StorageRecoveryFailed,

    // Validation errors (V)
    /// Request validation failed.
    InvalidInput,
    /// Node not found.
    NodeNotFound,
    /// Edge not found.
    EdgeNotFound,
    /// Property key not found.
    PropertyNotFound,
    /// Label not found.
    LabelNotFound,
    /// Type mismatch.
    TypeMismatch,

    // Internal errors (X)
    /// Unexpected internal error.
    Internal,
    /// Serialization/deserialization error.
    SerializationError,
    /// I/O error.
    IoError,
}

impl ErrorCode {
    /// Returns the string code (e.g. `"GRAFEO-Q001"`).
    #[must_use]
    pub const fn as_str(&self) -> &'static str {
        match self {
            Self::QuerySyntax => "GRAFEO-Q001",
            Self::QuerySemantic => "GRAFEO-Q002",
            Self::QueryTimeout => "GRAFEO-Q003",
            Self::QueryUnsupported => "GRAFEO-Q004",
            Self::QueryOptimization => "GRAFEO-Q005",
            Self::QueryExecution => "GRAFEO-Q006",

            Self::TransactionConflict => "GRAFEO-T001",
            Self::TransactionTimeout => "GRAFEO-T002",
            Self::TransactionReadOnly => "GRAFEO-T003",
            Self::TransactionInvalidState => "GRAFEO-T004",
            Self::TransactionSerialization => "GRAFEO-T005",
            Self::TransactionDeadlock => "GRAFEO-T006",
            Self::DatabaseClosed => "GRAFEO-T007",
            Self::IncompleteCommit => "GRAFEO-T008",

            Self::StorageFull => "GRAFEO-S001",
            Self::StorageCorrupted => "GRAFEO-S002",
            Self::StorageRecoveryFailed => "GRAFEO-S003",

            Self::InvalidInput => "GRAFEO-V001",
            Self::NodeNotFound => "GRAFEO-V002",
            Self::EdgeNotFound => "GRAFEO-V003",
            Self::PropertyNotFound => "GRAFEO-V004",
            Self::LabelNotFound => "GRAFEO-V005",
            Self::TypeMismatch => "GRAFEO-V006",

            Self::Internal => "GRAFEO-X001",
            Self::SerializationError => "GRAFEO-X002",
            Self::IoError => "GRAFEO-X003",
        }
    }

    /// Whether this error is retryable (client should retry the operation).
    #[must_use]
    pub const fn is_retryable(&self) -> bool {
        matches!(
            self,
            Self::TransactionConflict
                | Self::TransactionTimeout
                | Self::TransactionDeadlock
                | Self::QueryTimeout
        )
    }
}

impl fmt::Display for ErrorCode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// The main error type - covers everything that can go wrong in Grafeo.
///
/// Most methods return `Result<T, Error>`. Use pattern matching to handle
/// specific cases, or just propagate with `?`.
#[derive(Debug)]
#[non_exhaustive]
pub enum Error {
    /// A node was not found.
    NodeNotFound(crate::types::NodeId),

    /// An edge was not found.
    EdgeNotFound(crate::types::EdgeId),

    /// A property key was not found.
    PropertyNotFound(String),

    /// A label was not found.
    LabelNotFound(String),

    /// Type mismatch error.
    TypeMismatch {
        /// The expected type.
        expected: String,
        /// The actual type found.
        found: String,
    },

    /// Invalid value error.
    InvalidValue(String),

    /// Transaction error.
    Transaction(TransactionError),

    /// Storage error.
    Storage(StorageError),

    /// Query error.
    Query(QueryError),

    /// Serialization error.
    Serialization(String),

    /// I/O error.
    Io(std::io::Error),

    /// Internal error (should not happen in normal operation).
    Internal(String),
}

impl Error {
    /// Returns the machine-readable error code for this error.
    #[must_use]
    pub fn error_code(&self) -> ErrorCode {
        match self {
            Error::NodeNotFound(_) => ErrorCode::NodeNotFound,
            Error::EdgeNotFound(_) => ErrorCode::EdgeNotFound,
            Error::PropertyNotFound(_) => ErrorCode::PropertyNotFound,
            Error::LabelNotFound(_) => ErrorCode::LabelNotFound,
            Error::TypeMismatch { .. } => ErrorCode::TypeMismatch,
            Error::InvalidValue(_) => ErrorCode::InvalidInput,
            Error::Transaction(e) => e.error_code(),
            Error::Storage(e) => e.error_code(),
            Error::Query(e) => e.error_code(),
            Error::Serialization(_) => ErrorCode::SerializationError,
            Error::Io(_) => ErrorCode::IoError,
            Error::Internal(_) => ErrorCode::Internal,
        }
    }
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let code = self.error_code();
        match self {
            Error::NodeNotFound(id) => write!(f, "{code}: Node not found: {id}"),
            Error::EdgeNotFound(id) => write!(f, "{code}: Edge not found: {id}"),
            Error::PropertyNotFound(key) => write!(f, "{code}: Property not found: {key}"),
            Error::LabelNotFound(label) => write!(f, "{code}: Label not found: {label}"),
            Error::TypeMismatch { expected, found } => {
                write!(
                    f,
                    "{code}: Type mismatch: expected {expected}, found {found}"
                )
            }
            Error::InvalidValue(msg) => write!(f, "{code}: Invalid value: {msg}"),
            Error::Transaction(e) => write!(f, "{code}: {e}"),
            Error::Storage(e) => write!(f, "{code}: {e}"),
            Error::Query(e) => write!(f, "{e}"),
            Error::Serialization(msg) => write!(f, "{code}: Serialization error: {msg}"),
            Error::Io(e) => write!(f, "{code}: I/O error: {e}"),
            Error::Internal(msg) => write!(f, "{code}: Internal error: {msg}"),
        }
    }
}

impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Error::Io(e) => Some(e),
            Error::Transaction(e) => Some(e),
            Error::Storage(e) => Some(e),
            Error::Query(e) => Some(e),
            _ => None,
        }
    }
}

impl From<std::io::Error> for Error {
    fn from(e: std::io::Error) -> Self {
        Error::Io(e)
    }
}

/// Transaction-specific errors.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub enum TransactionError {
    /// Transaction was aborted.
    Aborted,

    /// Transaction commit failed due to conflict.
    Conflict,

    /// Write-write conflict with another transaction.
    WriteConflict(String),

    /// Serialization failure (SSI violation).
    ///
    /// Occurs when running at Serializable isolation level and a read-write
    /// conflict is detected (we read data that another committed transaction wrote).
    SerializationFailure(String),

    /// Deadlock detected.
    Deadlock,

    /// Transaction timed out.
    Timeout,

    /// Transaction is read-only but attempted a write.
    ReadOnly,

    /// Invalid transaction state.
    InvalidState(String),

    /// The database is closed: `close()` of a persistent database started,
    /// so it takes no more writes. Open it again to write.
    DatabaseClosed,

    /// An earlier commit did not complete (it panicked between its commit
    /// epoch and its completion), so the store may hold part of it: no
    /// transaction commits and nothing is checkpointed, saved or copied
    /// until the database is reopened. Reads still see every commit
    /// published before it.
    IncompleteCommit,
}

impl TransactionError {
    /// Returns the machine-readable error code for this transaction error.
    #[must_use]
    pub const fn error_code(&self) -> ErrorCode {
        match self {
            Self::Aborted | Self::Conflict | Self::WriteConflict(_) => {
                ErrorCode::TransactionConflict
            }
            Self::SerializationFailure(_) => ErrorCode::TransactionSerialization,
            Self::Deadlock => ErrorCode::TransactionDeadlock,
            Self::Timeout => ErrorCode::TransactionTimeout,
            Self::ReadOnly => ErrorCode::TransactionReadOnly,
            Self::InvalidState(_) => ErrorCode::TransactionInvalidState,
            Self::DatabaseClosed => ErrorCode::DatabaseClosed,
            Self::IncompleteCommit => ErrorCode::IncompleteCommit,
        }
    }
}

impl fmt::Display for TransactionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TransactionError::Aborted => write!(f, "Transaction aborted"),
            TransactionError::Conflict => write!(f, "Transaction conflict"),
            TransactionError::WriteConflict(msg) => write!(f, "Write conflict: {msg}"),
            TransactionError::SerializationFailure(msg) => {
                write!(f, "Serialization failure (SSI): {msg}")
            }
            TransactionError::Deadlock => write!(f, "Deadlock detected"),
            TransactionError::Timeout => write!(f, "Transaction timeout"),
            TransactionError::ReadOnly => write!(f, "Cannot write in read-only transaction"),
            TransactionError::InvalidState(msg) => write!(f, "Invalid transaction state: {msg}"),
            TransactionError::DatabaseClosed => write!(
                f,
                "the database is closed, so it takes no more writes: open it again to write"
            ),
            TransactionError::IncompleteCommit => write!(
                f,
                "an earlier commit did not complete, so no transaction can commit and nothing \
                 can be checkpointed, saved or copied: reopen the database (reads still see \
                 every commit published before it)"
            ),
        }
    }
}

impl std::error::Error for TransactionError {}

impl From<TransactionError> for Error {
    fn from(e: TransactionError) -> Self {
        Error::Transaction(e)
    }
}

/// Storage-specific errors.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub enum StorageError {
    /// Corruption detected in storage.
    Corruption(String),

    /// Storage is full.
    Full,

    /// Invalid WAL entry.
    InvalidWalEntry(String),

    /// Recovery failed.
    RecoveryFailed(String),

    /// Checkpoint failed.
    CheckpointFailed(String),
}

impl StorageError {
    /// Returns the machine-readable error code for this storage error.
    #[must_use]
    pub const fn error_code(&self) -> ErrorCode {
        match self {
            Self::Corruption(_) => ErrorCode::StorageCorrupted,
            Self::Full => ErrorCode::StorageFull,
            Self::InvalidWalEntry(_) | Self::CheckpointFailed(_) => ErrorCode::StorageCorrupted,
            Self::RecoveryFailed(_) => ErrorCode::StorageRecoveryFailed,
        }
    }
}

impl fmt::Display for StorageError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            StorageError::Corruption(msg) => write!(f, "Storage corruption: {msg}"),
            StorageError::Full => write!(f, "Storage is full"),
            StorageError::InvalidWalEntry(msg) => write!(f, "Invalid WAL entry: {msg}"),
            StorageError::RecoveryFailed(msg) => write!(f, "Recovery failed: {msg}"),
            StorageError::CheckpointFailed(msg) => write!(f, "Checkpoint failed: {msg}"),
        }
    }
}

impl std::error::Error for StorageError {}

impl From<StorageError> for Error {
    fn from(e: StorageError) -> Self {
        Error::Storage(e)
    }
}

/// A query error with source location and helpful hints.
///
/// When something goes wrong in a query (syntax error, unknown label, type
/// mismatch), you get one of these. The error message includes the location
/// in your query and often a suggestion for fixing it.
#[derive(Debug, Clone)]
pub struct QueryError {
    /// What category of error (lexer, syntax, semantic, etc.)
    pub kind: QueryErrorKind,
    /// Human-readable explanation of what went wrong.
    pub message: String,
    /// Where in the query the error occurred.
    pub span: Option<SourceSpan>,
    /// The original query text (for showing context).
    pub source_query: Option<String>,
    /// A suggestion for fixing the error.
    pub hint: Option<String>,
}

impl QueryError {
    /// Creates a new query error.
    pub fn new(kind: QueryErrorKind, message: impl Into<String>) -> Self {
        Self {
            kind,
            message: message.into(),
            span: None,
            source_query: None,
            hint: None,
        }
    }

    /// Creates a query timeout error.
    #[must_use]
    pub fn timeout() -> Self {
        Self::new(QueryErrorKind::Timeout, "Query exceeded timeout")
    }

    /// Creates a query timeout error that includes the configured limit.
    #[must_use]
    pub fn timeout_with_limit(limit: std::time::Duration) -> Self {
        let millis = limit.as_millis();
        let timeout_display = if millis >= 1000 && millis.is_multiple_of(1000) {
            format!("{}s", millis / 1000)
        } else {
            format!("{millis}ms")
        };
        Self::new(
            QueryErrorKind::Timeout,
            format!("Query exceeded the {timeout_display} timeout"),
        )
        .with_hint(
            "Increase with Config::with_query_timeout() or disable with Config::without_query_timeout()"
                .to_string(),
        )
    }

    /// Returns the machine-readable error code for this query error.
    #[must_use]
    pub const fn error_code(&self) -> ErrorCode {
        match self.kind {
            QueryErrorKind::Lexer | QueryErrorKind::Syntax => ErrorCode::QuerySyntax,
            QueryErrorKind::Semantic => ErrorCode::QuerySemantic,
            QueryErrorKind::Optimization => ErrorCode::QueryOptimization,
            QueryErrorKind::Execution => ErrorCode::QueryExecution,
            QueryErrorKind::Timeout => ErrorCode::QueryTimeout,
        }
    }

    /// Adds a source span to the error.
    #[must_use]
    pub fn with_span(mut self, span: SourceSpan) -> Self {
        self.span = Some(span);
        self
    }

    /// Adds the source query to the error.
    #[must_use]
    pub fn with_source(mut self, query: impl Into<String>) -> Self {
        self.source_query = Some(query.into());
        self
    }

    /// Adds a hint to the error.
    #[must_use]
    pub fn with_hint(mut self, hint: impl Into<String>) -> Self {
        self.hint = Some(hint.into());
        self
    }
}

impl fmt::Display for QueryError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}: {}", self.kind, self.message)?;

        if let (Some(span), Some(query)) = (&self.span, &self.source_query) {
            write!(f, "\n  --> query:{}:{}", span.line, span.column)?;

            // Extract and display the relevant line
            if let Some(line) = query.lines().nth(span.line.saturating_sub(1) as usize) {
                write!(f, "\n   |")?;
                write!(f, "\n {} | {}", span.line, line)?;
                write!(f, "\n   | ")?;

                // Add caret markers
                for _ in 0..span.column.saturating_sub(1) {
                    write!(f, " ")?;
                }
                for _ in span.start..span.end {
                    write!(f, "^")?;
                }
            }
        }

        if let Some(hint) = &self.hint {
            write!(f, "\n   |\n  help: {hint}")?;
        }

        Ok(())
    }
}

impl std::error::Error for QueryError {}

impl From<QueryError> for Error {
    fn from(e: QueryError) -> Self {
        Error::Query(e)
    }
}

/// The kind of query error.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum QueryErrorKind {
    /// Lexical error (invalid token).
    Lexer,
    /// Syntax error (parse failure).
    Syntax,
    /// Semantic error (type mismatch, unknown identifier, etc.).
    Semantic,
    /// Optimization error.
    Optimization,
    /// Execution error.
    Execution,
    /// Timeout error (query exceeded configured time limit).
    Timeout,
}

impl fmt::Display for QueryErrorKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            QueryErrorKind::Lexer => write!(f, "lexer error"),
            QueryErrorKind::Syntax => write!(f, "syntax error"),
            QueryErrorKind::Semantic => write!(f, "semantic error"),
            QueryErrorKind::Optimization => write!(f, "optimization error"),
            QueryErrorKind::Execution => write!(f, "execution error"),
            QueryErrorKind::Timeout => write!(f, "timeout error"),
        }
    }
}

/// A span in the source code.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SourceSpan {
    /// Byte offset of the start.
    pub start: usize,
    /// Byte offset of the end.
    pub end: usize,
    /// Line number (1-indexed).
    pub line: u32,
    /// Column number (1-indexed).
    pub column: u32,
}

impl SourceSpan {
    /// Creates a new source span.
    pub const fn new(start: usize, end: usize, line: u32, column: u32) -> Self {
        Self {
            start,
            end,
            line,
            column,
        }
    }
}

/// A type alias for `Result<T, Error>`.
pub type Result<T> = std::result::Result<T, Error>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_display() {
        let err = Error::NodeNotFound(crate::types::NodeId::new(42));
        assert_eq!(err.to_string(), "GRAFEO-V002: Node not found: 42");

        let err = Error::TypeMismatch {
            expected: "INT64".to_string(),
            found: "STRING".to_string(),
        };
        assert_eq!(
            err.to_string(),
            "GRAFEO-V006: Type mismatch: expected INT64, found STRING"
        );
    }

    #[test]
    fn database_closed_has_its_own_code_and_is_not_retryable() {
        let err = Error::Transaction(TransactionError::DatabaseClosed);
        assert_eq!(err.error_code(), ErrorCode::DatabaseClosed);
        assert_eq!(err.error_code().as_str(), "GRAFEO-T007");
        assert!(!err.error_code().is_retryable());
        assert_eq!(
            err.to_string(),
            "GRAFEO-T007: the database is closed, so it takes no more writes: open it again to write"
        );
    }

    #[test]
    fn an_incomplete_commit_has_its_own_code_and_is_not_retryable() {
        let err = Error::Transaction(TransactionError::IncompleteCommit);
        assert_eq!(err.error_code(), ErrorCode::IncompleteCommit);
        assert_eq!(err.error_code().as_str(), "GRAFEO-T008");
        assert!(
            !err.error_code().is_retryable(),
            "only a reopen helps, never a retry"
        );
        assert_eq!(
            err.to_string(),
            "GRAFEO-T008: an earlier commit did not complete, so no transaction can commit and \
             nothing can be checkpointed, saved or copied: reopen the database (reads still see \
             every commit published before it)"
        );
    }

    #[test]
    fn test_error_codes() {
        assert_eq!(
            Error::Internal("x".into()).error_code(),
            ErrorCode::Internal
        );
        assert_eq!(ErrorCode::Internal.as_str(), "GRAFEO-X001");
        assert!(!ErrorCode::Internal.is_retryable());

        assert_eq!(
            Error::Transaction(TransactionError::Conflict).error_code(),
            ErrorCode::TransactionConflict
        );
        assert!(ErrorCode::TransactionConflict.is_retryable());
        assert!(ErrorCode::QueryTimeout.is_retryable());
        assert!(!ErrorCode::StorageFull.is_retryable());
    }

    #[test]
    fn test_query_timeout() {
        let err = QueryError::timeout();
        assert_eq!(err.kind, QueryErrorKind::Timeout);
        assert_eq!(err.error_code(), ErrorCode::QueryTimeout);
        assert!(err.error_code().is_retryable());
        assert!(err.message.contains("timeout"));
    }

    #[test]
    fn test_query_error_formatting() {
        let query = "MATCH (n:Peron) RETURN n";
        let err = QueryError::new(QueryErrorKind::Semantic, "Unknown label 'Peron'")
            .with_span(SourceSpan::new(9, 14, 1, 10))
            .with_source(query)
            .with_hint("Did you mean 'Person'?");

        let msg = err.to_string();
        assert!(msg.contains("Unknown label 'Peron'"));
        assert!(msg.contains("query:1:10"));
        assert!(msg.contains("Did you mean 'Person'?"));
    }

    #[test]
    fn test_transaction_error() {
        let err: Error = TransactionError::Conflict.into();
        assert!(matches!(
            err,
            Error::Transaction(TransactionError::Conflict)
        ));
    }

    #[test]
    fn test_io_error_conversion() {
        let io_err = std::io::Error::new(std::io::ErrorKind::NotFound, "file not found");
        let err: Error = io_err.into();
        assert!(matches!(err, Error::Io(_)));
    }
}
