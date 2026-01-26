//! Error types for Graphos.

use std::fmt;

/// The main error type for Graphos operations.
#[derive(Debug)]
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

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::NodeNotFound(id) => write!(f, "Node not found: {id}"),
            Error::EdgeNotFound(id) => write!(f, "Edge not found: {id}"),
            Error::PropertyNotFound(key) => write!(f, "Property not found: {key}"),
            Error::LabelNotFound(label) => write!(f, "Label not found: {label}"),
            Error::TypeMismatch { expected, found } => {
                write!(f, "Type mismatch: expected {expected}, found {found}")
            }
            Error::InvalidValue(msg) => write!(f, "Invalid value: {msg}"),
            Error::Transaction(e) => write!(f, "Transaction error: {e}"),
            Error::Storage(e) => write!(f, "Storage error: {e}"),
            Error::Query(e) => write!(f, "Query error: {e}"),
            Error::Serialization(msg) => write!(f, "Serialization error: {msg}"),
            Error::Io(e) => write!(f, "I/O error: {e}"),
            Error::Internal(msg) => write!(f, "Internal error: {msg}"),
        }
    }
}

impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Error::Io(e) => Some(e),
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
pub enum TransactionError {
    /// Transaction was aborted.
    Aborted,

    /// Transaction commit failed due to conflict.
    Conflict,

    /// Deadlock detected.
    Deadlock,

    /// Transaction timed out.
    Timeout,

    /// Transaction is read-only but attempted a write.
    ReadOnly,

    /// Invalid transaction state.
    InvalidState(String),
}

impl fmt::Display for TransactionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TransactionError::Aborted => write!(f, "Transaction aborted"),
            TransactionError::Conflict => write!(f, "Transaction conflict"),
            TransactionError::Deadlock => write!(f, "Deadlock detected"),
            TransactionError::Timeout => write!(f, "Transaction timeout"),
            TransactionError::ReadOnly => write!(f, "Cannot write in read-only transaction"),
            TransactionError::InvalidState(msg) => write!(f, "Invalid transaction state: {msg}"),
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

/// Query-specific errors.
#[derive(Debug, Clone)]
pub struct QueryError {
    /// The kind of query error.
    pub kind: QueryErrorKind,
    /// Human-readable error message.
    pub message: String,
    /// Source span in the original query (if applicable).
    pub span: Option<SourceSpan>,
    /// The original query text.
    pub source_query: Option<String>,
    /// A hint for fixing the error.
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
}

impl fmt::Display for QueryErrorKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            QueryErrorKind::Lexer => write!(f, "lexer error"),
            QueryErrorKind::Syntax => write!(f, "syntax error"),
            QueryErrorKind::Semantic => write!(f, "semantic error"),
            QueryErrorKind::Optimization => write!(f, "optimization error"),
            QueryErrorKind::Execution => write!(f, "execution error"),
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
        assert_eq!(err.to_string(), "Node not found: 42");

        let err = Error::TypeMismatch {
            expected: "INT64".to_string(),
            found: "STRING".to_string(),
        };
        assert_eq!(err.to_string(), "Type mismatch: expected INT64, found STRING");
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
        assert!(matches!(err, Error::Transaction(TransactionError::Conflict)));
    }

    #[test]
    fn test_io_error_conversion() {
        let io_err = std::io::Error::new(std::io::ErrorKind::NotFound, "file not found");
        let err: Error = io_err.into();
        assert!(matches!(err, Error::Io(_)));
    }
}
