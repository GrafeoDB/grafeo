//! Language-agnostic error classification for bindings.
//!
//! Each binding maps [`ErrorCategory`] to its language-specific exception type
//! (Python `PyErr`, Node.js `napi::Error`, C `GrafeoStatus`, etc.) using a
//! single small match expression.

use grafeo_common::utils::error::Error;

/// Categories that all bindings map errors into.
///
/// These mirror the natural groupings in [`grafeo_common::utils::error::Error`]
/// and match what every binding was already doing independently.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ErrorCategory {
    /// Query parsing, semantic, or execution error.
    Query,
    /// Transaction conflict, timeout, or invalid state.
    Transaction,
    /// Storage-layer error (disk, memory limit).
    Storage,
    /// I/O error (file, network).
    Io,
    /// Serialization/deserialization failure.
    Serialization,
    /// Internal error (should not happen in normal operation).
    Internal,
    /// Catch-all for other database errors (not found, type mismatch, etc.).
    Database,
}

/// Classifies a Grafeo error into a binding-agnostic category.
#[must_use]
pub fn classify_error(err: &Error) -> ErrorCategory {
    match err {
        Error::Query(_) => ErrorCategory::Query,
        Error::Transaction(_) => ErrorCategory::Transaction,
        Error::Storage(_) => ErrorCategory::Storage,
        Error::Io(_) => ErrorCategory::Io,
        Error::Serialization(_) => ErrorCategory::Serialization,
        Error::Internal(_) => ErrorCategory::Internal,
        _ => ErrorCategory::Database,
    }
}

/// Returns the human-readable message for a Grafeo error.
#[must_use]
pub fn error_message(err: &Error) -> String {
    err.to_string()
}

#[cfg(test)]
mod tests {
    use grafeo_common::utils::error::{
        Error, QueryError, QueryErrorKind, StorageError, TransactionError,
    };

    use super::*;

    #[test]
    fn classifies_query_error() {
        let err = Error::Query(QueryError::new(QueryErrorKind::Syntax, "bad syntax"));
        assert_eq!(classify_error(&err), ErrorCategory::Query);
    }

    #[test]
    fn classifies_not_found_as_database() {
        let err = Error::NodeNotFound(grafeo_common::types::NodeId(42));
        assert_eq!(classify_error(&err), ErrorCategory::Database);
    }

    #[test]
    fn classifies_internal() {
        let err = Error::Internal("oops".into());
        assert_eq!(classify_error(&err), ErrorCategory::Internal);
    }

    #[test]
    fn classifies_transaction_error() {
        let err = Error::Transaction(TransactionError::Conflict);
        assert_eq!(classify_error(&err), ErrorCategory::Transaction);
    }

    #[test]
    fn classifies_storage_error() {
        let err = Error::Storage(StorageError::Full);
        assert_eq!(classify_error(&err), ErrorCategory::Storage);
    }

    #[test]
    fn classifies_io_error() {
        let err = Error::Io(std::io::Error::new(
            std::io::ErrorKind::NotFound,
            "file not found",
        ));
        assert_eq!(classify_error(&err), ErrorCategory::Io);
    }

    #[test]
    fn classifies_serialization_error() {
        let err = Error::Serialization("bad bytes".into());
        assert_eq!(classify_error(&err), ErrorCategory::Serialization);
    }

    #[test]
    fn error_message_is_non_empty() {
        let err = Error::Internal("something broke".into());
        let msg = error_message(&err);
        assert!(!msg.is_empty(), "msg is empty");
        assert!(msg.contains("something broke"));
    }
}
