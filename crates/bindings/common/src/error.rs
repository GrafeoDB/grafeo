//! Language-agnostic error classification for bindings.
//!
//! Each binding maps [`ErrorCategory`] to its language-specific exception type
//! (Python `PyErr`, Node.js `napi::Error`, C `GrafeoStatus`, etc.) using a
//! single small match expression.

use grafeo_common::utils::error::{Error, TransactionError};

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
    /// The database is closed and takes no more writes.
    DatabaseClosed,
    /// Storage-layer error (disk, memory limit).
    Storage,
    /// I/O error (file, network).
    Io,
    /// Serialization/deserialization failure of outside input.
    Serialization,
    /// A file Grafeo wrote is damaged (error code `GRAFEO-S002`).
    Corruption,
    /// Internal error (should not happen in normal operation).
    Internal,
    /// Catch-all for other database errors (not found, type mismatch, a
    /// commit that did not complete, etc.).
    Database,
}

/// Classifies a Grafeo error into a binding-agnostic category.
#[must_use]
pub fn classify_error(err: &Error) -> ErrorCategory {
    match err {
        Error::Query(_) => ErrorCategory::Query,
        Error::Transaction(TransactionError::DatabaseClosed) => ErrorCategory::DatabaseClosed,
        // Only a reopen helps, never a retry of the transaction.
        Error::Transaction(TransactionError::IncompleteCommit) => ErrorCategory::Database,
        Error::Transaction(_) => ErrorCategory::Transaction,
        Error::Storage(_) => ErrorCategory::Storage,
        Error::Io(_) => ErrorCategory::Io,
        Error::Serialization(_) => ErrorCategory::Serialization,
        Error::Corruption(_) => ErrorCategory::Corruption,
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
    fn classifies_database_closed_apart_from_other_transaction_errors() {
        let err = Error::Transaction(TransactionError::DatabaseClosed);
        assert_eq!(classify_error(&err), ErrorCategory::DatabaseClosed);
        let err = Error::Transaction(TransactionError::InvalidState("x".into()));
        assert_eq!(classify_error(&err), ErrorCategory::Transaction);
    }

    /// A commit that did not complete is no transaction error a retry could
    /// fix: only a reopen helps, so it is a database error with its own code.
    #[test]
    fn classifies_an_incomplete_commit_as_a_database_error() {
        let err = Error::Transaction(TransactionError::IncompleteCommit);
        assert_eq!(classify_error(&err), ErrorCategory::Database);
        assert_eq!(err.error_code().as_str(), "GRAFEO-T008");
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
        let err = Error::corruption("chunk checksum mismatch");
        assert_eq!(classify_error(&err), ErrorCategory::Corruption);
    }

    #[test]
    fn error_message_is_non_empty() {
        let err = Error::Internal("something broke".into());
        let msg = error_message(&err);
        assert!(!msg.is_empty(), "msg is empty");
        assert!(msg.contains("something broke"));
    }
}
