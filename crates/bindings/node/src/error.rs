//! Converts Rust errors to JavaScript exceptions.
//!
//! Type errors and invalid arguments become `InvalidArg` status errors,
//! while database, query, and transaction errors become `GenericFailure`.

use napi::Status;
use thiserror::Error;

/// Grafeo errors that translate to JavaScript Error instances.
#[derive(Error, Debug)]
pub enum NodeGrafeoError {
    #[error("Database error: {0}")]
    Database(String),

    #[error("Query error: {0}")]
    Query(String),

    #[error("Type error: {0}")]
    Type(String),

    #[error("Transaction error: {0}")]
    Transaction(String),

    #[error("Database closed: {0}")]
    DatabaseClosed(String),

    #[error("Invalid argument: {0}")]
    InvalidArgument(String),
}

impl From<NodeGrafeoError> for napi::Error {
    fn from(err: NodeGrafeoError) -> Self {
        match &err {
            NodeGrafeoError::InvalidArgument(_) | NodeGrafeoError::Type(_) => {
                napi::Error::new(Status::InvalidArg, err.to_string())
            }
            NodeGrafeoError::Database(_)
            | NodeGrafeoError::Query(_)
            | NodeGrafeoError::Transaction(_)
            | NodeGrafeoError::DatabaseClosed(_) => {
                napi::Error::new(Status::GenericFailure, err.to_string())
            }
        }
    }
}

impl From<grafeo_common::utils::error::Error> for NodeGrafeoError {
    fn from(err: grafeo_common::utils::error::Error) -> Self {
        use grafeo_bindings_common::error::{ErrorCategory, classify_error};
        let msg = err.to_string();
        match classify_error(&err) {
            ErrorCategory::Query => NodeGrafeoError::Query(msg),
            ErrorCategory::Transaction => NodeGrafeoError::Transaction(msg),
            ErrorCategory::DatabaseClosed => NodeGrafeoError::DatabaseClosed(msg),
            _ => NodeGrafeoError::Database(msg),
        }
    }
}
