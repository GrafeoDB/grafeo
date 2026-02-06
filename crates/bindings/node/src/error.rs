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
            | NodeGrafeoError::Transaction(_) => {
                napi::Error::new(Status::GenericFailure, err.to_string())
            }
        }
    }
}

impl From<grafeo_common::utils::error::Error> for NodeGrafeoError {
    fn from(err: grafeo_common::utils::error::Error) -> Self {
        match &err {
            grafeo_common::utils::error::Error::Query(_) => {
                NodeGrafeoError::Query(err.to_string())
            }
            grafeo_common::utils::error::Error::Transaction(_) => {
                NodeGrafeoError::Transaction(err.to_string())
            }
            _ => NodeGrafeoError::Database(err.to_string()),
        }
    }
}
