//! Converts Rust errors to JavaScript exceptions.
//!
//! Type errors and invalid arguments become `InvalidArg` status errors,
//! while database, query, and transaction errors become `GenericFailure`.
//!
//! An error from the database also carries its GQLSTATUS: napi builds a
//! thrown or rejected error from its status, message and `cause` only, so
//! the status travels as a `cause` whose message is [`GQLSTATUS_MARKER`]
//! and the code. The package's entry module (`grafeo.js`) moves it to
//! `err.gqlStatus` and removes that cause.

use grafeo_common::utils::GqlStatus;
use napi::Status;
use thiserror::Error;

/// The start of the message of the `cause` that carries an error's
/// GQLSTATUS (`GQLSTATUS 22012`); `error-status.js` matches the same text.
pub const GQLSTATUS_MARKER: &str = "GQLSTATUS ";

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

    /// An error from the database, with its GQLSTATUS.
    #[error("{error}")]
    WithStatus {
        error: Box<NodeGrafeoError>,
        gql_status: GqlStatus,
    },
}

impl From<NodeGrafeoError> for napi::Error {
    fn from(err: NodeGrafeoError) -> Self {
        match err {
            NodeGrafeoError::WithStatus { error, gql_status } => {
                let mut js = napi::Error::from(*error);
                js.set_cause(napi::Error::new(
                    Status::GenericFailure,
                    format!("{GQLSTATUS_MARKER}{}", gql_status.as_str()),
                ));
                js
            }
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
        let error = match classify_error(&err) {
            ErrorCategory::Query => NodeGrafeoError::Query(msg),
            ErrorCategory::Transaction => NodeGrafeoError::Transaction(msg),
            ErrorCategory::DatabaseClosed => NodeGrafeoError::DatabaseClosed(msg),
            _ => NodeGrafeoError::Database(msg),
        };
        NodeGrafeoError::WithStatus {
            error: Box::new(error),
            gql_status: GqlStatus::from(&err),
        }
    }
}
