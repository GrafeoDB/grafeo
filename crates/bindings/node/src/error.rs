//! Converts Rust errors to JavaScript exceptions.
//!
//! Type errors and invalid arguments become `InvalidArg` status errors,
//! while database, query, and transaction errors become `GenericFailure`.
//!
//! An error from the database, and one of a transaction the binding ends
//! itself, also carries its GQLSTATUS, its Grafeo code and whether a retry
//! can succeed: napi builds a thrown or rejected error from its status,
//! message and `cause` only, so these travel as a `cause` whose message is
//! [`GQLSTATUS_MARKER`], the status, the code and `1` or `0`
//! (`GQLSTATUS 40000 GRAFEO-T001 1`). The package's entry module
//! (`grafeo.js`) moves them to `err.gqlStatus`, `err.errorCode` and
//! `err.isRetryable` and removes that cause.

use grafeo_common::utils::GqlStatus;
use grafeo_common::utils::error::ErrorCode;
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

    /// An error from the database, with its GQLSTATUS and Grafeo code.
    #[error("{error}")]
    WithStatus {
        error: Box<NodeGrafeoError>,
        gql_status: GqlStatus,
        code: ErrorCode,
    },
}

impl NodeGrafeoError {
    /// An error of a transaction that ended (or is busy), which the binding
    /// raises itself: an invalid transaction state (GQLSTATUS 25000,
    /// `GRAFEO-T004`).
    pub fn transaction_state(message: impl Into<String>) -> Self {
        Self::WithStatus {
            error: Box::new(Self::Transaction(message.into())),
            gql_status: GqlStatus::INVALID_TX_STATE,
            code: ErrorCode::TransactionInvalidState,
        }
    }
}

impl From<NodeGrafeoError> for napi::Error {
    fn from(err: NodeGrafeoError) -> Self {
        match err {
            NodeGrafeoError::WithStatus {
                error,
                gql_status,
                code,
            } => {
                let mut js = napi::Error::from(*error);
                js.set_cause(napi::Error::new(
                    Status::GenericFailure,
                    format!(
                        "{GQLSTATUS_MARKER}{} {} {}",
                        gql_status.as_str(),
                        code.as_str(),
                        u8::from(code.is_retryable())
                    ),
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
            code: err.error_code(),
        }
    }
}
