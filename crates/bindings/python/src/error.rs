//! Converts Rust errors to Python exceptions.
//!
//! Type errors and invalid arguments become `ValueError`. Database, query,
//! and transaction errors become `GrafeoError` — a new subclass of
//! `RuntimeError` that carries the original `ErrorCode` and `is_retryable`
//! flag as attributes.
//!
//! `GrafeoError` inherits from `RuntimeError`, so existing code that catches
//! `RuntimeError` continues to work. New code can catch `GrafeoError`
//! specifically and inspect `e.error_code` / `e.is_retryable`. A write to a
//! closed database raises `DatabaseClosedError`, a damaged database file
//! `GrafeoCorruptionError`, and a database file another handle or process
//! holds `GrafeoLockedError`, all subclasses of `GrafeoError`.

use grafeo_common::utils::GqlStatus;
use grafeo_common::utils::error::ErrorCode;
use pyo3::create_exception;
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use thiserror::Error;

create_exception!(
    grafeo,
    GrafeoError,
    PyRuntimeError,
    "Grafeo-specific runtime error with machine-readable `error_code`, `gql_status` (the ISO GQL status, such as `22012`) and `is_retryable` attributes."
);

create_exception!(
    grafeo,
    DatabaseClosedError,
    GrafeoError,
    "Raised by a write to a database whose `close()` started (persistent databases only). Open it again to write."
);

create_exception!(
    grafeo,
    GrafeoCorruptionError,
    GrafeoError,
    "Raised when a file Grafeo wrote is damaged (error code GRAFEO-S002): a checksum, a header, a section or a WAL record that does not read back. The message names the file and, when known, the byte."
);

create_exception!(
    grafeo,
    GrafeoLockedError,
    GrafeoError,
    "Raised when a database file is open elsewhere (error code GRAFEO-S004, GQLSTATUS 08000): in another handle of this process, which must be closed first, or in another process. The message says which."
);

/// Grafeo errors that translate to Python exceptions.
#[derive(Error, Debug)]
pub enum PyGrafeoError {
    #[error("Database error: {message}")]
    Database {
        message: String,
        code: Option<ErrorCode>,
        gql_status: Option<GqlStatus>,
    },

    #[error("Query error: {message}")]
    Query {
        message: String,
        code: Option<ErrorCode>,
        gql_status: Option<GqlStatus>,
    },

    #[error("Type error: {0}")]
    Type(String),

    #[error("Transaction error: {message}")]
    Transaction {
        message: String,
        code: Option<ErrorCode>,
        gql_status: Option<GqlStatus>,
    },

    #[error("Database closed: {message}")]
    DatabaseClosed { message: String },

    #[error("Corrupt file: {message}")]
    Corruption { message: String },

    #[error("Database error: {message}")]
    Locked { message: String },

    #[error("Invalid argument: {0}")]
    InvalidArgument(String),
}

impl PyGrafeoError {
    /// Constructs a plain `Database` variant without a known error code.
    ///
    /// Used by call sites that produce strings (binding-internal errors
    /// with no upstream `Error` to classify).
    pub fn database<S: Into<String>>(message: S) -> Self {
        Self::Database {
            message: message.into(),
            code: None,
            gql_status: None,
        }
    }

    /// Constructs a plain `Query` variant without a known error code.
    pub fn query<S: Into<String>>(message: S) -> Self {
        Self::Query {
            message: message.into(),
            code: None,
            gql_status: None,
        }
    }

    /// Constructs a plain `Transaction` variant without a known error code.
    pub fn transaction<S: Into<String>>(message: S) -> Self {
        Self::Transaction {
            message: message.into(),
            code: None,
            gql_status: None,
        }
    }
}

impl From<PyGrafeoError> for PyErr {
    fn from(err: PyGrafeoError) -> Self {
        match err {
            PyGrafeoError::InvalidArgument(msg) | PyGrafeoError::Type(msg) => {
                PyValueError::new_err(msg)
            }
            PyGrafeoError::Database {
                message,
                code,
                gql_status,
            }
            | PyGrafeoError::Query {
                message,
                code,
                gql_status,
            }
            | PyGrafeoError::Transaction {
                message,
                code,
                gql_status,
            } => build_grafeo_error(message, code, gql_status),
            PyGrafeoError::DatabaseClosed { message } => Python::attach(|py| {
                let err = DatabaseClosedError::new_err(message);
                set_code(
                    &err,
                    py,
                    ErrorCode::DatabaseClosed,
                    GqlStatus::INVALID_TX_STATE,
                );
                err
            }),
            PyGrafeoError::Corruption { message } => Python::attach(|py| {
                let err = GrafeoCorruptionError::new_err(message);
                set_code(
                    &err,
                    py,
                    ErrorCode::StorageCorrupted,
                    GqlStatus::DATA_EXCEPTION,
                );
                err
            }),
            PyGrafeoError::Locked { message } => Python::attach(|py| {
                let err = GrafeoLockedError::new_err(message);
                set_code(
                    &err,
                    py,
                    ErrorCode::DatabaseLocked,
                    GqlStatus::CONNECTION_EXCEPTION,
                );
                err
            }),
        }
    }
}

/// Builds a `GrafeoError` with `error_code`, `gql_status` and `is_retryable`
/// attributes always populated. If the upstream error did not carry a
/// specific code, the generic `Internal` code and its status `22000` are used
/// (as the database reports an internal error), so that callers can rely on
/// the attributes being present on every `GrafeoError`.
fn build_grafeo_error(
    message: String,
    code: Option<ErrorCode>,
    gql_status: Option<GqlStatus>,
) -> PyErr {
    Python::attach(|py| {
        let err = GrafeoError::new_err(message);
        set_code(
            &err,
            py,
            code.unwrap_or(ErrorCode::Internal),
            gql_status.unwrap_or(GqlStatus::DATA_EXCEPTION),
        );
        err
    })
}

/// Sets the `error_code`, `gql_status` and `is_retryable` attributes of a
/// `GrafeoError`.
fn set_code(err: &PyErr, py: Python<'_>, code: ErrorCode, gql_status: GqlStatus) {
    let inst = err.value(py);
    let _ = inst.setattr("error_code", code.as_str());
    let _ = inst.setattr("gql_status", gql_status.as_str());
    let _ = inst.setattr("is_retryable", code.is_retryable());
}

/// The answer of a read of the database, or its error raised as a
/// `GrafeoError` (a read that fails is never an absent answer).
pub(crate) fn raised<T>(read: grafeo_common::utils::error::Result<T>) -> PyResult<T> {
    read.map_err(|error| PyGrafeoError::from(error).into())
}

impl From<grafeo_common::utils::error::Error> for PyGrafeoError {
    fn from(err: grafeo_common::utils::error::Error) -> Self {
        use grafeo_bindings_common::error::{ErrorCategory, classify_error};
        let code = Some(err.error_code());
        let gql_status = Some(GqlStatus::from(&err));
        let message = err.to_string();
        if err.error_code() == ErrorCode::DatabaseLocked {
            return PyGrafeoError::Locked { message };
        }
        match classify_error(&err) {
            ErrorCategory::Query => PyGrafeoError::Query {
                message,
                code,
                gql_status,
            },
            ErrorCategory::Transaction => PyGrafeoError::Transaction {
                message,
                code,
                gql_status,
            },
            ErrorCategory::DatabaseClosed => PyGrafeoError::DatabaseClosed { message },
            ErrorCategory::Corruption => PyGrafeoError::Corruption { message },
            _ => PyGrafeoError::Database {
                message,
                code,
                gql_status,
            },
        }
    }
}

/// Convenience type for functions that may fail with a Python-compatible error.
pub type PyGrafeoResult<T> = Result<T, PyGrafeoError>;
