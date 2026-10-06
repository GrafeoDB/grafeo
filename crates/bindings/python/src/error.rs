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
//! closed database raises `DatabaseClosedError`, a subclass of `GrafeoError`.

use grafeo_common::utils::error::ErrorCode;
use pyo3::create_exception;
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use thiserror::Error;

create_exception!(
    grafeo,
    GrafeoError,
    PyRuntimeError,
    "Grafeo-specific runtime error with machine-readable `error_code` and `is_retryable` attributes."
);

create_exception!(
    grafeo,
    DatabaseClosedError,
    GrafeoError,
    "Raised by a write to a database whose `close()` started (persistent databases only). Open it again to write."
);

/// Grafeo errors that translate to Python exceptions.
#[derive(Error, Debug)]
pub enum PyGrafeoError {
    #[error("Database error: {message}")]
    Database {
        message: String,
        code: Option<ErrorCode>,
    },

    #[error("Query error: {message}")]
    Query {
        message: String,
        code: Option<ErrorCode>,
    },

    #[error("Type error: {0}")]
    Type(String),

    #[error("Transaction error: {message}")]
    Transaction {
        message: String,
        code: Option<ErrorCode>,
    },

    #[error("Database closed: {message}")]
    DatabaseClosed { message: String },

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
        }
    }

    /// Constructs a plain `Query` variant without a known error code.
    pub fn query<S: Into<String>>(message: S) -> Self {
        Self::Query {
            message: message.into(),
            code: None,
        }
    }

    /// Constructs a plain `Transaction` variant without a known error code.
    pub fn transaction<S: Into<String>>(message: S) -> Self {
        Self::Transaction {
            message: message.into(),
            code: None,
        }
    }
}

impl From<PyGrafeoError> for PyErr {
    fn from(err: PyGrafeoError) -> Self {
        match err {
            PyGrafeoError::InvalidArgument(msg) | PyGrafeoError::Type(msg) => {
                PyValueError::new_err(msg)
            }
            PyGrafeoError::Database { message, code }
            | PyGrafeoError::Query { message, code }
            | PyGrafeoError::Transaction { message, code } => build_grafeo_error(message, code),
            PyGrafeoError::DatabaseClosed { message } => Python::attach(|py| {
                let err = DatabaseClosedError::new_err(message);
                set_code(&err, py, ErrorCode::DatabaseClosed);
                err
            }),
        }
    }
}

/// Builds a `GrafeoError` with `error_code` and `is_retryable` attributes
/// always populated. If the upstream error did not carry a specific code,
/// the generic `Internal` code is used so that callers can rely on both
/// attributes being present on every `GrafeoError`.
fn build_grafeo_error(message: String, code: Option<ErrorCode>) -> PyErr {
    Python::attach(|py| {
        let err = GrafeoError::new_err(message);
        set_code(&err, py, code.unwrap_or(ErrorCode::Internal));
        err
    })
}

/// Sets the `error_code` and `is_retryable` attributes of a `GrafeoError`.
fn set_code(err: &PyErr, py: Python<'_>, code: ErrorCode) {
    let inst = err.value(py);
    let _ = inst.setattr("error_code", code.as_str());
    let _ = inst.setattr("is_retryable", code.is_retryable());
}

impl From<grafeo_common::utils::error::Error> for PyGrafeoError {
    fn from(err: grafeo_common::utils::error::Error) -> Self {
        use grafeo_bindings_common::error::{ErrorCategory, classify_error};
        let code = Some(err.error_code());
        let message = err.to_string();
        match classify_error(&err) {
            ErrorCategory::Query => PyGrafeoError::Query { message, code },
            ErrorCategory::Transaction => PyGrafeoError::Transaction { message, code },
            ErrorCategory::DatabaseClosed => PyGrafeoError::DatabaseClosed { message },
            _ => PyGrafeoError::Database { message, code },
        }
    }
}

/// Convenience type for functions that may fail with a Python-compatible error.
pub type PyGrafeoResult<T> = Result<T, PyGrafeoError>;
