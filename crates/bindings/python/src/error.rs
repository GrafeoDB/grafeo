//! Converts Rust errors to Python exceptions.
//!
//! Type errors and invalid arguments become `ValueError`, while database,
//! query, and transaction errors become `RuntimeError`.

use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use thiserror::Error;

/// Grafeo errors that translate to Python exceptions.
#[derive(Error, Debug)]
pub enum PyGrafeoError {
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

impl From<PyGrafeoError> for PyErr {
    fn from(err: PyGrafeoError) -> Self {
        match err {
            PyGrafeoError::InvalidArgument(msg) | PyGrafeoError::Type(msg) => {
                PyValueError::new_err(msg)
            }
            PyGrafeoError::Database(msg)
            | PyGrafeoError::Query(msg)
            | PyGrafeoError::Transaction(msg) => PyRuntimeError::new_err(msg),
        }
    }
}

impl From<grafeo_common::utils::error::Error> for PyGrafeoError {
    fn from(err: grafeo_common::utils::error::Error) -> Self {
        match err {
            grafeo_common::utils::error::Error::Query(e) => PyGrafeoError::Query(e.to_string()),
            grafeo_common::utils::error::Error::Transaction(e) => {
                PyGrafeoError::Transaction(e.to_string())
            }
            other => PyGrafeoError::Database(other.to_string()),
        }
    }
}

/// Convenience type for functions that may fail with a Python-compatible error.
pub type PyGrafeoResult<T> = Result<T, PyGrafeoError>;
