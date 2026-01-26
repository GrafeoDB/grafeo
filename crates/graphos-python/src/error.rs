//! Error handling for Python bindings.

use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use thiserror::Error;

/// Python binding errors.
#[derive(Error, Debug)]
pub enum PyGraphosError {
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

impl From<PyGraphosError> for PyErr {
    fn from(err: PyGraphosError) -> Self {
        match err {
            PyGraphosError::InvalidArgument(msg) | PyGraphosError::Type(msg) => {
                PyValueError::new_err(msg)
            }
            PyGraphosError::Database(msg)
            | PyGraphosError::Query(msg)
            | PyGraphosError::Transaction(msg) => PyRuntimeError::new_err(msg),
        }
    }
}

/// Result type for Python bindings.
pub type PyGraphosResult<T> = Result<T, PyGraphosError>;
