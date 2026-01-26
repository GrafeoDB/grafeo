//! Physical operators for query execution.
//!
//! This module provides the physical operators that form the execution tree:
//!
//! - Scan: Read nodes/edges from storage
//! - Filter: Apply predicates to filter rows
//! - Project: Select and transform columns
//! - Join: Hash join and nested loop join
//! - Aggregate: Group by and aggregation functions

mod scan;
mod filter;
mod project;

pub use scan::ScanOperator;
pub use filter::FilterOperator;
pub use project::ProjectOperator;

use super::DataChunk;

/// Result of executing an operator.
pub type OperatorResult = Result<Option<DataChunk>, OperatorError>;

/// Error during operator execution.
#[derive(Debug, Clone)]
pub enum OperatorError {
    /// Type mismatch during execution.
    TypeMismatch { expected: String, found: String },
    /// Column not found.
    ColumnNotFound(String),
    /// Execution error.
    Execution(String),
}

impl std::fmt::Display for OperatorError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            OperatorError::TypeMismatch { expected, found } => {
                write!(f, "Type mismatch: expected {expected}, found {found}")
            }
            OperatorError::ColumnNotFound(name) => write!(f, "Column not found: {name}"),
            OperatorError::Execution(msg) => write!(f, "Execution error: {msg}"),
        }
    }
}

impl std::error::Error for OperatorError {}

/// Trait for physical operators.
pub trait Operator: Send + Sync {
    /// Returns the next chunk of data, or None if exhausted.
    fn next(&mut self) -> OperatorResult;

    /// Resets the operator to its initial state.
    fn reset(&mut self);

    /// Returns the name of this operator for debugging.
    fn name(&self) -> &'static str;
}
