//! Execution primitives for vectorized query processing.
//!
//! This module provides the core data structures for vectorized execution:
//!
//! - [`chunk`] - DataChunk for batched tuple processing
//! - [`vector`] - ValueVector for columnar storage
//! - [`selection`] - SelectionVector for filtering
//! - [`operators`] - Physical operators (scan, filter, project, join)

pub mod chunk;
pub mod operators;
pub mod selection;
pub mod vector;

pub use chunk::DataChunk;
pub use selection::SelectionVector;
pub use vector::ValueVector;
