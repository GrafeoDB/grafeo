//! Common utilities used throughout Grafeo.
//!
//! - [`error`] - Error types like [`Error`] and [`QueryError`](error::QueryError)
//! - [`hash`] - Fast hashing with FxHash (non-cryptographic)
//! - [`strings`] - String utilities for suggestions and fuzzy matching

pub mod error;
pub mod hash;
pub mod strings;

pub use error::{Error, Result};
pub use hash::FxHasher;
