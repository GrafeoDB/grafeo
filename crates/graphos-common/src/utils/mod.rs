//! Utility functions and helpers.
//!
//! This module provides common utilities used throughout Graphos:
//!
//! - [`error`] - Error types and result aliases
//! - [`hash`] - Fast hashing utilities

pub mod error;
pub mod hash;

pub use error::{Error, Result};
pub use hash::FxHasher;
