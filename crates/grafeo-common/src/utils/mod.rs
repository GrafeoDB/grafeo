//! Common utilities used throughout Grafeo.
//!
//! - [`error`] - Error types like [`Error`] and [`QueryError`](error::QueryError)
//! - [`gqlstatus`] - GQLSTATUS diagnostic codes (ISO/IEC 39075:2024, sec 23)
//! - [`hash`] - Fast hashing with FxHash (non-cryptographic)
//! - [`random`] - The random numbers of `rand()` and the other per-call functions
//! - [`strings`] - String utilities for suggestions and fuzzy matching

pub mod error;
pub mod gqlstatus;
pub mod hash;
pub mod random;
pub mod strings;

pub use error::{Error, Result};
pub use gqlstatus::{DiagnosticRecord, GqlStatus};
pub use hash::FxHasher;
