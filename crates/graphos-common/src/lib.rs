//! # graphos-common
//!
//! Foundation layer for Graphos: types, memory allocators, and utilities.
//!
//! This crate provides the fundamental building blocks used by all other
//! Graphos crates. It has no internal dependencies and should be kept minimal.
//!
//! ## Modules
//!
//! - [`types`] - Core type definitions (NodeId, EdgeId, Value, etc.)
//! - [`memory`] - Memory allocators (arena, bump, pool)
//! - [`utils`] - Utility functions and helpers (hashing, errors)

#![warn(missing_docs)]
#![warn(clippy::all)]
#![warn(clippy::pedantic)]

pub mod memory;
pub mod types;
pub mod utils;

// Re-export commonly used types at crate root
pub use types::{EdgeId, EpochId, LogicalType, NodeId, PropertyKey, Timestamp, TxId, Value};
pub use utils::error::{Error, Result};
