//! # grafeo-common
//!
//! The foundation layer - types and utilities used everywhere in Grafeo.
//!
//! You probably don't need to use this crate directly. The main `grafeo` crate
//! re-exports the types you'll actually use ([`NodeId`], [`EdgeId`], [`Value`]).
//!
//! If you're building extensions or diving into internals, here's what's here:
//!
//! ## Modules
//!
//! - [`types`] - Core types: [`NodeId`], [`EdgeId`], [`Value`], [`PropertyKey`]
//! - [`change`] - Change sets: a transaction's writes, for the WAL, undo and CDC
//! - [`collections`] - Type aliases for hash maps/sets with consistent hashing
//! - [`memory`] - Allocators for performance-critical paths (arenas, pools)
//! - [`mvcc`] - Version chains for snapshot isolation
//! - [`utils`] - Hashing, error types, and other helpers

#![deny(unsafe_code)]

pub mod change;
pub mod collections;
pub mod fmt;
pub mod memory;
pub mod mvcc;
pub mod storage;
pub mod testing;
pub mod tracing_macros;
pub mod types;
pub mod utils;

// The types you'll use most often
pub use mvcc::{Version, VersionChain, VersionInfo};
pub use types::{
    EdgeId, EpochId, LogicalType, NodeId, PropertyKey, Timestamp, TransactionId, Value,
};
pub use utils::error::{Error, Result};

// Tiered storage types (feature-gated)
#[cfg(feature = "tiered-storage")]
pub use mvcc::{ColdVersionRef, HotVersionRef, OptionalEpochId, VersionIndex, VersionRef};

// Temporal property versioning (feature-gated)
#[cfg(feature = "temporal")]
pub mod temporal;
#[cfg(feature = "temporal")]
pub use temporal::VersionLog;

// Encryption at rest (feature-gated)
#[cfg(feature = "encryption")]
pub mod encryption;
