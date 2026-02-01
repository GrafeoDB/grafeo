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
//! - [`memory`] - Allocators for performance-critical paths (arenas, pools)
//! - [`mvcc`] - Version chains for snapshot isolation
//! - [`utils`] - Hashing, error types, and other helpers

pub mod memory;
pub mod mvcc;
pub mod types;
pub mod utils;

// The types you'll use most often
pub use mvcc::{Version, VersionChain, VersionInfo};
pub use types::{EdgeId, EpochId, LogicalType, NodeId, PropertyKey, Timestamp, TxId, Value};
pub use utils::error::{Error, Result};

// Tiered storage types (feature-gated)
#[cfg(feature = "tiered-storage")]
pub use mvcc::{ColdVersionRef, HotVersionRef, OptionalEpochId, VersionIndex, VersionRef};
