//! Memory allocators for Graphos.
//!
//! This module provides specialized memory allocators optimized for
//! graph database workloads:
//!
//! - [`arena`] - Epoch-based arena allocator for structural sharing
//! - [`bump`] - Fast bump allocator for temporary allocations
//! - [`pool`] - Object pool for frequently allocated types

pub mod arena;
pub mod bump;
pub mod pool;

pub use arena::{Arena, ArenaAllocator};
pub use bump::BumpAllocator;
pub use pool::ObjectPool;
