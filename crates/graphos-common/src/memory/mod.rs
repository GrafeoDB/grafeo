//! Memory allocators for Graphos.
//!
//! This module provides specialized memory allocators optimized for
//! graph database workloads:
//!
//! - [`arena`] - Epoch-based arena allocator for structural sharing
//! - [`bump`] - Fast bump allocator for temporary allocations
//! - [`pool`] - Object pool for frequently allocated types
//! - [`buffer`] - Unified buffer manager for centralized memory management

pub mod arena;
pub mod buffer;
pub mod bump;
pub mod pool;

pub use arena::{Arena, ArenaAllocator};
pub use buffer::{
    BufferManager, BufferManagerConfig, BufferStats, MemoryConsumer, MemoryGrant, MemoryRegion,
    PressureLevel,
};
pub use bump::BumpAllocator;
pub use pool::ObjectPool;
