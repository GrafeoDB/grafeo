//! In-memory storage backend.
//!
//! The persistence layer (WAL, `.grafeo` container, async backends) has
//! moved to the `grafeo-storage` crate. This module retains only the
//! in-memory backend which bridges `grafeo-core::RowGroupStore` to the
//! storage interface.

pub mod memory;

pub use memory::MemoryBackend;
