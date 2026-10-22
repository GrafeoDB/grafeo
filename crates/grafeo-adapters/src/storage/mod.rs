//! In-memory storage backend.
//!
//! The persistence layer (WAL, `.grafeo` container, async backends) has
//! moved to the `grafeo-storage` crate. This module retains only the
//! in-memory backend which bridges `grafeo-core::LpgStore` to the
//! storage interface.

#[cfg(feature = "lpg")]
pub mod memory;

#[cfg(feature = "lpg")]
pub use memory::MemoryBackend;
