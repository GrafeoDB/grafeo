//! Storage backends for Graphos.
//!
//! This module provides different storage strategies:
//!
//! - [`memory`] - Pure in-memory storage (default)
//! - [`wal`] - Write-Ahead Log for durability

pub mod memory;
pub mod wal;

pub use memory::MemoryBackend;
pub use wal::WalManager;
