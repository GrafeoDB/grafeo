//! Storage backends - how your data gets persisted.
//!
//! | Backend | Speed | Durability | Use when |
//! | ------- | ----- | ---------- | -------- |
//! | [`memory`] | Fastest | None (data lost on restart) | Testing, prototyping |
//! | `wal` (feature-gated) | Fast | Survives crashes | Production workloads |
//!
//! The WAL (Write-Ahead Log) writes changes to disk before applying them,
//! so you can recover after crashes without losing committed transactions.
//! The WAL module requires filesystem I/O and is gated behind the `wal` feature.

pub mod memory;
#[cfg(feature = "wal")]
pub mod wal;

pub use memory::MemoryBackend;
#[cfg(feature = "wal")]
pub use wal::WalManager;
