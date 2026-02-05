//! Storage backends - how your data gets persisted.
//!
//! | Backend | Speed | Durability | Use when |
//! | ------- | ----- | ---------- | -------- |
//! | [`memory`] | Fastest | None (data lost on restart) | Testing, prototyping |
//! | [`wal`] | Fast | Survives crashes | Production workloads |
//!
//! The WAL (Write-Ahead Log) writes changes to disk before applying them,
//! so you can recover after crashes without losing committed transactions.

pub mod memory;
pub mod wal;

pub use memory::MemoryBackend;
pub use wal::WalManager;
