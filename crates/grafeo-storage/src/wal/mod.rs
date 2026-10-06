//! Write-Ahead Log - your safety net for crashes.
//!
//! Every mutation goes to the WAL before being applied to the main store.
//! If you crash mid-transaction, [`WalRecovery`] replays the log to restore
//! a consistent state. No committed data is lost.
//!
//! | Durability mode | What it does | When to use |
//! | --------------- | ------------ | ----------- |
//! | [`Sync`](DurabilityMode::Sync) | fsync after every commit | Can't lose any data |
//! | [`Batch`](DurabilityMode::Batch) | Periodic fsync | Balance of safety and speed |
//! | [`Adaptive`](DurabilityMode::Adaptive) | Self-tuning background sync | Variable disk latency |
//! | [`NoSync`](DurabilityMode::NoSync) | Let OS decide | Testing, when speed matters most |
//!
//! ## Adaptive Mode
//!
//! For workloads with variable disk latency, use [`Adaptive`](DurabilityMode::Adaptive)
//! mode with an [`AdaptiveFlusher`]:
//!
//! ```no_run
//! use grafeo_storage::wal::{WalManager, WalConfig, DurabilityMode, AdaptiveFlusher};
//! use std::sync::Arc;
//!
//! # fn main() -> grafeo_common::utils::error::Result<()> {
//! let config = WalConfig {
//!     durability: DurabilityMode::Adaptive { target_interval_ms: 100 },
//!     ..Default::default()
//! };
//! let wal = Arc::new(WalManager::with_config("wal_dir", config)?);
//! let flusher = AdaptiveFlusher::new(Arc::clone(&wal), 100);
//!
//! // Use wal normally - flusher handles background syncing
//! // Drop flusher for graceful shutdown with final flush
//! # Ok(())
//! # }
//! ```
//!
//! Choose [`WalManager`] for sync code, [`AsyncWalManager`] for async.

mod async_log;
#[cfg(feature = "async-storage")]
mod async_typed;
mod flusher;
mod log;
mod record;
mod recovery;
mod typed;

pub use async_log::AsyncWalManager;
#[cfg(feature = "async-storage")]
pub use async_typed::{AsyncLpgWal, AsyncTypedWal};
pub use flusher::{AdaptiveFlusher, FlusherStats};
pub use log::{CheckpointMetadata, DurabilityMode, WalCipher, WalConfig, WalManager};
pub use record::{
    GraphTypeAlterationKind, NamedConstraintKind, PropertyAlterationKind, TypeConstraintKind,
    WalEntry, WalRecord,
};
pub use recovery::{RecoveredWal, WalRecovery};
pub use typed::{LpgWal, TypedWal};
