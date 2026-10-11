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
//!
//! ## WAL v2
//!
//! The 0.6 log, not yet used by the engine: segment files named after the
//! log position (LSN) of their first frame, each with a 128-byte header that
//! names its database ([`SegmentHeader`]), holding frames with a 25-byte
//! header ([`FrameHeader`]). A transaction is one contiguous group of frames,
//! FIRST to LAST; the LAST frame is its commit marker. The storage layer
//! treats records as opaque bytes.
//!
//! - [`Wal`] writes groups through a [`GroupWriter`]: a failed write cuts the
//!   group back, a failed sync poisons the writer ([`GroupError`]).
//! - [`WalScan`] returns complete groups in log order, cuts torn tails
//!   ([`ScanEnd::cut_torn_tail`]) and refuses damage, gaps and the WAL of
//!   another database ([`WalError`]).
//! - Encrypted segments derive their key from a per-segment salt
//!   ([`CipherForSalt`]); each frame's associated data binds its position
//!   ([`frame_aad`]), and its checksum covers the ciphertext.

mod async_log;
#[cfg(feature = "async-storage")]
mod async_typed;
mod cipher;
mod error;
mod flusher;
mod frame;
mod log;
mod reader;
mod record;
mod recovery;
mod segment;
mod typed;
mod writer;

pub use async_log::AsyncWalManager;
#[cfg(feature = "async-storage")]
pub use async_typed::{AsyncLpgWal, AsyncTypedWal};
pub use cipher::{CipherForSalt, FRAME_AAD_BYTES, frame_aad};
pub use error::{GroupError, WalError};
pub use flusher::{AdaptiveFlusher, FlusherStats};
pub use frame::{
    FRAME_HEADER_BYTES, FRAME_PROLOGUE_BYTES, FRAME_TARGET, FrameFlags, FrameHeader,
    MAX_FRAME_PAYLOAD, MAX_FRAME_RECORD_BYTES, SEALED_OVERHEAD,
};
pub use log::{CheckpointMetadata, DurabilityMode, WalCipher, WalConfig, WalManager};
pub use reader::{GROUP_BUFFER_BYTES, GroupFrames, ScanEnd, ScanOptions, Tail, TailKind, WalScan};
pub use record::{
    GraphTypeAlterationKind, NamedConstraintKind, PropertyAlterationKind, TypeConstraintKind,
    WalEntry, WalRecord,
};
pub use recovery::{RecoveredWal, WalRecovery};
pub use segment::{
    KEY_CHECK_AAD_BYTES, KEY_CHECK_BYTES, SALT_BYTES, SEGMENT_HEADER_BYTES, SEGMENT_MAGIC,
    SEGMENT_VERSION, SegmentHeader, WalDirectory, is_unfinished_segment, list_wal_directory,
    parse_segment_file_name, segment_file_name, stored_key_check_aad,
};
pub use typed::{LpgWal, TypedWal};
pub use writer::{DEFAULT_SEGMENT_BYTES, GroupEnd, GroupWriter, Wal, WalOptions};
