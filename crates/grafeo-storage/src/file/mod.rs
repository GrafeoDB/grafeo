//! Single-file database format (`.grafeo`).
//!
//! This module implements a portable, crash-safe, single-file storage format.
//! At rest, only the `.grafeo` file exists. During operation a sidecar
//! WAL directory (`<path>.wal/`) captures in-flight mutations. Checkpoints
//! fold those mutations into the `.grafeo` file but retain the sidecar; it
//! is removed only on a clean `close()`.
//!
//! ## File layout
//!
//! | Offset | Size | Contents |
//! |--------|------|----------|
//! | 0 | 4 KiB | [`FileHeader`]: magic `GRAF`, version, page size |
//! | 4 KiB | 4 KiB | [`DbHeader`] slot 0 (H1) |
//! | 8 KiB | 4 KiB | [`DbHeader`] slot 1 (H2) |
//! | 12 KiB+ | variable | Snapshot data payload (bincode-encoded) |
//!
//! ## Crash safety
//!
//! Two database headers alternate writes. On checkpoint, the inactive slot
//! is overwritten with metadata pointing to the freshly written snapshot.
//! If the process crashes mid-write, the other header is still valid.

pub mod format;
pub mod header;
pub mod manager;

pub use format::{DbHeader, FileHeader, MAGIC};
pub use manager::GrafeoFileManager;
