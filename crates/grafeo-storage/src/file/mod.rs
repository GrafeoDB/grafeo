//! Single-file database format (`.grafeo`).
//!
//! This module implements a portable, crash-safe, single-file storage format.
//! At rest, only the `.grafeo` file exists. During operation a sidecar
//! WAL directory (`<path>.wal/`) captures in-flight mutations. Checkpoints
//! fold those mutations into the `.grafeo` file but retain the sidecar; it
//! is removed only on a clean `close()`.
//!
//! ## File layout (container format v3, written since 0.6.0)
//!
//! | Offset | Size | Contents |
//! |--------|------|----------|
//! | 0 | 4 KiB | [`FileHeaderV3`](v3::header::FileHeaderV3): magic `GRAF`, format version 3, page size, flags (encrypted), database id |
//! | 4 KiB | 4 KiB | [`DbHeaderV3`](v3::header::DbHeaderV3) slot 0 |
//! | 8 KiB | 4 KiB | [`DbHeaderV3`](v3::header::DbHeaderV3) slot 1 |
//! | 12 KiB+ | 4 KiB pages | Chunks and directory blocks of the images |
//!
//! An image is the state of one checkpoint: the chunks of its sections and
//! a chained directory listing them ([`v3::directory`]). A database header
//! points at the first directory block of its image; the valid header with
//! the higher iteration is the active one.
//!
//! ## Crash safety
//!
//! Checkpoints are copy-on-write ([`GrafeoFileManager::write_checkpoint`]):
//! the new image goes into pages the active image does not use, the file is
//! synced, and only then is the inactive header slot overwritten to point at
//! it (and synced). A crash before that header is on disk leaves the old
//! image active and intact; a torn header write fails its checksum, so the
//! other slot stays active.
//!
//! ## Files written by 0.5.x
//!
//! 0.5.x wrote container v1 (one snapshot blob) and v2 (a section directory
//! at 12 KiB), with bincode headers ([`format`](mod@format), [`header`]). [`detect`] tells
//! the layouts apart and [`legacy`] reads the old ones, so they can be
//! migrated; [`GrafeoFileManager`] opens only v3.

pub mod detect;
pub mod format;
pub mod header;
pub mod legacy;
pub mod manager;
pub mod v3;

pub use format::{DbHeader, FileHeader, MAGIC};
pub use manager::{CheckpointHeader, GrafeoFileManager};
