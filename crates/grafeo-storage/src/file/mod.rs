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
//! ## Sections and chunks
//!
//! Each section of an image is a sequence of chunks: a checkpoint hands the
//! container's writer to [`Section::write_to`], which writes the section's
//! chunks one at a time, and a read serves them back to
//! [`Section::read_from`] through an
//! [`ImageSource`](grafeo_common::storage::ImageSource). Every chunk is stored
//! on pages of its own, and its directory entry names its section type and
//! version, its [`ChunkKind`], graph, column, rows and codec, and holds the
//! CRC-32 of its stored bytes. The container treats the bytes as opaque: it
//! refuses two chunks of one section with one identity (kind, graph, column,
//! first row), and skips a chunk of an unknown section type or chunk kind only
//! when its entry marks it optional.
//!
//! A section of a checkpoint writes a metadata chunk and its column chunks or
//! stream pieces. A raw chunk holds a section whole: a 0.5.x file holds
//! each section as one, and the catalog section is still written so.
//!
//! [`Section::write_to`]: grafeo_common::storage::Section::write_to
//! [`Section::read_from`]: grafeo_common::storage::Section::read_from
//! [`ChunkKind`]: grafeo_common::storage::ChunkKind
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
pub use manager::{CheckpointHeader, GrafeoFileManager, ImageStats};
