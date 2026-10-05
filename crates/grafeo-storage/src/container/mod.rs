//! Section containers: the 0.5.x section directory, and memory-mapped
//! section access.
//!
//! The container treats section data as opaque `&[u8]` bytes. Since 0.6.0
//! `.grafeo` files use container format v3 (see `crate::file::v3`: chunks in
//! pages and a chained directory). This module keeps the 0.5.x section
//! directory, which the legacy reader (`crate::file::legacy`) uses to read
//! old files, and the mmap and spill helpers.
//!
//! ## File layout of container v2 (written by 0.5.x, read only)
//!
//! | Offset | Size | Contents |
//! |--------|------|----------|
//! | 0x0000 | 4 KiB | FileHeader (magic, format version) |
//! | 0x1000 | 4 KiB | DbHeader H1 (iteration, checksum) |
//! | 0x2000 | 4 KiB | DbHeader H2 (alternating copy) |
//! | 0x3000 | 4 KiB | Section Directory |
//! | 0x4000+ | variable | Section data (page-aligned) |

pub mod directory;

#[cfg(feature = "wal")]
pub mod mmap;

#[cfg(feature = "wal")]
pub mod page_fetcher;

#[cfg(feature = "wal")]
pub mod spill;

pub use directory::SectionDirectory;

#[cfg(feature = "wal")]
pub use mmap::MmapSection;

#[cfg(feature = "wal")]
pub use page_fetcher::{AccessHint, MmapPageFetcher, PageFetcher};

#[cfg(feature = "wal")]
pub use spill::write_and_mmap_spill_file;
