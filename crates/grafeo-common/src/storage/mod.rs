//! Storage section abstractions for the `.grafeo` container format.
//!
//! The container file stores data in typed **sections**, each independently
//! addressable and checksummed. This module defines the contract between
//! section serializers (in `grafeo-core`) and section I/O (in `grafeo-storage`).
//!
//! A section streams out and in as chunks ([`section`]), cut by the caps and
//! byte streams of [`chunk`]; an image ([`image`]) serves the sections of one
//! checkpoint, from a file or from memory. [`value_codec`] is a lossless
//! codec for property values, for the chunked sections to encode them with,
//! [`catalog_record`] holds the catalog's typed, framed records, and
//! [`log_record`] the WAL's, framed the same way.

pub mod catalog_record;
pub mod chunk;
pub mod image;
pub mod log_record;
pub mod page_fetcher;
pub mod section;
pub mod value_codec;

pub use catalog_record::{CatalogRecord, RECORD_REQUIRED, read_catalog_records};
pub use chunk::{
    ChunkCaps, ChunkIdentities, ChunkStreamReader, ChunkStreamWriter, read_stream, stream_error,
};
pub use image::{ImageSource, MemoryImage, MemorySection, ServedOnce};
pub use log_record::{LogRecord, LogRecordRef, MAX_RECORD_BYTES, read_log_records};
pub use page_fetcher::{AccessHint, PageFetcher};
pub use section::{
    ChunkKind, ChunkMeta, Section, SectionDirectoryEntry, SectionFlags, SectionMemoryConfig,
    SectionSink, SectionSource, SectionType, TierOverride, check_version, legacy_bytes, read_raw,
    write_raw,
};
pub use value_codec::{
    MAX_PROPERTY_VALUE_DEPTH, MAX_VALUE_DEPTH, decode_value, encode_value, encoded_len,
};
