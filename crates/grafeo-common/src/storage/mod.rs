//! Storage section abstractions for the `.grafeo` container format.
//!
//! The container file stores data in typed **sections**, each independently
//! addressable and checksummed. This module defines the contract between
//! section serializers (in `grafeo-core`) and section I/O (in `grafeo-storage`).

pub mod page_fetcher;
pub mod section;

pub use page_fetcher::{AccessHint, PageFetcher};
pub use section::{
    ChunkKind, ChunkMeta, Section, SectionDirectoryEntry, SectionFlags, SectionMemoryConfig,
    SectionSink, SectionSource, SectionType, TierOverride,
};
