//! Section types and traits for the `.grafeo` container format.
//!
//! A `.grafeo` file is a container of typed sections. Each section holds
//! one kind of data (LPG nodes, RDF triples, vector indexes, etc.) and
//! can be independently read, written, checksummed, and mmap'd.
//!
//! The [`Section`] trait is the contract between serializers (grafeo-core)
//! and the container I/O layer (grafeo-storage). Serializers produce opaque
//! bytes; the container writes them to disk without knowing the contents.

use std::sync::Arc;

use serde::{Deserialize, Serialize};

use crate::memory::buffer::SpillError;
use crate::storage::page_fetcher::PageFetcher;
use crate::utils::error::{Error, Result};

// ── Section Type ────────────────────────────────────────────────────

/// Identifies a section type in the container directory.
///
/// Types 1-9 are **data sections** (authoritative, cannot be rebuilt).
/// Types 10-19 are **index sections** (derived, can be rebuilt from data).
/// Types 20+ are reserved for future acceleration structures.
///
/// Values must stay below 256: container v3 stores the section type in one
/// byte (see [`to_u8`](Self::to_u8) and [`from_u8`](Self::from_u8)). The
/// values are part of the file format and never change.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[repr(u32)]
#[non_exhaustive]
pub enum SectionType {
    /// Schema definitions, index metadata, epoch, configuration.
    Catalog = 1,
    /// LPG nodes, edges, properties, named graphs.
    LpgStore = 2,
    /// RDF triples and named graphs.
    RdfStore = 3,
    /// Columnar CompactStore: read-only base for layered storage.
    CompactStore = 4,
    /// Layered overlay deletion log: ids of base entities the overlay
    /// has deleted but not yet merged. Persists tombstones so that a
    /// previously-deleted base node does not reappear after reload
    /// when the next compact has not yet run.
    OverlayDeletions = 5,

    /// Vector embeddings, HNSW topology, quantization data.
    VectorStore = 10,
    /// BM25 inverted index: term dictionary, postings lists.
    TextIndex = 11,
    /// RDF Ring index: wavelet trees, succinct permutations.
    RdfRing = 12,
    /// Property hash/btree indexes.
    PropertyIndex = 20,
}

impl SectionType {
    /// Whether this section type holds authoritative data (not rebuildable).
    #[must_use]
    pub const fn is_data_section(self) -> bool {
        (self as u32) < 10
    }

    /// Whether this section type holds a derived index (rebuildable from data).
    #[must_use]
    pub const fn is_index_section(self) -> bool {
        (self as u32) >= 10
    }

    /// The on-disk byte of this section type (its discriminant), the inverse
    /// of [`from_u8`](Self::from_u8).
    #[must_use]
    pub const fn to_u8(self) -> u8 {
        match self {
            Self::Catalog => 1,
            Self::LpgStore => 2,
            Self::RdfStore => 3,
            Self::CompactStore => 4,
            Self::OverlayDeletions => 5,
            Self::VectorStore => 10,
            Self::TextIndex => 11,
            Self::RdfRing => 12,
            Self::PropertyIndex => 20,
        }
    }

    /// Decodes a section type from its on-disk byte, or `None` for an unknown one.
    #[must_use]
    pub const fn from_u8(byte: u8) -> Option<Self> {
        match byte {
            1 => Some(Self::Catalog),
            2 => Some(Self::LpgStore),
            3 => Some(Self::RdfStore),
            4 => Some(Self::CompactStore),
            5 => Some(Self::OverlayDeletions),
            10 => Some(Self::VectorStore),
            11 => Some(Self::TextIndex),
            12 => Some(Self::RdfRing),
            20 => Some(Self::PropertyIndex),
            _ => None,
        }
    }
}

// ── Section Flags ───────────────────────────────────────────────────

/// Flags for a section entry in the container directory.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct SectionFlags {
    /// Bit 0: section is required (older binaries must refuse to open if unknown).
    /// When false, unknown section types can be safely skipped.
    pub required: bool,
    /// Bit 1: section data can be mmap'd for zero-copy access.
    pub mmap_able: bool,
}

impl SectionFlags {
    /// Pack flags into a single byte for on-disk storage.
    #[must_use]
    pub const fn to_byte(self) -> u8 {
        let mut flags = 0u8;
        if self.required {
            flags |= 0x01;
        }
        if self.mmap_able {
            flags |= 0x02;
        }
        flags
    }

    /// Unpack flags from a single byte.
    #[must_use]
    pub const fn from_byte(byte: u8) -> Self {
        Self {
            required: byte & 0x01 != 0,
            mmap_able: byte & 0x02 != 0,
        }
    }
}

impl SectionType {
    /// Default flags for this section type.
    #[must_use]
    pub const fn default_flags(self) -> SectionFlags {
        match self {
            Self::Catalog => SectionFlags {
                required: true,
                mmap_able: false,
            },
            Self::LpgStore => SectionFlags {
                required: true,
                mmap_able: false,
            },
            Self::RdfStore => SectionFlags {
                required: false,
                mmap_able: false,
            },
            Self::CompactStore => SectionFlags {
                required: true,
                mmap_able: true,
            },
            Self::OverlayDeletions => SectionFlags {
                // Marked non-required so older readers that don't know about
                // it can skip rather than refuse to open. Functionally the
                // section is authoritative for deletion durability, but a
                // reader that ignores it fails open (deleted base nodes
                // reappear) rather than failing closed (refuse to open).
                required: false,
                mmap_able: false,
            },
            Self::VectorStore | Self::TextIndex | Self::RdfRing | Self::PropertyIndex => {
                SectionFlags {
                    required: false,
                    mmap_able: true,
                }
            }
        }
    }
}

// ── Section Directory Entry ─────────────────────────────────────────

/// A single entry in the container's section directory.
///
/// Fixed 32-byte layout for on-disk storage:
///
/// | Offset | Size | Field |
/// |--------|------|-------|
/// | 0 | 4 | `section_type` (u32 LE) |
/// | 4 | 1 | `version` (u8) |
/// | 5 | 1 | `flags` (packed byte) |
/// | 6 | 2 | reserved (zero) |
/// | 8 | 8 | `offset` (u64 LE, byte offset from file start) |
/// | 16 | 8 | `length` (u64 LE, byte length of section data) |
/// | 24 | 4 | `checksum` (u32 LE, CRC-32 of section data) |
/// | 28 | 4 | reserved (zero) |
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SectionDirectoryEntry {
    /// Which section type this entry describes.
    pub section_type: SectionType,
    /// Per-section format version (allows independent evolution).
    pub version: u8,
    /// Section flags (required, mmap-able).
    pub flags: SectionFlags,
    /// Byte offset from file start where section data begins.
    pub offset: u64,
    /// Byte length of the section data.
    pub length: u64,
    /// CRC-32 checksum of the section data.
    pub checksum: u32,
}

impl SectionDirectoryEntry {
    /// Size of a directory entry on disk (fixed 32 bytes).
    pub const SIZE: usize = 32;
}

// ── Streaming Chunks ────────────────────────────────────────────────

/// What a chunk holds within its section.
///
/// The byte of each kind is part of the file format and never changes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
#[non_exhaustive]
pub enum ChunkKind {
    /// A section's bytes in its 0.5.x layout (or a test section's), passed to
    /// [`Section::deserialize`].
    Raw = 0,
    /// A section's metadata chunk: its layout byte, the caps it was written
    /// with, its graphs and columns.
    Meta = 1,
    /// One column of a table over a range of rows: presence bitmap, zone map,
    /// codec body.
    Column = 2,
    /// The older versions of a column's values over a range of rows (temporal
    /// property history).
    History = 3,
    /// A piece of a byte stream; `row_start` is the piece's offset in the
    /// stream, `column_id` the stream.
    Stream = 4,
}

impl ChunkKind {
    /// The on-disk byte for this kind.
    #[must_use]
    pub const fn to_byte(self) -> u8 {
        self as u8
    }

    /// Decodes an on-disk byte, or `None` for an unknown kind.
    #[must_use]
    pub fn from_byte(byte: u8) -> Option<Self> {
        match byte {
            0 => Some(Self::Raw),
            1 => Some(Self::Meta),
            2 => Some(Self::Column),
            3 => Some(Self::History),
            4 => Some(Self::Stream),
            _ => None,
        }
    }
}

/// A chunk as its section describes it; storage adds where it is.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ChunkMeta {
    /// What the chunk holds.
    pub kind: ChunkKind,
    /// Codec identifier of the chunk's bytes (0 for none).
    pub codec: u8,
    /// Graph the chunk belongs to (0 when not graph-specific).
    pub graph_id: u32,
    /// Column the chunk belongs to (0 when not column-specific); for a
    /// [`ChunkKind::Stream`] piece, the stream.
    pub column_id: u32,
    /// First row held by the chunk; for a [`ChunkKind::Stream`] piece, its
    /// byte offset in the stream.
    pub row_start: u64,
    /// Number of rows held by the chunk (0 for a stream piece).
    pub row_count: u32,
}

impl ChunkMeta {
    /// Metadata of a raw chunk: kind [`ChunkKind::Raw`], everything else zero.
    #[must_use]
    pub const fn raw() -> Self {
        Self {
            kind: ChunkKind::Raw,
            codec: 0,
            graph_id: 0,
            column_id: 0,
            row_start: 0,
            row_count: 0,
        }
    }

    /// Metadata of a section's metadata chunk: kind [`ChunkKind::Meta`],
    /// everything else zero.
    #[must_use]
    pub const fn meta() -> Self {
        Self {
            kind: ChunkKind::Meta,
            ..Self::raw()
        }
    }

    /// Metadata of a column chunk: `row_count` rows of column `column_id` of
    /// graph `graph_id` from row `row_start`, encoded with `codec`.
    #[must_use]
    pub const fn column(
        graph_id: u32,
        column_id: u32,
        row_start: u64,
        row_count: u32,
        codec: u8,
    ) -> Self {
        Self {
            kind: ChunkKind::Column,
            codec,
            graph_id,
            column_id,
            row_start,
            row_count,
        }
    }

    /// Metadata of a history chunk: the older versions of the values of the
    /// column chunk with the same graph, column and rows.
    #[must_use]
    pub const fn history(
        graph_id: u32,
        column_id: u32,
        row_start: u64,
        row_count: u32,
        codec: u8,
    ) -> Self {
        Self {
            kind: ChunkKind::History,
            ..Self::column(graph_id, column_id, row_start, row_count, codec)
        }
    }

    /// Metadata of a piece of byte stream `stream` of graph `graph_id`,
    /// starting at byte `offset` of the stream.
    #[must_use]
    pub const fn stream_piece(graph_id: u32, stream: u32, offset: u64) -> Self {
        Self {
            kind: ChunkKind::Stream,
            codec: 0,
            graph_id,
            column_id: stream,
            row_start: offset,
            row_count: 0,
        }
    }

    /// What must be unique within a section: (kind byte, graph, column, first
    /// row). The codec and the row count are not part of it.
    #[must_use]
    pub const fn identity(&self) -> (u8, u32, u32, u64) {
        (
            self.kind.to_byte(),
            self.graph_id,
            self.column_id,
            self.row_start,
        )
    }
}

/// Receives the chunks of a section as it streams out.
pub trait SectionSink {
    /// Appends one chunk.
    ///
    /// # Errors
    ///
    /// Returns an error if the chunk cannot be stored.
    fn write_chunk(&mut self, meta: ChunkMeta, bytes: &[u8]) -> Result<()>;
}

/// Serves the chunks of a section as it streams in.
pub trait SectionSource {
    /// Describes every chunk, in order.
    fn chunks(&self) -> &[ChunkMeta];

    /// Fetches the bytes of the chunk at `index`.
    ///
    /// # Errors
    ///
    /// Returns an error if `index` is out of range or the bytes cannot be read.
    fn fetch(&self, index: usize) -> Result<bytes::Bytes>;

    /// The section version every chunk of the section was written with (0
    /// for 0.5.x bytes).
    fn section_version(&self) -> u8;
}

/// The bytes of a section stored as exactly one raw chunk without a codec
/// (0.5.x bytes, or a section written through the raw defaults of
/// [`Section`]), `None` when the section has no raw chunk.
///
/// The one raw chunk is [`ChunkMeta::raw`] exactly: no codec, graph, column
/// or rows. The section version is not checked: a raw chunk is read the same
/// way whatever version wrote it.
///
/// # Errors
///
/// Returns [`Error::Serialization`] when raw chunks come several or next to
/// chunks of other kinds, or when the one raw chunk has a codec, a graph, a
/// column or rows; any error from fetching the chunk.
pub fn legacy_bytes(source: &dyn SectionSource) -> Result<Option<bytes::Bytes>> {
    let chunks = source.chunks();
    let raw = chunks
        .iter()
        .filter(|meta| meta.kind == ChunkKind::Raw)
        .count();
    match chunks {
        _ if raw == 0 => Ok(None),
        [meta] if *meta == ChunkMeta::raw() => Ok(Some(source.fetch(0)?)),
        [meta] => Err(Error::Serialization(format!(
            "the raw chunk of a section has codec {}, graph {}, column {}, first row {}, \
             rows {}; the raw chunk of 0.5.x section bytes has all of them 0",
            meta.codec, meta.graph_id, meta.column_id, meta.row_start, meta.row_count
        ))),
        _ => Err(Error::Serialization(format!(
            "a section holds {raw} raw chunks among {} chunks; 0.5.x section bytes are \
             exactly one raw chunk",
            chunks.len()
        ))),
    }
}

/// Refuses a source whose section version is not `expected`, naming the
/// section and both versions.
///
/// # Errors
///
/// Returns [`Error::Serialization`] when the version of `source` differs from
/// `expected`.
pub fn check_version(
    section_type: SectionType,
    source: &dyn SectionSource,
    expected: u8,
) -> Result<()> {
    let found = source.section_version();
    if found == expected {
        Ok(())
    } else {
        Err(Error::Serialization(format!(
            "section {section_type:?} has version {found}, this build reads version {expected}"
        )))
    }
}

// ── Section Trait ───────────────────────────────────────────────────

/// A serializable section for the `.grafeo` container.
///
/// Implemented in `grafeo-core` for each data model (LPG, RDF) and index
/// type (Vector, Text, Ring). The container I/O layer in `grafeo-storage`
/// calls `serialize()` and `deserialize()` without knowing the section internals.
///
/// The unified flush model uses this trait: the engine iterates all sections,
/// serializes dirty ones, and passes the bytes to the container writer.
pub trait Section: Send + Sync {
    /// The section type identifier.
    fn section_type(&self) -> SectionType;

    /// Per-section format version.
    fn version(&self) -> u8 {
        1
    }

    /// Serialize section contents to bytes.
    ///
    /// Called by the flush path (checkpoint, eviction, explicit CHECKPOINT).
    /// The returned bytes are opaque to the container writer.
    ///
    /// # Errors
    ///
    /// Returns an error if serialization fails (e.g., encoding error).
    fn serialize(&self) -> Result<Vec<u8>>;

    /// Populate section contents from bytes.
    ///
    /// Called during recovery (loading from container) or reload (mmap to RAM).
    ///
    /// # Errors
    ///
    /// Returns an error if deserialization fails (e.g., corrupt data, version mismatch).
    fn deserialize(&mut self, data: &[u8]) -> Result<()>;

    /// Stream section contents to `sink` as chunks.
    ///
    /// The default writes [`serialize`](Section::serialize) as one raw chunk.
    ///
    /// # Errors
    ///
    /// Returns an error if serialization fails or the sink rejects a chunk.
    fn write_to(&self, sink: &mut dyn SectionSink) -> Result<()> {
        sink.write_chunk(ChunkMeta::raw(), &self.serialize()?)
    }

    /// Populate section contents from the chunks of `source`.
    ///
    /// The default requires exactly one raw chunk without a codec and passes
    /// its bytes to [`deserialize`](Section::deserialize).
    ///
    /// # Errors
    ///
    /// Returns [`Error::Serialization`] unless the source holds exactly one
    /// raw chunk with codec 0, or any error from fetching or deserializing it.
    fn read_from(&mut self, source: &dyn SectionSource) -> Result<()> {
        match source.chunks() {
            [meta] if meta.kind == ChunkKind::Raw && meta.codec == 0 => {
                let bytes = source.fetch(0)?;
                self.deserialize(&bytes)
            }
            [meta] if meta.kind == ChunkKind::Raw => Err(Error::Serialization(format!(
                "section {:?}: the raw chunk has codec {}, which this section cannot decode",
                self.section_type(),
                meta.codec
            ))),
            [meta] => Err(Error::Serialization(format!(
                "section {:?}: expected one raw chunk, found one chunk of kind {:?}",
                self.section_type(),
                meta.kind
            ))),
            chunks => Err(Error::Serialization(format!(
                "section {:?}: expected one raw chunk, found {} chunks",
                self.section_type(),
                chunks.len()
            ))),
        }
    }

    /// Whether this section has been modified since the last flush.
    fn is_dirty(&self) -> bool;

    /// Mark the section as clean after a successful flush.
    fn mark_clean(&self);

    /// Estimated memory usage of this section in bytes.
    fn memory_usage(&self) -> usize;

    /// Switch to a mmap-backed read mode using bytes from `fetcher`.
    ///
    /// Called by the spill path after the section has been serialized
    /// to a spill file and that file has been memory-mapped. The
    /// `fetcher` lifetime is tied to the `Arc`: the section should
    /// retain the `Arc` for as long as it serves reads from the mmap.
    ///
    /// Implementations use interior mutability to swap their backing
    /// storage. Eager-deserialize sections may decode `fetcher.fetch(0,
    /// fetcher.len())` into a fresh in-memory copy and keep the
    /// `fetcher` alive only for OS page-cache warmth; zero-copy
    /// sections (a future addition) read directly from the fetcher on
    /// demand.
    ///
    /// # Errors
    ///
    /// The default returns [`SpillError::NotSupported`]. Concrete
    /// sections override this to enable spill-to-disk; failures during
    /// the swap should be reported via [`SpillError::IoError`] or
    /// another appropriate variant.
    fn swap_to_mmap(&self, _fetcher: Arc<dyn PageFetcher>) -> std::result::Result<(), SpillError> {
        Err(SpillError::NotSupported)
    }

    /// Release any mmap-backed view and return to a fully in-memory
    /// representation.
    ///
    /// The default is a no-op (already in-memory). Sections that
    /// override [`swap_to_mmap`](Section::swap_to_mmap) should also
    /// override this to drop their `Arc<dyn PageFetcher>` and, if
    /// needed, deserialize from a saved buffer.
    ///
    /// # Errors
    ///
    /// Returns a [`SpillError`] if the reload fails (for example,
    /// because the spill file is no longer readable).
    fn reload_to_ram(&self) -> std::result::Result<(), SpillError> {
        Ok(())
    }
}

// ── Tier Override ───────────────────────────────────────────────────

/// Controls whether a section stays in RAM, on disk, or is auto-managed.
///
/// The default (`Auto`) lets the [`BufferManager`](crate::memory::buffer::BufferManager)
/// decide based on memory pressure. Power users can pin a section to a
/// specific tier for predictable performance.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
#[non_exhaustive]
pub enum TierOverride {
    /// Memory-first, spill to disk when budget exceeded (default).
    #[default]
    Auto,
    /// Always keep in RAM. Fail with error if insufficient memory.
    ForceRam,
    /// Always use disk (mmap). Minimal RAM footprint.
    ForceDisk,
}

/// Per-section memory configuration.
///
/// Allows power users to cap individual sections or pin them to a tier.
/// Most users leave this at default (all sections auto-managed within the
/// global memory budget).
#[derive(Debug, Clone)]
pub struct SectionMemoryConfig {
    /// Hard cap on this section's RAM usage (bytes).
    /// `None` means the section participates in the global budget with no
    /// per-section cap. The BufferManager decides when to spill.
    pub max_ram: Option<usize>,
    /// Storage tier override.
    pub tier: TierOverride,
}

impl Default for SectionMemoryConfig {
    fn default() -> Self {
        Self {
            max_ram: None,
            tier: TierOverride::Auto,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn section_type_from_u8_covers_every_variant_and_refuses_unknown_bytes() {
        for section_type in [
            SectionType::Catalog,
            SectionType::LpgStore,
            SectionType::RdfStore,
            SectionType::CompactStore,
            SectionType::OverlayDeletions,
            SectionType::VectorStore,
            SectionType::TextIndex,
            SectionType::RdfRing,
            SectionType::PropertyIndex,
        ] {
            let byte = u8::try_from(section_type as u32).unwrap();
            assert_eq!(SectionType::from_u8(byte), Some(section_type));
        }
        for byte in [0u8, 6, 9, 13, 19, 21, 250] {
            assert_eq!(SectionType::from_u8(byte), None, "byte {byte}");
        }
    }

    /// Every section type, in declaration order.
    const EVERY_SECTION_TYPE: [SectionType; 9] = [
        SectionType::Catalog,
        SectionType::LpgStore,
        SectionType::RdfStore,
        SectionType::CompactStore,
        SectionType::OverlayDeletions,
        SectionType::VectorStore,
        SectionType::TextIndex,
        SectionType::RdfRing,
        SectionType::PropertyIndex,
    ];

    /// Position of `section_type` in [`EVERY_SECTION_TYPE`]. The match has no
    /// wildcard, so a new variant fails to compile here until it is listed.
    fn listed_position(section_type: SectionType) -> usize {
        match section_type {
            SectionType::Catalog => 0,
            SectionType::LpgStore => 1,
            SectionType::RdfStore => 2,
            SectionType::CompactStore => 3,
            SectionType::OverlayDeletions => 4,
            SectionType::VectorStore => 5,
            SectionType::TextIndex => 6,
            SectionType::RdfRing => 7,
            SectionType::PropertyIndex => 8,
        }
    }

    #[test]
    fn a_section_type_byte_round_trips_exactly_when_it_is_known() {
        for (position, section_type) in EVERY_SECTION_TYPE.into_iter().enumerate() {
            assert_eq!(
                listed_position(section_type),
                position,
                "{section_type:?} is listed once"
            );
            let byte = section_type.to_u8();
            assert_eq!(SectionType::from_u8(byte), Some(section_type));
            assert_eq!(
                u32::from(byte),
                section_type as u32,
                "{section_type:?}: the byte is the discriminant"
            );
        }
        let mut known = 0;
        for byte in 0..=u8::MAX {
            let decoded = SectionType::from_u8(byte);
            assert_eq!(
                decoded.map(SectionType::to_u8) == Some(byte),
                decoded.is_some(),
                "byte {byte}"
            );
            known += usize::from(decoded.is_some());
        }
        assert_eq!(
            known,
            EVERY_SECTION_TYPE.len(),
            "exactly one byte per section type decodes"
        );
    }

    #[test]
    fn section_type_classification() {
        assert!(SectionType::Catalog.is_data_section());
        assert!(SectionType::LpgStore.is_data_section());
        assert!(SectionType::RdfStore.is_data_section());
        assert!(!SectionType::VectorStore.is_data_section());

        assert!(!SectionType::Catalog.is_index_section());
        assert!(SectionType::VectorStore.is_index_section());
        assert!(SectionType::TextIndex.is_index_section());
        assert!(SectionType::RdfRing.is_index_section());
        assert!(SectionType::PropertyIndex.is_index_section());
    }

    #[test]
    fn section_flags_roundtrip() {
        let flags = SectionFlags {
            required: true,
            mmap_able: false,
        };
        assert_eq!(flags.to_byte(), 0x01);
        assert_eq!(SectionFlags::from_byte(0x01), flags);

        let flags = SectionFlags {
            required: false,
            mmap_able: true,
        };
        assert_eq!(flags.to_byte(), 0x02);
        assert_eq!(SectionFlags::from_byte(0x02), flags);

        let flags = SectionFlags {
            required: true,
            mmap_able: true,
        };
        assert_eq!(flags.to_byte(), 0x03);
        assert_eq!(SectionFlags::from_byte(0x03), flags);

        let empty = SectionFlags::default();
        assert_eq!(empty.to_byte(), 0x00);
        assert_eq!(SectionFlags::from_byte(0x00), empty);
    }

    #[test]
    fn default_flags_by_type() {
        let catalog = SectionType::Catalog.default_flags();
        assert!(catalog.required);
        assert!(!catalog.mmap_able);

        let vector = SectionType::VectorStore.default_flags();
        assert!(!vector.required);
        assert!(vector.mmap_able);

        let rdf = SectionType::RdfStore.default_flags();
        assert!(!rdf.required);
        assert!(
            !rdf.mmap_able,
            "data sections must be deserialized, not mmap'd"
        );
    }

    #[test]
    fn directory_entry_size() {
        assert_eq!(SectionDirectoryEntry::SIZE, 32);
    }

    #[test]
    fn alix_tier_override_variants() {
        assert_eq!(TierOverride::Auto, TierOverride::default());
        // Verify all variants are distinct
        assert_ne!(TierOverride::Auto, TierOverride::ForceRam);
        assert_ne!(TierOverride::Auto, TierOverride::ForceDisk);
        assert_ne!(TierOverride::ForceRam, TierOverride::ForceDisk);
    }

    #[test]
    fn gus_section_memory_config_default() {
        let config = SectionMemoryConfig::default();
        assert!(config.max_ram.is_none());
        assert_eq!(config.tier, TierOverride::Auto);
    }

    #[test]
    fn vincent_section_memory_config_with_cap() {
        let config = SectionMemoryConfig {
            max_ram: Some(1024 * 1024),
            tier: TierOverride::ForceRam,
        };
        assert_eq!(config.max_ram, Some(1024 * 1024));
        assert_eq!(config.tier, TierOverride::ForceRam);
    }

    #[test]
    fn jules_force_disk_tier() {
        let config = SectionMemoryConfig {
            max_ram: None,
            tier: TierOverride::ForceDisk,
        };
        assert_eq!(config.tier, TierOverride::ForceDisk);
    }

    #[test]
    fn mia_lpg_store_default_flags_distinct_from_rdf() {
        let lpg = SectionType::LpgStore.default_flags();
        let rdf = SectionType::RdfStore.default_flags();
        // LpgStore is required, RdfStore is not
        assert!(lpg.required);
        assert!(!rdf.required);
        // Data sections must be deserialized into RAM, not mmap'd
        assert!(!lpg.mmap_able, "LpgStore is a data section, not mmap-able");
        assert!(!rdf.mmap_able, "RdfStore is a data section, not mmap-able");
    }

    #[test]
    fn butch_index_section_default_flags_all_variants() {
        // All index section types share the same flags
        for section_type in [
            SectionType::VectorStore,
            SectionType::TextIndex,
            SectionType::RdfRing,
            SectionType::PropertyIndex,
        ] {
            let flags = section_type.default_flags();
            assert!(!flags.required, "{section_type:?} should not be required");
            assert!(flags.mmap_able, "{section_type:?} should be mmap-able");
        }
    }

    #[test]
    fn django_directory_entry_construction() {
        let entry = SectionDirectoryEntry {
            section_type: SectionType::LpgStore,
            version: 1,
            flags: SectionFlags {
                required: true,
                mmap_able: false,
            },
            offset: 4096,
            length: 8192,
            checksum: 0xDEAD_BEEF,
        };
        assert_eq!(entry.section_type, SectionType::LpgStore);
        assert_eq!(entry.version, 1);
        assert!(entry.flags.required);
        assert!(!entry.flags.mmap_able);
        assert_eq!(entry.offset, 4096);
        assert_eq!(entry.length, 8192);
        assert_eq!(entry.checksum, 0xDEAD_BEEF);
    }

    #[test]
    fn shosanna_section_type_is_data_vs_index_boundary() {
        // Data sections: discriminant < 10
        assert!(SectionType::Catalog.is_data_section());
        assert!(!SectionType::Catalog.is_index_section());

        // Index sections: discriminant >= 10
        assert!(SectionType::VectorStore.is_index_section());
        assert!(!SectionType::VectorStore.is_data_section());

        // PropertyIndex at discriminant 20 is still an index section
        assert!(SectionType::PropertyIndex.is_index_section());
        assert!(!SectionType::PropertyIndex.is_data_section());
    }

    #[test]
    fn hans_section_flags_extra_bits_ignored() {
        // Bits beyond 0 and 1 are ignored by from_byte
        let flags = SectionFlags::from_byte(0xFF);
        assert!(flags.required);
        assert!(flags.mmap_able);

        let flags = SectionFlags::from_byte(0xFC);
        assert!(!flags.required);
        assert!(!flags.mmap_able);
    }

    #[test]
    fn beatrix_directory_entry_clone_eq() {
        let entry = SectionDirectoryEntry {
            section_type: SectionType::RdfRing,
            version: 2,
            flags: SectionFlags {
                required: false,
                mmap_able: true,
            },
            offset: 0,
            length: 1024,
            checksum: 42,
        };
        let cloned = entry.clone();
        assert_eq!(entry, cloned);
    }

    /// Minimal Section trait implementation for testing default methods.
    struct StubSection {
        dirty: bool,
    }

    impl Section for StubSection {
        fn section_type(&self) -> SectionType {
            SectionType::LpgStore
        }

        fn serialize(&self) -> crate::utils::error::Result<Vec<u8>> {
            Ok(vec![1, 2, 3])
        }

        fn deserialize(&mut self, _data: &[u8]) -> crate::utils::error::Result<()> {
            Ok(())
        }

        fn is_dirty(&self) -> bool {
            self.dirty
        }

        fn mark_clean(&self) {}

        fn memory_usage(&self) -> usize {
            64
        }
    }

    #[test]
    fn mia_section_trait_default_version() {
        let stub = StubSection { dirty: false };
        // The default version() method returns 1
        assert_eq!(stub.version(), 1);
        assert_eq!(stub.section_type(), SectionType::LpgStore);
        assert!(!stub.is_dirty());
        assert_eq!(stub.memory_usage(), 64);
    }

    #[test]
    fn butch_section_trait_serialize_deserialize() {
        let mut stub = StubSection { dirty: true };
        assert!(stub.is_dirty());

        let data = stub.serialize().unwrap();
        assert_eq!(data, vec![1, 2, 3]);

        stub.deserialize(&[4, 5, 6]).unwrap();
        stub.mark_clean();
    }

    struct Bytes3(Vec<u8>);

    impl Section for Bytes3 {
        fn section_type(&self) -> SectionType {
            SectionType::Catalog
        }
        fn serialize(&self) -> Result<Vec<u8>> {
            Ok(self.0.clone())
        }
        fn deserialize(&mut self, data: &[u8]) -> Result<()> {
            self.0 = data.to_vec();
            Ok(())
        }
        fn is_dirty(&self) -> bool {
            false
        }
        fn mark_clean(&self) {}
        fn memory_usage(&self) -> usize {
            self.0.len()
        }
    }

    #[derive(Default)]
    struct VecSink(Vec<(ChunkMeta, Vec<u8>)>);

    impl SectionSink for VecSink {
        fn write_chunk(&mut self, meta: ChunkMeta, bytes: &[u8]) -> Result<()> {
            self.0.push((meta, bytes.to_vec()));
            Ok(())
        }
    }

    /// Chunk descriptions, chunk bytes and the section version.
    struct VecSource(Vec<ChunkMeta>, Vec<Vec<u8>>, u8);

    impl SectionSource for VecSource {
        fn chunks(&self) -> &[ChunkMeta] {
            &self.0
        }
        fn fetch(&self, index: usize) -> Result<bytes::Bytes> {
            Ok(bytes::Bytes::copy_from_slice(&self.1[index]))
        }
        fn section_version(&self) -> u8 {
            self.2
        }
    }

    #[test]
    fn a_section_without_chunks_writes_one_raw_chunk_and_reads_it_back() {
        let mut sink = VecSink::default();
        Bytes3(b"Amsterdam".to_vec()).write_to(&mut sink).unwrap();
        assert_eq!(sink.0, [(ChunkMeta::raw(), b"Amsterdam".to_vec())]);
        let mut back = Bytes3(Vec::new());
        back.read_from(&VecSource(
            vec![ChunkMeta::raw()],
            vec![b"Amsterdam".to_vec()],
            1,
        ))
        .unwrap();
        assert_eq!(back.0, b"Amsterdam");
    }

    #[test]
    fn the_raw_adapter_refuses_anything_but_one_raw_chunk() {
        let mut section = Bytes3(Vec::new());
        let none = section
            .read_from(&VecSource(vec![], vec![], 1))
            .unwrap_err()
            .to_string();
        assert!(none.contains("expected one raw chunk"), "{none}");
        let two = VecSource(vec![ChunkMeta::raw(); 2], vec![vec![1], vec![2]], 1);
        let two = section.read_from(&two).unwrap_err().to_string();
        assert!(two.contains("expected one raw chunk"), "{two}");
    }

    #[test]
    fn the_raw_adapter_refuses_a_raw_chunk_with_a_codec() {
        let mut section = Bytes3(Vec::new());
        let coded = ChunkMeta {
            codec: 3,
            ..ChunkMeta::raw()
        };
        let error = section
            .read_from(&VecSource(vec![coded], vec![b"Prague".to_vec()], 1))
            .unwrap_err()
            .to_string();
        assert!(error.contains("codec 3"), "{error}");
        assert!(section.0.is_empty(), "nothing was deserialized");
    }

    #[test]
    fn every_chunk_kind_round_trips_through_its_byte() {
        for kind in [
            ChunkKind::Raw,
            ChunkKind::Meta,
            ChunkKind::Column,
            ChunkKind::History,
            ChunkKind::Stream,
        ] {
            assert_eq!(ChunkKind::from_byte(kind.to_byte()), Some(kind));
        }
        assert_eq!(ChunkKind::from_byte(5), None);
        assert_eq!(ChunkKind::from_byte(88), None);
    }

    #[test]
    fn chunk_constructors_put_each_argument_in_its_field() {
        let column = ChunkMeta::column(3, 19, 88, 7, 2);
        assert_eq!(
            (
                column.kind,
                column.graph_id,
                column.column_id,
                column.row_start,
                column.row_count,
                column.codec
            ),
            (ChunkKind::Column, 3, 19, 88, 7, 2)
        );
        let history = ChunkMeta::history(3, 19, 88, 7, 2);
        assert_eq!(
            history,
            ChunkMeta {
                kind: ChunkKind::History,
                ..column
            }
        );
        let piece = ChunkMeta::stream_piece(3, 19, 88);
        assert_eq!(
            (
                piece.kind,
                piece.graph_id,
                piece.column_id,
                piece.row_start,
                piece.row_count,
                piece.codec
            ),
            (ChunkKind::Stream, 3, 19, 88, 0, 0)
        );
        assert_eq!(
            ChunkMeta::meta(),
            ChunkMeta {
                kind: ChunkKind::Meta,
                ..ChunkMeta::raw()
            }
        );
        assert_eq!(column.identity(), (ChunkKind::Column.to_byte(), 3, 19, 88));
        assert_eq!(
            ChunkMeta::column(3, 19, 88, 1, 0).identity(),
            column.identity(),
            "the row count and the codec are not part of the identity"
        );
        assert_ne!(history.identity(), column.identity(), "the kind is");
    }

    /// Closes the deferred A1 item: the branch for a chunk of another kind.
    #[test]
    fn the_raw_adapter_refuses_a_chunk_of_another_kind() {
        let mut section = Bytes3(Vec::new());
        let error = section
            .read_from(&VecSource(
                vec![ChunkMeta::meta()],
                vec![b"Gus".to_vec()],
                1,
            ))
            .unwrap_err()
            .to_string();
        assert!(
            error.contains("expected one raw chunk") && error.contains("Meta"),
            "{error}"
        );
        assert!(section.0.is_empty(), "nothing was deserialized");
    }

    #[test]
    fn legacy_bytes_are_one_raw_chunk_and_nothing_else() {
        let one = VecSource(vec![ChunkMeta::raw()], vec![b"Alix".to_vec()], 0);
        assert_eq!(legacy_bytes(&one).unwrap().as_deref(), Some(&b"Alix"[..]));
        let chunked = VecSource(vec![ChunkMeta::meta()], vec![vec![1]], 3);
        assert_eq!(
            legacy_bytes(&chunked).unwrap(),
            None,
            "a chunked section is not 0.5.x bytes"
        );
        let two = VecSource(vec![ChunkMeta::raw(); 2], vec![vec![1], vec![2]], 0);
        assert!(legacy_bytes(&two).is_err());
        let mixed = VecSource(
            vec![ChunkMeta::raw(), ChunkMeta::meta()],
            vec![vec![1], vec![2]],
            0,
        );
        assert!(legacy_bytes(&mixed).is_err());
        let coded = VecSource(
            vec![ChunkMeta {
                codec: 3,
                ..ChunkMeta::raw()
            }],
            vec![b"Paris".to_vec()],
            0,
        );
        let error = legacy_bytes(&coded).unwrap_err().to_string();
        assert!(error.contains("codec 3"), "{error}");
        for (case, placed) in [
            (
                "graph 3",
                ChunkMeta {
                    graph_id: 3,
                    ..ChunkMeta::raw()
                },
            ),
            (
                "column 19",
                ChunkMeta {
                    column_id: 19,
                    ..ChunkMeta::raw()
                },
            ),
            (
                "first row 88",
                ChunkMeta {
                    row_start: 88,
                    ..ChunkMeta::raw()
                },
            ),
            (
                "rows 19",
                ChunkMeta {
                    row_count: 19,
                    ..ChunkMeta::raw()
                },
            ),
        ] {
            let source = VecSource(vec![placed], vec![b"Paris".to_vec()], 0);
            let error = legacy_bytes(&source).map_or_else(
                |error| error.to_string(),
                |bytes| format!("accepted {bytes:?}"),
            );
            assert!(error.contains(case), "{case}: {error}");
        }
    }

    #[test]
    fn check_version_names_the_section_and_both_versions() {
        let source = VecSource(vec![ChunkMeta::meta()], vec![vec![1]], 2);
        check_version(SectionType::Catalog, &source, 2).unwrap();
        let error = check_version(SectionType::LpgStore, &source, 3)
            .unwrap_err()
            .to_string();
        assert!(
            error.contains("LpgStore")
                && error.contains("version 2")
                && error.contains("version 3"),
            "{error}"
        );
    }
}
