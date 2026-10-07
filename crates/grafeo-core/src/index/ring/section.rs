//! Ring Index section for `.grafeo` container persistence.
//!
//! Writes and reads the [`super::TripleRing`] through the [`Section`]
//! trait, so the ring survives a restart without being rebuilt from the
//! triples.
//!
//! ## Versions
//!
//! - **3 (current, written by [`Section::write_to`]):** a metadata chunk
//!   (bincode of `RingMeta`: the layout byte, the byte cap the section was
//!   written with, the number of triples), then six byte streams of graph 0,
//!   each cut into [`ChunkKind::Stream`] pieces of at most the cap: stream 0
//!   the packed term dictionary, streams 1 to 3 the packed wavelet trees of
//!   the subjects, predicates and objects, streams 4 and 5 the packed
//!   permutations from SPO to POS and to OSP order. Each stream holds the
//!   bytes of one part of the version 2 envelope (see
//!   [`super::packed_format`]), without the envelope. The term dictionary is
//!   packed whole before it is written (the ring keeps its terms in a heap
//!   dictionary); the wavelet trees and permutations are written straight
//!   from the ring. A store without a ring writes no chunk.
//! - **2 (0.5.x):** the four packed sub-formats composed under a `GRFR`
//!   envelope with a CRC32 trailer, as one raw chunk. [`Section::serialize`]
//!   still writes it, for the spill path, whose reads are mmap-friendly via
//!   `Bytes::from_owner` + per-level `BitVector::from_mmap`.
//! - **1 (0.5.x):** bincode, detected by the absence of the `GRFR` magic at
//!   offset 0; data flows through `TripleRing::load_from_bytes`.
//!
//! [`Section::read_from`] reads version 3, and passes the one raw chunk of
//! versions 1 and 2 to [`Section::deserialize`]. The next checkpoint writes
//! version 3.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use bytes::Bytes;
use serde::{Deserialize, Serialize};

use grafeo_common::memory::buffer::SpillError;
use grafeo_common::storage::chunk::{ChunkCaps, ChunkStreamWriter, read_stream, stream_error};
use grafeo_common::storage::page_fetcher::PageFetcher;
use grafeo_common::storage::section::{
    ChunkKind, ChunkMeta, Section, SectionSink, SectionSource, SectionType, check_version,
    legacy_bytes,
};
use grafeo_common::utils::error::{Error, Result};

use super::{PackedTermDictionary, write_permutation, write_wavelet_tree};
use crate::graph::rdf::RdfStore;

/// The version of the section this build writes: 1 was bincode, 2 the
/// packed envelope as one raw chunk, 3 a metadata chunk and six streams.
pub(crate) const RING_SECTION_VERSION: u8 = 3;

/// The layout byte of the metadata chunk.
const RING_LAYOUT: u8 = 1;

/// First 4 bytes of the v2 envelope; absent in v1 bincode output.
const V2_MAGIC: &[u8; 4] = b"GRFR";

/// The parts of the ring in the order of their streams: stream `i` holds
/// part `PARTS[i]`.
const PARTS: [&str; 6] = [
    "dictionary",
    "subjects",
    "predicates",
    "objects",
    "spo_to_pos",
    "spo_to_osp",
];

/// The metadata chunk of a version 3 ring section.
#[derive(Debug, Serialize, Deserialize)]
struct RingMeta {
    /// The layout of the section: [`RING_LAYOUT`].
    layout: u8,
    /// The byte cap of the stream pieces when the section was written.
    max_bytes: u32,
    /// The number of triples of the ring.
    num_triples: u64,
}

/// Section implementation for the RDF Ring Index.
///
/// Wraps an `Arc<RdfStore>` and writes its ring as the metadata chunk and
/// six streams of version 3, cut into pieces of at most the section's caps.
pub struct RdfRingSection {
    store: Arc<RdfStore>,
    caps: ChunkCaps,
    dirty: AtomicBool,
}

impl RdfRingSection {
    /// Creates a new Ring section backed by the given RDF store, writing
    /// with the caps of this thread ([`ChunkCaps::current`]).
    #[must_use]
    pub fn new(store: Arc<RdfStore>) -> Self {
        Self::with_caps(store, ChunkCaps::current())
    }

    /// Creates a new Ring section backed by the given RDF store, writing
    /// stream pieces of at most `caps.max_bytes`.
    #[must_use]
    pub fn with_caps(store: Arc<RdfStore>, caps: ChunkCaps) -> Self {
        Self {
            store,
            caps,
            dirty: AtomicBool::new(false),
        }
    }

    /// Marks the section as dirty (Ring was rebuilt or invalidated).
    pub fn mark_dirty(&self) {
        self.dirty.store(true, Ordering::Release);
    }
}

/// Writes one part of the ring as stream `stream` of graph 0, through
/// `write`.
///
/// An error of the sink comes back as the sink returned it, with its
/// variant; any other write error names the section.
fn write_stream(
    sink: &mut dyn SectionSink,
    caps: ChunkCaps,
    stream: u32,
    write: impl FnOnce(&mut dyn std::io::Write) -> std::io::Result<()>,
) -> Result<()> {
    let mut writer = ChunkStreamWriter::new(sink, 0, stream, caps);
    let written = write(&mut writer);
    // The writer keeps the first error of the sink, which `finish` returns
    // as it was; it writes the last piece only when nothing failed.
    let finished = writer.finish();
    match (written, finished) {
        (Ok(()), finished) => finished.map(|_| ()),
        (Err(_), Err(kept)) => Err(kept),
        (Err(error), Ok(_)) => Err(stream_error(SectionType::RdfRing, error)),
    }
}

/// Checks the chunks of a version 3 section: the metadata chunk first, then
/// only pieces of streams 0 to 5 of graph 0.
fn check_chunks(chunks: &[ChunkMeta]) -> Result<()> {
    match chunks.first() {
        Some(first) if *first == ChunkMeta::meta() => {}
        Some(first) => {
            return Err(Error::Serialization(format!(
                "section RdfRing: the first chunk is a {:?} chunk of graph {}, column {}, not \
                 the metadata chunk",
                first.kind, first.graph_id, first.column_id
            )));
        }
        None => {
            return Err(Error::Serialization(
                "section RdfRing: no metadata chunk, the section has no chunks".to_string(),
            ));
        }
    }
    for (index, meta) in chunks.iter().enumerate().skip(1) {
        if meta.kind != ChunkKind::Stream {
            return Err(Error::Serialization(format!(
                "section RdfRing: chunk {index} is a {:?} chunk; after its metadata chunk a ring \
                 section holds only pieces of streams 0 to 5 of graph 0",
                meta.kind
            )));
        }
        if meta.graph_id != 0 || meta.column_id as usize >= PARTS.len() {
            return Err(Error::Serialization(format!(
                "section RdfRing: chunk {index} is a piece of stream {} of graph {}; a ring \
                 section holds streams 0 to 5 of graph 0",
                meta.column_id, meta.graph_id
            )));
        }
        if meta.row_count != 0 {
            return Err(Error::Serialization(format!(
                "section RdfRing: chunk {index}, a piece of stream {}, has {} rows; stream \
                 pieces have none",
                meta.column_id, meta.row_count
            )));
        }
    }
    Ok(())
}

/// The most bytes bincode may read for the metadata chunk. `RingMeta` holds
/// no length prefix, so a crafted chunk cannot make bincode allocate; the
/// limit holds for the metadata of every stream section all the same. The
/// bytes written with it are those of `standard()`.
const META_LIMIT: usize = 1 << 24;

/// Decodes the metadata chunk: bincode of `RingMeta` with nothing after it,
/// in this build's layout, reading at most [`META_LIMIT`] bytes.
fn decode_meta(bytes: &[u8]) -> Result<RingMeta> {
    let config = bincode::config::standard().with_limit::<META_LIMIT>();
    let (meta, used): (RingMeta, usize) = bincode::serde::decode_from_slice(bytes, config)
        .map_err(|error| {
            Error::Serialization(format!(
                "section RdfRing: the metadata chunk does not decode: {error}"
            ))
        })?;
    if used != bytes.len() {
        return Err(Error::Serialization(format!(
            "section RdfRing: the metadata chunk holds {} bytes, its record {used}",
            bytes.len()
        )));
    }
    if meta.layout != RING_LAYOUT {
        return Err(Error::Serialization(format!(
            "section RdfRing: layout {}, this build reads layout {RING_LAYOUT}",
            meta.layout
        )));
    }
    Ok(meta)
}

impl Section for RdfRingSection {
    fn section_type(&self) -> SectionType {
        SectionType::RdfRing
    }

    fn version(&self) -> u8 {
        RING_SECTION_VERSION
    }

    /// The version 2 envelope (see the module doc), for the spill path.
    fn serialize(&self) -> Result<Vec<u8>> {
        match self.store.ring() {
            Some(ring) => Ok(super::serialize_triple_ring(&ring)),
            None => Ok(Vec::new()),
        }
    }

    /// Reads the bytes of versions 1 and 2 (see the module doc).
    fn deserialize(&mut self, data: &[u8]) -> Result<()> {
        if data.is_empty() {
            return Ok(());
        }
        // Phase 6g: detect v2 packed vs v1 bincode by magic bytes.
        let ring = if data.len() >= 4 && &data[0..4] == V2_MAGIC {
            super::deserialize_triple_ring(bytes::Bytes::copy_from_slice(data))
                .map_err(|e| Error::Serialization(e.to_string()))?
        } else {
            // v1 fallback: bincode-encoded TripleRing. Existing files keep
            // loading; the next checkpoint flushes them out as v2.
            super::TripleRing::load_from_bytes(data)
                .map_err(|e| Error::Serialization(e.to_string()))?
        };
        self.store.set_ring(ring);
        Ok(())
    }

    /// Writes version 3: the metadata chunk, then streams 0 to 5, one after
    /// the other (see the module doc). A store without a ring writes no
    /// chunk, so the image holds no ring section.
    ///
    /// # Errors
    ///
    /// Returns an error when the caps are 0, a part cannot be encoded, or
    /// the sink refuses a chunk.
    fn write_to(&self, sink: &mut dyn SectionSink) -> Result<()> {
        let Some(ring) = self.store.ring() else {
            return Ok(());
        };
        self.caps.validate()?;
        let meta = RingMeta {
            layout: RING_LAYOUT,
            max_bytes: self.caps.max_bytes,
            num_triples: ring.len() as u64,
        };
        let meta =
            bincode::serde::encode_to_vec(&meta, bincode::config::standard()).map_err(|error| {
                Error::Serialization(format!(
                    "section RdfRing: the metadata chunk does not encode: {error}"
                ))
            })?;
        sink.write_chunk(ChunkMeta::meta(), &meta)?;

        // The ring keeps its terms in a heap dictionary: the packed
        // dictionary is built whole, written, and dropped.
        let dictionary = PackedTermDictionary::from_term_dict(ring.dictionary());
        write_stream(sink, self.caps, 0, |out| dictionary.write_to(out))?;
        drop(dictionary);
        write_stream(sink, self.caps, 1, |out| {
            write_wavelet_tree(ring.subjects_wt(), out)
        })?;
        write_stream(sink, self.caps, 2, |out| {
            write_wavelet_tree(ring.predicates_wt(), out)
        })?;
        write_stream(sink, self.caps, 3, |out| {
            write_wavelet_tree(ring.objects_wt(), out)
        })?;
        write_stream(sink, self.caps, 4, |out| {
            write_permutation(ring.spo_to_pos_perm(), out)
        })?;
        write_stream(sink, self.caps, 5, |out| {
            write_permutation(ring.spo_to_osp_perm(), out)
        })
    }

    /// Reads version 3, or the one raw chunk of versions 1 and 2. Each
    /// stream is read into one buffer, which becomes its part's storage; the
    /// ring is set on the store only once all six parts are read, so a
    /// section that fails leaves the store's ring as it was.
    ///
    /// # Errors
    ///
    /// Returns an error when a chunk cannot be fetched, the section has
    /// another version, holds chunks a ring section does not hold, lacks a
    /// stream, or a part does not decode.
    fn read_from(&mut self, source: &dyn SectionSource) -> Result<()> {
        if let Some(bytes) = legacy_bytes(source)? {
            return self.deserialize(&bytes);
        }
        check_version(SectionType::RdfRing, source, RING_SECTION_VERSION)?;
        check_chunks(source.chunks())?;
        let meta = decode_meta(&source.fetch(0)?)?;
        let num_triples = usize::try_from(meta.num_triples).map_err(|_| {
            Error::Serialization(format!(
                "section RdfRing: {} triples do not fit this platform",
                meta.num_triples
            ))
        })?;
        let mut parts: Vec<Bytes> = Vec::with_capacity(PARTS.len());
        for (stream, part) in (0u32..).zip(PARTS) {
            let bytes = read_stream(source, 0, stream).map_err(|error| {
                stream_error(SectionType::RdfRing, std::io::Error::other(error))
            })?;
            if bytes.is_empty() {
                return Err(Error::Serialization(format!(
                    "section RdfRing: stream {stream} ({part}) is missing"
                )));
            }
            parts.push(bytes);
        }
        let [
            dictionary,
            subjects,
            predicates,
            objects,
            spo_to_pos,
            spo_to_osp,
        ] = <[Bytes; 6]>::try_from(parts).map_err(|parts| {
            Error::Internal(format!(
                "section RdfRing: {} parts read, the ring has six",
                parts.len()
            ))
        })?;
        let ring = super::assemble_triple_ring(
            num_triples,
            dictionary,
            subjects,
            predicates,
            objects,
            spo_to_pos,
            spo_to_osp,
        )
        .map_err(|error| Error::Serialization(format!("section RdfRing: {error}")))?;
        self.store.set_ring(ring);
        Ok(())
    }

    fn is_dirty(&self) -> bool {
        self.dirty.load(Ordering::Acquire)
    }

    fn mark_clean(&self) {
        self.dirty.store(false, Ordering::Release);
    }

    fn memory_usage(&self) -> usize {
        self.store.ring().map_or(0, |r| r.size_bytes())
    }

    /// Swaps the ring backing to a `Bytes` view sourced from `fetcher`.
    ///
    /// Phase 6 deferred → Phase 8 audit-fix: closes the loop on the v2
    /// packed Ring format. After the section serializes to a spill file
    /// and the file is mmap'd, the buffer manager calls this with a
    /// `MmapPageFetcher`. We copy the section bytes into a single
    /// owning `Bytes`; the v2 deserializer then constructs the ring with
    /// every bulk component (term dictionary, wavelet level bitvectors,
    /// permutation forward arrays) refcount-sharing slices of that
    /// `Bytes`. Reads thereafter are zero-copy against the shared buffer.
    ///
    /// The one allocation here is `~section_size` once, replacing the
    /// previous eager-deserialize that allocated `~3x` that for the
    /// reconstructed `HashMap`s. A future `PageFetcher::owned_bytes`
    /// override on the mmap impl could drop that copy too.
    ///
    /// v1 bincode buffers fall through to the bincode `load_from_bytes`
    /// path so legacy spill files still load.
    fn swap_to_mmap(&self, fetcher: Arc<dyn PageFetcher>) -> std::result::Result<(), SpillError> {
        let len = fetcher.len();
        if len == 0 {
            return Ok(());
        }

        let slice = fetcher
            .fetch(0, len)
            .map_err(|e| SpillError::IoError(e.to_string()))?;
        let data = bytes::Bytes::copy_from_slice(slice);

        let ring = if data.len() >= 4 && &data[0..4] == V2_MAGIC {
            super::deserialize_triple_ring(data).map_err(|e| SpillError::IoError(e.to_string()))?
        } else {
            super::TripleRing::load_from_bytes(&data)
                .map_err(|e| SpillError::IoError(e.to_string()))?
        };

        self.store.set_ring(ring);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::rdf::{Term, Triple};

    fn test_store() -> Arc<RdfStore> {
        let store = Arc::new(RdfStore::new());
        store.bulk_load(vec![
            Triple::new(
                Term::iri("http://ex.org/alix"),
                Term::iri("http://xmlns.com/foaf/0.1/name"),
                Term::literal("Alix"),
            ),
            Triple::new(
                Term::iri("http://ex.org/gus"),
                Term::iri("http://xmlns.com/foaf/0.1/name"),
                Term::literal("Gus"),
            ),
            Triple::new(
                Term::iri("http://ex.org/alix"),
                Term::iri("http://xmlns.com/foaf/0.1/knows"),
                Term::iri("http://ex.org/gus"),
            ),
        ]);
        store
    }

    #[test]
    fn section_type_is_rdf_ring() {
        let store = test_store();
        let section = RdfRingSection::new(store);
        assert_eq!(section.section_type(), SectionType::RdfRing);
        // 1 was bincode, 2 the packed envelope as one raw chunk, 3 streams.
        assert_eq!(section.version(), RING_SECTION_VERSION);
        assert_eq!(RING_SECTION_VERSION, 3);
    }

    #[test]
    fn section_dirty_tracking() {
        let store = test_store();
        let section = RdfRingSection::new(store);
        assert!(!section.is_dirty());
        section.mark_dirty();
        assert!(section.is_dirty());
        section.mark_clean();
        assert!(!section.is_dirty());
    }

    #[test]
    fn section_serialize_empty() {
        let store = Arc::new(RdfStore::new());
        let section = RdfRingSection::new(store);
        let bytes = section.serialize().unwrap();
        assert!(bytes.is_empty(), "{bytes:?}");
    }

    #[test]
    fn section_roundtrip() {
        let store = test_store();
        let section = RdfRingSection::new(Arc::clone(&store));

        // Serialize
        let bytes = section.serialize().unwrap();
        assert!(!bytes.is_empty(), "bytes is empty");

        // Create a fresh store and deserialize into it
        let store2 = Arc::new(RdfStore::new());
        let mut section2 = RdfRingSection::new(Arc::clone(&store2));
        section2.deserialize(&bytes).unwrap();

        // The loaded ring should have the same triple count
        let ring = store2.ring().expect("ring should be loaded");
        assert_eq!(ring.len(), 3);

        // Verify count operations work
        use crate::graph::rdf::TriplePattern;
        let name_pattern = TriplePattern {
            subject: None,
            predicate: Some(Term::iri("http://xmlns.com/foaf/0.1/name")),
            object: None,
        };
        assert_eq!(ring.count(&name_pattern), 2);
    }

    #[test]
    fn section_memory_usage() {
        let store = test_store();
        let section = RdfRingSection::new(store);
        assert!(section.memory_usage() > 0);
    }

    // ── Phase 6g: format detection + v1 → v2 migration ───────────────

    /// New writes produce a v2 buffer (starts with `GRFR` magic).
    #[test]
    fn alix_section_serialize_writes_v2_magic() {
        let store = test_store();
        let section = RdfRingSection::new(store);
        let bytes = section.serialize().unwrap();
        assert!(bytes.len() > 4);
        assert_eq!(&bytes[0..4], V2_MAGIC, "new writes must use v2 magic");
    }

    /// v1 bincode-encoded buffers still deserialize correctly (one-release
    /// fallback). The check uses save_to_bytes which produces v1 format
    /// directly — guaranteeing the migration path works for files written
    /// by older Grafeo versions.
    #[test]
    fn gus_section_v1_bincode_buffer_still_loads() {
        let original = test_store();
        let ring = original.ring().expect("ring built").as_ref().clone();
        // Encode as v1 bincode directly (bypass the section entry point).
        let v1_bytes = ring.save_to_bytes().unwrap();
        // Sanity: v1 bytes do NOT start with GRFR.
        assert_ne!(
            &v1_bytes[0..4],
            V2_MAGIC,
            "v1 bincode must not have GRFR magic"
        );

        // Deserialize via the section: should detect v1 and use the
        // bincode path.
        let store2 = Arc::new(RdfStore::new());
        let mut section2 = RdfRingSection::new(Arc::clone(&store2));
        section2.deserialize(&v1_bytes).unwrap();
        let restored = store2.ring().expect("ring loaded");
        assert_eq!(restored.len(), 3);
    }

    // ── Phase 6/8 audit-fix: swap_to_mmap end-to-end ──────────────────

    /// Minimal in-memory PageFetcher for testing swap_to_mmap.
    struct MemFetcher(Vec<u8>);

    impl PageFetcher for MemFetcher {
        fn fetch(&self, offset: usize, len: usize) -> std::io::Result<&[u8]> {
            let end = offset
                .checked_add(len)
                .ok_or_else(|| std::io::Error::other("overflow"))?;
            if end > self.0.len() {
                return Err(std::io::Error::from(std::io::ErrorKind::UnexpectedEof));
            }
            Ok(&self.0[offset..end])
        }

        fn len(&self) -> usize {
            self.0.len()
        }

        fn advise(
            &self,
            _offset: usize,
            _len: usize,
            _hint: grafeo_common::storage::page_fetcher::AccessHint,
        ) {
        }
    }

    /// `swap_to_mmap` rebuilds the ring from a `Bytes`-backed v2 buffer
    /// and queries against the swapped-in ring give correct results.
    #[test]
    fn shosanna_swap_to_mmap_serves_queries_from_bytes() {
        let original = test_store();
        let v2_bytes = {
            let section = RdfRingSection::new(Arc::clone(&original));
            section.serialize().unwrap()
        };

        // Fresh empty store; section starts with no ring.
        let store = Arc::new(RdfStore::new());
        assert!(store.ring().is_none());

        let section = RdfRingSection::new(Arc::clone(&store));
        let fetcher: Arc<dyn PageFetcher> = Arc::new(MemFetcher(v2_bytes));
        section.swap_to_mmap(fetcher).expect("swap_to_mmap");

        // Ring is now populated from the fetcher bytes.
        let ring = store.ring().expect("ring loaded via swap_to_mmap");
        assert_eq!(ring.len(), 3);

        // Query semantics still work against the post-swap ring.
        use crate::graph::rdf::TriplePattern;
        let name_pattern = TriplePattern {
            subject: None,
            predicate: Some(Term::iri("http://xmlns.com/foaf/0.1/name")),
            object: None,
        };
        assert_eq!(ring.count(&name_pattern), 2);
    }

    /// Empty fetcher (zero-length section) is a no-op, not an error.
    #[test]
    fn butch_swap_to_mmap_empty_fetcher_is_noop() {
        let store = Arc::new(RdfStore::new());
        let section = RdfRingSection::new(Arc::clone(&store));
        let fetcher: Arc<dyn PageFetcher> = Arc::new(MemFetcher(Vec::new()));
        section.swap_to_mmap(fetcher).expect("empty swap is ok");
        assert!(store.ring().is_none());
    }

    /// v1 bincode-encoded fetcher bytes still load via the legacy fallback.
    #[test]
    fn django_swap_to_mmap_v1_bincode_fetcher_still_loads() {
        let original = test_store();
        let ring = original.ring().expect("ring built").as_ref().clone();
        let v1_bytes = ring.save_to_bytes().unwrap();
        assert_ne!(&v1_bytes[0..4], V2_MAGIC);

        let store = Arc::new(RdfStore::new());
        let section = RdfRingSection::new(Arc::clone(&store));
        let fetcher: Arc<dyn PageFetcher> = Arc::new(MemFetcher(v1_bytes));
        section.swap_to_mmap(fetcher).expect("v1 fallback in swap");

        assert_eq!(store.ring().unwrap().len(), 3);
    }

    /// After a v1 read + a re-serialize, the new buffer is v2.
    /// Demonstrates the on-checkpoint migration.
    #[test]
    fn vincent_section_v1_then_resersialize_yields_v2() {
        let original = test_store();
        let ring = original.ring().expect("ring built").as_ref().clone();
        let v1_bytes = ring.save_to_bytes().unwrap();

        let store2 = Arc::new(RdfStore::new());
        let mut section2 = RdfRingSection::new(Arc::clone(&store2));
        section2.deserialize(&v1_bytes).unwrap();

        // Re-serialize: now in v2.
        let v2_bytes = section2.serialize().unwrap();
        assert_eq!(&v2_bytes[0..4], V2_MAGIC, "post-migration write is v2");

        // And v2 round-trips cleanly.
        let store3 = Arc::new(RdfStore::new());
        let mut section3 = RdfRingSection::new(Arc::clone(&store3));
        section3.deserialize(&v2_bytes).unwrap();
        assert_eq!(store3.ring().unwrap().len(), 3);
    }

    // ── Version 3: the ring in streams ───────────────────────────────

    use std::io::Write;

    use grafeo_common::storage::{ImageSource, MemoryImage};

    use crate::graph::rdf::TriplePattern;
    use crate::index::ring::{
        TripleRing, serialize_permutation, serialize_triple_ring, serialize_wavelet_tree,
    };

    /// Caps small enough that every stream of a 500-triple ring spans
    /// several pieces.
    const TINY: ChunkCaps = ChunkCaps {
        max_rows: 3,
        max_bytes: 64,
    };

    /// 500 distinct triples: 95 subjects, four predicates, objects that are
    /// IRIs, typed literals and literals with a language.
    fn five_hundred_triples() -> Vec<Triple> {
        let people = ["alix", "gus", "vincent", "mia", "jules"];
        let cities = ["amsterdam", "berlin", "paris", "prague"];
        (0..500usize)
            .map(|i| {
                let subject = Term::iri(format!("http://example.org/{}/{}", people[i % 5], i % 19));
                let (predicate, object) = match i % 3 {
                    0 => (
                        "knows",
                        Term::iri(format!(
                            "http://example.org/{}/{}",
                            people[(i / 3) % 5],
                            i % 88
                        )),
                    ),
                    1 => (
                        "lives_in",
                        Term::iri(format!("http://example.org/{}", cities[i % 4])),
                    ),
                    _ if i % 2 == 0 => (
                        "age",
                        Term::typed_literal(
                            i.to_string(),
                            "http://www.w3.org/2001/XMLSchema#integer",
                        ),
                    ),
                    _ => (
                        "motto",
                        Term::lang_literal(format!("{} {i}", cities[i % 4]), "nl"),
                    ),
                };
                Triple::new(
                    subject,
                    Term::iri(format!("http://example.org/{predicate}")),
                    object,
                )
            })
            .collect()
    }

    /// A store holding [`five_hundred_triples`] and the ring built from them.
    fn ringed_store() -> Arc<RdfStore> {
        let store = Arc::new(RdfStore::new());
        store.bulk_load(five_hundred_triples());
        assert_eq!(store.ring().expect("bulk_load builds the ring").len(), 500);
        store
    }

    /// The image `section` writes, as a checkpoint writes it.
    fn image_of(section: &RdfRingSection) -> MemoryImage {
        MemoryImage::from_sections(&[section as &dyn Section]).unwrap()
    }

    /// The ring section of `image`, read into a new store.
    fn read_back(image: &MemoryImage) -> Result<Arc<RdfStore>> {
        let store = Arc::new(RdfStore::new());
        let source = image
            .section_source(SectionType::RdfRing)
            .expect("the image has a ring section");
        RdfRingSection::new(Arc::clone(&store)).read_from(&*source)?;
        Ok(store)
    }

    /// The chunks of the ring section of `image`, with their bytes.
    fn chunks_of(image: &MemoryImage) -> Vec<(ChunkMeta, Vec<u8>)> {
        let source = image.section_source(SectionType::RdfRing).unwrap();
        (0..source.chunks().len())
            .map(|index| {
                (
                    source.chunks()[index],
                    source.fetch(index).unwrap().to_vec(),
                )
            })
            .collect()
    }

    /// A ring section of `version` holding exactly `chunks`.
    fn crafted(version: u8, chunks: &[(ChunkMeta, Vec<u8>)]) -> MemoryImage {
        let mut image = MemoryImage::new();
        image.begin_section(SectionType::RdfRing, version).unwrap();
        for (meta, bytes) in chunks {
            image.write_chunk(*meta, bytes).unwrap();
        }
        image
    }

    /// Writes part `part` of `ring` (`dictionary` is its packed dictionary)
    /// to `out`, in the order of the streams.
    fn write_part(
        ring: &TripleRing,
        dictionary: &PackedTermDictionary,
        part: u32,
        out: &mut dyn Write,
    ) -> std::io::Result<()> {
        match part {
            0 => dictionary.write_to(out),
            1 => write_wavelet_tree(ring.subjects_wt(), out),
            2 => write_wavelet_tree(ring.predicates_wt(), out),
            3 => write_wavelet_tree(ring.objects_wt(), out),
            4 => write_permutation(ring.spo_to_pos_perm(), out),
            5 => write_permutation(ring.spo_to_osp_perm(), out),
            _ => unreachable!("the ring has six parts"),
        }
    }

    /// Asserts that `restored` answers as `original` does: the same triple
    /// at every position, the same terms, and the same count for every
    /// pattern shape over the terms of every triple, and of absent terms.
    fn assert_same_ring(original: &TripleRing, restored: &TripleRing) {
        assert_eq!(restored.len(), original.len(), "triples");
        assert_eq!(restored.num_terms(), original.num_terms(), "terms");
        let mut terms: Vec<(Term, Term, Term)> = Vec::with_capacity(original.len() + 1);
        for index in 0..original.len() {
            let triple = original.get_spo(index).expect("a triple at every position");
            assert_eq!(
                restored.get_spo(index).as_ref(),
                Some(&triple),
                "triple at SPO position {index}"
            );
            assert_eq!(
                (restored.spo_to_pos(index), restored.spo_to_osp(index)),
                (original.spo_to_pos(index), original.spo_to_osp(index)),
                "POS and OSP positions of SPO position {index}"
            );
            terms.push(triple.into_parts());
        }
        let absent = Term::iri("http://example.org/berlin/88");
        terms.push((absent.clone(), absent.clone(), absent));
        for (subject, predicate, object) in &terms {
            for shape in 0..8u8 {
                let pattern = TriplePattern {
                    subject: (shape & 1 != 0).then(|| subject.clone()),
                    predicate: (shape & 2 != 0).then(|| predicate.clone()),
                    object: (shape & 4 != 0).then(|| object.clone()),
                };
                assert_eq!(
                    restored.count(&pattern),
                    original.count(&pattern),
                    "count of {pattern:?}"
                );
            }
        }
    }

    /// Each part's writer gives its serializer's bytes, written into one
    /// buffer or cut into stream pieces, and the six streams the section
    /// writes are the six parts of the 0.5.x envelope, in order.
    #[test]
    fn the_streamed_parts_equal_the_serialized_ones() {
        let ring = TripleRing::from_triples(five_hundred_triples().into_iter());
        let dictionary = PackedTermDictionary::from_term_dict(ring.dictionary());
        let serialized = [
            dictionary.to_bytes(),
            serialize_wavelet_tree(ring.subjects_wt()),
            serialize_wavelet_tree(ring.predicates_wt()),
            serialize_wavelet_tree(ring.objects_wt()),
            serialize_permutation(ring.spo_to_pos_perm()),
            serialize_permutation(ring.spo_to_osp_perm()),
        ];
        let mut pieces = MemoryImage::new();
        pieces.begin_section(SectionType::RdfRing, 3).unwrap();
        for (part, expected) in (0u32..).zip(&serialized) {
            let mut buffer = Vec::new();
            write_part(&ring, &dictionary, part, &mut buffer).unwrap();
            assert!(buffer == *expected, "part {part} written into one buffer");
            let mut writer = ChunkStreamWriter::new(&mut pieces, 0, part, TINY);
            write_part(&ring, &dictionary, part, &mut writer).unwrap();
            assert_eq!(
                writer.finish().unwrap(),
                expected.len() as u64,
                "part {part}"
            );
        }
        let source = pieces.section_source(SectionType::RdfRing).unwrap();
        for (part, expected) in (0u32..).zip(&serialized) {
            assert!(
                read_stream(&*source, 0, part).unwrap() == expected[..],
                "part {part} cut into pieces"
            );
        }

        let store = Arc::new(RdfStore::new());
        store.set_ring(ring.clone());
        let image = image_of(&RdfRingSection::with_caps(store, TINY));
        let source = image.section_source(SectionType::RdfRing).unwrap();
        let envelope = serialize_triple_ring(&ring);
        let mut body = Vec::new();
        for (part, expected) in (0u32..).zip(&serialized) {
            let stream = read_stream(&*source, 0, part).unwrap();
            assert!(stream == expected[..], "stream {part} of the section");
            body.extend_from_slice(&stream);
        }
        assert!(
            body == envelope[64..envelope.len() - 4],
            "the streams, one after the other, are the body of the 0.5.x envelope"
        );
    }

    /// A ring written with small caps reads back into a ring that answers
    /// every pattern as the original does; the section is a metadata chunk
    /// recording the caps and the triple count, then the six streams, each
    /// in several pieces.
    #[test]
    fn a_ring_round_trips_through_streams() {
        let store = ringed_store();
        let image = image_of(&RdfRingSection::with_caps(Arc::clone(&store), TINY));
        let chunks = chunks_of(&image);
        assert_eq!(
            chunks[0].0,
            ChunkMeta::meta(),
            "the metadata chunk comes first"
        );
        let (meta, used): (RingMeta, usize) =
            bincode::serde::decode_from_slice(&chunks[0].1, bincode::config::standard()).unwrap();
        assert_eq!(used, chunks[0].1.len());
        assert_eq!(
            (meta.layout, meta.max_bytes, meta.num_triples),
            (1, TINY.max_bytes, 500),
            "layout, caps and triple count"
        );
        for stream in 0..6u32 {
            let pieces = chunks
                .iter()
                .filter(|(meta, _)| meta.kind == ChunkKind::Stream && meta.column_id == stream)
                .count();
            assert!(pieces > 1, "stream {stream} spans {pieces} pieces");
        }
        assert!(
            chunks[1..]
                .iter()
                .all(|(meta, bytes)| meta.kind == ChunkKind::Stream
                    && meta.graph_id == 0
                    && meta.column_id < 6
                    && bytes.len() <= 64),
            "every other chunk is a piece of one of the six streams, within the caps"
        );

        let restored = read_back(&image).unwrap();
        assert_same_ring(&store.ring().unwrap(), &restored.ring().expect("a ring"));
    }

    /// A section built with `new` writes with the caps of its thread, read
    /// when it is built, and records them in its metadata chunk.
    #[test]
    fn the_section_takes_the_caps_of_its_thread() {
        let store = ringed_store();
        let section = grafeo_common::testing::chunk_caps::with_chunk_caps(TINY, || {
            RdfRingSection::new(Arc::clone(&store))
        });
        let chunks = chunks_of(&image_of(&section));
        let (meta, _): (RingMeta, usize) =
            bincode::serde::decode_from_slice(&chunks[0].1, bincode::config::standard()).unwrap();
        assert_eq!(meta.max_bytes, TINY.max_bytes, "the caps it was built with");
        assert!(
            chunks[1..]
                .iter()
                .all(|(_, bytes)| bytes.len() <= TINY.max_bytes as usize),
            "pieces of at most the caps of the thread that built the section"
        );
        let section = RdfRingSection::new(store);
        let chunks = chunks_of(&image_of(&section));
        assert_eq!(
            chunks.len(),
            7,
            "outside the override: the metadata chunk and one piece per stream"
        );
    }

    /// The section writes a ring in a defined order: the same ring written
    /// twice, and two rings built from the same triples in the same order,
    /// give the same chunks.
    #[test]
    fn writing_a_ring_twice_gives_the_same_chunks() {
        let store = ringed_store();
        let first = chunks_of(&image_of(&RdfRingSection::with_caps(
            Arc::clone(&store),
            TINY,
        )));
        let second = chunks_of(&image_of(&RdfRingSection::with_caps(
            Arc::clone(&store),
            TINY,
        )));
        assert!(first == second, "one ring written twice");
        let built = || {
            let store = Arc::new(RdfStore::new());
            store.set_ring(TripleRing::from_triples(five_hundred_triples().into_iter()));
            chunks_of(&image_of(&RdfRingSection::with_caps(store, TINY)))
        };
        assert!(built() == built(), "two rings built from the same triples");
    }

    /// A store without a ring writes no chunk, so the image holds no ring
    /// section and a load leaves the ring out, as it was.
    #[test]
    fn a_store_without_a_ring_writes_no_chunk() {
        let section = RdfRingSection::with_caps(Arc::new(RdfStore::new()), TINY);
        let image = image_of(&section);
        assert_eq!(image.chunk_count(), 0);
        assert!(image.section_source(SectionType::RdfRing).is_none());
    }

    /// The sections 0.5.x wrote, one raw chunk of the version 2 envelope or
    /// of version 1 bincode, still load; an empty one loads no ring.
    #[test]
    fn a_0_5_ring_section_still_loads() {
        let store = ringed_store();
        let original = store.ring().unwrap();
        let envelope = RdfRingSection::new(Arc::clone(&store)).serialize().unwrap();
        assert_eq!(&envelope[..4], V2_MAGIC);
        let bincode = original.save_to_bytes().unwrap();
        for (version, bytes) in [("version 2", envelope), ("version 1", bincode)] {
            let image = MemoryImage::from_raw(vec![(SectionType::RdfRing, bytes)]).unwrap();
            let restored = read_back(&image).unwrap();
            let ring = restored
                .ring()
                .unwrap_or_else(|| panic!("{version}: a ring"));
            assert_same_ring(&original, &ring);
        }
        let image = MemoryImage::from_raw(vec![(SectionType::RdfRing, Vec::new())]).unwrap();
        assert!(read_back(&image).unwrap().ring().is_none());
    }

    /// Chunk sequences no writer of this release produces are refused,
    /// naming what is wrong, and leave the store without a ring.
    #[test]
    fn crafted_ring_sections_are_refused() {
        let image = image_of(&RdfRingSection::with_caps(ringed_store(), TINY));
        let chunks = chunks_of(&image);
        let without = |drop: &dyn Fn(&ChunkMeta) -> bool| -> Vec<(ChunkMeta, Vec<u8>)> {
            chunks
                .iter()
                .filter(|(meta, _)| !drop(meta))
                .cloned()
                .collect()
        };
        let with_meta = |meta: &RingMeta| {
            let bytes = bincode::serde::encode_to_vec(meta, bincode::config::standard()).unwrap();
            let mut crafted = vec![(ChunkMeta::meta(), bytes)];
            crafted.extend(chunks[1..].iter().cloned());
            crafted
        };
        let with_chunk = |meta: ChunkMeta| {
            let mut crafted = chunks.clone();
            crafted.push((meta, b"Vincent".to_vec()));
            crafted
        };
        let piece = |meta: &ChunkMeta, stream: u32| {
            meta.kind == ChunkKind::Stream && meta.column_id == stream
        };
        let last_subjects = chunks
            .iter()
            .filter(|(meta, _)| piece(meta, 1))
            .map(|(meta, _)| meta.row_start)
            .max()
            .unwrap();
        let mut trailing = chunks.clone();
        trailing[0].1.push(19);
        // Layout 1, then max_bytes as a u64 varint (marker 253, eight
        // bytes): more than a u32 holds.
        let mut wide_cap = vec![(ChunkMeta::meta(), vec![1, 253, 0, 0, 0, 0, 1, 0, 0, 0, 0])];
        wide_cap.extend(chunks[1..].iter().cloned());
        // Layout 1, max_bytes 64, then num_triples as a u128 varint (marker
        // 254) of u64::MAX + 1: more than a u64 holds.
        let mut huge = vec![1u8, 64, 254];
        huge.extend_from_slice(&(u128::from(u64::MAX) + 1).to_le_bytes());
        let mut huge_count = vec![(ChunkMeta::meta(), huge)];
        huge_count.extend(chunks[1..].iter().cloned());
        let mut late_meta = chunks[1..].to_vec();
        late_meta.insert(1, chunks[0].clone());

        let cases: Vec<(&str, u8, Vec<(ChunkMeta, Vec<u8>)>, &[&str])> = vec![
            (
                "another version",
                2,
                chunks.clone(),
                &["RdfRing", "version 2", "version 3"],
            ),
            (
                "no metadata chunk",
                3,
                chunks[1..].to_vec(),
                &["RdfRing", "metadata chunk"],
            ),
            (
                "the metadata chunk after a piece",
                3,
                late_meta,
                &["RdfRing", "metadata chunk"],
            ),
            (
                "trailing bytes in the metadata",
                3,
                trailing,
                &["RdfRing", "metadata"],
            ),
            (
                "a byte cap wider than a u32",
                3,
                wide_cap,
                &["RdfRing", "metadata chunk does not decode"],
            ),
            (
                "a triple count wider than a u64",
                3,
                huge_count,
                &["RdfRing", "metadata chunk does not decode"],
            ),
            (
                "another layout",
                3,
                with_meta(&RingMeta {
                    layout: 2,
                    max_bytes: 64,
                    num_triples: 500,
                }),
                &["RdfRing", "layout 2"],
            ),
            (
                "another triple count",
                3,
                with_meta(&RingMeta {
                    layout: 1,
                    max_bytes: 64,
                    num_triples: 499,
                }),
                &["RdfRing", "499"],
            ),
            (
                "a seventh stream",
                3,
                with_chunk(ChunkMeta::stream_piece(0, 6, 0)),
                &["RdfRing", "stream 6"],
            ),
            (
                "a stream of another graph",
                3,
                with_chunk(ChunkMeta::stream_piece(3, 1, 0)),
                &["RdfRing", "graph 3"],
            ),
            (
                "a column chunk",
                3,
                with_chunk(ChunkMeta::column(0, 1, 0, 3, 0)),
                &["RdfRing", "Column"],
            ),
            (
                "a stream piece with rows",
                3,
                chunks
                    .iter()
                    .map(|(meta, bytes)| {
                        let meta = if piece(meta, 3) && meta.row_start == 0 {
                            ChunkMeta {
                                row_count: 88,
                                ..*meta
                            }
                        } else {
                            *meta
                        };
                        (meta, bytes.clone())
                    })
                    .collect(),
                &["RdfRing", "88 rows"],
            ),
            (
                "a missing stream",
                3,
                without(&|meta| piece(meta, 4)),
                &["RdfRing", "stream 4", "spo_to_pos"],
            ),
            (
                "a stream without its last piece",
                3,
                without(&|meta| piece(meta, 1) && meta.row_start == last_subjects),
                &["RdfRing", "wavelet"],
            ),
            (
                "a stream without its first piece",
                3,
                without(&|meta| piece(meta, 2) && meta.row_start == 0),
                &["RdfRing", "stream 2", "expected 0"],
            ),
        ];
        for (case, version, crafted_chunks, expected) in cases {
            let store = Arc::new(RdfStore::new());
            let image = crafted(version, &crafted_chunks);
            let source = image.section_source(SectionType::RdfRing).unwrap();
            let error = RdfRingSection::new(Arc::clone(&store))
                .read_from(&*source)
                .expect_err(case)
                .to_string();
            for needle in expected {
                assert!(error.contains(needle), "{case}: {error}");
            }
            assert!(store.ring().is_none(), "{case}: no ring is set");
        }
    }

    /// A ring section that fails to decode leaves the ring the store had as
    /// it was: the ring is set only once all six parts are read.
    #[test]
    fn a_failed_read_keeps_the_ring_the_store_had() {
        let image = image_of(&RdfRingSection::with_caps(ringed_store(), TINY));
        let mut chunks = chunks_of(&image);
        let objects = chunks
            .iter()
            .position(|(meta, _)| meta.kind == ChunkKind::Stream && meta.column_id == 3)
            .unwrap();
        chunks[objects].1[0] ^= 0xFF;
        let store = test_store();
        let before = store.ring().unwrap();
        let image = crafted(RING_SECTION_VERSION, &chunks);
        let source = image.section_source(SectionType::RdfRing).unwrap();
        let error = RdfRingSection::new(Arc::clone(&store))
            .read_from(&*source)
            .unwrap_err()
            .to_string();
        assert!(error.contains("bad magic"), "{error}");
        assert!(
            Arc::ptr_eq(&store.ring().unwrap(), &before),
            "the store keeps its ring"
        );
    }

    /// A sink error while the section streams out comes back as that error,
    /// with its variant.
    #[test]
    fn a_sink_error_while_streaming_is_returned() {
        /// Accepts `left` chunks, then refuses every chunk.
        struct Refusing {
            left: usize,
        }
        impl SectionSink for Refusing {
            fn write_chunk(&mut self, meta: ChunkMeta, _bytes: &[u8]) -> Result<()> {
                if self.left == 0 {
                    return Err(Error::Io(std::io::Error::new(
                        std::io::ErrorKind::StorageFull,
                        format!("no room for the chunk at offset {}", meta.row_start),
                    )));
                }
                self.left -= 1;
                Ok(())
            }
        }
        let section = RdfRingSection::with_caps(ringed_store(), TINY);
        for left in [0, 1, 5, 40] {
            let error = section
                .write_to(&mut Refusing { left })
                .expect_err("the sink refuses");
            assert!(
                matches!(&error, Error::Io(inner) if inner.kind() == std::io::ErrorKind::StorageFull),
                "after {left} chunks: {error:?}"
            );
        }
    }
    /// Caps of zero are refused before any chunk is written, the metadata
    /// chunk included.
    #[test]
    fn caps_of_zero_are_refused_before_any_chunk() {
        let caps = ChunkCaps {
            max_rows: 3,
            max_bytes: 0,
        };
        let section = RdfRingSection::with_caps(ringed_store(), caps);
        let mut image = MemoryImage::new();
        image.begin_section(SectionType::RdfRing, 3).unwrap();
        let error = section.write_to(&mut image).unwrap_err().to_string();
        assert!(error.contains("max_bytes 0"), "{error}");
        assert_eq!(image.chunk_count(), 0, "no chunk is written");
    }

    /// A stream holds exactly its part: bytes after any of the six parts
    /// are refused, naming the part.
    #[test]
    fn bytes_after_any_part_are_refused() {
        let chunks = chunks_of(&image_of(&RdfRingSection::with_caps(ringed_store(), TINY)));
        for (stream, part) in (0u32..).zip(PARTS) {
            let length: u64 = chunks
                .iter()
                .filter(|(meta, _)| meta.kind == ChunkKind::Stream && meta.column_id == stream)
                .map(|(_, bytes)| bytes.len() as u64)
                .sum();
            let mut with_junk = chunks.clone();
            with_junk.push((
                ChunkMeta::stream_piece(0, stream, length),
                b"Vincent in Amsterdam".to_vec(),
            ));
            let store = Arc::new(RdfStore::new());
            let image = crafted(RING_SECTION_VERSION, &with_junk);
            let source = image.section_source(SectionType::RdfRing).unwrap();
            let error = RdfRingSection::new(Arc::clone(&store))
                .read_from(&*source)
                .expect_err(part)
                .to_string();
            assert!(
                error.contains(part) && error.contains(&format!("{}", length + 20)),
                "{part}: {error}"
            );
            assert!(store.ring().is_none(), "{part}: no ring is set");
        }
    }

    /// Terms whose N-Triples string does not parse back to them (whitespace
    /// that parsing trims, a character it does not decode) are refused when
    /// read, so the ring is built again from the triples instead of
    /// answering for other terms. Two blank nodes that differ only in a
    /// trailing space would parse to one term and shift every later id.
    #[test]
    fn terms_that_do_not_survive_their_string_are_refused() {
        let blank_space = vec![
            Triple::new(
                Term::blank("b"),
                Term::iri("http://example.org/knows"),
                Term::iri("http://example.org/gus"),
            ),
            Triple::new(
                Term::blank("b "),
                Term::iri("http://example.org/knows"),
                Term::iri("http://example.org/mia"),
            ),
        ];
        let language = vec![Triple::new(
            Term::iri("http://example.org/mia"),
            Term::iri("http://example.org/motto"),
            Term::lang_literal("gezellig", "nl "),
        )];
        let accent = vec![Triple::new(
            Term::iri("http://example.org/mia"),
            Term::iri("http://example.org/lives_in"),
            Term::literal("Krak\u{f3}w"),
        )];
        for (case, extra) in [
            ("blank nodes", blank_space),
            ("a language tag", language),
            ("an accent", accent),
        ] {
            let mut triples = extra;
            triples.extend(five_hundred_triples());
            let store = Arc::new(RdfStore::new());
            store.bulk_load(triples);
            let image = image_of(&RdfRingSection::with_caps(store, TINY));
            let error = read_back(&image)
                .map(|restored| restored.ring().map(|ring| ring.len()))
                .expect_err(case)
                .to_string();
            assert!(
                error.contains("RdfRing") && error.contains("dictionary"),
                "{case}: {error}"
            );
        }
    }

    /// Two stores holding the same triples write the same ring section,
    /// however the triples came in: bulk loaded in either order, or
    /// inserted one by one and the ring rebuilt (the term ids follow the
    /// byte order of the terms, not the order of a hash set).
    #[test]
    fn stores_with_the_same_triples_write_the_same_ring() {
        let chunks_of_store =
            |store: Arc<RdfStore>| chunks_of(&image_of(&RdfRingSection::with_caps(store, TINY)));
        let forward = Arc::new(RdfStore::new());
        forward.bulk_load(five_hundred_triples());
        let backward = Arc::new(RdfStore::new());
        backward.bulk_load(five_hundred_triples().into_iter().rev());
        let inserted = Arc::new(RdfStore::new());
        for triple in five_hundred_triples() {
            inserted.insert(triple);
        }
        inserted.rebuild_ring();
        let expected = chunks_of_store(forward);
        assert!(
            chunks_of_store(backward) == expected,
            "bulk loaded in the other order"
        );
        assert!(
            chunks_of_store(inserted) == expected,
            "inserted one by one, the ring rebuilt"
        );
    }

    /// The version 2 envelope 0.5.x wrote for two triples (Vincent knows
    /// Mia, Mia lives in Amsterdam), with the term ids of 0.5.x: in the order
    /// the terms came, which is not their byte order.
    const ENVELOPE_0_5: &str = concat!(
        "47524652020000000200000000000000400000000000000012010000000000006201000000000000",
        "b20100000000000002020000000000001a0200000000000050444354010000000500000000000000",
        "76000000000000003c687474703a2f2f6578616d706c652e6f72672f76696e63656e743e3c687474",
        "703a2f2f6578616d706c652e6f72672f6b6e6f77733e3c687474703a2f2f6578616d706c652e6f72",
        "672f6d69613e3c687474703a2f2f6578616d706c652e6f72672f6c697665735f696e3e22416d7374",
        "657264616d2200000000000000001c0000000000000036000000000000004e000000000000006b00",
        "00000000000076000000000000000400000001000000030000000200000000000000575452450100",
        "00000100000000000000020000000000000002000000000000000200000000000000000000000000",
        "00000200000000000000020000000000000001000000000000000200000000000000575452450100",
        "00000100000000000000020000000000000002000000000000000200000000000000010000000000",
        "00000300000000000000020000000000000001000000000000000200000000000000575452450100",
        "00000100000000000000020000000000000002000000000000000200000000000000020000000000",
        "000004000000000000000200000000000000010000000000000002000000000000005045524d0100",
        "0000020000000000000000000000010000005045524d010000000200000000000000000000000100",
        "00005d298176",
    );

    /// A version 2 envelope as 0.5.x wrote it, pinned as bytes (no
    /// released fixture holds a ring), reads through `read_from` with the
    /// term ids it was written with.
    #[test]
    fn a_pinned_0_5_envelope_reads_through_read_from() {
        let bytes: Vec<u8> = (0..ENVELOPE_0_5.len())
            .step_by(2)
            .map(|at| u8::from_str_radix(&ENVELOPE_0_5[at..at + 2], 16).unwrap())
            .collect();
        assert_eq!(bytes.len(), 566);
        let image = MemoryImage::from_raw(vec![(SectionType::RdfRing, bytes)]).unwrap();
        let ring = read_back(&image).unwrap().ring().expect("a ring");
        let vincent_knows_mia = Triple::new(
            Term::iri("http://example.org/vincent"),
            Term::iri("http://example.org/knows"),
            Term::iri("http://example.org/mia"),
        );
        let mia_lives_in_amsterdam = Triple::new(
            Term::iri("http://example.org/mia"),
            Term::iri("http://example.org/lives_in"),
            Term::literal("Amsterdam"),
        );
        assert_eq!(
            (ring.get_spo(0), ring.get_spo(1), ring.get_spo(2)),
            (Some(vincent_knows_mia), Some(mia_lives_in_amsterdam), None),
            "the triples in the SPO order of the 0.5.x ids"
        );
        assert_eq!(
            ring.dictionary().get_term(0),
            Some(&Term::iri("http://example.org/vincent")),
            "the ids 0.5.x gave"
        );
        let mia = TriplePattern {
            subject: None,
            predicate: None,
            object: Some(Term::iri("http://example.org/mia")),
        };
        assert_eq!(ring.count(&mia), 1);
    }
}
