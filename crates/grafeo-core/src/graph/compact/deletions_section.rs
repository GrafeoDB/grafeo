//! [`Section`](grafeo_common::storage::section::Section) implementation for
//! the layered overlay deletion log.
//!
//! The [`LayeredStore`](crate::graph::compact::layered::LayeredStore)
//! tracks deletions of base-store entities as tombstones in its overlay:
//! a transaction's are pending until its commit stamps them, and a
//! rollback removes them. Without this section, the committed ones are
//! lost across a close/reopen cycle: the overlay scan in
//! `LayeredStore::with_overlay` cannot distinguish a deleted base node
//! (which has no overlay entry) from a base node that was never modified,
//! so previously-deleted base entities would silently reappear after
//! reload until the next `compact()` merges the overlay into the base.
//!
//! This section persists the deletion log alongside the rest of the
//! container: the ids of the committed tombstones only, as a checkpoint
//! writes the committed state and an open transaction may still roll its
//! deletes back. Reload restores them as tombstones committed at the
//! initial epoch, deleted for every reader.
//!
//! The section (version 2, since 0.6) is a metadata chunk (`DeletionsMeta`)
//! with the counts, then stream 0 with the node ids and stream 1 with the
//! edge ids, each a `u64` LE, strictly increasing. A 0.5.x file holds the
//! version 1 layout as one raw chunk, which still loads.

use std::io::{Read, Write};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use grafeo_common::storage::section::{Section, SectionType};
use grafeo_common::storage::{
    ChunkCaps, ChunkStreamReader, ChunkStreamWriter, SectionSink, SectionSource, check_version,
    legacy_bytes, stream_error,
};
use grafeo_common::types::{EdgeId, NodeId};
use grafeo_common::utils::error::{Error, Result};
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};

#[cfg(feature = "lpg")]
use super::layered::LayeredStore;
use super::section::{META_LAYOUT, in_section, read_meta, stream_write_error, write_meta};

/// Magic bytes identifying an OverlayDeletions section ("Grafeo Overlay
/// Deletion Log").
const MAGIC: [u8; 4] = *b"GODL";

/// The layout [`Section::serialize`] writes: magic, version, the counts and
/// ids, a CRC32. 0.5.x files hold it as one raw chunk.
const FORMAT_VERSION: u8 = 1;

/// The section version since 0.6: a metadata chunk with the counts, then the
/// node ids as stream 0 and the edge ids as stream 1.
const FORMAT_VERSION_CHUNKED: u8 = 2;

/// The stream holding the deleted node ids.
const NODE_STREAM: u32 = 0;

/// The stream holding the deleted edge ids.
const EDGE_STREAM: u32 = 1;

/// Ids a reader makes room for before it reads them: a crafted count must
/// not request a huge allocation, and the vector grows past this as ids
/// arrive.
const PREALLOCATED_IDS: usize = 1 << 16;

/// The metadata chunk of a version 2 section.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
struct DeletionsMeta {
    /// The layout byte of the compact module's metadata chunks.
    layout: u8,
    /// The byte cap the streams were cut with; readers accept any.
    max_bytes: u32,
    /// The ids stream 0 holds.
    nodes: u64,
    /// The ids stream 1 holds.
    edges: u64,
}

/// Snapshot of the layered overlay's deletion log (its committed
/// tombstones), ready to be serialized into the container or to seed a
/// freshly-loaded `LayeredStore`.
///
/// When constructed via [`Self::from_layered`], `is_dirty` / `mark_clean`
/// delegate to the layered store's own deletions-dirty flag so checkpoint
/// cycles only re-emit the section when the deletion log has actually
/// changed. The [`Self::empty`] constructor (used on the load path) holds
/// no layered store and tracks dirtiness locally as `false`, since
/// deserialized data is by definition already on disk.
pub struct OverlayDeletionsSection {
    payload: RwLock<DeletionsPayload>,
    /// Source of truth for `is_dirty` / `mark_clean` when this section was
    /// built from a live `LayeredStore`. `None` for sections constructed
    /// for the load path.
    #[cfg(feature = "lpg")]
    layered: Option<Arc<LayeredStore>>,
    /// Local dirty flag used when no `LayeredStore` is attached.
    local_dirty: AtomicBool,
    /// The caps the streams are cut with: [`ChunkCaps::current`] when the
    /// section was built.
    caps: ChunkCaps,
}

#[derive(Default, Clone, Debug)]
struct DeletionsPayload {
    nodes: Vec<NodeId>,
    edges: Vec<EdgeId>,
}

impl OverlayDeletionsSection {
    /// Creates a section by snapshotting the layered store's committed
    /// deletes. The snapshot is sorted (and deduplicated) so the
    /// on-disk byte representation is stable for the same set of ids.
    /// `is_dirty` / `mark_clean` proxy to the layered store, so a
    /// checkpoint that finds the deletion log unchanged since the last
    /// write skips re-emitting this section.
    #[cfg(feature = "lpg")]
    #[must_use]
    pub fn from_layered(layered: Arc<LayeredStore>) -> Self {
        let mut nodes = layered.snapshot_deleted_node_ids();
        let mut edges = layered.snapshot_deleted_edge_ids();
        nodes.sort_unstable();
        nodes.dedup();
        edges.sort_unstable();
        edges.dedup();
        Self {
            payload: RwLock::new(DeletionsPayload { nodes, edges }),
            layered: Some(layered),
            local_dirty: AtomicBool::new(false),
            caps: ChunkCaps::current(),
        }
    }

    /// Creates an empty section, used by the load path before
    /// [`Self::deserialize`] populates it. Has no attached layered store;
    /// `is_dirty` is `false` until the caller hands the deserialized
    /// payload back to the engine.
    #[must_use]
    pub fn empty() -> Self {
        Self {
            payload: RwLock::new(DeletionsPayload::default()),
            #[cfg(feature = "lpg")]
            layered: None,
            local_dirty: AtomicBool::new(false),
            caps: ChunkCaps::current(),
        }
    }

    /// Returns a clone of the snapshot's deleted node ids.
    #[must_use]
    pub fn deleted_node_ids(&self) -> Vec<NodeId> {
        self.payload.read().nodes.clone()
    }

    /// Returns a clone of the snapshot's deleted edge ids.
    #[must_use]
    pub fn deleted_edge_ids(&self) -> Vec<EdgeId> {
        self.payload.read().edges.clone()
    }

    /// Whether the snapshot carries no ids.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        let p = self.payload.read();
        p.nodes.is_empty() && p.edges.is_empty()
    }

    fn encode_payload(&self) -> Vec<u8> {
        let p = self.payload.read();
        // Header (8) + node_count (8) + nodes + edge_count (8) + edges + crc (4)
        let mut buf = Vec::with_capacity(8 + 8 + p.nodes.len() * 8 + 8 + p.edges.len() * 8 + 4);

        buf.extend_from_slice(&MAGIC);
        buf.push(FORMAT_VERSION);
        buf.extend_from_slice(&[0u8; 3]); // reserved

        // reason: id counts are bounded by entity counts in a single store,
        // which fit in u64 for any practical workload
        buf.extend_from_slice(&(p.nodes.len() as u64).to_le_bytes());
        for nid in &p.nodes {
            buf.extend_from_slice(&nid.0.to_le_bytes());
        }
        buf.extend_from_slice(&(p.edges.len() as u64).to_le_bytes());
        for eid in &p.edges {
            buf.extend_from_slice(&eid.0.to_le_bytes());
        }

        let crc = crc32fast::hash(&buf);
        buf.extend_from_slice(&crc.to_le_bytes());
        buf
    }

    fn decode_payload(data: &[u8]) -> Result<DeletionsPayload> {
        if data.len() < 8 + 8 + 8 + 4 {
            return Err(Error::Serialization(
                "OverlayDeletions section too short".into(),
            ));
        }
        if data[..4] != MAGIC {
            return Err(Error::Serialization(format!(
                "OverlayDeletions magic mismatch: expected {MAGIC:?}, got {:?}",
                &data[..4],
            )));
        }
        let version = data[4];
        if version != FORMAT_VERSION {
            return Err(Error::Serialization(format!(
                "unsupported OverlayDeletions section version {version}, expected {FORMAT_VERSION}",
            )));
        }

        // CRC verifies the entire prefix up to the trailing 4 bytes.
        let payload = &data[..data.len() - 4];
        let stored_crc = u32::from_le_bytes(data[data.len() - 4..].try_into().unwrap());
        let actual_crc = crc32fast::hash(payload);
        if stored_crc != actual_crc {
            return Err(Error::Serialization(format!(
                "OverlayDeletions CRC mismatch: stored {stored_crc:#010X}, computed {actual_crc:#010X}",
            )));
        }

        let mut pos = 8usize;
        let read_u64 = |buf: &[u8], pos: &mut usize| -> Result<u64> {
            if *pos + 8 > buf.len() {
                return Err(Error::Serialization(
                    "OverlayDeletions truncated mid-entry".into(),
                ));
            }
            let v = u64::from_le_bytes(buf[*pos..*pos + 8].try_into().unwrap());
            *pos += 8;
            Ok(v)
        };

        let node_count_u64 = read_u64(data, &mut pos)?;
        let node_count = usize::try_from(node_count_u64).map_err(|_| {
            Error::Serialization(format!(
                "OverlayDeletions node_count {node_count_u64} exceeds usize on this target",
            ))
        })?;
        // Sanity bound: each node id is 8 bytes, plus 8 bytes for edge_count
        // and 4 trailing CRC bytes. Reject obvious garbage early so we don't
        // pre-allocate huge vecs from a corrupt header.
        if node_count
            .checked_mul(8)
            .map_or(true, |n| pos + n + 8 + 4 > data.len())
        {
            return Err(Error::Serialization(format!(
                "OverlayDeletions node_count {node_count} exceeds section size",
            )));
        }
        let mut nodes = Vec::with_capacity(node_count);
        for _ in 0..node_count {
            nodes.push(NodeId(read_u64(data, &mut pos)?));
        }

        let edge_count_u64 = read_u64(data, &mut pos)?;
        let edge_count = usize::try_from(edge_count_u64).map_err(|_| {
            Error::Serialization(format!(
                "OverlayDeletions edge_count {edge_count_u64} exceeds usize on this target",
            ))
        })?;
        if edge_count
            .checked_mul(8)
            .map_or(true, |n| pos + n + 4 > data.len())
        {
            return Err(Error::Serialization(format!(
                "OverlayDeletions edge_count {edge_count} exceeds section size",
            )));
        }
        let mut edges = Vec::with_capacity(edge_count);
        for _ in 0..edge_count {
            edges.push(EdgeId(read_u64(data, &mut pos)?));
        }

        Ok(DeletionsPayload { nodes, edges })
    }

    /// Drains the snapshot into `(nodes, edges)`, leaving the section empty.
    /// Used by the load path to seed the layered store.
    pub fn take(&self) -> (Vec<NodeId>, Vec<EdgeId>) {
        let mut p = self.payload.write();
        let nodes = std::mem::take(&mut p.nodes);
        let edges = std::mem::take(&mut p.edges);
        (nodes, edges)
    }
}

/// Writes `ids` as stream `stream`, a `u64` LE each, refusing ids that are
/// not strictly increasing: the reader refuses them.
fn write_ids(
    sink: &mut dyn SectionSink,
    stream: u32,
    ids: impl Iterator<Item = u64>,
    caps: ChunkCaps,
    what: &str,
) -> Result<()> {
    let mut writer = ChunkStreamWriter::new(sink, 0, stream, caps);
    let mut previous = None;
    for id in ids {
        if let Some(previous) = previous
            && id <= previous
        {
            return Err(Error::Internal(format!(
                "section OverlayDeletions: {what} id {id} follows {previous}; deleted ids are \
                 written sorted and once each"
            )));
        }
        previous = Some(id);
        if let Err(error) = writer.write_all(&id.to_le_bytes()) {
            return Err(stream_write_error(
                writer,
                SectionType::OverlayDeletions,
                error,
            ));
        }
    }
    writer.finish().map(drop)
}

/// Reads the ids of stream `stream`, a `u64` LE each and strictly
/// increasing, and checks that the stream holds `count` of them.
fn read_ids<Id>(
    source: &dyn SectionSource,
    stream: u32,
    count: u64,
    what: &str,
    id: impl Fn(u64) -> Id,
) -> Result<Vec<Id>> {
    let refuse = |message: String| {
        Error::Serialization(format!(
            "section OverlayDeletions: stream {stream} {message}"
        ))
    };
    let mut reader = ChunkStreamReader::new(source, 0, stream);
    let mut ids = Vec::with_capacity(
        usize::try_from(count).map_or(PREALLOCATED_IDS, |count| count.min(PREALLOCATED_IDS)),
    );
    let mut read = 0u64;
    let mut previous = None;
    let mut word = [0u8; 8];
    while !reader.is_at_end() {
        if read == count {
            return Err(refuse(format!(
                "holds more than the {count} {what} ids its metadata chunk counts"
            )));
        }
        reader
            .read_exact(&mut word)
            .map_err(|error| stream_error(SectionType::OverlayDeletions, error))?;
        let value = u64::from_le_bytes(word);
        if let Some(previous) = previous
            && value <= previous
        {
            return Err(refuse(format!(
                "holds {what} id {value} after {previous}; the ids are strictly increasing"
            )));
        }
        previous = Some(value);
        ids.push(id(value));
        read += 1;
    }
    if read != count {
        return Err(refuse(format!(
            "holds {read} {what} ids, its metadata chunk counts {count}"
        )));
    }
    Ok(ids)
}

impl Section for OverlayDeletionsSection {
    fn section_type(&self) -> SectionType {
        SectionType::OverlayDeletions
    }

    fn version(&self) -> u8 {
        FORMAT_VERSION_CHUNKED
    }

    /// The version 1 layout, which 0.5.x files hold.
    fn serialize(&self) -> Result<Vec<u8>> {
        Ok(self.encode_payload())
    }

    fn deserialize(&mut self, data: &[u8]) -> Result<()> {
        let payload = Self::decode_payload(data)?;
        *self.payload.write() = payload;
        self.local_dirty.store(false, Ordering::Release);
        Ok(())
    }

    /// The metadata chunk with the counts, then the node ids as stream 0 and
    /// the edge ids as stream 1, cut at the section's byte cap.
    fn write_to(&self, sink: &mut dyn SectionSink) -> Result<()> {
        let payload = self.payload.read();
        let meta = DeletionsMeta {
            layout: META_LAYOUT,
            max_bytes: self.caps.max_bytes,
            nodes: payload.nodes.len() as u64,
            edges: payload.edges.len() as u64,
        };
        write_meta(sink, SectionType::OverlayDeletions, &meta)?;
        let nodes = payload.nodes.iter().map(NodeId::as_u64);
        write_ids(sink, NODE_STREAM, nodes, self.caps, "node")?;
        let edges = payload.edges.iter().map(EdgeId::as_u64);
        write_ids(sink, EDGE_STREAM, edges, self.caps, "edge")
    }

    /// A single raw chunk (0.5.x bytes) goes to the version 1 reader.
    /// Otherwise the section is version 2: its metadata chunk and two
    /// streams, each holding as many ids as the metadata chunk counts.
    fn read_from(&mut self, source: &dyn SectionSource) -> Result<()> {
        if let Some(bytes) =
            legacy_bytes(source).map_err(|e| in_section(SectionType::OverlayDeletions, e))?
        {
            return self.deserialize(&bytes);
        }
        check_version(
            SectionType::OverlayDeletions,
            source,
            FORMAT_VERSION_CHUNKED,
        )?;
        let meta: DeletionsMeta = read_meta(source, SectionType::OverlayDeletions, 2)?;
        let nodes = read_ids(source, NODE_STREAM, meta.nodes, "node", NodeId)?;
        let edges = read_ids(source, EDGE_STREAM, meta.edges, "edge", EdgeId)?;
        *self.payload.write() = DeletionsPayload { nodes, edges };
        self.local_dirty.store(false, Ordering::Release);
        Ok(())
    }

    fn is_dirty(&self) -> bool {
        #[cfg(feature = "lpg")]
        if let Some(ref layered) = self.layered {
            return layered.deletions_dirty();
        }
        self.local_dirty.load(Ordering::Acquire)
    }

    fn mark_clean(&self) {
        #[cfg(feature = "lpg")]
        if let Some(ref layered) = self.layered {
            layered.mark_deletions_clean();
            return;
        }
        self.local_dirty.store(false, Ordering::Release);
    }

    fn memory_usage(&self) -> usize {
        let p = self.payload.read();
        p.nodes.len() * std::mem::size_of::<NodeId>()
            + p.edges.len() * std::mem::size_of::<EdgeId>()
    }

    // Deletion log is small (a few KiB even for large workloads); the
    // default [`Section::swap_to_mmap`] reports `SpillError::NotSupported`,
    // which is what we want: there is no payoff in going through the
    // page-fetcher indirection for this section.
}

#[cfg(test)]
mod tests {
    use super::*;
    use grafeo_common::storage::{
        ChunkCaps, ChunkKind, ChunkMeta, ImageSource, MemoryImage, SectionSink,
    };
    use grafeo_common::testing::chunk_caps::with_chunk_caps;

    /// A section holding the node ids `nodes` and the edge ids `edges`, built
    /// with this thread's caps.
    fn section_with(nodes: &[u64], edges: &[u64]) -> OverlayDeletionsSection {
        OverlayDeletionsSection {
            payload: RwLock::new(DeletionsPayload {
                nodes: nodes.iter().map(|&id| NodeId(id)).collect(),
                edges: edges.iter().map(|&id| EdgeId(id)).collect(),
            }),
            #[cfg(feature = "lpg")]
            layered: None,
            local_dirty: AtomicBool::new(true),
            caps: ChunkCaps::current(),
        }
    }

    #[test]
    fn roundtrip_empty_payload() {
        let section = OverlayDeletionsSection::empty();
        let bytes = section.serialize().unwrap();

        let mut roundtrip = OverlayDeletionsSection::empty();
        roundtrip.deserialize(&bytes).unwrap();
        assert!(roundtrip.is_empty());
        assert!(
            roundtrip.deleted_node_ids().is_empty(),
            "{:?}",
            roundtrip.deleted_node_ids()
        );
        assert!(
            roundtrip.deleted_edge_ids().is_empty(),
            "{:?}",
            roundtrip.deleted_edge_ids()
        );
    }

    #[test]
    fn roundtrip_mixed_payload() {
        let section = section_with(&[1, 7, 42], &[3, 99]);
        let bytes = section.serialize().unwrap();

        let mut roundtrip = OverlayDeletionsSection::empty();
        roundtrip.deserialize(&bytes).unwrap();
        assert_eq!(
            roundtrip.deleted_node_ids(),
            vec![NodeId(1), NodeId(7), NodeId(42)]
        );
        assert_eq!(roundtrip.deleted_edge_ids(), vec![EdgeId(3), EdgeId(99)]);
    }

    #[test]
    fn rejects_bad_magic() {
        let mut bytes = OverlayDeletionsSection::empty().serialize().unwrap();
        bytes[0] = b'X';
        // Recompute CRC so the failure is the magic check, not CRC noise.
        let new_crc = crc32fast::hash(&bytes[..bytes.len() - 4]);
        let crc_offset = bytes.len() - 4;
        bytes[crc_offset..].copy_from_slice(&new_crc.to_le_bytes());

        let mut section = OverlayDeletionsSection::empty();
        let err = section
            .deserialize(&bytes)
            .expect_err("bad magic must fail");
        assert!(err.to_string().contains("magic"));
    }

    #[test]
    fn rejects_crc_mismatch() {
        let original = section_with(&[11], &[]);
        let mut bytes = original.serialize().unwrap();
        // Flip a node id byte after serialization so the trailing CRC no
        // longer matches.
        bytes[16] ^= 0xFF;

        let mut section = OverlayDeletionsSection::empty();
        let err = section
            .deserialize(&bytes)
            .expect_err("CRC mismatch must fail");
        assert!(err.to_string().contains("CRC mismatch"));
    }

    #[test]
    fn rejects_unreasonable_node_count() {
        // Construct a header that claims many more node ids than the section
        // body can possibly contain.
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&MAGIC);
        bytes.push(FORMAT_VERSION);
        bytes.extend_from_slice(&[0u8; 3]);
        bytes.extend_from_slice(&u64::MAX.to_le_bytes()); // claimed node count
        bytes.extend_from_slice(&0u64.to_le_bytes()); // edge count
        let crc = crc32fast::hash(&bytes);
        bytes.extend_from_slice(&crc.to_le_bytes());

        let mut section = OverlayDeletionsSection::empty();
        let err = section
            .deserialize(&bytes)
            .expect_err("absurd node_count must fail");
        assert!(err.to_string().contains("node_count"));
    }

    // ── Streams: section version 2 ──────────────────────────────────

    /// The image holding what `section` writes.
    fn image_of(section: &OverlayDeletionsSection) -> MemoryImage {
        MemoryImage::from_sections(&[section as &dyn Section]).unwrap()
    }

    /// The OverlayDeletions section of `image`, read into an empty section.
    fn load(image: &MemoryImage) -> Result<OverlayDeletionsSection> {
        let mut section = OverlayDeletionsSection::empty();
        section.read_from(&*image.section_source(SectionType::OverlayDeletions).unwrap())?;
        Ok(section)
    }

    /// A section of version `version` holding `chunks`, as a crafted file
    /// would.
    fn crafted(version: u8, chunks: &[(ChunkMeta, Vec<u8>)]) -> MemoryImage {
        let mut image = MemoryImage::new();
        image
            .begin_section(SectionType::OverlayDeletions, version)
            .unwrap();
        for (meta, bytes) in chunks {
            image.write_chunk(*meta, bytes).unwrap();
        }
        image
    }

    /// A metadata chunk with `layout` that counts `nodes` and `edges`.
    fn meta_chunk(layout: u8, nodes: u64, edges: u64) -> (ChunkMeta, Vec<u8>) {
        let meta = DeletionsMeta {
            layout,
            max_bytes: 16,
            nodes,
            edges,
        };
        let bytes = bincode::serde::encode_to_vec(meta, bincode::config::standard()).unwrap();
        (ChunkMeta::meta(), bytes)
    }

    /// One piece of stream `stream` holding `ids`, from its start.
    fn ids_piece(stream: u32, ids: &[u64]) -> (ChunkMeta, Vec<u8>) {
        let bytes = ids.iter().flat_map(|id| id.to_le_bytes()).collect();
        (ChunkMeta::stream_piece(0, stream, 0), bytes)
    }

    /// Each chunk of the section `image` holds, with its bytes.
    fn chunks_of(image: &MemoryImage) -> Vec<(ChunkMeta, bytes::Bytes)> {
        let source = image.section_source(SectionType::OverlayDeletions).unwrap();
        (0..source.chunks().len())
            .map(|index| (source.chunks()[index], source.fetch(index).unwrap()))
            .collect()
    }

    #[test]
    fn overlay_deletions_round_trip_through_streams() {
        // 19 node ids and 3 edge ids; pieces of 16 bytes hold two ids each.
        let nodes: Vec<u64> = (0..19).map(|i| 3 + 88 * i).collect();
        let edges = [3, 19, 88];
        let caps = ChunkCaps {
            max_rows: 3,
            max_bytes: 16,
        };
        let section = with_chunk_caps(caps, || section_with(&nodes, &edges));
        let image = image_of(&section);
        let source = image.section_source(SectionType::OverlayDeletions).unwrap();
        assert_eq!(source.section_version(), FORMAT_VERSION_CHUNKED);
        let chunks = source.chunks();
        assert_eq!(chunks[0], ChunkMeta::meta(), "the metadata chunk first");
        let meta: DeletionsMeta = bincode::serde::decode_from_slice(
            &source.fetch(0).unwrap(),
            bincode::config::standard(),
        )
        .unwrap()
        .0;
        assert_eq!(
            meta,
            DeletionsMeta {
                layout: 1,
                max_bytes: 16,
                nodes: 19,
                edges: 3
            }
        );
        let pieces = |stream: u32| {
            chunks
                .iter()
                .filter(|meta| **meta == ChunkMeta::stream_piece(0, stream, meta.row_start))
                .count()
        };
        assert_eq!((pieces(0), pieces(1)), (10, 2), "two ids per piece");
        assert_eq!(
            chunks.len(),
            13,
            "the metadata chunk and the pieces, nothing else"
        );

        let loaded = load(&image).unwrap();
        let node_ids: Vec<NodeId> = nodes.iter().map(|&id| NodeId(id)).collect();
        assert_eq!(loaded.deleted_node_ids(), node_ids);
        assert_eq!(
            loaded.deleted_edge_ids(),
            vec![EdgeId(3), EdgeId(19), EdgeId(88)]
        );
        assert!(!loaded.is_dirty(), "loaded deletions are on disk already");

        // Without deletions: the metadata chunk alone, and nothing comes back.
        let empty = image_of(&section_with(&[], &[]));
        assert_eq!(empty.chunk_count(), 1);
        assert!(load(&empty).unwrap().is_empty());
    }

    #[test]
    fn a_deletion_count_that_does_not_match_is_refused() {
        // Counts that match load.
        load(&crafted(
            2,
            &[
                meta_chunk(1, 2, 1),
                ids_piece(0, &[3, 19]),
                ids_piece(1, &[88]),
            ],
        ))
        .unwrap();

        let mut cut = ids_piece(0, &[3]);
        cut.1.truncate(5);
        for (case, chunks, expected) in [
            (
                "fewer node ids",
                vec![meta_chunk(1, 3, 0), ids_piece(0, &[3, 19])],
                "stream 0 holds 2 node ids, its metadata chunk counts 3",
            ),
            (
                "more node ids",
                vec![meta_chunk(1, 1, 0), ids_piece(0, &[3, 19])],
                "stream 0 holds more than the 1 node ids",
            ),
            (
                "node ids without a count",
                vec![meta_chunk(1, 0, 0), ids_piece(0, &[3])],
                "stream 0 holds more than the 0 node ids",
            ),
            (
                "more edge ids",
                vec![meta_chunk(1, 0, 1), ids_piece(1, &[3, 19])],
                "stream 1 holds more than the 1 edge ids",
            ),
            (
                "fewer edge ids",
                vec![meta_chunk(1, 0, 2), ids_piece(1, &[88])],
                "stream 1 holds 1 edge ids, its metadata chunk counts 2",
            ),
            ("an id cut short", vec![meta_chunk(1, 1, 0), cut], "ends"),
        ] {
            let error = load(&crafted(2, &chunks)).map(drop).unwrap_err();
            assert!(
                matches!(error, Error::Serialization(_)),
                "{case}: {error:?}"
            );
            let error = error.to_string();
            assert!(
                error.contains("OverlayDeletions") && error.contains(expected),
                "{case}: {error}"
            );
        }
    }

    /// The ids of a stream are strictly increasing, as `from_layered` sorts
    /// and deduplicates them: one set of deletions has one encoding. The
    /// reader refuses other ids, and the writer refuses to write them.
    #[test]
    fn deletion_ids_out_of_order_or_repeated_are_refused() {
        for (case, ids) in [("out of order", [19, 3]), ("repeated", [19, 19])] {
            for (stream, nodes, edges) in [(0, 2, 0), (1, 0, 2)] {
                let image = crafted(2, &[meta_chunk(1, nodes, edges), ids_piece(stream, &ids)]);
                let error = load(&image).map(drop).unwrap_err().to_string();
                assert!(
                    error.contains("OverlayDeletions")
                        && error.contains(&format!("stream {stream}"))
                        && error.contains("increasing"),
                    "{case}, stream {stream}: {error}"
                );
            }
            for (what, section) in [
                ("node", section_with(&ids, &[])),
                ("edge", section_with(&[], &ids)),
            ] {
                let mut image = crafted(2, &[]);
                let error = section.write_to(&mut image).unwrap_err();
                assert!(
                    matches!(&error, Error::Internal(message)
                        if message.contains(&format!("{what} id {} follows 19", ids[1]))),
                    "{case}, {what} ids: {error:?}"
                );
            }
        }
    }

    /// Chunk sequences no writer of this release produces are refused,
    /// naming the section and what is wrong.
    #[test]
    fn crafted_deletion_chunk_sequences_are_refused() {
        for (case, image, expected) in [
            (
                "an older section version",
                crafted(1, &[meta_chunk(1, 1, 0), ids_piece(0, &[3])]),
                "version 1",
            ),
            (
                "a newer metadata layout",
                crafted(2, &[meta_chunk(2, 1, 0), ids_piece(0, &[3])]),
                "layout 2",
            ),
            (
                "ids before the metadata chunk",
                crafted(2, &[ids_piece(0, &[3]), meta_chunk(1, 1, 0)]),
                "the first chunk is a piece of stream 0",
            ),
            (
                "a third stream",
                crafted(
                    2,
                    &[meta_chunk(1, 1, 0), ids_piece(0, &[3]), ids_piece(2, &[19])],
                ),
                "stream 2",
            ),
            (
                "a piece of another graph",
                crafted(
                    2,
                    &[
                        meta_chunk(1, 0, 0),
                        (
                            ChunkMeta::stream_piece(3, 0, 0),
                            19u64.to_le_bytes().to_vec(),
                        ),
                    ],
                ),
                "graph 3",
            ),
        ] {
            let error = load(&image).map(drop).unwrap_err().to_string();
            assert!(
                error.contains("OverlayDeletions") && error.contains(expected),
                "{case}: {error}"
            );
        }
    }

    /// A 0.5.x file holds the section as one raw chunk of the version 1
    /// layout, which `read_from` hands to the 0.5.x reader.
    #[test]
    fn a_0_5_overlay_deletions_section_still_loads() {
        let bytes = section_with(&[3, 19, 88], &[19]).serialize().unwrap();
        assert_eq!(bytes[4], FORMAT_VERSION, "the 0.5.x layout");
        let image = MemoryImage::from_raw(vec![(SectionType::OverlayDeletions, bytes)]).unwrap();
        let loaded = load(&image).unwrap();
        assert_eq!(
            loaded.deleted_node_ids(),
            vec![NodeId(3), NodeId(19), NodeId(88)]
        );
        assert_eq!(loaded.deleted_edge_ids(), vec![EdgeId(19)]);
    }

    /// The deletions of a layered store write the same chunks whatever order
    /// its sets hold them in.
    #[cfg(feature = "lpg")]
    #[test]
    fn deletions_write_the_same_chunks_whatever_order_they_were_made_in() {
        let written = |nodes: &[u64], edges: &[u64]| {
            let base = crate::graph::compact::CompactStoreBuilder::new()
                .build()
                .unwrap();
            let layered = Arc::new(LayeredStore::new(base, 100, 100).unwrap());
            layered.seed_deleted_from_base(
                nodes.iter().map(|&id| NodeId(id)),
                edges.iter().map(|&id| EdgeId(id)),
            );
            chunks_of(&image_of(&OverlayDeletionsSection::from_layered(layered)))
        };
        let first = written(&[88, 3, 19, 3], &[19, 3]);
        assert_eq!(
            first.len(),
            3,
            "the metadata chunk and one piece per stream"
        );
        assert_eq!(first, written(&[3, 19, 88], &[3, 19]));
        assert_eq!(first, written(&[19, 88, 3], &[3, 19, 3]));
    }

    /// Accepts `left` chunks, then refuses every chunk as a full disk would.
    struct FullDisk {
        left: usize,
    }

    impl SectionSink for FullDisk {
        fn write_chunk(&mut self, _meta: ChunkMeta, _bytes: &[u8]) -> Result<()> {
            if self.left == 0 {
                return Err(Error::Io(std::io::Error::new(
                    std::io::ErrorKind::StorageFull,
                    "no space left in Berlin",
                )));
            }
            self.left -= 1;
            Ok(())
        }
    }

    /// A sink error in a stream comes back as it was: a full disk stays an
    /// I/O error of its kind, whichever chunk it refuses.
    #[test]
    fn a_sink_error_in_a_stream_keeps_its_variant() {
        let caps = ChunkCaps {
            max_rows: 3,
            max_bytes: 16,
        };
        let section = with_chunk_caps(caps, || section_with(&[3, 19, 88], &[3]));
        let kinds: Vec<ChunkKind> = chunks_of(&image_of(&section))
            .iter()
            .map(|(meta, _)| meta.kind)
            .collect();
        assert_eq!(
            kinds,
            [
                ChunkKind::Meta,
                ChunkKind::Stream,
                ChunkKind::Stream,
                ChunkKind::Stream
            ],
            "the metadata chunk, two pieces of node ids, one of edge ids"
        );
        for left in 0..kinds.len() {
            let error = section.write_to(&mut FullDisk { left }).unwrap_err();
            assert!(
                matches!(&error, Error::Io(inner)
                    if inner.kind() == std::io::ErrorKind::StorageFull
                        && inner.to_string().contains("Berlin")),
                "after {left} chunks: {error:?}"
            );
        }
    }

    /// A count no stream could hold, next to an empty stream, is refused
    /// without making room for the ids it counts.
    #[test]
    fn a_deletion_count_beyond_any_stream_is_refused() {
        for (stream, nodes, edges, what) in [(0, u64::MAX, 0, "node"), (1, 0, u64::MAX, "edge")] {
            let error = load(&crafted(2, &[meta_chunk(1, nodes, edges)]))
                .map(drop)
                .unwrap_err();
            let expected = format!(
                "stream {stream} holds 0 {what} ids, its metadata chunk counts {}",
                u64::MAX
            );
            assert!(
                matches!(&error, Error::Serialization(message) if message.contains(&expected)),
                "{error:?}"
            );
        }
    }

    /// With pieces of 12 bytes, ids are split across pieces: the reader
    /// joins their halves.
    #[test]
    fn deletion_ids_split_across_pieces_round_trip() {
        let caps = ChunkCaps {
            max_rows: 3,
            max_bytes: 12,
        };
        let section = with_chunk_caps(caps, || section_with(&[3, 19, 88, 1988], &[19, 88]));
        let image = image_of(&section);
        let pieces: Vec<(u32, u64, usize)> = chunks_of(&image)
            .iter()
            .filter(|(meta, _)| meta.kind == ChunkKind::Stream)
            .map(|(meta, bytes)| (meta.column_id, meta.row_start, bytes.len()))
            .collect();
        assert_eq!(
            pieces,
            [(0, 0, 12), (0, 12, 12), (0, 24, 8), (1, 0, 12), (1, 12, 4)]
        );
        let loaded = load(&image).unwrap();
        assert_eq!(
            loaded.deleted_node_ids(),
            vec![NodeId(3), NodeId(19), NodeId(88), NodeId(1988)]
        );
        assert_eq!(loaded.deleted_edge_ids(), vec![EdgeId(19), EdgeId(88)]);
    }
}
