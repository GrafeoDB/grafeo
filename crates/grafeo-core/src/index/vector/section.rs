//! Vector Store section serializer for the `.grafeo` container format.
//!
//! Serializes HNSW topology (neighbor graphs) for all vector indexes.
//! Embeddings are not stored here: they live in LPG node properties and
//! are accessed via `VectorAccessor` during search.
//!
//! Persisting the topology eliminates the O(N log N) HNSW rebuild on
//! database open. For 1M vectors this saves 30-60 seconds of startup time.
//!
//! ## Format versioning (Phase 7b)
//!
//! The section transparently handles two on-disk formats:
//!
//! - **v2 paged (current):** packed envelope (`GVST` magic + index
//!   directory + per-index meta + per-index `GTOP` paged topology). Reads
//!   parse the directory and feed each topology blob into
//!   [`super::paged_topology::deserialize_topology`]. Writes always use
//!   this format.
//! - **v1 bincode (legacy):** preserved as a one-release fallback so
//!   existing `.grafeo` files keep loading after upgrade. Detected by
//!   the absence of the `GVST` magic at offset 0.
//!
//! On the next checkpoint after a v1→v2 read, the section serializes
//! the in-memory topologies as v2, completing the migration.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use bytes::Bytes;
use serde::{Deserialize, Serialize};

use grafeo_common::storage::section::{Section, SectionType};
use grafeo_common::types::NodeId;
use grafeo_common::utils::error::{Error, Result};

use super::paged_topology::{deserialize_topology, serialize_topology};
use super::{DistanceMetric, VectorIndexKind};

/// Current vector store section format version.
///
/// Phase 7b: bumped from 1 (bincode) to 2 (paged envelope).
const VECTOR_SECTION_VERSION: u8 = 2;

/// First 4 bytes of the v2 envelope; absent in v1 bincode output.
const V2_MAGIC: &[u8; 4] = b"GVST";

/// v2 envelope header size (magic + version + reserved + num_indexes).
const V2_HEADER_SIZE: usize = 16;

/// v2 directory entry size (meta_offset + meta_len + topology_offset + topology_len).
const V2_DIR_ENTRY_SIZE: usize = 32;

// ── v1 (legacy) snapshot types ─────────────────────────────────────

#[derive(Serialize, Deserialize)]
struct VectorStoreSnapshotV1 {
    version: u8,
    indexes: Vec<IndexSnapshotV1>,
}

#[derive(Serialize, Deserialize)]
struct IndexSnapshotV1 {
    /// Index key: "label:property"
    key: String,
    /// HNSW configuration
    dimensions: usize,
    metric: DistanceMetric,
    m: usize,
    ef_construction: usize,
    /// Topology
    entry_point: Option<NodeId>,
    max_level: usize,
    /// Node neighbors: Vec<(NodeId, Vec<Vec<NodeId>>)>
    nodes: Vec<(NodeId, Vec<Vec<NodeId>>)>,
}

// ── v2 (current) per-index metadata ────────────────────────────────

#[derive(Serialize, Deserialize)]
struct IndexMetaV2 {
    /// Index key: "label:property"
    key: String,
    dimensions: usize,
    metric: DistanceMetric,
    m: usize,
    ef_construction: usize,
}

// ── Section implementation ──────────────────────────────────────────

/// Vector Store section for the `.grafeo` container.
///
/// Wraps a collection of `(key, Arc<VectorIndexKind>)` pairs and serializes
/// their HNSW topologies for persistence.
pub struct VectorStoreSection {
    /// Vector indexes: (key, index) pairs from LpgStore::vector_index_entries()
    indexes: Vec<(String, Arc<VectorIndexKind>)>,
    dirty: AtomicBool,
}

impl VectorStoreSection {
    /// Create a new Vector Store section from the current indexes.
    pub fn new(indexes: Vec<(String, Arc<VectorIndexKind>)>) -> Self {
        Self {
            indexes,
            dirty: AtomicBool::new(false),
        }
    }

    /// Mark this section as dirty.
    pub fn mark_dirty(&self) {
        self.dirty.store(true, Ordering::Release);
    }
}

/// Serializes all in-memory indexes to the v2 paged envelope.
///
/// Layout: 16-byte header (`GVST` magic, version, num_indexes) + index
/// directory (32 bytes/entry: meta_offset, meta_len, topology_offset,
/// topology_len) + bincode'd metadata blobs + per-index `GTOP` paged
/// topology blobs.
fn serialize_v2(indexes: &[(String, Arc<VectorIndexKind>)]) -> Result<Vec<u8>> {
    let bincode_config = bincode::config::standard();

    // Build per-index meta blobs and topology blobs upfront so we can
    // compute absolute offsets.
    let mut meta_blobs: Vec<Vec<u8>> = Vec::with_capacity(indexes.len());
    let mut topology_blobs: Vec<Vec<u8>> = Vec::with_capacity(indexes.len());

    for (key, index) in indexes {
        let config = index.config();
        let meta = IndexMetaV2 {
            key: key.clone(),
            dimensions: config.dimensions,
            metric: config.metric,
            m: config.m,
            ef_construction: config.ef_construction,
        };
        let meta_bytes = bincode::serde::encode_to_vec(&meta, bincode_config).map_err(|e| {
            Error::Internal(format!("Vector Store v2 meta serialization failed: {e}"))
        })?;
        meta_blobs.push(meta_bytes);

        let (entry_point, max_level, nodes) = index.snapshot_topology();
        let topology_bytes = serialize_topology(entry_point, max_level, &nodes);
        topology_blobs.push(topology_bytes);
    }

    let n = indexes.len();
    let header_size = V2_HEADER_SIZE;
    let dir_size = n * V2_DIR_ENTRY_SIZE;
    let body_start = header_size + dir_size;

    // Compute absolute offsets for each meta + topology blob.
    let mut meta_offsets: Vec<u64> = Vec::with_capacity(n);
    let mut topology_offsets: Vec<u64> = Vec::with_capacity(n);
    let mut cursor = body_start;
    for blob in &meta_blobs {
        meta_offsets.push(cursor as u64);
        cursor += blob.len();
    }
    for blob in &topology_blobs {
        topology_offsets.push(cursor as u64);
        cursor += blob.len();
    }

    let mut buf = Vec::with_capacity(cursor);

    // Header
    buf.extend_from_slice(V2_MAGIC);
    buf.push(VECTOR_SECTION_VERSION);
    buf.extend_from_slice(&[0u8; 3]);
    buf.extend_from_slice(&(n as u64).to_le_bytes());
    debug_assert_eq!(buf.len(), V2_HEADER_SIZE);

    // Directory
    for i in 0..n {
        buf.extend_from_slice(&meta_offsets[i].to_le_bytes());
        buf.extend_from_slice(&(meta_blobs[i].len() as u64).to_le_bytes());
        buf.extend_from_slice(&topology_offsets[i].to_le_bytes());
        buf.extend_from_slice(&(topology_blobs[i].len() as u64).to_le_bytes());
    }
    debug_assert_eq!(buf.len(), header_size + dir_size);

    // Body: meta blobs first, then topology blobs (matches the offsets
    // computed above).
    for blob in &meta_blobs {
        buf.extend_from_slice(blob);
    }
    for blob in &topology_blobs {
        buf.extend_from_slice(blob);
    }

    Ok(buf)
}

/// Restores indexes from a v2 paged envelope.
fn deserialize_v2(data: &[u8], indexes: &mut [(String, Arc<VectorIndexKind>)]) -> Result<()> {
    let bincode_config = bincode::config::standard();

    if data.len() < V2_HEADER_SIZE {
        return Err(Error::Serialization(
            "Vector Store v2 header truncated".to_string(),
        ));
    }
    if &data[0..4] != V2_MAGIC {
        return Err(Error::Serialization(
            "Vector Store v2 bad magic".to_string(),
        ));
    }
    let version = data[4];
    if version != VECTOR_SECTION_VERSION {
        return Err(Error::Serialization(format!(
            "Vector Store v2 unsupported version: {version}"
        )));
    }
    let n_u64 = u64::from_le_bytes(
        data[8..16]
            .try_into()
            .expect("slice length 8 fits u64 array"),
    );
    let n =
        usize::try_from(n_u64).map_err(|_| Error::Serialization("v2 n_indexes overflow".into()))?;

    let dir_size = n
        .checked_mul(V2_DIR_ENTRY_SIZE)
        .ok_or_else(|| Error::Serialization("v2 directory size overflow".into()))?;
    let body_start = V2_HEADER_SIZE
        .checked_add(dir_size)
        .ok_or_else(|| Error::Serialization("v2 directory size overflow".into()))?;
    if data.len() < body_start {
        return Err(Error::Serialization(format!(
            "Vector Store v2 directory truncated: expected {body_start} bytes, got {}",
            data.len()
        )));
    }

    for i in 0..n {
        let dir_off = V2_HEADER_SIZE + i * V2_DIR_ENTRY_SIZE;
        let meta_off = u64::from_le_bytes(
            data[dir_off..dir_off + 8]
                .try_into()
                .expect("slice length 8 fits u64 array"),
        );
        let meta_len = u64::from_le_bytes(
            data[dir_off + 8..dir_off + 16]
                .try_into()
                .expect("slice length 8 fits u64 array"),
        );
        let topology_off = u64::from_le_bytes(
            data[dir_off + 16..dir_off + 24]
                .try_into()
                .expect("slice length 8 fits u64 array"),
        );
        let topology_len = u64::from_le_bytes(
            data[dir_off + 24..dir_off + 32]
                .try_into()
                .expect("slice length 8 fits u64 array"),
        );

        let meta_off_usize = usize::try_from(meta_off)
            .map_err(|_| Error::Serialization("v2 meta_off overflow".into()))?;
        let meta_len_usize = usize::try_from(meta_len)
            .map_err(|_| Error::Serialization("v2 meta_len overflow".into()))?;
        let topology_off_usize = usize::try_from(topology_off)
            .map_err(|_| Error::Serialization("v2 topology_off overflow".into()))?;
        let topology_len_usize = usize::try_from(topology_len)
            .map_err(|_| Error::Serialization("v2 topology_len overflow".into()))?;

        let meta_end = meta_off_usize
            .checked_add(meta_len_usize)
            .ok_or_else(|| Error::Serialization("v2 meta range overflow".into()))?;
        let topology_end = topology_off_usize
            .checked_add(topology_len_usize)
            .ok_or_else(|| Error::Serialization("v2 topology range overflow".into()))?;
        if meta_end > data.len() || topology_end > data.len() {
            return Err(Error::Serialization(format!(
                "Vector Store v2 directory entry {i} out of range"
            )));
        }

        let meta_bytes = &data[meta_off_usize..meta_end];
        let (meta, _): (IndexMetaV2, _) =
            bincode::serde::decode_from_slice(meta_bytes, bincode_config).map_err(|e| {
                Error::Serialization(format!("Vector Store v2 meta deserialization failed: {e}"))
            })?;

        // Find the matching index by key. v2 doesn't require ordering;
        // the section receives indexes in any order, so we look up by key.
        if let Some((_, index)) = indexes.iter().find(|(k, _)| *k == meta.key) {
            // Copy the topology bytes into a Bytes so the paged decoder
            // can hold them. Phase 7c will Bytes::from_owner the section
            // mmap directly and slice without copying.
            let topology_bytes = Bytes::copy_from_slice(&data[topology_off_usize..topology_end]);
            let (entry_point, max_level, nodes) =
                deserialize_topology(topology_bytes).map_err(|e| {
                    Error::Serialization(format!(
                        "Vector Store v2 topology decode failed for key '{}': {e}",
                        meta.key
                    ))
                })?;
            index.restore_topology(entry_point, max_level, nodes);
        }
    }

    Ok(())
}

/// Restores indexes from a v1 bincode envelope (legacy fallback).
fn deserialize_v1(data: &[u8], indexes: &mut [(String, Arc<VectorIndexKind>)]) -> Result<()> {
    let config = bincode::config::standard();
    let (snapshot, _): (VectorStoreSnapshotV1, _) = bincode::serde::decode_from_slice(data, config)
        .map_err(|e| {
            Error::Serialization(format!("Vector Store v1 deserialization failed: {e}"))
        })?;

    for idx_snap in snapshot.indexes {
        if let Some((_, index)) = indexes.iter().find(|(k, _)| *k == idx_snap.key) {
            index.restore_topology(idx_snap.entry_point, idx_snap.max_level, idx_snap.nodes);
        }
    }
    Ok(())
}

impl Section for VectorStoreSection {
    fn section_type(&self) -> SectionType {
        SectionType::VectorStore
    }

    fn version(&self) -> u8 {
        VECTOR_SECTION_VERSION
    }

    fn serialize(&self) -> Result<Vec<u8>> {
        serialize_v2(&self.indexes)
    }

    fn deserialize(&mut self, data: &[u8]) -> Result<()> {
        if data.is_empty() {
            return Ok(());
        }
        // Phase 7b: detect v2 packed vs v1 bincode by magic bytes.
        if data.len() >= 4 && &data[0..4] == V2_MAGIC {
            deserialize_v2(data, &mut self.indexes)
        } else {
            // v1 fallback: bincode-encoded VectorStoreSnapshotV1.
            // Existing files keep loading; the next checkpoint flushes
            // them out as v2.
            deserialize_v1(data, &mut self.indexes)
        }
    }

    fn is_dirty(&self) -> bool {
        self.dirty.load(Ordering::Acquire)
    }

    fn mark_clean(&self) {
        self.dirty.store(false, Ordering::Release);
    }

    fn memory_usage(&self) -> usize {
        self.indexes
            .iter()
            .map(|(_, idx)| idx.heap_memory_bytes())
            .sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::index::vector::{HnswConfig, HnswIndex};

    fn make_test_index() -> (String, Arc<VectorIndexKind>) {
        let config = HnswConfig::new(4, DistanceMetric::Cosine);
        let index = Arc::new(VectorIndexKind::Hnsw(HnswIndex::new(config)));

        // Manually set up a small topology via snapshot/restore
        let nodes = vec![
            (NodeId::new(1), vec![vec![NodeId::new(2), NodeId::new(3)]]),
            (NodeId::new(2), vec![vec![NodeId::new(1), NodeId::new(3)]]),
            (NodeId::new(3), vec![vec![NodeId::new(1), NodeId::new(2)]]),
        ];
        index.restore_topology(Some(NodeId::new(1)), 0, nodes);

        ("Item:embedding".to_string(), index)
    }

    fn make_v1_snapshot_bytes(key: &str) -> Vec<u8> {
        // Encode a v1 bincode snapshot directly so we can prove the
        // legacy fallback path on real bytes.
        let snapshot = VectorStoreSnapshotV1 {
            version: 1,
            indexes: vec![IndexSnapshotV1 {
                key: key.to_string(),
                dimensions: 4,
                metric: DistanceMetric::Cosine,
                m: 16,
                ef_construction: 200,
                entry_point: Some(NodeId::new(1)),
                max_level: 0,
                nodes: vec![
                    (NodeId::new(1), vec![vec![NodeId::new(2), NodeId::new(3)]]),
                    (NodeId::new(2), vec![vec![NodeId::new(1), NodeId::new(3)]]),
                    (NodeId::new(3), vec![vec![NodeId::new(1), NodeId::new(2)]]),
                ],
            }],
        };
        bincode::serde::encode_to_vec(&snapshot, bincode::config::standard())
            .expect("v1 bincode encode")
    }

    #[test]
    fn vector_section_round_trip() {
        let (key, index) = make_test_index();
        let section = VectorStoreSection::new(vec![(key.clone(), Arc::clone(&index))]);

        let bytes = section.serialize().expect("serialize should succeed");
        assert!(!bytes.is_empty(), "bytes is empty");

        // Create a fresh index with same config to restore into
        let config = index.config().clone();
        let fresh_index = Arc::new(VectorIndexKind::Hnsw(HnswIndex::new(config)));
        let mut section2 = VectorStoreSection::new(vec![(key, fresh_index.clone())]);
        section2
            .deserialize(&bytes)
            .expect("deserialize should succeed");

        assert_eq!(fresh_index.len(), 3);
        let (ep, ml, nodes) = fresh_index.snapshot_topology();
        assert_eq!(ep, Some(NodeId::new(1)));
        assert_eq!(ml, 0);
        assert_eq!(nodes.len(), 3);
    }

    #[test]
    fn vector_section_empty() {
        let section = VectorStoreSection::new(vec![]);
        let bytes = section.serialize().expect("serialize should succeed");

        let mut section2 = VectorStoreSection::new(vec![]);
        section2
            .deserialize(&bytes)
            .expect("deserialize should succeed");
    }

    #[test]
    fn vector_section_type() {
        let section = VectorStoreSection::new(vec![]);
        assert_eq!(section.section_type(), SectionType::VectorStore);
        // Phase 7b: bumped from 1 (bincode) to 2 (paged envelope).
        assert_eq!(section.version(), 2);
    }

    #[test]
    fn vector_section_dirty_tracking() {
        let section = VectorStoreSection::new(vec![]);
        assert!(!section.is_dirty());
        section.mark_dirty();
        assert!(section.is_dirty());
        section.mark_clean();
        assert!(!section.is_dirty());
    }

    // ── Phase 7b: format detection + v1 → v2 migration ───────────────

    /// New writes produce a v2 buffer (starts with `GVST` magic).
    #[test]
    fn alix_section_serialize_writes_v2_magic() {
        let (key, index) = make_test_index();
        let section = VectorStoreSection::new(vec![(key, Arc::clone(&index))]);
        let bytes = section.serialize().expect("serialize should succeed");
        assert!(bytes.len() > 4);
        assert_eq!(&bytes[0..4], V2_MAGIC, "new writes must use v2 magic");
    }

    /// v1 bincode-encoded buffers still deserialize correctly. The
    /// check uses a directly-constructed v1 snapshot, guaranteeing the
    /// migration path works for files written by older Grafeo versions.
    #[test]
    fn gus_section_v1_bincode_buffer_still_loads() {
        let v1_bytes = make_v1_snapshot_bytes("Item:embedding");
        // Sanity: v1 bytes do NOT start with GVST.
        assert_ne!(
            &v1_bytes[0..4],
            V2_MAGIC,
            "v1 bincode must not have GVST magic"
        );

        let config = HnswConfig::new(4, DistanceMetric::Cosine);
        let fresh = Arc::new(VectorIndexKind::Hnsw(HnswIndex::new(config)));
        let mut section =
            VectorStoreSection::new(vec![("Item:embedding".to_string(), Arc::clone(&fresh))]);
        section
            .deserialize(&v1_bytes)
            .expect("v1 fallback path must load");

        assert_eq!(fresh.len(), 3);
        let (ep, ml, nodes) = fresh.snapshot_topology();
        assert_eq!(ep, Some(NodeId::new(1)));
        assert_eq!(ml, 0);
        assert_eq!(nodes.len(), 3);
    }

    /// After a v1 read + a re-serialize, the new buffer is v2.
    /// Demonstrates the on-checkpoint migration.
    #[test]
    fn vincent_section_v1_then_reserialize_yields_v2() {
        let v1_bytes = make_v1_snapshot_bytes("Item:embedding");
        let config = HnswConfig::new(4, DistanceMetric::Cosine);
        let fresh = Arc::new(VectorIndexKind::Hnsw(HnswIndex::new(config)));
        let mut section =
            VectorStoreSection::new(vec![("Item:embedding".to_string(), Arc::clone(&fresh))]);
        section.deserialize(&v1_bytes).expect("v1 load");

        // Re-serialize: now in v2.
        let v2_bytes = section.serialize().expect("v2 serialize");
        assert_eq!(&v2_bytes[0..4], V2_MAGIC, "post-migration write is v2");

        // And v2 round-trips cleanly.
        let config2 = HnswConfig::new(4, DistanceMetric::Cosine);
        let restored = Arc::new(VectorIndexKind::Hnsw(HnswIndex::new(config2)));
        let mut section2 =
            VectorStoreSection::new(vec![("Item:embedding".to_string(), Arc::clone(&restored))]);
        section2.deserialize(&v2_bytes).expect("v2 load");
        assert_eq!(restored.len(), 3);
    }

    /// v2 with multiple indexes round-trips by key, including indexes
    /// with different shapes.
    #[test]
    fn jules_section_v2_multiple_indexes_round_trip() {
        let cfg_a = HnswConfig::new(4, DistanceMetric::Cosine);
        let idx_a = Arc::new(VectorIndexKind::Hnsw(HnswIndex::new(cfg_a)));
        idx_a.restore_topology(
            Some(NodeId::new(10)),
            1,
            vec![
                (NodeId::new(10), vec![vec![NodeId::new(20)], vec![]]),
                (NodeId::new(20), vec![vec![NodeId::new(10)]]),
            ],
        );

        let cfg_b = HnswConfig::new(8, DistanceMetric::Euclidean);
        let idx_b = Arc::new(VectorIndexKind::Hnsw(HnswIndex::new(cfg_b)));
        idx_b.restore_topology(
            Some(NodeId::new(100)),
            0,
            vec![(NodeId::new(100), vec![vec![]])],
        );

        let section = VectorStoreSection::new(vec![
            ("Doc:embedding".to_string(), Arc::clone(&idx_a)),
            ("User:embedding".to_string(), Arc::clone(&idx_b)),
        ]);
        let bytes = section.serialize().expect("v2 serialize");

        // Restore into fresh indexes and verify topology counts and
        // entry points match.
        let restored_a = Arc::new(VectorIndexKind::Hnsw(HnswIndex::new(HnswConfig::new(
            4,
            DistanceMetric::Cosine,
        ))));
        let restored_b = Arc::new(VectorIndexKind::Hnsw(HnswIndex::new(HnswConfig::new(
            8,
            DistanceMetric::Euclidean,
        ))));
        let mut section2 = VectorStoreSection::new(vec![
            ("Doc:embedding".to_string(), Arc::clone(&restored_a)),
            ("User:embedding".to_string(), Arc::clone(&restored_b)),
        ]);
        section2.deserialize(&bytes).expect("v2 load");

        assert_eq!(restored_a.len(), 2);
        assert_eq!(restored_b.len(), 1);
        let (ep_a, _, _) = restored_a.snapshot_topology();
        let (ep_b, _, _) = restored_b.snapshot_topology();
        assert_eq!(ep_a, Some(NodeId::new(10)));
        assert_eq!(ep_b, Some(NodeId::new(100)));
    }

    /// Truncated v2 envelope is rejected without panicking.
    #[test]
    fn shosanna_section_truncated_v2_rejected() {
        let (key, index) = make_test_index();
        let section = VectorStoreSection::new(vec![(key.clone(), Arc::clone(&index))]);
        let bytes = section.serialize().expect("v2 serialize");

        // Truncate to less than the v2 header.
        let truncated = &bytes[..8];
        let fresh = Arc::new(VectorIndexKind::Hnsw(HnswIndex::new(
            index.config().clone(),
        )));
        let mut section2 = VectorStoreSection::new(vec![(key, fresh)]);
        let err = section2
            .deserialize(truncated)
            .expect_err("must reject truncated v2");
        match err {
            Error::Serialization(_) => {}
            other => panic!("unexpected error variant: {other:?}"),
        }
    }
}
