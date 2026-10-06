//! [`Section`](grafeo_common::storage::section::Section) implementation for
//! the layered overlay deletion log.
//!
//! The [`LayeredStore`](crate::graph::compact::layered::LayeredStore)
//! tracks deletions of base-store entities in the in-memory
//! `deleted_from_base_nodes` / `deleted_from_base_edges` sets. Without
//! this section, those sets are lost across a close/reopen cycle: the
//! overlay scan in `LayeredStore::with_overlay` cannot distinguish a
//! deleted base node (which has no overlay entry) from a base node that
//! was never modified, so previously-deleted base entities would silently
//! reappear after reload until the next `compact()` merges the overlay
//! into the base.
//!
//! This section persists the deletion log alongside the rest of the
//! container so that reload restores the deleted sets verbatim.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use grafeo_common::storage::section::{Section, SectionType};
use grafeo_common::types::{EdgeId, NodeId};
use grafeo_common::utils::error::{Error, Result};
use parking_lot::RwLock;

#[cfg(feature = "lpg")]
use super::layered::LayeredStore;

/// Magic bytes identifying an OverlayDeletions section ("Grafeo Overlay
/// Deletion Log").
const MAGIC: [u8; 4] = *b"GODL";

/// Current section format version.
const FORMAT_VERSION: u8 = 1;

/// Snapshot of the layered overlay's deletion log, ready to be serialized
/// into the container or to seed a freshly-loaded `LayeredStore`.
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
}

#[derive(Default, Clone, Debug)]
struct DeletionsPayload {
    nodes: Vec<NodeId>,
    edges: Vec<EdgeId>,
}

impl OverlayDeletionsSection {
    /// Creates a section by snapshotting the layered store's current
    /// deletion sets. The snapshot is sorted (and deduplicated) so the
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

impl Section for OverlayDeletionsSection {
    fn section_type(&self) -> SectionType {
        SectionType::OverlayDeletions
    }

    fn version(&self) -> u8 {
        FORMAT_VERSION
    }

    fn serialize(&self) -> Result<Vec<u8>> {
        Ok(self.encode_payload())
    }

    fn deserialize(&mut self, data: &[u8]) -> Result<()> {
        let payload = Self::decode_payload(data)?;
        *self.payload.write() = payload;
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
    // which is what we want — there is no payoff in going through the
    // page-fetcher indirection for this section.
}

#[cfg(test)]
mod tests {
    use super::*;

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
        let section = OverlayDeletionsSection {
            payload: RwLock::new(DeletionsPayload {
                nodes: vec![NodeId(1), NodeId(7), NodeId(42)],
                edges: vec![EdgeId(3), EdgeId(99)],
            }),
            #[cfg(feature = "lpg")]
            layered: None,
            local_dirty: AtomicBool::new(true),
        };
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
        let original = OverlayDeletionsSection {
            payload: RwLock::new(DeletionsPayload {
                nodes: vec![NodeId(11)],
                edges: vec![],
            }),
            #[cfg(feature = "lpg")]
            layered: None,
            local_dirty: AtomicBool::new(true),
        };
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
}
