//! Compressed epoch storage for tiered hot/cold architecture.
//!
//! When epochs are no longer needed by active transactions, their data is frozen
//! from arena storage into compressed, immutable epoch blocks. This enables:
//!
//! - Batch deallocation of arena memory
//! - Better compression ratios (data is sorted and grouped)
//! - Zone maps for predicate pushdown (skip decompression for filtered blocks)
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────┐
//! │                         EpochStore                              │
//! │  ┌──────────────────────────────────────────────────────────┐   │
//! │  │  blocks: HashMap<EpochId, CompressedEpochBlock>          │   │
//! │  └──────────────────────────────────────────────────────────┘   │
//! │                              │                                  │
//! │              ┌───────────────┴───────────────┐                  │
//! │              ▼                               ▼                  │
//! │  ┌──────────────────────┐       ┌──────────────────────┐       │
//! │  │ CompressedEpochBlock │       │ CompressedEpochBlock │       │
//! │  │  ├─ header           │       │  ├─ header           │       │
//! │  │  │   └─ zone_map     │       │  │   └─ zone_map     │       │
//! │  │  ├─ node_index       │       │  ├─ node_index       │       │
//! │  │  ├─ node_data        │       │  ├─ node_data        │       │
//! │  │  ├─ edge_index       │       │  ├─ edge_index       │       │
//! │  │  └─ edge_data        │       │  └─ edge_data        │       │
//! │  └──────────────────────┘       └──────────────────────┘       │
//! └─────────────────────────────────────────────────────────────────┘
//! ```
//!
//! # Usage
//!
//! ```ignore
//! use grafeo_core::storage::EpochStore;
//!
//! let mut store = EpochStore::new();
//!
//! // Freeze an epoch (called by TransactionManager)
//! let node_entries = vec![(node_id, node_record), ...];
//! let edge_entries = vec![(edge_id, edge_record), ...];
//! let (node_refs, edge_refs) = store.freeze_epoch(epoch, &node_entries, &edge_entries);
//!
//! // Read back a node from cold storage
//! let node_record = store.get_node(&cold_ref)?;
//! ```

use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};

use grafeo_common::types::EpochId;
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};

use super::codec::CompressionCodec;
use crate::graph::lpg::{EdgeRecord, NodeRecord};

/// Compression type used for epoch blocks.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum CompressionType {
    /// No compression (raw bincode serialization).
    None,
    /// Dictionary encoding for strings/labels.
    Dictionary,
    /// Delta encoding for IDs.
    Delta,
    /// Combined dictionary + delta encoding.
    Combined,
}

impl Default for CompressionType {
    fn default() -> Self {
        Self::None
    }
}

impl From<CompressionCodec> for CompressionType {
    fn from(codec: CompressionCodec) -> Self {
        match codec {
            CompressionCodec::None => Self::None,
            CompressionCodec::Dictionary => Self::Dictionary,
            CompressionCodec::Delta | CompressionCodec::DeltaBitPacked { .. } => Self::Delta,
            _ => Self::None,
        }
    }
}

/// Zone map for predicate pushdown.
///
/// Zone maps track min/max values within a compressed block, enabling
/// query execution to skip decompression when filters can't match.
///
/// # Example
///
/// A query like `MATCH (n) WHERE n.id > 1000` can skip blocks where
/// `zone_map.max_node_id < 1000`.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ZoneMap {
    /// Minimum node ID in this block.
    pub min_node_id: u64,
    /// Maximum node ID in this block.
    pub max_node_id: u64,
    /// Minimum edge ID in this block.
    pub min_edge_id: u64,
    /// Maximum edge ID in this block.
    pub max_edge_id: u64,
    /// Minimum epoch for versions in this block.
    pub min_epoch: u64,
    /// Maximum epoch for versions in this block.
    pub max_epoch: u64,
    /// Number of nodes in this block.
    pub node_count: u32,
    /// Number of edges in this block.
    pub edge_count: u32,
}

impl ZoneMap {
    /// Creates a new zone map from node and edge records.
    #[must_use]
    pub fn from_records(
        nodes: &[(u64, NodeRecord)],
        edges: &[(u64, EdgeRecord)],
        epoch: EpochId,
    ) -> Self {
        let (min_node_id, max_node_id) = if nodes.is_empty() {
            (u64::MAX, 0)
        } else {
            let min = nodes.iter().map(|(id, _)| *id).min().unwrap_or(u64::MAX);
            let max = nodes.iter().map(|(id, _)| *id).max().unwrap_or(0);
            (min, max)
        };

        let (min_edge_id, max_edge_id) = if edges.is_empty() {
            (u64::MAX, 0)
        } else {
            let min = edges.iter().map(|(id, _)| *id).min().unwrap_or(u64::MAX);
            let max = edges.iter().map(|(id, _)| *id).max().unwrap_or(0);
            (min, max)
        };

        Self {
            min_node_id,
            max_node_id,
            min_edge_id,
            max_edge_id,
            min_epoch: epoch.as_u64(),
            max_epoch: epoch.as_u64(),
            node_count: nodes.len() as u32,
            edge_count: edges.len() as u32,
        }
    }

    /// Checks if a node ID might be in this block.
    #[must_use]
    pub fn might_contain_node(&self, node_id: u64) -> bool {
        self.node_count > 0 && node_id >= self.min_node_id && node_id <= self.max_node_id
    }

    /// Checks if an edge ID might be in this block.
    #[must_use]
    pub fn might_contain_edge(&self, edge_id: u64) -> bool {
        self.edge_count > 0 && edge_id >= self.min_edge_id && edge_id <= self.max_edge_id
    }
}

/// Header for a compressed epoch block.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EpochBlockHeader {
    /// Epoch this block represents.
    pub epoch: u64,
    /// Compression type used.
    pub compression_type: CompressionType,
    /// Zone map for predicate pushdown.
    pub zone_map: ZoneMap,
    /// Compressed size of node data in bytes.
    pub node_data_size: u32,
    /// Compressed size of edge data in bytes.
    pub edge_data_size: u32,
    /// Uncompressed size of node data in bytes.
    pub node_uncompressed_size: u32,
    /// Uncompressed size of edge data in bytes.
    pub edge_uncompressed_size: u32,
}

/// Index entry for locating an entity within compressed data.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct IndexEntry {
    /// Entity ID (NodeId or EdgeId as u64).
    pub entity_id: u64,
    /// Offset within the compressed data.
    pub offset: u32,
    /// Length of the serialized record.
    pub length: u16,
}

/// A compressed, immutable epoch block.
///
/// Contains all nodes and edges created during a single epoch,
/// compressed together for efficient storage and batch operations.
#[derive(Debug, Clone)]
pub struct CompressedEpochBlock {
    /// Block header with metadata.
    header: EpochBlockHeader,
    /// Index for locating nodes by ID.
    node_index: Vec<IndexEntry>,
    /// Compressed node data.
    node_data: Vec<u8>,
    /// Index for locating edges by ID.
    edge_index: Vec<IndexEntry>,
    /// Compressed edge data.
    edge_data: Vec<u8>,
}

impl CompressedEpochBlock {
    /// Creates a new compressed epoch block from node and edge records.
    ///
    /// # Arguments
    ///
    /// * `epoch` - The epoch being frozen
    /// * `nodes` - Node records to compress (id, record)
    /// * `edges` - Edge records to compress (id, record)
    ///
    /// # Returns
    ///
    /// A tuple containing:
    /// - The compressed block
    /// - Node index entries (for creating `ColdVersionRef`)
    /// - Edge index entries (for creating `ColdVersionRef`)
    #[must_use]
    pub fn from_records(
        epoch: EpochId,
        mut nodes: Vec<(u64, NodeRecord)>,
        mut edges: Vec<(u64, EdgeRecord)>,
    ) -> (Self, Vec<IndexEntry>, Vec<IndexEntry>) {
        // Sort by ID for better compression and binary search
        nodes.sort_unstable_by_key(|(id, _)| *id);
        edges.sort_unstable_by_key(|(id, _)| *id);

        // Build zone map
        let zone_map = ZoneMap::from_records(&nodes, &edges, epoch);

        // Serialize nodes
        let config = bincode::config::standard();
        let mut node_data = Vec::new();
        let mut node_index = Vec::with_capacity(nodes.len());

        for (id, record) in &nodes {
            let offset = node_data.len() as u32;
            let serialized = bincode::serde::encode_to_vec(record, config)
                .expect("NodeRecord serialization should not fail");
            let length = serialized.len() as u16;

            node_index.push(IndexEntry {
                entity_id: *id,
                offset,
                length,
            });
            node_data.extend_from_slice(&serialized);
        }

        // Serialize edges
        let mut edge_data = Vec::new();
        let mut edge_index = Vec::with_capacity(edges.len());

        for (id, record) in &edges {
            let offset = edge_data.len() as u32;
            let serialized = bincode::serde::encode_to_vec(record, config)
                .expect("EdgeRecord serialization should not fail");
            let length = serialized.len() as u16;

            edge_index.push(IndexEntry {
                entity_id: *id,
                offset,
                length,
            });
            edge_data.extend_from_slice(&serialized);
        }

        // Calculate uncompressed sizes (same as compressed for now)
        let node_uncompressed_size = node_data.len() as u32;
        let edge_uncompressed_size = edge_data.len() as u32;

        let header = EpochBlockHeader {
            epoch: epoch.as_u64(),
            compression_type: CompressionType::None, // Future: add compression
            zone_map,
            node_data_size: node_data.len() as u32,
            edge_data_size: edge_data.len() as u32,
            node_uncompressed_size,
            edge_uncompressed_size,
        };

        let block = Self {
            header,
            node_index: node_index.clone(),
            node_data,
            edge_index: edge_index.clone(),
            edge_data,
        };

        (block, node_index, edge_index)
    }

    /// Returns the epoch this block represents.
    #[must_use]
    pub fn epoch(&self) -> EpochId {
        EpochId::new(self.header.epoch)
    }

    /// Returns the block header.
    #[must_use]
    pub fn header(&self) -> &EpochBlockHeader {
        &self.header
    }

    /// Returns the zone map for predicate pushdown.
    #[must_use]
    pub fn zone_map(&self) -> &ZoneMap {
        &self.header.zone_map
    }

    /// Gets a node record by offset and length.
    ///
    /// This is the primary read path for cold storage.
    #[must_use]
    pub fn get_node(&self, offset: u32, length: u16) -> Option<NodeRecord> {
        let start = offset as usize;
        let end = start + length as usize;

        if end > self.node_data.len() {
            return None;
        }

        let data = &self.node_data[start..end];
        let config = bincode::config::standard();

        bincode::serde::decode_from_slice::<NodeRecord, _>(data, config)
            .ok()
            .map(|(record, _)| record)
    }

    /// Gets a node record by entity ID.
    ///
    /// Uses binary search on the sorted index.
    #[must_use]
    pub fn get_node_by_id(&self, node_id: u64) -> Option<NodeRecord> {
        // Quick zone map check
        if !self.header.zone_map.might_contain_node(node_id) {
            return None;
        }

        // Binary search in index
        let index_entry = self
            .node_index
            .binary_search_by_key(&node_id, |e| e.entity_id)
            .ok()
            .map(|idx| &self.node_index[idx])?;

        self.get_node(index_entry.offset, index_entry.length)
    }

    /// Gets an edge record by offset and length.
    ///
    /// This is the primary read path for cold storage.
    #[must_use]
    pub fn get_edge(&self, offset: u32, length: u16) -> Option<EdgeRecord> {
        let start = offset as usize;
        let end = start + length as usize;

        if end > self.edge_data.len() {
            return None;
        }

        let data = &self.edge_data[start..end];
        let config = bincode::config::standard();

        bincode::serde::decode_from_slice::<EdgeRecord, _>(data, config)
            .ok()
            .map(|(record, _)| record)
    }

    /// Gets an edge record by entity ID.
    ///
    /// Uses binary search on the sorted index.
    #[must_use]
    pub fn get_edge_by_id(&self, edge_id: u64) -> Option<EdgeRecord> {
        // Quick zone map check
        if !self.header.zone_map.might_contain_edge(edge_id) {
            return None;
        }

        // Binary search in index
        let index_entry = self
            .edge_index
            .binary_search_by_key(&edge_id, |e| e.entity_id)
            .ok()
            .map(|idx| &self.edge_index[idx])?;

        self.get_edge(index_entry.offset, index_entry.length)
    }

    /// Returns the total compressed size of this block.
    #[must_use]
    pub fn compressed_size(&self) -> usize {
        self.node_data.len() + self.edge_data.len()
    }

    /// Returns the compression ratio (uncompressed / compressed).
    #[must_use]
    pub fn compression_ratio(&self) -> f64 {
        let compressed = self.compressed_size();
        if compressed == 0 {
            return 1.0;
        }
        let uncompressed =
            (self.header.node_uncompressed_size + self.header.edge_uncompressed_size) as usize;
        uncompressed as f64 / compressed as f64
    }

    /// Returns the number of nodes in this block.
    #[must_use]
    pub fn node_count(&self) -> usize {
        self.node_index.len()
    }

    /// Returns the number of edges in this block.
    #[must_use]
    pub fn edge_count(&self) -> usize {
        self.edge_index.len()
    }
}

/// Manages all compressed epoch blocks.
///
/// The `EpochStore` is the central manager for cold storage. It handles:
///
/// - Freezing epochs from arena to compressed blocks
/// - Looking up entities in compressed storage
/// - Garbage collection of old epochs
///
/// # Thread Safety
///
/// The store uses `RwLock` for concurrent access. Multiple readers can
/// access the same block simultaneously, but writes (freeze/gc) require
/// exclusive access.
pub struct EpochStore {
    /// Epoch ID → compressed block.
    blocks: RwLock<HashMap<EpochId, CompressedEpochBlock>>,
    /// Total compressed bytes across all blocks.
    total_size: AtomicUsize,
    /// Number of frozen epochs.
    epoch_count: AtomicUsize,
}

impl Default for EpochStore {
    fn default() -> Self {
        Self::new()
    }
}

impl EpochStore {
    /// Creates a new empty epoch store.
    #[must_use]
    pub fn new() -> Self {
        Self {
            blocks: RwLock::new(HashMap::new()),
            total_size: AtomicUsize::new(0),
            epoch_count: AtomicUsize::new(0),
        }
    }

    /// Freezes an epoch from arena records into compressed storage.
    ///
    /// # Arguments
    ///
    /// * `epoch` - The epoch being frozen
    /// * `nodes` - Node records to freeze (id, record)
    /// * `edges` - Edge records to freeze (id, record)
    ///
    /// # Returns
    ///
    /// A tuple containing:
    /// - Node index entries (offset, length for each node)
    /// - Edge index entries (offset, length for each edge)
    ///
    /// These are used to create `ColdVersionRef` entries in `VersionIndex`.
    pub fn freeze_epoch(
        &self,
        epoch: EpochId,
        nodes: Vec<(u64, NodeRecord)>,
        edges: Vec<(u64, EdgeRecord)>,
    ) -> (Vec<IndexEntry>, Vec<IndexEntry>) {
        let (block, node_entries, edge_entries) =
            CompressedEpochBlock::from_records(epoch, nodes, edges);

        let size = block.compressed_size();

        let mut blocks = self.blocks.write();
        blocks.insert(epoch, block);

        self.total_size.fetch_add(size, Ordering::Relaxed);
        self.epoch_count.fetch_add(1, Ordering::Relaxed);

        (node_entries, edge_entries)
    }

    /// Gets a node record from cold storage.
    ///
    /// This is the primary read path for cold node data.
    #[must_use]
    pub fn get_node(&self, epoch: EpochId, offset: u32, length: u16) -> Option<NodeRecord> {
        let blocks = self.blocks.read();
        blocks.get(&epoch)?.get_node(offset, length)
    }

    /// Gets an edge record from cold storage.
    ///
    /// This is the primary read path for cold edge data.
    #[must_use]
    pub fn get_edge(&self, epoch: EpochId, offset: u32, length: u16) -> Option<EdgeRecord> {
        let blocks = self.blocks.read();
        blocks.get(&epoch)?.get_edge(offset, length)
    }

    /// Gets a node record by entity ID from a specific epoch.
    #[must_use]
    pub fn get_node_by_id(&self, epoch: EpochId, node_id: u64) -> Option<NodeRecord> {
        let blocks = self.blocks.read();
        blocks.get(&epoch)?.get_node_by_id(node_id)
    }

    /// Gets an edge record by entity ID from a specific epoch.
    #[must_use]
    pub fn get_edge_by_id(&self, epoch: EpochId, edge_id: u64) -> Option<EdgeRecord> {
        let blocks = self.blocks.read();
        blocks.get(&epoch)?.get_edge_by_id(edge_id)
    }

    /// Checks if an epoch has been frozen.
    #[must_use]
    pub fn contains_epoch(&self, epoch: EpochId) -> bool {
        self.blocks.read().contains_key(&epoch)
    }

    /// Returns the compressed block for an epoch.
    #[must_use]
    pub fn get_block(&self, epoch: EpochId) -> Option<CompressedEpochBlock> {
        self.blocks.read().get(&epoch).cloned()
    }

    /// Garbage collects epochs older than the watermark.
    ///
    /// # Returns
    ///
    /// The number of epochs removed.
    pub fn gc(&self, min_epoch: EpochId) -> usize {
        let mut blocks = self.blocks.write();
        let mut removed = 0;
        let mut freed_size = 0;

        blocks.retain(|epoch, block| {
            if epoch.as_u64() < min_epoch.as_u64() {
                freed_size += block.compressed_size();
                removed += 1;
                false
            } else {
                true
            }
        });

        if removed > 0 {
            self.total_size.fetch_sub(freed_size, Ordering::Relaxed);
            self.epoch_count.fetch_sub(removed, Ordering::Relaxed);
        }

        removed
    }

    /// Returns the total compressed size in bytes.
    #[must_use]
    pub fn total_size(&self) -> usize {
        self.total_size.load(Ordering::Relaxed)
    }

    /// Returns the number of frozen epochs.
    #[must_use]
    pub fn epoch_count(&self) -> usize {
        self.epoch_count.load(Ordering::Relaxed)
    }

    /// Returns statistics about the store.
    #[must_use]
    pub fn stats(&self) -> EpochStoreStats {
        let blocks = self.blocks.read();
        let mut total_nodes = 0;
        let mut total_edges = 0;
        let mut total_compressed = 0;
        let mut total_uncompressed = 0;

        for block in blocks.values() {
            total_nodes += block.node_count();
            total_edges += block.edge_count();
            total_compressed += block.compressed_size();
            total_uncompressed += (block.header.node_uncompressed_size
                + block.header.edge_uncompressed_size) as usize;
        }

        EpochStoreStats {
            epoch_count: blocks.len(),
            total_nodes,
            total_edges,
            total_compressed_bytes: total_compressed,
            total_uncompressed_bytes: total_uncompressed,
            compression_ratio: if total_compressed > 0 {
                total_uncompressed as f64 / total_compressed as f64
            } else {
                1.0
            },
        }
    }
}

/// Statistics about the epoch store.
#[derive(Debug, Clone)]
pub struct EpochStoreStats {
    /// Number of frozen epochs.
    pub epoch_count: usize,
    /// Total number of nodes in cold storage.
    pub total_nodes: usize,
    /// Total number of edges in cold storage.
    pub total_edges: usize,
    /// Total compressed bytes.
    pub total_compressed_bytes: usize,
    /// Total uncompressed bytes.
    pub total_uncompressed_bytes: usize,
    /// Overall compression ratio.
    pub compression_ratio: f64,
}

#[cfg(test)]
mod tests {
    use grafeo_common::types::{EdgeId, NodeId};

    use super::*;
    use crate::graph::lpg::{EdgeFlags, NodeFlags};

    fn make_node_record(id: u64, epoch: u64) -> NodeRecord {
        NodeRecord {
            id: NodeId::new(id),
            epoch: EpochId::new(epoch),
            props_offset: 0,
            label_count: 0,
            _reserved: 0,
            props_count: 0,
            flags: NodeFlags::default(),
            _padding: 0,
        }
    }

    fn make_edge_record(id: u64, src: u64, dst: u64, epoch: u64) -> EdgeRecord {
        EdgeRecord {
            id: EdgeId::new(id),
            src: NodeId::new(src),
            dst: NodeId::new(dst),
            type_id: 0,
            props_offset: 0,
            props_count: 0,
            flags: EdgeFlags::default(),
            epoch: EpochId::new(epoch),
        }
    }

    #[test]
    fn test_zone_map_creation() {
        let nodes = vec![
            (10, make_node_record(10, 1)),
            (20, make_node_record(20, 1)),
            (15, make_node_record(15, 1)),
        ];
        let edges = vec![
            (100, make_edge_record(100, 10, 20, 1)),
            (200, make_edge_record(200, 15, 20, 1)),
        ];

        let zone_map = ZoneMap::from_records(&nodes, &edges, EpochId::new(1));

        assert_eq!(zone_map.min_node_id, 10);
        assert_eq!(zone_map.max_node_id, 20);
        assert_eq!(zone_map.min_edge_id, 100);
        assert_eq!(zone_map.max_edge_id, 200);
        assert_eq!(zone_map.node_count, 3);
        assert_eq!(zone_map.edge_count, 2);
    }

    #[test]
    fn test_zone_map_predicate_pushdown() {
        let nodes = vec![(10, make_node_record(10, 1)), (20, make_node_record(20, 1))];
        let edges = vec![];

        let zone_map = ZoneMap::from_records(&nodes, &edges, EpochId::new(1));

        // Within range
        assert!(zone_map.might_contain_node(10));
        assert!(zone_map.might_contain_node(15));
        assert!(zone_map.might_contain_node(20));

        // Outside range
        assert!(!zone_map.might_contain_node(5));
        assert!(!zone_map.might_contain_node(25));

        // No edges
        assert!(!zone_map.might_contain_edge(100));
    }

    #[test]
    fn test_compressed_block_creation() {
        let nodes = vec![
            (1, make_node_record(1, 1)),
            (2, make_node_record(2, 1)),
            (3, make_node_record(3, 1)),
        ];
        let edges = vec![
            (10, make_edge_record(10, 1, 2, 1)),
            (20, make_edge_record(20, 2, 3, 1)),
        ];

        let (block, node_index, edge_index) =
            CompressedEpochBlock::from_records(EpochId::new(1), nodes, edges);

        assert_eq!(block.epoch().as_u64(), 1);
        assert_eq!(block.node_count(), 3);
        assert_eq!(block.edge_count(), 2);
        assert_eq!(node_index.len(), 3);
        assert_eq!(edge_index.len(), 2);
    }

    #[test]
    fn test_compressed_block_read_by_offset() {
        let nodes = vec![(1, make_node_record(1, 1)), (2, make_node_record(2, 1))];
        let edges = vec![(10, make_edge_record(10, 1, 2, 1))];

        let (block, node_index, edge_index) =
            CompressedEpochBlock::from_records(EpochId::new(1), nodes, edges);

        // Read node by offset
        let entry = &node_index[0];
        let node = block.get_node(entry.offset, entry.length).unwrap();
        assert_eq!(node.id.as_u64(), 1);

        // Read edge by offset
        let entry = &edge_index[0];
        let edge = block.get_edge(entry.offset, entry.length).unwrap();
        assert_eq!(edge.id.as_u64(), 10);
    }

    #[test]
    fn test_compressed_block_read_by_id() {
        let nodes = vec![
            (1, make_node_record(1, 1)),
            (5, make_node_record(5, 1)),
            (10, make_node_record(10, 1)),
        ];
        let edges = vec![
            (100, make_edge_record(100, 1, 5, 1)),
            (200, make_edge_record(200, 5, 10, 1)),
        ];

        let (block, _, _) = CompressedEpochBlock::from_records(EpochId::new(1), nodes, edges);

        // Read nodes by ID
        assert!(block.get_node_by_id(1).is_some());
        assert!(block.get_node_by_id(5).is_some());
        assert!(block.get_node_by_id(10).is_some());
        assert!(block.get_node_by_id(2).is_none()); // Not in block

        // Read edges by ID
        assert!(block.get_edge_by_id(100).is_some());
        assert!(block.get_edge_by_id(200).is_some());
        assert!(block.get_edge_by_id(150).is_none()); // Not in block
    }

    #[test]
    fn test_epoch_store_freeze_and_read() {
        let store = EpochStore::new();

        let nodes = vec![(1, make_node_record(1, 1)), (2, make_node_record(2, 1))];
        let edges = vec![(10, make_edge_record(10, 1, 2, 1))];

        let (node_entries, edge_entries) = store.freeze_epoch(EpochId::new(1), nodes, edges);

        assert_eq!(store.epoch_count(), 1);
        assert!(store.total_size() > 0);

        // Read node via cold ref
        let entry = &node_entries[0];
        let node = store
            .get_node(EpochId::new(1), entry.offset, entry.length)
            .unwrap();
        assert_eq!(node.id.as_u64(), 1);

        // Read edge via cold ref
        let entry = &edge_entries[0];
        let edge = store
            .get_edge(EpochId::new(1), entry.offset, entry.length)
            .unwrap();
        assert_eq!(edge.id.as_u64(), 10);
    }

    #[test]
    fn test_epoch_store_gc() {
        let store = EpochStore::new();

        // Freeze epochs 1, 2, 3
        for epoch in 1..=3 {
            let nodes = vec![(epoch, make_node_record(epoch, epoch))];
            store.freeze_epoch(EpochId::new(epoch), nodes, vec![]);
        }

        assert_eq!(store.epoch_count(), 3);

        // GC epochs < 3
        let removed = store.gc(EpochId::new(3));
        assert_eq!(removed, 2);
        assert_eq!(store.epoch_count(), 1);

        // Epoch 3 should still be accessible
        assert!(store.contains_epoch(EpochId::new(3)));
        assert!(!store.contains_epoch(EpochId::new(1)));
        assert!(!store.contains_epoch(EpochId::new(2)));
    }

    #[test]
    fn test_epoch_store_stats() {
        let store = EpochStore::new();

        let nodes = vec![
            (1, make_node_record(1, 1)),
            (2, make_node_record(2, 1)),
            (3, make_node_record(3, 1)),
        ];
        let edges = vec![
            (10, make_edge_record(10, 1, 2, 1)),
            (20, make_edge_record(20, 2, 3, 1)),
        ];

        store.freeze_epoch(EpochId::new(1), nodes, edges);

        let stats = store.stats();
        assert_eq!(stats.epoch_count, 1);
        assert_eq!(stats.total_nodes, 3);
        assert_eq!(stats.total_edges, 2);
        assert!(stats.total_compressed_bytes > 0);
    }

    #[test]
    fn test_empty_epoch_freeze() {
        let store = EpochStore::new();

        // Freeze empty epoch
        let (node_entries, edge_entries) = store.freeze_epoch(EpochId::new(1), vec![], vec![]);

        assert!(node_entries.is_empty());
        assert!(edge_entries.is_empty());
        assert_eq!(store.epoch_count(), 1);

        // Zone map should indicate no nodes/edges
        let block = store.get_block(EpochId::new(1)).unwrap();
        assert_eq!(block.zone_map().node_count, 0);
        assert_eq!(block.zone_map().edge_count, 0);
    }

    #[test]
    fn test_multiple_epochs() {
        let store = EpochStore::new();

        // Freeze multiple epochs
        for epoch in 1..=5 {
            let nodes: Vec<_> = (0..10)
                .map(|i| {
                    let id = epoch * 100 + i;
                    (id, make_node_record(id, epoch))
                })
                .collect();
            store.freeze_epoch(EpochId::new(epoch), nodes, vec![]);
        }

        assert_eq!(store.epoch_count(), 5);

        let stats = store.stats();
        assert_eq!(stats.total_nodes, 50); // 5 epochs * 10 nodes

        // Each epoch should be independently accessible
        for epoch in 1..=5 {
            let node_id = epoch * 100 + 5;
            let node = store.get_node_by_id(EpochId::new(epoch), node_id).unwrap();
            assert_eq!(node.id.as_u64(), node_id);
        }
    }
}
