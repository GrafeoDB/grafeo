//! Chunked adjacency lists with delta buffers.
//!
//! This is the primary edge storage structure, optimized for:
//! - O(1) amortized edge insertion
//! - Cache-friendly sequential scans
//! - MVCC-compatible copy-on-write at chunk granularity
//! - Optional backward adjacency for incoming edge queries

use graphos_common::types::{EdgeId, NodeId};
use graphos_common::utils::hash::{FxHashMap, FxHashSet};
use parking_lot::RwLock;
use smallvec::SmallVec;
use std::sync::atomic::{AtomicUsize, Ordering};

/// Default chunk capacity (number of edges per chunk).
const DEFAULT_CHUNK_CAPACITY: usize = 64;

/// Threshold for delta buffer compaction.
const DELTA_COMPACTION_THRESHOLD: usize = 1024;

/// A chunk of adjacency entries.
#[derive(Debug, Clone)]
struct AdjacencyChunk {
    /// Destination node IDs.
    destinations: Vec<NodeId>,
    /// Edge IDs (parallel to destinations).
    edge_ids: Vec<EdgeId>,
    /// Capacity of this chunk.
    capacity: usize,
}

impl AdjacencyChunk {
    fn new(capacity: usize) -> Self {
        Self {
            destinations: Vec::with_capacity(capacity),
            edge_ids: Vec::with_capacity(capacity),
            capacity,
        }
    }

    fn len(&self) -> usize {
        self.destinations.len()
    }

    fn is_full(&self) -> bool {
        self.destinations.len() >= self.capacity
    }

    fn push(&mut self, dst: NodeId, edge_id: EdgeId) -> bool {
        if self.is_full() {
            return false;
        }
        self.destinations.push(dst);
        self.edge_ids.push(edge_id);
        true
    }

    fn iter(&self) -> impl Iterator<Item = (NodeId, EdgeId)> + '_ {
        self.destinations
            .iter()
            .copied()
            .zip(self.edge_ids.iter().copied())
    }
}

/// Adjacency list for a single node.
#[derive(Debug)]
struct AdjacencyList {
    /// Compacted chunks of adjacency entries.
    chunks: Vec<AdjacencyChunk>,
    /// Delta buffer for recent insertions.
    delta_inserts: SmallVec<[(NodeId, EdgeId); 8]>,
    /// Set of deleted edge IDs.
    deleted: FxHashSet<EdgeId>,
}

impl AdjacencyList {
    fn new() -> Self {
        Self {
            chunks: Vec::new(),
            delta_inserts: SmallVec::new(),
            deleted: FxHashSet::default(),
        }
    }

    fn add_edge(&mut self, dst: NodeId, edge_id: EdgeId) {
        // Try to add to the last chunk
        if let Some(last) = self.chunks.last_mut() {
            if last.push(dst, edge_id) {
                return;
            }
        }

        // Add to delta buffer
        self.delta_inserts.push((dst, edge_id));
    }

    fn mark_deleted(&mut self, edge_id: EdgeId) {
        self.deleted.insert(edge_id);
    }

    fn compact(&mut self, chunk_capacity: usize) {
        if self.delta_inserts.is_empty() {
            return;
        }

        // Create new chunks from delta buffer
        // Check if last chunk has room, and if so, pop it to continue filling
        let last_has_room = self.chunks.last().is_some_and(|c| !c.is_full());
        let mut current_chunk = if last_has_room {
            // Invariant: is_some_and() returned true, so chunks is non-empty
            self.chunks
                .pop()
                .expect("chunks is non-empty: is_some_and() succeeded on previous line")
        } else {
            AdjacencyChunk::new(chunk_capacity)
        };

        for (dst, edge_id) in self.delta_inserts.drain(..) {
            if !current_chunk.push(dst, edge_id) {
                self.chunks.push(current_chunk);
                current_chunk = AdjacencyChunk::new(chunk_capacity);
                current_chunk.push(dst, edge_id);
            }
        }

        if current_chunk.len() > 0 {
            self.chunks.push(current_chunk);
        }
    }

    fn iter(&self) -> impl Iterator<Item = (NodeId, EdgeId)> + '_ {
        let deleted = &self.deleted;

        self.chunks
            .iter()
            .flat_map(|c| c.iter())
            .chain(self.delta_inserts.iter().copied())
            .filter(move |(_, edge_id)| !deleted.contains(edge_id))
    }

    fn neighbors(&self) -> impl Iterator<Item = NodeId> + '_ {
        self.iter().map(|(dst, _)| dst)
    }

    fn degree(&self) -> usize {
        self.iter().count()
    }
}

/// Chunked adjacency lists with delta buffers.
///
/// This is the primary structure for storing edge connectivity.
/// It supports efficient insertion, deletion (via tombstones),
/// and sequential scanning.
pub struct ChunkedAdjacency {
    /// Adjacency lists indexed by source node.
    lists: RwLock<FxHashMap<NodeId, AdjacencyList>>,
    /// Chunk capacity for new chunks.
    chunk_capacity: usize,
    /// Total number of edges (including deleted).
    edge_count: AtomicUsize,
    /// Number of deleted edges.
    deleted_count: AtomicUsize,
}

impl ChunkedAdjacency {
    /// Creates a new chunked adjacency structure.
    #[must_use]
    pub fn new() -> Self {
        Self::with_chunk_capacity(DEFAULT_CHUNK_CAPACITY)
    }

    /// Creates a new chunked adjacency with custom chunk capacity.
    #[must_use]
    pub fn with_chunk_capacity(capacity: usize) -> Self {
        Self {
            lists: RwLock::new(FxHashMap::default()),
            chunk_capacity: capacity,
            edge_count: AtomicUsize::new(0),
            deleted_count: AtomicUsize::new(0),
        }
    }

    /// Adds an edge from src to dst.
    pub fn add_edge(&self, src: NodeId, dst: NodeId, edge_id: EdgeId) {
        let mut lists = self.lists.write();
        lists
            .entry(src)
            .or_insert_with(AdjacencyList::new)
            .add_edge(dst, edge_id);
        self.edge_count.fetch_add(1, Ordering::Relaxed);
    }

    /// Marks an edge as deleted.
    pub fn mark_deleted(&self, src: NodeId, edge_id: EdgeId) {
        let mut lists = self.lists.write();
        if let Some(list) = lists.get_mut(&src) {
            list.mark_deleted(edge_id);
            self.deleted_count.fetch_add(1, Ordering::Relaxed);
        }
    }

    /// Returns an iterator over neighbors of a node.
    pub fn neighbors(&self, src: NodeId) -> impl Iterator<Item = NodeId> {
        let lists = self.lists.read();
        let neighbors: Vec<NodeId> = lists
            .get(&src)
            .map(|list| list.neighbors().collect())
            .unwrap_or_default();
        neighbors.into_iter()
    }

    /// Returns an iterator over (neighbor, edge_id) pairs.
    pub fn edges_from(&self, src: NodeId) -> impl Iterator<Item = (NodeId, EdgeId)> {
        let lists = self.lists.read();
        let edges: Vec<(NodeId, EdgeId)> = lists
            .get(&src)
            .map(|list| list.iter().collect())
            .unwrap_or_default();
        edges.into_iter()
    }

    /// Returns the out-degree of a node.
    pub fn out_degree(&self, src: NodeId) -> usize {
        let lists = self.lists.read();
        lists.get(&src).map_or(0, |list| list.degree())
    }

    /// Compacts all adjacency lists.
    pub fn compact(&self) {
        let mut lists = self.lists.write();
        for list in lists.values_mut() {
            list.compact(self.chunk_capacity);
        }
    }

    /// Compacts delta buffers that exceed the threshold.
    pub fn compact_if_needed(&self) {
        let mut lists = self.lists.write();
        for list in lists.values_mut() {
            if list.delta_inserts.len() >= DELTA_COMPACTION_THRESHOLD {
                list.compact(self.chunk_capacity);
            }
        }
    }

    /// Returns the total number of edges (including deleted).
    pub fn total_edge_count(&self) -> usize {
        self.edge_count.load(Ordering::Relaxed)
    }

    /// Returns the number of active (non-deleted) edges.
    pub fn active_edge_count(&self) -> usize {
        self.edge_count.load(Ordering::Relaxed) - self.deleted_count.load(Ordering::Relaxed)
    }

    /// Returns the number of nodes with adjacency lists.
    pub fn node_count(&self) -> usize {
        self.lists.read().len()
    }

    /// Clears all adjacency lists.
    pub fn clear(&self) {
        let mut lists = self.lists.write();
        lists.clear();
        self.edge_count.store(0, Ordering::Relaxed);
        self.deleted_count.store(0, Ordering::Relaxed);
    }
}

impl Default for ChunkedAdjacency {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_adjacency() {
        let adj = ChunkedAdjacency::new();

        adj.add_edge(NodeId::new(0), NodeId::new(1), EdgeId::new(0));
        adj.add_edge(NodeId::new(0), NodeId::new(2), EdgeId::new(1));
        adj.add_edge(NodeId::new(0), NodeId::new(3), EdgeId::new(2));

        let neighbors: Vec<_> = adj.neighbors(NodeId::new(0)).collect();
        assert_eq!(neighbors.len(), 3);
        assert!(neighbors.contains(&NodeId::new(1)));
        assert!(neighbors.contains(&NodeId::new(2)));
        assert!(neighbors.contains(&NodeId::new(3)));
    }

    #[test]
    fn test_out_degree() {
        let adj = ChunkedAdjacency::new();

        adj.add_edge(NodeId::new(0), NodeId::new(1), EdgeId::new(0));
        adj.add_edge(NodeId::new(0), NodeId::new(2), EdgeId::new(1));

        assert_eq!(adj.out_degree(NodeId::new(0)), 2);
        assert_eq!(adj.out_degree(NodeId::new(1)), 0);
    }

    #[test]
    fn test_mark_deleted() {
        let adj = ChunkedAdjacency::new();

        adj.add_edge(NodeId::new(0), NodeId::new(1), EdgeId::new(0));
        adj.add_edge(NodeId::new(0), NodeId::new(2), EdgeId::new(1));

        adj.mark_deleted(NodeId::new(0), EdgeId::new(0));

        let neighbors: Vec<_> = adj.neighbors(NodeId::new(0)).collect();
        assert_eq!(neighbors.len(), 1);
        assert!(neighbors.contains(&NodeId::new(2)));
    }

    #[test]
    fn test_edges_from() {
        let adj = ChunkedAdjacency::new();

        adj.add_edge(NodeId::new(0), NodeId::new(1), EdgeId::new(10));
        adj.add_edge(NodeId::new(0), NodeId::new(2), EdgeId::new(20));

        let edges: Vec<_> = adj.edges_from(NodeId::new(0)).collect();
        assert_eq!(edges.len(), 2);
        assert!(edges.contains(&(NodeId::new(1), EdgeId::new(10))));
        assert!(edges.contains(&(NodeId::new(2), EdgeId::new(20))));
    }

    #[test]
    fn test_compaction() {
        let adj = ChunkedAdjacency::with_chunk_capacity(4);

        // Add more edges than chunk capacity
        for i in 0..10 {
            adj.add_edge(NodeId::new(0), NodeId::new(i + 1), EdgeId::new(i));
        }

        adj.compact();

        // All edges should still be accessible
        let neighbors: Vec<_> = adj.neighbors(NodeId::new(0)).collect();
        assert_eq!(neighbors.len(), 10);
    }

    #[test]
    fn test_edge_counts() {
        let adj = ChunkedAdjacency::new();

        adj.add_edge(NodeId::new(0), NodeId::new(1), EdgeId::new(0));
        adj.add_edge(NodeId::new(0), NodeId::new(2), EdgeId::new(1));
        adj.add_edge(NodeId::new(1), NodeId::new(2), EdgeId::new(2));

        assert_eq!(adj.total_edge_count(), 3);
        assert_eq!(adj.active_edge_count(), 3);

        adj.mark_deleted(NodeId::new(0), EdgeId::new(0));

        assert_eq!(adj.total_edge_count(), 3);
        assert_eq!(adj.active_edge_count(), 2);
    }

    #[test]
    fn test_clear() {
        let adj = ChunkedAdjacency::new();

        adj.add_edge(NodeId::new(0), NodeId::new(1), EdgeId::new(0));
        adj.add_edge(NodeId::new(0), NodeId::new(2), EdgeId::new(1));

        adj.clear();

        assert_eq!(adj.total_edge_count(), 0);
        assert_eq!(adj.node_count(), 0);
    }
}
