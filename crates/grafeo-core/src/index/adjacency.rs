//! Chunked adjacency lists - the core data structure for graph traversal.
//!
//! Every graph database needs fast neighbor lookups. This implementation uses
//! chunked storage with delta buffers, giving you:
//!
//! - **O(1) amortized inserts** - new edges go into a delta buffer
//! - **Cache-friendly scans** - chunks are sized for L1/L2 cache
//! - **Soft deletes** - deletions don't require recompaction
//! - **Concurrent reads** - RwLock allows many simultaneous traversals
//! - **Compression** - cold chunks can be compressed using DeltaBitPacked

use crate::storage::{BitPackedInts, DeltaBitPacked};
use grafeo_common::types::{EdgeId, NodeId};
use grafeo_common::utils::hash::{FxHashMap, FxHashSet};
use parking_lot::RwLock;
use smallvec::SmallVec;
use std::sync::atomic::{AtomicUsize, Ordering};

/// Default chunk capacity (number of edges per chunk).
const DEFAULT_CHUNK_CAPACITY: usize = 64;

/// Threshold for delta buffer compaction.
///
/// Lower values reduce memory overhead and iteration cost for delta buffers,
/// but increase compaction frequency. 64 provides a good balance for typical workloads.
const DELTA_COMPACTION_THRESHOLD: usize = 64;

/// Threshold for cold chunk compression.
///
/// When the number of hot chunks exceeds this threshold, the oldest hot chunks
/// are compressed and moved to cold storage. This balances memory usage with
/// the cost of compression/decompression.
const COLD_COMPRESSION_THRESHOLD: usize = 4;

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

    /// Compresses this chunk into a `CompressedAdjacencyChunk`.
    ///
    /// The entries are sorted by destination node ID for better delta compression.
    /// Use this for cold chunks that won't be modified.
    #[allow(dead_code)]
    fn compress(&self) -> CompressedAdjacencyChunk {
        // Sort entries by destination for better delta compression
        let mut entries: Vec<_> = self
            .destinations
            .iter()
            .copied()
            .zip(self.edge_ids.iter().copied())
            .collect();
        entries.sort_by_key(|(dst, _)| dst.as_u64());

        // Extract sorted destinations and corresponding edge IDs
        let sorted_dsts: Vec<u64> = entries.iter().map(|(d, _)| d.as_u64()).collect();
        let sorted_edges: Vec<u64> = entries.iter().map(|(_, e)| e.as_u64()).collect();

        CompressedAdjacencyChunk {
            destinations: DeltaBitPacked::encode(&sorted_dsts),
            edge_ids: BitPackedInts::pack(&sorted_edges),
            count: entries.len(),
        }
    }
}

/// A compressed chunk of adjacency entries.
///
/// Uses DeltaBitPacked for destination node IDs (sorted for good compression)
/// and BitPackedInts for edge IDs. Typical compression ratio is 5-10x for
/// dense adjacency lists.
#[derive(Debug, Clone)]
struct CompressedAdjacencyChunk {
    /// Delta + bit-packed destination node IDs (sorted).
    destinations: DeltaBitPacked,
    /// Bit-packed edge IDs.
    edge_ids: BitPackedInts,
    /// Number of entries.
    count: usize,
}

impl CompressedAdjacencyChunk {
    /// Returns the number of entries in this chunk.
    fn len(&self) -> usize {
        self.count
    }

    /// Returns true if this chunk is empty.
    #[allow(dead_code)]
    fn is_empty(&self) -> bool {
        self.count == 0
    }

    /// Decompresses and iterates over all entries.
    fn iter(&self) -> impl Iterator<Item = (NodeId, EdgeId)> + '_ {
        let dsts = self.destinations.decode();
        let edges = self.edge_ids.unpack();

        dsts.into_iter()
            .zip(edges)
            .map(|(d, e)| (NodeId::new(d), EdgeId::new(e)))
    }

    /// Returns the approximate memory size in bytes.
    #[allow(dead_code)]
    fn memory_size(&self) -> usize {
        // DeltaBitPacked: 8 bytes base + packed deltas
        // BitPackedInts: packed data
        let dest_size = 8 + self.destinations.to_bytes().len();
        let edge_size = self.edge_ids.data().len() * 8;
        dest_size + edge_size
    }

    /// Returns the compression ratio compared to uncompressed storage.
    #[allow(dead_code)]
    fn compression_ratio(&self) -> f64 {
        if self.count == 0 {
            return 1.0;
        }
        let uncompressed = self.count * 16; // 8 bytes each for NodeId and EdgeId
        let compressed = self.memory_size();
        if compressed == 0 {
            return f64::INFINITY;
        }
        uncompressed as f64 / compressed as f64
    }
}

/// Adjacency list for a single node.
///
/// Uses a tiered storage model:
/// - **Hot chunks**: Recent data, uncompressed for fast modification
/// - **Cold chunks**: Older data, compressed for memory efficiency
/// - **Delta buffer**: Very recent insertions, not yet compacted
#[derive(Debug)]
struct AdjacencyList {
    /// Hot chunks (mutable, uncompressed) - for recent data.
    hot_chunks: Vec<AdjacencyChunk>,
    /// Cold chunks (immutable, compressed) - for older data.
    cold_chunks: Vec<CompressedAdjacencyChunk>,
    /// Delta buffer for recent insertions.
    /// Uses SmallVec with 16 inline entries for cache-friendly access.
    /// Most nodes have <16 recent insertions before compaction, so this
    /// avoids heap allocation in the common case.
    delta_inserts: SmallVec<[(NodeId, EdgeId); 16]>,
    /// Set of deleted edge IDs.
    deleted: FxHashSet<EdgeId>,
}

impl AdjacencyList {
    fn new() -> Self {
        Self {
            hot_chunks: Vec::new(),
            cold_chunks: Vec::new(),
            delta_inserts: SmallVec::new(),
            deleted: FxHashSet::default(),
        }
    }

    fn add_edge(&mut self, dst: NodeId, edge_id: EdgeId) {
        // Try to add to the last hot chunk
        if let Some(last) = self.hot_chunks.last_mut() {
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
        // Check if last hot chunk has room, and if so, pop it to continue filling
        let last_has_room = self.hot_chunks.last().is_some_and(|c| !c.is_full());
        let mut current_chunk = if last_has_room {
            // Invariant: is_some_and() returned true, so hot_chunks is non-empty
            self.hot_chunks
                .pop()
                .expect("hot_chunks is non-empty: is_some_and() succeeded on previous line")
        } else {
            AdjacencyChunk::new(chunk_capacity)
        };

        for (dst, edge_id) in self.delta_inserts.drain(..) {
            if !current_chunk.push(dst, edge_id) {
                self.hot_chunks.push(current_chunk);
                current_chunk = AdjacencyChunk::new(chunk_capacity);
                current_chunk.push(dst, edge_id);
            }
        }

        if current_chunk.len() > 0 {
            self.hot_chunks.push(current_chunk);
        }

        // Check if we should compress some hot chunks to cold
        self.maybe_compress_to_cold();
    }

    /// Compresses oldest hot chunks to cold storage if threshold exceeded.
    fn maybe_compress_to_cold(&mut self) {
        // Keep at least COLD_COMPRESSION_THRESHOLD hot chunks for write performance
        while self.hot_chunks.len() > COLD_COMPRESSION_THRESHOLD {
            // Remove the oldest (first) hot chunk
            let oldest = self.hot_chunks.remove(0);

            // Skip empty chunks
            if oldest.len() == 0 {
                continue;
            }

            // Compress and add to cold storage
            let compressed = oldest.compress();
            self.cold_chunks.push(compressed);
        }
    }

    /// Forces all hot chunks to be compressed to cold storage.
    ///
    /// Useful when memory pressure is high or the node is rarely accessed.
    #[allow(dead_code)]
    fn freeze_all(&mut self) {
        for chunk in self.hot_chunks.drain(..) {
            if chunk.len() > 0 {
                self.cold_chunks.push(chunk.compress());
            }
        }
    }

    fn iter(&self) -> impl Iterator<Item = (NodeId, EdgeId)> + '_ {
        let deleted = &self.deleted;

        // Iterate cold chunks first (oldest data)
        let cold_iter = self.cold_chunks.iter().flat_map(|c| c.iter());

        // Then hot chunks
        let hot_iter = self.hot_chunks.iter().flat_map(|c| c.iter());

        // Finally delta buffer (newest data)
        let delta_iter = self.delta_inserts.iter().copied();

        cold_iter
            .chain(hot_iter)
            .chain(delta_iter)
            .filter(move |(_, edge_id)| !deleted.contains(edge_id))
    }

    fn neighbors(&self) -> impl Iterator<Item = NodeId> + '_ {
        self.iter().map(|(dst, _)| dst)
    }

    fn degree(&self) -> usize {
        self.iter().count()
    }

    /// Returns the number of entries in hot storage.
    #[allow(dead_code)]
    fn hot_count(&self) -> usize {
        self.hot_chunks.iter().map(|c| c.len()).sum::<usize>() + self.delta_inserts.len()
    }

    /// Returns the number of entries in cold storage.
    #[allow(dead_code)]
    fn cold_count(&self) -> usize {
        self.cold_chunks.iter().map(|c| c.len()).sum()
    }

    /// Returns the approximate memory size in bytes.
    #[allow(dead_code)]
    fn memory_size(&self) -> usize {
        // Hot chunks: full uncompressed size
        let hot_size = self.hot_chunks.iter().map(|c| c.len() * 16).sum::<usize>();

        // Cold chunks: compressed size
        let cold_size = self
            .cold_chunks
            .iter()
            .map(|c| c.memory_size())
            .sum::<usize>();

        // Delta buffer
        let delta_size = self.delta_inserts.len() * 16;

        // Deleted set (rough estimate)
        let deleted_size = self.deleted.len() * 16;

        hot_size + cold_size + delta_size + deleted_size
    }

    /// Returns the compression ratio for cold storage.
    #[allow(dead_code)]
    fn cold_compression_ratio(&self) -> f64 {
        let total_cold_entries: usize = self.cold_chunks.iter().map(|c| c.len()).sum();
        if total_cold_entries == 0 {
            return 1.0;
        }

        let uncompressed_size = total_cold_entries * 16;
        let compressed_size: usize = self.cold_chunks.iter().map(|c| c.memory_size()).sum();

        if compressed_size == 0 {
            return f64::INFINITY;
        }

        uncompressed_size as f64 / compressed_size as f64
    }
}

/// The main structure for traversing graph edges.
///
/// Given a node, this tells you all its neighbors and the edges connecting them.
/// Internally uses chunked storage (64 edges per chunk) with a delta buffer for
/// recent inserts. Deletions are soft (tombstones) until compaction.
///
/// # Example
///
/// ```
/// use grafeo_core::index::ChunkedAdjacency;
/// use grafeo_common::types::{NodeId, EdgeId};
///
/// let adj = ChunkedAdjacency::new();
///
/// // Build a star graph: node 0 connects to nodes 1, 2, 3
/// adj.add_edge(NodeId::new(0), NodeId::new(1), EdgeId::new(100));
/// adj.add_edge(NodeId::new(0), NodeId::new(2), EdgeId::new(101));
/// adj.add_edge(NodeId::new(0), NodeId::new(3), EdgeId::new(102));
///
/// // Fast neighbor lookup
/// let neighbors = adj.neighbors(NodeId::new(0));
/// assert_eq!(neighbors.len(), 3);
/// ```
pub struct ChunkedAdjacency {
    /// Adjacency lists indexed by source node.
    /// Lock order: 10 (nested, acquired via LpgStore::forward_adj/backward_adj)
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

    /// Returns all neighbors of a node.
    ///
    /// Note: This allocates a Vec to collect neighbors while the internal lock
    /// is held, then returns the Vec. For traversal performance, consider using
    /// `edges_from` if you also need edge IDs, to avoid multiple lookups.
    #[must_use]
    pub fn neighbors(&self, src: NodeId) -> Vec<NodeId> {
        let lists = self.lists.read();
        lists
            .get(&src)
            .map(|list| list.neighbors().collect())
            .unwrap_or_default()
    }

    /// Returns all (neighbor, edge_id) pairs for outgoing edges from a node.
    ///
    /// Note: This allocates a Vec to collect edges while the internal lock
    /// is held, then returns the Vec. This is intentional to avoid holding
    /// the lock across iteration.
    #[must_use]
    pub fn edges_from(&self, src: NodeId) -> Vec<(NodeId, EdgeId)> {
        let lists = self.lists.read();
        lists
            .get(&src)
            .map(|list| list.iter().collect())
            .unwrap_or_default()
    }

    /// Returns the out-degree of a node (number of outgoing edges).
    ///
    /// For forward adjacency, this counts edges where `src` is the source.
    pub fn out_degree(&self, src: NodeId) -> usize {
        let lists = self.lists.read();
        lists.get(&src).map_or(0, |list| list.degree())
    }

    /// Returns the in-degree of a node (number of incoming edges).
    ///
    /// This is semantically equivalent to `out_degree` but named differently
    /// for use with backward adjacency where edges are stored in reverse.
    /// When called on `backward_adj`, this returns the count of edges
    /// where `node` is the destination.
    pub fn in_degree(&self, node: NodeId) -> usize {
        let lists = self.lists.read();
        lists.get(&node).map_or(0, |list| list.degree())
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

    /// Returns memory statistics for this adjacency structure.
    #[must_use]
    pub fn memory_stats(&self) -> AdjacencyMemoryStats {
        let lists = self.lists.read();

        let mut hot_entries = 0usize;
        let mut cold_entries = 0usize;
        let mut hot_bytes = 0usize;
        let mut cold_bytes = 0usize;

        for list in lists.values() {
            hot_entries += list.hot_count();
            cold_entries += list.cold_count();

            // Hot: uncompressed (16 bytes per entry)
            hot_bytes += list.hot_count() * 16;

            // Cold: compressed size from memory_size()
            for cold_chunk in &list.cold_chunks {
                cold_bytes += cold_chunk.memory_size();
            }
        }

        AdjacencyMemoryStats {
            hot_entries,
            cold_entries,
            hot_bytes,
            cold_bytes,
            node_count: lists.len(),
        }
    }

    /// Forces all hot chunks to be compressed for all adjacency lists.
    ///
    /// This is useful when memory pressure is high or during shutdown.
    pub fn freeze_all(&self) {
        let mut lists = self.lists.write();
        for list in lists.values_mut() {
            list.freeze_all();
        }
    }
}

/// Memory statistics for the adjacency structure.
#[derive(Debug, Clone)]
pub struct AdjacencyMemoryStats {
    /// Number of entries in hot (uncompressed) storage.
    pub hot_entries: usize,
    /// Number of entries in cold (compressed) storage.
    pub cold_entries: usize,
    /// Bytes used by hot storage.
    pub hot_bytes: usize,
    /// Bytes used by cold storage.
    pub cold_bytes: usize,
    /// Number of nodes with adjacency lists.
    pub node_count: usize,
}

impl AdjacencyMemoryStats {
    /// Returns the total number of entries.
    #[must_use]
    pub fn total_entries(&self) -> usize {
        self.hot_entries + self.cold_entries
    }

    /// Returns the total memory used in bytes.
    #[must_use]
    pub fn total_bytes(&self) -> usize {
        self.hot_bytes + self.cold_bytes
    }

    /// Returns the compression ratio for cold storage.
    ///
    /// Values > 1.0 indicate actual compression.
    #[must_use]
    pub fn cold_compression_ratio(&self) -> f64 {
        if self.cold_entries == 0 || self.cold_bytes == 0 {
            return 1.0;
        }
        let uncompressed = self.cold_entries * 16;
        uncompressed as f64 / self.cold_bytes as f64
    }

    /// Returns the overall compression ratio.
    #[must_use]
    pub fn overall_compression_ratio(&self) -> f64 {
        let total_entries = self.total_entries();
        if total_entries == 0 || self.total_bytes() == 0 {
            return 1.0;
        }
        let uncompressed = total_entries * 16;
        uncompressed as f64 / self.total_bytes() as f64
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

        let neighbors = adj.neighbors(NodeId::new(0));
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

        let neighbors = adj.neighbors(NodeId::new(0));
        assert_eq!(neighbors.len(), 1);
        assert!(neighbors.contains(&NodeId::new(2)));
    }

    #[test]
    fn test_edges_from() {
        let adj = ChunkedAdjacency::new();

        adj.add_edge(NodeId::new(0), NodeId::new(1), EdgeId::new(10));
        adj.add_edge(NodeId::new(0), NodeId::new(2), EdgeId::new(20));

        let edges = adj.edges_from(NodeId::new(0));
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
        let neighbors = adj.neighbors(NodeId::new(0));
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

    #[test]
    fn test_chunk_compression() {
        // Create a chunk with some edges
        let mut chunk = AdjacencyChunk::new(64);

        // Add edges with various destination IDs
        for i in 0..20 {
            chunk.push(NodeId::new(100 + i * 5), EdgeId::new(1000 + i));
        }

        // Compress the chunk
        let compressed = chunk.compress();

        // Verify all data is preserved
        assert_eq!(compressed.len(), 20);

        // Decompress and verify
        let entries: Vec<_> = compressed.iter().collect();
        assert_eq!(entries.len(), 20);

        // After compression, entries are sorted by destination
        // Verify destinations are sorted
        for window in entries.windows(2) {
            assert!(window[0].0.as_u64() <= window[1].0.as_u64());
        }

        // Verify all original destinations are present
        let original_dsts: std::collections::HashSet<_> =
            (0..20).map(|i| NodeId::new(100 + i * 5)).collect();
        let compressed_dsts: std::collections::HashSet<_> =
            entries.iter().map(|(d, _)| *d).collect();
        assert_eq!(original_dsts, compressed_dsts);

        // Check compression ratio (should be > 1.0 for sorted data)
        let ratio = compressed.compression_ratio();
        assert!(
            ratio > 1.0,
            "Expected compression ratio > 1.0, got {}",
            ratio
        );
    }

    #[test]
    fn test_empty_chunk_compression() {
        let chunk = AdjacencyChunk::new(64);
        let compressed = chunk.compress();

        assert_eq!(compressed.len(), 0);
        assert!(compressed.is_empty());
        assert_eq!(compressed.iter().count(), 0);
    }

    #[test]
    fn test_hot_to_cold_migration() {
        // Use small chunk capacity to trigger cold compression faster
        let adj = ChunkedAdjacency::with_chunk_capacity(8);

        // Add many edges to force multiple chunks and cold compression
        // With chunk_capacity=8 and COLD_COMPRESSION_THRESHOLD=4,
        // we need more than 4 * 8 = 32 edges to trigger cold compression
        for i in 0..100 {
            adj.add_edge(NodeId::new(0), NodeId::new(i + 1), EdgeId::new(i));
        }

        // Force compaction to trigger hot→cold migration
        adj.compact();

        // All edges should still be accessible
        let neighbors = adj.neighbors(NodeId::new(0));
        assert_eq!(neighbors.len(), 100);

        // Check memory stats
        let stats = adj.memory_stats();
        assert_eq!(stats.total_entries(), 100);

        // With 100 edges and threshold of 4 hot chunks (32 edges),
        // we should have some cold entries
        assert!(
            stats.cold_entries > 0,
            "Expected some cold entries, got {}",
            stats.cold_entries
        );
    }

    #[test]
    fn test_memory_stats() {
        let adj = ChunkedAdjacency::with_chunk_capacity(8);

        // Add edges
        for i in 0..20 {
            adj.add_edge(NodeId::new(0), NodeId::new(i + 1), EdgeId::new(i));
        }

        adj.compact();

        let stats = adj.memory_stats();
        assert_eq!(stats.total_entries(), 20);
        assert_eq!(stats.node_count, 1);
        assert!(stats.total_bytes() > 0);
    }

    #[test]
    fn test_freeze_all() {
        let adj = ChunkedAdjacency::with_chunk_capacity(8);

        // Add some edges
        for i in 0..30 {
            adj.add_edge(NodeId::new(0), NodeId::new(i + 1), EdgeId::new(i));
        }

        adj.compact();

        // Get initial stats
        let before = adj.memory_stats();

        // Freeze all hot chunks
        adj.freeze_all();

        // Get stats after freeze
        let after = adj.memory_stats();

        // All data should now be in cold storage
        assert_eq!(after.hot_entries, 0);
        assert_eq!(after.cold_entries, before.total_entries());

        // All edges should still be accessible
        let neighbors = adj.neighbors(NodeId::new(0));
        assert_eq!(neighbors.len(), 30);
    }

    #[test]
    fn test_cold_compression_ratio() {
        let adj = ChunkedAdjacency::with_chunk_capacity(8);

        // Add many edges with sequential IDs (compresses well)
        for i in 0..200 {
            adj.add_edge(NodeId::new(0), NodeId::new(100 + i), EdgeId::new(i));
        }

        adj.compact();
        adj.freeze_all();

        let stats = adj.memory_stats();

        // Sequential data should compress well
        let ratio = stats.cold_compression_ratio();
        assert!(
            ratio > 1.5,
            "Expected cold compression ratio > 1.5, got {}",
            ratio
        );
    }

    #[test]
    fn test_deleted_edges_with_cold_storage() {
        let adj = ChunkedAdjacency::with_chunk_capacity(8);

        // Add edges
        for i in 0..50 {
            adj.add_edge(NodeId::new(0), NodeId::new(i + 1), EdgeId::new(i));
        }

        adj.compact();

        // Delete some edges (some will be in cold storage after compaction)
        for i in (0..50).step_by(2) {
            adj.mark_deleted(NodeId::new(0), EdgeId::new(i));
        }

        // Should have half the edges
        let neighbors = adj.neighbors(NodeId::new(0));
        assert_eq!(neighbors.len(), 25);

        // Verify only odd-numbered destinations remain
        for neighbor in neighbors {
            assert!(neighbor.as_u64() % 2 == 0); // Original IDs were i+1, so even means odd i
        }
    }

    #[test]
    fn test_adjacency_list_memory_size() {
        let mut list = AdjacencyList::new();

        // Add edges
        for i in 0..50 {
            list.add_edge(NodeId::new(i + 1), EdgeId::new(i));
        }

        // Compact with small chunk capacity to get multiple chunks
        list.compact(8);

        let size = list.memory_size();
        assert!(size > 0);

        // Size should be roughly proportional to entry count
        // Each entry is 16 bytes uncompressed
        assert!(size <= 50 * 16 + 200); // Allow some overhead
    }

    #[test]
    fn test_cold_iteration_order() {
        let adj = ChunkedAdjacency::with_chunk_capacity(8);

        // Add edges in order
        for i in 0..50 {
            adj.add_edge(NodeId::new(0), NodeId::new(i + 1), EdgeId::new(i));
        }

        adj.compact();

        // Collect all edges
        let edges = adj.edges_from(NodeId::new(0));

        // All edges should be present
        assert_eq!(edges.len(), 50);

        // Verify edge IDs are present (may be reordered due to compression sorting)
        let edge_ids: std::collections::HashSet<_> = edges.iter().map(|(_, e)| *e).collect();
        for i in 0..50 {
            assert!(edge_ids.contains(&EdgeId::new(i)));
        }
    }

    #[test]
    fn test_in_degree() {
        let adj = ChunkedAdjacency::new();

        // Simulate backward adjacency: edges stored as (dst, src)
        // Edge 1->2: backward stores (2, 1)
        // Edge 3->2: backward stores (2, 3)
        adj.add_edge(NodeId::new(2), NodeId::new(1), EdgeId::new(0)); // 1->2 in backward
        adj.add_edge(NodeId::new(2), NodeId::new(3), EdgeId::new(1)); // 3->2 in backward

        // In-degree of node 2 is 2 (two edges point to it)
        assert_eq!(adj.in_degree(NodeId::new(2)), 2);

        // Node 1 has no incoming edges
        assert_eq!(adj.in_degree(NodeId::new(1)), 0);
    }

    #[test]
    fn test_bidirectional_edges() {
        let forward = ChunkedAdjacency::new();
        let backward = ChunkedAdjacency::new();

        // Add edge: 1 -> 2
        let edge_id = EdgeId::new(100);
        forward.add_edge(NodeId::new(1), NodeId::new(2), edge_id);
        backward.add_edge(NodeId::new(2), NodeId::new(1), edge_id); // Reverse for backward!

        // Forward: edges from node 1 → returns (dst=2, edge_id)
        let forward_edges = forward.edges_from(NodeId::new(1));
        assert_eq!(forward_edges.len(), 1);
        assert_eq!(forward_edges[0], (NodeId::new(2), edge_id));

        // Forward: node 2 has no outgoing edges
        assert_eq!(forward.edges_from(NodeId::new(2)).len(), 0);

        // Backward: edges to node 2 → stored as edges_from(2) → returns (src=1, edge_id)
        let backward_edges = backward.edges_from(NodeId::new(2));
        assert_eq!(backward_edges.len(), 1);
        assert_eq!(backward_edges[0], (NodeId::new(1), edge_id));

        // Backward: node 1 has no incoming edges
        assert_eq!(backward.edges_from(NodeId::new(1)).len(), 0);
    }

    #[test]
    fn test_bidirectional_chain() {
        // Test chain: A -> B -> C
        let forward = ChunkedAdjacency::new();
        let backward = ChunkedAdjacency::new();

        let a = NodeId::new(1);
        let b = NodeId::new(2);
        let c = NodeId::new(3);

        // Edge A -> B
        let edge_ab = EdgeId::new(10);
        forward.add_edge(a, b, edge_ab);
        backward.add_edge(b, a, edge_ab);

        // Edge B -> C
        let edge_bc = EdgeId::new(20);
        forward.add_edge(b, c, edge_bc);
        backward.add_edge(c, b, edge_bc);

        // Forward traversal from A: should reach B
        let from_a = forward.edges_from(a);
        assert_eq!(from_a.len(), 1);
        assert_eq!(from_a[0].0, b);

        // Forward traversal from B: should reach C
        let from_b = forward.edges_from(b);
        assert_eq!(from_b.len(), 1);
        assert_eq!(from_b[0].0, c);

        // Backward traversal to C: should find B
        let to_c = backward.edges_from(c);
        assert_eq!(to_c.len(), 1);
        assert_eq!(to_c[0].0, b);

        // Backward traversal to B: should find A
        let to_b = backward.edges_from(b);
        assert_eq!(to_b.len(), 1);
        assert_eq!(to_b[0].0, a);

        // Node A has no incoming edges
        assert_eq!(backward.edges_from(a).len(), 0);

        // Node C has no outgoing edges
        assert_eq!(forward.edges_from(c).len(), 0);
    }
}
