//! HNSW (Hierarchical Navigable Small World) index implementation.
//!
//! HNSW is a graph-based approximate nearest neighbor algorithm that builds
//! a multi-layer navigable small world graph. It provides:
//!
//! - **O(log n)** search complexity (approximate)
//! - **>95%** recall at k=10 with default settings
//!
//! This index is **topology-only**: it stores only the neighbor graph
//! structure, not the vectors themselves. Vectors are read on-the-fly
//! through a [`VectorAccessor`], which typically reads from property
//! storage, the single source of truth, halving memory usage for
//! vector workloads.
//!
//! # Algorithm Overview
//!
//! 1. **Multi-layer graph**: Nodes exist at multiple layers, with decreasing
//!    probability at higher layers (exponential distribution).
//! 2. **Greedy search**: Starting from the entry point at the top layer,
//!    greedily traverse to find the nearest node, then descend.
//! 3. **Beam search**: At the bottom layer, maintain a candidate set of
//!    size `ef` to find the k nearest neighbors.
//!
//! # Example
//!
//! ```
//! use grafeo_core::index::vector::{HnswIndex, HnswConfig, DistanceMetric, VectorAccessor};
//! use grafeo_common::types::NodeId;
//! use std::sync::Arc;
//! use std::collections::HashMap;
//!
//! let config = HnswConfig::new(384, DistanceMetric::Cosine);
//! let index = HnswIndex::new(config);
//!
//! // Build an accessor backed by a HashMap
//! let mut map: HashMap<NodeId, Arc<[f32]>> = HashMap::new();
//! let vec1: Arc<[f32]> = vec![0.1f32; 384].into();
//! map.insert(NodeId::new(1), vec1.clone());
//! let accessor = |id: NodeId| -> Option<Arc<[f32]>> { map.get(&id).cloned() };
//!
//! // Insert vectors
//! index.insert(NodeId::new(1), &vec1, &accessor);
//!
//! // Search for nearest neighbors
//! let query = vec![0.15f32; 384];
//! let results = index.search(&query, 10, &accessor);
//! ```
//!
//! # References
//!
//! - Malkov & Yashunin, "Efficient and robust approximate nearest neighbor
//!   search using Hierarchical Navigable Small World graphs" (2018)

use super::VectorAccessor;
use super::compute_distance;
use super::paged_topology::{MmapTopology, NeighborsIter as MmapNeighborsIter};
use crate::index::vector::HnswConfig;
use grafeo_common::types::NodeId;
use ordered_float::OrderedFloat;
use parking_lot::RwLock;
use rand::{RngExt, SeedableRng};
use std::collections::{BinaryHeap, HashMap, HashSet};
use std::sync::Arc;

/// A neighbor entry in the HNSW graph.
#[derive(Debug, Clone, Copy, PartialEq)]
struct Neighbor {
    id: NodeId,
    distance: f32,
}

impl Eq for Neighbor {}

impl PartialOrd for Neighbor {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Neighbor {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // Min-heap: smaller distance = higher priority
        OrderedFloat(other.distance).cmp(&OrderedFloat(self.distance))
    }
}

/// A candidate for the max-heap during search (furthest first).
#[derive(Debug, Clone, Copy, PartialEq)]
struct FurthestCandidate {
    id: NodeId,
    distance: f32,
}

impl Eq for FurthestCandidate {}

impl PartialOrd for FurthestCandidate {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for FurthestCandidate {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // Max-heap: larger distance = higher priority
        OrderedFloat(self.distance).cmp(&OrderedFloat(other.distance))
    }
}

/// Materializes every node's neighbor lists from an [`MmapTopology`]
/// into the heap representation expected by `snapshot_topology`.
///
/// Used only during checkpoint of an mmap-backed index.
fn snapshot_mmap_topology(topo: &MmapTopology) -> Vec<(NodeId, Vec<Vec<NodeId>>)> {
    let mut out = Vec::with_capacity(topo.len());
    for id in topo.iter_node_ids() {
        let mut layers: Vec<Vec<NodeId>> = Vec::new();
        let mut layer = 0usize;
        while let Some(iter) = topo.neighbors_at(id, layer) {
            layers.push(iter.collect());
            layer += 1;
        }
        out.push((id, layers));
    }
    out
}

/// Node data stored in the HNSW index (topology only, no vector data).
#[derive(Debug, Clone)]
struct HnswNode {
    /// Neighbors at each layer (layer 0 is the bottom).
    /// The node's max layer is `neighbors.len() - 1`.
    neighbors: Vec<Vec<NodeId>>,
}

/// Topology storage backend for [`HnswIndex`].
///
/// Two variants: [`Heap`](Self::Heap) is the build/mutation-friendly
/// representation (HashMap of node neighbor lists); [`Mmap`](Self::Mmap)
/// is a zero-copy view into a [`MmapTopology`] buffer, used when the
/// section was loaded from a `.grafeo` mmap. Reads are unified through
/// [`Self::neighbors_at`]; mutations require [`Self::Heap`].
enum TopologyBackend {
    /// Heap-resident, build-and-mutation friendly.
    Heap(HashMap<NodeId, HnswNode>),
    /// Zero-copy `Bytes`-backed view (Phase 7c). Read-only — mutations
    /// will panic.
    Mmap(MmapTopology),
}

impl TopologyBackend {
    fn new_heap() -> Self {
        Self::Heap(HashMap::new())
    }

    fn with_capacity(capacity: usize) -> Self {
        Self::Heap(HashMap::with_capacity(capacity))
    }

    fn len(&self) -> usize {
        match self {
            Self::Heap(map) => map.len(),
            Self::Mmap(topo) => topo.len(),
        }
    }

    fn is_empty(&self) -> bool {
        match self {
            Self::Heap(map) => map.is_empty(),
            Self::Mmap(topo) => topo.is_empty(),
        }
    }

    fn contains(&self, id: NodeId) -> bool {
        match self {
            Self::Heap(map) => map.contains_key(&id),
            Self::Mmap(topo) => topo.contains(id),
        }
    }

    /// Returns an iterator over the neighbors of `id` at the given
    /// `layer`, or `None` if absent.
    fn neighbors_at(&self, id: NodeId, layer: usize) -> Option<HnswNeighborsIter<'_>> {
        match self {
            Self::Heap(map) => map.get(&id).and_then(|node| {
                if layer < node.neighbors.len() {
                    Some(HnswNeighborsIter::Heap(node.neighbors[layer].iter()))
                } else {
                    None
                }
            }),
            Self::Mmap(topo) => topo.neighbors_at(id, layer).map(HnswNeighborsIter::Mmap),
        }
    }

    /// Borrow the heap representation for mutation, panicking if the
    /// backend is in [`Self::Mmap`] mode.
    fn as_heap_mut(&mut self) -> &mut HashMap<NodeId, HnswNode> {
        match self {
            Self::Heap(map) => map,
            Self::Mmap(_) => {
                panic!("HNSW topology is in mmap mode; cannot mutate. Reload to RAM first.")
            }
        }
    }
}

/// Unified iterator over neighbor IDs from either backend.
///
/// Yields one [`NodeId`] per neighbor; preserves source order.
pub enum HnswNeighborsIter<'a> {
    /// Iterating a heap-stored `Vec<NodeId>`.
    Heap(std::slice::Iter<'a, NodeId>),
    /// Iterating an mmap-backed packed neighbor list.
    Mmap(MmapNeighborsIter<'a>),
}

impl Iterator for HnswNeighborsIter<'_> {
    type Item = NodeId;

    fn next(&mut self) -> Option<NodeId> {
        match self {
            Self::Heap(iter) => iter.next().copied(),
            Self::Mmap(iter) => iter.next(),
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        match self {
            Self::Heap(iter) => iter.size_hint(),
            Self::Mmap(iter) => iter.size_hint(),
        }
    }
}

/// HNSW (Hierarchical Navigable Small World) index.
///
/// Thread-safe approximate nearest neighbor index supporting concurrent
/// reads and exclusive writes. This index is topology-only: vectors are
/// read through a [`VectorAccessor`] rather than stored internally.
pub struct HnswIndex {
    /// Index configuration.
    config: HnswConfig,
    /// Node storage. May be a heap HashMap (build/mutate path) or a
    /// zero-copy [`MmapTopology`] view (post-Phase-7c).
    nodes: RwLock<TopologyBackend>,
    /// Entry point for search (node at the highest layer).
    entry_point: RwLock<Option<NodeId>>,
    /// Current maximum layer in the index.
    max_level: RwLock<usize>,
    /// Random number generator for level selection.
    rng: RwLock<rand::rngs::StdRng>,
}

impl HnswIndex {
    /// Creates a new empty HNSW index with the given configuration.
    #[must_use]
    pub fn new(config: HnswConfig) -> Self {
        Self {
            config,
            nodes: RwLock::new(TopologyBackend::new_heap()),
            entry_point: RwLock::new(None),
            max_level: RwLock::new(0),
            rng: RwLock::new(rand::rngs::StdRng::from_rng(&mut rand::rng())),
        }
    }

    /// Creates a new HNSW index with pre-allocated capacity.
    ///
    /// Use this when you know the approximate number of vectors upfront
    /// to avoid HashMap rehashing during bulk insertion.
    #[must_use]
    pub fn with_capacity(config: HnswConfig, capacity: usize) -> Self {
        Self {
            config,
            nodes: RwLock::new(TopologyBackend::with_capacity(capacity)),
            entry_point: RwLock::new(None),
            max_level: RwLock::new(0),
            rng: RwLock::new(rand::rngs::StdRng::from_rng(&mut rand::rng())),
        }
    }

    /// Creates a new HNSW index with a fixed seed for reproducible results.
    #[must_use]
    pub fn with_seed(config: HnswConfig, seed: u64) -> Self {
        Self {
            config,
            nodes: RwLock::new(TopologyBackend::new_heap()),
            entry_point: RwLock::new(None),
            max_level: RwLock::new(0),
            rng: RwLock::new(rand::rngs::StdRng::seed_from_u64(seed)),
        }
    }

    /// Returns the index configuration.
    #[must_use]
    pub fn config(&self) -> &HnswConfig {
        &self.config
    }

    /// Returns the number of vectors in the index.
    #[must_use]
    pub fn len(&self) -> usize {
        self.nodes.read().len()
    }

    /// Returns true if the index is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.nodes.read().is_empty()
    }

    /// Snapshot the topology for serialization.
    ///
    /// Returns (entry_point, max_level, node_neighbors) where node_neighbors
    /// is a vec of (NodeId, neighbor_layers).
    ///
    /// Works on both heap and mmap backends; the mmap path materializes
    /// neighbor `Vec`s on the fly (used during checkpoint when the
    /// in-memory index is mmap-backed but needs to be re-serialized).
    #[must_use]
    pub fn snapshot_topology(&self) -> (Option<NodeId>, usize, Vec<(NodeId, Vec<Vec<NodeId>>)>) {
        let nodes = self.nodes.read();
        let entry_point = *self.entry_point.read();
        let max_level = *self.max_level.read();

        let mut node_data: Vec<(NodeId, Vec<Vec<NodeId>>)> = match &*nodes {
            TopologyBackend::Heap(map) => map
                .iter()
                .map(|(id, node)| (*id, node.neighbors.clone()))
                .collect(),
            TopologyBackend::Mmap(topo) => {
                // Read-out path used during checkpoint of an mmap-backed
                // index. Iterate every node by binary-searching the
                // page index. Materializes neighbor Vecs.
                snapshot_mmap_topology(topo)
            }
        };
        node_data.sort_by_key(|(id, _)| *id);

        (entry_point, max_level, node_data)
    }

    /// Restore topology from a snapshot. Replaces all current data.
    ///
    /// Always switches the backend to the heap representation; subsequent
    /// mutations work without needing a reload. Equivalent to constructing
    /// a fresh index and calling `insert` for each node, but skips the
    /// graph-build cost.
    pub fn restore_topology(
        &self,
        entry_point: Option<NodeId>,
        max_level: usize,
        node_data: Vec<(NodeId, Vec<Vec<NodeId>>)>,
    ) {
        let mut backend = self.nodes.write();
        let mut fresh: HashMap<NodeId, HnswNode> = HashMap::with_capacity(node_data.len());
        for (id, neighbors) in node_data {
            fresh.insert(id, HnswNode { neighbors });
        }
        *backend = TopologyBackend::Heap(fresh);
        *self.entry_point.write() = entry_point;
        *self.max_level.write() = max_level;
    }

    /// Adopt a [`MmapTopology`] as the topology backend (Phase 7c).
    ///
    /// Replaces any existing topology with a zero-copy view of the
    /// given mmap-backed buffer. Reads through the backend will serve
    /// from the [`bytes::Bytes`] without rebuilding a `HashMap`.
    /// Mutating operations ([`Self::insert`], [`Self::remove`]) will
    /// panic until the backend is reloaded into RAM via
    /// [`Self::restore_topology`].
    ///
    /// `entry_point` and `max_level` are taken from the topology header.
    pub fn adopt_mmap_topology(&self, topo: MmapTopology) {
        let entry_point = topo.entry_point();
        let max_level = topo.max_level();
        let mut backend = self.nodes.write();
        *backend = TopologyBackend::Mmap(topo);
        *self.entry_point.write() = entry_point;
        *self.max_level.write() = max_level;
    }

    /// Returns true if the backend is currently mmap-backed.
    #[must_use]
    pub fn is_mmap_backed(&self) -> bool {
        matches!(*self.nodes.read(), TopologyBackend::Mmap(_))
    }

    /// Returns estimated heap memory in bytes for the HNSW topology.
    ///
    /// In mmap mode, returns only the small struct overhead — the
    /// neighbor data lives in the mmap.
    #[must_use]
    pub fn heap_memory_bytes(&self) -> usize {
        let nodes = self.nodes.read();
        match &*nodes {
            TopologyBackend::Heap(map) => {
                let map_overhead = map.capacity()
                    * (std::mem::size_of::<NodeId>() + std::mem::size_of::<HnswNode>() + 1);
                let mut node_bytes = 0usize;
                for node in map.values() {
                    node_bytes += node.neighbors.capacity() * std::mem::size_of::<Vec<NodeId>>();
                    for layer in &node.neighbors {
                        node_bytes += layer.capacity() * std::mem::size_of::<NodeId>();
                    }
                }
                map_overhead + node_bytes
            }
            TopologyBackend::Mmap(_) => std::mem::size_of::<TopologyBackend>(),
        }
    }

    /// Inserts a vector with the given ID into the index.
    ///
    /// The vector is used during insertion to find neighbors and build
    /// the graph topology, but is **not** stored in the index.
    ///
    /// # Panics
    ///
    /// Panics if the vector dimensions don't match the configuration.
    pub fn insert(&self, id: NodeId, vector: &[f32], accessor: &impl VectorAccessor) {
        assert_eq!(
            vector.len(),
            self.config.dimensions,
            "Vector dimensions mismatch: expected {}, got {}",
            self.config.dimensions,
            vector.len()
        );

        // `set_node_property` uses insert for both first writes and updates.
        // Replacing the topology entry directly would leave reciprocal links
        // pointing at a node whose own neighbor lists were reset. Remove the
        // previous entry and reconnect its former neighbors before indexing the
        // replacement vector.
        self.remove_for_replacement(id, accessor);

        let level = self.random_level();

        // Create the new node (topology only)
        let node = HnswNode {
            neighbors: vec![Vec::new(); level + 1],
        };

        let mut nodes = self.nodes.write();
        let mut entry_point = self.entry_point.write();
        let mut max_level = self.max_level.write();

        // Insert path always operates on the heap backend; calling
        // `as_heap_mut` panics if the topology is mmap-backed. Reload
        // to RAM via `restore_topology` first if needed.

        // Capacity check + first-insertion path. Scoped so the mutable
        // borrow ends before the per-layer search loop reborrows `&nodes`.
        {
            let nodes_map = nodes.as_heap_mut();

            if let Some(max) = self.config.max_elements
                && !nodes_map.contains_key(&id)
            {
                let count = nodes_map.len();
                assert!(
                    count < max,
                    "HNSW index is full: max_elements={max}, current={count}"
                );
            }

            // First insertion
            if entry_point.is_none() {
                nodes_map.insert(id, node);
                *entry_point = Some(id);
                *max_level = level;
                return;
            }
        }

        let ep = entry_point.expect("entry_point confirmed Some above");
        let current_max_level = *max_level;

        // Insert the new node so subsequent searches can find it.
        nodes.as_heap_mut().insert(id, node);

        // Search from top to the level above the new node's max layer.
        let mut current_ep = ep;
        for lc in (level + 1..=current_max_level).rev() {
            current_ep = self.search_layer_single(&nodes, accessor, vector, current_ep, lc);
        }

        // For each layer from the new node's max layer down to 0
        for lc in (0..=level.min(current_max_level)).rev() {
            let m_max = if lc == 0 {
                self.config.m_max
            } else {
                self.config.m
            };

            // Find ef_construction nearest neighbors at this layer
            let neighbors = self.search_layer(
                &nodes,
                accessor,
                vector,
                current_ep,
                self.config.ef_construction,
                lc,
            );

            // Select neighbors using diversity-aware heuristic
            let selected = self.select_neighbors_heuristic(accessor, &neighbors, m_max);

            // First pass: link new node + identify who needs pruning.
            // Scope the mutable borrow tightly.
            let mut needs_pruning: Vec<NodeId> = Vec::new();
            {
                let nodes_map = nodes.as_heap_mut();
                if let Some(new_node) = nodes_map.get_mut(&id) {
                    new_node.neighbors[lc].clone_from(&selected);
                }

                for &neighbor_id in &selected {
                    if let Some(neighbor) = nodes_map.get_mut(&neighbor_id)
                        && neighbor.neighbors.len() > lc
                    {
                        neighbor.neighbors[lc].push(id);

                        if neighbor.neighbors[lc].len() > m_max {
                            needs_pruning.push(neighbor_id);
                        }
                    }
                }
            }

            // Second pass: compute distances for pruning (immutable read).
            let mut prune_data: Vec<(NodeId, Vec<(NodeId, f32)>)> = Vec::new();
            {
                let nodes_map = nodes.as_heap_mut();
                for neighbor_id in &needs_pruning {
                    if let Some(neighbor) = nodes_map.get(neighbor_id)
                        && neighbor.neighbors.len() > lc
                    {
                        let Some(base_vec) = accessor.get_vector(*neighbor_id) else {
                            continue;
                        };
                        let distances: Vec<(NodeId, f32)> = neighbor.neighbors[lc]
                            .iter()
                            .map(|&nid| {
                                let dist = accessor
                                    .get_vector(nid)
                                    .map_or(f32::MAX, |v| self.vector_distance(&base_vec, &v));
                                (nid, dist)
                            })
                            .collect();
                        prune_data.push((*neighbor_id, distances));
                    }
                }
            }

            // Third pass: apply pruning (mutable borrow).
            {
                let nodes_map = nodes.as_heap_mut();
                for (neighbor_id, distances) in prune_data {
                    if let Some(neighbor) = nodes_map.get_mut(&neighbor_id)
                        && neighbor.neighbors.len() > lc
                    {
                        Self::prune_neighbors_with_distances(
                            &mut neighbor.neighbors[lc],
                            &distances,
                            m_max,
                        );
                    }
                }
            }

            // Update entry point for next layer
            if !selected.is_empty() {
                current_ep = selected[0];
            }
        }

        // Update global entry point if needed
        if level > current_max_level {
            *entry_point = Some(id);
            *max_level = level;
        }
    }

    /// Remove an existing node before replacing its vector, preserving paths
    /// between the node's former neighbors.
    fn remove_for_replacement(&self, id: NodeId, accessor: &impl VectorAccessor) {
        let mut nodes = self.nodes.write();
        let mut entry_point = self.entry_point.write();
        let nodes_map = nodes.as_heap_mut();

        let Some(removed) = nodes_map.remove(&id) else {
            return;
        };

        for node in nodes_map.values_mut() {
            for neighbors in &mut node.neighbors {
                neighbors.retain(|&neighbor| neighbor != id);
            }
        }

        // Removing a bridge and immediately reinserting it can otherwise leave
        // its former neighborhoods disconnected. Connect those neighborhoods
        // at each shared level, then prune them back to the configured degree.
        for (level, former_neighbors) in removed.neighbors.iter().enumerate() {
            let existing: Vec<NodeId> = former_neighbors
                .iter()
                .copied()
                .filter(|neighbor| {
                    nodes_map
                        .get(neighbor)
                        .is_some_and(|node| node.neighbors.len() > level)
                })
                .collect();

            for &left in &existing {
                if let Some(node) = nodes_map.get_mut(&left) {
                    for &right in &existing {
                        if left != right && !node.neighbors[level].contains(&right) {
                            node.neighbors[level].push(right);
                        }
                    }
                }
            }

            let max_neighbors = if level == 0 {
                self.config.m_max
            } else {
                self.config.m
            };
            let mut prune_data: Vec<(NodeId, Vec<(NodeId, f32)>)> = Vec::new();
            for &neighbor_id in &existing {
                let Some(node) = nodes_map.get(&neighbor_id) else {
                    continue;
                };
                if node.neighbors[level].len() <= max_neighbors {
                    continue;
                }
                let Some(base_vector) = accessor.get_vector(neighbor_id) else {
                    continue;
                };
                let distances: Vec<(NodeId, f32)> = node.neighbors[level]
                    .iter()
                    .map(|&candidate| {
                        let distance = accessor.get_vector(candidate).map_or(f32::MAX, |vector| {
                            self.vector_distance(&base_vector, &vector)
                        });
                        (candidate, distance)
                    })
                    .collect();
                prune_data.push((neighbor_id, distances));
            }

            for (neighbor_id, distances) in prune_data {
                if let Some(node) = nodes_map.get_mut(&neighbor_id) {
                    Self::prune_neighbors_with_distances(
                        &mut node.neighbors[level],
                        &distances,
                        max_neighbors,
                    );
                }
            }
        }

        if *entry_point == Some(id) {
            *entry_point = removed
                .neighbors
                .iter()
                .rev()
                .flatten()
                .find(|neighbor| nodes_map.contains_key(neighbor))
                .copied()
                .or_else(|| nodes_map.keys().next().copied());
        }
    }

    /// Searches for the k nearest neighbors to the query vector.
    ///
    /// Returns a vector of (NodeId, distance) pairs sorted by distance
    /// (closest first).
    ///
    /// # Panics
    ///
    /// Panics if the query vector dimensions don't match the configuration.
    #[must_use]
    pub fn search(
        &self,
        query: &[f32],
        k: usize,
        accessor: &impl VectorAccessor,
    ) -> Vec<(NodeId, f32)> {
        self.search_with_ef(query, k, self.config.ef, accessor)
    }

    /// Searches with a custom ef (beam width) parameter.
    ///
    /// Higher ef values give better recall at the cost of latency.
    ///
    /// # Panics
    ///
    /// Panics if `query.len()` does not match the configured `dimensions`.
    #[must_use]
    pub fn search_with_ef(
        &self,
        query: &[f32],
        k: usize,
        ef: usize,
        accessor: &impl VectorAccessor,
    ) -> Vec<(NodeId, f32)> {
        assert_eq!(
            query.len(),
            self.config.dimensions,
            "Query dimensions mismatch: expected {}, got {}",
            self.config.dimensions,
            query.len()
        );

        let nodes = self.nodes.read();
        let entry_point = self.entry_point.read();
        let max_level = *self.max_level.read();

        if entry_point.is_none() || nodes.is_empty() {
            return Vec::new();
        }

        let ep = entry_point.expect("entry_point confirmed Some above");

        // Greedy search from top layer to layer 1
        let mut current_ep = ep;
        for lc in (1..=max_level).rev() {
            current_ep = self.search_layer_single(&nodes, accessor, query, current_ep, lc);
        }

        // Beam search at layer 0
        let ef_search = ef.max(k);
        let candidates = self.search_layer(&nodes, accessor, query, current_ep, ef_search, 0);

        // Return top k
        candidates
            .into_iter()
            .take(k)
            .map(|n| (n.id, n.distance))
            .collect()
    }

    /// Searches for the k nearest neighbors with an allowlist filter.
    ///
    /// Only nodes in the `allowlist` can appear in results. The HNSW graph
    /// is still fully traversed for connectivity; the filter only restricts
    /// the result set. The search beam width (`ef`) is automatically scaled
    /// based on the allowlist selectivity to maintain recall.
    ///
    /// Returns an empty vector if the allowlist is empty.
    #[must_use]
    pub fn search_with_filter(
        &self,
        query: &[f32],
        k: usize,
        allowlist: &HashSet<NodeId>,
        accessor: &impl VectorAccessor,
    ) -> Vec<(NodeId, f32)> {
        if allowlist.is_empty() {
            return Vec::new();
        }
        // Auto-scale ef based on selectivity ratio
        let total = self.nodes.read().len();
        let selectivity = if total == 0 {
            1.0
        } else {
            (allowlist.len() as f64 / total as f64).max(0.01)
        };
        // reason: ef scaled by selectivity is non-negative and bounded by .min(total)
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        let ef_scaled = ((self.config.ef as f64 / selectivity).ceil() as usize)
            .min(total)
            .max(k);
        self.search_with_ef_and_filter(query, k, ef_scaled, allowlist, accessor)
    }

    /// Searches with a custom ef (beam width) and an allowlist filter.
    ///
    /// Only nodes in the `allowlist` can appear in results. Higher ef values
    /// give better recall at the cost of latency.
    ///
    /// Returns an empty vector if the allowlist is empty.
    ///
    /// # Panics
    ///
    /// Panics if `query.len()` does not match the configured `dimensions`.
    #[must_use]
    pub fn search_with_ef_and_filter(
        &self,
        query: &[f32],
        k: usize,
        ef: usize,
        allowlist: &HashSet<NodeId>,
        accessor: &impl VectorAccessor,
    ) -> Vec<(NodeId, f32)> {
        if allowlist.is_empty() {
            return Vec::new();
        }

        assert_eq!(
            query.len(),
            self.config.dimensions,
            "Query dimensions mismatch: expected {}, got {}",
            self.config.dimensions,
            query.len()
        );

        let nodes = self.nodes.read();
        let entry_point = self.entry_point.read();
        let max_level = *self.max_level.read();

        if entry_point.is_none() || nodes.is_empty() {
            return Vec::new();
        }

        let ep = entry_point.expect("entry_point confirmed Some above");

        // Greedy search from top layer to layer 1
        let mut current_ep = ep;
        for lc in (1..=max_level).rev() {
            current_ep = self.search_layer_single(&nodes, accessor, query, current_ep, lc);
        }

        // Filtered beam search at layer 0
        let ef_search = ef.max(k);
        let candidates = self
            .search_layer_filtered(&nodes, accessor, query, current_ep, ef_search, 0, allowlist);

        // Return top k
        candidates
            .into_iter()
            .take(k)
            .map(|n| (n.id, n.distance))
            .collect()
    }

    /// Removes a vector from the index.
    ///
    /// Returns true if the vector was found and removed.
    pub fn remove(&self, id: NodeId) -> bool {
        let mut nodes = self.nodes.write();
        let mut entry_point = self.entry_point.write();

        let nodes_map = nodes.as_heap_mut();

        if nodes_map.remove(&id).is_none() {
            return false;
        }

        // Remove bidirectional links
        for (_, node) in nodes_map.iter_mut() {
            for neighbors in &mut node.neighbors {
                neighbors.retain(|&n| n != id);
            }
        }

        // Update entry point if needed
        if *entry_point == Some(id) {
            *entry_point = nodes_map.keys().next().copied();
        }

        true
    }

    /// Returns true if the index contains a vector with the given ID.
    #[must_use]
    pub fn contains(&self, id: NodeId) -> bool {
        self.nodes.read().contains(id)
    }

    /// Generates a random level for a new node.
    fn random_level(&self) -> usize {
        let mut rng = self.rng.write();
        let r: f64 = rng.random();
        // reason: HNSW level is non-negative (r in [0,1), -ln(r) >= 0), fits usize
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        let level = (-r.ln() * self.config.ml).floor() as usize;
        level
    }

    /// Single-element greedy search at a layer.
    fn search_layer_single(
        &self,
        nodes: &TopologyBackend,
        accessor: &impl VectorAccessor,
        query: &[f32],
        ep: NodeId,
        layer: usize,
    ) -> NodeId {
        let mut current = ep;
        let mut current_dist = self.node_distance(accessor, query, ep);

        loop {
            let mut changed = false;

            if let Some(neighbors) = nodes.neighbors_at(current, layer) {
                for neighbor in neighbors {
                    let dist = self.node_distance(accessor, query, neighbor);
                    if dist < current_dist {
                        current = neighbor;
                        current_dist = dist;
                        changed = true;
                    }
                }
            }

            if !changed {
                break;
            }
        }

        current
    }

    /// Beam search at a layer, returning ef nearest neighbors.
    fn search_layer(
        &self,
        nodes: &TopologyBackend,
        accessor: &impl VectorAccessor,
        query: &[f32],
        ep: NodeId,
        ef: usize,
        layer: usize,
    ) -> Vec<Neighbor> {
        let ep_dist = self.node_distance(accessor, query, ep);

        // Min-heap of candidates to explore
        let mut candidates: BinaryHeap<Neighbor> = BinaryHeap::new();
        candidates.push(Neighbor {
            id: ep,
            distance: ep_dist,
        });

        // Max-heap of current best (furthest = top)
        let mut results: BinaryHeap<FurthestCandidate> = BinaryHeap::new();
        results.push(FurthestCandidate {
            id: ep,
            distance: ep_dist,
        });

        let mut visited: HashSet<NodeId> =
            HashSet::with_capacity(nodes.len().min(ef.saturating_mul(2)));
        visited.insert(ep);

        while let Some(current) = candidates.pop() {
            // If the closest candidate is further than the furthest result, stop
            if let Some(furthest) = results.peek()
                && current.distance > furthest.distance
                && results.len() >= ef
            {
                break;
            }

            // Explore neighbors
            if let Some(neighbors) = nodes.neighbors_at(current.id, layer) {
                for neighbor in neighbors {
                    if visited.contains(&neighbor) {
                        continue;
                    }
                    visited.insert(neighbor);

                    let dist = self.node_distance(accessor, query, neighbor);

                    // Add to results if closer than furthest, or if we have room
                    let should_add =
                        results.len() < ef || results.peek().map_or(true, |f| dist < f.distance);

                    if should_add {
                        candidates.push(Neighbor {
                            id: neighbor,
                            distance: dist,
                        });
                        results.push(FurthestCandidate {
                            id: neighbor,
                            distance: dist,
                        });

                        // Keep only ef results
                        while results.len() > ef {
                            results.pop();
                        }
                    }
                }
            }
        }

        // Convert to sorted vec
        let mut result_vec: Vec<Neighbor> = results
            .into_iter()
            .map(|fc| Neighbor {
                id: fc.id,
                distance: fc.distance,
            })
            .collect();
        result_vec.sort_by_key(|a| OrderedFloat(a.distance));
        result_vec
    }

    /// Beam search at a layer with an allowlist filter on the result set.
    ///
    /// All nodes are visited for graph traversal (neighbor links followed),
    /// but only nodes in the `allowlist` can enter the result set. This
    /// preserves HNSW connectivity while restricting which nodes are returned.
    #[allow(clippy::too_many_arguments)]
    fn search_layer_filtered(
        &self,
        nodes: &TopologyBackend,
        accessor: &impl VectorAccessor,
        query: &[f32],
        ep: NodeId,
        ef: usize,
        layer: usize,
        allowlist: &HashSet<NodeId>,
    ) -> Vec<Neighbor> {
        let ep_dist = self.node_distance(accessor, query, ep);

        // Min-heap of candidates to explore
        let mut candidates: BinaryHeap<Neighbor> = BinaryHeap::new();
        candidates.push(Neighbor {
            id: ep,
            distance: ep_dist,
        });

        // best_seen tracks ALL visited candidates (for traversal termination)
        let mut best_seen: BinaryHeap<FurthestCandidate> = BinaryHeap::new();
        best_seen.push(FurthestCandidate {
            id: ep,
            distance: ep_dist,
        });

        // results only holds allowlisted nodes
        let mut results: BinaryHeap<FurthestCandidate> = BinaryHeap::new();
        if allowlist.contains(&ep) {
            results.push(FurthestCandidate {
                id: ep,
                distance: ep_dist,
            });
        }

        let mut visited: HashSet<NodeId> =
            HashSet::with_capacity(nodes.len().min(ef.saturating_mul(4)));
        visited.insert(ep);

        while let Some(current) = candidates.pop() {
            // Terminate when best candidate is worse than worst in best_seen
            if let Some(furthest) = best_seen.peek()
                && current.distance > furthest.distance
                && best_seen.len() >= ef
            {
                break;
            }

            // Explore neighbors
            if let Some(neighbors) = nodes.neighbors_at(current.id, layer) {
                for neighbor in neighbors {
                    if visited.contains(&neighbor) {
                        continue;
                    }
                    visited.insert(neighbor);

                    let dist = self.node_distance(accessor, query, neighbor);

                    // Update best_seen for traversal guidance
                    let should_explore = best_seen.len() < ef
                        || best_seen.peek().map_or(true, |f| dist < f.distance);

                    if should_explore {
                        candidates.push(Neighbor {
                            id: neighbor,
                            distance: dist,
                        });
                        best_seen.push(FurthestCandidate {
                            id: neighbor,
                            distance: dist,
                        });
                        while best_seen.len() > ef {
                            best_seen.pop();
                        }
                    }

                    // Only add to results if in allowlist
                    if allowlist.contains(&neighbor) {
                        let should_add = results.len() < ef
                            || results.peek().map_or(true, |f| dist < f.distance);
                        if should_add {
                            results.push(FurthestCandidate {
                                id: neighbor,
                                distance: dist,
                            });
                            while results.len() > ef {
                                results.pop();
                            }
                        }
                    }
                }
            }
        }

        // Convert to sorted vec
        let mut result_vec: Vec<Neighbor> = results
            .into_iter()
            .map(|fc| Neighbor {
                id: fc.id,
                distance: fc.distance,
            })
            .collect();
        result_vec.sort_by_key(|a| OrderedFloat(a.distance));
        result_vec
    }

    /// Selects neighbors using diversity-aware heuristic (Vamana-style).
    ///
    /// Instead of simply taking the M closest candidates, this checks whether
    /// each candidate is "covered" by an already-selected neighbor. A candidate
    /// is covered if any selected neighbor is closer to it than
    /// `alpha * distance(candidate, query)`. This preserves graph navigability
    /// by ensuring neighbors point to diverse regions of the space.
    fn select_neighbors_heuristic(
        &self,
        accessor: &impl VectorAccessor,
        candidates: &[Neighbor],
        m: usize,
    ) -> Vec<NodeId> {
        let alpha = self.config.alpha;
        let mut selected: Vec<(NodeId, Arc<[f32]>)> = Vec::with_capacity(m);

        for candidate in candidates {
            if selected.len() >= m {
                break;
            }
            let Some(cv) = accessor.get_vector(candidate.id) else {
                continue;
            };
            let covered = selected
                .iter()
                .any(|(_, sv)| self.vector_distance(&cv, sv) < alpha * candidate.distance);
            if !covered {
                selected.push((candidate.id, cv));
            }
        }

        selected.into_iter().map(|(id, _)| id).collect()
    }

    /// Prunes a neighbor list using distance-based diversity heuristic.
    ///
    /// Similar to `select_neighbors_heuristic` but operates on `(NodeId, f32)`
    /// distance pairs instead of `Neighbor` structs. Used during post-insert
    /// pruning where distances have already been computed.
    fn prune_neighbors_with_distances(
        neighbors: &mut Vec<NodeId>,
        distances: &[(NodeId, f32)],
        m: usize,
    ) {
        if neighbors.len() <= m {
            return;
        }

        // Sort by distance
        let mut sorted: Vec<_> = distances.to_vec();
        sorted.sort_by_key(|a| OrderedFloat(a.1));

        *neighbors = sorted.into_iter().take(m).map(|(id, _)| id).collect();
    }

    /// Computes distance between two raw vectors using the configured metric.
    #[inline]
    fn vector_distance(&self, a: &[f32], b: &[f32]) -> f32 {
        compute_distance(a, b, self.config.metric)
    }

    /// Computes the distance between a query vector and a stored node.
    fn node_distance(&self, accessor: &impl VectorAccessor, query: &[f32], id: NodeId) -> f32 {
        accessor
            .get_vector(id)
            .map_or(f32::MAX, |v| self.vector_distance(query, &v))
    }

    // ========================================================================
    // Batch Operations
    // ========================================================================

    /// Inserts multiple vectors in batch.
    ///
    /// This method inserts vectors sequentially into the HNSW graph structure
    /// but with optimized internal operations. For truly parallel construction
    /// of very large indexes, consider using multiple indexes and merging.
    ///
    /// # Arguments
    ///
    /// * `vectors` - Iterator of (NodeId, vector) pairs to insert
    /// * `accessor` - Vector accessor for reading vectors by ID
    ///
    /// # Panics
    ///
    /// Panics if any vector dimensions don't match the configuration.
    ///
    /// # Example
    ///
    /// ```
    /// use grafeo_core::index::vector::{HnswIndex, HnswConfig, DistanceMetric, VectorAccessor};
    /// use grafeo_common::types::NodeId;
    /// use std::sync::Arc;
    /// use std::collections::HashMap;
    ///
    /// let config = HnswConfig::new(384, DistanceMetric::Cosine);
    /// let index = HnswIndex::new(config);
    ///
    /// let vectors: Vec<(NodeId, Vec<f32>)> = (0..100)
    ///     .map(|i| (NodeId::new(i), vec![0.1f32; 384]))
    ///     .collect();
    ///
    /// // Build an accessor backed by a HashMap
    /// let map: HashMap<NodeId, Arc<[f32]>> = vectors
    ///     .iter()
    ///     .map(|(id, v)| (*id, Arc::from(v.as_slice())))
    ///     .collect();
    /// let accessor = move |id: NodeId| -> Option<Arc<[f32]>> { map.get(&id).cloned() };
    ///
    /// index.batch_insert(vectors.iter().map(|(id, v)| (*id, v.as_slice())), &accessor);
    /// ```
    pub fn batch_insert<'a, I>(&self, vectors: I, accessor: &impl VectorAccessor)
    where
        I: IntoIterator<Item = (NodeId, &'a [f32])>,
    {
        for (id, vector) in vectors {
            self.insert(id, vector, accessor);
        }
    }

    /// Searches for k nearest neighbors for multiple queries in parallel.
    ///
    /// This method runs multiple searches concurrently using rayon, providing
    /// significant speedup when you have many queries to execute.
    ///
    /// # Arguments
    ///
    /// * `queries` - Slice of query vectors (as `Vec<f32>` or similar)
    /// * `k` - Number of nearest neighbors to return for each query
    /// * `accessor` - Vector accessor for reading vectors by ID
    ///
    /// # Returns
    ///
    /// Vector of results, one per query. Each result is a vector of
    /// (NodeId, distance) pairs sorted by distance.
    ///
    /// # Panics
    ///
    /// Panics if any query vector dimensions don't match the configuration.
    ///
    /// # Example
    ///
    /// ```
    /// use grafeo_core::index::vector::{HnswIndex, HnswConfig, DistanceMetric, VectorAccessor};
    /// use grafeo_common::types::NodeId;
    /// use std::sync::Arc;
    /// use std::collections::HashMap;
    ///
    /// let config = HnswConfig::new(384, DistanceMetric::Cosine);
    /// let index = HnswIndex::new(config);
    ///
    /// // Build an accessor (empty for this example)
    /// let map: HashMap<NodeId, Arc<[f32]>> = HashMap::new();
    /// let accessor = move |id: NodeId| -> Option<Arc<[f32]>> { map.get(&id).cloned() };
    ///
    /// let queries: Vec<Vec<f32>> = vec![
    ///     vec![0.1f32; 384],
    ///     vec![0.2f32; 384],
    ///     vec![0.3f32; 384],
    /// ];
    ///
    /// let all_results = index.batch_search(&queries, 10, &accessor);
    /// assert_eq!(all_results.len(), 3);
    /// ```
    #[must_use]
    pub fn batch_search(
        &self,
        queries: &[Vec<f32>],
        k: usize,
        accessor: &impl VectorAccessor,
    ) -> Vec<Vec<(NodeId, f32)>> {
        #[cfg(feature = "parallel")]
        {
            use rayon::prelude::*;
            queries
                .par_iter()
                .map(|query| self.search(query, k, accessor))
                .collect()
        }
        #[cfg(not(feature = "parallel"))]
        {
            queries
                .iter()
                .map(|query| self.search(query, k, accessor))
                .collect()
        }
    }

    /// Searches for k nearest neighbors for multiple queries in parallel.
    ///
    /// This variant accepts query vectors as slices.
    #[must_use]
    pub fn batch_search_slices(
        &self,
        queries: &[&[f32]],
        k: usize,
        accessor: &impl VectorAccessor,
    ) -> Vec<Vec<(NodeId, f32)>> {
        #[cfg(feature = "parallel")]
        {
            use rayon::prelude::*;
            queries
                .par_iter()
                .map(|query| self.search(query, k, accessor))
                .collect()
        }
        #[cfg(not(feature = "parallel"))]
        {
            queries
                .iter()
                .map(|query| self.search(query, k, accessor))
                .collect()
        }
    }

    /// Searches with custom ef parameter for multiple queries in parallel.
    ///
    /// Higher ef values give better recall at the cost of latency.
    #[must_use]
    pub fn batch_search_with_ef(
        &self,
        queries: &[Vec<f32>],
        k: usize,
        ef: usize,
        accessor: &impl VectorAccessor,
    ) -> Vec<Vec<(NodeId, f32)>> {
        #[cfg(feature = "parallel")]
        {
            use rayon::prelude::*;
            queries
                .par_iter()
                .map(|query| self.search_with_ef(query, k, ef, accessor))
                .collect()
        }
        #[cfg(not(feature = "parallel"))]
        {
            queries
                .iter()
                .map(|query| self.search_with_ef(query, k, ef, accessor))
                .collect()
        }
    }

    /// Searches for k nearest neighbors for multiple queries with an allowlist filter.
    ///
    /// The beam width is automatically scaled based on allowlist selectivity.
    #[must_use]
    pub fn batch_search_with_filter(
        &self,
        queries: &[Vec<f32>],
        k: usize,
        allowlist: &HashSet<NodeId>,
        accessor: &impl VectorAccessor,
    ) -> Vec<Vec<(NodeId, f32)>> {
        #[cfg(feature = "parallel")]
        {
            use rayon::prelude::*;
            queries
                .par_iter()
                .map(|query| self.search_with_filter(query, k, allowlist, accessor))
                .collect()
        }
        #[cfg(not(feature = "parallel"))]
        {
            queries
                .iter()
                .map(|query| self.search_with_filter(query, k, allowlist, accessor))
                .collect()
        }
    }

    /// Searches with custom ef for multiple queries with an allowlist filter.
    #[must_use]
    pub fn batch_search_with_ef_and_filter(
        &self,
        queries: &[Vec<f32>],
        k: usize,
        ef: usize,
        allowlist: &HashSet<NodeId>,
        accessor: &impl VectorAccessor,
    ) -> Vec<Vec<(NodeId, f32)>> {
        #[cfg(feature = "parallel")]
        {
            use rayon::prelude::*;
            queries
                .par_iter()
                .map(|query| self.search_with_ef_and_filter(query, k, ef, allowlist, accessor))
                .collect()
        }
        #[cfg(not(feature = "parallel"))]
        {
            queries
                .iter()
                .map(|query| self.search_with_ef_and_filter(query, k, ef, allowlist, accessor))
                .collect()
        }
    }
}

impl std::fmt::Debug for HnswIndex {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HnswIndex")
            .field("config", &self.config)
            .field("len", &self.len())
            .field("max_level", &*self.max_level.read())
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::index::vector::DistanceMetric;

    fn create_test_vectors(n: usize, dim: usize) -> Vec<Vec<f32>> {
        (0..n)
            .map(|i| {
                (0..dim)
                    .map(|j| ((i * dim + j) as f32) / (n * dim) as f32)
                    .collect()
            })
            .collect()
    }

    /// Builds an accessor backed by a HashMap.
    fn make_accessor(map: &HashMap<NodeId, Arc<[f32]>>) -> impl VectorAccessor + '_ {
        move |id: NodeId| -> Option<Arc<[f32]>> { map.get(&id).cloned() }
    }

    #[test]
    fn test_hnsw_empty() {
        let config = HnswConfig::new(4, DistanceMetric::Euclidean);
        let index = HnswIndex::new(config);
        let map: HashMap<NodeId, Arc<[f32]>> = HashMap::new();
        let accessor = make_accessor(&map);

        assert!(index.is_empty());
        assert_eq!(index.len(), 0);
        assert!(
            index
                .search(&[0.0, 0.0, 0.0, 0.0], 10, &accessor)
                .is_empty()
        );
    }

    #[test]
    fn test_hnsw_single_insert() {
        let config = HnswConfig::new(4, DistanceMetric::Euclidean);
        let index = HnswIndex::new(config);

        let mut map: HashMap<NodeId, Arc<[f32]>> = HashMap::new();
        let v: Arc<[f32]> = vec![0.1, 0.2, 0.3, 0.4].into();
        map.insert(NodeId::new(1), v.clone());
        let accessor = make_accessor(&map);

        index.insert(NodeId::new(1), &v, &accessor);

        assert_eq!(index.len(), 1);
        assert!(index.contains(NodeId::new(1)));
        assert!(!index.contains(NodeId::new(2)));

        let results = index.search(&[0.1, 0.2, 0.3, 0.4], 1, &accessor);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0, NodeId::new(1));
        assert!(results[0].1 < 0.001); // Near-zero distance
    }

    #[test]
    fn test_hnsw_multiple_inserts() {
        let config = HnswConfig::new(4, DistanceMetric::Euclidean);
        let index = HnswIndex::with_seed(config, 42);

        let vectors = create_test_vectors(100, 4);

        let mut map: HashMap<NodeId, Arc<[f32]>> = HashMap::new();
        for (i, vec) in vectors.iter().enumerate() {
            let id = NodeId::new(i as u64 + 1);
            let arc: Arc<[f32]> = vec.as_slice().into();
            map.insert(id, arc);
        }
        let accessor = make_accessor(&map);

        for (i, vec) in vectors.iter().enumerate() {
            index.insert(NodeId::new(i as u64 + 1), vec, &accessor);
        }

        assert_eq!(index.len(), 100);

        // Search for nearest neighbors
        let query = &vectors[50];
        let results = index.search(query, 5, &accessor);

        assert_eq!(results.len(), 5);
        // The closest should be the vector itself
        assert_eq!(results[0].0, NodeId::new(51));
        assert!(results[0].1 < 0.001);
    }

    #[test]
    fn test_hnsw_search_returns_sorted() {
        let config = HnswConfig::new(4, DistanceMetric::Euclidean);
        let index = HnswIndex::with_seed(config, 42);

        let vectors = create_test_vectors(50, 4);

        let mut map: HashMap<NodeId, Arc<[f32]>> = HashMap::new();
        for (i, vec) in vectors.iter().enumerate() {
            let id = NodeId::new(i as u64 + 1);
            let arc: Arc<[f32]> = vec.as_slice().into();
            map.insert(id, arc);
        }
        let accessor = make_accessor(&map);

        for (i, vec) in vectors.iter().enumerate() {
            index.insert(NodeId::new(i as u64 + 1), vec, &accessor);
        }

        let query = [0.5, 0.5, 0.5, 0.5];
        let results = index.search(&query, 10, &accessor);

        // Verify sorted by distance
        for i in 1..results.len() {
            assert!(results[i - 1].1 <= results[i].1);
        }
    }

    #[test]
    fn test_hnsw_remove() {
        let config = HnswConfig::new(4, DistanceMetric::Euclidean);
        let index = HnswIndex::new(config);

        let mut map: HashMap<NodeId, Arc<[f32]>> = HashMap::new();
        map.insert(NodeId::new(1), vec![0.1, 0.2, 0.3, 0.4].into());
        map.insert(NodeId::new(2), vec![0.5, 0.6, 0.7, 0.8].into());
        let accessor = make_accessor(&map);

        index.insert(NodeId::new(1), &[0.1, 0.2, 0.3, 0.4], &accessor);
        index.insert(NodeId::new(2), &[0.5, 0.6, 0.7, 0.8], &accessor);

        assert_eq!(index.len(), 2);

        assert!(index.remove(NodeId::new(1)));
        assert_eq!(index.len(), 1);
        assert!(!index.contains(NodeId::new(1)));
        assert!(index.contains(NodeId::new(2)));

        // Removing again returns false
        assert!(!index.remove(NodeId::new(1)));
    }

    #[test]
    fn test_hnsw_cosine_metric() {
        let config = HnswConfig::new(4, DistanceMetric::Cosine);
        let index = HnswIndex::with_seed(config, 42);

        let mut map: HashMap<NodeId, Arc<[f32]>> = HashMap::new();
        map.insert(NodeId::new(1), vec![1.0, 0.0, 0.0, 0.0].into());
        map.insert(NodeId::new(2), vec![0.0, 1.0, 0.0, 0.0].into());
        map.insert(NodeId::new(3), vec![0.707, 0.707, 0.0, 0.0].into());
        let accessor = make_accessor(&map);

        // Insert normalized vectors
        index.insert(NodeId::new(1), &[1.0, 0.0, 0.0, 0.0], &accessor);
        index.insert(NodeId::new(2), &[0.0, 1.0, 0.0, 0.0], &accessor);
        index.insert(NodeId::new(3), &[0.707, 0.707, 0.0, 0.0], &accessor);

        // Query similar to node 1
        let results = index.search(&[0.9, 0.1, 0.0, 0.0], 3, &accessor);

        // Node 1 should be closest (most similar direction)
        assert_eq!(results[0].0, NodeId::new(1));
    }

    #[test]
    fn test_hnsw_ef_parameter() {
        let config = HnswConfig::new(4, DistanceMetric::Euclidean);
        let index = HnswIndex::with_seed(config, 42);

        let vectors = create_test_vectors(100, 4);

        let mut map: HashMap<NodeId, Arc<[f32]>> = HashMap::new();
        for (i, vec) in vectors.iter().enumerate() {
            let id = NodeId::new(i as u64 + 1);
            let arc: Arc<[f32]> = vec.as_slice().into();
            map.insert(id, arc);
        }
        let accessor = make_accessor(&map);

        for (i, vec) in vectors.iter().enumerate() {
            index.insert(NodeId::new(i as u64 + 1), vec, &accessor);
        }

        let query = [0.5, 0.5, 0.5, 0.5];

        // Higher ef should give same or better results
        let results_low = index.search_with_ef(&query, 5, 10, &accessor);
        let results_high = index.search_with_ef(&query, 5, 100, &accessor);

        assert_eq!(results_low.len(), 5);
        assert_eq!(results_high.len(), 5);

        // High ef should find equal or better (smaller) distances
        assert!(results_high[0].1 <= results_low[0].1);
    }

    #[test]
    #[should_panic(expected = "Vector dimensions mismatch")]
    fn test_hnsw_dimension_mismatch_insert() {
        let config = HnswConfig::new(4, DistanceMetric::Euclidean);
        let index = HnswIndex::new(config);
        let map: HashMap<NodeId, Arc<[f32]>> = HashMap::new();
        let accessor = make_accessor(&map);

        index.insert(NodeId::new(1), &[0.1, 0.2, 0.3], &accessor); // Wrong dimension
    }

    #[test]
    fn test_hnsw_max_elements_accepts_within_limit() {
        let config = HnswConfig::new(3, DistanceMetric::Euclidean).with_max_elements(3);
        let index = HnswIndex::new(config);
        let mut map: HashMap<NodeId, Arc<[f32]>> = HashMap::new();
        map.insert(NodeId::new(0), Arc::from([1.0f32, 0.0, 0.0].as_slice()));
        map.insert(NodeId::new(1), Arc::from([0.0f32, 1.0, 0.0].as_slice()));
        map.insert(NodeId::new(2), Arc::from([0.0f32, 0.0, 1.0].as_slice()));
        let accessor = make_accessor(&map);

        index.insert(NodeId::new(0), &[1.0, 0.0, 0.0], &accessor);
        index.insert(NodeId::new(1), &[0.0, 1.0, 0.0], &accessor);
        index.insert(NodeId::new(2), &[0.0, 0.0, 1.0], &accessor);
        assert_eq!(index.len(), 3);
    }

    #[test]
    #[should_panic(expected = "HNSW index is full")]
    fn test_hnsw_rejects_above_max_elements() {
        let config = HnswConfig::new(3, DistanceMetric::Euclidean).with_max_elements(2);
        let index = HnswIndex::new(config);
        let mut map: HashMap<NodeId, Arc<[f32]>> = HashMap::new();
        map.insert(NodeId::new(0), Arc::from([1.0f32, 0.0, 0.0].as_slice()));
        map.insert(NodeId::new(1), Arc::from([0.0f32, 1.0, 0.0].as_slice()));
        map.insert(NodeId::new(2), Arc::from([0.0f32, 0.0, 1.0].as_slice()));
        let accessor = make_accessor(&map);

        index.insert(NodeId::new(0), &[1.0, 0.0, 0.0], &accessor);
        index.insert(NodeId::new(1), &[0.0, 1.0, 0.0], &accessor);
        index.insert(NodeId::new(2), &[0.0, 0.0, 1.0], &accessor); // Should panic
    }

    #[test]
    #[should_panic(expected = "Query dimensions mismatch")]
    fn test_hnsw_dimension_mismatch_search() {
        let config = HnswConfig::new(4, DistanceMetric::Euclidean);
        let index = HnswIndex::new(config);

        let mut map: HashMap<NodeId, Arc<[f32]>> = HashMap::new();
        map.insert(NodeId::new(1), vec![0.1, 0.2, 0.3, 0.4].into());
        let accessor = make_accessor(&map);

        index.insert(NodeId::new(1), &[0.1, 0.2, 0.3, 0.4], &accessor);
        let _ = index.search(&[0.1, 0.2, 0.3], 1, &accessor); // Wrong dimension
    }

    #[test]
    fn test_hnsw_batch_insert() {
        let config = HnswConfig::new(4, DistanceMetric::Euclidean);
        let index = HnswIndex::with_seed(config, 42);

        let vectors = create_test_vectors(100, 4);

        let mut map: HashMap<NodeId, Arc<[f32]>> = HashMap::new();
        for (i, vec) in vectors.iter().enumerate() {
            let id = NodeId::new(i as u64 + 1);
            let arc: Arc<[f32]> = vec.as_slice().into();
            map.insert(id, arc);
        }
        let accessor = make_accessor(&map);

        let pairs: Vec<_> = vectors
            .iter()
            .enumerate()
            .map(|(i, v)| (NodeId::new(i as u64 + 1), v.as_slice()))
            .collect();

        index.batch_insert(pairs, &accessor);

        assert_eq!(index.len(), 100);

        // Verify search still works
        let results = index.search(&vectors[50], 5, &accessor);
        assert_eq!(results.len(), 5);
        assert_eq!(results[0].0, NodeId::new(51));
    }

    #[test]
    fn test_hnsw_batch_search() {
        let config = HnswConfig::new(4, DistanceMetric::Euclidean);
        let index = HnswIndex::with_seed(config, 42);

        let vectors = create_test_vectors(100, 4);

        let mut map: HashMap<NodeId, Arc<[f32]>> = HashMap::new();
        for (i, vec) in vectors.iter().enumerate() {
            let id = NodeId::new(i as u64 + 1);
            let arc: Arc<[f32]> = vec.as_slice().into();
            map.insert(id, arc);
        }
        let accessor = make_accessor(&map);

        for (i, vec) in vectors.iter().enumerate() {
            index.insert(NodeId::new(i as u64 + 1), vec, &accessor);
        }

        // Batch search with 5 queries
        let queries: Vec<Vec<f32>> = (0..5).map(|i| vectors[i * 20].clone()).collect();

        let all_results = index.batch_search(&queries, 3, &accessor);

        assert_eq!(all_results.len(), 5);
        for (i, results) in all_results.iter().enumerate() {
            assert_eq!(results.len(), 3);
            // First result should be the query vector itself
            assert_eq!(results[0].0, NodeId::new((i * 20 + 1) as u64));
            assert!(results[0].1 < 0.001);
        }
    }

    #[test]
    fn test_hnsw_batch_search_with_ef() {
        let config = HnswConfig::new(4, DistanceMetric::Euclidean);
        let index = HnswIndex::with_seed(config, 42);

        let vectors = create_test_vectors(100, 4);

        let mut map: HashMap<NodeId, Arc<[f32]>> = HashMap::new();
        for (i, vec) in vectors.iter().enumerate() {
            let id = NodeId::new(i as u64 + 1);
            let arc: Arc<[f32]> = vec.as_slice().into();
            map.insert(id, arc);
        }
        let accessor = make_accessor(&map);

        for (i, vec) in vectors.iter().enumerate() {
            index.insert(NodeId::new(i as u64 + 1), vec, &accessor);
        }

        let queries: Vec<Vec<f32>> = vec![vectors[25].clone(), vectors[75].clone()];

        // Search with higher ef for better recall
        let results = index.batch_search_with_ef(&queries, 5, 100, &accessor);

        assert_eq!(results.len(), 2);
        assert_eq!(results[0].len(), 5);
        assert_eq!(results[1].len(), 5);
    }

    #[test]
    fn test_hnsw_batch_search_empty_index() {
        let config = HnswConfig::new(4, DistanceMetric::Euclidean);
        let index = HnswIndex::new(config);
        let map: HashMap<NodeId, Arc<[f32]>> = HashMap::new();
        let accessor = make_accessor(&map);

        let queries = vec![vec![0.0f32, 0.0, 0.0, 0.0]];
        let results = index.batch_search(&queries, 10, &accessor);

        assert_eq!(results.len(), 1);
        assert!(results[0].is_empty());
    }

    /// Brute-force k-NN for recall verification.
    fn brute_force_knn(
        vectors: &[Vec<f32>],
        query: &[f32],
        k: usize,
        metric: DistanceMetric,
    ) -> Vec<usize> {
        let mut dists: Vec<(usize, f32)> = vectors
            .iter()
            .enumerate()
            .map(|(i, v)| (i, crate::index::vector::compute_distance(query, v, metric)))
            .collect();
        dists.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        dists.into_iter().take(k).map(|(i, _)| i).collect()
    }

    #[test]
    fn test_hnsw_recall_euclidean() {
        // 1000 vectors, 20 dimensions, matches ann-benchmarks random-xs profile
        let n = 1000;
        let dim = 20;
        let k = 10;
        let num_queries = 100;

        // Deterministic pseudo-random vectors via linear congruential generator
        let mut seed: u64 = 12345;
        let mut rand_f32 = || -> f32 {
            seed = seed.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
            ((seed >> 33) as f32) / (u32::MAX as f32)
        };

        let vectors: Vec<Vec<f32>> = (0..n)
            .map(|_| (0..dim).map(|_| rand_f32()).collect())
            .collect();

        let config = HnswConfig::new(dim, DistanceMetric::Euclidean).with_m(16);
        let index = HnswIndex::with_seed(config, 42);

        let mut map: HashMap<NodeId, Arc<[f32]>> = HashMap::new();
        for (i, vec) in vectors.iter().enumerate() {
            let id = NodeId::new(i as u64);
            let arc: Arc<[f32]> = vec.as_slice().into();
            map.insert(id, arc);
        }
        let accessor = make_accessor(&map);

        for (i, vec) in vectors.iter().enumerate() {
            index.insert(NodeId::new(i as u64), vec, &accessor);
        }

        // Measure recall over num_queries random queries
        let queries: Vec<Vec<f32>> = (0..num_queries)
            .map(|_| (0..dim).map(|_| rand_f32()).collect())
            .collect();

        let mut total_recall = 0.0f64;
        for query in &queries {
            let ground_truth = brute_force_knn(&vectors, query, k, DistanceMetric::Euclidean);
            let gt_set: std::collections::HashSet<u64> =
                ground_truth.iter().map(|&i| i as u64).collect();

            let results = index.search_with_ef(query, k, 50, &accessor);
            let found: std::collections::HashSet<u64> =
                results.iter().map(|(id, _)| id.as_u64()).collect();

            let overlap = gt_set.intersection(&found).count();
            total_recall += overlap as f64 / k as f64;
        }

        let avg_recall = total_recall / num_queries as f64;
        assert!(
            avg_recall >= 0.90,
            "Recall {avg_recall:.3} is below 0.90 threshold at M=16/ef=50"
        );
    }

    #[test]
    fn test_hnsw_recall_cosine() {
        let n = 500;
        let dim = 20;
        let k = 10;
        let num_queries = 50;

        let mut seed: u64 = 67890;
        let mut rand_f32 = || -> f32 {
            seed = seed.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
            ((seed >> 33) as f32) / (u32::MAX as f32)
        };

        let vectors: Vec<Vec<f32>> = (0..n)
            .map(|_| (0..dim).map(|_| rand_f32()).collect())
            .collect();

        let config = HnswConfig::new(dim, DistanceMetric::Cosine).with_m(16);
        let index = HnswIndex::with_seed(config, 42);

        let mut map: HashMap<NodeId, Arc<[f32]>> = HashMap::new();
        for (i, vec) in vectors.iter().enumerate() {
            let id = NodeId::new(i as u64);
            let arc: Arc<[f32]> = vec.as_slice().into();
            map.insert(id, arc);
        }
        let accessor = make_accessor(&map);

        for (i, vec) in vectors.iter().enumerate() {
            index.insert(NodeId::new(i as u64), vec, &accessor);
        }

        let queries: Vec<Vec<f32>> = (0..num_queries)
            .map(|_| (0..dim).map(|_| rand_f32()).collect())
            .collect();

        let mut total_recall = 0.0f64;
        for query in &queries {
            let ground_truth = brute_force_knn(&vectors, query, k, DistanceMetric::Cosine);
            let gt_set: std::collections::HashSet<u64> =
                ground_truth.iter().map(|&i| i as u64).collect();

            let results = index.search_with_ef(query, k, 50, &accessor);
            let found: std::collections::HashSet<u64> =
                results.iter().map(|(id, _)| id.as_u64()).collect();

            let overlap = gt_set.intersection(&found).count();
            total_recall += overlap as f64 / k as f64;
        }

        let avg_recall = total_recall / num_queries as f64;
        assert!(
            avg_recall >= 0.90,
            "Cosine recall {avg_recall:.3} is below 0.90 threshold at M=16/ef=50"
        );
    }

    #[test]
    fn test_diversity_pruning_prevents_clustering() {
        // Verify that diversity pruning selects diverse neighbors, not just closest
        let dim = 4;
        let config = HnswConfig::new(dim, DistanceMetric::Euclidean).with_m(4);
        let index = HnswIndex::with_seed(config, 42);

        let mut map: HashMap<NodeId, Arc<[f32]>> = HashMap::new();
        map.insert(NodeId::new(0), vec![0.0, 0.0, 0.0, 0.0].into());
        map.insert(NodeId::new(1), vec![0.01, 0.0, 0.0, 0.0].into());
        map.insert(NodeId::new(2), vec![0.02, 0.0, 0.0, 0.0].into());
        map.insert(NodeId::new(3), vec![0.03, 0.0, 0.0, 0.0].into());
        map.insert(NodeId::new(4), vec![0.04, 0.0, 0.0, 0.0].into());
        map.insert(NodeId::new(5), vec![0.0, 1.0, 0.0, 0.0].into());
        let accessor = make_accessor(&map);

        // Insert a cluster of very similar vectors and one outlier
        index.insert(NodeId::new(0), &[0.0, 0.0, 0.0, 0.0], &accessor);
        index.insert(NodeId::new(1), &[0.01, 0.0, 0.0, 0.0], &accessor);
        index.insert(NodeId::new(2), &[0.02, 0.0, 0.0, 0.0], &accessor);
        index.insert(NodeId::new(3), &[0.03, 0.0, 0.0, 0.0], &accessor);
        index.insert(NodeId::new(4), &[0.04, 0.0, 0.0, 0.0], &accessor);
        // Outlier in a different direction
        index.insert(NodeId::new(5), &[0.0, 1.0, 0.0, 0.0], &accessor);

        // Search for the outlier; it should be findable
        let results = index.search(&[0.0, 0.9, 0.0, 0.0], 1, &accessor);
        assert_eq!(results[0].0, NodeId::new(5));
    }

    // ── Edge case tests ─────────────────────────────────────────────

    #[test]
    fn test_single_vector() {
        let config = HnswConfig::new(3, DistanceMetric::Euclidean);
        let index = HnswIndex::new(config);

        let mut map: HashMap<NodeId, Arc<[f32]>> = HashMap::new();
        map.insert(NodeId::new(0), vec![1.0, 0.0, 0.0].into());
        let accessor = make_accessor(&map);

        index.insert(NodeId::new(0), &[1.0, 0.0, 0.0], &accessor);

        let results = index.search(&[1.0, 0.0, 0.0], 1, &accessor);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0, NodeId::new(0));
        assert!(results[0].1 < 0.01);
    }

    #[test]
    fn test_search_k_larger_than_index() {
        let config = HnswConfig::new(3, DistanceMetric::Euclidean);
        let index = HnswIndex::new(config);

        let mut map: HashMap<NodeId, Arc<[f32]>> = HashMap::new();
        map.insert(NodeId::new(0), vec![1.0, 0.0, 0.0].into());
        map.insert(NodeId::new(1), vec![0.0, 1.0, 0.0].into());
        let accessor = make_accessor(&map);

        index.insert(NodeId::new(0), &[1.0, 0.0, 0.0], &accessor);
        index.insert(NodeId::new(1), &[0.0, 1.0, 0.0], &accessor);

        // k=10 but only 2 vectors
        let results = index.search(&[1.0, 0.0, 0.0], 10, &accessor);
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn test_empty_index_search() {
        let config = HnswConfig::new(3, DistanceMetric::Euclidean);
        let index = HnswIndex::new(config);
        let map: HashMap<NodeId, Arc<[f32]>> = HashMap::new();
        let accessor = make_accessor(&map);

        let results = index.search(&[1.0, 0.0, 0.0], 5, &accessor);
        assert!(results.is_empty());
    }

    #[test]
    fn test_remove_and_search() {
        let config = HnswConfig::new(3, DistanceMetric::Euclidean);
        let index = HnswIndex::new(config);

        let mut map: HashMap<NodeId, Arc<[f32]>> = HashMap::new();
        map.insert(NodeId::new(0), vec![1.0, 0.0, 0.0].into());
        map.insert(NodeId::new(1), vec![0.0, 1.0, 0.0].into());
        map.insert(NodeId::new(2), vec![0.0, 0.0, 1.0].into());
        let accessor = make_accessor(&map);

        index.insert(NodeId::new(0), &[1.0, 0.0, 0.0], &accessor);
        index.insert(NodeId::new(1), &[0.0, 1.0, 0.0], &accessor);
        index.insert(NodeId::new(2), &[0.0, 0.0, 1.0], &accessor);

        index.remove(NodeId::new(1));
        let results = index.search(&[0.0, 1.0, 0.0], 3, &accessor);
        // Removed node should not appear
        assert!(results.iter().all(|(id, _)| *id != NodeId::new(1)));
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn test_duplicate_insert() {
        let config = HnswConfig::new(3, DistanceMetric::Euclidean);
        let index = HnswIndex::new(config);

        let mut map: HashMap<NodeId, Arc<[f32]>> = HashMap::new();
        map.insert(NodeId::new(0), vec![1.0, 0.0, 0.0].into());
        let accessor = make_accessor(&map);

        index.insert(NodeId::new(0), &[1.0, 0.0, 0.0], &accessor);

        // Update the accessor with the new vector for node 0
        let mut map2: HashMap<NodeId, Arc<[f32]>> = HashMap::new();
        map2.insert(NodeId::new(0), vec![0.0, 1.0, 0.0].into());
        let accessor2 = make_accessor(&map2);

        index.insert(NodeId::new(0), &[0.0, 1.0, 0.0], &accessor2); // Same ID, different vector

        assert_eq!(index.len(), 1);
        // Should use the latest vector
        let results = index.search(&[0.0, 1.0, 0.0], 1, &accessor2);
        assert_eq!(results[0].0, NodeId::new(0));
    }

    #[test]
    fn test_search_with_ef_zero() {
        let config = HnswConfig::new(3, DistanceMetric::Euclidean);
        let index = HnswIndex::new(config);

        let mut map: HashMap<NodeId, Arc<[f32]>> = HashMap::new();
        map.insert(NodeId::new(0), vec![1.0, 0.0, 0.0].into());
        let accessor = make_accessor(&map);

        index.insert(NodeId::new(0), &[1.0, 0.0, 0.0], &accessor);

        // ef=0 should still return results (search uses max(ef, k))
        let results = index.search_with_ef(&[1.0, 0.0, 0.0], 1, 0, &accessor);
        // Behavior may vary but should not panic
        assert!(results.len() <= 1);
    }

    #[test]
    fn test_all_metrics_search() {
        for metric in [
            DistanceMetric::Cosine,
            DistanceMetric::Euclidean,
            DistanceMetric::DotProduct,
            DistanceMetric::Manhattan,
        ] {
            let config = HnswConfig::new(3, metric);
            let index = HnswIndex::new(config);

            let mut map: HashMap<NodeId, Arc<[f32]>> = HashMap::new();
            map.insert(NodeId::new(0), vec![1.0, 0.0, 0.0].into());
            map.insert(NodeId::new(1), vec![0.0, 1.0, 0.0].into());
            let accessor = make_accessor(&map);

            index.insert(NodeId::new(0), &[1.0, 0.0, 0.0], &accessor);
            index.insert(NodeId::new(1), &[0.0, 1.0, 0.0], &accessor);

            let results = index.search(&[1.0, 0.0, 0.0], 2, &accessor);
            assert_eq!(results.len(), 2, "Failed for metric {metric:?}");
            assert_eq!(
                results[0].0,
                NodeId::new(0),
                "Closest not correct for metric {metric:?}"
            );
        }
    }

    #[test]
    fn test_batch_search_consistency() {
        let config = HnswConfig::new(3, DistanceMetric::Euclidean);
        let index = HnswIndex::new(config);

        let mut map: HashMap<NodeId, Arc<[f32]>> = HashMap::new();
        map.insert(NodeId::new(0), vec![1.0, 0.0, 0.0].into());
        map.insert(NodeId::new(1), vec![0.0, 1.0, 0.0].into());
        map.insert(NodeId::new(2), vec![0.0, 0.0, 1.0].into());
        let accessor = make_accessor(&map);

        index.insert(NodeId::new(0), &[1.0, 0.0, 0.0], &accessor);
        index.insert(NodeId::new(1), &[0.0, 1.0, 0.0], &accessor);
        index.insert(NodeId::new(2), &[0.0, 0.0, 1.0], &accessor);

        let queries: Vec<Vec<f32>> = vec![
            vec![1.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0],
            vec![0.0, 0.0, 1.0],
        ];

        let batch_results = index.batch_search(&queries, 1, &accessor);
        assert_eq!(batch_results.len(), 3);

        // Each query should find its exact match
        for (i, results) in batch_results.iter().enumerate() {
            assert_eq!(results[0].0, NodeId::new(i as u64));
        }
    }

    #[test]
    fn test_with_capacity_constructor() {
        let config = HnswConfig::new(3, DistanceMetric::Euclidean);
        let index = HnswIndex::with_capacity(config, 100);
        assert_eq!(index.len(), 0);
        assert!(index.is_empty());

        let mut map: HashMap<NodeId, Arc<[f32]>> = HashMap::new();
        map.insert(NodeId::new(0), vec![1.0, 0.0, 0.0].into());
        let accessor = make_accessor(&map);

        index.insert(NodeId::new(0), &[1.0, 0.0, 0.0], &accessor);
        assert_eq!(index.len(), 1);
        assert!(!index.is_empty());
    }

    #[test]
    fn test_high_m_value() {
        // M larger than number of nodes
        let config = HnswConfig::new(3, DistanceMetric::Euclidean).with_m(64);
        let index = HnswIndex::new(config);

        let mut map: HashMap<NodeId, Arc<[f32]>> = HashMap::new();
        map.insert(NodeId::new(0), vec![1.0, 0.0, 0.0].into());
        map.insert(NodeId::new(1), vec![0.0, 1.0, 0.0].into());
        let accessor = make_accessor(&map);

        index.insert(NodeId::new(0), &[1.0, 0.0, 0.0], &accessor);
        index.insert(NodeId::new(1), &[0.0, 1.0, 0.0], &accessor);

        let results = index.search(&[1.0, 0.0, 0.0], 2, &accessor);
        assert_eq!(results.len(), 2);
    }

    // ── Filtered search tests ─────────────────────────────────────

    #[test]
    fn test_filtered_search_returns_only_allowlisted() {
        let config = HnswConfig::new(4, DistanceMetric::Euclidean);
        let index = HnswIndex::with_seed(config, 42);

        let vectors = create_test_vectors(50, 4);

        let mut map: HashMap<NodeId, Arc<[f32]>> = HashMap::new();
        for (i, vec) in vectors.iter().enumerate() {
            let id = NodeId::new(i as u64 + 1);
            let arc: Arc<[f32]> = vec.as_slice().into();
            map.insert(id, arc);
        }
        let accessor = make_accessor(&map);

        for (i, vec) in vectors.iter().enumerate() {
            index.insert(NodeId::new(i as u64 + 1), vec, &accessor);
        }

        // Allowlist: only even-numbered nodes
        let allowlist: HashSet<NodeId> = (1..=50).filter(|i| i % 2 == 0).map(NodeId::new).collect();

        let results = index.search_with_filter(&vectors[25], 5, &allowlist, &accessor);
        assert!(!results.is_empty());
        assert!(results.len() <= 5);

        // Every result must be in the allowlist
        for (id, _) in &results {
            assert!(allowlist.contains(id), "Result {id:?} not in allowlist");
        }
    }

    #[test]
    fn test_filtered_search_empty_allowlist() {
        let config = HnswConfig::new(4, DistanceMetric::Euclidean);
        let index = HnswIndex::with_seed(config, 42);

        let vectors = create_test_vectors(20, 4);

        let mut map: HashMap<NodeId, Arc<[f32]>> = HashMap::new();
        for (i, vec) in vectors.iter().enumerate() {
            let id = NodeId::new(i as u64 + 1);
            let arc: Arc<[f32]> = vec.as_slice().into();
            map.insert(id, arc);
        }
        let accessor = make_accessor(&map);

        for (i, vec) in vectors.iter().enumerate() {
            index.insert(NodeId::new(i as u64 + 1), vec, &accessor);
        }

        let allowlist: HashSet<NodeId> = HashSet::new();
        let results = index.search_with_filter(&vectors[5], 5, &allowlist, &accessor);
        assert!(results.is_empty());
    }

    #[test]
    fn test_filtered_search_full_allowlist_matches_unfiltered() {
        let config = HnswConfig::new(4, DistanceMetric::Euclidean);
        let index = HnswIndex::with_seed(config, 42);

        let vectors = create_test_vectors(50, 4);

        let mut map: HashMap<NodeId, Arc<[f32]>> = HashMap::new();
        for (i, vec) in vectors.iter().enumerate() {
            let id = NodeId::new(i as u64 + 1);
            let arc: Arc<[f32]> = vec.as_slice().into();
            map.insert(id, arc);
        }
        let accessor = make_accessor(&map);

        for (i, vec) in vectors.iter().enumerate() {
            index.insert(NodeId::new(i as u64 + 1), vec, &accessor);
        }

        // Allowlist contains all nodes
        let allowlist: HashSet<NodeId> = (1..=50).map(NodeId::new).collect();
        let query = &vectors[25];

        let unfiltered = index.search_with_ef(query, 5, 200, &accessor);
        let filtered = index.search_with_ef_and_filter(query, 5, 200, &allowlist, &accessor);

        // With full allowlist, results should match unfiltered (same ef)
        assert_eq!(unfiltered.len(), filtered.len());
        for (u, f) in unfiltered.iter().zip(filtered.iter()) {
            assert_eq!(u.0, f.0);
        }
    }

    #[test]
    fn test_filtered_search_single_allowlisted_node() {
        let config = HnswConfig::new(4, DistanceMetric::Euclidean);
        let index = HnswIndex::with_seed(config, 42);

        let vectors = create_test_vectors(50, 4);

        let mut map: HashMap<NodeId, Arc<[f32]>> = HashMap::new();
        for (i, vec) in vectors.iter().enumerate() {
            let id = NodeId::new(i as u64 + 1);
            let arc: Arc<[f32]> = vec.as_slice().into();
            map.insert(id, arc);
        }
        let accessor = make_accessor(&map);

        for (i, vec) in vectors.iter().enumerate() {
            index.insert(NodeId::new(i as u64 + 1), vec, &accessor);
        }

        // Only one node allowed
        let allowlist: HashSet<NodeId> = [NodeId::new(30)].into_iter().collect();
        let results = index.search_with_filter(&vectors[25], 5, &allowlist, &accessor);

        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0, NodeId::new(30));
    }

    #[test]
    fn test_filtered_search_sorted_by_distance() {
        let config = HnswConfig::new(4, DistanceMetric::Euclidean);
        let index = HnswIndex::with_seed(config, 42);

        let vectors = create_test_vectors(100, 4);

        let mut map: HashMap<NodeId, Arc<[f32]>> = HashMap::new();
        for (i, vec) in vectors.iter().enumerate() {
            let id = NodeId::new(i as u64 + 1);
            let arc: Arc<[f32]> = vec.as_slice().into();
            map.insert(id, arc);
        }
        let accessor = make_accessor(&map);

        for (i, vec) in vectors.iter().enumerate() {
            index.insert(NodeId::new(i as u64 + 1), vec, &accessor);
        }

        let allowlist: HashSet<NodeId> =
            (1..=100).filter(|i| i % 3 == 0).map(NodeId::new).collect();

        let results = index.search_with_filter(&[0.5, 0.5, 0.5, 0.5], 10, &allowlist, &accessor);
        for i in 1..results.len() {
            assert!(results[i - 1].1 <= results[i].1);
        }
    }

    #[test]
    fn test_batch_filtered_search() {
        let config = HnswConfig::new(4, DistanceMetric::Euclidean);
        let index = HnswIndex::with_seed(config, 42);

        let vectors = create_test_vectors(100, 4);

        let mut map: HashMap<NodeId, Arc<[f32]>> = HashMap::new();
        for (i, vec) in vectors.iter().enumerate() {
            let id = NodeId::new(i as u64 + 1);
            let arc: Arc<[f32]> = vec.as_slice().into();
            map.insert(id, arc);
        }
        let accessor = make_accessor(&map);

        for (i, vec) in vectors.iter().enumerate() {
            index.insert(NodeId::new(i as u64 + 1), vec, &accessor);
        }

        // Allowlist: nodes 1..=50
        let allowlist: HashSet<NodeId> = (1..=50).map(NodeId::new).collect();
        let queries: Vec<Vec<f32>> = vec![vectors[10].clone(), vectors[70].clone()];

        let all_results = index.batch_search_with_filter(&queries, 5, &allowlist, &accessor);
        assert_eq!(all_results.len(), 2);

        for results in &all_results {
            for (id, _) in results {
                assert!(allowlist.contains(id));
            }
        }
    }

    #[test]
    // reason: test indices are small known values
    #[allow(clippy::cast_sign_loss, clippy::cast_possible_truncation)]
    fn test_filtered_search_ef_scaling() {
        // Verify that auto-scaling ef produces reasonable recall
        let n = 500;
        let dim = 8;
        let k = 10;
        let config = HnswConfig::new(dim, DistanceMetric::Euclidean).with_m(16);
        let index = HnswIndex::with_seed(config, 42);

        let mut seed: u64 = 99999;
        let mut rand_f32 = || -> f32 {
            seed = seed.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
            ((seed >> 33) as f32) / (u32::MAX as f32)
        };

        let vectors: Vec<Vec<f32>> = (0..n)
            .map(|_| (0..dim).map(|_| rand_f32()).collect())
            .collect();

        let mut map: HashMap<NodeId, Arc<[f32]>> = HashMap::new();
        for (i, vec) in vectors.iter().enumerate() {
            let id = NodeId::new(i as u64);
            let arc: Arc<[f32]> = vec.as_slice().into();
            map.insert(id, arc);
        }
        let accessor = make_accessor(&map);

        for (i, vec) in vectors.iter().enumerate() {
            index.insert(NodeId::new(i as u64), vec, &accessor);
        }

        // 20% allowlist, moderate selectivity
        let allowlist: HashSet<NodeId> = (0..n)
            .filter(|i| i % 5 == 0)
            .map(|i| NodeId::new(i as u64))
            .collect();

        let query: Vec<f32> = (0..dim).map(|_| rand_f32()).collect();

        // Brute-force ground truth (only among allowlisted nodes)
        let mut gt: Vec<(u64, f32)> = allowlist
            .iter()
            .map(|id| {
                let dist = crate::index::vector::compute_distance(
                    &query,
                    &vectors[id.as_u64() as usize],
                    DistanceMetric::Euclidean,
                );
                (id.as_u64(), dist)
            })
            .collect();
        gt.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        let gt_set: std::collections::HashSet<u64> = gt.iter().take(k).map(|(id, _)| *id).collect();

        let results = index.search_with_filter(&query, k, &allowlist, &accessor);
        let found: std::collections::HashSet<u64> =
            results.iter().map(|(id, _)| id.as_u64()).collect();

        let overlap = gt_set.intersection(&found).count();
        let recall = overlap as f64 / k as f64;
        assert!(
            recall >= 0.60,
            "Filtered recall {recall:.3} is below 0.60 threshold (20% selectivity)"
        );
    }

    #[test]
    fn test_filtered_search_cosine() {
        let config = HnswConfig::new(4, DistanceMetric::Cosine);
        let index = HnswIndex::with_seed(config, 42);

        let mut map: HashMap<NodeId, Arc<[f32]>> = HashMap::new();
        map.insert(NodeId::new(1), vec![1.0, 0.0, 0.0, 0.0].into());
        map.insert(NodeId::new(2), vec![0.0, 1.0, 0.0, 0.0].into());
        map.insert(NodeId::new(3), vec![0.707, 0.707, 0.0, 0.0].into());
        let accessor = make_accessor(&map);

        index.insert(NodeId::new(1), &[1.0, 0.0, 0.0, 0.0], &accessor);
        index.insert(NodeId::new(2), &[0.0, 1.0, 0.0, 0.0], &accessor);
        index.insert(NodeId::new(3), &[0.707, 0.707, 0.0, 0.0], &accessor);

        let allowlist: HashSet<NodeId> = [NodeId::new(2), NodeId::new(3)].into_iter().collect();
        let results = index.search_with_filter(&[0.9, 0.1, 0.0, 0.0], 2, &allowlist, &accessor);

        // Node 1 is closest overall but not in allowlist
        assert!(!results.is_empty());
        for (id, _) in &results {
            assert!(allowlist.contains(id));
        }
        // Node 3 should be closest among allowed
        assert_eq!(results[0].0, NodeId::new(3));
    }

    // ── Phase 7c-2: HnswIndex with mmap-backed topology ─────────────

    use crate::index::vector::paged_topology::{MmapTopology, serialize_topology};
    use bytes::Bytes;

    /// Build a small index in heap mode, snapshot its topology, swap
    /// the backend to mmap mode, and verify search returns identical
    /// results.
    #[test]
    fn alix_mmap_backed_search_matches_heap_search() {
        let config = HnswConfig::new(8, DistanceMetric::Euclidean);
        let heap_index = HnswIndex::with_seed(config.clone(), 42);

        let map: HashMap<NodeId, Arc<[f32]>> = (1..=20u64)
            .map(|i| {
                let v: Arc<[f32]> = (0..8u64)
                    .map(|j| (i.wrapping_mul(31).wrapping_add(j) % 17) as f32 / 17.0)
                    .collect::<Vec<_>>()
                    .into();
                (NodeId::new(i), v)
            })
            .collect();
        let accessor = |id: NodeId| -> Option<Arc<[f32]>> { map.get(&id).cloned() };

        for (id, v) in &map {
            heap_index.insert(*id, v, &accessor);
        }

        // Reference: search results from heap-backed index.
        let query: Vec<f32> = vec![0.1, 0.4, 0.6, 0.2, 0.8, 0.5, 0.3, 0.7];
        let heap_results = heap_index.search(&query, 5, &accessor);
        assert!(!heap_results.is_empty());

        // Snapshot + serialize + load back as mmap topology.
        let (ep, ml, nodes) = heap_index.snapshot_topology();
        let bytes = serialize_topology(ep, ml, &nodes);
        let topo = MmapTopology::from_bytes(Bytes::from(bytes)).expect("from_bytes");

        let mmap_index = HnswIndex::new(config);
        mmap_index.adopt_mmap_topology(topo);
        assert!(mmap_index.is_mmap_backed());

        let mmap_results = mmap_index.search(&query, 5, &accessor);

        // Same NodeIds in the same order, same distances.
        assert_eq!(mmap_results.len(), heap_results.len());
        for ((id_h, d_h), (id_m, d_m)) in heap_results.iter().zip(mmap_results.iter()) {
            assert_eq!(id_h, id_m);
            assert!((d_h - d_m).abs() < 1e-6);
        }
    }

    /// Mutating an mmap-backed index must panic with a clear message,
    /// not silently no-op or corrupt state.
    #[test]
    #[should_panic(expected = "mmap mode")]
    fn gus_mmap_backed_insert_panics() {
        let config = HnswConfig::new(4, DistanceMetric::Cosine);
        let nodes = vec![(NodeId::new(1), vec![vec![]])];
        let bytes = serialize_topology(Some(NodeId::new(1)), 0, &nodes);
        let topo = MmapTopology::from_bytes(Bytes::from(bytes)).expect("from_bytes");

        let index = HnswIndex::new(config);
        index.adopt_mmap_topology(topo);

        let map: HashMap<NodeId, Arc<[f32]>> = HashMap::new();
        let accessor = |id: NodeId| -> Option<Arc<[f32]>> { map.get(&id).cloned() };
        index.insert(NodeId::new(2), &[0.0, 0.0, 0.0, 0.0], &accessor);
    }

    /// Removing on an mmap-backed index must panic.
    #[test]
    #[should_panic(expected = "mmap mode")]
    fn vincent_mmap_backed_remove_panics() {
        let config = HnswConfig::new(4, DistanceMetric::Cosine);
        let nodes = vec![(NodeId::new(1), vec![vec![]])];
        let bytes = serialize_topology(Some(NodeId::new(1)), 0, &nodes);
        let topo = MmapTopology::from_bytes(Bytes::from(bytes)).expect("from_bytes");

        let index = HnswIndex::new(config);
        index.adopt_mmap_topology(topo);
        index.remove(NodeId::new(1));
    }

    /// `restore_topology` after `adopt_mmap_topology` must put the
    /// index back in heap mode and accept mutations again.
    #[test]
    fn jules_restore_topology_returns_to_heap_mode() {
        let config = HnswConfig::new(4, DistanceMetric::Cosine);
        let nodes = vec![(NodeId::new(1), vec![vec![]])];
        let bytes = serialize_topology(Some(NodeId::new(1)), 0, &nodes);
        let topo = MmapTopology::from_bytes(Bytes::from(bytes)).expect("from_bytes");

        let index = HnswIndex::new(config);
        index.adopt_mmap_topology(topo);
        assert!(index.is_mmap_backed());

        index.restore_topology(
            Some(NodeId::new(1)),
            0,
            vec![(NodeId::new(1), vec![vec![]])],
        );
        assert!(!index.is_mmap_backed());

        // Now insert should work without panicking.
        let map: HashMap<NodeId, Arc<[f32]>> = HashMap::new();
        let accessor = |id: NodeId| -> Option<Arc<[f32]>> { map.get(&id).cloned() };
        index.insert(NodeId::new(2), &[0.0, 0.0, 0.0, 0.0], &accessor);
        assert_eq!(index.len(), 2);
    }

    /// Heap memory savings: an mmap-backed index reports nearly zero
    /// heap usage (just the small struct), while the heap-backed
    /// equivalent reports significant overhead.
    #[test]
    fn mia_mmap_backed_heap_overhead_is_tiny() {
        let config = HnswConfig::new(8, DistanceMetric::Euclidean);
        let heap_index = HnswIndex::with_seed(config.clone(), 42);

        let map: HashMap<NodeId, Arc<[f32]>> = (1..=50u64)
            .map(|i| {
                let v: Arc<[f32]> = vec![0.1; 8].into();
                (NodeId::new(i), v)
            })
            .collect();
        let accessor = |id: NodeId| -> Option<Arc<[f32]>> { map.get(&id).cloned() };

        for (id, v) in &map {
            heap_index.insert(*id, v, &accessor);
        }
        let heap_bytes = heap_index.heap_memory_bytes();
        assert!(
            heap_bytes > 1000,
            "heap-mode should report > 1KB heap usage"
        );

        let (ep, ml, nodes) = heap_index.snapshot_topology();
        let bytes = serialize_topology(ep, ml, &nodes);
        let topo = MmapTopology::from_bytes(Bytes::from(bytes)).expect("from_bytes");

        let mmap_index = HnswIndex::new(config);
        mmap_index.adopt_mmap_topology(topo);
        let mmap_bytes = mmap_index.heap_memory_bytes();
        assert!(
            mmap_bytes < 256,
            "mmap-mode heap overhead should be < 256 bytes, got {mmap_bytes}"
        );
        assert!(
            mmap_bytes < heap_bytes / 10,
            "mmap-mode {mmap_bytes} should be far smaller than heap-mode {heap_bytes}"
        );
    }

    // ── Phase 7d: recall regression + variant coverage ──────────────

    /// Builds a 200-vector HNSW deterministically and runs the four
    /// search variants in both heap and mmap modes. Each variant must
    /// return identical (id, distance) sequences across modes.
    #[test]
    fn shosanna_all_search_variants_match_across_modes() {
        let config = HnswConfig::new(8, DistanceMetric::Euclidean);
        let heap_index = HnswIndex::with_seed(config.clone(), 7);

        // Deterministic pseudo-random vectors.
        let map: HashMap<NodeId, Arc<[f32]>> = (1..=200u64)
            .map(|i| {
                let v: Arc<[f32]> = (0..8u64)
                    .map(|j| {
                        let s = i.wrapping_mul(37).wrapping_add(j.wrapping_mul(101));
                        ((s % 1000) as f32) / 1000.0
                    })
                    .collect::<Vec<_>>()
                    .into();
                (NodeId::new(i), v)
            })
            .collect();
        let accessor = |id: NodeId| -> Option<Arc<[f32]>> { map.get(&id).cloned() };

        for (id, v) in &map {
            heap_index.insert(*id, v, &accessor);
        }

        let (ep, ml, nodes) = heap_index.snapshot_topology();
        let bytes = serialize_topology(ep, ml, &nodes);
        let topo = MmapTopology::from_bytes(Bytes::from(bytes)).expect("from_bytes");
        let mmap_index = HnswIndex::new(config);
        mmap_index.adopt_mmap_topology(topo);

        let query: Vec<f32> = vec![0.31, 0.42, 0.55, 0.18, 0.77, 0.91, 0.05, 0.62];

        // search()
        let h = heap_index.search(&query, 10, &accessor);
        let m = mmap_index.search(&query, 10, &accessor);
        assert_eq!(h.len(), m.len());
        for ((hid, hd), (mid, md)) in h.iter().zip(m.iter()) {
            assert_eq!(hid, mid);
            assert!((hd - md).abs() < 1e-6);
        }

        // search_with_ef()
        let h = heap_index.search_with_ef(&query, 10, 50, &accessor);
        let m = mmap_index.search_with_ef(&query, 10, 50, &accessor);
        assert_eq!(h, m);

        // search_with_filter()
        let allowlist: HashSet<NodeId> = (1..=100u64).map(NodeId::new).collect();
        let h = heap_index.search_with_filter(&query, 5, &allowlist, &accessor);
        let m = mmap_index.search_with_filter(&query, 5, &allowlist, &accessor);
        assert_eq!(h, m);

        // search_with_ef_and_filter()
        let h = heap_index.search_with_ef_and_filter(&query, 5, 80, &allowlist, &accessor);
        let m = mmap_index.search_with_ef_and_filter(&query, 5, 80, &allowlist, &accessor);
        assert_eq!(h, m);

        // batch_search()
        let queries = vec![query.clone(), vec![0.5; 8], vec![0.0; 8]];
        let h = heap_index.batch_search(&queries, 5, &accessor);
        let m = mmap_index.batch_search(&queries, 5, &accessor);
        assert_eq!(h, m);
    }

    /// Recall@k: search results from the heap-backed and mmap-backed
    /// indexes must be identical, so recall is trivially 100%. The
    /// test exists to fail loudly if a future refactor introduces any
    /// divergence (e.g. iterator order shift).
    #[test]
    fn butch_mmap_recall_at_10_is_100_percent() {
        let config = HnswConfig::new(16, DistanceMetric::Cosine);
        let heap_index = HnswIndex::with_seed(config.clone(), 1234);

        let map: HashMap<NodeId, Arc<[f32]>> = (1..=300u64)
            .map(|i| {
                let v: Arc<[f32]> = (0..16u64)
                    .map(|j| {
                        let s = i.wrapping_mul(53).wrapping_add(j.wrapping_mul(149));
                        ((s % 997) as f32) / 997.0
                    })
                    .collect::<Vec<_>>()
                    .into();
                (NodeId::new(i), v)
            })
            .collect();
        let accessor = |id: NodeId| -> Option<Arc<[f32]>> { map.get(&id).cloned() };

        for (id, v) in &map {
            heap_index.insert(*id, v, &accessor);
        }

        let (ep, ml, nodes) = heap_index.snapshot_topology();
        let bytes = serialize_topology(ep, ml, &nodes);
        let topo = MmapTopology::from_bytes(Bytes::from(bytes)).expect("from_bytes");
        let mmap_index = HnswIndex::new(config);
        mmap_index.adopt_mmap_topology(topo);

        // Run 20 different queries; each must match exactly.
        for q in 0..20u64 {
            let query: Vec<f32> = (0..16u64)
                .map(|j| {
                    let s = q.wrapping_mul(71).wrapping_add(j.wrapping_mul(211));
                    ((s % 991) as f32) / 991.0
                })
                .collect();

            let heap_results: HashSet<NodeId> = heap_index
                .search(&query, 10, &accessor)
                .into_iter()
                .map(|(id, _)| id)
                .collect();
            let mmap_results: HashSet<NodeId> = mmap_index
                .search(&query, 10, &accessor)
                .into_iter()
                .map(|(id, _)| id)
                .collect();

            // Recall@10 = |heap ∩ mmap| / |heap| = 1.0 because results
            // must be identical (deterministic byte-format read).
            let intersection = heap_results.intersection(&mmap_results).count();
            assert_eq!(
                intersection,
                heap_results.len(),
                "query {q}: recall@10 must be 100% (heap={heap_results:?}, mmap={mmap_results:?})"
            );
        }
    }

    /// Empty-allowlist filter must short-circuit cleanly in mmap mode.
    #[test]
    fn django_mmap_empty_allowlist_returns_empty() {
        let config = HnswConfig::new(4, DistanceMetric::Euclidean);
        let heap_index = HnswIndex::with_seed(config.clone(), 99);
        let map: HashMap<NodeId, Arc<[f32]>> = (1..=10u64)
            .map(|i| (NodeId::new(i), vec![0.1; 4].into()))
            .collect();
        let accessor = |id: NodeId| -> Option<Arc<[f32]>> { map.get(&id).cloned() };
        for (id, v) in &map {
            heap_index.insert(*id, v, &accessor);
        }

        let (ep, ml, nodes) = heap_index.snapshot_topology();
        let bytes = serialize_topology(ep, ml, &nodes);
        let topo = MmapTopology::from_bytes(Bytes::from(bytes)).expect("from_bytes");
        let mmap_index = HnswIndex::new(config);
        mmap_index.adopt_mmap_topology(topo);

        let allowlist: HashSet<NodeId> = HashSet::new();
        let results = mmap_index.search_with_filter(&[0.1; 4], 5, &allowlist, &accessor);
        assert!(results.is_empty());
    }

    /// Mmap-backed search on an empty index must not panic.
    #[test]
    fn beatrix_mmap_empty_topology_search_returns_empty() {
        let config = HnswConfig::new(4, DistanceMetric::Euclidean);
        let bytes = serialize_topology(None, 0, &[]);
        let topo = MmapTopology::from_bytes(Bytes::from(bytes)).expect("from_bytes");

        let mmap_index = HnswIndex::new(config);
        mmap_index.adopt_mmap_topology(topo);

        let map: HashMap<NodeId, Arc<[f32]>> = HashMap::new();
        let accessor = |id: NodeId| -> Option<Arc<[f32]>> { map.get(&id).cloned() };

        let results = mmap_index.search(&[0.1; 4], 5, &accessor);
        assert!(results.is_empty());
        assert_eq!(mmap_index.len(), 0);
        assert!(mmap_index.is_empty());
    }
}
