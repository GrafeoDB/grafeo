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

use super::compute_distance;
use super::paged_topology::{MmapTopology, NeighborsIter as MmapNeighborsIter};
use super::{TopologyVisitor, VectorAccessor};
use crate::index::vector::HnswConfig;
use grafeo_common::types::NodeId;
use grafeo_common::utils::error::Result;
use grafeo_common::utils::hash::FxHashSet;
use ordered_float::OrderedFloat;
use parking_lot::RwLock;
use rand::{RngExt, SeedableRng};
use std::collections::{BinaryHeap, HashMap, HashSet, VecDeque};
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

/// A breadth-first walk along the links of one level, from one node, that
/// looks for some nodes (see `HnswIndex::reconnect_unreached`). Reaching a
/// node known to reach them all reaches them all, so the walk stops there.
struct Walk<'a> {
    /// The nodes known to reach every node looked for.
    reaches_all: &'a FxHashSet<NodeId>,
    /// The nodes reached so far.
    visited: FxHashSet<NodeId>,
    /// The nodes looked for that the walk has not reached.
    unreached: FxHashSet<NodeId>,
    /// The reached nodes whose links the walk has yet to follow.
    queue: VecDeque<NodeId>,
    /// How many more nodes the walk may follow the links of.
    budget: usize,
}

impl<'a> Walk<'a> {
    /// A walk from `start` that looks for `targets`, following the links of
    /// at most `budget` nodes.
    fn new(
        start: NodeId,
        targets: &[NodeId],
        reaches_all: &'a FxHashSet<NodeId>,
        budget: usize,
    ) -> Self {
        let mut walk = Self {
            reaches_all,
            visited: FxHashSet::default(),
            unreached: targets.iter().copied().collect(),
            queue: VecDeque::new(),
            budget,
        };
        walk.reach(start);
        walk
    }

    /// Marks `id` reached; the walk follows its links later.
    fn reach(&mut self, id: NodeId) {
        if self.reaches_all.contains(&id) {
            self.unreached.clear();
        }
        self.unreached.remove(&id);
        if self.visited.insert(id) {
            self.queue.push_back(id);
        }
    }

    /// Walks on along the links at `level` until every target is reached,
    /// the budget is spent or no link leads further.
    fn run(&mut self, nodes_map: &HashMap<NodeId, HnswNode>, level: usize) {
        while !self.unreached.is_empty()
            && self.budget > 0
            && let Some(current) = self.queue.pop_front()
        {
            self.budget -= 1;
            let Some(node) = nodes_map.get(&current) else {
                continue;
            };
            for &next in node.neighbors.get(level).into_iter().flatten() {
                self.reach(next);
                if self.unreached.is_empty() {
                    return;
                }
            }
        }
    }
}

/// The most nodes [`HnswIndex::begin_restore`] sizes the topology map for up
/// front; a restore of more grows the map as nodes arrive.
const MAX_RESTORE_CAPACITY: usize = 1 << 16;

/// A neighbor reference that breaks the rules of a topology, as
/// [`HnswIndex::first_broken_link`] reports it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct BrokenLink {
    /// The node whose list holds the reference.
    pub node: NodeId,
    /// The level of that list.
    pub level: usize,
    /// The node listed.
    pub neighbor: NodeId,
    /// The levels the node listed has, `None` when it is not a node of the
    /// topology.
    pub neighbor_levels: Option<usize>,
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

    /// How many levels node `id` has, `None` when it is not a node.
    fn level_count(&self, id: NodeId) -> Option<usize> {
        match self {
            Self::Heap(map) => map.get(&id).map(|node| node.neighbors.len()),
            Self::Mmap(topo) => topo.contains(id).then(|| {
                let mut levels = 0;
                while topo.neighbors_at(id, levels).is_some() {
                    levels += 1;
                }
                levels
            }),
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

/// Picks a new entry point after the old one was removed: the node on the
/// highest level (lowest id on ties, for determinism), and records that level
/// as the index's top level. Search starts at the entry point's top level, so
/// a stale, higher `max_level` would leave the upper layers unreachable.
fn reset_entry_point(
    nodes_map: &HashMap<NodeId, HnswNode>,
    entry_point: &mut Option<NodeId>,
    max_level: &mut usize,
) {
    let best = nodes_map
        .iter()
        .map(|(&node_id, node)| (node.neighbors.len().saturating_sub(1), node_id))
        .max_by(|(level_a, id_a), (level_b, id_b)| level_a.cmp(level_b).then(id_b.cmp(id_a)));
    match best {
        Some((level, node_id)) => {
            *entry_point = Some(node_id);
            *max_level = level;
        }
        None => {
            *entry_point = None;
            *max_level = 0;
        }
    }
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

    /// Hands the topology to `visitor`: the header, then every node in
    /// increasing id order, under one read lock of the topology, the entry
    /// point and the level (so the visitor sees one consistent state).
    ///
    /// A heap topology lends each node's lists as they are, after sorting
    /// references to its nodes by id; an mmap-backed topology materializes
    /// one node's lists at a time, in buffers reused from node to node.
    ///
    /// The read locks are held until the visitor has taken the last node. A
    /// checkpoint writes the topology from inside the visit, so for as long
    /// as that write takes, inserts into this index wait, and so do searches
    /// that arrive after a waiting insert (`parking_lot`'s locks are fair:
    /// a waiting writer queues the readers behind it). Memory meanwhile: the
    /// checkpoint holds one piece and one node's bytes, besides the sorted
    /// references of a heap topology (16 bytes per node, O(node count)).
    ///
    /// # Errors
    ///
    /// Returns the first error `visitor` returns; no node is visited after it.
    pub fn visit_topology(&self, visitor: &mut dyn TopologyVisitor) -> Result<()> {
        let nodes = self.nodes.read();
        let entry_point = self.entry_point.read();
        let max_level = self.max_level.read();
        visitor.header(*entry_point, *max_level, nodes.len())?;
        match &*nodes {
            TopologyBackend::Heap(map) => {
                let mut sorted: Vec<(&NodeId, &HnswNode)> = map.iter().collect();
                sorted.sort_unstable_by_key(|(id, _)| **id);
                for (id, node) in sorted {
                    visitor.node(*id, &node.neighbors)?;
                }
            }
            TopologyBackend::Mmap(topo) => {
                // The page index is in id order. `layers` keeps the lists of
                // the node with the most layers so far, for the next nodes.
                let mut layers: Vec<Vec<NodeId>> = Vec::new();
                for id in topo.iter_node_ids() {
                    let mut count = 0;
                    while let Some(neighbors) = topo.neighbors_at(id, count) {
                        if count == layers.len() {
                            layers.push(Vec::new());
                        }
                        let layer = &mut layers[count];
                        layer.clear();
                        layer.extend(neighbors);
                        count += 1;
                    }
                    visitor.node(id, &layers[..count])?;
                }
            }
        }
        Ok(())
    }

    /// Replaces the topology with an empty heap map and sets the entry point
    /// and the level, for [`restore_node`](Self::restore_node) to fill in.
    ///
    /// `node_count` (the number of nodes the restore announces) only sizes
    /// the map, for at most 1 << 16 nodes up front: a count read from a file
    /// is not trusted with an allocation.
    pub fn begin_restore(&self, entry_point: Option<NodeId>, max_level: usize, node_count: usize) {
        let mut backend = self.nodes.write();
        *backend = TopologyBackend::with_capacity(node_count.min(MAX_RESTORE_CAPACITY));
        *self.entry_point.write() = entry_point;
        *self.max_level.write() = max_level;
    }

    /// Restores node `id` with its neighbor lists (layer 0 first), replacing
    /// any lists it had.
    ///
    /// # Panics
    ///
    /// Panics if the topology is mmap-backed: call
    /// [`begin_restore`](Self::begin_restore) first.
    pub fn restore_node(&self, id: NodeId, layers: Vec<Vec<NodeId>>) {
        self.nodes
            .write()
            .as_heap_mut()
            .insert(id, HnswNode { neighbors: layers });
    }

    /// The first neighbor reference, by node id and then level (and in list
    /// order within a list), that breaks the rules [`insert`](Self::insert)
    /// and [`remove`](Self::remove) keep: every node listed at a level is a
    /// node of the topology with a list at that level, and no node lists
    /// itself. `None` when every reference keeps them.
    ///
    /// Reads the topology in place under its read lock and allocates
    /// nothing: one pass over every list, and a lookup of each node listed
    /// (in an mmap topology, a search of its page index per level).
    pub(crate) fn first_broken_link(&self) -> Option<BrokenLink> {
        let nodes = self.nodes.read();
        let backend = &*nodes;
        let mut first: Option<BrokenLink> = None;
        let mut check =
            |node: NodeId, level: usize, neighbors: &mut dyn Iterator<Item = NodeId>| {
                if first.is_some_and(|found| (found.node, found.level) <= (node, level)) {
                    return;
                }
                for neighbor in neighbors {
                    let neighbor_levels = backend.level_count(neighbor);
                    if neighbor == node || neighbor_levels.is_none_or(|levels| levels <= level) {
                        first = Some(BrokenLink {
                            node,
                            level,
                            neighbor,
                            neighbor_levels,
                        });
                        return;
                    }
                }
            };
        match backend {
            TopologyBackend::Heap(map) => {
                for (&id, node) in map {
                    for (level, layer) in node.neighbors.iter().enumerate() {
                        check(id, level, &mut layer.iter().copied());
                    }
                }
            }
            TopologyBackend::Mmap(topo) => {
                for id in topo.iter_node_ids() {
                    let mut level = 0;
                    while let Some(mut neighbors) = topo.neighbors_at(id, level) {
                        check(id, level, &mut neighbors);
                        level += 1;
                    }
                }
            }
        }
        first
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

        let level = self.random_level();

        // Create the new node (topology only)
        let node = HnswNode {
            neighbors: vec![Vec::new(); level + 1],
        };

        let mut nodes = self.nodes.write();
        let mut entry_point = self.entry_point.write();
        let mut max_level = self.max_level.write();

        // Updating an existing vector (#374): take the old entry out first,
        // under the same locks, so searches never see its neighbor lists reset.
        self.take_out(
            nodes.as_heap_mut(),
            &mut entry_point,
            &mut max_level,
            id,
            accessor,
        );

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

            // Second pass: choose the pruned lists (immutable read), with the
            // same heuristic as the new node's own links (#391). Keeping the
            // nearest only dropped the links that lead away from a cluster or
            // along a line, and an insert whose links back were all dropped
            // was never found again. (Exact duplicates of more vectors than a
            // list holds still crowd a later insert out: none covers another.)
            let mut pruned: Vec<(NodeId, Vec<NodeId>)> = Vec::new();
            {
                let nodes_map = nodes.as_heap_mut();
                for neighbor_id in &needs_pruning {
                    if let Some(neighbor) = nodes_map.get(neighbor_id)
                        && neighbor.neighbors.len() > lc
                    {
                        let Some(base_vec) = self.measurable_vector(accessor, *neighbor_id) else {
                            continue;
                        };
                        let mut candidates: Vec<Neighbor> = neighbor.neighbors[lc]
                            .iter()
                            .map(|&nid| Neighbor {
                                id: nid,
                                distance: self.node_distance(accessor, &base_vec, nid),
                            })
                            .collect();
                        candidates.sort_by(|a, b| {
                            OrderedFloat(a.distance)
                                .cmp(&OrderedFloat(b.distance))
                                .then(a.id.cmp(&b.id))
                        });
                        pruned.push((
                            *neighbor_id,
                            self.select_neighbors_heuristic(accessor, &candidates, m_max),
                        ));
                    }
                }
            }

            // Third pass: apply pruning (mutable borrow).
            {
                let nodes_map = nodes.as_heap_mut();
                for (neighbor_id, kept) in pruned {
                    if let Some(neighbor) = nodes_map.get_mut(&neighbor_id)
                        && neighbor.neighbors.len() > lc
                    {
                        neighbor.neighbors[lc] = kept;
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

    /// Takes `id` out of the topology and mends the lists that led to it, so
    /// every path that went through it goes past it now: a removal (#600)
    /// and the replacement of a vector (#374) leave the other nodes
    /// reachable. Returns false (and changes nothing) when `id` is not in
    /// the index. Callers hold the topology locks.
    ///
    /// At each level of `id`, the nodes that linked to it (its in-neighbors)
    /// lose that link and gain links to its neighbors (the bypasses), the
    /// repair of the HNSW and DiskANN deletion algorithms, but without
    /// dropping any link they have: a dropped link could be the last one to
    /// some node. First every bypass that an in-neighbor cannot reach again
    /// within a bounded walk (a tight group of bypasses that link only each
    /// other, say) is linked (see
    /// [`reconnect_unreached`](Self::reconnect_unreached)); then the
    /// in-neighbors fill the room they have left with the bypasses the
    /// neighbor heuristic picks (diverse ones, as an insert links).
    ///
    /// Finding the in-neighbors takes one pass over every list (links are
    /// one-way, and no reverse index is kept), the pass that dropping the
    /// links to `id` needs anyway. Distances are read through `accessor`; the
    /// repair goes in id order, so the same index and removal give the same
    /// topology on every run.
    fn take_out(
        &self,
        nodes_map: &mut HashMap<NodeId, HnswNode>,
        entry_point: &mut Option<NodeId>,
        max_level: &mut usize,
        id: NodeId,
        accessor: &impl VectorAccessor,
    ) -> bool {
        let Some(removed) = nodes_map.remove(&id) else {
            return false;
        };

        // The removed node's neighbors that are still nodes with that level.
        let bypasses: Vec<Vec<NodeId>> = removed
            .neighbors
            .iter()
            .enumerate()
            .map(|(level, neighbors)| {
                let mut kept: Vec<NodeId> = Vec::with_capacity(neighbors.len());
                for &neighbor in neighbors {
                    let has_level = nodes_map
                        .get(&neighbor)
                        .is_some_and(|node| node.neighbors.len() > level);
                    if has_level && !kept.contains(&neighbor) {
                        kept.push(neighbor);
                    }
                }
                kept
            })
            .collect();

        // Drop every link to `id`, noting who had one.
        let mut in_neighbors: Vec<Vec<NodeId>> = vec![Vec::new(); removed.neighbors.len()];
        for (&node_id, node) in nodes_map.iter_mut() {
            for (level, neighbors) in node.neighbors.iter_mut().enumerate() {
                let before = neighbors.len();
                neighbors.retain(|&neighbor| neighbor != id);
                if neighbors.len() != before
                    && let Some(listed_by) = in_neighbors.get_mut(level)
                {
                    listed_by.push(node_id);
                }
            }
        }

        if *entry_point == Some(id) {
            reset_entry_point(nodes_map, entry_point, max_level);
        }

        for (level, listed_by) in in_neighbors.iter_mut().enumerate() {
            listed_by.sort_unstable();
            // A level nothing linked the removed node at was reached from it
            // as the entry point: the new one reaches its bypasses instead.
            let entry_source: Vec<NodeId> = entry_point
                .filter(|entry| {
                    listed_by.is_empty()
                        && nodes_map
                            .get(entry)
                            .is_some_and(|node| node.neighbors.len() > level)
                })
                .into_iter()
                .collect();
            let sources = if entry_source.is_empty() {
                &listed_by[..]
            } else {
                &entry_source[..]
            };
            self.reconnect_unreached(nodes_map, level, &bypasses[level], sources, accessor);
            self.link_past(nodes_map, level, &bypasses[level], listed_by, accessor);
        }
        true
    }

    /// The most neighbors a node keeps at `level`.
    fn max_neighbors(&self, level: usize) -> usize {
        if level == 0 {
            self.config.m_max
        } else {
            self.config.m
        }
    }

    /// The vector of `id` when the index can measure it (it has the
    /// configured size).
    fn measurable_vector(&self, accessor: &impl VectorAccessor, id: NodeId) -> Option<Arc<[f32]>> {
        accessor
            .get_vector(id)
            .filter(|vector| vector.len() == self.config.dimensions)
    }

    /// Makes every bypass at `level` reachable from every in-neighbor of the
    /// removed node (see [`take_out`](Self::take_out)), so that each path
    /// that went from an in-neighbor through the removed node to a bypass
    /// still goes from the one to the other: then every node a search
    /// reached before the removal, it reaches after.
    ///
    /// A walk along the links from each in-neighbor looks for the bypasses;
    /// it stops early at an in-neighbor already known to reach them all.
    /// Each bypass it misses is linked from the in-neighbor when that has
    /// room, else from the nearest bypass or in-neighbor the walk reached
    /// that has room, else from the in-neighbor all the same (whose list
    /// then holds one more than the most, until an insert prunes it); the
    /// walk goes on from there. A walk visits at most `8 * m * m` nodes for
    /// a level with at most `m` links per node, so a bypass further away is
    /// linked although it may be reachable, which costs a link, never a
    /// node.
    fn reconnect_unreached(
        &self,
        nodes_map: &mut HashMap<NodeId, HnswNode>,
        level: usize,
        bypasses: &[NodeId],
        sources: &[NodeId],
        accessor: &impl VectorAccessor,
    ) {
        let max_neighbors = self.max_neighbors(level);
        let mut reaches_all: FxHashSet<NodeId> = FxHashSet::default();
        for &source in sources {
            let mut walk = Walk::new(
                source,
                bypasses,
                &reaches_all,
                8 * max_neighbors * max_neighbors,
            );
            walk.run(nodes_map, level);
            for &bypass in bypasses {
                if !walk.unreached.contains(&bypass) {
                    continue;
                }
                let has_room = |host: &NodeId| {
                    nodes_map
                        .get(host)
                        .is_some_and(|node| node.neighbors[level].len() < max_neighbors)
                };
                let host = if has_room(&source) {
                    Some(source)
                } else {
                    self.measurable_vector(accessor, bypass).and_then(|vector| {
                        bypasses
                            .iter()
                            .chain(sources)
                            .copied()
                            .filter(|&host| {
                                host != bypass && walk.visited.contains(&host) && has_room(&host)
                            })
                            .map(|host| {
                                let distance = self.node_distance(accessor, &vector, host);
                                (OrderedFloat(distance), host)
                            })
                            .min()
                            .map(|(_, host)| host)
                    })
                }
                .unwrap_or(source);
                if let Some(node) = nodes_map.get_mut(&host) {
                    node.neighbors[level].push(bypass);
                }
                walk.reach(bypass);
                walk.run(nodes_map, level);
            }
            reaches_all.insert(source);
        }
    }

    /// Fills the room each in-neighbor of the removed node has at `level`
    /// with the bypasses the neighbor heuristic picks, measured from it and
    /// against the neighbors it has (see [`take_out`](Self::take_out)); one
    /// that links to no bypass gets the nearest one, so every path through
    /// the removed node goes on past it.
    fn link_past(
        &self,
        nodes_map: &mut HashMap<NodeId, HnswNode>,
        level: usize,
        bypasses: &[NodeId],
        in_neighbors: &[NodeId],
        accessor: &impl VectorAccessor,
    ) {
        let max_neighbors = self.max_neighbors(level);
        for &node_id in in_neighbors {
            let Some(current) = nodes_map.get(&node_id).map(|node| &node.neighbors[level]) else {
                continue;
            };
            let room = max_neighbors.saturating_sub(current.len());
            if room == 0 {
                continue;
            }
            let Some(base_vector) = self.measurable_vector(accessor, node_id) else {
                continue;
            };
            let links_past = current.iter().any(|neighbor| bypasses.contains(neighbor));
            let mut candidates: Vec<Neighbor> = bypasses
                .iter()
                .copied()
                .filter(|bypass| *bypass != node_id && !current.contains(bypass))
                .map(|bypass| Neighbor {
                    id: bypass,
                    distance: self.node_distance(accessor, &base_vector, bypass),
                })
                .collect();
            candidates.sort_by(|a, b| {
                OrderedFloat(a.distance)
                    .cmp(&OrderedFloat(b.distance))
                    .then(a.id.cmp(&b.id))
            });
            let mut chosen: Vec<Arc<[f32]>> = current
                .iter()
                .filter_map(|&neighbor| self.measurable_vector(accessor, neighbor))
                .collect();
            let mut picks = self.pick_diverse(accessor, &mut chosen, &candidates, room);
            if picks.is_empty()
                && !links_past
                && let Some(nearest) = candidates.first()
            {
                picks.push(nearest.id);
            }
            if let Some(node) = nodes_map.get_mut(&node_id) {
                node.neighbors[level].extend(picks);
            }
        }
    }

    /// Searches for the k nearest neighbors to the query vector.
    ///
    /// Returns a vector of (NodeId, distance) pairs sorted by distance
    /// (closest first). A query with another number of values than the
    /// configured `dimensions` cannot be measured against any vector and
    /// finds nothing: check it first with
    /// [`check_query_vector`](super::check_query_vector), which also refuses
    /// NaN and infinite values.
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
    /// Higher ef values give better recall at the cost of latency. A query
    /// of another size than the configured `dimensions` finds nothing (see
    /// [`search`](Self::search)).
    #[must_use]
    pub fn search_with_ef(
        &self,
        query: &[f32],
        k: usize,
        ef: usize,
        accessor: &impl VectorAccessor,
    ) -> Vec<(NodeId, f32)> {
        if query.len() != self.config.dimensions {
            return Vec::new();
        }

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

        // Return top k: the results hold only nodes with a vector (#594)
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
    /// Returns an empty vector if the allowlist is empty, or if the query
    /// has another size than the configured `dimensions` (see
    /// [`search`](Self::search)).
    #[must_use]
    pub fn search_with_ef_and_filter(
        &self,
        query: &[f32],
        k: usize,
        ef: usize,
        allowlist: &HashSet<NodeId>,
        accessor: &impl VectorAccessor,
    ) -> Vec<(NodeId, f32)> {
        if allowlist.is_empty() || query.len() != self.config.dimensions {
            return Vec::new();
        }

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

        // Return top k: the results hold only nodes with a vector (#594)
        candidates
            .into_iter()
            .take(k)
            .map(|n| (n.id, n.distance))
            .collect()
    }

    /// Removes a vector from the index.
    ///
    /// The nodes that linked to it choose their links anew from their other
    /// neighbors and its neighbors, measured with the vectors `accessor`
    /// reads, so every remaining vector stays reachable (#600). This takes
    /// one pass over every neighbor list.
    ///
    /// Returns true if the vector was found and removed.
    ///
    /// # Panics
    ///
    /// Panics if the topology is mmap-backed (see
    /// [`adopt_mmap_topology`](Self::adopt_mmap_topology)).
    pub fn remove(&self, id: NodeId, accessor: &impl VectorAccessor) -> bool {
        let mut nodes = self.nodes.write();
        let mut entry_point = self.entry_point.write();
        let mut max_level = self.max_level.write();
        self.take_out(
            nodes.as_heap_mut(),
            &mut entry_point,
            &mut max_level,
            id,
            accessor,
        )
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
        // A node without a vector is explored (its links lead on) but never
        // a result: the search reads no result twice to tell (#594).
        let ep_dist = self.distance_to(accessor, query, ep);

        // Min-heap of candidates to explore
        let mut candidates: BinaryHeap<Neighbor> = BinaryHeap::new();
        candidates.push(Neighbor {
            id: ep,
            distance: ep_dist.unwrap_or(f32::MAX),
        });

        // Max-heap of current best (furthest = top)
        let mut results: BinaryHeap<FurthestCandidate> = BinaryHeap::new();
        if let Some(distance) = ep_dist {
            results.push(FurthestCandidate { id: ep, distance });
        }

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

                    let Some(dist) = self.distance_to(accessor, query, neighbor) else {
                        // No vector: explored while there is room, as the
                        // furthest of all, never a result.
                        if results.len() < ef {
                            candidates.push(Neighbor {
                                id: neighbor,
                                distance: f32::MAX,
                            });
                        }
                        continue;
                    };

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
        // A node without a vector is explored but never a result (#594).
        let ep_vector = self.distance_to(accessor, query, ep);
        let ep_dist = ep_vector.unwrap_or(f32::MAX);

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
        if ep_vector.is_some() && allowlist.contains(&ep) {
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

                    let vector = self.distance_to(accessor, query, neighbor);
                    let dist = vector.unwrap_or(f32::MAX);

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

                    // Only add to results if in allowlist, with a vector
                    if vector.is_some() && allowlist.contains(&neighbor) {
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
    /// each candidate is "covered" by an already-selected neighbor: one that
    /// is closer to it, by a factor of `alpha`, than the node being linked
    /// (`alpha * distance(candidate, selected) < distance(candidate, node)`,
    /// the robust pruning of Vamana; `alpha = 1` is the HNSW heuristic, and a
    /// larger `alpha` covers fewer candidates, so it keeps more long links).
    /// This preserves graph navigability by ensuring neighbors point to
    /// diverse regions of the space.
    fn select_neighbors_heuristic(
        &self,
        accessor: &impl VectorAccessor,
        candidates: &[Neighbor],
        m: usize,
    ) -> Vec<NodeId> {
        self.pick_diverse(accessor, &mut Vec::with_capacity(m), candidates, m)
    }

    /// Picks up to `room` of `candidates` (sorted nearest first, with their
    /// distances to the node that links to them) that no vector in `chosen`
    /// covers, in the sense of
    /// [`select_neighbors_heuristic`](Self::select_neighbors_heuristic), and
    /// adds the vector of each pick to `chosen`. A candidate without a
    /// vector the index can measure is never picked.
    fn pick_diverse(
        &self,
        accessor: &impl VectorAccessor,
        chosen: &mut Vec<Arc<[f32]>>,
        candidates: &[Neighbor],
        room: usize,
    ) -> Vec<NodeId> {
        let alpha = self.config.alpha;
        let mut picks: Vec<NodeId> = Vec::with_capacity(room.min(candidates.len()));
        for candidate in candidates {
            if picks.len() >= room {
                break;
            }
            let Some(cv) = self.measurable_vector(accessor, candidate.id) else {
                continue;
            };
            let covered = chosen
                .iter()
                .any(|sv| alpha * self.vector_distance(&cv, sv) < candidate.distance);
            if !covered {
                chosen.push(cv);
                picks.push(candidate.id);
            }
        }
        picks
    }

    /// Computes distance between two raw vectors using the configured metric.
    #[inline]
    fn vector_distance(&self, a: &[f32], b: &[f32]) -> f32 {
        compute_distance(a, b, self.config.metric)
    }

    /// Computes the distance between a query vector and a stored node, reading
    /// the stored vector in place ([`VectorAccessor::with_vector`]).
    fn node_distance(&self, accessor: &impl VectorAccessor, query: &[f32], id: NodeId) -> f32 {
        self.distance_to(accessor, query, id).unwrap_or(f32::MAX)
    }

    /// The distance from `query` to the vector of `id`, or `None` when the
    /// node has no vector (a topology saved while its values could not be
    /// read can still name one, #594): one read tells both. A vector of
    /// another size than `query` cannot be measured and counts as none.
    fn distance_to(
        &self,
        accessor: &impl VectorAccessor,
        query: &[f32],
        id: NodeId,
    ) -> Option<f32> {
        let mut distance = None;
        accessor.with_vector(id, &mut |vector| {
            if vector.len() == query.len() {
                distance = Some(self.vector_distance(query, vector));
            }
        });
        distance
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
    /// (NodeId, distance) pairs sorted by distance; a query of another size
    /// than the configured `dimensions` finds nothing (see
    /// [`search`](Self::search)).
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

    /// A search reads each vector it returns once: the read that gives a
    /// candidate its distance also tells whether it has a vector, so no
    /// result is read again (a spilled vector is a read of its file, #594).
    #[test]
    fn a_search_reads_each_returned_vector_once() {
        let mut config = HnswConfig::new(2, DistanceMetric::Euclidean);
        // Every node on layer 0, where a search reads each node once.
        config.ml = 0.0;
        let index = HnswIndex::with_seed(config, 3);
        let map: HashMap<NodeId, Arc<[f32]>> = (1..=19_u64)
            .map(|i| (NodeId::new(i), vec![i as f32, 3.0].into()))
            .collect();
        let reads = std::sync::Mutex::new(HashMap::<NodeId, usize>::new());
        let counting = |id: NodeId| -> Option<Arc<[f32]>> {
            *reads.lock().unwrap().entry(id).or_default() += 1;
            map.get(&id).cloned()
        };
        for (id, vector) in &map {
            index.insert(*id, vector, &counting);
        }
        reads.lock().unwrap().clear();

        let hits = index.search(&[3.0, 3.0], 3, &counting);
        assert_eq!(hits.len(), 3);
        let reads = reads.lock().unwrap();
        for (id, _) in &hits {
            assert_eq!(reads[id], 1, "{id:?} was read more than once");
        }
    }

    /// A node the topology names but whose vector is gone (a topology saved
    /// while its values could not be read) is never a result, in a plain or
    /// a filtered search, and takes no result's place (#594).
    #[test]
    fn a_node_without_a_vector_is_never_a_result() {
        let mut config = HnswConfig::new(2, DistanceMetric::Euclidean);
        config.ml = 0.0;
        let index = HnswIndex::with_seed(config, 19);
        let mut map: HashMap<NodeId, Arc<[f32]>> = (1..=19_u64)
            .map(|i| (NodeId::new(i), vec![i as f32, 3.0].into()))
            .collect();
        {
            let accessor = make_accessor(&map);
            for (id, vector) in &map {
                index.insert(*id, vector, &accessor);
            }
        }
        map.remove(&NodeId::new(3));
        let accessor = make_accessor(&map);

        let hits = index.search(&[3.0, 3.0], 3, &accessor);
        let ids: Vec<u64> = hits.iter().map(|(id, _)| id.as_u64()).collect();
        assert_eq!(ids.len(), 3, "{ids:?}");
        assert!(!ids.contains(&3), "{ids:?}");

        let allowlist: HashSet<NodeId> = [2, 3, 4].into_iter().map(NodeId::new).collect();
        let mut filtered: Vec<u64> = index
            .search_with_filter(&[3.0, 3.0], 3, &allowlist, &accessor)
            .iter()
            .map(|(id, _)| id.as_u64())
            .collect();
        filtered.sort_unstable();
        assert_eq!(filtered, vec![2, 4]);
    }

    /// A node the topology names whose vector now has another size than the
    /// index (written without the index's upkeep) cannot be measured: it is
    /// never a result, a search past it does not panic, and an insert or a
    /// removal next to it does not either (#593).
    #[test]
    fn a_node_whose_vector_has_another_size_is_never_a_result() {
        let mut config = HnswConfig::new(2, DistanceMetric::Euclidean);
        config.ml = 0.0;
        let index = HnswIndex::with_seed(config, 88);
        let mut map: HashMap<NodeId, Arc<[f32]>> = (1..=19_u64)
            .map(|i| (NodeId::new(i), vec![i as f32, 3.0].into()))
            .collect();
        {
            let accessor = make_accessor(&map);
            for id in 1..=19_u64 {
                let vector = map[&NodeId::new(id)].clone();
                index.insert(NodeId::new(id), &vector, &accessor);
            }
        }
        map.insert(NodeId::new(3), vec![3.0, 3.0, 19.0].into());
        let ids: Vec<u64> = index
            .search(&[3.0, 3.0], 3, &make_accessor(&map))
            .iter()
            .map(|(id, _)| id.as_u64())
            .collect();
        assert_eq!(ids.len(), 3, "{ids:?}");
        assert!(!ids.contains(&3), "{ids:?}");

        map.insert(NodeId::new(88), vec![3.5, 3.0].into());
        let accessor = make_accessor(&map);
        index.insert(NodeId::new(88), &[3.5, 3.0], &accessor);
        assert!(index.remove(NodeId::new(4), &accessor));
        assert_eq!(
            index.search(&[3.5, 3.0], 1, &accessor)[0].0,
            NodeId::new(88)
        );
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
                .is_empty(),
            "expected no results"
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

        assert!(index.remove(NodeId::new(1), &accessor));
        assert_eq!(index.len(), 1);
        assert!(!index.contains(NodeId::new(1)));
        assert!(index.contains(NodeId::new(2)));

        // Removing again returns false
        assert!(!index.remove(NodeId::new(1), &accessor));
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

    /// #593: a query of another size than the index's cannot be measured
    /// against any vector: every search finds nothing, and none panics (a
    /// panic is a crash of the database). Callers report the error first
    /// (`check_query_vector`).
    #[test]
    fn a_query_of_another_size_finds_nothing_without_a_panic() {
        let config = HnswConfig::new(4, DistanceMetric::Euclidean);
        let index = HnswIndex::with_seed(config, 3);
        let map: HashMap<NodeId, Arc<[f32]>> = (1..=19u64)
            .map(|i| (NodeId::new(i), vec![i as f32, 3.0, 19.0, 88.0].into()))
            .collect();
        let accessor = make_accessor(&map);
        for id in 1..=19u64 {
            let vector = map[&NodeId::new(id)].clone();
            index.insert(NodeId::new(id), &vector, &accessor);
        }
        let allowlist: HashSet<NodeId> = map.keys().copied().collect();
        for query in [
            vec![3.0, 19.0, 88.0],
            vec![3.0, 19.0, 88.0, 3.0, 19.0],
            vec![],
        ] {
            assert!(index.search(&query, 3, &accessor).is_empty(), "{query:?}");
            assert!(
                index.search_with_ef(&query, 3, 64, &accessor).is_empty(),
                "{query:?}"
            );
            assert!(
                index
                    .search_with_filter(&query, 3, &allowlist, &accessor)
                    .is_empty(),
                "{query:?}"
            );
            assert!(
                index
                    .batch_search(std::slice::from_ref(&query), 3, &accessor)
                    .iter()
                    .all(Vec::is_empty),
                "{query:?}"
            );
        }
        // The index still answers a query of its size.
        assert_eq!(
            index.search(&[3.0, 3.0, 19.0, 88.0], 1, &accessor)[0].0,
            NodeId::new(3)
        );
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
        assert!(results[0].is_empty(), "expected empty");
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
        assert!(results.is_empty(), "{results:?}");
    }

    /// Builds a seeded 2-D index where every node is inserted once.
    fn seeded_index(
        seed: u64,
        points: &[(u64, [f32; 2])],
    ) -> (HnswIndex, HashMap<NodeId, Arc<[f32]>>) {
        let config = HnswConfig::new(2, DistanceMetric::Euclidean).with_m(4);
        let index = HnswIndex::with_seed(config, seed);
        let mut map: HashMap<NodeId, Arc<[f32]>> = HashMap::new();
        for &(id, v) in points {
            map.insert(NodeId::new(id), Arc::from(v.as_slice()));
        }
        for &(id, v) in points {
            index.insert(NodeId::new(id), &v, &make_accessor(&map));
        }
        (index, map)
    }

    /// Two tight clusters joined only through a few bridging points: a small
    /// graph where losing one node's links disconnects the rest.
    fn clustered_points() -> Vec<(u64, [f32; 2])> {
        let mut points = Vec::new();
        for i in 0..12u64 {
            let offset = i as f32 * 0.01;
            points.push((i, [offset, 0.0]));
            points.push((100 + i, [10.0 + offset, 0.0]));
        }
        for i in 0..4u64 {
            points.push((200 + i, [2.5 * (i as f32 + 1.0), 0.0]));
        }
        points
    }

    fn reachable_ids(index: &HnswIndex, map: &HashMap<NodeId, Arc<[f32]>>) -> Vec<NodeId> {
        let accessor = make_accessor(map);
        let mut ids: Vec<NodeId> = index
            .search_with_ef(&[5.0, 0.0], map.len(), 512, &accessor)
            .into_iter()
            .map(|(id, _)| id)
            .collect();
        ids.sort_unstable();
        ids
    }

    #[test]
    fn test_replacing_vectors_keeps_every_node_reachable() {
        // #374: re-inserting an indexed node used to reset its neighbor lists
        // and leave other nodes unreachable. Checked over many seeds.
        let points = clustered_points();
        let mut expected: Vec<NodeId> = points.iter().map(|&(id, _)| NodeId::new(id)).collect();
        expected.sort_unstable();
        for seed in 0..40 {
            let (index, map) = seeded_index(seed, &points);
            let accessor = make_accessor(&map);
            // Replace every bridge (same vector) several times, like repeated
            // set_node_property calls.
            for _ in 0..3 {
                for i in 0..4u64 {
                    let id = NodeId::new(200 + i);
                    let v = map[&id].clone();
                    index.insert(id, &v, &accessor);
                }
            }
            assert_eq!(index.len(), points.len(), "seed {seed}: node count changed");
            assert_eq!(reachable_ids(&index, &map), expected, "seed {seed}");
        }
    }

    #[test]
    fn test_replacing_vector_with_new_value_moves_the_node() {
        let points = clustered_points();
        let (index, mut map) = seeded_index(7, &points);
        let moved = NodeId::new(3);
        let new_vector: Arc<[f32]> = Arc::from([10.05f32, 0.0].as_slice());
        map.insert(moved, new_vector.clone());
        index.insert(moved, &new_vector, &make_accessor(&map));

        let accessor = make_accessor(&map);
        let nearest = index.search_with_ef(&[10.05, 0.0], 1, 64, &accessor);
        assert_eq!(nearest[0].0, moved, "the node is found at its new position");
        assert_eq!(index.len(), points.len());
    }

    /// The top level of the current entry point, which must equal `max_level`.
    fn entry_point_level(index: &HnswIndex) -> Option<usize> {
        let entry = (*index.entry_point.read())?;
        let nodes = index.nodes.read();
        match &*nodes {
            TopologyBackend::Heap(map) => map.get(&entry).map(|node| node.neighbors.len() - 1),
            TopologyBackend::Mmap(_) => None,
        }
    }

    /// Highest level of any node in the index.
    fn highest_level(index: &HnswIndex) -> usize {
        match &*index.nodes.read() {
            TopologyBackend::Heap(map) => map
                .values()
                .map(|node| node.neighbors.len() - 1)
                .max()
                .unwrap_or(0),
            TopologyBackend::Mmap(_) => 0,
        }
    }

    #[test]
    fn test_removing_entry_point_resets_top_level() {
        let points = clustered_points();
        for seed in 0..40 {
            let (index, map) = seeded_index(seed, &points);
            let entry = (*index.entry_point.read()).expect("entry point");
            assert!(index.remove(entry, &make_accessor(&map)));
            assert_eq!(
                *index.max_level.read(),
                entry_point_level(&index).expect("new entry point"),
                "seed {seed}: max_level must match the new entry point's level"
            );
            assert_eq!(
                *index.max_level.read(),
                highest_level(&index),
                "seed {seed}"
            );
        }
    }

    #[test]
    fn test_replacing_entry_point_resets_top_level() {
        let points = clustered_points();
        for seed in 0..40 {
            let (index, map) = seeded_index(seed, &points);
            let entry = (*index.entry_point.read()).expect("entry point");
            let v = map[&entry].clone();
            index.insert(entry, &v, &make_accessor(&map));
            assert_eq!(
                *index.max_level.read(),
                entry_point_level(&index).expect("entry point"),
                "seed {seed}"
            );
            assert_eq!(
                *index.max_level.read(),
                highest_level(&index),
                "seed {seed}"
            );
        }
    }

    #[test]
    fn test_removing_last_node_empties_index() {
        let (index, map) = seeded_index(1, &[(1, [0.0, 0.0])]);
        assert!(index.remove(NodeId::new(1), &make_accessor(&map)));
        assert!(index.entry_point.read().is_none());
        assert_eq!(*index.max_level.read(), 0);
        assert!(
            index
                .search(&[0.0, 0.0], 1, &make_accessor(&map))
                .is_empty(),
            "expected no results"
        );
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

        index.remove(NodeId::new(1), &accessor);
        let results = index.search(&[0.0, 1.0, 0.0], 3, &accessor);
        // Removed node should not appear
        assert!(results.iter().all(|(id, _)| *id != NodeId::new(1)));
        assert_eq!(results.len(), 2);
    }

    /// The next number of a deterministic generator (a linear congruential
    /// one), in `[0, 1)`: test data that is the same on every run.
    fn next_unit(state: &mut u64) -> f32 {
        *state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        // The top 24 bits, exact in an f32.
        (*state >> 40) as f32 / (1u64 << 24) as f32
    }

    /// The next number of the generator of [`next_unit`], below `bound`.
    fn next_below(state: &mut u64, bound: usize) -> usize {
        next_unit(state);
        let bound = u64::try_from(bound).expect("a test bound fits in u64");
        usize::try_from((*state >> 33) % bound).expect("below a usize bound")
    }

    /// The nodes of `map` that a search from `query` cannot find although
    /// it may visit the whole graph (`k` and `ef` above the index size):
    /// the nodes no path from the entry point reaches.
    fn unreachable_from(
        index: &HnswIndex,
        map: &HashMap<NodeId, Arc<[f32]>>,
        query: &[f32],
    ) -> Vec<NodeId> {
        let size = index.len();
        let found: HashSet<NodeId> = index
            .search_with_ef(query, size, size * 2 + 1, &make_accessor(map))
            .into_iter()
            .map(|(id, _)| id)
            .collect();
        let mut missing: Vec<NodeId> = map
            .keys()
            .copied()
            .filter(|id| !found.contains(id))
            .collect();
        missing.sort_unstable();
        missing
    }

    /// `count` vectors of `dimensions` values drawn uniformly from `[-1, 1)`
    /// with the generator seeded by `seed`.
    fn uniform_vectors(count: usize, dimensions: usize, seed: u64) -> Vec<Vec<f32>> {
        let mut state = seed;
        (0..count)
            .map(|_| {
                (0..dimensions)
                    .map(|_| next_unit(&mut state) * 2.0 - 1.0)
                    .collect()
            })
            .collect()
    }

    /// The mean share of the true `k` nearest neighbors (by brute force)
    /// that a search with beam width `ef` finds, over `queries`.
    fn recall_at(
        index: &HnswIndex,
        map: &HashMap<NodeId, Arc<[f32]>>,
        queries: &[Vec<f32>],
        k: usize,
        ef: usize,
    ) -> f64 {
        let metric = index.config().metric;
        let accessor = make_accessor(map);
        let mut found = 0usize;
        for query in queries {
            let mut exact: Vec<(NodeId, f32)> = map
                .iter()
                .map(|(id, vector)| {
                    (
                        *id,
                        crate::index::vector::compute_distance(query, vector, metric),
                    )
                })
                .collect();
            exact.sort_by(|a, b| {
                OrderedFloat(a.1)
                    .cmp(&OrderedFloat(b.1))
                    .then(a.0.cmp(&b.0))
            });
            let truth: HashSet<NodeId> = exact.iter().take(k).map(|(id, _)| *id).collect();
            found += index
                .search_with_ef(query, k, ef, &accessor)
                .iter()
                .filter(|(id, _)| truth.contains(id))
                .count();
        }
        found as f64 / (queries.len() * k) as f64
    }

    /// Builds a seeded index over `vectors` (node ids from 0, in order).
    fn build_index(
        config: HnswConfig,
        seed: u64,
        vectors: &[Vec<f32>],
    ) -> (HnswIndex, HashMap<NodeId, Arc<[f32]>>) {
        let index = HnswIndex::with_seed(config, seed);
        let map: HashMap<NodeId, Arc<[f32]>> = vectors
            .iter()
            .enumerate()
            .map(|(i, vector)| (NodeId::new(i as u64), vector.as_slice().into()))
            .collect();
        {
            let accessor = make_accessor(&map);
            for (i, vector) in vectors.iter().enumerate() {
                index.insert(NodeId::new(i as u64), vector, &accessor);
            }
        }
        (index, map)
    }

    const MIXED_DIMENSIONS: usize = 8;

    /// 190 vectors of 8 values from the generator seeded by `seed`: four
    /// tight clusters (whose members link mostly to each other), a line
    /// (which the heuristic links as a chain) and vectors spread out.
    fn mixed_vectors(seed: u64) -> HashMap<NodeId, Arc<[f32]>> {
        let mut state = seed.wrapping_add(88);
        (0..190u64)
            .map(|id| {
                let cluster = usize::try_from((id / 3) % 4).expect("below 4");
                let vector: Vec<f32> = match id % 3 {
                    0 => (0..MIXED_DIMENSIONS)
                        .map(|d| {
                            let center = if d == cluster { 19.0 } else { 0.0 };
                            center + next_unit(&mut state) * 0.05
                        })
                        .collect(),
                    1 => (0..MIXED_DIMENSIONS)
                        .map(|d| if d == 0 { id as f32 * 0.3 } else { 3.0 })
                        .collect(),
                    _ => (0..MIXED_DIMENSIONS)
                        .map(|_| next_unit(&mut state) * 19.0)
                        .collect(),
                };
                (NodeId::new(id), vector.into())
            })
            .collect()
    }

    /// An index over [`mixed_vectors`] with a small `m` (4, so 8 links at
    /// level 0), inserted in id order, and three queries from its corners.
    fn mixed_index(seed: u64) -> (HnswIndex, HashMap<NodeId, Arc<[f32]>>, Vec<Vec<f32>>) {
        let map = mixed_vectors(seed);
        let config = HnswConfig::new(MIXED_DIMENSIONS, DistanceMetric::Euclidean).with_m(4);
        let index = HnswIndex::with_seed(config, seed);
        let mut ids: Vec<NodeId> = map.keys().copied().collect();
        ids.sort_unstable();
        for id in &ids {
            let vector = map[id].clone();
            index.insert(*id, &vector, &make_accessor(&map));
        }
        let queries = vec![
            vec![0.0; MIXED_DIMENSIONS],
            vec![19.0; MIXED_DIMENSIONS],
            map[&NodeId::new(88)].to_vec(),
        ];
        (index, map, queries)
    }

    /// #391: inserts keep every vector reachable. Pruning a full list back
    /// to its nearest links dropped the only link to some inserts (a node
    /// past the end of a cluster loses its link back to the cluster), which
    /// a search then never found; the neighbor heuristic keeps the links
    /// that lead elsewhere.
    #[test]
    fn inserts_keep_every_vector_reachable() {
        for seed in 0..19u64 {
            let (index, map, queries) = mixed_index(seed);
            for query in &queries {
                assert_eq!(
                    unreachable_from(&index, &map, query),
                    Vec::<NodeId>::new(),
                    "seed {seed}, query {query:?}"
                );
            }
        }
    }

    /// A larger `alpha` covers fewer candidates, so the heuristic keeps more
    /// (longer) links, as the configuration documents: it used to cover
    /// more, keeping fewer.
    #[test]
    fn a_larger_alpha_keeps_more_links() {
        let vectors = uniform_vectors(300, 8, 19);
        let links = |alpha: f32| {
            let config = HnswConfig::new(8, DistanceMetric::Euclidean)
                .with_m(8)
                .with_alpha(alpha);
            let (index, _) = build_index(config, 3, &vectors);
            let (_, _, nodes) = index.snapshot_topology();
            nodes
                .iter()
                .map(|(_, levels)| levels[0].len())
                .sum::<usize>()
        };
        let standard = links(1.0);
        let relaxed = links(1.3);
        assert!(
            relaxed > standard,
            "alpha 1.3 keeps {relaxed} links at level 0, alpha 1.0 keeps {standard}"
        );
    }

    /// #391: the recall the neighbor heuristic gives on uniform vectors (the
    /// issue's shape, smaller): `m = 16` finds at least 95 in 100 of the 10
    /// nearest neighbors at `ef = 64` (98.9 now). On the issue's full shape
    /// (10,000 vectors of 128 values) recall is lower, as in FAISS's HNSW
    /// with the same parameters: that data is hard for any HNSW.
    #[test]
    fn recall_at_ten_reaches_95_percent_at_m_16() {
        let vectors = uniform_vectors(2_000, 64, 19);
        let queries = uniform_vectors(100, 64, 88);
        let config = HnswConfig::new(64, DistanceMetric::Cosine).with_m(16);
        let (index, map) = build_index(config, 3, &vectors);
        let recall = recall_at(&index, &map, &queries, 10, 64);
        assert!(recall >= 0.95, "recall@10 at ef=64 is {recall:.3}");
    }

    /// #600: the vectors of the issue, on a line. The heuristic links each
    /// node to its neighbors on the line only, so the line is a chain, and
    /// removing one of its links cut every node behind it off.
    #[test]
    fn removing_a_vector_keeps_the_vectors_behind_it_reachable() {
        for seed in 0..19 {
            let config = HnswConfig::new(4, DistanceMetric::Euclidean);
            let index = HnswIndex::with_seed(config, seed);
            let mut map: HashMap<NodeId, Arc<[f32]>> = (0..8u64)
                .map(|i| (NodeId::new(i), vec![1.0, i as f32 / 100.0, 0.0, 0.0].into()))
                .collect();
            for id in 0..8u64 {
                let vector = map[&NodeId::new(id)].clone();
                index.insert(NodeId::new(id), &vector, &make_accessor(&map));
            }

            // As a property removal does: the vector is gone first.
            map.remove(&NodeId::new(3));
            assert!(index.remove(NodeId::new(3), &make_accessor(&map)));

            for query in [[1.0, 0.0, 0.0, 0.0], [1.0, 0.07, 0.0, 0.0]] {
                assert_eq!(
                    unreachable_from(&index, &map, &query),
                    Vec::<NodeId>::new(),
                    "seed {seed}, query {query:?}: every remaining vector is found"
                );
            }
        }
    }

    /// #600: removing many vectors, one at a time and in a scrambled order,
    /// leaves every remaining one reachable after each removal: in tight
    /// clusters, along a line and spread out, with a small `m` so that each
    /// node has few links to lose.
    #[test]
    fn removing_many_vectors_keeps_every_remaining_vector_reachable() {
        for seed in 0..19u64 {
            let (index, mut map, queries) = mixed_index(seed);
            let mut state = seed.wrapping_add(3);
            let mut ids: Vec<NodeId> = map.keys().copied().collect();
            ids.sort_unstable();

            // Remove 3 of every 4 nodes, in a scrambled order.
            for step in 0..ids.len() {
                let pick = next_below(&mut state, ids.len());
                ids.swap(step, pick);
            }
            for &removed in ids.iter().take(ids.len() * 3 / 4) {
                map.remove(&removed);
                assert!(
                    index.remove(removed, &make_accessor(&map)),
                    "seed {seed}: {removed:?} was indexed"
                );
                for query in &queries {
                    assert_eq!(
                        unreachable_from(&index, &map, query),
                        Vec::<NodeId>::new(),
                        "seed {seed}: after removing {removed:?}"
                    );
                }
            }
            assert_eq!(index.len(), map.len());
        }
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
        assert!(!results.is_empty(), "results is empty");
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
        assert!(results.is_empty(), "{results:?}");
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
        assert!(!results.is_empty(), "results is empty");
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
        assert!(!heap_results.is_empty(), "heap_results is empty");

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
        let accessor = |_: NodeId| -> Option<Arc<[f32]>> { None };
        index.remove(NodeId::new(1), &accessor);
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
        assert!(results.is_empty(), "{results:?}");
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
        assert!(results.is_empty(), "{results:?}");
        assert_eq!(mmap_index.len(), 0);
        assert!(mmap_index.is_empty());
    }

    // ── Streaming the topology one node at a time ──────────────────

    use crate::index::vector::TopologyVisitor;
    use grafeo_common::utils::error::{Error, Result};

    /// Everything a visit hands over, in order.
    #[derive(Default)]
    struct Recorded {
        headers: Vec<(Option<NodeId>, usize, usize)>,
        nodes: Vec<(NodeId, Vec<Vec<NodeId>>)>,
        /// Refuse the node at this position of the visit.
        refuse_at: Option<usize>,
    }

    impl TopologyVisitor for Recorded {
        fn header(
            &mut self,
            entry_point: Option<NodeId>,
            max_level: usize,
            node_count: usize,
        ) -> Result<()> {
            self.headers.push((entry_point, max_level, node_count));
            Ok(())
        }

        fn node(&mut self, id: NodeId, layers: &[Vec<NodeId>]) -> Result<()> {
            if self.refuse_at == Some(self.nodes.len()) {
                return Err(Error::Internal(format!("Vincent refuses node {id:?}")));
            }
            self.nodes.push((id, layers.to_vec()));
            Ok(())
        }
    }

    /// 88 vectors of 3 dimensions on several levels (`ml` 1.0 puts about a
    /// third of the nodes above layer 0).
    /// The vectors of [`layered_index`].
    fn layered_vectors() -> HashMap<NodeId, Arc<[f32]>> {
        (1..=88u64)
            .map(|i| {
                let vector: Arc<[f32]> =
                    vec![(i * 3 % 19) as f32, (i * 19 % 88) as f32, i as f32].into();
                (NodeId::new(i * 3), vector)
            })
            .collect()
    }

    fn layered_index() -> HnswIndex {
        let mut config = HnswConfig::new(3, DistanceMetric::Euclidean);
        config.ml = 1.0;
        let index = HnswIndex::with_seed(config, 19);
        let map = layered_vectors();
        let accessor = make_accessor(&map);
        let mut ids: Vec<NodeId> = map.keys().copied().collect();
        ids.sort_unstable();
        for id in ids {
            index.insert(id, &map[&id], &accessor);
        }
        index
    }

    /// The heap topology is visited as `snapshot_topology` lists it: the
    /// header once, then every node in increasing id order with its lists.
    #[test]
    fn a_heap_topology_is_visited_in_id_order() {
        let index = layered_index();
        let (entry_point, max_level, nodes) = index.snapshot_topology();
        assert!(max_level >= 1, "the index has upper layers");

        let mut recorded = Recorded::default();
        index.visit_topology(&mut recorded).unwrap();
        assert_eq!(
            recorded.headers,
            [(entry_point, max_level, nodes.len())],
            "one header with the entry point, the level and the node count"
        );
        assert_eq!(
            recorded.nodes, nodes,
            "every node in id order, with its lists"
        );
    }

    /// An mmap-backed topology is visited with the same nodes and lists as
    /// the heap one it was serialized from, layers above 0 included.
    #[test]
    fn an_mmap_backed_topology_streams_node_by_node() {
        let heap = layered_index();
        let (entry_point, max_level, nodes) = heap.snapshot_topology();
        assert!(
            nodes.iter().any(|(_, layers)| layers.len() > 1),
            "some node has lists above layer 0"
        );
        let topo = MmapTopology::from_bytes(Bytes::from(serialize_topology(
            entry_point,
            max_level,
            &nodes,
        )))
        .unwrap();
        let mmap = HnswIndex::new(heap.config().clone());
        mmap.adopt_mmap_topology(topo);
        assert!(mmap.is_mmap_backed());

        let mut recorded = Recorded::default();
        mmap.visit_topology(&mut recorded).unwrap();
        assert_eq!(recorded.headers, [(entry_point, max_level, nodes.len())]);
        assert_eq!(recorded.nodes, nodes, "the same nodes and lists");
        assert!(mmap.is_mmap_backed(), "a visit leaves the backend as it is");
    }

    /// A visitor's error stops the visit and is returned as it is.
    #[test]
    fn a_visitor_error_stops_the_visit() {
        let index = layered_index();
        let mut recorded = Recorded {
            refuse_at: Some(3),
            ..Recorded::default()
        };
        let error = index.visit_topology(&mut recorded).unwrap_err();
        assert!(
            matches!(&error, Error::Internal(message) if message.contains("Vincent refuses")),
            "{error:?}"
        );
        assert_eq!(
            recorded.nodes.len(),
            3,
            "no node is visited after the error"
        );
    }

    /// `begin_restore` and `restore_node` rebuild a topology node by node,
    /// in heap mode also when the index was mmap-backed before.
    #[test]
    fn a_topology_is_restored_node_by_node() {
        let source = layered_index();
        let (entry_point, max_level, nodes) = source.snapshot_topology();

        let bytes =
            serialize_topology(Some(NodeId::new(88)), 0, &[(NodeId::new(88), vec![vec![]])]);
        let index = HnswIndex::new(source.config().clone());
        index.adopt_mmap_topology(MmapTopology::from_bytes(Bytes::from(bytes)).unwrap());

        index.begin_restore(entry_point, max_level, nodes.len());
        assert!(!index.is_mmap_backed(), "a restore replaces the mmap view");
        assert!(index.is_empty(), "a restore starts from an empty topology");
        for (id, layers) in nodes.iter().rev() {
            index.restore_node(*id, layers.clone());
        }
        assert_eq!(index.snapshot_topology(), (entry_point, max_level, nodes));
    }

    /// The node count a restore begins with only sizes the map, and a count
    /// no file could hold is not allocated up front.
    #[test]
    fn begin_restore_allocates_at_most_a_bounded_capacity() {
        let index = HnswIndex::new(HnswConfig::new(3, DistanceMetric::Cosine));
        index.begin_restore(Some(NodeId::new(3)), 19, usize::MAX);
        index.restore_node(NodeId::new(3), vec![vec![]]);
        assert_eq!(
            index.snapshot_topology(),
            (
                Some(NodeId::new(3)),
                19,
                vec![(NodeId::new(3), vec![vec![]])]
            )
        );
        // At most 1 << 16 nodes up front: under 4 MiB of map.
        assert!(
            index.heap_memory_bytes() < 1 << 22,
            "{} bytes held for one node",
            index.heap_memory_bytes()
        );
    }

    /// Every neighbor of a topology an index built is a node with a list at
    /// that level; a restored topology that lists anything else is reported,
    /// the lowest node id (then level) first, on a heap or mmap topology.
    #[test]
    fn a_broken_neighbor_reference_is_reported_lowest_node_first() {
        let id = NodeId::new;
        let built = layered_index();
        assert_eq!(built.first_broken_link(), None, "an index built by inserts");
        let vectors = layered_vectors();
        for removed in [15, 30, 45] {
            assert!(built.remove(id(removed), &make_accessor(&vectors)));
        }
        assert_eq!(built.first_broken_link(), None, "and after removals");
        let index = HnswIndex::new(HnswConfig::new(3, DistanceMetric::Cosine));
        let cases: Vec<(&str, Vec<(NodeId, Vec<Vec<NodeId>>)>, Option<BrokenLink>)> = vec![
            (
                "every neighbor a node with the level",
                vec![
                    (id(3), vec![vec![id(19)], vec![id(19)]]),
                    (id(19), vec![vec![id(3), id(88)], vec![id(3)]]),
                    (id(88), vec![vec![id(19)]]),
                ],
                None,
            ),
            (
                "a neighbor that is not a node, after one that is",
                vec![
                    (id(3), vec![vec![id(19)]]),
                    (id(19), vec![vec![id(3), id(7), id(88)]]),
                ],
                Some(BrokenLink {
                    node: id(19),
                    level: 0,
                    neighbor: id(7),
                    neighbor_levels: None,
                }),
            ),
            (
                "a neighbor without the level, and a later node listing itself",
                vec![
                    (id(3), vec![vec![id(19)]]),
                    (id(19), vec![vec![id(3)], vec![id(3)]]),
                    (id(88), vec![vec![id(88)]]),
                ],
                Some(BrokenLink {
                    node: id(19),
                    level: 1,
                    neighbor: id(3),
                    neighbor_levels: Some(1),
                }),
            ),
            (
                "a node listing itself before a later dangling neighbor",
                vec![
                    (id(3), vec![vec![id(19)], vec![id(3)]]),
                    (id(19), vec![vec![id(7)]]),
                ],
                Some(BrokenLink {
                    node: id(3),
                    level: 1,
                    neighbor: id(3),
                    neighbor_levels: Some(2),
                }),
            ),
        ];
        for (case, nodes, expected) in cases {
            index.restore_topology(Some(id(3)), 1, nodes.clone());
            assert_eq!(index.first_broken_link(), expected, "heap: {case}");
            let bytes = serialize_topology(Some(id(3)), 1, &nodes);
            index.adopt_mmap_topology(MmapTopology::from_bytes(Bytes::from(bytes)).unwrap());
            assert_eq!(index.first_broken_link(), expected, "mmap: {case}");
        }
    }
}
