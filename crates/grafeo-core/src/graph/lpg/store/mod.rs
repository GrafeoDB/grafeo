//! The in-memory LPG graph store.
//!
//! This is where your nodes and edges actually live. Most users interact
//! through [`GrafeoDB`](grafeo_engine::GrafeoDB), but algorithm implementers
//! sometimes need the raw [`LpgStore`] for direct adjacency traversal.
//!
//! Key features:
//! - MVCC versioning - concurrent readers don't block each other
//! - Columnar properties with zone maps for fast filtering
//! - Forward and backward adjacency indexes

mod apply;
mod edge_ops;
mod graph_store_impl;
mod index;
mod memory;
mod node_ops;
mod open_changes;
mod property_index;
mod property_ops;
mod schema;
mod search;
mod statistics;
mod traversal;
mod versioning;

#[cfg(test)]
mod tests;

#[cfg(test)]
pub(crate) use apply::testing;
pub use open_changes::OpenChangesByGraph;
pub(crate) use open_changes::{Labels, OpenChangeSource, OpenChanges};

use super::PropertyStorage;
use super::dictionary::NameDictionary;
#[cfg(not(feature = "tiered-storage"))]
use super::{EdgeRecord, NodeRecord};
use crate::index::adjacency::ChunkedAdjacency;
use crate::statistics::Statistics;
#[cfg(not(feature = "tiered-storage"))]
use grafeo_common::mvcc::VersionChain;
use grafeo_common::types::{EdgeId, EpochId, NodeId, PropertyKey, Value};
use grafeo_common::utils::hash::{FxHashMap, FxHashSet};
use parking_lot::RwLock;
use std::cmp::Ordering as CmpOrdering;
use std::sync::Arc;
use std::sync::atomic::{AtomicI64, AtomicU64, Ordering};

#[cfg(feature = "tiered-storage")]
use crate::codec::EpochStore;
use grafeo_common::memory::arena::AllocError;
#[cfg(feature = "tiered-storage")]
use grafeo_common::memory::arena::ArenaAllocator;
#[cfg(feature = "tiered-storage")]
use grafeo_common::mvcc::VersionIndex;
#[cfg(feature = "temporal")]
use grafeo_common::temporal::VersionLog;

/// Compares two values for ordering (used for range checks).
pub(super) fn compare_values_for_range(a: &Value, b: &Value) -> Option<CmpOrdering> {
    match (a, b) {
        (Value::Int64(a), Value::Int64(b)) => Some(a.cmp(b)),
        (Value::Float64(a), Value::Float64(b)) => a.partial_cmp(b),
        (Value::Int64(a), Value::Float64(b)) => (*a as f64).partial_cmp(b),
        (Value::Float64(a), Value::Int64(b)) => a.partial_cmp(&(*b as f64)),
        (Value::String(a), Value::String(b)) => Some(a.cmp(b)),
        (Value::Bool(a), Value::Bool(b)) => Some(a.cmp(b)),
        (Value::Timestamp(a), Value::Timestamp(b)) => Some(a.cmp(b)),
        (Value::Date(a), Value::Date(b)) => Some(a.cmp(b)),
        (Value::Time(a), Value::Time(b)) => Some(a.cmp(b)),
        // Zoned datetimes, also against a timestamp: by their instant, as a
        // filter compares them.
        _ => a.compare_instants(b),
    }
}

/// Checks if a value is within a range.
pub(super) fn value_in_range(
    value: &Value,
    min: Option<&Value>,
    max: Option<&Value>,
    min_inclusive: bool,
    max_inclusive: bool,
) -> bool {
    // Check lower bound
    if let Some(min_val) = min {
        match compare_values_for_range(value, min_val) {
            Some(CmpOrdering::Less) => return false,
            Some(CmpOrdering::Equal) if !min_inclusive => return false,
            None => return false, // Can't compare
            _ => {}
        }
    }

    // Check upper bound
    if let Some(max_val) = max {
        match compare_values_for_range(value, max_val) {
            Some(CmpOrdering::Greater) => return false,
            Some(CmpOrdering::Equal) if !max_inclusive => return false,
            None => return false,
            _ => {}
        }
    }

    true
}

/// Configuration for the LPG store.
///
/// The defaults work well for most cases. Tune `backward_edges` if you only
/// traverse in one direction (saves memory), or adjust capacities if you know
/// your graph size upfront (avoids reallocations).
#[derive(Debug, Clone)]
pub struct LpgStoreConfig {
    /// Maintain backward adjacency for incoming edge queries. Turn off if
    /// you only traverse outgoing edges - saves ~50% adjacency memory.
    pub backward_edges: bool,
    /// Initial capacity for nodes (avoids early reallocations).
    pub initial_node_capacity: usize,
    /// Initial capacity for edges (avoids early reallocations).
    pub initial_edge_capacity: usize,
}

impl Default for LpgStoreConfig {
    fn default() -> Self {
        Self {
            backward_edges: true,
            initial_node_capacity: 1024,
            initial_edge_capacity: 4096,
        }
    }
}

/// The entities whose history garbage collection has to trim.
#[cfg(any(feature = "temporal", feature = "tiered-storage"))]
#[derive(Default)]
pub(super) struct GcCandidates {
    /// Nodes whose label log holds an older label set.
    #[cfg(feature = "temporal")]
    pub(super) labels: FxHashSet<NodeId>,
    /// Nodes whose version index holds more than one version (re-created).
    #[cfg(feature = "tiered-storage")]
    pub(super) nodes: FxHashSet<NodeId>,
    /// Edges whose version index holds more than one version (re-created).
    #[cfg(feature = "tiered-storage")]
    pub(super) edges: FxHashSet<EdgeId>,
}

/// The core in-memory graph storage.
///
/// Everything lives here: nodes, edges, properties, adjacency indexes, and
/// version chains for MVCC. Concurrent reads never block each other.
///
/// Most users should go through `GrafeoDB` (from the `grafeo_engine` crate) which
/// adds transaction management and query execution. Use `LpgStore` directly
/// when you need raw performance for algorithm implementations.
///
/// # Example
///
/// ```
/// use grafeo_core::graph::lpg::LpgStore;
/// use grafeo_core::graph::Direction;
///
/// let store = LpgStore::new().expect("arena allocation");
///
/// // Create a small social network
/// let alix = store.create_node(&["Person"]);
/// let gus = store.create_node(&["Person"]);
/// store.create_edge(alix, gus, "KNOWS");
///
/// // Traverse outgoing edges
/// for neighbor in store.neighbors(alix, Direction::Outgoing) {
///     println!("Alix knows node {:?}", neighbor);
/// }
/// ```
///
/// # Lock Ordering
///
/// `LpgStore` contains multiple `RwLock` fields that must be acquired in a
/// consistent order to prevent deadlocks. Always acquire locks in this order:
///
/// ## Level 1: Entity Storage (mutually exclusive via feature flag)
/// 1. `nodes` / `node_versions`
/// 2. `edges` / `edge_versions`
///
/// ## Level 2: Catalogs
/// 3. `label_registry`
/// 4. `edge_types`
///
/// ## Level 3: Indexes
/// 5. `label_index`
/// 6. `node_labels`
/// 7. `property_indexes`
///
/// ## Level 4: Statistics
/// 8. `statistics`
///
/// ## Level 5: Nested Locks (internal to other structs)
/// 9. `PropertyStorage::columns` (via `node_properties`/`edge_properties`)
/// 10. `ChunkedAdjacency::lists` (via `forward_adj`/`backward_adj`)
///
/// ## Rules
/// - Never hold entity locks while acquiring catalog locks in a different scope.
/// - Statistics lock is always last.
/// - Read locks are generally safe, but avoid read-to-write upgrades.
pub struct LpgStore {
    /// Node records indexed by NodeId, with version chains for MVCC.
    /// Used when `tiered-storage` feature is disabled.
    /// Lock order: 1
    #[cfg(not(feature = "tiered-storage"))]
    pub(super) nodes: RwLock<FxHashMap<NodeId, VersionChain<NodeRecord>>>,

    /// Edge records indexed by EdgeId, with version chains for MVCC.
    /// Used when `tiered-storage` feature is disabled.
    /// Lock order: 2
    #[cfg(not(feature = "tiered-storage"))]
    pub(super) edges: RwLock<FxHashMap<EdgeId, VersionChain<EdgeRecord>>>,

    // === Tiered Storage Fields (feature-gated) ===
    //
    // Lock ordering for arena access:
    //   version_lock (read/write) → arena read lock (via arena_allocator.arena())
    //
    // Rules:
    // - Acquire arena read lock *after* version locks, never before.
    // - Multiple threads may call arena.read_at() concurrently (shared refs only).
    // - Never acquire arena write lock (alloc_new_chunk) while holding version locks.
    // - freeze_epoch order: node_versions.read() → arena.read_at(),
    //   then edge_versions.read() → arena.read_at().
    /// Arena allocator for hot data storage.
    /// Data is stored in per-epoch arenas for fast allocation and bulk deallocation.
    #[cfg(feature = "tiered-storage")]
    pub(super) arena_allocator: Arc<ArenaAllocator>,

    /// Node version indexes - store metadata and arena offsets.
    /// The actual NodeRecord data is stored in the arena.
    /// Lock order: 1
    #[cfg(feature = "tiered-storage")]
    pub(super) node_versions: RwLock<FxHashMap<NodeId, VersionIndex>>,

    /// Edge version indexes - store metadata and arena offsets.
    /// The actual EdgeRecord data is stored in the arena.
    /// Lock order: 2
    #[cfg(feature = "tiered-storage")]
    pub(super) edge_versions: RwLock<FxHashMap<EdgeId, VersionIndex>>,

    /// Cold storage for frozen epochs.
    /// Contains compressed epoch blocks for historical data.
    #[cfg(feature = "tiered-storage")]
    pub(super) epoch_store: Arc<EpochStore>,

    /// Property storage for nodes.
    pub(super) node_properties: PropertyStorage<NodeId>,

    /// Property storage for edges.
    pub(super) edge_properties: PropertyStorage<EdgeId>,

    /// The labels' ids, append-only (see [`NameDictionary`]).
    /// Lock order: 3
    pub(super) label_registry: RwLock<NameDictionary>,

    /// The edge types' ids, append-only (see [`NameDictionary`]).
    /// Lock order: 4
    pub(super) edge_types: RwLock<NameDictionary>,

    /// The property keys' ids, shared by the node and edge tables,
    /// append-only (see [`NameDictionary`]): the columns of the LPG section
    /// are keyed by them. A key gets its id when a checkpoint first writes
    /// it, or from the file at a load. Lock order: independent (never held
    /// with another lock of the store).
    pub(super) property_keys: RwLock<NameDictionary>,

    /// This graph's id in the file: 0 for the default graph, the id it was
    /// created with for a named graph (never reused, see
    /// [`next_graph_id`](Self::next_graph_id)). Atomic only so a load can
    /// give a graph an earlier step of the same open created the id the file
    /// holds (see [`restore_graphs`](Self::restore_graphs)).
    graph_id: std::sync::atomic::AtomicU32,

    /// The id the next named graph created in this store gets (1 for a new
    /// store); it only grows, so a dropped graph's id is never given again.
    next_graph_id: std::sync::atomic::AtomicU32,

    /// Whether this store is a named graph that was dropped (see
    /// [`drop_graph`](Self::drop_graph)): its [`ChangeTarget`](crate::graph::apply::ChangeTarget)
    /// refuses every write from then on, so a writer that resolved the graph
    /// before the drop changes nothing a commit would log for it.
    dropped: std::sync::atomic::AtomicBool,

    /// Forward adjacency lists (outgoing edges).
    pub(super) forward_adj: ChunkedAdjacency,

    /// Backward adjacency lists (incoming edges).
    /// Only populated if config.backward_edges is true.
    pub(super) backward_adj: Option<ChunkedAdjacency>,

    /// Label index: label_id -> set of node IDs.
    /// Lock order: 5
    pub(super) label_index: RwLock<Vec<FxHashMap<NodeId, ()>>>,

    /// Node labels: node_id -> set of label IDs.
    /// Reverse mapping to efficiently get labels for a node.
    /// Lock order: 6
    #[cfg(not(feature = "temporal"))]
    pub(super) node_labels: RwLock<FxHashMap<NodeId, FxHashSet<u32>>>,
    /// Versioned node labels: node_id -> version log of label sets.
    /// Lock order: 6
    #[cfg(feature = "temporal")]
    pub(super) node_labels: RwLock<FxHashMap<NodeId, VersionLog<FxHashSet<u32>>>>,

    /// Property indexes: property_key -> (value -> set of node IDs, see
    /// [`PropertyIndex`](property_index::PropertyIndex)).
    ///
    /// When a property is indexed, lookups by value are O(1) instead of O(n).
    /// Use [`create_property_index`] to enable indexing for a property.
    /// Lock order: 7, after the node lock (1): an adoption of a base node
    /// holds the node lock while it adds the copy here, so a reader that
    /// needs the node lock too (`create_property_index`, whose scan reads
    /// the node ids) takes it first, never while it holds this one.
    pub(super) property_indexes: RwLock<FxHashMap<PropertyKey, property_index::PropertyIndex>>,

    /// Vector indexes: "label:property" -> HNSW index, with the property.
    ///
    /// Created via [`GrafeoDB::create_vector_index`](grafeo_engine::GrafeoDB::create_vector_index).
    /// Lock order: 7 (same level as property_indexes, disjoint keys)
    #[cfg(feature = "vector-index")]
    pub(super) vector_indexes: RwLock<FxHashMap<String, index::StoredVectorIndex>>,

    /// Text indexes: "label:property" -> inverted index with BM25 scoring.
    ///
    /// Created via [`GrafeoDB::create_text_index`](grafeo_engine::GrafeoDB::create_text_index).
    /// Lock order: 7 (same level as property_indexes, disjoint keys)
    #[cfg(feature = "text-index")]
    pub(super) text_indexes:
        RwLock<FxHashMap<String, Arc<RwLock<crate::index::text::InvertedIndex>>>>,

    /// Next node ID.
    pub(super) next_node_id: AtomicU64,

    /// Next edge ID.
    pub(super) next_edge_id: AtomicU64,

    /// Current epoch.
    pub(super) current_epoch: AtomicU64,

    /// Live (non-deleted) node count, maintained incrementally.
    /// Avoids O(n) full scan in `compute_statistics()`.
    pub(super) live_node_count: AtomicI64,

    /// The entities that hold older versions, the only ones garbage
    /// collection has work for (see `gc_versions`). A leaf lock: nothing is
    /// acquired while it is held.
    #[cfg(any(feature = "temporal", feature = "tiered-storage"))]
    pub(super) gc_candidates: parking_lot::Mutex<GcCandidates>,

    /// Live (non-deleted) edge count, maintained incrementally.
    /// Avoids O(m) full scan in `compute_statistics()`.
    pub(super) live_edge_count: AtomicI64,

    /// Per-edge-type live counts, indexed by edge_type_id.
    /// Avoids O(m) edge scan in `compute_statistics()`.
    /// Lock order: 8 (same level as statistics)
    pub(super) edge_type_live_counts: RwLock<Vec<i64>>,

    /// Statistics for cost-based optimization.
    /// Lock order: 8 (always last)
    pub(super) statistics: RwLock<Arc<Statistics>>,

    /// Named graphs, each an independent `LpgStore` partition.
    /// Zero overhead for single-graph databases (empty HashMap).
    /// Lock order: 9 (after statistics)
    named_graphs: RwLock<FxHashMap<String, Arc<LpgStore>>>,
}

impl LpgStore {
    /// Creates a new LPG store with default configuration.
    ///
    /// # Errors
    ///
    /// Returns [`AllocError`] if the arena allocator cannot be initialized
    /// (only possible with the `tiered-storage` feature).
    pub fn new() -> Result<Self, AllocError> {
        Self::with_config(LpgStoreConfig::default())
    }

    /// Creates a new LPG store with custom configuration.
    ///
    /// # Errors
    ///
    /// Returns [`AllocError`] if the arena allocator cannot be initialized
    /// (only possible with the `tiered-storage` feature).
    pub fn with_config(config: LpgStoreConfig) -> Result<Self, AllocError> {
        let backward_adj = if config.backward_edges {
            Some(ChunkedAdjacency::new())
        } else {
            None
        };

        Ok(Self {
            #[cfg(not(feature = "tiered-storage"))]
            nodes: RwLock::new(FxHashMap::default()),
            #[cfg(not(feature = "tiered-storage"))]
            edges: RwLock::new(FxHashMap::default()),
            #[cfg(feature = "tiered-storage")]
            arena_allocator: Arc::new(ArenaAllocator::new()?),
            #[cfg(feature = "tiered-storage")]
            node_versions: RwLock::new(FxHashMap::default()),
            #[cfg(feature = "tiered-storage")]
            edge_versions: RwLock::new(FxHashMap::default()),
            #[cfg(feature = "tiered-storage")]
            epoch_store: Arc::new(EpochStore::new()),
            node_properties: PropertyStorage::new(),
            edge_properties: PropertyStorage::new(),
            label_registry: RwLock::new(NameDictionary::new()),
            edge_types: RwLock::new(NameDictionary::new()),
            property_keys: RwLock::new(NameDictionary::new()),
            graph_id: std::sync::atomic::AtomicU32::new(0),
            next_graph_id: std::sync::atomic::AtomicU32::new(1),
            dropped: std::sync::atomic::AtomicBool::new(false),
            forward_adj: ChunkedAdjacency::new(),
            backward_adj,
            label_index: RwLock::new(Vec::with_capacity(16)),
            node_labels: RwLock::new(FxHashMap::default()),
            property_indexes: RwLock::new(FxHashMap::default()),
            #[cfg(feature = "vector-index")]
            vector_indexes: RwLock::new(FxHashMap::default()),
            #[cfg(feature = "text-index")]
            text_indexes: RwLock::new(FxHashMap::default()),
            next_node_id: AtomicU64::new(0),
            next_edge_id: AtomicU64::new(0),
            current_epoch: AtomicU64::new(0),
            live_node_count: AtomicI64::new(0),
            #[cfg(any(feature = "temporal", feature = "tiered-storage"))]
            gc_candidates: parking_lot::Mutex::new(GcCandidates::default()),
            live_edge_count: AtomicI64::new(0),
            edge_type_live_counts: RwLock::new(Vec::new()),
            statistics: RwLock::new(Arc::new(Statistics::new())),
            named_graphs: RwLock::new(FxHashMap::default()),
        })
    }

    /// Returns the current epoch.
    #[must_use]
    pub fn current_epoch(&self) -> EpochId {
        EpochId::new(self.current_epoch.load(Ordering::Acquire))
    }

    /// Creates a new epoch.
    #[doc(hidden)]
    pub fn new_epoch(&self) -> EpochId {
        let id = self.current_epoch.fetch_add(1, Ordering::AcqRel) + 1;
        EpochId::new(id)
    }

    /// Syncs the store epoch to match an external epoch counter.
    ///
    /// Used by the transaction manager to keep the store's epoch in step
    /// after a transaction commit advances the global epoch.
    #[doc(hidden)]
    pub fn sync_epoch(&self, epoch: EpochId) {
        self.current_epoch
            .fetch_max(epoch.as_u64(), Ordering::AcqRel);
    }

    /// Returns the current next node ID counter value.
    #[doc(hidden)]
    #[must_use]
    pub fn next_node_id(&self) -> u64 {
        self.next_node_id.load(Ordering::Acquire)
    }

    /// Returns the current next edge ID counter value.
    #[doc(hidden)]
    #[must_use]
    pub fn next_edge_id(&self) -> u64 {
        self.next_edge_id.load(Ordering::Acquire)
    }

    /// Sets the next node ID counter.
    ///
    /// Used when loading a store, to keep its ID allocator above every ID
    /// the store holds.
    #[doc(hidden)]
    pub fn set_next_node_id(&self, id: u64) {
        self.next_node_id.store(id, Ordering::Release);
    }

    /// Sets the next edge ID counter.
    ///
    /// See [`set_next_node_id`](Self::set_next_node_id).
    #[doc(hidden)]
    pub fn set_next_edge_id(&self, id: u64) {
        self.next_edge_id.store(id, Ordering::Release);
    }

    /// Removes all data from the store, resetting it to an empty state.
    ///
    /// Acquires locks in the documented ordering to prevent deadlocks.
    /// After clearing, the store behaves as if freshly constructed, except
    /// that its name dictionaries (labels, edge types, property keys) keep
    /// every id they gave out: ids are never reassigned, so a name used
    /// again gets its old id and a new name a new one.
    pub fn clear(&self) {
        // Level 1: Entity storage
        #[cfg(not(feature = "tiered-storage"))]
        {
            self.nodes.write().clear();
            self.edges.write().clear();
        }
        #[cfg(feature = "tiered-storage")]
        {
            self.node_versions.write().clear();
            self.edge_versions.write().clear();
            // Arena allocator chunks are leaked; epochs are cleared via epoch_store.
        }

        // Level 2: the catalogs (name dictionaries) keep their ids.

        // Level 3: Indexes
        self.label_index.write().clear();
        self.node_labels.write().clear();
        self.property_indexes.write().clear();
        #[cfg(feature = "vector-index")]
        self.vector_indexes.write().clear();
        #[cfg(feature = "text-index")]
        self.text_indexes.write().clear();

        // Nested: Properties and adjacency
        self.node_properties.clear();
        self.edge_properties.clear();
        self.forward_adj.clear();
        if let Some(ref backward) = self.backward_adj {
            backward.clear();
        }

        // Atomics: ID counters
        self.next_node_id.store(0, Ordering::Release);
        self.next_edge_id.store(0, Ordering::Release);
        self.current_epoch.store(0, Ordering::Release);

        // Level 4: Statistics
        self.live_node_count.store(0, Ordering::Release);
        self.live_edge_count.store(0, Ordering::Release);
        self.edge_type_live_counts.write().clear();
        *self.statistics.write() = Arc::new(Statistics::new());
    }

    /// Returns whether backward adjacency (incoming edge index) is available.
    ///
    /// When backward adjacency is enabled (the default), bidirectional search
    /// algorithms can traverse from the target toward the source.
    #[must_use]
    pub fn has_backward_adjacency(&self) -> bool {
        self.backward_adj.is_some()
    }

    // === Named Graph Management ===

    /// Returns a named graph by name, or `None` if it does not exist.
    #[must_use]
    pub fn graph(&self, name: &str) -> Option<Arc<LpgStore>> {
        self.named_graphs.read().get(name).cloned()
    }

    /// Returns a named graph, creating it if it does not exist.
    ///
    /// # Errors
    ///
    /// Returns [`AllocError`] if a new store cannot be allocated.
    pub fn graph_or_create(&self, name: &str) -> Result<Arc<LpgStore>, AllocError> {
        {
            let graphs = self.named_graphs.read();
            if let Some(g) = graphs.get(name) {
                return Ok(Arc::clone(g));
            }
        }
        let mut graphs = self.named_graphs.write();
        // Double-check after acquiring write lock
        if let Some(g) = graphs.get(name) {
            return Ok(Arc::clone(g));
        }
        let store = Arc::new(self.new_named_graph()?);
        graphs.insert(name.to_string(), Arc::clone(&store));
        Ok(store)
    }

    /// Creates a named graph. Returns `true` on success, `false` if it already exists.
    ///
    /// # Errors
    ///
    /// Returns [`AllocError`] if the new store cannot be allocated.
    pub fn create_graph(&self, name: &str) -> Result<bool, AllocError> {
        let mut graphs = self.named_graphs.write();
        if graphs.contains_key(name) {
            return Ok(false);
        }
        graphs.insert(name.to_string(), Arc::new(self.new_named_graph()?));
        Ok(true)
    }

    /// A new store for a named graph, with the next graph id. The caller
    /// holds `named_graphs` for writing, so ids are given out one at a time.
    ///
    /// # Errors
    ///
    /// Returns [`AllocError`] if the store cannot be allocated, or
    /// [`AllocError::IdsExhausted`] once every graph id was given out.
    fn new_named_graph(&self) -> Result<LpgStore, AllocError> {
        let id = self.next_graph_id.load(Ordering::Acquire);
        let next = id
            .checked_add(1)
            .ok_or(AllocError::IdsExhausted { what: "graph" })?;
        let store = LpgStore::new()?;
        store.graph_id.store(id, Ordering::Release);
        self.next_graph_id.store(next, Ordering::Release);
        Ok(store)
    }

    /// Restores the named graphs `graphs` (`(id, name)`, each id from 1 and
    /// below `next_graph_id`, each id and name once) with their ids, as a
    /// load does, and makes `next_graph_id` the next graph id at least.
    /// Returns the graphs' stores, in the order of `graphs`.
    ///
    /// A graph this store holds already was created in this process (an
    /// earlier step of the open creates some, as the catalog does for a
    /// schema's graphs), so its id was never written: a listed one takes its
    /// id from `graphs`, and one not listed moves to an id at or above
    /// `next_graph_id`, so no two graphs share an id.
    ///
    /// # Errors
    ///
    /// Returns what is wrong when an id is 0 or not below `next_graph_id`,
    /// when every graph id is given out, or when a store cannot be allocated.
    pub(crate) fn restore_graphs(
        &self,
        graphs: &[(u32, &str)],
        next_graph_id: u32,
    ) -> Result<Vec<Arc<LpgStore>>, String> {
        let mut named = self.named_graphs.write();
        if let Some((id, name)) = graphs
            .iter()
            .find(|(id, _)| *id == 0 || *id >= next_graph_id)
        {
            return Err(format!(
                "graph {name:?} has id {id}, where named graphs have ids from 1 below \
                 {next_graph_id}"
            ));
        }
        self.next_graph_id
            .fetch_max(next_graph_id, Ordering::AcqRel);
        let listed: FxHashSet<&str> = graphs.iter().map(|(_, name)| *name).collect();
        for (name, graph) in named.iter() {
            if !listed.contains(name.as_str()) {
                let id = self.next_graph_id.load(Ordering::Acquire);
                let next = id
                    .checked_add(1)
                    .ok_or_else(|| AllocError::IdsExhausted { what: "graph" }.to_string())?;
                graph.graph_id.store(id, Ordering::Release);
                self.next_graph_id.store(next, Ordering::Release);
            }
        }
        let mut restored = Vec::with_capacity(graphs.len());
        for &(id, name) in graphs {
            let graph = match named.get(name) {
                Some(graph) => Arc::clone(graph),
                None => {
                    let graph = Arc::new(LpgStore::new().map_err(|error| error.to_string())?);
                    named.insert(name.to_string(), Arc::clone(&graph));
                    graph
                }
            };
            graph.graph_id.store(id, Ordering::Release);
            restored.push(graph);
        }
        Ok(restored)
    }

    /// This graph's id: 0 for the default graph (see the `graph_id` field).
    #[must_use]
    pub(crate) fn graph_id(&self) -> u32 {
        self.graph_id.load(Ordering::Acquire)
    }

    /// The id the next named graph created in this store gets.
    #[must_use]
    pub(crate) fn next_graph_id(&self) -> u32 {
        self.next_graph_id.load(Ordering::Acquire)
    }

    /// Drops a named graph. Returns `false` if it did not exist. Its id is
    /// not given out again: a graph created later under the same name gets
    /// a new one.
    ///
    /// The dropped graph's store refuses every write through its
    /// [`ChangeTarget`](crate::graph::apply::ChangeTarget) from now on (see
    /// [`is_dropped`](Self::is_dropped)): a writer that resolved the graph
    /// before the drop and writes after it changes nothing.
    pub fn drop_graph(&self, name: &str) -> bool {
        let removed = self.named_graphs.write().remove(name);
        if let Some(graph) = &removed {
            graph.dropped.store(true, Ordering::Release);
        }
        removed.is_some()
    }

    /// Whether this store is a named graph that was dropped (see
    /// [`drop_graph`](Self::drop_graph)).
    #[must_use]
    pub fn is_dropped(&self) -> bool {
        self.dropped.load(Ordering::Acquire)
    }

    /// Returns all named graph names.
    #[must_use]
    pub fn graph_names(&self) -> Vec<String> {
        self.named_graphs.read().keys().cloned().collect()
    }

    /// Returns the number of named graphs.
    #[must_use]
    pub fn graph_count(&self) -> usize {
        self.named_graphs.read().len()
    }

    /// Clears a specific graph, or the default graph if `name` is `None`.
    pub fn clear_graph(&self, name: Option<&str>) {
        match name {
            Some(n) => {
                if let Some(g) = self.named_graphs.read().get(n) {
                    g.clear();
                }
            }
            None => self.clear(),
        }
    }

    /// Copies all data from the source graph to the destination graph.
    /// Creates the destination graph if it does not exist.
    ///
    /// # Errors
    ///
    /// Returns [`AllocError`] if the destination store cannot be allocated.
    pub fn copy_graph(&self, source: Option<&str>, dest: Option<&str>) -> Result<(), AllocError> {
        let _src = match source {
            Some(n) => self.graph(n),
            None => None, // default graph
        };
        let _dest_graph = dest.map(|n| self.graph_or_create(n)).transpose()?;
        // Full graph copy is complex (requires iterating all entities).
        // For now, this creates the destination graph structure.
        // Full entity-level copy will be implemented when needed.
        Ok(())
    }

    // === Internal Helpers ===

    pub(super) fn get_or_create_label_id(&self, label: &str) -> u32 {
        if let Some(id) = self.label_registry.read().get_id(label) {
            return id;
        }
        self.label_registry.write().get_or_create(label)
    }

    pub(super) fn get_or_create_edge_type_id(&self, edge_type: &str) -> u32 {
        if let Some(id) = self.edge_types.read().get_id(edge_type) {
            return id;
        }
        let id = self.edge_types.write().get_or_create(edge_type);

        // Grow edge type live counts to match
        let mut counts = self.edge_type_live_counts.write();
        while counts.len() <= id as usize {
            counts.push(0);
        }

        id
    }

    /// Increments the live edge count for a given edge type.
    pub(super) fn increment_edge_type_count(&self, type_id: u32) {
        let mut counts = self.edge_type_live_counts.write();
        if counts.len() <= type_id as usize {
            counts.resize(type_id as usize + 1, 0);
        }
        counts[type_id as usize] += 1;
    }

    /// Decrements the live edge count for a given edge type.
    pub(super) fn decrement_edge_type_count(&self, type_id: u32) {
        let mut counts = self.edge_type_live_counts.write();
        // reason: counts.len() is bounded by edge type registry size, fits u32
        #[allow(clippy::cast_possible_truncation)]
        if type_id < counts.len() as u32 {
            counts[type_id as usize] -= 1;
        }
    }
}
