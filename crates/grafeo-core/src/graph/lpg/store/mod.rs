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

mod edge_ops;
mod graph_store_impl;
mod index;
mod memory;
mod node_ops;
mod property_ops;
mod schema;
mod search;
mod statistics;
mod traversal;
mod versioning;

#[cfg(test)]
mod tests;

use super::PropertyStorage;
#[cfg(not(feature = "tiered-storage"))]
use super::{EdgeRecord, NodeRecord};
use crate::index::adjacency::ChunkedAdjacency;
use crate::statistics::Statistics;
use arcstr::ArcStr;
use dashmap::DashMap;
#[cfg(not(feature = "tiered-storage"))]
use grafeo_common::mvcc::VersionChain;
use grafeo_common::types::{
    EdgeId, EpochId, HashableValue, NodeId, PropertyKey, TransactionId, Value,
};
use grafeo_common::utils::hash::{FxHashMap, FxHashSet};
use parking_lot::RwLock;
use std::cmp::Ordering as CmpOrdering;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicI64, AtomicU64, Ordering};

#[cfg(feature = "vector-index")]
use crate::index::vector::VectorIndexKind;

#[cfg(feature = "tiered-storage")]
use crate::codec::EpochStore;
use grafeo_common::memory::arena::AllocError;
#[cfg(feature = "tiered-storage")]
use grafeo_common::memory::arena::ArenaAllocator;
#[cfg(feature = "tiered-storage")]
use grafeo_common::mvcc::VersionIndex;
#[cfg(feature = "temporal")]
use grafeo_common::temporal::VersionLog;

/// Undo entry for a property mutation within a transaction.
///
/// Captures the previous state of a property so it can be restored on rollback.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub enum PropertyUndoEntry {
    /// A node property was changed or added.
    NodeProperty {
        /// The node that was modified.
        node_id: NodeId,
        /// The property key that was set or removed.
        key: PropertyKey,
        /// The previous value, or `None` if the property did not exist before.
        old_value: Option<Value>,
    },
    /// An edge property was changed or added.
    EdgeProperty {
        /// The edge that was modified.
        edge_id: EdgeId,
        /// The property key that was set or removed.
        key: PropertyKey,
        /// The previous value, or `None` if the property did not exist before.
        old_value: Option<Value>,
    },
    /// A label was added to a node.
    LabelAdded {
        /// The node that had a label added.
        node_id: NodeId,
        /// The label string that was added.
        label: String,
    },
    /// A label was removed from a node.
    LabelRemoved {
        /// The node that had a label removed.
        node_id: NodeId,
        /// The label string that was removed.
        label: String,
    },
    /// A node was deleted (for rollback restoration).
    NodeDeleted {
        /// The node that was deleted.
        node_id: NodeId,
        /// The labels the node had before deletion.
        labels: Vec<String>,
        /// The properties the node had before deletion.
        properties: Vec<(PropertyKey, Value)>,
    },
    /// An edge was deleted (for rollback restoration).
    EdgeDeleted {
        /// The edge that was deleted.
        edge_id: EdgeId,
        /// The source node.
        src: NodeId,
        /// The destination node.
        dst: NodeId,
        /// The edge type name.
        edge_type: String,
        /// The properties the edge had before deletion.
        properties: Vec<(PropertyKey, Value)>,
    },
}

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
        _ => None,
    }
}

/// Checks if a value is within a range.
pub(crate) fn value_in_range(
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

/// Bidirectional label name/ID registry.
///
/// Combines the name-to-ID and ID-to-name mappings behind a single lock,
/// reducing lock acquisitions on both the read path (`build_node`) and the
/// write path (`get_or_create_label_id`).
pub(super) struct LabelRegistry {
    /// Label name to numeric ID.
    name_to_id: FxHashMap<ArcStr, u32>,
    /// Numeric ID to label name (index = ID).
    id_to_name: Vec<ArcStr>,
}

impl LabelRegistry {
    fn new() -> Self {
        Self {
            name_to_id: FxHashMap::default(),
            id_to_name: Vec::new(),
        }
    }

    /// Looks up an existing label ID by name.
    pub(super) fn get_id(&self, name: &str) -> Option<u32> {
        self.name_to_id.get(name).copied()
    }

    /// Returns the label name for a given ID.
    pub(super) fn get_name(&self, id: u32) -> Option<&ArcStr> {
        self.id_to_name.get(id as usize)
    }

    /// Returns or creates a label ID for the given name.
    pub(super) fn get_or_create(&mut self, name: &str) -> u32 {
        if let Some(&id) = self.name_to_id.get(name) {
            return id;
        }
        let id = self.id_to_name.len() as u32;
        let label: ArcStr = name.into();
        self.name_to_id.insert(label.clone(), id);
        self.id_to_name.push(label);
        id
    }

    /// Returns the total number of distinct labels.
    pub(super) fn len(&self) -> usize {
        self.id_to_name.len()
    }

    /// Returns the ID-to-name slice for iteration.
    pub(super) fn names(&self) -> &[ArcStr] {
        &self.id_to_name
    }

    /// Clears all label mappings.
    pub(super) fn clear(&mut self) {
        self.name_to_id.clear();
        self.id_to_name.clear();
    }

    /// Estimates heap memory usage in bytes.
    pub(super) fn heap_bytes(&self) -> usize {
        let map_bytes = self.name_to_id.capacity()
            * (std::mem::size_of::<ArcStr>() + std::mem::size_of::<u32>());
        let vec_bytes = self.id_to_name.capacity() * std::mem::size_of::<ArcStr>();
        let string_bytes: usize = self.id_to_name.iter().map(|s| s.len()).sum();
        map_bytes + vec_bytes + string_bytes
    }
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
/// 4. `edge_type_to_id` + `id_to_edge_type`
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

    /// Bidirectional label name/ID registry.
    /// Lock order: 3
    pub(super) label_registry: RwLock<LabelRegistry>,

    /// Edge type name to ID mapping.
    /// Lock order: 4 (acquire with id_to_edge_type)
    pub(super) edge_type_to_id: RwLock<FxHashMap<ArcStr, u32>>,

    /// Edge type ID to name mapping.
    /// Lock order: 4 (acquire with edge_type_to_id)
    pub(super) id_to_edge_type: RwLock<Vec<ArcStr>>,

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

    /// Property indexes: property_key -> (value -> set of node IDs).
    ///
    /// When a property is indexed, lookups by value are O(1) instead of O(n).
    /// Use [`create_property_index`] to enable indexing for a property.
    /// Lock order: 7
    pub(super) property_indexes:
        RwLock<FxHashMap<PropertyKey, DashMap<HashableValue, FxHashSet<NodeId>>>>,

    /// Vector indexes: "label:property" -> HNSW index.
    ///
    /// Created via [`GrafeoDB::create_vector_index`](grafeo_engine::GrafeoDB::create_vector_index).
    /// Lock order: 7 (same level as property_indexes, disjoint keys)
    #[cfg(feature = "vector-index")]
    pub(super) vector_indexes: RwLock<FxHashMap<String, Arc<VectorIndexKind>>>,

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

    /// Whether statistics need full recomputation (e.g., after rollback).
    pub(super) needs_stats_recompute: AtomicBool,

    /// Named graphs, each an independent `LpgStore` partition.
    /// Zero overhead for single-graph databases (empty HashMap).
    /// Lock order: 9 (after statistics)
    named_graphs: RwLock<FxHashMap<String, Arc<LpgStore>>>,

    /// Undo log for property mutations within transactions.
    ///
    /// Maps transaction IDs to a list of undo entries that capture the
    /// previous property values. On rollback, entries are replayed in
    /// reverse order to restore properties. On commit, the entries are
    /// simply discarded.
    /// Lock order: 10 (after named_graphs, independent of other locks)
    property_undo_log: RwLock<FxHashMap<TransactionId, Vec<PropertyUndoEntry>>>,

    /// Optional callback fired after rollback_transaction_properties completes.
    pub(crate) on_rollback_hook: parking_lot::Mutex<Option<Box<dyn Fn(TransactionId) + Send + Sync>>>,
    /// Optional callback fired after commit_transaction_properties completes.
    pub(crate) on_commit_hook: parking_lot::Mutex<Option<Box<dyn Fn(TransactionId) + Send + Sync>>>,
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
            label_registry: RwLock::new(LabelRegistry::new()),
            edge_type_to_id: RwLock::new(FxHashMap::default()),
            id_to_edge_type: RwLock::new(Vec::new()),
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
            live_edge_count: AtomicI64::new(0),
            edge_type_live_counts: RwLock::new(Vec::new()),
            statistics: RwLock::new(Arc::new(Statistics::new())),
            needs_stats_recompute: AtomicBool::new(false),
            named_graphs: RwLock::new(FxHashMap::default()),
            property_undo_log: RwLock::new(FxHashMap::default()),
            on_rollback_hook: parking_lot::Mutex::new(None),
            on_commit_hook: parking_lot::Mutex::new(None),
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

    /// Removes all data from the store, resetting it to an empty state.
    ///
    /// Acquires locks in the documented ordering to prevent deadlocks.
    /// After clearing, the store behaves as if freshly constructed.
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

        // Level 2: Catalogs
        self.label_registry.write().clear();
        {
            self.edge_type_to_id.write().clear();
            self.id_to_edge_type.write().clear();
        }

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
        self.needs_stats_recompute.store(false, Ordering::Release);

        // Level 5: Undo log
        self.property_undo_log.write().clear();
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
        let store = Arc::new(LpgStore::new()?);
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
        graphs.insert(name.to_string(), Arc::new(LpgStore::new()?));
        Ok(true)
    }

    /// Drops a named graph. Returns `false` if it did not exist.
    pub fn drop_graph(&self, name: &str) -> bool {
        self.named_graphs.write().remove(name).is_some()
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
        {
            let type_to_id = self.edge_type_to_id.read();
            if let Some(&id) = type_to_id.get(edge_type) {
                return id;
            }
        }

        let mut type_to_id = self.edge_type_to_id.write();
        let mut id_to_type = self.id_to_edge_type.write();

        // Double-check
        if let Some(&id) = type_to_id.get(edge_type) {
            return id;
        }

        let id = id_to_type.len() as u32;
        let edge_type: ArcStr = edge_type.into();
        type_to_id.insert(edge_type.clone(), id);
        id_to_type.push(edge_type);

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
        if type_id < counts.len() as u32 {
            counts[type_id as usize] -= 1;
        }
    }

    /// Sets the next node ID counter.
    ///
    /// Used by `HybridStore::open` to ensure overlay IDs do not collide with
    /// compact-store IDs.
    pub(crate) fn set_next_node_id(&self, id: u64) {
        self.next_node_id.store(id, Ordering::Release);
    }

    /// Sets the next edge ID counter.
    ///
    /// Used by `HybridStore::open` to ensure overlay IDs do not collide with
    /// compact-store IDs.
    pub(crate) fn set_next_edge_id(&self, id: u64) {
        self.next_edge_id.store(id, Ordering::Release);
    }

    /// Registers a callback that fires after `rollback_transaction_properties`.
    pub(crate) fn set_on_rollback_hook(&self, hook: Box<dyn Fn(TransactionId) + Send + Sync>) {
        *self.on_rollback_hook.lock() = Some(hook);
    }

    /// Registers a callback that fires after `commit_transaction_properties`.
    pub(crate) fn set_on_commit_hook(&self, hook: Box<dyn Fn(TransactionId) + Send + Sync>) {
        *self.on_commit_hook.lock() = Some(hook);
    }
}
