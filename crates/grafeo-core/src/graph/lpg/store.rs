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

use super::property::CompareOp;
use super::{Edge, EdgeRecord, Node, NodeRecord, PropertyStorage};
use crate::graph::Direction;
use crate::index::adjacency::ChunkedAdjacency;
use crate::index::zone_map::ZoneMapEntry;
use crate::statistics::{EdgeTypeStatistics, LabelStatistics, Statistics};
use arcstr::ArcStr;
use dashmap::DashMap;
#[cfg(not(feature = "tiered-storage"))]
use grafeo_common::mvcc::VersionChain;
use grafeo_common::types::{EdgeId, EpochId, HashableValue, NodeId, PropertyKey, TxId, Value};
use grafeo_common::utils::hash::{FxHashMap, FxHashSet};
use parking_lot::RwLock;
use std::cmp::Ordering as CmpOrdering;
#[cfg(any(feature = "tiered-storage", feature = "vector-index"))]
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

#[cfg(feature = "vector-index")]
use crate::index::vector::HnswIndex;

/// Compares two values for ordering (used for range checks).
fn compare_values_for_range(a: &Value, b: &Value) -> Option<CmpOrdering> {
    match (a, b) {
        (Value::Int64(a), Value::Int64(b)) => Some(a.cmp(b)),
        (Value::Float64(a), Value::Float64(b)) => a.partial_cmp(b),
        (Value::String(a), Value::String(b)) => Some(a.cmp(b)),
        (Value::Bool(a), Value::Bool(b)) => Some(a.cmp(b)),
        _ => None,
    }
}

/// Checks if a value is within a range.
fn value_in_range(
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

// Tiered storage imports
#[cfg(feature = "tiered-storage")]
use crate::storage::EpochStore;
#[cfg(feature = "tiered-storage")]
use grafeo_common::memory::arena::ArenaAllocator;
#[cfg(feature = "tiered-storage")]
use grafeo_common::mvcc::{ColdVersionRef, HotVersionRef, VersionIndex, VersionRef};

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
/// let store = LpgStore::new();
///
/// // Create a small social network
/// let alice = store.create_node(&["Person"]);
/// let bob = store.create_node(&["Person"]);
/// store.create_edge(alice, bob, "KNOWS");
///
/// // Traverse outgoing edges
/// for neighbor in store.neighbors(alice, Direction::Outgoing) {
///     println!("Alice knows node {:?}", neighbor);
/// }
/// ```
///
/// # Lock Ordering
///
/// `LpgStore` contains multiple `RwLock` fields that must be acquired in a
/// consistent order to prevent deadlocks. Always acquire locks in this order:
///
/// ## Level 1 - Entity Storage (mutually exclusive via feature flag)
/// 1. `nodes` / `node_versions`
/// 2. `edges` / `edge_versions`
///
/// ## Level 2 - Catalogs (acquire as pairs when writing)
/// 3. `label_to_id` + `id_to_label`
/// 4. `edge_type_to_id` + `id_to_edge_type`
///
/// ## Level 3 - Indexes
/// 5. `label_index`
/// 6. `node_labels`
/// 7. `property_indexes`
///
/// ## Level 4 - Statistics
/// 8. `statistics`
///
/// ## Level 5 - Nested Locks (internal to other structs)
/// 9. `PropertyStorage::columns` (via `node_properties`/`edge_properties`)
/// 10. `ChunkedAdjacency::lists` (via `forward_adj`/`backward_adj`)
///
/// ## Rules
/// - Catalog pairs must be acquired together when writing.
/// - Never hold entity locks while acquiring catalog locks in a different scope.
/// - Statistics lock is always last.
/// - Read locks are generally safe, but avoid read-to-write upgrades.
pub struct LpgStore {
    /// Configuration.
    #[allow(dead_code)]
    config: LpgStoreConfig,

    /// Node records indexed by NodeId, with version chains for MVCC.
    /// Used when `tiered-storage` feature is disabled.
    /// Lock order: 1
    #[cfg(not(feature = "tiered-storage"))]
    nodes: RwLock<FxHashMap<NodeId, VersionChain<NodeRecord>>>,

    /// Edge records indexed by EdgeId, with version chains for MVCC.
    /// Used when `tiered-storage` feature is disabled.
    /// Lock order: 2
    #[cfg(not(feature = "tiered-storage"))]
    edges: RwLock<FxHashMap<EdgeId, VersionChain<EdgeRecord>>>,

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
    arena_allocator: Arc<ArenaAllocator>,

    /// Node version indexes - store metadata and arena offsets.
    /// The actual NodeRecord data is stored in the arena.
    /// Lock order: 1
    #[cfg(feature = "tiered-storage")]
    node_versions: RwLock<FxHashMap<NodeId, VersionIndex>>,

    /// Edge version indexes - store metadata and arena offsets.
    /// The actual EdgeRecord data is stored in the arena.
    /// Lock order: 2
    #[cfg(feature = "tiered-storage")]
    edge_versions: RwLock<FxHashMap<EdgeId, VersionIndex>>,

    /// Cold storage for frozen epochs.
    /// Contains compressed epoch blocks for historical data.
    #[cfg(feature = "tiered-storage")]
    epoch_store: Arc<EpochStore>,

    /// Property storage for nodes.
    node_properties: PropertyStorage<NodeId>,

    /// Property storage for edges.
    edge_properties: PropertyStorage<EdgeId>,

    /// Label name to ID mapping.
    /// Lock order: 3 (acquire with id_to_label)
    label_to_id: RwLock<FxHashMap<ArcStr, u32>>,

    /// Label ID to name mapping.
    /// Lock order: 3 (acquire with label_to_id)
    id_to_label: RwLock<Vec<ArcStr>>,

    /// Edge type name to ID mapping.
    /// Lock order: 4 (acquire with id_to_edge_type)
    edge_type_to_id: RwLock<FxHashMap<ArcStr, u32>>,

    /// Edge type ID to name mapping.
    /// Lock order: 4 (acquire with edge_type_to_id)
    id_to_edge_type: RwLock<Vec<ArcStr>>,

    /// Forward adjacency lists (outgoing edges).
    forward_adj: ChunkedAdjacency,

    /// Backward adjacency lists (incoming edges).
    /// Only populated if config.backward_edges is true.
    backward_adj: Option<ChunkedAdjacency>,

    /// Label index: label_id -> set of node IDs.
    /// Lock order: 5
    label_index: RwLock<Vec<FxHashMap<NodeId, ()>>>,

    /// Node labels: node_id -> set of label IDs.
    /// Reverse mapping to efficiently get labels for a node.
    /// Lock order: 6
    node_labels: RwLock<FxHashMap<NodeId, FxHashSet<u32>>>,

    /// Property indexes: property_key -> (value -> set of node IDs).
    ///
    /// When a property is indexed, lookups by value are O(1) instead of O(n).
    /// Use [`create_property_index`] to enable indexing for a property.
    /// Lock order: 7
    property_indexes: RwLock<FxHashMap<PropertyKey, DashMap<HashableValue, FxHashSet<NodeId>>>>,

    /// Vector indexes: "label:property" -> HNSW index.
    ///
    /// Created via [`GrafeoDB::create_vector_index`](grafeo_engine::GrafeoDB::create_vector_index).
    /// Lock order: 7 (same level as property_indexes, disjoint keys)
    #[cfg(feature = "vector-index")]
    vector_indexes: RwLock<FxHashMap<String, Arc<HnswIndex>>>,

    /// Next node ID.
    next_node_id: AtomicU64,

    /// Next edge ID.
    next_edge_id: AtomicU64,

    /// Current epoch.
    current_epoch: AtomicU64,

    /// Statistics for cost-based optimization.
    /// Lock order: 8 (always last)
    statistics: RwLock<Statistics>,

    /// Whether statistics need recomputation after mutations.
    needs_stats_recompute: AtomicBool,
}

impl LpgStore {
    /// Creates a new LPG store with default configuration.
    #[must_use]
    pub fn new() -> Self {
        Self::with_config(LpgStoreConfig::default())
    }

    /// Creates a new LPG store with custom configuration.
    #[must_use]
    pub fn with_config(config: LpgStoreConfig) -> Self {
        let backward_adj = if config.backward_edges {
            Some(ChunkedAdjacency::new())
        } else {
            None
        };

        Self {
            #[cfg(not(feature = "tiered-storage"))]
            nodes: RwLock::new(FxHashMap::default()),
            #[cfg(not(feature = "tiered-storage"))]
            edges: RwLock::new(FxHashMap::default()),
            #[cfg(feature = "tiered-storage")]
            arena_allocator: Arc::new(ArenaAllocator::new()),
            #[cfg(feature = "tiered-storage")]
            node_versions: RwLock::new(FxHashMap::default()),
            #[cfg(feature = "tiered-storage")]
            edge_versions: RwLock::new(FxHashMap::default()),
            #[cfg(feature = "tiered-storage")]
            epoch_store: Arc::new(EpochStore::new()),
            node_properties: PropertyStorage::new(),
            edge_properties: PropertyStorage::new(),
            label_to_id: RwLock::new(FxHashMap::default()),
            id_to_label: RwLock::new(Vec::new()),
            edge_type_to_id: RwLock::new(FxHashMap::default()),
            id_to_edge_type: RwLock::new(Vec::new()),
            forward_adj: ChunkedAdjacency::new(),
            backward_adj,
            label_index: RwLock::new(Vec::new()),
            node_labels: RwLock::new(FxHashMap::default()),
            property_indexes: RwLock::new(FxHashMap::default()),
            #[cfg(feature = "vector-index")]
            vector_indexes: RwLock::new(FxHashMap::default()),
            next_node_id: AtomicU64::new(0),
            next_edge_id: AtomicU64::new(0),
            current_epoch: AtomicU64::new(0),
            statistics: RwLock::new(Statistics::new()),
            needs_stats_recompute: AtomicBool::new(true),
            config,
        }
    }

    /// Returns the current epoch.
    #[must_use]
    pub fn current_epoch(&self) -> EpochId {
        EpochId::new(self.current_epoch.load(Ordering::Acquire))
    }

    /// Creates a new epoch.
    pub fn new_epoch(&self) -> EpochId {
        let id = self.current_epoch.fetch_add(1, Ordering::AcqRel) + 1;
        EpochId::new(id)
    }

    // === Node Operations ===

    /// Creates a new node with the given labels.
    ///
    /// Uses the system transaction for non-transactional operations.
    pub fn create_node(&self, labels: &[&str]) -> NodeId {
        self.needs_stats_recompute.store(true, Ordering::Relaxed);
        self.create_node_versioned(labels, self.current_epoch(), TxId::SYSTEM)
    }

    /// Creates a new node with the given labels within a transaction context.
    #[cfg(not(feature = "tiered-storage"))]
    pub fn create_node_versioned(&self, labels: &[&str], epoch: EpochId, tx_id: TxId) -> NodeId {
        let id = NodeId::new(self.next_node_id.fetch_add(1, Ordering::Relaxed));

        let mut record = NodeRecord::new(id, epoch);
        record.set_label_count(labels.len() as u16);

        // Store labels in node_labels map and label_index
        let mut node_label_set = FxHashSet::default();
        for label in labels {
            let label_id = self.get_or_create_label_id(*label);
            node_label_set.insert(label_id);

            // Update label index
            let mut index = self.label_index.write();
            while index.len() <= label_id as usize {
                index.push(FxHashMap::default());
            }
            index[label_id as usize].insert(id, ());
        }

        // Store node's labels
        self.node_labels.write().insert(id, node_label_set);

        // Create version chain with initial version
        let chain = VersionChain::with_initial(record, epoch, tx_id);
        self.nodes.write().insert(id, chain);
        id
    }

    /// Creates a new node with the given labels within a transaction context.
    /// (Tiered storage version: stores data in arena, metadata in VersionIndex)
    #[cfg(feature = "tiered-storage")]
    pub fn create_node_versioned(&self, labels: &[&str], epoch: EpochId, tx_id: TxId) -> NodeId {
        let id = NodeId::new(self.next_node_id.fetch_add(1, Ordering::Relaxed));

        let mut record = NodeRecord::new(id, epoch);
        record.set_label_count(labels.len() as u16);

        // Store labels in node_labels map and label_index
        let mut node_label_set = FxHashSet::default();
        for label in labels {
            let label_id = self.get_or_create_label_id(*label);
            node_label_set.insert(label_id);

            // Update label index
            let mut index = self.label_index.write();
            while index.len() <= label_id as usize {
                index.push(FxHashMap::default());
            }
            index[label_id as usize].insert(id, ());
        }

        // Store node's labels
        self.node_labels.write().insert(id, node_label_set);

        // Allocate record in arena and get offset (create epoch if needed)
        let arena = self.arena_allocator.arena_or_create(epoch);
        let (offset, _stored) = arena.alloc_value_with_offset(record);

        // Create HotVersionRef pointing to arena data
        let hot_ref = HotVersionRef::new(epoch, offset, tx_id);

        // Create or update version index
        let mut versions = self.node_versions.write();
        if let Some(index) = versions.get_mut(&id) {
            index.add_hot(hot_ref);
        } else {
            versions.insert(id, VersionIndex::with_initial(hot_ref));
        }

        id
    }

    /// Creates a new node with labels and properties.
    pub fn create_node_with_props(
        &self,
        labels: &[&str],
        properties: impl IntoIterator<Item = (impl Into<PropertyKey>, impl Into<Value>)>,
    ) -> NodeId {
        self.create_node_with_props_versioned(
            labels,
            properties,
            self.current_epoch(),
            TxId::SYSTEM,
        )
    }

    /// Creates a new node with labels and properties within a transaction context.
    #[cfg(not(feature = "tiered-storage"))]
    pub fn create_node_with_props_versioned(
        &self,
        labels: &[&str],
        properties: impl IntoIterator<Item = (impl Into<PropertyKey>, impl Into<Value>)>,
        epoch: EpochId,
        tx_id: TxId,
    ) -> NodeId {
        let id = self.create_node_versioned(labels, epoch, tx_id);

        for (key, value) in properties {
            let prop_key: PropertyKey = key.into();
            let prop_value: Value = value.into();
            // Update property index before setting the property
            self.update_property_index_on_set(id, &prop_key, &prop_value);
            self.node_properties.set(id, prop_key, prop_value);
        }

        // Update props_count in record
        let count = self.node_properties.get_all(id).len() as u16;
        if let Some(chain) = self.nodes.write().get_mut(&id) {
            if let Some(record) = chain.latest_mut() {
                record.props_count = count;
            }
        }

        id
    }

    /// Creates a new node with labels and properties within a transaction context.
    /// (Tiered storage version)
    #[cfg(feature = "tiered-storage")]
    pub fn create_node_with_props_versioned(
        &self,
        labels: &[&str],
        properties: impl IntoIterator<Item = (impl Into<PropertyKey>, impl Into<Value>)>,
        epoch: EpochId,
        tx_id: TxId,
    ) -> NodeId {
        let id = self.create_node_versioned(labels, epoch, tx_id);

        for (key, value) in properties {
            let prop_key: PropertyKey = key.into();
            let prop_value: Value = value.into();
            // Update property index before setting the property
            self.update_property_index_on_set(id, &prop_key, &prop_value);
            self.node_properties.set(id, prop_key, prop_value);
        }

        // Note: props_count in record is not updated for tiered storage.
        // The record is immutable once allocated in the arena.

        id
    }

    /// Gets a node by ID (latest visible version).
    #[must_use]
    pub fn get_node(&self, id: NodeId) -> Option<Node> {
        self.get_node_at_epoch(id, self.current_epoch())
    }

    /// Gets a node by ID at a specific epoch.
    #[must_use]
    #[cfg(not(feature = "tiered-storage"))]
    pub fn get_node_at_epoch(&self, id: NodeId, epoch: EpochId) -> Option<Node> {
        let nodes = self.nodes.read();
        let chain = nodes.get(&id)?;
        let record = chain.visible_at(epoch)?;

        if record.is_deleted() {
            return None;
        }

        let mut node = Node::new(id);

        // Get labels from node_labels map
        let id_to_label = self.id_to_label.read();
        let node_labels = self.node_labels.read();
        if let Some(label_ids) = node_labels.get(&id) {
            for &label_id in label_ids {
                if let Some(label) = id_to_label.get(label_id as usize) {
                    node.labels.push(label.clone());
                }
            }
        }

        // Get properties
        node.properties = self.node_properties.get_all(id).into_iter().collect();

        Some(node)
    }

    /// Gets a node by ID at a specific epoch.
    /// (Tiered storage version: reads from arena via VersionIndex)
    #[must_use]
    #[cfg(feature = "tiered-storage")]
    pub fn get_node_at_epoch(&self, id: NodeId, epoch: EpochId) -> Option<Node> {
        let versions = self.node_versions.read();
        let index = versions.get(&id)?;
        let version_ref = index.visible_at(epoch)?;

        // Read the record from arena
        let record = self.read_node_record(&version_ref)?;

        if record.is_deleted() {
            return None;
        }

        let mut node = Node::new(id);

        // Get labels from node_labels map
        let id_to_label = self.id_to_label.read();
        let node_labels = self.node_labels.read();
        if let Some(label_ids) = node_labels.get(&id) {
            for &label_id in label_ids {
                if let Some(label) = id_to_label.get(label_id as usize) {
                    node.labels.push(label.clone());
                }
            }
        }

        // Get properties
        node.properties = self.node_properties.get_all(id).into_iter().collect();

        Some(node)
    }

    /// Gets a node visible to a specific transaction.
    #[must_use]
    #[cfg(not(feature = "tiered-storage"))]
    pub fn get_node_versioned(&self, id: NodeId, epoch: EpochId, tx_id: TxId) -> Option<Node> {
        let nodes = self.nodes.read();
        let chain = nodes.get(&id)?;
        let record = chain.visible_to(epoch, tx_id)?;

        if record.is_deleted() {
            return None;
        }

        let mut node = Node::new(id);

        // Get labels from node_labels map
        let id_to_label = self.id_to_label.read();
        let node_labels = self.node_labels.read();
        if let Some(label_ids) = node_labels.get(&id) {
            for &label_id in label_ids {
                if let Some(label) = id_to_label.get(label_id as usize) {
                    node.labels.push(label.clone());
                }
            }
        }

        // Get properties
        node.properties = self.node_properties.get_all(id).into_iter().collect();

        Some(node)
    }

    /// Gets a node visible to a specific transaction.
    /// (Tiered storage version: reads from arena via VersionIndex)
    #[must_use]
    #[cfg(feature = "tiered-storage")]
    pub fn get_node_versioned(&self, id: NodeId, epoch: EpochId, tx_id: TxId) -> Option<Node> {
        let versions = self.node_versions.read();
        let index = versions.get(&id)?;
        let version_ref = index.visible_to(epoch, tx_id)?;

        // Read the record from arena
        let record = self.read_node_record(&version_ref)?;

        if record.is_deleted() {
            return None;
        }

        let mut node = Node::new(id);

        // Get labels from node_labels map
        let id_to_label = self.id_to_label.read();
        let node_labels = self.node_labels.read();
        if let Some(label_ids) = node_labels.get(&id) {
            for &label_id in label_ids {
                if let Some(label) = id_to_label.get(label_id as usize) {
                    node.labels.push(label.clone());
                }
            }
        }

        // Get properties
        node.properties = self.node_properties.get_all(id).into_iter().collect();

        Some(node)
    }

    /// Reads a NodeRecord from arena (hot) or epoch store (cold) using a VersionRef.
    #[cfg(feature = "tiered-storage")]
    #[allow(unsafe_code)]
    fn read_node_record(&self, version_ref: &VersionRef) -> Option<NodeRecord> {
        match version_ref {
            VersionRef::Hot(hot_ref) => {
                let arena = self.arena_allocator.arena(hot_ref.epoch);
                // SAFETY: The offset was returned by alloc_value_with_offset for a NodeRecord
                let record: &NodeRecord = unsafe { arena.read_at(hot_ref.arena_offset) };
                Some(*record)
            }
            VersionRef::Cold(cold_ref) => {
                // Read from compressed epoch store
                self.epoch_store
                    .get_node(cold_ref.epoch, cold_ref.block_offset, cold_ref.length)
            }
        }
    }

    /// Deletes a node and all its edges (using latest epoch).
    pub fn delete_node(&self, id: NodeId) -> bool {
        self.needs_stats_recompute.store(true, Ordering::Relaxed);
        self.delete_node_at_epoch(id, self.current_epoch())
    }

    /// Deletes a node at a specific epoch.
    #[cfg(not(feature = "tiered-storage"))]
    pub fn delete_node_at_epoch(&self, id: NodeId, epoch: EpochId) -> bool {
        let mut nodes = self.nodes.write();
        if let Some(chain) = nodes.get_mut(&id) {
            // Check if visible at this epoch (not already deleted)
            if let Some(record) = chain.visible_at(epoch) {
                if record.is_deleted() {
                    return false;
                }
            } else {
                // Not visible at this epoch (already deleted or doesn't exist)
                return false;
            }

            // Mark the version chain as deleted at this epoch
            chain.mark_deleted(epoch);

            // Remove from label index using node_labels map
            let mut index = self.label_index.write();
            let mut node_labels = self.node_labels.write();
            if let Some(label_ids) = node_labels.remove(&id) {
                for label_id in label_ids {
                    if let Some(set) = index.get_mut(label_id as usize) {
                        set.remove(&id);
                    }
                }
            }

            // Remove properties
            drop(nodes); // Release lock before removing properties
            drop(index);
            drop(node_labels);
            self.node_properties.remove_all(id);

            // Note: Caller should use delete_node_edges() first if detach is needed

            true
        } else {
            false
        }
    }

    /// Deletes a node at a specific epoch.
    /// (Tiered storage version)
    #[cfg(feature = "tiered-storage")]
    pub fn delete_node_at_epoch(&self, id: NodeId, epoch: EpochId) -> bool {
        let mut versions = self.node_versions.write();
        if let Some(index) = versions.get_mut(&id) {
            // Check if visible at this epoch
            if let Some(version_ref) = index.visible_at(epoch) {
                if let Some(record) = self.read_node_record(&version_ref) {
                    if record.is_deleted() {
                        return false;
                    }
                } else {
                    return false;
                }
            } else {
                return false;
            }

            // Mark as deleted in version index
            index.mark_deleted(epoch);

            // Remove from label index using node_labels map
            let mut label_index = self.label_index.write();
            let mut node_labels = self.node_labels.write();
            if let Some(label_ids) = node_labels.remove(&id) {
                for label_id in label_ids {
                    if let Some(set) = label_index.get_mut(label_id as usize) {
                        set.remove(&id);
                    }
                }
            }

            // Remove properties
            drop(versions);
            drop(label_index);
            drop(node_labels);
            self.node_properties.remove_all(id);

            true
        } else {
            false
        }
    }

    /// Deletes all edges connected to a node (implements DETACH DELETE).
    ///
    /// Call this before `delete_node()` if you want to remove a node that
    /// has edges. Grafeo doesn't auto-delete edges - you have to be explicit.
    #[cfg(not(feature = "tiered-storage"))]
    pub fn delete_node_edges(&self, node_id: NodeId) {
        // Get outgoing edges
        let outgoing: Vec<EdgeId> = self
            .forward_adj
            .edges_from(node_id)
            .into_iter()
            .map(|(_, edge_id)| edge_id)
            .collect();

        // Get incoming edges
        let incoming: Vec<EdgeId> = if let Some(ref backward) = self.backward_adj {
            backward
                .edges_from(node_id)
                .into_iter()
                .map(|(_, edge_id)| edge_id)
                .collect()
        } else {
            // No backward adjacency - scan all edges
            let epoch = self.current_epoch();
            self.edges
                .read()
                .iter()
                .filter_map(|(id, chain)| {
                    chain.visible_at(epoch).and_then(|r| {
                        if !r.is_deleted() && r.dst == node_id {
                            Some(*id)
                        } else {
                            None
                        }
                    })
                })
                .collect()
        };

        // Delete all edges
        for edge_id in outgoing.into_iter().chain(incoming) {
            self.delete_edge(edge_id);
        }
    }

    /// Deletes all edges connected to a node (implements DETACH DELETE).
    /// (Tiered storage version)
    #[cfg(feature = "tiered-storage")]
    pub fn delete_node_edges(&self, node_id: NodeId) {
        // Get outgoing edges
        let outgoing: Vec<EdgeId> = self
            .forward_adj
            .edges_from(node_id)
            .into_iter()
            .map(|(_, edge_id)| edge_id)
            .collect();

        // Get incoming edges
        let incoming: Vec<EdgeId> = if let Some(ref backward) = self.backward_adj {
            backward
                .edges_from(node_id)
                .into_iter()
                .map(|(_, edge_id)| edge_id)
                .collect()
        } else {
            // No backward adjacency - scan all edges
            let epoch = self.current_epoch();
            let versions = self.edge_versions.read();
            versions
                .iter()
                .filter_map(|(id, index)| {
                    index.visible_at(epoch).and_then(|vref| {
                        self.read_edge_record(&vref).and_then(|r| {
                            if !r.is_deleted() && r.dst == node_id {
                                Some(*id)
                            } else {
                                None
                            }
                        })
                    })
                })
                .collect()
        };

        // Delete all edges
        for edge_id in outgoing.into_iter().chain(incoming) {
            self.delete_edge(edge_id);
        }
    }

    /// Sets a property on a node.
    #[cfg(not(feature = "tiered-storage"))]
    pub fn set_node_property(&self, id: NodeId, key: &str, value: Value) {
        let prop_key: PropertyKey = key.into();

        // Update property index before setting the property (needs to read old value)
        self.update_property_index_on_set(id, &prop_key, &value);

        self.node_properties.set(id, prop_key, value);

        // Update props_count in record
        let count = self.node_properties.get_all(id).len() as u16;
        if let Some(chain) = self.nodes.write().get_mut(&id) {
            if let Some(record) = chain.latest_mut() {
                record.props_count = count;
            }
        }
    }

    /// Sets a property on a node.
    /// (Tiered storage version: properties stored separately, record is immutable)
    #[cfg(feature = "tiered-storage")]
    pub fn set_node_property(&self, id: NodeId, key: &str, value: Value) {
        let prop_key: PropertyKey = key.into();

        // Update property index before setting the property (needs to read old value)
        self.update_property_index_on_set(id, &prop_key, &value);

        self.node_properties.set(id, prop_key, value);
        // Note: props_count in record is not updated for tiered storage.
        // The record is immutable once allocated in the arena.
        // Property count can be derived from PropertyStorage if needed.
    }

    /// Sets a property on an edge.
    pub fn set_edge_property(&self, id: EdgeId, key: &str, value: Value) {
        self.edge_properties.set(id, key.into(), value);
    }

    /// Removes a property from a node.
    ///
    /// Returns the previous value if it existed, or None if the property didn't exist.
    #[cfg(not(feature = "tiered-storage"))]
    pub fn remove_node_property(&self, id: NodeId, key: &str) -> Option<Value> {
        let prop_key: PropertyKey = key.into();

        // Update property index before removing (needs to read old value)
        self.update_property_index_on_remove(id, &prop_key);

        let result = self.node_properties.remove(id, &prop_key);

        // Update props_count in record
        let count = self.node_properties.get_all(id).len() as u16;
        if let Some(chain) = self.nodes.write().get_mut(&id) {
            if let Some(record) = chain.latest_mut() {
                record.props_count = count;
            }
        }

        result
    }

    /// Removes a property from a node.
    /// (Tiered storage version)
    #[cfg(feature = "tiered-storage")]
    pub fn remove_node_property(&self, id: NodeId, key: &str) -> Option<Value> {
        let prop_key: PropertyKey = key.into();

        // Update property index before removing (needs to read old value)
        self.update_property_index_on_remove(id, &prop_key);

        self.node_properties.remove(id, &prop_key)
        // Note: props_count in record is not updated for tiered storage.
    }

    /// Removes a property from an edge.
    ///
    /// Returns the previous value if it existed, or None if the property didn't exist.
    pub fn remove_edge_property(&self, id: EdgeId, key: &str) -> Option<Value> {
        self.edge_properties.remove(id, &key.into())
    }

    /// Gets a single property from a node without loading all properties.
    ///
    /// This is O(1) vs O(properties) for `get_node().get_property()`.
    /// Use this for filter predicates where you only need one property value.
    ///
    /// # Example
    ///
    /// ```ignore
    /// // Fast: Direct single-property lookup
    /// let age = store.get_node_property(node_id, "age");
    ///
    /// // Slow: Loads all properties, then extracts one
    /// let age = store.get_node(node_id).and_then(|n| n.get_property("age").cloned());
    /// ```
    #[must_use]
    pub fn get_node_property(&self, id: NodeId, key: &PropertyKey) -> Option<Value> {
        self.node_properties.get(id, key)
    }

    /// Gets a single property from an edge without loading all properties.
    ///
    /// This is O(1) vs O(properties) for `get_edge().get_property()`.
    #[must_use]
    pub fn get_edge_property(&self, id: EdgeId, key: &PropertyKey) -> Option<Value> {
        self.edge_properties.get(id, key)
    }

    // === Batch Property Operations ===

    /// Gets a property for multiple nodes in a single batch operation.
    ///
    /// More efficient than calling [`Self::get_node_property`] in a loop because it
    /// reduces lock overhead and enables better cache utilization.
    ///
    /// # Example
    ///
    /// ```
    /// use grafeo_core::graph::lpg::LpgStore;
    /// use grafeo_common::types::{NodeId, PropertyKey, Value};
    ///
    /// let store = LpgStore::new();
    /// let n1 = store.create_node(&["Person"]);
    /// let n2 = store.create_node(&["Person"]);
    /// store.set_node_property(n1, "age", Value::from(25i64));
    /// store.set_node_property(n2, "age", Value::from(30i64));
    ///
    /// let ages = store.get_node_property_batch(&[n1, n2], &PropertyKey::new("age"));
    /// assert_eq!(ages, vec![Some(Value::from(25i64)), Some(Value::from(30i64))]);
    /// ```
    #[must_use]
    pub fn get_node_property_batch(&self, ids: &[NodeId], key: &PropertyKey) -> Vec<Option<Value>> {
        self.node_properties.get_batch(ids, key)
    }

    /// Gets all properties for multiple nodes in a single batch operation.
    ///
    /// Returns a vector of property maps, one per node ID (empty map if no properties).
    /// More efficient than calling [`Self::get_node`] in a loop.
    #[must_use]
    pub fn get_nodes_properties_batch(&self, ids: &[NodeId]) -> Vec<FxHashMap<PropertyKey, Value>> {
        self.node_properties.get_all_batch(ids)
    }

    /// Gets selected properties for multiple nodes (projection pushdown).
    ///
    /// This is more efficient than [`Self::get_nodes_properties_batch`] when you only
    /// need a subset of properties. It only iterates the requested columns instead of
    /// all columns.
    ///
    /// **Use this for**: Queries with explicit projections like `RETURN n.name, n.age`
    /// instead of `RETURN n` (which requires all properties).
    ///
    /// # Example
    ///
    /// ```
    /// use grafeo_core::graph::lpg::LpgStore;
    /// use grafeo_common::types::{PropertyKey, Value};
    ///
    /// let store = LpgStore::new();
    /// let n1 = store.create_node(&["Person"]);
    /// store.set_node_property(n1, "name", Value::from("Alice"));
    /// store.set_node_property(n1, "age", Value::from(30i64));
    /// store.set_node_property(n1, "email", Value::from("alice@example.com"));
    ///
    /// // Only fetch name and age (faster than get_nodes_properties_batch)
    /// let keys = vec![PropertyKey::new("name"), PropertyKey::new("age")];
    /// let props = store.get_nodes_properties_selective_batch(&[n1], &keys);
    ///
    /// assert_eq!(props[0].len(), 2); // Only name and age, not email
    /// ```
    #[must_use]
    pub fn get_nodes_properties_selective_batch(
        &self,
        ids: &[NodeId],
        keys: &[PropertyKey],
    ) -> Vec<FxHashMap<PropertyKey, Value>> {
        self.node_properties.get_selective_batch(ids, keys)
    }

    /// Gets selected properties for multiple edges (projection pushdown).
    ///
    /// Edge-property version of [`Self::get_nodes_properties_selective_batch`].
    #[must_use]
    pub fn get_edges_properties_selective_batch(
        &self,
        ids: &[EdgeId],
        keys: &[PropertyKey],
    ) -> Vec<FxHashMap<PropertyKey, Value>> {
        self.edge_properties.get_selective_batch(ids, keys)
    }

    /// Finds nodes where a property value is in a range.
    ///
    /// This is useful for queries like `n.age > 30` or `n.price BETWEEN 10 AND 100`.
    /// Uses zone maps to skip scanning when the range definitely doesn't match.
    ///
    /// # Arguments
    ///
    /// * `property` - The property to check
    /// * `min` - Optional lower bound (None for unbounded)
    /// * `max` - Optional upper bound (None for unbounded)
    /// * `min_inclusive` - Whether lower bound is inclusive (>= vs >)
    /// * `max_inclusive` - Whether upper bound is inclusive (<= vs <)
    ///
    /// # Example
    ///
    /// ```
    /// use grafeo_core::graph::lpg::LpgStore;
    /// use grafeo_common::types::Value;
    ///
    /// let store = LpgStore::new();
    /// let n1 = store.create_node(&["Person"]);
    /// let n2 = store.create_node(&["Person"]);
    /// store.set_node_property(n1, "age", Value::from(25i64));
    /// store.set_node_property(n2, "age", Value::from(35i64));
    ///
    /// // Find nodes where age > 30
    /// let result = store.find_nodes_in_range(
    ///     "age",
    ///     Some(&Value::from(30i64)),
    ///     None,
    ///     false, // exclusive lower bound
    ///     true,  // inclusive upper bound (doesn't matter since None)
    /// );
    /// assert_eq!(result.len(), 1); // Only n2 matches
    /// ```
    #[must_use]
    pub fn find_nodes_in_range(
        &self,
        property: &str,
        min: Option<&Value>,
        max: Option<&Value>,
        min_inclusive: bool,
        max_inclusive: bool,
    ) -> Vec<NodeId> {
        let key = PropertyKey::new(property);

        // Check zone map first - if no values could match, return empty
        if !self
            .node_properties
            .might_match_range(&key, min, max, min_inclusive, max_inclusive)
        {
            return Vec::new();
        }

        // Scan all nodes and filter by range
        self.node_ids()
            .into_iter()
            .filter(|&node_id| {
                self.node_properties
                    .get(node_id, &key)
                    .is_some_and(|v| value_in_range(&v, min, max, min_inclusive, max_inclusive))
            })
            .collect()
    }

    /// Finds nodes matching multiple property equality conditions.
    ///
    /// This is more efficient than intersecting multiple single-property lookups
    /// because it can use indexes when available and short-circuits on the first
    /// miss.
    ///
    /// # Example
    ///
    /// ```
    /// use grafeo_core::graph::lpg::LpgStore;
    /// use grafeo_common::types::Value;
    ///
    /// let store = LpgStore::new();
    /// let alice = store.create_node(&["Person"]);
    /// store.set_node_property(alice, "name", Value::from("Alice"));
    /// store.set_node_property(alice, "city", Value::from("NYC"));
    ///
    /// // Find nodes where name = "Alice" AND city = "NYC"
    /// let matches = store.find_nodes_by_properties(&[
    ///     ("name", Value::from("Alice")),
    ///     ("city", Value::from("NYC")),
    /// ]);
    /// assert!(matches.contains(&alice));
    /// ```
    #[must_use]
    pub fn find_nodes_by_properties(&self, conditions: &[(&str, Value)]) -> Vec<NodeId> {
        if conditions.is_empty() {
            return self.node_ids();
        }

        // Find the most selective condition (smallest result set) to start
        // If any condition has an index, use that first
        let mut best_start: Option<(usize, Vec<NodeId>)> = None;
        let indexes = self.property_indexes.read();

        for (i, (prop, value)) in conditions.iter().enumerate() {
            let key = PropertyKey::new(*prop);
            let hv = HashableValue::new(value.clone());

            if let Some(index) = indexes.get(&key) {
                let matches: Vec<NodeId> = index
                    .get(&hv)
                    .map(|nodes| nodes.iter().copied().collect())
                    .unwrap_or_default();

                // Short-circuit if any indexed condition has no matches
                if matches.is_empty() {
                    return Vec::new();
                }

                // Use smallest indexed result as starting point
                if best_start
                    .as_ref()
                    .is_none_or(|(_, best)| matches.len() < best.len())
                {
                    best_start = Some((i, matches));
                }
            }
        }
        drop(indexes);

        // Start from best indexed result or fall back to full node scan
        let (start_idx, mut candidates) = best_start.unwrap_or_else(|| {
            // No indexes available, start with first condition via full scan
            let (prop, value) = &conditions[0];
            (0, self.find_nodes_by_property(prop, value))
        });

        // Filter candidates through remaining conditions
        for (i, (prop, value)) in conditions.iter().enumerate() {
            if i == start_idx {
                continue;
            }

            let key = PropertyKey::new(*prop);
            candidates.retain(|&node_id| {
                self.node_properties
                    .get(node_id, &key)
                    .is_some_and(|v| v == *value)
            });

            // Short-circuit if no candidates remain
            if candidates.is_empty() {
                return Vec::new();
            }
        }

        candidates
    }

    // === Property Index Operations ===

    /// Creates an index on a node property for O(1) lookups by value.
    ///
    /// After creating an index, calls to [`Self::find_nodes_by_property`] will be
    /// O(1) instead of O(n) for this property. The index is automatically
    /// maintained when properties are set or removed.
    ///
    /// # Example
    ///
    /// ```
    /// use grafeo_core::graph::lpg::LpgStore;
    /// use grafeo_common::types::Value;
    ///
    /// let store = LpgStore::new();
    ///
    /// // Create nodes with an 'id' property
    /// let alice = store.create_node(&["Person"]);
    /// store.set_node_property(alice, "id", Value::from("alice_123"));
    ///
    /// // Create an index on the 'id' property
    /// store.create_property_index("id");
    ///
    /// // Now lookups by 'id' are O(1)
    /// let found = store.find_nodes_by_property("id", &Value::from("alice_123"));
    /// assert!(found.contains(&alice));
    /// ```
    pub fn create_property_index(&self, property: &str) {
        let key = PropertyKey::new(property);

        let mut indexes = self.property_indexes.write();
        if indexes.contains_key(&key) {
            return; // Already indexed
        }

        // Create the index and populate it with existing data
        let index: DashMap<HashableValue, FxHashSet<NodeId>> = DashMap::new();

        // Scan all nodes to build the index
        for node_id in self.node_ids() {
            if let Some(value) = self.node_properties.get(node_id, &key) {
                let hv = HashableValue::new(value);
                index.entry(hv).or_default().insert(node_id);
            }
        }

        indexes.insert(key, index);
    }

    /// Drops an index on a node property.
    ///
    /// Returns `true` if the index existed and was removed.
    pub fn drop_property_index(&self, property: &str) -> bool {
        let key = PropertyKey::new(property);
        self.property_indexes.write().remove(&key).is_some()
    }

    /// Returns `true` if the property has an index.
    #[must_use]
    pub fn has_property_index(&self, property: &str) -> bool {
        let key = PropertyKey::new(property);
        self.property_indexes.read().contains_key(&key)
    }

    /// Stores a vector index for a label+property pair.
    #[cfg(feature = "vector-index")]
    pub fn add_vector_index(&self, label: &str, property: &str, index: Arc<HnswIndex>) {
        let key = format!("{label}:{property}");
        self.vector_indexes.write().insert(key, index);
    }

    /// Retrieves the vector index for a label+property pair.
    #[cfg(feature = "vector-index")]
    #[must_use]
    pub fn get_vector_index(&self, label: &str, property: &str) -> Option<Arc<HnswIndex>> {
        let key = format!("{label}:{property}");
        self.vector_indexes.read().get(&key).cloned()
    }

    /// Finds all nodes that have a specific property value.
    ///
    /// If the property is indexed, this is O(1). Otherwise, it scans all nodes
    /// which is O(n). Use [`Self::create_property_index`] for frequently queried properties.
    ///
    /// # Example
    ///
    /// ```
    /// use grafeo_core::graph::lpg::LpgStore;
    /// use grafeo_common::types::Value;
    ///
    /// let store = LpgStore::new();
    /// store.create_property_index("city"); // Optional but makes lookups fast
    ///
    /// let alice = store.create_node(&["Person"]);
    /// let bob = store.create_node(&["Person"]);
    /// store.set_node_property(alice, "city", Value::from("NYC"));
    /// store.set_node_property(bob, "city", Value::from("NYC"));
    ///
    /// let nyc_people = store.find_nodes_by_property("city", &Value::from("NYC"));
    /// assert_eq!(nyc_people.len(), 2);
    /// ```
    #[must_use]
    pub fn find_nodes_by_property(&self, property: &str, value: &Value) -> Vec<NodeId> {
        let key = PropertyKey::new(property);
        let hv = HashableValue::new(value.clone());

        // Try indexed lookup first
        let indexes = self.property_indexes.read();
        if let Some(index) = indexes.get(&key) {
            if let Some(nodes) = index.get(&hv) {
                return nodes.iter().copied().collect();
            }
            return Vec::new();
        }
        drop(indexes);

        // Fall back to full scan
        self.node_ids()
            .into_iter()
            .filter(|&node_id| {
                self.node_properties
                    .get(node_id, &key)
                    .is_some_and(|v| v == *value)
            })
            .collect()
    }

    /// Updates property indexes when a property is set.
    fn update_property_index_on_set(&self, node_id: NodeId, key: &PropertyKey, new_value: &Value) {
        let indexes = self.property_indexes.read();
        if let Some(index) = indexes.get(key) {
            // Get old value to remove from index
            if let Some(old_value) = self.node_properties.get(node_id, key) {
                let old_hv = HashableValue::new(old_value);
                if let Some(mut nodes) = index.get_mut(&old_hv) {
                    nodes.remove(&node_id);
                    if nodes.is_empty() {
                        drop(nodes);
                        index.remove(&old_hv);
                    }
                }
            }

            // Add new value to index
            let new_hv = HashableValue::new(new_value.clone());
            index
                .entry(new_hv)
                .or_insert_with(FxHashSet::default)
                .insert(node_id);
        }
    }

    /// Updates property indexes when a property is removed.
    fn update_property_index_on_remove(&self, node_id: NodeId, key: &PropertyKey) {
        let indexes = self.property_indexes.read();
        if let Some(index) = indexes.get(key) {
            // Get old value to remove from index
            if let Some(old_value) = self.node_properties.get(node_id, key) {
                let old_hv = HashableValue::new(old_value);
                if let Some(mut nodes) = index.get_mut(&old_hv) {
                    nodes.remove(&node_id);
                    if nodes.is_empty() {
                        drop(nodes);
                        index.remove(&old_hv);
                    }
                }
            }
        }
    }

    /// Adds a label to a node.
    ///
    /// Returns true if the label was added, false if the node doesn't exist
    /// or already has the label.
    #[cfg(not(feature = "tiered-storage"))]
    pub fn add_label(&self, node_id: NodeId, label: &str) -> bool {
        let epoch = self.current_epoch();

        // Check if node exists
        let nodes = self.nodes.read();
        if let Some(chain) = nodes.get(&node_id) {
            if chain.visible_at(epoch).map_or(true, |r| r.is_deleted()) {
                return false;
            }
        } else {
            return false;
        }
        drop(nodes);

        // Get or create label ID
        let label_id = self.get_or_create_label_id(label);

        // Add to node_labels map
        let mut node_labels = self.node_labels.write();
        let label_set = node_labels
            .entry(node_id)
            .or_insert_with(FxHashSet::default);

        if label_set.contains(&label_id) {
            return false; // Already has this label
        }

        label_set.insert(label_id);
        drop(node_labels);

        // Add to label_index
        let mut index = self.label_index.write();
        if (label_id as usize) >= index.len() {
            index.resize(label_id as usize + 1, FxHashMap::default());
        }
        index[label_id as usize].insert(node_id, ());

        // Update label count in node record
        if let Some(chain) = self.nodes.write().get_mut(&node_id) {
            if let Some(record) = chain.latest_mut() {
                let count = self.node_labels.read().get(&node_id).map_or(0, |s| s.len());
                record.set_label_count(count as u16);
            }
        }

        true
    }

    /// Adds a label to a node.
    /// (Tiered storage version)
    #[cfg(feature = "tiered-storage")]
    pub fn add_label(&self, node_id: NodeId, label: &str) -> bool {
        let epoch = self.current_epoch();

        // Check if node exists
        let versions = self.node_versions.read();
        if let Some(index) = versions.get(&node_id) {
            if let Some(vref) = index.visible_at(epoch) {
                if let Some(record) = self.read_node_record(&vref) {
                    if record.is_deleted() {
                        return false;
                    }
                } else {
                    return false;
                }
            } else {
                return false;
            }
        } else {
            return false;
        }
        drop(versions);

        // Get or create label ID
        let label_id = self.get_or_create_label_id(label);

        // Add to node_labels map
        let mut node_labels = self.node_labels.write();
        let label_set = node_labels.entry(node_id).or_default();

        if label_set.contains(&label_id) {
            return false; // Already has this label
        }

        label_set.insert(label_id);
        drop(node_labels);

        // Add to label_index
        let mut index = self.label_index.write();
        if (label_id as usize) >= index.len() {
            index.resize(label_id as usize + 1, FxHashMap::default());
        }
        index[label_id as usize].insert(node_id, ());

        // Note: label_count in record is not updated for tiered storage.
        // The record is immutable once allocated in the arena.

        true
    }

    /// Removes a label from a node.
    ///
    /// Returns true if the label was removed, false if the node doesn't exist
    /// or doesn't have the label.
    #[cfg(not(feature = "tiered-storage"))]
    pub fn remove_label(&self, node_id: NodeId, label: &str) -> bool {
        let epoch = self.current_epoch();

        // Check if node exists
        let nodes = self.nodes.read();
        if let Some(chain) = nodes.get(&node_id) {
            if chain.visible_at(epoch).map_or(true, |r| r.is_deleted()) {
                return false;
            }
        } else {
            return false;
        }
        drop(nodes);

        // Get label ID
        let label_id = {
            let label_ids = self.label_to_id.read();
            match label_ids.get(label) {
                Some(&id) => id,
                None => return false, // Label doesn't exist
            }
        };

        // Remove from node_labels map
        let mut node_labels = self.node_labels.write();
        if let Some(label_set) = node_labels.get_mut(&node_id) {
            if !label_set.remove(&label_id) {
                return false; // Node doesn't have this label
            }
        } else {
            return false;
        }
        drop(node_labels);

        // Remove from label_index
        let mut index = self.label_index.write();
        if (label_id as usize) < index.len() {
            index[label_id as usize].remove(&node_id);
        }

        // Update label count in node record
        if let Some(chain) = self.nodes.write().get_mut(&node_id) {
            if let Some(record) = chain.latest_mut() {
                let count = self.node_labels.read().get(&node_id).map_or(0, |s| s.len());
                record.set_label_count(count as u16);
            }
        }

        true
    }

    /// Removes a label from a node.
    /// (Tiered storage version)
    #[cfg(feature = "tiered-storage")]
    pub fn remove_label(&self, node_id: NodeId, label: &str) -> bool {
        let epoch = self.current_epoch();

        // Check if node exists
        let versions = self.node_versions.read();
        if let Some(index) = versions.get(&node_id) {
            if let Some(vref) = index.visible_at(epoch) {
                if let Some(record) = self.read_node_record(&vref) {
                    if record.is_deleted() {
                        return false;
                    }
                } else {
                    return false;
                }
            } else {
                return false;
            }
        } else {
            return false;
        }
        drop(versions);

        // Get label ID
        let label_id = {
            let label_ids = self.label_to_id.read();
            match label_ids.get(label) {
                Some(&id) => id,
                None => return false, // Label doesn't exist
            }
        };

        // Remove from node_labels map
        let mut node_labels = self.node_labels.write();
        if let Some(label_set) = node_labels.get_mut(&node_id) {
            if !label_set.remove(&label_id) {
                return false; // Node doesn't have this label
            }
        } else {
            return false;
        }
        drop(node_labels);

        // Remove from label_index
        let mut index = self.label_index.write();
        if (label_id as usize) < index.len() {
            index[label_id as usize].remove(&node_id);
        }

        // Note: label_count in record is not updated for tiered storage.

        true
    }

    /// Returns the number of nodes (non-deleted at current epoch).
    #[must_use]
    #[cfg(not(feature = "tiered-storage"))]
    pub fn node_count(&self) -> usize {
        let epoch = self.current_epoch();
        self.nodes
            .read()
            .values()
            .filter_map(|chain| chain.visible_at(epoch))
            .filter(|r| !r.is_deleted())
            .count()
    }

    /// Returns the number of nodes (non-deleted at current epoch).
    /// (Tiered storage version)
    #[must_use]
    #[cfg(feature = "tiered-storage")]
    pub fn node_count(&self) -> usize {
        let epoch = self.current_epoch();
        let versions = self.node_versions.read();
        versions
            .iter()
            .filter(|(_, index)| {
                index.visible_at(epoch).map_or(false, |vref| {
                    self.read_node_record(&vref)
                        .map_or(false, |r| !r.is_deleted())
                })
            })
            .count()
    }

    /// Returns all node IDs in the store.
    ///
    /// This returns a snapshot of current node IDs. The returned vector
    /// excludes deleted nodes. Results are sorted by NodeId for deterministic
    /// iteration order.
    #[must_use]
    #[cfg(not(feature = "tiered-storage"))]
    pub fn node_ids(&self) -> Vec<NodeId> {
        let epoch = self.current_epoch();
        let mut ids: Vec<NodeId> = self
            .nodes
            .read()
            .iter()
            .filter_map(|(id, chain)| {
                chain
                    .visible_at(epoch)
                    .and_then(|r| if !r.is_deleted() { Some(*id) } else { None })
            })
            .collect();
        ids.sort_unstable();
        ids
    }

    /// Returns all node IDs in the store.
    /// (Tiered storage version)
    #[must_use]
    #[cfg(feature = "tiered-storage")]
    pub fn node_ids(&self) -> Vec<NodeId> {
        let epoch = self.current_epoch();
        let versions = self.node_versions.read();
        let mut ids: Vec<NodeId> = versions
            .iter()
            .filter_map(|(id, index)| {
                index.visible_at(epoch).and_then(|vref| {
                    self.read_node_record(&vref)
                        .and_then(|r| if !r.is_deleted() { Some(*id) } else { None })
                })
            })
            .collect();
        ids.sort_unstable();
        ids
    }

    // === Edge Operations ===

    /// Creates a new edge.
    pub fn create_edge(&self, src: NodeId, dst: NodeId, edge_type: &str) -> EdgeId {
        self.needs_stats_recompute.store(true, Ordering::Relaxed);
        self.create_edge_versioned(src, dst, edge_type, self.current_epoch(), TxId::SYSTEM)
    }

    /// Creates a new edge within a transaction context.
    #[cfg(not(feature = "tiered-storage"))]
    pub fn create_edge_versioned(
        &self,
        src: NodeId,
        dst: NodeId,
        edge_type: &str,
        epoch: EpochId,
        tx_id: TxId,
    ) -> EdgeId {
        let id = EdgeId::new(self.next_edge_id.fetch_add(1, Ordering::Relaxed));
        let type_id = self.get_or_create_edge_type_id(edge_type);

        let record = EdgeRecord::new(id, src, dst, type_id, epoch);
        let chain = VersionChain::with_initial(record, epoch, tx_id);
        self.edges.write().insert(id, chain);

        // Update adjacency
        self.forward_adj.add_edge(src, dst, id);
        if let Some(ref backward) = self.backward_adj {
            backward.add_edge(dst, src, id);
        }

        id
    }

    /// Creates a new edge within a transaction context.
    /// (Tiered storage version)
    #[cfg(feature = "tiered-storage")]
    pub fn create_edge_versioned(
        &self,
        src: NodeId,
        dst: NodeId,
        edge_type: &str,
        epoch: EpochId,
        tx_id: TxId,
    ) -> EdgeId {
        let id = EdgeId::new(self.next_edge_id.fetch_add(1, Ordering::Relaxed));
        let type_id = self.get_or_create_edge_type_id(edge_type);

        let record = EdgeRecord::new(id, src, dst, type_id, epoch);

        // Allocate record in arena and get offset (create epoch if needed)
        let arena = self.arena_allocator.arena_or_create(epoch);
        let (offset, _stored) = arena.alloc_value_with_offset(record);

        // Create HotVersionRef pointing to arena data
        let hot_ref = HotVersionRef::new(epoch, offset, tx_id);

        // Create or update version index
        let mut versions = self.edge_versions.write();
        if let Some(index) = versions.get_mut(&id) {
            index.add_hot(hot_ref);
        } else {
            versions.insert(id, VersionIndex::with_initial(hot_ref));
        }

        // Update adjacency
        self.forward_adj.add_edge(src, dst, id);
        if let Some(ref backward) = self.backward_adj {
            backward.add_edge(dst, src, id);
        }

        id
    }

    /// Creates a new edge with properties.
    pub fn create_edge_with_props(
        &self,
        src: NodeId,
        dst: NodeId,
        edge_type: &str,
        properties: impl IntoIterator<Item = (impl Into<PropertyKey>, impl Into<Value>)>,
    ) -> EdgeId {
        let id = self.create_edge(src, dst, edge_type);

        for (key, value) in properties {
            self.edge_properties.set(id, key.into(), value.into());
        }

        id
    }

    /// Gets an edge by ID (latest visible version).
    #[must_use]
    pub fn get_edge(&self, id: EdgeId) -> Option<Edge> {
        self.get_edge_at_epoch(id, self.current_epoch())
    }

    /// Gets an edge by ID at a specific epoch.
    #[must_use]
    #[cfg(not(feature = "tiered-storage"))]
    pub fn get_edge_at_epoch(&self, id: EdgeId, epoch: EpochId) -> Option<Edge> {
        let edges = self.edges.read();
        let chain = edges.get(&id)?;
        let record = chain.visible_at(epoch)?;

        if record.is_deleted() {
            return None;
        }

        let edge_type = {
            let id_to_type = self.id_to_edge_type.read();
            id_to_type.get(record.type_id as usize)?.clone()
        };

        let mut edge = Edge::new(id, record.src, record.dst, edge_type);

        // Get properties
        edge.properties = self.edge_properties.get_all(id).into_iter().collect();

        Some(edge)
    }

    /// Gets an edge by ID at a specific epoch.
    /// (Tiered storage version)
    #[must_use]
    #[cfg(feature = "tiered-storage")]
    pub fn get_edge_at_epoch(&self, id: EdgeId, epoch: EpochId) -> Option<Edge> {
        let versions = self.edge_versions.read();
        let index = versions.get(&id)?;
        let version_ref = index.visible_at(epoch)?;

        let record = self.read_edge_record(&version_ref)?;

        if record.is_deleted() {
            return None;
        }

        let edge_type = {
            let id_to_type = self.id_to_edge_type.read();
            id_to_type.get(record.type_id as usize)?.clone()
        };

        let mut edge = Edge::new(id, record.src, record.dst, edge_type);

        // Get properties
        edge.properties = self.edge_properties.get_all(id).into_iter().collect();

        Some(edge)
    }

    /// Gets an edge visible to a specific transaction.
    #[must_use]
    #[cfg(not(feature = "tiered-storage"))]
    pub fn get_edge_versioned(&self, id: EdgeId, epoch: EpochId, tx_id: TxId) -> Option<Edge> {
        let edges = self.edges.read();
        let chain = edges.get(&id)?;
        let record = chain.visible_to(epoch, tx_id)?;

        if record.is_deleted() {
            return None;
        }

        let edge_type = {
            let id_to_type = self.id_to_edge_type.read();
            id_to_type.get(record.type_id as usize)?.clone()
        };

        let mut edge = Edge::new(id, record.src, record.dst, edge_type);

        // Get properties
        edge.properties = self.edge_properties.get_all(id).into_iter().collect();

        Some(edge)
    }

    /// Gets an edge visible to a specific transaction.
    /// (Tiered storage version)
    #[must_use]
    #[cfg(feature = "tiered-storage")]
    pub fn get_edge_versioned(&self, id: EdgeId, epoch: EpochId, tx_id: TxId) -> Option<Edge> {
        let versions = self.edge_versions.read();
        let index = versions.get(&id)?;
        let version_ref = index.visible_to(epoch, tx_id)?;

        let record = self.read_edge_record(&version_ref)?;

        if record.is_deleted() {
            return None;
        }

        let edge_type = {
            let id_to_type = self.id_to_edge_type.read();
            id_to_type.get(record.type_id as usize)?.clone()
        };

        let mut edge = Edge::new(id, record.src, record.dst, edge_type);

        // Get properties
        edge.properties = self.edge_properties.get_all(id).into_iter().collect();

        Some(edge)
    }

    /// Reads an EdgeRecord from arena using a VersionRef.
    #[cfg(feature = "tiered-storage")]
    #[allow(unsafe_code)]
    fn read_edge_record(&self, version_ref: &VersionRef) -> Option<EdgeRecord> {
        match version_ref {
            VersionRef::Hot(hot_ref) => {
                let arena = self.arena_allocator.arena(hot_ref.epoch);
                // SAFETY: The offset was returned by alloc_value_with_offset for an EdgeRecord
                let record: &EdgeRecord = unsafe { arena.read_at(hot_ref.arena_offset) };
                Some(*record)
            }
            VersionRef::Cold(cold_ref) => {
                // Read from compressed epoch store
                self.epoch_store
                    .get_edge(cold_ref.epoch, cold_ref.block_offset, cold_ref.length)
            }
        }
    }

    /// Deletes an edge (using latest epoch).
    pub fn delete_edge(&self, id: EdgeId) -> bool {
        self.needs_stats_recompute.store(true, Ordering::Relaxed);
        self.delete_edge_at_epoch(id, self.current_epoch())
    }

    /// Deletes an edge at a specific epoch.
    #[cfg(not(feature = "tiered-storage"))]
    pub fn delete_edge_at_epoch(&self, id: EdgeId, epoch: EpochId) -> bool {
        let mut edges = self.edges.write();
        if let Some(chain) = edges.get_mut(&id) {
            // Get the visible record to check if deleted and get src/dst
            let (src, dst) = {
                match chain.visible_at(epoch) {
                    Some(record) => {
                        if record.is_deleted() {
                            return false;
                        }
                        (record.src, record.dst)
                    }
                    None => return false, // Not visible at this epoch (already deleted)
                }
            };

            // Mark the version chain as deleted
            chain.mark_deleted(epoch);

            drop(edges); // Release lock

            // Mark as deleted in adjacency (soft delete)
            self.forward_adj.mark_deleted(src, id);
            if let Some(ref backward) = self.backward_adj {
                backward.mark_deleted(dst, id);
            }

            // Remove properties
            self.edge_properties.remove_all(id);

            true
        } else {
            false
        }
    }

    /// Deletes an edge at a specific epoch.
    /// (Tiered storage version)
    #[cfg(feature = "tiered-storage")]
    pub fn delete_edge_at_epoch(&self, id: EdgeId, epoch: EpochId) -> bool {
        let mut versions = self.edge_versions.write();
        if let Some(index) = versions.get_mut(&id) {
            // Get the visible record to check if deleted and get src/dst
            let (src, dst) = {
                match index.visible_at(epoch) {
                    Some(version_ref) => {
                        if let Some(record) = self.read_edge_record(&version_ref) {
                            if record.is_deleted() {
                                return false;
                            }
                            (record.src, record.dst)
                        } else {
                            return false;
                        }
                    }
                    None => return false,
                }
            };

            // Mark as deleted in version index
            index.mark_deleted(epoch);

            drop(versions); // Release lock

            // Mark as deleted in adjacency (soft delete)
            self.forward_adj.mark_deleted(src, id);
            if let Some(ref backward) = self.backward_adj {
                backward.mark_deleted(dst, id);
            }

            // Remove properties
            self.edge_properties.remove_all(id);

            true
        } else {
            false
        }
    }

    /// Returns the number of edges (non-deleted at current epoch).
    #[must_use]
    #[cfg(not(feature = "tiered-storage"))]
    pub fn edge_count(&self) -> usize {
        let epoch = self.current_epoch();
        self.edges
            .read()
            .values()
            .filter_map(|chain| chain.visible_at(epoch))
            .filter(|r| !r.is_deleted())
            .count()
    }

    /// Returns the number of edges (non-deleted at current epoch).
    /// (Tiered storage version)
    #[must_use]
    #[cfg(feature = "tiered-storage")]
    pub fn edge_count(&self) -> usize {
        let epoch = self.current_epoch();
        let versions = self.edge_versions.read();
        versions
            .iter()
            .filter(|(_, index)| {
                index.visible_at(epoch).map_or(false, |vref| {
                    self.read_edge_record(&vref)
                        .map_or(false, |r| !r.is_deleted())
                })
            })
            .count()
    }

    /// Discards all uncommitted versions created by a transaction.
    ///
    /// This is called during transaction rollback to clean up uncommitted changes.
    /// The method removes version chain entries created by the specified transaction.
    #[cfg(not(feature = "tiered-storage"))]
    pub fn discard_uncommitted_versions(&self, tx_id: TxId) {
        // Remove uncommitted node versions
        {
            let mut nodes = self.nodes.write();
            for chain in nodes.values_mut() {
                chain.remove_versions_by(tx_id);
            }
            // Remove completely empty chains (no versions left)
            nodes.retain(|_, chain| !chain.is_empty());
        }

        // Remove uncommitted edge versions
        {
            let mut edges = self.edges.write();
            for chain in edges.values_mut() {
                chain.remove_versions_by(tx_id);
            }
            // Remove completely empty chains (no versions left)
            edges.retain(|_, chain| !chain.is_empty());
        }
    }

    /// Discards all uncommitted versions created by a transaction.
    /// (Tiered storage version)
    #[cfg(feature = "tiered-storage")]
    pub fn discard_uncommitted_versions(&self, tx_id: TxId) {
        // Remove uncommitted node versions
        {
            let mut versions = self.node_versions.write();
            for index in versions.values_mut() {
                index.remove_versions_by(tx_id);
            }
            // Remove completely empty indexes (no versions left)
            versions.retain(|_, index| !index.is_empty());
        }

        // Remove uncommitted edge versions
        {
            let mut versions = self.edge_versions.write();
            for index in versions.values_mut() {
                index.remove_versions_by(tx_id);
            }
            // Remove completely empty indexes (no versions left)
            versions.retain(|_, index| !index.is_empty());
        }
    }

    /// Freezes an epoch from hot (arena) storage to cold (compressed) storage.
    ///
    /// This is called by the transaction manager when an epoch becomes eligible
    /// for freezing (no active transactions can see it). The freeze process:
    ///
    /// 1. Collects all hot version refs for the epoch
    /// 2. Reads the corresponding records from arena
    /// 3. Compresses them into a `CompressedEpochBlock`
    /// 4. Updates `VersionIndex` entries to point to cold storage
    /// 5. The arena can be deallocated after all epochs in it are frozen
    ///
    /// # Arguments
    ///
    /// * `epoch` - The epoch to freeze
    ///
    /// # Returns
    ///
    /// The number of records frozen (nodes + edges).
    #[cfg(feature = "tiered-storage")]
    #[allow(unsafe_code)]
    pub fn freeze_epoch(&self, epoch: EpochId) -> usize {
        // Collect node records to freeze
        let mut node_records: Vec<(u64, NodeRecord)> = Vec::new();
        let mut node_hot_refs: Vec<(NodeId, HotVersionRef)> = Vec::new();

        {
            let versions = self.node_versions.read();
            for (node_id, index) in versions.iter() {
                for hot_ref in index.hot_refs_for_epoch(epoch) {
                    let arena = self.arena_allocator.arena(hot_ref.epoch);
                    // SAFETY: The offset was returned by alloc_value_with_offset for a NodeRecord
                    let record: &NodeRecord = unsafe { arena.read_at(hot_ref.arena_offset) };
                    node_records.push((node_id.as_u64(), *record));
                    node_hot_refs.push((*node_id, *hot_ref));
                }
            }
        }

        // Collect edge records to freeze
        let mut edge_records: Vec<(u64, EdgeRecord)> = Vec::new();
        let mut edge_hot_refs: Vec<(EdgeId, HotVersionRef)> = Vec::new();

        {
            let versions = self.edge_versions.read();
            for (edge_id, index) in versions.iter() {
                for hot_ref in index.hot_refs_for_epoch(epoch) {
                    let arena = self.arena_allocator.arena(hot_ref.epoch);
                    // SAFETY: The offset was returned by alloc_value_with_offset for an EdgeRecord
                    let record: &EdgeRecord = unsafe { arena.read_at(hot_ref.arena_offset) };
                    edge_records.push((edge_id.as_u64(), *record));
                    edge_hot_refs.push((*edge_id, *hot_ref));
                }
            }
        }

        let total_frozen = node_records.len() + edge_records.len();

        if total_frozen == 0 {
            return 0;
        }

        // Freeze to compressed storage
        let (node_entries, edge_entries) =
            self.epoch_store
                .freeze_epoch(epoch, node_records, edge_records);

        // Build lookup maps for index entries
        let node_entry_map: FxHashMap<u64, _> = node_entries
            .iter()
            .map(|e| (e.entity_id, (e.offset, e.length)))
            .collect();
        let edge_entry_map: FxHashMap<u64, _> = edge_entries
            .iter()
            .map(|e| (e.entity_id, (e.offset, e.length)))
            .collect();

        // Update version indexes to use cold refs
        {
            let mut versions = self.node_versions.write();
            for (node_id, hot_ref) in &node_hot_refs {
                if let Some(index) = versions.get_mut(node_id)
                    && let Some(&(offset, length)) = node_entry_map.get(&node_id.as_u64())
                {
                    let cold_ref = ColdVersionRef {
                        epoch,
                        block_offset: offset,
                        length,
                        created_by: hot_ref.created_by,
                        deleted_epoch: hot_ref.deleted_epoch,
                    };
                    index.freeze_epoch(epoch, std::iter::once(cold_ref));
                }
            }
        }

        {
            let mut versions = self.edge_versions.write();
            for (edge_id, hot_ref) in &edge_hot_refs {
                if let Some(index) = versions.get_mut(edge_id)
                    && let Some(&(offset, length)) = edge_entry_map.get(&edge_id.as_u64())
                {
                    let cold_ref = ColdVersionRef {
                        epoch,
                        block_offset: offset,
                        length,
                        created_by: hot_ref.created_by,
                        deleted_epoch: hot_ref.deleted_epoch,
                    };
                    index.freeze_epoch(epoch, std::iter::once(cold_ref));
                }
            }
        }

        total_frozen
    }

    /// Returns the epoch store for cold storage statistics.
    #[cfg(feature = "tiered-storage")]
    #[must_use]
    pub fn epoch_store(&self) -> &EpochStore {
        &self.epoch_store
    }

    /// Returns the number of distinct labels in the store.
    #[must_use]
    pub fn label_count(&self) -> usize {
        self.id_to_label.read().len()
    }

    /// Returns the number of distinct property keys in the store.
    ///
    /// This counts unique property keys across both nodes and edges.
    #[must_use]
    pub fn property_key_count(&self) -> usize {
        let node_keys = self.node_properties.column_count();
        let edge_keys = self.edge_properties.column_count();
        // Note: This may count some keys twice if the same key is used
        // for both nodes and edges. A more precise count would require
        // tracking unique keys across both storages.
        node_keys + edge_keys
    }

    /// Returns the number of distinct edge types in the store.
    #[must_use]
    pub fn edge_type_count(&self) -> usize {
        self.id_to_edge_type.read().len()
    }

    // === Traversal ===

    /// Iterates over neighbors of a node in the specified direction.
    ///
    /// This is the fast path for graph traversal - goes straight to the
    /// adjacency index without loading full node data.
    pub fn neighbors(
        &self,
        node: NodeId,
        direction: Direction,
    ) -> impl Iterator<Item = NodeId> + '_ {
        let forward: Box<dyn Iterator<Item = NodeId>> = match direction {
            Direction::Outgoing | Direction::Both => {
                Box::new(self.forward_adj.neighbors(node).into_iter())
            }
            Direction::Incoming => Box::new(std::iter::empty()),
        };

        let backward: Box<dyn Iterator<Item = NodeId>> = match direction {
            Direction::Incoming | Direction::Both => {
                if let Some(ref adj) = self.backward_adj {
                    Box::new(adj.neighbors(node).into_iter())
                } else {
                    Box::new(std::iter::empty())
                }
            }
            Direction::Outgoing => Box::new(std::iter::empty()),
        };

        forward.chain(backward)
    }

    /// Returns edges from a node with their targets.
    ///
    /// Returns an iterator of (target_node, edge_id) pairs.
    pub fn edges_from(
        &self,
        node: NodeId,
        direction: Direction,
    ) -> impl Iterator<Item = (NodeId, EdgeId)> + '_ {
        let forward: Box<dyn Iterator<Item = (NodeId, EdgeId)>> = match direction {
            Direction::Outgoing | Direction::Both => {
                Box::new(self.forward_adj.edges_from(node).into_iter())
            }
            Direction::Incoming => Box::new(std::iter::empty()),
        };

        let backward: Box<dyn Iterator<Item = (NodeId, EdgeId)>> = match direction {
            Direction::Incoming | Direction::Both => {
                if let Some(ref adj) = self.backward_adj {
                    Box::new(adj.edges_from(node).into_iter())
                } else {
                    Box::new(std::iter::empty())
                }
            }
            Direction::Outgoing => Box::new(std::iter::empty()),
        };

        forward.chain(backward)
    }

    /// Returns edges to a node (where the node is the destination).
    ///
    /// Returns (source_node, edge_id) pairs for all edges pointing TO this node.
    /// Uses the backward adjacency index for O(degree) lookup.
    ///
    /// # Example
    ///
    /// ```ignore
    /// // For edges: A->B, C->B
    /// let incoming = store.edges_to(B);
    /// // Returns: [(A, edge1), (C, edge2)]
    /// ```
    pub fn edges_to(&self, node: NodeId) -> Vec<(NodeId, EdgeId)> {
        if let Some(ref backward) = self.backward_adj {
            backward.edges_from(node)
        } else {
            // Fallback: scan all edges (slow but correct)
            self.all_edges()
                .filter_map(|edge| {
                    if edge.dst == node {
                        Some((edge.src, edge.id))
                    } else {
                        None
                    }
                })
                .collect()
        }
    }

    /// Returns the out-degree of a node (number of outgoing edges).
    ///
    /// Uses the forward adjacency index for O(1) lookup.
    #[must_use]
    pub fn out_degree(&self, node: NodeId) -> usize {
        self.forward_adj.out_degree(node)
    }

    /// Returns the in-degree of a node (number of incoming edges).
    ///
    /// Uses the backward adjacency index for O(1) lookup if available,
    /// otherwise falls back to scanning edges.
    #[must_use]
    pub fn in_degree(&self, node: NodeId) -> usize {
        if let Some(ref backward) = self.backward_adj {
            backward.in_degree(node)
        } else {
            // Fallback: count edges (slow)
            self.all_edges().filter(|edge| edge.dst == node).count()
        }
    }

    /// Gets the type of an edge by ID.
    #[must_use]
    #[cfg(not(feature = "tiered-storage"))]
    pub fn edge_type(&self, id: EdgeId) -> Option<ArcStr> {
        let edges = self.edges.read();
        let chain = edges.get(&id)?;
        let epoch = self.current_epoch();
        let record = chain.visible_at(epoch)?;
        let id_to_type = self.id_to_edge_type.read();
        id_to_type.get(record.type_id as usize).cloned()
    }

    /// Gets the type of an edge by ID.
    /// (Tiered storage version)
    #[must_use]
    #[cfg(feature = "tiered-storage")]
    pub fn edge_type(&self, id: EdgeId) -> Option<ArcStr> {
        let versions = self.edge_versions.read();
        let index = versions.get(&id)?;
        let epoch = self.current_epoch();
        let vref = index.visible_at(epoch)?;
        let record = self.read_edge_record(&vref)?;
        let id_to_type = self.id_to_edge_type.read();
        id_to_type.get(record.type_id as usize).cloned()
    }

    /// Returns all nodes with a specific label.
    ///
    /// Uses the label index for O(1) lookup per label. Returns a snapshot -
    /// concurrent modifications won't affect the returned vector. Results are
    /// sorted by NodeId for deterministic iteration order.
    pub fn nodes_by_label(&self, label: &str) -> Vec<NodeId> {
        let label_to_id = self.label_to_id.read();
        if let Some(&label_id) = label_to_id.get(label) {
            let index = self.label_index.read();
            if let Some(set) = index.get(label_id as usize) {
                let mut ids: Vec<NodeId> = set.keys().copied().collect();
                ids.sort_unstable();
                return ids;
            }
        }
        Vec::new()
    }

    // === Admin API: Iteration ===

    /// Returns an iterator over all nodes in the database.
    ///
    /// This creates a snapshot of all visible nodes at the current epoch.
    /// Useful for dump/export operations.
    #[cfg(not(feature = "tiered-storage"))]
    pub fn all_nodes(&self) -> impl Iterator<Item = Node> + '_ {
        let epoch = self.current_epoch();
        let node_ids: Vec<NodeId> = self
            .nodes
            .read()
            .iter()
            .filter_map(|(id, chain)| {
                chain
                    .visible_at(epoch)
                    .and_then(|r| if !r.is_deleted() { Some(*id) } else { None })
            })
            .collect();

        node_ids.into_iter().filter_map(move |id| self.get_node(id))
    }

    /// Returns an iterator over all nodes in the database.
    /// (Tiered storage version)
    #[cfg(feature = "tiered-storage")]
    pub fn all_nodes(&self) -> impl Iterator<Item = Node> + '_ {
        let node_ids = self.node_ids();
        node_ids.into_iter().filter_map(move |id| self.get_node(id))
    }

    /// Returns an iterator over all edges in the database.
    ///
    /// This creates a snapshot of all visible edges at the current epoch.
    /// Useful for dump/export operations.
    #[cfg(not(feature = "tiered-storage"))]
    pub fn all_edges(&self) -> impl Iterator<Item = Edge> + '_ {
        let epoch = self.current_epoch();
        let edge_ids: Vec<EdgeId> = self
            .edges
            .read()
            .iter()
            .filter_map(|(id, chain)| {
                chain
                    .visible_at(epoch)
                    .and_then(|r| if !r.is_deleted() { Some(*id) } else { None })
            })
            .collect();

        edge_ids.into_iter().filter_map(move |id| self.get_edge(id))
    }

    /// Returns an iterator over all edges in the database.
    /// (Tiered storage version)
    #[cfg(feature = "tiered-storage")]
    pub fn all_edges(&self) -> impl Iterator<Item = Edge> + '_ {
        let epoch = self.current_epoch();
        let versions = self.edge_versions.read();
        let edge_ids: Vec<EdgeId> = versions
            .iter()
            .filter_map(|(id, index)| {
                index.visible_at(epoch).and_then(|vref| {
                    self.read_edge_record(&vref)
                        .and_then(|r| if !r.is_deleted() { Some(*id) } else { None })
                })
            })
            .collect();

        edge_ids.into_iter().filter_map(move |id| self.get_edge(id))
    }

    /// Returns all label names in the database.
    pub fn all_labels(&self) -> Vec<String> {
        self.id_to_label
            .read()
            .iter()
            .map(|s| s.to_string())
            .collect()
    }

    /// Returns all edge type names in the database.
    pub fn all_edge_types(&self) -> Vec<String> {
        self.id_to_edge_type
            .read()
            .iter()
            .map(|s| s.to_string())
            .collect()
    }

    /// Returns all property keys used in the database.
    pub fn all_property_keys(&self) -> Vec<String> {
        let mut keys = std::collections::HashSet::new();
        for key in self.node_properties.keys() {
            keys.insert(key.to_string());
        }
        for key in self.edge_properties.keys() {
            keys.insert(key.to_string());
        }
        keys.into_iter().collect()
    }

    /// Returns an iterator over nodes with a specific label.
    pub fn nodes_with_label<'a>(&'a self, label: &str) -> impl Iterator<Item = Node> + 'a {
        let node_ids = self.nodes_by_label(label);
        node_ids.into_iter().filter_map(move |id| self.get_node(id))
    }

    /// Returns an iterator over edges with a specific type.
    #[cfg(not(feature = "tiered-storage"))]
    pub fn edges_with_type<'a>(&'a self, edge_type: &str) -> impl Iterator<Item = Edge> + 'a {
        let epoch = self.current_epoch();
        let type_to_id = self.edge_type_to_id.read();

        if let Some(&type_id) = type_to_id.get(edge_type) {
            let edge_ids: Vec<EdgeId> = self
                .edges
                .read()
                .iter()
                .filter_map(|(id, chain)| {
                    chain.visible_at(epoch).and_then(|r| {
                        if !r.is_deleted() && r.type_id == type_id {
                            Some(*id)
                        } else {
                            None
                        }
                    })
                })
                .collect();

            // Return a boxed iterator for the found edges
            Box::new(edge_ids.into_iter().filter_map(move |id| self.get_edge(id)))
                as Box<dyn Iterator<Item = Edge> + 'a>
        } else {
            // Return empty iterator
            Box::new(std::iter::empty()) as Box<dyn Iterator<Item = Edge> + 'a>
        }
    }

    /// Returns an iterator over edges with a specific type.
    /// (Tiered storage version)
    #[cfg(feature = "tiered-storage")]
    pub fn edges_with_type<'a>(&'a self, edge_type: &str) -> impl Iterator<Item = Edge> + 'a {
        let epoch = self.current_epoch();
        let type_to_id = self.edge_type_to_id.read();

        if let Some(&type_id) = type_to_id.get(edge_type) {
            let versions = self.edge_versions.read();
            let edge_ids: Vec<EdgeId> = versions
                .iter()
                .filter_map(|(id, index)| {
                    index.visible_at(epoch).and_then(|vref| {
                        self.read_edge_record(&vref).and_then(|r| {
                            if !r.is_deleted() && r.type_id == type_id {
                                Some(*id)
                            } else {
                                None
                            }
                        })
                    })
                })
                .collect();

            Box::new(edge_ids.into_iter().filter_map(move |id| self.get_edge(id)))
                as Box<dyn Iterator<Item = Edge> + 'a>
        } else {
            Box::new(std::iter::empty()) as Box<dyn Iterator<Item = Edge> + 'a>
        }
    }

    // === Zone Map Support ===

    /// Checks if a node property predicate might match any nodes.
    ///
    /// Uses zone maps for early filtering. Returns `true` if there might be
    /// matching nodes, `false` if there definitely aren't.
    #[must_use]
    pub fn node_property_might_match(
        &self,
        property: &PropertyKey,
        op: CompareOp,
        value: &Value,
    ) -> bool {
        self.node_properties.might_match(property, op, value)
    }

    /// Checks if an edge property predicate might match any edges.
    #[must_use]
    pub fn edge_property_might_match(
        &self,
        property: &PropertyKey,
        op: CompareOp,
        value: &Value,
    ) -> bool {
        self.edge_properties.might_match(property, op, value)
    }

    /// Gets the zone map for a node property.
    #[must_use]
    pub fn node_property_zone_map(&self, property: &PropertyKey) -> Option<ZoneMapEntry> {
        self.node_properties.zone_map(property)
    }

    /// Gets the zone map for an edge property.
    #[must_use]
    pub fn edge_property_zone_map(&self, property: &PropertyKey) -> Option<ZoneMapEntry> {
        self.edge_properties.zone_map(property)
    }

    /// Rebuilds zone maps for all properties.
    pub fn rebuild_zone_maps(&self) {
        self.node_properties.rebuild_zone_maps();
        self.edge_properties.rebuild_zone_maps();
    }

    // === Statistics ===

    /// Returns the current statistics.
    #[must_use]
    pub fn statistics(&self) -> Statistics {
        self.statistics.read().clone()
    }

    /// Recomputes statistics if they are stale (i.e., after mutations).
    ///
    /// Call this before reading statistics for query optimization.
    /// Avoids redundant recomputation if no mutations occurred.
    pub fn ensure_statistics_fresh(&self) {
        if self.needs_stats_recompute.swap(false, Ordering::Relaxed) {
            self.compute_statistics();
        }
    }

    /// Recomputes statistics from current data.
    ///
    /// Scans all labels and edge types to build cardinality estimates for the
    /// query optimizer. Call this periodically or after bulk data loads.
    #[cfg(not(feature = "tiered-storage"))]
    pub fn compute_statistics(&self) {
        let mut stats = Statistics::new();

        // Compute total counts
        stats.total_nodes = self.node_count() as u64;
        stats.total_edges = self.edge_count() as u64;

        // Compute per-label statistics
        let id_to_label = self.id_to_label.read();
        let label_index = self.label_index.read();

        for (label_id, label_name) in id_to_label.iter().enumerate() {
            let node_count = label_index
                .get(label_id)
                .map(|set| set.len() as u64)
                .unwrap_or(0);

            if node_count > 0 {
                // Estimate average degree
                let avg_out_degree = if stats.total_nodes > 0 {
                    stats.total_edges as f64 / stats.total_nodes as f64
                } else {
                    0.0
                };

                let label_stats =
                    LabelStatistics::new(node_count).with_degrees(avg_out_degree, avg_out_degree);

                stats.update_label(label_name.as_ref(), label_stats);
            }
        }

        // Compute per-edge-type statistics
        let id_to_edge_type = self.id_to_edge_type.read();
        let edges = self.edges.read();
        let epoch = self.current_epoch();

        let mut edge_type_counts: FxHashMap<u32, u64> = FxHashMap::default();
        for chain in edges.values() {
            if let Some(record) = chain.visible_at(epoch) {
                if !record.is_deleted() {
                    *edge_type_counts.entry(record.type_id).or_default() += 1;
                }
            }
        }

        for (type_id, count) in edge_type_counts {
            if let Some(type_name) = id_to_edge_type.get(type_id as usize) {
                let avg_degree = if stats.total_nodes > 0 {
                    count as f64 / stats.total_nodes as f64
                } else {
                    0.0
                };

                let edge_stats = EdgeTypeStatistics::new(count, avg_degree, avg_degree);
                stats.update_edge_type(type_name.as_ref(), edge_stats);
            }
        }

        *self.statistics.write() = stats;
    }

    /// Recomputes statistics from current data.
    /// (Tiered storage version)
    #[cfg(feature = "tiered-storage")]
    pub fn compute_statistics(&self) {
        let mut stats = Statistics::new();

        // Compute total counts
        stats.total_nodes = self.node_count() as u64;
        stats.total_edges = self.edge_count() as u64;

        // Compute per-label statistics
        let id_to_label = self.id_to_label.read();
        let label_index = self.label_index.read();

        for (label_id, label_name) in id_to_label.iter().enumerate() {
            let node_count = label_index.get(label_id).map_or(0, |set| set.len() as u64);

            if node_count > 0 {
                let avg_out_degree = if stats.total_nodes > 0 {
                    stats.total_edges as f64 / stats.total_nodes as f64
                } else {
                    0.0
                };

                let label_stats =
                    LabelStatistics::new(node_count).with_degrees(avg_out_degree, avg_out_degree);

                stats.update_label(label_name.as_ref(), label_stats);
            }
        }

        // Compute per-edge-type statistics
        let id_to_edge_type = self.id_to_edge_type.read();
        let versions = self.edge_versions.read();
        let epoch = self.current_epoch();

        let mut edge_type_counts: FxHashMap<u32, u64> = FxHashMap::default();
        for index in versions.values() {
            if let Some(vref) = index.visible_at(epoch)
                && let Some(record) = self.read_edge_record(&vref)
                && !record.is_deleted()
            {
                *edge_type_counts.entry(record.type_id).or_default() += 1;
            }
        }

        for (type_id, count) in edge_type_counts {
            if let Some(type_name) = id_to_edge_type.get(type_id as usize) {
                let avg_degree = if stats.total_nodes > 0 {
                    count as f64 / stats.total_nodes as f64
                } else {
                    0.0
                };

                let edge_stats = EdgeTypeStatistics::new(count, avg_degree, avg_degree);
                stats.update_edge_type(type_name.as_ref(), edge_stats);
            }
        }

        *self.statistics.write() = stats;
    }

    /// Estimates cardinality for a label scan.
    #[must_use]
    pub fn estimate_label_cardinality(&self, label: &str) -> f64 {
        self.statistics.read().estimate_label_cardinality(label)
    }

    /// Estimates average degree for an edge type.
    #[must_use]
    pub fn estimate_avg_degree(&self, edge_type: &str, outgoing: bool) -> f64 {
        self.statistics
            .read()
            .estimate_avg_degree(edge_type, outgoing)
    }

    // === Internal Helpers ===

    fn get_or_create_label_id(&self, label: &str) -> u32 {
        {
            let label_to_id = self.label_to_id.read();
            if let Some(&id) = label_to_id.get(label) {
                return id;
            }
        }

        let mut label_to_id = self.label_to_id.write();
        let mut id_to_label = self.id_to_label.write();

        // Double-check after acquiring write lock
        if let Some(&id) = label_to_id.get(label) {
            return id;
        }

        let id = id_to_label.len() as u32;

        let label: ArcStr = label.into();
        label_to_id.insert(label.clone(), id);
        id_to_label.push(label);

        id
    }

    fn get_or_create_edge_type_id(&self, edge_type: &str) -> u32 {
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

        id
    }

    // === Recovery Support ===

    /// Creates a node with a specific ID during recovery.
    ///
    /// This is used for WAL recovery to restore nodes with their original IDs.
    /// The caller must ensure IDs don't conflict with existing nodes.
    #[cfg(not(feature = "tiered-storage"))]
    pub fn create_node_with_id(&self, id: NodeId, labels: &[&str]) {
        let epoch = self.current_epoch();
        let mut record = NodeRecord::new(id, epoch);
        record.set_label_count(labels.len() as u16);

        // Store labels in node_labels map and label_index
        let mut node_label_set = FxHashSet::default();
        for label in labels {
            let label_id = self.get_or_create_label_id(*label);
            node_label_set.insert(label_id);

            // Update label index
            let mut index = self.label_index.write();
            while index.len() <= label_id as usize {
                index.push(FxHashMap::default());
            }
            index[label_id as usize].insert(id, ());
        }

        // Store node's labels
        self.node_labels.write().insert(id, node_label_set);

        // Create version chain with initial version (using SYSTEM tx for recovery)
        let chain = VersionChain::with_initial(record, epoch, TxId::SYSTEM);
        self.nodes.write().insert(id, chain);

        // Update next_node_id if necessary to avoid future collisions
        let id_val = id.as_u64();
        let _ = self
            .next_node_id
            .fetch_update(Ordering::SeqCst, Ordering::SeqCst, |current| {
                if id_val >= current {
                    Some(id_val + 1)
                } else {
                    None
                }
            });
    }

    /// Creates a node with a specific ID during recovery.
    /// (Tiered storage version)
    #[cfg(feature = "tiered-storage")]
    pub fn create_node_with_id(&self, id: NodeId, labels: &[&str]) {
        let epoch = self.current_epoch();
        let mut record = NodeRecord::new(id, epoch);
        record.set_label_count(labels.len() as u16);

        // Store labels in node_labels map and label_index
        let mut node_label_set = FxHashSet::default();
        for label in labels {
            let label_id = self.get_or_create_label_id(*label);
            node_label_set.insert(label_id);

            // Update label index
            let mut index = self.label_index.write();
            while index.len() <= label_id as usize {
                index.push(FxHashMap::default());
            }
            index[label_id as usize].insert(id, ());
        }

        // Store node's labels
        self.node_labels.write().insert(id, node_label_set);

        // Allocate record in arena and get offset (create epoch if needed)
        let arena = self.arena_allocator.arena_or_create(epoch);
        let (offset, _stored) = arena.alloc_value_with_offset(record);

        // Create HotVersionRef (using SYSTEM tx for recovery)
        let hot_ref = HotVersionRef::new(epoch, offset, TxId::SYSTEM);
        let mut versions = self.node_versions.write();
        versions.insert(id, VersionIndex::with_initial(hot_ref));

        // Update next_node_id if necessary to avoid future collisions
        let id_val = id.as_u64();
        let _ = self
            .next_node_id
            .fetch_update(Ordering::SeqCst, Ordering::SeqCst, |current| {
                if id_val >= current {
                    Some(id_val + 1)
                } else {
                    None
                }
            });
    }

    /// Creates an edge with a specific ID during recovery.
    ///
    /// This is used for WAL recovery to restore edges with their original IDs.
    #[cfg(not(feature = "tiered-storage"))]
    pub fn create_edge_with_id(&self, id: EdgeId, src: NodeId, dst: NodeId, edge_type: &str) {
        let epoch = self.current_epoch();
        let type_id = self.get_or_create_edge_type_id(edge_type);

        let record = EdgeRecord::new(id, src, dst, type_id, epoch);
        let chain = VersionChain::with_initial(record, epoch, TxId::SYSTEM);
        self.edges.write().insert(id, chain);

        // Update adjacency
        self.forward_adj.add_edge(src, dst, id);
        if let Some(ref backward) = self.backward_adj {
            backward.add_edge(dst, src, id);
        }

        // Update next_edge_id if necessary
        let id_val = id.as_u64();
        let _ = self
            .next_edge_id
            .fetch_update(Ordering::SeqCst, Ordering::SeqCst, |current| {
                if id_val >= current {
                    Some(id_val + 1)
                } else {
                    None
                }
            });
    }

    /// Creates an edge with a specific ID during recovery.
    /// (Tiered storage version)
    #[cfg(feature = "tiered-storage")]
    pub fn create_edge_with_id(&self, id: EdgeId, src: NodeId, dst: NodeId, edge_type: &str) {
        let epoch = self.current_epoch();
        let type_id = self.get_or_create_edge_type_id(edge_type);

        let record = EdgeRecord::new(id, src, dst, type_id, epoch);

        // Allocate record in arena and get offset (create epoch if needed)
        let arena = self.arena_allocator.arena_or_create(epoch);
        let (offset, _stored) = arena.alloc_value_with_offset(record);

        // Create HotVersionRef (using SYSTEM tx for recovery)
        let hot_ref = HotVersionRef::new(epoch, offset, TxId::SYSTEM);
        let mut versions = self.edge_versions.write();
        versions.insert(id, VersionIndex::with_initial(hot_ref));

        // Update adjacency
        self.forward_adj.add_edge(src, dst, id);
        if let Some(ref backward) = self.backward_adj {
            backward.add_edge(dst, src, id);
        }

        // Update next_edge_id if necessary
        let id_val = id.as_u64();
        let _ = self
            .next_edge_id
            .fetch_update(Ordering::SeqCst, Ordering::SeqCst, |current| {
                if id_val >= current {
                    Some(id_val + 1)
                } else {
                    None
                }
            });
    }

    /// Sets the current epoch during recovery.
    pub fn set_epoch(&self, epoch: EpochId) {
        self.current_epoch.store(epoch.as_u64(), Ordering::SeqCst);
    }
}

impl Default for LpgStore {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_node() {
        let store = LpgStore::new();

        let id = store.create_node(&["Person"]);
        assert!(id.is_valid());

        let node = store.get_node(id).unwrap();
        assert!(node.has_label("Person"));
        assert!(!node.has_label("Animal"));
    }

    #[test]
    fn test_create_node_with_props() {
        let store = LpgStore::new();

        let id = store.create_node_with_props(
            &["Person"],
            [("name", Value::from("Alice")), ("age", Value::from(30i64))],
        );

        let node = store.get_node(id).unwrap();
        assert_eq!(
            node.get_property("name").and_then(|v| v.as_str()),
            Some("Alice")
        );
        assert_eq!(
            node.get_property("age").and_then(|v| v.as_int64()),
            Some(30)
        );
    }

    #[test]
    fn test_delete_node() {
        let store = LpgStore::new();

        let id = store.create_node(&["Person"]);
        assert_eq!(store.node_count(), 1);

        assert!(store.delete_node(id));
        assert_eq!(store.node_count(), 0);
        assert!(store.get_node(id).is_none());

        // Double delete should return false
        assert!(!store.delete_node(id));
    }

    #[test]
    fn test_create_edge() {
        let store = LpgStore::new();

        let alice = store.create_node(&["Person"]);
        let bob = store.create_node(&["Person"]);

        let edge_id = store.create_edge(alice, bob, "KNOWS");
        assert!(edge_id.is_valid());

        let edge = store.get_edge(edge_id).unwrap();
        assert_eq!(edge.src, alice);
        assert_eq!(edge.dst, bob);
        assert_eq!(edge.edge_type.as_str(), "KNOWS");
    }

    #[test]
    fn test_neighbors() {
        let store = LpgStore::new();

        let a = store.create_node(&["Person"]);
        let b = store.create_node(&["Person"]);
        let c = store.create_node(&["Person"]);

        store.create_edge(a, b, "KNOWS");
        store.create_edge(a, c, "KNOWS");

        let outgoing: Vec<_> = store.neighbors(a, Direction::Outgoing).collect();
        assert_eq!(outgoing.len(), 2);
        assert!(outgoing.contains(&b));
        assert!(outgoing.contains(&c));

        let incoming: Vec<_> = store.neighbors(b, Direction::Incoming).collect();
        assert_eq!(incoming.len(), 1);
        assert!(incoming.contains(&a));
    }

    #[test]
    fn test_nodes_by_label() {
        let store = LpgStore::new();

        let p1 = store.create_node(&["Person"]);
        let p2 = store.create_node(&["Person"]);
        let _a = store.create_node(&["Animal"]);

        let persons = store.nodes_by_label("Person");
        assert_eq!(persons.len(), 2);
        assert!(persons.contains(&p1));
        assert!(persons.contains(&p2));

        let animals = store.nodes_by_label("Animal");
        assert_eq!(animals.len(), 1);
    }

    #[test]
    fn test_delete_edge() {
        let store = LpgStore::new();

        let a = store.create_node(&["Person"]);
        let b = store.create_node(&["Person"]);
        let edge_id = store.create_edge(a, b, "KNOWS");

        assert_eq!(store.edge_count(), 1);

        assert!(store.delete_edge(edge_id));
        assert_eq!(store.edge_count(), 0);
        assert!(store.get_edge(edge_id).is_none());
    }

    // === New tests for improved coverage ===

    #[test]
    fn test_lpg_store_config() {
        // Test with_config
        let config = LpgStoreConfig {
            backward_edges: false,
            initial_node_capacity: 100,
            initial_edge_capacity: 200,
        };
        let store = LpgStore::with_config(config);

        // Store should work but without backward adjacency
        let a = store.create_node(&["Person"]);
        let b = store.create_node(&["Person"]);
        store.create_edge(a, b, "KNOWS");

        // Outgoing should work
        let outgoing: Vec<_> = store.neighbors(a, Direction::Outgoing).collect();
        assert_eq!(outgoing.len(), 1);

        // Incoming should be empty (no backward adjacency)
        let incoming: Vec<_> = store.neighbors(b, Direction::Incoming).collect();
        assert_eq!(incoming.len(), 0);
    }

    #[test]
    fn test_epoch_management() {
        let store = LpgStore::new();

        let epoch0 = store.current_epoch();
        assert_eq!(epoch0.as_u64(), 0);

        let epoch1 = store.new_epoch();
        assert_eq!(epoch1.as_u64(), 1);

        let current = store.current_epoch();
        assert_eq!(current.as_u64(), 1);
    }

    #[test]
    fn test_node_properties() {
        let store = LpgStore::new();
        let id = store.create_node(&["Person"]);

        // Set and get property
        store.set_node_property(id, "name", Value::from("Alice"));
        let name = store.get_node_property(id, &"name".into());
        assert!(matches!(name, Some(Value::String(s)) if s.as_str() == "Alice"));

        // Update property
        store.set_node_property(id, "name", Value::from("Bob"));
        let name = store.get_node_property(id, &"name".into());
        assert!(matches!(name, Some(Value::String(s)) if s.as_str() == "Bob"));

        // Remove property
        let old = store.remove_node_property(id, "name");
        assert!(matches!(old, Some(Value::String(s)) if s.as_str() == "Bob"));

        // Property should be gone
        let name = store.get_node_property(id, &"name".into());
        assert!(name.is_none());

        // Remove non-existent property
        let none = store.remove_node_property(id, "nonexistent");
        assert!(none.is_none());
    }

    #[test]
    fn test_edge_properties() {
        let store = LpgStore::new();
        let a = store.create_node(&["Person"]);
        let b = store.create_node(&["Person"]);
        let edge_id = store.create_edge(a, b, "KNOWS");

        // Set and get property
        store.set_edge_property(edge_id, "since", Value::from(2020i64));
        let since = store.get_edge_property(edge_id, &"since".into());
        assert_eq!(since.and_then(|v| v.as_int64()), Some(2020));

        // Remove property
        let old = store.remove_edge_property(edge_id, "since");
        assert_eq!(old.and_then(|v| v.as_int64()), Some(2020));

        let since = store.get_edge_property(edge_id, &"since".into());
        assert!(since.is_none());
    }

    #[test]
    fn test_add_remove_label() {
        let store = LpgStore::new();
        let id = store.create_node(&["Person"]);

        // Add new label
        assert!(store.add_label(id, "Employee"));

        let node = store.get_node(id).unwrap();
        assert!(node.has_label("Person"));
        assert!(node.has_label("Employee"));

        // Adding same label again should fail
        assert!(!store.add_label(id, "Employee"));

        // Remove label
        assert!(store.remove_label(id, "Employee"));

        let node = store.get_node(id).unwrap();
        assert!(node.has_label("Person"));
        assert!(!node.has_label("Employee"));

        // Removing non-existent label should fail
        assert!(!store.remove_label(id, "Employee"));
        assert!(!store.remove_label(id, "NonExistent"));
    }

    #[test]
    fn test_add_label_to_nonexistent_node() {
        let store = LpgStore::new();
        let fake_id = NodeId::new(999);
        assert!(!store.add_label(fake_id, "Label"));
    }

    #[test]
    fn test_remove_label_from_nonexistent_node() {
        let store = LpgStore::new();
        let fake_id = NodeId::new(999);
        assert!(!store.remove_label(fake_id, "Label"));
    }

    #[test]
    fn test_node_ids() {
        let store = LpgStore::new();

        let n1 = store.create_node(&["Person"]);
        let n2 = store.create_node(&["Person"]);
        let n3 = store.create_node(&["Person"]);

        let ids = store.node_ids();
        assert_eq!(ids.len(), 3);
        assert!(ids.contains(&n1));
        assert!(ids.contains(&n2));
        assert!(ids.contains(&n3));

        // Delete one
        store.delete_node(n2);
        let ids = store.node_ids();
        assert_eq!(ids.len(), 2);
        assert!(!ids.contains(&n2));
    }

    #[test]
    fn test_delete_node_nonexistent() {
        let store = LpgStore::new();
        let fake_id = NodeId::new(999);
        assert!(!store.delete_node(fake_id));
    }

    #[test]
    fn test_delete_edge_nonexistent() {
        let store = LpgStore::new();
        let fake_id = EdgeId::new(999);
        assert!(!store.delete_edge(fake_id));
    }

    #[test]
    fn test_delete_edge_double() {
        let store = LpgStore::new();
        let a = store.create_node(&["Person"]);
        let b = store.create_node(&["Person"]);
        let edge_id = store.create_edge(a, b, "KNOWS");

        assert!(store.delete_edge(edge_id));
        assert!(!store.delete_edge(edge_id)); // Double delete
    }

    #[test]
    fn test_create_edge_with_props() {
        let store = LpgStore::new();
        let a = store.create_node(&["Person"]);
        let b = store.create_node(&["Person"]);

        let edge_id = store.create_edge_with_props(
            a,
            b,
            "KNOWS",
            [
                ("since", Value::from(2020i64)),
                ("weight", Value::from(1.0)),
            ],
        );

        let edge = store.get_edge(edge_id).unwrap();
        assert_eq!(
            edge.get_property("since").and_then(|v| v.as_int64()),
            Some(2020)
        );
        assert_eq!(
            edge.get_property("weight").and_then(|v| v.as_float64()),
            Some(1.0)
        );
    }

    #[test]
    fn test_delete_node_edges() {
        let store = LpgStore::new();

        let a = store.create_node(&["Person"]);
        let b = store.create_node(&["Person"]);
        let c = store.create_node(&["Person"]);

        store.create_edge(a, b, "KNOWS"); // a -> b
        store.create_edge(c, a, "KNOWS"); // c -> a

        assert_eq!(store.edge_count(), 2);

        // Delete all edges connected to a
        store.delete_node_edges(a);

        assert_eq!(store.edge_count(), 0);
    }

    #[test]
    fn test_neighbors_both_directions() {
        let store = LpgStore::new();

        let a = store.create_node(&["Person"]);
        let b = store.create_node(&["Person"]);
        let c = store.create_node(&["Person"]);

        store.create_edge(a, b, "KNOWS"); // a -> b
        store.create_edge(c, a, "KNOWS"); // c -> a

        // Direction::Both for node a
        let neighbors: Vec<_> = store.neighbors(a, Direction::Both).collect();
        assert_eq!(neighbors.len(), 2);
        assert!(neighbors.contains(&b)); // outgoing
        assert!(neighbors.contains(&c)); // incoming
    }

    #[test]
    fn test_edges_from() {
        let store = LpgStore::new();

        let a = store.create_node(&["Person"]);
        let b = store.create_node(&["Person"]);
        let c = store.create_node(&["Person"]);

        let e1 = store.create_edge(a, b, "KNOWS");
        let e2 = store.create_edge(a, c, "KNOWS");

        let edges: Vec<_> = store.edges_from(a, Direction::Outgoing).collect();
        assert_eq!(edges.len(), 2);
        assert!(edges.iter().any(|(_, e)| *e == e1));
        assert!(edges.iter().any(|(_, e)| *e == e2));

        // Incoming edges to b
        let incoming: Vec<_> = store.edges_from(b, Direction::Incoming).collect();
        assert_eq!(incoming.len(), 1);
        assert_eq!(incoming[0].1, e1);
    }

    #[test]
    fn test_edges_to() {
        let store = LpgStore::new();

        let a = store.create_node(&["Person"]);
        let b = store.create_node(&["Person"]);
        let c = store.create_node(&["Person"]);

        let e1 = store.create_edge(a, b, "KNOWS");
        let e2 = store.create_edge(c, b, "KNOWS");

        // Edges pointing TO b
        let to_b = store.edges_to(b);
        assert_eq!(to_b.len(), 2);
        assert!(to_b.iter().any(|(src, e)| *src == a && *e == e1));
        assert!(to_b.iter().any(|(src, e)| *src == c && *e == e2));
    }

    #[test]
    fn test_out_degree_in_degree() {
        let store = LpgStore::new();

        let a = store.create_node(&["Person"]);
        let b = store.create_node(&["Person"]);
        let c = store.create_node(&["Person"]);

        store.create_edge(a, b, "KNOWS");
        store.create_edge(a, c, "KNOWS");
        store.create_edge(c, b, "KNOWS");

        assert_eq!(store.out_degree(a), 2);
        assert_eq!(store.out_degree(b), 0);
        assert_eq!(store.out_degree(c), 1);

        assert_eq!(store.in_degree(a), 0);
        assert_eq!(store.in_degree(b), 2);
        assert_eq!(store.in_degree(c), 1);
    }

    #[test]
    fn test_edge_type() {
        let store = LpgStore::new();

        let a = store.create_node(&["Person"]);
        let b = store.create_node(&["Person"]);
        let edge_id = store.create_edge(a, b, "KNOWS");

        let edge_type = store.edge_type(edge_id);
        assert_eq!(edge_type.as_deref(), Some("KNOWS"));

        // Non-existent edge
        let fake_id = EdgeId::new(999);
        assert!(store.edge_type(fake_id).is_none());
    }

    #[test]
    fn test_count_methods() {
        let store = LpgStore::new();

        assert_eq!(store.label_count(), 0);
        assert_eq!(store.edge_type_count(), 0);
        assert_eq!(store.property_key_count(), 0);

        let a = store.create_node_with_props(&["Person"], [("age", Value::from(30i64))]);
        let b = store.create_node(&["Company"]);
        store.create_edge_with_props(a, b, "WORKS_AT", [("since", Value::from(2020i64))]);

        assert_eq!(store.label_count(), 2); // Person, Company
        assert_eq!(store.edge_type_count(), 1); // WORKS_AT
        assert_eq!(store.property_key_count(), 2); // age, since
    }

    #[test]
    fn test_all_nodes_and_edges() {
        let store = LpgStore::new();

        let a = store.create_node(&["Person"]);
        let b = store.create_node(&["Person"]);
        store.create_edge(a, b, "KNOWS");

        let nodes: Vec<_> = store.all_nodes().collect();
        assert_eq!(nodes.len(), 2);

        let edges: Vec<_> = store.all_edges().collect();
        assert_eq!(edges.len(), 1);
    }

    #[test]
    fn test_all_labels_and_edge_types() {
        let store = LpgStore::new();

        store.create_node(&["Person"]);
        store.create_node(&["Company"]);
        let a = store.create_node(&["Animal"]);
        let b = store.create_node(&["Animal"]);
        store.create_edge(a, b, "EATS");

        let labels = store.all_labels();
        assert_eq!(labels.len(), 3);
        assert!(labels.contains(&"Person".to_string()));
        assert!(labels.contains(&"Company".to_string()));
        assert!(labels.contains(&"Animal".to_string()));

        let edge_types = store.all_edge_types();
        assert_eq!(edge_types.len(), 1);
        assert!(edge_types.contains(&"EATS".to_string()));
    }

    #[test]
    fn test_all_property_keys() {
        let store = LpgStore::new();

        let a = store.create_node_with_props(&["Person"], [("name", Value::from("Alice"))]);
        let b = store.create_node_with_props(&["Person"], [("age", Value::from(30i64))]);
        store.create_edge_with_props(a, b, "KNOWS", [("since", Value::from(2020i64))]);

        let keys = store.all_property_keys();
        assert!(keys.contains(&"name".to_string()));
        assert!(keys.contains(&"age".to_string()));
        assert!(keys.contains(&"since".to_string()));
    }

    #[test]
    fn test_nodes_with_label() {
        let store = LpgStore::new();

        store.create_node(&["Person"]);
        store.create_node(&["Person"]);
        store.create_node(&["Company"]);

        let persons: Vec<_> = store.nodes_with_label("Person").collect();
        assert_eq!(persons.len(), 2);

        let companies: Vec<_> = store.nodes_with_label("Company").collect();
        assert_eq!(companies.len(), 1);

        let none: Vec<_> = store.nodes_with_label("NonExistent").collect();
        assert_eq!(none.len(), 0);
    }

    #[test]
    fn test_edges_with_type() {
        let store = LpgStore::new();

        let a = store.create_node(&["Person"]);
        let b = store.create_node(&["Person"]);
        let c = store.create_node(&["Company"]);

        store.create_edge(a, b, "KNOWS");
        store.create_edge(a, c, "WORKS_AT");

        let knows: Vec<_> = store.edges_with_type("KNOWS").collect();
        assert_eq!(knows.len(), 1);

        let works_at: Vec<_> = store.edges_with_type("WORKS_AT").collect();
        assert_eq!(works_at.len(), 1);

        let none: Vec<_> = store.edges_with_type("NonExistent").collect();
        assert_eq!(none.len(), 0);
    }

    #[test]
    fn test_nodes_by_label_nonexistent() {
        let store = LpgStore::new();
        store.create_node(&["Person"]);

        let empty = store.nodes_by_label("NonExistent");
        assert!(empty.is_empty());
    }

    #[test]
    fn test_statistics() {
        let store = LpgStore::new();

        let a = store.create_node(&["Person"]);
        let b = store.create_node(&["Person"]);
        let c = store.create_node(&["Company"]);

        store.create_edge(a, b, "KNOWS");
        store.create_edge(a, c, "WORKS_AT");

        store.compute_statistics();
        let stats = store.statistics();

        assert_eq!(stats.total_nodes, 3);
        assert_eq!(stats.total_edges, 2);

        // Estimates
        let person_card = store.estimate_label_cardinality("Person");
        assert!(person_card > 0.0);

        let avg_degree = store.estimate_avg_degree("KNOWS", true);
        assert!(avg_degree >= 0.0);
    }

    #[test]
    fn test_zone_maps() {
        let store = LpgStore::new();

        store.create_node_with_props(&["Person"], [("age", Value::from(25i64))]);
        store.create_node_with_props(&["Person"], [("age", Value::from(35i64))]);

        // Zone map should indicate possible matches (30 is within [25, 35] range)
        let might_match =
            store.node_property_might_match(&"age".into(), CompareOp::Eq, &Value::from(30i64));
        // Zone maps return true conservatively when value is within min/max range
        assert!(might_match);

        let zone = store.node_property_zone_map(&"age".into());
        assert!(zone.is_some());

        // Non-existent property
        let no_zone = store.node_property_zone_map(&"nonexistent".into());
        assert!(no_zone.is_none());

        // Edge zone maps
        let a = store.create_node(&["A"]);
        let b = store.create_node(&["B"]);
        store.create_edge_with_props(a, b, "REL", [("weight", Value::from(1.0))]);

        let edge_zone = store.edge_property_zone_map(&"weight".into());
        assert!(edge_zone.is_some());
    }

    #[test]
    fn test_rebuild_zone_maps() {
        let store = LpgStore::new();
        store.create_node_with_props(&["Person"], [("age", Value::from(25i64))]);

        // Should not panic
        store.rebuild_zone_maps();
    }

    #[test]
    fn test_create_node_with_id() {
        let store = LpgStore::new();

        let specific_id = NodeId::new(100);
        store.create_node_with_id(specific_id, &["Person", "Employee"]);

        let node = store.get_node(specific_id).unwrap();
        assert!(node.has_label("Person"));
        assert!(node.has_label("Employee"));

        // Next auto-generated ID should be > 100
        let next = store.create_node(&["Other"]);
        assert!(next.as_u64() > 100);
    }

    #[test]
    fn test_create_edge_with_id() {
        let store = LpgStore::new();

        let a = store.create_node(&["A"]);
        let b = store.create_node(&["B"]);

        let specific_id = EdgeId::new(500);
        store.create_edge_with_id(specific_id, a, b, "REL");

        let edge = store.get_edge(specific_id).unwrap();
        assert_eq!(edge.src, a);
        assert_eq!(edge.dst, b);
        assert_eq!(edge.edge_type.as_str(), "REL");

        // Next auto-generated ID should be > 500
        let next = store.create_edge(a, b, "OTHER");
        assert!(next.as_u64() > 500);
    }

    #[test]
    fn test_set_epoch() {
        let store = LpgStore::new();

        assert_eq!(store.current_epoch().as_u64(), 0);

        store.set_epoch(EpochId::new(42));
        assert_eq!(store.current_epoch().as_u64(), 42);
    }

    #[test]
    fn test_get_node_nonexistent() {
        let store = LpgStore::new();
        let fake_id = NodeId::new(999);
        assert!(store.get_node(fake_id).is_none());
    }

    #[test]
    fn test_get_edge_nonexistent() {
        let store = LpgStore::new();
        let fake_id = EdgeId::new(999);
        assert!(store.get_edge(fake_id).is_none());
    }

    #[test]
    fn test_multiple_labels() {
        let store = LpgStore::new();

        let id = store.create_node(&["Person", "Employee", "Manager"]);
        let node = store.get_node(id).unwrap();

        assert!(node.has_label("Person"));
        assert!(node.has_label("Employee"));
        assert!(node.has_label("Manager"));
        assert!(!node.has_label("Other"));
    }

    #[test]
    fn test_default_impl() {
        let store: LpgStore = Default::default();
        assert_eq!(store.node_count(), 0);
        assert_eq!(store.edge_count(), 0);
    }

    #[test]
    fn test_edges_from_both_directions() {
        let store = LpgStore::new();

        let a = store.create_node(&["A"]);
        let b = store.create_node(&["B"]);
        let c = store.create_node(&["C"]);

        let e1 = store.create_edge(a, b, "R1"); // a -> b
        let e2 = store.create_edge(c, a, "R2"); // c -> a

        // Both directions from a
        let edges: Vec<_> = store.edges_from(a, Direction::Both).collect();
        assert_eq!(edges.len(), 2);
        assert!(edges.iter().any(|(_, e)| *e == e1)); // outgoing
        assert!(edges.iter().any(|(_, e)| *e == e2)); // incoming
    }

    #[test]
    fn test_no_backward_adj_in_degree() {
        let config = LpgStoreConfig {
            backward_edges: false,
            initial_node_capacity: 10,
            initial_edge_capacity: 10,
        };
        let store = LpgStore::with_config(config);

        let a = store.create_node(&["A"]);
        let b = store.create_node(&["B"]);
        store.create_edge(a, b, "R");

        // in_degree should still work (falls back to scanning)
        let degree = store.in_degree(b);
        assert_eq!(degree, 1);
    }

    #[test]
    fn test_no_backward_adj_edges_to() {
        let config = LpgStoreConfig {
            backward_edges: false,
            initial_node_capacity: 10,
            initial_edge_capacity: 10,
        };
        let store = LpgStore::with_config(config);

        let a = store.create_node(&["A"]);
        let b = store.create_node(&["B"]);
        let e = store.create_edge(a, b, "R");

        // edges_to should still work (falls back to scanning)
        let edges = store.edges_to(b);
        assert_eq!(edges.len(), 1);
        assert_eq!(edges[0].1, e);
    }

    #[test]
    fn test_node_versioned_creation() {
        let store = LpgStore::new();

        let epoch = store.new_epoch();
        let tx_id = TxId::new(1);

        let id = store.create_node_versioned(&["Person"], epoch, tx_id);
        assert!(store.get_node(id).is_some());
    }

    #[test]
    fn test_edge_versioned_creation() {
        let store = LpgStore::new();

        let a = store.create_node(&["A"]);
        let b = store.create_node(&["B"]);

        let epoch = store.new_epoch();
        let tx_id = TxId::new(1);

        let edge_id = store.create_edge_versioned(a, b, "REL", epoch, tx_id);
        assert!(store.get_edge(edge_id).is_some());
    }

    #[test]
    fn test_node_with_props_versioned() {
        let store = LpgStore::new();

        let epoch = store.new_epoch();
        let tx_id = TxId::new(1);

        let id = store.create_node_with_props_versioned(
            &["Person"],
            [("name", Value::from("Alice"))],
            epoch,
            tx_id,
        );

        let node = store.get_node(id).unwrap();
        assert_eq!(
            node.get_property("name").and_then(|v| v.as_str()),
            Some("Alice")
        );
    }

    #[test]
    fn test_discard_uncommitted_versions() {
        let store = LpgStore::new();

        let epoch = store.new_epoch();
        let tx_id = TxId::new(42);

        // Create node with specific tx
        let node_id = store.create_node_versioned(&["Person"], epoch, tx_id);
        assert!(store.get_node(node_id).is_some());

        // Discard uncommitted versions for this tx
        store.discard_uncommitted_versions(tx_id);

        // Node should be gone (version chain was removed)
        assert!(store.get_node(node_id).is_none());
    }

    // === Property Index Tests ===

    #[test]
    fn test_property_index_create_and_lookup() {
        let store = LpgStore::new();

        // Create nodes with properties
        let alice = store.create_node(&["Person"]);
        let bob = store.create_node(&["Person"]);
        let charlie = store.create_node(&["Person"]);

        store.set_node_property(alice, "city", Value::from("NYC"));
        store.set_node_property(bob, "city", Value::from("NYC"));
        store.set_node_property(charlie, "city", Value::from("LA"));

        // Before indexing, lookup still works (via scan)
        let nyc_people = store.find_nodes_by_property("city", &Value::from("NYC"));
        assert_eq!(nyc_people.len(), 2);

        // Create index
        store.create_property_index("city");
        assert!(store.has_property_index("city"));

        // Indexed lookup should return same results
        let nyc_people = store.find_nodes_by_property("city", &Value::from("NYC"));
        assert_eq!(nyc_people.len(), 2);
        assert!(nyc_people.contains(&alice));
        assert!(nyc_people.contains(&bob));

        let la_people = store.find_nodes_by_property("city", &Value::from("LA"));
        assert_eq!(la_people.len(), 1);
        assert!(la_people.contains(&charlie));
    }

    #[test]
    fn test_property_index_maintained_on_update() {
        let store = LpgStore::new();

        // Create index first
        store.create_property_index("status");

        let node = store.create_node(&["Task"]);
        store.set_node_property(node, "status", Value::from("pending"));

        // Should find by initial value
        let pending = store.find_nodes_by_property("status", &Value::from("pending"));
        assert_eq!(pending.len(), 1);
        assert!(pending.contains(&node));

        // Update the property
        store.set_node_property(node, "status", Value::from("done"));

        // Old value should not find it
        let pending = store.find_nodes_by_property("status", &Value::from("pending"));
        assert!(pending.is_empty());

        // New value should find it
        let done = store.find_nodes_by_property("status", &Value::from("done"));
        assert_eq!(done.len(), 1);
        assert!(done.contains(&node));
    }

    #[test]
    fn test_property_index_maintained_on_remove() {
        let store = LpgStore::new();

        store.create_property_index("tag");

        let node = store.create_node(&["Item"]);
        store.set_node_property(node, "tag", Value::from("important"));

        // Should find it
        let found = store.find_nodes_by_property("tag", &Value::from("important"));
        assert_eq!(found.len(), 1);

        // Remove the property
        store.remove_node_property(node, "tag");

        // Should no longer find it
        let found = store.find_nodes_by_property("tag", &Value::from("important"));
        assert!(found.is_empty());
    }

    #[test]
    fn test_property_index_drop() {
        let store = LpgStore::new();

        store.create_property_index("key");
        assert!(store.has_property_index("key"));

        assert!(store.drop_property_index("key"));
        assert!(!store.has_property_index("key"));

        // Dropping non-existent index returns false
        assert!(!store.drop_property_index("key"));
    }

    #[test]
    fn test_property_index_multiple_values() {
        let store = LpgStore::new();

        store.create_property_index("age");

        // Create multiple nodes with same and different ages
        let n1 = store.create_node(&["Person"]);
        let n2 = store.create_node(&["Person"]);
        let n3 = store.create_node(&["Person"]);
        let n4 = store.create_node(&["Person"]);

        store.set_node_property(n1, "age", Value::from(25i64));
        store.set_node_property(n2, "age", Value::from(25i64));
        store.set_node_property(n3, "age", Value::from(30i64));
        store.set_node_property(n4, "age", Value::from(25i64));

        let age_25 = store.find_nodes_by_property("age", &Value::from(25i64));
        assert_eq!(age_25.len(), 3);

        let age_30 = store.find_nodes_by_property("age", &Value::from(30i64));
        assert_eq!(age_30.len(), 1);

        let age_40 = store.find_nodes_by_property("age", &Value::from(40i64));
        assert!(age_40.is_empty());
    }

    #[test]
    fn test_property_index_builds_from_existing_data() {
        let store = LpgStore::new();

        // Create nodes first
        let n1 = store.create_node(&["Person"]);
        let n2 = store.create_node(&["Person"]);
        store.set_node_property(n1, "email", Value::from("alice@example.com"));
        store.set_node_property(n2, "email", Value::from("bob@example.com"));

        // Create index after data exists
        store.create_property_index("email");

        // Index should include existing data
        let alice = store.find_nodes_by_property("email", &Value::from("alice@example.com"));
        assert_eq!(alice.len(), 1);
        assert!(alice.contains(&n1));

        let bob = store.find_nodes_by_property("email", &Value::from("bob@example.com"));
        assert_eq!(bob.len(), 1);
        assert!(bob.contains(&n2));
    }

    #[test]
    fn test_get_node_property_batch() {
        let store = LpgStore::new();

        let n1 = store.create_node(&["Person"]);
        let n2 = store.create_node(&["Person"]);
        let n3 = store.create_node(&["Person"]);

        store.set_node_property(n1, "age", Value::from(25i64));
        store.set_node_property(n2, "age", Value::from(30i64));
        // n3 has no age property

        let age_key = PropertyKey::new("age");
        let values = store.get_node_property_batch(&[n1, n2, n3], &age_key);

        assert_eq!(values.len(), 3);
        assert_eq!(values[0], Some(Value::from(25i64)));
        assert_eq!(values[1], Some(Value::from(30i64)));
        assert_eq!(values[2], None);
    }

    #[test]
    fn test_get_node_property_batch_empty() {
        let store = LpgStore::new();
        let key = PropertyKey::new("any");

        let values = store.get_node_property_batch(&[], &key);
        assert!(values.is_empty());
    }

    #[test]
    fn test_get_nodes_properties_batch() {
        let store = LpgStore::new();

        let n1 = store.create_node(&["Person"]);
        let n2 = store.create_node(&["Person"]);
        let n3 = store.create_node(&["Person"]);

        store.set_node_property(n1, "name", Value::from("Alice"));
        store.set_node_property(n1, "age", Value::from(25i64));
        store.set_node_property(n2, "name", Value::from("Bob"));
        // n3 has no properties

        let all_props = store.get_nodes_properties_batch(&[n1, n2, n3]);

        assert_eq!(all_props.len(), 3);
        assert_eq!(all_props[0].len(), 2); // name and age
        assert_eq!(all_props[1].len(), 1); // name only
        assert_eq!(all_props[2].len(), 0); // no properties

        assert_eq!(
            all_props[0].get(&PropertyKey::new("name")),
            Some(&Value::from("Alice"))
        );
        assert_eq!(
            all_props[1].get(&PropertyKey::new("name")),
            Some(&Value::from("Bob"))
        );
    }

    #[test]
    fn test_get_nodes_properties_batch_empty() {
        let store = LpgStore::new();

        let all_props = store.get_nodes_properties_batch(&[]);
        assert!(all_props.is_empty());
    }

    #[test]
    fn test_get_nodes_properties_selective_batch() {
        let store = LpgStore::new();

        let n1 = store.create_node(&["Person"]);
        let n2 = store.create_node(&["Person"]);

        // Set multiple properties
        store.set_node_property(n1, "name", Value::from("Alice"));
        store.set_node_property(n1, "age", Value::from(25i64));
        store.set_node_property(n1, "email", Value::from("alice@example.com"));
        store.set_node_property(n2, "name", Value::from("Bob"));
        store.set_node_property(n2, "age", Value::from(30i64));
        store.set_node_property(n2, "city", Value::from("NYC"));

        // Request only name and age (not email or city)
        let keys = vec![PropertyKey::new("name"), PropertyKey::new("age")];
        let props = store.get_nodes_properties_selective_batch(&[n1, n2], &keys);

        assert_eq!(props.len(), 2);

        // n1: should have name and age, but NOT email
        assert_eq!(props[0].len(), 2);
        assert_eq!(
            props[0].get(&PropertyKey::new("name")),
            Some(&Value::from("Alice"))
        );
        assert_eq!(
            props[0].get(&PropertyKey::new("age")),
            Some(&Value::from(25i64))
        );
        assert_eq!(props[0].get(&PropertyKey::new("email")), None);

        // n2: should have name and age, but NOT city
        assert_eq!(props[1].len(), 2);
        assert_eq!(
            props[1].get(&PropertyKey::new("name")),
            Some(&Value::from("Bob"))
        );
        assert_eq!(
            props[1].get(&PropertyKey::new("age")),
            Some(&Value::from(30i64))
        );
        assert_eq!(props[1].get(&PropertyKey::new("city")), None);
    }

    #[test]
    fn test_get_nodes_properties_selective_batch_empty_keys() {
        let store = LpgStore::new();

        let n1 = store.create_node(&["Person"]);
        store.set_node_property(n1, "name", Value::from("Alice"));

        // Request no properties
        let props = store.get_nodes_properties_selective_batch(&[n1], &[]);

        assert_eq!(props.len(), 1);
        assert!(props[0].is_empty()); // Empty map when no keys requested
    }

    #[test]
    fn test_get_nodes_properties_selective_batch_missing_keys() {
        let store = LpgStore::new();

        let n1 = store.create_node(&["Person"]);
        store.set_node_property(n1, "name", Value::from("Alice"));

        // Request a property that doesn't exist
        let keys = vec![PropertyKey::new("nonexistent"), PropertyKey::new("name")];
        let props = store.get_nodes_properties_selective_batch(&[n1], &keys);

        assert_eq!(props.len(), 1);
        assert_eq!(props[0].len(), 1); // Only name exists
        assert_eq!(
            props[0].get(&PropertyKey::new("name")),
            Some(&Value::from("Alice"))
        );
    }

    // === Range Query Tests ===

    #[test]
    fn test_find_nodes_in_range_inclusive() {
        let store = LpgStore::new();

        let n1 = store.create_node_with_props(&["Person"], [("age", Value::from(20i64))]);
        let n2 = store.create_node_with_props(&["Person"], [("age", Value::from(30i64))]);
        let n3 = store.create_node_with_props(&["Person"], [("age", Value::from(40i64))]);
        let _n4 = store.create_node_with_props(&["Person"], [("age", Value::from(50i64))]);

        // age >= 20 AND age <= 40 (inclusive both sides)
        let result = store.find_nodes_in_range(
            "age",
            Some(&Value::from(20i64)),
            Some(&Value::from(40i64)),
            true,
            true,
        );
        assert_eq!(result.len(), 3);
        assert!(result.contains(&n1));
        assert!(result.contains(&n2));
        assert!(result.contains(&n3));
    }

    #[test]
    fn test_find_nodes_in_range_exclusive() {
        let store = LpgStore::new();

        store.create_node_with_props(&["Person"], [("age", Value::from(20i64))]);
        let n2 = store.create_node_with_props(&["Person"], [("age", Value::from(30i64))]);
        store.create_node_with_props(&["Person"], [("age", Value::from(40i64))]);

        // age > 20 AND age < 40 (exclusive both sides)
        let result = store.find_nodes_in_range(
            "age",
            Some(&Value::from(20i64)),
            Some(&Value::from(40i64)),
            false,
            false,
        );
        assert_eq!(result.len(), 1);
        assert!(result.contains(&n2));
    }

    #[test]
    fn test_find_nodes_in_range_open_ended() {
        let store = LpgStore::new();

        store.create_node_with_props(&["Person"], [("age", Value::from(20i64))]);
        store.create_node_with_props(&["Person"], [("age", Value::from(30i64))]);
        let n3 = store.create_node_with_props(&["Person"], [("age", Value::from(40i64))]);
        let n4 = store.create_node_with_props(&["Person"], [("age", Value::from(50i64))]);

        // age >= 35 (no upper bound)
        let result = store.find_nodes_in_range("age", Some(&Value::from(35i64)), None, true, true);
        assert_eq!(result.len(), 2);
        assert!(result.contains(&n3));
        assert!(result.contains(&n4));

        // age <= 25 (no lower bound)
        let result = store.find_nodes_in_range("age", None, Some(&Value::from(25i64)), true, true);
        assert_eq!(result.len(), 1);
    }

    #[test]
    fn test_find_nodes_in_range_empty_result() {
        let store = LpgStore::new();

        store.create_node_with_props(&["Person"], [("age", Value::from(20i64))]);

        // Range that doesn't match anything
        let result = store.find_nodes_in_range(
            "age",
            Some(&Value::from(100i64)),
            Some(&Value::from(200i64)),
            true,
            true,
        );
        assert!(result.is_empty());
    }

    #[test]
    fn test_find_nodes_in_range_nonexistent_property() {
        let store = LpgStore::new();

        store.create_node_with_props(&["Person"], [("age", Value::from(20i64))]);

        let result = store.find_nodes_in_range(
            "weight",
            Some(&Value::from(50i64)),
            Some(&Value::from(100i64)),
            true,
            true,
        );
        assert!(result.is_empty());
    }

    // === Multi-Property Query Tests ===

    #[test]
    fn test_find_nodes_by_properties_multiple_conditions() {
        let store = LpgStore::new();

        let alice = store.create_node_with_props(
            &["Person"],
            [("name", Value::from("Alice")), ("city", Value::from("NYC"))],
        );
        store.create_node_with_props(
            &["Person"],
            [("name", Value::from("Bob")), ("city", Value::from("NYC"))],
        );
        store.create_node_with_props(
            &["Person"],
            [("name", Value::from("Alice")), ("city", Value::from("LA"))],
        );

        // Match name="Alice" AND city="NYC"
        let result = store.find_nodes_by_properties(&[
            ("name", Value::from("Alice")),
            ("city", Value::from("NYC")),
        ]);
        assert_eq!(result.len(), 1);
        assert!(result.contains(&alice));
    }

    #[test]
    fn test_find_nodes_by_properties_empty_conditions() {
        let store = LpgStore::new();

        store.create_node(&["Person"]);
        store.create_node(&["Person"]);

        // Empty conditions should return all nodes
        let result = store.find_nodes_by_properties(&[]);
        assert_eq!(result.len(), 2);
    }

    #[test]
    fn test_find_nodes_by_properties_no_match() {
        let store = LpgStore::new();

        store.create_node_with_props(&["Person"], [("name", Value::from("Alice"))]);

        let result = store.find_nodes_by_properties(&[("name", Value::from("Nobody"))]);
        assert!(result.is_empty());
    }

    #[test]
    fn test_find_nodes_by_properties_with_index() {
        let store = LpgStore::new();

        // Create index on name
        store.create_property_index("name");

        let alice = store.create_node_with_props(
            &["Person"],
            [("name", Value::from("Alice")), ("age", Value::from(30i64))],
        );
        store.create_node_with_props(
            &["Person"],
            [("name", Value::from("Bob")), ("age", Value::from(30i64))],
        );

        // Index should accelerate the lookup
        let result = store.find_nodes_by_properties(&[
            ("name", Value::from("Alice")),
            ("age", Value::from(30i64)),
        ]);
        assert_eq!(result.len(), 1);
        assert!(result.contains(&alice));
    }

    // === Cardinality Estimation Tests ===

    #[test]
    fn test_estimate_label_cardinality() {
        let store = LpgStore::new();

        store.create_node(&["Person"]);
        store.create_node(&["Person"]);
        store.create_node(&["Animal"]);

        store.ensure_statistics_fresh();

        let person_est = store.estimate_label_cardinality("Person");
        let animal_est = store.estimate_label_cardinality("Animal");
        let unknown_est = store.estimate_label_cardinality("Unknown");

        assert!(
            person_est >= 1.0,
            "Person should have cardinality >= 1, got {person_est}"
        );
        assert!(
            animal_est >= 1.0,
            "Animal should have cardinality >= 1, got {animal_est}"
        );
        // Unknown label should return some default (not panic)
        assert!(unknown_est >= 0.0);
    }

    #[test]
    fn test_estimate_avg_degree() {
        let store = LpgStore::new();

        let a = store.create_node(&["Person"]);
        let b = store.create_node(&["Person"]);
        let c = store.create_node(&["Person"]);

        store.create_edge(a, b, "KNOWS");
        store.create_edge(a, c, "KNOWS");
        store.create_edge(b, c, "KNOWS");

        store.ensure_statistics_fresh();

        let outgoing = store.estimate_avg_degree("KNOWS", true);
        let incoming = store.estimate_avg_degree("KNOWS", false);

        assert!(
            outgoing > 0.0,
            "Outgoing degree should be > 0, got {outgoing}"
        );
        assert!(
            incoming > 0.0,
            "Incoming degree should be > 0, got {incoming}"
        );
    }

    // === Delete operations ===

    #[test]
    fn test_delete_node_does_not_cascade() {
        let store = LpgStore::new();

        let a = store.create_node(&["A"]);
        let b = store.create_node(&["B"]);
        let e = store.create_edge(a, b, "KNOWS");

        assert!(store.delete_node(a));
        assert!(store.get_node(a).is_none());

        // Edges are NOT automatically deleted (non-detach delete)
        assert!(
            store.get_edge(e).is_some(),
            "Edge should survive non-detach node delete"
        );
    }

    #[test]
    fn test_delete_already_deleted_node() {
        let store = LpgStore::new();
        let a = store.create_node(&["A"]);

        assert!(store.delete_node(a));
        // Second delete should return false (already deleted)
        assert!(!store.delete_node(a));
    }

    #[test]
    fn test_delete_nonexistent_node() {
        let store = LpgStore::new();
        assert!(!store.delete_node(NodeId::new(999)));
    }
}
