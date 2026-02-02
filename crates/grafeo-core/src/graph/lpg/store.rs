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
#[cfg(not(feature = "tiered-storage"))]
use grafeo_common::mvcc::VersionChain;
use grafeo_common::types::{EdgeId, EpochId, NodeId, PropertyKey, TxId, Value};
use grafeo_common::utils::hash::{FxHashMap, FxHashSet};
use parking_lot::RwLock;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

// Tiered storage imports
#[cfg(feature = "tiered-storage")]
use grafeo_common::memory::arena::ArenaAllocator;
#[cfg(feature = "tiered-storage")]
use grafeo_common::mvcc::{ColdVersionRef, HotVersionRef, VersionIndex, VersionRef};
#[cfg(feature = "tiered-storage")]
use crate::storage::EpochStore;

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
pub struct LpgStore {
    /// Configuration.
    #[allow(dead_code)]
    config: LpgStoreConfig,

    /// Node records indexed by NodeId, with version chains for MVCC.
    /// Used when `tiered-storage` feature is disabled.
    #[cfg(not(feature = "tiered-storage"))]
    nodes: RwLock<FxHashMap<NodeId, VersionChain<NodeRecord>>>,

    /// Edge records indexed by EdgeId, with version chains for MVCC.
    /// Used when `tiered-storage` feature is disabled.
    #[cfg(not(feature = "tiered-storage"))]
    edges: RwLock<FxHashMap<EdgeId, VersionChain<EdgeRecord>>>,

    // === Tiered Storage Fields (feature-gated) ===

    /// Arena allocator for hot data storage.
    /// Data is stored in per-epoch arenas for fast allocation and bulk deallocation.
    #[cfg(feature = "tiered-storage")]
    arena_allocator: Arc<ArenaAllocator>,

    /// Node version indexes - store metadata and arena offsets.
    /// The actual NodeRecord data is stored in the arena.
    #[cfg(feature = "tiered-storage")]
    node_versions: RwLock<FxHashMap<NodeId, VersionIndex>>,

    /// Edge version indexes - store metadata and arena offsets.
    /// The actual EdgeRecord data is stored in the arena.
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
    label_to_id: RwLock<FxHashMap<Arc<str>, u32>>,

    /// Label ID to name mapping.
    id_to_label: RwLock<Vec<Arc<str>>>,

    /// Edge type name to ID mapping.
    edge_type_to_id: RwLock<FxHashMap<Arc<str>, u32>>,

    /// Edge type ID to name mapping.
    id_to_edge_type: RwLock<Vec<Arc<str>>>,

    /// Forward adjacency lists (outgoing edges).
    forward_adj: ChunkedAdjacency,

    /// Backward adjacency lists (incoming edges).
    /// Only populated if config.backward_edges is true.
    backward_adj: Option<ChunkedAdjacency>,

    /// Label index: label_id -> set of node IDs.
    label_index: RwLock<Vec<FxHashMap<NodeId, ()>>>,

    /// Node labels: node_id -> set of label IDs.
    /// Reverse mapping to efficiently get labels for a node.
    node_labels: RwLock<FxHashMap<NodeId, FxHashSet<u32>>>,

    /// Next node ID.
    next_node_id: AtomicU64,

    /// Next edge ID.
    next_edge_id: AtomicU64,

    /// Current epoch.
    current_epoch: AtomicU64,

    /// Statistics for cost-based optimization.
    statistics: RwLock<Statistics>,
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
            next_node_id: AtomicU64::new(0),
            next_edge_id: AtomicU64::new(0),
            current_epoch: AtomicU64::new(0),
            statistics: RwLock::new(Statistics::new()),
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
            self.node_properties.set(id, key.into(), value.into());
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
            self.node_properties.set(id, key.into(), value.into());
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
                Some(record.clone())
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
        self.node_properties.set(id, key.into(), value);

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
        self.node_properties.set(id, key.into(), value);
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
        let result = self.node_properties.remove(id, &key.into());

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
        self.node_properties.remove(id, &key.into())
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
                    self.read_node_record(&vref).and_then(|r| {
                        if !r.is_deleted() {
                            Some(*id)
                        } else {
                            None
                        }
                    })
                })
            })
            .collect();
        ids.sort_unstable();
        ids
    }

    // === Edge Operations ===

    /// Creates a new edge.
    pub fn create_edge(&self, src: NodeId, dst: NodeId, edge_type: &str) -> EdgeId {
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
                Some(record.clone())
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
                    node_records.push((node_id.as_u64(), record.clone()));
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
                    edge_records.push((edge_id.as_u64(), record.clone()));
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
            self.epoch_store.freeze_epoch(epoch, node_records, edge_records);

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
                if let Some(index) = versions.get_mut(node_id) {
                    if let Some(&(offset, length)) = node_entry_map.get(&node_id.as_u64()) {
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
        }

        {
            let mut versions = self.edge_versions.write();
            for (edge_id, hot_ref) in &edge_hot_refs {
                if let Some(index) = versions.get_mut(edge_id) {
                    if let Some(&(offset, length)) = edge_entry_map.get(&edge_id.as_u64()) {
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
    pub fn edge_type(&self, id: EdgeId) -> Option<Arc<str>> {
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
    pub fn edge_type(&self, id: EdgeId) -> Option<Arc<str>> {
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
                    self.read_edge_record(&vref).and_then(|r| {
                        if !r.is_deleted() {
                            Some(*id)
                        } else {
                            None
                        }
                    })
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
            let node_count = label_index
                .get(label_id)
                .map(|set| set.len() as u64)
                .unwrap_or(0);

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
            if let Some(vref) = index.visible_at(epoch) {
                if let Some(record) = self.read_edge_record(&vref) {
                    if !record.is_deleted() {
                        *edge_type_counts.entry(record.type_id).or_default() += 1;
                    }
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

        let label: Arc<str> = label.into();
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
        let edge_type: Arc<str> = edge_type.into();
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
        assert_eq!(edge.edge_type.as_ref(), "KNOWS");
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
}
