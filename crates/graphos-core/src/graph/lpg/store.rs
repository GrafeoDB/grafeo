//! LPG graph store implementation.

use super::property::CompareOp;
use super::{Edge, EdgeRecord, Node, NodeRecord, PropertyStorage};
use crate::graph::Direction;
use crate::index::adjacency::ChunkedAdjacency;
use crate::index::zone_map::ZoneMapEntry;
use crate::statistics::{EdgeTypeStatistics, LabelStatistics, Statistics};
use graphos_common::types::{EdgeId, EpochId, NodeId, PropertyKey, Value};
use graphos_common::utils::hash::FxHashMap;
use parking_lot::RwLock;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

/// Configuration for the LPG store.
#[derive(Debug, Clone)]
pub struct LpgStoreConfig {
    /// Whether to maintain backward adjacency lists.
    pub backward_edges: bool,
    /// Initial capacity for nodes.
    pub initial_node_capacity: usize,
    /// Initial capacity for edges.
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

/// The main LPG graph store.
///
/// This is the core storage for labeled property graphs, providing
/// efficient node/edge storage and adjacency indexing.
pub struct LpgStore {
    /// Configuration.
    #[allow(dead_code)]
    config: LpgStoreConfig,

    /// Node records indexed by NodeId.
    nodes: RwLock<FxHashMap<NodeId, NodeRecord>>,

    /// Edge records indexed by EdgeId.
    edges: RwLock<FxHashMap<EdgeId, EdgeRecord>>,

    /// Property storage for nodes.
    node_properties: PropertyStorage<NodeId>,

    /// Property storage for edges.
    edge_properties: PropertyStorage<EdgeId>,

    /// Label name to ID mapping.
    label_to_id: RwLock<FxHashMap<Arc<str>, u8>>,

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
            nodes: RwLock::new(FxHashMap::default()),
            edges: RwLock::new(FxHashMap::default()),
            node_properties: PropertyStorage::new(),
            edge_properties: PropertyStorage::new(),
            label_to_id: RwLock::new(FxHashMap::default()),
            id_to_label: RwLock::new(Vec::new()),
            edge_type_to_id: RwLock::new(FxHashMap::default()),
            id_to_edge_type: RwLock::new(Vec::new()),
            forward_adj: ChunkedAdjacency::new(),
            backward_adj,
            label_index: RwLock::new(Vec::new()),
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
    pub fn create_node(&self, labels: &[&str]) -> NodeId {
        let id = NodeId::new(self.next_node_id.fetch_add(1, Ordering::Relaxed));
        let epoch = self.current_epoch();

        let mut record = NodeRecord::new(id, epoch);

        // Set label bits
        for label in labels {
            let label_id = self.get_or_create_label_id(*label);
            record.set_label_bit(label_id);

            // Update label index
            let mut index = self.label_index.write();
            while index.len() <= label_id as usize {
                index.push(FxHashMap::default());
            }
            index[label_id as usize].insert(id, ());
        }

        self.nodes.write().insert(id, record);
        id
    }

    /// Creates a new node with labels and properties.
    pub fn create_node_with_props(
        &self,
        labels: &[&str],
        properties: impl IntoIterator<Item = (impl Into<PropertyKey>, impl Into<Value>)>,
    ) -> NodeId {
        let id = self.create_node(labels);

        for (key, value) in properties {
            self.node_properties.set(id, key.into(), value.into());
        }

        // Update props_count in record
        let count = self.node_properties.get_all(id).len() as u16;
        if let Some(record) = self.nodes.write().get_mut(&id) {
            record.props_count = count;
        }

        id
    }

    /// Gets a node by ID.
    #[must_use]
    pub fn get_node(&self, id: NodeId) -> Option<Node> {
        let nodes = self.nodes.read();
        let record = nodes.get(&id)?;

        if record.is_deleted() {
            return None;
        }

        let mut node = Node::new(id);

        // Get labels
        let id_to_label = self.id_to_label.read();
        for bit in record.label_bits_iter() {
            if let Some(label) = id_to_label.get(bit as usize) {
                node.labels.push(label.clone());
            }
        }

        // Get properties
        node.properties = self.node_properties.get_all(id).into_iter().collect();

        Some(node)
    }

    /// Deletes a node and all its edges.
    pub fn delete_node(&self, id: NodeId) -> bool {
        let mut nodes = self.nodes.write();
        if let Some(record) = nodes.get_mut(&id) {
            if record.is_deleted() {
                return false;
            }

            record.set_deleted(true);

            // Remove from label index
            let mut index = self.label_index.write();
            for bit in record.label_bits_iter() {
                if let Some(set) = index.get_mut(bit as usize) {
                    set.remove(&id);
                }
            }

            // Remove properties
            drop(nodes); // Release lock before removing properties
            self.node_properties.remove_all(id);

            // FIXME(delete-node): Delete incident edges when removing node

            true
        } else {
            false
        }
    }

    /// Returns the number of nodes.
    #[must_use]
    pub fn node_count(&self) -> usize {
        self.nodes
            .read()
            .values()
            .filter(|r| !r.is_deleted())
            .count()
    }

    /// Returns all node IDs in the store.
    ///
    /// This returns a snapshot of current node IDs. The returned vector
    /// excludes deleted nodes.
    #[must_use]
    pub fn node_ids(&self) -> Vec<NodeId> {
        self.nodes
            .read()
            .iter()
            .filter(|(_, r)| !r.is_deleted())
            .map(|(id, _)| *id)
            .collect()
    }

    // === Edge Operations ===

    /// Creates a new edge.
    pub fn create_edge(&self, src: NodeId, dst: NodeId, edge_type: &str) -> EdgeId {
        let id = EdgeId::new(self.next_edge_id.fetch_add(1, Ordering::Relaxed));
        let epoch = self.current_epoch();
        let type_id = self.get_or_create_edge_type_id(edge_type);

        let record = EdgeRecord::new(id, src, dst, type_id, epoch);
        self.edges.write().insert(id, record);

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

    /// Gets an edge by ID.
    #[must_use]
    pub fn get_edge(&self, id: EdgeId) -> Option<Edge> {
        let edges = self.edges.read();
        let record = edges.get(&id)?;

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

    /// Deletes an edge.
    pub fn delete_edge(&self, id: EdgeId) -> bool {
        let mut edges = self.edges.write();
        if let Some(record) = edges.get_mut(&id) {
            if record.is_deleted() {
                return false;
            }

            let src = record.src;
            let dst = record.dst;

            record.set_deleted(true);

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

    /// Returns the number of edges.
    #[must_use]
    pub fn edge_count(&self) -> usize {
        self.edges
            .read()
            .values()
            .filter(|r| !r.is_deleted())
            .count()
    }

    // === Traversal ===

    /// Returns an iterator over neighbors of a node.
    pub fn neighbors(
        &self,
        node: NodeId,
        direction: Direction,
    ) -> impl Iterator<Item = NodeId> + '_ {
        let forward: Box<dyn Iterator<Item = NodeId>> = match direction {
            Direction::Outgoing | Direction::Both => Box::new(self.forward_adj.neighbors(node)),
            Direction::Incoming => Box::new(std::iter::empty()),
        };

        let backward: Box<dyn Iterator<Item = NodeId>> = match direction {
            Direction::Incoming | Direction::Both => {
                if let Some(ref adj) = self.backward_adj {
                    Box::new(adj.neighbors(node))
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
            Direction::Outgoing | Direction::Both => Box::new(self.forward_adj.edges_from(node)),
            Direction::Incoming => Box::new(std::iter::empty()),
        };

        let backward: Box<dyn Iterator<Item = (NodeId, EdgeId)>> = match direction {
            Direction::Incoming | Direction::Both => {
                if let Some(ref adj) = self.backward_adj {
                    Box::new(adj.edges_from(node))
                } else {
                    Box::new(std::iter::empty())
                }
            }
            Direction::Outgoing => Box::new(std::iter::empty()),
        };

        forward.chain(backward)
    }

    /// Gets the type of an edge by ID.
    #[must_use]
    pub fn edge_type(&self, id: EdgeId) -> Option<Arc<str>> {
        let edges = self.edges.read();
        let record = edges.get(&id)?;
        let id_to_type = self.id_to_edge_type.read();
        id_to_type.get(record.type_id as usize).cloned()
    }

    /// Returns nodes with a specific label.
    pub fn nodes_by_label(&self, label: &str) -> Vec<NodeId> {
        let label_to_id = self.label_to_id.read();
        if let Some(&label_id) = label_to_id.get(label) {
            let index = self.label_index.read();
            if let Some(set) = index.get(label_id as usize) {
                return set.keys().copied().collect();
            }
        }
        Vec::new()
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

    /// Updates statistics from current data.
    ///
    /// This scans all labels and edge types to compute cardinality statistics.
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

                let label_stats = LabelStatistics::new(node_count)
                    .with_degrees(avg_out_degree, avg_out_degree);

                stats.update_label(label_name.as_ref(), label_stats);
            }
        }

        // Compute per-edge-type statistics
        let id_to_edge_type = self.id_to_edge_type.read();
        let edges = self.edges.read();

        let mut edge_type_counts: FxHashMap<u32, u64> = FxHashMap::default();
        for record in edges.values() {
            if !record.is_deleted() {
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
        self.statistics.read().estimate_avg_degree(edge_type, outgoing)
    }

    // === Internal Helpers ===

    fn get_or_create_label_id(&self, label: &str) -> u8 {
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

        let id = id_to_label.len() as u8;
        assert!(id < 64, "Maximum 64 labels supported");

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
