//! WAL overlay: in-memory mutation layer on top of mmap'd base data.
//!
//! When the LPG section is in the `OnDisk` tier (served via mmap), all
//! mutations are captured in this overlay. Reads check the overlay first,
//! then fall through to the mmap'd base. On checkpoint, the overlay is
//! merged into the section data and cleared.
//!
//! The overlay tracks three kinds of mutations:
//! - **Node mutations**: insert, update (labels/properties), delete
//! - **Edge mutations**: insert, update (properties), delete
//! - **Property mutations**: per-entity, per-key value changes
//!
//! The overlay does NOT store adjacency or index changes. Those are derived
//! from node/edge data on checkpoint (same as deserialization rebuild).

use std::collections::HashMap;

use grafeo_common::types::{EdgeId, NodeId, Value};
use parking_lot::RwLock;

/// A mutation recorded in the overlay.
#[derive(Debug, Clone)]
pub enum OverlayOp<T> {
    /// A new entity was inserted.
    Insert(T),
    /// An existing entity was updated (replacement data).
    Update(T),
    /// An entity was deleted.
    Delete,
}

impl<T> OverlayOp<T> {
    /// Returns `true` if this is a delete operation.
    #[must_use]
    pub fn is_delete(&self) -> bool {
        matches!(self, Self::Delete)
    }

    /// Returns the data if this is an insert or update.
    #[must_use]
    pub fn data(&self) -> Option<&T> {
        match self {
            Self::Insert(d) | Self::Update(d) => Some(d),
            Self::Delete => None,
        }
    }
}

/// Node data captured in the overlay.
#[derive(Debug, Clone)]
pub struct OverlayNode {
    /// Node labels.
    pub labels: Vec<String>,
    /// Node properties (latest values only).
    pub properties: HashMap<String, Value>,
}

/// Edge data captured in the overlay.
#[derive(Debug, Clone)]
pub struct OverlayEdge {
    /// Source node.
    pub src: NodeId,
    /// Destination node.
    pub dst: NodeId,
    /// Edge type.
    pub edge_type: String,
    /// Edge properties (latest values only).
    pub properties: HashMap<String, Value>,
}

/// In-memory overlay for mutations on top of mmap'd base data.
///
/// Thread-safe via internal `RwLock`s. Each mutation type has its own
/// lock to minimize contention.
#[derive(Debug)]
pub struct WalOverlay {
    /// Node mutations keyed by NodeId.
    nodes: RwLock<hashbrown::HashMap<NodeId, OverlayOp<OverlayNode>>>,
    /// Edge mutations keyed by EdgeId.
    edges: RwLock<hashbrown::HashMap<EdgeId, OverlayOp<OverlayEdge>>>,
    /// Count of mutations since last clear (for checkpoint decisions).
    mutation_count: std::sync::atomic::AtomicUsize,
}

impl WalOverlay {
    /// Creates a new empty overlay.
    #[must_use]
    pub fn new() -> Self {
        Self {
            nodes: RwLock::new(hashbrown::HashMap::new()),
            edges: RwLock::new(hashbrown::HashMap::new()),
            mutation_count: std::sync::atomic::AtomicUsize::new(0),
        }
    }

    // ── Node operations ────────────────────────────────────────────

    /// Records a node insertion.
    pub fn insert_node(&self, id: NodeId, labels: Vec<String>) {
        self.nodes.write().insert(
            id,
            OverlayOp::Insert(OverlayNode {
                labels,
                properties: HashMap::new(),
            }),
        );
        self.mutation_count
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }

    /// Records a node deletion.
    pub fn delete_node(&self, id: NodeId) {
        self.nodes.write().insert(id, OverlayOp::Delete);
        self.mutation_count
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }

    /// Records a node property change.
    pub fn set_node_property(&self, id: NodeId, key: String, value: Value) {
        let mut guard = self.nodes.write();
        match guard.get_mut(&id) {
            Some(OverlayOp::Insert(node)) | Some(OverlayOp::Update(node)) => {
                node.properties.insert(key, value);
            }
            Some(OverlayOp::Delete) => {
                // Node was deleted, ignore property set
            }
            None => {
                // Node exists in base data, record an update
                let mut props = HashMap::new();
                props.insert(key, value);
                guard.insert(
                    id,
                    OverlayOp::Update(OverlayNode {
                        labels: Vec::new(), // base labels unchanged
                        properties: props,
                    }),
                );
            }
        }
        self.mutation_count
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }

    /// Records a label addition to a node.
    pub fn add_node_label(&self, id: NodeId, label: String) {
        let mut guard = self.nodes.write();
        match guard.get_mut(&id) {
            Some(OverlayOp::Insert(node)) | Some(OverlayOp::Update(node)) => {
                if !node.labels.contains(&label) {
                    node.labels.push(label);
                }
            }
            Some(OverlayOp::Delete) => {}
            None => {
                guard.insert(
                    id,
                    OverlayOp::Update(OverlayNode {
                        labels: vec![label],
                        properties: HashMap::new(),
                    }),
                );
            }
        }
        self.mutation_count
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }

    /// Looks up a node in the overlay. Returns `None` if the node is not
    /// in the overlay (caller should check base data).
    #[must_use]
    pub fn get_node(&self, id: NodeId) -> Option<OverlayOp<OverlayNode>> {
        self.nodes.read().get(&id).cloned()
    }

    // ── Edge operations ────────────────────────────────────────────

    /// Records an edge insertion.
    pub fn insert_edge(&self, id: EdgeId, src: NodeId, dst: NodeId, edge_type: String) {
        self.edges.write().insert(
            id,
            OverlayOp::Insert(OverlayEdge {
                src,
                dst,
                edge_type,
                properties: HashMap::new(),
            }),
        );
        self.mutation_count
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }

    /// Records an edge deletion.
    pub fn delete_edge(&self, id: EdgeId) {
        self.edges.write().insert(id, OverlayOp::Delete);
        self.mutation_count
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }

    /// Records an edge property change.
    pub fn set_edge_property(&self, id: EdgeId, key: String, value: Value) {
        let mut guard = self.edges.write();
        match guard.get_mut(&id) {
            Some(OverlayOp::Insert(edge)) | Some(OverlayOp::Update(edge)) => {
                edge.properties.insert(key, value);
            }
            Some(OverlayOp::Delete) => {}
            None => {
                let mut props = HashMap::new();
                props.insert(key, value);
                guard.insert(
                    id,
                    OverlayOp::Update(OverlayEdge {
                        src: NodeId::new(0), // base values unchanged
                        dst: NodeId::new(0),
                        edge_type: String::new(),
                        properties: props,
                    }),
                );
            }
        }
        self.mutation_count
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }

    /// Looks up an edge in the overlay.
    #[must_use]
    pub fn get_edge(&self, id: EdgeId) -> Option<OverlayOp<OverlayEdge>> {
        self.edges.read().get(&id).cloned()
    }

    // ── Overlay state ──────────────────────────────────────────────

    /// Returns the total number of mutations recorded since the last clear.
    #[must_use]
    pub fn mutation_count(&self) -> usize {
        self.mutation_count
            .load(std::sync::atomic::Ordering::Relaxed)
    }

    /// Returns `true` if the overlay has any mutations.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.mutation_count() == 0
    }

    /// Returns the number of node entries in the overlay.
    #[must_use]
    pub fn node_count(&self) -> usize {
        self.nodes.read().len()
    }

    /// Returns the number of edge entries in the overlay.
    #[must_use]
    pub fn edge_count(&self) -> usize {
        self.edges.read().len()
    }

    /// Clears all overlay data. Called after a successful checkpoint
    /// that merged the overlay into the section.
    pub fn clear(&self) {
        self.nodes.write().clear();
        self.edges.write().clear();
        self.mutation_count
            .store(0, std::sync::atomic::Ordering::Relaxed);
    }

    /// Drains all node mutations from the overlay, returning them.
    pub fn drain_nodes(&self) -> hashbrown::HashMap<NodeId, OverlayOp<OverlayNode>> {
        let mut guard = self.nodes.write();
        std::mem::take(&mut *guard)
    }

    /// Drains all edge mutations from the overlay, returning them.
    pub fn drain_edges(&self) -> hashbrown::HashMap<EdgeId, OverlayOp<OverlayEdge>> {
        let mut guard = self.edges.write();
        std::mem::take(&mut *guard)
    }

    /// Approximate memory usage in bytes.
    #[must_use]
    pub fn approximate_memory_bytes(&self) -> usize {
        let node_count = self.nodes.read().len();
        let edge_count = self.edges.read().len();
        // ~256 bytes per node entry, ~256 per edge entry (rough estimate)
        (node_count + edge_count) * 256
    }
}

impl Default for WalOverlay {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty_overlay() {
        let overlay = WalOverlay::new();
        assert!(overlay.is_empty());
        assert_eq!(overlay.mutation_count(), 0);
        assert_eq!(overlay.node_count(), 0);
        assert_eq!(overlay.edge_count(), 0);
    }

    #[test]
    fn test_node_insert_and_lookup() {
        let overlay = WalOverlay::new();
        overlay.insert_node(NodeId::new(1), vec!["Person".to_string()]);

        let op = overlay.get_node(NodeId::new(1)).unwrap();
        let node = op.data().unwrap();
        assert_eq!(node.labels, vec!["Person"]);
        assert!(node.properties.is_empty());
        assert_eq!(overlay.mutation_count(), 1);
    }

    #[test]
    fn test_node_property_on_insert() {
        let overlay = WalOverlay::new();
        overlay.insert_node(NodeId::new(1), vec!["Person".to_string()]);
        overlay.set_node_property(
            NodeId::new(1),
            "name".to_string(),
            Value::String("Alix".into()),
        );

        let node = overlay.get_node(NodeId::new(1)).unwrap();
        let data = node.data().unwrap();
        assert_eq!(
            data.properties.get("name"),
            Some(&Value::String("Alix".into()))
        );
    }

    #[test]
    fn test_node_property_on_base() {
        let overlay = WalOverlay::new();
        // Node 1 is in base data (not in overlay), set a property
        overlay.set_node_property(
            NodeId::new(1),
            "name".to_string(),
            Value::String("Gus".into()),
        );

        let op = overlay.get_node(NodeId::new(1)).unwrap();
        match op {
            OverlayOp::Update(node) => {
                assert_eq!(
                    node.properties.get("name"),
                    Some(&Value::String("Gus".into()))
                );
            }
            _ => panic!("expected Update for base node property change"),
        }
    }

    #[test]
    fn test_node_delete() {
        let overlay = WalOverlay::new();
        overlay.insert_node(NodeId::new(1), vec!["Person".to_string()]);
        overlay.delete_node(NodeId::new(1));

        let op = overlay.get_node(NodeId::new(1)).unwrap();
        assert!(op.is_delete());
    }

    #[test]
    fn test_node_delete_ignores_property_set() {
        let overlay = WalOverlay::new();
        overlay.delete_node(NodeId::new(1));
        overlay.set_node_property(
            NodeId::new(1),
            "name".to_string(),
            Value::String("X".into()),
        );

        // Delete should persist (property set on deleted node is a no-op)
        let op = overlay.get_node(NodeId::new(1)).unwrap();
        assert!(op.is_delete());
    }

    #[test]
    fn test_edge_insert_and_lookup() {
        let overlay = WalOverlay::new();
        overlay.insert_edge(
            EdgeId::new(1),
            NodeId::new(1),
            NodeId::new(2),
            "KNOWS".to_string(),
        );

        let op = overlay.get_edge(EdgeId::new(1)).unwrap();
        let edge = op.data().unwrap();
        assert_eq!(edge.src, NodeId::new(1));
        assert_eq!(edge.dst, NodeId::new(2));
        assert_eq!(edge.edge_type, "KNOWS");
    }

    #[test]
    fn test_edge_delete() {
        let overlay = WalOverlay::new();
        overlay.insert_edge(
            EdgeId::new(1),
            NodeId::new(1),
            NodeId::new(2),
            "KNOWS".to_string(),
        );
        overlay.delete_edge(EdgeId::new(1));

        assert!(overlay.get_edge(EdgeId::new(1)).unwrap().is_delete());
    }

    #[test]
    fn test_clear() {
        let overlay = WalOverlay::new();
        overlay.insert_node(NodeId::new(1), vec![]);
        overlay.insert_edge(
            EdgeId::new(1),
            NodeId::new(1),
            NodeId::new(2),
            "E".to_string(),
        );

        assert!(!overlay.is_empty());
        overlay.clear();

        assert!(overlay.is_empty());
        assert_eq!(overlay.node_count(), 0);
        assert_eq!(overlay.edge_count(), 0);
        assert!(overlay.get_node(NodeId::new(1)).is_none());
    }

    #[test]
    fn test_drain() {
        let overlay = WalOverlay::new();
        overlay.insert_node(NodeId::new(1), vec!["A".to_string()]);
        overlay.insert_node(NodeId::new(2), vec!["B".to_string()]);
        overlay.insert_edge(
            EdgeId::new(1),
            NodeId::new(1),
            NodeId::new(2),
            "E".to_string(),
        );

        let drained_nodes = overlay.drain_nodes();
        assert_eq!(drained_nodes.len(), 2);
        assert_eq!(overlay.node_count(), 0); // drained

        let drained_edges = overlay.drain_edges();
        assert_eq!(drained_edges.len(), 1);
        assert_eq!(overlay.edge_count(), 0);
    }

    #[test]
    fn test_add_label_to_new_node() {
        let overlay = WalOverlay::new();
        overlay.insert_node(NodeId::new(1), vec!["Person".to_string()]);
        overlay.add_node_label(NodeId::new(1), "Employee".to_string());

        let node = overlay
            .get_node(NodeId::new(1))
            .unwrap()
            .data()
            .unwrap()
            .clone();
        assert!(node.labels.contains(&"Person".to_string()));
        assert!(node.labels.contains(&"Employee".to_string()));
    }

    #[test]
    fn test_add_label_to_base_node() {
        let overlay = WalOverlay::new();
        overlay.add_node_label(NodeId::new(1), "NewLabel".to_string());

        let op = overlay.get_node(NodeId::new(1)).unwrap();
        match op {
            OverlayOp::Update(node) => {
                assert_eq!(node.labels, vec!["NewLabel"]);
            }
            _ => panic!("expected Update"),
        }
    }

    #[test]
    fn test_mutation_count_tracks_all_ops() {
        let overlay = WalOverlay::new();
        overlay.insert_node(NodeId::new(1), vec![]);
        overlay.set_node_property(NodeId::new(1), "k".to_string(), Value::Int64(1));
        overlay.insert_edge(
            EdgeId::new(1),
            NodeId::new(1),
            NodeId::new(2),
            "E".to_string(),
        );
        overlay.delete_node(NodeId::new(2));
        overlay.delete_edge(EdgeId::new(1));

        assert_eq!(overlay.mutation_count(), 5);
    }

    #[test]
    fn test_approximate_memory() {
        let overlay = WalOverlay::new();
        assert_eq!(overlay.approximate_memory_bytes(), 0);

        overlay.insert_node(NodeId::new(1), vec![]);
        overlay.insert_edge(
            EdgeId::new(1),
            NodeId::new(1),
            NodeId::new(2),
            "E".to_string(),
        );

        assert!(overlay.approximate_memory_bytes() > 0);
    }

    #[test]
    fn test_edge_delete_ignores_property_set() {
        let overlay = WalOverlay::new();
        overlay.insert_edge(
            EdgeId::new(1),
            NodeId::new(1),
            NodeId::new(2),
            "KNOWS".to_string(),
        );
        overlay.delete_edge(EdgeId::new(1));

        // Property set on deleted edge should be a no-op
        overlay.set_edge_property(EdgeId::new(1), "weight".to_string(), Value::Float64(0.5));

        let op = overlay.get_edge(EdgeId::new(1)).unwrap();
        assert!(op.is_delete(), "delete should persist after property set");
    }

    #[test]
    fn test_edge_property_on_base_entity() {
        let overlay = WalOverlay::new();
        // Edge 1 exists in base data (not in overlay), set a property
        overlay.set_edge_property(EdgeId::new(1), "weight".to_string(), Value::Float64(0.9));

        let op = overlay.get_edge(EdgeId::new(1)).unwrap();
        match op {
            OverlayOp::Update(edge) => {
                assert_eq!(edge.properties.get("weight"), Some(&Value::Float64(0.9)));
            }
            _ => panic!("expected Update for base edge property change"),
        }
    }

    #[test]
    fn test_delete_base_node_then_property_ignored() {
        let overlay = WalOverlay::new();
        // Delete a base node (not previously in overlay)
        overlay.delete_node(NodeId::new(99));
        overlay.set_node_property(
            NodeId::new(99),
            "name".to_string(),
            Value::String("ghost".into()),
        );

        let op = overlay.get_node(NodeId::new(99)).unwrap();
        assert!(op.is_delete(), "base node delete should persist");
    }

    #[test]
    fn test_overlay_default_trait() {
        let overlay = WalOverlay::default();
        assert!(overlay.is_empty());
        assert_eq!(overlay.mutation_count(), 0);
    }

    #[test]
    fn test_overlay_op_is_delete() {
        let insert: OverlayOp<OverlayNode> = OverlayOp::Insert(OverlayNode {
            labels: vec![],
            properties: Default::default(),
        });
        assert!(!insert.is_delete());

        let delete: OverlayOp<OverlayNode> = OverlayOp::Delete;
        assert!(delete.is_delete());
    }

    #[test]
    fn test_overlay_op_data() {
        let node = OverlayNode {
            labels: vec!["Person".to_string()],
            properties: Default::default(),
        };
        let insert = OverlayOp::Insert(node);
        assert!(insert.data().is_some());
        assert_eq!(insert.data().unwrap().labels, vec!["Person"]);

        let delete: OverlayOp<OverlayNode> = OverlayOp::Delete;
        assert!(delete.data().is_none());
    }

    #[test]
    fn test_multiple_property_updates_on_same_node() {
        let overlay = WalOverlay::new();
        overlay.insert_node(NodeId::new(1), vec![]);
        overlay.set_node_property(
            NodeId::new(1),
            "name".to_string(),
            Value::String("Alix".into()),
        );
        overlay.set_node_property(
            NodeId::new(1),
            "name".to_string(),
            Value::String("Gus".into()),
        );

        let data = overlay
            .get_node(NodeId::new(1))
            .unwrap()
            .data()
            .unwrap()
            .clone();
        assert_eq!(
            data.properties.get("name"),
            Some(&Value::String("Gus".into())),
            "last property set should win"
        );
    }
}
