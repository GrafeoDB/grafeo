//! Lightweight mutation overlay for CompactStore.
//!
//! Runtime mutations (creates, deletes, property changes) accumulate here.
//! Reads merge DeltaBuffer state with the immutable snapshot. No MVCC —
//! locking is handled by the parent CompactStore via a mutex wrapper.

use arcstr::ArcStr;
use grafeo_common::types::{EdgeId, NodeId, PropertyKey, Value};
use grafeo_common::utils::hash::{FxHashMap, FxHashSet};
use smallvec::SmallVec;

use super::id::{encode_edge_id, encode_node_id};
use crate::graph::Direction;

/// Reserved table ID for delta-created nodes.
pub(super) const DELTA_NODE_TABLE_ID: u16 = 32000;

/// Reserved table ID for delta-created edges.
pub(super) const DELTA_EDGE_TABLE_ID: u16 = 32001;

/// A node created at runtime by the delta layer.
#[derive(Debug, Clone)]
pub(crate) struct DeltaNode {
    /// The synthetic [`NodeId`] assigned to this node.
    pub id: NodeId,
    /// Labels attached to this node.
    pub labels: SmallVec<[ArcStr; 2]>,
    /// Properties set on this node.
    pub properties: FxHashMap<PropertyKey, Value>,
}

/// An edge created at runtime by the delta layer.
#[derive(Debug, Clone)]
pub(crate) struct DeltaEdge {
    /// The synthetic [`EdgeId`] assigned to this edge.
    pub id: EdgeId,
    /// Source node.
    pub src: NodeId,
    /// Destination node.
    pub dst: NodeId,
    /// Relationship type.
    pub edge_type: ArcStr,
    /// Properties set on this edge.
    pub properties: FxHashMap<PropertyKey, Value>,
}

/// Accumulates runtime mutations against an immutable CompactStore snapshot.
///
/// All writes (node/edge creation, deletion, property updates, label changes)
/// are buffered here. At read time the caller merges delta state with the
/// underlying snapshot columns to produce a consistent view.
///
/// The buffer is **not** thread-safe — it targets single-threaded WASM
/// environments where no locking is needed.
#[derive(Debug, Clone, Default)]
pub struct DeltaBuffer {
    /// Nodes created in this delta session.
    created_nodes: Vec<DeltaNode>,
    /// Fast index: delta NodeId -> position in `created_nodes`.
    node_index: FxHashMap<NodeId, usize>,
    /// Edges created in this delta session.
    created_edges: Vec<DeltaEdge>,
    /// Fast index: delta EdgeId -> position in `created_edges`.
    edge_index: FxHashMap<EdgeId, usize>,

    /// Snapshot node IDs that have been logically deleted.
    deleted_nodes: FxHashSet<NodeId>,
    /// Snapshot edge IDs that have been logically deleted.
    deleted_edges: FxHashSet<EdgeId>,

    /// Per-node property overrides. `None` value = tombstone (property removed).
    node_property_overrides: FxHashMap<NodeId, FxHashMap<PropertyKey, Option<Value>>>,
    /// Per-edge property overrides. `None` value = tombstone (property removed).
    edge_property_overrides: FxHashMap<EdgeId, FxHashMap<PropertyKey, Option<Value>>>,

    /// Labels added to existing snapshot nodes.
    added_labels: FxHashMap<NodeId, FxHashSet<ArcStr>>,
    /// Labels removed from existing snapshot nodes.
    removed_labels: FxHashMap<NodeId, FxHashSet<ArcStr>>,

    /// Labels that have at least one delta-created node, enabling quick
    /// "does this label need a delta scan?" checks.
    label_overflow: FxHashSet<ArcStr>,

    /// Monotonic counter for the next delta node offset.
    next_delta_node: u64,
    /// Monotonic counter for the next delta edge offset.
    next_delta_edge: u64,
}

impl DeltaBuffer {
    /// Creates an empty delta buffer.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    // ------------------------------------------------------------------
    // Node creation / deletion
    // ------------------------------------------------------------------

    /// Creates a new delta node with the given labels and returns its [`NodeId`].
    ///
    /// The ID is synthesised from [`DELTA_NODE_TABLE_ID`] and a monotonically
    /// increasing offset so it will never collide with snapshot IDs.
    pub fn create_node(&mut self, labels: &[&str]) -> NodeId {
        let offset = self.next_delta_node;
        self.next_delta_node += 1;

        let id = encode_node_id(DELTA_NODE_TABLE_ID, offset);
        let label_vec: SmallVec<[ArcStr; 2]> = labels.iter().map(|s| ArcStr::from(*s)).collect();

        for l in &label_vec {
            self.label_overflow.insert(l.clone());
        }

        let idx = self.created_nodes.len();
        self.created_nodes.push(DeltaNode {
            id,
            labels: label_vec,
            properties: FxHashMap::default(),
        });
        self.node_index.insert(id, idx);

        id
    }

    /// Marks a node as deleted. Returns `true` if the node was not already
    /// deleted.
    ///
    /// If the node is a delta-created node, it is removed from `created_nodes`
    /// entirely so that `created_node_count()` only reflects live delta nodes.
    /// If it is a snapshot node, a tombstone is added to `deleted_nodes`.
    pub fn delete_node(&mut self, id: NodeId) -> bool {
        // If this is a delta-created node, remove it entirely.
        if let Some(idx) = self.node_index.remove(&id) {
            let removed = self.created_nodes.swap_remove(idx);
            // Update the index for the node that was swapped into position `idx`.
            if idx < self.created_nodes.len() {
                let swapped_id = self.created_nodes[idx].id;
                self.node_index.insert(swapped_id, idx);
            }
            // Clean up label_overflow: remove labels that no longer have any delta nodes.
            for label in &removed.labels {
                let still_has = self.created_nodes.iter().any(|n| n.labels.contains(label));
                if !still_has {
                    self.label_overflow.remove(label);
                }
            }
            return true;
        }
        // Otherwise it's a snapshot node — add tombstone.
        self.deleted_nodes.insert(id)
    }

    /// Returns `true` if the given node has been logically deleted.
    #[must_use]
    pub fn is_node_deleted(&self, id: NodeId) -> bool {
        self.deleted_nodes.contains(&id)
    }

    // ------------------------------------------------------------------
    // Edge creation / deletion
    // ------------------------------------------------------------------

    /// Creates a new delta edge and returns its [`EdgeId`].
    ///
    /// The ID is synthesised from [`DELTA_EDGE_TABLE_ID`] and a monotonically
    /// increasing offset.
    pub fn create_edge(&mut self, src: NodeId, dst: NodeId, edge_type: &str) -> EdgeId {
        let offset = self.next_delta_edge;
        self.next_delta_edge += 1;

        let id = encode_edge_id(DELTA_EDGE_TABLE_ID, offset);

        let idx = self.created_edges.len();
        self.created_edges.push(DeltaEdge {
            id,
            src,
            dst,
            edge_type: ArcStr::from(edge_type),
            properties: FxHashMap::default(),
        });
        self.edge_index.insert(id, idx);

        id
    }

    /// Marks an edge as deleted. Returns `true` if the edge was not already
    /// deleted.
    ///
    /// If the edge is a delta-created edge, it is removed from `created_edges`
    /// entirely so that `created_edge_count()` only reflects live delta edges.
    /// If it is a snapshot edge, a tombstone is added to `deleted_edges`.
    pub fn delete_edge(&mut self, id: EdgeId) -> bool {
        // If this is a delta-created edge, remove it entirely.
        if let Some(idx) = self.edge_index.remove(&id) {
            self.created_edges.swap_remove(idx);
            // Update the index for the edge that was swapped into position `idx`.
            if idx < self.created_edges.len() {
                let swapped_id = self.created_edges[idx].id;
                self.edge_index.insert(swapped_id, idx);
            }
            return true;
        }
        // Otherwise it's a snapshot edge — add tombstone.
        self.deleted_edges.insert(id)
    }

    /// Returns `true` if the given edge has been logically deleted.
    #[must_use]
    pub fn is_edge_deleted(&self, id: EdgeId) -> bool {
        self.deleted_edges.contains(&id)
    }

    // ------------------------------------------------------------------
    // Property overrides (nodes)
    // ------------------------------------------------------------------

    /// Sets (or overwrites) a property on a node.
    ///
    /// Works for both snapshot and delta-created nodes. Delta-created nodes
    /// store the value directly; snapshot nodes record an override that is
    /// merged at read time.
    pub fn set_node_property(&mut self, id: NodeId, key: &str, value: Value) {
        // Check if this is a delta-created node — store directly.
        if let Some(&idx) = self.node_index.get(&id) {
            self.created_nodes[idx]
                .properties
                .insert(PropertyKey::from(key), value);
            return;
        }

        // Snapshot node — record an override.
        self.node_property_overrides
            .entry(id)
            .or_default()
            .insert(PropertyKey::from(key), Some(value));
    }

    /// Removes a property from a node, returning the previous override value
    /// if one existed.
    ///
    /// For snapshot nodes this inserts a tombstone (`None`) so the reader
    /// knows to suppress the snapshot value.
    pub fn remove_node_property(&mut self, id: NodeId, key: &str) -> Option<Value> {
        let pk = PropertyKey::from(key);

        // Delta-created node — remove directly.
        if let Some(&idx) = self.node_index.get(&id) {
            return self.created_nodes[idx].properties.remove(&pk);
        }

        // Snapshot node — insert a tombstone and return any prior override.
        let overrides = self.node_property_overrides.entry(id).or_default();
        let prev = overrides.insert(pk, None);
        prev.flatten()
    }

    /// Looks up a property override for a snapshot node.
    ///
    /// Returns:
    /// - `None` — no override recorded (use snapshot value).
    /// - `Some(Some(value))` — override with a new value.
    /// - `Some(None)` — property was explicitly removed (tombstone).
    #[must_use]
    pub fn get_node_property_override(
        &self,
        id: NodeId,
        key: &PropertyKey,
    ) -> Option<Option<&Value>> {
        self.node_property_overrides
            .get(&id)?
            .get(key)
            .map(|opt| opt.as_ref())
    }

    // ------------------------------------------------------------------
    // Property overrides (edges)
    // ------------------------------------------------------------------

    /// Sets (or overwrites) a property on an edge.
    pub fn set_edge_property(&mut self, id: EdgeId, key: &str, value: Value) {
        if let Some(&idx) = self.edge_index.get(&id) {
            self.created_edges[idx]
                .properties
                .insert(PropertyKey::from(key), value);
            return;
        }

        self.edge_property_overrides
            .entry(id)
            .or_default()
            .insert(PropertyKey::from(key), Some(value));
    }

    /// Removes a property from an edge, returning the previous override value
    /// if one existed.
    pub fn remove_edge_property(&mut self, id: EdgeId, key: &str) -> Option<Value> {
        let pk = PropertyKey::from(key);

        if let Some(&idx) = self.edge_index.get(&id) {
            return self.created_edges[idx].properties.remove(&pk);
        }

        let overrides = self.edge_property_overrides.entry(id).or_default();
        let prev = overrides.insert(pk, None);
        prev.flatten()
    }

    /// Looks up a property override for a snapshot edge.
    ///
    /// Returns:
    /// - `None` — no override recorded (use snapshot value).
    /// - `Some(Some(value))` — override with a new value.
    /// - `Some(None)` — property was explicitly removed (tombstone).
    #[must_use]
    pub fn get_edge_property_override(
        &self,
        id: EdgeId,
        key: &PropertyKey,
    ) -> Option<Option<&Value>> {
        self.edge_property_overrides
            .get(&id)?
            .get(key)
            .map(|opt| opt.as_ref())
    }

    /// Returns all property overrides for a snapshot node.
    #[must_use]
    pub fn node_property_overrides_for(
        &self,
        id: NodeId,
    ) -> Option<&FxHashMap<PropertyKey, Option<Value>>> {
        self.node_property_overrides.get(&id)
    }

    /// Returns all property overrides for a snapshot edge.
    #[must_use]
    pub fn edge_property_overrides_for(
        &self,
        id: EdgeId,
    ) -> Option<&FxHashMap<PropertyKey, Option<Value>>> {
        self.edge_property_overrides.get(&id)
    }

    // ------------------------------------------------------------------
    // Label mutations
    // ------------------------------------------------------------------

    /// Adds a label to an existing snapshot node.
    pub fn add_label(&mut self, id: NodeId, label: &str) {
        // If this is a delta node, add directly.
        if let Some(&idx) = self.node_index.get(&id) {
            let dn = &mut self.created_nodes[idx];
            let arc = ArcStr::from(label);
            if !dn.labels.contains(&arc) {
                dn.labels.push(arc.clone());
                self.label_overflow.insert(arc);
            }
            return;
        }

        let arc = ArcStr::from(label);

        // Remove from removed_labels if it was previously removed.
        if let Some(set) = self.removed_labels.get_mut(&id) {
            set.remove(&arc);
        }

        self.added_labels.entry(id).or_default().insert(arc);
    }

    /// Removes a label from an existing snapshot node.
    pub fn remove_label(&mut self, id: NodeId, label: &str) {
        // Delta-created node — remove directly.
        if let Some(&idx) = self.node_index.get(&id) {
            let dn = &mut self.created_nodes[idx];
            let arc = ArcStr::from(label);
            dn.labels.retain(|l| l != &arc);
            return;
        }

        let arc = ArcStr::from(label);

        // Remove from added_labels if it was previously added.
        if let Some(set) = self.added_labels.get_mut(&id) {
            set.remove(&arc);
        }

        self.removed_labels.entry(id).or_default().insert(arc);
    }

    /// Returns labels that were added to `id` via the delta layer.
    #[must_use]
    pub fn added_labels_for(&self, id: NodeId) -> Option<&FxHashSet<ArcStr>> {
        self.added_labels.get(&id)
    }

    /// Returns labels that were removed from `id` via the delta layer.
    #[must_use]
    pub fn removed_labels_for(&self, id: NodeId) -> Option<&FxHashSet<ArcStr>> {
        self.removed_labels.get(&id)
    }

    // ------------------------------------------------------------------
    // Lookups
    // ------------------------------------------------------------------

    /// Returns the delta-created node with the given ID, if any.
    #[must_use]
    pub(crate) fn get_delta_node(&self, id: NodeId) -> Option<&DeltaNode> {
        let &idx = self.node_index.get(&id)?;
        self.created_nodes.get(idx)
    }

    /// Returns the delta-created edge with the given ID, if any.
    #[must_use]
    pub(crate) fn get_delta_edge(&self, id: EdgeId) -> Option<&DeltaEdge> {
        let &idx = self.edge_index.get(&id)?;
        self.created_edges.get(idx)
    }

    /// Returns all delta-created nodes that carry the given label.
    #[must_use]
    pub fn nodes_by_label(&self, label: &str) -> Vec<NodeId> {
        self.created_nodes
            .iter()
            .filter(|n| n.labels.iter().any(|l| l.as_str() == label))
            .map(|n| n.id)
            .collect()
    }

    /// Returns `true` if any delta-created node carries the given label.
    ///
    /// This is a fast O(1) check — the label overflow set is maintained
    /// incrementally during [`create_node`](Self::create_node).
    #[must_use]
    pub fn has_overflow(&self, label: &str) -> bool {
        self.label_overflow.contains(label)
    }

    /// Returns delta-created edges that touch `node` in the given
    /// [`Direction`].
    #[must_use]
    pub(crate) fn edges_for_node(&self, node: NodeId, direction: Direction) -> Vec<&DeltaEdge> {
        self.created_edges
            .iter()
            .filter(|e| match direction {
                Direction::Outgoing => e.src == node,
                Direction::Incoming => e.dst == node,
                Direction::Both => e.src == node || e.dst == node,
            })
            .collect()
    }

    // ------------------------------------------------------------------
    // Iteration / counts
    // ------------------------------------------------------------------

    /// Returns an iterator over all node property override entries.
    ///
    /// Each item is a `(NodeId, overrides_map)` pair for a snapshot node whose
    /// properties were modified at runtime.
    pub(crate) fn all_node_property_overrides(
        &self,
    ) -> impl Iterator<Item = (&NodeId, &FxHashMap<PropertyKey, Option<Value>>)> {
        self.node_property_overrides.iter()
    }

    /// Returns all delta-created nodes (for label enumeration, property scanning, etc.).
    pub(crate) fn all_created_nodes(&self) -> &[DeltaNode] {
        &self.created_nodes
    }

    /// Returns all delta-created edges (for edge type enumeration, etc.).
    pub(crate) fn all_created_edges(&self) -> &[DeltaEdge] {
        &self.created_edges
    }

    /// Returns an iterator over all delta-created node IDs.
    pub fn all_created_node_ids(&self) -> impl Iterator<Item = NodeId> + '_ {
        self.created_nodes.iter().map(|n| n.id)
    }

    /// Returns an iterator over all delta-created edge IDs.
    pub fn all_created_edge_ids(&self) -> impl Iterator<Item = EdgeId> + '_ {
        self.created_edges.iter().map(|e| e.id)
    }

    /// Returns the number of delta-created nodes.
    #[must_use]
    pub fn created_node_count(&self) -> usize {
        self.created_nodes.len()
    }

    /// Returns the number of delta-created edges.
    #[must_use]
    pub fn created_edge_count(&self) -> usize {
        self.created_edges.len()
    }

    /// Returns the number of deleted snapshot nodes.
    #[must_use]
    pub fn deleted_node_count(&self) -> usize {
        self.deleted_nodes.len()
    }

    /// Returns the number of deleted snapshot edges.
    #[must_use]
    pub fn deleted_edge_count(&self) -> usize {
        self.deleted_edges.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_and_delete_delta_node() {
        let mut buf = DeltaBuffer::new();

        let n1 = buf.create_node(&["Person", "Employee"]);
        assert_eq!(buf.created_node_count(), 1);

        let dn = buf.get_delta_node(n1).expect("should find created node");
        assert_eq!(dn.labels.len(), 2);
        assert_eq!(dn.labels[0].as_str(), "Person");
        assert_eq!(dn.labels[1].as_str(), "Employee");

        // Deleting a delta-created node removes it from created_nodes entirely.
        assert!(buf.delete_node(n1));
        assert_eq!(buf.created_node_count(), 0);
        assert!(buf.get_delta_node(n1).is_none());
    }

    #[test]
    fn test_snapshot_node_tombstone() {
        let mut buf = DeltaBuffer::new();

        // Pretend NodeId(42) is a snapshot node.
        let snap_id = NodeId::new(42);
        assert!(!buf.is_node_deleted(snap_id));

        buf.delete_node(snap_id);
        assert!(buf.is_node_deleted(snap_id));
    }

    #[test]
    fn test_property_override_on_snapshot_node() {
        let mut buf = DeltaBuffer::new();
        let snap_id = NodeId::new(100);
        let key = PropertyKey::from("name");

        // No override yet.
        assert!(buf.get_node_property_override(snap_id, &key).is_none());

        // Set override.
        buf.set_node_property(snap_id, "name", Value::String(ArcStr::from("Alice")));
        let ov = buf.get_node_property_override(snap_id, &key);
        assert_eq!(ov, Some(Some(&Value::String(ArcStr::from("Alice")))));

        // Remove produces a tombstone.
        let prev = buf.remove_node_property(snap_id, "name");
        assert_eq!(prev, Some(Value::String(ArcStr::from("Alice"))));

        let ov = buf.get_node_property_override(snap_id, &key);
        assert_eq!(ov, Some(None)); // tombstone
    }

    #[test]
    fn test_create_edge_and_query_by_direction() {
        let mut buf = DeltaBuffer::new();

        let n1 = buf.create_node(&["Person"]);
        let n2 = buf.create_node(&["Person"]);
        let n3 = buf.create_node(&["Company"]);

        let e1 = buf.create_edge(n1, n2, "KNOWS");
        let e2 = buf.create_edge(n1, n3, "WORKS_AT");
        let _e3 = buf.create_edge(n2, n3, "WORKS_AT");

        assert_eq!(buf.created_edge_count(), 3);

        // Outgoing from n1.
        let out = buf.edges_for_node(n1, Direction::Outgoing);
        assert_eq!(out.len(), 2);
        assert!(out.iter().any(|e| e.id == e1));
        assert!(out.iter().any(|e| e.id == e2));

        // Incoming to n3.
        let inc = buf.edges_for_node(n3, Direction::Incoming);
        assert_eq!(inc.len(), 2);

        // Both for n2 — one outgoing, one incoming.
        let both = buf.edges_for_node(n2, Direction::Both);
        assert_eq!(both.len(), 2);
    }

    #[test]
    fn test_label_overflow_tracking() {
        let mut buf = DeltaBuffer::new();

        assert!(!buf.has_overflow("Person"));

        buf.create_node(&["Person"]);
        assert!(buf.has_overflow("Person"));
        assert!(!buf.has_overflow("Company"));

        let ids = buf.nodes_by_label("Person");
        assert_eq!(ids.len(), 1);
    }

    #[test]
    fn test_edge_property_set_and_remove() {
        let mut buf = DeltaBuffer::new();
        let n1 = buf.create_node(&[]);
        let n2 = buf.create_node(&[]);
        let eid = buf.create_edge(n1, n2, "LIKES");

        buf.set_edge_property(eid, "weight", Value::Int64(5));
        let de = buf.get_delta_edge(eid).unwrap();
        assert_eq!(
            de.properties.get(&PropertyKey::from("weight")),
            Some(&Value::Int64(5))
        );

        let removed = buf.remove_edge_property(eid, "weight");
        assert_eq!(removed, Some(Value::Int64(5)));

        let de = buf.get_delta_edge(eid).unwrap();
        assert!(de.properties.get(&PropertyKey::from("weight")).is_none());
    }

    #[test]
    fn test_label_add_and_remove_on_snapshot_node() {
        let mut buf = DeltaBuffer::new();
        let snap = NodeId::new(7);

        buf.add_label(snap, "Admin");
        let added = buf.added_labels_for(snap).unwrap();
        assert!(added.contains("Admin"));

        buf.remove_label(snap, "Admin");
        // Should no longer appear in added.
        let added = buf.added_labels_for(snap);
        assert!(added.is_none() || !added.unwrap().contains("Admin"));

        let removed = buf.removed_labels_for(snap).unwrap();
        assert!(removed.contains("Admin"));
    }

    #[test]
    fn test_all_created_node_ids() {
        let mut buf = DeltaBuffer::new();
        let n1 = buf.create_node(&["A"]);
        let n2 = buf.create_node(&["B"]);

        let ids: Vec<NodeId> = buf.all_created_node_ids().collect();
        assert_eq!(ids, vec![n1, n2]);
    }

    #[test]
    fn test_delete_edge() {
        let mut buf = DeltaBuffer::new();
        let eid = EdgeId::new(999);

        assert!(!buf.is_edge_deleted(eid));
        assert!(buf.delete_edge(eid));
        assert!(buf.is_edge_deleted(eid));
        assert!(!buf.delete_edge(eid)); // already deleted
    }
}
