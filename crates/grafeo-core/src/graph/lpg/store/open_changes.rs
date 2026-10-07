//! The committed state of what the transactions open in a store changed
//! (#412).
//!
//! A transaction changes the store as it goes and records each change, with
//! what it replaced, in the store's undo log ([`PropertyUndoEntry`]): a
//! delete hides the node or edge at once and takes its labels (and, without
//! `temporal`, its values) out of the store, and without `temporal` values
//! and labels change in place. The first entry an open transaction recorded
//! for a value, a label or a deleted node or edge holds what was committed,
//! so [`LpgStore::open_changes`] indexes those first entries by entity, and a
//! checkpoint writes the committed state from them and the store.

#[cfg(not(feature = "temporal"))]
use std::collections::BTreeMap;

use arcstr::ArcStr;
use grafeo_common::types::{EdgeId, NodeId};
#[cfg(not(feature = "temporal"))]
use grafeo_common::types::{PropertyKey, Value};
use grafeo_common::utils::hash::{FxHashMap, FxHashSet};
use smallvec::SmallVec;

use super::{LpgStore, PropertyUndoEntry};

/// The labels of a node, as [`Node::labels`](crate::graph::lpg::Node) holds
/// them.
pub(crate) type Labels = SmallVec<[ArcStr; 2]>;

/// An edge an open transaction deleted, as it was committed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct DeletedEdge {
    /// The edge.
    pub(crate) id: EdgeId,
    /// Its source node.
    pub(crate) src: NodeId,
    /// Its target node.
    pub(crate) dst: NodeId,
    /// Its type.
    pub(crate) edge_type: ArcStr,
}

/// The committed state of what the transactions open in one store changed,
/// from their undo logs: the nodes and edges they deleted, the labels of the
/// nodes whose labels they changed and, without `temporal` (where values
/// change in place), the values they replaced. What an open transaction
/// created is left out: it was never committed.
///
/// Built by [`LpgStore::open_changes`]. It costs O(open changes) memory: an
/// id and a value per value replaced (a [`Value`] clone shares its data),
/// the labels and endpoints of each deleted node and edge, and one copy of
/// each label and edge type name.
#[derive(Debug, Default)]
pub(crate) struct OpenChanges {
    /// The nodes open transactions deleted, ascending.
    deleted_nodes: Vec<NodeId>,
    /// The edges open transactions deleted, by id ascending.
    deleted_edges: Vec<DeletedEdge>,
    /// How open transactions changed the labels of a node, for the nodes
    /// whose labels they changed and those they deleted.
    labels: FxHashMap<NodeId, LabelChanges>,
    /// Per property key, the committed value of each node whose value of it
    /// an open transaction replaced (`None` when it had none), by id
    /// ascending.
    #[cfg(not(feature = "temporal"))]
    node_values: BTreeMap<PropertyKey, Vec<(NodeId, Option<Value>)>>,
    /// The same for edges.
    #[cfg(not(feature = "temporal"))]
    edge_values: BTreeMap<PropertyKey, Vec<(EdgeId, Option<Value>)>>,
}

/// How open transactions changed the labels of one node.
#[derive(Debug, Default)]
struct LabelChanges {
    /// The labels the node had when an open transaction deleted it; `None`
    /// for a node not deleted.
    deleted: Option<Labels>,
    /// The first change of each label: `true` when it was removed (the
    /// committed node has it), `false` when it was added (it does not).
    first: Vec<(ArcStr, bool)>,
}

impl OpenChanges {
    /// The nodes open transactions deleted, ascending: they were committed,
    /// and are not visible now.
    #[must_use]
    pub(crate) fn deleted_nodes(&self) -> &[NodeId] {
        &self.deleted_nodes
    }

    /// The edges open transactions deleted, by id ascending, with the
    /// endpoints and type they were committed with.
    #[must_use]
    pub(crate) fn deleted_edges(&self) -> &[DeletedEdge] {
        &self.deleted_edges
    }

    /// The edge `id` as it was committed, when an open transaction deleted
    /// it.
    #[must_use]
    pub(crate) fn deleted_edge(&self, id: EdgeId) -> Option<&DeletedEdge> {
        self.deleted_edges
            .binary_search_by_key(&id, |edge| edge.id)
            .ok()
            .map(|at| &self.deleted_edges[at])
    }

    /// The committed labels of node `id`, from `current`, the labels the
    /// store gives it now: for a node an open transaction deleted, the labels
    /// it had then; then each label an open transaction added first is taken
    /// out, and each it removed first is put back. Applied to labels that are
    /// committed already (those of the store's epoch, with `temporal`), this
    /// changes nothing.
    #[must_use]
    pub(crate) fn committed_labels(&self, id: NodeId, current: impl FnOnce() -> Labels) -> Labels {
        let Some(changes) = self.labels.get(&id) else {
            return current();
        };
        let mut labels = match &changes.deleted {
            Some(labels) => labels.clone(),
            None => current(),
        };
        for (label, removed) in &changes.first {
            let has = labels.contains(label);
            if *removed && !has {
                labels.push(label.clone());
            } else if !*removed && has {
                labels.retain(|known| known != label);
            }
        }
        labels
    }

    /// Per property key, the committed value of each node whose value of it
    /// an open transaction replaced (`None` when it had none), by id
    /// ascending. The store holds the value the transaction wrote; the
    /// first value it replaced is the committed one, and for a node it
    /// deleted without changing the key first, the value it deleted.
    #[cfg(not(feature = "temporal"))]
    #[must_use]
    pub(crate) fn node_values(&self) -> &BTreeMap<PropertyKey, Vec<(NodeId, Option<Value>)>> {
        &self.node_values
    }

    /// [`node_values`](Self::node_values()) for edges.
    #[cfg(not(feature = "temporal"))]
    #[must_use]
    pub(crate) fn edge_values(&self) -> &BTreeMap<PropertyKey, Vec<(EdgeId, Option<Value>)>> {
        &self.edge_values
    }
}

impl LpgStore {
    /// The committed state of what the transactions open in this store
    /// changed (see [`OpenChanges`]), so a checkpoint writes what was
    /// committed while transactions are open. A named graph has its own
    /// store and log.
    ///
    /// The caller holds the transactional writes and rollbacks of this store
    /// (a checkpoint's write freeze) and commits from before this call until
    /// it has read the store: a change made in between is in the store but
    /// not here. Every entry of the log then belongs to a transaction that is
    /// still open, and at most one of them changed any one node or edge (the
    /// first writer wins). The log's read lock is held only while it is
    /// indexed.
    #[must_use]
    pub(crate) fn open_changes(&self) -> OpenChanges {
        let log = self.property_undo_log.read();
        let mut index = Index::default();
        for entries in log.values() {
            for entry in entries {
                index.add(entry);
            }
        }
        index.finish()
    }
}

/// Builds an [`OpenChanges`] from undo log entries, in log order.
#[derive(Default)]
struct Index<'l> {
    changes: OpenChanges,
    /// The nodes and edges an open transaction created: none of their
    /// entries is committed state.
    created_nodes: FxHashSet<NodeId>,
    created_edges: FxHashSet<EdgeId>,
    /// One shared copy of each label and edge type name met.
    names: FxHashMap<&'l str, ArcStr>,
}

impl<'l> Index<'l> {
    /// Adds an entry; within a transaction, entries come in the order they
    /// were recorded, so a creation comes before every other entry of what
    /// it created, and the first entry of a value or label is the committed
    /// one.
    fn add(&mut self, entry: &'l PropertyUndoEntry) {
        match entry {
            // A tombstone of a compacted base entity is no row of this
            // store's tables: the deletions section writes only committed
            // tombstones, so a pending one is left out already.
            PropertyUndoEntry::BaseNodeDeleted { .. }
            | PropertyUndoEntry::BaseEdgeDeleted { .. } => {}
            PropertyUndoEntry::NodeCreated { node_id } => {
                self.created_nodes.insert(*node_id);
            }
            PropertyUndoEntry::EdgeCreated { edge_id } => {
                self.created_edges.insert(*edge_id);
            }
            PropertyUndoEntry::NodeProperty {
                node_id,
                key,
                old_value,
            } => {
                if self.created_nodes.contains(node_id) {
                    return;
                }
                #[cfg(not(feature = "temporal"))]
                push_value(
                    &mut self.changes.node_values,
                    key,
                    *node_id,
                    old_value.clone(),
                );
                #[cfg(feature = "temporal")]
                let _ = (key, old_value);
            }
            PropertyUndoEntry::EdgeProperty {
                edge_id,
                key,
                old_value,
            } => {
                if self.created_edges.contains(edge_id) {
                    return;
                }
                #[cfg(not(feature = "temporal"))]
                push_value(
                    &mut self.changes.edge_values,
                    key,
                    *edge_id,
                    old_value.clone(),
                );
                #[cfg(feature = "temporal")]
                let _ = (key, old_value);
            }
            PropertyUndoEntry::LabelAdded { node_id, label } => {
                self.label(*node_id, label, false);
            }
            PropertyUndoEntry::LabelRemoved { node_id, label } => {
                self.label(*node_id, label, true);
            }
            PropertyUndoEntry::NodeDeleted {
                node_id,
                labels,
                properties,
            } => {
                if self.created_nodes.contains(node_id) {
                    return;
                }
                self.changes.deleted_nodes.push(*node_id);
                let labels: Labels = labels.iter().map(|label| self.name(label)).collect();
                self.changes
                    .labels
                    .entry(*node_id)
                    .or_default()
                    .deleted
                    .get_or_insert(labels);
                #[cfg(not(feature = "temporal"))]
                for (key, value) in properties {
                    push_value(
                        &mut self.changes.node_values,
                        key,
                        *node_id,
                        Some(value.clone()),
                    );
                }
                #[cfg(feature = "temporal")]
                let _ = properties;
            }
            PropertyUndoEntry::EdgeDeleted {
                edge_id,
                src,
                dst,
                edge_type,
                properties,
            } => {
                if self.created_edges.contains(edge_id) {
                    return;
                }
                let edge_type = self.name(edge_type);
                self.changes.deleted_edges.push(DeletedEdge {
                    id: *edge_id,
                    src: *src,
                    dst: *dst,
                    edge_type,
                });
                #[cfg(not(feature = "temporal"))]
                for (key, value) in properties {
                    push_value(
                        &mut self.changes.edge_values,
                        key,
                        *edge_id,
                        Some(value.clone()),
                    );
                }
                #[cfg(feature = "temporal")]
                let _ = properties;
            }
        }
    }

    /// Notes the change of `label` on `node_id` (`removed`, or added) if it
    /// is the label's first.
    fn label(&mut self, node_id: NodeId, label: &'l str, removed: bool) {
        if self.created_nodes.contains(&node_id) {
            return;
        }
        let label = self.name(label);
        let changes = self.changes.labels.entry(node_id).or_default();
        if !changes.first.iter().any(|(known, _)| *known == label) {
            changes.first.push((label, removed));
        }
    }

    /// The shared copy of `name`.
    fn name(&mut self, name: &'l str) -> ArcStr {
        self.names
            .entry(name)
            .or_insert_with(|| ArcStr::from(name))
            .clone()
    }

    /// Sorts what was indexed by id, keeping each value's first entry.
    fn finish(self) -> OpenChanges {
        let mut changes = self.changes;
        changes.deleted_nodes.sort_unstable();
        changes.deleted_nodes.dedup();
        changes.deleted_edges.sort_by_key(|edge| edge.id);
        changes.deleted_edges.dedup_by_key(|edge| edge.id);
        #[cfg(not(feature = "temporal"))]
        {
            for values in changes.node_values.values_mut() {
                keep_first_per_id(values);
            }
            for values in changes.edge_values.values_mut() {
                keep_first_per_id(values);
            }
        }
        changes
    }
}

/// Appends the value `id` had for `key` to the values of `key`.
#[cfg(not(feature = "temporal"))]
fn push_value<Id>(
    values: &mut BTreeMap<PropertyKey, Vec<(Id, Option<Value>)>>,
    key: &PropertyKey,
    id: Id,
    value: Option<Value>,
) {
    match values.get_mut(key) {
        Some(column) => column.push((id, value)),
        None => {
            values.insert(key.clone(), vec![(id, value)]);
        }
    }
}

/// Sorts `values` by id, keeping the first entry of each id: the entries
/// came in log order, and the sort is stable.
#[cfg(not(feature = "temporal"))]
fn keep_first_per_id<Id: Copy + Ord>(values: &mut Vec<(Id, Option<Value>)>) {
    values.sort_by_key(|(id, _)| *id);
    values.dedup_by_key(|(id, _)| *id);
}
