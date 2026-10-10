//! The committed state of what the transactions open in a store changed
//! (#412).
//!
//! A transaction changes the store as it goes and records each change, with
//! what it replaced: in the store's undo log ([`PropertyUndoEntry`]), or in
//! its change set ([`Change`], once the engine records those). A delete
//! hides the node or edge at once and takes its labels (and, without
//! `temporal`, its values) out of the store, and without `temporal` values
//! and labels change in place. The first entry an open transaction recorded
//! for a value, a label or a deleted node or edge holds what was committed,
//! so [`LpgStore::open_changes`] (from the undo logs) and
//! [`OpenChangesByGraph::index`] (from the change sets) index those first
//! entries by entity, and a checkpoint writes the committed state from them
//! and the store.

use std::borrow::Cow;
#[cfg(not(feature = "temporal"))]
use std::collections::BTreeMap;
use std::ops::Range;

use arcstr::ArcStr;
use grafeo_common::change::{Before, Change, DataOp, Table};
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
/// Built by [`LpgStore::open_changes`] or [`OpenChangesByGraph::index`]. It
/// costs O(open changes) memory: an id and a value per value replaced (a
/// [`Value`] clone shares its data), the labels and endpoints of each
/// deleted node and edge, and one copy of each label and edge type name.
#[derive(Debug, Default, Clone)]
#[cfg_attr(test, derive(PartialEq))]
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
#[derive(Debug, Default, Clone)]
#[cfg_attr(test, derive(PartialEq))]
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

    /// Whether a transaction open in this store has changed it: its undo
    /// log holds an entry. The store then differs from the committed state
    /// in what the transaction changed at once: the nodes and edges it
    /// deleted, the values and labels it changed in place (without
    /// `temporal`), and the vector and text indexes, which take its values
    /// as it writes them. A named graph has its own store and log.
    ///
    /// The answer holds while the caller holds the transactional writes and
    /// rollbacks of this store and commits (a checkpoint's or a copy's write
    /// freeze), as for the committed state a checkpoint reads from the undo
    /// log.
    #[must_use]
    pub fn has_open_changes(&self) -> bool {
        self.property_undo_log
            .read()
            .values()
            .any(|entries| !entries.is_empty())
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
        self.changes.sorted()
    }
}

impl OpenChanges {
    /// The view with each deleted node's labels sorted: two views of the
    /// same changes then compare equal whatever order the labels came in.
    #[cfg(test)]
    pub(crate) fn normalized(&self) -> Self {
        let mut view = self.clone();
        for changes in view.labels.values_mut() {
            if let Some(labels) = &mut changes.deleted {
                labels.sort();
            }
        }
        view
    }

    /// Sorts what was indexed in recorded order by id, keeping each value's
    /// first entry.
    fn sorted(mut self) -> Self {
        self.deleted_nodes.sort_unstable();
        self.deleted_nodes.dedup();
        self.deleted_edges.sort_by_key(|edge| edge.id);
        self.deleted_edges.dedup_by_key(|edge| edge.id);
        #[cfg(not(feature = "temporal"))]
        {
            for values in self.node_values.values_mut() {
                keep_first_per_id(values);
            }
            for values in self.edge_values.values_mut() {
                keep_first_per_id(values);
            }
        }
        self
    }
}

/// The committed state of what the transactions open in a store and its
/// named graphs changed, indexed from their change sets: what a checkpoint
/// writes for them while they stay open (see
/// [`LpgStoreSection::with_open_changes`](crate::graph::lpg::LpgStoreSection::with_open_changes)
/// and [`LpgStore::committed_copy_with`]). Per graph: the nodes and edges
/// they deleted, as committed, the labels of the nodes whose labels they
/// changed and, without `temporal` (where values change in place), the
/// values they replaced; what they created is left out.
///
/// The caller indexes the change sets of every open transaction while it
/// holds their writes, rollbacks and commits (a checkpoint's write freeze
/// and commit hold), and keeps holding them until the store is written: a
/// change made in between is in the store but not here. At most one open
/// transaction changed any one node or edge (the first writer wins), so the
/// first entry per entity, key or label holds the committed state.
#[derive(Debug, Default)]
pub struct OpenChangesByGraph {
    /// The default graph's.
    default: OpenChanges,
    /// Each named graph's, by name.
    named: FxHashMap<String, OpenChanges>,
}

impl OpenChangesByGraph {
    /// Indexes the entries of the open transactions' change sets. Each item
    /// is a graph and one transaction's entries in it, in recorded order
    /// (`ChangeSet::in_graph`): the graph's storage key, `None` for the
    /// default graph. RDF entries are left out (they apply at commit).
    pub fn index<'c, E>(sets: impl IntoIterator<Item = (Option<&'c str>, E)>) -> Self
    where
        E: IntoIterator<Item = &'c Change>,
    {
        let mut default = EntryIndex::default();
        let mut named: FxHashMap<&str, EntryIndex> = FxHashMap::default();
        for (graph, entries) in sets {
            let index = match graph {
                None => &mut default,
                Some(name) => named.entry(name).or_default(),
            };
            for change in entries {
                index.add(change);
            }
        }
        Self {
            default: default.changes.sorted(),
            named: named
                .into_iter()
                .map(|(name, index)| (name.to_string(), index.changes.sorted()))
                .collect(),
        }
    }

    /// What open transactions changed in the graph named `graph` (`None`
    /// for the default graph); `None` when they changed nothing there.
    pub(crate) fn of(&self, graph: Option<&str>) -> Option<&OpenChanges> {
        match graph {
            None => Some(&self.default),
            Some(name) => self.named.get(name),
        }
    }
}

/// Where a write of a store's committed state reads what the transactions
/// still open changed.
#[derive(Debug, Clone, Copy)]
pub(crate) enum OpenChangeSource<'v> {
    /// Each store's undo log ([`LpgStore::open_changes`]): the engine's
    /// writes today.
    UndoLogs,
    /// The open transactions' change sets, indexed.
    ChangeSets(&'v OpenChangesByGraph),
}

impl<'v> OpenChangeSource<'v> {
    /// What open transactions changed in `store`, the graph named `graph`
    /// (`None` for the default graph).
    pub(crate) fn changes_of(self, store: &LpgStore, graph: Option<&str>) -> Cow<'v, OpenChanges> {
        match self {
            Self::UndoLogs => Cow::Owned(store.open_changes()),
            Self::ChangeSets(view) => match view.of(graph) {
                Some(changes) => Cow::Borrowed(changes),
                None => Cow::Owned(OpenChanges::default()),
            },
        }
    }
}

/// Builds an [`OpenChanges`] from change-set entries: each transaction's in
/// recorded order, so a create comes before every other entry of what it
/// created, and the first entry of a value or label holds the committed
/// state. Entry by entry what [`Index`] does with undo log entries.
#[derive(Default)]
struct EntryIndex {
    changes: OpenChanges,
    /// The nodes and edges an open transaction created: none of their
    /// entries is committed state.
    created_nodes: FxHashSet<NodeId>,
    created_edges: FxHashSet<EdgeId>,
    /// The ids bulk writes reserved, by table: what they created is left
    /// out too.
    created_ranges: Vec<(Table, Range<u64>)>,
}

impl EntryIndex {
    /// Whether an open transaction created node `id`.
    fn created_node(&self, id: NodeId) -> bool {
        self.created_nodes.contains(&id) || self.in_created_range(Table::Nodes, id.as_u64())
    }

    /// Whether an open transaction created edge `id`.
    fn created_edge(&self, id: EdgeId) -> bool {
        self.created_edges.contains(&id) || self.in_created_range(Table::Edges, id.as_u64())
    }

    fn in_created_range(&self, table: Table, id: u64) -> bool {
        self.created_ranges
            .iter()
            .any(|(of, ids)| *of == table && ids.contains(&id))
    }

    /// Adds an entry. A before-image of another shape than its op's does not
    /// occur: the change set refuses one.
    fn add(&mut self, change: &Change) {
        let (op, before) = match change {
            Change::Data { op, before, .. } => (op, before),
            Change::Bulk(range) => {
                self.created_ranges.push((range.table, range.ids.clone()));
                return;
            }
        };
        match op {
            DataOp::CreateNode { id, .. } => {
                self.created_nodes.insert(*id);
            }
            DataOp::CreateEdge { id, .. } => {
                self.created_edges.insert(*id);
            }
            DataOp::SetNodeProperty { id, key, .. } | DataOp::RemoveNodeProperty { id, key } => {
                if self.created_node(*id) {
                    return;
                }
                #[cfg(not(feature = "temporal"))]
                if let Before::Value(old) = before {
                    push_value(&mut self.changes.node_values, key, *id, old.clone());
                }
                #[cfg(feature = "temporal")]
                let _ = key;
            }
            DataOp::SetEdgeProperty { id, key, .. } | DataOp::RemoveEdgeProperty { id, key } => {
                if self.created_edge(*id) {
                    return;
                }
                #[cfg(not(feature = "temporal"))]
                if let Before::Value(old) = before {
                    push_value(&mut self.changes.edge_values, key, *id, old.clone());
                }
                #[cfg(feature = "temporal")]
                let _ = key;
            }
            DataOp::AddNodeLabel { id, label } => self.label(*id, label, false),
            DataOp::RemoveNodeLabel { id, label } => self.label(*id, label, true),
            DataOp::DeleteNode { id } => {
                let Before::Node(image) = before else {
                    return;
                };
                if self.created_node(*id) {
                    return;
                }
                self.changes.deleted_nodes.push(*id);
                self.changes
                    .labels
                    .entry(*id)
                    .or_default()
                    .deleted
                    .get_or_insert_with(|| image.labels.clone());
                #[cfg(not(feature = "temporal"))]
                for (key, value) in &image.properties {
                    push_value(&mut self.changes.node_values, key, *id, Some(value.clone()));
                }
            }
            DataOp::DeleteEdge { id } => {
                let Before::Edge(image) = before else {
                    return;
                };
                if self.created_edge(*id) {
                    return;
                }
                self.changes.deleted_edges.push(DeletedEdge {
                    id: *id,
                    src: image.src,
                    dst: image.dst,
                    edge_type: image.edge_type.clone(),
                });
                #[cfg(not(feature = "temporal"))]
                for (key, value) in &image.properties {
                    push_value(&mut self.changes.edge_values, key, *id, Some(value.clone()));
                }
            }
            DataOp::InsertTriple { .. } | DataOp::DeleteTriple { .. } => {}
        }
    }

    /// Notes the change of `label` on `node_id` (`removed`, or added) if it
    /// is the label's first.
    fn label(&mut self, node_id: NodeId, label: &ArcStr, removed: bool) {
        if self.created_node(node_id) {
            return;
        }
        let changes = self.changes.labels.entry(node_id).or_default();
        if !changes.first.iter().any(|(known, _)| known == label) {
            changes.first.push((label.clone(), removed));
        }
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
