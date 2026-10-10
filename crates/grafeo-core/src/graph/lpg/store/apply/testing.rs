//! Test helpers: a store written through its change target, each write that
//! changed something recorded in a change set, as the engine's writer will.

use grafeo_common::change::{ChangeSet, DataOp, GraphSlot, Labels, Properties};
use grafeo_common::types::{ArcStr, EdgeId, EpochId, NodeId, PropertyKey, TransactionId, Value};

use crate::graph::apply::{Applied, ApplyError, ChangeTarget, Writer};
use crate::graph::lpg::LpgStore;

/// A transaction writer.
pub(crate) fn transaction(id: u64, snapshot: EpochId) -> Writer {
    Writer::Transaction {
        id: TransactionId::new(id),
        snapshot,
    }
}

/// Labels from names.
pub(crate) fn labels(names: &[&str]) -> Labels {
    names.iter().map(|name| ArcStr::from(*name)).collect()
}

/// Properties from `(key, value)` pairs.
pub(crate) fn properties(pairs: &[(&str, Value)]) -> Properties {
    pairs
        .iter()
        .map(|(key, value)| (PropertyKey::new(*key), value.clone()))
        .collect()
}

/// Writes one graph's store as `writer`, recording into a change set.
pub(crate) struct Recorder<'s> {
    /// The store written.
    pub(crate) store: &'s LpgStore,
    /// The graph's slot in the change sets written.
    pub(crate) slot: GraphSlot,
    /// Who writes.
    pub(crate) writer: Writer,
}

impl Recorder<'_> {
    /// Applies `op` and records it in `set` when it changed something.
    ///
    /// # Errors
    ///
    /// Returns what `apply` refused.
    ///
    /// # Panics
    ///
    /// Panics when the change set refuses the entry.
    pub(crate) fn write(&self, set: &mut ChangeSet, op: DataOp) -> Result<Applied, ApplyError> {
        let applied = self.store.apply(&op, self.writer)?;
        if let Applied::Changed { before, version } = &applied {
            set.push(self.slot, op, before.clone(), *version)
                .expect("the change set takes what apply reports");
        }
        Ok(applied)
    }

    /// Writes `op`, which must change something.
    fn changes(&self, set: &mut ChangeSet, op: DataOp) {
        let applied = self.write(set, op.clone()).unwrap();
        assert!(
            matches!(applied, Applied::Changed { .. } | Applied::Committed),
            "{op:?} changed nothing"
        );
    }

    /// Creates a node at a reserved id.
    pub(crate) fn create_node(
        &self,
        set: &mut ChangeSet,
        names: &[&str],
        values: &[(&str, Value)],
    ) -> NodeId {
        let id = NodeId::new(self.store.reserve_node_ids(1).unwrap().start);
        self.changes(
            set,
            DataOp::CreateNode {
                id,
                labels: labels(names),
                properties: properties(values),
            },
        );
        id
    }

    /// Creates an edge at a reserved id.
    pub(crate) fn create_edge(
        &self,
        set: &mut ChangeSet,
        src: NodeId,
        dst: NodeId,
        edge_type: &str,
        values: &[(&str, Value)],
    ) -> EdgeId {
        let id = EdgeId::new(self.store.reserve_edge_ids(1).unwrap().start);
        self.changes(
            set,
            DataOp::CreateEdge {
                id,
                src,
                dst,
                edge_type: ArcStr::from(edge_type),
                properties: properties(values),
            },
        );
        id
    }

    /// Deletes a node (its edges first, as a detach delete records them).
    pub(crate) fn delete_node(&self, set: &mut ChangeSet, id: NodeId) {
        self.changes(set, DataOp::DeleteNode { id });
    }

    /// Deletes an edge.
    pub(crate) fn delete_edge(&self, set: &mut ChangeSet, id: EdgeId) {
        self.changes(set, DataOp::DeleteEdge { id });
    }

    /// Sets a node's value.
    pub(crate) fn set_node(&self, set: &mut ChangeSet, id: NodeId, key: &str, value: Value) {
        self.changes(
            set,
            DataOp::SetNodeProperty {
                id,
                key: PropertyKey::new(key),
                value,
            },
        );
    }

    /// Removes a node's value, which it has.
    pub(crate) fn remove_node(&self, set: &mut ChangeSet, id: NodeId, key: &str) {
        self.changes(
            set,
            DataOp::RemoveNodeProperty {
                id,
                key: PropertyKey::new(key),
            },
        );
    }

    /// Sets an edge's value.
    pub(crate) fn set_edge(&self, set: &mut ChangeSet, id: EdgeId, key: &str, value: Value) {
        self.changes(
            set,
            DataOp::SetEdgeProperty {
                id,
                key: PropertyKey::new(key),
                value,
            },
        );
    }

    /// Removes an edge's value, which it has.
    pub(crate) fn remove_edge(&self, set: &mut ChangeSet, id: EdgeId, key: &str) {
        self.changes(
            set,
            DataOp::RemoveEdgeProperty {
                id,
                key: PropertyKey::new(key),
            },
        );
    }

    /// Adds a label the node lacks.
    pub(crate) fn add_label(&self, set: &mut ChangeSet, id: NodeId, label: &str) {
        self.changes(
            set,
            DataOp::AddNodeLabel {
                id,
                label: ArcStr::from(label),
            },
        );
    }

    /// Removes a label the node has.
    pub(crate) fn remove_label(&self, set: &mut ChangeSet, id: NodeId, label: &str) {
        self.changes(
            set,
            DataOp::RemoveNodeLabel {
                id,
                label: ArcStr::from(label),
            },
        );
    }
}
