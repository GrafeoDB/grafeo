//! The row-group store's raw writes, through `apply`.
//!
//! A raw write commits at once: it is an immediate write at the next epoch.
//! A `*_versioned` raw write, a create, a delete, a value or a label, is a
//! pending write of its transaction (at the store's epoch as its snapshot)
//! that no change set records (raw store handles, QJ): the transaction sees
//! it, no other reader does, and nothing commits it.

use std::sync::atomic::Ordering;

use grafeo_common::change::{Before, DataOp, Labels};
use grafeo_common::types::{ArcStr, EdgeId, EpochId, NodeId, PropertyKey, TransactionId, Value};
use grafeo_common::utils::error::{Error, Result};

use super::RowGroupStore;
use crate::graph::Direction;
use crate::graph::apply::{Applied, ApplyError, GraphBackend, Writer};

/// The crate's error for an `apply` that refused.
fn error(refused: &ApplyError) -> Error {
    Error::Internal(refused.to_string())
}

/// Whether `apply` changed something.
fn changed(applied: &std::result::Result<Applied, ApplyError>) -> bool {
    matches!(applied, Ok(Applied::Changed { .. } | Applied::Committed))
}

fn labels(names: &[&str]) -> Labels {
    names.iter().map(|name| ArcStr::from(*name)).collect()
}

impl RowGroupStore {
    /// Applies `op` as an immediate write at the next epoch.
    fn write_now(
        &self,
        op: &DataOp,
        before_images: bool,
    ) -> std::result::Result<Applied, ApplyError> {
        let epoch = EpochId::new(self.epoch.load(Ordering::Acquire) + 1);
        self.apply(
            op,
            Writer::Immediate {
                epoch,
                before_images,
            },
        )
    }

    /// Applies `op` as a pending write of `transaction`, at the store's
    /// epoch as its snapshot.
    fn write_as(
        &self,
        op: &DataOp,
        transaction: TransactionId,
    ) -> std::result::Result<Applied, ApplyError> {
        self.apply(
            op,
            Writer::Transaction {
                id: transaction,
                snapshot: EpochId::new(self.epoch.load(Ordering::Acquire)),
            },
        )
    }

    /// The value a removal took out, if any.
    fn removed(applied: std::result::Result<Applied, ApplyError>) -> Result<Option<Value>> {
        match applied {
            Ok(Applied::Changed {
                before: Before::Value(old),
                ..
            }) => Ok(old),
            Ok(_) => Ok(None),
            Err(refused) => Err(error(&refused)),
        }
    }

    fn reserve_node(&self) -> NodeId {
        self.reserve_node_ids(1)
            .map_or(NodeId::INVALID, |ids| NodeId::new(ids.start))
    }

    fn reserve_edge(&self) -> EdgeId {
        self.reserve_edge_ids(1)
            .map_or(EdgeId::INVALID, |ids| EdgeId::new(ids.start))
    }
}

/// The raw writes: each commits at once (an immediate write at the next
/// epoch), or, the `*_versioned` ones, applies as a pending write of its
/// transaction that no change set records. For tests, loaders and seeding;
/// the engine writes through [`GraphBackend`].
impl RowGroupStore {
    /// Creates a new node with the given labels.
    pub fn create_node(&self, names: &[&str]) -> NodeId {
        let id = self.reserve_node();
        let op = DataOp::CreateNode {
            id,
            labels: labels(names),
            properties: Vec::new(),
        };
        let _created = self.write_now(&op, false);
        id
    }

    /// Creates a new node within a transaction context.
    pub fn create_node_versioned(
        &self,
        names: &[&str],
        epoch: EpochId,
        transaction_id: TransactionId,
    ) -> NodeId {
        let id = self.reserve_node();
        let op = DataOp::CreateNode {
            id,
            labels: labels(names),
            properties: Vec::new(),
        };
        let _created = self.apply(
            &op,
            Writer::Transaction {
                id: transaction_id,
                snapshot: epoch,
            },
        );
        id
    }

    /// Creates a new edge between two nodes.
    ///
    /// The store does not check the endpoints: the caller makes sure both
    /// exist (the engine's writes go through
    /// [`create_edge_versioned`](Self::create_edge_versioned)).
    pub fn create_edge(&self, src: NodeId, dst: NodeId, edge_type: &str) -> EdgeId {
        let id = self.reserve_edge();
        let op = DataOp::CreateEdge {
            id,
            src,
            dst,
            edge_type: ArcStr::from(edge_type),
            properties: Vec::new(),
        };
        let _created = self.write_now(&op, false);
        id
    }

    /// Creates a new edge within a transaction context
    /// (`TransactionId::SYSTEM` outside one).
    ///
    /// The caller checks that its transaction sees both endpoints, and claims
    /// them against a concurrent delete (as `GraphWriter` does).
    ///
    /// # Errors
    ///
    /// Returns a write conflict, and creates nothing, when the store knows an
    /// endpoint is deleted, by a committed delete or one in progress: a
    /// compacted store's base node with a tombstone.
    pub fn create_edge_versioned(
        &self,
        src: NodeId,
        dst: NodeId,
        edge_type: &str,
        epoch: EpochId,
        transaction_id: TransactionId,
    ) -> Result<EdgeId> {
        let id = self.reserve_edge();
        let op = DataOp::CreateEdge {
            id,
            src,
            dst,
            edge_type: ArcStr::from(edge_type),
            properties: Vec::new(),
        };
        self.apply(
            &op,
            Writer::Transaction {
                id: transaction_id,
                snapshot: epoch,
            },
        )
        .map(|_| id)
        .map_err(|refused| error(&refused))
    }

    /// Creates multiple edges in batch (single lock acquisition). The
    /// endpoints are not checked, as for [`create_edge`](Self::create_edge).
    pub fn batch_create_edges(&self, edges: &[(NodeId, NodeId, &str)]) -> Vec<EdgeId> {
        edges
            .iter()
            .map(|(src, dst, edge_type)| self.create_edge(*src, *dst, edge_type))
            .collect()
    }

    /// Deletes a node. Returns `true` if the node existed.
    pub fn delete_node(&self, id: NodeId) -> bool {
        changed(&self.write_now(&DataOp::DeleteNode { id }, false))
    }

    /// Deletes a node within a transaction context. Returns `true` if the node existed.
    ///
    /// # Errors
    ///
    /// Returns an error, and deletes nothing, when the store cannot read the
    /// node it would delete (a record or value it cannot read).
    pub fn delete_node_versioned(
        &self,
        id: NodeId,
        epoch: EpochId,
        transaction_id: TransactionId,
    ) -> Result<bool> {
        let applied = self.apply(
            &DataOp::DeleteNode { id },
            Writer::Transaction {
                id: transaction_id,
                snapshot: epoch,
            },
        );
        match applied {
            Ok(applied) => Ok(!matches!(applied, Applied::Unchanged)),
            Err(refused) => Err(error(&refused)),
        }
    }

    /// Deletes all edges connected to a node (DETACH DELETE).
    pub fn delete_node_edges(&self, node_id: NodeId) {
        let now = self.now();
        let edges: Vec<EdgeId> = {
            let inner = self.inner.read();
            inner
                .adjacency(node_id, Direction::Both)
                .into_iter()
                .filter(|(_, edge)| inner.edge_visible(edge.as_u64(), now))
                .map(|(_, edge)| edge)
                .collect()
        };
        for id in edges {
            let _deleted = self.write_now(&DataOp::DeleteEdge { id }, false);
        }
    }

    /// Deletes an edge. Returns `true` if the edge existed.
    pub fn delete_edge(&self, id: EdgeId) -> bool {
        changed(&self.write_now(&DataOp::DeleteEdge { id }, false))
    }

    /// Deletes an edge within a transaction context. Returns `true` if the edge existed.
    pub fn delete_edge_versioned(
        &self,
        id: EdgeId,
        epoch: EpochId,
        transaction_id: TransactionId,
    ) -> bool {
        changed(&self.apply(
            &DataOp::DeleteEdge { id },
            Writer::Transaction {
                id: transaction_id,
                snapshot: epoch,
            },
        ))
    }

    /// Sets a property on a node.
    pub fn set_node_property(&self, id: NodeId, key: &str, value: Value) {
        let op = DataOp::SetNodeProperty {
            id,
            key: PropertyKey::new(key),
            value,
        };
        let _set = self.write_now(&op, false);
    }

    /// Sets a property on an edge.
    pub fn set_edge_property(&self, id: EdgeId, key: &str, value: Value) {
        let op = DataOp::SetEdgeProperty {
            id,
            key: PropertyKey::new(key),
            value,
        };
        let _set = self.write_now(&op, false);
    }

    /// Sets a node property within a transaction context: it writes the value as
    /// the transaction's pending version.
    ///
    /// # Errors
    ///
    /// Returns an error, and changes nothing, when the previous value cannot
    /// be read (a spilled value whose file cannot be read): a rollback could
    /// not restore it.
    pub fn set_node_property_versioned(
        &self,
        id: NodeId,
        key: &str,
        value: Value,
        transaction_id: TransactionId,
    ) -> Result<()> {
        let op = DataOp::SetNodeProperty {
            id,
            key: PropertyKey::new(key),
            value,
        };
        self.write_as(&op, transaction_id)
            .map(|_| ())
            .map_err(|refused| error(&refused))
    }

    /// Sets an edge property within a transaction context.
    pub fn set_edge_property_versioned(
        &self,
        id: EdgeId,
        key: &str,
        value: Value,
        transaction_id: TransactionId,
    ) {
        let op = DataOp::SetEdgeProperty {
            id,
            key: PropertyKey::new(key),
            value,
        };
        let _set = self.write_as(&op, transaction_id);
    }

    /// Removes a node property within a transaction context. Returns the
    /// previous value if it existed.
    ///
    /// # Errors
    ///
    /// Returns an error, and changes nothing, when the previous value cannot
    /// be read (a spilled value whose file cannot be read).
    pub fn remove_node_property_versioned(
        &self,
        id: NodeId,
        key: &str,
        transaction_id: TransactionId,
    ) -> Result<Option<Value>> {
        let op = DataOp::RemoveNodeProperty {
            id,
            key: PropertyKey::new(key),
        };
        Self::removed(self.write_as(&op, transaction_id))
    }

    /// Removes an edge property within a transaction context.
    ///
    /// # Errors
    ///
    /// Returns an error, and changes nothing, when the previous value cannot
    /// be read.
    pub fn remove_edge_property_versioned(
        &self,
        id: EdgeId,
        key: &str,
        transaction_id: TransactionId,
    ) -> Result<Option<Value>> {
        let op = DataOp::RemoveEdgeProperty {
            id,
            key: PropertyKey::new(key),
        };
        Self::removed(self.write_as(&op, transaction_id))
    }

    /// Adds a label within a transaction context.
    pub fn add_label_versioned(
        &self,
        node_id: NodeId,
        label: &str,
        transaction_id: TransactionId,
    ) -> bool {
        changed(&self.write_as(
            &DataOp::AddNodeLabel {
                id: node_id,
                label: ArcStr::from(label),
            },
            transaction_id,
        ))
    }

    /// Removes a label within a transaction context.
    pub fn remove_label_versioned(
        &self,
        node_id: NodeId,
        label: &str,
        transaction_id: TransactionId,
    ) -> bool {
        changed(&self.write_as(
            &DataOp::RemoveNodeLabel {
                id: node_id,
                label: ArcStr::from(label),
            },
            transaction_id,
        ))
    }

    /// Removes a property from a node. Returns the previous value if it existed.
    ///
    /// # Errors
    ///
    /// Returns an error, and changes nothing, when the value cannot be read
    /// (a spilled value whose file cannot be read): the caller logs the
    /// removal, and a hidden value reported absent would not be logged.
    pub fn remove_node_property(&self, id: NodeId, key: &str) -> Result<Option<Value>> {
        let op = DataOp::RemoveNodeProperty {
            id,
            key: PropertyKey::new(key),
        };
        Self::removed(self.write_now(&op, true))
    }

    /// Removes a property from an edge. Returns the previous value if it existed.
    ///
    /// # Errors
    ///
    /// Returns an error, and changes nothing, when the value cannot be read,
    /// as [`remove_node_property`](Self::remove_node_property) does.
    pub fn remove_edge_property(&self, id: EdgeId, key: &str) -> Result<Option<Value>> {
        let op = DataOp::RemoveEdgeProperty {
            id,
            key: PropertyKey::new(key),
        };
        Self::removed(self.write_now(&op, true))
    }

    /// Adds a label to a node. Returns `true` if the label was new.
    pub fn add_label(&self, node_id: NodeId, label: &str) -> bool {
        changed(&self.write_now(
            &DataOp::AddNodeLabel {
                id: node_id,
                label: ArcStr::from(label),
            },
            false,
        ))
    }

    /// Removes a label from a node. Returns `true` if the label existed.
    pub fn remove_label(&self, node_id: NodeId, label: &str) -> bool {
        changed(&self.write_now(
            &DataOp::RemoveNodeLabel {
                id: node_id,
                label: ArcStr::from(label),
            },
            false,
        ))
    }

    /// Creates a node with labels and properties: a create, then a set of
    /// each property, each an immediate write.
    pub fn create_node_with_props(
        &self,
        labels: &[&str],
        properties: &[(PropertyKey, Value)],
    ) -> NodeId {
        let id = self.create_node(labels);
        for (key, value) in properties {
            self.set_node_property(id, key.as_str(), value.clone());
        }
        id
    }

    /// Creates an edge with properties: a create, then a set of each
    /// property, each an immediate write.
    pub fn create_edge_with_props(
        &self,
        src: NodeId,
        dst: NodeId,
        edge_type: &str,
        properties: &[(PropertyKey, Value)],
    ) -> EdgeId {
        let id = self.create_edge(src, dst, edge_type);
        for (key, value) in properties {
            self.set_edge_property(id, key.as_str(), value.clone());
        }
        id
    }
}
