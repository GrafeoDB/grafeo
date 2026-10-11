//! The row-group store's raw writes (`GraphStoreMut`), through `apply`.
//!
//! A raw write commits at once: it is an immediate write at the next epoch.
//! A `*_versioned` raw write is a pending write of its transaction that no
//! change set records, as `LpgStore`'s are (raw store handles, QJ).

use std::sync::atomic::Ordering;

use grafeo_common::change::{Before, DataOp, Labels};
use grafeo_common::types::{ArcStr, EdgeId, EpochId, NodeId, PropertyKey, TransactionId, Value};
use grafeo_common::utils::error::{Error, Result};

use super::RowGroupStore;
use crate::graph::Direction;
use crate::graph::apply::{Applied, ApplyError, ChangeTarget, Writer};
use crate::graph::traits::GraphStoreMut;

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

impl GraphStoreMut for RowGroupStore {
    fn create_node(&self, names: &[&str]) -> NodeId {
        let id = self.reserve_node();
        let op = DataOp::CreateNode {
            id,
            labels: labels(names),
            properties: Vec::new(),
        };
        let _created = self.write_now(&op, false);
        id
    }

    fn create_node_versioned(
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

    fn create_edge(&self, src: NodeId, dst: NodeId, edge_type: &str) -> EdgeId {
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

    fn create_edge_versioned(
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

    fn batch_create_edges(&self, edges: &[(NodeId, NodeId, &str)]) -> Vec<EdgeId> {
        edges
            .iter()
            .map(|(src, dst, edge_type)| self.create_edge(*src, *dst, edge_type))
            .collect()
    }

    fn delete_node(&self, id: NodeId) -> bool {
        changed(&self.write_now(&DataOp::DeleteNode { id }, false))
    }

    fn delete_node_versioned(
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

    fn delete_node_edges(&self, node_id: NodeId) {
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

    fn delete_edge(&self, id: EdgeId) -> bool {
        changed(&self.write_now(&DataOp::DeleteEdge { id }, false))
    }

    fn delete_edge_versioned(
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

    fn set_node_property(&self, id: NodeId, key: &str, value: Value) {
        let op = DataOp::SetNodeProperty {
            id,
            key: PropertyKey::new(key),
            value,
        };
        let _set = self.write_now(&op, false);
    }

    fn set_edge_property(&self, id: EdgeId, key: &str, value: Value) {
        let op = DataOp::SetEdgeProperty {
            id,
            key: PropertyKey::new(key),
            value,
        };
        let _set = self.write_now(&op, false);
    }

    fn remove_node_property(&self, id: NodeId, key: &str) -> Result<Option<Value>> {
        let op = DataOp::RemoveNodeProperty {
            id,
            key: PropertyKey::new(key),
        };
        Self::removed(self.write_now(&op, true))
    }

    fn remove_edge_property(&self, id: EdgeId, key: &str) -> Result<Option<Value>> {
        let op = DataOp::RemoveEdgeProperty {
            id,
            key: PropertyKey::new(key),
        };
        Self::removed(self.write_now(&op, true))
    }

    fn add_label(&self, node_id: NodeId, label: &str) -> bool {
        changed(&self.write_now(
            &DataOp::AddNodeLabel {
                id: node_id,
                label: ArcStr::from(label),
            },
            false,
        ))
    }

    fn remove_label(&self, node_id: NodeId, label: &str) -> bool {
        changed(&self.write_now(
            &DataOp::RemoveNodeLabel {
                id: node_id,
                label: ArcStr::from(label),
            },
            false,
        ))
    }
}
