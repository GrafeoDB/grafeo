//! Bridge between mutation operators and the transaction manager's write tracking.

use std::sync::Arc;

use grafeo_common::types::{EdgeId, NodeId, TransactionId};
use grafeo_core::execution::operators::{OperatorError, WriteInProgress, WriteTracker};

use super::{GraphEntity, TransactionManager};

/// Implements [`WriteTracker`] by forwarding to [`TransactionManager::record_write`],
/// and marks writes in progress for checkpoints with the manager's write
/// freeze.
///
/// Created by the planner when a transaction is active, and passed to each
/// mutation operator so it can record writes for conflict detection.
pub struct TransactionWriteTracker {
    manager: Arc<TransactionManager>,
    /// The storage key of the graph the writes go to; `None` for the default
    /// graph.
    graph: Option<Arc<str>>,
}

impl TransactionWriteTracker {
    /// Creates a write tracker for writes to the default graph.
    pub fn new(manager: Arc<TransactionManager>) -> Self {
        Self {
            manager,
            graph: None,
        }
    }

    /// Records the writes as writes to the graph with storage key `graph`
    /// (`None`: the default graph).
    #[must_use]
    pub fn in_graph(mut self, graph: Option<&str>) -> Self {
        self.graph = graph.map(Arc::from);
        self
    }
}

impl WriteTracker for TransactionWriteTracker {
    fn write_in_progress(&self) -> WriteInProgress<'_> {
        self.manager.write_in_progress()
    }

    fn record_node_write(
        &self,
        transaction_id: TransactionId,
        node_id: NodeId,
    ) -> Result<(), OperatorError> {
        self.manager
            .record_write(
                transaction_id,
                GraphEntity::new(self.graph.clone(), node_id),
            )
            .map_err(|e| OperatorError::WriteConflict(e.to_string()))
    }

    fn record_node_delete(
        &self,
        transaction_id: TransactionId,
        node_id: NodeId,
    ) -> Result<(), OperatorError> {
        self.manager
            .record_delete(
                transaction_id,
                GraphEntity::new(self.graph.clone(), node_id),
            )
            .map_err(|e| OperatorError::WriteConflict(e.to_string()))
    }

    fn record_edge_endpoints(
        &self,
        transaction_id: TransactionId,
        src: NodeId,
        dst: NodeId,
    ) -> Result<(), OperatorError> {
        self.manager
            .record_endpoints(
                transaction_id,
                [
                    GraphEntity::new(self.graph.clone(), src),
                    GraphEntity::new(self.graph.clone(), dst),
                ],
            )
            .map_err(|e| OperatorError::WriteConflict(e.to_string()))
    }

    fn record_edge_write(
        &self,
        transaction_id: TransactionId,
        edge_id: EdgeId,
    ) -> Result<(), OperatorError> {
        self.manager
            .record_write(
                transaction_id,
                GraphEntity::new(self.graph.clone(), edge_id),
            )
            .map_err(|e| OperatorError::WriteConflict(e.to_string()))
    }
}
