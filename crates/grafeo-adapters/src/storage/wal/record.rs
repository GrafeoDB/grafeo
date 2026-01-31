//! WAL record types.

use grafeo_common::types::{EdgeId, NodeId, TxId, Value};
use serde::{Deserialize, Serialize};

/// A record in the Write-Ahead Log.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum WalRecord {
    /// Create a new node.
    CreateNode {
        /// Node ID.
        id: NodeId,
        /// Labels for the node.
        labels: Vec<String>,
    },

    /// Delete a node.
    DeleteNode {
        /// Node ID.
        id: NodeId,
    },

    /// Create a new edge.
    CreateEdge {
        /// Edge ID.
        id: EdgeId,
        /// Source node ID.
        src: NodeId,
        /// Destination node ID.
        dst: NodeId,
        /// Edge type.
        edge_type: String,
    },

    /// Delete an edge.
    DeleteEdge {
        /// Edge ID.
        id: EdgeId,
    },

    /// Set a property on a node.
    SetNodeProperty {
        /// Node ID.
        id: NodeId,
        /// Property key.
        key: String,
        /// Property value.
        value: Value,
    },

    /// Set a property on an edge.
    SetEdgeProperty {
        /// Edge ID.
        id: EdgeId,
        /// Property key.
        key: String,
        /// Property value.
        value: Value,
    },

    /// Add a label to a node.
    AddNodeLabel {
        /// Node ID.
        id: NodeId,
        /// Label to add.
        label: String,
    },

    /// Remove a label from a node.
    RemoveNodeLabel {
        /// Node ID.
        id: NodeId,
        /// Label to remove.
        label: String,
    },

    /// Transaction commit.
    TxCommit {
        /// Transaction ID.
        tx_id: TxId,
    },

    /// Transaction abort.
    TxAbort {
        /// Transaction ID.
        tx_id: TxId,
    },

    /// Checkpoint marker.
    Checkpoint {
        /// Transaction ID at checkpoint.
        tx_id: TxId,
    },
}
