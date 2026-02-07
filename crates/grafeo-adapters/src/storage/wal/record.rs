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

#[cfg(test)]
mod tests {
    use super::*;

    fn roundtrip(record: &WalRecord) -> WalRecord {
        let json = serde_json::to_string(record).unwrap();
        serde_json::from_str(&json).unwrap()
    }

    #[test]
    fn test_create_node_roundtrip() {
        let record = WalRecord::CreateNode {
            id: NodeId::new(1),
            labels: vec!["Person".to_string(), "Employee".to_string()],
        };
        let parsed = roundtrip(&record);
        match parsed {
            WalRecord::CreateNode { id, labels } => {
                assert_eq!(id, NodeId::new(1));
                assert_eq!(labels, vec!["Person", "Employee"]);
            }
            _ => panic!("Wrong variant"),
        }
    }

    #[test]
    fn test_delete_node_roundtrip() {
        let record = WalRecord::DeleteNode {
            id: NodeId::new(42),
        };
        let parsed = roundtrip(&record);
        match parsed {
            WalRecord::DeleteNode { id } => assert_eq!(id, NodeId::new(42)),
            _ => panic!("Wrong variant"),
        }
    }

    #[test]
    fn test_create_edge_roundtrip() {
        let record = WalRecord::CreateEdge {
            id: EdgeId::new(10),
            src: NodeId::new(1),
            dst: NodeId::new(2),
            edge_type: "KNOWS".to_string(),
        };
        let parsed = roundtrip(&record);
        match parsed {
            WalRecord::CreateEdge {
                id,
                src,
                dst,
                edge_type,
            } => {
                assert_eq!(id, EdgeId::new(10));
                assert_eq!(src, NodeId::new(1));
                assert_eq!(dst, NodeId::new(2));
                assert_eq!(edge_type, "KNOWS");
            }
            _ => panic!("Wrong variant"),
        }
    }

    #[test]
    fn test_delete_edge_roundtrip() {
        let record = WalRecord::DeleteEdge {
            id: EdgeId::new(99),
        };
        let parsed = roundtrip(&record);
        match parsed {
            WalRecord::DeleteEdge { id } => assert_eq!(id, EdgeId::new(99)),
            _ => panic!("Wrong variant"),
        }
    }

    #[test]
    fn test_set_node_property_roundtrip() {
        let record = WalRecord::SetNodeProperty {
            id: NodeId::new(5),
            key: "name".to_string(),
            value: Value::String("Alice".into()),
        };
        let parsed = roundtrip(&record);
        match parsed {
            WalRecord::SetNodeProperty { id, key, value } => {
                assert_eq!(id, NodeId::new(5));
                assert_eq!(key, "name");
                assert_eq!(value, Value::String("Alice".into()));
            }
            _ => panic!("Wrong variant"),
        }
    }

    #[test]
    fn test_set_edge_property_roundtrip() {
        let record = WalRecord::SetEdgeProperty {
            id: EdgeId::new(7),
            key: "weight".to_string(),
            value: Value::Float64(3.14),
        };
        let parsed = roundtrip(&record);
        match parsed {
            WalRecord::SetEdgeProperty { id, key, value } => {
                assert_eq!(id, EdgeId::new(7));
                assert_eq!(key, "weight");
                assert_eq!(value, Value::Float64(3.14));
            }
            _ => panic!("Wrong variant"),
        }
    }

    #[test]
    fn test_add_node_label_roundtrip() {
        let record = WalRecord::AddNodeLabel {
            id: NodeId::new(3),
            label: "Admin".to_string(),
        };
        let parsed = roundtrip(&record);
        match parsed {
            WalRecord::AddNodeLabel { id, label } => {
                assert_eq!(id, NodeId::new(3));
                assert_eq!(label, "Admin");
            }
            _ => panic!("Wrong variant"),
        }
    }

    #[test]
    fn test_remove_node_label_roundtrip() {
        let record = WalRecord::RemoveNodeLabel {
            id: NodeId::new(3),
            label: "Temp".to_string(),
        };
        let parsed = roundtrip(&record);
        match parsed {
            WalRecord::RemoveNodeLabel { id, label } => {
                assert_eq!(id, NodeId::new(3));
                assert_eq!(label, "Temp");
            }
            _ => panic!("Wrong variant"),
        }
    }

    #[test]
    fn test_tx_commit_roundtrip() {
        let record = WalRecord::TxCommit {
            tx_id: TxId::new(100),
        };
        let parsed = roundtrip(&record);
        match parsed {
            WalRecord::TxCommit { tx_id } => assert_eq!(tx_id, TxId::new(100)),
            _ => panic!("Wrong variant"),
        }
    }

    #[test]
    fn test_tx_abort_roundtrip() {
        let record = WalRecord::TxAbort {
            tx_id: TxId::new(200),
        };
        let parsed = roundtrip(&record);
        match parsed {
            WalRecord::TxAbort { tx_id } => assert_eq!(tx_id, TxId::new(200)),
            _ => panic!("Wrong variant"),
        }
    }

    #[test]
    fn test_checkpoint_roundtrip() {
        let record = WalRecord::Checkpoint {
            tx_id: TxId::new(50),
        };
        let parsed = roundtrip(&record);
        match parsed {
            WalRecord::Checkpoint { tx_id } => assert_eq!(tx_id, TxId::new(50)),
            _ => panic!("Wrong variant"),
        }
    }

    #[test]
    fn test_create_node_empty_labels() {
        let record = WalRecord::CreateNode {
            id: NodeId::new(0),
            labels: Vec::new(),
        };
        let parsed = roundtrip(&record);
        match parsed {
            WalRecord::CreateNode { labels, .. } => assert!(labels.is_empty()),
            _ => panic!("Wrong variant"),
        }
    }
}
