//! WAL record types and the [`WalEntry`] trait.

use grafeo_common::types::{EdgeId, EpochId, NodeId, TransactionId, Value};
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};

/// Trait for WAL record types, enabling type-safe WAL instances.
///
/// Implement this for each storage model's record enum (e.g., [`WalRecord`]
/// for LPG, a future `RdfWalRecord` for RDF). The [`TypedWal`](super::TypedWal)
/// wrapper uses these methods to handle durability decisions and transaction
/// semantics without knowing the concrete record type.
pub trait WalEntry: Serialize + DeserializeOwned + Send + Sync + std::fmt::Debug + Clone {
    /// Whether this record should force an immediate fsync in Sync durability mode.
    ///
    /// Returns `true` for commit markers.
    fn requires_sync(&self) -> bool;

    /// Whether this is a transaction commit record.
    fn is_commit(&self) -> bool;

    /// Whether this is a transaction abort record.
    fn is_abort(&self) -> bool;

    /// Whether this is a checkpoint record.
    fn is_checkpoint(&self) -> bool;

    /// Whether this is a metadata record (e.g., epoch advance).
    ///
    /// Metadata records are not part of any transaction and are always
    /// included in recovery output. They carry structural information
    /// used by backup and point-in-time recovery.
    fn is_metadata(&self) -> bool {
        false
    }

    /// Creates a checkpoint record for this WAL type.
    fn make_checkpoint(transaction_id: TransactionId) -> Self;
}

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

    /// Remove a property from a node.
    RemoveNodeProperty {
        /// Node ID.
        id: NodeId,
        /// Property key.
        key: String,
    },

    /// Remove a property from an edge.
    RemoveEdgeProperty {
        /// Edge ID.
        id: EdgeId,
        /// Property key.
        key: String,
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

    // === Schema DDL Records ===
    /// Register a node type definition.
    CreateNodeType {
        /// Type name (corresponds to a label).
        name: String,
        /// Property definitions: (name, data_type, nullable).
        properties: Vec<(String, String, bool)>,
        /// Constraints: (kind, property_names), where the kind is a
        /// [`TypeConstraintKind`].
        constraints: Vec<(String, Vec<String>)>,
    },

    /// Drop a node type definition.
    DropNodeType {
        /// Type name.
        name: String,
    },

    /// Register an edge type definition.
    CreateEdgeType {
        /// Type name.
        name: String,
        /// Property definitions: (name, data_type, nullable).
        properties: Vec<(String, String, bool)>,
        /// Constraints: (kind, property_names), where the kind is a
        /// [`TypeConstraintKind`].
        constraints: Vec<(String, Vec<String>)>,
    },

    /// Drop an edge type definition.
    DropEdgeType {
        /// Type name.
        name: String,
    },

    /// Create an index.
    CreateIndex {
        /// Index name.
        name: String,
        /// Target label.
        label: String,
        /// Target property.
        property: String,
        /// Index kind: "property", "text", "vector", "btree".
        index_type: String,
    },

    /// Drop an index.
    DropIndex {
        /// Index name.
        name: String,
    },

    /// Create a constraint.
    CreateConstraint {
        /// Constraint name.
        name: String,
        /// Target label.
        label: String,
        /// Target properties.
        properties: Vec<String>,
        /// Constraint kind, a [`NamedConstraintKind`].
        kind: String,
    },

    /// Drop a constraint.
    DropConstraint {
        /// Constraint name.
        name: String,
    },

    /// Register a graph type definition.
    CreateGraphType {
        /// Type name.
        name: String,
        /// Allowed node types.
        node_types: Vec<String>,
        /// Allowed edge types.
        edge_types: Vec<String>,
        /// Whether unlisted types are allowed.
        open: bool,
    },

    /// Drop a graph type definition.
    DropGraphType {
        /// Type name.
        name: String,
    },

    /// Register a schema namespace.
    CreateSchema {
        /// Schema name.
        name: String,
    },

    /// Drop a schema namespace.
    DropSchema {
        /// Schema name.
        name: String,
    },

    /// Alter a node type (add/drop properties).
    AlterNodeType {
        /// Type name.
        name: String,
        /// Alterations: ("add", prop_name, type, nullable) or ("drop", prop_name, "", false).
        /// The action is a [`PropertyAlterationKind`].
        alterations: Vec<(String, String, String, bool)>,
    },

    /// Alter an edge type (add/drop properties).
    AlterEdgeType {
        /// Type name.
        name: String,
        /// Alterations: ("add", prop_name, type, nullable) or ("drop", prop_name, "", false).
        /// The action is a [`PropertyAlterationKind`].
        alterations: Vec<(String, String, String, bool)>,
    },

    /// Alter a graph type (add/drop node/edge types).
    AlterGraphType {
        /// Graph type name.
        name: String,
        /// Alterations: (action, type_name), where the action is a
        /// [`GraphTypeAlterationKind`].
        alterations: Vec<(String, String)>,
    },

    /// Create a stored procedure.
    CreateProcedure {
        /// Procedure name.
        name: String,
        /// Parameters: (name, type).
        params: Vec<(String, String)>,
        /// Return columns: (name, type).
        returns: Vec<(String, String)>,
        /// Raw GQL body.
        body: String,
    },

    /// Drop a stored procedure.
    DropProcedure {
        /// Procedure name.
        name: String,
    },

    // === Named Graph Lifecycle ===
    /// Create a named graph partition.
    CreateNamedGraph {
        /// Graph name.
        name: String,
    },

    /// Drop a named graph partition.
    DropNamedGraph {
        /// Graph name.
        name: String,
    },

    /// Switch the WAL replay cursor to a named graph.
    ///
    /// Subsequent data mutation records apply to this graph until the next
    /// `SwitchGraph`. `None` switches back to the default graph.
    SwitchGraph {
        /// Target graph name, or `None` for the default graph.
        name: Option<String>,
    },

    // === RDF Records ===
    /// Insert an RDF triple into the default or a named graph.
    ///
    /// Term strings use N-Triples encoding: IRIs as `<iri>`, literals as
    /// `"value"`, typed literals as `"value"^^<type>`, blank nodes as `_:id`.
    InsertRdfTriple {
        /// Subject term (N-Triples encoding).
        subject: String,
        /// Predicate term (N-Triples encoding).
        predicate: String,
        /// Object term (N-Triples encoding).
        object: String,
        /// Target graph name (`None` = default graph).
        graph: Option<String>,
    },

    /// Delete an RDF triple from the default or a named graph.
    DeleteRdfTriple {
        /// Subject term (N-Triples encoding).
        subject: String,
        /// Predicate term (N-Triples encoding).
        predicate: String,
        /// Object term (N-Triples encoding).
        object: String,
        /// Target graph name (`None` = default graph).
        graph: Option<String>,
    },

    /// Clear all triples from an RDF graph.
    ClearRdfGraph {
        /// Graph name (`None` = default graph).
        graph: Option<String>,
    },

    /// Create an RDF named graph.
    CreateRdfGraph {
        /// Graph name.
        name: String,
    },

    /// Drop an RDF named graph.
    DropRdfGraph {
        /// Graph name (`None` = drop/clear default graph).
        name: Option<String>,
    },

    // === Transaction Control ===
    /// Transaction commit.
    TransactionCommit {
        /// Transaction ID.
        transaction_id: TransactionId,
    },

    /// Transaction abort.
    TransactionAbort {
        /// Transaction ID.
        transaction_id: TransactionId,
    },

    /// Checkpoint marker.
    Checkpoint {
        /// Transaction ID at checkpoint.
        transaction_id: TransactionId,
    },

    // === Metadata ===
    /// Marks an epoch boundary in the WAL.
    ///
    /// Logged after each `TransactionCommit` to record the new epoch value.
    /// Used by incremental backup to identify WAL segment boundaries and
    /// by point-in-time recovery to stop replay at a target epoch.
    /// Recovery treats this as metadata (no store mutation).
    EpochAdvance {
        /// The epoch after the commit.
        epoch: EpochId,
    },

    // === Standalone changes ===
    /// A change applied on its own (a named graph created or dropped, a
    /// catalog record put or dropped, an RDF graph operation), as the WAL v2
    /// log record that holds it: the framed bytes of a `LogRecord::Standalone`
    /// of grafeo-common's `storage::log_record`. This log carries the record
    /// until the WAL writes v2 groups; replay decodes it and applies it as
    /// the live statement did. Appended last, so every record before it
    /// keeps its encoding.
    Standalone {
        /// The framed log record.
        record: Vec<u8>,
    },
}

impl WalEntry for WalRecord {
    fn requires_sync(&self) -> bool {
        matches!(self, WalRecord::TransactionCommit { .. })
    }

    fn is_commit(&self) -> bool {
        matches!(self, WalRecord::TransactionCommit { .. })
    }

    fn is_abort(&self) -> bool {
        matches!(self, WalRecord::TransactionAbort { .. })
    }

    fn is_checkpoint(&self) -> bool {
        matches!(self, WalRecord::Checkpoint { .. })
    }

    fn is_metadata(&self) -> bool {
        matches!(self, WalRecord::EpochAdvance { .. })
    }

    fn make_checkpoint(transaction_id: TransactionId) -> Self {
        WalRecord::Checkpoint { transaction_id }
    }
}

// === Kinds stored as strings in schema records ===
//
// Schema records store their kinds as strings. Writers and replay both go
// through these enums, so the two cannot drift apart again: `ALTER GRAPH TYPE`
// wrote `"add_node_type"` while replay matched `"add_node"` (#422).

/// Declares a closed set of kinds with the string each one is stored as.
macro_rules! wal_kind {
    (
        $(#[$meta:meta])*
        $name:ident { $($(#[$variant_meta:meta])* $variant:ident => $text:literal,)+ }
    ) => {
        $(#[$meta])*
        #[derive(Debug, Clone, Copy, PartialEq, Eq)]
        pub enum $name {
            $($(#[$variant_meta])* $variant,)+
        }

        impl $name {
            /// The string this kind is stored as in the WAL.
            #[must_use]
            pub const fn as_str(self) -> &'static str {
                match self {
                    $(Self::$variant => $text,)+
                }
            }

            /// Parses a stored kind, or `None` for a kind this version does
            /// not know.
            #[must_use]
            pub fn parse(text: &str) -> Option<Self> {
                match text {
                    $($text => Some(Self::$variant),)+
                    _ => None,
                }
            }
        }
    };
}

wal_kind! {
    /// Kind of a type constraint in a [`WalRecord::CreateNodeType`] or
    /// [`WalRecord::CreateEdgeType`] record.
    TypeConstraintKind {
        /// The properties are unique among nodes (or edges) of the type.
        Unique => "unique",
        /// The properties form the type's primary key.
        PrimaryKey => "primary_key",
        /// The (single) property must be present.
        NotNull => "not_null",
    }
}

wal_kind! {
    /// Kind of a named constraint in a [`WalRecord::CreateConstraint`] record.
    NamedConstraintKind {
        /// The properties are unique among nodes with the label.
        Unique => "unique",
        /// The properties are present and unique together.
        NodeKey => "node_key",
        /// The properties are present (`NOT NULL`).
        NotNull => "not_null",
        /// The properties are present (`EXISTS`, another spelling of `NOT NULL`).
        Exists => "exists",
    }
}

wal_kind! {
    /// Action of an alteration in a [`WalRecord::AlterNodeType`] or
    /// [`WalRecord::AlterEdgeType`] record.
    PropertyAlterationKind {
        /// Add the property.
        Add => "add",
        /// Drop the property.
        Drop => "drop",
    }
}

wal_kind! {
    /// Action of an alteration in a [`WalRecord::AlterGraphType`] record.
    GraphTypeAlterationKind {
        /// Allow a node type in the graph type.
        AddNodeType => "add_node_type",
        /// Remove a node type from the graph type.
        DropNodeType => "drop_node_type",
        /// Allow an edge type in the graph type.
        AddEdgeType => "add_edge_type",
        /// Remove an edge type from the graph type.
        DropEdgeType => "drop_edge_type",
    }
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
            value: Value::String("Alix".into()),
        };
        let parsed = roundtrip(&record);
        match parsed {
            WalRecord::SetNodeProperty { id, key, value } => {
                assert_eq!(id, NodeId::new(5));
                assert_eq!(key, "name");
                assert_eq!(value, Value::String("Alix".into()));
            }
            _ => panic!("Wrong variant"),
        }
    }

    #[test]
    fn test_set_edge_property_roundtrip() {
        let record = WalRecord::SetEdgeProperty {
            id: EdgeId::new(7),
            key: "weight".to_string(),
            value: Value::Float64(std::f64::consts::PI),
        };
        let parsed = roundtrip(&record);
        match parsed {
            WalRecord::SetEdgeProperty { id, key, value } => {
                assert_eq!(id, EdgeId::new(7));
                assert_eq!(key, "weight");
                assert_eq!(value, Value::Float64(std::f64::consts::PI));
            }
            _ => panic!("Wrong variant"),
        }
    }

    #[test]
    fn test_remove_node_property_roundtrip() {
        let record = WalRecord::RemoveNodeProperty {
            id: NodeId::new(5),
            key: "age".to_string(),
        };
        let parsed = roundtrip(&record);
        match parsed {
            WalRecord::RemoveNodeProperty { id, key } => {
                assert_eq!(id, NodeId::new(5));
                assert_eq!(key, "age");
            }
            _ => panic!("Wrong variant"),
        }
    }

    #[test]
    fn test_remove_edge_property_roundtrip() {
        let record = WalRecord::RemoveEdgeProperty {
            id: EdgeId::new(7),
            key: "weight".to_string(),
        };
        let parsed = roundtrip(&record);
        match parsed {
            WalRecord::RemoveEdgeProperty { id, key } => {
                assert_eq!(id, EdgeId::new(7));
                assert_eq!(key, "weight");
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
        let record = WalRecord::TransactionCommit {
            transaction_id: TransactionId::new(100),
        };
        let parsed = roundtrip(&record);
        match parsed {
            WalRecord::TransactionCommit { transaction_id } => {
                assert_eq!(transaction_id, TransactionId::new(100));
            }
            _ => panic!("Wrong variant"),
        }
    }

    #[test]
    fn test_tx_abort_roundtrip() {
        let record = WalRecord::TransactionAbort {
            transaction_id: TransactionId::new(200),
        };
        let parsed = roundtrip(&record);
        match parsed {
            WalRecord::TransactionAbort { transaction_id } => {
                assert_eq!(transaction_id, TransactionId::new(200));
            }
            _ => panic!("Wrong variant"),
        }
    }

    #[test]
    fn test_checkpoint_roundtrip() {
        let record = WalRecord::Checkpoint {
            transaction_id: TransactionId::new(50),
        };
        let parsed = roundtrip(&record);
        match parsed {
            WalRecord::Checkpoint { transaction_id } => {
                assert_eq!(transaction_id, TransactionId::new(50));
            }
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
            WalRecord::CreateNode { labels, .. } => assert!(labels.is_empty(), "{labels:?}"),
            _ => panic!("Wrong variant"),
        }
    }

    #[test]
    fn test_create_named_graph_roundtrip() {
        let record = WalRecord::CreateNamedGraph {
            name: "analytics".to_string(),
        };
        let parsed = roundtrip(&record);
        match parsed {
            WalRecord::CreateNamedGraph { name } => assert_eq!(name, "analytics"),
            _ => panic!("Wrong variant"),
        }
    }

    #[test]
    fn test_drop_named_graph_roundtrip() {
        let record = WalRecord::DropNamedGraph {
            name: "temp".to_string(),
        };
        let parsed = roundtrip(&record);
        match parsed {
            WalRecord::DropNamedGraph { name } => assert_eq!(name, "temp"),
            _ => panic!("Wrong variant"),
        }
    }

    #[test]
    fn test_switch_graph_roundtrip() {
        // Switch to named graph
        let record = WalRecord::SwitchGraph {
            name: Some("analytics".to_string()),
        };
        let parsed = roundtrip(&record);
        match parsed {
            WalRecord::SwitchGraph { name } => assert_eq!(name, Some("analytics".to_string())),
            _ => panic!("Wrong variant"),
        }

        // Switch back to default
        let record = WalRecord::SwitchGraph { name: None };
        let parsed = roundtrip(&record);
        match parsed {
            WalRecord::SwitchGraph { name } => assert_eq!(name, None),
            _ => panic!("Wrong variant"),
        }
    }

    #[test]
    fn test_wal_entry_requires_sync() {
        use super::WalEntry;

        // Only TransactionCommit should require sync
        assert!(
            WalRecord::TransactionCommit {
                transaction_id: TransactionId::new(1)
            }
            .requires_sync()
        );

        assert!(
            !WalRecord::CreateNode {
                id: NodeId::new(1),
                labels: vec![]
            }
            .requires_sync()
        );

        assert!(
            !WalRecord::TransactionAbort {
                transaction_id: TransactionId::new(1)
            }
            .requires_sync()
        );

        assert!(
            !WalRecord::Checkpoint {
                transaction_id: TransactionId::new(1)
            }
            .requires_sync()
        );
    }

    #[test]
    fn test_wal_entry_transaction_markers() {
        use super::WalEntry;

        let commit = WalRecord::TransactionCommit {
            transaction_id: TransactionId::new(1),
        };
        assert!(commit.is_commit());
        assert!(!commit.is_abort());
        assert!(!commit.is_checkpoint());

        let abort = WalRecord::TransactionAbort {
            transaction_id: TransactionId::new(2),
        };
        assert!(!abort.is_commit());
        assert!(abort.is_abort());
        assert!(!abort.is_checkpoint());

        let checkpoint = WalRecord::Checkpoint {
            transaction_id: TransactionId::new(3),
        };
        assert!(!checkpoint.is_commit());
        assert!(!checkpoint.is_abort());
        assert!(checkpoint.is_checkpoint());

        // Data records are none of the above
        let data = WalRecord::CreateNode {
            id: NodeId::new(1),
            labels: vec![],
        };
        assert!(!data.is_commit());
        assert!(!data.is_abort());
        assert!(!data.is_checkpoint());
    }

    #[test]
    fn test_wal_entry_make_checkpoint() {
        use super::WalEntry;

        let record = WalRecord::make_checkpoint(TransactionId::new(42));
        match record {
            WalRecord::Checkpoint { transaction_id } => {
                assert_eq!(transaction_id, TransactionId::new(42));
            }
            _ => panic!("make_checkpoint should produce Checkpoint variant"),
        }
    }

    // =========================================================================
    // T1-05: Serialization round-trip for untested Value types
    // =========================================================================

    #[test]
    fn test_value_map_roundtrip() {
        use grafeo_common::types::PropertyKey;
        use std::collections::BTreeMap;
        use std::sync::Arc;

        let mut map = BTreeMap::new();
        map.insert(PropertyKey::from("name"), Value::String("Alix".into()));
        map.insert(PropertyKey::from("age"), Value::Int64(30));

        let record = WalRecord::SetNodeProperty {
            id: NodeId::new(1),
            key: "metadata".to_string(),
            value: Value::Map(Arc::new(map.clone())),
        };
        let parsed = roundtrip(&record);
        match parsed {
            WalRecord::SetNodeProperty { value, .. } => match value {
                Value::Map(m) => {
                    assert_eq!(m.len(), 2);
                    assert_eq!(m[&PropertyKey::from("name")], Value::String("Alix".into()));
                    assert_eq!(m[&PropertyKey::from("age")], Value::Int64(30));
                }
                other => panic!("Expected Map, got {other:?}"),
            },
            other => panic!("Wrong variant: {other:?}"),
        }
    }

    #[test]
    fn test_value_vector_roundtrip() {
        use std::sync::Arc;

        let embedding: Arc<[f32]> = Arc::from(vec![0.1_f32, 0.2, 0.3, 0.4]);
        let record = WalRecord::SetNodeProperty {
            id: NodeId::new(2),
            key: "embedding".to_string(),
            value: Value::Vector(embedding.clone()),
        };
        let parsed = roundtrip(&record);
        match parsed {
            WalRecord::SetNodeProperty { value, .. } => match value {
                Value::Vector(v) => {
                    assert_eq!(v.len(), 4);
                    assert!((v[0] - 0.1).abs() < f32::EPSILON);
                    assert!((v[3] - 0.4).abs() < f32::EPSILON);
                }
                other => panic!("Expected Vector, got {other:?}"),
            },
            other => panic!("Wrong variant: {other:?}"),
        }
    }

    #[test]
    fn test_value_timestamp_roundtrip() {
        use grafeo_common::types::Timestamp;

        let ts = Timestamp::from_secs(1_700_000_000); // 2023-11-14
        let record = WalRecord::SetNodeProperty {
            id: NodeId::new(3),
            key: "created_at".to_string(),
            value: Value::Timestamp(ts),
        };
        let parsed = roundtrip(&record);
        match parsed {
            WalRecord::SetNodeProperty { value, .. } => match value {
                Value::Timestamp(t) => {
                    assert_eq!(t.as_secs(), 1_700_000_000);
                }
                other => panic!("Expected Timestamp, got {other:?}"),
            },
            other => panic!("Wrong variant: {other:?}"),
        }
    }

    #[test]
    fn test_value_zoned_datetime_roundtrip() {
        use grafeo_common::types::{Timestamp, ZonedDatetime};

        let ts = Timestamp::from_secs(1_700_000_000);
        let zdt = ZonedDatetime::from_timestamp_offset(ts, 3600); // +01:00
        let record = WalRecord::SetNodeProperty {
            id: NodeId::new(4),
            key: "event_time".to_string(),
            value: Value::ZonedDatetime(zdt),
        };
        let parsed = roundtrip(&record);
        match parsed {
            WalRecord::SetNodeProperty { value, .. } => match value {
                Value::ZonedDatetime(z) => {
                    assert_eq!(z.as_timestamp().as_secs(), 1_700_000_000);
                    assert_eq!(z.offset_seconds(), 3600);
                }
                other => panic!("Expected ZonedDatetime, got {other:?}"),
            },
            other => panic!("Wrong variant: {other:?}"),
        }
    }

    #[test]
    fn test_value_path_roundtrip() {
        use std::sync::Arc;

        let nodes: Arc<[Value]> = Arc::from(vec![
            Value::String("node_A".into()),
            Value::String("node_B".into()),
            Value::String("node_C".into()),
        ]);
        let edges: Arc<[Value]> = Arc::from(vec![
            Value::String("edge_AB".into()),
            Value::String("edge_BC".into()),
        ]);
        let record = WalRecord::SetNodeProperty {
            id: NodeId::new(5),
            key: "route".to_string(),
            value: Value::Path {
                nodes: nodes.clone(),
                edges: edges.clone(),
            },
        };
        let parsed = roundtrip(&record);
        match parsed {
            WalRecord::SetNodeProperty { value, .. } => match value {
                Value::Path { nodes: n, edges: e } => {
                    assert_eq!(n.len(), 3);
                    assert_eq!(e.len(), 2);
                    assert_eq!(n[0], Value::String("node_A".into()));
                    assert_eq!(e[1], Value::String("edge_BC".into()));
                }
                other => panic!("Expected Path, got {other:?}"),
            },
            other => panic!("Wrong variant: {other:?}"),
        }
    }

    #[test]
    fn test_value_date_roundtrip() {
        use grafeo_common::types::Date;

        let date = Date::from_ymd(2024, 6, 15).unwrap();
        let record = WalRecord::SetNodeProperty {
            id: NodeId::new(6),
            key: "birthday".to_string(),
            value: Value::Date(date),
        };
        let parsed = roundtrip(&record);
        match parsed {
            WalRecord::SetNodeProperty { value, .. } => match value {
                Value::Date(d) => {
                    assert_eq!(d.year(), 2024);
                    assert_eq!(d.month(), 6);
                    assert_eq!(d.day(), 15);
                }
                other => panic!("Expected Date, got {other:?}"),
            },
            other => panic!("Wrong variant: {other:?}"),
        }
    }

    #[test]
    fn test_value_time_roundtrip() {
        use grafeo_common::types::Time;

        let time = Time::from_hms(14, 30, 45).unwrap().with_offset(3600);
        let record = WalRecord::SetNodeProperty {
            id: NodeId::new(7),
            key: "alarm".to_string(),
            value: Value::Time(time),
        };
        let parsed = roundtrip(&record);
        match parsed {
            WalRecord::SetNodeProperty { value, .. } => match value {
                Value::Time(t) => {
                    assert_eq!(t.hour(), 14);
                    assert_eq!(t.minute(), 30);
                    assert_eq!(t.second(), 45);
                    assert_eq!(t.offset_seconds(), Some(3600));
                }
                other => panic!("Expected Time, got {other:?}"),
            },
            other => panic!("Wrong variant: {other:?}"),
        }
    }

    #[test]
    fn test_value_duration_roundtrip() {
        use grafeo_common::types::Duration;

        let dur = Duration::new(14, 3, 4 * 3_600_000_000_000 + 5 * 60_000_000_000); // P1Y2M3DT4H5M
        let record = WalRecord::SetNodeProperty {
            id: NodeId::new(8),
            key: "interval".to_string(),
            value: Value::Duration(dur),
        };
        let parsed = roundtrip(&record);
        match parsed {
            WalRecord::SetNodeProperty { value, .. } => match value {
                Value::Duration(d) => {
                    assert_eq!(d.months(), 14);
                    assert_eq!(d.days(), 3);
                }
                other => panic!("Expected Duration, got {other:?}"),
            },
            other => panic!("Wrong variant: {other:?}"),
        }
    }

    #[test]
    fn test_epoch_advance_roundtrip() {
        let record = WalRecord::EpochAdvance {
            epoch: EpochId::new(42),
        };
        let parsed = roundtrip(&record);
        match parsed {
            WalRecord::EpochAdvance { epoch } => assert_eq!(epoch, EpochId::new(42)),
            other => panic!("Wrong variant: {other:?}"),
        }
    }

    #[test]
    fn test_epoch_advance_is_metadata() {
        let record = WalRecord::EpochAdvance {
            epoch: EpochId::new(1),
        };
        assert!(!record.requires_sync());
        assert!(!record.is_commit());
        assert!(!record.is_abort());
        assert!(!record.is_checkpoint());
        assert!(record.is_metadata());
    }

    #[test]
    fn test_non_metadata_records() {
        let commit = WalRecord::TransactionCommit {
            transaction_id: TransactionId::new(1),
        };
        assert!(!commit.is_metadata());

        let create = WalRecord::CreateNode {
            id: NodeId::new(1),
            labels: vec![],
        };
        assert!(!create.is_metadata());
    }

    /// The stored spellings are part of the WAL format: these are the strings
    /// 0.5.43 wrote, so its WAL files still replay.
    #[test]
    fn schema_kinds_keep_their_stored_spellings() {
        use GraphTypeAlterationKind as Graph;
        use PropertyAlterationKind as Property;
        use TypeConstraintKind as Constraint;

        let graph = [
            (Graph::AddNodeType, "add_node_type"),
            (Graph::DropNodeType, "drop_node_type"),
            (Graph::AddEdgeType, "add_edge_type"),
            (Graph::DropEdgeType, "drop_edge_type"),
        ];
        for (kind, text) in graph {
            assert_eq!(kind.as_str(), text);
            assert_eq!(Graph::parse(text), Some(kind));
        }
        for (kind, text) in [(Property::Add, "add"), (Property::Drop, "drop")] {
            assert_eq!(kind.as_str(), text);
            assert_eq!(Property::parse(text), Some(kind));
        }
        let constraints = [
            (Constraint::Unique, "unique"),
            (Constraint::PrimaryKey, "primary_key"),
            (Constraint::NotNull, "not_null"),
        ];
        for (kind, text) in constraints {
            assert_eq!(kind.as_str(), text);
            assert_eq!(Constraint::parse(text), Some(kind));
        }

        // What replay used to match for graph types was never written.
        assert_eq!(Graph::parse("add_node"), None);
        assert_eq!(Constraint::parse("check"), None);
    }
}
