//! Transaction-aware storage-level batch node/edge creation.
//!
//! Public contract (W1-GRAFEO-BATCH, plan §3.4): provider-neutral,
//! explicit-transaction-required batch mutation on [`Session`]. These methods
//! are **not** a loop over the row-oriented `create_node_with_props` /
//! `create_edge_with_props`; they delegate to a single storage-level batch
//! primitive ([`LpgStore::create_nodes_batch_versioned`] /
//! [`create_edges_batch_versioned`]) that hoists every per-row lock to once per
//! batch and allocates the ID range with one atomic.
//!
//! Locked contracts honored here:
//! - fail if no explicit transaction is active;
//! - prevalidate every property size before the first mutation;
//! - version every row under the active transaction ID (PENDING visibility);
//! - register every node and edge write with transaction management;
//! - preserve WAL records (identical record stream to the row path);
//! - buffer vector-index intents until commit;
//! - preserve returned ID order;
//! - never call the offline/unindexed bulk loaders.
//!
//! Commit/rollback need no batch-specific logic: rows are versioned identically
//! to the row path, so the transaction-wide `finalize_version_epochs` (commit)
//! and `discard_uncommitted_versions` (rollback) scans make the whole batch
//! visible or remove it atomically — including secondary structures published
//! eagerly by the create path (label/property indexes, adjacency, type counts).

use grafeo_common::types::{EdgeId, NodeId, PropertyKey, Value};
use grafeo_common::utils::error::{Error, Result, TransactionError};
use grafeo_core::graph::lpg::{BatchEdgeCreate, BatchNodeCreate};

use super::Session;
#[cfg(all(feature = "lpg", feature = "vector-index"))]
use super::VectorIndexIntent;

/// A node to create within a transactional batch.
///
/// Provider-neutral DTO. Properties are owned so the batch can prevalidate sizes
/// and then hand them to the storage layer in one shot.
#[derive(Debug, Clone)]
pub struct TransactionalNodeCreate {
    /// Labels applied to the node.
    pub labels: Vec<String>,
    /// Property key/value pairs, written in order.
    pub properties: Vec<(String, Value)>,
}

impl TransactionalNodeCreate {
    /// Creates a node spec with labels and no properties.
    #[must_use]
    pub fn new(labels: impl IntoIterator<Item = impl Into<String>>) -> Self {
        Self {
            labels: labels.into_iter().map(Into::into).collect(),
            properties: Vec::new(),
        }
    }

    /// Adds a property, returning self for chaining.
    #[must_use]
    pub fn with_property(mut self, key: impl Into<String>, value: impl Into<Value>) -> Self {
        self.properties.push((key.into(), value.into()));
        self
    }
}

/// An edge to create within a transactional batch.
#[derive(Debug, Clone)]
pub struct TransactionalEdgeCreate {
    /// Source node.
    pub source: NodeId,
    /// Target node.
    pub target: NodeId,
    /// Edge type name.
    pub edge_type: String,
    /// Property key/value pairs, written in order.
    pub properties: Vec<(String, Value)>,
}

impl TransactionalEdgeCreate {
    /// Creates an edge spec with no properties.
    #[must_use]
    pub fn new(source: NodeId, target: NodeId, edge_type: impl Into<String>) -> Self {
        Self {
            source,
            target,
            edge_type: edge_type.into(),
            properties: Vec::new(),
        }
    }

    /// Adds a property, returning self for chaining.
    #[must_use]
    pub fn with_property(mut self, key: impl Into<String>, value: impl Into<Value>) -> Self {
        self.properties.push((key.into(), value.into()));
        self
    }
}

impl Session {
    /// Batch-creates nodes inside the active transaction using one storage-level
    /// batch operation, returning the new IDs in input order.
    ///
    /// # Errors
    ///
    /// - if no explicit transaction is active;
    /// - if any property value exceeds the configured `max_property_size`
    ///   (checked for the whole batch before any mutation);
    /// - on a write-write conflict registering the batch with the transaction.
    #[cfg(feature = "lpg")]
    pub fn create_nodes_with_props_transactional(
        &self,
        nodes: &[TransactionalNodeCreate],
    ) -> Result<Vec<NodeId>> {
        let transaction_id = self.require_active_transaction()?;

        // Preflight: validate every property size before the first mutation.
        for node in nodes {
            for (key, value) in &node.properties {
                self.check_property_size(key, value)?;
            }
        }

        let (epoch, _) = self.get_transaction_context();
        #[cfg(feature = "vector-index")]
        let graph_name = self.active_graph_storage_key();

        // Labels are borrowed `&[&str]` in the storage DTO; build per-node label
        // slices that reference the caller's owned strings, then the storage DTOs.
        let label_slices: Vec<Vec<&str>> = nodes
            .iter()
            .map(|n| n.labels.iter().map(String::as_str).collect())
            .collect();
        let storage_nodes: Vec<BatchNodeCreate<'_>> = nodes
            .iter()
            .zip(label_slices.iter())
            .map(|(n, labels)| BatchNodeCreate {
                labels,
                properties: n
                    .properties
                    .iter()
                    .map(|(k, v)| (PropertyKey::new(k.clone()), v.clone()))
                    .collect(),
            })
            .collect();

        let store = self.active_lpg_store();
        let ids = store.create_nodes_batch_versioned(&storage_nodes, epoch, transaction_id);

        // Register every node write with transaction management (one lock).
        let entities: Vec<crate::transaction::EntityId> = ids
            .iter()
            .copied()
            .map(crate::transaction::EntityId::Node)
            .collect();
        self.transaction_manager
            .record_writes(transaction_id, &entities)?;

        // WAL: identical record stream to the row path (CreateNode + SetNodeProperty*).
        #[cfg(feature = "wal")]
        for (node, &id) in nodes.iter().zip(ids.iter()) {
            self.log_wal_record(&grafeo_storage::wal::WalRecord::CreateNode {
                id,
                labels: node.labels.clone(),
            });
            for (key, value) in &node.properties {
                self.log_wal_record(&grafeo_storage::wal::WalRecord::SetNodeProperty {
                    id,
                    key: key.clone(),
                    value: value.clone(),
                });
            }
        }

        // Vector intents: buffer until commit, exactly like the row path.
        #[cfg(feature = "vector-index")]
        for (node, &id) in nodes.iter().zip(ids.iter()) {
            for (key, value) in &node.properties {
                if matches!(value, Value::Vector(_)) {
                    self.push_vector_intent(VectorIndexIntent::Upsert {
                        graph_name: graph_name.clone(),
                        node_id: id,
                        property: key.clone(),
                    })?;
                }
            }
        }

        Ok(ids)
    }

    /// Batch-creates edges inside the active transaction using one storage-level
    /// batch operation, returning the new IDs in input order.
    ///
    /// # Errors
    ///
    /// - if no explicit transaction is active;
    /// - if any property value exceeds the configured `max_property_size`
    ///   (checked for the whole batch before any mutation);
    /// - on a write-write conflict registering the batch with the transaction.
    #[cfg(feature = "lpg")]
    pub fn create_edges_with_props_transactional(
        &self,
        edges: &[TransactionalEdgeCreate],
    ) -> Result<Vec<EdgeId>> {
        let transaction_id = self.require_active_transaction()?;

        for edge in edges {
            for (key, value) in &edge.properties {
                self.check_property_size(key, value)?;
            }
        }

        let (epoch, _) = self.get_transaction_context();

        let storage_edges: Vec<BatchEdgeCreate<'_>> = edges
            .iter()
            .map(|e| BatchEdgeCreate {
                source: e.source,
                target: e.target,
                edge_type: e.edge_type.as_str(),
                properties: e
                    .properties
                    .iter()
                    .map(|(k, v)| (PropertyKey::new(k.clone()), v.clone()))
                    .collect(),
            })
            .collect();

        let store = self.active_lpg_store();
        let ids = store.create_edges_batch_versioned(&storage_edges, epoch, transaction_id);

        let entities: Vec<crate::transaction::EntityId> = ids
            .iter()
            .copied()
            .map(crate::transaction::EntityId::Edge)
            .collect();
        self.transaction_manager
            .record_writes(transaction_id, &entities)?;

        #[cfg(feature = "wal")]
        for (edge, &id) in edges.iter().zip(ids.iter()) {
            self.log_wal_record(&grafeo_storage::wal::WalRecord::CreateEdge {
                id,
                src: edge.source,
                dst: edge.target,
                edge_type: edge.edge_type.clone(),
            });
            for (key, value) in &edge.properties {
                self.log_wal_record(&grafeo_storage::wal::WalRecord::SetEdgeProperty {
                    id,
                    key: key.clone(),
                    value: value.clone(),
                });
            }
        }

        Ok(ids)
    }

    /// Returns the active transaction ID or fails closed if none is active.
    ///
    /// The batch APIs are explicit-transaction-required by contract; unlike the
    /// row methods they never fall back to `TransactionId::SYSTEM` auto-commit.
    #[cfg(feature = "lpg")]
    fn require_active_transaction(&self) -> Result<grafeo_common::types::TransactionId> {
        self.current_transaction.lock().ok_or_else(|| {
            Error::Transaction(TransactionError::InvalidState(
                "transactional batch mutation requires an active explicit transaction".to_string(),
            ))
        })
    }
}
