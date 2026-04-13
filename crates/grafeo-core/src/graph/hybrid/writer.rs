//! GraphStoreMut implementation for HybridStore (routed writes).
//!
//! All writes delegate to the overlay `LpgStore`. For compact entity structural
//! mutations (delete, label change), we use lazy shadow records: a `NodeRecord`
//! is inserted into the overlay so the MVCC machinery can track the entity, and
//! labels are copied from the compact store so label queries continue to work.

use grafeo_common::mvcc::VersionChain;
use grafeo_common::types::{EdgeId, EpochId, NodeId, TransactionId, Value};

use super::EntityId;
use super::HybridStore;
use crate::graph::lpg::NodeRecord;
use crate::graph::traits::{GraphStore, GraphStoreMut};
use crate::graph::Direction;

impl HybridStore {
    /// Ensures a shadow `NodeRecord` exists in the overlay for a compact node.
    ///
    /// If the overlay already contains a record for `id`, this is a no-op.
    /// Otherwise, creates a new `NodeRecord` + `VersionChain` in the overlay
    /// and copies labels from the compact store so label queries still work.
    fn ensure_shadow(&self, id: NodeId) {
        // Already has a shadow (or was created in overlay) — nothing to do.
        if GraphStore::get_node(self.overlay.as_ref(), id).is_some() {
            return;
        }

        let epoch = self.overlay.current_epoch();
        let record = NodeRecord::new(id, epoch);
        let chain = VersionChain::with_initial(record, epoch, TransactionId::SYSTEM);
        self.overlay.insert_shadow_node(id, chain);

        // Copy labels from compact so label queries still work after shadowing.
        if let Some(node) = self.compact.read().get_node(id) {
            for label in &node.labels {
                self.overlay.add_label(id, label.as_str());
            }
        }
    }

    /// Ensures a shadow edge record exists in the overlay for a compact edge.
    ///
    /// For edges, deletion only needs the deleted_edges set (no MVCC chain
    /// needed because the HybridStore reader already checks deleted_edges).
    /// This is intentionally a no-op placeholder for symmetry; compact edge
    /// deletion is handled directly via the deleted_edges set.
    fn ensure_shadow_edge(&self, _id: EdgeId) {
        // Compact edge deletion is tracked purely via `deleted_edges` set.
        // No overlay shadow needed because the HybridStore reader checks
        // `is_edge_deleted()` before accessing either store.
    }
}

impl GraphStoreMut for HybridStore {
    // ---------------------------------------------------------------
    // Node creation
    // ---------------------------------------------------------------

    fn create_node(&self, labels: &[&str]) -> NodeId {
        self.overlay.create_node(labels)
    }

    fn create_node_versioned(
        &self,
        labels: &[&str],
        epoch: EpochId,
        transaction_id: TransactionId,
    ) -> NodeId {
        self.overlay
            .create_node_versioned(labels, epoch, transaction_id)
    }

    // ---------------------------------------------------------------
    // Edge creation
    // ---------------------------------------------------------------

    fn create_edge(&self, src: NodeId, dst: NodeId, edge_type: &str) -> EdgeId {
        self.overlay.create_edge(src, dst, edge_type)
    }

    fn create_edge_versioned(
        &self,
        src: NodeId,
        dst: NodeId,
        edge_type: &str,
        epoch: EpochId,
        transaction_id: TransactionId,
    ) -> EdgeId {
        self.overlay
            .create_edge_versioned(src, dst, edge_type, epoch, transaction_id)
    }

    fn batch_create_edges(&self, edges: &[(NodeId, NodeId, &str)]) -> Vec<EdgeId> {
        self.overlay.batch_create_edges(edges)
    }

    // ---------------------------------------------------------------
    // Deletion
    // ---------------------------------------------------------------

    fn delete_node(&self, id: NodeId) -> bool {
        if self.is_compact_node_id(id) {
            if self.is_node_deleted(id) {
                return false;
            }
            // Verify the node actually exists in compact
            if self.compact.read().get_node(id).is_none() {
                return false;
            }
            self.mark_node_dirty(id, None);
            self.ensure_shadow(id);
            self.overlay.delete_node(id);
            self.deleted_nodes.write().insert(id);
            true
        } else {
            self.overlay.delete_node(id)
        }
    }

    fn delete_node_versioned(
        &self,
        id: NodeId,
        epoch: EpochId,
        transaction_id: TransactionId,
    ) -> bool {
        if self.is_compact_node_id(id) {
            if self.is_node_deleted(id) {
                return false;
            }
            if self.compact.read().get_node(id).is_none() {
                return false;
            }
            self.mark_node_dirty(id, None);
            self.ensure_shadow(id);
            if transaction_id == TransactionId::SYSTEM {
                self.overlay.delete_node(id);
            } else {
                GraphStoreMut::delete_node_versioned(
                    self.overlay.as_ref(),
                    id,
                    epoch,
                    transaction_id,
                );
            }
            self.deleted_nodes.write().insert(id);

            // Cascade compact edges — collect and mark deleted (one lock at a time)
            let cascade_edges: Vec<EdgeId> = {
                let compact = self.compact.read();
                let mut deleted_edges = self.deleted_edges.write();
                compact
                    .edges_from(id, Direction::Both)
                    .into_iter()
                    .filter(|(_, eid)| deleted_edges.insert(*eid))
                    .map(|(_, eid)| eid)
                    .collect()
            };

            // Track node + cascaded edges for rollback (separate lock acquisition)
            {
                let mut pending = self.pending_deletions.write();
                let entries = pending.entry(transaction_id).or_default();
                entries.push(EntityId::Node(id));
                entries.extend(cascade_edges.into_iter().map(EntityId::Edge));
            }

            // Cascade overlay edges
            self.overlay.delete_node_edges(id);
            true
        } else if transaction_id == TransactionId::SYSTEM {
            self.overlay.delete_node(id)
        } else {
            GraphStoreMut::delete_node_versioned(
                self.overlay.as_ref(),
                id,
                epoch,
                transaction_id,
            )
        }
    }

    fn delete_node_edges(&self, node_id: NodeId) {
        // Non-versioned: no rollback tracking (matches non-versioned delete_node)
        if self.is_compact_node_id(node_id) {
            let compact = self.compact.read();
            let mut deleted_edges = self.deleted_edges.write();
            for (_, eid) in compact.edges_from(node_id, Direction::Both) {
                deleted_edges.insert(eid);
            }
        }
        self.overlay.delete_node_edges(node_id);
    }

    fn delete_edge(&self, id: EdgeId) -> bool {
        if self.is_compact_edge_id(id) {
            if self.is_edge_deleted(id) {
                return false;
            }
            // Verify the edge actually exists in compact
            if self.compact.read().get_edge(id).is_none() {
                return false;
            }
            self.ensure_shadow_edge(id);
            self.deleted_edges.write().insert(id);
            true
        } else {
            self.overlay.delete_edge(id)
        }
    }

    fn delete_edge_versioned(
        &self,
        id: EdgeId,
        epoch: EpochId,
        transaction_id: TransactionId,
    ) -> bool {
        if self.is_compact_edge_id(id) {
            if self.is_edge_deleted(id) {
                return false;
            }
            if self.compact.read().get_edge(id).is_none() {
                return false;
            }
            self.ensure_shadow_edge(id);
            self.deleted_edges.write().insert(id);
            // Lock dropped before acquiring pending_deletions
            self.pending_deletions.write()
                .entry(transaction_id)
                .or_default()
                .push(EntityId::Edge(id));
            true
        } else if transaction_id == TransactionId::SYSTEM {
            self.overlay.delete_edge(id)
        } else {
            GraphStoreMut::delete_edge_versioned(
                self.overlay.as_ref(),
                id,
                epoch,
                transaction_id,
            )
        }
    }

    // ---------------------------------------------------------------
    // Property mutation
    // ---------------------------------------------------------------

    fn set_node_property(&self, id: NodeId, key: &str, value: Value) {
        if self.is_compact_node_id(id) {
            self.mark_node_dirty(id, Some(key));
        }
        self.overlay.set_node_property(id, key, value);
    }

    fn set_edge_property(&self, id: EdgeId, key: &str, value: Value) {
        if self.is_compact_edge_id(id) {
            self.mark_edge_dirty(id, Some(key));
        }
        self.overlay.set_edge_property(id, key, value);
    }

    fn set_node_property_versioned(
        &self,
        id: NodeId,
        key: &str,
        value: Value,
        transaction_id: TransactionId,
    ) {
        if self.is_compact_node_id(id) {
            self.mark_node_dirty(id, Some(key));
        }
        self.overlay
            .set_node_property_versioned(id, key, value, transaction_id);
    }

    fn set_edge_property_versioned(
        &self,
        id: EdgeId,
        key: &str,
        value: Value,
        transaction_id: TransactionId,
    ) {
        if self.is_compact_edge_id(id) {
            self.mark_edge_dirty(id, Some(key));
        }
        self.overlay
            .set_edge_property_versioned(id, key, value, transaction_id);
    }

    fn remove_node_property(&self, id: NodeId, key: &str) -> Option<Value> {
        if self.is_compact_node_id(id) {
            self.mark_node_dirty(id, Some(key));
            let old = self.compact.read().get_node_property(id, &key.into());
            self.overlay.set_node_property(id, key, Value::Null);
            old
        } else {
            self.overlay.remove_node_property(id, key)
        }
    }

    fn remove_edge_property(&self, id: EdgeId, key: &str) -> Option<Value> {
        if self.is_compact_edge_id(id) {
            self.mark_edge_dirty(id, Some(key));
            let old = self.compact.read().get_edge_property(id, &key.into());
            self.overlay.set_edge_property(id, key, Value::Null);
            old
        } else {
            self.overlay.remove_edge_property(id, key)
        }
    }

    fn remove_node_property_versioned(
        &self,
        id: NodeId,
        key: &str,
        transaction_id: TransactionId,
    ) -> Option<Value> {
        if self.is_compact_node_id(id) {
            let old = self.compact.read().get_node_property(id, &key.into());
            self.overlay
                .set_node_property_versioned(id, key, Value::Null, transaction_id);
            old
        } else {
            self.overlay
                .remove_node_property_versioned(id, key, transaction_id)
        }
    }

    fn remove_edge_property_versioned(
        &self,
        id: EdgeId,
        key: &str,
        transaction_id: TransactionId,
    ) -> Option<Value> {
        if self.is_compact_edge_id(id) {
            let old = self.compact.read().get_edge_property(id, &key.into());
            self.overlay
                .set_edge_property_versioned(id, key, Value::Null, transaction_id);
            old
        } else {
            self.overlay
                .remove_edge_property_versioned(id, key, transaction_id)
        }
    }

    // ---------------------------------------------------------------
    // Label mutation
    // ---------------------------------------------------------------

    fn add_label(&self, node_id: NodeId, label: &str) -> bool {
        if self.is_compact_node_id(node_id) {
            self.mark_node_dirty(node_id, None);
            self.ensure_shadow(node_id);
        }
        self.overlay.add_label(node_id, label)
    }

    fn remove_label(&self, node_id: NodeId, label: &str) -> bool {
        if self.is_compact_node_id(node_id) {
            self.mark_node_dirty(node_id, None);
            self.ensure_shadow(node_id);
        }
        self.overlay.remove_label(node_id, label)
    }

    fn add_label_versioned(
        &self,
        node_id: NodeId,
        label: &str,
        transaction_id: TransactionId,
    ) -> bool {
        if self.is_compact_node_id(node_id) {
            self.mark_node_dirty(node_id, None);
            self.ensure_shadow(node_id);
        }
        self.overlay
            .add_label_versioned(node_id, label, transaction_id)
    }

    fn remove_label_versioned(
        &self,
        node_id: NodeId,
        label: &str,
        transaction_id: TransactionId,
    ) -> bool {
        if self.is_compact_node_id(node_id) {
            self.mark_node_dirty(node_id, None);
            self.ensure_shadow(node_id);
        }
        self.overlay
            .remove_label_versioned(node_id, label, transaction_id)
    }
}
