//! [`GraphStoreMut`] trait implementation for [`CompactStore`].
//!
//! All write operations (node/edge creation, deletion, property mutation,
//! and label changes) are delegated to the [`DeltaBuffer`](super::delta::DeltaBuffer).
//! The snapshot data is never modified — mutations are layered on top.

use grafeo_common::types::{EdgeId, EpochId, NodeId, PropertyKey, TransactionId, Value};

use super::CompactStore;
use crate::graph::Direction;
use crate::graph::traits::{GraphStore, GraphStoreMut};

use std::sync::atomic::Ordering as AtomicOrdering;

impl GraphStoreMut for CompactStore {
    fn create_node(&self, labels: &[&str]) -> NodeId {
        self.delta_has_content.store(true, AtomicOrdering::Relaxed);
        self.delta.lock().create_node(labels)
    }

    fn create_node_versioned(
        &self,
        labels: &[&str],
        _epoch: EpochId,
        _transaction_id: TransactionId,
    ) -> NodeId {
        self.create_node(labels)
    }

    fn create_edge(&self, src: NodeId, dst: NodeId, edge_type: &str) -> EdgeId {
        self.delta_has_content.store(true, AtomicOrdering::Relaxed);
        self.delta.lock().create_edge(src, dst, edge_type)
    }

    fn create_edge_versioned(
        &self,
        src: NodeId,
        dst: NodeId,
        edge_type: &str,
        _epoch: EpochId,
        _transaction_id: TransactionId,
    ) -> EdgeId {
        self.create_edge(src, dst, edge_type)
    }

    fn batch_create_edges(&self, edges: &[(NodeId, NodeId, &str)]) -> Vec<EdgeId> {
        self.delta_has_content.store(true, AtomicOrdering::Relaxed);
        let mut delta = self.delta.lock();
        edges
            .iter()
            .map(|(src, dst, edge_type)| delta.create_edge(*src, *dst, edge_type))
            .collect()
    }

    fn delete_node(&self, id: NodeId) -> bool {
        self.delta_has_content.store(true, AtomicOrdering::Relaxed);
        // First delete all edges incident to this node.
        // Collect edges before locking delta, since edges_from also locks it.
        let edges = self.edges_from(id, Direction::Both);
        let mut delta = self.delta.lock();
        for (_, eid) in edges {
            delta.delete_edge(eid);
        }
        delta.delete_node(id)
    }

    fn delete_node_versioned(
        &self,
        id: NodeId,
        _epoch: EpochId,
        _transaction_id: TransactionId,
    ) -> bool {
        self.delete_node(id)
    }

    fn delete_node_edges(&self, node_id: NodeId) {
        self.delta_has_content.store(true, AtomicOrdering::Relaxed);
        // Collect edges before locking delta, since edges_from also locks it.
        let edges = self.edges_from(node_id, Direction::Both);
        let mut delta = self.delta.lock();
        for (_, eid) in edges {
            delta.delete_edge(eid);
        }
    }

    fn delete_edge(&self, id: EdgeId) -> bool {
        self.delta_has_content.store(true, AtomicOrdering::Relaxed);
        self.delta.lock().delete_edge(id)
    }

    fn delete_edge_versioned(
        &self,
        id: EdgeId,
        _epoch: EpochId,
        _transaction_id: TransactionId,
    ) -> bool {
        self.delete_edge(id)
    }

    fn set_node_property(&self, id: NodeId, key: &str, value: Value) {
        self.delta_has_content.store(true, AtomicOrdering::Relaxed);
        self.delta.lock().set_node_property(id, key, value);
    }

    fn set_edge_property(&self, id: EdgeId, key: &str, value: Value) {
        self.delta_has_content.store(true, AtomicOrdering::Relaxed);
        self.delta.lock().set_edge_property(id, key, value);
    }

    fn remove_node_property(&self, id: NodeId, key: &str) -> Option<Value> {
        self.delta_has_content.store(true, AtomicOrdering::Relaxed);
        // Read the current visible value before tombstoning, since the trait
        // contract requires returning the previous value.
        let pk = PropertyKey::from(key);
        let current = self.get_node_property(id, &pk);
        let prev_override = self.delta.lock().remove_node_property(id, key);
        prev_override.or(current)
    }

    fn remove_edge_property(&self, id: EdgeId, key: &str) -> Option<Value> {
        self.delta_has_content.store(true, AtomicOrdering::Relaxed);
        let pk = PropertyKey::from(key);
        let current = self.get_edge_property(id, &pk);
        let prev_override = self.delta.lock().remove_edge_property(id, key);
        prev_override.or(current)
    }

    fn add_label(&self, node_id: NodeId, label: &str) -> bool {
        // Check if the node already has the label before adding.
        let has_label = self.get_node(node_id).is_some_and(|n| n.has_label(label));
        if has_label {
            return false;
        }
        self.delta_has_content.store(true, AtomicOrdering::Relaxed);
        self.delta.lock().add_label(node_id, label);
        true
    }

    fn remove_label(&self, node_id: NodeId, label: &str) -> bool {
        // Check if the node has the label before removing.
        let has_label = self.get_node(node_id).is_some_and(|n| n.has_label(label));
        if !has_label {
            return false;
        }
        self.delta_has_content.store(true, AtomicOrdering::Relaxed);
        self.delta.lock().remove_label(node_id, label);
        true
    }
}
