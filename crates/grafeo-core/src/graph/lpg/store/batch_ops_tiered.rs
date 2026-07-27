//! Tiered-storage implementation of the storage-level batch.
//!
//! Split from [`super::batch_ops`] to keep each file under the ~400 LOC budget.
//! The tiered store keeps records in a per-epoch arena and metadata in a version
//! index. Arena allocation is inherently per-record, but the version-index write
//! lock is still hoisted to once per batch (vs once per row in the row path).
//! Shares the DTOs and helper functions defined in `batch_ops.rs`.

#![cfg(feature = "tiered-storage")]

use super::LpgStore;
use super::batch_ops::{
    BatchEdgeCreate, BatchNodeCreate, build_edge_prop_rows, build_node_prop_rows, version_epoch_for,
};
use crate::graph::lpg::{EdgeRecord, NodeRecord};
use arcstr::ArcStr;
use grafeo_common::mvcc::{HotVersionRef, VersionIndex};
#[cfg(feature = "temporal")]
use grafeo_common::temporal::VersionLog;
use grafeo_common::types::{
    EdgeId, EpochId, HashableValue, NodeId, PropertyKey, TransactionId, Value,
};
use grafeo_common::utils::hash::FxHashSet;
use std::sync::atomic::Ordering;

impl LpgStore {
    /// Tiered-storage variant of
    /// [`create_nodes_batch_versioned`](LpgStore::create_nodes_batch_versioned).
    pub fn create_nodes_batch_versioned(
        &self,
        nodes: &[BatchNodeCreate<'_>],
        epoch: EpochId,
        transaction_id: TransactionId,
    ) -> Vec<NodeId> {
        let count = nodes.len();
        if count == 0 {
            return Vec::new();
        }
        let version_epoch = version_epoch_for(transaction_id, epoch);
        let base = self.next_node_id.fetch_add(count as u64, Ordering::Relaxed);
        let ids: Vec<NodeId> = (base..base + count as u64).map(NodeId::new).collect();

        let per_node_label_ids = self.resolve_label_ids_batch(nodes);
        {
            let mut index = self.label_index.write();
            for (label_ids, &id) in per_node_label_ids.iter().zip(ids.iter()) {
                for &label_id in label_ids {
                    if index.len() <= label_id as usize {
                        index.resize_with(label_id as usize + 1, Default::default);
                    }
                    index[label_id as usize].insert(id, ());
                }
            }
        }
        {
            let mut node_labels = self.node_labels.write();
            for (label_ids, &id) in per_node_label_ids.iter().zip(ids.iter()) {
                let set: FxHashSet<u32> = label_ids.iter().copied().collect();
                #[cfg(not(feature = "temporal"))]
                node_labels.insert(id, set);
                #[cfg(feature = "temporal")]
                node_labels.insert(id, VersionLog::with_value(version_epoch, set));
            }
        }

        // Allocate records in the epoch arena, then insert all version refs
        // under one version-index write lock.
        let arena = self
            .arena_allocator
            .arena_or_create(epoch)
            .expect("failed to create arena for epoch");
        let mut versions = self.node_versions.write();
        for (node, &id) in nodes.iter().zip(ids.iter()) {
            let mut record = NodeRecord::new(id, epoch);
            #[allow(clippy::cast_possible_truncation)]
            record.set_label_count(node.labels.len() as u16);
            let (offset, _stored) = arena
                .alloc_value_with_offset(record)
                .expect("arena allocation failed for node record");
            let hot_ref = HotVersionRef::new(version_epoch, epoch, offset, transaction_id);
            if let Some(index) = versions.get_mut(&id) {
                index.add_hot(hot_ref);
            } else {
                versions.insert(id, VersionIndex::with_initial(hot_ref));
            }
        }
        drop(versions);
        // reason: batch size bounded by practical graph limits, fits i64
        #[allow(clippy::cast_possible_wrap)]
        let count_i64 = count as i64;
        self.live_node_count.fetch_add(count_i64, Ordering::Relaxed);

        let prop_rows = build_node_prop_rows(nodes, &ids);
        self.apply_node_property_index_batch(&prop_rows);
        #[cfg(not(feature = "temporal"))]
        self.node_properties.set_batch(prop_rows);
        #[cfg(feature = "temporal")]
        self.node_properties.set_batch(prop_rows, epoch);

        ids
    }

    /// Tiered-storage variant of
    /// [`create_edges_batch_versioned`](LpgStore::create_edges_batch_versioned).
    pub fn create_edges_batch_versioned(
        &self,
        edges: &[BatchEdgeCreate<'_>],
        epoch: EpochId,
        transaction_id: TransactionId,
    ) -> Vec<EdgeId> {
        let count = edges.len();
        if count == 0 {
            return Vec::new();
        }
        let version_epoch = version_epoch_for(transaction_id, epoch);
        let base = self.next_edge_id.fetch_add(count as u64, Ordering::Relaxed);
        let ids: Vec<EdgeId> = (base..base + count as u64).map(EdgeId::new).collect();

        let type_ids = self.resolve_edge_type_ids_and_count_batch(edges);

        let arena = self
            .arena_allocator
            .arena_or_create(epoch)
            .expect("failed to create arena for epoch");
        let mut versions = self.edge_versions.write();
        for ((edge, &type_id), &id) in edges.iter().zip(type_ids.iter()).zip(ids.iter()) {
            let record = EdgeRecord::new(id, edge.source, edge.target, type_id, epoch);
            let (offset, _stored) = arena
                .alloc_value_with_offset(record)
                .expect("arena allocation failed for edge record");
            let hot_ref = HotVersionRef::new(version_epoch, epoch, offset, transaction_id);
            if let Some(index) = versions.get_mut(&id) {
                index.add_hot(hot_ref);
            } else {
                versions.insert(id, VersionIndex::with_initial(hot_ref));
            }
        }
        drop(versions);

        let adj_rows: Vec<(NodeId, NodeId, EdgeId)> = edges
            .iter()
            .zip(ids.iter())
            .map(|(e, &id)| (e.source, e.target, id))
            .collect();
        self.forward_adj.batch_add_edges(&adj_rows);
        if let Some(ref backward) = self.backward_adj {
            let back_rows: Vec<(NodeId, NodeId, EdgeId)> =
                adj_rows.iter().map(|&(s, d, id)| (d, s, id)).collect();
            backward.batch_add_edges(&back_rows);
        }
        // reason: batch size bounded by practical graph limits, fits i64
        #[allow(clippy::cast_possible_wrap)]
        let count_i64 = count as i64;
        self.live_edge_count.fetch_add(count_i64, Ordering::Relaxed);

        let prop_rows = build_edge_prop_rows(edges, &ids);
        #[cfg(not(feature = "temporal"))]
        self.edge_properties.set_batch(prop_rows);
        #[cfg(feature = "temporal")]
        self.edge_properties.set_batch(prop_rows, epoch);

        ids
    }

    fn resolve_label_ids_batch(&self, nodes: &[BatchNodeCreate<'_>]) -> Vec<Vec<u32>> {
        let mut per_node = Vec::with_capacity(nodes.len());
        let mut registry = self.label_registry.write();
        for node in nodes {
            let mut ids = Vec::with_capacity(node.labels.len());
            for label in node.labels {
                ids.push(registry.get_or_create(label));
            }
            per_node.push(ids);
        }
        per_node
    }

    fn apply_node_property_index_batch(&self, rows: &[(NodeId, PropertyKey, Value)]) {
        let indexes = self.property_indexes.read();
        if indexes.is_empty() {
            return;
        }
        for (id, key, value) in rows {
            if let Some(index) = indexes.get(key) {
                let hv = HashableValue::new(value.clone());
                index.entry(hv).or_default().insert(*id);
            }
        }
    }

    fn resolve_edge_type_ids_and_count_batch(&self, edges: &[BatchEdgeCreate<'_>]) -> Vec<u32> {
        let mut type_ids = Vec::with_capacity(edges.len());
        let mut type_to_id = self.edge_type_to_id.write();
        let mut id_to_type = self.id_to_edge_type.write();
        let mut counts = self.edge_type_live_counts.write();
        for edge in edges {
            let id = if let Some(&id) = type_to_id.get(edge.edge_type) {
                id
            } else {
                // reason: edge type registry size bounded, fits u32
                #[allow(clippy::cast_possible_truncation)]
                let id = id_to_type.len() as u32;
                let t: ArcStr = edge.edge_type.into();
                type_to_id.insert(t.clone(), id);
                id_to_type.push(t);
                while counts.len() <= id as usize {
                    counts.push(0);
                }
                id
            };
            type_ids.push(id);
        }
        for &type_id in &type_ids {
            counts[type_id as usize] += 1;
        }
        type_ids
    }
}
