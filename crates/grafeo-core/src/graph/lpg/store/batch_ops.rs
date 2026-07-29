//! Storage-level transactional batch node/edge creation.
//!
//! This module defines the provider-neutral batch DTOs and the shared helpers
//! behind `Session::create_nodes_with_props_transactional` /
//! `create_edges_with_props_transactional`. The storage-backend-specific
//! `LpgStore` impls live in sibling modules to keep each file small:
//!
//! - [`batch_ops_lpg`] — non-tiered (`VersionChain`) store;
//! - [`batch_ops_tiered`] — tiered arena/`VersionIndex` store.
//!
//! Unlike the row-oriented `create_node_with_props_versioned` /
//! `create_edge_versioned` paths — which re-acquire every `RwLock` (nodes/edges
//! map, label registry, label index, node labels, adjacency, edge-type tables,
//! property column map) once per row — the batch entry points hoist each of
//! those locks to **once per batch** and allocate the ID range with a single
//! atomic. That is the difference between a genuine storage-level batch and a
//! loop wrapped in a batch-shaped API.
//!
//! Correctness contract (mirrors the row path exactly, so commit/rollback need
//! no batch-specific logic):
//! - every record is versioned under `transaction_id` with `EpochId::PENDING`
//!   visibility when a real transaction is active (`SYSTEM` otherwise), so the
//!   transaction-wide `finalize_version_epochs` / `discard_uncommitted_versions`
//!   scans make the whole batch visible on commit and remove it on rollback;
//! - label, property, edge-type and adjacency secondary structures are updated
//!   identically to the row path; `discard_uncommitted_versions` erases those
//!   secondary entries for entities whose version chains become empty;
//! - returned IDs preserve input order.
//!
//! These methods never touch the offline/unindexed bulk loaders.

use grafeo_common::types::{EdgeId, EpochId, NodeId, PropertyKey, TransactionId, Value};

/// A node to create within a storage-level batch.
///
/// Provider-neutral: the engine layer builds these from its public DTOs.
/// `properties` is owned so the batch can hoist the property-column lock.
pub struct BatchNodeCreate<'a> {
    /// Labels applied to the node.
    pub labels: &'a [&'a str],
    /// Property key/value pairs, written in order.
    pub properties: Vec<(PropertyKey, Value)>,
}

/// An edge to create within a storage-level batch.
pub struct BatchEdgeCreate<'a> {
    /// Source node.
    pub source: NodeId,
    /// Target node.
    pub target: NodeId,
    /// Edge type name.
    pub edge_type: &'a str,
    /// Property key/value pairs, written in order.
    pub properties: Vec<(PropertyKey, Value)>,
}

/// Visibility epoch for a batch row: `PENDING` under a real transaction so it
/// stays invisible until commit, the real epoch for `SYSTEM` (auto-commit).
pub(super) fn version_epoch_for(transaction_id: TransactionId, epoch: EpochId) -> EpochId {
    if transaction_id == TransactionId::SYSTEM {
        epoch
    } else {
        EpochId::PENDING
    }
}

/// Counts distinct property keys (matches `get_all().len()` used by the row
/// path to compute `props_count`). Only the non-tiered store writes
/// `props_count` back onto the record.
#[cfg(not(feature = "tiered-storage"))]
pub(super) fn distinct_key_count(properties: &[(PropertyKey, Value)]) -> usize {
    use grafeo_common::utils::hash::FxHashSet;
    let mut seen = FxHashSet::default();
    for (key, _) in properties {
        seen.insert(key.clone());
    }
    seen.len()
}

/// Flattens node batch inputs into `(node_id, key, value)` property rows in
/// input order, for the hoisted property-column write.
pub(super) fn build_node_prop_rows(
    nodes: &[BatchNodeCreate<'_>],
    ids: &[NodeId],
) -> Vec<(NodeId, PropertyKey, Value)> {
    let total: usize = nodes.iter().map(|n| n.properties.len()).sum();
    let mut rows = Vec::with_capacity(total);
    for (node, &id) in nodes.iter().zip(ids.iter()) {
        for (key, value) in &node.properties {
            rows.push((id, key.clone(), value.clone()));
        }
    }
    rows
}

/// Flattens edge batch inputs into `(edge_id, key, value)` property rows in
/// input order, for the hoisted property-column write.
pub(super) fn build_edge_prop_rows(
    edges: &[BatchEdgeCreate<'_>],
    ids: &[EdgeId],
) -> Vec<(EdgeId, PropertyKey, Value)> {
    let total: usize = edges.iter().map(|e| e.properties.len()).sum();
    let mut rows = Vec::with_capacity(total);
    for (edge, &id) in edges.iter().zip(ids.iter()) {
        for (key, value) in &edge.properties {
            rows.push((id, key.clone(), value.clone()));
        }
    }
    rows
}
