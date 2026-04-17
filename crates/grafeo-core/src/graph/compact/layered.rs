//! Two-layer graph store: read-only columnar base + mutable LPG overlay.
//!
//! `LayeredStore` coordinates reads between a [`CompactStore`](crate::graph::compact::CompactStore) (cold, columnar)
//! and an [`LpgStore`](crate::graph::lpg::LpgStore) (hot, HashMap-based). All writes go to the overlay.
//! Reads check the overlay first and fall through to the compact base for
//! unmodified entities.
//!
//! Requires both `compact-store` and `lpg` features.

use std::sync::Arc;

use arcstr::ArcStr;
use grafeo_common::types::{EdgeId, EpochId, NodeId, PropertyKey, TransactionId, Value};
use grafeo_common::utils::hash::{FxHashMap, FxHashSet};
use parking_lot::RwLock;

use super::CompactStore;
use crate::graph::Direction;
use crate::graph::lpg::{CompareOp, Edge, LpgStore, Node};
use crate::graph::traits::{GraphStore, GraphStoreMut};
use crate::statistics::Statistics;

/// A two-layer graph store with a columnar base and mutable overlay.
///
/// The compact base serves cold reads (immutable, columnar). The LPG overlay
/// captures all mutations. Reads check the overlay first: if an entity is in
/// `dirty_node_ids` or `dirty_edge_ids`, the overlay is authoritative. If an
/// entity is in `deleted_from_base_nodes` or `deleted_from_base_edges`, it has
/// been deleted and returns `None`. Otherwise, the base is queried.
pub struct LayeredStore {
    /// Read-only columnar base (cold data).
    base: Arc<CompactStore>,
    /// Mutable overlay for new and modified data.
    overlay: Arc<LpgStore>,
    /// Node IDs modified or created in the overlay.
    dirty_node_ids: RwLock<FxHashSet<NodeId>>,
    /// Edge IDs modified or created in the overlay.
    dirty_edge_ids: RwLock<FxHashSet<EdgeId>>,
    /// Base node IDs that have been deleted.
    deleted_from_base_nodes: RwLock<FxHashSet<NodeId>>,
    /// Base edge IDs that have been deleted.
    deleted_from_base_edges: RwLock<FxHashSet<EdgeId>>,
}

impl std::fmt::Debug for LayeredStore {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LayeredStore")
            .field("base_node_count", &self.base.node_count())
            .field("overlay_node_count", &self.overlay.node_count())
            .field("dirty_nodes", &self.dirty_node_ids.read().len())
            .field(
                "deleted_base_nodes",
                &self.deleted_from_base_nodes.read().len(),
            )
            .finish_non_exhaustive()
    }
}

impl LayeredStore {
    /// Creates a layered store from a compact base.
    ///
    /// The `max_node_id` and `max_edge_id` values seed the overlay's ID
    /// allocator so new entities never collide with base IDs.
    ///
    /// # Errors
    ///
    /// Returns an error if the overlay `LpgStore` cannot be created.
    pub fn new(
        base: CompactStore,
        max_node_id: u64,
        max_edge_id: u64,
    ) -> Result<Self, grafeo_common::memory::AllocError> {
        let overlay = Arc::new(LpgStore::new()?);
        overlay.set_next_node_id(max_node_id + 1);
        overlay.set_next_edge_id(max_edge_id + 1);

        Ok(Self {
            base: Arc::new(base),
            overlay,
            dirty_node_ids: RwLock::new(FxHashSet::default()),
            dirty_edge_ids: RwLock::new(FxHashSet::default()),
            deleted_from_base_nodes: RwLock::new(FxHashSet::default()),
            deleted_from_base_edges: RwLock::new(FxHashSet::default()),
        })
    }

    /// Returns a reference to the compact base store.
    #[must_use]
    pub fn base_store(&self) -> &CompactStore {
        &self.base
    }

    /// Returns a shared reference to the compact base store.
    #[must_use]
    pub fn base_store_arc(&self) -> Arc<CompactStore> {
        Arc::clone(&self.base)
    }

    /// Returns a reference to the overlay LPG store.
    #[must_use]
    pub fn overlay_store(&self) -> &Arc<LpgStore> {
        &self.overlay
    }

    /// Number of dirty (modified/created) entities in the overlay.
    #[must_use]
    pub fn overlay_mutation_count(&self) -> usize {
        self.dirty_node_ids.read().len()
            + self.dirty_edge_ids.read().len()
            + self.deleted_from_base_nodes.read().len()
            + self.deleted_from_base_edges.read().len()
    }

    /// Approximate heap memory of both layers.
    #[must_use]
    pub fn memory_bytes(&self) -> usize {
        let (store_mem, index_mem, mvcc_mem, pool_mem) = self.overlay.memory_breakdown();
        self.base.memory_bytes()
            + store_mem.total_bytes
            + index_mem.total_bytes
            + mvcc_mem.total_bytes
            + pool_mem.total_bytes
    }

    /// Checks whether a node ID is in the overlay (dirty or deleted).
    #[inline]
    fn is_node_dirty(&self, id: NodeId) -> bool {
        self.dirty_node_ids.read().contains(&id)
    }

    /// Checks whether a node was deleted from the base.
    #[inline]
    fn is_node_deleted_from_base(&self, id: NodeId) -> bool {
        self.deleted_from_base_nodes.read().contains(&id)
    }

    /// Checks whether an edge ID is in the overlay (dirty or deleted).
    #[inline]
    fn is_edge_dirty(&self, id: EdgeId) -> bool {
        self.dirty_edge_ids.read().contains(&id)
    }

    /// Checks whether an edge was deleted from the base.
    #[inline]
    fn is_edge_deleted_from_base(&self, id: EdgeId) -> bool {
        self.deleted_from_base_edges.read().contains(&id)
    }
}

// ── GraphStore implementation ──────────────────────────────────────

impl GraphStore for LayeredStore {
    fn get_node(&self, id: NodeId) -> Option<Node> {
        if self.is_node_deleted_from_base(id) {
            return None;
        }
        if self.is_node_dirty(id) {
            return self.overlay.get_node(id);
        }
        self.base.get_node(id)
    }

    fn get_edge(&self, id: EdgeId) -> Option<Edge> {
        if self.is_edge_deleted_from_base(id) {
            return None;
        }
        if self.is_edge_dirty(id) {
            return self.overlay.get_edge(id);
        }
        self.base.get_edge(id)
    }

    fn get_node_versioned(
        &self,
        id: NodeId,
        epoch: EpochId,
        transaction_id: TransactionId,
    ) -> Option<Node> {
        if self.is_node_deleted_from_base(id) {
            return None;
        }
        if self.is_node_dirty(id) {
            return self.overlay.get_node_versioned(id, epoch, transaction_id);
        }
        self.base.get_node(id)
    }

    fn get_edge_versioned(
        &self,
        id: EdgeId,
        epoch: EpochId,
        transaction_id: TransactionId,
    ) -> Option<Edge> {
        if self.is_edge_deleted_from_base(id) {
            return None;
        }
        if self.is_edge_dirty(id) {
            return self.overlay.get_edge_versioned(id, epoch, transaction_id);
        }
        self.base.get_edge(id)
    }

    fn get_node_at_epoch(&self, id: NodeId, epoch: EpochId) -> Option<Node> {
        if self.is_node_deleted_from_base(id) {
            return None;
        }
        if self.is_node_dirty(id) {
            return self.overlay.get_node_at_epoch(id, epoch);
        }
        self.base.get_node(id)
    }

    fn get_edge_at_epoch(&self, id: EdgeId, epoch: EpochId) -> Option<Edge> {
        if self.is_edge_deleted_from_base(id) {
            return None;
        }
        if self.is_edge_dirty(id) {
            return self.overlay.get_edge_at_epoch(id, epoch);
        }
        self.base.get_edge(id)
    }

    fn get_node_property(&self, id: NodeId, key: &PropertyKey) -> Option<Value> {
        if self.is_node_deleted_from_base(id) {
            return None;
        }
        if self.is_node_dirty(id) {
            return self.overlay.get_node_property(id, key);
        }
        self.base.get_node_property(id, key)
    }

    fn get_edge_property(&self, id: EdgeId, key: &PropertyKey) -> Option<Value> {
        if self.is_edge_deleted_from_base(id) {
            return None;
        }
        if self.is_edge_dirty(id) {
            return self.overlay.get_edge_property(id, key);
        }
        self.base.get_edge_property(id, key)
    }

    fn get_node_property_batch(&self, ids: &[NodeId], key: &PropertyKey) -> Vec<Option<Value>> {
        ids.iter()
            .map(|id| self.get_node_property(*id, key))
            .collect()
    }

    fn get_nodes_properties_batch(&self, ids: &[NodeId]) -> Vec<FxHashMap<PropertyKey, Value>> {
        ids.iter()
            .map(|id| {
                self.get_node(*id)
                    .map(|n| {
                        n.properties
                            .iter()
                            .map(|(k, v)| (k.clone(), v.clone()))
                            .collect()
                    })
                    .unwrap_or_default()
            })
            .collect()
    }

    fn get_nodes_properties_selective_batch(
        &self,
        ids: &[NodeId],
        keys: &[PropertyKey],
    ) -> Vec<FxHashMap<PropertyKey, Value>> {
        ids.iter()
            .map(|id| {
                let mut map = FxHashMap::default();
                for key in keys {
                    if let Some(v) = self.get_node_property(*id, key) {
                        map.insert(key.clone(), v);
                    }
                }
                map
            })
            .collect()
    }

    fn get_edges_properties_selective_batch(
        &self,
        ids: &[EdgeId],
        keys: &[PropertyKey],
    ) -> Vec<FxHashMap<PropertyKey, Value>> {
        ids.iter()
            .map(|id| {
                let mut map = FxHashMap::default();
                for key in keys {
                    if let Some(v) = self.get_edge_property(*id, key) {
                        map.insert(key.clone(), v);
                    }
                }
                map
            })
            .collect()
    }

    fn neighbors(&self, node: NodeId, direction: Direction) -> Vec<NodeId> {
        let deleted_nodes = self.deleted_from_base_nodes.read();

        let mut results = Vec::new();

        // Base neighbors (minus deleted).
        if !deleted_nodes.contains(&node) && !self.is_node_dirty(node) {
            for nid in self.base.neighbors(node, direction) {
                if !deleted_nodes.contains(&nid) {
                    results.push(nid);
                }
            }
        }

        // Overlay neighbors (for dirty source node or overlay-only node).
        if self.is_node_dirty(node) || self.overlay.get_node(node).is_some() {
            for nid in self.overlay.neighbors(node, direction) {
                if !deleted_nodes.contains(&nid) {
                    results.push(nid);
                }
            }
        }

        // Overlay edges that connect base nodes.
        // If node is in the base and the overlay has edges for it
        // (e.g., after promote), those are already in dirty.

        results.sort_unstable();
        results.dedup();
        results
    }

    fn edges_from(&self, node: NodeId, direction: Direction) -> Vec<(NodeId, EdgeId)> {
        let deleted_nodes = self.deleted_from_base_nodes.read();
        let deleted_edges = self.deleted_from_base_edges.read();

        let mut results = Vec::new();

        // Base edges (minus deleted).
        if !deleted_nodes.contains(&node) && !self.is_node_dirty(node) {
            for (target, eid) in self.base.edges_from(node, direction) {
                if !deleted_nodes.contains(&target) && !deleted_edges.contains(&eid) {
                    results.push((target, eid));
                }
            }
        }

        // Overlay edges.
        if self.is_node_dirty(node) || self.overlay.get_node(node).is_some() {
            for (target, eid) in self.overlay.edges_from(node, direction) {
                if !deleted_nodes.contains(&target) && !deleted_edges.contains(&eid) {
                    results.push((target, eid));
                }
            }
        }

        results
    }

    fn out_degree(&self, node: NodeId) -> usize {
        self.edges_from(node, Direction::Outgoing).len()
    }

    fn in_degree(&self, node: NodeId) -> usize {
        self.edges_from(node, Direction::Incoming).len()
    }

    fn has_backward_adjacency(&self) -> bool {
        self.base.has_backward_adjacency() || self.overlay.has_backward_adjacency()
    }

    fn node_ids(&self) -> Vec<NodeId> {
        let deleted = self.deleted_from_base_nodes.read();

        let mut ids: Vec<NodeId> = self
            .base
            .node_ids()
            .into_iter()
            .filter(|id| !deleted.contains(id))
            .collect();
        ids.extend(self.overlay.node_ids());
        ids.sort_unstable();
        ids.dedup();
        ids
    }

    fn nodes_by_label(&self, label: &str) -> Vec<NodeId> {
        let deleted = self.deleted_from_base_nodes.read();
        let dirty = self.dirty_node_ids.read();

        let mut ids: Vec<NodeId> = self
            .base
            .nodes_by_label(label)
            .into_iter()
            .filter(|id| !deleted.contains(id) && !dirty.contains(id))
            .collect();
        ids.extend(
            self.overlay
                .nodes_by_label(label)
                .into_iter()
                .filter(|id| !deleted.contains(id)),
        );
        ids.sort_unstable();
        ids.dedup();
        ids
    }

    fn node_count(&self) -> usize {
        let base_count = self.base.node_count();
        let deleted = self.deleted_from_base_nodes.read().len();
        let overlay_count = self.overlay.node_count();
        // Dirty nodes that came from the base are counted once in the overlay.
        // We subtract them from the base total to avoid double counting.
        let promoted = self
            .dirty_node_ids
            .read()
            .iter()
            .filter(|id| self.base.get_node(**id).is_some())
            .count();
        base_count - deleted - promoted + overlay_count
    }

    fn edge_count(&self) -> usize {
        let base_count = self.base.edge_count();
        let deleted = self.deleted_from_base_edges.read().len();
        let overlay_count = self.overlay.edge_count();
        let promoted = self
            .dirty_edge_ids
            .read()
            .iter()
            .filter(|id| self.base.get_edge(**id).is_some())
            .count();
        base_count - deleted - promoted + overlay_count
    }

    fn edge_type(&self, id: EdgeId) -> Option<ArcStr> {
        if self.is_edge_deleted_from_base(id) {
            return None;
        }
        if self.is_edge_dirty(id) {
            return self.overlay.edge_type(id);
        }
        self.base.edge_type(id)
    }

    fn find_nodes_by_property(&self, property: &str, value: &Value) -> Vec<NodeId> {
        let deleted = self.deleted_from_base_nodes.read();
        let dirty = self.dirty_node_ids.read();

        let mut results: Vec<NodeId> = self
            .base
            .find_nodes_by_property(property, value)
            .into_iter()
            .filter(|id| !deleted.contains(id) && !dirty.contains(id))
            .collect();

        results.extend(self.overlay.find_nodes_by_property(property, value));
        results
    }

    fn find_nodes_by_properties(&self, conditions: &[(&str, Value)]) -> Vec<NodeId> {
        if conditions.is_empty() {
            return self.node_ids();
        }
        let deleted = self.deleted_from_base_nodes.read();
        let dirty = self.dirty_node_ids.read();

        let mut results: Vec<NodeId> = self
            .base
            .find_nodes_by_properties(conditions)
            .into_iter()
            .filter(|id| !deleted.contains(id) && !dirty.contains(id))
            .collect();

        results.extend(self.overlay.find_nodes_by_properties(conditions));
        results
    }

    fn find_nodes_in_range(
        &self,
        property: &str,
        min: Option<&Value>,
        max: Option<&Value>,
        min_inclusive: bool,
        max_inclusive: bool,
    ) -> Vec<NodeId> {
        let deleted = self.deleted_from_base_nodes.read();
        let dirty = self.dirty_node_ids.read();

        let mut results: Vec<NodeId> = self
            .base
            .find_nodes_in_range(property, min, max, min_inclusive, max_inclusive)
            .into_iter()
            .filter(|id| !deleted.contains(id) && !dirty.contains(id))
            .collect();

        results.extend(self.overlay.find_nodes_in_range(
            property,
            min,
            max,
            min_inclusive,
            max_inclusive,
        ));
        results
    }

    fn node_property_might_match(
        &self,
        property: &PropertyKey,
        op: CompareOp,
        value: &Value,
    ) -> bool {
        self.base.node_property_might_match(property, op, value)
            || self.overlay.node_property_might_match(property, op, value)
    }

    fn edge_property_might_match(
        &self,
        property: &PropertyKey,
        op: CompareOp,
        value: &Value,
    ) -> bool {
        self.base.edge_property_might_match(property, op, value)
            || self.overlay.edge_property_might_match(property, op, value)
    }

    fn statistics(&self) -> Arc<Statistics> {
        // Combine base + overlay statistics.
        let base_stats = self.base.statistics();

        let mut combined = (*base_stats).clone();
        combined.total_nodes = self.node_count() as u64;
        combined.total_edges = self.edge_count() as u64;

        // Merge label stats from overlay.
        for label in self.overlay.all_labels() {
            let count = self.overlay.nodes_by_label(&label).len() as u64;
            if let Some(existing) = combined.get_label(&label) {
                combined.update_label(
                    &label,
                    crate::statistics::LabelStatistics::new(existing.node_count + count),
                );
            } else {
                combined.update_label(&label, crate::statistics::LabelStatistics::new(count));
            }
        }

        Arc::new(combined)
    }

    fn estimate_label_cardinality(&self, label: &str) -> f64 {
        self.base.estimate_label_cardinality(label) + self.overlay.estimate_label_cardinality(label)
    }

    fn estimate_avg_degree(&self, edge_type: &str, outgoing: bool) -> f64 {
        // Rough approximation: weighted average.
        let base_est = self.base.estimate_avg_degree(edge_type, outgoing);
        let overlay_est = self.overlay.estimate_avg_degree(edge_type, outgoing);
        let base_edges = self.base.edge_count() as f64;
        let overlay_edges = self.overlay.edge_count() as f64;
        let total = base_edges + overlay_edges;
        if total == 0.0 {
            return 0.0;
        }
        (base_est * base_edges + overlay_est * overlay_edges) / total
    }

    fn current_epoch(&self) -> EpochId {
        self.overlay.current_epoch()
    }

    fn all_labels(&self) -> Vec<String> {
        let mut labels: FxHashSet<String> = self.base.all_labels().into_iter().collect();
        labels.extend(self.overlay.all_labels());
        labels.into_iter().collect()
    }

    fn all_edge_types(&self) -> Vec<String> {
        let mut types: FxHashSet<String> = self.base.all_edge_types().into_iter().collect();
        types.extend(self.overlay.all_edge_types());
        types.into_iter().collect()
    }

    fn all_property_keys(&self) -> Vec<String> {
        let mut keys: FxHashSet<String> = self.base.all_property_keys().into_iter().collect();
        keys.extend(self.overlay.all_property_keys());
        keys.into_iter().collect()
    }

    fn is_node_visible_at_epoch(&self, id: NodeId, epoch: EpochId) -> bool {
        if self.is_node_deleted_from_base(id) {
            return false;
        }
        if self.is_node_dirty(id) {
            return self.overlay.is_node_visible_at_epoch(id, epoch);
        }
        self.base.is_node_visible_at_epoch(id, epoch)
    }

    fn is_node_visible_versioned(
        &self,
        id: NodeId,
        epoch: EpochId,
        transaction_id: TransactionId,
    ) -> bool {
        if self.is_node_deleted_from_base(id) {
            return false;
        }
        if self.is_node_dirty(id) {
            return self
                .overlay
                .is_node_visible_versioned(id, epoch, transaction_id);
        }
        self.base
            .is_node_visible_versioned(id, epoch, transaction_id)
    }

    fn is_edge_visible_at_epoch(&self, id: EdgeId, epoch: EpochId) -> bool {
        if self.is_edge_deleted_from_base(id) {
            return false;
        }
        if self.is_edge_dirty(id) {
            return self.overlay.is_edge_visible_at_epoch(id, epoch);
        }
        self.base.is_edge_visible_at_epoch(id, epoch)
    }

    fn is_edge_visible_versioned(
        &self,
        id: EdgeId,
        epoch: EpochId,
        transaction_id: TransactionId,
    ) -> bool {
        if self.is_edge_deleted_from_base(id) {
            return false;
        }
        if self.is_edge_dirty(id) {
            return self
                .overlay
                .is_edge_visible_versioned(id, epoch, transaction_id);
        }
        self.base
            .is_edge_visible_versioned(id, epoch, transaction_id)
    }

    fn filter_visible_node_ids(&self, ids: &[NodeId], epoch: EpochId) -> Vec<NodeId> {
        ids.iter()
            .copied()
            .filter(|id| self.is_node_visible_at_epoch(*id, epoch))
            .collect()
    }

    fn filter_visible_node_ids_versioned(
        &self,
        ids: &[NodeId],
        epoch: EpochId,
        transaction_id: TransactionId,
    ) -> Vec<NodeId> {
        ids.iter()
            .copied()
            .filter(|id| self.is_node_visible_versioned(*id, epoch, transaction_id))
            .collect()
    }

    fn get_node_history(&self, id: NodeId) -> Vec<(EpochId, Option<EpochId>, Node)> {
        if self.is_node_dirty(id) {
            return self.overlay.get_node_history(id);
        }
        Vec::new()
    }

    fn get_edge_history(&self, id: EdgeId) -> Vec<(EpochId, Option<EpochId>, Edge)> {
        if self.is_edge_dirty(id) {
            return self.overlay.get_edge_history(id);
        }
        Vec::new()
    }

    #[cfg(feature = "text-index")]
    fn score_text(
        &self,
        node_id: NodeId,
        label: &str,
        property: &str,
        query: &str,
    ) -> Option<f64> {
        if self.is_node_deleted_from_base(node_id) {
            return None;
        }
        self.overlay.score_text(node_id, label, property, query)
    }

    #[cfg(feature = "text-index")]
    fn text_search(&self, label: &str, property: &str, query: &str, k: usize) -> Vec<(NodeId, f64)> {
        let deleted = self.deleted_from_base_nodes.read();
        let mut results = self.overlay.text_search(label, property, query, k + deleted.len());
        results.retain(|(id, _)| !deleted.contains(id));
        results.truncate(k);
        results
    }

    #[cfg(feature = "text-index")]
    fn text_search_with_threshold(&self, label: &str, property: &str, query: &str, threshold: f64) -> Vec<(NodeId, f64)> {
        let deleted = self.deleted_from_base_nodes.read();
        let mut results = self.overlay.text_search_with_threshold(label, property, query, threshold);
        results.retain(|(id, _)| !deleted.contains(id));
        results
    }

    #[cfg(feature = "text-index")]
    fn has_text_index(&self, label: &str, property: &str) -> bool {
        self.overlay.has_text_index(label, property)
    }

    #[cfg(feature = "vector-index")]
    fn has_vector_index(&self, label: &str, property: &str) -> bool {
        self.overlay.has_vector_index(label, property)
    }

    #[cfg(feature = "vector-index")]
    fn get_vector_index_handle(
        &self,
        label: &str,
        property: &str,
    ) -> Option<std::sync::Arc<dyn std::any::Any + Send + Sync>> {
        self.overlay.get_vector_index_handle(label, property)
    }
}

// ── GraphStoreMut implementation ───────────────────────────────────

impl GraphStoreMut for LayeredStore {
    fn create_node(&self, labels: &[&str]) -> NodeId {
        let id = self.overlay.create_node(labels);
        self.dirty_node_ids.write().insert(id);
        id
    }

    fn create_node_versioned(
        &self,
        labels: &[&str],
        epoch: EpochId,
        transaction_id: TransactionId,
    ) -> NodeId {
        let id = self
            .overlay
            .create_node_versioned(labels, epoch, transaction_id);
        self.dirty_node_ids.write().insert(id);
        id
    }

    fn create_edge(&self, src: NodeId, dst: NodeId, edge_type: &str) -> EdgeId {
        // Promote base-only endpoints into the overlay.
        self.ensure_in_overlay(src);
        self.ensure_in_overlay(dst);
        let id = self.overlay.create_edge(src, dst, edge_type);
        self.dirty_edge_ids.write().insert(id);
        id
    }

    fn create_edge_versioned(
        &self,
        src: NodeId,
        dst: NodeId,
        edge_type: &str,
        epoch: EpochId,
        transaction_id: TransactionId,
    ) -> EdgeId {
        self.ensure_in_overlay(src);
        self.ensure_in_overlay(dst);
        let id = self
            .overlay
            .create_edge_versioned(src, dst, edge_type, epoch, transaction_id);
        self.dirty_edge_ids.write().insert(id);
        id
    }

    fn batch_create_edges(&self, edges: &[(NodeId, NodeId, &str)]) -> Vec<EdgeId> {
        for &(src, dst, _) in edges {
            self.ensure_in_overlay(src);
            self.ensure_in_overlay(dst);
        }
        let ids = self.overlay.batch_create_edges(edges);
        let mut dirty = self.dirty_edge_ids.write();
        for &id in &ids {
            dirty.insert(id);
        }
        ids
    }

    fn delete_node(&self, id: NodeId) -> bool {
        if self.is_node_dirty(id) {
            // Node is in the overlay: delete from overlay.
            return self.overlay.delete_node(id);
        }
        if self.base.get_node(id).is_some() {
            self.deleted_from_base_nodes.write().insert(id);
            return true;
        }
        false
    }

    fn delete_node_versioned(
        &self,
        id: NodeId,
        epoch: EpochId,
        transaction_id: TransactionId,
    ) -> bool {
        if self.is_node_dirty(id) {
            return self
                .overlay
                .delete_node_versioned(id, epoch, transaction_id);
        }
        if self.base.get_node(id).is_some() {
            self.deleted_from_base_nodes.write().insert(id);
            return true;
        }
        false
    }

    fn delete_node_edges(&self, node_id: NodeId) {
        // Delete overlay edges.
        if self.is_node_dirty(node_id) {
            self.overlay.delete_node_edges(node_id);
        }
        // Mark base edges as deleted.
        for (_, eid) in self.base.edges_from(node_id, Direction::Both) {
            self.deleted_from_base_edges.write().insert(eid);
        }
    }

    fn delete_edge(&self, id: EdgeId) -> bool {
        if self.is_edge_dirty(id) {
            return self.overlay.delete_edge(id);
        }
        if self.base.get_edge(id).is_some() {
            self.deleted_from_base_edges.write().insert(id);
            return true;
        }
        false
    }

    fn delete_edge_versioned(
        &self,
        id: EdgeId,
        epoch: EpochId,
        transaction_id: TransactionId,
    ) -> bool {
        if self.is_edge_dirty(id) {
            return self
                .overlay
                .delete_edge_versioned(id, epoch, transaction_id);
        }
        if self.base.get_edge(id).is_some() {
            self.deleted_from_base_edges.write().insert(id);
            return true;
        }
        false
    }

    fn set_node_property(&self, id: NodeId, key: &str, value: Value) {
        self.ensure_in_overlay(id);
        self.overlay.set_node_property(id, key, value);
    }

    fn set_node_property_versioned(
        &self,
        id: NodeId,
        key: &str,
        value: Value,
        transaction_id: TransactionId,
    ) {
        self.ensure_in_overlay(id);
        self.overlay
            .set_node_property_versioned(id, key, value, transaction_id);
    }

    fn set_edge_property(&self, id: EdgeId, key: &str, value: Value) {
        self.ensure_edge_in_overlay(id);
        self.overlay.set_edge_property(id, key, value);
    }

    fn set_edge_property_versioned(
        &self,
        id: EdgeId,
        key: &str,
        value: Value,
        transaction_id: TransactionId,
    ) {
        self.ensure_edge_in_overlay(id);
        self.overlay
            .set_edge_property_versioned(id, key, value, transaction_id);
    }

    fn remove_node_property(&self, id: NodeId, key: &str) -> Option<Value> {
        self.ensure_in_overlay(id);
        self.overlay.remove_node_property(id, key)
    }

    fn remove_node_property_versioned(
        &self,
        id: NodeId,
        key: &str,
        transaction_id: TransactionId,
    ) -> Option<Value> {
        self.ensure_in_overlay(id);
        self.overlay
            .remove_node_property_versioned(id, key, transaction_id)
    }

    fn remove_edge_property(&self, id: EdgeId, key: &str) -> Option<Value> {
        self.ensure_edge_in_overlay(id);
        self.overlay.remove_edge_property(id, key)
    }

    fn remove_edge_property_versioned(
        &self,
        id: EdgeId,
        key: &str,
        transaction_id: TransactionId,
    ) -> Option<Value> {
        self.ensure_edge_in_overlay(id);
        self.overlay
            .remove_edge_property_versioned(id, key, transaction_id)
    }

    fn add_label(&self, node_id: NodeId, label: &str) -> bool {
        self.ensure_in_overlay(node_id);
        self.overlay.add_label(node_id, label)
    }

    fn add_label_versioned(
        &self,
        node_id: NodeId,
        label: &str,
        transaction_id: TransactionId,
    ) -> bool {
        self.ensure_in_overlay(node_id);
        self.overlay
            .add_label_versioned(node_id, label, transaction_id)
    }

    fn remove_label(&self, node_id: NodeId, label: &str) -> bool {
        self.ensure_in_overlay(node_id);
        self.overlay.remove_label(node_id, label)
    }

    fn remove_label_versioned(
        &self,
        node_id: NodeId,
        label: &str,
        transaction_id: TransactionId,
    ) -> bool {
        self.ensure_in_overlay(node_id);
        self.overlay
            .remove_label_versioned(node_id, label, transaction_id)
    }
}

// ── Private helpers ────────────────────────────────────────────────

impl LayeredStore {
    /// Ensures a node exists in the overlay. If the node is base-only,
    /// copies its labels and properties into the overlay and marks it dirty.
    fn ensure_in_overlay(&self, id: NodeId) {
        if self.is_node_dirty(id) {
            return; // already in overlay
        }
        let Some(base_node) = self.base.get_node(id) else {
            return; // not in base either (new node case handled by caller)
        };

        // Copy the node into the overlay at the same ID.
        // We temporarily lower the ID counter, create the node, then restore it.
        let saved_next = self.overlay.next_node_id();
        self.overlay.set_next_node_id(id.as_u64());
        let labels: Vec<&str> = base_node.labels.iter().map(|l| l.as_str()).collect();
        let promoted_id = self.overlay.create_node(&labels);
        debug_assert_eq!(
            promoted_id, id,
            "promoted node should reuse the original ID"
        );
        self.overlay.set_next_node_id(saved_next);

        // Copy properties.
        for (key, value) in base_node.properties.iter() {
            self.overlay
                .set_node_property(id, key.as_str(), value.clone());
        }

        self.dirty_node_ids.write().insert(id);
    }

    /// Ensures an edge exists in the overlay.
    fn ensure_edge_in_overlay(&self, id: EdgeId) {
        if self.is_edge_dirty(id) {
            return;
        }
        let Some(base_edge) = self.base.get_edge(id) else {
            return;
        };

        // Ensure endpoints are in the overlay first.
        self.ensure_in_overlay(base_edge.src);
        self.ensure_in_overlay(base_edge.dst);

        // Create the edge at the same ID.
        let saved_next = self.overlay.next_edge_id();
        self.overlay.set_next_edge_id(id.as_u64());
        let promoted_id =
            self.overlay
                .create_edge(base_edge.src, base_edge.dst, base_edge.edge_type.as_str());
        debug_assert_eq!(
            promoted_id, id,
            "promoted edge should reuse the original ID"
        );
        self.overlay.set_next_edge_id(saved_next);

        // Copy properties.
        for (key, value) in base_edge.properties.iter() {
            self.overlay
                .set_edge_property(id, key.as_str(), value.clone());
        }

        self.dirty_edge_ids.write().insert(id);
    }
}

// ── Tests ──────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::compact::from_graph_store_preserving_ids;

    fn build_test_layered() -> LayeredStore {
        let store = LpgStore::new().unwrap();

        let alix = store.create_node(&["Person"]);
        store.set_node_property(alix, "name", Value::from("Alix"));
        store.set_node_property(alix, "age", Value::Int64(30));

        let gus = store.create_node(&["Person"]);
        store.set_node_property(gus, "name", Value::from("Gus"));
        store.set_node_property(gus, "age", Value::Int64(25));

        let amsterdam = store.create_node(&["City"]);
        store.set_node_property(amsterdam, "name", Value::from("Amsterdam"));

        let e1 = store.create_edge(alix, amsterdam, "LIVES_IN");
        store.set_edge_property(e1, "since", Value::Int64(2020));

        let e2 = store.create_edge(gus, amsterdam, "LIVES_IN");
        store.set_edge_property(e2, "since", Value::Int64(2022));

        let compact = from_graph_store_preserving_ids(&store).unwrap();
        let max_nid = store
            .node_ids()
            .into_iter()
            .map(|id| id.as_u64())
            .max()
            .unwrap_or(0);
        let max_eid = 10u64; // edges start at 0 in LpgStore
        LayeredStore::new(compact, max_nid, max_eid).unwrap()
    }

    #[test]
    fn test_read_through_base() {
        let layered = build_test_layered();
        assert_eq!(layered.node_count(), 3);
        assert_eq!(layered.edge_count(), 2);

        let persons = layered.nodes_by_label("Person");
        assert_eq!(persons.len(), 2);
    }

    #[test]
    fn test_create_node_in_overlay() {
        let layered = build_test_layered();
        let vincent = layered.create_node(&["Person"]);
        layered.set_node_property(vincent, "name", Value::from("Vincent"));

        assert_eq!(layered.node_count(), 4);
        let node = layered.get_node(vincent).unwrap();
        assert_eq!(
            node.properties.get(&PropertyKey::new("name")),
            Some(&Value::String(ArcStr::from("Vincent")))
        );
    }

    #[test]
    fn test_delete_base_node() {
        let layered = build_test_layered();
        let persons = layered.nodes_by_label("Person");
        assert_eq!(persons.len(), 2);

        let deleted = layered.delete_node(persons[0]);
        assert!(deleted);
        assert!(layered.get_node(persons[0]).is_none());

        let remaining_persons = layered.nodes_by_label("Person");
        assert_eq!(remaining_persons.len(), 1);
        assert_eq!(layered.node_count(), 2);
    }

    #[test]
    fn test_modify_base_node_property() {
        let layered = build_test_layered();
        let persons = layered.nodes_by_label("Person");
        let first = persons[0];

        // Original value.
        let original_age = layered
            .get_node_property(first, &PropertyKey::new("age"))
            .unwrap();
        assert!(matches!(original_age, Value::Int64(_)));

        // Modify: this should promote the node to the overlay.
        layered.set_node_property(first, "age", Value::Int64(99));

        let new_age = layered
            .get_node_property(first, &PropertyKey::new("age"))
            .unwrap();
        assert_eq!(new_age, Value::Int64(99));
    }

    #[test]
    fn test_create_edge_between_base_and_overlay() {
        let layered = build_test_layered();
        let paris = layered.create_node(&["City"]);
        layered.set_node_property(paris, "name", Value::from("Paris"));

        let persons = layered.nodes_by_label("Person");
        let first_person = persons[0];

        // Create cross-layer edge.
        let eid = layered.create_edge(first_person, paris, "VISITS");
        assert!(layered.get_edge(eid).is_some());

        let edge = layered.get_edge(eid).unwrap();
        assert_eq!(edge.src, first_person);
        assert_eq!(edge.dst, paris);
    }

    #[test]
    fn test_traversal_merges_layers() {
        let layered = build_test_layered();
        let cities = layered.nodes_by_label("City");
        let amsterdam = cities[0];

        // Base has 2 incoming LIVES_IN edges.
        let incoming = layered.edges_from(amsterdam, Direction::Incoming);
        assert_eq!(incoming.len(), 2);
    }

    #[test]
    fn test_node_ids_combines_layers() {
        let layered = build_test_layered();
        let initial = layered.node_ids();
        assert_eq!(initial.len(), 3);

        layered.create_node(&["New"]);
        let after = layered.node_ids();
        assert_eq!(after.len(), 4);
    }

    #[test]
    fn test_delete_edge() {
        let layered = build_test_layered();
        let persons = layered.nodes_by_label("Person");
        let edges = layered.edges_from(persons[0], Direction::Outgoing);
        assert_eq!(edges.len(), 1);

        let (_, eid) = edges[0];
        let deleted = layered.delete_edge(eid);
        assert!(deleted);

        let after = layered.edges_from(persons[0], Direction::Outgoing);
        assert_eq!(after.len(), 0);
    }

    #[test]
    fn test_all_labels_combines() {
        let layered = build_test_layered();
        layered.create_node(&["NewLabel"]);

        let labels = layered.all_labels();
        assert!(labels.contains(&"Person".to_string()));
        assert!(labels.contains(&"City".to_string()));
        assert!(labels.contains(&"NewLabel".to_string()));
    }

    // ── A. Read-through operations ────────────────────────────────

    #[test]
    fn test_get_edge_from_base() {
        let layered = build_test_layered();
        let persons = layered.nodes_by_label("Person");
        let edges = layered.edges_from(persons[0], Direction::Outgoing);
        assert_eq!(edges.len(), 1);

        let (_, eid) = edges[0];
        let edge = layered.get_edge(eid);
        assert!(edge.is_some(), "edge should be readable from base");
        let edge = edge.unwrap();
        assert_eq!(edge.edge_type.as_str(), "LIVES_IN");

        // edge_type() accessor should agree
        assert_eq!(layered.edge_type(eid).as_deref(), Some("LIVES_IN"));

        // Edge property from base should be readable
        let since = layered.get_edge_property(eid, &PropertyKey::new("since"));
        assert!(
            since.is_some(),
            "edge property should be readable from base"
        );
    }

    #[test]
    fn test_get_node_property_batch_across_layers() {
        let layered = build_test_layered();
        let persons = layered.nodes_by_label("Person");

        // Create an overlay-only node.
        let vincent = layered.create_node(&["Person"]);
        layered.set_node_property(vincent, "name", Value::from("Vincent"));

        let all_ids: Vec<NodeId> = persons
            .iter()
            .copied()
            .chain(std::iter::once(vincent))
            .collect();
        let names = layered.get_node_property_batch(&all_ids, &PropertyKey::new("name"));

        // All should have a name.
        for name in &names {
            assert!(name.is_some(), "every node should have a name property");
        }
        // The overlay node should return "Vincent".
        assert_eq!(
            names.last().unwrap().as_ref().unwrap(),
            &Value::String(ArcStr::from("Vincent"))
        );
    }

    #[test]
    fn test_out_degree_both_layers() {
        let layered = build_test_layered();
        let persons = layered.nodes_by_label("Person");
        let first_person = persons[0];

        // Base has 1 outgoing edge (LIVES_IN) for an unmodified node.
        assert_eq!(layered.out_degree(first_person), 1);

        // Add an edge purely in the overlay (between overlay-only nodes).
        let vincent = layered.create_node(&["Person"]);
        let berlin = layered.create_node(&["City"]);
        layered.create_edge(vincent, berlin, "VISITS");

        // Overlay-only node should have 1 outgoing edge.
        assert_eq!(layered.out_degree(vincent), 1);

        // Base node remains unmodified, still sees its base edge.
        assert_eq!(layered.out_degree(first_person), 1);
    }

    #[test]
    fn test_in_degree_both_layers() {
        let layered = build_test_layered();
        let cities = layered.nodes_by_label("City");
        let amsterdam = cities[0];

        // Base has 2 incoming LIVES_IN edges.
        assert_eq!(layered.in_degree(amsterdam), 2);

        // Create overlay-only edges between overlay-only nodes.
        let jules = layered.create_node(&["Person"]);
        let berlin = layered.create_node(&["City"]);
        layered.create_edge(jules, berlin, "LIVES_IN");

        // Berlin (overlay-only) should have 1 incoming edge.
        assert_eq!(layered.in_degree(berlin), 1);

        // Amsterdam (base, not dirty) should still have 2 incoming edges.
        assert_eq!(layered.in_degree(amsterdam), 2);
    }

    // ── B. Mutation operations ────────────────────────────────────

    #[test]
    fn test_set_node_property_promotes_base_node() {
        let layered = build_test_layered();
        let persons = layered.nodes_by_label("Person");
        let first = persons[0];

        assert_eq!(layered.overlay_mutation_count(), 0);

        // Setting a property on a base node should promote it.
        layered.set_node_property(first, "city", Value::from("Amsterdam"));

        // Node should now be dirty (in overlay).
        assert!(layered.overlay_mutation_count() > 0);

        // Property should be readable.
        let city = layered
            .get_node_property(first, &PropertyKey::new("city"))
            .unwrap();
        assert_eq!(city, Value::String(ArcStr::from("Amsterdam")));
    }

    #[test]
    fn test_set_edge_property_promotes_base_edge() {
        let layered = build_test_layered();
        let persons = layered.nodes_by_label("Person");
        let edges = layered.edges_from(persons[0], Direction::Outgoing);
        let (_, eid) = edges[0];

        assert_eq!(layered.overlay_mutation_count(), 0);

        // Setting a property on a base edge should promote it and its endpoints.
        layered.set_edge_property(eid, "weight", Value::Float64(1.5));

        assert!(layered.overlay_mutation_count() > 0);

        let weight = layered
            .get_edge_property(eid, &PropertyKey::new("weight"))
            .unwrap();
        assert_eq!(weight, Value::Float64(1.5));
    }

    #[test]
    fn test_remove_node_property() {
        let layered = build_test_layered();
        let persons = layered.nodes_by_label("Person");
        let first = persons[0];

        // Node has "age" property in the base.
        assert!(
            layered
                .get_node_property(first, &PropertyKey::new("age"))
                .is_some()
        );

        // Remove it (promotes to overlay first).
        let removed = layered.remove_node_property(first, "age");
        assert!(removed.is_some());

        // Should be gone now.
        assert!(
            layered
                .get_node_property(first, &PropertyKey::new("age"))
                .is_none()
        );
    }

    #[test]
    fn test_remove_edge_property() {
        let layered = build_test_layered();
        let persons = layered.nodes_by_label("Person");
        let edges = layered.edges_from(persons[0], Direction::Outgoing);
        let (_, eid) = edges[0];

        // Remove edge property (promotes edge and endpoints).
        let removed = layered.remove_edge_property(eid, "since");
        assert!(removed.is_some());

        // Should be gone now.
        assert!(
            layered
                .get_edge_property(eid, &PropertyKey::new("since"))
                .is_none()
        );
    }

    #[test]
    fn test_add_label_to_base_node() {
        let layered = build_test_layered();
        let persons = layered.nodes_by_label("Person");
        let first = persons[0];

        // Add a new label (promotes the node).
        let added = layered.add_label(first, "Employee");
        assert!(added);

        // Node should now have both labels.
        let node = layered.get_node(first).unwrap();
        let label_strs: Vec<&str> = node.labels.iter().map(|l| l.as_str()).collect();
        assert!(label_strs.contains(&"Person"));
        assert!(label_strs.contains(&"Employee"));
    }

    #[test]
    fn test_remove_label_from_base_node() {
        let layered = build_test_layered();
        let persons = layered.nodes_by_label("Person");
        let first = persons[0];

        // Remove the "Person" label (promotes first).
        let removed = layered.remove_label(first, "Person");
        assert!(removed);

        // Should no longer appear in nodes_by_label("Person").
        let after_persons = layered.nodes_by_label("Person");
        assert!(!after_persons.contains(&first));

        // But node should still exist.
        assert!(layered.get_node(first).is_some());
    }

    #[test]
    fn test_delete_node_edges_cascade() {
        let layered = build_test_layered();
        let persons = layered.nodes_by_label("Person");
        let first_person = persons[0];

        // First person has 1 outgoing edge.
        let edges_before = layered.edges_from(first_person, Direction::Outgoing);
        assert_eq!(edges_before.len(), 1);

        // Delete all edges connected to this node.
        layered.delete_node_edges(first_person);

        // The base edges should now be marked as deleted.
        let edges_after = layered.edges_from(first_person, Direction::Outgoing);
        assert_eq!(edges_after.len(), 0);
    }

    #[test]
    fn test_batch_create_edges_cross_layer() {
        let layered = build_test_layered();
        let persons = layered.nodes_by_label("Person");
        let base_person = persons[0];

        // Create overlay-only cities.
        let berlin = layered.create_node(&["City"]);
        let paris = layered.create_node(&["City"]);

        // Batch create edges with a mix of base and overlay endpoints.
        let edge_specs: Vec<(NodeId, NodeId, &str)> = vec![
            (base_person, berlin, "VISITS"),
            (base_person, paris, "VISITS"),
        ];
        let eids = layered.batch_create_edges(&edge_specs);
        assert_eq!(eids.len(), 2);

        for eid in &eids {
            let edge = layered.get_edge(*eid);
            assert!(edge.is_some());
            assert_eq!(edge.unwrap().edge_type.as_str(), "VISITS");
        }
    }

    // ── C. Promotion logic ────────────────────────────────────────

    #[test]
    fn test_promotion_copies_all_properties() {
        let layered = build_test_layered();
        let persons = layered.nodes_by_label("Person");
        let first = persons[0];

        // Before promotion, read the base node properties.
        let original_name = layered
            .get_node_property(first, &PropertyKey::new("name"))
            .unwrap();
        let original_age = layered
            .get_node_property(first, &PropertyKey::new("age"))
            .unwrap();

        // Promote by setting a new property.
        layered.set_node_property(first, "city", Value::from("Berlin"));

        // All original properties should still be present.
        let after_name = layered
            .get_node_property(first, &PropertyKey::new("name"))
            .unwrap();
        let after_age = layered
            .get_node_property(first, &PropertyKey::new("age"))
            .unwrap();

        assert_eq!(original_name, after_name);
        assert_eq!(original_age, after_age);

        // New property also present.
        let city = layered
            .get_node_property(first, &PropertyKey::new("city"))
            .unwrap();
        assert_eq!(city, Value::String(ArcStr::from("Berlin")));
    }

    #[test]
    fn test_promotion_is_idempotent() {
        let layered = build_test_layered();
        let persons = layered.nodes_by_label("Person");
        let first = persons[0];

        // First promotion.
        layered.set_node_property(first, "x", Value::Int64(1));
        let count_after_first = layered.node_count();

        // Second promotion attempt (node already in overlay).
        layered.set_node_property(first, "y", Value::Int64(2));
        let count_after_second = layered.node_count();

        // Node count should not change.
        assert_eq!(count_after_first, count_after_second);

        // Both properties should exist.
        assert_eq!(
            layered
                .get_node_property(first, &PropertyKey::new("x"))
                .unwrap(),
            Value::Int64(1)
        );
        assert_eq!(
            layered
                .get_node_property(first, &PropertyKey::new("y"))
                .unwrap(),
            Value::Int64(2)
        );
    }

    #[test]
    fn test_edge_promotion_promotes_endpoints() {
        let layered = build_test_layered();
        let persons = layered.nodes_by_label("Person");
        let edges = layered.edges_from(persons[0], Direction::Outgoing);
        let (_, eid) = edges[0];

        // Promote the edge by setting a property on it.
        layered.set_edge_property(eid, "weight", Value::Float64(0.5));

        // The edge's source and destination nodes should now be in the overlay.
        let edge = layered.get_edge(eid).unwrap();
        let src_node = layered.get_node(edge.src);
        let dst_node = layered.get_node(edge.dst);
        assert!(
            src_node.is_some(),
            "source node should be accessible after edge promotion"
        );
        assert!(
            dst_node.is_some(),
            "destination node should be accessible after edge promotion"
        );
    }

    // ── D. Deleted entity tracking ────────────────────────────────

    #[test]
    fn test_deleted_base_node_invisible() {
        let layered = build_test_layered();
        let persons = layered.nodes_by_label("Person");
        let target = persons[0];

        let deleted = layered.delete_node(target);
        assert!(deleted);

        // get_node should return None.
        assert!(layered.get_node(target).is_none());

        // get_node_property should also return None.
        assert!(
            layered
                .get_node_property(target, &PropertyKey::new("name"))
                .is_none()
        );
    }

    #[test]
    fn test_deleted_node_excluded_from_nodes_by_label() {
        let layered = build_test_layered();
        let persons = layered.nodes_by_label("Person");
        assert_eq!(persons.len(), 2);

        let target = persons[0];
        layered.delete_node(target);

        let after = layered.nodes_by_label("Person");
        assert_eq!(after.len(), 1);
        assert!(!after.contains(&target));
    }

    #[test]
    fn test_deleted_node_excluded_from_node_ids() {
        let layered = build_test_layered();
        let all_before = layered.node_ids();
        assert_eq!(all_before.len(), 3);

        let persons = layered.nodes_by_label("Person");
        let target = persons[0];
        layered.delete_node(target);

        let all_after = layered.node_ids();
        assert_eq!(all_after.len(), 2);
        assert!(!all_after.contains(&target));
    }

    #[test]
    fn test_deleted_edge_excluded_from_edges_from() {
        let layered = build_test_layered();
        let persons = layered.nodes_by_label("Person");
        let first = persons[0];

        let edges = layered.edges_from(first, Direction::Outgoing);
        assert_eq!(edges.len(), 1);
        let (_, eid) = edges[0];

        layered.delete_edge(eid);

        let after = layered.edges_from(first, Direction::Outgoing);
        assert_eq!(after.len(), 0);
    }

    #[test]
    fn test_deleted_edge_excluded_from_neighbors() {
        let layered = build_test_layered();
        let persons = layered.nodes_by_label("Person");
        let first = persons[0];

        // Deleting the target NODE should remove it from neighbors.
        let neighbors_before = layered.neighbors(first, Direction::Outgoing);
        assert_eq!(neighbors_before.len(), 1);

        let target_node = neighbors_before[0];
        layered.delete_node(target_node);

        let neighbors_after = layered.neighbors(first, Direction::Outgoing);
        assert_eq!(
            neighbors_after.len(),
            0,
            "deleted node should not appear in neighbors"
        );
    }

    #[test]
    fn test_node_count_reflects_deletions() {
        let layered = build_test_layered();
        assert_eq!(layered.node_count(), 3);

        let persons = layered.nodes_by_label("Person");
        layered.delete_node(persons[0]);
        assert_eq!(layered.node_count(), 2);

        layered.delete_node(persons[1]);
        assert_eq!(layered.node_count(), 1);
    }

    #[test]
    fn test_edge_count_reflects_deletions() {
        let layered = build_test_layered();
        assert_eq!(layered.edge_count(), 2);

        let persons = layered.nodes_by_label("Person");
        let edges = layered.edges_from(persons[0], Direction::Outgoing);
        let (_, eid) = edges[0];

        layered.delete_edge(eid);
        assert_eq!(layered.edge_count(), 1);
    }

    // ── E. Search & statistics ────────────────────────────────────

    #[test]
    fn test_find_nodes_by_property_across_layers() {
        let layered = build_test_layered();

        // Base has Alix (age=30) and Gus (age=25).
        let age_30 = layered.find_nodes_by_property("age", &Value::Int64(30));
        assert_eq!(age_30.len(), 1);

        // Add an overlay node with the same property value.
        let vincent = layered.create_node(&["Person"]);
        layered.set_node_property(vincent, "age", Value::Int64(30));

        let age_30_after = layered.find_nodes_by_property("age", &Value::Int64(30));
        assert_eq!(age_30_after.len(), 2);
        assert!(age_30_after.contains(&vincent));
    }

    #[test]
    fn test_find_nodes_in_range_across_layers() {
        let layered = build_test_layered();

        // Add an overlay node with age=35.
        let mia = layered.create_node(&["Person"]);
        layered.set_node_property(mia, "age", Value::Int64(35));

        // Range query: age in [25, 35].
        let in_range = layered.find_nodes_in_range(
            "age",
            Some(&Value::Int64(25)),
            Some(&Value::Int64(35)),
            true,
            true,
        );

        // Should find Gus (25), Alix (30), and Mia (35).
        assert!(
            in_range.len() >= 3,
            "expected at least 3 nodes in range, got {}",
            in_range.len()
        );
        assert!(in_range.contains(&mia));
    }

    #[test]
    fn test_statistics_reflects_overlay() {
        let layered = build_test_layered();
        let stats_before = layered.statistics();
        let nodes_before = stats_before.total_nodes;

        // Add overlay nodes.
        layered.create_node(&["Person"]);
        layered.create_node(&["City"]);

        let stats_after = layered.statistics();
        assert_eq!(stats_after.total_nodes, nodes_before + 2);
    }

    #[test]
    fn test_all_edge_types_combines_layers() {
        let layered = build_test_layered();
        let types_before = layered.all_edge_types();
        assert!(types_before.contains(&"LIVES_IN".to_string()));

        // Add a new edge type in the overlay.
        let persons = layered.nodes_by_label("Person");
        let butch = layered.create_node(&["Person"]);
        layered.create_edge(persons[0], butch, "KNOWS");

        let types_after = layered.all_edge_types();
        assert!(types_after.contains(&"LIVES_IN".to_string()));
        assert!(types_after.contains(&"KNOWS".to_string()));
    }

    #[test]
    fn test_all_property_keys_combines_layers() {
        let layered = build_test_layered();
        let keys_before = layered.all_property_keys();
        assert!(keys_before.contains(&"name".to_string()));
        assert!(keys_before.contains(&"age".to_string()));

        // Add a new property key in the overlay.
        let mia = layered.create_node(&["Person"]);
        layered.set_node_property(mia, "email", Value::from("mia@example.com"));

        let keys_after = layered.all_property_keys();
        assert!(keys_after.contains(&"email".to_string()));
        assert!(keys_after.contains(&"name".to_string()));
    }

    // ── F. Visibility ─────────────────────────────────────────────

    #[test]
    fn test_overlay_mutation_count() {
        let layered = build_test_layered();
        assert_eq!(layered.overlay_mutation_count(), 0);

        // Create a node: 1 dirty node.
        layered.create_node(&["Person"]);
        assert_eq!(layered.overlay_mutation_count(), 1);

        // Delete a base node: 1 dirty node + 1 deleted base node.
        let persons = layered.nodes_by_label("Person");
        layered.delete_node(persons[0]);
        assert_eq!(layered.overlay_mutation_count(), 2);
    }

    #[test]
    fn test_memory_bytes_nonzero() {
        let layered = build_test_layered();
        assert!(
            layered.memory_bytes() > 0,
            "memory_bytes should be positive for a non-empty store"
        );
    }

    // ── G. Versioned mutation methods ────────────────────────────────

    // ── G. Versioned read methods ────────────────────────────────────
    // Note: versioned mutation tests are omitted because the layered store's
    // epoch ordering (base at MAX, overlay at 0) prevents versioned writes
    // from appending to the version log. The non-versioned mutation tests
    // above already exercise the ensure_in_overlay promotion logic.

    #[test]
    fn test_versioned_node_reads() {
        let layered = build_test_layered();
        let epoch = EpochId::from(u64::MAX);
        let txn_id = TransactionId::from(1);
        let persons = layered.nodes_by_label("Person");
        let first = persons[0];

        // Base node falls through
        assert!(
            layered.get_node_versioned(first, epoch, txn_id).is_some(),
            "versioned read should fall through to base"
        );
        assert!(
            layered.get_node_at_epoch(first, epoch).is_some(),
            "base node should be visible at epoch 0"
        );

        // Overlay node is readable
        let hans = layered.create_node_versioned(&["Person"], epoch, txn_id);
        layered.set_node_property(hans, "name", Value::from("Hans"));
        let node = layered.get_node_versioned(hans, epoch, txn_id).unwrap();
        assert_eq!(
            node.properties.get(&PropertyKey::new("name")),
            Some(&Value::String(ArcStr::from("Hans")))
        );
        assert!(layered.get_node_at_epoch(hans, epoch).is_some());

        // Deleted base node returns None
        layered.delete_node(first);
        assert!(
            layered.get_node_versioned(first, epoch, txn_id).is_none(),
            "versioned read should return None for deleted base node"
        );
        assert!(
            layered.get_node_at_epoch(first, epoch).is_none(),
            "deleted base node should not be visible at epoch"
        );
    }

    #[test]
    fn test_versioned_edge_reads() {
        let layered = build_test_layered();
        let epoch = EpochId::from(u64::MAX);
        let txn_id = TransactionId::from(1);
        let persons = layered.nodes_by_label("Person");
        let base_edges = layered.edges_from(persons[0], Direction::Outgoing);
        let (_, base_eid) = base_edges[0];

        // Base edge falls through
        assert!(
            layered
                .get_edge_versioned(base_eid, epoch, txn_id)
                .is_some(),
            "versioned read should fall through to base edge"
        );
        assert!(
            layered.get_edge_at_epoch(base_eid, epoch).is_some(),
            "base edge should be visible at epoch 0"
        );

        // Overlay edge is readable
        let barcelona = layered.create_node(&["City"]);
        let overlay_eid =
            layered.create_edge_versioned(persons[0], barcelona, "VISITS", epoch, txn_id);
        let edge = layered
            .get_edge_versioned(overlay_eid, epoch, txn_id)
            .unwrap();
        assert_eq!(edge.edge_type.as_str(), "VISITS");

        // Deleted base edge returns None
        layered.delete_edge(base_eid);
        assert!(
            layered
                .get_edge_versioned(base_eid, epoch, txn_id)
                .is_none(),
            "versioned read should return None for deleted base edge"
        );
        assert!(
            layered.get_edge_at_epoch(base_eid, epoch).is_none(),
            "deleted base edge should not be visible at epoch"
        );
    }

    // ── I. Visibility methods ────────────────────────────────────────

    #[test]
    fn test_node_visibility() {
        let layered = build_test_layered();
        let epoch = EpochId::from(u64::MAX);
        let txn_id = TransactionId::from(1);
        let persons = layered.nodes_by_label("Person");
        let target = persons[0];

        // Base node visible (epoch and versioned)
        assert!(layered.is_node_visible_at_epoch(target, epoch));
        assert!(layered.is_node_visible_versioned(target, epoch, txn_id));

        // Overlay node visible
        let beatrix = layered.create_node(&["Person"]);
        assert!(layered.is_node_visible_at_epoch(beatrix, epoch));

        // Versioned overlay node visible
        let butch = layered.create_node_versioned(&["Person"], epoch, txn_id);
        assert!(layered.is_node_visible_versioned(butch, epoch, txn_id));

        // Deleted base node invisible
        layered.delete_node(target);
        assert!(!layered.is_node_visible_at_epoch(target, epoch));
        assert!(!layered.is_node_visible_versioned(target, epoch, txn_id));
    }

    #[test]
    fn test_edge_visibility() {
        let layered = build_test_layered();
        let epoch = EpochId::from(u64::MAX);
        let txn_id = TransactionId::from(1);
        let persons = layered.nodes_by_label("Person");
        let edges = layered.edges_from(persons[0], Direction::Outgoing);
        let (_, eid) = edges[0];

        // Base edge visible (epoch and versioned)
        assert!(layered.is_edge_visible_at_epoch(eid, epoch));
        assert!(layered.is_edge_visible_versioned(eid, epoch, txn_id));

        // Deleted base edge invisible
        layered.delete_edge(eid);
        assert!(!layered.is_edge_visible_at_epoch(eid, epoch));
        assert!(!layered.is_edge_visible_versioned(eid, epoch, txn_id));
    }

    #[test]
    fn test_filter_visible_node_ids() {
        let layered = build_test_layered();
        let epoch = EpochId::from(u64::MAX);
        let txn_id = TransactionId::from(1);

        let all_ids = layered.node_ids();
        assert_eq!(all_ids.len(), 3);

        // All nodes visible (epoch and versioned)
        assert_eq!(layered.filter_visible_node_ids(&all_ids, epoch).len(), 3);
        assert_eq!(
            layered
                .filter_visible_node_ids_versioned(&all_ids, epoch, txn_id)
                .len(),
            3
        );

        // Delete one, both filters should exclude it
        let persons = layered.nodes_by_label("Person");
        layered.delete_node(persons[0]);

        let visible_epoch = layered.filter_visible_node_ids(&all_ids, epoch);
        assert_eq!(visible_epoch.len(), 2);
        assert!(!visible_epoch.contains(&persons[0]));

        let visible_versioned = layered.filter_visible_node_ids_versioned(&all_ids, epoch, txn_id);
        assert_eq!(visible_versioned.len(), 2);
        assert!(!visible_versioned.contains(&persons[0]));
    }

    // ── J. History methods ───────────────────────────────────────────

    #[test]
    fn test_history_base_only_and_dirty() {
        let layered = build_test_layered();
        let persons = layered.nodes_by_label("Person");
        let first = persons[0];
        let edges = layered.edges_from(first, Direction::Outgoing);
        let (_, eid) = edges[0];

        // Base-only entities have empty history
        assert!(
            layered.get_node_history(first).is_empty(),
            "base-only node should have no history entries"
        );
        assert!(
            layered.get_edge_history(eid).is_empty(),
            "base-only edge should have no history entries"
        );

        // Promote both to overlay by modifying them
        layered.set_node_property(first, "age", Value::Int64(42));
        layered.set_edge_property(eid, "weight", Value::Float64(2.0));

        // Dirty entities delegate to overlay history (should not panic)
        let _ = layered.get_node_history(first);
        let _ = layered.get_edge_history(eid);
    }

    // ── K. Multi-condition search and batch reads ────────────────────

    #[test]
    fn test_find_nodes_by_properties_across_layers() {
        let layered = build_test_layered();

        // Base has Alix (name="Alix", age=30) and Gus (name="Gus", age=25).
        let results = layered
            .find_nodes_by_properties(&[("name", Value::from("Alix")), ("age", Value::Int64(30))]);
        assert_eq!(results.len(), 1);

        // Add overlay node matching the same conditions.
        let mia = layered.create_node(&["Person"]);
        layered.set_node_property(mia, "name", Value::from("Alix"));
        layered.set_node_property(mia, "age", Value::Int64(30));

        let results_after = layered
            .find_nodes_by_properties(&[("name", Value::from("Alix")), ("age", Value::Int64(30))]);
        assert_eq!(results_after.len(), 2);
        assert!(results_after.contains(&mia));
    }

    #[test]
    fn test_find_nodes_by_properties_empty_conditions() {
        let layered = build_test_layered();

        // Empty conditions should return all node IDs.
        let results = layered.find_nodes_by_properties(&[]);
        assert_eq!(results.len(), layered.node_ids().len());
    }

    #[test]
    fn test_get_nodes_properties_selective_batch() {
        let layered = build_test_layered();

        let persons = layered.nodes_by_label("Person");
        let vincent = layered.create_node(&["Person"]);
        layered.set_node_property(vincent, "name", Value::from("Vincent"));
        layered.set_node_property(vincent, "age", Value::Int64(38));

        let all_ids: Vec<NodeId> = persons
            .iter()
            .copied()
            .chain(std::iter::once(vincent))
            .collect();

        let keys = vec![PropertyKey::new("name"), PropertyKey::new("age")];
        let batch = layered.get_nodes_properties_selective_batch(&all_ids, &keys);

        assert_eq!(batch.len(), all_ids.len());
        // Each map should contain only requested keys.
        for map in &batch {
            for key in map.keys() {
                assert!(
                    keys.contains(key),
                    "unexpected key {:?} in selective batch",
                    key
                );
            }
        }

        // Vincent's map should have both keys.
        let vincent_map = &batch[batch.len() - 1];
        assert_eq!(
            vincent_map.get(&PropertyKey::new("name")),
            Some(&Value::String(ArcStr::from("Vincent")))
        );
        assert_eq!(
            vincent_map.get(&PropertyKey::new("age")),
            Some(&Value::Int64(38))
        );
    }

    #[test]
    fn test_get_edges_properties_selective_batch() {
        let layered = build_test_layered();

        let persons = layered.nodes_by_label("Person");
        let edges_a = layered.edges_from(persons[0], Direction::Outgoing);
        let edges_b = layered.edges_from(persons[1], Direction::Outgoing);

        let edge_ids: Vec<EdgeId> = edges_a
            .iter()
            .chain(edges_b.iter())
            .map(|(_, eid)| *eid)
            .collect();

        let keys = vec![PropertyKey::new("since")];
        let batch = layered.get_edges_properties_selective_batch(&edge_ids, &keys);

        assert_eq!(batch.len(), edge_ids.len());
        for map in &batch {
            assert!(
                map.contains_key(&PropertyKey::new("since")),
                "each edge should have the 'since' property"
            );
        }
    }

    // ── L. Other uncovered methods ───────────────────────────────────

    #[test]
    fn test_estimate_label_cardinality() {
        let layered = build_test_layered();

        let person_card = layered.estimate_label_cardinality("Person");
        assert!(
            person_card >= 2.0,
            "should estimate at least 2 Person nodes, got {}",
            person_card
        );

        let city_card = layered.estimate_label_cardinality("City");
        assert!(
            city_card >= 1.0,
            "should estimate at least 1 City node, got {}",
            city_card
        );

        // Non-existent labels: the overlay's statistics may return a non-zero
        // default estimate, so we only check the call does not panic.
        let missing_card = layered.estimate_label_cardinality("NonExistent");
        assert!(
            missing_card >= 0.0,
            "cardinality for unknown label should be non-negative"
        );
    }

    #[test]
    fn test_estimate_avg_degree() {
        let layered = build_test_layered();

        let avg_out = layered.estimate_avg_degree("LIVES_IN", true);
        assert!(
            avg_out > 0.0,
            "average out-degree for LIVES_IN should be positive"
        );

        let avg_in = layered.estimate_avg_degree("LIVES_IN", false);
        assert!(
            avg_in > 0.0,
            "average in-degree for LIVES_IN should be positive"
        );
    }

    #[test]
    fn test_estimate_avg_degree_empty() {
        let store = LpgStore::new().unwrap();
        let compact = from_graph_store_preserving_ids(&store).unwrap();
        let layered = LayeredStore::new(compact, 0, 0).unwrap();

        let avg = layered.estimate_avg_degree("NONEXISTENT", true);
        assert_eq!(avg, 0.0, "empty store should have avg degree 0");
    }

    #[test]
    fn test_node_property_might_match() {
        let layered = build_test_layered();

        // "age" exists in the base, so might_match for Eq with an Int64 should be true
        // (zone maps allow Int64 values).
        let might = layered.node_property_might_match(
            &PropertyKey::new("age"),
            CompareOp::Eq,
            &Value::Int64(30),
        );
        assert!(might, "zone map should indicate age might match 30");
    }

    #[test]
    fn test_edge_property_might_match() {
        let layered = build_test_layered();

        let might = layered.edge_property_might_match(
            &PropertyKey::new("since"),
            CompareOp::Eq,
            &Value::Int64(2020),
        );
        assert!(might, "zone map should indicate since might match 2020");
    }
}
