//! GraphStore implementation for HybridStore (merged reads).
//!
//! Every read method follows the same merge protocol:
//!
//! 1. **Deleted?** -- if the entity was soft-deleted via the overlay, return `None`.
//! 2. **Overlay?** -- if the entity lives in the overlay (ID >= offset), delegate
//!    to the overlay `LpgStore`.
//! 3. **Compact?** -- fall back to the frozen `CompactStore`, then apply any
//!    property overrides the overlay may hold for that compact ID.
//!
//! Scan methods merge results from both stores, filtering out deleted compact
//! entities. Count methods combine compact and overlay counts minus deletions.

use std::sync::Arc;

use arcstr::ArcStr;
use grafeo_common::types::{EdgeId, EpochId, NodeId, PropertyKey, TransactionId, Value};
use grafeo_common::utils::hash::{FxHashMap, FxHashSet};

use super::{DirtySet, HybridStore};
use crate::graph::Direction;
use crate::graph::lpg::CompareOp;
use crate::graph::lpg::{Edge, Node};
use crate::graph::traits::GraphStore;
use crate::statistics::Statistics;

impl HybridStore {
    /// Reads a compact node and merges overlay property overrides onto it.
    fn compact_node_merged(&self, id: NodeId) -> Option<Node> {
        let compact = self.compact.read();
        let mut node = compact.get_node(id)?;
        drop(compact);

        // Fast path: clean entity — no overlay modifications
        if !self.is_node_dirty(id) {
            return Some(node);
        }

        // Apply overlay property overrides; skip Null tombstones (property removed).
        let overrides = self.overlay.get_node_properties_all(id);
        for (k, v) in overrides {
            if v == Value::Null {
                node.properties.remove(&k);
            } else {
                node.set_property(k, v);
            }
        }
        Some(node)
    }

    /// Reads a compact edge and merges overlay property overrides onto it.
    fn compact_edge_merged(&self, id: EdgeId) -> Option<Edge> {
        let compact = self.compact.read();
        let mut edge = compact.get_edge(id)?;
        drop(compact);

        // Fast path: clean entity
        if !self.is_edge_dirty(id) {
            return Some(edge);
        }

        // Apply overlay property overrides; skip Null tombstones (property removed).
        let overrides = self.overlay.get_edge_properties_all(id);
        for (k, v) in overrides {
            if v == Value::Null {
                edge.properties.remove(&k);
            } else {
                edge.set_property(k, v);
            }
        }
        Some(edge)
    }
}

impl GraphStore for HybridStore {
    // ---------------------------------------------------------------
    // Point lookups
    // ---------------------------------------------------------------

    fn get_node(&self, id: NodeId) -> Option<Node> {
        if self.is_node_deleted(id) {
            return None;
        }
        if self.is_compact_node_id(id) {
            self.compact_node_merged(id)
        } else {
            GraphStore::get_node(self.overlay.as_ref(), id)
        }
    }

    fn get_edge(&self, id: EdgeId) -> Option<Edge> {
        if self.is_edge_deleted(id) {
            return None;
        }
        if self.is_compact_edge_id(id) {
            self.compact_edge_merged(id)
        } else {
            GraphStore::get_edge(self.overlay.as_ref(), id)
        }
    }

    fn get_node_versioned(
        &self,
        id: NodeId,
        epoch: EpochId,
        transaction_id: TransactionId,
    ) -> Option<Node> {
        if self.is_node_deleted(id) {
            return None;
        }
        if self.is_compact_node_id(id) {
            // Compact data is always committed/visible
            self.compact_node_merged(id)
        } else {
            GraphStore::get_node_versioned(self.overlay.as_ref(), id, epoch, transaction_id)
        }
    }

    fn get_edge_versioned(
        &self,
        id: EdgeId,
        epoch: EpochId,
        transaction_id: TransactionId,
    ) -> Option<Edge> {
        if self.is_edge_deleted(id) {
            return None;
        }
        if self.is_compact_edge_id(id) {
            self.compact_edge_merged(id)
        } else {
            GraphStore::get_edge_versioned(self.overlay.as_ref(), id, epoch, transaction_id)
        }
    }

    fn get_node_at_epoch(&self, id: NodeId, epoch: EpochId) -> Option<Node> {
        if self.is_node_deleted(id) {
            return None;
        }
        if self.is_compact_node_id(id) {
            self.compact_node_merged(id)
        } else {
            GraphStore::get_node_at_epoch(self.overlay.as_ref(), id, epoch)
        }
    }

    fn get_edge_at_epoch(&self, id: EdgeId, epoch: EpochId) -> Option<Edge> {
        if self.is_edge_deleted(id) {
            return None;
        }
        if self.is_compact_edge_id(id) {
            self.compact_edge_merged(id)
        } else {
            GraphStore::get_edge_at_epoch(self.overlay.as_ref(), id, epoch)
        }
    }

    // ---------------------------------------------------------------
    // Property access (fast path)
    // ---------------------------------------------------------------

    fn get_node_property(&self, id: NodeId, key: &PropertyKey) -> Option<Value> {
        if self.is_node_deleted(id) {
            return None;
        }
        if self.is_compact_node_id(id) {
            if self.is_node_dirty(id) {
                // Overlay might have an override or tombstone
                if let Some(v) = self.overlay.get_node_property(id, key) {
                    return if v == Value::Null { None } else { Some(v) };
                }
            }
            self.compact.read().get_node_property(id, key)
        } else {
            self.overlay.get_node_property(id, key)
        }
    }

    fn get_edge_property(&self, id: EdgeId, key: &PropertyKey) -> Option<Value> {
        if self.is_edge_deleted(id) {
            return None;
        }
        if self.is_compact_edge_id(id) {
            if self.is_edge_dirty(id)
                && let Some(v) = self.overlay.get_edge_property(id, key)
            {
                return if v == Value::Null { None } else { Some(v) };
            }
            self.compact.read().get_edge_property(id, key)
        } else {
            self.overlay.get_edge_property(id, key)
        }
    }

    fn get_node_property_batch(&self, ids: &[NodeId], key: &PropertyKey) -> Vec<Option<Value>> {
        ids.iter()
            .map(|&id| self.get_node_property(id, key))
            .collect()
    }

    fn get_nodes_properties_batch(&self, ids: &[NodeId]) -> Vec<FxHashMap<PropertyKey, Value>> {
        ids.iter()
            .map(|&id| {
                if self.is_node_deleted(id) {
                    return FxHashMap::default();
                }
                if self.is_compact_node_id(id) {
                    let compact = self.compact.read();
                    let mut props: FxHashMap<PropertyKey, Value> = compact
                        .get_node(id)
                        .map(|n| n.properties.into_iter().collect())
                        .unwrap_or_default();
                    drop(compact);
                    // Merge overlay overrides; Null tombstones mean removal.
                    for (k, v) in self.overlay.get_node_properties_all(id) {
                        if v == Value::Null {
                            props.remove(&k);
                        } else {
                            props.insert(k, v);
                        }
                    }
                    props
                } else {
                    GraphStore::get_node(self.overlay.as_ref(), id)
                        .map(|n| n.properties.into_iter().collect())
                        .unwrap_or_default()
                }
            })
            .collect()
    }

    fn get_nodes_properties_selective_batch(
        &self,
        ids: &[NodeId],
        keys: &[PropertyKey],
    ) -> Vec<FxHashMap<PropertyKey, Value>> {
        ids.iter()
            .map(|&id| {
                let mut map = FxHashMap::default();
                for key in keys {
                    if let Some(v) = self.get_node_property(id, key) {
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
            .map(|&id| {
                let mut map = FxHashMap::default();
                for key in keys {
                    if let Some(v) = self.get_edge_property(id, key) {
                        map.insert(key.clone(), v);
                    }
                }
                map
            })
            .collect()
    }

    // ---------------------------------------------------------------
    // Traversal
    // ---------------------------------------------------------------

    fn neighbors(&self, node: NodeId, direction: Direction) -> Vec<NodeId> {
        let deleted_nodes = self.deleted_nodes.read();

        let mut result = Vec::new();

        // Compact edges (use edges_from so we can filter deleted edges too)
        if self.is_compact_node_id(node) {
            let deleted_edges = self.deleted_edges.read();
            let compact = self.compact.read();
            for (target, eid) in compact.edges_from(node, direction) {
                if !deleted_edges.contains(&eid) && !deleted_nodes.contains(&target) {
                    result.push(target);
                }
            }
        }

        // Overlay edges: overlay handles its own MVCC; just exclude deleted
        // compact nodes that might appear as targets.
        for nid in GraphStore::neighbors(self.overlay.as_ref(), node, direction) {
            if !deleted_nodes.contains(&nid) {
                result.push(nid);
            }
        }

        // Deduplicate (compact and overlay may produce the same neighbor)
        result.sort_unstable();
        result.dedup();
        result
    }

    fn edges_from(&self, node: NodeId, direction: Direction) -> Vec<(NodeId, EdgeId)> {
        let deleted_nodes = self.deleted_nodes.read();
        let deleted_edges = self.deleted_edges.read();

        let mut result = Vec::new();

        // Compact edges
        if self.is_compact_node_id(node) {
            let compact = self.compact.read();
            for (target, eid) in compact.edges_from(node, direction) {
                if !deleted_edges.contains(&eid) && !deleted_nodes.contains(&target) {
                    result.push((target, eid));
                }
            }
        }

        // Overlay edges
        for (target, eid) in GraphStore::edges_from(self.overlay.as_ref(), node, direction) {
            if !deleted_edges.contains(&eid) && !deleted_nodes.contains(&target) {
                result.push((target, eid));
            }
        }

        result
    }

    fn out_degree(&self, node: NodeId) -> usize {
        // Count edges, filtering deleted
        self.edges_from(node, Direction::Outgoing).len()
    }

    fn in_degree(&self, node: NodeId) -> usize {
        self.edges_from(node, Direction::Incoming).len()
    }

    fn has_backward_adjacency(&self) -> bool {
        let compact = self.compact.read();
        compact.has_backward_adjacency() || self.overlay.has_backward_adjacency()
    }

    // ---------------------------------------------------------------
    // Scans
    // ---------------------------------------------------------------

    fn node_ids(&self) -> Vec<NodeId> {
        let deleted = self.deleted_nodes.read();
        let mut ids = Vec::new();

        // Compact node IDs
        {
            let compact = self.compact.read();
            for id in compact.node_ids() {
                if !deleted.contains(&id) {
                    ids.push(id);
                }
            }
        }

        // Overlay node IDs
        for id in GraphStore::node_ids(self.overlay.as_ref()) {
            ids.push(id);
        }

        ids.sort_unstable();
        ids
    }

    fn all_node_ids(&self) -> Vec<NodeId> {
        // Same as node_ids for hybrid: compact is always committed
        self.node_ids()
    }

    fn nodes_by_label(&self, label: &str) -> Vec<NodeId> {
        let deleted = self.deleted_nodes.read();
        let mut ids = Vec::new();

        // Compact
        {
            let compact = self.compact.read();
            for id in compact.nodes_by_label(label) {
                if !deleted.contains(&id) {
                    ids.push(id);
                }
            }
        }

        // Overlay
        for id in GraphStore::nodes_by_label(self.overlay.as_ref(), label) {
            ids.push(id);
        }

        ids.sort_unstable();
        ids
    }

    fn node_count(&self) -> usize {
        let deleted_count = self.deleted_nodes.read().len();
        let overlay_count = GraphStore::node_count(self.overlay.as_ref());
        self.compact_node_count.load(std::sync::atomic::Ordering::Relaxed) - deleted_count + overlay_count
    }

    fn edge_count(&self) -> usize {
        let deleted_count = self.deleted_edges.read().len();
        let overlay_count = GraphStore::edge_count(self.overlay.as_ref());
        self.compact_edge_count.load(std::sync::atomic::Ordering::Relaxed) - deleted_count + overlay_count
    }

    // ---------------------------------------------------------------
    // Entity metadata
    // ---------------------------------------------------------------

    fn edge_type(&self, id: EdgeId) -> Option<ArcStr> {
        if self.is_edge_deleted(id) {
            return None;
        }
        if self.is_compact_edge_id(id) {
            let compact = self.compact.read();
            compact.edge_type(id)
        } else {
            GraphStore::edge_type(self.overlay.as_ref(), id)
        }
    }

    fn edge_type_versioned(
        &self,
        id: EdgeId,
        epoch: EpochId,
        transaction_id: TransactionId,
    ) -> Option<ArcStr> {
        if self.is_edge_deleted(id) {
            return None;
        }
        if self.is_compact_edge_id(id) {
            let compact = self.compact.read();
            compact.edge_type(id)
        } else {
            GraphStore::edge_type_versioned(self.overlay.as_ref(), id, epoch, transaction_id)
        }
    }

    // ---------------------------------------------------------------
    // Index introspection
    // ---------------------------------------------------------------

    fn has_property_index(&self, property: &str) -> bool {
        self.overlay.has_property_index(property)
    }

    // ---------------------------------------------------------------
    // Filtered search
    // ---------------------------------------------------------------

    fn find_nodes_by_property(&self, property: &str, value: &Value) -> Vec<NodeId> {
        let key = PropertyKey::new(property);
        let deleted = self.deleted_nodes.read();
        let dirty = self.dirty_nodes.read(); // held for entire scan — consistent snapshot
        let overlay = &self.overlay;
        let column_dirty = dirty.is_column_dirty(&key);
        let mut results = Vec::new();

        // Compact scan
        {
            let compact = self.compact.read();
            for id in compact.find_nodes_by_property(property, value) {
                if !deleted.contains(&id) {
                    if column_dirty && DirtySet::id_is_dirty_node(&dirty, id) {
                        if let Some(override_val) = overlay.get_node_property(id, &key) {
                            if &override_val == value {
                                results.push(id);
                            }
                        } else {
                            results.push(id);
                        }
                    } else {
                        results.push(id);
                    }
                }
            }
        }

        // Compact nodes with overlay overrides matching the NEW value
        // (only needed if this column has any overlay modifications)
        if column_dirty {
            let mut seen = FxHashSet::default();
            for id in &results {
                seen.insert(*id);
            }
            let target = value.clone();
            for id in overlay.node_ids_with_property_matching(&key, |v| v == &target) {
                if self.is_compact_node_id(id) && !deleted.contains(&id) && !seen.contains(&id) {
                    results.push(id);
                }
            }
        }

        // Overlay nodes (created in overlay, not compact)
        for id in GraphStore::find_nodes_by_property(overlay.as_ref(), property, value) {
            results.push(id);
        }

        results
    }

    fn find_nodes_by_properties(&self, conditions: &[(&str, Value)]) -> Vec<NodeId> {
        if conditions.is_empty() {
            return self.node_ids();
        }

        let deleted = self.deleted_nodes.read();
        let dirty = self.dirty_nodes.read();
        let overlay = &self.overlay;
        let any_column_dirty = conditions
            .iter()
            .any(|(prop, _)| dirty.is_column_dirty(&PropertyKey::new(*prop)));
        let mut results = Vec::new();
        let mut seen = FxHashSet::default();

        // Compact — post-filter only dirty entities against overlay overrides
        {
            let compact = self.compact.read();
            for id in compact.find_nodes_by_properties(conditions) {
                if !deleted.contains(&id) {
                    if any_column_dirty && DirtySet::id_is_dirty_node(&dirty, id) {
                        let mut still_matches = true;
                        for (prop, val) in conditions {
                            let key = PropertyKey::new(*prop);
                            if let Some(override_val) = overlay.get_node_property(id, &key)
                                && &override_val != val
                            {
                                still_matches = false;
                                break;
                            }
                        }
                        if still_matches {
                            results.push(id);
                            seen.insert(id);
                        }
                    } else {
                        results.push(id);
                        seen.insert(id);
                    }
                }
            }
        }

        // Compact nodes with overlay overrides (only if any condition column is dirty)
        if any_column_dirty
            && let Some((first_prop, first_val)) = conditions.first()
        {
                let first_key = PropertyKey::new(*first_prop);
                let target = first_val.clone();
                for id in overlay.node_ids_with_property_matching(&first_key, |v| v == &target) {
                    if self.is_compact_node_id(id) && !deleted.contains(&id) && !seen.contains(&id)
                    {
                        let mut all_match = true;
                        for (prop, val) in &conditions[1..] {
                            let key = PropertyKey::new(*prop);
                            let effective = overlay
                                .get_node_property(id, &key)
                                .or_else(|| self.compact.read().get_node_property(id, &key));
                            if effective.as_ref() != Some(val) {
                                all_match = false;
                                break;
                            }
                        }
                        if all_match {
                            results.push(id);
                        }
                    }
                }
        }
        // Overlay nodes
        for id in GraphStore::find_nodes_by_properties(overlay.as_ref(), conditions) {
            results.push(id);
        }

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
        let key = PropertyKey::new(property);
        let deleted = self.deleted_nodes.read();
        let dirty = self.dirty_nodes.read();
        let overlay = &self.overlay;
        let column_dirty = dirty.is_column_dirty(&key);
        let mut results = Vec::new();

        // Compact scan — only check overlay for dirty entities in dirty columns
        {
            let compact = self.compact.read();
            for id in compact.find_nodes_in_range(property, min, max, min_inclusive, max_inclusive) {
                if !deleted.contains(&id) {
                    if column_dirty && DirtySet::id_is_dirty_node(&dirty, id) {
                        if let Some(override_val) = overlay.get_node_property(id, &key) {
                            if crate::graph::lpg::value_in_range(
                                &override_val,
                                min,
                                max,
                                min_inclusive,
                                max_inclusive,
                            ) {
                                results.push(id);
                            }
                        } else {
                            results.push(id);
                        }
                    } else {
                        results.push(id);
                    }
                }
            }
        }

        // Compact nodes with overlay overrides in the NEW range
        if column_dirty {
            let mut seen = FxHashSet::default();
            for id in &results {
                seen.insert(*id);
            }
            let min_c = min.cloned();
            let max_c = max.cloned();
            for id in overlay.node_ids_with_property_matching(&key, |v| {
                crate::graph::lpg::value_in_range(
                    v,
                    min_c.as_ref(),
                    max_c.as_ref(),
                    min_inclusive,
                    max_inclusive,
                )
            }) {
                if self.is_compact_node_id(id) && !deleted.contains(&id) && !seen.contains(&id) {
                    results.push(id);
                }
            }
        }

        // Overlay nodes
        for id in GraphStore::find_nodes_in_range(
            overlay.as_ref(),
            property,
            min,
            max,
            min_inclusive,
            max_inclusive,
        ) {
            results.push(id);
        }

        results
    }

    // ---------------------------------------------------------------
    // Zone maps (skip pruning)
    // ---------------------------------------------------------------

    fn node_property_might_match(
        &self,
        property: &PropertyKey,
        op: CompareOp,
        value: &Value,
    ) -> bool {
        let compact = self.compact.read();
        compact.node_property_might_match(property, op, value)
            || GraphStore::node_property_might_match(self.overlay.as_ref(), property, op, value)
    }

    fn edge_property_might_match(
        &self,
        property: &PropertyKey,
        op: CompareOp,
        value: &Value,
    ) -> bool {
        let compact = self.compact.read();
        compact.edge_property_might_match(property, op, value)
            || GraphStore::edge_property_might_match(self.overlay.as_ref(), property, op, value)
    }

    // ---------------------------------------------------------------
    // Statistics
    // ---------------------------------------------------------------

    fn statistics(&self) -> Arc<Statistics> {
        // Return overlay stats for now; merged stats is a follow-up.
        GraphStore::statistics(self.overlay.as_ref())
    }

    fn estimate_label_cardinality(&self, label: &str) -> f64 {
        let compact = self.compact.read();
        let compact_card = compact.estimate_label_cardinality(label);
        let overlay_card =
            GraphStore::estimate_label_cardinality(self.overlay.as_ref(), label);
        compact_card + overlay_card
    }

    fn estimate_avg_degree(&self, edge_type: &str, outgoing: bool) -> f64 {
        let compact = self.compact.read();
        let compact_deg = compact.estimate_avg_degree(edge_type, outgoing);
        let overlay_deg =
            GraphStore::estimate_avg_degree(self.overlay.as_ref(), edge_type, outgoing);
        // Weighted average would be more accurate, but simple max is a
        // reasonable conservative estimate for the CBO.
        compact_deg.max(overlay_deg)
    }

    // ---------------------------------------------------------------
    // Epoch
    // ---------------------------------------------------------------

    fn current_epoch(&self) -> EpochId {
        self.overlay.current_epoch()
    }

    // ---------------------------------------------------------------
    // Schema introspection
    // ---------------------------------------------------------------

    fn all_labels(&self) -> Vec<String> {
        let mut labels: Vec<String> = {
            let compact = self.compact.read();
            compact.all_labels()
        };
        for l in GraphStore::all_labels(self.overlay.as_ref()) {
            if !labels.contains(&l) {
                labels.push(l);
            }
        }
        labels
    }

    fn all_edge_types(&self) -> Vec<String> {
        let mut types: Vec<String> = {
            let compact = self.compact.read();
            compact.all_edge_types()
        };
        for t in GraphStore::all_edge_types(self.overlay.as_ref()) {
            if !types.contains(&t) {
                types.push(t);
            }
        }
        types
    }

    fn all_property_keys(&self) -> Vec<String> {
        let mut keys: Vec<String> = {
            let compact = self.compact.read();
            compact.all_property_keys()
        };
        for k in GraphStore::all_property_keys(self.overlay.as_ref()) {
            if !keys.contains(&k) {
                keys.push(k);
            }
        }
        keys
    }

    // ---------------------------------------------------------------
    // Visibility checks
    // ---------------------------------------------------------------

    fn is_node_visible_at_epoch(&self, id: NodeId, epoch: EpochId) -> bool {
        if self.is_node_deleted(id) {
            return false;
        }
        if self.is_compact_node_id(id) {
            // Compact nodes are always visible (committed data)
            let compact = self.compact.read();
            compact.get_node(id).is_some()
        } else {
            GraphStore::is_node_visible_at_epoch(self.overlay.as_ref(), id, epoch)
        }
    }

    fn is_node_visible_versioned(
        &self,
        id: NodeId,
        epoch: EpochId,
        transaction_id: TransactionId,
    ) -> bool {
        if self.is_node_deleted(id) {
            return false;
        }
        if self.is_compact_node_id(id) {
            let compact = self.compact.read();
            compact.get_node(id).is_some()
        } else {
            GraphStore::is_node_visible_versioned(
                self.overlay.as_ref(),
                id,
                epoch,
                transaction_id,
            )
        }
    }

    fn is_edge_visible_at_epoch(&self, id: EdgeId, epoch: EpochId) -> bool {
        if self.is_edge_deleted(id) {
            return false;
        }
        if self.is_compact_edge_id(id) {
            let compact = self.compact.read();
            compact.get_edge(id).is_some()
        } else {
            GraphStore::is_edge_visible_at_epoch(self.overlay.as_ref(), id, epoch)
        }
    }

    fn is_edge_visible_versioned(
        &self,
        id: EdgeId,
        epoch: EpochId,
        transaction_id: TransactionId,
    ) -> bool {
        if self.is_edge_deleted(id) {
            return false;
        }
        if self.is_compact_edge_id(id) {
            let compact = self.compact.read();
            compact.get_edge(id).is_some()
        } else {
            GraphStore::is_edge_visible_versioned(
                self.overlay.as_ref(),
                id,
                epoch,
                transaction_id,
            )
        }
    }

    // ---------------------------------------------------------------
    // History
    // ---------------------------------------------------------------

    fn get_node_history(&self, id: NodeId) -> Vec<(EpochId, Option<EpochId>, Node)> {
        if self.is_compact_node_id(id) {
            // Compact store has no version history
            Vec::new()
        } else {
            GraphStore::get_node_history(self.overlay.as_ref(), id)
        }
    }

    fn get_edge_history(&self, id: EdgeId) -> Vec<(EpochId, Option<EpochId>, Edge)> {
        if self.is_compact_edge_id(id) {
            Vec::new()
        } else {
            GraphStore::get_edge_history(self.overlay.as_ref(), id)
        }
    }
}
