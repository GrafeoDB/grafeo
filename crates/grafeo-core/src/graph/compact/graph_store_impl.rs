//! [`GraphStore`](crate::graph::GraphStore) trait implementation for [`CompactStore`].
//!
//! All read operations (point lookups, traversal, scans, property access,
//! filtered search, statistics, and visibility checks) are implemented here.
//! The store is read-only: all data comes from immutable columnar tables.

use std::sync::Arc;

use arcstr::ArcStr;
use grafeo_common::types::{EdgeId, EpochId, NodeId, PropertyKey, TransactionId, Value};
use grafeo_common::utils::hash::{FxHashMap, FxHashSet};

use super::CompactStore;
use super::id::encode_node_id;
use crate::graph::Direction;
use crate::graph::lpg::CompareOp;
use crate::graph::lpg::{Edge, Node};
use crate::graph::traits::{GraphStore, GraphStoreSearch};
use crate::statistics::Statistics;

impl GraphStore for CompactStore {
    fn get_node(&self, id: NodeId) -> Option<Node> {
        let (table_id, offset) = self.resolve_node(id)?;
        let nt = self.resolve_node_table(table_id)?;
        let row = usize::try_from(offset).ok()?;
        if row >= nt.len() {
            return None;
        }

        let mut node = Node::new(id);
        node.add_label(nt.label());
        let props = nt.get_all_properties(row);
        for (k, v) in props {
            node.set_property(k, v);
        }
        Some(node)
    }

    fn get_edge(&self, id: EdgeId) -> Option<Edge> {
        let (rel_table_id, csr_position) = self.resolve_edge(id)?;
        let rt = self.resolve_rel_table(rel_table_id)?;
        let pos = u32::try_from(csr_position).ok()?;

        let src_compact = rt.source_node_id(pos)?;
        let dst_compact = rt.dest_node_id(pos)?;
        let src = self.to_original_node_id(src_compact);
        let dst = self.to_original_node_id(dst_compact);
        let edge_type = rt.edge_type().clone();

        let mut edge = Edge::new(id, src, dst, edge_type);
        let props = rt.get_all_edge_properties(pos as usize);
        for (k, v) in props {
            edge.set_property(k, v);
        }
        Some(edge)
    }

    fn get_node_versioned(
        &self,
        id: NodeId,
        _epoch: EpochId,
        _transaction_id: TransactionId,
    ) -> Option<Node> {
        self.get_node(id)
    }

    fn get_edge_versioned(
        &self,
        id: EdgeId,
        _epoch: EpochId,
        _transaction_id: TransactionId,
    ) -> Option<Edge> {
        self.get_edge(id)
    }

    fn get_node_at_epoch(&self, id: NodeId, _epoch: EpochId) -> Option<Node> {
        self.get_node(id)
    }

    fn get_edge_at_epoch(&self, id: EdgeId, _epoch: EpochId) -> Option<Edge> {
        self.get_edge(id)
    }

    fn get_node_property(&self, id: NodeId, key: &PropertyKey) -> Option<Value> {
        let (table_id, offset) = self.resolve_node(id)?;
        let nt = self.resolve_node_table(table_id)?;
        let row = usize::try_from(offset).ok()?;
        nt.get_property(row, key)
    }

    fn get_edge_property(&self, id: EdgeId, key: &PropertyKey) -> Option<Value> {
        let (rel_table_id, csr_position) = self.resolve_edge(id)?;
        let rt = self.resolve_rel_table(rel_table_id)?;
        let row = usize::try_from(csr_position).ok()?;
        rt.get_edge_property(row, key)
    }

    fn get_node_property_batch(&self, ids: &[NodeId], key: &PropertyKey) -> Vec<Option<Value>> {
        ids.iter()
            .map(|id| self.get_node_property(*id, key))
            .collect()
    }

    // The columns are in memory or in a mapped file: no read can fail.
    fn try_get_node_property_batch(
        &self,
        ids: &[NodeId],
        key: &PropertyKey,
    ) -> grafeo_common::utils::error::Result<Vec<Option<Value>>> {
        Ok(self.get_node_property_batch(ids, key))
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
        let Some((node_table_id, node_offset)) = self.resolve_node(node) else {
            return Vec::new();
        };
        let Ok(offset) = u32::try_from(node_offset) else {
            return Vec::new();
        };
        self.collect_edges(node_table_id, offset, direction)
            .into_iter()
            .map(|(target, _)| target)
            .collect()
    }

    fn edges_from(&self, node: NodeId, direction: Direction) -> Vec<(NodeId, EdgeId)> {
        let Some((node_table_id, node_offset)) = self.resolve_node(node) else {
            return Vec::new();
        };
        let Ok(offset) = u32::try_from(node_offset) else {
            return Vec::new();
        };
        self.collect_edges(node_table_id, offset, direction)
    }

    fn out_degree(&self, node: NodeId) -> usize {
        let Some((node_table_id, node_offset)) = self.resolve_node(node) else {
            return 0;
        };
        let Ok(offset) = u32::try_from(node_offset) else {
            return 0;
        };
        let mut degree = 0;
        if let Some(rel_ids) = self.src_rel_table_ids.get(node_table_id as usize) {
            for &rel_id in rel_ids {
                let rt = &self.rel_tables_by_id[rel_id as usize];
                degree += rt.out_degree(offset);
            }
        }
        degree
    }

    fn in_degree(&self, node: NodeId) -> usize {
        let Some((node_table_id, node_offset)) = self.resolve_node(node) else {
            return 0;
        };
        let Ok(offset) = u32::try_from(node_offset) else {
            return 0;
        };
        let mut degree = 0;
        if let Some(rel_ids) = self.dst_rel_table_ids.get(node_table_id as usize) {
            for &rel_id in rel_ids {
                let rt = &self.rel_tables_by_id[rel_id as usize];
                if let Some(d) = rt.in_degree(offset) {
                    degree += d;
                }
            }
        }
        degree
    }

    fn has_backward_adjacency(&self) -> bool {
        self.rel_tables_by_id.iter().any(|rt| rt.has_backward())
    }

    fn node_ids(&self) -> Vec<NodeId> {
        if let Some(ref map) = self.node_id_map {
            let mut ids: Vec<NodeId> = map.keys().copied().collect();
            ids.sort_unstable();
            ids
        } else {
            let mut ids = Vec::new();
            for nt in &self.node_tables_by_id {
                ids.extend(nt.node_ids());
            }
            ids.sort_unstable();
            ids
        }
    }

    fn nodes_by_label(&self, label: &str) -> Vec<NodeId> {
        let compact_ids = self
            .label_to_table_id
            .get(label)
            .map(|&tid| self.node_tables_by_id[tid as usize].node_ids())
            .unwrap_or_default();
        if self.preserves_ids() {
            compact_ids
                .into_iter()
                .map(|cid| self.to_original_node_id(cid))
                .collect()
        } else {
            compact_ids
        }
    }

    fn nodes_by_label_count(&self, label: &str) -> usize {
        // Row count is preserved across the compact->original ID mapping, so
        // NodeTable::len() is authoritative whether preserves_ids() is set or not.
        self.label_to_table_id
            .get(label)
            .map_or(0, |&tid| self.node_tables_by_id[tid as usize].len())
    }

    fn node_count(&self) -> usize {
        self.node_tables_by_id.iter().map(|nt| nt.len()).sum()
    }

    fn edge_count(&self) -> usize {
        self.rel_tables_by_id.iter().map(|rt| rt.num_edges()).sum()
    }

    fn edge_type(&self, id: EdgeId) -> Option<ArcStr> {
        let (rel_table_id, _) = self.resolve_edge(id)?;
        self.rel_table_id_to_type
            .get(rel_table_id as usize)
            .cloned()
    }

    /// The store is immutable: no transaction has edges of its own in it.
    fn edge_type_versioned(&self, id: EdgeId, _: EpochId, _: TransactionId) -> Option<ArcStr> {
        self.edge_type(id)
    }

    fn find_nodes_by_property(&self, property: &str, value: &Value) -> Vec<NodeId> {
        let key = PropertyKey::new(property);
        let mut results = Vec::new();
        for nt in &self.node_tables_by_id {
            if let Some(zm) = nt.zone_map(&key)
                && !zm.might_match(CompareOp::Eq, value)
            {
                continue;
            }
            if let Some(col) = nt.column(&key) {
                let table_id = nt.table_id();
                for offset in col.find_eq(value) {
                    let compact_id = encode_node_id(table_id, offset as u64);
                    results.push(self.to_original_node_id(compact_id));
                }
            }
        }
        results
    }

    fn find_nodes_by_properties(&self, conditions: &[(&str, Value)]) -> Vec<NodeId> {
        if conditions.is_empty() {
            return self.node_ids();
        }

        let (first_prop, first_val) = &conditions[0];
        let candidates = self.find_nodes_by_property(first_prop, first_val);

        if conditions.len() == 1 {
            return candidates;
        }

        candidates
            .into_iter()
            .filter(|nid| {
                for (prop, val) in &conditions[1..] {
                    let key = PropertyKey::new(*prop);
                    match self.get_node_property(*nid, &key) {
                        Some(ref v) if v == val => {}
                        _ => return false,
                    }
                }
                true
            })
            .collect()
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
        let mut results = Vec::new();

        for nt in &self.node_tables_by_id {
            if let Some(zm) = nt.zone_map(&key) {
                if let Some(min_val) = min {
                    let op = if min_inclusive {
                        CompareOp::Ge
                    } else {
                        CompareOp::Gt
                    };
                    if !zm.might_match(op, min_val) {
                        continue;
                    }
                }
                if let Some(max_val) = max {
                    let op = if max_inclusive {
                        CompareOp::Le
                    } else {
                        CompareOp::Lt
                    };
                    if !zm.might_match(op, max_val) {
                        continue;
                    }
                }
            }
            if let Some(col) = nt.column(&key) {
                let table_id = nt.table_id();
                for offset in col.find_in_range(min, max, min_inclusive, max_inclusive) {
                    let compact_id = encode_node_id(table_id, offset as u64);
                    results.push(self.to_original_node_id(compact_id));
                }
            }
        }

        results
    }

    fn node_property_might_match(
        &self,
        property: &PropertyKey,
        op: CompareOp,
        value: &Value,
    ) -> bool {
        let mut might_match = false;
        for nt in &self.node_tables_by_id {
            match nt.zone_map(property) {
                Some(zm) => {
                    if zm.might_match(op, value) {
                        return true;
                    }
                }
                None => {
                    // No stats for this property in this table: conservatively assume match
                    might_match = true;
                }
            }
        }
        might_match
    }

    fn edge_property_might_match(
        &self,
        _property: &PropertyKey,
        _op: CompareOp,
        _value: &Value,
    ) -> bool {
        // Conservative: no zone maps on edge properties
        true
    }

    fn statistics(&self) -> Arc<Statistics> {
        Arc::clone(&self.statistics)
    }

    fn estimate_label_cardinality(&self, label: &str) -> f64 {
        self.label_to_table_id
            .get(label)
            .and_then(|&tid| self.node_tables_by_id.get(tid as usize))
            .map_or(0.0, |nt| nt.len() as f64)
    }

    fn estimate_avg_degree(&self, edge_type: &str, outgoing: bool) -> f64 {
        let Some(rids) = self.edge_type_to_rel_id.get(edge_type) else {
            return 0.0;
        };
        let mut total_edges: usize = 0;
        let mut seen_tables = FxHashSet::default();
        for &rid in rids {
            let Some(rt) = self.rel_tables_by_id.get(rid as usize) else {
                continue;
            };
            total_edges += rt.num_edges();
            let table_id = if outgoing {
                rt.src_table_id()
            } else {
                rt.dst_table_id()
            };
            seen_tables.insert(table_id);
        }
        let total_nodes: usize = seen_tables
            .iter()
            .map(|&tid| self.resolve_node_table(tid).map_or(1, |nt| nt.len().max(1)))
            .sum();
        if total_nodes == 0 {
            return 0.0;
        }
        total_edges as f64 / total_nodes as f64
    }

    fn current_epoch(&self) -> EpochId {
        EpochId(1)
    }

    fn all_labels(&self) -> Vec<String> {
        self.table_id_to_label
            .iter()
            .map(|s| s.to_string())
            .collect()
    }

    fn all_edge_types(&self) -> Vec<String> {
        self.edge_type_to_rel_id
            .keys()
            .map(|s| s.to_string())
            .collect()
    }

    fn all_property_keys(&self) -> Vec<String> {
        let mut keys = FxHashSet::<String>::default();

        for nt in &self.node_tables_by_id {
            for pk in nt.property_keys() {
                keys.insert(pk.as_str().to_string());
            }
        }

        for rt in &self.rel_tables_by_id {
            for pk in rt.property_keys() {
                keys.insert(pk.as_str().to_string());
            }
        }

        keys.into_iter().collect()
    }

    fn get_node_history(&self, _id: NodeId) -> Vec<(EpochId, Option<EpochId>, Node)> {
        Vec::new()
    }

    fn get_edge_history(&self, _id: EdgeId) -> Vec<(EpochId, Option<EpochId>, Edge)> {
        Vec::new()
    }
}

impl GraphStoreSearch for CompactStore {
    fn find_nodes_in_range_iter<'a>(
        &'a self,
        property: &'a str,
        min: Option<&'a Value>,
        max: Option<&'a Value>,
        min_inclusive: bool,
        max_inclusive: bool,
    ) -> Box<dyn Iterator<Item = NodeId> + 'a> {
        let key = PropertyKey::new(property);

        let per_table = self.node_tables_by_id.iter().filter_map(move |nt| {
            // Whole-table skip via per-label zone map (existing behavior).
            if let Some(zm) = nt.zone_map(&key) {
                if let Some(min_val) = min {
                    let op = if min_inclusive {
                        CompareOp::Ge
                    } else {
                        CompareOp::Gt
                    };
                    if !zm.might_match(op, min_val) {
                        return None;
                    }
                }
                if let Some(max_val) = max {
                    let op = if max_inclusive {
                        CompareOp::Le
                    } else {
                        CompareOp::Lt
                    };
                    if !zm.might_match(op, max_val) {
                        return None;
                    }
                }
            }

            let col = nt.column(&key)?;
            let block_zones = nt.block_zone_maps_for(&key);
            let table_id = nt.table_id();
            let store = self;
            // reason: usize → u64 fits on every supported target (row count
            // bounded by u32::MAX per the section format).
            #[allow(clippy::cast_possible_truncation)]
            let iter = col
                .range_iter(block_zones, min, max, min_inclusive, max_inclusive)
                .map(move |offset| {
                    let compact_id = encode_node_id(table_id, offset as u64);
                    store.to_original_node_id(compact_id)
                });
            Some(iter)
        });

        Box::new(per_table.flatten())
    }
}
