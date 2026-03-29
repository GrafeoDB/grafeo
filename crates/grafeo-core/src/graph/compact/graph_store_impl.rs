//! [`GraphStore`] trait implementation for [`CompactStore`].
//!
//! All read operations (point lookups, traversal, scans, property access,
//! filtered search, statistics, and visibility checks) are implemented here.
//! Each method merges snapshot data from columnar node/rel tables with
//! runtime mutations stored in the [`DeltaBuffer`](super::delta::DeltaBuffer).

use std::sync::Arc;

use arcstr::ArcStr;
use grafeo_common::types::{EdgeId, EpochId, NodeId, PropertyKey, TransactionId, Value};
use grafeo_common::utils::hash::{FxHashMap, FxHashSet};

use super::CompactStore;
use super::id::{decode_edge_id, decode_node_id, encode_node_id};
use crate::graph::Direction;
use crate::graph::lpg::CompareOp;
use crate::graph::lpg::{Edge, Node};
use crate::graph::traits::GraphStore;
use crate::statistics::Statistics;

use std::sync::atomic::Ordering as AtomicOrdering;

impl GraphStore for CompactStore {
    fn get_node(&self, id: NodeId) -> Option<Node> {
        let (table_id, offset) = decode_node_id(id);
        let has_delta = self.delta_has_content.load(AtomicOrdering::Relaxed);

        // Delta-created node — only possible when delta has content.
        if Self::is_delta_node_id(table_id) {
            if !has_delta {
                return None;
            }
            let delta = self.delta.lock();
            let dn = delta.get_delta_node(id)?;
            if delta.is_node_deleted(id) {
                return None;
            }
            let mut node = Node::new(id);
            for l in &dn.labels {
                node.add_label(l.clone());
            }
            for (k, v) in &dn.properties {
                node.set_property(k.clone(), v.clone());
            }
            return Some(node);
        }

        let nt = self.resolve_node_table(table_id)?;
        if offset as usize >= nt.len() {
            return None;
        }

        // Fast path: no delta mutations — skip the lock entirely.
        if !has_delta {
            let mut node = Node::new(id);
            node.add_label(nt.label());
            let props = nt.get_all_properties(offset as usize);
            for (k, v) in props {
                node.set_property(k, v);
            }
            return Some(node);
        }

        let delta = self.delta.lock();

        // Snapshot node — check if deleted
        if delta.is_node_deleted(id) {
            return None;
        }

        let mut node = Node::new(id);

        // Label from schema
        node.add_label(nt.label());

        // Apply label mutations from delta
        if let Some(added) = delta.added_labels_for(id) {
            for l in added {
                node.add_label(l.clone());
            }
        }
        if let Some(removed) = delta.removed_labels_for(id) {
            for l in removed {
                node.remove_label(l.as_str());
            }
        }

        // Properties from columns
        let props = nt.get_all_properties(offset as usize);
        for (k, v) in props {
            node.set_property(k, v);
        }

        // Apply delta property overrides
        if let Some(overrides) = delta.node_property_overrides_for(id) {
            for (k, opt_v) in overrides {
                match opt_v {
                    Some(v) => {
                        node.set_property(k.clone(), v.clone());
                    }
                    None => {
                        node.remove_property(k.as_str());
                    }
                }
            }
        }

        Some(node)
    }

    fn get_edge(&self, id: EdgeId) -> Option<Edge> {
        let (rel_table_id, csr_position) = decode_edge_id(id);
        let has_delta = self.delta_has_content.load(AtomicOrdering::Relaxed);

        // Delta-created edge — only possible when delta has content.
        if Self::is_delta_edge_id(rel_table_id) {
            if !has_delta {
                return None;
            }
            let delta = self.delta.lock();
            let de = delta.get_delta_edge(id)?;
            if delta.is_edge_deleted(id) {
                return None;
            }
            let mut edge = Edge::new(id, de.src, de.dst, de.edge_type.clone());
            for (k, v) in &de.properties {
                edge.set_property(k.clone(), v.clone());
            }
            return Some(edge);
        }

        let rt = self.resolve_rel_table(rel_table_id)?;
        let pos = csr_position as u32;

        let src = rt.source_node_id(pos)?;
        let dst = rt.dest_node_id(pos)?;
        let edge_type = rt.edge_type().clone();

        // Fast path: no delta mutations — skip the lock entirely.
        if !has_delta {
            let mut edge = Edge::new(id, src, dst, edge_type);
            let props = rt.get_all_edge_properties(csr_position as usize);
            for (k, v) in props {
                edge.set_property(k, v);
            }
            return Some(edge);
        }

        let delta = self.delta.lock();

        // Snapshot edge — check if deleted
        if delta.is_edge_deleted(id) {
            return None;
        }

        let mut edge = Edge::new(id, src, dst, edge_type);

        // Properties from columns
        let props = rt.get_all_edge_properties(csr_position as usize);
        for (k, v) in props {
            edge.set_property(k, v);
        }

        // Apply delta property overrides
        if let Some(overrides) = delta.edge_property_overrides_for(id) {
            for (k, opt_v) in overrides {
                match opt_v {
                    Some(v) => {
                        edge.set_property(k.clone(), v.clone());
                    }
                    None => {
                        edge.remove_property(k.as_str());
                    }
                }
            }
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
        let (table_id, offset) = decode_node_id(id);
        let has_delta = self.delta_has_content.load(AtomicOrdering::Relaxed);

        // Delta-created node — only possible when delta has content.
        if Self::is_delta_node_id(table_id) {
            if !has_delta {
                return None;
            }
            let delta = self.delta.lock();
            let dn = delta.get_delta_node(id)?;
            if delta.is_node_deleted(id) {
                return None;
            }
            return dn.properties.get(key).cloned();
        }

        // Fast path: no delta mutations — read directly from snapshot.
        if !has_delta {
            let nt = self.resolve_node_table(table_id)?;
            return nt.get_property(offset as usize, key);
        }

        let delta = self.delta.lock();

        // Snapshot node — check if deleted
        if delta.is_node_deleted(id) {
            return None;
        }

        // Check delta override first
        if let Some(ov) = delta.get_node_property_override(id, key) {
            return ov.cloned();
        }

        // Read from snapshot column
        let nt = self.resolve_node_table(table_id)?;
        nt.get_property(offset as usize, key)
    }

    fn get_edge_property(&self, id: EdgeId, key: &PropertyKey) -> Option<Value> {
        let (rel_table_id, csr_position) = decode_edge_id(id);
        let has_delta = self.delta_has_content.load(AtomicOrdering::Relaxed);

        // Delta-created edge — only possible when delta has content.
        if Self::is_delta_edge_id(rel_table_id) {
            if !has_delta {
                return None;
            }
            let delta = self.delta.lock();
            let de = delta.get_delta_edge(id)?;
            if delta.is_edge_deleted(id) {
                return None;
            }
            return de.properties.get(key).cloned();
        }

        // Fast path: no delta mutations — read directly from snapshot.
        if !has_delta {
            let rt = self.resolve_rel_table(rel_table_id)?;
            return rt.get_edge_property(csr_position as usize, key);
        }

        let delta = self.delta.lock();

        // Snapshot edge — check if deleted
        if delta.is_edge_deleted(id) {
            return None;
        }

        // Check delta override first
        if let Some(ov) = delta.get_edge_property_override(id, key) {
            return ov.cloned();
        }

        // Read from snapshot column
        let rt = self.resolve_rel_table(rel_table_id)?;
        rt.get_edge_property(csr_position as usize, key)
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
        let (node_table_id, node_offset) = decode_node_id(node);
        let has_delta = self.delta_has_content.load(AtomicOrdering::Relaxed);

        // Delta-created nodes have no snapshot edges.
        if Self::is_delta_node_id(node_table_id) {
            if !has_delta {
                return Vec::new();
            }
            let delta = self.delta.lock();
            let mut result = Vec::new();
            for de in delta.edges_for_node(node, direction) {
                if !delta.is_edge_deleted(de.id) {
                    match direction {
                        Direction::Outgoing => result.push(de.dst),
                        Direction::Incoming => result.push(de.src),
                        Direction::Both => {
                            if de.src == node {
                                result.push(de.dst);
                            } else {
                                result.push(de.src);
                            }
                        }
                    }
                }
            }
            return result;
        }

        let tid = node_table_id as usize;

        // Fast path: no delta mutations — all edges are live, no delta edges.
        if !has_delta {
            let mut result = Vec::new();
            self.collect_snapshot_neighbors(tid, node_offset as u32, direction, &mut result);
            return result;
        }

        let delta = self.delta.lock();
        let mut result = Vec::new();

        if !delta.is_node_deleted(node) {
            self.collect_snapshot_neighbors_filtered(
                tid,
                node_offset as u32,
                direction,
                &delta,
                &mut result,
            );
        }

        // Delta edges
        for de in delta.edges_for_node(node, direction) {
            if !delta.is_edge_deleted(de.id) {
                match direction {
                    Direction::Outgoing => result.push(de.dst),
                    Direction::Incoming => result.push(de.src),
                    Direction::Both => {
                        if de.src == node {
                            result.push(de.dst);
                        } else {
                            result.push(de.src);
                        }
                    }
                }
            }
        }

        result
    }

    fn edges_from(&self, node: NodeId, direction: Direction) -> Vec<(NodeId, EdgeId)> {
        let has_delta = self.delta_has_content.load(AtomicOrdering::Relaxed);

        // Fast path: no delta mutations — snapshot only, no deletion checks.
        if !has_delta {
            let (node_table_id, node_offset) = decode_node_id(node);
            if Self::is_delta_node_id(node_table_id) {
                return Vec::new();
            }
            return self.snapshot_edges_no_delta(node_table_id, node_offset as u32, direction);
        }

        let delta = self.delta.lock();
        let mut results = self.snapshot_edges(node, direction, &delta);

        // Delta edges
        for de in delta.edges_for_node(node, direction) {
            if !delta.is_edge_deleted(de.id) {
                match direction {
                    Direction::Outgoing => results.push((de.dst, de.id)),
                    Direction::Incoming => results.push((de.src, de.id)),
                    Direction::Both => {
                        if de.src == node {
                            results.push((de.dst, de.id));
                        } else {
                            results.push((de.src, de.id));
                        }
                    }
                }
            }
        }

        results
    }

    fn out_degree(&self, node: NodeId) -> usize {
        let (node_table_id, node_offset) = decode_node_id(node);
        let has_delta = self.delta_has_content.load(AtomicOrdering::Relaxed);

        if Self::is_delta_node_id(node_table_id) {
            if !has_delta {
                return 0;
            }
            let delta = self.delta.lock();
            return delta
                .edges_for_node(node, Direction::Outgoing)
                .into_iter()
                .filter(|de| !delta.is_edge_deleted(de.id))
                .count();
        }

        // Fast path: no delta — just count snapshot edges.
        if !has_delta {
            let mut degree = 0;
            if let Some(rel_ids) = self.src_rel_table_ids.get(node_table_id as usize) {
                for &rel_id in rel_ids {
                    let rt = &self.rel_tables_by_id[rel_id as usize];
                    degree += rt.out_degree(node_offset as u32);
                }
            }
            return degree;
        }

        let delta = self.delta.lock();
        let mut degree = 0;

        if !delta.is_node_deleted(node)
            && let Some(rel_ids) = self.src_rel_table_ids.get(node_table_id as usize)
        {
            for &rel_id in rel_ids {
                let rt = &self.rel_tables_by_id[rel_id as usize];
                for (_, eid) in rt.edges_from_source(node_offset as u32) {
                    if !delta.is_edge_deleted(eid) {
                        degree += 1;
                    }
                }
            }
        }

        for de in delta.edges_for_node(node, Direction::Outgoing) {
            if !delta.is_edge_deleted(de.id) {
                degree += 1;
            }
        }

        degree
    }

    fn in_degree(&self, node: NodeId) -> usize {
        let (node_table_id, node_offset) = decode_node_id(node);
        let has_delta = self.delta_has_content.load(AtomicOrdering::Relaxed);

        if Self::is_delta_node_id(node_table_id) {
            if !has_delta {
                return 0;
            }
            let delta = self.delta.lock();
            return delta
                .edges_for_node(node, Direction::Incoming)
                .into_iter()
                .filter(|de| !delta.is_edge_deleted(de.id))
                .count();
        }

        // Fast path: no delta — just count snapshot edges.
        if !has_delta {
            let mut degree = 0;
            if let Some(rel_ids) = self.dst_rel_table_ids.get(node_table_id as usize) {
                for &rel_id in rel_ids {
                    let rt = &self.rel_tables_by_id[rel_id as usize];
                    if let Some(d) = rt.in_degree(node_offset as u32) {
                        degree += d;
                    }
                }
            }
            return degree;
        }

        let delta = self.delta.lock();
        let mut degree = 0;

        if !delta.is_node_deleted(node)
            && let Some(rel_ids) = self.dst_rel_table_ids.get(node_table_id as usize)
        {
            for &rel_id in rel_ids {
                let rt = &self.rel_tables_by_id[rel_id as usize];
                if let Some(edges) = rt.edges_to_target(node_offset as u32) {
                    for (_, eid) in edges {
                        if !delta.is_edge_deleted(eid) {
                            degree += 1;
                        }
                    }
                }
            }
        }

        for de in delta.edges_for_node(node, Direction::Incoming) {
            if !delta.is_edge_deleted(de.id) {
                degree += 1;
            }
        }

        degree
    }

    fn has_backward_adjacency(&self) -> bool {
        self.rel_tables_by_id.iter().any(|rt| rt.has_backward())
    }

    fn node_ids(&self) -> Vec<NodeId> {
        let has_delta = self.delta_has_content.load(AtomicOrdering::Relaxed);

        if !has_delta {
            let mut ids = Vec::new();
            for nt in &self.node_tables_by_id {
                ids.extend(nt.node_ids());
            }
            ids.sort_unstable();
            return ids;
        }

        let delta = self.delta.lock();
        let mut ids = Vec::new();

        // Snapshot node IDs (excluding deleted)
        for nt in &self.node_tables_by_id {
            for nid in nt.node_ids() {
                if !delta.is_node_deleted(nid) {
                    ids.push(nid);
                }
            }
        }

        // Delta-created node IDs (excluding deleted)
        for nid in delta.all_created_node_ids() {
            if !delta.is_node_deleted(nid) {
                ids.push(nid);
            }
        }

        ids.sort_unstable();
        ids
    }

    fn nodes_by_label(&self, label: &str) -> Vec<NodeId> {
        let has_delta = self.delta_has_content.load(AtomicOrdering::Relaxed);

        // Fast path: no delta — just return snapshot nodes for this label.
        if !has_delta {
            return self
                .label_to_table_id
                .get(label)
                .map(|&tid| self.node_tables_by_id[tid as usize].node_ids())
                .unwrap_or_default();
        }

        let delta = self.delta.lock();
        let mut ids = Vec::new();

        // Snapshot nodes for this label
        if let Some(&tid) = self.label_to_table_id.get(label) {
            let nt = &self.node_tables_by_id[tid as usize];
            for nid in nt.node_ids() {
                if !delta.is_node_deleted(nid) {
                    // Check if label was removed by delta
                    let label_removed = delta
                        .removed_labels_for(nid)
                        .is_some_and(|removed| removed.contains(label));
                    if !label_removed {
                        ids.push(nid);
                    }
                }
            }
        }

        // Snapshot nodes from OTHER tables that had this label added by delta
        for nt in &self.node_tables_by_id {
            if nt.label() == label {
                continue; // Already handled above
            }
            for nid in nt.node_ids() {
                if !delta.is_node_deleted(nid)
                    && let Some(added) = delta.added_labels_for(nid)
                    && added.contains(label)
                {
                    ids.push(nid);
                }
            }
        }

        // Delta-created nodes with this label
        if delta.has_overflow(label) {
            for nid in delta.nodes_by_label(label) {
                if !delta.is_node_deleted(nid) {
                    ids.push(nid);
                }
            }
        }

        ids
    }

    fn node_count(&self) -> usize {
        let snapshot_count: usize = self.node_tables_by_id.iter().map(|nt| nt.len()).sum();
        if !self.delta_has_content.load(AtomicOrdering::Relaxed) {
            return snapshot_count;
        }
        let delta = self.delta.lock();
        let created = delta.created_node_count();
        let deleted = delta.deleted_node_count();
        (snapshot_count + created).saturating_sub(deleted)
    }

    fn edge_count(&self) -> usize {
        let snapshot_count: usize = self.rel_tables_by_id.iter().map(|rt| rt.num_edges()).sum();
        if !self.delta_has_content.load(AtomicOrdering::Relaxed) {
            return snapshot_count;
        }
        let delta = self.delta.lock();
        let created = delta.created_edge_count();
        let deleted = delta.deleted_edge_count();
        (snapshot_count + created).saturating_sub(deleted)
    }

    fn edge_type(&self, id: EdgeId) -> Option<ArcStr> {
        let (rel_table_id, _) = decode_edge_id(id);
        let has_delta = self.delta_has_content.load(AtomicOrdering::Relaxed);

        if Self::is_delta_edge_id(rel_table_id) {
            if !has_delta {
                return None;
            }
            let delta = self.delta.lock();
            let de = delta.get_delta_edge(id)?;
            if delta.is_edge_deleted(id) {
                return None;
            }
            return Some(de.edge_type.clone());
        }

        if has_delta {
            let delta = self.delta.lock();
            if delta.is_edge_deleted(id) {
                return None;
            }
        }

        self.rel_table_id_to_type
            .get(rel_table_id as usize)
            .cloned()
    }

    fn find_nodes_by_property(&self, property: &str, value: &Value) -> Vec<NodeId> {
        let key = PropertyKey::new(property);
        let has_delta = self.delta_has_content.load(AtomicOrdering::Relaxed);

        // Fast path: no delta — scan snapshot only, no deletion/override checks.
        if !has_delta {
            let mut results = Vec::new();
            for nt in &self.node_tables_by_id {
                if let Some(zm) = nt.zone_map(&key)
                    && !zm.might_match(CompareOp::Eq, value)
                {
                    continue;
                }
                if let Some(col) = nt.column(&key) {
                    let table_id = nt.table_id();
                    for offset in 0..col.len() {
                        if let Some(v) = col.get(offset)
                            && &v == value
                        {
                            results.push(encode_node_id(table_id, offset as u64));
                        }
                    }
                }
            }
            return results;
        }

        let delta = self.delta.lock();
        let mut results = Vec::new();

        for nt in &self.node_tables_by_id {
            // Zone map check — skip table if no match possible
            if let Some(zm) = nt.zone_map(&key)
                && !zm.might_match(CompareOp::Eq, value)
            {
                continue;
            }

            if let Some(col) = nt.column(&key) {
                let table_id = nt.table_id();
                for offset in 0..col.len() {
                    let nid = encode_node_id(table_id, offset as u64);
                    if delta.is_node_deleted(nid) {
                        continue;
                    }
                    // Check delta override
                    if let Some(ov) = delta.get_node_property_override(nid, &key) {
                        if let Some(v) = ov
                            && v == value
                        {
                            results.push(nid);
                        }
                        continue;
                    }
                    if let Some(v) = col.get(offset)
                        && &v == value
                    {
                        results.push(nid);
                    }
                }
            }
        }

        // Delta-created nodes
        for nid in delta.all_created_node_ids() {
            if delta.is_node_deleted(nid) {
                continue;
            }
            if let Some(dn) = delta.get_delta_node(nid)
                && let Some(v) = dn.properties.get(&key)
                && v == value
            {
                results.push(nid);
            }
        }

        // Scan delta property overrides for snapshot nodes that were in tables
        // without this column or whose table was zone-map-pruned.
        for (&nid, overrides) in delta.all_node_property_overrides() {
            if delta.is_node_deleted(nid) {
                continue;
            }
            if results.contains(&nid) {
                continue; // already found via column scan
            }
            if let Some(Some(v)) = overrides.get(&key)
                && v == value
            {
                results.push(nid);
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
        let has_delta = self.delta_has_content.load(AtomicOrdering::Relaxed);

        // Fast path: no delta — scan snapshot only, no deletion/override checks.
        if !has_delta {
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
                    for offset in 0..col.len() {
                        if let Some(v) = col.get(offset)
                            && Self::value_in_range(&v, min, max, min_inclusive, max_inclusive)
                        {
                            results.push(encode_node_id(table_id, offset as u64));
                        }
                    }
                }
            }
            return results;
        }

        let delta = self.delta.lock();
        let mut results = Vec::new();

        for nt in &self.node_tables_by_id {
            // Zone map range pruning
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
                for offset in 0..col.len() {
                    let nid = encode_node_id(table_id, offset as u64);
                    if delta.is_node_deleted(nid) {
                        continue;
                    }
                    // Check delta override
                    if let Some(ov) = delta.get_node_property_override(nid, &key) {
                        if let Some(v) = ov
                            && Self::value_in_range(v, min, max, min_inclusive, max_inclusive)
                        {
                            results.push(nid);
                        }
                        continue;
                    }
                    if let Some(v) = col.get(offset)
                        && Self::value_in_range(&v, min, max, min_inclusive, max_inclusive)
                    {
                        results.push(nid);
                    }
                }
            }
        }

        // Delta-created nodes
        for nid in delta.all_created_node_ids() {
            if delta.is_node_deleted(nid) {
                continue;
            }
            if let Some(dn) = delta.get_delta_node(nid)
                && let Some(v) = dn.properties.get(&key)
                && Self::value_in_range(v, min, max, min_inclusive, max_inclusive)
            {
                results.push(nid);
            }
        }

        // Scan delta property overrides for snapshot nodes that were in tables
        // without this column or whose table was zone-map-pruned.
        for (&nid, overrides) in delta.all_node_property_overrides() {
            if delta.is_node_deleted(nid) {
                continue;
            }
            if results.contains(&nid) {
                continue; // already found via column scan
            }
            if let Some(Some(v)) = overrides.get(&key)
                && Self::value_in_range(v, min, max, min_inclusive, max_inclusive)
            {
                results.push(nid);
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
        // When delta has any content (created nodes, deleted nodes, or property
        // overrides), conservatively return true. An override could have set any
        // value on any snapshot node, making zone-map pruning unreliable.
        if self.delta_has_content.load(AtomicOrdering::Relaxed) {
            return true;
        }

        let mut might_match = false;
        for nt in &self.node_tables_by_id {
            match nt.zone_map(property) {
                Some(zm) => {
                    if zm.might_match(op, value) {
                        return true; // definitive match possible
                    }
                }
                None => {
                    // No stats for this property in this table — conservatively assume match
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
        self.statistics
            .read()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .clone()
    }

    fn estimate_label_cardinality(&self, label: &str) -> f64 {
        let base = self
            .label_to_table_id
            .get(label)
            .and_then(|&tid| self.node_tables_by_id.get(tid as usize))
            .map_or(0, |nt| nt.len());

        if !self.delta_has_content.load(AtomicOrdering::Relaxed) {
            return base as f64;
        }

        let delta = self.delta.lock();
        let delta_count = if delta.has_overflow(label) {
            delta.nodes_by_label(label).len()
        } else {
            0
        };

        (base + delta_count) as f64
    }

    fn estimate_avg_degree(&self, edge_type: &str, outgoing: bool) -> f64 {
        if let Some(rt) = self
            .edge_type_to_rel_id
            .get(edge_type)
            .and_then(|&rid| self.rel_tables_by_id.get(rid as usize))
        {
            let num_edges = rt.num_edges();
            if num_edges == 0 {
                return 0.0;
            }
            let num_nodes = if outgoing {
                self.resolve_node_table(rt.src_table_id())
                    .map_or(1, |nt| nt.len().max(1))
            } else {
                self.resolve_node_table(rt.dst_table_id())
                    .map_or(1, |nt| nt.len().max(1))
            };
            num_edges as f64 / num_nodes as f64
        } else {
            0.0
        }
    }

    fn current_epoch(&self) -> EpochId {
        EpochId(self.epoch.load(AtomicOrdering::Relaxed))
    }

    fn all_labels(&self) -> Vec<String> {
        // Trait requires Vec<String>; ArcStr::to_string() is a heap copy.
        let mut labels: Vec<String> = self
            .table_id_to_label
            .iter()
            .map(|s| s.to_string())
            .collect();
        if self.delta_has_content.load(AtomicOrdering::Relaxed) {
            let delta = self.delta.lock();
            for node in delta.all_created_nodes() {
                for label in &node.labels {
                    let s = label.to_string();
                    if !labels.contains(&s) {
                        labels.push(s);
                    }
                }
            }
        }
        labels
    }

    fn all_edge_types(&self) -> Vec<String> {
        let mut types: Vec<String> = self
            .rel_table_id_to_type
            .iter()
            .map(|s| s.to_string())
            .collect();
        if self.delta_has_content.load(AtomicOrdering::Relaxed) {
            let delta = self.delta.lock();
            for edge in delta.all_created_edges() {
                let s = edge.edge_type.to_string();
                if !types.contains(&s) {
                    types.push(s);
                }
            }
        }
        types
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
