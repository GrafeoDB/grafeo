//! CompactStore — a read-optimized columnar store for memory-constrained environments.
//!
//! Implements `GraphStoreMut` using per-label columnar tables and double-indexed
//! CSR adjacency. Designed for static snapshot data in WASM, edge workers, and
//! embedded devices. Fully behind `#[cfg(feature = "compact-store")]`.

/// Builder API for constructing a [`CompactStore`] from raw data.
pub mod builder;
/// Columnar codecs for node and edge properties.
pub mod column;
/// Compressed Sparse Row (CSR) adjacency representation.
pub mod csr;
/// Delta buffer for layering runtime mutations over the immutable snapshot.
pub mod delta;
mod graph_store_impl;
mod graph_store_mut_impl;
/// Node/edge ID encoding and decoding helpers.
pub mod id;
/// Per-label node tables with columnar property storage.
pub mod node_table;
/// Per-type relationship tables backed by forward/backward CSR.
pub mod rel_table;
/// Schema definitions for node tables and edge schemas.
pub mod schema;
#[cfg(test)]
mod tests;
/// Zone maps for skip-pruning predicate evaluation.
pub mod zone_map;

pub use builder::CompactStoreBuilder;

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64};

use arcstr::ArcStr;
use grafeo_common::types::{EdgeId, NodeId, Value};
use grafeo_common::utils::hash::FxHashMap;

use self::delta::{DELTA_EDGE_TABLE_ID, DELTA_NODE_TABLE_ID, DeltaBuffer};
use self::id::decode_node_id;
use self::node_table::NodeTable;
use self::rel_table::RelTable;
use self::zone_map::compare_values;
use crate::graph::Direction;
use crate::statistics::Statistics;

/// A read-optimized columnar graph store.
///
/// Node data is stored in per-label [`NodeTable`]s and edge data in per-type
/// [`RelTable`]s. The snapshot data is immutable — runtime mutations go to a
/// [`DeltaBuffer`] protected by a `parking_lot::Mutex`.
pub struct CompactStore {
    /// Node tables indexed by table_id for O(1) lookup from NodeId.
    node_tables_by_id: Vec<NodeTable>,
    /// table_id lookup from label string (for nodes_by_label).
    label_to_table_id: FxHashMap<ArcStr, u16>,
    /// Relationship tables indexed by rel_table_id for O(1) lookup from EdgeId.
    rel_tables_by_id: Vec<RelTable>,
    /// rel_table_id lookup from edge type string.
    edge_type_to_rel_id: FxHashMap<ArcStr, u16>,
    /// Lookup: table ID -> label.
    table_id_to_label: Vec<ArcStr>,
    /// Lookup: rel table ID -> edge type.
    rel_table_id_to_type: Vec<ArcStr>,
    /// Pre-computed: for each node table_id, the rel_table_ids where it is the source.
    src_rel_table_ids: Vec<Vec<u16>>,
    /// Pre-computed: for each node table_id, the rel_table_ids where it is the destination.
    dst_rel_table_ids: Vec<Vec<u16>>,
    /// Mutation overlay.
    delta: parking_lot::Mutex<DeltaBuffer>,
    /// Set to true when the first mutation occurs. Allows skipping the
    /// mutex lock on read paths when no mutations have happened.
    delta_has_content: AtomicBool,
    /// Current epoch.
    epoch: AtomicU64,
    /// Cached statistics.
    statistics: std::sync::RwLock<Arc<Statistics>>,
}

impl std::fmt::Debug for CompactStore {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CompactStore")
            .field("node_tables_by_id", &self.node_tables_by_id)
            .field("rel_tables_by_id", &self.rel_tables_by_id)
            .field("table_id_to_label", &self.table_id_to_label)
            .field("rel_table_id_to_type", &self.rel_table_id_to_type)
            .finish_non_exhaustive()
    }
}

impl CompactStore {
    /// Creates a new `CompactStore` from pre-built components.
    ///
    /// Prefer using [`CompactStoreBuilder`] which validates schemas and
    /// computes statistics automatically. This constructor is `pub(crate)`
    /// because it assumes all invariants are already satisfied.
    ///
    /// `node_tables_by_id` and `rel_tables_by_id` are indexed by their
    /// respective table IDs. The `label_to_table_id` and
    /// `edge_type_to_rel_id` maps enable string-based lookups.
    #[must_use]
    pub(crate) fn new(
        node_tables_by_id: Vec<NodeTable>,
        label_to_table_id: FxHashMap<ArcStr, u16>,
        rel_tables_by_id: Vec<RelTable>,
        edge_type_to_rel_id: FxHashMap<ArcStr, u16>,
        table_id_to_label: Vec<ArcStr>,
        rel_table_id_to_type: Vec<ArcStr>,
        statistics: Statistics,
    ) -> Self {
        // Pre-compute src/dst rel_table_id mappings per node table_id.
        let node_table_count = node_tables_by_id.len();
        let mut src_rel_table_ids = vec![Vec::new(); node_table_count];
        let mut dst_rel_table_ids = vec![Vec::new(); node_table_count];

        for (rel_idx, rt) in rel_tables_by_id.iter().enumerate() {
            let rel_id = rel_idx as u16;
            let src_tid = rt.src_table_id() as usize;
            let dst_tid = rt.dst_table_id() as usize;
            if src_tid < node_table_count {
                src_rel_table_ids[src_tid].push(rel_id);
            }
            if dst_tid < node_table_count {
                dst_rel_table_ids[dst_tid].push(rel_id);
            }
        }

        Self {
            node_tables_by_id,
            label_to_table_id,
            rel_tables_by_id,
            edge_type_to_rel_id,
            table_id_to_label,
            rel_table_id_to_type,
            src_rel_table_ids,
            dst_rel_table_ids,
            delta: parking_lot::Mutex::new(DeltaBuffer::new()),
            delta_has_content: AtomicBool::new(false),
            epoch: AtomicU64::new(1),
            statistics: std::sync::RwLock::new(Arc::new(statistics)),
        }
    }

    /// Resolves a table_id to its [`NodeTable`].
    #[inline]
    fn resolve_node_table(&self, table_id: u16) -> Option<&NodeTable> {
        self.node_tables_by_id.get(table_id as usize)
    }

    /// Resolves a rel_table_id to its [`RelTable`].
    #[inline]
    fn resolve_rel_table(&self, rel_table_id: u16) -> Option<&RelTable> {
        self.rel_tables_by_id.get(rel_table_id as usize)
    }

    /// Returns `true` if the given table_id belongs to the delta layer.
    #[inline]
    fn is_delta_node_id(table_id: u16) -> bool {
        table_id >= DELTA_NODE_TABLE_ID
    }

    /// Returns `true` if the given rel_table_id belongs to the delta layer.
    #[inline]
    fn is_delta_edge_id(rel_table_id: u16) -> bool {
        rel_table_id >= DELTA_EDGE_TABLE_ID
    }

    /// Returns a reference to the node table for the given label, if any.
    #[must_use]
    pub fn node_table(&self, label: &str) -> Option<&NodeTable> {
        let &tid = self.label_to_table_id.get(label)?;
        self.node_tables_by_id.get(tid as usize)
    }

    /// Returns a reference to the relationship table for the given edge type.
    #[must_use]
    pub fn rel_table(&self, edge_type: &str) -> Option<&RelTable> {
        let &rid = self.edge_type_to_rel_id.get(edge_type)?;
        self.rel_tables_by_id.get(rid as usize)
    }

    /// Returns the label for a given table ID, if valid.
    #[must_use]
    pub fn label_for_table_id(&self, table_id: u16) -> Option<&ArcStr> {
        self.table_id_to_label.get(table_id as usize)
    }

    /// Returns the edge type for a given rel table ID, if valid.
    #[must_use]
    pub fn edge_type_for_rel_table_id(&self, rel_table_id: u16) -> Option<&ArcStr> {
        self.rel_table_id_to_type.get(rel_table_id as usize)
    }

    /// Collects edges from snapshot RelTables for a given node in a direction,
    /// filtering out deleted edges via the delta buffer.
    fn snapshot_edges(
        &self,
        node: NodeId,
        direction: Direction,
        delta: &DeltaBuffer,
    ) -> Vec<(NodeId, EdgeId)> {
        let (node_table_id, node_offset) = decode_node_id(node);

        // For delta-created nodes, there are no snapshot edges.
        if Self::is_delta_node_id(node_table_id) {
            return Vec::new();
        }

        let tid = node_table_id as usize;
        let mut results = Vec::new();

        match direction {
            Direction::Outgoing => {
                if let Some(rel_ids) = self.src_rel_table_ids.get(tid) {
                    for &rel_id in rel_ids {
                        let rt = &self.rel_tables_by_id[rel_id as usize];
                        for (target, eid) in rt.edges_from_source(node_offset as u32) {
                            if !delta.is_edge_deleted(eid) {
                                results.push((target, eid));
                            }
                        }
                    }
                }
            }
            Direction::Incoming => {
                if let Some(rel_ids) = self.dst_rel_table_ids.get(tid) {
                    for &rel_id in rel_ids {
                        let rt = &self.rel_tables_by_id[rel_id as usize];
                        if let Some(edges) = rt.edges_to_target(node_offset as u32) {
                            for (source, eid) in edges {
                                if !delta.is_edge_deleted(eid) {
                                    results.push((source, eid));
                                }
                            }
                        }
                    }
                }
            }
            Direction::Both => {
                if let Some(rel_ids) = self.src_rel_table_ids.get(tid) {
                    for &rel_id in rel_ids {
                        let rt = &self.rel_tables_by_id[rel_id as usize];
                        for (target, eid) in rt.edges_from_source(node_offset as u32) {
                            if !delta.is_edge_deleted(eid) {
                                results.push((target, eid));
                            }
                        }
                    }
                }
                if let Some(rel_ids) = self.dst_rel_table_ids.get(tid) {
                    for &rel_id in rel_ids {
                        let rt = &self.rel_tables_by_id[rel_id as usize];
                        if let Some(edges) = rt.edges_to_target(node_offset as u32) {
                            for (source, eid) in edges {
                                if !delta.is_edge_deleted(eid) {
                                    results.push((source, eid));
                                }
                            }
                        }
                    }
                }
            }
        }

        results
    }

    /// Like `snapshot_edges` but without delta-deletion filtering.
    /// Used on the fast path when no mutations have occurred.
    fn snapshot_edges_no_delta(
        &self,
        node_table_id: u16,
        node_offset: u32,
        direction: Direction,
    ) -> Vec<(NodeId, EdgeId)> {
        let tid = node_table_id as usize;
        let mut results = Vec::new();

        match direction {
            Direction::Outgoing => {
                if let Some(rel_ids) = self.src_rel_table_ids.get(tid) {
                    for &rel_id in rel_ids {
                        let rt = &self.rel_tables_by_id[rel_id as usize];
                        results.extend(rt.edges_from_source(node_offset));
                    }
                }
            }
            Direction::Incoming => {
                if let Some(rel_ids) = self.dst_rel_table_ids.get(tid) {
                    for &rel_id in rel_ids {
                        let rt = &self.rel_tables_by_id[rel_id as usize];
                        if let Some(edges) = rt.edges_to_target(node_offset) {
                            results.extend(edges);
                        }
                    }
                }
            }
            Direction::Both => {
                if let Some(rel_ids) = self.src_rel_table_ids.get(tid) {
                    for &rel_id in rel_ids {
                        let rt = &self.rel_tables_by_id[rel_id as usize];
                        results.extend(rt.edges_from_source(node_offset));
                    }
                }
                if let Some(rel_ids) = self.dst_rel_table_ids.get(tid) {
                    for &rel_id in rel_ids {
                        let rt = &self.rel_tables_by_id[rel_id as usize];
                        if let Some(edges) = rt.edges_to_target(node_offset) {
                            results.extend(edges);
                        }
                    }
                }
            }
        }

        results
    }

    /// Collects snapshot neighbor NodeIds without checking delta deletions.
    fn collect_snapshot_neighbors(
        &self,
        tid: usize,
        node_offset: u32,
        direction: Direction,
        result: &mut Vec<NodeId>,
    ) {
        match direction {
            Direction::Outgoing => {
                if let Some(rel_ids) = self.src_rel_table_ids.get(tid) {
                    for &rel_id in rel_ids {
                        let rt = &self.rel_tables_by_id[rel_id as usize];
                        for (target, _) in rt.edges_from_source(node_offset) {
                            result.push(target);
                        }
                    }
                }
            }
            Direction::Incoming => {
                if let Some(rel_ids) = self.dst_rel_table_ids.get(tid) {
                    for &rel_id in rel_ids {
                        let rt = &self.rel_tables_by_id[rel_id as usize];
                        if let Some(edges) = rt.edges_to_target(node_offset) {
                            for (source, _) in edges {
                                result.push(source);
                            }
                        }
                    }
                }
            }
            Direction::Both => {
                if let Some(rel_ids) = self.src_rel_table_ids.get(tid) {
                    for &rel_id in rel_ids {
                        let rt = &self.rel_tables_by_id[rel_id as usize];
                        for (target, _) in rt.edges_from_source(node_offset) {
                            result.push(target);
                        }
                    }
                }
                if let Some(rel_ids) = self.dst_rel_table_ids.get(tid) {
                    for &rel_id in rel_ids {
                        let rt = &self.rel_tables_by_id[rel_id as usize];
                        if let Some(edges) = rt.edges_to_target(node_offset) {
                            for (source, _) in edges {
                                result.push(source);
                            }
                        }
                    }
                }
            }
        }
    }

    /// Collects snapshot neighbor NodeIds, filtering out deleted edges.
    fn collect_snapshot_neighbors_filtered(
        &self,
        tid: usize,
        node_offset: u32,
        direction: Direction,
        delta: &DeltaBuffer,
        result: &mut Vec<NodeId>,
    ) {
        match direction {
            Direction::Outgoing => {
                if let Some(rel_ids) = self.src_rel_table_ids.get(tid) {
                    for &rel_id in rel_ids {
                        let rt = &self.rel_tables_by_id[rel_id as usize];
                        for (target, eid) in rt.edges_from_source(node_offset) {
                            if !delta.is_edge_deleted(eid) {
                                result.push(target);
                            }
                        }
                    }
                }
            }
            Direction::Incoming => {
                if let Some(rel_ids) = self.dst_rel_table_ids.get(tid) {
                    for &rel_id in rel_ids {
                        let rt = &self.rel_tables_by_id[rel_id as usize];
                        if let Some(edges) = rt.edges_to_target(node_offset) {
                            for (source, eid) in edges {
                                if !delta.is_edge_deleted(eid) {
                                    result.push(source);
                                }
                            }
                        }
                    }
                }
            }
            Direction::Both => {
                if let Some(rel_ids) = self.src_rel_table_ids.get(tid) {
                    for &rel_id in rel_ids {
                        let rt = &self.rel_tables_by_id[rel_id as usize];
                        for (target, eid) in rt.edges_from_source(node_offset) {
                            if !delta.is_edge_deleted(eid) {
                                result.push(target);
                            }
                        }
                    }
                }
                if let Some(rel_ids) = self.dst_rel_table_ids.get(tid) {
                    for &rel_id in rel_ids {
                        let rt = &self.rel_tables_by_id[rel_id as usize];
                        if let Some(edges) = rt.edges_to_target(node_offset) {
                            for (source, eid) in edges {
                                if !delta.is_edge_deleted(eid) {
                                    result.push(source);
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    /// Returns a rough estimate of heap memory used by the snapshot data
    /// (node columns + CSR structures + edge property columns), in bytes.
    ///
    /// Does not include the [`DeltaBuffer`], `FxHashMap` overhead, or schema
    /// metadata. For precise measurement, use a heap profiler.
    #[must_use]
    pub fn memory_bytes(&self) -> usize {
        let node_bytes: usize = self
            .node_tables_by_id
            .iter()
            .map(|nt| nt.memory_bytes())
            .sum();
        let rel_bytes: usize = self
            .rel_tables_by_id
            .iter()
            .map(|rt| rt.memory_bytes())
            .sum();
        node_bytes + rel_bytes
    }

    /// Checks if a value falls within a range.
    fn value_in_range(
        val: &Value,
        min: Option<&Value>,
        max: Option<&Value>,
        min_inclusive: bool,
        max_inclusive: bool,
    ) -> bool {
        if let Some(min_val) = min {
            match compare_values(val, min_val) {
                Some(std::cmp::Ordering::Less) => return false,
                Some(std::cmp::Ordering::Equal) if !min_inclusive => return false,
                None => return false,
                _ => {}
            }
        }
        if let Some(max_val) = max {
            match compare_values(val, max_val) {
                Some(std::cmp::Ordering::Greater) => return false,
                Some(std::cmp::Ordering::Equal) if !max_inclusive => return false,
                None => return false,
                _ => {}
            }
        }
        true
    }
}
