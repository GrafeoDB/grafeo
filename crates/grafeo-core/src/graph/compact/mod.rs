//! CompactStore: a read-only columnar store for memory-constrained environments.
//!
//! Implements [`GraphStore`](crate::graph::traits::GraphStore) using per-label
//! columnar tables and double-indexed CSR adjacency. Designed for static
//! snapshot data in WASM, edge workers, and embedded devices.
//! Fully behind `#[cfg(feature = "compact-store")]`.

/// Builder API for constructing a [`CompactStore`] from raw data.
pub mod builder;
/// Columnar codecs for node and edge properties.
pub mod column;
/// Compressed Sparse Row (CSR) adjacency representation.
pub mod csr;
mod graph_store_impl;
/// Node/edge ID encoding and decoding helpers.
pub mod id;
/// Per-label node tables with columnar property storage.
pub mod node_table;
/// Per-type relationship tables backed by forward/backward CSR.
pub mod rel_table;
/// Schema definitions for node tables and edge schemas.
pub mod schema;
/// Serialization and deserialization for CompactStore.
pub mod serialize;
#[cfg(test)]
mod tests;
/// Zone maps for skip-pruning predicate evaluation.
pub mod zone_map;

pub use builder::{CompactStoreBuilder, from_graph_store, from_graph_store_incremental};

use std::sync::Arc;

use arcstr::ArcStr;
use grafeo_common::types::{EdgeId, NodeId};
use grafeo_common::utils::hash::FxHashMap;

use self::node_table::NodeTable;
use self::rel_table::RelTable;
use crate::graph::Direction;
use crate::statistics::Statistics;

/// A read-only columnar graph store.
///
/// Node data is stored in per-label [`NodeTable`]s and edge data in per-type
/// [`RelTable`]s. The store is immutable after construction: use
/// [`CompactStoreBuilder`] to populate it from raw data.
pub struct CompactStore {
    /// Node tables indexed by table_id for O(1) lookup from NodeId.
    node_tables_by_id: Vec<NodeTable>,
    /// table_id lookup from label string (for nodes_by_label).
    label_to_table_id: FxHashMap<ArcStr, u16>,
    /// Relationship tables indexed by rel_table_id for O(1) lookup from EdgeId.
    rel_tables_by_id: Vec<RelTable>,
    /// rel_table_id lookup from edge type string (one edge type may span
    /// multiple src/dst label combinations, so the value is a Vec).
    edge_type_to_rel_id: FxHashMap<ArcStr, Vec<u16>>,
    /// Lookup: table ID -> label.
    table_id_to_label: Vec<ArcStr>,
    /// Lookup: rel table ID -> edge type.
    rel_table_id_to_type: Vec<ArcStr>,
    /// Pre-computed: for each node table_id, the rel_table_ids where it is the source.
    src_rel_table_ids: Vec<Vec<u16>>,
    /// Pre-computed: for each node table_id, the rel_table_ids where it is the destination.
    dst_rel_table_ids: Vec<Vec<u16>>,
    /// Cached statistics.
    statistics: Arc<Statistics>,
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
    #[must_use]
    pub(crate) fn new(
        node_tables_by_id: Vec<NodeTable>,
        label_to_table_id: FxHashMap<ArcStr, u16>,
        rel_tables_by_id: Vec<RelTable>,
        edge_type_to_rel_id: FxHashMap<ArcStr, Vec<u16>>,
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
            statistics: Arc::new(statistics),
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

    /// Returns a reference to the node table for the given label, if any.
    #[must_use]
    pub fn node_table(&self, label: &str) -> Option<&NodeTable> {
        let &tid = self.label_to_table_id.get(label)?;
        self.node_tables_by_id.get(tid as usize)
    }

    /// Returns a reference to the first relationship table for the given edge type.
    ///
    /// When an edge type spans multiple label pairs, use [`Self::rel_tables_for_type`]
    /// to get all matching tables.
    #[must_use]
    pub fn rel_table(&self, edge_type: &str) -> Option<&RelTable> {
        let rids = self.edge_type_to_rel_id.get(edge_type)?;
        let &rid = rids.first()?;
        self.rel_tables_by_id.get(rid as usize)
    }

    /// Returns all relationship tables for the given edge type.
    #[must_use]
    pub fn rel_tables_for_type(&self, edge_type: &str) -> Vec<&RelTable> {
        self.edge_type_to_rel_id
            .get(edge_type)
            .map(|rids| {
                rids.iter()
                    .filter_map(|&rid| self.rel_tables_by_id.get(rid as usize))
                    .collect()
            })
            .unwrap_or_default()
    }

    /// Returns the label for a given table ID, if valid.
    #[must_use]
    pub fn label_for_table_id(&self, table_id: u16) -> Option<&ArcStr> {
        self.table_id_to_label.get(table_id as usize)
    }

    /// Returns the ordered slice of node table labels, indexed by table ID.
    ///
    /// Position `i` in the slice corresponds to `table_id = i`. Used by
    /// HybridStore to enumerate all tables when computing ID boundaries.
    #[must_use]
    pub fn table_id_to_label(&self) -> &[ArcStr] {
        &self.table_id_to_label
    }

    /// Returns the edge type for a given rel table ID, if valid.
    #[must_use]
    pub fn edge_type_for_rel_table_id(&self, rel_table_id: u16) -> Option<&ArcStr> {
        self.rel_table_id_to_type.get(rel_table_id as usize)
    }

    /// Collects edges from snapshot RelTables for a given node in a direction.
    fn collect_edges(
        &self,
        node_table_id: u16,
        node_offset: u32,
        direction: Direction,
    ) -> Vec<(NodeId, EdgeId)> {
        let tid = node_table_id as usize;
        let mut results = Vec::new();

        if matches!(direction, Direction::Outgoing | Direction::Both)
            && let Some(rel_ids) = self.src_rel_table_ids.get(tid)
        {
            for &rel_id in rel_ids {
                let rt = &self.rel_tables_by_id[rel_id as usize];
                results.extend(rt.edges_from_source(node_offset));
            }
        }

        if matches!(direction, Direction::Incoming | Direction::Both)
            && let Some(rel_ids) = self.dst_rel_table_ids.get(tid)
        {
            for &rel_id in rel_ids {
                let rt = &self.rel_tables_by_id[rel_id as usize];
                if let Some(edges) = rt.edges_to_target(node_offset) {
                    results.extend(edges);
                }
            }
        }

        results
    }

    /// Returns the node tables indexed by table_id.
    #[must_use]
    pub(crate) fn node_tables_by_id(&self) -> &[NodeTable] {
        &self.node_tables_by_id
    }

    /// Returns the relationship tables indexed by rel_table_id.
    #[must_use]
    pub(crate) fn rel_tables_by_id(&self) -> &[RelTable] {
        &self.rel_tables_by_id
    }

    /// Returns the label-to-table-id map.
    #[must_use]
    pub(crate) fn label_to_table_id_map(&self) -> &FxHashMap<ArcStr, u16> {
        &self.label_to_table_id
    }

    /// Returns the edge-type-to-rel-id map.
    #[must_use]
    pub(crate) fn edge_type_to_rel_id_map(&self) -> &FxHashMap<ArcStr, Vec<u16>> {
        &self.edge_type_to_rel_id
    }

    /// Returns the rel_table_id-to-type lookup.
    #[must_use]
    pub(crate) fn rel_table_id_to_type_slice(&self) -> &[ArcStr] {
        &self.rel_table_id_to_type
    }

    /// Returns a rough estimate of heap memory used by the snapshot data
    /// (node columns + CSR structures + edge property columns), in bytes.
    ///
    /// Does not include `FxHashMap` overhead or schema metadata. For precise
    /// measurement, use a heap profiler.
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
}
