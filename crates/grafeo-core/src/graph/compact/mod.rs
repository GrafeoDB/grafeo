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
/// Container section serialization for the layered overlay deletion log.
#[cfg(feature = "lpg")]
pub mod deletions_section;
/// Folds the compacted base of a database file into its LPG store.
#[cfg(feature = "lpg")]
pub mod fold;
mod graph_store_impl;
/// Node/edge ID encoding and decoding helpers.
pub mod id;
/// Two-layer store: columnar base + mutable LPG overlay.
#[cfg(feature = "lpg")]
pub mod layered;
/// Per-label node tables with columnar property storage.
pub mod node_table;
/// Per-type relationship tables backed by forward/backward CSR.
pub mod rel_table;
/// Schema definitions for node tables and edge schemas.
pub mod schema;
/// Container section serialization for CompactStore.
pub mod section;
#[cfg(test)]
mod tests;
/// Zone maps for skip-pruning predicate evaluation.
pub mod zone_map;

pub use builder::{CompactStoreBuilder, from_graph_store, from_graph_store_preserving_ids};

use std::sync::Arc;

use arcstr::ArcStr;
use grafeo_common::types::{EdgeId, NodeId};
use grafeo_common::utils::hash::FxHashMap;

use self::node_table::NodeTable;
use self::rel_table::RelTable;
use crate::graph::Direction;
use crate::statistics::{EdgeTypeStatistics, LabelStatistics, Statistics};

/// The key of the node table holding the nodes whose labels are `labels`:
/// the labels in name order without repeats, each with `\` and `|` escaped
/// by a `\`, joined with `|`. A single label without either character is its
/// own key.
///
/// A file names each node table by its key, and [`labels_of_key`] reads the
/// labels back from it, whatever they hold: a label `In|Out` and the labels
/// `In` and `Out` get different keys. The nodes without labels have the
/// empty key, so the empty label alone is keyed `|` (two empty labels, which
/// read back as one).
#[must_use]
pub(crate) fn label_set_key<S: AsRef<str>>(labels: &[S]) -> ArcStr {
    let mut sorted: Vec<&str> = labels.iter().map(AsRef::as_ref).collect();
    sorted.sort_unstable();
    sorted.dedup();
    if sorted == [""] {
        return ArcStr::from("|");
    }
    let mut key = String::new();
    for (index, label) in sorted.iter().enumerate() {
        if index > 0 {
            key.push('|');
        }
        for character in label.chars() {
            if matches!(character, '\\' | '|') {
                key.push('\\');
            }
            key.push(character);
        }
    }
    ArcStr::from(key)
}

/// The labels of a node table's key as [`label_set_key`] writes it, in name
/// order; none for the empty key. A `\` before anything but `\` or `|` is a
/// backslash of the label: builds before 0.6.0 wrote keys without escapes
/// (see [`labels_of_unescaped_key`]).
#[must_use]
pub(crate) fn labels_of_key(key: &str) -> Vec<ArcStr> {
    if key.is_empty() {
        return Vec::new();
    }
    let mut labels = Vec::new();
    let mut label = String::new();
    let mut characters = key.chars();
    while let Some(character) = characters.next() {
        match character {
            '\\' => match characters.next() {
                Some(escaped @ ('\\' | '|')) => label.push(escaped),
                Some(other) => {
                    label.push('\\');
                    label.push(other);
                }
                None => label.push('\\'),
            },
            '|' => labels.push(ArcStr::from(std::mem::take(&mut label))),
            other => label.push(other),
        }
    }
    labels.push(ArcStr::from(label));
    in_label_order(labels)
}

/// The labels of a node table's key as 0.5.x wrote it (the version 1, 2 and
/// 3 encodings): the node's labels joined with `|`, without escapes. A label
/// that held a `|` cannot be told from two labels there; it reads as two.
#[must_use]
pub(crate) fn labels_of_unescaped_key(key: &str) -> Vec<ArcStr> {
    if key.is_empty() {
        return Vec::new();
    }
    in_label_order(key.split('|').map(ArcStr::from).collect())
}

/// `labels` in name order without repeats.
fn in_label_order(mut labels: Vec<ArcStr>) -> Vec<ArcStr> {
    labels.sort_unstable();
    labels.dedup();
    labels
}

/// A read-only columnar graph store.
///
/// Node data is stored in [`NodeTable`]s, one per label set: a node with the
/// labels `Person` and `Actor` is a row of the table of that set, which every
/// read of either label reads. Edge data is stored in per-type
/// [`RelTable`]s. The store is immutable after construction: use
/// [`CompactStoreBuilder`] to populate it from raw data.
pub struct CompactStore {
    /// Node tables indexed by table_id for O(1) lookup from NodeId.
    node_tables_by_id: Vec<NodeTable>,
    /// table_id lookup from a table's key (see [`label_set_key`]).
    label_to_table_id: FxHashMap<ArcStr, u16>,
    /// The labels of each table's nodes, by table_id, in name order.
    table_labels: Vec<Vec<ArcStr>>,
    /// The tables whose nodes have a label, in table order (for
    /// `nodes_by_label`).
    label_to_table_ids: FxHashMap<ArcStr, Vec<u16>>,
    /// Relationship tables indexed by rel_table_id for O(1) lookup from EdgeId.
    rel_tables_by_id: Vec<RelTable>,
    /// rel_table_id lookup from edge type string (one edge type may span
    /// multiple src/dst label combinations, so the value is a Vec).
    edge_type_to_rel_id: FxHashMap<ArcStr, Vec<u16>>,
    /// Lookup: table ID -> the table's key.
    table_id_to_label: Vec<ArcStr>,
    /// Lookup: rel table ID -> edge type.
    rel_table_id_to_type: Vec<ArcStr>,
    /// Pre-computed: for each node table_id, the rel_table_ids where it is the source.
    src_rel_table_ids: Vec<Vec<u16>>,
    /// Pre-computed: for each node table_id, the rel_table_ids where it is the destination.
    dst_rel_table_ids: Vec<Vec<u16>>,
    /// Cached statistics.
    statistics: Arc<Statistics>,

    // ── ID-preserving maps (for layered store integration) ──────────
    /// Maps original `NodeId` to (table_id, row_offset). Present when the
    /// store was built with [`from_graph_store_preserving_ids`].
    node_id_map: Option<FxHashMap<NodeId, (u16, u64)>>,
    /// Maps original `EdgeId` to (rel_table_id, csr_position).
    edge_id_map: Option<FxHashMap<EdgeId, (u16, u64)>>,
    /// Reverse: table_id index -> vec of original `NodeId` per row offset.
    node_offset_to_id: Option<Vec<Vec<NodeId>>>,
    /// Reverse: rel_table_id index -> vec of original `EdgeId` per CSR position.
    edge_offset_to_id: Option<Vec<Vec<EdgeId>>>,
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
    /// Creates a new `CompactStore` from pre-built components:
    /// `table_labels` holds the labels of each node table's nodes, by table
    /// id, in name order. Derives the lookups and the statistics.
    ///
    /// Prefer using [`CompactStoreBuilder`] which validates schemas. This
    /// constructor is `pub(crate)` because it assumes all invariants are
    /// already satisfied.
    #[must_use]
    pub(crate) fn new(
        node_tables_by_id: Vec<NodeTable>,
        table_labels: Vec<Vec<ArcStr>>,
        rel_tables_by_id: Vec<RelTable>,
        edge_type_to_rel_id: FxHashMap<ArcStr, Vec<u16>>,
        rel_table_id_to_type: Vec<ArcStr>,
    ) -> Self {
        debug_assert_eq!(
            table_labels.len(),
            node_tables_by_id.len(),
            "one label set per node table"
        );
        let table_id_to_label: Vec<ArcStr> = node_tables_by_id
            .iter()
            .map(|table| ArcStr::from(table.label()))
            .collect();
        let mut label_to_table_id: FxHashMap<ArcStr, u16> = FxHashMap::default();
        let mut label_to_table_ids: FxHashMap<ArcStr, Vec<u16>> = FxHashMap::default();
        let mut label_counts: FxHashMap<&str, u64> = FxHashMap::default();
        for (table, (key, labels)) in node_tables_by_id
            .iter()
            .zip(table_id_to_label.iter().zip(&table_labels))
        {
            let table_id = table.table_id();
            label_to_table_id.insert(key.clone(), table_id);
            for label in labels {
                label_to_table_ids
                    .entry(label.clone())
                    .or_default()
                    .push(table_id);
                *label_counts.entry(label.as_str()).or_default() += table.len() as u64;
            }
        }

        let mut statistics = Statistics::new();
        for (label, count) in label_counts {
            statistics.update_label(label, LabelStatistics::new(count));
        }
        let mut edge_counts: FxHashMap<&str, u64> = FxHashMap::default();
        for (table, edge_type) in rel_tables_by_id.iter().zip(&rel_table_id_to_type) {
            *edge_counts.entry(edge_type.as_str()).or_default() += table.num_edges() as u64;
        }
        for (edge_type, count) in edge_counts {
            statistics.update_edge_type(edge_type, EdgeTypeStatistics::new(count, 0.0, 0.0));
        }
        statistics.total_nodes = node_tables_by_id.iter().map(|t| t.len() as u64).sum();
        statistics.total_edges = rel_tables_by_id.iter().map(|t| t.num_edges() as u64).sum();

        // Pre-compute src/dst rel_table_id mappings per node table_id.
        let node_table_count = node_tables_by_id.len();
        let mut src_rel_table_ids = vec![Vec::new(); node_table_count];
        let mut dst_rel_table_ids = vec![Vec::new(); node_table_count];

        debug_assert!(
            rel_tables_by_id.len() <= usize::from(id::MAX_TABLE_ID) + 1,
            "rel table count {} exceeds 15-bit limit; caller must validate",
            rel_tables_by_id.len()
        );
        for (rel_idx, rt) in rel_tables_by_id.iter().enumerate() {
            // Caller (CompactStoreBuilder::build) validates table count fits u16.
            let rel_id = u16::try_from(rel_idx).expect("caller validated table count");

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
            table_labels,
            label_to_table_ids,
            rel_tables_by_id,
            edge_type_to_rel_id,
            table_id_to_label,
            rel_table_id_to_type,
            src_rel_table_ids,
            dst_rel_table_ids,
            statistics: Arc::new(statistics),
            node_id_map: None,
            edge_id_map: None,
            node_offset_to_id: None,
            edge_offset_to_id: None,
        }
    }

    /// A store without nodes or edges that preserves ids (with empty id
    /// maps), as a compacted base that holds nothing.
    #[must_use]
    pub fn empty() -> Self {
        let mut store = Self::new(
            Vec::new(),
            Vec::new(),
            Vec::new(),
            FxHashMap::default(),
            Vec::new(),
        );
        store.set_id_maps(
            FxHashMap::default(),
            FxHashMap::default(),
            Vec::new(),
            Vec::new(),
        );
        store
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

    /// The labels of the nodes of table `table_id`, in name order (empty for
    /// a table that does not exist).
    #[inline]
    fn labels_of_table(&self, table_id: u16) -> &[ArcStr] {
        self.table_labels
            .get(usize::from(table_id))
            .map_or(&[], Vec::as_slice)
    }

    /// The ids of the tables whose nodes have `label`, in table order.
    #[inline]
    fn tables_with_label(&self, label: &str) -> &[u16] {
        self.label_to_table_ids
            .get(label)
            .map_or(&[], Vec::as_slice)
    }

    /// Returns the node table of the nodes whose only label is `label`, if
    /// any. Nodes that have `label` beside other labels are rows of the
    /// tables of their label sets.
    #[must_use]
    pub fn node_table(&self, label: &str) -> Option<&NodeTable> {
        let &tid = self.label_to_table_id.get(&label_set_key(&[label]))?;
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

    /// Whether `nodes_by_label(label)` holds `id`: the node is a row of a
    /// table whose label set has `label`.
    #[must_use]
    pub(crate) fn node_in_label(&self, id: NodeId, label: &str) -> bool {
        self.resolve_node(id).is_some_and(|(table, offset)| {
            self.labels_of_table(table)
                .binary_search_by(|held| held.as_str().cmp(label))
                .is_ok()
                && self
                    .resolve_node_table(table)
                    .zip(usize::try_from(offset).ok())
                    .is_some_and(|(rows, row)| row < rows.len())
        })
    }

    /// Returns the key of a table ID, if valid: the labels of its nodes in
    /// name order, joined with `|` (and `\` and `|` in a label escaped by a
    /// `\`); a table of single-label nodes has their label as its key.
    #[must_use]
    pub fn label_for_table_id(&self, table_id: u16) -> Option<&ArcStr> {
        self.table_id_to_label.get(table_id as usize)
    }

    /// Returns the edge type for a given rel table ID, if valid.
    #[must_use]
    pub fn edge_type_for_rel_table_id(&self, rel_table_id: u16) -> Option<&ArcStr> {
        self.rel_table_id_to_type.get(rel_table_id as usize)
    }

    /// Collects edges from snapshot RelTables for a given node in a direction.
    ///
    /// When ID-preserving, the returned `NodeId`/`EdgeId` values are translated
    /// back to the original IDs from the source store.
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

        // Translate compact-encoded IDs to original IDs when preserving.
        if self.preserves_ids() {
            for (target_id, edge_id) in &mut results {
                *target_id = self.to_original_node_id(*target_id);
                *edge_id = self.to_original_edge_id(*edge_id);
            }
        }

        results
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
        let id_map_bytes = self.id_map_memory_bytes();
        node_bytes + rel_bytes + id_map_bytes
    }

    // ── ID-preserving accessors ────────────────────────────────────

    /// Returns `true` if original IDs are preserved (built via
    /// [`from_graph_store_preserving_ids`]).
    #[must_use]
    pub fn preserves_ids(&self) -> bool {
        self.node_id_map.is_some()
    }

    /// Attaches ID maps to an already-built `CompactStore`.
    pub(crate) fn set_id_maps(
        &mut self,
        node_id_map: FxHashMap<NodeId, (u16, u64)>,
        edge_id_map: FxHashMap<EdgeId, (u16, u64)>,
        node_offset_to_id: Vec<Vec<NodeId>>,
        edge_offset_to_id: Vec<Vec<EdgeId>>,
    ) {
        self.node_id_map = Some(node_id_map);
        self.edge_id_map = Some(edge_id_map);
        self.node_offset_to_id = Some(node_offset_to_id);
        self.edge_offset_to_id = Some(edge_offset_to_id);
    }

    /// Resolves an input `NodeId` to (table_id, offset).
    ///
    /// When ID-preserving, looks up the original ID in the map.
    /// Otherwise, decodes the compact-encoded bits.
    #[inline]
    pub(crate) fn resolve_node(&self, id: NodeId) -> Option<(u16, u64)> {
        if let Some(ref map) = self.node_id_map {
            map.get(&id).copied()
        } else {
            Some(id::decode_node_id(id))
        }
    }

    /// Resolves an input `EdgeId` to (rel_table_id, csr_position).
    #[inline]
    pub(crate) fn resolve_edge(&self, id: EdgeId) -> Option<(u16, u64)> {
        if let Some(ref map) = self.edge_id_map {
            map.get(&id).copied()
        } else {
            Some(id::decode_edge_id(id))
        }
    }

    /// Translates a compact-encoded `NodeId` (from internal CSR/table lookups)
    /// back to the original preserved ID. No-op when not ID-preserving.
    #[inline]
    pub(crate) fn to_original_node_id(&self, compact_id: NodeId) -> NodeId {
        if let Some(ref offsets) = self.node_offset_to_id {
            let (table_id, offset) = id::decode_node_id(compact_id);
            offsets
                .get(table_id as usize)
                .and_then(|v| v.get(usize::try_from(offset).ok()?))
                .copied()
                .unwrap_or(compact_id)
        } else {
            compact_id
        }
    }

    /// Translates a compact-encoded `EdgeId` back to the original preserved ID.
    #[inline]
    pub(crate) fn to_original_edge_id(&self, compact_id: EdgeId) -> EdgeId {
        if let Some(ref offsets) = self.edge_offset_to_id {
            let (rel_table_id, csr_pos) = id::decode_edge_id(compact_id);
            offsets
                .get(rel_table_id as usize)
                .and_then(|v| v.get(usize::try_from(csr_pos).ok()?))
                .copied()
                .unwrap_or(compact_id)
        } else {
            compact_id
        }
    }

    /// Approximate heap cost of the ID maps.
    fn id_map_memory_bytes(&self) -> usize {
        // ~24 bytes per entry (key + value) in FxHashMap, plus Vec overhead.
        let node_map = self.node_id_map.as_ref().map_or(0, |m| m.len() * 24);
        let edge_map = self.edge_id_map.as_ref().map_or(0, |m| m.len() * 24);
        let node_rev = self
            .node_offset_to_id
            .as_ref()
            .map_or(0, |v| v.iter().map(|inner| inner.len() * 8).sum());
        let edge_rev = self
            .edge_offset_to_id
            .as_ref()
            .map_or(0, |v| v.iter().map(|inner| inner.len() * 8).sum());
        node_map + edge_map + node_rev + edge_rev
    }
}
