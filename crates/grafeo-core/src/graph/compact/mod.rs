//! The compacted base of a 0.5.x database file, read for migration.
//!
//! Up to 0.5.44, `compact()` wrote the default graph's data into a columnar
//! base (the `CompactStore` section: one node table per label set, one
//! relationship table per edge type and endpoint tables, with CSR
//! adjacency) and the deletes made since into the `OverlayDeletions`
//! section, beside the writes made since in the LPG section. 0.6.0 reads
//! those sections only to fold the base into the LPG store as such a file
//! opens (see [`fold::fold_0_5_base`]); nothing writes them any more.
//! Removed in 0.7.0 with the other 0.5.x readers.

/// Columnar codecs of the base's property columns.
mod column;
/// Compressed Sparse Row (CSR) adjacency.
mod csr;
/// The `OverlayDeletions` section.
#[cfg(feature = "lpg")]
mod deletions_section;
/// Folds the base into the LPG store.
#[cfg(feature = "lpg")]
pub mod fold;
/// Node and edge id encoding.
mod id;
/// Node tables with their property columns.
mod node_table;
/// Relationship tables over CSR adjacency.
mod rel_table;
/// Table and column schemas.
mod schema;
/// The `CompactStore` section.
mod section;
/// Zone maps of the base's column blocks.
mod zone_map;

use arcstr::ArcStr;
use grafeo_common::types::{EdgeId, NodeId};
use grafeo_common::utils::hash::FxHashMap;

use self::node_table::NodeTable;
use self::rel_table::RelTable;
use crate::graph::Direction;

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

/// The compacted base of a 0.5.x file, as its `CompactStore` section holds
/// it.
///
/// Node data is stored in [`NodeTable`]s, one per label set: a node with the
/// labels `Person` and `Actor` is a row of the table of that set. Edge data
/// is stored in per-type [`RelTable`]s. Read-only: the fold reads it once.
pub(crate) struct CompactStore {
    /// Node tables indexed by table_id for O(1) lookup from NodeId.
    node_tables_by_id: Vec<NodeTable>,
    /// The labels of each table's nodes, by table_id, in name order.
    table_labels: Vec<Vec<ArcStr>>,
    /// Relationship tables indexed by rel_table_id for O(1) lookup from EdgeId.
    rel_tables_by_id: Vec<RelTable>,
    /// Lookup: table ID -> the table's key.
    table_id_to_label: Vec<ArcStr>,
    /// Lookup: rel table ID -> edge type.
    rel_table_id_to_type: Vec<ArcStr>,
    /// Pre-computed: for each node table_id, the rel_table_ids where it is the source.
    src_rel_table_ids: Vec<Vec<u16>>,
    /// Pre-computed: for each node table_id, the rel_table_ids where it is the destination.
    dst_rel_table_ids: Vec<Vec<u16>>,

    // ── Id maps, when the base keeps the ids of its nodes and edges ───
    /// Maps original `NodeId` to (table_id, row_offset).
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
    /// id, in name order. Derives the lookups.
    ///
    /// The section reader checks the invariants this assumes.
    #[must_use]
    pub(crate) fn new(
        node_tables_by_id: Vec<NodeTable>,
        table_labels: Vec<Vec<ArcStr>>,
        rel_tables_by_id: Vec<RelTable>,
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
            // The section reader refuses more tables than a u16 numbers.
            let rel_id = u16::try_from(rel_idx).expect("the reader checked the table count");

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
            table_labels,
            rel_tables_by_id,
            table_id_to_label,
            rel_table_id_to_type,
            src_rel_table_ids,
            dst_rel_table_ids,
            node_id_map: None,
            edge_id_map: None,
            node_offset_to_id: None,
            edge_offset_to_id: None,
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

    /// The labels of the nodes of table `table_id`, in name order (empty for
    /// a table that does not exist).
    #[inline]
    fn labels_of_table(&self, table_id: u16) -> &[ArcStr] {
        self.table_labels
            .get(usize::from(table_id))
            .map_or(&[], Vec::as_slice)
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

    /// The ids of the base's nodes, in id order.
    pub(crate) fn node_ids(&self) -> Vec<NodeId> {
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

    /// The node `id`, with its labels and properties.
    pub(crate) fn get_node(&self, id: NodeId) -> Option<crate::graph::lpg::Node> {
        let (table_id, offset) = self.resolve_node(id)?;
        let nt = self.resolve_node_table(table_id)?;
        let row = usize::try_from(offset).ok()?;
        if row >= nt.len() {
            return None;
        }
        let mut node = crate::graph::lpg::Node::new(id);
        for label in self.labels_of_table(table_id) {
            node.add_label(label.clone());
        }
        for (key, value) in nt.get_all_properties(row) {
            node.set_property(key, value);
        }
        Some(node)
    }

    /// The edge `id`, with its endpoints, type and properties.
    pub(crate) fn get_edge(&self, id: EdgeId) -> Option<crate::graph::lpg::Edge> {
        let (rel_table_id, csr_position) = self.resolve_edge(id)?;
        let rt = self.resolve_rel_table(rel_table_id)?;
        let pos = u32::try_from(csr_position).ok()?;
        let src = self.to_original_node_id(rt.source_node_id(pos)?);
        let dst = self.to_original_node_id(rt.dest_node_id(pos)?);
        let mut edge = crate::graph::lpg::Edge::new(id, src, dst, rt.edge_type().clone());
        for (key, value) in rt.get_all_edge_properties(pos as usize) {
            edge.set_property(key, value);
        }
        Some(edge)
    }

    /// The ids of the edges that leave the node `node`.
    pub(crate) fn outgoing_edges(&self, node: NodeId) -> Vec<EdgeId> {
        let Some((node_table_id, node_offset)) = self.resolve_node(node) else {
            return Vec::new();
        };
        let Ok(offset) = u32::try_from(node_offset) else {
            return Vec::new();
        };
        self.collect_edges(node_table_id, offset, Direction::Outgoing)
            .into_iter()
            .map(|(_, edge)| edge)
            .collect()
    }

    // ── ID-preserving accessors ────────────────────────────────────

    /// Returns `true` if the base carries id maps back to the original
    /// node and edge IDs.
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
}
