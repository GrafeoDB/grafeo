//! What the engine reads of a store to report on it (P7): the memory it
//! holds by part, in its own terms ([`RowGroupStore::memory`]) and in
//! `LpgStore`'s shape ([`RowGroupStore::memory_breakdown`]), its edges at an
//! epoch, and every node and edge a committed reader sees now
//! ([`RowGroupStore::all_nodes`], [`RowGroupStore::all_edges`]).
//!
//! The engine reports one figure per database, so a root's figures cover
//! its named graphs: [`RowGroupStore::memory`], `heap_bytes`,
//! `version_bytes` and `cold_columns`. The counts and the iterators are of
//! the store alone, as `LpgStore`'s are.

use std::mem::size_of;
use std::sync::atomic::Ordering;

use grafeo_common::memory::usage::{IndexMemory, MvccMemory, StoreMemory, StringPoolMemory};
use grafeo_common::types::EpochId;
use grafeo_common::utils::hash::FxHashSet;

use super::bitset::Bitset;
use super::{Inner, NodeGroup, ROWS_PER_GROUP, Read, RowGroupStore};
use crate::graph::lpg::{Edge, Node};

/// The heap bytes a store holds, by part (an estimate, as `LpgStore`'s): a
/// root's with its named graphs'.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
#[non_exhaustive]
pub struct RowGroupMemory {
    /// The stores counted: the root and its named graphs.
    pub graphs: usize,
    /// The row groups of the node tables.
    pub node_groups: usize,
    /// The row groups of the edge tables.
    pub edge_groups: usize,
    /// The node row groups' entries in their tables.
    pub node_tables_bytes: usize,
    /// The edge row groups' entries in their tables, with each row's
    /// source, target and type.
    pub edge_tables_bytes: usize,
    /// The node property columns: their decoded values and presence bits.
    pub node_columns_bytes: usize,
    /// The edge property columns: their decoded values and presence bits.
    pub edge_columns_bytes: usize,
    /// The chunks the node property columns still hold encoded, as a load
    /// left them.
    pub node_cold_bytes: usize,
    /// The chunks the edge property columns still hold encoded.
    pub edge_cold_bytes: usize,
    /// The bytes of the node property columns' chunks in spill files: off
    /// the heap, so not in [`total_bytes`](Self::total_bytes).
    pub node_spilled_bytes: usize,
    /// The bytes of the edge property columns' chunks in spill files.
    pub edge_spilled_bytes: usize,
    /// The property columns as `LpgStore` counts them: one per key a table
    /// has a column of, in each graph (each row group holds a part of it).
    pub property_columns: usize,
    /// The columns of row groups that still hold their chunks encoded: a
    /// column decodes on its first use.
    pub cold_columns: usize,
    /// The cold columns of row groups whose chunks are in spill files.
    pub spilled_columns: usize,
    /// The node rows' version info: who created and deleted each row, and
    /// the update chains of their values and labels.
    pub node_versions_bytes: usize,
    /// The edge rows' version info, with the update chains of their values.
    pub edge_versions_bytes: usize,
    /// The nodes' outgoing edges.
    pub outgoing_bytes: usize,
    /// The nodes' incoming edges.
    pub incoming_bytes: usize,
    /// The bitmaps of rows per label.
    pub label_bitmaps_bytes: usize,
    /// The label dictionaries, with the counts of nodes per label.
    pub labels_bytes: usize,
    /// The edge type dictionaries, with the counts of edges per type.
    pub edge_types_bytes: usize,
    /// The property key dictionaries, which nodes and edges share.
    pub keys_bytes: usize,
    /// The labels the dictionaries hold, in each graph.
    pub label_count: usize,
    /// The edge types the dictionaries hold, in each graph.
    pub edge_type_count: usize,
}

impl RowGroupMemory {
    /// The heap bytes of every part.
    #[must_use]
    pub fn total_bytes(&self) -> usize {
        self.node_tables_bytes
            + self.edge_tables_bytes
            + self.node_columns_bytes
            + self.edge_columns_bytes
            + self.node_cold_bytes
            + self.edge_cold_bytes
            + self.node_versions_bytes
            + self.edge_versions_bytes
            + self.outgoing_bytes
            + self.incoming_bytes
            + self.label_bitmaps_bytes
            + self.labels_bytes
            + self.edge_types_bytes
            + self.keys_bytes
    }

    /// Adds `other`'s figures to these.
    fn add(&mut self, other: &Self) {
        self.graphs += other.graphs;
        self.node_groups += other.node_groups;
        self.edge_groups += other.edge_groups;
        self.node_tables_bytes += other.node_tables_bytes;
        self.edge_tables_bytes += other.edge_tables_bytes;
        self.node_columns_bytes += other.node_columns_bytes;
        self.edge_columns_bytes += other.edge_columns_bytes;
        self.node_cold_bytes += other.node_cold_bytes;
        self.edge_cold_bytes += other.edge_cold_bytes;
        self.node_spilled_bytes += other.node_spilled_bytes;
        self.edge_spilled_bytes += other.edge_spilled_bytes;
        self.property_columns += other.property_columns;
        self.cold_columns += other.cold_columns;
        self.spilled_columns += other.spilled_columns;
        self.node_versions_bytes += other.node_versions_bytes;
        self.edge_versions_bytes += other.edge_versions_bytes;
        self.outgoing_bytes += other.outgoing_bytes;
        self.incoming_bytes += other.incoming_bytes;
        self.label_bitmaps_bytes += other.label_bitmaps_bytes;
        self.labels_bytes += other.labels_bytes;
        self.edge_types_bytes += other.edge_types_bytes;
        self.keys_bytes += other.keys_bytes;
        self.label_count += other.label_count;
        self.edge_type_count += other.edge_type_count;
    }
}

impl Inner {
    /// What this store holds, without its named graphs.
    fn memory(&self) -> RowGroupMemory {
        let mut memory = RowGroupMemory {
            graphs: 1,
            node_groups: self.nodes.len(),
            edge_groups: self.edges.len(),
            labels_bytes: self.labels.heap_bytes()
                + self.counts.labels.capacity() * size_of::<i64>(),
            edge_types_bytes: self.edge_types.heap_bytes()
                + self.counts.edge_types.capacity() * size_of::<i64>(),
            keys_bytes: self.keys.heap_bytes(),
            label_count: self.labels.len(),
            edge_type_count: self.edge_types.len(),
            ..RowGroupMemory::default()
        };
        let mut node_keys: FxHashSet<u32> = FxHashSet::default();
        for group in self.nodes.values() {
            memory.node_tables_bytes += size_of::<u64>() + size_of::<NodeGroup>();
            memory.node_versions_bytes += group.versions.heap_bytes()
                + group.columns.updates_bytes()
                + group
                    .label_updates
                    .heap_bytes(|ids| std::mem::size_of_val::<[u32]>(ids));
            memory.label_bitmaps_bytes +=
                group.labels.values().map(Bitset::heap_bytes).sum::<usize>()
                    + group.labels.capacity() * (size_of::<u32>() + size_of::<Bitset>());
            memory.node_columns_bytes += group.columns.hot_bytes();
            memory.node_cold_bytes += group.columns.cold_bytes();
            memory.node_spilled_bytes += group.columns.spilled_bytes();
            memory.cold_columns += group.columns.cold_count();
            memory.spilled_columns += group.columns.spilled_count();
            memory.outgoing_bytes += group.outgoing.heap_bytes();
            memory.incoming_bytes += group.incoming.heap_bytes();
            node_keys.extend(group.columns.keys());
        }
        let mut edge_keys: FxHashSet<u32> = FxHashSet::default();
        for group in self.edges.values() {
            memory.edge_tables_bytes += size_of::<u64>()
                + size_of::<super::EdgeGroup>()
                + (group.src.capacity() + group.dst.capacity()) * size_of::<u64>()
                + group.edge_type.capacity() * size_of::<u32>();
            memory.edge_versions_bytes +=
                group.versions.heap_bytes() + group.columns.updates_bytes();
            memory.edge_columns_bytes += group.columns.hot_bytes();
            memory.edge_cold_bytes += group.columns.cold_bytes();
            memory.edge_spilled_bytes += group.columns.spilled_bytes();
            memory.cold_columns += group.columns.cold_count();
            memory.spilled_columns += group.columns.spilled_count();
            edge_keys.extend(group.columns.keys());
        }
        memory.property_columns = node_keys.len() + edge_keys.len();
        memory
    }

    /// The node ids `read` sees, ascending.
    fn visible_nodes(&self, read: Read) -> Vec<u64> {
        let mut ids = Vec::new();
        for (index, group) in &self.nodes {
            ids.extend(
                group
                    .versions
                    .rows()
                    .filter(|(_, version)| version.visible(read))
                    .map(|(row, _)| index * ROWS_PER_GROUP + row as u64),
            );
        }
        ids
    }

    /// The edge ids `read` sees, ascending.
    fn visible_edges(&self, read: Read) -> Vec<u64> {
        let mut ids = Vec::new();
        for (index, group) in &self.edges {
            ids.extend(
                group
                    .versions
                    .rows()
                    .filter(|(_, version)| version.visible(read))
                    .map(|(row, _)| index * ROWS_PER_GROUP + row as u64),
            );
        }
        ids
    }
}

impl RowGroupStore {
    /// The heap bytes it holds by part, with its named graphs' (an
    /// estimate). Each store is read under its own lock, one after the
    /// other, so a write meanwhile may leave the figures of two graphs from
    /// different moments.
    #[must_use]
    pub fn memory(&self) -> RowGroupMemory {
        let mut memory = self.inner.read().memory();
        for name in self.graph_names() {
            // Dropped since the names were read: not counted.
            if let Some(graph) = self.graph(&name) {
                memory.add(&graph.memory());
            }
        }
        memory
    }

    /// [`memory`](Self::memory) in the shape of `LpgStore`'s
    /// `memory_breakdown`, which the engine reports: the row groups and
    /// their property columns (with the chunks still encoded and the key
    /// dictionary) as the store, the adjacency and the label bitmaps as
    /// the indexes, the version info as the MVCC overhead, the label and
    /// edge type dictionaries as the string pool. The four totals add up to
    /// [`heap_bytes`](Self::heap_bytes).
    ///
    /// The version chains count the rows' version info and the update
    /// chains of their values and labels; their depth is reported as one,
    /// as the update chains are counted in bytes and not per row. The
    /// property, vector and text indexes of every graph are counted with the
    /// indexes (a named graph's vector and text indexes named
    /// `graph/label:property`), a property index as an estimate per entry.
    #[must_use]
    pub fn memory_breakdown(&self) -> (StoreMemory, IndexMemory, MvccMemory, StringPoolMemory) {
        let memory = self.memory();
        // `#[non_exhaustive]` in grafeo-common: built field by field.
        let mut store = StoreMemory::default();
        store.nodes_bytes = memory.node_tables_bytes;
        store.edges_bytes = memory.edge_tables_bytes;
        store.node_properties_bytes =
            memory.node_columns_bytes + memory.node_cold_bytes + memory.keys_bytes;
        store.edge_properties_bytes = memory.edge_columns_bytes + memory.edge_cold_bytes;
        store.property_column_count = memory.property_columns;
        store.compute_total();

        let mut indexes = IndexMemory::default();
        indexes.forward_adjacency_bytes = memory.outgoing_bytes;
        indexes.backward_adjacency_bytes = memory.incoming_bytes;
        indexes.label_index_bytes = memory.label_bitmaps_bytes;
        self.add_index_memory(&mut indexes);
        indexes.compute_total();

        let any_rows = memory.node_groups + memory.edge_groups > 0;
        let mut mvcc = MvccMemory::default();
        mvcc.node_version_chains_bytes = memory.node_versions_bytes;
        mvcc.edge_version_chains_bytes = memory.edge_versions_bytes;
        mvcc.average_chain_depth = if any_rows { 1.0 } else { 0.0 };
        mvcc.max_chain_depth = usize::from(any_rows);
        mvcc.compute_total();

        let mut string_pool = StringPoolMemory::default();
        string_pool.label_registry_bytes = memory.labels_bytes;
        string_pool.edge_type_registry_bytes = memory.edge_types_bytes;
        string_pool.label_count = memory.label_count;
        string_pool.edge_type_count = memory.edge_type_count;
        string_pool.compute_total();

        (store, indexes, mvcc, string_pool)
    }

    /// Adds the property, vector and text indexes of this store and its
    /// named graphs to `indexes`.
    fn add_index_memory(&self, indexes: &mut IndexMemory) {
        let mut graphs: Vec<(String, std::sync::Arc<Self>)> = self
            .graph_names()
            .into_iter()
            .filter_map(|name| self.graph(&name).map(|graph| (name, graph)))
            .collect();
        graphs.sort_by(|a, b| a.0.cmp(&b.0));
        let all = || {
            std::iter::once(("", self))
                .chain(graphs.iter().map(|(name, graph)| (name.as_str(), &**graph)))
        };
        for (_, store) in all() {
            for key in store.property_index_keys() {
                let entries = store.property_index_len(&key).unwrap_or(0);
                // As LpgStore estimates it: a value and a set of ids per
                // entry, and the map's own overhead.
                indexes.property_index_bytes += entries
                    * (size_of::<grafeo_common::types::HashableValue>()
                        + size_of::<FxHashSet<grafeo_common::types::NodeId>>()
                        + 32);
            }
        }
        #[cfg(any(feature = "vector-index", feature = "text-index"))]
        for (graph, store) in all() {
            store.add_search_index_memory(graph, indexes);
        }
        indexes.vector_indexes.sort_by(|a, b| a.name.cmp(&b.name));
        indexes.text_indexes.sort_by(|a, b| a.name.cmp(&b.name));
    }

    /// Adds the vector and text indexes of this store, the graph `graph` of
    /// its root (empty for the root itself), to `indexes`, named
    /// `graph/Label:property` for a named graph.
    #[cfg(any(feature = "vector-index", feature = "text-index"))]
    fn add_search_index_memory(&self, graph: &str, indexes: &mut IndexMemory) {
        let named = |key: String| {
            if graph.is_empty() {
                key
            } else {
                format!("{graph}/{key}")
            }
        };
        #[cfg(feature = "vector-index")]
        for (key, index) in self.vector_index_entries() {
            indexes
                .vector_indexes
                .push(grafeo_common::memory::NamedMemory {
                    name: named(key),
                    bytes: index.heap_memory_bytes(),
                    item_count: index.len(),
                });
        }
        #[cfg(feature = "text-index")]
        for (key, index) in self.text_index_entries() {
            let index = index.read();
            indexes
                .text_indexes
                .push(grafeo_common::memory::NamedMemory {
                    name: named(key),
                    bytes: index.heap_memory_bytes(),
                    item_count: index.len(),
                });
        }
    }

    /// The heap bytes its row groups and dictionaries hold, with its named
    /// graphs' (an estimate): [`memory`](Self::memory)'s total.
    #[must_use]
    pub fn heap_bytes(&self) -> usize {
        self.memory().total_bytes()
    }

    /// The heap bytes its rows' version info takes, with its named graphs'.
    #[must_use]
    pub fn version_bytes(&self) -> usize {
        let memory = self.memory();
        memory.node_versions_bytes + memory.edge_versions_bytes
    }

    /// The property columns of its row groups, and of its named graphs',
    /// that still hold their chunks as a load left them, encoded: a column
    /// decodes on its first use.
    #[must_use]
    pub fn cold_columns(&self) -> usize {
        self.memory().cold_columns
    }

    /// The next node id it gives out: every node id it holds is below it,
    /// and the ids below it are dense (a reserved id never used holds no
    /// row).
    #[doc(hidden)]
    #[must_use]
    pub fn next_node_id(&self) -> u64 {
        self.next_node_id.load(Ordering::Acquire)
    }

    /// The next edge id it gives out, as [`next_node_id`](Self::next_node_id)
    /// for edges: a scan of `0..next_edge_id()` meets every edge.
    #[doc(hidden)]
    #[must_use]
    pub fn next_edge_id(&self) -> u64 {
        self.next_edge_id.load(Ordering::Acquire)
    }

    /// The edges a committed reader at `epoch` sees, counted under one lock
    /// from the rows the row groups hold: what a scan of every id below
    /// [`next_edge_id`](Self::next_edge_id) with `is_edge_visible_at_epoch`
    /// counts. At the store's own epoch it is `edge_count`.
    #[must_use]
    pub fn edge_count_at_epoch(&self, epoch: EpochId) -> usize {
        let read = Read::at(epoch);
        self.inner
            .read()
            .edges
            .values()
            .map(|group| {
                group
                    .versions
                    .rows()
                    .filter(|(_, version)| version.visible(read))
                    .count()
            })
            .sum()
    }

    /// Every node a committed reader at the store's current epoch sees, in
    /// id order, with its labels and values, as `LpgStore`'s `all_nodes`:
    /// the ids are taken at once, each node is read as the iterator is
    /// advanced (at the same epoch), and a node a `collect` took out
    /// meanwhile is left out.
    pub fn all_nodes(&self) -> impl Iterator<Item = Node> + '_ {
        self.nodes_at(self.current_epoch())
    }

    /// Every edge a committed reader at the store's current epoch sees, in
    /// id order, with its ends, type and values, as `LpgStore`'s
    /// `all_edges` (see [`all_nodes`](Self::all_nodes)).
    pub fn all_edges(&self) -> impl Iterator<Item = Edge> + '_ {
        self.edges_at(self.current_epoch())
    }

    /// Every node a committed reader at `epoch` sees, in id order, with its
    /// labels and values at `epoch`: what a snapshot, a copy of the database
    /// or of a graph holds of the store (P5). Read as
    /// [`all_nodes`](Self::all_nodes) reads, one node at a time; `epoch` is
    /// at or after the horizon the store was last collected at.
    pub fn nodes_at(&self, epoch: EpochId) -> impl Iterator<Item = Node> + '_ {
        let read = Read::at(epoch);
        let ids = self.inner.read().visible_nodes(read);
        ids.into_iter()
            .filter_map(move |id| self.inner.read().read_node(id, read))
    }

    /// Every edge a committed reader at `epoch` sees, in id order, with its
    /// ends, type and values at `epoch` (see [`nodes_at`](Self::nodes_at)).
    pub fn edges_at(&self, epoch: EpochId) -> impl Iterator<Item = Edge> + '_ {
        let read = Read::at(epoch);
        let ids = self.inner.read().visible_edges(read);
        ids.into_iter()
            .filter_map(move |id| self.inner.read().read_edge(id, read))
    }
}

#[cfg(test)]
mod tests {
    use grafeo_common::storage::{ChunkCaps, ImageSource, MemoryImage, SectionType};
    use grafeo_common::types::{EdgeId, EpochId, NodeId, Value};

    use crate::graph::conformance::{Store, Tx, people};
    use crate::graph::lpg::{Edge, Node};
    use crate::graph::rowgroup::section::LPG_SECTION_VERSION;
    use crate::graph::rowgroup::{RowGroupMemory, RowGroupStore};
    use crate::graph::traits::GraphStore;

    /// A node as the iterators list it: id, sorted labels, values by key.
    type NodeRow = (u64, Vec<String>, Vec<(String, Value)>);
    /// An edge as the iterators list it: id, ends, type, values by key.
    type EdgeRow = (u64, u64, u64, String, Vec<(String, Value)>);

    fn node_row(node: Node) -> NodeRow {
        let mut labels: Vec<String> = node.labels.iter().map(ToString::to_string).collect();
        labels.sort();
        let mut values: Vec<(String, Value)> = node
            .properties
            .iter()
            .map(|(key, value)| (key.to_string(), value.clone()))
            .collect();
        values.sort_by(|(a, _), (b, _)| a.cmp(b));
        (node.id.as_u64(), labels, values)
    }

    fn edge_row(edge: Edge) -> EdgeRow {
        let mut values: Vec<(String, Value)> = edge
            .properties
            .iter()
            .map(|(key, value)| (key.to_string(), value.clone()))
            .collect();
        values.sort_by(|(a, _), (b, _)| a.cmp(b));
        (
            edge.id.as_u64(),
            edge.src.as_u64(),
            edge.dst.as_u64(),
            edge.edge_type.to_string(),
            values,
        )
    }

    /// Lists the nodes and edges, sorted by id.
    fn listed(
        nodes: impl Iterator<Item = Node>,
        edges: impl Iterator<Item = Edge>,
    ) -> (Vec<NodeRow>, Vec<EdgeRow>) {
        let mut nodes: Vec<NodeRow> = nodes.map(node_row).collect();
        nodes.sort_by_key(|(id, _, _)| *id);
        let mut edges: Vec<EdgeRow> = edges.map(edge_row).collect();
        edges.sort_by_key(|(id, ..)| *id);
        (nodes, edges)
    }

    /// People, then a commit that adds Vincent and deletes Mia with her
    /// edge, then an open transaction that adds Jules and an edge to him
    /// and deletes the edge from Alix to Gus, the open transaction left as
    /// it is. Returns Vincent.
    fn scene<S: Store>(store: &S) -> NodeId {
        let people = people(store).unwrap();
        let mut tx = Tx::begin(store, 10);
        let vincent = tx
            .create_node(
                &["Person"],
                &[
                    ("name", Value::from("Vincent")),
                    ("city", Value::from("Paris")),
                ],
            )
            .unwrap();
        tx.delete_edge(people.gus_mia).unwrap();
        tx.delete_node(people.mia).unwrap();
        tx.commit().unwrap();
        let mut open = Tx::begin(store, 11);
        let jules = open
            .create_node(&["Person"], &[("name", Value::from("Jules"))])
            .unwrap();
        open.create_edge(vincent, jules, "KNOWS", &[("since", Value::Int64(1994))])
            .unwrap();
        open.delete_edge(people.alix_gus).unwrap();
        vincent
    }

    /// The iterators list exactly the nodes and edges a committed reader
    /// sees, with their labels and values, in id order.
    #[test]
    fn the_iterators_list_the_visible_committed_entities() {
        let store = RowGroupStore::new();
        let vincent = scene(&store);
        let (nodes, edges) = listed(store.all_nodes(), store.all_edges());
        let text = |value: &str| Value::from(value);
        assert_eq!(
            nodes,
            [
                (
                    0,
                    vec!["Person".to_string()],
                    vec![
                        ("age".to_string(), Value::Int64(19)),
                        ("city".to_string(), text("Amsterdam")),
                        ("name".to_string(), text("Alix")),
                    ]
                ),
                (
                    1,
                    vec!["Employee".to_string(), "Person".to_string()],
                    vec![
                        ("age".to_string(), Value::Int64(3)),
                        ("name".to_string(), text("Gus")),
                    ]
                ),
                (
                    vincent.as_u64(),
                    vec!["Person".to_string()],
                    vec![
                        ("city".to_string(), text("Paris")),
                        ("name".to_string(), text("Vincent")),
                    ]
                ),
            ],
            "not Mia, deleted, nor Jules, not committed"
        );
        assert_eq!(
            edges,
            [(
                0,
                0,
                1,
                "KNOWS".to_string(),
                vec![("since".to_string(), Value::Int64(1988))]
            )],
            "Alix's edge to Gus until its delete commits; not the edge to Jules"
        );
        assert_eq!(
            store.all_nodes().map(|node| node.id).collect::<Vec<_>>(),
            [NodeId::new(0), NodeId::new(1), vincent],
            "in id order"
        );
    }

    /// The edge count at an epoch is what a scan of every id below the next
    /// edge id counts, at every epoch, with an edge not committed yet, and
    /// after a `collect`.
    #[test]
    fn the_edge_count_at_an_epoch_agrees_with_a_scan() {
        let store = RowGroupStore::new();
        let people = people(&store).unwrap();
        let loaded = store.current_epoch();
        let mut tx = Tx::begin(&store, 10);
        tx.create_edge(people.alix, people.mia, "KNOWS", &[])
            .unwrap();
        let added = tx.commit().unwrap();
        let mut tx = Tx::begin(&store, 11);
        tx.delete_edge(people.gus_mia).unwrap();
        let deleted = tx.commit().unwrap();
        let mut open = Tx::begin(&store, 12);
        // Left open: no committed reader sees it.
        open.create_edge(people.mia, people.alix, "KNOWS", &[])
            .unwrap();

        let scan = |epoch: EpochId| {
            (0..store.next_edge_id())
                .filter(|&id| store.is_edge_visible_at_epoch(EdgeId::new(id), epoch))
                .count()
        };
        let epochs = [EpochId::new(0), loaded, added, deleted];
        let counted: Vec<usize> = epochs
            .iter()
            .map(|&epoch| store.edge_count_at_epoch(epoch))
            .collect();
        assert_eq!(counted, [0, 2, 3, 2]);
        for epoch in epochs {
            assert_eq!(store.edge_count_at_epoch(epoch), scan(epoch), "{epoch:?}");
        }
        assert_eq!(store.edge_count_at_epoch(deleted), store.edge_count());
        assert_eq!(
            store.next_edge_id(),
            4,
            "the open transaction's edge has its id"
        );
        assert_eq!(store.next_node_id(), 3);

        // The open edge shares a vector with the deleted one, which keeps it
        // stamped: the roll back lets the collect take the delete out.
        open.roll_back().unwrap();
        assert_eq!(store.collect(deleted), 1, "the edge from Gus to Mia");
        assert_eq!(store.edge_count_at_epoch(deleted), 2);
        assert_eq!(scan(deleted), 2);
    }

    /// The breakdown adds up to the store's heap bytes, has bytes in each
    /// part the store holds, and grows with rows.
    #[test]
    fn the_breakdown_sums_to_its_parts_and_grows_with_rows() {
        let store = RowGroupStore::new();
        people(&store).unwrap();
        let memory = store.memory();
        assert_eq!(memory.total_bytes(), store.heap_bytes());
        let (data, indexes, mvcc, names) = store.memory_breakdown();
        assert_eq!(
            data.total_bytes + indexes.total_bytes + mvcc.total_bytes + names.total_bytes,
            memory.total_bytes(),
            "the four parts make the whole"
        );
        for (part, bytes) in [
            ("node tables", memory.node_tables_bytes),
            ("edge tables", memory.edge_tables_bytes),
            ("node columns", memory.node_columns_bytes),
            ("edge columns", memory.edge_columns_bytes),
            ("node versions", memory.node_versions_bytes),
            ("edge versions", memory.edge_versions_bytes),
            ("outgoing", memory.outgoing_bytes),
            ("incoming", memory.incoming_bytes),
            ("label bitmaps", memory.label_bitmaps_bytes),
            ("labels", memory.labels_bytes),
            ("edge types", memory.edge_types_bytes),
            ("keys", memory.keys_bytes),
        ] {
            assert!(bytes > 0, "{part}");
        }
        assert_eq!(
            (memory.graphs, memory.node_groups, memory.edge_groups),
            (1, 1, 1)
        );
        assert_eq!(
            (
                memory.cold_columns,
                memory.node_cold_bytes,
                memory.edge_cold_bytes
            ),
            (0, 0, 0),
            "nothing was loaded"
        );
        assert_eq!(
            data.property_column_count, 4,
            "the nodes' name, age and city, the edges' since"
        );
        assert_eq!((names.label_count, names.edge_type_count), (2, 1));
        assert_eq!(
            (indexes.property_index_bytes, indexes.node_labels_bytes),
            (0, 0)
        );
        assert_eq!(mvcc.max_chain_depth, 1);

        let mut tx = Tx::begin(&store, 10);
        for age in 0..3_000 {
            tx.create_node(
                &["Person"],
                &[("name", Value::from("Vincent")), ("age", Value::Int64(age))],
            )
            .unwrap();
        }
        tx.commit().unwrap();
        let grown = store.memory();
        assert!(grown.node_columns_bytes > memory.node_columns_bytes);
        assert!(grown.node_versions_bytes > memory.node_versions_bytes);
        assert!(grown.label_bitmaps_bytes > memory.label_bitmaps_bytes);
        assert_eq!(
            grown.edge_tables_bytes, memory.edge_tables_bytes,
            "no edge was added"
        );
        assert_eq!(grown.total_bytes(), store.heap_bytes());
    }

    /// `collect` shrinks the figures: a vector's version info collapses,
    /// and a row group left without rows goes with all it held.
    #[test]
    fn collect_shrinks_the_breakdown() {
        let store = RowGroupStore::new();
        let people = people(&store).unwrap();
        let mut tx = Tx::begin(&store, 10);
        let jules = tx
            .create_node(&["Person"], &[("name", Value::from("Jules"))])
            .unwrap();
        let epoch = tx.commit().unwrap();
        let stamped = store.memory();
        assert_eq!(store.collect(epoch), 0, "nothing was deleted");
        let collapsed = store.memory();
        assert!(
            collapsed.node_versions_bytes < stamped.node_versions_bytes,
            "Jules's stamps collapse"
        );
        assert_eq!(
            store.version_bytes(),
            collapsed.node_versions_bytes + collapsed.edge_versions_bytes
        );

        let mut tx = Tx::begin(&store, 11);
        for edge in [people.alix_gus, people.gus_mia] {
            tx.delete_edge(edge).unwrap();
        }
        for node in [people.alix, people.gus, people.mia, jules] {
            tx.delete_node(node).unwrap();
        }
        let epoch = tx.commit().unwrap();
        let deleted = store.memory();
        assert_eq!(store.collect(epoch), 6);
        let empty = store.memory();
        assert_eq!((empty.node_groups, empty.edge_groups), (0, 0));
        assert!(empty.total_bytes() < deleted.total_bytes());
        assert_eq!(
            empty.total_bytes(),
            empty.labels_bytes + empty.edge_types_bytes + empty.keys_bytes,
            "the dictionaries keep their names"
        );
        assert_eq!(empty.total_bytes(), store.heap_bytes());
        assert_eq!(store.memory_breakdown().2.max_chain_depth, 0);
    }

    /// A root with people, and trips where Vincent visited Paris. Returns
    /// Vincent.
    fn travels(root: &RowGroupStore) -> NodeId {
        people(root).unwrap();
        let trips = root.graph_or_create("trips").unwrap();
        let mut tx = Tx::begin(&*trips, 10);
        let vincent = tx
            .create_node(&["Person"], &[("name", Value::from("Vincent"))])
            .unwrap();
        let paris = tx
            .create_node(&["City"], &[("name", Value::from("Paris"))])
            .unwrap();
        tx.create_edge(vincent, paris, "VISITED", &[("since", Value::Int64(1994))])
            .unwrap();
        tx.commit().unwrap();
        vincent
    }

    /// A root without backward edges (10b), and its named graph, hold no
    /// incoming adjacency: the breakdown reports none, the outgoing bytes
    /// and every other part are a root's with them, and so after a write of
    /// 2,000 edges and a reload of the section into such a root.
    #[test]
    fn without_backward_edges_the_breakdown_has_no_incoming_bytes() {
        let (kept, scanned) = (
            RowGroupStore::new(),
            RowGroupStore::with_backward_edges(false),
        );
        for root in [&kept, &scanned] {
            travels(root);
            let mut tx = Tx::begin(root, 11);
            let nodes: Vec<NodeId> = (0..40)
                .map(|_| {
                    tx.create_node(&["Person"], &[("name", Value::from("Jules"))])
                        .unwrap()
                })
                .collect();
            for index in 0..2_000 {
                tx.create_edge(nodes[index % 40], nodes[(index * 7) % 40], "KNOWS", &[])
                    .unwrap();
            }
            tx.commit().unwrap();
        }
        let trips = scanned.graph("trips").unwrap();
        assert!(
            !trips.keeps_backward_edges(),
            "a named graph keeps none either"
        );
        assert_eq!(trips.memory().incoming_bytes, 0);
        let (with, without) = (kept.memory(), scanned.memory());
        assert!(with.incoming_bytes > 0);
        assert_eq!(without.incoming_bytes, 0);
        assert_eq!(
            RowGroupMemory {
                incoming_bytes: with.incoming_bytes,
                ..without.clone()
            },
            with,
            "every other part is the same"
        );
        assert_eq!(
            scanned.heap_bytes() + with.incoming_bytes,
            kept.heap_bytes()
        );
        let (_, indexes, _, _) = scanned.memory_breakdown();
        assert_eq!(indexes.backward_adjacency_bytes, 0);
        assert_eq!(
            indexes.forward_adjacency_bytes,
            kept.memory_breakdown().1.forward_adjacency_bytes
        );
        assert!(indexes.forward_adjacency_bytes > 0);

        let mut image = MemoryImage::new();
        image
            .begin_section(SectionType::LpgStore, LPG_SECTION_VERSION)
            .unwrap();
        kept.write_section(ChunkCaps::DEFAULT, &mut image).unwrap();
        let loaded = RowGroupStore::with_backward_edges(false);
        loaded
            .read_section(&*image.section_source(SectionType::LpgStore).unwrap())
            .unwrap();
        assert_eq!(loaded.memory().incoming_bytes, 0, "the file's are dropped");
        assert_eq!(loaded.memory().graphs, 2);
        assert_eq!(
            loaded.graph("trips").unwrap().in_degree(NodeId::new(1)),
            1,
            "Paris has its edge from Vincent, found in the edge table"
        );
    }

    /// A root's figures are its own and its named graphs': a dropped graph
    /// is no longer counted.
    #[test]
    fn a_root_counts_its_named_graphs() {
        let root = RowGroupStore::new();
        travels(&root);
        let trips = root.graph("trips").unwrap();
        let own = root.inner.read().memory();
        let mut expected = own.clone();
        expected.add(&trips.memory());
        let total = root.memory();
        assert_eq!(total, expected);
        assert_eq!(total.graphs, 2);
        assert!(trips.heap_bytes() > 0);
        assert_eq!(root.heap_bytes(), own.total_bytes() + trips.heap_bytes());
        assert_eq!(
            root.version_bytes(),
            own.node_versions_bytes + own.edge_versions_bytes + trips.version_bytes()
        );
        let (data, _, _, names) = root.memory_breakdown();
        assert_eq!(
            (names.label_count, names.edge_type_count),
            (4, 2),
            "Person and Employee, Person and City; KNOWS, VISITED"
        );
        assert_eq!(data.property_column_count, 4 + 2);
        assert_eq!(
            (root.node_count(), root.edge_count()),
            (3, 2),
            "the counts are the root's alone"
        );

        assert!(root.drop_graph("trips"));
        assert_eq!(root.memory(), own, "a dropped graph is not counted");
    }

    /// A loaded root's cold columns and chunks include its named graphs',
    /// and a read of a named graph's node decodes that graph's columns.
    #[test]
    fn a_loaded_roots_cold_columns_cover_its_named_graphs() {
        let root = RowGroupStore::new();
        let vincent = travels(&root);
        let mut image = MemoryImage::new();
        image
            .begin_section(SectionType::LpgStore, LPG_SECTION_VERSION)
            .unwrap();
        root.write_section(ChunkCaps::DEFAULT, &mut image).unwrap();
        let loaded = RowGroupStore::new();
        loaded
            .read_section(&*image.section_source(SectionType::LpgStore).unwrap())
            .unwrap();
        let trips = loaded.graph("trips").unwrap();
        let own = loaded.inner.read().memory();
        assert_eq!(own.cold_columns, 4, "name, age and city, and since");
        assert_eq!(trips.cold_columns(), 2, "name, and since");
        assert_eq!(loaded.cold_columns(), 6);
        assert!(trips.memory().node_cold_bytes > 0);

        assert!(trips.get_node(vincent).is_some());
        let decoded = trips.memory();
        assert_eq!(decoded.cold_columns, 1, "the edges' since stays encoded");
        assert_eq!(decoded.node_cold_bytes, 0);
        assert!(decoded.edge_cold_bytes > 0);
        assert_eq!(loaded.cold_columns(), 5);
        assert_eq!(loaded.heap_bytes(), loaded.memory().total_bytes());
    }
}
