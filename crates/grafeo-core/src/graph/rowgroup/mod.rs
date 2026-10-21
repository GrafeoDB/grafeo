//! The row-group store: one graph's nodes and edges as tables of row groups
//! (workstream H, masterplan X1 and 6.5), the store the engine holds its
//! graphs in (since 0.6.0, which deleted `LpgStore`).
//!
//! Row group `k` of a table holds the ids `[k * 65,536, (k + 1) * 65,536)`,
//! the rows of the file's row groups, and is created when its first row is.
//! In a row group:
//!
//! - each row's version: who created the row and who deleted it, each a
//!   transaction not committed yet or a commit epoch, kept per vector of
//!   2,048 rows and collapsed to one bit per row once no reader can tell
//!   the rows' versions apart (`versions`,
//!   [`RowGroupStore::collect`](crate::graph::rowgroup::RowGroupStore::collect));
//! - one column per property key: a typed vector with presence bits
//!   (`column`), holding each row's newest value;
//! - for nodes, one bitmap of rows per label (FD5 in memory) and the nodes'
//!   outgoing and incoming edges, each node's sorted by edge type and other
//!   node, with a hot delta for the edges added since (FD6 in memory,
//!   `adjacency`); a store built without backward edges
//!   ([`RowGroupStore::with_backward_edges`](crate::graph::rowgroup::RowGroupStore::with_backward_edges))
//!   keeps the outgoing ones only, and finds a node's incoming edges in the
//!   edge table instead;
//! - for edges, the source, target and type of each row;
//! - per column, and for the nodes' labels, update chains per vector of
//!   2,048 rows: what the rows held before the writes some readers must not
//!   see yet (`updates`, H2b).
//!
//! Every write is versioned: a create is seen by its transaction alone
//! until the commit, a delete takes effect at its commit epoch, and a reader
//! at an earlier epoch keeps what a later commit deleted. A value or label
//! write puts the newest value in place and keeps the one it replaced in an
//! undo node of its vector, stamped as a create is, so a reader that does
//! not see the write reads the kept value (#412, DuckDB's update chains).
//! Writes go through [`ChangeTarget`](crate::graph::apply::ChangeTarget) (`apply`), the read
//! traits read; one lock guards the store (H1a).
//!
//! A store is also a root of named graphs, each a store of its own
//! (`graphs`), each with its own property, text and
//! vector indexes, which the writes keep and the reads at a view recheck
//! (`index`). What the engine reads to report on a store, its memory by part
//! and every node and edge, is in `diagnostics`; the store's epoch and its
//! garbage collection over the named graphs in `epochs`, the collect itself
//! over the list of vectors with versions in `collect`. The rows the readers
//! of earlier formats, the snapshots and the copies of graph data load come
//! in through the replay writer (`load`, a
//! [`LoadTarget`](crate::graph::load::LoadTarget)), so the section reader
//! migrates the LPG sections of 0.5.x and of version 3 into a root. Under a
//! memory limit a node property column spills into files the engine writes
//! and maps, and reads from them without coming back onto the heap (`spill`).
//! A database that keeps history (a retention window) keeps the same
//! versions longer, and checkpoints them as `History` chunks; what time
//! travel needs besides, the history calls and the floor of what the store
//! answers, is in `history`.

mod adjacency;
mod adjacency_chunk;
mod apply;
mod bitset;
mod collect;
mod column;
mod diagnostics;
mod epochs;
mod graphs;
mod history;
mod index;
mod load;
mod read;
pub mod section;
mod spill;
mod updates;
mod versions;
mod write;

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};

use grafeo_common::change::{EdgeImage, Labels, NodeImage, Properties};
use grafeo_common::types::{EdgeId, EpochId, NodeId, PropertyKey, PropertyMap, ReadView, Value};
use grafeo_common::utils::hash::FxHashMap;
use parking_lot::RwLock;

use self::adjacency::{Adjacency, Adjacent};
use self::bitset::Bitset;
use self::column::Columns;
pub use self::diagnostics::RowGroupMemory;
#[cfg(any(feature = "text-index", feature = "vector-index"))]
pub use self::index::IndexSearch;
pub use self::spill::{ColumnSpill, ColumnSpillReport};
use self::updates::Updates;
#[cfg(test)]
pub(crate) use self::versions::VECTOR_ROWS;
use self::versions::Versions;
use crate::graph::lpg::dictionary::NameDictionary;
use crate::graph::lpg::{Edge, Node};

/// The rows of a row group, as in the file (the section's `max_rows`).
pub const ROWS_PER_GROUP: u64 = 65_536;

/// A version stamp nothing holds: a row never created, or not deleted.
const ABSENT: u64 = u64::MAX;

/// The bit that marks a stamp as a transaction's, not committed yet; the
/// other bits hold its id. A stamp without it is a commit epoch.
const PENDING: u64 = 1 << 63;

/// The row group and row of an id.
fn locate(id: u64) -> (u64, usize) {
    (
        id / ROWS_PER_GROUP,
        usize::try_from(id % ROWS_PER_GROUP).expect("a row below 65,536"),
    )
}

/// Who reads: a transaction at its snapshot (its own pending versions
/// included), or a committed reader at an epoch.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Read {
    Transaction { id: u64, snapshot: u64 },
    At(u64),
}

impl Read {
    /// Whether this reader sees what `stamp` made.
    fn sees(self, stamp: u64) -> bool {
        if stamp == ABSENT {
            return false;
        }
        if stamp & PENDING != 0 {
            return matches!(self, Self::Transaction { id, .. } if id == stamp & !PENDING);
        }
        match self {
            Self::Transaction { snapshot, .. } => stamp <= snapshot,
            Self::At(epoch) => stamp <= epoch,
        }
    }

    /// A transaction's reader.
    fn transaction(id: grafeo_common::types::TransactionId, snapshot: EpochId) -> Self {
        Self::Transaction {
            id: id.as_u64(),
            snapshot: snapshot.as_u64(),
        }
    }

    /// A committed reader at `epoch`.
    fn at(epoch: EpochId) -> Self {
        Self::At(epoch.as_u64())
    }

    /// The reader of a view.
    fn view(view: ReadView) -> Self {
        match view.transaction {
            Some(id) => Self::transaction(id, view.epoch),
            None => Self::at(view.epoch),
        }
    }
}

/// A row's version: who created it and who deleted it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct RowVersion {
    created: u64,
    deleted: u64,
}

impl RowVersion {
    /// No row.
    const NONE: Self = Self {
        created: ABSENT,
        deleted: ABSENT,
    };

    /// Whether `read` sees the row: it sees the create and not the delete.
    fn visible(self, read: Read) -> bool {
        read.sees(self.created) && !read.sees(self.deleted)
    }
}

/// Grows `values` so `row` is in it.
fn grow<T: Clone>(values: &mut Vec<T>, row: usize, filler: T) {
    if row >= values.len() {
        values.resize(row + 1, filler);
    }
}

/// The ids of the labels whose bitmaps hold `row`, ascending.
fn label_ids(labels: &FxHashMap<u32, Bitset>, row: usize) -> Vec<u32> {
    let mut ids: Vec<u32> = labels
        .iter()
        .filter(|(_, rows)| rows.get(row))
        .map(|(id, _)| *id)
        .collect();
    ids.sort_unstable();
    ids
}

/// A row group of the node table.
#[derive(Debug, Default)]
struct NodeGroup {
    versions: Versions,
    /// The rows of each label, by label id: the rows' newest labels.
    labels: FxHashMap<u32, Bitset>,
    /// The label ids rows had before the label writes some readers must not
    /// see, ascending.
    label_updates: Updates<Box<[u32]>>,
    columns: Columns,
    /// The rows' outgoing edges.
    outgoing: Adjacency,
    /// The rows' incoming edges.
    incoming: Adjacency,
    /// The vectors the store lists, a bit each (`collect`).
    listed: u32,
}

impl NodeGroup {
    fn version(&self, row: usize) -> RowVersion {
        self.versions.get(row)
    }

    /// The row's newest label ids, ascending.
    fn labels_of(&self, row: usize) -> Vec<u32> {
        label_ids(&self.labels, row)
    }

    /// The row's label ids as `read` sees them, ascending.
    fn labels_at(&self, row: usize, read: Read) -> Vec<u32> {
        if !self.label_updates.is_empty()
            && let Some(kept) = self.label_updates.saved(row, read)
        {
            return kept.to_vec();
        }
        self.labels_of(row)
    }

    /// Whether the row has label id `label` as `read` sees it.
    fn has_label_at(&self, row: usize, label: u32, read: Read) -> bool {
        if !self.label_updates.is_empty()
            && let Some(kept) = self.label_updates.saved(row, read)
        {
            return kept.binary_search(&label).is_ok();
        }
        self.labels.get(&label).is_some_and(|rows| rows.get(row))
    }

    /// Gives the row exactly the labels `ids`.
    fn set_labels(&mut self, row: usize, ids: &[u32]) {
        for (id, rows) in &mut self.labels {
            rows.put(row, ids.contains(id));
        }
        for id in ids {
            self.labels.entry(*id).or_default().set(row);
        }
    }

    /// One direction of the group's adjacency.
    fn adjacency(&self, outgoing: bool) -> &Adjacency {
        if outgoing {
            &self.outgoing
        } else {
            &self.incoming
        }
    }

    fn adjacency_mut(&mut self, outgoing: bool) -> &mut Adjacency {
        if outgoing {
            &mut self.outgoing
        } else {
            &mut self.incoming
        }
    }

    /// Takes the row out whole: its version, labels, values and adjacency.
    fn remove(&mut self, row: usize) {
        self.versions.remove(row);
        self.clear(row);
    }

    /// Takes out what a row that is gone held: labels, values, adjacency,
    /// and what the chains kept of them.
    fn clear(&mut self, row: usize) {
        self.set_labels(row, &[]);
        if !self.label_updates.is_empty() {
            self.label_updates.remove_row(row);
        }
        self.columns.clear_row(row);
        self.outgoing.clear(row);
        self.incoming.clear(row);
    }
}

/// A row group of the edge table.
#[derive(Debug, Default)]
struct EdgeGroup {
    versions: Versions,
    src: Vec<u64>,
    dst: Vec<u64>,
    edge_type: Vec<u32>,
    columns: Columns,
    /// The vectors the store lists, a bit each (`collect`).
    listed: u32,
}

impl EdgeGroup {
    fn version(&self, row: usize) -> RowVersion {
        self.versions.get(row)
    }

    /// The row's source, target and type id; the row exists.
    fn ends(&self, row: usize) -> (u64, u64, u32) {
        (self.src[row], self.dst[row], self.edge_type[row])
    }

    fn set_ends(&mut self, row: usize, src: u64, dst: u64, edge_type: u32) {
        grow(&mut self.src, row, 0);
        grow(&mut self.dst, row, 0);
        grow(&mut self.edge_type, row, 0);
        self.src[row] = src;
        self.dst[row] = dst;
        self.edge_type[row] = edge_type;
    }

    fn remove(&mut self, row: usize) {
        self.versions.remove(row);
        self.columns.clear_row(row);
    }
}

/// The committed counts: nodes, edges, nodes per label id, edges per type
/// id. They move when a write is stamped, never at a transaction's write.
#[derive(Debug, Default)]
struct Counts {
    nodes: i64,
    edges: i64,
    labels: Vec<i64>,
    edge_types: Vec<i64>,
}

impl Counts {
    fn label(&mut self, id: u32, delta: i64) {
        let index = id as usize;
        grow(&mut self.labels, index, 0);
        self.labels[index] += delta;
    }

    fn edge_type(&mut self, id: u32, delta: i64) {
        let index = id as usize;
        grow(&mut self.edge_types, index, 0);
        self.edge_types[index] += delta;
    }
}

/// Whether a store keeps each node's incoming edges, as `LpgStore`'s
/// backward adjacency (`backward_edges`): fixed when the store is built.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
enum IncomingEdges {
    /// Kept with each node, as the outgoing ones are.
    #[default]
    Kept,
    /// Not kept: a read of a node's incoming edges scans the edge table's
    /// targets (see [`Inner::incoming_by_scan`]).
    NotKept,
}

/// Everything the store's lock guards.
#[derive(Debug, Default)]
struct Inner {
    nodes: BTreeMap<u64, NodeGroup>,
    edges: BTreeMap<u64, EdgeGroup>,
    labels: NameDictionary,
    edge_types: NameDictionary,
    keys: NameDictionary,
    counts: Counts,
    /// The property, text and vector indexes (`index`).
    indexes: index::Indexes,
    /// The row groups with a listed vector (`collect`).
    listed: collect::Listed,
    /// Whether the nodes' incoming edges are kept.
    incoming_edges: IncomingEdges,
}

impl Inner {
    /// The node's group and row, if the row exists.
    fn node(&self, id: u64) -> Option<(&NodeGroup, usize)> {
        let (group, row) = locate(id);
        let group = self.nodes.get(&group)?;
        group.versions.exists(row).then_some((group, row))
    }

    fn node_mut(&mut self, id: u64) -> Option<(&mut NodeGroup, usize)> {
        let (group, row) = locate(id);
        let group = self.nodes.get_mut(&group)?;
        if group.versions.exists(row) {
            Some((group, row))
        } else {
            None
        }
    }

    fn edge(&self, id: u64) -> Option<(&EdgeGroup, usize)> {
        let (group, row) = locate(id);
        let group = self.edges.get(&group)?;
        group.versions.exists(row).then_some((group, row))
    }

    fn edge_mut(&mut self, id: u64) -> Option<(&mut EdgeGroup, usize)> {
        let (group, row) = locate(id);
        let group = self.edges.get_mut(&group)?;
        if group.versions.exists(row) {
            Some((group, row))
        } else {
            None
        }
    }

    /// The node's group and row, if `read` sees the node: one lookup of
    /// its vector's versions, a bit test while the vector is flat.
    fn node_seen(&self, id: u64, read: Read) -> Option<(&NodeGroup, usize)> {
        let (group, row) = locate(id);
        let group = self.nodes.get(&group)?;
        group.versions.visible(row, read).then_some((group, row))
    }

    /// The edge's group and row, if `read` sees the edge (see
    /// [`node_seen`](Self::node_seen)).
    fn edge_seen(&self, id: u64, read: Read) -> Option<(&EdgeGroup, usize)> {
        let (group, row) = locate(id);
        let group = self.edges.get(&group)?;
        group.versions.visible(row, read).then_some((group, row))
    }

    fn node_visible(&self, id: u64, read: Read) -> bool {
        self.node_seen(id, read).is_some()
    }

    fn edge_visible(&self, id: u64, read: Read) -> bool {
        self.edge_seen(id, read).is_some()
    }

    /// The names of label ids, in their order.
    fn label_names(&self, ids: &[u32]) -> Labels {
        ids.iter()
            .filter_map(|id| self.labels.get_name(*id).cloned())
            .collect()
    }

    /// The newest values of a row's columns, sorted by key name.
    fn values(&self, columns: &Columns, row: usize) -> Properties {
        self.named(columns.row(row))
    }

    /// The values of a row's columns as `read` sees them, sorted by key
    /// name.
    fn values_at(&self, columns: &Columns, row: usize, read: Read) -> Properties {
        self.named(columns.row_at(row, read))
    }

    /// Values by key id as values by key name, sorted by it.
    fn named(&self, values: impl Iterator<Item = (u32, Value)>) -> Properties {
        let mut values: Properties = values
            .filter_map(|(key, value)| {
                self.keys
                    .get_name(key)
                    .map(|name| (PropertyKey::new(name.as_str()), value))
            })
            .collect();
        values.sort_by(|(a, _), (b, _)| a.cmp(b));
        values
    }

    /// The node's labels and values as they are now.
    fn node_image(&self, id: u64) -> Option<NodeImage> {
        let (group, row) = self.node(id)?;
        Some(NodeImage {
            labels: self.label_names(&group.labels_of(row)),
            properties: self.values(&group.columns, row),
        })
    }

    /// The edge's ends, type and values as they are now.
    fn edge_image(&self, id: u64) -> Option<EdgeImage> {
        let (group, row) = self.edge(id)?;
        let (src, dst, edge_type) = group.ends(row);
        Some(EdgeImage {
            src: NodeId::new(src),
            dst: NodeId::new(dst),
            edge_type: self.edge_types.get_name(edge_type)?.clone(),
            properties: self.values(&group.columns, row),
            deleted_with: None,
        })
    }

    /// The node as `read` sees it.
    fn read_node(&self, id: u64, read: Read) -> Option<Node> {
        let (group, row) = self.node(id)?;
        if !group.version(row).visible(read) {
            return None;
        }
        Some(Node {
            id: NodeId::new(id),
            labels: self.label_names(&group.labels_at(row, read)),
            properties: self
                .values_at(&group.columns, row, read)
                .into_iter()
                .collect::<PropertyMap>(),
        })
    }

    /// The edge as `read` sees it.
    fn read_edge(&self, id: u64, read: Read) -> Option<Edge> {
        let (group, row) = self.edge(id)?;
        if !group.version(row).visible(read) {
            return None;
        }
        let (src, dst, edge_type) = group.ends(row);
        Some(Edge {
            id: EdgeId::new(id),
            src: NodeId::new(src),
            dst: NodeId::new(dst),
            edge_type: self.edge_types.get_name(edge_type)?.clone(),
            properties: self
                .values_at(&group.columns, row, read)
                .into_iter()
                .collect::<PropertyMap>(),
        })
    }

    /// Whether `read` sees an edge of node `id`, either way: without the
    /// incoming edges kept, its incoming ones by a scan of the edge table.
    fn sees_an_edge_of(&self, id: u64, read: Read) -> bool {
        self.node(id).is_some_and(|(group, row)| {
            group
                .outgoing
                .of(row)
                .any(|adjacent| self.edge_visible(adjacent.edge, read))
                || if self.keeps_incoming() {
                    group
                        .incoming
                        .of(row)
                        .any(|adjacent| self.edge_visible(adjacent.edge, read))
                } else {
                    self.incoming_by_scan(id)
                        .any(|adjacent| self.edge_visible(adjacent.edge, read))
                }
        })
    }

    /// Whether the nodes' incoming edges are kept.
    fn keeps_incoming(&self) -> bool {
        self.incoming_edges == IncomingEdges::Kept
    }

    /// The directions of adjacency the store keeps for an edge's two ends:
    /// (node, other node, outgoing) for its source, and for its target when
    /// the incoming edges are kept.
    fn kept_ends(&self, src: u64, dst: u64) -> impl Iterator<Item = (u64, u64, bool)> + use<> {
        let incoming = self.keeps_incoming();
        [(src, dst, true), (dst, src, false)]
            .into_iter()
            .filter(move |(_, _, outgoing)| *outgoing || incoming)
    }

    /// Node `node`'s incoming edges as the edge table holds them, by edge
    /// id: every edge row with that target not collected yet, deleted ones
    /// included, as the incoming adjacency would list them. A store that
    /// keeps no incoming edges reads them so (`LpgStore` without backward
    /// adjacency scans its edges alike): one pass over the target column of
    /// every edge row group.
    fn incoming_by_scan(&self, node: u64) -> impl Iterator<Item = Adjacent> + '_ {
        self.edges.iter().flat_map(move |(index, group)| {
            group
                .dst
                .iter()
                .enumerate()
                .filter(move |(row, dst)| **dst == node && group.versions.exists(*row))
                .map(move |(row, _)| Adjacent {
                    edge_type: group.edge_type[row],
                    other: group.src[row],
                    edge: index * ROWS_PER_GROUP + row as u64,
                })
        })
    }

    /// The label id of `name`, which every name a node had keeps (the
    /// dictionary forgets none).
    fn label_id(&self, name: &str) -> Option<u32> {
        self.labels.get_id(name)
    }
}

/// One graph's store of row groups.
#[derive(Debug, Default)]
pub struct RowGroupStore {
    inner: RwLock<Inner>,
    next_node_id: AtomicU64,
    next_edge_id: AtomicU64,
    /// The latest epoch a write was stamped at.
    epoch: AtomicU64,
    /// Whether the store lists a vector (`collect`): a collect while it
    /// lists none returns at once, without taking the lock.
    has_listed: AtomicBool,
    /// The writes stamped at once and the entries commits stamped since the
    /// last collect: each may have left a version that only a reader before
    /// its epoch needs (`collect`).
    settled: AtomicUsize,
    /// One past the horizon of the last collect when it left committed
    /// versions some reader at that horizon may still need, 0 when it left
    /// none (`collect`).
    held_at: AtomicU64,
    /// Its named graphs, its graph id, and whether it was dropped (`graphs`).
    graphs: graphs::Graphs,
    /// Whether the database keeps history (a retention window, H2c): replay
    /// then keeps what it replaces as an immediate write does (`history`).
    history: AtomicBool,
    /// The oldest epoch the store answers reads at: the newest horizon it
    /// was collected at, or the one its history was loaded from (`history`).
    floor: AtomicU64,
}

/// What a collect took out: rows, and the values and label sets kept for
/// older readers.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
struct Collected {
    rows: usize,
    values: usize,
}

impl RowGroupStore {
    /// An empty store that keeps each node's incoming edges.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// An empty store that keeps each node's incoming edges, or with
    /// `backward_edges` false keeps none, as `LpgStoreConfig::backward_edges`
    /// and the database's `Config::backward_edges` say: an edge is then in
    /// its source's adjacency only, a read of a node's incoming edges scans
    /// the edge table, [`has_backward_adjacency`](crate::graph::GraphStore::has_backward_adjacency)
    /// is false so a planner can avoid them, and a checkpoint writes no
    /// incoming adjacency. Its named graphs are built the same way.
    #[must_use]
    pub fn with_backward_edges(backward_edges: bool) -> Self {
        let store = Self::default();
        store.inner.write().incoming_edges = if backward_edges {
            IncomingEdges::Kept
        } else {
            IncomingEdges::NotKept
        };
        store
    }

    /// Whether it keeps each node's incoming edges (see
    /// [`with_backward_edges`](Self::with_backward_edges)).
    #[must_use]
    pub fn keeps_backward_edges(&self) -> bool {
        self.inner.read().keeps_incoming()
    }

    /// The vectors of rows, nodes and edges, whose version info holds
    /// stamps (the others hold one bit per row).
    #[cfg(test)]
    pub(crate) fn stamped_vectors(&self) -> usize {
        let inner = self.inner.read();
        inner
            .nodes
            .values()
            .map(|group| group.versions.stamped())
            .sum::<usize>()
            + inner
                .edges
                .values()
                .map(|group| group.versions.stamped())
                .sum::<usize>()
    }

    /// The label, edge type and property key dictionaries, each as its
    /// names with their ids (ids ascending) and its next id.
    #[cfg(test)]
    pub(crate) fn dictionaries(&self) -> [(Vec<(u32, String)>, u32); 3] {
        let inner = self.inner.read();
        let listed = |dictionary: &NameDictionary| {
            let names = dictionary
                .iter()
                .map(|(id, name)| (id, name.to_string()))
                .collect();
            (names, dictionary.next_id())
        };
        [
            listed(&inner.labels),
            listed(&inner.edge_types),
            listed(&inner.keys),
        ]
    }

    /// The undo nodes of the value and label chains, nodes and edges.
    #[cfg(test)]
    pub(crate) fn update_nodes(&self) -> usize {
        let inner = self.inner.read();
        inner
            .nodes
            .values()
            .map(|group| group.columns.update_nodes() + group.label_updates.nodes())
            .sum::<usize>()
            + inner
                .edges
                .values()
                .map(|group| group.columns.update_nodes())
                .sum::<usize>()
    }

    /// The reader at the store's current epoch.
    fn now(&self) -> Read {
        Read::At(self.epoch.load(Ordering::Acquire))
    }

    /// The view of a committed reader at the store's current epoch.
    fn now_view(&self) -> ReadView {
        ReadView::committed(EpochId::new(self.epoch.load(Ordering::Acquire)))
    }
}

#[cfg(test)]
mod tests {
    use super::{ABSENT, PENDING, Read, RowVersion, locate};

    /// A row `collect` takes out leaves no label, value or adjacency entry
    /// in its row group, nor does its edge.
    #[test]
    fn a_collected_row_leaves_nothing_behind() {
        use crate::graph::conformance::{Tx, people};

        let store = super::RowGroupStore::new();
        let people = people(&store).unwrap();
        let mut tx = Tx::begin(&store, 10);
        tx.delete_edge(people.gus_mia).unwrap();
        tx.delete_node(people.mia).unwrap();
        let epoch = tx.commit().unwrap();
        let (_, mia) = locate(people.mia.as_u64());
        let (_, gus) = locate(people.gus.as_u64());
        let (_, edge) = locate(people.gus_mia.as_u64());
        {
            let inner = store.inner.read();
            let nodes = &inner.nodes[&0];
            assert_eq!(
                nodes.labels_of(mia).len(),
                1,
                "Mia is a Person until the collect"
            );
            assert_eq!(nodes.outgoing.of(gus).count(), 1);
        }
        assert_eq!(store.collect(epoch), 2);
        let inner = store.inner.read();
        let nodes = &inner.nodes[&0];
        assert_eq!(nodes.labels_of(mia), Vec::<u32>::new());
        assert_eq!(
            nodes.columns.row(mia).count(),
            0,
            "no value of Mia's is left"
        );
        assert_eq!(nodes.incoming.of(mia).count(), 0);
        assert_eq!(
            nodes.outgoing.of(gus).count(),
            0,
            "nor Gus's entry of the edge"
        );
        assert_eq!(
            inner.edges[&0].columns.row(edge).count(),
            0,
            "nor the edge's since"
        );
    }

    #[test]
    fn ids_map_to_their_row_group_and_row() {
        assert_eq!(locate(0), (0, 0));
        assert_eq!(locate(65_535), (0, 65_535));
        assert_eq!(locate(65_536), (1, 0));
        assert_eq!(locate((1 << 40) + 2), (1 << 24, 2));
    }

    #[test]
    fn a_reader_sees_committed_versions_up_to_its_epoch_and_its_own_pending_ones() {
        let transaction = 88;
        let committed = RowVersion {
            created: 3,
            deleted: ABSENT,
        };
        assert!(!committed.visible(Read::At(2)));
        assert!(committed.visible(Read::At(3)));
        let deleted_later = RowVersion {
            created: 3,
            deleted: 19,
        };
        assert!(deleted_later.visible(Read::At(18)) && !deleted_later.visible(Read::At(19)));
        let pending = RowVersion {
            created: PENDING | transaction,
            deleted: ABSENT,
        };
        assert!(pending.visible(Read::Transaction {
            id: 88,
            snapshot: 0
        }));
        assert!(!pending.visible(Read::Transaction {
            id: 7,
            snapshot: 1_000
        }));
        assert!(!pending.visible(Read::At(1_000)));
        let deleting = RowVersion {
            created: 3,
            deleted: PENDING | transaction,
        };
        assert!(!deleting.visible(Read::Transaction {
            id: 88,
            snapshot: 3
        }));
        assert!(deleting.visible(Read::Transaction { id: 7, snapshot: 3 }));
        assert!(
            deleting.visible(Read::At(3)),
            "a delete takes effect at its commit"
        );
        assert_eq!(RowVersion::NONE.created, ABSENT);
    }
}
