//! The row-group store: one graph's nodes and edges as tables of row groups
//! (workstream H, masterplan X1 and 6.5). Not used by the engine yet: it
//! switches from `LpgStore` once, at H2d.
//!
//! Row group `k` of a table holds the ids `[k * 65,536, (k + 1) * 65,536)`,
//! the rows of the file's row groups, and is created when its first row is.
//! In a row group:
//!
//! - each row's version: who created the row and who deleted it, each a
//!   transaction not committed yet or a commit epoch;
//! - one column per property key: a typed vector with presence bits
//!   (`column`);
//! - for nodes, one bitmap of rows per label (FD5 in memory) and the nodes'
//!   outgoing and incoming edges, each node's sorted by edge type and other
//!   node, with a hot delta for the edges added since (FD6 in memory,
//!   `adjacency`);
//! - for edges, the source, target and type of each row.
//!
//! Creates and deletes are versioned: a create is seen by its transaction
//! alone until the commit, a delete takes effect at its commit epoch, and a
//! reader at an earlier epoch keeps what a later commit deleted. Values and
//! labels are written in place, as `LpgStore`'s default build does, so every
//! reader sees an open transaction's values and labels (#412) until H2b's
//! update chains. Writes go through [`ChangeTarget`](crate::graph::apply::ChangeTarget) (`apply`), the read
//! traits read; one lock guards the store (H1a).

mod adjacency;
mod adjacency_chunk;
mod apply;
mod bitset;
mod column;
mod read;
pub mod section;
mod write;

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicU64, Ordering};

use grafeo_common::change::{EdgeImage, Labels, NodeImage, Properties};
use grafeo_common::types::{EdgeId, EpochId, NodeId, PropertyKey, PropertyMap};
use grafeo_common::utils::hash::FxHashMap;
use parking_lot::RwLock;

use self::adjacency::{Adjacency, Adjacent};
use self::bitset::Bitset;
use self::column::Columns;
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

    /// Whether the row exists (created, by anyone, and not undone).
    fn exists(self) -> bool {
        self.created != ABSENT
    }

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

/// A row group of the node table.
#[derive(Debug, Default)]
struct NodeGroup {
    versions: Vec<RowVersion>,
    /// The rows of each label, by label id.
    labels: FxHashMap<u32, Bitset>,
    columns: Columns,
    /// The rows' outgoing edges.
    outgoing: Adjacency,
    /// The rows' incoming edges.
    incoming: Adjacency,
}

impl NodeGroup {
    fn version(&self, row: usize) -> RowVersion {
        self.versions.get(row).copied().unwrap_or(RowVersion::NONE)
    }

    fn version_mut(&mut self, row: usize) -> &mut RowVersion {
        grow(&mut self.versions, row, RowVersion::NONE);
        &mut self.versions[row]
    }

    /// The row's label ids, ascending.
    fn labels_of(&self, row: usize) -> Vec<u32> {
        let mut ids: Vec<u32> = self
            .labels
            .iter()
            .filter(|(_, rows)| rows.get(row))
            .map(|(id, _)| *id)
            .collect();
        ids.sort_unstable();
        ids
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
        *self.version_mut(row) = RowVersion::NONE;
        self.set_labels(row, &[]);
        self.columns.clear_row(row);
        self.outgoing.clear(row);
        self.incoming.clear(row);
    }

    fn heap_bytes(&self) -> usize {
        self.versions.capacity() * std::mem::size_of::<RowVersion>()
            + self.labels.values().map(Bitset::heap_bytes).sum::<usize>()
            + self.columns.heap_bytes()
            + self.outgoing.heap_bytes()
            + self.incoming.heap_bytes()
    }
}

/// A row group of the edge table.
#[derive(Debug, Default)]
struct EdgeGroup {
    versions: Vec<RowVersion>,
    src: Vec<u64>,
    dst: Vec<u64>,
    edge_type: Vec<u32>,
    columns: Columns,
}

impl EdgeGroup {
    fn version(&self, row: usize) -> RowVersion {
        self.versions.get(row).copied().unwrap_or(RowVersion::NONE)
    }

    fn version_mut(&mut self, row: usize) -> &mut RowVersion {
        grow(&mut self.versions, row, RowVersion::NONE);
        &mut self.versions[row]
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
        *self.version_mut(row) = RowVersion::NONE;
        self.columns.clear_row(row);
    }

    fn heap_bytes(&self) -> usize {
        self.versions.capacity() * std::mem::size_of::<RowVersion>()
            + (self.src.capacity() + self.dst.capacity()) * std::mem::size_of::<u64>()
            + self.edge_type.capacity() * std::mem::size_of::<u32>()
            + self.columns.heap_bytes()
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

/// Everything the store's lock guards.
#[derive(Debug, Default)]
struct Inner {
    nodes: BTreeMap<u64, NodeGroup>,
    edges: BTreeMap<u64, EdgeGroup>,
    labels: NameDictionary,
    edge_types: NameDictionary,
    keys: NameDictionary,
    counts: Counts,
}

impl Inner {
    /// The node's group and row, if the row exists.
    fn node(&self, id: u64) -> Option<(&NodeGroup, usize)> {
        let (group, row) = locate(id);
        let group = self.nodes.get(&group)?;
        group.version(row).exists().then_some((group, row))
    }

    fn node_mut(&mut self, id: u64) -> Option<(&mut NodeGroup, usize)> {
        let (group, row) = locate(id);
        let group = self.nodes.get_mut(&group)?;
        if group.version(row).exists() {
            Some((group, row))
        } else {
            None
        }
    }

    fn edge(&self, id: u64) -> Option<(&EdgeGroup, usize)> {
        let (group, row) = locate(id);
        let group = self.edges.get(&group)?;
        group.version(row).exists().then_some((group, row))
    }

    fn edge_mut(&mut self, id: u64) -> Option<(&mut EdgeGroup, usize)> {
        let (group, row) = locate(id);
        let group = self.edges.get_mut(&group)?;
        if group.version(row).exists() {
            Some((group, row))
        } else {
            None
        }
    }

    fn node_visible(&self, id: u64, read: Read) -> bool {
        self.node(id)
            .is_some_and(|(group, row)| group.version(row).visible(read))
    }

    fn edge_visible(&self, id: u64, read: Read) -> bool {
        self.edge(id)
            .is_some_and(|(group, row)| group.version(row).visible(read))
    }

    /// The names of label ids, in their order.
    fn label_names(&self, ids: &[u32]) -> Labels {
        ids.iter()
            .filter_map(|id| self.labels.get_name(*id).cloned())
            .collect()
    }

    /// The values of a row's columns, sorted by key name.
    fn values(&self, columns: &Columns, row: usize) -> Properties {
        let mut values: Properties = columns
            .row(row)
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
            labels: self.label_names(&group.labels_of(row)),
            properties: self
                .values(&group.columns, row)
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
                .values(&group.columns, row)
                .into_iter()
                .collect::<PropertyMap>(),
        })
    }

    /// Whether `read` sees an edge of node `id`, either way.
    fn sees_an_edge_of(&self, id: u64, read: Read) -> bool {
        self.node(id).is_some_and(|(group, row)| {
            group
                .outgoing
                .of(row)
                .chain(group.incoming.of(row))
                .any(|adjacent| self.edge_visible(adjacent.edge, read))
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
}

impl RowGroupStore {
    /// An empty store.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// The heap bytes its row groups and dictionaries hold (an estimate).
    #[must_use]
    pub fn heap_bytes(&self) -> usize {
        let inner = self.inner.read();
        inner
            .nodes
            .values()
            .map(NodeGroup::heap_bytes)
            .sum::<usize>()
            + inner
                .edges
                .values()
                .map(EdgeGroup::heap_bytes)
                .sum::<usize>()
            + inner.labels.heap_bytes()
            + inner.edge_types.heap_bytes()
            + inner.keys.heap_bytes()
    }

    /// The property columns of its row groups that still hold their chunks
    /// as a load left them, encoded: a column decodes on its first use.
    #[must_use]
    pub fn cold_columns(&self) -> usize {
        let inner = self.inner.read();
        inner
            .nodes
            .values()
            .map(|group| group.columns.cold_count())
            .sum::<usize>()
            + inner
                .edges
                .values()
                .map(|group| group.columns.cold_count())
                .sum::<usize>()
    }

    /// Moves the store's epoch to `epoch` if it is later.
    fn sync_epoch(&self, epoch: EpochId) {
        self.epoch.fetch_max(epoch.as_u64(), Ordering::AcqRel);
    }

    /// The reader at the store's current epoch.
    fn now(&self) -> Read {
        Read::At(self.epoch.load(Ordering::Acquire))
    }
}

#[cfg(test)]
mod tests {
    use super::{ABSENT, PENDING, Read, RowVersion, locate};

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
        assert!(!RowVersion::NONE.exists());
    }
}
