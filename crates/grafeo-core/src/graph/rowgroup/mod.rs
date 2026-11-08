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
//! Writes go through [`GraphBackend`] (`apply`), the read
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

mod account;
mod adjacency;
mod adjacency_chunk;
mod apply;
mod bitset;
mod checkpoint;
mod collect;
mod column;
mod compress;
mod degree_samples;
mod diagnostics;
mod epochs;
mod graphs;
mod history;
mod index;
mod key_column;
mod load;
mod read;
pub mod section;
mod slots;
mod spill;
mod updates;
mod versions;
mod write;

use std::collections::BTreeMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};

use grafeo_common::change::{DataOp, EdgeImage, Labels, NodeImage, Properties, Table};
use grafeo_common::memory::buffer::BufferManager;
use grafeo_common::types::{EdgeId, EpochId, NodeId, PropertyKey, PropertyMap, ReadView, Value};
use grafeo_common::utils::hash::FxHashMap;
use parking_lot::RwLock;
use smallvec::SmallVec;

use self::account::{LINES, Line, Lines, StoreAccount, add_lines};
use self::adjacency::{Adjacency, Adjacent};
use self::bitset::Bitset;
pub use self::checkpoint::CheckpointSnapshot;
use self::checkpoint::Marks;
use self::column::Columns;
pub use self::compress::CompressReport;
pub use self::diagnostics::RowGroupMemory;
#[cfg(any(feature = "text-index", feature = "vector-index"))]
pub use self::index::IndexSearch;
pub use self::key_column::KeyColumnQuery;
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

/// Grows `values` so `row` is in it: how an edge row group's ends and the
/// counts grow, whose capacity the account reads.
fn grow<T: Clone>(values: &mut Vec<T>, row: usize, filler: T) {
    if row >= values.len() {
        values.resize(row + 1, filler);
    }
}

/// The label ids a row held before a label write, ascending, as a label
/// chain keeps them: up to four in place, so the chain of a label write
/// keeps them without an allocation (a row rarely has more).
type LabelIds = SmallVec<[u32; 4]>;

/// The ids of the labels whose bitmaps hold `row`, ascending, as a label
/// chain keeps them.
fn kept_label_ids(labels: &FxHashMap<u32, Bitset>, row: usize) -> LabelIds {
    let mut ids: LabelIds = labels
        .iter()
        .filter(|(_, rows)| rows.get(row))
        .map(|(id, _)| *id)
        .collect();
    ids.sort_unstable();
    ids
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
    label_updates: Updates<LabelIds>,
    columns: Columns,
    /// The rows' outgoing edges.
    outgoing: Adjacency,
    /// The rows' incoming edges.
    incoming: Adjacency,
    /// The vectors the store lists, a bit each (`collect`).
    listed: u32,
    /// What changed since the store's baseline image (`checkpoint`).
    marks: Marks,
    /// The heap bytes of the label bitmaps' words, kept as the labels
    /// change (`account`).
    label_bytes: usize,
}

/// `count` less `bytes`, for a count a part keeps of what it holds, which
/// never goes below what it takes out: saturating, so a drift can neither
/// panic under the store's lock nor wrap into a huge grant (a debug build
/// stops at it).
fn sub_count(count: usize, bytes: usize) -> usize {
    debug_assert!(bytes <= count, "a count drifted: {count} less {bytes}");
    count.saturating_sub(bytes)
}

/// The heap bytes of a row group's entry in its table: its index and its
/// struct.
const fn table_entry_bytes<G>() -> usize {
    std::mem::size_of::<u64>() + std::mem::size_of::<G>()
}

/// The heap bytes of the map of a node row group's label bitmaps, without
/// their words.
fn label_map_bytes(labels: &FxHashMap<u32, Bitset>) -> usize {
    labels.capacity() * (std::mem::size_of::<u32>() + std::mem::size_of::<Bitset>())
}

impl NodeGroup {
    /// What it holds, line by line, as the account counts it: a few loads
    /// per part (see `account`).
    fn lines(&self) -> Lines {
        let mut lines = [0; LINES];
        self.add_lines_to(&mut lines);
        lines
    }

    /// Adds [`lines`](Self::lines) to `lines`, in place.
    #[inline]
    fn add_lines_to(&self, lines: &mut Lines) {
        let columns = self.columns.counts();
        let mut add = |line: Line, bytes: usize| {
            lines[line.index()] = lines[line.index()].wrapping_add(bytes);
        };
        add(
            Line::NodeTables,
            table_entry_bytes::<Self>() + self.marks.heap_bytes(),
        );
        add(Line::NodeColumns, columns.hot);
        add(Line::NodeCold, columns.cold);
        add(Line::NodeColdChunks, columns.chunks);
        add(
            Line::NodeVersions,
            self.versions
                .counted_bytes()
                .wrapping_add(columns.updates)
                .wrapping_add(self.label_updates.counted_bytes()),
        );
        add(
            Line::LabelBitmaps,
            self.label_bytes + label_map_bytes(&self.labels),
        );
        add(Line::Outgoing, self.outgoing.counted_bytes());
        add(Line::Incoming, self.incoming.counted_bytes());
    }

    /// What it holds, line by line, by a walk of its parts: the oracle of
    /// [`lines`](Self::lines), and what a row group dropped whole takes out
    /// of the account.
    fn walk_lines(&self) -> Lines {
        let mut lines = [0; LINES];
        lines[Line::NodeTables.index()] = table_entry_bytes::<Self>() + self.marks.heap_bytes();
        lines[Line::NodeColumns.index()] = self.columns.hot_bytes();
        lines[Line::NodeCold.index()] = self.columns.cold_meta_bytes();
        lines[Line::NodeColdChunks.index()] = self.columns.cold_chunk_bytes();
        lines[Line::NodeVersions.index()] = self.versions.heap_bytes()
            + self.columns.updates_bytes()
            + self.label_updates.heap_bytes(|ids| {
                if ids.spilled() {
                    ids.capacity() * std::mem::size_of::<u32>()
                } else {
                    0
                }
            });
        lines[Line::LabelBitmaps.index()] =
            self.labels.values().map(Bitset::heap_bytes).sum::<usize>()
                + label_map_bytes(&self.labels);
        lines[Line::Outgoing.index()] = self.outgoing.heap_bytes();
        lines[Line::Incoming.index()] = self.incoming.heap_bytes();
        lines
    }

    /// Sets the counts its parts keep to what they hold: after a load that
    /// built them without counting.
    fn retally(&mut self) {
        self.versions.retally();
        self.columns.retally();
        self.outgoing.retally();
        self.incoming.retally();
        self.label_bytes = self.labels.values().map(Bitset::heap_bytes).sum();
    }

    /// Drops the bitmap of label `label`, for a label no row has.
    fn forget_label(&mut self, label: u32) {
        if let Some(rows) = self.labels.remove(&label) {
            self.label_bytes = sub_count(self.label_bytes, rows.heap_bytes());
        }
    }

    /// Sets (`add`) or clears row `row` in the bitmap of label `label`.
    fn put_label(&mut self, label: u32, row: usize, add: bool) {
        let rows = self.labels.entry(label).or_default();
        let before = rows.heap_bytes();
        rows.put(row, add);
        self.label_bytes += rows.heap_bytes().saturating_sub(before);
    }
    fn version(&self, row: usize) -> RowVersion {
        self.versions.get(row)
    }

    /// Gives back the slots below the group's first vector with a row in any
    /// version, in each structure where they are due (see
    /// [`trims`](slots::trims)): no reader sees a row below it. Called by the
    /// collect, after the rows it collected are taken out.
    fn trim_head(&mut self) {
        // The lists of the rows the collect took out leave once they are
        // most of the lists, which also starts the offsets at the lowest
        // row left with a list.
        self.outgoing.merge_if_removed_dominate();
        self.incoming.merge_if_removed_dominate();
        let Some(head) = self.versions.first_vector_row() else {
            return;
        };
        self.columns.trim_dead_head(head);
        self.outgoing.trim_dead_head(head);
        self.incoming.trim_dead_head(head);
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

    /// The row's labels as `read` sees them as a bit mask (bit `i` for
    /// label id `i`), `None` for a label id of 64 or more. Reads the label
    /// bitmaps without listing the ids.
    fn label_mask_at(&self, row: usize, read: Read) -> Option<u64> {
        let bit = |id: u32| (id < 64).then(|| 1_u64 << id);
        if !self.label_updates.is_empty()
            && let Some(kept) = self.label_updates.saved(row, read)
        {
            return kept.iter().try_fold(0, |mask, &id| Some(mask | bit(id)?));
        }
        let mut mask = 0;
        for (&id, rows) in &self.labels {
            if rows.get(row) {
                mask |= bit(id)?;
            }
        }
        Some(mask)
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
        let mut grown = 0;
        for (id, rows) in &mut self.labels {
            let before = rows.heap_bytes();
            rows.put(row, ids.contains(id));
            grown += rows.heap_bytes() - before;
        }
        for id in ids {
            let rows = self.labels.entry(*id).or_default();
            let before = rows.heap_bytes();
            rows.set(row);
            grown += rows.heap_bytes() - before;
        }
        self.label_bytes += grown;
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
    /// The rows' ends and types, from the vector of the group's first row
    /// (see [`GroupRows`](slots::GroupRows)).
    src: slots::GroupRows<u64>,
    dst: slots::GroupRows<u64>,
    edge_type: slots::GroupRows<u32>,
    columns: Columns,
    /// The vectors the store lists, a bit each (`collect`).
    listed: u32,
    /// What changed since the store's baseline image (`checkpoint`).
    marks: Marks,
}

impl EdgeGroup {
    fn version(&self, row: usize) -> RowVersion {
        self.versions.get(row)
    }

    /// The row's source, target and type id; the row exists.
    fn ends(&self, row: usize) -> (u64, u64, u32) {
        (
            *self.src.at(row),
            *self.dst.at(row),
            *self.edge_type.at(row),
        )
    }

    /// The heap bytes of its rows' ends and types: the one figure of their
    /// size the walk and the account read.
    fn ends_bytes(&self) -> usize {
        self.src.heap_bytes() + self.dst.heap_bytes() + self.edge_type.heap_bytes()
    }

    /// What it holds, line by line, as the account counts it (see
    /// [`NodeGroup::lines`]).
    fn lines(&self) -> Lines {
        let mut lines = [0; LINES];
        self.add_lines_to(&mut lines);
        lines
    }

    /// Adds [`lines`](Self::lines) to `lines`, in place.
    #[inline]
    fn add_lines_to(&self, lines: &mut Lines) {
        let columns = self.columns.counts();
        let mut add = |line: Line, bytes: usize| {
            lines[line.index()] = lines[line.index()].wrapping_add(bytes);
        };
        add(
            Line::EdgeTables,
            table_entry_bytes::<Self>() + self.ends_bytes() + self.marks.heap_bytes(),
        );
        add(Line::EdgeColumns, columns.hot);
        add(Line::EdgeCold, columns.cold);
        add(Line::EdgeColdChunks, columns.chunks);
        add(
            Line::EdgeVersions,
            self.versions.counted_bytes().wrapping_add(columns.updates),
        );
    }

    /// What it holds, line by line, by a walk of its parts (see
    /// [`NodeGroup::walk_lines`]).
    fn walk_lines(&self) -> Lines {
        let mut lines = [0; LINES];
        lines[Line::EdgeTables.index()] =
            table_entry_bytes::<Self>() + self.ends_bytes() + self.marks.heap_bytes();
        lines[Line::EdgeColumns.index()] = self.columns.hot_bytes();
        lines[Line::EdgeCold.index()] = self.columns.cold_meta_bytes();
        lines[Line::EdgeColdChunks.index()] = self.columns.cold_chunk_bytes();
        lines[Line::EdgeVersions.index()] =
            self.versions.heap_bytes() + self.columns.updates_bytes();
        lines
    }

    /// Sets the counts its parts keep to what they hold (see
    /// [`NodeGroup::retally`]).
    fn retally(&mut self) {
        self.versions.retally();
        self.columns.retally();
    }

    fn set_ends(&mut self, row: usize, src: u64, dst: u64, edge_type: u32) {
        self.src.put(row, src, 0);
        self.dst.put(row, dst, 0);
        self.edge_type.put(row, edge_type, 0);
    }

    /// Gives back the slots below the group's first vector with a row in any
    /// version, in each structure where they are due (see
    /// [`trims`](slots::trims)): no reader sees a row below it. Called by the
    /// collect, after the rows it collected are taken out and unlinked.
    fn trim_head(&mut self) {
        let Some(head) = self.versions.first_vector_row() else {
            return;
        };
        self.src.trim_dead_head(head);
        self.dst.trim_dead_head(head);
        self.edge_type.trim_dead_head(head);
        self.columns.trim_dead_head(head);
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
    /// Moves label id `id`'s count by `delta`; returns the bytes its list
    /// grew by (counted by the caller: rare, at a new label id).
    fn label(&mut self, id: u32, delta: i64) -> usize {
        let index = id as usize;
        let before = self.labels.capacity();
        grow(&mut self.labels, index, 0);
        self.labels[index] += delta;
        (self.labels.capacity() - before) * std::mem::size_of::<i64>()
    }

    /// Moves edge type id `id`'s count by `delta`; returns the bytes its
    /// list grew by.
    fn edge_type(&mut self, id: u32, delta: i64) -> usize {
        let index = id as usize;
        let before = self.edge_types.capacity();
        grow(&mut self.edge_types, index, 0);
        self.edge_types[index] += delta;
        (self.edge_types.capacity() - before) * std::mem::size_of::<i64>()
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
    /// The image of a database file the store's chunks were last written to
    /// or read from, which the marks of its row groups are relative to;
    /// `None`: the next checkpoint writes every chunk (`checkpoint`).
    baseline: Option<grafeo_common::storage::ImageId>,
    /// What it holds, counted as it grows, and the grants that cover it
    /// (`account`).
    account: StoreAccount,
    /// The row groups the writes since the account last settled touched,
    /// with what they held before (`Bracket`).
    bracket: Bracket,
}

/// The row groups a bracket holds at most: an op touches three at most (an
/// edge's row group and its ends'), and the writes of one row or one
/// statement step keep to a few.
const BRACKET_GROUPS: usize = 4;

/// The ops a bracket takes before it closes: what an admission sees not yet
/// counted stays bounded (it keeps headroom for it, see `account`).
const BRACKET_OPS: u32 = 64;

/// What a debug build reads before a commit to check it after (see
/// `Inner::check_commit`).
#[cfg(debug_assertions)]
struct CommitCheck {
    groups: Vec<(bool, u64)>,
    before: Lines,
    counted: Lines,
}

/// One reading of the account around a run of writes: the row groups the
/// writes touched, each taken in at its first touch with what it held, and
/// what the indexes held when the run began (the other shared parts count
/// their growth where it happens). A commit, a rollback, a write stamped at
/// once, a full bracket or any other change of the store closes it: the
/// account adds what the parts hold then less what they held, once for the
/// run. So a transaction's writes cost the account a membership test each,
/// and its commit one reading of what they touched.
#[derive(Debug, Default)]
struct Bracket {
    /// Whether a run is open.
    open: bool,
    /// The row groups taken in, by table and index; the first `len` hold.
    groups: [Option<(Table, u64)>; BRACKET_GROUPS],
    len: usize,
    /// The ops of the run.
    ops: u32,
    /// What the indexes and the groups taken in held at their first touch.
    before: Lines,
    /// What they hold when the bracket closes.
    after: Lines,
}

impl Bracket {
    /// Whether it holds row group `index` of `table`.
    fn holds(&self, table: Table, index: u64) -> bool {
        self.groups[..self.len].contains(&Some((table, index)))
    }
}

/// The row groups an op may change, each once.
#[derive(Debug, Clone, Copy)]
enum Touched {
    /// Up to two node row groups (an edge's ends) and an edge row group.
    Groups {
        nodes: [Option<u64>; 2],
        edge: Option<u64>,
    },
    /// The row groups `first..=last` of a table (a bulk range).
    Range { table: Table, first: u64, last: u64 },
}

impl Touched {
    /// The row group of the node or edge `op` writes, and an edge's ends'.
    fn of(op: &DataOp) -> Self {
        let group = |id: u64| locate(id).0;
        let (nodes, edge) = match op {
            DataOp::CreateNode { id, .. }
            | DataOp::DeleteNode { id }
            | DataOp::SetNodeProperty { id, .. }
            | DataOp::RemoveNodeProperty { id, .. }
            | DataOp::AddNodeLabel { id, .. }
            | DataOp::RemoveNodeLabel { id, .. } => ([Some(group(id.as_u64())), None], None),
            DataOp::CreateEdge { id, src, dst, .. } => {
                let (src, dst) = (group(src.as_u64()), group(dst.as_u64()));
                (
                    [Some(src), (dst != src).then_some(dst)],
                    Some(group(id.as_u64())),
                )
            }
            DataOp::DeleteEdge { id }
            | DataOp::SetEdgeProperty { id, .. }
            | DataOp::RemoveEdgeProperty { id, .. } => ([None, None], Some(group(id.as_u64()))),
            DataOp::InsertTriple { .. } | DataOp::DeleteTriple { .. } => ([None, None], None),
        };
        Self::Groups { nodes, edge }
    }

    /// No row group: an op on the shared parts alone (an index, a
    /// dictionary).
    const fn none() -> Self {
        Self::Groups {
            nodes: [None, None],
            edge: None,
        }
    }

    /// The row groups a bulk range of `table` over `ids` spans.
    fn range(table: Table, ids: &std::ops::Range<u64>) -> Self {
        if ids.is_empty() {
            return Self::none();
        }
        Self::Range {
            table,
            first: locate(ids.start).0,
            last: locate(ids.end - 1).0,
        }
    }
}

impl Inner {
    /// What the store holds outside its row groups, line by line, as the
    /// account counts it: its dictionaries and counts, its lists for the
    /// collect and its indexes.
    fn shared_lines(&self) -> Lines {
        let mut lines = [0; LINES];
        let (listed_nodes, listed_edges) = self.listed.heap_bytes();
        lines[Line::NodeVersions.index()] = listed_nodes;
        lines[Line::EdgeVersions.index()] = listed_edges;
        lines[Line::Labels.index()] = self.labels.counted_bytes()
            + self.counts.labels.capacity() * std::mem::size_of::<i64>();
        lines[Line::EdgeTypes.index()] = self.edge_types.counted_bytes()
            + self.counts.edge_types.capacity() * std::mem::size_of::<i64>();
        lines[Line::Keys.index()] = self.keys.counted_bytes();
        if !self.indexes.is_empty() {
            let [property, text, vector] = self.indexes.counted_bytes();
            lines[Line::PropertyIndexes.index()] = property;
            lines[Line::TextIndexes.index()] = text;
            lines[Line::VectorIndexes.index()] = vector;
        }
        lines
    }

    /// [`shared_lines`](Self::shared_lines) by a walk of every part.
    fn shared_walk_lines(&self) -> Lines {
        let mut lines = [0; LINES];
        let (listed_nodes, listed_edges) = self.listed.heap_bytes();
        lines[Line::NodeVersions.index()] = listed_nodes;
        lines[Line::EdgeVersions.index()] = listed_edges;
        lines[Line::Labels.index()] =
            self.labels.heap_bytes() + self.counts.labels.capacity() * std::mem::size_of::<i64>();
        lines[Line::EdgeTypes.index()] = self.edge_types.heap_bytes()
            + self.counts.edge_types.capacity() * std::mem::size_of::<i64>();
        lines[Line::Keys.index()] = self.keys.heap_bytes();
        let [property, text, vector] = self.indexes.walk_bytes();
        lines[Line::PropertyIndexes.index()] = property;
        lines[Line::TextIndexes.index()] = text;
        lines[Line::VectorIndexes.index()] = vector;
        lines
    }

    /// Everything the store holds, line by line, by a walk of every part:
    /// the account's oracle, and what a load counts.
    fn walk_lines(&self) -> Lines {
        let mut lines = self.shared_walk_lines();
        for group in self.nodes.values() {
            add_lines(&mut lines, &group.walk_lines());
        }
        for group in self.edges.values() {
            add_lines(&mut lines, &group.walk_lines());
        }
        lines
    }

    /// What `touched` and the store's shared parts hold, line by line, as
    /// the account counts it. Read before and after an op, the difference
    /// is what the op changed.
    fn lines_of(&self, touched: Touched) -> Lines {
        let mut lines = self.shared_lines();
        match touched {
            Touched::Groups { nodes, edge } => {
                for index in nodes.into_iter().flatten() {
                    if let Some(group) = self.nodes.get(&index) {
                        group.add_lines_to(&mut lines);
                    }
                }
                if let Some(group) = edge.and_then(|index| self.edges.get(&index)) {
                    group.add_lines_to(&mut lines);
                }
            }
            Touched::Range {
                table: Table::Nodes,
                first,
                last,
            } => {
                for group in self.nodes.range(first..=last).map(|(_, group)| group) {
                    group.add_lines_to(&mut lines);
                }
            }
            Touched::Range {
                table: Table::Edges,
                first,
                last,
            } => {
                for group in self.edges.range(first..=last).map(|(_, group)| group) {
                    group.add_lines_to(&mut lines);
                }
            }
        }
        lines
    }

    /// Takes the row groups of `touched` into the bracket (see `Bracket`),
    /// opening it first: what an op does to them is counted when the bracket
    /// closes. A run of writes to the groups it holds costs a test of each
    /// against the few it holds; a bracket that cannot take them, or that
    /// took its ops, closes first.
    #[inline]
    fn touch(&mut self, touched: Touched) {
        if let Touched::Groups { nodes, edge } = touched
            && self.bracket.open
            && self.bracket.ops < BRACKET_OPS
            && nodes[0].is_none_or(|index| self.bracket.holds(Table::Nodes, index))
            && nodes[1].is_none_or(|index| self.bracket.holds(Table::Nodes, index))
            && edge.is_none_or(|index| self.bracket.holds(Table::Edges, index))
        {
            self.bracket.ops += 1;
            return;
        }
        self.touch_new(touched);
    }

    /// [`touch`](Self::touch) of row groups the bracket does not hold yet, or
    /// of a closed or full bracket.
    fn touch_new(&mut self, touched: Touched) {
        let Touched::Groups { nodes, edge } = touched else {
            // A bulk range: counted on its own.
            self.close_bracket();
            return;
        };
        let wanted = [
            nodes[0].map(|index| (Table::Nodes, index)),
            nodes[1].map(|index| (Table::Nodes, index)),
            edge.map(|index| (Table::Edges, index)),
        ];
        let new = wanted
            .iter()
            .flatten()
            .filter(|(table, index)| !self.bracket.holds(*table, *index))
            .count();
        if self.bracket.ops >= BRACKET_OPS || self.bracket.len + new > BRACKET_GROUPS {
            self.close_bracket();
        }
        if !self.bracket.open {
            self.bracket.open = true;
            self.add_bracket_shared(true);
        }
        self.bracket.ops += 1;
        for (table, index) in wanted.into_iter().flatten() {
            if self.bracket.holds(table, index) {
                continue;
            }
            match table {
                Table::Nodes => {
                    if let Some(group) = self.nodes.get(&index) {
                        group.add_lines_to(&mut self.bracket.before);
                    }
                }
                Table::Edges => {
                    if let Some(group) = self.edges.get(&index) {
                        group.add_lines_to(&mut self.bracket.before);
                    }
                }
            }
            self.bracket.groups[self.bracket.len] = Some((table, index));
            self.bracket.len += 1;
        }
    }

    /// Adds to the bracket's reading (`before`) or to `self.bracket.after`
    /// what the shared parts a write may change behind the bracket's back
    /// hold: the indexes, when the store has any (writes keep them through
    /// shared references). The other shared parts (dictionaries, counts, the
    /// collect's lists) count their growth where it happens, so the bracket
    /// does not read them.
    #[inline]
    fn add_bracket_shared(&mut self, before: bool) {
        if self.indexes.is_empty() {
            return;
        }
        let [property, text, vector] = self.indexes.counted_bytes();
        let lines = if before {
            &mut self.bracket.before
        } else {
            &mut self.bracket.after
        };
        for (line, bytes) in [
            (Line::PropertyIndexes, property),
            (Line::TextIndexes, text),
            (Line::VectorIndexes, vector),
        ] {
            lines[line.index()] = lines[line.index()].wrapping_add(bytes);
        }
    }

    /// Closes the bracket (see `Bracket`): the account adds what the groups
    /// it took in (and the indexes) hold now, less what they held, and
    /// settles.
    fn close_bracket(&mut self) {
        if !self.bracket.open {
            return;
        }
        self.bracket.after = [0; LINES];
        self.add_bracket_shared(false);
        for &(table, index) in self.bracket.groups[..self.bracket.len].iter().flatten() {
            match table {
                Table::Nodes => {
                    if let Some(group) = self.nodes.get(&index) {
                        group.add_lines_to(&mut self.bracket.after);
                    }
                }
                Table::Edges => {
                    if let Some(group) = self.edges.get(&index) {
                        group.add_lines_to(&mut self.bracket.after);
                    }
                }
            }
        }
        let (before, after) = (self.bracket.before, self.bracket.after);
        self.account.moved(&before, &after);
        self.bracket = Bracket::default();
        self.account.settle();
    }

    /// Counts what `change` does to the parts `touched` names and to the
    /// shared parts, after closing the bracket.
    fn counting<R>(&mut self, touched: Touched, change: impl FnOnce(&mut Self) -> R) -> R {
        self.close_bracket();
        let before = self.lines_of(touched);
        let result = change(self);
        let after = self.lines_of(touched);
        self.account.moved(&before, &after);
        result
    }

    /// Counts everything the store holds again, by a walk: after a load, or
    /// a change of the whole store. The counts the parts keep start from
    /// what they hold.
    fn recount(&mut self) {
        // A walk counts every write: the open bracket goes with its reading.
        self.bracket = Bracket::default();
        for group in self.nodes.values_mut() {
            group.retally();
        }
        for group in self.edges.values_mut() {
            group.retally();
        }
        self.note_key_columns();
        let lines = self.walk_lines();
        self.account.reset(lines);
    }

    /// Notes the property keys each table has a column of, from its row
    /// groups: after a load, or after a collect dropped a row group.
    fn note_key_columns(&mut self) {
        let node_keys: Vec<u32> = self
            .nodes
            .values()
            .flat_map(|group| group.columns.keys())
            .collect();
        let edge_keys: Vec<u32> = self
            .edges
            .values()
            .flat_map(|group| group.columns.keys())
            .collect();
        self.account.set_key_columns(Table::Nodes, node_keys);
        self.account.set_key_columns(Table::Edges, edge_keys);
    }

    /// About what `op` thaws: the decoded bytes of the cold columns its
    /// values go to. Nothing to look up while the store holds no cold
    /// column.
    fn thaw_bytes(&self, op: &DataOp) -> usize {
        if !self.account.holds_cold() {
            return 0;
        }
        let of = |group: Option<&Columns>, key: &str| {
            group
                .zip(self.keys.get_id(key))
                .map_or(0, |(columns, key)| columns.thaw_bytes(key))
        };
        let nodes = |id: NodeId| {
            self.nodes
                .get(&locate(id.as_u64()).0)
                .map(|group| &group.columns)
        };
        let edges = |id: EdgeId| {
            self.edges
                .get(&locate(id.as_u64()).0)
                .map(|group| &group.columns)
        };
        match op {
            DataOp::CreateNode { id, properties, .. } => properties
                .iter()
                .map(|(key, _)| of(nodes(*id), key.as_str()))
                .sum(),
            DataOp::CreateEdge { id, properties, .. } => properties
                .iter()
                .map(|(key, _)| of(edges(*id), key.as_str()))
                .sum(),
            DataOp::SetNodeProperty { id, key, .. } | DataOp::RemoveNodeProperty { id, key } => {
                of(nodes(*id), key.as_str())
            }
            DataOp::SetEdgeProperty { id, key, .. } | DataOp::RemoveEdgeProperty { id, key } => {
                of(edges(*id), key.as_str())
            }
            _ => 0,
        }
    }

    /// What a commit of `entries` touches, read before it, for
    /// [`check_commit`](Self::check_commit): its row groups (each once), what
    /// they and the shared parts hold, and what the account counts.
    #[cfg(debug_assertions)]
    fn commit_check(&self, entries: &[&grafeo_common::change::Change]) -> CommitCheck {
        use grafeo_common::change::Change;

        let mut groups: Vec<(bool, u64)> = Vec::new();
        for change in entries {
            match change {
                Change::Data { op, .. } => {
                    if let Touched::Groups { nodes, edge } = Touched::of(op) {
                        groups.extend(nodes.into_iter().flatten().map(|index| (true, index)));
                        groups.extend(edge.map(|index| (false, index)));
                    }
                }
                Change::Bulk(range) => {
                    if let Touched::Range { table, first, last } =
                        Touched::range(range.table, &range.ids)
                    {
                        let nodes = table == Table::Nodes;
                        groups.extend((first..=last).map(|index| (nodes, index)));
                    }
                }
            }
        }
        groups.sort_unstable();
        groups.dedup();
        let before = self.commit_lines(&groups);
        CommitCheck {
            groups,
            before,
            counted: self.account.counted_lines(),
        }
    }

    /// What the row groups of a commit and the shared parts hold.
    #[cfg(debug_assertions)]
    fn commit_lines(&self, groups: &[(bool, u64)]) -> Lines {
        let mut lines = self.shared_lines();
        for &(nodes, index) in groups {
            if nodes {
                if let Some(group) = self.nodes.get(&index) {
                    group.add_lines_to(&mut lines);
                }
            } else if let Some(group) = self.edges.get(&index) {
                group.add_lines_to(&mut lines);
            }
        }
        lines
    }

    /// Checks that a commit counted exactly what a bracket around it would
    /// have: what its row groups and the shared parts moved by equals what
    /// the account moved by.
    #[cfg(debug_assertions)]
    fn check_commit(&self, check: &CommitCheck) {
        let after = self.commit_lines(&check.groups);
        let counted = self.account.counted_lines();
        for index in 0..LINES {
            let held = after[index].wrapping_sub(check.before[index]);
            let moved = counted[index].wrapping_sub(check.counted[index]);
            assert_eq!(
                held,
                moved,
                "a commit moved {:?} by {held} bytes and counted {moved}",
                account::line_at(index)
            );
        }
    }

    /// Settles the account after a change: its grants cover what it counts.
    /// Closes the bracket first.
    fn settle(&mut self) {
        self.close_bracket();
        self.account.settle();
    }

    /// Settles the account without hysteresis, after a spill or a
    /// compression: what it freed leaves the buffer manager's count.
    fn settle_tight(&mut self) {
        self.close_bracket();
        self.account.settle_tight();
    }

    /// Whether `op` may grow the index buffers: an edge's create (the
    /// adjacency), or a node's write while the store has an index.
    fn may_grow_indexes(&self, op: &DataOp) -> bool {
        match op {
            DataOp::CreateEdge { .. } => true,
            DataOp::CreateNode { .. }
            | DataOp::SetNodeProperty { .. }
            | DataOp::RemoveNodeProperty { .. }
            | DataOp::AddNodeLabel { .. }
            | DataOp::RemoveNodeLabel { .. } => !self.indexes.is_empty(),
            DataOp::DeleteNode { .. }
            | DataOp::DeleteEdge { .. }
            | DataOp::SetEdgeProperty { .. }
            | DataOp::RemoveEdgeProperty { .. }
            | DataOp::InsertTriple { .. }
            | DataOp::DeleteTriple { .. } => false,
        }
    }
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
    #[inline]
    fn node_seen(&self, id: u64, read: Read) -> Option<(&NodeGroup, usize)> {
        let (group, row) = locate(id);
        let group = self.nodes.get(&group)?;
        group.versions.visible(row, read).then_some((group, row))
    }

    /// The edge's group and row, if `read` sees the edge (see
    /// [`node_seen`](Self::node_seen)).
    #[inline]
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
                    .map(|name| (PropertyKey::new(name.clone()), value))
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
            properties: PropertyMap::from_distinct(self.values_at(&group.columns, row, read)),
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
            properties: PropertyMap::from_distinct(self.values_at(&group.columns, row, read)),
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
            let first = group.dst.first_row();
            group
                .dst
                .as_slice()
                .iter()
                .enumerate()
                .map(move |(offset, dst)| (first + offset, dst))
                .filter(move |(row, dst)| **dst == node && group.versions.exists(*row))
                .map(move |(row, _)| Adjacent {
                    edge_type: *group.edge_type.at(row),
                    other: *group.src.at(row),
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
    /// The degrees it has sampled for the planner (`degree_samples`).
    degree_samples: degree_samples::DegreeSamples,
    /// The buffer manager its grants come from, once a database attaches
    /// one (`account`).
    memory: account::Attached,
    /// The memory consumers a database registered for it, unregistered when
    /// it drops.
    registered: account::Registered,
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

    /// Takes the memory it holds from `manager` from now on: the grants of
    /// its account cover what it counts, and a write past the memory limit
    /// is refused (once admission refuses). A database attaches its buffer
    /// manager once, after building it; each named graph gets the root's.
    /// A store keeps the first manager it gets.
    pub fn attach_memory(&self, manager: Arc<BufferManager>) {
        if !self.memory.set(Arc::clone(&manager)) {
            return;
        }
        self.inner.write().account.attach(Arc::clone(&manager));
        self.degree_samples.attach(&manager);
        for name in self.graph_names() {
            if let Some(graph) = self.graph(&name) {
                graph.attach_memory(Arc::clone(&manager));
            }
        }
    }

    /// Unregisters the memory consumer `name` from `manager` when the store
    /// drops: a consumer that reads the store holds it weakly, and leaves
    /// with it.
    pub fn unregister_on_drop(&self, manager: &Arc<BufferManager>, name: &'static str) {
        self.registered.add(manager, name);
    }

    /// Runs `f` with this store's reads' thaws, and its named graphs',
    /// granted whatever the memory limit: for the work an open must finish
    /// (the index and text rebuilds), which never refuses (O1). Outside it,
    /// a read whose thaw the limit has no room for decodes the one chunk it
    /// needs instead.
    pub fn with_unrefused<T>(&self, f: impl FnOnce() -> T) -> T {
        let others: Vec<_> = self
            .graph_names()
            .iter()
            .filter_map(|name| self.graph(name))
            .map(|graph| graph.inner.read().account.thaws())
            .collect();
        let _unrefused = self.inner.read().account.unrefused(others);
        f()
    }

    /// The nodes it counts at its latest epoch, without a scan: the figure
    /// of an index build's estimate.
    #[must_use]
    pub fn nodes_counted(&self) -> usize {
        usize::try_from(self.inner.read().counts.nodes.max(0)).unwrap_or(usize::MAX)
    }

    /// The buffer manager it takes its memory from, once attached.
    #[must_use]
    pub fn memory_manager(&self) -> Option<&Arc<BufferManager>> {
        self.memory.get()
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
