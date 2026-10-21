//! What a checkpoint needs of the row-group store to write only what changed
//! (H3, #430), and to write without holding writers off.
//!
//! Every row group carries marks of what changed in it since the image its
//! store's chunks were last written to or read from (the store's baseline):
//! rows created or deleted, the property keys and labels whose values or
//! rows changed (each with the vectors of 2,048 rows it changed in), and the
//! nodes' outgoing and incoming adjacency. A write
//! sets them as it takes effect for readers: a write stamped at once
//! (replay, an immediate write) when it is applied, a transaction's at its
//! commit, when `stamp` gives it its epoch. A rollback undoes what nothing
//! but its transaction saw, so it marks nothing.
//!
//! A checkpoint takes the marks while commits are held off, with the epoch
//! E its image is written at ([`CheckpointSnapshot::take`]): every commit
//! at or before E marked what it changed before the snapshot, every later
//! one marks the store's new marks. With commits going on again, the
//! section writer reads each graph at E, gathering a file row group's values
//! at a time under the store's read lock and encoding and writing them with
//! the lock released (the engine pins a read at E, so no collect takes a
//! version it reads), and per file row group of the LPG section:
//!
//! - writes nothing new for a group without marks: the new image points at
//!   the chunks of the image it follows ([`SectionSink::reuse_chunk`]);
//! - re-encodes only the marked columns of a group whose rows did not
//!   change (no create or delete), within the chunk boundaries of that
//!   image and only for the chunks whose rows the marks name, and its
//!   adjacency when it changed;
//! - writes every chunk of any other group, as a full checkpoint does.
//!
//! A row group checks the image's row count of it besides its marks: a group
//! without history whose count changed is written anew whatever the marks
//! say.
//!
//! A full checkpoint is the case where every group is marked, or where the
//! store's baseline is not the image the checkpoint follows (a new file, a
//! copy, a store loaded or written elsewhere, or a checkpoint that failed):
//! the same path, with nothing to reuse. Once the image is active the
//! engine makes it every written graph's baseline
//! ([`CheckpointSnapshot::written_as`]); after a failure no graph has one
//! ([`CheckpointSnapshot::failed`]), so the next checkpoint writes every
//! chunk.
//!
//! [`SectionSink::reuse_chunk`]: grafeo_common::storage::SectionSink::reuse_chunk

use std::collections::BTreeMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use grafeo_common::change::{BulkRange, DataOp, Table};
use grafeo_common::storage::ImageId;
use grafeo_common::types::EpochId;

use super::versions::VECTOR_ROWS;
use super::{Inner, ROWS_PER_GROUP, RowGroupStore, locate};

/// The columns of a row group (property keys, or labels) whose values or
/// rows changed, each with the vectors of 2,048 rows it changed in, a bit
/// each: by column id (the dictionaries give out small ids, densely), so a
/// write marks with one index; empty until the first mark.
#[derive(Debug, Default, Clone)]
pub(super) struct ColumnMarks(Vec<u32>);

impl ColumnMarks {
    /// Marks column `id` changed at row `row` of the group.
    #[inline]
    pub(super) fn mark(&mut self, id: u32, row: usize) {
        let index = id as usize;
        if index >= self.0.len() {
            self.0.resize(index + 1, 0);
        }
        self.0[index] |= 1 << (row / VECTOR_ROWS);
    }

    /// Whether no column changed.
    pub(super) fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    /// The columns that changed, ascending.
    pub(super) fn ids(&self) -> impl Iterator<Item = u32> + '_ {
        self.0
            .iter()
            .enumerate()
            .filter(|(_, mask)| **mask != 0)
            .map(|(id, _)| u32::try_from(id).expect("a dictionary id"))
    }

    /// Whether column `id` changed in one of `rows`, rows of the group.
    pub(super) fn touches(&self, id: u32, rows: std::ops::Range<usize>) -> bool {
        let Some(mask) = self.0.get(id as usize) else {
            return false;
        };
        if rows.is_empty() {
            return false;
        }
        (rows.start / VECTOR_ROWS..=(rows.end - 1) / VECTOR_ROWS)
            .any(|vector| mask & (1 << vector) != 0)
    }
}

/// What changed in a row group since its store's baseline image.
#[derive(Debug, Default, Clone)]
pub(super) struct Marks {
    /// Rows were created or deleted: every chunk of the group changes.
    pub(super) rows: bool,
    /// The nodes' outgoing adjacency changed.
    pub(super) outgoing: bool,
    /// The nodes' incoming adjacency changed.
    pub(super) incoming: bool,
    /// The property keys whose values changed, by key id.
    pub(super) keys: ColumnMarks,
    /// The labels whose rows changed, by label id (node groups only).
    pub(super) labels: ColumnMarks,
}

impl Marks {
    /// Marks the nodes' outgoing (`outgoing`) or incoming adjacency changed.
    pub(super) fn mark_adjacency(&mut self, outgoing: bool) {
        if outgoing {
            self.outgoing = true;
        } else {
            self.incoming = true;
        }
    }

    /// Whether nothing changed.
    pub(super) fn is_empty(&self) -> bool {
        !self.rows
            && !self.outgoing
            && !self.incoming
            && self.keys.is_empty()
            && self.labels.is_empty()
    }
}

/// What a checkpoint took of one graph while commits were held: the marks
/// of its row groups, the image they are relative to, and the next ids and
/// dictionary sizes at the checkpoint's epoch.
#[derive(Debug, Default)]
pub(super) struct GraphMarks {
    /// The image the graph's chunks were last written to or read from.
    pub(super) baseline: Option<ImageId>,
    /// The marks of the node row groups with some, by row group.
    pub(super) nodes: BTreeMap<u64, Marks>,
    /// The marks of the edge row groups with some.
    pub(super) edges: BTreeMap<u64, Marks>,
    pub(super) next_node_id: u64,
    pub(super) next_edge_id: u64,
    /// The next ids of the label, edge type and property key dictionaries.
    pub(super) next_names: [u32; 3],
}

/// A graph of a [`CheckpointSnapshot`].
#[derive(Debug)]
pub(super) struct SnapshotGraph {
    pub(super) id: u32,
    pub(super) name: String,
    pub(super) store: Arc<RowGroupStore>,
    pub(super) marks: GraphMarks,
}

/// What a checkpoint takes of a root and its named graphs while it holds
/// commits off, to write the LPG section at its epoch afterwards, with
/// commits going on (see the module docs): the graphs it writes, each with
/// the marks of what changed since its baseline image and the next ids at
/// the epoch.
///
/// The marks it took are the only record of what changed since each
/// graph's baseline: a snapshot dropped before
/// [`written_as`](Self::written_as), after an error or a panic, does what
/// [`failed`](Self::failed) does, so the next checkpoint writes every chunk.
#[derive(Debug)]
pub struct CheckpointSnapshot {
    pub(super) epoch: EpochId,
    pub(super) history_from: Option<EpochId>,
    pub(super) next_graph_id: u32,
    /// The graphs, ids ascending: the default graph first.
    pub(super) graphs: Vec<SnapshotGraph>,
    /// Whether the image holding it is written and active.
    written: AtomicBool,
}

impl CheckpointSnapshot {
    /// Takes the snapshot of `root` and its named graphs at `epoch`, with
    /// the history from `history_from` (an epoch before `epoch`) for a
    /// database that keeps history: each graph's marks, which its store
    /// starts over, and the next ids and dictionary sizes now.
    ///
    /// The caller holds commits off (no commit is between its epoch and its
    /// completion, none at or before `epoch` is still to be stamped) and,
    /// until the section is written, a read pinned at `history_from` or, without
    /// history, at `epoch`: the section reads the graphs at `epoch`.
    #[must_use]
    pub fn take(root: &Arc<RowGroupStore>, epoch: EpochId, history_from: Option<EpochId>) -> Self {
        let mut named: Vec<(String, Arc<RowGroupStore>)> = root
            .graphs
            .named
            .read()
            .iter()
            .map(|(name, graph)| (name.clone(), Arc::clone(graph)))
            .collect();
        named.sort_unstable_by_key(|(_, graph)| graph.graph_id());
        let mut graphs = Vec::with_capacity(named.len() + 1);
        graphs.push(SnapshotGraph {
            id: 0,
            name: String::new(),
            store: Arc::clone(root),
            marks: root.take_checkpoint_marks(),
        });
        for (name, store) in named {
            graphs.push(SnapshotGraph {
                id: store.graph_id(),
                name,
                marks: store.take_checkpoint_marks(),
                store,
            });
        }
        Self {
            epoch,
            history_from: history_from.filter(|from| *from < epoch),
            // Read after the graphs: above every graph id taken.
            next_graph_id: root.next_graph_id(),
            graphs,
            written: AtomicBool::new(false),
        }
    }

    /// The epoch the section is written at.
    #[must_use]
    pub fn epoch(&self) -> EpochId {
        self.epoch
    }

    /// The image holding the section is written and active as `image`:
    /// every graph's chunks are from now on those of `image`, and the marks
    /// its writes set since the snapshot are what changed since.
    pub fn written_as(&self, image: ImageId) {
        self.written.store(true, Ordering::Release);
        for graph in &self.graphs {
            graph.store.inner.write().baseline = Some(image);
        }
    }

    /// The checkpoint failed: the marks it took are gone, so no graph knows
    /// what changed since its baseline any more, and the next checkpoint
    /// writes every chunk.
    pub fn failed(&self) {
        for graph in &self.graphs {
            graph.store.inner.write().baseline = None;
        }
    }
}

impl Drop for CheckpointSnapshot {
    fn drop(&mut self) {
        if !self.written.load(Ordering::Acquire) {
            self.failed();
        }
    }
}

impl RowGroupStore {
    /// Takes the marks of every row group, which start over, with the
    /// baseline they are relative to and the next ids and dictionary sizes
    /// now (see [`CheckpointSnapshot::take`]).
    fn take_checkpoint_marks(&self) -> GraphMarks {
        let mut inner = self.inner.write();
        let mut marks = GraphMarks {
            baseline: inner.baseline,
            next_node_id: self.next_node_id.load(Ordering::Acquire),
            next_edge_id: self.next_edge_id.load(Ordering::Acquire),
            next_names: [
                inner.labels.next_id(),
                inner.edge_types.next_id(),
                inner.keys.next_id(),
            ],
            ..GraphMarks::default()
        };
        for (index, group) in &mut inner.nodes {
            if !group.marks.is_empty() {
                marks.nodes.insert(*index, std::mem::take(&mut group.marks));
            }
        }
        for (index, group) in &mut inner.edges {
            if !group.marks.is_empty() {
                marks.edges.insert(*index, std::mem::take(&mut group.marks));
            }
        }
        marks
    }

    /// Makes `image` the image this store's chunks are in, as a load from
    /// it leaves them: the next checkpoint into the file whose active image
    /// it is writes only what changed since. A store loaded from anything
    /// else (a migration, a copy, a snapshot) has none, and its next
    /// checkpoint writes every chunk.
    pub fn set_checkpoint_baseline(&self, image: ImageId) {
        self.inner.write().baseline = Some(image);
    }

    /// The image this store's chunks are in, `None` when the next
    /// checkpoint writes every chunk.
    #[must_use]
    pub fn checkpoint_baseline(&self) -> Option<ImageId> {
        self.inner.read().baseline
    }
}

impl Inner {
    /// Marks node row group of `id` with `mark`, when the group exists.
    fn mark_node(&mut self, id: u64, mark: impl FnOnce(&mut Marks)) {
        if let Some(group) = self.nodes.get_mut(&locate(id).0) {
            mark(&mut group.marks);
        }
    }

    /// Marks edge row group of `id` with `mark`, when the group exists.
    fn mark_edge(&mut self, id: u64, mark: impl FnOnce(&mut Marks)) {
        if let Some(group) = self.edges.get_mut(&locate(id).0) {
            mark(&mut group.marks);
        }
    }

    /// Marks an edge's row group and its ends' adjacency.
    fn mark_edge_and_ends(&mut self, id: u64, ends: Option<(u64, u64)>) {
        self.mark_edge(id, |marks| marks.rows = true);
        if let Some((src, dst)) = ends {
            self.mark_node(src, |marks| marks.outgoing = true);
            self.mark_node(dst, |marks| marks.incoming = true);
        }
    }

    /// Marks what `op` changes, once it takes effect for readers.
    pub(super) fn mark_op(&mut self, op: &DataOp) {
        match op {
            DataOp::CreateNode { id, .. } | DataOp::DeleteNode { id } => {
                self.mark_node(id.as_u64(), |marks| marks.rows = true);
            }
            DataOp::CreateEdge { id, src, dst, .. } => {
                self.mark_edge_and_ends(id.as_u64(), Some((src.as_u64(), dst.as_u64())));
            }
            DataOp::DeleteEdge { id } => {
                let ends = self.edge(id.as_u64()).map(|(group, row)| {
                    let (src, dst, _) = group.ends(row);
                    (src, dst)
                });
                self.mark_edge_and_ends(id.as_u64(), ends);
            }
            DataOp::SetNodeProperty { id, key, .. } | DataOp::RemoveNodeProperty { id, key } => {
                if let Some(key) = self.keys.get_id(key.as_str()) {
                    let row = locate(id.as_u64()).1;
                    self.mark_node(id.as_u64(), |marks| marks.keys.mark(key, row));
                }
            }
            DataOp::SetEdgeProperty { id, key, .. } | DataOp::RemoveEdgeProperty { id, key } => {
                if let Some(key) = self.keys.get_id(key.as_str()) {
                    let row = locate(id.as_u64()).1;
                    self.mark_edge(id.as_u64(), |marks| marks.keys.mark(key, row));
                }
            }
            DataOp::AddNodeLabel { id, label } | DataOp::RemoveNodeLabel { id, label } => {
                if let Some(label) = self.label_id(label) {
                    let row = locate(id.as_u64()).1;
                    self.mark_node(id.as_u64(), |marks| marks.labels.mark(label, row));
                }
            }
            DataOp::InsertTriple { .. } | DataOp::DeleteTriple { .. } => {}
        }
    }

    /// Marks the rows of a bulk range, and for edges their ends' adjacency.
    pub(super) fn mark_bulk(&mut self, range: &BulkRange) {
        if range.ids.is_empty() {
            return;
        }
        let first = locate(range.ids.start).0;
        let last = locate(range.ids.end - 1).0;
        match range.table {
            Table::Nodes => {
                for (_, group) in self.nodes.range_mut(first..=last) {
                    group.marks.rows = true;
                }
            }
            Table::Edges => {
                let mut ends: Vec<(u64, u64)> = Vec::new();
                for (index, group) in self.edges.range_mut(first..=last) {
                    group.marks.rows = true;
                    let base = index * ROWS_PER_GROUP;
                    for (row, _) in group.versions.rows() {
                        let id = base + row as u64;
                        if range.ids.contains(&id) {
                            let (src, dst, _) = group.ends(row);
                            ends.push((src, dst));
                        }
                    }
                }
                for (src, dst) in ends {
                    self.mark_node(src, |marks| marks.outgoing = true);
                    self.mark_node(dst, |marks| marks.incoming = true);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests;
