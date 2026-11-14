//! What committed RDF writes changed, for readers at an earlier snapshot
//! (#718).
//!
//! The store holds the latest committed triples only. Each commit also
//! publishes an [`OverlayEntry`] at its epoch, in the same hold of the
//! dataset's apply guard as its apply: the triples it inserted and deleted,
//! and the graph operations it ran, in recorded order, leaving out the ones
//! that changed nothing. A reader at snapshot `S` reads the store and undoes what
//! the entries after `S` did ([`Snapshot`]): a triple's state at `S` is the
//! one before its first change after `S`, present when that change was a
//! delete and absent when it was an insert. A graph that a `DROP` or `CLEAR`
//! after `S` swapped out is read from the partition the entry keeps.
//!
//! Entries are kept while a reader pins an epoch below theirs
//! ([`RdfOverlay::prune_below`]); the overlay keeps nothing for history.

use std::collections::VecDeque;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};

use grafeo_common::types::EpochId;
use grafeo_common::utils::hash::{FxHashMap, FxHashSet};
use parking_lot::{RecursiveRwLock, RwLock};

use super::{RdfStore, Triple};

/// A triple by its three dataset term ids: the key the store's indexes use,
/// exact for equal terms.
pub type TripleIds = [u64; 3];

/// A graph by its key: `None` for the default graph.
pub type GraphKey = Option<Arc<str>>;

/// One change a commit made, in the order it made them.
///
/// A triple change names no graph: it is in the graph of the last
/// [`In`](Self::In) before it, or the default graph when there is none, so
/// a run of changes in one graph names it once.
#[derive(Debug, Clone)]
pub enum OverlayChange {
    /// The triple changes that follow, up to the next `In`, are in `graph`.
    In(GraphKey),
    /// A run of triples one apply inserted, each new to the graph and
    /// carrying the ids the store keys it by.
    Inserts(Vec<Arc<Triple>>),
    /// `triple` was deleted.
    Delete {
        /// The triple's term ids.
        ids: TripleIds,
        /// The triple.
        triple: Arc<Triple>,
    },
    /// A graph operation on `graph`.
    Graph {
        /// The graph.
        graph: GraphKey,
        /// What happened to it.
        change: GraphChange,
    },
}

/// What a graph operation did to a graph.
#[derive(Debug, Clone)]
pub enum GraphChange {
    /// The graph was created, empty.
    Created,
    /// The graph was dropped; `before` is its partition as the drop found it.
    Dropped {
        /// The partition the drop swapped out.
        before: Arc<RdfStore>,
    },
    /// The graph was emptied; `before` is its partition as the clear found it.
    Cleared {
        /// The partition the clear swapped out.
        before: Arc<RdfStore>,
    },
}

/// What one commit changed, at its epoch.
#[derive(Debug)]
pub struct OverlayEntry {
    epoch: EpochId,
    changes: Vec<OverlayChange>,
}

impl OverlayEntry {
    /// The changes of the commit at `epoch`, in recorded order.
    #[must_use]
    pub fn new(epoch: EpochId, changes: Vec<OverlayChange>) -> Self {
        Self { epoch, changes }
    }

    /// The epoch of the commit.
    #[must_use]
    pub fn epoch(&self) -> EpochId {
        self.epoch
    }

    /// The changes of the commit, in recorded order.
    #[must_use]
    pub fn changes(&self) -> &[OverlayChange] {
        &self.changes
    }
}

/// A view at an epoch whose overlay entries are pruned: it would miss the
/// commits after it. A reader that pins its epoch never meets it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SnapshotGone {
    /// The epoch asked for.
    pub epoch: EpochId,
    /// The epoch the overlay is pruned through.
    pub pruned_through: EpochId,
}

impl std::fmt::Display for SnapshotGone {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "no RDF snapshot at epoch {}: the changes after it are pruned through epoch {}",
            self.epoch.as_u64(),
            self.pruned_through.as_u64()
        )
    }
}

impl std::error::Error for SnapshotGone {}

/// The entries of the commits a reader may still need, in commit order.
#[derive(Debug, Default)]
pub struct RdfOverlay {
    entries: RwLock<VecDeque<Arc<OverlayEntry>>>,
    /// The epoch of the newest entry, 0 with none: one atomic for a reader
    /// that finds no commit after its snapshot, the common case.
    newest: AtomicU64,
    /// How many entries are kept: one atomic for a commit that checks
    /// whether there is anything to prune.
    held: AtomicUsize,
    /// The epoch the entries are pruned through: a view below it would miss
    /// commits.
    pruned_through: AtomicU64,
    /// How many times a view extended its changes, over all views: for
    /// tests (none on the fast path).
    views_extended: AtomicU64,
}

impl RdfOverlay {
    /// Publishes `entry`, whose epoch is above every entry's so far.
    ///
    /// # Panics
    ///
    /// When the epoch is not above the newest entry's: commits publish in
    /// epoch order.
    pub fn publish(&self, entry: OverlayEntry) {
        let mut entries = self.entries.write();
        assert!(
            entries.back().is_none_or(|last| last.epoch < entry.epoch),
            "overlay entries are published in epoch order"
        );
        let epoch = entry.epoch.as_u64();
        entries.push_back(Arc::new(entry));
        self.held.store(entries.len(), Ordering::Release);
        self.newest.store(epoch, Ordering::Release);
    }

    /// The epoch of the newest entry, 0 with none.
    #[must_use]
    pub fn newest_epoch(&self) -> u64 {
        self.newest.load(Ordering::Acquire)
    }

    /// The entry of the commit at `epoch`, while it is kept.
    #[must_use]
    pub fn entry_at(&self, epoch: EpochId) -> Option<Arc<OverlayEntry>> {
        if self.newest_epoch() < epoch.as_u64() {
            return None;
        }
        self.entries
            .read()
            .iter()
            .rev()
            .find(|entry| entry.epoch == epoch)
            .cloned()
    }

    /// The entries of the commits after `epoch`, in commit order.
    #[must_use]
    pub fn since(&self, epoch: EpochId) -> Vec<Arc<OverlayEntry>> {
        if self.newest_epoch() <= epoch.as_u64() {
            return Vec::new();
        }
        self.entries
            .read()
            .iter()
            .filter(|entry| entry.epoch > epoch)
            .cloned()
            .collect()
    }

    /// Drops the entries at or below `epoch`: no reader pins an epoch below
    /// it, so none needs them. A view below `epoch` is refused from now on
    /// ([`pruned_through`](Self::pruned_through)).
    pub fn prune_below(&self, epoch: EpochId) {
        let mut entries = self.entries.write();
        self.pruned_through
            .fetch_max(epoch.as_u64(), Ordering::AcqRel);
        while entries.front().is_some_and(|entry| entry.epoch <= epoch) {
            entries.pop_front();
        }
        self.held.store(entries.len(), Ordering::Release);
    }

    /// The epoch the entries are pruned through, 0 before the first prune.
    #[must_use]
    pub fn pruned_through(&self) -> EpochId {
        EpochId::new(self.pruned_through.load(Ordering::Acquire))
    }

    /// How many entries are kept.
    #[must_use]
    pub fn len(&self) -> usize {
        self.held.load(Ordering::Acquire)
    }

    /// Whether no entry is kept: one atomic load.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// How many times a view extended its changes, over all views of this
    /// overlay: 0 while every reader takes the fast path.
    #[must_use]
    pub fn views_extended(&self) -> u64 {
        self.views_extended.load(Ordering::Relaxed)
    }
}

/// A graph as a reader at a snapshot sees it, where commits after the
/// snapshot changed it.
#[derive(Debug, Default)]
pub struct GraphAtSnapshot {
    /// Triples the store holds that were absent at the snapshot: their first
    /// change after it was an insert.
    pub absent: FxHashSet<TripleIds>,
    /// Triples present at the snapshot that a later commit deleted: their
    /// first change after it was a delete.
    pub present: Vec<Arc<Triple>>,
    /// Where the graph's triples are read from.
    pub partition: Partition,
}

/// Where a reader at a snapshot reads a graph's triples from.
#[derive(Debug, Default, Clone)]
pub enum Partition {
    /// The store's current partition, adjusted by `absent` and `present`.
    #[default]
    Current,
    /// The partition a later `DROP` or `CLEAR` swapped out, adjusted by the
    /// changes made to it after the snapshot and before the swap.
    Retained(Arc<RdfStore>),
    /// The graph did not exist at the snapshot: a later commit created it.
    Absent,
}

/// What the commits after one snapshot changed, per graph: each triple and
/// graph by its first change after the snapshot.
#[derive(Debug, Default)]
pub struct SnapshotChanges {
    graphs: FxHashMap<GraphKey, GraphAtSnapshot>,
    /// The triples whose first change after the snapshot is recorded: a
    /// later change of one is past it and never replaces it.
    seen: FxHashSet<(GraphKey, TripleIds)>,
    /// The graphs a `DROP`, `CLEAR` or `CREATE` after the snapshot swapped:
    /// their later triple changes are in the new partition, which the reader
    /// does not read.
    swapped: FxHashSet<GraphKey>,
}

impl SnapshotChanges {
    /// Adds the changes of `entry`, a commit after the snapshot and after
    /// every entry added so far.
    fn add(&mut self, entry: &OverlayEntry) {
        static DEFAULT: GraphKey = None;
        let mut graph = &DEFAULT;
        for change in &entry.changes {
            match change {
                OverlayChange::In(next) => graph = next,
                OverlayChange::Inserts(run) => {
                    for triple in run {
                        let ids = *triple.raw_ids();
                        if self.first_change(graph, ids) {
                            self.graphs
                                .entry(graph.clone())
                                .or_default()
                                .absent
                                .insert(ids);
                        }
                    }
                }
                OverlayChange::Delete { ids, triple } => {
                    if self.first_change(graph, *ids) {
                        self.graphs
                            .entry(graph.clone())
                            .or_default()
                            .present
                            .push(Arc::clone(triple));
                    }
                }
                OverlayChange::Graph {
                    graph: swapped,
                    change,
                } => {
                    if !self.swapped.insert(swapped.clone()) {
                        continue;
                    }
                    self.graphs.entry(swapped.clone()).or_default().partition = match change {
                        // Created after the snapshot: it did not exist then.
                        GraphChange::Created => Partition::Absent,
                        GraphChange::Dropped { before } | GraphChange::Cleared { before } => {
                            Partition::Retained(Arc::clone(before))
                        }
                    };
                }
            }
        }
    }

    /// Whether a change of `ids` in `graph` is its first after the
    /// snapshot, in the partition the reader reads: one after the graph's
    /// first swap is in the new partition.
    fn first_change(&mut self, graph: &GraphKey, ids: TripleIds) -> bool {
        !self.swapped.contains(graph) && self.seen.insert((graph.clone(), ids))
    }

    /// What the commits after the snapshot changed in `graph`, `None` when
    /// they changed nothing there.
    #[must_use]
    pub fn graph(&self, graph: &GraphKey) -> Option<&GraphAtSnapshot> {
        self.graphs.get(graph)
    }

    /// The graphs the commits after the snapshot changed.
    pub fn graphs(&self) -> impl Iterator<Item = (&GraphKey, &GraphAtSnapshot)> {
        self.graphs.iter()
    }
}

/// What a snapshot has worked out, up to the entry at `cursor`.
#[derive(Debug)]
struct SnapshotState {
    /// The epoch of the last entry added (the snapshot's own epoch before
    /// the first).
    cursor: u64,
    changes: SnapshotChanges,
}

/// A reader's snapshot: its epoch, and what the commits after it changed.
///
/// The changes are worked out at the first lookup that needs them and grow
/// as later commits publish: a commit after the map was built can hold the
/// first change after the snapshot of a triple the map has not seen. A
/// lookup brings them up to the overlay's newest entry inside the store's
/// apply guard, so the store and the changes it reads agree. The parts of
/// one query share them.
#[derive(Debug)]
pub struct Snapshot {
    epoch: EpochId,
    overlay: Arc<RdfOverlay>,
    state: RecursiveRwLock<SnapshotState>,
    /// How many times the changes were extended, for tests: none on the
    /// fast path.
    extensions: AtomicU64,
}

impl Snapshot {
    /// The snapshot at `epoch` of `overlay`.
    #[must_use]
    pub fn at(overlay: &Arc<RdfOverlay>, epoch: EpochId) -> Self {
        Self {
            epoch,
            overlay: Arc::clone(overlay),
            state: RecursiveRwLock::new(SnapshotState {
                cursor: epoch.as_u64(),
                changes: SnapshotChanges::default(),
            }),
            extensions: AtomicU64::new(0),
        }
    }

    /// The epoch the reader reads at.
    #[must_use]
    pub fn epoch(&self) -> EpochId {
        self.epoch
    }

    /// What the commits after the snapshot changed, up to the overlay's
    /// newest entry; `None` when no commit followed it (the store is then
    /// read as it is, and nothing is built). Called inside the dataset's
    /// apply guard, under which the newest entry does not move: a nested
    /// call on the same thread finds the changes up to date and takes their
    /// read side again (recursively, so a waiting extension by another
    /// reader of the snapshot cannot block it).
    pub fn changes(&self) -> Option<ChangesGuard<'_>> {
        let newest = self.overlay.newest_epoch();
        if newest <= self.epoch.as_u64() {
            return None;
        }
        {
            let state = self.state.read();
            if state.cursor >= newest {
                return Some(ChangesGuard(state));
            }
        }
        let mut state = self.state.write();
        if state.cursor < newest {
            for entry in self.overlay.since(EpochId::new(state.cursor)) {
                state.changes.add(&entry);
                state.cursor = state.cursor.max(entry.epoch.as_u64());
            }
            self.extensions.fetch_add(1, Ordering::Relaxed);
            self.overlay.views_extended.fetch_add(1, Ordering::Relaxed);
        }
        Some(ChangesGuard(
            parking_lot::RecursiveRwLockWriteGuard::downgrade(state),
        ))
    }

    /// How many times the changes were extended.
    #[must_use]
    pub fn extensions(&self) -> u64 {
        self.extensions.load(Ordering::Relaxed)
    }
}

/// The changes of a snapshot, held for one lookup.
pub struct ChangesGuard<'a>(parking_lot::RecursiveRwLockReadGuard<'a, SnapshotState>);

impl std::ops::Deref for ChangesGuard<'_> {
    type Target = SnapshotChanges;

    fn deref(&self) -> &SnapshotChanges {
        &self.0.changes
    }
}

/// The state a read sees.
///
/// Every read of the store takes one, so a read that forgets the reader's
/// snapshot does not compile. A reader takes the view at its pinned epoch
/// ([`RdfStore::view_at`]); the newest committed state can only be named,
/// with the reason it is the right one ([`LatestUse`]).
#[derive(Debug, Clone)]
pub struct RdfView(ViewKind);

#[derive(Debug, Clone)]
enum ViewKind {
    Latest(LatestUse),
    Snapshot(Arc<Snapshot>),
}

/// Why a read takes the newest committed state instead of a snapshot.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LatestUse {
    /// Applying a commit, which reads what it changes.
    Apply,
    /// Recovery and loads at open, before any reader exists.
    Recovery,
    /// A checkpoint, copy or export that holds commits off while it reads.
    Checkpoint,
    /// Optimizer statistics and memory accounting: estimates, not answers.
    Statistics,
    /// A test, or a caller of the store outside any database.
    Test,
}

impl RdfView {
    /// The newest committed state, for `why`.
    #[must_use]
    pub const fn latest(why: LatestUse) -> Self {
        Self(ViewKind::Latest(why))
    }

    /// The view at `epoch` of `overlay`.
    #[must_use]
    pub fn at(overlay: &Arc<RdfOverlay>, epoch: EpochId) -> Self {
        Self(ViewKind::Snapshot(Arc::new(Snapshot::at(overlay, epoch))))
    }

    /// Why the view takes the newest committed state, `None` for a
    /// snapshot.
    #[must_use]
    pub fn latest_use(&self) -> Option<LatestUse> {
        match &self.0 {
            ViewKind::Latest(why) => Some(*why),
            ViewKind::Snapshot(_) => None,
        }
    }

    /// The epoch of the snapshot, `None` for the newest state.
    #[must_use]
    pub fn epoch(&self) -> Option<EpochId> {
        match &self.0 {
            ViewKind::Latest(_) => None,
            ViewKind::Snapshot(snapshot) => Some(snapshot.epoch()),
        }
    }

    /// What the commits after the view changed, `None` for the newest state
    /// or a snapshot no commit followed. Called inside the store's apply
    /// guard.
    #[must_use]
    pub fn changes(&self) -> Option<ChangesGuard<'_>> {
        match &self.0 {
            ViewKind::Latest(_) => None,
            ViewKind::Snapshot(snapshot) => snapshot.changes(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::rdf::Term;

    fn triple(subject: &str) -> Arc<Triple> {
        Arc::new(Triple::new(
            Term::iri(format!("http://example.org/{subject}")),
            Term::iri("http://example.org/name"),
            Term::literal(subject),
        ))
    }

    /// The triple of `subject` as a store holds it, with the raw ids `ids`.
    fn stored(subject: &str, ids: TripleIds) -> Arc<Triple> {
        Arc::new(
            Triple::new(
                Term::iri(format!("http://example.org/{subject}")),
                Term::iri("http://example.org/name"),
                Term::literal(subject),
            )
            .with_raw_ids(ids),
        )
    }

    /// An insert into `graph`, after the `In` that names it.
    fn insert(graph: Option<&str>, ids: TripleIds, name: &str) -> Vec<OverlayChange> {
        vec![
            OverlayChange::In(graph.map(Arc::from)),
            OverlayChange::Inserts(vec![stored(name, ids)]),
        ]
    }

    /// A delete from `graph`, after the `In` that names it.
    fn delete(graph: Option<&str>, ids: TripleIds, name: &str) -> Vec<OverlayChange> {
        vec![
            OverlayChange::In(graph.map(Arc::from)),
            OverlayChange::Delete {
                ids,
                triple: triple(name),
            },
        ]
    }

    fn overlay(entries: Vec<(u64, Vec<OverlayChange>)>) -> Arc<RdfOverlay> {
        let overlay = Arc::new(RdfOverlay::default());
        for (epoch, changes) in entries {
            overlay.publish(OverlayEntry::new(EpochId::new(epoch), changes));
        }
        overlay
    }

    #[test]
    fn a_snapshot_no_commit_followed_needs_no_changes() {
        let overlay = overlay(vec![(3, insert(None, [1, 2, 3], "Alix"))]);
        assert!(RdfView::at(&overlay, EpochId::new(3)).changes().is_none());
        assert!(RdfView::at(&overlay, EpochId::new(19)).changes().is_none());
        assert!(RdfView::latest(LatestUse::Test).changes().is_none());
    }

    #[test]
    fn the_first_change_after_the_snapshot_decides() {
        // At 2: Gus present, Alix absent. 3 deletes Gus, inserts Alix; 4 puts Gus
        // back and deletes Alix again. A reader at 2 sees Gus and not Alix.
        let overlay = overlay(vec![
            (
                3,
                [
                    delete(None, [7, 2, 7], "Gus"),
                    insert(None, [1, 2, 1], "Alix"),
                ]
                .concat(),
            ),
            (
                4,
                [
                    insert(None, [7, 2, 7], "Gus"),
                    delete(None, [1, 2, 1], "Alix"),
                ]
                .concat(),
            ),
        ]);
        let view = RdfView::at(&overlay, EpochId::new(2));
        let changes = view.changes().unwrap();
        let graph = changes.graph(&None).unwrap();
        assert!(
            graph.absent.contains(&[1, 2, 1]),
            "Alix was inserted after 2"
        );
        assert!(!graph.absent.contains(&[7, 2, 7]));
        assert_eq!(graph.present.len(), 1, "Gus was deleted after 2");
        assert!(matches!(graph.partition, Partition::Current));

        // A reader at 3 sees the opposite.
        let view = RdfView::at(&overlay, EpochId::new(3));
        let changes = view.changes().unwrap();
        let graph = changes.graph(&None).unwrap();
        assert!(graph.absent.contains(&[7, 2, 7]));
        assert_eq!(graph.present.len(), 1);
    }

    #[test]
    fn a_swapped_graph_is_read_from_its_retained_partition() {
        let before = Arc::new(RdfStore::new());
        let overlay = overlay(vec![
            (3, insert(Some("g"), [1, 2, 1], "Alix")),
            (
                4,
                [
                    vec![OverlayChange::Graph {
                        graph: Some(Arc::from("g")),
                        change: GraphChange::Cleared {
                            before: Arc::clone(&before),
                        },
                    }],
                    insert(Some("g"), [7, 2, 7], "Gus"),
                ]
                .concat(),
            ),
            (
                5,
                vec![OverlayChange::Graph {
                    graph: Some(Arc::from("h")),
                    change: GraphChange::Created,
                }],
            ),
        ]);
        let view = RdfView::at(&overlay, EpochId::new(2));
        let changes = view.changes().unwrap();
        let g = changes.graph(&Some(Arc::from("g"))).unwrap();
        assert!(matches!(&g.partition, Partition::Retained(kept) if Arc::ptr_eq(kept, &before)));
        assert!(
            g.absent.contains(&[1, 2, 1]),
            "inserted before the clear, after 2"
        );
        assert!(
            !g.absent.contains(&[7, 2, 7]),
            "inserted after the swap, into the new partition"
        );
        let h = changes.graph(&Some(Arc::from("h"))).unwrap();
        assert!(matches!(h.partition, Partition::Absent), "created after 2");
    }

    /// A triple change is in the graph of the last `In` before it, and in the
    /// default graph before the first.
    #[test]
    fn a_triple_change_is_in_the_graph_of_the_last_in() {
        let overlay = overlay(vec![(
            3,
            vec![
                OverlayChange::Inserts(vec![stored("Alix", [1, 2, 1])]),
                OverlayChange::In(Some(Arc::from("g"))),
                OverlayChange::Inserts(vec![stored("Gus", [7, 2, 7])]),
                OverlayChange::In(None),
                OverlayChange::Delete {
                    ids: [8, 2, 8],
                    triple: triple("Vincent"),
                },
            ],
        )]);
        let view = RdfView::at(&overlay, EpochId::new(2));
        let changes = view.changes().unwrap();
        let default = changes.graph(&None).unwrap();
        assert!(default.absent.contains(&[1, 2, 1]) && !default.absent.contains(&[7, 2, 7]));
        assert_eq!(default.present.len(), 1, "Vincent, after In(None)");
        let g = changes.graph(&Some(Arc::from("g"))).unwrap();
        assert!(g.absent.contains(&[7, 2, 7]) && g.absent.len() == 1);
        assert_eq!(g.present, Vec::new());
    }

    /// The fast path builds nothing: a snapshot no commit followed reads
    /// the store as it is.
    #[test]
    fn the_fast_path_builds_no_changes() {
        let overlay = overlay(vec![(3, insert(None, [1, 2, 1], "Alix"))]);
        let RdfView(ViewKind::Snapshot(snapshot)) = RdfView::at(&overlay, EpochId::new(3)) else {
            unreachable!()
        };
        assert!(snapshot.changes().is_none());
        assert_eq!(snapshot.extensions(), 0);
    }

    /// A commit after the changes were built can hold the first change after
    /// the snapshot of a triple they have not seen: the next lookup adds it,
    /// and a first change already recorded is never replaced.
    #[test]
    fn later_commits_extend_the_changes() {
        let overlay = overlay(vec![(3, insert(None, [1, 2, 1], "Alix"))]);
        let view = RdfView::at(&overlay, EpochId::new(2));
        {
            let changes = view.changes().unwrap();
            let graph = changes.graph(&None).unwrap();
            assert!(graph.absent.contains(&[1, 2, 1]));
            assert_eq!(graph.present, Vec::new());
        }
        // Gus, untouched when the changes were built, is deleted later; Alix's
        // later delete is past her first change after 2.
        overlay.publish(OverlayEntry::new(
            EpochId::new(4),
            [
                delete(None, [7, 2, 7], "Gus"),
                delete(None, [1, 2, 1], "Alix"),
            ]
            .concat(),
        ));
        let changes = view.changes().unwrap();
        let graph = changes.graph(&None).unwrap();
        assert_eq!(graph.present.len(), 1, "Gus was present at 2");
        assert!(graph.absent.contains(&[1, 2, 1]), "Alix stays absent at 2");
        let RdfView(ViewKind::Snapshot(snapshot)) = &view else {
            unreachable!()
        };
        assert_eq!(snapshot.extensions(), 2);
        drop(changes);
        let _again = view.changes().unwrap();
        assert_eq!(snapshot.extensions(), 2, "nothing new to add");
    }

    /// A triple present at the snapshot whose graph is dropped, then put back
    /// into the new graph and deleted there, is still present at the snapshot:
    /// collecting stops at the graph's first swap.
    #[test]
    fn changes_after_a_swap_belong_to_the_new_partition() {
        let before = Arc::new(RdfStore::new());
        let overlay = overlay(vec![
            (
                3,
                vec![OverlayChange::Graph {
                    graph: Some(Arc::from("g")),
                    change: GraphChange::Dropped {
                        before: Arc::clone(&before),
                    },
                }],
            ),
            (4, insert(Some("g"), [1, 2, 1], "Alix")),
            (5, delete(Some("g"), [1, 2, 1], "Alix")),
        ]);
        let view = RdfView::at(&overlay, EpochId::new(2));
        let changes = view.changes().unwrap();
        let g = changes.graph(&Some(Arc::from("g"))).unwrap();
        assert!(matches!(&g.partition, Partition::Retained(kept) if Arc::ptr_eq(kept, &before)));
        assert!(g.absent.is_empty() && g.present.is_empty());
    }

    #[test]
    fn pruning_keeps_the_entries_a_reader_may_need() {
        let overlay = overlay(vec![
            (3, insert(None, [1, 2, 1], "Alix")),
            (4, insert(None, [7, 2, 7], "Gus")),
            (5, insert(None, [8, 2, 8], "Vincent")),
        ]);
        overlay.prune_below(EpochId::new(3));
        assert_eq!(overlay.len(), 2);
        assert_eq!(
            overlay.since(EpochId::new(2)).len(),
            2,
            "what is left of it"
        );
        overlay.prune_below(EpochId::new(5));
        assert!(overlay.is_empty());
        assert_eq!(overlay.newest_epoch(), 5, "the newest epoch stays");
        assert!(overlay.since(EpochId::new(4)).is_empty());
    }

    #[test]
    #[should_panic(expected = "epoch order")]
    fn entries_are_published_in_epoch_order() {
        let overlay = overlay(vec![(4, vec![])]);
        overlay.publish(OverlayEntry::new(EpochId::new(4), vec![]));
    }

    /// A run of inserts reads as each of its triples: a view before it sees
    /// none of them, a view at it or after sees all of them (no change map),
    /// and the first change after a view decides across runs.
    #[test]
    fn a_run_of_inserts_reads_triple_by_triple() {
        let run = vec![
            stored("Alix", [1, 2, 1]),
            stored("Gus", [7, 2, 7]),
            stored("Mia", [8, 2, 8]),
        ];
        let overlay = overlay(vec![
            (3, vec![OverlayChange::Inserts(run.clone())]),
            (
                4,
                vec![OverlayChange::Delete {
                    ids: [7, 2, 7],
                    triple: Arc::clone(&run[1]),
                }],
            ),
            (5, vec![OverlayChange::Inserts(vec![Arc::clone(&run[1])])]),
        ]);
        let before = RdfView::at(&overlay, EpochId::new(2));
        let changes = before.changes().unwrap();
        let graph = changes.graph(&None).unwrap();
        assert_eq!(graph.absent.len(), 3, "none of the run at 2");
        assert_eq!(graph.present, Vec::new());
        drop(changes);

        let between = RdfView::at(&overlay, EpochId::new(3));
        let changes = between.changes().unwrap();
        let graph = changes.graph(&None).unwrap();
        assert!(graph.absent.is_empty(), "the run is there at 3");
        assert_eq!(
            graph
                .present
                .iter()
                .map(|t| *t.raw_ids())
                .collect::<Vec<_>>(),
            [[7, 2, 7]],
            "Gus is present at 3: deleted after it, inserted again later"
        );
        drop(changes);

        assert!(RdfView::at(&overlay, EpochId::new(5)).changes().is_none());
    }

    /// A pruned entry drops its run: the run's triples are held by nothing
    /// else of the overlay.
    #[test]
    fn a_pruned_run_drops_its_triples() {
        let alix = stored("Alix", [1, 2, 1]);
        let overlay = overlay(vec![(
            3,
            vec![OverlayChange::Inserts(vec![Arc::clone(&alix)])],
        )]);
        assert_eq!(Arc::strong_count(&alix), 2);
        assert!(overlay.entry_at(EpochId::new(3)).is_some());
        overlay.prune_below(EpochId::new(3));
        assert_eq!(
            Arc::strong_count(&alix),
            1,
            "the entry and its run are gone"
        );
        assert!(overlay.entry_at(EpochId::new(3)).is_none());
    }
}
