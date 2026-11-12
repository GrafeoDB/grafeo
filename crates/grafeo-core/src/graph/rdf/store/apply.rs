//! A commit's RDF changes, applied in recorded order under one hold of the
//! dataset's apply guard, and published for readers at earlier snapshots
//! (#718).
//!
//! A graph operation swaps a partition out instead of changing it in place:
//! a `DROP` removes the graph's partition, a `CLEAR` puts an empty one in its
//! place, a `CREATE` adds one. The partition swapped out goes into the
//! overlay entry, where a reader at an earlier snapshot still reads it.

use std::sync::Arc;

use grafeo_common::change::RdfGraphOp;
use grafeo_common::storage::log_record::RdfGraphTarget;
use grafeo_common::testing::commit_hook::{RdfApplyStep, run_during_rdf_apply};
use grafeo_common::types::EpochId;

use super::super::overlay::{
    GraphChange, GraphKey, OverlayChange, OverlayEntry, RdfOverlay, RdfView, SnapshotGone,
};
use super::{Index, RdfStore, Triple};

/// One change a commit applies, in recorded order.
#[derive(Debug, Clone)]
pub enum CommitChange {
    /// Inserts `triple` into `graph` (`None` for the default graph), which a
    /// named graph that does not exist yet comes into being for.
    Insert {
        /// The graph.
        graph: GraphKey,
        /// The triple.
        triple: Triple,
    },
    /// Deletes `triple` from `graph`; a graph that does not hold it is left
    /// as it is.
    Delete {
        /// The graph.
        graph: GraphKey,
        /// The triple.
        triple: Triple,
    },
    /// A whole-graph operation.
    Graph(RdfGraphOp),
}

/// Whether an apply publishes what it changed for readers at earlier
/// snapshots.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ApplyMode {
    /// A live commit at this epoch, above every snapshot a reader holds: its
    /// overlay entry is published before the guard is released.
    Publish(EpochId),
    /// Replay and loads at open: no reader exists, nothing is published.
    Recovery,
}

/// What an apply changed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Applied {
    /// For each change, in order, whether it changed the store: an insert of
    /// a triple the graph held, a delete of one it did not, a `CREATE` of a
    /// graph that exists and a `DROP` of a named graph that does not changed
    /// nothing.
    pub changed: Vec<bool>,
}

/// The overlay changes an apply collects, `None` when it publishes nothing.
type Collected = Option<Vec<OverlayChange>>;

impl RdfStore {
    /// Applies `changes` in order under one hold of the dataset's apply
    /// guard, and with [`ApplyMode::Publish`] publishes the overlay entry of
    /// the changes that changed the store before it releases the guard: a
    /// reader at an earlier snapshot sees the commit nowhere, a reader after
    /// it sees all of it. Called on the dataset's default graph.
    pub fn apply_commit(&self, mode: ApplyMode, changes: Vec<CommitChange>) -> Applied {
        let guard = self.apply_guard.write();
        run_during_rdf_apply(RdfApplyStep::GuardTaken);
        let mut collected: Collected = match mode {
            ApplyMode::Publish(_) => Some(Vec::new()),
            ApplyMode::Recovery => None,
        };
        let mut changed = Vec::with_capacity(changes.len());
        let mut changes = changes.into_iter().peekable();
        let mut first = true;
        while let Some(change) = changes.next() {
            match change {
                CommitChange::Insert { graph, triple } => {
                    // A run of inserts into one graph goes in one batch.
                    let mut run = vec![triple];
                    while let Some(CommitChange::Insert { graph: next, .. }) = changes.peek()
                        && *next == graph
                    {
                        if let Some(CommitChange::Insert { triple, .. }) = changes.next() {
                            run.push(triple);
                        }
                    }
                    self.apply_inserts(&graph, run, &mut changed, &mut collected);
                }
                CommitChange::Delete { graph, triple } => {
                    changed.push(self.apply_delete(&graph, &triple, &mut collected));
                }
                CommitChange::Graph(op) => {
                    changed.push(self.apply_graph_change(&op, &mut collected));
                }
            }
            if first && changes.peek().is_some() {
                first = false;
                run_during_rdf_apply(RdfApplyStep::Halfway);
            }
        }
        run_during_rdf_apply(RdfApplyStep::BeforePublish);
        if let (ApplyMode::Publish(epoch), Some(collected)) = (mode, collected)
            && !collected.is_empty()
        {
            self.overlay.publish(OverlayEntry::new(epoch, collected));
        }
        drop(guard);
        Applied { changed }
    }

    /// Drops the overlay entries at or below `horizon`, the read horizon: no
    /// reader pins an epoch below it.
    pub fn prune_overlay(&self, horizon: EpochId) {
        self.overlay.prune_below(horizon);
    }

    /// Whether the overlay holds an entry: one atomic load, so a commit with
    /// nothing to prune pays no lock.
    #[must_use]
    pub fn overlay_holds_entries(&self) -> bool {
        !self.overlay.is_empty()
    }

    /// The dataset's overlay, for its counters.
    #[must_use]
    pub fn overlay(&self) -> &RdfOverlay {
        &self.overlay
    }

    /// The view at `epoch`, a reader's pinned snapshot.
    ///
    /// # Errors
    ///
    /// [`SnapshotGone`] when the overlay is pruned past `epoch`: a view there
    /// would miss the commits after it.
    pub fn view_at(&self, epoch: EpochId) -> Result<RdfView, SnapshotGone> {
        let pruned_through = self.overlay.pruned_through();
        if epoch < pruned_through {
            return Err(SnapshotGone {
                epoch,
                pruned_through,
            });
        }
        Ok(RdfView::at(&self.overlay, epoch))
    }

    /// Inserts `triples` into `graph`, one batch, recording each one that is
    /// new, and the graph's creation when it did not exist.
    fn apply_inserts(
        &self,
        graph: &GraphKey,
        triples: Vec<Triple>,
        changed: &mut Vec<bool>,
        collected: &mut Collected,
    ) {
        let target = graph
            .as_ref()
            .map(|name| self.partition_or_create(name, collected));
        let target = target.as_deref().unwrap_or(self);
        let start = changed.len();
        let valid: Vec<bool> = triples.iter().map(Triple::is_valid).collect();
        let mut new = Vec::with_capacity(triples.len());
        target.insert_stored_unguarded(
            triples
                .into_iter()
                .filter(Triple::is_valid)
                .map(|triple| self.stored(triple)),
            |triple, inserted| {
                new.push(inserted);
                if inserted && let Some(collected) = collected.as_mut() {
                    collected.push(OverlayChange::Insert {
                        graph: graph.clone(),
                        ids: *triple.raw_ids(),
                        triple: Arc::clone(triple),
                    });
                }
            },
        );
        let mut new = new.into_iter();
        changed.extend(
            valid
                .into_iter()
                .map(|valid| valid && new.next().unwrap_or(false)),
        );
        debug_assert!(changed.len() >= start);
    }

    /// Deletes `triple` from `graph`, recording it when the graph held it.
    fn apply_delete(&self, graph: &GraphKey, triple: &Triple, collected: &mut Collected) -> bool {
        let partition = match graph {
            None => None,
            Some(name) => match self.graph(name) {
                Some(partition) => Some(partition),
                None => return false,
            },
        };
        let target = partition.as_deref().unwrap_or(self);
        let Some(removed) = target.remove_unguarded(triple) else {
            return false;
        };
        if let Some(collected) = collected.as_mut() {
            collected.push(OverlayChange::Delete {
                graph: graph.clone(),
                ids: *removed.raw_ids(),
                triple: removed,
            });
        }
        true
    }

    /// Applies a whole-graph operation as swaps of partitions, recording
    /// each. Returns what [`apply_graph_op`](Self::apply_graph_op) returns.
    fn apply_graph_change(&self, op: &RdfGraphOp, collected: &mut Collected) -> bool {
        match op {
            RdfGraphOp::Create { name } => {
                if self.graph(name).is_some() {
                    return false;
                }
                self.partition_or_create(name, collected);
                true
            }
            RdfGraphOp::Drop {
                target: RdfGraphTarget::Named(name),
            } => self.drop_partition(name, collected),
            RdfGraphOp::Drop { target } | RdfGraphOp::Clear { target } => {
                match target {
                    RdfGraphTarget::Default => self.swap_default(collected),
                    RdfGraphTarget::Named(name) => self.swap_named(name, collected),
                    RdfGraphTarget::AllNamed => self.drop_every_partition(collected),
                    RdfGraphTarget::All => {
                        self.swap_default(collected);
                        self.drop_every_partition(collected);
                    }
                }
                true
            }
            RdfGraphOp::Copy { source, target } => {
                if source != target {
                    let triples = self.graph_triples(source.as_deref());
                    self.empty_graph(target.as_deref(), collected);
                    self.insert_triples(target.as_deref(), triples, collected);
                }
                true
            }
            RdfGraphOp::Move { source, target } => {
                if source != target {
                    let triples = self.graph_triples(source.as_deref());
                    self.empty_graph(target.as_deref(), collected);
                    self.insert_triples(target.as_deref(), triples, collected);
                    match source.as_deref() {
                        None => self.swap_default(collected),
                        Some(name) => {
                            self.drop_partition(name, collected);
                        }
                    }
                }
                true
            }
            RdfGraphOp::Add { source, target } => {
                if source != target {
                    let triples = self.graph_triples(source.as_deref());
                    self.insert_triples(target.as_deref(), triples, collected);
                }
                true
            }
        }
    }

    /// The partition of the named graph `name`, created (and recorded) when
    /// the dataset has none.
    fn partition_or_create(&self, name: &str, collected: &mut Collected) -> Arc<RdfStore> {
        if let Some(partition) = self.graph(name) {
            return partition;
        }
        let partition = Arc::new(self.named_graph_store(name));
        self.named_graphs
            .write()
            .insert(name.to_string(), Arc::clone(&partition));
        if let Some(collected) = collected.as_mut() {
            collected.push(OverlayChange::Graph {
                graph: Some(Arc::from(name)),
                change: GraphChange::Created,
            });
        }
        partition
    }

    /// Removes the named graph `name`'s partition, recording it; `false`
    /// when there is none.
    fn drop_partition(&self, name: &str, collected: &mut Collected) -> bool {
        let Some(before) = self.named_graphs.write().remove(name) else {
            return false;
        };
        if let Some(collected) = collected.as_mut() {
            collected.push(OverlayChange::Graph {
                graph: Some(Arc::from(name)),
                change: GraphChange::Dropped { before },
            });
        }
        true
    }

    /// Removes every named graph's partition, recording each in name order.
    fn drop_every_partition(&self, collected: &mut Collected) {
        let graphs = std::mem::take(&mut *self.named_graphs.write());
        let Some(collected) = collected.as_mut() else {
            return;
        };
        let mut graphs: Vec<(String, Arc<RdfStore>)> = graphs.into_iter().collect();
        graphs.sort_unstable_by(|(a, _), (b, _)| a.cmp(b));
        collected.extend(
            graphs
                .into_iter()
                .map(|(name, before)| OverlayChange::Graph {
                    graph: Some(Arc::from(name.as_str())),
                    change: GraphChange::Dropped { before },
                }),
        );
    }

    /// Puts an empty partition in the place of the named graph `name`'s,
    /// recording the one swapped out; a graph that does not exist stays so.
    fn swap_named(&self, name: &str, collected: &mut Collected) {
        let before = {
            let mut graphs = self.named_graphs.write();
            let Some(slot) = graphs.get_mut(name) else {
                return;
            };
            std::mem::replace(slot, Arc::new(self.named_graph_store(name)))
        };
        if let Some(collected) = collected.as_mut() {
            collected.push(OverlayChange::Graph {
                graph: Some(Arc::from(name)),
                change: GraphChange::Cleared { before },
            });
        }
    }

    /// Empties the default graph by moving its contents into a partition of
    /// their own, recorded as swapped out; an empty default graph is left as
    /// it is.
    fn swap_default(&self, collected: &mut Collected) {
        if self.triples.read().is_empty() {
            return;
        }
        let before = self.take_default_contents();
        if let Some(collected) = collected.as_mut() {
            collected.push(OverlayChange::Graph {
                graph: None,
                change: GraphChange::Cleared {
                    before: Arc::new(before),
                },
            });
        }
    }

    /// Empties `graph` for a `COPY` or `MOVE` onto it: a named graph that
    /// does not exist is created.
    fn empty_graph(&self, graph: Option<&str>, collected: &mut Collected) {
        match graph {
            None => self.swap_default(collected),
            Some(name) if self.graph(name).is_some() => self.swap_named(name, collected),
            Some(name) => {
                self.partition_or_create(name, collected);
            }
        }
    }

    /// The triples of `graph` (`None` for the default graph) as they are.
    fn graph_triples(&self, graph: Option<&str>) -> Vec<Arc<Triple>> {
        match graph {
            None => self.triples_latest(),
            Some(name) => self
                .graph(name)
                .map(|partition| partition.triples_latest())
                .unwrap_or_default(),
        }
    }

    /// Inserts `triples`, held by this dataset already, into `graph`, which
    /// a named graph that does not exist comes into being for.
    fn insert_triples(
        &self,
        graph: Option<&str>,
        triples: Vec<Arc<Triple>>,
        collected: &mut Collected,
    ) {
        let key: GraphKey = graph.map(Arc::from);
        let target = graph.map(|name| self.partition_or_create(name, collected));
        let target = target.as_deref().unwrap_or(self);
        target.insert_stored_unguarded(triples.into_iter(), |triple, inserted| {
            if inserted && let Some(collected) = collected.as_mut() {
                collected.push(OverlayChange::Insert {
                    graph: key.clone(),
                    ids: *triple.raw_ids(),
                    triple: Arc::clone(triple),
                });
            }
        });
    }

    /// The default graph's triples and indexes, moved into a partition of
    /// their own (map moves, nothing copied), with the memory they hold; the
    /// default graph keeps empty maps, and its ring and statistics go.
    fn take_default_contents(&self) -> RdfStore {
        let mut before = RdfStore::with_config(self.named_graph_config());
        *before.triples.get_mut() = std::mem::take(&mut *self.triples.write());
        *before.subject_index.get_mut() = std::mem::take(&mut *self.subject_index.write());
        *before.predicate_index.get_mut() = std::mem::take(&mut *self.predicate_index.write());
        *before.object_index.get_mut() = std::mem::replace(
            &mut *self.object_index.write(),
            self.config.index_objects.then(Index::default),
        );
        *before.sp_index.get_mut() = std::mem::take(&mut *self.sp_index.write());
        *before.po_index.get_mut() = std::mem::take(&mut *self.po_index.write());
        *before.os_index.get_mut() = std::mem::take(&mut *self.os_index.write());
        before.dictionary = Arc::clone(&self.dictionary);
        before.memory = Arc::clone(&self.memory);
        before.owns_dictionary = false;
        before.apply_guard = Arc::clone(&self.apply_guard);
        before.overlay = Arc::clone(&self.overlay);
        before.key = None;
        // The count and the grant go with the maps they cover.
        let moved = self.counted.swap(0, std::sync::atomic::Ordering::Relaxed);
        before
            .counted
            .store(moved, std::sync::atomic::Ordering::Relaxed);
        *before.grant.get_mut() = self.grant.lock().take();
        self.counted
            .store(self.own_walk_bytes(), std::sync::atomic::Ordering::Relaxed);
        self.note_changes(usize::MAX);
        self.settle();
        before.settle();
        before
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::rdf::{Term, TriplePattern};

    const G: &str = "http://ex.org/g";
    const H: &str = "http://ex.org/h";

    fn named(name: &str) -> Triple {
        Triple::new(
            Term::iri(format!("http://ex.org/{name}")),
            Term::iri("http://ex.org/name"),
            Term::literal(name),
        )
    }

    fn graph(name: Option<&str>) -> GraphKey {
        name.map(Arc::from)
    }

    fn insert(into: Option<&str>, name: &str) -> CommitChange {
        CommitChange::Insert {
            graph: graph(into),
            triple: named(name),
        }
    }

    fn delete(from: Option<&str>, name: &str) -> CommitChange {
        CommitChange::Delete {
            graph: graph(from),
            triple: named(name),
        }
    }

    fn op(op: RdfGraphOp) -> CommitChange {
        CommitChange::Graph(op)
    }

    fn epoch(epoch: u64) -> EpochId {
        EpochId::new(epoch)
    }

    /// The names in `graph` as `view` sees them, sorted.
    fn names(store: &RdfStore, view: &RdfView, graph: Option<&str>) -> Vec<String> {
        let triples = match graph {
            None => store.find(view, &TriplePattern::any()),
            Some(name) => store
                .graph_at(view, name)
                .map(|partition| partition.find(view, &TriplePattern::any()))
                .unwrap_or_default(),
        };
        let mut names: Vec<String> = triples
            .iter()
            .map(|triple| match triple.object() {
                Term::Literal(literal) => literal.value().to_string(),
                other => other.to_string(),
            })
            .collect();
        names.sort();
        names
    }

    fn latest() -> RdfView {
        RdfView::Latest
    }

    /// A store at epoch 1: Alix and Gus in the default graph, Vincent and
    /// Jules in g.
    fn seeded() -> RdfStore {
        let store = RdfStore::new();
        store.apply_commit(
            ApplyMode::Recovery,
            vec![
                insert(None, "Alix"),
                insert(None, "Gus"),
                insert(Some(G), "Vincent"),
                insert(Some(G), "Jules"),
            ],
        );
        store
    }

    /// What the entries after `after` hold, as (graph, kind, name or graph
    /// change) in order.
    fn entries(store: &RdfStore, after: u64) -> Vec<Vec<String>> {
        store
            .overlay()
            .since(epoch(after))
            .iter()
            .map(|entry| {
                entry
                    .changes()
                    .iter()
                    .map(|change| match change {
                        OverlayChange::Insert { graph, triple, .. } => {
                            format!("{} +{}", key(graph), triple.subject())
                        }
                        OverlayChange::Delete { graph, triple, .. } => {
                            format!("{} -{}", key(graph), triple.subject())
                        }
                        OverlayChange::Graph { graph, change } => format!(
                            "{} {}",
                            key(graph),
                            match change {
                                GraphChange::Created => "created",
                                GraphChange::Dropped { .. } => "dropped",
                                GraphChange::Cleared { .. } => "cleared",
                            }
                        ),
                    })
                    .collect()
            })
            .collect()
    }

    fn key(graph: &GraphKey) -> &str {
        graph.as_deref().unwrap_or("default")
    }

    #[test]
    fn a_commit_writing_two_graphs_publishes_one_entry_in_recorded_order() {
        let store = seeded();
        let applied = store.apply_commit(
            ApplyMode::Publish(epoch(2)),
            vec![
                insert(Some(G), "Mia"),
                insert(None, "Butch"),
                delete(Some(G), "Vincent"),
                insert(Some(G), "Django"),
            ],
        );
        assert_eq!(applied.changed, [true, true, true, true]);
        assert_eq!(
            entries(&store, 1),
            [[
                format!("{G} +<http://ex.org/Mia>"),
                "default +<http://ex.org/Butch>".to_string(),
                format!("{G} -<http://ex.org/Vincent>"),
                format!("{G} +<http://ex.org/Django>"),
            ]]
        );
        assert_eq!(store.overlay().len(), 1);
    }

    /// A no-op in the entry would undo a triple another commit inserted for
    /// a later reader: an insert of a triple the graph holds and a delete of
    /// one it does not are left out.
    #[test]
    fn changes_that_change_nothing_are_not_in_the_entry() {
        let store = seeded();
        let applied = store.apply_commit(
            ApplyMode::Publish(epoch(2)),
            vec![
                insert(None, "Alix"),
                delete(None, "Shosanna"),
                delete(Some(H), "Alix"),
                op(RdfGraphOp::Create {
                    name: G.to_string(),
                }),
                op(RdfGraphOp::Drop {
                    target: RdfGraphTarget::Named(H.to_string()),
                }),
            ],
        );
        assert_eq!(applied.changed, [false, false, false, false, false]);
        assert!(!store.overlay_holds_entries(), "nothing changed: no entry");
        let view = store.view_at(epoch(1)).unwrap();
        assert_eq!(names(&store, &view, None), ["Alix", "Gus"]);
        assert_eq!(store.graph_names(&latest()), [G]);
    }

    #[test]
    fn a_reader_at_an_earlier_snapshot_reads_the_state_before_the_commit() {
        let store = seeded();
        let view = store.view_at(epoch(1)).unwrap();
        store.apply_commit(
            ApplyMode::Publish(epoch(2)),
            vec![
                delete(None, "Alix"),
                insert(None, "Mia"),
                delete(Some(G), "Jules"),
                insert(Some(G), "Hans"),
            ],
        );
        assert_eq!(names(&store, &view, None), ["Alix", "Gus"]);
        assert_eq!(names(&store, &view, Some(G)), ["Jules", "Vincent"]);
        assert_eq!(names(&store, &latest(), None), ["Gus", "Mia"]);
        assert_eq!(names(&store, &latest(), Some(G)), ["Hans", "Vincent"]);
        let after = store.view_at(epoch(2)).unwrap();
        assert_eq!(names(&store, &after, None), ["Gus", "Mia"]);
    }

    /// A `CLEAR` or `DROP` of the default graph swaps its contents out: a
    /// reader before it still reads them, also after later inserts.
    #[test]
    fn the_swap_of_the_default_graph_retains_its_contents() {
        for target in [RdfGraphTarget::Default, RdfGraphTarget::All] {
            for drop in [true, false] {
                let store = seeded();
                let view = store.view_at(epoch(1)).unwrap();
                let change = if drop {
                    RdfGraphOp::Drop {
                        target: target.clone(),
                    }
                } else {
                    RdfGraphOp::Clear {
                        target: target.clone(),
                    }
                };
                store.apply_commit(ApplyMode::Publish(epoch(2)), vec![op(change)]);
                store.apply_commit(ApplyMode::Publish(epoch(3)), vec![insert(None, "Mia")]);
                let case = format!("{target:?}, drop {drop}");
                assert_eq!(names(&store, &view, None), ["Alix", "Gus"], "{case}");
                assert_eq!(
                    names(&store, &view, Some(G)),
                    ["Jules", "Vincent"],
                    "{case}"
                );
                assert_eq!(store.graph_names(&view), [G], "{case}");
                assert_eq!(names(&store, &latest(), None), ["Mia"], "{case}");
                let at_two = store.view_at(epoch(2)).unwrap();
                assert_eq!(names(&store, &at_two, None), Vec::<String>::new(), "{case}");
                assert!(store.account_mismatches().is_empty(), "{case}");
            }
        }
    }

    #[test]
    fn named_graphs_cleared_dropped_or_created_keep_their_state_for_a_reader() {
        let store = seeded();
        store.apply_commit(ApplyMode::Recovery, vec![insert(Some(H), "Beatrix")]);
        let view = store.view_at(epoch(1)).unwrap();
        let applied = store.apply_commit(
            ApplyMode::Publish(epoch(2)),
            vec![
                op(RdfGraphOp::Clear {
                    target: RdfGraphTarget::Named(G.to_string()),
                }),
                insert(Some(G), "Mia"),
                op(RdfGraphOp::Drop {
                    target: RdfGraphTarget::Named(H.to_string()),
                }),
                insert(Some("http://ex.org/new"), "Django"),
            ],
        );
        assert_eq!(applied.changed, [true, true, true, true]);
        assert_eq!(names(&store, &view, Some(G)), ["Jules", "Vincent"]);
        assert_eq!(names(&store, &view, Some(H)), ["Beatrix"]);
        assert_eq!(
            names(&store, &view, Some("http://ex.org/new")),
            Vec::<String>::new()
        );
        let mut seen = store.graph_names(&view);
        seen.sort();
        assert_eq!(
            seen,
            [G, H],
            "a graph made after the snapshot is not listed"
        );
        assert_eq!(names(&store, &latest(), Some(G)), ["Mia"]);
        let mut now = store.graph_names(&latest());
        now.sort();
        assert_eq!(now, [G, "http://ex.org/new"]);
    }

    #[test]
    fn a_copy_move_or_add_reads_its_source_and_keeps_the_old_target_for_a_reader() {
        let cases = [
            (
                RdfGraphOp::Copy {
                    source: None,
                    target: Some(G.to_string()),
                },
                vec!["Alix", "Gus"],
                vec!["Alix", "Gus"],
            ),
            (
                RdfGraphOp::Move {
                    source: None,
                    target: Some(G.to_string()),
                },
                vec!["Alix", "Gus"],
                vec![],
            ),
            (
                RdfGraphOp::Add {
                    source: None,
                    target: Some(G.to_string()),
                },
                vec!["Alix", "Gus", "Jules", "Vincent"],
                vec!["Alix", "Gus"],
            ),
            (
                RdfGraphOp::Copy {
                    source: Some(G.to_string()),
                    target: None,
                },
                vec!["Jules", "Vincent"],
                vec!["Jules", "Vincent"],
            ),
        ];
        for (change, target_now, default_now) in cases {
            let store = seeded();
            let view = store.view_at(epoch(1)).unwrap();
            store.apply_commit(ApplyMode::Publish(epoch(2)), vec![op(change.clone())]);
            let case = format!("{change:?}");
            assert_eq!(names(&store, &view, None), ["Alix", "Gus"], "{case}");
            assert_eq!(
                names(&store, &view, Some(G)),
                ["Jules", "Vincent"],
                "{case}"
            );
            let target = match &change {
                RdfGraphOp::Copy { target, .. }
                | RdfGraphOp::Add { target, .. }
                | RdfGraphOp::Move { target, .. } => target.as_deref(),
                _ => unreachable!("the cases copy, move or add"),
            };
            assert_eq!(names(&store, &latest(), target), target_now, "{case}");
            if target.is_some() {
                assert_eq!(names(&store, &latest(), None), default_now, "{case}");
            }
        }
    }

    #[test]
    fn recovery_applies_and_publishes_nothing() {
        let store = seeded();
        let applied = store.apply_commit(
            ApplyMode::Recovery,
            vec![
                delete(None, "Alix"),
                op(RdfGraphOp::Drop {
                    target: RdfGraphTarget::Named(G.to_string()),
                }),
            ],
        );
        assert_eq!(applied.changed, [true, true]);
        assert!(!store.overlay_holds_entries());
        assert_eq!(names(&store, &latest(), None), ["Gus"]);
        assert_eq!(store.graph_names(&latest()), Vec::<String>::new());
    }

    #[test]
    fn pruning_drops_the_entries_no_reader_needs_and_refuses_views_below() {
        let store = seeded();
        store.apply_commit(ApplyMode::Publish(epoch(2)), vec![insert(None, "Mia")]);
        store.apply_commit(ApplyMode::Publish(epoch(3)), vec![insert(None, "Hans")]);
        assert_eq!(store.overlay().len(), 2);
        store.prune_overlay(epoch(2));
        assert_eq!(store.overlay().len(), 1);
        assert_eq!(
            store.view_at(epoch(1)).unwrap_err(),
            SnapshotGone {
                epoch: epoch(1),
                pruned_through: epoch(2),
            }
        );
        let view = store.view_at(epoch(2)).unwrap();
        assert_eq!(names(&store, &view, None), ["Alix", "Gus", "Mia"]);
        store.prune_overlay(epoch(3));
        assert!(!store.overlay_holds_entries());
    }

    #[test]
    fn a_reader_on_the_fast_path_extends_no_view() {
        let store = seeded();
        let view = store.view_at(epoch(1)).unwrap();
        assert_eq!(names(&store, &view, None), ["Alix", "Gus"]);
        assert_eq!(store.overlay().views_extended(), 0);
        store.apply_commit(ApplyMode::Publish(epoch(2)), vec![insert(None, "Mia")]);
        assert_eq!(names(&store, &view, None), ["Alix", "Gus"]);
        assert_eq!(store.overlay().views_extended(), 1);
    }

    /// The default graph and its named graphs share one guard: a reader of
    /// either waits while a commit holds it, then reads its snapshot.
    #[cfg(feature = "testing-statement-injection")]
    #[test]
    fn a_reader_waits_for_the_apply_and_reads_its_snapshot() {
        use grafeo_common::testing::commit_hook::during_next_rdf_apply;
        use std::time::Duration;

        for step in [
            RdfApplyStep::GuardTaken,
            RdfApplyStep::Halfway,
            RdfApplyStep::BeforePublish,
        ] {
            for graph in [None, Some(G)] {
                let store = Arc::new(seeded());
                let view = store.view_at(epoch(1)).unwrap();
                let (sender, receiver) = std::sync::mpsc::channel();
                // A named graph is read through its own partition, taken
                // before the commit: its reads take the dataset's guard too.
                let read_from = match graph {
                    None => Arc::clone(&store),
                    Some(name) => store.graph(name).unwrap(),
                };
                during_next_rdf_apply(step, move || {
                    let reader = std::thread::spawn(move || names(&read_from, &view, None));
                    std::thread::sleep(Duration::from_millis(50));
                    assert!(
                        !reader.is_finished(),
                        "{step:?}: the reader waits for the guard"
                    );
                    sender.send(reader).unwrap();
                });
                store.apply_commit(
                    ApplyMode::Publish(epoch(2)),
                    vec![
                        delete(None, "Alix"),
                        insert(Some(G), "Mia"),
                        delete(Some(G), "Jules"),
                    ],
                );
                let read = receiver.recv().unwrap().join().unwrap();
                let before = if graph.is_none() {
                    ["Alix", "Gus"]
                } else {
                    ["Jules", "Vincent"]
                };
                assert_eq!(read, before, "{step:?}, {graph:?}");
            }
        }
    }
}
