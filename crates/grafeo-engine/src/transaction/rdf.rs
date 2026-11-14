//! RDF writes in a transaction's change set (#414, #718).
//!
//! An RDF update records each triple it inserts or deletes, and each
//! whole-graph operation (`CREATE`, `DROP`, `CLEAR`, `COPY`, `MOVE`, `ADD`),
//! in its transaction's change set, in order, and changes nothing in the
//! store: the commit applies the entries in that order
//! ([`TransactionChanges::apply_rdf`]) and logs them as one group. A
//! rollback, a rollback to a savepoint and a failed statement drop entries
//! with the rest of the set, so they leave nothing in the store, the WAL or
//! the change feed.
//!
//! The transaction's reads see its own writes: [`PendingTriples`] keeps, per
//! graph, what the transaction's graph operations left of it (its base) and
//! whether each triple it wrote is there once it commits, and
//! [`RdfWriter::find`] lays that over the store at the transaction's view.
//!
//! A graph operation claims each graph it changes: a second transaction that
//! changes one of them fails with a write conflict, at its statement while
//! the first is open and at its commit when the first committed after it
//! began. A graph operation and a triple write in the same graph do not
//! conflict.

use grafeo_core::graph::rdf::RdfView;
use std::sync::Arc;

use grafeo_common::change::{Change, ChangeSet, DataOp, RdfGraphOp};
use grafeo_common::storage::log_record::RdfGraphTarget;
#[cfg(any(rdf_language, test))]
use grafeo_common::utils::error::Result;
use grafeo_common::utils::error::{Error, TransactionError};
use grafeo_common::utils::hash::{FxHashMap, FxHashSet};
use grafeo_core::execution::operators::{
    OperatorError, RdfPathPendingGraph, RdfPathReadControl, RdfPathReadOverlay,
};
use grafeo_core::graph::rdf::{RdfStore, Triple, TriplePattern};

use super::{TransactionChanges, TransactionManager};

/// The triples of one graph a transaction wrote: whether each is there once
/// the transaction commits.
type GraphTriples = FxHashMap<Arc<Triple>, bool>;

/// What the transaction's own graph operations left of a graph, under the
/// triples it wrote there since.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
enum PendingBase {
    /// The graph as the transaction's view has it.
    #[default]
    View,
    /// Emptied (`CLEAR`, or `DROP` of the default graph): it exists and holds
    /// only what the transaction inserted since.
    Emptied,
    /// Created by the transaction: it exists, empty but for its inserts.
    Created,
    /// Dropped: absent, unless the transaction inserted into it since.
    Dropped,
}

/// One graph as the transaction's own writes leave it.
#[derive(Debug, Default)]
struct PendingGraph {
    /// What its graph operations left of it.
    base: PendingBase,
    /// Its triple writes since: whether each triple is there.
    triples: GraphTriples,
    /// Whether it inserted a triple since its last graph operation: the
    /// commit creates the graph, even when a later delete leaves it empty.
    inserted: bool,
}

impl PendingGraph {
    /// The graph after an operation that left it `base`.
    fn based(base: PendingBase) -> Self {
        Self {
            base,
            ..Self::default()
        }
    }

    /// Whether the transaction's committed triples are hidden: its graph
    /// operations replaced them.
    fn hides_store(&self) -> bool {
        self.base != PendingBase::View
    }

    /// Whether `triple` is there for the transaction, when its own writes
    /// decide: its last write of the triple, or a base that hides the store.
    fn state(&self, triple: &Triple) -> Option<bool> {
        self.triples
            .get(triple)
            .copied()
            .or_else(|| self.hides_store().then_some(false))
    }

    /// Whether the graph exists for the transaction, when its own writes
    /// decide; `None` leaves it to the view.
    fn exists(&self) -> Option<bool> {
        match self.base {
            PendingBase::Created | PendingBase::Emptied => Some(true),
            PendingBase::Dropped => Some(self.inserted),
            PendingBase::View => self.inserted.then_some(true),
        }
    }

    /// Whether nothing of the transaction is in the graph.
    fn is_untouched(&self) -> bool {
        self.base == PendingBase::View && self.triples.is_empty()
    }
}

/// One graph's existing transaction-owned net, borrowed only while its
/// change-set lock is held. Reading it makes no copy of pending triples.
struct PathPendingGraph<'a> {
    graph: Option<&'a PendingGraph>,
}

impl RdfPathPendingGraph for PathPendingGraph<'_> {
    fn state(&self, triple: &Triple) -> Option<bool> {
        self.graph?.triples.get(triple).copied()
    }

    fn hides_store(&self) -> bool {
        self.graph.is_some_and(PendingGraph::hides_store)
    }

    fn visit_present(
        &self,
        pattern: &TriplePattern,
        control: &mut RdfPathReadControl<'_>,
        visit: &mut dyn FnMut(
            &Triple,
            &mut RdfPathReadControl<'_>,
        ) -> std::result::Result<(), OperatorError>,
    ) -> std::result::Result<(), OperatorError> {
        if let Some(graph) = self.graph {
            for (triple, &present) in &graph.triples {
                control.poll()?;
                if present && pattern.matches(triple) {
                    visit(triple, control)?;
                }
            }
        }
        Ok(())
    }
}

/// Query-owned adapter; the pending map stays owned by the transaction.
struct PathReadOverlay {
    changes: Arc<TransactionChanges>,
}

impl RdfPathReadOverlay for PathReadOverlay {
    fn with_graph(
        &self,
        graph: Option<&str>,
        control: &mut RdfPathReadControl<'_>,
        read: &mut dyn FnMut(
            &dyn RdfPathPendingGraph,
            &mut RdfPathReadControl<'_>,
        ) -> std::result::Result<(), OperatorError>,
    ) -> std::result::Result<(), OperatorError> {
        self.changes
            .with_path_pending(control, |pending, _, control| {
                read(&pending.path_graph(graph), control)
            })
    }

    fn visit_graph_names(
        &self,
        control: &mut RdfPathReadControl<'_>,
        visit: &mut dyn FnMut(
            &str,
            bool,
            &mut RdfPathReadControl<'_>,
        ) -> std::result::Result<(), OperatorError>,
    ) -> std::result::Result<(), OperatorError> {
        // The pending state is recomputed on a rollback, so it holds only what
        // survives.
        self.changes
            .with_path_pending(control, |pending, _, control| {
                for (name, graph) in &pending.named {
                    control.poll()?;
                    if let Some(exists) = graph.exists() {
                        visit(name, exists, control)?;
                    }
                }
                Ok(())
            })
    }

    fn retained_bytes(&self) -> usize {
        std::mem::size_of::<Self>().saturating_add(2 * std::mem::size_of::<usize>())
    }
}

/// The net effect of a transaction's RDF entries, per graph: what its graph
/// operations left of each graph, and for each triple it inserted or deleted
/// since, whether the triple is there once it commits (its last write of the
/// triple decides).
#[derive(Debug, Default)]
pub(crate) struct PendingTriples {
    /// The default graph's.
    default: PendingGraph,
    /// Each named graph's, by name.
    named: FxHashMap<String, PendingGraph>,
    /// How many graph operations the transaction recorded: one of a
    /// statement's own transaction (`COPY`, `MOVE` or `ADD` applied whole at
    /// its commit) leaves no base, but the commit still has it to apply.
    graph_ops: usize,
}

impl PendingTriples {
    /// Borrows one graph's net for a native path lookup.
    fn path_graph(&self, graph: Option<&str>) -> PathPendingGraph<'_> {
        PathPendingGraph {
            graph: self.graph(graph),
        }
    }

    /// The net effect of the RDF entries of `set`.
    pub(crate) fn of(set: &ChangeSet) -> Self {
        let mut pending = Self::default();
        for change in set.entries() {
            match change {
                Change::Data { graph, op, .. } => {
                    let (triple, present) = match op {
                        DataOp::InsertTriple { triple } => (triple, true),
                        DataOp::DeleteTriple { triple } => (triple, false),
                        _ => continue,
                    };
                    let key = set.graph(*graph).and_then(|graph| graph.key.as_deref());
                    pending.note(key, Arc::new(Triple::from(&**triple)), present);
                }
                Change::RdfGraph { op, .. } => pending.note_graph_op(op),
                Change::Bulk(_) => {}
            }
        }
        pending
    }

    /// The graph `graph` (`None` for the default graph), if the transaction
    /// touched it.
    fn graph(&self, graph: Option<&str>) -> Option<&PendingGraph> {
        match graph {
            None => Some(&self.default),
            Some(name) => self.named.get(name),
        }
    }

    /// Notes a write of `triple` in `graph` (`None` for the default graph):
    /// the triple is there after it (`present`) or not.
    pub(crate) fn note(&mut self, graph: Option<&str>, triple: Arc<Triple>, present: bool) {
        let graph = match graph {
            None => &mut self.default,
            Some(name) => self.named.entry(name.to_string()).or_default(),
        };
        graph.inserted |= present;
        graph.triples.insert(triple, present);
    }

    /// Notes a graph operation: the graph it changes starts over from the
    /// base it leaves, and the triple writes before it in that graph are
    /// gone. An operation on several graphs, and a `COPY`, `MOVE` or `ADD`,
    /// is recorded expanded where a read can follow it (see
    /// [`RdfWriter::graph_op`]), so only the count notes it here.
    pub(crate) fn note_graph_op(&mut self, op: &RdfGraphOp) {
        self.graph_ops += 1;
        match op {
            RdfGraphOp::Create { name } => {
                self.named
                    .insert(name.clone(), PendingGraph::based(PendingBase::Created));
            }
            RdfGraphOp::Drop {
                target: RdfGraphTarget::Named(name),
            } => {
                self.named
                    .insert(name.clone(), PendingGraph::based(PendingBase::Dropped));
            }
            RdfGraphOp::Clear {
                target: RdfGraphTarget::Named(name),
            } => {
                let base = match self.named.get(name).map(|graph| graph.base) {
                    Some(PendingBase::Created) => PendingBase::Created,
                    _ => PendingBase::Emptied,
                };
                self.named.insert(name.clone(), PendingGraph::based(base));
            }
            RdfGraphOp::Drop {
                target: RdfGraphTarget::Default,
            }
            | RdfGraphOp::Clear {
                target: RdfGraphTarget::Default,
            } => self.default = PendingGraph::based(PendingBase::Emptied),
            RdfGraphOp::Drop { .. }
            | RdfGraphOp::Clear { .. }
            | RdfGraphOp::Copy { .. }
            | RdfGraphOp::Move { .. }
            | RdfGraphOp::Add { .. } => {}
        }
    }

    /// Whether `triple` is in `graph` once the transaction commits, if the
    /// transaction's own writes decide: its last write of the triple there,
    /// or a graph operation that replaced the graph's committed triples.
    pub(crate) fn state(&self, graph: Option<&str>, triple: &Triple) -> Option<bool> {
        self.graph(graph)?.state(triple)
    }

    /// Whether the transaction wrote no triple and ran no graph operation.
    pub(crate) fn is_empty(&self) -> bool {
        self.default.is_untouched() && self.named.is_empty() && self.graph_ops == 0
    }

    /// Whether the named graph `name` exists for the transaction, when its
    /// own writes decide; `None` leaves it to the view.
    pub(crate) fn graph_exists(&self, name: &str) -> Option<bool> {
        self.named.get(name).and_then(PendingGraph::exists)
    }

    /// The named graphs the transaction sees, from `view`, its view's: without
    /// those it dropped, with those it created or inserted into (an emptied
    /// one too, as its commit creates them).
    pub(crate) fn graph_names(&self, mut view: Vec<String>) -> Vec<String> {
        view.retain(|name| self.graph_exists(name) != Some(false));
        view.extend(
            self.named
                .iter()
                .filter(|(_, graph)| graph.exists() == Some(true))
                .map(|(name, _)| name.clone()),
        );
        view.sort_unstable();
        view.dedup();
        view
    }

    /// The graphs whose committed triples the transaction's graph operations
    /// hide (`None` for the default graph).
    pub(crate) fn hidden_graphs(&self) -> Vec<Option<String>> {
        let default = self.default.hides_store().then_some(None);
        default
            .into_iter()
            .chain(
                self.named
                    .iter()
                    .filter(|(_, graph)| graph.hides_store())
                    .map(|(name, _)| Some(name.clone())),
            )
            .collect()
    }

    /// The triples the transaction wrote that match `pattern`, in the graphs
    /// `graphs` names (as [`RdfStore::find_in_graphs`] reads them), each with
    /// its graph and whether it is there once the transaction commits.
    pub(crate) fn matching(
        &self,
        pattern: &TriplePattern,
        graphs: Option<&[&str]>,
    ) -> Vec<(Option<String>, Arc<Triple>, bool)> {
        let mut found = Vec::new();
        let mut add = |graph: Option<&str>, triples: &GraphTriples| {
            for (triple, &present) in triples {
                if pattern.matches(triple) {
                    found.push((graph.map(ToString::to_string), Arc::clone(triple), present));
                }
            }
        };
        match graphs {
            None => add(None, &self.default.triples),
            Some([]) => {
                for (name, graph) in &self.named {
                    add(Some(name), &graph.triples);
                }
            }
            Some(names) => {
                for name in names {
                    if let Some(graph) = self.named.get(*name) {
                        add(Some(name), &graph.triples);
                    }
                }
            }
        }
        found
    }

    /// The first triple that these writes and `other`'s, two transactions',
    /// both wrote in one graph where either deletes it, with its graph: the
    /// writes conflict there. Two inserts of one triple do not: the state
    /// after both is the same in either order. Probes the larger side with
    /// the smaller.
    pub(crate) fn conflict_with<'a>(
        &'a self,
        other: &'a Self,
    ) -> Option<(Option<&'a str>, &'a Triple)> {
        if let Some(triple) = graph_conflict(&self.default.triples, &other.default.triples) {
            return Some((None, triple));
        }
        let (small, large) = if self.named.len() <= other.named.len() {
            (&self.named, &other.named)
        } else {
            (&other.named, &self.named)
        };
        small.iter().find_map(|(name, graph)| {
            let theirs = large.get(name)?;
            graph_conflict(&graph.triples, &theirs.triples)
                .map(|triple| (Some(name.as_str()), triple))
        })
    }
}

/// The first triple both graphs' writes hold where either deletes it (see
/// [`PendingTriples::conflict_with`]).
fn graph_conflict<'a>(one: &'a GraphTriples, other: &'a GraphTriples) -> Option<&'a Triple> {
    let (small, large) = if one.len() <= other.len() {
        (one, other)
    } else {
        (other, one)
    };
    small
        .iter()
        .find(|(triple, present)| {
            large
                .get(*triple)
                .is_some_and(|theirs| !(**present && *theirs))
        })
        .map(|(triple, _)| &**triple)
}

/// The error of a commit that loses to a transaction that committed after it
/// began and wrote `triple` in `graph` (`None` for the default graph) where
/// one of the two deletes it.
pub(crate) fn triple_conflict(graph: Option<&str>, triple: &Triple) -> Error {
    let graph = graph.map_or_else(
        || "the default graph".to_string(),
        |name| format!("<{name}>"),
    );
    Error::Transaction(TransactionError::WriteConflict(format!(
        "on RDF triple {} {} {} in {graph}: a transaction that committed after this one began \
         wrote it",
        triple.subject(),
        triple.predicate(),
        triple.object()
    )))
}

/// How a statement of a transaction writes RDF: its triples and graph
/// operations go to the transaction's change set, and its reads see them
/// over the store.
#[derive(Clone)]
pub(crate) struct RdfWriter {
    /// The store the commit applies the entries to.
    store: Arc<RdfStore>,
    /// The transaction's changes.
    changes: Arc<TransactionChanges>,
    /// The transaction manager: the write freeze around each record, and the
    /// claims of graph operations.
    manager: Arc<TransactionManager>,
    /// Whether a `COPY`, `MOVE` or `ADD` is recorded as what the transaction
    /// read (the target's swap and one insert per source triple): in a
    /// transaction later statements read. A statement's own transaction
    /// records it whole, applied at its commit.
    expands_copies: bool,
    /// The view the transaction reads the store at: its snapshot.
    view: RdfView,
}

impl RdfWriter {
    /// This writer, in a transaction later statements read: a `COPY`, `MOVE`
    /// or `ADD` is recorded as what the transaction read.
    #[cfg(rdf_language)]
    pub(crate) fn expanding_copies(mut self) -> Self {
        self.expands_copies = true;
        self
    }

    /// Borrowed, bounded pending reader for the native path operator. The
    /// adapter retains no store or manager and allocates no pending snapshot.
    pub(crate) fn path_overlay(&self) -> Arc<dyn RdfPathReadOverlay> {
        Arc::new(PathReadOverlay {
            changes: Arc::clone(&self.changes),
        })
    }

    /// The writer of the transaction whose changes are `changes`, reading
    /// `store` at the transaction's snapshot. Only an RDF query writes
    /// through one.
    ///
    /// # Errors
    ///
    /// When the store has no view at the snapshot any more (see
    /// [`TransactionChanges::rdf_view`]).
    #[cfg(any(rdf_language, test))]
    pub(crate) fn new(
        store: Arc<RdfStore>,
        changes: Arc<TransactionChanges>,
        manager: Arc<TransactionManager>,
    ) -> Result<Self> {
        let view = changes.rdf_view(&store)?;
        Ok(Self {
            store,
            changes,
            manager,
            expands_copies: false,
            view,
        })
    }

    /// The writer of a private transaction outside any database (see
    /// [`update_privately`]): its manager's epochs are not the store's, so
    /// it reads the newest committed state.
    #[cfg(any(rdf_language, test))]
    pub(crate) fn outside_any_database(
        store: Arc<RdfStore>,
        changes: Arc<TransactionChanges>,
        manager: Arc<TransactionManager>,
    ) -> Self {
        Self {
            store,
            changes,
            manager,
            expands_copies: false,
            view: RdfView::latest(grafeo_core::graph::rdf::LatestUse::Test),
        }
    }

    /// The view the transaction reads the store at.
    #[cfg(any(rdf_language, test))]
    pub(crate) fn view(&self) -> &RdfView {
        &self.view
    }

    /// Records the insert of `triple` into `graph` (`None` for the default
    /// graph). Returns whether the triple is new, as the transaction sees
    /// the graph; an insert of a triple it sees there records nothing.
    ///
    /// # Errors
    ///
    /// When the change set refuses the entry (a broken invariant).
    pub(crate) fn insert(
        &self,
        graph: Option<&str>,
        triple: Triple,
    ) -> std::result::Result<bool, OperatorError> {
        if self.sees(graph, &triple) {
            return Ok(false);
        }
        self.record(graph, triple, true)?;
        Ok(true)
    }

    /// Records the delete of `triple` from `graph` (`None` for the default
    /// graph). Returns whether the transaction sees the triple there; a
    /// delete of one it does not see records nothing.
    ///
    /// # Errors
    ///
    /// When the change set refuses the entry (a broken invariant).
    pub(crate) fn delete(
        &self,
        graph: Option<&str>,
        triple: &Triple,
    ) -> std::result::Result<bool, OperatorError> {
        if !self.sees(graph, triple) {
            return Ok(false);
        }
        self.record(graph, triple.clone(), false)?;
        Ok(true)
    }

    /// Records a write of `triple` in `graph`, in the write freeze: a
    /// checkpoint never reads the change sets halfway through it.
    fn record(
        &self,
        graph: Option<&str>,
        triple: Triple,
        insert: bool,
    ) -> std::result::Result<(), OperatorError> {
        let _writing = self.manager.write_in_progress();
        self.changes
            .record_triple(graph, triple, insert)
            .map_err(OperatorError::from)
    }

    /// Whether the transaction sees `triple` in `graph`: its own writes
    /// decide (its last write of the triple, or a graph operation that
    /// replaced the graph's committed triples), the store otherwise.
    fn sees(&self, graph: Option<&str>, triple: &Triple) -> bool {
        self.changes
            .pending_triple(graph, triple)
            .unwrap_or_else(|| holds(&self.store, &self.view, graph, triple))
    }

    /// Whether the transaction wrote a triple or ran a graph operation that
    /// it has not committed.
    pub(crate) fn has_pending(&self) -> bool {
        self.changes.has_pending_triples()
    }

    /// Whether the named graph `name` exists for the transaction: as its own
    /// graph operations and inserts leave it, the view otherwise.
    pub(crate) fn graph_exists(&self, name: &str) -> bool {
        self.changes
            .read_pending(|pending| pending.graph_exists(name))
            .unwrap_or_else(|| self.store.graph_at(&self.view, name).is_some())
    }

    /// The named graphs the transaction sees, sorted: the view's, without
    /// those it dropped, with those it created or inserted into.
    pub(crate) fn graph_names(&self) -> Vec<String> {
        let view = self.store.graph_names(&self.view);
        self.changes
            .read_pending(|pending| pending.graph_names(view))
    }

    /// The triples matching `pattern` in the graphs `graphs` names (as
    /// [`RdfStore::find_in_graphs`] reads them), each with its graph, as the
    /// transaction sees them: what the store holds in the graphs its graph
    /// operations left as they were, without the triples it deleted, with
    /// those it inserted.
    pub(crate) fn find(
        &self,
        pattern: &TriplePattern,
        graphs: Option<&[&str]>,
    ) -> Vec<(Option<String>, Arc<Triple>)> {
        let mut found = self.store.find_in_graphs(&self.view, pattern, graphs);
        let (pending, hidden) = self.changes.read_pending(|pending| {
            if pending.is_empty() {
                return (Vec::new(), Vec::new());
            }
            (pending.matching(pattern, graphs), pending.hidden_graphs())
        });
        if pending.is_empty() && hidden.is_empty() {
            return found;
        }
        {
            let deleted: FxHashSet<(Option<&str>, &Triple)> = pending
                .iter()
                .filter(|(_, _, present)| !present)
                .map(|(graph, triple, _)| (graph.as_deref(), triple.as_ref()))
                .collect();
            if !deleted.is_empty() || !hidden.is_empty() {
                found.retain(|(graph, triple)| {
                    !hidden.contains(graph)
                        && !deleted.contains(&(graph.as_deref(), triple.as_ref()))
                });
            }
        }
        for (graph, triple, present) in pending {
            // A triple the store holds already is in `found`, unless its
            // graph's committed triples are hidden.
            if present
                && (hidden.contains(&graph)
                    || !holds(&self.store, &self.view, graph.as_deref(), &triple))
            {
                found.push((graph, triple));
            }
        }
        found
    }

    /// The triples of `graph` (`None` for the default graph) as the
    /// transaction sees them.
    fn graph_triples(&self, graph: Option<&str>) -> Vec<Arc<Triple>> {
        let names = graph.map(|name| [name]);
        self.find(
            &TriplePattern::any(),
            names.as_ref().map(<[&str; 1]>::as_slice),
        )
        .into_iter()
        .map(|(_, triple)| triple)
        .collect()
    }

    /// Records the whole-graph operation `op` in the transaction, once
    /// `check` passed (whether it applies: `false` skips it, as `SILENT` does
    /// for a graph that is not there). Each graph it changes is claimed
    /// first: another open transaction that changes one fails it. An
    /// operation on `NAMED` or `ALL` is recorded per graph the transaction
    /// sees; with [`expanding_copies`](Self::expanding_copies), a `COPY`,
    /// `MOVE` or `ADD` as the target's swap, one insert per source triple the
    /// transaction sees and, for a `MOVE`, the source's drop or clear. A copy,
    /// move or add of a graph onto itself changes nothing (SPARQL 1.1 Update,
    /// sections 3.2.3 and 3.2.4): once its check passed, it records nothing.
    ///
    /// # Errors
    ///
    /// A write conflict naming the graph (`GRAFEO-T001`); the error of
    /// `check`; the change set's refusal (a memory limit, a broken
    /// invariant).
    pub(crate) fn graph_op(
        &self,
        op: RdfGraphOp,
        check: impl FnOnce(&Self) -> std::result::Result<bool, OperatorError>,
    ) -> std::result::Result<(), OperatorError> {
        if !check(self)? || onto_itself(&op) {
            return Ok(());
        }
        let (drops, adds, moves) = (
            matches!(op, RdfGraphOp::Drop { .. }),
            matches!(op, RdfGraphOp::Add { .. }),
            matches!(op, RdfGraphOp::Move { .. }),
        );
        match op {
            RdfGraphOp::Drop { target } | RdfGraphOp::Clear { target }
                if matches!(target, RdfGraphTarget::AllNamed | RdfGraphTarget::All) =>
            {
                // As the store applies them: `CLEAR` of `NAMED` or `ALL` takes
                // the named graphs away, as `DROP` does.
                if matches!(target, RdfGraphTarget::All) {
                    self.record_graph_op(if drops {
                        RdfGraphOp::Drop {
                            target: RdfGraphTarget::Default,
                        }
                    } else {
                        RdfGraphOp::Clear {
                            target: RdfGraphTarget::Default,
                        }
                    })?;
                }
                for name in self.graph_names() {
                    self.record_graph_op(RdfGraphOp::Drop {
                        target: RdfGraphTarget::Named(name),
                    })?;
                }
                Ok(())
            }
            RdfGraphOp::Copy { source, target }
            | RdfGraphOp::Move { source, target }
            | RdfGraphOp::Add { source, target }
                if self.expands_copies =>
            {
                let triples = self.graph_triples(source.as_deref());
                if !adds {
                    self.record_graph_op(match &target {
                        None => RdfGraphOp::Clear {
                            target: RdfGraphTarget::Default,
                        },
                        Some(name) if self.graph_exists(name) => RdfGraphOp::Clear {
                            target: RdfGraphTarget::Named(name.clone()),
                        },
                        Some(name) => RdfGraphOp::Create { name: name.clone() },
                    })?;
                }
                for triple in triples {
                    self.insert(target.as_deref(), Triple::clone(&triple))?;
                }
                if moves {
                    self.record_graph_op(match source {
                        None => RdfGraphOp::Clear {
                            target: RdfGraphTarget::Default,
                        },
                        Some(name) => RdfGraphOp::Drop {
                            target: RdfGraphTarget::Named(name),
                        },
                    })?;
                }
                Ok(())
            }
            op => self.record_graph_op(op),
        }
    }

    /// Claims the graphs `op` changes, then records it, in the write freeze.
    fn record_graph_op(&self, op: RdfGraphOp) -> std::result::Result<(), OperatorError> {
        for claim in crate::database::standalone::rdf_graph_claims(&op) {
            self.manager
                .record_key(self.changes.id(), claim)
                .map_err(OperatorError::from)?;
        }
        let _writing = self.manager.write_in_progress();
        self.changes
            .record_rdf_graph(op)
            .map_err(OperatorError::from)
    }
}

/// Runs `body`, an RDF update outside a session (a query processor of its
/// own, a test), in a private transaction of `manager`: the triples it
/// records apply to `store` once it succeeds, and none of them when it
/// fails. Nothing is logged or reported to change data capture. `manager`
/// keeps no epochs of the store, so the update reads the newest committed
/// state and publishes nothing for readers at a snapshot (as a write through
/// the store itself).
///
/// # Errors
///
/// The error of `body`; the error of beginning or committing the
/// transaction (a closed database, a commit that did not complete).
#[cfg(any(rdf_language, test))]
pub(crate) fn update_privately<T>(
    store: &Arc<RdfStore>,
    manager: &Arc<TransactionManager>,
    body: impl FnOnce(RdfWriter) -> Result<T>,
) -> Result<T> {
    let (transaction, changes) = manager.begin_private()?;
    let writer = RdfWriter::outside_any_database(
        Arc::clone(store),
        Arc::clone(&changes),
        Arc::clone(manager),
    );
    let value = match body(writer) {
        Ok(value) => value,
        Err(error) => {
            let _ = manager.abort(transaction);
            return Err(error);
        }
    };
    let commit = match manager.start_commit(transaction) {
        Ok(commit) => commit,
        Err(error) => {
            let _ = manager.abort(transaction);
            return Err(error);
        }
    };
    // A set that fails to apply drops the guard uncompleted, which poisons
    // the manager.
    changes.apply_rdf(store, grafeo_core::graph::rdf::ApplyMode::Unversioned)?;
    commit.complete()?;
    Ok(value)
}

/// Drops the overlay entries of `store` that no reader of `manager` needs,
/// after a commit published one: those at or below the read horizon. A
/// store with none pays one atomic load.
pub(crate) fn prune_rdf_overlay(store: &RdfStore, manager: &TransactionManager) {
    if store.overlay_holds_entries() {
        store.prune_overlay(manager.read_horizon());
    }
}

/// The N-Triples strings of a triple's terms, as a change event holds them.
#[cfg(feature = "cdc")]
pub(crate) fn ntriples_terms(
    triple: &grafeo_common::storage::log_record::TripleRecord,
) -> (String, String, String) {
    use grafeo_core::graph::rdf::Term;

    (
        Term::from(&triple.subject).to_string(),
        Term::from(&triple.predicate).to_string(),
        Term::from(&triple.object).to_string(),
    )
}

/// Whether `view` sees `triple` in `graph` (`None` for the default graph).
fn holds(store: &RdfStore, view: &RdfView, graph: Option<&str>, triple: &Triple) -> bool {
    match graph {
        None => store.contains(view, triple),
        Some(name) => store
            .graph_at(view, name)
            .is_some_and(|graph| graph.contains(view, triple)),
    }
}

/// Whether `op` is a copy, move or add of a graph onto itself.
fn onto_itself(op: &RdfGraphOp) -> bool {
    match op {
        RdfGraphOp::Copy { source, target }
        | RdfGraphOp::Move { source, target }
        | RdfGraphOp::Add { source, target } => source == target,
        RdfGraphOp::Create { .. } | RdfGraphOp::Drop { .. } | RdfGraphOp::Clear { .. } => false,
    }
}

#[cfg(test)]
mod tests {
    use grafeo_core::graph::rdf::Term;

    use super::*;

    fn triple(subject: &str) -> Arc<Triple> {
        Arc::new(Triple::new(
            Term::iri(format!("http://example.org/{subject}")),
            Term::iri("http://example.org/knows"),
            Term::iri("http://example.org/gus"),
        ))
    }

    /// The last write of a triple decides, per graph; a graph scope reads as
    /// `RdfStore::find_in_graphs` does.
    #[test]
    fn pending_triples_keep_the_last_write_per_graph() {
        let paris = "http://example.org/paris";
        let mut pending = PendingTriples::default();
        assert!(pending.is_empty());
        pending.note(None, triple("alix"), true);
        pending.note(Some(paris), triple("alix"), true);
        pending.note(None, triple("alix"), false);
        pending.note(Some(paris), triple("mia"), false);

        assert_eq!(pending.state(None, &triple("alix")), Some(false));
        assert_eq!(pending.state(Some(paris), &triple("alix")), Some(true));
        assert_eq!(pending.state(Some(paris), &triple("gus")), None);
        assert_eq!(
            pending.state(Some("http://example.org/berlin"), &triple("alix")),
            None
        );

        let any = TriplePattern::any();
        let graphs = |graphs: Option<&[&str]>| {
            let mut found: Vec<(Option<String>, String, bool)> = pending
                .matching(&any, graphs)
                .into_iter()
                .map(|(graph, triple, present)| (graph, triple.subject().to_string(), present))
                .collect();
            found.sort();
            found
        };
        let alix = "<http://example.org/alix>".to_string();
        let mia = "<http://example.org/mia>".to_string();
        assert_eq!(graphs(None), [(None, alix.clone(), false)]);
        let named = vec![
            (Some(paris.to_string()), alix, true),
            (Some(paris.to_string()), mia, false),
        ];
        assert_eq!(graphs(Some(&[])), named);
        assert_eq!(graphs(Some(&[paris])), named);
        assert_eq!(graphs(Some(&["http://example.org/berlin"])), Vec::new());
    }

    fn named(name: &str) -> RdfGraphTarget {
        RdfGraphTarget::Named(name.to_string())
    }

    /// A graph operation starts the graph over from the base it leaves: the
    /// triple writes before it there are gone, its committed triples are
    /// hidden, and the writes after it are what the graph holds.
    #[test]
    fn a_graph_operation_replaces_what_the_transaction_reads_of_its_graph() {
        let (g, h) = ("http://example.org/g", "http://example.org/h");
        let mut pending = PendingTriples::default();
        pending.note(Some(g), triple("alix"), true);
        pending.note_graph_op(&RdfGraphOp::Clear { target: named(g) });
        assert_eq!(pending.state(Some(g), &triple("alix")), Some(false));
        assert_eq!(
            pending.state(Some(g), &triple("gus")),
            Some(false),
            "the committed triples are hidden"
        );
        assert_eq!(
            pending.state(Some(h), &triple("gus")),
            None,
            "h is the view's"
        );
        assert_eq!(
            pending.graph_exists(g),
            Some(true),
            "a cleared graph exists"
        );

        pending.note_graph_op(&RdfGraphOp::Drop { target: named(g) });
        assert_eq!(pending.graph_exists(g), Some(false));
        pending.note(Some(g), triple("mia"), true);
        assert_eq!(
            pending.graph_exists(g),
            Some(true),
            "an insert recreates it"
        );
        assert_eq!(pending.state(Some(g), &triple("mia")), Some(true));

        pending.note_graph_op(&RdfGraphOp::Clear {
            target: RdfGraphTarget::Default,
        });
        assert_eq!(pending.state(None, &triple("vincent")), Some(false));
        assert_eq!(pending.hidden_graphs(), [None, Some(g.to_string())]);
        assert!(!pending.is_empty());
    }

    /// The graphs the transaction sees: the view's without those it dropped,
    /// with those it created, cleared or inserted into; a graph whose only
    /// own write is a delete stays the view's.
    #[test]
    fn the_graph_list_follows_the_transactions_operations() {
        let mut pending = PendingTriples::default();
        pending.note_graph_op(&RdfGraphOp::Drop {
            target: named("berlin"),
        });
        pending.note_graph_op(&RdfGraphOp::Create {
            name: "paris".into(),
        });
        pending.note_graph_op(&RdfGraphOp::Clear {
            target: named("paris"),
        });
        pending.note(Some("rome"), triple("alix"), true);
        pending.note(Some("rome"), triple("alix"), false);
        pending.note(Some("oslo"), triple("gus"), false);
        let view = ["berlin", "oslo", "vienna"].map(String::from).to_vec();
        assert_eq!(
            pending.graph_names(view),
            ["oslo", "paris", "rome", "vienna"],
            "rome: its commit creates it even with its triple deleted again"
        );
        assert_eq!(
            pending.graph_exists("paris"),
            Some(true),
            "a created graph stays created when cleared"
        );
        assert_eq!(pending.graph_exists("oslo"), None);
    }

    /// The count of graph operations keeps a transaction with only a whole
    /// `COPY` (a statement's own) from reading as empty: its commit has it to
    /// apply. A savepoint rollback recomputes the state from the entries.
    #[test]
    fn a_whole_copy_counts_and_the_state_follows_the_entries() {
        let mut pending = PendingTriples::default();
        pending.note_graph_op(&RdfGraphOp::Copy {
            source: None,
            target: Some("paris".into()),
        });
        assert!(!pending.is_empty());
        assert_eq!(pending.graph_exists("paris"), None);

        let mut set = ChangeSet::new();
        let slot = set
            .slot(grafeo_common::change::GraphRef {
                model: grafeo_common::change::DataModel::Rdf,
                key: Some("paris".into()),
            })
            .unwrap();
        set.push_rdf_graph(
            slot,
            RdfGraphOp::Drop {
                target: named("paris"),
            },
        )
        .unwrap();
        let recomputed = PendingTriples::of(&set);
        assert_eq!(recomputed.graph_exists("paris"), Some(false));
        assert!(PendingTriples::of(&ChangeSet::new()).is_empty());
    }
}
