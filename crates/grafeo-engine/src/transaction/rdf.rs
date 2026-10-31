//! RDF writes in a transaction's change set (#414).
//!
//! An RDF update records each triple it inserts or deletes in its
//! transaction's change set, as an entry of the triple's graph, and changes
//! nothing in the store: the commit applies the entries in order
//! ([`TransactionChanges::apply_triples`]), then logs and reports to change
//! data capture the ones that changed the store. A rollback, a rollback to a
//! savepoint and a failed statement drop entries with the rest of the set,
//! so they leave nothing in the store, the WAL or the change feed.
//!
//! The transaction's reads see its own writes: [`PendingTriples`] keeps, per
//! graph, whether each triple it wrote is there once it commits, and
//! [`RdfWriter::find`] lays that over the store.
//!
//! A whole-graph operation (`CREATE`, `DROP`, `CLEAR`, `COPY`, `MOVE`,
//! `ADD`) is a standalone change (see `database::standalone`): it takes
//! effect at once, also inside a transaction, and is refused with a write
//! conflict while an open transaction, the caller's own included, has RDF
//! entries in a graph it changes.

use std::sync::Arc;

use grafeo_common::change::{Change, ChangeSet, DataOp, RdfGraphOp};
use grafeo_common::storage::log_record::RdfGraphTarget;
use grafeo_common::utils::error::{Error, Result, TransactionError};
use grafeo_common::utils::hash::{FxHashMap, FxHashSet};
use grafeo_core::execution::operators::{
    OperatorError, RdfPathPendingGraph, RdfPathReadControl, RdfPathReadOverlay,
};
use grafeo_core::graph::rdf::{RdfStore, Triple, TriplePattern};

use super::{CommitsHeld, TransactionChanges, TransactionManager};

/// The triples of one graph a transaction wrote: whether each is there once
/// the transaction commits.
type GraphTriples = FxHashMap<Arc<Triple>, bool>;

/// One graph's existing transaction-owned net, borrowed only while its
/// change-set lock is held. Reading it makes no copy of pending triples.
struct PathPendingGraph<'a> {
    triples: Option<&'a GraphTriples>,
}

impl RdfPathPendingGraph for PathPendingGraph<'_> {
    fn state(&self, triple: &Triple) -> Option<bool> {
        self.triples?.get(triple).copied()
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
        if let Some(triples) = self.triples {
            for (triple, &present) in triples {
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
            &mut RdfPathReadControl<'_>,
        ) -> std::result::Result<(), OperatorError>,
    ) -> std::result::Result<(), OperatorError> {
        self.changes.with_path_pending(control, |_, set, control| {
            // An insert creates its graph even if a later delete leaves the
            // graph empty. Borrow the surviving records, so rollback removes
            // that existence too; delete-only references create no graph.
            for change in set.entries() {
                control.poll()?;
                let Change::Data {
                    graph,
                    op: DataOp::InsertTriple { .. },
                    ..
                } = change
                else {
                    continue;
                };
                if let Some(name) = set.graph(*graph).and_then(|graph| graph.key.as_deref()) {
                    visit(name, control)?;
                }
            }
            Ok(())
        })
    }

    fn retained_bytes(&self) -> usize {
        std::mem::size_of::<Self>().saturating_add(2 * std::mem::size_of::<usize>())
    }
}

/// The net effect of a transaction's RDF entries, per graph: for each triple
/// it inserted or deleted, whether the triple is there once it commits (its
/// last write of the triple decides).
#[derive(Default)]
pub(crate) struct PendingTriples {
    /// The default graph's.
    default: GraphTriples,
    /// Each named graph's, by name.
    named: FxHashMap<String, GraphTriples>,
}

impl PendingTriples {
    /// Borrows one graph's net for a native path lookup.
    fn path_graph(&self, graph: Option<&str>) -> PathPendingGraph<'_> {
        PathPendingGraph {
            triples: match graph {
                None => Some(&self.default),
                Some(name) => self.named.get(name),
            },
        }
    }

    /// The net effect of the RDF entries of `set`.
    pub(crate) fn of(set: &ChangeSet) -> Self {
        let mut pending = Self::default();
        for change in set.entries() {
            let Change::Data { graph, op, .. } = change else {
                continue;
            };
            let (triple, present) = match op {
                DataOp::InsertTriple { triple } => (triple, true),
                DataOp::DeleteTriple { triple } => (triple, false),
                _ => continue,
            };
            let key = set.graph(*graph).and_then(|graph| graph.key.as_deref());
            pending.note(key, Arc::new(Triple::from(&**triple)), present);
        }
        pending
    }

    /// Notes a write of `triple` in `graph` (`None` for the default graph):
    /// the triple is there after it (`present`) or not.
    pub(crate) fn note(&mut self, graph: Option<&str>, triple: Arc<Triple>, present: bool) {
        let triples = match graph {
            None => &mut self.default,
            Some(name) => self.named.entry(name.to_string()).or_default(),
        };
        triples.insert(triple, present);
    }

    /// Whether `triple` is in `graph` once the transaction commits, if the
    /// transaction wrote it there.
    pub(crate) fn state(&self, graph: Option<&str>, triple: &Triple) -> Option<bool> {
        match graph {
            None => self.default.get(triple).copied(),
            Some(name) => self.named.get(name)?.get(triple).copied(),
        }
    }

    /// Whether the transaction wrote no triple.
    pub(crate) fn is_empty(&self) -> bool {
        self.default.is_empty() && self.named.is_empty()
    }

    /// The first graph the transaction wrote that `graph` takes, as an
    /// error names it: "the default graph", or the name in angle brackets.
    pub(crate) fn written_graph(&self, graph: impl Fn(Option<&str>) -> bool) -> Option<String> {
        if !self.default.is_empty() && graph(None) {
            return Some("the default graph".to_string());
        }
        self.named
            .keys()
            .find(|name| graph(Some(name)))
            .map(|name| format!("<{name}>"))
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
            None => add(None, &self.default),
            Some([]) => {
                for (name, triples) in &self.named {
                    add(Some(name), triples);
                }
            }
            Some(names) => {
                for name in names {
                    if let Some(triples) = self.named.get(*name) {
                        add(Some(name), triples);
                    }
                }
            }
        }
        found
    }
}

/// How a statement of a transaction writes RDF: its triples go to the
/// transaction's change set, and its reads see them over the store.
#[derive(Clone)]
pub(crate) struct RdfWriter {
    /// The store the commit applies the triples to.
    store: Arc<RdfStore>,
    /// The transaction's changes.
    changes: Arc<TransactionChanges>,
    /// The transaction manager: the write freeze around each record, and
    /// the hold of a whole-graph operation.
    manager: Arc<TransactionManager>,
    /// The WAL a whole-graph operation is logged to, when the database logs.
    #[cfg(feature = "wal")]
    wal: Option<Arc<grafeo_storage::wal::Wal>>,
}

impl RdfWriter {
    /// Borrowed, bounded pending reader for the native path operator. The
    /// adapter retains no store or manager and allocates no pending snapshot.
    pub(crate) fn path_overlay(&self) -> Arc<dyn RdfPathReadOverlay> {
        Arc::new(PathReadOverlay {
            changes: Arc::clone(&self.changes),
        })
    }

    /// The writer of the transaction whose changes are `changes`. Only an
    /// RDF query writes through one.
    #[cfg(any(rdf_language, test))]
    pub(crate) fn new(
        store: Arc<RdfStore>,
        changes: Arc<TransactionChanges>,
        manager: Arc<TransactionManager>,
        #[cfg(feature = "wal")] wal: Option<Arc<grafeo_storage::wal::Wal>>,
    ) -> Self {
        Self {
            store,
            changes,
            manager,
            #[cfg(feature = "wal")]
            wal,
        }
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

    /// Whether the transaction sees `triple` in `graph`: its own last write
    /// of it decides, the store otherwise.
    fn sees(&self, graph: Option<&str>, triple: &Triple) -> bool {
        self.changes
            .pending_triple(graph, triple)
            .unwrap_or_else(|| holds(&self.store, graph, triple))
    }

    /// Whether the transaction wrote a triple that it has not committed.
    pub(crate) fn has_pending(&self) -> bool {
        self.changes.has_pending_triples()
    }

    /// The triples matching `pattern` in the graphs `graphs` names (as
    /// [`RdfStore::find_in_graphs`] reads them), each with its graph, as the
    /// transaction sees them: what the store holds, without the triples the
    /// transaction deleted, with those it inserted.
    pub(crate) fn find(
        &self,
        pattern: &TriplePattern,
        graphs: Option<&[&str]>,
    ) -> Vec<(Option<String>, Arc<Triple>)> {
        let mut found = self.store.find_in_graphs(pattern, graphs);
        let pending = self.changes.pending_triples_matching(pattern, graphs);
        if pending.is_empty() {
            return found;
        }
        {
            let deleted: FxHashSet<(Option<&str>, &Triple)> = pending
                .iter()
                .filter(|(_, _, present)| !present)
                .map(|(graph, triple, _)| (graph.as_deref(), triple.as_ref()))
                .collect();
            if !deleted.is_empty() {
                found.retain(|(graph, triple)| {
                    !deleted.contains(&(graph.as_deref(), triple.as_ref()))
                });
            }
        }
        for (graph, triple, present) in pending {
            // A triple the store holds already is in `found`.
            if present && !holds(&self.store, graph.as_deref(), &triple) {
                found.push((graph, triple));
            }
        }
        found
    }

    /// Runs the whole-graph operation `op` as a standalone change: holds
    /// commits and the writes of open transactions off, refuses it while an
    /// open transaction has RDF entries in a graph it changes, runs `check`
    /// (whether the operation applies: `false` skips it, as `SILENT` does
    /// for a graph that is not there), then logs it as a WAL group of its
    /// own and applies it.
    ///
    /// # Errors
    ///
    /// A write conflict naming the graph (`GRAFEO-T001`); the error of
    /// `check`; the error of the hold (the database is closed, or a commit did
    /// not complete) or of writing the group, which applies nothing. Each
    /// keeps its code.
    pub(crate) fn graph_op(
        &self,
        op: RdfGraphOp,
        check: impl FnOnce(&RdfStore) -> std::result::Result<bool, OperatorError>,
    ) -> std::result::Result<(), OperatorError> {
        // A copy, move or add of a graph onto itself changes nothing (SPARQL
        // 1.1 Update, sections 3.2.3 and 3.2.4): once its source check
        // passed, it is not held, checked against open changes or logged.
        if onto_itself(&op) {
            check(&self.store)?;
            return Ok(());
        }
        let held = crate::database::standalone::hold(&self.manager, true)?;
        refuse_graph_op_with_open_changes(&self.manager, &held, &op)?;
        if !check(&self.store)? {
            return Ok(());
        }
        crate::database::standalone::commit_rdf(
            op,
            &held,
            #[cfg(feature = "wal")]
            self.wal.as_deref(),
            &self.store,
            &self.manager,
        )
        .map_err(OperatorError::from)
    }
}

/// Runs `body`, an RDF update outside a session (a query processor of its
/// own, a test), in a private transaction of `manager`: the triples it
/// records apply to `store` once it succeeds, and none of them when it
/// fails. Nothing is logged or reported to change data capture.
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
    let writer = RdfWriter::new(
        Arc::clone(store),
        Arc::clone(&changes),
        Arc::clone(manager),
        #[cfg(feature = "wal")]
        None,
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
    changes.apply_triples(store)?;
    commit.complete()?;
    Ok(value)
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

/// Whether the store holds `triple` in `graph` (`None` for the default
/// graph).
fn holds(store: &RdfStore, graph: Option<&str>, triple: &Triple) -> bool {
    match graph {
        None => store.contains(triple),
        Some(name) => store
            .graph(name)
            .is_some_and(|graph| graph.contains(triple)),
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

/// Whether the whole-graph operation `op` changes `graph` (`None` for the
/// default graph): the target of each operation, and the source of a move.
/// A create changes no triple an open transaction wrote.
fn changes_graph(op: &RdfGraphOp, graph: Option<&str>) -> bool {
    let named = |target: &Option<String>| target.as_deref() == graph;
    match op {
        RdfGraphOp::Create { .. } => false,
        RdfGraphOp::Drop { target } | RdfGraphOp::Clear { target } => match target {
            RdfGraphTarget::Default => graph.is_none(),
            RdfGraphTarget::Named(name) => graph == Some(name.as_str()),
            RdfGraphTarget::AllNamed => graph.is_some(),
            RdfGraphTarget::All => true,
        },
        RdfGraphOp::Copy { target, .. } | RdfGraphOp::Add { target, .. } => named(target),
        RdfGraphOp::Move { source, target } => named(source) || named(target),
    }
}

/// Refuses the whole-graph operation `op` while an open transaction (the
/// caller's own too) has RDF entries in a graph it changes: they would apply
/// at that transaction's commit, after the operation. The caller holds
/// commits off and the writes of open transactions out (`held`), so no
/// transaction records an entry in the graph between this check and the
/// operation.
///
/// # Errors
///
/// A write conflict naming the graph.
fn refuse_graph_op_with_open_changes(
    manager: &TransactionManager,
    held: &CommitsHeld<'_>,
    op: &RdfGraphOp,
) -> Result<()> {
    let written = manager
        .open_change_sets(held)
        .iter()
        .find_map(|changes| changes.written_rdf_graph(|graph| changes_graph(op, graph)));
    match written {
        None => Ok(()),
        Some(name) => Err(Error::Transaction(TransactionError::WriteConflict(
            format!(
                "RDF graph {name} has changes of an open transaction: run the graph operation \
                 once that transaction commits or rolls back"
            ),
        ))),
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

        assert_eq!(
            pending.written_graph(|graph| graph.is_none()).as_deref(),
            Some("the default graph")
        );
        assert_eq!(
            pending.written_graph(|graph| graph.is_some()),
            Some(format!("<{paris}>"))
        );
        assert_eq!(pending.written_graph(|graph| graph == Some("x")), None);
    }

    /// The graphs an operation changes, for the check against open
    /// transactions: every target, and a move's source.
    #[test]
    fn a_graph_operation_changes_its_targets_and_a_moves_source() {
        let named = |name: &str| Some(name.to_string());
        let paris = Some("paris");
        let cases = [
            (
                RdfGraphOp::Create {
                    name: "paris".into(),
                },
                [false, false, false],
            ),
            (
                RdfGraphOp::Clear {
                    target: RdfGraphTarget::Default,
                },
                [true, false, false],
            ),
            (
                RdfGraphOp::Drop {
                    target: RdfGraphTarget::Named("paris".into()),
                },
                [false, true, false],
            ),
            (
                RdfGraphOp::Clear {
                    target: RdfGraphTarget::AllNamed,
                },
                [false, true, true],
            ),
            (
                RdfGraphOp::Drop {
                    target: RdfGraphTarget::All,
                },
                [true, true, true],
            ),
            (
                RdfGraphOp::Copy {
                    source: named("paris"),
                    target: None,
                },
                [true, false, false],
            ),
            (
                RdfGraphOp::Add {
                    source: None,
                    target: named("paris"),
                },
                [false, true, false],
            ),
            (
                RdfGraphOp::Move {
                    source: named("paris"),
                    target: named("berlin"),
                },
                [false, true, true],
            ),
        ];
        for (op, expected) in cases {
            let seen = [None, paris, Some("berlin")].map(|graph| changes_graph(&op, graph));
            assert_eq!(seen, expected, "{op:?}");
        }
    }
}
