//! What a transaction changed: its change set, with the store of each graph
//! it writes.
//!
//! Every write of a transaction (a statement, the direct API of a session or
//! of the database, a batch) goes through a
//! [`GraphWriter`](grafeo_core::execution::operators::GraphWriter) that
//! applies it to the graph's store and records it here, one entry per
//! change ([`GraphRecorder`]). The set is then all a commit, a rollback, a
//! rollback to a savepoint and a checkpoint need: the commit stamps each
//! graph's entries with the commit epoch ([`TransactionChanges::stamp`]),
//! logs them and reports them to change data capture; a rollback undoes
//! them, last to first, through the same stores
//! ([`TransactionChanges::undo_after`]); a checkpoint reads the committed
//! state of what open transactions changed from their entries
//! ([`TransactionChanges::index`]). Stores keep nothing per transaction.
//!
//! The store of each graph is resolved once, at the transaction's first
//! write in that graph, and kept with the set: the commit and the undo use
//! that handle and never look the graph up by name again, so a graph dropped
//! and created again under its name meanwhile is never stamped or undone
//! with ids that mean other entities there.

use std::sync::Arc;

use grafeo_common::change::{
    Before, Change, ChangeMark, ChangeSet, DataModel, DataOp, Entity, GraphRef, GraphSlot,
    PendingVersion,
};
use grafeo_common::types::{ArcStr, EpochId, TransactionId};
use grafeo_common::utils::error::{Error, Result, TransactionError};
use grafeo_core::execution::operators::{
    ChangeRecorder, OperatorError, Recording, WriteClaim, WriteClaims, WriteInProgress, WriteTarget,
};
use grafeo_core::graph::apply::{ApplyError, ChangeTarget, UndoSupport, Writer};
use parking_lot::Mutex;

use super::{GraphEntity, TransactionManager};

/// One transaction's changes: its change set and, per graph, the store the
/// entries were applied to.
pub(crate) struct TransactionChanges {
    /// The transaction.
    id: TransactionId,
    /// The epoch it reads at.
    snapshot: EpochId,
    /// The set and the stores. Lock order: taken after the write freeze
    /// and the transaction manager's lock, never before them.
    state: Mutex<State>,
}

/// The mutable part of [`TransactionChanges`].
struct State {
    /// The entries, in the order they were applied.
    set: ChangeSet,
    /// Per slot index: the slot and its graph's store; `None` for an RDF
    /// graph, whose entries apply at commit and are never stamped or undone.
    targets: Vec<Option<(GraphSlot, Arc<dyn ChangeTarget>)>>,
    /// The position before the first entry, which a rollback undoes to.
    start: ChangeMark,
}

/// Why an undo did not restore everything.
#[derive(Debug)]
pub(crate) enum UndoFailure {
    /// The entries to undo include writes to a graph whose store has no
    /// undo (a store the database was built on); the error names the graph.
    /// A rollback to a savepoint is refused before anything is undone.
    External(String),
    /// A store failed to undo an entry: a broken invariant. The database
    /// must not commit anything afterwards.
    Broken(ApplyError),
}

impl TransactionChanges {
    /// The changes of transaction `id`, which reads at `snapshot`: none yet.
    pub(crate) fn new(id: TransactionId, snapshot: EpochId) -> Self {
        let set = ChangeSet::new();
        let start = set.mark();
        Self {
            id,
            snapshot,
            state: Mutex::new(State {
                set,
                targets: Vec::new(),
                start,
            }),
        }
    }

    /// The transaction.
    pub(crate) fn id(&self) -> TransactionId {
        self.id
    }

    /// The epoch the transaction reads at.
    pub(crate) fn snapshot(&self) -> EpochId {
        self.snapshot
    }

    /// The slot of the labeled property graph with storage key `key` (`None`
    /// for the default graph), whose store is `target`: the first call for
    /// a graph keeps `target` as the store its entries are stamped and
    /// undone through.
    ///
    /// # Errors
    ///
    /// Fails when the transaction wrote the graph through another store: the
    /// graph was dropped and created again since its first write.
    pub(crate) fn bind(
        &self,
        key: Option<&str>,
        target: Arc<dyn ChangeTarget>,
    ) -> Result<GraphSlot> {
        let mut state = self.state.lock();
        let slot = state.set.slot(GraphRef {
            model: DataModel::Lpg,
            key: key.map(ArcStr::from),
        })?;
        let at = slot.index();
        if state.targets.len() <= at {
            state.targets.resize_with(at + 1, || None);
        }
        match &state.targets[at] {
            None => state.targets[at] = Some((slot, target)),
            Some((_, bound)) if same_store(bound, &target) => {}
            Some(_) => {
                return Err(Error::Transaction(TransactionError::InvalidState(format!(
                    "graph '{}' was dropped and created again while this transaction wrote it",
                    graph_name(key)
                ))));
            }
        }
        Ok(slot)
    }

    /// A recording of this transaction's writes to the graph with storage
    /// key `key` through `target`, its claims made through `manager`.
    ///
    /// # Errors
    ///
    /// As [`bind`](Self::bind).
    pub(crate) fn recording(
        self: &Arc<Self>,
        manager: &Arc<TransactionManager>,
        key: Option<&str>,
        target: WriteTarget,
    ) -> Result<Recording> {
        let store: Arc<dyn ChangeTarget> = match &target {
            WriteTarget::Store(store) => Arc::clone(store),
            WriteTarget::External(store) => Arc::clone(store) as Arc<dyn ChangeTarget>,
        };
        let slot = self.bind(key, store)?;
        let recorder = GraphRecorder {
            changes: Arc::clone(self),
            claims: TransactionClaims::new(Arc::clone(manager), self.id, key),
            slot,
        };
        Ok(Recording {
            target,
            recorder: Arc::new(recorder),
        })
    }

    /// Records a change applied to the graph of `slot`.
    fn record(
        &self,
        slot: GraphSlot,
        op: DataOp,
        before: Before,
        version: PendingVersion,
    ) -> Result<()> {
        self.state.lock().set.push(slot, op, before, version)
    }

    /// The position after the last entry, for a savepoint.
    pub(crate) fn mark(&self) -> ChangeMark {
        self.state.lock().set.mark()
    }

    /// The position before the first entry: a rollback undoes to it.
    pub(crate) fn start(&self) -> ChangeMark {
        self.state.lock().start
    }

    /// Whether the transaction changed nothing (that is still recorded).
    #[cfg(any(feature = "wal", feature = "cdc"))]
    pub(crate) fn is_empty(&self) -> bool {
        self.state.lock().set.is_empty()
    }

    /// Undoes the entries recorded after `mark`, last to first, per graph
    /// through the store kept for it, and drops them from the set. Entries
    /// in a graph whose store has no undo are dropped and the graph is
    /// returned: its store keeps those writes. With `refuse_external`, such
    /// entries refuse the whole undo instead, before anything changes (a
    /// rollback to a savepoint). The caller holds the write freeze.
    ///
    /// # Errors
    ///
    /// [`UndoFailure::External`] with `refuse_external`;
    /// [`UndoFailure::Broken`] when a store fails to undo an entry (the
    /// entries are dropped either way).
    pub(crate) fn undo_after(
        &self,
        mark: ChangeMark,
        refuse_external: bool,
    ) -> std::result::Result<Option<String>, UndoFailure> {
        let mut state = self.state.lock();
        let without_undo = Self::graph_without_undo(&state, mark);
        if refuse_external && let Some(graph) = without_undo {
            return Err(UndoFailure::External(graph));
        }
        let tail = state.set.split_off(mark);
        if tail.is_empty() {
            return Ok(None);
        }
        let mut failure = None;
        for (slot, target) in state.targets.iter().flatten() {
            if target.undo_support() != UndoSupport::Exact {
                continue;
            }
            let mut entries = tail.iter().filter(|change| change.graph() == *slot);
            if let Err(error) = target.undo(self.id, &mut entries) {
                failure.get_or_insert(error);
            }
        }
        match failure {
            Some(error) => Err(UndoFailure::Broken(error)),
            None => Ok(without_undo),
        }
    }

    /// The name of a graph without undo that has entries after `mark`.
    fn graph_without_undo(state: &State, mark: ChangeMark) -> Option<String> {
        let after = state.set.after(mark);
        state.targets.iter().flatten().find_map(|(slot, target)| {
            (target.undo_support() == UndoSupport::None
                && after.iter().any(|change| change.graph() == *slot))
            .then(|| {
                graph_name(
                    state
                        .set
                        .graph(*slot)
                        .and_then(|graph| graph.key.as_deref()),
                )
            })
        })
    }

    /// Commits the entries at `epoch`, per graph through the store kept for
    /// it: their pending versions get the epoch, and the stores' counters
    /// and epochs follow.
    ///
    /// # Errors
    ///
    /// The first store that fails: a broken invariant, after which the
    /// commit must not complete (the caller drops its commit guard, which
    /// poisons the database).
    pub(crate) fn stamp(&self, epoch: EpochId) -> std::result::Result<(), ApplyError> {
        let state = self.state.lock();
        for (slot, target) in state.targets.iter().flatten() {
            target.stamp(self.id, &mut state.set.in_graph(*slot), epoch)?;
        }
        Ok(())
    }

    /// The storage keys of the labeled property graphs the transaction
    /// wrote (`None` for the default graph).
    pub(crate) fn written_graphs(&self) -> Vec<Option<String>> {
        let state = self.state.lock();
        state
            .targets
            .iter()
            .flatten()
            .filter(|(slot, _)| state.set.in_graph(*slot).next().is_some())
            .filter_map(|(slot, _)| state.set.graph(*slot))
            .map(|graph| graph.key.as_ref().map(ToString::to_string))
            .collect()
    }

    /// The nodes and edges the transaction wrote: created, changed or
    /// deleted, each once.
    pub(crate) fn written_entities(&self) -> (u64, u64) {
        let state = self.state.lock();
        let mut seen: grafeo_common::utils::hash::FxHashSet<(GraphSlot, Entity)> =
            grafeo_common::utils::hash::FxHashSet::default();
        for change in state.set.entries() {
            if let Change::Data { graph, op, .. } = change
                && let Some(entity) = op.entity()
            {
                seen.insert((*graph, entity));
            }
        }
        let nodes = seen
            .iter()
            .filter(|(_, entity)| matches!(entity, Entity::Node(_)))
            .count();
        (nodes as u64, (seen.len() - nodes) as u64)
    }

    /// Runs `read` on the change set.
    #[cfg(any(feature = "wal", feature = "cdc"))]
    pub(crate) fn read<R>(&self, read: impl FnOnce(&ChangeSet) -> R) -> R {
        read(&self.state.lock().set)
    }

    /// Whether the transaction has an entry in a labeled property graph.
    pub(crate) fn writes_any_graph(&self) -> bool {
        self.writes_where(|_| true)
    }

    /// Whether the transaction has an entry in the labeled property graph
    /// with storage key `key` (`None` for the default graph).
    #[cfg(any(feature = "vector-index", feature = "text-index"))]
    pub(crate) fn writes_graph(&self, key: Option<&str>) -> bool {
        self.writes_where(|graph| graph == key)
    }

    /// Whether the transaction has an entry in a labeled property graph
    /// whose storage key `graph` takes.
    fn writes_where(&self, graph: impl Fn(Option<&str>) -> bool) -> bool {
        let state = self.state.lock();
        state.set.entries().iter().any(|change| {
            state.set.graph(change.graph()).is_some_and(|written| {
                written.model == DataModel::Lpg && graph(written.key.as_deref())
            })
        })
    }

    /// The committed state of what the transactions of `sets` changed, from
    /// their entries, for a checkpoint: the caller holds their writes,
    /// rollbacks and commits until the store is written (see
    /// [`OpenChangesByGraph`](grafeo_core::graph::lpg::OpenChangesByGraph)).
    #[cfg(feature = "lpg")]
    pub(crate) fn index(sets: &[Arc<Self>]) -> grafeo_core::graph::lpg::OpenChangesByGraph {
        let states: Vec<_> = sets.iter().map(|changes| changes.state.lock()).collect();
        let graphs = states.iter().flat_map(|state| {
            state.targets.iter().flatten().map(move |(slot, _)| {
                (
                    state
                        .set
                        .graph(*slot)
                        .and_then(|graph| graph.key.as_deref()),
                    state.set.in_graph(*slot),
                )
            })
        });
        grafeo_core::graph::lpg::OpenChangesByGraph::index(graphs)
    }
}

/// Whether two handles are the same store.
fn same_store(a: &Arc<dyn ChangeTarget>, b: &Arc<dyn ChangeTarget>) -> bool {
    std::ptr::addr_eq(Arc::as_ptr(a), Arc::as_ptr(b))
}

/// How errors name a graph by its storage key.
fn graph_name(key: Option<&str>) -> String {
    key.unwrap_or("default").to_string()
}

/// What the error of [`kept_by_external_store`] says after the graph.
const KEPT_BY_EXTERNAL_STORE: &str =
    "is a store the database was built on, which has no undo: it keeps this transaction's writes";

/// The error for writes a store without undo keeps: a rollback undid the
/// transaction's other writes, and the graph's store keeps these.
pub(crate) fn kept_by_external_store(graph: &str) -> Error {
    Error::Transaction(TransactionError::InvalidState(format!(
        "graph '{graph}' {KEPT_BY_EXTERNAL_STORE}"
    )))
}

/// Whether `error` is one of [`kept_by_external_store`].
pub(crate) fn is_kept_by_external_store(error: &Error) -> bool {
    matches!(
        error,
        Error::Transaction(TransactionError::InvalidState(message))
            if message.ends_with(KEPT_BY_EXTERNAL_STORE)
    )
}

/// The claims of a transaction's writes to one graph, through the
/// transaction manager (first writer wins), and the write freeze around
/// them. A writer of a transaction gets them with its recording; one that
/// writes through the store's versioned methods (a `QueryProcessor` with a
/// transaction context) gets them alone.
pub(crate) struct TransactionClaims {
    manager: Arc<TransactionManager>,
    transaction: TransactionId,
    /// The graph's storage key; `None` for the default graph.
    graph: Option<Arc<str>>,
}

impl TransactionClaims {
    /// The claims of `transaction` in the graph with storage key `graph`.
    pub(crate) fn new(
        manager: Arc<TransactionManager>,
        transaction: TransactionId,
        graph: Option<&str>,
    ) -> Self {
        Self {
            manager,
            transaction,
            graph: graph.map(Arc::from),
        }
    }

    fn entity(&self, entity: impl Into<super::EntityId>) -> GraphEntity {
        GraphEntity::new(self.graph.clone(), entity)
    }
}

impl WriteClaims for TransactionClaims {
    fn claim(&self, claim: WriteClaim) -> std::result::Result<(), OperatorError> {
        let transaction = self.transaction;
        match claim {
            WriteClaim::Node(id) => self.manager.record_write(transaction, self.entity(id)),
            WriteClaim::NodeDelete(id) => self.manager.record_delete(transaction, self.entity(id)),
            WriteClaim::Edge(id) => self.manager.record_write(transaction, self.entity(id)),
            WriteClaim::Endpoints(src, dst) => self
                .manager
                .record_endpoints(transaction, [self.entity(src), self.entity(dst)]),
        }
        .map_err(|error| OperatorError::WriteConflict(error.to_string()))
    }

    fn write_in_progress(&self) -> Option<WriteInProgress<'_>> {
        Some(self.manager.write_in_progress())
    }
}

/// A transaction's recording of its writes to one graph: the claims go to
/// the transaction manager, the entries to the transaction's change set.
struct GraphRecorder {
    changes: Arc<TransactionChanges>,
    claims: TransactionClaims,
    /// The graph's slot in the set.
    slot: GraphSlot,
}

impl WriteClaims for GraphRecorder {
    fn claim(&self, claim: WriteClaim) -> std::result::Result<(), OperatorError> {
        self.claims.claim(claim)
    }

    fn write_in_progress(&self) -> Option<WriteInProgress<'_>> {
        self.claims.write_in_progress()
    }
}

impl ChangeRecorder for GraphRecorder {
    fn writer(&self) -> Writer {
        Writer::Transaction {
            id: self.changes.id(),
            snapshot: self.changes.snapshot(),
        }
    }

    fn record(
        &self,
        op: DataOp,
        before: Before,
        version: PendingVersion,
    ) -> std::result::Result<(), OperatorError> {
        self.changes
            .record(self.slot, op, before, version)
            .map_err(|error| {
                // The store holds a change no undo knows of.
                self.claims.manager.poison(&format!(
                    "transaction {:?} could not record a change it applied: {error}",
                    self.changes.id
                ));
                OperatorError::Execution(error.to_string())
            })
    }
}
