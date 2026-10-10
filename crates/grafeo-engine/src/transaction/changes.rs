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
//!
//! A batch call is a bulk write: its ids are one reserved range per table,
//! one entry however many rows, which the commit stamps and a rollback
//! undoes as a range; its rows are kept with the range only when the commit
//! logs them or reports them to change data capture
//! ([`TransactionChanges::set_keeps_bulk_rows`]). An import holds commits
//! off for its whole run and writes its rows to the WAL as it goes
//! ([`StreamingChanges`]), keeping none.
//!
//! Graph commands, schema statements and the index API change nothing a
//! change set holds: each statement or call collects its changes in a
//! [`StandaloneChange`], which is checked, logged as a group of its own and
//! applied at once, also inside a transaction, whose rollback keeps it.

use std::ops::Range;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

#[cfg(feature = "lpg")]
use grafeo_common::change::StandaloneOp;
use grafeo_common::change::{
    Before, BulkRange, Change, ChangeMark, ChangeSet, DataModel, DataOp, Entity, GraphRef,
    GraphSlot, PendingVersion, Table,
};
use grafeo_common::types::{ArcStr, EpochId, TransactionId};
use grafeo_common::utils::error::{Error, Result, TransactionError};
use grafeo_core::execution::operators::{
    BulkRows, ChangeRecorder, OperatorError, Recording, WriteClaim, WriteClaims, WriteInProgress,
    WriteTarget,
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
    /// Whether the rows of the transaction's bulk writes are kept with
    /// their ranges, for the commit's log and change data capture (see
    /// [`set_keeps_bulk_rows`](Self::set_keeps_bulk_rows)).
    keeps_bulk_rows: AtomicBool,
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
    /// The net effect of the RDF entries, which the transaction's reads lay
    /// over the store.
    #[cfg(feature = "triple-store")]
    triples: super::rdf::PendingTriples,
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
            keeps_bulk_rows: AtomicBool::new(true),
            state: Mutex::new(State {
                set,
                targets: Vec::new(),
                start,
                #[cfg(feature = "triple-store")]
                triples: super::rdf::PendingTriples::default(),
            }),
        }
    }

    /// The transaction.
    pub(crate) fn id(&self) -> TransactionId {
        self.id
    }

    /// Sets whether the rows of the transaction's bulk writes (batch calls)
    /// are kept with their ranges for the commit: they must be when its
    /// commit writes the WAL or reports change data capture events, which
    /// read them. Kept until told otherwise, so a commit that reads them
    /// never misses one.
    pub(crate) fn set_keeps_bulk_rows(&self, keep: bool) {
        self.keeps_bulk_rows.store(keep, Ordering::Release);
    }

    /// Records that a bulk write reserved `ids` in `table` of the graph of
    /// `slot`: one entry for the range.
    fn record_bulk(&self, slot: GraphSlot, table: Table, ids: Range<u64>) -> Result<()> {
        self.state.lock().set.push_bulk(BulkRange {
            graph: slot,
            table,
            ids,
        })
    }

    /// Keeps `rows` with the range a bulk write in the graph of `slot`
    /// recorded last.
    fn record_bulk_rows(&self, slot: GraphSlot, rows: Vec<DataOp>) -> Result<()> {
        self.state.lock().set.push_bulk_rows(slot, rows)
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
        #[cfg(feature = "triple-store")]
        if tail
            .iter()
            .any(|change| matches!(change, Change::Data { op, .. } if op.model() == DataModel::Rdf))
        {
            state.triples = super::rdf::PendingTriples::of(&state.set);
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

    /// Records the insert (`insert`) or the delete of `triple` in the RDF
    /// graph `graph` (`None` for the default graph). The store changes at
    /// the commit ([`apply_triples`](Self::apply_triples)); the
    /// transaction's reads see it before (see
    /// [`pending_triple`](Self::pending_triple)).
    ///
    /// # Errors
    ///
    /// When the set refuses the entry: it then records nothing.
    #[cfg(feature = "triple-store")]
    pub(crate) fn record_triple(
        &self,
        graph: Option<&str>,
        triple: grafeo_core::graph::rdf::Triple,
        insert: bool,
    ) -> Result<()> {
        use grafeo_common::storage::log_record::TripleRecord;

        let mut state = self.state.lock();
        let slot = state.set.slot(GraphRef {
            model: DataModel::Rdf,
            key: graph.map(ArcStr::from),
        })?;
        let record = Box::new(TripleRecord::from(&triple));
        let op = if insert {
            DataOp::InsertTriple { triple: record }
        } else {
            DataOp::DeleteTriple { triple: record }
        };
        state
            .set
            .push(slot, op, Before::Absent, PendingVersion::Created)?;
        state.triples.note(graph, Arc::new(triple), insert);
        Ok(())
    }

    /// Whether `triple` is in the RDF graph `graph` once the transaction
    /// commits, if the transaction wrote it there.
    #[cfg(feature = "triple-store")]
    pub(crate) fn pending_triple(
        &self,
        graph: Option<&str>,
        triple: &grafeo_core::graph::rdf::Triple,
    ) -> Option<bool> {
        self.state.lock().triples.state(graph, triple)
    }

    /// Borrows the existing RDF net and entries under a deadline-aware lock. The reader
    /// may acquire a store index afterwards, matching `apply_triples`' lock
    /// order, but its callbacks must not reenter storage or transaction state.
    #[cfg(feature = "triple-store")]
    pub(crate) fn with_path_pending(
        &self,
        control: &mut grafeo_core::execution::operators::RdfPathReadControl<'_>,
        read: impl FnOnce(
            &super::rdf::PendingTriples,
            &ChangeSet,
            &mut grafeo_core::execution::operators::RdfPathReadControl<'_>,
        ) -> std::result::Result<(), OperatorError>,
    ) -> std::result::Result<(), OperatorError> {
        loop {
            let wait = control.lock_wait()?;
            if let Some(state) = self.state.try_lock_for(wait) {
                control.poll()?;
                return read(&state.triples, &state.set, control);
            }
        }
    }

    /// The triples the transaction wrote that match `pattern` in the graphs
    /// `graphs` names, each with its graph and whether it is there once the
    /// transaction commits (see `PendingTriples::matching`).
    #[cfg(feature = "triple-store")]
    pub(crate) fn pending_triples_matching(
        &self,
        pattern: &grafeo_core::graph::rdf::TriplePattern,
        graphs: Option<&[&str]>,
    ) -> Vec<(Option<String>, Arc<grafeo_core::graph::rdf::Triple>, bool)> {
        let state = self.state.lock();
        if state.triples.is_empty() {
            return Vec::new();
        }
        state.triples.matching(pattern, graphs)
    }

    /// Whether the transaction has an RDF entry.
    #[cfg(feature = "triple-store")]
    pub(crate) fn has_pending_triples(&self) -> bool {
        !self.state.lock().triples.is_empty()
    }

    /// The first RDF graph that `graph` takes in which the transaction has
    /// an entry, as an error names it (see `PendingTriples::written_graph`).
    #[cfg(feature = "triple-store")]
    pub(crate) fn written_rdf_graph(&self, graph: impl Fn(Option<&str>) -> bool) -> Option<String> {
        self.state.lock().triples.written_graph(graph)
    }

    /// Applies the RDF entries to `store`, in recorded order, at the commit,
    /// and drops the ones that changed nothing there (a triple another
    /// transaction inserted or deleted first since this one wrote it): the
    /// log and change data capture then hear only of the triples the commit
    /// changed.
    ///
    /// # Errors
    ///
    /// When the set refuses an entry it held before (a broken invariant):
    /// the commit must not complete.
    #[cfg(feature = "triple-store")]
    pub(crate) fn apply_triples(&self, store: &grafeo_core::graph::rdf::RdfStore) -> Result<()> {
        use grafeo_core::graph::rdf::Triple;

        let mut state = self.state.lock();
        if state.triples.is_empty() {
            return Ok(());
        }
        let mut unchanged = Vec::new();
        for (at, change) in state.set.entries().iter().enumerate() {
            let Change::Data { graph, op, .. } = change else {
                continue;
            };
            let key = state
                .set
                .graph(*graph)
                .and_then(|graph| graph.key.as_deref());
            let changed = match op {
                DataOp::InsertTriple { triple } => store.insert_into(key, Triple::from(&**triple)),
                DataOp::DeleteTriple { triple } => store.remove_from(key, &Triple::from(&**triple)),
                _ => continue,
            };
            if !changed {
                unchanged.push(at);
            }
        }
        if unchanged.is_empty() {
            return Ok(());
        }
        // Take the entries out and put back those that changed the store.
        let start = state.start;
        let entries = state.set.split_off(start);
        let mut unchanged = unchanged.into_iter().peekable();
        for (at, change) in entries.into_iter().enumerate() {
            if unchanged.next_if_eq(&at).is_some() {
                continue;
            }
            match change {
                Change::Data {
                    graph,
                    op,
                    before,
                    version,
                } => state.set.push(graph, op, before, version)?,
                Change::Bulk(range) => state.set.push_bulk(range)?,
            }
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
    /// deleted, each once. A bulk range counts each of its ids: a bulk write
    /// that did not create them all failed, and its rollback removed the
    /// range.
    pub(crate) fn written_entities(&self) -> (u64, u64) {
        let state = self.state.lock();
        let mut seen: grafeo_common::utils::hash::FxHashSet<(GraphSlot, Entity)> =
            grafeo_common::utils::hash::FxHashSet::default();
        let (mut bulk_nodes, mut bulk_edges) = (0_u64, 0_u64);
        for change in state.set.entries() {
            match change {
                Change::Data { graph, op, .. } => {
                    if let Some(entity) = op.entity() {
                        seen.insert((*graph, entity));
                    }
                }
                Change::Bulk(range) => {
                    let ids = range.ids.end - range.ids.start;
                    match range.table {
                        Table::Nodes => bulk_nodes += ids,
                        Table::Edges => bulk_edges += ids,
                    }
                }
            }
        }
        let nodes = seen
            .iter()
            .filter(|(_, entity)| matches!(entity, Entity::Node(_)))
            .count();
        (
            nodes as u64 + bulk_nodes,
            (seen.len() - nodes) as u64 + bulk_edges,
        )
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
    #[cfg(any(feature = "lpg", feature = "vector-index", feature = "text-index"))]
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
/// them. A writer of a transaction gets them with its recording.
struct TransactionClaims {
    manager: Arc<TransactionManager>,
    transaction: TransactionId,
    /// The graph's storage key; `None` for the default graph.
    graph: Option<Arc<str>>,
}

impl TransactionClaims {
    /// The claims of `transaction` in the graph with storage key `graph`.
    fn new(
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

impl GraphRecorder {
    /// The error of a change the set refused although the store applied
    /// it (`what`): the store holds a change no undo or log knows of, so
    /// the database is poisoned.
    fn refused(&self, what: &str, error: &Error) -> OperatorError {
        self.claims.manager.poison(&format!(
            "transaction {:?} could not record {what} it applied: {error}",
            self.changes.id
        ));
        OperatorError::Internal(error.to_string())
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
            .map_err(|error| self.refused("a change", &error))
    }

    fn bulk(&self) -> Option<BulkRows> {
        Some(if self.changes.keeps_bulk_rows.load(Ordering::Acquire) {
            BulkRows::Keep
        } else {
            BulkRows::Drop
        })
    }

    fn record_bulk(&self, table: Table, ids: Range<u64>) -> std::result::Result<(), OperatorError> {
        // Recorded before any row is applied: a refused range changed
        // nothing.
        self.changes
            .record_bulk(self.slot, table, ids)
            .map_err(OperatorError::from)
    }

    fn record_bulk_rows(&self, rows: Vec<DataOp>) -> std::result::Result<(), OperatorError> {
        self.changes
            .record_bulk_rows(self.slot, rows)
            .map_err(|error| self.refused("the rows of a bulk write", &error))
    }
}

/// The records a [`StreamingChanges`] writes to the WAL at a time, about
/// 64 KiB of frames: what it holds of the log however many rows it writes.
#[cfg(feature = "wal")]
pub(crate) const STREAMED_RECORDS: usize = 2048;

/// The changes of a bulk write that holds commits off for its whole run (an
/// import, an RDF batch insert): a streaming change set, whose memory does
/// not grow with the rows it writes.
///
/// Its ids are reserved in ranges ([`reserve`](Self::reserve)), each one
/// entry of its change set however many rows it creates there; its rows are
/// applied as pending versions of its own transaction
/// ([`apply`](Self::apply)) and written to the WAL as they go, a run of
/// records at a time, without a commit marker. Nothing else writes the WAL
/// meanwhile, since commits wait, so the records stay one group, which its
/// commit closes with the marker ([`commit`](Self::commit)) before it
/// stamps the ranges. Until then a crash leaves records that recovery drops
/// (no marker closes them), and a failure, or a panic, drops it unfinished,
/// which undoes the ranges and closes the records with an abort marker.
/// Records of changes applied after the marker (RDF triples) are written the
/// same way ([`log`](Self::log)).
///
/// A bulk write reports no change data capture events.
pub(crate) struct StreamingChanges {
    /// The transaction manager, which it poisons when it cannot undo or
    /// close what it wrote.
    manager: Arc<TransactionManager>,
    /// The writer of its rows: an id no other transaction has (see
    /// [`TransactionManager::bulk_writer`]).
    transaction: TransactionId,
    /// The epoch it reads at.
    snapshot: EpochId,
    /// Its ranges: one entry each.
    set: ChangeSet,
    /// The store its ranges are in, with the graph's slot, once it reserved
    /// one.
    target: Option<(GraphSlot, Arc<dyn ChangeTarget>)>,
    /// The WAL, with the records not written yet.
    #[cfg(feature = "wal")]
    log: Option<StreamedLog>,
    /// Whether it committed or aborted.
    finished: bool,
}

/// The WAL a [`StreamingChanges`] writes, with its pending records.
#[cfg(feature = "wal")]
struct StreamedLog {
    wal: Arc<grafeo_storage::wal::LpgWal>,
    /// At most [`STREAMED_RECORDS`] records, written next.
    pending: Vec<grafeo_storage::wal::WalRecord>,
    /// Whether records were written: an abort closes them.
    written: bool,
}

impl StreamingChanges {
    /// The changes of a bulk write by `transaction`, reading at `snapshot`
    /// (see [`TransactionManager::bulk_writer`]), not logged. The caller
    /// holds commits off until it commits or aborts them.
    pub(crate) fn new(
        manager: Arc<TransactionManager>,
        (transaction, snapshot): (TransactionId, EpochId),
    ) -> Self {
        Self {
            manager,
            transaction,
            snapshot,
            set: ChangeSet::new(),
            target: None,
            #[cfg(feature = "wal")]
            log: None,
            finished: false,
        }
    }

    /// The same changes, logged to `wal` when there is one.
    #[cfg(feature = "wal")]
    pub(crate) fn logged_to(mut self, wal: Option<Arc<grafeo_storage::wal::LpgWal>>) -> Self {
        self.log = wal.map(|wal| StreamedLog {
            wal,
            pending: Vec::with_capacity(STREAMED_RECORDS),
            written: false,
        });
        self
    }

    /// Reserves `count` consecutive ids in `table` of `target`, the default
    /// graph's store, and records them as one entry.
    ///
    /// # Errors
    ///
    /// Fails when the ids are exhausted, or the write reserved ids in
    /// another store before.
    pub(crate) fn reserve(
        &mut self,
        target: &Arc<dyn ChangeTarget>,
        table: Table,
        count: usize,
    ) -> Result<Range<u64>> {
        let slot = match &self.target {
            Some((slot, bound)) if same_store(bound, target) => *slot,
            Some(_) => {
                return Err(Error::Internal(
                    "a bulk write reserves its ids in one store".to_string(),
                ));
            }
            None => {
                let slot = self.set.slot(GraphRef {
                    model: DataModel::Lpg,
                    key: None,
                })?;
                self.target = Some((slot, Arc::clone(target)));
                slot
            }
        };
        let count = u64::try_from(count)
            .map_err(|_| Error::Internal(format!("{count} rows: more ids than a store has")))?;
        let ids = match table {
            Table::Nodes => target.reserve_node_ids(count),
            Table::Edges => target.reserve_edge_ids(count),
        }
        .map_err(|error| Error::Internal(error.to_string()))?;
        self.set.push_bulk(BulkRange {
            graph: slot,
            table,
            ids: ids.clone(),
        })?;
        Ok(ids)
    }

    /// Applies `op`, a create at an id of a range it reserved, as a pending
    /// version of its transaction (see
    /// [`ChangeTarget::apply_bulk_row`]: an edge's endpoints are the
    /// caller's to have created), and logs it.
    ///
    /// # Errors
    ///
    /// Fails when the store refuses the row (nothing of it applied), or
    /// the WAL write of a run of records fails.
    pub(crate) fn apply(&mut self, op: DataOp) -> Result<()> {
        let Some((_, target)) = &self.target else {
            return Err(Error::Internal(
                "a bulk write's row before it reserved ids".to_string(),
            ));
        };
        let writer = Writer::Transaction {
            id: self.transaction,
            snapshot: self.snapshot,
        };
        target
            .apply_bulk_row(&op, writer)
            .map_err(|error| Error::Internal(error.to_string()))?;
        #[cfg(feature = "wal")]
        if let Some(log) = &mut self.log {
            super::v1_group::push_v1_records(&op, None, |record| log.pending.push(record));
            if log.pending.len() >= STREAMED_RECORDS {
                log.write()?;
            }
        }
        Ok(())
    }

    /// Logs `record`, of a change applied after the commit marker (an RDF
    /// triple).
    ///
    /// # Errors
    ///
    /// Fails when the WAL write of a run of records fails.
    #[cfg(all(feature = "wal", feature = "triple-store"))]
    pub(crate) fn log(&mut self, record: grafeo_storage::wal::WalRecord) -> Result<()> {
        if let Some(log) = &mut self.log {
            log.pending.push(record);
            if log.pending.len() >= STREAMED_RECORDS {
                log.write()?;
            }
        }
        Ok(())
    }

    /// Whether it changed nothing yet: no range, no record.
    pub(crate) fn is_empty(&self) -> bool {
        #[cfg(feature = "wal")]
        let logged = self
            .log
            .as_ref()
            .is_some_and(|log| log.written || !log.pending.is_empty());
        #[cfg(not(feature = "wal"))]
        let logged = false;
        self.set.is_empty() && !logged
    }

    /// The memory its change set holds, in bytes: its ranges, whatever
    /// their size.
    #[cfg(test)]
    pub(crate) fn approx_bytes(&self) -> usize {
        self.set.approx_bytes()
    }

    /// The records not written to the WAL yet: at most a run.
    #[cfg(all(test, feature = "wal"))]
    pub(crate) fn pending_records(&self) -> usize {
        self.log.as_ref().map_or(0, |log| log.pending.len())
    }

    /// Commits it: writes the rest of its records and the commit marker,
    /// which closes them, as one group, then stamps its ranges at `epoch`
    /// (the epoch advance follows the marker). Without an epoch it has no
    /// ranges (an RDF batch insert, whose triples the caller applies once
    /// this returns). The caller publishes the epoch.
    ///
    /// A WAL write of the marker that fails is reported and the commit goes
    /// on, as a transaction's commit does.
    ///
    /// # Errors
    ///
    /// Fails when it has ranges and no epoch, or the store cannot stamp its
    /// ranges (a broken invariant: the transaction manager is poisoned).
    pub(crate) fn commit(mut self, epoch: Option<EpochId>) -> Result<()> {
        self.finished = true;
        #[cfg(feature = "wal")]
        if let Some(log) = &mut self.log {
            use grafeo_storage::wal::WalRecord;
            log.pending.push(WalRecord::TransactionCommit {
                transaction_id: self.transaction,
            });
            if let Some(epoch) = epoch {
                log.pending.push(WalRecord::EpochAdvance { epoch });
            }
            if let Err(error) = log.write() {
                grafeo_common::grafeo_warn!("Failed to write a bulk write to the WAL: {}", error);
            }
        }
        if let Some((slot, target)) = &self.target {
            let Some(epoch) = epoch else {
                let message = "a bulk write with ranges committed without an epoch".to_string();
                self.manager.poison(&message);
                return Err(Error::Internal(message));
            };
            target
                .stamp(self.transaction, &mut self.set.in_graph(*slot), epoch)
                .map_err(|error| {
                    let message =
                        format!("the commit of a bulk write could not stamp its ranges: {error}");
                    self.manager.poison(&message);
                    Error::Internal(message)
                })?;
        }
        Ok(())
    }

    /// Aborts it (when it is dropped before it committed): drops its pending
    /// records, closes the ones written with an abort marker, so no later
    /// commit marker commits them, and undoes its ranges.
    ///
    /// # Errors
    ///
    /// Fails, after poisoning the transaction manager, when the abort marker
    /// cannot be written (a later commit marker would commit the records
    /// written) or the store cannot undo a range.
    fn abort(&mut self) -> Result<()> {
        self.finished = true;
        let mut failure = None;
        #[cfg(feature = "wal")]
        if let Some(log) = &mut self.log {
            log.pending.clear();
            if log.written {
                log.pending
                    .push(grafeo_storage::wal::WalRecord::TransactionAbort {
                        transaction_id: self.transaction,
                    });
                if let Err(error) = log.write() {
                    failure = Some(format!(
                        "a bulk write that failed could not close its WAL records: {error}"
                    ));
                }
            }
        }
        if let Some((slot, target)) = &self.target {
            let mut entries = self.set.in_graph(*slot);
            if let Err(error) = target.undo(self.transaction, &mut entries) {
                failure.get_or_insert(format!(
                    "a bulk write that failed could not undo its rows: {error}"
                ));
            }
        }
        match failure {
            Some(message) => {
                self.manager.poison(&message);
                Err(Error::Internal(message))
            }
            None => Ok(()),
        }
    }
}

#[cfg(feature = "wal")]
impl StreamedLog {
    /// Writes the pending records, and empties them.
    fn write(&mut self) -> Result<()> {
        // Set first: a write that fails may have written some of them.
        self.written = true;
        let written = grafeo_common::testing::crash::maybe_fail("streaming_changes:write")
            .and_then(|()| self.wal.log_batch(&self.pending));
        self.pending.clear();
        written?;
        // Tests crash here: records written, no marker closes them.
        grafeo_common::testing::crash::maybe_crash("streaming_changes:after_records");
        Ok(())
    }
}

impl Drop for StreamingChanges {
    /// A bulk write dropped before it committed (an error the caller passed
    /// on, a panic) is aborted; what cannot be aborted poisons the database.
    fn drop(&mut self) {
        if !self.finished {
            let _ = self.abort();
        }
    }
}

/// What one statement or call changes outside every change set: a named
/// graph created or dropped, catalog records put or dropped (types,
/// constraints, procedures, schemas, indexes and their names). The
/// statement checks it against the catalog and the store while it holds
/// commits off, then it is logged as one group of its own and applied op by
/// op, through the function replay applies its records with (validate, log,
/// apply). It takes effect at once, also inside a transaction, whose
/// rollback keeps it.
#[cfg(feature = "lpg")]
#[derive(Default)]
pub(crate) struct StandaloneChange {
    /// The ops, in the order they apply, each with the index a statement
    /// built for it (a vector or text index put), installed as it applies.
    ops: Vec<(StandaloneOp, Option<BuiltIndex>)>,
}

#[cfg(feature = "lpg")]
impl StandaloneChange {
    /// A change of nothing yet.
    pub(crate) fn new() -> Self {
        Self::default()
    }

    /// Adds `op`, applied after the ops added before it.
    pub(crate) fn push(&mut self, op: StandaloneOp) {
        self.ops.push((op, None));
    }

    /// Adds `op`, the put of an index record, with the index the statement
    /// built for it from the data: the op installs it instead of building
    /// it again.
    #[cfg(any(feature = "vector-index", feature = "text-index"))]
    pub(crate) fn push_built(&mut self, op: StandaloneOp, index: BuiltIndex) {
        self.ops.push((op, Some(index)));
    }

    /// Whether the change changes nothing: it logs no group then.
    pub(crate) fn is_empty(&self) -> bool {
        self.ops.is_empty()
    }

    /// The ops, in the order they apply.
    #[cfg(feature = "wal")]
    pub(crate) fn ops(&self) -> impl Iterator<Item = &StandaloneOp> {
        self.ops.iter().map(|(op, _)| op)
    }

    /// The ops with the indexes built for them, in the order they apply.
    pub(crate) fn into_ops(self) -> Vec<(StandaloneOp, Option<BuiltIndex>)> {
        self.ops
    }
}

/// An index a statement built from the data while it checked its change,
/// installed when the op that puts its record applies. Replay builds the
/// index from the data again.
#[cfg(feature = "lpg")]
pub(crate) enum BuiltIndex {
    /// A vector index.
    #[cfg(feature = "vector-index")]
    Vector(grafeo_core::index::vector::VectorIndexKind),
    /// A text index.
    #[cfg(feature = "text-index")]
    Text(grafeo_core::index::text::InvertedIndex),
}

#[cfg(all(test, feature = "triple-store"))]
mod path_overlay_tests {
    use super::super::rdf::RdfWriter;
    use super::{TransactionChanges, TransactionManager};
    use grafeo_common::types::{EpochId, TransactionId};
    use grafeo_core::execution::operators::{
        Operator, OperatorError, PathStep, RdfPathConfig, RdfPathGraph, RdfPathOperator,
        RdfPathPendingGraph, RdfPathReadControl, RdfPathReadOverlay,
    };
    use grafeo_core::graph::rdf::{RdfStore, Term, Triple, TriplePattern};
    use std::sync::Arc;
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::time::{Duration, Instant};

    enum Probe {
        Lock,
        ExpiredLock,
        Pending,
    }

    /// Reaches the real adapter after native traversal starts. Expiration is
    /// placed at the indicated stage, so initial State admission cannot pass
    /// one of these controls merely by returning an early timeout.
    struct ProbeOverlay {
        inner: Arc<dyn RdfPathReadOverlay>,
        entered: Arc<AtomicBool>,
        deadline: Instant,
        probe: Probe,
    }

    impl ProbeOverlay {
        fn expire(&self, control: &mut RdfPathReadControl<'_>) {
            control.poll().unwrap();
            self.entered.store(true, Ordering::SeqCst);
            std::thread::sleep(
                self.deadline.saturating_duration_since(Instant::now()) + Duration::from_millis(1),
            );
        }
    }

    impl RdfPathReadOverlay for ProbeOverlay {
        fn with_graph(
            &self,
            graph: Option<&str>,
            control: &mut RdfPathReadControl<'_>,
            read: &mut dyn FnMut(
                &dyn RdfPathPendingGraph,
                &mut RdfPathReadControl<'_>,
            ) -> Result<(), OperatorError>,
        ) -> Result<(), OperatorError> {
            match self.probe {
                Probe::Lock => {
                    control.poll()?;
                    self.entered.store(true, Ordering::SeqCst);
                    self.inner.with_graph(graph, control, read)
                }
                Probe::ExpiredLock => {
                    self.expire(control);
                    self.inner.with_graph(graph, control, read)
                }
                Probe::Pending => self
                    .inner
                    .with_graph(graph, control, &mut |pending, control| {
                        self.expire(control);
                        let mut visits = 0;
                        let result = pending.visit_present(
                            &TriplePattern::with_predicate(Term::iri("p")),
                            control,
                            &mut |_, _| {
                                visits += 1;
                                Ok(())
                            },
                        );
                        assert_eq!(visits, 0);
                        assert!(matches!(result, Err(OperatorError::Timeout)));
                        result
                    }),
            }
        }

        fn visit_graph_names(
            &self,
            control: &mut RdfPathReadControl<'_>,
            visit: &mut dyn FnMut(&str, &mut RdfPathReadControl<'_>) -> Result<(), OperatorError>,
        ) -> Result<(), OperatorError> {
            self.inner.visit_graph_names(control, visit)
        }

        fn retained_bytes(&self) -> usize {
            self.inner.retained_bytes() + std::mem::size_of::<Self>()
        }
    }

    fn path(changes: &Arc<TransactionChanges>, probe: Probe) -> (RdfPathOperator, Arc<AtomicBool>) {
        let store = Arc::new(RdfStore::new());
        let writer = RdfWriter::new(
            Arc::clone(&store),
            Arc::clone(changes),
            Arc::new(TransactionManager::new()),
            #[cfg(feature = "wal")]
            None,
        );
        let deadline = Instant::now() + Duration::from_millis(100);
        let entered = Arc::new(AtomicBool::new(false));
        let overlay = Arc::new(ProbeOverlay {
            inner: writer.path_overlay(),
            entered: Arc::clone(&entered),
            deadline,
            probe,
        });
        let config = RdfPathConfig {
            subject: Some(Term::iri("s")),
            object: None,
            subject_var: None,
            object_var: None,
            graph_var: None,
            min_hops: false,
            path: PathStep::Predicate("p".into()),
            graph: RdfPathGraph::Default,
            companions: false,
            transaction_id: None,
            chunk_capacity: 16,
        };
        (
            RdfPathOperator::new(store, config)
                .with_read_overlay(Some(overlay))
                .with_deadline(Some(deadline)),
            entered,
        )
    }

    #[test]
    fn path_overlay_deadline_bounds_changes_lock_wait() {
        let changes = Arc::new(TransactionChanges::new(
            TransactionId::new(2),
            EpochId::new(0),
        ));
        std::thread::scope(|scope| {
            let (held_tx, held_rx) = std::sync::mpsc::channel();
            let (release_tx, release_rx) = std::sync::mpsc::channel();
            let changes_ref = &changes;
            scope.spawn(move || {
                let _guard = changes_ref.state.lock();
                held_tx.send(()).unwrap();
                // Release even if the path regresses to an unbounded lock wait,
                // so this control fails instead of hanging the test suite.
                let _ = release_rx.recv_timeout(Duration::from_secs(2));
            });
            held_rx.recv().unwrap();
            let (mut path, entered) = path(&changes, Probe::Lock);
            let result = path.next();
            assert!(changes.state.try_lock().is_none());
            release_tx.send(()).unwrap();
            assert!(entered.load(Ordering::SeqCst));
            assert!(matches!(result, Err(OperatorError::Timeout)));
            assert!(path.next().unwrap().is_none());
        });
    }

    #[test]
    fn path_overlay_refuses_expired_changes_lock_read() {
        let changes = Arc::new(TransactionChanges::new(
            TransactionId::new(2),
            EpochId::new(0),
        ));
        let (mut path, entered) = path(&changes, Probe::ExpiredLock);
        assert!(matches!(path.next(), Err(OperatorError::Timeout)));
        assert!(entered.load(Ordering::SeqCst));
        assert!(path.next().unwrap().is_none());
        assert!(changes.state.try_lock().is_some());
    }

    #[test]
    fn path_overlay_polls_deleted_and_rejected_pending_entries() {
        for present in [false, true] {
            let changes = Arc::new(TransactionChanges::new(
                TransactionId::new(2),
                EpochId::new(0),
            ));
            for n in 0..64 {
                changes
                    .record_triple(
                        None,
                        Triple::new(
                            Term::iri("s"),
                            Term::iri("other"),
                            Term::literal(n.to_string()),
                        ),
                        present,
                    )
                    .unwrap();
            }
            let (mut path, entered) = path(&changes, Probe::Pending);
            assert!(matches!(path.next(), Err(OperatorError::Timeout)));
            assert!(entered.load(Ordering::SeqCst));
            // The reflexive, zero-column row was buffered before traversal;
            // an error discards it and releases the borrowed changes lock.
            assert!(path.next().unwrap().is_none());
            assert!(changes.state.try_lock().is_some());
        }
    }
}
