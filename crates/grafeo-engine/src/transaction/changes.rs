//! What a transaction changed: its change set, with the store of each graph
//! it writes.
//!
//! Every write of a transaction (a statement, the direct API of a session or
//! of the database, a batch) goes through a
//! [`GraphWriter`](grafeo_core::execution::operators::GraphWriter) that
//! applies it to the graph's store and records it here, one entry per
//! change ([`GraphRecorder`]). The set is then all a commit, a rollback, a
//! rollback to a savepoint and a checkpoint need: the commit hands each
//! graph's entries to its backend's commit at the commit epoch
//! ([`TransactionChanges::commit`]), logs them and reports them to change
//! data capture; a rollback hands them to the backend's rollback
//! ([`TransactionChanges::rollback`], [`TransactionChanges::rollback_to`]);
//! a checkpoint reads the committed state of what open transactions changed
//! from their entries ([`TransactionChanges::index`]). This is the one log:
//! backends keep no log of their own.
//!
//! The backend of each graph is resolved once, at the transaction's first
//! write in that graph, which opens the backend's transaction there
//! ([`GraphBackend::begin`]); both are kept with the set: the commit and the
//! rollbacks use them and never look the graph up by name again, so a graph
//! dropped and created again under its name meanwhile is never committed or
//! rolled back with ids that mean other entities there.
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

use grafeo_common::change::StandaloneOp;
use grafeo_common::change::{
    Before, BulkRange, Change, ChangeMark, ChangeSet, DataModel, DataOp, Entity, GraphRef,
    GraphSlot, PendingVersion, Table,
};
use grafeo_common::types::{ArcStr, EpochId, NodeId, TransactionId};
use grafeo_common::utils::error::{Error, Result, TransactionError};
use grafeo_core::execution::operators::{
    BulkRows, ChangeRecorder, OperatorError, Recording, WriteClaim, WriteClaims, WriteInProgress,
};
use grafeo_core::graph::apply::{ApplyError, BackendTransaction, GraphBackend, Writer};
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
    /// Per slot index: the slot, its graph's backend and the transaction
    /// open in it; `None` for an RDF graph, whose entries apply at commit
    /// and are never committed or rolled back through a backend.
    targets: Vec<Option<Bound>>,
    /// The position before the first entry, which a rollback undoes to.
    start: ChangeMark,
    /// The net effect of the RDF entries, which the transaction's reads lay
    /// over the store.
    #[cfg(feature = "triple-store")]
    triples: super::rdf::PendingTriples,
}

/// A graph a transaction writes: its slot, its backend and the backend's
/// transaction, opened at the first write there.
struct Bound {
    slot: GraphSlot,
    backend: Arc<dyn GraphBackend>,
    transaction: BackendTransaction,
}

/// The most entries a change set kept for another transaction has room for
/// (see [`TransactionChanges::is_worth_keeping`]).
const SPARE_ENTRIES: usize = 4_096;

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

    /// Empties this set in place for another transaction, as the one that
    /// had it ends: no entry, graph or pending triple, with the buffers kept
    /// (see [`TransactionManager::recycle`](super::TransactionManager::recycle)).
    pub(crate) fn empty(&mut self) {
        let state = self.state.get_mut();
        state.set.clear();
        state.targets.clear();
        state.start = state.set.mark();
        #[cfg(feature = "triple-store")]
        {
            state.triples = super::rdf::PendingTriples::default();
        }
    }

    /// Makes this set, [emptied](Self::empty), the changes of transaction
    /// `id`, which reads at `snapshot`, as [`new`](Self::new) makes them.
    pub(crate) fn reuse(&mut self, id: TransactionId, snapshot: EpochId) {
        self.id = id;
        self.snapshot = snapshot;
        *self.keeps_bulk_rows.get_mut() = true;
    }

    /// Whether the set's buffers are small enough to keep for another
    /// transaction: a large transaction's are dropped, so a spare set holds
    /// no more than a few thousand entries' room.
    pub(crate) fn is_worth_keeping(&mut self) -> bool {
        self.state.get_mut().set.capacity() <= SPARE_ENTRIES
    }

    /// The transaction.
    #[cfg(test)]
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
    /// for the default graph), whose backend is `target`: the first call for
    /// a graph opens the transaction in `target` and keeps both, the backend
    /// its entries are committed and rolled back through.
    ///
    /// # Errors
    ///
    /// Fails when the transaction wrote the graph through another store (the
    /// graph was dropped and created again since its first write), or when
    /// the backend cannot open the transaction.
    pub(crate) fn bind(
        &self,
        key: Option<&str>,
        target: Arc<dyn GraphBackend>,
    ) -> Result<(GraphSlot, BackendTransaction)> {
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
            None => {
                let transaction = target.begin(self.id, self.snapshot).map_err(|error| {
                    Error::Transaction(TransactionError::InvalidState(format!(
                        "graph '{}' could not open the transaction: {error}",
                        graph_name(key)
                    )))
                })?;
                state.targets[at] = Some(Bound {
                    slot,
                    backend: target,
                    transaction,
                });
                Ok((slot, transaction))
            }
            Some(bound) if same_store(&bound.backend, &target) => Ok((slot, bound.transaction)),
            Some(_) => Err(Error::Transaction(TransactionError::InvalidState(format!(
                "graph '{}' was dropped and created again while this transaction wrote it",
                graph_name(key)
            )))),
        }
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
        target: Arc<dyn GraphBackend>,
    ) -> Result<Recording> {
        let (slot, transaction) = self.bind(key, Arc::clone(&target))?;
        let recorder = GraphRecorder {
            changes: Arc::clone(self),
            claims: TransactionClaims::new(Arc::clone(manager), self.id, key),
            slot,
            transaction,
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

    /// Whether the transaction changed nothing (that is still recorded).
    #[cfg(feature = "wal")]
    pub(crate) fn is_empty(&self) -> bool {
        self.state.lock().set.is_empty()
    }

    /// Rolls the transaction back: every entry goes, and each graph's
    /// backend rolls its transaction back with the graph's entries, which
    /// ends it there: a second rollback (the transaction manager's abort
    /// after a session's) asks no backend again. The caller holds the write
    /// freeze.
    ///
    /// # Errors
    ///
    /// The first backend that fails: a broken invariant, after which the
    /// database must not commit anything (the caller poisons it). Every
    /// backend is asked either way, and the entries are dropped.
    pub(crate) fn rollback(&self) -> std::result::Result<(), ApplyError> {
        let mut state = self.state.lock();
        let start = state.start;
        let all = state.set.split_off(start);
        Self::forget_triples(&mut state, &all);
        let mut failure = None;
        for bound in std::mem::take(&mut state.targets).iter().flatten() {
            let mut entries = all.iter().filter(|change| change.graph() == bound.slot);
            if let Err(error) = bound.backend.rollback(bound.transaction, &mut entries) {
                failure.get_or_insert(error);
            }
        }
        failure.map_or(Ok(()), Err)
    }

    /// Rolls the transaction back to `mark` (a savepoint, or the position
    /// before a statement that failed): the entries recorded after it go,
    /// and each graph's backend rolls them back; the transaction stays open.
    /// The caller holds the write freeze.
    ///
    /// # Errors
    ///
    /// As [`rollback`](Self::rollback).
    pub(crate) fn rollback_to(&self, mark: ChangeMark) -> std::result::Result<(), ApplyError> {
        let mut state = self.state.lock();
        let tail = state.set.split_off(mark);
        if tail.is_empty() {
            return Ok(());
        }
        Self::forget_triples(&mut state, &tail);
        let mut failure = None;
        for bound in state.targets.iter().flatten() {
            let mut entries = tail.iter().filter(|change| change.graph() == bound.slot);
            if let Err(error) = bound
                .backend
                .rollback_to(bound.transaction, mark, &mut entries)
            {
                failure.get_or_insert(error);
            }
        }
        failure.map_or(Ok(()), Err)
    }

    /// Recomputes the pending triples when `dropped` held RDF entries.
    #[cfg(feature = "triple-store")]
    fn forget_triples(state: &mut State, dropped: &[Change]) {
        if dropped
            .iter()
            .any(|change| matches!(change, Change::Data { op, .. } if op.model() == DataModel::Rdf))
        {
            state.triples = super::rdf::PendingTriples::of(&state.set);
        }
    }

    /// Without the triple store no entry is of RDF: nothing to recompute.
    #[cfg(not(feature = "triple-store"))]
    fn forget_triples(_state: &mut State, _dropped: &[Change]) {}

    /// Commits the entries at `epoch`: each graph's backend commits its
    /// transaction with the graph's entries (the row-group store gives
    /// their pending versions the epoch, and its counters and epoch follow).
    ///
    /// # Errors
    ///
    /// The first backend that fails: a broken invariant, after which the
    /// commit must not complete (the caller drops its commit guard, which
    /// poisons the database).
    pub(crate) fn commit(&self, epoch: EpochId) -> std::result::Result<(), ApplyError> {
        let state = self.state.lock();
        for bound in state.targets.iter().flatten() {
            bound.backend.commit(
                bound.transaction,
                &mut state.set.in_graph(bound.slot),
                epoch,
            )?;
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
            .filter(|bound| state.set.in_graph(bound.slot).next().is_some())
            .filter_map(|bound| state.set.graph(bound.slot))
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

    /// Whether the transaction has an entry in the labeled property graph
    /// with storage key `key` (`None` for the default graph).
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
}

/// Whether two handles are the same store.
fn same_store(a: &Arc<dyn GraphBackend>, b: &Arc<dyn GraphBackend>) -> bool {
    std::ptr::addr_eq(Arc::as_ptr(a), Arc::as_ptr(b))
}

/// How errors name a graph by its storage key.
fn graph_name(key: Option<&str>) -> String {
    key.unwrap_or("default").to_string()
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
        .map_err(|error| match error {
            // Not a conflict: no write can commit until a reopen, which the
            // caller must learn (a conflict reads as retryable). A conflict
            // the manager found passes as it is, so its message is not
            // wrapped in a second conflict.
            error @ Error::Transaction(
                TransactionError::IncompleteCommit | TransactionError::WriteConflict(_),
            ) => OperatorError::Wrapped(Box::new(error)),
            error => OperatorError::WriteConflict(error.to_string()),
        })
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
    /// The transaction open in the graph's backend, whose writer the writes
    /// apply as.
    transaction: BackendTransaction,
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
        self.transaction.writer()
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

    fn change_mark(&self) -> Option<ChangeMark> {
        Some(self.changes.mark())
    }

    fn nodes_written_after(&self, mark: ChangeMark, nodes: &mut Vec<NodeId>) -> Option<ChangeMark> {
        Some(
            self.changes
                .state
                .lock()
                .set
                .nodes_written_after(self.slot, mark, nodes),
        )
    }
}

/// The changes of a bulk write that holds commits off for its whole run (an
/// import, an RDF batch insert): a streaming change set, whose memory does
/// not grow with the rows it writes.
///
/// Its ids are reserved in ranges ([`reserve`](Self::reserve)), each one
/// entry of its change set however many rows it creates there; its rows are
/// applied as pending versions of its own transaction
/// ([`apply`](Self::apply)) and written to the WAL as they go, as the
/// records of one group, which holds the WAL's writer until it ends. Nothing
/// else writes the WAL meanwhile, since commits wait. Its commit
/// ([`commit`](Self::commit)) writes the group's LAST frame before it stamps
/// the ranges; until then a crash leaves a group that recovery drops (no LAST
/// frame), and a failure, or a panic, drops it unfinished, which cuts the
/// group back and undoes the ranges. Records of changes applied after the
/// commit (RDF triples) are written into the same group
/// ([`log_triple`](Self::log_triple)).
///
/// A bulk write reports no change data capture events.
pub(crate) struct StreamingChanges<'wal> {
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
    /// The backend its ranges are in, with the graph's slot and the
    /// transaction opened there, once it reserved one.
    target: Option<Bound>,
    /// The storage key of the graph that backend is (`None` for the
    /// default graph), which its records name (see [`Self::in_graph`]).
    graph: Option<ArcStr>,
    /// The WAL and the group it writes.
    #[cfg(feature = "wal")]
    log: Option<StreamedLog<'wal>>,
    /// Whether it committed or aborted.
    finished: bool,
    /// The lifetime of the WAL it writes to: the same type in every build,
    /// with or without the `wal` feature (#445).
    _wal: std::marker::PhantomData<&'wal ()>,
}

/// The WAL a [`StreamingChanges`] writes, and its group once it began.
#[cfg(feature = "wal")]
struct StreamedLog<'wal> {
    wal: &'wal grafeo_storage::wal::Wal,
    /// The epoch the group's changes take (its `GroupBegin` names it).
    epoch: EpochId,
    /// The group, from the first record on.
    group: Option<grafeo_storage::wal::GroupWriter<'wal>>,
    /// The bytes of the record being written; reused for every record.
    scratch: Vec<u8>,
}

#[cfg(feature = "wal")]
impl<'wal> StreamingChanges<'wal> {
    /// The same changes, logged to `wal` when there is one, as a group at
    /// `epoch`: the epoch the bulk write commits at (commits wait, so it is
    /// known from the start).
    #[cfg(feature = "wal")]
    pub(crate) fn logged_to(
        mut self,
        wal: Option<&'wal grafeo_storage::wal::Wal>,
        epoch: EpochId,
    ) -> Self {
        self.log = wal.map(|wal| StreamedLog {
            wal,
            epoch,
            group: None,
            scratch: Vec::new(),
        });
        self
    }
}

impl StreamingChanges<'_> {
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
            graph: None,
            #[cfg(feature = "wal")]
            log: None,
            finished: false,
            _wal: std::marker::PhantomData,
        }
    }

    /// The same changes, in the graph with storage key `graph` (`None` for
    /// the default graph), whose store [`reserve`](Self::reserve) is given:
    /// its records name that graph, so a replay applies them there.
    pub(crate) fn in_graph(mut self, graph: Option<&str>) -> Self {
        self.graph = graph.map(ArcStr::from);
        self
    }

    /// Reserves `count` consecutive ids in `table` of `target`, the store of
    /// the changes' graph (see [`in_graph`](Self::in_graph)), and records
    /// them as one entry.
    ///
    /// # Errors
    ///
    /// Fails when the ids are exhausted, or the write reserved ids in
    /// another store before.
    pub(crate) fn reserve(
        &mut self,
        target: &Arc<dyn GraphBackend>,
        table: Table,
        count: usize,
    ) -> Result<Range<u64>> {
        let slot = match &self.target {
            Some(bound) if same_store(&bound.backend, target) => bound.slot,
            Some(_) => {
                return Err(Error::Internal(
                    "a bulk write reserves its ids in one store".to_string(),
                ));
            }
            None => {
                let slot = self.set.slot(GraphRef {
                    model: DataModel::Lpg,
                    key: self.graph.clone(),
                })?;
                let transaction = target
                    .begin(self.transaction, self.snapshot)
                    .map_err(|error| Error::Internal(error.to_string()))?;
                self.target = Some(Bound {
                    slot,
                    backend: Arc::clone(target),
                    transaction,
                });
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
    /// [`GraphBackend::apply_bulk_row`]: an edge's endpoints are the
    /// caller's to have created), and logs it.
    ///
    /// # Errors
    ///
    /// Fails when the store refuses the row (nothing of it applied), or
    /// the WAL write of a frame fails.
    pub(crate) fn apply(&mut self, op: DataOp) -> Result<()> {
        let Some(bound) = &self.target else {
            return Err(Error::Internal(
                "a bulk write's row before it reserved ids".to_string(),
            ));
        };
        bound
            .backend
            .apply_bulk_row(&op, bound.transaction.writer())
            .map_err(|error| Error::Internal(error.to_string()))?;
        #[cfg(feature = "wal")]
        if let Some(log) = &mut self.log {
            log.push(
                self.transaction,
                &grafeo_common::storage::LogRecordRef::Data {
                    graph: None,
                    op: &op,
                },
            )?;
        }
        Ok(())
    }

    /// Logs `op`, a triple inserted into or deleted from the RDF graph
    /// `graph`, which the caller applies once the bulk write commits.
    ///
    /// # Errors
    ///
    /// Fails when the WAL write of a frame fails.
    #[cfg(all(feature = "wal", feature = "triple-store"))]
    pub(crate) fn log_triple(&mut self, graph: Option<&str>, op: &DataOp) -> Result<()> {
        if let Some(log) = &mut self.log {
            log.push(
                self.transaction,
                &grafeo_common::storage::LogRecordRef::Data { graph, op },
            )?;
        }
        Ok(())
    }

    /// Whether it changed nothing yet: no range, no record.
    pub(crate) fn is_empty(&self) -> bool {
        #[cfg(feature = "wal")]
        let logged = self.log.as_ref().is_some_and(|log| log.group.is_some());
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

    /// Commits it: writes its group's LAST frame (and syncs it as the
    /// durability mode asks), then stamps its ranges at `epoch`. Without an
    /// epoch it has no ranges (an RDF batch insert, whose triples the caller
    /// applies once this returns). The caller publishes the epoch.
    ///
    /// # Errors
    ///
    /// - The group was not written: nothing of the bulk write is in the log,
    ///   and it is aborted ([`TransactionError::LogWriteFailed`]).
    /// - The group was written but not made durable: the bulk write may
    ///   survive, so the database is poisoned
    ///   ([`TransactionError::CommitOutcomeUnknown`]).
    /// - It has ranges and no epoch, or the store cannot stamp its ranges (a
    ///   broken invariant: the transaction manager is poisoned).
    ///
    /// [`TransactionError::LogWriteFailed`]: grafeo_common::utils::error::TransactionError::LogWriteFailed
    /// [`TransactionError::CommitOutcomeUnknown`]: grafeo_common::utils::error::TransactionError::CommitOutcomeUnknown
    pub(crate) fn commit(mut self, epoch: Option<EpochId>) -> Result<()> {
        #[cfg(feature = "wal")]
        if let Some(group) = self.log.as_mut().and_then(|log| log.group.take()) {
            grafeo_common::testing::crash::maybe_crash("streaming_changes:after_records");
            match group.finish() {
                Ok(_) => {}
                Err(error @ grafeo_storage::wal::GroupError::OutcomeUnknown(_)) => {
                    self.finished = true;
                    let error = crate::database::wal_log::commit_error(error);
                    self.manager.poison(&format!(
                        "a bulk write may be in the WAL unapplied: {error}"
                    ));
                    return Err(error);
                }
                // Not written: the drop aborts it (undoes the ranges).
                Err(error) => return Err(crate::database::wal_log::commit_error(error)),
            }
        }
        self.finished = true;
        if let Some(bound) = &self.target {
            let Some(epoch) = epoch else {
                let message = "a bulk write with ranges committed without an epoch".to_string();
                self.manager.poison(&message);
                return Err(Error::Internal(message));
            };
            bound
                .backend
                .commit(bound.transaction, &mut self.set.in_graph(bound.slot), epoch)
                .map_err(|error| {
                    let message =
                        format!("the commit of a bulk write could not stamp its ranges: {error}");
                    self.manager.poison(&message);
                    Error::Internal(message)
                })?;
        }
        Ok(())
    }

    /// Aborts it (when it is dropped before it committed): drops its group,
    /// which cuts it back, so no record of it stays in the log, and undoes its
    /// ranges.
    ///
    /// # Errors
    ///
    /// Fails, after poisoning the transaction manager, when the store cannot
    /// undo a range.
    fn abort(&mut self) -> Result<()> {
        self.finished = true;
        #[cfg(feature = "wal")]
        if let Some(log) = &mut self.log {
            drop(log.group.take());
        }
        if let Some(bound) = &self.target {
            let mut entries = self.set.in_graph(bound.slot);
            if let Err(error) = bound.backend.rollback(bound.transaction, &mut entries) {
                let message = format!("a bulk write that failed could not undo its rows: {error}");
                self.manager.poison(&message);
                return Err(Error::Internal(message));
            }
        }
        Ok(())
    }
}

#[cfg(feature = "wal")]
impl StreamedLog<'_> {
    /// Encodes `record` into the group, which it begins with its
    /// `GroupBegin` on the first record.
    ///
    /// # Errors
    ///
    /// The record's own error (over a limit), or the commit error of a frame
    /// write that failed (the group is cut back).
    fn push(
        &mut self,
        transaction: TransactionId,
        record: &grafeo_common::storage::LogRecordRef<'_>,
    ) -> Result<()> {
        use grafeo_common::storage::LogRecordRef;
        use grafeo_common::storage::log_record::{GroupBegin, Origin};

        use crate::database::wal_log::commit_error;

        grafeo_common::testing::crash::maybe_fail("streaming_changes:write")?;
        if self.group.is_none() {
            let mut group = self.wal.begin_group(transaction).map_err(commit_error)?;
            self.scratch.clear();
            LogRecordRef::GroupBegin(GroupBegin {
                epoch: self.epoch,
                timestamp_ms: crate::database::wal_log::now_ms(),
                origin: Origin::Bulk,
            })
            .encode_framed(&mut self.scratch)?;
            group.push(&self.scratch).map_err(commit_error)?;
            self.group = Some(group);
        }
        self.scratch.clear();
        record.encode_framed(&mut self.scratch)?;
        let group = self.group.as_mut().expect("the group began above");
        group.push(&self.scratch).map_err(commit_error)
    }
}

impl Drop for StreamingChanges<'_> {
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
#[derive(Default)]
pub(crate) struct StandaloneChange {
    /// The ops, in the order they apply, each with the index a statement
    /// built for it (a vector or text index put), installed as it applies.
    ops: Vec<(StandaloneOp, Option<BuiltIndex>)>,
}

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
    use std::sync::Arc;
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::time::{Duration, Instant};

    use grafeo_common::types::{EpochId, TransactionId};
    use grafeo_core::execution::operators::{
        Operator, OperatorError, PathStep, RdfPathConfig, RdfPathGraph, RdfPathOperator,
        RdfPathPendingGraph, RdfPathReadControl, RdfPathReadOverlay,
    };
    use grafeo_core::graph::rdf::{RdfStore, Term, Triple, TriplePattern};

    use super::super::rdf::RdfWriter;
    use super::{TransactionChanges, TransactionManager};

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
            transaction_id: None,
            chunk_capacity: 16,
            zero_length_in_graph: false,
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
