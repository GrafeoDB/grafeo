//! The direct write API: each call checked through a [`GraphWriter`] and
//! committed at an epoch of its own.
//!
//! While no transaction is open, a single call commits at once: it holds the
//! transaction manager's idle gate, so no transaction can begin meanwhile
//! and there is nothing it could conflict with, and its writes are applied
//! and stamped at its new epoch as they happen ([`Writer::Immediate`], the
//! path replay takes, lenient), which it publishes when done. Every check of
//! a single call runs before it writes. Its changes retain before-images
//! until the call succeeds: an error or panic undoes them before the epoch
//! is published, and writes no WAL group or change events.
//!
//! While a transaction is open, and for every batch, a call is a private
//! transaction: one without a session, registered with the transaction
//! manager like any other (see
//! [`TransactionManager::begin_private`](crate::transaction::TransactionManager)).
//! It writes through the graph's store, recording each write in its change
//! set (a batch as one bulk write: one entry for the ids it reserved per
//! table, see [`GraphWriter::create_nodes`]), and commits when the call
//! succeeds: its changes are stamped with the commit epoch, logged and
//! reported to change data capture as a transaction's are. A call that fails
//! or panics is rolled back through its change set, also a batch whose later
//! row fails: it leaves nothing, and its ids are not handed out again. Its
//! claims make it conflict with an open transaction that wrote the same
//! node or edge first, and an open transaction conflicts with it the same
//! way; such calls on different nodes and edges run side by side, from any
//! number of threads, beside open transactions. Like a session's
//! transaction, a call begins between commits, never in the middle of one.
//!
//! Both kinds log the same WAL records and report the same change events,
//! built from the entries their writes record.
//!
//! A database on a read-only or an external store runs each call as an
//! implicit transaction of a session instead.
//!
//! [`Writer::Immediate`]: grafeo_core::graph::apply::Writer::Immediate

use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

use grafeo_common::change::{
    Before, ChangeSet, DataModel, DataOp, GraphRef, GraphSlot, PendingVersion,
};
use grafeo_common::types::{
    ArcStr, EdgeId, EpochId, NodeId, PropertyKey, ReadView, TransactionId, Value,
};
use grafeo_common::utils::error::{Error, QueryError, QueryErrorKind, Result};
use grafeo_core::execution::operators::{
    ChangeRecorder, GraphWriter, NewEdge, OperatorError, Recording, WriteClaim, WriteClaims,
    WriteInProgress,
};
use grafeo_core::graph::apply::{GraphBackend, Writer};
use grafeo_core::graph::lpg::{Edge, Node};
use grafeo_core::graph::rowgroup::RowGroupStore;
use grafeo_core::graph::{Direction, GraphStore};

use super::GrafeoDB;
use crate::catalog::CatalogConstraintValidator;
use crate::session::graph_storage_key;
use crate::transaction::{CommitsHeld, TransactionChanges, TransactionManager};

/// Keep small-call buffers warm without retaining a large detach's log.
const CACHED_IMMEDIATE_ENTRIES: usize = 64;

/// The recorder of the direct call that commits at once ([`Writer::Immediate`]):
/// one per database, used only by the call that holds the transaction
/// manager's idle gate. It claims nothing (no transaction is open), and keeps
/// the entries until the call succeeds or its writes have been undone.
pub(crate) struct ImmediateRecorder {
    manager: Arc<TransactionManager>,
    /// The running call's epoch.
    epoch: AtomicU64,
    /// Whether a call is recording entries and their before-images.
    keep: AtomicBool,
    /// A backend apply was entered but has not supplied its undo entry.
    unrecorded: AtomicBool,
    /// The running call's entries and its graph's slot, while it keeps them.
    entries: parking_lot::Mutex<(ChangeSet, Option<GraphSlot>)>,
}

/// The completed call's changes, held in their reusable buffer until its
/// logging or rollback finishes. Every exit releases their values.
struct ImmediateChanges<'a> {
    entries: parking_lot::MutexGuard<'a, (ChangeSet, Option<GraphSlot>)>,
}

impl std::ops::Deref for ImmediateChanges<'_> {
    type Target = ChangeSet;

    fn deref(&self) -> &Self::Target {
        &self.entries.0
    }
}

impl Drop for ImmediateChanges<'_> {
    fn drop(&mut self) {
        if self.entries.0.capacity() > CACHED_IMMEDIATE_ENTRIES {
            self.entries.0 = ChangeSet::new();
        } else {
            self.entries.0.clear();
        }
    }
}

impl ImmediateRecorder {
    /// A recorder for the direct calls of the database `manager` runs.
    pub(crate) fn new(manager: Arc<TransactionManager>) -> Self {
        Self {
            manager,
            epoch: AtomicU64::new(0),
            keep: AtomicBool::new(false),
            unrecorded: AtomicBool::new(false),
            entries: parking_lot::Mutex::new((ChangeSet::new(), None)),
        }
    }

    /// Starts a call at `epoch` in the graph with storage key `graph`,
    /// retaining its undo entries until success or rollback.
    fn start(&self, epoch: EpochId, graph: Option<&str>) -> Result<()> {
        self.epoch.store(epoch.as_u64(), Ordering::Release);
        self.unrecorded.store(false, Ordering::Release);
        let mut entries = self.entries.lock();
        debug_assert!(entries.0.is_empty());
        let slot = entries.0.slot(GraphRef {
            model: DataModel::Lpg,
            key: graph.map(ArcStr::from),
        })?;
        entries.1 = Some(slot);
        self.keep.store(true, Ordering::Release);
        Ok(())
    }

    /// Ends recording and borrows the call's entries until logging or undo
    /// finishes. Dropping the guard clears or releases the reusable buffer.
    fn finish(&self) -> ImmediateChanges<'_> {
        self.keep.store(false, Ordering::Release);
        let mut entries = self.entries.lock();
        entries.1 = None;
        ImmediateChanges { entries }
    }
}

impl WriteClaims for ImmediateRecorder {
    /// Nothing to claim: the idle gate keeps every transaction out.
    fn claim(&self, _claim: WriteClaim) -> std::result::Result<(), OperatorError> {
        Ok(())
    }

    /// None: the call holds commits off for its whole run, so no
    /// checkpoint runs meanwhile.
    fn write_in_progress(&self) -> Option<WriteInProgress<'_>> {
        None
    }
}

impl ChangeRecorder for ImmediateRecorder {
    fn writer(&self) -> Writer {
        // GraphWriter asks for this immediately before applying an operation.
        // If it unwinds before record(), its undo authority is incomplete.
        self.unrecorded.store(true, Ordering::Release);
        Writer::Immediate {
            epoch: EpochId::new(self.epoch.load(Ordering::Acquire)),
            before_images: self.keep.load(Ordering::Acquire),
        }
    }

    fn record(
        &self,
        op: DataOp,
        before: Before,
        version: PendingVersion,
    ) -> std::result::Result<(), OperatorError> {
        if !self.keep.load(Ordering::Acquire) {
            return Ok(());
        }
        let mut entries = self.entries.lock();
        let (set, slot) = &mut *entries;
        let Some(slot) = *slot else {
            return Ok(());
        };
        let recorded = set.push(slot, op, before, version).map_err(|error| {
            // Applied already: an incomplete undo log cannot restore it.
            self.manager.poison(&format!(
                "a direct write could not record a change it applied: {error}"
            ));
            OperatorError::Internal(error.to_string())
        });
        if recorded.is_ok() {
            self.unrecorded.store(false, Ordering::Release);
        }
        recorded
    }
}

/// A direct call that commits at once, between its start and its finish.
struct ImmediateCall<'a> {
    /// Holds commits off until the call's epoch is published.
    commits: CommitsHeld<'a>,
    /// The database's root store, which follows the call's epoch.
    root: Arc<RowGroupStore>,
    /// The epoch the call commits at.
    epoch: EpochId,
    /// The call's writer.
    writer: GraphWriter,
}

/// The graph a direct call works in.
#[derive(Clone, Copy)]
pub(crate) enum DirectTarget<'a> {
    /// The graph `set_current_graph` and `set_current_schema` select, or the
    /// default graph when they select none. A selected graph that no longer
    /// exists is an error.
    Current,
    /// A named graph of `schema`, which must exist (a graph handle).
    Named {
        schema: Option<&'a str>,
        name: &'a str,
    },
}

/// An edge to create with [`GrafeoDB::batch_create_edges`]: its endpoints,
/// type and properties.
#[derive(Debug, Clone, PartialEq)]
pub struct BatchEdge {
    /// The source node.
    pub src: NodeId,
    /// The target node.
    pub dst: NodeId,
    /// The edge type.
    pub edge_type: String,
    /// The edge's properties.
    pub properties: HashMap<PropertyKey, Value>,
}

impl BatchEdge {
    /// An edge without properties.
    #[must_use]
    pub fn new(src: NodeId, dst: NodeId, edge_type: impl Into<String>) -> Self {
        Self {
            src,
            dst,
            edge_type: edge_type.into(),
            properties: HashMap::new(),
        }
    }

    /// The edge with `properties`.
    #[must_use]
    pub fn with_properties(
        mut self,
        properties: impl IntoIterator<Item = (impl Into<PropertyKey>, impl Into<Value>)>,
    ) -> Self {
        self.properties = properties
            .into_iter()
            .map(|(key, value)| (key.into(), value.into()))
            .collect();
        self
    }
}

/// The direct API on one graph, shared by [`GrafeoDB`] (its current graph)
/// and [`GraphHandle`](super::GraphHandle) (the handle's graph).
pub(crate) struct DirectCalls<'a> {
    db: &'a GrafeoDB,
    target: DirectTarget<'a>,
}

impl GrafeoDB {
    /// The direct API on `target`.
    pub(crate) fn direct<'a>(&'a self, target: DirectTarget<'a>) -> DirectCalls<'a> {
        DirectCalls { db: self, target }
    }

    /// Whether direct calls run as private transactions on the built-in
    /// store: not on a read-only database, and not on an external store.
    fn writes_privately(&self) -> bool {
        !self.read_only && self.root_store().is_some() && self.external_read_store.is_none()
    }

    /// The built-in store of the graph `target` names and its storage key
    /// (`None`: the default graph), or `None` without a built-in store.
    ///
    /// # Errors
    ///
    /// Returns an error if `target` names a graph that does not exist.
    fn direct_store(
        &self,
        target: DirectTarget<'_>,
    ) -> Result<Option<(Arc<RowGroupStore>, Option<String>)>> {
        let Some(root) = self.root_store() else {
            return Ok(None);
        };
        let key = match target {
            DirectTarget::Current => graph_storage_key(
                self.current_schema.read().as_deref(),
                self.current_graph.read().as_deref(),
            ),
            DirectTarget::Named { schema, name } => graph_storage_key(schema, Some(name)),
        };
        let Some(key) = key else {
            return Ok(Some((root, None)));
        };
        if let Some(store) = root.graph(&key) {
            return Ok(Some((store, Some(key))));
        }
        // The graph was dropped, or its schema: never fall back to another.
        let name = match target {
            DirectTarget::Named { name, .. } => name.to_string(),
            DirectTarget::Current => self
                .current_graph
                .read()
                .clone()
                .unwrap_or_else(|| "default".to_string()),
        };
        Err(missing_graph(&name))
    }

    /// The store the direct API reads the graph `target` names from: its own
    /// store for a named graph; for the default graph the store queries read,
    /// which is the external store of a database built with `with_store` or
    /// `with_read_store`.
    ///
    /// # Errors
    ///
    /// Returns an error if `target` names a graph that does not exist.
    pub(crate) fn read_store(&self, target: DirectTarget<'_>) -> Result<Arc<dyn GraphStore>> {
        Ok(match self.direct_store(target)? {
            Some((store, Some(_))) => store,
            _ => self.graph_store(),
        })
    }

    /// Runs one direct call (a batch is one too) on `target`: a single call
    /// while no transaction is open commits at once; otherwise the call is a
    /// private transaction; on a read-only or external store, an implicit
    /// transaction of a session.
    fn write_direct<T>(
        &self,
        target: DirectTarget<'_>,
        batch: bool,
        write: impl FnOnce(&GraphWriter) -> std::result::Result<T, OperatorError>,
    ) -> Result<T> {
        if let Some((store, graph)) = self.direct_store(target)?
            && self.writes_privately()
        {
            if !batch && let Some(_gate) = self.transaction_manager.idle_gate() {
                return self.write_immediate(&store, graph.as_deref(), write);
            }
            return self.write_private(&store, graph.as_deref(), write);
        }
        let session = match target {
            DirectTarget::Current => self.session(),
            DirectTarget::Named { schema, name } => self.graph_in(schema, name)?.session()?,
        };
        session.write(write)
    }

    /// Runs one direct call on `store` (the graph with storage key `graph`)
    /// that commits at once, at a new epoch, which it publishes when done.
    /// The caller holds the idle gate; this holds commits off (see
    /// [`TransactionManager::hold_commits_for_change`](crate::transaction::TransactionManager))
    /// from its epoch until it is published, so no checkpoint holds part of
    /// it (lock order: the idle gate, then the commit lock, as in `begin`).
    /// Fails, like a commit, once the database is closed.
    fn write_immediate<T>(
        &self,
        store: &Arc<RowGroupStore>,
        graph: Option<&str>,
        write: impl FnOnce(&GraphWriter) -> std::result::Result<T, OperatorError>,
    ) -> Result<T> {
        // Only this shell is generic: one copy per kind of call.
        let call = self.start_immediate(store, graph)?;
        let outcome =
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| write(&call.writer)));
        match outcome {
            Ok(Ok(value)) => {
                self.finish_immediate(call, store)?;
                Ok(value)
            }
            Ok(Err(error)) => {
                let error = crate::query::executor::convert_operator_error(error);
                match self.rollback_immediate(call, store) {
                    Ok(()) => Err(error),
                    Err(undo) => Err(Error::Internal(format!("{error}; {undo}"))),
                }
            }
            Err(panic) => {
                if self.immediate_recorder().unrecorded.load(Ordering::Acquire) {
                    self.transaction_manager
                        .poison("a direct write panicked before its applied change was recorded");
                }
                let _ = self.rollback_immediate(call, store);
                std::panic::resume_unwind(panic)
            }
        }
    }

    /// Undoes an unsuccessful call while its epoch is still unpublished.
    /// A failed undo fences later writes; it is never reported as success.
    fn rollback_immediate(
        &self,
        call: ImmediateCall<'_>,
        store: &Arc<RowGroupStore>,
    ) -> Result<()> {
        let ImmediateCall {
            commits,
            root: _,
            epoch,
            writer,
        } = call;
        drop(writer);
        let changes = self.immediate_recorder().finish();
        // A recording failure may have omitted an applied operation. Do not
        // mistake undoing its recorded prefix for a complete rollback.
        let undone =
            self.transaction_manager
                .check_no_incomplete_commit()
                .and_then(|()| {
                    store.rollback_immediate(epoch, changes.entries()).map_err(|failure| {
                    let message = format!(
                        "the rollback of a direct write could not undo its changes: {failure}"
                    );
                    self.transaction_manager.poison(&message);
                    Error::Internal(message)
                })
                });
        drop(changes);
        drop(commits);
        undone
    }

    /// Starts a direct call that commits at once (see
    /// [`write_immediate`](Self::write_immediate)): holds commits off, moves
    /// the stores to the call's epoch and builds its writer.
    fn start_immediate<'a>(
        &'a self,
        store: &Arc<RowGroupStore>,
        graph: Option<&str>,
    ) -> Result<ImmediateCall<'a>> {
        let commits = self.transaction_manager.hold_commits_for_change()?;
        let root = self.lpg_store()?;
        let epoch = EpochId::new(self.transaction_manager.current_epoch().as_u64() + 1);
        // The stores read at the new epoch from here: the call sees every
        // commit, and stamps its writes (and the store's history) at it.
        root.sync_epoch(epoch);
        store.sync_epoch(epoch);
        // Tests start a checkpoint or `close()` here (the call's epoch has
        // moved, nothing is published yet), which must wait.
        #[cfg(feature = "testing-statement-injection")]
        grafeo_common::testing::commit_hook::run_during_held_change();

        let recorder = self.immediate_recorder();
        recorder.start(epoch, graph)?;
        let writer = self.direct_writer(
            store,
            graph,
            (epoch, None),
            Recording {
                target: Arc::clone(store) as Arc<dyn GraphBackend>,
                recorder: Arc::clone(recorder) as Arc<dyn ChangeRecorder>,
            },
        );
        // GraphWriter construction reads the writer context too; no apply
        // has run at this point. Subsequent writer() calls precede applies.
        recorder.unrecorded.store(false, Ordering::Release);
        Ok(ImmediateCall {
            commits,
            root,
            epoch,
            writer,
        })
    }

    /// Finishes a direct call that commits at once: logs what it wrote,
    /// publishes its epoch, reports its changes to change data capture.
    ///
    /// # Errors
    ///
    /// [`TransactionError::CommitOutcomeUnknown`] when its WAL group is not
    /// written or not made durable: its writes are applied and stamped at an
    /// epoch nobody reads, and cannot be taken back, so the epoch is not
    /// published and the database is poisoned; a reopen decides from the log
    /// whether the call happened.
    ///
    /// [`TransactionError::CommitOutcomeUnknown`]: grafeo_common::utils::error::TransactionError::CommitOutcomeUnknown
    fn finish_immediate(&self, call: ImmediateCall<'_>, store: &Arc<RowGroupStore>) -> Result<()> {
        let ImmediateCall {
            commits,
            root,
            epoch,
            writer,
        } = call;
        drop(writer);
        let changes = self.immediate_recorder().finish();
        // While the database keeps history, the time of the call: its WAL
        // group carries it, and the time index records it once published.
        let at = self.transaction_manager.commit_time();

        #[cfg(feature = "wal")]
        if let Some(wal) = &self.wal
            && !changes.is_empty()
            && let Err(error) = Self::log_immediate(wal, &changes, epoch, at)
        {
            self.transaction_manager.poison(&format!(
                "a direct write at epoch {epoch:?} is applied but not in the WAL: {error}"
            ));
            return Err(Error::Transaction(
                grafeo_common::utils::error::TransactionError::CommitOutcomeUnknown(
                    error.to_string(),
                ),
            ));
        }
        // Reported before the epoch is published, while commits are held
        // off: in epoch order, as a commit reports its changes.
        #[cfg(feature = "cdc")]
        if self.cdc_active() {
            self.cdc_log.record_commit(&changes, epoch);
        }
        drop(changes);
        self.transaction_manager.sync_epoch(epoch);
        if let Some(at) = at {
            self.transaction_manager.record_commit_time(epoch, at);
        }
        drop(commits);
        self.prune_versions(&root, store);
        Ok(())
    }

    /// Writes the WAL group of a direct call that committed at once at
    /// `epoch`, which wrote `changes`.
    ///
    /// # Errors
    ///
    /// The group's error: it is cut back when it was not written.
    #[cfg(feature = "wal")]
    fn log_immediate(
        wal: &grafeo_storage::wal::Wal,
        changes: &grafeo_common::change::ChangeSet,
        epoch: EpochId,
        at: Option<u64>,
    ) -> Result<()> {
        let mut group = wal
            .begin_group(TransactionId::SYSTEM)
            .map_err(crate::database::wal_log::commit_error)?;
        crate::database::wal_log::write_group_at(
            &mut group,
            changes,
            epoch,
            grafeo_common::storage::log_record::Origin::Direct,
            at,
        )?;
        group
            .finish()
            .map(|_| ())
            .map_err(crate::database::wal_log::commit_error)
    }

    /// The writer of a direct call on `store` (the graph with storage key
    /// `graph`), reading at `context` (an epoch, and the transaction for a
    /// private one) and writing through `recording`, checked against the
    /// catalog.
    fn direct_writer(
        &self,
        store: &Arc<RowGroupStore>,
        graph: Option<&str>,
        context: (EpochId, Option<TransactionId>),
        recording: Recording,
    ) -> GraphWriter {
        // A graph of a schema has the storage key `schema/graph`: the types of
        // that schema check the call.
        let schema = graph
            .and_then(|key| key.split_once('/'))
            .map(|(schema, _)| schema);
        let mut validator = CatalogConstraintValidator::new(Arc::clone(&self.catalog))
            .with_store(Arc::clone(store) as Arc<dyn GraphStore>)
            .with_max_property_size(self.config.max_property_size())
            .with_transaction_context(context.0, context.1)
            .with_schema(schema)
            .with_key_claims(Arc::clone(&self.transaction_manager), graph);
        if let Some(graph) = graph {
            validator = validator.with_graph_name(graph);
        }
        GraphWriter::new(Arc::clone(store) as Arc<dyn GraphStore>, recording)
            .with_validator(Arc::new(validator))
    }

    /// Whether a commit of the direct API reads what its writes did after
    /// they are applied: when it logs them to the WAL or reports them to
    /// change data capture. Without, a batch keeps no rows for commit;
    /// immediate calls still retain before-images until success or rollback.
    fn reads_committed_rows(&self) -> bool {
        #[cfg(feature = "wal")]
        let wal = self.wal.is_some();
        #[cfg(not(feature = "wal"))]
        let wal = false;
        #[cfg(feature = "cdc")]
        let cdc = self.cdc_active();
        #[cfg(not(feature = "cdc"))]
        let cdc = false;
        wal || cdc
    }

    /// The recorder of the direct calls that commit at once.
    fn immediate_recorder(&self) -> &Arc<ImmediateRecorder> {
        self.immediate_writes.get_or_init(|| {
            Arc::new(ImmediateRecorder::new(Arc::clone(
                &self.transaction_manager,
            )))
        })
    }

    /// After a direct call that wrote `store`, collects the versions no
    /// reader needs any more in the root store and `store`, at the read
    /// horizon, when a collect is due for them (see
    /// [`AfterWrite`](crate::database::collect::AfterWrite)).
    fn prune_versions(&self, root: &Arc<RowGroupStore>, store: &Arc<RowGroupStore>) {
        crate::database::collect::AfterWrite::next(&self.commit_counter, self.config.gc_interval())
            .run(
                &self.transaction_manager,
                &[&**root as &dyn GraphBackend, &**store],
            );
    }

    /// Runs one direct call on `store` (the graph with storage key `graph`)
    /// as a private transaction: commits it when `write` succeeds, rolls it
    /// back when it fails or panics (and raises the panic again). It begins
    /// once a commit or checkpoint in progress is done, and fails, like a
    /// commit, once the database is closed.
    fn write_private<T>(
        &self,
        store: &Arc<RowGroupStore>,
        graph: Option<&str>,
        write: impl FnOnce(&GraphWriter) -> std::result::Result<T, OperatorError>,
    ) -> Result<T> {
        // Only this shell is generic: one copy per kind of call.
        let (transaction, changes, writer) = self.start_private(store, graph)?;
        // A panic in the call is handled like an error, then raised again:
        // its writes are rolled back first.
        let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| write(&writer)));
        drop(writer);
        match outcome {
            Ok(Ok(value)) => {
                self.commit_private(transaction, &changes, store)?;
                Ok(value)
            }
            Ok(Err(error)) => Err(self.fail_private(transaction, &changes, error)),
            Err(panic) => {
                let _ = self.rollback_private(transaction, &changes);
                std::panic::resume_unwind(panic)
            }
        }
    }

    /// Begins a direct call as a private transaction (see
    /// [`write_private`](Self::write_private)) and builds its writer.
    fn start_private(
        &self,
        store: &Arc<RowGroupStore>,
        graph: Option<&str>,
    ) -> Result<(TransactionId, Arc<TransactionChanges>, GraphWriter)> {
        let (transaction, changes) = self.transaction_manager.begin_private()?;
        // A batch's rows are kept for the commit only when it logs them or
        // reports them to change data capture.
        changes.set_keeps_bulk_rows(self.reads_committed_rows());
        let recording = match changes.recording(
            &self.transaction_manager,
            graph,
            Arc::clone(store) as Arc<dyn GraphBackend>,
        ) {
            Ok(recording) => recording,
            Err(error) => {
                let _ = self.transaction_manager.abort(transaction);
                return Err(error);
            }
        };
        let writer = self.direct_writer(
            store,
            graph,
            (changes.snapshot(), Some(transaction)),
            recording,
        );
        Ok((transaction, changes, writer))
    }

    /// Rolls back the private transaction `transaction` whose call failed
    /// with `error`; the error to return.
    fn fail_private(
        &self,
        transaction: TransactionId,
        changes: &TransactionChanges,
        error: OperatorError,
    ) -> Error {
        let error = crate::query::executor::convert_operator_error(error);
        match self.rollback_private(transaction, changes) {
            Ok(()) => error,
            Err(undo) => Error::Internal(format!("{error}; {undo}")),
        }
    }

    /// Commits the private transaction `transaction`, which wrote
    /// `changes` in `store`: stamps them with the commit epoch, logs them
    /// as one WAL group, reports them to change data capture, publishes the
    /// epoch.
    fn commit_private(
        &self,
        transaction: TransactionId,
        changes: &TransactionChanges,
        store: &Arc<RowGroupStore>,
    ) -> Result<()> {
        let commit = match self.transaction_manager.start_commit(transaction) {
            Ok(commit) => commit,
            Err(error) => {
                let _ = self.rollback_private(transaction, changes);
                return Err(error);
            }
        };
        let epoch = commit.epoch();
        // Tests start a checkpoint or `close()` here (the commit holds its
        // epoch, nothing is published yet), which must wait for it.
        #[cfg(feature = "testing-statement-injection")]
        grafeo_common::testing::commit_hook::run_during_held_change();

        // The group is in the log before anything of the commit is stamped
        // (#498): one that is not written takes the commit back, and the
        // call is rolled back.
        #[cfg(feature = "wal")]
        let (commit, logged) = match &self.wal {
            Some(wal) => match crate::database::wal_log::write_commit(
                wal,
                commit,
                transaction,
                changes,
                grafeo_common::storage::log_record::Origin::Direct,
            ) {
                Ok(written) => written,
                Err(error) => {
                    if self.transaction_manager.state(transaction)
                        == Some(crate::transaction::TransactionState::Active)
                    {
                        let _ = self.rollback_private(transaction, changes);
                    }
                    return Err(error);
                }
            },
            None => (commit, None),
        };

        // As a session's commit: the commit lock goes once the group is
        // written, the sync comes after it, then the ordered publish.
        let publish = commit.written();
        #[cfg(feature = "wal")]
        if let (Some(wal), Some(logged)) = (&self.wal, logged) {
            crate::database::wal_log::sync_commit(wal, logged)?;
        }

        let root = self.lpg_store()?;
        let published = publish.complete_with(|| {
            // A store that fails to stamp leaves the commit half done: it is
            // not published, which poisons the database.
            changes.commit(epoch).map_err(|error| {
                Error::Internal(format!(
                    "the commit of a direct write could not stamp its changes: {error}"
                ))
            })?;

            // Reported in epoch order, before the epoch is published.
            #[cfg(feature = "cdc")]
            if self.cdc_active() {
                changes.read(|set| self.cdc_log.record_commit(set, epoch));
            }

            // The database has one epoch: the root store follows every
            // commit (the stamp moved the written store's).
            root.sync_epoch(epoch);
            Ok(())
        });
        #[cfg(feature = "wal")]
        let published = published
            .map_err(|error| crate::database::wal_log::publish_error(error, logged.is_some()));
        published?;

        self.prune_versions(&root, store);
        Ok(())
    }

    /// Rolls the private transaction `transaction` back: undoes `changes`,
    /// last to first, and aborts it.
    ///
    /// # Errors
    ///
    /// When a store fails to undo, which poisons the database.
    fn rollback_private(
        &self,
        transaction: TransactionId,
        changes: &TransactionChanges,
    ) -> Result<()> {
        let undone = {
            let _writing = self.transaction_manager.write_in_progress();
            changes.rollback()
        };
        let _ = self.transaction_manager.abort(transaction);
        match undone {
            Ok(()) => Ok(()),
            Err(failure) => {
                let message =
                    format!("the rollback of a direct write could not undo its changes: {failure}");
                self.transaction_manager.poison(&message);
                Err(Error::Internal(message))
            }
        }
    }
}

impl DirectCalls<'_> {
    fn write<T>(
        &self,
        write: impl FnOnce(&GraphWriter) -> std::result::Result<T, OperatorError>,
    ) -> Result<T> {
        self.db.write_direct(self.target, false, write)
    }

    /// A batch: one call, all of its rows or none, so always a private
    /// transaction, which can undo the rows before a failing one.
    fn write_batch<T>(
        &self,
        write: impl FnOnce(&GraphWriter) -> std::result::Result<T, OperatorError>,
    ) -> Result<T> {
        self.db.write_direct(self.target, true, write)
    }

    /// The store of the graph, for reads (see [`GrafeoDB::read_store`]).
    fn store(&self) -> Result<Arc<dyn GraphStore>> {
        self.db.read_store(self.target)
    }

    pub(crate) fn create_node_with_props(
        &self,
        labels: &[&str],
        properties: impl IntoIterator<Item = (impl Into<PropertyKey>, impl Into<Value>)>,
    ) -> Result<NodeId> {
        let labels: Vec<String> = labels.iter().map(|label| (*label).to_string()).collect();
        let properties = direct_properties(properties);
        self.write(|writer| writer.create_node(&labels, properties))
    }

    pub(crate) fn create_edge_with_props(
        &self,
        src: NodeId,
        dst: NodeId,
        edge_type: &str,
        properties: impl IntoIterator<Item = (impl Into<PropertyKey>, impl Into<Value>)>,
    ) -> Result<EdgeId> {
        let properties = direct_properties(properties);
        self.write(|writer| create_edge(writer, src, dst, edge_type, properties))
    }

    pub(crate) fn set_node_property(&self, id: NodeId, key: &str, value: Value) -> Result<()> {
        self.write(|writer| set_node_property(writer, id, key, value))
    }

    pub(crate) fn set_edge_property(&self, id: EdgeId, key: &str, value: Value) -> Result<()> {
        self.write(|writer| set_edge_property(writer, id, key, value))
    }

    pub(crate) fn remove_node_property(&self, id: NodeId, key: &str) -> Result<bool> {
        self.write(
            // The direct API reports a missing node as `false`; a query that
            // writes to one fails (see `GraphWriter`).
            |writer| {
                if writer.has_node(id)? {
                    writer.remove_node_property(id, key)
                } else {
                    Ok(false)
                }
            },
        )
    }

    pub(crate) fn remove_edge_property(&self, id: EdgeId, key: &str) -> Result<bool> {
        self.write(|writer| {
            if writer.has_edge(id)? {
                writer.remove_edge_property(id, key)
            } else {
                Ok(false)
            }
        })
    }

    pub(crate) fn add_node_label(&self, id: NodeId, label: &str) -> Result<bool> {
        self.write(|writer| add_node_label(writer, id, label))
    }

    pub(crate) fn remove_node_label(&self, id: NodeId, label: &str) -> Result<bool> {
        self.write(|writer| remove_node_label(writer, id, label))
    }

    pub(crate) fn delete_node(&self, id: NodeId) -> Result<bool> {
        self.write(|writer| writer.delete_node(id, false))
    }

    pub(crate) fn detach_delete_node(&self, id: NodeId) -> Result<bool> {
        self.write(|writer| writer.delete_node(id, true))
    }

    pub(crate) fn delete_edge(&self, id: EdgeId) -> Result<bool> {
        self.write(|writer| writer.delete_edge(id))
    }

    pub(crate) fn batch_create_nodes(
        &self,
        label: &str,
        property: &str,
        vectors: Vec<Vec<f32>>,
    ) -> Result<Vec<NodeId>> {
        self.write_batch(|writer| create_vector_nodes(writer, label, property, vectors))
    }

    pub(crate) fn batch_create_nodes_with_labels(
        &self,
        labels: &[&str],
        properties_list: Vec<HashMap<PropertyKey, Value>>,
    ) -> Result<Vec<NodeId>> {
        self.write_batch(|writer| create_nodes(writer, labels, properties_list))
    }

    pub(crate) fn batch_create_edges(&self, edges: Vec<BatchEdge>) -> Result<Vec<EdgeId>> {
        self.write_batch(|writer| create_edges(writer, edges))
    }

    pub(crate) fn get_node(&self, id: NodeId) -> Result<Option<Node>> {
        let view = ReadView::committed(self.db.read_epoch());
        self.store()?.node(id, view)
    }

    pub(crate) fn get_edge(&self, id: EdgeId) -> Result<Option<Edge>> {
        let view = ReadView::committed(self.db.read_epoch());
        self.store()?.edge(id, view)
    }

    pub(crate) fn get_node_edges(
        &self,
        node: NodeId,
        direction: Direction,
        edge_types: &[&str],
    ) -> Result<Vec<(NodeId, EdgeId)>> {
        let view = ReadView::committed(self.db.read_epoch());
        node_edges(&*self.store()?, node, direction, edge_types, view)
    }
}

/// The edges of `node` in `direction` that `view` sees, as `(neighbor,
/// edge)` pairs, only those whose type is one of `edge_types` (as stored)
/// unless it is empty; none when the view does not see the node.
/// `Direction::Both` lists a self-loop once (see [`GraphStore::edges`]).
pub(crate) fn node_edges(
    store: &dyn GraphStore,
    node: NodeId,
    direction: Direction,
    edge_types: &[&str],
    view: ReadView,
) -> Result<Vec<(NodeId, EdgeId)>> {
    let types: Vec<ArcStr> = edge_types.iter().map(|name| ArcStr::from(*name)).collect();
    store.edges(node, direction, &types, view)
}

// === The writes of the direct API, shared with `Session` ===

/// Creates an edge between two nodes the writer sees.
pub(crate) fn create_edge(
    writer: &GraphWriter,
    src: NodeId,
    dst: NodeId,
    edge_type: &str,
    properties: Vec<(String, Value)>,
) -> std::result::Result<EdgeId, OperatorError> {
    for endpoint in [src, dst] {
        if !writer.has_node(endpoint)? {
            return Err(missing_node(endpoint));
        }
    }
    writer.create_edge(src, dst, edge_type, properties)
}

/// Sets a property of a node the writer sees.
pub(crate) fn set_node_property(
    writer: &GraphWriter,
    id: NodeId,
    key: &str,
    value: Value,
) -> std::result::Result<(), OperatorError> {
    if !writer.has_node(id)? {
        return Err(missing_node(id));
    }
    writer.set_node_properties(id, &[(key.to_string(), value)], false)
}

/// Sets a property of an edge the writer sees.
pub(crate) fn set_edge_property(
    writer: &GraphWriter,
    id: EdgeId,
    key: &str,
    value: Value,
) -> std::result::Result<(), OperatorError> {
    if !writer.has_edge(id)? {
        return Err(OperatorError::from(Error::EdgeNotFound(id)));
    }
    writer.set_edge_properties(id, &[(key.to_string(), value)], false)
}

/// Adds a label; whether the node exists and lacked it.
pub(crate) fn add_node_label(
    writer: &GraphWriter,
    id: NodeId,
    label: &str,
) -> std::result::Result<bool, OperatorError> {
    if !writer.has_node(id)? {
        return Ok(false);
    }
    Ok(writer.add_labels(id, &[label.to_string()])? == 1)
}

/// Removes a label; whether the node exists and had it.
pub(crate) fn remove_node_label(
    writer: &GraphWriter,
    id: NodeId,
    label: &str,
) -> std::result::Result<bool, OperatorError> {
    if !writer.has_node(id)? {
        return Ok(false);
    }
    Ok(writer.remove_labels(id, &[label.to_string()])? == 1)
}

/// Creates one node with `label` per vector, the vector as `property`, as
/// one bulk write (see [`GraphWriter::create_nodes`]).
pub(crate) fn create_vector_nodes(
    writer: &GraphWriter,
    label: &str,
    property: &str,
    vectors: Vec<Vec<f32>>,
) -> std::result::Result<Vec<NodeId>, OperatorError> {
    let rows = vectors
        .into_iter()
        .map(|vector| vec![(property.to_string(), Value::Vector(vector.into()))])
        .collect();
    writer.create_nodes(&[label.to_string()], rows)
}

/// Creates one node with all of `labels` per property map, as one bulk
/// write (see [`GraphWriter::create_nodes`]).
pub(crate) fn create_nodes(
    writer: &GraphWriter,
    labels: &[&str],
    properties_list: Vec<HashMap<PropertyKey, Value>>,
) -> std::result::Result<Vec<NodeId>, OperatorError> {
    let labels: Vec<String> = labels.iter().map(|label| (*label).to_string()).collect();
    let rows = properties_list.into_iter().map(direct_properties).collect();
    writer.create_nodes(&labels, rows)
}

/// Creates the edges, each between two nodes the writer sees, as one bulk
/// write (see [`GraphWriter::create_edges`]).
pub(crate) fn create_edges(
    writer: &GraphWriter,
    edges: Vec<BatchEdge>,
) -> std::result::Result<Vec<EdgeId>, OperatorError> {
    let edges = edges
        .into_iter()
        .map(|edge| NewEdge {
            src: edge.src,
            dst: edge.dst,
            edge_type: edge.edge_type,
            properties: direct_properties(edge.properties),
        })
        .collect();
    writer.create_edges(edges)
}

/// The properties of a direct write as `(key, value)` pairs.
pub(crate) fn direct_properties(
    properties: impl IntoIterator<Item = (impl Into<PropertyKey>, impl Into<Value>)>,
) -> Vec<(String, Value)> {
    properties
        .into_iter()
        .map(|(key, value)| {
            let key: PropertyKey = key.into();
            (key.as_str().to_string(), value.into())
        })
        .collect()
}

/// The error for a direct write to a node that does not exist.
fn missing_node(id: NodeId) -> OperatorError {
    OperatorError::from(Error::NodeNotFound(id))
}

/// The error for a graph handle whose graph does not exist.
pub(crate) fn missing_graph(name: &str) -> Error {
    Error::from(QueryError::new(
        QueryErrorKind::InvalidReference,
        format!("Graph '{name}' does not exist"),
    ))
}

#[cfg(all(
    test,
    feature = "wal",
    feature = "grafeo-file",
    feature = "cdc",
    feature = "gql"
))]
mod tests {
    use grafeo_common::types::EpochId;

    use super::*;
    use crate::cdc::EntityId;
    use crate::config::{Config, StorageFormat};

    /// Runs a direct call on `db` that creates a `label` node and panics.
    fn panicking_call(db: &GrafeoDB, batch: bool, label: &str) -> NodeId {
        let created = parking_lot::Mutex::new(None);
        let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            db.write_direct::<()>(DirectTarget::Current, batch, |writer| {
                *created.lock() = Some(writer.create_node(&[label.to_string()], Vec::new())?);
                panic!("the call fails halfway");
            })
        }));
        assert!(outcome.is_err(), "the panic comes through");
        created.into_inner().unwrap()
    }

    /// Tells [`a_panicking_call_leaves_nothing_for_the_next`], run in a child
    /// process, where its database is.
    const CHILD_PATH_VAR: &str = "GRAFEO_DIRECT_PANICKING_CALL_PATH";

    /// A batch or single call that panics leaves nothing: its changes are
    /// undone, and the next call writes none of its WAL records or events.
    /// The calls run in a child
    /// process that exits without `close()`, so the reopen replays the WAL.
    #[test]
    fn a_panicking_call_leaves_nothing_for_the_next() {
        let config = |path: &std::path::Path| {
            Config::persistent(path)
                .with_storage_format(StorageFormat::Auto)
                .with_cdc()
        };
        if let Some(path) = std::env::var_os(CHILD_PATH_VAR) {
            let db = GrafeoDB::with_config(config(std::path::Path::new(&path))).unwrap();
            writes_and_panics(&db);
            // Crash: no close(), no checkpoint, no destructors.
            std::process::exit(0);
        }
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("db.grafeo");
        let status = grafeo_common::testing::child_process::run(
            std::process::Command::new(std::env::current_exe().unwrap())
                .args([
                    "--exact",
                    "database::direct::tests::a_panicking_call_leaves_nothing_for_the_next",
                    "--nocapture",
                ])
                .env(CHILD_PATH_VAR, &path),
        )
        .unwrap();
        assert!(status.success(), "the child process failed");
        assert!(path.exists(), "the child process created no database");

        let db = GrafeoDB::with_config(config(&path)).unwrap();
        let labels = db
            .execute("MATCH (n) RETURN labels(n)[0] AS label ORDER BY label")
            .unwrap();
        assert_eq!(labels.rows(), [[Value::from("Person")]]);
        db.close().unwrap();
    }

    /// The calls of [`a_panicking_call_leaves_nothing_for_the_next`], with
    /// what memory holds after them.
    fn writes_and_panics(db: &GrafeoDB) {
        let batch = panicking_call(db, true, "Batch");
        let single = panicking_call(db, false, "Single");
        let alix = db
            .create_node_with_props(&["Person"], [("name", Value::from("Alix"))])
            .unwrap();

        assert!(
            db.get_node(batch).unwrap().is_none(),
            "the batch's node is discarded"
        );
        assert!(db.get_node(single).unwrap().is_none());
        let events: Vec<EntityId> = db
            .changes_between(EpochId::new(0), db.current_epoch())
            .unwrap()
            .into_iter()
            .map(|event| event.entity_id)
            .collect();
        assert_eq!(events, [EntityId::Node(alix)]);
    }
}

#[cfg(test)]
mod immediate_failure_controls {
    use super::*;

    #[test]
    fn finished_changes_release_values_and_bound_capacity_on_every_exit() {
        for count in [1_u64, 65] {
            for outcome in 0..3 {
                let recorder = ImmediateRecorder::new(Arc::new(TransactionManager::new()));
                recorder.start(EpochId::new(1), None).unwrap();
                let payload = Arc::new(std::collections::BTreeMap::from([(
                    PropertyKey::new("kept"),
                    Value::from("x".repeat(300)),
                )]));
                for raw in 1..=count {
                    recorder
                        .record(
                            DataOp::SetNodeProperty {
                                id: NodeId::new(raw),
                                key: PropertyKey::new("value"),
                                value: Value::Map(Arc::clone(&payload)),
                            },
                            Before::Value(None),
                            PendingVersion::Created,
                        )
                        .unwrap();
                }
                let result =
                    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| -> Result<()> {
                        let changes = recorder.finish();
                        assert_eq!(changes.entries().len(), usize::try_from(count).unwrap());
                        assert_eq!(
                            Arc::strong_count(&payload),
                            usize::try_from(count).unwrap() + 1
                        );
                        match outcome {
                            0 => Ok(()),
                            1 => Err(Error::Internal("finished call failed".to_string())),
                            _ => panic!("finished call unwinds"),
                        }
                    }));
                match outcome {
                    0 => result.unwrap().unwrap(),
                    1 => assert!(matches!(
                        result.unwrap(),
                        Err(Error::Internal(message)) if message == "finished call failed"
                    )),
                    _ => assert_eq!(
                        result.unwrap_err().downcast_ref::<&str>(),
                        Some(&"finished call unwinds")
                    ),
                }
                assert_eq!(Arc::strong_count(&payload), 1);
                let cached = recorder.entries.lock();
                assert!(cached.0.is_empty());
                assert_eq!(cached.0.graphs(), []);
                assert!(cached.0.capacity() <= CACHED_IMMEDIATE_ENTRIES);
                assert!(cached.1.is_none());
            }
        }
    }

    #[test]
    fn a_large_direct_call_releases_its_undo_capacity() {
        let db = GrafeoDB::new_in_memory();
        let a = db.create_node(&["A"]).unwrap();
        let b = db.create_node(&["B"]).unwrap();
        for _ in 0..80 {
            db.create_edge(a, b, "REL").unwrap();
        }
        db.detach_delete_node(a).unwrap();
        assert_eq!(db.edge_count().unwrap(), 0);
        assert!(db.immediate_recorder().entries.lock().0.capacity() <= CACHED_IMMEDIATE_ENTRIES);
        db.create_node(&["Next"]).unwrap();
        assert!(db.immediate_recorder().entries.lock().0.capacity() <= CACHED_IMMEDIATE_ENTRIES);
    }

    #[test]
    fn a_panic_before_any_apply_leaves_the_database_usable() {
        let db = GrafeoDB::new_in_memory();
        let before = db.current_epoch();
        let panic = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            db.write_direct::<()>(DirectTarget::Current, false, |_| panic!("before apply"))
        }));
        assert!(panic.is_err());
        assert_eq!(db.current_epoch(), before);
        assert_eq!(db.node_count().unwrap(), 0);
        db.create_node(&["After"]).unwrap();
        assert_eq!(db.node_count().unwrap(), 1);
    }

    #[test]
    fn an_unrecorded_backend_panic_fences_later_writes() {
        let db = GrafeoDB::new_in_memory();
        let before = db.current_epoch();
        let store = db.lpg_store().unwrap();
        let panic = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            db.write_direct::<()>(DirectTarget::Current, false, |_| {
                let id = NodeId::new(store.reserve_node_ids(1).unwrap().start);
                store
                    .apply(
                        &DataOp::CreateNode {
                            id,
                            labels: vec![ArcStr::from("Unrecorded")].into(),
                            properties: Vec::new(),
                        },
                        db.immediate_recorder().writer(),
                    )
                    .unwrap();
                panic!("backend unwinds before recording");
            })
        }));
        assert!(panic.is_err());
        assert_eq!(db.current_epoch(), before);
        assert_eq!(
            db.node_count().unwrap(),
            0,
            "the unpublished node is hidden"
        );
        assert!(matches!(
            db.create_node(&["Refused"]),
            Err(Error::Transaction(
                grafeo_common::utils::error::TransactionError::IncompleteCommit
            ))
        ));
    }
}
