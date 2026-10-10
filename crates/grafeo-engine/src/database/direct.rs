//! The direct write API: each call checked through a [`GraphWriter`] and
//! committed at an epoch of its own.
//!
//! While no transaction is open, a single call commits at once: it holds the
//! transaction manager's idle gate, so no transaction can begin meanwhile
//! and there is nothing it could conflict with, and its writes are applied
//! and stamped at its new epoch as they happen ([`Writer::Immediate`], the
//! path replay takes, lenient), which it publishes when done. Every check of
//! a single call runs before it writes, so the call cannot half-apply; should
//! it ever fail after a write, what it wrote is committed and logged, so the
//! WAL matches memory. A call that fails still uses up its epoch.
//!
//! While a transaction is open, and for every batch, a call is a private
//! transaction: one without a session, registered with the transaction
//! manager like any other (see
//! [`TransactionManager::begin_private`](crate::transaction::TransactionManager)).
//! It writes through the graph's store, recording each write in its change
//! set, and commits when the call succeeds: its changes are stamped with the
//! commit epoch, logged and reported to change data capture as a
//! transaction's are. A call that fails or panics is rolled back through its
//! change set, also a batch whose later row fails: it leaves nothing. Its
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
use grafeo_common::types::{ArcStr, EdgeId, EpochId, NodeId, PropertyKey, TransactionId, Value};
use grafeo_common::utils::error::{Error, QueryError, QueryErrorKind, Result};
use grafeo_core::execution::operators::{
    ChangeRecorder, GraphWriter, OperatorError, Recording, WriteClaim, WriteClaims,
    WriteInProgress, WriteTarget,
};
use grafeo_core::graph::apply::{ChangeTarget, Writer};
use grafeo_core::graph::lpg::{Edge, LpgStore, Node};
use grafeo_core::graph::{GraphStoreMut, GraphStoreSearch};

use super::GrafeoDB;
use crate::catalog::CatalogConstraintValidator;
use crate::session::graph_storage_key;
use crate::transaction::{CommitsHeld, TransactionChanges, TransactionManager};

/// The recorder of the direct call that commits at once ([`Writer::Immediate`]):
/// one per database, used only by the call that holds the transaction
/// manager's idle gate. It claims nothing (no transaction is open), and keeps
/// the entries only when the WAL or change data capture reads them.
pub(crate) struct ImmediateRecorder {
    manager: Arc<TransactionManager>,
    /// The running call's epoch.
    epoch: AtomicU64,
    /// Whether the running call keeps its entries (and builds their
    /// before-images).
    keep: AtomicBool,
    /// The running call's entries and its graph's slot, while it keeps them.
    entries: parking_lot::Mutex<(ChangeSet, Option<GraphSlot>)>,
}

impl ImmediateRecorder {
    /// A recorder for the direct calls of the database `manager` runs.
    pub(crate) fn new(manager: Arc<TransactionManager>) -> Self {
        Self {
            manager,
            epoch: AtomicU64::new(0),
            keep: AtomicBool::new(false),
            entries: parking_lot::Mutex::new((ChangeSet::new(), None)),
        }
    }

    /// Starts a call that commits at `epoch` in the graph with storage key
    /// `graph`, keeping its entries when `keep`.
    fn start(&self, epoch: EpochId, graph: Option<&str>, keep: bool) -> Result<()> {
        self.epoch.store(epoch.as_u64(), Ordering::Release);
        self.keep.store(keep, Ordering::Release);
        if keep {
            let mut entries = self.entries.lock();
            let mut set = ChangeSet::new();
            let slot = set.slot(GraphRef {
                model: DataModel::Lpg,
                key: graph.map(ArcStr::from),
            })?;
            *entries = (set, Some(slot));
        }
        Ok(())
    }

    /// The running call's entries, taken out (none when it kept none).
    fn finish(&self) -> ChangeSet {
        if !self.keep.swap(false, Ordering::AcqRel) {
            return ChangeSet::new();
        }
        let mut entries = self.entries.lock();
        entries.1 = None;
        std::mem::take(&mut entries.0)
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
        set.push(slot, op, before, version).map_err(|error| {
            // Committed already: the log would miss it.
            self.manager.poison(&format!(
                "a direct write could not record a change it committed: {error}"
            ));
            OperatorError::Execution(error.to_string())
        })
    }
}

/// A direct call that commits at once, between its start and its finish.
struct ImmediateCall<'a> {
    /// Holds commits off until the call's epoch is published.
    commits: CommitsHeld<'a>,
    /// The database's root store, which follows the call's epoch.
    root: Arc<LpgStore>,
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
    ) -> Result<Option<(Arc<LpgStore>, Option<String>)>> {
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
    pub(crate) fn read_store(&self, target: DirectTarget<'_>) -> Result<Arc<dyn GraphStoreSearch>> {
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
        store: &Arc<LpgStore>,
        graph: Option<&str>,
        write: impl FnOnce(&GraphWriter) -> std::result::Result<T, OperatorError>,
    ) -> Result<T> {
        // Only this shell is generic: one copy per kind of call.
        let call = self.start_immediate(store, graph)?;
        // A panic in the call is handled like an error, then raised again:
        // what it wrote is committed and logged first.
        let outcome =
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| write(&call.writer)));
        self.finish_immediate(call, store);
        match outcome {
            Ok(result) => result.map_err(crate::query::executor::convert_operator_error),
            Err(panic) => std::panic::resume_unwind(panic),
        }
    }

    /// Starts a direct call that commits at once (see
    /// [`write_immediate`](Self::write_immediate)): holds commits off, moves
    /// the stores to the call's epoch and builds its writer.
    fn start_immediate<'a>(
        &'a self,
        store: &Arc<LpgStore>,
        graph: Option<&str>,
    ) -> Result<ImmediateCall<'a>> {
        let commits = self.transaction_manager.hold_commits_for_change()?;
        let root = self.lpg_store();
        let epoch = EpochId::new(self.transaction_manager.current_epoch().as_u64() + 1);
        // The stores read at the new epoch from here: the call sees every
        // commit, and stamps its writes (and the store's history) at it.
        root.sync_epoch(epoch);
        store.sync_epoch(epoch);
        // Tests start a checkpoint or `close()` here (the call's epoch has
        // moved, nothing is published yet), which must wait.
        #[cfg(feature = "testing-statement-injection")]
        grafeo_common::testing::commit_hook::run_during_held_change();

        let keep = {
            #[cfg(feature = "wal")]
            let wal = self.wal.is_some();
            #[cfg(not(feature = "wal"))]
            let wal = false;
            #[cfg(feature = "cdc")]
            let cdc = self.cdc_active();
            #[cfg(not(feature = "cdc"))]
            let cdc = false;
            wal || cdc
        };
        let recorder = self.immediate_recorder();
        recorder.start(epoch, graph, keep)?;
        let writer = self.direct_writer(
            store,
            graph,
            (epoch, None),
            Recording {
                target: WriteTarget::Store(Arc::clone(store) as Arc<dyn ChangeTarget>),
                recorder: Arc::clone(recorder) as Arc<dyn ChangeRecorder>,
            },
        );
        Ok(ImmediateCall {
            commits,
            root,
            epoch,
            writer,
        })
    }

    /// Finishes a direct call that commits at once: logs what it wrote,
    /// publishes its epoch, reports its changes to change data capture.
    fn finish_immediate(&self, call: ImmediateCall<'_>, store: &Arc<LpgStore>) {
        let ImmediateCall {
            commits,
            root,
            epoch,
            writer,
        } = call;
        drop(writer);
        let changes = self.immediate_recorder().finish();

        #[cfg(feature = "wal")]
        if let Some(wal) = &self.wal
            && !changes.is_empty()
        {
            use grafeo_storage::wal::WalRecord;
            let group = crate::transaction::wal_buffer::build_group(
                crate::transaction::v1_group::v1_records(&changes),
                &[
                    WalRecord::TransactionCommit {
                        transaction_id: TransactionId::SYSTEM,
                    },
                    WalRecord::EpochAdvance { epoch },
                ],
            );
            if let Err(e) = wal.log_batch(&group) {
                grafeo_common::grafeo_warn!("Failed to write a direct write to the WAL: {}", e);
            }
        }
        self.transaction_manager.sync_epoch(epoch);
        drop(commits);
        #[cfg(feature = "cdc")]
        if self.cdc_active() && !changes.is_empty() {
            let events = crate::cdc::events_for_changes(&changes, &self.cdc_log);
            self.cdc_log
                .record_batch(crate::cdc::fold_into_creates(events).into_iter().map(
                    |mut event| {
                        event.epoch = epoch;
                        event
                    },
                ));
        }
        // A build without the log and change data capture keeps no entries.
        #[cfg(not(any(feature = "wal", feature = "cdc")))]
        drop(changes);
        self.prune_versions(&root, store);
    }

    /// The writer of a direct call on `store` (the graph with storage key
    /// `graph`), reading at `context` (an epoch, and the transaction for a
    /// private one) and writing through `recording`, checked against the
    /// catalog.
    fn direct_writer(
        &self,
        store: &Arc<LpgStore>,
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
            .with_store(Arc::clone(store) as Arc<dyn GraphStoreSearch>)
            .with_max_property_size(self.config.max_property_size)
            .with_transaction_context(context.0, context.1)
            .with_schema(schema);
        if let Some(graph) = graph {
            validator = validator.with_graph_name(graph);
        }
        GraphWriter::new(Arc::clone(store) as Arc<dyn GraphStoreMut>)
            .with_validator(Arc::new(validator))
            .with_recording(recording)
    }

    /// The recorder of the direct calls that commit at once.
    fn immediate_recorder(&self) -> &Arc<ImmediateRecorder> {
        self.immediate_writes.get_or_init(|| {
            Arc::new(ImmediateRecorder::new(Arc::clone(
                &self.transaction_manager,
            )))
        })
    }

    /// Every `gc_interval` commits, prunes the versions no reader needs, in
    /// the root store and `store`.
    fn prune_versions(&self, root: &Arc<LpgStore>, store: &Arc<LpgStore>) {
        if self.config.gc_interval > 0 {
            let count = self.commit_counter.fetch_add(1, Ordering::Relaxed) + 1;
            if count.is_multiple_of(self.config.gc_interval) {
                let min_epoch = self.transaction_manager.min_active_epoch();
                root.gc_versions(min_epoch);
                if !Arc::ptr_eq(root, store) {
                    store.gc_versions(min_epoch);
                }
                self.transaction_manager.gc();
            }
        }
    }

    /// Runs one direct call on `store` (the graph with storage key `graph`)
    /// as a private transaction: commits it when `write` succeeds, rolls it
    /// back when it fails or panics (and raises the panic again). It begins
    /// once a commit or checkpoint in progress is done, and fails, like a
    /// commit, once the database is closed.
    fn write_private<T>(
        &self,
        store: &Arc<LpgStore>,
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
        store: &Arc<LpgStore>,
        graph: Option<&str>,
    ) -> Result<(TransactionId, Arc<TransactionChanges>, GraphWriter)> {
        let (transaction, changes) = self.transaction_manager.begin_private()?;
        let recording = match changes.recording(
            &self.transaction_manager,
            graph,
            WriteTarget::Store(Arc::clone(store) as Arc<dyn ChangeTarget>),
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
        store: &Arc<LpgStore>,
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

        // A store that fails to stamp leaves the commit half done: the
        // guard, dropped uncompleted, poisons the database.
        changes.stamp(epoch).map_err(|error| {
            Error::Internal(format!(
                "the commit of a direct write could not stamp its changes: {error}"
            ))
        })?;

        #[cfg(feature = "wal")]
        if let Some(wal) = &self.wal
            && !changes.is_empty()
        {
            use grafeo_storage::wal::WalRecord;
            let records = changes.read(crate::transaction::v1_group::v1_records);
            let group = crate::transaction::wal_buffer::build_group(
                records,
                &[
                    WalRecord::TransactionCommit {
                        transaction_id: transaction,
                    },
                    WalRecord::EpochAdvance { epoch },
                ],
            );
            if let Err(e) = wal.log_batch(&group) {
                grafeo_common::grafeo_warn!("Failed to write a direct write to the WAL: {}", e);
            }
        }

        #[cfg(feature = "cdc")]
        if self.cdc_active() && !changes.is_empty() {
            let events = changes.read(|set| crate::cdc::events_for_changes(set, &self.cdc_log));
            self.cdc_log
                .record_batch(crate::cdc::fold_into_creates(events).into_iter().map(
                    |mut event| {
                        event.epoch = epoch;
                        event
                    },
                ));
        }

        // The database has one epoch: the root store follows every commit
        // (the stamp moved the written store's).
        let root = self.lpg_store();
        root.sync_epoch(epoch);
        commit.complete();

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
            changes.undo_after(changes.start(), false)
        };
        let _ = self.transaction_manager.abort(transaction);
        match undone {
            Ok(_) => Ok(()),
            Err(failure) => {
                let message = format!(
                    "the rollback of a direct write could not undo its changes: {failure:?}"
                );
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
    fn store(&self) -> Result<Arc<dyn GraphStoreSearch>> {
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
                if writer.has_node(id) {
                    writer.remove_node_property(id, key)
                } else {
                    Ok(false)
                }
            },
        )
    }

    pub(crate) fn remove_edge_property(&self, id: EdgeId, key: &str) -> Result<bool> {
        self.write(|writer| {
            if writer.has_edge(id) {
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
        let epoch = self.db.read_epoch();
        Ok(self.store()?.get_node_at_epoch(id, epoch))
    }

    pub(crate) fn get_edge(&self, id: EdgeId) -> Result<Option<Edge>> {
        let epoch = self.db.read_epoch();
        Ok(self.store()?.get_edge_at_epoch(id, epoch))
    }
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
        if !writer.has_node(endpoint) {
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
    if !writer.has_node(id) {
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
    if !writer.has_edge(id) {
        return Err(OperatorError::Execution(format!(
            "edge {} does not exist",
            id.as_u64()
        )));
    }
    writer.set_edge_properties(id, &[(key.to_string(), value)], false)
}

/// Adds a label; whether the node exists and lacked it.
pub(crate) fn add_node_label(
    writer: &GraphWriter,
    id: NodeId,
    label: &str,
) -> std::result::Result<bool, OperatorError> {
    if !writer.has_node(id) {
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
    if !writer.has_node(id) {
        return Ok(false);
    }
    Ok(writer.remove_labels(id, &[label.to_string()])? == 1)
}

/// Creates one node with `label` per vector, the vector as `property`.
pub(crate) fn create_vector_nodes(
    writer: &GraphWriter,
    label: &str,
    property: &str,
    vectors: Vec<Vec<f32>>,
) -> std::result::Result<Vec<NodeId>, OperatorError> {
    let labels = [label.to_string()];
    vectors
        .into_iter()
        .map(|vector| {
            writer.create_node(
                &labels,
                vec![(property.to_string(), Value::Vector(vector.into()))],
            )
        })
        .collect()
}

/// Creates one node with all of `labels` per property map.
pub(crate) fn create_nodes(
    writer: &GraphWriter,
    labels: &[&str],
    properties_list: Vec<HashMap<PropertyKey, Value>>,
) -> std::result::Result<Vec<NodeId>, OperatorError> {
    let labels: Vec<String> = labels.iter().map(|label| (*label).to_string()).collect();
    properties_list
        .into_iter()
        .map(|properties| writer.create_node(&labels, direct_properties(properties)))
        .collect()
}

/// Creates the edges, each between two nodes the writer sees.
pub(crate) fn create_edges(
    writer: &GraphWriter,
    edges: Vec<BatchEdge>,
) -> std::result::Result<Vec<EdgeId>, OperatorError> {
    edges
        .into_iter()
        .map(|edge| {
            create_edge(
                writer,
                edge.src,
                edge.dst,
                &edge.edge_type,
                direct_properties(edge.properties),
            )
        })
        .collect()
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
    OperatorError::Execution(format!("node {} does not exist", id.as_u64()))
}

/// The error for a graph handle whose graph does not exist.
pub(crate) fn missing_graph(name: &str) -> Error {
    Error::Query(QueryError::new(
        QueryErrorKind::Semantic,
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

    /// A batch that panics leaves nothing: its versions are gone, and the
    /// next call writes none of its WAL records or change events. A single
    /// call writes in place, so what it wrote before the panic is committed,
    /// as after an error: the WAL matches memory. The calls run in a child
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
        assert_eq!(
            labels.rows(),
            [[Value::from("Person")], [Value::from("Single")]]
        );
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
            db.get_node(batch).is_none(),
            "the batch's node is discarded"
        );
        assert!(db.get_node(single).is_some());
        let events: Vec<EntityId> = db
            .changes_between(EpochId::new(0), db.current_epoch())
            .unwrap()
            .into_iter()
            .map(|event| event.entity_id)
            .collect();
        assert_eq!(events, [EntityId::Node(single), EntityId::Node(alix)]);
    }
}
