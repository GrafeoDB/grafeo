//! Transaction manager.

use std::collections::VecDeque;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

use grafeo_common::types::{EdgeId, EpochId, NodeId, Timestamp, TransactionId};
use grafeo_common::utils::error::{Error, Result, TransactionError};
use grafeo_common::utils::hash::{FxHashMap, FxHashSet};
use grafeo_core::execution::operators::WriteInProgress;
use parking_lot::{Condvar, Mutex, MutexGuard, RwLock, RwLockWriteGuard};

use super::changes::TransactionChanges;
use super::history::{History, HistoryImage};
use crate::config::HistoryRetention;

/// State of a transaction.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum TransactionState {
    /// Transaction is active.
    Active,
    /// The commit is decided (its epoch is assigned) and its versions, events
    /// and WAL records are being written; it becomes `Committed` when that is
    /// done. Its writes still conflict with other transactions' writes.
    Committing,
    /// Transaction is committed.
    Committed,
    /// Transaction is aborted.
    Aborted,
}

/// Transaction isolation level.
///
/// Controls the consistency guarantees and performance tradeoffs for transactions.
///
/// # Comparison
///
/// | Level | Dirty Reads | Non-Repeatable Reads | Phantom Reads | Write Skew |
/// |-------|-------------|----------------------|---------------|------------|
/// | ReadCommitted | No | Yes | Yes | Yes |
/// | SnapshotIsolation | No | No | No | Yes |
/// | Serializable | No | No | No | No |
///
/// # Performance
///
/// Higher isolation levels require more bookkeeping:
/// - `ReadCommitted`: Only tracks writes
/// - `SnapshotIsolation`: Tracks writes + snapshot versioning
/// - `Serializable`: Tracks writes + reads + SSI validation
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[non_exhaustive]
pub enum IsolationLevel {
    /// Read Committed: sees only committed data, but may see different
    /// versions of the same row within a transaction.
    ///
    /// Lowest overhead, highest throughput, but weaker consistency.
    ReadCommitted,

    /// Snapshot Isolation (default): each transaction sees a consistent
    /// snapshot as of transaction start. Prevents non-repeatable reads
    /// and phantom reads.
    ///
    /// Vulnerable to write skew anomaly.
    #[default]
    SnapshotIsolation,

    /// Serializable Snapshot Isolation (SSI): provides full serializability
    /// by detecting read-write conflicts in addition to write-write conflicts.
    ///
    /// Prevents all anomalies including write skew, but may abort more
    /// transactions due to stricter conflict detection.
    Serializable,
}

/// Entity identifier for write tracking.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum EntityId {
    /// A node.
    Node(NodeId),
    /// An edge.
    Edge(EdgeId),
}

impl From<NodeId> for EntityId {
    fn from(id: NodeId) -> Self {
        Self::Node(id)
    }
}

impl From<EdgeId> for EntityId {
    fn from(id: EdgeId) -> Self {
        Self::Edge(id)
    }
}

/// A node or edge of one graph: the unit of conflict detection.
///
/// Named graphs number their nodes and edges on their own, so the same ID
/// in two graphs is two entities, and writes to them never conflict.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct GraphEntity {
    /// The graph's storage key; `None` for the default graph.
    pub graph: Option<Arc<str>>,
    /// The node or edge.
    pub entity: EntityId,
}

impl GraphEntity {
    /// An entity of the graph with storage key `graph` (`None`: the
    /// default graph).
    #[must_use]
    pub fn new(graph: Option<Arc<str>>, entity: impl Into<EntityId>) -> Self {
        Self {
            graph,
            entity: entity.into(),
        }
    }
}

impl From<EntityId> for GraphEntity {
    fn from(entity: EntityId) -> Self {
        Self::new(None, entity)
    }
}

impl From<NodeId> for GraphEntity {
    fn from(id: NodeId) -> Self {
        Self::new(None, id)
    }
}

impl From<EdgeId> for GraphEntity {
    fn from(id: EdgeId) -> Self {
        Self::new(None, id)
    }
}

impl std::fmt::Display for GraphEntity {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self.entity)?;
        if let Some(graph) = &self.graph {
            write!(f, " in graph '{graph}'")?;
        }
        Ok(())
    }
}

/// Information about an active transaction.
pub struct TransactionInfo {
    /// Transaction state.
    pub state: TransactionState,
    /// Isolation level for this transaction.
    pub isolation_level: IsolationLevel,
    /// Start epoch (snapshot epoch for reads).
    pub start_epoch: EpochId,
    /// Set of entities written by this transaction.
    pub write_set: FxHashSet<GraphEntity>,
    /// Set of entities read by this transaction (for serializable isolation).
    pub read_set: FxHashSet<GraphEntity>,
    /// The nodes this transaction deleted (in `write_set` too): another
    /// transaction's edge to one of them conflicts with the delete.
    pub delete_set: FxHashSet<GraphEntity>,
    /// The nodes the edges this transaction created end at, claimed against
    /// a delete by another transaction (see
    /// [`TransactionManager::record_endpoints`]).
    pub endpoint_set: FxHashSet<GraphEntity>,
    /// What the transaction changed, while it is open or committing: a
    /// checkpoint reads the committed state of what open transactions
    /// changed from it (see [`TransactionManager::open_change_sets`]).
    /// Dropped once the transaction is committed or aborted.
    pub(crate) changes: Option<Arc<TransactionChanges>>,
    /// Whether this is a private transaction (a direct call, see
    /// [`TransactionManager::begin_private`]): nobody asks for its state, so
    /// it leaves the manager as soon as no other transaction can conflict
    /// with it.
    private: bool,
}

impl TransactionInfo {
    /// Creates a new transaction info with the given isolation level.
    fn new(
        start_epoch: EpochId,
        isolation_level: IsolationLevel,
        changes: Arc<TransactionChanges>,
        private: bool,
    ) -> Self {
        Self {
            state: TransactionState::Active,
            isolation_level,
            start_epoch,
            write_set: FxHashSet::default(),
            read_set: FxHashSet::default(),
            delete_set: FxHashSet::default(),
            endpoint_set: FxHashSet::default(),
            changes: Some(changes),
            private,
        }
    }

    /// Whether `claim` of `entity` by another transaction conflicts with
    /// this transaction's writes and claims, and why.
    fn conflict_with(&self, claim: Claim, entity: &GraphEntity) -> Option<String> {
        let written = || self.write_set.contains(entity);
        match claim {
            Claim::Write if written() => Some(format!("Write-write conflict on entity {entity}")),
            Claim::Delete if written() => Some(format!("Write-write conflict on entity {entity}")),
            Claim::Delete if self.endpoint_set.contains(entity) => Some(format!(
                "Write conflict on entity {entity}: another transaction creates an edge to the \
                 node this transaction deletes"
            )),
            Claim::Endpoint if self.delete_set.contains(entity) => Some(format!(
                "Write conflict on entity {entity}: another transaction deletes the node this \
                 transaction creates an edge to"
            )),
            _ => None,
        }
    }
}

/// How a transaction claims an entity for conflict detection
/// (first-writer-wins between open transactions, and at commit against the
/// transactions that committed after it began).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Claim {
    /// A write: conflicts with another transaction's writes (deletes
    /// included).
    Write,
    /// The delete of a node: a write, which also conflicts with another
    /// transaction's claim on the node as an edge's endpoint.
    Delete,
    /// The endpoint of an edge the transaction creates: conflicts only with
    /// another transaction's delete of the node.
    Endpoint,
}

/// Manages transactions and MVCC versioning.
///
/// # A commit that does not complete
///
/// A commit stamps its versions with its epoch before the epoch is published
/// (see `CommitGuard`). When the commit code panics in between, the stamped
/// versions stay invisible only as long as no higher epoch is published: a
/// later commit would publish its own epoch, and with it the stamped part of
/// the failed commit. So the first commit that does not complete poisons the
/// manager: from then on every commit fails with an error saying a commit did
/// not complete and the database must be reopened, and so does everything
/// that would persist or copy the store, which holds the stamped versions
/// (checkpoints, saves, backups and copies of the database call
/// [`check_no_incomplete_commit`](Self::check_no_incomplete_commit)).
/// Transactions still begin, and reads see every epoch published before the
/// failed commit.
pub struct TransactionManager {
    /// Next transaction ID.
    next_transaction_id: AtomicU64,
    /// The last epoch handed out: a commit takes the next one in
    /// [`start_commit`](Self::start_commit).
    assigned_epoch: AtomicU64,
    /// The last epoch whose commit is complete, the one readers see
    /// ([`current_epoch`](Self::current_epoch)). It trails `assigned_epoch`
    /// while a commit is being written, so no snapshot holds part of one.
    published_epoch: AtomicU64,
    /// Number of currently active transactions (for fast-path conflict skip).
    active_count: AtomicU64,
    /// Active transactions.
    transactions: RwLock<FxHashMap<TransactionId, TransactionInfo>>,
    /// Committed transaction epochs (for conflict detection).
    /// Maps TransactionId -> commit epoch.
    committed_epochs: RwLock<FxHashMap<TransactionId, EpochId>>,
    /// Held by a direct call that commits at once while no transaction is
    /// open (see [`idle_gate`](Self::idle_gate)), and briefly by every
    /// [`begin`](Self::begin) and [`begin_private`](Self::begin_private), so
    /// no transaction starts while such a call runs.
    ///
    /// Lock order: the idle gate before the commit lock (`begin` and such a
    /// call take both, in that order).
    idle_gate: Mutex<()>,
    /// Held by a commit from its epoch until it is complete (see
    /// [`CommitGuard`]), by a change outside a commit (a schema statement, a
    /// graph command, a direct call that commits at once) while it runs, by
    /// a checkpoint or a copy of the store
    /// while it builds its image (see [`hold_commits`](Self::hold_commits)),
    /// and briefly by every [`begin`](Self::begin) and
    /// [`begin_private`](Self::begin_private) and by
    /// [`close_for_writes`](Self::close_for_writes): commits complete one at a
    /// time, in epoch order, no transaction starts in the middle of one, and
    /// no image holds part of one.
    ///
    /// Lock order: a checkpoint takes the file's checkpoint guard before this
    /// lock, and the idle gate comes before it too; the write freeze comes
    /// after it. Nothing that holds it waits for
    /// the checkpoint timer thread (`close()` stops the timer before it takes
    /// it).
    commit_lock: Mutex<()>,
    /// The write freeze: held shared by every store change of an open
    /// transaction while it runs (a write, see
    /// [`write_in_progress`](Self::write_in_progress), and the undo of a
    /// rollback or a rollback to a savepoint), and exclusively by a
    /// checkpoint or a copy of the store while it builds its image (see
    /// [`hold_commits`](Self::hold_commits)): they read a store and change
    /// logs that do not move. Changes outside
    /// any commit ([`hold_commits_for_change`](Self::hold_commits_for_change))
    /// do not take it: they hold the commit lock, which already keeps every
    /// checkpoint out.
    ///
    /// Lock order: the commit lock before this one, so a checkpoint that
    /// holds the commit lock waits here only for store changes in progress,
    /// which never wait for the commit lock, the checkpoint guard or anything
    /// else a checkpoint holds. It is not reentrant (a fair lock): no thread
    /// asks for it while it holds it, in either mode, because a checkpoint
    /// waiting in between would block the second request.
    write_freeze: RwLock<()>,
    /// Set when a commit did not complete (its [`CommitGuard`] was dropped
    /// without `complete`): no commit can run afterwards.
    poisoned: AtomicBool,
    /// Set by [`close_for_writes`](Self::close_for_writes) when a persistent
    /// database closes: no commit and no write outside a transaction can run
    /// afterwards.
    closed: AtomicBool,
    /// The lowest epoch that is never published: the epoch of a commit that
    /// did not complete (nothing at or above it is published, see
    /// [`Publish`]), 0 after a broken invariant ([`poison`](Self::poison)),
    /// `u64::MAX` otherwise.
    unpublishable_from: AtomicU64,
    /// Whose turn it is to publish: held while a commit publishes its epoch,
    /// and while a waiter checks the published epoch, so no wake-up is lost.
    ///
    /// Lock order: after the commit lock (a hold waits for the written
    /// commits while it holds the commit lock), before the `transactions`
    /// lock.
    publish_turn: Mutex<()>,
    /// Signalled when an epoch is published, and when a commit does not
    /// complete.
    published: Condvar,
    /// The readers that hold the read horizon back, by the epoch each reads
    /// at: the snapshot of every open and committing transaction, and every
    /// read outside a transaction that runs now (see
    /// [`pin_read`](Self::pin_read)). The read horizon is its oldest epoch,
    /// or the published epoch when it holds none, so it never walks the
    /// finished transactions.
    ///
    /// Lock order: after the `transactions` lock (a transaction's snapshot
    /// is counted and released while that lock is held); the read horizon,
    /// a pin and its drop take this one alone, and the collect horizon takes
    /// the history's locks after it.
    reads: Arc<ReadPins>,
    /// The history: its retention, the commit times and the floor of what a
    /// read may ask for (see `history`).
    history: History,
}

/// The epochs the readers that hold the read horizon back read at, each with
/// how many such readers: open and committing transactions by their
/// snapshot, reads outside a transaction by the epoch they pinned.
///
/// Ascending, each epoch once with a count above 0: every reader is counted
/// at the published epoch, read under the lock, which only moves forward, so
/// a new epoch goes at the back. The oldest is the front, and the buffer is
/// kept from one reader to the next (a reader that begins and ends allocates
/// nothing).
#[derive(Debug, Default)]
struct ReadPins {
    epochs: Mutex<VecDeque<(u64, usize)>>,
}

impl ReadPins {
    /// Counts a reader at the published epoch, read while the lock is held,
    /// and returns that epoch: the read horizon reads the published epoch
    /// under the same lock, so it never passes a reader it has not counted.
    fn hold_published(&self, published: &AtomicU64) -> EpochId {
        let mut epochs = self.epochs.lock();
        let epoch = published.load(Ordering::Acquire);
        match epochs.back_mut() {
            Some((newest, readers)) if *newest == epoch => *readers += 1,
            _ => {
                debug_assert!(
                    epochs.back().is_none_or(|(newest, _)| *newest < epoch),
                    "the published epoch only moves forward"
                );
                epochs.push_back((epoch, 1));
            }
        }
        EpochId::new(epoch)
    }

    /// Counts a reader at `epoch`, at or before the published epoch, in its
    /// place among the readers (ascending).
    fn hold_at(epochs: &mut VecDeque<(u64, usize)>, epoch: u64) {
        match epochs.binary_search_by_key(&epoch, |(at, _)| *at) {
            Ok(index) => epochs[index].1 += 1,
            Err(index) => epochs.insert(index, (epoch, 1)),
        }
    }

    /// Releases one reader at `epoch`.
    fn release(&self, epoch: EpochId) {
        let mut epochs = self.epochs.lock();
        if let Ok(index) = epochs.binary_search_by_key(&epoch.as_u64(), |(at, _)| *at) {
            let readers = &mut epochs[index].1;
            *readers -= 1;
            if *readers == 0 {
                epochs.remove(index);
            }
        }
    }

    /// The oldest epoch a reader reads at, `None` with no reader.
    fn oldest(epochs: &VecDeque<(u64, usize)>) -> Option<u64> {
        epochs.front().map(|(epoch, _)| *epoch)
    }
}

/// A read outside any transaction, pinned at the epoch it reads at from
/// [`TransactionManager::pin_read`] until it is dropped (also by the unwind
/// of a panic in the read): the read horizon stays at that epoch or before,
/// so garbage collection keeps every version the read may still need (#412
/// decision Q6: every read statement outside a transaction pins its epoch
/// for its run).
#[must_use = "a read is pinned only while the pin lives"]
pub(crate) struct ReadPin {
    pins: Arc<ReadPins>,
    epoch: EpochId,
}

impl ReadPin {
    /// The epoch the read is pinned at: the published epoch when it was
    /// pinned.
    #[cfg(test)]
    #[must_use]
    pub(crate) fn epoch(&self) -> EpochId {
        self.epoch
    }
}

impl std::fmt::Debug for ReadPin {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ReadPin")
            .field("epoch", &self.epoch)
            .finish_non_exhaustive()
    }
}

impl Drop for ReadPin {
    fn drop(&mut self) {
        self.pins.release(self.epoch);
    }
}

/// Commits held off (see [`TransactionManager::hold_commits`]): while this
/// lives, no commit is between its epoch and its completion, and no commit,
/// transaction start or write outside a transaction can begin. Held for a
/// checkpoint or a copy of the store, it also freezes the store: no store
/// change of an open transaction is in progress or can begin.
#[must_use = "commits are held off only while the guard lives"]
pub(crate) struct CommitsHeld<'a> {
    // Fields drop in order: the freeze is released before the commit lock,
    // the reverse of the order they are taken in.
    _writes: Option<RwLockWriteGuard<'a, ()>>,
    _commit: MutexGuard<'a, ()>,
}

/// A commit in progress, from [`TransactionManager::start_commit`] until it
/// is published. Until then the transaction is
/// [`TransactionState::Committing`]: it still counts as open, its writes
/// still conflict with other transactions' writes, and readers do not see
/// its epoch yet.
///
/// The guard holds the commit lock: no other commit takes an epoch, and no
/// [`begin`](TransactionManager::begin) runs. It ends in one of three ways:
///
/// - [`abort`](Self::abort): the commit's log write failed and nothing of it
///   remains. The epoch goes back, and the transaction is active again for
///   the caller's rollback.
/// - [`written`](Self::written): the commit's log group is written. The
///   commit lock is released, so the next commit starts, and the returned
///   [`Publish`] publishes this one in its turn.
/// - [`complete`](Self::complete): both at once, while the lock is held.
///
/// A guard dropped without any of them (only when the commit code panics)
/// does not report the transaction as committed and does not publish its
/// epoch: it stays `Committing`, so its half-written entities stay locked
/// against other writers, and only the commit lock is released. It also
/// poisons the manager: versions the commit already stamped with its epoch
/// would become visible with the epoch of any later commit, so every later
/// [`start_commit`](TransactionManager::start_commit) fails until the
/// database is reopened (see [`TransactionManager`]).
#[must_use = "a commit ends with `complete()`, `written()` or `abort()`"]
pub(crate) struct CommitGuard<'a> {
    // Fields drop in order: an unfinished commit poisons the manager before
    // the commit lock is released, so a commit waiting for the lock sees it.
    commit: Publish<'a>,
    _commit: MutexGuard<'a, ()>,
}

impl CommitGuard<'_> {
    /// The commit epoch.
    pub(crate) fn epoch(&self) -> EpochId {
        self.commit.epoch
    }

    /// When the commit happened, while the database keeps history (`None`
    /// otherwise): its WAL group carries this time, so a replay records the
    /// time the live commit recorded.
    #[cfg(feature = "wal")]
    pub(crate) fn commit_time(&self) -> Option<u64> {
        self.commit.commit_time
    }

    /// Publishes the commit while the commit lock is held: the transaction is
    /// committed, readers see its epoch, and what waited for it can run.
    ///
    /// # Errors
    ///
    /// As [`Publish::complete`].
    pub(crate) fn complete(self) -> Result<()> {
        let Self { commit, _commit } = self;
        commit.complete()
    }

    /// Takes the commit back before anything of it is written or stamped
    /// (its log write failed and was cut back): the epoch is free again,
    /// which is safe because the commit lock is still held and no later
    /// commit took one, and the transaction is active again, with its claims,
    /// for the caller to roll back. Nothing is published or poisoned.
    #[cfg(any(test, feature = "wal"))]
    pub(crate) fn abort(self) {
        let Self {
            mut commit,
            _commit,
        } = self;
        let manager = commit.manager;
        {
            let mut transactions = manager.transactions.write();
            let mut committed = manager.committed_epochs.write();
            if let Some(info) = transactions.get_mut(&commit.transaction_id) {
                info.state = TransactionState::Active;
            }
            committed.remove(&commit.transaction_id);
            // The last epoch handed out is this commit's (the commit lock).
            manager
                .assigned_epoch
                .store(commit.epoch.as_u64() - 1, Ordering::SeqCst);
        }
        commit.finished = true;
    }
}

impl<'a> CommitGuard<'a> {
    /// The commit's group is in the log: releases the commit lock, so the
    /// next commit can take its epoch and write, and returns the handle that
    /// publishes this one after every commit below it.
    pub(crate) fn written(self) -> Publish<'a> {
        let Self { commit, _commit } = self;
        commit
    }
}

/// A written commit that waits to be published (see
/// [`CommitGuard::written`]): its log group is written, the commit lock is
/// released, and its transaction is still [`TransactionState::Committing`].
///
/// Commits publish in the order of their epochs, so a reader that sees an
/// epoch sees every commit below it: [`complete`](Self::complete) waits for
/// the commit below this one.
///
/// A handle dropped without `complete` is a commit that did not complete, as
/// a dropped [`CommitGuard`] is: the manager is poisoned, and neither this
/// epoch nor any above it is published.
#[must_use = "a written commit is committed only after `complete()`"]
pub(crate) struct Publish<'a> {
    manager: &'a TransactionManager,
    transaction_id: TransactionId,
    epoch: EpochId,
    /// When the commit happened, while the database keeps history: recorded
    /// in the time index as it publishes, and the time of its WAL group.
    commit_time: Option<u64>,
    /// Whether the commit was published or taken back.
    finished: bool,
}

impl Publish<'_> {
    /// The commit epoch.
    #[cfg(test)]
    pub(crate) fn epoch(&self) -> EpochId {
        self.epoch
    }

    /// Publishes the commit once every commit below it is published: the
    /// transaction is committed, its claims are released, readers see its
    /// epoch, and what waited for it can run.
    ///
    /// # Errors
    ///
    /// As [`complete_with`](Self::complete_with).
    pub(crate) fn complete(self) -> Result<()> {
        self.complete_with(|| Ok(()))
    }

    /// Publishes the commit as [`complete`](Self::complete) does, once
    /// `ordered` ran in the commit's turn: after every commit below it is
    /// published and before any above it publishes. The commit's steps whose
    /// order must follow the epochs go there: stamping its versions, applying
    /// its RDF triples, reporting its change events, moving the store's epoch.
    /// `ordered` must not wait for anything that waits for a commit to be
    /// published (it holds the publish turn).
    ///
    /// # Errors
    ///
    /// [`TransactionError::IncompleteCommit`] when a commit at a lower epoch
    /// did not complete meanwhile (or the manager was poisoned): this one is
    /// then not published either (its epoch would show the other's stamped
    /// versions) and stays `Committing`, in the log as it was written, until
    /// the database is reopened. The error of `ordered`, which leaves the
    /// commit incomplete as a dropped handle does.
    pub(crate) fn complete_with(mut self, ordered: impl FnOnce() -> Result<()>) -> Result<()> {
        let manager = self.manager;
        let mut turn = manager.publish_turn.lock();
        loop {
            if self.epoch.as_u64() >= manager.unpublishable_from.load(Ordering::Acquire) {
                // Reported here: the drop of this handle says nothing more.
                self.finished = true;
                return Err(Error::Transaction(TransactionError::IncompleteCommit));
            }
            if manager.published_epoch.load(Ordering::Acquire) + 1 >= self.epoch.as_u64() {
                break;
            }
            manager.published.wait(&mut turn);
        }
        if let Err(error) = ordered() {
            // Not published: the handle, dropped unfinished, poisons the
            // manager once the turn is free.
            drop(turn);
            return Err(error);
        }
        {
            let mut transactions = manager.transactions.write();
            if let Some(info) = transactions.get_mut(&self.transaction_id) {
                info.state = TransactionState::Committed;
                // Stamped: what it changed is committed state now.
                info.changes = None;
                // It reads no more: its snapshot holds the horizon no more.
                manager.reads.release(info.start_epoch);
                // A private transaction no open one began before: only a
                // transaction that began before this commit was published
                // can conflict with it, and each is counted while the
                // `transactions` lock is held (see `register`), so its
                // claims are of no further use.
                if info.private && manager.active_count.load(Ordering::Acquire) == 1 {
                    transactions.remove(&self.transaction_id);
                    manager
                        .committed_epochs
                        .write()
                        .remove(&self.transaction_id);
                }
            }
            // Counted down under the `transactions` lock, like the count up
            // of a transaction that begins.
            manager.active_count.fetch_sub(1, Ordering::Release);
            // Commits publish one at a time and in epoch order (the turn), so
            // the published epoch only moves forward.
            manager
                .published_epoch
                .fetch_max(self.epoch.as_u64(), Ordering::Release);
        }
        // In the turn, so the times are recorded in epoch order.
        if let Some(time) = self.commit_time {
            manager.history.record(self.epoch, time);
        }
        self.finished = true;
        drop(turn);
        manager.published.notify_all();
        Ok(())
    }
}

impl Drop for Publish<'_> {
    fn drop(&mut self) {
        if self.finished {
            return;
        }
        grafeo_common::grafeo_error!(
            "commit of transaction {:?} at epoch {:?} did not complete; its writes stay \
             locked, and no transaction can commit until the database is reopened",
            self.transaction_id,
            self.epoch
        );
        self.manager.incomplete_from(self.epoch);
    }
}

impl TransactionManager {
    /// Creates a new transaction manager.
    #[must_use]
    pub fn new() -> Self {
        Self {
            // Start at 2 to avoid collision with TransactionId::SYSTEM (which is 1)
            // TransactionId::INVALID = u64::MAX, TransactionId::SYSTEM = 1, user transactions start at 2
            next_transaction_id: AtomicU64::new(2),
            assigned_epoch: AtomicU64::new(0),
            published_epoch: AtomicU64::new(0),
            active_count: AtomicU64::new(0),
            transactions: RwLock::new(FxHashMap::default()),
            committed_epochs: RwLock::new(FxHashMap::default()),
            idle_gate: Mutex::new(()),
            commit_lock: Mutex::new(()),
            write_freeze: RwLock::new(()),
            poisoned: AtomicBool::new(false),
            closed: AtomicBool::new(false),
            unpublishable_from: AtomicU64::new(u64::MAX),
            publish_turn: Mutex::new(()),
            published: Condvar::new(),
            reads: Arc::new(ReadPins::default()),
            history: History::default(),
        }
    }

    /// A manager whose commits take their times from `clock` (milliseconds
    /// since the Unix epoch) once history is on: for tests of the retention
    /// window.
    #[cfg(test)]
    pub(crate) fn with_history_clock(clock: super::history::Clock) -> Self {
        Self {
            history: History::with_clock(clock),
            ..Self::new()
        }
    }

    /// Begins a new transaction with the default isolation level (Snapshot Isolation).
    pub fn begin(&self) -> TransactionId {
        self.begin_with_isolation(IsolationLevel::default())
    }

    /// Begins a new transaction with the specified isolation level.
    pub fn begin_with_isolation(&self, isolation_level: IsolationLevel) -> TransactionId {
        // Wait for a direct call that commits at once (it needs no open
        // transaction), not for a commit in progress: the snapshot is the
        // published epoch, which holds every commit up to it, complete, and
        // none of one still being written or synced.
        let _gate = self.idle_gate.lock();
        self.register(isolation_level, false).0
    }

    /// Exclusive access for a direct call that commits at once: `Some` while
    /// no transaction is open. No transaction can begin until the guard is
    /// dropped, so such a call cannot conflict with one and needs no
    /// claims; nor with another such call, which waits for the guard.
    pub(crate) fn idle_gate(&self) -> Option<MutexGuard<'_, ()>> {
        let gate = self.idle_gate.lock();
        // Acquire pairs with the release of a commit's completion: the call
        // sees everything the last commit wrote.
        (self.active_count.load(Ordering::Acquire) == 0).then_some(gate)
    }

    /// Begins a private transaction: one without a session, for a direct
    /// call of the database or a graph handle while a transaction is open
    /// (or a batch), committed or rolled back by the call itself. Like
    /// [`begin`](Self::begin) it waits for a direct call holding the idle
    /// gate, not for a commit in progress: it reads at the published epoch.
    /// It does not wait for open transactions either: its writes are claimed
    /// like any transaction's, so it conflicts with an open (or committing)
    /// transaction that wrote the same entity first, and the other way round.
    ///
    /// # Errors
    ///
    /// Fails once the database is closed for writes (see
    /// [`check_open`](Self::check_open)): the call's commit would fail.
    pub(crate) fn begin_private(&self) -> Result<(TransactionId, Arc<TransactionChanges>)> {
        let _gate = self.idle_gate.lock();
        self.check_open()?;
        Ok(self.register(IsolationLevel::default(), true))
    }

    /// The writer of a bulk write that holds commits off for its whole run
    /// (an import, see
    /// [`StreamingChanges`](crate::transaction::StreamingChanges)): a
    /// transaction id no other transaction has, reading at the published
    /// epoch. Not registered: the bulk write creates only new nodes and
    /// edges, which no other transaction sees before it commits, so none
    /// conflicts with it; and it commits or rolls back itself while it holds
    /// commits off (`_commits`), so no checkpoint, copy or other commit sees
    /// it halfway.
    pub(crate) fn bulk_writer(&self, _commits: &CommitsHeld<'_>) -> (TransactionId, EpochId) {
        let transaction_id =
            TransactionId::new(self.next_transaction_id.fetch_add(1, Ordering::Relaxed));
        (transaction_id, self.current_epoch())
    }

    /// Registers a new active transaction reading at the published epoch,
    /// with its (empty) change set.
    fn register(
        &self,
        isolation_level: IsolationLevel,
        private: bool,
    ) -> (TransactionId, Arc<TransactionChanges>) {
        let transaction_id =
            TransactionId::new(self.next_transaction_id.fetch_add(1, Ordering::Relaxed));
        // Counted while the `transactions` lock is held: a commit that
        // publishes meanwhile sees this transaction or does not, never half
        // (it drops a private transaction's claims only when no other is
        // counted, see `Publish::complete`). The snapshot is read and held
        // under the readers' lock, so the read horizon, which reads the
        // published epoch under it, never passes a snapshot it has not seen.
        let mut transactions = self.transactions.write();
        let epoch = self.reads.hold_published(&self.published_epoch);
        let changes = Arc::new(TransactionChanges::new(transaction_id, epoch));
        let info = TransactionInfo::new(epoch, isolation_level, Arc::clone(&changes), private);
        transactions.insert(transaction_id, info);
        self.active_count.fetch_add(1, Ordering::Release);
        drop(transactions);
        (transaction_id, changes)
    }

    /// What transaction `transaction_id` changed so far, while it is open.
    pub(crate) fn changes(&self, transaction_id: TransactionId) -> Option<Arc<TransactionChanges>> {
        self.transactions
            .read()
            .get(&transaction_id)
            .and_then(|info| info.changes.clone())
    }

    /// The change sets of the open transactions, for a checkpoint or a copy
    /// of the store that holds commits off (`_commits`): no commit is in
    /// progress, and no write or rollback of an open transaction, so the sets
    /// and the stores hold the same changes until the hold is released.
    pub(crate) fn open_change_sets(
        &self,
        _commits: &CommitsHeld<'_>,
    ) -> Vec<Arc<TransactionChanges>> {
        self.transactions
            .read()
            .values()
            .filter(|info| info.state == TransactionState::Active)
            .filter_map(|info| info.changes.clone())
            .collect()
    }

    /// Poisons the manager after a broken invariant that leaves the store
    /// in a state no commit may build on (a change applied but not
    /// recorded, an undo that failed): from now on every commit and
    /// checkpoint fails, as after a commit that did not complete.
    pub(crate) fn poison(&self, reason: &str) {
        grafeo_common::grafeo_error!(
            "{reason}; no transaction can commit until the database is reopened"
        );
        // Nothing more is published: no commit may build on this store.
        self.incomplete_from(EpochId::new(0));
    }

    /// Records that the commit at `epoch` did not complete: the manager is
    /// poisoned, no epoch at or above it is published, and everything that
    /// waits for an epoch to be published is woken to see it.
    fn incomplete_from(&self, epoch: EpochId) {
        self.poisoned.store(true, Ordering::Release);
        self.unpublishable_from
            .fetch_min(epoch.as_u64(), Ordering::AcqRel);
        self.wake_publish_waiters();
    }

    /// Wakes whatever waits for an epoch to be published. Taking the turn
    /// first means a waiter is either before its check, and sees the new
    /// state, or already waiting, and is woken.
    fn wake_publish_waiters(&self) {
        drop(self.publish_turn.lock());
        self.published.notify_all();
    }

    /// Waits until every written commit is published (see [`Publish`]):
    /// called with the commit lock held, so no further commit is written
    /// meanwhile. Returns at once when a commit did not complete: the
    /// commits at and above it are never published.
    fn wait_for_written_commits(&self) {
        let mut turn = self.publish_turn.lock();
        while !self.has_incomplete_commit()
            && self.published_epoch.load(Ordering::Acquire)
                < self.assigned_epoch.load(Ordering::Acquire)
        {
            self.published.wait(&mut turn);
        }
    }

    /// Returns the isolation level of a transaction.
    pub fn isolation_level(&self, transaction_id: TransactionId) -> Option<IsolationLevel> {
        self.transactions
            .read()
            .get(&transaction_id)
            .map(|info| info.isolation_level)
    }

    /// Records a write operation for the transaction.
    ///
    /// Uses first-writer-wins: if another active transaction has already
    /// written to the same entity, returns a write-write conflict error
    /// immediately (before the caller mutates the store).
    ///
    /// # Errors
    ///
    /// Returns an error if the transaction is not active or if another
    /// active transaction has already written to the same entity.
    pub fn record_write(
        &self,
        transaction_id: TransactionId,
        entity: impl Into<GraphEntity>,
    ) -> Result<()> {
        self.record_claims(transaction_id, Claim::Write, [entity.into()])
    }

    /// Records the delete of a node: a write, as
    /// [`record_write`](Self::record_write) records one, that also conflicts
    /// with an open transaction that creates an edge to the node (see
    /// [`record_endpoints`](Self::record_endpoints)), first writer wins; at
    /// commit, with one that did and committed after this one began.
    ///
    /// # Errors
    ///
    /// Returns an error if the transaction is not active, or if another open
    /// transaction wrote the node or claimed it as an edge's endpoint.
    pub fn record_delete(
        &self,
        transaction_id: TransactionId,
        entity: impl Into<GraphEntity>,
    ) -> Result<()> {
        self.record_claims(transaction_id, Claim::Delete, [entity.into()])
    }

    /// Claims the endpoints of an edge the transaction is about to create,
    /// so no committed edge ends at a deleted node: an open transaction that
    /// deletes one of them conflicts with this one, first writer wins, and at
    /// commit so does one that deleted one and committed after this one
    /// began. A claim does not conflict with other writes of the node or
    /// with other claims on it: transactions that create edges to one node,
    /// or set its properties, go on side by side.
    ///
    /// # Errors
    ///
    /// Returns an error if the transaction is not active, or if another open
    /// transaction deletes one of the endpoints; nothing is claimed then.
    pub fn record_endpoints(
        &self,
        transaction_id: TransactionId,
        endpoints: [GraphEntity; 2],
    ) -> Result<()> {
        self.record_claims(transaction_id, Claim::Endpoint, endpoints)
    }

    /// Records `claim` of each of `entities` for `transaction_id`, after
    /// checking it against the writes and claims of the other open
    /// transactions (first-writer-wins): all of them, or none on a conflict.
    /// After a commit that did not complete nothing can commit, so a write
    /// fails with that error first, not with a conflict a retry cannot get
    /// past (the incomplete commit keeps its claims until the reopen).
    fn record_claims<const N: usize>(
        &self,
        transaction_id: TransactionId,
        claim: Claim,
        entities: [GraphEntity; N],
    ) -> Result<()> {
        self.check_no_incomplete_commit()?;
        let mut txns = self.transactions.write();

        // First-writer-wins conflict detection. Skip the scan when only one
        // transaction is active (common case for auto-commit). A commit in
        // progress still holds its writes.
        if self.active_count.load(Ordering::Relaxed) > 1 {
            for (other_tx, other_info) in txns.iter() {
                if *other_tx == transaction_id
                    || !matches!(
                        other_info.state,
                        TransactionState::Active | TransactionState::Committing
                    )
                {
                    continue;
                }
                if let Some(conflict) = entities
                    .iter()
                    .find_map(|entity| other_info.conflict_with(claim, entity))
                {
                    return Err(Error::Transaction(TransactionError::WriteConflict(
                        conflict,
                    )));
                }
            }
        }

        // Single lookup: get_mut for both state check and write_set insert
        let info = txns.get_mut(&transaction_id).ok_or_else(|| {
            Error::Transaction(TransactionError::InvalidState(
                "Transaction not found".to_string(),
            ))
        })?;

        if info.state != TransactionState::Active {
            return Err(Error::Transaction(TransactionError::InvalidState(
                "Transaction is not active".to_string(),
            )));
        }

        for entity in entities {
            match claim {
                Claim::Write => {
                    info.write_set.insert(entity);
                }
                Claim::Delete => {
                    info.write_set.insert(entity.clone());
                    info.delete_set.insert(entity);
                }
                Claim::Endpoint => {
                    info.endpoint_set.insert(entity);
                }
            }
        }
        Ok(())
    }

    /// Records a read operation for the transaction (for serializable isolation).
    ///
    /// # Errors
    ///
    /// Returns an error if the transaction is not active.
    pub fn record_read(
        &self,
        transaction_id: TransactionId,
        entity: impl Into<GraphEntity>,
    ) -> Result<()> {
        let mut txns = self.transactions.write();
        let info = txns.get_mut(&transaction_id).ok_or_else(|| {
            Error::Transaction(TransactionError::InvalidState(
                "Transaction not found".to_string(),
            ))
        })?;

        if info.state != TransactionState::Active {
            return Err(Error::Transaction(TransactionError::InvalidState(
                "Transaction is not active".to_string(),
            )));
        }

        info.read_set.insert(entity.into());
        Ok(())
    }

    /// Commits a transaction with conflict detection.
    ///
    /// # Conflict Detection
    ///
    /// - **All isolation levels**: Write-write conflicts (two transactions writing
    ///   to the same entity) are always detected and cause the second committer to abort.
    ///
    /// - **Serializable only**: Read-write conflicts (SSI validation) are additionally
    ///   checked. If transaction T1 read an entity that another transaction T2 wrote,
    ///   and T2 committed after T1 started, T1 will abort. This prevents write skew.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The transaction is not active
    /// - There's a write-write conflict with another committed transaction
    /// - (Serializable only) There's a read-write conflict (SSI violation)
    /// - An earlier commit did not complete (see [`TransactionManager`]): no
    ///   transaction commits until the database is reopened
    /// - The database is closed: `close()` of a persistent database started
    /// - A store fails to stamp the transaction's changes: the commit does
    ///   not complete, which poisons the manager
    ///
    /// The changes the transaction recorded (the writes of a
    /// [`QueryProcessor`](crate::query::QueryProcessor) with its transaction
    /// context) are stamped with the commit epoch, so they are visible once
    /// it is published. They are not logged: a session's commit logs its
    /// transaction.
    pub fn commit(&self, transaction_id: TransactionId) -> Result<EpochId> {
        let changes = self.changes(transaction_id);
        let commit = self.start_commit(transaction_id)?;
        let epoch = commit.epoch();
        if let Some(changes) = changes {
            // A store that fails leaves the commit half stamped: the guard,
            // dropped uncompleted, poisons the manager.
            changes.stamp(epoch).map_err(|error| {
                Error::Internal(format!(
                    "the commit of transaction {transaction_id:?} could not stamp its changes: \
                     {error}"
                ))
            })?;
        }
        commit.complete()?;
        Ok(epoch)
    }

    /// Commits a transaction like [`commit`](Self::commit), but completes the
    /// commit only when the returned guard is dropped (see [`CommitGuard`]):
    /// the caller writes the commit's versions, events and WAL records first.
    ///
    /// # Errors
    ///
    /// As [`commit`](Self::commit); the transaction then stays active.
    pub(crate) fn start_commit(&self, transaction_id: TransactionId) -> Result<CommitGuard<'_>> {
        let commit_lock = self.commit_lock.lock();
        // A commit that did not complete sets this before it releases the
        // commit lock: any epoch published after it would publish its
        // stamped versions too.
        self.check_no_incomplete_commit()?;
        self.check_open()?;
        // Lock ordering: transactions first, then committed_epochs (matches gc()).
        // Both held as write locks to ensure state and epoch are updated atomically,
        // preventing a race where another thread sees state == Committed but the
        // epoch is not yet in committed_epochs.
        let mut txns = self.transactions.write();
        let mut committed = self.committed_epochs.write();

        // First, validate the transaction exists and is active
        let ours = txns.get(&transaction_id).ok_or_else(|| {
            Error::Transaction(TransactionError::InvalidState(
                "Transaction not found".to_string(),
            ))
        })?;
        if ours.state != TransactionState::Active {
            return Err(Error::Transaction(TransactionError::InvalidState(
                "Transaction is not active".to_string(),
            )));
        }
        let our_start_epoch = ours.start_epoch;

        // Only a transaction that committed after ours began can conflict
        // with it, and each commit advances the epoch: if the epoch has not
        // moved since, there is nothing to check. (Commits hold the
        // `transactions` lock, so none can happen during the checks.)
        let others_committed =
            self.assigned_epoch.load(Ordering::Acquire) > our_start_epoch.as_u64();

        // Check for write-write conflicts with transactions that committed
        // after our snapshot (i.e., concurrent writers to the same entities).
        // Transactions committed before our start_epoch are part of our visible
        // snapshot, so overwriting their values is not a conflict.
        if others_committed && !ours.write_set.is_empty() {
            for (other_tx, commit_epoch) in committed.iter() {
                if *other_tx != transaction_id && commit_epoch.as_u64() > our_start_epoch.as_u64() {
                    // Check if that transaction wrote to any of our entities
                    if let Some(other_info) = txns.get(other_tx) {
                        for entity in &ours.write_set {
                            if other_info.write_set.contains(entity) {
                                return Err(Error::Transaction(TransactionError::WriteConflict(
                                    format!("Write-write conflict on entity {entity}"),
                                )));
                            }
                        }
                    }
                }
            }
        }

        // The same for the endpoints of our new edges and the nodes we
        // deleted: a transaction that committed after our snapshot deleted
        // a node we link to, or linked to a node we delete (neither saw the
        // other's change, so its commit would leave a dangling edge).
        if others_committed && !(ours.endpoint_set.is_empty() && ours.delete_set.is_empty()) {
            for (other_tx, commit_epoch) in committed.iter() {
                if *other_tx != transaction_id
                    && commit_epoch.as_u64() > our_start_epoch.as_u64()
                    && let Some(other_info) = txns.get(other_tx)
                {
                    let conflict = ours
                        .endpoint_set
                        .iter()
                        .find_map(|entity| other_info.conflict_with(Claim::Endpoint, entity))
                        .or_else(|| {
                            ours.delete_set
                                .iter()
                                .find_map(|entity| other_info.conflict_with(Claim::Delete, entity))
                        });
                    if let Some(conflict) = conflict {
                        return Err(Error::Transaction(TransactionError::WriteConflict(
                            conflict,
                        )));
                    }
                }
            }
        }

        // SSI validation for Serializable isolation level.
        // Check for read-write conflicts: if we read an entity that another
        // transaction (that committed after we started) wrote, we have a
        // "rw-antidependency" which can cause write skew.
        //
        // With both transactions.write() and committed_epochs.write() held,
        // no concurrent commit can insert into committed_epochs or change
        // transaction state during our validation window. A single pass over
        // committed_epochs is sufficient.
        if others_committed
            && ours.isolation_level == IsolationLevel::Serializable
            && !ours.read_set.is_empty()
        {
            for (other_tx, commit_epoch) in committed.iter() {
                if *other_tx != transaction_id && commit_epoch.as_u64() > our_start_epoch.as_u64() {
                    // Check if that transaction wrote to any entity we read
                    if let Some(other_info) = txns.get(other_tx) {
                        for entity in &ours.read_set {
                            if other_info.write_set.contains(entity) {
                                return Err(Error::Transaction(
                                    TransactionError::SerializationFailure(format!(
                                        "Read-write conflict on entity {entity}: \
                                         another transaction modified data we read"
                                    )),
                                ));
                            }
                        }
                    }
                }
            }
        }

        // Commit successful: advance epoch atomically.
        // SeqCst ensures all threads see commits in a consistent total order.
        let commit_epoch = EpochId::new(self.assigned_epoch.fetch_add(1, Ordering::SeqCst) + 1);

        // Update state and record commit epoch atomically (both write locks
        // held). The transaction stays counted as active until the guard is
        // dropped.
        if let Some(info) = txns.get_mut(&transaction_id) {
            info.state = TransactionState::Committing;
        }
        committed.insert(transaction_id, commit_epoch);

        Ok(CommitGuard {
            commit: Publish {
                manager: self,
                transaction_id,
                epoch: commit_epoch,
                // Under the commit lock: the times follow the epochs.
                commit_time: self.history.commit_time(),
                finished: false,
            },
            _commit: commit_lock,
        })
    }

    /// Holds commits off for a checkpoint or a copy of the store: while the
    /// returned guard lives, no commit is between its epoch and its
    /// completion, and no commit, transaction start or write outside a
    /// transaction can begin. It also freezes the store (see `write_freeze`):
    /// no store change of an open transaction, a write or the undo of a
    /// rollback, is in progress or can begin, so the image is read from a
    /// store and change logs that do not move. Waits for a commit in
    /// progress to complete, then for the store changes in progress.
    ///
    /// Lock order: the commit lock, then the write freeze. The calling
    /// thread must not have a write in progress (see
    /// [`write_in_progress`](Self::write_in_progress)): it would wait for
    /// itself.
    ///
    /// # Errors
    ///
    /// Fails, once the commit lock is held, when a commit did not complete
    /// (see [`check_no_incomplete_commit`](Self::check_no_incomplete_commit)):
    /// the store holds its stamped versions, which no image may contain.
    pub(crate) fn hold_commits(&self) -> Result<CommitsHeld<'_>> {
        let commit = self.commit_lock.lock();
        self.wait_for_written_commits();
        self.check_no_incomplete_commit()?;
        let writes = self.write_freeze.write();
        Ok(CommitsHeld {
            _writes: Some(writes),
            _commit: commit,
        })
    }

    /// Holds commits off for a change that takes effect at once and logs its
    /// own WAL group outside any commit (a schema statement, a graph command,
    /// an RDF update or a write outside a transaction), for as long as the
    /// guard lives: a checkpoint, a copy or `close()` sees all of it or none
    /// of it. Waits for a commit or checkpoint in progress, then fails once
    /// the database is closed (see [`check_open`](Self::check_open)) or after
    /// a commit that did not complete.
    ///
    /// It does not freeze the store (see `write_freeze`): the writes of open
    /// transactions go on meanwhile.
    ///
    /// # Errors
    ///
    /// [`TransactionError::IncompleteCommit`] or
    /// [`TransactionError::DatabaseClosed`].
    pub(crate) fn hold_commits_for_change(&self) -> Result<CommitsHeld<'_>> {
        let commit = self.commit_lock.lock();
        self.wait_for_written_commits();
        self.check_no_incomplete_commit()?;
        self.check_open()?;
        Ok(CommitsHeld {
            _writes: None,
            _commit: commit,
        })
    }

    /// Marks a store change of an open transaction as in progress, for as
    /// long as the returned guard lives: a write and its record in the
    /// transaction's change set (through the transaction's change recorder),
    /// or the undo of a rollback or a rollback to a savepoint. It waits while a
    /// checkpoint or a copy of the store holds commits off (see
    /// [`hold_commits`](Self::hold_commits)), and they wait for it.
    ///
    /// Not reentrant: the calling thread must not hold one already, nor
    /// hold commits off for a checkpoint or a copy (see `write_freeze`). It
    /// may hold the commit lock (taken before this), but nothing else a
    /// checkpoint waits for while it holds commits off.
    pub(crate) fn write_in_progress(&self) -> WriteInProgress<'_> {
        self.write_freeze.read()
    }

    /// Closes the database for writes: from now on every commit and every
    /// write outside a transaction fails (see [`check_open`](Self::check_open)).
    /// Waits for a commit in progress to complete. A persistent database
    /// calls this when it closes, before its final checkpoint: a write that
    /// ran after that checkpoint would be written only to the WAL the close
    /// removes. Transactions still begin, and reads still work.
    pub(crate) fn close_for_writes(&self) {
        let _commit = self.commit_lock.lock();
        self.wait_for_written_commits();
        // Set while the commit lock is held: a commit or a write outside a
        // transaction waiting for the lock sees it.
        self.closed.store(true, Ordering::Release);
    }

    /// Fails once the database is closed for writes (see
    /// [`close_for_writes`](Self::close_for_writes)). Every commit, and every
    /// write outside a transaction, calls this while it holds the commit
    /// lock.
    ///
    /// # Errors
    ///
    /// Returns [`TransactionError::DatabaseClosed`].
    pub(crate) fn check_open(&self) -> Result<()> {
        if self.closed.load(Ordering::Acquire) {
            return Err(Error::Transaction(TransactionError::DatabaseClosed));
        }
        Ok(())
    }

    /// Whether a commit did not complete (see [`TransactionManager`]): its
    /// stamped versions are in the store, so from then on no transaction
    /// commits and nothing may persist or copy the store.
    #[must_use]
    pub fn has_incomplete_commit(&self) -> bool {
        self.poisoned.load(Ordering::Acquire)
    }

    /// Fails when a commit did not complete (see
    /// [`has_incomplete_commit`](Self::has_incomplete_commit)). Every commit,
    /// checkpoint and copy of the store calls this first.
    ///
    /// # Errors
    ///
    /// Returns [`TransactionError::IncompleteCommit`], which says a commit
    /// did not complete and the database must be reopened.
    pub fn check_no_incomplete_commit(&self) -> Result<()> {
        if self.has_incomplete_commit() {
            return Err(Error::Transaction(TransactionError::IncompleteCommit));
        }
        Ok(())
    }

    /// Aborts a transaction, undoing the changes it recorded that nobody
    /// undid yet (the writes of a
    /// [`QueryProcessor`](crate::query::QueryProcessor) with its transaction
    /// context; a session and a direct call undo their own first).
    ///
    /// # Errors
    ///
    /// Returns an error if the transaction is not active, or when a store
    /// fails to undo a change, which poisons the manager (the transaction is
    /// aborted either way).
    pub fn abort(&self, transaction_id: TransactionId) -> Result<()> {
        let undone = match self.changes(transaction_id) {
            Some(changes) => {
                // A store change in progress: a checkpoint never reads the
                // store or the change set halfway through the undo.
                let _writing = self.write_in_progress();
                changes.undo_after(changes.start(), false)
            }
            None => Ok(None),
        };
        let undo_error = match undone {
            Err(super::UndoFailure::Broken(error)) => {
                let message = format!(
                    "the abort of transaction {transaction_id:?} could not undo its changes: \
                     {error}"
                );
                self.poison(&message);
                Some(Error::Internal(message))
            }
            // A store without undo keeps its writes: the caller chose it.
            Ok(_) | Err(super::UndoFailure::External(_)) => None,
        };
        self.abort_registered(transaction_id)?;
        undo_error.map_or(Ok(()), Err)
    }

    /// Marks transaction `transaction_id` aborted.
    fn abort_registered(&self, transaction_id: TransactionId) -> Result<()> {
        let mut txns = self.transactions.write();

        let info = txns.get_mut(&transaction_id).ok_or_else(|| {
            Error::Transaction(TransactionError::InvalidState(
                "Transaction not found".to_string(),
            ))
        })?;

        if info.state != TransactionState::Active {
            return Err(Error::Transaction(TransactionError::InvalidState(
                "Transaction is not active".to_string(),
            )));
        }

        info.state = TransactionState::Aborted;
        info.changes = None;
        self.reads.release(info.start_epoch);
        // An aborted transaction conflicts with nobody: a private one, whose
        // state nobody asks for, leaves at once (others at the next `gc`).
        if info.private {
            txns.remove(&transaction_id);
        }
        self.active_count.fetch_sub(1, Ordering::Relaxed);
        Ok(())
    }

    /// Returns the write set of a transaction.
    ///
    /// This returns a copy of the entities written by this transaction,
    /// used for rollback to discard uncommitted versions.
    ///
    /// # Errors
    ///
    /// Returns a `TransactionError::InvalidState` if the transaction is not found.
    pub fn get_write_set(&self, transaction_id: TransactionId) -> Result<FxHashSet<GraphEntity>> {
        let txns = self.transactions.read();
        let info = txns.get(&transaction_id).ok_or_else(|| {
            Error::Transaction(TransactionError::InvalidState(
                "Transaction not found".to_string(),
            ))
        })?;
        Ok(info.write_set.clone())
    }

    /// Replaces the write set of a transaction (used for savepoint rollback).
    ///
    /// # Errors
    ///
    /// Returns an error if the transaction is not found.
    pub fn reset_write_set(
        &self,
        transaction_id: TransactionId,
        write_set: FxHashSet<GraphEntity>,
    ) -> Result<()> {
        let mut txns = self.transactions.write();
        let info = txns.get_mut(&transaction_id).ok_or_else(|| {
            Error::Transaction(TransactionError::InvalidState(
                "Transaction not found".to_string(),
            ))
        })?;
        info.write_set = write_set;
        Ok(())
    }

    /// Aborts all active transactions.
    ///
    /// Used during database shutdown.
    pub fn abort_all_active(&self) {
        let mut txns = self.transactions.write();
        for info in txns.values_mut() {
            if info.state == TransactionState::Active {
                info.state = TransactionState::Aborted;
                info.changes = None;
                self.reads.release(info.start_epoch);
                self.active_count.fetch_sub(1, Ordering::Relaxed);
            }
        }
    }

    /// Returns the state of a transaction.
    pub fn state(&self, transaction_id: TransactionId) -> Option<TransactionState> {
        self.transactions
            .read()
            .get(&transaction_id)
            .map(|info| info.state)
    }

    /// Returns the start epoch of a transaction.
    pub fn start_epoch(&self, transaction_id: TransactionId) -> Option<EpochId> {
        self.transactions
            .read()
            .get(&transaction_id)
            .map(|info| info.start_epoch)
    }

    /// Returns the current epoch: the last one whose commit is complete, so a
    /// snapshot at this epoch never holds part of a commit.
    #[must_use]
    pub fn current_epoch(&self) -> EpochId {
        EpochId::new(self.published_epoch.load(Ordering::Acquire))
    }

    /// Synchronizes the epoch counter to at least the given value.
    ///
    /// Used after snapshot import and WAL recovery to align the
    /// TransactionManager epoch with the store epoch, and by a write outside
    /// any transaction (which runs while no commit is in progress) to publish
    /// its epoch.
    pub fn sync_epoch(&self, epoch: EpochId) {
        self.assigned_epoch
            .fetch_max(epoch.as_u64(), Ordering::SeqCst);
        self.published_epoch
            .fetch_max(epoch.as_u64(), Ordering::SeqCst);
        self.wake_publish_waiters();
    }

    /// The read horizon: the oldest epoch any reader may still read at, now
    /// or from now on. Garbage collection keeps every version a reader at
    /// the horizon or later can see (`gc_versions`, `collect`).
    ///
    /// It is the oldest of: the snapshot of every open transaction and of
    /// every committing one (a commit whose publish has not come, or never
    /// comes after an unknown outcome, still reads at its snapshot and its
    /// versions stay pending), the epoch of every read pinned outside a
    /// transaction (`pin_read`: every read statement outside a transaction
    /// pins for its run), and the published epoch, at which every reader
    /// that starts from now on reads.
    ///
    /// Each of them is counted by the epoch it reads at as it begins (a
    /// transaction's snapshot until its commit is published or it aborts,
    /// a pin until it is dropped), so this reads the oldest counted epoch
    /// and costs the same however many transactions finished. A reader's
    /// epoch is read and counted under the lock this reads them under, so a
    /// reader that starts meanwhile is either counted or reads at the
    /// published epoch read here or a later one.
    #[must_use]
    pub fn read_horizon(&self) -> EpochId {
        let readers = self.reads.epochs.lock();
        let published = self.published_epoch.load(Ordering::Acquire);
        EpochId::new(ReadPins::oldest(&readers).map_or(published, |oldest| oldest.min(published)))
    }

    /// Pins a read outside any transaction at the published epoch, for as
    /// long as the returned pin lives: the [read horizon](Self::read_horizon)
    /// stays at its epoch or before. The read may read at that epoch or at
    /// any later one it loads after the pin (the published epoch only moves
    /// forward). Read statements outside a transaction pin for their run
    /// (#412 decision Q6), so collecting at the horizon never takes a version
    /// such a statement still reads.
    pub(crate) fn pin_read(&self) -> ReadPin {
        let epoch = self.reads.hold_published(&self.published_epoch);
        ReadPin {
            pins: Arc::clone(&self.reads),
            epoch,
        }
    }

    /// Pins a read at `epoch`, a time-travel read's (one after the
    /// published epoch reads at the published one), for as long as the
    /// returned pin lives: the collect horizon stays at its epoch or before,
    /// so the versions it reads stay while it runs.
    ///
    /// # Errors
    ///
    /// [`Error::InvalidValue`] for an epoch before the history floor (see
    /// [`history_floor`](Self::history_floor)): "history before ... is not
    /// retained". The floor is read under the lock the collect horizon moves
    /// it under, so a read pinned here is never collected past.
    pub(crate) fn pin_read_at(&self, epoch: EpochId) -> Result<ReadPin> {
        let mut epochs = self.reads.epochs.lock();
        if epoch < self.history.floor() {
            return Err(self.history.before_floor(epoch));
        }
        let epoch = epoch.min(EpochId::new(self.published_epoch.load(Ordering::Acquire)));
        ReadPins::hold_at(&mut epochs, epoch.as_u64());
        Ok(ReadPin {
            pins: Arc::clone(&self.reads),
            epoch,
        })
    }

    /// The horizon garbage collection runs at: the older of the
    /// [read horizon](Self::read_horizon) and the start of the database's
    /// retention window (none while history is off, so it is the read
    /// horizon then, as without history). Every version a reader at it or
    /// later reads is kept, and it becomes the history floor: from now on a
    /// read before it is refused. The engine's collects run at it.
    #[must_use]
    pub fn collect_horizon(&self) -> EpochId {
        let readers = self.reads.epochs.lock();
        let published = self.published_epoch.load(Ordering::Acquire);
        let read = ReadPins::oldest(&readers).map_or(published, |oldest| oldest.min(published));
        let horizon = match self.history.retention_horizon() {
            Some(start) => read.min(start),
            None => read,
        };
        let horizon = EpochId::new(horizon);
        self.history.collected_at(horizon);
        horizon
    }

    /// The oldest epoch a read may still ask for: the newest collect
    /// horizon (see [`collect_horizon`](Self::collect_horizon)). A
    /// time-travel read before it is refused.
    #[must_use]
    pub fn history_floor(&self) -> EpochId {
        self.history.floor()
    }

    /// Sets how long the database keeps history, from now on (see
    /// [`HistoryRetention`]): turned on, the commit times are recorded from
    /// the published epoch on; turned off, the time index goes, and the
    /// collect horizon is the read horizon again.
    pub fn set_history_retention(&self, retention: HistoryRetention) {
        self.history
            .set_retention(retention, self.published_epoch.load(Ordering::Acquire));
    }

    /// How long the database keeps history.
    #[must_use]
    pub fn history_retention(&self) -> HistoryRetention {
        self.history.retention()
    }

    /// The epoch a read at `timestamp` reads at: the last one committed at
    /// or before it (at the resolution of a millisecond).
    ///
    /// # Errors
    ///
    /// [`Error::InvalidValue`] while the database keeps no history, and for a
    /// time before what it keeps.
    pub fn epoch_at(&self, timestamp: Timestamp) -> Result<EpochId> {
        self.history.epoch_at(timestamp)
    }

    /// Now on the clock commit times come from, in milliseconds since the
    /// Unix epoch.
    pub(crate) fn history_now(&self) -> u64 {
        self.history.now()
    }

    /// The time a write outside a transaction that commits at once takes,
    /// under the commit lock, while the database keeps history: its WAL
    /// group carries it, and [`record_commit_time`](Self::record_commit_time)
    /// records it once its epoch is published.
    pub(crate) fn commit_time(&self) -> Option<u64> {
        self.history.commit_time()
    }

    /// Records that `epoch` committed at `millis`: a write outside a
    /// transaction, once published, and a replayed WAL group. Nothing while
    /// history is off.
    pub(crate) fn record_commit_time(&self, epoch: EpochId, millis: u64) {
        self.history.record(epoch, millis);
    }

    /// What a checkpoint writes of the history, `None` while it is off.
    pub(crate) fn history_image(&self) -> Option<HistoryImage> {
        self.history.image()
    }

    /// Sets the retention an open starts with, before it loads the file and
    /// replays the log (see [`anchor_history`](Self::anchor_history)).
    pub(crate) fn open_history(&self, retention: HistoryRetention) {
        self.history.open(retention);
    }

    /// Takes in what a load found of the history: the image of a file's
    /// `TIME_INDEX` section, or, for a file without one, the epoch of its
    /// checkpoint, before which nothing is retained.
    #[cfg(any(test, feature = "lpg"))]
    pub(crate) fn restore_history_or_floor(
        &self,
        image: Option<HistoryImage>,
        checkpoint: EpochId,
    ) {
        match image {
            Some(image) => self.history.restore(image),
            None => self.history.collected_at(checkpoint),
        }
    }

    /// At the end of an open or a new database: a history that is on and
    /// holds no commit at or after the published epoch starts there, now,
    /// so a read at a time from now on finds its epoch.
    pub(crate) fn anchor_history(&self) {
        self.history
            .anchor(self.published_epoch.load(Ordering::Acquire));
    }

    /// Returns the number of active transactions.
    #[must_use]
    pub fn active_count(&self) -> usize {
        self.transactions
            .read()
            .values()
            .filter(|info| info.state == TransactionState::Active)
            .count()
    }

    /// Cleans up completed transactions that are no longer needed for conflict detection.
    ///
    /// A committed transaction's write set must be preserved until all transactions
    /// that started before its commit have completed. This ensures write-write
    /// conflict detection works correctly.
    ///
    /// Returns the number of transactions cleaned up.
    pub fn gc(&self) -> usize {
        let mut txns = self.transactions.write();
        let mut committed = self.committed_epochs.write();

        // Find the minimum start epoch among active transactions
        let min_active_start = txns
            .values()
            .filter(|info| info.state == TransactionState::Active)
            .map(|info| info.start_epoch)
            .min();

        let initial_count = txns.len();

        // Collect transactions safe to remove
        let to_remove: Vec<TransactionId> = txns
            .iter()
            .filter(|(transaction_id, info)| {
                match info.state {
                    // Never remove active transactions or a commit in progress
                    TransactionState::Active | TransactionState::Committing => false,
                    TransactionState::Aborted => true, // Always safe to remove aborted transactions
                    TransactionState::Committed => {
                        // Only remove committed transactions if their commit epoch
                        // is older than all active transactions' start epochs
                        if let Some(min_start) = min_active_start {
                            if let Some(commit_epoch) = committed.get(*transaction_id) {
                                // Safe to remove if committed before all active txns started
                                commit_epoch.as_u64() < min_start.as_u64()
                            } else {
                                // No commit epoch recorded, keep it to be safe
                                false
                            }
                        } else {
                            // No active transactions, safe to remove all committed
                            true
                        }
                    }
                }
            })
            .map(|(id, _)| *id)
            .collect();

        for id in &to_remove {
            txns.remove(id);
            committed.remove(id);
        }

        initial_count - txns.len()
    }

    /// Marks a transaction as committed at a specific epoch.
    ///
    /// Used during recovery to restore transaction state.
    pub fn mark_committed(&self, transaction_id: TransactionId, epoch: EpochId) {
        self.committed_epochs.write().insert(transaction_id, epoch);
    }

    /// Returns the last assigned transaction ID.
    ///
    /// Returns `None` if no transactions have been started yet.
    #[must_use]
    pub fn last_assigned_transaction_id(&self) -> Option<TransactionId> {
        let next = self.next_transaction_id.load(Ordering::Relaxed);
        if next > 1 {
            Some(TransactionId::new(next - 1))
        } else {
            None
        }
    }

    /// Returns the commit epoch of a transaction, if committed.
    #[cfg(test)]
    pub fn committed_epoch(&self, transaction_id: TransactionId) -> Option<EpochId> {
        self.committed_epochs.read().get(&transaction_id).copied()
    }
}

impl Default for TransactionManager {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_begin_commit() {
        let mgr = TransactionManager::new();

        let tx = mgr.begin();
        assert_eq!(mgr.state(tx), Some(TransactionState::Active));

        let commit_epoch = mgr.commit(tx).unwrap();
        assert_eq!(mgr.state(tx), Some(TransactionState::Committed));
        assert!(commit_epoch.as_u64() > 0);
    }

    #[test]
    fn test_begin_abort() {
        let mgr = TransactionManager::new();

        let tx = mgr.begin();
        mgr.abort(tx).unwrap();
        assert_eq!(mgr.state(tx), Some(TransactionState::Aborted));
    }

    #[test]
    fn test_epoch_advancement() {
        let mgr = TransactionManager::new();

        let initial_epoch = mgr.current_epoch();

        let tx = mgr.begin();
        let commit_epoch = mgr.commit(tx).unwrap();

        assert!(mgr.current_epoch().as_u64() > initial_epoch.as_u64());
        assert!(commit_epoch.as_u64() > initial_epoch.as_u64());
    }

    #[test]
    fn test_gc_preserves_needed_write_sets() {
        let mgr = TransactionManager::new();

        let tx1 = mgr.begin();
        let tx2 = mgr.begin();

        mgr.commit(tx1).unwrap();
        // tx2 still active - started before tx1 committed

        assert_eq!(mgr.active_count(), 1);

        // GC should NOT remove tx1 because tx2 might need its write set for conflict detection
        let cleaned = mgr.gc();
        assert_eq!(cleaned, 0);

        // Both transactions should remain
        assert_eq!(mgr.state(tx1), Some(TransactionState::Committed));
        assert_eq!(mgr.state(tx2), Some(TransactionState::Active));
    }

    #[test]
    fn test_gc_removes_old_commits() {
        let mgr = TransactionManager::new();

        // tx1 commits at epoch 1
        let tx1 = mgr.begin();
        mgr.commit(tx1).unwrap();

        // tx2 starts at epoch 1, commits at epoch 2
        let tx2 = mgr.begin();
        mgr.commit(tx2).unwrap();

        // tx3 starts at epoch 2
        let tx3 = mgr.begin();

        // At this point:
        // - tx1 committed at epoch 1, tx3 started at epoch 2 → tx1 commit < tx3 start → safe to GC
        // - tx2 committed at epoch 2, tx3 started at epoch 2 → tx2 commit >= tx3 start → NOT safe
        let cleaned = mgr.gc();
        assert_eq!(cleaned, 1); // Only tx1 removed

        assert_eq!(mgr.state(tx1), None);
        assert_eq!(mgr.state(tx2), Some(TransactionState::Committed)); // Preserved for conflict detection
        assert_eq!(mgr.state(tx3), Some(TransactionState::Active));

        // After tx3 commits, tx2 can be GC'd
        mgr.commit(tx3).unwrap();
        let cleaned = mgr.gc();
        assert_eq!(cleaned, 2); // tx2 and tx3 both cleaned (no active transactions)
    }

    #[test]
    fn test_gc_removes_aborted() {
        let mgr = TransactionManager::new();

        let tx1 = mgr.begin();
        let tx2 = mgr.begin();

        mgr.abort(tx1).unwrap();
        // tx2 still active

        // Aborted transactions are always safe to remove
        let cleaned = mgr.gc();
        assert_eq!(cleaned, 1);

        assert_eq!(mgr.state(tx1), None);
        assert_eq!(mgr.state(tx2), Some(TransactionState::Active));
    }

    #[test]
    fn test_write_tracking() {
        let mgr = TransactionManager::new();

        let tx = mgr.begin();

        // Record writes
        mgr.record_write(tx, NodeId::new(1)).unwrap();
        mgr.record_write(tx, NodeId::new(2)).unwrap();
        mgr.record_write(tx, EdgeId::new(100)).unwrap();

        // Should commit successfully (no conflicts)
        assert!(mgr.commit(tx).is_ok());
    }

    /// With no reader the horizon is the published epoch; with open
    /// transactions, the oldest snapshot, until it ends.
    #[test]
    fn the_read_horizon_is_the_oldest_open_snapshot() {
        let mgr = TransactionManager::new();
        assert_eq!(
            mgr.read_horizon(),
            mgr.current_epoch(),
            "no reader: the published epoch"
        );

        let tx1 = mgr.begin();
        let epoch1 = mgr.start_epoch(tx1).unwrap();
        let tx2 = mgr.begin();
        mgr.commit(tx2).unwrap();
        let _tx3 = mgr.begin();
        assert_eq!(mgr.read_horizon(), epoch1, "the earliest open snapshot");

        mgr.abort(tx1).unwrap();
        assert_eq!(mgr.read_horizon(), mgr.current_epoch());
    }

    /// A read outside any transaction holds the horizon at the epoch it
    /// pinned while later commits publish; two reads pinned at one epoch
    /// hold it until both end, and then it moves to the published epoch.
    #[test]
    fn a_pinned_read_holds_the_read_horizon_back() {
        let mgr = TransactionManager::new();
        let first = mgr.begin();
        let committed = mgr.commit(first).unwrap();
        let read = mgr.pin_read();
        let again = mgr.pin_read();
        assert_eq!((read.epoch(), again.epoch()), (committed, committed));
        for _ in 0..2 {
            let tx = mgr.begin();
            mgr.commit(tx).unwrap();
        }
        assert_eq!(mgr.current_epoch(), EpochId::new(committed.as_u64() + 2));
        assert_eq!(mgr.read_horizon(), committed, "the pinned read");
        drop(read);
        assert_eq!(mgr.read_horizon(), committed, "the second read");
        let later = mgr.pin_read();
        drop(again);
        assert_eq!(mgr.read_horizon(), later.epoch(), "the later read");
        drop(later);
        assert_eq!(mgr.read_horizon(), mgr.current_epoch());
    }

    /// A pin dropped by the unwind of a panic in its read unpins it.
    #[test]
    fn a_read_that_panics_unpins() {
        let mgr = TransactionManager::new();
        let tx = mgr.begin();
        let pinned = mgr.commit(tx).unwrap();
        let read = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _pin = mgr.pin_read();
            let tx = mgr.begin();
            mgr.commit(tx).unwrap();
            assert_eq!(mgr.read_horizon(), pinned);
            panic!("the read fails");
        }));
        assert!(read.is_err());
        assert_eq!(mgr.read_horizon(), mgr.current_epoch());
        assert!(mgr.current_epoch() > pinned);
    }

    /// A committing transaction (its epoch assigned, its publish not come)
    /// still reads at its snapshot, which holds the horizon back after a
    /// later snapshot's commit published; the horizon never passes the
    /// published epoch for an epoch assigned but not published, and moves
    /// once the commit publishes.
    #[test]
    fn a_committing_transaction_holds_the_read_horizon_back() {
        let mgr = TransactionManager::new();
        let slow = mgr.begin();
        let snapshot = mgr.start_epoch(slow).unwrap();
        let fast = mgr.begin();
        mgr.commit(fast).unwrap();
        let published = mgr.current_epoch();
        assert!(published > snapshot);

        let publish = mgr.start_commit(slow).unwrap().written();
        assert_eq!(mgr.state(slow), Some(TransactionState::Committing));
        assert_eq!(mgr.read_horizon(), snapshot, "the committing snapshot");
        let reader = mgr.begin();
        assert_eq!(mgr.start_epoch(reader), Some(published));
        mgr.abort(reader).unwrap();
        assert_eq!(mgr.read_horizon(), snapshot);

        let epoch = publish.epoch();
        publish.complete().unwrap();
        assert_eq!(mgr.read_horizon(), epoch, "published: no reader before it");
    }

    /// While a commit is written and not published, the horizon stays at
    /// the published epoch: a reader that starts meanwhile reads there.
    #[test]
    fn the_read_horizon_stays_at_the_published_epoch_during_a_commit() {
        let mgr = TransactionManager::new();
        let tx = mgr.begin();
        let published = mgr.current_epoch();
        let publish = mgr.start_commit(tx).unwrap().written();
        // The committing transaction's snapshot is the published epoch; the
        // horizon does not reach the epoch it was assigned.
        assert!(publish.epoch() > published);
        assert_eq!(mgr.read_horizon(), published);
        let read = mgr.pin_read();
        assert_eq!(read.epoch(), published, "a read reads what is published");
        publish.complete().unwrap();
        assert_eq!(mgr.read_horizon(), published, "the read is pinned");
        drop(read);
        assert_eq!(mgr.read_horizon(), mgr.current_epoch());
    }

    /// Every way a transaction finishes lets go of its snapshot, before any
    /// `gc` drops it: a commit (private or not), an abort, the abort of every
    /// active one; a commit taken back keeps the snapshot until the
    /// transaction's rollback.
    #[test]
    fn a_finished_transaction_no_longer_holds_the_read_horizon() {
        let mgr = TransactionManager::new();
        let oldest = mgr.begin();
        let snapshot = mgr.start_epoch(oldest).unwrap();
        for round in 0..100 {
            let tx = mgr.begin();
            if round % 2 == 0 {
                mgr.commit(tx).unwrap();
            } else {
                mgr.abort(tx).unwrap();
            }
            let (private, _) = mgr.begin_private().unwrap();
            mgr.commit(private).unwrap();
        }
        assert_eq!(mgr.read_horizon(), snapshot, "the oldest open snapshot");
        mgr.abort(oldest).unwrap();
        assert_eq!(mgr.read_horizon(), mgr.current_epoch(), "no reader left");

        // A commit whose log write failed is taken back: the transaction is
        // active again and still reads at its snapshot.
        let taken_back = mgr.begin();
        let snapshot = mgr.start_epoch(taken_back).unwrap();
        let other = mgr.begin();
        mgr.commit(other).unwrap();
        mgr.start_commit(taken_back).unwrap().abort();
        assert_eq!(mgr.state(taken_back), Some(TransactionState::Active));
        assert_eq!(mgr.read_horizon(), snapshot, "the commit taken back");
        mgr.abort(taken_back).unwrap();
        assert_eq!(mgr.read_horizon(), mgr.current_epoch());

        let first = mgr.begin();
        let held = mgr.start_epoch(first).unwrap();
        let tx = mgr.begin();
        mgr.commit(tx).unwrap();
        let _second = mgr.begin();
        assert_eq!(mgr.read_horizon(), held);
        mgr.abort_all_active();
        assert_eq!(
            mgr.read_horizon(),
            mgr.current_epoch(),
            "all aborted at once"
        );
    }

    #[test]
    fn test_abort_all_active() {
        let mgr = TransactionManager::new();

        let tx1 = mgr.begin();
        let tx2 = mgr.begin();
        let tx3 = mgr.begin();

        mgr.commit(tx1).unwrap();
        // tx2 and tx3 still active

        mgr.abort_all_active();

        assert_eq!(mgr.state(tx1), Some(TransactionState::Committed)); // Already committed
        assert_eq!(mgr.state(tx2), Some(TransactionState::Aborted));
        assert_eq!(mgr.state(tx3), Some(TransactionState::Aborted));
    }

    #[test]
    fn test_start_epoch_snapshot() {
        let mgr = TransactionManager::new();

        // Start epoch for tx1
        let tx1 = mgr.begin();
        let start1 = mgr.start_epoch(tx1).unwrap();

        // Commit tx1, advancing epoch
        mgr.commit(tx1).unwrap();

        // Start tx2 after epoch advanced
        let tx2 = mgr.begin();
        let start2 = mgr.start_epoch(tx2).unwrap();

        // tx2 should have a later start epoch
        assert!(start2.as_u64() > start1.as_u64());
    }

    #[test]
    fn test_write_write_conflict_detection() {
        let mgr = TransactionManager::new();

        // Both transactions start at the same epoch
        let tx1 = mgr.begin();
        let tx2 = mgr.begin();

        // First writer succeeds
        let entity = NodeId::new(42);
        mgr.record_write(tx1, entity).unwrap();

        // Second writer is rejected immediately (first-writer-wins)
        let result = mgr.record_write(tx2, entity);
        assert!(result.is_err());
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("Write-write conflict"),
            "Expected write-write conflict error"
        );

        // First commit succeeds (no conflict at commit time either)
        let result1 = mgr.commit(tx1);
        assert!(result1.is_ok());
    }

    #[test]
    fn test_commit_epoch_monotonicity() {
        let mgr = TransactionManager::new();

        let mut epochs = Vec::new();

        // Commit multiple transactions and verify epochs are strictly increasing
        for _ in 0..10 {
            let tx = mgr.begin();
            let epoch = mgr.commit(tx).unwrap();
            epochs.push(epoch.as_u64());
        }

        // Verify strict monotonicity
        for i in 1..epochs.len() {
            assert!(
                epochs[i] > epochs[i - 1],
                "Epoch {} ({}) should be greater than epoch {} ({})",
                i,
                epochs[i],
                i - 1,
                epochs[i - 1]
            );
        }
    }

    #[test]
    fn test_concurrent_commits_via_threads() {
        use std::sync::Arc;
        use std::thread;

        let mgr = Arc::new(TransactionManager::new());
        let num_threads = 10;
        let commits_per_thread = 100;

        let handles: Vec<_> = (0..num_threads)
            .map(|_| {
                let mgr = Arc::clone(&mgr);
                thread::spawn(move || {
                    let mut epochs = Vec::new();
                    for _ in 0..commits_per_thread {
                        let tx = mgr.begin();
                        let epoch = mgr.commit(tx).unwrap();
                        epochs.push(epoch.as_u64());
                    }
                    epochs
                })
            })
            .collect();

        let mut all_epochs: Vec<u64> = handles
            .into_iter()
            .flat_map(|h| h.join().unwrap())
            .collect();

        // All epochs should be unique (no duplicates)
        all_epochs.sort_unstable();
        let unique_count = all_epochs.len();
        all_epochs.dedup();
        assert_eq!(
            all_epochs.len(),
            unique_count,
            "All commit epochs should be unique"
        );

        // Final epoch should equal number of commits
        // num_threads=10, commits_per_thread=100, product is 1000
        // reason: value is non-negative by preceding validation
        #[allow(clippy::cast_sign_loss)]
        let expected_epoch = (num_threads * commits_per_thread) as u64;
        assert_eq!(
            mgr.current_epoch().as_u64(),
            expected_epoch,
            "Final epoch should equal total commits"
        );
    }

    #[test]
    fn test_isolation_level_default() {
        let mgr = TransactionManager::new();

        let tx = mgr.begin();
        assert_eq!(
            mgr.isolation_level(tx),
            Some(IsolationLevel::SnapshotIsolation)
        );
    }

    #[test]
    fn test_isolation_level_explicit() {
        let mgr = TransactionManager::new();

        let transaction_rc = mgr.begin_with_isolation(IsolationLevel::ReadCommitted);
        let transaction_si = mgr.begin_with_isolation(IsolationLevel::SnapshotIsolation);
        let transaction_ser = mgr.begin_with_isolation(IsolationLevel::Serializable);

        assert_eq!(
            mgr.isolation_level(transaction_rc),
            Some(IsolationLevel::ReadCommitted)
        );
        assert_eq!(
            mgr.isolation_level(transaction_si),
            Some(IsolationLevel::SnapshotIsolation)
        );
        assert_eq!(
            mgr.isolation_level(transaction_ser),
            Some(IsolationLevel::Serializable)
        );
    }

    #[test]
    fn test_ssi_read_write_conflict_detected() {
        let mgr = TransactionManager::new();

        // tx1 starts with Serializable isolation
        let tx1 = mgr.begin_with_isolation(IsolationLevel::Serializable);

        // tx2 starts and will modify an entity
        let tx2 = mgr.begin();

        // tx1 reads entity 42
        let entity = NodeId::new(42);
        mgr.record_read(tx1, entity).unwrap();

        // tx2 writes to the same entity and commits
        mgr.record_write(tx2, entity).unwrap();
        mgr.commit(tx2).unwrap();

        // tx1 tries to commit - should fail due to SSI read-write conflict
        let result = mgr.commit(tx1);
        assert!(result.is_err());
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("Serialization failure"),
            "Expected serialization failure error"
        );
    }

    #[test]
    fn test_ssi_no_conflict_when_not_serializable() {
        let mgr = TransactionManager::new();

        // tx1 starts with default Snapshot Isolation
        let tx1 = mgr.begin();

        // tx2 starts and will modify an entity
        let tx2 = mgr.begin();

        // tx1 reads entity 42
        let entity = NodeId::new(42);
        mgr.record_read(tx1, entity).unwrap();

        // tx2 writes to the same entity and commits
        mgr.record_write(tx2, entity).unwrap();
        mgr.commit(tx2).unwrap();

        // tx1 should commit successfully (SI doesn't check read-write conflicts)
        let result = mgr.commit(tx1);
        assert!(
            result.is_ok(),
            "Snapshot Isolation should not detect read-write conflicts"
        );
    }

    #[test]
    fn test_ssi_no_conflict_when_write_before_read() {
        let mgr = TransactionManager::new();

        // tx1 writes and commits first
        let tx1 = mgr.begin();
        let entity = NodeId::new(42);
        mgr.record_write(tx1, entity).unwrap();
        mgr.commit(tx1).unwrap();

        // tx2 starts AFTER tx1 committed and reads the entity
        let tx2 = mgr.begin_with_isolation(IsolationLevel::Serializable);
        mgr.record_read(tx2, entity).unwrap();

        // tx2 should commit successfully (tx1 committed before tx2 started)
        let result = mgr.commit(tx2);
        assert!(
            result.is_ok(),
            "Should not conflict when writer committed before reader started"
        );
    }

    #[test]
    fn test_write_skew_prevented_by_ssi() {
        // Classic write skew scenario:
        // Account A = 50, Account B = 50, constraint: A + B >= 0
        // T1 reads A, B, writes A = A - 100
        // T2 reads A, B, writes B = B - 100
        // Without SSI, both could commit violating the constraint.

        let mgr = TransactionManager::new();

        let account_a = NodeId::new(1);
        let account_b = NodeId::new(2);

        // T1 and T2 both start with Serializable isolation
        let tx1 = mgr.begin_with_isolation(IsolationLevel::Serializable);
        let tx2 = mgr.begin_with_isolation(IsolationLevel::Serializable);

        // Both read both accounts
        mgr.record_read(tx1, account_a).unwrap();
        mgr.record_read(tx1, account_b).unwrap();
        mgr.record_read(tx2, account_a).unwrap();
        mgr.record_read(tx2, account_b).unwrap();

        // T1 writes to A, T2 writes to B (no write-write conflict)
        mgr.record_write(tx1, account_a).unwrap();
        mgr.record_write(tx2, account_b).unwrap();

        // T1 commits first
        let result1 = mgr.commit(tx1);
        assert!(result1.is_ok(), "First commit should succeed");

        // T2 tries to commit - should fail because it read account_a which T1 wrote
        let result2 = mgr.commit(tx2);
        assert!(result2.is_err(), "Second commit should fail due to SSI");
        assert!(
            result2
                .unwrap_err()
                .to_string()
                .contains("Serialization failure"),
            "Expected serialization failure error for write skew prevention"
        );
    }

    #[test]
    fn test_read_committed_allows_non_repeatable_reads() {
        let mgr = TransactionManager::new();

        // tx1 starts with ReadCommitted isolation
        let tx1 = mgr.begin_with_isolation(IsolationLevel::ReadCommitted);
        let entity = NodeId::new(42);

        // tx1 reads entity
        mgr.record_read(tx1, entity).unwrap();

        // tx2 writes and commits
        let tx2 = mgr.begin();
        mgr.record_write(tx2, entity).unwrap();
        mgr.commit(tx2).unwrap();

        // tx1 can still commit (ReadCommitted allows non-repeatable reads)
        let result = mgr.commit(tx1);
        assert!(
            result.is_ok(),
            "ReadCommitted should allow non-repeatable reads"
        );
    }

    #[test]
    fn test_isolation_level_debug() {
        assert_eq!(
            format!("{:?}", IsolationLevel::ReadCommitted),
            "ReadCommitted"
        );
        assert_eq!(
            format!("{:?}", IsolationLevel::SnapshotIsolation),
            "SnapshotIsolation"
        );
        assert_eq!(
            format!("{:?}", IsolationLevel::Serializable),
            "Serializable"
        );
    }

    #[test]
    fn test_isolation_level_default_trait() {
        let default: IsolationLevel = Default::default();
        assert_eq!(default, IsolationLevel::SnapshotIsolation);
    }

    #[test]
    fn test_ssi_concurrent_reads_no_conflict() {
        let mgr = TransactionManager::new();

        let entity = NodeId::new(42);

        // Both transactions read the same entity
        let tx1 = mgr.begin_with_isolation(IsolationLevel::Serializable);
        let tx2 = mgr.begin_with_isolation(IsolationLevel::Serializable);

        mgr.record_read(tx1, entity).unwrap();
        mgr.record_read(tx2, entity).unwrap();

        // Both should commit successfully (read-read is not a conflict)
        assert!(mgr.commit(tx1).is_ok());
        assert!(mgr.commit(tx2).is_ok());
    }

    #[test]
    fn test_ssi_write_write_conflict() {
        let mgr = TransactionManager::new();

        let entity = NodeId::new(42);

        // Both transactions attempt to write the same entity
        let tx1 = mgr.begin_with_isolation(IsolationLevel::Serializable);
        let tx2 = mgr.begin_with_isolation(IsolationLevel::Serializable);

        // First writer succeeds
        mgr.record_write(tx1, entity).unwrap();

        // Second writer is rejected immediately (first-writer-wins)
        let result = mgr.record_write(tx2, entity);
        assert!(
            result.is_err(),
            "Second record_write should fail with write-write conflict"
        );

        // First commit succeeds
        assert!(mgr.commit(tx1).is_ok());
    }

    #[test]
    fn test_ssi_concurrent_commit_race() {
        // Regression test: with the old read-then-upgrade lock pattern,
        // two concurrent SSI commits could both succeed when one should
        // have been aborted due to a read-write conflict (write skew).
        use std::sync::Arc;

        let mgr = Arc::new(TransactionManager::new());

        // Run many iterations to exercise the race window
        for _ in 0..100 {
            let entity_a = NodeId::new(1);
            let entity_b = NodeId::new(2);

            // Classic write skew setup: both transactions read both entities,
            // then each writes to a different one.
            let tx1 = mgr.begin_with_isolation(IsolationLevel::Serializable);
            let tx2 = mgr.begin_with_isolation(IsolationLevel::Serializable);

            mgr.record_read(tx1, entity_a).unwrap();
            mgr.record_read(tx1, entity_b).unwrap();
            mgr.record_read(tx2, entity_a).unwrap();
            mgr.record_read(tx2, entity_b).unwrap();

            mgr.record_write(tx1, entity_a).unwrap();
            mgr.record_write(tx2, entity_b).unwrap();

            // Commit tx1 first so it's in committed_epochs
            mgr.commit(tx1).unwrap();

            // tx2 should be rejected: it read entity_a which tx1 wrote
            let result = mgr.commit(tx2);
            assert!(
                result.is_err(),
                "SSI should detect read-write conflict on entity_a"
            );

            // Abort the rejected transaction so its write set is cleared
            // before the next iteration.
            let _ = mgr.abort(tx2);
            mgr.gc();
        }
    }

    #[test]
    fn test_ssi_concurrent_commit_barrier() {
        // Stress test with barrier synchronization to maximize the chance
        // of concurrent commit() calls overlapping.
        use std::sync::{Arc, Barrier};
        use std::thread;

        let mgr = Arc::new(TransactionManager::new());
        let mut both_ok_count = 0;

        for _ in 0..50 {
            let entity_a = NodeId::new(1);
            let entity_b = NodeId::new(2);

            let tx1 = mgr.begin_with_isolation(IsolationLevel::Serializable);
            let tx2 = mgr.begin_with_isolation(IsolationLevel::Serializable);

            mgr.record_read(tx1, entity_a).unwrap();
            mgr.record_read(tx1, entity_b).unwrap();
            mgr.record_read(tx2, entity_a).unwrap();
            mgr.record_read(tx2, entity_b).unwrap();

            mgr.record_write(tx1, entity_a).unwrap();
            mgr.record_write(tx2, entity_b).unwrap();

            let mgr1 = Arc::clone(&mgr);
            let mgr2 = Arc::clone(&mgr);
            let barrier = Arc::new(Barrier::new(2));
            let b1 = Arc::clone(&barrier);
            let b2 = Arc::clone(&barrier);

            let h1 = thread::spawn(move || {
                b1.wait();
                mgr1.commit(tx1)
            });
            let h2 = thread::spawn(move || {
                b2.wait();
                mgr2.commit(tx2)
            });

            let r1 = h1.join().unwrap();
            let r2 = h2.join().unwrap();

            if r1.is_ok() && r2.is_ok() {
                both_ok_count += 1;
            }

            // Clean up
            if r1.is_err() {
                let _ = mgr.abort(tx1);
            }
            if r2.is_err() {
                let _ = mgr.abort(tx2);
            }
            mgr.gc();
        }

        // At most one should succeed per iteration (write skew prevention).
        // With the fix, both_ok_count should always be 0.
        assert_eq!(
            both_ok_count, 0,
            "SSI must prevent both concurrent write-skew commits from succeeding"
        );
    }

    #[test]
    fn test_committed_epoch_present_after_commit() {
        // Verify that after commit(), the committed_epochs entry is always
        // present (no window where state is Committed but epoch is missing).
        let mgr = TransactionManager::new();

        let tx = mgr.begin();
        mgr.record_write(tx, NodeId::new(1)).unwrap();
        let epoch = mgr.commit(tx).unwrap();

        // committed_epoch must be available immediately after commit returns
        assert_eq!(
            mgr.committed_epoch(tx),
            Some(epoch),
            "committed_epochs must contain tx immediately after commit()"
        );
    }

    /// Until a commit is complete, a direct call that commits at once cannot
    /// start and another transaction cannot write what the committing one
    /// wrote; completing the commit allows both again.
    #[test]
    fn a_commit_holds_its_writes_until_it_is_complete() {
        let mgr = TransactionManager::new();
        let tx = mgr.begin();
        let other = mgr.begin();
        mgr.record_write(tx, NodeId::new(1)).unwrap();

        let commit = mgr.start_commit(tx).unwrap();
        assert_eq!(mgr.state(tx), Some(TransactionState::Committing));
        assert_eq!(mgr.committed_epoch(tx), Some(commit.epoch()));
        assert!(matches!(
            mgr.record_write(other, NodeId::new(1)),
            Err(Error::Transaction(TransactionError::WriteConflict(_)))
        ));
        mgr.record_write(other, NodeId::new(2)).unwrap();
        mgr.abort(other).unwrap();
        assert!(
            mgr.idle_gate().is_none(),
            "a direct call that commits at once waits for the commit"
        );

        commit.complete().unwrap();
        assert_eq!(mgr.state(tx), Some(TransactionState::Committed));
        assert!(mgr.changes(tx).is_none(), "a commit drops its change set");
        assert!(mgr.idle_gate().is_some());
        let next = mgr.begin();
        mgr.record_write(next, NodeId::new(1)).unwrap();
    }

    /// A private transaction (a direct call) begins during a commit without
    /// waiting for it, at the published epoch, registered with its change
    /// set; it conflicts with the committing transaction on what that one
    /// wrote, and with an open transaction that wrote an entity first.
    #[test]
    fn a_private_transaction_begins_during_a_commit_at_the_published_epoch() {
        let mgr = TransactionManager::new();
        let open = mgr.begin();
        mgr.record_write(open, NodeId::new(3)).unwrap();
        let tx = mgr.begin();
        mgr.record_write(tx, NodeId::new(19)).unwrap();
        let published = mgr.current_epoch();
        let commit = mgr.start_commit(tx).unwrap();

        let (direct, changes) = mgr.begin_private().unwrap();
        assert_eq!(mgr.start_epoch(direct), Some(published));
        assert_eq!((changes.id(), changes.snapshot()), (direct, published));
        assert!(mgr.changes(direct).is_some(), "registered with its set");
        for (node, whose) in [(3, "the open transaction"), (19, "the committing one")] {
            assert!(
                matches!(
                    mgr.record_write(direct, NodeId::new(node)),
                    Err(Error::Transaction(TransactionError::WriteConflict(_)))
                ),
                "{whose} wrote node {node} first"
            );
        }
        commit.complete().unwrap();
        mgr.abort(direct).unwrap();
        assert!(mgr.changes(direct).is_none(), "an abort drops its set");
    }

    /// A transaction begins during a commit without waiting for it: it
    /// starts at the published epoch, which holds none of the commit; one
    /// that begins once the commit is published starts at its epoch.
    #[test]
    fn begin_does_not_wait_for_a_commit_in_progress() {
        let mgr = TransactionManager::new();
        let tx = mgr.begin();
        let published = mgr.current_epoch();
        let commit = mgr.start_commit(tx).unwrap();
        let epoch = commit.epoch();

        let during = mgr.begin();
        assert_eq!(mgr.start_epoch(during), Some(published));
        let publish = commit.written();
        let written = mgr.begin();
        assert_eq!(
            mgr.start_epoch(written),
            Some(published),
            "written is not published"
        );
        publish.complete().unwrap();
        let after = mgr.begin();
        assert_eq!(mgr.start_epoch(after), Some(epoch));
    }

    /// Readers see a commit's epoch only once the commit is complete, so a
    /// snapshot never holds part of a commit.
    #[test]
    fn a_commit_publishes_its_epoch_when_complete() {
        let mgr = TransactionManager::new();
        let before = mgr.current_epoch();
        let tx = mgr.begin();
        let commit = mgr.start_commit(tx).unwrap();
        let epoch = commit.epoch();
        assert!(epoch > before);
        assert_eq!(mgr.current_epoch(), before);

        commit.complete().unwrap();
        assert_eq!(mgr.current_epoch(), epoch);
    }

    // ---- A commit in phases (W4.1) ----

    /// A commit whose log write failed is aborted: its epoch goes back (no
    /// later commit took one yet), nothing is published or poisoned, and the
    /// transaction is active again for the caller's rollback, still holding
    /// its claims until then. The next commit takes the same epoch.
    ///
    /// After `a_cancelled_prepare_releases_claims_without_publishing` of
    /// pull request #614 (temporaryfix), with the epoch given back.
    #[test]
    fn an_aborted_commit_gives_its_epoch_back_and_keeps_the_manager_usable() {
        let mgr = TransactionManager::new();
        let tx = mgr.begin();
        mgr.record_write(tx, NodeId::new(1)).unwrap();
        let commit = mgr.start_commit(tx).unwrap();
        assert_eq!(commit.epoch(), EpochId::new(1));
        commit.abort();

        assert_eq!(mgr.state(tx), Some(TransactionState::Active));
        assert_eq!(mgr.committed_epoch(tx), None);
        assert_eq!(mgr.current_epoch(), EpochId::new(0));
        assert!(!mgr.has_incomplete_commit());
        let other = mgr.begin();
        assert!(
            matches!(
                mgr.record_write(other, NodeId::new(1)),
                Err(Error::Transaction(TransactionError::WriteConflict(_)))
            ),
            "the transaction of an aborted commit holds its claim until it is rolled back"
        );
        mgr.abort(other).unwrap();
        mgr.abort(tx).unwrap();
        assert_eq!(mgr.active_count(), 0);
        assert!(mgr.idle_gate().is_some());

        // Every assigned epoch is published: a hold does not wait.
        std::thread::scope(|scope| {
            let _end = EndWaits(&mgr);
            let manager = &mgr;
            let (finished, _) = spawn_and_wait(scope, PATIENTLY, move || {
                drop(manager.hold_commits_for_change().unwrap());
                drop(manager.hold_commits().unwrap());
            });
            assert!(
                finished,
                "a hold waits for the epoch of the aborted commit, which nothing publishes"
            );
        });
        let next = mgr.begin();
        mgr.record_write(next, NodeId::new(1))
            .expect("the rolled-back transaction holds no claim");
        assert_eq!(
            mgr.commit(next).unwrap(),
            EpochId::new(1),
            "the next commit takes the epoch the aborted one gave back"
        );
    }

    /// A written commit has released the commit lock, so the next one starts,
    /// but its transaction is still committing: its claims stay, readers do
    /// not see its epoch, and a direct call that commits at once waits.
    #[test]
    fn claims_stay_until_publish() {
        let mgr = TransactionManager::new();
        let before = mgr.begin();
        let tx = mgr.begin();
        mgr.record_write(tx, NodeId::new(1)).unwrap();
        let publish = mgr.start_commit(tx).unwrap().written();
        let epoch = publish.epoch();

        assert_eq!(mgr.state(tx), Some(TransactionState::Committing));
        assert_eq!(mgr.current_epoch(), EpochId::new(0));
        let after = mgr.begin();
        assert_eq!(
            mgr.start_epoch(after),
            Some(EpochId::new(0)),
            "a transaction begins while a written commit waits to publish, below its epoch"
        );
        for other in [before, after] {
            assert!(matches!(
                mgr.record_write(other, NodeId::new(1)),
                Err(Error::Transaction(TransactionError::WriteConflict(_)))
            ));
        }
        assert!(mgr.idle_gate().is_none());

        // The next commit runs meanwhile, and publishes after this one.
        mgr.record_write(after, NodeId::new(2)).unwrap();
        let second = mgr.start_commit(after).unwrap().written();
        let second_epoch = second.epoch();
        assert_eq!(second_epoch.as_u64(), epoch.as_u64() + 1);

        publish.complete().unwrap();
        assert_eq!(mgr.state(tx), Some(TransactionState::Committed));
        assert_eq!(mgr.current_epoch(), epoch);
        second.complete().unwrap();
        assert_eq!(mgr.current_epoch(), second_epoch);
        mgr.abort(before).unwrap();
        let next = mgr.begin();
        mgr.record_write(next, NodeId::new(1)).unwrap();
    }

    /// Eight written commits publish in the order of their epochs, whatever
    /// order their committers finish in: none publishes before the one
    /// below it, so no reader sees an epoch with a gap under it.
    #[test]
    fn publish_follows_epoch_order_under_eight_committers() {
        let mgr = TransactionManager::new();
        let mut written: Vec<_> = (0..8)
            .map(|_| {
                let tx = mgr.begin();
                mgr.start_commit(tx).unwrap().written()
            })
            .collect();
        let epochs: Vec<u64> = written
            .iter()
            .map(|publish| publish.epoch().as_u64())
            .collect();
        assert_eq!(epochs, (1..=8).collect::<Vec<u64>>());
        let first = written.remove(0);

        let manager = &mgr;
        std::thread::scope(|scope| {
            // The highest epochs first: each waits for the one below it.
            let waiting: Vec<_> = written
                .into_iter()
                .rev()
                .map(|publish| {
                    let epoch = publish.epoch();
                    let (finished, handle) = spawn_and_wait(scope, BRIEFLY, move || {
                        publish.complete().unwrap();
                        // Published in order: nothing below is missing.
                        assert!(manager.current_epoch() >= epoch);
                    });
                    assert!(!finished, "epoch {epoch:?} was published before epoch 1");
                    handle
                })
                .collect();
            assert_eq!(mgr.current_epoch(), EpochId::new(0));
            assert!(mgr.idle_gate().is_none(), "eight commits are still open");

            first.complete().unwrap();
            for handle in waiting {
                handle.join().unwrap();
            }
        });
        assert_eq!(mgr.current_epoch(), EpochId::new(8));
        assert!(mgr.idle_gate().is_some(), "every commit is published");
        assert!(!mgr.has_incomplete_commit());
    }

    /// A checkpoint's hold, a change outside a commit and `close()` take the
    /// commit lock and then wait until every written commit is published:
    /// the committed state they see, log or close on holds every commit that
    /// is in the log.
    #[test]
    fn a_hold_waits_for_every_written_commit_to_publish() {
        type Hold = fn(&TransactionManager);
        let holds: [(&str, Hold); 3] = [
            ("hold_commits", |mgr| drop(mgr.hold_commits().unwrap())),
            ("hold_commits_for_change", |mgr| {
                drop(mgr.hold_commits_for_change().unwrap());
            }),
            ("close_for_writes", TransactionManager::close_for_writes),
        ];
        for (name, hold) in holds {
            let mgr = TransactionManager::new();
            let tx = mgr.begin();
            let publish = mgr.start_commit(tx).unwrap().written();
            let epoch = publish.epoch();
            let manager = &mgr;
            std::thread::scope(|scope| {
                let (finished, held) = spawn_and_wait(scope, BRIEFLY, move || {
                    hold(manager);
                    manager.current_epoch()
                });
                assert!(
                    !finished,
                    "{name} returned before the written commit was published"
                );
                publish.complete().unwrap();
                assert_eq!(held.join().unwrap(), epoch, "{name}");
            });
        }
    }

    /// A publish handle dropped without `complete` (the commit code panicked
    /// after its group was written) is a commit that did not complete: its
    /// epoch is never published, and neither is any epoch above it, since
    /// that would show its stamped versions. Later commits fail, a written
    /// one at its publish, and a hold fails instead of waiting for ever.
    #[test]
    fn a_dropped_publish_handle_poisons_as_an_incomplete_commit() {
        let mgr = TransactionManager::new();
        let (first, tx, later) = (mgr.begin(), mgr.begin(), mgr.begin());
        mgr.record_write(tx, NodeId::new(1)).unwrap();
        let below = mgr.start_commit(first).unwrap().written();
        let publish = mgr.start_commit(tx).unwrap().written();
        let above = mgr.start_commit(later).unwrap().written();
        let below_epoch = below.epoch();

        drop(publish);
        assert!(mgr.has_incomplete_commit());
        assert_eq!(mgr.state(tx), Some(TransactionState::Committing));
        std::thread::scope(|scope| {
            let _end = EndWaits(&mgr);
            let above = publishing(scope, above);
            assert!(matches!(
                above.recv_timeout(PATIENTLY),
                Ok(Err(Error::Transaction(TransactionError::IncompleteCommit)))
            ));
            assert_eq!(mgr.state(later), Some(TransactionState::Committing));
            assert_eq!(mgr.current_epoch(), EpochId::new(0));
            // The commit below it holds nothing of it: it is published.
            let below = publishing(scope, below);
            assert!(matches!(below.recv_timeout(PATIENTLY), Ok(Ok(()))));
        });
        assert_eq!(mgr.state(first), Some(TransactionState::Committed));
        assert_eq!(mgr.current_epoch(), below_epoch);

        let next = mgr.begin();
        assert!(matches!(
            mgr.start_commit(next),
            Err(Error::Transaction(TransactionError::IncompleteCommit))
        ));
        assert!(matches!(
            mgr.hold_commits(),
            Err(Error::Transaction(TransactionError::IncompleteCommit))
        ));
        assert!(matches!(
            mgr.hold_commits_for_change(),
            Err(Error::Transaction(TransactionError::IncompleteCommit))
        ));
        // Returns: the database closes on what was published.
        mgr.close_for_writes();
        assert_eq!(mgr.current_epoch(), below_epoch);
    }

    /// After a broken invariant ([`TransactionManager::poison`]) nothing is
    /// published any more, a written commit included: no commit may build
    /// on that store.
    #[test]
    fn a_poisoned_manager_publishes_nothing_more() {
        let mgr = TransactionManager::new();
        let tx = mgr.begin();
        let written = mgr.start_commit(tx).unwrap().written();
        mgr.poison("an undo failed");
        std::thread::scope(|scope| {
            let _end = EndWaits(&mgr);
            let written = publishing(scope, written);
            assert!(matches!(
                written.recv_timeout(PATIENTLY),
                Ok(Err(Error::Transaction(TransactionError::IncompleteCommit)))
            ));
        });
        assert_eq!(mgr.current_epoch(), EpochId::new(0));
    }

    /// A commit that waits to publish when a commit below it does not
    /// complete is woken and fails: it does not wait for ever.
    #[test]
    fn a_waiting_publish_fails_when_the_commit_below_it_does_not_complete() {
        let mgr = TransactionManager::new();
        let tx = mgr.begin();
        let later = mgr.begin();
        let publish = mgr.start_commit(tx).unwrap().written();
        let above = mgr.start_commit(later).unwrap().written();
        std::thread::scope(|scope| {
            let _end = EndWaits(&mgr);
            let above = publishing(scope, above);
            assert!(
                above.recv_timeout(BRIEFLY).is_err(),
                "published before the commit below it"
            );
            drop(publish);
            assert!(matches!(
                above.recv_timeout(PATIENTLY),
                Ok(Err(Error::Transaction(TransactionError::IncompleteCommit)))
            ));
        });
        assert_eq!(mgr.current_epoch(), EpochId::new(0));
    }

    /// Ends every wait for an epoch when a test fails, so a failed assertion
    /// fails the test instead of leaving a thread waiting for a commit that
    /// never publishes.
    struct EndWaits<'a>(&'a TransactionManager);

    impl Drop for EndWaits<'_> {
        fn drop(&mut self) {
            if std::thread::panicking() {
                // Both ways a wait ends: refused, or its turn has come.
                self.0.incomplete_from(EpochId::new(0));
                self.0.sync_epoch(EpochId::new(u64::MAX - 1));
            }
        }
    }

    /// Publishes `publish` on a scoped thread; the result arrives on the
    /// returned channel once it is published or refused.
    fn publishing<'scope>(
        scope: &'scope std::thread::Scope<'scope, '_>,
        publish: Publish<'scope>,
    ) -> std::sync::mpsc::Receiver<Result<()>> {
        let (done, result) = std::sync::mpsc::channel();
        scope.spawn(move || {
            let _ = done.send(publish.complete());
        });
        result
    }

    /// `complete` on the guard publishes while it holds the commit lock, as
    /// every caller did before the phases: nothing starts in between.
    #[test]
    fn a_commit_completed_under_its_lock_is_published_at_once() {
        let mgr = TransactionManager::new();
        let tx = mgr.begin();
        let commit = mgr.start_commit(tx).unwrap();
        let epoch = commit.epoch();
        commit.complete().unwrap();
        assert_eq!(mgr.current_epoch(), epoch);
        assert_eq!(mgr.state(tx), Some(TransactionState::Committed));
        assert!(mgr.idle_gate().is_some());
    }

    /// A commit that does not complete (its code panicked between the commit
    /// decision and `complete`) is not reported as committed and does not
    /// publish its epoch: its writes stay locked against other transactions,
    /// which get the error of the failed commit (not a retryable conflict)
    /// when they write, and the commit lock is released, so transactions
    /// still begin (no transaction commits afterwards, see the next test).
    #[test]
    fn a_commit_that_does_not_complete_keeps_its_writes_locked() {
        let mgr = TransactionManager::new();
        let before = mgr.current_epoch();
        let tx = mgr.begin();
        let other = mgr.begin();
        mgr.record_write(tx, NodeId::new(1)).unwrap();

        let unwound = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _commit = mgr.start_commit(tx).unwrap();
            panic!("stamping failed");
        }));
        assert!(unwound.is_err());

        assert_eq!(mgr.state(tx), Some(TransactionState::Committing));
        assert_eq!(mgr.current_epoch(), before);
        assert!(matches!(
            mgr.record_write(other, NodeId::new(1)),
            Err(Error::Transaction(TransactionError::IncompleteCommit))
        ));
        let next = mgr.begin();
        assert_eq!(mgr.start_epoch(next), Some(before));
    }

    /// A commit that does not complete may have stamped versions with its
    /// epoch, which readers never see. A later commit would publish a higher
    /// epoch, and with it that part of the failed commit, so no transaction
    /// commits afterwards (not even one that wrote nothing) and the published
    /// epoch never moves past the failed one. Transactions still begin, at
    /// the last published epoch.
    #[test]
    fn after_a_commit_that_does_not_complete_no_transaction_commits() {
        let mgr = TransactionManager::new();
        let first = mgr.begin();
        let published = mgr.commit(first).unwrap();

        let tx = mgr.begin();
        mgr.record_write(tx, NodeId::new(3)).unwrap();
        let commit = mgr.start_commit(tx).unwrap();
        let failed = commit.epoch();
        drop(commit);

        for wrote in [true, false] {
            let next = mgr.begin();
            assert_eq!(
                mgr.start_epoch(next),
                Some(published),
                "a new transaction reads what was published before the failed commit"
            );
            if wrote {
                assert!(
                    matches!(
                        mgr.record_write(next, NodeId::new(19)),
                        Err(Error::Transaction(TransactionError::IncompleteCommit))
                    ),
                    "a write fails with the failed commit's error"
                );
            }
            let Err(error) = mgr.start_commit(next) else {
                panic!("a commit after an incomplete one (wrote: {wrote}) succeeded");
            };
            assert!(
                matches!(
                    &error,
                    Error::Transaction(TransactionError::IncompleteCommit)
                ),
                "a typed error: {error:?}"
            );
            assert_eq!(error.error_code().as_str(), "GRAFEO-T008");
            let message = error.to_string();
            assert!(
                message.contains("did not complete") && message.contains("reopen"),
                "the error says a commit did not complete and the database must be \
                 reopened: {message}"
            );
            assert!(
                mgr.commit(next).is_err(),
                "commit refuses as start_commit does"
            );
            mgr.abort(next).unwrap();
        }
        assert_eq!(
            mgr.current_epoch(),
            published,
            "the published epoch stays before the failed epoch {failed:?}"
        );
        assert_eq!(mgr.state(tx), Some(TransactionState::Committing));
    }

    /// `commit` completes at once: nothing waits for it afterwards.
    #[test]
    fn commit_completes_at_once() {
        let mgr = TransactionManager::new();
        let tx = mgr.begin();
        mgr.record_write(tx, NodeId::new(1)).unwrap();
        mgr.commit(tx).unwrap();

        assert_eq!(mgr.state(tx), Some(TransactionState::Committed));
        assert!(mgr.idle_gate().is_some());
        let next = mgr.begin();
        mgr.record_write(next, NodeId::new(1)).unwrap();
    }

    /// How long work that should wait gets to finish anyway.
    const BRIEFLY: std::time::Duration = std::time::Duration::from_millis(100);

    /// How long work that should not wait gets to finish: long, so a loaded
    /// machine does not fail the test, and finite, so a wait fails it
    /// instead of hanging.
    const PATIENTLY: std::time::Duration = std::time::Duration::from_secs(30);

    /// Runs `work` on a scoped thread; returns whether it finished within
    /// `wait`, and its handle.
    fn spawn_and_wait<'scope, T: Send + 'scope>(
        scope: &'scope std::thread::Scope<'scope, '_>,
        wait: std::time::Duration,
        work: impl FnOnce() -> T + Send + 'scope,
    ) -> (bool, std::thread::ScopedJoinHandle<'scope, T>) {
        let (done, finished) = std::sync::mpsc::channel();
        let handle = scope.spawn(move || {
            let result = work();
            let _ = done.send(());
            result
        });
        (finished.recv_timeout(wait).is_ok(), handle)
    }

    /// A checkpoint's hold waits for a store change of an open transaction
    /// in progress, and once it holds commits off, a store change waits for
    /// it: the image is read from a store that does not move.
    #[test]
    fn a_checkpoint_hold_and_a_write_in_progress_wait_for_each_other() {
        let mgr = TransactionManager::new();
        let manager = &mgr;
        std::thread::scope(|scope| {
            let writing = mgr.write_in_progress();
            let (finished, checkpoint) = spawn_and_wait(scope, BRIEFLY, move || {
                drop(manager.hold_commits().unwrap());
            });
            assert!(!finished, "the hold waits for the write in progress");
            drop(writing);
            checkpoint.join().unwrap();

            let held = mgr.hold_commits().unwrap();
            let (finished, write) = spawn_and_wait(scope, BRIEFLY, move || {
                drop(manager.write_in_progress());
            });
            assert!(!finished, "a write waits for the checkpoint's hold");
            drop(held);
            write.join().unwrap();
        });
    }

    /// Writes in progress do not wait for each other, and a change outside
    /// any commit (a schema statement, a graph command) holds commits off
    /// without freezing the store: writes of open
    /// transactions go on, and it does not wait for them.
    #[test]
    fn a_change_outside_a_commit_does_not_freeze_the_store() {
        let mgr = TransactionManager::new();
        let manager = &mgr;
        std::thread::scope(|scope| {
            let writing = mgr.write_in_progress();
            let (finished, other) = spawn_and_wait(scope, PATIENTLY, move || {
                drop(manager.write_in_progress());
            });
            assert!(finished, "another write runs alongside");
            other.join().unwrap();

            let (finished, change) = spawn_and_wait(scope, PATIENTLY, move || {
                drop(manager.hold_commits_for_change().unwrap());
            });
            assert!(finished, "the change does not wait for the write");
            change.join().unwrap();
            drop(writing);

            let held = mgr.hold_commits_for_change().unwrap();
            let (finished, write) = spawn_and_wait(scope, PATIENTLY, move || {
                drop(manager.write_in_progress());
            });
            assert!(finished, "a write does not wait for the change");
            write.join().unwrap();
            drop(held);
        });
    }

    // === Edge endpoints claimed against deletes ===

    /// Alix, Gus and Vincent as entities of the default graph.
    fn people() -> (GraphEntity, GraphEntity, GraphEntity) {
        (
            GraphEntity::from(NodeId::new(3)),
            GraphEntity::from(NodeId::new(19)),
            GraphEntity::from(NodeId::new(88)),
        )
    }

    /// Whether `result` is a write conflict.
    fn is_conflict(result: &Result<impl std::fmt::Debug>) -> bool {
        matches!(
            result,
            Err(Error::Transaction(TransactionError::WriteConflict(_)))
        )
    }

    /// A delete of a node that an open transaction claimed as an edge's
    /// endpoint is a conflict; a delete of another node is not.
    #[test]
    fn a_delete_conflicts_with_an_open_claim_on_the_node() {
        let mgr = TransactionManager::new();
        let (alix, gus, vincent) = people();
        let linker = mgr.begin();
        let deleter = mgr.begin();
        mgr.record_endpoints(linker, [alix, gus.clone()]).unwrap();

        let deleted = mgr.record_delete(deleter, gus);
        assert!(is_conflict(&deleted), "got {deleted:?}");
        mgr.record_delete(deleter, vincent).unwrap();
        mgr.commit(linker).unwrap();
        mgr.commit(deleter).unwrap();
    }

    /// A claim on a node an open transaction deletes is a conflict, and
    /// claims neither endpoint: once the delete is rolled back, another
    /// transaction deletes the other endpoint without a conflict.
    #[test]
    fn a_claim_conflicts_with_an_open_delete_and_claims_nothing() {
        let mgr = TransactionManager::new();
        let (alix, gus, _) = people();
        let deleter = mgr.begin();
        let linker = mgr.begin();
        mgr.record_delete(deleter, gus.clone()).unwrap();

        let claimed = mgr.record_endpoints(linker, [alix.clone(), gus]);
        assert!(is_conflict(&claimed), "got {claimed:?}");
        mgr.abort(deleter).unwrap();
        let other = mgr.begin();
        mgr.record_delete(other, alix)
            .expect("the failed claim left Alix unclaimed");
    }

    /// Claims conflict with deletes only: transactions that create edges to
    /// one node, and one that writes the node, all commit.
    #[test]
    fn claims_do_not_conflict_with_claims_or_writes() {
        let mgr = TransactionManager::new();
        let (alix, gus, vincent) = people();
        let alix_links = mgr.begin();
        let vincent_links = mgr.begin();
        let writer = mgr.begin();
        mgr.record_endpoints(alix_links, [alix, gus.clone()])
            .unwrap();
        mgr.record_endpoints(vincent_links, [vincent, gus.clone()])
            .unwrap();
        mgr.record_write(writer, gus).unwrap();
        mgr.commit(alix_links).unwrap();
        mgr.commit(vincent_links).unwrap();
        mgr.commit(writer).unwrap();
    }

    /// A transaction whose new edge ends at a node another transaction
    /// deleted and committed after it began fails its commit.
    #[test]
    fn a_claim_fails_the_commit_after_a_later_committed_delete() {
        let mgr = TransactionManager::new();
        let (alix, gus, _) = people();
        let linker = mgr.begin();
        let deleter = mgr.begin();
        mgr.record_delete(deleter, gus.clone()).unwrap();
        mgr.commit(deleter).unwrap();

        mgr.record_endpoints(linker, [alix, gus])
            .expect("the delete is committed: no open transaction holds it");
        let committed = mgr.commit(linker);
        assert!(is_conflict(&committed), "got {committed:?}");
    }

    /// The other way around: a delete fails its commit when another
    /// transaction committed an edge to the node after the delete's
    /// transaction began.
    #[test]
    fn a_delete_fails_the_commit_after_a_later_committed_claim() {
        let mgr = TransactionManager::new();
        let (alix, gus, _) = people();
        let deleter = mgr.begin();
        let linker = mgr.begin();
        mgr.record_endpoints(linker, [alix, gus.clone()]).unwrap();
        mgr.commit(linker).unwrap();

        mgr.record_delete(deleter, gus).unwrap();
        let committed = mgr.commit(deleter);
        assert!(is_conflict(&committed), "got {committed:?}");
    }

    /// A claim committed before a transaction began is part of its snapshot
    /// (it sees the edge, and deletes it with the node): no conflict.
    #[test]
    fn a_claim_committed_before_the_transaction_began_does_not_conflict() {
        let mgr = TransactionManager::new();
        let (alix, gus, _) = people();
        let linker = mgr.begin();
        mgr.record_endpoints(linker, [alix, gus.clone()]).unwrap();
        mgr.commit(linker).unwrap();

        let deleter = mgr.begin();
        mgr.record_delete(deleter, gus).unwrap();
        mgr.commit(deleter).unwrap();
    }

    /// Claims of one node id in two graphs are claims of two nodes: a delete
    /// in one graph does not conflict with an edge in the other.
    #[test]
    fn claims_in_different_graphs_do_not_conflict() {
        let mgr = TransactionManager::new();
        let gus = NodeId::new(19);
        let linker = mgr.begin();
        let deleter = mgr.begin();
        mgr.record_endpoints(
            linker,
            [
                GraphEntity::new(Some(Arc::from("Paris")), gus),
                GraphEntity::new(Some(Arc::from("Paris")), gus),
            ],
        )
        .unwrap();
        mgr.record_delete(deleter, GraphEntity::new(Some(Arc::from("Prague")), gus))
            .unwrap();
        mgr.commit(linker).unwrap();
        mgr.commit(deleter).unwrap();
    }
}
