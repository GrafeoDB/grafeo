//! Transaction manager.

use std::collections::HashSet;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use grafeo_common::types::{EdgeId, EpochId, NodeId, TransactionId};
use grafeo_common::utils::error::{Error, Result, TransactionError};
use grafeo_common::utils::hash::FxHashMap;
use parking_lot::{Mutex, MutexGuard, RwLock};

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
    pub write_set: HashSet<GraphEntity>,
    /// Set of entities read by this transaction (for serializable isolation).
    pub read_set: HashSet<GraphEntity>,
}

impl TransactionInfo {
    /// Creates a new transaction info with the given isolation level.
    fn new(start_epoch: EpochId, isolation_level: IsolationLevel) -> Self {
        Self {
            state: TransactionState::Active,
            isolation_level,
            start_epoch,
            write_set: HashSet::new(),
            read_set: HashSet::new(),
        }
    }
}

/// Manages transactions and MVCC versioning.
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
    /// Held by a write outside any transaction (see [`idle_gate`](Self::idle_gate))
    /// and briefly by every [`begin`](Self::begin), so no transaction starts
    /// while such a write runs.
    idle_gate: Mutex<()>,
    /// Held by a commit from its epoch until it is complete (see
    /// [`CommitGuard`]) and briefly by every [`begin`](Self::begin): commits
    /// complete one at a time, in epoch order, and no transaction starts in
    /// the middle of one.
    commit_lock: Mutex<()>,
}

/// A commit in progress, from [`TransactionManager::start_commit`] until
/// [`complete`](Self::complete). Until then the transaction is
/// [`TransactionState::Committing`]: it still counts as open, so no write
/// outside a transaction can start (see [`TransactionManager::idle_gate`]),
/// its writes still conflict with other transactions' writes, readers do not
/// see its epoch yet, and no other commit and no
/// [`begin`](TransactionManager::begin) can run. Complete it once the
/// commit's versions, events and WAL records are written.
///
/// A guard dropped without `complete` (only when the commit code panics) does
/// not report the transaction as committed: it stays `Committing`, so its
/// half-written entities stay locked against other writers, and only the
/// commit lock is released.
#[must_use = "a commit is complete only after `complete()`"]
pub(crate) struct CommitGuard<'a> {
    manager: &'a TransactionManager,
    transaction_id: TransactionId,
    epoch: EpochId,
    completed: bool,
    _commit: MutexGuard<'a, ()>,
}

impl CommitGuard<'_> {
    /// The commit epoch.
    pub(crate) fn epoch(&self) -> EpochId {
        self.epoch
    }

    /// Completes the commit: the transaction is committed, readers see its
    /// epoch, and what waited for it can run.
    pub(crate) fn complete(mut self) {
        self.completed = true;
    }
}

impl Drop for CommitGuard<'_> {
    fn drop(&mut self) {
        if !self.completed {
            grafeo_common::grafeo_error!(
                "commit of transaction {:?} at epoch {:?} did not complete; its writes stay locked",
                self.transaction_id,
                self.epoch
            );
            return;
        }
        if let Some(info) = self
            .manager
            .transactions
            .write()
            .get_mut(&self.transaction_id)
        {
            info.state = TransactionState::Committed;
        }
        // Commits complete one at a time and in epoch order (the commit
        // lock), so the published epoch only moves forward. It is published
        // before the count drops, so a write outside a transaction that finds
        // no transaction open also finds this commit published.
        self.manager
            .published_epoch
            .fetch_max(self.epoch.as_u64(), Ordering::Release);
        // Release pairs with the acquire in `idle_gate`: a write outside a
        // transaction sees everything the commit wrote.
        self.manager.active_count.fetch_sub(1, Ordering::Release);
        // The commit lock is released after this, with `_commit`.
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
        }
    }

    /// Begins a new transaction with the default isolation level (Snapshot Isolation).
    pub fn begin(&self) -> TransactionId {
        self.begin_with_isolation(IsolationLevel::default())
    }

    /// Begins a new transaction with the specified isolation level.
    pub fn begin_with_isolation(&self, isolation_level: IsolationLevel) -> TransactionId {
        // Wait for a write outside any transaction to finish, and for a
        // commit in progress: the snapshot holds every commit up to its
        // epoch, complete.
        let _gate = self.idle_gate.lock();
        let _commit = self.commit_lock.lock();
        let transaction_id =
            TransactionId::new(self.next_transaction_id.fetch_add(1, Ordering::Relaxed));
        let epoch = self.current_epoch();

        let info = TransactionInfo::new(epoch, isolation_level);
        self.transactions.write().insert(transaction_id, info);
        self.active_count.fetch_add(1, Ordering::Relaxed);
        transaction_id
    }

    /// Exclusive access for a write outside any transaction: `Some` while no
    /// transaction is open. No transaction can begin until the guard is
    /// dropped, so such a write cannot conflict with one and needs no
    /// conflict tracking.
    pub(crate) fn idle_gate(&self) -> Option<MutexGuard<'_, ()>> {
        let gate = self.idle_gate.lock();
        (self.active_count.load(Ordering::Acquire) == 0).then_some(gate)
    }

    /// A transaction id that no transaction uses, for a write outside any
    /// transaction that versions its changes so it can undo them (a batch
    /// holding [`idle_gate`](Self::idle_gate)).
    pub(crate) fn reserve_transaction_id(&self) -> TransactionId {
        TransactionId::new(self.next_transaction_id.fetch_add(1, Ordering::Relaxed))
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
        let entity = entity.into();
        let mut txns = self.transactions.write();

        // First-writer-wins conflict detection. Skip the scan when only one
        // transaction is active (common case for auto-commit). A commit in
        // progress still holds its writes.
        if self.active_count.load(Ordering::Relaxed) > 1 {
            for (other_tx, other_info) in txns.iter() {
                if *other_tx != transaction_id
                    && matches!(
                        other_info.state,
                        TransactionState::Active | TransactionState::Committing
                    )
                    && other_info.write_set.contains(&entity)
                {
                    return Err(Error::Transaction(TransactionError::WriteConflict(
                        format!("Write-write conflict on entity {entity}"),
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

        info.write_set.insert(entity);
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
    pub fn commit(&self, transaction_id: TransactionId) -> Result<EpochId> {
        let commit = self.start_commit(transaction_id)?;
        let epoch = commit.epoch();
        commit.complete();
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
            manager: self,
            transaction_id,
            epoch: commit_epoch,
            completed: false,
            _commit: commit_lock,
        })
    }

    /// Aborts a transaction.
    ///
    /// # Errors
    ///
    /// Returns an error if the transaction is not active.
    pub fn abort(&self, transaction_id: TransactionId) -> Result<()> {
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
    pub fn get_write_set(&self, transaction_id: TransactionId) -> Result<HashSet<GraphEntity>> {
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
        write_set: HashSet<GraphEntity>,
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
    }

    /// Returns the minimum epoch that must be preserved for active transactions.
    ///
    /// This is used for garbage collection - versions visible at this epoch
    /// must be preserved.
    #[must_use]
    pub fn min_active_epoch(&self) -> EpochId {
        let txns = self.transactions.read();
        txns.values()
            .filter(|info| info.state == TransactionState::Active)
            .map(|info| info.start_epoch)
            .min()
            .unwrap_or_else(|| self.current_epoch())
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

    #[test]
    fn test_min_active_epoch() {
        let mgr = TransactionManager::new();

        // No active transactions - should return current epoch
        assert_eq!(mgr.min_active_epoch(), mgr.current_epoch());

        // Start some transactions
        let tx1 = mgr.begin();
        let epoch1 = mgr.start_epoch(tx1).unwrap();

        // Advance epoch
        let tx2 = mgr.begin();
        mgr.commit(tx2).unwrap();

        let _tx3 = mgr.begin();

        // min_active_epoch should be tx1's start epoch (earliest active)
        assert_eq!(mgr.min_active_epoch(), epoch1);
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

    /// Until a commit is complete, a write outside any transaction cannot
    /// start and another transaction cannot write what the committing one
    /// wrote; dropping the guard allows both again.
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
            "a write outside a transaction waits for the commit"
        );

        commit.complete();
        assert_eq!(mgr.state(tx), Some(TransactionState::Committed));
        assert!(mgr.idle_gate().is_some());
        let next = mgr.begin();
        mgr.record_write(next, NodeId::new(1)).unwrap();
    }

    /// A transaction begins between commits, never during one: it waits for
    /// the commit in progress and starts at its epoch.
    #[test]
    fn begin_waits_for_a_commit_in_progress() {
        let mgr = TransactionManager::new();
        let tx = mgr.begin();
        let commit = mgr.start_commit(tx).unwrap();
        let epoch = commit.epoch();

        let manager = &mgr;
        std::thread::scope(|scope| {
            let (began, waited) = std::sync::mpsc::channel();
            let late = scope.spawn(move || {
                let late = manager.begin();
                began.send(()).unwrap();
                late
            });
            assert!(
                waited
                    .recv_timeout(std::time::Duration::from_millis(100))
                    .is_err(),
                "begin returned while the commit was in progress"
            );
            commit.complete();
            let late = late.join().unwrap();
            assert_eq!(mgr.start_epoch(late), Some(epoch));
        });
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

        commit.complete();
        assert_eq!(mgr.current_epoch(), epoch);
    }

    /// A commit that does not complete (its code panicked between the commit
    /// decision and `complete`) is not reported as committed and does not
    /// publish its epoch: its writes stay locked against other transactions,
    /// and the commit lock is released, so the database keeps working.
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
            Err(Error::Transaction(TransactionError::WriteConflict(_)))
        ));
        let next = mgr.begin();
        assert_eq!(mgr.start_epoch(next), Some(before));
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
}
