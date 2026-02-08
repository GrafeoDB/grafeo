//! Transaction manager.

use std::collections::HashSet;
use std::sync::atomic::{AtomicU64, Ordering};

use grafeo_common::types::{EdgeId, EpochId, NodeId, TxId};
use grafeo_common::utils::error::{Error, Result, TransactionError};
use grafeo_common::utils::hash::FxHashMap;
use parking_lot::RwLock;

/// State of a transaction.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TxState {
    /// Transaction is active.
    Active,
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

/// Information about an active transaction.
pub struct TxInfo {
    /// Transaction state.
    pub state: TxState,
    /// Isolation level for this transaction.
    pub isolation_level: IsolationLevel,
    /// Start epoch (snapshot epoch for reads).
    pub start_epoch: EpochId,
    /// Set of entities written by this transaction.
    pub write_set: HashSet<EntityId>,
    /// Set of entities read by this transaction (for serializable isolation).
    pub read_set: HashSet<EntityId>,
}

impl TxInfo {
    /// Creates a new transaction info with the given isolation level.
    fn new(start_epoch: EpochId, isolation_level: IsolationLevel) -> Self {
        Self {
            state: TxState::Active,
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
    next_tx_id: AtomicU64,
    /// Current epoch.
    current_epoch: AtomicU64,
    /// Active transactions.
    transactions: RwLock<FxHashMap<TxId, TxInfo>>,
    /// Committed transaction epochs (for conflict detection).
    /// Maps TxId -> commit epoch.
    committed_epochs: RwLock<FxHashMap<TxId, EpochId>>,
}

impl TransactionManager {
    /// Creates a new transaction manager.
    #[must_use]
    pub fn new() -> Self {
        Self {
            // Start at 2 to avoid collision with TxId::SYSTEM (which is 1)
            // TxId::INVALID = 0, TxId::SYSTEM = 1, user transactions start at 2
            next_tx_id: AtomicU64::new(2),
            current_epoch: AtomicU64::new(0),
            transactions: RwLock::new(FxHashMap::default()),
            committed_epochs: RwLock::new(FxHashMap::default()),
        }
    }

    /// Begins a new transaction with the default isolation level (Snapshot Isolation).
    pub fn begin(&self) -> TxId {
        self.begin_with_isolation(IsolationLevel::default())
    }

    /// Begins a new transaction with the specified isolation level.
    pub fn begin_with_isolation(&self, isolation_level: IsolationLevel) -> TxId {
        let tx_id = TxId::new(self.next_tx_id.fetch_add(1, Ordering::Relaxed));
        let epoch = EpochId::new(self.current_epoch.load(Ordering::Acquire));

        let info = TxInfo::new(epoch, isolation_level);
        self.transactions.write().insert(tx_id, info);
        tx_id
    }

    /// Returns the isolation level of a transaction.
    pub fn isolation_level(&self, tx_id: TxId) -> Option<IsolationLevel> {
        self.transactions
            .read()
            .get(&tx_id)
            .map(|info| info.isolation_level)
    }

    /// Records a write operation for the transaction.
    ///
    /// # Errors
    ///
    /// Returns an error if the transaction is not active.
    pub fn record_write(&self, tx_id: TxId, entity: impl Into<EntityId>) -> Result<()> {
        let mut txns = self.transactions.write();
        let info = txns.get_mut(&tx_id).ok_or_else(|| {
            Error::Transaction(TransactionError::InvalidState(
                "Transaction not found".to_string(),
            ))
        })?;

        if info.state != TxState::Active {
            return Err(Error::Transaction(TransactionError::InvalidState(
                "Transaction is not active".to_string(),
            )));
        }

        info.write_set.insert(entity.into());
        Ok(())
    }

    /// Records a read operation for the transaction (for serializable isolation).
    ///
    /// # Errors
    ///
    /// Returns an error if the transaction is not active.
    pub fn record_read(&self, tx_id: TxId, entity: impl Into<EntityId>) -> Result<()> {
        let mut txns = self.transactions.write();
        let info = txns.get_mut(&tx_id).ok_or_else(|| {
            Error::Transaction(TransactionError::InvalidState(
                "Transaction not found".to_string(),
            ))
        })?;

        if info.state != TxState::Active {
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
    pub fn commit(&self, tx_id: TxId) -> Result<EpochId> {
        let mut txns = self.transactions.write();
        let committed = self.committed_epochs.read();

        // First, validate the transaction exists and is active
        let (our_isolation, our_start_epoch, our_write_set, our_read_set) = {
            let info = txns.get(&tx_id).ok_or_else(|| {
                Error::Transaction(TransactionError::InvalidState(
                    "Transaction not found".to_string(),
                ))
            })?;

            if info.state != TxState::Active {
                return Err(Error::Transaction(TransactionError::InvalidState(
                    "Transaction is not active".to_string(),
                )));
            }

            (
                info.isolation_level,
                info.start_epoch,
                info.write_set.clone(),
                info.read_set.clone(),
            )
        };

        // Check for write-write conflicts with other committed transactions
        for (other_tx, other_info) in txns.iter() {
            if *other_tx == tx_id {
                continue;
            }
            if other_info.state == TxState::Committed {
                // Check if any of our writes conflict with their writes
                for entity in &our_write_set {
                    if other_info.write_set.contains(entity) {
                        return Err(Error::Transaction(TransactionError::WriteConflict(
                            format!("Write-write conflict on entity {:?}", entity),
                        )));
                    }
                }
            }
        }

        // Also check against recently committed transactions
        for (other_tx, commit_epoch) in committed.iter() {
            if *other_tx != tx_id && commit_epoch.as_u64() > our_start_epoch.as_u64() {
                // Check if that transaction wrote to any of our entities
                if let Some(other_info) = txns.get(other_tx) {
                    for entity in &our_write_set {
                        if other_info.write_set.contains(entity) {
                            return Err(Error::Transaction(TransactionError::WriteConflict(
                                format!("Write-write conflict on entity {:?}", entity),
                            )));
                        }
                    }
                }
            }
        }

        // SSI validation for Serializable isolation level
        // Check for read-write conflicts: if we read an entity that another
        // transaction (that committed after we started) wrote, we have a
        // "rw-antidependency" which can cause write skew.
        if our_isolation == IsolationLevel::Serializable && !our_read_set.is_empty() {
            for (other_tx, commit_epoch) in committed.iter() {
                if *other_tx != tx_id && commit_epoch.as_u64() > our_start_epoch.as_u64() {
                    // Check if that transaction wrote to any entity we read
                    if let Some(other_info) = txns.get(other_tx) {
                        for entity in &our_read_set {
                            if other_info.write_set.contains(entity) {
                                return Err(Error::Transaction(
                                    TransactionError::SerializationFailure(format!(
                                        "Read-write conflict on entity {:?}: \
                                         another transaction modified data we read",
                                        entity
                                    )),
                                ));
                            }
                        }
                    }
                }
            }

            // Also check against transactions that are already marked committed
            // but not yet in committed_epochs map
            for (other_tx, other_info) in txns.iter() {
                if *other_tx == tx_id {
                    continue;
                }
                if other_info.state == TxState::Committed {
                    // If we can see their write set and we read something they wrote
                    for entity in &our_read_set {
                        if other_info.write_set.contains(entity) {
                            // Check if they committed after we started
                            if let Some(commit_epoch) = committed.get(other_tx)
                                && commit_epoch.as_u64() > our_start_epoch.as_u64()
                            {
                                return Err(Error::Transaction(
                                    TransactionError::SerializationFailure(format!(
                                        "Read-write conflict on entity {:?}: \
                                             another transaction modified data we read",
                                        entity
                                    )),
                                ));
                            }
                        }
                    }
                }
            }
        }

        // Commit successful - advance epoch atomically
        // SeqCst ensures all threads see commits in a consistent total order
        let commit_epoch = EpochId::new(self.current_epoch.fetch_add(1, Ordering::SeqCst) + 1);

        // Now update state
        if let Some(info) = txns.get_mut(&tx_id) {
            info.state = TxState::Committed;
        }

        // Record commit epoch (need to drop read lock first)
        drop(committed);
        self.committed_epochs.write().insert(tx_id, commit_epoch);

        Ok(commit_epoch)
    }

    /// Aborts a transaction.
    ///
    /// # Errors
    ///
    /// Returns an error if the transaction is not active.
    pub fn abort(&self, tx_id: TxId) -> Result<()> {
        let mut txns = self.transactions.write();

        let info = txns.get_mut(&tx_id).ok_or_else(|| {
            Error::Transaction(TransactionError::InvalidState(
                "Transaction not found".to_string(),
            ))
        })?;

        if info.state != TxState::Active {
            return Err(Error::Transaction(TransactionError::InvalidState(
                "Transaction is not active".to_string(),
            )));
        }

        info.state = TxState::Aborted;
        Ok(())
    }

    /// Returns the write set of a transaction.
    ///
    /// This returns a copy of the entities written by this transaction,
    /// used for rollback to discard uncommitted versions.
    pub fn get_write_set(&self, tx_id: TxId) -> Result<HashSet<EntityId>> {
        let txns = self.transactions.read();
        let info = txns.get(&tx_id).ok_or_else(|| {
            Error::Transaction(TransactionError::InvalidState(
                "Transaction not found".to_string(),
            ))
        })?;
        Ok(info.write_set.clone())
    }

    /// Aborts all active transactions.
    ///
    /// Used during database shutdown.
    pub fn abort_all_active(&self) {
        let mut txns = self.transactions.write();
        for info in txns.values_mut() {
            if info.state == TxState::Active {
                info.state = TxState::Aborted;
            }
        }
    }

    /// Returns the state of a transaction.
    pub fn state(&self, tx_id: TxId) -> Option<TxState> {
        self.transactions.read().get(&tx_id).map(|info| info.state)
    }

    /// Returns the start epoch of a transaction.
    pub fn start_epoch(&self, tx_id: TxId) -> Option<EpochId> {
        self.transactions
            .read()
            .get(&tx_id)
            .map(|info| info.start_epoch)
    }

    /// Returns the current epoch.
    #[must_use]
    pub fn current_epoch(&self) -> EpochId {
        EpochId::new(self.current_epoch.load(Ordering::Acquire))
    }

    /// Returns the minimum epoch that must be preserved for active transactions.
    ///
    /// This is used for garbage collection - versions visible at this epoch
    /// must be preserved.
    #[must_use]
    pub fn min_active_epoch(&self) -> EpochId {
        let txns = self.transactions.read();
        txns.values()
            .filter(|info| info.state == TxState::Active)
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
            .filter(|info| info.state == TxState::Active)
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
            .filter(|info| info.state == TxState::Active)
            .map(|info| info.start_epoch)
            .min();

        let initial_count = txns.len();

        // Collect transactions safe to remove
        let to_remove: Vec<TxId> = txns
            .iter()
            .filter(|(tx_id, info)| {
                match info.state {
                    TxState::Active => false, // Never remove active transactions
                    TxState::Aborted => true, // Always safe to remove aborted transactions
                    TxState::Committed => {
                        // Only remove committed transactions if their commit epoch
                        // is older than all active transactions' start epochs
                        if let Some(min_start) = min_active_start {
                            if let Some(commit_epoch) = committed.get(*tx_id) {
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
    pub fn mark_committed(&self, tx_id: TxId, epoch: EpochId) {
        self.committed_epochs.write().insert(tx_id, epoch);
    }

    /// Returns the last assigned transaction ID.
    ///
    /// Returns `None` if no transactions have been started yet.
    #[must_use]
    pub fn last_assigned_tx_id(&self) -> Option<TxId> {
        let next = self.next_tx_id.load(Ordering::Relaxed);
        if next > 1 {
            Some(TxId::new(next - 1))
        } else {
            None
        }
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
        assert_eq!(mgr.state(tx), Some(TxState::Active));

        let commit_epoch = mgr.commit(tx).unwrap();
        assert_eq!(mgr.state(tx), Some(TxState::Committed));
        assert!(commit_epoch.as_u64() > 0);
    }

    #[test]
    fn test_begin_abort() {
        let mgr = TransactionManager::new();

        let tx = mgr.begin();
        mgr.abort(tx).unwrap();
        assert_eq!(mgr.state(tx), Some(TxState::Aborted));
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
        assert_eq!(mgr.state(tx1), Some(TxState::Committed));
        assert_eq!(mgr.state(tx2), Some(TxState::Active));
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
        assert_eq!(mgr.state(tx2), Some(TxState::Committed)); // Preserved for conflict detection
        assert_eq!(mgr.state(tx3), Some(TxState::Active));

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
        assert_eq!(mgr.state(tx2), Some(TxState::Active));
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

        assert_eq!(mgr.state(tx1), Some(TxState::Committed)); // Already committed
        assert_eq!(mgr.state(tx2), Some(TxState::Aborted));
        assert_eq!(mgr.state(tx3), Some(TxState::Aborted));
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

        // Both try to write to the same entity
        let entity = NodeId::new(42);
        mgr.record_write(tx1, entity).unwrap();
        mgr.record_write(tx2, entity).unwrap();

        // First commit succeeds
        let result1 = mgr.commit(tx1);
        assert!(result1.is_ok());

        // Second commit should fail due to write-write conflict
        let result2 = mgr.commit(tx2);
        assert!(result2.is_err());
        assert!(
            result2
                .unwrap_err()
                .to_string()
                .contains("Write-write conflict"),
            "Expected write-write conflict error"
        );
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
        assert_eq!(
            mgr.current_epoch().as_u64(),
            (num_threads * commits_per_thread) as u64,
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

        let tx_rc = mgr.begin_with_isolation(IsolationLevel::ReadCommitted);
        let tx_si = mgr.begin_with_isolation(IsolationLevel::SnapshotIsolation);
        let tx_ser = mgr.begin_with_isolation(IsolationLevel::Serializable);

        assert_eq!(
            mgr.isolation_level(tx_rc),
            Some(IsolationLevel::ReadCommitted)
        );
        assert_eq!(
            mgr.isolation_level(tx_si),
            Some(IsolationLevel::SnapshotIsolation)
        );
        assert_eq!(
            mgr.isolation_level(tx_ser),
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

        // Both transactions write the same entity
        let tx1 = mgr.begin_with_isolation(IsolationLevel::Serializable);
        let tx2 = mgr.begin_with_isolation(IsolationLevel::Serializable);

        mgr.record_write(tx1, entity).unwrap();
        mgr.record_write(tx2, entity).unwrap();

        // First commit succeeds
        assert!(mgr.commit(tx1).is_ok());

        // Second commit fails (write-write conflict)
        let result = mgr.commit(tx2);
        assert!(result.is_err());
    }
}
