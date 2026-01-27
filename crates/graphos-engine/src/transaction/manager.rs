//! Transaction manager.

use graphos_common::types::{EpochId, TxId};
use graphos_common::utils::error::{Error, Result, TransactionError};
use graphos_common::utils::hash::FxHashMap;
use parking_lot::RwLock;
use std::sync::atomic::{AtomicU64, Ordering};

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

/// Information about an active transaction.
struct TxInfo {
    /// Transaction state.
    state: TxState,
    /// Start epoch.
    #[allow(dead_code)]
    start_epoch: EpochId,
}

/// Manages transactions and MVCC versioning.
pub struct TransactionManager {
    /// Next transaction ID.
    next_tx_id: AtomicU64,
    /// Current epoch.
    current_epoch: AtomicU64,
    /// Active transactions.
    transactions: RwLock<FxHashMap<TxId, TxInfo>>,
}

impl TransactionManager {
    /// Creates a new transaction manager.
    #[must_use]
    pub fn new() -> Self {
        Self {
            next_tx_id: AtomicU64::new(1),
            current_epoch: AtomicU64::new(0),
            transactions: RwLock::new(FxHashMap::default()),
        }
    }

    /// Begins a new transaction.
    pub fn begin(&self) -> TxId {
        let tx_id = TxId::new(self.next_tx_id.fetch_add(1, Ordering::Relaxed));
        let epoch = EpochId::new(self.current_epoch.load(Ordering::Acquire));

        let info = TxInfo {
            state: TxState::Active,
            start_epoch: epoch,
        };

        self.transactions.write().insert(tx_id, info);
        tx_id
    }

    /// Commits a transaction.
    ///
    /// # Errors
    ///
    /// Returns an error if the transaction is not active.
    pub fn commit(&self, tx_id: TxId) -> Result<()> {
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

        info.state = TxState::Committed;

        // Advance epoch
        self.current_epoch.fetch_add(1, Ordering::AcqRel);

        Ok(())
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

    /// Returns the state of a transaction.
    pub fn state(&self, tx_id: TxId) -> Option<TxState> {
        self.transactions.read().get(&tx_id).map(|info| info.state)
    }

    /// Returns the current epoch.
    #[must_use]
    pub fn current_epoch(&self) -> EpochId {
        EpochId::new(self.current_epoch.load(Ordering::Acquire))
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

    /// Cleans up completed transactions.
    pub fn gc(&self) {
        let mut txns = self.transactions.write();
        txns.retain(|_, info| info.state == TxState::Active);
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

        mgr.commit(tx).unwrap();
        assert_eq!(mgr.state(tx), Some(TxState::Committed));
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
        mgr.commit(tx).unwrap();

        assert!(mgr.current_epoch().as_u64() > initial_epoch.as_u64());
    }

    #[test]
    fn test_gc() {
        let mgr = TransactionManager::new();

        let tx1 = mgr.begin();
        let tx2 = mgr.begin();

        mgr.commit(tx1).unwrap();
        // tx2 still active

        assert_eq!(mgr.active_count(), 1);

        mgr.gc();

        // Committed transaction should be cleaned up
        assert_eq!(mgr.state(tx1), None);
        // Active transaction should remain
        assert_eq!(mgr.state(tx2), Some(TxState::Active));
    }
}
