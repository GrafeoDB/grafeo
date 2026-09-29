//! Two-phase commit with inspection and metadata.
//!
//! `PreparedCommit` lets you inspect pending changes before finalizing a
//! transaction. This is useful for external integrations that need to
//! validate, audit, or attach metadata before committing.
//!
//! # Example
//!
//! ```no_run
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! use grafeo_engine::GrafeoDB;
//!
//! let db = GrafeoDB::new_in_memory();
//! let mut session = db.session();
//!
//! session.begin_transaction()?;
//! session.execute("INSERT (:Person {name: 'Alix'})")?;
//!
//! let mut prepared = session.prepare_commit()?;
//! let info = prepared.info();
//! assert_eq!(info.nodes_written, 1);
//!
//! prepared.set_metadata("audit_user", "admin");
//! prepared.commit()?;
//! # Ok(())
//! # }
//! ```

use std::collections::HashMap;

use grafeo_common::types::{EpochId, TransactionId};
use grafeo_common::utils::error::{Error, Result, TransactionError};

use crate::Session;
use crate::transaction::EntityId;

/// Summary of pending transaction mutations.
#[derive(Debug, Clone)]
pub struct CommitInfo {
    /// Transaction ID.
    pub txn_id: TransactionId,
    /// Snapshot epoch the transaction read from.
    pub start_epoch: EpochId,
    /// Number of node entities in the write set.
    pub nodes_written: u64,
    /// Number of edge entities in the write set.
    pub edges_written: u64,
}

/// A transaction that has been validated and is ready to commit.
///
/// Created by [`Session::prepare_commit`]. The mutable borrow on the session
/// prevents any concurrent operations while the commit is pending.
///
/// If dropped without calling [`commit`](Self::commit) or [`abort`](Self::abort),
/// the transaction is automatically rolled back.
pub struct PreparedCommit<'a> {
    session: &'a mut Session,
    metadata: HashMap<String, String>,
    info: CommitInfo,
    finalized: bool,
}

impl<'a> PreparedCommit<'a> {
    /// Creates a new prepared commit from a session with an active transaction.
    pub(crate) fn new(session: &'a mut Session) -> Result<Self> {
        let transaction_id = session.current_transaction_id().ok_or_else(|| {
            Error::Transaction(TransactionError::InvalidState(
                "No active transaction to prepare".to_string(),
            ))
        })?;

        let start_epoch = session
            .transaction_manager()
            .start_epoch(transaction_id)
            .unwrap_or(EpochId::new(0));

        // The entities the transaction created, changed or deleted.
        let write_set = session
            .transaction_manager()
            .get_write_set(transaction_id)?;
        let (mut nodes_written, mut edges_written) = (0, 0);
        for entity in &write_set {
            match entity {
                EntityId::Node(_) => nodes_written += 1,
                EntityId::Edge(_) => edges_written += 1,
            }
        }

        let info = CommitInfo {
            txn_id: transaction_id,
            start_epoch,
            nodes_written,
            edges_written,
        };

        Ok(Self {
            session,
            metadata: HashMap::new(),
            info,
            finalized: false,
        })
    }

    /// Returns the commit info (mutation summary).
    #[must_use]
    pub fn info(&self) -> &CommitInfo {
        &self.info
    }

    /// Attaches metadata to this commit.
    ///
    /// Metadata is available for logging, auditing, or CDC consumers.
    pub fn set_metadata(&mut self, key: impl Into<String>, value: impl Into<String>) {
        self.metadata.insert(key.into(), value.into());
    }

    /// Returns the attached metadata.
    #[must_use]
    pub fn metadata(&self) -> &HashMap<String, String> {
        &self.metadata
    }

    /// Finalizes the commit, persisting all changes.
    ///
    /// Consumes self to prevent double-commit.
    ///
    /// # Errors
    ///
    /// Returns an error if the commit fails (write-write conflict, SSI violation, etc.).
    pub fn commit(mut self) -> Result<EpochId> {
        self.finalized = true;
        self.session.commit()?;
        Ok(self.session.transaction_manager().current_epoch())
    }

    /// Explicitly aborts the transaction, discarding all changes.
    ///
    /// # Errors
    ///
    /// Returns an error if the rollback fails.
    pub fn abort(mut self) -> Result<()> {
        self.finalized = true;
        self.session.rollback()
    }
}

impl Drop for PreparedCommit<'_> {
    fn drop(&mut self) {
        if !self.finalized {
            // Best-effort rollback on drop
            let _ = self.session.rollback();
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::GrafeoDB;

    #[test]
    fn test_prepared_commit_basic() {
        let db = GrafeoDB::new_in_memory();
        let mut session = db.session();

        session.begin_transaction().unwrap();
        session.execute("INSERT (:Person {name: 'Alix'})").unwrap();

        let prepared = session.prepare_commit().unwrap();
        let info = prepared.info();

        // The counts come from the transaction's write set.
        assert_eq!((info.nodes_written, info.edges_written), (1, 0));

        let epoch = prepared.commit().unwrap();
        assert!(epoch.as_u64() > 0);

        // After commit, finalize_version_epochs() converts PENDING to the
        // real commit epoch, making the data visible.
        assert_eq!(db.node_count(), 1, "Node should be visible after commit");
    }

    #[test]
    fn test_prepared_commit_with_edges() {
        let db = GrafeoDB::new_in_memory();
        let mut session = db.session();

        // Create nodes first, commit, then create edge in a second transaction.
        // This avoids the issue where MATCH within the same transaction cannot
        // find PENDING-epoch nodes for the cross-product join.
        session.begin_transaction().unwrap();
        session.execute("INSERT (:Person {name: 'Alix'})").unwrap();
        session.execute("INSERT (:Person {name: 'Gus'})").unwrap();
        session.commit().unwrap();

        assert_eq!(
            db.node_count(),
            2,
            "Both nodes should be visible after first commit"
        );

        // Second transaction: create edge between the now-committed nodes.
        session.begin_transaction().unwrap();
        session
            .execute(
                "MATCH (a:Person {name: 'Alix'}), (b:Person {name: 'Gus'}) INSERT (a)-[:KNOWS]->(b)",
            )
            .unwrap();

        let prepared = session.prepare_commit().unwrap();
        prepared.commit().unwrap();

        // Verify everything is visible after commit.
        assert_eq!(
            db.node_count(),
            2,
            "Both nodes should be visible after commit"
        );
        let session2 = db.session();
        let result = session2
            .execute("MATCH (a)-[:KNOWS]->(b) RETURN a.name, b.name")
            .unwrap();
        assert_eq!(result.row_count(), 1, "Edge should be visible after commit");
    }

    #[test]
    fn test_prepared_commit_metadata() {
        let db = GrafeoDB::new_in_memory();
        let mut session = db.session();

        session.begin_transaction().unwrap();
        session.execute("INSERT (:Person {name: 'Alix'})").unwrap();

        let mut prepared = session.prepare_commit().unwrap();
        prepared.set_metadata("audit_user", "admin");
        prepared.set_metadata("source", "api");

        assert_eq!(prepared.metadata().len(), 2);
        assert_eq!(prepared.metadata().get("audit_user").unwrap(), "admin");

        prepared.commit().unwrap();
    }

    #[test]
    fn test_prepared_commit_abort() {
        let db = GrafeoDB::new_in_memory();
        let mut session = db.session();

        session.begin_transaction().unwrap();
        session.execute("INSERT (:Person {name: 'Alix'})").unwrap();

        let prepared = session.prepare_commit().unwrap();
        prepared.abort().unwrap();

        // Data should not be visible after abort
        let result = session.execute("MATCH (n:Person) RETURN n").unwrap();
        assert_eq!(result.rows.len(), 0);
    }

    #[test]
    fn test_prepared_commit_drop_rollback() {
        let db = GrafeoDB::new_in_memory();
        let mut session = db.session();

        session.begin_transaction().unwrap();
        session.execute("INSERT (:Person {name: 'Alix'})").unwrap();

        {
            let _prepared = session.prepare_commit().unwrap();
            // Drop without commit or abort
        }

        // Data should not be visible after drop-induced rollback
        let result = session.execute("MATCH (n:Person) RETURN n").unwrap();
        assert_eq!(result.rows.len(), 0);
    }

    #[test]
    fn test_prepared_commit_no_transaction() {
        let db = GrafeoDB::new_in_memory();
        let mut session = db.session();

        // Should fail when no transaction is active
        let result = session.prepare_commit();
        assert!(result.is_err());
    }
}
