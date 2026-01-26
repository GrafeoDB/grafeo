//! Session management.

use crate::database::QueryResult;
use crate::transaction::TransactionManager;
use graphos_common::types::{NodeId, TxId, Value};
use graphos_common::utils::error::Result;
use graphos_core::graph::lpg::LpgStore;
use std::sync::Arc;

/// A session for interacting with the database.
///
/// Sessions provide isolation between concurrent users and
/// manage transaction state.
pub struct Session {
    /// The underlying store.
    store: Arc<LpgStore>,
    /// Transaction manager.
    tx_manager: Arc<TransactionManager>,
    /// Current transaction ID (if any).
    current_tx: Option<TxId>,
    /// Whether the session is in auto-commit mode.
    auto_commit: bool,
}

impl Session {
    /// Creates a new session.
    pub(crate) fn new(store: Arc<LpgStore>, tx_manager: Arc<TransactionManager>) -> Self {
        Self {
            store,
            tx_manager,
            current_tx: None,
            auto_commit: true,
        }
    }

    /// Executes a query.
    ///
    /// # Errors
    ///
    /// Returns an error if the query fails.
    pub fn execute(&self, query: &str) -> Result<QueryResult> {
        #[cfg(feature = "gql")]
        {
            use graphos_adapters::query::gql;

            // Parse the query
            let _ast = gql::parse(query)?;

            // TODO: Bind, plan, optimize, execute
            // For now, return an empty result
            Ok(QueryResult::new(vec![]))
        }

        #[cfg(not(feature = "gql"))]
        {
            Err(graphos_common::utils::error::Error::Internal(
                "No query language enabled".to_string(),
            ))
        }
    }

    /// Begins a new transaction.
    ///
    /// # Errors
    ///
    /// Returns an error if a transaction is already active.
    pub fn begin_tx(&mut self) -> Result<()> {
        if self.current_tx.is_some() {
            return Err(graphos_common::utils::error::Error::Transaction(
                graphos_common::utils::error::TransactionError::InvalidState(
                    "Transaction already active".to_string(),
                ),
            ));
        }

        let tx_id = self.tx_manager.begin();
        self.current_tx = Some(tx_id);
        Ok(())
    }

    /// Commits the current transaction.
    ///
    /// # Errors
    ///
    /// Returns an error if no transaction is active.
    pub fn commit(&mut self) -> Result<()> {
        let tx_id = self.current_tx.take().ok_or_else(|| {
            graphos_common::utils::error::Error::Transaction(
                graphos_common::utils::error::TransactionError::InvalidState(
                    "No active transaction".to_string(),
                ),
            )
        })?;

        self.tx_manager.commit(tx_id)
    }

    /// Aborts the current transaction.
    ///
    /// # Errors
    ///
    /// Returns an error if no transaction is active.
    pub fn rollback(&mut self) -> Result<()> {
        let tx_id = self.current_tx.take().ok_or_else(|| {
            graphos_common::utils::error::Error::Transaction(
                graphos_common::utils::error::TransactionError::InvalidState(
                    "No active transaction".to_string(),
                ),
            )
        })?;

        self.tx_manager.abort(tx_id)
    }

    /// Returns whether a transaction is active.
    #[must_use]
    pub fn in_transaction(&self) -> bool {
        self.current_tx.is_some()
    }

    /// Sets auto-commit mode.
    pub fn set_auto_commit(&mut self, auto_commit: bool) {
        self.auto_commit = auto_commit;
    }

    /// Returns whether auto-commit is enabled.
    #[must_use]
    pub fn auto_commit(&self) -> bool {
        self.auto_commit
    }

    /// Creates a node directly (bypassing query execution).
    ///
    /// This is a low-level API for testing and direct manipulation.
    pub fn create_node(&self, labels: &[&str]) -> NodeId {
        self.store.create_node(labels)
    }

    /// Creates a node with properties.
    pub fn create_node_with_props(
        &self,
        labels: &[&str],
        properties: impl IntoIterator<Item = (&str, Value)>,
    ) -> NodeId {
        self.store.create_node_with_props(
            labels,
            properties.into_iter().map(|(k, v)| (k, v)),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::database::GraphosDB;

    #[test]
    fn test_session_create_node() {
        let db = GraphosDB::new_in_memory();
        let session = db.session();

        let id = session.create_node(&["Person"]);
        assert!(id.is_valid());
        assert_eq!(db.node_count(), 1);
    }

    #[test]
    fn test_session_transaction() {
        let db = GraphosDB::new_in_memory();
        let mut session = db.session();

        assert!(!session.in_transaction());

        session.begin_tx().unwrap();
        assert!(session.in_transaction());

        session.commit().unwrap();
        assert!(!session.in_transaction());
    }

    #[test]
    fn test_session_rollback() {
        let db = GraphosDB::new_in_memory();
        let mut session = db.session();

        session.begin_tx().unwrap();
        session.rollback().unwrap();
        assert!(!session.in_transaction());
    }
}
