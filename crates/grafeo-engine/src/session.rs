//! Lightweight handles for database interaction.
//!
//! A session is your conversation with the database. Each session can have
//! its own transaction state, so concurrent sessions don't interfere with
//! each other. Sessions are cheap to create - spin up as many as you need.

use std::sync::Arc;

use grafeo_common::types::{EpochId, NodeId, TxId, Value};
use grafeo_common::utils::error::Result;
use grafeo_core::graph::lpg::LpgStore;
#[cfg(feature = "rdf")]
use grafeo_core::graph::rdf::RdfStore;

use crate::config::AdaptiveConfig;
use crate::database::QueryResult;
use crate::transaction::TransactionManager;

/// Your handle to the database - execute queries and manage transactions.
///
/// Get one from [`GrafeoDB::session()`](crate::GrafeoDB::session). Each session
/// tracks its own transaction state, so you can have multiple concurrent
/// sessions without them interfering.
pub struct Session {
    /// The underlying store.
    store: Arc<LpgStore>,
    /// RDF triple store (if RDF feature is enabled).
    #[cfg(feature = "rdf")]
    #[allow(dead_code)]
    rdf_store: Arc<RdfStore>,
    /// Transaction manager.
    tx_manager: Arc<TransactionManager>,
    /// Current transaction ID (if any).
    current_tx: Option<TxId>,
    /// Whether the session is in auto-commit mode.
    auto_commit: bool,
    /// Adaptive execution configuration.
    #[allow(dead_code)]
    adaptive_config: AdaptiveConfig,
    /// Whether to use factorized execution for multi-hop queries.
    factorized_execution: bool,
}

impl Session {
    /// Creates a new session.
    #[allow(dead_code)]
    pub(crate) fn new(store: Arc<LpgStore>, tx_manager: Arc<TransactionManager>) -> Self {
        Self {
            store,
            #[cfg(feature = "rdf")]
            rdf_store: Arc::new(RdfStore::new()),
            tx_manager,
            current_tx: None,
            auto_commit: true,
            adaptive_config: AdaptiveConfig::default(),
            factorized_execution: true,
        }
    }

    /// Creates a new session with adaptive execution configuration.
    #[allow(dead_code)]
    pub(crate) fn with_adaptive(
        store: Arc<LpgStore>,
        tx_manager: Arc<TransactionManager>,
        adaptive_config: AdaptiveConfig,
        factorized_execution: bool,
    ) -> Self {
        Self {
            store,
            #[cfg(feature = "rdf")]
            rdf_store: Arc::new(RdfStore::new()),
            tx_manager,
            current_tx: None,
            auto_commit: true,
            adaptive_config,
            factorized_execution,
        }
    }

    /// Creates a new session with RDF store and adaptive configuration.
    #[cfg(feature = "rdf")]
    pub(crate) fn with_rdf_store_and_adaptive(
        store: Arc<LpgStore>,
        rdf_store: Arc<RdfStore>,
        tx_manager: Arc<TransactionManager>,
        adaptive_config: AdaptiveConfig,
        factorized_execution: bool,
    ) -> Self {
        Self {
            store,
            rdf_store,
            tx_manager,
            current_tx: None,
            auto_commit: true,
            adaptive_config,
            factorized_execution,
        }
    }

    /// Executes a GQL query.
    ///
    /// # Errors
    ///
    /// Returns an error if the query fails to parse or execute.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use grafeo_engine::GrafeoDB;
    ///
    /// let db = GrafeoDB::new_in_memory();
    /// let session = db.session();
    ///
    /// // Create a node
    /// session.execute("INSERT (:Person {name: 'Alice', age: 30})")?;
    ///
    /// // Query nodes
    /// let result = session.execute("MATCH (n:Person) RETURN n.name, n.age")?;
    /// for row in result {
    ///     println!("{:?}", row);
    /// }
    /// ```
    #[cfg(feature = "gql")]
    pub fn execute(&self, query: &str) -> Result<QueryResult> {
        use crate::query::{
            Executor, Planner, binder::Binder, gql_translator, optimizer::Optimizer,
        };

        // Parse and translate the query to a logical plan
        let logical_plan = gql_translator::translate(query)?;

        // Semantic validation
        let mut binder = Binder::new();
        let _binding_context = binder.bind(&logical_plan)?;

        // Optimize the plan
        let optimizer = Optimizer::new();
        let optimized_plan = optimizer.optimize(logical_plan)?;

        // Get transaction context for MVCC visibility
        let (viewing_epoch, tx_id) = self.get_transaction_context();

        // Convert to physical plan with transaction context
        let planner = Planner::with_context(
            Arc::clone(&self.store),
            Arc::clone(&self.tx_manager),
            tx_id,
            viewing_epoch,
        )
        .with_factorized_execution(self.factorized_execution);
        let mut physical_plan = planner.plan(&optimized_plan)?;

        // Execute the plan
        let executor = Executor::with_columns(physical_plan.columns.clone());
        executor.execute(physical_plan.operator.as_mut())
    }

    /// Executes a GQL query with parameters.
    ///
    /// # Errors
    ///
    /// Returns an error if the query fails to parse or execute.
    #[cfg(feature = "gql")]
    pub fn execute_with_params(
        &self,
        query: &str,
        params: std::collections::HashMap<String, Value>,
    ) -> Result<QueryResult> {
        use crate::query::processor::{QueryLanguage, QueryProcessor};

        // Get transaction context for MVCC visibility
        let (viewing_epoch, tx_id) = self.get_transaction_context();

        // Create processor with transaction context
        let processor =
            QueryProcessor::for_lpg_with_tx(Arc::clone(&self.store), Arc::clone(&self.tx_manager));

        // Apply transaction context if in a transaction
        let processor = if let Some(tx_id) = tx_id {
            processor.with_tx_context(viewing_epoch, tx_id)
        } else {
            processor
        };

        processor.process(query, QueryLanguage::Gql, Some(&params))
    }

    /// Executes a GQL query with parameters.
    ///
    /// # Errors
    ///
    /// Returns an error if no query language is enabled.
    #[cfg(not(any(feature = "gql", feature = "cypher")))]
    pub fn execute_with_params(
        &self,
        _query: &str,
        _params: std::collections::HashMap<String, Value>,
    ) -> Result<QueryResult> {
        Err(grafeo_common::utils::error::Error::Internal(
            "No query language enabled".to_string(),
        ))
    }

    /// Executes a GQL query.
    ///
    /// # Errors
    ///
    /// Returns an error if no query language is enabled.
    #[cfg(not(any(feature = "gql", feature = "cypher")))]
    pub fn execute(&self, _query: &str) -> Result<QueryResult> {
        Err(grafeo_common::utils::error::Error::Internal(
            "No query language enabled".to_string(),
        ))
    }

    /// Executes a Cypher query.
    ///
    /// # Errors
    ///
    /// Returns an error if the query fails to parse or execute.
    #[cfg(feature = "cypher")]
    pub fn execute_cypher(&self, query: &str) -> Result<QueryResult> {
        use crate::query::{
            Executor, Planner, binder::Binder, cypher_translator, optimizer::Optimizer,
        };

        // Parse and translate the query to a logical plan
        let logical_plan = cypher_translator::translate(query)?;

        // Semantic validation
        let mut binder = Binder::new();
        let _binding_context = binder.bind(&logical_plan)?;

        // Optimize the plan
        let optimizer = Optimizer::new();
        let optimized_plan = optimizer.optimize(logical_plan)?;

        // Get transaction context for MVCC visibility
        let (viewing_epoch, tx_id) = self.get_transaction_context();

        // Convert to physical plan with transaction context
        let planner = Planner::with_context(
            Arc::clone(&self.store),
            Arc::clone(&self.tx_manager),
            tx_id,
            viewing_epoch,
        )
        .with_factorized_execution(self.factorized_execution);
        let mut physical_plan = planner.plan(&optimized_plan)?;

        // Execute the plan
        let executor = Executor::with_columns(physical_plan.columns.clone());
        executor.execute(physical_plan.operator.as_mut())
    }

    /// Executes a Gremlin query.
    ///
    /// # Errors
    ///
    /// Returns an error if the query fails to parse or execute.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use grafeo_engine::GrafeoDB;
    ///
    /// let db = GrafeoDB::new_in_memory();
    /// let session = db.session();
    ///
    /// // Create some nodes first
    /// session.create_node(&["Person"]);
    ///
    /// // Query using Gremlin
    /// let result = session.execute_gremlin("g.V().hasLabel('Person')")?;
    /// ```
    #[cfg(feature = "gremlin")]
    pub fn execute_gremlin(&self, query: &str) -> Result<QueryResult> {
        use crate::query::{
            Executor, Planner, binder::Binder, gremlin_translator, optimizer::Optimizer,
        };

        // Parse and translate the query to a logical plan
        let logical_plan = gremlin_translator::translate(query)?;

        // Semantic validation
        let mut binder = Binder::new();
        let _binding_context = binder.bind(&logical_plan)?;

        // Optimize the plan
        let optimizer = Optimizer::new();
        let optimized_plan = optimizer.optimize(logical_plan)?;

        // Get transaction context for MVCC visibility
        let (viewing_epoch, tx_id) = self.get_transaction_context();

        // Convert to physical plan with transaction context
        let planner = Planner::with_context(
            Arc::clone(&self.store),
            Arc::clone(&self.tx_manager),
            tx_id,
            viewing_epoch,
        )
        .with_factorized_execution(self.factorized_execution);
        let mut physical_plan = planner.plan(&optimized_plan)?;

        // Execute the plan
        let executor = Executor::with_columns(physical_plan.columns.clone());
        executor.execute(physical_plan.operator.as_mut())
    }

    /// Executes a Gremlin query with parameters.
    ///
    /// # Errors
    ///
    /// Returns an error if the query fails to parse or execute.
    #[cfg(feature = "gremlin")]
    pub fn execute_gremlin_with_params(
        &self,
        query: &str,
        params: std::collections::HashMap<String, Value>,
    ) -> Result<QueryResult> {
        use crate::query::processor::{QueryLanguage, QueryProcessor};

        // Get transaction context for MVCC visibility
        let (viewing_epoch, tx_id) = self.get_transaction_context();

        // Create processor with transaction context
        let processor =
            QueryProcessor::for_lpg_with_tx(Arc::clone(&self.store), Arc::clone(&self.tx_manager));

        // Apply transaction context if in a transaction
        let processor = if let Some(tx_id) = tx_id {
            processor.with_tx_context(viewing_epoch, tx_id)
        } else {
            processor
        };

        processor.process(query, QueryLanguage::Gremlin, Some(&params))
    }

    /// Executes a GraphQL query against the LPG store.
    ///
    /// # Errors
    ///
    /// Returns an error if the query fails to parse or execute.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use grafeo_engine::GrafeoDB;
    ///
    /// let db = GrafeoDB::new_in_memory();
    /// let session = db.session();
    ///
    /// // Create some nodes first
    /// session.create_node(&["User"]);
    ///
    /// // Query using GraphQL
    /// let result = session.execute_graphql("query { user { id name } }")?;
    /// ```
    #[cfg(feature = "graphql")]
    pub fn execute_graphql(&self, query: &str) -> Result<QueryResult> {
        use crate::query::{
            Executor, Planner, binder::Binder, graphql_translator, optimizer::Optimizer,
        };

        // Parse and translate the query to a logical plan
        let logical_plan = graphql_translator::translate(query)?;

        // Semantic validation
        let mut binder = Binder::new();
        let _binding_context = binder.bind(&logical_plan)?;

        // Optimize the plan
        let optimizer = Optimizer::new();
        let optimized_plan = optimizer.optimize(logical_plan)?;

        // Get transaction context for MVCC visibility
        let (viewing_epoch, tx_id) = self.get_transaction_context();

        // Convert to physical plan with transaction context
        let planner = Planner::with_context(
            Arc::clone(&self.store),
            Arc::clone(&self.tx_manager),
            tx_id,
            viewing_epoch,
        )
        .with_factorized_execution(self.factorized_execution);
        let mut physical_plan = planner.plan(&optimized_plan)?;

        // Execute the plan
        let executor = Executor::with_columns(physical_plan.columns.clone());
        executor.execute(physical_plan.operator.as_mut())
    }

    /// Executes a GraphQL query with parameters.
    ///
    /// # Errors
    ///
    /// Returns an error if the query fails to parse or execute.
    #[cfg(feature = "graphql")]
    pub fn execute_graphql_with_params(
        &self,
        query: &str,
        params: std::collections::HashMap<String, Value>,
    ) -> Result<QueryResult> {
        use crate::query::processor::{QueryLanguage, QueryProcessor};

        // Get transaction context for MVCC visibility
        let (viewing_epoch, tx_id) = self.get_transaction_context();

        // Create processor with transaction context
        let processor =
            QueryProcessor::for_lpg_with_tx(Arc::clone(&self.store), Arc::clone(&self.tx_manager));

        // Apply transaction context if in a transaction
        let processor = if let Some(tx_id) = tx_id {
            processor.with_tx_context(viewing_epoch, tx_id)
        } else {
            processor
        };

        processor.process(query, QueryLanguage::GraphQL, Some(&params))
    }

    /// Executes a SPARQL query.
    ///
    /// # Errors
    ///
    /// Returns an error if the query fails to parse or execute.
    #[cfg(all(feature = "sparql", feature = "rdf"))]
    pub fn execute_sparql(&self, query: &str) -> Result<QueryResult> {
        use crate::query::{
            Executor, optimizer::Optimizer, planner_rdf::RdfPlanner, sparql_translator,
        };

        // Parse and translate the SPARQL query to a logical plan
        let logical_plan = sparql_translator::translate(query)?;

        // Optimize the plan
        let optimizer = Optimizer::new();
        let optimized_plan = optimizer.optimize(logical_plan)?;

        // Convert to physical plan using RDF planner
        let planner = RdfPlanner::new(Arc::clone(&self.rdf_store)).with_tx_id(self.current_tx);
        let mut physical_plan = planner.plan(&optimized_plan)?;

        // Execute the plan
        let executor = Executor::with_columns(physical_plan.columns.clone());
        executor.execute(physical_plan.operator.as_mut())
    }

    /// Executes a SPARQL query with parameters.
    ///
    /// # Errors
    ///
    /// Returns an error if the query fails to parse or execute.
    #[cfg(all(feature = "sparql", feature = "rdf"))]
    pub fn execute_sparql_with_params(
        &self,
        query: &str,
        _params: std::collections::HashMap<String, Value>,
    ) -> Result<QueryResult> {
        // TODO: Implement parameter substitution for SPARQL
        // For now, just execute the query without parameters
        self.execute_sparql(query)
    }

    /// Begins a new transaction.
    ///
    /// # Errors
    ///
    /// Returns an error if a transaction is already active.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use grafeo_engine::GrafeoDB;
    ///
    /// let db = GrafeoDB::new_in_memory();
    /// let mut session = db.session();
    ///
    /// session.begin_tx()?;
    /// session.execute("INSERT (:Person {name: 'Alice'})")?;
    /// session.execute("INSERT (:Person {name: 'Bob'})")?;
    /// session.commit()?; // Both inserts committed atomically
    /// ```
    pub fn begin_tx(&mut self) -> Result<()> {
        if self.current_tx.is_some() {
            return Err(grafeo_common::utils::error::Error::Transaction(
                grafeo_common::utils::error::TransactionError::InvalidState(
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
    /// Makes all changes since [`begin_tx`](Self::begin_tx) permanent.
    ///
    /// # Errors
    ///
    /// Returns an error if no transaction is active.
    pub fn commit(&mut self) -> Result<()> {
        let tx_id = self.current_tx.take().ok_or_else(|| {
            grafeo_common::utils::error::Error::Transaction(
                grafeo_common::utils::error::TransactionError::InvalidState(
                    "No active transaction".to_string(),
                ),
            )
        })?;

        // Commit RDF store pending operations
        #[cfg(feature = "rdf")]
        self.rdf_store.commit_tx(tx_id);

        self.tx_manager.commit(tx_id).map(|_| ())
    }

    /// Aborts the current transaction.
    ///
    /// Discards all changes since [`begin_tx`](Self::begin_tx).
    ///
    /// # Errors
    ///
    /// Returns an error if no transaction is active.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use grafeo_engine::GrafeoDB;
    ///
    /// let db = GrafeoDB::new_in_memory();
    /// let mut session = db.session();
    ///
    /// session.begin_tx()?;
    /// session.execute("INSERT (:Person {name: 'Alice'})")?;
    /// session.rollback()?; // Insert is discarded
    /// ```
    pub fn rollback(&mut self) -> Result<()> {
        let tx_id = self.current_tx.take().ok_or_else(|| {
            grafeo_common::utils::error::Error::Transaction(
                grafeo_common::utils::error::TransactionError::InvalidState(
                    "No active transaction".to_string(),
                ),
            )
        })?;

        // Discard uncommitted versions in the LPG store
        self.store.discard_uncommitted_versions(tx_id);

        // Discard pending operations in the RDF store
        #[cfg(feature = "rdf")]
        self.rdf_store.rollback_tx(tx_id);

        // Mark transaction as aborted in the manager
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

    /// Returns the current transaction context for MVCC visibility.
    ///
    /// Returns `(viewing_epoch, tx_id)` where:
    /// - `viewing_epoch` is the epoch at which to check version visibility
    /// - `tx_id` is the current transaction ID (if in a transaction)
    #[must_use]
    fn get_transaction_context(&self) -> (EpochId, Option<TxId>) {
        if let Some(tx_id) = self.current_tx {
            // In a transaction - use the transaction's start epoch
            let epoch = self
                .tx_manager
                .start_epoch(tx_id)
                .unwrap_or_else(|| self.tx_manager.current_epoch());
            (epoch, Some(tx_id))
        } else {
            // No transaction - use current epoch
            (self.tx_manager.current_epoch(), None)
        }
    }

    /// Creates a node directly (bypassing query execution).
    ///
    /// This is a low-level API for testing and direct manipulation.
    /// If a transaction is active, the node will be versioned with the transaction ID.
    pub fn create_node(&self, labels: &[&str]) -> NodeId {
        let (epoch, tx_id) = self.get_transaction_context();
        self.store
            .create_node_versioned(labels, epoch, tx_id.unwrap_or(TxId::SYSTEM))
    }

    /// Creates a node with properties.
    ///
    /// If a transaction is active, the node will be versioned with the transaction ID.
    pub fn create_node_with_props<'a>(
        &self,
        labels: &[&str],
        properties: impl IntoIterator<Item = (&'a str, Value)>,
    ) -> NodeId {
        let (epoch, tx_id) = self.get_transaction_context();
        self.store.create_node_with_props_versioned(
            labels,
            properties.into_iter().map(|(k, v)| (k, v)),
            epoch,
            tx_id.unwrap_or(TxId::SYSTEM),
        )
    }

    /// Creates an edge between two nodes.
    ///
    /// This is a low-level API for testing and direct manipulation.
    /// If a transaction is active, the edge will be versioned with the transaction ID.
    pub fn create_edge(
        &self,
        src: NodeId,
        dst: NodeId,
        edge_type: &str,
    ) -> grafeo_common::types::EdgeId {
        let (epoch, tx_id) = self.get_transaction_context();
        self.store
            .create_edge_versioned(src, dst, edge_type, epoch, tx_id.unwrap_or(TxId::SYSTEM))
    }
}

#[cfg(test)]
mod tests {
    use crate::database::GrafeoDB;

    #[test]
    fn test_session_create_node() {
        let db = GrafeoDB::new_in_memory();
        let session = db.session();

        let id = session.create_node(&["Person"]);
        assert!(id.is_valid());
        assert_eq!(db.node_count(), 1);
    }

    #[test]
    fn test_session_transaction() {
        let db = GrafeoDB::new_in_memory();
        let mut session = db.session();

        assert!(!session.in_transaction());

        session.begin_tx().unwrap();
        assert!(session.in_transaction());

        session.commit().unwrap();
        assert!(!session.in_transaction());
    }

    #[test]
    fn test_session_transaction_context() {
        let db = GrafeoDB::new_in_memory();
        let mut session = db.session();

        // Without transaction - context should have current epoch and no tx_id
        let (_epoch1, tx_id1) = session.get_transaction_context();
        assert!(tx_id1.is_none());

        // Start a transaction
        session.begin_tx().unwrap();
        let (epoch2, tx_id2) = session.get_transaction_context();
        assert!(tx_id2.is_some());
        // Transaction should have a valid epoch
        let _ = epoch2; // Use the variable

        // Commit and verify
        session.commit().unwrap();
        let (epoch3, tx_id3) = session.get_transaction_context();
        assert!(tx_id3.is_none());
        // Epoch should have advanced after commit
        assert!(epoch3.as_u64() >= epoch2.as_u64());
    }

    #[test]
    fn test_session_rollback() {
        let db = GrafeoDB::new_in_memory();
        let mut session = db.session();

        session.begin_tx().unwrap();
        session.rollback().unwrap();
        assert!(!session.in_transaction());
    }

    #[test]
    fn test_session_rollback_discards_versions() {
        use grafeo_common::types::TxId;

        let db = GrafeoDB::new_in_memory();

        // Create a node outside of any transaction (at system level)
        let node_before = db.store().create_node(&["Person"]);
        assert!(node_before.is_valid());
        assert_eq!(db.node_count(), 1, "Should have 1 node before transaction");

        // Start a transaction
        let mut session = db.session();
        session.begin_tx().unwrap();
        let tx_id = session.current_tx.unwrap();

        // Create a node versioned with the transaction's ID
        let epoch = db.store().current_epoch();
        let node_in_tx = db.store().create_node_versioned(&["Person"], epoch, tx_id);
        assert!(node_in_tx.is_valid());

        // Should see 2 nodes at this point
        assert_eq!(db.node_count(), 2, "Should have 2 nodes during transaction");

        // Rollback the transaction
        session.rollback().unwrap();
        assert!(!session.in_transaction());

        // The node created in the transaction should be discarded
        // Only the first node should remain visible
        let count_after = db.node_count();
        assert_eq!(
            count_after, 1,
            "Rollback should discard uncommitted node, but got {count_after}"
        );

        // The original node should still be accessible
        let current_epoch = db.store().current_epoch();
        assert!(
            db.store()
                .get_node_versioned(node_before, current_epoch, TxId::SYSTEM)
                .is_some(),
            "Original node should still exist"
        );

        // The node created in the transaction should not be accessible
        assert!(
            db.store()
                .get_node_versioned(node_in_tx, current_epoch, TxId::SYSTEM)
                .is_none(),
            "Transaction node should be gone"
        );
    }

    #[test]
    fn test_session_create_node_in_transaction() {
        // Test that session.create_node() is transaction-aware
        let db = GrafeoDB::new_in_memory();

        // Create a node outside of any transaction
        let node_before = db.create_node(&["Person"]);
        assert!(node_before.is_valid());
        assert_eq!(db.node_count(), 1, "Should have 1 node before transaction");

        // Start a transaction and create a node through the session
        let mut session = db.session();
        session.begin_tx().unwrap();

        // Create a node through session.create_node() - should be versioned with tx
        let node_in_tx = session.create_node(&["Person"]);
        assert!(node_in_tx.is_valid());

        // Should see 2 nodes at this point
        assert_eq!(db.node_count(), 2, "Should have 2 nodes during transaction");

        // Rollback the transaction
        session.rollback().unwrap();

        // The node created via session.create_node() should be discarded
        let count_after = db.node_count();
        assert_eq!(
            count_after, 1,
            "Rollback should discard node created via session.create_node(), but got {count_after}"
        );
    }

    #[test]
    fn test_session_create_node_with_props_in_transaction() {
        use grafeo_common::types::Value;

        // Test that session.create_node_with_props() is transaction-aware
        let db = GrafeoDB::new_in_memory();

        // Create a node outside of any transaction
        db.create_node(&["Person"]);
        assert_eq!(db.node_count(), 1, "Should have 1 node before transaction");

        // Start a transaction and create a node with properties
        let mut session = db.session();
        session.begin_tx().unwrap();

        let node_in_tx =
            session.create_node_with_props(&["Person"], [("name", Value::String("Alice".into()))]);
        assert!(node_in_tx.is_valid());

        // Should see 2 nodes
        assert_eq!(db.node_count(), 2, "Should have 2 nodes during transaction");

        // Rollback the transaction
        session.rollback().unwrap();

        // The node should be discarded
        let count_after = db.node_count();
        assert_eq!(
            count_after, 1,
            "Rollback should discard node created via session.create_node_with_props()"
        );
    }

    #[cfg(feature = "gql")]
    mod gql_tests {
        use super::*;

        #[test]
        fn test_gql_query_execution() {
            let db = GrafeoDB::new_in_memory();
            let session = db.session();

            // Create some test data
            session.create_node(&["Person"]);
            session.create_node(&["Person"]);
            session.create_node(&["Animal"]);

            // Execute a GQL query
            let result = session.execute("MATCH (n:Person) RETURN n").unwrap();

            // Should return 2 Person nodes
            assert_eq!(result.row_count(), 2);
            assert_eq!(result.column_count(), 1);
            assert_eq!(result.columns[0], "n");
        }

        #[test]
        fn test_gql_empty_result() {
            let db = GrafeoDB::new_in_memory();
            let session = db.session();

            // No data in database
            let result = session.execute("MATCH (n:Person) RETURN n").unwrap();

            assert_eq!(result.row_count(), 0);
        }

        #[test]
        fn test_gql_parse_error() {
            let db = GrafeoDB::new_in_memory();
            let session = db.session();

            // Invalid GQL syntax
            let result = session.execute("MATCH (n RETURN n");

            assert!(result.is_err());
        }

        #[test]
        fn test_gql_relationship_traversal() {
            let db = GrafeoDB::new_in_memory();
            let session = db.session();

            // Create a graph: Alice -> Bob, Alice -> Charlie
            let alice = session.create_node(&["Person"]);
            let bob = session.create_node(&["Person"]);
            let charlie = session.create_node(&["Person"]);

            session.create_edge(alice, bob, "KNOWS");
            session.create_edge(alice, charlie, "KNOWS");

            // Execute a path query: MATCH (a:Person)-[:KNOWS]->(b:Person) RETURN a, b
            let result = session
                .execute("MATCH (a:Person)-[:KNOWS]->(b:Person) RETURN a, b")
                .unwrap();

            // Should return 2 rows (Alice->Bob, Alice->Charlie)
            assert_eq!(result.row_count(), 2);
            assert_eq!(result.column_count(), 2);
            assert_eq!(result.columns[0], "a");
            assert_eq!(result.columns[1], "b");
        }

        #[test]
        fn test_gql_relationship_with_type_filter() {
            let db = GrafeoDB::new_in_memory();
            let session = db.session();

            // Create a graph: Alice -KNOWS-> Bob, Alice -WORKS_WITH-> Charlie
            let alice = session.create_node(&["Person"]);
            let bob = session.create_node(&["Person"]);
            let charlie = session.create_node(&["Person"]);

            session.create_edge(alice, bob, "KNOWS");
            session.create_edge(alice, charlie, "WORKS_WITH");

            // Query only KNOWS relationships
            let result = session
                .execute("MATCH (a:Person)-[:KNOWS]->(b:Person) RETURN a, b")
                .unwrap();

            // Should return only 1 row (Alice->Bob)
            assert_eq!(result.row_count(), 1);
        }

        #[test]
        fn test_gql_semantic_error_undefined_variable() {
            let db = GrafeoDB::new_in_memory();
            let session = db.session();

            // Reference undefined variable 'x' in RETURN
            let result = session.execute("MATCH (n:Person) RETURN x");

            // Should fail with semantic error
            assert!(result.is_err());
            let err = match result {
                Err(e) => e,
                Ok(_) => panic!("Expected error"),
            };
            assert!(
                err.to_string().contains("Undefined variable"),
                "Expected undefined variable error, got: {}",
                err
            );
        }

        #[test]
        fn test_gql_where_clause_property_filter() {
            use grafeo_common::types::Value;

            let db = GrafeoDB::new_in_memory();
            let session = db.session();

            // Create people with ages
            session.create_node_with_props(&["Person"], [("age", Value::Int64(25))]);
            session.create_node_with_props(&["Person"], [("age", Value::Int64(35))]);
            session.create_node_with_props(&["Person"], [("age", Value::Int64(45))]);

            // Query with WHERE clause: age > 30
            let result = session
                .execute("MATCH (n:Person) WHERE n.age > 30 RETURN n")
                .unwrap();

            // Should return 2 people (ages 35 and 45)
            assert_eq!(result.row_count(), 2);
        }

        #[test]
        fn test_gql_where_clause_equality() {
            use grafeo_common::types::Value;

            let db = GrafeoDB::new_in_memory();
            let session = db.session();

            // Create people with names
            session.create_node_with_props(&["Person"], [("name", Value::String("Alice".into()))]);
            session.create_node_with_props(&["Person"], [("name", Value::String("Bob".into()))]);
            session.create_node_with_props(&["Person"], [("name", Value::String("Alice".into()))]);

            // Query with WHERE clause: name = "Alice"
            let result = session
                .execute("MATCH (n:Person) WHERE n.name = \"Alice\" RETURN n")
                .unwrap();

            // Should return 2 people named Alice
            assert_eq!(result.row_count(), 2);
        }

        #[test]
        fn test_gql_return_property_access() {
            use grafeo_common::types::Value;

            let db = GrafeoDB::new_in_memory();
            let session = db.session();

            // Create people with names and ages
            session.create_node_with_props(
                &["Person"],
                [
                    ("name", Value::String("Alice".into())),
                    ("age", Value::Int64(30)),
                ],
            );
            session.create_node_with_props(
                &["Person"],
                [
                    ("name", Value::String("Bob".into())),
                    ("age", Value::Int64(25)),
                ],
            );

            // Query returning properties
            let result = session
                .execute("MATCH (n:Person) RETURN n.name, n.age")
                .unwrap();

            // Should return 2 rows with name and age columns
            assert_eq!(result.row_count(), 2);
            assert_eq!(result.column_count(), 2);
            assert_eq!(result.columns[0], "n.name");
            assert_eq!(result.columns[1], "n.age");

            // Check that we get actual values
            let names: Vec<&Value> = result.rows.iter().map(|r| &r[0]).collect();
            assert!(names.contains(&&Value::String("Alice".into())));
            assert!(names.contains(&&Value::String("Bob".into())));
        }

        #[test]
        fn test_gql_return_mixed_expressions() {
            use grafeo_common::types::Value;

            let db = GrafeoDB::new_in_memory();
            let session = db.session();

            // Create a person
            session.create_node_with_props(&["Person"], [("name", Value::String("Alice".into()))]);

            // Query returning both node and property
            let result = session
                .execute("MATCH (n:Person) RETURN n, n.name")
                .unwrap();

            assert_eq!(result.row_count(), 1);
            assert_eq!(result.column_count(), 2);
            assert_eq!(result.columns[0], "n");
            assert_eq!(result.columns[1], "n.name");

            // Second column should be the name
            assert_eq!(result.rows[0][1], Value::String("Alice".into()));
        }
    }

    #[cfg(feature = "cypher")]
    mod cypher_tests {
        use super::*;

        #[test]
        fn test_cypher_query_execution() {
            let db = GrafeoDB::new_in_memory();
            let session = db.session();

            // Create some test data
            session.create_node(&["Person"]);
            session.create_node(&["Person"]);
            session.create_node(&["Animal"]);

            // Execute a Cypher query
            let result = session.execute_cypher("MATCH (n:Person) RETURN n").unwrap();

            // Should return 2 Person nodes
            assert_eq!(result.row_count(), 2);
            assert_eq!(result.column_count(), 1);
            assert_eq!(result.columns[0], "n");
        }

        #[test]
        fn test_cypher_empty_result() {
            let db = GrafeoDB::new_in_memory();
            let session = db.session();

            // No data in database
            let result = session.execute_cypher("MATCH (n:Person) RETURN n").unwrap();

            assert_eq!(result.row_count(), 0);
        }

        #[test]
        fn test_cypher_parse_error() {
            let db = GrafeoDB::new_in_memory();
            let session = db.session();

            // Invalid Cypher syntax
            let result = session.execute_cypher("MATCH (n RETURN n");

            assert!(result.is_err());
        }
    }
}
