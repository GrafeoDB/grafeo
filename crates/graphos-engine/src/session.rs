//! Session management.

use std::sync::Arc;

use graphos_common::types::{NodeId, TxId, Value};
use graphos_common::utils::error::Result;
use graphos_core::graph::lpg::LpgStore;

use crate::database::QueryResult;
use crate::transaction::TransactionManager;

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

    /// Executes a GQL query.
    ///
    /// # Errors
    ///
    /// Returns an error if the query fails to parse or execute.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use graphos_engine::GraphosDB;
    ///
    /// let db = GraphosDB::new_in_memory();
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
        use crate::query::{binder::Binder, gql_translator, optimizer::Optimizer, Executor, Planner};

        // Parse and translate the query to a logical plan
        let logical_plan = gql_translator::translate(query)?;

        // Semantic validation
        let mut binder = Binder::new();
        let _binding_context = binder.bind(&logical_plan)?;

        // Optimize the plan
        let optimizer = Optimizer::new();
        let optimized_plan = optimizer.optimize(logical_plan)?;

        // Convert to physical plan
        let planner = Planner::new(Arc::clone(&self.store));
        let mut physical_plan = planner.plan(&optimized_plan)?;

        // Execute the plan
        let executor = Executor::with_columns(physical_plan.columns.clone());
        executor.execute(physical_plan.operator.as_mut())
    }

    /// Executes a GQL query.
    ///
    /// # Errors
    ///
    /// Returns an error if no query language is enabled.
    #[cfg(not(any(feature = "gql", feature = "cypher")))]
    pub fn execute(&self, _query: &str) -> Result<QueryResult> {
        Err(graphos_common::utils::error::Error::Internal(
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
        use crate::query::{binder::Binder, cypher_translator, optimizer::Optimizer, Executor, Planner};

        // Parse and translate the query to a logical plan
        let logical_plan = cypher_translator::translate(query)?;

        // Semantic validation
        let mut binder = Binder::new();
        let _binding_context = binder.bind(&logical_plan)?;

        // Optimize the plan
        let optimizer = Optimizer::new();
        let optimized_plan = optimizer.optimize(logical_plan)?;

        // Convert to physical plan
        let planner = Planner::new(Arc::clone(&self.store));
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
    /// use graphos_engine::GraphosDB;
    ///
    /// let db = GraphosDB::new_in_memory();
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
        use crate::query::{binder::Binder, gremlin_translator, optimizer::Optimizer, Executor, Planner};

        // Parse and translate the query to a logical plan
        let logical_plan = gremlin_translator::translate(query)?;

        // Semantic validation
        let mut binder = Binder::new();
        let _binding_context = binder.bind(&logical_plan)?;

        // Optimize the plan
        let optimizer = Optimizer::new();
        let optimized_plan = optimizer.optimize(logical_plan)?;

        // Convert to physical plan
        let planner = Planner::new(Arc::clone(&self.store));
        let mut physical_plan = planner.plan(&optimized_plan)?;

        // Execute the plan
        let executor = Executor::with_columns(physical_plan.columns.clone());
        executor.execute(physical_plan.operator.as_mut())
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
    /// use graphos_engine::GraphosDB;
    ///
    /// let db = GraphosDB::new_in_memory();
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
        use crate::query::{binder::Binder, graphql_translator, optimizer::Optimizer, Executor, Planner};

        // Parse and translate the query to a logical plan
        let logical_plan = graphql_translator::translate(query)?;

        // Semantic validation
        let mut binder = Binder::new();
        let _binding_context = binder.bind(&logical_plan)?;

        // Optimize the plan
        let optimizer = Optimizer::new();
        let optimized_plan = optimizer.optimize(logical_plan)?;

        // Convert to physical plan
        let planner = Planner::new(Arc::clone(&self.store));
        let mut physical_plan = planner.plan(&optimized_plan)?;

        // Execute the plan
        let executor = Executor::with_columns(physical_plan.columns.clone());
        executor.execute(physical_plan.operator.as_mut())
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
    /// use graphos_engine::GraphosDB;
    ///
    /// let db = GraphosDB::new_in_memory();
    /// let mut session = db.session();
    ///
    /// session.begin_tx()?;
    /// session.execute("INSERT (:Person {name: 'Alice'})")?;
    /// session.execute("INSERT (:Person {name: 'Bob'})")?;
    /// session.commit()?; // Both inserts committed atomically
    /// ```
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
    /// Makes all changes since [`begin_tx`](Self::begin_tx) permanent.
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
    /// Discards all changes since [`begin_tx`](Self::begin_tx).
    ///
    /// # Errors
    ///
    /// Returns an error if no transaction is active.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use graphos_engine::GraphosDB;
    ///
    /// let db = GraphosDB::new_in_memory();
    /// let mut session = db.session();
    ///
    /// session.begin_tx()?;
    /// session.execute("INSERT (:Person {name: 'Alice'})")?;
    /// session.rollback()?; // Insert is discarded
    /// ```
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
    pub fn create_node_with_props<'a>(
        &self,
        labels: &[&str],
        properties: impl IntoIterator<Item = (&'a str, Value)>,
    ) -> NodeId {
        self.store
            .create_node_with_props(labels, properties.into_iter().map(|(k, v)| (k, v)))
    }

    /// Creates an edge between two nodes.
    ///
    /// This is a low-level API for testing and direct manipulation.
    pub fn create_edge(
        &self,
        src: NodeId,
        dst: NodeId,
        edge_type: &str,
    ) -> graphos_common::types::EdgeId {
        self.store.create_edge(src, dst, edge_type)
    }
}

#[cfg(test)]
mod tests {
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

    #[cfg(feature = "gql")]
    mod gql_tests {
        use super::*;

        #[test]
        fn test_gql_query_execution() {
            let db = GraphosDB::new_in_memory();
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
            let db = GraphosDB::new_in_memory();
            let session = db.session();

            // No data in database
            let result = session.execute("MATCH (n:Person) RETURN n").unwrap();

            assert_eq!(result.row_count(), 0);
        }

        #[test]
        fn test_gql_parse_error() {
            let db = GraphosDB::new_in_memory();
            let session = db.session();

            // Invalid GQL syntax
            let result = session.execute("MATCH (n RETURN n");

            assert!(result.is_err());
        }

        #[test]
        fn test_gql_relationship_traversal() {
            let db = GraphosDB::new_in_memory();
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
            let db = GraphosDB::new_in_memory();
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
            let db = GraphosDB::new_in_memory();
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
            use graphos_common::types::Value;

            let db = GraphosDB::new_in_memory();
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
            use graphos_common::types::Value;

            let db = GraphosDB::new_in_memory();
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
            use graphos_common::types::Value;

            let db = GraphosDB::new_in_memory();
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
            use graphos_common::types::Value;

            let db = GraphosDB::new_in_memory();
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
            let db = GraphosDB::new_in_memory();
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
            let db = GraphosDB::new_in_memory();
            let session = db.session();

            // No data in database
            let result = session.execute_cypher("MATCH (n:Person) RETURN n").unwrap();

            assert_eq!(result.row_count(), 0);
        }

        #[test]
        fn test_cypher_parse_error() {
            let db = GraphosDB::new_in_memory();
            let session = db.session();

            // Invalid Cypher syntax
            let result = session.execute_cypher("MATCH (n RETURN n");

            assert!(result.is_err());
        }
    }
}
