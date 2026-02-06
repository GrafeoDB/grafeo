//! Lightweight handles for database interaction.
//!
//! A session is your conversation with the database. Each session can have
//! its own transaction state, so concurrent sessions don't interfere with
//! each other. Sessions are cheap to create - spin up as many as you need.

use std::sync::Arc;

use grafeo_common::types::{EdgeId, EpochId, NodeId, TxId, Value};
use grafeo_common::utils::error::Result;
use grafeo_core::graph::Direction;
use grafeo_core::graph::lpg::{Edge, LpgStore, Node};
#[cfg(feature = "rdf")]
use grafeo_core::graph::rdf::RdfStore;

use crate::config::AdaptiveConfig;
use crate::database::QueryResult;
use crate::query::cache::QueryCache;
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
    /// Query cache shared across sessions.
    query_cache: Arc<QueryCache>,
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
    pub(crate) fn new(
        store: Arc<LpgStore>,
        tx_manager: Arc<TransactionManager>,
        query_cache: Arc<QueryCache>,
    ) -> Self {
        Self {
            store,
            #[cfg(feature = "rdf")]
            rdf_store: Arc::new(RdfStore::new()),
            tx_manager,
            query_cache,
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
        query_cache: Arc<QueryCache>,
        adaptive_config: AdaptiveConfig,
        factorized_execution: bool,
    ) -> Self {
        Self {
            store,
            #[cfg(feature = "rdf")]
            rdf_store: Arc::new(RdfStore::new()),
            tx_manager,
            query_cache,
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
        query_cache: Arc<QueryCache>,
        adaptive_config: AdaptiveConfig,
        factorized_execution: bool,
    ) -> Self {
        Self {
            store,
            rdf_store,
            tx_manager,
            query_cache,
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
            Executor, Planner, binder::Binder, cache::CacheKey, gql_translator,
            optimizer::Optimizer, processor::QueryLanguage,
        };

        let start_time = std::time::Instant::now();

        // Create cache key for this query
        let cache_key = CacheKey::new(query, QueryLanguage::Gql);

        // Try to get cached optimized plan
        let optimized_plan = if let Some(cached_plan) = self.query_cache.get_optimized(&cache_key) {
            // Cache hit - skip parsing, translation, binding, and optimization
            cached_plan
        } else {
            // Cache miss - run full pipeline

            // Parse and translate the query to a logical plan
            let logical_plan = gql_translator::translate(query)?;

            // Semantic validation
            let mut binder = Binder::new();
            let _binding_context = binder.bind(&logical_plan)?;

            // Optimize the plan
            let optimizer = Optimizer::from_store(&self.store);
            let plan = optimizer.optimize(logical_plan)?;

            // Cache the optimized plan for future use
            self.query_cache.put_optimized(cache_key, plan.clone());

            plan
        };

        // Get transaction context for MVCC visibility
        let (viewing_epoch, tx_id) = self.get_transaction_context();

        // Convert to physical plan with transaction context
        // (Physical planning cannot be cached as it depends on transaction state)
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
        let mut result = executor.execute(physical_plan.operator.as_mut())?;

        // Add execution metrics
        let elapsed_ms = start_time.elapsed().as_secs_f64() * 1000.0;
        let rows_scanned = result.rows.len() as u64;
        result.execution_time_ms = Some(elapsed_ms);
        result.rows_scanned = Some(rows_scanned);

        Ok(result)
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
            Executor, Planner, binder::Binder, cache::CacheKey, cypher_translator,
            optimizer::Optimizer, processor::QueryLanguage,
        };

        // Create cache key for this query
        let cache_key = CacheKey::new(query, QueryLanguage::Cypher);

        // Try to get cached optimized plan
        let optimized_plan = if let Some(cached_plan) = self.query_cache.get_optimized(&cache_key) {
            cached_plan
        } else {
            // Parse and translate the query to a logical plan
            let logical_plan = cypher_translator::translate(query)?;

            // Semantic validation
            let mut binder = Binder::new();
            let _binding_context = binder.bind(&logical_plan)?;

            // Optimize the plan
            let optimizer = Optimizer::from_store(&self.store);
            let plan = optimizer.optimize(logical_plan)?;

            // Cache the optimized plan
            self.query_cache.put_optimized(cache_key, plan.clone());

            plan
        };

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
        let optimizer = Optimizer::from_store(&self.store);
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
        let optimizer = Optimizer::from_store(&self.store);
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
        let optimizer = Optimizer::from_store(&self.store);
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

    /// Begins a transaction with a specific isolation level.
    ///
    /// See [`begin_tx`](Self::begin_tx) for the default (`SnapshotIsolation`).
    ///
    /// # Errors
    ///
    /// Returns an error if a transaction is already active.
    pub fn begin_tx_with_isolation(
        &mut self,
        isolation_level: crate::transaction::IsolationLevel,
    ) -> Result<()> {
        if self.current_tx.is_some() {
            return Err(grafeo_common::utils::error::Error::Transaction(
                grafeo_common::utils::error::TransactionError::InvalidState(
                    "Transaction already active".to_string(),
                ),
            ));
        }

        let tx_id = self.tx_manager.begin_with_isolation(isolation_level);
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

    // =========================================================================
    // Direct Lookup APIs (bypass query planning for O(1) point reads)
    // =========================================================================

    /// Gets a node by ID directly, bypassing query planning.
    ///
    /// This is the fastest way to retrieve a single node when you know its ID.
    /// Skips parsing, binding, optimization, and physical planning entirely.
    ///
    /// # Performance
    ///
    /// - Time complexity: O(1) average case
    /// - No lock contention (uses DashMap internally)
    /// - ~20-30x faster than equivalent MATCH query
    ///
    /// # Example
    ///
    /// ```ignore
    /// let session = db.session();
    /// let node_id = session.create_node(&["Person"]);
    ///
    /// // Direct lookup - O(1), no query planning
    /// let node = session.get_node(node_id);
    /// assert!(node.is_some());
    /// ```
    #[must_use]
    pub fn get_node(&self, id: NodeId) -> Option<Node> {
        let (epoch, tx_id) = self.get_transaction_context();
        self.store
            .get_node_versioned(id, epoch, tx_id.unwrap_or(TxId::SYSTEM))
    }

    /// Gets a single property from a node by ID, bypassing query planning.
    ///
    /// More efficient than `get_node()` when you only need one property,
    /// as it avoids loading the full node with all properties.
    ///
    /// # Performance
    ///
    /// - Time complexity: O(1) average case
    /// - No query planning overhead
    ///
    /// # Example
    ///
    /// ```ignore
    /// let session = db.session();
    /// let id = session.create_node_with_props(&["Person"], [("name", "Alice".into())]);
    ///
    /// // Direct property access - O(1)
    /// let name = session.get_node_property(id, "name");
    /// assert_eq!(name, Some(Value::String("Alice".into())));
    /// ```
    #[must_use]
    pub fn get_node_property(&self, id: NodeId, key: &str) -> Option<Value> {
        self.get_node(id)
            .and_then(|node| node.get_property(key).cloned())
    }

    /// Gets an edge by ID directly, bypassing query planning.
    ///
    /// # Performance
    ///
    /// - Time complexity: O(1) average case
    /// - No lock contention
    #[must_use]
    pub fn get_edge(&self, id: EdgeId) -> Option<Edge> {
        let (epoch, tx_id) = self.get_transaction_context();
        self.store
            .get_edge_versioned(id, epoch, tx_id.unwrap_or(TxId::SYSTEM))
    }

    /// Gets outgoing neighbors of a node directly, bypassing query planning.
    ///
    /// Returns (neighbor_id, edge_id) pairs for all outgoing edges.
    ///
    /// # Performance
    ///
    /// - Time complexity: O(degree) where degree is the number of outgoing edges
    /// - Uses adjacency index for direct access
    /// - ~10-20x faster than equivalent MATCH query
    ///
    /// # Example
    ///
    /// ```ignore
    /// let session = db.session();
    /// let alice = session.create_node(&["Person"]);
    /// let bob = session.create_node(&["Person"]);
    /// session.create_edge(alice, bob, "KNOWS");
    ///
    /// // Direct neighbor lookup - O(degree)
    /// let neighbors = session.get_neighbors_outgoing(alice);
    /// assert_eq!(neighbors.len(), 1);
    /// assert_eq!(neighbors[0].0, bob);
    /// ```
    #[must_use]
    pub fn get_neighbors_outgoing(&self, node: NodeId) -> Vec<(NodeId, EdgeId)> {
        self.store.edges_from(node, Direction::Outgoing).collect()
    }

    /// Gets incoming neighbors of a node directly, bypassing query planning.
    ///
    /// Returns (neighbor_id, edge_id) pairs for all incoming edges.
    ///
    /// # Performance
    ///
    /// - Time complexity: O(degree) where degree is the number of incoming edges
    /// - Uses backward adjacency index for direct access
    #[must_use]
    pub fn get_neighbors_incoming(&self, node: NodeId) -> Vec<(NodeId, EdgeId)> {
        self.store.edges_from(node, Direction::Incoming).collect()
    }

    /// Gets outgoing neighbors filtered by edge type, bypassing query planning.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let neighbors = session.get_neighbors_outgoing_by_type(alice, "KNOWS");
    /// ```
    #[must_use]
    pub fn get_neighbors_outgoing_by_type(
        &self,
        node: NodeId,
        edge_type: &str,
    ) -> Vec<(NodeId, EdgeId)> {
        self.store
            .edges_from(node, Direction::Outgoing)
            .filter(|(_, edge_id)| {
                self.get_edge(*edge_id)
                    .is_some_and(|e| e.edge_type.as_str() == edge_type)
            })
            .collect()
    }

    /// Checks if a node exists, bypassing query planning.
    ///
    /// # Performance
    ///
    /// - Time complexity: O(1)
    /// - Fastest existence check available
    #[must_use]
    pub fn node_exists(&self, id: NodeId) -> bool {
        self.get_node(id).is_some()
    }

    /// Checks if an edge exists, bypassing query planning.
    #[must_use]
    pub fn edge_exists(&self, id: EdgeId) -> bool {
        self.get_edge(id).is_some()
    }

    /// Gets the degree (number of edges) of a node.
    ///
    /// Returns (outgoing_degree, incoming_degree).
    #[must_use]
    pub fn get_degree(&self, node: NodeId) -> (usize, usize) {
        let out = self.store.out_degree(node);
        let in_degree = self.store.in_degree(node);
        (out, in_degree)
    }

    /// Batch lookup of multiple nodes by ID.
    ///
    /// More efficient than calling `get_node()` in a loop because it
    /// amortizes overhead.
    ///
    /// # Performance
    ///
    /// - Time complexity: O(n) where n is the number of IDs
    /// - Better cache utilization than individual lookups
    #[must_use]
    pub fn get_nodes_batch(&self, ids: &[NodeId]) -> Vec<Option<Node>> {
        let (epoch, tx_id) = self.get_transaction_context();
        let tx = tx_id.unwrap_or(TxId::SYSTEM);
        ids.iter()
            .map(|&id| self.store.get_node_versioned(id, epoch, tx))
            .collect()
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

    // ==================== Direct Lookup API Tests ====================

    mod direct_lookup_tests {
        use super::*;
        use grafeo_common::types::Value;

        #[test]
        fn test_get_node() {
            let db = GrafeoDB::new_in_memory();
            let session = db.session();

            let id = session.create_node(&["Person"]);
            let node = session.get_node(id);

            assert!(node.is_some());
            let node = node.unwrap();
            assert_eq!(node.id, id);
        }

        #[test]
        fn test_get_node_not_found() {
            use grafeo_common::types::NodeId;

            let db = GrafeoDB::new_in_memory();
            let session = db.session();

            // Try to get a non-existent node
            let node = session.get_node(NodeId::new(9999));
            assert!(node.is_none());
        }

        #[test]
        fn test_get_node_property() {
            let db = GrafeoDB::new_in_memory();
            let session = db.session();

            let id = session
                .create_node_with_props(&["Person"], [("name", Value::String("Alice".into()))]);

            let name = session.get_node_property(id, "name");
            assert_eq!(name, Some(Value::String("Alice".into())));

            // Non-existent property
            let missing = session.get_node_property(id, "missing");
            assert!(missing.is_none());
        }

        #[test]
        fn test_get_edge() {
            let db = GrafeoDB::new_in_memory();
            let session = db.session();

            let alice = session.create_node(&["Person"]);
            let bob = session.create_node(&["Person"]);
            let edge_id = session.create_edge(alice, bob, "KNOWS");

            let edge = session.get_edge(edge_id);
            assert!(edge.is_some());
            let edge = edge.unwrap();
            assert_eq!(edge.id, edge_id);
            assert_eq!(edge.src, alice);
            assert_eq!(edge.dst, bob);
        }

        #[test]
        fn test_get_edge_not_found() {
            use grafeo_common::types::EdgeId;

            let db = GrafeoDB::new_in_memory();
            let session = db.session();

            let edge = session.get_edge(EdgeId::new(9999));
            assert!(edge.is_none());
        }

        #[test]
        fn test_get_neighbors_outgoing() {
            let db = GrafeoDB::new_in_memory();
            let session = db.session();

            let alice = session.create_node(&["Person"]);
            let bob = session.create_node(&["Person"]);
            let carol = session.create_node(&["Person"]);

            session.create_edge(alice, bob, "KNOWS");
            session.create_edge(alice, carol, "KNOWS");

            let neighbors = session.get_neighbors_outgoing(alice);
            assert_eq!(neighbors.len(), 2);

            let neighbor_ids: Vec<_> = neighbors.iter().map(|(node_id, _)| *node_id).collect();
            assert!(neighbor_ids.contains(&bob));
            assert!(neighbor_ids.contains(&carol));
        }

        #[test]
        fn test_get_neighbors_incoming() {
            let db = GrafeoDB::new_in_memory();
            let session = db.session();

            let alice = session.create_node(&["Person"]);
            let bob = session.create_node(&["Person"]);
            let carol = session.create_node(&["Person"]);

            session.create_edge(bob, alice, "KNOWS");
            session.create_edge(carol, alice, "KNOWS");

            let neighbors = session.get_neighbors_incoming(alice);
            assert_eq!(neighbors.len(), 2);

            let neighbor_ids: Vec<_> = neighbors.iter().map(|(node_id, _)| *node_id).collect();
            assert!(neighbor_ids.contains(&bob));
            assert!(neighbor_ids.contains(&carol));
        }

        #[test]
        fn test_get_neighbors_outgoing_by_type() {
            let db = GrafeoDB::new_in_memory();
            let session = db.session();

            let alice = session.create_node(&["Person"]);
            let bob = session.create_node(&["Person"]);
            let company = session.create_node(&["Company"]);

            session.create_edge(alice, bob, "KNOWS");
            session.create_edge(alice, company, "WORKS_AT");

            let knows_neighbors = session.get_neighbors_outgoing_by_type(alice, "KNOWS");
            assert_eq!(knows_neighbors.len(), 1);
            assert_eq!(knows_neighbors[0].0, bob);

            let works_neighbors = session.get_neighbors_outgoing_by_type(alice, "WORKS_AT");
            assert_eq!(works_neighbors.len(), 1);
            assert_eq!(works_neighbors[0].0, company);

            // No edges of this type
            let no_neighbors = session.get_neighbors_outgoing_by_type(alice, "LIKES");
            assert!(no_neighbors.is_empty());
        }

        #[test]
        fn test_node_exists() {
            use grafeo_common::types::NodeId;

            let db = GrafeoDB::new_in_memory();
            let session = db.session();

            let id = session.create_node(&["Person"]);

            assert!(session.node_exists(id));
            assert!(!session.node_exists(NodeId::new(9999)));
        }

        #[test]
        fn test_edge_exists() {
            use grafeo_common::types::EdgeId;

            let db = GrafeoDB::new_in_memory();
            let session = db.session();

            let alice = session.create_node(&["Person"]);
            let bob = session.create_node(&["Person"]);
            let edge_id = session.create_edge(alice, bob, "KNOWS");

            assert!(session.edge_exists(edge_id));
            assert!(!session.edge_exists(EdgeId::new(9999)));
        }

        #[test]
        fn test_get_degree() {
            let db = GrafeoDB::new_in_memory();
            let session = db.session();

            let alice = session.create_node(&["Person"]);
            let bob = session.create_node(&["Person"]);
            let carol = session.create_node(&["Person"]);

            // Alice knows Bob and Carol (2 outgoing)
            session.create_edge(alice, bob, "KNOWS");
            session.create_edge(alice, carol, "KNOWS");
            // Bob knows Alice (1 incoming for Alice)
            session.create_edge(bob, alice, "KNOWS");

            let (out_degree, in_degree) = session.get_degree(alice);
            assert_eq!(out_degree, 2);
            assert_eq!(in_degree, 1);

            // Node with no edges
            let lonely = session.create_node(&["Person"]);
            let (out, in_deg) = session.get_degree(lonely);
            assert_eq!(out, 0);
            assert_eq!(in_deg, 0);
        }

        #[test]
        fn test_get_nodes_batch() {
            let db = GrafeoDB::new_in_memory();
            let session = db.session();

            let alice = session.create_node(&["Person"]);
            let bob = session.create_node(&["Person"]);
            let carol = session.create_node(&["Person"]);

            let nodes = session.get_nodes_batch(&[alice, bob, carol]);
            assert_eq!(nodes.len(), 3);
            assert!(nodes[0].is_some());
            assert!(nodes[1].is_some());
            assert!(nodes[2].is_some());

            // With non-existent node
            use grafeo_common::types::NodeId;
            let nodes_with_missing = session.get_nodes_batch(&[alice, NodeId::new(9999), carol]);
            assert_eq!(nodes_with_missing.len(), 3);
            assert!(nodes_with_missing[0].is_some());
            assert!(nodes_with_missing[1].is_none()); // Missing node
            assert!(nodes_with_missing[2].is_some());
        }

        #[test]
        fn test_auto_commit_setting() {
            let db = GrafeoDB::new_in_memory();
            let mut session = db.session();

            // Default is auto-commit enabled
            assert!(session.auto_commit());

            session.set_auto_commit(false);
            assert!(!session.auto_commit());

            session.set_auto_commit(true);
            assert!(session.auto_commit());
        }

        #[test]
        fn test_transaction_double_begin_error() {
            let db = GrafeoDB::new_in_memory();
            let mut session = db.session();

            session.begin_tx().unwrap();
            let result = session.begin_tx();

            assert!(result.is_err());
            // Clean up
            session.rollback().unwrap();
        }

        #[test]
        fn test_commit_without_transaction_error() {
            let db = GrafeoDB::new_in_memory();
            let mut session = db.session();

            let result = session.commit();
            assert!(result.is_err());
        }

        #[test]
        fn test_rollback_without_transaction_error() {
            let db = GrafeoDB::new_in_memory();
            let mut session = db.session();

            let result = session.rollback();
            assert!(result.is_err());
        }

        #[test]
        fn test_create_edge_in_transaction() {
            let db = GrafeoDB::new_in_memory();
            let mut session = db.session();

            // Create nodes outside transaction
            let alice = session.create_node(&["Person"]);
            let bob = session.create_node(&["Person"]);

            // Create edge in transaction
            session.begin_tx().unwrap();
            let edge_id = session.create_edge(alice, bob, "KNOWS");

            // Edge should be visible in the transaction
            assert!(session.edge_exists(edge_id));

            // Commit
            session.commit().unwrap();

            // Edge should still be visible
            assert!(session.edge_exists(edge_id));
        }

        #[test]
        fn test_neighbors_empty_node() {
            let db = GrafeoDB::new_in_memory();
            let session = db.session();

            let lonely = session.create_node(&["Person"]);

            assert!(session.get_neighbors_outgoing(lonely).is_empty());
            assert!(session.get_neighbors_incoming(lonely).is_empty());
            assert!(
                session
                    .get_neighbors_outgoing_by_type(lonely, "KNOWS")
                    .is_empty()
            );
        }
    }
}
