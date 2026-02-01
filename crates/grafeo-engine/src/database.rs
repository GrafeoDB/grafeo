//! The main database struct and operations.
//!
//! Start here with [`GrafeoDB`] - it's your handle to everything.

use std::path::Path;
use std::sync::Arc;

use parking_lot::RwLock;

use grafeo_adapters::storage::wal::{WalConfig, WalManager, WalRecord, WalRecovery};
use grafeo_common::memory::buffer::{BufferManager, BufferManagerConfig};
use grafeo_common::utils::error::Result;
use grafeo_core::graph::lpg::LpgStore;
#[cfg(feature = "rdf")]
use grafeo_core::graph::rdf::RdfStore;

use crate::config::Config;
use crate::session::Session;
use crate::transaction::TransactionManager;

/// Your handle to a Grafeo database.
///
/// Start here. Create one with [`new_in_memory()`](Self::new_in_memory) for
/// quick experiments, or [`open()`](Self::open) for persistent storage.
/// Then grab a [`session()`](Self::session) to start querying.
///
/// # Examples
///
/// ```
/// use grafeo_engine::GrafeoDB;
///
/// // Quick in-memory database
/// let db = GrafeoDB::new_in_memory();
///
/// // Add some data
/// db.create_node(&["Person"]);
///
/// // Query it
/// let session = db.session();
/// let result = session.execute("MATCH (p:Person) RETURN p")?;
/// # Ok::<(), grafeo_common::utils::error::Error>(())
/// ```
pub struct GrafeoDB {
    /// Database configuration.
    config: Config,
    /// The underlying graph store.
    store: Arc<LpgStore>,
    /// RDF triple store (if RDF feature is enabled).
    #[cfg(feature = "rdf")]
    rdf_store: Arc<RdfStore>,
    /// Transaction manager.
    tx_manager: Arc<TransactionManager>,
    /// Unified buffer manager.
    buffer_manager: Arc<BufferManager>,
    /// Write-ahead log manager (if durability is enabled).
    wal: Option<Arc<WalManager>>,
    /// Whether the database is open.
    is_open: RwLock<bool>,
}

impl GrafeoDB {
    /// Creates an in-memory database - fast to create, gone when dropped.
    ///
    /// Use this for tests, experiments, or when you don't need persistence.
    /// For data that survives restarts, use [`open()`](Self::open) instead.
    ///
    /// # Examples
    ///
    /// ```
    /// use grafeo_engine::GrafeoDB;
    ///
    /// let db = GrafeoDB::new_in_memory();
    /// let session = db.session();
    /// session.execute("INSERT (:Person {name: 'Alice'})")?;
    /// # Ok::<(), grafeo_common::utils::error::Error>(())
    /// ```
    #[must_use]
    pub fn new_in_memory() -> Self {
        Self::with_config(Config::in_memory()).expect("In-memory database creation should not fail")
    }

    /// Opens a database at the given path, creating it if it doesn't exist.
    ///
    /// If you've used this path before, Grafeo recovers your data from the
    /// write-ahead log automatically. First open on a new path creates an
    /// empty database.
    ///
    /// # Errors
    ///
    /// Returns an error if the path isn't writable or recovery fails.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use grafeo_engine::GrafeoDB;
    ///
    /// let db = GrafeoDB::open("./my_social_network")?;
    /// # Ok::<(), grafeo_common::utils::error::Error>(())
    /// ```
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        Self::with_config(Config::persistent(path.as_ref()))
    }

    /// Creates a database with custom configuration.
    ///
    /// Use this when you need fine-grained control over memory limits,
    /// thread counts, or persistence settings. For most cases,
    /// [`new_in_memory()`](Self::new_in_memory) or [`open()`](Self::open)
    /// are simpler.
    ///
    /// # Errors
    ///
    /// Returns an error if the database can't be created or recovery fails.
    ///
    /// # Examples
    ///
    /// ```
    /// use grafeo_engine::{GrafeoDB, Config};
    ///
    /// // In-memory with a 512MB limit
    /// let config = Config::in_memory()
    ///     .with_memory_limit(512 * 1024 * 1024);
    ///
    /// let db = GrafeoDB::with_config(config)?;
    /// # Ok::<(), grafeo_common::utils::error::Error>(())
    /// ```
    pub fn with_config(config: Config) -> Result<Self> {
        let store = Arc::new(LpgStore::new());
        #[cfg(feature = "rdf")]
        let rdf_store = Arc::new(RdfStore::new());
        let tx_manager = Arc::new(TransactionManager::new());

        // Create buffer manager with configured limits
        let buffer_config = BufferManagerConfig {
            budget: config.memory_limit.unwrap_or_else(|| {
                (BufferManagerConfig::detect_system_memory() as f64 * 0.75) as usize
            }),
            spill_path: config
                .spill_path
                .clone()
                .or_else(|| config.path.as_ref().map(|p| p.join("spill"))),
            ..BufferManagerConfig::default()
        };
        let buffer_manager = BufferManager::new(buffer_config);

        // Initialize WAL if persistence is enabled
        let wal = if config.wal_enabled {
            if let Some(ref db_path) = config.path {
                // Create database directory if it doesn't exist
                std::fs::create_dir_all(db_path)?;

                let wal_path = db_path.join("wal");

                // Check if WAL exists and recover if needed
                if wal_path.exists() {
                    let recovery = WalRecovery::new(&wal_path);
                    let records = recovery.recover()?;
                    Self::apply_wal_records(&store, &records)?;
                }

                // Open/create WAL manager
                let wal_config = WalConfig::default();
                let wal_manager = WalManager::with_config(&wal_path, wal_config)?;
                Some(Arc::new(wal_manager))
            } else {
                None
            }
        } else {
            None
        };

        Ok(Self {
            config,
            store,
            #[cfg(feature = "rdf")]
            rdf_store,
            tx_manager,
            buffer_manager,
            wal,
            is_open: RwLock::new(true),
        })
    }

    /// Applies WAL records to restore the database state.
    fn apply_wal_records(store: &LpgStore, records: &[WalRecord]) -> Result<()> {
        for record in records {
            match record {
                WalRecord::CreateNode { id, labels } => {
                    let label_refs: Vec<&str> = labels.iter().map(|s| s.as_str()).collect();
                    store.create_node_with_id(*id, &label_refs);
                }
                WalRecord::DeleteNode { id } => {
                    store.delete_node(*id);
                }
                WalRecord::CreateEdge {
                    id,
                    src,
                    dst,
                    edge_type,
                } => {
                    store.create_edge_with_id(*id, *src, *dst, edge_type);
                }
                WalRecord::DeleteEdge { id } => {
                    store.delete_edge(*id);
                }
                WalRecord::SetNodeProperty { id, key, value } => {
                    store.set_node_property(*id, key, value.clone());
                }
                WalRecord::SetEdgeProperty { id, key, value } => {
                    store.set_edge_property(*id, key, value.clone());
                }
                WalRecord::AddNodeLabel { id, label } => {
                    store.add_label(*id, label);
                }
                WalRecord::RemoveNodeLabel { id, label } => {
                    store.remove_label(*id, label);
                }
                WalRecord::TxCommit { .. }
                | WalRecord::TxAbort { .. }
                | WalRecord::Checkpoint { .. } => {
                    // Transaction control records don't need replay action
                    // (recovery already filtered to only committed transactions)
                }
            }
        }
        Ok(())
    }

    /// Opens a new session for running queries.
    ///
    /// Sessions are cheap to create - spin up as many as you need. Each
    /// gets its own transaction context, so concurrent sessions won't
    /// block each other on reads.
    ///
    /// # Examples
    ///
    /// ```
    /// use grafeo_engine::GrafeoDB;
    ///
    /// let db = GrafeoDB::new_in_memory();
    /// let session = db.session();
    ///
    /// // Run queries through the session
    /// let result = session.execute("MATCH (n) RETURN count(n)")?;
    /// # Ok::<(), grafeo_common::utils::error::Error>(())
    /// ```
    #[must_use]
    pub fn session(&self) -> Session {
        #[cfg(feature = "rdf")]
        {
            Session::with_rdf_store_and_adaptive(
                Arc::clone(&self.store),
                Arc::clone(&self.rdf_store),
                Arc::clone(&self.tx_manager),
                self.config.adaptive.clone(),
                self.config.factorized_execution,
            )
        }
        #[cfg(not(feature = "rdf"))]
        {
            Session::with_adaptive(
                Arc::clone(&self.store),
                Arc::clone(&self.tx_manager),
                self.config.adaptive.clone(),
                self.config.factorized_execution,
            )
        }
    }

    /// Returns the adaptive execution configuration.
    #[must_use]
    pub fn adaptive_config(&self) -> &crate::config::AdaptiveConfig {
        &self.config.adaptive
    }

    /// Runs a query directly on the database.
    ///
    /// A convenience method that creates a temporary session behind the
    /// scenes. If you're running multiple queries, grab a
    /// [`session()`](Self::session) instead to avoid the overhead.
    ///
    /// # Errors
    ///
    /// Returns an error if parsing or execution fails.
    pub fn execute(&self, query: &str) -> Result<QueryResult> {
        let session = self.session();
        session.execute(query)
    }

    /// Executes a query with parameters and returns the result.
    ///
    /// # Errors
    ///
    /// Returns an error if the query fails.
    pub fn execute_with_params(
        &self,
        query: &str,
        params: std::collections::HashMap<String, grafeo_common::types::Value>,
    ) -> Result<QueryResult> {
        let session = self.session();
        session.execute_with_params(query, params)
    }

    /// Executes a Cypher query and returns the result.
    ///
    /// # Errors
    ///
    /// Returns an error if the query fails.
    #[cfg(feature = "cypher")]
    pub fn execute_cypher(&self, query: &str) -> Result<QueryResult> {
        let session = self.session();
        session.execute_cypher(query)
    }

    /// Executes a Cypher query with parameters and returns the result.
    ///
    /// # Errors
    ///
    /// Returns an error if the query fails.
    #[cfg(feature = "cypher")]
    pub fn execute_cypher_with_params(
        &self,
        query: &str,
        params: std::collections::HashMap<String, grafeo_common::types::Value>,
    ) -> Result<QueryResult> {
        use crate::query::processor::{QueryLanguage, QueryProcessor};

        // Create processor
        let processor = QueryProcessor::for_lpg(Arc::clone(&self.store));
        processor.process(query, QueryLanguage::Cypher, Some(&params))
    }

    /// Executes a Gremlin query and returns the result.
    ///
    /// # Errors
    ///
    /// Returns an error if the query fails.
    #[cfg(feature = "gremlin")]
    pub fn execute_gremlin(&self, query: &str) -> Result<QueryResult> {
        let session = self.session();
        session.execute_gremlin(query)
    }

    /// Executes a Gremlin query with parameters and returns the result.
    ///
    /// # Errors
    ///
    /// Returns an error if the query fails.
    #[cfg(feature = "gremlin")]
    pub fn execute_gremlin_with_params(
        &self,
        query: &str,
        params: std::collections::HashMap<String, grafeo_common::types::Value>,
    ) -> Result<QueryResult> {
        let session = self.session();
        session.execute_gremlin_with_params(query, params)
    }

    /// Executes a GraphQL query and returns the result.
    ///
    /// # Errors
    ///
    /// Returns an error if the query fails.
    #[cfg(feature = "graphql")]
    pub fn execute_graphql(&self, query: &str) -> Result<QueryResult> {
        let session = self.session();
        session.execute_graphql(query)
    }

    /// Executes a GraphQL query with parameters and returns the result.
    ///
    /// # Errors
    ///
    /// Returns an error if the query fails.
    #[cfg(feature = "graphql")]
    pub fn execute_graphql_with_params(
        &self,
        query: &str,
        params: std::collections::HashMap<String, grafeo_common::types::Value>,
    ) -> Result<QueryResult> {
        let session = self.session();
        session.execute_graphql_with_params(query, params)
    }

    /// Executes a SPARQL query and returns the result.
    ///
    /// SPARQL queries operate on the RDF triple store.
    ///
    /// # Errors
    ///
    /// Returns an error if the query fails.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use grafeo_engine::GrafeoDB;
    ///
    /// let db = GrafeoDB::new_in_memory();
    /// let result = db.execute_sparql("SELECT ?s ?p ?o WHERE { ?s ?p ?o }")?;
    /// ```
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
        let planner = RdfPlanner::new(Arc::clone(&self.rdf_store));
        let mut physical_plan = planner.plan(&optimized_plan)?;

        // Execute the plan
        let executor = Executor::with_columns(physical_plan.columns.clone());
        executor.execute(physical_plan.operator.as_mut())
    }

    /// Returns the RDF store.
    ///
    /// This provides direct access to the RDF store for triple operations.
    #[cfg(feature = "rdf")]
    #[must_use]
    pub fn rdf_store(&self) -> &Arc<RdfStore> {
        &self.rdf_store
    }

    /// Executes a query and returns a single scalar value.
    ///
    /// # Errors
    ///
    /// Returns an error if the query fails or doesn't return exactly one row.
    pub fn query_scalar<T: FromValue>(&self, query: &str) -> Result<T> {
        let result = self.execute(query)?;
        result.scalar()
    }

    /// Returns the configuration.
    #[must_use]
    pub fn config(&self) -> &Config {
        &self.config
    }

    /// Returns the underlying store.
    ///
    /// This provides direct access to the LPG store for algorithm implementations.
    #[must_use]
    pub fn store(&self) -> &Arc<LpgStore> {
        &self.store
    }

    /// Returns the buffer manager for memory-aware operations.
    #[must_use]
    pub fn buffer_manager(&self) -> &Arc<BufferManager> {
        &self.buffer_manager
    }

    /// Closes the database, flushing all pending writes.
    ///
    /// For persistent databases, this ensures everything is safely on disk.
    /// Called automatically when the database is dropped, but you can call
    /// it explicitly if you need to guarantee durability at a specific point.
    ///
    /// # Errors
    ///
    /// Returns an error if the WAL can't be flushed (check disk space/permissions).
    pub fn close(&self) -> Result<()> {
        let mut is_open = self.is_open.write();
        if !*is_open {
            return Ok(());
        }

        // Commit and checkpoint WAL
        if let Some(ref wal) = self.wal {
            let epoch = self.store.current_epoch();

            // Use the last assigned transaction ID, or create a checkpoint-only tx
            let checkpoint_tx = self.tx_manager.last_assigned_tx_id().unwrap_or_else(|| {
                // No transactions have been started; begin one for checkpoint
                self.tx_manager.begin()
            });

            // Log a TxCommit to mark all pending records as committed
            wal.log(&WalRecord::TxCommit {
                tx_id: checkpoint_tx,
            })?;

            // Then checkpoint
            wal.checkpoint(checkpoint_tx, epoch)?;
            wal.sync()?;
        }

        *is_open = false;
        Ok(())
    }

    /// Returns the WAL manager if available.
    #[must_use]
    pub fn wal(&self) -> Option<&Arc<WalManager>> {
        self.wal.as_ref()
    }

    /// Logs a WAL record if WAL is enabled.
    fn log_wal(&self, record: &WalRecord) -> Result<()> {
        if let Some(ref wal) = self.wal {
            wal.log(record)?;
        }
        Ok(())
    }

    /// Returns the number of nodes in the database.
    #[must_use]
    pub fn node_count(&self) -> usize {
        self.store.node_count()
    }

    /// Returns the number of edges in the database.
    #[must_use]
    pub fn edge_count(&self) -> usize {
        self.store.edge_count()
    }

    /// Returns the number of distinct labels in the database.
    #[must_use]
    pub fn label_count(&self) -> usize {
        self.store.label_count()
    }

    /// Returns the number of distinct property keys in the database.
    #[must_use]
    pub fn property_key_count(&self) -> usize {
        self.store.property_key_count()
    }

    /// Returns the number of distinct edge types in the database.
    #[must_use]
    pub fn edge_type_count(&self) -> usize {
        self.store.edge_type_count()
    }

    // === Node Operations ===

    /// Creates a node with the given labels and returns its ID.
    ///
    /// Labels categorize nodes - think of them like tags. A node can have
    /// multiple labels (e.g., `["Person", "Employee"]`).
    ///
    /// # Examples
    ///
    /// ```
    /// use grafeo_engine::GrafeoDB;
    ///
    /// let db = GrafeoDB::new_in_memory();
    /// let alice = db.create_node(&["Person"]);
    /// let company = db.create_node(&["Company", "Startup"]);
    /// ```
    pub fn create_node(&self, labels: &[&str]) -> grafeo_common::types::NodeId {
        let id = self.store.create_node(labels);

        // Log to WAL if enabled
        if let Err(e) = self.log_wal(&WalRecord::CreateNode {
            id,
            labels: labels.iter().map(|s| s.to_string()).collect(),
        }) {
            tracing::warn!("Failed to log CreateNode to WAL: {}", e);
        }

        id
    }

    /// Creates a new node with labels and properties.
    ///
    /// If WAL is enabled, the operation is logged for durability.
    pub fn create_node_with_props(
        &self,
        labels: &[&str],
        properties: impl IntoIterator<
            Item = (
                impl Into<grafeo_common::types::PropertyKey>,
                impl Into<grafeo_common::types::Value>,
            ),
        >,
    ) -> grafeo_common::types::NodeId {
        // Collect properties first so we can log them to WAL
        let props: Vec<(
            grafeo_common::types::PropertyKey,
            grafeo_common::types::Value,
        )> = properties
            .into_iter()
            .map(|(k, v)| (k.into(), v.into()))
            .collect();

        let id = self
            .store
            .create_node_with_props(labels, props.iter().map(|(k, v)| (k.clone(), v.clone())));

        // Log node creation to WAL
        if let Err(e) = self.log_wal(&WalRecord::CreateNode {
            id,
            labels: labels.iter().map(|s| s.to_string()).collect(),
        }) {
            tracing::warn!("Failed to log CreateNode to WAL: {}", e);
        }

        // Log each property to WAL for full durability
        for (key, value) in props {
            if let Err(e) = self.log_wal(&WalRecord::SetNodeProperty {
                id,
                key: key.to_string(),
                value,
            }) {
                tracing::warn!("Failed to log SetNodeProperty to WAL: {}", e);
            }
        }

        id
    }

    /// Gets a node by ID.
    #[must_use]
    pub fn get_node(
        &self,
        id: grafeo_common::types::NodeId,
    ) -> Option<grafeo_core::graph::lpg::Node> {
        self.store.get_node(id)
    }

    /// Deletes a node and all its edges.
    ///
    /// If WAL is enabled, the operation is logged for durability.
    pub fn delete_node(&self, id: grafeo_common::types::NodeId) -> bool {
        let result = self.store.delete_node(id);

        if result {
            if let Err(e) = self.log_wal(&WalRecord::DeleteNode { id }) {
                tracing::warn!("Failed to log DeleteNode to WAL: {}", e);
            }
        }

        result
    }

    /// Sets a property on a node.
    ///
    /// If WAL is enabled, the operation is logged for durability.
    pub fn set_node_property(
        &self,
        id: grafeo_common::types::NodeId,
        key: &str,
        value: grafeo_common::types::Value,
    ) {
        // Log to WAL first
        if let Err(e) = self.log_wal(&WalRecord::SetNodeProperty {
            id,
            key: key.to_string(),
            value: value.clone(),
        }) {
            tracing::warn!("Failed to log SetNodeProperty to WAL: {}", e);
        }

        self.store.set_node_property(id, key, value);
    }

    /// Adds a label to an existing node.
    ///
    /// Returns `true` if the label was added, `false` if the node doesn't exist
    /// or already has the label.
    ///
    /// # Examples
    ///
    /// ```
    /// use grafeo_engine::GrafeoDB;
    ///
    /// let db = GrafeoDB::new_in_memory();
    /// let alice = db.create_node(&["Person"]);
    ///
    /// // Promote Alice to Employee
    /// let added = db.add_node_label(alice, "Employee");
    /// assert!(added);
    /// ```
    pub fn add_node_label(&self, id: grafeo_common::types::NodeId, label: &str) -> bool {
        let result = self.store.add_label(id, label);

        if result {
            // Log to WAL if enabled
            if let Err(e) = self.log_wal(&WalRecord::AddNodeLabel {
                id,
                label: label.to_string(),
            }) {
                tracing::warn!("Failed to log AddNodeLabel to WAL: {}", e);
            }
        }

        result
    }

    /// Removes a label from a node.
    ///
    /// Returns `true` if the label was removed, `false` if the node doesn't exist
    /// or doesn't have the label.
    ///
    /// # Examples
    ///
    /// ```
    /// use grafeo_engine::GrafeoDB;
    ///
    /// let db = GrafeoDB::new_in_memory();
    /// let alice = db.create_node(&["Person", "Employee"]);
    ///
    /// // Remove Employee status
    /// let removed = db.remove_node_label(alice, "Employee");
    /// assert!(removed);
    /// ```
    pub fn remove_node_label(&self, id: grafeo_common::types::NodeId, label: &str) -> bool {
        let result = self.store.remove_label(id, label);

        if result {
            // Log to WAL if enabled
            if let Err(e) = self.log_wal(&WalRecord::RemoveNodeLabel {
                id,
                label: label.to_string(),
            }) {
                tracing::warn!("Failed to log RemoveNodeLabel to WAL: {}", e);
            }
        }

        result
    }

    /// Gets all labels for a node.
    ///
    /// Returns `None` if the node doesn't exist.
    ///
    /// # Examples
    ///
    /// ```
    /// use grafeo_engine::GrafeoDB;
    ///
    /// let db = GrafeoDB::new_in_memory();
    /// let alice = db.create_node(&["Person", "Employee"]);
    ///
    /// let labels = db.get_node_labels(alice).unwrap();
    /// assert!(labels.contains(&"Person".to_string()));
    /// assert!(labels.contains(&"Employee".to_string()));
    /// ```
    #[must_use]
    pub fn get_node_labels(&self, id: grafeo_common::types::NodeId) -> Option<Vec<String>> {
        self.store
            .get_node(id)
            .map(|node| node.labels.iter().map(|s| s.to_string()).collect())
    }

    // === Edge Operations ===

    /// Creates an edge (relationship) between two nodes.
    ///
    /// Edges connect nodes and have a type that describes the relationship.
    /// They're directed - the order of `src` and `dst` matters.
    ///
    /// # Examples
    ///
    /// ```
    /// use grafeo_engine::GrafeoDB;
    ///
    /// let db = GrafeoDB::new_in_memory();
    /// let alice = db.create_node(&["Person"]);
    /// let bob = db.create_node(&["Person"]);
    ///
    /// // Alice knows Bob (directed: Alice -> Bob)
    /// let edge = db.create_edge(alice, bob, "KNOWS");
    /// ```
    pub fn create_edge(
        &self,
        src: grafeo_common::types::NodeId,
        dst: grafeo_common::types::NodeId,
        edge_type: &str,
    ) -> grafeo_common::types::EdgeId {
        let id = self.store.create_edge(src, dst, edge_type);

        // Log to WAL if enabled
        if let Err(e) = self.log_wal(&WalRecord::CreateEdge {
            id,
            src,
            dst,
            edge_type: edge_type.to_string(),
        }) {
            tracing::warn!("Failed to log CreateEdge to WAL: {}", e);
        }

        id
    }

    /// Creates a new edge with properties.
    ///
    /// If WAL is enabled, the operation is logged for durability.
    pub fn create_edge_with_props(
        &self,
        src: grafeo_common::types::NodeId,
        dst: grafeo_common::types::NodeId,
        edge_type: &str,
        properties: impl IntoIterator<
            Item = (
                impl Into<grafeo_common::types::PropertyKey>,
                impl Into<grafeo_common::types::Value>,
            ),
        >,
    ) -> grafeo_common::types::EdgeId {
        // Collect properties first so we can log them to WAL
        let props: Vec<(
            grafeo_common::types::PropertyKey,
            grafeo_common::types::Value,
        )> = properties
            .into_iter()
            .map(|(k, v)| (k.into(), v.into()))
            .collect();

        let id = self.store.create_edge_with_props(
            src,
            dst,
            edge_type,
            props.iter().map(|(k, v)| (k.clone(), v.clone())),
        );

        // Log edge creation to WAL
        if let Err(e) = self.log_wal(&WalRecord::CreateEdge {
            id,
            src,
            dst,
            edge_type: edge_type.to_string(),
        }) {
            tracing::warn!("Failed to log CreateEdge to WAL: {}", e);
        }

        // Log each property to WAL for full durability
        for (key, value) in props {
            if let Err(e) = self.log_wal(&WalRecord::SetEdgeProperty {
                id,
                key: key.to_string(),
                value,
            }) {
                tracing::warn!("Failed to log SetEdgeProperty to WAL: {}", e);
            }
        }

        id
    }

    /// Gets an edge by ID.
    #[must_use]
    pub fn get_edge(
        &self,
        id: grafeo_common::types::EdgeId,
    ) -> Option<grafeo_core::graph::lpg::Edge> {
        self.store.get_edge(id)
    }

    /// Deletes an edge.
    ///
    /// If WAL is enabled, the operation is logged for durability.
    pub fn delete_edge(&self, id: grafeo_common::types::EdgeId) -> bool {
        let result = self.store.delete_edge(id);

        if result {
            if let Err(e) = self.log_wal(&WalRecord::DeleteEdge { id }) {
                tracing::warn!("Failed to log DeleteEdge to WAL: {}", e);
            }
        }

        result
    }

    /// Sets a property on an edge.
    ///
    /// If WAL is enabled, the operation is logged for durability.
    pub fn set_edge_property(
        &self,
        id: grafeo_common::types::EdgeId,
        key: &str,
        value: grafeo_common::types::Value,
    ) {
        // Log to WAL first
        if let Err(e) = self.log_wal(&WalRecord::SetEdgeProperty {
            id,
            key: key.to_string(),
            value: value.clone(),
        }) {
            tracing::warn!("Failed to log SetEdgeProperty to WAL: {}", e);
        }
        self.store.set_edge_property(id, key, value);
    }

    /// Removes a property from a node.
    ///
    /// Returns true if the property existed and was removed, false otherwise.
    pub fn remove_node_property(&self, id: grafeo_common::types::NodeId, key: &str) -> bool {
        // Note: RemoveProperty WAL records not yet implemented, but operation works in memory
        self.store.remove_node_property(id, key).is_some()
    }

    /// Removes a property from an edge.
    ///
    /// Returns true if the property existed and was removed, false otherwise.
    pub fn remove_edge_property(&self, id: grafeo_common::types::EdgeId, key: &str) -> bool {
        // Note: RemoveProperty WAL records not yet implemented, but operation works in memory
        self.store.remove_edge_property(id, key).is_some()
    }

    // =========================================================================
    // ADMIN API: Introspection
    // =========================================================================

    /// Returns true if this database is backed by a file (persistent).
    ///
    /// In-memory databases return false.
    #[must_use]
    pub fn is_persistent(&self) -> bool {
        self.config.path.is_some()
    }

    /// Returns the database file path, if persistent.
    ///
    /// In-memory databases return None.
    #[must_use]
    pub fn path(&self) -> Option<&Path> {
        self.config.path.as_deref()
    }

    /// Returns high-level database information.
    ///
    /// Includes node/edge counts, persistence status, and mode (LPG/RDF).
    #[must_use]
    pub fn info(&self) -> crate::admin::DatabaseInfo {
        crate::admin::DatabaseInfo {
            mode: crate::admin::DatabaseMode::Lpg,
            node_count: self.store.node_count(),
            edge_count: self.store.edge_count(),
            is_persistent: self.is_persistent(),
            path: self.config.path.clone(),
            wal_enabled: self.config.wal_enabled,
            version: env!("CARGO_PKG_VERSION").to_string(),
        }
    }

    /// Returns detailed database statistics.
    ///
    /// Includes counts, memory usage, and index information.
    #[must_use]
    pub fn detailed_stats(&self) -> crate::admin::DatabaseStats {
        let disk_bytes = self.config.path.as_ref().and_then(|p| {
            if p.exists() {
                Self::calculate_disk_usage(p).ok()
            } else {
                None
            }
        });

        crate::admin::DatabaseStats {
            node_count: self.store.node_count(),
            edge_count: self.store.edge_count(),
            label_count: self.store.label_count(),
            edge_type_count: self.store.edge_type_count(),
            property_key_count: self.store.property_key_count(),
            index_count: 0, // TODO: implement index tracking
            memory_bytes: self.buffer_manager.allocated(),
            disk_bytes,
        }
    }

    /// Calculates total disk usage for the database directory.
    fn calculate_disk_usage(path: &Path) -> Result<usize> {
        let mut total = 0usize;
        if path.is_dir() {
            for entry in std::fs::read_dir(path)? {
                let entry = entry?;
                let metadata = entry.metadata()?;
                if metadata.is_file() {
                    total += metadata.len() as usize;
                } else if metadata.is_dir() {
                    total += Self::calculate_disk_usage(&entry.path())?;
                }
            }
        }
        Ok(total)
    }

    /// Returns schema information (labels, edge types, property keys).
    ///
    /// For LPG mode, returns label and edge type information.
    /// For RDF mode, returns predicate and named graph information.
    #[must_use]
    pub fn schema(&self) -> crate::admin::SchemaInfo {
        let labels = self
            .store
            .all_labels()
            .into_iter()
            .map(|name| crate::admin::LabelInfo {
                name: name.clone(),
                count: self.store.nodes_with_label(&name).count(),
            })
            .collect();

        let edge_types = self
            .store
            .all_edge_types()
            .into_iter()
            .map(|name| crate::admin::EdgeTypeInfo {
                name: name.clone(),
                count: self.store.edges_with_type(&name).count(),
            })
            .collect();

        let property_keys = self.store.all_property_keys();

        crate::admin::SchemaInfo::Lpg(crate::admin::LpgSchemaInfo {
            labels,
            edge_types,
            property_keys,
        })
    }

    /// Returns RDF schema information.
    ///
    /// Only available when the RDF feature is enabled.
    #[cfg(feature = "rdf")]
    #[must_use]
    pub fn rdf_schema(&self) -> crate::admin::SchemaInfo {
        let stats = self.rdf_store.stats();

        let predicates = self
            .rdf_store
            .predicates()
            .into_iter()
            .map(|predicate| {
                let count = self.rdf_store.triples_with_predicate(&predicate).len();
                crate::admin::PredicateInfo {
                    iri: predicate.to_string(),
                    count,
                }
            })
            .collect();

        crate::admin::SchemaInfo::Rdf(crate::admin::RdfSchemaInfo {
            predicates,
            named_graphs: Vec::new(), // Named graphs not yet implemented in RdfStore
            subject_count: stats.subject_count,
            object_count: stats.object_count,
        })
    }

    /// Validates database integrity.
    ///
    /// Checks for:
    /// - Dangling edge references (edges pointing to non-existent nodes)
    /// - Internal index consistency
    ///
    /// Returns a list of errors and warnings. Empty errors = valid.
    #[must_use]
    pub fn validate(&self) -> crate::admin::ValidationResult {
        let mut result = crate::admin::ValidationResult::default();

        // Check for dangling edge references
        for edge in self.store.all_edges() {
            if self.store.get_node(edge.src).is_none() {
                result.errors.push(crate::admin::ValidationError {
                    code: "DANGLING_SRC".to_string(),
                    message: format!(
                        "Edge {} references non-existent source node {}",
                        edge.id.0, edge.src.0
                    ),
                    context: Some(format!("edge:{}", edge.id.0)),
                });
            }
            if self.store.get_node(edge.dst).is_none() {
                result.errors.push(crate::admin::ValidationError {
                    code: "DANGLING_DST".to_string(),
                    message: format!(
                        "Edge {} references non-existent destination node {}",
                        edge.id.0, edge.dst.0
                    ),
                    context: Some(format!("edge:{}", edge.id.0)),
                });
            }
        }

        // Add warnings for potential issues
        if self.store.node_count() > 0 && self.store.edge_count() == 0 {
            result.warnings.push(crate::admin::ValidationWarning {
                code: "NO_EDGES".to_string(),
                message: "Database has nodes but no edges".to_string(),
                context: None,
            });
        }

        result
    }

    /// Returns WAL (Write-Ahead Log) status.
    ///
    /// Returns None if WAL is not enabled.
    #[must_use]
    pub fn wal_status(&self) -> crate::admin::WalStatus {
        if let Some(ref wal) = self.wal {
            crate::admin::WalStatus {
                enabled: true,
                path: self.config.path.as_ref().map(|p| p.join("wal")),
                size_bytes: wal.size_bytes(),
                record_count: wal.record_count() as usize,
                last_checkpoint: wal.last_checkpoint_timestamp(),
                current_epoch: self.store.current_epoch().as_u64(),
            }
        } else {
            crate::admin::WalStatus {
                enabled: false,
                path: None,
                size_bytes: 0,
                record_count: 0,
                last_checkpoint: None,
                current_epoch: self.store.current_epoch().as_u64(),
            }
        }
    }

    /// Forces a WAL checkpoint.
    ///
    /// Flushes all pending WAL records to the main storage.
    ///
    /// # Errors
    ///
    /// Returns an error if the checkpoint fails.
    pub fn wal_checkpoint(&self) -> Result<()> {
        if let Some(ref wal) = self.wal {
            let epoch = self.store.current_epoch();
            let tx_id = self
                .tx_manager
                .last_assigned_tx_id()
                .unwrap_or_else(|| self.tx_manager.begin());
            wal.checkpoint(tx_id, epoch)?;
            wal.sync()?;
        }
        Ok(())
    }

    // =========================================================================
    // ADMIN API: Persistence Control
    // =========================================================================

    /// Saves the database to a file path.
    ///
    /// - If in-memory: creates a new persistent database at path
    /// - If file-backed: creates a copy at the new path
    ///
    /// The original database remains unchanged.
    ///
    /// # Errors
    ///
    /// Returns an error if the save operation fails.
    pub fn save(&self, path: impl AsRef<Path>) -> Result<()> {
        let path = path.as_ref();

        // Create target database with WAL enabled
        let target_config = Config::persistent(path);
        let target = Self::with_config(target_config)?;

        // Copy all nodes using WAL-enabled methods
        for node in self.store.all_nodes() {
            let label_refs: Vec<&str> = node.labels.iter().map(|s| &**s).collect();
            target.store.create_node_with_id(node.id, &label_refs);

            // Log to WAL
            target.log_wal(&WalRecord::CreateNode {
                id: node.id,
                labels: node.labels.iter().map(|s| s.to_string()).collect(),
            })?;

            // Copy properties
            for (key, value) in node.properties {
                target
                    .store
                    .set_node_property(node.id, key.as_str(), value.clone());
                target.log_wal(&WalRecord::SetNodeProperty {
                    id: node.id,
                    key: key.to_string(),
                    value,
                })?;
            }
        }

        // Copy all edges using WAL-enabled methods
        for edge in self.store.all_edges() {
            target
                .store
                .create_edge_with_id(edge.id, edge.src, edge.dst, &edge.edge_type);

            // Log to WAL
            target.log_wal(&WalRecord::CreateEdge {
                id: edge.id,
                src: edge.src,
                dst: edge.dst,
                edge_type: edge.edge_type.to_string(),
            })?;

            // Copy properties
            for (key, value) in edge.properties {
                target
                    .store
                    .set_edge_property(edge.id, key.as_str(), value.clone());
                target.log_wal(&WalRecord::SetEdgeProperty {
                    id: edge.id,
                    key: key.to_string(),
                    value,
                })?;
            }
        }

        // Checkpoint and close the target database
        target.close()?;

        Ok(())
    }

    /// Creates an in-memory copy of this database.
    ///
    /// Returns a new database that is completely independent.
    /// Useful for:
    /// - Testing modifications without affecting the original
    /// - Faster operations when persistence isn't needed
    ///
    /// # Errors
    ///
    /// Returns an error if the copy operation fails.
    pub fn to_memory(&self) -> Result<Self> {
        let config = Config::in_memory();
        let target = Self::with_config(config)?;

        // Copy all nodes
        for node in self.store.all_nodes() {
            let label_refs: Vec<&str> = node.labels.iter().map(|s| &**s).collect();
            target.store.create_node_with_id(node.id, &label_refs);

            // Copy properties
            for (key, value) in node.properties {
                target.store.set_node_property(node.id, key.as_str(), value);
            }
        }

        // Copy all edges
        for edge in self.store.all_edges() {
            target
                .store
                .create_edge_with_id(edge.id, edge.src, edge.dst, &edge.edge_type);

            // Copy properties
            for (key, value) in edge.properties {
                target.store.set_edge_property(edge.id, key.as_str(), value);
            }
        }

        Ok(target)
    }

    /// Opens a database file and loads it entirely into memory.
    ///
    /// The returned database has no connection to the original file.
    /// Changes will NOT be written back to the file.
    ///
    /// # Errors
    ///
    /// Returns an error if the file can't be opened or loaded.
    pub fn open_in_memory(path: impl AsRef<Path>) -> Result<Self> {
        // Open the source database (triggers WAL recovery)
        let source = Self::open(path)?;

        // Create in-memory copy
        let target = source.to_memory()?;

        // Close the source (releases file handles)
        source.close()?;

        Ok(target)
    }

    // =========================================================================
    // ADMIN API: Iteration
    // =========================================================================

    /// Returns an iterator over all nodes in the database.
    ///
    /// Useful for dump/export operations.
    pub fn iter_nodes(&self) -> impl Iterator<Item = grafeo_core::graph::lpg::Node> + '_ {
        self.store.all_nodes()
    }

    /// Returns an iterator over all edges in the database.
    ///
    /// Useful for dump/export operations.
    pub fn iter_edges(&self) -> impl Iterator<Item = grafeo_core::graph::lpg::Edge> + '_ {
        self.store.all_edges()
    }
}

impl Drop for GrafeoDB {
    fn drop(&mut self) {
        if let Err(e) = self.close() {
            tracing::error!("Error closing database: {}", e);
        }
    }
}

/// The result of running a query.
///
/// Contains rows and columns, like a table. Use [`iter()`](Self::iter) to
/// loop through rows, or [`scalar()`](Self::scalar) if you expect a single value.
///
/// # Examples
///
/// ```
/// use grafeo_engine::GrafeoDB;
///
/// let db = GrafeoDB::new_in_memory();
/// db.create_node(&["Person"]);
///
/// let result = db.execute("MATCH (p:Person) RETURN count(p) AS total")?;
///
/// // Check what we got
/// println!("Columns: {:?}", result.columns);
/// println!("Rows: {}", result.row_count());
///
/// // Iterate through results
/// for row in result.iter() {
///     println!("{:?}", row);
/// }
/// # Ok::<(), grafeo_common::utils::error::Error>(())
/// ```
#[derive(Debug)]
pub struct QueryResult {
    /// Column names from the RETURN clause.
    pub columns: Vec<String>,
    /// Column types - useful for distinguishing NodeId/EdgeId from plain integers.
    pub column_types: Vec<grafeo_common::types::LogicalType>,
    /// The actual result rows.
    pub rows: Vec<Vec<grafeo_common::types::Value>>,
}

impl QueryResult {
    /// Creates a new empty query result.
    #[must_use]
    pub fn new(columns: Vec<String>) -> Self {
        let len = columns.len();
        Self {
            columns,
            column_types: vec![grafeo_common::types::LogicalType::Any; len],
            rows: Vec::new(),
        }
    }

    /// Creates a new empty query result with column types.
    #[must_use]
    pub fn with_types(
        columns: Vec<String>,
        column_types: Vec<grafeo_common::types::LogicalType>,
    ) -> Self {
        Self {
            columns,
            column_types,
            rows: Vec::new(),
        }
    }

    /// Returns the number of rows.
    #[must_use]
    pub fn row_count(&self) -> usize {
        self.rows.len()
    }

    /// Returns the number of columns.
    #[must_use]
    pub fn column_count(&self) -> usize {
        self.columns.len()
    }

    /// Returns true if the result is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.rows.is_empty()
    }

    /// Extracts a single value from the result.
    ///
    /// Use this when your query returns exactly one row with one column,
    /// like `RETURN count(n)` or `RETURN sum(p.amount)`.
    ///
    /// # Errors
    ///
    /// Returns an error if the result has multiple rows or columns.
    pub fn scalar<T: FromValue>(&self) -> Result<T> {
        if self.rows.len() != 1 || self.columns.len() != 1 {
            return Err(grafeo_common::utils::error::Error::InvalidValue(
                "Expected single value".to_string(),
            ));
        }
        T::from_value(&self.rows[0][0])
    }

    /// Returns an iterator over the rows.
    pub fn iter(&self) -> impl Iterator<Item = &Vec<grafeo_common::types::Value>> {
        self.rows.iter()
    }
}

/// Converts a [`Value`](grafeo_common::types::Value) to a concrete Rust type.
///
/// Implemented for common types like `i64`, `f64`, `String`, and `bool`.
/// Used by [`QueryResult::scalar()`] to extract typed values.
pub trait FromValue: Sized {
    /// Attempts the conversion, returning an error on type mismatch.
    fn from_value(value: &grafeo_common::types::Value) -> Result<Self>;
}

impl FromValue for i64 {
    fn from_value(value: &grafeo_common::types::Value) -> Result<Self> {
        value
            .as_int64()
            .ok_or_else(|| grafeo_common::utils::error::Error::TypeMismatch {
                expected: "INT64".to_string(),
                found: value.type_name().to_string(),
            })
    }
}

impl FromValue for f64 {
    fn from_value(value: &grafeo_common::types::Value) -> Result<Self> {
        value
            .as_float64()
            .ok_or_else(|| grafeo_common::utils::error::Error::TypeMismatch {
                expected: "FLOAT64".to_string(),
                found: value.type_name().to_string(),
            })
    }
}

impl FromValue for String {
    fn from_value(value: &grafeo_common::types::Value) -> Result<Self> {
        value.as_str().map(String::from).ok_or_else(|| {
            grafeo_common::utils::error::Error::TypeMismatch {
                expected: "STRING".to_string(),
                found: value.type_name().to_string(),
            }
        })
    }
}

impl FromValue for bool {
    fn from_value(value: &grafeo_common::types::Value) -> Result<Self> {
        value
            .as_bool()
            .ok_or_else(|| grafeo_common::utils::error::Error::TypeMismatch {
                expected: "BOOL".to_string(),
                found: value.type_name().to_string(),
            })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_in_memory_database() {
        let db = GrafeoDB::new_in_memory();
        assert_eq!(db.node_count(), 0);
        assert_eq!(db.edge_count(), 0);
    }

    #[test]
    fn test_database_config() {
        let config = Config::in_memory().with_threads(4).with_query_logging();

        let db = GrafeoDB::with_config(config).unwrap();
        assert_eq!(db.config().threads, 4);
        assert!(db.config().query_logging);
    }

    #[test]
    fn test_database_session() {
        let db = GrafeoDB::new_in_memory();
        let _session = db.session();
        // Session should be created successfully
    }

    #[test]
    fn test_persistent_database_recovery() {
        use grafeo_common::types::Value;
        use tempfile::tempdir;

        let dir = tempdir().unwrap();
        let db_path = dir.path().join("test_db");

        // Create database and add some data
        {
            let db = GrafeoDB::open(&db_path).unwrap();

            let alice = db.create_node(&["Person"]);
            db.set_node_property(alice, "name", Value::from("Alice"));

            let bob = db.create_node(&["Person"]);
            db.set_node_property(bob, "name", Value::from("Bob"));

            let _edge = db.create_edge(alice, bob, "KNOWS");

            // Explicitly close to flush WAL
            db.close().unwrap();
        }

        // Reopen and verify data was recovered
        {
            let db = GrafeoDB::open(&db_path).unwrap();

            assert_eq!(db.node_count(), 2);
            assert_eq!(db.edge_count(), 1);

            // Verify nodes exist
            let node0 = db.get_node(grafeo_common::types::NodeId::new(0));
            assert!(node0.is_some());

            let node1 = db.get_node(grafeo_common::types::NodeId::new(1));
            assert!(node1.is_some());
        }
    }

    #[test]
    fn test_wal_logging() {
        use tempfile::tempdir;

        let dir = tempdir().unwrap();
        let db_path = dir.path().join("wal_test_db");

        let db = GrafeoDB::open(&db_path).unwrap();

        // Create some data
        let node = db.create_node(&["Test"]);
        db.delete_node(node);

        // WAL should have records
        if let Some(wal) = db.wal() {
            assert!(wal.record_count() > 0);
        }

        db.close().unwrap();
    }
}
