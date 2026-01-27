//! GraphosDB main database struct.

use std::path::Path;
use std::sync::Arc;

use parking_lot::RwLock;

use graphos_common::memory::buffer::{BufferManager, BufferManagerConfig};
use graphos_common::utils::error::Result;
use graphos_core::graph::lpg::LpgStore;

use crate::config::Config;
use crate::session::Session;
use crate::transaction::TransactionManager;

/// The main Graphos database.
pub struct GraphosDB {
    /// Database configuration.
    config: Config,
    /// The underlying graph store.
    store: Arc<LpgStore>,
    /// Transaction manager.
    tx_manager: Arc<TransactionManager>,
    /// Unified buffer manager.
    buffer_manager: Arc<BufferManager>,
    /// Whether the database is open.
    is_open: RwLock<bool>,
}

impl GraphosDB {
    /// Creates a new in-memory database.
    ///
    /// # Examples
    ///
    /// ```
    /// use graphos_engine::GraphosDB;
    ///
    /// let db = GraphosDB::new_in_memory();
    /// let session = db.session();
    /// ```
    #[must_use]
    pub fn new_in_memory() -> Self {
        Self::with_config(Config::in_memory())
    }

    /// Opens or creates a database at the given path.
    ///
    /// # Errors
    ///
    /// Returns an error if the database cannot be opened or created.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use graphos_engine::GraphosDB;
    ///
    /// let db = GraphosDB::open("./my_database").expect("Failed to open database");
    /// ```
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        Ok(Self::with_config(Config::persistent(path.as_ref())))
    }

    /// Creates a database with the given configuration.
    ///
    /// # Examples
    ///
    /// ```
    /// use graphos_engine::{GraphosDB, Config};
    ///
    /// let config = Config::in_memory()
    ///     .with_memory_limit(512 * 1024 * 1024); // 512MB
    ///
    /// let db = GraphosDB::with_config(config);
    /// ```
    #[must_use]
    pub fn with_config(config: Config) -> Self {
        let store = Arc::new(LpgStore::new());
        let tx_manager = Arc::new(TransactionManager::new());

        // Create buffer manager with configured limits
        let buffer_config = BufferManagerConfig {
            budget: config.memory_limit.unwrap_or_else(|| {
                (BufferManagerConfig::detect_system_memory() as f64 * 0.75) as usize
            }),
            spill_path: config.spill_path.clone().or_else(|| {
                config.path.as_ref().map(|p| p.join("spill"))
            }),
            ..BufferManagerConfig::default()
        };
        let buffer_manager = BufferManager::new(buffer_config);

        Self {
            config,
            store,
            tx_manager,
            buffer_manager,
            is_open: RwLock::new(true),
        }
    }

    /// Creates a new session for interacting with the database.
    ///
    /// # Examples
    ///
    /// ```
    /// use graphos_engine::GraphosDB;
    ///
    /// let db = GraphosDB::new_in_memory();
    /// let session = db.session();
    /// // Use session for queries and transactions
    /// ```
    #[must_use]
    pub fn session(&self) -> Session {
        Session::new(Arc::clone(&self.store), Arc::clone(&self.tx_manager))
    }

    /// Executes a query and returns the result.
    ///
    /// This is a convenience method that creates a session, executes the query,
    /// and returns the result.
    ///
    /// # Errors
    ///
    /// Returns an error if the query fails.
    pub fn execute(&self, query: &str) -> Result<QueryResult> {
        let session = self.session();
        session.execute(query)
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

    /// Closes the database.
    pub fn close(&self) {
        *self.is_open.write() = false;
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

    // === Node Operations ===

    /// Creates a new node with the given labels.
    pub fn create_node(&self, labels: &[&str]) -> graphos_common::types::NodeId {
        self.store.create_node(labels)
    }

    /// Creates a new node with labels and properties.
    pub fn create_node_with_props(
        &self,
        labels: &[&str],
        properties: impl IntoIterator<
            Item = (
                impl Into<graphos_common::types::PropertyKey>,
                impl Into<graphos_common::types::Value>,
            ),
        >,
    ) -> graphos_common::types::NodeId {
        self.store.create_node_with_props(labels, properties)
    }

    /// Gets a node by ID.
    #[must_use]
    pub fn get_node(
        &self,
        id: graphos_common::types::NodeId,
    ) -> Option<graphos_core::graph::lpg::Node> {
        self.store.get_node(id)
    }

    /// Deletes a node and all its edges.
    pub fn delete_node(&self, id: graphos_common::types::NodeId) -> bool {
        self.store.delete_node(id)
    }

    // === Edge Operations ===

    /// Creates a new edge between two nodes.
    pub fn create_edge(
        &self,
        src: graphos_common::types::NodeId,
        dst: graphos_common::types::NodeId,
        edge_type: &str,
    ) -> graphos_common::types::EdgeId {
        self.store.create_edge(src, dst, edge_type)
    }

    /// Creates a new edge with properties.
    pub fn create_edge_with_props(
        &self,
        src: graphos_common::types::NodeId,
        dst: graphos_common::types::NodeId,
        edge_type: &str,
        properties: impl IntoIterator<
            Item = (
                impl Into<graphos_common::types::PropertyKey>,
                impl Into<graphos_common::types::Value>,
            ),
        >,
    ) -> graphos_common::types::EdgeId {
        self.store
            .create_edge_with_props(src, dst, edge_type, properties)
    }

    /// Gets an edge by ID.
    #[must_use]
    pub fn get_edge(
        &self,
        id: graphos_common::types::EdgeId,
    ) -> Option<graphos_core::graph::lpg::Edge> {
        self.store.get_edge(id)
    }

    /// Deletes an edge.
    pub fn delete_edge(&self, id: graphos_common::types::EdgeId) -> bool {
        self.store.delete_edge(id)
    }
}

impl Drop for GraphosDB {
    fn drop(&mut self) {
        self.close();
    }
}

/// Result of a query execution.
pub struct QueryResult {
    /// Column names.
    pub columns: Vec<String>,
    /// Result rows.
    pub rows: Vec<Vec<graphos_common::types::Value>>,
}

impl QueryResult {
    /// Creates a new empty query result.
    #[must_use]
    pub fn new(columns: Vec<String>) -> Self {
        Self {
            columns,
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

    /// Gets a single scalar value from the result.
    ///
    /// # Errors
    ///
    /// Returns an error if the result doesn't have exactly one row and one column.
    pub fn scalar<T: FromValue>(&self) -> Result<T> {
        if self.rows.len() != 1 || self.columns.len() != 1 {
            return Err(graphos_common::utils::error::Error::InvalidValue(
                "Expected single value".to_string(),
            ));
        }
        T::from_value(&self.rows[0][0])
    }

    /// Returns an iterator over the rows.
    pub fn iter(&self) -> impl Iterator<Item = &Vec<graphos_common::types::Value>> {
        self.rows.iter()
    }
}

/// Trait for converting from Value.
pub trait FromValue: Sized {
    /// Converts from a Value.
    ///
    /// # Errors
    ///
    /// Returns an error if the conversion fails.
    fn from_value(value: &graphos_common::types::Value) -> Result<Self>;
}

impl FromValue for i64 {
    fn from_value(value: &graphos_common::types::Value) -> Result<Self> {
        value
            .as_int64()
            .ok_or_else(|| graphos_common::utils::error::Error::TypeMismatch {
                expected: "INT64".to_string(),
                found: value.type_name().to_string(),
            })
    }
}

impl FromValue for f64 {
    fn from_value(value: &graphos_common::types::Value) -> Result<Self> {
        value
            .as_float64()
            .ok_or_else(|| graphos_common::utils::error::Error::TypeMismatch {
                expected: "FLOAT64".to_string(),
                found: value.type_name().to_string(),
            })
    }
}

impl FromValue for String {
    fn from_value(value: &graphos_common::types::Value) -> Result<Self> {
        value.as_str().map(String::from).ok_or_else(|| {
            graphos_common::utils::error::Error::TypeMismatch {
                expected: "STRING".to_string(),
                found: value.type_name().to_string(),
            }
        })
    }
}

impl FromValue for bool {
    fn from_value(value: &graphos_common::types::Value) -> Result<Self> {
        value
            .as_bool()
            .ok_or_else(|| graphos_common::utils::error::Error::TypeMismatch {
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
        let db = GraphosDB::new_in_memory();
        assert_eq!(db.node_count(), 0);
        assert_eq!(db.edge_count(), 0);
    }

    #[test]
    fn test_database_config() {
        let config = Config::in_memory().with_threads(4).with_query_logging();

        let db = GraphosDB::with_config(config);
        assert_eq!(db.config().threads, 4);
        assert!(db.config().query_logging);
    }

    #[test]
    fn test_database_session() {
        let db = GraphosDB::new_in_memory();
        let _session = db.session();
        // Session should be created successfully
    }
}
