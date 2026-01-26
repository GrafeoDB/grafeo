//! GraphosDB main database struct.

use crate::config::Config;
use crate::session::Session;
use crate::transaction::TransactionManager;
use graphos_adapters::storage::MemoryBackend;
use graphos_common::utils::error::Result;
use graphos_core::graph::lpg::LpgStore;
use parking_lot::RwLock;
use std::path::Path;
use std::sync::Arc;

/// The main Graphos database.
pub struct GraphosDB {
    /// Database configuration.
    config: Config,
    /// The underlying graph store.
    store: Arc<LpgStore>,
    /// Transaction manager.
    tx_manager: Arc<TransactionManager>,
    /// Whether the database is open.
    is_open: RwLock<bool>,
}

impl GraphosDB {
    /// Creates a new in-memory database.
    #[must_use]
    pub fn new_in_memory() -> Self {
        Self::with_config(Config::in_memory())
    }

    /// Opens or creates a database at the given path.
    ///
    /// # Errors
    ///
    /// Returns an error if the database cannot be opened or created.
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        Ok(Self::with_config(Config::persistent(path.as_ref())))
    }

    /// Creates a database with the given configuration.
    #[must_use]
    pub fn with_config(config: Config) -> Self {
        let store = Arc::new(LpgStore::new());
        let tx_manager = Arc::new(TransactionManager::new());

        Self {
            config,
            store,
            tx_manager,
            is_open: RwLock::new(true),
        }
    }

    /// Creates a new session for interacting with the database.
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

    /// Returns the underlying store (for internal use).
    #[must_use]
    pub(crate) fn store(&self) -> &Arc<LpgStore> {
        &self.store
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
        value.as_int64().ok_or_else(|| {
            graphos_common::utils::error::Error::TypeMismatch {
                expected: "INT64".to_string(),
                found: value.type_name().to_string(),
            }
        })
    }
}

impl FromValue for f64 {
    fn from_value(value: &graphos_common::types::Value) -> Result<Self> {
        value.as_float64().ok_or_else(|| {
            graphos_common::utils::error::Error::TypeMismatch {
                expected: "FLOAT64".to_string(),
                found: value.type_name().to_string(),
            }
        })
    }
}

impl FromValue for String {
    fn from_value(value: &graphos_common::types::Value) -> Result<Self> {
        value
            .as_str()
            .map(String::from)
            .ok_or_else(|| graphos_common::utils::error::Error::TypeMismatch {
                expected: "STRING".to_string(),
                found: value.type_name().to_string(),
            })
    }
}

impl FromValue for bool {
    fn from_value(value: &graphos_common::types::Value) -> Result<Self> {
        value.as_bool().ok_or_else(|| {
            graphos_common::utils::error::Error::TypeMismatch {
                expected: "BOOL".to_string(),
                found: value.type_name().to_string(),
            }
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
        let config = Config::in_memory()
            .with_threads(4)
            .with_query_logging();

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
