//! Transaction support for the Node.js API.

use std::sync::Arc;

use napi::bindgen_prelude::*;
use napi_derive::napi;
use parking_lot::RwLock;

use grafeo_engine::database::GrafeoDB;

use crate::error::NodeGrafeoError;
use crate::query::QueryResult;

/// A database transaction with explicit commit/rollback.
///
/// In Node.js 22+, use with `using` for automatic cleanup:
/// ```js
/// using tx = db.beginTransaction();
/// await tx.execute("INSERT (:Person {name: 'Alix'})");
/// tx.commit();
/// // auto-rollback if commit not called
/// ```
#[napi]
pub struct Transaction {
    db: Arc<RwLock<GrafeoDB>>,
    /// Shared with `spawn_blocking` tasks so queries run off the JS thread.
    session: Arc<parking_lot::Mutex<Option<grafeo_engine::session::Session>>>,
    committed: bool,
    rolled_back: bool,
}

#[napi]
impl Transaction {
    /// Execute a GQL query within this transaction.
    #[napi]
    pub async fn execute(
        &self,
        query: String,
        params: Option<serde_json::Value>,
    ) -> Result<QueryResult> {
        self.execute_language_impl("gql", query, params).await
    }

    /// Commit the transaction.
    #[napi]
    pub fn commit(&mut self) -> Result<()> {
        if self.committed {
            return Err(NodeGrafeoError::Transaction("Already committed".into()).into());
        }
        if self.rolled_back {
            return Err(NodeGrafeoError::Transaction("Already rolled back".into()).into());
        }
        let mut session_guard = self.session.lock();
        if let Some(ref mut session) = *session_guard {
            session.commit().map_err(NodeGrafeoError::from)?;
        }
        self.committed = true;
        Ok(())
    }

    /// Roll back the transaction.
    #[napi]
    pub fn rollback(&mut self) -> Result<()> {
        if self.committed {
            return Err(NodeGrafeoError::Transaction("Already committed".into()).into());
        }
        if self.rolled_back {
            return Err(NodeGrafeoError::Transaction("Already rolled back".into()).into());
        }
        let mut session_guard = self.session.lock();
        if let Some(ref mut session) = *session_guard {
            session.rollback().map_err(NodeGrafeoError::from)?;
        }
        self.rolled_back = true;
        Ok(())
    }

    /// Whether the transaction is still active.
    #[napi(getter, js_name = "isActive")]
    pub fn is_active(&self) -> bool {
        !self.committed && !self.rolled_back
    }
}

impl Transaction {
    /// Shared implementation for all language-specific execute methods.
    ///
    /// The query runs on a blocking worker thread (like `Database.execute`)
    /// so a long transaction query does not stall the Node.js event loop.
    async fn execute_language_impl(
        &self,
        language: &'static str,
        query: String,
        params: Option<serde_json::Value>,
    ) -> Result<QueryResult> {
        if self.committed || self.rolled_back {
            return Err(
                NodeGrafeoError::Transaction("Transaction is no longer active".into()).into(),
            );
        }
        let param_map = grafeo_bindings_common::json::json_params_to_map(params.as_ref())
            .map_err(|msg| napi::Error::from(NodeGrafeoError::InvalidArgument(msg)))?;

        let session = Arc::clone(&self.session);
        let mut result = tokio::task::spawn_blocking(move || -> Result<_> {
            let session_guard = session.lock();
            let session = session_guard.as_ref().ok_or_else(|| {
                napi::Error::from(NodeGrafeoError::Transaction(
                    "Transaction is no longer active".into(),
                ))
            })?;
            Ok(session
                .execute_language(&query, language, param_map)
                .map_err(NodeGrafeoError::from)?)
        })
        .await
        .map_err(|e| napi::Error::from_reason(e.to_string()))??;

        let db = self.db.read();
        let (nodes, edges) = crate::database::extract_entities(&result, &db);
        let columns = std::mem::take(&mut result.columns);
        let exec_time = result.execution_time_ms;
        let scanned = result.rows_scanned;

        Ok(QueryResult::with_metrics(
            columns,
            result.into_rows(),
            nodes,
            edges,
            exec_time,
            scanned,
        ))
    }

    pub(crate) fn new(db: Arc<RwLock<GrafeoDB>>, isolation_level: Option<&str>) -> Result<Self> {
        // Parse isolation level string
        let level = match isolation_level {
            Some("read_committed") => {
                Some(grafeo_engine::transaction::IsolationLevel::ReadCommitted)
            }
            Some("serializable") => Some(grafeo_engine::transaction::IsolationLevel::Serializable),
            Some("snapshot") | None => None, // snapshot is the default
            Some(other) => {
                return Err(NodeGrafeoError::InvalidArgument(format!(
                    "Unknown isolation level '{}'. Use 'read_committed', 'snapshot', or 'serializable'",
                    other
                ))
                .into());
            }
        };

        let mut session = {
            let db_guard = db.read();
            db_guard.session()
        };

        // Begin the transaction with the specified isolation level
        if let Some(level) = level {
            session
                .begin_transaction_with_isolation(level)
                .map_err(NodeGrafeoError::from)?;
        } else {
            session.begin_transaction().map_err(NodeGrafeoError::from)?;
        }

        Ok(Self {
            db,
            session: Arc::new(parking_lot::Mutex::new(Some(session))),
            committed: false,
            rolled_back: false,
        })
    }
}

impl Drop for Transaction {
    fn drop(&mut self) {
        // Auto-rollback on drop if not explicitly committed or rolled back
        if !self.committed && !self.rolled_back {
            let mut session_guard = self.session.lock();
            if let Some(ref mut session) = *session_guard {
                let _ = session.rollback();
            }
        }
    }
}

// Language-specific execute methods in separate impl blocks so `#[napi]`
// only generates C callback symbols when the feature is active.

#[cfg(feature = "cypher")]
#[napi]
impl Transaction {
    /// Execute a Cypher query within this transaction.
    #[napi(js_name = "executeCypher")]
    pub async fn execute_cypher(
        &self,
        query: String,
        params: Option<serde_json::Value>,
    ) -> Result<QueryResult> {
        self.execute_language_impl("cypher", query, params).await
    }
}

#[cfg(feature = "sql-pgq")]
#[napi]
impl Transaction {
    /// Execute a SQL/PGQ query (SQL:2023 GRAPH_TABLE) within this transaction.
    #[napi(js_name = "executeSql")]
    pub async fn execute_sql(
        &self,
        query: String,
        params: Option<serde_json::Value>,
    ) -> Result<QueryResult> {
        self.execute_language_impl("sql", query, params).await
    }
}

#[cfg(feature = "gremlin")]
#[napi]
impl Transaction {
    /// Execute a Gremlin query within this transaction.
    #[napi(js_name = "executeGremlin")]
    pub async fn execute_gremlin(
        &self,
        query: String,
        params: Option<serde_json::Value>,
    ) -> Result<QueryResult> {
        self.execute_language_impl("gremlin", query, params).await
    }
}

#[cfg(feature = "graphql")]
#[napi]
impl Transaction {
    /// Execute a GraphQL query within this transaction.
    #[napi(js_name = "executeGraphql")]
    pub async fn execute_graphql(
        &self,
        query: String,
        params: Option<serde_json::Value>,
    ) -> Result<QueryResult> {
        self.execute_language_impl("graphql", query, params).await
    }
}

#[cfg(feature = "sparql")]
#[napi]
impl Transaction {
    /// Execute a SPARQL query within this transaction.
    #[napi(js_name = "executeSparql")]
    pub async fn execute_sparql(
        &self,
        query: String,
        params: Option<serde_json::Value>,
    ) -> Result<QueryResult> {
        self.execute_language_impl("sparql", query, params).await
    }
}
