//! Transaction support for the Node.js API.

use std::sync::Arc;
use std::sync::atomic::{AtomicU8, Ordering};

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
    /// Set to `None` once the transaction is committed or rolled back, so a
    /// query still queued on a worker finds it closed instead of running
    /// outside the transaction.
    session: Arc<parking_lot::Mutex<Option<grafeo_engine::session::Session>>>,
    /// One of `ACTIVE`, `COMMITTED`, `ROLLED_BACK`. Atomic because async
    /// execute futures read it off the JS thread.
    state: AtomicU8,
}

const ACTIVE: u8 = 0;
const COMMITTED: u8 = 1;
const ROLLED_BACK: u8 = 2;

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
    ///
    /// Throws if a query from this transaction is still running: await it
    /// first. Closing never waits on a running query, so it cannot stall the
    /// event loop.
    #[napi]
    pub fn commit(&self) -> Result<()> {
        self.finish(COMMITTED)
    }

    /// Roll back the transaction.
    ///
    /// Throws if a query from this transaction is still running: await it
    /// first.
    #[napi]
    pub fn rollback(&self) -> Result<()> {
        self.finish(ROLLED_BACK)
    }

    /// Whether the transaction is still active.
    #[napi(getter, js_name = "isActive")]
    pub fn is_active(&self) -> bool {
        self.state.load(Ordering::Acquire) == ACTIVE
    }
}

impl Transaction {
    /// Commits or rolls back (`target` is `COMMITTED` or `ROLLED_BACK`).
    fn finish(&self, target: u8) -> Result<()> {
        match self.state.load(Ordering::Acquire) {
            COMMITTED => {
                return Err(NodeGrafeoError::Transaction("Already committed".into()).into());
            }
            ROLLED_BACK => {
                return Err(NodeGrafeoError::Transaction("Already rolled back".into()).into());
            }
            _ => {}
        }
        // try_lock: the lock is held only while a query runs on a worker,
        // and blocking here would stall the whole event loop until it ends.
        let mut session_guard = self.session.try_lock().ok_or_else(|| {
            napi::Error::from(NodeGrafeoError::Transaction(
                "A query is still running in this transaction; await it before calling commit() or rollback()".into(),
            ))
        })?;
        if let Some(session) = session_guard.as_mut() {
            if target == COMMITTED {
                session.commit().map_err(NodeGrafeoError::from)?;
            } else {
                session.rollback().map_err(NodeGrafeoError::from)?;
            }
        }
        *session_guard = None;
        self.state.store(target, Ordering::Release);
        Ok(())
    }

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
        if self.state.load(Ordering::Acquire) != ACTIVE {
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
            state: AtomicU8::new(ACTIVE),
        })
    }
}

impl Drop for Transaction {
    fn drop(&mut self) {
        // Auto-rollback on drop if not explicitly committed or rolled back.
        // If a query still holds the lock, don't block the JS thread (drop
        // runs from GC finalizers): the worker task owns the other `Arc`, and
        // `Session`'s own `Drop` rolls back once that query finishes.
        if self.state.load(Ordering::Acquire) == ACTIVE
            && let Some(mut session_guard) = self.session.try_lock()
            && let Some(session) = session_guard.as_mut()
        {
            let _ = session.rollback();
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
