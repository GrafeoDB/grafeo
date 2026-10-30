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
/// `tx[Symbol.dispose]()` rolls back a transaction that is still active, so
/// where `using` is supported (Node.js 24+, or TypeScript compiling it down)
/// it cleans up on its own:
/// ```js
/// using tx = db.beginTransaction();
/// await tx.execute("INSERT (:Person {name: 'Alix'})");
/// tx.commit();
/// // auto-rollback if commit not called
/// ```
///
/// A commit that fails (a write conflict, for example) ends the transaction
/// all the same: `isActive` is `false` after it and `rollback()` does
/// nothing, so a `catch` that rolls back sees the commit's error.
#[napi]
pub struct Transaction {
    db: Arc<RwLock<GrafeoDB>>,
    /// Shared with `spawn_blocking` tasks so queries run off the JS thread.
    /// Set to `None` once the transaction is committed or rolled back, so a
    /// query still queued on a worker finds it closed instead of running
    /// outside the transaction.
    session: Arc<parking_lot::Mutex<Option<grafeo_engine::session::Session>>>,
    /// One of `ACTIVE`, `COMMITTED`, `ROLLED_BACK`, `COMMIT_FAILED`. Atomic
    /// because async execute futures read it off the JS thread.
    state: AtomicU8,
}

const ACTIVE: u8 = 0;
const COMMITTED: u8 = 1;
const ROLLED_BACK: u8 = 2;
/// A commit was tried and failed: the transaction ended without its changes.
const COMMIT_FAILED: u8 = 3;

#[napi]
impl Transaction {
    /// Execute a GQL query within this transaction.
    #[napi]
    pub async fn execute(
        &self,
        query: String,
        #[napi(ts_arg_type = "any | undefined | null")] params: Option<crate::types::Params>,
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
    /// first. After a commit that failed it does nothing: that commit ended
    /// the transaction.
    #[napi]
    pub fn rollback(&self) -> Result<()> {
        if self.state.load(Ordering::Acquire) == COMMIT_FAILED {
            return Ok(());
        }
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
                return Err(NodeGrafeoError::transaction_state("Already committed").into());
            }
            ROLLED_BACK => {
                return Err(NodeGrafeoError::transaction_state("Already rolled back").into());
            }
            COMMIT_FAILED => {
                return Err(NodeGrafeoError::transaction_state(
                    "Transaction is no longer active: its commit failed",
                )
                .into());
            }
            _ => {}
        }
        // try_lock: the lock is held only while a query runs on a worker,
        // and blocking here would stall the whole event loop until it ends.
        let mut session_guard = self.session.try_lock().ok_or_else(|| {
            napi::Error::from(NodeGrafeoError::transaction_state(
                "A query is still running in this transaction; await it before calling commit() or rollback()",
            ))
        })?;
        let outcome = match session_guard.as_mut() {
            Some(session) if target == COMMITTED => {
                let outcome = session.commit();
                // A commit refused before it began leaves the transaction
                // open: end it, as a failed commit does.
                if outcome.is_err() && session.in_transaction() {
                    let _ = session.rollback();
                }
                outcome
            }
            Some(session) => session.rollback(),
            None => Ok(()),
        };
        // A commit ends the transaction whether it succeeds or fails.
        if outcome.is_ok() || target == COMMITTED {
            *session_guard = None;
            let state = if outcome.is_ok() {
                target
            } else {
                COMMIT_FAILED
            };
            self.state.store(state, Ordering::Release);
        }
        outcome.map_err(|error| NodeGrafeoError::from(error).into())
    }

    /// Shared implementation for all language-specific execute methods.
    ///
    /// The query runs on a blocking worker thread (like `Database.execute`)
    /// so a long transaction query does not stall the Node.js event loop.
    async fn execute_language_impl(
        &self,
        language: &'static str,
        query: String,
        params: Option<crate::types::Params>,
    ) -> Result<QueryResult> {
        if self.state.load(Ordering::Acquire) != ACTIVE {
            return Err(
                NodeGrafeoError::transaction_state("Transaction is no longer active").into(),
            );
        }
        let param_map = params.map(crate::types::Params::into_map).transpose()?;

        let session = Arc::clone(&self.session);
        let mut result = tokio::task::spawn_blocking(move || -> Result<_> {
            let session_guard = session.lock();
            let session = session_guard.as_ref().ok_or_else(|| {
                napi::Error::from(NodeGrafeoError::transaction_state(
                    "Transaction is no longer active",
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
        let counters = result.counters;
        let profile = result.profile.take();

        Ok(QueryResult::with_metrics(
            columns,
            result.into_rows(),
            nodes,
            edges,
            exec_time,
            scanned,
        )
        .with_counters(counters)
        .with_profile(profile))
    }

    pub(crate) fn new(db: Arc<RwLock<GrafeoDB>>, isolation_level: Option<&str>) -> Result<Self> {
        let session = db.read().session();
        Self::begin(db, session, isolation_level)
    }

    /// Begins a transaction in `session`, which works in the graph it selects.
    pub(crate) fn begin(
        db: Arc<RwLock<GrafeoDB>>,
        mut session: grafeo_engine::Session,
        isolation_level: Option<&str>,
    ) -> Result<Self> {
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

        // Begin the transaction with the specified isolation level
        if let Some(level) = level {
            session
                .begin_transaction_with_isolation(level)
                .map_err(NodeGrafeoError::from)?;
        } else {
            session.begin_transaction().map_err(NodeGrafeoError::from)?;
        }
        // It ends with commit() or rollback(), never with a statement.
        session.refuse_transaction_statements();

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
        #[napi(ts_arg_type = "any | undefined | null")] params: Option<crate::types::Params>,
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
        #[napi(ts_arg_type = "any | undefined | null")] params: Option<crate::types::Params>,
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
        #[napi(ts_arg_type = "any | undefined | null")] params: Option<crate::types::Params>,
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
        #[napi(ts_arg_type = "any | undefined | null")] params: Option<crate::types::Params>,
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
        #[napi(ts_arg_type = "any | undefined | null")] params: Option<crate::types::Params>,
    ) -> Result<QueryResult> {
        self.execute_language_impl("sparql", query, params).await
    }
}
