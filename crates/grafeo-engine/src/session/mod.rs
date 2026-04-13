//! Lightweight handles for database interaction.
//!
//! A session is your conversation with the database. Each session can have
//! its own transaction state, so concurrent sessions don't interfere with
//! each other. Sessions are cheap to create - spin up as many as you need.

#[cfg(feature = "triple-store")]
mod rdf;

use std::sync::Arc;
use std::sync::atomic::AtomicUsize;
#[cfg(feature = "lpg")]
use std::sync::atomic::Ordering;
use std::time::{Duration, Instant};

#[cfg(feature = "lpg")]
use grafeo_common::grafeo_debug_span;
#[cfg(feature = "lpg")]
use grafeo_common::types::{EdgeId, NodeId};
use grafeo_common::types::{EpochId, TransactionId, Value};
use grafeo_common::utils::error::Result;
use grafeo_common::{grafeo_info_span, grafeo_warn};
#[cfg(feature = "lpg")]
use grafeo_core::graph::Direction;
#[cfg(feature = "lpg")]
use grafeo_core::graph::lpg::LpgStore;
#[cfg(feature = "lpg")]
use grafeo_core::graph::lpg::{Edge, Node};
#[cfg(feature = "triple-store")]
use grafeo_core::graph::rdf::RdfStore;
use grafeo_core::graph::{GraphStore, GraphStoreMut};

use crate::catalog::{Catalog, CatalogConstraintValidator};
use crate::config::{AdaptiveConfig, GraphModel};
use crate::database::QueryResult;
use crate::query::cache::QueryCache;
use crate::transaction::TransactionManager;

/// Storage key suffix for the implicit default graph within a schema.
/// Auto-created by `CREATE SCHEMA` and auto-dropped by `DROP SCHEMA`.
const SCHEMA_DEFAULT_GRAPH: &str = "__default__";

/// Parses a DDL default-value literal string into a [`Value`].
///
/// Handles string literals (single- or double-quoted), integers, floats,
/// booleans (`true`/`false`), and `NULL`.
fn parse_default_literal(text: &str) -> Value {
    if text.eq_ignore_ascii_case("null") {
        return Value::Null;
    }
    if text.eq_ignore_ascii_case("true") {
        return Value::Bool(true);
    }
    if text.eq_ignore_ascii_case("false") {
        return Value::Bool(false);
    }
    // String literal: strip surrounding quotes
    if (text.starts_with('\'') && text.ends_with('\''))
        || (text.starts_with('"') && text.ends_with('"'))
    {
        return Value::String(text[1..text.len() - 1].into());
    }
    // Try integer, then float
    if let Ok(i) = text.parse::<i64>() {
        return Value::Int64(i);
    }
    if let Ok(f) = text.parse::<f64>() {
        return Value::Float64(f);
    }
    // Fallback: treat as string
    Value::String(text.into())
}

/// Runtime configuration for creating a new session.
///
/// Groups the shared parameters passed to all session constructors, keeping
/// call sites readable and avoiding long argument lists.
pub(crate) struct SessionConfig {
    pub transaction_manager: Arc<TransactionManager>,
    pub query_cache: Arc<QueryCache>,
    pub catalog: Arc<Catalog>,
    pub adaptive_config: AdaptiveConfig,
    pub factorized_execution: bool,
    pub graph_model: GraphModel,
    pub query_timeout: Option<Duration>,
    pub commit_counter: Arc<AtomicUsize>,
    pub gc_interval: usize,
    /// When true, the session permanently blocks all mutations.
    pub read_only: bool,
    /// The identity bound to this session (for permission checks).
    pub identity: crate::auth::Identity,
    /// Named graph projections shared with the database.
    #[cfg(feature = "lpg")]
    pub projections: Arc<
        parking_lot::RwLock<
            std::collections::HashMap<String, Arc<grafeo_core::graph::GraphProjection>>,
        >,
    >,
    /// For hybrid stores: the overlay `LpgStore` that should receive MVCC
    /// finalization on commit (finalize_version_epochs, sync_epoch).
    ///
    /// When `Some`, `resolve_store` returns this store instead of a fresh
    /// empty one, ensuring that mutations committed via an external write
    /// store (HybridStore) become visible to subsequent read-only sessions.
    #[cfg(all(feature = "lpg", feature = "compact-store"))]
    pub hybrid_overlay: Option<Arc<LpgStore>>,
}

/// Your handle to the database - execute queries and manage transactions.
///
/// Get one from [`GrafeoDB::session()`](crate::GrafeoDB::session). Each session
/// tracks its own transaction state, so you can have multiple concurrent
/// sessions without them interfering.
pub struct Session {
    /// The underlying store.
    #[cfg(feature = "lpg")]
    store: Arc<LpgStore>,
    /// Graph store trait object for pluggable storage backends (read path).
    graph_store: Arc<dyn GraphStore>,
    /// Writable graph store (None for read-only databases).
    graph_store_mut: Option<Arc<dyn GraphStoreMut>>,
    /// Schema and metadata catalog shared across sessions.
    catalog: Arc<Catalog>,
    /// RDF triple store (if RDF feature is enabled).
    #[cfg(feature = "triple-store")]
    rdf_store: Arc<RdfStore>,
    /// Transaction manager.
    transaction_manager: Arc<TransactionManager>,
    /// Query cache shared across sessions.
    query_cache: Arc<QueryCache>,
    /// Current transaction ID (if any). Behind a Mutex so that GQL commands
    /// (`START TRANSACTION`, `COMMIT`, `ROLLBACK`) can manage transactions
    /// from within `execute(&self)`.
    current_transaction: parking_lot::Mutex<Option<TransactionId>>,
    /// Whether the current transaction is read-only (blocks mutations).
    read_only_tx: parking_lot::Mutex<bool>,
    /// Whether the database itself is read-only (set at open time, never changes).
    /// When true, `read_only_tx` is always true regardless of transaction flags.
    db_read_only: bool,
    /// The identity bound to this session (determines permission level).
    identity: crate::auth::Identity,
    /// Whether the session is in auto-commit mode.
    auto_commit: bool,
    /// Adaptive execution configuration.
    #[allow(dead_code)] // Stored for future adaptive re-optimization during execution
    adaptive_config: AdaptiveConfig,
    /// Whether to use factorized execution for multi-hop queries.
    factorized_execution: bool,
    /// The graph data model this session operates on.
    graph_model: GraphModel,
    /// Maximum time a query may run before being cancelled.
    query_timeout: Option<Duration>,
    /// Shared commit counter for triggering auto-GC.
    commit_counter: Arc<AtomicUsize>,
    /// GC every N commits (0 = disabled).
    gc_interval: usize,
    /// Node count at the start of the current transaction (for PreparedCommit stats).
    transaction_start_node_count: AtomicUsize,
    /// Edge count at the start of the current transaction (for PreparedCommit stats).
    transaction_start_edge_count: AtomicUsize,
    /// WAL for logging schema changes.
    #[cfg(feature = "wal")]
    wal: Option<Arc<grafeo_storage::wal::LpgWal>>,
    /// Shared WAL graph context tracker for named graph awareness.
    #[cfg(feature = "wal")]
    wal_graph_context: Option<Arc<parking_lot::Mutex<Option<String>>>>,
    /// CDC log for change tracking.
    #[cfg(feature = "cdc")]
    cdc_log: Arc<crate::cdc::CdcLog>,
    /// Buffered CDC events for the current transaction.
    /// Flushed to `cdc_log` on commit, discarded on rollback.
    #[cfg(feature = "cdc")]
    cdc_pending_events: Option<Arc<parking_lot::Mutex<Vec<crate::cdc::ChangeEvent>>>>,
    /// Current graph name (for multi-graph USE GRAPH support). None = default graph.
    current_graph: parking_lot::Mutex<Option<String>>,
    /// Current schema name (ISO/IEC 39075 Section 4.7.3: independent from session graph).
    /// None = "not set" (uses default schema).
    current_schema: parking_lot::Mutex<Option<String>>,
    /// Session time zone override.
    time_zone: parking_lot::Mutex<Option<String>>,
    /// Session-level parameters (SET PARAMETER).
    session_params:
        parking_lot::Mutex<std::collections::HashMap<String, grafeo_common::types::Value>>,
    /// Override epoch for time-travel queries (None = use transaction/current epoch).
    viewing_epoch_override: parking_lot::Mutex<Option<EpochId>>,
    /// Savepoints within the current transaction.
    savepoints: parking_lot::Mutex<Vec<SavepointState>>,
    /// Nesting depth for nested transactions (0 = outermost).
    /// Nested `START TRANSACTION` creates an auto-savepoint; nested `COMMIT`
    /// releases it, nested `ROLLBACK` rolls back to it.
    transaction_nesting_depth: parking_lot::Mutex<u32>,
    /// Named graphs touched during the current transaction (for cross-graph atomicity).
    /// `None` represents the default graph. Populated at `BEGIN` time and on each
    /// `USE GRAPH` / `SESSION SET GRAPH` switch within a transaction.
    touched_graphs: parking_lot::Mutex<Vec<Option<String>>>,
    /// Shared metrics registry (populated when the `metrics` feature is enabled).
    #[cfg(feature = "metrics")]
    pub(crate) metrics: Option<Arc<crate::metrics::MetricsRegistry>>,
    /// Transaction start time for duration tracking.
    #[cfg(feature = "metrics")]
    tx_start_time: parking_lot::Mutex<Option<Instant>>,
    /// Named graph projections shared with the database.
    #[cfg(feature = "lpg")]
    projections: Arc<
        parking_lot::RwLock<
            std::collections::HashMap<String, Arc<grafeo_core::graph::GraphProjection>>,
        >,
    >,
}

/// Per-graph savepoint snapshot, capturing the store state at the time of the savepoint.
#[derive(Clone)]
struct GraphSavepoint {
    graph_name: Option<String>,
    next_node_id: u64,
    next_edge_id: u64,
    undo_log_position: usize,
}

/// Savepoint state: name + per-graph snapshots + the graph that was active.
#[derive(Clone)]
struct SavepointState {
    name: String,
    graph_snapshots: Vec<GraphSavepoint>,
    /// The graph that was active when the savepoint was created.
    /// Reserved for future use (e.g., restoring graph context on rollback).
    #[allow(dead_code)]
    active_graph: Option<String>,
    /// CDC event buffer position at savepoint creation.
    /// On rollback-to-savepoint, the buffer is truncated to this position.
    #[cfg(feature = "cdc")]
    cdc_event_position: usize,
}

impl Session {
    /// Creates a new session with adaptive execution configuration.
    #[cfg(feature = "lpg")]
    #[allow(dead_code)] // Used when lpg enabled without triple-store
    pub(crate) fn with_adaptive(store: Arc<LpgStore>, cfg: SessionConfig) -> Self {
        let graph_store = Arc::clone(&store) as Arc<dyn GraphStore>;
        let graph_store_mut = Some(Arc::clone(&store) as Arc<dyn GraphStoreMut>);
        Self {
            store,
            graph_store,
            graph_store_mut,
            catalog: cfg.catalog,
            #[cfg(feature = "triple-store")]
            rdf_store: Arc::new(RdfStore::new()),
            transaction_manager: cfg.transaction_manager,
            query_cache: cfg.query_cache,
            current_transaction: parking_lot::Mutex::new(None),
            read_only_tx: parking_lot::Mutex::new(cfg.read_only),
            db_read_only: cfg.read_only,
            identity: cfg.identity,
            auto_commit: true,
            adaptive_config: cfg.adaptive_config,
            factorized_execution: cfg.factorized_execution,
            graph_model: cfg.graph_model,
            query_timeout: cfg.query_timeout,
            commit_counter: cfg.commit_counter,
            gc_interval: cfg.gc_interval,
            transaction_start_node_count: AtomicUsize::new(0),
            transaction_start_edge_count: AtomicUsize::new(0),
            #[cfg(feature = "wal")]
            wal: None,
            #[cfg(feature = "wal")]
            wal_graph_context: None,
            #[cfg(feature = "cdc")]
            cdc_log: Arc::new(crate::cdc::CdcLog::new()),
            #[cfg(feature = "cdc")]
            cdc_pending_events: None,
            current_graph: parking_lot::Mutex::new(None),
            current_schema: parking_lot::Mutex::new(None),
            time_zone: parking_lot::Mutex::new(None),
            session_params: parking_lot::Mutex::new(std::collections::HashMap::new()),
            viewing_epoch_override: parking_lot::Mutex::new(None),
            savepoints: parking_lot::Mutex::new(Vec::new()),
            transaction_nesting_depth: parking_lot::Mutex::new(0),
            touched_graphs: parking_lot::Mutex::new(Vec::new()),
            #[cfg(feature = "metrics")]
            metrics: None,
            #[cfg(feature = "metrics")]
            tx_start_time: parking_lot::Mutex::new(None),
            projections: cfg.projections,
        }
    }

    /// Sets the WAL for this session (shared with the database).
    ///
    /// This also wraps `graph_store` in a [`WalGraphStore`] so that mutation
    /// operators (INSERT, DELETE, SET via queries) log to the WAL.
    #[cfg(all(feature = "wal", feature = "lpg"))]
    pub(crate) fn set_wal(
        &mut self,
        wal: Arc<grafeo_storage::wal::LpgWal>,
        wal_graph_context: Arc<parking_lot::Mutex<Option<String>>>,
    ) {
        // Wrap the graph store so query-engine mutations are WAL-logged
        let wal_store = Arc::new(crate::database::wal_store::WalGraphStore::new(
            Arc::clone(&self.store),
            Arc::clone(&wal),
            Arc::clone(&wal_graph_context),
        ));
        self.graph_store = Arc::clone(&wal_store) as Arc<dyn GraphStore>;
        self.graph_store_mut = Some(wal_store as Arc<dyn GraphStoreMut>);
        self.wal = Some(wal);
        self.wal_graph_context = Some(wal_graph_context);
    }

    /// Sets the CDC log for this session (shared with the database).
    ///
    /// Wraps the current write store with a `CdcGraphStore` decorator so
    /// that all session mutations (INSERT, SET, DELETE via query execution)
    /// buffer CDC events. The buffer is flushed to the `CdcLog` on commit
    /// and discarded on rollback.
    #[cfg(feature = "cdc")]
    pub(crate) fn set_cdc_log(&mut self, cdc_log: Arc<crate::cdc::CdcLog>) {
        // Wrap the WRITE store only with CdcGraphStore to intercept mutations.
        // The read store (self.graph_store) is left unchanged for zero read overhead.
        if let Some(ref write_store) = self.graph_store_mut {
            let cdc_store = Arc::new(crate::database::cdc_store::CdcGraphStore::new(
                Arc::clone(write_store),
                Arc::clone(&cdc_log),
            ));
            self.cdc_pending_events = Some(cdc_store.pending_events());
            self.graph_store_mut = Some(cdc_store as Arc<dyn grafeo_core::graph::GraphStoreMut>);
        }
        self.cdc_log = cdc_log;
    }

    /// Sets the metrics registry for this session (shared with the database).
    #[cfg(feature = "metrics")]
    pub(crate) fn set_metrics(&mut self, metrics: Arc<crate::metrics::MetricsRegistry>) {
        self.metrics = Some(metrics);
    }

    /// Creates a session backed by an external graph store.
    ///
    /// The external store handles all data operations. Transaction management
    /// (begin/commit/rollback) is not supported for external stores.
    ///
    /// # Errors
    ///
    /// Returns an error if the internal arena allocation fails (out of memory).
    pub(crate) fn with_external_store(
        read_store: Arc<dyn GraphStore>,
        write_store: Option<Arc<dyn GraphStoreMut>>,
        cfg: SessionConfig,
    ) -> Result<Self> {
        #[cfg(all(feature = "lpg", feature = "compact-store"))]
        let store = if let Some(overlay) = cfg.hybrid_overlay {
            overlay
        } else {
            Arc::new(LpgStore::new()?)
        };
        #[cfg(all(feature = "lpg", not(feature = "compact-store")))]
        let store = Arc::new(LpgStore::new()?);
        Ok(Self {
            #[cfg(feature = "lpg")]
            store,
            graph_store: read_store,
            graph_store_mut: write_store,
            catalog: cfg.catalog,
            #[cfg(feature = "triple-store")]
            rdf_store: Arc::new(RdfStore::new()),
            transaction_manager: cfg.transaction_manager,
            query_cache: cfg.query_cache,
            current_transaction: parking_lot::Mutex::new(None),
            read_only_tx: parking_lot::Mutex::new(cfg.read_only),
            db_read_only: cfg.read_only,
            identity: cfg.identity,
            auto_commit: true,
            adaptive_config: cfg.adaptive_config,
            factorized_execution: cfg.factorized_execution,
            graph_model: cfg.graph_model,
            query_timeout: cfg.query_timeout,
            commit_counter: cfg.commit_counter,
            gc_interval: cfg.gc_interval,
            transaction_start_node_count: AtomicUsize::new(0),
            transaction_start_edge_count: AtomicUsize::new(0),
            #[cfg(feature = "wal")]
            wal: None,
            #[cfg(feature = "wal")]
            wal_graph_context: None,
            #[cfg(feature = "cdc")]
            cdc_log: Arc::new(crate::cdc::CdcLog::new()),
            #[cfg(feature = "cdc")]
            cdc_pending_events: None,
            current_graph: parking_lot::Mutex::new(None),
            current_schema: parking_lot::Mutex::new(None),
            time_zone: parking_lot::Mutex::new(None),
            session_params: parking_lot::Mutex::new(std::collections::HashMap::new()),
            viewing_epoch_override: parking_lot::Mutex::new(None),
            savepoints: parking_lot::Mutex::new(Vec::new()),
            transaction_nesting_depth: parking_lot::Mutex::new(0),
            touched_graphs: parking_lot::Mutex::new(Vec::new()),
            #[cfg(feature = "metrics")]
            metrics: None,
            #[cfg(feature = "metrics")]
            tx_start_time: parking_lot::Mutex::new(None),
            #[cfg(feature = "lpg")]
            projections: cfg.projections,
        })
    }

    /// Returns the graph model this session operates on.
    #[must_use]
    pub fn graph_model(&self) -> GraphModel {
        self.graph_model
    }

    /// Returns the identity bound to this session.
    #[must_use]
    pub fn identity(&self) -> &crate::auth::Identity {
        &self.identity
    }

    // === Session State Management ===

    /// Sets the current graph for this session (USE GRAPH).
    pub fn use_graph(&self, name: &str) {
        *self.current_graph.lock() = Some(name.to_string());
    }

    /// Returns the current graph name, if any.
    #[must_use]
    pub fn current_graph(&self) -> Option<String> {
        self.current_graph.lock().clone()
    }

    /// Sets the current schema for this session (SESSION SET SCHEMA).
    ///
    /// Per ISO/IEC 39075 Section 7.1 GR1, this is independent of the session graph.
    pub fn set_schema(&self, name: &str) {
        *self.current_schema.lock() = Some(name.to_string());
    }

    /// Returns the current schema name, if any.
    ///
    /// `None` means "not set", which resolves to the default schema.
    #[must_use]
    pub fn current_schema(&self) -> Option<String> {
        self.current_schema.lock().clone()
    }

    /// Computes the effective storage key for a graph, accounting for schema context.
    ///
    /// Per ISO/IEC 39075 Section 17.2, graphs resolve relative to the current schema.
    /// Uses `/` as separator since it is invalid in GQL identifiers.
    fn effective_graph_key(&self, graph_name: &str) -> String {
        let schema = self.current_schema.lock().clone();
        match schema {
            Some(s) => format!("{s}/{graph_name}"),
            None => graph_name.to_string(),
        }
    }

    /// Computes the effective storage key for a type, accounting for schema context.
    ///
    /// Mirrors `effective_graph_key()`: types resolve relative to the current schema.
    fn effective_type_key(&self, type_name: &str) -> String {
        let schema = self.current_schema.lock().clone();
        match schema {
            Some(s) => format!("{s}/{type_name}"),
            None => type_name.to_string(),
        }
    }

    /// Returns the effective storage key for the current graph, accounting for schema.
    ///
    /// Combines `current_schema` and `current_graph` into a flat lookup key.
    fn active_graph_storage_key(&self) -> Option<String> {
        let graph = self.current_graph.lock().clone();
        let schema = self.current_schema.lock().clone();
        match (&schema, &graph) {
            (None, None) => None,
            (Some(s), None) => Some(format!("{s}/{SCHEMA_DEFAULT_GRAPH}")),
            (None, Some(name)) if name.eq_ignore_ascii_case("default") => None,
            (Some(s), Some(name)) if name.eq_ignore_ascii_case("default") => {
                Some(format!("{s}/{SCHEMA_DEFAULT_GRAPH}"))
            }
            (None, Some(name)) => Some(name.clone()),
            (Some(s), Some(g)) => Some(format!("{s}/{g}")),
        }
    }

    /// Returns the graph store for the currently active graph.
    ///
    /// If `current_graph` is `None` or `"default"`, returns the session's
    /// default `graph_store` (already WAL-wrapped for the default graph).
    /// Otherwise looks up the named graph in the root store and wraps it
    /// in a [`WalGraphStore`] so mutations are WAL-logged with the correct
    /// graph context.
    fn active_store(&self) -> Arc<dyn GraphStore> {
        let key = self.active_graph_storage_key();
        match key {
            None => Arc::clone(&self.graph_store),
            #[cfg(feature = "lpg")]
            Some(ref name) => match self.store.graph(name) {
                Some(named_store) => {
                    #[cfg(feature = "wal")]
                    if let (Some(wal), Some(ctx)) = (&self.wal, &self.wal_graph_context) {
                        return Arc::new(crate::database::wal_store::WalGraphStore::new_for_graph(
                            named_store,
                            Arc::clone(wal),
                            name.clone(),
                            Arc::clone(ctx),
                        )) as Arc<dyn GraphStore>;
                    }
                    named_store as Arc<dyn GraphStore>
                }
                None => Arc::clone(&self.graph_store),
            },
            #[cfg(not(feature = "lpg"))]
            Some(_) => Arc::clone(&self.graph_store),
        }
    }

    /// Returns the writable store for the active graph, if available.
    ///
    /// Returns `None` for read-only databases. For named graphs, wraps
    /// the store with WAL logging when durability is enabled.
    fn active_write_store(&self) -> Option<Arc<dyn GraphStoreMut>> {
        let key = self.active_graph_storage_key();
        match key {
            None => self.graph_store_mut.as_ref().map(Arc::clone),
            #[cfg(feature = "lpg")]
            Some(ref name) => match self.store.graph(name) {
                Some(named_store) => {
                    let mut store: Arc<dyn GraphStoreMut> = named_store;

                    #[cfg(feature = "wal")]
                    if let (Some(wal), Some(ctx)) = (&self.wal, &self.wal_graph_context) {
                        store = Arc::new(crate::database::wal_store::WalGraphStore::new_for_graph(
                            // WAL needs Arc<LpgStore>, get it fresh
                            self.store
                                .graph(name)
                                .unwrap_or_else(|| Arc::clone(&self.store)),
                            Arc::clone(wal),
                            name.clone(),
                            Arc::clone(ctx),
                        ));
                    }

                    #[cfg(feature = "cdc")]
                    if let Some(ref pending) = self.cdc_pending_events {
                        store = Arc::new(crate::database::cdc_store::CdcGraphStore::wrap(
                            store,
                            Arc::clone(&self.cdc_log),
                            Arc::clone(pending),
                        ));
                    }

                    Some(store)
                }
                None => self.graph_store_mut.as_ref().map(Arc::clone),
            },
            #[cfg(not(feature = "lpg"))]
            Some(_) => self.graph_store_mut.as_ref().map(Arc::clone),
        }
    }

    /// Returns the concrete `LpgStore` for the currently active graph.
    ///
    /// Used by direct CRUD methods that need the concrete store type
    /// for versioned operations.
    #[cfg(feature = "lpg")]
    fn active_lpg_store(&self) -> Arc<LpgStore> {
        let key = self.active_graph_storage_key();
        match key {
            None => Arc::clone(&self.store),
            Some(ref name) => self
                .store
                .graph(name)
                .unwrap_or_else(|| Arc::clone(&self.store)),
        }
    }

    /// Resolves a graph name to a concrete `LpgStore`.
    /// `None` and `"default"` resolve to the session's root store.
    #[cfg(feature = "lpg")]
    fn resolve_store(&self, graph_name: &Option<String>) -> Arc<LpgStore> {
        match graph_name {
            None => Arc::clone(&self.store),
            Some(name) if name.eq_ignore_ascii_case("default") => Arc::clone(&self.store),
            Some(name) => self
                .store
                .graph(name)
                .unwrap_or_else(|| Arc::clone(&self.store)),
        }
    }

    /// Records the current graph as "touched" if a transaction is active.
    ///
    /// Uses the full storage key (schema/graph) so that commit/rollback
    /// can resolve the correct store via `resolve_store`.
    fn track_graph_touch(&self) {
        if self.current_transaction.lock().is_some() {
            let key = self.active_graph_storage_key();
            let mut touched = self.touched_graphs.lock();
            if !touched.contains(&key) {
                touched.push(key);
            }
        }
    }

    /// Sets the session time zone.
    pub fn set_time_zone(&self, tz: &str) {
        *self.time_zone.lock() = Some(tz.to_string());
    }

    /// Returns the session time zone, if set.
    #[must_use]
    pub fn time_zone(&self) -> Option<String> {
        self.time_zone.lock().clone()
    }

    /// Sets a session parameter.
    pub fn set_parameter(&self, key: &str, value: grafeo_common::types::Value) {
        self.session_params.lock().insert(key.to_string(), value);
    }

    /// Gets a session parameter by cloning it.
    #[must_use]
    pub fn get_parameter(&self, key: &str) -> Option<grafeo_common::types::Value> {
        self.session_params.lock().get(key).cloned()
    }

    /// Resets all session state to defaults (ISO/IEC 39075 Section 7.2).
    pub fn reset_session(&self) {
        *self.current_schema.lock() = None;
        *self.current_graph.lock() = None;
        *self.time_zone.lock() = None;
        self.session_params.lock().clear();
        *self.viewing_epoch_override.lock() = None;
    }

    /// Resets only the session schema (Section 7.2 GR1).
    pub fn reset_schema(&self) {
        *self.current_schema.lock() = None;
    }

    /// Resets only the session graph (Section 7.2 GR2).
    pub fn reset_graph(&self) {
        *self.current_graph.lock() = None;
    }

    /// Resets only the session time zone (Section 7.2 GR3).
    pub fn reset_time_zone(&self) {
        *self.time_zone.lock() = None;
    }

    /// Resets only session parameters (Section 7.2 GR4).
    pub fn reset_parameters(&self) {
        self.session_params.lock().clear();
    }

    // --- Time-travel API ---

    /// Sets a viewing epoch override for time-travel queries.
    ///
    /// While set, all queries on this session see the database as it existed
    /// at the given epoch. Use [`clear_viewing_epoch`](Self::clear_viewing_epoch)
    /// to return to normal behavior.
    pub fn set_viewing_epoch(&self, epoch: EpochId) {
        *self.viewing_epoch_override.lock() = Some(epoch);
    }

    /// Clears the viewing epoch override, returning to normal behavior.
    pub fn clear_viewing_epoch(&self) {
        *self.viewing_epoch_override.lock() = None;
    }

    /// Returns the current viewing epoch override, if any.
    #[must_use]
    pub fn viewing_epoch(&self) -> Option<EpochId> {
        *self.viewing_epoch_override.lock()
    }

    /// Returns all versions of a node with their creation/deletion epochs.
    ///
    /// Properties and labels reflect the current state (not versioned per-epoch).
    #[cfg(feature = "lpg")]
    #[must_use]
    pub fn get_node_history(&self, id: NodeId) -> Vec<(EpochId, Option<EpochId>, Node)> {
        self.active_lpg_store().get_node_history(id)
    }

    /// Returns all versions of an edge with their creation/deletion epochs.
    ///
    /// Properties reflect the current state (not versioned per-epoch).
    #[cfg(feature = "lpg")]
    #[must_use]
    pub fn get_edge_history(&self, id: EdgeId) -> Vec<(EpochId, Option<EpochId>, Edge)> {
        self.active_lpg_store().get_edge_history(id)
    }

    /// Checks that the session's graph model supports LPG operations.
    fn require_lpg(&self, language: &str) -> Result<()> {
        if self.graph_model == GraphModel::Rdf {
            return Err(grafeo_common::utils::error::Error::Internal(format!(
                "This is an RDF database. {language} queries require an LPG database."
            )));
        }
        Ok(())
    }

    /// Checks that the session's identity is permitted to execute the given
    /// statement kind. Returns an error if the role is insufficient.
    ///
    /// Short-circuits for Admin identities (the common case for embedded use
    /// and benchmarks) to avoid any overhead on the hot path.
    #[inline]
    fn require_permission(&self, kind: crate::auth::StatementKind) -> Result<()> {
        // Fast path: Admin can do everything
        if self.identity.can_admin() {
            return Ok(());
        }
        crate::auth::check_permission(&self.identity, kind).map_err(|denied| {
            grafeo_common::utils::error::Error::Query(grafeo_common::utils::error::QueryError::new(
                grafeo_common::utils::error::QueryErrorKind::Semantic,
                denied.to_string(),
            ))
        })
    }

    /// Executes a session or transaction command, returning an empty result.
    #[cfg(feature = "gql")]
    fn execute_session_command(
        &self,
        cmd: grafeo_adapters::query::gql::ast::SessionCommand,
    ) -> Result<QueryResult> {
        use grafeo_adapters::query::gql::ast::SessionCommand;
        #[cfg(feature = "lpg")]
        use grafeo_adapters::query::gql::ast::TransactionIsolationLevel;
        use grafeo_common::utils::error::{Error, QueryError, QueryErrorKind};

        // Check role-based permission for graph management commands
        match &cmd {
            SessionCommand::CreateGraph { .. }
            | SessionCommand::DropGraph { .. }
            | SessionCommand::CreateProjection { .. }
            | SessionCommand::DropProjection { .. } => {
                self.require_permission(crate::auth::StatementKind::Write)?;
            }
            _ => {} // Session state + transaction control: always allowed
        }

        // Check per-graph grants for graph-scoped commands
        if self.identity.has_grants() {
            match &cmd {
                SessionCommand::CreateGraph { name, .. }
                | SessionCommand::DropGraph { name, .. } => {
                    if !self
                        .identity
                        .can_access_graph(name, crate::auth::Role::ReadWrite)
                    {
                        return Err(Error::Query(QueryError::new(
                            QueryErrorKind::Semantic,
                            format!(
                                "permission denied: no grant for graph '{name}' (user: {})",
                                self.identity.user_id()
                            ),
                        )));
                    }
                }
                _ => {}
            }
        }

        // Block DDL in read-only transactions (ISO/IEC 39075 Section 8)
        if *self.read_only_tx.lock() {
            match &cmd {
                SessionCommand::CreateGraph { .. }
                | SessionCommand::DropGraph { .. }
                | SessionCommand::CreateProjection { .. }
                | SessionCommand::DropProjection { .. } => {
                    return Err(Error::Transaction(
                        grafeo_common::utils::error::TransactionError::ReadOnly,
                    ));
                }
                _ => {} // Session state + transaction control allowed
            }
        }

        match cmd {
            #[cfg(feature = "lpg")]
            SessionCommand::CreateGraph {
                name,
                if_not_exists,
                typed,
                like_graph,
                copy_of,
                open: _,
            } => {
                // ISO/IEC 39075 Section 12.4: graphs are created within the current schema
                let storage_key = self.effective_graph_key(&name);

                // Validate source graph exists for LIKE / AS COPY OF
                if let Some(ref src) = like_graph {
                    let src_key = self.effective_graph_key(src);
                    if self.store.graph(&src_key).is_none() {
                        return Err(Error::Query(QueryError::new(
                            QueryErrorKind::Semantic,
                            format!("Source graph '{src}' does not exist"),
                        )));
                    }
                }
                if let Some(ref src) = copy_of {
                    let src_key = self.effective_graph_key(src);
                    if self.store.graph(&src_key).is_none() {
                        return Err(Error::Query(QueryError::new(
                            QueryErrorKind::Semantic,
                            format!("Source graph '{src}' does not exist"),
                        )));
                    }
                }

                let created = self
                    .store
                    .create_graph(&storage_key)
                    .map_err(|e| Error::Internal(e.to_string()))?;
                if !created && !if_not_exists {
                    return Err(Error::Query(QueryError::new(
                        QueryErrorKind::Semantic,
                        format!("Graph '{name}' already exists"),
                    )));
                }
                if created {
                    #[cfg(feature = "wal")]
                    self.log_schema_wal(&grafeo_storage::wal::WalRecord::CreateNamedGraph {
                        name: storage_key.clone(),
                    });
                }

                // AS COPY OF: copy data from source graph
                if let Some(ref src) = copy_of {
                    let src_key = self.effective_graph_key(src);
                    self.store
                        .copy_graph(Some(&src_key), Some(&storage_key))
                        .map_err(|e| Error::Internal(e.to_string()))?;
                }

                // Bind to graph type if specified.
                // If the parser produced a '/' in the name it is already a qualified
                // "schema/type" key; otherwise resolve against the current schema.
                if let Some(type_name) = typed
                    && let Err(e) = self.catalog.bind_graph_type(
                        &storage_key,
                        if type_name.contains('/') {
                            type_name.clone()
                        } else {
                            self.effective_type_key(&type_name)
                        },
                    )
                {
                    return Err(Error::Query(QueryError::new(
                        QueryErrorKind::Semantic,
                        e.to_string(),
                    )));
                }

                // LIKE: copy graph type binding from source
                if let Some(ref src) = like_graph {
                    let src_key = self.effective_graph_key(src);
                    if let Some(src_type) = self.catalog.get_graph_type_binding(&src_key) {
                        let _ = self.catalog.bind_graph_type(&storage_key, src_type);
                    }
                }

                Ok(QueryResult::empty())
            }
            #[cfg(feature = "lpg")]
            SessionCommand::DropGraph { name, if_exists } => {
                let storage_key = self.effective_graph_key(&name);
                let dropped = self.store.drop_graph(&storage_key);
                if !dropped && !if_exists {
                    return Err(Error::Query(QueryError::new(
                        QueryErrorKind::Semantic,
                        format!("Graph '{name}' does not exist"),
                    )));
                }
                if dropped {
                    #[cfg(feature = "wal")]
                    self.log_schema_wal(&grafeo_storage::wal::WalRecord::DropNamedGraph {
                        name: storage_key.clone(),
                    });
                    // If this session was using the dropped graph, reset to default
                    let mut current = self.current_graph.lock();
                    if current
                        .as_deref()
                        .is_some_and(|g| g.eq_ignore_ascii_case(&name))
                    {
                        *current = None;
                    }
                }
                Ok(QueryResult::empty())
            }
            #[cfg(feature = "lpg")]
            SessionCommand::UseGraph(name) => {
                // Check per-graph grant before switching
                if self.identity.has_grants()
                    && !name.eq_ignore_ascii_case("default")
                    && !self
                        .identity
                        .can_access_graph(&name, crate::auth::Role::ReadOnly)
                {
                    return Err(Error::Query(QueryError::new(
                        QueryErrorKind::Semantic,
                        format!(
                            "permission denied: no grant for graph '{name}' (user: {})",
                            self.identity.user_id()
                        ),
                    )));
                }
                // Verify graph exists (resolve within current schema)
                let effective_key = self.effective_graph_key(&name);
                if !name.eq_ignore_ascii_case("default")
                    && self.store.graph(&effective_key).is_none()
                {
                    return Err(Error::Query(QueryError::new(
                        QueryErrorKind::Semantic,
                        format!("Graph '{name}' does not exist"),
                    )));
                }
                self.use_graph(&name);
                // Track the new graph if in a transaction
                self.track_graph_touch();
                Ok(QueryResult::empty())
            }
            #[cfg(feature = "lpg")]
            SessionCommand::SessionSetGraph(name) => {
                // ISO/IEC 39075 Section 7.1 GR2: set session graph (resolved within current schema)
                // Check per-graph grant before switching (same as USE GRAPH)
                if self.identity.has_grants()
                    && !name.eq_ignore_ascii_case("default")
                    && !self
                        .identity
                        .can_access_graph(&name, crate::auth::Role::ReadOnly)
                {
                    return Err(Error::Query(QueryError::new(
                        QueryErrorKind::Semantic,
                        format!(
                            "permission denied: no grant for graph '{name}' (user: {})",
                            self.identity.user_id()
                        ),
                    )));
                }
                let effective_key = self.effective_graph_key(&name);
                if !name.eq_ignore_ascii_case("default")
                    && self.store.graph(&effective_key).is_none()
                {
                    return Err(Error::Query(QueryError::new(
                        QueryErrorKind::Semantic,
                        format!("Graph '{name}' does not exist"),
                    )));
                }
                self.use_graph(&name);
                // Track the new graph if in a transaction
                self.track_graph_touch();
                Ok(QueryResult::empty())
            }
            SessionCommand::SessionSetSchema(name) => {
                // ISO/IEC 39075 Section 7.1 GR1: set session schema (independent of graph)
                if !self.catalog.schema_exists(&name) {
                    return Err(Error::Query(QueryError::new(
                        QueryErrorKind::Semantic,
                        format!("Schema '{name}' does not exist"),
                    )));
                }
                self.set_schema(&name);
                Ok(QueryResult::empty())
            }
            SessionCommand::SessionSetTimeZone(tz) => {
                self.set_time_zone(&tz);
                Ok(QueryResult::empty())
            }
            SessionCommand::SessionSetParameter(key, expr) => {
                if key.eq_ignore_ascii_case("viewing_epoch") {
                    match Self::eval_integer_literal(&expr) {
                        Some(n) if n >= 0 => {
                            self.set_viewing_epoch(EpochId::new(n as u64));
                            Ok(QueryResult::status(format!("Set viewing_epoch to {n}")))
                        }
                        _ => Err(Error::Query(QueryError::new(
                            QueryErrorKind::Semantic,
                            "viewing_epoch must be a non-negative integer literal",
                        ))),
                    }
                } else {
                    // For now, store parameter name with Null value.
                    // Full expression evaluation would require building and executing a plan.
                    self.set_parameter(&key, Value::Null);
                    Ok(QueryResult::empty())
                }
            }
            SessionCommand::SessionReset(target) => {
                use grafeo_adapters::query::gql::ast::SessionResetTarget;
                match target {
                    SessionResetTarget::All => self.reset_session(),
                    SessionResetTarget::Schema => self.reset_schema(),
                    SessionResetTarget::Graph => self.reset_graph(),
                    SessionResetTarget::TimeZone => self.reset_time_zone(),
                    SessionResetTarget::Parameters => self.reset_parameters(),
                }
                Ok(QueryResult::empty())
            }
            SessionCommand::SessionClose => {
                self.reset_session();
                Ok(QueryResult::empty())
            }
            #[cfg(feature = "lpg")]
            SessionCommand::StartTransaction {
                read_only,
                isolation_level,
            } => {
                let engine_level = isolation_level.map(|l| match l {
                    TransactionIsolationLevel::ReadCommitted => {
                        crate::transaction::IsolationLevel::ReadCommitted
                    }
                    TransactionIsolationLevel::SnapshotIsolation => {
                        crate::transaction::IsolationLevel::SnapshotIsolation
                    }
                    TransactionIsolationLevel::Serializable => {
                        crate::transaction::IsolationLevel::Serializable
                    }
                });
                self.begin_transaction_inner(read_only, engine_level)?;
                Ok(QueryResult::status("Transaction started"))
            }
            #[cfg(feature = "lpg")]
            SessionCommand::Commit => {
                self.commit_inner()?;
                Ok(QueryResult::status("Transaction committed"))
            }
            #[cfg(feature = "lpg")]
            SessionCommand::Rollback => {
                self.rollback_inner()?;
                Ok(QueryResult::status("Transaction rolled back"))
            }
            #[cfg(feature = "lpg")]
            SessionCommand::Savepoint(name) => {
                self.savepoint(&name)?;
                Ok(QueryResult::status(format!("Savepoint '{name}' created")))
            }
            #[cfg(feature = "lpg")]
            SessionCommand::RollbackToSavepoint(name) => {
                self.rollback_to_savepoint(&name)?;
                Ok(QueryResult::status(format!(
                    "Rolled back to savepoint '{name}'"
                )))
            }
            #[cfg(feature = "lpg")]
            SessionCommand::ReleaseSavepoint(name) => {
                self.release_savepoint(&name)?;
                Ok(QueryResult::status(format!("Savepoint '{name}' released")))
            }
            #[cfg(feature = "lpg")]
            SessionCommand::CreateProjection {
                name,
                node_labels,
                edge_types,
            } => {
                use grafeo_core::graph::{GraphProjection, ProjectionSpec};
                use std::collections::hash_map::Entry;

                let spec = ProjectionSpec::new()
                    .with_node_labels(node_labels)
                    .with_edge_types(edge_types);

                let store = self.active_store();
                let projection = Arc::new(GraphProjection::new(store, spec));
                let mut projections = self.projections.write();
                match projections.entry(name.clone()) {
                    Entry::Occupied(_) => Err(Error::Query(QueryError::new(
                        QueryErrorKind::Semantic,
                        format!("Projection '{name}' already exists"),
                    ))),
                    Entry::Vacant(e) => {
                        e.insert(projection);
                        Ok(QueryResult::status(format!("Projection '{name}' created")))
                    }
                }
            }
            #[cfg(feature = "lpg")]
            SessionCommand::DropProjection { name } => {
                let removed = self.projections.write().remove(&name).is_some();
                if !removed {
                    return Err(Error::Query(QueryError::new(
                        QueryErrorKind::Semantic,
                        format!("Projection '{name}' does not exist"),
                    )));
                }
                Ok(QueryResult::status(format!("Projection '{name}' dropped")))
            }
            #[cfg(feature = "lpg")]
            SessionCommand::ShowProjections => {
                let mut names: Vec<String> = self.projections.read().keys().cloned().collect();
                names.sort();
                let rows: Vec<Vec<Value>> =
                    names.into_iter().map(|n| vec![Value::from(n)]).collect();
                Ok(QueryResult {
                    columns: vec!["name".to_string()],
                    column_types: Vec::new(),
                    rows,
                    ..QueryResult::empty()
                })
            }
            #[cfg(not(feature = "lpg"))]
            _ => Err(grafeo_common::utils::error::Error::Internal(
                "This command requires the `lpg` feature".to_string(),
            )),
        }
    }

    /// Logs a WAL record for a schema change (no-op if WAL is not enabled).
    #[cfg(feature = "wal")]
    fn log_schema_wal(&self, record: &grafeo_storage::wal::WalRecord) {
        if let Some(ref wal) = self.wal
            && let Err(e) = wal.log(record)
        {
            grafeo_warn!("Failed to log schema change to WAL: {}", e);
        }
    }

    /// Executes a schema DDL command, returning a status result.
    #[cfg(all(feature = "lpg", feature = "gql"))]
    fn execute_schema_command(
        &self,
        cmd: grafeo_adapters::query::gql::ast::SchemaStatement,
    ) -> Result<QueryResult> {
        use crate::catalog::{
            EdgeTypeDefinition, NodeTypeDefinition, PropertyDataType, TypedProperty,
        };
        use grafeo_adapters::query::gql::ast::SchemaStatement;
        use grafeo_common::utils::error::{Error, QueryError, QueryErrorKind};
        #[cfg(feature = "wal")]
        use grafeo_storage::wal::WalRecord;

        /// Logs a WAL record for schema changes. Compiles to nothing without `wal`.
        macro_rules! wal_log {
            ($self:expr, $record:expr) => {
                #[cfg(feature = "wal")]
                $self.log_schema_wal(&$record);
            };
        }

        let result = match cmd {
            SchemaStatement::CreateNodeType(stmt) => {
                let effective_name = self.effective_type_key(&stmt.name);
                #[cfg(feature = "wal")]
                let props_for_wal: Vec<(String, String, bool)> = stmt
                    .properties
                    .iter()
                    .map(|p| (p.name.clone(), p.data_type.clone(), p.nullable))
                    .collect();
                let def = NodeTypeDefinition {
                    name: effective_name.clone(),
                    properties: stmt
                        .properties
                        .iter()
                        .map(|p| TypedProperty {
                            name: p.name.clone(),
                            data_type: PropertyDataType::from_type_name(&p.data_type),
                            nullable: p.nullable,
                            default_value: p
                                .default_value
                                .as_ref()
                                .map(|s| parse_default_literal(s)),
                        })
                        .collect(),
                    constraints: Vec::new(),
                    parent_types: stmt.parent_types.clone(),
                };
                let result = if stmt.or_replace {
                    let _ = self.catalog.drop_node_type(&effective_name);
                    self.catalog.register_node_type(def)
                } else {
                    self.catalog.register_node_type(def)
                };
                match result {
                    Ok(()) => {
                        wal_log!(
                            self,
                            WalRecord::CreateNodeType {
                                name: effective_name.clone(),
                                properties: props_for_wal,
                                constraints: Vec::new(),
                            }
                        );
                        Ok(QueryResult::status(format!(
                            "Created node type '{}'",
                            stmt.name
                        )))
                    }
                    Err(e) if stmt.if_not_exists => {
                        let _ = e;
                        Ok(QueryResult::status("No change"))
                    }
                    Err(e) => Err(Error::Query(QueryError::new(
                        QueryErrorKind::Semantic,
                        e.to_string(),
                    ))),
                }
            }
            SchemaStatement::CreateEdgeType(stmt) => {
                let effective_name = self.effective_type_key(&stmt.name);
                #[cfg(feature = "wal")]
                let props_for_wal: Vec<(String, String, bool)> = stmt
                    .properties
                    .iter()
                    .map(|p| (p.name.clone(), p.data_type.clone(), p.nullable))
                    .collect();
                let def = EdgeTypeDefinition {
                    name: effective_name.clone(),
                    properties: stmt
                        .properties
                        .iter()
                        .map(|p| TypedProperty {
                            name: p.name.clone(),
                            data_type: PropertyDataType::from_type_name(&p.data_type),
                            nullable: p.nullable,
                            default_value: p
                                .default_value
                                .as_ref()
                                .map(|s| parse_default_literal(s)),
                        })
                        .collect(),
                    constraints: Vec::new(),
                    source_node_types: stmt.source_node_types.clone(),
                    target_node_types: stmt.target_node_types.clone(),
                };
                let result = if stmt.or_replace {
                    let _ = self.catalog.drop_edge_type_def(&effective_name);
                    self.catalog.register_edge_type_def(def)
                } else {
                    self.catalog.register_edge_type_def(def)
                };
                match result {
                    Ok(()) => {
                        wal_log!(
                            self,
                            WalRecord::CreateEdgeType {
                                name: effective_name.clone(),
                                properties: props_for_wal,
                                constraints: Vec::new(),
                            }
                        );
                        Ok(QueryResult::status(format!(
                            "Created edge type '{}'",
                            stmt.name
                        )))
                    }
                    Err(e) if stmt.if_not_exists => {
                        let _ = e;
                        Ok(QueryResult::status("No change"))
                    }
                    Err(e) => Err(Error::Query(QueryError::new(
                        QueryErrorKind::Semantic,
                        e.to_string(),
                    ))),
                }
            }
            SchemaStatement::CreateVectorIndex(stmt) => {
                Self::create_vector_index_on_store(
                    &self.active_lpg_store(),
                    &stmt.node_label,
                    &stmt.property,
                    stmt.dimensions,
                    stmt.metric.as_deref(),
                )?;
                wal_log!(
                    self,
                    WalRecord::CreateIndex {
                        name: stmt.name.clone(),
                        label: stmt.node_label.clone(),
                        property: stmt.property.clone(),
                        index_type: "vector".to_string(),
                    }
                );
                Ok(QueryResult::status(format!(
                    "Created vector index '{}'",
                    stmt.name
                )))
            }
            SchemaStatement::DropNodeType { name, if_exists } => {
                let effective_name = self.effective_type_key(&name);
                match self.catalog.drop_node_type(&effective_name) {
                    Ok(()) => {
                        wal_log!(
                            self,
                            WalRecord::DropNodeType {
                                name: effective_name
                            }
                        );
                        Ok(QueryResult::status(format!("Dropped node type '{name}'")))
                    }
                    Err(e) if if_exists => {
                        let _ = e;
                        Ok(QueryResult::status("No change"))
                    }
                    Err(e) => Err(Error::Query(QueryError::new(
                        QueryErrorKind::Semantic,
                        e.to_string(),
                    ))),
                }
            }
            SchemaStatement::DropEdgeType { name, if_exists } => {
                let effective_name = self.effective_type_key(&name);
                match self.catalog.drop_edge_type_def(&effective_name) {
                    Ok(()) => {
                        wal_log!(
                            self,
                            WalRecord::DropEdgeType {
                                name: effective_name
                            }
                        );
                        Ok(QueryResult::status(format!("Dropped edge type '{name}'")))
                    }
                    Err(e) if if_exists => {
                        let _ = e;
                        Ok(QueryResult::status("No change"))
                    }
                    Err(e) => Err(Error::Query(QueryError::new(
                        QueryErrorKind::Semantic,
                        e.to_string(),
                    ))),
                }
            }
            SchemaStatement::CreateIndex(stmt) => {
                use crate::catalog::IndexType as CatalogIndexType;
                use grafeo_adapters::query::gql::ast::IndexKind;
                let active = self.active_lpg_store();
                let index_type_str = match stmt.index_kind {
                    IndexKind::Property => "property",
                    IndexKind::BTree => "btree",
                    IndexKind::Text => "text",
                    IndexKind::Vector => "vector",
                };
                match stmt.index_kind {
                    IndexKind::Property | IndexKind::BTree => {
                        for prop in &stmt.properties {
                            active.create_property_index(prop);
                        }
                    }
                    IndexKind::Text => {
                        for prop in &stmt.properties {
                            Self::create_text_index_on_store(&active, &stmt.label, prop)?;
                        }
                    }
                    IndexKind::Vector => {
                        for prop in &stmt.properties {
                            Self::create_vector_index_on_store(
                                &active,
                                &stmt.label,
                                prop,
                                stmt.options.dimensions,
                                stmt.options.metric.as_deref(),
                            )?;
                        }
                    }
                }
                // Register each property index in the catalog for SHOW INDEXES
                // and DROP INDEX name-based lookup.
                let catalog_index_type = match stmt.index_kind {
                    IndexKind::Property => CatalogIndexType::Hash,
                    IndexKind::BTree => CatalogIndexType::BTree,
                    IndexKind::Text => CatalogIndexType::FullText,
                    IndexKind::Vector => CatalogIndexType::Hash,
                };
                let label_id = self.catalog.get_or_create_label(&stmt.label);
                for prop in &stmt.properties {
                    let prop_id = self.catalog.get_or_create_property_key(prop);
                    self.catalog
                        .create_index(&stmt.name, label_id, prop_id, catalog_index_type);
                }
                #[cfg(feature = "wal")]
                for prop in &stmt.properties {
                    wal_log!(
                        self,
                        WalRecord::CreateIndex {
                            name: stmt.name.clone(),
                            label: stmt.label.clone(),
                            property: prop.clone(),
                            index_type: index_type_str.to_string(),
                        }
                    );
                }
                Ok(QueryResult::status(format!(
                    "Created {} index '{}'",
                    index_type_str, stmt.name
                )))
            }
            SchemaStatement::DropIndex { name, if_exists } => {
                // Look up the index by name in the catalog to find the
                // underlying property, then drop from both catalog and store.
                if let Some(index_id) = self.catalog.find_index_by_name(&name) {
                    let def = self.catalog.get_index(index_id);
                    self.catalog.drop_index(index_id);
                    if let Some(def) = def
                        && let Some(prop_name) =
                            self.catalog.get_property_key_name(def.property_key)
                    {
                        self.active_lpg_store().drop_property_index(&prop_name);
                    }
                    wal_log!(self, WalRecord::DropIndex { name: name.clone() });
                    Ok(QueryResult::status(format!("Dropped index '{name}'")))
                } else if if_exists {
                    Ok(QueryResult::status("No change".to_string()))
                } else {
                    Err(Error::Query(QueryError::new(
                        QueryErrorKind::Semantic,
                        format!("Index '{name}' does not exist"),
                    )))
                }
            }
            SchemaStatement::CreateConstraint(stmt) => {
                use crate::catalog::TypeConstraint;
                use grafeo_adapters::query::gql::ast::ConstraintKind;
                let kind_str = match stmt.constraint_kind {
                    ConstraintKind::Unique => "unique",
                    ConstraintKind::NodeKey => "node_key",
                    ConstraintKind::NotNull => "not_null",
                    ConstraintKind::Exists => "exists",
                };
                let constraint_name = stmt
                    .name
                    .clone()
                    .unwrap_or_else(|| format!("{}_{kind_str}", stmt.label));

                // Register constraint in catalog type definitions
                match stmt.constraint_kind {
                    ConstraintKind::Unique => {
                        for prop in &stmt.properties {
                            let label_id = self.catalog.get_or_create_label(&stmt.label);
                            let prop_id = self.catalog.get_or_create_property_key(prop);
                            let _ = self.catalog.add_unique_constraint(label_id, prop_id);
                        }
                        let _ = self.catalog.add_constraint_to_type(
                            &stmt.label,
                            TypeConstraint::Unique(stmt.properties.clone()),
                        );
                    }
                    ConstraintKind::NodeKey => {
                        for prop in &stmt.properties {
                            let label_id = self.catalog.get_or_create_label(&stmt.label);
                            let prop_id = self.catalog.get_or_create_property_key(prop);
                            let _ = self.catalog.add_unique_constraint(label_id, prop_id);
                            let _ = self.catalog.add_required_property(label_id, prop_id);
                        }
                        let _ = self.catalog.add_constraint_to_type(
                            &stmt.label,
                            TypeConstraint::PrimaryKey(stmt.properties.clone()),
                        );
                    }
                    ConstraintKind::NotNull | ConstraintKind::Exists => {
                        for prop in &stmt.properties {
                            let label_id = self.catalog.get_or_create_label(&stmt.label);
                            let prop_id = self.catalog.get_or_create_property_key(prop);
                            let _ = self.catalog.add_required_property(label_id, prop_id);
                            let _ = self.catalog.add_constraint_to_type(
                                &stmt.label,
                                TypeConstraint::NotNull(prop.clone()),
                            );
                        }
                    }
                }

                wal_log!(
                    self,
                    WalRecord::CreateConstraint {
                        name: constraint_name.clone(),
                        label: stmt.label.clone(),
                        properties: stmt.properties.clone(),
                        kind: kind_str.to_string(),
                    }
                );
                Ok(QueryResult::status(format!(
                    "Created {kind_str} constraint '{constraint_name}'"
                )))
            }
            SchemaStatement::DropConstraint { name, if_exists } => {
                let _ = if_exists;
                wal_log!(self, WalRecord::DropConstraint { name: name.clone() });
                Ok(QueryResult::status(format!("Dropped constraint '{name}'")))
            }
            SchemaStatement::CreateGraphType(stmt) => {
                use crate::catalog::GraphTypeDefinition;
                use grafeo_adapters::query::gql::ast::InlineElementType;

                let effective_name = self.effective_type_key(&stmt.name);

                // GG04: LIKE clause copies type from existing graph
                let (mut node_types, mut edge_types, open) =
                    if let Some(ref like_graph) = stmt.like_graph {
                        // Infer types from the graph's bound type, or use its existing types
                        if let Some(type_name) = self.catalog.get_graph_type_binding(like_graph) {
                            if let Some(existing) = self
                                .catalog
                                .schema()
                                .and_then(|s| s.get_graph_type(&type_name))
                            {
                                (
                                    existing.allowed_node_types.clone(),
                                    existing.allowed_edge_types.clone(),
                                    existing.open,
                                )
                            } else {
                                (Vec::new(), Vec::new(), true)
                            }
                        } else {
                            // GG22: Infer from graph data (labels used in graph)
                            let nt = self.catalog.all_node_type_names();
                            let et = self.catalog.all_edge_type_names();
                            if nt.is_empty() && et.is_empty() {
                                (Vec::new(), Vec::new(), true)
                            } else {
                                (nt, et, false)
                            }
                        }
                    } else {
                        // Prefix element type names with schema for consistency
                        let nt = stmt
                            .node_types
                            .iter()
                            .map(|n| self.effective_type_key(n))
                            .collect();
                        let et = stmt
                            .edge_types
                            .iter()
                            .map(|n| self.effective_type_key(n))
                            .collect();
                        (nt, et, stmt.open)
                    };

                // GG03: Register inline element types and add their names
                for inline in &stmt.inline_types {
                    match inline {
                        InlineElementType::Node {
                            name,
                            properties,
                            key_labels,
                            ..
                        } => {
                            let inline_effective = self.effective_type_key(name);
                            let def = NodeTypeDefinition {
                                name: inline_effective.clone(),
                                properties: properties
                                    .iter()
                                    .map(|p| TypedProperty {
                                        name: p.name.clone(),
                                        data_type: PropertyDataType::from_type_name(&p.data_type),
                                        nullable: p.nullable,
                                        default_value: None,
                                    })
                                    .collect(),
                                constraints: Vec::new(),
                                parent_types: key_labels.clone(),
                            };
                            // Register or replace so inline defs override existing
                            self.catalog.register_or_replace_node_type(def);
                            #[cfg(feature = "wal")]
                            {
                                let props_for_wal: Vec<(String, String, bool)> = properties
                                    .iter()
                                    .map(|p| (p.name.clone(), p.data_type.clone(), p.nullable))
                                    .collect();
                                self.log_schema_wal(&WalRecord::CreateNodeType {
                                    name: inline_effective.clone(),
                                    properties: props_for_wal,
                                    constraints: Vec::new(),
                                });
                            }
                            if !node_types.contains(&inline_effective) {
                                node_types.push(inline_effective);
                            }
                        }
                        InlineElementType::Edge {
                            name,
                            properties,
                            source_node_types,
                            target_node_types,
                            ..
                        } => {
                            let inline_effective = self.effective_type_key(name);
                            let def = EdgeTypeDefinition {
                                name: inline_effective.clone(),
                                properties: properties
                                    .iter()
                                    .map(|p| TypedProperty {
                                        name: p.name.clone(),
                                        data_type: PropertyDataType::from_type_name(&p.data_type),
                                        nullable: p.nullable,
                                        default_value: None,
                                    })
                                    .collect(),
                                constraints: Vec::new(),
                                source_node_types: source_node_types.clone(),
                                target_node_types: target_node_types.clone(),
                            };
                            self.catalog.register_or_replace_edge_type_def(def);
                            #[cfg(feature = "wal")]
                            {
                                let props_for_wal: Vec<(String, String, bool)> = properties
                                    .iter()
                                    .map(|p| (p.name.clone(), p.data_type.clone(), p.nullable))
                                    .collect();
                                self.log_schema_wal(&WalRecord::CreateEdgeType {
                                    name: inline_effective.clone(),
                                    properties: props_for_wal,
                                    constraints: Vec::new(),
                                });
                            }
                            if !edge_types.contains(&inline_effective) {
                                edge_types.push(inline_effective);
                            }
                        }
                    }
                }

                let def = GraphTypeDefinition {
                    name: effective_name.clone(),
                    allowed_node_types: node_types.clone(),
                    allowed_edge_types: edge_types.clone(),
                    open,
                };
                let result = if stmt.or_replace {
                    // Drop existing first, ignore error if not found
                    let _ = self.catalog.drop_graph_type(&effective_name);
                    self.catalog.register_graph_type(def)
                } else {
                    self.catalog.register_graph_type(def)
                };
                match result {
                    Ok(()) => {
                        wal_log!(
                            self,
                            WalRecord::CreateGraphType {
                                name: effective_name.clone(),
                                node_types,
                                edge_types,
                                open,
                            }
                        );
                        Ok(QueryResult::status(format!(
                            "Created graph type '{}'",
                            stmt.name
                        )))
                    }
                    Err(e) if stmt.if_not_exists => {
                        let _ = e;
                        Ok(QueryResult::status("No change"))
                    }
                    Err(e) => Err(Error::Query(QueryError::new(
                        QueryErrorKind::Semantic,
                        e.to_string(),
                    ))),
                }
            }
            SchemaStatement::DropGraphType { name, if_exists } => {
                let effective_name = self.effective_type_key(&name);
                match self.catalog.drop_graph_type(&effective_name) {
                    Ok(()) => {
                        wal_log!(
                            self,
                            WalRecord::DropGraphType {
                                name: effective_name
                            }
                        );
                        Ok(QueryResult::status(format!("Dropped graph type '{name}'")))
                    }
                    Err(e) if if_exists => {
                        let _ = e;
                        Ok(QueryResult::status("No change"))
                    }
                    Err(e) => Err(Error::Query(QueryError::new(
                        QueryErrorKind::Semantic,
                        e.to_string(),
                    ))),
                }
            }
            SchemaStatement::CreateSchema {
                name,
                if_not_exists,
            } => match self.catalog.register_schema_namespace(name.clone()) {
                Ok(()) => {
                    wal_log!(self, WalRecord::CreateSchema { name: name.clone() });
                    // Auto-create the schema's default graph partition so that
                    // SESSION SET SCHEMA + queries work without an explicit graph.
                    let default_key = format!("{name}/{SCHEMA_DEFAULT_GRAPH}");
                    if self.store.create_graph(&default_key).unwrap_or(false) {
                        wal_log!(self, WalRecord::CreateNamedGraph { name: default_key });
                    }
                    Ok(QueryResult::status(format!("Created schema '{name}'")))
                }
                Err(e) if if_not_exists => {
                    let _ = e;
                    Ok(QueryResult::status("No change"))
                }
                Err(e) => Err(Error::Query(QueryError::new(
                    QueryErrorKind::Semantic,
                    e.to_string(),
                ))),
            },
            SchemaStatement::DropSchema { name, if_exists } => {
                // ISO/IEC 39075 Section 12.3: schema must be empty before dropping.
                // The auto-created __default__ graph is exempt from the check.
                let prefix = format!("{name}/");
                let default_graph_key = format!("{name}/{SCHEMA_DEFAULT_GRAPH}");
                let has_graphs = self
                    .store
                    .graph_names()
                    .iter()
                    .any(|g| g.starts_with(&prefix) && *g != default_graph_key);
                let has_types = self
                    .catalog
                    .all_node_type_names()
                    .iter()
                    .any(|n| n.starts_with(&prefix))
                    || self
                        .catalog
                        .all_edge_type_names()
                        .iter()
                        .any(|n| n.starts_with(&prefix))
                    || self
                        .catalog
                        .all_graph_type_names()
                        .iter()
                        .any(|n| n.starts_with(&prefix));
                if has_graphs || has_types {
                    return Err(Error::Query(QueryError::new(
                        QueryErrorKind::Semantic,
                        format!("Schema '{name}' is not empty: drop all graphs and types first"),
                    )));
                }
                match self.catalog.drop_schema_namespace(&name) {
                    Ok(()) => {
                        wal_log!(self, WalRecord::DropSchema { name: name.clone() });
                        // Drop the auto-created default graph partition
                        if self.store.drop_graph(&default_graph_key) {
                            wal_log!(
                                self,
                                WalRecord::DropNamedGraph {
                                    name: default_graph_key,
                                }
                            );
                        }
                        // If this session was using the dropped schema, reset it
                        let mut current = self.current_schema.lock();
                        if current
                            .as_deref()
                            .is_some_and(|s| s.eq_ignore_ascii_case(&name))
                        {
                            *current = None;
                        }
                        Ok(QueryResult::status(format!("Dropped schema '{name}'")))
                    }
                    Err(e) if if_exists => {
                        let _ = e;
                        Ok(QueryResult::status("No change"))
                    }
                    Err(e) => Err(Error::Query(QueryError::new(
                        QueryErrorKind::Semantic,
                        e.to_string(),
                    ))),
                }
            }
            SchemaStatement::AlterNodeType(stmt) => {
                use grafeo_adapters::query::gql::ast::TypeAlteration;
                let effective_name = self.effective_type_key(&stmt.name);
                let mut wal_alts = Vec::new();
                for alt in &stmt.alterations {
                    match alt {
                        TypeAlteration::AddProperty(prop) => {
                            let typed = TypedProperty {
                                name: prop.name.clone(),
                                data_type: PropertyDataType::from_type_name(&prop.data_type),
                                nullable: prop.nullable,
                                default_value: prop
                                    .default_value
                                    .as_ref()
                                    .map(|s| parse_default_literal(s)),
                            };
                            self.catalog
                                .alter_node_type_add_property(&effective_name, typed)
                                .map_err(|e| {
                                    Error::Query(QueryError::new(
                                        QueryErrorKind::Semantic,
                                        e.to_string(),
                                    ))
                                })?;
                            wal_alts.push((
                                "add".to_string(),
                                prop.name.clone(),
                                prop.data_type.clone(),
                                prop.nullable,
                            ));
                        }
                        TypeAlteration::DropProperty(name) => {
                            self.catalog
                                .alter_node_type_drop_property(&effective_name, name)
                                .map_err(|e| {
                                    Error::Query(QueryError::new(
                                        QueryErrorKind::Semantic,
                                        e.to_string(),
                                    ))
                                })?;
                            wal_alts.push(("drop".to_string(), name.clone(), String::new(), false));
                        }
                    }
                }
                wal_log!(
                    self,
                    WalRecord::AlterNodeType {
                        name: effective_name,
                        alterations: wal_alts,
                    }
                );
                Ok(QueryResult::status(format!(
                    "Altered node type '{}'",
                    stmt.name
                )))
            }
            SchemaStatement::AlterEdgeType(stmt) => {
                use grafeo_adapters::query::gql::ast::TypeAlteration;
                let effective_name = self.effective_type_key(&stmt.name);
                let mut wal_alts = Vec::new();
                for alt in &stmt.alterations {
                    match alt {
                        TypeAlteration::AddProperty(prop) => {
                            let typed = TypedProperty {
                                name: prop.name.clone(),
                                data_type: PropertyDataType::from_type_name(&prop.data_type),
                                nullable: prop.nullable,
                                default_value: prop
                                    .default_value
                                    .as_ref()
                                    .map(|s| parse_default_literal(s)),
                            };
                            self.catalog
                                .alter_edge_type_add_property(&effective_name, typed)
                                .map_err(|e| {
                                    Error::Query(QueryError::new(
                                        QueryErrorKind::Semantic,
                                        e.to_string(),
                                    ))
                                })?;
                            wal_alts.push((
                                "add".to_string(),
                                prop.name.clone(),
                                prop.data_type.clone(),
                                prop.nullable,
                            ));
                        }
                        TypeAlteration::DropProperty(name) => {
                            self.catalog
                                .alter_edge_type_drop_property(&effective_name, name)
                                .map_err(|e| {
                                    Error::Query(QueryError::new(
                                        QueryErrorKind::Semantic,
                                        e.to_string(),
                                    ))
                                })?;
                            wal_alts.push(("drop".to_string(), name.clone(), String::new(), false));
                        }
                    }
                }
                wal_log!(
                    self,
                    WalRecord::AlterEdgeType {
                        name: effective_name,
                        alterations: wal_alts,
                    }
                );
                Ok(QueryResult::status(format!(
                    "Altered edge type '{}'",
                    stmt.name
                )))
            }
            SchemaStatement::AlterGraphType(stmt) => {
                use grafeo_adapters::query::gql::ast::GraphTypeAlteration;
                let effective_name = self.effective_type_key(&stmt.name);
                let mut wal_alts = Vec::new();
                for alt in &stmt.alterations {
                    match alt {
                        GraphTypeAlteration::AddNodeType(name) => {
                            self.catalog
                                .alter_graph_type_add_node_type(&effective_name, name.clone())
                                .map_err(|e| {
                                    Error::Query(QueryError::new(
                                        QueryErrorKind::Semantic,
                                        e.to_string(),
                                    ))
                                })?;
                            wal_alts.push(("add_node_type".to_string(), name.clone()));
                        }
                        GraphTypeAlteration::DropNodeType(name) => {
                            self.catalog
                                .alter_graph_type_drop_node_type(&effective_name, name)
                                .map_err(|e| {
                                    Error::Query(QueryError::new(
                                        QueryErrorKind::Semantic,
                                        e.to_string(),
                                    ))
                                })?;
                            wal_alts.push(("drop_node_type".to_string(), name.clone()));
                        }
                        GraphTypeAlteration::AddEdgeType(name) => {
                            self.catalog
                                .alter_graph_type_add_edge_type(&effective_name, name.clone())
                                .map_err(|e| {
                                    Error::Query(QueryError::new(
                                        QueryErrorKind::Semantic,
                                        e.to_string(),
                                    ))
                                })?;
                            wal_alts.push(("add_edge_type".to_string(), name.clone()));
                        }
                        GraphTypeAlteration::DropEdgeType(name) => {
                            self.catalog
                                .alter_graph_type_drop_edge_type(&effective_name, name)
                                .map_err(|e| {
                                    Error::Query(QueryError::new(
                                        QueryErrorKind::Semantic,
                                        e.to_string(),
                                    ))
                                })?;
                            wal_alts.push(("drop_edge_type".to_string(), name.clone()));
                        }
                    }
                }
                wal_log!(
                    self,
                    WalRecord::AlterGraphType {
                        name: effective_name,
                        alterations: wal_alts,
                    }
                );
                Ok(QueryResult::status(format!(
                    "Altered graph type '{}'",
                    stmt.name
                )))
            }
            SchemaStatement::CreateProcedure(stmt) => {
                use crate::catalog::ProcedureDefinition;

                let def = ProcedureDefinition {
                    name: stmt.name.clone(),
                    params: stmt
                        .params
                        .iter()
                        .map(|p| (p.name.clone(), p.param_type.clone()))
                        .collect(),
                    returns: stmt
                        .returns
                        .iter()
                        .map(|r| (r.name.clone(), r.return_type.clone()))
                        .collect(),
                    body: stmt.body.clone(),
                };

                if stmt.or_replace {
                    self.catalog.replace_procedure(def).map_err(|e| {
                        Error::Query(QueryError::new(QueryErrorKind::Semantic, e.to_string()))
                    })?;
                } else {
                    match self.catalog.register_procedure(def) {
                        Ok(()) => {}
                        Err(_) if stmt.if_not_exists => {
                            return Ok(QueryResult::empty());
                        }
                        Err(e) => {
                            return Err(Error::Query(QueryError::new(
                                QueryErrorKind::Semantic,
                                e.to_string(),
                            )));
                        }
                    }
                }

                wal_log!(
                    self,
                    WalRecord::CreateProcedure {
                        name: stmt.name.clone(),
                        params: stmt
                            .params
                            .iter()
                            .map(|p| (p.name.clone(), p.param_type.clone()))
                            .collect(),
                        returns: stmt
                            .returns
                            .iter()
                            .map(|r| (r.name.clone(), r.return_type.clone()))
                            .collect(),
                        body: stmt.body,
                    }
                );
                Ok(QueryResult::status(format!(
                    "Created procedure '{}'",
                    stmt.name
                )))
            }
            SchemaStatement::DropProcedure { name, if_exists } => {
                match self.catalog.drop_procedure(&name) {
                    Ok(()) => {}
                    Err(_) if if_exists => {
                        return Ok(QueryResult::empty());
                    }
                    Err(e) => {
                        return Err(Error::Query(QueryError::new(
                            QueryErrorKind::Semantic,
                            e.to_string(),
                        )));
                    }
                }
                wal_log!(self, WalRecord::DropProcedure { name: name.clone() });
                Ok(QueryResult::status(format!("Dropped procedure '{name}'")))
            }
            SchemaStatement::ShowIndexes => {
                return self.execute_show_indexes();
            }
            SchemaStatement::ShowConstraints => {
                return self.execute_show_constraints();
            }
            SchemaStatement::ShowNodeTypes => {
                return self.execute_show_node_types();
            }
            SchemaStatement::ShowEdgeTypes => {
                return self.execute_show_edge_types();
            }
            SchemaStatement::ShowGraphTypes => {
                return self.execute_show_graph_types();
            }
            SchemaStatement::ShowGraphType(name) => {
                return self.execute_show_graph_type(&name);
            }
            SchemaStatement::ShowCurrentGraphType => {
                return self.execute_show_current_graph_type();
            }
            SchemaStatement::ShowGraphs => {
                return self.execute_show_graphs();
            }
            SchemaStatement::ShowSchemas => {
                return self.execute_show_schemas();
            }
        };

        // Invalidate all cached query plans after any successful DDL change.
        // DDL is rare, so clearing the entire cache is cheap and correct.
        if result.is_ok() {
            self.query_cache.clear();
        }

        result
    }

    /// Creates a vector index on the store by scanning existing nodes.
    #[cfg(all(feature = "lpg", feature = "gql", feature = "vector-index"))]
    fn create_vector_index_on_store(
        store: &LpgStore,
        label: &str,
        property: &str,
        dimensions: Option<usize>,
        metric: Option<&str>,
    ) -> Result<()> {
        use grafeo_common::types::{PropertyKey, Value};
        use grafeo_common::utils::error::Error;
        use grafeo_core::index::vector::{DistanceMetric, HnswConfig, HnswIndex, VectorIndexKind};

        let metric = match metric {
            Some(m) => DistanceMetric::from_str(m).ok_or_else(|| {
                Error::Internal(format!(
                    "Unknown distance metric '{m}'. Use: cosine, euclidean, dot_product, manhattan"
                ))
            })?,
            None => DistanceMetric::Cosine,
        };

        let prop_key = PropertyKey::new(property);
        let mut found_dims: Option<usize> = dimensions;
        let mut vectors: Vec<(grafeo_common::types::NodeId, Vec<f32>)> = Vec::new();

        for node in store.nodes_with_label(label) {
            if let Some(Value::Vector(v)) = node.properties.get(&prop_key) {
                if let Some(expected) = found_dims {
                    if v.len() != expected {
                        return Err(Error::Internal(format!(
                            "Vector dimension mismatch: expected {expected}, found {} on node {}",
                            v.len(),
                            node.id.0
                        )));
                    }
                } else {
                    found_dims = Some(v.len());
                }
                vectors.push((node.id, v.to_vec()));
            }
        }

        let Some(dims) = found_dims else {
            return Err(Error::Internal(format!(
                "No vector properties found on :{label}({property}) and no dimensions specified"
            )));
        };

        let config = HnswConfig::new(dims, metric);
        let index = HnswIndex::with_capacity(config, vectors.len());
        let accessor = grafeo_core::index::vector::PropertyVectorAccessor::new(store, property);
        for (node_id, vec) in &vectors {
            index.insert(*node_id, vec, &accessor);
        }

        store.add_vector_index(label, property, Arc::new(VectorIndexKind::Hnsw(index)));
        Ok(())
    }

    /// Stub for when vector-index feature is not enabled.
    #[cfg(all(feature = "lpg", feature = "gql", not(feature = "vector-index")))]
    fn create_vector_index_on_store(
        _store: &LpgStore,
        _label: &str,
        _property: &str,
        _dimensions: Option<usize>,
        _metric: Option<&str>,
    ) -> Result<()> {
        Err(grafeo_common::utils::error::Error::Internal(
            "Vector index support requires the 'vector-index' feature".to_string(),
        ))
    }

    /// Creates a text index on the store by scanning existing nodes.
    #[cfg(all(feature = "lpg", feature = "gql", feature = "text-index"))]
    fn create_text_index_on_store(store: &LpgStore, label: &str, property: &str) -> Result<()> {
        use grafeo_common::types::{PropertyKey, Value};
        use grafeo_core::index::text::{BM25Config, InvertedIndex};

        let mut index = InvertedIndex::new(BM25Config::default());
        let prop_key = PropertyKey::new(property);

        let nodes = store.nodes_by_label(label);
        for node_id in nodes {
            if let Some(Value::String(text)) = store.get_node_property(node_id, &prop_key) {
                index.insert(node_id, text.as_str());
            }
        }

        store.add_text_index(label, property, Arc::new(parking_lot::RwLock::new(index)));
        Ok(())
    }

    /// Stub for when text-index feature is not enabled.
    #[cfg(all(feature = "lpg", feature = "gql", not(feature = "text-index")))]
    fn create_text_index_on_store(_store: &LpgStore, _label: &str, _property: &str) -> Result<()> {
        Err(grafeo_common::utils::error::Error::Internal(
            "Text index support requires the 'text-index' feature".to_string(),
        ))
    }

    /// Returns a table of all indexes from the catalog.
    fn execute_show_indexes(&self) -> Result<QueryResult> {
        let indexes = self.catalog.all_indexes();
        let columns = vec![
            "name".to_string(),
            "type".to_string(),
            "label".to_string(),
            "property".to_string(),
        ];
        let rows: Vec<Vec<Value>> = indexes
            .into_iter()
            .map(|def| {
                let label_name = self
                    .catalog
                    .get_label_name(def.label)
                    .unwrap_or_else(|| "?".into());
                let prop_name = self
                    .catalog
                    .get_property_key_name(def.property_key)
                    .unwrap_or_else(|| "?".into());
                vec![
                    Value::from(def.name),
                    Value::from(format!("{:?}", def.index_type)),
                    Value::from(&*label_name),
                    Value::from(&*prop_name),
                ]
            })
            .collect();
        Ok(QueryResult {
            columns,
            column_types: Vec::new(),
            rows,
            ..QueryResult::empty()
        })
    }

    /// Returns a table of all constraints (currently metadata-only).
    fn execute_show_constraints(&self) -> Result<QueryResult> {
        // Constraints are tracked in WAL but not yet in a queryable catalog.
        // Return an empty table with the expected schema.
        Ok(QueryResult {
            columns: vec![
                "name".to_string(),
                "type".to_string(),
                "label".to_string(),
                "properties".to_string(),
            ],
            column_types: Vec::new(),
            rows: Vec::new(),
            ..QueryResult::empty()
        })
    }

    /// Returns a table of all registered node types in the current schema.
    fn execute_show_node_types(&self) -> Result<QueryResult> {
        let columns = vec![
            "name".to_string(),
            "properties".to_string(),
            "constraints".to_string(),
            "parents".to_string(),
        ];
        let schema = self.current_schema.lock().clone();
        let all_names = self.catalog.all_node_type_names();
        let type_names: Vec<String> = match &schema {
            Some(s) => {
                let prefix = format!("{s}/");
                all_names
                    .into_iter()
                    .filter_map(|n| n.strip_prefix(&prefix).map(String::from))
                    .collect()
            }
            None => all_names.into_iter().filter(|n| !n.contains('/')).collect(),
        };
        let rows: Vec<Vec<Value>> = type_names
            .into_iter()
            .filter_map(|name| {
                let lookup = match &schema {
                    Some(s) => format!("{s}/{name}"),
                    None => name.clone(),
                };
                let def = self.catalog.get_node_type(&lookup)?;
                let props: Vec<String> = def
                    .properties
                    .iter()
                    .map(|p| {
                        let nullable = if p.nullable { "" } else { " NOT NULL" };
                        format!("{} {}{}", p.name, p.data_type, nullable)
                    })
                    .collect();
                let constraints: Vec<String> =
                    def.constraints.iter().map(|c| format!("{c:?}")).collect();
                let parents = def.parent_types.join(", ");
                Some(vec![
                    Value::from(name),
                    Value::from(props.join(", ")),
                    Value::from(constraints.join(", ")),
                    Value::from(parents),
                ])
            })
            .collect();
        Ok(QueryResult {
            columns,
            column_types: Vec::new(),
            rows,
            ..QueryResult::empty()
        })
    }

    /// Returns a table of all registered edge types in the current schema.
    fn execute_show_edge_types(&self) -> Result<QueryResult> {
        let columns = vec![
            "name".to_string(),
            "properties".to_string(),
            "source_types".to_string(),
            "target_types".to_string(),
        ];
        let schema = self.current_schema.lock().clone();
        let all_names = self.catalog.all_edge_type_names();
        let type_names: Vec<String> = match &schema {
            Some(s) => {
                let prefix = format!("{s}/");
                all_names
                    .into_iter()
                    .filter_map(|n| n.strip_prefix(&prefix).map(String::from))
                    .collect()
            }
            None => all_names.into_iter().filter(|n| !n.contains('/')).collect(),
        };
        let rows: Vec<Vec<Value>> = type_names
            .into_iter()
            .filter_map(|name| {
                let lookup = match &schema {
                    Some(s) => format!("{s}/{name}"),
                    None => name.clone(),
                };
                let def = self.catalog.get_edge_type_def(&lookup)?;
                let props: Vec<String> = def
                    .properties
                    .iter()
                    .map(|p| {
                        let nullable = if p.nullable { "" } else { " NOT NULL" };
                        format!("{} {}{}", p.name, p.data_type, nullable)
                    })
                    .collect();
                let src = def.source_node_types.join(", ");
                let tgt = def.target_node_types.join(", ");
                Some(vec![
                    Value::from(name),
                    Value::from(props.join(", ")),
                    Value::from(src),
                    Value::from(tgt),
                ])
            })
            .collect();
        Ok(QueryResult {
            columns,
            column_types: Vec::new(),
            rows,
            ..QueryResult::empty()
        })
    }

    /// Returns a table of all registered graph types in the current schema.
    fn execute_show_graph_types(&self) -> Result<QueryResult> {
        let columns = vec![
            "name".to_string(),
            "open".to_string(),
            "node_types".to_string(),
            "edge_types".to_string(),
        ];
        let schema = self.current_schema.lock().clone();
        let all_names = self.catalog.all_graph_type_names();
        let type_names: Vec<String> = match &schema {
            Some(s) => {
                let prefix = format!("{s}/");
                all_names
                    .into_iter()
                    .filter_map(|n| n.strip_prefix(&prefix).map(String::from))
                    .collect()
            }
            None => all_names.into_iter().filter(|n| !n.contains('/')).collect(),
        };
        let rows: Vec<Vec<Value>> = type_names
            .into_iter()
            .filter_map(|name| {
                let lookup = match &schema {
                    Some(s) => format!("{s}/{name}"),
                    None => name.clone(),
                };
                let def = self.catalog.get_graph_type_def(&lookup)?;
                // Strip schema prefix from allowed type names for display
                let strip = |n: &String| -> String {
                    match &schema {
                        Some(s) => n.strip_prefix(&format!("{s}/")).unwrap_or(n).to_string(),
                        None => n.clone(),
                    }
                };
                let node_types: Vec<String> = def.allowed_node_types.iter().map(strip).collect();
                let edge_types: Vec<String> = def.allowed_edge_types.iter().map(strip).collect();
                Some(vec![
                    Value::from(name),
                    Value::from(def.open),
                    Value::from(node_types.join(", ")),
                    Value::from(edge_types.join(", ")),
                ])
            })
            .collect();
        Ok(QueryResult {
            columns,
            column_types: Vec::new(),
            rows,
            ..QueryResult::empty()
        })
    }

    /// Returns the list of named graphs visible in the current schema context.
    ///
    /// When a session schema is set, only graphs belonging to that schema are
    /// shown (their compound prefix is stripped). When no schema is set, graphs
    /// without a schema prefix are shown (the default schema).
    #[cfg(feature = "lpg")]
    fn execute_show_graphs(&self) -> Result<QueryResult> {
        let schema = self.current_schema.lock().clone();
        let all_names = self.store.graph_names();

        let mut names: Vec<String> = match &schema {
            Some(s) => {
                let prefix = format!("{s}/");
                all_names
                    .into_iter()
                    .filter_map(|n| n.strip_prefix(&prefix).map(String::from))
                    .filter(|n| n != SCHEMA_DEFAULT_GRAPH)
                    .collect()
            }
            None => all_names.into_iter().filter(|n| !n.contains('/')).collect(),
        };
        names.sort();

        let rows: Vec<Vec<Value>> = names.into_iter().map(|n| vec![Value::from(n)]).collect();
        Ok(QueryResult {
            columns: vec!["name".to_string()],
            column_types: Vec::new(),
            rows,
            ..QueryResult::empty()
        })
    }

    /// Returns the list of all schema namespaces.
    fn execute_show_schemas(&self) -> Result<QueryResult> {
        let mut names = self.catalog.schema_names();
        names.sort();
        let rows: Vec<Vec<Value>> = names.into_iter().map(|n| vec![Value::from(n)]).collect();
        Ok(QueryResult {
            columns: vec!["name".to_string()],
            column_types: Vec::new(),
            rows,
            ..QueryResult::empty()
        })
    }

    /// Returns detailed info for a specific graph type.
    fn execute_show_graph_type(&self, name: &str) -> Result<QueryResult> {
        use grafeo_common::utils::error::{Error, QueryError, QueryErrorKind};

        let def = self.catalog.get_graph_type_def(name).ok_or_else(|| {
            Error::Query(QueryError::new(
                QueryErrorKind::Semantic,
                format!("Graph type '{name}' not found"),
            ))
        })?;

        let columns = vec![
            "name".to_string(),
            "open".to_string(),
            "node_types".to_string(),
            "edge_types".to_string(),
        ];
        let rows = vec![vec![
            Value::from(def.name),
            Value::from(def.open),
            Value::from(def.allowed_node_types.join(", ")),
            Value::from(def.allowed_edge_types.join(", ")),
        ]];
        Ok(QueryResult {
            columns,
            column_types: Vec::new(),
            rows,
            ..QueryResult::empty()
        })
    }

    /// Returns the graph type bound to the current graph.
    fn execute_show_current_graph_type(&self) -> Result<QueryResult> {
        let graph_name = self
            .current_graph()
            .unwrap_or_else(|| "default".to_string());
        let columns = vec![
            "graph".to_string(),
            "graph_type".to_string(),
            "open".to_string(),
            "node_types".to_string(),
            "edge_types".to_string(),
        ];

        if let Some(type_name) = self.catalog.get_graph_type_binding(&graph_name)
            && let Some(def) = self.catalog.get_graph_type_def(&type_name)
        {
            let rows = vec![vec![
                Value::from(graph_name),
                Value::from(type_name),
                Value::from(def.open),
                Value::from(def.allowed_node_types.join(", ")),
                Value::from(def.allowed_edge_types.join(", ")),
            ]];
            return Ok(QueryResult {
                columns,
                column_types: Vec::new(),
                rows,
                ..QueryResult::empty()
            });
        }

        // No graph type binding found
        Ok(QueryResult {
            columns,
            column_types: Vec::new(),
            rows: vec![vec![
                Value::from(graph_name),
                Value::Null,
                Value::Null,
                Value::Null,
                Value::Null,
            ]],
            ..QueryResult::empty()
        })
    }

    /// Executes a GQL query.
    ///
    /// # Errors
    ///
    /// Returns an error if the query fails to parse or execute.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// use grafeo_engine::GrafeoDB;
    ///
    /// let db = GrafeoDB::new_in_memory();
    /// let session = db.session();
    ///
    /// // Create a node
    /// session.execute("INSERT (:Person {name: 'Alix', age: 30})")?;
    ///
    /// // Query nodes
    /// let result = session.execute("MATCH (n:Person) RETURN n.name, n.age")?;
    /// for row in result.rows() {
    ///     println!("{:?}", row);
    /// }
    /// # Ok(())
    /// # }
    /// ```
    #[cfg(feature = "gql")]
    pub fn execute(&self, query: &str) -> Result<QueryResult> {
        self.require_lpg("GQL")?;

        use crate::query::{
            Executor, binder::Binder, cache::CacheKey, optimizer::Optimizer,
            processor::QueryLanguage, translators::gql,
        };

        let _span = grafeo_info_span!(
            "grafeo::session::execute",
            language = "gql",
            query_len = query.len(),
        );

        #[cfg(not(target_arch = "wasm32"))]
        let start_time = std::time::Instant::now();

        // Parse and translate, checking for session/schema commands first
        let translation = gql::translate_full(query)?;
        let logical_plan = match translation {
            gql::GqlTranslationResult::SessionCommand(cmd) => {
                return self.execute_session_command(cmd);
            }
            #[cfg(feature = "lpg")]
            gql::GqlTranslationResult::SchemaCommand(cmd) => {
                // All DDL requires Admin role
                self.require_permission(crate::auth::StatementKind::Admin)?;
                if *self.read_only_tx.lock() {
                    return Err(grafeo_common::utils::error::Error::Transaction(
                        grafeo_common::utils::error::TransactionError::ReadOnly,
                    ));
                }
                return self.execute_schema_command(cmd);
            }
            gql::GqlTranslationResult::Plan(plan) => {
                // Only walk the operator tree when it matters: non-admin
                // identities need permission checks, read-only transactions
                // need mutation blocking. Admin sessions in auto-commit mode
                // skip the tree walk entirely.
                let read_only = *self.read_only_tx.lock();
                let need_check = read_only || !self.identity.can_admin();
                let is_mutation = need_check && plan.root.has_mutations();
                if is_mutation {
                    self.require_permission(crate::auth::StatementKind::Write)?;
                }
                if read_only && is_mutation {
                    return Err(grafeo_common::utils::error::Error::Transaction(
                        grafeo_common::utils::error::TransactionError::ReadOnly,
                    ));
                }
                plan
            }
            #[cfg(not(feature = "lpg"))]
            gql::GqlTranslationResult::SchemaCommand(_) => {
                return Err(grafeo_common::utils::error::Error::Internal(
                    "Schema commands require the `lpg` feature".to_string(),
                ));
            }
        };

        // Create cache key for this query
        let cache_key = CacheKey::with_graph(query, QueryLanguage::Gql, self.current_graph());

        // Try to get cached optimized plan, or use the plan we just translated
        let optimized_plan = if let Some(cached_plan) = self.query_cache.get_optimized(&cache_key) {
            cached_plan
        } else {
            // Semantic validation
            let mut binder = Binder::new();
            let _binding_context = binder.bind(&logical_plan)?;

            // Optimize the plan
            let active = self.active_store();
            let optimizer = Optimizer::from_graph_store(&*active);
            let plan = optimizer.optimize(logical_plan)?;

            // Cache the optimized plan for future use
            self.query_cache.put_optimized(cache_key, plan.clone());

            plan
        };

        // Resolve the active store for query execution
        let active = self.active_store();

        // EXPLAIN: annotate pushdown hints and return the plan tree
        if optimized_plan.explain {
            use crate::query::processor::{annotate_pushdown_hints, explain_result};
            let mut plan = optimized_plan;
            annotate_pushdown_hints(&mut plan.root, active.as_ref());
            return Ok(explain_result(&plan));
        }

        // PROFILE: execute with per-operator instrumentation
        if optimized_plan.profile {
            let has_mutations = optimized_plan.root.has_mutations();
            return self.with_auto_commit(has_mutations, || {
                let (viewing_epoch, transaction_id) = self.get_transaction_context();
                let planner = self.create_planner_for_store(
                    Arc::clone(&active),
                    viewing_epoch,
                    transaction_id,
                );
                let (mut physical_plan, entries) = planner.plan_profiled(&optimized_plan)?;

                let executor = Executor::with_columns(physical_plan.columns.clone())
                    .with_deadline(self.query_deadline());
                let _result = executor.execute(physical_plan.operator.as_mut())?;

                let total_time_ms;
                #[cfg(not(target_arch = "wasm32"))]
                {
                    total_time_ms = start_time.elapsed().as_secs_f64() * 1000.0;
                }
                #[cfg(target_arch = "wasm32")]
                {
                    total_time_ms = 0.0;
                }

                let profile_tree = crate::query::profile::build_profile_tree(
                    &optimized_plan.root,
                    &mut entries.into_iter(),
                );
                Ok(crate::query::profile::profile_result(
                    &profile_tree,
                    total_time_ms,
                ))
            });
        }

        let has_mutations = optimized_plan.root.has_mutations();

        let result = self.with_auto_commit(has_mutations, || {
            // Get transaction context for MVCC visibility
            let (viewing_epoch, transaction_id) = self.get_transaction_context();

            // Convert to physical plan with transaction context
            // (Physical planning cannot be cached as it depends on transaction state)
            // Safe to use read-only fast path when: this query has no mutations AND
            // there is no active transaction that may have prior uncommitted writes.
            let has_active_tx = self.current_transaction.lock().is_some();
            let read_only = !has_mutations && !has_active_tx;
            let planner = self.create_planner_for_store_with_read_only(
                Arc::clone(&active),
                viewing_epoch,
                transaction_id,
                read_only,
            );
            let mut physical_plan = planner.plan(&optimized_plan)?;

            // Execute the plan
            let executor = Executor::with_columns(physical_plan.columns.clone())
                .with_deadline(self.query_deadline());
            let mut result = executor.execute(physical_plan.operator.as_mut())?;

            // Add execution metrics
            let rows_scanned = result.rows.len() as u64;
            #[cfg(not(target_arch = "wasm32"))]
            {
                let elapsed_ms = start_time.elapsed().as_secs_f64() * 1000.0;
                result.execution_time_ms = Some(elapsed_ms);
            }
            result.rows_scanned = Some(rows_scanned);

            Ok(result)
        });

        // Record metrics for this query execution.
        #[cfg(feature = "metrics")]
        {
            #[cfg(not(target_arch = "wasm32"))]
            let elapsed_ms = Some(start_time.elapsed().as_secs_f64() * 1000.0);
            #[cfg(target_arch = "wasm32")]
            let elapsed_ms = None;
            self.record_query_metrics("gql", elapsed_ms, &result);
        }

        result
    }

    /// Executes a GQL query with visibility at the specified epoch.
    ///
    /// This enables time-travel queries: the query sees the database
    /// as it existed at the given epoch.
    ///
    /// # Errors
    ///
    /// Returns an error if parsing or execution fails.
    #[cfg(feature = "gql")]
    pub fn execute_at_epoch(&self, query: &str, epoch: EpochId) -> Result<QueryResult> {
        let previous = self.viewing_epoch_override.lock().replace(epoch);
        let result = self.execute(query);
        *self.viewing_epoch_override.lock() = previous;
        result
    }

    /// Executes a GQL query at a specific epoch with optional parameters.
    ///
    /// Combines epoch-based time travel with parameterized queries.
    ///
    /// # Errors
    ///
    /// Returns an error if parsing or execution fails.
    #[cfg(feature = "gql")]
    pub fn execute_at_epoch_with_params(
        &self,
        query: &str,
        epoch: EpochId,
        params: Option<std::collections::HashMap<String, Value>>,
    ) -> Result<QueryResult> {
        let previous = self.viewing_epoch_override.lock().replace(epoch);
        let result = if let Some(p) = params {
            self.execute_with_params(query, p)
        } else {
            self.execute(query)
        };
        *self.viewing_epoch_override.lock() = previous;
        result
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
        self.require_lpg("GQL")?;

        use crate::query::processor::{QueryLanguage, QueryProcessor};

        // Reject writes if the identity lacks permission. Parse the query
        // to determine mutation status reliably (the text heuristic has false
        // negatives that could bypass authorization).
        let has_mutations = if self.identity.can_write() {
            // Fast path: identity can write, use heuristic for auto-commit only
            Self::query_looks_like_mutation(query)
        } else {
            // Restricted identity: parse to check mutations reliably
            use crate::query::translators::gql;
            match gql::translate(query) {
                Ok(plan) if plan.root.has_mutations() => {
                    self.require_permission(crate::auth::StatementKind::Write)?;
                    true
                }
                Ok(_) => false,
                // Parse error: let the processor handle it below
                Err(_) => Self::query_looks_like_mutation(query),
            }
        };
        let active = self.active_store();

        self.with_auto_commit(has_mutations, || {
            // Get transaction context for MVCC visibility
            let (viewing_epoch, transaction_id) = self.get_transaction_context();

            // Create processor with transaction context
            let processor = QueryProcessor::for_stores_with_transaction(
                Arc::clone(&active),
                self.active_write_store(),
                Arc::clone(&self.transaction_manager),
            )?;

            // Apply transaction context if in a transaction
            let processor = if let Some(transaction_id) = transaction_id {
                processor.with_transaction_context(viewing_epoch, transaction_id)
            } else {
                processor
            };

            processor.process(query, QueryLanguage::Gql, Some(&params))
        })
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
            Executor, binder::Binder, cache::CacheKey, optimizer::Optimizer,
            processor::QueryLanguage, translators::cypher,
        };

        // Handle schema DDL and SHOW commands before the normal query path
        let translation = cypher::translate_full(query)?;
        match translation {
            #[cfg(feature = "lpg")]
            cypher::CypherTranslationResult::SchemaCommand(cmd) => {
                use grafeo_common::utils::error::{
                    Error as GrafeoError, QueryError, QueryErrorKind,
                };
                self.require_permission(crate::auth::StatementKind::Admin)?;
                if *self.read_only_tx.lock() {
                    return Err(GrafeoError::Query(QueryError::new(
                        QueryErrorKind::Semantic,
                        "Cannot execute schema DDL in a read-only transaction",
                    )));
                }
                return self.execute_schema_command(cmd);
            }
            #[cfg(not(feature = "lpg"))]
            cypher::CypherTranslationResult::SchemaCommand(_) => {
                return Err(grafeo_common::utils::error::Error::Internal(
                    "Schema DDL requires the `lpg` feature".to_string(),
                ));
            }
            cypher::CypherTranslationResult::ShowIndexes => {
                return self.execute_show_indexes();
            }
            cypher::CypherTranslationResult::ShowConstraints => {
                return self.execute_show_constraints();
            }
            cypher::CypherTranslationResult::ShowCurrentGraphType => {
                return self.execute_show_current_graph_type();
            }
            cypher::CypherTranslationResult::Plan(_) => {
                // Fall through to normal execution below
            }
        }

        #[cfg(not(target_arch = "wasm32"))]
        let start_time = std::time::Instant::now();

        // Create cache key for this query
        let cache_key = CacheKey::with_graph(query, QueryLanguage::Cypher, self.current_graph());

        // Try to get cached optimized plan
        let optimized_plan = if let Some(cached_plan) = self.query_cache.get_optimized(&cache_key) {
            cached_plan
        } else {
            // Parse and translate the query to a logical plan
            let logical_plan = cypher::translate(query)?;

            // Semantic validation
            let mut binder = Binder::new();
            let _binding_context = binder.bind(&logical_plan)?;

            // Optimize the plan
            let active = self.active_store();
            let optimizer = Optimizer::from_graph_store(&*active);
            let plan = optimizer.optimize(logical_plan)?;

            // Cache the optimized plan
            self.query_cache.put_optimized(cache_key, plan.clone());

            plan
        };

        // Check role-based permission for mutations
        if optimized_plan.root.has_mutations() {
            self.require_permission(crate::auth::StatementKind::Write)?;
        }

        // Resolve the active store for query execution
        let active = self.active_store();

        // EXPLAIN
        if optimized_plan.explain {
            use crate::query::processor::{annotate_pushdown_hints, explain_result};
            let mut plan = optimized_plan;
            annotate_pushdown_hints(&mut plan.root, active.as_ref());
            return Ok(explain_result(&plan));
        }

        // PROFILE
        if optimized_plan.profile {
            let has_mutations = optimized_plan.root.has_mutations();
            return self.with_auto_commit(has_mutations, || {
                let (viewing_epoch, transaction_id) = self.get_transaction_context();
                let planner = self.create_planner_for_store(
                    Arc::clone(&active),
                    viewing_epoch,
                    transaction_id,
                );
                let (mut physical_plan, entries) = planner.plan_profiled(&optimized_plan)?;

                let executor = Executor::with_columns(physical_plan.columns.clone())
                    .with_deadline(self.query_deadline());
                let _result = executor.execute(physical_plan.operator.as_mut())?;

                let total_time_ms;
                #[cfg(not(target_arch = "wasm32"))]
                {
                    total_time_ms = start_time.elapsed().as_secs_f64() * 1000.0;
                }
                #[cfg(target_arch = "wasm32")]
                {
                    total_time_ms = 0.0;
                }

                let profile_tree = crate::query::profile::build_profile_tree(
                    &optimized_plan.root,
                    &mut entries.into_iter(),
                );
                Ok(crate::query::profile::profile_result(
                    &profile_tree,
                    total_time_ms,
                ))
            });
        }

        let has_mutations = optimized_plan.root.has_mutations();

        let result = self.with_auto_commit(has_mutations, || {
            // Get transaction context for MVCC visibility
            let (viewing_epoch, transaction_id) = self.get_transaction_context();

            // Convert to physical plan with transaction context
            let planner =
                self.create_planner_for_store(Arc::clone(&active), viewing_epoch, transaction_id);
            let mut physical_plan = planner.plan(&optimized_plan)?;

            // Execute the plan
            let executor = Executor::with_columns(physical_plan.columns.clone())
                .with_deadline(self.query_deadline());
            executor.execute(physical_plan.operator.as_mut())
        });

        #[cfg(feature = "metrics")]
        {
            #[cfg(not(target_arch = "wasm32"))]
            let elapsed_ms = Some(start_time.elapsed().as_secs_f64() * 1000.0);
            #[cfg(target_arch = "wasm32")]
            let elapsed_ms = None;
            self.record_query_metrics("cypher", elapsed_ms, &result);
        }

        result
    }

    /// Executes a Gremlin query.
    ///
    /// # Errors
    ///
    /// Returns an error if the query fails to parse or execute.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
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
    /// # Ok(())
    /// # }
    /// ```
    #[cfg(feature = "gremlin")]
    pub fn execute_gremlin(&self, query: &str) -> Result<QueryResult> {
        use crate::query::{Executor, binder::Binder, optimizer::Optimizer, translators::gremlin};

        #[cfg(all(feature = "metrics", not(target_arch = "wasm32")))]
        let start_time = Instant::now();

        // Parse and translate the query to a logical plan
        let logical_plan = gremlin::translate(query)?;

        // Semantic validation
        let mut binder = Binder::new();
        let _binding_context = binder.bind(&logical_plan)?;

        // Optimize the plan
        let active = self.active_store();
        let optimizer = Optimizer::from_graph_store(&*active);
        let optimized_plan = optimizer.optimize(logical_plan)?;

        let has_mutations = optimized_plan.root.has_mutations();
        if has_mutations {
            self.require_permission(crate::auth::StatementKind::Write)?;
        }

        let result = self.with_auto_commit(has_mutations, || {
            // Get transaction context for MVCC visibility
            let (viewing_epoch, transaction_id) = self.get_transaction_context();

            // Convert to physical plan with transaction context
            let planner =
                self.create_planner_for_store(Arc::clone(&active), viewing_epoch, transaction_id);
            let mut physical_plan = planner.plan(&optimized_plan)?;

            // Execute the plan
            let executor = Executor::with_columns(physical_plan.columns.clone())
                .with_deadline(self.query_deadline());
            executor.execute(physical_plan.operator.as_mut())
        });

        #[cfg(feature = "metrics")]
        {
            #[cfg(not(target_arch = "wasm32"))]
            let elapsed_ms = Some(start_time.elapsed().as_secs_f64() * 1000.0);
            #[cfg(target_arch = "wasm32")]
            let elapsed_ms = None;
            self.record_query_metrics("gremlin", elapsed_ms, &result);
        }

        result
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
        use crate::query::{
            Executor, binder::Binder, optimizer::Optimizer, processor::substitute_params,
            translators::gremlin,
        };

        #[cfg(all(feature = "metrics", not(target_arch = "wasm32")))]
        let start_time = Instant::now();

        // Parse and translate the query to a logical plan
        let mut logical_plan = gremlin::translate(query)?;

        // Substitute parameters
        substitute_params(&mut logical_plan, &params)?;

        // Semantic validation
        let mut binder = Binder::new();
        let _binding_context = binder.bind(&logical_plan)?;

        // Optimize the plan
        let active = self.active_store();
        let optimizer = Optimizer::from_graph_store(&*active);
        let optimized_plan = optimizer.optimize(logical_plan)?;

        let has_mutations = optimized_plan.root.has_mutations();
        if has_mutations {
            self.require_permission(crate::auth::StatementKind::Write)?;
        }

        let result = self.with_auto_commit(has_mutations, || {
            let (viewing_epoch, transaction_id) = self.get_transaction_context();
            let planner =
                self.create_planner_for_store(Arc::clone(&active), viewing_epoch, transaction_id);
            let mut physical_plan = planner.plan(&optimized_plan)?;
            let executor = Executor::with_columns(physical_plan.columns.clone())
                .with_deadline(self.query_deadline());
            executor.execute(physical_plan.operator.as_mut())
        });

        #[cfg(feature = "metrics")]
        {
            #[cfg(not(target_arch = "wasm32"))]
            let elapsed_ms = Some(start_time.elapsed().as_secs_f64() * 1000.0);
            #[cfg(target_arch = "wasm32")]
            let elapsed_ms = None;
            self.record_query_metrics("gremlin", elapsed_ms, &result);
        }

        result
    }

    /// Executes a GraphQL query against the LPG store.
    ///
    /// # Errors
    ///
    /// Returns an error if the query fails to parse or execute.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
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
    /// # Ok(())
    /// # }
    /// ```
    #[cfg(feature = "graphql")]
    pub fn execute_graphql(&self, query: &str) -> Result<QueryResult> {
        use crate::query::{
            Executor, binder::Binder, optimizer::Optimizer, processor::substitute_params,
            translators::graphql,
        };

        #[cfg(all(feature = "metrics", not(target_arch = "wasm32")))]
        let start_time = Instant::now();

        let mut logical_plan = graphql::translate(query)?;

        // Substitute default parameter values from variable declarations
        if !logical_plan.default_params.is_empty() {
            let defaults = logical_plan.default_params.clone();
            substitute_params(&mut logical_plan, &defaults)?;
        }

        let mut binder = Binder::new();
        let _binding_context = binder.bind(&logical_plan)?;

        let active = self.active_store();
        let optimizer = Optimizer::from_graph_store(&*active);
        let optimized_plan = optimizer.optimize(logical_plan)?;
        let has_mutations = optimized_plan.root.has_mutations();
        if has_mutations {
            self.require_permission(crate::auth::StatementKind::Write)?;
        }

        let result = self.with_auto_commit(has_mutations, || {
            let (viewing_epoch, transaction_id) = self.get_transaction_context();
            let planner =
                self.create_planner_for_store(Arc::clone(&active), viewing_epoch, transaction_id);
            let mut physical_plan = planner.plan(&optimized_plan)?;
            let executor = Executor::with_columns(physical_plan.columns.clone())
                .with_deadline(self.query_deadline());
            executor.execute(physical_plan.operator.as_mut())
        });

        #[cfg(feature = "metrics")]
        {
            #[cfg(not(target_arch = "wasm32"))]
            let elapsed_ms = Some(start_time.elapsed().as_secs_f64() * 1000.0);
            #[cfg(target_arch = "wasm32")]
            let elapsed_ms = None;
            self.record_query_metrics("graphql", elapsed_ms, &result);
        }

        result
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
        use crate::query::{
            Executor, binder::Binder, optimizer::Optimizer, processor::substitute_params,
            translators::graphql,
        };

        #[cfg(all(feature = "metrics", not(target_arch = "wasm32")))]
        let start_time = Instant::now();

        // Parse and translate the query to a logical plan
        let mut logical_plan = graphql::translate(query)?;

        // Merge default params with caller-supplied params
        if !logical_plan.default_params.is_empty() {
            let mut merged = logical_plan.default_params.clone();
            merged.extend(params.iter().map(|(k, v)| (k.clone(), v.clone())));
            substitute_params(&mut logical_plan, &merged)?;
        } else {
            substitute_params(&mut logical_plan, &params)?;
        }

        // Semantic validation
        let mut binder = Binder::new();
        let _binding_context = binder.bind(&logical_plan)?;

        // Optimize the plan
        let active = self.active_store();
        let optimizer = Optimizer::from_graph_store(&*active);
        let optimized_plan = optimizer.optimize(logical_plan)?;

        let has_mutations = optimized_plan.root.has_mutations();
        if has_mutations {
            self.require_permission(crate::auth::StatementKind::Write)?;
        }

        let result = self.with_auto_commit(has_mutations, || {
            let (viewing_epoch, transaction_id) = self.get_transaction_context();
            let planner =
                self.create_planner_for_store(Arc::clone(&active), viewing_epoch, transaction_id);
            let mut physical_plan = planner.plan(&optimized_plan)?;
            let executor = Executor::with_columns(physical_plan.columns.clone())
                .with_deadline(self.query_deadline());
            executor.execute(physical_plan.operator.as_mut())
        });

        #[cfg(feature = "metrics")]
        {
            #[cfg(not(target_arch = "wasm32"))]
            let elapsed_ms = Some(start_time.elapsed().as_secs_f64() * 1000.0);
            #[cfg(target_arch = "wasm32")]
            let elapsed_ms = None;
            self.record_query_metrics("graphql", elapsed_ms, &result);
        }

        result
    }

    /// Executes a SQL/PGQ query (SQL:2023 GRAPH_TABLE).
    ///
    /// # Errors
    ///
    /// Returns an error if the query fails to parse or execute.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// use grafeo_engine::GrafeoDB;
    ///
    /// let db = GrafeoDB::new_in_memory();
    /// let session = db.session();
    ///
    /// let result = session.execute_sql(
    ///     "SELECT * FROM GRAPH_TABLE (
    ///         MATCH (n:Person)
    ///         COLUMNS (n.name AS name)
    ///     )"
    /// )?;
    /// # Ok(())
    /// # }
    /// ```
    #[cfg(feature = "sql-pgq")]
    pub fn execute_sql(&self, query: &str) -> Result<QueryResult> {
        use crate::query::{
            Executor, binder::Binder, cache::CacheKey, optimizer::Optimizer, plan::LogicalOperator,
            processor::QueryLanguage, translators::sql_pgq,
        };

        #[cfg(all(feature = "metrics", not(target_arch = "wasm32")))]
        let start_time = Instant::now();

        // Parse and translate (always needed to check for DDL)
        let logical_plan = sql_pgq::translate(query)?;

        // Handle DDL statements directly (they don't go through the query pipeline)
        if let LogicalOperator::CreatePropertyGraph(ref cpg) = logical_plan.root {
            self.require_permission(crate::auth::StatementKind::Admin)?;
            return Ok(QueryResult {
                columns: vec!["status".into()],
                column_types: vec![grafeo_common::types::LogicalType::String],
                rows: vec![vec![Value::from(format!(
                    "Property graph '{}' created ({} node tables, {} edge tables)",
                    cpg.name,
                    cpg.node_tables.len(),
                    cpg.edge_tables.len()
                ))]],
                execution_time_ms: None,
                rows_scanned: None,
                status_message: None,
                gql_status: grafeo_common::utils::GqlStatus::SUCCESS,
            });
        }

        let cache_key = CacheKey::with_graph(query, QueryLanguage::SqlPgq, self.current_graph());

        let optimized_plan = if let Some(cached_plan) = self.query_cache.get_optimized(&cache_key) {
            cached_plan
        } else {
            let mut binder = Binder::new();
            let _binding_context = binder.bind(&logical_plan)?;
            let active = self.active_store();
            let optimizer = Optimizer::from_graph_store(&*active);
            let plan = optimizer.optimize(logical_plan)?;
            self.query_cache.put_optimized(cache_key, plan.clone());
            plan
        };

        let active = self.active_store();
        let has_mutations = optimized_plan.root.has_mutations();
        if has_mutations {
            self.require_permission(crate::auth::StatementKind::Write)?;
        }

        let result = self.with_auto_commit(has_mutations, || {
            let (viewing_epoch, transaction_id) = self.get_transaction_context();
            let planner =
                self.create_planner_for_store(Arc::clone(&active), viewing_epoch, transaction_id);
            let mut physical_plan = planner.plan(&optimized_plan)?;
            let executor = Executor::with_columns(physical_plan.columns.clone())
                .with_deadline(self.query_deadline());
            executor.execute(physical_plan.operator.as_mut())
        });

        #[cfg(feature = "metrics")]
        {
            #[cfg(not(target_arch = "wasm32"))]
            let elapsed_ms = Some(start_time.elapsed().as_secs_f64() * 1000.0);
            #[cfg(target_arch = "wasm32")]
            let elapsed_ms = None;
            self.record_query_metrics("sql", elapsed_ms, &result);
        }

        result
    }

    /// Executes a SQL/PGQ query with parameters.
    ///
    /// # Errors
    ///
    /// Returns an error if the query fails to parse or execute.
    #[cfg(feature = "sql-pgq")]
    pub fn execute_sql_with_params(
        &self,
        query: &str,
        params: std::collections::HashMap<String, Value>,
    ) -> Result<QueryResult> {
        use crate::query::processor::{QueryLanguage, QueryProcessor};

        #[cfg(all(feature = "metrics", not(target_arch = "wasm32")))]
        let start_time = Instant::now();

        let has_mutations = if self.identity.can_write() {
            Self::query_looks_like_mutation(query)
        } else {
            use crate::query::translators::sql_pgq;
            match sql_pgq::translate(query) {
                Ok(plan) if plan.root.has_mutations() => {
                    self.require_permission(crate::auth::StatementKind::Write)?;
                    true
                }
                Ok(_) => false,
                Err(_) => Self::query_looks_like_mutation(query),
            }
        };
        if has_mutations {
            self.require_permission(crate::auth::StatementKind::Write)?;
        }
        let active = self.active_store();

        let result = self.with_auto_commit(has_mutations, || {
            let (viewing_epoch, transaction_id) = self.get_transaction_context();
            let processor = QueryProcessor::for_stores_with_transaction(
                Arc::clone(&active),
                self.active_write_store(),
                Arc::clone(&self.transaction_manager),
            )?;
            let processor = if let Some(transaction_id) = transaction_id {
                processor.with_transaction_context(viewing_epoch, transaction_id)
            } else {
                processor
            };
            processor.process(query, QueryLanguage::SqlPgq, Some(&params))
        });

        #[cfg(feature = "metrics")]
        {
            #[cfg(not(target_arch = "wasm32"))]
            let elapsed_ms = Some(start_time.elapsed().as_secs_f64() * 1000.0);
            #[cfg(target_arch = "wasm32")]
            let elapsed_ms = None;
            self.record_query_metrics("sql", elapsed_ms, &result);
        }

        result
    }

    /// Executes a query in the specified language by name.
    ///
    /// Supported language names: `"gql"`, `"cypher"`, `"gremlin"`, `"graphql"`,
    /// `"graphql-rdf"`, `"sparql"`, `"sql"`. Each requires the corresponding feature flag.
    ///
    /// # Errors
    ///
    /// Returns an error if the language is unknown/disabled or the query fails.
    pub fn execute_language(
        &self,
        query: &str,
        language: &str,
        params: Option<std::collections::HashMap<String, Value>>,
    ) -> Result<QueryResult> {
        let _span = grafeo_info_span!(
            "grafeo::session::execute",
            language,
            query_len = query.len(),
        );
        match language {
            "gql" => {
                if let Some(p) = params {
                    self.execute_with_params(query, p)
                } else {
                    self.execute(query)
                }
            }
            #[cfg(feature = "cypher")]
            "cypher" => {
                if let Some(p) = params {
                    use crate::query::processor::{QueryLanguage, QueryProcessor};

                    #[cfg(all(feature = "metrics", not(target_arch = "wasm32")))]
                    let start_time = Instant::now();

                    let has_mutations = if self.identity.can_write() {
                        Self::query_looks_like_mutation(query)
                    } else {
                        use crate::query::translators::cypher;
                        match cypher::translate(query) {
                            Ok(plan) if plan.root.has_mutations() => {
                                self.require_permission(crate::auth::StatementKind::Write)?;
                                true
                            }
                            Ok(_) => false,
                            Err(_) => Self::query_looks_like_mutation(query),
                        }
                    };
                    let active = self.active_store();
                    let result = self.with_auto_commit(has_mutations, || {
                        let processor = QueryProcessor::for_stores_with_transaction(
                            Arc::clone(&active),
                            self.active_write_store(),
                            Arc::clone(&self.transaction_manager),
                        )?;
                        let (viewing_epoch, transaction_id) = self.get_transaction_context();
                        let processor = if let Some(transaction_id) = transaction_id {
                            processor.with_transaction_context(viewing_epoch, transaction_id)
                        } else {
                            processor
                        };
                        processor.process(query, QueryLanguage::Cypher, Some(&p))
                    });

                    #[cfg(feature = "metrics")]
                    {
                        #[cfg(not(target_arch = "wasm32"))]
                        let elapsed_ms = Some(start_time.elapsed().as_secs_f64() * 1000.0);
                        #[cfg(target_arch = "wasm32")]
                        let elapsed_ms = None;
                        self.record_query_metrics("cypher", elapsed_ms, &result);
                    }

                    result
                } else {
                    self.execute_cypher(query)
                }
            }
            #[cfg(feature = "gremlin")]
            "gremlin" => {
                if let Some(p) = params {
                    self.execute_gremlin_with_params(query, p)
                } else {
                    self.execute_gremlin(query)
                }
            }
            #[cfg(feature = "graphql")]
            "graphql" => {
                if let Some(p) = params {
                    self.execute_graphql_with_params(query, p)
                } else {
                    self.execute_graphql(query)
                }
            }
            #[cfg(all(feature = "graphql", feature = "triple-store"))]
            "graphql-rdf" => {
                if let Some(p) = params {
                    self.execute_graphql_rdf_with_params(query, p)
                } else {
                    self.execute_graphql_rdf(query)
                }
            }
            #[cfg(feature = "sql-pgq")]
            "sql" | "sql-pgq" => {
                if let Some(p) = params {
                    self.execute_sql_with_params(query, p)
                } else {
                    self.execute_sql(query)
                }
            }
            #[cfg(all(feature = "sparql", feature = "triple-store"))]
            "sparql" => {
                if let Some(p) = params {
                    self.execute_sparql_with_params(query, p)
                } else {
                    self.execute_sparql(query)
                }
            }
            other => Err(grafeo_common::utils::error::Error::Query(
                grafeo_common::utils::error::QueryError::new(
                    grafeo_common::utils::error::QueryErrorKind::Semantic,
                    format!("Unknown query language: '{other}'"),
                ),
            )),
        }
    }

    /// Begins a new transaction.
    ///
    /// # Errors
    ///
    /// Returns an error if a transaction is already active.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// use grafeo_engine::GrafeoDB;
    ///
    /// let db = GrafeoDB::new_in_memory();
    /// let mut session = db.session();
    ///
    /// session.begin_transaction()?;
    /// session.execute("INSERT (:Person {name: 'Alix'})")?;
    /// session.execute("INSERT (:Person {name: 'Gus'})")?;
    /// session.commit()?; // Both inserts committed atomically
    /// # Ok(())
    /// # }
    /// ```
    /// Clears all cached query plans.
    ///
    /// The plan cache is shared across all sessions on the same database,
    /// so clearing from one session affects all sessions.
    pub fn clear_plan_cache(&self) {
        self.query_cache.clear();
    }

    /// Begins a new transaction on this session.
    ///
    /// Uses the default isolation level (`SnapshotIsolation`).
    ///
    /// # Errors
    ///
    /// Returns an error if a transaction is already active.
    #[cfg(feature = "lpg")]
    pub fn begin_transaction(&mut self) -> Result<()> {
        self.begin_transaction_inner(false, None)
    }

    /// Begins a transaction with a specific isolation level.
    ///
    /// See [`begin_transaction`](Self::begin_transaction) for the default (`SnapshotIsolation`).
    ///
    /// # Errors
    ///
    /// Returns an error if a transaction is already active.
    #[cfg(feature = "lpg")]
    pub fn begin_transaction_with_isolation(
        &mut self,
        isolation_level: crate::transaction::IsolationLevel,
    ) -> Result<()> {
        self.begin_transaction_inner(false, Some(isolation_level))
    }

    /// Core transaction begin logic, usable from both `&mut self` and `&self` paths.
    #[cfg(feature = "lpg")]
    fn begin_transaction_inner(
        &self,
        read_only: bool,
        isolation_level: Option<crate::transaction::IsolationLevel>,
    ) -> Result<()> {
        let _span = grafeo_debug_span!("grafeo::tx::begin", read_only);
        let mut current = self.current_transaction.lock();
        if current.is_some() {
            // Nested transaction: create an auto-savepoint instead of a new tx.
            drop(current);
            let mut depth = self.transaction_nesting_depth.lock();
            *depth += 1;
            let sp_name = format!("_nested_tx_{}", *depth);
            self.savepoint(&sp_name)?;
            return Ok(());
        }

        let active = self.active_lpg_store();
        self.transaction_start_node_count
            .store(active.node_count(), Ordering::Relaxed);
        self.transaction_start_edge_count
            .store(active.edge_count(), Ordering::Relaxed);
        let transaction_id = if let Some(level) = isolation_level {
            self.transaction_manager.begin_with_isolation(level)
        } else {
            self.transaction_manager.begin()
        };
        *current = Some(transaction_id);
        *self.read_only_tx.lock() = read_only || self.db_read_only;

        // Record the initial graph as "touched" for cross-graph atomicity.
        // Uses the full storage key (schema/graph) for schema-scoped resolution.
        let key = self.active_graph_storage_key();
        let mut touched = self.touched_graphs.lock();
        touched.clear();
        touched.push(key);

        #[cfg(feature = "metrics")]
        {
            crate::metrics::record_metric!(self.metrics, tx_active, inc);
            #[cfg(not(target_arch = "wasm32"))]
            {
                *self.tx_start_time.lock() = Some(Instant::now());
            }
        }

        Ok(())
    }

    /// Commits the current transaction.
    ///
    /// Makes all changes since [`begin_transaction`](Self::begin_transaction) permanent.
    ///
    /// # Errors
    ///
    /// Returns an error if no transaction is active.
    #[cfg(feature = "lpg")]
    pub fn commit(&mut self) -> Result<()> {
        self.commit_inner()
    }

    /// Core commit logic, usable from both `&mut self` and `&self` paths.
    #[cfg(feature = "lpg")]
    fn commit_inner(&self) -> Result<()> {
        let _span = grafeo_debug_span!("grafeo::tx::commit");
        // Nested transaction: release the auto-savepoint (changes are preserved).
        {
            let mut depth = self.transaction_nesting_depth.lock();
            if *depth > 0 {
                let sp_name = format!("_nested_tx_{depth}");
                *depth -= 1;
                drop(depth);
                return self.release_savepoint(&sp_name);
            }
        }

        let transaction_id = self.current_transaction.lock().take().ok_or_else(|| {
            grafeo_common::utils::error::Error::Transaction(
                grafeo_common::utils::error::TransactionError::InvalidState(
                    "No active transaction".to_string(),
                ),
            )
        })?;

        // Validate the transaction first (conflict detection) before committing data.
        // If this fails, we rollback the data changes instead of making them permanent.
        let touched = self.touched_graphs.lock().clone();
        let commit_epoch = match self.transaction_manager.commit(transaction_id) {
            Ok(epoch) => epoch,
            Err(e) => {
                // Conflict detected: rollback the data changes
                for graph_name in &touched {
                    let store = self.resolve_store(graph_name);
                    store.rollback_transaction_properties(transaction_id);
                }
                #[cfg(feature = "triple-store")]
                self.rollback_rdf_transaction(transaction_id);
                // Discard buffered CDC events on conflict rollback
                #[cfg(feature = "cdc")]
                if let Some(ref pending) = self.cdc_pending_events {
                    pending.lock().clear();
                }
                *self.read_only_tx.lock() = self.db_read_only;
                self.savepoints.lock().clear();
                self.touched_graphs.lock().clear();
                #[cfg(feature = "metrics")]
                {
                    crate::metrics::record_metric!(self.metrics, tx_active, dec);
                    crate::metrics::record_metric!(self.metrics, tx_conflicts, inc);
                    #[cfg(not(target_arch = "wasm32"))]
                    if let Some(start) = self.tx_start_time.lock().take() {
                        let duration_ms = start.elapsed().as_secs_f64() * 1000.0;
                        crate::metrics::record_metric!(
                            self.metrics,
                            tx_duration,
                            observe duration_ms
                        );
                    }
                }
                return Err(e);
            }
        };

        // Finalize PENDING epochs: make uncommitted versions visible at the commit epoch.
        for graph_name in &touched {
            let store = self.resolve_store(graph_name);
            store.finalize_version_epochs(transaction_id, commit_epoch);
        }

        // Commit succeeded: discard undo logs (make changes permanent)
        #[cfg(feature = "triple-store")]
        self.commit_rdf_transaction(transaction_id);

        for graph_name in &touched {
            let store = self.resolve_store(graph_name);
            store.commit_transaction_properties(transaction_id);
        }

        // Flush buffered CDC events now that the transaction is committed.
        // All buffered events have PENDING epoch; assign the real commit_epoch.
        // Uses record_batch to acquire the write lock once per commit.
        #[cfg(feature = "cdc")]
        if let Some(ref pending) = self.cdc_pending_events {
            let events: Vec<crate::cdc::ChangeEvent> = pending.lock().drain(..).collect();
            self.cdc_log.record_batch(events.into_iter().map(|mut e| {
                e.epoch = commit_epoch;
                e
            }));
        }

        // Log transaction commit and epoch advance to WAL so that crash
        // recovery can identify committed transactions and their epoch
        // boundaries. Without these markers, WAL recovery discards all
        // records as uncommitted (fixes #252 for the crash scenario).
        #[cfg(feature = "wal")]
        if let Some(ref wal) = self.wal {
            use grafeo_storage::wal::WalRecord;
            if let Err(e) = wal.log(&WalRecord::TransactionCommit { transaction_id }) {
                grafeo_warn!("Failed to log transaction commit to WAL: {}", e);
            }
            if let Err(e) = wal.log(&WalRecord::EpochAdvance {
                epoch: commit_epoch,
            }) {
                grafeo_warn!("Failed to log epoch advance to WAL: {}", e);
            }
        }

        // Sync epoch for all touched graphs so that convenience lookups
        // (edge_type, get_edge, get_node) can see versions at the latest epoch.
        let current_epoch = self.transaction_manager.current_epoch();
        for graph_name in &touched {
            let store = self.resolve_store(graph_name);
            store.sync_epoch(current_epoch);
        }

        // Reset read-only flag, clear savepoints and touched graphs
        *self.read_only_tx.lock() = self.db_read_only;
        self.savepoints.lock().clear();
        self.touched_graphs.lock().clear();

        // Auto-GC: periodically prune old MVCC versions
        if self.gc_interval > 0 {
            let count = self.commit_counter.fetch_add(1, Ordering::Relaxed) + 1;
            if count.is_multiple_of(self.gc_interval) {
                let min_epoch = self.transaction_manager.min_active_epoch();
                for graph_name in &touched {
                    let store = self.resolve_store(graph_name);
                    store.gc_versions(min_epoch);
                }
                self.transaction_manager.gc();
                #[cfg(feature = "metrics")]
                crate::metrics::record_metric!(self.metrics, gc_runs, inc);
            }
        }

        #[cfg(feature = "metrics")]
        {
            crate::metrics::record_metric!(self.metrics, tx_active, dec);
            crate::metrics::record_metric!(self.metrics, tx_committed, inc);
            #[cfg(not(target_arch = "wasm32"))]
            if let Some(start) = self.tx_start_time.lock().take() {
                let duration_ms = start.elapsed().as_secs_f64() * 1000.0;
                crate::metrics::record_metric!(self.metrics, tx_duration, observe duration_ms);
            }
        }

        Ok(())
    }

    /// Aborts the current transaction.
    ///
    /// Discards all changes since [`begin_transaction`](Self::begin_transaction).
    ///
    /// # Errors
    ///
    /// Returns an error if no transaction is active.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// use grafeo_engine::GrafeoDB;
    ///
    /// let db = GrafeoDB::new_in_memory();
    /// let mut session = db.session();
    ///
    /// session.begin_transaction()?;
    /// session.execute("INSERT (:Person {name: 'Alix'})")?;
    /// session.rollback()?; // Insert is discarded
    /// # Ok(())
    /// # }
    /// ```
    #[cfg(feature = "lpg")]
    pub fn rollback(&mut self) -> Result<()> {
        self.rollback_inner()
    }

    /// Core rollback logic, usable from both `&mut self` and `&self` paths.
    #[cfg(feature = "lpg")]
    fn rollback_inner(&self) -> Result<()> {
        let _span = grafeo_debug_span!("grafeo::tx::rollback");
        // Nested transaction: rollback to the auto-savepoint.
        {
            let mut depth = self.transaction_nesting_depth.lock();
            if *depth > 0 {
                let sp_name = format!("_nested_tx_{depth}");
                *depth -= 1;
                drop(depth);
                return self.rollback_to_savepoint(&sp_name);
            }
        }

        let transaction_id = self.current_transaction.lock().take().ok_or_else(|| {
            grafeo_common::utils::error::Error::Transaction(
                grafeo_common::utils::error::TransactionError::InvalidState(
                    "No active transaction".to_string(),
                ),
            )
        })?;

        // Reset read-only flag
        *self.read_only_tx.lock() = self.db_read_only;

        // Discard uncommitted versions in ALL touched LPG stores (cross-graph atomicity).
        let touched = self.touched_graphs.lock().clone();
        for graph_name in &touched {
            let store = self.resolve_store(graph_name);
            store.discard_uncommitted_versions(transaction_id);
        }

        // Discard pending operations in the RDF store
        #[cfg(feature = "triple-store")]
        self.rollback_rdf_transaction(transaction_id);

        // Discard buffered CDC events on rollback
        #[cfg(feature = "cdc")]
        if let Some(ref pending) = self.cdc_pending_events {
            pending.lock().clear();
        }

        // Clear savepoints and touched graphs
        self.savepoints.lock().clear();
        self.touched_graphs.lock().clear();

        // Mark transaction as aborted in the manager
        let result = self.transaction_manager.abort(transaction_id);

        #[cfg(feature = "metrics")]
        if result.is_ok() {
            crate::metrics::record_metric!(self.metrics, tx_active, dec);
            crate::metrics::record_metric!(self.metrics, tx_rolled_back, inc);
            #[cfg(not(target_arch = "wasm32"))]
            if let Some(start) = self.tx_start_time.lock().take() {
                let duration_ms = start.elapsed().as_secs_f64() * 1000.0;
                crate::metrics::record_metric!(self.metrics, tx_duration, observe duration_ms);
            }
        }

        result
    }

    /// Creates a named savepoint within the current transaction.
    ///
    /// The savepoint captures the current node/edge ID counters so that
    /// [`rollback_to_savepoint`](Self::rollback_to_savepoint) can discard
    /// entities created after this point.
    ///
    /// # Errors
    ///
    /// Returns an error if no transaction is active.
    #[cfg(feature = "lpg")]
    pub fn savepoint(&self, name: &str) -> Result<()> {
        let tx_id = self.current_transaction.lock().ok_or_else(|| {
            grafeo_common::utils::error::Error::Transaction(
                grafeo_common::utils::error::TransactionError::InvalidState(
                    "No active transaction".to_string(),
                ),
            )
        })?;

        // Capture state for every graph touched so far.
        let touched = self.touched_graphs.lock().clone();
        let graph_snapshots: Vec<GraphSavepoint> = touched
            .iter()
            .map(|graph_name| {
                let store = self.resolve_store(graph_name);
                GraphSavepoint {
                    graph_name: graph_name.clone(),
                    next_node_id: store.peek_next_node_id(),
                    next_edge_id: store.peek_next_edge_id(),
                    undo_log_position: store.property_undo_log_position(tx_id),
                }
            })
            .collect();

        self.savepoints.lock().push(SavepointState {
            name: name.to_string(),
            graph_snapshots,
            active_graph: self.current_graph.lock().clone(),
            #[cfg(feature = "cdc")]
            cdc_event_position: self
                .cdc_pending_events
                .as_ref()
                .map_or(0, |p| p.lock().len()),
        });
        Ok(())
    }

    /// Rolls back to a named savepoint, undoing all writes made after it.
    ///
    /// The savepoint and any savepoints created after it are removed.
    /// Entities with IDs >= the savepoint snapshot are discarded.
    ///
    /// # Errors
    ///
    /// Returns an error if no transaction is active or the savepoint does not exist.
    #[cfg(feature = "lpg")]
    pub fn rollback_to_savepoint(&self, name: &str) -> Result<()> {
        let transaction_id = self.current_transaction.lock().ok_or_else(|| {
            grafeo_common::utils::error::Error::Transaction(
                grafeo_common::utils::error::TransactionError::InvalidState(
                    "No active transaction".to_string(),
                ),
            )
        })?;

        let mut savepoints = self.savepoints.lock();

        // Find the savepoint by name (search from the end for nested savepoints)
        let pos = savepoints
            .iter()
            .rposition(|sp| sp.name == name)
            .ok_or_else(|| {
                grafeo_common::utils::error::Error::Transaction(
                    grafeo_common::utils::error::TransactionError::InvalidState(format!(
                        "Savepoint '{name}' not found"
                    )),
                )
            })?;

        let sp_state = savepoints[pos].clone();

        // Remove this savepoint and all later ones
        savepoints.truncate(pos);
        drop(savepoints);

        // Roll back each graph that was captured in the savepoint.
        for gs in &sp_state.graph_snapshots {
            let store = self.resolve_store(&gs.graph_name);

            // Replay property/label undo entries recorded after the savepoint
            store.rollback_transaction_properties_to(transaction_id, gs.undo_log_position);

            // Discard entities created after the savepoint
            let current_next_node = store.peek_next_node_id();
            let current_next_edge = store.peek_next_edge_id();

            let node_ids: Vec<NodeId> = (gs.next_node_id..current_next_node)
                .map(NodeId::new)
                .collect();
            let edge_ids: Vec<EdgeId> = (gs.next_edge_id..current_next_edge)
                .map(EdgeId::new)
                .collect();

            if !node_ids.is_empty() || !edge_ids.is_empty() {
                store.discard_entities_by_id(transaction_id, &node_ids, &edge_ids);
            }
        }

        // Also roll back any graphs that were touched AFTER the savepoint
        // but not captured in it. These need full discard since the savepoint
        // didn't include them.
        let touched = self.touched_graphs.lock().clone();
        for graph_name in &touched {
            let already_captured = sp_state
                .graph_snapshots
                .iter()
                .any(|gs| gs.graph_name == *graph_name);
            if !already_captured {
                let store = self.resolve_store(graph_name);
                store.discard_uncommitted_versions(transaction_id);
            }
        }

        // Truncate CDC event buffer to the savepoint position.
        #[cfg(feature = "cdc")]
        if let Some(ref pending) = self.cdc_pending_events {
            pending.lock().truncate(sp_state.cdc_event_position);
        }

        // Restore touched_graphs to only the graphs that were known at savepoint time.
        let mut touched = self.touched_graphs.lock();
        touched.clear();
        for gs in &sp_state.graph_snapshots {
            if !touched.contains(&gs.graph_name) {
                touched.push(gs.graph_name.clone());
            }
        }

        Ok(())
    }

    /// Releases (removes) a named savepoint without rolling back.
    ///
    /// # Errors
    ///
    /// Returns an error if no transaction is active or the savepoint does not exist.
    pub fn release_savepoint(&self, name: &str) -> Result<()> {
        let _tx_id = self.current_transaction.lock().ok_or_else(|| {
            grafeo_common::utils::error::Error::Transaction(
                grafeo_common::utils::error::TransactionError::InvalidState(
                    "No active transaction".to_string(),
                ),
            )
        })?;

        let mut savepoints = self.savepoints.lock();
        let pos = savepoints
            .iter()
            .rposition(|sp| sp.name == name)
            .ok_or_else(|| {
                grafeo_common::utils::error::Error::Transaction(
                    grafeo_common::utils::error::TransactionError::InvalidState(format!(
                        "Savepoint '{name}' not found"
                    )),
                )
            })?;
        savepoints.remove(pos);
        Ok(())
    }

    /// Returns whether a transaction is active.
    #[must_use]
    pub fn in_transaction(&self) -> bool {
        self.current_transaction.lock().is_some()
    }

    /// Returns the current transaction ID, if any.
    #[must_use]
    pub(crate) fn current_transaction_id(&self) -> Option<TransactionId> {
        *self.current_transaction.lock()
    }

    /// Returns a reference to the transaction manager.
    #[must_use]
    pub(crate) fn transaction_manager(&self) -> &TransactionManager {
        &self.transaction_manager
    }

    /// Returns the store's current node count and the count at transaction start.
    #[cfg(feature = "lpg")]
    #[must_use]
    pub(crate) fn node_count_delta(&self) -> (usize, usize) {
        (
            self.transaction_start_node_count.load(Ordering::Relaxed),
            self.active_lpg_store().node_count(),
        )
    }

    /// Returns the store's current edge count and the count at transaction start.
    #[cfg(feature = "lpg")]
    #[must_use]
    pub(crate) fn edge_count_delta(&self) -> (usize, usize) {
        (
            self.transaction_start_edge_count.load(Ordering::Relaxed),
            self.active_lpg_store().edge_count(),
        )
    }

    /// Prepares the current transaction for a two-phase commit.
    ///
    /// Returns a [`PreparedCommit`](crate::transaction::PreparedCommit) that
    /// lets you inspect pending changes and attach metadata before finalizing.
    /// The mutable borrow prevents concurrent operations while the commit is
    /// pending.
    ///
    /// If the `PreparedCommit` is dropped without calling `commit()` or
    /// `abort()`, the transaction is automatically rolled back.
    ///
    /// # Errors
    ///
    /// Returns an error if no transaction is active.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// use grafeo_engine::GrafeoDB;
    ///
    /// let db = GrafeoDB::new_in_memory();
    /// let mut session = db.session();
    ///
    /// session.begin_transaction()?;
    /// session.execute("INSERT (:Person {name: 'Alix'})")?;
    ///
    /// let mut prepared = session.prepare_commit()?;
    /// println!("Nodes written: {}", prepared.info().nodes_written);
    /// prepared.set_metadata("audit_user", "admin");
    /// prepared.commit()?;
    /// # Ok(())
    /// # }
    /// ```
    #[cfg(feature = "lpg")]
    pub fn prepare_commit(&mut self) -> Result<crate::transaction::PreparedCommit<'_>> {
        crate::transaction::PreparedCommit::new(self)
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

    /// Returns `true` if auto-commit should wrap this execution.
    ///
    /// Auto-commit kicks in when: the session is in auto-commit mode,
    /// no explicit transaction is active, and the query mutates data.
    fn needs_auto_commit(&self, has_mutations: bool) -> bool {
        self.auto_commit && has_mutations && self.current_transaction.lock().is_none()
    }

    /// Wraps `body` in an automatic begin/commit when [`needs_auto_commit`]
    /// returns `true`. On error the transaction is rolled back.
    #[cfg(feature = "lpg")]
    fn with_auto_commit<F>(&self, has_mutations: bool, body: F) -> Result<QueryResult>
    where
        F: FnOnce() -> Result<QueryResult>,
    {
        if self.needs_auto_commit(has_mutations) {
            self.begin_transaction_inner(false, None)?;
            match body() {
                Ok(result) => {
                    self.commit_inner()?;
                    Ok(result)
                }
                Err(e) => {
                    let _ = self.rollback_inner();
                    Err(e)
                }
            }
        } else {
            body()
        }
    }

    /// Non-LPG stub: no auto-commit wrapping (SPARQL UPDATE is atomic).
    #[cfg(not(feature = "lpg"))]
    fn with_auto_commit<F>(&self, _has_mutations: bool, body: F) -> Result<QueryResult>
    where
        F: FnOnce() -> Result<QueryResult>,
    {
        body()
    }

    /// Quick heuristic: returns `true` when the query text looks like it
    /// performs a mutation. Used by `_with_params` paths that go through the
    /// `QueryProcessor` (where the logical plan isn't available before
    /// execution). False negatives are harmless: the data just won't be
    /// auto-committed, which matches the prior behaviour.
    fn query_looks_like_mutation(query: &str) -> bool {
        let upper = query.to_ascii_uppercase();
        upper.contains("INSERT")
            || upper.contains("CREATE")
            || upper.contains("DELETE")
            || upper.contains("MERGE")
            || upper.contains("SET")
            || upper.contains("REMOVE")
            || upper.contains("DROP")
            || upper.contains("ALTER")
    }

    /// Computes the wall-clock deadline for query execution.
    #[must_use]
    fn query_deadline(&self) -> Option<Instant> {
        #[cfg(not(target_arch = "wasm32"))]
        {
            self.query_timeout.map(|d| Instant::now() + d)
        }
        #[cfg(target_arch = "wasm32")]
        {
            let _ = &self.query_timeout;
            None
        }
    }

    /// Records query metrics for any language.
    ///
    /// Called after query execution to update counters, latency histogram,
    /// and per-language tracking. `elapsed_ms` should be `None` on WASM
    /// where `Instant` is unavailable.
    #[cfg(feature = "metrics")]
    fn record_query_metrics(
        &self,
        language: &str,
        elapsed_ms: Option<f64>,
        result: &Result<crate::database::QueryResult>,
    ) {
        use crate::metrics::record_metric;

        record_metric!(self.metrics, query_count, inc);
        if let Some(ref reg) = self.metrics {
            reg.query_count_by_language.increment(language);
        }
        if let Some(ms) = elapsed_ms {
            record_metric!(self.metrics, query_latency, observe ms);
        }
        match result {
            Ok(r) => {
                let returned = r.rows.len() as u64;
                record_metric!(self.metrics, rows_returned, add returned);
                if let Some(scanned) = r.rows_scanned {
                    record_metric!(self.metrics, rows_scanned, add scanned);
                }
            }
            Err(e) => {
                record_metric!(self.metrics, query_errors, inc);
                // Detect timeout errors
                let msg = e.to_string();
                if msg.contains("exceeded timeout") {
                    record_metric!(self.metrics, query_timeouts, inc);
                }
            }
        }
    }

    /// Evaluates a simple integer literal from a session parameter expression.
    fn eval_integer_literal(expr: &grafeo_adapters::query::gql::ast::Expression) -> Option<i64> {
        use grafeo_adapters::query::gql::ast::{Expression, Literal};
        match expr {
            Expression::Literal(Literal::Integer(n)) => Some(*n),
            _ => None,
        }
    }

    /// Returns the current transaction context for MVCC visibility.
    ///
    /// Returns `(viewing_epoch, transaction_id)` where:
    /// - `viewing_epoch` is the epoch at which to check version visibility
    /// - `transaction_id` is the current transaction ID (if in a transaction)
    #[must_use]
    fn get_transaction_context(&self) -> (EpochId, Option<TransactionId>) {
        // Time-travel override takes precedence (read-only, no tx context)
        if let Some(epoch) = *self.viewing_epoch_override.lock() {
            return (epoch, None);
        }

        if let Some(transaction_id) = *self.current_transaction.lock() {
            // In a transaction: use the transaction's start epoch
            let epoch = self
                .transaction_manager
                .start_epoch(transaction_id)
                .unwrap_or_else(|| self.transaction_manager.current_epoch());
            (epoch, Some(transaction_id))
        } else {
            // No transaction: use current epoch
            (self.transaction_manager.current_epoch(), None)
        }
    }

    /// Creates a planner with transaction context and constraint validator.
    ///
    /// The `store` parameter is the graph store to plan against (use
    /// `self.active_store()` for graph-aware execution).
    fn create_planner_for_store(
        &self,
        store: Arc<dyn GraphStore>,
        viewing_epoch: EpochId,
        transaction_id: Option<TransactionId>,
    ) -> crate::query::Planner {
        self.create_planner_for_store_with_read_only(store, viewing_epoch, transaction_id, false)
    }

    fn create_planner_for_store_with_read_only(
        &self,
        store: Arc<dyn GraphStore>,
        viewing_epoch: EpochId,
        transaction_id: Option<TransactionId>,
        read_only: bool,
    ) -> crate::query::Planner {
        use crate::query::Planner;
        use grafeo_core::execution::operators::{LazyValue, SessionContext};

        // Capture store reference for lazy introspection (only computed if info()/schema() called).
        let info_store = Arc::clone(&store);
        let schema_store = Arc::clone(&store);

        let session_context = SessionContext {
            current_schema: self.current_schema(),
            current_graph: self.current_graph(),
            db_info: LazyValue::new(move || Self::build_info_value(&*info_store)),
            schema_info: LazyValue::new(move || Self::build_schema_value(&*schema_store)),
        };

        let write_store = self.active_write_store();

        let mut planner = Planner::with_context(
            Arc::clone(&store),
            write_store,
            Arc::clone(&self.transaction_manager),
            transaction_id,
            viewing_epoch,
        )
        .with_factorized_execution(self.factorized_execution)
        .with_catalog(Arc::clone(&self.catalog))
        .with_session_context(session_context)
        .with_read_only(read_only);

        // Attach the constraint validator for schema enforcement
        let validator =
            CatalogConstraintValidator::new(Arc::clone(&self.catalog)).with_store(store);
        planner = planner.with_validator(Arc::new(validator));

        planner
    }

    /// Builds a `Value::Map` for the `info()` introspection function.
    fn build_info_value(store: &dyn GraphStore) -> Value {
        use grafeo_common::types::PropertyKey;
        use std::collections::BTreeMap;

        let mut map = BTreeMap::new();
        map.insert(PropertyKey::from("mode"), Value::String("lpg".into()));
        map.insert(
            PropertyKey::from("node_count"),
            Value::Int64(store.node_count() as i64),
        );
        map.insert(
            PropertyKey::from("edge_count"),
            Value::Int64(store.edge_count() as i64),
        );
        map.insert(
            PropertyKey::from("version"),
            Value::String(env!("CARGO_PKG_VERSION").into()),
        );
        Value::Map(map.into())
    }

    /// Builds a `Value::Map` for the `schema()` introspection function.
    fn build_schema_value(store: &dyn GraphStore) -> Value {
        use grafeo_common::types::PropertyKey;
        use std::collections::BTreeMap;

        let labels: Vec<Value> = store
            .all_labels()
            .into_iter()
            .map(|l| Value::String(l.into()))
            .collect();
        let edge_types: Vec<Value> = store
            .all_edge_types()
            .into_iter()
            .map(|t| Value::String(t.into()))
            .collect();
        let property_keys: Vec<Value> = store
            .all_property_keys()
            .into_iter()
            .map(|k| Value::String(k.into()))
            .collect();

        let mut map = BTreeMap::new();
        map.insert(PropertyKey::from("labels"), Value::List(labels.into()));
        map.insert(
            PropertyKey::from("edge_types"),
            Value::List(edge_types.into()),
        );
        map.insert(
            PropertyKey::from("property_keys"),
            Value::List(property_keys.into()),
        );
        Value::Map(map.into())
    }

    /// Creates a node directly (bypassing query execution).
    ///
    /// This is a low-level API for testing and direct manipulation.
    /// If a transaction is active, the node will be versioned with the transaction ID.
    #[cfg(feature = "lpg")]
    pub fn create_node(&self, labels: &[&str]) -> NodeId {
        let (epoch, transaction_id) = self.get_transaction_context();
        self.active_lpg_store().create_node_versioned(
            labels,
            epoch,
            transaction_id.unwrap_or(TransactionId::SYSTEM),
        )
    }

    /// Creates a node with properties.
    ///
    /// If a transaction is active, the node will be versioned with the transaction ID.
    #[cfg(feature = "lpg")]
    pub fn create_node_with_props<'a>(
        &self,
        labels: &[&str],
        properties: impl IntoIterator<Item = (&'a str, Value)>,
    ) -> NodeId {
        let (epoch, transaction_id) = self.get_transaction_context();
        self.active_lpg_store().create_node_with_props_versioned(
            labels,
            properties,
            epoch,
            transaction_id.unwrap_or(TransactionId::SYSTEM),
        )
    }

    /// Creates an edge between two nodes.
    ///
    /// This is a low-level API for testing and direct manipulation.
    /// If a transaction is active, the edge will be versioned with the transaction ID.
    #[cfg(feature = "lpg")]
    pub fn create_edge(
        &self,
        src: NodeId,
        dst: NodeId,
        edge_type: &str,
    ) -> grafeo_common::types::EdgeId {
        let (epoch, transaction_id) = self.get_transaction_context();
        self.active_lpg_store().create_edge_versioned(
            src,
            dst,
            edge_type,
            epoch,
            transaction_id.unwrap_or(TransactionId::SYSTEM),
        )
    }

    /// Creates an edge with properties within the active transaction context.
    #[cfg(feature = "lpg")]
    pub fn create_edge_with_props<'a>(
        &self,
        src: NodeId,
        dst: NodeId,
        edge_type: &str,
        properties: impl IntoIterator<Item = (&'a str, Value)>,
    ) -> grafeo_common::types::EdgeId {
        let (epoch, transaction_id) = self.get_transaction_context();
        let tid = transaction_id.unwrap_or(TransactionId::SYSTEM);
        let store = self.active_lpg_store();
        let eid = store.create_edge_versioned(src, dst, edge_type, epoch, tid);
        for (key, value) in properties {
            store.set_edge_property_versioned(eid, key, value, tid);
        }
        eid
    }

    /// Sets a node property within the active transaction context.
    #[cfg(feature = "lpg")]
    pub fn set_node_property(&self, id: NodeId, key: &str, value: Value) {
        let (_, transaction_id) = self.get_transaction_context();
        if let Some(tid) = transaction_id {
            self.active_lpg_store()
                .set_node_property_versioned(id, key, value, tid);
        } else {
            self.active_lpg_store().set_node_property(id, key, value);
        }
    }

    /// Sets an edge property within the active transaction context.
    #[cfg(feature = "lpg")]
    pub fn set_edge_property(&self, id: grafeo_common::types::EdgeId, key: &str, value: Value) {
        let (_, transaction_id) = self.get_transaction_context();
        if let Some(tid) = transaction_id {
            self.active_lpg_store()
                .set_edge_property_versioned(id, key, value, tid);
        } else {
            self.active_lpg_store().set_edge_property(id, key, value);
        }
    }

    /// Deletes a node within the active transaction context.
    #[cfg(feature = "lpg")]
    pub fn delete_node(&self, id: NodeId) -> bool {
        let (epoch, transaction_id) = self.get_transaction_context();
        if let Some(tid) = transaction_id {
            self.active_lpg_store()
                .delete_node_versioned(id, epoch, tid)
        } else {
            self.active_lpg_store().delete_node(id)
        }
    }

    /// Deletes an edge within the active transaction context.
    #[cfg(feature = "lpg")]
    pub fn delete_edge(&self, id: grafeo_common::types::EdgeId) -> bool {
        let (epoch, transaction_id) = self.get_transaction_context();
        if let Some(tid) = transaction_id {
            self.active_lpg_store()
                .delete_edge_versioned(id, epoch, tid)
        } else {
            self.active_lpg_store().delete_edge(id)
        }
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
    /// ```no_run
    /// # use grafeo_engine::GrafeoDB;
    /// # let db = GrafeoDB::new_in_memory();
    /// let session = db.session();
    /// let node_id = session.create_node(&["Person"]);
    ///
    /// // Direct lookup - O(1), no query planning
    /// let node = session.get_node(node_id);
    /// assert!(node.is_some());
    /// ```
    #[cfg(feature = "lpg")]
    #[must_use]
    pub fn get_node(&self, id: NodeId) -> Option<Node> {
        let (epoch, transaction_id) = self.get_transaction_context();
        self.active_lpg_store().get_node_versioned(
            id,
            epoch,
            transaction_id.unwrap_or(TransactionId::SYSTEM),
        )
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
    /// ```no_run
    /// # use grafeo_engine::GrafeoDB;
    /// # use grafeo_common::types::Value;
    /// # let db = GrafeoDB::new_in_memory();
    /// let session = db.session();
    /// let id = session.create_node_with_props(&["Person"], [("name", "Alix".into())]);
    ///
    /// // Direct property access - O(1)
    /// let name = session.get_node_property(id, "name");
    /// assert_eq!(name, Some(Value::String("Alix".into())));
    /// ```
    #[cfg(feature = "lpg")]
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
    #[cfg(feature = "lpg")]
    #[must_use]
    pub fn get_edge(&self, id: EdgeId) -> Option<Edge> {
        let (epoch, transaction_id) = self.get_transaction_context();
        self.active_lpg_store().get_edge_versioned(
            id,
            epoch,
            transaction_id.unwrap_or(TransactionId::SYSTEM),
        )
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
    /// ```no_run
    /// # use grafeo_engine::GrafeoDB;
    /// # let db = GrafeoDB::new_in_memory();
    /// let session = db.session();
    /// let alix = session.create_node(&["Person"]);
    /// let gus = session.create_node(&["Person"]);
    /// session.create_edge(alix, gus, "KNOWS");
    ///
    /// // Direct neighbor lookup - O(degree)
    /// let neighbors = session.get_neighbors_outgoing(alix);
    /// assert_eq!(neighbors.len(), 1);
    /// assert_eq!(neighbors[0].0, gus);
    /// ```
    #[cfg(feature = "lpg")]
    #[must_use]
    pub fn get_neighbors_outgoing(&self, node: NodeId) -> Vec<(NodeId, EdgeId)> {
        self.active_lpg_store()
            .edges_from(node, Direction::Outgoing)
            .collect()
    }

    /// Gets incoming neighbors of a node directly, bypassing query planning.
    ///
    /// Returns (neighbor_id, edge_id) pairs for all incoming edges.
    ///
    /// # Performance
    ///
    /// - Time complexity: O(degree) where degree is the number of incoming edges
    /// - Uses backward adjacency index for direct access
    #[cfg(feature = "lpg")]
    #[must_use]
    pub fn get_neighbors_incoming(&self, node: NodeId) -> Vec<(NodeId, EdgeId)> {
        self.active_lpg_store()
            .edges_from(node, Direction::Incoming)
            .collect()
    }

    /// Gets outgoing neighbors filtered by edge type, bypassing query planning.
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use grafeo_engine::GrafeoDB;
    /// # let db = GrafeoDB::new_in_memory();
    /// # let session = db.session();
    /// # let alix = session.create_node(&["Person"]);
    /// let neighbors = session.get_neighbors_outgoing_by_type(alix, "KNOWS");
    /// ```
    #[cfg(feature = "lpg")]
    #[must_use]
    pub fn get_neighbors_outgoing_by_type(
        &self,
        node: NodeId,
        edge_type: &str,
    ) -> Vec<(NodeId, EdgeId)> {
        self.active_lpg_store()
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
    #[cfg(feature = "lpg")]
    #[must_use]
    pub fn node_exists(&self, id: NodeId) -> bool {
        self.get_node(id).is_some()
    }

    /// Checks if an edge exists, bypassing query planning.
    #[cfg(feature = "lpg")]
    #[must_use]
    pub fn edge_exists(&self, id: EdgeId) -> bool {
        self.get_edge(id).is_some()
    }

    /// Gets the degree (number of edges) of a node.
    ///
    /// Returns (outgoing_degree, incoming_degree).
    #[cfg(feature = "lpg")]
    #[must_use]
    pub fn get_degree(&self, node: NodeId) -> (usize, usize) {
        let active = self.active_lpg_store();
        let out = active.out_degree(node);
        let in_degree = active.in_degree(node);
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
    #[cfg(feature = "lpg")]
    #[must_use]
    pub fn get_nodes_batch(&self, ids: &[NodeId]) -> Vec<Option<Node>> {
        let (epoch, transaction_id) = self.get_transaction_context();
        let tx = transaction_id.unwrap_or(TransactionId::SYSTEM);
        let active = self.active_lpg_store();
        ids.iter()
            .map(|&id| active.get_node_versioned(id, epoch, tx))
            .collect()
    }

    // ── Change Data Capture ─────────────────────────────────────────────

    /// Returns the full change history for an entity (node or edge).
    ///
    /// # Errors
    ///
    /// Currently infallible, but returns `Result` for forward compatibility.
    #[cfg(feature = "cdc")]
    pub fn history(
        &self,
        entity_id: impl Into<crate::cdc::EntityId>,
    ) -> Result<Vec<crate::cdc::ChangeEvent>> {
        Ok(self.cdc_log.history(entity_id.into()))
    }

    /// Returns change events for an entity since the given epoch.
    ///
    /// # Errors
    ///
    /// Currently infallible, but returns `Result` for forward compatibility.
    #[cfg(feature = "cdc")]
    pub fn history_since(
        &self,
        entity_id: impl Into<crate::cdc::EntityId>,
        since_epoch: EpochId,
    ) -> Result<Vec<crate::cdc::ChangeEvent>> {
        Ok(self.cdc_log.history_since(entity_id.into(), since_epoch))
    }

    /// Returns all change events across all entities in an epoch range.
    ///
    /// # Errors
    ///
    /// Currently infallible, but returns `Result` for forward compatibility.
    #[cfg(feature = "cdc")]
    pub fn changes_between(
        &self,
        start_epoch: EpochId,
        end_epoch: EpochId,
    ) -> Result<Vec<crate::cdc::ChangeEvent>> {
        Ok(self.cdc_log.changes_between(start_epoch, end_epoch))
    }
}

impl Drop for Session {
    fn drop(&mut self) {
        // Auto-rollback any active transaction to prevent leaked MVCC state,
        // dangling write locks, and uncommitted versions lingering in the store.
        #[cfg(feature = "lpg")]
        if self.in_transaction() {
            let _ = self.rollback_inner();
        }

        #[cfg(feature = "metrics")]
        if let Some(ref reg) = self.metrics {
            reg.session_active
                .fetch_sub(1, std::sync::atomic::Ordering::Relaxed);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::parse_default_literal;
    use crate::database::GrafeoDB;
    use grafeo_common::types::Value;

    // -----------------------------------------------------------------------
    // parse_default_literal
    // -----------------------------------------------------------------------

    #[test]
    fn parse_default_literal_null() {
        assert_eq!(parse_default_literal("null"), Value::Null);
        assert_eq!(parse_default_literal("NULL"), Value::Null);
        assert_eq!(parse_default_literal("Null"), Value::Null);
    }

    #[test]
    fn parse_default_literal_bool() {
        assert_eq!(parse_default_literal("true"), Value::Bool(true));
        assert_eq!(parse_default_literal("TRUE"), Value::Bool(true));
        assert_eq!(parse_default_literal("false"), Value::Bool(false));
        assert_eq!(parse_default_literal("FALSE"), Value::Bool(false));
    }

    #[test]
    fn parse_default_literal_string_single_quoted() {
        assert_eq!(
            parse_default_literal("'hello'"),
            Value::String("hello".into())
        );
    }

    #[test]
    fn parse_default_literal_string_double_quoted() {
        assert_eq!(
            parse_default_literal("\"world\""),
            Value::String("world".into())
        );
    }

    #[test]
    fn parse_default_literal_integer() {
        assert_eq!(parse_default_literal("42"), Value::Int64(42));
        assert_eq!(parse_default_literal("-7"), Value::Int64(-7));
        assert_eq!(parse_default_literal("0"), Value::Int64(0));
    }

    #[test]
    fn parse_default_literal_float() {
        assert_eq!(parse_default_literal("9.81"), Value::Float64(9.81_f64));
        assert_eq!(parse_default_literal("-0.5"), Value::Float64(-0.5));
    }

    #[test]
    fn parse_default_literal_fallback_string() {
        // Not a recognized literal, not quoted, not a number
        assert_eq!(
            parse_default_literal("some_identifier"),
            Value::String("some_identifier".into())
        );
    }

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

        session.begin_transaction().unwrap();
        assert!(session.in_transaction());

        session.commit().unwrap();
        assert!(!session.in_transaction());
    }

    #[test]
    fn test_session_transaction_context() {
        let db = GrafeoDB::new_in_memory();
        let mut session = db.session();

        // Without transaction - context should have current epoch and no transaction_id
        let (_epoch1, transaction_id1) = session.get_transaction_context();
        assert!(transaction_id1.is_none());

        // Start a transaction
        session.begin_transaction().unwrap();
        let (epoch2, transaction_id2) = session.get_transaction_context();
        assert!(transaction_id2.is_some());
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

        session.begin_transaction().unwrap();
        session.rollback().unwrap();
        assert!(!session.in_transaction());
    }

    #[test]
    fn test_session_rollback_discards_versions() {
        use grafeo_common::types::TransactionId;

        let db = GrafeoDB::new_in_memory();

        // Create a node outside of any transaction (at system level)
        let node_before = db.store().create_node(&["Person"]);
        assert!(node_before.is_valid());
        assert_eq!(db.node_count(), 1, "Should have 1 node before transaction");

        // Start a transaction
        let mut session = db.session();
        session.begin_transaction().unwrap();
        let transaction_id = session.current_transaction.lock().unwrap();

        // Create a node versioned with the transaction's ID
        let epoch = db.store().current_epoch();
        let node_in_tx = db
            .store()
            .create_node_versioned(&["Person"], epoch, transaction_id);
        assert!(node_in_tx.is_valid());

        // Uncommitted nodes use EpochId::PENDING, so they are invisible to
        // non-versioned lookups like node_count(). Verify the node is visible
        // only through the owning transaction.
        assert_eq!(
            db.node_count(),
            1,
            "PENDING nodes should be invisible to non-versioned node_count()"
        );
        assert!(
            db.store()
                .get_node_versioned(node_in_tx, epoch, transaction_id)
                .is_some(),
            "Transaction node should be visible to its own transaction"
        );

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
                .get_node_versioned(node_before, current_epoch, TransactionId::SYSTEM)
                .is_some(),
            "Original node should still exist"
        );

        // The node created in the transaction should not be accessible
        assert!(
            db.store()
                .get_node_versioned(node_in_tx, current_epoch, TransactionId::SYSTEM)
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
        session.begin_transaction().unwrap();
        let transaction_id = session.current_transaction.lock().unwrap();

        // Create a node through session.create_node() - should be versioned with tx
        let node_in_tx = session.create_node(&["Person"]);
        assert!(node_in_tx.is_valid());

        // Uncommitted nodes use EpochId::PENDING, so they are invisible to
        // non-versioned lookups. Verify the node is visible only to its own tx.
        assert_eq!(
            db.node_count(),
            1,
            "PENDING nodes should be invisible to non-versioned node_count()"
        );
        let epoch = db.store().current_epoch();
        assert!(
            db.store()
                .get_node_versioned(node_in_tx, epoch, transaction_id)
                .is_some(),
            "Transaction node should be visible to its own transaction"
        );

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
        session.begin_transaction().unwrap();
        let transaction_id = session.current_transaction.lock().unwrap();

        let node_in_tx =
            session.create_node_with_props(&["Person"], [("name", Value::String("Alix".into()))]);
        assert!(node_in_tx.is_valid());

        // Uncommitted nodes use EpochId::PENDING, so they are invisible to
        // non-versioned lookups. Verify the node is visible only to its own tx.
        assert_eq!(
            db.node_count(),
            1,
            "PENDING nodes should be invisible to non-versioned node_count()"
        );
        let epoch = db.store().current_epoch();
        assert!(
            db.store()
                .get_node_versioned(node_in_tx, epoch, transaction_id)
                .is_some(),
            "Transaction node should be visible to its own transaction"
        );

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

            // Create a graph: Alix -> Gus, Alix -> Vincent
            let alix = session.create_node(&["Person"]);
            let gus = session.create_node(&["Person"]);
            let vincent = session.create_node(&["Person"]);

            session.create_edge(alix, gus, "KNOWS");
            session.create_edge(alix, vincent, "KNOWS");

            // Execute a path query: MATCH (a:Person)-[:KNOWS]->(b:Person) RETURN a, b
            let result = session
                .execute("MATCH (a:Person)-[:KNOWS]->(b:Person) RETURN a, b")
                .unwrap();

            // Should return 2 rows (Alix->Gus, Alix->Vincent)
            assert_eq!(result.row_count(), 2);
            assert_eq!(result.column_count(), 2);
            assert_eq!(result.columns[0], "a");
            assert_eq!(result.columns[1], "b");
        }

        #[test]
        fn test_gql_relationship_with_type_filter() {
            let db = GrafeoDB::new_in_memory();
            let session = db.session();

            // Create a graph: Alix -KNOWS-> Gus, Alix -WORKS_WITH-> Vincent
            let alix = session.create_node(&["Person"]);
            let gus = session.create_node(&["Person"]);
            let vincent = session.create_node(&["Person"]);

            session.create_edge(alix, gus, "KNOWS");
            session.create_edge(alix, vincent, "WORKS_WITH");

            // Query only KNOWS relationships
            let result = session
                .execute("MATCH (a:Person)-[:KNOWS]->(b:Person) RETURN a, b")
                .unwrap();

            // Should return only 1 row (Alix->Gus)
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
            let Err(err) = result else {
                panic!("Expected error")
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
            session.create_node_with_props(&["Person"], [("name", Value::String("Alix".into()))]);
            session.create_node_with_props(&["Person"], [("name", Value::String("Gus".into()))]);
            session.create_node_with_props(&["Person"], [("name", Value::String("Alix".into()))]);

            // Query with WHERE clause: name = "Alix"
            let result = session
                .execute("MATCH (n:Person) WHERE n.name = \"Alix\" RETURN n")
                .unwrap();

            // Should return 2 people named Alix
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
                    ("name", Value::String("Alix".into())),
                    ("age", Value::Int64(30)),
                ],
            );
            session.create_node_with_props(
                &["Person"],
                [
                    ("name", Value::String("Gus".into())),
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
            assert!(names.contains(&&Value::String("Alix".into())));
            assert!(names.contains(&&Value::String("Gus".into())));
        }

        #[test]
        fn test_gql_return_mixed_expressions() {
            use grafeo_common::types::Value;

            let db = GrafeoDB::new_in_memory();
            let session = db.session();

            // Create a person
            session.create_node_with_props(&["Person"], [("name", Value::String("Alix".into()))]);

            // Query returning both node and property
            let result = session
                .execute("MATCH (n:Person) RETURN n, n.name")
                .unwrap();

            assert_eq!(result.row_count(), 1);
            assert_eq!(result.column_count(), 2);
            assert_eq!(result.columns[0], "n");
            assert_eq!(result.columns[1], "n.name");

            // Second column should be the name
            assert_eq!(result.rows[0][1], Value::String("Alix".into()));
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
                .create_node_with_props(&["Person"], [("name", Value::String("Alix".into()))]);

            let name = session.get_node_property(id, "name");
            assert_eq!(name, Some(Value::String("Alix".into())));

            // Non-existent property
            let missing = session.get_node_property(id, "missing");
            assert!(missing.is_none());
        }

        #[test]
        fn test_get_edge() {
            let db = GrafeoDB::new_in_memory();
            let session = db.session();

            let alix = session.create_node(&["Person"]);
            let gus = session.create_node(&["Person"]);
            let edge_id = session.create_edge(alix, gus, "KNOWS");

            let edge = session.get_edge(edge_id);
            assert!(edge.is_some());
            let edge = edge.unwrap();
            assert_eq!(edge.id, edge_id);
            assert_eq!(edge.src, alix);
            assert_eq!(edge.dst, gus);
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

            let alix = session.create_node(&["Person"]);
            let gus = session.create_node(&["Person"]);
            let harm = session.create_node(&["Person"]);

            session.create_edge(alix, gus, "KNOWS");
            session.create_edge(alix, harm, "KNOWS");

            let neighbors = session.get_neighbors_outgoing(alix);
            assert_eq!(neighbors.len(), 2);

            let neighbor_ids: Vec<_> = neighbors.iter().map(|(node_id, _)| *node_id).collect();
            assert!(neighbor_ids.contains(&gus));
            assert!(neighbor_ids.contains(&harm));
        }

        #[test]
        fn test_get_neighbors_incoming() {
            let db = GrafeoDB::new_in_memory();
            let session = db.session();

            let alix = session.create_node(&["Person"]);
            let gus = session.create_node(&["Person"]);
            let harm = session.create_node(&["Person"]);

            session.create_edge(gus, alix, "KNOWS");
            session.create_edge(harm, alix, "KNOWS");

            let neighbors = session.get_neighbors_incoming(alix);
            assert_eq!(neighbors.len(), 2);

            let neighbor_ids: Vec<_> = neighbors.iter().map(|(node_id, _)| *node_id).collect();
            assert!(neighbor_ids.contains(&gus));
            assert!(neighbor_ids.contains(&harm));
        }

        #[test]
        fn test_get_neighbors_outgoing_by_type() {
            let db = GrafeoDB::new_in_memory();
            let session = db.session();

            let alix = session.create_node(&["Person"]);
            let gus = session.create_node(&["Person"]);
            let company = session.create_node(&["Company"]);

            session.create_edge(alix, gus, "KNOWS");
            session.create_edge(alix, company, "WORKS_AT");

            let knows_neighbors = session.get_neighbors_outgoing_by_type(alix, "KNOWS");
            assert_eq!(knows_neighbors.len(), 1);
            assert_eq!(knows_neighbors[0].0, gus);

            let works_neighbors = session.get_neighbors_outgoing_by_type(alix, "WORKS_AT");
            assert_eq!(works_neighbors.len(), 1);
            assert_eq!(works_neighbors[0].0, company);

            // No edges of this type
            let no_neighbors = session.get_neighbors_outgoing_by_type(alix, "LIKES");
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

            let alix = session.create_node(&["Person"]);
            let gus = session.create_node(&["Person"]);
            let edge_id = session.create_edge(alix, gus, "KNOWS");

            assert!(session.edge_exists(edge_id));
            assert!(!session.edge_exists(EdgeId::new(9999)));
        }

        #[test]
        fn test_get_degree() {
            let db = GrafeoDB::new_in_memory();
            let session = db.session();

            let alix = session.create_node(&["Person"]);
            let gus = session.create_node(&["Person"]);
            let harm = session.create_node(&["Person"]);

            // Alix knows Gus and Harm (2 outgoing)
            session.create_edge(alix, gus, "KNOWS");
            session.create_edge(alix, harm, "KNOWS");
            // Gus knows Alix (1 incoming for Alix)
            session.create_edge(gus, alix, "KNOWS");

            let (out_degree, in_degree) = session.get_degree(alix);
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

            let alix = session.create_node(&["Person"]);
            let gus = session.create_node(&["Person"]);
            let harm = session.create_node(&["Person"]);

            let nodes = session.get_nodes_batch(&[alix, gus, harm]);
            assert_eq!(nodes.len(), 3);
            assert!(nodes[0].is_some());
            assert!(nodes[1].is_some());
            assert!(nodes[2].is_some());

            // With non-existent node
            use grafeo_common::types::NodeId;
            let nodes_with_missing = session.get_nodes_batch(&[alix, NodeId::new(9999), harm]);
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
        fn test_transaction_double_begin_nests() {
            let db = GrafeoDB::new_in_memory();
            let mut session = db.session();

            session.begin_transaction().unwrap();
            // Second begin_transaction creates a nested transaction (auto-savepoint)
            let result = session.begin_transaction();
            assert!(result.is_ok());
            // Commit the inner (releases savepoint)
            session.commit().unwrap();
            // Commit the outer
            session.commit().unwrap();
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
            let alix = session.create_node(&["Person"]);
            let gus = session.create_node(&["Person"]);

            // Create edge in transaction
            session.begin_transaction().unwrap();
            let edge_id = session.create_edge(alix, gus, "KNOWS");

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

    #[test]
    fn test_auto_gc_triggers_on_commit_interval() {
        use crate::config::Config;

        let config = Config::in_memory().with_gc_interval(2);
        let db = GrafeoDB::with_config(config).unwrap();
        let mut session = db.session();

        // First commit: counter = 1, no GC (not a multiple of 2)
        session.begin_transaction().unwrap();
        session.create_node(&["A"]);
        session.commit().unwrap();

        // Second commit: counter = 2, GC should trigger (multiple of 2)
        session.begin_transaction().unwrap();
        session.create_node(&["B"]);
        session.commit().unwrap();

        // Verify the database is still functional after GC
        assert_eq!(db.node_count(), 2);
    }

    #[test]
    fn test_query_timeout_config_propagates_to_session() {
        use crate::config::Config;
        use std::time::Duration;

        let config = Config::in_memory().with_query_timeout(Duration::from_secs(5));
        let db = GrafeoDB::with_config(config).unwrap();
        let session = db.session();

        // Verify the session has a query deadline (timeout was set)
        assert!(session.query_deadline().is_some());
    }

    #[test]
    fn test_no_query_timeout_returns_no_deadline() {
        let db = GrafeoDB::new_in_memory();
        let session = db.session();

        // Default config has no timeout
        assert!(session.query_deadline().is_none());
    }

    #[test]
    fn test_graph_model_accessor() {
        use crate::config::GraphModel;

        let db = GrafeoDB::new_in_memory();
        let session = db.session();

        assert_eq!(session.graph_model(), GraphModel::Lpg);
    }

    #[cfg(feature = "gql")]
    #[test]
    fn test_external_store_session() {
        use grafeo_core::graph::GraphStoreMut;
        use std::sync::Arc;

        let config = crate::config::Config::in_memory();
        let store =
            Arc::new(grafeo_core::graph::lpg::LpgStore::new().unwrap()) as Arc<dyn GraphStoreMut>;
        let db = GrafeoDB::with_store(store, config).unwrap();

        let mut session = db.session();

        // Use an explicit transaction so that INSERT and MATCH share the same
        // transaction context. With PENDING epochs, uncommitted versions are
        // only visible to the owning transaction.
        session.begin_transaction().unwrap();
        session.execute("INSERT (:Test {name: 'hello'})").unwrap();

        // Verify we can query through it within the same transaction
        let result = session.execute("MATCH (n:Test) RETURN n.name").unwrap();
        assert_eq!(result.row_count(), 1);

        session.commit().unwrap();
    }

    // ==================== Session Command Tests ====================

    #[cfg(feature = "gql")]
    mod session_command_tests {
        use super::*;
        use grafeo_common::types::Value;

        #[test]
        fn test_use_graph_sets_current_graph() {
            let db = GrafeoDB::new_in_memory();
            let session = db.session();

            // Create the graph first, then USE it
            session.execute("CREATE GRAPH mydb").unwrap();
            session.execute("USE GRAPH mydb").unwrap();

            assert_eq!(session.current_graph(), Some("mydb".to_string()));
        }

        #[test]
        fn test_use_graph_nonexistent_errors() {
            let db = GrafeoDB::new_in_memory();
            let session = db.session();

            let result = session.execute("USE GRAPH doesnotexist");
            assert!(result.is_err());
            let err = result.unwrap_err().to_string();
            assert!(
                err.contains("does not exist"),
                "Expected 'does not exist' error, got: {err}"
            );
        }

        #[test]
        fn test_use_graph_default_always_valid() {
            let db = GrafeoDB::new_in_memory();
            let session = db.session();

            // "default" is always valid, even without CREATE GRAPH
            session.execute("USE GRAPH default").unwrap();
            assert_eq!(session.current_graph(), Some("default".to_string()));
        }

        #[test]
        fn test_session_set_graph() {
            let db = GrafeoDB::new_in_memory();
            let session = db.session();

            session.execute("CREATE GRAPH analytics").unwrap();
            session.execute("SESSION SET GRAPH analytics").unwrap();
            assert_eq!(session.current_graph(), Some("analytics".to_string()));
        }

        #[test]
        fn test_session_set_graph_nonexistent_errors() {
            let db = GrafeoDB::new_in_memory();
            let session = db.session();

            let result = session.execute("SESSION SET GRAPH nosuchgraph");
            assert!(result.is_err());
        }

        #[test]
        fn test_session_set_time_zone() {
            let db = GrafeoDB::new_in_memory();
            let session = db.session();

            assert_eq!(session.time_zone(), None);

            session.execute("SESSION SET TIME ZONE 'UTC'").unwrap();
            assert_eq!(session.time_zone(), Some("UTC".to_string()));

            session
                .execute("SESSION SET TIME ZONE 'America/New_York'")
                .unwrap();
            assert_eq!(session.time_zone(), Some("America/New_York".to_string()));
        }

        #[test]
        fn test_session_set_parameter() {
            let db = GrafeoDB::new_in_memory();
            let session = db.session();

            session
                .execute("SESSION SET PARAMETER $timeout = 30")
                .unwrap();

            // Parameter is stored (value is Null for now, since expression
            // evaluation is not yet wired up)
            assert!(session.get_parameter("timeout").is_some());
        }

        #[test]
        fn test_session_reset_clears_all_state() {
            let db = GrafeoDB::new_in_memory();
            let session = db.session();

            // Set various session state
            session.execute("CREATE GRAPH analytics").unwrap();
            session.execute("SESSION SET GRAPH analytics").unwrap();
            session.execute("SESSION SET TIME ZONE 'UTC'").unwrap();
            session
                .execute("SESSION SET PARAMETER $limit = 100")
                .unwrap();

            // Verify state was set
            assert!(session.current_graph().is_some());
            assert!(session.time_zone().is_some());
            assert!(session.get_parameter("limit").is_some());

            // Reset everything
            session.execute("SESSION RESET").unwrap();

            assert_eq!(session.current_graph(), None);
            assert_eq!(session.time_zone(), None);
            assert!(session.get_parameter("limit").is_none());
        }

        #[test]
        fn test_session_close_clears_state() {
            let db = GrafeoDB::new_in_memory();
            let session = db.session();

            session.execute("CREATE GRAPH analytics").unwrap();
            session.execute("SESSION SET GRAPH analytics").unwrap();
            session.execute("SESSION SET TIME ZONE 'UTC'").unwrap();

            session.execute("SESSION CLOSE").unwrap();

            assert_eq!(session.current_graph(), None);
            assert_eq!(session.time_zone(), None);
        }

        #[test]
        fn test_create_graph() {
            let db = GrafeoDB::new_in_memory();
            let session = db.session();

            session.execute("CREATE GRAPH mydb").unwrap();

            // Should be able to USE it now
            session.execute("USE GRAPH mydb").unwrap();
            assert_eq!(session.current_graph(), Some("mydb".to_string()));
        }

        #[test]
        fn test_create_graph_duplicate_errors() {
            let db = GrafeoDB::new_in_memory();
            let session = db.session();

            session.execute("CREATE GRAPH mydb").unwrap();
            let result = session.execute("CREATE GRAPH mydb");

            assert!(result.is_err());
            let err = result.unwrap_err().to_string();
            assert!(
                err.contains("already exists"),
                "Expected 'already exists' error, got: {err}"
            );
        }

        #[test]
        fn test_create_graph_if_not_exists() {
            let db = GrafeoDB::new_in_memory();
            let session = db.session();

            session.execute("CREATE GRAPH mydb").unwrap();
            // Should succeed silently with IF NOT EXISTS
            session.execute("CREATE GRAPH IF NOT EXISTS mydb").unwrap();
        }

        #[test]
        fn test_drop_graph() {
            let db = GrafeoDB::new_in_memory();
            let session = db.session();

            session.execute("CREATE GRAPH mydb").unwrap();
            session.execute("DROP GRAPH mydb").unwrap();

            // Should no longer be usable
            let result = session.execute("USE GRAPH mydb");
            assert!(result.is_err());
        }

        #[test]
        fn test_drop_graph_nonexistent_errors() {
            let db = GrafeoDB::new_in_memory();
            let session = db.session();

            let result = session.execute("DROP GRAPH nosuchgraph");
            assert!(result.is_err());
            let err = result.unwrap_err().to_string();
            assert!(
                err.contains("does not exist"),
                "Expected 'does not exist' error, got: {err}"
            );
        }

        #[test]
        fn test_drop_graph_if_exists() {
            let db = GrafeoDB::new_in_memory();
            let session = db.session();

            // Should succeed silently with IF EXISTS
            session.execute("DROP GRAPH IF EXISTS nosuchgraph").unwrap();
        }

        #[test]
        fn test_start_transaction_via_gql() {
            let db = GrafeoDB::new_in_memory();
            let session = db.session();

            session.execute("START TRANSACTION").unwrap();
            assert!(session.in_transaction());
            session.execute("INSERT (:Person {name: 'Alix'})").unwrap();
            session.execute("COMMIT").unwrap();
            assert!(!session.in_transaction());

            let result = session.execute("MATCH (n:Person) RETURN n.name").unwrap();
            assert_eq!(result.rows.len(), 1);
        }

        #[test]
        fn test_start_transaction_read_only_blocks_insert() {
            let db = GrafeoDB::new_in_memory();
            let session = db.session();

            session.execute("START TRANSACTION READ ONLY").unwrap();
            let result = session.execute("INSERT (:Person {name: 'Alix'})");
            assert!(result.is_err());
            let err = result.unwrap_err().to_string();
            assert!(
                err.contains("read-only"),
                "Expected read-only error, got: {err}"
            );
            session.execute("ROLLBACK").unwrap();
        }

        #[test]
        fn test_start_transaction_read_only_allows_reads() {
            let db = GrafeoDB::new_in_memory();
            let mut session = db.session();
            session.begin_transaction().unwrap();
            session.execute("INSERT (:Person {name: 'Alix'})").unwrap();
            session.commit().unwrap();

            session.execute("START TRANSACTION READ ONLY").unwrap();
            let result = session.execute("MATCH (n:Person) RETURN n.name").unwrap();
            assert_eq!(result.rows.len(), 1);
            session.execute("COMMIT").unwrap();
        }

        #[test]
        fn test_rollback_via_gql() {
            let db = GrafeoDB::new_in_memory();
            let session = db.session();

            session.execute("START TRANSACTION").unwrap();
            session.execute("INSERT (:Person {name: 'Alix'})").unwrap();
            session.execute("ROLLBACK").unwrap();

            let result = session.execute("MATCH (n:Person) RETURN n.name").unwrap();
            assert!(result.rows.is_empty());
        }

        #[test]
        fn test_start_transaction_with_isolation_level() {
            let db = GrafeoDB::new_in_memory();
            let session = db.session();

            session
                .execute("START TRANSACTION ISOLATION LEVEL SERIALIZABLE")
                .unwrap();
            assert!(session.in_transaction());
            session.execute("ROLLBACK").unwrap();
        }

        #[test]
        fn test_session_commands_return_empty_result() {
            let db = GrafeoDB::new_in_memory();
            let session = db.session();

            session.execute("CREATE GRAPH test").unwrap();
            let result = session.execute("SESSION SET GRAPH test").unwrap();
            assert_eq!(result.row_count(), 0);
            assert_eq!(result.column_count(), 0);
        }

        #[test]
        fn test_current_graph_default_is_none() {
            let db = GrafeoDB::new_in_memory();
            let session = db.session();

            assert_eq!(session.current_graph(), None);
        }

        #[test]
        fn test_time_zone_default_is_none() {
            let db = GrafeoDB::new_in_memory();
            let session = db.session();

            assert_eq!(session.time_zone(), None);
        }

        #[test]
        fn test_session_state_independent_across_sessions() {
            let db = GrafeoDB::new_in_memory();
            let session1 = db.session();
            let session2 = db.session();

            session1.execute("CREATE GRAPH first").unwrap();
            session1.execute("CREATE GRAPH second").unwrap();
            session1.execute("SESSION SET GRAPH first").unwrap();
            session2.execute("SESSION SET GRAPH second").unwrap();

            assert_eq!(session1.current_graph(), Some("first".to_string()));
            assert_eq!(session2.current_graph(), Some("second".to_string()));
        }

        #[test]
        fn test_show_node_types() {
            let db = GrafeoDB::new_in_memory();
            let session = db.session();

            session
                .execute("CREATE NODE TYPE Person (name STRING NOT NULL, age INTEGER)")
                .unwrap();

            let result = session.execute("SHOW NODE TYPES").unwrap();
            assert_eq!(
                result.columns,
                vec!["name", "properties", "constraints", "parents"]
            );
            assert_eq!(result.rows.len(), 1);
            // First column is the type name
            assert_eq!(result.rows[0][0], Value::from("Person"));
        }

        #[test]
        fn test_show_edge_types() {
            let db = GrafeoDB::new_in_memory();
            let session = db.session();

            session
                .execute("CREATE EDGE TYPE KNOWS CONNECTING (Person) TO (Person) (since INTEGER)")
                .unwrap();

            let result = session.execute("SHOW EDGE TYPES").unwrap();
            assert_eq!(
                result.columns,
                vec!["name", "properties", "source_types", "target_types"]
            );
            assert_eq!(result.rows.len(), 1);
            assert_eq!(result.rows[0][0], Value::from("KNOWS"));
        }

        #[test]
        fn test_show_graph_types() {
            let db = GrafeoDB::new_in_memory();
            let session = db.session();

            session
                .execute("CREATE NODE TYPE Person (name STRING)")
                .unwrap();
            session
                .execute(
                    "CREATE GRAPH TYPE social (\
                        NODE TYPE Person (name STRING)\
                    )",
                )
                .unwrap();

            let result = session.execute("SHOW GRAPH TYPES").unwrap();
            assert_eq!(
                result.columns,
                vec!["name", "open", "node_types", "edge_types"]
            );
            assert_eq!(result.rows.len(), 1);
            assert_eq!(result.rows[0][0], Value::from("social"));
        }

        #[test]
        fn test_show_graph_type_named() {
            let db = GrafeoDB::new_in_memory();
            let session = db.session();

            session
                .execute("CREATE NODE TYPE Person (name STRING)")
                .unwrap();
            session
                .execute(
                    "CREATE GRAPH TYPE social (\
                        NODE TYPE Person (name STRING)\
                    )",
                )
                .unwrap();

            let result = session.execute("SHOW GRAPH TYPE social").unwrap();
            assert_eq!(result.rows.len(), 1);
            assert_eq!(result.rows[0][0], Value::from("social"));
        }

        #[test]
        fn test_show_graph_type_not_found() {
            let db = GrafeoDB::new_in_memory();
            let session = db.session();

            let result = session.execute("SHOW GRAPH TYPE nonexistent");
            assert!(result.is_err());
        }

        #[test]
        fn test_show_indexes_via_gql() {
            let db = GrafeoDB::new_in_memory();
            let session = db.session();

            let result = session.execute("SHOW INDEXES").unwrap();
            assert_eq!(result.columns, vec!["name", "type", "label", "property"]);
        }

        #[test]
        fn test_show_constraints_via_gql() {
            let db = GrafeoDB::new_in_memory();
            let session = db.session();

            let result = session.execute("SHOW CONSTRAINTS").unwrap();
            assert_eq!(result.columns, vec!["name", "type", "label", "properties"]);
        }

        #[test]
        fn test_pattern_form_graph_type_roundtrip() {
            let db = GrafeoDB::new_in_memory();
            let session = db.session();

            // Register the types first
            session
                .execute("CREATE NODE TYPE Person (name STRING NOT NULL)")
                .unwrap();
            session
                .execute("CREATE NODE TYPE City (name STRING)")
                .unwrap();
            session
                .execute("CREATE EDGE TYPE KNOWS (since INTEGER)")
                .unwrap();
            session.execute("CREATE EDGE TYPE LIVES_IN").unwrap();

            // Create graph type using pattern form
            session
                .execute(
                    "CREATE GRAPH TYPE social (\
                        (:Person {name STRING NOT NULL})-[:KNOWS {since INTEGER}]->(:Person),\
                        (:Person)-[:LIVES_IN]->(:City)\
                    )",
                )
                .unwrap();

            // Verify it was created
            let result = session.execute("SHOW GRAPH TYPE social").unwrap();
            assert_eq!(result.rows.len(), 1);
            assert_eq!(result.rows[0][0], Value::from("social"));
        }
    }
}
