//! RDF-specific session methods.
//!
//! This module consolidates all RDF functionality from the session layer.
//! The entire module is gated behind `#[cfg(feature = "triple-store")]` in the parent.

use std::sync::Arc;
#[cfg(feature = "lpg")]
use std::sync::atomic::AtomicUsize;
#[cfg(all(feature = "metrics", not(target_arch = "wasm32")))]
use std::time::Instant;

use grafeo_common::types::{TransactionId, Value};
use grafeo_common::utils::error::Result;
#[cfg(feature = "lpg")]
use grafeo_core::graph::lpg::LpgStore;
#[cfg(feature = "lpg")]
use grafeo_core::graph::rdf::RdfStore;
#[cfg(feature = "lpg")]
use grafeo_core::graph::{GraphStoreMut, GraphStoreSearch};

use crate::database::QueryResult;

use super::Session;
#[cfg(feature = "lpg")]
use super::SessionConfig;

impl Session {
    /// Creates a new session with RDF store and adaptive configuration.
    #[cfg(feature = "lpg")]
    pub(crate) fn with_rdf_store_and_adaptive(
        store: Arc<LpgStore>,
        rdf_store: Arc<RdfStore>,
        cfg: SessionConfig,
    ) -> Self {
        let graph_store = Arc::clone(&store) as Arc<dyn GraphStoreSearch>;
        let graph_store_mut = Some(Arc::clone(&store) as Arc<dyn GraphStoreMut>);
        Self {
            store,
            lpg_backend: super::LpgBackend::Active,
            graph_store,
            graph_store_mut,
            catalog: cfg.catalog,
            rdf_store,
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
            max_property_size: cfg.max_property_size,
            buffer_manager: cfg.buffer_manager,
            commit_counter: cfg.commit_counter,
            gc_interval: cfg.gc_interval,
            transaction_start_node_count: AtomicUsize::new(0),
            transaction_start_edge_count: AtomicUsize::new(0),
            active_streams: AtomicUsize::new(0),
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

    /// Executes a GraphQL query against the RDF store.
    ///
    /// # Errors
    ///
    /// Returns an error if the query fails to parse or execute.
    #[cfg(feature = "graphql")]
    pub fn execute_graphql_rdf(&self, query: &str) -> Result<QueryResult> {
        use crate::query::{
            optimizer::Optimizer, planner::rdf::RdfPlanner, translators::graphql_rdf,
        };

        #[cfg(all(feature = "metrics", not(target_arch = "wasm32")))]
        let start_time = Instant::now();

        let logical_plan = graphql_rdf::translate(query, "http://example.org/")?;
        let active = self.active_store();
        let optimizer = Optimizer::from_graph_store(&*active);
        let optimized_plan = optimizer.optimize(logical_plan)?;

        if !self.identity.can_admin() && optimized_plan.root.has_mutations() {
            self.require_permission(crate::auth::StatementKind::Write)?;
        }

        let planner = RdfPlanner::new(Arc::clone(&self.rdf_store))
            .with_transaction_id(*self.current_transaction.lock());
        #[cfg(feature = "wal")]
        let planner = planner.with_wal(self.wal.clone());
        #[cfg(all(feature = "cdc", feature = "lpg"))]
        let planner =
            planner.with_cdc_log(Some(Arc::clone(&self.cdc_log)), self.store.current_epoch());
        let mut physical_plan = planner.plan(&optimized_plan)?;

        let executor = self.make_executor(physical_plan.columns.clone());
        let result = executor.execute(physical_plan.operator.as_mut());

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

    /// Executes a GraphQL query against the RDF store with parameters.
    ///
    /// # Errors
    ///
    /// Returns an error if the query fails to parse or execute.
    #[cfg(feature = "graphql")]
    pub fn execute_graphql_rdf_with_params(
        &self,
        query: &str,
        params: std::collections::HashMap<String, Value>,
    ) -> Result<QueryResult> {
        use crate::query::{
            optimizer::Optimizer, planner::rdf::RdfPlanner, processor::substitute_params,
            translators::graphql_rdf,
        };

        #[cfg(all(feature = "metrics", not(target_arch = "wasm32")))]
        let start_time = Instant::now();

        // Parse and translate the query to a logical plan
        let mut logical_plan = graphql_rdf::translate(query, "http://example.org/")?;

        // Substitute parameters
        substitute_params(&mut logical_plan, &params)?;

        // Optimize the plan
        let rdf_stats = self.rdf_store.get_or_collect_statistics();
        let optimizer = Optimizer::from_rdf_statistics((*rdf_stats).clone());
        let optimized_plan = optimizer.optimize(logical_plan)?;

        // Check role-based permission for mutations
        if !self.identity.can_admin() && optimized_plan.root.has_mutations() {
            self.require_permission(crate::auth::StatementKind::Write)?;
        }

        // EXPLAIN: return the logical plan tree without executing
        if optimized_plan.explain {
            use crate::query::processor::explain_result;
            return Ok(explain_result(&optimized_plan));
        }

        let planner = RdfPlanner::new(Arc::clone(&self.rdf_store))
            .with_transaction_id(*self.current_transaction.lock());
        #[cfg(feature = "wal")]
        let planner = planner.with_wal(self.wal.clone());
        #[cfg(all(feature = "cdc", feature = "lpg"))]
        let planner =
            planner.with_cdc_log(Some(Arc::clone(&self.cdc_log)), self.store.current_epoch());
        let mut physical_plan = planner.plan(&optimized_plan)?;

        let executor = self.make_executor(physical_plan.columns.clone());
        let result = executor.execute(physical_plan.operator.as_mut());

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

    /// Executes a SPARQL query or update against this session.
    ///
    /// # Atomicity (transaction handle)
    ///
    /// A single SPARQL Update statement is applied atomically. A *sequence* of
    /// updates is atomic only when run inside one explicit transaction on this
    /// same session handle: call `begin_transaction` before the sequence and
    /// `commit` after it. Statements executed without an open transaction
    /// auto-commit individually, so a failure partway through a multi-statement
    /// sequence leaves the earlier statements applied. Updates issued through a
    /// different handle than the one holding the open transaction are not
    /// enlisted in that transaction.
    ///
    /// # Errors
    ///
    /// Returns an error if the query fails to parse or execute.
    #[cfg(feature = "sparql")]
    pub fn execute_sparql(&self, query: &str) -> Result<QueryResult> {
        use crate::query::{optimizer::Optimizer, planner::rdf::RdfPlanner, translators::sparql};

        #[cfg(all(feature = "metrics", not(target_arch = "wasm32")))]
        let start_time = Instant::now();

        let logical_plan = sparql::translate(query)?;
        let rdf_stats = self.rdf_store.get_or_collect_statistics();
        let optimizer = Optimizer::from_rdf_statistics((*rdf_stats).clone());
        let optimized_plan = optimizer.optimize(logical_plan)?;

        // Check role-based permission for mutations (skip tree walk for admin)
        if !self.identity.can_admin() && optimized_plan.root.has_mutations() {
            self.require_permission(crate::auth::StatementKind::Write)?;
        }

        // EXPLAIN: return the logical plan tree without executing
        if optimized_plan.explain {
            use crate::query::processor::explain_result;
            return Ok(explain_result(&optimized_plan));
        }

        let planner = RdfPlanner::new(Arc::clone(&self.rdf_store))
            .with_transaction_id(*self.current_transaction.lock());
        #[cfg(feature = "wal")]
        let planner = planner.with_wal(self.wal.clone());
        #[cfg(all(feature = "cdc", feature = "lpg"))]
        let planner =
            planner.with_cdc_log(Some(Arc::clone(&self.cdc_log)), self.store.current_epoch());
        let mut physical_plan = planner.plan(&optimized_plan)?;

        let executor = self.make_executor(physical_plan.columns.clone());
        let result = executor.execute(physical_plan.operator.as_mut());

        #[cfg(feature = "metrics")]
        {
            #[cfg(not(target_arch = "wasm32"))]
            let elapsed_ms = Some(start_time.elapsed().as_secs_f64() * 1000.0);
            #[cfg(target_arch = "wasm32")]
            let elapsed_ms = None;
            self.record_query_metrics("sparql", elapsed_ms, &result);
        }

        result
    }

    /// Executes a SPARQL query with parameters.
    ///
    /// # Errors
    ///
    /// Returns an error if the query fails to parse or execute.
    #[cfg(feature = "sparql")]
    pub fn execute_sparql_with_params(
        &self,
        query: &str,
        params: std::collections::HashMap<String, Value>,
    ) -> Result<QueryResult> {
        use crate::query::{
            optimizer::Optimizer, planner::rdf::RdfPlanner, processor::substitute_params,
            translators::sparql,
        };

        #[cfg(all(feature = "metrics", not(target_arch = "wasm32")))]
        let start_time = Instant::now();

        let mut logical_plan = sparql::translate(query)?;
        substitute_params(&mut logical_plan, &params)?;

        let rdf_stats = self.rdf_store.get_or_collect_statistics();
        let optimizer = Optimizer::from_rdf_statistics((*rdf_stats).clone());
        let optimized_plan = optimizer.optimize(logical_plan)?;

        // Check role-based permission for mutations (skip tree walk for admin)
        if !self.identity.can_admin() && optimized_plan.root.has_mutations() {
            self.require_permission(crate::auth::StatementKind::Write)?;
        }

        // EXPLAIN: return the logical plan tree without executing
        if optimized_plan.explain {
            use crate::query::processor::explain_result;
            return Ok(explain_result(&optimized_plan));
        }

        let planner = RdfPlanner::new(Arc::clone(&self.rdf_store))
            .with_transaction_id(*self.current_transaction.lock());
        #[cfg(feature = "wal")]
        let planner = planner.with_wal(self.wal.clone());
        #[cfg(all(feature = "cdc", feature = "lpg"))]
        let planner =
            planner.with_cdc_log(Some(Arc::clone(&self.cdc_log)), self.store.current_epoch());
        let mut physical_plan = planner.plan(&optimized_plan)?;

        let executor = self.make_executor(physical_plan.columns.clone());
        let result = executor.execute(physical_plan.operator.as_mut());

        #[cfg(feature = "metrics")]
        {
            #[cfg(not(target_arch = "wasm32"))]
            let elapsed_ms = Some(start_time.elapsed().as_secs_f64() * 1000.0);
            #[cfg(target_arch = "wasm32")]
            let elapsed_ms = None;
            self.record_query_metrics("sparql", elapsed_ms, &result);
        }

        result
    }

    /// Commits RDF transaction state.
    ///
    /// Called from the main commit path to finalize RDF changes.
    pub(super) fn commit_rdf_transaction(&self, transaction_id: TransactionId) {
        self.rdf_store.commit_transaction(transaction_id);
    }

    /// Rolls back RDF transaction state.
    ///
    /// Called from the main commit-conflict and rollback paths to discard RDF changes.
    pub(super) fn rollback_rdf_transaction(&self, transaction_id: TransactionId) {
        self.rdf_store.rollback_transaction(transaction_id);
    }

    /// Validates the default graph against SHACL shapes in a named graph.
    ///
    /// # Errors
    ///
    /// Returns an error if shape parsing fails or the shapes graph doesn't exist.
    #[cfg(feature = "shacl")]
    pub fn validate_shacl(
        &self,
        shapes_graph: &str,
    ) -> Result<grafeo_core::graph::rdf::shacl::ValidationReport> {
        crate::validation::validate_shacl(self, &self.rdf_store, shapes_graph)
    }

    /// Validates a named data graph against shapes in another named graph.
    ///
    /// Both SHACL Core constraints and SHACL-SPARQL constraints are scoped to
    /// the named data graph (SPARQL queries receive `FROM <data_graph_name>`).
    ///
    /// # Errors
    ///
    /// Returns an error if shape parsing fails or either graph doesn't exist.
    #[cfg(feature = "shacl")]
    pub fn validate_shacl_graph(
        &self,
        data_graph_name: &str,
        shapes_graph_name: &str,
    ) -> Result<grafeo_core::graph::rdf::shacl::ValidationReport> {
        let data_store = self.rdf_store.graph(data_graph_name).ok_or_else(|| {
            grafeo_common::utils::error::Error::Internal(format!(
                "Named graph '{data_graph_name}' not found"
            ))
        })?;
        let shapes_store = self.rdf_store.graph(shapes_graph_name).ok_or_else(|| {
            grafeo_common::utils::error::Error::Internal(format!(
                "Named graph '{shapes_graph_name}' not found"
            ))
        })?;
        let executor =
            crate::validation::SessionSparqlExecutor::with_graph(self, data_graph_name.to_string());
        grafeo_core::graph::rdf::shacl::validate(&data_store, &shapes_store, Some(&executor))
            .map_err(|e| grafeo_common::utils::error::Error::Internal(e.to_string()))
    }
}
