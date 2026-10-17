//! RDF-specific session methods.
//!
//! This module consolidates all RDF functionality from the session layer.
//! The entire module is gated behind `#[cfg(feature = "triple-store")]` in the parent.

use std::sync::Arc;
#[cfg(feature = "lpg")]
use std::sync::atomic::AtomicUsize;
#[cfg(not(target_arch = "wasm32"))]
use std::time::Instant;

use grafeo_common::types::Value;
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
    /// Plans native RDF paths with the session's pending writes and resource limits.
    fn make_rdf_planner(
        &self,
        deadline: Option<std::time::Instant>,
    ) -> crate::query::planner::rdf::RdfPlanner {
        let budget = self.plan_options.path_search_budget;
        #[cfg(feature = "spill")]
        let budget = self
            .buffer_manager
            .as_ref()
            .map_or(budget, |bm| budget.min(bm.available()));
        crate::query::planner::rdf::RdfPlanner::new(Arc::clone(&self.rdf_store))
            .with_shuffle_unordered(self.plan_options.shuffle_unordered)
            .with_writer(self.rdf_writer())
            .with_path_search_budget(budget)
            .with_deadline(deadline)
    }

    /// Creates a session that reads and writes `store` and `rdf_store`.
    #[cfg(feature = "lpg")]
    pub(crate) fn with_rdf_store(
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
            plan_options: super::PlanOptions {
                factorized_execution: cfg.factorized_execution,
                shuffle_unordered: cfg.shuffle_unordered,
                reachability: true,
                path_search_budget: cfg.path_search_budget,
            },
            graph_model: cfg.graph_model,
            query_timeout: cfg.query_timeout,
            max_property_size: cfg.max_property_size,
            #[cfg(feature = "spill")]
            buffer_manager: cfg.buffer_manager,
            commit_counter: cfg.commit_counter,
            gc_interval: cfg.gc_interval,
            active_streams: AtomicUsize::new(0),
            #[cfg(feature = "wal")]
            wal: None,
            #[cfg(feature = "cdc")]
            cdc_log: Arc::new(crate::cdc::CdcLog::new()),
            #[cfg(feature = "cdc")]
            records_cdc: false,
            current_graph: parking_lot::Mutex::new(None),
            current_schema: parking_lot::Mutex::new(None),
            time_zone: parking_lot::Mutex::new(None),
            session_params: parking_lot::Mutex::new(std::collections::HashMap::new()),
            viewing_epoch_override: parking_lot::Mutex::new(None),
            savepoints: parking_lot::Mutex::new(Vec::new()),
            transaction_nesting_depth: parking_lot::Mutex::new(0),
            changes: parking_lot::Mutex::new(None),
            external_target: None,
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
        use crate::query::{optimizer::Optimizer, translators::graphql_rdf};

        let logical_plan = graphql_rdf::translate(query, "http://example.org/")?;
        let active = self.active_store();
        let optimizer = Optimizer::from_graph_store(&*active);
        let optimized_plan = optimizer.optimize(logical_plan)?;
        self.run_rdf_plan(&optimized_plan, "graphql")
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
            optimizer::Optimizer, processor::substitute_params, translators::graphql_rdf,
        };

        // Parse and translate the query to a logical plan
        let mut logical_plan = graphql_rdf::translate(query, "http://example.org/")?;

        // Substitute parameters
        substitute_params(&mut logical_plan, &params)?;

        // Optimize the plan
        let rdf_stats = self.rdf_store.get_or_collect_statistics();
        let optimizer = Optimizer::from_rdf_statistics((*rdf_stats).clone());
        let optimized_plan = optimizer.optimize(logical_plan)?;
        self.run_rdf_plan(&optimized_plan, "graphql")
    }

    /// Executes a SPARQL query or update against this session.
    ///
    /// Without an open transaction each update commits on its own; to apply a
    /// sequence of updates atomically, wrap it in `begin_transaction` and
    /// `commit` on this session.
    ///
    /// # Errors
    ///
    /// Returns an error if the query fails to parse or execute.
    #[cfg(feature = "sparql")]
    pub fn execute_sparql(&self, query: &str) -> Result<QueryResult> {
        use crate::query::{optimizer::Optimizer, translators::sparql};

        let logical_plan = sparql::translate(query)?;
        let rdf_stats = self.rdf_store.get_or_collect_statistics();
        let optimizer = Optimizer::from_rdf_statistics((*rdf_stats).clone());
        let optimized_plan = optimizer.optimize(logical_plan)?;
        self.run_rdf_plan(&optimized_plan, "sparql")
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
            optimizer::Optimizer, processor::substitute_params, translators::sparql,
        };

        let mut logical_plan = sparql::translate(query)?;
        substitute_params(&mut logical_plan, &params)?;

        let rdf_stats = self.rdf_store.get_or_collect_statistics();
        let optimizer = Optimizer::from_rdf_statistics((*rdf_stats).clone());
        let optimized_plan = optimizer.optimize(logical_plan)?;
        self.run_rdf_plan(&optimized_plan, "sparql")
    }

    /// Runs `plan`, an optimized SPARQL or GraphQL plan in `language`, in
    /// this session. A read sees what the session's open transaction wrote.
    /// An update runs in that transaction, and its writes are undone when it
    /// fails; without one it runs in a transaction of its own, which commits
    /// when it succeeds (see [`in_statement_transaction`]). `EXPLAIN` shows
    /// the plan without running it, `PROFILE` runs it and shows what each
    /// operator did.
    ///
    /// [`in_statement_transaction`]: Self::in_statement_transaction
    ///
    /// # Errors
    ///
    /// The error of the statement. An update also fails, and changes
    /// nothing, on a session that may not write, and once the database is
    /// closed.
    pub(crate) fn run_rdf_plan(
        &self,
        optimized_plan: &crate::query::plan::LogicalPlan,
        language: &'static str,
    ) -> Result<QueryResult> {
        #[cfg(not(target_arch = "wasm32"))]
        let start_time = Instant::now();

        // A write needs a writable session: a writing role, and no read-only
        // transaction or database.
        let mutates = optimized_plan.root.has_mutations();
        if mutates && (!self.identity.can_admin() || *self.read_only_tx.lock()) {
            self.check_writable()?;
        }

        // EXPLAIN: return the logical plan tree without executing
        if optimized_plan.explain {
            use crate::query::processor::explain_result;
            return Ok(explain_result(optimized_plan));
        }

        // A write fails once the database is closed, also inside a
        // transaction (whose commit would fail) and for an admin identity,
        // which skips the other write checks.
        if mutates {
            self.transaction_manager.check_open()?;
        }

        let deadline = self.query_deadline();
        let result = self.in_statement_transaction(mutates, || {
            let planner = self.make_rdf_planner(deadline);
            if optimized_plan.profile {
                let (mut physical_plan, entries) = planner.plan_profiled(optimized_plan)?;
                let executor = self
                    .make_executor(physical_plan.columns.clone())
                    .with_deadline(deadline);
                let _result = executor.execute(physical_plan.operator.as_mut())?;
                #[cfg(not(target_arch = "wasm32"))]
                let total_time_ms = start_time.elapsed().as_secs_f64() * 1000.0;
                #[cfg(target_arch = "wasm32")]
                let total_time_ms = 0.0;
                let profile_tree = crate::query::profile::build_profile_tree(
                    &optimized_plan.root,
                    &mut entries.into_iter(),
                );
                return Ok(crate::query::profile::profile_result(
                    &profile_tree,
                    total_time_ms,
                ));
            }
            let mut physical_plan = planner.plan(optimized_plan)?;
            let executor = self
                .make_executor(physical_plan.columns.clone())
                .with_deadline(deadline);
            executor.execute(physical_plan.operator.as_mut())
        });

        #[cfg(feature = "metrics")]
        {
            #[cfg(not(target_arch = "wasm32"))]
            let elapsed_ms = Some(start_time.elapsed().as_secs_f64() * 1000.0);
            #[cfg(target_arch = "wasm32")]
            let elapsed_ms = None;
            self.record_query_metrics(language, elapsed_ms, &result);
        }
        #[cfg(not(feature = "metrics"))]
        let _ = language;

        result
    }

    /// The writer of the session's open transaction, through which an RDF
    /// update records its triples and a read sees them; `None` without one.
    fn rdf_writer(&self) -> Option<crate::transaction::RdfWriter> {
        let changes = self.changes.lock().clone()?;
        Some(crate::transaction::RdfWriter::new(
            Arc::clone(&self.rdf_store),
            changes,
            Arc::clone(&self.transaction_manager),
            #[cfg(feature = "wal")]
            self.wal().cloned(),
        ))
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
            grafeo_common::utils::error::Error::InvalidValue(format!(
                "there is no named graph '{data_graph_name}' to validate"
            ))
        })?;
        let shapes_store = self.rdf_store.graph(shapes_graph_name).ok_or_else(|| {
            grafeo_common::utils::error::Error::InvalidValue(format!(
                "there is no named graph '{shapes_graph_name}' to read the shapes from"
            ))
        })?;
        let executor =
            crate::validation::SessionSparqlExecutor::with_graph(self, data_graph_name.to_string());
        grafeo_core::graph::rdf::shacl::validate(&data_store, &shapes_store, Some(&executor))
            .map_err(|e| grafeo_common::utils::error::Error::Internal(e.to_string()))
    }
}
