//! The session methods that run SPARQL and GraphQL queries on the RDF
//! store, which only a build with one of those languages has.

use grafeo_common::clock::Instant;
use grafeo_common::types::Value;
use grafeo_common::utils::error::Result;
use std::sync::Arc;

use crate::database::QueryResult;

use super::super::Session;

impl Session {
    /// The RDF view a statement reads at: its transaction's snapshot (built
    /// at the transaction's first RDF statement and kept with its changes),
    /// or, outside a transaction, the published epoch, read inside the
    /// statement's pin (see `in_statement_transaction`), so no entry above
    /// it is pruned while the statement runs.
    ///
    /// # Errors
    ///
    /// While the session has a viewing epoch set: RDF data keeps no
    /// history, so a read at an earlier epoch is refused rather than read
    /// at the present.
    pub(crate) fn rdf_view(&self) -> Result<grafeo_core::graph::rdf::RdfView> {
        if self.viewing_epoch().is_some() {
            return Err(grafeo_common::utils::error::QueryError::unsupported(
                "RDF data keeps no history: a SPARQL or RDF read cannot run at a viewing epoch; \
                 clear it with set_viewing_epoch(None) first",
            )
            .into());
        }
        if let Some(changes) = self.changes.lock().clone() {
            return changes.rdf_view(&self.rdf_store);
        }
        self.rdf_store
            .view_at(self.transaction_manager.current_epoch())
            .map_err(|error| grafeo_common::utils::error::Error::Internal(error.to_string()))
    }

    /// Runs `read`, a read of the RDF store outside a query plan (SHACL
    /// validation), at one view: in the open transaction at its snapshot, or
    /// in a pinned read at the published epoch.
    #[cfg(feature = "shacl")]
    pub(crate) fn with_rdf_view<T>(
        &self,
        read: impl FnOnce(&grafeo_core::graph::rdf::RdfView) -> Result<T>,
    ) -> Result<T> {
        self.in_statement_transaction(false, || read(&self.rdf_view()?))
    }

    /// Plans native RDF paths with the session's pending writes and resource
    /// limits, reading the store at `view`; `expands_copies` as
    /// [`rdf_writer`](Self::rdf_writer) takes it.
    fn make_rdf_planner(
        &self,
        deadline: Option<Instant>,
        view: grafeo_core::graph::rdf::RdfView,
        expands_copies: bool,
    ) -> Result<crate::query::planner::rdf::RdfPlanner> {
        // The one budget LPG path searches read too (#492): not narrowed by the memory free now.
        let budget = self.plan_options.path_search_budget;
        Ok(
            crate::query::planner::rdf::RdfPlanner::new(Arc::clone(&self.rdf_store), view)
                .in_session()
                .with_shuffle_unordered(self.plan_options.shuffle_unordered)
                .with_writer(self.rdf_writer(expands_copies)?)
                .with_path_search_budget(budget)
                .with_deadline(deadline)
                .with_warnings(self.statement_warnings())
                .with_exists_run_once(
                    self.plan_options
                        .rules
                        .has(crate::query::optimizer::PlannerRule::ExistsRunOnce),
                ),
        )
    }

    /// Executes a GraphQL query against the RDF store.
    ///
    /// # Errors
    ///
    /// Returns an error if the query fails to parse or execute.
    #[cfg(feature = "graphql")]
    pub fn execute_graphql_rdf(&self, query: &str) -> Result<QueryResult> {
        self.timed(|| self.execute_graphql_rdf_untimed(query))
    }

    /// [`execute_graphql_rdf`](Self::execute_graphql_rdf) without the time of the whole call.
    #[cfg(feature = "graphql")]
    fn execute_graphql_rdf_untimed(&self, query: &str) -> Result<QueryResult> {
        use crate::query::{optimizer::Optimizer, translators::graphql_rdf};

        let mut logical_plan =
            self.translated(graphql_rdf::translate(query, "http://example.org/")?)?;
        crate::query::processor::substitute_with_defaults(&mut logical_plan, None)?;
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
        self.timed(|| self.execute_graphql_rdf_with_params_untimed(query, params))
    }

    /// [`execute_graphql_rdf_with_params`](Self::execute_graphql_rdf_with_params) without the time of the whole call.
    #[cfg(feature = "graphql")]
    fn execute_graphql_rdf_with_params_untimed(
        &self,
        query: &str,
        params: std::collections::HashMap<String, Value>,
    ) -> Result<QueryResult> {
        use crate::query::{
            optimizer::Optimizer, processor::substitute_with_defaults, translators::graphql_rdf,
        };

        // Parse and translate the query to a logical plan
        let mut logical_plan =
            self.translated(graphql_rdf::translate(query, "http://example.org/")?)?;

        // The query's own defaults, then the parameters given
        substitute_with_defaults(&mut logical_plan, Some(&params))?;

        // Optimize the plan
        let optimizer = Optimizer::for_rdf_plan(&self.rdf_store, &logical_plan);
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
        self.timed(|| self.execute_sparql_untimed(query))
    }

    /// [`execute_sparql`](Self::execute_sparql) without the time of the whole call.
    #[cfg(feature = "sparql")]
    fn execute_sparql_untimed(&self, query: &str) -> Result<QueryResult> {
        use crate::query::{optimizer::Optimizer, translators::sparql};

        let logical_plan = self.translated(sparql::translate(query)?)?;
        let optimizer = Optimizer::for_rdf_plan(&self.rdf_store, &logical_plan);
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
        self.timed(|| self.execute_sparql_with_params_untimed(query, params))
    }

    /// [`execute_sparql_with_params`](Self::execute_sparql_with_params) without the time of the whole call.
    #[cfg(feature = "sparql")]
    fn execute_sparql_with_params_untimed(
        &self,
        query: &str,
        params: std::collections::HashMap<String, Value>,
    ) -> Result<QueryResult> {
        use crate::query::{optimizer::Optimizer, translators::sparql};

        let logical_plan = self.translated(sparql::translate_with_bindings(query, &params)?)?;

        let optimizer = Optimizer::for_rdf_plan(&self.rdf_store, &logical_plan);
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
        let start_time = Instant::now();
        // In a transaction the session opened, later statements read what a
        // `COPY`, `MOVE` or `ADD` wrote; a statement's own transaction
        // commits it whole.
        let expands_copies = self.current_transaction.lock().is_some();

        // A write needs a writable session: a writing role, and no read-only
        // transaction or database.
        let mutates = optimized_plan.root.has_mutations();
        if mutates && (!self.identity.can_admin() || *self.read_only_tx.lock()) {
            self.check_writable()?;
        }

        // EXPLAIN: return the logical plan tree without executing
        if optimized_plan.explain {
            use crate::query::processor::explain_result;
            return Ok(explain_result(
                optimized_plan,
                crate::query::optimizer::PlannerRules::all(),
            ));
        }

        // A write fails once the database is closed, also inside a
        // transaction (whose commit would fail) and for an admin identity,
        // which skips the other write checks.
        if mutates {
            self.transaction_manager.check_open()?;
        }

        let deadline = self.query_deadline();
        let result = self.in_statement_transaction(mutates, || {
            // The statement's memory, which its operators capture while it
            // is planned.
            let _memory = self.enter_statement_memory();
            let planner = self
                .make_rdf_planner(deadline, self.rdf_view()?, expands_copies)?
                .with_query_values(language == "graphql");
            if optimized_plan.profile {
                let (mut physical_plan, entries) = planner.plan_profiled(optimized_plan)?;
                let executor = self
                    .executor_with_deadline(physical_plan.columns.clone())
                    .with_deadline(deadline);
                let _result = executor.execute(physical_plan.operator.as_mut())?;
                let total_time_ms = start_time.elapsed().as_secs_f64() * 1000.0;
                let profile_tree =
                    crate::query::profile::build_profile_tree(&optimized_plan.root, entries)?;
                return Ok(
                    crate::query::profile::profile_result(&profile_tree, total_time_ms)
                        .with_warnings(planner.warnings().raised()),
                );
            }
            let mut physical_plan = planner.plan(optimized_plan)?;
            let executor = self
                .executor_with_deadline(physical_plan.columns.clone())
                .with_deadline(deadline)
                .with_read_counter(planner.read_counter())
                .with_warnings(planner.warnings());
            executor
                .execute(physical_plan.operator.as_mut())
                .map(|result| result.with_kind(optimized_plan.result_kind))
        });

        #[cfg(feature = "metrics")]
        {
            let elapsed_ms = start_time.elapsed().as_secs_f64() * 1000.0;
            self.record_query_metrics(language, elapsed_ms, &result);
        }
        #[cfg(not(feature = "metrics"))]
        let _ = language;

        result
    }

    /// The writer of the session's open transaction, through which an RDF
    /// update records its triples and graph operations and a read sees them;
    /// `None` without one. With `expands_copies` (a transaction the session
    /// opened, which later statements read), a `COPY`, `MOVE` or `ADD` is
    /// recorded as what it read.
    fn rdf_writer(&self, expands_copies: bool) -> Result<Option<crate::transaction::RdfWriter>> {
        let Some(changes) = self.changes.lock().clone() else {
            return Ok(None);
        };
        let writer = crate::transaction::RdfWriter::new(
            Arc::clone(&self.rdf_store),
            changes,
            Arc::clone(&self.transaction_manager),
        )?;
        Ok(Some(if expands_copies {
            writer.expanding_copies()
        } else {
            writer
        }))
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
        self.with_rdf_view(|view| {
            let data_store = self
                .rdf_store
                .graph_at(view, data_graph_name)
                .ok_or_else(|| {
                    grafeo_common::utils::error::Error::InvalidValue(format!(
                        "there is no named graph '{data_graph_name}' to validate"
                    ))
                })?;
            let shapes_store = self
                .rdf_store
                .graph_at(view, shapes_graph_name)
                .ok_or_else(|| {
                    grafeo_common::utils::error::Error::InvalidValue(format!(
                        "there is no named graph '{shapes_graph_name}' to read the shapes from"
                    ))
                })?;
            let executor = crate::validation::SessionSparqlExecutor::with_graph(
                self,
                data_graph_name.to_string(),
            );
            grafeo_core::graph::rdf::shacl::validate(
                data_store.at(view),
                shapes_store.at(view),
                Some(&executor),
            )
            .map_err(|e| grafeo_common::utils::error::Error::Internal(e.to_string()))
        })
    }
}
