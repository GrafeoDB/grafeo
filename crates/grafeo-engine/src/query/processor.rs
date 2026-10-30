//! Query processor that orchestrates the query pipeline.
//!
//! The `QueryProcessor` is the central component that executes queries through
//! the full pipeline: Parse → Bind → Optimize → Plan → Execute.
//!
//! It supports multiple query languages (GQL, Cypher, Gremlin, GraphQL) for LPG
//! and SPARQL for RDF (when the `rdf` feature is enabled).

use std::collections::HashMap;
use std::sync::Arc;

use grafeo_common::grafeo_debug_span;
use grafeo_common::types::{EpochId, TransactionId, Value};
use grafeo_common::utils::error::{Error, Result};
use grafeo_core::execution::operators::Recording;
use grafeo_core::graph::GraphStore;
use grafeo_core::graph::apply::GraphBackend;
use grafeo_core::graph::rowgroup::RowGroupStore;

use crate::catalog::Catalog;
use crate::database::QueryResult;
use crate::query::binder::Binder;
use crate::query::executor::Executor;
use crate::query::optimizer::Optimizer;
use crate::query::plan::{LogicalExpression, LogicalOperator, LogicalPlan};
use crate::query::planner::Planner;
use crate::transaction::TransactionManager;

/// Supported query languages.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum QueryLanguage {
    /// GQL (ISO/IEC 39075:2024) - default for LPG
    #[cfg(feature = "gql")]
    Gql,
    /// openCypher 9.0
    #[cfg(feature = "cypher")]
    Cypher,
    /// Apache TinkerPop Gremlin
    #[cfg(feature = "gremlin")]
    Gremlin,
    /// GraphQL for LPG
    #[cfg(feature = "graphql")]
    GraphQL,
    /// SQL/PGQ (SQL:2023 GRAPH_TABLE)
    #[cfg(feature = "sql-pgq")]
    SqlPgq,
    /// SPARQL 1.1 for RDF
    #[cfg(feature = "sparql")]
    Sparql,
    /// GraphQL for RDF
    #[cfg(all(feature = "graphql", feature = "triple-store"))]
    GraphQLRdf,
}

impl QueryLanguage {
    /// Returns whether this language targets LPG (vs RDF).
    #[must_use]
    pub const fn is_lpg(&self) -> bool {
        match self {
            #[cfg(feature = "gql")]
            Self::Gql => true,
            #[cfg(feature = "cypher")]
            Self::Cypher => true,
            #[cfg(feature = "gremlin")]
            Self::Gremlin => true,
            #[cfg(feature = "graphql")]
            Self::GraphQL => true,
            #[cfg(feature = "sql-pgq")]
            Self::SqlPgq => true,
            #[cfg(feature = "sparql")]
            Self::Sparql => false,
            #[cfg(all(feature = "graphql", feature = "triple-store"))]
            Self::GraphQLRdf => false,
            #[allow(unreachable_patterns)]
            _ => false,
        }
    }
}

/// Query parameters for prepared statements.
pub type QueryParams = HashMap<String, Value>;

/// Processes queries through the full pipeline.
///
/// The processor holds references to the stores and provides a unified
/// interface for executing queries in any supported language.
///
/// # Example
///
/// ```no_run
/// # use std::sync::Arc;
/// # use grafeo_core::graph::rowgroup::RowGroupStore;
/// use grafeo_engine::query::processor::{QueryProcessor, QueryLanguage};
///
/// # fn main() -> grafeo_common::utils::error::Result<()> {
/// let store = Arc::new(RowGroupStore::new());
/// let processor = QueryProcessor::for_lpg(store);
/// let result = processor.process("MATCH (n:Person) RETURN n", QueryLanguage::Gql, None)?;
/// # Ok(())
/// # }
/// ```
pub struct QueryProcessor {
    /// The built-in store, when the processor runs on one (`None` on a
    /// backend of one's own).
    lpg_store: Option<Arc<RowGroupStore>>,
    /// The store its reads read.
    graph_store: Arc<dyn GraphStore>,
    /// The backend a transaction's writes apply in and record their changes
    /// against (see
    /// [`with_transaction_context`](Self::with_transaction_context)); `None`
    /// when read-only.
    write_store: Option<Arc<dyn GraphBackend>>,
    /// Transaction manager for MVCC operations.
    transaction_manager: Arc<TransactionManager>,
    /// Catalog for schema and index metadata.
    catalog: Arc<Catalog>,
    /// Query optimizer.
    optimizer: Optimizer,
    /// Current transaction context (if any).
    transaction_context: Option<(EpochId, TransactionId)>,
    /// The storage key of the graph the store holds (`None`: the default
    /// graph), see [`with_graph`](Self::with_graph).
    graph: Option<String>,
    /// Whether the processor's own transaction manager follows the store's
    /// epoch (see `for_lpg`).
    follows_store_epoch: bool,
    /// RDF store for triple pattern queries (optional).
    #[cfg(feature = "triple-store")]
    rdf_store: Option<Arc<grafeo_core::graph::rdf::RdfStore>>,
}

impl QueryProcessor {
    /// Creates a new query processor for LPG queries, with a transaction
    /// manager of its own, which follows the store's epoch: a statement
    /// outside a transaction reads every write the store committed before
    /// it, the store's direct writes (`create_node` and the like, each at an
    /// epoch of its own) included.
    #[must_use]
    pub fn for_lpg(store: Arc<RowGroupStore>) -> Self {
        let optimizer = Optimizer::from_graph_store(&*store);
        let graph_store = Arc::clone(&store) as Arc<dyn GraphStore>;
        let write_store = Some(Arc::clone(&store) as Arc<dyn GraphBackend>);
        Self {
            lpg_store: Some(store),
            graph_store,
            write_store,
            transaction_manager: Arc::new(TransactionManager::new()),
            catalog: Arc::new(Catalog::new()),
            optimizer,
            transaction_context: None,
            graph: None,
            follows_store_epoch: true,
            #[cfg(feature = "triple-store")]
            rdf_store: None,
        }
    }

    /// Creates a new query processor with a transaction manager.
    #[must_use]
    pub fn for_lpg_with_transaction(
        store: Arc<RowGroupStore>,
        transaction_manager: Arc<TransactionManager>,
    ) -> Self {
        let optimizer = Optimizer::from_graph_store(&*store);
        let graph_store = Arc::clone(&store) as Arc<dyn GraphStore>;
        let write_store = Some(Arc::clone(&store) as Arc<dyn GraphBackend>);
        Self {
            lpg_store: Some(store),
            graph_store,
            write_store,
            transaction_manager,
            catalog: Arc::new(Catalog::new()),
            optimizer,
            transaction_context: None,
            graph: None,
            follows_store_epoch: false,
            #[cfg(feature = "triple-store")]
            rdf_store: None,
        }
    }

    /// Creates a query processor on a backend of one's own ([`GraphBackend`]).
    ///
    /// # Errors
    ///
    /// Returns an error if the internal arena allocation fails (out of memory).
    pub fn for_graph_store_with_transaction(
        store: Arc<dyn GraphBackend>,
        transaction_manager: Arc<TransactionManager>,
    ) -> Result<Self> {
        let optimizer = Optimizer::from_graph_store(&*store);
        let read_store = Arc::clone(&store) as Arc<dyn GraphStore>;
        Ok(Self {
            lpg_store: None,
            graph_store: read_store,
            write_store: Some(store),
            transaction_manager,
            catalog: Arc::new(Catalog::new()),
            optimizer,
            transaction_context: None,
            graph: None,
            follows_store_epoch: false,
            #[cfg(feature = "triple-store")]
            rdf_store: None,
        })
    }

    /// Creates a query processor from split read/write stores.
    ///
    /// # Errors
    ///
    /// Returns an error if the internal arena allocation fails (out of memory).
    pub fn for_stores_with_transaction(
        read_store: Arc<dyn GraphStore>,
        write_store: Option<Arc<dyn GraphBackend>>,
        transaction_manager: Arc<TransactionManager>,
    ) -> Result<Self> {
        let optimizer = Optimizer::from_graph_store(&*read_store);
        Ok(Self {
            lpg_store: None,
            graph_store: read_store,
            write_store,
            transaction_manager,
            catalog: Arc::new(Catalog::new()),
            optimizer,
            transaction_context: None,
            graph: None,
            follows_store_epoch: false,
            #[cfg(feature = "triple-store")]
            rdf_store: None,
        })
    }

    /// Sets the transaction context: the processor reads at `viewing_epoch`
    /// and writes as `transaction_id`, a transaction of its transaction
    /// manager. Its writes are claimed against other open transactions'
    /// (first writer wins) in the graph [`with_graph`](Self::with_graph)
    /// names, and recorded in the transaction's change set: committing the
    /// transaction (a session's commit, or
    /// [`TransactionManager::commit`]) makes them visible at its epoch,
    /// aborting it undoes them. A write once the transaction is no longer
    /// open fails.
    #[must_use]
    pub fn with_transaction_context(
        mut self,
        viewing_epoch: EpochId,
        transaction_id: TransactionId,
    ) -> Self {
        self.transaction_context = Some((viewing_epoch, transaction_id));
        self
    }

    /// Names the graph the processor's store holds, by its storage key
    /// (`name`, or `schema/name` for a graph of a schema): a processor in a
    /// transaction claims what it writes in that graph, so it conflicts with
    /// a session's transaction that writes the same node or edge there, and
    /// never with one that writes the same id in another graph. Without
    /// it, the store is the default graph's.
    #[must_use]
    pub fn with_graph(mut self, graph: impl Into<String>) -> Self {
        self.graph = Some(graph.into());
        self
    }

    /// Sets a custom catalog.
    #[must_use]
    pub fn with_catalog(mut self, catalog: Arc<Catalog>) -> Self {
        self.catalog = catalog;
        self
    }

    /// Sets a custom optimizer.
    #[must_use]
    pub fn with_optimizer(mut self, optimizer: Optimizer) -> Self {
        self.optimizer = optimizer;
        self
    }

    /// Processes a query string and returns results.
    ///
    /// Pipeline:
    /// 1. Parse (language-specific parser → AST)
    /// 2. Translate (AST → LogicalPlan)
    /// 3. Bind (semantic validation)
    /// 4. Optimize (filter pushdown, join reorder, etc.)
    /// 5. Plan (logical → physical operators)
    /// 6. Execute (run operators, collect results)
    ///
    /// # Arguments
    ///
    /// * `query` - The query string
    /// * `language` - Which query language to use
    /// * `params` - Optional query parameters for prepared statements
    ///
    /// # Errors
    ///
    /// Returns an error if any stage of the pipeline fails.
    pub fn process(
        &self,
        query: &str,
        language: QueryLanguage,
        params: Option<&QueryParams>,
    ) -> Result<QueryResult> {
        if language.is_lpg() {
            self.process_lpg(query, language, params)
        } else {
            #[cfg(feature = "triple-store")]
            {
                self.process_rdf(query, language, params)
            }
            #[cfg(not(feature = "triple-store"))]
            {
                Err(grafeo_common::utils::error::Error::from(
                    grafeo_common::utils::error::QueryError::unsupported(
                        "this build has no RDF support (the `triple-store` feature)",
                    ),
                ))
            }
        }
    }

    /// Processes an LPG query (GQL, Cypher, Gremlin, GraphQL).
    fn process_lpg(
        &self,
        query: &str,
        language: QueryLanguage,
        params: Option<&QueryParams>,
    ) -> Result<QueryResult> {
        let start_time = grafeo_common::clock::Instant::now();

        // 1. Parse and translate to logical plan
        let mut logical_plan = self.translate_lpg(query, language)?;

        // 2. Substitute parameters if provided (merge defaults from the plan first)
        let has_defaults = !logical_plan.default_params.is_empty();
        // A parameter nobody supplied fails here, before planning; only an
        // EXPLAIN without parameters shows the plan with them unresolved.
        if params.is_some() || has_defaults || !logical_plan.explain {
            let merged = if has_defaults {
                let mut merged = logical_plan.default_params.clone();
                if let Some(params) = params {
                    merged.extend(params.iter().map(|(k, v)| (k.clone(), v.clone())));
                }
                merged
            } else {
                params.cloned().unwrap_or_default()
            };
            substitute_params(&mut logical_plan, &merged)?;
        }

        // 3. Semantic validation
        let mut binder = Binder::new();
        let _binding_context = binder.bind(&logical_plan)?;

        // 4. Optimize the plan
        let optimized_plan = self.optimizer.optimize(logical_plan)?;

        // A processor with its own transaction manager reads what the
        // store's direct writes stamped (each at an epoch of its own) outside
        // a transaction: its manager follows the store's epoch.
        if self.follows_store_epoch
            && self.transaction_context.is_none()
            && let Some(store) = &self.lpg_store
        {
            self.transaction_manager.sync_epoch(store.current_epoch());
        }

        // 4a. EXPLAIN: annotate pushdown hints and return the plan tree
        if optimized_plan.explain {
            let mut plan = optimized_plan;
            annotate_pushdown_hints(&mut plan.root, self.graph_store.as_ref());
            return Ok(explain_result(&plan));
        }

        // A write outside a transaction runs in one of its own, which
        // commits when the statement succeeds and rolls back when it fails:
        // its writes apply in the backend's transaction like any other.
        if self.transaction_context.is_none()
            && optimized_plan.root.has_mutations()
            && self.write_store.is_some()
        {
            let transaction = self.transaction_manager.begin();
            let epoch = self
                .transaction_manager
                .start_epoch(transaction)
                .unwrap_or_else(|| self.transaction_manager.current_epoch());
            return match self.run_plan(&optimized_plan, Some((epoch, transaction)), start_time) {
                Ok(result) => {
                    self.transaction_manager.commit(transaction)?;
                    Ok(result)
                }
                Err(error) => {
                    // The statement's error is the one to report; a failed
                    // rollback poisons the manager, which the next write
                    // reports.
                    let _aborted = self.transaction_manager.abort(transaction);
                    Err(error)
                }
            };
        }
        self.run_plan(&optimized_plan, self.transaction_context, start_time)
    }

    /// Plans and runs `optimized_plan` in `context` (its epoch and
    /// transaction), or outside a transaction.
    fn run_plan(
        &self,
        optimized_plan: &LogicalPlan,
        context: Option<(EpochId, TransactionId)>,
        start_time: grafeo_common::clock::Instant,
    ) -> Result<QueryResult> {
        // 5. Convert to physical plan with transaction context
        // Read-only fast path: safe when no mutations AND no active transaction
        // (an active transaction may have prior uncommitted writes from earlier statements)
        let is_read_only = !optimized_plan.root.has_mutations() && context.is_none();
        // Outside a transaction the statement reads at the published epoch,
        // pinned for its run (#412 decision Q6); dropped when it returns.
        let _pin = context
            .is_none()
            .then(|| self.transaction_manager.pin_read());
        let planner = if let Some((epoch, transaction_id)) = context {
            Planner::with_context(
                Arc::clone(&self.graph_store),
                self.write_store.as_ref().map(Arc::clone),
                Arc::clone(&self.transaction_manager),
                Some(transaction_id),
                epoch,
            )
            .with_recording(self.recording(transaction_id)?)
        } else {
            Planner::with_context(
                Arc::clone(&self.graph_store),
                self.write_store.as_ref().map(Arc::clone),
                Arc::clone(&self.transaction_manager),
                None,
                self.transaction_manager.current_epoch(),
            )
        }
        .with_read_only(is_read_only);
        let mut physical_plan = planner.plan(optimized_plan)?;

        // 6. Execute and collect results
        let executor = Executor::with_columns(physical_plan.columns.clone())
            .with_read_counter(planner.read_counter());
        let result = executor.execute(physical_plan.operator.as_mut())?;

        // Add execution metrics (the executor set `rows_scanned`).
        let result = QueryResult {
            execution_time_ms: Some(start_time.elapsed().as_secs_f64() * 1000.0),
            ..result
        };

        Ok(result)
    }

    /// Where the writes of `transaction_id` go and are recorded: the write
    /// store and the transaction's changes in the processor's graph. `None`
    /// without a write store, or when the transaction is not open (a write
    /// then fails).
    ///
    /// # Errors
    ///
    /// Fails when the transaction wrote the graph through another store.
    fn recording(&self, transaction_id: TransactionId) -> Result<Option<Recording>> {
        let Some(target) = &self.write_store else {
            return Ok(None);
        };
        let Some(changes) = self.transaction_manager.changes(transaction_id) else {
            return Ok(None);
        };
        changes
            .recording(
                &self.transaction_manager,
                self.graph.as_deref(),
                Arc::clone(target),
            )
            .map(Some)
    }

    /// Translates an LPG query to a logical plan.
    fn translate_lpg(&self, query: &str, language: QueryLanguage) -> Result<LogicalPlan> {
        let _span = grafeo_debug_span!("grafeo::query::parse", ?language);
        match language {
            #[cfg(feature = "gql")]
            QueryLanguage::Gql => {
                use crate::query::translators::gql;
                gql::translate(query)
            }
            #[cfg(feature = "cypher")]
            QueryLanguage::Cypher => {
                use crate::query::translators::cypher;
                cypher::translate(query)
            }
            #[cfg(feature = "gremlin")]
            QueryLanguage::Gremlin => {
                use crate::query::translators::gremlin;
                gremlin::translate(query)
            }
            #[cfg(feature = "graphql")]
            QueryLanguage::GraphQL => {
                use crate::query::translators::graphql;
                graphql::translate(query)
            }
            #[cfg(feature = "sql-pgq")]
            QueryLanguage::SqlPgq => {
                use crate::query::translators::sql_pgq;
                sql_pgq::translate(query)
            }
            #[allow(unreachable_patterns)]
            _ => Err(Error::Internal(format!(
                "Language {:?} is not an LPG language ({} bytes of query)",
                language,
                query.len()
            ))),
        }
    }

    /// The built-in store, when the processor runs on one.
    #[must_use]
    pub fn lpg_store(&self) -> Option<&Arc<RowGroupStore>> {
        self.lpg_store.as_ref()
    }

    /// Returns a reference to the catalog.
    #[must_use]
    pub fn catalog(&self) -> &Arc<Catalog> {
        &self.catalog
    }

    /// Returns a reference to the optimizer.
    #[must_use]
    pub fn optimizer(&self) -> &Optimizer {
        &self.optimizer
    }
}

impl QueryProcessor {
    /// Returns a reference to the transaction manager.
    #[must_use]
    pub fn transaction_manager(&self) -> &Arc<TransactionManager> {
        &self.transaction_manager
    }
}

// =========================================================================
// RDF-specific methods (gated behind `rdf` feature)
// =========================================================================

#[cfg(feature = "triple-store")]
impl QueryProcessor {
    /// Creates a new query processor with both LPG and RDF stores.
    #[must_use]
    pub fn with_rdf(
        lpg_store: Arc<RowGroupStore>,
        rdf_store: Arc<grafeo_core::graph::rdf::RdfStore>,
    ) -> Self {
        let optimizer = Optimizer::from_graph_store(&*lpg_store);
        let graph_store = Arc::clone(&lpg_store) as Arc<dyn GraphStore>;
        let write_store = Some(Arc::clone(&lpg_store) as Arc<dyn GraphBackend>);
        Self {
            lpg_store: Some(lpg_store),
            graph_store,
            write_store,
            transaction_manager: Arc::new(TransactionManager::new()),
            catalog: Arc::new(Catalog::new()),
            optimizer,
            transaction_context: None,
            graph: None,
            follows_store_epoch: false,
            rdf_store: Some(rdf_store),
        }
    }

    /// Returns a reference to the RDF store (if configured).
    #[must_use]
    pub fn rdf_store(&self) -> Option<&Arc<grafeo_core::graph::rdf::RdfStore>> {
        self.rdf_store.as_ref()
    }

    /// Processes an RDF query (SPARQL, GraphQL-RDF).
    fn process_rdf(
        &self,
        query: &str,
        language: QueryLanguage,
        params: Option<&QueryParams>,
    ) -> Result<QueryResult> {
        use crate::query::planner::rdf::RdfPlanner;

        let rdf_store = self.rdf_store.as_ref().ok_or_else(|| {
            Error::Internal("RDF store not configured for this processor".to_string())
        })?;

        // 1. Parse and translate to logical plan (SPARQL binds its
        // parameters as initial bindings)
        let mut logical_plan = self.translate_rdf(query, language, params)?;

        // 2. Substitute parameters if provided (merge defaults from the plan first)
        substitute_with_defaults(&mut logical_plan, params)?;

        // 3. Semantic validation
        let mut binder = Binder::new();
        let _binding_context = binder.bind(&logical_plan)?;

        // 3. Optimize the plan (use RDF statistics for cost-based optimization)
        let rdf_optimizer = {
            let stats = rdf_store.get_or_collect_statistics();
            Optimizer::from_rdf_statistics((*stats).clone())
        };
        let optimized_plan = rdf_optimizer.optimize(logical_plan)?;

        // 3a. EXPLAIN: return the plan tree without executing.
        // Includes physical operator names by planning with profiling to collect entries.
        if optimized_plan.explain {
            let planner = RdfPlanner::new(Arc::clone(rdf_store));
            let (_, entries) = planner.plan_profiled(&optimized_plan)?;
            return Ok(physical_explain_result(&optimized_plan, entries));
        }

        // 4. Plan and execute. An update runs in a private transaction of
        // the processor's, whose triples apply once it succeeds.
        // GraphQL over RDF gives plain values, SPARQL gives terms.
        #[cfg(all(feature = "graphql", feature = "triple-store"))]
        let query_values = language == QueryLanguage::GraphQLRdf;
        #[cfg(not(all(feature = "graphql", feature = "triple-store")))]
        let query_values = false;
        let run = |writer: Option<crate::transaction::RdfWriter>| -> Result<QueryResult> {
            let planner = RdfPlanner::new(Arc::clone(rdf_store))
                .with_writer(writer)
                .with_query_values(query_values);

            // EXPLAIN ANALYZE (PROFILE): execute with instrumentation, report stats.
            if optimized_plan.profile {
                let (mut physical_plan, entries) = planner.plan_profiled(&optimized_plan)?;

                let start = grafeo_common::clock::Instant::now();
                let executor = Executor::with_columns(physical_plan.columns.clone());
                let _result = executor.execute(physical_plan.operator.as_mut())?;
                let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;

                let tree = crate::query::profile::build_profile_tree(
                    &optimized_plan.root,
                    &mut entries.into_iter(),
                );
                return Ok(crate::query::profile::profile_result(&tree, elapsed_ms));
            }

            let mut physical_plan = planner.plan(&optimized_plan)?;
            let executor = Executor::with_columns(physical_plan.columns.clone());
            executor
                .execute(physical_plan.operator.as_mut())
                .map(|result| result.with_kind(optimized_plan.result_kind))
        };
        if optimized_plan.root.has_mutations() {
            crate::transaction::update_privately(rdf_store, &self.transaction_manager, |writer| {
                run(Some(writer))
            })
        } else {
            run(None)
        }
    }

    /// Translates an RDF query to a logical plan; a SPARQL query binds
    /// `params` as its initial bindings.
    fn translate_rdf(
        &self,
        query: &str,
        language: QueryLanguage,
        params: Option<&std::collections::HashMap<String, Value>>,
    ) -> Result<LogicalPlan> {
        // Only SPARQL takes them here.
        #[cfg(not(feature = "sparql"))]
        let _ = params;
        match language {
            #[cfg(feature = "sparql")]
            QueryLanguage::Sparql => {
                use crate::query::translators::sparql;
                sparql::translate_with_bindings(query, &params.cloned().unwrap_or_default())
            }
            #[cfg(all(feature = "graphql", feature = "triple-store"))]
            QueryLanguage::GraphQLRdf => {
                use crate::query::translators::graphql_rdf;
                // Default namespace for GraphQL-RDF queries
                graphql_rdf::translate(query, "http://example.org/")
            }
            // Only a build with a property graph language has other ones.
            #[cfg(lpg_language)]
            _ => Err(Error::Internal(format!(
                "Language {:?} is not an RDF language",
                language
            ))),
        }
    }
}

/// Annotates filter operators in the plan with pushdown hints.
///
/// Walks the whole plan tree looking for `Filter -> NodeScan` patterns and
/// names the path the planner takes for each (see `plan_filter`): a seek, a
/// hash join on values, an index lookup, a range scan or a label-first scan
/// (see [`scan_hint`](crate::query::planner::lpg::filter::scan_hint)). A
/// node scan below filters that require more of its labels shows the label
/// the planner scans, below any operator, as the planner chooses it for every
/// filter it plans (see
/// [`with_smallest_scan_label`](crate::query::planner::lpg::scan::with_smallest_scan_label)).
/// `op` is the statement's root: a statement that writes plans no hash join
/// on values.
pub(crate) fn annotate_pushdown_hints(
    op: &mut LogicalOperator,
    store: &dyn grafeo_core::graph::GraphStore,
) {
    let writes = op.has_mutations();
    annotate_hints(op, store, writes, false);
}

/// [`annotate_pushdown_hints`] below the root of a statement that `writes`,
/// in a part of it that runs `after_a_write` of the statement (see
/// [`right_side_runs_after_a_write`](crate::query::planner::lpg::after_write::right_side_runs_after_a_write)
/// and
/// [`subquery_runs_after_a_write`](crate::query::planner::lpg::after_write::subquery_runs_after_a_write)),
/// where the planner looks nothing up while planning.
fn annotate_hints(
    op: &mut LogicalOperator,
    store: &dyn grafeo_core::graph::GraphStore,
    writes: bool,
    after_a_write: bool,
) {
    use crate::query::planner::lpg::after_write::{
        right_side_runs_after_a_write, subquery_runs_after_a_write,
    };

    match op {
        LogicalOperator::Join(crate::query::plan::JoinOp { left, right, .. })
        | LogicalOperator::LeftJoin(crate::query::plan::LeftJoinOp { left, right, .. }) => {
            let later = after_a_write || right_side_runs_after_a_write(left);
            annotate_hints(left, store, writes, after_a_write);
            annotate_hints(right, store, writes, later);
            return;
        }
        LogicalOperator::Apply(apply) => {
            let later = after_a_write || subquery_runs_after_a_write(apply);
            annotate_hints(&mut apply.input, store, writes, after_a_write);
            annotate_hints(&mut apply.subplan, store, writes, later);
            return;
        }
        _ => {}
    }
    if let LogicalOperator::Filter(filter) = op {
        // The label the planner scans, chosen at the top of a chain of filters
        if let Some(reordered) = crate::query::planner::lpg::scan::with_scan_labels(filter, store) {
            *filter = reordered;
        }
        // Recurse into children first
        annotate_hints(&mut filter.input, store, writes, after_a_write);
        // A subquery of the predicate is planned like any plan, with the
        // labels it scans chosen the same way.
        let _ = crate::query::planner::lpg::subquery::visit_subqueries(
            &mut filter.predicate,
            &mut |subquery| {
                if let crate::query::plan::LogicalExpression::ExistsSubquery(plan)
                | crate::query::plan::LogicalExpression::CountSubquery(plan)
                | crate::query::plan::LogicalExpression::ValueSubquery(plan) = subquery
                {
                    annotate_hints(plan, store, writes, after_a_write);
                }
                Ok(())
            },
        );

        // A seek or a hash join replaces the scan below the filter
        let replaced = crate::query::planner::lpg::seek::checked_scan(filter).and_then(|below| {
            seek_hint(&filter.predicate, below.scan, store, after_a_write)
                .or_else(|| value_join_hint(filter, writes))
        });
        if let Some(hint) = replaced {
            // It takes over the checks between it and the scan: they run in
            // its filter or on the nodes it scans, not on a scan of their own.
            filter.pushdown_hint = Some(hint);
            let mut below = filter.input.as_mut();
            while let LogicalOperator::Filter(check) = below {
                check.pushdown_hint = None;
                below = check.input.as_mut();
            }
        } else if let LogicalOperator::NodeScan(scan) = filter.input.as_ref() {
            filter.pushdown_hint =
                crate::query::planner::lpg::filter::scan_hint(&filter.predicate, scan, store);
        }
        return;
    }
    if let LogicalOperator::Merge(merge) = op {
        merge.index_hint = merge_index_hint(merge, store);
    }
    let taken = std::mem::replace(op, LogicalOperator::Empty);
    *op = taken.map_children(|mut child| {
        annotate_hints(&mut child, store, writes, after_a_write);
        child
    });
}

/// The property whose index `merge` looks its node up through: the first of
/// its match properties with a property index and a value that is not a
/// null literal, as the merge operator picks it (a null value matches no
/// index entry, so a row with one scans the label).
fn merge_index_hint(
    merge: &crate::query::plan::MergeOp,
    store: &dyn grafeo_core::graph::GraphStore,
) -> Option<String> {
    use crate::query::plan::LogicalExpression;
    use grafeo_common::types::Value;

    merge
        .match_properties
        .iter()
        .find(|(key, value)| {
            !matches!(value, LogicalExpression::Literal(Value::Null))
                && store.has_property_index(key)
        })
        .map(|(key, _)| key.clone())
}

/// The seek the planner makes of a filter over `scan`: an ID, or an indexed
/// property per input row, or once for a filter that
/// runs `after_a_write` of the statement. None after a write in the scan's
/// input, which the scan reads whole first.
fn seek_hint(
    predicate: &LogicalExpression,
    scan: &crate::query::plan::NodeScanOp,
    store: &dyn grafeo_core::graph::GraphStore,
    after_a_write: bool,
) -> Option<crate::query::plan::PushdownHint> {
    use crate::query::plan::PushdownHint;
    use grafeo_core::execution::operators::SeekKey;

    if scan
        .input
        .as_deref()
        .is_some_and(LogicalOperator::has_mutations)
    {
        return None;
    }
    let seek = crate::query::planner::lpg::seek::choose_seek(
        predicate,
        scan,
        |p| store.has_property_index(p),
        after_a_write,
    )?;
    Some(match seek.key {
        SeekKey::Id => PushdownHint::IdSeek,
        SeekKey::Property(property) => PushdownHint::IndexLookup { property },
    })
}

/// The hash join on values the planner makes of `filter` in a statement that
/// `writes` or not, if any (see
/// [`value_join`](crate::query::planner::lpg::value_join::value_join)).
fn value_join_hint(
    filter: &crate::query::plan::FilterOp,
    writes: bool,
) -> Option<crate::query::plan::PushdownHint> {
    let join = crate::query::planner::lpg::value_join::value_join(filter, writes)?;
    Some(crate::query::plan::PushdownHint::HashJoin {
        keys: join
            .keys
            .iter()
            .map(|key| (key.scanned.clone(), key.row.clone()))
            .collect(),
    })
}

/// Builds a `QueryResult` containing the EXPLAIN plan tree text.
pub(crate) fn explain_result(plan: &LogicalPlan) -> QueryResult {
    let tree_text = plan.root.explain_tree();
    QueryResult {
        columns: vec!["plan".to_string()],
        column_types: vec![grafeo_common::types::LogicalType::String],
        rows: vec![vec![Value::String(tree_text.into())]],
        execution_time_ms: None,
        rows_scanned: None,
        status_message: None,
        gql_status: grafeo_common::utils::GqlStatus::SUCCESS,
        counters: Default::default(),
        profile: None,
        kind: crate::database::ResultKind::Rows,
    }
}

/// Formats a physical EXPLAIN result showing both the logical plan and the
/// physical operator names mapped to each logical operator.
#[cfg(feature = "triple-store")]
pub(crate) fn physical_explain_result(
    plan: &LogicalPlan,
    entries: Vec<crate::query::profile::ProfileEntry>,
) -> QueryResult {
    let tree = crate::query::profile::build_profile_tree(&plan.root, &mut entries.into_iter());

    let mut output = String::new();
    format_physical_node(&mut output, &tree, 0);

    QueryResult {
        columns: vec!["plan".to_string()],
        column_types: vec![grafeo_common::types::LogicalType::String],
        rows: vec![vec![Value::String(output.into())]],
        execution_time_ms: None,
        rows_scanned: None,
        status_message: None,
        gql_status: grafeo_common::utils::GqlStatus::SUCCESS,
        counters: Default::default(),
        profile: None,
        kind: crate::database::ResultKind::Rows,
    }
}

/// Recursively formats a physical plan node showing operator name and label.
#[cfg(feature = "triple-store")]
fn format_physical_node(out: &mut String, node: &crate::query::profile::ProfileNode, depth: usize) {
    use std::fmt::Write;
    let indent = "  ".repeat(depth);
    let _ = writeln!(out, "{indent}{} {}", node.name, node.label);
    for child in &node.children {
        format_physical_node(out, child, depth + 1);
    }
}

/// Substitutes the parameters of `plan`: the defaults its query declares
/// (GraphQL variables with a default), then `params` over them. A parameter
/// nobody supplied fails here, before planning; only an `EXPLAIN` without
/// parameters shows the plan with them unresolved.
///
/// # Errors
///
/// As [`substitute_params`].
#[cfg(feature = "triple-store")]
pub(crate) fn substitute_with_defaults(
    plan: &mut LogicalPlan,
    params: Option<&QueryParams>,
) -> Result<()> {
    let has_defaults = !plan.default_params.is_empty();
    if params.is_none() && !has_defaults && plan.explain {
        return Ok(());
    }
    let merged = if has_defaults {
        let mut merged = plan.default_params.clone();
        if let Some(params) = params {
            merged.extend(params.iter().map(|(k, v)| (k.clone(), v.clone())));
        }
        merged
    } else {
        params.cloned().unwrap_or_default()
    };
    substitute_params(plan, &merged)
}

/// `params`, or an error when a value nests lists, maps and paths deeper
/// than a stored value may: a query would clone, compare and drop it with a
/// call per level, and a deep enough one overflows the stack. The refused
/// parameters are dropped without recursion
/// ([`Value::drop_without_recursion`]), as their drop alone would overflow
/// it too.
///
/// # Errors
///
/// An invalid-value error naming the parameter.
#[cfg(any(
    feature = "gql",
    feature = "cypher",
    feature = "gremlin",
    feature = "graphql",
    feature = "sql-pgq"
))]
pub(crate) fn refuse_deep_params(
    params: std::collections::HashMap<String, Value>,
) -> Result<std::collections::HashMap<String, Value>> {
    let Some(name) = params
        .iter()
        .find(|(_, value)| grafeo_common::storage::value_codec::nests_too_deep(value))
        .map(|(name, _)| name.clone())
    else {
        return Ok(params);
    };
    for value in params.into_values() {
        value.drop_without_recursion();
    }
    Err(Error::InvalidValue(format!(
        "parameter ${name}: the value nests lists, maps and paths more than {} levels deep",
        grafeo_common::storage::MAX_PROPERTY_VALUE_DEPTH
    )))
}

/// Substitutes parameters in a logical plan with their values.
///
/// # Errors
///
/// Returns an error if a referenced parameter is not found in `params`.
pub fn substitute_params(plan: &mut LogicalPlan, params: &QueryParams) -> Result<()> {
    // A value nested deeper than a stored one is refused before a query
    // clones, compares or drops it, which recurse once per level: a deep
    // enough one would overflow the stack (and trap a WASM module).
    if let Some((name, _)) = params
        .iter()
        .find(|(_, value)| grafeo_common::storage::value_codec::nests_too_deep(value))
    {
        return Err(Error::InvalidValue(format!(
            "parameter ${name}: the value nests lists, maps and paths more than {} levels deep",
            grafeo_common::storage::MAX_PROPERTY_VALUE_DEPTH
        )));
    }
    substitute_in_operator(&mut plan.root, params)
}

/// Recursively substitutes parameters in an operator.
fn substitute_in_operator(op: &mut LogicalOperator, params: &QueryParams) -> Result<()> {
    #[allow(clippy::wildcard_imports)]
    use crate::query::plan::*;

    match op {
        LogicalOperator::Filter(filter) => {
            substitute_in_expression(&mut filter.predicate, params)?;
            substitute_in_operator(&mut filter.input, params)?;
        }
        LogicalOperator::Return(ret) => {
            for item in &mut ret.items {
                // An unaliased column that reads a parameter is named after
                // the query text (`$x`), not after the value replacing it.
                let name = item.alias.is_none().then(|| {
                    crate::query::planner::common::output_column_name(None, &item.expression)
                });
                substitute_in_expression(&mut item.expression, params)?;
                if let Some(name) = name
                    && name
                        != crate::query::planner::common::output_column_name(None, &item.expression)
                {
                    item.alias = Some(name);
                }
            }
            substitute_in_operator(&mut ret.input, params)?;
        }
        LogicalOperator::Project(proj) => {
            for p in &mut proj.projections {
                substitute_in_expression(&mut p.expression, params)?;
            }
            substitute_in_operator(&mut proj.input, params)?;
        }
        LogicalOperator::NodeScan(scan) => {
            if let Some(input) = &mut scan.input {
                substitute_in_operator(input, params)?;
            }
        }
        LogicalOperator::EdgeScan(scan) => {
            if let Some(input) = &mut scan.input {
                substitute_in_operator(input, params)?;
            }
        }
        LogicalOperator::Expand(expand) => {
            substitute_in_operator(&mut expand.input, params)?;
        }
        LogicalOperator::Join(join) => {
            substitute_in_operator(&mut join.left, params)?;
            substitute_in_operator(&mut join.right, params)?;
            for cond in &mut join.conditions {
                substitute_in_expression(&mut cond.left, params)?;
                substitute_in_expression(&mut cond.right, params)?;
            }
        }
        LogicalOperator::LeftJoin(join) => {
            substitute_in_operator(&mut join.left, params)?;
            substitute_in_operator(&mut join.right, params)?;
            if let Some(cond) = &mut join.condition {
                substitute_in_expression(cond, params)?;
            }
        }
        LogicalOperator::Aggregate(agg) => {
            for expr in &mut agg.group_by {
                substitute_in_expression(expr, params)?;
            }
            for agg_expr in &mut agg.aggregates {
                for expr in [&mut agg_expr.expression, &mut agg_expr.expression2]
                    .into_iter()
                    .flatten()
                {
                    substitute_in_expression(expr, params)?;
                }
            }
            if let Some(having) = &mut agg.having {
                substitute_in_expression(having, params)?;
            }
            substitute_in_operator(&mut agg.input, params)?;
        }
        LogicalOperator::Sort(sort) => {
            for key in &mut sort.keys {
                substitute_in_expression(&mut key.expression, params)?;
            }
            substitute_in_operator(&mut sort.input, params)?;
        }
        LogicalOperator::Limit(limit) => {
            resolve_count_param(&mut limit.count, params)?;
            substitute_in_operator(&mut limit.input, params)?;
        }
        LogicalOperator::Skip(skip) => {
            resolve_count_param(&mut skip.count, params)?;
            substitute_in_operator(&mut skip.input, params)?;
        }
        LogicalOperator::Distinct(distinct) => {
            substitute_in_operator(&mut distinct.input, params)?;
        }
        LogicalOperator::CreateNode(create) => {
            for (_, expr) in &mut create.properties {
                substitute_in_expression(expr, params)?;
            }
            if let Some(input) = &mut create.input {
                substitute_in_operator(input, params)?;
            }
        }
        LogicalOperator::CreateEdge(create) => {
            for (_, expr) in &mut create.properties {
                substitute_in_expression(expr, params)?;
            }
            substitute_in_operator(&mut create.input, params)?;
        }
        LogicalOperator::Create(create) => {
            for expr in create.property_values_mut() {
                substitute_in_expression(expr, params)?;
            }
            if let Some(input) = &mut create.input {
                substitute_in_operator(input, params)?;
            }
        }
        LogicalOperator::DeleteNode(delete) => {
            substitute_in_operator(&mut delete.input, params)?;
        }
        LogicalOperator::DeleteEdge(delete) => {
            substitute_in_operator(&mut delete.input, params)?;
        }
        LogicalOperator::SetProperty(set) => {
            for (_, expr) in &mut set.properties {
                substitute_in_expression(expr, params)?;
            }
            substitute_in_operator(&mut set.input, params)?;
        }
        LogicalOperator::Union(union) => {
            for input in &mut union.inputs {
                substitute_in_operator(input, params)?;
            }
        }
        LogicalOperator::AntiJoin(anti) => {
            substitute_in_operator(&mut anti.left, params)?;
            substitute_in_operator(&mut anti.right, params)?;
        }
        LogicalOperator::MarkJoin(mark) => {
            substitute_in_operator(&mut mark.left, params)?;
            substitute_in_operator(&mut mark.right, params)?;
        }
        LogicalOperator::Bind(bind) => {
            substitute_in_expression(&mut bind.expression, params)?;
            substitute_in_operator(&mut bind.input, params)?;
        }
        LogicalOperator::PropertyPath(_) | LogicalOperator::NamedGraphs(_) => {}
        LogicalOperator::TripleScan(scan) => {
            if let Some(input) = &mut scan.input {
                substitute_in_operator(input, params)?;
            }
        }
        LogicalOperator::Unwind(unwind) => {
            substitute_in_expression(&mut unwind.expression, params)?;
            substitute_in_operator(&mut unwind.input, params)?;
        }
        LogicalOperator::MapCollect(mc) => {
            substitute_in_operator(&mut mc.input, params)?;
        }
        LogicalOperator::Merge(merge) => {
            for (_, expr) in &mut merge.match_properties {
                substitute_in_expression(expr, params)?;
            }
            for (_, expr) in &mut merge.on_create {
                substitute_in_expression(expr, params)?;
            }
            for (_, expr) in &mut merge.on_match {
                substitute_in_expression(expr, params)?;
            }
            substitute_in_operator(&mut merge.input, params)?;
        }
        LogicalOperator::MergeRelationship(merge_rel) => {
            for (_, expr) in &mut merge_rel.match_properties {
                substitute_in_expression(expr, params)?;
            }
            for (_, expr) in &mut merge_rel.on_create {
                substitute_in_expression(expr, params)?;
            }
            for (_, expr) in &mut merge_rel.on_match {
                substitute_in_expression(expr, params)?;
            }
            substitute_in_operator(&mut merge_rel.input, params)?;
        }
        LogicalOperator::AddLabel(add_label) => {
            substitute_in_operator(&mut add_label.input, params)?;
        }
        LogicalOperator::RemoveLabel(remove_label) => {
            substitute_in_operator(&mut remove_label.input, params)?;
        }
        LogicalOperator::ShortestPath(sp) => {
            if let Some(condition) = &mut sp.edge_condition {
                substitute_in_expression(&mut condition.predicate, params)?;
            }
            substitute_in_operator(&mut sp.input, params)?;
        }
        // SPARQL Update operators
        LogicalOperator::InsertTriple(insert) => {
            if let Some(ref mut input) = insert.input {
                substitute_in_operator(input, params)?;
            }
        }
        LogicalOperator::DeleteTriple(delete) => {
            if let Some(ref mut input) = delete.input {
                substitute_in_operator(input, params)?;
            }
        }
        LogicalOperator::Modify(modify) => {
            substitute_in_operator(&mut modify.where_clause, params)?;
        }
        LogicalOperator::ClearGraph(_)
        | LogicalOperator::CreateGraph(_)
        | LogicalOperator::DropGraph(_)
        | LogicalOperator::LoadGraph(_)
        | LogicalOperator::CopyGraph(_)
        | LogicalOperator::MoveGraph(_)
        | LogicalOperator::AddGraph(_) => {}
        LogicalOperator::HorizontalAggregate(op) => {
            substitute_in_operator(&mut op.input, params)?;
        }
        LogicalOperator::Empty => {}
        LogicalOperator::VectorScan(scan) => {
            substitute_in_expression(&mut scan.query_vector, params)?;
            if let Some(ref mut input) = scan.input {
                substitute_in_operator(input, params)?;
            }
        }
        LogicalOperator::VectorJoin(join) => {
            substitute_in_expression(&mut join.query_vector, params)?;
            substitute_in_operator(&mut join.input, params)?;
        }
        LogicalOperator::TextScan(scan) => {
            substitute_in_expression(&mut scan.query, params)?;
        }
        LogicalOperator::Except(except) => {
            substitute_in_operator(&mut except.left, params)?;
            substitute_in_operator(&mut except.right, params)?;
        }
        LogicalOperator::Intersect(intersect) => {
            substitute_in_operator(&mut intersect.left, params)?;
            substitute_in_operator(&mut intersect.right, params)?;
        }
        LogicalOperator::Otherwise(otherwise) => {
            substitute_in_operator(&mut otherwise.left, params)?;
            substitute_in_operator(&mut otherwise.right, params)?;
        }
        LogicalOperator::Apply(apply) => {
            substitute_in_operator(&mut apply.input, params)?;
            substitute_in_operator(&mut apply.subplan, params)?;
        }
        // ParameterScan and DelimScan have no expressions to substitute
        LogicalOperator::ParameterScan(_) | LogicalOperator::DelimScan(_) => {}
        LogicalOperator::MultiWayJoin(mwj) => {
            for input in &mut mwj.inputs {
                substitute_in_operator(input, params)?;
            }
            for cond in &mut mwj.conditions {
                substitute_in_expression(&mut cond.left, params)?;
                substitute_in_expression(&mut cond.right, params)?;
            }
        }
        // DDL operators have no expressions to substitute
        LogicalOperator::CreatePropertyGraph(_) | LogicalOperator::DropPropertyGraph(_) => {}
        // Procedure arguments take their parameters here like any other
        // expression, so a parameter runs exactly like the same literal and a
        // missing one fails before the procedure runs.
        LogicalOperator::CallProcedure(call) => {
            for argument in &mut call.arguments {
                substitute_in_expression(argument, params)?;
            }
        }
        // LoadData: file path is a literal, no parameter substitution needed
        LogicalOperator::LoadData(_) => {}
        // Construct: template uses variables, substitute in the WHERE input
        LogicalOperator::Construct(construct) => {
            substitute_in_operator(&mut construct.input, params)?;
        }
    }
    Ok(())
}

/// Resolves a `CountExpr::Parameter` by looking up the parameter value.
fn resolve_count_param(
    count: &mut crate::query::plan::CountExpr,
    params: &QueryParams,
) -> Result<()> {
    use crate::query::plan::CountExpr;
    use grafeo_common::utils::error::QueryError;

    if let CountExpr::Parameter(name) = count {
        let value = params.get(name.as_str()).ok_or_else(|| {
            Error::from(QueryError::invalid_reference(format!(
                "Missing parameter for SKIP/LIMIT: ${name}"
            )))
        })?;
        *count = CountExpr::Literal(CountExpr::count_of(value)?);
    }
    Ok(())
}

/// Substitutes parameters in an expression with their values.
fn substitute_in_expression(expr: &mut LogicalExpression, params: &QueryParams) -> Result<()> {
    use crate::query::plan::LogicalExpression;

    match expr {
        LogicalExpression::Parameter(name) => {
            if let Some(value) = params.get(name) {
                *expr = LogicalExpression::Literal(value.clone());
            } else {
                return Err(Error::from(grafeo_common::utils::error::QueryError::new(
                    grafeo_common::utils::error::QueryErrorKind::InvalidReference,
                    format!("Missing parameter: ${name}"),
                )));
            }
        }
        LogicalExpression::Binary { left, right, .. } => {
            substitute_in_expression(left, params)?;
            substitute_in_expression(right, params)?;
        }
        LogicalExpression::Unary { operand, .. } => {
            substitute_in_expression(operand, params)?;
        }
        LogicalExpression::FunctionCall { args, .. }
        | LogicalExpression::SparqlCall { args, .. } => {
            for arg in args {
                substitute_in_expression(arg, params)?;
            }
        }
        LogicalExpression::List(items) => {
            for item in items {
                substitute_in_expression(item, params)?;
            }
        }
        LogicalExpression::Map(pairs) => {
            for (_, value) in pairs {
                substitute_in_expression(value, params)?;
            }
        }
        LogicalExpression::IndexAccess { base, index } => {
            substitute_in_expression(base, params)?;
            substitute_in_expression(index, params)?;
        }
        LogicalExpression::MapAccess { base, .. } => substitute_in_expression(base, params)?,
        LogicalExpression::SliceAccess { base, start, end } => {
            substitute_in_expression(base, params)?;
            if let Some(s) = start {
                substitute_in_expression(s, params)?;
            }
            if let Some(e) = end {
                substitute_in_expression(e, params)?;
            }
        }
        LogicalExpression::Case {
            operand,
            when_clauses,
            else_clause,
        } => {
            if let Some(op) = operand {
                substitute_in_expression(op, params)?;
            }
            for (cond, result) in when_clauses {
                substitute_in_expression(cond, params)?;
                substitute_in_expression(result, params)?;
            }
            if let Some(el) = else_clause {
                substitute_in_expression(el, params)?;
            }
        }
        LogicalExpression::Property { .. }
        | LogicalExpression::Variable(_)
        | LogicalExpression::Literal(_)
        | LogicalExpression::Labels(_)
        | LogicalExpression::Type(_)
        | LogicalExpression::Id(_) => {}
        LogicalExpression::ListComprehension {
            list_expr,
            filter_expr,
            map_expr,
            ..
        } => {
            substitute_in_expression(list_expr, params)?;
            if let Some(filter) = filter_expr {
                substitute_in_expression(filter, params)?;
            }
            substitute_in_expression(map_expr, params)?;
        }
        LogicalExpression::ListPredicate {
            list_expr,
            predicate,
            ..
        } => {
            substitute_in_expression(list_expr, params)?;
            substitute_in_expression(predicate, params)?;
        }
        LogicalExpression::ExistsSubquery(subplan)
        | LogicalExpression::CountSubquery(subplan)
        | LogicalExpression::ValueSubquery(subplan) => {
            substitute_in_operator(subplan, params)?;
        }
        LogicalExpression::PatternComprehension { projection, .. } => {
            substitute_in_expression(projection, params)?;
        }
        LogicalExpression::MapProjection { entries, .. } => {
            for entry in entries {
                if let crate::query::plan::MapProjectionEntry::LiteralEntry(_, expr) = entry {
                    substitute_in_expression(expr, params)?;
                }
            }
        }
        LogicalExpression::Reduce {
            initial,
            list,
            expression,
            ..
        } => {
            substitute_in_expression(initial, params)?;
            substitute_in_expression(list, params)?;
            substitute_in_expression(expression, params)?;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use grafeo_common::types::ReadView;
    use grafeo_common::types::{NodeId, PropertyKey};
    use grafeo_core::graph::GraphStore;

    /// A node with `labels` and the values `properties`.
    fn with_props<const N: usize>(
        store: &RowGroupStore,
        labels: &[&str],
        properties: [(&str, Value); N],
    ) -> NodeId {
        let properties: Vec<(PropertyKey, Value)> = properties
            .into_iter()
            .map(|(key, value)| (PropertyKey::new(key), value))
            .collect();
        store.create_node_with_props(labels, &properties)
    }

    #[test]
    fn test_query_language_is_lpg() {
        #[cfg(feature = "gql")]
        assert!(QueryLanguage::Gql.is_lpg());
        #[cfg(feature = "cypher")]
        assert!(QueryLanguage::Cypher.is_lpg());
        #[cfg(feature = "sparql")]
        assert!(!QueryLanguage::Sparql.is_lpg());
    }

    #[test]
    fn test_processor_creation() {
        let store = Arc::new(RowGroupStore::new());
        let processor = QueryProcessor::for_lpg(store);
        assert_eq!(
            processor
                .lpg_store()
                .expect("a processor on the built-in store")
                .node_count(ReadView::latest())
                .unwrap(),
            0
        );
    }

    #[cfg(feature = "gql")]
    #[test]
    fn test_process_simple_gql() {
        let store = Arc::new(RowGroupStore::new());
        store.create_node(&["Person"]);
        store.create_node(&["Person"]);

        let processor = QueryProcessor::for_lpg(store);
        let result = processor
            .process("MATCH (n:Person) RETURN n", QueryLanguage::Gql, None)
            .unwrap();

        assert_eq!(result.row_count(), 2);
        assert_eq!(result.columns[0], "n");
    }

    #[cfg(feature = "cypher")]
    #[test]
    fn test_process_simple_cypher() {
        let store = Arc::new(RowGroupStore::new());
        store.create_node(&["Person"]);

        let processor = QueryProcessor::for_lpg(store);
        let result = processor
            .process("MATCH (n:Person) RETURN n", QueryLanguage::Cypher, None)
            .unwrap();

        assert_eq!(result.row_count(), 1);
    }

    #[cfg(feature = "gql")]
    #[test]
    fn test_process_with_params() {
        let store = Arc::new(RowGroupStore::new());
        with_props(&store, &["Person"], [("age", Value::Int64(25))]);
        with_props(&store, &["Person"], [("age", Value::Int64(35))]);
        with_props(&store, &["Person"], [("age", Value::Int64(45))]);

        let processor = QueryProcessor::for_lpg(store);

        // Query with parameter
        let mut params = HashMap::new();
        params.insert("min_age".to_string(), Value::Int64(30));

        let result = processor
            .process(
                "MATCH (n:Person) WHERE n.age > $min_age RETURN n",
                QueryLanguage::Gql,
                Some(&params),
            )
            .unwrap();

        // Should return 2 people (ages 35 and 45)
        assert_eq!(result.row_count(), 2);
    }

    #[cfg(feature = "gql")]
    #[test]
    fn test_missing_param_error() {
        let store = Arc::new(RowGroupStore::new());
        store.create_node(&["Person"]);

        let processor = QueryProcessor::for_lpg(store);

        // Query with parameter but empty params map (missing the required param)
        let params: HashMap<String, Value> = HashMap::new();
        let result = processor.process(
            "MATCH (n:Person) WHERE n.age > $min_age RETURN n",
            QueryLanguage::Gql,
            Some(&params),
        );

        // Should fail with missing parameter error
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(
            err.to_string().contains("Missing parameter"),
            "Expected 'Missing parameter' error, got: {}",
            err
        );
    }

    #[cfg(feature = "gql")]
    #[test]
    fn test_params_in_filter_with_property() {
        // Tests parameter substitution in WHERE clause with property comparison
        let store = Arc::new(RowGroupStore::new());
        with_props(&store, &["Num"], [("value", Value::Int64(10))]);
        with_props(&store, &["Num"], [("value", Value::Int64(20))]);

        let processor = QueryProcessor::for_lpg(store);

        let mut params = HashMap::new();
        params.insert("threshold".to_string(), Value::Int64(15));

        let result = processor
            .process(
                "MATCH (n:Num) WHERE n.value > $threshold RETURN n.value",
                QueryLanguage::Gql,
                Some(&params),
            )
            .unwrap();

        // Only value=20 matches > 15
        assert_eq!(result.row_count(), 1);
        let row = &result.rows[0];
        assert_eq!(row[0], Value::Int64(20));
    }

    #[cfg(feature = "gql")]
    #[test]
    fn test_params_in_multiple_where_conditions() {
        // Tests multiple parameters in WHERE clause with AND
        let store = Arc::new(RowGroupStore::new());
        with_props(
            &store,
            &["Person"],
            [("age", Value::Int64(25)), ("score", Value::Int64(80))],
        );
        with_props(
            &store,
            &["Person"],
            [("age", Value::Int64(35)), ("score", Value::Int64(90))],
        );
        with_props(
            &store,
            &["Person"],
            [("age", Value::Int64(45)), ("score", Value::Int64(70))],
        );

        let processor = QueryProcessor::for_lpg(store);

        let mut params = HashMap::new();
        params.insert("min_age".to_string(), Value::Int64(30));
        params.insert("min_score".to_string(), Value::Int64(75));

        let result = processor
            .process(
                "MATCH (n:Person) WHERE n.age > $min_age AND n.score > $min_score RETURN n",
                QueryLanguage::Gql,
                Some(&params),
            )
            .unwrap();

        // Only the person with age=35, score=90 matches both conditions
        assert_eq!(result.row_count(), 1);
    }

    #[cfg(feature = "gql")]
    #[test]
    fn test_params_with_in_list() {
        // Tests parameter as a value checked against IN list
        let store = Arc::new(RowGroupStore::new());
        with_props(
            &store,
            &["Item"],
            [("status", Value::String("active".into()))],
        );
        with_props(
            &store,
            &["Item"],
            [("status", Value::String("pending".into()))],
        );
        with_props(
            &store,
            &["Item"],
            [("status", Value::String("deleted".into()))],
        );

        let processor = QueryProcessor::for_lpg(store);

        // Check if a parameter value matches any of the statuses
        let mut params = HashMap::new();
        params.insert("target".to_string(), Value::String("active".into()));

        let result = processor
            .process(
                "MATCH (n:Item) WHERE n.status = $target RETURN n",
                QueryLanguage::Gql,
                Some(&params),
            )
            .unwrap();

        assert_eq!(result.row_count(), 1);
    }

    #[cfg(feature = "gql")]
    #[test]
    fn test_params_same_type_comparison() {
        // Tests that same-type parameter comparisons work correctly
        let store = Arc::new(RowGroupStore::new());
        with_props(&store, &["Data"], [("value", Value::Int64(100))]);
        with_props(&store, &["Data"], [("value", Value::Int64(50))]);

        let processor = QueryProcessor::for_lpg(store);

        // Compare int property with int parameter
        let mut params = HashMap::new();
        params.insert("threshold".to_string(), Value::Int64(75));

        let result = processor
            .process(
                "MATCH (n:Data) WHERE n.value > $threshold RETURN n",
                QueryLanguage::Gql,
                Some(&params),
            )
            .unwrap();

        // Only value=100 matches > 75
        assert_eq!(result.row_count(), 1);
    }

    #[cfg(feature = "gql")]
    #[test]
    fn test_process_empty_result_has_columns() {
        // Tests that empty results still have correct column names
        let store = Arc::new(RowGroupStore::new());
        // Don't create any nodes

        let processor = QueryProcessor::for_lpg(store);
        let result = processor
            .process(
                "MATCH (n:Person) RETURN n.name AS name, n.age AS age",
                QueryLanguage::Gql,
                None,
            )
            .unwrap();

        assert_eq!(result.row_count(), 0);
        assert_eq!(result.columns.len(), 2);
        assert_eq!(result.columns[0], "name");
        assert_eq!(result.columns[1], "age");
    }

    #[cfg(feature = "gql")]
    #[test]
    fn test_params_string_equality() {
        // Tests string parameter equality comparison
        let store = Arc::new(RowGroupStore::new());
        with_props(&store, &["Item"], [("name", Value::String("alpha".into()))]);
        with_props(&store, &["Item"], [("name", Value::String("beta".into()))]);
        with_props(&store, &["Item"], [("name", Value::String("gamma".into()))]);

        let processor = QueryProcessor::for_lpg(store);

        let mut params = HashMap::new();
        params.insert("target".to_string(), Value::String("beta".into()));

        let result = processor
            .process(
                "MATCH (n:Item) WHERE n.name = $target RETURN n.name",
                QueryLanguage::Gql,
                Some(&params),
            )
            .unwrap();

        assert_eq!(result.row_count(), 1);
        assert_eq!(result.rows[0][0], Value::String("beta".into()));
    }

    #[cfg(feature = "cypher")]
    #[test]
    fn test_params_in_exists_subquery() {
        // Regression: parameters inside EXISTS/COUNT/VALUE subqueries were not
        // substituted, causing type mismatches or silently wrong results.
        let store = Arc::new(RowGroupStore::new());
        let alix = with_props(
            &store,
            &["Person"],
            [("name", Value::String("Alix".into()))],
        );
        let gus = with_props(&store, &["Person"], [("name", Value::String("Gus".into()))]);
        let _jules = with_props(
            &store,
            &["Person"],
            [("name", Value::String("Jules".into()))],
        );

        // Alix follows Gus (but not Jules)
        store.create_edge(alix, gus, "FOLLOWS");

        let processor = QueryProcessor::for_lpg(store);

        // Find people NOT followed by the viewer ($viewer)
        let mut params = HashMap::new();
        params.insert("viewer".to_string(), Value::String("Alix".into()));

        let result = processor
            .process(
                "MATCH (p:Person) \
                 WHERE p.name <> $viewer \
                   AND NOT EXISTS { MATCH (v:Person)-[:FOLLOWS]->(p) WHERE v.name = $viewer } \
                 RETURN p.name ORDER BY p.name",
                QueryLanguage::Cypher,
                Some(&params),
            )
            .unwrap();

        // Alix follows Gus, so only Jules should be returned
        assert_eq!(result.row_count(), 1);
        assert_eq!(result.rows[0][0], Value::String("Jules".into()));
    }
}
