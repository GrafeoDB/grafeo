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
#[cfg(feature = "lpg")]
use grafeo_core::graph::lpg::LpgStore;
use grafeo_core::graph::{GraphStore, GraphStoreMut};

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
/// # use grafeo_core::graph::lpg::LpgStore;
/// use grafeo_engine::query::processor::{QueryProcessor, QueryLanguage};
///
/// # fn main() -> grafeo_common::utils::error::Result<()> {
/// let store = Arc::new(LpgStore::new().unwrap());
/// let processor = QueryProcessor::for_lpg(store);
/// let result = processor.process("MATCH (n:Person) RETURN n", QueryLanguage::Gql, None)?;
/// # Ok(())
/// # }
/// ```
pub struct QueryProcessor {
    /// LPG store for property graph queries.
    #[cfg(feature = "lpg")]
    lpg_store: Arc<LpgStore>,
    /// Graph store trait object for pluggable storage backends (read path).
    graph_store: Arc<dyn GraphStore>,
    /// Writable graph store (None when read-only).
    write_store: Option<Arc<dyn GraphStoreMut>>,
    /// Transaction manager for MVCC operations.
    transaction_manager: Arc<TransactionManager>,
    /// Catalog for schema and index metadata.
    catalog: Arc<Catalog>,
    /// Query optimizer.
    optimizer: Optimizer,
    /// Current transaction context (if any).
    transaction_context: Option<(EpochId, TransactionId)>,
    /// RDF store for triple pattern queries (optional).
    #[cfg(feature = "triple-store")]
    rdf_store: Option<Arc<grafeo_core::graph::rdf::RdfStore>>,
}

impl QueryProcessor {
    /// Creates a new query processor for LPG queries.
    #[cfg(feature = "lpg")]
    #[must_use]
    pub fn for_lpg(store: Arc<LpgStore>) -> Self {
        let optimizer = Optimizer::from_store(&store);
        let graph_store = Arc::clone(&store) as Arc<dyn GraphStore>;
        let write_store = Some(Arc::clone(&store) as Arc<dyn GraphStoreMut>);
        Self {
            lpg_store: store,
            graph_store,
            write_store,
            transaction_manager: Arc::new(TransactionManager::new()),
            catalog: Arc::new(Catalog::new()),
            optimizer,
            transaction_context: None,
            #[cfg(feature = "triple-store")]
            rdf_store: None,
        }
    }

    /// Creates a new query processor with a transaction manager.
    #[cfg(feature = "lpg")]
    #[must_use]
    pub fn for_lpg_with_transaction(
        store: Arc<LpgStore>,
        transaction_manager: Arc<TransactionManager>,
    ) -> Self {
        let optimizer = Optimizer::from_store(&store);
        let graph_store = Arc::clone(&store) as Arc<dyn GraphStore>;
        let write_store = Some(Arc::clone(&store) as Arc<dyn GraphStoreMut>);
        Self {
            lpg_store: store,
            graph_store,
            write_store,
            transaction_manager,
            catalog: Arc::new(Catalog::new()),
            optimizer,
            transaction_context: None,
            #[cfg(feature = "triple-store")]
            rdf_store: None,
        }
    }

    /// Creates a query processor backed by any `GraphStoreMut` implementation.
    ///
    /// # Errors
    ///
    /// Returns an error if the internal arena allocation fails (out of memory).
    pub fn for_graph_store_with_transaction(
        store: Arc<dyn GraphStoreMut>,
        transaction_manager: Arc<TransactionManager>,
    ) -> Result<Self> {
        let optimizer = Optimizer::from_graph_store(&*store);
        let read_store = Arc::clone(&store) as Arc<dyn GraphStore>;
        Ok(Self {
            #[cfg(feature = "lpg")]
            lpg_store: Arc::new(LpgStore::new()?),
            graph_store: read_store,
            write_store: Some(store),
            transaction_manager,
            catalog: Arc::new(Catalog::new()),
            optimizer,
            transaction_context: None,
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
        write_store: Option<Arc<dyn GraphStoreMut>>,
        transaction_manager: Arc<TransactionManager>,
    ) -> Result<Self> {
        let optimizer = Optimizer::from_graph_store(&*read_store);
        Ok(Self {
            #[cfg(feature = "lpg")]
            lpg_store: Arc::new(LpgStore::new()?),
            graph_store: read_store,
            write_store,
            transaction_manager,
            catalog: Arc::new(Catalog::new()),
            optimizer,
            transaction_context: None,
            #[cfg(feature = "triple-store")]
            rdf_store: None,
        })
    }

    /// Sets the transaction context for MVCC visibility.
    ///
    /// This should be called when the processor is used within a transaction.
    #[must_use]
    pub fn with_transaction_context(
        mut self,
        viewing_epoch: EpochId,
        transaction_id: TransactionId,
    ) -> Self {
        self.transaction_context = Some((viewing_epoch, transaction_id));
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
                Err(Error::Internal(
                    "RDF support not enabled. Compile with --features rdf".to_string(),
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
        #[cfg(not(target_arch = "wasm32"))]
        let start_time = std::time::Instant::now();

        // 1. Parse and translate to logical plan
        let mut logical_plan = self.translate_lpg(query, language)?;

        // 2. Substitute parameters if provided (merge defaults from the plan first)
        let has_defaults = !logical_plan.default_params.is_empty();
        if params.is_some() || has_defaults {
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

        // 4a. EXPLAIN: annotate pushdown hints and return the plan tree
        if optimized_plan.explain {
            let mut plan = optimized_plan;
            annotate_pushdown_hints(&mut plan.root, self.graph_store.as_ref());
            return Ok(explain_result(&plan));
        }

        // 5. Convert to physical plan with transaction context
        // Read-only fast path: safe when no mutations AND no active transaction
        // (an active transaction may have prior uncommitted writes from earlier statements)
        let is_read_only =
            !optimized_plan.root.has_mutations() && self.transaction_context.is_none();
        let planner = if let Some((epoch, transaction_id)) = self.transaction_context {
            Planner::with_context(
                Arc::clone(&self.graph_store),
                self.write_store.as_ref().map(Arc::clone),
                Arc::clone(&self.transaction_manager),
                Some(transaction_id),
                epoch,
            )
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
        let mut physical_plan = planner.plan(&optimized_plan)?;

        // 6. Execute and collect results
        let executor = Executor::with_columns(physical_plan.columns.clone());
        let mut result = executor.execute(physical_plan.operator.as_mut())?;

        // Add execution metrics
        let rows_scanned = result.rows.len() as u64; // Approximate: rows returned
        #[cfg(not(target_arch = "wasm32"))]
        {
            let elapsed_ms = start_time.elapsed().as_secs_f64() * 1000.0;
            result.execution_time_ms = Some(elapsed_ms);
        }
        result.rows_scanned = Some(rows_scanned);

        Ok(result)
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
                "Language {:?} is not an LPG language",
                language
            ))),
        }
    }

    /// Returns a reference to the LPG store.
    #[cfg(feature = "lpg")]
    #[must_use]
    pub fn lpg_store(&self) -> &Arc<LpgStore> {
        &self.lpg_store
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
    #[cfg(feature = "lpg")]
    #[must_use]
    pub fn with_rdf(
        lpg_store: Arc<LpgStore>,
        rdf_store: Arc<grafeo_core::graph::rdf::RdfStore>,
    ) -> Self {
        let optimizer = Optimizer::from_store(&lpg_store);
        let graph_store = Arc::clone(&lpg_store) as Arc<dyn GraphStore>;
        let write_store = Some(Arc::clone(&lpg_store) as Arc<dyn GraphStoreMut>);
        Self {
            lpg_store,
            graph_store,
            write_store,
            transaction_manager: Arc::new(TransactionManager::new()),
            catalog: Arc::new(Catalog::new()),
            optimizer,
            transaction_context: None,
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

        // 1. Parse and translate to logical plan
        let mut logical_plan = self.translate_rdf(query, language)?;

        // 2. Substitute parameters if provided (merge defaults from the plan first)
        let has_defaults = !logical_plan.default_params.is_empty();
        if params.is_some() || has_defaults {
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

        // 3b. EXPLAIN ANALYZE (PROFILE): execute with instrumentation, report stats.
        if optimized_plan.profile {
            let planner = RdfPlanner::new(Arc::clone(rdf_store));
            let (mut physical_plan, entries) = planner.plan_profiled(&optimized_plan)?;

            let start = std::time::Instant::now();
            let executor = Executor::with_columns(physical_plan.columns.clone());
            let _result = executor.execute(physical_plan.operator.as_mut())?;
            let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;

            let tree = crate::query::profile::build_profile_tree(
                &optimized_plan.root,
                &mut entries.into_iter(),
            );
            return Ok(crate::query::profile::profile_result(&tree, elapsed_ms));
        }

        // 4. Convert to physical plan (using RDF planner)
        let planner = RdfPlanner::new(Arc::clone(rdf_store));
        let mut physical_plan = planner.plan(&optimized_plan)?;

        // 5. Execute and collect results
        let executor = Executor::with_columns(physical_plan.columns.clone());
        executor.execute(physical_plan.operator.as_mut())
    }

    /// Translates an RDF query to a logical plan.
    fn translate_rdf(&self, query: &str, language: QueryLanguage) -> Result<LogicalPlan> {
        match language {
            #[cfg(feature = "sparql")]
            QueryLanguage::Sparql => {
                use crate::query::translators::sparql;
                sparql::translate(query)
            }
            #[cfg(all(feature = "graphql", feature = "triple-store"))]
            QueryLanguage::GraphQLRdf => {
                use crate::query::translators::graphql_rdf;
                // Default namespace for GraphQL-RDF queries
                graphql_rdf::translate(query, "http://example.org/")
            }
            _ => Err(Error::Internal(format!(
                "Language {:?} is not an RDF language",
                language
            ))),
        }
    }
}

/// Annotates filter operators in the plan with pushdown hints.
///
/// Walks the plan tree looking for `Filter -> NodeScan` patterns and checks
/// whether a property index exists for equality predicates.
pub(crate) fn annotate_pushdown_hints(
    op: &mut LogicalOperator,
    store: &dyn grafeo_core::graph::GraphStore,
) {
    #[allow(clippy::wildcard_imports)]
    use crate::query::plan::*;

    match op {
        LogicalOperator::Filter(filter) => {
            // Recurse into children first
            annotate_pushdown_hints(&mut filter.input, store);

            // Annotate this filter if it sits on top of a NodeScan
            if let LogicalOperator::NodeScan(scan) = filter.input.as_ref() {
                filter.pushdown_hint = infer_pushdown(&filter.predicate, scan, store);
            }
        }
        LogicalOperator::NodeScan(op) => {
            if let Some(input) = &mut op.input {
                annotate_pushdown_hints(input, store);
            }
        }
        LogicalOperator::EdgeScan(op) => {
            if let Some(input) = &mut op.input {
                annotate_pushdown_hints(input, store);
            }
        }
        LogicalOperator::Expand(op) => annotate_pushdown_hints(&mut op.input, store),
        LogicalOperator::Project(op) => annotate_pushdown_hints(&mut op.input, store),
        LogicalOperator::Join(op) => {
            annotate_pushdown_hints(&mut op.left, store);
            annotate_pushdown_hints(&mut op.right, store);
        }
        LogicalOperator::Aggregate(op) => annotate_pushdown_hints(&mut op.input, store),
        LogicalOperator::Limit(op) => annotate_pushdown_hints(&mut op.input, store),
        LogicalOperator::Skip(op) => annotate_pushdown_hints(&mut op.input, store),
        LogicalOperator::Sort(op) => annotate_pushdown_hints(&mut op.input, store),
        LogicalOperator::Distinct(op) => annotate_pushdown_hints(&mut op.input, store),
        LogicalOperator::Return(op) => annotate_pushdown_hints(&mut op.input, store),
        LogicalOperator::Union(op) => {
            for input in &mut op.inputs {
                annotate_pushdown_hints(input, store);
            }
        }
        LogicalOperator::Apply(op) => {
            annotate_pushdown_hints(&mut op.input, store);
            annotate_pushdown_hints(&mut op.subplan, store);
        }
        LogicalOperator::Otherwise(op) => {
            annotate_pushdown_hints(&mut op.left, store);
            annotate_pushdown_hints(&mut op.right, store);
        }
        _ => {}
    }
}

/// Infers the pushdown strategy for a filter predicate over a node scan.
fn infer_pushdown(
    predicate: &LogicalExpression,
    scan: &crate::query::plan::NodeScanOp,
    store: &dyn grafeo_core::graph::GraphStore,
) -> Option<crate::query::plan::PushdownHint> {
    #[allow(clippy::wildcard_imports)]
    use crate::query::plan::*;

    match predicate {
        // Equality: n.prop = value
        LogicalExpression::Binary { left, op, right } if *op == BinaryOp::Eq => {
            if let Some(prop) = extract_property_name(left, &scan.variable)
                .or_else(|| extract_property_name(right, &scan.variable))
            {
                if store.has_property_index(&prop) {
                    return Some(PushdownHint::IndexLookup { property: prop });
                }
                if scan.label.is_some() {
                    return Some(PushdownHint::LabelFirst);
                }
            }
            None
        }
        // Range: n.prop > value, n.prop < value, etc.
        LogicalExpression::Binary {
            left,
            op: BinaryOp::Gt | BinaryOp::Ge | BinaryOp::Lt | BinaryOp::Le,
            right,
        } => {
            if let Some(prop) = extract_property_name(left, &scan.variable)
                .or_else(|| extract_property_name(right, &scan.variable))
            {
                if store.has_property_index(&prop) {
                    return Some(PushdownHint::RangeScan { property: prop });
                }
                if scan.label.is_some() {
                    return Some(PushdownHint::LabelFirst);
                }
            }
            None
        }
        // AND: check the left side (first conjunct) for pushdown
        LogicalExpression::Binary {
            left,
            op: BinaryOp::And,
            ..
        } => infer_pushdown(left, scan, store),
        _ => {
            // Any other predicate on a labeled scan gets label-first
            if scan.label.is_some() {
                Some(PushdownHint::LabelFirst)
            } else {
                None
            }
        }
    }
}

/// Extracts the property name if the expression is `Property { variable, property }`
/// and the variable matches the scan variable.
fn extract_property_name(expr: &LogicalExpression, scan_var: &str) -> Option<String> {
    if let LogicalExpression::Property { variable, property } = expr
        && variable == scan_var
    {
        Some(property.clone())
    } else {
        None
    }
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
    }
}

/// Formats a physical EXPLAIN result showing both the logical plan and the
/// physical operator names mapped to each logical operator.
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
    }
}

/// Recursively formats a physical plan node showing operator name and label.
fn format_physical_node(out: &mut String, node: &crate::query::profile::ProfileNode, depth: usize) {
    use std::fmt::Write;
    let indent = "  ".repeat(depth);
    let _ = writeln!(out, "{indent}{} {}", node.name, node.label);
    for child in &node.children {
        format_physical_node(out, child, depth + 1);
    }
}

/// Substitutes parameters in a logical plan with their values.
///
/// # Errors
///
/// Returns an error if a referenced parameter is not found in `params`.
pub fn substitute_params(plan: &mut LogicalPlan, params: &QueryParams) -> Result<()> {
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
                substitute_in_expression(&mut item.expression, params)?;
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
                if let Some(expr) = &mut agg_expr.expression {
                    substitute_in_expression(expr, params)?;
                }
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
        LogicalOperator::Bind(bind) => {
            substitute_in_expression(&mut bind.expression, params)?;
            substitute_in_operator(&mut bind.input, params)?;
        }
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
        // ParameterScan has no expressions to substitute
        LogicalOperator::ParameterScan(_) => {}
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
        LogicalOperator::CreatePropertyGraph(_) => {}
        // Procedure calls: arguments could contain parameters but we handle at execution time
        LogicalOperator::CallProcedure(_) => {}
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
    use grafeo_common::utils::error::{QueryError, QueryErrorKind};

    if let CountExpr::Parameter(name) = count {
        let value = params.get(name.as_str()).ok_or_else(|| {
            Error::Query(QueryError::new(
                QueryErrorKind::Semantic,
                format!("Missing parameter for SKIP/LIMIT: ${name}"),
            ))
        })?;
        let n = match value {
            Value::Int64(i) if *i >= 0 => *i as usize,
            Value::Int64(i) => {
                return Err(Error::Query(QueryError::new(
                    QueryErrorKind::Semantic,
                    format!("SKIP/LIMIT parameter ${name} must be non-negative, got {i}"),
                )));
            }
            other => {
                return Err(Error::Query(QueryError::new(
                    QueryErrorKind::Semantic,
                    format!("SKIP/LIMIT parameter ${name} must be an integer, got {other:?}"),
                )));
            }
        };
        *count = CountExpr::Literal(n);
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
                return Err(Error::Internal(format!("Missing parameter: ${}", name)));
            }
        }
        LogicalExpression::Binary { left, right, .. } => {
            substitute_in_expression(left, params)?;
            substitute_in_expression(right, params)?;
        }
        LogicalExpression::Unary { operand, .. } => {
            substitute_in_expression(operand, params)?;
        }
        LogicalExpression::FunctionCall { args, .. } => {
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
        let store = Arc::new(LpgStore::new().unwrap());
        let processor = QueryProcessor::for_lpg(store);
        assert!(processor.lpg_store().node_count() == 0);
    }

    #[cfg(feature = "gql")]
    #[test]
    fn test_process_simple_gql() {
        let store = Arc::new(LpgStore::new().unwrap());
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
        let store = Arc::new(LpgStore::new().unwrap());
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
        let store = Arc::new(LpgStore::new().unwrap());
        store.create_node_with_props(&["Person"], [("age", Value::Int64(25))]);
        store.create_node_with_props(&["Person"], [("age", Value::Int64(35))]);
        store.create_node_with_props(&["Person"], [("age", Value::Int64(45))]);

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
        let store = Arc::new(LpgStore::new().unwrap());
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
        let store = Arc::new(LpgStore::new().unwrap());
        store.create_node_with_props(&["Num"], [("value", Value::Int64(10))]);
        store.create_node_with_props(&["Num"], [("value", Value::Int64(20))]);

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
        let store = Arc::new(LpgStore::new().unwrap());
        store.create_node_with_props(
            &["Person"],
            [("age", Value::Int64(25)), ("score", Value::Int64(80))],
        );
        store.create_node_with_props(
            &["Person"],
            [("age", Value::Int64(35)), ("score", Value::Int64(90))],
        );
        store.create_node_with_props(
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
        let store = Arc::new(LpgStore::new().unwrap());
        store.create_node_with_props(&["Item"], [("status", Value::String("active".into()))]);
        store.create_node_with_props(&["Item"], [("status", Value::String("pending".into()))]);
        store.create_node_with_props(&["Item"], [("status", Value::String("deleted".into()))]);

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
        let store = Arc::new(LpgStore::new().unwrap());
        store.create_node_with_props(&["Data"], [("value", Value::Int64(100))]);
        store.create_node_with_props(&["Data"], [("value", Value::Int64(50))]);

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
        let store = Arc::new(LpgStore::new().unwrap());
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
        let store = Arc::new(LpgStore::new().unwrap());
        store.create_node_with_props(&["Item"], [("name", Value::String("alpha".into()))]);
        store.create_node_with_props(&["Item"], [("name", Value::String("beta".into()))]);
        store.create_node_with_props(&["Item"], [("name", Value::String("gamma".into()))]);

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
        // Tests parameter substitution inside NOT EXISTS { ... } subquery.
        // This is a common Cypher pattern for anti-joins, e.g. "users not followed by viewer".
        let store = Arc::new(LpgStore::new().unwrap());
        let alice = store.create_node_with_props(&["Person"], [("name", Value::String("alice".into()))]);
        let bob = store.create_node_with_props(&["Person"], [("name", Value::String("bob".into()))]);
        let _charlie = store.create_node_with_props(&["Person"], [("name", Value::String("charlie".into()))]);

        // alice follows bob (but not charlie)
        store.create_edge(alice, bob, "FOLLOWS");

        let processor = QueryProcessor::for_lpg(store);

        // Find people NOT followed by the viewer ($viewer_name)
        let mut params = HashMap::new();
        params.insert("viewer_name".to_string(), Value::String("alice".into()));

        let result = processor
            .process(
                "MATCH (p:Person) WHERE p.name <> $viewer_name AND NOT EXISTS { MATCH (v:Person)-[:FOLLOWS]->(p) WHERE v.name = $viewer_name } RETURN p.name ORDER BY p.name",
                QueryLanguage::Cypher,
                Some(&params),
            )
            .unwrap();

        // alice follows bob, so only charlie should be returned
        assert_eq!(result.row_count(), 1);
        assert_eq!(result.rows[0][0], Value::String("charlie".into()));
    }
}
