//! Query processor that orchestrates the query pipeline.
//!
//! The `QueryProcessor` is the central component that executes queries through
//! the full pipeline: Parse → Bind → Optimize → Plan → Execute.
//!
//! It supports multiple query languages (GQL, Cypher, Gremlin, GraphQL) for LPG
//! and SPARQL for RDF (when the `rdf` feature is enabled).

use std::collections::HashMap;
use std::sync::Arc;

use grafeo_common::types::{EpochId, TxId, Value};
use grafeo_common::utils::error::{Error, Result};
use grafeo_core::graph::lpg::LpgStore;

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
    /// SPARQL 1.1 for RDF
    #[cfg(feature = "sparql")]
    Sparql,
    /// GraphQL for RDF
    #[cfg(all(feature = "graphql", feature = "rdf"))]
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
            #[cfg(feature = "sparql")]
            Self::Sparql => false,
            #[cfg(all(feature = "graphql", feature = "rdf"))]
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
/// ```ignore
/// use grafeo_engine::query::processor::{QueryProcessor, QueryLanguage};
///
/// let processor = QueryProcessor::for_lpg(store);
/// let result = processor.process("MATCH (n:Person) RETURN n", QueryLanguage::Gql, None)?;
/// ```
pub struct QueryProcessor {
    /// LPG store for property graph queries.
    lpg_store: Arc<LpgStore>,
    /// Transaction manager for MVCC operations.
    tx_manager: Arc<TransactionManager>,
    /// Catalog for schema and index metadata.
    catalog: Arc<Catalog>,
    /// Query optimizer.
    optimizer: Optimizer,
    /// Current transaction context (if any).
    tx_context: Option<(EpochId, TxId)>,
    /// RDF store for triple pattern queries (optional).
    #[cfg(feature = "rdf")]
    rdf_store: Option<Arc<grafeo_core::graph::rdf::RdfStore>>,
}

impl QueryProcessor {
    /// Creates a new query processor for LPG queries.
    #[must_use]
    pub fn for_lpg(store: Arc<LpgStore>) -> Self {
        Self {
            lpg_store: store,
            tx_manager: Arc::new(TransactionManager::new()),
            catalog: Arc::new(Catalog::new()),
            optimizer: Optimizer::new(),
            tx_context: None,
            #[cfg(feature = "rdf")]
            rdf_store: None,
        }
    }

    /// Creates a new query processor with a transaction manager.
    #[must_use]
    pub fn for_lpg_with_tx(store: Arc<LpgStore>, tx_manager: Arc<TransactionManager>) -> Self {
        Self {
            lpg_store: store,
            tx_manager,
            catalog: Arc::new(Catalog::new()),
            optimizer: Optimizer::new(),
            tx_context: None,
            #[cfg(feature = "rdf")]
            rdf_store: None,
        }
    }

    /// Creates a new query processor with both LPG and RDF stores.
    #[cfg(feature = "rdf")]
    #[must_use]
    pub fn with_rdf(
        lpg_store: Arc<LpgStore>,
        rdf_store: Arc<grafeo_core::graph::rdf::RdfStore>,
    ) -> Self {
        Self {
            lpg_store,
            tx_manager: Arc::new(TransactionManager::new()),
            catalog: Arc::new(Catalog::new()),
            optimizer: Optimizer::new(),
            tx_context: None,
            rdf_store: Some(rdf_store),
        }
    }

    /// Sets the transaction context for MVCC visibility.
    ///
    /// This should be called when the processor is used within a transaction.
    #[must_use]
    pub fn with_tx_context(mut self, viewing_epoch: EpochId, tx_id: TxId) -> Self {
        self.tx_context = Some((viewing_epoch, tx_id));
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
            #[cfg(feature = "rdf")]
            {
                self.process_rdf(query, language, params)
            }
            #[cfg(not(feature = "rdf"))]
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
        // 1. Parse and translate to logical plan
        let mut logical_plan = self.translate_lpg(query, language)?;

        // 2. Substitute parameters if provided
        if let Some(params) = params {
            substitute_params(&mut logical_plan, params)?;
        }

        // 3. Semantic validation
        let mut binder = Binder::new();
        let _binding_context = binder.bind(&logical_plan)?;

        // 4. Optimize the plan
        let optimized_plan = self.optimizer.optimize(logical_plan)?;

        // 5. Convert to physical plan with transaction context
        let planner = if let Some((epoch, tx_id)) = self.tx_context {
            Planner::with_context(
                Arc::clone(&self.lpg_store),
                Arc::clone(&self.tx_manager),
                Some(tx_id),
                epoch,
            )
        } else {
            Planner::with_context(
                Arc::clone(&self.lpg_store),
                Arc::clone(&self.tx_manager),
                None,
                self.tx_manager.current_epoch(),
            )
        };
        let mut physical_plan = planner.plan(&optimized_plan)?;

        // 6. Execute and collect results
        let executor = Executor::with_columns(physical_plan.columns.clone());
        executor.execute(physical_plan.operator.as_mut())
    }

    /// Translates an LPG query to a logical plan.
    fn translate_lpg(&self, query: &str, language: QueryLanguage) -> Result<LogicalPlan> {
        match language {
            #[cfg(feature = "gql")]
            QueryLanguage::Gql => {
                use crate::query::gql_translator;
                gql_translator::translate(query)
            }
            #[cfg(feature = "cypher")]
            QueryLanguage::Cypher => {
                use crate::query::cypher_translator;
                cypher_translator::translate(query)
            }
            #[cfg(feature = "gremlin")]
            QueryLanguage::Gremlin => {
                use crate::query::gremlin_translator;
                gremlin_translator::translate(query)
            }
            #[cfg(feature = "graphql")]
            QueryLanguage::GraphQL => {
                use crate::query::graphql_translator;
                graphql_translator::translate(query)
            }
            #[allow(unreachable_patterns)]
            _ => Err(Error::Internal(format!(
                "Language {:?} is not an LPG language",
                language
            ))),
        }
    }

    /// Processes an RDF query (SPARQL, GraphQL-RDF).
    #[cfg(feature = "rdf")]
    fn process_rdf(
        &self,
        query: &str,
        language: QueryLanguage,
        _params: Option<&QueryParams>,
    ) -> Result<QueryResult> {
        use crate::query::planner_rdf::RdfPlanner;

        let rdf_store = self.rdf_store.as_ref().ok_or_else(|| {
            Error::Internal("RDF store not configured for this processor".to_string())
        })?;

        // 1. Parse and translate to logical plan
        let logical_plan = self.translate_rdf(query, language)?;

        // 2. Semantic validation
        let mut binder = Binder::new();
        let _binding_context = binder.bind(&logical_plan)?;

        // 3. Optimize the plan
        let optimized_plan = self.optimizer.optimize(logical_plan)?;

        // 4. Convert to physical plan (using RDF planner)
        let planner = RdfPlanner::new(Arc::clone(rdf_store));
        let mut physical_plan = planner.plan(&optimized_plan)?;

        // 5. Execute and collect results
        let executor = Executor::with_columns(physical_plan.columns.clone());
        executor.execute(physical_plan.operator.as_mut())
    }

    /// Translates an RDF query to a logical plan.
    #[cfg(feature = "rdf")]
    fn translate_rdf(&self, query: &str, language: QueryLanguage) -> Result<LogicalPlan> {
        match language {
            #[cfg(feature = "sparql")]
            QueryLanguage::Sparql => {
                use crate::query::sparql_translator;
                sparql_translator::translate(query)
            }
            #[cfg(all(feature = "graphql", feature = "rdf"))]
            QueryLanguage::GraphQLRdf => {
                use crate::query::graphql_rdf_translator;
                // Default namespace for GraphQL-RDF queries
                graphql_rdf_translator::translate(query, "http://example.org/")
            }
            _ => Err(Error::Internal(format!(
                "Language {:?} is not an RDF language",
                language
            ))),
        }
    }

    /// Returns a reference to the LPG store.
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

    /// Returns a reference to the RDF store (if configured).
    #[cfg(feature = "rdf")]
    #[must_use]
    pub fn rdf_store(&self) -> Option<&Arc<grafeo_core::graph::rdf::RdfStore>> {
        self.rdf_store.as_ref()
    }
}

// Legacy compatibility: Keep the old API working
impl QueryProcessor {
    /// Creates a new query processor (legacy API).
    ///
    /// This creates a processor with an empty in-memory store.
    /// Prefer using [`QueryProcessor::for_lpg`] with an explicit store.
    #[must_use]
    #[deprecated(since = "0.1.0", note = "Use QueryProcessor::for_lpg() instead")]
    pub fn new() -> Self {
        Self::for_lpg(Arc::new(LpgStore::new()))
    }

    /// Processes a query using default GQL language (legacy API).
    ///
    /// # Errors
    ///
    /// Returns an error if the query fails.
    #[cfg(feature = "gql")]
    #[deprecated(since = "0.1.0", note = "Use process() with explicit language")]
    pub fn process_legacy(&self, query: &str) -> Result<QueryResult> {
        self.process(query, QueryLanguage::Gql, None)
    }

    /// Returns a reference to the transaction manager.
    #[must_use]
    pub fn tx_manager(&self) -> &Arc<TransactionManager> {
        &self.tx_manager
    }
}

impl Default for QueryProcessor {
    fn default() -> Self {
        Self::for_lpg(Arc::new(LpgStore::new()))
    }
}

/// Substitutes parameters in a logical plan with their values.
fn substitute_params(plan: &mut LogicalPlan, params: &QueryParams) -> Result<()> {
    substitute_in_operator(&mut plan.root, params)
}

/// Recursively substitutes parameters in an operator.
fn substitute_in_operator(op: &mut LogicalOperator, params: &QueryParams) -> Result<()> {
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
            substitute_in_operator(&mut limit.input, params)?;
        }
        LogicalOperator::Skip(skip) => {
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
        LogicalOperator::AddLabel(add_label) => {
            substitute_in_operator(&mut add_label.input, params)?;
        }
        LogicalOperator::RemoveLabel(remove_label) => {
            substitute_in_operator(&mut remove_label.input, params)?;
        }
        LogicalOperator::ShortestPath(sp) => {
            substitute_in_operator(&mut sp.input, params)?;
        }
        LogicalOperator::Empty => {}
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
        LogicalExpression::ExistsSubquery(_) | LogicalExpression::CountSubquery(_) => {
            // Subqueries would need recursive parameter substitution
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
        let store = Arc::new(LpgStore::new());
        let processor = QueryProcessor::for_lpg(store);
        assert!(processor.lpg_store().node_count() == 0);
    }

    #[cfg(feature = "gql")]
    #[test]
    fn test_process_simple_gql() {
        let store = Arc::new(LpgStore::new());
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
        let store = Arc::new(LpgStore::new());
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
        let store = Arc::new(LpgStore::new());
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
        let store = Arc::new(LpgStore::new());
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
}
