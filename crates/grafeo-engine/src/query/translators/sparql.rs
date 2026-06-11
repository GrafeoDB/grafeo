//! SPARQL to LogicalPlan translator.
//!
//! Translates SPARQL 1.1 AST to the common logical plan representation.

use super::common::{wrap_distinct, wrap_filter, wrap_limit, wrap_skip, wrap_sort};
use crate::query::plan::{
    AddGraphOp, AggregateExpr, AggregateFunction, AggregateOp, AntiJoinOp, BinaryOp, BindOp,
    ClearGraphOp, ConstructOp, CopyGraphOp, CreateGraphOp, DatasetRestriction, DeleteTripleOp,
    DropGraphOp, InsertTripleOp, JoinCondition, JoinOp, JoinType, LeftJoinOp, LoadGraphOp,
    LogicalExpression, LogicalOperator, LogicalPlan, ModifyOp, MoveGraphOp, ProjectOp, Projection,
    SortKey, SortOrder, TripleComponent, TripleScanOp, TripleTemplate, UnaryOp, UnionOp,
};
use grafeo_adapters::query::sparql::{self, ast};
use grafeo_common::types::Value;
use grafeo_common::utils::error::{Error, QueryError, QueryErrorKind, Result};
use std::collections::HashMap;
use std::sync::atomic::{AtomicU32, Ordering};

/// Global counter for generating unique query IDs (blank node scoping).
static QUERY_ID_COUNTER: AtomicU32 = AtomicU32::new(0);

/// Translates a SPARQL query string to a logical plan.
///
/// # Errors
///
/// Returns an error if the query cannot be parsed or translated.
pub fn translate(query: &str) -> Result<LogicalPlan> {
    // Check for EXPLAIN [ANALYZE] prefix (case-insensitive, non-standard extension).
    // EXPLAIN: show physical plan without executing.
    // EXPLAIN ANALYZE: execute with profiling, show actual vs estimated stats.
    let trimmed = query.trim_start();
    let (explain, profile, actual_query) = if trimmed
        .get(..7)
        .is_some_and(|s| s.eq_ignore_ascii_case("EXPLAIN"))
        && trimmed
            .as_bytes()
            .get(7)
            .is_some_and(u8::is_ascii_whitespace)
    {
        let rest = trimmed[7..].trim_start();
        if rest
            .get(..7)
            .is_some_and(|s| s.eq_ignore_ascii_case("ANALYZE"))
            && rest.as_bytes().get(7).is_some_and(u8::is_ascii_whitespace)
        {
            // EXPLAIN ANALYZE: execute with profiling
            (false, true, rest[7..].trim_start())
        } else {
            // EXPLAIN: show plan only
            (true, false, rest)
        }
    } else {
        (false, false, query)
    };

    let sparql_query = sparql::parse(actual_query)?;
    let mut translator = SparqlTranslator::new();
    let mut plan = translator.translate_query(&sparql_query)?;
    plan.explain = explain;
    plan.profile = profile;
    Ok(plan)
}

/// Translator from SPARQL AST to LogicalPlan.
struct SparqlTranslator {
    /// Prefix mappings for IRI resolution.
    prefixes: HashMap<String, String>,
    /// Base IRI for relative IRI resolution.
    base: Option<String>,
    /// Counter for generating anonymous variables.
    anon_counter: u32,
    /// Stack of active graph contexts (pushed/popped around GRAPH patterns).
    graph_context_stack: Vec<TripleComponent>,
    /// Unique ID for this query (used for blank node scoping).
    query_id: u32,
    /// Dataset restriction from FROM / FROM NAMED clauses, if any.
    dataset: Option<DatasetRestriction>,
}

impl SparqlTranslator {
    fn new() -> Self {
        Self {
            prefixes: HashMap::new(),
            base: None,
            anon_counter: 0,
            graph_context_stack: Vec::new(),
            query_id: QUERY_ID_COUNTER.fetch_add(1, Ordering::Relaxed),
            dataset: None,
        }
    }

    fn translate_query(&mut self, query: &ast::Query) -> Result<LogicalPlan> {
        // Process prologue
        if let Some(base) = &query.base {
            self.base = Some(base.as_str().to_string());
        }

        for prefix in &query.prefixes {
            self.prefixes
                .insert(prefix.prefix.clone(), prefix.namespace.as_str().to_string());
        }

        // Translate query form
        match &query.query_form {
            ast::QueryForm::Select(select) => self.translate_select(select),
            ast::QueryForm::Ask(ask) => self.translate_ask(ask),
            ast::QueryForm::Construct(construct) => self.translate_construct(construct),
            ast::QueryForm::Describe(describe) => self.translate_describe(describe),
            ast::QueryForm::Update(update) => self.translate_update(update),
        }
    }

    fn translate_select(&mut self, select: &ast::SelectQuery) -> Result<LogicalPlan> {
        // Apply dataset restriction from FROM / FROM NAMED clauses
        self.dataset = self.translate_dataset_clause(&select.dataset);

        // Start with the WHERE clause pattern
        let mut plan = self.translate_graph_pattern(&select.where_clause)?;

        // Clear dataset after translating the WHERE clause
        self.dataset = None;

        // Check if projection contains aggregates (handles both explicit GROUP BY and implicit aggregation)
        let has_aggregates = Self::has_aggregates_in_projection(&select.projection);

        // Apply GROUP BY if present, OR create aggregate for implicit aggregation
        if has_aggregates || select.solution_modifiers.group_by.is_some() {
            let (aggregates, _group_exprs) = self.extract_aggregates_for_select(select)?;

            // Get explicit GROUP BY expressions, or empty vec for whole-dataset aggregation
            let group_by_exprs = if let Some(group_by) = &select.solution_modifiers.group_by {
                group_by
                    .iter()
                    .map(|g| self.translate_group_condition(g))
                    .collect::<Result<Vec<_>>>()?
            } else {
                // No GROUP BY means aggregate over entire dataset (empty group_by)
                Vec::new()
            };

            // Translate HAVING: rewrite aggregate calls as variable references
            // to the computed aggregate column aliases
            let having_expr = if let Some(having) = &select.solution_modifiers.having {
                let raw = self.translate_expression(having)?;
                Some(Self::rewrite_aggregates_as_refs(&raw, &aggregates))
            } else {
                None
            };

            plan = LogicalOperator::Aggregate(AggregateOp {
                group_by: group_by_exprs,
                aggregates,
                input: Box::new(plan),
                having: having_expr,
            });
        }

        // Apply projection before ORDER BY so that computed aliases
        // (e.g. `SELECT (CONCAT(?a, ?b) AS ?full) ... ORDER BY ?full`)
        // are available to the sort operator.
        if !has_aggregates {
            let projections = self.translate_projection(&select.projection)?;
            if !projections.is_empty() {
                plan = LogicalOperator::Project(ProjectOp {
                    projections,
                    input: Box::new(plan),
                    pass_through_input: false,
                });
            }
        }

        // Apply ORDER BY (after projection so aliases are resolvable)
        if let Some(order_by) = &select.solution_modifiers.order_by {
            let keys = order_by
                .iter()
                .map(|oc| {
                    Ok(SortKey {
                        expression: self.translate_expression(&oc.expression)?,
                        order: match oc.direction {
                            ast::SortDirection::Ascending => SortOrder::Ascending,
                            ast::SortDirection::Descending => SortOrder::Descending,
                        },
                        nulls: None,
                    })
                })
                .collect::<Result<Vec<_>>>()?;

            plan = wrap_sort(plan, keys);
        }

        // Apply DISTINCT (after projection, before OFFSET/LIMIT).
        // REDUCED is a no-op: the spec allows returning all rows unchanged.
        if matches!(select.modifier, ast::SelectModifier::Distinct) {
            plan = wrap_distinct(plan);
        }

        // Apply OFFSET
        if let Some(offset) = select.solution_modifiers.offset {
            // reason: SPARQL u64 offset fits usize on 64-bit targets
            #[allow(clippy::cast_possible_truncation)]
            let skip_n = offset as usize;
            plan = wrap_skip(plan, skip_n);
        }

        // Apply LIMIT
        if let Some(limit) = select.solution_modifiers.limit {
            // reason: SPARQL u64 limit fits usize on 64-bit targets
            #[allow(clippy::cast_possible_truncation)]
            let limit_n = limit as usize;
            plan = wrap_limit(plan, limit_n);
        }

        Ok(LogicalPlan::new(plan))
    }

    fn translate_ask(&mut self, ask: &ast::AskQuery) -> Result<LogicalPlan> {
        // Apply dataset restriction from FROM / FROM NAMED clauses
        self.dataset = self.translate_dataset_clause(&ask.dataset);

        // ASK returns true if the pattern has any matches
        let plan = self.translate_graph_pattern(&ask.where_clause)?;

        // Clear dataset after translating the WHERE clause
        self.dataset = None;

        // Limit to 1 result for efficiency
        let plan = wrap_limit(plan, 1);

        Ok(LogicalPlan::new(plan))
    }

    fn translate_construct(&mut self, construct: &ast::ConstructQuery) -> Result<LogicalPlan> {
        // Apply dataset restriction from FROM / FROM NAMED clauses
        self.dataset = self.translate_dataset_clause(&construct.dataset);

        // Evaluate the WHERE pattern to produce variable bindings
        let mut plan = self.translate_graph_pattern(&construct.where_clause)?;

        // Clear dataset after translating the WHERE clause
        self.dataset = None;

        // Apply solution modifiers to the WHERE output
        if let Some(limit) = construct.solution_modifiers.limit {
            // reason: SPARQL u64 limit fits usize on 64-bit targets
            #[allow(clippy::cast_possible_truncation)]
            let limit_n = limit as usize;
            plan = wrap_limit(plan, limit_n);
        }

        // Translate template triples: substitute variable bindings from WHERE.
        // Use translate_data_term for subject/object so that blank nodes become
        // TripleComponent::BlankNode (constants in output) rather than variables
        // that would fail to bind against the WHERE clause results.
        let mut templates = Vec::new();
        for tp in &construct.template {
            let subject = self.translate_data_term(&tp.subject)?;
            // CONSTRUCT templates use simple predicates (IRIs/variables), not paths
            let predicate = match &tp.predicate {
                ast::PropertyPath::Predicate(iri) => TripleComponent::Iri(self.resolve_iri(iri)),
                ast::PropertyPath::Variable(name) => TripleComponent::Variable(name.clone()),
                ast::PropertyPath::RdfType => TripleComponent::Iri(
                    "http://www.w3.org/1999/02/22-rdf-syntax-ns#type".to_string(),
                ),
                _ => continue, // Skip complex property paths in templates
            };
            let object = self.translate_data_term(&tp.object)?;
            templates.push(TripleTemplate {
                subject,
                predicate,
                object,
                graph: None,
            });
        }

        Ok(LogicalPlan::new(LogicalOperator::Construct(ConstructOp {
            templates,
            input: Box::new(plan),
        })))
    }

    fn translate_describe(&mut self, describe: &ast::DescribeQuery) -> Result<LogicalPlan> {
        // Apply dataset restriction from FROM / FROM NAMED clauses
        self.dataset = self.translate_dataset_clause(&describe.dataset);

        // DESCRIBE implements Concise Bounded Description (CBD):
        // Return all triples where the described resource is the subject.
        let pred_var = format!("__describe_p{}", self.next_anon());
        let obj_var = format!("__describe_o{}", self.next_anon());

        let mut cbd_scans: Vec<LogicalOperator> = Vec::new();
        for resource in &describe.resources {
            let subject = match resource {
                ast::VariableOrIri::Iri(iri) => TripleComponent::Iri(self.resolve_iri(iri)),
                ast::VariableOrIri::Variable(name) => TripleComponent::Variable(name.clone()),
            };
            cbd_scans.push(self.make_triple_scan(
                subject,
                TripleComponent::Variable(pred_var.clone()),
                TripleComponent::Variable(obj_var.clone()),
                None,
            ));
        }

        let cbd_plan = if cbd_scans.len() == 1 {
            cbd_scans.into_iter().next().expect("single CBD scan")
        } else {
            LogicalOperator::Union(UnionOp { inputs: cbd_scans })
        };

        let plan = if let Some(where_clause) = &describe.where_clause {
            let where_plan = self.translate_graph_pattern(where_clause)?;
            self.join_patterns(where_plan, cbd_plan)
        } else {
            cbd_plan
        };

        // Clear dataset after translating the WHERE clause
        self.dataset = None;

        Ok(LogicalPlan::new(plan))
    }

    // ==================== SPARQL Update Translation ====================

    fn translate_update(&mut self, update: &ast::UpdateOperation) -> Result<LogicalPlan> {
        match update {
            ast::UpdateOperation::InsertData { data } => self.translate_insert_data(data),
            ast::UpdateOperation::DeleteData { data } => self.translate_delete_data(data),
            ast::UpdateOperation::DeleteWhere { pattern } => self.translate_delete_where(pattern),
            ast::UpdateOperation::Modify {
                with_graph,
                delete_template,
                insert_template,
                using_clauses,
                where_clause,
            } => self.translate_modify(
                with_graph,
                delete_template,
                insert_template,
                using_clauses,
                where_clause,
            ),
            ast::UpdateOperation::Load {
                silent,
                source,
                destination,
            } => self.translate_load(*silent, source, destination.as_ref()),
            ast::UpdateOperation::Clear { silent, target } => self.translate_clear(*silent, target),
            ast::UpdateOperation::Drop { silent, target } => self.translate_drop(*silent, target),
            ast::UpdateOperation::Create { silent, graph } => self.translate_create(*silent, graph),
            ast::UpdateOperation::Copy {
                silent,
                source,
                destination,
            } => self.translate_copy(*silent, source, destination),
            ast::UpdateOperation::Move {
                silent,
                source,
                destination,
            } => self.translate_move(*silent, source, destination),
            ast::UpdateOperation::Add {
                silent,
                source,
                destination,
            } => self.translate_add(*silent, source, destination),
        }
    }

    fn translate_insert_data(&mut self, data: &[ast::QuadPattern]) -> Result<LogicalPlan> {
        // Build a sequence of InsertTriple operators
        let mut ops = Vec::new();
        for quad in data {
            let subject = self.translate_data_term(&quad.triple.subject)?;
            let predicate = self.translate_property_path(&quad.triple.predicate)?;
            let object = self.translate_data_term(&quad.triple.object)?;
            let graph = quad.graph.as_ref().map(|g| self.resolve_variable_or_iri(g));

            ops.push(LogicalOperator::InsertTriple(InsertTripleOp {
                subject,
                predicate,
                object,
                graph,
                input: None,
            }));
        }

        // Combine all inserts into a sequence using Union
        if ops.is_empty() {
            Ok(LogicalPlan::new(LogicalOperator::Empty))
        } else if ops.len() == 1 {
            Ok(LogicalPlan::new(
                ops.into_iter().next().expect("single-element iterator"),
            ))
        } else {
            Ok(LogicalPlan::new(LogicalOperator::Union(UnionOp {
                inputs: ops,
            })))
        }
    }

    /// Translates a triple term in a data context (INSERT DATA / DELETE DATA).
    ///
    /// Blank nodes become `TripleComponent::BlankNode` instead of variables,
    /// because data operations have no WHERE clause to bind variables against.
    fn translate_data_term(&mut self, term: &ast::TripleTerm) -> Result<TripleComponent> {
        match term {
            ast::TripleTerm::BlankNode(bnode) => {
                let label = match bnode {
                    ast::BlankNode::Labeled(label) => {
                        format!("q{}_{label}", self.query_id)
                    }
                    ast::BlankNode::Anonymous(_) => {
                        let anon = self.next_anon();
                        format!("q{}_anon{anon}", self.query_id)
                    }
                };
                Ok(TripleComponent::BlankNode(label))
            }
            // Non-blank-node terms use the standard translation
            other => self.translate_triple_term(other),
        }
    }

    fn translate_delete_data(&mut self, data: &[ast::QuadPattern]) -> Result<LogicalPlan> {
        // Build a sequence of DeleteTriple operators
        let mut ops = Vec::new();
        for quad in data {
            let subject = self.translate_triple_term(&quad.triple.subject)?;
            let predicate = self.translate_property_path(&quad.triple.predicate)?;
            let object = self.translate_triple_term(&quad.triple.object)?;
            let graph = quad.graph.as_ref().map(|g| self.resolve_variable_or_iri(g));

            ops.push(LogicalOperator::DeleteTriple(DeleteTripleOp {
                subject,
                predicate,
                object,
                graph,
                input: None,
            }));
        }

        if ops.is_empty() {
            Ok(LogicalPlan::new(LogicalOperator::Empty))
        } else if ops.len() == 1 {
            Ok(LogicalPlan::new(
                ops.into_iter().next().expect("single-element iterator"),
            ))
        } else {
            Ok(LogicalPlan::new(LogicalOperator::Union(UnionOp {
                inputs: ops,
            })))
        }
    }

    fn translate_delete_where(&mut self, pattern: &ast::GraphPattern) -> Result<LogicalPlan> {
        // DELETE WHERE uses the pattern both for matching and deletion
        // First, translate the pattern to get bindings
        let match_plan = self.translate_graph_pattern(pattern)?;

        // For DELETE WHERE, the WHERE pattern is the same as the delete template
        // We extract triples from the pattern and create delete operations
        let triples = Self::extract_triples_from_pattern(pattern);

        // Build delete operators with the match plan as input
        let mut ops = Vec::new();
        for (triple, graph) in &triples {
            let subject = self.translate_triple_term(&triple.subject)?;
            let predicate = self.translate_property_path(&triple.predicate)?;
            let object = self.translate_triple_term(&triple.object)?;
            // A GRAPH-wrapped DELETE WHERE deletes from the named graph; a
            // variable graph (`GRAPH ?g`) resolves to a "?"-prefixed name that
            // the executor rejects loudly (fail-closed).
            let graph = graph.as_ref().map(|g| self.resolve_variable_or_iri(g));

            ops.push(LogicalOperator::DeleteTriple(DeleteTripleOp {
                subject,
                predicate,
                object,
                graph,
                input: Some(Box::new(match_plan.clone())),
            }));
        }

        if ops.is_empty() {
            Ok(LogicalPlan::new(match_plan))
        } else if ops.len() == 1 {
            Ok(LogicalPlan::new(
                ops.into_iter().next().expect("single-element iterator"),
            ))
        } else {
            Ok(LogicalPlan::new(LogicalOperator::Union(UnionOp {
                inputs: ops,
            })))
        }
    }

    /// Extracts the triple templates from a DELETE WHERE pattern, tagging each
    /// with its enclosing named graph (if any) so the delete routes to the
    /// correct sub-store. Descends into GRAPH blocks via the `NamedGraph` arm;
    /// dropping that arm is what made a GRAPH-wrapped DELETE WHERE a silent
    /// no-op (the template came back empty).
    fn extract_triples_from_pattern(
        pattern: &ast::GraphPattern,
    ) -> Vec<(ast::TriplePattern, Option<ast::VariableOrIri>)> {
        Self::extract_triples_with_graph(pattern, None)
    }

    fn extract_triples_with_graph(
        pattern: &ast::GraphPattern,
        graph: Option<&ast::VariableOrIri>,
    ) -> Vec<(ast::TriplePattern, Option<ast::VariableOrIri>)> {
        match pattern {
            ast::GraphPattern::Basic(triples) => triples
                .iter()
                .map(|t| (t.clone(), graph.cloned()))
                .collect(),
            ast::GraphPattern::Group(patterns) => patterns
                .iter()
                .flat_map(|p| Self::extract_triples_with_graph(p, graph))
                .collect(),
            ast::GraphPattern::NamedGraph {
                graph: inner_graph,
                pattern: inner_pattern,
            } => Self::extract_triples_with_graph(inner_pattern, Some(inner_graph)),
            _ => Vec::new(),
        }
    }

    fn translate_modify(
        &mut self,
        with_graph: &Option<ast::Iri>,
        delete_template: &Option<Vec<ast::QuadPattern>>,
        insert_template: &Option<Vec<ast::QuadPattern>>,
        using_clauses: &[ast::UsingClause],
        where_clause: &ast::GraphPattern,
    ) -> Result<LogicalPlan> {
        // USING / USING NAMED is parsed but not yet honoured by the executor.
        // Fail closed rather than silently dropping the clause and mis-scoping
        // the WHERE evaluation against the wrong dataset.
        if !using_clauses.is_empty() {
            return Err(Error::Query(QueryError::new(
                QueryErrorKind::Semantic,
                "SPARQL Update USING / USING NAMED is not yet supported",
            )));
        }

        let default_graph = with_graph.as_ref().map(|g| self.resolve_iri(g));

        // The WITH clause redefines the WHERE clause's default graph as <g>
        // (SPARQL 1.1 Update, DELETE/INSERT): an unqualified WHERE pattern must
        // bind rows from the named graph, not the empty default graph. Redirect
        // the dataset before building the WHERE plan (mirroring translate_select)
        // so a WITH-scoped Modify does not silently no-op for lack of bindings.
        // A GRAPH-wrapped WHERE still binds via the graph context stack and is
        // unaffected.
        if let Some(ref graph) = default_graph {
            self.dataset = Some(DatasetRestriction {
                default_graphs: vec![graph.clone()],
                named_graphs: Vec::new(),
            });
        }

        // Translate the WHERE clause - this will be evaluated once and shared
        let where_plan = self.translate_graph_pattern(where_clause)?;

        // Clear the dataset redirect after translating the WHERE clause
        self.dataset = None;

        // Build DELETE templates
        let mut delete_templates = Vec::new();
        if let Some(delete_quads) = delete_template {
            for quad in delete_quads {
                let subject = self.translate_triple_term(&quad.triple.subject)?;
                let predicate = self.translate_property_path(&quad.triple.predicate)?;
                let object = self.translate_triple_term(&quad.triple.object)?;
                let graph = quad
                    .graph
                    .as_ref()
                    .map(|g| self.resolve_variable_or_iri(g))
                    .or_else(|| default_graph.clone());

                delete_templates.push(TripleTemplate {
                    subject,
                    predicate,
                    object,
                    graph,
                });
            }
        }

        // Build INSERT templates
        let mut insert_templates = Vec::new();
        if let Some(insert_quads) = insert_template {
            for quad in insert_quads {
                let subject = self.translate_triple_term(&quad.triple.subject)?;
                let predicate = self.translate_property_path(&quad.triple.predicate)?;
                let object = self.translate_triple_term(&quad.triple.object)?;
                let graph = quad
                    .graph
                    .as_ref()
                    .map(|g| self.resolve_variable_or_iri(g))
                    .or_else(|| default_graph.clone());

                insert_templates.push(TripleTemplate {
                    subject,
                    predicate,
                    object,
                    graph,
                });
            }
        }

        // Use ModifyOp which handles SPARQL MODIFY semantics correctly:
        // 1. Evaluate WHERE once
        // 2. Apply DELETE templates
        // 3. Apply INSERT templates (using same bindings)
        Ok(LogicalPlan::new(LogicalOperator::Modify(ModifyOp {
            delete_templates,
            insert_templates,
            where_clause: Box::new(where_plan),
            graph: default_graph,
        })))
    }

    fn translate_load(
        &mut self,
        silent: bool,
        source: &ast::Iri,
        destination: Option<&ast::Iri>,
    ) -> Result<LogicalPlan> {
        Ok(LogicalPlan::new(LogicalOperator::LoadGraph(LoadGraphOp {
            source: self.resolve_iri(source),
            destination: destination.map(|d| self.resolve_iri(d)),
            silent,
        })))
    }

    fn translate_clear(&mut self, silent: bool, target: &ast::GraphTarget) -> Result<LogicalPlan> {
        let graph = self.translate_graph_target(target);
        Ok(LogicalPlan::new(LogicalOperator::ClearGraph(
            ClearGraphOp { graph, silent },
        )))
    }

    fn translate_drop(&mut self, silent: bool, target: &ast::GraphTarget) -> Result<LogicalPlan> {
        let graph = self.translate_graph_target(target);
        Ok(LogicalPlan::new(LogicalOperator::DropGraph(DropGraphOp {
            graph,
            silent,
        })))
    }

    fn translate_create(&mut self, silent: bool, graph: &ast::Iri) -> Result<LogicalPlan> {
        Ok(LogicalPlan::new(LogicalOperator::CreateGraph(
            CreateGraphOp {
                graph: self.resolve_iri(graph),
                silent,
            },
        )))
    }

    fn translate_copy(
        &mut self,
        silent: bool,
        source: &ast::GraphTarget,
        destination: &ast::GraphTarget,
    ) -> Result<LogicalPlan> {
        Ok(LogicalPlan::new(LogicalOperator::CopyGraph(CopyGraphOp {
            source: self.translate_graph_target(source),
            destination: self.translate_graph_target(destination),
            silent,
        })))
    }

    fn translate_move(
        &mut self,
        silent: bool,
        source: &ast::GraphTarget,
        destination: &ast::GraphTarget,
    ) -> Result<LogicalPlan> {
        Ok(LogicalPlan::new(LogicalOperator::MoveGraph(MoveGraphOp {
            source: self.translate_graph_target(source),
            destination: self.translate_graph_target(destination),
            silent,
        })))
    }

    fn translate_add(
        &mut self,
        silent: bool,
        source: &ast::GraphTarget,
        destination: &ast::GraphTarget,
    ) -> Result<LogicalPlan> {
        Ok(LogicalPlan::new(LogicalOperator::AddGraph(AddGraphOp {
            source: self.translate_graph_target(source),
            destination: self.translate_graph_target(destination),
            silent,
        })))
    }

    fn translate_graph_target(&self, target: &ast::GraphTarget) -> Option<String> {
        match target {
            ast::GraphTarget::Default => None,
            ast::GraphTarget::Named(iri) => Some(self.resolve_iri(iri)),
            ast::GraphTarget::All => Some(String::new()), // Empty string represents "all"
        }
    }

    fn resolve_variable_or_iri(&self, var_or_iri: &ast::VariableOrIri) -> String {
        match var_or_iri {
            ast::VariableOrIri::Variable(name) => format!("?{}", name),
            ast::VariableOrIri::Iri(iri) => self.resolve_iri(iri),
        }
    }

    fn translate_projection(&mut self, projection: &ast::Projection) -> Result<Vec<Projection>> {
        match projection {
            ast::Projection::Wildcard => Ok(Vec::new()), // Empty means select all
            ast::Projection::Variables(vars) => vars
                .iter()
                .map(|pv| {
                    Ok(Projection {
                        expression: self.translate_expression(&pv.expression)?,
                        alias: pv.alias.clone(),
                    })
                })
                .collect(),
        }
    }

    fn translate_graph_pattern(&mut self, pattern: &ast::GraphPattern) -> Result<LogicalOperator> {
        match pattern {
            ast::GraphPattern::Basic(triples) => self.translate_basic_pattern(triples),

            ast::GraphPattern::Group(patterns) => {
                // Process patterns in document order so that BIND, OPTIONAL,
                // MINUS, etc. see the variables introduced by preceding
                // patterns. FILTER and FILTER NOT EXISTS/EXISTS scope over
                // the entire group (SPARQL spec), so they are collected and
                // applied last.
                let mut filter_exprs: Vec<&ast::Expression> = Vec::new();
                let mut not_exists_patterns: Vec<&ast::GraphPattern> = Vec::new();
                let mut exists_patterns: Vec<&ast::GraphPattern> = Vec::new();

                let mut plan = LogicalOperator::Empty;

                for p in patterns {
                    match p {
                        ast::GraphPattern::Filter(expr) => {
                            // Collect filters for group-level application
                            match expr {
                                ast::Expression::NotExists(inner) => {
                                    not_exists_patterns.push(inner);
                                }
                                ast::Expression::Exists(inner) => {
                                    exists_patterns.push(inner);
                                }
                                _ => filter_exprs.push(expr),
                            }
                        }
                        ast::GraphPattern::Bind {
                            expression,
                            variable,
                        } => {
                            let expr = self.translate_expression(expression)?;
                            plan = LogicalOperator::Bind(BindOp {
                                expression: expr,
                                variable: variable.clone(),
                                input: Box::new(plan),
                            });
                        }
                        ast::GraphPattern::Optional(inner) => {
                            let inner_plan = self.translate_graph_pattern(inner)?;
                            if matches!(plan, LogicalOperator::Empty) {
                                plan = inner_plan;
                            } else {
                                plan = LogicalOperator::LeftJoin(LeftJoinOp {
                                    left: Box::new(plan),
                                    right: Box::new(inner_plan),
                                    condition: None,
                                });
                            }
                        }
                        ast::GraphPattern::Minus(inner) => {
                            let inner_plan = self.translate_graph_pattern(inner)?;
                            if !matches!(plan, LogicalOperator::Empty) {
                                plan = LogicalOperator::AntiJoin(AntiJoinOp {
                                    left: Box::new(plan),
                                    right: Box::new(inner_plan),
                                });
                            }
                        }
                        // InlineData with UNDEF entries: translate as a Union of
                        // filtered copies of the base plan.
                        ast::GraphPattern::InlineData(data)
                            if data
                                .values
                                .iter()
                                .any(|row| row.iter().any(Option::is_none)) =>
                        {
                            if matches!(plan, LogicalOperator::Empty) {
                                plan = self.translate_graph_pattern(
                                    &ast::GraphPattern::InlineData(data.clone()),
                                )?;
                            } else {
                                let mut branches = Vec::new();
                                for row in &data.values {
                                    let mut conditions: Vec<LogicalExpression> = Vec::new();
                                    for (var, val) in data.variables.iter().zip(row.iter()) {
                                        if let Some(dv) = val {
                                            let value = self.data_value_to_value(dv);
                                            conditions.push(LogicalExpression::Binary {
                                                left: Box::new(LogicalExpression::Variable(
                                                    var.clone(),
                                                )),
                                                op: BinaryOp::Eq,
                                                right: Box::new(LogicalExpression::Literal(value)),
                                            });
                                        }
                                    }
                                    if conditions.is_empty() {
                                        branches.push(plan.clone());
                                    } else {
                                        let combined = conditions
                                            .into_iter()
                                            .reduce(|acc, c| LogicalExpression::Binary {
                                                left: Box::new(acc),
                                                op: BinaryOp::And,
                                                right: Box::new(c),
                                            })
                                            .expect("conditions non-empty");
                                        branches.push(wrap_filter(plan.clone(), combined));
                                    }
                                }
                                plan = if branches.len() == 1 {
                                    branches.into_iter().next().expect("single branch")
                                } else {
                                    LogicalOperator::Union(UnionOp { inputs: branches })
                                };
                            }
                        }
                        _ => {
                            let p_plan = self.translate_graph_pattern(p)?;
                            plan = self.join_patterns(plan, p_plan);
                        }
                    }
                }

                // Apply FILTER NOT EXISTS as anti joins
                for inner in not_exists_patterns {
                    let inner_plan = self.translate_graph_pattern(inner)?;
                    if !matches!(plan, LogicalOperator::Empty) {
                        plan = LogicalOperator::AntiJoin(AntiJoinOp {
                            left: Box::new(plan),
                            right: Box::new(inner_plan),
                        });
                    }
                }

                // 4c. Apply FILTER EXISTS as semi joins (inner join)
                for inner in exists_patterns {
                    let inner_plan = self.translate_graph_pattern(inner)?;
                    if !matches!(plan, LogicalOperator::Empty) {
                        plan = self.join_patterns(plan, inner_plan);
                    }
                }

                // 5. Apply FILTER expressions last (they scope over entire group)
                if !filter_exprs.is_empty() {
                    let predicates: Vec<LogicalExpression> = filter_exprs
                        .into_iter()
                        .map(|e| self.translate_expression(e))
                        .collect::<Result<Vec<_>>>()?;

                    // Combine all predicates with AND
                    let combined = predicates
                        .into_iter()
                        .reduce(|acc, pred| LogicalExpression::Binary {
                            left: Box::new(acc),
                            op: BinaryOp::And,
                            right: Box::new(pred),
                        })
                        .expect("predicates non-empty after is_empty check");

                    plan = wrap_filter(plan, combined);
                }

                Ok(plan)
            }

            ast::GraphPattern::Optional(inner) => {
                // Standalone OPTIONAL - handled in Group translation, but support direct call
                self.translate_graph_pattern(inner)
            }

            ast::GraphPattern::Union(alternatives) => {
                let inputs = alternatives
                    .iter()
                    .map(|p| self.translate_graph_pattern(p))
                    .collect::<Result<Vec<_>>>()?;

                Ok(LogicalOperator::Union(UnionOp { inputs }))
            }

            ast::GraphPattern::Minus(inner) => {
                // Standalone MINUS - handled in Group translation, but support direct call
                self.translate_graph_pattern(inner)
            }

            ast::GraphPattern::Filter(expr) => {
                // Standalone FILTER - handled in Group translation, but support direct call
                // This can happen when Filter is the top-level pattern
                let predicate = self.translate_expression(expr)?;
                Ok(wrap_filter(LogicalOperator::Empty, predicate))
            }

            ast::GraphPattern::Bind {
                expression,
                variable,
            } => {
                // Standalone BIND - handled in Group translation, but support direct call
                let expr = self.translate_expression(expression)?;
                Ok(LogicalOperator::Bind(BindOp {
                    expression: expr,
                    variable: variable.clone(),
                    input: Box::new(LogicalOperator::Empty),
                }))
            }

            ast::GraphPattern::NamedGraph { graph, pattern } => {
                let graph_component = match graph {
                    ast::VariableOrIri::Variable(name) => TripleComponent::Variable(name.clone()),
                    ast::VariableOrIri::Iri(iri) => TripleComponent::Iri(self.resolve_iri(iri)),
                };
                self.graph_context_stack.push(graph_component);
                let plan = self.translate_graph_pattern(pattern);
                self.graph_context_stack.pop();
                plan
            }

            ast::GraphPattern::SubSelect(subquery) => {
                let plan = self.translate_select(subquery)?;
                Ok(plan.root)
            }

            ast::GraphPattern::Service { .. } => Err(Error::Query(QueryError::new(
                QueryErrorKind::Semantic,
                "SPARQL SERVICE (federated queries) is not yet supported",
            ))),

            ast::GraphPattern::InlineData(data) => {
                // VALUES clause: each row becomes a chain of BIND operators
                // starting from Empty, and all rows are combined with UNION.
                if data.values.is_empty() {
                    return Ok(LogicalOperator::Empty);
                }
                let mut branches = Vec::new();
                for row in &data.values {
                    let mut plan = LogicalOperator::Empty;
                    for (var, val) in data.variables.iter().zip(row.iter()) {
                        if let Some(dv) = val {
                            let value = self.data_value_to_value(dv);
                            plan = LogicalOperator::Bind(BindOp {
                                expression: LogicalExpression::Literal(value),
                                variable: var.clone(),
                                input: Box::new(plan),
                            });
                        }
                    }
                    branches.push(plan);
                }
                if branches.len() == 1 {
                    Ok(branches
                        .into_iter()
                        .next()
                        .expect("single-element iterator"))
                } else {
                    Ok(LogicalOperator::Union(UnionOp { inputs: branches }))
                }
            }
        }
    }

    fn translate_basic_pattern(
        &mut self,
        triples: &[ast::TriplePattern],
    ) -> Result<LogicalOperator> {
        if triples.is_empty() {
            return Ok(LogicalOperator::Empty);
        }

        let mut plan = LogicalOperator::Empty;

        for triple in triples {
            let triple_scan = self.translate_triple_pattern(triple)?;
            plan = self.join_patterns(plan, triple_scan);
        }

        Ok(plan)
    }

    fn translate_triple_pattern(&mut self, triple: &ast::TriplePattern) -> Result<LogicalOperator> {
        // Handle Sequence property paths: expand into chained triple patterns
        // e.g. ?person foaf:knows/foaf:name ?name  becomes:
        //   ?person foaf:knows ?_anon0 . ?_anon0 foaf:name ?name
        if let ast::PropertyPath::Sequence(paths) = &triple.predicate {
            let subject = self.translate_triple_term(&triple.subject)?;
            let object = self.translate_triple_term(&triple.object)?;
            let graph = self.graph_context_stack.last().cloned();

            let mut current_subject = subject;
            let mut plan = LogicalOperator::Empty;

            for (i, path) in paths.iter().enumerate() {
                let next_object = if i == paths.len() - 1 {
                    object.clone()
                } else {
                    TripleComponent::Variable(format!("_:seq{}", self.next_anon()))
                };

                let step = if self.is_simple_path(path) {
                    let pred = self.translate_property_path(path)?;
                    self.make_triple_scan(
                        current_subject.clone(),
                        pred,
                        next_object.clone(),
                        graph.clone(),
                    )
                } else {
                    // Complex path (ZeroOrMore, OneOrMore, etc.): recurse
                    let sub_triple = ast::TriplePattern {
                        subject: self.triple_component_to_term(&current_subject),
                        predicate: path.clone(),
                        object: self.triple_component_to_term(&next_object),
                    };
                    self.translate_triple_pattern(&sub_triple)?
                };

                plan = self.join_patterns(plan, step);
                current_subject = next_object;
            }

            return Ok(plan);
        }

        // Handle Alternative property paths: translate as Union of triple scans
        if let ast::PropertyPath::Alternative(alternatives) = &triple.predicate {
            let subject = self.translate_triple_term(&triple.subject)?;
            let object = self.translate_triple_term(&triple.object)?;
            let graph = self.graph_context_stack.last().cloned();

            let mut branches = Vec::new();
            for alt_path in alternatives {
                let pred = self.translate_property_path(alt_path)?;
                branches.push(self.make_triple_scan(
                    subject.clone(),
                    pred,
                    object.clone(),
                    graph.clone(),
                ));
            }

            return Ok(LogicalOperator::Union(UnionOp { inputs: branches }));
        }

        // Handle OneOrMore (path+): bounded expansion
        if let ast::PropertyPath::OneOrMore(inner) = &triple.predicate {
            return self.translate_one_or_more_path(triple, inner);
        }

        // Handle ZeroOrMore (path*): bounded expansion
        if let ast::PropertyPath::ZeroOrMore(inner) = &triple.predicate {
            return self.translate_zero_or_more_path(triple, inner);
        }

        // Handle Inverse (^path): swap subject and object, translate inner path
        if let ast::PropertyPath::Inverse(inner) = &triple.predicate {
            let swapped = ast::TriplePattern {
                subject: triple.object.clone(),
                predicate: *inner.clone(),
                object: triple.subject.clone(),
            };
            return self.translate_triple_pattern(&swapped);
        }

        // Handle ZeroOrOne (path?): union of reflexive 0-hop and 1-hop
        if let ast::PropertyPath::ZeroOrOne(inner) = &triple.predicate {
            return self.translate_zero_or_one_path(triple, inner);
        }

        // Handle Negation: !(iri1|^iri2) scans all triples and filters out excluded predicates
        if let ast::PropertyPath::Negation(negated_iris) = &triple.predicate {
            return self.translate_negated_property_set(triple, negated_iris);
        }

        let subject = self.translate_triple_term(&triple.subject)?;
        let predicate = self.translate_property_path(&triple.predicate)?;
        let object = self.translate_triple_term(&triple.object)?;

        Ok(self.make_triple_scan(
            subject,
            predicate,
            object,
            self.graph_context_stack.last().cloned(),
        ))
    }

    fn translate_triple_term(&mut self, term: &ast::TripleTerm) -> Result<TripleComponent> {
        match term {
            ast::TripleTerm::Variable(name) => Ok(TripleComponent::Variable(name.clone())),
            ast::TripleTerm::Iri(iri) => Ok(TripleComponent::Iri(self.resolve_iri(iri))),
            ast::TripleTerm::Literal(lit) => {
                if let Some(lang) = &lit.language {
                    Ok(TripleComponent::LangLiteral {
                        value: lit.value.clone(),
                        lang: lang.clone(),
                    })
                } else {
                    let value = self.literal_to_value(lit);
                    Ok(TripleComponent::Literal(value))
                }
            }
            ast::TripleTerm::BlankNode(bnode) => {
                // Treat blank nodes as variables, scoped by query_id
                match bnode {
                    ast::BlankNode::Labeled(label) => Ok(TripleComponent::Variable(format!(
                        "_:q{}_{label}",
                        self.query_id
                    ))),
                    ast::BlankNode::Anonymous(_) => {
                        let anon = self.next_anon();
                        let var = format!("_:q{}_anon{anon}", self.query_id);
                        Ok(TripleComponent::Variable(var))
                    }
                }
            }
        }
    }

    /// Returns true if the property path is a simple predicate (IRI, variable, or rdf:type).
    fn is_simple_path(&self, path: &ast::PropertyPath) -> bool {
        matches!(
            path,
            ast::PropertyPath::Predicate(_)
                | ast::PropertyPath::Variable(_)
                | ast::PropertyPath::RdfType
        )
    }

    /// Converts a `TripleComponent` back to an AST `TripleTerm` for recursive translation.
    fn triple_component_to_term(&self, component: &TripleComponent) -> ast::TripleTerm {
        match component {
            TripleComponent::Variable(name) => ast::TripleTerm::Variable(name.clone()),
            TripleComponent::Iri(iri) => ast::TripleTerm::Iri(ast::Iri(iri.clone())),
            TripleComponent::Literal(val) => ast::TripleTerm::Literal(ast::Literal {
                value: val.to_string(),
                datatype: None,
                language: None,
            }),
            TripleComponent::LangLiteral { value, lang } => {
                ast::TripleTerm::Literal(ast::Literal {
                    value: value.clone(),
                    datatype: None,
                    language: Some(lang.clone()),
                })
            }
            TripleComponent::BlankNode(label) => {
                ast::TripleTerm::BlankNode(ast::BlankNode::Labeled(label.clone()))
            }
        }
    }

    fn translate_property_path(&mut self, path: &ast::PropertyPath) -> Result<TripleComponent> {
        match path {
            ast::PropertyPath::Predicate(iri) => Ok(TripleComponent::Iri(self.resolve_iri(iri))),
            ast::PropertyPath::Variable(name) => Ok(TripleComponent::Variable(name.clone())),
            ast::PropertyPath::RdfType => Ok(TripleComponent::Iri(
                "http://www.w3.org/1999/02/22-rdf-syntax-ns#type".to_string(),
            )),
            // Complex property paths are not fully supported yet
            _ => Err(Error::Query(QueryError::new(
                QueryErrorKind::Semantic,
                "Complex property paths not yet supported",
            ))),
        }
    }

    fn translate_expression(&mut self, expr: &ast::Expression) -> Result<LogicalExpression> {
        match expr {
            ast::Expression::Variable(name) => Ok(LogicalExpression::Variable(name.clone())),

            ast::Expression::Iri(iri) => Ok(LogicalExpression::Literal(Value::String(
                self.resolve_iri(iri).into(),
            ))),

            ast::Expression::Literal(lit) => {
                Ok(LogicalExpression::Literal(self.literal_to_value(lit)))
            }

            ast::Expression::Binary {
                left,
                operator,
                right,
            } => {
                // Detect language-tagged literal comparisons: ?var = "value"@lang
                // or "value"@lang = ?var. Rewrite to check both lexical value and
                // language tag so that "Barcelona"@es does not match "Barcelona"@ca.
                if matches!(
                    operator,
                    ast::BinaryOperator::Equal | ast::BinaryOperator::NotEqual
                ) && let Some(expanded) =
                    self.try_expand_lang_comparison(left, *operator, right)?
                {
                    return Ok(expanded);
                }

                let left = self.translate_expression(left)?;
                let right = self.translate_expression(right)?;
                let op = self.translate_binary_op(*operator);
                Ok(LogicalExpression::Binary {
                    left: Box::new(left),
                    op,
                    right: Box::new(right),
                })
            }

            ast::Expression::Unary { operator, operand } => {
                let operand = self.translate_expression(operand)?;
                // Unary plus is a no-op: just return the operand unchanged
                if *operator == ast::UnaryOperator::Plus {
                    return Ok(operand);
                }
                let op = self.translate_unary_op(*operator);
                Ok(LogicalExpression::Unary {
                    op,
                    operand: Box::new(operand),
                })
            }

            ast::Expression::FunctionCall {
                function,
                arguments,
            } => {
                let name = self.translate_function_name(function);
                let args = arguments
                    .iter()
                    .map(|a| self.translate_expression(a))
                    .collect::<Result<Vec<_>>>()?;
                Ok(LogicalExpression::FunctionCall {
                    name,
                    args,
                    distinct: false,
                })
            }

            ast::Expression::Bound(var) => {
                // BOUND(?x) checks if variable is bound
                Ok(LogicalExpression::FunctionCall {
                    name: "BOUND".to_string(),
                    args: vec![LogicalExpression::Variable(var.clone())],
                    distinct: false,
                })
            }

            ast::Expression::Conditional {
                condition,
                then_expression,
                else_expression,
            } => {
                let cond = self.translate_expression(condition)?;
                let then_expr = self.translate_expression(then_expression)?;
                let else_expr = self.translate_expression(else_expression)?;
                Ok(LogicalExpression::Case {
                    operand: None,
                    when_clauses: vec![(cond, then_expr)],
                    else_clause: Some(Box::new(else_expr)),
                })
            }

            ast::Expression::Coalesce(exprs) => {
                let args = exprs
                    .iter()
                    .map(|e| self.translate_expression(e))
                    .collect::<Result<Vec<_>>>()?;
                Ok(LogicalExpression::FunctionCall {
                    name: "COALESCE".to_string(),
                    args,
                    distinct: false,
                })
            }

            ast::Expression::Exists(pattern) => {
                // EXISTS { pattern } - check if pattern has any matches
                let subquery = self.translate_graph_pattern(pattern)?;
                Ok(LogicalExpression::ExistsSubquery(Box::new(subquery)))
            }

            ast::Expression::NotExists(pattern) => {
                // NOT EXISTS { pattern } - negate the EXISTS check
                let subquery = self.translate_graph_pattern(pattern)?;
                Ok(LogicalExpression::Unary {
                    op: UnaryOp::Not,
                    operand: Box::new(LogicalExpression::ExistsSubquery(Box::new(subquery))),
                })
            }

            ast::Expression::In { expression, list } => {
                let expr = self.translate_expression(expression)?;
                let items = list
                    .iter()
                    .map(|e| self.translate_expression(e))
                    .collect::<Result<Vec<_>>>()?;
                Ok(LogicalExpression::Binary {
                    left: Box::new(expr),
                    op: BinaryOp::In,
                    right: Box::new(LogicalExpression::List(items)),
                })
            }

            ast::Expression::NotIn { expression, list } => {
                let expr = self.translate_expression(expression)?;
                let items = list
                    .iter()
                    .map(|e| self.translate_expression(e))
                    .collect::<Result<Vec<_>>>()?;
                Ok(LogicalExpression::Unary {
                    op: UnaryOp::Not,
                    operand: Box::new(LogicalExpression::Binary {
                        left: Box::new(expr),
                        op: BinaryOp::In,
                        right: Box::new(LogicalExpression::List(items)),
                    }),
                })
            }

            ast::Expression::Aggregate(agg) => self.translate_aggregate_expression(agg),

            ast::Expression::Bracketed(inner) => self.translate_expression(inner),
        }
    }

    fn translate_aggregate_expression(
        &mut self,
        agg: &ast::AggregateExpression,
    ) -> Result<LogicalExpression> {
        let (func_name, distinct) = match agg {
            ast::AggregateExpression::Count { distinct, .. } => ("COUNT", *distinct),
            ast::AggregateExpression::Sum { distinct, .. } => ("SUM", *distinct),
            ast::AggregateExpression::Average { distinct, .. } => ("AVG", *distinct),
            ast::AggregateExpression::Minimum { .. } => ("MIN", false),
            ast::AggregateExpression::Maximum { .. } => ("MAX", false),
            ast::AggregateExpression::Sample { .. } => ("SAMPLE", false),
            ast::AggregateExpression::GroupConcat { distinct, .. } => ("GROUP_CONCAT", *distinct),
        };

        let args = match agg {
            ast::AggregateExpression::Count { expression, .. } => {
                if let Some(expr) = expression {
                    vec![self.translate_expression(expr)?]
                } else {
                    vec![]
                }
            }
            ast::AggregateExpression::Sum { expression, .. }
            | ast::AggregateExpression::Average { expression, .. }
            | ast::AggregateExpression::Minimum { expression, .. }
            | ast::AggregateExpression::Maximum { expression, .. }
            | ast::AggregateExpression::Sample { expression, .. }
            | ast::AggregateExpression::GroupConcat { expression, .. } => {
                vec![self.translate_expression(expression)?]
            }
        };

        Ok(LogicalExpression::FunctionCall {
            name: func_name.to_string(),
            args,
            distinct,
        })
    }

    fn translate_binary_op(&self, op: ast::BinaryOperator) -> BinaryOp {
        match op {
            ast::BinaryOperator::Or => BinaryOp::Or,
            ast::BinaryOperator::And => BinaryOp::And,
            ast::BinaryOperator::Equal => BinaryOp::Eq,
            ast::BinaryOperator::NotEqual => BinaryOp::Ne,
            ast::BinaryOperator::LessThan => BinaryOp::Lt,
            ast::BinaryOperator::LessOrEqual => BinaryOp::Le,
            ast::BinaryOperator::GreaterThan => BinaryOp::Gt,
            ast::BinaryOperator::GreaterOrEqual => BinaryOp::Ge,
            ast::BinaryOperator::Add => BinaryOp::Add,
            ast::BinaryOperator::Subtract => BinaryOp::Sub,
            ast::BinaryOperator::Multiply => BinaryOp::Mul,
            ast::BinaryOperator::Divide => BinaryOp::Div,
        }
    }

    fn translate_unary_op(&self, op: ast::UnaryOperator) -> UnaryOp {
        match op {
            ast::UnaryOperator::Not => UnaryOp::Not,
            ast::UnaryOperator::Plus => {
                unreachable!("unary plus is handled as a no-op at the call site")
            }
            ast::UnaryOperator::Minus => UnaryOp::Neg,
        }
    }

    fn translate_function_name(&self, func: &ast::FunctionName) -> String {
        match func {
            ast::FunctionName::BuiltIn(builtin) => format!("{:?}", builtin).to_uppercase(),
            ast::FunctionName::Custom(iri) => self.resolve_iri(iri),
        }
    }

    fn translate_group_condition(
        &mut self,
        cond: &ast::GroupCondition,
    ) -> Result<LogicalExpression> {
        match cond {
            ast::GroupCondition::Variable(name) => Ok(LogicalExpression::Variable(name.clone())),
            ast::GroupCondition::Expression { expression, .. } => {
                self.translate_expression(expression)
            }
            ast::GroupCondition::BuiltInCall(expr) => self.translate_expression(expr),
        }
    }

    fn extract_aggregates_for_select(
        &mut self,
        select: &ast::SelectQuery,
    ) -> Result<(Vec<AggregateExpr>, Vec<LogicalExpression>)> {
        let mut aggregates = Vec::new();
        let mut group_by = Vec::new();

        if let ast::Projection::Variables(vars) = &select.projection {
            for pv in vars {
                if self.is_aggregate_expression(&pv.expression) {
                    if let Some(agg) = self.extract_aggregate(&pv.expression, &pv.alias)? {
                        aggregates.push(agg);
                    }
                } else {
                    group_by.push(self.translate_expression(&pv.expression)?);
                }
            }
        }

        Ok((aggregates, group_by))
    }

    fn is_aggregate_expression(&self, expr: &ast::Expression) -> bool {
        matches!(expr, ast::Expression::Aggregate(_))
    }

    /// Recursively checks if an expression contains any aggregate function.
    fn contains_aggregate(expr: &ast::Expression) -> bool {
        match expr {
            ast::Expression::Aggregate(_) => true,
            ast::Expression::Binary { left, right, .. } => {
                Self::contains_aggregate(left) || Self::contains_aggregate(right)
            }
            ast::Expression::Unary { operand, .. } => Self::contains_aggregate(operand),
            ast::Expression::FunctionCall { arguments, .. } => {
                arguments.iter().any(Self::contains_aggregate)
            }
            ast::Expression::Bracketed(inner) => Self::contains_aggregate(inner),
            ast::Expression::Conditional {
                condition,
                then_expression,
                else_expression,
            } => {
                Self::contains_aggregate(condition)
                    || Self::contains_aggregate(then_expression)
                    || Self::contains_aggregate(else_expression)
            }
            ast::Expression::Coalesce(exprs) => exprs.iter().any(Self::contains_aggregate),
            ast::Expression::In { expression, list }
            | ast::Expression::NotIn { expression, list } => {
                Self::contains_aggregate(expression) || list.iter().any(Self::contains_aggregate)
            }
            _ => false,
        }
    }

    /// Checks if the SELECT projection contains any aggregate expressions.
    fn has_aggregates_in_projection(projection: &ast::Projection) -> bool {
        match projection {
            ast::Projection::Wildcard => false,
            ast::Projection::Variables(vars) => vars
                .iter()
                .any(|pv| Self::contains_aggregate(&pv.expression)),
        }
    }

    fn extract_aggregate(
        &mut self,
        expr: &ast::Expression,
        alias: &Option<String>,
    ) -> Result<Option<AggregateExpr>> {
        if let ast::Expression::Aggregate(agg) = expr {
            let (func, expr_inner, distinct, separator) = match agg {
                ast::AggregateExpression::Count {
                    distinct,
                    expression,
                } => {
                    // COUNT(?expr) uses CountNonNull to skip NULLs;
                    // COUNT(*) (no expression) uses Count to count all rows.
                    let func = if expression.is_some() {
                        AggregateFunction::CountNonNull
                    } else {
                        AggregateFunction::Count
                    };
                    (
                        func,
                        expression.as_ref().map(|e| e.as_ref()),
                        *distinct,
                        None,
                    )
                }
                ast::AggregateExpression::Sum {
                    distinct,
                    expression,
                } => (
                    AggregateFunction::Sum,
                    Some(expression.as_ref()),
                    *distinct,
                    None,
                ),
                ast::AggregateExpression::Average {
                    distinct,
                    expression,
                } => (
                    AggregateFunction::Avg,
                    Some(expression.as_ref()),
                    *distinct,
                    None,
                ),
                ast::AggregateExpression::Minimum { expression } => (
                    AggregateFunction::Min,
                    Some(expression.as_ref()),
                    false,
                    None,
                ),
                ast::AggregateExpression::Maximum { expression } => (
                    AggregateFunction::Max,
                    Some(expression.as_ref()),
                    false,
                    None,
                ),
                ast::AggregateExpression::Sample { expression } => (
                    AggregateFunction::Sample,
                    Some(expression.as_ref()),
                    false,
                    None,
                ),
                ast::AggregateExpression::GroupConcat {
                    distinct,
                    expression,
                    separator,
                } => (
                    AggregateFunction::GroupConcat,
                    Some(expression.as_ref()),
                    *distinct,
                    separator.clone(),
                ),
            };

            let expression = if let Some(e) = expr_inner {
                Some(self.translate_expression(e)?)
            } else {
                None
            };

            Ok(Some(AggregateExpr {
                function: func,
                expression,
                expression2: None,
                distinct,
                alias: alias.clone(),
                percentile: None, // SPARQL doesn't support percentile functions
                separator,
            }))
        } else {
            Ok(None)
        }
    }

    /// Rewrites aggregate function calls in a HAVING expression as variable
    /// references to the already-computed aggregate column aliases.
    fn rewrite_aggregates_as_refs(
        expr: &LogicalExpression,
        aggregates: &[AggregateExpr],
    ) -> LogicalExpression {
        match expr {
            LogicalExpression::FunctionCall { name, .. } => {
                // Match aggregate function names to their aliases
                let upper = name.to_uppercase();
                if matches!(
                    upper.as_str(),
                    "COUNT" | "SUM" | "AVG" | "MIN" | "MAX" | "SAMPLE" | "GROUP_CONCAT"
                ) {
                    // Find the matching aggregate by function name.
                    // CountNonNull is the physical variant of COUNT(expr),
                    // so treat it as matching "COUNT" for HAVING rewriting.
                    for agg in aggregates {
                        let agg_name = format!("{:?}", agg.function).to_uppercase();
                        let matches_name =
                            agg_name == upper || (upper == "COUNT" && agg_name == "COUNTNONNULL");
                        if matches_name && agg.alias.is_some() {
                            return LogicalExpression::Variable(
                                agg.alias.clone().expect("alias checked by is_some guard"),
                            );
                        }
                    }
                }
                expr.clone()
            }
            LogicalExpression::Binary { left, op, right } => LogicalExpression::Binary {
                left: Box::new(Self::rewrite_aggregates_as_refs(left, aggregates)),
                op: *op,
                right: Box::new(Self::rewrite_aggregates_as_refs(right, aggregates)),
            },
            LogicalExpression::Unary { op, operand } => LogicalExpression::Unary {
                op: *op,
                operand: Box::new(Self::rewrite_aggregates_as_refs(operand, aggregates)),
            },
            _ => expr.clone(),
        }
    }

    fn join_patterns(&self, left: LogicalOperator, right: LogicalOperator) -> LogicalOperator {
        if matches!(left, LogicalOperator::Empty) {
            return right;
        }
        if matches!(right, LogicalOperator::Empty) {
            return left;
        }

        // Collect output variables from each side and build explicit join
        // conditions for shared variables. This enables the DPccp optimizer
        // to reorder SPARQL triple patterns by selectivity.
        let left_vars = Self::collect_operator_variables(&left);
        let right_vars = Self::collect_operator_variables(&right);

        let mut conditions = Vec::new();
        for var in &left_vars {
            if right_vars.contains(var) {
                conditions.push(JoinCondition {
                    left: LogicalExpression::Variable(var.clone()),
                    right: LogicalExpression::Variable(var.clone()),
                });
            }
        }

        LogicalOperator::Join(JoinOp {
            left: Box::new(left),
            right: Box::new(right),
            join_type: JoinType::Inner,
            conditions,
        })
    }

    /// Collects all variable names produced by an operator subtree.
    ///
    /// Walks `TripleScan`, `Join`, `LeftJoin`, `Union`, `Filter`, `Bind`,
    /// and `Project` operators to gather the set of variable names that
    /// appear in the output.
    fn collect_operator_variables(op: &LogicalOperator) -> Vec<String> {
        let mut vars = Vec::new();
        Self::collect_variables_recursive(op, &mut vars);
        vars
    }

    fn collect_variables_recursive(op: &LogicalOperator, vars: &mut Vec<String>) {
        match op {
            LogicalOperator::TripleScan(scan) => {
                if let TripleComponent::Variable(v) = &scan.subject
                    && !vars.contains(v)
                {
                    vars.push(v.clone());
                }
                if let TripleComponent::Variable(v) = &scan.predicate
                    && !vars.contains(v)
                {
                    vars.push(v.clone());
                }
                if let TripleComponent::Variable(v) = &scan.object
                    && !vars.contains(v)
                {
                    vars.push(v.clone());
                }
                if let Some(TripleComponent::Variable(v)) = &scan.graph
                    && !vars.contains(v)
                {
                    vars.push(v.clone());
                }
                if let Some(input) = &scan.input {
                    Self::collect_variables_recursive(input, vars);
                }
            }
            LogicalOperator::Join(join) => {
                Self::collect_variables_recursive(&join.left, vars);
                Self::collect_variables_recursive(&join.right, vars);
            }
            LogicalOperator::LeftJoin(join) => {
                Self::collect_variables_recursive(&join.left, vars);
                Self::collect_variables_recursive(&join.right, vars);
            }
            LogicalOperator::Union(union) => {
                // Union outputs variables from both branches
                for input in &union.inputs {
                    Self::collect_variables_recursive(input, vars);
                }
            }
            LogicalOperator::Filter(filter) => {
                Self::collect_variables_recursive(&filter.input, vars);
            }
            LogicalOperator::Bind(bind) => {
                Self::collect_variables_recursive(&bind.input, vars);
                if !vars.contains(&bind.variable) {
                    vars.push(bind.variable.clone());
                }
            }
            LogicalOperator::Project(proj) => {
                Self::collect_variables_recursive(&proj.input, vars);
            }
            _ => {}
        }
    }

    fn resolve_iri(&self, iri: &ast::Iri) -> String {
        let iri_str = iri.as_str();

        // Check if it's a prefixed name
        if let Some(colon_pos) = iri_str.find(':') {
            let prefix = &iri_str[..colon_pos];
            let local = &iri_str[colon_pos + 1..];

            if let Some(namespace) = self.prefixes.get(prefix) {
                return format!("{}{}", namespace, local);
            }
        }

        // Return as-is if no prefix match or already a full IRI
        iri_str.to_string()
    }

    /// Rewrites `?var = "value"@lang` to `?var = "value" AND LANG(?var) = "lang"`,
    /// and `?var != "value"@lang` to `?var != "value" OR LANG(?var) != "lang"`.
    ///
    /// Returns `None` when neither side is a language-tagged literal (the caller
    /// falls through to the normal binary translation path).
    fn try_expand_lang_comparison(
        &mut self,
        left: &ast::Expression,
        operator: ast::BinaryOperator,
        right: &ast::Expression,
    ) -> Result<Option<LogicalExpression>> {
        // Detect which side is the language-tagged literal and which is a variable.
        let (var_expr, lang_lit) = match (left, right) {
            (_, ast::Expression::Literal(lit)) if lit.language.is_some() => (left, lit),
            (ast::Expression::Literal(lit), _) if lit.language.is_some() => (right, lit),
            _ => return Ok(None),
        };

        let lang_tag = lang_lit.language.as_ref().expect("checked above");
        let translated_var = self.translate_expression(var_expr)?;
        let value_literal =
            LogicalExpression::Literal(Value::String(lang_lit.value.clone().into()));
        let lang_literal = LogicalExpression::Literal(Value::String(lang_tag.clone().into()));

        // Build LANG(?var)
        let lang_call = LogicalExpression::FunctionCall {
            name: "LANG".to_string(),
            args: vec![translated_var.clone()],
            distinct: false,
        };

        let (value_op, lang_op, combine_op) = if operator == ast::BinaryOperator::Equal {
            (BinaryOp::Eq, BinaryOp::Eq, BinaryOp::And)
        } else {
            // NotEqual: either value differs OR lang tag differs
            (BinaryOp::Ne, BinaryOp::Ne, BinaryOp::Or)
        };

        let value_cmp = LogicalExpression::Binary {
            left: Box::new(translated_var),
            op: value_op,
            right: Box::new(value_literal),
        };
        let lang_cmp = LogicalExpression::Binary {
            left: Box::new(lang_call),
            op: lang_op,
            right: Box::new(lang_literal),
        };

        Ok(Some(LogicalExpression::Binary {
            left: Box::new(value_cmp),
            op: combine_op,
            right: Box::new(lang_cmp),
        }))
    }

    fn literal_to_value(&self, lit: &ast::Literal) -> Value {
        // Check for typed literals
        if let Some(datatype) = &lit.datatype {
            let dt = self.resolve_iri(datatype);
            match dt.as_str() {
                "http://www.w3.org/2001/XMLSchema#integer"
                | "http://www.w3.org/2001/XMLSchema#int"
                | "http://www.w3.org/2001/XMLSchema#long" => {
                    if let Ok(n) = lit.value.parse::<i64>() {
                        return Value::Int64(n);
                    }
                }
                "http://www.w3.org/2001/XMLSchema#decimal"
                | "http://www.w3.org/2001/XMLSchema#double"
                | "http://www.w3.org/2001/XMLSchema#float" => {
                    if let Ok(n) = lit.value.parse::<f64>() {
                        return Value::Float64(n);
                    }
                }
                "http://www.w3.org/2001/XMLSchema#boolean" => {
                    return Value::Bool(lit.value == "true" || lit.value == "1");
                }
                "http://www.w3.org/2001/XMLSchema#date" => {
                    if let Some(d) = grafeo_common::types::Date::parse(&lit.value) {
                        return Value::Date(d);
                    }
                }
                "http://www.w3.org/2001/XMLSchema#time" => {
                    if let Some(t) = grafeo_common::types::Time::parse(&lit.value) {
                        return Value::Time(t);
                    }
                }
                "http://www.w3.org/2001/XMLSchema#duration"
                | "http://www.w3.org/2001/XMLSchema#dayTimeDuration"
                | "http://www.w3.org/2001/XMLSchema#yearMonthDuration" => {
                    if let Some(d) = grafeo_common::types::Duration::parse(&lit.value) {
                        return Value::Duration(d);
                    }
                }
                "http://www.w3.org/2001/XMLSchema#dateTime" => {
                    // Prefer ZonedDatetime when the value has an explicit offset,
                    // so that local date/time and timezone are preserved for
                    // YEAR/MONTH/DAY/HOURS/MINUTES/SECONDS/TIMEZONE/TZ functions.
                    if let Some(zdt) = grafeo_common::types::ZonedDatetime::parse(&lit.value) {
                        return Value::ZonedDatetime(zdt);
                    }
                    // Fall back to Timestamp for values without offset
                    if let Some(pos) = lit.value.find('T')
                        && let (Some(d), Some(t)) = (
                            grafeo_common::types::Date::parse(&lit.value[..pos]),
                            grafeo_common::types::Time::parse(&lit.value[pos + 1..]),
                        )
                    {
                        return Value::Timestamp(grafeo_common::types::Timestamp::from_date_time(
                            d, t,
                        ));
                    }
                }
                _ => {}
            }
        }

        // Default to string
        Value::String(lit.value.clone().into())
    }

    /// Converts a SPARQL `DataValue` (from VALUES inline data) to a `Value`.
    fn data_value_to_value(&self, dv: &ast::DataValue) -> Value {
        match dv {
            ast::DataValue::Iri(iri) => Value::String(self.resolve_iri(iri).into()),
            ast::DataValue::Literal(lit) => self.literal_to_value(lit),
        }
    }

    /// Translates a negated property set `!(iri1|^iri2)`.
    ///
    /// For forward IRIs: scans `?s ?p ?o` and filters out excluded predicates.
    /// For inverse IRIs: scans `?o ?p ?s` (swapped) and filters out excluded predicates.
    /// Mixed sets produce a `Union` of forward and inverse branches.
    fn translate_negated_property_set(
        &mut self,
        triple: &ast::TriplePattern,
        negated_iris: &[ast::NegatedIri],
    ) -> Result<LogicalOperator> {
        let subject = self.translate_triple_term(&triple.subject)?;
        let object = self.translate_triple_term(&triple.object)?;
        let graph = self.graph_context_stack.last().cloned();

        let forward_iris: Vec<&ast::Iri> = negated_iris
            .iter()
            .filter(|ni| !ni.inverse)
            .map(|ni| &ni.iri)
            .collect();
        let inverse_iris: Vec<&ast::Iri> = negated_iris
            .iter()
            .filter(|ni| ni.inverse)
            .map(|ni| &ni.iri)
            .collect();

        let has_forward = !forward_iris.is_empty() || inverse_iris.is_empty();
        let has_inverse = !inverse_iris.is_empty();

        let build_branch = |translator: &mut Self,
                            subj: TripleComponent,
                            obj: TripleComponent,
                            excluded: &[&ast::Iri]|
         -> Result<LogicalOperator> {
            let pred_var = format!("_:neg_pred{}", translator.next_anon());
            let scan = translator.make_triple_scan(
                subj,
                TripleComponent::Variable(pred_var.clone()),
                obj,
                graph.clone(),
            );

            if excluded.is_empty() {
                return Ok(scan);
            }

            // Build filter: _:neg_pred != iri1 AND _:neg_pred != iri2 AND ...
            let conditions: Vec<LogicalExpression> = excluded
                .iter()
                .map(|iri| LogicalExpression::Binary {
                    left: Box::new(LogicalExpression::Variable(pred_var.clone())),
                    op: BinaryOp::Ne,
                    right: Box::new(LogicalExpression::Literal(Value::String(
                        translator.resolve_iri(iri).into(),
                    ))),
                })
                .collect();

            let predicate = conditions
                .into_iter()
                .reduce(|left, right| LogicalExpression::Binary {
                    left: Box::new(left),
                    op: BinaryOp::And,
                    right: Box::new(right),
                })
                .expect("excluded non-empty after is_empty check");

            Ok(wrap_filter(scan, predicate))
        };

        if has_forward && has_inverse {
            // Union of forward scan (excluding forward IRIs) and
            // inverse scan with swapped s/o (excluding inverse IRIs)
            let forward_branch =
                build_branch(self, subject.clone(), object.clone(), &forward_iris)?;
            let inverse_branch = build_branch(self, object, subject, &inverse_iris)?;
            Ok(LogicalOperator::Union(UnionOp {
                inputs: vec![forward_branch, inverse_branch],
            }))
        } else if has_inverse {
            // Only inverse exclusions: scan with swapped subject/object
            build_branch(self, object, subject, &inverse_iris)
        } else {
            // Only forward exclusions (most common case)
            build_branch(self, subject, object, &forward_iris)
        }
    }

    /// Translates a `OneOrMore` property path (`path+`) using bounded expansion.
    ///
    /// Expands to a `Union` of sequences from depth 1 to `MAX_DEPTH`, wrapped
    /// in `Distinct` to deduplicate rows that appear at multiple depths.
    fn translate_one_or_more_path(
        &mut self,
        triple: &ast::TriplePattern,
        inner_path: &ast::PropertyPath,
    ) -> Result<LogicalOperator> {
        const MAX_DEPTH: usize = 50;

        let subject = self.translate_triple_term(&triple.subject)?;
        let object = self.translate_triple_term(&triple.object)?;
        let graph = self.graph_context_stack.last().cloned();

        let mut branches = Vec::new();

        for depth in 1..=MAX_DEPTH {
            let branch =
                self.translate_fixed_depth_path(inner_path, &subject, &object, &graph, depth)?;
            branches.push(branch);
        }

        let union = LogicalOperator::Union(UnionOp { inputs: branches });

        // Wrap in Distinct to deduplicate across depths
        Ok(wrap_distinct(union))
    }

    /// Translates a `ZeroOrMore` property path (`path*`) using bounded expansion.
    ///
    /// Includes reflexive (0-hop) matches for every node that participates as
    /// subject or object of the predicate, plus the same 1..`MAX_DEPTH` expansion
    /// used by `OneOrMore`.
    fn translate_zero_or_more_path(
        &mut self,
        triple: &ast::TriplePattern,
        inner_path: &ast::PropertyPath,
    ) -> Result<LogicalOperator> {
        const MAX_DEPTH: usize = 50;

        let subject = self.translate_triple_term(&triple.subject)?;
        let object = self.translate_triple_term(&triple.object)?;
        let graph = self.graph_context_stack.last().cloned();

        let mut branches = Vec::new();

        // 0-hop reflexive branches
        self.add_reflexive_branches(&subject, &object, inner_path, &graph, &mut branches)?;

        // 1+ hops: same as OneOrMore (wrapped in projection to match reflexive column count)
        for depth in 1..=MAX_DEPTH {
            let branch =
                self.translate_fixed_depth_path(inner_path, &subject, &object, &graph, depth)?;
            branches.push(self.project_path_endpoints(&subject, &object, branch));
        }

        let union = LogicalOperator::Union(UnionOp { inputs: branches });
        Ok(wrap_distinct(union))
    }

    /// Translates a `ZeroOrOne` property path (`path?`).
    ///
    /// Produces a union of 0-hop reflexive matches and exactly 1-hop matches,
    /// then deduplicates. Same structure as `translate_zero_or_more_path` but
    /// bounded to depth 0..1 instead of 0..MAX_DEPTH.
    fn translate_zero_or_one_path(
        &mut self,
        triple: &ast::TriplePattern,
        inner_path: &ast::PropertyPath,
    ) -> Result<LogicalOperator> {
        let subject = self.translate_triple_term(&triple.subject)?;
        let object = self.translate_triple_term(&triple.object)?;
        let graph = self.graph_context_stack.last().cloned();

        let mut branches = Vec::new();

        // 0-hop reflexive branches
        self.add_reflexive_branches(&subject, &object, inner_path, &graph, &mut branches)?;

        // 1-hop: exactly one traversal of the predicate (wrapped to match reflexive column count)
        let one_hop = self.translate_fixed_depth_path(inner_path, &subject, &object, &graph, 1)?;
        branches.push(self.project_path_endpoints(&subject, &object, one_hop));

        let union = LogicalOperator::Union(UnionOp { inputs: branches });
        Ok(wrap_distinct(union))
    }

    /// Adds 0-hop reflexive branches for `ZeroOrMore` and `ZeroOrOne` paths.
    fn add_reflexive_branches(
        &mut self,
        subject: &TripleComponent,
        object: &TripleComponent,
        inner_path: &ast::PropertyPath,
        graph: &Option<TripleComponent>,
        branches: &mut Vec<LogicalOperator>,
    ) -> Result<()> {
        if matches!(subject, TripleComponent::Variable(_)) {
            let fresh_obj = TripleComponent::Variable(format!("_:refl{}", self.next_anon()));
            let pred = self.translate_property_path(inner_path)?;
            let subj_scan = self.make_triple_scan(subject.clone(), pred, fresh_obj, graph.clone());
            let subj_reflexive = self.project_reflexive(subject, object, subj_scan)?;
            branches.push(subj_reflexive);

            let fresh_subj = TripleComponent::Variable(format!("_:refl{}", self.next_anon()));
            let pred2 = self.translate_property_path(inner_path)?;
            let obj_scan = self.make_triple_scan(fresh_subj, pred2, object.clone(), graph.clone());
            let obj_reflexive = self.project_reflexive_from_object(subject, object, obj_scan)?;
            branches.push(obj_reflexive);
        } else if let TripleComponent::Variable(obj_var) = object {
            let subj_expr = self.triple_component_to_expression(subject);
            let reflexive = LogicalOperator::Bind(BindOp {
                expression: subj_expr,
                variable: obj_var.clone(),
                input: Box::new(LogicalOperator::Empty),
            });
            branches.push(reflexive);
        }
        Ok(())
    }

    /// Wraps a depth-branch operator in a projection that outputs only the
    /// subject and object variables, stripping companion columns so that all
    /// union branches have consistent column counts for DISTINCT.
    fn project_path_endpoints(
        &self,
        subject: &TripleComponent,
        object: &TripleComponent,
        input: LogicalOperator,
    ) -> LogicalOperator {
        let mut projections = Vec::new();
        if let TripleComponent::Variable(s) = subject {
            projections.push(Projection {
                expression: LogicalExpression::Variable(s.clone()),
                alias: Some(s.clone()),
            });
        }
        if let TripleComponent::Variable(o) = object {
            projections.push(Projection {
                expression: LogicalExpression::Variable(o.clone()),
                alias: Some(o.clone()),
            });
        }
        if projections.is_empty() {
            return input;
        }
        LogicalOperator::Project(ProjectOp {
            projections,
            input: Box::new(input),
            pass_through_input: false,
        })
    }

    /// Translates a property path at a fixed depth (number of hops).
    ///
    /// Depth 1 produces a single `TripleScan`. Depth N chains N scans with
    /// freshly generated intermediate variables joined together.
    fn translate_fixed_depth_path(
        &mut self,
        path: &ast::PropertyPath,
        subject: &TripleComponent,
        object: &TripleComponent,
        graph: &Option<TripleComponent>,
        depth: usize,
    ) -> Result<LogicalOperator> {
        if depth == 1 {
            let predicate = self.translate_property_path(path)?;
            return Ok(self.make_triple_scan(
                subject.clone(),
                predicate,
                object.clone(),
                graph.clone(),
            ));
        }

        // Multiple hops: chain triple scans with intermediate variables
        let mut current_subject = subject.clone();
        let mut plan = LogicalOperator::Empty;
        let mut first = true;

        for i in 0..depth {
            let next_object = if i == depth - 1 {
                object.clone()
            } else {
                TripleComponent::Variable(format!("_:path{}", self.next_anon()))
            };

            let predicate = self.translate_property_path(path)?;
            let scan = self.make_triple_scan(
                current_subject,
                predicate,
                next_object.clone(),
                graph.clone(),
            );

            if first {
                plan = scan;
                first = false;
            } else {
                plan = self.join_patterns(plan, scan);
            }

            current_subject = next_object;
        }

        Ok(plan)
    }

    /// Projects a scan so that the subject value appears as both the subject
    /// and object output variables, producing reflexive (0-hop) rows.
    fn project_reflexive(
        &self,
        subject: &TripleComponent,
        object: &TripleComponent,
        input: LogicalOperator,
    ) -> Result<LogicalOperator> {
        let subj_expr = self.triple_component_to_expression(subject);
        let obj_var = match object {
            TripleComponent::Variable(v) => v.clone(),
            _ => return Ok(input),
        };
        let mut projections = Vec::new();
        // Keep subject variable in output if it is a variable
        if let TripleComponent::Variable(s_var) = subject {
            projections.push(Projection {
                expression: LogicalExpression::Variable(s_var.clone()),
                alias: Some(s_var.clone()),
            });
        }
        projections.push(Projection {
            expression: subj_expr,
            alias: Some(obj_var),
        });
        Ok(LogicalOperator::Project(ProjectOp {
            projections,
            input: Box::new(input),
            pass_through_input: false,
        }))
    }

    /// Projects a scan so that the object value appears as both the subject
    /// and object output variables, producing reflexive (0-hop) rows.
    fn project_reflexive_from_object(
        &self,
        subject: &TripleComponent,
        object: &TripleComponent,
        input: LogicalOperator,
    ) -> Result<LogicalOperator> {
        let obj_expr = self.triple_component_to_expression(object);
        let subj_var = match subject {
            TripleComponent::Variable(v) => v.clone(),
            _ => return Ok(input),
        };
        let mut projections = vec![Projection {
            expression: obj_expr,
            alias: Some(subj_var),
        }];
        // Keep object variable in output if it is a variable
        if let TripleComponent::Variable(o_var) = object {
            projections.push(Projection {
                expression: LogicalExpression::Variable(o_var.clone()),
                alias: Some(o_var.clone()),
            });
        }
        Ok(LogicalOperator::Project(ProjectOp {
            projections,
            input: Box::new(input),
            pass_through_input: false,
        }))
    }

    /// Converts a `TripleComponent` to a `LogicalExpression` for use in projections.
    fn triple_component_to_expression(&self, component: &TripleComponent) -> LogicalExpression {
        match component {
            TripleComponent::Variable(name) => LogicalExpression::Variable(name.clone()),
            TripleComponent::Iri(iri) => {
                LogicalExpression::Literal(Value::String(iri.clone().into()))
            }
            TripleComponent::Literal(val) => LogicalExpression::Literal(val.clone()),
            TripleComponent::LangLiteral { value, .. } => {
                LogicalExpression::Literal(Value::String(value.clone().into()))
            }
            TripleComponent::BlankNode(label) => {
                LogicalExpression::Literal(Value::String(format!("_:{label}").into()))
            }
        }
    }

    /// Converts a parsed `DatasetClause` into a `DatasetRestriction` for the logical plan.
    fn translate_dataset_clause(
        &self,
        dataset: &Option<ast::DatasetClause>,
    ) -> Option<DatasetRestriction> {
        let clause = dataset.as_ref()?;

        let default_graphs: Vec<String> = clause
            .default_graphs
            .iter()
            .map(|iri| self.resolve_iri(iri))
            .collect();
        let named_graphs: Vec<String> = clause
            .named_graphs
            .iter()
            .map(|iri| self.resolve_iri(iri))
            .collect();

        // Only create a restriction if at least one FROM or FROM NAMED is present
        if default_graphs.is_empty() && named_graphs.is_empty() {
            return None;
        }

        Some(DatasetRestriction {
            default_graphs,
            named_graphs,
        })
    }

    /// Creates a `TripleScanOp` with the current graph context and dataset restriction.
    fn make_triple_scan(
        &self,
        subject: TripleComponent,
        predicate: TripleComponent,
        object: TripleComponent,
        graph: Option<TripleComponent>,
    ) -> LogicalOperator {
        LogicalOperator::TripleScan(TripleScanOp {
            subject,
            predicate,
            object,
            graph,
            input: None,
            dataset: self.dataset.clone(),
        })
    }

    fn next_anon(&mut self) -> u32 {
        let n = self.anon_counter;
        self.anon_counter += 1;
        n
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::query::plan::{LimitOp, SkipOp, SortOp};

    // === Basic SELECT Tests ===

    #[test]
    fn test_translate_simple_select() {
        let query = "SELECT ?x WHERE { ?x ?y ?z }";
        let result = translate(query);
        assert!(result.is_ok());
    }

    #[test]
    fn test_translate_select_with_prefix() {
        let query = r#"
            PREFIX foaf: <http://xmlns.com/foaf/0.1/>
            SELECT ?name
            WHERE { ?x foaf:name ?name }
        "#;
        let result = translate(query);
        assert!(result.is_ok());
    }

    #[test]
    fn test_translate_select_wildcard() {
        let query = "SELECT * WHERE { ?x ?y ?z }";
        let result = translate(query);
        assert!(result.is_ok());
    }

    #[test]
    fn test_translate_select_distinct() {
        let query = "SELECT DISTINCT ?x WHERE { ?x ?y ?z }";
        let result = translate(query);
        assert!(result.is_ok());

        let plan = result.unwrap();
        fn find_distinct(op: &LogicalOperator) -> bool {
            match op {
                LogicalOperator::Distinct(_) => true,
                LogicalOperator::Project(p) => find_distinct(&p.input),
                _ => false,
            }
        }
        assert!(find_distinct(&plan.root));
    }

    // === Filter Tests ===

    #[test]
    fn test_translate_select_with_filter() {
        let query = "SELECT ?x WHERE { ?x ?y ?z FILTER(?z > 10) }";
        let result = translate(query);
        assert!(result.is_ok());
    }

    #[test]
    fn test_translate_filter_equality() {
        let query = r#"SELECT ?x WHERE { ?x ?y ?z FILTER(?z = "test") }"#;
        let result = translate(query);
        assert!(result.is_ok());
    }

    #[test]
    fn test_translate_filter_and() {
        let query = "SELECT ?x WHERE { ?x ?y ?z FILTER(?z > 10 && ?z < 100) }";
        let result = translate(query);
        assert!(result.is_ok());
    }

    #[test]
    fn test_translate_filter_or() {
        let query = r#"SELECT ?x WHERE { ?x ?y ?z FILTER(?z = 1 || ?z = 2) }"#;
        let result = translate(query);
        assert!(result.is_ok());
    }

    #[test]
    fn test_translate_filter_bound() {
        let query = "SELECT ?x WHERE { ?x ?y ?z FILTER(BOUND(?z)) }";
        let result = translate(query);
        assert!(result.is_ok());
    }

    // === ASK Query Tests ===

    #[test]
    fn test_translate_ask() {
        let query = "ASK { ?x ?y ?z }";
        let result = translate(query);
        assert!(result.is_ok());

        let plan = result.unwrap();
        // ASK should have a Limit(1)
        fn find_limit(op: &LogicalOperator) -> Option<&LimitOp> {
            match op {
                LogicalOperator::Limit(l) => Some(l),
                _ => None,
            }
        }
        let limit = find_limit(&plan.root).expect("Expected Limit");
        assert_eq!(limit.count, 1);
    }

    // === Solution Modifiers Tests ===

    #[test]
    fn test_translate_select_with_limit() {
        let query = "SELECT ?x WHERE { ?x ?y ?z } LIMIT 10";
        let result = translate(query);
        assert!(result.is_ok());
    }

    #[test]
    fn test_translate_select_with_offset() {
        let query = "SELECT ?x WHERE { ?x ?y ?z } OFFSET 5";
        let result = translate(query);
        assert!(result.is_ok());

        let plan = result.unwrap();
        fn find_skip(op: &LogicalOperator) -> Option<&SkipOp> {
            match op {
                LogicalOperator::Skip(s) => Some(s),
                LogicalOperator::Project(p) => find_skip(&p.input),
                _ => None,
            }
        }
        let skip = find_skip(&plan.root).expect("Expected Skip");
        assert_eq!(skip.count, 5);
    }

    #[test]
    fn test_translate_select_with_order_by() {
        let query = "SELECT ?x WHERE { ?x ?y ?z } ORDER BY ?z";
        let result = translate(query);
        assert!(result.is_ok());

        let plan = result.unwrap();
        fn find_sort(op: &LogicalOperator) -> Option<&SortOp> {
            match op {
                LogicalOperator::Sort(s) => Some(s),
                LogicalOperator::Project(p) => find_sort(&p.input),
                _ => None,
            }
        }
        assert!(find_sort(&plan.root).is_some());
    }

    #[test]
    fn test_translate_select_with_order_by_desc() {
        let query = "SELECT ?x WHERE { ?x ?y ?z } ORDER BY DESC(?z)";
        let result = translate(query);
        assert!(result.is_ok());

        let plan = result.unwrap();
        fn find_sort(op: &LogicalOperator) -> Option<&SortOp> {
            match op {
                LogicalOperator::Sort(s) => Some(s),
                LogicalOperator::Project(p) => find_sort(&p.input),
                _ => None,
            }
        }
        let sort = find_sort(&plan.root).expect("Expected Sort");
        assert_eq!(sort.keys[0].order, SortOrder::Descending);
    }

    // === Graph Pattern Tests ===

    #[test]
    fn test_translate_union() {
        let query = "SELECT ?x WHERE { { ?x ?y ?z } UNION { ?x ?a ?b } }";
        let result = translate(query);
        assert!(result.is_ok());

        let plan = result.unwrap();
        fn find_union(op: &LogicalOperator) -> bool {
            match op {
                LogicalOperator::Union(_) => true,
                LogicalOperator::Project(p) => find_union(&p.input),
                _ => false,
            }
        }
        assert!(find_union(&plan.root));
    }

    #[test]
    fn test_translate_optional() {
        let query = "SELECT ?x ?name WHERE { ?x ?y ?z OPTIONAL { ?x ?p ?name } }";
        let result = translate(query);
        assert!(result.is_ok());
    }

    #[test]
    fn test_translate_bind() {
        let query = "SELECT ?x ?doubled WHERE { ?x ?y ?z BIND(?z * 2 AS ?doubled) }";
        let result = translate(query);
        assert!(result.is_ok());
    }

    // === Aggregate Tests ===

    #[test]
    fn test_translate_count() {
        let query = "SELECT (COUNT(?x) AS ?cnt) WHERE { ?x ?y ?z }";
        let result = translate(query);
        assert!(result.is_ok());
    }

    #[test]
    fn test_translate_group_by() {
        let query = "SELECT ?y (COUNT(?x) AS ?cnt) WHERE { ?x ?y ?z } GROUP BY ?y";
        let result = translate(query);
        assert!(result.is_ok());

        let plan = result.unwrap();
        fn find_aggregate(op: &LogicalOperator) -> Option<&AggregateOp> {
            match op {
                LogicalOperator::Aggregate(a) => Some(a),
                LogicalOperator::Project(p) => find_aggregate(&p.input),
                _ => None,
            }
        }
        let agg = find_aggregate(&plan.root).expect("Expected Aggregate");
        assert!(!agg.group_by.is_empty());
    }

    // === Expression Tests ===

    #[test]
    fn test_translate_arithmetic_expression() {
        let query = "SELECT (?x + ?y AS ?sum) WHERE { ?x ?p ?y }";
        let result = translate(query);
        assert!(result.is_ok());
    }

    #[test]
    fn test_translate_string_function() {
        let query = r#"SELECT ?x WHERE { ?x ?y ?z FILTER(CONTAINS(?z, "test")) }"#;
        let result = translate(query);
        assert!(result.is_ok());
    }

    // === CONSTRUCT and DESCRIBE Tests ===

    #[test]
    fn test_translate_construct() {
        let query = "CONSTRUCT { ?x ?y ?z } WHERE { ?x ?y ?z }";
        let result = translate(query);
        assert!(result.is_ok());
    }

    #[test]
    fn test_translate_describe() {
        let query = "DESCRIBE ?x WHERE { ?x ?y ?z }";
        let result = translate(query);
        assert!(result.is_ok());
    }

    // === Multiple Triple Patterns ===

    #[test]
    fn test_translate_multiple_triples() {
        let query = "SELECT ?x ?name ?age WHERE { ?x ?y ?name . ?x ?z ?age }";
        let result = translate(query);
        assert!(result.is_ok());
    }

    // === Literal Types ===

    #[test]
    fn test_translate_literal_types() {
        let query = r#"SELECT ?x WHERE { ?x ?y 42 . ?x ?z "hello" . ?x ?w true }"#;
        let result = translate(query);
        assert!(result.is_ok());
    }

    // === Helper Function Tests ===

    #[test]
    fn test_translator_new() {
        let translator = SparqlTranslator::new();
        assert!(translator.prefixes.is_empty());
        assert!(translator.base.is_none());
        assert_eq!(translator.anon_counter, 0);
    }

    #[test]
    fn test_translator_next_anon() {
        let mut translator = SparqlTranslator::new();
        assert_eq!(translator.next_anon(), 0);
        assert_eq!(translator.next_anon(), 1);
        assert_eq!(translator.next_anon(), 2);
    }

    // === SPARQL Update Tests ===

    #[test]
    fn test_translate_insert_data() {
        let query = r#"INSERT DATA { <http://ex.org/s> <http://ex.org/p> "value" }"#;
        let result = translate(query);
        assert!(result.is_ok());

        let plan = result.unwrap();
        fn has_insert_triple(op: &LogicalOperator) -> bool {
            match op {
                LogicalOperator::InsertTriple(_) => true,
                LogicalOperator::Union(u) => u.inputs.iter().any(has_insert_triple),
                _ => false,
            }
        }
        assert!(has_insert_triple(&plan.root));
    }

    #[test]
    fn test_translate_delete_data() {
        let query = r#"DELETE DATA { <http://ex.org/s> <http://ex.org/p> "value" }"#;
        let result = translate(query);
        assert!(result.is_ok());

        let plan = result.unwrap();
        fn has_delete_triple(op: &LogicalOperator) -> bool {
            match op {
                LogicalOperator::DeleteTriple(_) => true,
                LogicalOperator::Union(u) => u.inputs.iter().any(has_delete_triple),
                _ => false,
            }
        }
        assert!(has_delete_triple(&plan.root));
    }

    #[test]
    fn test_translate_delete_where() {
        let query = r#"DELETE WHERE { ?s <http://ex.org/p> ?o }"#;
        let result = translate(query);
        assert!(result.is_ok());
    }

    #[test]
    fn test_translate_modify_delete_insert() {
        let query = r#"
            DELETE { ?s <http://ex.org/old> ?o }
            INSERT { ?s <http://ex.org/new> ?o }
            WHERE { ?s <http://ex.org/old> ?o }
        "#;
        let result = translate(query);
        assert!(result.is_ok());
    }

    #[test]
    fn test_translate_clear_graph() {
        let query = "CLEAR DEFAULT";
        let result = translate(query);
        assert!(result.is_ok());

        let plan = result.unwrap();
        assert!(matches!(plan.root, LogicalOperator::ClearGraph(_)));
    }

    #[test]
    fn test_translate_drop_graph() {
        let query = "DROP GRAPH <http://example.org/graph>";
        let result = translate(query);
        assert!(result.is_ok());

        let plan = result.unwrap();
        assert!(matches!(plan.root, LogicalOperator::DropGraph(_)));
    }

    #[test]
    fn test_translate_create_graph() {
        let query = "CREATE GRAPH <http://example.org/newgraph>";
        let result = translate(query);
        assert!(result.is_ok());

        let plan = result.unwrap();
        assert!(matches!(plan.root, LogicalOperator::CreateGraph(_)));
    }

    #[test]
    fn test_translate_copy_graph() {
        let query = "COPY DEFAULT TO <http://example.org/backup>";
        let result = translate(query);
        assert!(result.is_ok());

        let plan = result.unwrap();
        assert!(matches!(plan.root, LogicalOperator::CopyGraph(_)));
    }

    #[test]
    fn test_translate_move_graph() {
        let query = "MOVE <http://example.org/old> TO <http://example.org/new>";
        let result = translate(query);
        assert!(result.is_ok());

        let plan = result.unwrap();
        assert!(matches!(plan.root, LogicalOperator::MoveGraph(_)));
    }

    #[test]
    fn test_translate_add_graph() {
        let query = "ADD <http://example.org/source> TO <http://example.org/dest>";
        let result = translate(query);
        assert!(result.is_ok());

        let plan = result.unwrap();
        assert!(matches!(plan.root, LogicalOperator::AddGraph(_)));
    }

    #[test]
    fn test_translate_load_graph() {
        let query = "LOAD <http://example.org/data.ttl>";
        let result = translate(query);
        assert!(result.is_ok());

        let plan = result.unwrap();
        assert!(matches!(plan.root, LogicalOperator::LoadGraph(_)));
    }

    #[test]
    fn test_translate_load_into_graph() {
        let query = "LOAD <http://example.org/data.ttl> INTO GRAPH <http://example.org/target>";
        let result = translate(query);
        assert!(result.is_ok());

        let plan = result.unwrap();
        if let LogicalOperator::LoadGraph(load) = &plan.root {
            assert!(load.destination.is_some());
        } else {
            panic!("Expected LoadGraph operator");
        }
    }

    #[test]
    fn test_translate_silent_operations() {
        let query = "DROP SILENT GRAPH <http://example.org/graph>";
        let result = translate(query);
        assert!(result.is_ok());

        let plan = result.unwrap();
        if let LogicalOperator::DropGraph(drop) = &plan.root {
            assert!(drop.silent);
        } else {
            panic!("Expected DropGraph operator");
        }
    }

    // === BIND Expression Tests ===

    #[test]
    fn test_translate_bind_with_concat() {
        let query = r#"
            PREFIX foaf: <http://xmlns.com/foaf/0.1/>
            SELECT ?name ?label
            WHERE {
                ?person foaf:name ?name .
                BIND (CONCAT(?name, " test") AS ?label)
            }
        "#;
        let result = translate(query);
        assert!(
            result.is_ok(),
            "BIND translation failed: {:?}",
            result.err()
        );
        let plan = result.unwrap();

        fn find_bind(op: &LogicalOperator) -> bool {
            match op {
                LogicalOperator::Bind(_) => true,
                LogicalOperator::Project(p) => find_bind(&p.input),
                LogicalOperator::Filter(f) => find_bind(&f.input),
                _ => false,
            }
        }
        assert!(find_bind(&plan.root), "Expected Bind operator in plan");
    }

    // === VALUES Inline Data Tests ===

    #[test]
    fn test_translate_values_inline_data() {
        let query = r#"
            PREFIX foaf: <http://xmlns.com/foaf/0.1/>
            SELECT ?name
            WHERE {
                VALUES ?person { <http://ex.org/alix> }
                ?person foaf:name ?name .
            }
        "#;
        let result = translate(query);
        assert!(
            result.is_ok(),
            "VALUES translation failed: {:?}",
            result.err()
        );
        let plan = result.unwrap();

        // VALUES with a single value produces a Bind chain joined with the
        // triple pattern. Walk the tree and verify we find at least one Bind.
        fn find_bind(op: &LogicalOperator) -> bool {
            match op {
                LogicalOperator::Bind(_) => true,
                LogicalOperator::Project(p) => find_bind(&p.input),
                LogicalOperator::Filter(f) => find_bind(&f.input),
                LogicalOperator::Join(j) => find_bind(&j.left) || find_bind(&j.right),
                LogicalOperator::Union(u) => u.inputs.iter().any(find_bind),
                _ => false,
            }
        }
        assert!(find_bind(&plan.root), "Expected Bind from VALUES clause");
    }

    // === OneOrMore Property Path Tests ===

    #[test]
    fn test_translate_one_or_more_property_path() {
        let query = "SELECT ?s ?o WHERE { ?s <http://ex.org/p>+ ?o }";
        let result = translate(query);
        assert!(
            result.is_ok(),
            "OneOrMore path translation failed: {:?}",
            result.err()
        );
        let plan = result.unwrap();

        // OneOrMore produces Distinct(Union(...))
        fn find_distinct(op: &LogicalOperator) -> bool {
            match op {
                LogicalOperator::Distinct(_) => true,
                LogicalOperator::Project(p) => find_distinct(&p.input),
                _ => false,
            }
        }
        assert!(
            find_distinct(&plan.root),
            "Expected Distinct wrapping the bounded expansion"
        );

        fn find_union_inside_distinct(op: &LogicalOperator) -> bool {
            match op {
                LogicalOperator::Distinct(d) => matches!(*d.input, LogicalOperator::Union(_)),
                LogicalOperator::Project(p) => find_union_inside_distinct(&p.input),
                _ => false,
            }
        }
        assert!(
            find_union_inside_distinct(&plan.root),
            "Expected Union inside Distinct for OneOrMore path"
        );
    }

    // === ZeroOrMore Property Path Tests ===

    #[test]
    fn test_translate_zero_or_more_property_path() {
        let query = "SELECT ?s ?o WHERE { ?s <http://ex.org/p>* ?o }";
        let result = translate(query);
        assert!(
            result.is_ok(),
            "ZeroOrMore path translation failed: {:?}",
            result.err()
        );
        let plan = result.unwrap();

        // ZeroOrMore produces Distinct(Union(...)) with reflexive branches
        fn find_distinct(op: &LogicalOperator) -> bool {
            match op {
                LogicalOperator::Distinct(_) => true,
                LogicalOperator::Project(p) => find_distinct(&p.input),
                _ => false,
            }
        }
        assert!(
            find_distinct(&plan.root),
            "Expected Distinct wrapping the bounded expansion"
        );

        // The Union should have more branches than OneOrMore (reflexive + depth branches)
        fn count_union_branches(op: &LogicalOperator) -> Option<usize> {
            match op {
                LogicalOperator::Distinct(d) => {
                    if let LogicalOperator::Union(u) = d.input.as_ref() {
                        Some(u.inputs.len())
                    } else {
                        None
                    }
                }
                LogicalOperator::Project(p) => count_union_branches(&p.input),
                _ => None,
            }
        }
        let branch_count = count_union_branches(&plan.root)
            .expect("Expected Union inside Distinct for ZeroOrMore path");
        // ZeroOrMore has 2 reflexive branches + MAX_DEPTH (10) depth branches = 12
        assert!(
            branch_count > 10,
            "ZeroOrMore should have reflexive branches plus depth branches, got {}",
            branch_count
        );
    }

    // === Sequence Property Path Tests ===

    #[test]
    fn test_translate_sequence_property_path() {
        let query = r#"
            SELECT ?name
            WHERE {
                ?person <http://ex.org/knows>/<http://ex.org/name> ?name
            }
        "#;
        let result = translate(query);
        assert!(
            result.is_ok(),
            "Sequence path translation failed: {:?}",
            result.err()
        );
        let plan = result.unwrap();

        // Sequence path expands into joined triple scans. Count TripleScan operators.
        fn count_triple_scans(op: &LogicalOperator) -> usize {
            match op {
                LogicalOperator::TripleScan(_) => 1,
                LogicalOperator::Project(p) => count_triple_scans(&p.input),
                LogicalOperator::Filter(f) => count_triple_scans(&f.input),
                LogicalOperator::Join(j) => {
                    count_triple_scans(&j.left) + count_triple_scans(&j.right)
                }
                _ => 0,
            }
        }
        let scan_count = count_triple_scans(&plan.root);
        assert!(
            scan_count >= 2,
            "Sequence path should produce at least 2 joined TripleScans, got {}",
            scan_count
        );
    }

    // === Alternative Property Path Tests ===

    #[test]
    fn test_translate_alternative_property_path() {
        let query = "SELECT ?v WHERE { ?s <http://a>|<http://b> ?v }";
        let result = translate(query);
        assert!(
            result.is_ok(),
            "Alternative path translation failed: {:?}",
            result.err()
        );
        let plan = result.unwrap();

        // Alternative path produces a Union of TripleScans
        fn find_union(op: &LogicalOperator) -> bool {
            match op {
                LogicalOperator::Union(_) => true,
                LogicalOperator::Project(p) => find_union(&p.input),
                _ => false,
            }
        }
        assert!(
            find_union(&plan.root),
            "Expected Union for alternative property path"
        );

        fn count_union_branches(op: &LogicalOperator) -> Option<usize> {
            match op {
                LogicalOperator::Union(u) => Some(u.inputs.len()),
                LogicalOperator::Project(p) => count_union_branches(&p.input),
                _ => None,
            }
        }
        let branch_count =
            count_union_branches(&plan.root).expect("Expected Union in plan for alternative path");
        assert_eq!(
            branch_count, 2,
            "Alternative path with 2 predicates should have 2 Union branches"
        );
    }

    // === Inverse Property Path Tests ===

    #[test]
    fn test_translate_inverse_property_path() {
        let query = "SELECT ?s WHERE { ?o ^<http://ex.org/knows> ?s }";
        let result = translate(query);
        assert!(
            result.is_ok(),
            "Inverse path translation failed: {:?}",
            result.err()
        );
        let plan = result.unwrap();

        // Inverse path swaps subject and object, so we should get a TripleScan
        // where the predicate is the inner IRI.
        fn find_triple_scan(op: &LogicalOperator) -> Option<&TripleScanOp> {
            match op {
                LogicalOperator::TripleScan(ts) => Some(ts),
                LogicalOperator::Project(p) => find_triple_scan(&p.input),
                LogicalOperator::Filter(f) => find_triple_scan(&f.input),
                LogicalOperator::Join(j) => {
                    find_triple_scan(&j.left).or_else(|| find_triple_scan(&j.right))
                }
                _ => None,
            }
        }
        let scan = find_triple_scan(&plan.root).expect("Expected TripleScan for inverse path");
        // The predicate should be the IRI (not the inverse wrapper)
        assert!(
            matches!(&scan.predicate, TripleComponent::Iri(_)),
            "Expected IRI predicate in TripleScan after inverse, got {:?}",
            scan.predicate
        );
    }

    // === translate() returns Ok for various path types ===

    #[test]
    fn test_translate_ok_for_named_iri_path() {
        let query = "SELECT ?s ?o WHERE { ?s <http://ex.org/rel> ?o }";
        let result = translate(query);
        assert!(
            result.is_ok(),
            "Named IRI path should translate: {:?}",
            result.err()
        );
    }

    #[test]
    fn test_translate_ok_for_zero_or_one_path() {
        let query = "SELECT ?s ?o WHERE { ?s <http://ex.org/rel>? ?o }";
        let result = translate(query);
        assert!(
            result.is_ok(),
            "ZeroOrOne path should translate: {:?}",
            result.err()
        );
    }

    #[test]
    fn test_translate_ok_for_rdf_type_shorthand() {
        let query = "SELECT ?s WHERE { ?s a <http://ex.org/Person> }";
        let result = translate(query);
        assert!(
            result.is_ok(),
            "'a' (rdf:type) shorthand should translate: {:?}",
            result.err()
        );
    }

    #[test]
    fn test_translate_ok_for_negated_property_set() {
        let query = "SELECT ?s ?o WHERE { ?s !<http://ex.org/skip> ?o }";
        let result = translate(query);
        assert!(
            result.is_ok(),
            "Negated property set should translate: {:?}",
            result.err()
        );
    }

    // === Basic SELECT verification ===

    #[test]
    fn test_translate_basic_select_structure() {
        let query = "SELECT ?x ?y WHERE { ?x <http://ex.org/p> ?y }";
        let result = translate(query);
        assert!(result.is_ok());

        let plan = result.unwrap();
        // Top-level should be a Project
        fn find_project(op: &LogicalOperator) -> Option<&ProjectOp> {
            match op {
                LogicalOperator::Project(p) => Some(p),
                _ => None,
            }
        }
        let project = find_project(&plan.root).expect("Expected Project operator at top level");
        assert_eq!(
            project.projections.len(),
            2,
            "SELECT ?x ?y should produce 2 projections"
        );
    }

    // === OPTIONAL pattern ===

    #[test]
    fn test_translate_optional_produces_left_join() {
        let query = "SELECT ?x ?name WHERE { ?x <http://ex.org/type> ?t OPTIONAL { ?x <http://ex.org/name> ?name } }";
        let result = translate(query);
        assert!(
            result.is_ok(),
            "OPTIONAL should translate: {:?}",
            result.err()
        );

        let plan = result.unwrap();
        fn find_left_join(op: &LogicalOperator) -> bool {
            match op {
                LogicalOperator::LeftJoin(_) => true,
                LogicalOperator::Project(p) => find_left_join(&p.input),
                LogicalOperator::Filter(f) => find_left_join(&f.input),
                _ => false,
            }
        }
        assert!(
            find_left_join(&plan.root),
            "OPTIONAL should produce a LeftJoin operator in the plan"
        );
    }

    // === SERVICE clause tests (federated queries not supported) ===

    #[test]
    fn test_service_clause_returns_explicit_error() {
        let result =
            translate("SELECT ?x WHERE { SERVICE <http://example.org/sparql> { ?x ?p ?o } }");
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(
            err.contains("SERVICE"),
            "Error should mention SERVICE, got: {err}"
        );
    }

    #[test]
    fn test_service_silent_clause_also_errors() {
        let result = translate(
            "SELECT ?x WHERE { SERVICE SILENT <http://example.org/sparql> { ?x ?p ?o } }",
        );
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(
            err.contains("SERVICE"),
            "SILENT SERVICE should also error, got: {err}"
        );
    }

    #[test]
    fn test_service_clause_with_local_patterns_still_errors() {
        // Ensure SERVICE is not silently executed locally
        let result =
            translate("SELECT ?x ?y WHERE { ?x ?p ?y . SERVICE <http://remote/> { ?y ?q ?z } }");
        assert!(result.is_err());
    }

    #[test]
    fn test_query_without_service_still_works() {
        // Regression: ensure normal queries still translate fine
        let result = translate("SELECT ?x ?y WHERE { ?x ?p ?y }");
        assert!(result.is_ok());
    }

    // === SAMPLE aggregate tests (3G verification) ===

    #[test]
    fn test_sample_aggregate_translates() {
        let result = translate(
            "SELECT (SAMPLE(?name) AS ?sampleName) WHERE { ?x <http://example.org/name> ?name }",
        );
        assert!(
            result.is_ok(),
            "SAMPLE aggregate should translate: {:?}",
            result.err()
        );
    }

    #[test]
    fn test_group_concat_with_separator_translates() {
        let result = translate(
            r#"SELECT (GROUP_CONCAT(?name; separator=", ") AS ?names) WHERE { ?x <http://example.org/name> ?name }"#,
        );
        assert!(
            result.is_ok(),
            "GROUP_CONCAT with separator should translate: {:?}",
            result.err()
        );
    }

    #[test]
    fn test_group_concat_without_separator_translates() {
        let result = translate(
            "SELECT (GROUP_CONCAT(?name) AS ?names) WHERE { ?x <http://example.org/name> ?name }",
        );
        assert!(
            result.is_ok(),
            "GROUP_CONCAT without separator should translate: {:?}",
            result.err()
        );
    }

    // === Property path depth tests (3D verification) ===

    #[test]
    fn test_property_path_plus_translates_beyond_10() {
        // With MAX_DEPTH=50, a + path should expand to more than 10 levels
        let result = translate("SELECT ?x ?y WHERE { ?x <http://example.org/knows>+ ?y }");
        assert!(
            result.is_ok(),
            "Property path + should translate: {:?}",
            result.err()
        );
    }

    #[test]
    fn test_property_path_star_translates() {
        let result = translate("SELECT ?x ?y WHERE { ?x <http://example.org/knows>* ?y }");
        assert!(
            result.is_ok(),
            "Property path * should translate: {:?}",
            result.err()
        );
    }

    #[test]
    fn test_property_path_optional_translates() {
        let result = translate("SELECT ?x ?y WHERE { ?x <http://example.org/knows>? ?y }");
        assert!(
            result.is_ok(),
            "Property path ? should translate: {:?}",
            result.err()
        );
    }
}
