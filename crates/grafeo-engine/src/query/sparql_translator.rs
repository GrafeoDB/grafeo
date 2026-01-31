//! SPARQL to LogicalPlan translator.
//!
//! Translates SPARQL 1.1 AST to the common logical plan representation.

use crate::query::plan::{
    AddGraphOp, AggregateExpr, AggregateFunction, AggregateOp, AntiJoinOp, BinaryOp, BindOp,
    ClearGraphOp, CopyGraphOp, CreateGraphOp, DeleteTripleOp, DistinctOp, DropGraphOp, FilterOp,
    InsertTripleOp, JoinOp, JoinType, LeftJoinOp, LimitOp, LoadGraphOp, LogicalExpression,
    LogicalOperator, LogicalPlan, ModifyOp, MoveGraphOp, ProjectOp, Projection, SkipOp, SortKey,
    SortOp, SortOrder, TripleComponent, TripleScanOp, TripleTemplate, UnaryOp, UnionOp,
};
use grafeo_adapters::query::sparql::{self, ast};
use grafeo_common::types::Value;
use grafeo_common::utils::error::{Error, Result};
use std::collections::HashMap;

/// Translates a SPARQL query string to a logical plan.
///
/// # Errors
///
/// Returns an error if the query cannot be parsed or translated.
pub fn translate(query: &str) -> Result<LogicalPlan> {
    let sparql_query = sparql::parse(query)?;
    let mut translator = SparqlTranslator::new();
    translator.translate_query(&sparql_query)
}

/// Translator from SPARQL AST to LogicalPlan.
struct SparqlTranslator {
    /// Prefix mappings for IRI resolution.
    prefixes: HashMap<String, String>,
    /// Base IRI for relative IRI resolution.
    base: Option<String>,
    /// Counter for generating anonymous variables.
    anon_counter: u32,
}

impl SparqlTranslator {
    fn new() -> Self {
        Self {
            prefixes: HashMap::new(),
            base: None,
            anon_counter: 0,
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
        // Start with the WHERE clause pattern
        let mut plan = self.translate_graph_pattern(&select.where_clause)?;

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

            plan = LogicalOperator::Aggregate(AggregateOp {
                group_by: group_by_exprs,
                aggregates,
                input: Box::new(plan),
                having: None, // SPARQL HAVING handled as separate Filter below
            });

            // Apply HAVING if present
            if let Some(having) = &select.solution_modifiers.having {
                plan = LogicalOperator::Filter(FilterOp {
                    predicate: self.translate_expression(having)?,
                    input: Box::new(plan),
                });
            }
        }

        // Apply ORDER BY
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
                    })
                })
                .collect::<Result<Vec<_>>>()?;

            plan = LogicalOperator::Sort(SortOp {
                keys,
                input: Box::new(plan),
            });
        }

        // Apply OFFSET
        if let Some(offset) = select.solution_modifiers.offset {
            plan = LogicalOperator::Skip(SkipOp {
                count: offset as usize,
                input: Box::new(plan),
            });
        }

        // Apply LIMIT
        if let Some(limit) = select.solution_modifiers.limit {
            plan = LogicalOperator::Limit(LimitOp {
                count: limit as usize,
                input: Box::new(plan),
            });
        }

        // Apply DISTINCT/REDUCED
        if select.modifier == ast::SelectModifier::Distinct {
            plan = LogicalOperator::Distinct(DistinctOp {
                input: Box::new(plan),
                columns: None,
            });
        }

        // Apply projection (but NOT for aggregate queries - aggregate already produces correct columns)
        // For aggregate queries, the AggregateOp outputs columns with proper aliases
        if !has_aggregates {
            let projections = self.translate_projection(&select.projection)?;
            if !projections.is_empty() {
                plan = LogicalOperator::Project(ProjectOp {
                    projections,
                    input: Box::new(plan),
                });
            }
        }

        Ok(LogicalPlan::new(plan))
    }

    fn translate_ask(&mut self, ask: &ast::AskQuery) -> Result<LogicalPlan> {
        // ASK returns true if the pattern has any matches
        let plan = self.translate_graph_pattern(&ask.where_clause)?;

        // Limit to 1 result for efficiency
        let plan = LogicalOperator::Limit(LimitOp {
            count: 1,
            input: Box::new(plan),
        });

        Ok(LogicalPlan::new(plan))
    }

    fn translate_construct(&mut self, construct: &ast::ConstructQuery) -> Result<LogicalPlan> {
        // For CONSTRUCT, we need to evaluate the WHERE pattern and then
        // produce triples according to the template
        let plan = self.translate_graph_pattern(&construct.where_clause)?;

        // Apply solution modifiers
        let mut plan = plan;
        if let Some(limit) = construct.solution_modifiers.limit {
            plan = LogicalOperator::Limit(LimitOp {
                count: limit as usize,
                input: Box::new(plan),
            });
        }

        // The template will be processed at execution time
        Ok(LogicalPlan::new(plan))
    }

    fn translate_describe(&mut self, describe: &ast::DescribeQuery) -> Result<LogicalPlan> {
        // DESCRIBE returns information about resources
        if let Some(where_clause) = &describe.where_clause {
            let plan = self.translate_graph_pattern(where_clause)?;
            Ok(LogicalPlan::new(plan))
        } else {
            // DESCRIBE with just IRIs - create a scan for each resource
            Ok(LogicalPlan::new(LogicalOperator::Empty))
        }
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
                using_clauses: _,
                where_clause,
            } => self.translate_modify(with_graph, delete_template, insert_template, where_clause),
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
            let subject = self.translate_triple_term(&quad.triple.subject)?;
            let predicate = self.translate_property_path(&quad.triple.predicate)?;
            let object = self.translate_triple_term(&quad.triple.object)?;
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
            Ok(LogicalPlan::new(ops.into_iter().next().unwrap()))
        } else {
            Ok(LogicalPlan::new(LogicalOperator::Union(UnionOp {
                inputs: ops,
            })))
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
            Ok(LogicalPlan::new(ops.into_iter().next().unwrap()))
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
        for triple in &triples {
            let subject = self.translate_triple_term(&triple.subject)?;
            let predicate = self.translate_property_path(&triple.predicate)?;
            let object = self.translate_triple_term(&triple.object)?;

            ops.push(LogicalOperator::DeleteTriple(DeleteTripleOp {
                subject,
                predicate,
                object,
                graph: None, // Default graph
                input: Some(Box::new(match_plan.clone())),
            }));
        }

        if ops.is_empty() {
            Ok(LogicalPlan::new(match_plan))
        } else if ops.len() == 1 {
            Ok(LogicalPlan::new(ops.into_iter().next().unwrap()))
        } else {
            Ok(LogicalPlan::new(LogicalOperator::Union(UnionOp {
                inputs: ops,
            })))
        }
    }

    fn extract_triples_from_pattern(pattern: &ast::GraphPattern) -> Vec<ast::TriplePattern> {
        match pattern {
            ast::GraphPattern::Basic(triples) => triples.clone(),
            ast::GraphPattern::Group(patterns) => patterns
                .iter()
                .flat_map(Self::extract_triples_from_pattern)
                .collect(),
            _ => Vec::new(),
        }
    }

    fn translate_modify(
        &mut self,
        with_graph: &Option<ast::Iri>,
        delete_template: &Option<Vec<ast::QuadPattern>>,
        insert_template: &Option<Vec<ast::QuadPattern>>,
        where_clause: &ast::GraphPattern,
    ) -> Result<LogicalPlan> {
        // Translate the WHERE clause - this will be evaluated once and shared
        let where_plan = self.translate_graph_pattern(where_clause)?;

        let default_graph = with_graph.as_ref().map(|g| self.resolve_iri(g));

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
                // Categorize patterns by type for proper composition
                let mut basic_patterns: Vec<&ast::GraphPattern> = Vec::new();
                let mut filter_exprs: Vec<&ast::Expression> = Vec::new();
                let mut optional_patterns: Vec<&ast::GraphPattern> = Vec::new();
                let mut minus_patterns: Vec<&ast::GraphPattern> = Vec::new();
                let mut bind_patterns: Vec<(&ast::Expression, &String)> = Vec::new();

                for p in patterns {
                    match p {
                        ast::GraphPattern::Filter(expr) => filter_exprs.push(expr),
                        ast::GraphPattern::Optional(inner) => optional_patterns.push(inner),
                        ast::GraphPattern::Minus(inner) => minus_patterns.push(inner),
                        ast::GraphPattern::Bind {
                            expression,
                            variable,
                        } => bind_patterns.push((expression, variable)),
                        _ => basic_patterns.push(p),
                    }
                }

                // 1. Translate and join basic/required patterns
                let mut plan = LogicalOperator::Empty;
                for p in basic_patterns {
                    let p_plan = self.translate_graph_pattern(p)?;
                    plan = self.join_patterns(plan, p_plan);
                }

                // 2. Apply BIND expressions (adds computed columns)
                for (expression, variable) in bind_patterns {
                    let expr = self.translate_expression(expression)?;
                    plan = LogicalOperator::Bind(BindOp {
                        expression: expr,
                        variable: variable.clone(),
                        input: Box::new(plan),
                    });
                }

                // 3. Apply OPTIONAL patterns (left outer joins)
                for inner in optional_patterns {
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

                // 4. Apply MINUS patterns (anti joins)
                for inner in minus_patterns {
                    let inner_plan = self.translate_graph_pattern(inner)?;
                    if !matches!(plan, LogicalOperator::Empty) {
                        plan = LogicalOperator::AntiJoin(AntiJoinOp {
                            left: Box::new(plan),
                            right: Box::new(inner_plan),
                        });
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
                        .unwrap();

                    plan = LogicalOperator::Filter(FilterOp {
                        predicate: combined,
                        input: Box::new(plan),
                    });
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
                Ok(LogicalOperator::Filter(FilterOp {
                    predicate,
                    input: Box::new(LogicalOperator::Empty),
                }))
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

            ast::GraphPattern::NamedGraph { graph: _, pattern } => {
                // For named graph, we add the graph as context
                self.translate_graph_pattern(pattern)
            }

            ast::GraphPattern::SubSelect(subquery) => {
                let plan = self.translate_select(subquery)?;
                Ok(plan.root)
            }

            ast::GraphPattern::Service {
                endpoint: _,
                pattern,
                silent: _,
            } => {
                // SERVICE queries remote endpoints - for now, translate the pattern
                self.translate_graph_pattern(pattern)
            }

            ast::GraphPattern::InlineData(_data) => {
                // VALUES clause - inline data
                // For now, return empty; full implementation needs a Values operator
                Ok(LogicalOperator::Empty)
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
        let subject = self.translate_triple_term(&triple.subject)?;
        let predicate = self.translate_property_path(&triple.predicate)?;
        let object = self.translate_triple_term(&triple.object)?;

        Ok(LogicalOperator::TripleScan(TripleScanOp {
            subject,
            predicate,
            object,
            graph: None,
            input: None,
        }))
    }

    fn translate_triple_term(&mut self, term: &ast::TripleTerm) -> Result<TripleComponent> {
        match term {
            ast::TripleTerm::Variable(name) => Ok(TripleComponent::Variable(name.clone())),
            ast::TripleTerm::Iri(iri) => Ok(TripleComponent::Iri(self.resolve_iri(iri))),
            ast::TripleTerm::Literal(lit) => {
                let value = self.literal_to_value(lit);
                Ok(TripleComponent::Literal(value))
            }
            ast::TripleTerm::BlankNode(bnode) => {
                // Treat blank nodes as variables
                match bnode {
                    ast::BlankNode::Labeled(label) => {
                        Ok(TripleComponent::Variable(format!("_:{}", label)))
                    }
                    ast::BlankNode::Anonymous(_) => {
                        let var = format!("_:anon{}", self.next_anon());
                        Ok(TripleComponent::Variable(var))
                    }
                }
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
            _ => Err(Error::Internal(
                "Complex property paths not yet supported".to_string(),
            )),
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

            ast::Expression::Exists(_pattern) => {
                // EXISTS { pattern } - check if pattern has any matches
                Ok(LogicalExpression::FunctionCall {
                    name: "EXISTS".to_string(),
                    args: vec![],
                    distinct: false,
                })
            }

            ast::Expression::NotExists(_pattern) => {
                // NOT EXISTS { pattern }
                Ok(LogicalExpression::FunctionCall {
                    name: "NOT_EXISTS".to_string(),
                    args: vec![],
                    distinct: false,
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
            ast::UnaryOperator::Plus => UnaryOp::Not, // No direct mapping, use identity
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
            let (func, expr_inner, distinct) = match agg {
                ast::AggregateExpression::Count {
                    distinct,
                    expression,
                } => (
                    AggregateFunction::Count,
                    expression.as_ref().map(|e| e.as_ref()),
                    *distinct,
                ),
                ast::AggregateExpression::Sum {
                    distinct,
                    expression,
                } => (AggregateFunction::Sum, Some(expression.as_ref()), *distinct),
                ast::AggregateExpression::Average {
                    distinct,
                    expression,
                } => (AggregateFunction::Avg, Some(expression.as_ref()), *distinct),
                ast::AggregateExpression::Minimum { expression } => {
                    (AggregateFunction::Min, Some(expression.as_ref()), false)
                }
                ast::AggregateExpression::Maximum { expression } => {
                    (AggregateFunction::Max, Some(expression.as_ref()), false)
                }
                ast::AggregateExpression::Sample { expression } => {
                    // Map SAMPLE to Collect for now
                    (AggregateFunction::Collect, Some(expression.as_ref()), false)
                }
                ast::AggregateExpression::GroupConcat {
                    distinct,
                    expression,
                    ..
                } => (
                    AggregateFunction::Collect,
                    Some(expression.as_ref()),
                    *distinct,
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
                distinct,
                alias: alias.clone(),
                percentile: None, // SPARQL doesn't support percentile functions
            }))
        } else {
            Ok(None)
        }
    }

    fn join_patterns(&self, left: LogicalOperator, right: LogicalOperator) -> LogicalOperator {
        if matches!(left, LogicalOperator::Empty) {
            return right;
        }
        if matches!(right, LogicalOperator::Empty) {
            return left;
        }

        // For basic patterns, use inner join on shared variables
        LogicalOperator::Join(JoinOp {
            left: Box::new(left),
            right: Box::new(right),
            join_type: JoinType::Inner,
            conditions: vec![], // Shared variables are implicit join conditions
        })
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
                _ => {}
            }
        }

        // Default to string
        Value::String(lit.value.clone().into())
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
}
