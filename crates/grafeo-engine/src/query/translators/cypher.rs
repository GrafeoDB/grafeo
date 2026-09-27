//! Cypher AST to Logical Plan translator.
//!
//! Translates parsed Cypher queries into the common logical plan representation
//! that can be optimized and executed.

use super::common::{
    build_left_join_with_predicates, check_union_columns, combine_with_and, is_aggregate_function,
    to_aggregate_function, wrap_distinct, wrap_filter, wrap_limit, wrap_return, wrap_skip,
    wrap_sort,
};
use crate::query::plan::{
    AddLabelOp, AggregateExpr, AggregateFunction, AggregateOp, ApplyOp, BinaryOp, CallProcedureOp,
    CountExpr, CreateEdgeOp, CreateNodeOp, DeleteEdgeOp, DeleteNodeOp, ExpandDirection, ExpandOp,
    JoinCondition, JoinOp, JoinType, LeftJoinOp, ListPredicateKind, LoadDataFormat, LoadDataOp,
    LogicalExpression, LogicalOperator, LogicalPlan, MapProjectionEntry, MergeOp,
    MergeRelationshipOp, NodeScanOp, ParameterScanOp, PathMode, ProcedureYield, ProjectOp,
    Projection, RemoveLabelOp, ReturnItem, SetPropertyOp, ShortestPathOp, SortKey, SortOrder,
    UnaryOp, UnionOp, UnwindOp,
};
use grafeo_adapters::query::cypher::{self, ast};
use grafeo_common::types::Value;
use grafeo_common::utils::error::{Error, QueryError, QueryErrorKind, Result};
use std::cell::{Cell, RefCell};
use std::collections::{HashMap, HashSet};

/// Result of translating a Cypher query: either a plan or a schema DDL command.
#[non_exhaustive]
pub enum CypherTranslationResult {
    /// Regular query or mutation, produces a logical plan.
    Plan(LogicalPlan),
    /// Schema DDL (CREATE/DROP INDEX, CREATE/DROP CONSTRAINT).
    SchemaCommand(grafeo_adapters::query::gql::ast::SchemaStatement),
    /// SHOW INDEXES introspection.
    ShowIndexes,
    /// SHOW CONSTRAINTS introspection.
    ShowConstraints,
    /// SHOW CURRENT GRAPH TYPE introspection.
    ShowCurrentGraphType,
}

/// Translates a Cypher query string to a logical plan.
///
/// # Errors
///
/// Returns an error if parsing fails or the query is a schema command
/// that cannot be represented as a logical plan.
pub fn translate(query: &str) -> Result<LogicalPlan> {
    match translate_full(query)? {
        CypherTranslationResult::Plan(plan) => Ok(plan),
        _ => Err(Error::Query(QueryError::new(
            QueryErrorKind::Semantic,
            "Schema commands cannot be translated to a logical plan",
        ))),
    }
}

/// Translates a Cypher query, returning either a plan or a schema command.
///
/// # Errors
///
/// Returns an error if parsing fails or the AST contains unsupported constructs.
pub fn translate_full(query: &str) -> Result<CypherTranslationResult> {
    let statement = cypher::parse(query)?;
    let translator = CypherTranslator::new();
    translator.translate_statement_full(&statement)
}

/// Cypher AST to logical plan translator.
struct CypherTranslator {
    /// Variables bound to edges (from MATCH relationship patterns or MERGE relationships).
    /// Used to set `is_edge: true` on `SetPropertyOp` when the SET target is an edge variable.
    edge_variables: RefCell<HashSet<String>>,
    /// Counter for generating unique anonymous variable names.
    anon_counter: Cell<u32>,
    /// Alias-to-output-column-name mapping from the most recent RETURN/WITH clause.
    /// Used by ORDER BY to resolve alias references to actual output column names.
    return_aliases: RefCell<HashMap<String, String>>,
}

impl CypherTranslator {
    fn new() -> Self {
        Self {
            edge_variables: RefCell::new(HashSet::new()),
            anon_counter: Cell::new(0),
            return_aliases: RefCell::new(HashMap::new()),
        }
    }

    /// Generates a unique anonymous variable name.
    fn next_anon_var(&self) -> String {
        let id = self.anon_counter.get();
        self.anon_counter.set(id + 1);
        format!("_anon_{id}")
    }

    /// Records a variable as an edge variable.
    fn register_edge_variable(&self, variable: &str) {
        self.edge_variables
            .borrow_mut()
            .insert(variable.to_string());
    }

    /// Returns true if the variable was bound to an edge.
    fn is_edge_variable(&self, variable: &str) -> bool {
        self.edge_variables.borrow().contains(variable)
    }

    fn translate_statement(&self, stmt: &ast::Statement) -> Result<LogicalPlan> {
        match stmt {
            ast::Statement::Query(query) => self.translate_query(query),
            ast::Statement::Create(create) => self.translate_create_statement(create),
            ast::Statement::Merge(merge) => self.translate_merge_statement(merge),
            ast::Statement::Delete(_) => Err(Error::Query(QueryError::new(
                QueryErrorKind::Semantic,
                "Standalone DELETE requires a preceding MATCH clause. Use: MATCH (n) WHERE ... DELETE n",
            ))),
            ast::Statement::Set(_) => Err(Error::Query(QueryError::new(
                QueryErrorKind::Semantic,
                "Standalone SET requires a preceding MATCH clause. Use: MATCH (n) WHERE ... SET n.prop = value",
            ))),
            ast::Statement::Remove(_) => Err(Error::Query(QueryError::new(
                QueryErrorKind::Semantic,
                "Standalone REMOVE requires a preceding MATCH clause. Use: MATCH (n) WHERE ... REMOVE n.prop",
            ))),
            ast::Statement::Union { queries, all } => {
                let inputs: Vec<LogicalOperator> = queries
                    .iter()
                    .map(|q| {
                        let plan = self.translate_query(q)?;
                        Ok(plan.root)
                    })
                    .collect::<Result<Vec<_>>>()?;
                check_union_columns(&inputs)?;

                let union_op = LogicalOperator::Union(UnionOp { inputs });

                // UNION (not ALL) removes duplicates
                let root = if *all {
                    union_op
                } else {
                    wrap_distinct(union_op)
                };

                Ok(LogicalPlan::new(root))
            }
            ast::Statement::Explain(inner) => {
                let mut plan = self.translate_statement(inner)?;
                plan.explain = true;
                Ok(plan)
            }
            ast::Statement::Profile(inner) => {
                let mut plan = self.translate_statement(inner)?;
                plan.profile = true;
                Ok(plan)
            }
            ast::Statement::Schema(_)
            | ast::Statement::ShowIndexes
            | ast::Statement::ShowConstraints
            | ast::Statement::ShowCurrentGraphType => Err(Error::Query(QueryError::new(
                QueryErrorKind::Semantic,
                "Schema commands should be routed through translate_statement_full",
            ))),
        }
    }

    fn translate_statement_full(&self, stmt: &ast::Statement) -> Result<CypherTranslationResult> {
        match stmt {
            ast::Statement::Schema(schema) => {
                Ok(CypherTranslationResult::SchemaCommand(schema.clone()))
            }
            ast::Statement::ShowIndexes => Ok(CypherTranslationResult::ShowIndexes),
            ast::Statement::ShowConstraints => Ok(CypherTranslationResult::ShowConstraints),
            ast::Statement::ShowCurrentGraphType => {
                Ok(CypherTranslationResult::ShowCurrentGraphType)
            }
            other => {
                let plan = self.translate_statement(other)?;
                Ok(CypherTranslationResult::Plan(plan))
            }
        }
    }

    fn translate_query(&self, query: &ast::Query) -> Result<LogicalPlan> {
        let mut plan: Option<LogicalOperator> = None;

        for clause in &query.clauses {
            plan = Some(self.translate_clause(clause, plan)?);
        }

        let root = plan.ok_or_else(|| {
            Error::Query(QueryError::new(QueryErrorKind::Semantic, "Empty query"))
        })?;
        Ok(LogicalPlan::new(root))
    }

    fn translate_clause(
        &self,
        clause: &ast::Clause,
        input: Option<LogicalOperator>,
    ) -> Result<LogicalOperator> {
        match clause {
            ast::Clause::Match(match_clause) => self.translate_match(match_clause, input),
            ast::Clause::OptionalMatch(match_clause) => {
                self.translate_optional_match(match_clause, input)
            }
            ast::Clause::Where(where_clause) => self.translate_where(where_clause, input),
            ast::Clause::With(with_clause) => self.translate_with(with_clause, input),
            ast::Clause::Return(return_clause) => self.translate_return(return_clause, input),
            ast::Clause::Unwind(unwind_clause) => self.translate_unwind(unwind_clause, input),
            ast::Clause::OrderBy(order_by) => self.translate_order_by(order_by, input),
            ast::Clause::Skip(expr) => self.translate_skip(expr, input),
            ast::Clause::Limit(expr) => self.translate_limit(expr, input),
            ast::Clause::Create(create_clause) => {
                self.translate_create_clause(create_clause, input)
            }
            ast::Clause::Merge(merge_clause) => self.translate_merge(merge_clause, input),
            ast::Clause::Delete(delete_clause) => self.translate_delete(delete_clause, input),
            ast::Clause::Set(set_clause) => self.translate_set(set_clause, input),
            ast::Clause::Remove(remove_clause) => self.translate_remove(remove_clause, input),
            ast::Clause::Call(call) => self.translate_call_clause(call, input),
            ast::Clause::CallSubquery(inner_query) => {
                self.translate_call_subquery(inner_query, input)
            }
            ast::Clause::ForEach(foreach) => self.translate_foreach(foreach, input),
            ast::Clause::LoadCsv(load_csv) => self.translate_load_csv(load_csv),
        }
    }

    fn translate_load_csv(&self, load_csv: &ast::LoadCsvClause) -> Result<LogicalOperator> {
        Ok(LogicalOperator::LoadData(LoadDataOp {
            format: LoadDataFormat::Csv,
            with_headers: load_csv.with_headers,
            path: load_csv.path.clone(),
            variable: load_csv.variable.clone(),
            field_terminator: load_csv.field_terminator,
        }))
    }

    fn translate_call_clause(
        &self,
        call: &ast::CallClause,
        _input: Option<LogicalOperator>,
    ) -> Result<LogicalOperator> {
        let arguments = call
            .arguments
            .iter()
            .map(|a| self.translate_expression(a))
            .collect::<Result<Vec<_>>>()?;

        let yield_items = call.yield_items.as_ref().map(|items| {
            items
                .iter()
                .map(|item| ProcedureYield {
                    field_name: item.field_name.clone(),
                    alias: item.alias.clone(),
                })
                .collect()
        });

        Ok(LogicalOperator::CallProcedure(CallProcedureOp {
            name: call.procedure_name.clone(),
            arguments,
            yield_items,
        }))
    }

    /// Translates `CALL { subquery }` to an Apply operator.
    ///
    /// When the inner subquery starts with `WITH <vars>` and there is an outer
    /// input, the WITH items are treated as variable imports from the outer scope.
    /// The imported variable names are recorded in `ApplyOp.shared_variables` so
    /// the planner can wire them through `ParameterState`.
    fn translate_call_subquery(
        &self,
        inner: &ast::Query,
        input: Option<LogicalOperator>,
    ) -> Result<LogicalOperator> {
        // Detect importing WITH: if the first clause is WITH and we have outer input,
        // extract the imported variable names and start the inner plan from a
        // ParameterScan instead of Empty.
        let mut shared_variables = Vec::new();
        let mut inner_plan: Option<LogicalOperator> = None;
        let mut clauses_iter = inner.clauses.iter();

        if input.is_some()
            && let Some(ast::Clause::With(with_clause)) = inner.clauses.first()
        {
            if with_clause.is_wildcard {
                // WITH * imports all outer variables
                shared_variables.push("*".to_string());
            } else {
                for item in &with_clause.items {
                    if let ast::Expression::Variable(name) = &item.expression {
                        let var_name = item.alias.as_deref().unwrap_or(name);
                        shared_variables.push(var_name.to_string());
                    }
                }
            }
            if !shared_variables.is_empty() {
                // Skip the importing WITH and start from a ParameterScan
                clauses_iter.next();
                inner_plan = Some(LogicalOperator::ParameterScan(ParameterScanOp {
                    columns: shared_variables.clone(),
                }));
            }
        }

        // Translate the remaining inner subquery clauses
        for clause in clauses_iter {
            inner_plan = Some(self.translate_clause(clause, inner_plan)?);
        }
        let inner_plan = inner_plan.ok_or_else(|| {
            Error::Query(QueryError::new(
                QueryErrorKind::Semantic,
                "CALL subquery requires at least one clause",
            ))
        })?;

        match input {
            Some(outer) => Ok(LogicalOperator::Apply(ApplyOp {
                input: Box::new(outer),
                subplan: Box::new(inner_plan),
                shared_variables,
                optional: false,
            })),
            None => Ok(inner_plan),
        }
    }

    /// Translates `FOREACH (var IN list | clauses)` to Unwind + mutation pipeline.
    fn translate_foreach(
        &self,
        foreach: &ast::ForEachClause,
        input: Option<LogicalOperator>,
    ) -> Result<LogicalOperator> {
        let input = input.ok_or_else(|| {
            Error::Query(QueryError::new(
                QueryErrorKind::Semantic,
                "FOREACH requires preceding input (e.g., a MATCH clause)",
            ))
        })?;

        let list_expr = self.translate_expression(&foreach.list)?;

        // Unwind the list into individual rows
        let unwind = LogicalOperator::Unwind(UnwindOp {
            input: Box::new(input),
            expression: list_expr,
            variable: foreach.variable.clone(),
            ordinality_var: None,
            offset_var: None,
        });

        // Chain the inner mutation clauses
        let mut plan = unwind;
        for clause in &foreach.clauses {
            plan = self.translate_clause(clause, Some(plan))?;
        }

        Ok(plan)
    }

    /// Extracts all named variables from a Cypher AST pattern.
    fn pattern_variables(pattern: &ast::Pattern) -> HashSet<String> {
        let mut vars = HashSet::new();
        match pattern {
            ast::Pattern::Node(node) => {
                if let Some(v) = &node.variable {
                    vars.insert(v.clone());
                }
            }
            ast::Pattern::Path(path) => {
                if let Some(v) = &path.start.variable {
                    vars.insert(v.clone());
                }
                for rel in &path.chain {
                    if let Some(v) = &rel.variable {
                        vars.insert(v.clone());
                    }
                    if let Some(v) = &rel.target.variable {
                        vars.insert(v.clone());
                    }
                }
            }
            ast::Pattern::NamedPath { name, pattern, .. } => {
                vars.insert(name.clone());
                vars.extend(Self::pattern_variables(pattern));
            }
        }
        vars
    }

    /// Translates comma-separated patterns, creating proper joins for shared
    /// variables instead of cross products.
    ///
    /// The first pattern receives `input` (to chain with prior clauses like
    /// UNWIND). Subsequent patterns that share variables with earlier patterns
    /// are translated independently and joined via `JoinOp` with equality
    /// conditions on the shared variables.
    fn translate_comma_patterns(
        &self,
        patterns: &[ast::Pattern],
        input: Option<LogicalOperator>,
    ) -> Result<LogicalOperator> {
        if patterns.is_empty() {
            return Err(Error::Query(QueryError::new(
                QueryErrorKind::Semantic,
                "Empty MATCH pattern",
            )));
        }

        // Single pattern: fast path, no join logic needed
        if patterns.len() == 1 {
            return self.translate_pattern(&patterns[0], input);
        }

        // Multiple patterns: detect shared variables and create joins
        let pattern_vars: Vec<HashSet<String>> =
            patterns.iter().map(Self::pattern_variables).collect();

        let mut plan = self.translate_pattern(&patterns[0], input)?;
        let mut bound_vars = pattern_vars[0].clone();

        for (index, pattern) in patterns.iter().enumerate().skip(1) {
            let current_vars = &pattern_vars[index];
            let shared: Vec<String> = current_vars.intersection(&bound_vars).cloned().collect();

            if shared.is_empty() {
                // No shared variables: chain as input (cross product)
                plan = self.translate_pattern(pattern, Some(plan))?;
            } else {
                // Shared variables: translate independently and inner join
                let right = self.translate_pattern(pattern, None)?;
                let conditions = shared
                    .iter()
                    .map(|var| JoinCondition {
                        left: LogicalExpression::Variable(var.clone()),
                        right: LogicalExpression::Variable(var.clone()),
                    })
                    .collect();
                plan = LogicalOperator::Join(JoinOp {
                    left: Box::new(plan),
                    right: Box::new(right),
                    join_type: JoinType::Inner,
                    conditions,
                });
            }

            bound_vars.extend(current_vars.iter().cloned());
        }

        Ok(plan)
    }

    fn translate_match(
        &self,
        match_clause: &ast::MatchClause,
        input: Option<LogicalOperator>,
    ) -> Result<LogicalOperator> {
        self.translate_comma_patterns(&match_clause.patterns, input)
    }

    fn translate_optional_match(
        &self,
        match_clause: &ast::MatchClause,
        input: Option<LogicalOperator>,
    ) -> Result<LogicalOperator> {
        // OPTIONAL MATCH uses LEFT JOIN semantics
        let input = input.ok_or_else(|| {
            Error::Query(QueryError::new(
                QueryErrorKind::Semantic,
                "OPTIONAL MATCH requires input",
            ))
        })?;

        // Build the right side with proper shared variable joins
        let right = self.translate_comma_patterns(&match_clause.patterns, None)?;

        Ok(LogicalOperator::LeftJoin(LeftJoinOp {
            left: Box::new(input),
            right: Box::new(right),
            condition: None,
        }))
    }

    fn translate_pattern(
        &self,
        pattern: &ast::Pattern,
        input: Option<LogicalOperator>,
    ) -> Result<LogicalOperator> {
        match pattern {
            ast::Pattern::Node(node_pattern) => self.translate_node_pattern(node_pattern, input),
            ast::Pattern::Path(path_pattern) => self.translate_path_pattern(path_pattern, input),
            ast::Pattern::NamedPath {
                name,
                path_function,
                pattern,
            } => {
                // Check if this is a path function (shortestPath/allShortestPaths)
                if let Some(func) = path_function {
                    self.translate_shortest_path(name, *func, pattern, input)
                } else {
                    // Pass the path alias through to the inner pattern
                    self.translate_pattern_with_alias(pattern, input, Some(name.clone()))
                }
            }
        }
    }

    fn translate_node_pattern(
        &self,
        node: &ast::NodePattern,
        input: Option<LogicalOperator>,
    ) -> Result<LogicalOperator> {
        let variable = node
            .variable
            .clone()
            .unwrap_or_else(|| self.next_anon_var());
        let label = node.labels.first().cloned();

        let mut plan = LogicalOperator::NodeScan(NodeScanOp {
            variable: variable.clone(),
            label,
            input: input.map(Box::new),
        });

        // Add hasLabel filters for additional labels (AND semantics).
        // First label is used in NodeScan for scan-time filtering; remaining
        // labels are checked via post-scan Filter.
        if node.labels.len() > 1 {
            let mut combined: Option<LogicalExpression> = None;
            for extra_label in &node.labels[1..] {
                let check = LogicalExpression::FunctionCall {
                    name: "hasLabel".into(),
                    args: vec![
                        LogicalExpression::Variable(variable.clone()),
                        LogicalExpression::Literal(Value::String(extra_label.clone().into())),
                    ],
                    distinct: false,
                };
                combined = Some(match combined {
                    None => check,
                    Some(prev) => LogicalExpression::Binary {
                        left: Box::new(prev),
                        op: crate::query::plan::BinaryOp::And,
                        right: Box::new(check),
                    },
                });
            }
            if let Some(predicate) = combined {
                plan = wrap_filter(plan, predicate);
            }
        }

        // Add filter for inline properties (e.g., {city: 'NYC'})
        if !node.properties.is_empty() {
            let predicate = self.build_property_predicate(&variable, &node.properties)?;
            plan = wrap_filter(plan, predicate);
        }

        Ok(plan)
    }

    /// Builds a predicate expression for property filters like {name: 'Alix', city: 'NYC'}.
    ///
    /// When a pattern property is null (e.g., `{key: null}`), we emit `IS NULL`
    /// instead of equality, because `null = null` is `null` (falsy) in
    /// three-valued logic but a null property pattern should match absent or
    /// null-valued properties.
    fn build_property_predicate(
        &self,
        variable: &str,
        properties: &[(String, ast::Expression)],
    ) -> Result<LogicalExpression> {
        let predicates = properties
            .iter()
            .map(|(prop_name, prop_value)| {
                let left = LogicalExpression::Property {
                    variable: variable.to_string(),
                    property: prop_name.clone(),
                };
                let right = self.translate_expression(prop_value)?;
                if matches!(right, LogicalExpression::Literal(Value::Null)) {
                    Ok(LogicalExpression::Unary {
                        op: UnaryOp::IsNull,
                        operand: Box::new(left),
                    })
                } else {
                    Ok(LogicalExpression::Binary {
                        left: Box::new(left),
                        op: BinaryOp::Eq,
                        right: Box::new(right),
                    })
                }
            })
            .collect::<Result<Vec<_>>>()?;

        combine_with_and(predicates)
    }

    fn translate_path_pattern(
        &self,
        path: &ast::PathPattern,
        input: Option<LogicalOperator>,
    ) -> Result<LogicalOperator> {
        self.translate_path_pattern_with_alias(path, input, None)
    }

    fn translate_path_pattern_with_alias(
        &self,
        path: &ast::PathPattern,
        input: Option<LogicalOperator>,
        path_alias: Option<String>,
    ) -> Result<LogicalOperator> {
        let mut plan = self.translate_node_pattern(&path.start, input)?;

        for rel in &path.chain {
            plan = self.translate_relationship_pattern_with_alias(rel, plan, path_alias.clone())?;
        }

        Ok(plan)
    }

    /// Translates a pattern with an optional path alias.
    fn translate_pattern_with_alias(
        &self,
        pattern: &ast::Pattern,
        input: Option<LogicalOperator>,
        path_alias: Option<String>,
    ) -> Result<LogicalOperator> {
        match pattern {
            ast::Pattern::Node(node_pattern) => self.translate_node_pattern(node_pattern, input),
            ast::Pattern::Path(path_pattern) => {
                self.translate_path_pattern_with_alias(path_pattern, input, path_alias)
            }
            ast::Pattern::NamedPath {
                name,
                path_function,
                pattern: inner,
            } => {
                // Use the outer path alias if none was passed, otherwise use the inner one
                let alias = path_alias.or_else(|| Some(name.clone()));
                if let Some(func) = path_function {
                    self.translate_shortest_path(name, *func, inner, input)
                } else {
                    self.translate_pattern_with_alias(inner, input, alias)
                }
            }
        }
    }

    fn translate_shortest_path(
        &self,
        path_alias: &str,
        path_function: ast::PathFunction,
        pattern: &ast::Pattern,
        input: Option<LogicalOperator>,
    ) -> Result<LogicalOperator> {
        // Extract the path pattern from the inner pattern
        let path = match pattern {
            ast::Pattern::Path(p) => p,
            ast::Pattern::Node(_) => {
                return Err(Error::Query(QueryError::new(
                    QueryErrorKind::Semantic,
                    "shortestPath requires a path pattern, not a node",
                )));
            }
            ast::Pattern::NamedPath { pattern: inner, .. } => {
                // Recursively get the path pattern
                if let ast::Pattern::Path(p) = inner.as_ref() {
                    p
                } else {
                    return Err(Error::Query(QueryError::new(
                        QueryErrorKind::Semantic,
                        "shortestPath requires a path pattern",
                    )));
                }
            }
        };

        // Scan for the source node first
        let source_var = path
            .start
            .variable
            .clone()
            .unwrap_or_else(|| "_src".to_string());
        let source_label = path.start.labels.first().cloned();

        let mut plan = LogicalOperator::NodeScan(NodeScanOp {
            variable: source_var.clone(),
            label: source_label,
            input: input.map(Box::new),
        });

        // Apply property filters on the source node if any
        for (key, value) in &path.start.properties {
            let filter_expr = LogicalExpression::Binary {
                left: Box::new(LogicalExpression::Property {
                    variable: source_var.clone(),
                    property: key.clone(),
                }),
                op: BinaryOp::Eq,
                right: Box::new(self.translate_expression(value)?),
            };
            plan = wrap_filter(plan, filter_expr);
        }

        // Get the target node info from the relationship chain
        // shortestPath typically has one relationship in the chain
        if let Some(rel) = path.chain.first() {
            let target_var = rel
                .target
                .variable
                .clone()
                .unwrap_or_else(|| "_tgt".to_string());
            let target_label = rel.target.labels.first().cloned();

            // Scan for target node
            plan = LogicalOperator::NodeScan(NodeScanOp {
                variable: target_var.clone(),
                label: target_label,
                input: Some(Box::new(plan)),
            });

            // Apply property filters on the target node if any
            for (key, value) in &rel.target.properties {
                let filter_expr = LogicalExpression::Binary {
                    left: Box::new(LogicalExpression::Property {
                        variable: target_var.clone(),
                        property: key.clone(),
                    }),
                    op: BinaryOp::Eq,
                    right: Box::new(self.translate_expression(value)?),
                };
                plan = wrap_filter(plan, filter_expr);
            }

            let direction = match rel.direction {
                ast::Direction::Outgoing => ExpandDirection::Outgoing,
                ast::Direction::Incoming => ExpandDirection::Incoming,
                ast::Direction::Undirected => ExpandDirection::Both,
            };

            let edge_types = rel.types.clone();
            let all_paths = matches!(path_function, ast::PathFunction::AllShortestPaths);

            plan = LogicalOperator::ShortestPath(ShortestPathOp {
                input: Box::new(plan),
                source_var,
                target_var,
                edge_types,
                direction,
                path_alias: path_alias.to_string(),
                all_paths,
            });
        }

        Ok(plan)
    }

    fn translate_relationship_pattern_with_alias(
        &self,
        rel: &ast::RelationshipPattern,
        input: LogicalOperator,
        path_alias: Option<String>,
    ) -> Result<LogicalOperator> {
        let from_variable = Self::get_last_variable(&input)?;
        // An edge with a property map needs a variable to filter on, even when
        // the pattern leaves it anonymous: `-[:T {w: 1}]->`, `-[*1..2 {w: 1}]->`.
        let edge_variable = rel
            .variable
            .clone()
            .or_else(|| (!rel.properties.is_empty()).then(|| self.next_anon_var()));
        if let Some(ref ev) = edge_variable {
            self.register_edge_variable(ev);
        }
        let edge_variable_for_filter = edge_variable.clone();
        let edge_types = rel.types.clone();
        let to_variable = rel
            .target
            .variable
            .clone()
            .unwrap_or_else(|| self.next_anon_var());
        let target_label = rel.target.labels.first().cloned();

        let direction = match rel.direction {
            ast::Direction::Outgoing => ExpandDirection::Outgoing,
            ast::Direction::Incoming => ExpandDirection::Incoming,
            ast::Direction::Undirected => ExpandDirection::Both,
        };

        let (min_hops, max_hops) = if let Some(range) = &rel.length {
            (range.min.unwrap_or(1), range.max)
        } else {
            (1, Some(1))
        };

        // Detect cycle pattern: (s)-[*]->(s) where source == target variable.
        // The expand must use a temporary target, then filter for equality.
        let is_cycle = to_variable == from_variable;
        let expand_target = if is_cycle {
            self.next_anon_var()
        } else {
            to_variable.clone()
        };

        let expand = LogicalOperator::Expand(ExpandOp {
            from_variable,
            to_variable: expand_target.clone(),
            edge_variable,
            direction,
            edge_types,
            min_hops,
            max_hops,
            input: Box::new(input),
            path_alias,
            path_mode: PathMode::Walk,
        });

        // For cycle patterns, enforce that expanded target == original source
        let expand = if is_cycle {
            wrap_filter(
                expand,
                LogicalExpression::Binary {
                    left: Box::new(LogicalExpression::FunctionCall {
                        name: "id".into(),
                        args: vec![LogicalExpression::Variable(expand_target)],
                        distinct: false,
                    }),
                    op: BinaryOp::Eq,
                    right: Box::new(LogicalExpression::FunctionCall {
                        name: "id".into(),
                        args: vec![LogicalExpression::Variable(to_variable.clone())],
                        distinct: false,
                    }),
                },
            )
        } else {
            expand
        };

        let mut result = if let Some(label) = target_label {
            wrap_filter(
                expand,
                LogicalExpression::FunctionCall {
                    name: "hasLabel".into(),
                    args: vec![
                        LogicalExpression::Variable(to_variable.clone()),
                        LogicalExpression::Literal(Value::from(label)),
                    ],
                    distinct: false,
                },
            )
        } else {
            expand
        };

        // Apply property filters on the edge: -[r {since: 2020}]->
        if !rel.properties.is_empty()
            && let Some(ref ev) = edge_variable_for_filter
        {
            let predicate = self.build_property_predicate(ev, &rel.properties)?;
            result = wrap_filter(result, predicate);
        }

        // Apply inline WHERE clause from relationship pattern: -[r WHERE expr]->
        if let Some(where_expr) = &rel.where_clause {
            let predicate = self.translate_expression(where_expr)?;
            result = wrap_filter(result, predicate);
        }

        // Apply property filters on the target node: ()-[r]->(o {id: "X"})
        if !rel.target.properties.is_empty() {
            let predicate = self.build_property_predicate(&to_variable, &rel.target.properties)?;
            result = wrap_filter(result, predicate);
        }

        Ok(result)
    }

    fn translate_where(
        &self,
        where_clause: &ast::WhereClause,
        input: Option<LogicalOperator>,
    ) -> Result<LogicalOperator> {
        let input = input.ok_or_else(|| {
            Error::Query(QueryError::new(
                QueryErrorKind::Semantic,
                "WHERE requires input",
            ))
        })?;
        let predicate = self.translate_expression(&where_clause.predicate)?;

        // When the input is a LeftJoin (from OPTIONAL MATCH), classify the
        // predicate so right-side references become join conditions rather
        // than post-filters (which would incorrectly eliminate NULL rows).
        if let LogicalOperator::LeftJoin(left_join) = input {
            let (join, post_filter) =
                build_left_join_with_predicates(*left_join.left, *left_join.right, Some(predicate));
            if let Some(pf) = post_filter {
                Ok(wrap_filter(join, pf))
            } else {
                Ok(join)
            }
        } else {
            Ok(wrap_filter(input, predicate))
        }
    }

    fn translate_with(
        &self,
        with_clause: &ast::WithClause,
        input: Option<LogicalOperator>,
    ) -> Result<LogicalOperator> {
        // WITH can work with or without prior input (e.g., standalone WITH [1,2,3] AS nums)
        // If there's no input, use Empty which produces a single row for projection evaluation
        let input = input.unwrap_or(LogicalOperator::Empty);

        // WITH *: skip projection, all variables pass through unchanged
        if with_clause.is_wildcard {
            let mut plan = input;

            if let Some(where_clause) = &with_clause.where_clause {
                let predicate = self.translate_expression(&where_clause.predicate)?;
                plan = wrap_filter(plan, predicate);
            }

            if with_clause.distinct {
                plan = wrap_distinct(plan);
            }

            return Ok(plan);
        }

        // Check if WITH contains aggregate functions (e.g. WITH collect(n) AS people)
        let has_aggregates = with_clause
            .items
            .iter()
            .any(|item| contains_aggregate(&item.expression));

        let mut plan = if has_aggregates {
            let (aggregates, group_by, post_return) =
                self.extract_aggregates_and_groups_from_items(&with_clause.items)?;

            let agg_op = LogicalOperator::Aggregate(AggregateOp {
                group_by,
                aggregates,
                input: Box::new(input),
                having: None,
            });

            if let Some(return_items) = post_return {
                let projections = return_items
                    .into_iter()
                    .map(|item| Projection {
                        expression: item.expression,
                        alias: item.alias,
                    })
                    .collect();
                LogicalOperator::Project(ProjectOp {
                    projections,
                    input: Box::new(agg_op),
                    pass_through_input: false,
                })
            } else {
                agg_op
            }
        } else {
            let projections: Vec<Projection> = with_clause
                .items
                .iter()
                .map(|item| {
                    Ok(Projection {
                        expression: self.translate_expression(&item.expression)?,
                        alias: item.alias.clone(),
                    })
                })
                .collect::<Result<_>>()?;

            // Rewrite pattern comprehensions into Apply + Aggregate(Collect)
            let has_pattern_comp = projections.iter().any(|p| {
                matches!(
                    &p.expression,
                    LogicalExpression::PatternComprehension { .. }
                )
            });
            let (input, projections) = if has_pattern_comp {
                let items: Vec<ReturnItem> = projections
                    .into_iter()
                    .map(|p| ReturnItem {
                        expression: p.expression,
                        alias: p.alias,
                    })
                    .collect();
                let (rewritten_input, rewritten_items) =
                    self.rewrite_pattern_comprehensions(input, items)?;
                let projections = rewritten_items
                    .into_iter()
                    .map(|item| Projection {
                        expression: item.expression,
                        alias: item.alias,
                    })
                    .collect();
                (rewritten_input, projections)
            } else {
                (input, projections)
            };

            LogicalOperator::Project(ProjectOp {
                projections,
                input: Box::new(input),
                pass_through_input: false,
            })
        };

        if let Some(where_clause) = &with_clause.where_clause {
            let predicate = self.translate_expression(&where_clause.predicate)?;
            plan = wrap_filter(plan, predicate);
        }

        if with_clause.distinct {
            plan = wrap_distinct(plan);
        }

        Ok(plan)
    }

    fn translate_unwind(
        &self,
        unwind_clause: &ast::UnwindClause,
        input: Option<LogicalOperator>,
    ) -> Result<LogicalOperator> {
        // UNWIND can work with or without prior input
        // If there's no input, create an implicit single row (Empty with one result)
        let input = input.unwrap_or(LogicalOperator::Empty);

        let expression = self.translate_expression(&unwind_clause.expression)?;

        Ok(LogicalOperator::Unwind(UnwindOp {
            expression,
            variable: unwind_clause.variable.clone(),
            ordinality_var: None,
            offset_var: None,
            input: Box::new(input),
        }))
    }

    fn translate_merge_statement(&self, merge: &ast::MergeClause) -> Result<LogicalPlan> {
        let op = self.translate_merge(merge, None)?;
        Ok(LogicalPlan::new(op))
    }

    fn translate_merge(
        &self,
        merge_clause: &ast::MergeClause,
        input: Option<LogicalOperator>,
    ) -> Result<LogicalOperator> {
        let input = input.unwrap_or(LogicalOperator::Empty);
        let pattern = &merge_clause.pattern;

        // Check if this is a relationship (path) pattern
        let path = match pattern {
            ast::Pattern::Path(path) if !path.chain.is_empty() => Some(path),
            ast::Pattern::NamedPath { pattern: inner, .. } => match inner.as_ref() {
                ast::Pattern::Path(path) if !path.chain.is_empty() => Some(path),
                _ => None,
            },
            _ => None,
        };

        if let Some(path) = path {
            return self.translate_merge_relationship(path, merge_clause, input);
        }

        // Node-only MERGE
        let node = match pattern {
            ast::Pattern::Node(n) => n,
            ast::Pattern::Path(path) => &path.start,
            ast::Pattern::NamedPath { pattern: inner, .. } => match inner.as_ref() {
                ast::Pattern::Node(n) => n,
                ast::Pattern::Path(path) => &path.start,
                _ => {
                    return Err(Error::Query(QueryError::new(
                        QueryErrorKind::Semantic,
                        "MERGE NamedPath must contain a node or path",
                    )));
                }
            },
        };

        let variable = node
            .variable
            .clone()
            .unwrap_or_else(|| format!("_merge_{}", 0));
        let labels: Vec<String> = node.labels.clone();

        let match_properties: Vec<(String, LogicalExpression)> = node
            .properties
            .iter()
            .map(|(k, v)| Ok((k.clone(), self.translate_expression(v)?)))
            .collect::<Result<Vec<_>>>()?;

        let on_create: Vec<(String, LogicalExpression)> =
            if let Some(set_clause) = &merge_clause.on_create {
                self.extract_set_properties(set_clause)?
            } else {
                Vec::new()
            };

        let on_match: Vec<(String, LogicalExpression)> =
            if let Some(set_clause) = &merge_clause.on_match {
                self.extract_set_properties(set_clause)?
            } else {
                Vec::new()
            };

        Ok(LogicalOperator::Merge(MergeOp {
            variable,
            labels,
            match_properties,
            on_create,
            on_match,
            input: Box::new(input),
        }))
    }

    fn translate_merge_relationship(
        &self,
        path: &ast::PathPattern,
        merge_clause: &ast::MergeClause,
        input: LogicalOperator,
    ) -> Result<LogicalOperator> {
        let mut current_input = input;

        // Extract source node variable
        let source_variable = path.start.variable.clone().ok_or_else(|| {
            Error::Query(QueryError::new(
                QueryErrorKind::Semantic,
                "MERGE relationship pattern requires a source node variable",
            ))
        })?;

        // If source node has labels or properties, it's an inline definition:
        // emit a MergeOp to create-or-match the node first.
        if !path.start.labels.is_empty() || !path.start.properties.is_empty() {
            let node_props: Vec<(String, LogicalExpression)> = path
                .start
                .properties
                .iter()
                .map(|(k, v)| Ok((k.clone(), self.translate_expression(v)?)))
                .collect::<Result<Vec<_>>>()?;

            current_input = LogicalOperator::Merge(MergeOp {
                variable: source_variable.clone(),
                labels: path.start.labels.clone(),
                match_properties: node_props,
                on_create: Vec::new(),
                on_match: Vec::new(),
                input: Box::new(current_input),
            });
        }

        // Extract the first (and only) relationship segment
        let rel = path.chain.first().ok_or_else(|| {
            Error::Query(QueryError::new(
                QueryErrorKind::Semantic,
                "MERGE relationship pattern is empty",
            ))
        })?;

        // Extract relationship variable
        let variable = rel
            .variable
            .clone()
            .unwrap_or_else(|| "_merge_rel_0".to_string());
        self.register_edge_variable(&variable);

        // Extract relationship type
        let edge_type = rel.types.first().cloned().ok_or_else(|| {
            Error::Query(QueryError::new(
                QueryErrorKind::Semantic,
                "MERGE relationship pattern requires a relationship type",
            ))
        })?;

        // Extract target node variable
        let target_variable = rel.target.variable.clone().ok_or_else(|| {
            Error::Query(QueryError::new(
                QueryErrorKind::Semantic,
                "MERGE relationship pattern requires a target node variable",
            ))
        })?;

        // If target node has labels or properties, emit a MergeOp for it too.
        if !rel.target.labels.is_empty() || !rel.target.properties.is_empty() {
            let node_props: Vec<(String, LogicalExpression)> = rel
                .target
                .properties
                .iter()
                .map(|(k, v)| Ok((k.clone(), self.translate_expression(v)?)))
                .collect::<Result<Vec<_>>>()?;

            current_input = LogicalOperator::Merge(MergeOp {
                variable: target_variable.clone(),
                labels: rel.target.labels.clone(),
                match_properties: node_props,
                on_create: Vec::new(),
                on_match: Vec::new(),
                input: Box::new(current_input),
            });
        }

        // Extract relationship properties
        let match_properties: Vec<(String, LogicalExpression)> = rel
            .properties
            .iter()
            .map(|(k, v)| Ok((k.clone(), self.translate_expression(v)?)))
            .collect::<Result<Vec<_>>>()?;

        let on_create: Vec<(String, LogicalExpression)> =
            if let Some(set_clause) = &merge_clause.on_create {
                self.extract_set_properties(set_clause)?
            } else {
                Vec::new()
            };

        let on_match: Vec<(String, LogicalExpression)> =
            if let Some(set_clause) = &merge_clause.on_match {
                self.extract_set_properties(set_clause)?
            } else {
                Vec::new()
            };

        Ok(LogicalOperator::MergeRelationship(MergeRelationshipOp {
            variable,
            source_variable,
            target_variable,
            edge_type,
            match_properties,
            on_create,
            on_match,
            input: Box::new(current_input),
        }))
    }

    /// Extracts properties from a SET clause.
    fn extract_set_properties(
        &self,
        set_clause: &ast::SetClause,
    ) -> Result<Vec<(String, LogicalExpression)>> {
        let mut properties = Vec::new();
        for item in &set_clause.items {
            match item {
                ast::SetItem::Property {
                    variable: _,
                    property,
                    value,
                } => {
                    properties.push((property.clone(), self.translate_expression(value)?));
                }
                ast::SetItem::AllProperties {
                    variable: _,
                    properties: prop_expr,
                } => {
                    // n = {props} - extract all properties from the map
                    if let ast::Expression::Map(pairs) = prop_expr {
                        for (k, v) in pairs {
                            properties.push((k.clone(), self.translate_expression(v)?));
                        }
                    }
                }
                ast::SetItem::MergeProperties {
                    variable: _,
                    properties: prop_expr,
                } => {
                    // n += {props} - merge properties
                    if let ast::Expression::Map(pairs) = prop_expr {
                        for (k, v) in pairs {
                            properties.push((k.clone(), self.translate_expression(v)?));
                        }
                    }
                }
                ast::SetItem::Labels { .. } => {
                    // Labels are handled separately
                }
            }
        }
        Ok(properties)
    }

    fn translate_return(
        &self,
        return_clause: &ast::ReturnClause,
        input: Option<LogicalOperator>,
    ) -> Result<LogicalOperator> {
        // Standalone RETURN (e.g. RETURN 2 * 3) uses Empty as a single-row source
        let input = input.unwrap_or(LogicalOperator::Empty);

        // Record alias-to-output-column mappings for ORDER BY alias resolution.
        // For non-aggregate RETURN, output columns use aliases directly.
        // For aggregate RETURN without post_return, group-by columns use
        // expression_to_string names, and aggregate columns use their aliases.
        self.return_aliases.borrow_mut().clear();

        // Check if RETURN contains aggregate functions
        let has_aggregates = match &return_clause.items {
            ast::ReturnItems::All => false,
            ast::ReturnItems::Explicit(items) => items
                .iter()
                .any(|item| contains_aggregate(&item.expression)),
        };

        if has_aggregates {
            // Extract aggregates and group-by expressions.
            // When a return item wraps an aggregate in a binary/unary expression
            // (e.g. `count(n) > 0 AS exists`), we decompose it into:
            //   1. An aggregate (`count(n)` with synthetic alias)
            //   2. A post-aggregate projection (`_agg_0 > 0 AS exists`)
            let items = match &return_clause.items {
                ast::ReturnItems::All => {
                    return Err(Error::Query(QueryError::new(
                        QueryErrorKind::Semantic,
                        "Cannot use RETURN * with aggregates",
                    )));
                }
                ast::ReturnItems::Explicit(items) => items,
            };
            // With aliases (e.g. `n.city AS city`) the post-Return renames the
            // columns, which ORDER BY alias resolution and result naming need.
            let (aggregates, group_by, post_return) =
                self.extract_aggregates_and_groups_from_items(items)?;

            // Register aggregate output column names so ORDER BY can
            // reference them. Group-by columns use expression_to_string
            // format (e.g. "o.status"), aggregate columns use their alias.
            {
                let mut aliases = self.return_aliases.borrow_mut();
                for gb in &group_by {
                    let col = crate::query::planner::common::expression_to_string(gb);
                    aliases.insert(col.clone(), col);
                }
                for agg in &aggregates {
                    if let Some(ref alias) = agg.alias {
                        aliases.insert(alias.clone(), alias.clone());
                    }
                }
            }

            let agg_op = LogicalOperator::Aggregate(AggregateOp {
                group_by,
                aggregates,
                input: Box::new(input),
                having: None,
            });

            if let Some(return_items) = post_return {
                // Post-projection renames columns using aliases.
                // Register alias -> alias (identity) so ORDER BY resolves
                // directly since the Return outputs alias names.
                {
                    let mut aliases = self.return_aliases.borrow_mut();
                    for ri in &return_items {
                        if let Some(ref alias) = ri.alias {
                            let a = alias.clone();
                            aliases.insert(a.clone(), a);
                        }
                    }
                }
                Ok(wrap_return(agg_op, return_items, return_clause.distinct))
            } else {
                Ok(agg_op)
            }
        } else {
            // Normal return without aggregates
            let items = match &return_clause.items {
                ast::ReturnItems::All => {
                    vec![ReturnItem {
                        expression: LogicalExpression::Variable("*".into()),
                        alias: None,
                    }]
                }
                ast::ReturnItems::Explicit(items) => items
                    .iter()
                    .map(|item| {
                        Ok(ReturnItem {
                            expression: self.translate_expression(&item.expression)?,
                            alias: item.alias.clone(),
                        })
                    })
                    .collect::<Result<_>>()?,
            };

            // Rewrite pattern comprehensions into Apply + Aggregate(Collect)
            let has_pattern_comp = items.iter().any(|item| {
                matches!(
                    &item.expression,
                    LogicalExpression::PatternComprehension { .. }
                )
            });
            if has_pattern_comp {
                let (rewritten_input, rewritten_items) =
                    self.rewrite_pattern_comprehensions(input, items)?;
                Ok(wrap_return(
                    rewritten_input,
                    rewritten_items,
                    return_clause.distinct,
                ))
            } else {
                Ok(wrap_return(input, items, return_clause.distinct))
            }
        }
    }

    /// Extracts aggregate and group-by expressions from RETURN items.
    ///
    /// Returns `(aggregates, group_by, post_return)`. The Aggregate operator
    /// outputs its grouping keys first, then its aggregates, named by alias or
    /// [`aggregate_column_name`](crate::query::planner::common::aggregate_column_name).
    /// `post_return` is `Some(...)` whenever that output differs from what the
    /// items ask for: an item wraps an aggregate (`count(n) > 0 AS exists`),
    /// an item has an alias, or a grouping key follows an aggregate (so
    /// columns must be reordered). A post-aggregate projection must then be
    /// chained.
    fn extract_aggregates_and_groups_from_items(
        &self,
        items: &[ast::ProjectionItem],
    ) -> Result<(
        Vec<AggregateExpr>,
        Vec<LogicalExpression>,
        Option<Vec<ReturnItem>>,
    )> {
        let mut aggregates = Vec::new();
        let mut group_by = Vec::new();
        let mut needs_post_return = false;
        let mut post_return_items = Vec::new();
        let mut agg_counter: u32 = 0;
        let mut seen_aggregate = false;

        for item in items {
            if let Some(mut agg_expr) = self.try_extract_aggregate(&item.expression, &item.alias)? {
                // Direct aggregate (e.g. `count(n) AS cnt`). An unaliased one is
                // named after its source text (`count(n)`); the name doubles as
                // the alias so the binder resolves the post-Return reference.
                let column = agg_expr.alias.clone().unwrap_or_else(|| {
                    crate::query::planner::common::aggregate_column_name(&agg_expr)
                });
                agg_expr.alias = Some(column.clone());
                aggregates.push(agg_expr);
                post_return_items.push(ReturnItem {
                    expression: LogicalExpression::Variable(column),
                    alias: item.alias.clone(),
                });
                seen_aggregate = true;
            } else if contains_aggregate(&item.expression) {
                // Wrapped aggregate (e.g. `count(n) > 0 AS exists`)
                needs_post_return = true;
                seen_aggregate = true;

                // Extract all aggregates and build a substitute expression
                // with variable references replacing each aggregate.
                let substitute = self.extract_wrapped_aggregates(
                    &item.expression,
                    &mut agg_counter,
                    &mut aggregates,
                )?;
                // Unaliased, the column is named after the written expression
                // (`sum(n.v) + 1`), not the substitute (`_agg_0 + 1`).
                let alias = item.alias.clone().or_else(|| {
                    self.translate_expression(&item.expression)
                        .ok()
                        .map(|expr| crate::query::planner::common::expression_to_string(&expr))
                });
                post_return_items.push(ReturnItem {
                    expression: substitute,
                    alias,
                });
            } else {
                // Non-aggregate expression: group-by key. Keys come first in
                // the Aggregate's output, so a key after an aggregate needs the
                // post-return to restore the item order.
                needs_post_return |= seen_aggregate;
                let expr = self.translate_expression(&item.expression)?;
                group_by.push(expr.clone());
                // In the post-return, reference the Aggregate's output column
                // by its generated name. The Aggregate already extracts
                // property values, so we reference the column, not re-evaluate.
                let col_name = crate::query::planner::common::expression_to_string(&expr);
                post_return_items.push(ReturnItem {
                    expression: LogicalExpression::Variable(col_name),
                    alias: item.alias.clone(),
                });
            }
        }

        let has_aliases = items.iter().any(|item| item.alias.is_some());
        if needs_post_return || has_aliases {
            Ok((aggregates, group_by, Some(post_return_items)))
        } else {
            Ok((aggregates, group_by, None))
        }
    }

    /// Extracts all aggregates from inside a wrapping expression, assigning
    /// each a synthetic alias (`_agg_{counter}`), and returns the expression
    /// with every aggregate replaced by a variable reference to its alias.
    ///
    /// For `count(n) > 0`:
    /// - pushes `count(n)` with alias `_agg_0` into `aggregates_out`
    /// - returns `Variable("_agg_0") > Literal(0)`
    ///
    /// For `sum(x) / count(x)`:
    /// - pushes `sum(x)` as `_agg_0` and `count(x)` as `_agg_1`
    /// - returns `Variable("_agg_0") / Variable("_agg_1")`
    fn extract_wrapped_aggregates(
        &self,
        expr: &ast::Expression,
        agg_counter: &mut u32,
        aggregates_out: &mut Vec<AggregateExpr>,
    ) -> Result<LogicalExpression> {
        match expr {
            ast::Expression::FunctionCall { name, args, .. } => {
                // Check if the function itself is an aggregate
                let alias = format!("_agg_{agg_counter}");
                if let Some(agg) = self.try_extract_aggregate(expr, &Some(alias.clone()))? {
                    *agg_counter += 1;
                    aggregates_out.push(agg);
                    return Ok(LogicalExpression::Variable(alias));
                }
                // Non-aggregate function wrapping aggregate arguments,
                // e.g. size(collect(DISTINCT n.v)). Recurse into each argument.
                let mut translated_args = Vec::with_capacity(args.len());
                for arg in args {
                    if contains_aggregate(arg) {
                        translated_args.push(self.extract_wrapped_aggregates(
                            arg,
                            agg_counter,
                            aggregates_out,
                        )?);
                    } else {
                        translated_args.push(self.translate_expression(arg)?);
                    }
                }
                Ok(LogicalExpression::FunctionCall {
                    name: name.clone(),
                    args: translated_args,
                    distinct: false,
                })
            }
            ast::Expression::Binary { left, op, right } => {
                let binary_op = self.translate_binary_op(*op)?;
                let left_sub = if contains_aggregate(left) {
                    self.extract_wrapped_aggregates(left, agg_counter, aggregates_out)?
                } else {
                    self.translate_expression(left)?
                };
                let right_sub = if contains_aggregate(right) {
                    self.extract_wrapped_aggregates(right, agg_counter, aggregates_out)?
                } else {
                    self.translate_expression(right)?
                };
                Ok(LogicalExpression::Binary {
                    left: Box::new(left_sub),
                    op: binary_op,
                    right: Box::new(right_sub),
                })
            }
            ast::Expression::Unary { op, operand } => {
                let sub = self.extract_wrapped_aggregates(operand, agg_counter, aggregates_out)?;
                // Unary positive is identity: just return the operand
                if *op == ast::UnaryOp::Pos {
                    return Ok(sub);
                }
                let unary_op = self.translate_unary_op(*op)?;
                Ok(LogicalExpression::Unary {
                    op: unary_op,
                    operand: Box::new(sub),
                })
            }
            ast::Expression::Case {
                input,
                whens,
                else_clause,
            } => {
                let operand = match input {
                    Some(inp) if contains_aggregate(inp) => Some(Box::new(
                        self.extract_wrapped_aggregates(inp, agg_counter, aggregates_out)?,
                    )),
                    Some(inp) => Some(Box::new(self.translate_expression(inp)?)),
                    None => None,
                };
                let mut when_clauses = Vec::with_capacity(whens.len());
                for (cond, then) in whens {
                    let cond_expr = if contains_aggregate(cond) {
                        self.extract_wrapped_aggregates(cond, agg_counter, aggregates_out)?
                    } else {
                        self.translate_expression(cond)?
                    };
                    let then_expr = if contains_aggregate(then) {
                        self.extract_wrapped_aggregates(then, agg_counter, aggregates_out)?
                    } else {
                        self.translate_expression(then)?
                    };
                    when_clauses.push((cond_expr, then_expr));
                }
                let else_expr = match else_clause {
                    Some(el) if contains_aggregate(el) => Some(Box::new(
                        self.extract_wrapped_aggregates(el, agg_counter, aggregates_out)?,
                    )),
                    Some(el) => Some(Box::new(self.translate_expression(el)?)),
                    None => None,
                };
                Ok(LogicalExpression::Case {
                    operand,
                    when_clauses,
                    else_clause: else_expr,
                })
            }
            _ => Err(Error::Query(QueryError::new(
                QueryErrorKind::Semantic,
                "Unsupported expression wrapping an aggregate",
            ))),
        }
    }

    /// Tries to extract an aggregate expression from an AST expression.
    fn try_extract_aggregate(
        &self,
        expr: &ast::Expression,
        alias: &Option<String>,
    ) -> Result<Option<AggregateExpr>> {
        match expr {
            ast::Expression::FunctionCall {
                name,
                args,
                distinct,
            } => {
                if let Some(function) = to_aggregate_function(name) {
                    // count(*) is represented as FunctionCall with Variable("*") arg
                    let is_count_star = function == AggregateFunction::Count
                        && args.len() == 1
                        && matches!(&args[0], ast::Expression::Variable(v) if v == "*");
                    let expression = if args.is_empty() || is_count_star {
                        None
                    } else {
                        Some(self.translate_expression(&args[0])?)
                    };
                    // Extract percentile parameter for percentile functions
                    let percentile = if matches!(
                        function,
                        AggregateFunction::PercentileDisc | AggregateFunction::PercentileCont
                    ) && args.len() >= 2
                    {
                        // Second argument is the percentile value
                        if let ast::Expression::Literal(ast::Literal::Float(p)) = &args[1] {
                            Some((*p).clamp(0.0, 1.0))
                        } else if let ast::Expression::Literal(ast::Literal::Integer(p)) = &args[1]
                        {
                            Some((*p as f64).clamp(0.0, 1.0))
                        } else {
                            Some(0.5) // Default to median
                        }
                    } else {
                        None
                    };

                    // COUNT(expr) uses CountNonNull to skip NULLs;
                    // COUNT(*) uses Count to count all rows.
                    let function = if function == AggregateFunction::Count
                        && !is_count_star
                        && expression.is_some()
                    {
                        AggregateFunction::CountNonNull
                    } else {
                        function
                    };

                    Ok(Some(AggregateExpr {
                        function,
                        expression,
                        expression2: None,
                        distinct: *distinct,
                        alias: alias.clone(),
                        percentile,
                        separator: None,
                    }))
                } else {
                    Ok(None)
                }
            }
            _ => Ok(None),
        }
    }

    fn translate_order_by(
        &self,
        order_by: &ast::OrderByClause,
        input: Option<LogicalOperator>,
    ) -> Result<LogicalOperator> {
        let input = input.ok_or_else(|| {
            Error::Query(QueryError::new(
                QueryErrorKind::Semantic,
                "ORDER BY requires input",
            ))
        })?;

        let aliases = self.return_aliases.borrow();
        let keys: Vec<SortKey> = order_by
            .items
            .iter()
            .map(|item| {
                // Resolve alias references: if ORDER BY uses a variable
                // that matches a RETURN alias, substitute with the actual
                // output column name from the preceding RETURN/Aggregate.
                let expression = if let ast::Expression::Variable(name) = &item.expression {
                    if let Some(col_name) = aliases.get(name) {
                        LogicalExpression::Variable(col_name.clone())
                    } else {
                        self.translate_expression(&item.expression)?
                    }
                } else if let ast::Expression::PropertyAccess { base, property } = &item.expression
                {
                    // After aggregation, entity variables (o, c, d) no longer
                    // exist. Rewrite o.status to Variable("o.status") which
                    // matches the aggregate output column name.
                    if let ast::Expression::Variable(var) = base.as_ref() {
                        let col_dot = format!("{var}.{property}");
                        if aliases.get(&col_dot).is_some() {
                            LogicalExpression::Variable(col_dot)
                        } else {
                            self.translate_expression(&item.expression)?
                        }
                    } else {
                        self.translate_expression(&item.expression)?
                    }
                } else {
                    self.translate_expression(&item.expression)?
                };
                Ok(SortKey {
                    expression,
                    order: match item.direction {
                        ast::SortDirection::Asc => SortOrder::Ascending,
                        ast::SortDirection::Desc => SortOrder::Descending,
                    },
                    nulls: None,
                })
            })
            .collect::<Result<_>>()?;

        Ok(wrap_sort(input, keys))
    }

    fn translate_skip(
        &self,
        expr: &ast::Expression,
        input: Option<LogicalOperator>,
    ) -> Result<LogicalOperator> {
        let input = input.ok_or_else(|| {
            Error::Query(QueryError::new(
                QueryErrorKind::Semantic,
                "SKIP requires input",
            ))
        })?;
        let count = self.eval_as_count_expr(expr)?;

        Ok(wrap_skip(input, count))
    }

    fn translate_limit(
        &self,
        expr: &ast::Expression,
        input: Option<LogicalOperator>,
    ) -> Result<LogicalOperator> {
        let input = input.ok_or_else(|| {
            Error::Query(QueryError::new(
                QueryErrorKind::Semantic,
                "LIMIT requires input",
            ))
        })?;
        let count = self.eval_as_count_expr(expr)?;

        Ok(wrap_limit(input, count))
    }

    fn translate_create_clause(
        &self,
        create_clause: &ast::CreateClause,
        input: Option<LogicalOperator>,
    ) -> Result<LogicalOperator> {
        let mut plan = input;

        for pattern in &create_clause.patterns {
            plan = Some(self.translate_create_pattern(pattern, plan)?);
        }

        plan.ok_or_else(|| {
            Error::Query(QueryError::new(
                QueryErrorKind::Semantic,
                "Empty CREATE pattern",
            ))
        })
    }

    fn translate_create_pattern(
        &self,
        pattern: &ast::Pattern,
        input: Option<LogicalOperator>,
    ) -> Result<LogicalOperator> {
        match pattern {
            ast::Pattern::Node(node) => {
                let variable = node
                    .variable
                    .clone()
                    .unwrap_or_else(|| self.next_anon_var());
                let labels = node.labels.clone();
                let properties: Vec<(String, LogicalExpression)> = node
                    .properties
                    .iter()
                    .map(|(k, v)| Ok((k.clone(), self.translate_expression(v)?)))
                    .collect::<Result<_>>()?;

                Ok(LogicalOperator::CreateNode(CreateNodeOp {
                    variable,
                    labels,
                    properties,
                    input: input.map(Box::new),
                }))
            }
            ast::Pattern::Path(path) => {
                let mut current =
                    self.translate_create_pattern(&ast::Pattern::Node(path.start.clone()), input)?;

                for rel in &path.chain {
                    let from_variable = self.get_last_node_variable(&Some(current.clone()))?;
                    let to_variable = rel
                        .target
                        .variable
                        .clone()
                        .unwrap_or_else(|| self.next_anon_var());
                    let edge_type = rel
                        .types
                        .first()
                        .cloned()
                        .unwrap_or_else(|| "RELATED".to_string());

                    let target_labels = rel.target.labels.clone();
                    let target_props: Vec<(String, LogicalExpression)> = rel
                        .target
                        .properties
                        .iter()
                        .map(|(k, v)| Ok((k.clone(), self.translate_expression(v)?)))
                        .collect::<Result<_>>()?;

                    current = LogicalOperator::CreateNode(CreateNodeOp {
                        variable: to_variable.clone(),
                        labels: target_labels,
                        properties: target_props,
                        input: Some(Box::new(current)),
                    });

                    let edge_props: Vec<(String, LogicalExpression)> = rel
                        .properties
                        .iter()
                        .map(|(k, v)| Ok((k.clone(), self.translate_expression(v)?)))
                        .collect::<Result<_>>()?;

                    current = LogicalOperator::CreateEdge(CreateEdgeOp {
                        variable: rel.variable.clone(),
                        from_variable,
                        to_variable,
                        edge_type,
                        properties: edge_props,
                        input: Box::new(current),
                    });
                }

                Ok(current)
            }
            ast::Pattern::NamedPath { pattern, .. } => {
                self.translate_create_pattern(pattern, input)
            }
        }
    }

    fn translate_create_statement(&self, create: &ast::CreateClause) -> Result<LogicalPlan> {
        let mut plan: Option<LogicalOperator> = None;

        for pattern in &create.patterns {
            plan = Some(self.translate_create_pattern(pattern, plan)?);
        }

        let root = plan.ok_or_else(|| {
            Error::Query(QueryError::new(QueryErrorKind::Semantic, "Empty CREATE"))
        })?;
        Ok(LogicalPlan::new(root))
    }

    fn translate_delete(
        &self,
        delete_clause: &ast::DeleteClause,
        input: Option<LogicalOperator>,
    ) -> Result<LogicalOperator> {
        let input = input.ok_or_else(|| {
            Error::Query(QueryError::new(
                QueryErrorKind::Semantic,
                "DELETE requires input",
            ))
        })?;

        let mut plan = input;

        // Delete each expression (typically variables)
        for expr in &delete_clause.expressions {
            if let ast::Expression::Variable(var) = expr {
                if self.is_edge_variable(var) {
                    plan = LogicalOperator::DeleteEdge(DeleteEdgeOp {
                        variable: var.clone(),
                        input: Box::new(plan),
                    });
                } else {
                    plan = LogicalOperator::DeleteNode(DeleteNodeOp {
                        variable: var.clone(),
                        detach: delete_clause.detach,
                        input: Box::new(plan),
                    });
                }
            } else {
                return Err(Error::Query(QueryError::new(
                    QueryErrorKind::Semantic,
                    "DELETE only supports variable expressions",
                )));
            }
        }

        Ok(plan)
    }

    fn translate_set(
        &self,
        set_clause: &ast::SetClause,
        input: Option<LogicalOperator>,
    ) -> Result<LogicalOperator> {
        let input = input.ok_or_else(|| {
            Error::Query(QueryError::new(
                QueryErrorKind::Semantic,
                "SET requires input",
            ))
        })?;

        let mut plan = input;

        // Group items by variable
        for item in &set_clause.items {
            match item {
                ast::SetItem::Property {
                    variable,
                    property,
                    value,
                } => {
                    // SET n.prop = value
                    let value_expr = self.translate_expression(value)?;
                    plan = LogicalOperator::SetProperty(SetPropertyOp {
                        variable: variable.clone(),
                        properties: vec![(property.clone(), value_expr)],
                        replace: false,
                        is_edge: self.is_edge_variable(variable),
                        input: Box::new(plan),
                    });
                }
                ast::SetItem::AllProperties {
                    variable,
                    properties,
                } => {
                    // SET n = {...} or SET n = m
                    let value_expr = self.translate_expression(properties)?;
                    plan = LogicalOperator::SetProperty(SetPropertyOp {
                        variable: variable.clone(),
                        properties: vec![("*".to_string(), value_expr)],
                        replace: true,
                        is_edge: self.is_edge_variable(variable),
                        input: Box::new(plan),
                    });
                }
                ast::SetItem::MergeProperties {
                    variable,
                    properties,
                } => {
                    // SET n += {...}
                    let value_expr = self.translate_expression(properties)?;
                    plan = LogicalOperator::SetProperty(SetPropertyOp {
                        variable: variable.clone(),
                        properties: vec![("*".to_string(), value_expr)],
                        replace: false,
                        is_edge: self.is_edge_variable(variable),
                        input: Box::new(plan),
                    });
                }
                ast::SetItem::Labels { variable, labels } => {
                    // SET n:Label1:Label2 adds labels to the node
                    plan = LogicalOperator::AddLabel(AddLabelOp {
                        variable: variable.clone(),
                        labels: labels.clone(),
                        input: Box::new(plan),
                    });
                }
            }
        }

        Ok(plan)
    }

    fn translate_remove(
        &self,
        remove_clause: &ast::RemoveClause,
        input: Option<LogicalOperator>,
    ) -> Result<LogicalOperator> {
        let input = input.ok_or_else(|| {
            Error::Query(QueryError::new(
                QueryErrorKind::Semantic,
                "REMOVE requires input",
            ))
        })?;

        let mut plan = input;

        for item in &remove_clause.items {
            match item {
                ast::RemoveItem::Property { variable, property } => {
                    // REMOVE n.prop sets the property to null
                    plan = LogicalOperator::SetProperty(SetPropertyOp {
                        variable: variable.clone(),
                        properties: vec![(
                            property.clone(),
                            LogicalExpression::Literal(Value::Null),
                        )],
                        replace: false,
                        is_edge: self.is_edge_variable(variable),
                        input: Box::new(plan),
                    });
                }
                ast::RemoveItem::Labels { variable, labels } => {
                    // REMOVE n:Label removes labels from the node
                    plan = LogicalOperator::RemoveLabel(RemoveLabelOp {
                        variable: variable.clone(),
                        labels: labels.clone(),
                        input: Box::new(plan),
                    });
                }
            }
        }

        Ok(plan)
    }

    fn translate_expression(&self, expr: &ast::Expression) -> Result<LogicalExpression> {
        match expr {
            ast::Expression::Literal(lit) => self.translate_literal(lit),
            ast::Expression::Variable(name) => Ok(LogicalExpression::Variable(name.clone())),
            ast::Expression::Parameter(name) => Ok(LogicalExpression::Parameter(name.clone())),
            ast::Expression::PropertyAccess { base, property } => {
                if let ast::Expression::Variable(var) = base.as_ref() {
                    Ok(LogicalExpression::Property {
                        variable: var.clone(),
                        property: property.clone(),
                    })
                } else {
                    Err(Error::Query(QueryError::new(
                        QueryErrorKind::Semantic,
                        "Nested property access not supported",
                    )))
                }
            }
            ast::Expression::IndexAccess { base, index } => {
                let base_expr = self.translate_expression(base)?;
                let index_expr = self.translate_expression(index)?;
                Ok(LogicalExpression::IndexAccess {
                    base: Box::new(base_expr),
                    index: Box::new(index_expr),
                })
            }
            ast::Expression::SliceAccess { base, start, end } => {
                let base_expr = self.translate_expression(base)?;
                let start_expr = start
                    .as_ref()
                    .map(|s| self.translate_expression(s))
                    .transpose()?
                    .map(Box::new);
                let end_expr = end
                    .as_ref()
                    .map(|e| self.translate_expression(e))
                    .transpose()?
                    .map(Box::new);
                Ok(LogicalExpression::SliceAccess {
                    base: Box::new(base_expr),
                    start: start_expr,
                    end: end_expr,
                })
            }
            ast::Expression::Binary { left, op, right } => {
                let left_expr = self.translate_expression(left)?;
                let right_expr = self.translate_expression(right)?;
                let binary_op = self.translate_binary_op(*op)?;

                Ok(LogicalExpression::Binary {
                    left: Box::new(left_expr),
                    op: binary_op,
                    right: Box::new(right_expr),
                })
            }
            ast::Expression::Unary { op, operand } => {
                let operand_expr = self.translate_expression(operand)?;
                // Unary positive is identity: just return the operand
                if *op == ast::UnaryOp::Pos {
                    return Ok(operand_expr);
                }
                let unary_op = self.translate_unary_op(*op)?;

                Ok(LogicalExpression::Unary {
                    op: unary_op,
                    operand: Box::new(operand_expr),
                })
            }
            ast::Expression::FunctionCall { name, args, .. } => {
                // Special handling for length() on path variables
                // When length(p) is called where p is a path alias, we convert it
                // to a variable reference to the path length column
                if name.to_lowercase() == "length"
                    && args.len() == 1
                    && let ast::Expression::Variable(var_name) = &args[0]
                {
                    // Check if this looks like a path variable
                    // Path lengths are stored in columns named _path_length_{alias}
                    return Ok(LogicalExpression::Variable(format!(
                        "_path_length_{}",
                        var_name
                    )));
                }

                let translated_args: Vec<LogicalExpression> = args
                    .iter()
                    .map(|a| self.translate_expression(a))
                    .collect::<Result<_>>()?;

                Ok(LogicalExpression::FunctionCall {
                    name: name.clone(),
                    args: translated_args,
                    distinct: false,
                })
            }
            ast::Expression::List(items) => {
                let translated: Vec<LogicalExpression> = items
                    .iter()
                    .map(|i| self.translate_expression(i))
                    .collect::<Result<_>>()?;

                Ok(LogicalExpression::List(translated))
            }
            ast::Expression::Map(pairs) => {
                let translated: Vec<(String, LogicalExpression)> = pairs
                    .iter()
                    .map(|(k, v)| Ok((k.clone(), self.translate_expression(v)?)))
                    .collect::<Result<_>>()?;
                Ok(LogicalExpression::Map(translated))
            }
            ast::Expression::Case {
                input,
                whens,
                else_clause,
            } => {
                let translated_operand = if let Some(op) = input {
                    Some(Box::new(self.translate_expression(op)?))
                } else {
                    None
                };

                let translated_when: Vec<(LogicalExpression, LogicalExpression)> = whens
                    .iter()
                    .map(|(when, then)| {
                        Ok((
                            self.translate_expression(when)?,
                            self.translate_expression(then)?,
                        ))
                    })
                    .collect::<Result<_>>()?;

                let translated_else = if let Some(el) = else_clause {
                    Some(Box::new(self.translate_expression(el)?))
                } else {
                    None
                };

                Ok(LogicalExpression::Case {
                    operand: translated_operand,
                    when_clauses: translated_when,
                    else_clause: translated_else,
                })
            }
            ast::Expression::ListComprehension {
                variable,
                list,
                filter,
                projection,
            } => {
                let list_expr = self.translate_expression(list)?;
                let filter_expr = filter
                    .as_ref()
                    .map(|f| self.translate_expression(f))
                    .transpose()?
                    .map(Box::new);
                // If no projection, use the variable itself as the map expression
                let map_expr = if let Some(proj) = projection {
                    self.translate_expression(proj)?
                } else {
                    LogicalExpression::Variable(variable.clone())
                };

                Ok(LogicalExpression::ListComprehension {
                    variable: variable.clone(),
                    list_expr: Box::new(list_expr),
                    filter_expr,
                    map_expr: Box::new(map_expr),
                })
            }
            ast::Expression::ListPredicate {
                kind,
                variable,
                list,
                predicate,
            } => {
                let ir_kind = match kind {
                    ast::ListPredicateKind::All => ListPredicateKind::All,
                    ast::ListPredicateKind::Any => ListPredicateKind::Any,
                    ast::ListPredicateKind::None => ListPredicateKind::None,
                    ast::ListPredicateKind::Single => ListPredicateKind::Single,
                };
                Ok(LogicalExpression::ListPredicate {
                    kind: ir_kind,
                    variable: variable.clone(),
                    list_expr: Box::new(self.translate_expression(list)?),
                    predicate: Box::new(self.translate_expression(predicate)?),
                })
            }
            ast::Expression::PatternComprehension {
                pattern,
                where_clause,
                projection,
            } => {
                // Build a subplan from the pattern
                let pattern_plan = self.translate_pattern(pattern, None)?;
                // Apply optional WHERE filter
                let subplan = if let Some(where_expr) = where_clause {
                    let pred = self.translate_expression(where_expr)?;
                    wrap_filter(pattern_plan, pred)
                } else {
                    pattern_plan
                };
                let proj = self.translate_expression(projection)?;
                Ok(LogicalExpression::PatternComprehension {
                    subplan: Box::new(subplan),
                    projection: Box::new(proj),
                })
            }
            ast::Expression::MapProjection { base, entries } => {
                let ir_entries = entries
                    .iter()
                    .map(|entry| match entry {
                        ast::MapProjectionEntry::PropertySelector(name) => {
                            Ok(MapProjectionEntry::PropertySelector(name.clone()))
                        }
                        ast::MapProjectionEntry::LiteralEntry(key, expr) => {
                            let translated = self.translate_expression(expr)?;
                            Ok(MapProjectionEntry::LiteralEntry(key.clone(), translated))
                        }
                        ast::MapProjectionEntry::AllProperties => {
                            Ok(MapProjectionEntry::AllProperties)
                        }
                    })
                    .collect::<Result<Vec<_>>>()?;
                Ok(LogicalExpression::MapProjection {
                    base: base.clone(),
                    entries: ir_entries,
                })
            }
            ast::Expression::Reduce {
                accumulator,
                initial,
                variable,
                list,
                expression,
            } => Ok(LogicalExpression::Reduce {
                accumulator: accumulator.clone(),
                initial: Box::new(self.translate_expression(initial)?),
                variable: variable.clone(),
                list: Box::new(self.translate_expression(list)?),
                expression: Box::new(self.translate_expression(expression)?),
            }),
            ast::Expression::Exists(inner_query) => {
                let inner_plan = self.translate_exists_subquery(inner_query)?;
                Ok(LogicalExpression::ExistsSubquery(Box::new(inner_plan)))
            }
            ast::Expression::CountSubquery(inner_query) => {
                let inner_plan = self.translate_exists_subquery(inner_query)?;
                Ok(LogicalExpression::CountSubquery(Box::new(inner_plan)))
            }
        }
    }

    /// Translates the inner query of an EXISTS subquery to a `LogicalOperator`.
    fn translate_exists_subquery(&self, query: &ast::Query) -> Result<LogicalOperator> {
        let mut plan: Option<LogicalOperator> = None;

        for clause in &query.clauses {
            match clause {
                ast::Clause::Match(m) => {
                    plan = Some(self.translate_match(m, plan)?);
                }
                ast::Clause::Where(w) => {
                    plan = Some(self.translate_where(w, plan)?);
                }
                _ => {
                    return Err(Error::Query(QueryError::new(
                        QueryErrorKind::Semantic,
                        "EXISTS subquery only supports MATCH and WHERE clauses",
                    )));
                }
            }
        }

        plan.ok_or_else(|| {
            Error::Query(QueryError::new(
                QueryErrorKind::Semantic,
                "EXISTS subquery requires at least one MATCH clause",
            ))
        })
    }

    fn translate_literal(&self, lit: &ast::Literal) -> Result<LogicalExpression> {
        let value = match lit {
            ast::Literal::Null => Value::Null,
            ast::Literal::Bool(b) => Value::Bool(*b),
            ast::Literal::Integer(i) => Value::Int64(*i),
            ast::Literal::Float(f) => Value::Float64(*f),
            ast::Literal::String(s) => Value::from(s.as_str()),
        };
        Ok(LogicalExpression::Literal(value))
    }

    fn translate_binary_op(&self, op: ast::BinaryOp) -> Result<BinaryOp> {
        Ok(match op {
            ast::BinaryOp::Eq => BinaryOp::Eq,
            ast::BinaryOp::Ne => BinaryOp::Ne,
            ast::BinaryOp::Lt => BinaryOp::Lt,
            ast::BinaryOp::Le => BinaryOp::Le,
            ast::BinaryOp::Gt => BinaryOp::Gt,
            ast::BinaryOp::Ge => BinaryOp::Ge,
            ast::BinaryOp::And => BinaryOp::And,
            ast::BinaryOp::Or => BinaryOp::Or,
            ast::BinaryOp::Xor => BinaryOp::Xor,
            ast::BinaryOp::Add => BinaryOp::Add,
            ast::BinaryOp::Sub => BinaryOp::Sub,
            ast::BinaryOp::Mul => BinaryOp::Mul,
            ast::BinaryOp::Div => BinaryOp::Div,
            ast::BinaryOp::Mod => BinaryOp::Mod,
            ast::BinaryOp::Pow => BinaryOp::Pow,
            ast::BinaryOp::Concat => BinaryOp::Concat,
            ast::BinaryOp::StartsWith => BinaryOp::StartsWith,
            ast::BinaryOp::EndsWith => BinaryOp::EndsWith,
            ast::BinaryOp::Contains => BinaryOp::Contains,
            ast::BinaryOp::RegexMatch => BinaryOp::Regex,
            ast::BinaryOp::In => BinaryOp::In,
        })
    }

    fn translate_unary_op(&self, op: ast::UnaryOp) -> Result<UnaryOp> {
        Ok(match op {
            ast::UnaryOp::Not => UnaryOp::Not,
            ast::UnaryOp::Neg => UnaryOp::Neg,
            ast::UnaryOp::Pos => {
                return Err(Error::Query(QueryError::new(
                    QueryErrorKind::Semantic,
                    "Unary positive not yet supported",
                )));
            }
            ast::UnaryOp::IsNull => UnaryOp::IsNull,
            ast::UnaryOp::IsNotNull => UnaryOp::IsNotNull,
        })
    }

    fn eval_as_count_expr(&self, expr: &ast::Expression) -> Result<CountExpr> {
        match expr {
            ast::Expression::Literal(ast::Literal::Integer(i)) => {
                // Clamp negative values to 0 (LIMIT -1 returns empty, not an error)
                // reason: clamped to >= 0 by .max(0)
                #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
                let n = (*i).max(0) as usize;
                Ok(CountExpr::Literal(n))
            }
            ast::Expression::Parameter(name) => Ok(CountExpr::Parameter(name.clone())),
            _ => Err(Error::Query(QueryError::new(
                QueryErrorKind::Semantic,
                "Expected integer literal or parameter for SKIP/LIMIT",
            ))),
        }
    }

    fn get_last_variable(plan: &LogicalOperator) -> Result<String> {
        match plan {
            LogicalOperator::NodeScan(scan) => Ok(scan.variable.clone()),
            LogicalOperator::Expand(expand) => Ok(expand.to_variable.clone()),
            LogicalOperator::Filter(filter) => Self::get_last_variable(&filter.input),
            LogicalOperator::Project(project) => Self::get_last_variable(&project.input),
            _ => Err(Error::Query(QueryError::new(
                QueryErrorKind::Semantic,
                "Cannot get variable from operator",
            ))),
        }
    }

    fn get_last_node_variable(&self, plan: &Option<LogicalOperator>) -> Result<String> {
        match plan {
            Some(LogicalOperator::CreateNode(node)) => Ok(node.variable.clone()),
            Some(LogicalOperator::NodeScan(scan)) => Ok(scan.variable.clone()),
            Some(LogicalOperator::CreateEdge(edge)) => Ok(edge.to_variable.clone()),
            Some(other) => self.get_last_node_variable(&self.extract_input(other)),
            None => Err(Error::Query(QueryError::new(
                QueryErrorKind::Semantic,
                "No previous node variable",
            ))),
        }
    }

    fn extract_input(&self, plan: &LogicalOperator) -> Option<LogicalOperator> {
        match plan {
            LogicalOperator::CreateNode(n) => n.input.as_ref().map(|b| b.as_ref().clone()),
            LogicalOperator::CreateEdge(e) => Some(e.input.as_ref().clone()),
            LogicalOperator::Filter(f) => Some(f.input.as_ref().clone()),
            _ => None,
        }
    }

    // ========================================================================
    // Pattern comprehension rewrite helpers
    // ========================================================================

    /// Extracts the anchor (start) variable from a pattern subplan.
    ///
    /// Walks down the operator tree following the input chain to find
    /// the leaf `NodeScan`, returning its variable name. This is the
    /// variable that needs to be imported from the outer scope.
    fn extract_anchor_variable(op: &LogicalOperator) -> Option<String> {
        match op {
            LogicalOperator::NodeScan(scan) if scan.input.is_none() => Some(scan.variable.clone()),
            LogicalOperator::NodeScan(scan) => Self::extract_anchor_variable(scan.input.as_ref()?),
            LogicalOperator::Expand(expand) => Self::extract_anchor_variable(&expand.input),
            LogicalOperator::Filter(filter) => Self::extract_anchor_variable(&filter.input),
            _ => None,
        }
    }

    /// Replaces the leaf `NodeScan(variable)` in a pattern subplan with
    /// `ParameterScan(columns: [variable])`, for correlated execution.
    fn replace_anchor_with_parameter_scan(op: LogicalOperator, anchor: &str) -> LogicalOperator {
        match op {
            LogicalOperator::NodeScan(scan) if scan.variable == anchor && scan.input.is_none() => {
                LogicalOperator::ParameterScan(ParameterScanOp {
                    columns: vec![anchor.to_string()],
                })
            }
            LogicalOperator::Expand(mut expand) => {
                let new_input = Self::replace_anchor_with_parameter_scan(*expand.input, anchor);
                expand.input = Box::new(new_input);
                LogicalOperator::Expand(expand)
            }
            LogicalOperator::Filter(mut filter) => {
                let new_input = Self::replace_anchor_with_parameter_scan(*filter.input, anchor);
                filter.input = Box::new(new_input);
                LogicalOperator::Filter(filter)
            }
            other => other,
        }
    }

    /// Rewrites pattern comprehensions in return items into Apply + Aggregate.
    ///
    /// For each `PatternComprehension` found in the items:
    /// 1. Extracts the anchor variable from the subplan
    /// 2. Replaces the leaf NodeScan with ParameterScan
    /// 3. Wraps the subplan in `Aggregate(collect(projection) AS alias)`
    /// 4. Wraps the current input in `Apply(shared_variables: [anchor])`
    /// 5. Replaces the expression with `Variable(alias)`
    fn rewrite_pattern_comprehensions(
        &self,
        input: LogicalOperator,
        items: Vec<ReturnItem>,
    ) -> Result<(LogicalOperator, Vec<ReturnItem>)> {
        let mut current_input = input;
        let mut rewritten_items = Vec::with_capacity(items.len());

        for item in items {
            if let LogicalExpression::PatternComprehension {
                ref subplan,
                ref projection,
            } = item.expression
            {
                // 1. Extract anchor variable
                let anchor = Self::extract_anchor_variable(subplan).ok_or_else(|| {
                    Error::Query(QueryError::new(
                        QueryErrorKind::Semantic,
                        "Pattern comprehension must start with a node pattern",
                    ))
                })?;

                // 2. Generate alias for the collected list
                let alias = item.alias.clone().unwrap_or_else(|| self.next_anon_var());

                // 3. Replace anchor NodeScan with ParameterScan
                let rewritten_subplan =
                    Self::replace_anchor_with_parameter_scan(*subplan.clone(), &anchor);

                // 4. Wrap in Aggregate(collect(projection) AS alias)
                let inner_plan = LogicalOperator::Aggregate(AggregateOp {
                    group_by: vec![],
                    aggregates: vec![AggregateExpr {
                        function: AggregateFunction::Collect,
                        expression: Some(*projection.clone()),
                        expression2: None,
                        distinct: false,
                        alias: Some(alias.clone()),
                        percentile: None,
                        separator: None,
                    }],
                    input: Box::new(rewritten_subplan),
                    having: None,
                });

                // 5. Wrap outer input in Apply
                current_input = LogicalOperator::Apply(ApplyOp {
                    input: Box::new(current_input),
                    subplan: Box::new(inner_plan),
                    shared_variables: vec![anchor],
                    optional: false,
                });

                // 6. Replace expression with Variable reference
                rewritten_items.push(ReturnItem {
                    expression: LogicalExpression::Variable(alias.clone()),
                    alias: Some(alias),
                });
            } else {
                rewritten_items.push(item);
            }
        }

        Ok((current_input, rewritten_items))
    }
}

/// Checks if an AST expression contains an aggregate function call.
fn contains_aggregate(expr: &ast::Expression) -> bool {
    match expr {
        ast::Expression::FunctionCall { name, args, .. } => {
            is_aggregate_function(name) || args.iter().any(contains_aggregate)
        }
        ast::Expression::Binary { left, right, .. } => {
            contains_aggregate(left) || contains_aggregate(right)
        }
        ast::Expression::Unary { operand, .. } => contains_aggregate(operand),
        ast::Expression::Case {
            input,
            whens,
            else_clause,
        } => {
            input.as_deref().is_some_and(contains_aggregate)
                || whens
                    .iter()
                    .any(|(w, t)| contains_aggregate(w) || contains_aggregate(t))
                || else_clause.as_deref().is_some_and(contains_aggregate)
        }
        ast::Expression::List(items) => items.iter().any(contains_aggregate),
        ast::Expression::ListComprehension {
            filter, projection, ..
        } => {
            filter.as_deref().is_some_and(contains_aggregate)
                || projection.as_deref().is_some_and(contains_aggregate)
        }
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::query::plan::{FilterOp, LimitOp, SkipOp, SortOp};

    // === Basic MATCH Tests ===

    #[test]
    fn test_translate_simple_match() {
        let plan = translate("MATCH (n:Person) RETURN n").unwrap();

        if let LogicalOperator::Return(ret) = &plan.root {
            assert_eq!(ret.items.len(), 1);
            if let LogicalOperator::NodeScan(scan) = ret.input.as_ref() {
                assert_eq!(scan.variable, "n");
                assert_eq!(scan.label, Some("Person".into()));
            } else {
                panic!("Expected NodeScan");
            }
        } else {
            panic!("Expected Return");
        }
    }

    #[test]
    fn test_translate_match_with_where() {
        let plan = translate("MATCH (n:Person) WHERE n.age > 30 RETURN n").unwrap();

        if let LogicalOperator::Return(ret) = &plan.root {
            if let LogicalOperator::Filter(filter) = ret.input.as_ref() {
                if let LogicalExpression::Binary { op, .. } = &filter.predicate {
                    assert_eq!(*op, BinaryOp::Gt);
                }
            } else {
                panic!("Expected Filter");
            }
        } else {
            panic!("Expected Return");
        }
    }

    #[test]
    fn test_translate_match_return_distinct() {
        let plan = translate("MATCH (n:Person) RETURN DISTINCT n.name").unwrap();

        if let LogicalOperator::Return(ret) = &plan.root {
            assert!(ret.distinct);
        } else {
            panic!("Expected Return");
        }
    }

    #[test]
    fn test_translate_match_return_all() {
        let plan = translate("MATCH (n:Person) RETURN *").unwrap();

        if let LogicalOperator::Return(ret) = &plan.root {
            assert_eq!(ret.items.len(), 1);
            if let LogicalExpression::Variable(v) = &ret.items[0].expression {
                assert_eq!(v, "*");
            }
        } else {
            panic!("Expected Return");
        }
    }

    // === Path Pattern Tests ===

    #[test]
    fn test_translate_outgoing_relationship() {
        let plan = translate("MATCH (a:Person)-[:KNOWS]->(b:Person) RETURN a, b").unwrap();

        // Find the Expand operator
        fn find_expand(op: &LogicalOperator) -> Option<&ExpandOp> {
            match op {
                LogicalOperator::Expand(e) => Some(e),
                LogicalOperator::Return(r) => find_expand(&r.input),
                LogicalOperator::Filter(f) => find_expand(&f.input),
                _ => None,
            }
        }

        let expand = find_expand(&plan.root).expect("Expected Expand");
        assert_eq!(expand.direction, ExpandDirection::Outgoing);
        assert_eq!(expand.edge_types, vec!["KNOWS".to_string()]);
    }

    #[test]
    fn test_translate_incoming_relationship() {
        let plan = translate("MATCH (a:Person)<-[:KNOWS]-(b:Person) RETURN a, b").unwrap();

        fn find_expand(op: &LogicalOperator) -> Option<&ExpandOp> {
            match op {
                LogicalOperator::Expand(e) => Some(e),
                LogicalOperator::Return(r) => find_expand(&r.input),
                LogicalOperator::Filter(f) => find_expand(&f.input),
                _ => None,
            }
        }

        let expand = find_expand(&plan.root).expect("Expected Expand");
        assert_eq!(expand.direction, ExpandDirection::Incoming);
    }

    #[test]
    fn test_translate_variable_length_path() {
        let plan = translate("MATCH (a:Person)-[:KNOWS*1..3]->(b:Person) RETURN a, b").unwrap();

        fn find_expand(op: &LogicalOperator) -> Option<&ExpandOp> {
            match op {
                LogicalOperator::Expand(e) => Some(e),
                LogicalOperator::Return(r) => find_expand(&r.input),
                LogicalOperator::Filter(f) => find_expand(&f.input),
                _ => None,
            }
        }

        let expand = find_expand(&plan.root).expect("Expected Expand");
        assert_eq!(expand.min_hops, 1);
        assert_eq!(expand.max_hops, Some(3));
    }

    // === Mutation Tests ===

    #[test]
    fn test_translate_create_node() {
        let plan = translate("CREATE (n:Person {name: 'Alix'})").unwrap();

        if let LogicalOperator::CreateNode(create) = &plan.root {
            assert_eq!(create.variable, "n");
            assert_eq!(create.labels, vec!["Person".to_string()]);
            assert_eq!(create.properties.len(), 1);
            assert_eq!(create.properties[0].0, "name");
        } else {
            panic!("Expected CreateNode, got {:?}", plan.root);
        }
    }

    #[test]
    fn test_translate_create_path() {
        let plan = translate("CREATE (a:Person)-[:KNOWS]->(b:Person)").unwrap();

        // Should have CreateEdge at root
        if let LogicalOperator::CreateEdge(edge) = &plan.root {
            assert_eq!(edge.edge_type, "KNOWS");
            // Input should be CreateNode for b
            if let LogicalOperator::CreateNode(node_b) = edge.input.as_ref() {
                assert_eq!(node_b.variable, "b");
            }
        } else {
            panic!("Expected CreateEdge, got {:?}", plan.root);
        }
    }

    #[test]
    fn test_translate_delete_node() {
        let plan = translate("MATCH (n:Person) DELETE n").unwrap();

        if let LogicalOperator::DeleteNode(delete) = &plan.root {
            assert_eq!(delete.variable, "n");
            if let LogicalOperator::NodeScan(scan) = delete.input.as_ref() {
                assert_eq!(scan.variable, "n");
                assert_eq!(scan.label, Some("Person".into()));
            } else {
                panic!("Expected NodeScan input");
            }
        } else {
            panic!("Expected DeleteNode, got {:?}", plan.root);
        }
    }

    #[test]
    fn test_translate_set_property() {
        let plan = translate("MATCH (n:Person) SET n.name = 'Gus' RETURN n").unwrap();

        if let LogicalOperator::Return(ret) = &plan.root {
            if let LogicalOperator::SetProperty(set) = ret.input.as_ref() {
                assert_eq!(set.variable, "n");
                assert_eq!(set.properties.len(), 1);
                assert_eq!(set.properties[0].0, "name");
                assert!(!set.replace);
            } else {
                panic!("Expected SetProperty");
            }
        } else {
            panic!("Expected Return, got {:?}", plan.root);
        }
    }

    #[test]
    fn test_translate_set_multiple_properties() {
        let plan = translate("MATCH (n:Person) SET n.name = 'Alix', n.age = 30 RETURN n").unwrap();

        if let LogicalOperator::Return(ret) = &plan.root {
            // SET creates chained SetProperty operators
            if let LogicalOperator::SetProperty(set2) = ret.input.as_ref() {
                if let LogicalOperator::SetProperty(set1) = set2.input.as_ref() {
                    // Properties are set in order
                    assert_eq!(set1.properties[0].0, "name");
                    assert_eq!(set2.properties[0].0, "age");
                } else {
                    panic!("Expected nested SetProperty");
                }
            } else {
                panic!("Expected SetProperty");
            }
        } else {
            panic!("Expected Return");
        }
    }

    #[test]
    fn test_translate_remove_property() {
        let plan = translate("MATCH (n:Person) REMOVE n.name RETURN n").unwrap();

        if let LogicalOperator::Return(ret) = &plan.root {
            if let LogicalOperator::SetProperty(set) = ret.input.as_ref() {
                // REMOVE property is translated to SET property = null
                assert_eq!(set.variable, "n");
                assert_eq!(set.properties.len(), 1);
                assert_eq!(set.properties[0].0, "name");
                // Value should be Null
                if let LogicalExpression::Literal(Value::Null) = &set.properties[0].1 {
                    // OK
                } else {
                    panic!("Expected Null value for REMOVE");
                }
            } else {
                panic!("Expected SetProperty");
            }
        } else {
            panic!("Expected Return, got {:?}", plan.root);
        }
    }

    #[test]
    fn test_translate_remove_label() {
        let plan = translate("MATCH (n:Person:Admin) REMOVE n:Admin RETURN n").unwrap();

        if let LogicalOperator::Return(ret) = &plan.root {
            if let LogicalOperator::RemoveLabel(remove) = ret.input.as_ref() {
                assert_eq!(remove.variable, "n");
                assert_eq!(remove.labels, vec!["Admin".to_string()]);
            } else {
                panic!("Expected RemoveLabel");
            }
        } else {
            panic!("Expected Return, got {:?}", plan.root);
        }
    }

    // === WITH, UNWIND, ORDER BY, SKIP, LIMIT Tests ===

    #[test]
    fn test_translate_with_clause() {
        let plan = translate("MATCH (n:Person) WITH n.name AS name RETURN name").unwrap();

        // Find Project operator
        fn find_project(op: &LogicalOperator) -> Option<&ProjectOp> {
            match op {
                LogicalOperator::Project(p) => Some(p),
                LogicalOperator::Return(r) => find_project(&r.input),
                LogicalOperator::Filter(f) => find_project(&f.input),
                _ => None,
            }
        }

        let project = find_project(&plan.root).expect("Expected Project");
        assert_eq!(project.projections.len(), 1);
        assert_eq!(project.projections[0].alias.as_deref(), Some("name"));
    }

    #[test]
    fn test_translate_with_distinct() {
        let plan = translate("MATCH (n:Person) WITH DISTINCT n.city AS city RETURN city").unwrap();

        // Find Distinct operator
        fn find_distinct(op: &LogicalOperator) -> bool {
            match op {
                LogicalOperator::Distinct(_) => true,
                LogicalOperator::Return(r) => find_distinct(&r.input),
                LogicalOperator::Project(p) => find_distinct(&p.input),
                LogicalOperator::Filter(f) => find_distinct(&f.input),
                _ => false,
            }
        }

        assert!(find_distinct(&plan.root));
    }

    #[test]
    fn test_translate_unwind() {
        let plan = translate("UNWIND [1, 2, 3] AS x RETURN x").unwrap();

        // Find Unwind operator
        fn find_unwind(op: &LogicalOperator) -> Option<&UnwindOp> {
            match op {
                LogicalOperator::Unwind(u) => Some(u),
                LogicalOperator::Return(r) => find_unwind(&r.input),
                _ => None,
            }
        }

        let unwind = find_unwind(&plan.root).expect("Expected Unwind");
        assert_eq!(unwind.variable, "x");
    }

    #[test]
    fn test_translate_order_by() {
        let plan = translate("MATCH (n:Person) RETURN n ORDER BY n.name").unwrap();

        fn find_sort(op: &LogicalOperator) -> Option<&SortOp> {
            match op {
                LogicalOperator::Sort(s) => Some(s),
                LogicalOperator::Return(r) => find_sort(&r.input),
                _ => None,
            }
        }

        let sort = find_sort(&plan.root).expect("Expected Sort");
        assert_eq!(sort.keys.len(), 1);
        assert_eq!(sort.keys[0].order, SortOrder::Ascending);
    }

    #[test]
    fn test_translate_order_by_desc() {
        let plan = translate("MATCH (n:Person) RETURN n ORDER BY n.age DESC").unwrap();

        fn find_sort(op: &LogicalOperator) -> Option<&SortOp> {
            match op {
                LogicalOperator::Sort(s) => Some(s),
                LogicalOperator::Return(r) => find_sort(&r.input),
                _ => None,
            }
        }

        let sort = find_sort(&plan.root).expect("Expected Sort");
        assert_eq!(sort.keys[0].order, SortOrder::Descending);
    }

    #[test]
    fn test_translate_limit() {
        let plan = translate("MATCH (n:Person) RETURN n LIMIT 10").unwrap();

        fn find_limit(op: &LogicalOperator) -> Option<&LimitOp> {
            match op {
                LogicalOperator::Limit(l) => Some(l),
                LogicalOperator::Return(r) => find_limit(&r.input),
                _ => None,
            }
        }

        let limit = find_limit(&plan.root).expect("Expected Limit");
        assert_eq!(limit.count, 10);
    }

    #[test]
    fn test_translate_skip() {
        let plan = translate("MATCH (n:Person) RETURN n SKIP 5").unwrap();

        fn find_skip(op: &LogicalOperator) -> Option<&SkipOp> {
            match op {
                LogicalOperator::Skip(s) => Some(s),
                LogicalOperator::Return(r) => find_skip(&r.input),
                LogicalOperator::Limit(l) => find_skip(&l.input),
                _ => None,
            }
        }

        let skip = find_skip(&plan.root).expect("Expected Skip");
        assert_eq!(skip.count, 5);
    }

    // === MERGE Tests ===

    #[test]
    fn test_translate_merge() {
        let plan = translate("MERGE (n:Person {name: 'Alix'})").unwrap();

        if let LogicalOperator::Merge(merge) = &plan.root {
            assert_eq!(merge.variable, "n");
            assert_eq!(merge.labels, vec!["Person".to_string()]);
            assert_eq!(merge.match_properties.len(), 1);
            assert_eq!(merge.match_properties[0].0, "name");
        } else {
            panic!("Expected Merge, got {:?}", plan.root);
        }
    }

    #[test]
    fn test_translate_merge_on_create() {
        let plan =
            translate("MERGE (n:Person {name: 'Alix'}) ON CREATE SET n.created = true").unwrap();

        if let LogicalOperator::Merge(merge) = &plan.root {
            assert_eq!(merge.on_create.len(), 1);
            assert_eq!(merge.on_create[0].0, "created");
        } else {
            panic!("Expected Merge, got {:?}", plan.root);
        }
    }

    // === Expression Tests ===

    #[test]
    fn test_translate_list_expression() {
        // Cypher requires MATCH before RETURN, so use UNWIND to test list
        let plan = translate("UNWIND [1, 2, 3] AS x RETURN x").unwrap();

        fn find_unwind(op: &LogicalOperator) -> Option<&UnwindOp> {
            match op {
                LogicalOperator::Unwind(u) => Some(u),
                LogicalOperator::Return(r) => find_unwind(&r.input),
                _ => None,
            }
        }

        let unwind = find_unwind(&plan.root).expect("Expected Unwind");
        if let LogicalExpression::List(items) = &unwind.expression {
            assert_eq!(items.len(), 3);
        } else {
            panic!("Expected List expression");
        }
    }

    #[test]
    fn test_translate_map_expression() {
        // Test map in CREATE with properties
        let plan = translate("CREATE (n:Person {name: 'Alix', age: 30})").unwrap();

        if let LogicalOperator::CreateNode(create) = &plan.root {
            assert_eq!(create.properties.len(), 2);
        } else {
            panic!("Expected CreateNode");
        }
    }

    #[test]
    fn test_translate_function_call() {
        // Use toUpper which is a simple function
        let plan = translate("MATCH (n:Person) RETURN toUpper(n.name)").unwrap();

        if let LogicalOperator::Return(ret) = &plan.root {
            if let LogicalExpression::FunctionCall { name, args, .. } = &ret.items[0].expression {
                assert_eq!(name.to_lowercase(), "toupper");
                assert_eq!(args.len(), 1);
            } else {
                panic!("Expected FunctionCall, got {:?}", ret.items[0].expression);
            }
        } else {
            panic!("Expected Return");
        }
    }

    #[test]
    fn test_translate_case_expression() {
        let plan =
            translate("MATCH (n:Person) RETURN CASE WHEN n.age > 18 THEN 'adult' ELSE 'minor' END")
                .unwrap();

        if let LogicalOperator::Return(ret) = &plan.root {
            if let LogicalExpression::Case {
                when_clauses,
                else_clause,
                ..
            } = &ret.items[0].expression
            {
                assert_eq!(when_clauses.len(), 1);
                assert!(else_clause.is_some());
            } else {
                panic!("Expected Case expression");
            }
        } else {
            panic!("Expected Return");
        }
    }

    #[test]
    fn test_translate_parameter() {
        let plan = translate("MATCH (n:Person) WHERE n.name = $name RETURN n").unwrap();

        fn find_filter(op: &LogicalOperator) -> Option<&FilterOp> {
            match op {
                LogicalOperator::Filter(f) => Some(f),
                LogicalOperator::Return(r) => find_filter(&r.input),
                _ => None,
            }
        }

        let filter = find_filter(&plan.root).expect("Expected Filter");
        if let LogicalExpression::Binary { right, .. } = &filter.predicate {
            if let LogicalExpression::Parameter(p) = right.as_ref() {
                assert_eq!(p, "name");
            } else {
                panic!("Expected Parameter");
            }
        }
    }

    // === Error Handling Tests ===

    #[test]
    fn test_translate_binary_op_all() {
        let translator = CypherTranslator::new();

        // Test all supported binary ops
        assert_eq!(
            translator.translate_binary_op(ast::BinaryOp::Eq).unwrap(),
            BinaryOp::Eq
        );
        assert_eq!(
            translator.translate_binary_op(ast::BinaryOp::Ne).unwrap(),
            BinaryOp::Ne
        );
        assert_eq!(
            translator.translate_binary_op(ast::BinaryOp::Lt).unwrap(),
            BinaryOp::Lt
        );
        assert_eq!(
            translator.translate_binary_op(ast::BinaryOp::Le).unwrap(),
            BinaryOp::Le
        );
        assert_eq!(
            translator.translate_binary_op(ast::BinaryOp::Gt).unwrap(),
            BinaryOp::Gt
        );
        assert_eq!(
            translator.translate_binary_op(ast::BinaryOp::Ge).unwrap(),
            BinaryOp::Ge
        );
        assert_eq!(
            translator.translate_binary_op(ast::BinaryOp::And).unwrap(),
            BinaryOp::And
        );
        assert_eq!(
            translator.translate_binary_op(ast::BinaryOp::Or).unwrap(),
            BinaryOp::Or
        );
        assert_eq!(
            translator.translate_binary_op(ast::BinaryOp::Xor).unwrap(),
            BinaryOp::Xor
        );
        assert_eq!(
            translator.translate_binary_op(ast::BinaryOp::Add).unwrap(),
            BinaryOp::Add
        );
        assert_eq!(
            translator.translate_binary_op(ast::BinaryOp::Sub).unwrap(),
            BinaryOp::Sub
        );
        assert_eq!(
            translator.translate_binary_op(ast::BinaryOp::Mul).unwrap(),
            BinaryOp::Mul
        );
        assert_eq!(
            translator.translate_binary_op(ast::BinaryOp::Div).unwrap(),
            BinaryOp::Div
        );
        assert_eq!(
            translator.translate_binary_op(ast::BinaryOp::Mod).unwrap(),
            BinaryOp::Mod
        );
    }

    #[test]
    fn test_translate_unary_op_all() {
        let translator = CypherTranslator::new();

        assert_eq!(
            translator.translate_unary_op(ast::UnaryOp::Not).unwrap(),
            UnaryOp::Not
        );
        assert_eq!(
            translator.translate_unary_op(ast::UnaryOp::Neg).unwrap(),
            UnaryOp::Neg
        );
        assert_eq!(
            translator.translate_unary_op(ast::UnaryOp::IsNull).unwrap(),
            UnaryOp::IsNull
        );
        assert_eq!(
            translator
                .translate_unary_op(ast::UnaryOp::IsNotNull)
                .unwrap(),
            UnaryOp::IsNotNull
        );
    }

    #[test]
    fn test_translate_literal_types() {
        let translator = CypherTranslator::new();

        // Test all literal types
        let null_lit = translator.translate_literal(&ast::Literal::Null).unwrap();
        assert!(matches!(null_lit, LogicalExpression::Literal(Value::Null)));

        let bool_lit = translator
            .translate_literal(&ast::Literal::Bool(true))
            .unwrap();
        assert!(matches!(
            bool_lit,
            LogicalExpression::Literal(Value::Bool(true))
        ));

        let int_lit = translator
            .translate_literal(&ast::Literal::Integer(42))
            .unwrap();
        assert!(matches!(
            int_lit,
            LogicalExpression::Literal(Value::Int64(42))
        ));

        let float_lit = translator
            .translate_literal(&ast::Literal::Float(std::f64::consts::PI))
            .unwrap();
        if let LogicalExpression::Literal(Value::Float64(f)) = float_lit {
            assert!((f - std::f64::consts::PI).abs() < 0.001);
        } else {
            panic!("Expected Float64");
        }
    }

    #[test]
    fn test_translate_multiple_match_clauses() {
        // Two independent MATCH clauses should produce a valid plan
        let plan = translate(
            "MATCH (a:Person) WHERE a.name = 'Alix' MATCH (b:Person) WHERE b.name = 'Gus' RETURN a.name, b.name",
        )
        .unwrap();

        // The plan should have a Return at the root
        assert!(matches!(&plan.root, LogicalOperator::Return(_)));
    }

    #[test]
    fn test_translate_merge_with_relationship() {
        // MERGE with a relationship pattern should produce MergeRelationship operator
        let plan =
            translate("MATCH (a {id: 'x'}), (b {id: 'y'}) MERGE (a)-[r:KNOWS]->(b) RETURN r")
                .unwrap();

        // The plan root should be a Return
        if let LogicalOperator::Return(ret) = &plan.root {
            // The input to Return should be MergeRelationship
            assert!(
                matches!(ret.input.as_ref(), LogicalOperator::MergeRelationship(_)),
                "Expected MergeRelationship, got: {:?}",
                std::mem::discriminant(ret.input.as_ref())
            );
        } else {
            panic!("Expected Return at root");
        }
    }

    #[test]
    fn test_translate_set_edge_property_is_edge_true() {
        // SET on an edge variable from MATCH should produce is_edge: true
        let plan = translate("MATCH (a)-[r:KNOWS]->(b) SET r.weight = 1.0 RETURN r").unwrap();

        // Walk the plan tree to find SetProperty
        fn find_set_property(op: &LogicalOperator) -> Option<&SetPropertyOp> {
            match op {
                LogicalOperator::SetProperty(set_op) => Some(set_op),
                LogicalOperator::Return(ret) => find_set_property(&ret.input),
                _ => None,
            }
        }

        let set_op = find_set_property(&plan.root).expect("Expected SetProperty in plan");
        assert!(
            set_op.is_edge,
            "SET on edge variable 'r' should have is_edge: true"
        );
        assert_eq!(set_op.variable, "r");
    }

    #[test]
    fn test_translate_set_node_property_is_edge_false() {
        // SET on a node variable should produce is_edge: false
        let plan = translate("MATCH (n:Person) SET n.age = 30 RETURN n").unwrap();

        fn find_set_property(op: &LogicalOperator) -> Option<&SetPropertyOp> {
            match op {
                LogicalOperator::SetProperty(set_op) => Some(set_op),
                LogicalOperator::Return(ret) => find_set_property(&ret.input),
                _ => None,
            }
        }

        let set_op = find_set_property(&plan.root).expect("Expected SetProperty in plan");
        assert!(
            !set_op.is_edge,
            "SET on node variable 'n' should have is_edge: false"
        );
        assert_eq!(set_op.variable, "n");
    }

    #[test]
    fn test_translate_set_edge_after_merge_relationship() {
        // SET on edge variable from MERGE should also have is_edge: true
        let plan = translate(
            "MATCH (a {id: 'x'}), (b {id: 'y'}) MERGE (a)-[r:KNOWS]->(b) SET r.since = '2024' RETURN r",
        )
        .unwrap();

        fn find_set_property(op: &LogicalOperator) -> Option<&SetPropertyOp> {
            match op {
                LogicalOperator::SetProperty(set_op) => Some(set_op),
                LogicalOperator::Return(ret) => find_set_property(&ret.input),
                _ => None,
            }
        }

        let set_op = find_set_property(&plan.root).expect("Expected SetProperty in plan");
        assert!(
            set_op.is_edge,
            "SET on MERGE edge variable 'r' should have is_edge: true"
        );
        assert_eq!(set_op.variable, "r");
    }

    #[test]
    fn test_translate_aggregate_count() {
        let plan = translate("MATCH (n:Person) RETURN count(n)").unwrap();
        // Should produce an Aggregate operator
        fn has_aggregate(op: &LogicalOperator) -> bool {
            match op {
                LogicalOperator::Aggregate(_) => true,
                LogicalOperator::Return(ret) => has_aggregate(&ret.input),
                _ => false,
            }
        }
        assert!(has_aggregate(&plan.root), "Expected Aggregate operator");
    }

    #[test]
    fn test_translate_case_inside_aggregate() {
        // sum(CASE WHEN n.type = 'x' THEN 1 ELSE 0 END) should be detected as aggregate
        let plan = translate(
            "MATCH (n:Person) RETURN sum(CASE WHEN n.type = 'source' THEN 1 ELSE 0 END) AS cnt",
        )
        .unwrap();
        fn has_aggregate(op: &LogicalOperator) -> bool {
            match op {
                LogicalOperator::Aggregate(_) => true,
                LogicalOperator::Return(ret) => has_aggregate(&ret.input),
                _ => false,
            }
        }
        assert!(
            has_aggregate(&plan.root),
            "Expected Aggregate operator for CASE inside aggregate"
        );
    }

    #[test]
    fn test_translate_case_wrapping_aggregate() {
        // CASE WHEN count(*) > 0 should also be detected as containing an aggregate
        let plan = translate(
            "MATCH (n:Person) RETURN CASE WHEN count(*) > 0 THEN 'yes' ELSE 'no' END AS result",
        )
        .unwrap();
        fn has_aggregate(op: &LogicalOperator) -> bool {
            match op {
                LogicalOperator::Aggregate(_) => true,
                LogicalOperator::Return(ret) => has_aggregate(&ret.input),
                _ => false,
            }
        }
        assert!(
            has_aggregate(&plan.root),
            "Expected Aggregate operator for CASE wrapping aggregate"
        );
    }

    #[test]
    fn test_translate_union_all() {
        let plan =
            translate("MATCH (n:Person) RETURN n.name UNION ALL MATCH (m:Animal) RETURN m.name")
                .unwrap();
        assert!(
            matches!(&plan.root, LogicalOperator::Union(_)),
            "Expected Union at root, got {:?}",
            std::mem::discriminant(&plan.root)
        );
    }

    #[test]
    fn test_translate_union_without_all_applies_distinct() {
        let plan = translate("MATCH (n:Person) RETURN n.name UNION MATCH (m:Animal) RETURN m.name")
            .unwrap();
        // UNION (without ALL) should wrap the result in Distinct
        assert!(
            matches!(&plan.root, LogicalOperator::Distinct(_)),
            "Expected Distinct at root for UNION without ALL, got {:?}",
            std::mem::discriminant(&plan.root)
        );
        if let LogicalOperator::Distinct(distinct) = &plan.root {
            assert!(
                matches!(distinct.input.as_ref(), LogicalOperator::Union(_)),
                "Expected Union inside Distinct"
            );
        }
    }

    #[test]
    fn test_translate_call_procedure() {
        let plan = translate("CALL db.labels()").unwrap();
        assert!(
            matches!(&plan.root, LogicalOperator::CallProcedure(_)),
            "Expected CallProcedure, got {:?}",
            std::mem::discriminant(&plan.root)
        );
        if let LogicalOperator::CallProcedure(call) = &plan.root {
            assert_eq!(call.name, vec!["db", "labels"]);
            assert!(call.arguments.is_empty());
            assert!(call.yield_items.is_none());
        }
    }

    #[test]
    fn test_translate_call_with_args_and_yield() {
        let plan = translate("CALL db.index.fulltext('Person', 'name') YIELD status").unwrap();
        if let LogicalOperator::CallProcedure(call) = &plan.root {
            assert_eq!(call.name, vec!["db", "index", "fulltext"]);
            assert_eq!(call.arguments.len(), 2);
            assert!(call.yield_items.is_some());
            let yields = call.yield_items.as_ref().unwrap();
            assert_eq!(yields.len(), 1);
            assert_eq!(yields[0].field_name, "status");
        } else {
            panic!("Expected CallProcedure");
        }
    }

    // === EXISTS Subquery Tests ===

    #[test]
    fn test_translate_exists_subquery() {
        let plan =
            translate("MATCH (n:Person) WHERE EXISTS { MATCH (n)-[:KNOWS]->() } RETURN n").unwrap();

        // Plan should be Return -> Filter -> NodeScan
        // with the Filter predicate being ExistsSubquery
        if let LogicalOperator::Return(ret) = &plan.root {
            if let LogicalOperator::Filter(filter) = ret.input.as_ref() {
                assert!(
                    matches!(&filter.predicate, LogicalExpression::ExistsSubquery(_)),
                    "Expected ExistsSubquery predicate, got {:?}",
                    filter.predicate
                );
            } else {
                panic!("Expected Filter, got {:?}", ret.input);
            }
        } else {
            panic!("Expected Return");
        }
    }

    // === Map Projection Tests ===

    #[test]
    fn test_translate_map_projection() {
        let plan = translate("MATCH (p:Person) RETURN p { .name, .age }").unwrap();

        if let LogicalOperator::Return(ret) = &plan.root {
            assert_eq!(ret.items.len(), 1);
            if let LogicalExpression::MapProjection { base, entries } = &ret.items[0].expression {
                assert_eq!(base, "p");
                assert_eq!(entries.len(), 2);
                assert!(
                    matches!(&entries[0], MapProjectionEntry::PropertySelector(s) if s == "name")
                );
                assert!(
                    matches!(&entries[1], MapProjectionEntry::PropertySelector(s) if s == "age")
                );
            } else {
                panic!(
                    "Expected MapProjection expression, got {:?}",
                    ret.items[0].expression
                );
            }
        } else {
            panic!("Expected Return");
        }
    }

    // === reduce() Tests ===

    #[test]
    fn test_translate_reduce() {
        let plan = translate("MATCH (n) RETURN reduce(acc = 0, x IN [1,2,3] | acc + x)").unwrap();

        // Walk past possible Aggregate wrapping to find the Reduce expression
        fn find_reduce_expr(op: &LogicalOperator) -> Option<&LogicalExpression> {
            match op {
                LogicalOperator::Return(ret) => {
                    for item in &ret.items {
                        if matches!(&item.expression, LogicalExpression::Reduce { .. }) {
                            return Some(&item.expression);
                        }
                    }
                    find_reduce_expr(&ret.input)
                }
                LogicalOperator::Aggregate(agg) => {
                    // Check group_by expressions
                    for expr in &agg.group_by {
                        if matches!(expr, LogicalExpression::Reduce { .. }) {
                            return Some(expr);
                        }
                    }
                    find_reduce_expr(&agg.input)
                }
                _ => None,
            }
        }

        let reduce_expr =
            find_reduce_expr(&plan.root).expect("Expected Reduce expression in the plan");

        if let LogicalExpression::Reduce {
            accumulator,
            initial,
            variable,
            list,
            expression,
        } = reduce_expr
        {
            assert_eq!(accumulator, "acc");
            assert!(matches!(
                initial.as_ref(),
                LogicalExpression::Literal(Value::Int64(0))
            ));
            assert_eq!(variable, "x");
            // The list should be a List of 3 items
            if let LogicalExpression::List(items) = list.as_ref() {
                assert_eq!(items.len(), 3);
            } else {
                panic!("Expected List for reduce iteration, got {:?}", list);
            }
            // The body should be acc + x (Binary Add)
            if let LogicalExpression::Binary { op, .. } = expression.as_ref() {
                assert_eq!(*op, BinaryOp::Add);
            } else {
                panic!("Expected Binary Add in reduce body, got {:?}", expression);
            }
        } else {
            panic!("Expected Reduce, got {:?}", reduce_expr);
        }
    }

    // === Pattern Comprehension Tests ===

    #[test]
    fn test_translate_pattern_comprehension() {
        let plan = translate("MATCH (p:Person) RETURN [(p)-[:KNOWS]->(f) | f.name]").unwrap();

        // After rewrite: Return -> Apply -> NodeScan
        // The PatternComprehension is rewritten to Apply + Aggregate(Collect) + ParameterScan
        if let LogicalOperator::Return(ret) = &plan.root {
            assert_eq!(ret.items.len(), 1);
            // Expression should now be a Variable reference (not PatternComprehension)
            assert!(
                matches!(&ret.items[0].expression, LogicalExpression::Variable(_)),
                "Expected Variable after rewrite, got {:?}",
                ret.items[0].expression
            );
            // The input should be an Apply operator
            if let LogicalOperator::Apply(apply) = ret.input.as_ref() {
                assert_eq!(apply.shared_variables, vec!["p".to_string()]);
                // Inner plan should be Aggregate(Collect)
                if let LogicalOperator::Aggregate(agg) = apply.subplan.as_ref() {
                    assert_eq!(agg.aggregates.len(), 1);
                    assert_eq!(agg.aggregates[0].function, AggregateFunction::Collect);
                    // Aggregate input should contain an Expand over ParameterScan
                    fn has_parameter_scan(op: &LogicalOperator) -> bool {
                        match op {
                            LogicalOperator::ParameterScan(_) => true,
                            LogicalOperator::Expand(e) => has_parameter_scan(&e.input),
                            LogicalOperator::Filter(f) => has_parameter_scan(&f.input),
                            _ => false,
                        }
                    }
                    assert!(
                        has_parameter_scan(&agg.input),
                        "Expected ParameterScan in inner plan, got {:?}",
                        agg.input
                    );
                } else {
                    panic!(
                        "Expected Aggregate in Apply subplan, got {:?}",
                        apply.subplan
                    );
                }
            } else {
                panic!("Expected Apply as Return input, got {:?}", ret.input);
            }
        } else {
            panic!("Expected Return");
        }
    }

    // === COUNT Subquery Tests ===

    #[test]
    fn test_translate_count_subquery() {
        let plan =
            translate("MATCH (p:Person) RETURN COUNT { MATCH (p)-[:KNOWS]->() } AS cnt").unwrap();

        // The COUNT subquery should appear as a CountSubquery expression
        fn find_count_subquery(op: &LogicalOperator) -> bool {
            match op {
                LogicalOperator::Return(ret) => {
                    ret.items
                        .iter()
                        .any(|item| matches!(&item.expression, LogicalExpression::CountSubquery(_)))
                        || find_count_subquery(&ret.input)
                }
                LogicalOperator::Aggregate(agg) => {
                    agg.group_by
                        .iter()
                        .any(|expr| matches!(expr, LogicalExpression::CountSubquery(_)))
                        || find_count_subquery(&agg.input)
                }
                _ => false,
            }
        }

        assert!(
            find_count_subquery(&plan.root),
            "Expected CountSubquery in the plan, got {:?}",
            plan.root
        );
    }

    // === CALL Subquery Tests ===

    #[test]
    fn test_translate_call_subquery() {
        let plan = translate(
            "MATCH (p:Person) CALL { WITH p MATCH (p)-[:KNOWS]->(f) RETURN count(f) AS cnt } RETURN p.name, cnt",
        )
        .unwrap();

        // The plan should have an Apply operator for the CALL subquery
        fn find_apply(op: &LogicalOperator) -> bool {
            match op {
                LogicalOperator::Apply(_) => true,
                LogicalOperator::Return(ret) => find_apply(&ret.input),
                LogicalOperator::Filter(f) => find_apply(&f.input),
                LogicalOperator::Sort(s) => find_apply(&s.input),
                _ => false,
            }
        }

        assert!(
            find_apply(&plan.root),
            "Expected Apply operator for CALL subquery"
        );

        // Verify the final RETURN has two items
        if let LogicalOperator::Return(ret) = &plan.root {
            assert_eq!(
                ret.items.len(),
                2,
                "Expected 2 return items (p.name and cnt)"
            );
        } else {
            panic!("Expected Return at root");
        }
    }

    // === FOREACH Tests ===

    #[test]
    fn test_translate_foreach() {
        let plan = translate("MATCH (n:Person) FOREACH (x IN [1,2,3] | SET n.x = 1)").unwrap();

        // FOREACH translates to Unwind + mutation pipeline
        // The plan should contain an Unwind operator somewhere
        fn find_unwind(op: &LogicalOperator) -> bool {
            match op {
                LogicalOperator::Unwind(_) => true,
                LogicalOperator::SetProperty(s) => find_unwind(&s.input),
                LogicalOperator::Filter(f) => find_unwind(&f.input),
                LogicalOperator::Return(r) => find_unwind(&r.input),
                _ => false,
            }
        }

        assert!(
            find_unwind(&plan.root),
            "FOREACH should produce an Unwind operator in the plan"
        );

        // The root should be a SetProperty (the inner SET clause)
        assert!(
            matches!(&plan.root, LogicalOperator::SetProperty(_)),
            "Expected SetProperty at root for FOREACH with SET, got {:?}",
            std::mem::discriminant(&plan.root)
        );
    }

    // === Basic Query Translation Tests ===

    #[test]
    fn test_translate_match_with_multiple_labels() {
        // Multi-label node patterns
        let plan = translate("MATCH (n:Person:Employee) RETURN n").unwrap();
        assert!(matches!(&plan.root, LogicalOperator::Return(_)));
    }

    #[test]
    fn test_translate_standalone_return() {
        // RETURN without MATCH (pure expression evaluation)
        let plan = translate("RETURN 2 * 3").unwrap();
        if let LogicalOperator::Return(ret) = &plan.root {
            assert_eq!(ret.items.len(), 1);
            // The expression should be a Binary Mul of 2 * 3
            if let LogicalExpression::Binary { op, left, right } = &ret.items[0].expression {
                assert_eq!(*op, BinaryOp::Mul);
                assert!(matches!(
                    left.as_ref(),
                    LogicalExpression::Literal(Value::Int64(2))
                ));
                assert!(matches!(
                    right.as_ref(),
                    LogicalExpression::Literal(Value::Int64(3))
                ));
            } else {
                panic!("Expected Binary expression");
            }
        } else {
            panic!("Expected Return");
        }
    }

    #[test]
    fn test_translate_undirected_relationship() {
        let plan = translate("MATCH (a:Person)-[:FRIEND]-(b:Person) RETURN a, b").unwrap();

        fn find_expand(op: &LogicalOperator) -> Option<&ExpandOp> {
            match op {
                LogicalOperator::Expand(e) => Some(e),
                LogicalOperator::Return(r) => find_expand(&r.input),
                LogicalOperator::Filter(f) => find_expand(&f.input),
                _ => None,
            }
        }

        let expand = find_expand(&plan.root).expect("Expected Expand");
        assert_eq!(expand.direction, ExpandDirection::Both);
        assert_eq!(expand.edge_types, vec!["FRIEND".to_string()]);
    }

    #[test]
    fn test_translate_match_with_return_alias() {
        let plan = translate("MATCH (n:Person) RETURN n.name AS personName").unwrap();

        if let LogicalOperator::Return(ret) = &plan.root {
            assert_eq!(ret.items.len(), 1);
            assert_eq!(ret.items[0].alias.as_deref(), Some("personName"));
        } else {
            panic!("Expected Return");
        }
    }

    #[test]
    fn test_translate_multiple_return_items() {
        let plan = translate("MATCH (n:Person) RETURN n.name, n.age, n.city").unwrap();

        if let LogicalOperator::Return(ret) = &plan.root {
            assert_eq!(ret.items.len(), 3);
        } else {
            panic!("Expected Return");
        }
    }

    #[test]
    fn test_translate_list_predicate_all() {
        let plan = translate("MATCH (n) WHERE all(x IN [1,2,3] WHERE x > 0) RETURN n").unwrap();

        fn find_filter(op: &LogicalOperator) -> Option<&FilterOp> {
            match op {
                LogicalOperator::Filter(f) => Some(f),
                LogicalOperator::Return(r) => find_filter(&r.input),
                _ => None,
            }
        }

        let filter = find_filter(&plan.root).expect("Expected Filter");
        assert!(
            matches!(
                &filter.predicate,
                LogicalExpression::ListPredicate {
                    kind: ListPredicateKind::All,
                    ..
                }
            ),
            "Expected ListPredicate(All), got {:?}",
            filter.predicate
        );
    }

    #[test]
    fn test_translate_list_comprehension() {
        let plan = translate("MATCH (n) RETURN [x IN [1,2,3] WHERE x > 1 | x * 2]").unwrap();

        if let LogicalOperator::Return(ret) = &plan.root {
            assert_eq!(ret.items.len(), 1);
            assert!(
                matches!(
                    &ret.items[0].expression,
                    LogicalExpression::ListComprehension { .. }
                ),
                "Expected ListComprehension, got {:?}",
                ret.items[0].expression
            );
        } else {
            panic!("Expected Return");
        }
    }
}
