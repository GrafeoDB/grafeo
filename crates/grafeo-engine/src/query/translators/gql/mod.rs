//! GQL to LogicalPlan translator.
//!
//! Translates GQL AST to the common logical plan representation.

mod aggregate;
mod expression;
mod pattern;

use std::collections::{HashMap, HashSet};

use super::common::{
    build_left_join_with_predicates, check_union_columns, combine_with_and, flatten_and_conjuncts,
    is_aggregate_function, is_binary_set_function, join_and_conjuncts, references_any,
    to_aggregate_function, wrap_distinct, wrap_filter, wrap_limit, wrap_return, wrap_skip,
    wrap_sort,
};
use crate::query::plan::{
    self as plan, AddLabelOp, AggregateExpr, AggregateFunction, AggregateOp, ApplyOp, BinaryOp,
    CallProcedureOp, CountExpr, CreateEdgeOp, CreateNodeOp, DeleteNodeOp, EntityKind, ExceptOp,
    ExpandDirection, ExpandOp, HorizontalAggregateOp, IntersectOp, JoinCondition, JoinOp, JoinType,
    LeftJoinOp, LoadDataFormat, LoadDataOp, LogicalExpression, LogicalOperator, LogicalPlan,
    MergeOp, MergeRelationshipOp, NodeScanOp, NullsOrdering, OtherwiseOp, ParameterScanOp,
    PathMode, ProcedureYield, ProjectOp, Projection, RemoveLabelOp, ReturnItem, SetPropertyOp,
    ShortestPathOp, SortKey, SortOrder, UnaryOp, UnionOp, UnwindOp,
};
#[cfg(test)]
use crate::query::plan::{FilterOp, LimitOp, SkipOp};
use grafeo_adapters::query::gql::{self, ast};
use grafeo_common::types::Value;
use grafeo_common::utils::error::{Error, QueryError, QueryErrorKind, Result};

/// Result of translating a GQL query: either a logical plan, session command, or schema command.
#[derive(Debug)]
#[non_exhaustive]
pub enum GqlTranslationResult {
    /// A query plan to execute (or EXPLAIN if `plan.explain` is true).
    Plan(LogicalPlan),
    /// A session or transaction command (not a query plan).
    SessionCommand(ast::SessionCommand),
    /// A schema DDL command (CREATE/DROP TYPE, INDEX, CONSTRAINT).
    SchemaCommand(ast::SchemaStatement),
}

/// Translates a GQL query string to a logical plan.
///
/// Session/transaction commands (USE GRAPH, COMMIT, etc.) return an error.
/// Use [`translate_full`] to handle both plans and session commands.
///
/// # Errors
///
/// Returns an error if the query cannot be parsed or translated.
pub fn translate(query: &str) -> Result<LogicalPlan> {
    match translate_full(query)? {
        GqlTranslationResult::Plan(plan) => Ok(plan),
        GqlTranslationResult::SessionCommand(_) => Err(Error::Query(QueryError::new(
            QueryErrorKind::Semantic,
            "Session commands cannot be executed as queries",
        ))),
        GqlTranslationResult::SchemaCommand(_) => Err(Error::Query(QueryError::new(
            QueryErrorKind::Semantic,
            "Schema DDL commands cannot be executed as queries",
        ))),
    }
}

/// Translates a GQL query string, returning either a logical plan or session command.
///
/// # Errors
///
/// Returns an error if the query cannot be parsed or translated.
pub fn translate_full(query: &str) -> Result<GqlTranslationResult> {
    let statement = gql::parse(query)?;
    let translator = GqlTranslator::new();
    translator.translate_statement_full(&statement)
}

/// Translator from GQL AST to LogicalPlan.
struct GqlTranslator {
    /// Edge variables from variable-length expand patterns (group-list variables).
    /// Maps edge variable name to the path alias used for `_path_edges_{alias}` lookup.
    group_list_variables: std::cell::RefCell<HashMap<String, String>>,
}

impl GqlTranslator {
    fn new() -> Self {
        Self {
            group_list_variables: std::cell::RefCell::new(HashMap::new()),
        }
    }

    fn translate_statement_full(&self, stmt: &ast::Statement) -> Result<GqlTranslationResult> {
        match stmt {
            ast::Statement::SessionCommand(cmd) => {
                Ok(GqlTranslationResult::SessionCommand(cmd.clone()))
            }
            ast::Statement::Schema(schema) => {
                Ok(GqlTranslationResult::SchemaCommand(schema.clone()))
            }
            ast::Statement::Explain(inner) => {
                let mut plan = self.translate_statement(inner)?;
                plan.explain = true;
                Ok(GqlTranslationResult::Plan(plan))
            }
            ast::Statement::Profile(inner) => {
                let mut plan = self.translate_statement(inner)?;
                plan.profile = true;
                Ok(GqlTranslationResult::Plan(plan))
            }
            other => self
                .translate_statement(other)
                .map(GqlTranslationResult::Plan),
        }
    }

    fn translate_statement(&self, stmt: &ast::Statement) -> Result<LogicalPlan> {
        match stmt {
            ast::Statement::Query(query) => self.translate_query(query),
            ast::Statement::DataModification(dm) => self.translate_data_modification(dm),
            ast::Statement::Schema(_) => Err(Error::Query(QueryError::new(
                QueryErrorKind::Semantic,
                "Schema DDL commands are handled before query planning",
            ))),
            ast::Statement::Call(call) => self.translate_call(call),
            ast::Statement::CompositeQuery { left, op, right } => {
                self.translate_composite_query(left, *op, right)
            }
            ast::Statement::SessionCommand(_) => Err(Error::Query(QueryError::new(
                QueryErrorKind::Semantic,
                "Session commands cannot be executed as queries",
            ))),
            ast::Statement::Explain(inner) | ast::Statement::Profile(inner) => {
                self.translate_statement(inner)
            }
        }
    }

    fn translate_composite_query(
        &self,
        left: &ast::Statement,
        op: ast::CompositeOp,
        right: &ast::Statement,
    ) -> Result<LogicalPlan> {
        let left_plan = self.translate_statement(left)?;
        let right_plan = self.translate_statement(right)?;

        match op {
            ast::CompositeOp::Union | ast::CompositeOp::UnionAll => {
                let inputs = vec![left_plan.root, right_plan.root];
                check_union_columns(&inputs)?;
                let union_op = LogicalOperator::Union(UnionOp { inputs });
                let root = if op == ast::CompositeOp::UnionAll {
                    union_op
                } else {
                    wrap_distinct(union_op)
                };
                Ok(LogicalPlan::new(root))
            }
            ast::CompositeOp::Except | ast::CompositeOp::ExceptAll => {
                let root = LogicalOperator::Except(ExceptOp {
                    left: Box::new(left_plan.root),
                    right: Box::new(right_plan.root),
                    all: matches!(op, ast::CompositeOp::ExceptAll),
                });
                Ok(LogicalPlan::new(root))
            }
            ast::CompositeOp::Intersect | ast::CompositeOp::IntersectAll => {
                let root = LogicalOperator::Intersect(IntersectOp {
                    left: Box::new(left_plan.root),
                    right: Box::new(right_plan.root),
                    all: matches!(op, ast::CompositeOp::IntersectAll),
                });
                Ok(LogicalPlan::new(root))
            }
            ast::CompositeOp::Otherwise => {
                let root = LogicalOperator::Otherwise(OtherwiseOp {
                    left: Box::new(left_plan.root),
                    right: Box::new(right_plan.root),
                });
                Ok(LogicalPlan::new(root))
            }
            ast::CompositeOp::Next => {
                // NEXT (linear composition): output of left feeds as input to right.
                // Translate as Apply: for each row from left, execute right with bound variables.
                let root = LogicalOperator::Apply(ApplyOp {
                    input: Box::new(left_plan.root),
                    subplan: Box::new(right_plan.root),
                    shared_variables: Vec::new(),
                    optional: false,
                });
                Ok(LogicalPlan::new(root))
            }
        }
    }

    fn translate_call(&self, call: &ast::CallStatement) -> Result<LogicalPlan> {
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

        let mut plan = LogicalOperator::CallProcedure(CallProcedureOp {
            name: call.procedure_name.clone(),
            arguments,
            yield_items,
        });

        // Apply WHERE filter on yielded rows
        if let Some(where_clause) = &call.where_clause {
            let predicate = self.translate_expression(&where_clause.expression)?;
            plan = wrap_filter(plan, predicate);
        }

        // Apply RETURN clause (with ORDER BY, SKIP, LIMIT)
        // Order: RETURN first (closest to input), then Sort, Skip, Limit wrap it.
        // This ensures RETURN aliases are visible to ORDER BY in the binder.
        if let Some(return_clause) = &call.return_clause {
            // Check if RETURN contains aggregate functions (e.g. count(label))
            let has_aggregates = !return_clause.items.is_empty()
                && return_clause
                    .items
                    .iter()
                    .any(|item| contains_aggregate(&item.expression));

            if has_aggregates {
                let (aggregates, auto_group_by, post_return) = self.extract_aggregates_and_groups(
                    &return_clause.items,
                    !return_clause.group_by.is_empty(),
                )?;
                // Explicit GROUP BY wins over the keys implied by the items.
                let group_by = if return_clause.group_by.is_empty() {
                    auto_group_by
                } else {
                    return_clause
                        .group_by
                        .iter()
                        .map(|e| self.translate_expression(e))
                        .collect::<Result<Vec<_>>>()?
                };

                plan = LogicalOperator::Aggregate(AggregateOp {
                    group_by,
                    aggregates,
                    input: Box::new(plan),
                    having: None,
                });

                // Collect aggregate output column names for ORDER BY rewriting.
                // After aggregation the original entity variable no longer exists,
                // so property references like `n.prop` must become flat variable
                // references when they match an output column.
                let agg_output_columns: HashSet<String> = post_return
                    .as_ref()
                    .map(|items| {
                        items
                            .iter()
                            .filter_map(|ri| {
                                ri.alias.clone().or_else(|| {
                                    if let LogicalExpression::Variable(v) = &ri.expression {
                                        Some(v.clone())
                                    } else {
                                        None
                                    }
                                })
                            })
                            .collect()
                    })
                    .unwrap_or_default();

                if let Some(return_items) = post_return {
                    plan = wrap_return(plan, return_items, return_clause.distinct);
                }

                // Apply ORDER BY with aggregate-aware rewriting
                if let Some(order_by) = &return_clause.order_by {
                    let keys = order_by
                        .items
                        .iter()
                        .map(|item| {
                            let mut expression = self.translate_expression(&item.expression)?;
                            if let LogicalExpression::Property { .. } = &expression {
                                let col_name = crate::query::planner::common::expression_to_string(
                                    &expression,
                                );
                                if agg_output_columns.contains(&col_name) {
                                    expression = LogicalExpression::Variable(col_name);
                                }
                            }
                            Ok(SortKey {
                                expression,
                                order: match item.order {
                                    ast::SortOrder::Asc => SortOrder::Ascending,
                                    ast::SortOrder::Desc => SortOrder::Descending,
                                },
                                nulls: item.nulls.map(|n| match n {
                                    ast::NullsOrdering::First => NullsOrdering::First,
                                    ast::NullsOrdering::Last => NullsOrdering::Last,
                                }),
                            })
                        })
                        .collect::<Result<Vec<_>>>()?;

                    plan = wrap_sort(plan, keys);
                }

                // Apply SKIP
                if let Some(skip_expr) = &return_clause.skip
                    && let ast::Expression::Literal(ast::Literal::Integer(n)) = skip_expr
                {
                    // reason: SKIP/LIMIT literals are non-negative in practice
                    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
                    let count = *n as usize;
                    plan = wrap_skip(plan, count);
                }

                // Apply LIMIT
                if let Some(limit_expr) = &return_clause.limit
                    && let ast::Expression::Literal(ast::Literal::Integer(n)) = limit_expr
                {
                    // reason: SKIP/LIMIT literals are non-negative in practice
                    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
                    let count = *n as usize;
                    plan = wrap_limit(plan, count);
                }
            } else if !return_clause.items.is_empty() {
                let return_items = return_clause
                    .items
                    .iter()
                    .map(|item| {
                        Ok(ReturnItem {
                            expression: self.translate_expression(&item.expression)?,
                            alias: item.alias.clone(),
                        })
                    })
                    .collect::<Result<Vec<_>>>()?;

                plan = wrap_return(plan, return_items, return_clause.distinct);
            }

            // Apply ORDER BY/SKIP/LIMIT for non-aggregate CALL queries.
            // Aggregate CALL queries handle these inside their own branch
            // with aggregate-aware ORDER BY rewriting.
            if !has_aggregates {
                if let Some(order_by) = &return_clause.order_by {
                    let keys = order_by
                        .items
                        .iter()
                        .map(|item| {
                            Ok(SortKey {
                                expression: self.translate_expression(&item.expression)?,
                                order: match item.order {
                                    ast::SortOrder::Asc => SortOrder::Ascending,
                                    ast::SortOrder::Desc => SortOrder::Descending,
                                },
                                nulls: item.nulls.map(|n| match n {
                                    ast::NullsOrdering::First => NullsOrdering::First,
                                    ast::NullsOrdering::Last => NullsOrdering::Last,
                                }),
                            })
                        })
                        .collect::<Result<Vec<_>>>()?;

                    plan = wrap_sort(plan, keys);
                }

                if let Some(skip_expr) = &return_clause.skip
                    && let ast::Expression::Literal(ast::Literal::Integer(n)) = skip_expr
                {
                    // reason: SKIP/LIMIT literals are non-negative in practice
                    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
                    let count = *n as usize;
                    plan = wrap_skip(plan, count);
                }
            }

            // Apply LIMIT (non-aggregate path)
            if !has_aggregates
                && let Some(limit_expr) = &return_clause.limit
                && let ast::Expression::Literal(ast::Literal::Integer(n)) = limit_expr
            {
                // reason: SKIP/LIMIT literals are non-negative in practice
                #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
                let count = *n as usize;
                plan = wrap_limit(plan, count);
            }
        }

        Ok(LogicalPlan::new(plan))
    }

    fn translate_query(&self, query: &ast::QueryStatement) -> Result<LogicalPlan> {
        let mut plan = LogicalOperator::Empty;
        let mut where_applied = false;

        // Process clauses in source order for correct variable scoping.
        // When ordered_clauses is populated, use it; otherwise fall back to
        // the legacy field-based ordering (for backward compatibility).
        if !query.ordered_clauses.is_empty() {
            for clause in &query.ordered_clauses {
                // Apply WHERE filter before the first mutation clause so that
                // MATCH ... WHERE ... CREATE/DELETE operates on filtered rows.
                if !where_applied
                    && matches!(
                        clause,
                        ast::QueryClause::Create(_)
                            | ast::QueryClause::Delete(_)
                            | ast::QueryClause::Set(_)
                            | ast::QueryClause::Merge(_)
                    )
                {
                    if let Some(where_clause) = &query.where_clause {
                        let predicate = self.translate_expression(&where_clause.expression)?;
                        plan = self.apply_where_with_left_join_awareness(plan, predicate);
                    }
                    where_applied = true;
                }

                match clause {
                    ast::QueryClause::Match(match_clause) => {
                        if matches!(plan, LogicalOperator::Empty) && match_clause.optional {
                            // OPTIONAL MATCH as the first clause: left join with
                            // an implicit unit table so unmatched patterns produce
                            // a single row of NULLs instead of zero rows.
                            let match_plan = self.translate_match(match_clause)?;
                            plan = LogicalOperator::LeftJoin(LeftJoinOp {
                                left: Box::new(LogicalOperator::Empty),
                                right: Box::new(match_plan),
                                condition: None,
                            });
                        } else if matches!(plan, LogicalOperator::Empty) {
                            // No prior input: standard MATCH
                            plan = self.translate_match(match_clause)?;
                        } else if match_clause.optional {
                            // OPTIONAL MATCH: left join (prior vars on left, match on right)
                            let match_plan = self.translate_match(match_clause)?;
                            plan = LogicalOperator::LeftJoin(LeftJoinOp {
                                left: Box::new(plan),
                                right: Box::new(match_plan),
                                condition: None,
                            });
                        } else {
                            // Non-optional MATCH after prior clauses (UNWIND, etc.)
                            // Pass current plan as input so the MATCH's NodeScan creates
                            // a nested loop join, keeping prior variables (like UNWIND
                            // variables) in scope for property filters.
                            let input = std::mem::replace(&mut plan, LogicalOperator::Empty);
                            plan = self.translate_match_with_input(match_clause, Some(input))?;
                        }
                    }
                    ast::QueryClause::Unwind(unwind_clause) => {
                        let expression = self.translate_expression(&unwind_clause.expression)?;
                        plan = LogicalOperator::Unwind(UnwindOp {
                            expression,
                            variable: unwind_clause.alias.clone(),
                            ordinality_var: None,
                            offset_var: None,
                            input: Box::new(plan),
                        });
                    }
                    ast::QueryClause::For(unwind_clause) => {
                        let expression = self.translate_expression(&unwind_clause.expression)?;
                        plan = LogicalOperator::Unwind(UnwindOp {
                            expression,
                            variable: unwind_clause.alias.clone(),
                            ordinality_var: unwind_clause.ordinality_var.clone(),
                            offset_var: unwind_clause.offset_var.clone(),
                            input: Box::new(plan),
                        });
                    }
                    ast::QueryClause::Create(create_clause) => {
                        // An INSERT that starts the statement has no input rows.
                        plan = if matches!(plan, LogicalOperator::Empty) {
                            self.insert_chain(&create_clause.patterns)?.0
                        } else {
                            self.translate_create_patterns(&create_clause.patterns, plan)?
                        };
                    }
                    ast::QueryClause::Delete(delete_clause) => {
                        plan = self.translate_delete_targets(
                            &delete_clause.targets,
                            delete_clause.detach,
                            plan,
                        )?;
                    }
                    ast::QueryClause::Set(set_clause) => {
                        for assignment in &set_clause.assignments {
                            let value = self.translate_expression(&assignment.value)?;
                            plan = LogicalOperator::SetProperty(SetPropertyOp {
                                variable: assignment.variable.clone(),
                                properties: vec![(assignment.property.clone(), value)],
                                replace: false,
                                is_edge: false,
                                input: Box::new(plan),
                            });
                        }
                        for map_assign in &set_clause.map_assignments {
                            let value = self.translate_expression(&map_assign.map_expr)?;
                            plan = LogicalOperator::SetProperty(SetPropertyOp {
                                variable: map_assign.variable.clone(),
                                properties: vec![("*".to_string(), value)],
                                replace: map_assign.replace,
                                is_edge: false,
                                input: Box::new(plan),
                            });
                        }
                        for label_op in &set_clause.label_operations {
                            plan = LogicalOperator::AddLabel(AddLabelOp {
                                variable: label_op.variable.clone(),
                                labels: label_op.labels.clone(),
                                input: Box::new(plan),
                            });
                        }
                    }
                    ast::QueryClause::Merge(merge_clause) => {
                        plan = self.translate_merge(merge_clause, plan)?;
                    }
                    ast::QueryClause::Let(bindings) => {
                        // LET var = expr translates to a Project that adds the bound
                        // variables as additional columns in the current pipeline.
                        let mut projections = Vec::new();
                        for (name, expr) in bindings {
                            let logical_expr = self.translate_expression(expr)?;
                            projections.push(Projection {
                                expression: logical_expr,
                                alias: Some(name.clone()),
                            });
                        }
                        plan = LogicalOperator::Project(ProjectOp {
                            projections,
                            input: Box::new(plan),
                            pass_through_input: true,
                        });
                    }
                    ast::QueryClause::InlineCall { subquery, optional } => {
                        plan = self.translate_inline_call(subquery, plan, *optional)?;
                    }
                    ast::QueryClause::CallProcedure(call_stmt) => {
                        // CALL procedure(...) within a query context
                        let call_plan = self.translate_call(call_stmt)?.root;
                        if matches!(plan, LogicalOperator::Empty) {
                            plan = call_plan;
                        } else {
                            plan = LogicalOperator::Apply(ApplyOp {
                                input: Box::new(plan),
                                subplan: Box::new(call_plan),
                                shared_variables: Vec::new(),
                                optional: false,
                            });
                        }
                    }
                    ast::QueryClause::LoadData(load_clause) => {
                        let load_plan = self.translate_load_data(load_clause);
                        if matches!(plan, LogicalOperator::Empty) {
                            plan = load_plan;
                        } else {
                            // Cross join with existing plan
                            plan = LogicalOperator::Join(JoinOp {
                                left: Box::new(plan),
                                right: Box::new(load_plan),
                                join_type: JoinType::Cross,
                                conditions: vec![],
                            });
                        }
                    }
                }
            }
        } else {
            // Legacy path: process MATCH, then UNWIND, then MERGE separately
            for match_clause in &query.match_clauses {
                let match_plan = self.translate_match(match_clause)?;
                if matches!(plan, LogicalOperator::Empty) && match_clause.optional {
                    plan = LogicalOperator::LeftJoin(LeftJoinOp {
                        left: Box::new(LogicalOperator::Empty),
                        right: Box::new(match_plan),
                        condition: None,
                    });
                } else if matches!(plan, LogicalOperator::Empty) {
                    plan = match_plan;
                } else if match_clause.optional {
                    plan = LogicalOperator::LeftJoin(LeftJoinOp {
                        left: Box::new(plan),
                        right: Box::new(match_plan),
                        condition: None,
                    });
                } else {
                    plan = LogicalOperator::Join(JoinOp {
                        left: Box::new(plan),
                        right: Box::new(match_plan),
                        join_type: JoinType::Cross,
                        conditions: vec![],
                    });
                }
            }

            for unwind_clause in &query.unwind_clauses {
                let expression = self.translate_expression(&unwind_clause.expression)?;
                plan = LogicalOperator::Unwind(UnwindOp {
                    expression,
                    variable: unwind_clause.alias.clone(),
                    ordinality_var: unwind_clause.ordinality_var.clone(),
                    offset_var: unwind_clause.offset_var.clone(),
                    input: Box::new(plan),
                });
            }

            for merge_clause in &query.merge_clauses {
                plan = self.translate_merge(merge_clause, plan)?;
            }
        }

        // Apply WHERE filter (skip if already applied before a mutation clause)
        if !where_applied && let Some(where_clause) = &query.where_clause {
            let predicate = self.translate_expression(&where_clause.expression)?;
            plan = self.apply_where_with_left_join_awareness(plan, predicate);
        }

        // Legacy path: handle SET/REMOVE/CREATE/DELETE from individual fields.
        // When ordered_clauses is used, these are already processed above.
        if query.ordered_clauses.is_empty() {
            for set_clause in &query.set_clauses {
                for assignment in &set_clause.assignments {
                    let value = self.translate_expression(&assignment.value)?;
                    plan = LogicalOperator::SetProperty(SetPropertyOp {
                        variable: assignment.variable.clone(),
                        properties: vec![(assignment.property.clone(), value)],
                        replace: false,
                        is_edge: false,
                        input: Box::new(plan),
                    });
                }
                for map_assign in &set_clause.map_assignments {
                    let value = self.translate_expression(&map_assign.map_expr)?;
                    plan = LogicalOperator::SetProperty(SetPropertyOp {
                        variable: map_assign.variable.clone(),
                        properties: vec![("*".to_string(), value)],
                        replace: map_assign.replace,
                        is_edge: false,
                        input: Box::new(plan),
                    });
                }
                for label_op in &set_clause.label_operations {
                    plan = LogicalOperator::AddLabel(AddLabelOp {
                        variable: label_op.variable.clone(),
                        labels: label_op.labels.clone(),
                        input: Box::new(plan),
                    });
                }
            }

            for create_clause in &query.create_clauses {
                plan = self.translate_create_patterns(&create_clause.patterns, plan)?;
            }

            for delete_clause in &query.delete_clauses {
                plan = self.translate_delete_targets(
                    &delete_clause.targets,
                    delete_clause.detach,
                    plan,
                )?;
            }
        }

        // REMOVE clauses (not yet in ordered_clauses, always process)
        for remove_clause in &query.remove_clauses {
            for label_op in &remove_clause.label_operations {
                plan = LogicalOperator::RemoveLabel(RemoveLabelOp {
                    variable: label_op.variable.clone(),
                    labels: label_op.labels.clone(),
                    input: Box::new(plan),
                });
            }
            for (variable, property) in &remove_clause.property_removals {
                plan = LogicalOperator::SetProperty(SetPropertyOp {
                    variable: variable.clone(),
                    properties: vec![(property.clone(), LogicalExpression::Literal(Value::Null))],
                    replace: false,
                    is_edge: false,
                    input: Box::new(plan),
                });
            }
        }

        // Handle WITH clauses (projection for query chaining)
        for with_clause in &query.with_clauses {
            if !with_clause.is_wildcard {
                // Check if WITH contains aggregate functions (e.g. WITH count(n) AS cnt)
                let has_aggregates = with_clause
                    .items
                    .iter()
                    .any(|item| contains_aggregate(&item.expression));

                if has_aggregates {
                    let (aggregates, auto_group_by, post_return) =
                        self.extract_aggregates_and_groups(&with_clause.items, false)?;

                    // Split the WHERE into HAVING (aggregate-referencing
                    // conjuncts) and a post-aggregate filter (the rest).
                    // This handles mixed predicates like
                    // `WHERE a.name = 'Alix' AND cnt > 2` correctly.
                    let aggregate_aliases: Vec<String> =
                        aggregates.iter().filter_map(|a| a.alias.clone()).collect();
                    let (having, post_agg_filter) =
                        if let Some(where_clause) = &with_clause.where_clause {
                            let pred = self.translate_expression(&where_clause.expression)?;
                            let conjuncts = flatten_and_conjuncts(&pred);
                            let (having_parts, filter_parts): (Vec<_>, Vec<_>) = conjuncts
                                .into_iter()
                                .partition(|c| references_any(c, &aggregate_aliases));
                            (
                                join_and_conjuncts(having_parts.into_iter().cloned().collect()),
                                join_and_conjuncts(filter_parts.into_iter().cloned().collect()),
                            )
                        } else {
                            (None, None)
                        };

                    plan = LogicalOperator::Aggregate(AggregateOp {
                        group_by: auto_group_by,
                        aggregates,
                        input: Box::new(plan),
                        having,
                    });

                    // Apply post-aggregate projection if aggregates were wrapped
                    // in expressions (e.g. WITH count(n) + 1 AS cnt_plus_one)
                    if let Some(post_items) = post_return {
                        let post_projections: Vec<Projection> = post_items
                            .into_iter()
                            .map(|item| Projection {
                                expression: item.expression,
                                alias: item.alias,
                            })
                            .collect();
                        plan = LogicalOperator::Project(ProjectOp {
                            projections: post_projections,
                            input: Box::new(plan),
                            pass_through_input: false,
                        });
                    }

                    // Apply non-aggregate WHERE conjuncts as a post-aggregate filter.
                    if let Some(filter_pred) = post_agg_filter {
                        plan = wrap_filter(plan, filter_pred);
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

                    plan = LogicalOperator::Project(ProjectOp {
                        projections,
                        input: Box::new(plan),
                        pass_through_input: false,
                    });
                }
            }
            // WITH * skips projection: all variables pass through unchanged

            // Handle LET bindings attached to this WITH clause.
            // LET adds new columns without replacing existing ones.
            if !with_clause.let_bindings.is_empty() {
                let mut let_projections = Vec::new();
                for (name, expr) in &with_clause.let_bindings {
                    let logical_expr = self.translate_expression(expr)?;
                    let_projections.push(Projection {
                        expression: logical_expr,
                        alias: Some(name.clone()),
                    });
                }
                plan = LogicalOperator::Project(ProjectOp {
                    projections: let_projections,
                    input: Box::new(plan),
                    pass_through_input: true,
                });
            }

            // Apply WHERE filter if present in WITH clause.
            // For aggregate WITH clauses, the WHERE was already split into
            // HAVING + post-aggregate filter above, so skip here.
            if let Some(where_clause) = &with_clause.where_clause {
                let has_agg = with_clause
                    .items
                    .iter()
                    .any(|item| contains_aggregate(&item.expression));
                if !has_agg {
                    let predicate = self.translate_expression(&where_clause.expression)?;
                    plan = wrap_filter(plan, predicate);
                }
            }

            // Handle DISTINCT
            if with_clause.distinct {
                plan = wrap_distinct(plan);
            }
        }

        // FINISH: consume input, return empty result (mutations already applied)
        if query.return_clause.is_finish {
            // Wrap in a Limit(0) to consume input but return no rows
            plan = wrap_limit(plan, 0);
            return Ok(LogicalPlan::new(plan));
        }

        // Check if RETURN contains aggregate functions
        let has_aggregates = !query.return_clause.is_wildcard
            && query
                .return_clause
                .items
                .iter()
                .any(|item| contains_aggregate(&item.expression));

        if has_aggregates {
            // Extract aggregate and group-by expressions.
            // When a return item wraps an aggregate in a binary/unary expression
            // (e.g. `count(n) > 0 AS exists`), we decompose it into:
            //   1. An aggregate (`count(n)` with synthetic alias)
            //   2. A post-aggregate projection (`_agg_0 > 0 AS exists`)
            let (aggregates, auto_group_by, post_return) = self.extract_aggregates_and_groups(
                &query.return_clause.items,
                !query.return_clause.group_by.is_empty(),
            )?;

            // Separate horizontal aggregates (over group-list variables from
            // variable-length paths) from regular aggregates.
            let glv = self.group_list_variables.borrow();
            let mut regular_aggregates = Vec::new();
            for agg_expr in aggregates {
                if let Some(ref expr) = agg_expr.expression
                    && let LogicalExpression::Property { variable, property } = expr
                    && let Some(path_alias) = glv.get(variable)
                {
                    // Same name as a regular aggregate, which the post-Return
                    // built by `extract_aggregates_and_groups` refers to.
                    let alias = agg_expr.alias.clone().unwrap_or_else(|| {
                        crate::query::planner::common::aggregate_column_name(&agg_expr)
                    });
                    plan = LogicalOperator::HorizontalAggregate(HorizontalAggregateOp {
                        list_column: format!("_path_edges_{}", path_alias),
                        entity_kind: EntityKind::Edge,
                        function: agg_expr.function,
                        property: property.clone(),
                        alias,
                        input: Box::new(plan),
                    });
                    continue;
                }
                regular_aggregates.push(agg_expr);
            }
            drop(glv);

            // Use explicit GROUP BY if provided, otherwise use auto-detected
            let group_by = if query.return_clause.group_by.is_empty() {
                auto_group_by
            } else {
                query
                    .return_clause
                    .group_by
                    .iter()
                    .map(|e| self.translate_expression(e))
                    .collect::<Result<Vec<_>>>()?
            };

            // Translate HAVING clause if present
            let having = if let Some(having_clause) = &query.having_clause {
                Some(self.translate_expression(&having_clause.expression)?)
            } else {
                None
            };

            let agg_op = if regular_aggregates.is_empty() && group_by.is_empty() {
                // All aggregates were horizontal, no need for a HashAggregate
                plan
            } else {
                LogicalOperator::Aggregate(AggregateOp {
                    group_by,
                    aggregates: regular_aggregates,
                    input: Box::new(plan),
                    having,
                })
            };

            // Collect aggregate output column names before post_return is consumed.
            // These are used to rewrite ORDER BY property references (e.g. `a.species`)
            // into flat variable references (e.g. Variable("a.species")) since after
            // aggregation the original entity variable no longer exists.
            let agg_output_columns: std::collections::HashSet<String> = post_return
                .as_ref()
                .map(|items| {
                    items
                        .iter()
                        .filter_map(|ri| {
                            ri.alias.clone().or_else(|| {
                                if let LogicalExpression::Variable(v) = &ri.expression {
                                    Some(v.clone())
                                } else {
                                    None
                                }
                            })
                        })
                        .collect()
                })
                .unwrap_or_default();

            if let Some(return_items) = post_return {
                plan = wrap_return(agg_op, return_items, query.return_clause.distinct);
            } else {
                plan = agg_op;
            }

            // Apply ORDER BY for aggregate queries.
            // Sort keys that reference a property on a match variable (e.g. a.species)
            // are rewritten to a flat variable reference when that property matches
            // an aggregate output column, since the entity variable no longer exists
            // after aggregation.
            if let Some(order_by) = &query.return_clause.order_by {
                let keys = order_by
                    .items
                    .iter()
                    .map(|item| {
                        let mut expression = self.translate_expression(&item.expression)?;
                        if let LogicalExpression::Property { .. } = &expression {
                            let col_name =
                                crate::query::planner::common::expression_to_string(&expression);
                            if agg_output_columns.contains(&col_name) {
                                expression = LogicalExpression::Variable(col_name);
                            }
                        }
                        Ok(SortKey {
                            expression,
                            order: match item.order {
                                ast::SortOrder::Asc => SortOrder::Ascending,
                                ast::SortOrder::Desc => SortOrder::Descending,
                            },
                            nulls: item.nulls.map(|n| match n {
                                ast::NullsOrdering::First => NullsOrdering::First,
                                ast::NullsOrdering::Last => NullsOrdering::Last,
                            }),
                        })
                    })
                    .collect::<Result<Vec<_>>>()?;

                plan = wrap_sort(plan, keys);
            }

            // Note: For aggregate queries, we don't add a Return operator
            // because Aggregate already produces the final output
        } else {
            // Apply RETURN first (closest to input), then Sort wraps it.
            // This ensures RETURN aliases are visible to ORDER BY in the binder.
            let mut return_items = if query.return_clause.is_wildcard {
                // RETURN *: emit a wildcard marker that the planner expands
                vec![ReturnItem {
                    expression: LogicalExpression::Variable("*".into()),
                    alias: None,
                }]
            } else {
                query
                    .return_clause
                    .items
                    .iter()
                    .map(|item| {
                        Ok(ReturnItem {
                            expression: self.translate_expression(&item.expression)?,
                            alias: item.alias.clone(),
                        })
                    })
                    .collect::<Result<Vec<_>>>()?
            };

            // Lift VALUE subqueries: wrap with Apply and replace expression
            // with a Variable reference to the inner plan's output column.
            for item in &mut return_items {
                if let LogicalExpression::ValueSubquery(inner_plan) = &item.expression {
                    // Determine the output column name from the inner plan's RETURN
                    let col_name =
                        Self::extract_return_column_name(inner_plan).unwrap_or_else(|| {
                            item.alias.clone().unwrap_or_else(|| "__value".to_string())
                        });

                    plan = LogicalOperator::Apply(ApplyOp {
                        input: Box::new(plan),
                        subplan: inner_plan.clone(),
                        shared_variables: vec![],
                        optional: false,
                    });

                    item.expression = LogicalExpression::Variable(col_name);
                }
            }

            plan = wrap_return(plan, return_items, query.return_clause.distinct);

            // Apply ORDER BY (wraps Return so aliases are visible)
            if let Some(order_by) = &query.return_clause.order_by {
                let keys = order_by
                    .items
                    .iter()
                    .map(|item| {
                        Ok(SortKey {
                            expression: self.translate_expression(&item.expression)?,
                            order: match item.order {
                                ast::SortOrder::Asc => SortOrder::Ascending,
                                ast::SortOrder::Desc => SortOrder::Descending,
                            },
                            nulls: item.nulls.map(|n| match n {
                                ast::NullsOrdering::First => NullsOrdering::First,
                                ast::NullsOrdering::Last => NullsOrdering::Last,
                            }),
                        })
                    })
                    .collect::<Result<Vec<_>>>()?;

                plan = wrap_sort(plan, keys);
            }
        }

        if let Some(skip_expr) = &query.return_clause.skip {
            plan = wrap_skip(plan, Self::eval_as_count_expr(skip_expr)?);
        }

        if let Some(limit_expr) = &query.return_clause.limit {
            plan = wrap_limit(plan, Self::eval_as_count_expr(limit_expr)?);
        }

        Ok(LogicalPlan::new(plan))
    }

    fn eval_as_count_expr(expr: &ast::Expression) -> Result<CountExpr> {
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

    fn substitute_let_bindings(
        expr: LogicalExpression,
        bindings: &[(String, LogicalExpression)],
    ) -> LogicalExpression {
        match expr {
            LogicalExpression::Variable(ref name) => {
                for (bind_name, bind_expr) in bindings {
                    if bind_name == name {
                        return bind_expr.clone();
                    }
                }
                expr
            }
            LogicalExpression::Binary { left, op, right } => LogicalExpression::Binary {
                left: Box::new(Self::substitute_let_bindings(*left, bindings)),
                op,
                right: Box::new(Self::substitute_let_bindings(*right, bindings)),
            },
            LogicalExpression::Unary { op, operand } => LogicalExpression::Unary {
                op,
                operand: Box::new(Self::substitute_let_bindings(*operand, bindings)),
            },
            LogicalExpression::FunctionCall {
                name,
                args,
                distinct,
            } => LogicalExpression::FunctionCall {
                name,
                args: args
                    .into_iter()
                    .map(|a| Self::substitute_let_bindings(a, bindings))
                    .collect(),
                distinct,
            },
            other => other,
        }
    }

    /// Extracts all named variables from a GQL AST pattern.
    fn pattern_variables(pattern: &ast::Pattern) -> HashSet<String> {
        let mut vars = HashSet::new();
        match pattern {
            ast::Pattern::Node(node) => {
                if let Some(v) = &node.variable {
                    vars.insert(v.clone());
                }
            }
            ast::Pattern::Path(path) => {
                if let Some(v) = &path.source.variable {
                    vars.insert(v.clone());
                }
                for edge in &path.edges {
                    if let Some(v) = &edge.variable {
                        vars.insert(v.clone());
                    }
                    if let Some(v) = &edge.target.variable {
                        vars.insert(v.clone());
                    }
                }
            }
            ast::Pattern::Quantified {
                pattern,
                subpath_var,
                ..
            } => {
                if let Some(v) = subpath_var {
                    vars.insert(v.clone());
                }
                vars.extend(Self::pattern_variables(pattern));
            }
            ast::Pattern::Union(patterns) | ast::Pattern::MultisetUnion(patterns) => {
                for p in patterns {
                    vars.extend(Self::pattern_variables(p));
                }
            }
        }
        vars
    }

    /// Translates a MATCH clause with an optional initial input.
    ///
    /// When `initial_input` is provided (e.g. from a preceding UNWIND), the
    /// first pattern's NodeScan receives it as input. This creates a nested
    /// loop join that keeps prior variables (like UNWIND variables) in scope
    /// so that property filters like `{id: x}` can reference them.
    ///
    /// When multiple comma-separated patterns share variables, creates proper
    /// `JoinOp` operators with equality conditions instead of cross products.
    fn translate_match_with_input(
        &self,
        match_clause: &ast::MatchClause,
        initial_input: Option<LogicalOperator>,
    ) -> Result<LogicalOperator> {
        let mut path_mode = match match_clause.path_mode {
            Some(ast::PathMode::Walk) | None => PathMode::Walk,
            Some(ast::PathMode::Trail) => PathMode::Trail,
            Some(ast::PathMode::Simple) => PathMode::Simple,
            Some(ast::PathMode::Acyclic) => PathMode::Acyclic,
        };

        // Match mode overrides path mode: DIFFERENT EDGES = Trail, REPEATABLE ELEMENTS = Walk
        if let Some(mode) = &match_clause.match_mode {
            match mode {
                ast::MatchMode::DifferentEdges => path_mode = PathMode::Trail,
                ast::MatchMode::RepeatableElements => path_mode = PathMode::Walk,
            }
        }

        // Search prefix determines whether we use shortest path operators
        let use_shortest = matches!(
            &match_clause.search_prefix,
            Some(
                ast::PathSearchPrefix::AnyShortest
                    | ast::PathSearchPrefix::AllShortest
                    | ast::PathSearchPrefix::ShortestK(_)
                    | ast::PathSearchPrefix::ShortestKGroups(_)
            )
        );
        // ANY returns any single matching path (not necessarily shortest)
        let use_any_limit = matches!(
            &match_clause.search_prefix,
            Some(ast::PathSearchPrefix::Any)
        );

        // Collect variables for each pattern to detect shared variables
        let pattern_vars: Vec<HashSet<String>> = match_clause
            .patterns
            .iter()
            .map(|ap| Self::pattern_variables(&ap.pattern))
            .collect();

        let mut plan: Option<LogicalOperator> = initial_input;
        let mut bound_vars: HashSet<String> = HashSet::new();

        for (index, aliased_pattern) in match_clause.patterns.iter().enumerate() {
            let current_vars = &pattern_vars[index];
            let shared: Vec<String> = current_vars.intersection(&bound_vars).cloned().collect();

            // Determine the input for this pattern: if shared variables exist,
            // translate independently and join; otherwise chain as before.
            let pattern_input = if shared.is_empty() { plan.take() } else { None };

            // Check per-pattern search prefix (e.g., p = ANY SHORTEST (...))
            let per_pattern_shortest = matches!(
                &aliased_pattern.search_prefix,
                Some(
                    ast::PathSearchPrefix::AnyShortest
                        | ast::PathSearchPrefix::AllShortest
                        | ast::PathSearchPrefix::ShortestK(_)
                        | ast::PathSearchPrefix::ShortestKGroups(_)
                )
            );

            let pattern_plan = if let Some(path_function) = &aliased_pattern.path_function {
                self.translate_shortest_path(
                    &aliased_pattern.pattern,
                    aliased_pattern.alias.as_deref(),
                    *path_function,
                    pattern_input,
                )?
            } else if use_shortest || per_pattern_shortest {
                let prefix = aliased_pattern
                    .search_prefix
                    .as_ref()
                    .or(match_clause.search_prefix.as_ref());
                let pf = match prefix {
                    Some(ast::PathSearchPrefix::AllShortest) => ast::PathFunction::AllShortestPaths,
                    _ => ast::PathFunction::ShortestPath,
                };
                self.translate_shortest_path(
                    &aliased_pattern.pattern,
                    aliased_pattern.alias.as_deref(),
                    pf,
                    pattern_input,
                )?
            } else {
                self.translate_pattern_with_alias(
                    &aliased_pattern.pattern,
                    pattern_input,
                    aliased_pattern.alias.as_deref(),
                    path_mode,
                )?
            };

            if !shared.is_empty() {
                // Join on shared variables
                let left = plan
                    .take()
                    .expect("bound_vars non-empty implies plan exists");
                let conditions = shared
                    .iter()
                    .map(|var| JoinCondition {
                        left: LogicalExpression::Variable(var.clone()),
                        right: LogicalExpression::Variable(var.clone()),
                    })
                    .collect();
                plan = Some(LogicalOperator::Join(JoinOp {
                    left: Box::new(left),
                    right: Box::new(pattern_plan),
                    join_type: JoinType::Inner,
                    conditions,
                }));
            } else {
                plan = Some(pattern_plan);
            }

            bound_vars.extend(current_vars.iter().cloned());
        }

        let mut result = plan.ok_or_else(|| {
            Error::Query(QueryError::new(
                QueryErrorKind::Semantic,
                "Empty MATCH clause",
            ))
        })?;

        // ANY (without SHORTEST): wrap in LIMIT 1 to return a single matching path.
        // ANY SHORTEST uses ShortestPathOperator which inherently returns one
        // path per source/target pair, so no LIMIT 1 wrapper is needed.
        if use_any_limit {
            result = wrap_limit(result, 1);
        }

        Ok(result)
    }

    /// Translates `CALL { subquery }` to an Apply operator with proper scope.
    ///
    /// When the subquery starts with `WITH <vars>`, the variables are treated
    /// Translates a `LoadDataClause` to a `LoadData` logical operator.
    fn translate_load_data(&self, load: &ast::LoadDataClause) -> LogicalOperator {
        let format = match load.format {
            ast::LoadFormat::Csv => LoadDataFormat::Csv,
            ast::LoadFormat::Jsonl => LoadDataFormat::Jsonl,
            ast::LoadFormat::Parquet => LoadDataFormat::Parquet,
        };
        LogicalOperator::LoadData(LoadDataOp {
            format,
            with_headers: load.with_headers,
            path: load.path.clone(),
            variable: load.variable.clone(),
            field_terminator: load.field_terminator,
        })
    }

    /// as imports from the outer scope: a `ParameterScan` replaces `Empty` as
    /// the inner plan root and `shared_variables` is populated so the planner
    /// can wire them through `ParameterState`.
    fn translate_inline_call(
        &self,
        subquery: &ast::QueryStatement,
        outer: LogicalOperator,
        optional: bool,
    ) -> Result<LogicalOperator> {
        let has_outer = !matches!(outer, LogicalOperator::Empty);

        // Detect importing WITH: extract shared variable names and skip it.
        let mut shared_variables = Vec::new();
        let skip_with = if has_outer && !subquery.with_clauses.is_empty() {
            let first_with = &subquery.with_clauses[0];
            if first_with.is_wildcard {
                shared_variables.push("*".to_string());
                true
            } else {
                for item in &first_with.items {
                    if let ast::Expression::Variable(name) = &item.expression {
                        let var_name = item.alias.as_deref().unwrap_or(name);
                        shared_variables.push(var_name.to_string());
                    }
                }
                !shared_variables.is_empty()
            }
        } else {
            false
        };

        // Build the inner plan: start from ParameterScan when importing variables.
        let inner_plan = if skip_with && !shared_variables.is_empty() {
            // Translate the subquery but override the first WITH clause:
            // start from ParameterScan instead of Empty, skip the importing WITH.
            let mut plan = LogicalOperator::ParameterScan(ParameterScanOp {
                columns: shared_variables.clone(),
            });

            // Process MATCH clauses
            for match_clause in &subquery.match_clauses {
                if match_clause.optional {
                    let match_plan = self.translate_match(match_clause)?;
                    plan = LogicalOperator::LeftJoin(LeftJoinOp {
                        left: Box::new(plan),
                        right: Box::new(match_plan),
                        condition: None,
                    });
                } else {
                    let input = std::mem::replace(&mut plan, LogicalOperator::Empty);
                    plan = self.translate_match_with_input(match_clause, Some(input))?;
                }
            }

            // Apply WHERE filter
            if let Some(where_clause) = &subquery.where_clause {
                let predicate = self.translate_expression(&where_clause.expression)?;
                plan = wrap_filter(plan, predicate);
            }

            // Process remaining WITH clauses (skip the first importing one)
            for with_clause in subquery.with_clauses.iter().skip(1) {
                if !with_clause.is_wildcard {
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
                    plan = LogicalOperator::Project(ProjectOp {
                        projections,
                        input: Box::new(plan),
                        pass_through_input: false,
                    });
                }
                // Handle LET bindings in inline call WITH clause
                if !with_clause.let_bindings.is_empty() {
                    let mut let_projections = Vec::new();
                    for (name, expr) in &with_clause.let_bindings {
                        let logical_expr = self.translate_expression(expr)?;
                        let_projections.push(Projection {
                            expression: logical_expr,
                            alias: Some(name.clone()),
                        });
                    }
                    plan = LogicalOperator::Project(ProjectOp {
                        projections: let_projections,
                        input: Box::new(plan),
                        pass_through_input: true,
                    });
                }
                if let Some(wc) = &with_clause.where_clause {
                    let predicate = self.translate_expression(&wc.expression)?;
                    plan = wrap_filter(plan, predicate);
                }
            }

            // Translate RETURN clause
            let has_aggregates = !subquery.return_clause.is_wildcard
                && subquery
                    .return_clause
                    .items
                    .iter()
                    .any(|item| contains_aggregate(&item.expression));

            if has_aggregates {
                let (aggregates, auto_group_by, post_return) = self.extract_aggregates_and_groups(
                    &subquery.return_clause.items,
                    !subquery.return_clause.group_by.is_empty(),
                )?;
                let group_by = if subquery.return_clause.group_by.is_empty() {
                    auto_group_by
                } else {
                    subquery
                        .return_clause
                        .group_by
                        .iter()
                        .map(|e| self.translate_expression(e))
                        .collect::<Result<Vec<_>>>()?
                };
                let agg_op = LogicalOperator::Aggregate(AggregateOp {
                    group_by,
                    aggregates,
                    input: Box::new(plan),
                    having: None,
                });
                plan = if let Some(return_items) = post_return {
                    wrap_return(agg_op, return_items, subquery.return_clause.distinct)
                } else {
                    agg_op
                };
            } else {
                let return_items = subquery
                    .return_clause
                    .items
                    .iter()
                    .map(|item| {
                        Ok(ReturnItem {
                            expression: self.translate_expression(&item.expression)?,
                            alias: item.alias.clone(),
                        })
                    })
                    .collect::<Result<Vec<_>>>()?;
                plan = wrap_return(plan, return_items, subquery.return_clause.distinct);
            }
            plan
        } else {
            // No importing WITH: translate the entire subquery independently
            self.translate_query(subquery)?.root
        };

        // Wire the inner plan to the outer plan
        if has_outer {
            Ok(LogicalOperator::Apply(ApplyOp {
                input: Box::new(outer),
                subplan: Box::new(inner_plan),
                shared_variables,
                optional,
            }))
        } else {
            // No outer input: just use the inner plan directly
            Ok(inner_plan)
        }
    }

    fn translate_match(&self, match_clause: &ast::MatchClause) -> Result<LogicalOperator> {
        self.translate_match_with_input(match_clause, None)
    }

    /// Applies a WHERE predicate with awareness of LeftJoin semantics.
    ///
    /// When the current plan ends with a LeftJoin (from OPTIONAL MATCH),
    /// right-side predicates are pushed into the join instead of being
    /// placed as a post-filter (which would incorrectly eliminate NULL rows).
    fn apply_where_with_left_join_awareness(
        &self,
        plan: LogicalOperator,
        predicate: LogicalExpression,
    ) -> LogicalOperator {
        if let LogicalOperator::LeftJoin(left_join) = plan {
            let (join, post_filter) =
                build_left_join_with_predicates(*left_join.left, *left_join.right, Some(predicate));
            if let Some(pf) = post_filter {
                wrap_filter(join, pf)
            } else {
                join
            }
        } else {
            wrap_filter(plan, predicate)
        }
    }

    /// Translates a shortestPath pattern into a logical operator.
    fn translate_shortest_path(
        &self,
        pattern: &ast::Pattern,
        alias: Option<&str>,
        path_function: ast::PathFunction,
        input: Option<LogicalOperator>,
    ) -> Result<LogicalOperator> {
        // Extract source and target from the pattern
        let (source_node, target_node, edge_types, direction) = match pattern {
            ast::Pattern::Path(path) => {
                let target_node = if let Some(edge) = path.edges.last() {
                    &edge.target
                } else {
                    return Err(Error::Query(QueryError::new(
                        QueryErrorKind::Semantic,
                        "shortestPath requires a path pattern",
                    )));
                };
                let edge_types = path
                    .edges
                    .first()
                    .map(|e| e.types.clone())
                    .unwrap_or_default();
                let direction =
                    path.edges
                        .first()
                        .map_or(ExpandDirection::Both, |e| match e.direction {
                            ast::EdgeDirection::Outgoing => ExpandDirection::Outgoing,
                            ast::EdgeDirection::Incoming => ExpandDirection::Incoming,
                            ast::EdgeDirection::Undirected => ExpandDirection::Both,
                        });
                (&path.source, target_node, edge_types, direction)
            }
            ast::Pattern::Node(_)
            | ast::Pattern::Quantified { .. }
            | ast::Pattern::Union(_)
            | ast::Pattern::MultisetUnion(_) => {
                return Err(Error::Query(QueryError::new(
                    QueryErrorKind::Semantic,
                    "shortestPath requires a simple path pattern",
                )));
            }
        };

        // Get variable names
        let source_var = source_node
            .variable
            .clone()
            .unwrap_or_else(|| format!("_anon_{}", rand_id()));
        let target_var = target_node
            .variable
            .clone()
            .unwrap_or_else(|| format!("_anon_{}", rand_id()));

        // For shortestPath, we need to scan source and target nodes separately
        // (not expand between them - the ShortestPathOperator will find the path)

        // Scan source node first
        let source_plan = self.translate_node_pattern(source_node, input)?;

        // Scan target node (cross-product with source)
        let target_plan = self.translate_node_pattern(target_node, Some(source_plan))?;

        // Wrap with ShortestPath operator
        Ok(LogicalOperator::ShortestPath(ShortestPathOp {
            input: Box::new(target_plan),
            source_var,
            target_var,
            edge_types,
            direction,
            path_alias: alias.unwrap_or("_path").to_string(),
            all_paths: matches!(path_function, ast::PathFunction::AllShortestPaths),
        }))
    }

    fn translate_pattern_with_alias(
        &self,
        pattern: &ast::Pattern,
        input: Option<LogicalOperator>,
        path_alias: Option<&str>,
        path_mode: PathMode,
    ) -> Result<LogicalOperator> {
        match pattern {
            ast::Pattern::Node(node) => self.translate_node_pattern(node, input),
            ast::Pattern::Path(path) => {
                self.translate_path_pattern_with_alias(path, input, path_alias, path_mode)
            }
            ast::Pattern::Quantified {
                pattern,
                min,
                max,
                subpath_var,
                path_mode: inner_path_mode,
                where_clause,
            } => {
                // G049: Inner path mode overrides the outer path mode if set.
                let effective_mode = inner_path_mode.as_ref().map_or(path_mode, |m| match m {
                    ast::PathMode::Walk => PathMode::Walk,
                    ast::PathMode::Trail => PathMode::Trail,
                    ast::PathMode::Simple => PathMode::Simple,
                    ast::PathMode::Acyclic => PathMode::Acyclic,
                });

                // G048: subpath_var takes precedence as path alias for the
                // quantified pattern. Fall back to outer path_alias if not set.
                let effective_alias = subpath_var.as_deref().or(path_alias);

                // Quantified path pattern: repeat the inner pattern min..max times.
                // For now, translate as a variable-length expansion if the inner
                // pattern is a simple single-edge path.
                let mut result = match pattern.as_ref() {
                    ast::Pattern::Path(path) if path.edges.len() == 1 => {
                        // Single-edge quantified: equivalent to variable-length edge
                        let mut modified_path = path.clone();
                        let edge = &mut modified_path.edges[0];
                        edge.min_hops = Some(*min);
                        edge.max_hops = *max;
                        self.translate_path_pattern_with_alias(
                            &modified_path,
                            input,
                            effective_alias,
                            effective_mode,
                        )
                    }
                    _ => {
                        // Multi-edge or complex quantified patterns: translate the
                        // inner pattern once (future: iterate with backtracking).
                        self.translate_pattern_with_alias(
                            pattern,
                            input,
                            effective_alias,
                            effective_mode,
                        )
                    }
                }?;

                // G050: Apply WHERE clause as a filter on the quantified pattern output
                if let Some(where_expr) = where_clause {
                    let filter_expr = self.translate_expression(where_expr)?;
                    result = wrap_filter(result, filter_expr);
                }

                Ok(result)
            }
            ast::Pattern::Union(patterns) => {
                // Union of alternative patterns: UNION ALL of each translated pattern
                let inputs: Vec<LogicalOperator> = patterns
                    .iter()
                    .map(|p| {
                        self.translate_pattern_with_alias(p, input.clone(), path_alias, path_mode)
                    })
                    .collect::<Result<Vec<_>>>()?;
                Ok(LogicalOperator::Union(UnionOp { inputs }))
            }
            ast::Pattern::MultisetUnion(patterns) => {
                // G030: Multiset (bag) union preserves duplicates
                let inputs: Vec<LogicalOperator> = patterns
                    .iter()
                    .map(|p| {
                        self.translate_pattern_with_alias(p, input.clone(), path_alias, path_mode)
                    })
                    .collect::<Result<Vec<_>>>()?;
                Ok(LogicalOperator::Union(UnionOp { inputs }))
            }
        }
    }

    fn translate_data_modification(
        &self,
        dm: &ast::DataModificationStatement,
    ) -> Result<LogicalPlan> {
        match dm {
            ast::DataModificationStatement::Insert(insert) => self.translate_insert(insert),
            ast::DataModificationStatement::Delete(delete) => self.translate_delete(delete),
            ast::DataModificationStatement::Set(set) => self.translate_set(set),
        }
    }

    fn translate_delete(&self, delete: &ast::DeleteStatement) -> Result<LogicalPlan> {
        // DELETE requires a preceding MATCH clause to identify what to delete.
        // For standalone DELETE, we need to scan and delete the specified variables.
        // This is typically used as: MATCH (n:Label) DELETE n

        if delete.targets.is_empty() {
            return Err(Error::Query(QueryError::new(
                QueryErrorKind::Semantic,
                "DELETE requires at least one target",
            )));
        }

        // Extract the first variable name for the scan
        let first_var = match &delete.targets[0] {
            ast::DeleteTarget::Variable(name) => name.clone(),
            ast::DeleteTarget::Expression(_) => "__delete_expr_0".to_string(),
        };

        // Create a scan to find the entities to delete
        let scan = LogicalOperator::NodeScan(NodeScanOp {
            variable: first_var.clone(),
            label: None,
            input: None,
        });

        let plan = self.translate_delete_targets(&delete.targets, delete.detach, scan)?;
        Ok(LogicalPlan::new(plan))
    }

    /// Translates a list of delete targets into a chain of delete operators.
    /// For `DeleteTarget::Variable`, emits a `DeleteNodeOp` directly.
    /// For `DeleteTarget::Expression` (GD04), projects the expression into a
    /// synthetic variable then deletes that variable.
    fn translate_delete_targets(
        &self,
        targets: &[ast::DeleteTarget],
        detach: bool,
        mut plan: LogicalOperator,
    ) -> Result<LogicalOperator> {
        for (i, target) in targets.iter().enumerate() {
            match target {
                ast::DeleteTarget::Variable(name) => {
                    plan = LogicalOperator::DeleteNode(DeleteNodeOp {
                        variable: name.clone(),
                        detach,
                        input: Box::new(plan),
                    });
                }
                ast::DeleteTarget::Expression(expr) => {
                    // GD04: evaluate the expression, bind to a synthetic variable,
                    // then delete that variable.
                    let synthetic_var = format!("__delete_expr_{i}");
                    let logical_expr = self.translate_expression(expr)?;
                    plan = LogicalOperator::Project(ProjectOp {
                        projections: vec![Projection {
                            expression: logical_expr,
                            alias: Some(synthetic_var.clone()),
                        }],
                        input: Box::new(plan),
                        pass_through_input: true,
                    });
                    plan = LogicalOperator::DeleteNode(DeleteNodeOp {
                        variable: synthetic_var,
                        detach,
                        input: Box::new(plan),
                    });
                }
            }
        }
        Ok(plan)
    }

    fn translate_set(&self, set: &ast::SetStatement) -> Result<LogicalPlan> {
        // SET requires a preceding MATCH clause to identify what to update.
        // For standalone SET, we error - it should be part of a query.

        if set.assignments.is_empty() {
            return Err(Error::Query(QueryError::new(
                QueryErrorKind::Semantic,
                "SET requires at least one assignment",
            )));
        }

        // Group assignments by variable
        let first_assignment = &set.assignments[0];
        let var = &first_assignment.variable;

        // Create a scan to find the entity to update
        let scan = LogicalOperator::NodeScan(NodeScanOp {
            variable: var.clone(),
            label: None,
            input: None,
        });

        // Build property assignments for this variable
        let properties: Vec<(String, LogicalExpression)> = set
            .assignments
            .iter()
            .filter(|a| &a.variable == var)
            .map(|a| Ok((a.property.clone(), self.translate_expression(&a.value)?)))
            .collect::<Result<_>>()?;

        let plan = LogicalOperator::SetProperty(SetPropertyOp {
            variable: var.clone(),
            properties,
            replace: false,
            is_edge: false,
            input: Box::new(scan),
        });

        Ok(LogicalPlan::new(plan))
    }

    fn translate_insert(&self, insert: &ast::InsertStatement) -> Result<LogicalPlan> {
        let (plan, last_variable) = self.insert_chain(&insert.patterns)?;
        let ret = wrap_return(
            plan,
            vec![ReturnItem {
                expression: LogicalExpression::Variable(last_variable),
                alias: None,
            }],
            false,
        );
        Ok(LogicalPlan::new(ret))
    }

    /// Builds the CreateNode / CreateEdge chain of an INSERT that starts a
    /// statement (no input rows). Returns the plan and the last variable
    /// created. Used for a standalone INSERT and for the first INSERT clause of
    /// a query such as `INSERT (a) INSERT (b) RETURN a, b`.
    fn insert_chain(&self, patterns: &[ast::Pattern]) -> Result<(LogicalOperator, String)> {
        if patterns.is_empty() {
            return Err(Error::Query(QueryError::new(
                QueryErrorKind::Semantic,
                "Empty INSERT statement",
            )));
        }

        // Chain CreateNode operators for all patterns.
        // First pattern gets input: None, subsequent ones chain via input: Some(prev).
        let mut plan: Option<LogicalOperator> = None;
        let mut last_variable = String::new();

        for pattern in patterns {
            match pattern {
                ast::Pattern::Node(node) => {
                    let variable = node
                        .variable
                        .clone()
                        .unwrap_or_else(|| format!("_anon_{}", rand_id()));

                    let properties = node
                        .properties
                        .iter()
                        .map(|(k, v)| Ok((k.clone(), self.translate_expression(v)?)))
                        .collect::<Result<Vec<_>>>()?;

                    plan = Some(LogicalOperator::CreateNode(CreateNodeOp {
                        variable: variable.clone(),
                        labels: node.labels.clone(),
                        properties,
                        input: plan.map(Box::new),
                    }));
                    last_variable = variable;
                }
                ast::Pattern::Path(path) => {
                    // Decompose path into CreateNode + CreateEdge chain
                    let source_var = path
                        .source
                        .variable
                        .clone()
                        .unwrap_or_else(|| format!("_anon_{}", rand_id()));

                    if !path.source.labels.is_empty() {
                        let source_props: Vec<(String, LogicalExpression)> = path
                            .source
                            .properties
                            .iter()
                            .map(|(k, v)| Ok((k.clone(), self.translate_expression(v)?)))
                            .collect::<Result<Vec<_>>>()?;
                        plan = Some(LogicalOperator::CreateNode(CreateNodeOp {
                            variable: source_var.clone(),
                            labels: path.source.labels.clone(),
                            properties: source_props,
                            input: plan.map(Box::new),
                        }));
                    }

                    let mut current_src = source_var;
                    for edge in &path.edges {
                        let target_var = edge
                            .target
                            .variable
                            .clone()
                            .unwrap_or_else(|| format!("_anon_{}", rand_id()));

                        if !edge.target.labels.is_empty() {
                            let target_props: Vec<(String, LogicalExpression)> = edge
                                .target
                                .properties
                                .iter()
                                .map(|(k, v)| Ok((k.clone(), self.translate_expression(v)?)))
                                .collect::<Result<Vec<_>>>()?;
                            plan = Some(LogicalOperator::CreateNode(CreateNodeOp {
                                variable: target_var.clone(),
                                labels: edge.target.labels.clone(),
                                properties: target_props,
                                input: plan.map(Box::new),
                            }));
                        }

                        let edge_type = edge.types.first().cloned().unwrap_or_default();
                        let edge_props: Vec<(String, LogicalExpression)> = edge
                            .properties
                            .iter()
                            .map(|(k, v)| Ok((k.clone(), self.translate_expression(v)?)))
                            .collect::<Result<Vec<_>>>()?;

                        let (from, to) = match edge.direction {
                            ast::EdgeDirection::Incoming => (target_var.clone(), current_src),
                            _ => (current_src, target_var.clone()),
                        };

                        plan = Some(LogicalOperator::CreateEdge(CreateEdgeOp {
                            variable: edge.variable.clone(),
                            edge_type,
                            from_variable: from,
                            to_variable: to,
                            properties: edge_props,
                            input: Box::new(plan.unwrap_or(LogicalOperator::Empty)),
                        }));
                        last_variable.clone_from(&target_var);
                        current_src = target_var;
                    }
                }
                ast::Pattern::Quantified { .. }
                | ast::Pattern::Union(_)
                | ast::Pattern::MultisetUnion(_) => {
                    return Err(Error::Query(QueryError::new(
                        QueryErrorKind::Semantic,
                        "INSERT does not support quantified or union patterns",
                    )));
                }
            }
        }

        let plan = plan.ok_or_else(|| {
            Error::Query(QueryError::new(
                QueryErrorKind::Semantic,
                "INSERT must create at least one node",
            ))
        })?;
        Ok((plan, last_variable))
    }

    /// Translates a subquery to a logical operator (without Return).
    ///
    /// When the WHERE clause references variables not defined by the inner MATCH
    /// patterns, a `ParameterScan` join is added so the filter planner can set up
    /// correlated execution via `ApplyOperator`.
    fn translate_subquery_to_operator(
        &self,
        query: &ast::QueryStatement,
    ) -> Result<LogicalOperator> {
        let mut plan = LogicalOperator::Empty;

        // Collect variables defined by inner MATCH patterns
        let mut inner_defined = std::collections::HashSet::new();
        for match_clause in &query.match_clauses {
            Self::collect_pattern_variables(&match_clause.patterns, &mut inner_defined);
            let match_plan = self.translate_match(match_clause)?;
            plan = if matches!(plan, LogicalOperator::Empty) {
                match_plan
            } else {
                LogicalOperator::Join(JoinOp {
                    left: Box::new(plan),
                    right: Box::new(match_plan),
                    join_type: JoinType::Cross,
                    conditions: vec![],
                })
            };
        }

        if let Some(where_clause) = &query.where_clause {
            // Detect outer variable references in WHERE
            let mut referenced = std::collections::HashSet::new();
            Self::collect_ast_expression_variables(&where_clause.expression, &mut referenced);
            let outer_refs: Vec<String> = referenced.difference(&inner_defined).cloned().collect();

            // If there are outer references, add ParameterScan for correlation
            if !outer_refs.is_empty() {
                plan = LogicalOperator::Join(JoinOp {
                    left: Box::new(LogicalOperator::ParameterScan(ParameterScanOp {
                        columns: outer_refs,
                    })),
                    right: Box::new(plan),
                    join_type: JoinType::Cross,
                    conditions: vec![],
                });
            }

            let predicate = self.translate_expression(&where_clause.expression)?;
            plan = wrap_filter(plan, predicate);
        }

        Ok(plan)
    }

    /// Returns true if the RETURN clause is a single count() aggregate.
    fn is_count_aggregate_return(ret: &ast::ReturnClause) -> bool {
        if ret.items.len() != 1 {
            return false;
        }
        matches!(
            &ret.items[0].expression,
            ast::Expression::FunctionCall { name, .. } if name.eq_ignore_ascii_case("count")
        )
    }

    /// Extracts the first output column name from a Return operator in a logical plan.
    fn extract_return_column_name(plan: &LogicalOperator) -> Option<String> {
        match plan {
            LogicalOperator::Return(ret) => {
                let item = ret.items.first()?;
                if let Some(alias) = &item.alias {
                    Some(alias.clone())
                } else {
                    // Derive name from expression
                    match &item.expression {
                        LogicalExpression::Variable(name) => Some(name.clone()),
                        LogicalExpression::Property { variable, property } => {
                            Some(format!("{variable}.{property}"))
                        }
                        _ => None,
                    }
                }
            }
            // Walk through wrapping operators to find the Return
            LogicalOperator::Sort(s) => Self::extract_return_column_name(&s.input),
            LogicalOperator::Limit(l) => Self::extract_return_column_name(&l.input),
            LogicalOperator::Distinct(d) => Self::extract_return_column_name(&d.input),
            _ => None,
        }
    }

    /// Collects variable names defined by match patterns (nodes and edges).
    fn collect_pattern_variables(
        patterns: &[ast::AliasedPattern],
        vars: &mut std::collections::HashSet<String>,
    ) {
        for aliased in patterns {
            Self::collect_pattern_vars_inner(&aliased.pattern, vars);
        }
    }

    /// Recursively collects variables from a single Pattern enum.
    fn collect_pattern_vars_inner(
        pattern: &ast::Pattern,
        vars: &mut std::collections::HashSet<String>,
    ) {
        match pattern {
            ast::Pattern::Node(node) => {
                if let Some(v) = &node.variable {
                    vars.insert(v.clone());
                }
            }
            ast::Pattern::Path(path) => {
                if let Some(v) = &path.source.variable {
                    vars.insert(v.clone());
                }
                for edge in &path.edges {
                    if let Some(v) = &edge.variable {
                        vars.insert(v.clone());
                    }
                    if let Some(v) = &edge.target.variable {
                        vars.insert(v.clone());
                    }
                }
            }
            ast::Pattern::Quantified { pattern: inner, .. } => {
                Self::collect_pattern_vars_inner(inner, vars);
            }
            ast::Pattern::Union(patterns) | ast::Pattern::MultisetUnion(patterns) => {
                for p in patterns {
                    Self::collect_pattern_vars_inner(p, vars);
                }
            }
        }
    }

    /// Collects all variable names referenced in an AST expression (for property access).
    fn collect_ast_expression_variables(
        expr: &ast::Expression,
        vars: &mut std::collections::HashSet<String>,
    ) {
        match expr {
            ast::Expression::PropertyAccess { variable, .. } => {
                vars.insert(variable.clone());
            }
            ast::Expression::Variable(name) => {
                vars.insert(name.clone());
            }
            ast::Expression::Binary { left, right, .. } => {
                Self::collect_ast_expression_variables(left, vars);
                Self::collect_ast_expression_variables(right, vars);
            }
            ast::Expression::Unary { operand, .. } => {
                Self::collect_ast_expression_variables(operand, vars);
            }
            ast::Expression::FunctionCall { args, .. } => {
                for arg in args {
                    Self::collect_ast_expression_variables(arg, vars);
                }
            }
            _ => {}
        }
    }
}

/// Generate a simple random-ish ID for anonymous variables.
fn rand_id() -> u32 {
    use std::sync::atomic::{AtomicU32, Ordering};
    static COUNTER: AtomicU32 = AtomicU32::new(0);
    COUNTER.fetch_add(1, Ordering::Relaxed)
}

use aggregate::contains_aggregate;

#[cfg(test)]
mod tests {
    use super::*;

    // === Basic MATCH Tests ===

    #[test]
    fn test_translate_simple_match() {
        let query = "MATCH (n:Person) RETURN n";
        let result = translate(query);
        assert!(result.is_ok());

        let plan = result.unwrap();
        if let LogicalOperator::Return(ret) = &plan.root {
            assert_eq!(ret.items.len(), 1);
            assert!(!ret.distinct);
        } else {
            panic!("Expected Return operator");
        }
    }

    #[test]
    fn test_translate_match_with_where() {
        let query = "MATCH (n:Person) WHERE n.age > 30 RETURN n.name";
        let result = translate(query);
        assert!(result.is_ok());

        let plan = result.unwrap();
        if let LogicalOperator::Return(ret) = &plan.root {
            // Should have Filter as input
            if let LogicalOperator::Filter(filter) = ret.input.as_ref() {
                if let LogicalExpression::Binary { op, .. } = &filter.predicate {
                    assert_eq!(*op, BinaryOp::Gt);
                } else {
                    panic!("Expected binary expression");
                }
            } else {
                panic!("Expected Filter operator");
            }
        } else {
            panic!("Expected Return operator");
        }
    }

    #[test]
    fn test_translate_match_without_label() {
        let query = "MATCH (n) RETURN n";
        let result = translate(query);
        assert!(result.is_ok());

        let plan = result.unwrap();
        if let LogicalOperator::Return(ret) = &plan.root {
            if let LogicalOperator::NodeScan(scan) = ret.input.as_ref() {
                assert!(scan.label.is_none());
            } else {
                panic!("Expected NodeScan operator");
            }
        } else {
            panic!("Expected Return operator");
        }
    }

    #[test]
    fn test_translate_match_distinct() {
        let query = "MATCH (n:Person) RETURN DISTINCT n.name";
        let result = translate(query);
        assert!(result.is_ok());

        let plan = result.unwrap();
        if let LogicalOperator::Return(ret) = &plan.root {
            assert!(ret.distinct);
        } else {
            panic!("Expected Return operator");
        }
    }

    // === Filter and Predicate Tests ===

    #[test]
    fn test_translate_filter_equality() {
        let query = "MATCH (n:Person) WHERE n.name = 'Alix' RETURN n";
        let result = translate(query);
        assert!(result.is_ok());

        let plan = result.unwrap();
        // Navigate to find Filter
        fn find_filter(op: &LogicalOperator) -> Option<&FilterOp> {
            match op {
                LogicalOperator::Filter(f) => Some(f),
                LogicalOperator::Return(r) => find_filter(&r.input),
                _ => None,
            }
        }

        let filter = find_filter(&plan.root).expect("Expected Filter");
        if let LogicalExpression::Binary { op, .. } = &filter.predicate {
            assert_eq!(*op, BinaryOp::Eq);
        }
    }

    #[test]
    fn test_translate_filter_and() {
        let query = "MATCH (n:Person) WHERE n.age > 20 AND n.age < 40 RETURN n";
        let result = translate(query);
        assert!(result.is_ok());

        let plan = result.unwrap();
        fn find_filter(op: &LogicalOperator) -> Option<&FilterOp> {
            match op {
                LogicalOperator::Filter(f) => Some(f),
                LogicalOperator::Return(r) => find_filter(&r.input),
                _ => None,
            }
        }

        let filter = find_filter(&plan.root).expect("Expected Filter");
        if let LogicalExpression::Binary { op, .. } = &filter.predicate {
            assert_eq!(*op, BinaryOp::And);
        }
    }

    #[test]
    fn test_translate_filter_or() {
        let query = "MATCH (n:Person) WHERE n.name = 'Alix' OR n.name = 'Gus' RETURN n";
        let result = translate(query);
        assert!(result.is_ok());

        let plan = result.unwrap();
        fn find_filter(op: &LogicalOperator) -> Option<&FilterOp> {
            match op {
                LogicalOperator::Filter(f) => Some(f),
                LogicalOperator::Return(r) => find_filter(&r.input),
                _ => None,
            }
        }

        let filter = find_filter(&plan.root).expect("Expected Filter");
        if let LogicalExpression::Binary { op, .. } = &filter.predicate {
            assert_eq!(*op, BinaryOp::Or);
        }
    }

    #[test]
    fn test_translate_filter_not() {
        let query = "MATCH (n:Person) WHERE NOT n.active RETURN n";
        let result = translate(query);
        assert!(result.is_ok());

        let plan = result.unwrap();
        fn find_filter(op: &LogicalOperator) -> Option<&FilterOp> {
            match op {
                LogicalOperator::Filter(f) => Some(f),
                LogicalOperator::Return(r) => find_filter(&r.input),
                _ => None,
            }
        }

        let filter = find_filter(&plan.root).expect("Expected Filter");
        if let LogicalExpression::Unary { op, .. } = &filter.predicate {
            assert_eq!(*op, UnaryOp::Not);
        }
    }

    // === Path Pattern / Join Tests ===

    #[test]
    fn test_translate_path_pattern() {
        let query = "MATCH (a:Person)-[:KNOWS]->(b:Person) RETURN a, b";
        let result = translate(query);
        assert!(result.is_ok());

        let plan = result.unwrap();
        // Find Expand operator
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
    fn test_translate_incoming_path() {
        let query = "MATCH (a:Person)<-[:KNOWS]-(b:Person) RETURN a, b";
        let result = translate(query);
        assert!(result.is_ok());

        let plan = result.unwrap();
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
    fn test_translate_undirected_path() {
        let query = "MATCH (a:Person)-[:KNOWS]-(b:Person) RETURN a, b";
        let result = translate(query);
        assert!(result.is_ok());

        let plan = result.unwrap();
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
    }

    // === Aggregation Tests ===

    #[test]
    fn test_translate_count_aggregate() {
        let query = "MATCH (n:Person) RETURN COUNT(n)";
        let result = translate(query);
        assert!(result.is_ok());

        let plan = result.unwrap();
        if let LogicalOperator::Aggregate(agg) = &plan.root {
            assert_eq!(agg.aggregates.len(), 1);
            // COUNT(expr) uses CountNonNull to ensure we fetch values for DISTINCT support
            assert_eq!(agg.aggregates[0].function, AggregateFunction::CountNonNull);
        } else {
            panic!("Expected Aggregate operator, got {:?}", plan.root);
        }
    }

    #[test]
    fn test_translate_sum_aggregate() {
        let query = "MATCH (n:Person) RETURN SUM(n.age)";
        let result = translate(query);
        assert!(result.is_ok());

        let plan = result.unwrap();
        if let LogicalOperator::Aggregate(agg) = &plan.root {
            assert_eq!(agg.aggregates.len(), 1);
            assert_eq!(agg.aggregates[0].function, AggregateFunction::Sum);
        } else {
            panic!("Expected Aggregate operator");
        }
    }

    #[test]
    fn test_translate_group_by_aggregate() {
        let query = "MATCH (n:Person) RETURN n.city, COUNT(n)";
        let result = translate(query);
        assert!(result.is_ok());

        let plan = result.unwrap();
        if let LogicalOperator::Aggregate(agg) = &plan.root {
            assert_eq!(agg.group_by.len(), 1); // n.city
            assert_eq!(agg.aggregates.len(), 1); // COUNT(n)
        } else {
            panic!("Expected Aggregate operator");
        }
    }

    // === Ordering and Pagination Tests ===

    #[test]
    fn test_translate_order_by() {
        let query = "MATCH (n:Person) RETURN n ORDER BY n.name";
        let result = translate(query);
        assert!(result.is_ok());

        let plan = result.unwrap();
        // Sort wraps Return so that RETURN aliases are visible to ORDER BY
        if let LogicalOperator::Sort(sort) = &plan.root {
            assert_eq!(sort.keys.len(), 1);
            assert_eq!(sort.keys[0].order, SortOrder::Ascending);
            if let LogicalOperator::Return(_ret) = sort.input.as_ref() {
                // Return is the inner operator, as expected
            } else {
                panic!("Expected Return operator inside Sort");
            }
        } else {
            panic!("Expected Sort operator");
        }
    }

    #[test]
    fn test_translate_limit() {
        let query = "MATCH (n:Person) RETURN n LIMIT 10";
        let result = translate(query);
        assert!(result.is_ok());

        let plan = result.unwrap();
        // Find Limit
        fn find_limit(op: &LogicalOperator) -> Option<&LimitOp> {
            match op {
                LogicalOperator::Limit(l) => Some(l),
                LogicalOperator::Return(r) => find_limit(&r.input),
                LogicalOperator::Sort(s) => find_limit(&s.input),
                _ => None,
            }
        }

        let limit = find_limit(&plan.root).expect("Expected Limit");
        assert_eq!(limit.count, 10);
    }

    #[test]
    fn test_translate_skip() {
        let query = "MATCH (n:Person) RETURN n SKIP 5";
        let result = translate(query);
        assert!(result.is_ok());

        let plan = result.unwrap();
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

    // === Mutation Tests ===

    #[test]
    fn test_translate_insert_node() {
        let query = "INSERT (n:Person {name: 'Alix', age: 30})";
        let result = translate(query);
        assert!(result.is_ok());

        let plan = result.unwrap();
        // Find CreateNode
        fn find_create(op: &LogicalOperator) -> bool {
            match op {
                LogicalOperator::CreateNode(_) => true,
                LogicalOperator::Return(r) => find_create(&r.input),
                _ => false,
            }
        }

        assert!(find_create(&plan.root));
    }

    #[test]
    fn test_translate_delete() {
        let query = "DELETE n";
        let result = translate(query);
        assert!(result.is_ok());

        let plan = result.unwrap();
        if let LogicalOperator::DeleteNode(del) = &plan.root {
            assert_eq!(del.variable, "n");
        } else {
            panic!("Expected DeleteNode operator");
        }
    }

    #[test]
    fn test_translate_set() {
        // SET is not a standalone statement in GQL, test the translator method directly
        let translator = GqlTranslator::new();
        let set_stmt = ast::SetStatement {
            assignments: vec![ast::PropertyAssignment {
                variable: "n".to_string(),
                property: "name".to_string(),
                value: ast::Expression::Literal(ast::Literal::String("Gus".to_string())),
            }],
            span: None,
        };

        let result = translator.translate_set(&set_stmt);
        assert!(result.is_ok());

        let plan = result.unwrap();
        if let LogicalOperator::SetProperty(set) = &plan.root {
            assert_eq!(set.variable, "n");
            assert_eq!(set.properties.len(), 1);
            assert_eq!(set.properties[0].0, "name");
        } else {
            panic!("Expected SetProperty operator");
        }
    }

    // === Expression Translation Tests ===

    #[test]
    fn test_translate_literals() {
        let query = "MATCH (n) WHERE n.count = 42 AND n.active = true AND n.rate = 3.14 RETURN n";
        let result = translate(query);
        assert!(result.is_ok());
    }

    #[test]
    fn test_translate_parameter() {
        let query = "MATCH (n:Person) WHERE n.name = $name RETURN n";
        let result = translate(query);
        assert!(result.is_ok());

        let plan = result.unwrap();
        fn find_filter(op: &LogicalOperator) -> Option<&FilterOp> {
            match op {
                LogicalOperator::Filter(f) => Some(f),
                LogicalOperator::Return(r) => find_filter(&r.input),
                _ => None,
            }
        }

        let filter = find_filter(&plan.root).expect("Expected Filter");
        if let LogicalExpression::Binary { right, .. } = &filter.predicate {
            if let LogicalExpression::Parameter(name) = right.as_ref() {
                assert_eq!(name, "name");
            } else {
                panic!("Expected Parameter");
            }
        }
    }

    // === Error Handling Tests ===

    #[test]
    fn test_translate_empty_delete_error() {
        // Create translator directly to test empty delete
        let translator = GqlTranslator::new();
        let delete = ast::DeleteStatement {
            targets: vec![],
            detach: false,
            span: None,
        };
        let result = translator.translate_delete(&delete);
        assert!(result.is_err());
    }

    #[test]
    fn test_translate_empty_set_error() {
        let translator = GqlTranslator::new();
        let set = ast::SetStatement {
            assignments: vec![],
            span: None,
        };
        let result = translator.translate_set(&set);
        assert!(result.is_err());
    }

    #[test]
    fn test_translate_empty_insert_error() {
        let translator = GqlTranslator::new();
        let insert = ast::InsertStatement {
            patterns: vec![],
            span: None,
        };
        let result = translator.translate_insert(&insert);
        assert!(result.is_err());
    }

    // === Helper Function Tests ===

    #[test]
    fn test_is_aggregate_function() {
        assert!(is_aggregate_function("COUNT"));
        assert!(is_aggregate_function("count"));
        assert!(is_aggregate_function("SUM"));
        assert!(is_aggregate_function("AVG"));
        assert!(is_aggregate_function("MIN"));
        assert!(is_aggregate_function("MAX"));
        assert!(is_aggregate_function("COLLECT"));
        assert!(!is_aggregate_function("UPPER"));
        assert!(!is_aggregate_function("RANDOM"));
    }

    #[test]
    fn test_to_aggregate_function() {
        assert_eq!(
            to_aggregate_function("COUNT"),
            Some(AggregateFunction::Count)
        );
        assert_eq!(to_aggregate_function("sum"), Some(AggregateFunction::Sum));
        assert_eq!(to_aggregate_function("Avg"), Some(AggregateFunction::Avg));
        assert_eq!(to_aggregate_function("min"), Some(AggregateFunction::Min));
        assert_eq!(to_aggregate_function("MAX"), Some(AggregateFunction::Max));
        assert_eq!(
            to_aggregate_function("collect"),
            Some(AggregateFunction::Collect)
        );
        assert_eq!(to_aggregate_function("UNKNOWN"), None);
    }

    #[test]
    fn test_contains_aggregate() {
        let count_expr = ast::Expression::FunctionCall {
            name: "COUNT".to_string(),
            args: vec![],
            distinct: false,
        };
        assert!(contains_aggregate(&count_expr));

        let upper_expr = ast::Expression::FunctionCall {
            name: "UPPER".to_string(),
            args: vec![],
            distinct: false,
        };
        assert!(!contains_aggregate(&upper_expr));

        let var_expr = ast::Expression::Variable("n".to_string());
        assert!(!contains_aggregate(&var_expr));
    }

    #[test]
    fn test_binary_op_translation() {
        let translator = GqlTranslator::new();

        assert_eq!(
            translator.translate_binary_op(ast::BinaryOp::Eq),
            BinaryOp::Eq
        );
        assert_eq!(
            translator.translate_binary_op(ast::BinaryOp::Ne),
            BinaryOp::Ne
        );
        assert_eq!(
            translator.translate_binary_op(ast::BinaryOp::Lt),
            BinaryOp::Lt
        );
        assert_eq!(
            translator.translate_binary_op(ast::BinaryOp::Le),
            BinaryOp::Le
        );
        assert_eq!(
            translator.translate_binary_op(ast::BinaryOp::Gt),
            BinaryOp::Gt
        );
        assert_eq!(
            translator.translate_binary_op(ast::BinaryOp::Ge),
            BinaryOp::Ge
        );
        assert_eq!(
            translator.translate_binary_op(ast::BinaryOp::And),
            BinaryOp::And
        );
        assert_eq!(
            translator.translate_binary_op(ast::BinaryOp::Or),
            BinaryOp::Or
        );
        assert_eq!(
            translator.translate_binary_op(ast::BinaryOp::Add),
            BinaryOp::Add
        );
        assert_eq!(
            translator.translate_binary_op(ast::BinaryOp::Sub),
            BinaryOp::Sub
        );
        assert_eq!(
            translator.translate_binary_op(ast::BinaryOp::Mul),
            BinaryOp::Mul
        );
        assert_eq!(
            translator.translate_binary_op(ast::BinaryOp::Div),
            BinaryOp::Div
        );
        assert_eq!(
            translator.translate_binary_op(ast::BinaryOp::Mod),
            BinaryOp::Mod
        );
        assert_eq!(
            translator.translate_binary_op(ast::BinaryOp::Like),
            BinaryOp::Like
        );
        assert_eq!(
            translator.translate_binary_op(ast::BinaryOp::In),
            BinaryOp::In
        );
    }

    #[test]
    fn test_unary_op_translation() {
        let translator = GqlTranslator::new();

        assert_eq!(
            translator.translate_unary_op(ast::UnaryOp::Not),
            UnaryOp::Not
        );
        assert_eq!(
            translator.translate_unary_op(ast::UnaryOp::Neg),
            UnaryOp::Neg
        );
        assert_eq!(
            translator.translate_unary_op(ast::UnaryOp::IsNull),
            UnaryOp::IsNull
        );
        assert_eq!(
            translator.translate_unary_op(ast::UnaryOp::IsNotNull),
            UnaryOp::IsNotNull
        );
    }

    // === ShortestPath Tests ===

    #[test]
    fn test_translate_shortest_path() {
        let query = "MATCH p = shortestPath((a:Person)-[:KNOWS]->(b:Person)) RETURN p";
        let result = translate(query);
        assert!(
            result.is_ok(),
            "shortestPath should translate: {:?}",
            result.err()
        );
        let plan = result.unwrap();

        fn find_shortest_path(op: &LogicalOperator) -> bool {
            match op {
                LogicalOperator::ShortestPath(_) => true,
                LogicalOperator::Return(r) => find_shortest_path(&r.input),
                _ => false,
            }
        }
        assert!(
            find_shortest_path(&plan.root),
            "Plan should contain ShortestPath operator"
        );
    }

    #[test]
    fn test_translate_all_shortest_paths() {
        let query = "MATCH p = allShortestPaths((a)-[:ROAD]-(b)) RETURN p";
        let result = translate(query);
        assert!(
            result.is_ok(),
            "allShortestPaths should translate: {:?}",
            result.err()
        );
    }

    // === CASE expression ===

    #[test]
    fn test_translate_case_expression() {
        let query = "MATCH (n:Person) RETURN CASE WHEN n.age > 18 THEN 'adult' ELSE 'minor' END AS category";
        let result = translate(query);
        assert!(
            result.is_ok(),
            "CASE expression should translate: {:?}",
            result.err()
        );
    }

    // === UNWIND ===

    #[test]
    fn test_translate_unwind() {
        let query = "UNWIND [1, 2, 3] AS x RETURN x";
        let result = translate(query);
        assert!(
            result.is_ok(),
            "UNWIND should translate: {:?}",
            result.err()
        );
        let plan = result.unwrap();

        fn find_unwind(op: &LogicalOperator) -> bool {
            match op {
                LogicalOperator::Unwind(_) => true,
                LogicalOperator::Return(r) => find_unwind(&r.input),
                _ => false,
            }
        }
        assert!(find_unwind(&plan.root), "Plan should contain Unwind");
    }

    // === MERGE ===

    #[test]
    fn test_translate_merge() {
        let query = "MERGE (n:Person {name: 'Alix'}) RETURN n";
        let result = translate(query);
        assert!(result.is_ok(), "MERGE should translate: {:?}", result.err());
        let plan = result.unwrap();

        fn find_merge(op: &LogicalOperator) -> bool {
            match op {
                LogicalOperator::Merge(_) => true,
                LogicalOperator::Return(r) => find_merge(&r.input),
                _ => false,
            }
        }
        assert!(find_merge(&plan.root), "Plan should contain Merge");
    }

    #[test]
    fn test_translate_merge_with_on_create() {
        let query = "MERGE (n:Person {name: 'Alix'}) ON CREATE SET n.created = true RETURN n";
        let result = translate(query);
        assert!(
            result.is_ok(),
            "MERGE ON CREATE should translate: {:?}",
            result.err()
        );
    }

    // === WITH clause ===

    #[test]
    fn test_translate_with_clause() {
        let query = "MATCH (n:Person) WITH n.name AS name WHERE name = 'Alix' RETURN name";
        let result = translate(query);
        assert!(
            result.is_ok(),
            "WITH clause should translate: {:?}",
            result.err()
        );
    }

    // === Label operations ===

    #[test]
    fn test_translate_add_label() {
        let query = "MATCH (n:Person) SET n:Employee RETURN n";
        let result = translate(query);
        assert!(
            result.is_ok(),
            "SET label should translate: {:?}",
            result.err()
        );
    }

    #[test]
    fn test_translate_remove_label() {
        let query = "MATCH (n:Person) REMOVE n:Employee RETURN n";
        let result = translate(query);
        assert!(
            result.is_ok(),
            "REMOVE label should translate: {:?}",
            result.err()
        );
    }

    // === Multiple aggregates ===

    #[test]
    fn test_translate_multiple_aggregates() {
        let query = "MATCH (n:Person) RETURN count(n) AS cnt, sum(n.age) AS total_age, avg(n.age) AS avg_age";
        let result = translate(query);
        assert!(
            result.is_ok(),
            "Multiple aggregates should translate: {:?}",
            result.err()
        );
    }

    #[test]
    fn test_translate_group_by_with_having_like_filter() {
        // Use WHERE after aggregation (emulating HAVING)
        let query = "MATCH (n:Person) RETURN n.city AS city, count(n) AS cnt ORDER BY cnt DESC";
        let result = translate(query);
        assert!(
            result.is_ok(),
            "GROUP BY with ORDER BY should translate: {:?}",
            result.err()
        );
    }

    // === GqlTranslationResult enum Tests ===

    #[test]
    fn test_translate_full_returns_plan_for_query() {
        let query = "MATCH (n:Person) RETURN n";
        let result = translate_full(query);
        assert!(
            result.is_ok(),
            "translate_full should succeed: {:?}",
            result.err()
        );
        assert!(
            matches!(result.unwrap(), GqlTranslationResult::Plan(_)),
            "translate_full should return Plan for a query"
        );
    }

    #[test]
    fn test_translate_full_returns_session_command() {
        let query = "COMMIT";
        let result = translate_full(query);
        assert!(
            result.is_ok(),
            "translate_full should succeed for COMMIT: {:?}",
            result.err()
        );
        assert!(
            matches!(result.unwrap(), GqlTranslationResult::SessionCommand(_)),
            "translate_full should return SessionCommand for COMMIT"
        );
    }

    // === translate() vs session commands ===

    #[test]
    fn test_translate_returns_ok_for_query() {
        let query = "MATCH (n) RETURN n";
        let result = translate(query);
        assert!(result.is_ok(), "translate should succeed for a query");
    }

    #[test]
    fn test_translate_returns_err_for_session_command() {
        let query = "COMMIT";
        let result = translate(query);
        assert!(
            result.is_err(),
            "translate should return Err for session commands"
        );
    }

    // === Set Operations ===

    #[test]
    fn test_translate_except() {
        let query = "MATCH (a:Person) RETURN a EXCEPT MATCH (b:Employee) RETURN b";
        let result = translate(query);
        assert!(
            result.is_ok(),
            "EXCEPT should translate: {:?}",
            result.err()
        );

        let plan = result.unwrap();
        assert!(
            matches!(plan.root, LogicalOperator::Except(_)),
            "Expected Except operator, got {:?}",
            plan.root
        );
    }

    #[test]
    fn test_translate_intersect() {
        let query = "MATCH (a:Person) RETURN a INTERSECT MATCH (b:Employee) RETURN b";
        let result = translate(query);
        assert!(
            result.is_ok(),
            "INTERSECT should translate: {:?}",
            result.err()
        );

        let plan = result.unwrap();
        assert!(
            matches!(plan.root, LogicalOperator::Intersect(_)),
            "Expected Intersect operator, got {:?}",
            plan.root
        );
    }

    #[test]
    fn test_translate_otherwise() {
        let query = "MATCH (a:Person) RETURN a OTHERWISE MATCH (b:Employee) RETURN b";
        let result = translate(query);
        assert!(
            result.is_ok(),
            "OTHERWISE should translate: {:?}",
            result.err()
        );

        let plan = result.unwrap();
        assert!(
            matches!(plan.root, LogicalOperator::Otherwise(_)),
            "Expected Otherwise operator, got {:?}",
            plan.root
        );
    }

    // === FINISH ===

    #[test]
    fn test_translate_finish() {
        let query = "MATCH (n:Person) FINISH";
        let result = translate(query);
        assert!(
            result.is_ok(),
            "FINISH should translate: {:?}",
            result.err()
        );

        let plan = result.unwrap();
        // FINISH is translated as Limit(0)
        if let LogicalOperator::Limit(limit) = &plan.root {
            assert_eq!(limit.count, 0, "FINISH should produce Limit(0)");
        } else {
            panic!("Expected Limit operator for FINISH, got {:?}", plan.root);
        }
    }

    // === Element WHERE on nodes ===

    #[test]
    fn test_translate_element_where_on_node() {
        let query = "MATCH (n:Person WHERE n.age > 30) RETURN n";
        let result = translate(query);
        assert!(
            result.is_ok(),
            "Element WHERE should translate: {:?}",
            result.err()
        );

        let plan = result.unwrap();
        // Walk the plan to find a Filter with a > predicate
        fn find_gt_filter(op: &LogicalOperator) -> bool {
            match op {
                LogicalOperator::Filter(f) => {
                    if let LogicalExpression::Binary { op, .. } = &f.predicate {
                        *op == BinaryOp::Gt || find_gt_filter(&f.input)
                    } else {
                        find_gt_filter(&f.input)
                    }
                }
                LogicalOperator::Return(r) => find_gt_filter(&r.input),
                _ => false,
            }
        }
        assert!(
            find_gt_filter(&plan.root),
            "Expected a Filter with Gt predicate from element WHERE clause"
        );
    }

    // === NULLIF desugaring ===

    #[test]
    fn test_translate_nullif_desugaring() {
        let query = "MATCH (n:Person) RETURN nullif(n.age, 0) AS age";
        let result = translate(query);
        assert!(
            result.is_ok(),
            "NULLIF should translate: {:?}",
            result.err()
        );

        let plan = result.unwrap();
        // NULLIF(x, y) desugars to CASE WHEN x = y THEN NULL ELSE x END.
        // The Return operator should contain a Case expression.
        fn find_case_in_return(op: &LogicalOperator) -> bool {
            if let LogicalOperator::Return(ret) = op {
                ret.items
                    .iter()
                    .any(|item| matches!(item.expression, LogicalExpression::Case { .. }))
            } else {
                false
            }
        }
        assert!(
            find_case_in_return(&plan.root),
            "Expected NULLIF to desugar into a CASE expression in RETURN"
        );
    }
}
