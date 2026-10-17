//! GQL to LogicalPlan translator.
//!
//! Translates GQL AST to the common logical plan representation.

mod aggregate;
mod expression;
mod pattern;

use std::collections::{HashMap, HashSet};

use super::common::{
    CallImports, GeneratedNames, build_left_join_with_predicates, call_imports,
    check_branch_columns, collect_expression_variables, combine_with_and,
    comma_part_join_variables, comma_part_reads_earlier_rows, expand_subquery_return_star,
    flatten_and_conjuncts, has_all_labels, is_aggregate_function, is_binary_set_function,
    join_and_conjuncts, no_result, optional_join, push_set_property, references_any,
    to_aggregate_function, with_imports, wrap_distinct, wrap_filter, wrap_limit, wrap_return,
    wrap_skip, wrap_sort,
};
use crate::query::plan::{
    self as plan, AddLabelOp, AggregateExpr, AggregateFunction, AggregateOp, ApplyOp, BinaryOp,
    CallProcedureOp, CountExpr, CreateElement, CreateOp, DeleteNodeOp, EntityKind, ExceptOp,
    ExpandDirection, ExpandOp, HorizontalAggregateOp, IntersectOp, JoinCondition, JoinOp, JoinType,
    LeftJoinOp, LoadDataFormat, LoadDataOp, LogicalExpression, LogicalOperator, LogicalPlan,
    MergeOp, MergeRelationshipOp, NodeScanOp, NullsOrdering, OtherwiseOp, ParameterScanOp,
    PathMode, ProcedureYield, ProjectOp, Projection, RemoveLabelOp, ReturnItem, ReturnOp,
    SetPropertyOp, ShortestPathEdgeCondition, ShortestPathOp, SortKey, SortOrder, UnaryOp, UnionOp,
    UnwindOp,
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
    let translator = GqlTranslator::new(query);
    match translator.translate_statement_full(&statement)? {
        GqlTranslationResult::Plan(plan) => Ok(GqlTranslationResult::Plan(
            crate::query::limits::check_plan_depth(plan)?,
        )),
        other => Ok(other),
    }
}

/// Translator from GQL AST to LogicalPlan.
struct GqlTranslator {
    /// Edge variables from variable-length expand patterns (group-list variables),
    /// each bound to the list of its path's edges, with the path alias of the
    /// pattern. An aggregate over one is computed per row
    /// (see `take_horizontal_aggregates`).
    group_list_variables: std::cell::RefCell<HashMap<String, String>>,
    /// The variables of the row the `CALL` subquery being translated runs
    /// for (`None` outside one, or when they are not known): what a nested
    /// subquery's `RETURN *` leaves out.
    call_scope: std::cell::RefCell<Option<HashSet<String>>>,
    /// The names made up for anonymous elements and helper columns.
    names: GeneratedNames,
    /// The variables the `CALL` body being translated imports, which a
    /// `WITH` in it passes on whether it names them or not.
    call_imports: CallImports,
}

/// The rows a query passes to the one after `NEXT`: its final `RETURN` as a
/// `WITH` (a projection), under its `ORDER BY`, `SKIP` and `LIMIT`. An
/// unaliased item is a column named after it; `RETURN *` passes every column
/// on. Without `DISTINCT` the rows are ordered before the projection, so an
/// `ORDER BY` key can read what the `RETURN` leaves out (and reads an alias
/// through its expression). A plan that ends otherwise (an aggregation)
/// passes its rows on as they are.
fn return_as_with(plan: LogicalOperator) -> LogicalOperator {
    match plan {
        LogicalOperator::Return(ret) => {
            if ret.items.iter().any(
                |item| matches!(&item.expression, LogicalExpression::Variable(name) if name == "*"),
            ) {
                return if ret.distinct {
                    wrap_distinct(*ret.input)
                } else {
                    *ret.input
                };
            }
            let projections = ret
                .items
                .into_iter()
                .map(|item| {
                    let alias = match (&item.alias, &item.expression) {
                        (Some(alias), _) => Some(alias.clone()),
                        (None, LogicalExpression::Variable(_)) => None,
                        (None, expression) => Some(
                            crate::query::planner::common::expression_to_string(expression),
                        ),
                    };
                    Projection {
                        expression: item.expression,
                        alias,
                    }
                })
                .collect();
            let project = LogicalOperator::Project(ProjectOp {
                projections,
                input: ret.input,
                pass_through_input: false,
            });
            if ret.distinct {
                wrap_distinct(project)
            } else {
                project
            }
        }
        LogicalOperator::Sort(mut sort) => match *sort.input {
            LogicalOperator::Return(ret) if !ret.distinct => {
                match keys_before_return(&sort.keys, &ret) {
                    Some(keys) => {
                        sort.keys = keys;
                        sort.input = ret.input;
                        return_as_with(LogicalOperator::Return(ReturnOp {
                            input: Box::new(LogicalOperator::Sort(sort)),
                            ..ret
                        }))
                    }
                    // A key reads an alias that cannot be replaced: order the
                    // projected rows (the key then reads only what they hold).
                    None => {
                        sort.input = Box::new(return_as_with(LogicalOperator::Return(ret)));
                        LogicalOperator::Sort(sort)
                    }
                }
            }
            input => {
                sort.input = Box::new(return_as_with(input));
                LogicalOperator::Sort(sort)
            }
        },
        LogicalOperator::Skip(mut skip) => {
            skip.input = Box::new(return_as_with(*skip.input));
            LogicalOperator::Skip(skip)
        }
        LogicalOperator::Limit(mut limit) => {
            limit.input = Box::new(return_as_with(*limit.input));
            LogicalOperator::Limit(limit)
        }
        LogicalOperator::Distinct(mut distinct) => {
            distinct.input = Box::new(return_as_with(*distinct.input));
            LogicalOperator::Distinct(distinct)
        }
        other => other,
    }
}

/// The sort `keys` of a `RETURN` rewritten to read the `RETURN`'s input: each
/// alias replaced by its expression (a property of an alias that is a
/// variable by that variable's property). `None` when a key still reads an
/// alias, such as one inside a `CASE`, or a property of an alias that is not
/// a variable.
fn keys_before_return(keys: &[SortKey], ret: &ReturnOp) -> Option<Vec<SortKey>> {
    let aliases: Vec<(String, LogicalExpression)> = ret
        .items
        .iter()
        .filter_map(|item| {
            item.alias
                .as_ref()
                .map(|alias| (alias.clone(), item.expression.clone()))
        })
        .collect();
    let input_names = ret.input.bound_variables(None);
    keys.iter()
        .map(|key| {
            let expression =
                GqlTranslator::substitute_let_bindings(key.expression.clone(), &aliases);
            let mut read = HashSet::new();
            collect_expression_variables(&expression, &mut read);
            let reads_an_alias = read.iter().any(|name| {
                aliases.iter().any(|(alias, _)| alias == name)
                    && input_names
                        .as_ref()
                        .is_none_or(|names| !names.contains(name))
            });
            (!reads_an_alias).then(|| SortKey {
                expression,
                ..key.clone()
            })
        })
        .collect()
}

/// The sort key of an `ORDER BY` item that sorts on `expression`.
///
/// Without `NULLS FIRST` or `NULLS LAST`, GQL puts nulls last in both
/// directions. ISO/IEC 39075 leaves this default to the implementation, as SQL
/// does; nulls last both ways is what Microsoft Fabric's GQL documents. Cypher
/// keeps openCypher's order, where null is the largest value (see
/// `physical_null_order`).
fn sort_key(item: &ast::OrderByItem, expression: LogicalExpression) -> SortKey {
    SortKey {
        expression,
        order: match item.order {
            ast::SortOrder::Asc => SortOrder::Ascending,
            ast::SortOrder::Desc => SortOrder::Descending,
        },
        nulls: Some(match item.nulls {
            Some(ast::NullsOrdering::First) => NullsOrdering::First,
            Some(ast::NullsOrdering::Last) | None => NullsOrdering::Last,
        }),
    }
}

/// Whether a statement's plan ends without a result: in a `RETURN` of no
/// items (see `no_result`), as a write without `RETURN` and `FINISH` do.
fn ends_without_a_result(plan: &LogicalOperator) -> bool {
    matches!(plan, LogicalOperator::Return(ret) if ret.items.is_empty())
}

/// Whether a query ends with a result: a `RETURN` with items or `RETURN *`.
/// A `CALL` body without one (no `RETURN`, or `FINISH`) runs for its writes
/// and passes each row on once, as it came in.
fn returns_rows(query: &ast::QueryStatement) -> bool {
    let result = &query.return_clause;
    !result.is_finish && (result.is_wildcard || !result.items.is_empty())
}

/// Combines two queries with a set operator, or with `NEXT` (the right one
/// runs for each row of the left one; a right side that is a query or a lone
/// write reads the left one's rows instead, see `translate_composite_query`).
fn combine_queries(
    op: ast::CompositeOp,
    left: LogicalOperator,
    right: LogicalOperator,
) -> Result<LogicalOperator> {
    Ok(match op {
        ast::CompositeOp::Union | ast::CompositeOp::UnionAll => {
            let inputs = vec![left, right];
            check_branch_columns("UNION", &inputs)?;
            let union_op = LogicalOperator::Union(UnionOp { inputs });
            if op == ast::CompositeOp::UnionAll {
                union_op
            } else {
                wrap_distinct(union_op)
            }
        }
        ast::CompositeOp::Except | ast::CompositeOp::ExceptAll => {
            let branches = [left, right];
            check_branch_columns("EXCEPT", &branches)?;
            let [left, right] = branches;
            LogicalOperator::Except(ExceptOp {
                left: Box::new(left),
                right: Box::new(right),
                all: matches!(op, ast::CompositeOp::ExceptAll),
            })
        }
        ast::CompositeOp::Intersect | ast::CompositeOp::IntersectAll => {
            let branches = [left, right];
            check_branch_columns("INTERSECT", &branches)?;
            let [left, right] = branches;
            LogicalOperator::Intersect(IntersectOp {
                left: Box::new(left),
                right: Box::new(right),
                all: matches!(op, ast::CompositeOp::IntersectAll),
            })
        }
        ast::CompositeOp::Otherwise => {
            let branches = [left, right];
            check_branch_columns("OTHERWISE", &branches)?;
            let [left, right] = branches;
            LogicalOperator::Otherwise(OtherwiseOp {
                left: Box::new(left),
                right: Box::new(right),
            })
        }
        // NEXT (linear composition): output of left feeds as input to right.
        // Translate as Apply: for each row from left, execute right with bound variables.
        ast::CompositeOp::Next => LogicalOperator::Apply(ApplyOp {
            input: Box::new(left),
            subplan: Box::new(right),
            shared_variables: Vec::new(),
            optional: false,
            unit: false,
        }),
    })
}

impl GqlTranslator {
    /// A translator for the statement `query`, whose text the generated
    /// names skip.
    fn new(query: &str) -> Self {
        Self {
            group_list_variables: std::cell::RefCell::new(HashMap::new()),
            call_scope: std::cell::RefCell::new(None),
            names: GeneratedNames::new(query),
            call_imports: CallImports::default(),
        }
    }

    /// A name for an anonymous element, one the statement does not spell.
    fn anonymous_name(&self) -> String {
        self.names.next("_anon_")
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
            ast::Statement::DataModification(dm) => {
                self.translate_data_modification(dm, LogicalOperator::Empty)
            }
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
        // NEXT: the statement after it reads the rows the one before returns,
        // as a query reads the rows of a WITH; a lone INSERT or DELETE writes
        // for each, and a procedure call runs for each. After one without a
        // result (no RETURN, or FINISH) it reads one empty row, as at the
        // start of a statement: the one before runs once, for its writes,
        // and nothing it binds is a variable after it.
        if op == ast::CompositeOp::Next {
            let input = if ends_without_a_result(&left_plan.root) {
                LogicalOperator::Apply(ApplyOp {
                    input: Box::new(LogicalOperator::Empty),
                    subplan: Box::new(left_plan.root),
                    shared_variables: Vec::new(),
                    optional: false,
                    unit: true,
                })
            } else {
                return_as_with(left_plan.root)
            };
            return match right {
                ast::Statement::Query(right_query) => self.translate_query_from(right_query, input),
                ast::Statement::DataModification(dm) => self.translate_data_modification(dm, input),
                other => {
                    let right_plan = self.translate_statement(other)?;
                    Ok(LogicalPlan::new(combine_queries(
                        op,
                        input,
                        right_plan.root,
                    )?))
                }
            };
        }
        let right_plan = self.translate_statement(right)?;
        Ok(LogicalPlan::new(combine_queries(
            op,
            left_plan.root,
            right_plan.root,
        )?))
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

            // GROUP BY makes one row per group, also without an aggregate
            let groups_rows = !return_clause.items.is_empty() && !return_clause.group_by.is_empty();
            if has_aggregates || groups_rows {
                let (aggregates, auto_group_by, mut post_return) = self
                    .extract_aggregates_and_groups(
                        &return_clause.items,
                        !return_clause.group_by.is_empty(),
                    )?;
                // Explicit GROUP BY wins over the keys implied by the items.
                let group_by = if return_clause.group_by.is_empty() {
                    auto_group_by
                } else {
                    let keys = self.translate_group_by(
                        &return_clause.group_by,
                        &return_clause.items,
                        &plan,
                    )?;
                    self.resolve_grouped_items(
                        &return_clause.items,
                        post_return.as_deref_mut().unwrap_or_default(),
                        &keys,
                        &aggregates,
                    )?;
                    keys
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
                            Ok(sort_key(item, expression))
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
                            Ok(sort_key(item, self.translate_expression(&item.expression)?))
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
        self.translate_query_from(query, LogicalOperator::Empty)
    }

    /// Translates `query` on the rows of `input`: `Empty` for a query of its
    /// own, the outer row's variables for a `CALL` subquery.
    fn translate_query_from(
        &self,
        query: &ast::QueryStatement,
        input: LogicalOperator,
    ) -> Result<LogicalPlan> {
        let mut plan = input;

        // Each clause reads the rows of the ones before it, in source order
        // (ISO GQL's linear statement: a WHERE or FILTER, an ORDER BY or a
        // LIMIT applies where it stands). A query without ordered clauses (a
        // subquery, SELECT ... FROM) has its parts in the per-kind fields.
        if !query.ordered_clauses.is_empty() {
            // Whether the clause before is an OPTIONAL MATCH, whose graph
            // pattern a WHERE right after it belongs to
            let mut after_optional_match = false;
            for clause in &query.ordered_clauses {
                // A clause may walk the plan below it recursively (the
                // variables it binds): stop before that is too deep.
                crate::query::limits::check_partial_plan_depth(&mut plan)?;
                match clause {
                    ast::QueryClause::Match(match_clause) => {
                        if matches!(plan, LogicalOperator::Empty) && match_clause.optional {
                            // OPTIONAL MATCH as the first clause: left join with
                            // an implicit unit table so unmatched patterns produce
                            // a single row of NULLs instead of zero rows.
                            let match_plan = self.translate_match(match_clause)?;
                            plan = optional_join(
                                LogicalOperator::Empty,
                                match_plan,
                                self.call_scope.borrow().as_ref(),
                            );
                        } else if matches!(plan, LogicalOperator::Empty) {
                            // No prior input: standard MATCH
                            plan = self.translate_match(match_clause)?;
                        } else if match_clause.optional {
                            // OPTIONAL MATCH: left join (prior vars on left, match on right)
                            let match_plan = self.translate_match(match_clause)?;
                            plan =
                                optional_join(plan, match_plan, self.call_scope.borrow().as_ref());
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
                        plan = self.translate_insert(&create_clause.patterns, plan)?;
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
                            plan = push_set_property(
                                plan,
                                &assignment.variable,
                                assignment.property.clone(),
                                value,
                                false,
                            );
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
                    ast::QueryClause::Remove(remove_clause) => {
                        plan = Self::apply_remove(plan, remove_clause);
                    }
                    ast::QueryClause::With(with_clause) => {
                        plan = self.apply_with(plan, with_clause)?;
                    }
                    ast::QueryClause::InlineCall {
                        subquery,
                        combined,
                        optional,
                        scope,
                    } => {
                        plan = self.translate_inline_call(
                            subquery,
                            combined,
                            plan,
                            *optional,
                            scope.as_deref(),
                        )?;
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
                                unit: false,
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
                    // A WHERE right after an OPTIONAL MATCH is part of it and
                    // keeps the rows without a match; a FILTER, and a WHERE
                    // anywhere else, filters the rows so far.
                    ast::QueryClause::Filter(filter) => {
                        plan = self.apply_statement_where(plan, filter, after_optional_match)?;
                    }
                    ast::QueryClause::OrderByAndPage(page) => {
                        plan = self.apply_order_by_and_page(plan, page)?;
                    }
                }
                after_optional_match =
                    matches!(clause, ast::QueryClause::Match(found) if found.optional);
            }
        } else {
            // Legacy path: process MATCH, then UNWIND, then MERGE separately
            for match_clause in &query.match_clauses {
                let match_plan = self.translate_match(match_clause)?;
                if matches!(plan, LogicalOperator::Empty) && match_clause.optional {
                    plan = optional_join(
                        LogicalOperator::Empty,
                        match_plan,
                        self.call_scope.borrow().as_ref(),
                    );
                } else if matches!(plan, LogicalOperator::Empty) {
                    plan = match_plan;
                } else if match_clause.optional {
                    plan = optional_join(plan, match_plan, self.call_scope.borrow().as_ref());
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

            if let Some(where_clause) = &query.where_clause {
                let after_optional_match = query.unwind_clauses.is_empty()
                    && query.merge_clauses.is_empty()
                    && query.match_clauses.last().is_some_and(|last| last.optional);
                plan = self.apply_statement_where(plan, where_clause, after_optional_match)?;
            }
        }

        // Legacy path: handle SET/REMOVE/CREATE/DELETE from individual fields.
        // When ordered_clauses is used, these are already processed above.
        if query.ordered_clauses.is_empty() {
            for set_clause in &query.set_clauses {
                for assignment in &set_clause.assignments {
                    let value = self.translate_expression(&assignment.value)?;
                    plan = push_set_property(
                        plan,
                        &assignment.variable,
                        assignment.property.clone(),
                        value,
                        false,
                    );
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

        // REMOVE clauses not among the ordered clauses (statements built
        // without them) apply here, after the rest.
        if !query
            .ordered_clauses
            .iter()
            .any(|clause| matches!(clause, ast::QueryClause::Remove(_)))
        {
            for remove_clause in &query.remove_clauses {
                plan = Self::apply_remove(plan, remove_clause);
            }
        }

        // WITH clauses not among the ordered clauses (statements built
        // without them) apply here, after the rest.
        if !query
            .ordered_clauses
            .iter()
            .any(|clause| matches!(clause, ast::QueryClause::With(_)))
        {
            for with_clause in &query.with_clauses {
                plan = self.apply_with(plan, with_clause)?;
            }
        }

        // FINISH: the statement has no result (ISO GQL's omitted result: no
        // rows and no columns); the input runs to its end for its writes. A
        // NEXT after it reads one empty row, as after a write without RETURN
        // (see `translate_composite_query`).
        if query.return_clause.is_finish {
            return Ok(LogicalPlan::new(no_result(wrap_limit(plan, 0))));
        }

        // Aggregates over a group variable are computed per row first: the
        // items read their columns (see `take_horizontal_aggregates`)
        let aggregate::WithoutHorizontalAggregates {
            items: return_items,
            order_by: return_order_by,
            having: having_condition,
            plan: horizontal_plan,
        } = self.take_horizontal_aggregates(
            &query.return_clause.items,
            query.return_clause.order_by.as_ref(),
            query
                .having_clause
                .as_ref()
                .map(|having| &having.expression),
            plan,
        )?;
        plan = horizontal_plan;

        // Check if RETURN contains aggregate functions
        let has_aggregates = !query.return_clause.is_wildcard
            && return_items
                .iter()
                .any(|item| contains_aggregate(&item.expression));

        // GROUP BY makes one row per group (ISO/IEC 39075:2024 <group by
        // clause>), and so does HAVING, also when the RETURN list has no
        // aggregate.
        let groups_rows = !query.return_clause.is_wildcard
            && (!query.return_clause.group_by.is_empty() || query.having_clause.is_some());
        // An aggregate in HAVING is computed per group like the ones of the
        // RETURN list, in a column of its own that the result leaves out.
        let having_aggregates = having_condition.as_ref().is_some_and(contains_aggregate);

        if has_aggregates || groups_rows {
            // Extract aggregate and group-by expressions.
            // When a return item wraps an aggregate in a binary/unary expression
            // (e.g. `count(n) > 0 AS exists`), we decompose it into:
            //   1. An aggregate (`count(n)` with synthetic alias)
            //   2. A post-aggregate projection (`_agg_0 > 0 AS exists`)
            // The post-Return also leaves out the columns of HAVING's
            // aggregates.
            let (mut aggregates, auto_group_by, mut post_return) = self
                .extract_aggregates_and_groups(
                    &return_items,
                    !query.return_clause.group_by.is_empty() || having_aggregates,
                )?;
            // HAVING's aggregates are numbered after the `_agg_N` ones of the
            // RETURN list.
            let mut having_counter = u32::try_from(
                aggregates
                    .iter()
                    .filter(|aggregate| {
                        aggregate
                            .alias
                            .as_deref()
                            .is_some_and(|alias| alias.starts_with("_agg_"))
                    })
                    .count(),
            )
            .map_err(|_| {
                Error::Query(QueryError::new(
                    QueryErrorKind::Semantic,
                    "too many aggregates in one RETURN",
                ))
            })?;

            // Use explicit GROUP BY if provided, otherwise use auto-detected
            let group_by = if query.return_clause.group_by.is_empty() {
                auto_group_by
            } else {
                let keys =
                    self.translate_group_by(&query.return_clause.group_by, &return_items, &plan)?;
                self.resolve_grouped_items(
                    &return_items,
                    post_return.as_deref_mut().unwrap_or_default(),
                    &keys,
                    &aggregates,
                )?;
                keys
            };

            // Translate HAVING clause if present; its aggregates become
            // columns of the Aggregate (see `having_aggregates`), and its
            // grouping keys and RETURN aliases read the Aggregate's output.
            let having_scope = aggregate::HavingScope::new(&group_by, post_return.as_deref());
            let having = match &having_condition {
                Some(condition) if having_aggregates => {
                    Some(having_scope.resolve(self.extract_wrapped_aggregates(
                        condition,
                        &mut having_counter,
                        &mut aggregates,
                    )?))
                }
                Some(condition) => {
                    Some(having_scope.resolve(self.translate_expression(condition)?))
                }
                None => None,
            };

            let agg_op = LogicalOperator::Aggregate(AggregateOp {
                group_by,
                aggregates,
                input: Box::new(plan),
                having,
            });

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
            if let Some(order_by) = &return_order_by {
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
                        Ok(sort_key(item, expression))
                    })
                    .collect::<Result<Vec<_>>>()?;

                plan = wrap_sort(plan, keys);
            }

            // Note: For aggregate queries, we don't add a Return operator
            // because Aggregate already produces the final output
        } else {
            // Apply RETURN first (closest to input), then Sort wraps it.
            // This ensures RETURN aliases are visible to ORDER BY in the binder.
            let return_items = if query.return_clause.is_wildcard {
                // RETURN *: emit a wildcard marker that the planner expands
                vec![ReturnItem {
                    expression: LogicalExpression::Variable("*".into()),
                    alias: None,
                }]
            } else {
                return_items
                    .iter()
                    .map(|item| {
                        Ok(ReturnItem {
                            expression: self.translate_expression(&item.expression)?,
                            alias: item.alias.clone(),
                        })
                    })
                    .collect::<Result<Vec<_>>>()?
            };

            plan = wrap_return(plan, return_items, query.return_clause.distinct);

            // Apply ORDER BY (wraps Return so aliases are visible)
            if let Some(order_by) = &return_order_by {
                let keys = order_by
                    .items
                    .iter()
                    .map(|item| Ok(sort_key(item, self.translate_expression(&item.expression)?)))
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
            // A property of a binding that is a variable reads that
            // variable's property.
            LogicalExpression::Property {
                ref variable,
                ref property,
            } => match bindings.iter().find(|(name, _)| name == variable) {
                Some((_, LogicalExpression::Variable(target))) => LogicalExpression::Property {
                    variable: target.clone(),
                    property: property.clone(),
                },
                _ => expr,
            },
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

    /// The variable of the node a pattern starts from, if it names one.
    fn pattern_start(pattern: &ast::Pattern) -> Option<&str> {
        match pattern {
            ast::Pattern::Node(node) => node.variable.as_deref(),
            ast::Pattern::Path(path) => path.source.variable.as_deref(),
            ast::Pattern::Quantified { .. }
            | ast::Pattern::Union(_)
            | ast::Pattern::MultisetUnion(_) => None,
        }
    }

    /// Translates a MATCH clause with an optional initial input.
    ///
    /// When `initial_input` is provided (e.g. from a preceding UNWIND), the
    /// first pattern's NodeScan receives it as input. This creates a nested
    /// loop join that keeps prior variables (like UNWIND variables) in scope
    /// so that property filters like `{id: x}` can reference them.
    ///
    /// When multiple comma-separated patterns share variables, creates proper
    /// `JoinOp` operators with equality conditions instead of cross products;
    /// the variables of the input count as shared too (see
    /// [`comma_part_join_variables`]).
    fn translate_match_with_input(
        &self,
        match_clause: &ast::MatchClause,
        initial_input: Option<LogicalOperator>,
    ) -> Result<LogicalOperator> {
        // The match mode and the path mode are independent (ISO/IEC
        // 39075:2024 16.4 and 16.6): REPEATABLE ELEMENTS, the default, keeps
        // the path mode. DIFFERENT EDGES binds no edge twice in the whole
        // graph pattern, which a filter over every edge checks below (each
        // edge pattern then needs a variable); a path that repeats an edge
        // can never match, so its expands and searches may stop at a
        // repeated edge.
        let different_edges = match_clause.match_mode == Some(ast::MatchMode::DifferentEdges);
        // The path mode of a pattern: its own, else the clause's
        let path_mode_of = |aliased: &ast::AliasedPattern| {
            let mode = match aliased.path_mode.or(match_clause.path_mode) {
                Some(ast::PathMode::Walk) | None => PathMode::Walk,
                Some(ast::PathMode::Trail) => PathMode::Trail,
                Some(ast::PathMode::Simple) => PathMode::Simple,
                Some(ast::PathMode::Acyclic) => PathMode::Acyclic,
            };
            if different_edges && mode == PathMode::Walk {
                PathMode::Trail
            } else {
                mode
            }
        };
        let keeps_different_edges =
            |pattern: &ast::AliasedPattern| pattern.keep == Some(ast::MatchMode::DifferentEdges);
        let named_edges;
        let match_clause =
            if different_edges || match_clause.patterns.iter().any(keeps_different_edges) {
                named_edges = pattern::with_named_edges(match_clause, &self.names);
                &named_edges
            } else {
                match_clause
            };

        // The selection of a selective pattern (ISO/IEC 39075:2024 16.6): its
        // own search prefix, else the clause's; `ALL` selects nothing. A
        // lone node pattern has one path in each partition, itself.
        let selection_of = |aliased: &ast::AliasedPattern| {
            if let Some(path_function) = aliased.path_function {
                return Some(match path_function {
                    ast::PathFunction::ShortestPath => plan::PathSelection::Shortest(1),
                    ast::PathFunction::AllShortestPaths => plan::PathSelection::ShortestGroups(1),
                });
            }
            if matches!(aliased.pattern, ast::Pattern::Node(_)) {
                return None;
            }
            match aliased
                .search_prefix
                .as_ref()
                .or(match_clause.search_prefix.as_ref())?
            {
                ast::PathSearchPrefix::All => None,
                ast::PathSearchPrefix::Any => Some(plan::PathSelection::Any(1)),
                ast::PathSearchPrefix::AnyK(count) => Some(plan::PathSelection::Any(*count)),
                ast::PathSearchPrefix::AnyShortest => Some(plan::PathSelection::Shortest(1)),
                ast::PathSearchPrefix::AllShortest => Some(plan::PathSelection::ShortestGroups(1)),
                ast::PathSearchPrefix::ShortestK(count) => {
                    Some(plan::PathSelection::Shortest(*count))
                }
                ast::PathSearchPrefix::ShortestKGroups(count) => {
                    Some(plan::PathSelection::ShortestGroups(*count))
                }
            }
        };

        // Collect variables for each pattern to detect shared variables
        let pattern_vars: Vec<HashSet<String>> = match_clause
            .patterns
            .iter()
            .map(|ap| Self::pattern_variables(&ap.pattern))
            .collect();

        // The variables the input binds, those of the outer row for a
        // subquery's `WITH *` included (none when they are not known here)
        let input_vars = initial_input
            .as_ref()
            .and_then(|input| input.bound_variables(self.call_scope.borrow().as_ref()))
            .unwrap_or_default();

        let mut plan: Option<LogicalOperator> = initial_input;
        let mut bound_vars: HashSet<String> = HashSet::new();

        for (index, aliased_pattern) in match_clause.patterns.iter().enumerate() {
            let current_vars = &pattern_vars[index];
            let mut shared = comma_part_join_variables(
                current_vars,
                Self::pattern_start(&aliased_pattern.pattern),
                &bound_vars,
                &input_vars,
            );

            let selection = selection_of(aliased_pattern);
            let path_mode = path_mode_of(aliased_pattern);

            // A path search through an edge bound before is limited to that
            // edge, so it goes on from the rows that bind it: searched on its
            // own and joined after, it would pick its paths without it.
            let earlier: HashSet<String> = bound_vars.union(&input_vars).cloned().collect();
            if selection.is_some()
                && let ast::Pattern::Path(path) = &aliased_pattern.pattern
                && path
                    .edges
                    .iter()
                    .filter_map(|edge| edge.variable.as_ref())
                    .any(|name| earlier.contains(name))
            {
                shared.clear();
            }

            // The part, going on from `pattern_input` when there is one.
            let translate_part = |pattern_input: Option<LogicalOperator>| {
                if let Some(selection) = selection {
                    self.translate_path_search(
                        &aliased_pattern.pattern,
                        aliased_pattern.alias.as_deref(),
                        PathSearch {
                            selection,
                            path_mode,
                        },
                        pattern_input,
                        &earlier,
                    )
                } else {
                    self.translate_pattern_with_alias(
                        &aliased_pattern.pattern,
                        pattern_input,
                        aliased_pattern.alias.as_deref(),
                        path_mode,
                    )
                }
            };

            // Determine the input for this pattern: if shared variables exist,
            // translate independently and join; otherwise go on from the rows
            // before (a cross product, or an expand from the bound start). A
            // part that reads a value of those rows goes on from them too.
            let pattern_input = if shared.is_empty() { plan.take() } else { None };
            let mut pattern_plan = translate_part(pattern_input)?;
            if !shared.is_empty()
                && comma_part_reads_earlier_rows(
                    &pattern_plan,
                    current_vars,
                    &bound_vars,
                    &input_vars,
                )
            {
                shared.clear();
                pattern_plan = translate_part(plan.take())?;
            }

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

        // DIFFERENT EDGES: no two edge patterns of the clause bind the same
        // edge. KEEP DIFFERENT EDGES asks the same of the edge patterns of one
        // path pattern.
        if different_edges {
            let patterns: Vec<&ast::Pattern> =
                match_clause.patterns.iter().map(|p| &p.pattern).collect();
            if let Some(predicate) = pattern::different_edges(&patterns)? {
                result = wrap_filter(result, predicate);
            }
        }
        for aliased in match_clause
            .patterns
            .iter()
            .filter(|p| keeps_different_edges(p))
        {
            if let Some(predicate) = pattern::different_edges(&[&aliased.pattern])? {
                result = wrap_filter(result, predicate);
            }
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

    /// Applies a REMOVE clause to `plan`: its label removals, then its
    /// property removals.
    fn apply_remove(
        mut plan: LogicalOperator,
        remove_clause: &ast::RemoveClause,
    ) -> LogicalOperator {
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
        plan
    }

    /// Applies a WITH clause to `plan`: its projection (or aggregation), the
    /// LET bindings attached to it, its WHERE and DISTINCT. The clauses after
    /// it read the rows it passes on.
    fn apply_with(
        &self,
        mut plan: LogicalOperator,
        with_clause: &ast::WithClause,
    ) -> Result<LogicalOperator> {
        // Whether the WITH groups its rows by aggregates; its WHERE is then
        // split around the Aggregate
        let mut has_aggregates = false;
        if !with_clause.is_wildcard {
            if with_clause.items.iter().any(|item| {
                item.alias.is_none() && !matches!(item.expression, ast::Expression::Variable(_))
            }) {
                return Err(super::common::unaliased_with_expression());
            }

            // Aggregates over a group variable are computed per row first
            // (see `take_horizontal_aggregates`)
            let aggregate::WithoutHorizontalAggregates {
                items,
                plan: horizontal_plan,
                ..
            } = self.take_horizontal_aggregates(&with_clause.items, None, None, plan)?;
            plan = horizontal_plan;

            // In a CALL body, the imports this WITH leaves out pass on too.
            let projected: Vec<(String, bool)> = with_clause
                .items
                .iter()
                .filter_map(|item| match (&item.alias, &item.expression) {
                    (Some(alias), ast::Expression::Variable(name)) => {
                        Some((alias.clone(), alias == name))
                    }
                    (Some(alias), _) => Some((alias.clone(), false)),
                    (None, ast::Expression::Variable(name)) => Some((name.clone(), true)),
                    (None, _) => None,
                })
                .collect();
            let missing = self.call_imports.left_out(&projected);

            // Check if WITH contains aggregate functions (e.g. WITH count(n) AS cnt)
            has_aggregates = items
                .iter()
                .any(|item| contains_aggregate(&item.expression));

            if has_aggregates {
                let (aggregates, auto_group_by, post_return) =
                    self.extract_aggregates_and_groups(&items, false)?;

                // Split the WHERE into HAVING (aggregate-referencing
                // conjuncts) and a post-aggregate filter (the rest).
                // This handles mixed predicates like
                // `WHERE a.name = 'Alix' AND cnt > 2` correctly. A conjunct
                // that reads an import the aggregation leaves out filters
                // after the import is back.
                let aggregate_aliases: Vec<String> =
                    aggregates.iter().filter_map(|a| a.alias.clone()).collect();
                let (having, post_agg_filter) = if let Some(where_clause) =
                    &with_clause.where_clause
                {
                    let pred = self.translate_expression(&where_clause.expression)?;
                    let conjuncts = flatten_and_conjuncts(&pred);
                    let (having_parts, filter_parts): (Vec<_>, Vec<_>) =
                        conjuncts.into_iter().partition(|c| {
                            references_any(c, &aggregate_aliases) && !references_any(c, &missing)
                        });
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
                plan = with_imports(plan, missing, true);

                // Apply non-aggregate WHERE conjuncts as a post-aggregate filter.
                if let Some(filter_pred) = post_agg_filter {
                    plan = wrap_filter(plan, filter_pred);
                }
            } else {
                let projections: Vec<Projection> = items
                    .iter()
                    .map(|item| {
                        Ok(Projection {
                            expression: self.translate_expression(&item.expression)?,
                            alias: item.alias.clone(),
                        })
                    })
                    .collect::<Result<_>>()?;

                plan = with_imports(
                    LogicalOperator::Project(ProjectOp {
                        projections,
                        input: Box::new(plan),
                        pass_through_input: false,
                    }),
                    missing,
                    false,
                );
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
        if let Some(where_clause) = &with_clause.where_clause
            && !has_aggregates
        {
            let predicate = self.translate_expression(&where_clause.expression)?;
            plan = wrap_filter(plan, predicate);
        }

        // Handle DISTINCT
        if with_clause.distinct {
            plan = wrap_distinct(plan);
        }
        Ok(plan)
    }

    /// Applies an `<order by and page statement>` before the result statement
    /// to `plan`: its ORDER BY, then its OFFSET, then its LIMIT. The clauses
    /// after it read only the rows it keeps, in its order.
    fn apply_order_by_and_page(
        &self,
        mut plan: LogicalOperator,
        page: &ast::OrderByAndPage,
    ) -> Result<LogicalOperator> {
        if let Some(order_by) = &page.order_by {
            let keys = order_by
                .items
                .iter()
                .map(|item| self.translate_sort_key(item))
                .collect::<Result<Vec<_>>>()?;
            plan = wrap_sort(plan, keys);
        }
        if let Some(offset) = &page.offset {
            plan = wrap_skip(plan, Self::eval_as_count_expr(offset)?);
        }
        if let Some(limit) = &page.limit {
            plan = wrap_limit(plan, Self::eval_as_count_expr(limit)?);
        }
        Ok(plan)
    }

    /// Translates one ORDER BY item to a sort key.
    fn translate_sort_key(&self, item: &ast::OrderByItem) -> Result<SortKey> {
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
    }

    /// Translates `CALL { subquery }` to an `Apply` that runs the subquery for
    /// each row of `outer`. As in GQL, the subquery sees the outer row's
    /// variables: all of them, or the ones its variable scope clause names
    /// (`CALL (a, b) { ... }`; none for `CALL () { ... }`). It starts from a
    /// `ParameterScan` of them, which the planner fills for each row through
    /// `ParameterState`. They stay in scope for the whole body: a `WITH` in
    /// it passes them on whether it names them or not (see [`CallImports`]).
    fn translate_inline_call(
        &self,
        subquery: &ast::QueryStatement,
        combined: &[(ast::CompositeOp, ast::QueryStatement)],
        outer: LogicalOperator,
        optional: bool,
        scope: Option<&[String]>,
    ) -> Result<LogicalOperator> {
        // A CALL that comes first has no outer row to see: it runs once, on
        // one empty row (`Empty`). A scope clause there names variables the
        // binder reports as undefined.
        let shared_variables: Vec<String> = match scope {
            Some(names) => names.to_vec(),
            None if matches!(outer, LogicalOperator::Empty) => Vec::new(),
            None => vec!["*".to_string()],
        };
        let input = if shared_variables.is_empty() {
            LogicalOperator::Empty
        } else {
            LogicalOperator::ParameterScan(ParameterScanOp {
                columns: shared_variables.clone(),
            })
        };
        // The outer row's variables: what a `RETURN *` of the subquery leaves
        // out, and the scope of a CALL nested in it.
        let outer_names = outer.bound_variables(self.call_scope.borrow().as_ref());
        let enclosing = self.call_scope.replace(outer_names.clone());
        let imports = call_imports(&shared_variables, outer_names.as_ref());
        // Each query combined in the body starts from the same outer row,
        // with the same imports.
        let translate_part = |part: &ast::QueryStatement| -> Result<LogicalOperator> {
            let mut plan = self
                .call_imports
                .within(imports.clone(), || {
                    self.translate_query_from(part, input.clone())
                })?
                .root;
            expand_subquery_return_star(&mut plan, outer_names.as_ref())?;
            Ok(plan)
        };
        let inner = translate_part(subquery).and_then(|first| {
            combined.iter().try_fold(first, |plan, (op, part)| {
                combine_queries(*op, plan, translate_part(part)?)
            })
        });
        self.call_scope.replace(enclosing);
        let inner_plan = inner?;
        // A body that ends without a result (no RETURN, or FINISH) runs for
        // its writes and passes each row on once, as it came in.
        let last = combined.last().map_or(subquery, |(_, part)| part);
        Ok(LogicalOperator::Apply(ApplyOp {
            input: Box::new(outer),
            subplan: Box::new(inner_plan),
            shared_variables,
            optional,
            unit: !returns_rows(last),
        }))
    }

    fn translate_match(&self, match_clause: &ast::MatchClause) -> Result<LogicalOperator> {
        self.translate_match_with_input(match_clause, None)
    }

    /// Applies the WHERE or FILTER of the statement to `plan`.
    ///
    /// After an OPTIONAL MATCH (`after_optional_match`) a WHERE is part of
    /// its graph pattern (ISO/IEC 39075:2024 16.4): it decides which matches
    /// count and keeps every row of the clauses before it, so its conjuncts go
    /// into the optional side or the join condition (see
    /// [`build_left_join_with_predicates`]). A FILTER, and a WHERE after any
    /// other clause, filters the rows, also those of a MATCH whose questioned
    /// edge (`->?`) is a left join of its own.
    fn apply_statement_where(
        &self,
        plan: LogicalOperator,
        where_clause: &ast::WhereClause,
        after_optional_match: bool,
    ) -> Result<LogicalOperator> {
        let predicate = self.translate_expression(&where_clause.expression)?;
        let of_optional_match = after_optional_match && !where_clause.filter;
        Ok(
            if of_optional_match && let LogicalOperator::LeftJoin(left_join) = plan {
                let (join, post_filter) = build_left_join_with_predicates(
                    left_join,
                    Some(predicate),
                    self.call_scope.borrow().as_ref(),
                );
                if let Some(pf) = post_filter {
                    wrap_filter(join, pf)
                } else {
                    join
                }
            } else {
                wrap_filter(plan, predicate)
            },
        )
    }

    /// Translates a selective path pattern (a path search prefix other than
    /// `ALL`, or `shortestPath`) into a path search: for each input row and
    /// each pair of endpoints, the paths of `search.selection` among those
    /// of `search.path_mode` (ISO/IEC 39075:2024 16.6). The edge pattern's
    /// own conditions hold during the search, before the selection.
    ///
    /// `ANY` searches from the source to every node it reaches, unless the
    /// target is bound already (by the input rows, or as the source): one
    /// search per source, instead of one per pair of a source and a node.
    fn translate_path_search(
        &self,
        pattern: &ast::Pattern,
        alias: Option<&str>,
        search: PathSearch,
        input: Option<LogicalOperator>,
        bound: &HashSet<String>,
    ) -> Result<LogicalOperator> {
        // One edge pattern between two node patterns
        let any = matches!(search.selection, plan::PathSelection::Any(_));
        let what = if any { "an ANY" } else { "a shortest" };
        let ast::Pattern::Path(path) = pattern else {
            return Err(Error::Query(QueryError::new(
                QueryErrorKind::Semantic,
                format!(
                    "{what} path search over a parenthesized, alternated or node-only path \
                     pattern is not supported: it runs over one edge pattern between two nodes"
                ),
            )));
        };
        let edge = match path.edges.as_slice() {
            [edge] => edge,
            [] => {
                return Err(Error::Query(QueryError::new(
                    QueryErrorKind::Semantic,
                    format!("{what} path search needs an edge pattern between two nodes"),
                )));
            }
            // Only the first edge pattern would be searched, and the nodes
            // between the edge patterns ignored
            _ => {
                return Err(Error::Query(QueryError::new(
                    QueryErrorKind::Semantic,
                    format!("{what} path search over more than one edge pattern is not supported"),
                )));
            }
        };
        let direction = match edge.direction {
            ast::EdgeDirection::Outgoing => ExpandDirection::Outgoing,
            ast::EdgeDirection::Incoming => ExpandDirection::Incoming,
            ast::EdgeDirection::Undirected => ExpandDirection::Both,
        };
        // The path must fit the edge's quantifier: `->+` needs at least
        // one hop, and an edge without one is a single hop.
        let (min_hops, max_hops) = pattern::edge_hop_bounds(edge);
        let quantified = edge.min_hops.is_some() || edge.max_hops.is_some();

        // The search finds the paths between the endpoints, so they are not
        // expanded between: the source is scanned, and the target too unless
        // the search binds it. An anonymous endpoint gets a variable the
        // search can name.
        let named = |node: &ast::NodePattern| {
            let mut node = node.clone();
            node.variable.get_or_insert_with(|| self.anonymous_name());
            node
        };
        let (source_node, target_node) = (named(&path.source), named(&edge.target));
        let source_var = source_node.variable.clone().unwrap_or_default();
        let target_var = target_node.variable.clone().unwrap_or_default();

        // The edge variable binds the edges of each path, unless a pattern
        // before bound it: the path then takes that edge, which the search
        // checks for each edge like the edge pattern's own conditions
        let bound_edge = edge.variable.as_ref().filter(|name| bound.contains(*name));
        if let (Some(name), true) = (bound_edge, quantified) {
            return Err(Error::Query(QueryError::new(
                QueryErrorKind::Semantic,
                format!(
                    "'{name}' is bound to one edge before, so a quantified edge pattern cannot \
                     bind it to a list of edges"
                ),
            )));
        }
        let candidate = match (&edge.variable, bound_edge) {
            (Some(name), None) => name.clone(),
            _ => self.anonymous_name(),
        };
        let mut conditions = Vec::new();
        if let Some(name) = bound_edge {
            conditions.push(LogicalExpression::Binary {
                left: Box::new(LogicalExpression::Id(candidate.clone())),
                op: BinaryOp::Eq,
                right: Box::new(LogicalExpression::Id(name.clone())),
            });
        }
        if !edge.properties.is_empty() {
            conditions.push(self.build_property_predicate(&candidate, &edge.properties)?);
        }
        if let Some(where_expr) = &edge.where_clause {
            conditions.push(self.translate_expression(where_expr)?);
        }
        let edge_condition =
            join_and_conjuncts(conditions).map(|predicate| ShortestPathEdgeCondition {
                variable: candidate,
                predicate,
            });

        // ANY searches to every node, unless the target is bound (by the
        // input rows or as the source) or the edge condition reads it, which
        // the search to every node has not bound yet
        let reads_target = edge_condition.as_ref().is_some_and(|condition| {
            let mut read = HashSet::new();
            collect_expression_variables(&condition.predicate, &mut read);
            read.contains(&target_var)
        });
        let binds_target = any
            && target_var != source_var
            && !(input.is_some() && bound.contains(&target_var))
            && !reads_target;
        let source_plan = self.translate_node_pattern(&source_node, input)?;
        let search_input = if binds_target {
            source_plan
        } else {
            self.translate_node_pattern(&target_node, Some(source_plan))?
        };

        let path_alias = alias.map_or_else(|| self.names.next("_anon_path_"), String::from);
        let edge_variable = edge.variable.clone().filter(|_| bound_edge.is_none());
        // A sum over the edges of the path (`sum(e.w)`) reads its edge column
        if quantified && let Some(name) = &edge_variable {
            self.group_list_variables
                .borrow_mut()
                .insert(name.clone(), path_alias.clone());
        }

        let search = LogicalOperator::ShortestPath(ShortestPathOp {
            input: Box::new(search_input),
            source_var,
            target_var,
            edge_types: edge.types.clone(),
            direction,
            path_alias,
            selection: search.selection,
            path_mode: search.path_mode,
            binds_target,
            min_hops,
            max_hops,
            edge_variable,
            quantified,
            edge_condition,
        });
        // The target's labels, properties and WHERE hold for each node the
        // search binds: they read the target alone, so they keep or drop
        // whole partitions, after the selection as before it
        let target_conditions = !target_node.labels.is_empty()
            || target_node.label_expression.is_some()
            || !target_node.properties.is_empty()
            || target_node.where_clause.is_some();
        if binds_target && target_conditions {
            return self.translate_node_pattern(&target_node, Some(search));
        }
        Ok(search)
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

    /// Translates a data-modifying statement of one clause (a lone `INSERT`
    /// or `DELETE`) on the rows of `input`: `Empty` for a statement of its
    /// own, the rows the statement before a `NEXT` passes on. It has no
    /// `RETURN` or `FINISH`, so it has no result (an omitted result, ISO/IEC
    /// 39075:2024 13.1; see `no_result`): it returned the last node an INSERT
    /// created.
    fn translate_data_modification(
        &self,
        dm: &ast::DataModificationStatement,
        input: LogicalOperator,
    ) -> Result<LogicalPlan> {
        let plan = match dm {
            ast::DataModificationStatement::Insert(insert) => {
                self.translate_insert(&insert.patterns, input)?
            }
            ast::DataModificationStatement::Delete(delete) => {
                self.translate_delete(delete, input)?
            }
            ast::DataModificationStatement::Set(set) => self.translate_set(set, input)?,
        };
        Ok(LogicalPlan::new(no_result(plan)))
    }

    /// Translates a lone DELETE on the rows of `input`. Its targets read the
    /// variables those rows bind: one that starts its statement (`Empty`)
    /// has none, so the binder refuses `DELETE w` (it scanned the graph for
    /// `w` and deleted every node).
    fn translate_delete(
        &self,
        delete: &ast::DeleteStatement,
        input: LogicalOperator,
    ) -> Result<LogicalOperator> {
        if delete.targets.is_empty() {
            return Err(Error::Query(QueryError::new(
                QueryErrorKind::Semantic,
                "DELETE requires at least one target",
            )));
        }
        self.translate_delete_targets(&delete.targets, delete.detach, input)
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

    /// Translates a lone SET on the rows of `input`, whose variables its
    /// assignments read (the parser makes none; a SET is a clause of a
    /// query).
    fn translate_set(
        &self,
        set: &ast::SetStatement,
        input: LogicalOperator,
    ) -> Result<LogicalOperator> {
        if set.assignments.is_empty() {
            return Err(Error::Query(QueryError::new(
                QueryErrorKind::Semantic,
                "SET requires at least one assignment",
            )));
        }

        // Group assignments by variable
        let first_assignment = &set.assignments[0];
        let var = &first_assignment.variable;

        // Build property assignments for this variable
        let properties: Vec<(String, LogicalExpression)> = set
            .assignments
            .iter()
            .filter(|a| &a.variable == var)
            .map(|a| Ok((a.property.clone(), self.translate_expression(&a.value)?)))
            .collect::<Result<_>>()?;

        Ok(LogicalOperator::SetProperty(SetPropertyOp {
            variable: var.clone(),
            properties,
            replace: false,
            is_edge: false,
            input: Box::new(input),
        }))
    }

    /// Translates the patterns of an INSERT on the rows of `input`: `Empty`
    /// when the INSERT starts its statement.
    fn translate_insert(
        &self,
        patterns: &[ast::Pattern],
        input: LogicalOperator,
    ) -> Result<LogicalOperator> {
        if matches!(input, LogicalOperator::Empty) {
            self.insert_chain(patterns)
        } else {
            self.translate_create_patterns(patterns, input)
        }
    }

    /// Builds the [`CreateOp`] of an INSERT that starts a statement (no input
    /// rows): a standalone INSERT, or the first INSERT clause of a query such
    /// as `INSERT (a) INSERT (b) RETURN a, b`.
    fn insert_chain(&self, patterns: &[ast::Pattern]) -> Result<LogicalOperator> {
        if patterns.is_empty() {
            return Err(Error::Query(QueryError::new(
                QueryErrorKind::Semantic,
                "Empty INSERT statement",
            )));
        }

        let mut elements = Vec::new();
        // With no input rows, a variable is bound only if this INSERT created
        // it earlier: `INSERT (a:A), (a)-[:T]->(b)` creates `a` once and `b`.
        let mut bound: HashSet<String> = HashSet::new();

        for pattern in patterns {
            match pattern {
                ast::Pattern::Node(node) => {
                    let (variable, is_new) =
                        pattern::insert_endpoint(node, &mut bound, true, &self.names)?;
                    if is_new {
                        elements.push(self.created_node(&variable, node)?);
                    }
                }
                ast::Pattern::Path(path) => {
                    let (source_var, is_new) =
                        pattern::insert_endpoint(&path.source, &mut bound, true, &self.names)?;
                    if is_new {
                        elements.push(self.created_node(&source_var, &path.source)?);
                    }

                    let mut current_src = source_var;
                    for edge in &path.edges {
                        let (target_var, is_new) =
                            pattern::insert_endpoint(&edge.target, &mut bound, true, &self.names)?;
                        if is_new {
                            elements.push(self.created_node(&target_var, &edge.target)?);
                        }

                        let (from, to) = match edge.direction {
                            ast::EdgeDirection::Incoming => (target_var.clone(), current_src),
                            _ => (current_src, target_var.clone()),
                        };
                        elements.push(self.created_edge(edge, from, to)?);
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

        if elements.is_empty() {
            return Err(Error::Query(QueryError::new(
                QueryErrorKind::Semantic,
                "INSERT must create at least one node",
            )));
        }
        Ok(LogicalOperator::Create(CreateOp {
            elements,
            input: None,
        }))
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
            // Each MATCH goes on from the ones before it, as in the outer
            // query, so a variable in two clauses is the same node or edge.
            plan = if match_clause.optional {
                optional_join(
                    plan,
                    self.translate_match(match_clause)?,
                    self.call_scope.borrow().as_ref(),
                )
            } else if matches!(plan, LogicalOperator::Empty) {
                self.translate_match(match_clause)?
            } else {
                self.translate_match_with_input(match_clause, Some(plan))?
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

    /// If the RETURN clause is a single `count()` aggregate of at most one
    /// argument, returns that argument (`None` for `count(*)`) and whether it
    /// is counted DISTINCT.
    fn count_aggregate_return(ret: &ast::ReturnClause) -> Option<(Option<&ast::Expression>, bool)> {
        let [item] = ret.items.as_slice() else {
            return None;
        };
        match &item.expression {
            ast::Expression::FunctionCall {
                name,
                args,
                distinct,
            } if name.eq_ignore_ascii_case("count") && args.len() <= 1 => {
                Some((args.first(), *distinct))
            }
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

/// The selection and path mode of a selective path pattern (see
/// `GqlTranslator::translate_path_search`).
#[derive(Debug, Clone, Copy)]
struct PathSearch {
    /// Which paths of each pair of endpoints the search keeps.
    selection: plan::PathSelection,
    /// The paths the search follows.
    path_mode: PathMode,
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
        // Find the Create that creates the node
        fn find_create(op: &LogicalOperator) -> bool {
            match op {
                LogicalOperator::Create(create) => matches!(
                    create.elements.as_slice(),
                    [CreateElement::Node { variable, labels, .. }]
                        if variable == "n" && labels == &["Person"]
                ),
                LogicalOperator::Return(r) => find_create(&r.input),
                _ => false,
            }
        }

        assert!(find_create(&plan.root));
    }

    #[test]
    fn test_translate_insert_of_many_patterns_is_one_create() {
        let patterns: Vec<String> = (0..1_000)
            .map(|i| format!("(:Person {{v: {i}}})-[:KNOWS]->(:Person)"))
            .collect();
        let plan = translate(&format!("INSERT {}", patterns.join(", "))).unwrap();
        let LogicalOperator::Return(ret) = &plan.root else {
            panic!("a Return over the Create: {:?}", plan.root)
        };
        let LogicalOperator::Create(create) = ret.input.as_ref() else {
            panic!("one Create for every pattern: {:?}", ret.input)
        };
        assert!(create.input.is_none());
        assert_eq!(
            create.elements.len(),
            3_000,
            "two nodes and an edge per pattern"
        );
        let CreateElement::Edge {
            from_variable,
            to_variable,
            ..
        } = &create.elements[2]
        else {
            panic!("the edge after its endpoints")
        };
        assert_eq!(
            create.variables().take(2).collect::<Vec<_>>(),
            [from_variable, to_variable]
        );
    }

    #[test]
    fn test_translate_delete() {
        let query = "DELETE n";
        let result = translate(query);
        assert!(result.is_ok());

        // A lone DELETE has no result: a RETURN of no items over the write.
        let plan = result.unwrap();
        let LogicalOperator::Return(ret) = &plan.root else {
            panic!("Expected Return, got {:?}", plan.root);
        };
        assert!(ret.items.is_empty(), "no columns: {ret:?}");
        if let LogicalOperator::DeleteNode(del) = ret.input.as_ref() {
            assert_eq!(del.variable, "n");
        } else {
            panic!("Expected DeleteNode operator");
        }
    }

    #[test]
    fn test_translate_set() {
        // SET is not a standalone statement in GQL, test the translator method directly
        let translator = GqlTranslator::new("");
        let set_stmt = ast::SetStatement {
            assignments: vec![ast::PropertyAssignment {
                variable: "n".to_string(),
                property: "name".to_string(),
                value: ast::Expression::Literal(ast::Literal::String("Gus".to_string())),
            }],
            span: None,
        };

        let plan = translator
            .translate_data_modification(
                &ast::DataModificationStatement::Set(set_stmt),
                LogicalOperator::Empty,
            )
            .unwrap();

        // A lone SET has no result: a RETURN of no items over the write.
        let LogicalOperator::Return(ret) = &plan.root else {
            panic!("Expected Return, got {:?}", plan.root);
        };
        assert!(ret.items.is_empty(), "no columns: {ret:?}");
        if let LogicalOperator::SetProperty(set) = ret.input.as_ref() {
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
        let translator = GqlTranslator::new("");
        let delete = ast::DeleteStatement {
            targets: vec![],
            detach: false,
            span: None,
        };
        let result = translator.translate_delete(&delete, LogicalOperator::Empty);
        assert!(result.is_err());
    }

    #[test]
    fn test_translate_empty_set_error() {
        let translator = GqlTranslator::new("");
        let set = ast::SetStatement {
            assignments: vec![],
            span: None,
        };
        let result = translator.translate_set(&set, LogicalOperator::Empty);
        assert!(result.is_err());
    }

    #[test]
    fn test_translate_empty_insert_error() {
        let translator = GqlTranslator::new("");
        let insert = ast::InsertStatement {
            patterns: vec![],
            span: None,
        };
        let result = translator.translate_insert(&insert.patterns, LogicalOperator::Empty);
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
        let translator = GqlTranslator::new("");

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
        let translator = GqlTranslator::new("");

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
        let query = "MATCH (n:Person) RETURN n EXCEPT MATCH (n:Employee) RETURN n";
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
        let query = "MATCH (n:Person) RETURN n INTERSECT MATCH (n:Employee) RETURN n";
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
        let query = "MATCH (n:Person) RETURN n OTHERWISE MATCH (n:Employee) RETURN n";
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
        // FINISH is a statement without a result (a RETURN of no items) over
        // a Limit(0)
        let LogicalOperator::Return(ret) = &plan.root else {
            panic!(
                "Expected a RETURN of no items for FINISH, got {:?}",
                plan.root
            );
        };
        assert!(ret.items.is_empty(), "FINISH returns no columns");
        if let LogicalOperator::Limit(limit) = ret.input.as_ref() {
            assert_eq!(limit.count, 0, "FINISH should produce Limit(0)");
        } else {
            panic!("Expected Limit operator for FINISH, got {:?}", ret.input);
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
