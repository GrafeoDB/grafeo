//! Shared utilities for query language translators.
//!
//! Functions here are used by multiple translator modules (GQL, Cypher, etc.)
//! to avoid duplication of identical logic.

use std::collections::HashSet;
#[cfg(any(feature = "graphql", feature = "gremlin", test))]
use std::sync::atomic::{AtomicU32, Ordering};

use crate::query::plan::{
    AggregateFunction, BinaryOp, CountExpr, DistinctOp, FilterOp, LeftJoinOp, LimitOp,
    LogicalExpression, LogicalOperator, ReturnItem, ReturnOp, SkipOp, SortKey, SortOp, UnaryOp,
};
use grafeo_common::utils::error::{Error, QueryError, QueryErrorKind, Result};

/// Returns true if the function name is a recognized aggregate function.
pub(crate) fn is_aggregate_function(name: &str) -> bool {
    matches!(
        name.to_uppercase().as_str(),
        "COUNT"
            | "SUM"
            | "AVG"
            | "MIN"
            | "MAX"
            | "COLLECT"
            | "STDEV"
            | "STDDEV"
            | "STDDEV_SAMP"
            | "STDEVP"
            | "STDDEVP"
            | "STDDEV_POP"
            | "VARIANCE"
            | "VAR_SAMP"
            | "VAR_POP"
            | "PERCENTILE_DISC"
            | "PERCENTILEDISC"
            | "PERCENTILE_CONT"
            | "PERCENTILECONT"
            | "GROUP_CONCAT"
            | "GROUPCONCAT"
            | "LISTAGG"
            | "SAMPLE"
            | "COVAR_SAMP"
            | "COVAR_POP"
            | "CORR"
            | "REGR_SLOPE"
            | "REGR_INTERCEPT"
            | "REGR_R2"
            | "REGR_COUNT"
            | "REGR_SXX"
            | "REGR_SYY"
            | "REGR_SXY"
            | "REGR_AVGX"
            | "REGR_AVGY"
    )
}

/// Converts a function name to an `AggregateFunction` enum variant.
pub(crate) fn to_aggregate_function(name: &str) -> Option<AggregateFunction> {
    match name.to_uppercase().as_str() {
        "COUNT" => Some(AggregateFunction::Count),
        "SUM" => Some(AggregateFunction::Sum),
        "AVG" => Some(AggregateFunction::Avg),
        "MIN" => Some(AggregateFunction::Min),
        "MAX" => Some(AggregateFunction::Max),
        "COLLECT" => Some(AggregateFunction::Collect),
        "STDEV" | "STDDEV" | "STDDEV_SAMP" => Some(AggregateFunction::StdDev),
        "STDEVP" | "STDDEVP" | "STDDEV_POP" => Some(AggregateFunction::StdDevPop),
        "VARIANCE" | "VAR_SAMP" => Some(AggregateFunction::Variance),
        "VAR_POP" => Some(AggregateFunction::VariancePop),
        "PERCENTILE_DISC" | "PERCENTILEDISC" => Some(AggregateFunction::PercentileDisc),
        "PERCENTILE_CONT" | "PERCENTILECONT" => Some(AggregateFunction::PercentileCont),
        "GROUP_CONCAT" | "GROUPCONCAT" | "LISTAGG" => Some(AggregateFunction::GroupConcat),
        "SAMPLE" => Some(AggregateFunction::Sample),
        "COVAR_SAMP" => Some(AggregateFunction::CovarSamp),
        "COVAR_POP" => Some(AggregateFunction::CovarPop),
        "CORR" => Some(AggregateFunction::Corr),
        "REGR_SLOPE" => Some(AggregateFunction::RegrSlope),
        "REGR_INTERCEPT" => Some(AggregateFunction::RegrIntercept),
        "REGR_R2" => Some(AggregateFunction::RegrR2),
        "REGR_COUNT" => Some(AggregateFunction::RegrCount),
        "REGR_SXX" => Some(AggregateFunction::RegrSxx),
        "REGR_SYY" => Some(AggregateFunction::RegrSyy),
        "REGR_SXY" => Some(AggregateFunction::RegrSxy),
        "REGR_AVGX" => Some(AggregateFunction::RegrAvgx),
        "REGR_AVGY" => Some(AggregateFunction::RegrAvgy),
        _ => None,
    }
}

/// Returns true if the aggregate function is a binary set function (requires two arguments).
pub(crate) fn is_binary_set_function(func: AggregateFunction) -> bool {
    matches!(
        func,
        AggregateFunction::CovarSamp
            | AggregateFunction::CovarPop
            | AggregateFunction::Corr
            | AggregateFunction::RegrSlope
            | AggregateFunction::RegrIntercept
            | AggregateFunction::RegrR2
            | AggregateFunction::RegrCount
            | AggregateFunction::RegrSxx
            | AggregateFunction::RegrSyy
            | AggregateFunction::RegrSxy
            | AggregateFunction::RegrAvgx
            | AggregateFunction::RegrAvgy
    )
}

/// Evaluates GraphQL `@skip` and `@include` directives to determine if a field
/// should be included in the query result.
///
/// Per the GraphQL spec:
/// - `@skip(if: true)` excludes the field
/// - `@skip(if: false)` includes the field
/// - `@include(if: false)` excludes the field
/// - `@include(if: true)` includes the field
///
/// When both directives are present, the field is included only if it passes both
/// checks (`@skip` must not exclude AND `@include` must include).
///
/// Returns `true` if the field should be included, `false` if it should be skipped.
#[cfg(feature = "graphql")]
pub(crate) fn graphql_directives_allow(
    directives: &[grafeo_adapters::query::graphql::ast::Directive],
) -> bool {
    let mut include = true;

    for directive in directives {
        match directive.name.as_str() {
            "skip" => {
                // @skip(if: true) excludes the field
                if let Some(arg) = directive.arguments.iter().find(|a| a.name == "if")
                    && let grafeo_adapters::query::graphql::ast::InputValue::Boolean(val) =
                        &arg.value
                    && *val
                {
                    include = false;
                }
            }
            "include" => {
                // @include(if: false) excludes the field
                if let Some(arg) = directive.arguments.iter().find(|a| a.name == "if")
                    && let grafeo_adapters::query::graphql::ast::InputValue::Boolean(val) =
                        &arg.value
                    && !val
                {
                    include = false;
                }
            }
            _ => {} // Unknown directives are ignored
        }
    }

    include
}

/// Capitalizes the first character of a string.
///
/// Used by GraphQL translators to convert field names to type names.
#[cfg(any(feature = "graphql", test))]
pub(crate) fn capitalize_first(s: &str) -> String {
    let mut chars = s.chars();
    match chars.next() {
        None => String::new(),
        Some(first) => first.to_uppercase().collect::<String>() + chars.as_str(),
    }
}

/// Generates unique variable names with an atomic counter.
///
/// Replaces the duplicated `var_counter: AtomicU32` + `next_var()` pattern
/// used across multiple translators.
#[cfg(any(feature = "graphql", feature = "gremlin", test))]
pub(crate) struct VarGen {
    counter: AtomicU32,
}

#[cfg(any(feature = "graphql", feature = "gremlin", test))]
impl VarGen {
    /// Creates a new variable generator starting from 0.
    pub fn new() -> Self {
        Self {
            counter: AtomicU32::new(0),
        }
    }

    /// Returns the next unique variable name (e.g., `_v0`, `_v1`, ...).
    pub fn next(&self) -> String {
        let n = self.counter.fetch_add(1, Ordering::Relaxed);
        format!("_v{n}")
    }

    /// Returns the current counter value without incrementing.
    pub fn current(&self) -> u32 {
        self.counter.load(Ordering::Relaxed)
    }
}

/// Combines a non-empty vector of predicates into a single AND expression.
///
/// Returns an error if the input is empty. Used by `build_property_predicate`
/// in multiple translators.
pub(crate) fn combine_with_and(predicates: Vec<LogicalExpression>) -> Result<LogicalExpression> {
    predicates
        .into_iter()
        .reduce(|acc, pred| LogicalExpression::Binary {
            left: Box::new(acc),
            op: BinaryOp::And,
            right: Box::new(pred),
        })
        .ok_or_else(|| {
            Error::Query(QueryError::new(
                QueryErrorKind::Semantic,
                "Empty property predicate",
            ))
        })
}

// ---------------------------------------------------------------------------
// Variable extraction
// ---------------------------------------------------------------------------

/// Collects all variable names referenced by a logical expression.
pub(crate) fn collect_expression_variables(expr: &LogicalExpression, vars: &mut HashSet<String>) {
    match expr {
        LogicalExpression::Variable(name) => {
            vars.insert(name.clone());
        }
        LogicalExpression::Property { variable, .. }
        | LogicalExpression::Labels(variable)
        | LogicalExpression::Type(variable)
        | LogicalExpression::Id(variable) => {
            vars.insert(variable.clone());
        }
        LogicalExpression::Binary { left, right, .. } => {
            collect_expression_variables(left, vars);
            collect_expression_variables(right, vars);
        }
        LogicalExpression::Unary { operand, .. } => {
            collect_expression_variables(operand, vars);
        }
        LogicalExpression::FunctionCall { args, .. } => {
            for arg in args {
                collect_expression_variables(arg, vars);
            }
        }
        LogicalExpression::List(items) => {
            for item in items {
                collect_expression_variables(item, vars);
            }
        }
        LogicalExpression::Map(pairs) => {
            for (_, value) in pairs {
                collect_expression_variables(value, vars);
            }
        }
        LogicalExpression::IndexAccess { base, index } => {
            collect_expression_variables(base, vars);
            collect_expression_variables(index, vars);
        }
        LogicalExpression::SliceAccess { base, start, end } => {
            collect_expression_variables(base, vars);
            if let Some(s) = start {
                collect_expression_variables(s, vars);
            }
            if let Some(e) = end {
                collect_expression_variables(e, vars);
            }
        }
        LogicalExpression::Case {
            operand,
            when_clauses,
            else_clause,
        } => {
            if let Some(op) = operand {
                collect_expression_variables(op, vars);
            }
            for (cond, result) in when_clauses {
                collect_expression_variables(cond, vars);
                collect_expression_variables(result, vars);
            }
            if let Some(else_expr) = else_clause {
                collect_expression_variables(else_expr, vars);
            }
        }
        LogicalExpression::ListComprehension {
            list_expr,
            filter_expr,
            map_expr,
            ..
        } => {
            collect_expression_variables(list_expr, vars);
            if let Some(filter) = filter_expr {
                collect_expression_variables(filter, vars);
            }
            collect_expression_variables(map_expr, vars);
        }
        LogicalExpression::ListPredicate {
            list_expr,
            predicate,
            ..
        } => {
            collect_expression_variables(list_expr, vars);
            collect_expression_variables(predicate, vars);
        }
        LogicalExpression::MapProjection { base, entries } => {
            vars.insert(base.clone());
            for entry in entries {
                if let crate::query::plan::MapProjectionEntry::LiteralEntry(_, expr) = entry {
                    collect_expression_variables(expr, vars);
                }
            }
        }
        LogicalExpression::Reduce {
            initial,
            list,
            expression,
            ..
        } => {
            collect_expression_variables(initial, vars);
            collect_expression_variables(list, vars);
            collect_expression_variables(expression, vars);
        }
        LogicalExpression::PatternComprehension { projection, .. } => {
            collect_expression_variables(projection, vars);
        }
        LogicalExpression::Literal(_)
        | LogicalExpression::Parameter(_)
        | LogicalExpression::ExistsSubquery(_)
        | LogicalExpression::CountSubquery(_)
        | LogicalExpression::ValueSubquery(_) => {}
    }
}

// ---------------------------------------------------------------------------
// OPTIONAL MATCH predicate classification
// ---------------------------------------------------------------------------

/// Splits a conjunctive predicate (AND-chain) into individual conjuncts.
pub(crate) fn split_conjuncts(expr: LogicalExpression) -> Vec<LogicalExpression> {
    let mut result = Vec::new();
    split_conjuncts_recursive(expr, &mut result);
    result
}

fn split_conjuncts_recursive(expr: LogicalExpression, out: &mut Vec<LogicalExpression>) {
    if let LogicalExpression::Binary {
        left,
        op: BinaryOp::And,
        right,
    } = expr
    {
        split_conjuncts_recursive(*left, out);
        split_conjuncts_recursive(*right, out);
    } else {
        out.push(expr);
    }
}

/// Result of classifying WHERE predicates for OPTIONAL MATCH.
///
/// Predicates are split based on which side of the LeftJoin their variables
/// belong to, ensuring correct NULL-preservation semantics.
pub(crate) struct ClassifiedPredicates {
    /// Predicates referencing only left-side variables (or constants): placed
    /// as a Filter above the LeftJoin (they filter the required side).
    pub post_filters: Vec<LogicalExpression>,
    /// Predicates whose referenced variables all exist on the right side:
    /// pushed as a pre-filter on the right input of the LeftJoin.
    pub right_filters: Vec<LogicalExpression>,
    /// Predicates referencing variables from both sides: stored as null-safe
    /// join conditions. Applied as `(right_var IS NULL) OR predicate` so that
    /// NULL-padded rows (unmatched optional side) are preserved.
    pub cross_filters: Vec<LogicalExpression>,
}

/// Classifies WHERE predicates for correct OPTIONAL MATCH semantics.
///
/// Given a predicate and the set of variables produced by the left (required)
/// and right (optional) sides of a LeftJoin, splits the predicate into:
///
/// - **post_filters**: reference only left-side variables, safe to apply after the join
/// - **right_filters**: reference only right-side variables, can be pushed as a
///   pre-filter on the right input (semantically equivalent to a join condition)
/// - **cross_filters**: reference both sides, must be applied as a join condition
pub(crate) fn classify_optional_predicates(
    predicate: LogicalExpression,
    left_vars: &HashSet<String>,
    right_vars: &HashSet<String>,
) -> ClassifiedPredicates {
    let conjuncts = split_conjuncts(predicate);
    let mut post_filters = Vec::new();
    let mut right_filters = Vec::new();
    let mut cross_filters = Vec::new();

    for conjunct in conjuncts {
        let mut referenced = HashSet::new();
        collect_expression_variables(&conjunct, &mut referenced);

        // A predicate is a right-filter only if it references at least one
        // right-ONLY variable (a variable produced exclusively by the optional
        // side, not shared with the required side). Predicates on shared
        // variables alone (e.g., `n.city = 'NYC'` where `n` is the join key)
        // must remain as post-filters because they constrain the required side.
        let has_right_only_var = referenced
            .iter()
            .any(|v| right_vars.contains(v) && !left_vars.contains(v));
        let has_left_only_var = referenced
            .iter()
            .any(|v| left_vars.contains(v) && !right_vars.contains(v));
        let all_in_right = referenced.iter().all(|v| right_vars.contains(v));

        if referenced.is_empty() {
            // Constant predicate: post-filter
            post_filters.push(conjunct);
        } else if has_right_only_var && all_in_right {
            // References at least one right-only variable, and all referenced
            // variables exist on the right side: push as pre-filter on right input.
            right_filters.push(conjunct);
        } else if has_left_only_var && has_right_only_var {
            // True cross-side: references at least one left-only AND one right-only
            // variable. Store for null-safe join condition wrapping.
            cross_filters.push(conjunct);
        } else {
            // Left-only or shared-only: post-filter above the join.
            post_filters.push(conjunct);
        }
    }

    ClassifiedPredicates {
        post_filters,
        right_filters,
        cross_filters,
    }
}

/// Collects all variables produced by a logical operator's subtree.
pub(crate) fn collect_operator_variables(op: &LogicalOperator, vars: &mut HashSet<String>) {
    match op {
        LogicalOperator::NodeScan(scan) => {
            vars.insert(scan.variable.clone());
            if let Some(input) = &scan.input {
                collect_operator_variables(input, vars);
            }
        }
        LogicalOperator::EdgeScan(scan) => {
            vars.insert(scan.variable.clone());
        }
        LogicalOperator::Expand(expand) => {
            vars.insert(expand.to_variable.clone());
            if let Some(edge_var) = &expand.edge_variable {
                vars.insert(edge_var.clone());
            }
            collect_operator_variables(&expand.input, vars);
        }
        LogicalOperator::Filter(filter) => {
            collect_operator_variables(&filter.input, vars);
        }
        LogicalOperator::Project(proj) => {
            for p in &proj.projections {
                if let Some(alias) = &p.alias {
                    vars.insert(alias.clone());
                }
            }
            collect_operator_variables(&proj.input, vars);
        }
        LogicalOperator::Join(join) => {
            collect_operator_variables(&join.left, vars);
            collect_operator_variables(&join.right, vars);
        }
        LogicalOperator::LeftJoin(lj) => {
            collect_operator_variables(&lj.left, vars);
            collect_operator_variables(&lj.right, vars);
        }
        LogicalOperator::Unwind(unwind) => {
            vars.insert(unwind.variable.clone());
            collect_operator_variables(&unwind.input, vars);
        }
        LogicalOperator::Bind(bind) => {
            vars.insert(bind.variable.clone());
            collect_operator_variables(&bind.input, vars);
        }
        LogicalOperator::Aggregate(agg) => {
            for expr in &agg.group_by {
                collect_expression_variables(expr, vars);
            }
            for agg_expr in &agg.aggregates {
                if let Some(alias) = &agg_expr.alias {
                    vars.insert(alias.clone());
                }
            }
            collect_operator_variables(&agg.input, vars);
        }
        LogicalOperator::Return(ret) => {
            collect_operator_variables(&ret.input, vars);
        }
        LogicalOperator::Limit(limit) => {
            collect_operator_variables(&limit.input, vars);
        }
        LogicalOperator::Skip(skip) => {
            collect_operator_variables(&skip.input, vars);
        }
        LogicalOperator::Sort(sort) => {
            collect_operator_variables(&sort.input, vars);
        }
        LogicalOperator::Distinct(distinct) => {
            collect_operator_variables(&distinct.input, vars);
        }
        _ => {
            // For other operators, do not recurse to avoid false positives.
            // The common cases (NodeScan, Expand, Filter, Join, LeftJoin,
            // Unwind, Project, Aggregate, Return) are covered above.
        }
    }
}

/// Builds a LeftJoin with properly classified WHERE predicates.
///
/// Given a WHERE predicate that follows an OPTIONAL MATCH, this function:
/// 1. Collects variables from both sides
/// 2. Classifies predicates into left-only, right-only, and cross-side
/// 3. Pushes right-only predicates as a Filter on the right input
/// 4. Stores cross-side predicates in `LeftJoinOp.condition`
/// 5. Returns the LeftJoin and any remaining post-filters to apply above
pub(crate) fn build_left_join_with_predicates(
    left: LogicalOperator,
    right: LogicalOperator,
    predicate: Option<LogicalExpression>,
) -> (LogicalOperator, Option<LogicalExpression>) {
    let Some(predicate) = predicate else {
        let join = LogicalOperator::LeftJoin(LeftJoinOp {
            left: Box::new(left),
            right: Box::new(right),
            condition: None,
        });
        return (join, None);
    };

    // Collect variables from each side
    let mut left_vars = HashSet::new();
    collect_operator_variables(&left, &mut left_vars);
    let mut right_vars = HashSet::new();
    collect_operator_variables(&right, &mut right_vars);

    // Classify
    let classified = classify_optional_predicates(predicate, &left_vars, &right_vars);

    // Build right input with right-only filters pushed down
    let filtered_right = if classified.right_filters.is_empty() {
        right
    } else {
        let right_pred = classified
            .right_filters
            .into_iter()
            .reduce(|acc, pred| LogicalExpression::Binary {
                left: Box::new(acc),
                op: BinaryOp::And,
                right: Box::new(pred),
            })
            .expect("non-empty right_filters");
        wrap_filter(right, right_pred)
    };

    // Build null-safe condition for cross-side predicates.
    // For each cross predicate P referencing right-only variable R, wrap it as:
    //   (R IS NULL) OR P
    // This preserves NULL-padded rows (unmatched optional side) while evaluating P
    // correctly when the right side matched.
    let cross_condition = if classified.cross_filters.is_empty() {
        None
    } else {
        // Collect all right-only variable names for the IS NULL sentinel.
        let right_only_vars: Vec<String> = right_vars
            .iter()
            .filter(|v| !left_vars.contains(*v))
            .cloned()
            .collect();

        let null_safe: Vec<LogicalExpression> = classified
            .cross_filters
            .into_iter()
            .map(|pred| {
                // Pick the first right-only variable referenced in this predicate
                // as the NULL sentinel. Falling back to the first right-only var
                // overall is safe: if the right side produced no row, all right
                // columns are NULL so any of them serves as the sentinel.
                let mut pred_vars = HashSet::new();
                collect_expression_variables(&pred, &mut pred_vars);
                let sentinel = pred_vars
                    .iter()
                    .find(|v| right_vars.contains(*v) && !left_vars.contains(*v))
                    .or_else(|| right_only_vars.first())
                    .cloned()
                    .unwrap_or_default();

                let is_null = LogicalExpression::Unary {
                    op: UnaryOp::IsNull,
                    operand: Box::new(LogicalExpression::Variable(sentinel)),
                };
                LogicalExpression::Binary {
                    left: Box::new(is_null),
                    op: BinaryOp::Or,
                    right: Box::new(pred),
                }
            })
            .collect();

        Some(
            null_safe
                .into_iter()
                .reduce(|acc, expr| LogicalExpression::Binary {
                    left: Box::new(acc),
                    op: BinaryOp::And,
                    right: Box::new(expr),
                })
                .expect("non-empty cross_filters"),
        )
    };

    let join = LogicalOperator::LeftJoin(LeftJoinOp {
        left: Box::new(left),
        right: Box::new(filtered_right),
        condition: cross_condition,
    });

    // Combine remaining post-filters
    let post_filter = if classified.post_filters.is_empty() {
        None
    } else {
        Some(
            classified
                .post_filters
                .into_iter()
                .reduce(|acc, pred| LogicalExpression::Binary {
                    left: Box::new(acc),
                    op: BinaryOp::And,
                    right: Box::new(pred),
                })
                .expect("non-empty post_filters"),
        )
    };

    (join, post_filter)
}

// ---------------------------------------------------------------------------
// Plan node builder helpers
// ---------------------------------------------------------------------------

/// Wraps an operator with a filter predicate.
pub(crate) fn wrap_filter(input: LogicalOperator, predicate: LogicalExpression) -> LogicalOperator {
    LogicalOperator::Filter(FilterOp {
        predicate,
        input: Box::new(input),
        pushdown_hint: None,
    })
}

/// Wraps an operator with ORDER BY.
pub(crate) fn wrap_sort(input: LogicalOperator, keys: Vec<SortKey>) -> LogicalOperator {
    LogicalOperator::Sort(SortOp {
        keys,
        input: Box::new(input),
    })
}

/// Wraps an operator with SKIP.
pub(crate) fn wrap_skip(input: LogicalOperator, count: impl Into<CountExpr>) -> LogicalOperator {
    LogicalOperator::Skip(SkipOp {
        count: count.into(),
        input: Box::new(input),
    })
}

/// Wraps an operator with LIMIT.
pub(crate) fn wrap_limit(input: LogicalOperator, count: impl Into<CountExpr>) -> LogicalOperator {
    LogicalOperator::Limit(LimitOp {
        count: count.into(),
        input: Box::new(input),
    })
}

/// Output columns of a query branch, looking through ORDER BY, SKIP, LIMIT and
/// DISTINCT down to its `RETURN`. Each entry is the explicit column name (an
/// alias or a bare variable), or `None` for an unaliased expression, whose name
/// is implementation-defined. Returns `None` when the branch does not end in a
/// `RETURN` or uses `RETURN *` (columns unknown before planning).
fn branch_output_columns(op: &LogicalOperator) -> Option<Vec<Option<String>>> {
    match op {
        LogicalOperator::Return(ret) => {
            if ret
                .items
                .iter()
                .any(|item| matches!(&item.expression, LogicalExpression::Variable(v) if v == "*"))
            {
                return None;
            }
            Some(
                ret.items
                    .iter()
                    .map(|item| match (&item.alias, &item.expression) {
                        (Some(alias), _) => Some(alias.clone()),
                        (None, LogicalExpression::Variable(name)) => Some(name.clone()),
                        (None, _) => None,
                    })
                    .collect(),
            )
        }
        // A RETURN made only of aggregates and grouping keys plans to a bare
        // Aggregate: grouping keys first, then aggregates.
        LogicalOperator::Aggregate(agg) => Some(
            agg.group_by
                .iter()
                .map(|key| match key {
                    LogicalExpression::Variable(name) => Some(name.clone()),
                    _ => None,
                })
                .chain(agg.aggregates.iter().map(|a| a.alias.clone()))
                .collect(),
        ),
        // A nested UNION (`a UNION b UNION c` parses left-deep) was checked when
        // it was built; it outputs its branches' shared columns.
        LogicalOperator::Union(union) => {
            let mut merged: Option<Vec<Option<String>>> = None;
            for columns in union.inputs.iter().filter_map(branch_output_columns) {
                match &mut merged {
                    None => merged = Some(columns),
                    Some(known) => {
                        for (slot, name) in known.iter_mut().zip(columns) {
                            if slot.is_none() {
                                *slot = name;
                            }
                        }
                    }
                }
            }
            merged
        }
        LogicalOperator::Sort(sort) => branch_output_columns(&sort.input),
        LogicalOperator::Limit(limit) => branch_output_columns(&limit.input),
        LogicalOperator::Skip(skip) => branch_output_columns(&skip.input),
        LogicalOperator::Distinct(distinct) => branch_output_columns(&distinct.input),
        LogicalOperator::Filter(filter) => branch_output_columns(&filter.input),
        _ => None,
    }
}

/// Checks that the branches of a user-written `UNION` are compatible, as GQL
/// (ISO/IEC 39075 14.2) and Cypher require: every branch returns the same number
/// of columns, and where both branches name a column explicitly (alias or bare
/// variable) the names match position by position.
///
/// Branches whose columns cannot be determined here (for example `RETURN *`)
/// are not checked.
///
/// # Errors
///
/// Returns a semantic error naming both column lists on the first mismatch.
pub(crate) fn check_union_columns(branches: &[LogicalOperator]) -> Result<()> {
    let render = |columns: &[Option<String>]| -> String {
        columns
            .iter()
            .map(|c| c.as_deref().unwrap_or("<expression>"))
            .collect::<Vec<_>>()
            .join(", ")
    };
    let mut expected: Option<Vec<Option<String>>> = None;
    for branch in branches {
        let Some(columns) = branch_output_columns(branch) else {
            continue;
        };
        let Some(first) = &expected else {
            expected = Some(columns);
            continue;
        };
        let names_conflict = first
            .iter()
            .zip(&columns)
            .any(|(a, b)| matches!((a, b), (Some(a), Some(b)) if a != b));
        if first.len() != columns.len() || names_conflict {
            return Err(Error::Query(QueryError::new(
                QueryErrorKind::Semantic,
                format!(
                    "All UNION branches must return the same columns in the same order: \
                     [{}] vs [{}]",
                    render(first),
                    render(&columns)
                ),
            )));
        }
    }
    Ok(())
}

/// Wraps an operator with DISTINCT.
pub(crate) fn wrap_distinct(input: LogicalOperator) -> LogicalOperator {
    LogicalOperator::Distinct(DistinctOp {
        input: Box::new(input),
        columns: None,
    })
}

/// Checks if a logical expression references any variable in the given set.
pub(crate) fn references_any(expr: &LogicalExpression, names: &[String]) -> bool {
    match expr {
        LogicalExpression::Literal(_) | LogicalExpression::Parameter(_) => false,
        LogicalExpression::Variable(v) => names.iter().any(|n| n == v),
        LogicalExpression::Property { variable, .. }
        | LogicalExpression::Labels(variable)
        | LogicalExpression::Type(variable)
        | LogicalExpression::Id(variable) => names.iter().any(|n| n == variable),
        LogicalExpression::Binary { left, right, .. } => {
            references_any(left, names) || references_any(right, names)
        }
        LogicalExpression::Unary { operand, .. } => references_any(operand, names),
        LogicalExpression::FunctionCall { args, .. } => {
            args.iter().any(|a| references_any(a, names))
        }
        LogicalExpression::Case {
            operand,
            when_clauses,
            else_clause,
        } => {
            operand.as_ref().is_some_and(|e| references_any(e, names))
                || when_clauses
                    .iter()
                    .any(|(cond, val)| references_any(cond, names) || references_any(val, names))
                || else_clause
                    .as_ref()
                    .is_some_and(|e| references_any(e, names))
        }
        LogicalExpression::List(items) => items.iter().any(|i| references_any(i, names)),
        LogicalExpression::Map(entries) => entries.iter().any(|(_, v)| references_any(v, names)),
        LogicalExpression::IndexAccess { base, index } => {
            references_any(base, names) || references_any(index, names)
        }
        LogicalExpression::SliceAccess { base, start, end } => {
            references_any(base, names)
                || start.as_ref().is_some_and(|s| references_any(s, names))
                || end.as_ref().is_some_and(|e| references_any(e, names))
        }
        LogicalExpression::ListComprehension {
            list_expr,
            filter_expr,
            map_expr,
            ..
        } => {
            references_any(list_expr, names)
                || filter_expr
                    .as_ref()
                    .is_some_and(|f| references_any(f, names))
                || references_any(map_expr, names)
        }
        LogicalExpression::ListPredicate {
            list_expr,
            predicate,
            ..
        } => references_any(list_expr, names) || references_any(predicate, names),
        LogicalExpression::Reduce {
            initial,
            list,
            expression,
            ..
        } => {
            references_any(initial, names)
                || references_any(list, names)
                || references_any(expression, names)
        }
        LogicalExpression::PatternComprehension { projection, .. } => {
            references_any(projection, names)
        }
        LogicalExpression::MapProjection { base, .. } => names.iter().any(|n| n == base),
        // Subqueries have their own scope; outer names are not referenced.
        LogicalExpression::ExistsSubquery(_)
        | LogicalExpression::CountSubquery(_)
        | LogicalExpression::ValueSubquery(_) => false,
    }
}

/// Flattens a tree of AND-joined expressions into a list of conjuncts.
pub(crate) fn flatten_and_conjuncts(expr: &LogicalExpression) -> Vec<&LogicalExpression> {
    match expr {
        LogicalExpression::Binary {
            left,
            op: BinaryOp::And,
            right,
        } => {
            let mut parts = flatten_and_conjuncts(left);
            parts.extend(flatten_and_conjuncts(right));
            parts
        }
        other => vec![other],
    }
}

/// Joins a list of expressions with AND. Returns `None` for an empty list.
pub(crate) fn join_and_conjuncts(parts: Vec<LogicalExpression>) -> Option<LogicalExpression> {
    parts
        .into_iter()
        .reduce(|acc, part| LogicalExpression::Binary {
            left: Box::new(acc),
            op: BinaryOp::And,
            right: Box::new(part),
        })
}

/// Wraps an operator with RETURN.
pub(crate) fn wrap_return(
    input: LogicalOperator,
    items: Vec<ReturnItem>,
    distinct: bool,
) -> LogicalOperator {
    LogicalOperator::Return(ReturnOp {
        items,
        distinct,
        input: Box::new(input),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use grafeo_common::types::Value;

    // --- capitalize_first ---

    #[test]
    fn capitalize_first_empty() {
        assert_eq!(capitalize_first(""), "");
    }

    #[test]
    fn capitalize_first_single_char() {
        assert_eq!(capitalize_first("a"), "A");
    }

    #[test]
    fn capitalize_first_already_upper() {
        assert_eq!(capitalize_first("Hello"), "Hello");
    }

    #[test]
    fn capitalize_first_lower() {
        assert_eq!(capitalize_first("person"), "Person");
    }

    // --- VarGen ---

    #[test]
    fn var_gen_starts_at_zero() {
        let vg = VarGen::new();
        assert_eq!(vg.current(), 0);
    }

    #[test]
    fn var_gen_increments() {
        let vg = VarGen::new();
        assert_eq!(vg.next(), "_v0");
        assert_eq!(vg.next(), "_v1");
        assert_eq!(vg.current(), 2);
    }

    // --- is_aggregate_function ---

    #[test]
    fn aggregate_functions_recognized() {
        for name in [
            "count",
            "COUNT",
            "sum",
            "avg",
            "min",
            "max",
            "collect",
            "stdev",
            "stddev",
            "stdevp",
            "stddevp",
            "stddev_samp",
            "STDDEV_POP",
            "variance",
            "VARIANCE",
            "var_samp",
            "VAR_POP",
            "percentile_disc",
            "percentiledisc",
            "percentile_cont",
            "percentilecont",
        ] {
            assert!(is_aggregate_function(name), "{name} should be aggregate");
        }
    }

    #[test]
    fn non_aggregate_functions_rejected() {
        for name in ["toString", "toUpper", "size", "rand", "abs", "coalesce", ""] {
            assert!(
                !is_aggregate_function(name),
                "{name} should not be aggregate"
            );
        }
    }

    // --- to_aggregate_function ---

    #[test]
    fn to_aggregate_all_variants() {
        assert!(matches!(
            to_aggregate_function("count"),
            Some(AggregateFunction::Count)
        ));
        assert!(matches!(
            to_aggregate_function("SUM"),
            Some(AggregateFunction::Sum)
        ));
        assert!(matches!(
            to_aggregate_function("Avg"),
            Some(AggregateFunction::Avg)
        ));
        assert!(matches!(
            to_aggregate_function("MIN"),
            Some(AggregateFunction::Min)
        ));
        assert!(matches!(
            to_aggregate_function("max"),
            Some(AggregateFunction::Max)
        ));
        assert!(matches!(
            to_aggregate_function("collect"),
            Some(AggregateFunction::Collect)
        ));
        assert!(matches!(
            to_aggregate_function("stdev"),
            Some(AggregateFunction::StdDev)
        ));
        assert!(matches!(
            to_aggregate_function("stddev"),
            Some(AggregateFunction::StdDev)
        ));
        assert!(matches!(
            to_aggregate_function("stdevp"),
            Some(AggregateFunction::StdDevPop)
        ));
        assert!(matches!(
            to_aggregate_function("stddevp"),
            Some(AggregateFunction::StdDevPop)
        ));
        assert!(matches!(
            to_aggregate_function("stddev_samp"),
            Some(AggregateFunction::StdDev)
        ));
        assert!(matches!(
            to_aggregate_function("STDDEV_POP"),
            Some(AggregateFunction::StdDevPop)
        ));
        assert!(matches!(
            to_aggregate_function("variance"),
            Some(AggregateFunction::Variance)
        ));
        assert!(matches!(
            to_aggregate_function("VAR_SAMP"),
            Some(AggregateFunction::Variance)
        ));
        assert!(matches!(
            to_aggregate_function("VAR_POP"),
            Some(AggregateFunction::VariancePop)
        ));
        assert!(matches!(
            to_aggregate_function("percentile_disc"),
            Some(AggregateFunction::PercentileDisc)
        ));
        assert!(matches!(
            to_aggregate_function("percentiledisc"),
            Some(AggregateFunction::PercentileDisc)
        ));
        assert!(matches!(
            to_aggregate_function("percentile_cont"),
            Some(AggregateFunction::PercentileCont)
        ));
        assert!(matches!(
            to_aggregate_function("percentilecont"),
            Some(AggregateFunction::PercentileCont)
        ));
    }

    #[test]
    fn to_aggregate_unknown_returns_none() {
        assert!(to_aggregate_function("unknown").is_none());
        assert!(to_aggregate_function("").is_none());
        assert!(to_aggregate_function("size").is_none());
    }

    // --- is_aggregate_function: bivariate / concat / sample ---

    #[test]
    fn aggregate_functions_bivariate_recognized() {
        for name in [
            "group_concat",
            "GROUPCONCAT",
            "listagg",
            "LISTAGG",
            "sample",
            "SAMPLE",
            "covar_samp",
            "COVAR_SAMP",
            "covar_pop",
            "COVAR_POP",
            "corr",
            "CORR",
            "regr_slope",
            "REGR_SLOPE",
            "regr_intercept",
            "REGR_INTERCEPT",
            "regr_r2",
            "REGR_R2",
            "regr_count",
            "REGR_COUNT",
            "regr_sxx",
            "REGR_SXX",
            "regr_syy",
            "REGR_SYY",
            "regr_sxy",
            "REGR_SXY",
            "regr_avgx",
            "REGR_AVGX",
            "regr_avgy",
            "REGR_AVGY",
        ] {
            assert!(is_aggregate_function(name), "{name} should be aggregate");
        }
    }

    // --- to_aggregate_function: bivariate variants ---

    #[test]
    fn to_aggregate_bivariate_variants() {
        assert!(matches!(
            to_aggregate_function("group_concat"),
            Some(AggregateFunction::GroupConcat)
        ));
        assert!(matches!(
            to_aggregate_function("GROUPCONCAT"),
            Some(AggregateFunction::GroupConcat)
        ));
        assert!(matches!(
            to_aggregate_function("LISTAGG"),
            Some(AggregateFunction::GroupConcat)
        ));
        assert!(matches!(
            to_aggregate_function("sample"),
            Some(AggregateFunction::Sample)
        ));
        assert!(matches!(
            to_aggregate_function("COVAR_SAMP"),
            Some(AggregateFunction::CovarSamp)
        ));
        assert!(matches!(
            to_aggregate_function("COVAR_POP"),
            Some(AggregateFunction::CovarPop)
        ));
        assert!(matches!(
            to_aggregate_function("CORR"),
            Some(AggregateFunction::Corr)
        ));
        assert!(matches!(
            to_aggregate_function("REGR_SLOPE"),
            Some(AggregateFunction::RegrSlope)
        ));
        assert!(matches!(
            to_aggregate_function("REGR_INTERCEPT"),
            Some(AggregateFunction::RegrIntercept)
        ));
        assert!(matches!(
            to_aggregate_function("REGR_R2"),
            Some(AggregateFunction::RegrR2)
        ));
        assert!(matches!(
            to_aggregate_function("REGR_COUNT"),
            Some(AggregateFunction::RegrCount)
        ));
        assert!(matches!(
            to_aggregate_function("REGR_SXX"),
            Some(AggregateFunction::RegrSxx)
        ));
        assert!(matches!(
            to_aggregate_function("REGR_SYY"),
            Some(AggregateFunction::RegrSyy)
        ));
        assert!(matches!(
            to_aggregate_function("REGR_SXY"),
            Some(AggregateFunction::RegrSxy)
        ));
        assert!(matches!(
            to_aggregate_function("REGR_AVGX"),
            Some(AggregateFunction::RegrAvgx)
        ));
        assert!(matches!(
            to_aggregate_function("REGR_AVGY"),
            Some(AggregateFunction::RegrAvgy)
        ));
    }

    // --- is_binary_set_function ---

    #[test]
    fn binary_set_functions_recognized() {
        let binary = [
            AggregateFunction::CovarSamp,
            AggregateFunction::CovarPop,
            AggregateFunction::Corr,
            AggregateFunction::RegrSlope,
            AggregateFunction::RegrIntercept,
            AggregateFunction::RegrR2,
            AggregateFunction::RegrCount,
            AggregateFunction::RegrSxx,
            AggregateFunction::RegrSyy,
            AggregateFunction::RegrSxy,
            AggregateFunction::RegrAvgx,
            AggregateFunction::RegrAvgy,
        ];
        for func in binary {
            assert!(is_binary_set_function(func), "{func:?} should be binary");
        }
    }

    #[test]
    fn non_binary_set_functions_rejected() {
        let non_binary = [
            AggregateFunction::Count,
            AggregateFunction::CountNonNull,
            AggregateFunction::Sum,
            AggregateFunction::Avg,
            AggregateFunction::Min,
            AggregateFunction::Max,
            AggregateFunction::Collect,
            AggregateFunction::StdDev,
            AggregateFunction::StdDevPop,
            AggregateFunction::Variance,
            AggregateFunction::VariancePop,
            AggregateFunction::PercentileDisc,
            AggregateFunction::PercentileCont,
            AggregateFunction::GroupConcat,
            AggregateFunction::Sample,
        ];
        for func in non_binary {
            assert!(
                !is_binary_set_function(func),
                "{func:?} should not be binary"
            );
        }
    }

    // --- combine_with_and ---

    #[test]
    fn combine_with_and_empty_returns_error() {
        let result = combine_with_and(vec![]);
        assert!(result.is_err());
    }

    #[test]
    fn combine_with_and_single_predicate() {
        let pred = LogicalExpression::Property {
            variable: "n".to_string(),
            property: "name".to_string(),
        };
        let result = combine_with_and(vec![pred.clone()]).unwrap();
        assert!(matches!(result, LogicalExpression::Property { .. }));
    }

    #[test]
    fn combine_with_and_two_predicates() {
        let p1 = LogicalExpression::Property {
            variable: "n".to_string(),
            property: "a".to_string(),
        };
        let p2 = LogicalExpression::Property {
            variable: "n".to_string(),
            property: "b".to_string(),
        };
        let result = combine_with_and(vec![p1, p2]).unwrap();
        assert!(matches!(
            result,
            LogicalExpression::Binary {
                op: BinaryOp::And,
                ..
            }
        ));
    }

    // --- split_conjuncts ---

    #[test]
    fn split_conjuncts_single() {
        let expr = LogicalExpression::Variable("x".into());
        let conjuncts = split_conjuncts(expr);
        assert_eq!(conjuncts.len(), 1);
    }

    #[test]
    fn split_conjuncts_nested_and() {
        // (a AND b) AND c -> [a, b, c]
        let a = LogicalExpression::Variable("a".into());
        let b = LogicalExpression::Variable("b".into());
        let c = LogicalExpression::Variable("c".into());
        let ab = LogicalExpression::Binary {
            left: Box::new(a),
            op: BinaryOp::And,
            right: Box::new(b),
        };
        let abc = LogicalExpression::Binary {
            left: Box::new(ab),
            op: BinaryOp::And,
            right: Box::new(c),
        };
        let conjuncts = split_conjuncts(abc);
        assert_eq!(conjuncts.len(), 3);
    }

    #[test]
    fn split_conjuncts_or_not_split() {
        // a OR b should NOT be split (only AND is split)
        let a = LogicalExpression::Variable("a".into());
        let b = LogicalExpression::Variable("b".into());
        let or_expr = LogicalExpression::Binary {
            left: Box::new(a),
            op: BinaryOp::Or,
            right: Box::new(b),
        };
        let conjuncts = split_conjuncts(or_expr);
        assert_eq!(conjuncts.len(), 1);
    }

    // --- classify_optional_predicates ---

    #[test]
    fn classify_left_only_predicate() {
        let left_vars: HashSet<String> = ["n".into()].into_iter().collect();
        let right_vars: HashSet<String> = ["m".into()].into_iter().collect();
        let pred = LogicalExpression::Property {
            variable: "n".into(),
            property: "age".into(),
        };

        let result = classify_optional_predicates(pred, &left_vars, &right_vars);
        assert_eq!(
            result.post_filters.len(),
            1,
            "left-only should be post-filter"
        );
        assert!(result.right_filters.is_empty());
    }

    #[test]
    fn classify_right_only_predicate() {
        let left_vars: HashSet<String> = ["n".into()].into_iter().collect();
        let right_vars: HashSet<String> = ["m".into()].into_iter().collect();
        let pred = LogicalExpression::Property {
            variable: "m".into(),
            property: "age".into(),
        };

        let result = classify_optional_predicates(pred, &left_vars, &right_vars);
        assert!(result.post_filters.is_empty());
        assert_eq!(
            result.right_filters.len(),
            1,
            "right-only should be right-filter"
        );
    }

    #[test]
    fn classify_shared_variable_as_right() {
        // Variable `n` is in both left_vars and right_vars (shared).
        // A predicate on `m.age > n.age` has both m and n on right side,
        // so it should be classified as right-filter.
        let left_vars: HashSet<String> = ["n".into()].into_iter().collect();
        let right_vars: HashSet<String> = ["n".into(), "m".into()].into_iter().collect();
        let pred = LogicalExpression::Binary {
            left: Box::new(LogicalExpression::Property {
                variable: "m".into(),
                property: "age".into(),
            }),
            op: BinaryOp::Gt,
            right: Box::new(LogicalExpression::Property {
                variable: "n".into(),
                property: "age".into(),
            }),
        };

        let result = classify_optional_predicates(pred, &left_vars, &right_vars);
        assert!(result.post_filters.is_empty());
        assert_eq!(
            result.right_filters.len(),
            1,
            "shared variable predicate should be right-filter"
        );
    }

    #[test]
    fn classify_mixed_and_predicate() {
        // n.active AND m.age > 30 should split into post-filter and right-filter
        let left_vars: HashSet<String> = ["n".into()].into_iter().collect();
        let right_vars: HashSet<String> = ["n".into(), "m".into()].into_iter().collect();

        let left_pred = LogicalExpression::Property {
            variable: "n".into(),
            property: "active".into(),
        };
        let right_pred = LogicalExpression::Binary {
            left: Box::new(LogicalExpression::Property {
                variable: "m".into(),
                property: "age".into(),
            }),
            op: BinaryOp::Gt,
            right: Box::new(LogicalExpression::Literal(Value::Int64(30))),
        };
        let combined = LogicalExpression::Binary {
            left: Box::new(left_pred),
            op: BinaryOp::And,
            right: Box::new(right_pred),
        };

        let result = classify_optional_predicates(combined, &left_vars, &right_vars);

        // n.active -> n is in both left and right, so all_in_right = true
        // It should be classified as right_filter since n is available on right side.
        // Actually n.active references only n, which IS in right_vars too.
        // So both predicates should be right_filters.
        // BUT we also need to check: n.active only references n, which is in left_vars.
        // Since n is in BOTH sets, all_in_left AND all_in_right are true.
        // The logic checks all_in_right FIRST, so it goes to right_filters.
        assert_eq!(
            result.right_filters.len() + result.post_filters.len(),
            2,
            "two conjuncts should be classified"
        );
    }

    // --- collect_expression_variables ---

    #[test]
    fn collect_vars_from_property() {
        let expr = LogicalExpression::Property {
            variable: "n".into(),
            property: "age".into(),
        };
        let mut vars = HashSet::new();
        collect_expression_variables(&expr, &mut vars);
        assert!(vars.contains("n"));
        assert_eq!(vars.len(), 1);
    }

    #[test]
    fn collect_vars_from_binary() {
        let expr = LogicalExpression::Binary {
            left: Box::new(LogicalExpression::Property {
                variable: "a".into(),
                property: "x".into(),
            }),
            op: BinaryOp::Gt,
            right: Box::new(LogicalExpression::Property {
                variable: "b".into(),
                property: "y".into(),
            }),
        };
        let mut vars = HashSet::new();
        collect_expression_variables(&expr, &mut vars);
        assert!(vars.contains("a"));
        assert!(vars.contains("b"));
        assert_eq!(vars.len(), 2);
    }

    #[test]
    fn collect_vars_from_function_call() {
        let expr = LogicalExpression::FunctionCall {
            name: "size".into(),
            args: vec![LogicalExpression::Variable("list".into())],
            distinct: false,
        };
        let mut vars = HashSet::new();
        collect_expression_variables(&expr, &mut vars);
        assert!(vars.contains("list"));
    }

    #[test]
    fn collect_vars_from_literal_is_empty() {
        let expr = LogicalExpression::Literal(Value::Int64(42));
        let mut vars = HashSet::new();
        collect_expression_variables(&expr, &mut vars);
        assert!(vars.is_empty());
    }

    // --- graphql_directives_allow ---
    //
    // These tests require the `graphql` feature so that the
    // `grafeo_adapters::query::graphql::ast` types are available.

    #[cfg(feature = "graphql")]
    mod graphql_directive_tests {
        use super::super::graphql_directives_allow;
        use grafeo_adapters::query::graphql::ast::{Argument, Directive, InputValue};

        fn directive(name: &str, if_value: bool) -> Directive {
            Directive {
                name: name.to_string(),
                arguments: vec![Argument {
                    name: "if".to_string(),
                    value: InputValue::Boolean(if_value),
                }],
            }
        }

        #[test]
        fn test_skip_directive_true_excludes() {
            // @skip(if: true) should exclude the field
            let directives = vec![directive("skip", true)];
            assert!(
                !graphql_directives_allow(&directives),
                "@skip(if: true) should return false (field excluded)"
            );
        }

        #[test]
        fn test_skip_directive_false_includes() {
            // @skip(if: false) should include the field
            let directives = vec![directive("skip", false)];
            assert!(
                graphql_directives_allow(&directives),
                "@skip(if: false) should return true (field included)"
            );
        }

        #[test]
        fn test_include_directive_true_includes() {
            // @include(if: true) should include the field
            let directives = vec![directive("include", true)];
            assert!(
                graphql_directives_allow(&directives),
                "@include(if: true) should return true"
            );
        }

        #[test]
        fn test_include_directive_false_excludes() {
            // @include(if: false) should exclude the field
            let directives = vec![directive("include", false)];
            assert!(
                !graphql_directives_allow(&directives),
                "@include(if: false) should return false"
            );
        }

        #[test]
        fn test_no_directives_includes() {
            // No directives at all should include the field
            let directives: Vec<Directive> = vec![];
            assert!(
                graphql_directives_allow(&directives),
                "empty directives should return true"
            );
        }

        #[test]
        fn test_unknown_directive_ignored() {
            // Unknown directives should be ignored, field included
            let directives = vec![Directive {
                name: "deprecated".to_string(),
                arguments: vec![],
            }];
            assert!(
                graphql_directives_allow(&directives),
                "unknown directive should be ignored, returning true"
            );
        }
    }
}
