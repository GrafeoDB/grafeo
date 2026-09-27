//! Shared planning utilities for LPG and RDF planners.
//!
//! These free functions build physical operators from pre-planned children,
//! eliminating duplication between `Planner` (LPG) and `RdfPlanner`.
//! Each function takes already-planned input operators and column lists,
//! plus a schema derivation function to handle LPG vs RDF type differences.

use crate::query::plan::{
    BinaryOp, ListPredicateKind, LogicalExpression, MapProjectionEntry, UnaryOp,
};
use grafeo_common::types::{LogicalType, Value};
use grafeo_common::utils::error::{Error, Result};
use grafeo_core::execution::operators::{
    DistinctOperator, ExceptOperator, HashJoinOperator, IntersectOperator,
    JoinType as PhysicalJoinType, LimitOperator, Operator, OtherwiseOperator, ProjectExpr,
    ProjectOperator, SkipOperator, UnionOperator,
};

/// Builds a LIMIT physical operator.
pub(crate) fn build_limit(
    input: Box<dyn Operator>,
    columns: Vec<String>,
    count: usize,
    schema: Vec<LogicalType>,
) -> (Box<dyn Operator>, Vec<String>) {
    let operator = Box::new(LimitOperator::new(input, count, schema));
    (operator, columns)
}

/// Builds a SKIP physical operator.
pub(crate) fn build_skip(
    input: Box<dyn Operator>,
    columns: Vec<String>,
    count: usize,
    schema: Vec<LogicalType>,
) -> (Box<dyn Operator>, Vec<String>) {
    let operator = Box::new(SkipOperator::new(input, count, schema));
    (operator, columns)
}

/// Builds a DISTINCT physical operator.
///
/// Handles both full-row dedup and column-specific dedup (when `distinct.columns` is set).
pub(crate) fn build_distinct(
    input: Box<dyn Operator>,
    columns: Vec<String>,
    distinct_columns: Option<&[String]>,
    schema: Vec<LogicalType>,
) -> (Box<dyn Operator>, Vec<String>) {
    let operator: Box<dyn Operator> = if let Some(dist_cols) = distinct_columns {
        let col_indices: Vec<usize> = dist_cols
            .iter()
            .filter_map(|name| columns.iter().position(|c| c == name))
            .collect();
        if col_indices.is_empty() {
            Box::new(DistinctOperator::new(input, schema))
        } else {
            Box::new(DistinctOperator::on_columns(input, col_indices, schema))
        }
    } else {
        Box::new(DistinctOperator::new(input, schema))
    };
    (operator, columns)
}

/// Builds a UNION physical operator from multiple pre-planned inputs.
pub(crate) fn build_union(
    inputs: Vec<Box<dyn Operator>>,
    columns: Vec<String>,
    schema: Vec<LogicalType>,
) -> Result<(Box<dyn Operator>, Vec<String>)> {
    if inputs.is_empty() {
        return Err(Error::Internal(
            "Union requires at least one input".to_string(),
        ));
    }
    let operator = Box::new(UnionOperator::new(inputs, schema));
    Ok((operator, columns))
}

/// Builds an EXCEPT physical operator.
pub(crate) fn build_except(
    left: Box<dyn Operator>,
    right: Box<dyn Operator>,
    columns: Vec<String>,
    all: bool,
    schema: Vec<LogicalType>,
) -> (Box<dyn Operator>, Vec<String>) {
    let operator = Box::new(ExceptOperator::new(left, right, all, schema));
    (operator, columns)
}

/// Builds an INTERSECT physical operator.
pub(crate) fn build_intersect(
    left: Box<dyn Operator>,
    right: Box<dyn Operator>,
    columns: Vec<String>,
    all: bool,
    schema: Vec<LogicalType>,
) -> (Box<dyn Operator>, Vec<String>) {
    let operator = Box::new(IntersectOperator::new(left, right, all, schema));
    (operator, columns)
}

/// Builds an OTHERWISE physical operator.
pub(crate) fn build_otherwise(
    left: Box<dyn Operator>,
    right: Box<dyn Operator>,
    columns: Vec<String>,
) -> (Box<dyn Operator>, Vec<String>) {
    let operator = Box::new(OtherwiseOperator::new(left, right));
    (operator, columns)
}

/// Builds an INNER JOIN physical operator.
///
/// Finds shared variables between left and right column lists for join keys,
/// then creates a hash join with inner semantics. Deduplicates shared columns
/// by projecting away right-side columns that already appear on the left.
/// Falls back to cross join when no shared variables exist.
///
/// When `cardinalities` is provided as `(left_card, right_card)`, the smaller
/// side is placed as the build side for better memory and cache performance.
#[cfg(feature = "triple-store")]
pub(crate) fn build_inner_join(
    left: Box<dyn Operator>,
    right: Box<dyn Operator>,
    left_columns: &[String],
    right_columns: &[String],
    left_types: &[LogicalType],
    right_types: &[LogicalType],
    cardinalities: Option<(f64, f64)>,
) -> (Box<dyn Operator>, Vec<String>, Vec<LogicalType>) {
    let (probe_keys, build_keys) = find_shared_join_keys(left_columns, right_columns);

    let join_type = if probe_keys.is_empty() {
        PhysicalJoinType::Cross
    } else {
        PhysicalJoinType::Inner
    };

    // Decide whether to swap sides: build on the smaller input.
    // Only swap for equi-joins (not cross joins) and when left is significantly larger.
    let swap_sides = matches!(join_type, PhysicalJoinType::Inner)
        && cardinalities.is_some_and(|(left_card, right_card)| right_card < left_card * 0.8);

    if swap_sides {
        // Swap: right becomes probe, left becomes build
        // Output order is right+left from the join, then we project back to left+right
        let mut join_columns: Vec<String> = right_columns.to_vec();
        join_columns.extend(left_columns.iter().cloned());
        let mut join_schema: Vec<LogicalType> = right_types.to_vec();
        join_schema.extend(left_types.iter().cloned());

        let join_op: Box<dyn Operator> = Box::new(HashJoinOperator::new(
            right,      // probe (larger)
            left,       // build (smaller, materialized)
            build_keys, // swapped: right keys become probe keys
            probe_keys, // swapped: left keys become build keys
            join_type,
            join_schema.clone(),
        ));

        // Remap to logical left+right order and deduplicate shared columns
        let right_count = right_columns.len();
        let left_set: std::collections::HashSet<&str> =
            left_columns.iter().map(String::as_str).collect();

        // Build projection: first map left columns (which are at offset right_count in physical output)
        let mut proj_indices: Vec<usize> =
            (0..left_columns.len()).map(|i| right_count + i).collect();
        let mut output_columns: Vec<String> = left_columns.to_vec();
        // Then add right columns not already in left
        for (right_idx, right_col) in right_columns.iter().enumerate() {
            if !left_set.contains(right_col.as_str()) {
                proj_indices.push(right_idx);
                output_columns.push(right_col.clone());
            }
        }

        let proj_exprs: Vec<ProjectExpr> = proj_indices
            .iter()
            .map(|&i| ProjectExpr::Column(i))
            .collect();
        let proj_types: Vec<LogicalType> = proj_indices
            .iter()
            .map(|&i| join_schema[i].clone())
            .collect();
        let output_types = proj_types.clone();
        let operator = Box::new(ProjectOperator::new(join_op, proj_exprs, proj_types));
        (operator, output_columns, output_types)
    } else {
        // Normal order: left = probe, right = build
        let mut join_columns: Vec<String> = left_columns.to_vec();
        join_columns.extend(right_columns.iter().cloned());
        let mut join_schema: Vec<LogicalType> = left_types.to_vec();
        join_schema.extend(right_types.iter().cloned());

        let join_op: Box<dyn Operator> = Box::new(HashJoinOperator::new(
            left,
            right,
            probe_keys,
            build_keys,
            join_type,
            join_schema.clone(),
        ));

        // Deduplicate: keep left columns, then only right columns not already on the left
        let left_set: std::collections::HashSet<&str> =
            left_columns.iter().map(String::as_str).collect();
        let mut keep_indices: Vec<usize> = (0..left_columns.len()).collect();
        let mut output_columns: Vec<String> = left_columns.to_vec();
        for (right_idx, right_col) in right_columns.iter().enumerate() {
            if !left_set.contains(right_col.as_str()) {
                keep_indices.push(left_columns.len() + right_idx);
                output_columns.push(right_col.clone());
            }
        }

        // If there are duplicates, add a ProjectOperator to strip them
        if keep_indices.len() < join_columns.len() {
            let proj_exprs: Vec<ProjectExpr> = keep_indices
                .iter()
                .map(|&i| ProjectExpr::Column(i))
                .collect();
            let proj_types: Vec<LogicalType> = keep_indices
                .iter()
                .map(|&i| join_schema[i].clone())
                .collect();
            let output_types = proj_types.clone();
            let operator = Box::new(ProjectOperator::new(join_op, proj_exprs, proj_types));
            (operator, output_columns, output_types)
        } else {
            (join_op, output_columns, join_schema)
        }
    }
}

/// Builds an ANTI JOIN physical operator.
///
/// Finds shared variables between left and right column lists for join keys,
/// then creates a hash join with anti semantics (only left rows with no match).
///
/// Per the SPARQL 1.1 spec, MINUS with no shared variables between left and
/// right is a no-op: two solutions are compatible only if they agree on all
/// shared variables, so when there are none, no solutions are compatible and
/// nothing is removed.
pub(crate) fn build_anti_join(
    left: Box<dyn Operator>,
    right: Box<dyn Operator>,
    left_columns: Vec<String>,
    right_columns: &[String],
    schema: Vec<LogicalType>,
) -> (Box<dyn Operator>, Vec<String>) {
    let (probe_keys, build_keys) = find_shared_join_keys(&left_columns, right_columns);

    // No shared variables: MINUS is a no-op (keep all left rows).
    if probe_keys.is_empty() {
        return (left, left_columns);
    }

    let operator: Box<dyn Operator> = Box::new(HashJoinOperator::new(
        left,
        right,
        probe_keys,
        build_keys,
        PhysicalJoinType::Anti,
        schema,
    ));
    (operator, left_columns)
}

/// Builds a SEMI JOIN physical operator.
///
/// Finds shared variables between left and right column lists for join keys,
/// then creates a hash join with semi semantics (only left rows with a match).
#[cfg(feature = "triple-store")]
pub(crate) fn build_semi_join(
    left: Box<dyn Operator>,
    right: Box<dyn Operator>,
    left_columns: Vec<String>,
    right_columns: &[String],
    schema: Vec<LogicalType>,
) -> (Box<dyn Operator>, Vec<String>) {
    let (probe_keys, build_keys) = find_shared_join_keys(&left_columns, right_columns);

    let operator: Box<dyn Operator> = Box::new(HashJoinOperator::new(
        left,
        right,
        probe_keys,
        build_keys,
        PhysicalJoinType::Semi,
        schema,
    ));
    (operator, left_columns)
}

/// Builds a LEFT JOIN physical operator.
///
/// Joins left and right sides, deduplicates shared columns by projecting away
/// right-side columns that already appear on the left.
pub(crate) fn build_left_join(
    left: Box<dyn Operator>,
    right: Box<dyn Operator>,
    left_columns: &[String],
    right_columns: &[String],
    left_types: &[LogicalType],
    right_types: &[LogicalType],
) -> (Box<dyn Operator>, Vec<String>, Vec<LogicalType>) {
    let (probe_keys, build_keys) = find_shared_join_keys(left_columns, right_columns);

    // Full join outputs all left + all right columns
    let mut join_columns: Vec<String> = left_columns.to_vec();
    join_columns.extend(right_columns.iter().cloned());
    let mut join_schema: Vec<LogicalType> = left_types.to_vec();
    join_schema.extend(right_types.iter().cloned());

    let join_op: Box<dyn Operator> = Box::new(HashJoinOperator::new(
        left,
        right,
        probe_keys,
        build_keys,
        PhysicalJoinType::Left,
        join_schema.clone(),
    ));

    // Deduplicate: keep left columns, then only right columns not already on the left
    let left_set: std::collections::HashSet<&str> =
        left_columns.iter().map(String::as_str).collect();
    let mut keep_indices: Vec<usize> = (0..left_columns.len()).collect();
    let mut output_columns: Vec<String> = left_columns.to_vec();
    for (right_idx, right_col) in right_columns.iter().enumerate() {
        if !left_set.contains(right_col.as_str()) {
            keep_indices.push(left_columns.len() + right_idx);
            output_columns.push(right_col.clone());
        }
    }

    // If there are duplicates, add a ProjectOperator to strip them
    if keep_indices.len() < join_columns.len() {
        let proj_exprs: Vec<ProjectExpr> = keep_indices
            .iter()
            .map(|&i| ProjectExpr::Column(i))
            .collect();
        let proj_types: Vec<LogicalType> = keep_indices
            .iter()
            .map(|&i| join_schema[i].clone())
            .collect();
        let output_types = proj_types.clone();
        let operator = Box::new(ProjectOperator::new(join_op, proj_exprs, proj_types));
        (operator, output_columns, output_types)
    } else {
        (join_op, output_columns, join_schema)
    }
}

/// Finds shared variable names between two column lists and returns
/// `(left_indices, right_indices)` for use as join keys.
fn find_shared_join_keys(left: &[String], right: &[String]) -> (Vec<usize>, Vec<usize>) {
    let mut probe_keys = Vec::new();
    let mut build_keys = Vec::new();
    for (right_idx, right_col) in right.iter().enumerate() {
        if let Some(left_idx) = left.iter().position(|c| c == right_col) {
            probe_keys.push(left_idx);
            build_keys.push(right_idx);
        }
    }
    (probe_keys, build_keys)
}

/// Column name that `resolve_expression_to_column` looks up for `expr`.
///
/// - `Variable(name)`: `"name"`
/// - `Property { variable, property }`: `"{variable}_{property}"` (LPG projections)
/// - anything else: `"__expr_{expr:?}"`
///
/// Planners that inject synthetic columns (aggregate and sort augmenting
/// projections, LPG and RDF) must name them through this function so the
/// resolver finds them.
pub(crate) fn resolved_column_name(expr: &LogicalExpression) -> String {
    match expr {
        LogicalExpression::Variable(name) => name.clone(),
        LogicalExpression::Property { variable, property } => {
            format!("{variable}_{property}")
        }
        _ => format!("__expr_{expr:?}"),
    }
}

/// Output column name for a Return/Project item: `alias` if set, otherwise
/// `expression_to_string(expr)`. Shared by the LPG and RDF planners and the
/// top-K rewrite's column prediction, which must never disagree.
pub(crate) fn output_column_name(alias: Option<&str>, expr: &LogicalExpression) -> String {
    alias.map_or_else(|| expression_to_string(expr), str::to_string)
}

/// Resolves a logical expression to a column index in the given variable-column map.
///
/// Mirrors [`resolved_column_name`]: the column name looked up is whatever that
/// function returns for `expr`. `context` is appended to error messages
/// (e.g. `" for ORDER BY"`, or `""` for aggregations).
///
/// NOTE: The expression *collection* loops (which build the synthetic columns that this
/// function resolves) are intentionally NOT shared, because the LPG and RDF planners use
/// different `convert_expression` signatures (method on `&self` vs free function).
pub(crate) fn resolve_expression_to_column(
    expr: &LogicalExpression,
    variable_columns: &std::collections::HashMap<String, usize>,
    context: &str,
) -> Result<usize> {
    let col_name = resolved_column_name(expr);
    variable_columns
        .get(&col_name)
        .copied()
        .ok_or_else(|| match expr {
            LogicalExpression::Variable(name) => {
                Error::Internal(format!("Variable '{name}' not found{context}"))
            }
            LogicalExpression::Property { variable, property } => Error::Internal(format!(
                "Property column '{col_name}' not found{context} (from {variable}.{property})"
            )),
            _ => Error::Internal(format!(
                "Cannot resolve expression to column{context}: {expr:?}"
            )),
        })
}

/// Converts a logical expression to a human-readable string for column naming.
///
/// Used when a `RETURN` item has no alias. The text follows the source syntax
/// so structurally different expressions get different names (`id(a)` and
/// `id(b)`, `n.a + n.b` and `n.c + n.d`, `1` and `1.0`). Subqueries cannot be
/// rendered from the plan and get a short label (`EXISTS {...}`); two of those
/// unaliased in one `RETURN` collide and are rejected by `QueryResult`, so
/// alias them.
pub(crate) fn expression_to_string(expr: &LogicalExpression) -> String {
    match expr {
        LogicalExpression::Variable(name) => name.clone(),
        LogicalExpression::Property { variable, property } => {
            format!("{variable}.{property}")
        }
        LogicalExpression::Literal(value) => literal_to_string(value),
        LogicalExpression::Parameter(name) => format!("${name}"),
        LogicalExpression::FunctionCall {
            name,
            args,
            distinct,
        } => {
            let rendered = join_expressions(args);
            if *distinct {
                format!("{name}(DISTINCT {rendered})")
            } else {
                format!("{name}({rendered})")
            }
        }
        LogicalExpression::IndexAccess { base, index } => {
            format!(
                "{}[{}]",
                expression_to_string(base),
                expression_to_string(index)
            )
        }
        LogicalExpression::SliceAccess { base, start, end } => {
            let start = start
                .as_deref()
                .map(expression_to_string)
                .unwrap_or_default();
            let end = end.as_deref().map(expression_to_string).unwrap_or_default();
            format!("{}[{start}..{end}]", expression_to_string(base))
        }
        LogicalExpression::Binary { left, op, right } => format!(
            "{} {} {}",
            operand_to_string(left),
            binary_op_symbol(*op),
            operand_to_string(right)
        ),
        LogicalExpression::Unary { op, operand } => {
            let inner = operand_to_string(operand);
            match op {
                UnaryOp::Not => format!("NOT {inner}"),
                UnaryOp::Neg => format!("-{inner}"),
                UnaryOp::IsNull => format!("{inner} IS NULL"),
                UnaryOp::IsNotNull => format!("{inner} IS NOT NULL"),
            }
        }
        LogicalExpression::Labels(variable) => format!("labels({variable})"),
        LogicalExpression::Type(variable) => format!("type({variable})"),
        LogicalExpression::Id(variable) => format!("id({variable})"),
        LogicalExpression::List(items) => format!("[{}]", join_expressions(items)),
        LogicalExpression::Map(entries) => {
            let inner = entries
                .iter()
                .map(|(key, value)| format!("{key}: {}", expression_to_string(value)))
                .collect::<Vec<_>>()
                .join(", ");
            format!("{{{inner}}}")
        }
        LogicalExpression::MapProjection { base, entries } => {
            let inner = entries
                .iter()
                .map(|entry| match entry {
                    MapProjectionEntry::PropertySelector(property) => format!(".{property}"),
                    MapProjectionEntry::LiteralEntry(key, value) => {
                        format!("{key}: {}", expression_to_string(value))
                    }
                    MapProjectionEntry::AllProperties => ".*".to_string(),
                })
                .collect::<Vec<_>>()
                .join(", ");
            format!("{base}{{{inner}}}")
        }
        LogicalExpression::Case {
            operand,
            when_clauses,
            else_clause,
        } => {
            let mut out = String::from("CASE");
            if let Some(operand) = operand {
                out.push(' ');
                out.push_str(&expression_to_string(operand));
            }
            for (when, then) in when_clauses {
                out.push_str(" WHEN ");
                out.push_str(&expression_to_string(when));
                out.push_str(" THEN ");
                out.push_str(&expression_to_string(then));
            }
            if let Some(else_clause) = else_clause {
                out.push_str(" ELSE ");
                out.push_str(&expression_to_string(else_clause));
            }
            out.push_str(" END");
            out
        }
        LogicalExpression::ExistsSubquery(_) => "EXISTS {...}".to_string(),
        LogicalExpression::CountSubquery(_) => "COUNT {...}".to_string(),
        LogicalExpression::ValueSubquery(_) => "VALUE {...}".to_string(),
        LogicalExpression::Reduce {
            accumulator,
            initial,
            variable,
            list,
            expression,
        } => format!(
            "reduce({accumulator} = {}, {variable} IN {} | {})",
            expression_to_string(initial),
            expression_to_string(list),
            expression_to_string(expression)
        ),
        LogicalExpression::ListComprehension {
            variable,
            list_expr,
            filter_expr,
            map_expr,
        } => {
            let mut out = format!("[{variable} IN {}", expression_to_string(list_expr));
            if let Some(filter) = filter_expr {
                out.push_str(" WHERE ");
                out.push_str(&expression_to_string(filter));
            }
            // `[x IN list WHERE p]` projects the iteration variable itself.
            if !matches!(map_expr.as_ref(), LogicalExpression::Variable(v) if v == variable) {
                out.push_str(" | ");
                out.push_str(&expression_to_string(map_expr));
            }
            out.push(']');
            out
        }
        LogicalExpression::ListPredicate {
            kind,
            variable,
            list_expr,
            predicate,
        } => {
            let function = match kind {
                ListPredicateKind::All => "all",
                ListPredicateKind::Any => "any",
                ListPredicateKind::None => "none",
                ListPredicateKind::Single => "single",
            };
            format!(
                "{function}({variable} IN {} WHERE {})",
                expression_to_string(list_expr),
                expression_to_string(predicate)
            )
        }
        LogicalExpression::PatternComprehension { projection, .. } => {
            format!("[(...) | {}]", expression_to_string(projection))
        }
    }
}

/// Renders a literal as it would be written in a query: `1`, `1.0`, `'x'`,
/// `true`, `NULL`, `[1, 2]`, `{a: 1}`. Floats keep their decimal point so `1`
/// and `1.0` stay distinct; strings are single-quoted with `'` and `\`
/// escaped.
fn literal_to_string(value: &Value) -> String {
    match value {
        Value::Null => "NULL".to_string(),
        Value::Bool(b) => b.to_string(),
        Value::Int64(i) => i.to_string(),
        Value::Float64(f) => format!("{f:?}"),
        Value::String(s) => format!("'{}'", s.replace('\\', "\\\\").replace('\'', "\\'")),
        Value::List(items) => {
            let inner = items
                .iter()
                .map(literal_to_string)
                .collect::<Vec<_>>()
                .join(", ");
            format!("[{inner}]")
        }
        Value::Map(entries) => {
            let inner = entries
                .iter()
                .map(|(key, value)| format!("{key}: {}", literal_to_string(value)))
                .collect::<Vec<_>>()
                .join(", ");
            format!("{{{inner}}}")
        }
        Value::Vector(values) => {
            let inner = values
                .iter()
                .map(|v| format!("{v:?}"))
                .collect::<Vec<_>>()
                .join(", ");
            format!("vector([{inner}])")
        }
        other => other.to_string(),
    }
}

/// Renders an operand of a binary or unary expression, parenthesizing nested
/// operators so `(a + b) * c` and `a + b * c` get different names.
fn operand_to_string(expr: &LogicalExpression) -> String {
    match expr {
        LogicalExpression::Binary { .. } | LogicalExpression::Unary { .. } => {
            format!("({})", expression_to_string(expr))
        }
        _ => expression_to_string(expr),
    }
}

fn join_expressions(exprs: &[LogicalExpression]) -> String {
    exprs
        .iter()
        .map(expression_to_string)
        .collect::<Vec<_>>()
        .join(", ")
}

/// Column name of an unaliased aggregate: `count(*)`, `count(n)`,
/// `sum(DISTINCT n.age)`, `covar_pop(x, y)`, `percentile_cont(n.age, 0.5)`,
/// `group_concat(n.name, ', ')`. Every argument that changes the result is
/// rendered, so distinct aggregates get distinct names. Shared by the LPG and
/// RDF planners.
pub(crate) fn aggregate_column_name(aggregate: &crate::query::plan::AggregateExpr) -> String {
    let function = aggregate_function_name(aggregate.function);
    let mut args: Vec<String> = Vec::with_capacity(3);
    match &aggregate.expression {
        Some(expr) => args.push(expression_to_string(expr)),
        None => args.push("*".to_string()),
    }
    if let Some(second) = &aggregate.expression2 {
        args.push(expression_to_string(second));
    }
    if let Some(percentile) = aggregate.percentile {
        args.push(literal_to_string(&Value::Float64(percentile)));
    }
    if let Some(separator) = &aggregate.separator {
        args.push(literal_to_string(&Value::String(separator.as_str().into())));
    }
    let distinct = if aggregate.distinct { "DISTINCT " } else { "" };
    format!("{function}({distinct}{})", args.join(", "))
}

/// Surface name of an aggregate function, for column names.
fn aggregate_function_name(function: crate::query::plan::AggregateFunction) -> &'static str {
    use crate::query::plan::AggregateFunction as F;
    match function {
        F::Count | F::CountNonNull => "count",
        F::Sum => "sum",
        F::Avg => "avg",
        F::Min => "min",
        F::Max => "max",
        F::Collect => "collect",
        F::StdDev => "stdev",
        F::StdDevPop => "stdevp",
        F::Variance => "var_samp",
        F::VariancePop => "var_pop",
        F::PercentileDisc => "percentile_disc",
        F::PercentileCont => "percentile_cont",
        F::GroupConcat => "group_concat",
        F::Sample => "sample",
        F::CovarSamp => "covar_samp",
        F::CovarPop => "covar_pop",
        F::Corr => "corr",
        F::RegrSlope => "regr_slope",
        F::RegrIntercept => "regr_intercept",
        F::RegrR2 => "regr_r2",
        F::RegrCount => "regr_count",
        F::RegrSxx => "regr_sxx",
        F::RegrSyy => "regr_syy",
        F::RegrSxy => "regr_sxy",
        F::RegrAvgx => "regr_avgx",
        F::RegrAvgy => "regr_avgy",
    }
}

/// Surface symbol of a binary operator, for column names.
fn binary_op_symbol(op: BinaryOp) -> &'static str {
    match op {
        BinaryOp::Eq => "=",
        BinaryOp::Ne => "<>",
        BinaryOp::Lt => "<",
        BinaryOp::Le => "<=",
        BinaryOp::Gt => ">",
        BinaryOp::Ge => ">=",
        BinaryOp::And => "AND",
        BinaryOp::Or => "OR",
        BinaryOp::Xor => "XOR",
        BinaryOp::Add => "+",
        BinaryOp::Sub => "-",
        BinaryOp::Mul => "*",
        BinaryOp::Div => "/",
        BinaryOp::Mod => "%",
        BinaryOp::Concat => "||",
        BinaryOp::StartsWith => "STARTS WITH",
        BinaryOp::EndsWith => "ENDS WITH",
        BinaryOp::Contains => "CONTAINS",
        BinaryOp::In => "IN",
        BinaryOp::Like => "LIKE",
        BinaryOp::Regex => "=~",
        BinaryOp::Pow => "^",
    }
}

#[cfg(all(test, feature = "triple-store"))]
mod tests {
    use super::*;
    use grafeo_common::types::LogicalType;
    use grafeo_core::execution::DataChunk;
    use grafeo_core::execution::operators::{Operator, OperatorResult};

    struct MockOperator {
        chunk: Option<DataChunk>,
    }

    impl MockOperator {
        fn new(chunk: DataChunk) -> Self {
            Self { chunk: Some(chunk) }
        }
    }

    impl Operator for MockOperator {
        fn next(&mut self) -> OperatorResult {
            Ok(self.chunk.take())
        }

        fn reset(&mut self) {}

        fn name(&self) -> &'static str {
            "Mock"
        }

        fn into_any(self: Box<Self>) -> Box<dyn std::any::Any + Send> {
            self
        }
    }

    /// Creates a `DataChunk` with the given schema and pushes two rows of dummy data.
    fn make_chunk(types: &[LogicalType]) -> DataChunk {
        let mut chunk = DataChunk::with_capacity(types, 2);
        for (i, t) in types.iter().enumerate() {
            let col = chunk.column_mut(i).unwrap();
            match t {
                LogicalType::String => {
                    col.push_string("v1");
                    col.push_string("v2");
                }
                LogicalType::Int64 => {
                    col.push_int64(1);
                    col.push_int64(2);
                }
                _ => {}
            }
        }
        chunk.set_count(2);
        chunk
    }

    /// Cardinalities where right (100) < left (1000) * 0.8 trigger the swap branch.
    #[test]
    fn test_inner_join_swap_sides_preserves_types() {
        let left_cols = vec!["s".to_string(), "name".to_string()];
        let right_cols = vec!["s".to_string(), "age".to_string()];
        let left_types = vec![LogicalType::String, LogicalType::String];
        let right_types = vec![LogicalType::String, LogicalType::Int64];

        let left_op: Box<dyn Operator> = Box::new(MockOperator::new(make_chunk(&left_types)));
        let right_op: Box<dyn Operator> = Box::new(MockOperator::new(make_chunk(&right_types)));

        let (_, output_columns, output_types) = build_inner_join(
            left_op,
            right_op,
            &left_cols,
            &right_cols,
            &left_types,
            &right_types,
            Some((1000.0, 100.0)),
        );

        assert_eq!(output_columns, vec!["s", "name", "age"]);
        // Types must match logical column order, not physical swap order
        assert_eq!(
            output_types,
            vec![LogicalType::String, LogicalType::String, LogicalType::Int64]
        );
    }

    /// Cardinalities where left (100) < right (1000) skip the swap.
    #[test]
    fn test_inner_join_no_swap_preserves_types() {
        let left_cols = vec!["s".to_string(), "name".to_string()];
        let right_cols = vec!["s".to_string(), "age".to_string()];
        let left_types = vec![LogicalType::String, LogicalType::String];
        let right_types = vec![LogicalType::String, LogicalType::Int64];

        let left_op: Box<dyn Operator> = Box::new(MockOperator::new(make_chunk(&left_types)));
        let right_op: Box<dyn Operator> = Box::new(MockOperator::new(make_chunk(&right_types)));

        let (_, output_columns, output_types) = build_inner_join(
            left_op,
            right_op,
            &left_cols,
            &right_cols,
            &left_types,
            &right_types,
            Some((100.0, 1000.0)),
        );

        assert_eq!(output_columns, vec!["s", "name", "age"]);
        assert_eq!(
            output_types,
            vec![LogicalType::String, LogicalType::String, LogicalType::Int64]
        );
    }

    /// Disjoint columns produce a cross join (no swap regardless of cardinalities).
    #[test]
    fn test_inner_join_cross_join_types() {
        let left_cols = vec!["a".to_string()];
        let right_cols = vec!["b".to_string()];
        let left_types = vec![LogicalType::String];
        let right_types = vec![LogicalType::Int64];

        let left_op: Box<dyn Operator> = Box::new(MockOperator::new(make_chunk(&left_types)));
        let right_op: Box<dyn Operator> = Box::new(MockOperator::new(make_chunk(&right_types)));

        let (_, output_columns, output_types) = build_inner_join(
            left_op,
            right_op,
            &left_cols,
            &right_cols,
            &left_types,
            &right_types,
            Some((1000.0, 100.0)),
        );

        assert_eq!(output_columns, vec!["a", "b"]);
        assert_eq!(output_types, vec![LogicalType::String, LogicalType::Int64]);
    }

    /// Left join deduplicates shared columns and preserves types.
    #[test]
    fn test_left_join_preserves_types() {
        let left_cols = vec!["s".to_string(), "name".to_string()];
        let right_cols = vec!["s".to_string(), "age".to_string()];
        let left_types = vec![LogicalType::String, LogicalType::String];
        let right_types = vec![LogicalType::String, LogicalType::Int64];

        let left_op: Box<dyn Operator> = Box::new(MockOperator::new(make_chunk(&left_types)));
        let right_op: Box<dyn Operator> = Box::new(MockOperator::new(make_chunk(&right_types)));

        let (_, output_columns, output_types) = build_left_join(
            left_op,
            right_op,
            &left_cols,
            &right_cols,
            &left_types,
            &right_types,
        );

        assert_eq!(output_columns, vec!["s", "name", "age"]);
        assert_eq!(
            output_types,
            vec![LogicalType::String, LogicalType::String, LogicalType::Int64]
        );
    }
}
