//! Shared planning utilities for LPG and RDF planners.
//!
//! These free functions build physical operators from pre-planned children,
//! eliminating duplication between `Planner` (LPG) and `RdfPlanner`.
//! Each function takes already-planned input operators and column lists,
//! plus a schema derivation function to handle LPG vs RDF type differences.

use crate::query::plan::{BinaryOp, LogicalExpression, UnaryOp};
use grafeo_common::types::LogicalType;
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

/// Resolves a logical expression to a column index in the given variable-column map.
///
/// Handles three expression kinds:
/// - `Variable(name)`: direct lookup in `variable_columns`
/// - `Property { variable, property }`: lookup of `"{variable}_{property}"` (LPG projections)
/// - Complex expressions: lookup of `"__expr_{expr:?}"` (synthetic columns)
///
/// `context` is appended to error messages (e.g. `" for ORDER BY"`, or `""` for aggregations).
///
/// NOTE: The expression *collection* loops (which build the synthetic columns that this
/// function resolves) are intentionally NOT shared, because the LPG and RDF planners use
/// different `convert_expression` signatures (method on `&self` vs free function).
pub(crate) fn resolve_expression_to_column(
    expr: &LogicalExpression,
    variable_columns: &std::collections::HashMap<String, usize>,
    context: &str,
) -> Result<usize> {
    match expr {
        LogicalExpression::Variable(name) => variable_columns
            .get(name)
            .copied()
            .ok_or_else(|| Error::Internal(format!("Variable '{name}' not found{context}"))),
        LogicalExpression::Property { variable, property } => {
            let col_name = format!("{variable}_{property}");
            variable_columns.get(&col_name).copied().ok_or_else(|| {
                Error::Internal(format!(
                    "Property column '{col_name}' not found{context} (from {variable}.{property})"
                ))
            })
        }
        _ => {
            let col_name = format!("__expr_{expr:?}");
            variable_columns.get(&col_name).copied().ok_or_else(|| {
                Error::Internal(format!(
                    "Cannot resolve expression to column{context}: {expr:?}"
                ))
            })
        }
    }
}

/// Converts a logical expression to a human-readable string for column naming.
///
/// Used by `output_column_name` as the fallback when a Return/Project item
/// has no explicit alias. The goal is two-fold:
///
/// 1. The string should look like the source expression (users see it as a
///    column header).
/// 2. Two structurally distinct expressions should produce distinct strings,
///    so `RETURN n.a + n.b, n.c + n.d` doesn't yield two columns named the
///    same thing — downstream lookups by name would silently shadow.
///
/// We can't fully meet (2) without sacrificing (1); heavy expressions
/// (CASE / subqueries / comprehensions) collapse to a short generic label.
/// Two such expressions in one Return without aliases will still collide —
/// the right answer there is to alias them. This function covers every
/// arithmetic/logical/scalar shape users commonly write inline.
pub(crate) fn expression_to_string(expr: &LogicalExpression) -> String {
    match expr {
        LogicalExpression::Variable(name) => name.clone(),
        LogicalExpression::Property { variable, property } => {
            format!("{variable}.{property}")
        }
        LogicalExpression::Literal(value) => format!("{value:?}"),
        LogicalExpression::Parameter(name) => format!("${name}"),
        LogicalExpression::FunctionCall { name, args, .. } => {
            // Include argument signatures so `count(n)` and `count(m)` don't
            // collapse to the same column. Empty arg lists keep the "()" form.
            let inner = args
                .iter()
                .map(expression_to_string)
                .collect::<Vec<_>>()
                .join(", ");
            format!("{name}({inner})")
        }
        LogicalExpression::IndexAccess { base, index } => {
            format!(
                "{}[{}]",
                expression_to_string(base),
                expression_to_string(index)
            )
        }
        LogicalExpression::SliceAccess { base, start, end } => {
            let s = start
                .as_deref()
                .map(expression_to_string)
                .unwrap_or_default();
            let e = end.as_deref().map(expression_to_string).unwrap_or_default();
            format!("{}[{s}..{e}]", expression_to_string(base))
        }
        LogicalExpression::Binary { op, left, right } => {
            format!(
                "({} {} {})",
                expression_to_string(left),
                binary_op_symbol(*op),
                expression_to_string(right)
            )
        }
        LogicalExpression::Unary { op, operand } => match op {
            UnaryOp::IsNull => format!("({} IS NULL)", expression_to_string(operand)),
            UnaryOp::IsNotNull => format!("({} IS NOT NULL)", expression_to_string(operand)),
            UnaryOp::Not => format!("(NOT {})", expression_to_string(operand)),
            UnaryOp::Neg => format!("(-{})", expression_to_string(operand)),
        },
        LogicalExpression::Labels(v) => format!("labels({v})"),
        LogicalExpression::Type(v) => format!("type({v})"),
        LogicalExpression::Id(v) => format!("id({v})"),
        LogicalExpression::List(items) => {
            let inner = items
                .iter()
                .map(expression_to_string)
                .collect::<Vec<_>>()
                .join(", ");
            format!("[{inner}]")
        }
        LogicalExpression::Map(entries) => {
            let inner = entries
                .iter()
                .map(|(k, v)| format!("{k}: {}", expression_to_string(v)))
                .collect::<Vec<_>>()
                .join(", ");
            format!("{{{inner}}}")
        }
        LogicalExpression::MapProjection { base, .. } => format!("{base}{{...}}"),
        // Heavy expressions — short generic labels. Two unaliased instances
        // in the same Return still collide; the user should alias them.
        LogicalExpression::Case { .. } => "case".to_string(),
        LogicalExpression::ExistsSubquery(_) => "exists".to_string(),
        LogicalExpression::CountSubquery(_) => "count".to_string(),
        LogicalExpression::ValueSubquery(_) => "subquery".to_string(),
        LogicalExpression::Reduce { .. } => "reduce".to_string(),
        LogicalExpression::ListComprehension { .. } => "list_comprehension".to_string(),
        LogicalExpression::ListPredicate { kind, .. } => format!("{kind:?}").to_lowercase(),
        LogicalExpression::PatternComprehension { .. } => "pattern_comprehension".to_string(),
    }
}

/// Cypher-style symbol for a `BinaryOp`. Used by `expression_to_string` so
/// auto-generated column names read like the source expression.
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
