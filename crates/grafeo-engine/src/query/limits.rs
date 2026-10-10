//! The size limit of a statement's plan.
//!
//! The parsers bound how deep a statement nests (see
//! [`MAX_NESTING_DEPTH`](grafeo_adapters::query::limits::MAX_NESTING_DEPTH)),
//! but a flat statement can still translate to a deep plan: each clause,
//! each hop of a path and each triple pattern adds an operator on top of the
//! previous one. Binding, optimizing, planning and running the plan recurse
//! once per operator and once per level of each expression, so a plan deeper
//! than the stack allows used to overflow it (#573). Every translator checks
//! the plan it built with [`check_plan_depth`], and a translator that walks
//! the plan it is building recursively checks it as it goes
//! (`check_partial_plan_depth`).
//!
//! Lists are not deep: the patterns of one `INSERT` or `CREATE` are one
//! [`CreateOp`](super::plan::CreateOp), a list of `SET` assignments of
//! constants is one operator, and a list of conditions (`AND`, `OR`, the
//! entries of a property map) is a balanced tree
//! ([`LogicalExpression::balanced`]).

use grafeo_common::types::Value;
use grafeo_common::utils::error::{Error, QueryError, QueryErrorKind, Result};

use super::plan::{
    AggregateExpr, LogicalExpression, LogicalOperator, LogicalPlan, MapProjectionEntry,
};

/// How deep a statement's plan may nest: the operators from the root down to
/// a leaf, with the levels of an expression or subquery inside one of them.
///
/// A statement whose plan nests deeper fails with an error that names this
/// limit before anything runs: a chain of more than about 120 clauses (`MATCH`,
/// `WITH`, `UNWIND`, `MERGE`, ...) or path hops. Split it, or run one clause
/// for the rows of a list parameter with `UNWIND`.
pub const MAX_PLAN_DEPTH: usize = 128;

/// Returns `plan` when it nests at most [`MAX_PLAN_DEPTH`] levels deep.
///
/// # Errors
///
/// Returns a semantic query error that names the limit when `plan` nests
/// deeper. The plan is then taken apart without recursion: dropping it the
/// usual way would recurse as deep as it nests.
pub fn check_plan_depth(plan: LogicalPlan) -> Result<LogicalPlan> {
    if plan_depth(&plan.root, MAX_PLAN_DEPTH + 1) <= MAX_PLAN_DEPTH {
        return Ok(plan);
    }
    dismantle(plan.root);
    Err(plan_depth_error())
}

/// Fails when the plan a translator is building nests deeper than
/// [`MAX_PLAN_DEPTH`] already, and then takes it apart (leaving
/// [`LogicalOperator::Empty`]). A translator checks its plan before each
/// clause, as translating a clause may walk the plan below it recursively.
#[cfg(any(feature = "gql", feature = "sparql"))]
pub(crate) fn check_partial_plan_depth(plan: &mut LogicalOperator) -> Result<()> {
    if plan_depth(plan, MAX_PLAN_DEPTH + 1) <= MAX_PLAN_DEPTH {
        return Ok(());
    }
    dismantle(std::mem::replace(plan, LogicalOperator::Empty));
    Err(plan_depth_error())
}

/// The error of a statement whose plan nests deeper than
/// [`MAX_PLAN_DEPTH`].
pub(crate) fn plan_depth_error() -> Error {
    Error::Query(QueryError::new(
        QueryErrorKind::Semantic,
        format!(
            "the statement nests deeper than the plan depth limit of {MAX_PLAN_DEPTH} levels: \
             split it into smaller statements (a chain of many clauses can UNWIND a list \
             parameter instead)"
        ),
    ))
}

/// Drops `root` without recursion: each operator and expression is emptied
/// of its children before it is dropped, and the children wait in a list.
fn dismantle(root: LogicalOperator) {
    let mut operators = vec![root];
    let mut expressions = Vec::new();
    while !operators.is_empty() || !expressions.is_empty() {
        if let Some(mut op) = operators.pop() {
            take_operator_expressions(&mut op, &mut expressions);
            let shell = op.map_children(|child| {
                operators.push(child);
                LogicalOperator::Empty
            });
            drop(shell);
        }
        if let Some(mut expression) = expressions.pop() {
            take_expression_children(&mut expression, &mut expressions, &mut operators);
            drop(expression);
        }
    }
}

/// A placeholder for an expression taken out of its parent.
fn hole() -> LogicalExpression {
    LogicalExpression::Literal(Value::Null)
}

/// A node of a plan: an operator or an expression inside one.
enum Node<'a> {
    Operator(&'a LogicalOperator),
    Expression(&'a LogicalExpression),
}

/// How many levels the plan below `root` nests, counting each operator and
/// each level of an expression, up to `cap`. Walks the plan without
/// recursion, so a plan of any depth is measured on a small stack.
fn plan_depth(root: &LogicalOperator, cap: usize) -> usize {
    let mut deepest = 0;
    let mut pending = vec![(Node::Operator(root), 1)];
    while let Some((node, depth)) = pending.pop() {
        deepest = deepest.max(depth);
        if deepest >= cap {
            break;
        }
        let below = depth + 1;
        match node {
            Node::Operator(op) => {
                pending.extend(
                    op.children()
                        .into_iter()
                        .map(|child| (Node::Operator(child), below)),
                );
                operator_expressions(op, &mut |expression| {
                    pending.push((Node::Expression(expression), below));
                });
            }
            Node::Expression(expression) => {
                expression_children(expression, &mut |child| pending.push((child, below)));
            }
        }
    }
    deepest
}

/// Calls `visit` with each expression `op` holds itself (not those of its
/// input operators).
fn operator_expressions<'a>(
    op: &'a LogicalOperator,
    visit: &mut impl FnMut(&'a LogicalExpression),
) {
    let mut properties = |properties: &'a [(String, LogicalExpression)]| {
        for (_, expression) in properties {
            visit(expression);
        }
    };
    match op {
        LogicalOperator::Filter(op) => visit(&op.predicate),
        LogicalOperator::Project(op) => op.projections.iter().for_each(|p| visit(&p.expression)),
        LogicalOperator::Aggregate(op) => {
            op.group_by.iter().for_each(&mut *visit);
            for AggregateExpr {
                expression,
                expression2,
                ..
            } in &op.aggregates
            {
                expression.iter().chain(expression2).for_each(&mut *visit);
            }
            op.having.iter().for_each(&mut *visit);
        }
        LogicalOperator::Sort(op) => op.keys.iter().for_each(|key| visit(&key.expression)),
        LogicalOperator::Return(op) => op.items.iter().for_each(|item| visit(&item.expression)),
        LogicalOperator::Join(op) => {
            for condition in &op.conditions {
                visit(&condition.left);
                visit(&condition.right);
            }
        }
        LogicalOperator::MultiWayJoin(op) => {
            for condition in &op.conditions {
                visit(&condition.left);
                visit(&condition.right);
            }
        }
        LogicalOperator::LeftJoin(op) => op.condition.iter().for_each(&mut *visit),
        LogicalOperator::CreateNode(op) => properties(&op.properties),
        LogicalOperator::CreateEdge(op) => properties(&op.properties),
        LogicalOperator::Create(op) => op.property_values().for_each(&mut *visit),
        LogicalOperator::SetProperty(op) => properties(&op.properties),
        LogicalOperator::Merge(op) => {
            properties(&op.match_properties);
            properties(&op.on_create);
            properties(&op.on_match);
        }
        LogicalOperator::MergeRelationship(op) => {
            properties(&op.match_properties);
            properties(&op.on_create);
            properties(&op.on_match);
        }
        LogicalOperator::Bind(op) => visit(&op.expression),
        LogicalOperator::Unwind(op) => visit(&op.expression),
        LogicalOperator::ShortestPath(op) => {
            if let Some(condition) = &op.edge_condition {
                visit(&condition.predicate);
            }
        }
        LogicalOperator::VectorScan(op) => visit(&op.query_vector),
        LogicalOperator::VectorJoin(op) => visit(&op.query_vector),
        LogicalOperator::TextScan(op) => visit(&op.query),
        LogicalOperator::CallProcedure(op) => op.arguments.iter().for_each(&mut *visit),
        // Operators without expressions of their own.
        LogicalOperator::NodeScan(_)
        | LogicalOperator::EdgeScan(_)
        | LogicalOperator::Expand(_)
        | LogicalOperator::Limit(_)
        | LogicalOperator::Skip(_)
        | LogicalOperator::Distinct(_)
        | LogicalOperator::DeleteNode(_)
        | LogicalOperator::DeleteEdge(_)
        | LogicalOperator::AddLabel(_)
        | LogicalOperator::RemoveLabel(_)
        | LogicalOperator::Empty
        | LogicalOperator::TripleScan(_)
        | LogicalOperator::PropertyPath(_)
        | LogicalOperator::Union(_)
        | LogicalOperator::AntiJoin(_)
        | LogicalOperator::Construct(_)
        | LogicalOperator::MapCollect(_)
        | LogicalOperator::InsertTriple(_)
        | LogicalOperator::DeleteTriple(_)
        | LogicalOperator::Modify(_)
        | LogicalOperator::ClearGraph(_)
        | LogicalOperator::CreateGraph(_)
        | LogicalOperator::DropGraph(_)
        | LogicalOperator::LoadGraph(_)
        | LogicalOperator::CopyGraph(_)
        | LogicalOperator::MoveGraph(_)
        | LogicalOperator::AddGraph(_)
        | LogicalOperator::HorizontalAggregate(_)
        | LogicalOperator::Except(_)
        | LogicalOperator::Intersect(_)
        | LogicalOperator::Otherwise(_)
        | LogicalOperator::Apply(_)
        | LogicalOperator::ParameterScan(_)
        | LogicalOperator::CreatePropertyGraph(_)
        | LogicalOperator::LoadData(_) => {}
    }
}

/// Calls `visit` with each expression and subquery operator directly inside
/// `expression`.
fn expression_children<'a>(expression: &'a LogicalExpression, visit: &mut impl FnMut(Node<'a>)) {
    let mut expressions = |expressions: &mut dyn Iterator<Item = &'a LogicalExpression>| {
        for child in expressions {
            visit(Node::Expression(child));
        }
    };
    match expression {
        LogicalExpression::Binary { left, right, .. } => {
            expressions(&mut [&**left, &**right].into_iter());
        }
        LogicalExpression::Unary { operand, .. } => expressions(&mut std::iter::once(&**operand)),
        LogicalExpression::FunctionCall { args, .. } | LogicalExpression::List(args) => {
            expressions(&mut args.iter());
        }
        LogicalExpression::Map(entries) => expressions(&mut entries.iter().map(|(_, e)| e)),
        LogicalExpression::IndexAccess { base, index } => {
            expressions(&mut [&**base, &**index].into_iter());
        }
        LogicalExpression::MapAccess { base, .. } => expressions(&mut std::iter::once(&**base)),
        LogicalExpression::SliceAccess { base, start, end } => expressions(
            &mut std::iter::once(&**base)
                .chain(start.as_deref())
                .chain(end.as_deref()),
        ),
        LogicalExpression::Case {
            operand,
            when_clauses,
            else_clause,
        } => expressions(
            &mut operand
                .as_deref()
                .into_iter()
                .chain(when_clauses.iter().flat_map(|(when, then)| [when, then]))
                .chain(else_clause.as_deref()),
        ),
        LogicalExpression::ListComprehension {
            list_expr,
            filter_expr,
            map_expr,
            ..
        } => expressions(
            &mut [&**list_expr, &**map_expr]
                .into_iter()
                .chain(filter_expr.as_deref()),
        ),
        LogicalExpression::ListPredicate {
            list_expr,
            predicate,
            ..
        } => expressions(&mut [&**list_expr, &**predicate].into_iter()),
        LogicalExpression::ExistsSubquery(subplan)
        | LogicalExpression::CountSubquery(subplan)
        | LogicalExpression::ValueSubquery(subplan) => visit(Node::Operator(subplan)),
        LogicalExpression::MapProjection { entries, .. } => {
            expressions(&mut entries.iter().filter_map(|entry| match entry {
                MapProjectionEntry::LiteralEntry(_, value) => Some(value),
                MapProjectionEntry::PropertySelector(_) | MapProjectionEntry::AllProperties => None,
            }));
        }
        LogicalExpression::Reduce {
            initial,
            list,
            expression,
            ..
        } => expressions(&mut [&**initial, &**list, &**expression].into_iter()),
        LogicalExpression::PatternComprehension {
            subplan,
            projection,
        } => {
            visit(Node::Operator(subplan));
            visit(Node::Expression(projection));
        }
        LogicalExpression::Literal(_)
        | LogicalExpression::Variable(_)
        | LogicalExpression::Property { .. }
        | LogicalExpression::Parameter(_)
        | LogicalExpression::Labels(_)
        | LogicalExpression::Type(_)
        | LogicalExpression::Id(_) => {}
    }
}

/// Moves the expressions `op` holds itself into `out` (see
/// [`operator_expressions`]), leaving placeholders.
fn take_operator_expressions(op: &mut LogicalOperator, out: &mut Vec<LogicalExpression>) {
    let mut take = |expression: &mut LogicalExpression| {
        out.push(std::mem::replace(expression, hole()));
    };
    match op {
        LogicalOperator::Filter(op) => take(&mut op.predicate),
        LogicalOperator::Project(op) => {
            op.projections
                .iter_mut()
                .for_each(|p| take(&mut p.expression));
        }
        LogicalOperator::Aggregate(op) => {
            op.group_by.iter_mut().for_each(&mut take);
            for aggregate in &mut op.aggregates {
                aggregate
                    .expression
                    .iter_mut()
                    .chain(aggregate.expression2.iter_mut())
                    .for_each(&mut take);
            }
            op.having.iter_mut().for_each(&mut take);
        }
        LogicalOperator::Sort(op) => op.keys.iter_mut().for_each(|key| take(&mut key.expression)),
        LogicalOperator::Return(op) => {
            op.items
                .iter_mut()
                .for_each(|item| take(&mut item.expression));
        }
        LogicalOperator::Join(op) => {
            for condition in &mut op.conditions {
                take(&mut condition.left);
                take(&mut condition.right);
            }
        }
        LogicalOperator::MultiWayJoin(op) => {
            for condition in &mut op.conditions {
                take(&mut condition.left);
                take(&mut condition.right);
            }
        }
        LogicalOperator::LeftJoin(op) => op.condition.iter_mut().for_each(&mut take),
        LogicalOperator::CreateNode(op) => op.properties.iter_mut().for_each(|(_, e)| take(e)),
        LogicalOperator::CreateEdge(op) => op.properties.iter_mut().for_each(|(_, e)| take(e)),
        LogicalOperator::Create(op) => op.property_values_mut().for_each(&mut take),
        LogicalOperator::SetProperty(op) => op.properties.iter_mut().for_each(|(_, e)| take(e)),
        LogicalOperator::Merge(op) => op
            .match_properties
            .iter_mut()
            .chain(op.on_create.iter_mut())
            .chain(op.on_match.iter_mut())
            .for_each(|(_, e)| take(e)),
        LogicalOperator::MergeRelationship(op) => op
            .match_properties
            .iter_mut()
            .chain(op.on_create.iter_mut())
            .chain(op.on_match.iter_mut())
            .for_each(|(_, e)| take(e)),
        LogicalOperator::Bind(op) => take(&mut op.expression),
        LogicalOperator::Unwind(op) => take(&mut op.expression),
        LogicalOperator::ShortestPath(op) => {
            if let Some(condition) = &mut op.edge_condition {
                take(&mut condition.predicate);
            }
        }
        LogicalOperator::VectorScan(op) => take(&mut op.query_vector),
        LogicalOperator::VectorJoin(op) => take(&mut op.query_vector),
        LogicalOperator::TextScan(op) => take(&mut op.query),
        LogicalOperator::CallProcedure(op) => op.arguments.iter_mut().for_each(&mut take),
        LogicalOperator::NodeScan(_)
        | LogicalOperator::EdgeScan(_)
        | LogicalOperator::Expand(_)
        | LogicalOperator::Limit(_)
        | LogicalOperator::Skip(_)
        | LogicalOperator::Distinct(_)
        | LogicalOperator::DeleteNode(_)
        | LogicalOperator::DeleteEdge(_)
        | LogicalOperator::AddLabel(_)
        | LogicalOperator::RemoveLabel(_)
        | LogicalOperator::Empty
        | LogicalOperator::TripleScan(_)
        | LogicalOperator::PropertyPath(_)
        | LogicalOperator::Union(_)
        | LogicalOperator::AntiJoin(_)
        | LogicalOperator::Construct(_)
        | LogicalOperator::MapCollect(_)
        | LogicalOperator::InsertTriple(_)
        | LogicalOperator::DeleteTriple(_)
        | LogicalOperator::Modify(_)
        | LogicalOperator::ClearGraph(_)
        | LogicalOperator::CreateGraph(_)
        | LogicalOperator::DropGraph(_)
        | LogicalOperator::LoadGraph(_)
        | LogicalOperator::CopyGraph(_)
        | LogicalOperator::MoveGraph(_)
        | LogicalOperator::AddGraph(_)
        | LogicalOperator::HorizontalAggregate(_)
        | LogicalOperator::Except(_)
        | LogicalOperator::Intersect(_)
        | LogicalOperator::Otherwise(_)
        | LogicalOperator::Apply(_)
        | LogicalOperator::ParameterScan(_)
        | LogicalOperator::CreatePropertyGraph(_)
        | LogicalOperator::LoadData(_) => {}
    }
}

/// Moves the expressions and subquery operators directly inside
/// `expression` into `expressions` and `operators` (see
/// [`expression_children`]), leaving placeholders.
fn take_expression_children(
    expression: &mut LogicalExpression,
    expressions: &mut Vec<LogicalExpression>,
    operators: &mut Vec<LogicalOperator>,
) {
    let mut take = |child: &mut LogicalExpression| {
        expressions.push(std::mem::replace(child, hole()));
    };
    let mut take_operator = |subplan: &mut LogicalOperator| {
        operators.push(std::mem::replace(subplan, LogicalOperator::Empty));
    };
    match expression {
        LogicalExpression::Binary { left, right, .. } => {
            take(left);
            take(right);
        }
        LogicalExpression::Unary { operand, .. } => take(operand),
        LogicalExpression::FunctionCall { args, .. } | LogicalExpression::List(args) => {
            args.iter_mut().for_each(&mut take);
        }
        LogicalExpression::Map(entries) => entries.iter_mut().for_each(|(_, e)| take(e)),
        LogicalExpression::IndexAccess { base, index } => {
            take(base);
            take(index);
        }
        LogicalExpression::MapAccess { base, .. } => take(base),
        LogicalExpression::SliceAccess { base, start, end } => {
            take(base);
            start.iter_mut().chain(end.iter_mut()).for_each(|e| take(e));
        }
        LogicalExpression::Case {
            operand,
            when_clauses,
            else_clause,
        } => {
            if let Some(operand) = operand {
                take(operand);
            }
            for (when, then) in when_clauses {
                take(when);
                take(then);
            }
            if let Some(else_clause) = else_clause {
                take(else_clause);
            }
        }
        LogicalExpression::ListComprehension {
            list_expr,
            filter_expr,
            map_expr,
            ..
        } => {
            take(list_expr);
            take(map_expr);
            if let Some(filter_expr) = filter_expr {
                take(filter_expr);
            }
        }
        LogicalExpression::ListPredicate {
            list_expr,
            predicate,
            ..
        } => {
            take(list_expr);
            take(predicate);
        }
        LogicalExpression::ExistsSubquery(subplan)
        | LogicalExpression::CountSubquery(subplan)
        | LogicalExpression::ValueSubquery(subplan) => take_operator(subplan),
        LogicalExpression::MapProjection { entries, .. } => {
            for entry in entries {
                if let MapProjectionEntry::LiteralEntry(_, value) = entry {
                    take(value);
                }
            }
        }
        LogicalExpression::Reduce {
            initial,
            list,
            expression,
            ..
        } => {
            take(initial);
            take(list);
            take(expression);
        }
        LogicalExpression::PatternComprehension {
            subplan,
            projection,
        } => {
            take_operator(subplan);
            take(projection);
        }
        LogicalExpression::Literal(_)
        | LogicalExpression::Variable(_)
        | LogicalExpression::Property { .. }
        | LogicalExpression::Parameter(_)
        | LogicalExpression::Labels(_)
        | LogicalExpression::Type(_)
        | LogicalExpression::Id(_) => {}
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::query::plan::{CreateNodeOp, FilterOp};
    use grafeo_common::types::Value;

    /// `size` node creations, each on top of the previous one.
    fn creations(size: usize) -> LogicalOperator {
        (0..size).fold(LogicalOperator::Empty, |input, i| {
            LogicalOperator::CreateNode(CreateNodeOp {
                variable: format!("n{i}"),
                labels: Vec::new(),
                properties: Vec::new(),
                input: Some(Box::new(input)),
            })
        })
    }

    /// An expression `size` levels deep: `NOT NOT ... true`.
    fn negations(size: usize) -> LogicalExpression {
        (1..size).fold(LogicalExpression::Literal(Value::Bool(true)), |inner, _| {
            LogicalExpression::Unary {
                op: crate::query::plan::UnaryOp::Not,
                operand: Box::new(inner),
            }
        })
    }

    #[test]
    fn operators_and_expressions_each_nest_one_level() {
        // `size` creations over the empty leaf: size + 1 operators.
        assert_eq!(plan_depth(&creations(3), usize::MAX), 4);
        let filter = LogicalOperator::Filter(FilterOp {
            predicate: negations(19),
            input: Box::new(creations(3)),
            pushdown_hint: None,
        });
        // The filter, then the deeper of its input (4) and its predicate (19).
        assert_eq!(plan_depth(&filter, usize::MAX), 20);
    }

    #[test]
    fn a_subquery_nests_inside_its_expression() {
        let filter = LogicalOperator::Filter(FilterOp {
            predicate: LogicalExpression::ExistsSubquery(Box::new(creations(88))),
            input: Box::new(LogicalOperator::Empty),
            pushdown_hint: None,
        });
        // The filter, its predicate, then 89 operators of the subquery.
        assert_eq!(plan_depth(&filter, usize::MAX), 91);
    }

    #[test]
    fn the_limit_is_inclusive_and_named_in_the_error() {
        // MAX_PLAN_DEPTH levels: MAX_PLAN_DEPTH - 1 creations over the leaf.
        let at_limit = LogicalPlan::new(creations(MAX_PLAN_DEPTH - 1));
        let kept = check_plan_depth(at_limit).expect("a plan at the limit is kept");
        assert_eq!(plan_depth(&kept.root, usize::MAX), MAX_PLAN_DEPTH);
        let beyond = LogicalPlan::new(creations(MAX_PLAN_DEPTH));
        let error = check_plan_depth(beyond).unwrap_err().to_string();
        assert!(
            error.contains(&format!("limit of {MAX_PLAN_DEPTH} levels")),
            "{error}"
        );
    }

    /// A plan 100,000 levels deep, in its operators, in an expression and in
    /// a subquery inside an expression.
    fn far_beyond_the_limit() -> LogicalPlan {
        let subquery_filter = LogicalOperator::Filter(FilterOp {
            predicate: negations(100_000),
            input: Box::new(creations(100_000)),
            pushdown_hint: None,
        });
        LogicalPlan::new(LogicalOperator::Filter(FilterOp {
            predicate: LogicalExpression::ExistsSubquery(Box::new(subquery_filter)),
            input: Box::new(creations(100_000)),
            pushdown_hint: None,
        }))
    }

    #[test]
    fn a_plan_far_beyond_the_limit_is_measured_and_dropped_on_a_small_stack() {
        // Built on a large stack, refused (and so taken apart) on a small one:
        // the usual drop would recurse 100,000 levels deep.
        let plan = std::thread::Builder::new()
            .stack_size(256 << 20)
            .spawn(far_beyond_the_limit)
            .unwrap()
            .join()
            .unwrap();
        let refused = std::thread::Builder::new()
            .stack_size(64 << 10)
            .spawn(move || check_plan_depth(plan).is_err())
            .unwrap()
            .join()
            .unwrap();
        assert!(refused, "100,000 levels are beyond the limit");
    }
}
