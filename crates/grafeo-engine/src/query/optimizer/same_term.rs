//! A `sameTerm` between two columns, enforced where the two columns first meet.
//!
//! RDF columns hold term ids, and one term has one id, so `sameTerm(?a, ?b)` over a pattern that
//! binds both is the pattern with `?b` read as `?a`: within one triple pattern the scan then keeps
//! the rows whose positions hold the same id (a repeated variable, see the planner's
//! `triple_scan`), and across a join the two sides share the column, which the join keys on. So the
//! rewrite renames `?b` to `?a` in the pattern and drops the filter, and binds `?b` to `?a` again
//! above the pattern only where something above still reads `?b`.
//!
//! It covers two filters:
//! - a user's `FILTER(sameTerm(?a, ?b))`;
//! - the check a `GRAPH ?g` group makes when its pattern names `?g` itself:
//!   `Bind(?g := H)` over `Filter(!BOUND(?g) || SAMETERM(?g, H))`, with `H` the hidden name the
//!   pattern reads its graph under. Where the pattern always binds `?g`, the `!BOUND` disjunct is
//!   false on every row, the filter is `sameTerm(?g, H)`, and the `Bind` is then a no-op.
//!
//! The rename is only made where it is exact: both columns bound on every row of the pattern
//! (`always_bound`), and the pattern made of triple scans, inner joins and filters, so a scan
//! position is the only thing that binds either. Elsewhere (an `OPTIONAL`'s right side, an
//! `EXISTS`, a `MINUS`, a column a side may leave unbound) the filter stays as built. `=` is
//! value equality, not term identity, and is never rewritten.

use std::collections::HashSet;

use crate::query::plan::{
    BindOp, JoinCondition, JoinType, LogicalExpression, LogicalOperator, TripleComponent, UnaryOp,
    always_bound, collect_expression_variables, is_hidden_variable,
};

/// `root` with every `sameTerm` between two columns enforced where the two first meet (see the
/// module docs).
pub(super) fn push_same_term(root: LogicalOperator) -> LogicalOperator {
    rewrite(root, None)
}

/// `op` rewritten, with `needed` the variables an operator above reads (`None`: not known, so
/// every one).
fn rewrite(op: LogicalOperator, needed: Option<&HashSet<String>>) -> LogicalOperator {
    match op {
        LogicalOperator::Bind(bind) => match graph_check(bind) {
            Ok(renamed) => rewrite(renamed, needed),
            Err(bind) => {
                let below = with_expression(needed, &bind.expression);
                LogicalOperator::Bind(BindOp {
                    input: Box::new(rewrite(*bind.input, below.as_ref())),
                    ..bind
                })
            }
        },
        LogicalOperator::Filter(filter) => {
            if let Some((keep, replace)) = same_term_columns(&filter.predicate)
                && let Some(renamed) = renamed_pattern(&filter.input, &keep, &replace)
            {
                let below = rewrite(renamed, needed);
                return if is_hidden_variable(&replace)
                    || needed.is_some_and(|needed| !needed.contains(&replace))
                {
                    below
                } else {
                    LogicalOperator::Bind(BindOp {
                        expression: LogicalExpression::Variable(keep),
                        variable: replace,
                        input: Box::new(below),
                    })
                };
            }
            let below = with_expression(needed, &filter.predicate);
            let mut filter = filter;
            filter.input = Box::new(rewrite(*filter.input, below.as_ref()));
            LogicalOperator::Filter(filter)
        }
        LogicalOperator::Project(mut project) => {
            let mut below = HashSet::new();
            for projection in &project.projections {
                collect_expression_variables(&projection.expression, &mut below);
            }
            let below = if project.pass_through_input {
                needed.map(|needed| below.union(needed).cloned().collect())
            } else {
                Some(below)
            };
            project.input = Box::new(rewrite(*project.input, below.as_ref()));
            LogicalOperator::Project(project)
        }
        LogicalOperator::Limit(mut limit) => {
            limit.input = Box::new(rewrite(*limit.input, needed));
            LogicalOperator::Limit(limit)
        }
        LogicalOperator::Skip(mut skip) => {
            skip.input = Box::new(rewrite(*skip.input, needed));
            LogicalOperator::Skip(skip)
        }
        LogicalOperator::Distinct(mut distinct) => {
            distinct.input = Box::new(rewrite(*distinct.input, needed));
            LogicalOperator::Distinct(distinct)
        }
        other => other.map_children(|child| rewrite(child, None)),
    }
}

/// `needed` with the variables `expression` reads (`None` stays `None`).
fn with_expression(
    needed: Option<&HashSet<String>>,
    expression: &LogicalExpression,
) -> Option<HashSet<String>> {
    needed.map(|needed| {
        let mut needed = needed.clone();
        collect_expression_variables(expression, &mut needed);
        needed
    })
}

/// A `GRAPH ?g` group's check, `Bind(?g := H)` over `Filter(!BOUND(?g) || SAMETERM(?g, H))` over
/// a pattern that always binds `?g`: the pattern with `H` read as `?g` (the filter and the `Bind`
/// are then no-ops). Otherwise `bind`, as it was.
fn graph_check(bind: BindOp) -> Result<LogicalOperator, BindOp> {
    let LogicalExpression::Variable(hidden) = &bind.expression else {
        return Err(bind);
    };
    let LogicalOperator::Filter(filter) = bind.input.as_ref() else {
        return Err(bind);
    };
    let LogicalExpression::Binary { left, right, op } = &filter.predicate else {
        return Err(bind);
    };
    let is_or = matches!(op, crate::query::plan::BinaryOp::Or);
    let unbound = matches!(
        left.as_ref(),
        LogicalExpression::Unary { op: UnaryOp::Not, operand }
            if matches!(operand.as_ref(), LogicalExpression::SparqlCall { name, args }
                if name.eq_ignore_ascii_case("BOUND")
                    && matches!(args.as_slice(), [LogicalExpression::Variable(v)] if *v == bind.variable))
    );
    let same = same_term_columns(right).is_some_and(|(a, b)| {
        (a == bind.variable && b == *hidden) || (b == bind.variable && a == *hidden)
    });
    if !(is_or && unbound && same && is_hidden_variable(hidden)) {
        return Err(bind);
    }
    match renamed_pattern(&filter.input, &bind.variable, hidden) {
        Some(renamed) => Ok(renamed),
        None => Err(bind),
    }
}

/// The two columns of `sameTerm(?a, ?b)` (any case of the name), as the column kept and the one
/// read as it: a hidden name is the one replaced. `None` for any other expression, and for one
/// column against itself.
fn same_term_columns(predicate: &LogicalExpression) -> Option<(String, String)> {
    let LogicalExpression::SparqlCall { name, args } = predicate else {
        return None;
    };
    if !name.eq_ignore_ascii_case("sameTerm") {
        return None;
    }
    let [
        LogicalExpression::Variable(a),
        LogicalExpression::Variable(b),
    ] = args.as_slice()
    else {
        return None;
    };
    if a == b {
        return None;
    }
    Some(if is_hidden_variable(a) && !is_hidden_variable(b) {
        (b.clone(), a.clone())
    } else {
        (a.clone(), b.clone())
    })
}

/// `pattern` with `replace` read as `keep`, where that is exact (see the module docs): the
/// pattern binds both on every row, and is made of triple scans, inner joins and filters without
/// a subquery. `None` otherwise.
fn renamed_pattern(
    pattern: &LogicalOperator,
    keep: &str,
    replace: &str,
) -> Option<LogicalOperator> {
    let bound = always_bound(pattern)?;
    if !(bound.contains(keep) && bound.contains(replace) && is_plain_pattern(pattern)) {
        return None;
    }
    Some(rename(pattern.clone(), replace, keep))
}

/// Whether `op` is triple scans, inner joins and filters without a subquery: a scan position is
/// then the only thing that binds a variable.
fn is_plain_pattern(op: &LogicalOperator) -> bool {
    match op {
        LogicalOperator::TripleScan(scan) => scan.input.is_none(),
        LogicalOperator::Join(join) => {
            join.join_type == JoinType::Inner
                && is_plain_pattern(&join.left)
                && is_plain_pattern(&join.right)
        }
        LogicalOperator::Filter(filter) => {
            !crate::query::plan::has_subquery(&filter.predicate) && is_plain_pattern(&filter.input)
        }
        _ => false,
    }
}

/// `op`, a plain pattern (see [`is_plain_pattern`]), with every `from` read as `to`.
fn rename(op: LogicalOperator, from: &str, to: &str) -> LogicalOperator {
    let component = |component: TripleComponent| match component {
        TripleComponent::Variable(name) if name == from => {
            TripleComponent::Variable(to.to_string())
        }
        other => other,
    };
    match op {
        LogicalOperator::TripleScan(mut scan) => {
            scan.subject = component(scan.subject);
            scan.predicate = component(scan.predicate);
            scan.object = component(scan.object);
            scan.graph = scan.graph.map(component);
            LogicalOperator::TripleScan(scan)
        }
        LogicalOperator::Join(mut join) => {
            join.left = Box::new(rename(*join.left, from, to));
            join.right = Box::new(rename(*join.right, from, to));
            join.conditions = join
                .conditions
                .into_iter()
                .map(|condition| JoinCondition {
                    left: rename_expression(condition.left, from, to),
                    right: rename_expression(condition.right, from, to),
                })
                .collect();
            LogicalOperator::Join(join)
        }
        LogicalOperator::Filter(mut filter) => {
            filter.predicate = rename_expression(filter.predicate, from, to);
            filter.input = Box::new(rename(*filter.input, from, to));
            LogicalOperator::Filter(filter)
        }
        other => other,
    }
}

/// `expression` with every variable `from` read as `to`.
fn rename_expression(expression: LogicalExpression, from: &str, to: &str) -> LogicalExpression {
    match expression {
        LogicalExpression::Variable(name) if name == from => {
            LogicalExpression::Variable(to.to_string())
        }
        other => other.map_children(|child| rename_expression(child, from, to)),
    }
}
