//! `EXISTS` subqueries as delim joins: the subquery runs once over the
//! distinct values of the variables it shares with the rows it filters.
//!
//! `WHERE EXISTS { MATCH (n)-[:R]->(t) WHERE ... }` holds for a row when the
//! subquery finds a match from its `n`. Run as a semi-join over the whole
//! graph, the subquery reads (and evaluates its `WHERE` on) every node, also
//! the ones no row holds: a `10 / n.zero` there fails on a node the query
//! never asked about, and a selective query reads the whole graph. Run per
//! row, it runs as often as there are rows. A delim join (as DuckDB calls it)
//! reads the rows once, gives the subquery the rows' distinct values of the
//! shared variables to start from, and keeps the rows whose values it
//! matched: one run, over exactly the values the rows hold.
//!
//! An `EXISTS` inside an `OR` or `NOT` cannot drop rows: it becomes a mark
//! join, which keeps every row and adds whether the subquery matched it as a
//! boolean column that the filter reads in its place. Its guard (the other
//! conditions of its `AND`, see `visit_guarded_subqueries`) filters the
//! values it starts from, so a row whose other condition is false starts no
//! run of it.

use std::collections::HashSet;

use crate::query::plan::{
    BinaryOp, DelimScanOp, FilterOp, JoinCondition, JoinOp, JoinType, LogicalExpression,
    LogicalOperator, MarkJoinOp, UnaryOp,
};
use crate::query::planner::lpg::subquery::{
    expression_names, subplan_variables, visit_guarded_subqueries,
};

/// `op` with each top-level `EXISTS` and `NOT EXISTS` of its filters that
/// reads the filtered rows' variables planned as a delim join (a semi or
/// anti join whose right side starts from a [`DelimScanOp`]), and each other
/// such `EXISTS` of them as a mark join ([`MarkJoinOp`]). A filter keeps
/// what does not fit: an `EXISTS` the node's own edges answer (the edge
/// check, which is faster), one whose guard reads more than the variables
/// it shares (planned per row, where the guard skips rows), any above a
/// write, and shapes the rewrite does not know.
pub(super) fn decorrelate_exists(op: LogicalOperator) -> LogicalOperator {
    let mut marks = 0;
    decorrelate(op, &mut marks)
}

/// [`decorrelate_exists`], naming the mark columns from `marks` on.
fn decorrelate(op: LogicalOperator, marks: &mut usize) -> LogicalOperator {
    // This frame is on the stack once per level of the plan (see
    // `crate::query::limits`): the rewrite runs in a frame of its own, and
    // only for a filter whose predicate holds an `EXISTS`.
    let op = op.map_children(|child| decorrelate(child, marks));
    match op {
        LogicalOperator::Filter(filter) if has_exists(&filter.predicate) => {
            decorrelate_filter(filter, marks)
        }
        other => other,
    }
}

/// Whether `predicate` holds an `EXISTS` in an expression the rewrite
/// reaches: a conjunct, or under `AND`, `OR`, `NOT`, other operators, a
/// function's arguments and `CASE`. Read without a copy: every filter of
/// every plan asks. An `EXISTS` elsewhere runs per row, as before.
#[inline(never)]
fn has_exists(predicate: &LogicalExpression) -> bool {
    match predicate {
        LogicalExpression::ExistsSubquery(_) => true,
        LogicalExpression::Binary { left, right, .. } => has_exists(left) || has_exists(right),
        LogicalExpression::Unary { operand, .. } => has_exists(operand),
        LogicalExpression::FunctionCall { args, .. } | LogicalExpression::List(args) => {
            args.iter().any(has_exists)
        }
        LogicalExpression::Case {
            operand,
            when_clauses,
            else_clause,
        } => {
            operand.as_deref().is_some_and(has_exists)
                || when_clauses
                    .iter()
                    .any(|(condition, result)| has_exists(condition) || has_exists(result))
                || else_clause.as_deref().is_some_and(has_exists)
        }
        _ => false,
    }
}

/// [`decorrelate_exists`] of one filter that holds an `EXISTS`.
#[inline(never)]
fn decorrelate_filter(filter: FilterOp, marks: &mut usize) -> LogicalOperator {
    // The RDF planner has its own EXISTS (and no delim join).
    if filter.input.has_mutations() || holds_triples(&filter.input) {
        return LogicalOperator::Filter(filter);
    }
    let Some(bound) = filter.input.bound_variables(None) else {
        return LogicalOperator::Filter(filter);
    };
    let mut conjuncts = Vec::new();
    and_chain(filter.predicate.clone(), &mut conjuncts);
    let mut joins = Vec::new();
    let mut plain = Vec::new();
    let mut other = Vec::new();
    let mut marked = Vec::new();
    for conjunct in conjuncts {
        match delim_join_of(&conjunct, &bound) {
            Some(join) => joins.push(join),
            None if has_subquery(&conjunct) => {
                let before = marked.len();
                let conjunct = mark_exists(conjunct, &bound, marks, &mut marked);
                // The labels a branch of it requires (`(n:A AND EXISTS
                // {...}) OR (n:B AND ...)`) filter first, so a scan of
                // every node reads only those labels' nodes and the mark
                // join starts from them alone (as the planner does for a
                // subquery it runs per row).
                if marked.len() > before
                    && let LogicalOperator::NodeScan(scan) = filter.input.as_ref()
                    && let Some(check) = crate::query::planner::lpg::scan::implied_label_check(
                        &conjunct,
                        &scan.variable,
                    )
                {
                    plain.push(check);
                }
                other.push(conjunct);
            }
            None => plain.push(conjunct),
        }
    }
    if joins.is_empty() && marked.is_empty() {
        return LogicalOperator::Filter(filter);
    }
    // The plain conjuncts filter first, so the subqueries start from the
    // values of the rows they keep; the other subqueries filter last.
    let mut input = *filter.input;
    if let Some(predicate) = and_of(plain) {
        input = LogicalOperator::Filter(FilterOp {
            predicate,
            input: Box::new(input),
            pushdown_hint: None,
        });
    }
    for (subplan, join_type, shared) in joins {
        input = LogicalOperator::Join(JoinOp {
            left: Box::new(input),
            right: Box::new(subplan),
            join_type,
            conditions: shared
                .into_iter()
                .map(|name| JoinCondition {
                    left: LogicalExpression::Variable(name.clone()),
                    right: LogicalExpression::Variable(name),
                })
                .collect(),
        });
    }
    for (right, shared, column) in marked {
        input = LogicalOperator::MarkJoin(MarkJoinOp {
            left: Box::new(input),
            right: Box::new(right),
            shared,
            column,
        });
    }
    match and_of(other) {
        Some(predicate) => LogicalOperator::Filter(FilterOp {
            predicate,
            input: Box::new(input),
            pushdown_hint: None,
        }),
        None => input,
    }
}

/// `conjunct` with each `EXISTS` in it that fits a delim join (see
/// [`delim_join_of`]) read from the column of a mark join, which it adds to
/// `marked` as (right side, shared variables, column). An `EXISTS` with a
/// guard that reads only the shared variables starts from the values the
/// guard passes; one whose guard reads more stays, to run per row.
fn mark_exists(
    mut conjunct: LogicalExpression,
    bound: &HashSet<String>,
    marks: &mut usize,
    marked: &mut Vec<(LogicalOperator, Vec<String>, String)>,
) -> LogicalExpression {
    let _ = visit_guarded_subqueries(&mut conjunct, None, &mut |subquery, guard| {
        if !matches!(subquery, LogicalExpression::ExistsSubquery(_)) {
            return Ok(());
        }
        let Some((mut right, _, shared)) = delim_join_of(subquery, bound) else {
            return Ok(());
        };
        if let Some(guard) = guard {
            let mut names = HashSet::new();
            if !expression_names(guard, &mut names)
                || !names.iter().all(|name| shared.contains(name))
            {
                return Ok(());
            }
            right = guard_values(right, guard);
        }
        let column = mark_column(marks, bound);
        marked.push((right, shared, column.clone()));
        *subquery = LogicalExpression::Variable(column);
        Ok(())
    });
    conjunct
}

/// `plan` with its [`DelimScanOp`] under a filter by `guard`: the values a
/// guard rejects start no run of the subquery. A row of such a value is
/// marked unmatched, which its `AND` makes false anyway.
fn guard_values(plan: LogicalOperator, guard: &LogicalExpression) -> LogicalOperator {
    match plan {
        LogicalOperator::DelimScan(scan) => LogicalOperator::Filter(FilterOp {
            predicate: guard.clone(),
            input: Box::new(LogicalOperator::DelimScan(scan)),
            pushdown_hint: None,
        }),
        other => other.map_children(|child| guard_values(child, guard)),
    }
}

/// The next mark column's name, not one of the rows' variables.
fn mark_column(marks: &mut usize, bound: &HashSet<String>) -> String {
    loop {
        let name = format!("__mark_{marks}");
        *marks += 1;
        if !bound.contains(&name) {
            return name;
        }
    }
}

/// The right side, join type and shared variables of the delim join of
/// `conjunct`, an `EXISTS` or `NOT EXISTS` over rows that bind `bound`, when
/// it fits.
fn delim_join_of(
    conjunct: &LogicalExpression,
    bound: &HashSet<String>,
) -> Option<(LogicalOperator, JoinType, Vec<String>)> {
    let (subplan, join_type) = match conjunct {
        LogicalExpression::ExistsSubquery(subplan) => (subplan.as_ref(), JoinType::Semi),
        LogicalExpression::Unary {
            op: UnaryOp::Not,
            operand,
        } => match operand.as_ref() {
            LogicalExpression::ExistsSubquery(subplan) => (subplan.as_ref(), JoinType::Anti),
            _ => return None,
        },
        _ => return None,
    };
    if subplan.has_mutations() || holds_triples(subplan) || answered_by_edge_check(subplan, bound) {
        return None;
    }
    let names = subplan_variables(subplan)?;
    let mut shared: Vec<String> = names
        .into_iter()
        .filter(|name| bound.contains(name))
        .collect();
    if shared.is_empty() {
        return None;
    }
    shared.sort();
    let rewritten = start_from_values(subplan.clone(), &shared)?;
    // The join matches on the shared variables: the subquery must give them.
    let given = rewritten.bound_variables(None)?;
    shared
        .iter()
        .all(|name| given.contains(name))
        .then_some((rewritten, join_type, shared))
}

/// `subplan` starting from the distinct values of `shared`: its scan of a
/// shared node becomes the [`DelimScanOp`] (with a check of the scan's
/// label); with no such scan (a subquery that reads the row only in its
/// `WHERE`), its first leaf is crossed with it, scanned once. `None` for a
/// subquery with scans of two shared nodes, a parameter scan or no leaf.
fn start_from_values(subplan: LogicalOperator, shared: &[String]) -> Option<LogicalOperator> {
    let mut scans = 0;
    count_shared_scans(&subplan, shared, &mut scans)?;
    let values = LogicalOperator::DelimScan(DelimScanOp {
        columns: shared.to_vec(),
    });
    match scans {
        // A pattern that reaches a shared node by an expand starts better
        // from its own scan (the planner's semi-join): crossing it with the
        // values would pair each value with every node it scans.
        0 if binds_shared(&subplan, shared) => None,
        0 => {
            let mut values = Some(values);
            let crossed = cross_first_leaf(subplan, &mut values);
            values.is_none().then_some(crossed)
        }
        1 => Some(replace_shared_scan(subplan, shared, values)),
        _ => None,
    }
}

/// Counts the places `plan` can start from the shared values: scans of
/// `shared` nodes with no input, and the parameter scan of shared variables
/// the translator starts a subquery from when it reads them only in its
/// `WHERE`. `None` for a plan of other operators than scans, expands,
/// filters and inner or cross joins (a `CALL`, whose parameter scans are its
/// own, an `OPTIONAL MATCH`, a `RETURN`), or a parameter scan of anything
/// else.
fn count_shared_scans(plan: &LogicalOperator, shared: &[String], scans: &mut usize) -> Option<()> {
    match plan {
        LogicalOperator::Apply(_) | LogicalOperator::DelimScan(_) => None,
        LogicalOperator::ParameterScan(scan) => {
            if !scan.columns.iter().all(|column| shared.contains(column)) {
                return None;
            }
            *scans += 1;
            Some(())
        }
        LogicalOperator::NodeScan(scan) if scan.input.is_none() => {
            if shared.contains(&scan.variable) {
                *scans += 1;
            }
            Some(())
        }
        // The shapes whose rows a start from the values leaves as they are:
        // an OPTIONAL MATCH (a left join from one row), an aggregate or a
        // RETURN would read them differently.
        LogicalOperator::NodeScan(_) | LogicalOperator::Expand(_) | LogicalOperator::Filter(_) => {
            plan.children()
                .into_iter()
                .try_for_each(|child| count_shared_scans(child, shared, scans))
        }
        LogicalOperator::Join(join)
            if matches!(join.join_type, JoinType::Inner | JoinType::Cross) =>
        {
            plan.children()
                .into_iter()
                .try_for_each(|child| count_shared_scans(child, shared, scans))
        }
        _ => None,
    }
}

/// Whether an expand of `plan` reaches or crosses a `shared` variable.
fn binds_shared(plan: &LogicalOperator, shared: &[String]) -> bool {
    if let LogicalOperator::Expand(expand) = plan
        && (shared.contains(&expand.from_variable)
            || shared.contains(&expand.to_variable)
            || expand
                .edge_variable
                .as_ref()
                .is_some_and(|edge| shared.contains(edge)))
    {
        return true;
    }
    plan.children()
        .into_iter()
        .any(|child| binds_shared(child, shared))
}

/// `plan` with its scan of a `shared` node replaced by `values`, under a
/// check of the scan's label.
fn replace_shared_scan(
    plan: LogicalOperator,
    shared: &[String],
    values: LogicalOperator,
) -> LogicalOperator {
    let mut values = Some(values);
    replace_in(plan, shared, &mut values)
}

fn replace_in(
    plan: LogicalOperator,
    shared: &[String],
    values: &mut Option<LogicalOperator>,
) -> LogicalOperator {
    match plan {
        // The shared values in place of the outer row's.
        LogicalOperator::ParameterScan(_) if values.is_some() => {
            values.take().expect("checked above")
        }
        LogicalOperator::NodeScan(scan)
            if scan.input.is_none() && shared.contains(&scan.variable) && values.is_some() =>
        {
            let start = values.take().expect("checked above");
            match scan.label {
                Some(label) => LogicalOperator::Filter(FilterOp {
                    predicate: LogicalExpression::FunctionCall {
                        callee: crate::query::plan::Callee::Builtin(
                            crate::query::plan::BuiltinFunction::HasLabel,
                        ),
                        name: "hasLabel".to_string(),
                        args: vec![
                            LogicalExpression::Variable(scan.variable),
                            LogicalExpression::Literal(label.into()),
                        ],
                        distinct: false,
                    },
                    input: Box::new(start),
                    pushdown_hint: None,
                }),
                None => start,
            }
        }
        other => other.map_children(|child| replace_in(child, shared, values)),
    }
}

/// `plan` with its first leaf crossed with `values`, which it takes.
fn cross_first_leaf(
    plan: LogicalOperator,
    values: &mut Option<LogicalOperator>,
) -> LogicalOperator {
    let is_leaf = plan.children().is_empty();
    if is_leaf {
        return match values.take() {
            Some(values) => LogicalOperator::Join(JoinOp {
                left: Box::new(values),
                right: Box::new(plan),
                join_type: JoinType::Cross,
                conditions: Vec::new(),
            }),
            None => plan,
        };
    }
    plan.map_children(|child| {
        if values.is_some() {
            cross_first_leaf(child, values)
        } else {
            child
        }
    })
}

/// Whether a per-row check answers `subplan` faster than a join, as
/// `Planner::extract_exists_pattern` and `Planner::exists_fast_path_fits`
/// decide it, which is faster than any join: a single expand of at least one
/// hop from a plain scan, starting at its named source (or its target when
/// the source is anonymous) that the row binds, with an edge new to the row,
/// and for one edge an end new to it (a longer path may end at a node of the
/// row: the check searches until it reaches it).
fn answered_by_edge_check(subplan: &LogicalOperator, bound: &HashSet<String>) -> bool {
    // A label of the far end (`(n)-[:USES]->(:Method)`) is a check on the
    // one node each edge reaches: the per-row check that stops at the
    // first such edge stays faster than expanding every edge for a join.
    let mut end_labels = Vec::new();
    let mut below = subplan;
    while let LogicalOperator::Filter(filter) = below {
        end_labels.push(&filter.predicate);
        below = &filter.input;
    }
    let LogicalOperator::Expand(expand) = below else {
        return false;
    };
    let LogicalOperator::NodeScan(source) = expand.input.as_ref() else {
        return false;
    };
    if expand.min_hops != 1 || source.input.is_some() {
        return false;
    }
    let one_edge = expand.max_hops == Some(1) && !expand.quantified;
    if !one_edge && !crate::query::planner::lpg::reachability::reaches_as_walks(expand) {
        return false;
    }
    let from_anonymous = expand.from_variable.starts_with("_anon_");
    let to_anonymous = expand.to_variable.starts_with("_anon_");
    let (start, end) = match (from_anonymous, to_anonymous) {
        (true, true) => return false,
        (true, false) if source.label.is_some() && !one_edge => return false,
        (true, false) => (&expand.to_variable, &expand.from_variable),
        (false, _) if source.label.is_some() => return false,
        (false, _) => (&expand.from_variable, &expand.to_variable),
    };
    bound.contains(start)
        && (!one_edge || !bound.contains(end))
        && expand
            .edge_variable
            .as_ref()
            .is_none_or(|edge| !bound.contains(edge))
        && (end_labels.is_empty()
            || (one_edge
                && !bound.contains(end)
                && end_labels
                    .iter()
                    .all(|predicate| only_label_checks(predicate, end))))
}

/// Whether `predicate` is an `AND` of label checks of `variable`.
fn only_label_checks(predicate: &LogicalExpression, variable: &str) -> bool {
    match predicate {
        LogicalExpression::Binary {
            left,
            op: BinaryOp::And,
            right,
        } => only_label_checks(left, variable) && only_label_checks(right, variable),
        other => crate::query::planner::lpg::scan::label_check(other, variable).is_some(),
    }
}

/// The members of the `AND` chain `expression`, in order.
fn and_chain(expression: LogicalExpression, out: &mut Vec<LogicalExpression>) {
    match expression {
        LogicalExpression::Binary {
            left,
            op: BinaryOp::And,
            right,
        } => {
            and_chain(*left, out);
            and_chain(*right, out);
        }
        member => out.push(member),
    }
}

/// The `AND` of `members`, `None` for none.
fn and_of(members: Vec<LogicalExpression>) -> Option<LogicalExpression> {
    members
        .into_iter()
        .reduce(|left, right| LogicalExpression::Binary {
            left: Box::new(left),
            op: BinaryOp::And,
            right: Box::new(right),
        })
}

/// Whether `plan` scans RDF triples.
fn holds_triples(plan: &LogicalOperator) -> bool {
    matches!(plan, LogicalOperator::TripleScan(_)) || plan.children().into_iter().any(holds_triples)
}

/// Whether `expression` holds a subquery.
fn has_subquery(expression: &LogicalExpression) -> bool {
    let mut found = false;
    let mut probe = expression.clone();
    let _ = crate::query::planner::lpg::subquery::visit_subqueries(&mut probe, &mut |_| {
        found = true;
        Ok(())
    });
    found
}
