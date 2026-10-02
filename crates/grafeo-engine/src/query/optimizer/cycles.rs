//! Patterns that come back to a node or an edge their input already binds.

use std::collections::HashSet;

use crate::query::plan::{BinaryOp, FilterOp, LogicalExpression, LogicalOperator};

/// Rewrites every expand that binds a variable its input already binds: a
/// target node, such as the last hop of `(a)-->(b)-->(a)` or of
/// `MATCH (a)-->(b) MATCH (b)-->(a)`, or an edge, such as `r` in
/// `MATCH ()-[r]->() MATCH (x)-[r]->(y)`. The expand binds a fresh variable
/// instead, under a filter that it is the same node or edge. Binding the
/// variable itself again would return every path of that shape instead of
/// the ones through the bound node or edge.
pub(super) fn close_cycles(op: LogicalOperator) -> LogicalOperator {
    let mut taken = HashSet::new();
    plan_names(&op, &mut taken);
    close(op, &mut FreshNames { taken, next: 0 }, None)
}

/// Names for the fresh variables, `_cycle_end_N` for a node and
/// `_bound_edge_N` for an edge, skipping the ones the plan already uses: a
/// user may name a variable `_cycle_end_0` too.
struct FreshNames {
    taken: HashSet<String>,
    next: usize,
}

impl FreshNames {
    fn next(&mut self, prefix: &str) -> String {
        loop {
            let name = format!("{prefix}_{}", self.next);
            self.next += 1;
            if !self.taken.contains(&name) {
                return name;
            }
        }
    }
}

/// Rewrites `op` and every operator below it. `imports` holds the variables
/// of the row a subquery runs for, the ones `CALL { WITH * ... }` imports.
fn close(
    op: LogicalOperator,
    fresh: &mut FreshNames,
    imports: Option<&HashSet<String>>,
) -> LogicalOperator {
    if let LogicalOperator::Apply(mut apply) = op {
        apply.input = Box::new(close(*apply.input, fresh, imports));
        let outer = bound_variables(&apply.input, imports);
        apply.subplan = Box::new(close(*apply.subplan, fresh, outer.as_ref()));
        return LogicalOperator::Apply(apply);
    }
    let op = op.map_children(|child| close(child, fresh, imports));
    let LogicalOperator::Expand(mut expand) = op else {
        return op;
    };
    let Some(bound) = bound_variables(&expand.input, imports) else {
        return LogicalOperator::Expand(expand);
    };
    let mut checks = Vec::new();
    if bound.contains(&expand.to_variable) {
        let target = std::mem::replace(&mut expand.to_variable, fresh.next("_cycle_end"));
        checks.push(same(&expand.to_variable, target));
    }
    if let Some(edge) = &mut expand.edge_variable
        && bound.contains(edge.as_str())
    {
        let bound_edge = std::mem::replace(edge, fresh.next("_bound_edge"));
        checks.push(same(edge, bound_edge));
    }
    let Some(predicate) = checks
        .into_iter()
        .reduce(|left, right| LogicalExpression::Binary {
            left: Box::new(left),
            op: BinaryOp::And,
            right: Box::new(right),
        })
    else {
        return LogicalOperator::Expand(expand);
    };
    LogicalOperator::Filter(FilterOp {
        predicate,
        input: Box::new(LogicalOperator::Expand(expand)),
        pushdown_hint: None,
    })
}

/// `fresh = bound`: the fresh variable is the node or edge bound before.
fn same(fresh: &str, bound: String) -> LogicalExpression {
    LogicalExpression::Binary {
        left: Box::new(LogicalExpression::Variable(fresh.to_string())),
        op: BinaryOp::Eq,
        right: Box::new(LogicalExpression::Variable(bound)),
    }
}

/// Adds every name that `op` or an operator below it binds.
fn plan_names(op: &LogicalOperator, names: &mut HashSet<String>) {
    let bound: Vec<&String> = match op {
        LogicalOperator::NodeScan(scan) => vec![&scan.variable],
        LogicalOperator::EdgeScan(scan) => vec![&scan.variable],
        LogicalOperator::Expand(expand) => [&expand.from_variable, &expand.to_variable]
            .into_iter()
            .chain(&expand.edge_variable)
            .chain(&expand.path_alias)
            .collect(),
        LogicalOperator::Project(project) => project
            .projections
            .iter()
            .filter_map(|projection| projection.alias.as_ref())
            .collect(),
        LogicalOperator::Aggregate(aggregate) => aggregate
            .aggregates
            .iter()
            .filter_map(|aggregate| aggregate.alias.as_ref())
            .collect(),
        LogicalOperator::HorizontalAggregate(aggregate) => {
            vec![&aggregate.list_column, &aggregate.alias]
        }
        LogicalOperator::Return(ret) => ret
            .items
            .iter()
            .filter_map(|item| item.alias.as_ref())
            .collect(),
        LogicalOperator::CreateNode(create) => vec![&create.variable],
        LogicalOperator::CreateEdge(create) => [&create.from_variable, &create.to_variable]
            .into_iter()
            .chain(&create.variable)
            .collect(),
        LogicalOperator::Merge(merge) => vec![&merge.variable],
        LogicalOperator::MergeRelationship(merge) => vec![
            &merge.variable,
            &merge.source_variable,
            &merge.target_variable,
        ],
        LogicalOperator::Bind(bind) => vec![&bind.variable],
        LogicalOperator::Unwind(unwind) => std::iter::once(&unwind.variable)
            .chain(&unwind.ordinality_var)
            .chain(&unwind.offset_var)
            .collect(),
        LogicalOperator::MapCollect(collect) => {
            vec![&collect.key_var, &collect.value_var, &collect.alias]
        }
        LogicalOperator::ShortestPath(path) => {
            vec![&path.source_var, &path.target_var, &path.path_alias]
        }
        LogicalOperator::VectorScan(scan) => vec![&scan.variable],
        LogicalOperator::VectorJoin(join) => std::iter::once(&join.right_variable)
            .chain(&join.score_variable)
            .collect(),
        LogicalOperator::TextScan(scan) => std::iter::once(&scan.variable)
            .chain(&scan.score_column)
            .collect(),
        LogicalOperator::ParameterScan(scan) => scan.columns.iter().collect(),
        LogicalOperator::CallProcedure(call) => call
            .yield_items
            .iter()
            .flatten()
            .map(|item| item.alias.as_ref().unwrap_or(&item.field_name))
            .collect(),
        LogicalOperator::LoadData(load) => vec![&load.variable],
        _ => Vec::new(),
    };
    names.extend(bound.into_iter().cloned());
    for child in op.children() {
        plan_names(child, names);
    }
}

/// The variables the rows of `op` hold, or `None` for an operator this pass
/// does not model, which then keeps its plan as it is. A `WITH` (`Project`)
/// holds only what it projects; `imports` is what a subquery's `WITH *`
/// imports.
fn bound_variables(
    op: &LogicalOperator,
    imports: Option<&HashSet<String>>,
) -> Option<HashSet<String>> {
    let mut bound = HashSet::new();
    match op {
        LogicalOperator::Empty => {}
        LogicalOperator::NodeScan(scan) => {
            if let Some(input) = &scan.input {
                bound = bound_variables(input, imports)?;
            }
            bound.insert(scan.variable.clone());
        }
        LogicalOperator::EdgeScan(scan) => {
            if let Some(input) = &scan.input {
                bound = bound_variables(input, imports)?;
            }
            bound.insert(scan.variable.clone());
        }
        LogicalOperator::Expand(expand) => {
            bound = bound_variables(&expand.input, imports)?;
            bound.insert(expand.to_variable.clone());
            bound.extend(expand.edge_variable.iter().cloned());
            bound.extend(expand.path_alias.iter().cloned());
        }
        LogicalOperator::Filter(filter) => return bound_variables(&filter.input, imports),
        LogicalOperator::Limit(limit) => return bound_variables(&limit.input, imports),
        LogicalOperator::Skip(skip) => return bound_variables(&skip.input, imports),
        LogicalOperator::Sort(sort) => return bound_variables(&sort.input, imports),
        LogicalOperator::Distinct(distinct) => return bound_variables(&distinct.input, imports),
        LogicalOperator::Project(project) => {
            if project.pass_through_input {
                bound = bound_variables(&project.input, imports)?;
            }
            for projection in &project.projections {
                match (&projection.alias, &projection.expression) {
                    (Some(alias), _) => {
                        bound.insert(alias.clone());
                    }
                    (None, LogicalExpression::Variable(name)) => {
                        bound.insert(name.clone());
                    }
                    _ => {}
                }
            }
        }
        LogicalOperator::Aggregate(aggregate) => {
            for key in &aggregate.group_by {
                if let LogicalExpression::Variable(name) = key {
                    bound.insert(name.clone());
                }
            }
            bound.extend(aggregate.aggregates.iter().filter_map(|a| a.alias.clone()));
        }
        LogicalOperator::Unwind(unwind) => {
            bound = bound_variables(&unwind.input, imports)?;
            bound.insert(unwind.variable.clone());
            bound.extend(unwind.ordinality_var.iter().cloned());
            bound.extend(unwind.offset_var.iter().cloned());
        }
        LogicalOperator::Bind(bind) => {
            bound = bound_variables(&bind.input, imports)?;
            bound.insert(bind.variable.clone());
        }
        LogicalOperator::Join(join) => {
            bound = bound_variables(&join.left, imports)?;
            bound.extend(bound_variables(&join.right, imports)?);
        }
        LogicalOperator::LeftJoin(join) => {
            bound = bound_variables(&join.left, imports)?;
            bound.extend(bound_variables(&join.right, imports)?);
        }
        // A subquery starts from the variables it imports from the row it
        // runs for: the ones its `WITH` names, or all of them for `WITH *`.
        LogicalOperator::ParameterScan(scan) => {
            if scan.columns.iter().any(|column| column == "*") {
                return imports.cloned();
            }
            bound.extend(scan.columns.iter().cloned());
        }
        // `CALL { ... }` adds the columns its subquery returns to each row.
        LogicalOperator::Apply(apply) => {
            bound = bound_variables(&apply.input, imports)?;
            returned_variables(&apply.subplan, &mut bound);
        }
        _ => return None,
    }
    Some(bound)
}

/// Adds the variables a subquery's `RETURN` names. A `RETURN *` adds none:
/// this pass then leaves a later pattern on them as it is.
fn returned_variables(op: &LogicalOperator, bound: &mut HashSet<String>) {
    match op {
        LogicalOperator::Return(ret) => {
            for item in &ret.items {
                match (&item.alias, &item.expression) {
                    (Some(alias), _) => {
                        bound.insert(alias.clone());
                    }
                    (None, LogicalExpression::Variable(name)) if name != "*" => {
                        bound.insert(name.clone());
                    }
                    _ => {}
                }
            }
        }
        LogicalOperator::Sort(sort) => returned_variables(&sort.input, bound),
        LogicalOperator::Limit(limit) => returned_variables(&limit.input, bound),
        LogicalOperator::Skip(skip) => returned_variables(&skip.input, bound),
        LogicalOperator::Distinct(distinct) => returned_variables(&distinct.input, bound),
        _ => {}
    }
}
