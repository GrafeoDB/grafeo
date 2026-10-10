//! Patterns that come back to a node or an edge their input already binds.

use std::collections::HashSet;

use crate::query::plan::{BinaryOp, CreateElement, FilterOp, LogicalExpression, LogicalOperator};

/// Rewrites every expand that binds a variable its input already binds: a
/// target node, such as the last hop of `(a)-->(b)-->(a)` or of
/// `MATCH (a)-->(b) MATCH (b)-->(a)`, or an edge, such as `r` in
/// `MATCH ()-[r]->() MATCH (x)-[r]->(y)`. The expand binds a fresh variable
/// instead, under a filter that it is the same node or edge. Binding the
/// variable itself again would return every path of that shape instead of
/// the ones through the bound node or edge.
pub(crate) fn close_cycles(op: LogicalOperator) -> LogicalOperator {
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
        let outer = apply.input.bound_variables(imports);
        apply.subplan = Box::new(close(*apply.subplan, fresh, outer.as_ref()));
        return LogicalOperator::Apply(apply);
    }
    let op = op.map_children(|child| close(child, fresh, imports));
    let LogicalOperator::Expand(mut expand) = op else {
        return op;
    };
    let Some(bound) = expand.input.bound_variables(imports) else {
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
        LogicalOperator::Create(create) => create
            .elements
            .iter()
            .flat_map(|element| match element {
                CreateElement::Node { variable, .. } => vec![variable],
                CreateElement::Edge {
                    variable,
                    from_variable,
                    to_variable,
                    ..
                } => [from_variable, to_variable]
                    .into_iter()
                    .chain(variable)
                    .collect(),
            })
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
            [&path.source_var, &path.target_var, &path.path_alias]
                .into_iter()
                .chain(&path.edge_variable)
                .collect()
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
