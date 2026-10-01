//! Paths that come back to a variable they already bound.

use std::collections::HashSet;

use crate::query::plan::{BinaryOp, FilterOp, LogicalExpression, LogicalOperator};

/// Rewrites every expand whose target its input already binds, such as the
/// last hop of `(a)-->(b)-->(a)` or of `MATCH (a)-->(b) MATCH (b)-->(a)`,
/// into an expand to a fresh variable under a filter that it is the same
/// node. Expanding to the bound variable itself would bind it again and
/// return every path of that shape instead of the cycles.
pub(super) fn close_cycles(op: LogicalOperator) -> LogicalOperator {
    let mut taken = HashSet::new();
    plan_names(&op, &mut taken);
    close(op, &mut FreshNames { taken, next: 0 })
}

/// Names for the closing nodes, `_cycle_end_N`, skipping the ones the plan
/// already uses: a user may name a variable `_cycle_end_0` too.
struct FreshNames {
    taken: HashSet<String>,
    next: usize,
}

impl FreshNames {
    fn next(&mut self) -> String {
        loop {
            let name = format!("_cycle_end_{}", self.next);
            self.next += 1;
            if !self.taken.contains(&name) {
                return name;
            }
        }
    }
}

fn close(op: LogicalOperator, fresh: &mut FreshNames) -> LogicalOperator {
    let op = op.map_children(|child| close(child, fresh));
    let LogicalOperator::Expand(mut expand) = op else {
        return op;
    };
    if !bound_variables(&expand.input).is_some_and(|bound| bound.contains(&expand.to_variable)) {
        return LogicalOperator::Expand(expand);
    }
    let target = std::mem::replace(&mut expand.to_variable, fresh.next());
    let end = LogicalExpression::Variable(expand.to_variable.clone());
    LogicalOperator::Filter(FilterOp {
        predicate: LogicalExpression::Binary {
            left: Box::new(end),
            op: BinaryOp::Eq,
            right: Box::new(LogicalExpression::Variable(target)),
        },
        input: Box::new(LogicalOperator::Expand(expand)),
        pushdown_hint: None,
    })
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
/// holds only what it projects.
fn bound_variables(op: &LogicalOperator) -> Option<HashSet<String>> {
    let mut bound = HashSet::new();
    match op {
        LogicalOperator::Empty => {}
        LogicalOperator::NodeScan(scan) => {
            if let Some(input) = &scan.input {
                bound = bound_variables(input)?;
            }
            bound.insert(scan.variable.clone());
        }
        LogicalOperator::EdgeScan(scan) => {
            if let Some(input) = &scan.input {
                bound = bound_variables(input)?;
            }
            bound.insert(scan.variable.clone());
        }
        LogicalOperator::Expand(expand) => {
            bound = bound_variables(&expand.input)?;
            bound.insert(expand.to_variable.clone());
            bound.extend(expand.edge_variable.iter().cloned());
            bound.extend(expand.path_alias.iter().cloned());
        }
        LogicalOperator::Filter(filter) => return bound_variables(&filter.input),
        LogicalOperator::Limit(limit) => return bound_variables(&limit.input),
        LogicalOperator::Skip(skip) => return bound_variables(&skip.input),
        LogicalOperator::Sort(sort) => return bound_variables(&sort.input),
        LogicalOperator::Distinct(distinct) => return bound_variables(&distinct.input),
        LogicalOperator::Project(project) => {
            if project.pass_through_input {
                bound = bound_variables(&project.input)?;
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
            bound = bound_variables(&unwind.input)?;
            bound.insert(unwind.variable.clone());
            bound.extend(unwind.ordinality_var.iter().cloned());
            bound.extend(unwind.offset_var.iter().cloned());
        }
        LogicalOperator::Bind(bind) => {
            bound = bound_variables(&bind.input)?;
            bound.insert(bind.variable.clone());
        }
        LogicalOperator::Join(join) => {
            bound = bound_variables(&join.left)?;
            bound.extend(bound_variables(&join.right)?);
        }
        LogicalOperator::LeftJoin(join) => {
            bound = bound_variables(&join.left)?;
            bound.extend(bound_variables(&join.right)?);
        }
        _ => return None,
    }
    Some(bound)
}
