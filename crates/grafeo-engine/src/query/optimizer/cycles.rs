//! Paths that come back to a variable they already bound.

use std::collections::HashSet;

use crate::query::plan::{BinaryOp, FilterOp, LogicalExpression, LogicalOperator};

/// Rewrites every expand whose target its input already binds, such as the
/// last hop of `(a)-->(b)-->(a)` or of `MATCH (a)-->(b) MATCH (b)-->(a)`,
/// into an expand to a fresh variable under a filter that it is the same
/// node. Expanding to the bound variable itself would bind it again and
/// return every path of that shape instead of the cycles.
pub(super) fn close_cycles(op: LogicalOperator) -> LogicalOperator {
    let mut fresh = 0;
    close(op, &mut fresh)
}

fn close(op: LogicalOperator, fresh: &mut usize) -> LogicalOperator {
    let op = op.map_children(|child| close(child, fresh));
    let LogicalOperator::Expand(mut expand) = op else {
        return op;
    };
    if !bound_variables(&expand.input).is_some_and(|bound| bound.contains(&expand.to_variable)) {
        return LogicalOperator::Expand(expand);
    }
    let target = std::mem::replace(&mut expand.to_variable, format!("_cycle_end_{fresh}"));
    *fresh += 1;
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
