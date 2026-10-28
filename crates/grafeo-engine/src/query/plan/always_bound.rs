//! The variables a SPARQL plan binds in every row it produces.
//!
//! Two SPARQL solutions are compatible when they agree on the variables both
//! bind (SPARQL 1.1 sec 18.3): a variable one side may leave unbound (the
//! right side of an OPTIONAL, one UNION branch, a BIND whose expression can
//! fail) agrees with any value. A join hashes on the shared variables both
//! sides always bind and checks the others per pair.

use std::collections::HashSet;

use super::{LogicalExpression, LogicalOperator, TripleComponent};

/// The variables every row of `op` binds, or `None` for an operator this does
/// not know, whose variables are then taken as always bound.
pub(crate) fn always_bound(op: &LogicalOperator) -> Option<HashSet<String>> {
    Some(match op {
        LogicalOperator::TripleScan(scan) => {
            let mut bound = match &scan.input {
                Some(input) => always_bound(input)?,
                None => HashSet::new(),
            };
            let components = [&scan.subject, &scan.predicate, &scan.object];
            bound.extend(variables(components.into_iter().chain(&scan.graph)));
            bound
        }
        LogicalOperator::PropertyPath(path) => {
            variables([&path.subject, &path.object].into_iter().chain(&path.graph)).collect()
        }
        LogicalOperator::Filter(filter) => always_bound(&filter.input)?,
        LogicalOperator::Limit(limit) => always_bound(&limit.input)?,
        LogicalOperator::Skip(skip) => always_bound(&skip.input)?,
        LogicalOperator::Sort(sort) => always_bound(&sort.input)?,
        LogicalOperator::Distinct(distinct) => always_bound(&distinct.input)?,
        LogicalOperator::Join(join) => {
            let mut bound = always_bound(&join.left)?;
            bound.extend(always_bound(&join.right)?);
            bound
        }
        LogicalOperator::MultiWayJoin(join) => {
            let mut bound = HashSet::new();
            for input in &join.inputs {
                bound.extend(always_bound(input)?);
            }
            bound
        }
        LogicalOperator::LeftJoin(join) => always_bound(&join.left)?,
        LogicalOperator::AntiJoin(join) => always_bound(&join.left)?,
        LogicalOperator::Union(union) => {
            let mut branches = union.inputs.iter();
            let Some(first) = branches.next() else {
                return Some(HashSet::new());
            };
            let mut bound = always_bound(first)?;
            for branch in branches {
                let other = always_bound(branch)?;
                bound.retain(|variable| other.contains(variable));
            }
            bound
        }
        LogicalOperator::Bind(bind) => {
            let mut bound = always_bound(&bind.input)?;
            if matches!(bind.expression, LogicalExpression::Literal(_)) {
                bound.insert(bind.variable.clone());
            }
            bound
        }
        LogicalOperator::Project(project) => {
            let input = always_bound(&project.input)?;
            let mut bound = if project.pass_through_input {
                input.clone()
            } else {
                HashSet::new()
            };
            for projection in &project.projections {
                if let Some(name) = renamed(&projection.expression, &projection.alias, &input) {
                    bound.insert(name);
                }
            }
            bound
        }
        LogicalOperator::Return(ret) => {
            let input = always_bound(&ret.input)?;
            ret.items
                .iter()
                .filter_map(|item| renamed(&item.expression, &item.alias, &input))
                .collect()
        }
        LogicalOperator::Aggregate(aggregate) => {
            let input = always_bound(&aggregate.input)?;
            aggregate
                .group_by
                .iter()
                .filter_map(|key| match key {
                    LogicalExpression::Variable(name) if input.contains(name) => Some(name.clone()),
                    _ => None,
                })
                .collect()
        }
        LogicalOperator::NamedGraphs(graphs) => HashSet::from([graphs.variable.clone()]),
        LogicalOperator::Empty => HashSet::new(),
        _ => return None,
    })
}

/// The variables among `components`.
fn variables<'a>(
    components: impl Iterator<Item = &'a TripleComponent>,
) -> impl Iterator<Item = String> {
    components.filter_map(|component| match component {
        TripleComponent::Variable(name) => Some(name.clone()),
        _ => None,
    })
}

/// The output name of a projected variable that `input` always binds.
fn renamed(
    expression: &LogicalExpression,
    alias: &Option<String>,
    input: &HashSet<String>,
) -> Option<String> {
    match expression {
        LogicalExpression::Variable(name) if input.contains(name) => {
            Some(alias.clone().unwrap_or_else(|| name.clone()))
        }
        _ => None,
    }
}
