//! An `ORDER BY ... LIMIT` reads what cannot raise after its top-k.

use std::collections::HashSet;

use super::Optimizer;
use crate::query::plan::{
    LimitOp, LogicalExpression, LogicalOperator, ProjectOp, Projection, ReturnItem, ReturnOp,
};

impl Optimizer {
    /// Moves the items of the `RETURN` under the statement's `ORDER BY ... LIMIT`
    /// that cannot raise, and that no sort key reads, above the limit: a property of
    /// a node or an edge the pattern binds (null for an `OPTIONAL MATCH`'s
    /// null), a literal, a parameter. They are then read for the rows the limit
    /// keeps instead of for every row. The `RETURN` below keeps the sort keys,
    /// every other item, and the variables the moved items read; the one above
    /// gives the columns in their order and with their names.
    ///
    /// Nothing moves under `DISTINCT` (its set needs every item), for an item
    /// without a name, when a sort key reads a name the `RETURN` does not make,
    /// or when no property moves (nothing to gain): an unaliased property
    /// (`RETURN m.content ORDER BY ...`) and a `SKIP` under the `LIMIT` (which
    /// puts no sort right under it) keep today's plan. A function call or a
    /// property of a value `WITH` or `UNWIND` binds stays: it may raise, and an
    /// error must neither move nor disappear.
    ///
    /// The limit is the first one down the statement's chain of single-input
    /// operators from the root (through `RETURN`, a projection and `SKIP`),
    /// where an `ORDER BY ... LIMIT` of a query sits; finding it builds
    /// nothing, so a plan without one costs nothing.
    pub(super) fn project_after_top_k(op: &mut LogicalOperator) {
        let mut op = op;
        loop {
            match op {
                LogicalOperator::Limit(_) => break,
                LogicalOperator::Return(ReturnOp { input, .. })
                | LogicalOperator::Project(ProjectOp { input, .. })
                | LogicalOperator::Skip(crate::query::plan::SkipOp { input, .. }) => {
                    op = input.as_mut();
                }
                _ => return,
            }
        }
        if let LogicalOperator::Limit(limit) = std::mem::replace(op, LogicalOperator::Empty) {
            *op = Self::defer_past(limit);
        }
    }

    /// [`project_after_top_k`](Self::project_after_top_k) of one limit.
    fn defer_past(mut limit: LimitOp) -> LogicalOperator {
        let LogicalOperator::Sort(sort) = limit.input.as_mut() else {
            return LogicalOperator::Limit(limit);
        };
        let LogicalOperator::Return(ret) = sort.input.as_mut() else {
            return LogicalOperator::Limit(limit);
        };
        // Only a property read gains anything: nothing is built for a `RETURN`
        // without one (an aggregate's columns, plain variables).
        if ret.distinct
            || !ret
                .items
                .iter()
                .any(|item| matches!(item.expression, LogicalExpression::Property { .. }))
        {
            return LogicalOperator::Limit(limit);
        }
        let name = |item: &ReturnItem| -> Option<String> {
            item.alias.clone().or_else(|| match &item.expression {
                LogicalExpression::Variable(variable) => Some(variable.clone()),
                _ => None,
            })
        };
        let mut sorted = HashSet::new();
        for key in &sort.keys {
            Self::collect_variables(&key.expression, &mut sorted);
        }
        let Some(names) = ret.items.iter().map(name).collect::<Option<Vec<_>>>() else {
            return LogicalOperator::Limit(limit);
        };
        if !sorted.iter().all(|variable| names.contains(variable)) {
            return LogicalOperator::Limit(limit);
        }
        let moves: Vec<bool> = ret
            .items
            .iter()
            .zip(&names)
            .map(|(item, name)| {
                !sorted.contains(name)
                    && match &item.expression {
                        LogicalExpression::Literal(_) | LogicalExpression::Parameter(_) => true,
                        LogicalExpression::Property { variable, .. } => {
                            binds_entity(&ret.input, variable)
                        }
                        _ => false,
                    }
            })
            .collect();
        // The variables the moved properties read, passed through below; a
        // kept item of that name must be the variable itself.
        let mut passed: Vec<String> = Vec::new();
        for (item, moved) in ret.items.iter().zip(&moves) {
            if let (true, LogicalExpression::Property { variable, .. }) = (moved, &item.expression)
                && !passed.contains(variable)
            {
                passed.push(variable.clone());
            }
        }
        if passed.is_empty() {
            return LogicalOperator::Limit(limit);
        }
        let clashes = ret.items.iter().zip(&names).zip(&moves).any(|((item, name), moved)| {
            !moved
                && passed.contains(name)
                && !matches!(&item.expression, LogicalExpression::Variable(variable) if variable == name)
        });
        if clashes {
            return LogicalOperator::Limit(limit);
        }

        let mut below = Vec::with_capacity(ret.items.len() + passed.len());
        let mut above = Vec::with_capacity(ret.items.len());
        for ((item, name), moved) in std::mem::take(&mut ret.items)
            .into_iter()
            .zip(names)
            .zip(moves)
        {
            if moved {
                above.push(item);
            } else {
                above.push(ReturnItem {
                    expression: LogicalExpression::Variable(name),
                    alias: item.alias.clone(),
                });
                below.push(item);
            }
        }
        for variable in passed {
            // A kept column of that name is the variable itself (no clash above);
            // `m AS node` makes no column `m`.
            let made = below.iter().any(|item| {
                item.alias.as_deref().map_or_else(
                    || matches!(&item.expression, LogicalExpression::Variable(v) if *v == variable),
                    |alias| alias == variable,
                )
            });
            if !made {
                below.push(ReturnItem {
                    expression: LogicalExpression::Variable(variable.clone()),
                    alias: Some(variable),
                });
            }
        }
        // Below, a projection as `WITH` makes one: a node or an edge stays its
        // id, read by the batch above, where a `RETURN` would build every row's
        // whole value before the top-k.
        let input = std::mem::replace(&mut ret.input, Box::new(LogicalOperator::Empty));
        *sort.input = LogicalOperator::Project(ProjectOp {
            projections: below
                .into_iter()
                .map(|item| Projection {
                    expression: item.expression,
                    alias: item.alias,
                })
                .collect(),
            input,
            pass_through_input: false,
        });
        LogicalOperator::Return(ReturnOp {
            items: above,
            distinct: false,
            input: Box::new(LogicalOperator::Limit(limit)),
        })
    }
}

/// Whether `op` binds `variable` to a node or a single edge of its pattern
/// (a scan, a one-hop expand, either side of a join), whose property reads
/// cannot raise. A variable another operator binds (`WITH`, `UNWIND`, a
/// variable-length edge list) is not.
fn binds_entity(op: &LogicalOperator, variable: &str) -> bool {
    match op {
        LogicalOperator::NodeScan(scan) => {
            scan.variable == variable
                || scan
                    .input
                    .as_deref()
                    .is_some_and(|input| binds_entity(input, variable))
        }
        LogicalOperator::Expand(expand) => {
            expand.to_variable == variable
                || (expand.edge_variable.as_deref() == Some(variable)
                    && expand.min_hops == 1
                    && expand.max_hops == Some(1))
                || binds_entity(&expand.input, variable)
        }
        LogicalOperator::Filter(filter) => binds_entity(&filter.input, variable),
        LogicalOperator::Join(join) => {
            binds_entity(&join.left, variable) || binds_entity(&join.right, variable)
        }
        LogicalOperator::LeftJoin(join) => {
            binds_entity(&join.left, variable) || binds_entity(&join.right, variable)
        }
        _ => false,
    }
}
