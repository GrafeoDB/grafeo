//! A statement's last `RETURN` that gives its aggregates as they are adds nothing: the
//! aggregate already names its columns so. `MATCH (n:Person) RETURN count(n)` is the count
//! alone, as it was planned before every aggregate got a projection over it (to read an id
//! back as its node), and every pass after this one walks one operator less.

use crate::query::plan::{
    AggregateFunction, AggregateOp, LogicalExpression, LogicalOperator, ReturnOp,
};

/// `root` without its `Return` when that `Return` is the plan's last operator and gives the
/// aggregate below it as it is (see [`gives_as_it_is`]); `root` as it is otherwise. Only the
/// root: anything over the `Return` (an `ORDER BY`, a `SKIP`, a `LIMIT`) keeps it.
pub(super) fn without_identity_return(root: LogicalOperator) -> LogicalOperator {
    match root {
        LogicalOperator::Return(ret) if gives_as_it_is(&ret) => *ret.input,
        other => other,
    }
}

/// Whether `ret` gives its input's columns as they are: not `DISTINCT`, over an aggregate with
/// no grouping key and no `HAVING`, one item per aggregate in its order, each the aggregate's
/// column under its own name, and each aggregate one whose value is never a node or an edge (a
/// `collect(n)` or a `min(n)` is read back as nodes through the `Return`).
fn gives_as_it_is(ret: &ReturnOp) -> bool {
    let LogicalOperator::Aggregate(AggregateOp {
        group_by,
        aggregates,
        having: None,
        ..
    }) = ret.input.as_ref()
    else {
        return false;
    };
    !ret.distinct
        && group_by.is_empty()
        && ret.items.len() == aggregates.len()
        && ret.items.iter().zip(aggregates).all(|(item, aggregate)| {
            let Some(column) = aggregate.alias.as_deref() else {
                return false;
            };
            matches!(&item.expression, LogicalExpression::Variable(name) if name == column)
                && item.alias.as_deref().is_none_or(|alias| alias == column)
                && matches!(
                    aggregate.function,
                    AggregateFunction::Count
                        | AggregateFunction::CountNonNull
                        | AggregateFunction::Sum
                        | AggregateFunction::Avg
                )
        })
}

#[cfg(test)]
mod tests {
    use super::without_identity_return;
    use crate::query::plan::{
        AggregateExpr, AggregateFunction, AggregateOp, LimitOp, LogicalExpression, LogicalOperator,
        NodeScanOp, ReturnItem, ReturnOp,
    };

    fn aggregate(function: AggregateFunction, column: &str) -> AggregateExpr {
        AggregateExpr {
            function,
            expression: Some(LogicalExpression::Variable("n".to_string())),
            expression2: None,
            distinct: false,
            alias: Some(column.to_string()),
            percentile: None,
            separator: None,
            keep_nulls: false,
        }
    }

    fn item(column: &str, alias: Option<&str>) -> ReturnItem {
        ReturnItem {
            expression: LogicalExpression::Variable(column.to_string()),
            alias: alias.map(str::to_string),
        }
    }

    /// `RETURN items` over an aggregate of `aggregates`, grouped by `keys`, over a scan of `n`.
    fn plan(
        items: Vec<ReturnItem>,
        aggregates: Vec<AggregateExpr>,
        keys: Vec<LogicalExpression>,
        distinct: bool,
    ) -> LogicalOperator {
        LogicalOperator::Return(ReturnOp {
            items,
            distinct,
            input: Box::new(LogicalOperator::Aggregate(AggregateOp {
                group_by: keys,
                aggregates,
                input: Box::new(LogicalOperator::NodeScan(NodeScanOp {
                    variable: "n".to_string(),
                    label: Some("Person".to_string()),
                    input: None,
                })),
                having: None,
            })),
        })
    }

    fn dropped(op: LogicalOperator) -> bool {
        matches!(without_identity_return(op), LogicalOperator::Aggregate(_))
    }

    #[test]
    fn a_return_of_its_aggregates_as_they_are_goes() {
        let count = || aggregate(AggregateFunction::Count, "count(n)");
        let sum = || aggregate(AggregateFunction::Sum, "sum(n.age)");
        assert!(dropped(plan(
            vec![item("count(n)", None)],
            vec![count()],
            vec![],
            false
        )));
        assert!(dropped(plan(
            vec![item("count(n)", Some("count(n)")), item("sum(n.age)", None)],
            vec![count(), sum()],
            vec![],
            false,
        )));
    }

    #[test]
    fn every_other_return_stays() {
        let count = || aggregate(AggregateFunction::Count, "count(n)");
        let sum = || aggregate(AggregateFunction::Sum, "sum(n.age)");
        let cases = [
            (
                "DISTINCT",
                plan(vec![item("count(n)", None)], vec![count()], vec![], true),
            ),
            (
                "a renamed column",
                plan(
                    vec![item("count(n)", Some("people"))],
                    vec![count()],
                    vec![],
                    false,
                ),
            ),
            (
                "a grouping key",
                plan(
                    vec![item("n", None), item("count(n)", None)],
                    vec![count()],
                    vec![LogicalExpression::Variable("n".to_string())],
                    false,
                ),
            ),
            (
                "a node read back (collect)",
                plan(
                    vec![item("collect(n)", None)],
                    vec![aggregate(AggregateFunction::Collect, "collect(n)")],
                    vec![],
                    false,
                ),
            ),
            (
                "a node read back (min)",
                plan(
                    vec![item("min(n)", None)],
                    vec![aggregate(AggregateFunction::Min, "min(n)")],
                    vec![],
                    false,
                ),
            ),
            (
                "another order",
                plan(
                    vec![item("sum(n.age)", None), item("count(n)", None)],
                    vec![count(), sum()],
                    vec![],
                    false,
                ),
            ),
            (
                "fewer columns",
                plan(
                    vec![item("count(n)", None)],
                    vec![count(), sum()],
                    vec![],
                    false,
                ),
            ),
        ];
        for (case, op) in cases {
            assert!(!dropped(op), "{case}");
        }
        // HAVING.
        let mut having = plan(vec![item("count(n)", None)], vec![count()], vec![], false);
        if let LogicalOperator::Return(ret) = &mut having
            && let LogicalOperator::Aggregate(agg) = ret.input.as_mut()
        {
            agg.having = Some(LogicalExpression::Variable("count(n)".to_string()));
        }
        assert!(!dropped(having), "HAVING");
        // Not the root: a LIMIT over it keeps it.
        let limited = LogicalOperator::Limit(LimitOp {
            count: crate::query::plan::CountExpr::Literal(1),
            input: Box::new(plan(
                vec![item("count(n)", None)],
                vec![count()],
                vec![],
                false,
            )),
        });
        assert!(matches!(
            without_identity_return(limited),
            LogicalOperator::Limit(limit) if matches!(*limit.input, LogicalOperator::Return(_))
        ));
    }
}
