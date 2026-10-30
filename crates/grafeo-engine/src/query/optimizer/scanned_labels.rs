//! Label checks that the node scan below them makes true.

use crate::query::plan::{BinaryOp, FilterOp, LogicalExpression, LogicalOperator};
use crate::query::planner::lpg::scan::label_check;

/// `op` without the label checks a node scan below them guarantees: in
/// `MATCH (n:Method) WHERE 'Method' IN labels(n)` (or `n:Method`, or
/// `hasLabel(n, 'Method')`) every row of the scan has the label, so the
/// conjunct is dropped, and a filter that held nothing else goes with it.
///
/// Only conjuncts at the top of a filter's `AND` chain, in a chain of filters
/// right above the scan, are dropped: there the check holds for every row,
/// anywhere else (under an `OR` or a `NOT`, above an expand) it can be false.
pub(super) fn drop_scanned_label_checks(op: LogicalOperator) -> LogicalOperator {
    let op = op.map_children(drop_scanned_label_checks);
    let LogicalOperator::Filter(filter) = op else {
        return op;
    };
    let Some((variable, label)) = scanned_label(&filter.input) else {
        return LogicalOperator::Filter(filter);
    };
    let (variable, label) = (variable.to_string(), label.to_string());
    match without_checks(filter.predicate, &variable, &label) {
        Some(predicate) => LogicalOperator::Filter(FilterOp {
            predicate,
            ..filter
        }),
        None => *filter.input,
    }
}

/// The variable and label of the node scan at the bottom of a chain of
/// filters, when the scan reads one label.
fn scanned_label(mut op: &LogicalOperator) -> Option<(&str, &str)> {
    while let LogicalOperator::Filter(inner) = op {
        op = &inner.input;
    }
    match op {
        LogicalOperator::NodeScan(scan) => Some((scan.variable.as_str(), scan.label.as_deref()?)),
        _ => None,
    }
}

/// `predicate` without its top-level conjuncts that check `label` on
/// `variable`; `None` when nothing else is left.
fn without_checks(
    predicate: LogicalExpression,
    variable: &str,
    label: &str,
) -> Option<LogicalExpression> {
    match predicate {
        LogicalExpression::Binary {
            left,
            op: BinaryOp::And,
            right,
        } => {
            let left = without_checks(*left, variable, label);
            let right = without_checks(*right, variable, label);
            match (left, right) {
                (Some(left), Some(right)) => Some(LogicalExpression::Binary {
                    left: Box::new(left),
                    op: BinaryOp::And,
                    right: Box::new(right),
                }),
                (kept, None) | (None, kept) => kept,
            }
        }
        conjunct if label_check(&conjunct, variable) == Some(label) => None,
        conjunct => Some(conjunct),
    }
}
