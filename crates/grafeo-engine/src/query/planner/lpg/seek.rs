//! Node seeks: a filter over a node scan that pins the node by ID or by an
//! indexed property becomes a lookup per input row instead of a scan per row.

use std::collections::HashSet;

use grafeo_core::execution::operators::{
    ExpressionPredicate, FilterOperator, NodeSeekOperator, Operator, SeekKey, SingleRowOperator,
};
use grafeo_core::graph::GraphStoreSearch;

use super::{Arc, HashMap, Result};
use crate::query::optimizer::movable_variables;
use crate::query::plan::{BinaryOp, FilterOp, LogicalExpression, LogicalOperator, NodeScanOp};

/// The seek the planner makes of a filter over a node scan.
pub(crate) struct SeekChoice<'a> {
    /// What the seek looks nodes up by.
    pub key: SeekKey,
    /// The value looked up, evaluated for each input row.
    pub value: &'a LogicalExpression,
    /// For `IN`: the value is a list and each element is looked up.
    pub list: bool,
}

/// Chooses the seek for a filter over `scan`, if one applies.
///
/// A conjunct of the filter must pin the scanned node: `id(v) = e`, or
/// `v.p = e` with `p` indexed, or the `IN` form of either, where `e` does not
/// read `v` (it may read the scan's input, parameters and literals). An ID is
/// preferred over a property. A scan without input with a constant property
/// key is left to the plan-time index lookup, which resolves it once, unless
/// the filter runs `after_a_write` of its statement: that lookup would not
/// see the write (see `Planner::reads_the_store_as_planned`), the seek looks
/// the key up when its one input row arrives.
pub(crate) fn choose_seek<'a>(
    predicate: &'a LogicalExpression,
    scan: &NodeScanOp,
    has_index: impl Fn(&str) -> bool,
    after_a_write: bool,
) -> Option<SeekChoice<'a>> {
    let mut conjuncts = Vec::new();
    split_conjuncts(predicate, &mut conjuncts);
    let mut by_property = None;
    for conjunct in conjuncts {
        let Some(choice) = seek_of(conjunct, &scan.variable, &has_index) else {
            continue;
        };
        let Some(variables) = value_variables(choice.value) else {
            continue;
        };
        if variables.contains(&scan.variable) || (scan.input.is_none() && !variables.is_empty()) {
            continue;
        }
        match choice.key {
            SeekKey::Id => return Some(choice),
            SeekKey::Property(_) => {
                let looked_up_while_planning = scan.input.is_none() && !after_a_write;
                if !looked_up_while_planning && by_property.is_none() {
                    by_property = Some(choice);
                }
            }
        }
    }
    by_property
}

/// The node scan a filter's seek replaces, with the filters between them.
pub(crate) struct CheckedScan<'a> {
    /// The scan below the filter.
    pub scan: &'a NodeScanOp,
    /// The scan, as the operator it is in the plan.
    pub operator: &'a LogicalOperator,
    /// The filters between the filter and the scan, from the bottom up (the
    /// order they run in): each as the operator it is in the plan, and its
    /// predicate.
    pub checks: Vec<(&'a LogicalOperator, &'a LogicalExpression)>,
}

/// The node scan below `filter`, directly or below filters that check the
/// scanned node alone: each reads no other variable and may move (see
/// [`movable_variables`]), like the checks of the other labels of a pattern
/// such as `(t:Graph:TypeDefinition)` (see `scan.rs`). A seek for `filter`
/// replaces the scan, and checks these predicates on the nodes it finds.
pub(crate) fn checked_scan(filter: &FilterOp) -> Option<CheckedScan<'_>> {
    let mut checks = Vec::new();
    let mut below = filter.input.as_ref();
    while let LogicalOperator::Filter(check) = below {
        checks.push((below, &check.predicate));
        below = &check.input;
    }
    let LogicalOperator::NodeScan(scan) = below else {
        return None;
    };
    let checks_the_scan = |predicate: &LogicalExpression| {
        movable_variables(predicate)
            .is_some_and(|variables| variables.iter().all(|v| *v == scan.variable))
    };
    if !checks
        .iter()
        .all(|(_, predicate)| checks_the_scan(predicate))
    {
        return None;
    }
    checks.reverse();
    Some(CheckedScan {
        scan,
        operator: below,
        checks,
    })
}

pub(super) fn split_conjuncts<'a>(
    expr: &'a LogicalExpression,
    out: &mut Vec<&'a LogicalExpression>,
) {
    match expr {
        LogicalExpression::Binary {
            left,
            op: BinaryOp::And,
            right,
        } => {
            split_conjuncts(left, out);
            split_conjuncts(right, out);
        }
        other => out.push(other),
    }
}

/// The seek a single conjunct allows: `key = value`, `value = key` or
/// `key IN value`, where the key is `id(variable)` or an indexed property of
/// `variable`.
fn seek_of<'a>(
    conjunct: &'a LogicalExpression,
    variable: &str,
    has_index: &impl Fn(&str) -> bool,
) -> Option<SeekChoice<'a>> {
    let LogicalExpression::Binary { left, op, right } = conjunct else {
        return None;
    };
    let key = |side: &LogicalExpression| match side {
        LogicalExpression::Id(v) if v == variable => Some(SeekKey::Id),
        LogicalExpression::FunctionCall { name, args, .. }
            if name.eq_ignore_ascii_case("id")
                && matches!(args.as_slice(), [LogicalExpression::Variable(v)] if v == variable) =>
        {
            Some(SeekKey::Id)
        }
        LogicalExpression::Property {
            variable: v,
            property,
        } if v == variable && has_index(property) => Some(SeekKey::Property(property.clone())),
        _ => None,
    };
    match op {
        BinaryOp::Eq => key(left)
            .map(|key| SeekChoice {
                key,
                value: right,
                list: false,
            })
            .or_else(|| {
                key(right).map(|key| SeekChoice {
                    key,
                    value: left,
                    list: false,
                })
            }),
        BinaryOp::In => key(left).map(|key| SeekChoice {
            key,
            value: right,
            list: true,
        }),
        _ => None,
    }
}

/// The variables a seek value reads, or `None` for an expression a seek does
/// not evaluate (subqueries, comprehensions, functions whose value changes
/// between calls). The keys of a value join follow the same rule (see
/// `value_join.rs`).
pub(super) fn value_variables(expr: &LogicalExpression) -> Option<HashSet<String>> {
    let mut variables = HashSet::new();
    collect_value_variables(expr, &mut variables).then_some(variables)
}

/// Whether the function `name` called with `arity` arguments returns the same
/// value for the same arguments, so a seek can evaluate a key once and probe
/// the index with it while the filter above it reads the same value.
///
/// An allowlist on purpose: a function not listed (every clock and random
/// function among them) keeps the scan and its filter, so a missing name costs
/// speed, never rows. The evaluator dispatches functions by name, so this list
/// cannot come from it yet (#540). The reachability search of variable-length
/// expands uses it too (see `reachability.rs`).
pub(super) fn deterministic(name: &str, arity: usize) -> bool {
    const PURE: [&str; 39] = [
        "tostring",
        "tostringornull",
        "tointeger",
        "toint",
        "tointegerornull",
        "tofloat",
        "tofloatornull",
        "toboolean",
        "tobooleanornull",
        "tolower",
        "toupper",
        "lower",
        "upper",
        "trim",
        "ltrim",
        "rtrim",
        "btrim",
        "substring",
        "left",
        "right",
        "replace",
        "split",
        "reverse",
        "size",
        "length",
        "char_length",
        "character_length",
        "abs",
        "ceil",
        "floor",
        "round",
        "sign",
        "sqrt",
        "coalesce",
        "head",
        "last",
        "tail",
        "keys",
        "properties",
    ];
    // Temporal constructors: from their arguments (`date('2026-09-30')`), or
    // from the clock without any.
    const FROM_ARGUMENTS: [&str; 10] = [
        "date",
        "time",
        "datetime",
        "localdatetime",
        "local_datetime",
        "localtime",
        "local_time",
        "zoneddatetime",
        "zoned_datetime",
        "duration",
    ];
    PURE.iter().any(|f| name.eq_ignore_ascii_case(f))
        || (arity > 0 && FROM_ARGUMENTS.iter().any(|f| name.eq_ignore_ascii_case(f)))
}

fn collect_value_variables(expr: &LogicalExpression, out: &mut HashSet<String>) -> bool {
    match expr {
        LogicalExpression::Literal(_) | LogicalExpression::Parameter(_) => true,
        LogicalExpression::Variable(name)
        | LogicalExpression::Property { variable: name, .. }
        | LogicalExpression::Id(name)
        | LogicalExpression::Labels(name)
        | LogicalExpression::Type(name) => {
            out.insert(name.clone());
            true
        }
        LogicalExpression::Binary { left, right, .. } => {
            collect_value_variables(left, out) && collect_value_variables(right, out)
        }
        LogicalExpression::Unary { operand, .. } => collect_value_variables(operand, out),
        LogicalExpression::IndexAccess { base, index } => {
            collect_value_variables(base, out) && collect_value_variables(index, out)
        }
        LogicalExpression::MapAccess { base, .. } => collect_value_variables(base, out),
        LogicalExpression::List(items) => {
            items.iter().all(|item| collect_value_variables(item, out))
        }
        LogicalExpression::Map(entries) => entries
            .iter()
            .all(|(_, value)| collect_value_variables(value, out)),
        LogicalExpression::FunctionCall { name, args, .. } => {
            deterministic(name, args.len())
                && args.iter().all(|arg| collect_value_variables(arg, out))
        }
        _ => false,
    }
}

impl super::Planner {
    /// Plans a filter over a node scan (see [`checked_scan`]) as a seek when
    /// [`choose_seek`] finds one: the scan's input rows, each joined with the
    /// nodes its key selects, and a filter on top, which still decides every
    /// row with the checks between the filter and the scan, then the filter's
    /// own predicate.
    pub(super) fn try_plan_filter_with_node_seek(
        &self,
        filter: &FilterOp,
    ) -> Result<Option<(Box<dyn Operator>, Vec<String>)>> {
        let Some(CheckedScan {
            scan,
            operator,
            checks,
        }) = checked_scan(filter)
        else {
            return Ok(None);
        };
        // A seek looks a key up when its input row arrives: after a write in
        // the input it would miss what the input writes for later rows (the
        // scan reads its whole input first, see `plan_node_scan`).
        if scan
            .input
            .as_deref()
            .is_some_and(LogicalOperator::has_mutations)
        {
            return Ok(None);
        }
        // A property index holds the values of now (see
        // `reads_the_current_store`); an ID is no index.
        let current = self.reads_the_current_store();
        let after_a_write = !self.reads_the_store_as_planned();
        let Some(seek) = choose_seek(
            &filter.predicate,
            scan,
            |property| current && self.store.has_property_index(property),
            after_a_write,
        ) else {
            return Ok(None);
        };

        let (input, mut columns) = match &scan.input {
            Some(input) => self.plan_operator(input)?,
            None => (
                Box::new(SingleRowOperator::new()) as Box<dyn Operator>,
                Vec::new(),
            ),
        };
        // PROFILE entries for the scan and the checks the seek absorbs, in
        // the order the plan's tree lists them (children first).
        self.record_absorbed_scan_entry("NodeSeek", operator);
        for (check, _) in &checks {
            self.record_absorbed_scan_entry("Filter", check);
        }
        // `first`, the checks, then the filter's predicate: the order the
        // chain of filters runs them in.
        let checked = |first: Option<LogicalExpression>| {
            first
                .into_iter()
                .chain(checks.iter().map(|(_, predicate)| (*predicate).clone()))
                .rev()
                .fold(filter.predicate.clone(), |rest, check| {
                    LogicalExpression::Binary {
                        left: Box::new(check),
                        op: BinaryOp::And,
                        right: Box::new(rest),
                    }
                })
        };
        let store = Arc::clone(&self.store) as Arc<dyn GraphStoreSearch>;
        let expression =
            |expr: &LogicalExpression, columns: &[String]| -> Result<ExpressionPredicate> {
                let variable_columns: HashMap<String, usize> = columns
                    .iter()
                    .enumerate()
                    .map(|(i, name)| (name.clone(), i))
                    .collect();
                Ok(ExpressionPredicate::new(
                    self.convert_expression(expr)?,
                    variable_columns,
                    Arc::clone(&store),
                )
                .with_transaction_context(self.viewing_epoch, self.transaction_id)
                .with_session_context(self.session_context.clone()))
            };

        if columns.contains(&scan.variable) {
            // The input binds the variable already (a correlated subquery):
            // no lookup, the filter checks the bound node, labels included.
            let label = scan
                .label
                .as_ref()
                .map(|label| LogicalExpression::FunctionCall {
                    name: "hasLabel".to_string(),
                    args: vec![
                        LogicalExpression::Variable(scan.variable.clone()),
                        LogicalExpression::Literal(label.as_str().into()),
                    ],
                    distinct: false,
                });
            let predicate = expression(&checked(label), &columns)?;
            return Ok(Some((
                Box::new(FilterOperator::new(input, Box::new(predicate))),
                columns,
            )));
        }

        let key = expression(seek.value, &columns)?;
        let mut rows =
            NodeSeekOperator::new(Arc::clone(&store), input, seek.key, key, scan.label.clone())
                .with_transaction_context(self.viewing_epoch, self.transaction_id);
        if seek.list {
            rows = rows.with_list_key();
        }
        columns.push(scan.variable.clone());
        let predicate = expression(&checked(None), &columns)?;
        Ok(Some((
            Box::new(FilterOperator::new(Box::new(rows), Box::new(predicate))),
            columns,
        )))
    }
}

#[cfg(test)]
mod tests {
    use grafeo_common::types::Value;

    use super::*;

    fn has_label(label: &str) -> LogicalExpression {
        LogicalExpression::FunctionCall {
            name: "hasLabel".to_string(),
            args: vec![
                LogicalExpression::Variable("t".to_string()),
                LogicalExpression::Literal(Value::from(label)),
            ],
            distinct: false,
        }
    }

    fn equals(left: LogicalExpression, right: LogicalExpression) -> LogicalExpression {
        LogicalExpression::Binary {
            left: Box::new(left),
            op: BinaryOp::Eq,
            right: Box::new(right),
        }
    }

    fn property(variable: &str, name: &str) -> LogicalExpression {
        LogicalExpression::Property {
            variable: variable.to_string(),
            property: name.to_string(),
        }
    }

    /// `t.filePath = path`, over the filters with `checks` (top down) over a
    /// scan of `t:Graph`.
    fn keyed_over(checks: Vec<LogicalExpression>) -> FilterOp {
        let scan = LogicalOperator::NodeScan(NodeScanOp {
            variable: "t".to_string(),
            label: Some("Graph".to_string()),
            input: None,
        });
        let input = checks.into_iter().rev().fold(scan, |input, predicate| {
            LogicalOperator::Filter(FilterOp {
                predicate,
                pushdown_hint: None,
                input: Box::new(input),
            })
        });
        FilterOp {
            predicate: equals(
                property("t", "filePath"),
                LogicalExpression::Variable("path".to_string()),
            ),
            pushdown_hint: None,
            input: Box::new(input),
        }
    }

    #[test]
    fn the_checks_of_the_scanned_node_are_listed_in_the_order_they_run() {
        let kind = equals(
            property("t", "kind"),
            LogicalExpression::Literal(Value::from("struct")),
        );
        let filter = keyed_over(vec![has_label("TypeDefinition"), kind.clone()]);
        let checked = checked_scan(&filter).expect("both filters check t alone");
        assert_eq!(checked.scan.label.as_deref(), Some("Graph"));
        assert!(matches!(checked.operator, LogicalOperator::NodeScan(_)));
        let predicates: Vec<String> = checked
            .checks
            .iter()
            .map(|(_, predicate)| format!("{predicate:?}"))
            .collect();
        assert_eq!(
            predicates,
            [
                format!("{kind:?}"),
                format!("{:?}", has_label("TypeDefinition"))
            ]
        );

        let direct = keyed_over(Vec::new());
        assert!(
            checked_scan(&direct).is_some_and(|checked| checked.checks.is_empty()),
            "a filter right on the scan has no checks"
        );
    }

    /// A filter between that reads another variable, holds a subquery or
    /// calls a volatile function is no check of the scanned node alone.
    #[test]
    fn other_filters_between_keep_the_scan() {
        let other_variable = equals(property("t", "owner"), property("f", "name"));
        let volatile = LogicalExpression::Binary {
            left: Box::new(LogicalExpression::FunctionCall {
                name: "rand".to_string(),
                args: Vec::new(),
                distinct: false,
            }),
            op: BinaryOp::Lt,
            right: Box::new(property("t", "share")),
        };
        let subquery =
            LogicalExpression::ExistsSubquery(Box::new(LogicalOperator::NodeScan(NodeScanOp {
                variable: "t".to_string(),
                label: Some("TypeDefinition".to_string()),
                input: None,
            })));
        for check in [other_variable, volatile, subquery] {
            let filter = keyed_over(vec![has_label("TypeDefinition"), check.clone()]);
            assert!(checked_scan(&filter).is_none(), "{check:?}");
        }
    }
}
