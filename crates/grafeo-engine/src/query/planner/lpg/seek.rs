//! Node seeks: a filter over a node scan that pins the node by ID or by an
//! indexed property becomes a lookup per input row instead of a scan per row.

use std::collections::HashSet;

use grafeo_core::execution::operators::{
    ExpressionPredicate, FilterOperator, NodeSeekOperator, Operator, SeekKey, SingleRowOperator,
};
use grafeo_core::graph::GraphStoreSearch;

use super::{Arc, HashMap, Result};
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
/// key is left to the plan-time index lookup, which resolves it once.
pub(crate) fn choose_seek<'a>(
    predicate: &'a LogicalExpression,
    scan: &NodeScanOp,
    has_index: impl Fn(&str) -> bool,
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
                let constant = scan.input.is_none();
                if !constant && by_property.is_none() {
                    by_property = Some(choice);
                }
            }
        }
    }
    by_property
}

fn split_conjuncts<'a>(expr: &'a LogicalExpression, out: &mut Vec<&'a LogicalExpression>) {
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
/// between calls).
fn value_variables(expr: &LogicalExpression) -> Option<HashSet<String>> {
    let mut variables = HashSet::new();
    collect_value_variables(expr, &mut variables).then_some(variables)
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
            const CHANGING: [&str; 5] = ["rand", "random", "randomuuid", "uuid", "timestamp"];
            !CHANGING.iter().any(|f| name.eq_ignore_ascii_case(f))
                && args.iter().all(|arg| collect_value_variables(arg, out))
        }
        _ => false,
    }
}

impl super::Planner {
    /// Plans a filter over a node scan as a seek when [`choose_seek`] finds
    /// one: the scan's input rows, each joined with the nodes its key
    /// selects, and the filter on top, which still decides every row.
    pub(super) fn try_plan_filter_with_node_seek(
        &self,
        filter: &FilterOp,
    ) -> Result<Option<(Box<dyn Operator>, Vec<String>)>> {
        let LogicalOperator::NodeScan(scan) = filter.input.as_ref() else {
            return Ok(None);
        };
        let Some(seek) = choose_seek(&filter.predicate, scan, |property| {
            self.store.has_property_index(property)
        }) else {
            return Ok(None);
        };

        let (input, mut columns) = match &scan.input {
            Some(input) => self.plan_operator(input)?,
            None => (
                Box::new(SingleRowOperator::new()) as Box<dyn Operator>,
                Vec::new(),
            ),
        };
        self.record_absorbed_scan_entry("NodeSeek", &filter.input);
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
            // no lookup, the filter checks the bound node, label included.
            let predicate = match &scan.label {
                Some(label) => LogicalExpression::Binary {
                    left: Box::new(filter.predicate.clone()),
                    op: BinaryOp::And,
                    right: Box::new(LogicalExpression::FunctionCall {
                        name: "hasLabel".to_string(),
                        args: vec![
                            LogicalExpression::Variable(scan.variable.clone()),
                            LogicalExpression::Literal(label.as_str().into()),
                        ],
                        distinct: false,
                    }),
                },
                None => filter.predicate.clone(),
            };
            let predicate = expression(&predicate, &columns)?;
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
        let predicate = expression(&filter.predicate, &columns)?;
        Ok(Some((
            Box::new(FilterOperator::new(Box::new(rows), Box::new(predicate))),
            columns,
        )))
    }
}
