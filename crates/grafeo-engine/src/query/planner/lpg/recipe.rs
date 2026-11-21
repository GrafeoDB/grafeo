//! A prepared plan's physical shape, made once and opened per call.
//!
//! A prepared statement's cached plan keeps its parameters. Its recipe records, for each
//! operator of a covered kind, what the physical planner builds there, and the plan's
//! variable names. Each call opens it with its own values and view, through the builders the
//! planner uses, instead of planning again. The recipe walks beside the cached logical plan,
//! which holds everything else, and holds no store, view, transaction, memory or value. A plan
//! with any other kind has no recipe and is planned per call as before.

use std::collections::{HashMap, HashSet};

use grafeo_common::types::Value;
use grafeo_common::utils::error::{Error, Result};

use super::Planner;
use crate::query::plan::{
    BinaryOp, BuiltinFunction, Callee, FilterOp, LogicalExpression, LogicalOperator, LogicalPlan,
};
use crate::query::planner::PhysicalPlan;

/// The kinds a recipe covers, by name: the set a test pins, so it grows only on purpose.
#[cfg(test)]
pub(crate) const COVERED: [&str; 5] = ["Return", "Expand", "Filter", "Seek", "NodeScan"];

/// A prepared plan's recipe: what the physical planner builds for each operator, and the
/// plan's variable names, which the column names it makes up must skip.
#[derive(Debug)]
pub(crate) struct Recipe {
    root: Step,
    names: HashSet<String>,
    /// The parameters the plan reads, all in its seek.
    parameters: Vec<String>,
}

/// What the physical planner builds for one operator of a prepared plan, in the plan's order.
#[derive(Debug)]
enum Step {
    /// A `RETURN` of plain properties and variables over its input.
    Return(Box<Step>),
    /// One hop over its input.
    Expand(Box<Step>),
    /// A check of labels alone over its input.
    Filter(Box<Step>),
    /// A start a key pins: a filter that equates a property of a scanned node with a value or
    /// a parameter, over the scan, read by the planner's lookup.
    Seek,
    /// A scan of a label's nodes.
    NodeScan,
}

impl Step {
    /// The step's kind, as [`COVERED`] names it: a new step must be named here.
    #[cfg(test)]
    fn kind(&self) -> &'static str {
        match self {
            Self::Return(_) => "Return",
            Self::Expand(_) => "Expand",
            Self::Filter(_) => "Filter",
            Self::Seek => "Seek",
            Self::NodeScan => "NodeScan",
        }
    }

    /// The kinds of this step and the steps below it.
    #[cfg(test)]
    fn kinds(&self, into: &mut Vec<&'static str>) {
        into.push(self.kind());
        if let Self::Return(input) | Self::Expand(input) | Self::Filter(input) = self {
            input.kinds(into);
        }
    }
}

/// A plan that has no recipe: the first kind it holds outside [`COVERED`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Uncovered {
    /// The kind, by name.
    pub(crate) kind: &'static str,
}

/// What one call opens a recipe with: the planner the call builds, as a call builds it today,
/// whose builders and lookup `open` uses but which plans nothing, and the call's values.
pub(crate) struct OpenContext<'a> {
    /// The call's planner: its store, view, transaction, rules and memory.
    pub(crate) planner: &'a Planner,
    /// The call's values, by parameter name.
    pub(crate) values: &'a HashMap<String, Value>,
}

/// The recipe of `plan`, or the first kind it holds that no recipe covers.
pub(crate) fn recipe_of(plan: &LogicalPlan) -> std::result::Result<Recipe, Uncovered> {
    let root = step_of(&plan.root)?;
    let mut names = HashSet::new();
    crate::query::optimizer::plan_names(&plan.root, &mut names);
    let mut parameters = Vec::new();
    seek_parameters(&plan.root, &mut parameters);
    Ok(Recipe {
        root,
        names,
        parameters,
    })
}

/// The parameters of the seek in `op`, the one place a covered plan reads one.
fn seek_parameters(op: &LogicalOperator, parameters: &mut Vec<String>) {
    if let LogicalOperator::Filter(filter) = op
        && let LogicalExpression::Binary { left, right, .. } = &filter.predicate
    {
        for side in [left, right] {
            if let LogicalExpression::Parameter(name) = side.as_ref() {
                parameters.push(name.clone());
            }
        }
    }
    for child in op.children() {
        seek_parameters(child, parameters);
    }
}

fn step_of(op: &LogicalOperator) -> std::result::Result<Step, Uncovered> {
    let below = |input: &LogicalOperator| step_of(input).map(Box::new);
    let uncovered = |kind| Err(Uncovered { kind });
    match op {
        LogicalOperator::Return(ret) => {
            let plain = |item: &crate::query::plan::ReturnItem| {
                matches!(
                    item.expression,
                    LogicalExpression::Property { .. } | LogicalExpression::Variable(_)
                )
            };
            if ret.items.is_empty() || ret.distinct || !ret.items.iter().all(plain) {
                return uncovered("ReturnItems");
            }
            Ok(Step::Return(below(&ret.input)?))
        }
        LogicalOperator::Expand(expand) => {
            if expand.is_variable_length() {
                return uncovered("VariableLengthExpand");
            }
            if expand.path_alias.is_some() || expand.quantified {
                return uncovered("PathExpand");
            }
            Ok(Step::Expand(below(&expand.input)?))
        }
        LogicalOperator::Filter(filter) => {
            if filter.pushdown_hint.is_none() && pins_its_scan(filter) {
                return Ok(Step::Seek);
            }
            if filter.pushdown_hint.is_some()
                || matches!(filter.input.as_ref(), LogicalOperator::NodeScan(_))
                || !checks_labels_alone(&filter.predicate)
            {
                return uncovered("FilterOther");
            }
            Ok(Step::Filter(below(&filter.input)?))
        }
        LogicalOperator::NodeScan(scan) if scan.input.is_none() => Ok(Step::NodeScan),
        other => uncovered(kind_name(other)),
    }
}

/// Whether `filter` equates a property of the node its input scans, with no input of its
/// own, with a literal or a parameter.
fn pins_its_scan(filter: &FilterOp) -> bool {
    let LogicalOperator::NodeScan(scan) = filter.input.as_ref() else {
        return false;
    };
    if scan.input.is_some() {
        return false;
    }
    let LogicalExpression::Binary {
        left,
        op: BinaryOp::Eq,
        right,
    } = &filter.predicate
    else {
        return false;
    };
    let is_key = |side: &LogicalExpression| matches!(side, LogicalExpression::Property { variable, .. } if *variable == scan.variable);
    let is_value = |side: &LogicalExpression| {
        matches!(
            side,
            LogicalExpression::Literal(_) | LogicalExpression::Parameter(_)
        )
    };
    (is_key(left) && is_value(right)) || (is_value(left) && is_key(right))
}

/// Whether `predicate` is label checks alone, joined by `AND`.
fn checks_labels_alone(predicate: &LogicalExpression) -> bool {
    match predicate {
        LogicalExpression::Binary {
            left,
            op: BinaryOp::And,
            right,
        } => checks_labels_alone(left) && checks_labels_alone(right),
        LogicalExpression::FunctionCall {
            callee: Callee::Builtin(BuiltinFunction::HasLabel),
            args,
            ..
        } => args.iter().all(|arg| {
            matches!(
                arg,
                LogicalExpression::Variable(_) | LogicalExpression::Literal(Value::String(_))
            )
        }),
        _ => false,
    }
}

/// The name of `op`'s kind, as [`Uncovered`] reports it.
fn kind_name(op: &LogicalOperator) -> &'static str {
    match op {
        LogicalOperator::Sort(_) => "Sort",
        LogicalOperator::Aggregate(_) => "Aggregate",
        LogicalOperator::Limit(_) => "Limit",
        LogicalOperator::Skip(_) => "Skip",
        LogicalOperator::Project(_) => "Project",
        LogicalOperator::Join(_) => "Join",
        LogicalOperator::LeftJoin(_) => "LeftJoin",
        LogicalOperator::Distinct(_) => "Distinct",
        LogicalOperator::NodeScan(_) => "NodeScanWithInput",
        _ => "Other",
    }
}

impl Recipe {
    /// The kinds of the recipe's steps, top first.
    #[cfg(test)]
    pub(crate) fn kinds(&self) -> Vec<&'static str> {
        let mut kinds = Vec::new();
        self.root.kinds(&mut kinds);
        kinds
    }

    /// The values a call opens this recipe with: the plan's defaults, then `given`; `None`
    /// when one of its parameters has neither, so the call fills the plan as before and fails
    /// as it did.
    pub(crate) fn values_of(
        &self,
        plan: &LogicalPlan,
        given: &HashMap<String, Value>,
    ) -> Option<HashMap<String, Value>> {
        if given
            .values()
            .any(grafeo_common::storage::value_codec::nests_too_deep)
        {
            return None;
        }
        let mut values = plan.default_params.clone();
        values.extend(
            given
                .iter()
                .map(|(name, value)| (name.clone(), value.clone())),
        );
        self.parameters
            .iter()
            .all(|name| values.contains_key(name))
            .then_some(values)
    }

    /// The operators of one call of `plan`, the cached plan this recipe was made from, at the
    /// call's planner and values: `None` when this call is planned as before instead (a key
    /// that is neither an integer nor a string, or one the planner's lookup does not read).
    ///
    /// # Errors
    ///
    /// The errors of building the operators, as planning raises them, and a parameter the
    /// call does not give.
    pub(crate) fn open(
        &self,
        plan: &LogicalPlan,
        ctx: &OpenContext<'_>,
    ) -> Result<Option<PhysicalPlan>> {
        let planner = ctx.planner;
        let _statement =
            super::value_join::StatementScope::enter(&planner.statement_writes, &plan.root);
        planner.check_statement_scope();
        planner.plan_names.borrow_mut().clone_from(&self.names);
        planner.session_context.warnings.raise_all(plan.warnings);
        let Some((operator, columns)) = open_step(&self.root, &plan.root, ctx)? else {
            return Ok(None);
        };
        let operator = planner.shuffled_root(plan, operator);
        Ok(Some(PhysicalPlan { operator, columns }))
    }
}

type Built = (
    Box<dyn grafeo_core::execution::operators::Operator>,
    Vec<String>,
);

/// The operators of `step`, made from `op`, its operator in the cached plan.
fn open_step(step: &Step, op: &LogicalOperator, ctx: &OpenContext<'_>) -> Result<Option<Built>> {
    let planner = ctx.planner;
    let mismatch = || Error::Internal("a recipe does not match its plan".to_string());
    match (step, op) {
        (Step::Return(input), LogicalOperator::Return(ret)) => {
            let Some((input_op, columns)) = open_step(input, &ret.input, ctx)? else {
                return Ok(None);
            };
            planner
                .plan_return_with_input(ret, input_op, columns)
                .map(Some)
        }
        (Step::Expand(input), LogicalOperator::Expand(expand)) => {
            let Some((input_op, columns)) = open_step(input, &expand.input, ctx)? else {
                return Ok(None);
            };
            planner
                .expand_over(expand, None, input_op, columns)
                .map(Some)
        }
        (Step::Filter(input), LogicalOperator::Filter(filter)) => {
            let Some((input_op, columns)) = open_step(input, &filter.input, ctx)? else {
                return Ok(None);
            };
            planner.filter_over(filter, input_op, columns).map(Some)
        }
        (Step::Seek, LogicalOperator::Filter(filter)) => {
            let Some(filled) = with_the_calls_key(filter, ctx.values)? else {
                return Ok(None);
            };
            planner.try_plan_filter_with_property_index(&filled)
        }
        (Step::NodeScan, LogicalOperator::NodeScan(scan)) => planner.plan_node_scan(scan).map(Some),
        _ => Err(mismatch()),
    }
}

/// `filter`, a seek's filter over its input-less scan, with its key filled with the call's
/// value: `None` when that value is neither an integer nor a string, which the planner's
/// lookup reads with rules of their own (the probes of other kinds, a null).
fn with_the_calls_key(
    filter: &FilterOp,
    values: &HashMap<String, Value>,
) -> Result<Option<FilterOp>> {
    let fill = |side: &LogicalExpression| -> Result<Option<LogicalExpression>> {
        let value = match side {
            LogicalExpression::Parameter(name) => match values.get(name) {
                Some(value) => value.clone(),
                None => return Ok(None),
            },
            LogicalExpression::Literal(value) => value.clone(),
            other => return Ok(Some(other.clone())),
        };
        Ok(matches!(value, Value::Int64(_) | Value::String(_))
            .then(|| LogicalExpression::Literal(value)))
    };
    let LogicalExpression::Binary { left, op, right } = &filter.predicate else {
        return Ok(None);
    };
    let (Some(left), Some(right)) = (fill(left)?, fill(right)?) else {
        return Ok(None);
    };
    Ok(Some(FilterOp {
        predicate: LogicalExpression::Binary {
            left: Box::new(left),
            op: *op,
            right: Box::new(right),
        },
        pushdown_hint: None,
        input: filter.input.clone(),
    }))
}
