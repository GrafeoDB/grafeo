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
use super::project::{limit_over, sort_input, takes_top_k};
use crate::query::plan::{
    BinaryOp, BuiltinFunction, Callee, CountClauses, CountExpr, FilterOp, LimitOp,
    LogicalExpression, LogicalOperator, LogicalPlan, SkipOp,
};
use crate::query::planner::PhysicalPlan;

/// The kinds a recipe covers, by name: the set a test pins, so it grows only on purpose.
#[cfg(test)]
pub(crate) const COVERED: [&str; 8] = [
    "Return", "Expand", "Filter", "Seek", "NodeScan", "Sort", "Limit", "Skip",
];

/// A prepared plan's recipe: what the physical planner builds for each operator, and the
/// plan's variable names, which the column names it makes up must skip.
#[derive(Debug)]
pub(crate) struct Recipe {
    root: Step,
    names: HashSet<String>,
    /// The parameters the plan reads: its seek's key and its counts.
    parameters: Vec<String>,
    /// The parameters of its LIMIT and SKIP counts, which a call checks before it opens.
    counts: Vec<String>,
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
    /// An ORDER BY of properties and variables, over the step of its input's plan
    /// ([`sort_input`]: the RETURN's input when the sort adds its keys to that RETURN).
    Sort(Box<Step>),
    /// A LIMIT over its input: over a sort, a top-k when the count takes one
    /// ([`takes_top_k`]).
    Limit(Box<Step>),
    /// A SKIP over its input.
    Skip(Box<Step>),
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
            Self::Sort(_) => "Sort",
            Self::Limit(_) => "Limit",
            Self::Skip(_) => "Skip",
        }
    }

    /// The kinds of this step and the steps below it.
    #[cfg(test)]
    fn kinds(&self, into: &mut Vec<&'static str>) {
        into.push(self.kind());
        if let Self::Return(input)
        | Self::Expand(input)
        | Self::Filter(input)
        | Self::Sort(input)
        | Self::Limit(input)
        | Self::Skip(input) = self
        {
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
    let mut counts = Vec::new();
    parameters_of(&plan.root, &mut parameters, &mut counts);
    Ok(Recipe {
        root,
        names,
        parameters,
        counts,
    })
}

/// The parameters `op` reads, in the places a covered plan reads one: the seek's key, and the
/// counts of its LIMITs and SKIPs, which also go in `counts`.
fn parameters_of(op: &LogicalOperator, parameters: &mut Vec<String>, counts: &mut Vec<String>) {
    match op {
        LogicalOperator::Filter(filter) => {
            if let LogicalExpression::Binary { left, right, .. } = &filter.predicate {
                for side in [left, right] {
                    if let LogicalExpression::Parameter(name) = side.as_ref() {
                        parameters.push(name.clone());
                    }
                }
            }
        }
        LogicalOperator::Limit(LimitOp {
            count: CountExpr::Parameter(name),
            ..
        })
        | LogicalOperator::Skip(SkipOp {
            count: CountExpr::Parameter(name),
            ..
        }) => {
            parameters.push(name.clone());
            counts.push(name.clone());
        }
        _ => {}
    }
    for child in op.children() {
        parameters_of(child, parameters, counts);
    }
}

fn step_of(op: &LogicalOperator) -> std::result::Result<Step, Uncovered> {
    let below = |input: &LogicalOperator| step_of(input).map(Box::new);
    let uncovered = |kind| Err(Uncovered { kind });
    match op {
        LogicalOperator::Return(ret) => {
            if !returns_plainly(ret) {
                return uncovered("ReturnItems");
            }
            Ok(Step::Return(below(&ret.input)?))
        }
        LogicalOperator::Sort(sort) => {
            let plain = |key: &crate::query::plan::SortKey| is_plain(&key.expression);
            if !sort.keys.iter().all(plain) {
                return uncovered("SortKeys");
            }
            // A sort that adds its keys to the RETURN builds that RETURN itself.
            if let LogicalOperator::Return(ret) = sort.input.as_ref()
                && !returns_plainly(ret)
            {
                return uncovered("ReturnItems");
            }
            Ok(Step::Sort(below(sort_input(sort))?))
        }
        LogicalOperator::Limit(limit) => Ok(Step::Limit(below(&limit.input)?)),
        LogicalOperator::Skip(skip) => Ok(Step::Skip(below(&skip.input)?)),
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

/// Whether `expression` is a property or a variable.
fn is_plain(expression: &LogicalExpression) -> bool {
    matches!(
        expression,
        LogicalExpression::Property { .. } | LogicalExpression::Variable(_)
    )
}

/// Whether `ret` returns plain properties and variables, not distinct.
fn returns_plainly(ret: &crate::query::plan::ReturnOp) -> bool {
    !ret.items.is_empty()
        && !ret.distinct
        && ret.items.iter().all(|item| is_plain(&item.expression))
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
        LogicalOperator::Aggregate(_) => "Aggregate",
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
    /// when one of its parameters has neither, or a count is not one (`CountExpr::count_in`),
    /// so the call fills the plan as before and fails as it did, before any other check.
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
        let given_all = self.parameters.iter().all(|name| values.contains_key(name));
        let counts_are_counts = || {
            self.counts
                .iter()
                .all(|name| CountExpr::count_in(&values[name], "").is_ok())
        };
        (given_all && counts_are_counts()).then_some(values)
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
        let Some((operator, columns)) = open_step(&self.root, &plan.root, plan.count_clauses, ctx)?
        else {
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

/// The operators of `step`, made from `op`, its operator in the cached plan, whose counts' errors
/// name `clauses`.
fn open_step(
    step: &Step,
    op: &LogicalOperator,
    clauses: CountClauses,
    ctx: &OpenContext<'_>,
) -> Result<Option<Built>> {
    let planner = ctx.planner;
    let open = |step: &Step, op: &LogicalOperator| open_step(step, op, clauses, ctx);
    // A count with the call's value, as filling the plan resolves it, with its errors.
    let count_of = |count: &CountExpr, clause: &str| -> Result<usize> {
        let mut count = count.clone();
        crate::query::processor::resolve_count_param(&mut count, ctx.values, clause)?;
        Ok(count.value())
    };
    let mismatch = || Error::Internal("a recipe does not match its plan".to_string());
    match (step, op) {
        (Step::Return(input), LogicalOperator::Return(ret)) => {
            let Some((input_op, columns)) = open(input, &ret.input)? else {
                return Ok(None);
            };
            planner
                .plan_return_with_input(ret, input_op, columns)
                .map(Some)
        }
        (Step::Expand(input), LogicalOperator::Expand(expand)) => {
            let Some((input_op, columns)) = open(input, &expand.input)? else {
                return Ok(None);
            };
            planner
                .expand_over(expand, None, input_op, columns)
                .map(Some)
        }
        (Step::Filter(input), LogicalOperator::Filter(filter)) => {
            let Some((input_op, columns)) = open(input, &filter.input)? else {
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
        (Step::Sort(input), LogicalOperator::Sort(sort)) => {
            let Some((input_op, columns)) = open(input, sort_input(sort))? else {
                return Ok(None);
            };
            planner.sort_over(sort, None, input_op, columns).map(Some)
        }
        (Step::Limit(input), LogicalOperator::Limit(limit)) => {
            let count = count_of(&limit.count, clauses.limit())?;
            if let (Step::Sort(sort_step), LogicalOperator::Sort(sort)) =
                (input.as_ref(), limit.input.as_ref())
                && takes_top_k(count)
            {
                let Some((input_op, columns)) = open(sort_step, sort_input(sort))? else {
                    return Ok(None);
                };
                return planner
                    .sort_over(sort, Some(count), input_op, columns)
                    .map(Some);
            }
            let opened = planner.below_limit(limit, count, || open(input, &limit.input))?;
            Ok(opened.map(|(input_op, columns)| limit_over(limit, count, input_op, columns)))
        }
        (Step::Skip(input), LogicalOperator::Skip(skip)) => {
            let Some((input_op, columns)) = open(input, &skip.input)? else {
                return Ok(None);
            };
            let count = count_of(&skip.count, clauses.skip())?;
            Ok(Some(crate::query::planner::common::build_skip(
                input_op, columns, count,
            )))
        }
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
