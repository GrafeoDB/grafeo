//! The order of a pattern's inner joins, for the property graph.
//!
//! One search over connected sets of relations, smallest first. Each connected
//! pair of sets considers a hash join; where one side is a single hop and the
//! other binds one of its ends, also an expand from that end; where the other
//! binds both ends, a hop that closes on the bound node. So a plan may hash two
//! selective chains and then expand. The memo holds choices with their rows and
//! cost, and the operator tree is built once, for the winner.
//!
//! A filter that may raise stays with its relation: over the hop, after every check
//! the hop makes, in the order written. Its hop then runs on fewer rows than the
//! written plan's, so it meets no row that plan would not, and an order may start
//! from a selective filter whatever it can raise. The one difference: an empty other
//! side lets a written plan end before it evaluates the filter.
//!
//! Ties break by the written order, so a query gets the same plan on every run.

use std::cell::RefCell;
use std::collections::{HashMap, HashSet};

use super::closing_hops::compared_variables;
use super::cost::measured;
use super::cycles::{FreshNames, same};
use super::{IncomingEdges, Optimizer, PlannerRule, any_operator, cannot_raise, movable_variables};
use grafeo_core::graph::Direction;

use crate::query::plan::{
    BuiltinFunction, Callee, ExpandDirection, ExpandOp, FilterOp, JoinCondition, JoinOp, JoinType,
    LogicalExpression, LogicalOperator, NodeScanOp, PathMode,
};
use crate::query::planner::lpg::scan::label_check;
use crate::query::planner::lpg::seek::split_conjuncts;

/// The most relations a search takes: past this the written order stands.
const MAX_RELATIONS: usize = 12;

/// One input of the join tree.
struct Relation {
    /// The relation as it runs when it comes first and does not take part as a
    /// hop.
    operator: LogicalOperator,
    /// The ids of the variables it binds, as bits.
    binds: u64,
    /// The relation as a single hop, when it is one.
    hop: Option<Hop>,
}

/// A relation that is one hop from a node scan.
struct Hop {
    source: String,
    target: String,
    edge: Option<String>,
    edge_types: Vec<String>,
    direction: ExpandDirection,
    path_mode: PathMode,
    /// The label its scan reads.
    source_label: Option<String>,
    /// The label a check requires of the target, which a scan of it can read.
    target_label: Option<String>,
    /// Whether the labels above are the hop's own, not another relation's: only
    /// its own need a check, and a relation that has one checks it.
    own_labels: (bool, bool),
    /// The checks of the source alone, in the order they run.
    source_checks: Vec<LogicalExpression>,
    /// The checks of the target alone, without `target_label`'s.
    target_checks: Vec<LogicalExpression>,
    /// The other checks, which read the edge or both ends.
    hop_checks: Vec<LogicalExpression>,
    /// The checks that may raise, on the source alone with nothing but the source
    /// read before them, in the order they run: where the hop runs from its source,
    /// they keep their written place below the expand.
    source_raising: Vec<LogicalExpression>,
    /// The other checks that may raise, in the order they run.
    raising: Vec<LogicalExpression>,
    /// What a source row makes.
    forward_factor: PerRow,
    /// What a target row makes, when the store can read that direction.
    backward_factor: Option<PerRow>,
    /// The hop's rows and cost when it comes first, from its source.
    forward_leaf: (f64, f64),
    /// The same from its target, when the store can read that direction.
    backward_leaf: Option<(f64, f64)>,
}

/// What a hop makes of a row it runs from.
#[derive(Clone, Copy)]
struct PerRow {
    /// The lists it asks for: the share of rows past the checks at its start.
    calls: f64,
    /// The edges it reads, before the checks of its far end drop any.
    read: f64,
    /// The rows it keeps.
    kept: f64,
}

/// A condition other than the equality of one variable on both sides.
struct ForeignCondition {
    left_relation_binds: u64,
    right_relation_binds: u64,
    condition: JoinCondition,
}

#[derive(Clone, Copy)]
enum Choice {
    /// A relation alone, from its target when `backward`.
    Leaf { relation: usize, backward: bool },
    /// The right side builds, the left probes.
    Hash { left: usize, right: usize },
    /// A hop over the plan of `input`, from its target when `backward`.
    /// `closes`: the input binds both ends.
    Step {
        input: usize,
        relation: usize,
        backward: bool,
        closes: bool,
    },
}

#[derive(Clone, Copy)]
struct Plan {
    rows: f64,
    cost: f64,
    choice: Choice,
}

/// The names a plan's searches share.
pub(super) struct PlanNames<'a> {
    /// The variables made for the checks that close a cycle.
    closing: &'a HashSet<String>,
    /// New names, none a variable of the whole plan has and none given twice: a name
    /// made in one tree must not meet a variable bound above it.
    fresh: RefCell<FreshNames>,
}

impl<'a> PlanNames<'a> {
    /// The plan's `fresh` names, whose `closing` variables were made for such checks.
    pub(super) fn new(closing: &'a HashSet<String>, fresh: FreshNames) -> Self {
        Self {
            closing,
            fresh: RefCell::new(fresh),
        }
    }
}

/// What the search reads.
struct Search<'a> {
    relations: Vec<Relation>,
    foreign: Vec<ForeignCondition>,
    /// The variables by id.
    variables: Vec<String>,
    /// The nodes a variable can be, by id.
    domains: Vec<f64>,
    /// Whether a closing hop over another hop can run with it as one intersection.
    intersects: bool,
    /// Per hop and way it runs (from its source, from its target), what it meets at the
    /// ends other hops reach, with the hop that reaches them: read for a comma pattern.
    reached: Vec<[Vec<(usize, f64)>; 2]>,
    fresh: &'a RefCell<FreshNames>,
}

/// The cheapest order of the inner join tree `op`, whose columns the caller reads
/// by name, with the filters that read several relations above it. `None` when `op`
/// is no tree the search can take, which then stays as written:
///
/// - a relation other than scans, expands and joins, or one that writes;
/// - a condition that reads anything but variables and properties;
/// - a filter that must stay where it is (see `stays_in_place`);
/// - a filter that may raise inside a relation that is no hop;
/// - a filter that may raise when `stops_early`: the written order may stop before
///   the row that raises;
/// - no connected way to join every relation, or too many relations.
pub(super) fn best_order(
    optimizer: &Optimizer,
    op: &LogicalOperator,
    plan_names: &PlanNames<'_>,
    stops_early: bool,
) -> Option<LogicalOperator> {
    let closing_names = plan_names.closing;
    if stops_early && holds_raising_filter(op, closing_names) {
        return None;
    }
    let mut inputs: Vec<LogicalOperator> = Vec::new();
    let mut conditions = Vec::new();
    let mut over_joins = Vec::new();
    // The relations of a comma pattern each run whole in the written plan. The hops of
    // a path do so only from its start: a later hop's checks see the rows that reach it.
    let is_comma_pattern = matches!(op, LogicalOperator::Join(_));
    match op {
        LogicalOperator::Join(_) => {
            let mut parts = Vec::new();
            collect_inner_joins(op, &mut parts, &mut conditions, &mut over_joins)?;
            inputs.extend(parts.into_iter().cloned());
            // Comma patterns sample their degrees; a single path keeps the counts.
            optimizer.cost_model.hop_degrees()?.start_sampling();
        }
        _ => inputs = cycle_hops(op, closing_names)?,
    }
    if inputs.len() < 2 || inputs.len() > MAX_RELATIONS {
        return None;
    }

    let mut names: HashMap<String, usize> = HashMap::new();
    let mut variables: Vec<String> = Vec::new();
    let mut lifted = Vec::new();
    let mut relations = Vec::with_capacity(inputs.len());
    for (position, input) in inputs.into_iter().enumerate() {
        let bound = super::bound_columns(&input)?;
        if input.has_mutations() || bound.len() > 64 {
            return None;
        }
        let mut sorted: Vec<&String> = bound.iter().collect();
        sorted.sort_unstable();
        let mut binds = 0_u64;
        for name in sorted {
            let next = variables.len();
            let id = *names.entry(name.clone()).or_insert(next);
            if id == next {
                variables.push(name.clone());
            }
            binds |= 1_u64.checked_shl(u32::try_from(id).ok()?)?;
        }
        let (hop, kept) = parse_hop(&input, is_comma_pattern || position == 0, &mut lifted)?;
        if hop.is_none() && !runs_anywhere(&input) {
            return None;
        }
        relations.push(Relation {
            operator: kept,
            binds,
            hop,
        });
    }

    // The join equates a variable across relations, so the label one of them
    // knows holds in the others too, and tells a hop's degree from that end.
    let mut labels: HashMap<String, String> = HashMap::new();
    for hop in relations
        .iter()
        .filter_map(|relation| relation.hop.as_ref())
    {
        for (variable, label) in [
            (&hop.source, &hop.source_label),
            (&hop.target, &hop.target_label),
        ] {
            if let Some(label) = label {
                labels
                    .entry(variable.clone())
                    .or_insert_with(|| label.clone());
            }
        }
    }
    for hop in relations
        .iter_mut()
        .filter_map(|relation| relation.hop.as_mut())
    {
        hop.price(&labels, optimizer);
    }

    let reached = if is_comma_pattern {
        reached_degrees(&relations, optimizer)
    } else {
        Vec::new()
    };

    let mut foreign = Vec::new();
    for condition in conditions {
        foreign.push(foreign_condition(condition, &relations, &names)?);
    }

    let degrees = optimizer.cost_model.hop_degrees()?;
    let domains = variables
        .iter()
        .map(|name| degrees.domain(name, op))
        .collect();
    let mut search = Search {
        relations,
        foreign,
        variables,
        domains,
        intersects: optimizer.rules.has(PlannerRule::ExpandInto)
            && optimizer.rules.has(PlannerRule::IntersectExpand)
            && optimizer.incoming_edges == IncomingEdges::Kept,
        reached,
        fresh: &plan_names.fresh,
    };
    // Filters over a join inside the tree go above it too: later, so they overtake
    // nothing.
    lifted.extend(over_joins);
    let memo = search.run(optimizer);
    let full = (1_usize << search.relations.len()) - 1;
    memo[full]?;

    let tree = search.build(&memo, full);
    let ordered = match LogicalExpression::conjunction(lifted) {
        None => tree,
        // A selective filter may now run late: the written tree may be cheaper.
        Some(predicate) => {
            let lifted_tree = LogicalOperator::Filter(FilterOp {
                predicate,
                pushdown_hint: None,
                input: Box::new(tree),
            });
            let cost = |plan: &LogicalOperator| {
                optimizer
                    .cost_model
                    .estimate_tree(plan, &optimizer.card_estimator)
                    .total()
            };
            if cost(&lifted_tree) >= cost(op) {
                return None;
            }
            lifted_tree
        }
    };
    Some(ordered)
}

/// The second stage's reads for the hops of `relations`, in written order, each with each
/// hop that reaches one of its ends.
fn reached_degrees(relations: &[Relation], optimizer: &Optimizer) -> Vec<[Vec<(usize, f64)>; 2]> {
    let Some(degrees) = optimizer.cost_model.hop_degrees() else {
        return Vec::new();
    };
    let mut reached: Vec<[Vec<(usize, f64)>; 2]> =
        relations.iter().map(|_| [Vec::new(), Vec::new()]).collect();
    for (index, on) in relations.iter().enumerate() {
        let Some(on) = &on.hop else {
            continue;
        };
        let on_types: Vec<&str> = on.edge_types.iter().map(String::as_str).collect();
        for (way, (end, direction)) in
            [(&on.source, on.direction), (&on.target, flip(on.direction))]
                .into_iter()
                .enumerate()
        {
            for (other, via) in relations.iter().enumerate() {
                let Some(via) = via.hop.as_ref().filter(|_| other != index) else {
                    continue;
                };
                let (start_label, via_direction) = if via.target == *end {
                    (&via.source_label, via.direction)
                } else if via.source == *end {
                    (&via.target_label, flip(via.direction))
                } else {
                    continue;
                };
                let via_types: Vec<&str> = via.edge_types.iter().map(String::as_str).collect();
                if let Some(degree) = degrees.reached(
                    start_label.as_deref(),
                    &via_types,
                    store_direction(via_direction),
                    &on_types,
                    store_direction(direction),
                ) {
                    reached[index][way].push((other, degree));
                }
            }
        }
    }
    reached
}

/// `direction` as the store names it.
fn store_direction(direction: ExpandDirection) -> Direction {
    match direction {
        ExpandDirection::Outgoing => Direction::Outgoing,
        ExpandDirection::Incoming => Direction::Incoming,
        ExpandDirection::Both => Direction::Both,
    }
}

/// The inputs and conditions of the inner joins at the top of `op`, and the
/// conjuncts of filters over a join inside it, in the order they run. A condition
/// on one variable on both sides is the variables' sharing, and needs no entry.
/// `None` for a condition the search cannot read, or a filter that must stay.
fn collect_inner_joins<'a>(
    op: &'a LogicalOperator,
    inputs: &mut Vec<&'a LogicalOperator>,
    conditions: &mut Vec<&'a JoinCondition>,
    over_joins: &mut Vec<LogicalExpression>,
) -> Option<()> {
    match op {
        LogicalOperator::Filter(filter)
            if matches!(
                filter.input.as_ref(),
                LogicalOperator::Join(join) if join.join_type == JoinType::Inner
            ) =>
        {
            collect_inner_joins(&filter.input, inputs, conditions, over_joins)?;
            let mut conjuncts = Vec::new();
            split_conjuncts(&filter.predicate, &mut conjuncts);
            for conjunct in conjuncts {
                movable_variables(conjunct)?;
                over_joins.push(conjunct.clone());
            }
        }
        LogicalOperator::Join(join) if join.join_type == JoinType::Inner => {
            collect_inner_joins(&join.left, inputs, conditions, over_joins)?;
            collect_inner_joins(&join.right, inputs, conditions, over_joins)?;
            for condition in &join.conditions {
                match (&condition.left, &condition.right) {
                    (LogicalExpression::Variable(a), LogicalExpression::Variable(b)) if a == b => {}
                    (left, right) => {
                        read_variable(left)?;
                        read_variable(right)?;
                        conditions.push(condition);
                    }
                }
            }
        }
        other => inputs.push(other),
    }
    Some(())
}

/// The hops of `op`, a path of single hops from a node scan that comes back to a
/// node it binds, as relations of their own, each from a scan of its source, its
/// closing check taken up by binding the node itself. Its other filters go with the
/// hop they stand over, and the search lifts what a hop alone cannot read. `None`
/// for anything else.
fn cycle_hops(
    op: &LogicalOperator,
    closing_names: &HashSet<String>,
) -> Option<Vec<LogicalOperator>> {
    let mut hops: Vec<(ExpandOp, Vec<LogicalExpression>)> = Vec::new();
    let mut above = Vec::new();
    let mut at = op;
    loop {
        match at {
            LogicalOperator::Filter(filter) => {
                if filter.pushdown_hint.is_some() {
                    return None;
                }
                let mut conjuncts = Vec::new();
                split_conjuncts(&filter.predicate, &mut conjuncts);
                above.extend(conjuncts.into_iter().cloned());
                at = &filter.input;
            }
            LogicalOperator::Expand(expand) => {
                if expand.is_variable_length()
                    || expand.path_alias.is_some()
                    || expand.path_mode != PathMode::Walk
                    || hops
                        .last()
                        .is_some_and(|(upper, _)| upper.from_variable != expand.to_variable)
                {
                    return None;
                }
                hops.push((expand.clone(), std::mem::take(&mut above)));
                at = &expand.input;
            }
            LogicalOperator::NodeScan(scan) if scan.input.is_none() => break,
            _ => return None,
        }
    }
    if hops.len() < 2 {
        return None;
    }
    // The filters over the scan check the first hop's source.
    let LogicalOperator::NodeScan(scan) = at else {
        return None;
    };
    let source_checks = above;
    let mut closed = false;
    let bound_before: Vec<String> = std::iter::once(scan.variable.clone())
        .chain(hops.iter().rev().map(|(hop, _)| hop.to_variable.clone()))
        .collect();
    let mut relations = Vec::with_capacity(hops.len());
    for (position, (mut hop, mut checks)) in hops.into_iter().rev().enumerate() {
        // A closing check over this hop: the hop binds the node itself.
        // Only a name made for that check: a name of the query's may be read above.
        if closing_names.contains(&hop.to_variable)
            && let Some(index) = checks.iter().position(|check| {
                closing_pair(check, &hop.to_variable)
                    .is_some_and(|target| bound_before[..=position].contains(&target))
            })
            && let Some(target) = closing_pair(&checks[index], &hop.to_variable)
        {
            checks.remove(index);
            hop.to_variable = target;
            closed = true;
        }
        let source = LogicalOperator::NodeScan(NodeScanOp {
            variable: hop.from_variable.clone(),
            label: (position == 0).then(|| scan.label.clone()).flatten(),
            input: None,
        });
        hop.input = Box::new(filtered(
            source,
            if position == 0 {
                source_checks.clone()
            } else {
                Vec::new()
            },
        ));
        relations.push(filtered(LogicalOperator::Expand(hop), checks));
    }
    closed.then_some(relations)
}

/// The other variable of `variable = other`, or of `id(variable) = id(other)`.
fn closing_pair(check: &LogicalExpression, variable: &str) -> Option<String> {
    let (a, b) = compared_variables(check)?;
    if a == variable {
        Some(b.to_string())
    } else if b == variable {
        Some(a.to_string())
    } else {
        None
    }
}

/// The variable `expression` reads, when it reads one directly.
fn read_variable(expression: &LogicalExpression) -> Option<&str> {
    match expression {
        LogicalExpression::Variable(variable) => Some(variable),
        LogicalExpression::Property { variable, .. } => Some(variable),
        _ => None,
    }
}

/// `condition` with the relations that bind its two sides, which must differ.
fn foreign_condition(
    condition: &JoinCondition,
    relations: &[Relation],
    names: &HashMap<String, usize>,
) -> Option<ForeignCondition> {
    let binding = |expression: &LogicalExpression| {
        let id = *names.get(read_variable(expression)?)?;
        let bit = 1_u64 << id;
        let holders: Vec<u64> = relations
            .iter()
            .map(|relation| relation.binds)
            .filter(|binds| binds & bit != 0)
            .collect();
        match holders.as_slice() {
            [binds] => Some(*binds),
            _ => None,
        }
    };
    let (left, right) = (binding(&condition.left)?, binding(&condition.right)?);
    (left != right).then(|| ForeignCondition {
        left_relation_binds: left,
        right_relation_binds: right,
        condition: condition.clone(),
    })
}

/// Whether a filter in `op` may raise, the checks made to close a cycle aside: they
/// compare two nodes.
fn holds_raising_filter(op: &LogicalOperator, closing_names: &HashSet<String>) -> bool {
    let closes_a_cycle = |conjunct: &LogicalExpression| {
        compared_variables(conjunct)
            .is_some_and(|(a, b)| closing_names.contains(a) || closing_names.contains(b))
    };
    any_operator(op, &|op| {
        let LogicalOperator::Filter(filter) = op else {
            return false;
        };
        let mut conjuncts = Vec::new();
        split_conjuncts(&filter.predicate, &mut conjuncts);
        conjuncts
            .into_iter()
            .any(|conjunct| !cannot_raise(conjunct) && !closes_a_cycle(conjunct))
    })
}

/// Whether `op` is made of scans, expands, joins and checks that cannot raise, so
/// it may run ahead of the checks of other relations.
fn runs_anywhere(op: &LogicalOperator) -> bool {
    match op {
        LogicalOperator::NodeScan(scan) => scan.input.as_deref().is_none_or(runs_anywhere),
        LogicalOperator::EdgeScan(scan) => scan.input.as_deref().is_none_or(runs_anywhere),
        LogicalOperator::Expand(expand) => runs_anywhere(&expand.input),
        LogicalOperator::Filter(filter) => {
            cannot_raise(&filter.predicate) && runs_anywhere(&filter.input)
        }
        LogicalOperator::Join(join) => runs_anywhere(&join.left) && runs_anywhere(&join.right),
        _ => false,
    }
}

/// `op` as a hop and as it runs alone, when it is one hop from a scan without
/// input: its checks sorted by what they read, and those that cannot stay with it
/// added to `lifted` in the order they run: what reads other relations, and with
/// `keeps_raising` false what may raise. What may raise and comes after a lifted check
/// goes too, so it overtakes none. `Some((None, op))` for another relation, and `None`
/// for a check that must stay where it is.
fn parse_hop(
    op: &LogicalOperator,
    keeps_raising: bool,
    lifted: &mut Vec<LogicalExpression>,
) -> Option<(Option<Hop>, LogicalOperator)> {
    let mut above = Vec::new();
    let mut below = op;
    while let LogicalOperator::Filter(filter) = below {
        above.push(&filter.predicate);
        below = &filter.input;
    }
    let LogicalOperator::Expand(expand) = below else {
        return Some((None, op.clone()));
    };
    let mut under = Vec::new();
    let mut source_op = expand.input.as_ref();
    while let LogicalOperator::Filter(filter) = source_op {
        under.push(&filter.predicate);
        source_op = &filter.input;
    }
    let LogicalOperator::NodeScan(scan) = source_op else {
        return Some((None, op.clone()));
    };
    if scan.input.is_some()
        || expand.is_variable_length()
        || expand.path_alias.is_some()
        || expand.from_variable != scan.variable
        || expand.from_variable == expand.to_variable
    {
        return Some((None, op.clone()));
    }

    let mut hop = Hop {
        source: expand.from_variable.clone(),
        target: expand.to_variable.clone(),
        edge: expand.edge_variable.clone(),
        edge_types: expand.edge_types.clone(),
        direction: expand.direction,
        path_mode: expand.path_mode,
        source_label: scan.label.clone(),
        target_label: None,
        own_labels: (false, false),
        source_checks: Vec::new(),
        target_checks: Vec::new(),
        hop_checks: Vec::new(),
        source_raising: Vec::new(),
        raising: Vec::new(),
        forward_factor: PerRow {
            calls: 1.0,
            read: 1.0,
            kept: 1.0,
        },
        backward_factor: None,
        forward_leaf: (1.0, 0.0),
        backward_leaf: None,
    };
    // In the order they run: the bottom filter first.
    let predicates = under.iter().rev().chain(above.iter().rev());
    let mut conjuncts = Vec::new();
    for predicate in predicates {
        split_conjuncts(predicate, &mut conjuncts);
    }
    let mut any_lifted = false;
    let mut only_source_read = true;
    for conjunct in conjuncts {
        let reads = movable_variables(conjunct)?;
        let on_source = reads.iter().all(|name| *name == hop.source);
        let reads_others = !reads.iter().all(|name| {
            *name == hop.source || *name == hop.target || hop.edge.as_ref() == Some(name)
        });
        let raises = !cannot_raise(conjunct);
        only_source_read &= on_source;
        if reads_others || (raises && (!keeps_raising || any_lifted)) {
            // It runs above them all.
            any_lifted = true;
            lifted.push(conjunct.clone());
        } else if raises {
            if only_source_read {
                hop.source_raising.push(conjunct.clone());
            } else {
                hop.raising.push(conjunct.clone());
            }
        } else if on_source {
            hop.source_checks.push(conjunct.clone());
        } else if reads.iter().all(|name| *name == hop.target) {
            match label_check(conjunct, &hop.target) {
                Some(label) if hop.target_label.is_none() => {
                    hop.target_label = Some(label.to_string());
                }
                _ => hop.target_checks.push(conjunct.clone()),
            }
        } else {
            hop.hop_checks.push(conjunct.clone());
        }
    }

    hop.own_labels = (hop.source_label.is_some(), hop.target_label.is_some());
    let forward = hop.leaf(false);
    Some((Some(hop), forward))
}

/// The rows of a relation the search takes whole, and its cost in measured time.
fn priced_whole(op: &LogicalOperator, optimizer: &Optimizer) -> (f64, f64) {
    (
        optimizer.card_estimator.estimate(op),
        optimizer
            .cost_model
            .estimate_tree(op, &optimizer.card_estimator)
            .total()
            * measured::OLD_UNIT,
    )
}

/// `direction` seen from the other end.
fn flip(direction: ExpandDirection) -> ExpandDirection {
    match direction {
        ExpandDirection::Outgoing => ExpandDirection::Incoming,
        ExpandDirection::Incoming => ExpandDirection::Outgoing,
        ExpandDirection::Both => ExpandDirection::Both,
    }
}

/// `input` under the `checks`, when there are any.
fn filtered(input: LogicalOperator, checks: Vec<LogicalExpression>) -> LogicalOperator {
    match LogicalExpression::conjunction(checks) {
        Some(predicate) => LogicalOperator::Filter(FilterOp {
            predicate,
            pushdown_hint: None,
            input: Box::new(input),
        }),
        None => input,
    }
}

/// `input` under `checks`, then under each of `raising` in its own filter, so a row a
/// check leaves null goes no further.
fn checked(
    input: LogicalOperator,
    checks: Vec<LogicalExpression>,
    raising: &[LogicalExpression],
) -> LogicalOperator {
    raising
        .iter()
        .fold(filtered(input, checks), |input, check| {
            filtered(input, vec![check.clone()])
        })
}

/// Whether the hop checks a label on the far end it reaches, from its target when
/// `backward`.
fn far_label(hop: &Hop, backward: bool) -> bool {
    if backward {
        hop.own_labels.0
    } else {
        hop.own_labels.1
    }
}

/// The check that `variable` has `label`.
fn has_label(variable: &str, label: &str) -> LogicalExpression {
    LogicalExpression::FunctionCall {
        callee: Callee::Builtin(BuiltinFunction::HasLabel),
        name: "hasLabel".to_string(),
        args: vec![
            LogicalExpression::Variable(variable.to_string()),
            LogicalExpression::Literal(label.into()),
        ],
        distinct: false,
    }
}

impl Hop {
    /// Prices the hop, taking the `labels` the join's other relations know of its
    /// ends as its own: the inner join keeps no row without them, so a scan of the
    /// label gives the same rows, and the label tells the hop's degree from that end.
    fn price(&mut self, labels: &HashMap<String, String>, optimizer: &Optimizer) {
        if self.source_label.is_none() {
            self.source_label = labels.get(&self.source).cloned();
        }
        if self.target_label.is_none() {
            self.target_label = labels.get(&self.target).cloned();
        }
        // One relation has one number of rows, read from either end: the smaller,
        // which holds the label check of the other end. A set's rows then do not
        // depend on its order, which its cheapest plan alone stands for.
        let rows_of = |op: &LogicalOperator| optimizer.card_estimator.estimate(op);
        // A list holds what the checks of its far end go on to drop, and only the rows
        // past the checks of the start ask for one.
        let read_from = |backward: bool| {
            let scan = self.scan_of(backward);
            let scanned = rows_of(&scan).max(1.0);
            let start = self.at_the_start(scan, backward, false);
            let calls = rows_of(&start);
            let far_end = if backward { &self.source } else { &self.target };
            let expanded = self.expand(start, backward, far_end.clone());
            let read = rows_of(&expanded);
            let kept = rows_of(&self.after_the_hop(expanded, backward, Vec::new()));
            (scanned, calls, read, kept)
        };
        let forward = read_from(false);
        let readable = flip(self.direction) != ExpandDirection::Incoming
            || optimizer.incoming_edges == IncomingEdges::Kept;
        let backward = readable.then(|| read_from(true));
        let rows = backward.map_or(forward.3, |(.., kept)| kept.min(forward.3));
        let priced = |(scanned, calls, read, _): (f64, f64, f64, f64), backward: bool| {
            let cost = scanned * measured::SCAN_ROW
                + measured::hop(calls, read, rows, far_label(self, backward));
            (
                (rows, cost),
                PerRow {
                    calls: calls / scanned,
                    read: read / scanned,
                    kept: rows / scanned,
                },
            )
        };
        let forward = priced(forward, false);
        let backward = backward.map(|read| priced(read, true));
        (self.forward_leaf, self.forward_factor) = forward;
        self.backward_leaf = backward.map(|(leaf, _)| leaf);
        self.backward_factor = backward.map(|(_, factor)| factor);
    }

    /// The scan the hop starts from.
    fn scan_of(&self, backward: bool) -> LogicalOperator {
        let (variable, label) = if backward {
            (&self.target, &self.target_label)
        } else {
            (&self.source, &self.source_label)
        };
        LogicalOperator::NodeScan(NodeScanOp {
            variable: variable.clone(),
            label: label.clone(),
            input: None,
        })
    }

    /// The checks of the end the hop runs from or to, with its own label when
    /// `with_label`: a bound end has no scan to read the label for it.
    fn checks_of_end(&self, target: bool, with_label: bool) -> Vec<LogicalExpression> {
        let (variable, label, own, checks) = if target {
            (
                &self.target,
                &self.target_label,
                self.own_labels.1,
                &self.target_checks,
            )
        } else {
            (
                &self.source,
                &self.source_label,
                self.own_labels.0,
                &self.source_checks,
            )
        };
        let label = label
            .iter()
            .filter(|_| with_label && own)
            .map(|label| has_label(variable, label));
        label.chain(checks.iter().cloned()).collect()
    }

    /// The hop over `input`, from its target when `backward`, binding `to`.
    fn expand(&self, input: LogicalOperator, backward: bool, to: String) -> LogicalOperator {
        let (from, direction) = if backward {
            (&self.target, flip(self.direction))
        } else {
            (&self.source, self.direction)
        };
        LogicalOperator::Expand(ExpandOp {
            from_variable: from.clone(),
            to_variable: to,
            edge_variable: self.edge.clone(),
            direction,
            edge_types: self.edge_types.clone(),
            min_hops: 1,
            max_hops: Some(1),
            input: Box::new(input),
            path_alias: None,
            path_mode: self.path_mode,
            quantified: false,
        })
    }

    /// `input` under the checks of the end the hop runs from.
    fn at_the_start(
        &self,
        input: LogicalOperator,
        backward: bool,
        with_label: bool,
    ) -> LogicalOperator {
        let raising: &[LogicalExpression] = if backward { &[] } else { &self.source_raising };
        checked(input, self.checks_of_end(backward, with_label), raising)
    }

    /// The `expanded` hop under `first`, its other checks and last what may raise.
    fn after_the_hop(
        &self,
        expanded: LogicalOperator,
        backward: bool,
        first: Vec<LogicalExpression>,
    ) -> LogicalOperator {
        let mut after = first;
        // A far end has no scan to read its label, bound or not.
        after.extend(self.checks_of_end(!backward, true));
        after.extend(self.hop_checks.iter().cloned());
        let source_raising: &[LogicalExpression] =
            if backward { &self.source_raising } else { &[] };
        checked(
            checked(expanded, after, source_raising),
            Vec::new(),
            &self.raising,
        )
    }

    /// The hop as it runs first.
    fn leaf(&self, backward: bool) -> LogicalOperator {
        let start = self.at_the_start(self.scan_of(backward), backward, false);
        let far_end = if backward { &self.source } else { &self.target };
        let expanded = self.expand(start, backward, far_end.clone());
        self.after_the_hop(expanded, backward, Vec::new())
    }

    /// The hop over the rows of `input`, which bind the end it runs from, and
    /// the far end too when `closes`.
    fn step(
        &self,
        input: LogicalOperator,
        backward: bool,
        closes: bool,
        fresh: &mut FreshNames,
    ) -> LogicalOperator {
        let far_end = if backward { &self.source } else { &self.target };
        let start = self.at_the_start(input, backward, true);
        let mut first = Vec::new();
        let to = if closes {
            let name = fresh.next("_cycle_end");
            first.push(same(&name, far_end.clone()));
            name
        } else {
            far_end.clone()
        };
        let expanded = self.expand(start, backward, to);
        self.after_the_hop(expanded, backward, first)
    }
}

impl Search<'_> {
    /// The cheapest plan of each connected set of relations, by the bits of its
    /// relations. Sets run in ascending order, so a set's parts are done first.
    fn run(&self, optimizer: &Optimizer) -> Vec<Option<Plan>> {
        let count = self.relations.len();
        let sets = 1_usize << count;
        let connected = self.connected_sets();
        let mut binds = vec![0_u64; sets];
        for set in 1..sets {
            let lowest = set.trailing_zeros() as usize;
            binds[set] = binds[set & (set - 1)] | self.relations[lowest].binds;
        }
        let mut memo: Vec<Option<Plan>> = vec![None; sets];
        for (index, relation) in self.relations.iter().enumerate() {
            memo[1 << index] = Some(self.leaf_plan(index, relation, optimizer));
        }
        for set in 1..sets {
            if set.count_ones() < 2 || !connected[set] {
                continue;
            }
            let mut best: Option<Plan> = None;
            let mut left = (set - 1) & set;
            while left > 0 {
                let right = set ^ left;
                if let (Some(left_plan), Some(right_plan)) = (memo[left], memo[right])
                    && (binds[left] & binds[right] != 0 || self.foreign_between(left, right))
                {
                    keep_cheaper(
                        &mut best,
                        self.hash(left, right, left_plan, right_plan, &binds),
                    );
                    if right.is_power_of_two() {
                        let relation = right.trailing_zeros() as usize;
                        for step in self.steps(left, left_plan, relation, &binds, &memo) {
                            keep_cheaper(&mut best, step);
                        }
                    }
                }
                left = (left - 1) & set;
            }
            memo[set] = best;
        }
        memo
    }

    /// Which sets of relations connect by shared variables or conditions.
    fn connected_sets(&self) -> Vec<bool> {
        let count = self.relations.len();
        let touches = |a: usize, b: usize| {
            self.relations[a].binds & self.relations[b].binds != 0
                || self.foreign_between(1 << a, 1 << b)
        };
        let neighbors: Vec<usize> = (0..count)
            .map(|a| {
                (0..count)
                    .filter(|&b| b != a && touches(a, b))
                    .fold(0, |set, b| set | (1 << b))
            })
            .collect();
        (0..1_usize << count)
            .map(|set| {
                if set == 0 {
                    return false;
                }
                let mut reached = set.isolate_lowest_one();
                loop {
                    let mut next = reached;
                    for (index, around) in neighbors.iter().enumerate() {
                        if reached & (1 << index) != 0 {
                            next |= around & set;
                        }
                    }
                    if next == reached {
                        return reached == set;
                    }
                    reached = next;
                }
            })
            .collect()
    }

    /// The bits of the variables a set of relations binds.
    fn binds_of(&self, set: usize) -> u64 {
        self.relations
            .iter()
            .enumerate()
            .filter(|(index, _)| set & (1 << index) != 0)
            .fold(0, |binds, (_, relation)| binds | relation.binds)
    }

    /// Whether a condition other than a shared variable connects the two sets.
    fn foreign_between(&self, left: usize, right: usize) -> bool {
        if self.foreign.is_empty() {
            return false;
        }
        let (left, right) = (self.binds_of(left), self.binds_of(right));
        self.foreign.iter().any(|condition| {
            let (a, b) = (
                condition.left_relation_binds,
                condition.right_relation_binds,
            );
            (a & left == a && b & right == b) || (b & left == b && a & right == a)
        })
    }

    fn leaf_plan(&self, index: usize, relation: &Relation, optimizer: &Optimizer) -> Plan {
        let leaf = |backward, (rows, cost)| Plan {
            rows,
            cost,
            choice: Choice::Leaf {
                relation: index,
                backward,
            },
        };
        match &relation.hop {
            None => {
                let (rows, cost) = priced_whole(&relation.operator, optimizer);
                leaf(false, (rows, cost))
            }
            Some(hop) => {
                let mut best = Some(leaf(false, hop.forward_leaf));
                if let Some(backward) = hop.backward_leaf {
                    keep_cheaper(&mut best, leaf(true, backward));
                }
                best.unwrap_or_else(|| leaf(false, hop.forward_leaf))
            }
        }
    }

    /// The nodes a variable may be, which bounds its distinct values.
    fn domain_of_bits(&self, bits: u64) -> impl Iterator<Item = f64> + '_ {
        self.domains
            .iter()
            .enumerate()
            .filter(move |(id, _)| bits & (1_u64 << id) != 0)
            .map(|(_, domain)| *domain)
    }

    fn hash(
        &self,
        left: usize,
        right: usize,
        left_plan: Plan,
        right_plan: Plan,
        binds: &[u64],
    ) -> Plan {
        let mut rows = left_plan.rows * right_plan.rows;
        for domain in self.domain_of_bits(binds[left] & binds[right]) {
            let distinct = left_plan.rows.min(domain).max(right_plan.rows.min(domain));
            rows /= distinct.max(1.0);
        }
        if self.foreign_between(left, right) {
            rows *= 0.1;
        }
        let rows = rows.max(1.0);
        Plan {
            rows,
            cost: left_plan.cost
                + right_plan.cost
                + measured::hash_join(
                    right_plan.rows,
                    left_plan.rows,
                    rows,
                    (binds[left] | binds[right]).count_ones(),
                ),
            choice: Choice::Hash { left, right },
        }
    }

    /// The ways to run the hop `relation` over the plan of `input`, which shares
    /// no edge variable with it and no other condition.
    fn steps(
        &self,
        input: usize,
        plan: Plan,
        relation: usize,
        binds: &[u64],
        memo: &[Option<Plan>],
    ) -> Vec<Plan> {
        let Some(hop) = &self.relations[relation].hop else {
            return Vec::new();
        };
        if self.foreign_between(input, 1 << relation)
            || hop
                .edge
                .as_ref()
                .and_then(|edge| self.variable_bit(edge))
                .is_some_and(|edge| binds[input] & edge != 0)
        {
            return Vec::new();
        }
        let bound = |name: &str| {
            self.variable_bit(name)
                .is_some_and(|bit| binds[input] & bit != 0)
        };
        let (source, target) = (bound(&hop.source), bound(&hop.target));
        let closes = source && target;
        let mut steps = Vec::with_capacity(2);
        let mut consider = |backward: bool, per_row: PerRow| {
            let far_end = if backward { &hop.source } else { &hop.target };
            let per_row = self.met_at_reached_ends(relation, backward, input, per_row);
            let mut rows = plan.rows * per_row.kept;
            if closes {
                let domain = self
                    .variable_bit(far_end)
                    .and_then(|bit| self.domain_of_bits(bit).next())
                    .unwrap_or(1.0);
                rows /= domain.max(1.0);
            }
            let rows = rows.max(1.0);
            let cost = if closes {
                let lookups = plan.cost + measured::closing_hop(plan.rows, rows);
                self.intersection_cost(plan, hop, rows, memo)
                    .map_or(lookups, |intersection| intersection.min(lookups))
            } else {
                plan.cost
                    + measured::hop(
                        plan.rows * per_row.calls,
                        plan.rows * per_row.read,
                        rows,
                        far_label(hop, backward),
                    )
            };
            steps.push(Plan {
                rows,
                cost,
                choice: Choice::Step {
                    input,
                    relation,
                    backward,
                    closes,
                },
            });
        };
        if source {
            consider(false, hop.forward_factor);
        }
        if target && let Some(factor) = hop.backward_factor {
            consider(true, factor);
        }
        steps
    }

    /// What the hop `relation` makes of a row of `input`, whose relations reached the end
    /// it runs from: the second stage's reads in place of the first's, and the rows kept
    /// in the same share of them, so a label check after the hop still counts. Of several
    /// hops that reach it the largest: a variable reached twice is more likely a hub, and
    /// a high estimate costs little where a low one costs much.
    fn met_at_reached_ends(
        &self,
        relation: usize,
        backward: bool,
        input: usize,
        per_row: PerRow,
    ) -> PerRow {
        let Some(reached) = self.reached.get(relation) else {
            return per_row;
        };
        let largest = reached[usize::from(backward)]
            .iter()
            .filter(|(other, _)| input & (1 << other) != 0)
            .map(|(_, degree)| *degree)
            .fold(None, |largest: Option<f64>, degree| {
                Some(largest.map_or(degree, |so_far| so_far.max(degree)))
            });
        let Some(read) = largest else {
            return per_row;
        };
        let kept_share = if per_row.read > 0.0 {
            per_row.kept / per_row.read
        } else {
            1.0
        };
        PerRow {
            calls: per_row.calls,
            read,
            kept: read * kept_share,
        }
    }

    /// The cost of the plan `below` with the closing `hop` over it, when the two run as
    /// one intersection: `below` ends in a hop that reaches an end of `hop`, with no
    /// check between that may raise (see `Optimizer::intersect_closing_hops`). It reads
    /// the lists from the two bound nodes, in place of that hop's list and the lookups.
    fn intersection_cost(
        &self,
        below: Plan,
        hop: &Hop,
        rows: f64,
        memo: &[Option<Plan>],
    ) -> Option<f64> {
        let Choice::Step {
            input,
            relation,
            backward,
            closes: false,
        } = below.choice
        else {
            return None;
        };
        let near = self.relations[relation].hop.as_ref()?;
        let meets_source = if backward { &near.source } else { &near.target } == &hop.source;
        let meets_target = if backward { &near.source } else { &near.target } == &hop.target;
        if !self.intersects
            || !(meets_source || meets_target)
            || !near.raising.is_empty()
            || !near.source_raising.is_empty()
            || !hop.source_raising.is_empty()
        {
            return None;
        }
        let input = memo[input]?;
        let near_read = if backward {
            near.backward_factor?
        } else {
            near.forward_factor
        };
        // The closing hop's list is read from its other end.
        let far_read = if meets_source {
            hop.backward_factor?
        } else {
            hop.forward_factor
        };
        Some(input.cost + measured::intersection(input.rows, near_read.read, far_read.read, rows))
    }

    fn variable_bit(&self, name: &str) -> Option<u64> {
        self.variables
            .iter()
            .position(|variable| variable == name)
            .map(|id| 1_u64 << id)
    }

    /// The operator tree of the plan of `set`.
    fn build(&mut self, memo: &[Option<Plan>], set: usize) -> LogicalOperator {
        let Some(plan) = memo[set] else {
            return LogicalOperator::Empty;
        };
        match plan.choice {
            Choice::Leaf { relation, backward } => {
                let relation = &self.relations[relation];
                match &relation.hop {
                    Some(hop) => hop.leaf(backward),
                    None => relation.operator.clone(),
                }
            }
            Choice::Hash { left, right } => {
                let conditions = self.conditions_between(left, right);
                let left = self.build(memo, left);
                let right = self.build(memo, right);
                LogicalOperator::Join(JoinOp {
                    left: Box::new(left),
                    right: Box::new(right),
                    join_type: JoinType::Inner,
                    conditions,
                })
            }
            Choice::Step {
                input,
                relation,
                backward,
                closes,
            } => {
                let input = self.build(memo, input);
                match self.relations[relation].hop.as_ref() {
                    Some(hop) => hop.step(input, backward, closes, &mut self.fresh.borrow_mut()),
                    None => input,
                }
            }
        }
    }

    /// The conditions of a hash join of two sets: every shared variable, in the
    /// order of its id, and the other conditions between them.
    fn conditions_between(&self, left: usize, right: usize) -> Vec<JoinCondition> {
        let (left_binds, right_binds) = (self.binds_of(left), self.binds_of(right));
        let shared = left_binds & right_binds;
        let mut conditions: Vec<JoinCondition> = self
            .variables
            .iter()
            .enumerate()
            .filter(|(id, _)| shared & (1_u64 << id) != 0)
            .map(|(_, name)| JoinCondition {
                left: LogicalExpression::Variable(name.clone()),
                right: LogicalExpression::Variable(name.clone()),
            })
            .collect();
        for foreign in &self.foreign {
            let (a, b) = (foreign.left_relation_binds, foreign.right_relation_binds);
            if a & left_binds == a && b & right_binds == b {
                conditions.push(foreign.condition.clone());
            } else if b & left_binds == b && a & right_binds == a {
                conditions.push(JoinCondition {
                    left: foreign.condition.right.clone(),
                    right: foreign.condition.left.clone(),
                });
            }
        }
        conditions
    }
}

/// Keeps the cheaper plan; the earlier one on a tie.
fn keep_cheaper(best: &mut Option<Plan>, plan: Plan) {
    if best.is_none_or(|kept| plan.cost < kept.cost) {
        *best = Some(plan);
    }
}
