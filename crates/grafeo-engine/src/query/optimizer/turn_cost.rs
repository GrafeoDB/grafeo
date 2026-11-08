//! Whether a hop whose far end a key pins starts at that end: the two starts priced in the
//! measured units, by the counts, the far index and a sample.
//!
//! A literal far value is known: the plan is cached under its text, so the far end is priced by
//! the nodes its index finds for it, read up to the most at which the turn could still win. A
//! parameter's plan serves every value, so it turns only on an index of one value per node, where
//! the mean is every value's count. The walk back must win clearly, and its degree weighs each end
//! by the edges that reach it, which reads high: a high reading keeps the source.

use std::collections::HashSet;

use grafeo_core::execution::operators::SeekKey;
use grafeo_core::graph::Direction;

use super::cost::measured::{
    EXPAND_OUTPUT, KEY_COLUMN_CHECK, KEY_COLUMN_ROW, LABEL_CHECK, LIST_CALL, LIST_ENTRY, LOOKUP,
    PROPERTY_CHECK, SCAN_ROW,
};
use super::degrees::Candidates;
use super::{Optimizer, PlannerRule};
use crate::query::plan::{BinaryOp, ExpandDirection, ExpandOp, LogicalExpression, NodeScanOp};
use crate::query::plan::{FilterOp, LogicalOperator};
use crate::query::planner::lpg::scan::label_check;
use crate::query::planner::lpg::seek::{self, LookupLabels, SeekChoice};
use grafeo_common::types::Value;

/// How many times as much the source's plan must cost for the far end's to win: a guess to test.
const TURN_FACTOR: f64 = 2.0;

/// The most nodes of a key a hop looks up (`TARGET_KEY_CAP` of the expand): past this it lists.
const HOP_KEY_CAP: f64 = 1024.0;

/// A hop that could start at its far end.
pub(super) struct TurnCandidate<'a> {
    /// The conjuncts of the filters over the hop.
    pub written: &'a [LogicalExpression],
    /// The conjuncts of the checks of the source.
    pub checked: &'a [LogicalExpression],
    pub expand: &'a ExpandOp,
    /// The scan of the source.
    pub scan: &'a NodeScanOp,
    /// The variables of the rows the scan runs for.
    pub input_vars: &'a HashSet<String>,
    pub source_labels: &'a LookupLabels<'a>,
    pub target_labels: &'a LookupLabels<'a>,
}

impl Optimizer {
    /// Whether the plan that starts at the far end costs clearly less than the one that starts at
    /// the source. Without a count, a sample or an index's distinct values there is no comparison,
    /// and the source stays: a store that cannot tell loses nothing against what it had.
    ///
    /// One shape turns by rule, as it did before the comparison: a source with no label, pinned by
    /// an equality no index serves, whose far end the far index finds small. Its start's list is
    /// unknown (no label to sample), so pricing cannot tell its hub from its leaf, and the walk
    /// back reads no more than the scan of every node's key.
    #[inline(never)]
    pub(super) fn turn_wins(&self, candidate: &TurnCandidate<'_>) -> bool {
        self.turn_costs(candidate).is_some_and(|costs| {
            (costs.far_read && unlabelled_with_an_unindexed_pin(self, candidate))
                || costs.source > TURN_FACTOR * costs.far
        })
    }

    /// What the two plans cost, the source's first, and whether the far end was read rather than
    /// estimated. The rows of the scan's input multiply both.
    fn turn_costs(&self, candidate: &TurnCandidate<'_>) -> Option<TurnCosts> {
        let degrees = self.cost_model.hop_degrees()?;
        let expand = candidate.expand;
        let types: Vec<&str> = expand.edge_types.iter().map(String::as_str).collect();
        let (direction, back) = match expand.direction {
            ExpandDirection::Outgoing => (Direction::Outgoing, Direction::Incoming),
            ExpandDirection::Incoming => (Direction::Incoming, Direction::Outgoing),
            ExpandDirection::Both => (Direction::Both, Direction::Both),
        };

        let source = expand.from_variable.as_str();
        let target = expand.to_variable.as_str();
        let far_pin = candidate.written.iter().find_map(|conjunct| {
            seek::pin_of(
                conjunct,
                target,
                |key| self.index_serves(candidate.target_labels, key),
                candidate.input_vars,
            )
        })?;
        let mean_far_nodes = self.nodes_a_pin_finds(&far_pin, candidate.target_labels)?;

        let start_label = candidate.source_labels.required().first().copied();
        let far_label = candidate.target_labels.required().first().copied();
        let walked_back = degrees.backward_degree(start_label, far_label, &types, direction)?;
        let source_has_label = !candidate.source_labels.required().is_empty();
        let source_checks_property = candidate
            .checked
            .iter()
            .any(|check| label_check(check, source).is_none());
        // An entry either walk emits pays its row and the checks of the end it reaches: the far
        // plan the source's, the source plan the far end's (its label, then its key).
        let emitted = |checks_label: bool, checks_property: bool| {
            EXPAND_OUTPUT
                + if checks_label { LABEL_CHECK } else { 0.0 }
                + if checks_property { PROPERTY_CHECK } else { 0.0 }
        };
        let per_entry_back = LIST_ENTRY + emitted(source_has_label, source_checks_property);
        let per_entry_out = emitted(!candidate.target_labels.required().is_empty(), true);

        let source_pin = candidate.checked.iter().find_map(|check| {
            seek::pin_of(
                check,
                source,
                |key| self.index_serves(candidate.source_labels, key),
                candidate.input_vars,
            )
        });
        let (rows, start) = match source_pin {
            Some(pin) => {
                let rows = self.nodes_a_pin_finds(&pin, candidate.source_labels)?;
                (rows, rows.max(1.0) * LOOKUP)
            }
            None => {
                let scanned = degrees.nodes_with(candidate.scan.label.as_deref()) as f64;
                // A scan whose checks each compare a key with a value no row gives reads the
                // keys' columns; any other check reads the row's property.
                let checks_on_columns = candidate.scan.input.is_none()
                    && degrees.reads_key_columns()
                    && candidate
                        .checked
                        .iter()
                        .filter(|check| label_check(check, source).is_none())
                        .all(|check| compares_a_key_with_a_constant(check, source));
                let per_row = SCAN_ROW
                    + match (source_checks_property, checks_on_columns) {
                        (false, _) => 0.0,
                        (true, true) => KEY_COLUMN_CHECK,
                        (true, false) => PROPERTY_CHECK,
                    };
                // An equality with no index: the rows are what a read of its values finds at the
                // least, which errs towards the source, and the scan is still priced whole, so a
                // large label can still lose to a far end few nodes hold.
                let unindexed = candidate
                    .checked
                    .iter()
                    .find_map(|check| seek::pin_of(check, source, |_| true, candidate.input_vars))
                    .and_then(|pin| Some((values_of(&pin)?, pin)));
                match unindexed {
                    Some((values, pin)) => {
                        let reads_the_column = candidate.scan.input.is_none()
                            && reads_no_row(&pin)
                            && self.rules.has(PlannerRule::KeyColumnLookup)
                            && degrees.reads_key_columns();
                        let per_node = if reads_the_column {
                            KEY_COLUMN_ROW
                        } else {
                            per_row
                        };
                        (values, scanned * per_node)
                    }
                    None => (
                        self.rows_after_checks(candidate, scanned),
                        scanned * per_row,
                    ),
                }
            }
        };
        let mean_degree =
            degrees.start_degree(candidate.source_labels.required(), &types, direction)?;
        // The most the source can cost, whatever the far end holds: every entry of its lists read
        // and emitted. Past the far count at which even that loses, the turn cannot win.
        let most_source =
            start + LOOKUP + rows * (LIST_CALL + mean_degree * (LIST_ENTRY + per_entry_out));
        let far = self.far_end(
            &far_pin,
            candidate.target_labels,
            mean_far_nodes,
            most_source / (TURN_FACTOR * (LOOKUP + LIST_CALL)),
            &types,
            back,
        )?;
        let (far_nodes, far_entries, far_read) = match far {
            FarEnd::Estimated(nodes) => (nodes, nodes * walked_back.max(1.0), false),
            FarEnd::Read { nodes, entries } => (nodes, entries, true),
        };
        // A start's list: at the mean for an estimated far end, at the largest the sample saw when
        // the far end was read (a pinned start may be a hub, and the far side is then exact).
        let degree = if far_read {
            degrees
                .sampled_max_degree(candidate.source_labels.required(), &types, direction)
                .map_or(mean_degree, |max| max.max(mean_degree))
        } else {
            mean_degree
        };
        let takes_a_key = !far_pin.list && far_nodes <= HOP_KEY_CAP && reads_no_row(&far_pin);
        let source_cost = if takes_a_key {
            // The key looked up once, then per row the cheaper of its list and a lookup per far
            // node, emitting the entries that reach the far end: no more than the far end has.
            let walk = LIST_CALL + degree * LIST_ENTRY;
            let reached = (rows * degree).min(far_entries);
            start + LOOKUP + rows * walk.min(far_nodes * LOOKUP) + reached * per_entry_out
        } else {
            start + rows * (LIST_CALL + degree * (LIST_ENTRY + per_entry_out))
        };
        let far_cost = far_nodes * (LOOKUP + LIST_CALL) + far_entries * per_entry_back;
        Some(TurnCosts {
            source: source_cost,
            far: far_cost,
            far_read,
        })
    }

    /// The far end the pin finds, for the turn's price. A literal by its index's candidates and
    /// their lists, read up to `beyond`, the far count past which the turn cannot win (`None` past
    /// it, the source staying). A parameter by the mean only on an index of one value per node,
    /// every value's count (`None` otherwise). Anything else, and a store that cannot read
    /// candidates, by the mean.
    fn far_end(
        &self,
        pin: &SeekChoice<'_>,
        labels: &LookupLabels<'_>,
        mean: f64,
        beyond: f64,
        types: &[&str],
        back: Direction,
    ) -> Option<FarEnd> {
        let SeekKey::Property(key) = &pin.key else {
            return Some(FarEnd::Estimated(mean));
        };
        if pin.list {
            return Some(FarEnd::Estimated(mean));
        }
        match pin.value {
            LogicalExpression::Literal(value) => {
                let degrees = self.cost_model.hop_degrees()?;
                #[allow(
                    clippy::cast_possible_truncation,
                    clippy::cast_sign_loss,
                    reason = "`beyond` is a non-negative count, capped at `usize::MAX` by the cast"
                )]
                let limit = beyond.max(0.0).floor() as usize;
                match degrees.candidates_of(labels.required(), key, value, limit, types, back) {
                    Candidates::Few { nodes, entries } => Some(FarEnd::Read {
                        nodes: nodes as f64,
                        entries: entries as f64,
                    }),
                    Candidates::Many => None,
                    Candidates::Unknown => Some(FarEnd::Estimated(mean)),
                }
            }
            LogicalExpression::Parameter(_) => (mean <= 1.0).then_some(FarEnd::Estimated(mean)),
            _ => Some(FarEnd::Estimated(mean)),
        }
    }

    /// How many nodes `pin` finds: the nodes its index holds over the values it holds, per value.
    /// For a key-wide index that is every node's, which reads high.
    fn nodes_a_pin_finds(&self, pin: &SeekChoice<'_>, labels: &LookupLabels<'_>) -> Option<f64> {
        let values = values_of(pin)?;
        let SeekKey::Property(key) = &pin.key else {
            return Some(values);
        };
        let degrees = self.cost_model.hop_degrees()?;
        let per_label = |label: Option<&str>| -> Option<f64> {
            let read: &[&str] = label.as_slice();
            let distinct = degrees.distinct_values(read, key)?;
            Some(degrees.nodes_with(label) as f64 / distinct.max(1) as f64)
        };
        let read = self.index_read(labels, key)?;
        let per_value = if read.is_empty() {
            per_label(None)?
        } else {
            let mut nodes = 0.0;
            for label in read {
                nodes += per_label(Some(label))?;
            }
            nodes
        };
        Some(values * per_value)
    }

    /// The rows left of `scanned` nodes after the checks of the source, by the estimator.
    fn rows_after_checks(&self, candidate: &TurnCandidate<'_>, scanned: f64) -> f64 {
        let scan = LogicalOperator::NodeScan(NodeScanOp {
            variable: candidate.scan.variable.clone(),
            label: candidate.scan.label.clone(),
            input: None,
        });
        match super::conjunction(candidate.checked.to_vec()) {
            Some(predicate) => self
                .card_estimator
                .estimate(&LogicalOperator::Filter(FilterOp {
                    predicate,
                    input: Box::new(scan),
                    pushdown_hint: None,
                }))
                .min(scanned),
            None => scanned,
        }
    }
}

/// What the two plans of a turn cost.
struct TurnCosts {
    source: f64,
    far: f64,
    /// Whether the far end's nodes and lists were read rather than estimated.
    far_read: bool,
}

/// The far end of a turn: its nodes and the entries a walk back reads.
enum FarEnd {
    /// From the counts: nodes, each walked back by the sampled degree.
    Estimated(f64),
    /// Read from the index and the lists.
    Read { nodes: f64, entries: f64 },
}

/// Whether the source has no label and an equality no index serves pins it: the shape that turns
/// by rule when its far end is small (see [`Optimizer::turn_wins`]).
fn unlabelled_with_an_unindexed_pin(optimizer: &Optimizer, candidate: &TurnCandidate<'_>) -> bool {
    let source = candidate.expand.from_variable.as_str();
    candidate.scan.label.is_none()
        && candidate.source_labels.required().is_empty()
        && candidate.checked.iter().any(|check| {
            seek::pin_of(check, source, |_| true, candidate.input_vars)
                .is_some_and(|pin| reads_no_row(&pin) && !pin.list)
        })
        && !candidate.checked.iter().any(|check| {
            seek::pin_of(
                check,
                source,
                |key| optimizer.index_serves(candidate.source_labels, key),
                candidate.input_vars,
            )
            .is_some()
        })
}

/// Whether `check` compares a key of `variable` with a value no row gives, by `<`, `<=`, `>` or
/// `>=` (`a.age > 90`, `$low <= a.age`), which a filter over a scan reads from the key's column.
fn compares_a_key_with_a_constant(check: &LogicalExpression, variable: &str) -> bool {
    let LogicalExpression::Binary { left, op, right } = check else {
        return false;
    };
    if !matches!(
        op,
        BinaryOp::Lt | BinaryOp::Le | BinaryOp::Gt | BinaryOp::Ge
    ) {
        return false;
    }
    let is_key = |side: &LogicalExpression| matches!(side, LogicalExpression::Property { variable: of, .. } if of == variable);
    let is_constant = |side: &LogicalExpression| {
        matches!(
            side,
            LogicalExpression::Literal(_) | LogicalExpression::Parameter(_)
        )
    };
    (is_key(left) && is_constant(right)) || (is_constant(left) && is_key(right))
}

/// Whether the value of `pin` reads no row, so a hop can find its nodes once.
fn reads_no_row(pin: &SeekChoice<'_>) -> bool {
    matches!(
        pin.value,
        LogicalExpression::Literal(_) | LogicalExpression::Parameter(_)
    )
}

/// How many values `pin` looks for, `None` for a list known only when it runs.
fn values_of(pin: &SeekChoice<'_>) -> Option<f64> {
    if pin.list {
        list_length(pin.value).map(|length| length as f64)
    } else {
        Some(1.0)
    }
}

/// How many values a list of constants holds.
fn list_length(list: &LogicalExpression) -> Option<usize> {
    match list {
        LogicalExpression::Literal(Value::List(items)) => Some(items.len()),
        LogicalExpression::List(items) => Some(items.len()),
        _ => None,
    }
}
