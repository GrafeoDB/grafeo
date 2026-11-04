//! Whether a hop whose far end a key pins starts at that end: the two starts priced in the
//! measured units, by the counts, the far index and a sample.
//!
//! The value is unknown here: a cached plan serves other values, so a hub the sample cannot see
//! costs a walk back through all of it. So the walk back must win clearly, and its degree weighs
//! each end by the edges that reach it, which reads high: a high reading keeps the source.

use std::collections::HashSet;

use grafeo_core::execution::operators::SeekKey;
use grafeo_core::graph::Direction;

use super::Optimizer;
use super::cost::measured::{
    EXPAND_OUTPUT, LABEL_CHECK, LIST_CALL, LIST_ENTRY, LOOKUP, PROPERTY_CHECK, SCAN_ROW,
};
use crate::query::plan::{ExpandDirection, ExpandOp, LogicalExpression, NodeScanOp};
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
    #[inline(never)]
    pub(super) fn turn_wins(&self, candidate: &TurnCandidate<'_>) -> bool {
        self.turn_costs(candidate)
            .is_some_and(|(source, far)| source > TURN_FACTOR * far)
    }

    /// What the two plans cost, the source's first. The rows of the scan's input multiply both.
    fn turn_costs(&self, candidate: &TurnCandidate<'_>) -> Option<(f64, f64)> {
        let degrees = self.cost_model.hop_degrees()?;
        let expand = candidate.expand;
        let types: Vec<&str> = expand.edge_types.iter().map(String::as_str).collect();
        let direction = match expand.direction {
            ExpandDirection::Outgoing => Direction::Outgoing,
            ExpandDirection::Incoming => Direction::Incoming,
            ExpandDirection::Both => Direction::Both,
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
        let far_nodes = self.nodes_a_pin_finds(&far_pin, candidate.target_labels)?;

        let start_label = candidate.source_labels.required().first().copied();
        let far_label = candidate.target_labels.required().first().copied();
        let walked_back = degrees.backward_degree(start_label, far_label, &types, direction)?;
        let far_checks_label = !candidate.source_labels.required().is_empty();
        let far_checks_property = candidate
            .checked
            .iter()
            .any(|check| label_check(check, source).is_none());
        let per_entry_back = LIST_ENTRY
            + EXPAND_OUTPUT
            + if far_checks_label { LABEL_CHECK } else { 0.0 }
            + if far_checks_property {
                PROPERTY_CHECK
            } else {
                0.0
            };
        let far_first =
            far_nodes * (LOOKUP + LIST_CALL) + far_nodes * walked_back.max(1.0) * per_entry_back;

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
                let per_row = SCAN_ROW
                    + if far_checks_property {
                        PROPERTY_CHECK
                    } else {
                        0.0
                    };
                (
                    self.rows_after_checks(candidate, scanned),
                    scanned * per_row,
                )
            }
        };
        let degree =
            degrees.counted_degree(candidate.source_labels.required(), &types, direction)?;
        let walk = LIST_CALL + degree * LIST_ENTRY;
        let takes_a_key = !far_pin.list && far_nodes <= HOP_KEY_CAP && reads_no_row(&far_pin);
        let (find_key, per_row) = if takes_a_key {
            (LOOKUP, walk.min(far_nodes * LOOKUP))
        } else {
            (
                0.0,
                LIST_CALL + degree * (LIST_ENTRY + EXPAND_OUTPUT + PROPERTY_CHECK),
            )
        };
        Some((start + find_key + rows * per_row, far_first))
    }

    /// How many nodes `pin` finds: the nodes its index holds over the values it holds, per value.
    /// For a key-wide index that is every node's, which reads high.
    fn nodes_a_pin_finds(&self, pin: &SeekChoice<'_>, labels: &LookupLabels<'_>) -> Option<f64> {
        let values = if pin.list {
            list_length(pin.value)? as f64
        } else {
            1.0
        };
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

/// Whether the value of `pin` reads no row, so a hop can find its nodes once.
fn reads_no_row(pin: &SeekChoice<'_>) -> bool {
    matches!(
        pin.value,
        LogicalExpression::Literal(_) | LogicalExpression::Parameter(_)
    )
}

/// How many values a list of constants holds.
fn list_length(list: &LogicalExpression) -> Option<usize> {
    match list {
        LogicalExpression::Literal(Value::List(items)) => Some(items.len()),
        LogicalExpression::List(items) => Some(items.len()),
        _ => None,
    }
}
