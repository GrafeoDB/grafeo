//! How many edges a hop finds per node.
//!
//! The one function the planner reads: a type's edges over the nodes of the end the
//! hop runs from, and for a comma pattern, whose order depends on it, the smaller of
//! that and a sample of the end's label (see `GraphStore::sampled_degree`). An
//! unlabelled end has no nodes to sample, so its degree stays blind to direction, and
//! a sample may miss a label's hubs: a slower plan, never other rows. The second stage
//! reads the ends a hop reached, which is what a path through them meets.

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, OnceLock};

use grafeo_core::graph::Direction;
use grafeo_core::graph::traits::{DEGREE_SAMPLE_NODES, GraphStore};
use grafeo_core::statistics::Statistics;

use crate::query::plan::{ExpandDirection, ExpandOp, LogicalExpression, LogicalOperator};
use crate::query::planner::lpg::scan::label_check;
use crate::query::planner::lpg::seek::split_conjuncts;

/// The reads one plan may spend on degrees of reached ends: entries are taken in written
/// order until they run out, and the rest use the first stage until a later plan reads
/// them. A placeholder until the first plan's time is measured.
const REACHED_READS_PER_PLAN: u64 = 8_192;

/// The most reads a plan's choices of where a hop starts spend on samples, apart from the join
/// search's, so that they leave its plan as it was. A placeholder until measured.
const TURN_READS_PER_PLAN: u64 = 1_024;

/// What a read of a far key's candidates found (see [`HopDegrees::candidates_of`]).
pub(super) enum Candidates {
    /// At most the limit: the nodes, and the entries their lists hold.
    Few { nodes: usize, entries: usize },
    /// More than the limit.
    Many,
    /// A store that cannot say.
    Unknown,
}

/// The counts a hop's degree derives from, and the store that samples it.
pub struct HopDegrees {
    /// The store's counts, shared, so a plan with no hop pays no copy of them.
    statistics: Arc<Statistics>,
    /// The store to sample, for a session's optimizer.
    sampler: OnceLock<Arc<dyn GraphStore>>,
    /// Whether the join search asked: plain queries sample nothing.
    sampling: AtomicBool,
    /// The reads left for degrees of reached ends in this plan.
    reached_reads_left: AtomicU64,
    /// The reads left for the choices of where a hop starts in this plan.
    turn_reads_left: AtomicU64,
}

impl std::fmt::Debug for HopDegrees {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HopDegrees")
            .field("total_nodes", &self.statistics.total_nodes)
            .finish_non_exhaustive()
    }
}

impl HopDegrees {
    /// The counts in `statistics`.
    #[must_use]
    pub fn from_statistics(statistics: Arc<Statistics>) -> Self {
        Self {
            statistics,
            sampler: OnceLock::new(),
            sampling: AtomicBool::new(false),
            reached_reads_left: AtomicU64::new(REACHED_READS_PER_PLAN),
            turn_reads_left: AtomicU64::new(TURN_READS_PER_PLAN),
        }
    }

    /// The degree a hop meets at the ends another hop reached (see
    /// `GraphStore::sampled_reached_degree`), `None` unless the join search asked and the
    /// plan's reads allow it.
    pub(super) fn reached(
        &self,
        start_label: Option<&str>,
        via_types: &[&str],
        via_direction: Direction,
        on_types: &[&str],
        on_direction: Direction,
    ) -> Option<f64> {
        if !self.sampling.load(Ordering::Relaxed) {
            return None;
        }
        let sampler = self.sampler.get()?;
        let mut left = self.reached_reads_left.load(Ordering::Relaxed);
        let degree = sampler.sampled_reached_degree_within(
            start_label,
            via_types,
            via_direction,
            on_types,
            on_direction,
            &mut left,
        );
        self.reached_reads_left.store(left, Ordering::Relaxed);
        degree
    }

    /// The degree at the far end of a hop from a node of `start_label`, over `direction` there and
    /// back, weighted by the edges that reach each end, so that a hub weighs what a walk back from
    /// it reads. When a sample of the source reaches nothing, as a hub among many starts leaves it,
    /// the far side's lists of `far_label` tell instead. `None` for a store that cannot sample, or
    /// past this plan's reads for it.
    pub(super) fn backward_degree(
        &self,
        start_label: Option<&str>,
        far_label: Option<&str>,
        types: &[&str],
        direction: Direction,
    ) -> Option<f64> {
        let back = match direction {
            Direction::Outgoing => Direction::Incoming,
            Direction::Incoming => Direction::Outgoing,
            both => both,
        };
        let sampler = self.sampler.get()?;
        let mut left = self.turn_reads_left.load(Ordering::Relaxed);
        let degree = sampler.sampled_reached_degree_within(
            start_label,
            types,
            direction,
            types,
            back,
            &mut left,
        );
        let degree =
            degree.or_else(|| sampler.far_walk_back_within(far_label?, types, back, &mut left));
        self.turn_reads_left.store(left, Ordering::Relaxed);
        degree
    }

    /// Whether the store can read an equality's key from the key's column, and so without a row.
    pub(super) fn reads_key_columns(&self) -> bool {
        self.sampler
            .get()
            .is_some_and(|store| store.reads_key_columns())
    }

    /// The nodes with `labels` the index that serves them on `key` may find equal to `value`, and
    /// the entries their lists of `types` hold in `direction`, read up to `limit` candidates.
    /// Candidates may include values `=` tells apart, and entries removed ones, so it reads high.
    pub(super) fn candidates_of(
        &self,
        labels: &[&str],
        key: &str,
        value: &grafeo_common::types::Value,
        limit: usize,
        types: &[&str],
        direction: Direction,
    ) -> Candidates {
        let Some(store) = self.sampler.get() else {
            return Candidates::Unknown;
        };
        let latest = grafeo_common::types::ReadView::latest();
        let nodes = match store.find_few_indexed_maybe_equal(labels, key, value, limit, latest) {
            Ok(Some(nodes)) => nodes,
            Ok(None) => return Candidates::Many,
            Err(_) => return Candidates::Unknown,
        };
        let types: Vec<grafeo_common::types::ArcStr> =
            types.iter().map(|name| (*name).into()).collect();
        let mut entries = 0;
        for node in &nodes {
            let Some(listed) = store.edge_entries_at_most(*node, direction, &types, latest) else {
                return Candidates::Unknown;
            };
            entries += listed;
        }
        Candidates::Few {
            nodes: nodes.len(),
            entries,
        }
    }

    /// The largest degree over `types` in `direction` a sample of the rarest of `labels` saw, by
    /// type, added up: a start's list is at most this long, but for one the sample missed. `None`
    /// without a label or a sample.
    pub(super) fn sampled_max_degree(
        &self,
        labels: &[&str],
        types: &[&str],
        direction: Direction,
    ) -> Option<f64> {
        let store = self.sampler.get()?;
        let label = labels
            .iter()
            .copied()
            .min_by_key(|label| self.nodes_of(label).unwrap_or(0))?;
        let mut max = 0.0;
        for edge_type in types {
            max += store.sampled_max_degree(label, edge_type, direction)?;
        }
        (!types.is_empty()).then_some(max)
    }

    /// How many distinct values the store's index for `labels` and `key` holds.
    pub(super) fn distinct_values(&self, labels: &[&str], key: &str) -> Option<u64> {
        self.sampler.get()?.indexed_distinct_values(labels, key)
    }

    /// How many nodes have `label`, or how many there are for none.
    pub(super) fn nodes_with(&self, label: Option<&str>) -> u64 {
        label.map_or(self.statistics.total_nodes, |label| {
            self.nodes_of(label).unwrap_or(0)
        })
    }

    /// The degree from a node with `labels` as the turn prices a start's list: by a sample of the
    /// label where the store samples, else by the counts (see [`Self::of`]). The counts read high
    /// for a small label of a type other labels hold most edges of.
    pub(super) fn start_degree(
        &self,
        labels: &[&str],
        types: &[&str],
        direction: Direction,
    ) -> Option<f64> {
        self.degree(
            self.sampler.get().map(AsRef::as_ref),
            labels,
            types,
            direction,
        )
    }

    /// The nodes with `label`, as counted.
    fn nodes_of(&self, label: &str) -> Option<u64> {
        self.statistics
            .labels
            .get(label)
            .map(|label| label.node_count)
    }

    /// Samples `store` once the join search asks (see [`Self::start_sampling`]).
    pub(super) fn sample_from(&self, store: Arc<dyn GraphStore>) {
        let _ = self.sampler.set(store);
    }

    /// The join search is about to read degrees, whose order depends on them.
    pub(super) fn start_sampling(&self) {
        self.sampling.store(true, Ordering::Relaxed);
    }

    /// The edges per row of `expand`, `None` when the counts do not say: no edge
    /// type named, or no node to count them over.
    #[must_use]
    pub fn of(&self, expand: &ExpandOp) -> Option<f64> {
        let mut labels = Vec::new();
        labels_of(&expand.input, &expand.from_variable, &mut labels);
        let direction = match expand.direction {
            ExpandDirection::Outgoing => Direction::Outgoing,
            ExpandDirection::Incoming => Direction::Incoming,
            ExpandDirection::Both => Direction::Both,
        };
        let types: Vec<&str> = expand.edge_types.iter().map(String::as_str).collect();
        let sampler = self
            .sampling
            .load(Ordering::Relaxed)
            .then(|| self.sampler.get())
            .flatten();
        self.degree(sampler.map(AsRef::as_ref), &labels, &types, direction)
    }

    /// The degree from a node with `labels`, the one with the fewest nodes counting,
    /// over `types` added up, sampled from `sampler` when there is one.
    fn degree(
        &self,
        sampler: Option<&dyn GraphStore>,
        labels: &[&str],
        types: &[&str],
        direction: Direction,
    ) -> Option<f64> {
        if types.is_empty() {
            return None;
        }
        let label = labels
            .iter()
            .copied()
            .min_by_key(|label| self.nodes_of(label).unwrap_or(0));
        let nodes = match label {
            Some(label) => self.nodes_of(label).unwrap_or(0),
            None => self.statistics.total_nodes,
        };
        if nodes == 0 {
            return None;
        }
        let per_direction = if direction == Direction::Both {
            2.0
        } else {
            1.0
        };
        Some(
            types
                .iter()
                .map(|edge_type| {
                    let edges = self
                        .statistics
                        .edge_types
                        .get(*edge_type)
                        .map_or(0, |edge_type| edge_type.edge_count);
                    let derived = edges as f64 / nodes as f64 * per_direction;
                    // A 0 from part of a label may only have missed the nodes with
                    // edges, and would price the hop as free.
                    let sampled = label
                        .zip(sampler)
                        .and_then(|(label, sampler)| {
                            sampler.sampled_degree(label, edge_type, direction)
                        })
                        .filter(|sampled| *sampled > 0.0 || nodes <= DEGREE_SAMPLE_NODES as u64);
                    // A label's mean cannot exceed its share of the type's edges.
                    sampled.map_or(derived, |sampled| sampled.min(derived))
                })
                .sum(),
        )
    }

    /// The nodes `variable` can be in `op`: the count of the label `op` checks or
    /// scans it by, the smallest of several, else all nodes. At least one.
    #[must_use]
    pub fn domain(&self, variable: &str, op: &LogicalOperator) -> f64 {
        let mut labels = Vec::new();
        labels_of(op, variable, &mut labels);
        let nodes = labels
            .iter()
            .filter_map(|label| self.nodes_of(label))
            .min()
            .unwrap_or(self.statistics.total_nodes);
        (nodes as f64).max(1.0)
    }
}

/// The degree the planner reads for a node with `labels` over `types`, sampled from
/// `store`: the smaller of the sample and the type's edges over the label's nodes.
/// For tests; 0 when the counts do not say.
#[doc(hidden)]
#[must_use]
pub fn hop_degree(
    store: &dyn GraphStore,
    labels: &[&str],
    types: &[&str],
    direction: Direction,
) -> f64 {
    HopDegrees::from_statistics(store.statistics())
        .degree(Some(store), labels, types, direction)
        .unwrap_or(0.0)
}

/// The labels `op` checks or scans `variable` by.
fn labels_of<'a>(op: &'a LogicalOperator, variable: &str, labels: &mut Vec<&'a str>) {
    match op {
        LogicalOperator::NodeScan(scan) => {
            if scan.variable == variable {
                labels.extend(scan.label.as_deref());
            }
            if let Some(input) = scan.input.as_deref() {
                labels_of(input, variable, labels);
            }
        }
        LogicalOperator::Filter(filter) => {
            checked_labels(&filter.predicate, variable, labels);
            labels_of(&filter.input, variable, labels);
        }
        LogicalOperator::Expand(expand) => labels_of(&expand.input, variable, labels),
        LogicalOperator::Join(join) => {
            labels_of(&join.left, variable, labels);
            labels_of(&join.right, variable, labels);
        }
        _ => {}
    }
}

/// The labels top-level conjuncts of `predicate` require of `variable`.
fn checked_labels<'a>(predicate: &'a LogicalExpression, variable: &str, labels: &mut Vec<&'a str>) {
    let mut conjuncts = Vec::new();
    split_conjuncts(predicate, &mut conjuncts);
    labels.extend(
        conjuncts
            .into_iter()
            .filter_map(|conjunct| label_check(conjunct, variable)),
    );
}
