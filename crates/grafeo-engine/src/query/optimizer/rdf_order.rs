//! The order of the patterns of a SPARQL group: each chain of inner joins is
//! rebuilt to start at its most selective input (a small `VALUES` block, a
//! pattern with constants) and to add next the cheapest input that shares a
//! variable with what is bound already, as the per-predicate counts of the
//! store's statistics estimate it (a pattern whose subject is bound reads a
//! subject's triples of its predicate). The planner then looks the triples
//! of each pattern up from the rows before it (see the RDF planner's
//! `triple_lookup`). An inner join commutes, so the rows are the same; the
//! join graph of the general reordering keys a relation by one variable,
//! which every pattern of a star on one subject shares, so it reorders no
//! SPARQL group.

use std::collections::{BTreeMap, BTreeSet, HashSet};

use std::sync::Arc;

use grafeo_core::graph::rdf::RdfStore;
use grafeo_core::statistics::RdfStatistics;

use crate::query::plan::{
    GraphsRead, JoinCondition, JoinOp, JoinType, LogicalExpression, LogicalOperator,
    TripleComponent, TripleScanOp,
};

/// What the order is estimated from: the store's statistics (per-predicate
/// counts and totals), when the optimizer was made from RDF statistics,
/// shared with the cardinality estimator, and those of the named graphs the
/// plan's patterns and paths read (each its own store, with its own
/// statistics).
pub(super) struct RdfOrder {
    statistics: Arc<RdfStatistics>,
    graphs: BTreeMap<String, Arc<RdfStatistics>>,
}

#[expect(
    clippy::cast_precision_loss,
    reason = "estimates: a count past 2^53 rounds, which no order depends on"
)]
fn estimate_of(count: u64) -> f64 {
    count as f64
}

impl RdfOrder {
    pub(super) fn new(statistics: Arc<RdfStatistics>) -> Self {
        Self {
            statistics,
            graphs: BTreeMap::new(),
        }
    }

    /// The statistics of the named graphs the plan reads, by name: every
    /// existing graph [`graphs_read`] names.
    pub(super) fn set_graphs(&mut self, graphs: BTreeMap<String, Arc<RdfStatistics>>) {
        self.graphs = graphs;
    }

    /// The statistics of the graphs `read` names: the default graph's, or those of the named
    /// graphs the plan reads that the store holds (a graph it does not hold has none).
    fn statistics_of(&self, read: GraphsRead<'_>) -> Vec<&RdfStatistics> {
        match read {
            GraphsRead::Default => vec![self.statistics.as_ref()],
            GraphsRead::Named(names) => names
                .into_iter()
                .filter_map(|name| self.graphs.get(name).map(Arc::as_ref))
                .collect(),
            GraphsRead::EveryNamed => self.graphs.values().map(Arc::as_ref).collect(),
        }
    }

    /// The rows `scan` is estimated to give, unbound, per subject and per
    /// object (see [`Cost::Pattern`]): from the statistics of each graph it
    /// reads, summed over them, so a graph it does not find gives none.
    fn pattern_rows(&self, scan: &TripleScanOp) -> [f64; 3] {
        self.statistics_of(scan.graphs_read())
            .into_iter()
            .fold([0.0; 3], |sum, statistics| {
                let rows = pattern_rows_in(statistics, scan);
                [sum[0] + rows[0], sum[1] + rows[1], sum[2] + rows[2]]
            })
    }

    /// The triples of the graphs a chain's patterns and paths read, each graph once: what an
    /// input whose rows are not known is estimated at. The default graph's when none reads one.
    fn chain_total(&self, inputs: &[&LogicalOperator]) -> f64 {
        let mut read: Vec<&RdfStatistics> = Vec::new();
        for input in inputs {
            let graphs = match input {
                LogicalOperator::TripleScan(scan) => scan.graphs_read(),
                LogicalOperator::PropertyPath(path) => path.graphs_read(),
                _ => continue,
            };
            for statistics in self.statistics_of(graphs) {
                if !read.iter().any(|seen| std::ptr::eq(*seen, statistics)) {
                    read.push(statistics);
                }
            }
        }
        if read.is_empty() {
            read.push(&self.statistics);
        }
        estimate_of(read.iter().map(|statistics| statistics.total_triples).sum()).max(1.0)
    }

    /// `op` with every chain of inner joins in it ordered.
    pub(super) fn order(&self, op: LogicalOperator) -> LogicalOperator {
        match op {
            LogicalOperator::Join(join) if join.join_type == JoinType::Inner => {
                let mut inputs = Vec::new();
                inputs_of(&join.left, &mut inputs);
                inputs_of(&join.right, &mut inputs);
                // Two inputs (the planner joins from either side), an input
                // whose variables are not known, or the written order of a
                // left-deep chain (each join adds one input, so the first one
                // starts it): the chain stays as written, with its
                // conditions, and only its inputs are ordered inside. A
                // nested chain in the written order is rebuilt: its first
                // input would be joined last.
                let ordered = if inputs.len() < 3 {
                    None
                } else {
                    self.chain(&inputs)
                };
                let as_written = left_deep(&join);
                let Some((order, variables)) = ordered.filter(|(order, _)| {
                    !as_written || order.iter().enumerate().any(|(written, &at)| written != at)
                }) else {
                    return self.order_inputs(LogicalOperator::Join(join));
                };
                let mut leaves = Vec::new();
                flatten(LogicalOperator::Join(join), &mut leaves);
                let mut slots: Vec<Option<LogicalOperator>> = leaves
                    .into_iter()
                    .map(|leaf| Some(self.order(leaf)))
                    .collect();
                rebuild(
                    order
                        .into_iter()
                        .map(|i| {
                            (
                                slots[i].take().expect("each leaf once"),
                                variables[i].clone(),
                            )
                        })
                        .collect(),
                )
            }
            LogicalOperator::Filter(mut op) => {
                op.input = Box::new(self.order(*op.input));
                LogicalOperator::Filter(op)
            }
            LogicalOperator::Project(mut op) => {
                op.input = Box::new(self.order(*op.input));
                LogicalOperator::Project(op)
            }
            LogicalOperator::Distinct(mut op) => {
                op.input = Box::new(self.order(*op.input));
                LogicalOperator::Distinct(op)
            }
            LogicalOperator::Sort(mut op) => {
                op.input = Box::new(self.order(*op.input));
                LogicalOperator::Sort(op)
            }
            LogicalOperator::Limit(mut op) => {
                op.input = Box::new(self.order(*op.input));
                LogicalOperator::Limit(op)
            }
            LogicalOperator::Skip(mut op) => {
                op.input = Box::new(self.order(*op.input));
                LogicalOperator::Skip(op)
            }
            LogicalOperator::Aggregate(mut op) => {
                op.input = Box::new(self.order(*op.input));
                LogicalOperator::Aggregate(op)
            }
            LogicalOperator::Bind(mut op) => {
                op.input = Box::new(self.order(*op.input));
                LogicalOperator::Bind(op)
            }
            LogicalOperator::Construct(mut op) => {
                op.input = Box::new(self.order(*op.input));
                LogicalOperator::Construct(op)
            }
            LogicalOperator::LeftJoin(mut op) => {
                op.left = Box::new(self.order(*op.left));
                op.right = Box::new(self.order(*op.right));
                LogicalOperator::LeftJoin(op)
            }
            LogicalOperator::AntiJoin(mut op) => {
                op.left = Box::new(self.order(*op.left));
                op.right = Box::new(self.order(*op.right));
                LogicalOperator::AntiJoin(op)
            }
            LogicalOperator::Union(mut op) => {
                op.inputs = op
                    .inputs
                    .into_iter()
                    .map(|input| self.order(input))
                    .collect();
                LogicalOperator::Union(op)
            }
            other => other,
        }
    }

    /// `op`, a chain of inner joins kept as written, with each of its inputs
    /// ordered inside (the chain's own joins are not looked at again).
    fn order_inputs(&self, op: LogicalOperator) -> LogicalOperator {
        match op {
            LogicalOperator::Join(mut join) if join.join_type == JoinType::Inner => {
                join.left = Box::new(self.order_inputs(*join.left));
                join.right = Box::new(self.order_inputs(*join.right));
                LogicalOperator::Join(join)
            }
            other => self.order(other),
        }
    }

    /// The order the inputs of a chain are best joined in, by their written
    /// positions, with each input's variables: the most selective first, then
    /// always the cheapest that shares a variable with the ones before (or
    /// has one row, which joins as cheaply disconnected), any when none does,
    /// the written order among equal estimates. `None` when an input's
    /// variables are not known. Each input is costed once, so a step only
    /// compares numbers.
    fn chain(&self, inputs: &[&LogicalOperator]) -> Option<(Vec<usize>, Vec<Vec<String>>)> {
        let total = self.chain_total(inputs);
        let mut names: Vec<String> = Vec::new();
        let mut index = |name: &str| match names.iter().position(|known| known == name) {
            Some(i) => i,
            None => {
                names.push(name.to_string());
                names.len() - 1
            }
        };
        let costed: Vec<Costed> = inputs
            .iter()
            .map(|input| self.costed(input, &mut index, total))
            .collect::<Option<_>>()?;
        let mut bound = vec![false; names.len()];
        let mut remaining: Vec<usize> = (0..inputs.len()).collect();
        let mut order = Vec::with_capacity(inputs.len());
        while !remaining.is_empty() {
            let connected = |i: usize| costed[i].variables.iter().any(|&v| bound[v]);
            let mut best: Option<(usize, f64, bool)> = None;
            for &i in &remaining {
                let estimate = costed[i].estimate(&bound);
                let eligible = connected(i) || estimate <= 1.0;
                let better = match best {
                    None => true,
                    // An eligible input beats any other; then the estimate.
                    Some((_, best_estimate, best_eligible)) => {
                        (eligible && !best_eligible)
                            || (eligible == best_eligible && estimate < best_estimate)
                    }
                };
                if better {
                    best = Some((i, estimate, eligible));
                }
            }
            let (best, _, _) = best.expect("an input remains");
            remaining.retain(|&i| i != best);
            for &v in &costed[best].variables {
                bound[v] = true;
            }
            order.push(best);
        }
        let variables = costed
            .iter()
            .map(|input| input.variables.iter().map(|&v| names[v].clone()).collect())
            .collect();
        Some((order, variables))
    }

    /// What the order reads of `leaf`: its variables (by `index`) and how
    /// its rows depend on the ones bound before it, `total` rows when they
    /// are not known; `None` when its variables are not known.
    fn costed(
        &self,
        leaf: &LogicalOperator,
        index: &mut impl FnMut(&str) -> usize,
        total: f64,
    ) -> Option<Costed> {
        let mut end = |component: &TripleComponent| match component.as_variable() {
            Some(name) => End::Variable(index(name)),
            None => End::Constant,
        };
        let (cost, mut variables) = match leaf {
            LogicalOperator::TripleScan(scan) => {
                let (subject, object) = (end(&scan.subject), end(&scan.object));
                let rows = self.pattern_rows(scan);
                (
                    Cost::Pattern {
                        subject,
                        object,
                        rows,
                    },
                    Vec::new(),
                )
            }
            LogicalOperator::PropertyPath(path) => {
                let (subject, object) = (end(&path.subject), end(&path.object));
                // From the graphs the path searches, as a pattern's rows are.
                let graphs = self.statistics_of(path.graphs_read());
                let one_end =
                    estimate_of(graphs.iter().map(|g| g.subject_count).sum::<u64>().max(1));
                let total =
                    estimate_of(graphs.iter().map(|g| g.total_triples).sum::<u64>()).max(1.0);
                (
                    Cost::Path {
                        subject,
                        object,
                        one_end,
                        total,
                    },
                    Vec::new(),
                )
            }
            other => (Cost::Rows(rows_of(other).unwrap_or(total)), Vec::new()),
        };
        let found = leaf_variables(leaf)?;
        let mut found: Vec<&String> = found.iter().collect();
        found.sort();
        variables.extend(found.into_iter().map(|name| index(name)));
        Some(Costed { variables, cost })
    }
}

/// The rows `scan` is estimated to give in the graph of `statistics`,
/// unbound, per subject and per object: from its predicate's counts, else
/// from the graph's totals.
fn pattern_rows_in(statistics: &RdfStatistics, scan: &TripleScanOp) -> [f64; 3] {
    let total = estimate_of(statistics.total_triples).max(1.0);
    let counted = match &scan.predicate {
        TripleComponent::Iri(iri) => statistics.get_predicate(&format!("<{iri}>")),
        _ => None,
    };
    match counted {
        Some(counted) => {
            let triples = estimate_of(counted.triple_count);
            [
                triples,
                triples / estimate_of(counted.distinct_subjects.max(1)),
                triples / estimate_of(counted.distinct_objects.max(1)),
            ]
        }
        None => [
            if scan.predicate.as_variable().is_some() {
                total
            } else {
                total / estimate_of(statistics.predicate_count.max(1))
            },
            total / estimate_of(statistics.subject_count.max(1)),
            total / estimate_of(statistics.object_count.max(1)),
        ],
    }
}

/// Adds to `names` the named graphs the patterns and paths of `op` read:
/// those they name, and for `GRAPH ?g` the dataset's or every graph of
/// `store`.
pub(super) fn graphs_read(op: &LogicalOperator, store: &RdfStore, names: &mut BTreeSet<String>) {
    let read = match op {
        LogicalOperator::TripleScan(scan) => scan.graphs_read(),
        LogicalOperator::PropertyPath(path) => path.graphs_read(),
        _ => GraphsRead::Default,
    };
    match read {
        GraphsRead::Default => {}
        GraphsRead::Named(read) => names.extend(read.into_iter().map(str::to_string)),
        GraphsRead::EveryNamed => names.extend(store.graph_names_latest()),
    }
    for child in op.children() {
        graphs_read(child, store, names);
    }
}

/// An end of a pattern or path: a constant, or a variable by its index.
#[derive(Clone, Copy)]
enum End {
    Constant,
    Variable(usize),
}

impl End {
    fn fixed(self, bound: &[bool]) -> bool {
        match self {
            Self::Constant => true,
            Self::Variable(v) => bound[v],
        }
    }
}

/// How a leaf's rows depend on the variables bound before it.
enum Cost {
    /// A triple pattern: its rows with neither end fixed, with the subject
    /// fixed and with the object fixed (one with both).
    Pattern {
        subject: End,
        object: End,
        rows: [f64; 3],
    },
    /// A property path: one row with both ends fixed (it stops at the first
    /// match), what one end reaches (up to every subject), or every triple.
    Path {
        subject: End,
        object: End,
        one_end: f64,
        total: f64,
    },
    /// Rows that do not depend on what is bound (a VALUES block, or an input
    /// the order knows nothing of, at the size of the graphs its chain reads).
    Rows(f64),
}

/// A leaf as the order compares it.
struct Costed {
    variables: Vec<usize>,
    cost: Cost,
}

impl Costed {
    /// How many rows the leaf gives when the variables `bound` are known.
    fn estimate(&self, bound: &[bool]) -> f64 {
        let estimate = match &self.cost {
            Cost::Pattern {
                subject,
                object,
                rows,
            } => match (subject.fixed(bound), object.fixed(bound)) {
                (true, true) => 1.0,
                (true, false) => rows[1],
                (false, true) => rows[2],
                (false, false) => rows[0],
            },
            Cost::Path {
                subject,
                object,
                one_end,
                total,
            } => match (subject.fixed(bound), object.fixed(bound)) {
                (true, true) => 1.0,
                (true, false) | (false, true) => *one_end,
                (false, false) => *total,
            },
            Cost::Rows(rows) => *rows,
        };
        estimate.max(1.0)
    }
}

/// The rows of a VALUES block (a union of one row each), `None` for any
/// other operator.
fn rows_of(op: &LogicalOperator) -> Option<f64> {
    match op {
        LogicalOperator::Empty => Some(1.0),
        LogicalOperator::Bind(bind) => rows_of(&bind.input),
        LogicalOperator::Union(union) => union.inputs.iter().map(rows_of).sum(),
        _ => None,
    }
}

/// Whether the chain of inner joins `join` is left-deep: no join of it has
/// an inner join on its right.
fn left_deep(join: &JoinOp) -> bool {
    let mut join = join;
    loop {
        if matches!(join.right.as_ref(), LogicalOperator::Join(right) if right.join_type == JoinType::Inner)
        {
            return false;
        }
        match join.left.as_ref() {
            LogicalOperator::Join(left) if left.join_type == JoinType::Inner => join = left,
            _ => return true,
        }
    }
}

/// The inputs of the chain of inner joins `op`, in written order, by
/// reference.
fn inputs_of<'a>(op: &'a LogicalOperator, inputs: &mut Vec<&'a LogicalOperator>) {
    match op {
        LogicalOperator::Join(join) if join.join_type == JoinType::Inner => {
            inputs_of(&join.left, inputs);
            inputs_of(&join.right, inputs);
        }
        other => inputs.push(other),
    }
}

/// The inputs of the chain of inner joins `op`, in written order.
fn flatten(op: LogicalOperator, leaves: &mut Vec<LogicalOperator>) {
    match op {
        LogicalOperator::Join(join) if join.join_type == JoinType::Inner => {
            flatten(*join.left, leaves);
            flatten(*join.right, leaves);
        }
        other => leaves.push(other),
    }
}

/// The variables `leaf` binds in some row, `None` when that is not known.
fn leaf_variables(leaf: &LogicalOperator) -> Option<HashSet<String>> {
    match leaf {
        LogicalOperator::TripleScan(scan) if scan.input.is_none() => Some(
            [&scan.subject, &scan.predicate, &scan.object]
                .into_iter()
                .chain(scan.graph.as_ref())
                .filter_map(|component| component.as_variable().map(str::to_string))
                .collect(),
        ),
        other => crate::query::plan::always_bound(other),
    }
}

/// The left-deep chain of `leaves`, each joined on the variables it shares
/// with the ones before it (as the translator writes them).
fn rebuild(leaves: Vec<(LogicalOperator, Vec<String>)>) -> LogicalOperator {
    let mut leaves = leaves.into_iter();
    let (mut chain, mut seen) = leaves.next().expect("a chain has an input");
    for (leaf, variables) in leaves {
        let mut shared: Vec<&String> = variables.iter().filter(|v| seen.contains(*v)).collect();
        shared.sort();
        let conditions = shared
            .into_iter()
            .map(|name| JoinCondition {
                left: LogicalExpression::Variable(name.clone()),
                right: LogicalExpression::Variable(name.clone()),
            })
            .collect();
        seen.extend(variables);
        chain = LogicalOperator::Join(JoinOp {
            left: Box::new(chain),
            right: Box::new(leaf),
            join_type: JoinType::Inner,
            conditions,
        });
    }
    chain
}
