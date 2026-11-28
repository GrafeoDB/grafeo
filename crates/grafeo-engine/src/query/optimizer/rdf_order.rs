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
    DistinctOp, GraphsRead, JoinCondition, JoinOp, JoinType, LogicalExpression, LogicalOperator,
    ProjectOp, Projection, TripleComponent, TripleScanOp,
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

    /// The probability `scan` holds with each of its variables bound (a
    /// check), from the statistics of each graph it reads: its triples over
    /// its subjects times its objects, summed over the graphs, at most 1.
    fn check_holds(&self, scan: &TripleScanOp) -> f64 {
        self.statistics_of(scan.graphs_read())
            .into_iter()
            .map(|statistics| check_holds_in(statistics, scan))
            .sum::<f64>()
            .clamp(0.0, 1.0)
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
                let Some((shape, variables)) = ordered.filter(|(shape, _)| match shape {
                    Shape::LeftDeep(order) => {
                        !as_written || order.iter().enumerate().any(|(written, &at)| written != at)
                    }
                    Shape::Split(_) => true,
                }) else {
                    return self.order_inputs(LogicalOperator::Join(join));
                };
                let mut leaves = Vec::new();
                flatten(LogicalOperator::Join(join), &mut leaves);
                let mut slots: Vec<Option<LogicalOperator>> = leaves
                    .into_iter()
                    .map(|leaf| Some(self.order(leaf)))
                    .collect();
                match shape {
                    Shape::LeftDeep(order) => rebuild(
                        order
                            .into_iter()
                            .map(|i| {
                                (
                                    slots[i].take().expect("each leaf once"),
                                    variables[i].clone(),
                                )
                            })
                            .collect(),
                    ),
                    Shape::Split(split) => split.build(&mut slots, &variables),
                }
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

    /// How the inputs of a chain are best joined, by their written positions
    /// (see [`greedy`], and [`four_cycle`] for a cycle of four through an
    /// anchored node), with each input's variables. `None` when an input's
    /// variables are not known. Each input is costed once, so a step only
    /// compares numbers.
    fn chain(&self, inputs: &[&LogicalOperator]) -> Option<(Shape, Vec<Vec<String>>)> {
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
        let order = greedy(
            &costed,
            (0..inputs.len()).collect(),
            &mut vec![false; names.len()],
        );
        let variables = costed
            .iter()
            .map(|input| input.variables.iter().map(|&v| names[v].clone()).collect())
            .collect();
        let shape = four_cycle(inputs, &costed, &order, &names)
            .map_or(Shape::LeftDeep(order), Shape::Split);
        Some((shape, variables))
    }

    /// How the rows of `scan` depend on its ends (by `end`).
    fn pattern_cost(
        &self,
        scan: &TripleScanOp,
        end: &mut impl FnMut(&TripleComponent) -> End,
    ) -> Cost {
        Cost::Pattern {
            subject: end(&scan.subject),
            object: end(&scan.object),
            rows: self.pattern_rows(scan),
            holds: self.check_holds(scan),
        }
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
            LogicalOperator::TripleScan(scan) => (self.pattern_cost(scan, &mut end), Vec::new()),
            // A UNION of patterns (an alternative path, `{ ?s :p ?o } UNION
            // { ?s :q ?o }`): each branch by its own ends.
            LogicalOperator::Union(union) => {
                let branches: Option<Vec<(Cost, Vec<usize>)>> = union
                    .inputs
                    .iter()
                    .map(|branch| {
                        let scan = union_branch_pattern(branch)?;
                        let cost = self.pattern_cost(scan, &mut end);
                        let variables = scan_components(scan)
                            .filter_map(|component| match end(component) {
                                End::Variable(v) => Some(v),
                                End::Constant => None,
                            })
                            .collect();
                        Some((cost, variables))
                    })
                    .collect();
                match branches {
                    Some(branches) => (Cost::Union(branches), Vec::new()),
                    None => (Cost::Rows(rows_of(leaf).unwrap_or(total)), Vec::new()),
                }
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

/// The probability a check of `scan` holds in the graph of `statistics`:
/// its predicate's triples over its subjects times its objects; for a
/// predicate the statistics do not list (none in the graph, or inserted
/// since they were collected, see [`pattern_rows_in`]) an average
/// predicate's triples over the graph's subjects times its objects; for a
/// variable predicate the graph's totals.
fn check_holds_in(statistics: &RdfStatistics, scan: &TripleScanOp) -> f64 {
    let (triples, subjects, objects) = match &scan.predicate {
        TripleComponent::Iri(iri) => match statistics.get_predicate(&format!("<{iri}>")) {
            Some(counted) => (
                estimate_of(counted.triple_count),
                counted.distinct_subjects,
                counted.distinct_objects,
            ),
            None => (
                estimate_of(statistics.total_triples)
                    / estimate_of(statistics.predicate_count.max(1)),
                statistics.subject_count,
                statistics.object_count,
            ),
        },
        _ => (
            estimate_of(statistics.total_triples),
            statistics.subject_count,
            statistics.object_count,
        ),
    };
    triples / (estimate_of(subjects.max(1)) * estimate_of(objects.max(1)))
}

/// The pattern a UNION branch is: a scan, or a projection that keeps the
/// names of one (the inverse branch of an alternative path); `None` for
/// any other branch (a sequence, a filter, a renaming).
fn union_branch_pattern(branch: &LogicalOperator) -> Option<&TripleScanOp> {
    match branch {
        LogicalOperator::TripleScan(scan) if scan.input.is_none() => Some(scan),
        LogicalOperator::Project(project)
            if project.projections.iter().all(|projection| {
                matches!(&projection.expression, LogicalExpression::Variable(name)
                    if projection.alias.as_ref().is_none_or(|alias| alias == name))
            }) =>
        {
            match project.input.as_ref() {
                LogicalOperator::TripleScan(scan) if scan.input.is_none() => Some(scan),
                _ => None,
            }
        }
        _ => None,
    }
}

/// The components of `scan` that hold a variable: subject, predicate,
/// object and graph.
fn scan_components(scan: &TripleScanOp) -> impl Iterator<Item = &TripleComponent> {
    [&scan.subject, &scan.predicate, &scan.object]
        .into_iter()
        .chain(scan.graph.as_ref())
        .filter(|component| component.as_variable().is_some())
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
    /// fixed and with the object fixed (one with both), and the probability
    /// it holds as a check (every variable of it bound).
    Pattern {
        subject: End,
        object: End,
        rows: [f64; 3],
        holds: f64,
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
    /// A UNION of patterns: the sum of its branches, each with its
    /// variables (a branch may bind one the others do not).
    Union(Vec<(Cost, Vec<usize>)>),
}

/// A leaf as the order compares it.
struct Costed {
    variables: Vec<usize>,
    cost: Cost,
}

impl Costed {
    /// How many rows the leaf gives when the variables `bound` are known: at
    /// least one, except a check (a pattern, or a UNION of them, with every
    /// variable bound), which only removes rows and gives the probability
    /// it holds.
    fn estimate(&self, bound: &[bool]) -> f64 {
        let check = self.variables.iter().all(|&v| bound[v]);
        let rows = self.cost.rows(bound, check);
        match self.cost {
            Cost::Pattern { .. } | Cost::Union(_) if check => rows,
            _ => rows.max(1.0),
        }
    }
}

impl Cost {
    /// The rows of the leaf (see [`Costed::estimate`]), `check` when none of
    /// its variables is new.
    fn rows(&self, bound: &[bool], check: bool) -> f64 {
        match self {
            Self::Pattern { holds, .. } if check => *holds,
            Self::Pattern {
                subject,
                object,
                rows,
                ..
            } => match (subject.fixed(bound), object.fixed(bound)) {
                (true, true) => 1.0,
                (true, false) => rows[1],
                (false, true) => rows[2],
                (false, false) => rows[0],
            }
            .max(1.0),
            Self::Path {
                subject,
                object,
                one_end,
                total,
            } => match (subject.fixed(bound), object.fixed(bound)) {
                (true, true) => 1.0,
                (true, false) | (false, true) => *one_end,
                (false, false) => *total,
            },
            Self::Rows(rows) => *rows,
            Self::Union(branches) => branches
                .iter()
                .map(|(cost, variables)| cost.rows(bound, variables.iter().all(|&v| bound[v])))
                .sum(),
        }
    }
}

/// How the inputs of a chain are joined, by their written positions.
enum Shape {
    /// One left-deep chain, in this order.
    LeftDeep(Vec<usize>),
    /// A cycle of four through an anchored node, split at the node opposite
    /// it (see [`four_cycle`]).
    Split(Split),
}

/// A cycle of four nodes a, b, c and d through `a`, the node its first input
/// binds, joined as two halves that meet at `c`: each half reads the
/// two-step paths from `a` to `c` on its side, where a left-deep chain reads
/// every three-step path from `a`. The inputs, each list in its order:
struct Split {
    /// What `a` hangs off (its first input among them): in both halves.
    anchor: Vec<usize>,
    /// Half b: the anchor, the edge a-b, what hangs off b, the edge b-c.
    half_b: Vec<usize>,
    /// Half d after its anchor: the edge a-d, what hangs off d, the edge d-c.
    half_d: Vec<usize>,
    /// What hangs off c, joined to half b before half d joins.
    far: Vec<usize>,
    /// The variables of half d's anchor: `a`, and those the anchor binds that
    /// half d's inputs name (the graph variable every pattern names, if the
    /// anchor binds it).
    keep: Vec<String>,
}

impl Split {
    /// The plan: half b, what hangs off c, then half d, joined on the
    /// variables they share (`a`, `c` and the graph variable). Half d's
    /// anchor is the anchor's distinct values of `keep`, so a row of `a` that
    /// the anchor gives more than once multiplies the result once, as in a
    /// left-deep chain.
    fn build(
        self,
        slots: &mut [Option<LogicalOperator>],
        variables: &[Vec<String>],
    ) -> LogicalOperator {
        let names = |positions: &[usize]| -> Vec<String> {
            let mut names: Vec<String> = positions
                .iter()
                .flat_map(|&i| variables[i].iter().cloned())
                .collect();
            names.sort();
            names.dedup();
            names
        };
        let anchor = rebuild(
            self.anchor
                .iter()
                .map(|&i| {
                    (
                        slots[i].clone().expect("each leaf once"),
                        variables[i].clone(),
                    )
                })
                .collect(),
        );
        let mut take = |i: usize| {
            (
                slots[i].take().expect("each leaf once"),
                variables[i].clone(),
            )
        };
        let half_b = (
            rebuild(self.half_b.iter().map(|&i| take(i)).collect()),
            names(&self.half_b),
        );
        let anchor_d = LogicalOperator::Distinct(DistinctOp {
            input: Box::new(LogicalOperator::Project(ProjectOp {
                projections: self
                    .keep
                    .iter()
                    .map(|name| Projection {
                        expression: LogicalExpression::Variable(name.clone()),
                        alias: None,
                    })
                    .collect(),
                input: Box::new(anchor),
                pass_through_input: false,
            })),
            columns: None,
        });
        let mut d_leaves = vec![(anchor_d, self.keep.clone())];
        d_leaves.extend(self.half_d.iter().map(|&i| take(i)));
        let mut d_names = names(&self.half_d);
        d_names.extend(self.keep.iter().cloned());
        d_names.sort();
        d_names.dedup();
        let half_d = (rebuild(d_leaves), d_names);
        let mut outer = vec![half_b];
        outer.extend(self.far.iter().map(|&i| take(i)));
        outer.push(half_d);
        rebuild(outer)
    }
}

/// `positions` in the order they are best joined in, from the variables
/// `bound` (which it binds as it goes): the most selective first, then always
/// the cheapest that shares a variable with the ones before (or has one row,
/// which joins as cheaply disconnected), any when none does, the written
/// order among equal estimates.
fn greedy(costed: &[Costed], mut remaining: Vec<usize>, bound: &mut [bool]) -> Vec<usize> {
    let mut order = Vec::with_capacity(remaining.len());
    while !remaining.is_empty() {
        let mut best: Option<(usize, f64, bool)> = None;
        for &i in &remaining {
            let estimate = costed[i].estimate(bound);
            let eligible = costed[i].variables.iter().any(|&v| bound[v]) || estimate <= 1.0;
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
    order
}

/// The rows after each step of `order`, starting from `rows` rows with
/// `bound` known (which it binds as it goes): their sum, and the last.
fn walk(costed: &[Costed], order: &[usize], bound: &mut [bool], mut rows: f64) -> (f64, f64) {
    let mut sum = 0.0;
    for &i in order {
        rows *= costed[i].estimate(bound);
        for &v in &costed[i].variables {
            bound[v] = true;
        }
        sum += rows;
    }
    (sum, rows)
}

/// The split of a chain that is a cycle of four through an anchored node,
/// when its estimates make it cheaper than the left-deep `order`; `None` for
/// any other chain.
///
/// - The core: an input naming one variable (besides a graph variable every
///   pattern names) hangs off it, one naming two is an edge between them.
///   A variable on at most one edge is peeled, and it and what hangs off it
///   hang off the edge's other end. What is left must be four variables on
///   two edges each, a cycle a-b-c-d with each edge a pattern or a UNION of
///   patterns with a fixed predicate. An input naming none or more than two,
///   a variable left on no edge, or any other core is no cycle of four.
/// - `a` is what the first input of `order` hangs off, `c` the node opposite.
/// - The cost: the rows after each step, summed. The split reads half b's
///   steps, half d's from the anchor's rows, and the chain's own final rows
///   (both give the same rows); it is taken only when that is less than the
///   left-deep sum.
fn four_cycle(
    inputs: &[&LogicalOperator],
    costed: &[Costed],
    order: &[usize],
    names: &[String],
) -> Option<Split> {
    if inputs.len() < 4 {
        return None;
    }
    let graph = graph_variable(inputs).and_then(|g| names.iter().position(|name| *name == g));
    let own: Vec<Vec<usize>> = costed
        .iter()
        .map(|input| {
            let mut own: Vec<usize> = input
                .variables
                .iter()
                .copied()
                .filter(|&v| Some(v) != graph)
                .collect();
            own.sort_unstable();
            own.dedup();
            own
        })
        .collect();
    if own.iter().any(|own| own.is_empty() || own.len() > 2) {
        return None;
    }
    let mut named = vec![false; names.len()];
    for &v in own.iter().flatten() {
        named[v] = true;
    }
    // A peel takes one variable and one edge, and four of each must be left:
    // a chain with another count of edges than of variables has no such core.
    let edge_count = own.iter().filter(|own| own.len() == 2).count();
    if edge_count < 4 || edge_count != named.iter().filter(|&&named| named).count() {
        return None;
    }
    let mut edges: Vec<Vec<usize>> = vec![Vec::new(); names.len()];
    let mut hang: Vec<Vec<usize>> = vec![Vec::new(); names.len()];
    for (i, own) in own.iter().enumerate() {
        match own[..] {
            [v] => hang[v].push(i),
            [v, w] => {
                edges[v].push(i);
                edges[w].push(i);
            }
            _ => return None,
        }
    }
    let other = |edge: usize, v: usize| own[edge].iter().copied().find(|&w| w != v);
    let mut alive = vec![true; inputs.len()];
    let mut peeled = vec![false; names.len()];
    let mut queue: Vec<usize> = (0..names.len()).rev().filter(|&v| named[v]).collect();
    while let Some(v) = queue.pop() {
        if peeled[v] {
            continue;
        }
        let live: Vec<usize> = edges[v].iter().copied().filter(|&e| alive[e]).collect();
        if let [edge] = live[..] {
            let w = other(edge, v)?;
            alive[edge] = false;
            peeled[v] = true;
            let moved = std::mem::take(&mut hang[v]);
            hang[w].extend(moved);
            hang[w].push(edge);
            queue.push(w);
        }
    }
    let live =
        |v: usize| -> Vec<usize> { edges[v].iter().copied().filter(|&e| alive[e]).collect() };
    let core: Vec<usize> = (0..names.len())
        .filter(|&v| named[v] && !peeled[v])
        .collect();
    if core.len() != 4
        || core.iter().any(|&v| live(v).len() != 2)
        || core
            .iter()
            .flat_map(|&v| live(v))
            .any(|edge| !is_edge(inputs[edge]))
    {
        return None;
    }
    let a = core
        .iter()
        .copied()
        .find(|&v| hang[v].contains(&order[0]))?;
    // b is the neighbour whose edge the order reaches first.
    let mut from_a = live(a);
    from_a.sort_by_key(|edge| order.iter().position(|i| i == edge));
    let (edge_ab, edge_ad) = (from_a[0], from_a[1]);
    let (b, d) = (other(edge_ab, a)?, other(edge_ad, a)?);
    let edge_bc = live(b).into_iter().find(|&edge| edge != edge_ab)?;
    let edge_dc = live(d).into_iter().find(|&edge| edge != edge_ad)?;
    let c = other(edge_bc, b)?;
    if b == d || c == a || other(edge_dc, d)? != c {
        return None;
    }
    let fresh = || vec![false; names.len()];
    let (left_deep, last) = walk(costed, order, &mut fresh(), 1.0);
    let anchor = greedy(costed, hang[a].clone(), &mut fresh());
    let (_, anchor_rows) = walk(costed, &anchor, &mut fresh(), 1.0);
    let mut half_b_inputs = hang[a].clone();
    half_b_inputs.push(edge_ab);
    half_b_inputs.extend(&hang[b]);
    half_b_inputs.push(edge_bc);
    let half_b = greedy(costed, half_b_inputs, &mut fresh());
    let mut bound_b = fresh();
    let (half_b_rows, _) = walk(costed, &half_b, &mut bound_b, 1.0);
    let mut half_d_inputs = vec![edge_ad];
    half_d_inputs.extend(&hang[d]);
    half_d_inputs.push(edge_dc);
    // Half d's anchor keeps `a` and what else its inputs bind that half d's
    // inputs name (a graph variable): never a variable it does not bind (a
    // `VALUES` outside `GRAPH ?g` binds no `?g`, and the halves then join on
    // the one their edges bind).
    let mut kept: Vec<usize> = hang[a]
        .iter()
        .flat_map(|&input| costed[input].variables.iter().copied())
        .filter(|&v| {
            v == a
                || half_d_inputs
                    .iter()
                    .any(|&input| costed[input].variables.contains(&v))
        })
        .collect();
    kept.sort_unstable_by_key(|&v| (v != a, v));
    kept.dedup();
    let mut bound_d = fresh();
    for &v in &kept {
        bound_d[v] = true;
    }
    let half_d = greedy(costed, half_d_inputs, &mut bound_d.clone());
    let (half_d_rows, _) = walk(costed, &half_d, &mut bound_d, anchor_rows);
    if half_b_rows + half_d_rows + last >= left_deep {
        return None;
    }
    let far = greedy(costed, hang[c].clone(), &mut bound_b);
    let keep = kept.iter().map(|&v| names[v].clone()).collect();
    Some(Split {
        anchor,
        half_b,
        half_d,
        far,
        keep,
    })
}

/// The graph variable every pattern of `inputs` reads (`GRAPH ?g { ... }`),
/// `None` when a pattern reads another graph or none names one.
fn graph_variable(inputs: &[&LogicalOperator]) -> Option<String> {
    let mut graphs = inputs.iter().flat_map(|input| -> Vec<Option<&str>> {
        match input {
            LogicalOperator::TripleScan(scan) => {
                vec![scan.graph.as_ref().and_then(TripleComponent::as_variable)]
            }
            LogicalOperator::Union(union) => union
                .inputs
                .iter()
                .map(|branch| {
                    union_branch_pattern(branch)
                        .and_then(|scan| scan.graph.as_ref())
                        .and_then(TripleComponent::as_variable)
                })
                .collect(),
            _ => Vec::new(),
        }
    });
    let first = graphs.next()??;
    graphs
        .all(|graph| graph == Some(first))
        .then(|| first.to_string())
}

/// Whether `input` is an edge a cycle can split at: a pattern, or a UNION of
/// patterns (an alternative path), each with a fixed predicate.
fn is_edge(input: &LogicalOperator) -> bool {
    let fixed =
        |scan: &TripleScanOp| scan.input.is_none() && scan.predicate.as_variable().is_none();
    match input {
        LogicalOperator::TripleScan(scan) => fixed(scan),
        LogicalOperator::Union(union) => union
            .inputs
            .iter()
            .all(|branch| union_branch_pattern(branch).is_some_and(fixed)),
        _ => false,
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

#[cfg(all(test, feature = "sparql"))]
mod tests {
    use super::*;

    /// 100 persons (`v:Person`, one `v:id` each) and 100 posts (`v:Post`): two classes. KNOWS:
    /// 6 links out of each person (0.06 per check of one direction); `v:likes` 20 per person,
    /// `v:likes2` 50; `v:tag` one per liked thing.
    fn store() -> Arc<RdfStore> {
        use std::fmt::Write as _;
        let v = "http://example.org/v/";
        let rdf_type = "http://www.w3.org/1999/02/22-rdf-syntax-ns#type";
        let mut text = String::new();
        for i in 0..100 {
            let person = format!("<{v}p{i}>");
            writeln!(text, "{person} <{rdf_type}> <{v}Person> .").unwrap();
            writeln!(text, "<{v}post{i}> <{rdf_type}> <{v}Post> .").unwrap();
            writeln!(text, "{person} <{v}id> \"p{i}\" .").unwrap();
            for k in 1..=6 {
                writeln!(text, "{person} <{v}KNOWS> <{v}p{}> .", (i + k) % 100).unwrap();
            }
            for j in 0..20 {
                writeln!(text, "{person} <{v}likes> <{v}l{}> .", i * 20 + j).unwrap();
            }
            for j in 0..50 {
                writeln!(text, "{person} <{v}likes2> <{v}m{}> .", i * 50 + j).unwrap();
            }
        }
        for j in 0..2000 {
            writeln!(text, "<{v}l{j}> <{v}tag> <{v}t{}> .", j % 10).unwrap();
        }
        let store = Arc::new(RdfStore::new());
        store.load_ntriples(text.as_bytes()).unwrap();
        store
    }

    /// A term as the tests write it: `?x`, an IRI's last segment, a literal as `"lit"`.
    fn term(component: &TripleComponent) -> String {
        match component {
            TripleComponent::Variable(name) => format!("?{name}"),
            TripleComponent::Iri(iri) => iri.rsplit(['/', '#']).next().unwrap_or(iri).to_string(),
            _ => "\"lit\"".to_string(),
        }
    }

    /// A leaf as the tests write it: `?s p ?o` for a pattern, `union(a | b)`, `values`, `path`.
    fn leaf(op: &LogicalOperator) -> String {
        match op {
            LogicalOperator::TripleScan(scan) => format!(
                "{} {} {}",
                term(&scan.subject),
                term(&scan.predicate),
                term(&scan.object)
            ),
            LogicalOperator::Project(project) => leaf(&project.input),
            LogicalOperator::Union(_) if rows_of(op).is_some() => "values".to_string(),
            LogicalOperator::Union(union) => format!(
                "union({})",
                union
                    .inputs
                    .iter()
                    .map(leaf)
                    .collect::<Vec<_>>()
                    .join(" | ")
            ),
            LogicalOperator::Bind(_) | LogicalOperator::Empty => "values".to_string(),
            LogicalOperator::PropertyPath(_) => "path".to_string(),
            LogicalOperator::Join(_) => "join".to_string(),
            _ => "other".to_string(),
        }
    }

    /// The leaves of the first chain of inner joins of `query`, in the order the pass gives.
    fn order(query: &str) -> Vec<String> {
        fn chain(op: &LogicalOperator) -> Option<&LogicalOperator> {
            match op {
                LogicalOperator::Join(join) if join.join_type == JoinType::Inner => Some(op),
                other => other.children().into_iter().find_map(chain),
            }
        }
        let query = format!("PREFIX v: <http://example.org/v/> SELECT * WHERE {{ {query} }}");
        let plan = crate::query::translators::sparql::translate(
            &query,
            &grafeo_adapters::query::limits::ParseLimits::default(),
        )
        .unwrap();
        let ordered = RdfOrder::new(store().get_or_collect_statistics()).order(plan.root);
        let mut leaves = Vec::new();
        inputs_of(chain(&ordered).expect("a chain"), &mut leaves);
        leaves.into_iter().map(leaf).collect()
    }

    /// The first chain of inner joins of `query` after the pass, as a tree: `(left . right)` for
    /// a join, `distinct(...)` for half d's anchor, a leaf as [`leaf`] writes it.
    fn joined(query: &str) -> String {
        fn chain(op: &LogicalOperator) -> Option<&LogicalOperator> {
            match op {
                LogicalOperator::Join(join) if join.join_type == JoinType::Inner => Some(op),
                other => other.children().into_iter().find_map(chain),
            }
        }
        fn render(op: &LogicalOperator) -> String {
            match op {
                LogicalOperator::Join(join) if join.join_type == JoinType::Inner => {
                    format!("({} . {})", render(&join.left), render(&join.right))
                }
                LogicalOperator::Distinct(distinct) => match distinct.input.as_ref() {
                    LogicalOperator::Project(project) => {
                        format!("distinct({})", render(&project.input))
                    }
                    other => format!("distinct({})", render(other)),
                },
                other => leaf(other),
            }
        }
        let query = format!("PREFIX v: <http://example.org/v/> SELECT * WHERE {{ {query} }}");
        let plan = crate::query::translators::sparql::translate(
            &query,
            &grafeo_adapters::query::limits::ParseLimits::default(),
        )
        .unwrap();
        let ordered = RdfOrder::new(store().get_or_collect_statistics()).order(plan.root);
        render(chain(&ordered).expect("a chain"))
    }

    const CYCLE: &str = "?a v:id \"p0\" . ?a (v:KNOWS|^v:KNOWS) ?b . ?b (v:KNOWS|^v:KNOWS) ?c .
         ?c (v:KNOWS|^v:KNOWS) ?d . ?d (v:KNOWS|^v:KNOWS) ?a";

    #[test]
    fn a_four_cycle_through_an_anchor_joins_two_halves_at_the_far_node() {
        let step =
            |from: &str, to: &str| format!("union(?{from} KNOWS ?{to} | ?{to} KNOWS ?{from})");
        // Half b: the anchor, a-b, b-c; half d: the anchor's distinct a, a-d, d-c; joined on a, c.
        assert_eq!(
            joined(CYCLE),
            format!(
                "(((?a id \"lit\" . {}) . {}) . ((distinct(?a id \"lit\") . {}) . {}))",
                step("a", "b"),
                step("b", "c"),
                step("d", "a"),
                step("c", "d"),
            )
        );
    }

    #[test]
    fn what_hangs_off_each_node_stays_on_its_side() {
        let tree = joined(&format!(
            "{CYCLE} . ?b a v:Person . ?c a v:Person . ?d v:id ?did"
        ));
        let (half_b, half_d) = tree.split_once("distinct(").expect("split");
        assert!(half_b.contains("?b type Person"), "{tree}");
        assert!(
            half_b.contains("?c type Person"),
            "c joins after half b: {tree}"
        );
        assert!(half_d.contains("?d id ?did"), "{tree}");
    }

    #[test]
    fn a_triangle_or_a_chord_keeps_one_chain() {
        let triangle = "?a v:id \"p0\" . ?a (v:KNOWS|^v:KNOWS) ?b . ?b (v:KNOWS|^v:KNOWS) ?c .
             ?c (v:KNOWS|^v:KNOWS) ?a";
        for query in [
            triangle.to_string(),
            format!("{CYCLE} . ?a (v:KNOWS|^v:KNOWS) ?c"),
            format!("{CYCLE} . ?a (v:KNOWS|^v:KNOWS) ?b"),
        ] {
            assert!(!joined(&query).contains("distinct("), "{query}");
        }
    }

    const CLOSING: &str = "union(?c KNOWS ?a | ?a KNOWS ?c)";

    #[test]
    fn a_closing_check_goes_before_a_class_check_and_an_expansion() {
        // Closing 0.12, the class check 0.5, the id 1.0.
        let order = order(
            "?a v:id \"p0\" . ?a (v:KNOWS|^v:KNOWS) ?b . ?b (v:KNOWS|^v:KNOWS) ?c .
             ?c a v:Person . ?c v:id ?cid . ?c (v:KNOWS|^v:KNOWS) ?a",
        );
        assert_eq!(
            order,
            [
                "?a id \"lit\"",
                "union(?a KNOWS ?b | ?b KNOWS ?a)",
                "union(?b KNOWS ?c | ?c KNOWS ?b)",
                CLOSING,
                "?c type Person",
                "?c id ?cid",
            ]
        );
    }

    #[test]
    fn a_written_union_of_patterns_orders_as_the_path() {
        let order = order(
            "?a v:id \"p0\" . ?a v:KNOWS ?b . ?b v:KNOWS ?c .
             ?c a v:Person . ?c v:id ?cid . { ?c v:KNOWS ?a } UNION { ?a v:KNOWS ?c }",
        );
        assert_eq!(order[3], CLOSING, "{order:?}");
    }

    #[test]
    fn a_union_with_one_end_bound_is_the_sum_of_its_branches() {
        // KNOWS both ways: 12 per person, against 20 likes.
        let order = order("?a v:id \"p0\" . ?a v:likes ?l . ?a (v:KNOWS|^v:KNOWS) ?b");
        assert_eq!(
            order,
            [
                "?a id \"lit\"",
                "union(?a KNOWS ?b | ?b KNOWS ?a)",
                "?a likes ?l"
            ]
        );
    }

    #[test]
    fn a_union_with_no_end_bound_is_the_sum_of_its_triples() {
        // 1,200 KNOWS triples both ways against 2,000 likes and 2,000 tags.
        let order = order("?y v:likes ?l . ?l v:tag ?t . ?x (v:KNOWS|^v:KNOWS) ?y");
        assert_eq!(
            order,
            [
                "union(?x KNOWS ?y | ?y KNOWS ?x)",
                "?y likes ?l",
                "?l tag ?t"
            ]
        );
    }

    #[test]
    fn branches_that_bind_different_variables_are_each_costed_by_their_own_ends() {
        // With ?c bound: 6 KNOWS + 20 likes, after the id (1.0), before 50 likes2.
        let order = order(
            "?c v:id \"p0\" . ?c v:likes2 ?m . ?c v:id ?cid .
             { ?c v:KNOWS ?a } UNION { ?c v:likes ?l }",
        );
        assert_eq!(
            order,
            [
                "?c id \"lit\"",
                "?c id ?cid",
                "union(?c KNOWS ?a | ?c likes ?l)",
                "?c likes2 ?m",
            ]
        );
    }

    #[test]
    fn a_union_with_a_sequence_branch_keeps_its_fixed_count() {
        let order = order(
            "?a v:id \"p0\" . ?a v:KNOWS ?c . ?c a v:Person . ?c v:id ?cid .
             ?c (v:KNOWS/v:KNOWS|v:KNOWS) ?a",
        );
        assert!(order.last().unwrap().starts_with("union("), "{order:?}");
    }

    #[test]
    fn values_keeps_its_rows() {
        let order = order("VALUES ?a { v:p0 v:p1 } ?a v:likes ?l . ?a (v:KNOWS|^v:KNOWS) ?b");
        assert_eq!(
            order,
            ["values", "union(?a KNOWS ?b | ?b KNOWS ?a)", "?a likes ?l"]
        );
    }

    #[test]
    fn a_variable_predicate_binds_a_variable_and_keeps_one_row() {
        // Both ends bound by the VALUES row: the class check (0.5) first; then ?p (1.0) and the
        // id (1.0) as written.
        let order = order("VALUES (?a ?c) { (v:p0 v:p1) } ?c ?p ?a . ?c v:id ?cid . ?c a v:Person");
        assert_eq!(
            order,
            ["values", "?c type Person", "?c ?p ?a", "?c id ?cid"]
        );
    }

    #[test]
    fn a_predicate_the_statistics_do_not_list_holds_as_an_average_one() {
        // Statistics go stale between collections (a tenth of the triples changed): a predicate
        // inserted since is not listed, and is not known to hold no triple.
        fn scan(op: &LogicalOperator) -> Option<&TripleScanOp> {
            match op {
                LogicalOperator::TripleScan(scan) => Some(scan),
                other => other.children().into_iter().find_map(scan),
            }
        }
        let query = "PREFIX v: <http://example.org/v/> SELECT * WHERE { ?c v:inserted_since ?a }";
        let plan = crate::query::translators::sparql::translate(
            query,
            &grafeo_adapters::query::limits::ParseLimits::default(),
        )
        .unwrap();
        let statistics = store().get_or_collect_statistics();
        let holds = check_holds_in(&statistics, scan(&plan.root).expect("a pattern"));
        let average = estimate_of(statistics.total_triples)
            / estimate_of(statistics.predicate_count)
            / (estimate_of(statistics.subject_count) * estimate_of(statistics.object_count));
        assert!(holds > 0.0);
        assert!(
            (holds - average).abs() <= average * 1e-12,
            "{holds} against {average}"
        );
    }

    #[test]
    fn a_check_on_a_predicate_with_no_triples_goes_first_as_written() {
        // Both ends bound by the VALUES row: the two checks on predicates the statistics do not
        // list, as written (an average predicate's chance, far below the class check's 0.5),
        // then the class check, then the id.
        let order = order(
            "VALUES (?a ?c) { (v:p0 v:p1) } ?c v:id ?cid . ?c a v:Person .
             ?c v:never2 ?a . ?c v:never ?a",
        );
        assert_eq!(
            order,
            [
                "values",
                "?c never2 ?a",
                "?c never ?a",
                "?c type Person",
                "?c id ?cid",
            ]
        );
    }

    #[test]
    fn a_path_check_keeps_one_row() {
        // The class check (0.5) first; the path (1.0) and the id (1.0) as written.
        let order =
            order("?a v:id \"p0\" . ?a v:KNOWS ?c . ?c v:KNOWS+ ?a . ?c v:id ?cid . ?c a v:Person");
        assert_eq!(
            order,
            [
                "?a id \"lit\"",
                "?a KNOWS ?c",
                "?c type Person",
                "path",
                "?c id ?cid"
            ]
        );
    }
}
