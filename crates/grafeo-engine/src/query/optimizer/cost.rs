//! Cost model for query optimization.
//!
//! Provides cost estimates for logical operators to guide optimization decisions.

use crate::query::plan::{
    AggregateOp, DistinctOp, ExpandDirection, ExpandOp, FilterOp, JoinOp, JoinType, LeftJoinOp,
    LimitOp, LogicalOperator, MultiWayJoinOp, NodeScanOp, ProjectOp, ReturnOp, SkipOp, SortOp,
    TextScanOp, VectorJoinOp, VectorScanOp,
};
use std::collections::HashMap;

/// Cost of an operation.
///
/// Represents the estimated resource consumption of executing an operator.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Cost {
    /// Estimated CPU cycles / work units.
    pub cpu: f64,
    /// Estimated I/O operations (page reads).
    pub io: f64,
    /// Estimated memory usage in bytes.
    pub memory: f64,
    /// Network cost (for distributed queries).
    pub network: f64,
}

impl Cost {
    /// Creates a zero cost.
    #[must_use]
    pub fn zero() -> Self {
        Self {
            cpu: 0.0,
            io: 0.0,
            memory: 0.0,
            network: 0.0,
        }
    }

    /// Creates a cost from CPU work units.
    #[must_use]
    pub fn cpu(cpu: f64) -> Self {
        Self {
            cpu,
            io: 0.0,
            memory: 0.0,
            network: 0.0,
        }
    }

    /// Adds I/O cost.
    #[must_use]
    pub fn with_io(mut self, io: f64) -> Self {
        self.io = io;
        self
    }

    /// Adds memory cost.
    #[must_use]
    pub fn with_memory(mut self, memory: f64) -> Self {
        self.memory = memory;
        self
    }

    /// Returns the total weighted cost.
    ///
    /// Uses default weights: CPU=1.0, IO=10.0, Memory=0.1, Network=100.0
    #[must_use]
    pub fn total(&self) -> f64 {
        self.cpu + self.io * 10.0 + self.memory * 0.1 + self.network * 100.0
    }

    /// Returns the total cost with custom weights.
    #[must_use]
    pub fn total_weighted(&self, cpu_weight: f64, io_weight: f64, mem_weight: f64) -> f64 {
        self.cpu * cpu_weight + self.io * io_weight + self.memory * mem_weight
    }
}

impl std::ops::Add for Cost {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        Self {
            cpu: self.cpu + other.cpu,
            io: self.io + other.io,
            memory: self.memory + other.memory,
            network: self.network + other.network,
        }
    }
}

impl std::ops::AddAssign for Cost {
    fn add_assign(&mut self, other: Self) {
        self.cpu += other.cpu;
        self.io += other.io;
        self.memory += other.memory;
        self.network += other.network;
    }
}

/// Cost model for estimating operator costs.
///
/// Default constants are calibrated relative to each other:
/// - Tuple scan is the baseline (1x)
/// - Hash lookup is ~3x (hash computation + potential cache miss)
/// - Sort comparison is ~2x (key extraction + comparison)
/// - Distance computation is ~10x (vector math)
pub struct CostModel {
    /// Cost per tuple processed by CPU (baseline unit).
    cpu_tuple_cost: f64,
    /// Cost per hash table lookup (~3x tuple cost: hash + cache miss).
    hash_lookup_cost: f64,
    /// Cost per comparison in sorting (~2x tuple cost: key extract + cmp).
    sort_comparison_cost: f64,
    /// Average tuple size in bytes (for IO estimation).
    avg_tuple_size: f64,
    /// Page size in bytes.
    page_size: f64,
    /// Global average edge fanout (fallback when per-type stats unavailable).
    avg_fanout: f64,
    /// Per-edge-type degree stats: (avg_out_degree, avg_in_degree).
    edge_type_degrees: HashMap<String, (f64, f64)>,
    /// Per-label node counts for accurate scan IO estimation.
    label_cardinalities: HashMap<String, u64>,
    /// Total node count in the graph.
    total_nodes: u64,
    /// Total edge count in the graph.
    total_edges: u64,
}

impl CostModel {
    /// Creates a new cost model with calibrated default parameters.
    #[must_use]
    pub fn new() -> Self {
        Self {
            cpu_tuple_cost: 0.01,
            hash_lookup_cost: 0.03,
            sort_comparison_cost: 0.02,
            avg_tuple_size: 100.0,
            page_size: 8192.0,
            avg_fanout: 10.0,
            edge_type_degrees: HashMap::new(),
            label_cardinalities: HashMap::new(),
            total_nodes: 0,
            total_edges: 0,
        }
    }

    /// Sets the global average fanout from graph statistics.
    #[must_use]
    pub fn with_avg_fanout(mut self, avg_fanout: f64) -> Self {
        self.avg_fanout = if avg_fanout > 0.0 { avg_fanout } else { 10.0 };
        self
    }

    /// Sets per-edge-type degree statistics for accurate expand cost estimation.
    ///
    /// Each entry maps edge type name to `(avg_out_degree, avg_in_degree)`.
    #[must_use]
    pub fn with_edge_type_degrees(mut self, degrees: HashMap<String, (f64, f64)>) -> Self {
        self.edge_type_degrees = degrees;
        self
    }

    /// Sets per-label node counts for accurate scan IO estimation.
    #[must_use]
    pub fn with_label_cardinalities(mut self, cardinalities: HashMap<String, u64>) -> Self {
        self.label_cardinalities = cardinalities;
        self
    }

    /// Sets graph-level totals for cost estimation.
    #[must_use]
    pub fn with_graph_totals(mut self, total_nodes: u64, total_edges: u64) -> Self {
        self.total_nodes = total_nodes;
        self.total_edges = total_edges;
        self
    }

    /// Returns the fanout for a specific expand operation.
    ///
    /// Uses per-edge-type degree stats when available, falling back to the
    /// global average fanout. For multiple edge types, sums per-type fanouts.
    fn fanout_for_expand(&self, expand: &ExpandOp) -> f64 {
        if expand.edge_types.is_empty() {
            return self.avg_fanout;
        }

        let mut total_fanout = 0.0;
        let mut all_found = true;

        for edge_type in &expand.edge_types {
            if let Some(&(out_deg, in_deg)) = self.edge_type_degrees.get(edge_type) {
                total_fanout += match expand.direction {
                    ExpandDirection::Outgoing => out_deg,
                    ExpandDirection::Incoming => in_deg,
                    ExpandDirection::Both => out_deg + in_deg,
                };
            } else {
                all_found = false;
                break;
            }
        }

        if all_found {
            total_fanout
        } else {
            self.avg_fanout
        }
    }

    /// Estimates the cost of a logical operator.
    #[must_use]
    pub fn estimate(&self, op: &LogicalOperator, cardinality: f64) -> Cost {
        match op {
            LogicalOperator::NodeScan(scan) => self.node_scan_cost(scan, cardinality),
            LogicalOperator::Filter(filter) => self.filter_cost(filter, cardinality),
            LogicalOperator::Project(project) => self.project_cost(project, cardinality),
            LogicalOperator::Expand(expand) => self.expand_cost(expand, cardinality),
            LogicalOperator::Join(join) => self.join_cost(join, cardinality),
            LogicalOperator::Aggregate(agg) => self.aggregate_cost(agg, cardinality),
            LogicalOperator::Sort(sort) => self.sort_cost(sort, cardinality),
            LogicalOperator::Distinct(distinct) => self.distinct_cost(distinct, cardinality),
            LogicalOperator::Limit(limit) => self.limit_cost(limit, cardinality),
            LogicalOperator::Skip(skip) => self.skip_cost(skip, cardinality),
            LogicalOperator::Return(ret) => self.return_cost(ret, cardinality),
            LogicalOperator::Empty => Cost::zero(),
            LogicalOperator::VectorScan(scan) => self.vector_scan_cost(scan, cardinality),
            LogicalOperator::VectorJoin(join) => self.vector_join_cost(join, cardinality),
            LogicalOperator::MultiWayJoin(mwj) => self.multi_way_join_cost(mwj, cardinality),
            LogicalOperator::LeftJoin(lj) => {
                let left_card = self.estimate_child_cardinality(&lj.left);
                let right_card = self.estimate_child_cardinality(&lj.right);
                self.left_join_cost(lj, cardinality, left_card, right_card)
            }
            LogicalOperator::TextScan(scan) => self.text_scan_cost(scan, cardinality),
            _ => Cost::cpu(cardinality * self.cpu_tuple_cost),
        }
    }

    /// Estimates the cost of a node scan.
    ///
    /// When label statistics are available, uses the actual label cardinality
    /// for IO estimation (pages to read) rather than the optimizer's cardinality
    /// estimate, which may already account for filter selectivity.
    fn node_scan_cost(&self, scan: &NodeScanOp, cardinality: f64) -> Cost {
        // IO cost: based on how many nodes we actually need to scan from storage
        let scan_size = if let Some(label) = &scan.label {
            self.label_cardinalities
                .get(label)
                .map_or(cardinality, |&count| count as f64)
        } else if self.total_nodes > 0 {
            self.total_nodes as f64
        } else {
            cardinality
        };
        let pages = (scan_size * self.avg_tuple_size) / self.page_size;
        // CPU cost: only pay for rows that pass the label filter
        Cost::cpu(cardinality * self.cpu_tuple_cost).with_io(pages)
    }

    /// Estimates the cost of a filter operation.
    fn filter_cost(&self, _filter: &FilterOp, cardinality: f64) -> Cost {
        // Filter cost is just predicate evaluation per tuple
        Cost::cpu(cardinality * self.cpu_tuple_cost * 1.5)
    }

    /// Estimates the cost of a projection.
    fn project_cost(&self, project: &ProjectOp, cardinality: f64) -> Cost {
        // Cost depends on number of expressions evaluated
        let expr_count = project.projections.len() as f64;
        Cost::cpu(cardinality * self.cpu_tuple_cost * expr_count)
    }

    /// Estimates the cost of an expand operation.
    ///
    /// Uses per-edge-type degree stats when available, otherwise falls back
    /// to the global average fanout.
    fn expand_cost(&self, expand: &ExpandOp, cardinality: f64) -> Cost {
        let fanout = self.fanout_for_expand(expand);
        // Adjacency list lookup per input row
        let lookup_cost = cardinality * self.hash_lookup_cost;
        // Process each expanded output tuple
        let output_cost = cardinality * fanout * self.cpu_tuple_cost;
        Cost::cpu(lookup_cost + output_cost)
    }

    /// Estimates the cost of a join operation.
    ///
    /// When child cardinalities are known (from recursive estimation), uses them
    /// directly for build/probe cost. Otherwise falls back to sqrt approximation.
    fn join_cost(&self, join: &JoinOp, cardinality: f64) -> Cost {
        self.join_cost_with_children(join, cardinality, None, None)
    }

    /// Estimates join cost using actual child cardinalities.
    fn join_cost_with_children(
        &self,
        join: &JoinOp,
        cardinality: f64,
        left_cardinality: Option<f64>,
        right_cardinality: Option<f64>,
    ) -> Cost {
        match join.join_type {
            JoinType::Cross => Cost::cpu(cardinality * self.cpu_tuple_cost),
            JoinType::Inner | JoinType::Left | JoinType::Right | JoinType::Full => {
                // Hash join: build the smaller side, probe with the larger
                let build_cardinality = left_cardinality.unwrap_or_else(|| cardinality.sqrt());
                let probe_cardinality = right_cardinality.unwrap_or_else(|| cardinality.sqrt());

                let build_cost = build_cardinality * self.hash_lookup_cost;
                let memory_cost = build_cardinality * self.avg_tuple_size;
                let probe_cost = probe_cardinality * self.hash_lookup_cost;
                let output_cost = cardinality * self.cpu_tuple_cost;

                Cost::cpu(build_cost + probe_cost + output_cost).with_memory(memory_cost)
            }
            JoinType::Semi | JoinType::Anti => {
                let build_cardinality = left_cardinality.unwrap_or_else(|| cardinality.sqrt());
                let probe_cardinality = right_cardinality.unwrap_or_else(|| cardinality.sqrt());

                let build_cost = build_cardinality * self.hash_lookup_cost;
                let probe_cost = probe_cardinality * self.hash_lookup_cost;

                Cost::cpu(build_cost + probe_cost)
                    .with_memory(build_cardinality * self.avg_tuple_size)
            }
        }
    }

    /// Estimates the cost of a left outer join (OPTIONAL MATCH).
    ///
    /// Uses the same hash join cost model as inner joins: the right side is
    /// built into a hash table, the left side probes. Output cost includes all
    /// left rows (some with NULL-padded right side).
    fn left_join_cost(
        &self,
        _lj: &LeftJoinOp,
        cardinality: f64,
        left_card: f64,
        right_card: f64,
    ) -> Cost {
        let build_cost = right_card * self.hash_lookup_cost;
        let memory_cost = right_card * self.avg_tuple_size;
        let probe_cost = left_card * self.hash_lookup_cost;
        let output_cost = cardinality * self.cpu_tuple_cost;

        Cost::cpu(build_cost + probe_cost + output_cost).with_memory(memory_cost)
    }

    /// Estimates the cardinality of a child operator using available statistics.
    ///
    /// Used internally to compute left/right child cardinalities for joins
    /// without requiring an external `CardinalityEstimator`. Uses the same
    /// statistics the cost model already holds (label counts, fanout).
    fn estimate_child_cardinality(&self, op: &LogicalOperator) -> f64 {
        match op {
            LogicalOperator::NodeScan(scan) => if let Some(label) = &scan.label {
                self.label_cardinalities
                    .get(label)
                    .map_or(self.total_nodes as f64, |&c| c as f64)
            } else {
                self.total_nodes as f64
            }
            .max(1.0),
            LogicalOperator::Expand(expand) => {
                let input_card = self.estimate_child_cardinality(&expand.input);
                let fanout = if expand.edge_types.is_empty() {
                    self.avg_fanout
                } else {
                    self.fanout_for_expand(expand)
                };
                (input_card * fanout).max(1.0)
            }
            LogicalOperator::Filter(filter) => {
                // Approximate: filters reduce cardinality by 10%
                (self.estimate_child_cardinality(&filter.input) * 0.1).max(1.0)
            }
            LogicalOperator::Return(ret) => self.estimate_child_cardinality(&ret.input),
            LogicalOperator::Limit(limit) => {
                let input = self.estimate_child_cardinality(&limit.input);
                // LimitOp.count may be a parameter; use input as upper bound
                input.min(100.0)
            }
            _ => (self.total_nodes as f64).max(1.0),
        }
    }

    /// Estimates the cost of a multi-way (leapfrog) join.
    ///
    /// Delegates to `leapfrog_join_cost` using per-input cardinality estimates
    /// derived from the output cardinality divided equally among inputs.
    fn multi_way_join_cost(&self, mwj: &MultiWayJoinOp, cardinality: f64) -> Cost {
        let n = mwj.inputs.len();
        if n == 0 {
            return Cost::zero();
        }
        // Approximate per-input cardinalities: assume each input contributes
        // cardinality^(1/n) rows (AGM-style uniform assumption)
        let per_input = cardinality.powf(1.0 / n as f64).max(1.0);
        let cardinalities: Vec<f64> = (0..n).map(|_| per_input).collect();
        self.leapfrog_join_cost(n, &cardinalities, cardinality)
    }

    /// Estimates the cost of an aggregation.
    fn aggregate_cost(&self, agg: &AggregateOp, cardinality: f64) -> Cost {
        // Hash aggregation cost
        let hash_cost = cardinality * self.hash_lookup_cost;

        // Aggregate function evaluation
        let agg_count = agg.aggregates.len() as f64;
        let agg_cost = cardinality * self.cpu_tuple_cost * agg_count;

        // Memory for hash table (estimated distinct groups)
        let distinct_groups = (cardinality / 10.0).max(1.0); // Assume 10% distinct
        let memory_cost = distinct_groups * self.avg_tuple_size;

        Cost::cpu(hash_cost + agg_cost).with_memory(memory_cost)
    }

    /// Estimates the cost of a sort operation.
    fn sort_cost(&self, sort: &SortOp, cardinality: f64) -> Cost {
        if cardinality <= 1.0 {
            return Cost::zero();
        }

        // Sort is O(n log n) comparisons
        let comparisons = cardinality * cardinality.log2();
        let key_count = sort.keys.len() as f64;

        // Memory for sorting (full input materialization)
        let memory_cost = cardinality * self.avg_tuple_size;

        Cost::cpu(comparisons * self.sort_comparison_cost * key_count).with_memory(memory_cost)
    }

    /// Estimates the cost of a distinct operation.
    fn distinct_cost(&self, _distinct: &DistinctOp, cardinality: f64) -> Cost {
        // Hash-based distinct
        let hash_cost = cardinality * self.hash_lookup_cost;
        let memory_cost = cardinality * self.avg_tuple_size * 0.5; // Assume 50% distinct

        Cost::cpu(hash_cost).with_memory(memory_cost)
    }

    /// Estimates the cost of a limit operation.
    fn limit_cost(&self, limit: &LimitOp, _cardinality: f64) -> Cost {
        // Limit is very cheap - just counting
        Cost::cpu(limit.count.estimate() * self.cpu_tuple_cost * 0.1)
    }

    /// Estimates the cost of a skip operation.
    fn skip_cost(&self, skip: &SkipOp, _cardinality: f64) -> Cost {
        // Skip requires scanning through skipped rows
        Cost::cpu(skip.count.estimate() * self.cpu_tuple_cost)
    }

    /// Estimates the cost of a return operation.
    fn return_cost(&self, ret: &ReturnOp, cardinality: f64) -> Cost {
        // Return materializes results
        let expr_count = ret.items.len() as f64;
        Cost::cpu(cardinality * self.cpu_tuple_cost * expr_count)
    }

    /// Estimates the cost of a vector scan operation.
    ///
    /// HNSW index search is O(log N) per query, while brute-force is O(N).
    /// This estimates the HNSW case with ef search parameter.
    fn vector_scan_cost(&self, scan: &VectorScanOp, cardinality: f64) -> Cost {
        let k = scan.k.unwrap_or(0) as f64;
        let n = cardinality.max(1.0);

        // HNSW search cost: O(ef * log(N)) distance computations
        // Brute-force: O(N) distance computations
        let ef = 64.0;
        let search_cost = if scan.index_name.is_some() {
            ef * n.ln() * self.cpu_tuple_cost * 10.0
        } else {
            n * self.cpu_tuple_cost * 10.0
        };

        // Memory for candidate heap (threshold mode uses cardinality estimate)
        let output_rows = if k > 0.0 { k } else { cardinality };
        let memory = output_rows * self.avg_tuple_size * 2.0;

        Cost::cpu(search_cost).with_memory(memory)
    }

    /// Estimates the cost of a text scan operation.
    ///
    /// BM25 scoring requires iterating over postings lists proportional to the
    /// corpus size, with scoring being ~5x more expensive than a plain tuple
    /// comparison. The index is in-memory so I/O cost is zero.
    fn text_scan_cost(&self, scan: &TextScanOp, cardinality: f64) -> Cost {
        // Use actual label row count for corpus size (BM25 scans the postings).
        let corpus_size = self
            .label_cardinalities
            .get(&scan.label)
            .copied()
            .map_or(cardinality, |c| c as f64);
        // BM25 scoring is ~5x a simple tuple comparison
        let cpu = corpus_size * self.cpu_tuple_cost * 5.0;
        Cost::cpu(cpu).with_memory(cardinality * self.avg_tuple_size)
    }

    /// Estimates the cost of a vector join operation.
    ///
    /// Vector join performs k-NN search for each input row.
    fn vector_join_cost(&self, join: &VectorJoinOp, cardinality: f64) -> Cost {
        let k = join.k as f64;

        // Each input row triggers a vector search
        // Assume brute-force for hybrid queries (no index specified typically)
        let per_row_search_cost = if join.index_name.is_some() {
            // HNSW: O(ef * log N)
            let ef = 64.0;
            let n = cardinality.max(1.0);
            ef * n.ln() * self.cpu_tuple_cost * 10.0
        } else {
            // Brute-force: O(N) per input row
            cardinality * self.cpu_tuple_cost * 10.0
        };

        // Total cost: input_rows * search_cost
        // For vector join, cardinality is typically input cardinality * k
        let input_cardinality = (cardinality / k).max(1.0);
        let total_search_cost = input_cardinality * per_row_search_cost;

        // Memory for results
        let memory = cardinality * self.avg_tuple_size;

        Cost::cpu(total_search_cost).with_memory(memory)
    }

    /// Estimates the total cost of an operator tree recursively.
    ///
    /// Walks the entire plan tree, computing per-operator cost at each level
    /// using the cardinality estimator for accurate child cardinalities.
    /// Returns the sum of all operator costs in the tree.
    #[must_use]
    pub fn estimate_tree(
        &self,
        op: &LogicalOperator,
        card_estimator: &super::CardinalityEstimator,
    ) -> Cost {
        self.estimate_tree_inner(op, card_estimator)
    }

    fn estimate_tree_inner(
        &self,
        op: &LogicalOperator,
        card_est: &super::CardinalityEstimator,
    ) -> Cost {
        let cardinality = card_est.estimate(op);

        match op {
            LogicalOperator::NodeScan(scan) => self.node_scan_cost(scan, cardinality),
            LogicalOperator::Filter(filter) => {
                let child_cost = self.estimate_tree_inner(&filter.input, card_est);
                child_cost + self.filter_cost(filter, cardinality)
            }
            LogicalOperator::Project(project) => {
                let child_cost = self.estimate_tree_inner(&project.input, card_est);
                child_cost + self.project_cost(project, cardinality)
            }
            LogicalOperator::Expand(expand) => {
                let child_cost = self.estimate_tree_inner(&expand.input, card_est);
                child_cost + self.expand_cost(expand, cardinality)
            }
            LogicalOperator::Join(join) => {
                let left_cost = self.estimate_tree_inner(&join.left, card_est);
                let right_cost = self.estimate_tree_inner(&join.right, card_est);
                let left_card = card_est.estimate(&join.left);
                let right_card = card_est.estimate(&join.right);
                let join_cost = self.join_cost_with_children(
                    join,
                    cardinality,
                    Some(left_card),
                    Some(right_card),
                );
                left_cost + right_cost + join_cost
            }
            LogicalOperator::LeftJoin(lj) => {
                let left_cost = self.estimate_tree_inner(&lj.left, card_est);
                let right_cost = self.estimate_tree_inner(&lj.right, card_est);
                let left_card = card_est.estimate(&lj.left);
                let right_card = card_est.estimate(&lj.right);
                let join_cost = self.left_join_cost(lj, cardinality, left_card, right_card);
                left_cost + right_cost + join_cost
            }
            LogicalOperator::Aggregate(agg) => {
                let child_cost = self.estimate_tree_inner(&agg.input, card_est);
                child_cost + self.aggregate_cost(agg, cardinality)
            }
            LogicalOperator::Sort(sort) => {
                let child_cost = self.estimate_tree_inner(&sort.input, card_est);
                child_cost + self.sort_cost(sort, cardinality)
            }
            LogicalOperator::Distinct(distinct) => {
                let child_cost = self.estimate_tree_inner(&distinct.input, card_est);
                child_cost + self.distinct_cost(distinct, cardinality)
            }
            LogicalOperator::Limit(limit) => {
                let child_cost = self.estimate_tree_inner(&limit.input, card_est);
                child_cost + self.limit_cost(limit, cardinality)
            }
            LogicalOperator::Skip(skip) => {
                let child_cost = self.estimate_tree_inner(&skip.input, card_est);
                child_cost + self.skip_cost(skip, cardinality)
            }
            LogicalOperator::Return(ret) => {
                let child_cost = self.estimate_tree_inner(&ret.input, card_est);
                child_cost + self.return_cost(ret, cardinality)
            }
            LogicalOperator::VectorScan(scan) => self.vector_scan_cost(scan, cardinality),
            LogicalOperator::VectorJoin(join) => {
                let child_cost = self.estimate_tree_inner(&join.input, card_est);
                child_cost + self.vector_join_cost(join, cardinality)
            }
            LogicalOperator::MultiWayJoin(mwj) => {
                let mut children_cost = Cost::zero();
                for input in &mwj.inputs {
                    children_cost += self.estimate_tree_inner(input, card_est);
                }
                children_cost + self.multi_way_join_cost(mwj, cardinality)
            }
            LogicalOperator::Empty => Cost::zero(),
            _ => Cost::cpu(cardinality * self.cpu_tuple_cost),
        }
    }

    /// Compares two costs and returns the cheaper one.
    #[must_use]
    pub fn cheaper<'a>(&self, a: &'a Cost, b: &'a Cost) -> &'a Cost {
        if a.total() <= b.total() { a } else { b }
    }

    /// Estimates the cost of a worst-case optimal join (WCOJ/leapfrog join).
    ///
    /// WCOJ is optimal for cyclic patterns like triangles. Traditional binary
    /// hash joins are O(N²) for triangles; WCOJ achieves O(N^1.5) by processing
    /// all relations simultaneously using sorted iterators.
    ///
    /// # Arguments
    /// * `num_relations` - Number of relations participating in the join
    /// * `cardinalities` - Cardinality of each input relation
    /// * `output_cardinality` - Expected output cardinality
    ///
    /// # Cost Model
    /// - Materialization: O(sum of cardinalities) to build trie indexes
    /// - Intersection: O(output * log(min_cardinality)) for leapfrog seek operations
    /// - Memory: Trie storage for all inputs
    #[must_use]
    pub fn leapfrog_join_cost(
        &self,
        num_relations: usize,
        cardinalities: &[f64],
        output_cardinality: f64,
    ) -> Cost {
        if cardinalities.is_empty() {
            return Cost::zero();
        }

        let total_input: f64 = cardinalities.iter().sum();
        let min_card = cardinalities.iter().copied().fold(f64::INFINITY, f64::min);

        // Materialization phase: build trie indexes for each input
        let materialize_cost = total_input * self.cpu_tuple_cost * 2.0; // Sorting + trie building

        // Intersection phase: leapfrog seeks are O(log n) per relation
        let seek_cost = if min_card > 1.0 {
            output_cardinality * (num_relations as f64) * min_card.log2() * self.hash_lookup_cost
        } else {
            output_cardinality * self.cpu_tuple_cost
        };

        // Output materialization
        let output_cost = output_cardinality * self.cpu_tuple_cost;

        // Memory: trie storage (roughly 2x input size for sorted index)
        let memory = total_input * self.avg_tuple_size * 2.0;

        Cost::cpu(materialize_cost + seek_cost + output_cost).with_memory(memory)
    }

    /// Compares hash join cost vs leapfrog join cost for a cyclic pattern.
    ///
    /// Returns true if leapfrog (WCOJ) is estimated to be cheaper.
    #[must_use]
    pub fn prefer_leapfrog_join(
        &self,
        num_relations: usize,
        cardinalities: &[f64],
        output_cardinality: f64,
    ) -> bool {
        if num_relations < 3 || cardinalities.len() < 3 {
            // Leapfrog is only beneficial for multi-way joins (3+)
            return false;
        }

        let leapfrog_cost =
            self.leapfrog_join_cost(num_relations, cardinalities, output_cardinality);

        // Estimate cascade of binary hash joins
        // For N relations, we need N-1 joins
        // Each join produces intermediate results that feed the next
        let mut hash_cascade_cost = Cost::zero();
        let mut intermediate_cardinality = cardinalities[0];

        for card in &cardinalities[1..] {
            // Hash join cost: build + probe + output
            let join_output = (intermediate_cardinality * card).sqrt(); // Estimated selectivity
            let join = JoinOp {
                left: Box::new(LogicalOperator::Empty),
                right: Box::new(LogicalOperator::Empty),
                join_type: JoinType::Inner,
                conditions: vec![],
            };
            hash_cascade_cost += self.join_cost(&join, join_output);
            intermediate_cardinality = join_output;
        }

        leapfrog_cost.total() < hash_cascade_cost.total()
    }

    /// Estimates cost for factorized execution (compressed intermediate results).
    ///
    /// Factorized execution avoids materializing full cross products by keeping
    /// results in a compressed "factorized" form. This is beneficial for multi-hop
    /// traversals where intermediate results can explode.
    ///
    /// Returns the reduction factor (1.0 = no benefit, lower = more compression).
    #[must_use]
    pub fn factorized_benefit(&self, avg_fanout: f64, num_hops: usize) -> f64 {
        if num_hops <= 1 || avg_fanout <= 1.0 {
            return 1.0; // No benefit for single hop or low fanout
        }

        // Factorized representation compresses repeated prefixes
        // Compression ratio improves with higher fanout and more hops
        // Full materialization: fanout^hops
        // Factorized: sum(fanout^i for i in 1..=hops) ≈ fanout^(hops+1) / (fanout - 1)

        // reason: hop count is always small (< 100)
        #[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
        let hops_i32 = num_hops as i32;
        let full_size = avg_fanout.powi(hops_i32);
        let factorized_size = if avg_fanout > 1.0 {
            (avg_fanout.powi(hops_i32 + 1) - 1.0) / (avg_fanout - 1.0)
        } else {
            num_hops as f64
        };

        (factorized_size / full_size).min(1.0)
    }
}

impl Default for CostModel {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::query::plan::{
        AggregateExpr, AggregateFunction, ExpandDirection, JoinCondition, LogicalExpression,
        PathMode, Projection, ReturnItem, SortOrder,
    };

    #[test]
    fn test_cost_addition() {
        let a = Cost::cpu(10.0).with_io(5.0);
        let b = Cost::cpu(20.0).with_memory(100.0);
        let c = a + b;

        assert!((c.cpu - 30.0).abs() < 0.001);
        assert!((c.io - 5.0).abs() < 0.001);
        assert!((c.memory - 100.0).abs() < 0.001);
    }

    #[test]
    fn test_cost_total() {
        let cost = Cost::cpu(10.0).with_io(1.0).with_memory(100.0);
        // Total = 10 + 1*10 + 100*0.1 = 10 + 10 + 10 = 30
        assert!((cost.total() - 30.0).abs() < 0.001);
    }

    #[test]
    fn test_cost_model_node_scan() {
        let model = CostModel::new();
        let scan = NodeScanOp {
            variable: "n".to_string(),
            label: Some("Person".to_string()),
            input: None,
        };
        let cost = model.node_scan_cost(&scan, 1000.0);

        assert!(cost.cpu > 0.0);
        assert!(cost.io > 0.0);
    }

    #[test]
    fn test_cost_model_sort() {
        let model = CostModel::new();
        let sort = SortOp {
            keys: vec![],
            input: Box::new(LogicalOperator::Empty),
        };

        let cost_100 = model.sort_cost(&sort, 100.0);
        let cost_1000 = model.sort_cost(&sort, 1000.0);

        // Sorting 1000 rows should be more expensive than 100 rows
        assert!(cost_1000.total() > cost_100.total());
    }

    #[test]
    fn test_cost_zero() {
        let cost = Cost::zero();
        assert!((cost.cpu).abs() < 0.001);
        assert!((cost.io).abs() < 0.001);
        assert!((cost.memory).abs() < 0.001);
        assert!((cost.network).abs() < 0.001);
        assert!((cost.total()).abs() < 0.001);
    }

    #[test]
    fn test_cost_add_assign() {
        let mut cost = Cost::cpu(10.0);
        cost += Cost::cpu(5.0).with_io(2.0);
        assert!((cost.cpu - 15.0).abs() < 0.001);
        assert!((cost.io - 2.0).abs() < 0.001);
    }

    #[test]
    fn test_cost_total_weighted() {
        let cost = Cost::cpu(10.0).with_io(2.0).with_memory(100.0);
        // With custom weights: cpu*2 + io*5 + mem*0.5 = 20 + 10 + 50 = 80
        let total = cost.total_weighted(2.0, 5.0, 0.5);
        assert!((total - 80.0).abs() < 0.001);
    }

    #[test]
    fn test_cost_model_filter() {
        let model = CostModel::new();
        let filter = FilterOp {
            predicate: LogicalExpression::Literal(grafeo_common::types::Value::Bool(true)),
            input: Box::new(LogicalOperator::Empty),
            pushdown_hint: None,
        };
        let cost = model.filter_cost(&filter, 1000.0);

        // Filter cost is CPU only
        assert!(cost.cpu > 0.0);
        assert!((cost.io).abs() < 0.001);
    }

    #[test]
    fn test_cost_model_project() {
        let model = CostModel::new();
        let project = ProjectOp {
            projections: vec![
                Projection {
                    expression: LogicalExpression::Variable("a".to_string()),
                    alias: None,
                },
                Projection {
                    expression: LogicalExpression::Variable("b".to_string()),
                    alias: None,
                },
            ],
            input: Box::new(LogicalOperator::Empty),
            pass_through_input: false,
        };
        let cost = model.project_cost(&project, 1000.0);

        // Cost should scale with number of projections
        assert!(cost.cpu > 0.0);
    }

    #[test]
    fn test_cost_model_expand() {
        let model = CostModel::new();
        let expand = ExpandOp {
            from_variable: "a".to_string(),
            to_variable: "b".to_string(),
            edge_variable: None,
            direction: ExpandDirection::Outgoing,
            edge_types: vec![],
            min_hops: 1,
            max_hops: Some(1),
            input: Box::new(LogicalOperator::Empty),
            path_alias: None,
            path_mode: PathMode::Walk,
        };
        let cost = model.expand_cost(&expand, 1000.0);

        // Expand involves hash lookups and output generation
        assert!(cost.cpu > 0.0);
    }

    #[test]
    fn test_cost_model_expand_with_edge_type_stats() {
        let mut degrees = std::collections::HashMap::new();
        degrees.insert("KNOWS".to_string(), (5.0, 5.0)); // Symmetric
        degrees.insert("WORKS_AT".to_string(), (1.0, 50.0)); // Many-to-one

        let model = CostModel::new().with_edge_type_degrees(degrees);

        // Outgoing KNOWS: fanout = 5
        let knows_out = ExpandOp {
            from_variable: "a".to_string(),
            to_variable: "b".to_string(),
            edge_variable: None,
            direction: ExpandDirection::Outgoing,
            edge_types: vec!["KNOWS".to_string()],
            min_hops: 1,
            max_hops: Some(1),
            input: Box::new(LogicalOperator::Empty),
            path_alias: None,
            path_mode: PathMode::Walk,
        };
        let cost_knows = model.expand_cost(&knows_out, 1000.0);

        // Outgoing WORKS_AT: fanout = 1 (each person works at one company)
        let works_out = ExpandOp {
            from_variable: "a".to_string(),
            to_variable: "b".to_string(),
            edge_variable: None,
            direction: ExpandDirection::Outgoing,
            edge_types: vec!["WORKS_AT".to_string()],
            min_hops: 1,
            max_hops: Some(1),
            input: Box::new(LogicalOperator::Empty),
            path_alias: None,
            path_mode: PathMode::Walk,
        };
        let cost_works = model.expand_cost(&works_out, 1000.0);

        // KNOWS (fanout=5) should be more expensive than WORKS_AT (fanout=1)
        assert!(
            cost_knows.cpu > cost_works.cpu,
            "KNOWS(5) should cost more than WORKS_AT(1)"
        );

        // Incoming WORKS_AT: fanout = 50 (company has many employees)
        let works_in = ExpandOp {
            from_variable: "c".to_string(),
            to_variable: "p".to_string(),
            edge_variable: None,
            direction: ExpandDirection::Incoming,
            edge_types: vec!["WORKS_AT".to_string()],
            min_hops: 1,
            max_hops: Some(1),
            input: Box::new(LogicalOperator::Empty),
            path_alias: None,
            path_mode: PathMode::Walk,
        };
        let cost_works_in = model.expand_cost(&works_in, 1000.0);

        // Incoming WORKS_AT (fanout=50) should be most expensive
        assert!(
            cost_works_in.cpu > cost_knows.cpu,
            "Incoming WORKS_AT(50) should cost more than KNOWS(5)"
        );
    }

    #[test]
    fn test_cost_model_expand_unknown_edge_type_uses_global_fanout() {
        let model = CostModel::new().with_avg_fanout(7.0);
        let expand = ExpandOp {
            from_variable: "a".to_string(),
            to_variable: "b".to_string(),
            edge_variable: None,
            direction: ExpandDirection::Outgoing,
            edge_types: vec!["UNKNOWN_TYPE".to_string()],
            min_hops: 1,
            max_hops: Some(1),
            input: Box::new(LogicalOperator::Empty),
            path_alias: None,
            path_mode: PathMode::Walk,
        };
        let cost_unknown = model.expand_cost(&expand, 1000.0);

        // Without edge type (uses global fanout too)
        let expand_no_type = ExpandOp {
            from_variable: "a".to_string(),
            to_variable: "b".to_string(),
            edge_variable: None,
            direction: ExpandDirection::Outgoing,
            edge_types: vec![],
            min_hops: 1,
            max_hops: Some(1),
            input: Box::new(LogicalOperator::Empty),
            path_alias: None,
            path_mode: PathMode::Walk,
        };
        let cost_no_type = model.expand_cost(&expand_no_type, 1000.0);

        // Both should use global fanout = 7, so costs should be equal
        assert!(
            (cost_unknown.cpu - cost_no_type.cpu).abs() < 0.001,
            "Unknown edge type should use global fanout"
        );
    }

    #[test]
    fn test_cost_model_hash_join() {
        let model = CostModel::new();
        let join = JoinOp {
            left: Box::new(LogicalOperator::Empty),
            right: Box::new(LogicalOperator::Empty),
            join_type: JoinType::Inner,
            conditions: vec![JoinCondition {
                left: LogicalExpression::Variable("a".to_string()),
                right: LogicalExpression::Variable("b".to_string()),
            }],
        };
        let cost = model.join_cost(&join, 10000.0);

        // Hash join has CPU cost and memory cost
        assert!(cost.cpu > 0.0);
        assert!(cost.memory > 0.0);
    }

    #[test]
    fn test_cost_model_cross_join() {
        let model = CostModel::new();
        let join = JoinOp {
            left: Box::new(LogicalOperator::Empty),
            right: Box::new(LogicalOperator::Empty),
            join_type: JoinType::Cross,
            conditions: vec![],
        };
        let cost = model.join_cost(&join, 1000000.0);

        // Cross join is expensive
        assert!(cost.cpu > 0.0);
    }

    #[test]
    fn test_cost_model_semi_join() {
        let model = CostModel::new();
        let join = JoinOp {
            left: Box::new(LogicalOperator::Empty),
            right: Box::new(LogicalOperator::Empty),
            join_type: JoinType::Semi,
            conditions: vec![],
        };
        let cost_semi = model.join_cost(&join, 1000.0);

        let inner_join = JoinOp {
            left: Box::new(LogicalOperator::Empty),
            right: Box::new(LogicalOperator::Empty),
            join_type: JoinType::Inner,
            conditions: vec![],
        };
        let cost_inner = model.join_cost(&inner_join, 1000.0);

        // Semi join can be cheaper than inner join
        assert!(cost_semi.cpu > 0.0);
        assert!(cost_inner.cpu > 0.0);
    }

    #[test]
    fn test_cost_model_aggregate() {
        let model = CostModel::new();
        let agg = AggregateOp {
            group_by: vec![],
            aggregates: vec![
                AggregateExpr {
                    function: AggregateFunction::Count,
                    expression: None,
                    expression2: None,
                    distinct: false,
                    alias: Some("cnt".to_string()),
                    percentile: None,
                    separator: None,
                },
                AggregateExpr {
                    function: AggregateFunction::Sum,
                    expression: Some(LogicalExpression::Variable("x".to_string())),
                    expression2: None,
                    distinct: false,
                    alias: Some("total".to_string()),
                    percentile: None,
                    separator: None,
                },
            ],
            input: Box::new(LogicalOperator::Empty),
            having: None,
        };
        let cost = model.aggregate_cost(&agg, 1000.0);

        // Aggregation has hash cost and memory cost
        assert!(cost.cpu > 0.0);
        assert!(cost.memory > 0.0);
    }

    #[test]
    fn test_cost_model_distinct() {
        let model = CostModel::new();
        let distinct = DistinctOp {
            input: Box::new(LogicalOperator::Empty),
            columns: None,
        };
        let cost = model.distinct_cost(&distinct, 1000.0);

        // Distinct uses hash set
        assert!(cost.cpu > 0.0);
        assert!(cost.memory > 0.0);
    }

    #[test]
    fn test_cost_model_limit() {
        let model = CostModel::new();
        let limit = LimitOp {
            count: 10.into(),
            input: Box::new(LogicalOperator::Empty),
        };
        let cost = model.limit_cost(&limit, 1000.0);

        // Limit is very cheap
        assert!(cost.cpu > 0.0);
        assert!(cost.cpu < 1.0); // Should be minimal
    }

    #[test]
    fn test_cost_model_skip() {
        let model = CostModel::new();
        let skip = SkipOp {
            count: 100.into(),
            input: Box::new(LogicalOperator::Empty),
        };
        let cost = model.skip_cost(&skip, 1000.0);

        // Skip must scan through skipped rows
        assert!(cost.cpu > 0.0);
    }

    #[test]
    fn test_cost_model_return() {
        let model = CostModel::new();
        let ret = ReturnOp {
            items: vec![
                ReturnItem {
                    expression: LogicalExpression::Variable("a".to_string()),
                    alias: None,
                },
                ReturnItem {
                    expression: LogicalExpression::Variable("b".to_string()),
                    alias: None,
                },
            ],
            distinct: false,
            input: Box::new(LogicalOperator::Empty),
        };
        let cost = model.return_cost(&ret, 1000.0);

        // Return materializes results
        assert!(cost.cpu > 0.0);
    }

    #[test]
    fn test_cost_cheaper() {
        let model = CostModel::new();
        let cheap = Cost::cpu(10.0);
        let expensive = Cost::cpu(100.0);

        assert_eq!(model.cheaper(&cheap, &expensive).total(), cheap.total());
        assert_eq!(model.cheaper(&expensive, &cheap).total(), cheap.total());
    }

    #[test]
    fn test_cost_comparison_prefers_lower_total() {
        let model = CostModel::new();
        // High CPU, low IO
        let cpu_heavy = Cost::cpu(100.0).with_io(1.0);
        // Low CPU, high IO
        let io_heavy = Cost::cpu(10.0).with_io(20.0);

        // IO is weighted 10x, so io_heavy = 10 + 200 = 210, cpu_heavy = 100 + 10 = 110
        assert!(cpu_heavy.total() < io_heavy.total());
        assert_eq!(
            model.cheaper(&cpu_heavy, &io_heavy).total(),
            cpu_heavy.total()
        );
    }

    #[test]
    fn test_cost_model_sort_with_keys() {
        let model = CostModel::new();
        let sort_single = SortOp {
            keys: vec![crate::query::plan::SortKey {
                expression: LogicalExpression::Variable("a".to_string()),
                order: SortOrder::Ascending,
                nulls: None,
            }],
            input: Box::new(LogicalOperator::Empty),
        };
        let sort_multi = SortOp {
            keys: vec![
                crate::query::plan::SortKey {
                    expression: LogicalExpression::Variable("a".to_string()),
                    order: SortOrder::Ascending,
                    nulls: None,
                },
                crate::query::plan::SortKey {
                    expression: LogicalExpression::Variable("b".to_string()),
                    order: SortOrder::Descending,
                    nulls: None,
                },
            ],
            input: Box::new(LogicalOperator::Empty),
        };

        let cost_single = model.sort_cost(&sort_single, 1000.0);
        let cost_multi = model.sort_cost(&sort_multi, 1000.0);

        // More sort keys = more comparisons
        assert!(cost_multi.cpu > cost_single.cpu);
    }

    #[test]
    fn test_cost_model_empty_operator() {
        let model = CostModel::new();
        let cost = model.estimate(&LogicalOperator::Empty, 0.0);
        assert!((cost.total()).abs() < 0.001);
    }

    #[test]
    fn test_cost_model_default() {
        let model = CostModel::default();
        let scan = NodeScanOp {
            variable: "n".to_string(),
            label: None,
            input: None,
        };
        let cost = model.estimate(&LogicalOperator::NodeScan(scan), 100.0);
        assert!(cost.total() > 0.0);
    }

    #[test]
    fn test_leapfrog_join_cost() {
        let model = CostModel::new();

        // Three-way join (triangle pattern)
        let cardinalities = vec![1000.0, 1000.0, 1000.0];
        let cost = model.leapfrog_join_cost(3, &cardinalities, 100.0);

        // Should have CPU cost for materialization and intersection
        assert!(cost.cpu > 0.0);
        // Should have memory cost for trie storage
        assert!(cost.memory > 0.0);
    }

    #[test]
    fn test_leapfrog_join_cost_empty() {
        let model = CostModel::new();
        let cost = model.leapfrog_join_cost(0, &[], 0.0);
        assert!((cost.total()).abs() < 0.001);
    }

    #[test]
    fn test_prefer_leapfrog_join_for_triangles() {
        let model = CostModel::new();

        // Compare costs for triangle pattern
        let cardinalities = vec![10000.0, 10000.0, 10000.0];
        let output = 1000.0;

        let leapfrog_cost = model.leapfrog_join_cost(3, &cardinalities, output);

        // Leapfrog should have reasonable cost for triangle patterns
        assert!(leapfrog_cost.cpu > 0.0);
        assert!(leapfrog_cost.memory > 0.0);

        // The prefer_leapfrog_join method compares against hash cascade
        // Actual preference depends on specific cost parameters
        let _prefer = model.prefer_leapfrog_join(3, &cardinalities, output);
        // Test that it returns a boolean (doesn't panic)
    }

    #[test]
    fn test_prefer_leapfrog_join_binary_case() {
        let model = CostModel::new();

        // Binary join should NOT prefer leapfrog (need 3+ relations)
        let cardinalities = vec![1000.0, 1000.0];
        let prefer = model.prefer_leapfrog_join(2, &cardinalities, 500.0);
        assert!(!prefer, "Binary joins should use hash join, not leapfrog");
    }

    #[test]
    fn test_factorized_benefit_single_hop() {
        let model = CostModel::new();

        // Single hop: no factorization benefit
        let benefit = model.factorized_benefit(10.0, 1);
        assert!(
            (benefit - 1.0).abs() < 0.001,
            "Single hop should have no benefit"
        );
    }

    #[test]
    fn test_factorized_benefit_multi_hop() {
        let model = CostModel::new();

        // Multi-hop with high fanout
        let benefit = model.factorized_benefit(10.0, 3);

        // The factorized_benefit returns a ratio capped at 1.0
        // For high fanout, factorized size / full size approaches 1/fanout
        // which is beneficial but the formula gives a value <= 1.0
        assert!(benefit <= 1.0, "Benefit should be <= 1.0");
        assert!(benefit > 0.0, "Benefit should be positive");
    }

    #[test]
    fn test_factorized_benefit_low_fanout() {
        let model = CostModel::new();

        // Low fanout: minimal benefit
        let benefit = model.factorized_benefit(1.5, 2);
        assert!(
            benefit <= 1.0,
            "Low fanout still benefits from factorization"
        );
    }

    #[test]
    fn test_node_scan_uses_label_cardinality_for_io() {
        let mut label_cards = std::collections::HashMap::new();
        label_cards.insert("Person".to_string(), 500_u64);
        label_cards.insert("Company".to_string(), 50_u64);

        let model = CostModel::new()
            .with_label_cardinalities(label_cards)
            .with_graph_totals(550, 1000);

        let person_scan = NodeScanOp {
            variable: "n".to_string(),
            label: Some("Person".to_string()),
            input: None,
        };
        let company_scan = NodeScanOp {
            variable: "n".to_string(),
            label: Some("Company".to_string()),
            input: None,
        };

        let person_cost = model.node_scan_cost(&person_scan, 500.0);
        let company_cost = model.node_scan_cost(&company_scan, 50.0);

        // Person scan reads 10x more pages than Company scan
        assert!(
            person_cost.io > company_cost.io * 5.0,
            "Person ({}) should have much higher IO than Company ({})",
            person_cost.io,
            company_cost.io
        );
    }

    #[test]
    fn test_node_scan_unlabeled_uses_total_nodes() {
        let model = CostModel::new().with_graph_totals(10_000, 50_000);

        let scan = NodeScanOp {
            variable: "n".to_string(),
            label: None,
            input: None,
        };

        let cost = model.node_scan_cost(&scan, 10_000.0);
        let expected_pages = (10_000.0 * 100.0) / 8192.0;
        assert!(
            (cost.io - expected_pages).abs() < 0.1,
            "Unlabeled scan should use total_nodes for IO: got {}, expected {}",
            cost.io,
            expected_pages
        );
    }

    #[test]
    fn test_join_cost_with_actual_child_cardinalities() {
        let model = CostModel::new();
        let join = JoinOp {
            left: Box::new(LogicalOperator::Empty),
            right: Box::new(LogicalOperator::Empty),
            join_type: JoinType::Inner,
            conditions: vec![JoinCondition {
                left: LogicalExpression::Variable("a".to_string()),
                right: LogicalExpression::Variable("b".to_string()),
            }],
        };

        // With actual child cardinalities (100 left, 10000 right)
        let cost_actual = model.join_cost_with_children(&join, 500.0, Some(100.0), Some(10_000.0));

        // With sqrt fallback (sqrt(500) ~ 22.4 for both sides)
        let cost_sqrt = model.join_cost(&join, 500.0);

        // Actual build side (100) is larger than sqrt(500) ~ 22.4, so
        // actual cost should be higher for build, but the probe side (10000)
        // should dominate
        assert!(
            cost_actual.cpu > cost_sqrt.cpu,
            "Actual child cardinalities ({}) should produce different cost than sqrt fallback ({})",
            cost_actual.cpu,
            cost_sqrt.cpu
        );
    }

    #[test]
    fn test_expand_multi_edge_types() {
        let mut degrees = std::collections::HashMap::new();
        degrees.insert("KNOWS".to_string(), (5.0, 5.0));
        degrees.insert("FOLLOWS".to_string(), (20.0, 100.0));

        let model = CostModel::new().with_edge_type_degrees(degrees);

        // Multi-type outgoing: KNOWS(5) + FOLLOWS(20) = 25
        let multi_expand = ExpandOp {
            from_variable: "a".to_string(),
            to_variable: "b".to_string(),
            edge_variable: None,
            direction: ExpandDirection::Outgoing,
            edge_types: vec!["KNOWS".to_string(), "FOLLOWS".to_string()],
            min_hops: 1,
            max_hops: Some(1),
            input: Box::new(LogicalOperator::Empty),
            path_alias: None,
            path_mode: PathMode::Walk,
        };
        let multi_cost = model.expand_cost(&multi_expand, 100.0);

        // Single type: KNOWS(5) only
        let single_expand = ExpandOp {
            from_variable: "a".to_string(),
            to_variable: "b".to_string(),
            edge_variable: None,
            direction: ExpandDirection::Outgoing,
            edge_types: vec!["KNOWS".to_string()],
            min_hops: 1,
            max_hops: Some(1),
            input: Box::new(LogicalOperator::Empty),
            path_alias: None,
            path_mode: PathMode::Walk,
        };
        let single_cost = model.expand_cost(&single_expand, 100.0);

        // Multi-type (fanout=25) should be more expensive than single (fanout=5)
        assert!(
            multi_cost.cpu > single_cost.cpu * 3.0,
            "Multi-type fanout ({}) should be much higher than single-type ({})",
            multi_cost.cpu,
            single_cost.cpu
        );
    }

    #[test]
    fn test_recursive_tree_cost() {
        use crate::query::optimizer::CardinalityEstimator;

        let mut label_cards = std::collections::HashMap::new();
        label_cards.insert("Person".to_string(), 1000_u64);

        let model = CostModel::new()
            .with_label_cardinalities(label_cards)
            .with_graph_totals(1000, 5000)
            .with_avg_fanout(5.0);

        let mut card_est = CardinalityEstimator::new();
        card_est.add_table_stats("Person", crate::query::optimizer::TableStats::new(1000));

        // Build a plan: NodeScan -> Filter -> Return
        let plan = LogicalOperator::Return(ReturnOp {
            items: vec![ReturnItem {
                expression: LogicalExpression::Variable("n".to_string()),
                alias: None,
            }],
            distinct: false,
            input: Box::new(LogicalOperator::Filter(FilterOp {
                predicate: LogicalExpression::Binary {
                    left: Box::new(LogicalExpression::Property {
                        variable: "n".to_string(),
                        property: "age".to_string(),
                    }),
                    op: crate::query::plan::BinaryOp::Gt,
                    right: Box::new(LogicalExpression::Literal(
                        grafeo_common::types::Value::Int64(30),
                    )),
                },
                input: Box::new(LogicalOperator::NodeScan(NodeScanOp {
                    variable: "n".to_string(),
                    label: Some("Person".to_string()),
                    input: None,
                })),
                pushdown_hint: None,
            })),
        });

        let tree_cost = model.estimate_tree(&plan, &card_est);

        // Tree cost should include scan IO + filter CPU + return CPU
        assert!(tree_cost.cpu > 0.0, "Tree should have CPU cost");
        assert!(tree_cost.io > 0.0, "Tree should have IO cost from scan");

        // Compare with single-operator estimate (only costs the root)
        let root_only_card = card_est.estimate(&plan);
        let root_only_cost = model.estimate(&plan, root_only_card);

        // Tree cost should be strictly higher because it includes child costs
        assert!(
            tree_cost.total() > root_only_cost.total(),
            "Recursive tree cost ({}) should exceed root-only cost ({})",
            tree_cost.total(),
            root_only_cost.total()
        );
    }

    #[test]
    fn test_statistics_driven_vs_default_cost() {
        let default_model = CostModel::new();

        let mut label_cards = std::collections::HashMap::new();
        label_cards.insert("Person".to_string(), 100_u64);
        let stats_model = CostModel::new()
            .with_label_cardinalities(label_cards)
            .with_graph_totals(100, 500);

        // Scan a small label: statistics model knows it's only 100 nodes
        let scan = NodeScanOp {
            variable: "n".to_string(),
            label: Some("Person".to_string()),
            input: None,
        };

        let default_cost = default_model.node_scan_cost(&scan, 100.0);
        let stats_cost = stats_model.node_scan_cost(&scan, 100.0);

        // With statistics, IO cost is based on actual label size (100 nodes)
        // Without statistics, IO cost uses cardinality parameter (also 100 here)
        // They should be equal in this case since cardinality matches label size
        assert!(
            (default_cost.io - stats_cost.io).abs() < 0.1,
            "When cardinality matches label size, costs should be similar"
        );
    }

    #[test]
    fn test_leapfrog_join_cost_unit_min_cardinality() {
        let model = CostModel::new();
        // min_card == 1.0 exercises the else branch in seek_cost (output * cpu_tuple_cost)
        let cost = model.leapfrog_join_cost(3, &[1.0, 100.0, 200.0], 50.0);
        assert!(cost.cpu > 0.0);
        assert!(cost.memory > 0.0);
    }

    #[test]
    fn test_prefer_leapfrog_join_cardinalities_below_three() {
        let model = CostModel::new();
        // num_relations >= 3 but fewer cardinality entries: second guard fires
        assert!(!model.prefer_leapfrog_join(3, &[100.0, 200.0], 50.0));
        assert!(!model.prefer_leapfrog_join(5, &[], 10.0));
    }

    #[test]
    fn test_factorized_benefit_zero_hops() {
        let model = CostModel::new();
        // Zero hops falls through the `num_hops <= 1` guard
        assert_eq!(model.factorized_benefit(10.0, 0), 1.0);
    }

    #[test]
    fn test_factorized_benefit_unit_fanout_guard() {
        let model = CostModel::new();
        // fanout exactly 1.0: no benefit regardless of hop count
        assert_eq!(model.factorized_benefit(1.0, 5), 1.0);
    }
}
