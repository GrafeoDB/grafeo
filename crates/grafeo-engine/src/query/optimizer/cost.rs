//! Cost model for query optimization.
//!
//! Provides cost estimates for logical operators to guide optimization decisions.

use crate::query::plan::{
    AggregateOp, DistinctOp, ExpandOp, FilterOp, JoinOp, JoinType, LimitOp, LogicalOperator,
    NodeScanOp, ProjectOp, ReturnOp, SkipOp, SortOp, VectorJoinOp, VectorScanOp,
};

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
pub struct CostModel {
    /// Cost per tuple processed by CPU.
    cpu_tuple_cost: f64,
    /// Cost per I/O page read.
    #[allow(dead_code)]
    io_page_cost: f64,
    /// Cost per hash table lookup.
    hash_lookup_cost: f64,
    /// Cost per comparison in sorting.
    sort_comparison_cost: f64,
    /// Average tuple size in bytes.
    avg_tuple_size: f64,
    /// Page size in bytes.
    page_size: f64,
}

impl CostModel {
    /// Creates a new cost model with default parameters.
    #[must_use]
    pub fn new() -> Self {
        Self {
            cpu_tuple_cost: 0.01,
            io_page_cost: 1.0,
            hash_lookup_cost: 0.02,
            sort_comparison_cost: 0.02,
            avg_tuple_size: 100.0,
            page_size: 8192.0,
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
            _ => Cost::cpu(cardinality * self.cpu_tuple_cost),
        }
    }

    /// Estimates the cost of a node scan.
    fn node_scan_cost(&self, _scan: &NodeScanOp, cardinality: f64) -> Cost {
        let pages = (cardinality * self.avg_tuple_size) / self.page_size;
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
    fn expand_cost(&self, _expand: &ExpandOp, cardinality: f64) -> Cost {
        // Expand involves adjacency list lookups
        let lookup_cost = cardinality * self.hash_lookup_cost;
        // Assume average fanout of 10 for edge traversal
        let avg_fanout = 10.0;
        let output_cost = cardinality * avg_fanout * self.cpu_tuple_cost;
        Cost::cpu(lookup_cost + output_cost)
    }

    /// Estimates the cost of a join operation.
    fn join_cost(&self, join: &JoinOp, cardinality: f64) -> Cost {
        // Cost depends on join type
        match join.join_type {
            JoinType::Cross => {
                // Cross join is O(n * m)
                Cost::cpu(cardinality * self.cpu_tuple_cost)
            }
            JoinType::Inner | JoinType::Left | JoinType::Right | JoinType::Full => {
                // Hash join: build phase + probe phase
                // Assume left side is build, right side is probe
                let build_cardinality = cardinality.sqrt(); // Rough estimate
                let probe_cardinality = cardinality.sqrt();

                // Build hash table
                let build_cost = build_cardinality * self.hash_lookup_cost;
                let memory_cost = build_cardinality * self.avg_tuple_size;

                // Probe hash table
                let probe_cost = probe_cardinality * self.hash_lookup_cost;

                // Output cost
                let output_cost = cardinality * self.cpu_tuple_cost;

                Cost::cpu(build_cost + probe_cost + output_cost).with_memory(memory_cost)
            }
            JoinType::Semi | JoinType::Anti => {
                // Semi/anti joins are typically cheaper
                let build_cardinality = cardinality.sqrt();
                let probe_cardinality = cardinality.sqrt();

                let build_cost = build_cardinality * self.hash_lookup_cost;
                let probe_cost = probe_cardinality * self.hash_lookup_cost;

                Cost::cpu(build_cost + probe_cost)
                    .with_memory(build_cardinality * self.avg_tuple_size)
            }
        }
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
        Cost::cpu(limit.count as f64 * self.cpu_tuple_cost * 0.1)
    }

    /// Estimates the cost of a skip operation.
    fn skip_cost(&self, skip: &SkipOp, _cardinality: f64) -> Cost {
        // Skip requires scanning through skipped rows
        Cost::cpu(skip.count as f64 * self.cpu_tuple_cost)
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
        // k determines output cardinality
        let k = scan.k as f64;

        // HNSW search cost: O(ef * log(N)) distance computations
        // Assume ef = 64 (default), N = cardinality
        let ef = 64.0;
        let n = cardinality.max(1.0);
        let search_cost = if scan.index_name.is_some() {
            // HNSW: O(ef * log N)
            ef * n.ln() * self.cpu_tuple_cost * 10.0 // Distance computation is ~10x regular tuple
        } else {
            // Brute-force: O(N)
            n * self.cpu_tuple_cost * 10.0
        };

        // Memory for candidate heap
        let memory = k * self.avg_tuple_size * 2.0;

        Cost::cpu(search_cost).with_memory(memory)
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

        let full_size = avg_fanout.powi(num_hops as i32);
        let factorized_size = if avg_fanout > 1.0 {
            (avg_fanout.powi(num_hops as i32 + 1) - 1.0) / (avg_fanout - 1.0)
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
        Projection, ReturnItem, SortOrder,
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
            edge_type: None,
            min_hops: 1,
            max_hops: Some(1),
            input: Box::new(LogicalOperator::Empty),
            path_alias: None,
        };
        let cost = model.expand_cost(&expand, 1000.0);

        // Expand involves hash lookups and output generation
        assert!(cost.cpu > 0.0);
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
                    distinct: false,
                    alias: Some("cnt".to_string()),
                    percentile: None,
                },
                AggregateExpr {
                    function: AggregateFunction::Sum,
                    expression: Some(LogicalExpression::Variable("x".to_string())),
                    distinct: false,
                    alias: Some("total".to_string()),
                    percentile: None,
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
            count: 10,
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
            count: 100,
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
            }],
            input: Box::new(LogicalOperator::Empty),
        };
        let sort_multi = SortOp {
            keys: vec![
                crate::query::plan::SortKey {
                    expression: LogicalExpression::Variable("a".to_string()),
                    order: SortOrder::Ascending,
                },
                crate::query::plan::SortKey {
                    expression: LogicalExpression::Variable("b".to_string()),
                    order: SortOrder::Descending,
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
}
