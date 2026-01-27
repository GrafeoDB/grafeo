//! Cost model for query optimization.
//!
//! Provides cost estimates for logical operators to guide optimization decisions.

use crate::query::plan::{
    AggregateOp, DistinctOp, ExpandOp, FilterOp, JoinOp, JoinType, LimitOp, LogicalOperator,
    NodeScanOp, ProjectOp, ReturnOp, SkipOp, SortOp,
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

    /// Compares two costs and returns the cheaper one.
    #[must_use]
    pub fn cheaper<'a>(&self, a: &'a Cost, b: &'a Cost) -> &'a Cost {
        if a.total() <= b.total() {
            a
        } else {
            b
        }
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
}
