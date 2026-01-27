//! Cardinality estimation for query optimization.
//!
//! Estimates the number of rows produced by each operator in a query plan.

use crate::query::plan::{
    AggregateOp, BinaryOp, DistinctOp, ExpandOp, FilterOp, JoinOp, JoinType, LimitOp,
    LogicalExpression, LogicalOperator, NodeScanOp, ProjectOp, SkipOp, SortOp, UnaryOp,
};
use std::collections::HashMap;

/// Statistics for a table/label.
#[derive(Debug, Clone)]
pub struct TableStats {
    /// Total number of rows.
    pub row_count: u64,
    /// Column statistics.
    pub columns: HashMap<String, ColumnStats>,
}

impl TableStats {
    /// Creates new table statistics.
    #[must_use]
    pub fn new(row_count: u64) -> Self {
        Self {
            row_count,
            columns: HashMap::new(),
        }
    }

    /// Adds column statistics.
    pub fn with_column(mut self, name: &str, stats: ColumnStats) -> Self {
        self.columns.insert(name.to_string(), stats);
        self
    }
}

/// Statistics for a column.
#[derive(Debug, Clone)]
pub struct ColumnStats {
    /// Number of distinct values.
    pub distinct_count: u64,
    /// Number of null values.
    pub null_count: u64,
    /// Minimum value (if orderable).
    pub min_value: Option<f64>,
    /// Maximum value (if orderable).
    pub max_value: Option<f64>,
}

impl ColumnStats {
    /// Creates new column statistics.
    #[must_use]
    pub fn new(distinct_count: u64) -> Self {
        Self {
            distinct_count,
            null_count: 0,
            min_value: None,
            max_value: None,
        }
    }

    /// Sets the null count.
    #[must_use]
    pub fn with_nulls(mut self, null_count: u64) -> Self {
        self.null_count = null_count;
        self
    }

    /// Sets the min/max range.
    #[must_use]
    pub fn with_range(mut self, min: f64, max: f64) -> Self {
        self.min_value = Some(min);
        self.max_value = Some(max);
        self
    }
}

/// Cardinality estimator.
pub struct CardinalityEstimator {
    /// Statistics for each label/table.
    table_stats: HashMap<String, TableStats>,
    /// Default row count for unknown tables.
    default_row_count: u64,
    /// Default selectivity for unknown predicates.
    default_selectivity: f64,
    /// Average edge fanout (outgoing edges per node).
    avg_fanout: f64,
}

impl CardinalityEstimator {
    /// Creates a new cardinality estimator with default settings.
    #[must_use]
    pub fn new() -> Self {
        Self {
            table_stats: HashMap::new(),
            default_row_count: 1000,
            default_selectivity: 0.1,
            avg_fanout: 10.0,
        }
    }

    /// Adds statistics for a table/label.
    pub fn add_table_stats(&mut self, name: &str, stats: TableStats) {
        self.table_stats.insert(name.to_string(), stats);
    }

    /// Sets the average edge fanout.
    pub fn set_avg_fanout(&mut self, fanout: f64) {
        self.avg_fanout = fanout;
    }

    /// Estimates the cardinality of a logical operator.
    #[must_use]
    pub fn estimate(&self, op: &LogicalOperator) -> f64 {
        match op {
            LogicalOperator::NodeScan(scan) => self.estimate_node_scan(scan),
            LogicalOperator::Filter(filter) => self.estimate_filter(filter),
            LogicalOperator::Project(project) => self.estimate_project(project),
            LogicalOperator::Expand(expand) => self.estimate_expand(expand),
            LogicalOperator::Join(join) => self.estimate_join(join),
            LogicalOperator::Aggregate(agg) => self.estimate_aggregate(agg),
            LogicalOperator::Sort(sort) => self.estimate_sort(sort),
            LogicalOperator::Distinct(distinct) => self.estimate_distinct(distinct),
            LogicalOperator::Limit(limit) => self.estimate_limit(limit),
            LogicalOperator::Skip(skip) => self.estimate_skip(skip),
            LogicalOperator::Return(ret) => self.estimate(&ret.input),
            LogicalOperator::Empty => 0.0,
            _ => self.default_row_count as f64,
        }
    }

    /// Estimates node scan cardinality.
    fn estimate_node_scan(&self, scan: &NodeScanOp) -> f64 {
        if let Some(label) = &scan.label {
            if let Some(stats) = self.table_stats.get(label) {
                return stats.row_count as f64;
            }
        }
        // No label filter - scan all nodes
        self.default_row_count as f64
    }

    /// Estimates filter cardinality.
    fn estimate_filter(&self, filter: &FilterOp) -> f64 {
        let input_cardinality = self.estimate(&filter.input);
        let selectivity = self.estimate_selectivity(&filter.predicate);
        (input_cardinality * selectivity).max(1.0)
    }

    /// Estimates projection cardinality (same as input).
    fn estimate_project(&self, project: &ProjectOp) -> f64 {
        self.estimate(&project.input)
    }

    /// Estimates expand cardinality.
    fn estimate_expand(&self, expand: &ExpandOp) -> f64 {
        let input_cardinality = self.estimate(&expand.input);

        // Apply fanout based on edge type
        let fanout = if expand.edge_type.is_some() {
            // Specific edge type typically has lower fanout
            self.avg_fanout * 0.5
        } else {
            self.avg_fanout
        };

        // Handle variable-length paths
        let path_multiplier = if expand.max_hops.unwrap_or(1) > 1 {
            let min = expand.min_hops as f64;
            let max = expand.max_hops.unwrap_or(expand.min_hops + 3) as f64;
            // Geometric series approximation
            (fanout.powf(max + 1.0) - fanout.powf(min)) / (fanout - 1.0)
        } else {
            fanout
        };

        (input_cardinality * path_multiplier).max(1.0)
    }

    /// Estimates join cardinality.
    fn estimate_join(&self, join: &JoinOp) -> f64 {
        let left_card = self.estimate(&join.left);
        let right_card = self.estimate(&join.right);

        match join.join_type {
            JoinType::Cross => left_card * right_card,
            JoinType::Inner => {
                // Assume join selectivity based on conditions
                let selectivity = if join.conditions.is_empty() {
                    1.0 // Cross join
                } else {
                    // Estimate based on number of conditions
                    0.1_f64.powi(join.conditions.len() as i32)
                };
                (left_card * right_card * selectivity).max(1.0)
            }
            JoinType::Left => {
                // Left join returns at least all left rows
                let inner_card = self.estimate_join(&JoinOp {
                    left: join.left.clone(),
                    right: join.right.clone(),
                    join_type: JoinType::Inner,
                    conditions: join.conditions.clone(),
                });
                inner_card.max(left_card)
            }
            JoinType::Right => {
                // Right join returns at least all right rows
                let inner_card = self.estimate_join(&JoinOp {
                    left: join.left.clone(),
                    right: join.right.clone(),
                    join_type: JoinType::Inner,
                    conditions: join.conditions.clone(),
                });
                inner_card.max(right_card)
            }
            JoinType::Full => {
                // Full join returns at least max(left, right)
                let inner_card = self.estimate_join(&JoinOp {
                    left: join.left.clone(),
                    right: join.right.clone(),
                    join_type: JoinType::Inner,
                    conditions: join.conditions.clone(),
                });
                inner_card.max(left_card.max(right_card))
            }
            JoinType::Semi => {
                // Semi join returns at most left cardinality
                (left_card * self.default_selectivity).max(1.0)
            }
            JoinType::Anti => {
                // Anti join returns at most left cardinality
                (left_card * (1.0 - self.default_selectivity)).max(1.0)
            }
        }
    }

    /// Estimates aggregation cardinality.
    fn estimate_aggregate(&self, agg: &AggregateOp) -> f64 {
        let input_cardinality = self.estimate(&agg.input);

        if agg.group_by.is_empty() {
            // Global aggregation - single row
            1.0
        } else {
            // Group by - estimate distinct groups
            // Assume each group key reduces cardinality by 10
            let group_reduction = 10.0_f64.powi(agg.group_by.len() as i32);
            (input_cardinality / group_reduction).max(1.0)
        }
    }

    /// Estimates sort cardinality (same as input).
    fn estimate_sort(&self, sort: &SortOp) -> f64 {
        self.estimate(&sort.input)
    }

    /// Estimates distinct cardinality.
    fn estimate_distinct(&self, distinct: &DistinctOp) -> f64 {
        let input_cardinality = self.estimate(&distinct.input);
        // Assume 50% distinct by default
        (input_cardinality * 0.5).max(1.0)
    }

    /// Estimates limit cardinality.
    fn estimate_limit(&self, limit: &LimitOp) -> f64 {
        let input_cardinality = self.estimate(&limit.input);
        (limit.count as f64).min(input_cardinality)
    }

    /// Estimates skip cardinality.
    fn estimate_skip(&self, skip: &SkipOp) -> f64 {
        let input_cardinality = self.estimate(&skip.input);
        (input_cardinality - skip.count as f64).max(0.0)
    }

    /// Estimates the selectivity of a predicate (0.0 to 1.0).
    fn estimate_selectivity(&self, expr: &LogicalExpression) -> f64 {
        match expr {
            LogicalExpression::Binary { left, op, right } => {
                self.estimate_binary_selectivity(left, *op, right)
            }
            LogicalExpression::Unary { op, operand } => {
                self.estimate_unary_selectivity(*op, operand)
            }
            LogicalExpression::Literal(value) => {
                // Boolean literal
                if let graphos_common::types::Value::Bool(b) = value {
                    if *b {
                        1.0
                    } else {
                        0.0
                    }
                } else {
                    self.default_selectivity
                }
            }
            _ => self.default_selectivity,
        }
    }

    /// Estimates binary expression selectivity.
    fn estimate_binary_selectivity(
        &self,
        _left: &LogicalExpression,
        op: BinaryOp,
        _right: &LogicalExpression,
    ) -> f64 {
        match op {
            // Equality is typically very selective
            BinaryOp::Eq => 0.01,
            // Inequality is very unselective
            BinaryOp::Ne => 0.99,
            // Range predicates
            BinaryOp::Lt | BinaryOp::Le | BinaryOp::Gt | BinaryOp::Ge => 0.33,
            // Logical operators
            BinaryOp::And => {
                // AND reduces selectivity (multiply)
                self.default_selectivity * self.default_selectivity
            }
            BinaryOp::Or => {
                // OR increases selectivity (1 - (1-s1)(1-s2))
                1.0 - (1.0 - self.default_selectivity) * (1.0 - self.default_selectivity)
            }
            // String operations
            BinaryOp::StartsWith => 0.1,
            BinaryOp::EndsWith => 0.1,
            BinaryOp::Contains => 0.1,
            BinaryOp::Like => 0.1,
            // Collection membership
            BinaryOp::In => 0.1,
            // Other operations
            _ => self.default_selectivity,
        }
    }

    /// Estimates unary expression selectivity.
    fn estimate_unary_selectivity(&self, op: UnaryOp, _operand: &LogicalExpression) -> f64 {
        match op {
            UnaryOp::Not => 1.0 - self.default_selectivity,
            UnaryOp::IsNull => 0.05, // Assume 5% nulls
            UnaryOp::IsNotNull => 0.95,
            UnaryOp::Neg => 1.0, // Negation doesn't change cardinality
        }
    }

    /// Gets statistics for a column.
    fn get_column_stats(&self, label: &str, column: &str) -> Option<&ColumnStats> {
        self.table_stats.get(label)?.columns.get(column)
    }

    /// Estimates equality selectivity using column statistics.
    #[allow(dead_code)]
    fn estimate_equality_with_stats(&self, label: &str, column: &str) -> f64 {
        if let Some(stats) = self.get_column_stats(label, column) {
            if stats.distinct_count > 0 {
                return 1.0 / stats.distinct_count as f64;
            }
        }
        0.01 // Default for equality
    }

    /// Estimates range selectivity using column statistics.
    #[allow(dead_code)]
    fn estimate_range_with_stats(
        &self,
        label: &str,
        column: &str,
        lower: Option<f64>,
        upper: Option<f64>,
    ) -> f64 {
        if let Some(stats) = self.get_column_stats(label, column) {
            if let (Some(min), Some(max)) = (stats.min_value, stats.max_value) {
                let range = max - min;
                if range <= 0.0 {
                    return 1.0;
                }

                let effective_lower = lower.unwrap_or(min).max(min);
                let effective_upper = upper.unwrap_or(max).min(max);

                let overlap = (effective_upper - effective_lower).max(0.0);
                return (overlap / range).min(1.0).max(0.0);
            }
        }
        0.33 // Default for range
    }
}

impl Default for CardinalityEstimator {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::query::plan::{FilterOp, JoinCondition, NodeScanOp};
    use graphos_common::types::Value;

    #[test]
    fn test_node_scan_with_stats() {
        let mut estimator = CardinalityEstimator::new();
        estimator.add_table_stats("Person", TableStats::new(5000));

        let scan = LogicalOperator::NodeScan(NodeScanOp {
            variable: "n".to_string(),
            label: Some("Person".to_string()),
            input: None,
        });

        let cardinality = estimator.estimate(&scan);
        assert!((cardinality - 5000.0).abs() < 0.001);
    }

    #[test]
    fn test_filter_reduces_cardinality() {
        let mut estimator = CardinalityEstimator::new();
        estimator.add_table_stats("Person", TableStats::new(1000));

        let filter = LogicalOperator::Filter(FilterOp {
            predicate: LogicalExpression::Binary {
                left: Box::new(LogicalExpression::Property {
                    variable: "n".to_string(),
                    property: "age".to_string(),
                }),
                op: BinaryOp::Eq,
                right: Box::new(LogicalExpression::Literal(Value::Int64(30))),
            },
            input: Box::new(LogicalOperator::NodeScan(NodeScanOp {
                variable: "n".to_string(),
                label: Some("Person".to_string()),
                input: None,
            })),
        });

        let cardinality = estimator.estimate(&filter);
        // Equality selectivity is 0.01, so 1000 * 0.01 = 10
        assert!(cardinality < 1000.0);
        assert!(cardinality >= 1.0);
    }

    #[test]
    fn test_join_cardinality() {
        let mut estimator = CardinalityEstimator::new();
        estimator.add_table_stats("Person", TableStats::new(1000));
        estimator.add_table_stats("Company", TableStats::new(100));

        let join = LogicalOperator::Join(JoinOp {
            left: Box::new(LogicalOperator::NodeScan(NodeScanOp {
                variable: "p".to_string(),
                label: Some("Person".to_string()),
                input: None,
            })),
            right: Box::new(LogicalOperator::NodeScan(NodeScanOp {
                variable: "c".to_string(),
                label: Some("Company".to_string()),
                input: None,
            })),
            join_type: JoinType::Inner,
            conditions: vec![JoinCondition {
                left: LogicalExpression::Property {
                    variable: "p".to_string(),
                    property: "company_id".to_string(),
                },
                right: LogicalExpression::Property {
                    variable: "c".to_string(),
                    property: "id".to_string(),
                },
            }],
        });

        let cardinality = estimator.estimate(&join);
        // Should be less than cross product
        assert!(cardinality < 1000.0 * 100.0);
    }

    #[test]
    fn test_limit_caps_cardinality() {
        let mut estimator = CardinalityEstimator::new();
        estimator.add_table_stats("Person", TableStats::new(1000));

        let limit = LogicalOperator::Limit(LimitOp {
            count: 10,
            input: Box::new(LogicalOperator::NodeScan(NodeScanOp {
                variable: "n".to_string(),
                label: Some("Person".to_string()),
                input: None,
            })),
        });

        let cardinality = estimator.estimate(&limit);
        assert!((cardinality - 10.0).abs() < 0.001);
    }

    #[test]
    fn test_aggregate_reduces_cardinality() {
        let mut estimator = CardinalityEstimator::new();
        estimator.add_table_stats("Person", TableStats::new(1000));

        // Global aggregation
        let global_agg = LogicalOperator::Aggregate(AggregateOp {
            group_by: vec![],
            aggregates: vec![],
            input: Box::new(LogicalOperator::NodeScan(NodeScanOp {
                variable: "n".to_string(),
                label: Some("Person".to_string()),
                input: None,
            })),
        });

        let cardinality = estimator.estimate(&global_agg);
        assert!((cardinality - 1.0).abs() < 0.001);

        // Group by aggregation
        let group_agg = LogicalOperator::Aggregate(AggregateOp {
            group_by: vec![LogicalExpression::Property {
                variable: "n".to_string(),
                property: "city".to_string(),
            }],
            aggregates: vec![],
            input: Box::new(LogicalOperator::NodeScan(NodeScanOp {
                variable: "n".to_string(),
                label: Some("Person".to_string()),
                input: None,
            })),
        });

        let cardinality = estimator.estimate(&group_agg);
        // Should be less than input
        assert!(cardinality < 1000.0);
    }
}
