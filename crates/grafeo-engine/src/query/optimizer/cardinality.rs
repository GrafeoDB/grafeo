//! Cardinality estimation for query optimization.
//!
//! Estimates the number of rows produced by each operator in a query plan.
//!
//! # Equi-Depth Histograms
//!
//! This module provides equi-depth histogram support for accurate selectivity
//! estimation of range predicates. Unlike equi-width histograms that divide
//! the value range into equal-sized buckets, equi-depth histograms divide
//! the data into buckets with approximately equal numbers of rows.
//!
//! Benefits:
//! - Better estimates for skewed data distributions
//! - More accurate range selectivity than assuming uniform distribution
//! - Adaptive to actual data characteristics

use crate::query::plan::{
    AggregateOp, BinaryOp, DistinctOp, ExpandOp, FilterOp, JoinOp, JoinType, LimitOp,
    LogicalExpression, LogicalOperator, NodeScanOp, ProjectOp, SkipOp, SortOp, UnaryOp,
    VectorJoinOp, VectorScanOp,
};
use std::collections::HashMap;

/// A bucket in an equi-depth histogram.
///
/// Each bucket represents a range of values and the frequency of rows
/// falling within that range. In an equi-depth histogram, all buckets
/// contain approximately the same number of rows.
#[derive(Debug, Clone)]
pub struct HistogramBucket {
    /// Lower bound of the bucket (inclusive).
    pub lower_bound: f64,
    /// Upper bound of the bucket (exclusive, except for the last bucket).
    pub upper_bound: f64,
    /// Number of rows in this bucket.
    pub frequency: u64,
    /// Number of distinct values in this bucket.
    pub distinct_count: u64,
}

impl HistogramBucket {
    /// Creates a new histogram bucket.
    #[must_use]
    pub fn new(lower_bound: f64, upper_bound: f64, frequency: u64, distinct_count: u64) -> Self {
        Self {
            lower_bound,
            upper_bound,
            frequency,
            distinct_count,
        }
    }

    /// Returns the width of this bucket.
    #[must_use]
    pub fn width(&self) -> f64 {
        self.upper_bound - self.lower_bound
    }

    /// Checks if a value falls within this bucket.
    #[must_use]
    pub fn contains(&self, value: f64) -> bool {
        value >= self.lower_bound && value < self.upper_bound
    }

    /// Returns the fraction of this bucket covered by the given range.
    #[must_use]
    pub fn overlap_fraction(&self, lower: Option<f64>, upper: Option<f64>) -> f64 {
        let effective_lower = lower.unwrap_or(self.lower_bound).max(self.lower_bound);
        let effective_upper = upper.unwrap_or(self.upper_bound).min(self.upper_bound);

        let bucket_width = self.width();
        if bucket_width <= 0.0 {
            return if effective_lower <= self.lower_bound && effective_upper >= self.upper_bound {
                1.0
            } else {
                0.0
            };
        }

        let overlap = (effective_upper - effective_lower).max(0.0);
        (overlap / bucket_width).min(1.0)
    }
}

/// An equi-depth histogram for selectivity estimation.
///
/// Equi-depth histograms partition data into buckets where each bucket
/// contains approximately the same number of rows. This provides more
/// accurate selectivity estimates than assuming uniform distribution,
/// especially for skewed data.
///
/// # Example
///
/// ```ignore
/// use grafeo_engine::query::optimizer::cardinality::EquiDepthHistogram;
///
/// // Build a histogram from sorted values
/// let values = vec![1.0, 2.0, 3.0, 4.0, 5.0, 10.0, 20.0, 30.0, 40.0, 50.0];
/// let histogram = EquiDepthHistogram::build(&values, 4);
///
/// // Estimate selectivity for age > 25
/// let selectivity = histogram.range_selectivity(Some(25.0), None);
/// ```
#[derive(Debug, Clone)]
pub struct EquiDepthHistogram {
    /// The histogram buckets, sorted by lower_bound.
    buckets: Vec<HistogramBucket>,
    /// Total number of rows represented by this histogram.
    total_rows: u64,
}

impl EquiDepthHistogram {
    /// Creates a new histogram from pre-built buckets.
    #[must_use]
    pub fn new(buckets: Vec<HistogramBucket>) -> Self {
        let total_rows = buckets.iter().map(|b| b.frequency).sum();
        Self {
            buckets,
            total_rows,
        }
    }

    /// Builds an equi-depth histogram from a sorted slice of values.
    ///
    /// # Arguments
    /// * `values` - A sorted slice of numeric values
    /// * `num_buckets` - The desired number of buckets
    ///
    /// # Returns
    /// An equi-depth histogram with approximately equal row counts per bucket.
    #[must_use]
    pub fn build(values: &[f64], num_buckets: usize) -> Self {
        if values.is_empty() || num_buckets == 0 {
            return Self {
                buckets: Vec::new(),
                total_rows: 0,
            };
        }

        let num_buckets = num_buckets.min(values.len());
        let rows_per_bucket = (values.len() + num_buckets - 1) / num_buckets;
        let mut buckets = Vec::with_capacity(num_buckets);

        let mut start_idx = 0;
        while start_idx < values.len() {
            let end_idx = (start_idx + rows_per_bucket).min(values.len());
            let lower_bound = values[start_idx];
            let upper_bound = if end_idx < values.len() {
                values[end_idx]
            } else {
                // For the last bucket, extend slightly beyond the max value
                values[end_idx - 1] + 1.0
            };

            // Count distinct values in this bucket
            let bucket_values = &values[start_idx..end_idx];
            let distinct_count = count_distinct(bucket_values);

            buckets.push(HistogramBucket::new(
                lower_bound,
                upper_bound,
                (end_idx - start_idx) as u64,
                distinct_count,
            ));

            start_idx = end_idx;
        }

        Self::new(buckets)
    }

    /// Returns the number of buckets in this histogram.
    #[must_use]
    pub fn num_buckets(&self) -> usize {
        self.buckets.len()
    }

    /// Returns the total number of rows represented.
    #[must_use]
    pub fn total_rows(&self) -> u64 {
        self.total_rows
    }

    /// Returns the histogram buckets.
    #[must_use]
    pub fn buckets(&self) -> &[HistogramBucket] {
        &self.buckets
    }

    /// Estimates selectivity for a range predicate.
    ///
    /// # Arguments
    /// * `lower` - Lower bound (None for unbounded)
    /// * `upper` - Upper bound (None for unbounded)
    ///
    /// # Returns
    /// Estimated fraction of rows matching the range (0.0 to 1.0).
    #[must_use]
    pub fn range_selectivity(&self, lower: Option<f64>, upper: Option<f64>) -> f64 {
        if self.buckets.is_empty() || self.total_rows == 0 {
            return 0.33; // Default fallback
        }

        let mut matching_rows = 0.0;

        for bucket in &self.buckets {
            // Check if this bucket overlaps with the range
            let bucket_lower = bucket.lower_bound;
            let bucket_upper = bucket.upper_bound;

            // Skip buckets entirely outside the range
            if let Some(l) = lower {
                if bucket_upper <= l {
                    continue;
                }
            }
            if let Some(u) = upper {
                if bucket_lower >= u {
                    continue;
                }
            }

            // Calculate the fraction of this bucket covered by the range
            let overlap = bucket.overlap_fraction(lower, upper);
            matching_rows += overlap * bucket.frequency as f64;
        }

        (matching_rows / self.total_rows as f64).min(1.0).max(0.0)
    }

    /// Estimates selectivity for an equality predicate.
    ///
    /// Uses the distinct count within matching buckets for better accuracy.
    #[must_use]
    pub fn equality_selectivity(&self, value: f64) -> f64 {
        if self.buckets.is_empty() || self.total_rows == 0 {
            return 0.01; // Default fallback
        }

        // Find the bucket containing this value
        for bucket in &self.buckets {
            if bucket.contains(value) {
                // Assume uniform distribution within bucket
                if bucket.distinct_count > 0 {
                    return (bucket.frequency as f64
                        / bucket.distinct_count as f64
                        / self.total_rows as f64)
                        .min(1.0);
                }
            }
        }

        // Value not in any bucket - very low selectivity
        0.001
    }

    /// Gets the minimum value in the histogram.
    #[must_use]
    pub fn min_value(&self) -> Option<f64> {
        self.buckets.first().map(|b| b.lower_bound)
    }

    /// Gets the maximum value in the histogram.
    #[must_use]
    pub fn max_value(&self) -> Option<f64> {
        self.buckets.last().map(|b| b.upper_bound)
    }
}

/// Counts distinct values in a sorted slice.
fn count_distinct(sorted_values: &[f64]) -> u64 {
    if sorted_values.is_empty() {
        return 0;
    }

    let mut count = 1u64;
    let mut prev = sorted_values[0];

    for &val in &sorted_values[1..] {
        if (val - prev).abs() > f64::EPSILON {
            count += 1;
            prev = val;
        }
    }

    count
}

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
    /// Equi-depth histogram for accurate selectivity estimation.
    pub histogram: Option<EquiDepthHistogram>,
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
            histogram: None,
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

    /// Sets the equi-depth histogram for this column.
    #[must_use]
    pub fn with_histogram(mut self, histogram: EquiDepthHistogram) -> Self {
        self.histogram = Some(histogram);
        self
    }

    /// Builds column statistics with histogram from raw values.
    ///
    /// This is a convenience method that computes all statistics from the data.
    ///
    /// # Arguments
    /// * `values` - The column values (will be sorted internally)
    /// * `num_buckets` - Number of histogram buckets to create
    #[must_use]
    pub fn from_values(mut values: Vec<f64>, num_buckets: usize) -> Self {
        if values.is_empty() {
            return Self::new(0);
        }

        // Sort values for histogram building
        values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let min = values.first().copied();
        let max = values.last().copied();
        let distinct_count = count_distinct(&values);
        let histogram = EquiDepthHistogram::build(&values, num_buckets);

        Self {
            distinct_count,
            null_count: 0,
            min_value: min,
            max_value: max,
            histogram: Some(histogram),
        }
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
            LogicalOperator::VectorScan(scan) => self.estimate_vector_scan(scan),
            LogicalOperator::VectorJoin(join) => self.estimate_vector_join(join),
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

    /// Estimates vector scan cardinality.
    ///
    /// Vector scan returns at most k results (the k nearest neighbors).
    /// With similarity/distance filters, it may return fewer.
    fn estimate_vector_scan(&self, scan: &VectorScanOp) -> f64 {
        let base_k = scan.k as f64;

        // Apply filter selectivity if thresholds are specified
        let selectivity = if scan.min_similarity.is_some() || scan.max_distance.is_some() {
            // Assume 70% of results pass threshold filters
            0.7
        } else {
            1.0
        };

        (base_k * selectivity).max(1.0)
    }

    /// Estimates vector join cardinality.
    ///
    /// Vector join produces up to k results per input row.
    fn estimate_vector_join(&self, join: &VectorJoinOp) -> f64 {
        let input_cardinality = self.estimate(&join.input);
        let k = join.k as f64;

        // Apply filter selectivity if thresholds are specified
        let selectivity = if join.min_similarity.is_some() || join.max_distance.is_some() {
            0.7
        } else {
            1.0
        };

        (input_cardinality * k * selectivity).max(1.0)
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
                if let grafeo_common::types::Value::Bool(b) = value {
                    if *b { 1.0 } else { 0.0 }
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
        left: &LogicalExpression,
        op: BinaryOp,
        right: &LogicalExpression,
    ) -> f64 {
        match op {
            // Equality - try histogram-based estimation
            BinaryOp::Eq => {
                if let Some(selectivity) = self.try_equality_selectivity(left, right) {
                    return selectivity;
                }
                0.01
            }
            // Inequality is very unselective
            BinaryOp::Ne => 0.99,
            // Range predicates - use histogram if available
            BinaryOp::Lt | BinaryOp::Le | BinaryOp::Gt | BinaryOp::Ge => {
                if let Some(selectivity) = self.try_range_selectivity(left, op, right) {
                    return selectivity;
                }
                0.33
            }
            // Logical operators - recursively estimate sub-expressions
            BinaryOp::And => {
                let left_sel = self.estimate_selectivity(left);
                let right_sel = self.estimate_selectivity(right);
                // AND reduces selectivity (multiply assuming independence)
                left_sel * right_sel
            }
            BinaryOp::Or => {
                let left_sel = self.estimate_selectivity(left);
                let right_sel = self.estimate_selectivity(right);
                // OR: P(A ∪ B) = P(A) + P(B) - P(A ∩ B)
                // Assuming independence: P(A ∩ B) = P(A) * P(B)
                (left_sel + right_sel - left_sel * right_sel).min(1.0)
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

    /// Tries to estimate equality selectivity using histograms.
    fn try_equality_selectivity(
        &self,
        left: &LogicalExpression,
        right: &LogicalExpression,
    ) -> Option<f64> {
        // Extract property access and literal value
        let (label, column, value) = self.extract_column_and_value(left, right)?;

        // Get column stats with histogram
        let stats = self.get_column_stats(&label, &column)?;

        // Try histogram-based estimation
        if let Some(ref histogram) = stats.histogram {
            return Some(histogram.equality_selectivity(value));
        }

        // Fall back to distinct count estimation
        if stats.distinct_count > 0 {
            return Some(1.0 / stats.distinct_count as f64);
        }

        None
    }

    /// Tries to estimate range selectivity using histograms.
    fn try_range_selectivity(
        &self,
        left: &LogicalExpression,
        op: BinaryOp,
        right: &LogicalExpression,
    ) -> Option<f64> {
        // Extract property access and literal value
        let (label, column, value) = self.extract_column_and_value(left, right)?;

        // Get column stats
        let stats = self.get_column_stats(&label, &column)?;

        // Determine the range based on operator
        let (lower, upper) = match op {
            BinaryOp::Lt => (None, Some(value)),
            BinaryOp::Le => (None, Some(value + f64::EPSILON)),
            BinaryOp::Gt => (Some(value + f64::EPSILON), None),
            BinaryOp::Ge => (Some(value), None),
            _ => return None,
        };

        // Try histogram-based estimation first
        if let Some(ref histogram) = stats.histogram {
            return Some(histogram.range_selectivity(lower, upper));
        }

        // Fall back to min/max range estimation
        if let (Some(min), Some(max)) = (stats.min_value, stats.max_value) {
            let range = max - min;
            if range <= 0.0 {
                return Some(1.0);
            }

            let effective_lower = lower.unwrap_or(min).max(min);
            let effective_upper = upper.unwrap_or(max).min(max);
            let overlap = (effective_upper - effective_lower).max(0.0);
            return Some((overlap / range).min(1.0).max(0.0));
        }

        None
    }

    /// Extracts column information and literal value from a comparison.
    ///
    /// Returns (label, column_name, numeric_value) if the expression is
    /// a comparison between a property access and a numeric literal.
    fn extract_column_and_value(
        &self,
        left: &LogicalExpression,
        right: &LogicalExpression,
    ) -> Option<(String, String, f64)> {
        // Try left as property, right as literal
        if let Some(result) = self.try_extract_property_literal(left, right) {
            return Some(result);
        }

        // Try right as property, left as literal
        self.try_extract_property_literal(right, left)
    }

    /// Tries to extract property and literal from a specific ordering.
    fn try_extract_property_literal(
        &self,
        property_expr: &LogicalExpression,
        literal_expr: &LogicalExpression,
    ) -> Option<(String, String, f64)> {
        // Extract property access
        let (variable, property) = match property_expr {
            LogicalExpression::Property { variable, property } => {
                (variable.clone(), property.clone())
            }
            _ => return None,
        };

        // Extract numeric literal
        let value = match literal_expr {
            LogicalExpression::Literal(grafeo_common::types::Value::Int64(n)) => *n as f64,
            LogicalExpression::Literal(grafeo_common::types::Value::Float64(f)) => *f,
            _ => return None,
        };

        // Try to find a label for this variable from table stats
        // Use the variable name as a heuristic label lookup
        // In practice, the optimizer would track which labels variables are bound to
        for label in self.table_stats.keys() {
            if let Some(stats) = self.table_stats.get(label) {
                if stats.columns.contains_key(&property) {
                    return Some((label.clone(), property, value));
                }
            }
        }

        // If no stats found but we have the property, return with variable as label
        Some((variable, property, value))
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
    use crate::query::plan::{
        DistinctOp, ExpandDirection, ExpandOp, FilterOp, JoinCondition, NodeScanOp, ProjectOp,
        Projection, ReturnItem, ReturnOp, SkipOp, SortKey, SortOp, SortOrder,
    };
    use grafeo_common::types::Value;

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
            having: None,
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
            having: None,
        });

        let cardinality = estimator.estimate(&group_agg);
        // Should be less than input
        assert!(cardinality < 1000.0);
    }

    #[test]
    fn test_node_scan_without_stats() {
        let estimator = CardinalityEstimator::new();

        let scan = LogicalOperator::NodeScan(NodeScanOp {
            variable: "n".to_string(),
            label: Some("Unknown".to_string()),
            input: None,
        });

        let cardinality = estimator.estimate(&scan);
        // Should return default (1000)
        assert!((cardinality - 1000.0).abs() < 0.001);
    }

    #[test]
    fn test_node_scan_no_label() {
        let estimator = CardinalityEstimator::new();

        let scan = LogicalOperator::NodeScan(NodeScanOp {
            variable: "n".to_string(),
            label: None,
            input: None,
        });

        let cardinality = estimator.estimate(&scan);
        // Should scan all nodes (default)
        assert!((cardinality - 1000.0).abs() < 0.001);
    }

    #[test]
    fn test_filter_inequality_selectivity() {
        let mut estimator = CardinalityEstimator::new();
        estimator.add_table_stats("Person", TableStats::new(1000));

        let filter = LogicalOperator::Filter(FilterOp {
            predicate: LogicalExpression::Binary {
                left: Box::new(LogicalExpression::Property {
                    variable: "n".to_string(),
                    property: "age".to_string(),
                }),
                op: BinaryOp::Ne,
                right: Box::new(LogicalExpression::Literal(Value::Int64(30))),
            },
            input: Box::new(LogicalOperator::NodeScan(NodeScanOp {
                variable: "n".to_string(),
                label: Some("Person".to_string()),
                input: None,
            })),
        });

        let cardinality = estimator.estimate(&filter);
        // Inequality selectivity is 0.99, so 1000 * 0.99 = 990
        assert!(cardinality > 900.0);
    }

    #[test]
    fn test_filter_range_selectivity() {
        let mut estimator = CardinalityEstimator::new();
        estimator.add_table_stats("Person", TableStats::new(1000));

        let filter = LogicalOperator::Filter(FilterOp {
            predicate: LogicalExpression::Binary {
                left: Box::new(LogicalExpression::Property {
                    variable: "n".to_string(),
                    property: "age".to_string(),
                }),
                op: BinaryOp::Gt,
                right: Box::new(LogicalExpression::Literal(Value::Int64(30))),
            },
            input: Box::new(LogicalOperator::NodeScan(NodeScanOp {
                variable: "n".to_string(),
                label: Some("Person".to_string()),
                input: None,
            })),
        });

        let cardinality = estimator.estimate(&filter);
        // Range selectivity is 0.33, so 1000 * 0.33 = 330
        assert!(cardinality < 500.0);
        assert!(cardinality > 100.0);
    }

    #[test]
    fn test_filter_and_selectivity() {
        let mut estimator = CardinalityEstimator::new();
        estimator.add_table_stats("Person", TableStats::new(1000));

        // Test AND with two equality predicates
        // Each equality has selectivity 0.01, so AND gives 0.01 * 0.01 = 0.0001
        let filter = LogicalOperator::Filter(FilterOp {
            predicate: LogicalExpression::Binary {
                left: Box::new(LogicalExpression::Binary {
                    left: Box::new(LogicalExpression::Property {
                        variable: "n".to_string(),
                        property: "city".to_string(),
                    }),
                    op: BinaryOp::Eq,
                    right: Box::new(LogicalExpression::Literal(Value::String("NYC".into()))),
                }),
                op: BinaryOp::And,
                right: Box::new(LogicalExpression::Binary {
                    left: Box::new(LogicalExpression::Property {
                        variable: "n".to_string(),
                        property: "age".to_string(),
                    }),
                    op: BinaryOp::Eq,
                    right: Box::new(LogicalExpression::Literal(Value::Int64(30))),
                }),
            },
            input: Box::new(LogicalOperator::NodeScan(NodeScanOp {
                variable: "n".to_string(),
                label: Some("Person".to_string()),
                input: None,
            })),
        });

        let cardinality = estimator.estimate(&filter);
        // AND reduces selectivity (multiply): 0.01 * 0.01 = 0.0001
        // 1000 * 0.0001 = 0.1, min is 1.0
        assert!(cardinality < 100.0);
        assert!(cardinality >= 1.0);
    }

    #[test]
    fn test_filter_or_selectivity() {
        let mut estimator = CardinalityEstimator::new();
        estimator.add_table_stats("Person", TableStats::new(1000));

        // Test OR with two equality predicates
        // Each equality has selectivity 0.01
        // OR gives: 0.01 + 0.01 - (0.01 * 0.01) = 0.0199
        let filter = LogicalOperator::Filter(FilterOp {
            predicate: LogicalExpression::Binary {
                left: Box::new(LogicalExpression::Binary {
                    left: Box::new(LogicalExpression::Property {
                        variable: "n".to_string(),
                        property: "city".to_string(),
                    }),
                    op: BinaryOp::Eq,
                    right: Box::new(LogicalExpression::Literal(Value::String("NYC".into()))),
                }),
                op: BinaryOp::Or,
                right: Box::new(LogicalExpression::Binary {
                    left: Box::new(LogicalExpression::Property {
                        variable: "n".to_string(),
                        property: "city".to_string(),
                    }),
                    op: BinaryOp::Eq,
                    right: Box::new(LogicalExpression::Literal(Value::String("LA".into()))),
                }),
            },
            input: Box::new(LogicalOperator::NodeScan(NodeScanOp {
                variable: "n".to_string(),
                label: Some("Person".to_string()),
                input: None,
            })),
        });

        let cardinality = estimator.estimate(&filter);
        // OR: 0.01 + 0.01 - 0.0001 ≈ 0.0199, so 1000 * 0.0199 ≈ 19.9
        assert!(cardinality < 100.0);
        assert!(cardinality >= 1.0);
    }

    #[test]
    fn test_filter_literal_true() {
        let mut estimator = CardinalityEstimator::new();
        estimator.add_table_stats("Person", TableStats::new(1000));

        let filter = LogicalOperator::Filter(FilterOp {
            predicate: LogicalExpression::Literal(Value::Bool(true)),
            input: Box::new(LogicalOperator::NodeScan(NodeScanOp {
                variable: "n".to_string(),
                label: Some("Person".to_string()),
                input: None,
            })),
        });

        let cardinality = estimator.estimate(&filter);
        // Literal true has selectivity 1.0
        assert!((cardinality - 1000.0).abs() < 0.001);
    }

    #[test]
    fn test_filter_literal_false() {
        let mut estimator = CardinalityEstimator::new();
        estimator.add_table_stats("Person", TableStats::new(1000));

        let filter = LogicalOperator::Filter(FilterOp {
            predicate: LogicalExpression::Literal(Value::Bool(false)),
            input: Box::new(LogicalOperator::NodeScan(NodeScanOp {
                variable: "n".to_string(),
                label: Some("Person".to_string()),
                input: None,
            })),
        });

        let cardinality = estimator.estimate(&filter);
        // Literal false has selectivity 0.0, but min is 1.0
        assert!((cardinality - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_unary_not_selectivity() {
        let mut estimator = CardinalityEstimator::new();
        estimator.add_table_stats("Person", TableStats::new(1000));

        let filter = LogicalOperator::Filter(FilterOp {
            predicate: LogicalExpression::Unary {
                op: UnaryOp::Not,
                operand: Box::new(LogicalExpression::Literal(Value::Bool(true))),
            },
            input: Box::new(LogicalOperator::NodeScan(NodeScanOp {
                variable: "n".to_string(),
                label: Some("Person".to_string()),
                input: None,
            })),
        });

        let cardinality = estimator.estimate(&filter);
        // NOT inverts selectivity
        assert!(cardinality < 1000.0);
    }

    #[test]
    fn test_unary_is_null_selectivity() {
        let mut estimator = CardinalityEstimator::new();
        estimator.add_table_stats("Person", TableStats::new(1000));

        let filter = LogicalOperator::Filter(FilterOp {
            predicate: LogicalExpression::Unary {
                op: UnaryOp::IsNull,
                operand: Box::new(LogicalExpression::Variable("x".to_string())),
            },
            input: Box::new(LogicalOperator::NodeScan(NodeScanOp {
                variable: "n".to_string(),
                label: Some("Person".to_string()),
                input: None,
            })),
        });

        let cardinality = estimator.estimate(&filter);
        // IS NULL has selectivity 0.05
        assert!(cardinality < 100.0);
    }

    #[test]
    fn test_expand_cardinality() {
        let mut estimator = CardinalityEstimator::new();
        estimator.add_table_stats("Person", TableStats::new(100));

        let expand = LogicalOperator::Expand(ExpandOp {
            from_variable: "a".to_string(),
            to_variable: "b".to_string(),
            edge_variable: None,
            direction: ExpandDirection::Outgoing,
            edge_type: None,
            min_hops: 1,
            max_hops: Some(1),
            input: Box::new(LogicalOperator::NodeScan(NodeScanOp {
                variable: "a".to_string(),
                label: Some("Person".to_string()),
                input: None,
            })),
            path_alias: None,
        });

        let cardinality = estimator.estimate(&expand);
        // Expand multiplies by fanout (10)
        assert!(cardinality > 100.0);
    }

    #[test]
    fn test_expand_with_edge_type_filter() {
        let mut estimator = CardinalityEstimator::new();
        estimator.add_table_stats("Person", TableStats::new(100));

        let expand = LogicalOperator::Expand(ExpandOp {
            from_variable: "a".to_string(),
            to_variable: "b".to_string(),
            edge_variable: None,
            direction: ExpandDirection::Outgoing,
            edge_type: Some("KNOWS".to_string()),
            min_hops: 1,
            max_hops: Some(1),
            input: Box::new(LogicalOperator::NodeScan(NodeScanOp {
                variable: "a".to_string(),
                label: Some("Person".to_string()),
                input: None,
            })),
            path_alias: None,
        });

        let cardinality = estimator.estimate(&expand);
        // With edge type, fanout is reduced by half
        assert!(cardinality > 100.0);
    }

    #[test]
    fn test_expand_variable_length() {
        let mut estimator = CardinalityEstimator::new();
        estimator.add_table_stats("Person", TableStats::new(100));

        let expand = LogicalOperator::Expand(ExpandOp {
            from_variable: "a".to_string(),
            to_variable: "b".to_string(),
            edge_variable: None,
            direction: ExpandDirection::Outgoing,
            edge_type: None,
            min_hops: 1,
            max_hops: Some(3),
            input: Box::new(LogicalOperator::NodeScan(NodeScanOp {
                variable: "a".to_string(),
                label: Some("Person".to_string()),
                input: None,
            })),
            path_alias: None,
        });

        let cardinality = estimator.estimate(&expand);
        // Variable length path has much higher cardinality
        assert!(cardinality > 500.0);
    }

    #[test]
    fn test_join_cross_product() {
        let mut estimator = CardinalityEstimator::new();
        estimator.add_table_stats("Person", TableStats::new(100));
        estimator.add_table_stats("Company", TableStats::new(50));

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
            join_type: JoinType::Cross,
            conditions: vec![],
        });

        let cardinality = estimator.estimate(&join);
        // Cross join = 100 * 50 = 5000
        assert!((cardinality - 5000.0).abs() < 0.001);
    }

    #[test]
    fn test_join_left_outer() {
        let mut estimator = CardinalityEstimator::new();
        estimator.add_table_stats("Person", TableStats::new(1000));
        estimator.add_table_stats("Company", TableStats::new(10));

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
            join_type: JoinType::Left,
            conditions: vec![JoinCondition {
                left: LogicalExpression::Variable("p".to_string()),
                right: LogicalExpression::Variable("c".to_string()),
            }],
        });

        let cardinality = estimator.estimate(&join);
        // Left join returns at least all left rows
        assert!(cardinality >= 1000.0);
    }

    #[test]
    fn test_join_semi() {
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
            join_type: JoinType::Semi,
            conditions: vec![],
        });

        let cardinality = estimator.estimate(&join);
        // Semi join returns at most left cardinality
        assert!(cardinality <= 1000.0);
    }

    #[test]
    fn test_join_anti() {
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
            join_type: JoinType::Anti,
            conditions: vec![],
        });

        let cardinality = estimator.estimate(&join);
        // Anti join returns at most left cardinality
        assert!(cardinality <= 1000.0);
        assert!(cardinality >= 1.0);
    }

    #[test]
    fn test_project_preserves_cardinality() {
        let mut estimator = CardinalityEstimator::new();
        estimator.add_table_stats("Person", TableStats::new(1000));

        let project = LogicalOperator::Project(ProjectOp {
            projections: vec![Projection {
                expression: LogicalExpression::Variable("n".to_string()),
                alias: None,
            }],
            input: Box::new(LogicalOperator::NodeScan(NodeScanOp {
                variable: "n".to_string(),
                label: Some("Person".to_string()),
                input: None,
            })),
        });

        let cardinality = estimator.estimate(&project);
        assert!((cardinality - 1000.0).abs() < 0.001);
    }

    #[test]
    fn test_sort_preserves_cardinality() {
        let mut estimator = CardinalityEstimator::new();
        estimator.add_table_stats("Person", TableStats::new(1000));

        let sort = LogicalOperator::Sort(SortOp {
            keys: vec![SortKey {
                expression: LogicalExpression::Variable("n".to_string()),
                order: SortOrder::Ascending,
            }],
            input: Box::new(LogicalOperator::NodeScan(NodeScanOp {
                variable: "n".to_string(),
                label: Some("Person".to_string()),
                input: None,
            })),
        });

        let cardinality = estimator.estimate(&sort);
        assert!((cardinality - 1000.0).abs() < 0.001);
    }

    #[test]
    fn test_distinct_reduces_cardinality() {
        let mut estimator = CardinalityEstimator::new();
        estimator.add_table_stats("Person", TableStats::new(1000));

        let distinct = LogicalOperator::Distinct(DistinctOp {
            input: Box::new(LogicalOperator::NodeScan(NodeScanOp {
                variable: "n".to_string(),
                label: Some("Person".to_string()),
                input: None,
            })),
            columns: None,
        });

        let cardinality = estimator.estimate(&distinct);
        // Distinct assumes 50% unique
        assert!((cardinality - 500.0).abs() < 0.001);
    }

    #[test]
    fn test_skip_reduces_cardinality() {
        let mut estimator = CardinalityEstimator::new();
        estimator.add_table_stats("Person", TableStats::new(1000));

        let skip = LogicalOperator::Skip(SkipOp {
            count: 100,
            input: Box::new(LogicalOperator::NodeScan(NodeScanOp {
                variable: "n".to_string(),
                label: Some("Person".to_string()),
                input: None,
            })),
        });

        let cardinality = estimator.estimate(&skip);
        assert!((cardinality - 900.0).abs() < 0.001);
    }

    #[test]
    fn test_return_preserves_cardinality() {
        let mut estimator = CardinalityEstimator::new();
        estimator.add_table_stats("Person", TableStats::new(1000));

        let ret = LogicalOperator::Return(ReturnOp {
            items: vec![ReturnItem {
                expression: LogicalExpression::Variable("n".to_string()),
                alias: None,
            }],
            distinct: false,
            input: Box::new(LogicalOperator::NodeScan(NodeScanOp {
                variable: "n".to_string(),
                label: Some("Person".to_string()),
                input: None,
            })),
        });

        let cardinality = estimator.estimate(&ret);
        assert!((cardinality - 1000.0).abs() < 0.001);
    }

    #[test]
    fn test_empty_cardinality() {
        let estimator = CardinalityEstimator::new();
        let cardinality = estimator.estimate(&LogicalOperator::Empty);
        assert!((cardinality).abs() < 0.001);
    }

    #[test]
    fn test_table_stats_with_column() {
        let stats = TableStats::new(1000).with_column(
            "age",
            ColumnStats::new(50).with_nulls(10).with_range(0.0, 100.0),
        );

        assert_eq!(stats.row_count, 1000);
        let col = stats.columns.get("age").unwrap();
        assert_eq!(col.distinct_count, 50);
        assert_eq!(col.null_count, 10);
        assert!((col.min_value.unwrap() - 0.0).abs() < 0.001);
        assert!((col.max_value.unwrap() - 100.0).abs() < 0.001);
    }

    #[test]
    fn test_estimator_default() {
        let estimator = CardinalityEstimator::default();
        let scan = LogicalOperator::NodeScan(NodeScanOp {
            variable: "n".to_string(),
            label: None,
            input: None,
        });
        let cardinality = estimator.estimate(&scan);
        assert!((cardinality - 1000.0).abs() < 0.001);
    }

    #[test]
    fn test_set_avg_fanout() {
        let mut estimator = CardinalityEstimator::new();
        estimator.add_table_stats("Person", TableStats::new(100));
        estimator.set_avg_fanout(5.0);

        let expand = LogicalOperator::Expand(ExpandOp {
            from_variable: "a".to_string(),
            to_variable: "b".to_string(),
            edge_variable: None,
            direction: ExpandDirection::Outgoing,
            edge_type: None,
            min_hops: 1,
            max_hops: Some(1),
            input: Box::new(LogicalOperator::NodeScan(NodeScanOp {
                variable: "a".to_string(),
                label: Some("Person".to_string()),
                input: None,
            })),
            path_alias: None,
        });

        let cardinality = estimator.estimate(&expand);
        // With fanout 5: 100 * 5 = 500
        assert!((cardinality - 500.0).abs() < 0.001);
    }

    #[test]
    fn test_multiple_group_by_keys_reduce_cardinality() {
        // The current implementation uses a simplified model where more group by keys
        // results in greater reduction (dividing by 10^num_keys). This is a simplification
        // that works for most cases where group by keys are correlated.
        let mut estimator = CardinalityEstimator::new();
        estimator.add_table_stats("Person", TableStats::new(10000));

        let single_group = LogicalOperator::Aggregate(AggregateOp {
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
            having: None,
        });

        let multi_group = LogicalOperator::Aggregate(AggregateOp {
            group_by: vec![
                LogicalExpression::Property {
                    variable: "n".to_string(),
                    property: "city".to_string(),
                },
                LogicalExpression::Property {
                    variable: "n".to_string(),
                    property: "country".to_string(),
                },
            ],
            aggregates: vec![],
            input: Box::new(LogicalOperator::NodeScan(NodeScanOp {
                variable: "n".to_string(),
                label: Some("Person".to_string()),
                input: None,
            })),
            having: None,
        });

        let single_card = estimator.estimate(&single_group);
        let multi_card = estimator.estimate(&multi_group);

        // Both should reduce cardinality from input
        assert!(single_card < 10000.0);
        assert!(multi_card < 10000.0);
        // Both should be at least 1
        assert!(single_card >= 1.0);
        assert!(multi_card >= 1.0);
    }

    // ============= Histogram Tests =============

    #[test]
    fn test_histogram_build_uniform() {
        // Build histogram from uniformly distributed data
        let values: Vec<f64> = (0..100).map(|i| i as f64).collect();
        let histogram = EquiDepthHistogram::build(&values, 10);

        assert_eq!(histogram.num_buckets(), 10);
        assert_eq!(histogram.total_rows(), 100);

        // Each bucket should have approximately 10 rows
        for bucket in histogram.buckets() {
            assert!(bucket.frequency >= 9 && bucket.frequency <= 11);
        }
    }

    #[test]
    fn test_histogram_build_skewed() {
        // Build histogram from skewed data (many small values, few large)
        let mut values: Vec<f64> = (0..80).map(|i| i as f64).collect();
        values.extend((0..20).map(|i| 1000.0 + i as f64));
        let histogram = EquiDepthHistogram::build(&values, 5);

        assert_eq!(histogram.num_buckets(), 5);
        assert_eq!(histogram.total_rows(), 100);

        // Each bucket should have ~20 rows despite skewed data
        for bucket in histogram.buckets() {
            assert!(bucket.frequency >= 18 && bucket.frequency <= 22);
        }
    }

    #[test]
    fn test_histogram_range_selectivity_full() {
        let values: Vec<f64> = (0..100).map(|i| i as f64).collect();
        let histogram = EquiDepthHistogram::build(&values, 10);

        // Full range should have selectivity ~1.0
        let selectivity = histogram.range_selectivity(None, None);
        assert!((selectivity - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_histogram_range_selectivity_half() {
        let values: Vec<f64> = (0..100).map(|i| i as f64).collect();
        let histogram = EquiDepthHistogram::build(&values, 10);

        // Values >= 50 should be ~50% (half the data)
        let selectivity = histogram.range_selectivity(Some(50.0), None);
        assert!(selectivity > 0.4 && selectivity < 0.6);
    }

    #[test]
    fn test_histogram_range_selectivity_quarter() {
        let values: Vec<f64> = (0..100).map(|i| i as f64).collect();
        let histogram = EquiDepthHistogram::build(&values, 10);

        // Values < 25 should be ~25%
        let selectivity = histogram.range_selectivity(None, Some(25.0));
        assert!(selectivity > 0.2 && selectivity < 0.3);
    }

    #[test]
    fn test_histogram_equality_selectivity() {
        let values: Vec<f64> = (0..100).map(|i| i as f64).collect();
        let histogram = EquiDepthHistogram::build(&values, 10);

        // Equality on 100 distinct values should be ~1%
        let selectivity = histogram.equality_selectivity(50.0);
        assert!(selectivity > 0.005 && selectivity < 0.02);
    }

    #[test]
    fn test_histogram_empty() {
        let histogram = EquiDepthHistogram::build(&[], 10);

        assert_eq!(histogram.num_buckets(), 0);
        assert_eq!(histogram.total_rows(), 0);

        // Default selectivity for empty histogram
        let selectivity = histogram.range_selectivity(Some(0.0), Some(100.0));
        assert!((selectivity - 0.33).abs() < 0.01);
    }

    #[test]
    fn test_histogram_bucket_overlap() {
        let bucket = HistogramBucket::new(10.0, 20.0, 100, 10);

        // Full overlap
        assert!((bucket.overlap_fraction(Some(10.0), Some(20.0)) - 1.0).abs() < 0.01);

        // Half overlap (lower half)
        assert!((bucket.overlap_fraction(Some(10.0), Some(15.0)) - 0.5).abs() < 0.01);

        // Half overlap (upper half)
        assert!((bucket.overlap_fraction(Some(15.0), Some(20.0)) - 0.5).abs() < 0.01);

        // No overlap (below)
        assert!((bucket.overlap_fraction(Some(0.0), Some(5.0))).abs() < 0.01);

        // No overlap (above)
        assert!((bucket.overlap_fraction(Some(25.0), Some(30.0))).abs() < 0.01);
    }

    #[test]
    fn test_column_stats_from_values() {
        let values = vec![10.0, 20.0, 30.0, 40.0, 50.0, 20.0, 30.0, 40.0];
        let stats = ColumnStats::from_values(values, 4);

        assert_eq!(stats.distinct_count, 5); // 10, 20, 30, 40, 50
        assert!(stats.min_value.is_some());
        assert!((stats.min_value.unwrap() - 10.0).abs() < 0.01);
        assert!(stats.max_value.is_some());
        assert!((stats.max_value.unwrap() - 50.0).abs() < 0.01);
        assert!(stats.histogram.is_some());
    }

    #[test]
    fn test_column_stats_with_histogram_builder() {
        let values: Vec<f64> = (0..100).map(|i| i as f64).collect();
        let histogram = EquiDepthHistogram::build(&values, 10);

        let stats = ColumnStats::new(100)
            .with_range(0.0, 99.0)
            .with_histogram(histogram);

        assert!(stats.histogram.is_some());
        assert_eq!(stats.histogram.as_ref().unwrap().num_buckets(), 10);
    }

    #[test]
    fn test_filter_with_histogram_stats() {
        let mut estimator = CardinalityEstimator::new();

        // Create stats with histogram for age column
        let age_values: Vec<f64> = (18..80).map(|i| i as f64).collect();
        let histogram = EquiDepthHistogram::build(&age_values, 10);
        let age_stats = ColumnStats::new(62)
            .with_range(18.0, 79.0)
            .with_histogram(histogram);

        estimator.add_table_stats(
            "Person",
            TableStats::new(1000).with_column("age", age_stats),
        );

        // Filter: age > 50
        // Age range is 18-79, so >50 is about (79-50)/(79-18) = 29/61 ≈ 47.5%
        let filter = LogicalOperator::Filter(FilterOp {
            predicate: LogicalExpression::Binary {
                left: Box::new(LogicalExpression::Property {
                    variable: "n".to_string(),
                    property: "age".to_string(),
                }),
                op: BinaryOp::Gt,
                right: Box::new(LogicalExpression::Literal(Value::Int64(50))),
            },
            input: Box::new(LogicalOperator::NodeScan(NodeScanOp {
                variable: "n".to_string(),
                label: Some("Person".to_string()),
                input: None,
            })),
        });

        let cardinality = estimator.estimate(&filter);

        // With histogram, should get more accurate estimate than default 0.33
        // Expected: ~47.5% of 1000 = ~475
        assert!(cardinality > 300.0 && cardinality < 600.0);
    }

    #[test]
    fn test_filter_equality_with_histogram() {
        let mut estimator = CardinalityEstimator::new();

        // Create stats with histogram
        let values: Vec<f64> = (0..100).map(|i| i as f64).collect();
        let histogram = EquiDepthHistogram::build(&values, 10);
        let stats = ColumnStats::new(100)
            .with_range(0.0, 99.0)
            .with_histogram(histogram);

        estimator.add_table_stats("Data", TableStats::new(1000).with_column("value", stats));

        // Filter: value = 50
        let filter = LogicalOperator::Filter(FilterOp {
            predicate: LogicalExpression::Binary {
                left: Box::new(LogicalExpression::Property {
                    variable: "d".to_string(),
                    property: "value".to_string(),
                }),
                op: BinaryOp::Eq,
                right: Box::new(LogicalExpression::Literal(Value::Int64(50))),
            },
            input: Box::new(LogicalOperator::NodeScan(NodeScanOp {
                variable: "d".to_string(),
                label: Some("Data".to_string()),
                input: None,
            })),
        });

        let cardinality = estimator.estimate(&filter);

        // With 100 distinct values, selectivity should be ~1%
        // 1000 * 0.01 = 10
        assert!(cardinality >= 1.0 && cardinality < 50.0);
    }

    #[test]
    fn test_histogram_min_max() {
        let values: Vec<f64> = vec![5.0, 10.0, 15.0, 20.0, 25.0];
        let histogram = EquiDepthHistogram::build(&values, 2);

        assert_eq!(histogram.min_value(), Some(5.0));
        // Max is the upper bound of the last bucket
        assert!(histogram.max_value().is_some());
    }
}
