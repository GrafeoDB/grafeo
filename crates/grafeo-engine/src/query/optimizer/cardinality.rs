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
    AggregateOp, BinaryOp, DistinctOp, ExpandOp, FilterOp, JoinOp, JoinType, LeftJoinOp, LimitOp,
    LogicalExpression, LogicalOperator, MultiWayJoinOp, NodeScanOp, ProjectOp, SkipOp, SortOp,
    TextScanOp, TripleComponent, TripleScanOp, UnaryOp, VectorJoinOp, VectorScanOp,
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
/// ```no_run
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
            if let Some(l) = lower
                && bucket_upper <= l
            {
                continue;
            }
            if let Some(u) = upper
                && bucket_lower >= u
            {
                continue;
            }

            // Calculate the fraction of this bucket covered by the range
            let overlap = bucket.overlap_fraction(lower, upper);
            matching_rows += overlap * bucket.frequency as f64;
        }

        (matching_rows / self.total_rows as f64).clamp(0.0, 1.0)
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
/// Counts the number of top-level AND conjuncts in an expression.
///
/// For `(A AND B) AND (C AND D)` returns 4; for a single non-AND expression
/// returns 1. Used to estimate selectivity of multi-predicate join conditions.
fn count_and_conjuncts(expr: &LogicalExpression) -> usize {
    match expr {
        LogicalExpression::Binary {
            op: BinaryOp::And,
            left,
            right,
        } => count_and_conjuncts(left) + count_and_conjuncts(right),
        _ => 1,
    }
}

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

/// Configurable selectivity defaults for cardinality estimation.
///
/// Controls the assumed selectivity for various predicate types when
/// histogram or column statistics are unavailable. Adjusting these
/// values can improve plan quality for workloads with known skew.
#[derive(Debug, Clone)]
pub struct SelectivityConfig {
    /// Selectivity for unknown predicates (default: 0.1).
    pub default: f64,
    /// Selectivity for equality predicates without stats (default: 0.01).
    pub equality: f64,
    /// Selectivity for inequality predicates (default: 0.99).
    pub inequality: f64,
    /// Selectivity for range predicates without stats (default: 0.33).
    pub range: f64,
    /// Selectivity for string operations: STARTS WITH, ENDS WITH, CONTAINS, LIKE (default: 0.1).
    pub string_ops: f64,
    /// Selectivity for IN membership (default: 0.1).
    pub membership: f64,
    /// Selectivity for IS NULL (default: 0.05).
    pub is_null: f64,
    /// Selectivity for IS NOT NULL (default: 0.95).
    pub is_not_null: f64,
    /// Fraction assumed distinct for DISTINCT operations (default: 0.5).
    pub distinct_fraction: f64,
}

impl SelectivityConfig {
    /// Creates a new config with standard database defaults.
    #[must_use]
    pub fn new() -> Self {
        Self {
            default: 0.1,
            equality: 0.01,
            inequality: 0.99,
            range: 0.33,
            string_ops: 0.1,
            membership: 0.1,
            is_null: 0.05,
            is_not_null: 0.95,
            distinct_fraction: 0.5,
        }
    }
}

impl Default for SelectivityConfig {
    fn default() -> Self {
        Self::new()
    }
}

/// A single estimate-vs-actual observation for analysis.
#[derive(Debug, Clone)]
pub struct EstimationEntry {
    /// Human-readable label for the operator (e.g., "NodeScan(Person)").
    pub operator: String,
    /// The cardinality estimate produced by the optimizer.
    pub estimated: f64,
    /// The actual row count observed at execution time.
    pub actual: f64,
}

impl EstimationEntry {
    /// Returns the estimation error ratio (actual / estimated).
    ///
    /// Values near 1.0 indicate accurate estimates.
    /// Values > 1.0 indicate underestimation.
    /// Values < 1.0 indicate overestimation.
    #[must_use]
    pub fn error_ratio(&self) -> f64 {
        if self.estimated.abs() < f64::EPSILON {
            if self.actual.abs() < f64::EPSILON {
                1.0
            } else {
                f64::INFINITY
            }
        } else {
            self.actual / self.estimated
        }
    }
}

/// Collects estimate vs actual cardinality data for query plan analysis.
///
/// After executing a query, call [`record()`](Self::record) for each
/// operator with its estimated and actual cardinalities. Then use
/// [`should_replan()`](Self::should_replan) to decide whether the plan
/// should be re-optimized.
#[derive(Debug, Clone, Default)]
pub struct EstimationLog {
    /// Recorded entries.
    entries: Vec<EstimationEntry>,
    /// Error ratio threshold that triggers re-planning (default: 10.0).
    ///
    /// If any operator's error ratio exceeds this, `should_replan()` returns true.
    replan_threshold: f64,
}

impl EstimationLog {
    /// Creates a new estimation log with the given re-planning threshold.
    #[must_use]
    pub fn new(replan_threshold: f64) -> Self {
        Self {
            entries: Vec::new(),
            replan_threshold,
        }
    }

    /// Records an estimate-vs-actual observation.
    pub fn record(&mut self, operator: impl Into<String>, estimated: f64, actual: f64) {
        self.entries.push(EstimationEntry {
            operator: operator.into(),
            estimated,
            actual,
        });
    }

    /// Returns all recorded entries.
    #[must_use]
    pub fn entries(&self) -> &[EstimationEntry] {
        &self.entries
    }

    /// Returns whether any operator's estimation error exceeds the threshold,
    /// indicating the plan should be re-optimized.
    #[must_use]
    pub fn should_replan(&self) -> bool {
        self.entries.iter().any(|e| {
            let ratio = e.error_ratio();
            ratio > self.replan_threshold || ratio < 1.0 / self.replan_threshold
        })
    }

    /// Returns the maximum error ratio across all entries.
    #[must_use]
    pub fn max_error_ratio(&self) -> f64 {
        self.entries
            .iter()
            .map(|e| {
                let r = e.error_ratio();
                // Normalize so both over- and under-estimation are > 1.0
                if r < 1.0 { 1.0 / r } else { r }
            })
            .fold(1.0_f64, f64::max)
    }

    /// Clears all entries.
    pub fn clear(&mut self) {
        self.entries.clear();
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
    /// Configurable selectivity defaults.
    selectivity_config: SelectivityConfig,
    /// RDF statistics for triple pattern cardinality estimation.
    rdf_statistics: Option<grafeo_core::statistics::RdfStatistics>,
}

impl CardinalityEstimator {
    /// Creates a new cardinality estimator with default settings.
    #[must_use]
    pub fn new() -> Self {
        let config = SelectivityConfig::new();
        Self {
            table_stats: HashMap::new(),
            default_row_count: 1000,
            default_selectivity: config.default,
            avg_fanout: 10.0,
            selectivity_config: config,
            rdf_statistics: None,
        }
    }

    /// Creates a new cardinality estimator with custom selectivity configuration.
    #[must_use]
    pub fn with_selectivity_config(config: SelectivityConfig) -> Self {
        Self {
            table_stats: HashMap::new(),
            default_row_count: 1000,
            default_selectivity: config.default,
            avg_fanout: 10.0,
            selectivity_config: config,
            rdf_statistics: None,
        }
    }

    /// Returns the current selectivity configuration.
    #[must_use]
    pub fn selectivity_config(&self) -> &SelectivityConfig {
        &self.selectivity_config
    }

    /// Creates an estimation log with the default re-planning threshold (10x).
    #[must_use]
    pub fn create_estimation_log() -> EstimationLog {
        EstimationLog::new(10.0)
    }

    /// Creates a cardinality estimator pre-populated from store statistics.
    ///
    /// Maps `LabelStatistics` to `TableStats` and computes the average edge
    /// fanout from `EdgeTypeStatistics`. Falls back to defaults for any
    /// missing statistics.
    #[must_use]
    pub fn from_statistics(stats: &grafeo_core::statistics::Statistics) -> Self {
        let mut estimator = Self::new();

        // Use total node count as default for unlabeled scans
        if stats.total_nodes > 0 {
            estimator.default_row_count = stats.total_nodes;
        }

        // Convert label statistics to optimizer table stats
        for (label, label_stats) in &stats.labels {
            let mut table_stats = TableStats::new(label_stats.node_count);

            // Map property statistics (distinct count for selectivity estimation)
            for (prop, col_stats) in &label_stats.properties {
                let optimizer_col =
                    ColumnStats::new(col_stats.distinct_count).with_nulls(col_stats.null_count);
                table_stats = table_stats.with_column(prop, optimizer_col);
            }

            estimator.add_table_stats(label, table_stats);
        }

        // Compute average fanout from edge type statistics
        if !stats.edge_types.is_empty() {
            let total_out_degree: f64 = stats.edge_types.values().map(|e| e.avg_out_degree).sum();
            estimator.avg_fanout = total_out_degree / stats.edge_types.len() as f64;
        } else if stats.total_nodes > 0 {
            estimator.avg_fanout = stats.total_edges as f64 / stats.total_nodes as f64;
        }

        // Clamp fanout to a reasonable minimum
        if estimator.avg_fanout < 1.0 {
            estimator.avg_fanout = 1.0;
        }

        estimator
    }

    /// Creates a cardinality estimator from RDF statistics.
    ///
    /// Uses triple pattern cardinality estimates for `TripleScan` operators
    /// and join selectivity from per-predicate statistics.
    #[must_use]
    pub fn from_rdf_statistics(rdf_stats: grafeo_core::statistics::RdfStatistics) -> Self {
        let mut estimator = Self::new();
        if rdf_stats.total_triples > 0 {
            estimator.default_row_count = rdf_stats.total_triples;
        }
        estimator.rdf_statistics = Some(rdf_stats);
        estimator
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
            LogicalOperator::MultiWayJoin(mwj) => self.estimate_multi_way_join(mwj),
            LogicalOperator::LeftJoin(lj) => self.estimate_left_join(lj),
            LogicalOperator::TripleScan(scan) => self.estimate_triple_scan(scan),
            LogicalOperator::TextScan(scan) => self.estimate_text_scan(scan),
            _ => self.default_row_count as f64,
        }
    }

    /// Estimates node scan cardinality.
    fn estimate_node_scan(&self, scan: &NodeScanOp) -> f64 {
        if let Some(label) = &scan.label
            && let Some(stats) = self.table_stats.get(label)
        {
            return stats.row_count as f64;
        }
        // No label filter - scan all nodes
        self.default_row_count as f64
    }

    /// Estimates triple scan cardinality using RDF statistics.
    ///
    /// If RDF statistics are available, uses the pattern binding (which positions
    /// are bound vs variable) to produce accurate estimates. Otherwise falls back
    /// to the default row count.
    fn estimate_triple_scan(&self, scan: &TripleScanOp) -> f64 {
        // If there's an input, the triple scan is chained: multiply input cardinality
        // by the per-row expansion factor.
        let base = if let Some(ref input) = scan.input {
            self.estimate(input)
        } else {
            1.0
        };

        let Some(rdf_stats) = &self.rdf_statistics else {
            return if scan.input.is_some() {
                base * self.default_row_count as f64
            } else {
                self.default_row_count as f64
            };
        };

        let subject_bound = matches!(
            scan.subject,
            TripleComponent::Iri(_)
                | TripleComponent::Literal(_)
                | TripleComponent::LangLiteral { .. }
        );
        let object_bound = matches!(
            scan.object,
            TripleComponent::Iri(_)
                | TripleComponent::Literal(_)
                | TripleComponent::LangLiteral { .. }
        );
        let predicate_iri = match &scan.predicate {
            TripleComponent::Iri(iri) => Some(iri.as_str()),
            _ => None,
        };

        let pattern_card = rdf_stats.estimate_triple_pattern_cardinality(
            subject_bound,
            predicate_iri,
            object_bound,
        );

        if scan.input.is_some() {
            // Chained scan: each input row expands by the pattern's selectivity
            let selectivity = if rdf_stats.total_triples > 0 {
                pattern_card / rdf_stats.total_triples as f64
            } else {
                1.0
            };
            (base * pattern_card * selectivity).max(1.0)
        } else {
            pattern_card.max(1.0)
        }
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
        let fanout = if !expand.edge_types.is_empty() {
            // Specific edge type(s) typically have lower fanout
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
                    // reason: join condition count is always small (< 100)
                    #[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
                    let exp = join.conditions.len() as i32;
                    0.1_f64.powi(exp)
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

    /// Estimates left join cardinality (OPTIONAL MATCH).
    ///
    /// A left outer join preserves all left rows, so the output is at least
    /// `left_cardinality`. When the right side matches, the output may be
    /// larger (one left row can match multiple right rows).
    ///
    /// When the join carries a cross-side condition (null-safe predicates),
    /// each AND-conjunct reduces the selectivity estimate.
    fn estimate_left_join(&self, lj: &LeftJoinOp) -> f64 {
        let left_card = self.estimate(&lj.left);
        let right_card = self.estimate(&lj.right);

        // Adjust selectivity based on the number of AND conjuncts in the
        // cross-side condition: each equality reduces match probability.
        let condition_selectivity = if let Some(cond) = &lj.condition {
            let n = count_and_conjuncts(cond);
            // reason: conjunct count is always small (< 100)
            #[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
            let exp = n as i32;
            self.default_selectivity.powi(exp)
        } else {
            self.default_selectivity
        };

        // Estimate as inner join cardinality, but guaranteed at least left_card
        let inner_estimate = left_card * right_card * condition_selectivity;
        inner_estimate.max(left_card).max(1.0)
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
            // reason: group-by key count is always small (< 100)
            #[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
            let exp = agg.group_by.len() as i32;
            let group_reduction = 10.0_f64.powi(exp);
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
        (input_cardinality * self.selectivity_config.distinct_fraction).max(1.0)
    }

    /// Estimates limit cardinality.
    fn estimate_limit(&self, limit: &LimitOp) -> f64 {
        let input_cardinality = self.estimate(&limit.input);
        limit.count.estimate().min(input_cardinality)
    }

    /// Estimates skip cardinality.
    fn estimate_skip(&self, skip: &SkipOp) -> f64 {
        let input_cardinality = self.estimate(&skip.input);
        (input_cardinality - skip.count.estimate()).max(0.0)
    }

    /// Estimates vector scan cardinality.
    ///
    /// Vector scan returns at most k results (the k nearest neighbors).
    /// With similarity/distance filters, it may return fewer.
    fn estimate_vector_scan(&self, scan: &VectorScanOp) -> f64 {
        if let Some(k) = scan.k {
            // Top-k mode: at most k results, reduced by filter selectivity
            let selectivity = if scan.min_similarity.is_some() || scan.max_distance.is_some() {
                0.7 // Assume 70% of results pass threshold filters
            } else {
                1.0
            };
            (k as f64 * selectivity).max(1.0)
        } else {
            // Threshold-only mode: estimate 20% of indexed nodes match
            let base = scan
                .label
                .as_deref()
                .and_then(|l| self.table_stats.get(l))
                .map_or(self.default_row_count as f64, |s| s.row_count as f64);
            (base * 0.2).max(1.0)
        }
    }

    /// Estimates text scan cardinality using BM25 index.
    ///
    /// In top-k mode, returns at most k results. In threshold mode, assumes
    /// 10% of the label's documents match the query.
    fn estimate_text_scan(&self, scan: &TextScanOp) -> f64 {
        if let Some(k) = scan.k {
            // Top-k mode: at most k results
            return k as f64;
        }
        // Threshold mode: estimate 10% of indexed documents match
        let default_selectivity = 0.1;
        let base = if let Some(stats) = self.table_stats.get(&scan.label) {
            stats.row_count as f64
        } else {
            self.default_row_count as f64
        };
        (base * default_selectivity).max(1.0)
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

    /// Estimates multi-way join cardinality using the AGM bound heuristic.
    ///
    /// For a cyclic join of N relations, the AGM (Atserias-Grohe-Marx) bound
    /// gives min(cardinality)^(N/2) as a worst-case output size estimate.
    fn estimate_multi_way_join(&self, mwj: &MultiWayJoinOp) -> f64 {
        if mwj.inputs.is_empty() {
            return 0.0;
        }
        let cardinalities: Vec<f64> = mwj
            .inputs
            .iter()
            .map(|input| self.estimate(input))
            .collect();
        let min_card = cardinalities.iter().copied().fold(f64::INFINITY, f64::min);
        let n = cardinalities.len() as f64;
        // AGM bound: min(cardinality)^(n/2)
        (min_card.powf(n / 2.0)).max(1.0)
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
                self.selectivity_config.equality
            }
            // Inequality is very unselective
            BinaryOp::Ne => self.selectivity_config.inequality,
            // Range predicates - use histogram if available
            BinaryOp::Lt | BinaryOp::Le | BinaryOp::Gt | BinaryOp::Ge => {
                if let Some(selectivity) = self.try_range_selectivity(left, op, right) {
                    return selectivity;
                }
                self.selectivity_config.range
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
            BinaryOp::StartsWith | BinaryOp::EndsWith | BinaryOp::Contains | BinaryOp::Like => {
                self.selectivity_config.string_ops
            }
            // Collection membership
            BinaryOp::In => self.selectivity_config.membership,
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
            return Some((overlap / range).clamp(0.0, 1.0));
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
            if let Some(stats) = self.table_stats.get(label)
                && stats.columns.contains_key(&property)
            {
                return Some((label.clone(), property, value));
            }
        }

        // If no stats found but we have the property, return with variable as label
        Some((variable, property, value))
    }

    /// Estimates unary expression selectivity.
    fn estimate_unary_selectivity(&self, op: UnaryOp, _operand: &LogicalExpression) -> f64 {
        match op {
            UnaryOp::Not => 1.0 - self.default_selectivity,
            UnaryOp::IsNull => self.selectivity_config.is_null,
            UnaryOp::IsNotNull => self.selectivity_config.is_not_null,
            UnaryOp::Neg => 1.0, // Negation doesn't change cardinality
        }
    }

    /// Gets statistics for a column.
    fn get_column_stats(&self, label: &str, column: &str) -> Option<&ColumnStats> {
        self.table_stats.get(label)?.columns.get(column)
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
        DistinctOp, ExpandDirection, ExpandOp, FilterOp, JoinCondition, NodeScanOp, PathMode,
        ProjectOp, Projection, ReturnItem, ReturnOp, SkipOp, SortKey, SortOp, SortOrder,
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
            pushdown_hint: None,
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
            count: 10.into(),
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
            pushdown_hint: None,
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
            pushdown_hint: None,
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
            pushdown_hint: None,
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
            pushdown_hint: None,
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
            pushdown_hint: None,
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
            pushdown_hint: None,
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
            pushdown_hint: None,
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
            pushdown_hint: None,
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
            edge_types: vec![],
            min_hops: 1,
            max_hops: Some(1),
            input: Box::new(LogicalOperator::NodeScan(NodeScanOp {
                variable: "a".to_string(),
                label: Some("Person".to_string()),
                input: None,
            })),
            path_alias: None,
            path_mode: PathMode::Walk,
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
            edge_types: vec!["KNOWS".to_string()],
            min_hops: 1,
            max_hops: Some(1),
            input: Box::new(LogicalOperator::NodeScan(NodeScanOp {
                variable: "a".to_string(),
                label: Some("Person".to_string()),
                input: None,
            })),
            path_alias: None,
            path_mode: PathMode::Walk,
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
            edge_types: vec![],
            min_hops: 1,
            max_hops: Some(3),
            input: Box::new(LogicalOperator::NodeScan(NodeScanOp {
                variable: "a".to_string(),
                label: Some("Person".to_string()),
                input: None,
            })),
            path_alias: None,
            path_mode: PathMode::Walk,
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
            pass_through_input: false,
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
                nulls: None,
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
            count: 100.into(),
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
            edge_types: vec![],
            min_hops: 1,
            max_hops: Some(1),
            input: Box::new(LogicalOperator::NodeScan(NodeScanOp {
                variable: "a".to_string(),
                label: Some("Person".to_string()),
                input: None,
            })),
            path_alias: None,
            path_mode: PathMode::Walk,
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
            pushdown_hint: None,
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
            pushdown_hint: None,
        });

        let cardinality = estimator.estimate(&filter);

        // With 100 distinct values, selectivity should be ~1%
        // 1000 * 0.01 = 10
        assert!((1.0..50.0).contains(&cardinality));
    }

    #[test]
    fn test_histogram_min_max() {
        let values: Vec<f64> = vec![5.0, 10.0, 15.0, 20.0, 25.0];
        let histogram = EquiDepthHistogram::build(&values, 2);

        assert_eq!(histogram.min_value(), Some(5.0));
        // Max is the upper bound of the last bucket
        assert!(histogram.max_value().is_some());
    }

    // ==================== SelectivityConfig Tests ====================

    #[test]
    fn test_selectivity_config_defaults() {
        let config = SelectivityConfig::new();
        assert!((config.default - 0.1).abs() < f64::EPSILON);
        assert!((config.equality - 0.01).abs() < f64::EPSILON);
        assert!((config.inequality - 0.99).abs() < f64::EPSILON);
        assert!((config.range - 0.33).abs() < f64::EPSILON);
        assert!((config.string_ops - 0.1).abs() < f64::EPSILON);
        assert!((config.membership - 0.1).abs() < f64::EPSILON);
        assert!((config.is_null - 0.05).abs() < f64::EPSILON);
        assert!((config.is_not_null - 0.95).abs() < f64::EPSILON);
        assert!((config.distinct_fraction - 0.5).abs() < f64::EPSILON);
    }

    #[test]
    fn test_custom_selectivity_config() {
        let config = SelectivityConfig {
            equality: 0.05,
            range: 0.25,
            ..SelectivityConfig::new()
        };
        let estimator = CardinalityEstimator::with_selectivity_config(config);
        assert!((estimator.selectivity_config().equality - 0.05).abs() < f64::EPSILON);
        assert!((estimator.selectivity_config().range - 0.25).abs() < f64::EPSILON);
    }

    #[test]
    fn test_custom_selectivity_affects_estimation() {
        // Default: equality = 0.01 → 1000 * 0.01 = 10
        let mut default_est = CardinalityEstimator::new();
        default_est.add_table_stats("Person", TableStats::new(1000));

        let filter = LogicalOperator::Filter(FilterOp {
            predicate: LogicalExpression::Binary {
                left: Box::new(LogicalExpression::Property {
                    variable: "n".to_string(),
                    property: "name".to_string(),
                }),
                op: BinaryOp::Eq,
                right: Box::new(LogicalExpression::Literal(Value::String("Alix".into()))),
            },
            input: Box::new(LogicalOperator::NodeScan(NodeScanOp {
                variable: "n".to_string(),
                label: Some("Person".to_string()),
                input: None,
            })),
            pushdown_hint: None,
        });

        let default_card = default_est.estimate(&filter);

        // Custom: equality = 0.2 → 1000 * 0.2 = 200
        let config = SelectivityConfig {
            equality: 0.2,
            ..SelectivityConfig::new()
        };
        let mut custom_est = CardinalityEstimator::with_selectivity_config(config);
        custom_est.add_table_stats("Person", TableStats::new(1000));

        let custom_card = custom_est.estimate(&filter);

        assert!(custom_card > default_card);
        assert!((custom_card - 200.0).abs() < 1.0);
    }

    #[test]
    fn test_custom_range_selectivity() {
        let config = SelectivityConfig {
            range: 0.5,
            ..SelectivityConfig::new()
        };
        let mut estimator = CardinalityEstimator::with_selectivity_config(config);
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
            pushdown_hint: None,
        });

        let cardinality = estimator.estimate(&filter);
        // 1000 * 0.5 = 500
        assert!((cardinality - 500.0).abs() < 1.0);
    }

    #[test]
    fn test_custom_distinct_fraction() {
        let config = SelectivityConfig {
            distinct_fraction: 0.8,
            ..SelectivityConfig::new()
        };
        let mut estimator = CardinalityEstimator::with_selectivity_config(config);
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
        // 1000 * 0.8 = 800
        assert!((cardinality - 800.0).abs() < 1.0);
    }

    // ==================== EstimationLog Tests ====================

    #[test]
    fn test_estimation_log_basic() {
        let mut log = EstimationLog::new(10.0);
        log.record("NodeScan(Person)", 1000.0, 1200.0);
        log.record("Filter(age > 30)", 100.0, 90.0);

        assert_eq!(log.entries().len(), 2);
        assert!(!log.should_replan()); // 1.2x and 0.9x are within 10x threshold
    }

    #[test]
    fn test_estimation_log_triggers_replan() {
        let mut log = EstimationLog::new(10.0);
        log.record("NodeScan(Person)", 100.0, 5000.0); // 50x underestimate

        assert!(log.should_replan());
    }

    #[test]
    fn test_estimation_log_overestimate_triggers_replan() {
        let mut log = EstimationLog::new(5.0);
        log.record("Filter", 1000.0, 100.0); // 10x overestimate → ratio = 0.1

        assert!(log.should_replan()); // 0.1 < 1/5 = 0.2
    }

    #[test]
    fn test_estimation_entry_error_ratio() {
        let entry = EstimationEntry {
            operator: "test".into(),
            estimated: 100.0,
            actual: 200.0,
        };
        assert!((entry.error_ratio() - 2.0).abs() < f64::EPSILON);

        let perfect = EstimationEntry {
            operator: "test".into(),
            estimated: 100.0,
            actual: 100.0,
        };
        assert!((perfect.error_ratio() - 1.0).abs() < f64::EPSILON);

        let zero_est = EstimationEntry {
            operator: "test".into(),
            estimated: 0.0,
            actual: 0.0,
        };
        assert!((zero_est.error_ratio() - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_estimation_log_max_error_ratio() {
        let mut log = EstimationLog::new(10.0);
        log.record("A", 100.0, 300.0); // 3x
        log.record("B", 100.0, 50.0); // 2x (normalized: 1/0.5 = 2)
        log.record("C", 100.0, 100.0); // 1x

        assert!((log.max_error_ratio() - 3.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_estimation_log_clear() {
        let mut log = EstimationLog::new(10.0);
        log.record("A", 100.0, 100.0);
        assert_eq!(log.entries().len(), 1);

        log.clear();
        assert!(log.entries().is_empty());
        assert!(!log.should_replan());
    }

    #[test]
    fn test_create_estimation_log() {
        let log = CardinalityEstimator::create_estimation_log();
        assert!(log.entries().is_empty());
        assert!(!log.should_replan());
    }

    #[test]
    fn test_equality_selectivity_empty_histogram() {
        let hist = EquiDepthHistogram::new(vec![]);
        // Empty histogram returns fixed fallback
        assert_eq!(hist.equality_selectivity(5.0), 0.01);
    }

    #[test]
    fn test_equality_selectivity_value_in_bucket() {
        let values: Vec<f64> = (1..=10).map(|i| i as f64).collect();
        let hist = EquiDepthHistogram::build(&values, 2);
        let sel = hist.equality_selectivity(3.0);
        assert!(sel > 0.0);
        assert!(sel <= 1.0);
    }

    #[test]
    fn test_equality_selectivity_value_outside_all_buckets() {
        let values: Vec<f64> = (1..=10).map(|i| i as f64).collect();
        let hist = EquiDepthHistogram::build(&values, 2);
        // Value far outside range
        let sel = hist.equality_selectivity(9999.0);
        assert_eq!(sel, 0.001);
    }

    #[test]
    fn test_histogram_min_max_empty() {
        let hist = EquiDepthHistogram::new(vec![]);
        assert_eq!(hist.min_value(), None);
        assert_eq!(hist.max_value(), None);
    }

    #[test]
    fn test_histogram_min_max_single_bucket() {
        let hist = EquiDepthHistogram::new(vec![HistogramBucket::new(1.0, 10.0, 5, 5)]);
        assert_eq!(hist.min_value(), Some(1.0));
        assert_eq!(hist.max_value(), Some(10.0));
    }

    #[test]
    fn test_histogram_min_max_multi_bucket() {
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0, 10.0, 20.0];
        let hist = EquiDepthHistogram::build(&values, 3);
        let min = hist.min_value().unwrap();
        let max = hist.max_value().unwrap();
        assert!((min - 1.0).abs() < 1e-9, "min should be 1.0, got {min}");
        assert!(max >= 20.0, "max should be >= last value, got {max}");
    }

    #[test]
    fn test_count_and_conjuncts_single_expression() {
        use crate::query::plan::LogicalExpression;
        let expr = LogicalExpression::Literal(Value::Bool(true));
        assert_eq!(count_and_conjuncts(&expr), 1);
    }

    #[test]
    fn test_count_and_conjuncts_flat_and() {
        use crate::query::plan::{BinaryOp, LogicalExpression};
        let expr = LogicalExpression::Binary {
            left: Box::new(LogicalExpression::Literal(Value::Bool(true))),
            op: BinaryOp::And,
            right: Box::new(LogicalExpression::Literal(Value::Bool(false))),
        };
        assert_eq!(count_and_conjuncts(&expr), 2);
    }

    #[test]
    fn test_count_and_conjuncts_nested_and() {
        use crate::query::plan::{BinaryOp, LogicalExpression};
        let ab = LogicalExpression::Binary {
            left: Box::new(LogicalExpression::Literal(Value::Bool(true))),
            op: BinaryOp::And,
            right: Box::new(LogicalExpression::Literal(Value::Bool(false))),
        };
        let cd = LogicalExpression::Binary {
            left: Box::new(LogicalExpression::Literal(Value::Int64(1))),
            op: BinaryOp::And,
            right: Box::new(LogicalExpression::Literal(Value::Int64(2))),
        };
        let expr = LogicalExpression::Binary {
            left: Box::new(ab),
            op: BinaryOp::And,
            right: Box::new(cd),
        };
        assert_eq!(count_and_conjuncts(&expr), 4);
    }

    #[test]
    fn test_count_distinct_empty() {
        assert_eq!(count_distinct(&[]), 0);
    }

    #[test]
    fn test_count_distinct_all_unique() {
        assert_eq!(count_distinct(&[1.0, 2.0, 3.0, 4.0]), 4);
    }

    #[test]
    fn test_count_distinct_with_duplicates() {
        assert_eq!(count_distinct(&[1.0, 1.0, 2.0, 2.0, 3.0]), 3);
    }

    #[test]
    fn test_count_distinct_all_same() {
        assert_eq!(count_distinct(&[5.0, 5.0, 5.0]), 1);
    }

    #[test]
    fn test_count_distinct_single_value() {
        assert_eq!(count_distinct(&[42.0]), 1);
    }
}
