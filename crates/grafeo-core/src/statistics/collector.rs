//! Collecting and storing graph statistics.
//!
//! Use [`StatisticsCollector`] to stream values through and build statistics,
//! or construct [`ColumnStatistics`] directly if you already know the numbers.
//! The [`Statistics`] struct holds everything the optimizer needs.

use super::histogram::Histogram;
use grafeo_common::types::Value;
use std::collections::HashMap;

/// A property key identifier.
pub type PropertyKey = String;

/// Everything the optimizer knows about the data - cardinalities, distributions, degrees.
///
/// This is the main struct the query planner consults when choosing between
/// different execution strategies.
#[derive(Debug, Clone, Default)]
pub struct Statistics {
    /// Per-label statistics.
    pub labels: HashMap<String, LabelStatistics>,
    /// Per-edge-type statistics.
    pub edge_types: HashMap<String, EdgeTypeStatistics>,
    /// Per-property statistics (across all labels).
    pub properties: HashMap<PropertyKey, ColumnStatistics>,
    /// Total node count.
    pub total_nodes: u64,
    /// Total edge count.
    pub total_edges: u64,
}

impl Statistics {
    /// Creates a new empty statistics object.
    pub fn new() -> Self {
        Self::default()
    }

    /// Updates label statistics.
    pub fn update_label(&mut self, label: &str, stats: LabelStatistics) {
        self.labels.insert(label.to_string(), stats);
    }

    /// Updates edge type statistics.
    pub fn update_edge_type(&mut self, edge_type: &str, stats: EdgeTypeStatistics) {
        self.edge_types.insert(edge_type.to_string(), stats);
    }

    /// Updates property statistics.
    pub fn update_property(&mut self, property: &str, stats: ColumnStatistics) {
        self.properties.insert(property.to_string(), stats);
    }

    /// Gets label statistics.
    pub fn get_label(&self, label: &str) -> Option<&LabelStatistics> {
        self.labels.get(label)
    }

    /// Gets edge type statistics.
    pub fn get_edge_type(&self, edge_type: &str) -> Option<&EdgeTypeStatistics> {
        self.edge_types.get(edge_type)
    }

    /// Gets property statistics.
    pub fn get_property(&self, property: &str) -> Option<&ColumnStatistics> {
        self.properties.get(property)
    }

    /// Estimates the cardinality of a label scan.
    pub fn estimate_label_cardinality(&self, label: &str) -> f64 {
        self.labels
            .get(label)
            .map(|s| s.node_count as f64)
            .unwrap_or(1000.0) // Default estimate if no statistics
    }

    /// Estimates the average degree for an edge type.
    pub fn estimate_avg_degree(&self, edge_type: &str, outgoing: bool) -> f64 {
        self.edge_types
            .get(edge_type)
            .map(|s| {
                if outgoing {
                    s.avg_out_degree
                } else {
                    s.avg_in_degree
                }
            })
            .unwrap_or(10.0) // Default estimate
    }

    /// Estimates selectivity of an equality predicate.
    pub fn estimate_equality_selectivity(&self, property: &str, _value: &Value) -> f64 {
        self.properties
            .get(property)
            .map(|s| {
                if s.distinct_count > 0 {
                    1.0 / s.distinct_count as f64
                } else {
                    0.5
                }
            })
            .unwrap_or(0.5)
    }

    /// Estimates selectivity of a range predicate.
    pub fn estimate_range_selectivity(
        &self,
        property: &str,
        lower: Option<&Value>,
        upper: Option<&Value>,
    ) -> f64 {
        self.properties
            .get(property)
            .and_then(|s| s.histogram.as_ref())
            .map(|h| h.estimate_range_selectivity(lower, upper, true, true))
            .unwrap_or(0.33) // Default for range predicates
    }
}

/// Statistics for nodes with a particular label (like "Person" or "Company").
#[derive(Debug, Clone)]
pub struct LabelStatistics {
    /// Number of nodes with this label.
    pub node_count: u64,
    /// Average outgoing degree.
    pub avg_out_degree: f64,
    /// Average incoming degree.
    pub avg_in_degree: f64,
    /// Per-property statistics for nodes with this label.
    pub properties: HashMap<PropertyKey, ColumnStatistics>,
}

impl LabelStatistics {
    /// Creates new label statistics.
    pub fn new(node_count: u64) -> Self {
        Self {
            node_count,
            avg_out_degree: 0.0,
            avg_in_degree: 0.0,
            properties: HashMap::new(),
        }
    }

    /// Sets the average degrees.
    pub fn with_degrees(mut self, out_degree: f64, in_degree: f64) -> Self {
        self.avg_out_degree = out_degree;
        self.avg_in_degree = in_degree;
        self
    }

    /// Adds property statistics.
    pub fn with_property(mut self, property: &str, stats: ColumnStatistics) -> Self {
        self.properties.insert(property.to_string(), stats);
        self
    }
}

/// Alias for table statistics (used in relational contexts).
pub type TableStatistics = LabelStatistics;

/// Statistics for edges of a particular type (like "KNOWS" or "WORKS_AT").
#[derive(Debug, Clone)]
pub struct EdgeTypeStatistics {
    /// Number of edges of this type.
    pub edge_count: u64,
    /// Average outgoing degree (edges per source node).
    pub avg_out_degree: f64,
    /// Average incoming degree (edges per target node).
    pub avg_in_degree: f64,
    /// Per-property statistics for edges of this type.
    pub properties: HashMap<PropertyKey, ColumnStatistics>,
}

impl EdgeTypeStatistics {
    /// Creates new edge type statistics.
    pub fn new(edge_count: u64, avg_out_degree: f64, avg_in_degree: f64) -> Self {
        Self {
            edge_count,
            avg_out_degree,
            avg_in_degree,
            properties: HashMap::new(),
        }
    }

    /// Adds property statistics.
    pub fn with_property(mut self, property: &str, stats: ColumnStatistics) -> Self {
        self.properties.insert(property.to_string(), stats);
        self
    }
}

/// Detailed statistics about a property's values - min, max, histogram, null ratio.
#[derive(Debug, Clone)]
pub struct ColumnStatistics {
    /// Number of distinct values.
    pub distinct_count: u64,
    /// Total number of values (including nulls).
    pub total_count: u64,
    /// Number of null values.
    pub null_count: u64,
    /// Minimum value (if applicable).
    pub min_value: Option<Value>,
    /// Maximum value (if applicable).
    pub max_value: Option<Value>,
    /// Average value (for numeric types).
    pub avg_value: Option<f64>,
    /// Equi-depth histogram (for selectivity estimation).
    pub histogram: Option<Histogram>,
    /// Most common values with their frequencies.
    pub most_common: Vec<(Value, f64)>,
}

impl ColumnStatistics {
    /// Creates new column statistics with basic info.
    pub fn new(distinct_count: u64, total_count: u64, null_count: u64) -> Self {
        Self {
            distinct_count,
            total_count,
            null_count,
            min_value: None,
            max_value: None,
            avg_value: None,
            histogram: None,
            most_common: Vec::new(),
        }
    }

    /// Sets min/max values.
    pub fn with_min_max(mut self, min: Value, max: Value) -> Self {
        self.min_value = Some(min);
        self.max_value = Some(max);
        self
    }

    /// Sets the average value.
    pub fn with_avg(mut self, avg: f64) -> Self {
        self.avg_value = Some(avg);
        self
    }

    /// Sets the histogram.
    pub fn with_histogram(mut self, histogram: Histogram) -> Self {
        self.histogram = Some(histogram);
        self
    }

    /// Sets the most common values.
    pub fn with_most_common(mut self, values: Vec<(Value, f64)>) -> Self {
        self.most_common = values;
        self
    }

    /// Returns the null fraction.
    pub fn null_fraction(&self) -> f64 {
        if self.total_count == 0 {
            0.0
        } else {
            self.null_count as f64 / self.total_count as f64
        }
    }

    /// Estimates selectivity for an equality predicate.
    pub fn estimate_equality_selectivity(&self, value: &Value) -> f64 {
        // Check most common values first
        for (mcv, freq) in &self.most_common {
            if mcv == value {
                return *freq;
            }
        }

        // Use histogram if available
        if let Some(ref hist) = self.histogram {
            return hist.estimate_equality_selectivity(value);
        }

        // Fall back to uniform distribution assumption
        if self.distinct_count > 0 {
            1.0 / self.distinct_count as f64
        } else {
            0.0
        }
    }

    /// Estimates selectivity for a range predicate.
    pub fn estimate_range_selectivity(&self, lower: Option<&Value>, upper: Option<&Value>) -> f64 {
        if let Some(ref hist) = self.histogram {
            return hist.estimate_range_selectivity(lower, upper, true, true);
        }

        // Without histogram, use min/max if available
        match (&self.min_value, &self.max_value, lower, upper) {
            (Some(min), Some(max), Some(l), Some(u)) => {
                // Linear interpolation
                estimate_linear_range(min, max, l, u)
            }
            (Some(_), Some(_), Some(_), None) => 0.5, // Greater than
            (Some(_), Some(_), None, Some(_)) => 0.5, // Less than
            _ => 0.33,                                // Default
        }
    }
}

/// Estimates range selectivity using linear interpolation.
fn estimate_linear_range(min: &Value, max: &Value, lower: &Value, upper: &Value) -> f64 {
    match (min, max, lower, upper) {
        (
            Value::Int64(min_v),
            Value::Int64(max_v),
            Value::Int64(lower_v),
            Value::Int64(upper_v),
        ) => {
            let total_range = (max_v - min_v) as f64;
            if total_range <= 0.0 {
                return 1.0;
            }

            let effective_lower = (*lower_v).max(*min_v);
            let effective_upper = (*upper_v).min(*max_v);

            if effective_upper < effective_lower {
                return 0.0;
            }

            (effective_upper - effective_lower) as f64 / total_range
        }
        (
            Value::Float64(min_v),
            Value::Float64(max_v),
            Value::Float64(lower_v),
            Value::Float64(upper_v),
        ) => {
            let total_range = max_v - min_v;
            if total_range <= 0.0 {
                return 1.0;
            }

            let effective_lower = lower_v.max(*min_v);
            let effective_upper = upper_v.min(*max_v);

            if effective_upper < effective_lower {
                return 0.0;
            }

            (effective_upper - effective_lower) / total_range
        }
        _ => 0.33,
    }
}

/// Streams values through to build statistics automatically.
///
/// Call [`add()`](Self::add) for each value, then [`build()`](Self::build)
/// to get the final [`ColumnStatistics`] with histogram and most common values.
#[allow(dead_code)] // Used in tests and future cardinality estimation
pub struct StatisticsCollector {
    /// Values collected for histogram building.
    values: Vec<Value>,
    /// Distinct value tracker.
    distinct: std::collections::HashSet<String>,
    /// Running min.
    min: Option<Value>,
    /// Running max.
    max: Option<Value>,
    /// Running sum (for numeric).
    sum: f64,
    /// Null count.
    null_count: u64,
    /// Value frequency counter.
    frequencies: HashMap<String, u64>,
}

#[allow(dead_code)] // Used in tests and future cardinality estimation
impl StatisticsCollector {
    /// Creates a new statistics collector.
    pub fn new() -> Self {
        Self {
            values: Vec::new(),
            distinct: std::collections::HashSet::new(),
            min: None,
            max: None,
            sum: 0.0,
            null_count: 0,
            frequencies: HashMap::new(),
        }
    }

    /// Adds a value to the collector.
    pub fn add(&mut self, value: Value) {
        if matches!(value, Value::Null) {
            self.null_count += 1;
            return;
        }

        // Track distinct values
        let key = format!("{value:?}");
        self.distinct.insert(key.clone());

        // Track frequencies
        *self.frequencies.entry(key).or_insert(0) += 1;

        // Track min/max
        self.update_min_max(&value);

        // Track sum for numeric
        if let Some(v) = value_to_f64(&value) {
            self.sum += v;
        }

        self.values.push(value);
    }

    fn update_min_max(&mut self, value: &Value) {
        // Update min
        match &self.min {
            None => self.min = Some(value.clone()),
            Some(current) => {
                if compare_values(value, current) == Some(std::cmp::Ordering::Less) {
                    self.min = Some(value.clone());
                }
            }
        }

        // Update max
        match &self.max {
            None => self.max = Some(value.clone()),
            Some(current) => {
                if compare_values(value, current) == Some(std::cmp::Ordering::Greater) {
                    self.max = Some(value.clone());
                }
            }
        }
    }

    /// Builds column statistics from collected data.
    pub fn build(mut self, num_histogram_buckets: usize, num_mcv: usize) -> ColumnStatistics {
        let total_count = self.values.len() as u64 + self.null_count;
        let distinct_count = self.distinct.len() as u64;

        let avg = if !self.values.is_empty() {
            Some(self.sum / self.values.len() as f64)
        } else {
            None
        };

        // Build histogram
        self.values
            .sort_by(|a, b| compare_values(a, b).unwrap_or(std::cmp::Ordering::Equal));
        let histogram = if self.values.len() >= num_histogram_buckets {
            Some(Histogram::build(&self.values, num_histogram_buckets))
        } else {
            None
        };

        // Find most common values
        let total_non_null = self.values.len() as f64;
        let mut freq_vec: Vec<_> = self.frequencies.into_iter().collect();
        freq_vec.sort_by(|a, b| b.1.cmp(&a.1));

        let most_common: Vec<(Value, f64)> = freq_vec
            .into_iter()
            .take(num_mcv)
            .filter_map(|(key, count)| {
                // Try to parse the key back to a value (simplified)
                let freq = count as f64 / total_non_null;
                // This is a simplification - we'd need to store actual values
                if key.starts_with("Int64(") {
                    let num_str = key.trim_start_matches("Int64(").trim_end_matches(')');
                    num_str.parse::<i64>().ok().map(|n| (Value::Int64(n), freq))
                } else if key.starts_with("String(") {
                    let s = key
                        .trim_start_matches("String(Arc(\"")
                        .trim_end_matches("\"))");
                    Some((Value::String(s.to_string().into()), freq))
                } else {
                    None
                }
            })
            .collect();

        let mut stats = ColumnStatistics::new(distinct_count, total_count, self.null_count);

        if let Some(min) = self.min
            && let Some(max) = self.max
        {
            stats = stats.with_min_max(min, max);
        }

        if let Some(avg) = avg {
            stats = stats.with_avg(avg);
        }

        if let Some(hist) = histogram {
            stats = stats.with_histogram(hist);
        }

        if !most_common.is_empty() {
            stats = stats.with_most_common(most_common);
        }

        stats
    }
}

impl Default for StatisticsCollector {
    fn default() -> Self {
        Self::new()
    }
}

/// Converts a value to f64.
#[allow(dead_code)] // Used by StatisticsCollector
fn value_to_f64(value: &Value) -> Option<f64> {
    match value {
        Value::Int64(i) => Some(*i as f64),
        Value::Float64(f) => Some(*f),
        _ => None,
    }
}

/// Compares two values.
#[allow(dead_code)] // Used by StatisticsCollector
fn compare_values(a: &Value, b: &Value) -> Option<std::cmp::Ordering> {
    match (a, b) {
        (Value::Int64(a), Value::Int64(b)) => Some(a.cmp(b)),
        (Value::Float64(a), Value::Float64(b)) => a.partial_cmp(b),
        (Value::String(a), Value::String(b)) => Some(a.cmp(b)),
        (Value::Bool(a), Value::Bool(b)) => Some(a.cmp(b)),
        (Value::Int64(a), Value::Float64(b)) => (*a as f64).partial_cmp(b),
        (Value::Float64(a), Value::Int64(b)) => a.partial_cmp(&(*b as f64)),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_statistics_collector() {
        let mut collector = StatisticsCollector::new();

        for i in 0..100 {
            collector.add(Value::Int64(i % 10)); // Values 0-9, each appearing 10 times
        }

        let stats = collector.build(10, 5);

        assert_eq!(stats.distinct_count, 10);
        assert_eq!(stats.total_count, 100);
        assert_eq!(stats.null_count, 0);
        assert_eq!(stats.min_value, Some(Value::Int64(0)));
        assert_eq!(stats.max_value, Some(Value::Int64(9)));
    }

    #[test]
    fn test_statistics_with_nulls() {
        let mut collector = StatisticsCollector::new();

        collector.add(Value::Int64(1));
        collector.add(Value::Null);
        collector.add(Value::Int64(2));
        collector.add(Value::Null);
        collector.add(Value::Int64(3));

        let stats = collector.build(5, 3);

        assert_eq!(stats.total_count, 5);
        assert_eq!(stats.null_count, 2);
        assert_eq!(stats.distinct_count, 3);
        assert!((stats.null_fraction() - 0.4).abs() < 0.01);
    }

    #[test]
    fn test_label_statistics() {
        let stats = LabelStatistics::new(1000)
            .with_degrees(5.0, 3.0)
            .with_property(
                "age",
                ColumnStatistics::new(50, 1000, 10)
                    .with_min_max(Value::Int64(0), Value::Int64(100)),
            );

        assert_eq!(stats.node_count, 1000);
        assert_eq!(stats.avg_out_degree, 5.0);
        assert!(stats.properties.contains_key("age"));
    }

    #[test]
    fn test_database_statistics() {
        let mut db_stats = Statistics::new();

        db_stats.update_label(
            "Person",
            LabelStatistics::new(10000).with_degrees(10.0, 10.0),
        );

        db_stats.update_edge_type("KNOWS", EdgeTypeStatistics::new(50000, 5.0, 5.0));

        assert_eq!(db_stats.estimate_label_cardinality("Person"), 10000.0);
        assert_eq!(db_stats.estimate_label_cardinality("Unknown"), 1000.0); // Default

        assert_eq!(db_stats.estimate_avg_degree("KNOWS", true), 5.0);
    }
}
