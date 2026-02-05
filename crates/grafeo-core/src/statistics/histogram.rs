//! Equi-depth histograms for understanding value distributions.
//!
//! When the optimizer sees `WHERE age > 30`, it needs to know what fraction of
//! rows match. Histograms split the value range into buckets of roughly equal
//! row counts, letting us estimate selectivity without scanning the data.

use grafeo_common::types::Value;
use std::cmp::Ordering;

/// One slice of the value distribution - a range with its row count.
#[derive(Debug, Clone)]
pub struct HistogramBucket {
    /// Lower bound (inclusive).
    pub lower: Value,
    /// Upper bound (inclusive).
    pub upper: Value,
    /// Number of distinct values in this bucket.
    pub distinct_count: u64,
    /// Number of values in this bucket.
    pub row_count: u64,
}

impl HistogramBucket {
    /// Creates a new histogram bucket.
    pub fn new(lower: Value, upper: Value, distinct_count: u64, row_count: u64) -> Self {
        Self {
            lower,
            upper,
            distinct_count,
            row_count,
        }
    }

    /// Checks if a value falls within this bucket.
    pub fn contains(&self, value: &Value) -> bool {
        compare_values(value, &self.lower) != Some(Ordering::Less)
            && compare_values(value, &self.upper) != Some(Ordering::Greater)
    }
}

/// Divides a column's value range into buckets of roughly equal row counts.
///
/// Build one with [`build()`](Self::build) from sorted values, then use
/// [`estimate_equality_selectivity()`](Self::estimate_equality_selectivity)
/// or [`estimate_range_selectivity()`](Self::estimate_range_selectivity)
/// to predict how many rows will match a predicate.
#[derive(Debug, Clone)]
pub struct Histogram {
    /// Histogram buckets, sorted by lower bound.
    buckets: Vec<HistogramBucket>,
    /// Total number of rows represented.
    total_rows: u64,
    /// Total number of distinct values.
    total_distinct: u64,
}

impl Histogram {
    /// Creates a new histogram with the given buckets.
    pub fn new(buckets: Vec<HistogramBucket>) -> Self {
        let total_rows = buckets.iter().map(|b| b.row_count).sum();
        let total_distinct = buckets.iter().map(|b| b.distinct_count).sum();

        Self {
            buckets,
            total_rows,
            total_distinct,
        }
    }

    /// Creates an equi-depth histogram from sorted values.
    ///
    /// # Arguments
    /// * `sorted_values` - Values sorted in ascending order.
    /// * `num_buckets` - Target number of buckets.
    pub fn build(sorted_values: &[Value], num_buckets: usize) -> Self {
        if sorted_values.is_empty() {
            return Self::new(Vec::new());
        }

        let num_buckets = num_buckets.max(1).min(sorted_values.len());
        let rows_per_bucket = sorted_values.len() / num_buckets;

        let mut buckets = Vec::with_capacity(num_buckets);
        let mut current_start = 0;

        for i in 0..num_buckets {
            let end = if i == num_buckets - 1 {
                sorted_values.len()
            } else {
                current_start + rows_per_bucket
            };

            if current_start >= sorted_values.len() {
                break;
            }

            let bucket_values = &sorted_values[current_start..end];
            // Invariant: slice is non-empty because:
            // - current_start < sorted_values.len() (guard on line 89)
            // - end > current_start (rows_per_bucket >= 1 or end = len > current_start)
            let lower = bucket_values
                .first()
                .expect("bucket_values is non-empty: current_start < end")
                .clone();
            let upper = bucket_values
                .last()
                .expect("bucket_values is non-empty: current_start < end")
                .clone();

            // Count distinct values in bucket
            let mut distinct = 1u64;
            for j in 1..bucket_values.len() {
                if bucket_values[j] != bucket_values[j - 1] {
                    distinct += 1;
                }
            }

            buckets.push(HistogramBucket::new(
                lower,
                upper,
                distinct,
                bucket_values.len() as u64,
            ));

            current_start = end;
        }

        Self::new(buckets)
    }

    /// Returns the number of buckets.
    pub fn bucket_count(&self) -> usize {
        self.buckets.len()
    }

    /// Returns the buckets.
    pub fn buckets(&self) -> &[HistogramBucket] {
        &self.buckets
    }

    /// Returns the total row count.
    pub fn total_rows(&self) -> u64 {
        self.total_rows
    }

    /// Returns the total distinct count.
    pub fn total_distinct(&self) -> u64 {
        self.total_distinct
    }

    /// Estimates the selectivity of an equality predicate.
    ///
    /// Returns the estimated fraction of rows that match the value.
    pub fn estimate_equality_selectivity(&self, value: &Value) -> f64 {
        if self.total_rows == 0 {
            return 0.0;
        }

        // Find the bucket containing the value
        for bucket in &self.buckets {
            if bucket.contains(value) {
                // Assume uniform distribution within bucket
                if bucket.distinct_count == 0 {
                    return 0.0;
                }
                return (bucket.row_count as f64 / bucket.distinct_count as f64)
                    / self.total_rows as f64;
            }
        }

        // Value not in histogram - assume very low selectivity
        1.0 / self.total_rows as f64
    }

    /// Estimates the selectivity of a range predicate.
    ///
    /// # Arguments
    /// * `lower` - Lower bound (None for unbounded).
    /// * `upper` - Upper bound (None for unbounded).
    /// * `lower_inclusive` - Whether lower bound is inclusive.
    /// * `upper_inclusive` - Whether upper bound is inclusive.
    pub fn estimate_range_selectivity(
        &self,
        lower: Option<&Value>,
        upper: Option<&Value>,
        lower_inclusive: bool,
        upper_inclusive: bool,
    ) -> f64 {
        if self.total_rows == 0 {
            return 0.0;
        }

        let mut matching_rows = 0.0;

        for bucket in &self.buckets {
            // Check if bucket overlaps with range
            let bucket_in_range = match (lower, upper) {
                (None, None) => true,
                (Some(l), None) => compare_values(&bucket.upper, l) != Some(Ordering::Less),
                (None, Some(u)) => compare_values(&bucket.lower, u) != Some(Ordering::Greater),
                (Some(l), Some(u)) => {
                    compare_values(&bucket.upper, l) != Some(Ordering::Less)
                        && compare_values(&bucket.lower, u) != Some(Ordering::Greater)
                }
            };

            if !bucket_in_range {
                continue;
            }

            // Estimate the fraction of the bucket that's in range
            let bucket_fraction = estimate_bucket_overlap(
                &bucket.lower,
                &bucket.upper,
                lower,
                upper,
                lower_inclusive,
                upper_inclusive,
            );

            matching_rows += bucket.row_count as f64 * bucket_fraction;
        }

        matching_rows / self.total_rows as f64
    }

    /// Estimates the selectivity of a less-than predicate.
    pub fn estimate_less_than_selectivity(&self, value: &Value, inclusive: bool) -> f64 {
        self.estimate_range_selectivity(None, Some(value), true, inclusive)
    }

    /// Estimates the selectivity of a greater-than predicate.
    pub fn estimate_greater_than_selectivity(&self, value: &Value, inclusive: bool) -> f64 {
        self.estimate_range_selectivity(Some(value), None, inclusive, true)
    }
}

/// Estimates the fraction of a bucket that overlaps with a range.
fn estimate_bucket_overlap(
    bucket_lower: &Value,
    bucket_upper: &Value,
    range_lower: Option<&Value>,
    range_upper: Option<&Value>,
    _lower_inclusive: bool,
    _upper_inclusive: bool,
) -> f64 {
    // For simplicity, use linear interpolation based on value positions
    // This is a rough estimate but works well for numeric values

    // If range fully contains bucket, return 1.0
    let range_contains_lower = range_lower
        .map(|l| compare_values(bucket_lower, l) != Some(Ordering::Less))
        .unwrap_or(true);
    let range_contains_upper = range_upper
        .map(|u| compare_values(bucket_upper, u) != Some(Ordering::Greater))
        .unwrap_or(true);

    if range_contains_lower && range_contains_upper {
        return 1.0;
    }

    // For partial overlap, estimate fraction
    // This is simplified - a real implementation would use numeric interpolation
    match (bucket_lower, bucket_upper) {
        (Value::Int64(bl), Value::Int64(bu)) => {
            let bucket_range = (bu - bl) as f64;
            if bucket_range <= 0.0 {
                return 1.0;
            }

            let effective_lower = range_lower
                .and_then(|l| {
                    if let Value::Int64(li) = l {
                        Some(*li)
                    } else {
                        None
                    }
                })
                .unwrap_or(*bl);

            let effective_upper = range_upper
                .and_then(|u| {
                    if let Value::Int64(ui) = u {
                        Some(*ui)
                    } else {
                        None
                    }
                })
                .unwrap_or(*bu);

            let overlap_lower = effective_lower.max(*bl);
            let overlap_upper = effective_upper.min(*bu);

            if overlap_upper < overlap_lower {
                return 0.0;
            }

            (overlap_upper - overlap_lower) as f64 / bucket_range
        }
        (Value::Float64(bl), Value::Float64(bu)) => {
            let bucket_range = bu - bl;
            if bucket_range <= 0.0 {
                return 1.0;
            }

            let effective_lower = range_lower
                .and_then(|l| {
                    if let Value::Float64(li) = l {
                        Some(*li)
                    } else {
                        None
                    }
                })
                .unwrap_or(*bl);

            let effective_upper = range_upper
                .and_then(|u| {
                    if let Value::Float64(ui) = u {
                        Some(*ui)
                    } else {
                        None
                    }
                })
                .unwrap_or(*bu);

            let overlap_lower = effective_lower.max(*bl);
            let overlap_upper = effective_upper.min(*bu);

            if overlap_upper < overlap_lower {
                return 0.0;
            }

            (overlap_upper - overlap_lower) / bucket_range
        }
        _ => {
            // For non-numeric types, assume 0.5 for partial overlap
            0.5
        }
    }
}

/// Compares two values.
fn compare_values(a: &Value, b: &Value) -> Option<Ordering> {
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

    fn create_int_values(values: &[i64]) -> Vec<Value> {
        values.iter().map(|&v| Value::Int64(v)).collect()
    }

    #[test]
    fn test_histogram_build() {
        let values = create_int_values(&[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
        let hist = Histogram::build(&values, 2);

        assert_eq!(hist.bucket_count(), 2);
        assert_eq!(hist.total_rows(), 10);
    }

    #[test]
    fn test_equality_selectivity() {
        let values = create_int_values(&[1, 1, 2, 2, 2, 3, 3, 3, 3, 4]);
        let hist = Histogram::build(&values, 4);

        // Value 3 appears 4 times out of 10
        let sel = hist.estimate_equality_selectivity(&Value::Int64(3));
        assert!(sel > 0.0 && sel < 1.0);
    }

    #[test]
    fn test_range_selectivity() {
        let values = create_int_values(&[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
        let hist = Histogram::build(&values, 5);

        // Range 1-5 should be about 50%
        let sel = hist.estimate_range_selectivity(
            Some(&Value::Int64(1)),
            Some(&Value::Int64(5)),
            true,
            true,
        );
        assert!(sel >= 0.3 && sel <= 0.7);

        // Range 1-10 should be 100%
        let sel_full = hist.estimate_range_selectivity(
            Some(&Value::Int64(1)),
            Some(&Value::Int64(10)),
            true,
            true,
        );
        assert!((sel_full - 1.0).abs() < 0.1);
    }

    #[test]
    fn test_less_than_selectivity() {
        let values = create_int_values(&[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
        let hist = Histogram::build(&values, 5);

        // Less than 5 should be about 40%
        let sel = hist.estimate_less_than_selectivity(&Value::Int64(5), false);
        assert!(sel > 0.0 && sel < 1.0);
    }

    #[test]
    fn test_empty_histogram() {
        let hist = Histogram::build(&[], 5);

        assert_eq!(hist.bucket_count(), 0);
        assert_eq!(hist.total_rows(), 0);
        assert_eq!(hist.estimate_equality_selectivity(&Value::Int64(5)), 0.0);
    }
}
