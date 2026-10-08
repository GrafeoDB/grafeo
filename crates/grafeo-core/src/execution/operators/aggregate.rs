//! Aggregation operators for GROUP BY and aggregation functions.
//!
//! This module provides:
//! - [`HashAggregateOperator`]: Hash-based grouping with aggregation functions
//! - [`SimpleAggregateOperator`]: Global aggregation without GROUP BY
//!
//! Shared types ([`AggregateFunction`], [`AggregateExpr`], [`HashableValue`]) live in
//! the [`super::accumulator`] module.

use grafeo_common::types::{HashableValue as GroupValueKey, LogicalType, Value};
use indexmap::IndexMap;
use std::collections::HashSet;

use super::accumulator::{AggregateExpr, AggregateFunction, HashableValue};
use super::{Operator, OperatorError, OperatorResult};
use crate::execution::DataChunk;
use crate::execution::chunk::{ColumnTypes, DataChunkBuilder, copied_column_type};

/// State for a single aggregation computation.
///
/// Used by both the pull-based [`HashAggregateOperator`] and the push-based
/// `AggregatePushOperator`.
/// Supports all [`AggregateFunction`] variants including Welford's algorithm
/// for online statistics, Kahan summation, distinct tracking, and bivariate
/// regression functions.
#[derive(Debug, Clone)]
#[allow(missing_docs)]
pub enum AggregateState {
    /// Count state.
    Count(i64),
    /// Count distinct state (count, seen values).
    CountDistinct(i64, HashSet<HashableValue>),
    /// Sum state (integer sum, count of values added).
    SumInt(i64, i64),
    /// Sum distinct state (integer sum, count, seen values).
    SumIntDistinct(i64, i64, HashSet<HashableValue>),
    /// Sum state (float sum, count of values added).
    SumFloat(f64, f64, i64),
    /// Sum distinct state (float sum, compensation, count, seen values).
    SumFloatDistinct(f64, f64, i64, HashSet<HashableValue>),
    /// Average state (sum, count).
    Avg(f64, i64),
    /// Average distinct state (sum, count, seen values).
    AvgDistinct(f64, i64, HashSet<HashableValue>),
    /// Min state.
    Min(Option<Value>),
    /// Max state.
    Max(Option<Value>),
    /// First state.
    First(Option<Value>),
    /// Last state.
    Last(Option<Value>),
    /// Collect state.
    Collect(Vec<Value>),
    /// Collect distinct state (values, seen).
    CollectDistinct(Vec<Value>, HashSet<HashableValue>),
    /// Sample standard deviation state using Welford's algorithm (count, mean, M2).
    StdDev { count: i64, mean: f64, m2: f64 },
    /// Population standard deviation state using Welford's algorithm (count, mean, M2).
    StdDevPop { count: i64, mean: f64, m2: f64 },
    /// Discrete percentile state (values, percentile).
    PercentileDisc { values: Vec<f64>, percentile: f64 },
    /// Continuous percentile state (values, percentile).
    PercentileCont { values: Vec<f64>, percentile: f64 },
    /// GROUP_CONCAT / LISTAGG state (collected string values, separator).
    GroupConcat(Vec<String>, String),
    /// GROUP_CONCAT / LISTAGG distinct state (collected string values, separator, seen).
    GroupConcatDistinct(Vec<String>, String, HashSet<HashableValue>),
    /// SAMPLE state (first non-null value encountered).
    Sample(Option<Value>),
    /// Sample variance state using Welford's algorithm (count, mean, M2).
    Variance { count: i64, mean: f64, m2: f64 },
    /// Population variance state using Welford's algorithm (count, mean, M2).
    VariancePop { count: i64, mean: f64, m2: f64 },
    /// Two-variable online statistics (Welford generalization for covariance/regression).
    Bivariate {
        /// Which binary set function this state will finalize to.
        kind: AggregateFunction,
        count: i64,
        mean_x: f64,
        mean_y: f64,
        m2_x: f64,
        m2_y: f64,
        c_xy: f64,
    },
}

impl AggregateState {
    /// Creates initial state for an aggregation function.
    pub fn new(
        function: AggregateFunction,
        distinct: bool,
        percentile: Option<f64>,
        separator: Option<&str>,
    ) -> Self {
        match (function, distinct) {
            (AggregateFunction::Count | AggregateFunction::CountNonNull, false) => {
                AggregateState::Count(0)
            }
            (AggregateFunction::Count | AggregateFunction::CountNonNull, true) => {
                AggregateState::CountDistinct(0, HashSet::new())
            }
            (AggregateFunction::Sum, false) => AggregateState::SumInt(0, 0),
            (AggregateFunction::Sum, true) => AggregateState::SumIntDistinct(0, 0, HashSet::new()),
            (AggregateFunction::Avg, false) => AggregateState::Avg(0.0, 0),
            (AggregateFunction::Avg, true) => AggregateState::AvgDistinct(0.0, 0, HashSet::new()),
            (AggregateFunction::Min, _) => AggregateState::Min(None), // MIN/MAX don't need distinct
            (AggregateFunction::Max, _) => AggregateState::Max(None),
            (AggregateFunction::First, _) => AggregateState::First(None),
            (AggregateFunction::Last, _) => AggregateState::Last(None),
            (AggregateFunction::Collect, false) => AggregateState::Collect(Vec::new()),
            (AggregateFunction::Collect, true) => {
                AggregateState::CollectDistinct(Vec::new(), HashSet::new())
            }
            // Statistical functions (Welford's algorithm for online computation)
            (AggregateFunction::StdDev, _) => AggregateState::StdDev {
                count: 0,
                mean: 0.0,
                m2: 0.0,
            },
            (AggregateFunction::StdDevPop, _) => AggregateState::StdDevPop {
                count: 0,
                mean: 0.0,
                m2: 0.0,
            },
            (AggregateFunction::PercentileDisc, _) => AggregateState::PercentileDisc {
                values: Vec::new(),
                percentile: percentile.unwrap_or(0.5),
            },
            (AggregateFunction::PercentileCont, _) => AggregateState::PercentileCont {
                values: Vec::new(),
                percentile: percentile.unwrap_or(0.5),
            },
            (AggregateFunction::GroupConcat, false) => {
                AggregateState::GroupConcat(Vec::new(), separator.unwrap_or(" ").to_string())
            }
            (AggregateFunction::GroupConcat, true) => AggregateState::GroupConcatDistinct(
                Vec::new(),
                separator.unwrap_or(" ").to_string(),
                HashSet::new(),
            ),
            (AggregateFunction::Sample, _) => AggregateState::Sample(None),
            // Binary set functions (all share the same Bivariate state)
            (
                AggregateFunction::CovarSamp
                | AggregateFunction::CovarPop
                | AggregateFunction::Corr
                | AggregateFunction::RegrSlope
                | AggregateFunction::RegrIntercept
                | AggregateFunction::RegrR2
                | AggregateFunction::RegrCount
                | AggregateFunction::RegrSxx
                | AggregateFunction::RegrSyy
                | AggregateFunction::RegrSxy
                | AggregateFunction::RegrAvgx
                | AggregateFunction::RegrAvgy,
                _,
            ) => AggregateState::Bivariate {
                kind: function,
                count: 0,
                mean_x: 0.0,
                mean_y: 0.0,
                m2_x: 0.0,
                m2_y: 0.0,
                c_xy: 0.0,
            },
            (AggregateFunction::Variance, _) => AggregateState::Variance {
                count: 0,
                mean: 0.0,
                m2: 0.0,
            },
            (AggregateFunction::VariancePop, _) => AggregateState::VariancePop {
                count: 0,
                mean: 0.0,
                m2: 0.0,
            },
        }
    }

    /// Updates the state with a new value.
    ///
    /// For `COUNT(*)`, pass `None` to count all rows. For column-specific
    /// aggregates, pass `Some(value)`. A null operand leaves every state as it
    /// was: aggregates skip nulls, also as the first operand of a group.
    pub fn update(&mut self, value: Option<Value>) {
        if matches!(value, Some(Value::Null)) {
            return;
        }
        match self {
            AggregateState::Count(count) => {
                *count += 1;
            }
            AggregateState::CountDistinct(count, seen) => {
                if let Some(ref v) = value {
                    let hashable = HashableValue::from(v);
                    if seen.insert(hashable) {
                        *count += 1;
                    }
                }
            }
            AggregateState::SumInt(sum, count) => {
                if let Some(Value::Int64(v)) = value {
                    *sum += v;
                    *count += 1;
                } else if let Some(Value::Float64(v)) = value {
                    // Convert to float sum, carrying count forward
                    *self = AggregateState::SumFloat(*sum as f64 + v, 0.0, *count + 1);
                } else if let Some(ref v) = value {
                    // RDF stores numeric literals as strings - try to parse
                    if let Some(num) = value_to_f64(v) {
                        *self = AggregateState::SumFloat(*sum as f64 + num, 0.0, *count + 1);
                    }
                }
            }
            AggregateState::SumIntDistinct(sum, count, seen) => {
                if let Some(ref v) = value {
                    let hashable = HashableValue::from(v);
                    if seen.insert(hashable) {
                        if let Value::Int64(i) = v {
                            *sum += i;
                            *count += 1;
                        } else if let Value::Float64(f) = v {
                            // Convert to float distinct: move the seen set instead of cloning
                            let moved_seen = std::mem::take(seen);
                            *self = AggregateState::SumFloatDistinct(
                                *sum as f64 + f,
                                0.0,
                                *count + 1,
                                moved_seen,
                            );
                        } else if let Some(num) = value_to_f64(v) {
                            // RDF string-encoded numerics
                            let moved_seen = std::mem::take(seen);
                            *self = AggregateState::SumFloatDistinct(
                                *sum as f64 + num,
                                0.0,
                                *count + 1,
                                moved_seen,
                            );
                        }
                    }
                }
            }
            AggregateState::SumFloat(sum, comp, count) => {
                if let Some(ref v) = value {
                    // Use value_to_f64 which now handles strings
                    if let Some(num) = value_to_f64(v) {
                        // Kahan compensated summation to reduce rounding error
                        let y = num - *comp;
                        let t = *sum + y;
                        *comp = (t - *sum) - y;
                        *sum = t;
                        *count += 1;
                    }
                }
            }
            AggregateState::SumFloatDistinct(sum, comp, count, seen) => {
                if let Some(ref v) = value {
                    let hashable = HashableValue::from(v);
                    if seen.insert(hashable)
                        && let Some(num) = value_to_f64(v)
                    {
                        let y = num - *comp;
                        let t = *sum + y;
                        *comp = (t - *sum) - y;
                        *sum = t;
                        *count += 1;
                    }
                }
            }
            AggregateState::Avg(sum, count) => {
                if let Some(ref v) = value
                    && let Some(num) = value_to_f64(v)
                {
                    *sum += num;
                    *count += 1;
                }
            }
            AggregateState::AvgDistinct(sum, count, seen) => {
                if let Some(ref v) = value {
                    let hashable = HashableValue::from(v);
                    if seen.insert(hashable)
                        && let Some(num) = value_to_f64(v)
                    {
                        *sum += num;
                        *count += 1;
                    }
                }
            }
            AggregateState::Min(min) => {
                if let Some(v) = value {
                    match min {
                        None => *min = Some(v),
                        Some(current) => {
                            if compare_values(&v, current) == Some(std::cmp::Ordering::Less) {
                                *min = Some(v);
                            }
                        }
                    }
                }
            }
            AggregateState::Max(max) => {
                if let Some(v) = value {
                    match max {
                        None => *max = Some(v),
                        Some(current) => {
                            if compare_values(&v, current) == Some(std::cmp::Ordering::Greater) {
                                *max = Some(v);
                            }
                        }
                    }
                }
            }
            AggregateState::First(first) => {
                if first.is_none() {
                    *first = value;
                }
            }
            AggregateState::Last(last) => {
                if value.is_some() {
                    *last = value;
                }
            }
            AggregateState::Collect(list) => {
                if let Some(v) = value {
                    list.push(v);
                }
            }
            AggregateState::CollectDistinct(list, seen) => {
                if let Some(v) = value {
                    let hashable = HashableValue::from(&v);
                    if seen.insert(hashable) {
                        list.push(v);
                    }
                }
            }
            // Statistical functions using Welford's online algorithm
            AggregateState::StdDev { count, mean, m2 }
            | AggregateState::StdDevPop { count, mean, m2 }
            | AggregateState::Variance { count, mean, m2 }
            | AggregateState::VariancePop { count, mean, m2 } => {
                if let Some(ref v) = value
                    && let Some(x) = value_to_f64(v)
                {
                    *count += 1;
                    let delta = x - *mean;
                    *mean += delta / *count as f64;
                    let delta2 = x - *mean;
                    *m2 += delta * delta2;
                }
            }
            AggregateState::PercentileDisc { values, .. }
            | AggregateState::PercentileCont { values, .. } => {
                if let Some(ref v) = value
                    && let Some(x) = value_to_f64(v)
                {
                    values.push(x);
                }
            }
            AggregateState::GroupConcat(list, _sep) => {
                if let Some(v) = value {
                    list.push(agg_value_to_string(&v));
                }
            }
            AggregateState::GroupConcatDistinct(list, _sep, seen) => {
                if let Some(v) = value {
                    let hashable = HashableValue::from(&v);
                    if seen.insert(hashable) {
                        list.push(agg_value_to_string(&v));
                    }
                }
            }
            AggregateState::Sample(sample) => {
                if sample.is_none() {
                    *sample = value;
                }
            }
            AggregateState::Bivariate { .. } => {
                // Bivariate functions require two values; use update_bivariate() instead.
                // Single-value update is a no-op for bivariate state.
            }
        }
    }

    /// Updates a bivariate (two-variable) aggregate state with a pair of values.
    ///
    /// Uses the two-variable Welford online algorithm for numerically stable computation
    /// of covariance and related statistics. Skips the update if either value is null.
    pub fn update_bivariate(&mut self, y_val: Option<Value>, x_val: Option<Value>) {
        if let AggregateState::Bivariate {
            count,
            mean_x,
            mean_y,
            m2_x,
            m2_y,
            c_xy,
            ..
        } = self
        {
            // Skip if either value is null (SQL semantics: exclude non-pairs)
            if let (Some(y), Some(x)) = (&y_val, &x_val)
                && let (Some(y_f), Some(x_f)) = (value_to_f64(y), value_to_f64(x))
            {
                *count += 1;
                let n = *count as f64;
                let dx = x_f - *mean_x;
                let dy = y_f - *mean_y;
                *mean_x += dx / n;
                *mean_y += dy / n;
                let dx2 = x_f - *mean_x; // post-update delta
                let dy2 = y_f - *mean_y; // post-update delta
                *m2_x += dx * dx2;
                *m2_y += dy * dy2;
                *c_xy += dx * dy2;
            }
        }
    }

    /// Finalizes the state and returns the result value.
    pub fn finalize(&self) -> Value {
        match self {
            AggregateState::Count(count) | AggregateState::CountDistinct(count, _) => {
                Value::Int64(*count)
            }
            AggregateState::SumInt(sum, count) | AggregateState::SumIntDistinct(sum, count, _) => {
                if *count == 0 {
                    Value::Null
                } else {
                    Value::Int64(*sum)
                }
            }
            AggregateState::SumFloat(sum, _, count)
            | AggregateState::SumFloatDistinct(sum, _, count, _) => {
                if *count == 0 {
                    Value::Null
                } else {
                    Value::Float64(*sum)
                }
            }
            AggregateState::Avg(sum, count) | AggregateState::AvgDistinct(sum, count, _) => {
                if *count == 0 {
                    Value::Null
                } else {
                    Value::Float64(*sum / *count as f64)
                }
            }
            AggregateState::Min(min) => min.clone().unwrap_or(Value::Null),
            AggregateState::Max(max) => max.clone().unwrap_or(Value::Null),
            AggregateState::First(first) => first.clone().unwrap_or(Value::Null),
            AggregateState::Last(last) => last.clone().unwrap_or(Value::Null),
            AggregateState::Collect(list) | AggregateState::CollectDistinct(list, _) => {
                Value::List(list.clone().into())
            }
            // Sample standard deviation: sqrt(M2 / (n - 1))
            AggregateState::StdDev { count, m2, .. } => {
                if *count < 2 {
                    Value::Null
                } else {
                    Value::Float64((*m2 / (*count - 1) as f64).sqrt())
                }
            }
            // Population standard deviation: sqrt(M2 / n)
            AggregateState::StdDevPop { count, m2, .. } => {
                if *count == 0 {
                    Value::Null
                } else {
                    Value::Float64((*m2 / *count as f64).sqrt())
                }
            }
            // Sample variance: M2 / (n - 1)
            AggregateState::Variance { count, m2, .. } => {
                if *count < 2 {
                    Value::Null
                } else {
                    Value::Float64(*m2 / (*count - 1) as f64)
                }
            }
            // Population variance: M2 / n
            AggregateState::VariancePop { count, m2, .. } => {
                if *count == 0 {
                    Value::Null
                } else {
                    Value::Float64(*m2 / *count as f64)
                }
            }
            // Discrete percentile: return actual value at percentile position
            AggregateState::PercentileDisc { values, percentile } => {
                if values.is_empty() {
                    Value::Null
                } else {
                    let mut sorted = values.clone();
                    sorted.sort_by(|a, b| compare_floats(*a, *b));
                    // Index calculation per SQL standard: floor(p * (n - 1))
                    // reason: percentile index is bounded by sorted.len(), fits usize
                    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
                    let index = (percentile * (sorted.len() - 1) as f64).floor() as usize;
                    Value::Float64(sorted[index])
                }
            }
            // Continuous percentile: interpolate between values
            AggregateState::PercentileCont { values, percentile } => {
                if values.is_empty() {
                    Value::Null
                } else {
                    let mut sorted = values.clone();
                    sorted.sort_by(|a, b| compare_floats(*a, *b));
                    // Linear interpolation per SQL standard
                    let rank = percentile * (sorted.len() - 1) as f64;
                    // reason: rank is bounded by sorted.len() - 1, fits usize
                    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
                    let lower_idx = rank.floor() as usize;
                    // reason: rank is a non-negative f64 from percentile calculation, fits usize
                    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
                    let upper_idx = rank.ceil() as usize;
                    if lower_idx == upper_idx {
                        Value::Float64(sorted[lower_idx])
                    } else {
                        let fraction = rank - lower_idx as f64;
                        let result =
                            sorted[lower_idx] + fraction * (sorted[upper_idx] - sorted[lower_idx]);
                        Value::Float64(result)
                    }
                }
            }
            // GROUP_CONCAT: join strings with space separator (SPARQL default)
            AggregateState::GroupConcat(list, sep)
            | AggregateState::GroupConcatDistinct(list, sep, _) => {
                Value::String(list.join(sep).into())
            }
            // SAMPLE: return the first non-null value seen
            AggregateState::Sample(sample) => sample.clone().unwrap_or(Value::Null),
            // Binary set functions: dispatch on kind
            AggregateState::Bivariate {
                kind,
                count,
                mean_x,
                mean_y,
                m2_x,
                m2_y,
                c_xy,
            } => {
                let n = *count;
                match kind {
                    AggregateFunction::CovarSamp => {
                        if n < 2 {
                            Value::Null
                        } else {
                            Value::Float64(*c_xy / (n - 1) as f64)
                        }
                    }
                    AggregateFunction::CovarPop => {
                        if n == 0 {
                            Value::Null
                        } else {
                            Value::Float64(*c_xy / n as f64)
                        }
                    }
                    AggregateFunction::Corr => {
                        if n == 0 || *m2_x == 0.0 || *m2_y == 0.0 {
                            Value::Null
                        } else {
                            Value::Float64(*c_xy / (*m2_x * *m2_y).sqrt())
                        }
                    }
                    AggregateFunction::RegrSlope => {
                        if n == 0 || *m2_x == 0.0 {
                            Value::Null
                        } else {
                            Value::Float64(*c_xy / *m2_x)
                        }
                    }
                    AggregateFunction::RegrIntercept => {
                        if n == 0 || *m2_x == 0.0 {
                            Value::Null
                        } else {
                            let slope = *c_xy / *m2_x;
                            Value::Float64(*mean_y - slope * *mean_x)
                        }
                    }
                    AggregateFunction::RegrR2 => {
                        if n == 0 || *m2_x == 0.0 || *m2_y == 0.0 {
                            Value::Null
                        } else {
                            Value::Float64((*c_xy * *c_xy) / (*m2_x * *m2_y))
                        }
                    }
                    AggregateFunction::RegrCount => Value::Int64(n),
                    AggregateFunction::RegrSxx => {
                        if n == 0 {
                            Value::Null
                        } else {
                            Value::Float64(*m2_x)
                        }
                    }
                    AggregateFunction::RegrSyy => {
                        if n == 0 {
                            Value::Null
                        } else {
                            Value::Float64(*m2_y)
                        }
                    }
                    AggregateFunction::RegrSxy => {
                        if n == 0 {
                            Value::Null
                        } else {
                            Value::Float64(*c_xy)
                        }
                    }
                    AggregateFunction::RegrAvgx => {
                        if n == 0 {
                            Value::Null
                        } else {
                            Value::Float64(*mean_x)
                        }
                    }
                    AggregateFunction::RegrAvgy => {
                        if n == 0 {
                            Value::Null
                        } else {
                            Value::Float64(*mean_y)
                        }
                    }
                    _ => Value::Null, // non-bivariate functions never reach here
                }
            }
        }
    }
}

use super::value_utils::{compare_floats, compare_values, value_to_f64};

/// Converts a Value to its string representation for GROUP_CONCAT.
fn agg_value_to_string(val: &Value) -> String {
    match val {
        Value::String(s) => s.to_string(),
        Value::Int64(i) => i.to_string(),
        Value::Float64(f) => f.to_string(),
        Value::Bool(b) => b.to_string(),
        Value::Null => String::new(),
        other => format!("{other:?}"),
    }
}

/// A group key for hash-based aggregation.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct GroupKey(Vec<GroupValueKey>);

impl GroupKey {
    /// Creates a group key from column values.
    fn from_row(chunk: &DataChunk, row: usize, group_columns: &[usize]) -> Self {
        let parts: Vec<GroupValueKey> = group_columns
            .iter()
            .map(|&col_idx| {
                chunk
                    .column(col_idx)
                    .and_then(|col| col.get_value(row))
                    .unwrap_or(Value::Null)
                    .into()
            })
            .collect();
        GroupKey(parts)
    }

    /// Converts the group key back to values.
    fn to_values(&self) -> Vec<Value> {
        self.0.iter().map(|value| value.0.clone()).collect()
    }
}

/// Hash-based aggregate operator.
///
/// Groups input by key columns and computes aggregations for each group. A
/// group key keeps its input column's type: a node or edge stays one.
pub struct HashAggregateOperator {
    /// Child operator to read from.
    child: Box<dyn Operator>,
    /// Columns to group by.
    group_columns: Vec<usize>,
    /// Aggregation expressions.
    aggregates: Vec<AggregateExpr>,
    /// Output schema.
    output_schema: Vec<LogicalType>,
    /// The column types of the input chunks.
    input_types: ColumnTypes,
    /// Ordered map: group key -> aggregate states (IndexMap for deterministic iteration order).
    groups: IndexMap<GroupKey, Vec<AggregateState>>,
    /// Whether aggregation is complete.
    aggregation_complete: bool,
    /// Results iterator.
    results: Option<std::vec::IntoIter<(GroupKey, Vec<AggregateState>)>>,
}

impl HashAggregateOperator {
    /// Creates a new hash aggregate operator.
    ///
    /// # Arguments
    /// * `child` - Child operator to read from.
    /// * `group_columns` - Column indices to group by.
    /// * `aggregates` - Aggregation expressions.
    /// * `output_schema` - Schema of the output (group columns + aggregate results).
    pub fn new(
        child: Box<dyn Operator>,
        group_columns: Vec<usize>,
        aggregates: Vec<AggregateExpr>,
        output_schema: Vec<LogicalType>,
    ) -> Self {
        Self {
            child,
            group_columns,
            aggregates,
            output_schema,
            input_types: ColumnTypes::default(),
            groups: IndexMap::new(),
            aggregation_complete: false,
            results: None,
        }
    }

    /// The output column types: the group keys' input types (see
    /// [`ColumnTypes`]; declared before any input), then the aggregates'.
    fn output_types(&self) -> Vec<LogicalType> {
        let mut types = self.output_schema.clone();
        for (i, &column) in self.group_columns.iter().enumerate() {
            if let (Some(key_type), Some(input_type)) =
                (types.get_mut(i), self.input_types.types().get(column))
            {
                *key_type = copied_column_type(input_type, Some(&*key_type));
            }
        }
        types
    }

    /// Decomposes this operator for push-based conversion.
    pub fn into_parts(self) -> (Box<dyn Operator>, Vec<usize>, Vec<AggregateExpr>) {
        (self.child, self.group_columns, self.aggregates)
    }

    /// Performs the aggregation.
    fn aggregate(&mut self) -> Result<(), OperatorError> {
        while let Some(chunk) = self.child.next()? {
            self.input_types.add(&chunk);
            for row in chunk.selected_indices() {
                let key = GroupKey::from_row(&chunk, row, &self.group_columns);

                // Get or create aggregate states for this group
                let states = self.groups.entry(key).or_insert_with(|| {
                    self.aggregates
                        .iter()
                        .map(|agg| {
                            AggregateState::new(
                                agg.function,
                                agg.distinct,
                                agg.percentile,
                                agg.separator.as_deref(),
                            )
                        })
                        .collect()
                });

                // Update each aggregate
                for (i, agg) in self.aggregates.iter().enumerate() {
                    // Binary set functions: read two column values
                    if agg.column2.is_some() {
                        let y_val = agg
                            .column
                            .and_then(|col| chunk.column(col).and_then(|c| c.get_value(row)));
                        let x_val = agg
                            .column2
                            .and_then(|col| chunk.column(col).and_then(|c| c.get_value(row)));
                        states[i].update_bivariate(y_val, x_val);
                        continue;
                    }

                    let value = match (agg.function, agg.distinct) {
                        // COUNT(*) without DISTINCT doesn't need a value
                        (AggregateFunction::Count, false) => None,
                        // COUNT DISTINCT needs the actual value to track unique values
                        (AggregateFunction::Count, true) => agg
                            .column
                            .and_then(|col| chunk.column(col).and_then(|c| c.get_value(row))),
                        _ => agg
                            .column
                            .and_then(|col| chunk.column(col).and_then(|c| c.get_value(row))),
                    };

                    // For COUNT without DISTINCT, always update. For others, skip nulls.
                    match (agg.function, agg.distinct) {
                        (AggregateFunction::Count, false) => states[i].update(None),
                        (AggregateFunction::Count, true) => {
                            // COUNT DISTINCT needs the value to track unique values
                            if value.is_some() && !matches!(value, Some(Value::Null)) {
                                states[i].update(value);
                            }
                        }
                        (AggregateFunction::CountNonNull, _) => {
                            if value.is_some() && !matches!(value, Some(Value::Null)) {
                                states[i].update(value);
                            }
                        }
                        _ => {
                            if value.is_some() && !matches!(value, Some(Value::Null)) {
                                states[i].update(value);
                            }
                        }
                    }
                }
            }
        }

        self.aggregation_complete = true;

        // Convert to results iterator (IndexMap::drain takes a range)
        let results: Vec<_> = self.groups.drain(..).collect();
        self.results = Some(results.into_iter());

        Ok(())
    }
}

impl Operator for HashAggregateOperator {
    fn next(&mut self) -> OperatorResult {
        // Perform aggregation if not done
        if !self.aggregation_complete {
            self.aggregate()?;
        }

        // Special case: no groups (global aggregation with no data)
        if self.groups.is_empty() && self.results.is_none() && self.group_columns.is_empty() {
            // For global aggregation (no GROUP BY), return one row with initial values
            let mut builder = DataChunkBuilder::with_capacity(&self.output_schema, 1);

            for agg in &self.aggregates {
                let state = AggregateState::new(
                    agg.function,
                    agg.distinct,
                    agg.percentile,
                    agg.separator.as_deref(),
                );
                let value = state.finalize();
                if let Some(col) = builder.column_mut(self.group_columns.len()) {
                    col.push_value(value);
                }
            }
            builder.advance_row();

            self.results = Some(Vec::new().into_iter()); // Mark as done
            return Ok(Some(builder.finish()));
        }

        let types = self.output_types();
        let Some(results) = &mut self.results else {
            return Ok(None);
        };

        let mut builder = DataChunkBuilder::with_capacity(&types, 2048);

        for (key, states) in results.by_ref() {
            // Output group key columns
            let key_values = key.to_values();
            for (i, value) in key_values.into_iter().enumerate() {
                if let Some(col) = builder.column_mut(i) {
                    col.push_value(value);
                }
            }

            // Output aggregate results
            for (i, state) in states.iter().enumerate() {
                let col_idx = self.group_columns.len() + i;
                if let Some(col) = builder.column_mut(col_idx) {
                    col.push_value(state.finalize());
                }
            }

            builder.advance_row();

            if builder.is_full() {
                return Ok(Some(builder.finish()));
            }
        }

        if builder.row_count() > 0 {
            Ok(Some(builder.finish()))
        } else {
            Ok(None)
        }
    }

    fn reset(&mut self) {
        self.child.reset();
        self.input_types = ColumnTypes::default();
        self.groups.clear();
        self.aggregation_complete = false;
        self.results = None;
    }

    fn name(&self) -> &'static str {
        "HashAggregate"
    }

    fn into_any(self: Box<Self>) -> Box<dyn std::any::Any + Send> {
        self
    }
}

/// Simple (non-grouping) aggregate operator for global aggregations.
///
/// Used when there's no GROUP BY clause - aggregates all input into a single row.
pub struct SimpleAggregateOperator {
    /// Child operator.
    child: Box<dyn Operator>,
    /// Aggregation expressions.
    aggregates: Vec<AggregateExpr>,
    /// Output schema.
    output_schema: Vec<LogicalType>,
    /// Aggregate states.
    states: Vec<AggregateState>,
    /// Whether aggregation is complete.
    done: bool,
}

impl SimpleAggregateOperator {
    /// Creates a new simple aggregate operator.
    pub fn new(
        child: Box<dyn Operator>,
        aggregates: Vec<AggregateExpr>,
        output_schema: Vec<LogicalType>,
    ) -> Self {
        let states = aggregates
            .iter()
            .map(|agg| {
                AggregateState::new(
                    agg.function,
                    agg.distinct,
                    agg.percentile,
                    agg.separator.as_deref(),
                )
            })
            .collect();

        Self {
            child,
            aggregates,
            output_schema,
            states,
            done: false,
        }
    }
}

impl Operator for SimpleAggregateOperator {
    fn next(&mut self) -> OperatorResult {
        if self.done {
            return Ok(None);
        }

        // Process all input
        while let Some(chunk) = self.child.next()? {
            for row in chunk.selected_indices() {
                for (i, agg) in self.aggregates.iter().enumerate() {
                    // Binary set functions: read two column values
                    if agg.column2.is_some() {
                        let y_val = agg
                            .column
                            .and_then(|col| chunk.column(col).and_then(|c| c.get_value(row)));
                        let x_val = agg
                            .column2
                            .and_then(|col| chunk.column(col).and_then(|c| c.get_value(row)));
                        self.states[i].update_bivariate(y_val, x_val);
                        continue;
                    }

                    let value = match (agg.function, agg.distinct) {
                        // COUNT(*) without DISTINCT doesn't need a value
                        (AggregateFunction::Count, false) => None,
                        // COUNT DISTINCT needs the actual value to track unique values
                        (AggregateFunction::Count, true) => agg
                            .column
                            .and_then(|col| chunk.column(col).and_then(|c| c.get_value(row))),
                        _ => agg
                            .column
                            .and_then(|col| chunk.column(col).and_then(|c| c.get_value(row))),
                    };

                    match (agg.function, agg.distinct) {
                        (AggregateFunction::Count, false) => self.states[i].update(None),
                        (AggregateFunction::Count, true) => {
                            // COUNT DISTINCT needs the value to track unique values
                            if value.is_some() && !matches!(value, Some(Value::Null)) {
                                self.states[i].update(value);
                            }
                        }
                        (AggregateFunction::CountNonNull, _) => {
                            if value.is_some() && !matches!(value, Some(Value::Null)) {
                                self.states[i].update(value);
                            }
                        }
                        _ => {
                            if value.is_some() && !matches!(value, Some(Value::Null)) {
                                self.states[i].update(value);
                            }
                        }
                    }
                }
            }
        }

        // Output single result row
        let mut builder = DataChunkBuilder::with_capacity(&self.output_schema, 1);

        for (i, state) in self.states.iter().enumerate() {
            if let Some(col) = builder.column_mut(i) {
                col.push_value(state.finalize());
            }
        }
        builder.advance_row();

        self.done = true;
        Ok(Some(builder.finish()))
    }

    fn reset(&mut self) {
        self.child.reset();
        self.states = self
            .aggregates
            .iter()
            .map(|agg| {
                AggregateState::new(
                    agg.function,
                    agg.distinct,
                    agg.percentile,
                    agg.separator.as_deref(),
                )
            })
            .collect();
        self.done = false;
    }

    fn name(&self) -> &'static str {
        "SimpleAggregate"
    }

    fn into_any(self: Box<Self>) -> Box<dyn std::any::Any + Send> {
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::execution::chunk::DataChunkBuilder;
    use arcstr::ArcStr;
    use std::sync::Arc;

    #[test]
    fn group_identity_preserves_typed_values_across_chunks() {
        use crate::execution::vector::ValueVector;
        use grafeo_common::types::HashableValue as ValueKey;
        use std::collections::HashMap;

        let vector = Value::Vector(Arc::from([1.0_f32, 2.0]));
        let values = vec![
            Value::Int64(0),
            Value::Float64(0.0),
            Value::Float64(-0.0),
            Value::Float64(f64::from_bits(0x7ff8_0000_0000_0001)),
            Value::Float64(f64::from_bits(0x7ff8_0000_0000_0002)),
            Value::List(Arc::from([Value::Int64(0)])),
            Value::List(Arc::from([Value::Float64(0.0)])),
            Value::String(ArcStr::from(format!("{vector:?}"))),
            vector,
            Value::Null,
        ];
        let chunks = (0..2)
            .map(|_| DataChunk::new(vec![ValueVector::from_values(&values)]))
            .collect();
        let mut aggregate = HashAggregateOperator::new(
            Box::new(MockOperator::new(chunks)),
            vec![0],
            vec![AggregateExpr::count_star()],
            vec![LogicalType::Any, LogicalType::Int64],
        );
        let result = aggregate.next().unwrap().unwrap();
        let actual: HashMap<ValueKey, i64> = result
            .selected_indices()
            .map(|row| {
                (
                    result.column(0).unwrap().get_value(row).unwrap().into(),
                    result.column(1).unwrap().get_int64(row).unwrap(),
                )
            })
            .collect();
        let expected: HashMap<ValueKey, i64> =
            values.into_iter().map(|value| (value.into(), 2)).collect();
        assert_eq!(result.row_count(), expected.len());
        assert_eq!(actual, expected);
        assert!(aggregate.next().unwrap().is_none());
    }

    struct MockOperator {
        chunks: Vec<DataChunk>,
        position: usize,
    }

    impl MockOperator {
        fn new(chunks: Vec<DataChunk>) -> Self {
            Self {
                chunks,
                position: 0,
            }
        }
    }

    impl Operator for MockOperator {
        fn next(&mut self) -> OperatorResult {
            if self.position < self.chunks.len() {
                let chunk = std::mem::replace(&mut self.chunks[self.position], DataChunk::empty());
                self.position += 1;
                Ok(Some(chunk))
            } else {
                Ok(None)
            }
        }

        fn reset(&mut self) {
            self.position = 0;
        }

        fn name(&self) -> &'static str {
            "Mock"
        }

        fn into_any(self: Box<Self>) -> Box<dyn std::any::Any + Send> {
            self
        }
    }

    /// A node group key keeps its column type; the count its declared one.
    #[test]
    fn group_keys_keep_their_column_types() {
        use grafeo_common::types::NodeId;

        let mut builder = DataChunkBuilder::new(&[LogicalType::Node]);
        for node in [101, 101, 102] {
            builder
                .column_mut(0)
                .unwrap()
                .push_node_id(NodeId::new(node));
            builder.advance_row();
        }
        let mut aggregate = HashAggregateOperator::new(
            Box::new(MockOperator::new(vec![builder.finish()])),
            vec![0],
            vec![AggregateExpr::count_star()],
            vec![LogicalType::Any, LogicalType::Int64],
        );

        let chunk = aggregate.next().unwrap().unwrap();
        assert_eq!(
            chunk.column_types(),
            [LogicalType::Node, LogicalType::Int64]
        );
        let mut groups: Vec<(u64, Option<Value>)> = chunk
            .selected_indices()
            .map(|row| {
                (
                    chunk.column(0).unwrap().get_node_id(row).unwrap().as_u64(),
                    chunk.column(1).unwrap().get_value(row),
                )
            })
            .collect();
        groups.sort_by_key(|(node, _)| *node);
        assert_eq!(
            groups,
            [(101, Some(Value::Int64(2))), (102, Some(Value::Int64(1)))]
        );
    }

    fn create_test_chunk() -> DataChunk {
        // Create: [(group, value)] = [(1, 10), (1, 20), (2, 30), (2, 40), (2, 50)]
        let mut builder = DataChunkBuilder::new(&[LogicalType::Int64, LogicalType::Int64]);

        let data = [(1i64, 10i64), (1, 20), (2, 30), (2, 40), (2, 50)];
        for (group, value) in data {
            builder.column_mut(0).unwrap().push_int64(group);
            builder.column_mut(1).unwrap().push_int64(value);
            builder.advance_row();
        }

        builder.finish()
    }

    #[test]
    fn test_simple_count() {
        let mock = MockOperator::new(vec![create_test_chunk()]);

        let mut agg = SimpleAggregateOperator::new(
            Box::new(mock),
            vec![AggregateExpr::count_star()],
            vec![LogicalType::Int64],
        );

        let result = agg.next().unwrap().unwrap();
        assert_eq!(result.row_count(), 1);
        assert_eq!(result.column(0).unwrap().get_int64(0), Some(5));

        // Should be done
        assert!(agg.next().unwrap().is_none());
    }

    #[test]
    fn test_simple_sum() {
        let mock = MockOperator::new(vec![create_test_chunk()]);

        let mut agg = SimpleAggregateOperator::new(
            Box::new(mock),
            vec![AggregateExpr::sum(1)], // Sum of column 1
            vec![LogicalType::Int64],
        );

        let result = agg.next().unwrap().unwrap();
        assert_eq!(result.row_count(), 1);
        // Sum: 10 + 20 + 30 + 40 + 50 = 150
        assert_eq!(result.column(0).unwrap().get_int64(0), Some(150));
    }

    #[test]
    fn test_simple_avg() {
        let mock = MockOperator::new(vec![create_test_chunk()]);

        let mut agg = SimpleAggregateOperator::new(
            Box::new(mock),
            vec![AggregateExpr::avg(1)],
            vec![LogicalType::Float64],
        );

        let result = agg.next().unwrap().unwrap();
        assert_eq!(result.row_count(), 1);
        // Avg: 150 / 5 = 30.0
        let avg = result.column(0).unwrap().get_float64(0).unwrap();
        assert!((avg - 30.0).abs() < 0.001);
    }

    #[test]
    fn test_simple_min_max() {
        let mock = MockOperator::new(vec![create_test_chunk()]);

        let mut agg = SimpleAggregateOperator::new(
            Box::new(mock),
            vec![AggregateExpr::min(1), AggregateExpr::max(1)],
            vec![LogicalType::Int64, LogicalType::Int64],
        );

        let result = agg.next().unwrap().unwrap();
        assert_eq!(result.row_count(), 1);
        assert_eq!(result.column(0).unwrap().get_int64(0), Some(10)); // Min
        assert_eq!(result.column(1).unwrap().get_int64(0), Some(50)); // Max
    }

    #[test]
    fn test_sum_with_string_values() {
        // Test SUM with string values (like RDF stores numeric literals)
        let mut builder = DataChunkBuilder::new(&[LogicalType::String]);
        builder.column_mut(0).unwrap().push_string("30");
        builder.advance_row();
        builder.column_mut(0).unwrap().push_string("25");
        builder.advance_row();
        builder.column_mut(0).unwrap().push_string("35");
        builder.advance_row();
        let chunk = builder.finish();

        let mock = MockOperator::new(vec![chunk]);
        let mut agg = SimpleAggregateOperator::new(
            Box::new(mock),
            vec![AggregateExpr::sum(0)],
            vec![LogicalType::Float64],
        );

        let result = agg.next().unwrap().unwrap();
        assert_eq!(result.row_count(), 1);
        // Should parse strings and sum: 30 + 25 + 35 = 90
        let sum_val = result.column(0).unwrap().get_float64(0).unwrap();
        assert!(
            (sum_val - 90.0).abs() < 0.001,
            "Expected 90.0, got {}",
            sum_val
        );
    }

    #[test]
    fn test_grouped_aggregation() {
        let mock = MockOperator::new(vec![create_test_chunk()]);

        // GROUP BY column 0, SUM(column 1)
        let mut agg = HashAggregateOperator::new(
            Box::new(mock),
            vec![0],                     // Group by column 0
            vec![AggregateExpr::sum(1)], // Sum of column 1
            vec![LogicalType::Int64, LogicalType::Int64],
        );

        let mut results: Vec<(i64, i64)> = Vec::new();
        while let Some(chunk) = agg.next().unwrap() {
            for row in chunk.selected_indices() {
                let group = chunk.column(0).unwrap().get_int64(row).unwrap();
                let sum = chunk.column(1).unwrap().get_int64(row).unwrap();
                results.push((group, sum));
            }
        }

        results.sort_by_key(|(g, _)| *g);
        assert_eq!(results.len(), 2);
        assert_eq!(results[0], (1, 30)); // Group 1: 10 + 20 = 30
        assert_eq!(results[1], (2, 120)); // Group 2: 30 + 40 + 50 = 120
    }

    #[test]
    fn test_grouped_count() {
        let mock = MockOperator::new(vec![create_test_chunk()]);

        // GROUP BY column 0, COUNT(*)
        let mut agg = HashAggregateOperator::new(
            Box::new(mock),
            vec![0],
            vec![AggregateExpr::count_star()],
            vec![LogicalType::Int64, LogicalType::Int64],
        );

        let mut results: Vec<(i64, i64)> = Vec::new();
        while let Some(chunk) = agg.next().unwrap() {
            for row in chunk.selected_indices() {
                let group = chunk.column(0).unwrap().get_int64(row).unwrap();
                let count = chunk.column(1).unwrap().get_int64(row).unwrap();
                results.push((group, count));
            }
        }

        results.sort_by_key(|(g, _)| *g);
        assert_eq!(results.len(), 2);
        assert_eq!(results[0], (1, 2)); // Group 1: 2 rows
        assert_eq!(results[1], (2, 3)); // Group 2: 3 rows
    }

    #[test]
    fn test_multiple_aggregates() {
        let mock = MockOperator::new(vec![create_test_chunk()]);

        // GROUP BY column 0, COUNT(*), SUM(column 1), AVG(column 1)
        let mut agg = HashAggregateOperator::new(
            Box::new(mock),
            vec![0],
            vec![
                AggregateExpr::count_star(),
                AggregateExpr::sum(1),
                AggregateExpr::avg(1),
            ],
            vec![
                LogicalType::Int64,   // Group key
                LogicalType::Int64,   // COUNT
                LogicalType::Int64,   // SUM
                LogicalType::Float64, // AVG
            ],
        );

        let mut results: Vec<(i64, i64, i64, f64)> = Vec::new();
        while let Some(chunk) = agg.next().unwrap() {
            for row in chunk.selected_indices() {
                let group = chunk.column(0).unwrap().get_int64(row).unwrap();
                let count = chunk.column(1).unwrap().get_int64(row).unwrap();
                let sum = chunk.column(2).unwrap().get_int64(row).unwrap();
                let avg = chunk.column(3).unwrap().get_float64(row).unwrap();
                results.push((group, count, sum, avg));
            }
        }

        results.sort_by_key(|(g, _, _, _)| *g);
        assert_eq!(results.len(), 2);

        // Group 1: COUNT=2, SUM=30, AVG=15.0
        assert_eq!(results[0].0, 1);
        assert_eq!(results[0].1, 2);
        assert_eq!(results[0].2, 30);
        assert!((results[0].3 - 15.0).abs() < 0.001);

        // Group 2: COUNT=3, SUM=120, AVG=40.0
        assert_eq!(results[1].0, 2);
        assert_eq!(results[1].1, 3);
        assert_eq!(results[1].2, 120);
        assert!((results[1].3 - 40.0).abs() < 0.001);
    }

    fn create_test_chunk_with_duplicates() -> DataChunk {
        // Create data with duplicate values in column 1
        // [(group, value)] = [(1, 10), (1, 10), (1, 20), (2, 30), (2, 30), (2, 30)]
        // GROUP 1: values [10, 10, 20] -> distinct count = 2
        // GROUP 2: values [30, 30, 30] -> distinct count = 1
        let mut builder = DataChunkBuilder::new(&[LogicalType::Int64, LogicalType::Int64]);

        let data = [(1i64, 10i64), (1, 10), (1, 20), (2, 30), (2, 30), (2, 30)];
        for (group, value) in data {
            builder.column_mut(0).unwrap().push_int64(group);
            builder.column_mut(1).unwrap().push_int64(value);
            builder.advance_row();
        }

        builder.finish()
    }

    #[test]
    fn test_count_distinct() {
        let mock = MockOperator::new(vec![create_test_chunk_with_duplicates()]);

        // COUNT(DISTINCT column 1)
        let mut agg = SimpleAggregateOperator::new(
            Box::new(mock),
            vec![AggregateExpr::count(1).with_distinct()],
            vec![LogicalType::Int64],
        );

        let result = agg.next().unwrap().unwrap();
        assert_eq!(result.row_count(), 1);
        // Total distinct values: 10, 20, 30 = 3 distinct values
        assert_eq!(result.column(0).unwrap().get_int64(0), Some(3));
    }

    #[test]
    fn test_grouped_count_distinct() {
        let mock = MockOperator::new(vec![create_test_chunk_with_duplicates()]);

        // GROUP BY column 0, COUNT(DISTINCT column 1)
        let mut agg = HashAggregateOperator::new(
            Box::new(mock),
            vec![0],
            vec![AggregateExpr::count(1).with_distinct()],
            vec![LogicalType::Int64, LogicalType::Int64],
        );

        let mut results: Vec<(i64, i64)> = Vec::new();
        while let Some(chunk) = agg.next().unwrap() {
            for row in chunk.selected_indices() {
                let group = chunk.column(0).unwrap().get_int64(row).unwrap();
                let count = chunk.column(1).unwrap().get_int64(row).unwrap();
                results.push((group, count));
            }
        }

        results.sort_by_key(|(g, _)| *g);
        assert_eq!(results.len(), 2);
        assert_eq!(results[0], (1, 2)); // Group 1: [10, 10, 20] -> 2 distinct values
        assert_eq!(results[1], (2, 1)); // Group 2: [30, 30, 30] -> 1 distinct value
    }

    #[test]
    fn test_sum_distinct() {
        let mock = MockOperator::new(vec![create_test_chunk_with_duplicates()]);

        // SUM(DISTINCT column 1)
        let mut agg = SimpleAggregateOperator::new(
            Box::new(mock),
            vec![AggregateExpr::sum(1).with_distinct()],
            vec![LogicalType::Int64],
        );

        let result = agg.next().unwrap().unwrap();
        assert_eq!(result.row_count(), 1);
        // Sum of distinct values: 10 + 20 + 30 = 60
        assert_eq!(result.column(0).unwrap().get_int64(0), Some(60));
    }

    #[test]
    fn test_avg_distinct() {
        let mock = MockOperator::new(vec![create_test_chunk_with_duplicates()]);

        // AVG(DISTINCT column 1)
        let mut agg = SimpleAggregateOperator::new(
            Box::new(mock),
            vec![AggregateExpr::avg(1).with_distinct()],
            vec![LogicalType::Float64],
        );

        let result = agg.next().unwrap().unwrap();
        assert_eq!(result.row_count(), 1);
        // Avg of distinct values: (10 + 20 + 30) / 3 = 20.0
        let avg = result.column(0).unwrap().get_float64(0).unwrap();
        assert!((avg - 20.0).abs() < 0.001);
    }

    fn create_statistical_test_chunk() -> DataChunk {
        // Create data: [2, 4, 4, 4, 5, 5, 7, 9]
        // Mean = 5.0, Sample StdDev = 2.138, Population StdDev = 2.0
        let mut builder = DataChunkBuilder::new(&[LogicalType::Int64]);

        for value in [2i64, 4, 4, 4, 5, 5, 7, 9] {
            builder.column_mut(0).unwrap().push_int64(value);
            builder.advance_row();
        }

        builder.finish()
    }

    #[test]
    fn test_stdev_sample() {
        let mock = MockOperator::new(vec![create_statistical_test_chunk()]);

        let mut agg = SimpleAggregateOperator::new(
            Box::new(mock),
            vec![AggregateExpr::stdev(0)],
            vec![LogicalType::Float64],
        );

        let result = agg.next().unwrap().unwrap();
        assert_eq!(result.row_count(), 1);
        // Sample standard deviation of [2, 4, 4, 4, 5, 5, 7, 9]
        // Mean = 5.0, Variance = 32/7 = 4.571, StdDev = 2.138
        let stdev = result.column(0).unwrap().get_float64(0).unwrap();
        assert!((stdev - 2.138).abs() < 0.01);
    }

    #[test]
    fn test_stdev_population() {
        let mock = MockOperator::new(vec![create_statistical_test_chunk()]);

        let mut agg = SimpleAggregateOperator::new(
            Box::new(mock),
            vec![AggregateExpr::stdev_pop(0)],
            vec![LogicalType::Float64],
        );

        let result = agg.next().unwrap().unwrap();
        assert_eq!(result.row_count(), 1);
        // Population standard deviation of [2, 4, 4, 4, 5, 5, 7, 9]
        // Mean = 5.0, Variance = 32/8 = 4.0, StdDev = 2.0
        let stdev = result.column(0).unwrap().get_float64(0).unwrap();
        assert!((stdev - 2.0).abs() < 0.01);
    }

    #[test]
    fn test_percentile_disc() {
        let mock = MockOperator::new(vec![create_statistical_test_chunk()]);

        // Median (50th percentile discrete)
        let mut agg = SimpleAggregateOperator::new(
            Box::new(mock),
            vec![AggregateExpr::percentile_disc(0, 0.5)],
            vec![LogicalType::Float64],
        );

        let result = agg.next().unwrap().unwrap();
        assert_eq!(result.row_count(), 1);
        // Sorted: [2, 4, 4, 4, 5, 5, 7, 9], index = floor(0.5 * 7) = 3, value = 4
        let percentile = result.column(0).unwrap().get_float64(0).unwrap();
        assert!((percentile - 4.0).abs() < 0.01);
    }

    #[test]
    fn test_percentile_cont() {
        let mock = MockOperator::new(vec![create_statistical_test_chunk()]);

        // Median (50th percentile continuous)
        let mut agg = SimpleAggregateOperator::new(
            Box::new(mock),
            vec![AggregateExpr::percentile_cont(0, 0.5)],
            vec![LogicalType::Float64],
        );

        let result = agg.next().unwrap().unwrap();
        assert_eq!(result.row_count(), 1);
        // Sorted: [2, 4, 4, 4, 5, 5, 7, 9], rank = 0.5 * 7 = 3.5
        // Interpolate between index 3 (4) and index 4 (5): 4 + 0.5 * (5 - 4) = 4.5
        let percentile = result.column(0).unwrap().get_float64(0).unwrap();
        assert!((percentile - 4.5).abs() < 0.01);
    }

    #[test]
    fn test_percentile_extremes() {
        // Test 0th and 100th percentiles
        let mock = MockOperator::new(vec![create_statistical_test_chunk()]);

        let mut agg = SimpleAggregateOperator::new(
            Box::new(mock),
            vec![
                AggregateExpr::percentile_disc(0, 0.0),
                AggregateExpr::percentile_disc(0, 1.0),
            ],
            vec![LogicalType::Float64, LogicalType::Float64],
        );

        let result = agg.next().unwrap().unwrap();
        assert_eq!(result.row_count(), 1);
        // 0th percentile = minimum = 2
        let p0 = result.column(0).unwrap().get_float64(0).unwrap();
        assert!((p0 - 2.0).abs() < 0.01);
        // 100th percentile = maximum = 9
        let p100 = result.column(1).unwrap().get_float64(0).unwrap();
        assert!((p100 - 9.0).abs() < 0.01);
    }

    /// NaN sorts after every other number in percentiles too. Sorting it as
    /// equal to everything was not a total order: the sort could panic.
    #[test]
    fn percentiles_sort_nan_last() {
        let input = || {
            let mut builder = DataChunkBuilder::new(&[LogicalType::Float64]);
            for value in [3.0, f64::NAN, 1.0, 2.0, f64::NAN, 4.0, 0.5, 2.5] {
                builder.column_mut(0).unwrap().push_float64(value);
                builder.advance_row();
            }
            MockOperator::new(vec![builder.finish()])
        };
        let mut agg = SimpleAggregateOperator::new(
            Box::new(input()),
            vec![
                AggregateExpr::percentile_disc(0, 0.5),
                AggregateExpr::percentile_cont(0, 0.5),
                AggregateExpr::percentile_disc(0, 1.0),
            ],
            vec![LogicalType::Float64; 3],
        );

        let result = agg.next().unwrap().unwrap();
        // Sorted: [0.5, 1, 2, 2.5, 3, 4, NaN, NaN]; rank 0.5 * 7 = 3.5.
        assert_eq!(result.column(0).unwrap().get_float64(0), Some(2.5));
        assert_eq!(result.column(1).unwrap().get_float64(0), Some(2.75));
        assert!(result.column(2).unwrap().get_float64(0).unwrap().is_nan());
    }

    #[test]
    fn test_stdev_single_value() {
        // Single value should return null for sample stdev
        let mut builder = DataChunkBuilder::new(&[LogicalType::Int64]);
        builder.column_mut(0).unwrap().push_int64(42);
        builder.advance_row();
        let chunk = builder.finish();

        let mock = MockOperator::new(vec![chunk]);

        let mut agg = SimpleAggregateOperator::new(
            Box::new(mock),
            vec![AggregateExpr::stdev(0)],
            vec![LogicalType::Float64],
        );

        let result = agg.next().unwrap().unwrap();
        assert_eq!(result.row_count(), 1);
        // Sample stdev of single value is undefined (null)
        assert!(matches!(
            result.column(0).unwrap().get_value(0),
            Some(Value::Null)
        ));
    }

    #[test]
    fn test_first_and_last() {
        let mock = MockOperator::new(vec![create_test_chunk()]);

        let mut agg = SimpleAggregateOperator::new(
            Box::new(mock),
            vec![AggregateExpr::first(1), AggregateExpr::last(1)],
            vec![LogicalType::Int64, LogicalType::Int64],
        );

        let result = agg.next().unwrap().unwrap();
        assert_eq!(result.row_count(), 1);
        // First: 10, Last: 50
        assert_eq!(result.column(0).unwrap().get_int64(0), Some(10));
        assert_eq!(result.column(1).unwrap().get_int64(0), Some(50));
    }

    #[test]
    fn test_collect() {
        let mock = MockOperator::new(vec![create_test_chunk()]);

        let mut agg = SimpleAggregateOperator::new(
            Box::new(mock),
            vec![AggregateExpr::collect(1)],
            vec![LogicalType::Any],
        );

        let result = agg.next().unwrap().unwrap();
        let val = result.column(0).unwrap().get_value(0).unwrap();
        if let Value::List(items) = val {
            assert_eq!(items.len(), 5);
        } else {
            panic!("Expected List value");
        }
    }

    #[test]
    fn test_collect_distinct() {
        let mock = MockOperator::new(vec![create_test_chunk_with_duplicates()]);

        let mut agg = SimpleAggregateOperator::new(
            Box::new(mock),
            vec![AggregateExpr::collect(1).with_distinct()],
            vec![LogicalType::Any],
        );

        let result = agg.next().unwrap().unwrap();
        let val = result.column(0).unwrap().get_value(0).unwrap();
        if let Value::List(items) = val {
            // [10, 10, 20, 30, 30, 30] -> distinct: [10, 20, 30]
            assert_eq!(items.len(), 3);
        } else {
            panic!("Expected List value");
        }
    }

    #[test]
    fn test_group_concat() {
        let mut builder = DataChunkBuilder::new(&[LogicalType::String]);
        for s in ["hello", "world", "foo"] {
            builder.column_mut(0).unwrap().push_string(s);
            builder.advance_row();
        }
        let chunk = builder.finish();
        let mock = MockOperator::new(vec![chunk]);

        let agg_expr = AggregateExpr {
            function: AggregateFunction::GroupConcat,
            column: Some(0),
            column2: None,
            distinct: false,
            alias: None,
            percentile: None,
            separator: None,
        };

        let mut agg =
            SimpleAggregateOperator::new(Box::new(mock), vec![agg_expr], vec![LogicalType::String]);

        let result = agg.next().unwrap().unwrap();
        let val = result.column(0).unwrap().get_value(0).unwrap();
        assert_eq!(val, Value::String("hello world foo".into()));
    }

    #[test]
    fn test_sample() {
        let mock = MockOperator::new(vec![create_test_chunk()]);

        let agg_expr = AggregateExpr {
            function: AggregateFunction::Sample,
            column: Some(1),
            column2: None,
            distinct: false,
            alias: None,
            percentile: None,
            separator: None,
        };

        let mut agg =
            SimpleAggregateOperator::new(Box::new(mock), vec![agg_expr], vec![LogicalType::Int64]);

        let result = agg.next().unwrap().unwrap();
        // Sample should return the first non-null value (10)
        assert_eq!(result.column(0).unwrap().get_int64(0), Some(10));
    }

    /// A null operand leaves every state as it was, also as the first operand
    /// of an empty state: each aggregate finalizes as if the nulls were never
    /// there, and one that saw only nulls as if it saw nothing. (`None` is a
    /// row for `COUNT(*)`, not an operand.)
    #[test]
    fn a_null_operand_leaves_every_state_unchanged() {
        let functions = [
            AggregateFunction::Count,
            AggregateFunction::CountNonNull,
            AggregateFunction::Sum,
            AggregateFunction::Avg,
            AggregateFunction::Min,
            AggregateFunction::Max,
            AggregateFunction::First,
            AggregateFunction::Last,
            AggregateFunction::Collect,
            AggregateFunction::StdDev,
            AggregateFunction::StdDevPop,
            AggregateFunction::Variance,
            AggregateFunction::VariancePop,
            AggregateFunction::PercentileDisc,
            AggregateFunction::PercentileCont,
            AggregateFunction::GroupConcat,
            AggregateFunction::Sample,
            AggregateFunction::CovarSamp,
            AggregateFunction::Corr,
            AggregateFunction::RegrCount,
        ];
        let operands = [19, 3, 88, 3].map(Value::Int64);
        for function in functions {
            for distinct in [false, true] {
                let new_state = || AggregateState::new(function, distinct, Some(0.5), Some("|"));
                let update = |state: &mut AggregateState, operand: &Value| {
                    if matches!(state, AggregateState::Bivariate { .. }) {
                        state.update_bivariate(Some(operand.clone()), Some(Value::Int64(19)));
                    } else {
                        state.update(Some(operand.clone()));
                    }
                };
                let mut without_nulls = new_state();
                let mut with_nulls = new_state();
                let mut only_nulls = new_state();
                update(&mut with_nulls, &Value::Null);
                update(&mut only_nulls, &Value::Null);
                for operand in &operands {
                    update(&mut without_nulls, operand);
                    update(&mut with_nulls, operand);
                    update(&mut with_nulls, &Value::Null);
                    update(&mut only_nulls, &Value::Null);
                }
                let case = format!("{function:?}, distinct: {distinct}");
                assert_eq!(with_nulls.finalize(), without_nulls.finalize(), "{case}");
                assert_eq!(only_nulls.finalize(), new_state().finalize(), "{case}");
            }
        }
    }

    #[test]
    fn test_variance_sample() {
        let mock = MockOperator::new(vec![create_statistical_test_chunk()]);

        let agg_expr = AggregateExpr {
            function: AggregateFunction::Variance,
            column: Some(0),
            column2: None,
            distinct: false,
            alias: None,
            percentile: None,
            separator: None,
        };

        let mut agg = SimpleAggregateOperator::new(
            Box::new(mock),
            vec![agg_expr],
            vec![LogicalType::Float64],
        );

        let result = agg.next().unwrap().unwrap();
        // Sample variance of [2, 4, 4, 4, 5, 5, 7, 9]: M2/(n-1) = 32/7 = 4.571
        let variance = result.column(0).unwrap().get_float64(0).unwrap();
        assert!((variance - 32.0 / 7.0).abs() < 0.01);
    }

    #[test]
    fn test_variance_population() {
        let mock = MockOperator::new(vec![create_statistical_test_chunk()]);

        let agg_expr = AggregateExpr {
            function: AggregateFunction::VariancePop,
            column: Some(0),
            column2: None,
            distinct: false,
            alias: None,
            percentile: None,
            separator: None,
        };

        let mut agg = SimpleAggregateOperator::new(
            Box::new(mock),
            vec![agg_expr],
            vec![LogicalType::Float64],
        );

        let result = agg.next().unwrap().unwrap();
        // Population variance: M2/n = 32/8 = 4.0
        let variance = result.column(0).unwrap().get_float64(0).unwrap();
        assert!((variance - 4.0).abs() < 0.01);
    }

    #[test]
    fn test_variance_single_value() {
        let mut builder = DataChunkBuilder::new(&[LogicalType::Int64]);
        builder.column_mut(0).unwrap().push_int64(42);
        builder.advance_row();
        let chunk = builder.finish();
        let mock = MockOperator::new(vec![chunk]);

        let agg_expr = AggregateExpr {
            function: AggregateFunction::Variance,
            column: Some(0),
            column2: None,
            distinct: false,
            alias: None,
            percentile: None,
            separator: None,
        };

        let mut agg = SimpleAggregateOperator::new(
            Box::new(mock),
            vec![agg_expr],
            vec![LogicalType::Float64],
        );

        let result = agg.next().unwrap().unwrap();
        // Sample variance of single value is undefined (null)
        assert!(matches!(
            result.column(0).unwrap().get_value(0),
            Some(Value::Null)
        ));
    }

    #[test]
    fn test_empty_aggregation() {
        // No input rows: COUNT should be 0, SUM/AVG/MIN/MAX should be NULL
        // (ISO/IEC 39075 Section 20.9)
        let mock = MockOperator::new(vec![]);

        let mut agg = SimpleAggregateOperator::new(
            Box::new(mock),
            vec![
                AggregateExpr::count_star(),
                AggregateExpr::sum(0),
                AggregateExpr::avg(0),
                AggregateExpr::min(0),
                AggregateExpr::max(0),
            ],
            vec![
                LogicalType::Int64,
                LogicalType::Int64,
                LogicalType::Float64,
                LogicalType::Int64,
                LogicalType::Int64,
            ],
        );

        let result = agg.next().unwrap().unwrap();
        assert_eq!(result.column(0).unwrap().get_int64(0), Some(0)); // COUNT
        assert!(matches!(
            result.column(1).unwrap().get_value(0),
            Some(Value::Null)
        )); // SUM
        assert!(matches!(
            result.column(2).unwrap().get_value(0),
            Some(Value::Null)
        )); // AVG
        assert!(matches!(
            result.column(3).unwrap().get_value(0),
            Some(Value::Null)
        )); // MIN
        assert!(matches!(
            result.column(4).unwrap().get_value(0),
            Some(Value::Null)
        )); // MAX
    }

    #[test]
    fn test_stdev_pop_single_value() {
        // Single value should return 0 for population stdev
        let mut builder = DataChunkBuilder::new(&[LogicalType::Int64]);
        builder.column_mut(0).unwrap().push_int64(42);
        builder.advance_row();
        let chunk = builder.finish();

        let mock = MockOperator::new(vec![chunk]);

        let mut agg = SimpleAggregateOperator::new(
            Box::new(mock),
            vec![AggregateExpr::stdev_pop(0)],
            vec![LogicalType::Float64],
        );

        let result = agg.next().unwrap().unwrap();
        assert_eq!(result.row_count(), 1);
        // Population stdev of single value is 0
        let stdev = result.column(0).unwrap().get_float64(0).unwrap();
        assert!((stdev - 0.0).abs() < 0.01);
    }

    #[test]
    fn test_hash_aggregate_into_any() {
        let mock = MockOperator::new(vec![]);
        let op = HashAggregateOperator::new(
            Box::new(mock),
            vec![0],
            vec![AggregateExpr::count_star()],
            vec![LogicalType::Int64, LogicalType::Int64],
        );
        let any = Box::new(op).into_any();
        assert!(any.downcast::<HashAggregateOperator>().is_ok());
    }

    #[test]
    fn test_simple_aggregate_into_any() {
        let mock = MockOperator::new(vec![]);
        let op = SimpleAggregateOperator::new(
            Box::new(mock),
            vec![AggregateExpr::count_star()],
            vec![LogicalType::Int64],
        );
        let any = Box::new(op).into_any();
        assert!(any.downcast::<SimpleAggregateOperator>().is_ok());
    }

    #[test]
    fn test_hash_aggregate_into_parts() {
        let mock = MockOperator::new(vec![]);
        let op = HashAggregateOperator::new(
            Box::new(mock),
            vec![0, 2],
            vec![AggregateExpr::sum(1), AggregateExpr::count_star()],
            vec![LogicalType::Int64, LogicalType::Int64, LogicalType::Int64],
        );
        let (mut child, group_columns, aggregates) = op.into_parts();
        assert_eq!(group_columns, vec![0, 2]);
        assert_eq!(aggregates.len(), 2);
        assert!(child.next().unwrap().is_none());
    }
}
