//! Aggregation operators for GROUP BY and aggregation functions.
//!
//! This module provides:
//! - `HashAggregateOperator`: Hash-based grouping with aggregation functions
//! - Various aggregation functions: COUNT, SUM, AVG, MIN, MAX, etc.

use indexmap::IndexMap;
use std::collections::HashSet;

use grafeo_common::types::{LogicalType, Value};

/// A wrapper for Value that can be hashed (for DISTINCT tracking).
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
enum HashableValue {
    Null,
    Bool(bool),
    Int64(i64),
    Float64Bits(u64),
    String(String),
    Other(String),
}

impl From<&Value> for HashableValue {
    fn from(v: &Value) -> Self {
        match v {
            Value::Null => HashableValue::Null,
            Value::Bool(b) => HashableValue::Bool(*b),
            Value::Int64(i) => HashableValue::Int64(*i),
            Value::Float64(f) => HashableValue::Float64Bits(f.to_bits()),
            Value::String(s) => HashableValue::String(s.to_string()),
            other => HashableValue::Other(format!("{other:?}")),
        }
    }
}

impl From<Value> for HashableValue {
    fn from(v: Value) -> Self {
        Self::from(&v)
    }
}

use super::{Operator, OperatorError, OperatorResult};
use crate::execution::DataChunk;
use crate::execution::chunk::DataChunkBuilder;

/// Aggregation function types.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AggregateFunction {
    /// Count of rows (COUNT(*)).
    Count,
    /// Count of non-null values (COUNT(column)).
    CountNonNull,
    /// Sum of values.
    Sum,
    /// Average of values.
    Avg,
    /// Minimum value.
    Min,
    /// Maximum value.
    Max,
    /// First value in the group.
    First,
    /// Last value in the group.
    Last,
    /// Collect values into a list.
    Collect,
    /// Sample standard deviation (STDEV).
    StdDev,
    /// Population standard deviation (STDEVP).
    StdDevPop,
    /// Discrete percentile (PERCENTILE_DISC).
    PercentileDisc,
    /// Continuous percentile (PERCENTILE_CONT).
    PercentileCont,
}

/// An aggregation expression.
#[derive(Debug, Clone)]
pub struct AggregateExpr {
    /// The aggregation function.
    pub function: AggregateFunction,
    /// Column index to aggregate (None for COUNT(*)).
    pub column: Option<usize>,
    /// Whether to aggregate distinct values only.
    pub distinct: bool,
    /// Output alias (for naming the result column).
    pub alias: Option<String>,
    /// Percentile parameter for PERCENTILE_DISC/PERCENTILE_CONT (0.0 to 1.0).
    pub percentile: Option<f64>,
}

impl AggregateExpr {
    /// Creates a COUNT(*) expression.
    pub fn count_star() -> Self {
        Self {
            function: AggregateFunction::Count,
            column: None,
            distinct: false,
            alias: None,
            percentile: None,
        }
    }

    /// Creates a COUNT(column) expression.
    pub fn count(column: usize) -> Self {
        Self {
            function: AggregateFunction::CountNonNull,
            column: Some(column),
            distinct: false,
            alias: None,
            percentile: None,
        }
    }

    /// Creates a SUM(column) expression.
    pub fn sum(column: usize) -> Self {
        Self {
            function: AggregateFunction::Sum,
            column: Some(column),
            distinct: false,
            alias: None,
            percentile: None,
        }
    }

    /// Creates an AVG(column) expression.
    pub fn avg(column: usize) -> Self {
        Self {
            function: AggregateFunction::Avg,
            column: Some(column),
            distinct: false,
            alias: None,
            percentile: None,
        }
    }

    /// Creates a MIN(column) expression.
    pub fn min(column: usize) -> Self {
        Self {
            function: AggregateFunction::Min,
            column: Some(column),
            distinct: false,
            alias: None,
            percentile: None,
        }
    }

    /// Creates a MAX(column) expression.
    pub fn max(column: usize) -> Self {
        Self {
            function: AggregateFunction::Max,
            column: Some(column),
            distinct: false,
            alias: None,
            percentile: None,
        }
    }

    /// Creates a FIRST(column) expression.
    pub fn first(column: usize) -> Self {
        Self {
            function: AggregateFunction::First,
            column: Some(column),
            distinct: false,
            alias: None,
            percentile: None,
        }
    }

    /// Creates a LAST(column) expression.
    pub fn last(column: usize) -> Self {
        Self {
            function: AggregateFunction::Last,
            column: Some(column),
            distinct: false,
            alias: None,
            percentile: None,
        }
    }

    /// Creates a COLLECT(column) expression.
    pub fn collect(column: usize) -> Self {
        Self {
            function: AggregateFunction::Collect,
            column: Some(column),
            distinct: false,
            alias: None,
            percentile: None,
        }
    }

    /// Creates a STDEV(column) expression (sample standard deviation).
    pub fn stdev(column: usize) -> Self {
        Self {
            function: AggregateFunction::StdDev,
            column: Some(column),
            distinct: false,
            alias: None,
            percentile: None,
        }
    }

    /// Creates a STDEVP(column) expression (population standard deviation).
    pub fn stdev_pop(column: usize) -> Self {
        Self {
            function: AggregateFunction::StdDevPop,
            column: Some(column),
            distinct: false,
            alias: None,
            percentile: None,
        }
    }

    /// Creates a PERCENTILE_DISC(column, percentile) expression.
    ///
    /// # Arguments
    /// * `column` - Column index to aggregate
    /// * `percentile` - Percentile value between 0.0 and 1.0 (e.g., 0.5 for median)
    pub fn percentile_disc(column: usize, percentile: f64) -> Self {
        Self {
            function: AggregateFunction::PercentileDisc,
            column: Some(column),
            distinct: false,
            alias: None,
            percentile: Some(percentile.clamp(0.0, 1.0)),
        }
    }

    /// Creates a PERCENTILE_CONT(column, percentile) expression.
    ///
    /// # Arguments
    /// * `column` - Column index to aggregate
    /// * `percentile` - Percentile value between 0.0 and 1.0 (e.g., 0.5 for median)
    pub fn percentile_cont(column: usize, percentile: f64) -> Self {
        Self {
            function: AggregateFunction::PercentileCont,
            column: Some(column),
            distinct: false,
            alias: None,
            percentile: Some(percentile.clamp(0.0, 1.0)),
        }
    }

    /// Sets the distinct flag.
    pub fn with_distinct(mut self) -> Self {
        self.distinct = true;
        self
    }

    /// Sets the output alias.
    pub fn with_alias(mut self, alias: impl Into<String>) -> Self {
        self.alias = Some(alias.into());
        self
    }
}

/// State for a single aggregation computation.
#[derive(Debug, Clone)]
enum AggregateState {
    /// Count state.
    Count(i64),
    /// Count distinct state (count, seen values).
    CountDistinct(i64, HashSet<HashableValue>),
    /// Sum state (integer).
    SumInt(i64),
    /// Sum distinct state (integer, seen values).
    SumIntDistinct(i64, HashSet<HashableValue>),
    /// Sum state (float).
    SumFloat(f64),
    /// Sum distinct state (float, seen values).
    SumFloatDistinct(f64, HashSet<HashableValue>),
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
}

impl AggregateState {
    /// Creates initial state for an aggregation function.
    fn new(function: AggregateFunction, distinct: bool, percentile: Option<f64>) -> Self {
        match (function, distinct) {
            (AggregateFunction::Count | AggregateFunction::CountNonNull, false) => {
                AggregateState::Count(0)
            }
            (AggregateFunction::Count | AggregateFunction::CountNonNull, true) => {
                AggregateState::CountDistinct(0, HashSet::new())
            }
            (AggregateFunction::Sum, false) => AggregateState::SumInt(0),
            (AggregateFunction::Sum, true) => AggregateState::SumIntDistinct(0, HashSet::new()),
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
        }
    }

    /// Updates the state with a new value.
    fn update(&mut self, value: Option<Value>) {
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
            AggregateState::SumInt(sum) => {
                if let Some(Value::Int64(v)) = value {
                    *sum += v;
                } else if let Some(Value::Float64(v)) = value {
                    // Convert to float sum
                    *self = AggregateState::SumFloat(*sum as f64 + v);
                } else if let Some(ref v) = value {
                    // RDF stores numeric literals as strings - try to parse
                    if let Some(num) = value_to_f64(v) {
                        *self = AggregateState::SumFloat(*sum as f64 + num);
                    }
                }
            }
            AggregateState::SumIntDistinct(sum, seen) => {
                if let Some(ref v) = value {
                    let hashable = HashableValue::from(v);
                    if seen.insert(hashable) {
                        if let Value::Int64(i) = v {
                            *sum += i;
                        } else if let Value::Float64(f) = v {
                            // Convert to float distinct
                            let seen_clone = seen.clone();
                            *self = AggregateState::SumFloatDistinct(*sum as f64 + f, seen_clone);
                        } else if let Some(num) = value_to_f64(v) {
                            // RDF string-encoded numerics
                            let seen_clone = seen.clone();
                            *self = AggregateState::SumFloatDistinct(*sum as f64 + num, seen_clone);
                        }
                    }
                }
            }
            AggregateState::SumFloat(sum) => {
                if let Some(ref v) = value {
                    // Use value_to_f64 which now handles strings
                    if let Some(num) = value_to_f64(v) {
                        *sum += num;
                    }
                }
            }
            AggregateState::SumFloatDistinct(sum, seen) => {
                if let Some(ref v) = value {
                    let hashable = HashableValue::from(v);
                    if seen.insert(hashable) {
                        if let Some(num) = value_to_f64(v) {
                            *sum += num;
                        }
                    }
                }
            }
            AggregateState::Avg(sum, count) => {
                if let Some(ref v) = value {
                    if let Some(num) = value_to_f64(v) {
                        *sum += num;
                        *count += 1;
                    }
                }
            }
            AggregateState::AvgDistinct(sum, count, seen) => {
                if let Some(ref v) = value {
                    let hashable = HashableValue::from(v);
                    if seen.insert(hashable) {
                        if let Some(num) = value_to_f64(v) {
                            *sum += num;
                            *count += 1;
                        }
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
            | AggregateState::StdDevPop { count, mean, m2 } => {
                if let Some(ref v) = value {
                    if let Some(x) = value_to_f64(v) {
                        *count += 1;
                        let delta = x - *mean;
                        *mean += delta / *count as f64;
                        let delta2 = x - *mean;
                        *m2 += delta * delta2;
                    }
                }
            }
            AggregateState::PercentileDisc { values, .. }
            | AggregateState::PercentileCont { values, .. } => {
                if let Some(ref v) = value {
                    if let Some(x) = value_to_f64(v) {
                        values.push(x);
                    }
                }
            }
        }
    }

    /// Finalizes the state and returns the result value.
    fn finalize(&self) -> Value {
        match self {
            AggregateState::Count(count) | AggregateState::CountDistinct(count, _) => {
                Value::Int64(*count)
            }
            AggregateState::SumInt(sum) | AggregateState::SumIntDistinct(sum, _) => {
                Value::Int64(*sum)
            }
            AggregateState::SumFloat(sum) | AggregateState::SumFloatDistinct(sum, _) => {
                Value::Float64(*sum)
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
            // Discrete percentile: return actual value at percentile position
            AggregateState::PercentileDisc { values, percentile } => {
                if values.is_empty() {
                    Value::Null
                } else {
                    let mut sorted = values.clone();
                    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                    // Index calculation per SQL standard: floor(p * (n - 1))
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
                    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                    // Linear interpolation per SQL standard
                    let rank = percentile * (sorted.len() - 1) as f64;
                    let lower_idx = rank.floor() as usize;
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
        }
    }
}

/// Convert a value to f64 for numeric aggregations.
/// Supports RDF values stored as strings by attempting numeric parsing.
fn value_to_f64(value: &Value) -> Option<f64> {
    match value {
        Value::Int64(i) => Some(*i as f64),
        Value::Float64(f) => Some(*f),
        // RDF stores numeric literals as strings - try to parse them
        Value::String(s) => s.parse::<f64>().ok(),
        _ => None,
    }
}

/// Compare two values.
/// Supports RDF values stored as strings by attempting numeric parsing.
fn compare_values(a: &Value, b: &Value) -> Option<std::cmp::Ordering> {
    match (a, b) {
        (Value::Int64(a), Value::Int64(b)) => Some(a.cmp(b)),
        (Value::Float64(a), Value::Float64(b)) => a.partial_cmp(b),
        (Value::String(a), Value::String(b)) => {
            // Try numeric comparison first if both look like numbers
            if let (Ok(a_num), Ok(b_num)) = (a.parse::<f64>(), b.parse::<f64>()) {
                a_num.partial_cmp(&b_num)
            } else {
                Some(a.cmp(b))
            }
        }
        (Value::Bool(a), Value::Bool(b)) => Some(a.cmp(b)),
        (Value::Int64(a), Value::Float64(b)) => (*a as f64).partial_cmp(b),
        (Value::Float64(a), Value::Int64(b)) => a.partial_cmp(&(*b as f64)),
        // String-to-numeric comparisons for RDF
        (Value::String(s), Value::Int64(i)) => s.parse::<f64>().ok()?.partial_cmp(&(*i as f64)),
        (Value::String(s), Value::Float64(f)) => s.parse::<f64>().ok()?.partial_cmp(f),
        (Value::Int64(i), Value::String(s)) => (*i as f64).partial_cmp(&s.parse::<f64>().ok()?),
        (Value::Float64(f), Value::String(s)) => f.partial_cmp(&s.parse::<f64>().ok()?),
        _ => None,
    }
}

/// A group key for hash-based aggregation.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct GroupKey(Vec<GroupKeyPart>);

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
enum GroupKeyPart {
    Null,
    Bool(bool),
    Int64(i64),
    String(String),
}

impl GroupKey {
    /// Creates a group key from column values.
    fn from_row(chunk: &DataChunk, row: usize, group_columns: &[usize]) -> Self {
        let parts: Vec<GroupKeyPart> = group_columns
            .iter()
            .map(|&col_idx| {
                chunk
                    .column(col_idx)
                    .and_then(|col| col.get_value(row))
                    .map(|v| match v {
                        Value::Null => GroupKeyPart::Null,
                        Value::Bool(b) => GroupKeyPart::Bool(b),
                        Value::Int64(i) => GroupKeyPart::Int64(i),
                        Value::Float64(f) => GroupKeyPart::Int64(f.to_bits() as i64),
                        Value::String(s) => GroupKeyPart::String(s.to_string()),
                        _ => GroupKeyPart::String(format!("{v:?}")),
                    })
                    .unwrap_or(GroupKeyPart::Null)
            })
            .collect();
        GroupKey(parts)
    }

    /// Converts the group key back to values.
    fn to_values(&self) -> Vec<Value> {
        self.0
            .iter()
            .map(|part| match part {
                GroupKeyPart::Null => Value::Null,
                GroupKeyPart::Bool(b) => Value::Bool(*b),
                GroupKeyPart::Int64(i) => Value::Int64(*i),
                GroupKeyPart::String(s) => Value::String(s.clone().into()),
            })
            .collect()
    }
}

/// Hash-based aggregate operator.
///
/// Groups input by key columns and computes aggregations for each group.
pub struct HashAggregateOperator {
    /// Child operator to read from.
    child: Box<dyn Operator>,
    /// Columns to group by.
    group_columns: Vec<usize>,
    /// Aggregation expressions.
    aggregates: Vec<AggregateExpr>,
    /// Output schema.
    output_schema: Vec<LogicalType>,
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
            groups: IndexMap::new(),
            aggregation_complete: false,
            results: None,
        }
    }

    /// Performs the aggregation.
    fn aggregate(&mut self) -> Result<(), OperatorError> {
        while let Some(chunk) = self.child.next()? {
            for row in chunk.selected_indices() {
                let key = GroupKey::from_row(&chunk, row, &self.group_columns);

                // Get or create aggregate states for this group
                let states = self.groups.entry(key).or_insert_with(|| {
                    self.aggregates
                        .iter()
                        .map(|agg| AggregateState::new(agg.function, agg.distinct, agg.percentile))
                        .collect()
                });

                // Update each aggregate
                for (i, agg) in self.aggregates.iter().enumerate() {
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
                let state = AggregateState::new(agg.function, agg.distinct, agg.percentile);
                let value = state.finalize();
                if let Some(col) = builder.column_mut(self.group_columns.len()) {
                    col.push_value(value);
                }
            }
            builder.advance_row();

            self.results = Some(Vec::new().into_iter()); // Mark as done
            return Ok(Some(builder.finish()));
        }

        let results = match &mut self.results {
            Some(r) => r,
            None => return Ok(None),
        };

        let mut builder = DataChunkBuilder::with_capacity(&self.output_schema, 2048);

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
        self.groups.clear();
        self.aggregation_complete = false;
        self.results = None;
    }

    fn name(&self) -> &'static str {
        "HashAggregate"
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
            .map(|agg| AggregateState::new(agg.function, agg.distinct, agg.percentile))
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
            .map(|agg| AggregateState::new(agg.function, agg.distinct, agg.percentile))
            .collect();
        self.done = false;
    }

    fn name(&self) -> &'static str {
        "SimpleAggregate"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::execution::chunk::DataChunkBuilder;

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
}
