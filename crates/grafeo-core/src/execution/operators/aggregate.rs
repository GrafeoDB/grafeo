//! Aggregation operators for GROUP BY and aggregation functions.
//!
//! This module provides:
//! - [`HashAggregateOperator`]: Hash-based grouping with aggregation functions
//! - [`SimpleAggregateOperator`]: Global aggregation without GROUP BY
//!
//! Shared types ([`AggregateFunction`], [`AggregateExpr`], [`HashableValue`]) live in
//! the [`super::accumulator`] module.

use indexmap::IndexMap;
use std::collections::HashSet;

use grafeo_common::types::{Decimal, LogicalType, Value};

use super::accumulator::{AggregateExpr, AggregateFunction, HashableValue, RowKey};
use super::{Operator, OperatorError, OperatorResult};
use crate::execution::chunk::{ColumnTypes, DataChunkBuilder, copied_column_type};

#[cfg(feature = "spill")]
#[path = "aggregate_admitted.rs"]
mod admitted;

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
    /// Sum state (integer sum, count of values added). The sum is kept in
    /// 128 bits, which no count of 64-bit values reaches the end of, and
    /// checked once at [`finalize`](Self::finalize).
    SumInt(i128, i64),
    /// Sum distinct state (integer sum, count, seen values), as `SumInt`.
    SumIntDistinct(i128, i64, HashSet<HashableValue>),
    /// Sum state (float sum, count of values added).
    SumFloat(f64, f64, i64),
    /// Sum distinct state (float sum, compensation, count, seen values).
    SumFloatDistinct(f64, f64, i64, HashSet<HashableValue>),
    /// Exact sum of decimals and integers (`None` once it needs more than
    /// 38 digits, which fails at [`finalize`](Self::finalize)), count of
    /// values added.
    SumDecimal(Option<Decimal>, i64),
    /// Average state (sum, count).
    Avg(f64, i64),
    /// Exact average of decimals and integers: their sum (`None` once it
    /// needs more than 38 digits) and count.
    AvgDecimal(Option<Decimal>, i64),
    /// Average distinct state (sum, count, seen values).
    AvgDistinct(f64, i64, HashSet<HashableValue>),
    /// Min state: the least value, in the order of ORDER BY (openCypher
    /// orderability), whatever the types of the values.
    Min(Option<Value>),
    /// Max state: the greatest value, in the order of ORDER BY.
    Max(Option<Value>),
    /// SPARQL MIN state: RDF literals held as text, where two that read as
    /// numbers compare by those numbers (see [`AggregateExpr::rdf_literals`]).
    MinOfRdfLiterals(Option<Value>),
    /// SPARQL MAX state, see [`AggregateState::MinOfRdfLiterals`].
    MaxOfRdfLiterals(Option<Value>),
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
    /// DISTINCT for an aggregate without a distinct state of its own (the
    /// statistical and binary set functions): `inner` sees each operand, or
    /// each `(y, x)` pair, once.
    Distinct {
        /// The identities of the operands passed on so far.
        seen: HashSet<HashableValue>,
        /// The state of the aggregate itself; never another `Distinct`.
        inner: Box<AggregateState>,
    },
}

/// The decimal of a float sum that holds a whole number exactly (a sum of
/// integers, or no value yet): integers below 2^53 are exact in a float.
/// An integer sum as an exact decimal, `None` past 38 digits.
fn exact_sum(sum: i128) -> Option<Decimal> {
    Decimal::new(sum, 0).ok()
}

/// The error of a decimal sum past 38 digits.
fn decimal_sum_out_of_range() -> OperatorError {
    OperatorError::numeric_out_of_range("sum of more than 38 digits")
}

fn exact_whole(sum: f64) -> Option<Decimal> {
    const EXACT: f64 = 9_007_199_254_740_992.0;
    if sum.fract() == 0.0 && sum.abs() < EXACT {
        Decimal::from_f64(sum).ok()
    } else {
        None
    }
}

impl AggregateState {
    /// Creates the initial state of `expr`'s aggregate: [`new`](Self::new)
    /// with its parameters, and MIN and MAX of RDF literals for SPARQL.
    #[must_use]
    pub fn for_expr(expr: &AggregateExpr) -> Self {
        let state = Self::new(
            expr.function,
            expr.distinct,
            expr.percentile,
            expr.separator.as_deref(),
        );
        match state {
            AggregateState::Min(min) if expr.rdf_literals => AggregateState::MinOfRdfLiterals(min),
            AggregateState::Max(max) if expr.rdf_literals => AggregateState::MaxOfRdfLiterals(max),
            // SPARQL's AVG divides by XPath's rules: the average of integers
            // and decimals is a decimal (a double operand makes it a double).
            AggregateState::Avg(..) if expr.rdf_literals => {
                AggregateState::AvgDecimal(Some(Decimal::from_i64(0)), 0)
            }
            AggregateState::AvgDistinct(..) if expr.rdf_literals => AggregateState::Distinct {
                seen: HashSet::new(),
                inner: Box::new(AggregateState::AvgDecimal(Some(Decimal::from_i64(0)), 0)),
            },
            state => state,
        }
    }

    /// Creates initial state for an aggregation function.
    pub fn new(
        function: AggregateFunction,
        distinct: bool,
        percentile: Option<f64>,
        separator: Option<&str>,
    ) -> Self {
        let state = Self::new_counting_copies(function, distinct, percentile, separator);
        // The statistical and binary set functions keep no identities of
        // their own: DISTINCT drops the copies before they see them.
        if distinct
            && matches!(
                state,
                AggregateState::StdDev { .. }
                    | AggregateState::StdDevPop { .. }
                    | AggregateState::Variance { .. }
                    | AggregateState::VariancePop { .. }
                    | AggregateState::PercentileDisc { .. }
                    | AggregateState::PercentileCont { .. }
                    | AggregateState::Bivariate { .. }
            )
        {
            AggregateState::Distinct {
                seen: HashSet::new(),
                inner: Box::new(state),
            }
        } else {
            state
        }
    }

    /// The state of `function`, with the identities DISTINCT needs for the
    /// functions that track them themselves (count, sum, avg, collect and
    /// group_concat). MIN, MAX and SAMPLE give the same result for any
    /// number of copies of a value; FIRST and LAST follow the input order.
    fn new_counting_copies(
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
                    *sum += i128::from(v);
                    *count += 1;
                } else if let Some(Value::Decimal(d)) = value {
                    let total = exact_sum(*sum).and_then(|whole| whole.checked_add(&d).ok());
                    *self = AggregateState::SumDecimal(total, *count + 1);
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
                            *sum += i128::from(*i);
                            *count += 1;
                        } else if let Value::Decimal(d) = v {
                            let total = exact_sum(*sum).and_then(|whole| whole.checked_add(d).ok());
                            *self = AggregateState::Distinct {
                                seen: std::mem::take(seen),
                                inner: Box::new(AggregateState::SumDecimal(total, *count + 1)),
                            };
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
            AggregateState::SumDecimal(total, count) => match value {
                Some(Value::Decimal(d)) => {
                    *total = total.and_then(|t| t.checked_add(&d).ok());
                    *count += 1;
                }
                Some(Value::Int64(i)) => {
                    *total = total.and_then(|t| t.checked_add(&Decimal::from_i64(i)).ok());
                    *count += 1;
                }
                // A float (or an RDF numeric string) makes the sum a float;
                // a sum past 38 digits stays null.
                Some(ref v) => {
                    if let (Some(t), Some(num)) = (*total, value_to_f64(v)) {
                        *self = AggregateState::SumFloat(t.to_f64() + num, 0.0, *count + 1);
                    }
                }
                None => {}
            },
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
            AggregateState::Avg(sum, count) if matches!(value, Some(Value::Decimal(_))) => {
                // A whole sum came from integers (or no value yet): the
                // average becomes exact.
                if let Some(Value::Decimal(d)) = value {
                    if let Some(whole) = exact_whole(*sum) {
                        let total = whole.checked_add(&d).ok();
                        *self = AggregateState::AvgDecimal(total, *count + 1);
                    } else {
                        *sum += d.to_f64();
                        *count += 1;
                    }
                }
            }
            AggregateState::AvgDecimal(total, count) => match value {
                Some(Value::Decimal(d)) => {
                    *total = total.and_then(|t| t.checked_add(&d).ok());
                    *count += 1;
                }
                Some(Value::Int64(i)) => {
                    *total = total.and_then(|t| t.checked_add(&Decimal::from_i64(i)).ok());
                    *count += 1;
                }
                Some(ref v) => {
                    if let (Some(t), Some(num)) = (*total, value_to_f64(v)) {
                        *self = AggregateState::Avg(t.to_f64() + num, *count + 1);
                    }
                }
                None => {}
            },
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
                    if let Value::Decimal(d) = v
                        && let Some(whole) = exact_whole(*sum)
                    {
                        if seen.insert(hashable) {
                            let total = whole.checked_add(d).ok();
                            *self = AggregateState::Distinct {
                                seen: std::mem::take(seen),
                                inner: Box::new(AggregateState::AvgDecimal(total, *count + 1)),
                            };
                        }
                    } else if seen.insert(hashable)
                        && let Some(num) = value_to_f64(v)
                    {
                        *sum += num;
                        *count += 1;
                    }
                }
            }
            // MIN and MAX order values as ORDER BY does, a total order over
            // values of every type, so the result does not depend on the
            // order of the input (a value never replaced one of another type).
            AggregateState::Min(min) => {
                if let Some(v) = value
                    && min
                        .as_ref()
                        .is_none_or(|current| compare_values_total(&v, current).is_lt())
                {
                    *min = Some(v);
                }
            }
            AggregateState::Max(max) => {
                if let Some(v) = value
                    && max
                        .as_ref()
                        .is_none_or(|current| compare_values_total(&v, current).is_gt())
                {
                    *max = Some(v);
                }
            }
            AggregateState::MinOfRdfLiterals(min) => {
                if let Some(v) = value
                    && min.as_ref().is_none_or(|current| {
                        compare_values(&v, current) == Some(std::cmp::Ordering::Less)
                    })
                {
                    *min = Some(v);
                }
            }
            AggregateState::MaxOfRdfLiterals(max) => {
                if let Some(v) = value
                    && max.as_ref().is_none_or(|current| {
                        compare_values(&v, current) == Some(std::cmp::Ordering::Greater)
                    })
                {
                    *max = Some(v);
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
            AggregateState::Distinct { seen, inner } => match &value {
                Some(operand) => {
                    if seen.insert(HashableValue::from(operand)) {
                        inner.update(value);
                    }
                }
                None => inner.update(None),
            },
        }
    }

    /// Updates a bivariate (two-variable) aggregate state with a pair of values.
    ///
    /// Uses the two-variable Welford online algorithm for numerically stable computation
    /// of covariance and related statistics. Skips the update if either value is null.
    pub fn update_bivariate(&mut self, y_val: Option<Value>, x_val: Option<Value>) {
        // DISTINCT counts each (y, x) pair once; a pair with a null is no
        // pair (see below), so it is no identity either.
        if let AggregateState::Distinct { seen, inner } = self {
            if let (Some(y), Some(x)) = (&y_val, &x_val)
                && !y.is_null()
                && !x.is_null()
                && seen.insert(HashableValue::from(&Value::List(
                    vec![y.clone(), x.clone()].into(),
                )))
            {
                inner.update_bivariate(y_val, x_val);
            }
            return;
        }
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

    /// The result of the aggregate `expr` this state computed, as
    /// [`finalize`](Self::finalize) gives it, except that SPARQL's `SUM` and
    /// `AVG` of a group without values are `0` (SPARQL 1.1, 18.5.1.3 and
    /// 18.5.1.4), where GQL and Cypher give null.
    ///
    /// # Errors
    ///
    /// As [`finalize`](Self::finalize).
    pub fn finalize_for(&self, expr: &AggregateExpr) -> Result<Value, OperatorError> {
        if expr.rdf_literals
            && matches!(
                expr.function,
                AggregateFunction::Sum | AggregateFunction::Avg
            )
            && self.value_count() == Some(0)
        {
            return Ok(Value::Int64(0));
        }
        self.finalize()
    }

    /// How many values a sum or average state has taken in; `None` for the
    /// other aggregates.
    fn value_count(&self) -> Option<i64> {
        match self {
            AggregateState::SumInt(_, count)
            | AggregateState::SumIntDistinct(_, count, _)
            | AggregateState::SumFloat(_, _, count)
            | AggregateState::SumFloatDistinct(_, _, count, _)
            | AggregateState::SumDecimal(_, count)
            | AggregateState::Avg(_, count)
            | AggregateState::AvgDecimal(_, count)
            | AggregateState::AvgDistinct(_, count, _) => Some(*count),
            AggregateState::Distinct { inner, .. } => inner.value_count(),
            _ => None,
        }
    }

    /// Finalizes the state and returns the result value.
    ///
    /// # Errors
    ///
    /// An integer sum outside the 64-bit range, and a decimal sum or average
    /// whose sum needs more than 38 digits, are out of range (22003).
    pub fn finalize(&self) -> Result<Value, OperatorError> {
        Ok(match self {
            AggregateState::Count(count) | AggregateState::CountDistinct(count, _) => {
                Value::Int64(*count)
            }
            AggregateState::SumInt(sum, count) | AggregateState::SumIntDistinct(sum, count, _) => {
                if *count == 0 {
                    Value::Null
                } else {
                    Value::Int64(
                        i64::try_from(*sum).map_err(|_| {
                            OperatorError::numeric_out_of_range(format!("sum {sum}"))
                        })?,
                    )
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
            AggregateState::SumDecimal(_, 0) | AggregateState::AvgDecimal(_, 0) => Value::Null,
            AggregateState::SumDecimal(total, _) => {
                Value::Decimal(total.ok_or_else(decimal_sum_out_of_range)?)
            }
            // The average of decimals by the rule of decimal division.
            AggregateState::AvgDecimal(total, count) => {
                let total = total.ok_or_else(decimal_sum_out_of_range)?;
                Value::Decimal(
                    total
                        .checked_div(&Decimal::from_i64(*count))
                        .map_err(|_| decimal_sum_out_of_range())?,
                )
            }
            AggregateState::Min(extreme)
            | AggregateState::Max(extreme)
            | AggregateState::MinOfRdfLiterals(extreme)
            | AggregateState::MaxOfRdfLiterals(extreme) => extreme.clone().unwrap_or(Value::Null),
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
            AggregateState::Distinct { inner, .. } => return inner.finalize(),
        })
    }

    /// Whether this state takes `(y, x)` pairs through
    /// [`update_bivariate`](Self::update_bivariate): a binary set function,
    /// with or without DISTINCT.
    #[must_use]
    pub fn is_bivariate(&self) -> bool {
        match self {
            AggregateState::Bivariate { .. } => true,
            AggregateState::Distinct { inner, .. } => inner.is_bivariate(),
            _ => false,
        }
    }
}

use super::value_utils::{compare_floats, compare_values, compare_values_total, value_to_f64};

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

/// A group of the hash aggregate: the values of its grouping keys, as the
/// group's first row has them (a key keeps its value in its own type), and
/// the states of its aggregates.
struct Group {
    key_values: Vec<Value>,
    states: Vec<AggregateState>,
}

/// Group state shared with the existing push aggregate implementation.
#[derive(Clone)]
pub(crate) struct AdmittedGroupState {
    pub(crate) key_values: Vec<Value>,
    pub(crate) accumulators: Vec<AggregateState>,
}

/// The bounded output cursor used by both grouped operator interfaces.
#[cfg(feature = "spill")]
pub(crate) struct GroupedOutput {
    pub(crate) cursor: crate::execution::spill::OwnedPartitionDrainCursor<AdmittedGroupState>,
    pub(crate) types: Vec<LogicalType>,
    pub(crate) group_columns: usize,
    pub(crate) rdf_aggregates: Vec<(usize, AggregateExpr)>,
    // The schema is destroyed before the reservation that owns it.
    pub(crate) schema_owner: crate::execution::memory::GroupedMemoryOwner,
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
    /// The groups by their key (IndexMap for deterministic iteration order).
    /// Keys whose values are the same values (`3` and `3.0`) are one key,
    /// see [`RowKey`].
    groups: IndexMap<RowKey, Group>,
    /// Whether aggregation is complete.
    aggregation_complete: bool,
    /// Results iterator.
    results: Option<std::vec::IntoIter<(RowKey, Group)>>,
    #[cfg(feature = "spill")]
    memory_context: Option<(
        crate::execution::memory::OperatorMemoryContext,
        Option<std::time::Instant>,
    )>,
    #[cfg(feature = "spill")]
    accounted_backend: Option<super::push::SpillableAggregatePushOperator>,
    #[cfg(feature = "spill")]
    accounted_output: Option<GroupedOutput>,
    // The backend's copied configuration drops before this reservation.
    #[cfg(feature = "spill")]
    backend_config_owner: Option<crate::execution::memory::GroupedMemoryOwner>,
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
            #[cfg(feature = "spill")]
            memory_context: None,
            #[cfg(feature = "spill")]
            accounted_backend: None,
            #[cfg(feature = "spill")]
            accounted_output: None,
            #[cfg(feature = "spill")]
            backend_config_owner: None,
        }
    }

    /// Uses admitted grouped state while keeping this pull operator and its child.
    #[cfg(feature = "spill")]
    #[must_use]
    pub fn with_memory_context(
        mut self,
        ctx: crate::execution::memory::OperatorMemoryContext,
        deadline: Option<std::time::Instant>,
    ) -> Self {
        self.memory_context = Some((ctx, deadline));
        self
    }

    #[cfg(feature = "spill")]
    fn check_accounted_deadline(&self) -> Result<(), OperatorError> {
        #[cfg(not(target_arch = "wasm32"))]
        if let Some((_, Some(deadline))) = &self.memory_context
            && std::time::Instant::now() >= *deadline
        {
            return Err(OperatorError::Timeout);
        }
        Ok(())
    }

    #[cfg(feature = "spill")]
    fn initialize_accounted(&mut self) -> Result<(), OperatorError> {
        use crate::execution::memory::{
            GroupedMemoryOwner, checked_grouped_bytes, grouped_memory_limit,
        };
        let ctx = self
            .memory_context
            .as_ref()
            .expect("memory-aware pull aggregate")
            .0
            .clone();
        let mut bytes =
            checked_grouped_bytes(self.group_columns.len(), std::mem::size_of::<usize>())?
                .checked_add(checked_grouped_bytes(
                    self.aggregates.len(),
                    std::mem::size_of::<AggregateExpr>(),
                )?)
                .ok_or_else(grouped_memory_limit)?;
        for expression in &self.aggregates {
            for text in [&expression.alias, &expression.separator]
                .into_iter()
                .flatten()
            {
                bytes = bytes
                    .checked_add(text.len())
                    .ok_or_else(grouped_memory_limit)?;
            }
        }
        let owner =
            GroupedMemoryOwner::try_new(std::sync::Arc::clone(ctx.buffer_manager()), bytes)?;
        let mut columns = Vec::new();
        columns
            .try_reserve_exact(self.group_columns.len())
            .map_err(|_| grouped_memory_limit())?;
        columns.extend(self.group_columns.iter().copied());
        let mut aggregates = Vec::new();
        aggregates
            .try_reserve_exact(self.aggregates.len())
            .map_err(|_| grouped_memory_limit())?;
        aggregates.extend(self.aggregates.iter().cloned());
        self.accounted_backend = Some(
            super::push::SpillableAggregatePushOperator::with_memory_context(
                columns, aggregates, ctx,
            ),
        );
        self.backend_config_owner = Some(owner);
        Ok(())
    }

    #[cfg(feature = "spill")]
    fn clear_accounted(&mut self) {
        self.accounted_backend = None;
        self.accounted_output = None;
        self.backend_config_owner = None;
    }

    #[cfg(feature = "spill")]
    fn next_accounted(&mut self) -> OperatorResult {
        self.check_accounted_deadline()?;
        if !self.aggregation_complete {
            self.initialize_accounted()?;
            loop {
                self.check_accounted_deadline()?;
                let Some(chunk) = self.child.next()? else {
                    break;
                };
                self.check_accounted_deadline()?;
                self.accounted_backend
                    .as_mut()
                    .expect("admitted pull backend")
                    .ingest_accounted(chunk)?;
            }
            self.check_accounted_deadline()?;
            self.accounted_output = self
                .accounted_backend
                .as_mut()
                .expect("admitted pull backend")
                .take_accounted_output(Some(&self.output_schema))?;
            self.accounted_backend = None;
            self.backend_config_owner = None;
            self.aggregation_complete = true;
        }
        self.check_accounted_deadline()?;
        let next = match self.accounted_output.as_mut() {
            Some(output) => output.next_chunk()?,
            None => None,
        };
        self.check_accounted_deadline()?;
        if next.is_none() {
            self.accounted_output = None;
        }
        Ok(next)
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
                // Get or create aggregate states for this group; a new group
                // keeps the key values of its first row.
                let group_columns = &self.group_columns;
                let aggregates = &self.aggregates;
                let states = &mut self
                    .groups
                    .entry(RowKey::from_row(&chunk, row, group_columns))
                    .or_insert_with(|| Group {
                        key_values: RowKey::values_of(&chunk, row, group_columns),
                        states: aggregates.iter().map(AggregateState::for_expr).collect(),
                    })
                    .states;

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
        #[cfg(feature = "spill")]
        if !self.group_columns.is_empty() && self.memory_context.is_some() {
            let result = self.next_accounted();
            if result.is_err() {
                self.clear_accounted();
                self.aggregation_complete = true;
            }
            return result;
        }
        // Perform aggregation if not done
        if !self.aggregation_complete {
            self.aggregate()?;
        }

        // Special case: no groups (global aggregation with no data)
        if self.groups.is_empty() && self.results.is_none() && self.group_columns.is_empty() {
            // For global aggregation (no GROUP BY), return one row with initial values
            let mut builder = DataChunkBuilder::with_capacity(&self.output_schema, 1);

            for agg in &self.aggregates {
                let value = AggregateState::for_expr(agg).finalize_for(agg)?;
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

        for (_, group) in results.by_ref() {
            // Output group key columns
            for (i, value) in group.key_values.into_iter().enumerate() {
                if let Some(col) = builder.column_mut(i) {
                    col.push_value(value);
                }
            }

            // Output aggregate results
            for (i, (state, expr)) in group.states.iter().zip(&self.aggregates).enumerate() {
                let col_idx = self.group_columns.len() + i;
                if let Some(col) = builder.column_mut(col_idx) {
                    col.push_value(state.finalize_for(expr)?);
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
        #[cfg(feature = "spill")]
        self.clear_accounted();
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
        let states = aggregates.iter().map(AggregateState::for_expr).collect();

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

        for (i, (state, expr)) in self.states.iter().zip(&self.aggregates).enumerate() {
            if let Some(col) = builder.column_mut(i) {
                col.push_value(state.finalize_for(expr)?);
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
            .map(AggregateState::for_expr)
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
    use crate::execution::DataChunk;
    use crate::execution::chunk::DataChunkBuilder;

    #[cfg(feature = "spill")]
    fn accounted_pull_fixture() -> (
        tempfile::TempDir,
        std::sync::Arc<grafeo_common::memory::buffer::BufferManager>,
        std::sync::Arc<crate::execution::spill::SpillManager>,
        crate::execution::memory::OperatorMemoryContext,
    ) {
        use crate::execution::memory::OperatorMemoryContext;
        use crate::execution::spill::SpillManager;
        use grafeo_common::memory::buffer::{BufferManager, BufferManagerConfig};
        use std::sync::Arc;

        let directory = tempfile::TempDir::new().unwrap();
        let manager = BufferManager::new(BufferManagerConfig {
            hard_limit_fraction: 1.0,
            ..BufferManagerConfig::with_budget(8 * 1024 * 1024)
        });
        let spill = Arc::new(SpillManager::new(directory.path()).unwrap());
        let context = OperatorMemoryContext::new(Arc::clone(&manager), Arc::clone(&spill));
        (directory, manager, spill, context)
    }

    #[cfg(feature = "spill")]
    struct ControlledGroupedRows {
        phase: usize,
        manager: std::sync::Arc<grafeo_common::memory::buffer::BufferManager>,
        spill: std::sync::Arc<crate::execution::spill::SpillManager>,
        observed_spill: std::sync::Arc<std::sync::atomic::AtomicBool>,
        fail_after_spill: bool,
    }

    #[cfg(feature = "spill")]
    impl Operator for ControlledGroupedRows {
        fn next(&mut self) -> OperatorResult {
            match self.phase {
                0 => {
                    self.phase = 1;
                    let mut builder = DataChunkBuilder::with_capacity(&[LogicalType::Int64], 2051);
                    for key in 0..2051 {
                        builder.column_mut(0).unwrap().push_int64(key);
                        builder.advance_row();
                    }
                    Ok(Some(builder.finish()))
                }
                1 => {
                    self.phase = 2;
                    // The registered consumer defers the request until ingestion.
                    assert_eq!(self.manager.spill_all(), 0);
                    let mut builder = DataChunkBuilder::with_capacity(&[LogicalType::Int64], 1);
                    builder.column_mut(0).unwrap().push_int64(0);
                    builder.advance_row();
                    Ok(Some(builder.finish()))
                }
                2 => {
                    self.phase = 3;
                    if self.spill.active_file_count() == 0 || self.spill.spilled_bytes() == 0 {
                        return Err(OperatorError::Internal(
                            "the source must observe a spill".into(),
                        ));
                    }
                    self.observed_spill
                        .store(true, std::sync::atomic::Ordering::SeqCst);
                    if self.fail_after_spill {
                        Err(OperatorError::Timeout)
                    } else {
                        Ok(None)
                    }
                }
                _ => Ok(None),
            }
        }

        fn reset(&mut self) {
            self.phase = 0;
            self.observed_spill
                .store(false, std::sync::atomic::Ordering::SeqCst);
        }

        fn name(&self) -> &'static str {
            "ControlledGroupedRows"
        }

        fn into_any(self: Box<Self>) -> Box<dyn std::any::Any + Send> {
            self
        }
    }

    #[test]
    #[cfg(feature = "spill")]
    fn accounted_pull_preserves_rdf_empty_value_sum_and_average() {
        for rdf_literals in [false, true] {
            let (_directory, manager, spill, context) = accounted_pull_fixture();
            let mut builder =
                DataChunkBuilder::with_capacity(&[LogicalType::Int64, LogicalType::Any], 1);
            builder.column_mut(0).unwrap().push_int64(7);
            builder.column_mut(1).unwrap().push_value(Value::Null);
            builder.advance_row();
            let mut expressions = vec![
                AggregateExpr::sum(1),
                AggregateExpr::avg(1),
                AggregateExpr::sum(1).with_distinct(),
                AggregateExpr::avg(1).with_distinct(),
            ];
            for expr in &mut expressions {
                expr.rdf_literals = rdf_literals;
            }
            let mut aggregate = HashAggregateOperator::new(
                Box::new(MockOperator::new(vec![builder.finish()])),
                vec![0],
                expressions,
                vec![LogicalType::Any; 5],
            )
            .with_memory_context(context, None);
            let chunk = aggregate.next().unwrap().unwrap();
            assert_eq!(chunk.len(), 1);
            assert_eq!(chunk.column(0).unwrap().get_value(0), Some(Value::Int64(7)));
            let expected = if rdf_literals {
                Value::Int64(0)
            } else {
                Value::Null
            };
            for column in 1..5 {
                assert_eq!(
                    chunk.column(column).unwrap().get_value(0),
                    Some(expected.clone())
                );
            }
            assert!(manager.allocated() > 0);
            drop(chunk);
            drop(aggregate);
            assert_eq!(manager.allocated(), 0);
            assert_eq!(manager.stats().consumer_count, 0);
            assert_eq!(spill.active_file_count(), 0);
        }
    }

    #[test]
    #[cfg(feature = "spill")]
    fn accounted_pull_preserves_entity_keys_and_declared_scalar_schema() {
        use grafeo_common::types::{EdgeId, NodeId};
        let (_directory, manager, spill, context) = accounted_pull_fixture();
        let mut builder = DataChunkBuilder::with_capacity(
            &[LogicalType::Node, LogicalType::Edge, LogicalType::Int64],
            3,
        );
        for (node, edge, value) in [(101, 11, 3), (101, 11, 4), (102, 12, 5)] {
            builder
                .column_mut(0)
                .unwrap()
                .push_node_id(NodeId::new(node));
            builder
                .column_mut(1)
                .unwrap()
                .push_edge_id(EdgeId::new(edge));
            builder.column_mut(2).unwrap().push_int64(value);
            builder.advance_row();
        }
        let mut aggregate = HashAggregateOperator::new(
            Box::new(MockOperator::new(vec![builder.finish()])),
            vec![0, 1],
            vec![AggregateExpr::sum(2)],
            vec![LogicalType::Any, LogicalType::Any, LogicalType::Int64],
        )
        .with_memory_context(context, None);
        let chunk = aggregate.next().unwrap().unwrap();
        assert_eq!(
            chunk.column_types(),
            [LogicalType::Node, LogicalType::Edge, LogicalType::Int64]
        );
        let mut rows: Vec<_> = chunk
            .selected_indices()
            .map(|row| {
                (
                    chunk.column(0).unwrap().get_node_id(row).unwrap().as_u64(),
                    chunk.column(1).unwrap().get_edge_id(row).unwrap().as_u64(),
                    chunk.column(2).unwrap().get_int64(row).unwrap(),
                )
            })
            .collect();
        rows.sort_unstable();
        assert_eq!(rows, [(101, 11, 7), (102, 12, 5)]);
        assert!(manager.allocated() > 0);
        drop(chunk);
        drop(aggregate);
        assert_eq!(manager.allocated(), 0);
        assert_eq!(spill.active_file_count(), 0);
    }

    #[test]
    #[cfg(feature = "spill")]
    fn accounted_pull_partial_drain_reset_and_drop_release_grants_and_files() {
        use std::sync::{
            Arc,
            atomic::{AtomicBool, Ordering},
        };
        let (_directory, manager, spill, context) = accounted_pull_fixture();
        let observed_spill = Arc::new(AtomicBool::new(false));
        let source = ControlledGroupedRows {
            phase: 0,
            manager: Arc::clone(&manager),
            spill: Arc::clone(&spill),
            observed_spill: Arc::clone(&observed_spill),
            fail_after_spill: false,
        };
        let mut aggregate = HashAggregateOperator::new(
            Box::new(source),
            vec![0],
            vec![AggregateExpr::count_star()],
            vec![LogicalType::Int64, LogicalType::Int64],
        )
        .with_memory_context(context, None);
        for reset in [true, false] {
            let chunk = aggregate.next().unwrap().unwrap();
            assert!(observed_spill.load(Ordering::SeqCst));
            assert_eq!(chunk.len(), crate::execution::chunk::DEFAULT_CHUNK_SIZE);
            drop(chunk);
            assert!(
                manager.allocated() > 0,
                "the unfinished drain retains its grants"
            );
            if reset {
                aggregate.reset();
                assert_eq!(manager.allocated(), 0);
                assert_eq!(spill.active_file_count(), 0);
            }
        }
        drop(aggregate);
        assert_eq!(manager.allocated(), 0);
        assert_eq!(spill.active_file_count(), 0);
        assert_eq!(spill.spilled_bytes(), 0);
    }

    #[test]
    #[cfg(all(feature = "spill", not(target_arch = "wasm32")))]
    fn accounted_pull_deadline_after_observed_spill_cleans_state_and_resets() {
        use grafeo_common::memory::buffer::MemoryRegion;
        use std::sync::{
            Arc,
            atomic::{AtomicBool, Ordering},
        };
        let (directory, manager, spill, context) = accounted_pull_fixture();
        let held = manager
            .try_allocate(1024, MemoryRegion::ExecutionBuffers)
            .unwrap();
        let observed_spill = Arc::new(AtomicBool::new(false));
        let source = ControlledGroupedRows {
            phase: 0,
            manager: Arc::clone(&manager),
            spill: Arc::clone(&spill),
            observed_spill: Arc::clone(&observed_spill),
            fail_after_spill: false,
        };
        let mut aggregate = HashAggregateOperator::new(
            Box::new(source),
            vec![0],
            vec![AggregateExpr::count_star()],
            vec![LogicalType::Int64, LogicalType::Int64],
        )
        .with_memory_context(context, None);
        let first = aggregate.next().unwrap().unwrap();
        assert!(observed_spill.load(Ordering::SeqCst));
        assert_eq!(first.len(), crate::execution::chunk::DEFAULT_CHUNK_SIZE);
        let mut seen = vec![false; 2051];
        for row in first.selected_indices() {
            let key = usize::try_from(first.column(0).unwrap().get_int64(row).unwrap()).unwrap();
            assert!(key < seen.len() && !seen[key]);
            seen[key] = true;
            assert_eq!(
                first.column(1).unwrap().get_int64(row),
                Some(if key == 0 { 2 } else { 1 })
            );
        }
        drop(first);
        assert!(manager.allocated() > held.size());
        // Expire the actual stored deadline after a witnessed spill and output.
        // The first batch may already have consumed the spilled file.
        aggregate.memory_context.as_mut().unwrap().1 = Some(std::time::Instant::now());
        assert!(matches!(aggregate.next(), Err(OperatorError::Timeout)));
        assert_eq!(manager.allocated(), held.size());
        assert_eq!(spill.active_file_count(), 0);
        assert_eq!(spill.spilled_bytes(), 0);
        assert_eq!(std::fs::read_dir(directory.path()).unwrap().count(), 0);

        aggregate.memory_context.as_mut().unwrap().1 = None;
        aggregate.reset();
        seen.fill(false);
        let mut rows = 0;
        let mut total = 0;
        while let Some(chunk) = aggregate.next().unwrap() {
            for row in chunk.selected_indices() {
                let key =
                    usize::try_from(chunk.column(0).unwrap().get_int64(row).unwrap()).unwrap();
                assert!(key < seen.len() && !seen[key]);
                seen[key] = true;
                let count = chunk.column(1).unwrap().get_int64(row).unwrap();
                assert_eq!(count, if key == 0 { 2 } else { 1 });
                total += count;
                rows += 1;
            }
        }
        assert!(observed_spill.load(Ordering::SeqCst));
        assert_eq!((rows, total), (2051, 2052));
        assert!(seen.into_iter().all(|present| present));
        drop(aggregate);
        assert_eq!(manager.allocated(), held.size());
        assert_eq!(spill.active_file_count(), 0);
        assert_eq!(spill.spilled_bytes(), 0);
        assert_eq!(std::fs::read_dir(directory.path()).unwrap().count(), 0);
        drop(held);
        assert_eq!(manager.allocated(), 0);
    }

    #[test]
    #[cfg(feature = "spill")]
    fn accounted_pull_source_timeout_after_observed_spill_cleans_state() {
        use std::sync::{
            Arc,
            atomic::{AtomicBool, Ordering},
        };
        let (_directory, manager, spill, context) = accounted_pull_fixture();
        let observed_spill = Arc::new(AtomicBool::new(false));
        let source = ControlledGroupedRows {
            phase: 0,
            manager: Arc::clone(&manager),
            spill: Arc::clone(&spill),
            observed_spill: Arc::clone(&observed_spill),
            fail_after_spill: true,
        };
        let mut aggregate = HashAggregateOperator::new(
            Box::new(source),
            vec![0],
            vec![AggregateExpr::count_star()],
            vec![LogicalType::Int64, LogicalType::Int64],
        )
        .with_memory_context(context, None);
        assert!(matches!(aggregate.next(), Err(OperatorError::Timeout)));
        assert!(
            observed_spill.load(Ordering::SeqCst),
            "the failure follows an actual spill"
        );
        assert_eq!(manager.allocated(), 0);
        assert_eq!(spill.active_file_count(), 0);
        assert_eq!(spill.spilled_bytes(), 0);
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
            rdf_literals: false,
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
            rdf_literals: false,
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
                    if state.is_bivariate() {
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
                assert_eq!(
                    with_nulls.finalize().unwrap(),
                    without_nulls.finalize().unwrap(),
                    "{case}"
                );
                assert_eq!(
                    only_nulls.finalize().unwrap(),
                    new_state().finalize().unwrap(),
                    "{case}"
                );
            }
        }
    }

    /// With DISTINCT a statistical aggregate sees each value once, and a
    /// binary set function each `(y, x)` pair once: over values with copies
    /// it gives what the plain aggregate gives over each value once.
    #[test]
    fn distinct_statistical_and_binary_set_functions_see_each_value_once() {
        let functions = [
            AggregateFunction::StdDev,
            AggregateFunction::StdDevPop,
            AggregateFunction::Variance,
            AggregateFunction::VariancePop,
            AggregateFunction::PercentileDisc,
            AggregateFunction::PercentileCont,
            AggregateFunction::CovarSamp,
            AggregateFunction::CovarPop,
            AggregateFunction::Corr,
            AggregateFunction::RegrSlope,
            AggregateFunction::RegrIntercept,
            AggregateFunction::RegrR2,
            AggregateFunction::RegrCount,
            AggregateFunction::RegrSxx,
            AggregateFunction::RegrSyy,
            AggregateFunction::RegrSxy,
            AggregateFunction::RegrAvgx,
            AggregateFunction::RegrAvgy,
        ];
        for function in functions {
            let aggregate = |distinct: bool, xs: &[i64]| {
                let mut state = AggregateState::new(function, distinct, Some(0.5), None);
                for &x in xs {
                    if state.is_bivariate() {
                        let y = 2 * x + x % 19;
                        state.update_bivariate(Some(Value::Int64(y)), Some(Value::Int64(x)));
                    } else {
                        state.update(Some(Value::Int64(x)));
                    }
                }
                state.finalize().unwrap()
            };
            let with_copies = [3, 3, 19, 3, 88];
            let once = [3, 19, 88];
            assert_ne!(
                aggregate(false, &with_copies),
                aggregate(false, &once),
                "{function:?}: the copies must change the plain aggregate"
            );
            assert_eq!(
                aggregate(true, &with_copies),
                aggregate(false, &once),
                "{function:?} with DISTINCT"
            );
        }

        // Pairs that share a y or an x are different pairs
        let mut count = AggregateState::new(AggregateFunction::RegrCount, true, None, None);
        for (y, x) in [(19, 3), (19, 3), (88, 3), (19, 88)] {
            count.update_bivariate(Some(Value::Int64(y)), Some(Value::Int64(x)));
        }
        assert_eq!(count.finalize().unwrap(), Value::Int64(3));
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
            rdf_literals: false,
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
            rdf_literals: false,
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
            rdf_literals: false,
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

    /// One chunk of one column of any value per value of `values`.
    fn values_chunk(values: &[Value]) -> DataChunk {
        let mut builder = DataChunkBuilder::new(&[LogicalType::Any]);
        for value in values {
            builder.column_mut(0).unwrap().push_value(value.clone());
            builder.advance_row();
        }
        builder.finish()
    }

    /// The `(key, count(*))` rows of grouping `values` by themselves.
    fn grouped_counts(values: &[Value]) -> Vec<(Value, Value)> {
        let mut aggregate = HashAggregateOperator::new(
            Box::new(MockOperator::new(vec![values_chunk(values)])),
            vec![0],
            vec![AggregateExpr::count_star()],
            vec![LogicalType::Any, LogicalType::Int64],
        );
        let mut rows = Vec::new();
        while let Some(chunk) = aggregate.next().unwrap() {
            for row in chunk.selected_indices() {
                rows.push((
                    chunk.column(0).unwrap().get_value(row).unwrap(),
                    chunk.column(1).unwrap().get_value(row).unwrap(),
                ));
            }
        }
        rows
    }

    /// A grouping key keeps its value in its own type (a float key came back
    /// as the integer of its bits), and keys that are the same value (`3`
    /// and `3.0`, `-0.0` and `0.0`, two NaN) are one group, which keeps the
    /// value of its first row.
    #[test]
    fn group_keys_are_equivalent_values_and_keep_the_first() {
        let nan = Value::Float64(f64::NAN);
        let rows = grouped_counts(&[
            Value::Float64(1.5),
            Value::Float64(3.0),
            Value::Int64(3),
            Value::Float64(-0.0),
            Value::Float64(0.0),
            nan.clone(),
            Value::Float64(f64::from_bits(f64::NAN.to_bits() | 1)),
            Value::Float64(1.5),
        ]);
        assert_eq!(rows.len(), 4, "{rows:?}");
        assert_eq!(rows[0], (Value::Float64(1.5), Value::Int64(2)));
        assert_eq!(rows[1], (Value::Float64(3.0), Value::Int64(2)));
        assert_eq!(rows[2].1, Value::Int64(2));
        assert!(
            matches!(rows[2].0, Value::Float64(zero) if zero == 0.0 && zero.is_sign_negative()),
            "{rows:?}"
        );
        assert!(
            matches!(rows[3], (Value::Float64(n), Value::Int64(2)) if n.is_nan()),
            "{rows:?}"
        );
    }

    /// Lists, vectors and paths group by all their items: a vector key stays
    /// a vector (it became text, one group per first item and length), and
    /// paths of one length are not one group.
    #[test]
    fn composite_group_keys_compare_every_item() {
        let path = |middle: i64| Value::Path {
            nodes: vec![Value::Int64(1), Value::Int64(middle), Value::Int64(9)].into(),
            edges: vec![Value::Int64(10 + middle), Value::Int64(20 + middle)].into(),
        };
        let rows = grouped_counts(&[
            Value::Vector(vec![1.0_f32, 2.0].into()),
            Value::Vector(vec![1.0_f32, 3.0].into()),
            path(2),
            path(3),
            Value::List(vec![Value::Int64(3)].into()),
            Value::List(vec![Value::Float64(3.0)].into()),
        ]);
        assert_eq!(
            rows,
            [
                (Value::Vector(vec![1.0_f32, 2.0].into()), Value::Int64(1)),
                (Value::Vector(vec![1.0_f32, 3.0].into()), Value::Int64(1)),
                (path(2), Value::Int64(1)),
                (path(3), Value::Int64(1)),
                (Value::List(vec![Value::Int64(3)].into()), Value::Int64(2)),
            ]
        );
    }

    /// The `[sum, avg]` of `values`, through the aggregate states and
    /// `finalize_for`, also with DISTINCT.
    fn sum_and_avg(values: &[Option<Value>], rdf_literals: bool, distinct: bool) -> [Value; 2] {
        [AggregateExpr::sum(0), AggregateExpr::avg(0)].map(|mut expr| {
            expr.rdf_literals = rdf_literals;
            expr.distinct = distinct;
            let mut state = AggregateState::for_expr(&expr);
            for value in values {
                state.update(value.clone());
            }
            state.finalize_for(&expr).unwrap()
        })
    }

    /// SPARQL's SUM and AVG of no values are 0 (SPARQL 1.1, 18.5.1.3 and
    /// 18.5.1.4), also when the only operands are unbound; GQL and Cypher give
    /// null. With values, both finish as before.
    #[test]
    fn sum_and_avg_of_no_values() {
        for distinct in [false, true] {
            let zero = Value::Int64(0);
            assert_eq!(
                sum_and_avg(&[], true, distinct),
                [zero.clone(), zero.clone()]
            );
            assert_eq!(
                sum_and_avg(&[None, None], true, distinct),
                [zero.clone(), zero]
            );
            assert_eq!(
                sum_and_avg(&[], false, distinct),
                [Value::Null, Value::Null]
            );
            assert_eq!(
                sum_and_avg(&[None], false, distinct),
                [Value::Null, Value::Null]
            );
            let [sum, avg] = sum_and_avg(
                &[Some(Value::Int64(1)), Some(Value::Int64(2))],
                true,
                distinct,
            );
            assert_eq!(sum, Value::Int64(3));
            assert_eq!(avg, Value::Decimal(Decimal::parse("1.5").unwrap()));
            let [sum, _] = sum_and_avg(&[Some(Value::Float64(0.5))], true, distinct);
            assert_eq!(sum, Value::Float64(0.5));
        }
    }

    /// The `[min, max]` of `values`, through the aggregate states.
    fn extremes(values: &[Value], rdf_literals: bool) -> [Value; 2] {
        [AggregateExpr::min(0), AggregateExpr::max(0)].map(|mut expr| {
            expr.rdf_literals = rdf_literals;
            let mut state = AggregateState::for_expr(&expr);
            for value in values {
                state.update(Some(value.clone()));
            }
            state.finalize().unwrap()
        })
    }

    /// MIN and MAX order values as ORDER BY does (openCypher orderability),
    /// so mixed types give the same result in any input order: a string
    /// before numbers, lists item by item with a prefix first, NaN after
    /// every number; nulls are skipped. A value never replaced one of another
    /// type, so `['a', 3, 1]` gave `'a'` for both.
    #[test]
    fn min_and_max_use_the_order_of_order_by() {
        let a = Value::String("a".into());
        for values in [
            [Value::Int64(3), a.clone(), Value::Int64(1)],
            [a.clone(), Value::Int64(3), Value::Int64(1)],
            [Value::Int64(1), Value::Int64(3), a.clone()],
        ] {
            assert_eq!(
                extremes(&values, false),
                [a.clone(), Value::Int64(3)],
                "{values:?}"
            );
        }
        let short = Value::List(vec![Value::Int64(1)].into());
        let long = Value::List(vec![Value::Int64(1), Value::Int64(2)].into());
        assert_eq!(
            extremes(&[long.clone(), short.clone()], false),
            [short, long]
        );
        let [min, max] = extremes(
            &[Value::Int64(3), Value::Float64(f64::NAN), Value::Null],
            false,
        );
        assert_eq!(min, Value::Int64(3));
        assert!(matches!(max, Value::Float64(n) if n.is_nan()), "{max:?}");
        // Strings by code point, also when they read as numbers.
        assert_eq!(
            extremes(
                &[Value::String("9".into()), Value::String("10".into())],
                false
            ),
            [Value::String("10".into()), Value::String("9".into())]
        );
    }

    /// SPARQL's MIN and MAX compare RDF literals held as text by their
    /// numbers when both read as numbers.
    #[test]
    fn min_and_max_of_rdf_literals_compare_numbers_in_text() {
        assert_eq!(
            extremes(
                &[Value::String("9".into()), Value::String("10".into())],
                true
            ),
            [Value::String("9".into()), Value::String("10".into())]
        );
    }

    /// COUNT(DISTINCT) and COLLECT(DISTINCT) take every value once by its
    /// identity: paths of one length are different values, `3` and `3.0` one.
    #[test]
    fn distinct_aggregates_take_each_value_once_by_identity() {
        let path = |middle: i64| Value::Path {
            nodes: vec![Value::Int64(1), Value::Int64(middle)].into(),
            edges: vec![Value::Int64(10 + middle)].into(),
        };
        let values = [
            path(2),
            path(3),
            path(2),
            Value::Int64(3),
            Value::Float64(3.0),
        ];
        let mut count = AggregateState::new(AggregateFunction::Count, true, None, None);
        let mut collect = AggregateState::new(AggregateFunction::Collect, true, None, None);
        for value in values {
            count.update(Some(value.clone()));
            collect.update(Some(value));
        }
        assert_eq!(count.finalize().unwrap(), Value::Int64(3));
        assert_eq!(
            collect.finalize().unwrap(),
            Value::List(vec![path(2), path(3), Value::Int64(3)].into())
        );
    }
}
