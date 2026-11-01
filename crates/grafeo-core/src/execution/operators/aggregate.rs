//! Aggregation operators for GROUP BY and aggregation functions.
//!
//! This module provides:
//! - [`HashAggregateOperator`]: Hash-based grouping with aggregation functions
//! - [`SimpleAggregateOperator`]: Global aggregation without GROUP BY
//!
//! Shared types ([`AggregateFunction`], [`AggregateExpr`], [`HashableValue`]) live in
//! the [`super::accumulator`] module.

use grafeo_common::types::TemporalComparison;
use grafeo_common::utils::hash::FxHashSet as HashSet;
use indexmap::IndexMap;

use grafeo_common::types::{Decimal, DecimalSum, Duration, LogicalType, Value};

use super::accumulator::{AggregateExpr, AggregateFunction, HashableValue, RowKey};
use super::{Operator, OperatorError, OperatorResult};
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
    /// Exact sum of decimals and integers, the same in any order (a sum
    /// past 38 digits fails at [`finalize`](Self::finalize)), count of
    /// values added.
    SumDecimal(DecimalSum, i64),
    /// Average state (sum, count).
    Avg(f64, i64),
    /// Exact average of decimals and integers: their sum and count.
    AvgDecimal(DecimalSum, i64),
    /// Average distinct state (sum, count, seen values).
    AvgDistinct(f64, i64, HashSet<HashableValue>),
    /// Min state: the least value, in the order of ORDER BY (openCypher
    /// orderability), whatever the types of the values.
    Min(Option<Value>),
    /// Max state: the greatest value, in the order of ORDER BY.
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
    /// DISTINCT for an aggregate without a distinct state of its own (the
    /// statistical and binary set functions): `inner` sees each operand, or
    /// each `(y, x)` pair, once.
    Distinct {
        /// The identities of the operands passed on so far.
        seen: HashSet<HashableValue>,
        /// The state of the aggregate itself; never another `Distinct`.
        inner: Box<AggregateState>,
    },
    /// Sum of durations (`None` once a component overflows, which fails at
    /// [`finalize`](Self::finalize)), count of values added.
    SumDuration(Option<Duration>, i64),
    /// Average of durations: their sum (`None` once it overflows) and count.
    AvgDuration(Option<Duration>, i64),
    /// A SUM or AVG that met an operand it does not take (a string, a
    /// boolean, numbers mixed with durations): [`finalize`](Self::finalize)
    /// fails with this message as a type error (22G03), as openCypher makes
    /// it.
    InvalidOperand(String),
    /// SPARQL's SUM and AVG (see [`AggregateExpr::rdf_literals`]): the
    /// numbers go to `inner`, an unbound operand is skipped, and an operand
    /// that is no number is an error that makes the group's result unbound
    /// ([`SparqlError`](Self::SparqlError)), as SPARQL 1.1 section 18.5
    /// defines aggregate errors.
    SparqlNumbers(Box<AggregateState>),
    /// A SPARQL SUM or AVG that met an error: its result is unbound.
    SparqlError,
}

/// The error of a sum of durations past the duration range.
fn duration_sum_out_of_range() -> OperatorError {
    OperatorError::numeric_out_of_range("sum of durations past the duration range")
}

/// `a + b`, component by component, `None` when one overflows.
fn checked_duration_sum(a: Duration, b: Duration) -> Option<Duration> {
    Some(Duration::new(
        a.months().checked_add(b.months())?,
        a.days().checked_add(b.days())?,
        a.nanos().checked_add(b.nanos())?,
    ))
}

/// The error of a decimal sum past 38 digits.
fn decimal_sum_out_of_range() -> OperatorError {
    OperatorError::numeric_out_of_range("sum of more than 38 digits")
}

/// The decimal of a float sum that holds a whole number exactly (a sum of
/// integers, or no value yet): integers below 2^53 are exact in a float.
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
    /// with its parameters, and SPARQL's SUM and AVG for RDF operands.
    #[must_use]
    pub fn for_expr(expr: &AggregateExpr) -> Self {
        let state = Self::new(
            expr.function,
            expr.distinct,
            expr.percentile,
            expr.separator.as_deref(),
        );
        match state {
            // SPARQL's AVG divides by XPath's rules: the average of integers
            // and decimals is a decimal (a double operand makes it a double).
            AggregateState::Avg(..) if expr.rdf_literals => AggregateState::SparqlNumbers(
                Box::new(AggregateState::AvgDecimal(DecimalSum::new(), 0)),
            ),
            AggregateState::AvgDistinct(..) if expr.rdf_literals => {
                AggregateState::SparqlNumbers(Box::new(AggregateState::Distinct {
                    seen: HashSet::default(),
                    inner: Box::new(AggregateState::AvgDecimal(DecimalSum::new(), 0)),
                }))
            }
            AggregateState::AvgDecimal(..) if expr.rdf_literals => {
                AggregateState::SparqlNumbers(Box::new(state))
            }
            state
                if expr.rdf_literals
                    && matches!(
                        expr.function,
                        AggregateFunction::Sum | AggregateFunction::Avg
                    ) =>
            {
                AggregateState::SparqlNumbers(Box::new(state))
            }
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
                seen: HashSet::default(),
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
                AggregateState::CountDistinct(0, HashSet::default())
            }
            (AggregateFunction::Sum, false) => AggregateState::SumInt(0, 0),
            (AggregateFunction::Sum, true) => {
                AggregateState::SumIntDistinct(0, 0, HashSet::default())
            }
            (AggregateFunction::Avg, false) => AggregateState::Avg(0.0, 0),
            (AggregateFunction::Avg, true) => {
                AggregateState::AvgDistinct(0.0, 0, HashSet::default())
            }
            (AggregateFunction::Min, _) => AggregateState::Min(None), // MIN/MAX don't need distinct
            (AggregateFunction::Max, _) => AggregateState::Max(None),
            (AggregateFunction::First, _) => AggregateState::First(None),
            (AggregateFunction::Last, _) => AggregateState::Last(None),
            (AggregateFunction::Collect, false) => AggregateState::Collect(Vec::new()),
            (AggregateFunction::Collect, true) => {
                AggregateState::CollectDistinct(Vec::new(), HashSet::default())
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
                HashSet::default(),
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
        self.update_under(value, TemporalComparison::TypesApart);
    }

    /// Updates the state with a new value (see [`Self::update`]), DISTINCT,
    /// MIN and MAX reading a local datetime against a zoned one as
    /// `temporal` reads it (the statement's `=` and `<`).
    pub fn update_under(&mut self, value: Option<Value>, temporal: TemporalComparison) {
        if matches!(value, Some(Value::Null)) {
            return;
        }
        if let Some(operand) = &value
            && let Some(next) = self.after_a_non_numeric_operand(operand)
        {
            *self = next;
            return;
        }
        match self {
            AggregateState::Count(count) => {
                *count += 1;
            }
            AggregateState::CountDistinct(count, seen) => {
                if let Some(ref v) = value {
                    let hashable = HashableValue::under(v, temporal);
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
                    let mut total = DecimalSum::from_i128(*sum);
                    total.add(&d);
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
                    let hashable = HashableValue::under(v, temporal);
                    if seen.insert(hashable) {
                        if let Value::Int64(i) = v {
                            *sum += i128::from(*i);
                            *count += 1;
                        } else if let Value::Decimal(d) = v {
                            let mut total = DecimalSum::from_i128(*sum);
                            total.add(d);
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
                    total.add(&d);
                    *count += 1;
                }
                Some(Value::Int64(i)) => {
                    total.add(&Decimal::from_i64(i));
                    *count += 1;
                }
                // A float (or an RDF numeric string) makes the sum a float.
                Some(ref v) => {
                    if let Some(num) = value_to_f64(v) {
                        *self = AggregateState::SumFloat(total.to_f64() + num, 0.0, *count + 1);
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
                    let hashable = HashableValue::under(v, temporal);
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
                        let mut total = DecimalSum::new();
                        total.add(&whole);
                        total.add(&d);
                        *self = AggregateState::AvgDecimal(total, *count + 1);
                    } else {
                        *sum += d.to_f64();
                        *count += 1;
                    }
                }
            }
            AggregateState::AvgDecimal(total, count) => match value {
                Some(Value::Decimal(d)) => {
                    total.add(&d);
                    *count += 1;
                }
                Some(Value::Int64(i)) => {
                    total.add(&Decimal::from_i64(i));
                    *count += 1;
                }
                Some(ref v) => {
                    if let Some(num) = value_to_f64(v) {
                        *self = AggregateState::Avg(total.to_f64() + num, *count + 1);
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
                    let hashable = HashableValue::under(v, temporal);
                    if let Value::Decimal(d) = v
                        && let Some(whole) = exact_whole(*sum)
                    {
                        if seen.insert(hashable) {
                            let mut total = DecimalSum::new();
                            total.add(&whole);
                            total.add(d);
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
                    && min.as_ref().is_none_or(|current| {
                        compare_values_total(&temporal.normalize(&v), &temporal.normalize(current))
                            .is_lt()
                    })
                {
                    *min = Some(v);
                }
            }
            AggregateState::Max(max) => {
                if let Some(v) = value
                    && max.as_ref().is_none_or(|current| {
                        compare_values_total(&temporal.normalize(&v), &temporal.normalize(current))
                            .is_gt()
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
                    let hashable = HashableValue::under(&v, temporal);
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
                    let hashable = HashableValue::under(&v, temporal);
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
                    if seen.insert(HashableValue::under(operand, temporal)) {
                        inner.update_under(value, temporal);
                    }
                }
                None => inner.update_under(None, temporal),
            },
            AggregateState::SumDuration(total, count)
            | AggregateState::AvgDuration(total, count) => {
                if let Some(Value::Duration(d)) = value {
                    *total = total.and_then(|t| checked_duration_sum(t, d));
                    *count += 1;
                }
            }
            AggregateState::InvalidOperand(_) => {}
            AggregateState::SparqlNumbers(inner) => match value {
                Some(Value::Int64(_) | Value::Float64(_) | Value::Decimal(_)) | None => {
                    inner.update_under(value, temporal);
                }
                Some(_) => *self = AggregateState::SparqlError,
            },
            AggregateState::SparqlError => {}
        }
    }

    /// The state a SUM or AVG moves to on an operand that is not a number:
    /// a sum or average of durations on the first operand, a duration, and
    /// [`InvalidOperand`](Self::InvalidOperand) on any other value, and on
    /// a number among durations or a duration among numbers. `None` for a
    /// number (and a duration among durations), which the state takes, and
    /// for the other aggregates.
    fn after_a_non_numeric_operand(&self, operand: &Value) -> Option<AggregateState> {
        let (function, count, seen, durations) = match self {
            AggregateState::SumInt(_, count) | AggregateState::SumFloat(_, _, count) => {
                ("sum", *count, None, false)
            }
            AggregateState::SumDecimal(_, count) => ("sum", *count, None, false),
            AggregateState::SumIntDistinct(_, count, seen)
            | AggregateState::SumFloatDistinct(_, _, count, seen) => {
                ("sum", *count, Some(seen), false)
            }
            AggregateState::Avg(_, count) | AggregateState::AvgDecimal(_, count) => {
                ("avg", *count, None, false)
            }
            AggregateState::AvgDistinct(_, count, seen) => ("avg", *count, Some(seen), false),
            AggregateState::SumDuration(_, count) => ("sum", *count, None, true),
            AggregateState::AvgDuration(_, count) => ("avg", *count, None, true),
            _ => return None,
        };
        let number = matches!(
            operand,
            Value::Int64(_) | Value::Float64(_) | Value::Decimal(_)
        );
        match operand {
            _ if number && !durations => None,
            Value::Duration(_) if durations => None,
            Value::Duration(d) if count == 0 => {
                let first = if function == "sum" {
                    AggregateState::SumDuration(Some(*d), 1)
                } else {
                    AggregateState::AvgDuration(Some(*d), 1)
                };
                Some(match seen {
                    Some(seen) => {
                        let mut seen = seen.clone();
                        seen.insert(HashableValue::from(operand));
                        AggregateState::Distinct {
                            seen,
                            inner: Box::new(first),
                        }
                    }
                    None => first,
                })
            }
            _ if number || matches!(operand, Value::Duration(_)) => {
                Some(AggregateState::InvalidOperand(format!(
                    "{function} takes numbers or durations, not both: got {operand}"
                )))
            }
            other => Some(AggregateState::InvalidOperand(format!(
                "{function} takes numbers or durations, got a {} value: {other}",
                other.type_name()
            ))),
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
            AggregateState::SumDuration(_, count) | AggregateState::AvgDuration(_, count) => {
                Some(*count)
            }
            AggregateState::Distinct { inner, .. } | AggregateState::SparqlNumbers(inner) => {
                inner.value_count()
            }
            _ => None,
        }
    }

    /// Finalizes the state and returns the result value.
    ///
    /// # Errors
    ///
    /// An integer sum outside the 64-bit range, a decimal sum whose value
    /// needs more than 38 digits and a duration sum that does not fit are
    /// out of range (22003); a decimal average always fits.
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
            AggregateState::SumDuration(_, 0) | AggregateState::AvgDuration(_, 0) => Value::Null,
            AggregateState::SumDuration(total, _) => {
                Value::Duration(total.ok_or_else(duration_sum_out_of_range)?)
            }
            // The average of durations: their sum divided by the count, each
            // component's fraction carried down (see `Duration::checked_div_decimal`).
            AggregateState::AvgDuration(total, count) => Value::Duration(
                total
                    .ok_or_else(duration_sum_out_of_range)?
                    .checked_div_decimal(&Decimal::from_i64(*count))
                    .map_err(|_| duration_sum_out_of_range())?,
            ),
            AggregateState::InvalidOperand(message) => {
                return Err(OperatorError::invalid_value_type(message.clone()));
            }
            AggregateState::SparqlNumbers(inner) => return inner.finalize(),
            AggregateState::SparqlError => Value::Null,
            // Checked once it is final: a step past 38 digits that later
            // rows took back is no error (#669).
            AggregateState::SumDecimal(total, _) => {
                Value::Decimal(total.total().map_err(|_| decimal_sum_out_of_range())?)
            }
            // The average of decimals by the rule of decimal division; it
            // always fits, as no average is beyond the largest row.
            AggregateState::AvgDecimal(total, count) => Value::Decimal(
                u64::try_from(*count)
                    .ok()
                    .and_then(|count| total.average(count).ok())
                    .ok_or_else(decimal_sum_out_of_range)?,
            ),
            AggregateState::Min(extreme) | AggregateState::Max(extreme) => {
                extreme.clone().unwrap_or(Value::Null)
            }
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

use super::value_utils::{compare_floats, compare_values_total, value_to_f64};

/// Converts a Value to its string representation for GROUP_CONCAT.
fn agg_value_to_string(val: &Value) -> String {
    match val {
        Value::String(s) => s.to_string(),
        Value::Int64(i) => i.to_string(),
        Value::Float64(f) => super::value_utils::float_text(*f),
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
    /// How the keys read a local datetime against a zoned one (see
    /// [`TemporalComparison::normalize`]); `TypesApart` reads every value
    /// as itself.
    temporal: TemporalComparison,
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
            temporal: TemporalComparison::TypesApart,
        }
    }

    /// The operator with `temporal` as the reading of its group keys and of
    /// its aggregates' DISTINCT, MIN and MAX (the statement's `=`).
    #[must_use]
    pub fn with_temporal(mut self, temporal: TemporalComparison) -> Self {
        self.temporal = temporal;
        for aggregate in &mut self.aggregates {
            aggregate.temporal = temporal;
        }
        self
    }

    /// How the group keys read temporal values (see [`Self::with_temporal`]).
    #[must_use]
    pub fn temporal(&self) -> TemporalComparison {
        self.temporal
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
            crate::execution::deadline::check()?;
            self.input_types.add(&chunk);
            for row in chunk.selected_indices() {
                // Get or create aggregate states for this group; a new group
                // keeps the key values of its first row.
                let group_columns = &self.group_columns;
                let aggregates = &self.aggregates;
                let states = &mut self
                    .groups
                    .entry(RowKey::from_row(&chunk, row, group_columns, self.temporal))
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
                        (AggregateFunction::Count, false) => {
                            states[i].update_under(None, agg.temporal);
                        }
                        (AggregateFunction::Count, true) => {
                            // COUNT DISTINCT needs the value to track unique values
                            if value.is_some() && !matches!(value, Some(Value::Null)) {
                                states[i].update_under(value, agg.temporal);
                            }
                        }
                        (AggregateFunction::CountNonNull, _) => {
                            if value.is_some() && !matches!(value, Some(Value::Null)) {
                                states[i].update_under(value, agg.temporal);
                            }
                        }
                        _ => {
                            if value.is_some() && !matches!(value, Some(Value::Null)) {
                                states[i].update_under(value, agg.temporal);
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

    /// The operator with `temporal` as the reading of its aggregates'
    /// DISTINCT, MIN and MAX (the statement's `=` and `<`).
    #[must_use]
    pub fn with_temporal(mut self, temporal: TemporalComparison) -> Self {
        for aggregate in &mut self.aggregates {
            aggregate.temporal = temporal;
        }
        self
    }
}

impl Operator for SimpleAggregateOperator {
    fn next(&mut self) -> OperatorResult {
        if self.done {
            return Ok(None);
        }

        // Process all input
        while let Some(chunk) = self.child.next()? {
            crate::execution::deadline::check()?;
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
                        (AggregateFunction::Count, false) => {
                            self.states[i].update_under(None, agg.temporal);
                        }
                        (AggregateFunction::Count, true) => {
                            // COUNT DISTINCT needs the value to track unique values
                            if value.is_some() && !matches!(value, Some(Value::Null)) {
                                self.states[i].update_under(value, agg.temporal);
                            }
                        }
                        (AggregateFunction::CountNonNull, _) => {
                            if value.is_some() && !matches!(value, Some(Value::Null)) {
                                self.states[i].update_under(value, agg.temporal);
                            }
                        }
                        _ => {
                            if value.is_some() && !matches!(value, Some(Value::Null)) {
                                self.states[i].update_under(value, agg.temporal);
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
        let strings = || {
            let mut builder = DataChunkBuilder::new(&[LogicalType::String]);
            for text in ["30", "25", "35"] {
                builder.column_mut(0).unwrap().push_string(text);
                builder.advance_row();
            }
            builder.finish()
        };

        // GQL and Cypher: a string is no number, so SUM fails (22G03)
        let mut agg = SimpleAggregateOperator::new(
            Box::new(MockOperator::new(vec![strings()])),
            vec![AggregateExpr::sum(0)],
            vec![LogicalType::Float64],
        );
        let error = agg.next().unwrap_err().to_string();
        assert!(error.contains("sum takes numbers or durations"), "{error}");

        // SPARQL: an operand that is no number is an error, and the sum is
        // unbound (the RDF planner gives SUM the numbers of its terms).
        let mut sparql_sum = AggregateExpr::sum(0);
        sparql_sum.rdf_literals = true;
        let mut agg = SimpleAggregateOperator::new(
            Box::new(MockOperator::new(vec![strings()])),
            vec![sparql_sum],
            vec![LogicalType::Float64],
        );
        let result = agg.next().unwrap().unwrap();
        assert_eq!(result.row_count(), 1);
        assert_eq!(result.column(0).unwrap().get_value(0), Some(Value::Null));
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
            temporal: TemporalComparison::TypesApart,
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
            temporal: TemporalComparison::TypesApart,
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
            temporal: TemporalComparison::TypesApart,
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
            temporal: TemporalComparison::TypesApart,
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
            temporal: TemporalComparison::TypesApart,
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
    fn min_and_max_of_rdf_terms_follow_the_sparql_order() {
        use grafeo_common::types::rdf::{Literal, Term};
        let term = |term: Term| Value::RdfTerm(std::sync::Arc::new(term));
        let nine = term(Term::typed_literal("9", Literal::XSD_INTEGER));
        let ten = term(Term::typed_literal("10", Literal::XSD_INTEGER));
        let iri = term(Term::iri("http://example.org/a"));
        assert_eq!(
            extremes(&[ten.clone(), iri.clone(), nine], false),
            [iri, ten]
        );
    }

    /// A SPARQL SUM or AVG meets an operand that is no number: the group's
    /// result is unbound, whatever comes after it.
    #[test]
    fn a_sparql_sum_of_a_term_that_is_no_number_is_unbound() {
        let text = Some(Value::String("9".into()));
        let one = Some(Value::Int64(1));
        assert_eq!(
            sum_and_avg(&[one.clone(), text, one], true, false),
            [Value::Null, Value::Null]
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
