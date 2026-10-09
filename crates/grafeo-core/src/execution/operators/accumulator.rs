//! Shared accumulator types for both pull-based and push-based aggregate operators.
//!
//! Provides the canonical definitions of [`AggregateFunction`], [`AggregateExpr`],
//! [`AggregateState`], and [`HashableValue`] used by both `aggregate.rs` (pull)
//! and `push/aggregate/mod.rs`.

// Re-export AggregateState so both pull and push operators import from one place.
pub use super::aggregate::AggregateState;

use grafeo_common::types::Value;

use crate::execution::DataChunk;

/// Aggregation function types.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
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
    /// Sample variance (VAR_SAMP / VARIANCE).
    Variance,
    /// Population variance (VAR_POP).
    VariancePop,
    /// Discrete percentile (PERCENTILE_DISC).
    PercentileDisc,
    /// Continuous percentile (PERCENTILE_CONT).
    PercentileCont,
    /// Concatenate values with separator (GROUP_CONCAT).
    GroupConcat,
    /// Return an arbitrary value from the group (SAMPLE).
    Sample,
    /// Sample covariance (COVAR_SAMP(y, x)).
    CovarSamp,
    /// Population covariance (COVAR_POP(y, x)).
    CovarPop,
    /// Pearson correlation coefficient (CORR(y, x)).
    Corr,
    /// Regression slope (REGR_SLOPE(y, x)).
    RegrSlope,
    /// Regression intercept (REGR_INTERCEPT(y, x)).
    RegrIntercept,
    /// Coefficient of determination (REGR_R2(y, x)).
    RegrR2,
    /// Regression count of non-null pairs (REGR_COUNT(y, x)).
    RegrCount,
    /// Regression sum of squares for x (REGR_SXX(y, x)).
    RegrSxx,
    /// Regression sum of squares for y (REGR_SYY(y, x)).
    RegrSyy,
    /// Regression sum of cross-products (REGR_SXY(y, x)).
    RegrSxy,
    /// Regression average of x (REGR_AVGX(y, x)).
    RegrAvgx,
    /// Regression average of y (REGR_AVGY(y, x)).
    RegrAvgy,
}

/// An aggregation expression.
#[derive(Debug, Clone)]
pub struct AggregateExpr {
    /// The aggregation function.
    pub function: AggregateFunction,
    /// Column index to aggregate (None for COUNT(*), y column for binary set functions).
    pub column: Option<usize>,
    /// Second column index for binary set functions (x column for COVAR, CORR, REGR_*).
    pub column2: Option<usize>,
    /// Whether to aggregate distinct values only.
    pub distinct: bool,
    /// Output alias (for naming the result column).
    pub alias: Option<String>,
    /// Percentile parameter for PERCENTILE_DISC/PERCENTILE_CONT (0.0 to 1.0).
    pub percentile: Option<f64>,
    /// Separator string for GROUP_CONCAT / LISTAGG.
    pub separator: Option<String>,
    /// Whether the operands are RDF literals held as text (SPARQL): MIN and
    /// MAX then compare two strings that read as numbers by those numbers.
    /// Otherwise MIN and MAX order their values as ORDER BY does.
    pub rdf_literals: bool,
}

impl AggregateExpr {
    /// Creates a COUNT(*) expression.
    pub fn count_star() -> Self {
        Self {
            function: AggregateFunction::Count,
            column: None,
            column2: None,
            distinct: false,
            alias: None,
            percentile: None,
            separator: None,
            rdf_literals: false,
        }
    }

    /// Creates a COUNT(column) expression.
    pub fn count(column: usize) -> Self {
        Self {
            function: AggregateFunction::CountNonNull,
            column: Some(column),
            column2: None,
            distinct: false,
            alias: None,
            percentile: None,
            separator: None,
            rdf_literals: false,
        }
    }

    /// Creates a SUM(column) expression.
    pub fn sum(column: usize) -> Self {
        Self {
            function: AggregateFunction::Sum,
            column: Some(column),
            column2: None,
            distinct: false,
            alias: None,
            percentile: None,
            separator: None,
            rdf_literals: false,
        }
    }

    /// Creates an AVG(column) expression.
    pub fn avg(column: usize) -> Self {
        Self {
            function: AggregateFunction::Avg,
            column: Some(column),
            column2: None,
            distinct: false,
            alias: None,
            percentile: None,
            separator: None,
            rdf_literals: false,
        }
    }

    /// Creates a MIN(column) expression.
    pub fn min(column: usize) -> Self {
        Self {
            function: AggregateFunction::Min,
            column: Some(column),
            column2: None,
            distinct: false,
            alias: None,
            percentile: None,
            separator: None,
            rdf_literals: false,
        }
    }

    /// Creates a MAX(column) expression.
    pub fn max(column: usize) -> Self {
        Self {
            function: AggregateFunction::Max,
            column: Some(column),
            column2: None,
            distinct: false,
            alias: None,
            percentile: None,
            separator: None,
            rdf_literals: false,
        }
    }

    /// Creates a FIRST(column) expression.
    pub fn first(column: usize) -> Self {
        Self {
            function: AggregateFunction::First,
            column: Some(column),
            column2: None,
            distinct: false,
            alias: None,
            percentile: None,
            separator: None,
            rdf_literals: false,
        }
    }

    /// Creates a LAST(column) expression.
    pub fn last(column: usize) -> Self {
        Self {
            function: AggregateFunction::Last,
            column: Some(column),
            column2: None,
            distinct: false,
            alias: None,
            percentile: None,
            separator: None,
            rdf_literals: false,
        }
    }

    /// Creates a COLLECT(column) expression.
    pub fn collect(column: usize) -> Self {
        Self {
            function: AggregateFunction::Collect,
            column: Some(column),
            column2: None,
            distinct: false,
            alias: None,
            percentile: None,
            separator: None,
            rdf_literals: false,
        }
    }

    /// Creates a STDEV(column) expression (sample standard deviation).
    pub fn stdev(column: usize) -> Self {
        Self {
            function: AggregateFunction::StdDev,
            column: Some(column),
            column2: None,
            distinct: false,
            alias: None,
            percentile: None,
            separator: None,
            rdf_literals: false,
        }
    }

    /// Creates a STDEVP(column) expression (population standard deviation).
    pub fn stdev_pop(column: usize) -> Self {
        Self {
            function: AggregateFunction::StdDevPop,
            column: Some(column),
            column2: None,
            distinct: false,
            alias: None,
            percentile: None,
            separator: None,
            rdf_literals: false,
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
            column2: None,
            distinct: false,
            alias: None,
            percentile: Some(percentile.clamp(0.0, 1.0)),
            separator: None,
            rdf_literals: false,
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
            column2: None,
            distinct: false,
            alias: None,
            percentile: Some(percentile.clamp(0.0, 1.0)),
            separator: None,
            rdf_literals: false,
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

/// The identity of a value under equivalence: what grouping keys, DISTINCT
/// and UNION compare.
///
/// Two values have the same identity when they are the same value, the
/// equivalence of openCypher and the "not distinct" of ISO/IEC 39075:2024:
/// numbers by their value, so `3` and `3.0` are one and `-0.0` is `0.0`; NaN
/// is NaN; lists, maps, paths and vectors item by item; zoned datetimes and
/// times with an offset by their instant; every other value as `=` compares
/// it. Nodes and edges are their IDs, or maps that hold their IDs.
///
/// The identity holds the value's [representative](Self::representative):
/// one value for all the values it stands for.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct HashableValue(grafeo_common::types::HashableValue);

impl HashableValue {
    /// The value that stands for every value of this identity: an integral
    /// float in the range of `i64` as that integer, NaN as one NaN, a zoned
    /// datetime or a time with an offset at UTC, and lists, maps, paths and
    /// vectors of representatives. Two values have the same identity exactly
    /// when their representatives are equal and serialize to the same bytes
    /// (counters aside, whose replicas serialize in any order).
    #[must_use]
    pub fn representative(&self) -> &Value {
        self.0.inner()
    }
}

impl From<&Value> for HashableValue {
    fn from(value: &Value) -> Self {
        Self(grafeo_common::types::HashableValue(representative(value)))
    }
}

impl From<Value> for HashableValue {
    fn from(value: Value) -> Self {
        let representative = match value {
            Value::Float64(float) => representative_float(float),
            Value::Vector(_)
            | Value::ZonedDatetime(_)
            | Value::Time(_)
            | Value::List(_)
            | Value::Map(_)
            | Value::Path { .. } => representative(&value),
            // Every other value stands for itself: no copy.
            other => other,
        };
        Self(grafeo_common::types::HashableValue(representative))
    }
}

/// A grouping, DISTINCT or UNION key: the identity of each of a row's values
/// in the key columns (see [`HashableValue`]), so the rows whose key values
/// are the same values have one key.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub(crate) struct RowKey(Vec<HashableValue>);

impl RowKey {
    /// The key of `values`.
    pub(crate) fn of(values: &[Value]) -> Self {
        Self(values.iter().map(HashableValue::from).collect())
    }

    /// The values of `row` in `columns`, a missing one as null.
    pub(crate) fn values_of(chunk: &DataChunk, row: usize, columns: &[usize]) -> Vec<Value> {
        columns
            .iter()
            .map(|&column| {
                chunk
                    .column(column)
                    .and_then(|values| values.get_value(row))
                    .unwrap_or(Value::Null)
            })
            .collect()
    }

    /// The key of `row` in `columns`.
    pub(crate) fn from_row(chunk: &DataChunk, row: usize, columns: &[usize]) -> Self {
        Self(
            columns
                .iter()
                .map(|&column| Self::identity(chunk, row, column))
                .collect(),
        )
    }

    /// The key of `row` in every column of `chunk`.
    pub(crate) fn from_all_columns(chunk: &DataChunk, row: usize) -> Self {
        Self(
            (0..chunk.column_count())
                .map(|column| Self::identity(chunk, row, column))
                .collect(),
        )
    }

    /// The identity of the value of `row` in `column`, a missing one as null.
    fn identity(chunk: &DataChunk, row: usize, column: usize) -> HashableValue {
        HashableValue::from(
            chunk
                .column(column)
                .and_then(|values| values.get_value(row))
                .unwrap_or(Value::Null),
        )
    }

    /// The representatives of the key's values: equal keys have equal
    /// representatives that serialize to the same bytes, so a spilled
    /// partition can file groups under them.
    pub(crate) fn representatives(&self) -> Vec<Value> {
        self.0
            .iter()
            .map(|identity| identity.representative().clone())
            .collect()
    }
}

/// The representative of `value`'s identity (see [`HashableValue`]).
fn representative(value: &Value) -> Value {
    match value {
        Value::Float64(float) => representative_float(*float),
        Value::Vector(items) => {
            Value::Vector(items.iter().map(|&x| representative_f32(x)).collect())
        }
        Value::ZonedDatetime(datetime) => Value::ZonedDatetime(
            grafeo_common::types::ZonedDatetime::from_timestamp_offset(datetime.as_timestamp(), 0),
        ),
        Value::Time(time) => Value::Time(representative_time(*time)),
        Value::List(items) => Value::List(items.iter().map(representative).collect()),
        Value::Map(map) => Value::Map(std::sync::Arc::new(
            map.iter()
                .map(|(key, item)| (key.clone(), representative(item)))
                .collect(),
        )),
        Value::Path { nodes, edges } => Value::Path {
            nodes: nodes.iter().map(representative).collect(),
            edges: edges.iter().map(representative).collect(),
        },
        other => other.clone(),
    }
}

/// A float's representative: the integer it equals when it is integral and
/// in the range of `i64` (`3.0` is `3`, `-0.0` is `0`), one NaN for every
/// NaN, else the float itself. An integer converted to a float would round
/// above 2^53 and take `2^53 + 1` for `2^53`; the float converted to the
/// integer is exact.
fn representative_float(float: f64) -> Value {
    // 2^63, the smallest float above every i64.
    const BEYOND_I64: f64 = 9_223_372_036_854_775_808.0;
    if float.is_nan() {
        return Value::Float64(f64::NAN);
    }
    let whole = float.trunc();
    if whole == float && (-BEYOND_I64..BEYOND_I64).contains(&whole) {
        #[allow(
            clippy::cast_possible_truncation,
            reason = "`whole` is an integer in [-2^63, 2^63), checked above, so the cast is exact"
        )]
        let integer = whole as i64;
        return Value::Int64(integer);
    }
    Value::Float64(float)
}

/// A vector item's representative: one NaN for every NaN, and `0.0` for
/// `-0.0`.
fn representative_f32(item: f32) -> f32 {
    if item.is_nan() {
        f32::NAN
    } else if item == 0.0 {
        // `-0.0 == 0.0`: both become `0.0`.
        0.0
    } else {
        item
    }
}

/// A time's representative: a time with an offset at UTC (`14:00+01:00` is
/// `13:00Z`), a local time as it is.
fn representative_time(time: grafeo_common::types::Time) -> grafeo_common::types::Time {
    const NANOS_PER_DAY: i64 = 86_400 * 1_000_000_000;
    let at_utc = || {
        let offset = time.offset_seconds()?;
        let local = i64::try_from(time.as_nanos()).ok()?;
        let utc = (local - i64::from(offset) * 1_000_000_000).rem_euclid(NANOS_PER_DAY);
        Some(grafeo_common::types::Time::from_nanos(u64::try_from(utc).ok()?)?.with_offset(0))
    };
    at_utc().unwrap_or(time)
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;
    use std::sync::Arc;

    use grafeo_common::types::{PropertyKey, Time, Timestamp, ZonedDatetime};

    use super::*;

    fn same(a: &Value, b: &Value) -> bool {
        HashableValue::from(a) == HashableValue::from(b)
    }

    fn list(items: Vec<Value>) -> Value {
        Value::List(items.into())
    }

    fn path(nodes: Vec<Value>, edges: Vec<Value>) -> Value {
        Value::Path {
            nodes: nodes.into(),
            edges: edges.into(),
        }
    }

    /// Numbers are one identity by their value (openCypher equivalence): an
    /// integer and a float of one value, `-0.0` and `0.0`, and every NaN.
    #[test]
    fn numbers_have_one_identity_per_value() {
        assert!(same(&Value::Int64(3), &Value::Float64(3.0)));
        assert!(same(&Value::Float64(-0.0), &Value::Float64(0.0)));
        assert!(same(&Value::Float64(-0.0), &Value::Int64(0)));
        assert!(same(
            &Value::Float64(f64::NAN),
            &Value::Float64(f64::from_bits(f64::NAN.to_bits() | 1))
        ));
        assert!(same(
            &Value::Float64(-9_223_372_036_854_775_808.0),
            &Value::Int64(i64::MIN)
        ));
        assert!(!same(&Value::Float64(1.5), &Value::Int64(1)));
        assert!(!same(&Value::Float64(f64::NAN), &Value::Int64(0)));
        // An integer must not be rounded to a float to compare: 2^53 + 1 is
        // not the float 2^53, and 2^63 is no i64.
        assert!(!same(
            &Value::Int64(9_007_199_254_740_993),
            &Value::Float64(9_007_199_254_740_992.0)
        ));
        assert!(!same(
            &Value::Float64(9_223_372_036_854_775_808.0),
            &Value::Int64(i64::MAX)
        ));
    }

    /// Lists, maps, paths and vectors compare item by item; a path is not
    /// the same as another path of its length, and two byte strings or vectors
    /// that share their first item and length are not one.
    #[test]
    fn composite_values_compare_item_by_item() {
        assert!(same(
            &list(vec![Value::Int64(3), Value::Float64(1.5)]),
            &list(vec![Value::Float64(3.0), Value::Float64(1.5)])
        ));
        assert!(!same(
            &list(vec![Value::Int64(3)]),
            &list(vec![Value::Int64(3), Value::Int64(3)])
        ));
        let map =
            |value: Value| Value::Map(Arc::new(BTreeMap::from([(PropertyKey::new("k"), value)])));
        assert!(same(&map(Value::Int64(3)), &map(Value::Float64(3.0))));
        assert!(!same(&map(Value::Int64(3)), &map(Value::Int64(19))));
        assert!(same(
            &path(
                vec![Value::Int64(1), Value::Int64(2)],
                vec![Value::Int64(10)]
            ),
            &path(
                vec![Value::Int64(1), Value::Int64(2)],
                vec![Value::Int64(10)]
            )
        ));
        assert!(!same(
            &path(
                vec![Value::Int64(1), Value::Int64(2)],
                vec![Value::Int64(10)]
            ),
            &path(
                vec![Value::Int64(1), Value::Int64(3)],
                vec![Value::Int64(11)]
            )
        ));
        assert!(!same(
            &Value::Vector(vec![1.0_f32, 2.0].into()),
            &Value::Vector(vec![1.0_f32, 3.0].into())
        ));
        assert!(same(
            &Value::Vector(vec![-0.0_f32, f32::NAN].into()),
            &Value::Vector(vec![0.0_f32, f32::from_bits(f32::NAN.to_bits() | 1)].into())
        ));
        assert!(!same(
            &Value::Bytes(vec![3, 19].into()),
            &Value::Bytes(vec![3, 88].into())
        ));
    }

    /// Values of different types are different values, numbers aside.
    #[test]
    fn values_of_different_types_have_different_identities() {
        assert!(!same(&Value::String("3".into()), &Value::Int64(3)));
        assert!(!same(&Value::Bool(true), &Value::Int64(1)));
        assert!(!same(&list(vec![]), &Value::Map(Arc::new(BTreeMap::new()))));
        assert!(!same(&Value::Null, &Value::Int64(0)));
        assert!(!same(
            &list(vec![Value::Float64(1.0)]),
            &Value::Vector(vec![1.0_f32].into())
        ));
    }

    /// Zoned datetimes and times with an offset are their instant, as `=`
    /// compares them; a local time is not a time at UTC.
    #[test]
    fn zoned_datetimes_and_times_are_their_instant() {
        let instant = Timestamp::from_micros(1_700_000_000_000_000);
        assert!(same(
            &Value::ZonedDatetime(ZonedDatetime::from_timestamp_offset(instant, 3600)),
            &Value::ZonedDatetime(ZonedDatetime::from_timestamp_offset(instant, -7200))
        ));
        let at = |hour| Time::from_hms(hour, 0, 0).unwrap();
        assert!(same(
            &Value::Time(at(14).with_offset(3600)),
            &Value::Time(at(13).with_offset(0))
        ));
        assert!(same(
            &Value::Time(at(0).with_offset(3600)),
            &Value::Time(at(23).with_offset(0))
        ));
        assert!(!same(
            &Value::Time(at(13)),
            &Value::Time(at(13).with_offset(0))
        ));
    }

    /// The representatives of one identity serialize to the same bytes, so
    /// a spilled aggregate partition files `3` and `3.0` under one group.
    #[cfg(feature = "spill")]
    #[test]
    fn representatives_of_one_identity_serialize_alike() {
        use crate::execution::spill::serialize_value;

        let bytes = |value: &Value| {
            let mut bytes = Vec::new();
            serialize_value(HashableValue::from(value).representative(), &mut bytes).unwrap();
            bytes
        };
        let instant = Timestamp::from_micros(1_700_000_000_000_000);
        for (a, b) in [
            (Value::Int64(3), Value::Float64(3.0)),
            (Value::Float64(-0.0), Value::Float64(0.0)),
            (
                Value::Float64(f64::NAN),
                Value::Float64(f64::from_bits(f64::NAN.to_bits() | 1)),
            ),
            (
                list(vec![Value::Int64(19), Value::Float64(-0.0)]),
                list(vec![Value::Float64(19.0), Value::Int64(0)]),
            ),
            (
                Value::ZonedDatetime(ZonedDatetime::from_timestamp_offset(instant, 3600)),
                Value::ZonedDatetime(ZonedDatetime::from_timestamp_offset(instant, 0)),
            ),
        ] {
            assert_eq!(bytes(&a), bytes(&b), "{a:?} and {b:?}");
        }
    }

    /// A row key compares each column's value by its identity.
    #[test]
    fn row_keys_compare_their_values_by_identity() {
        assert_eq!(
            RowKey::of(&[Value::Int64(3), Value::String("Alix".into())]),
            RowKey::of(&[Value::Float64(3.0), Value::String("Alix".into())])
        );
        assert_ne!(
            RowKey::of(&[Value::Int64(3), Value::String("Alix".into())]),
            RowKey::of(&[Value::Int64(3), Value::String("Gus".into())])
        );
        assert_eq!(
            RowKey::of(&[Value::Float64(3.0)]).representatives(),
            [Value::Int64(3)]
        );
    }
}
