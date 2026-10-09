//! Shared value comparison and conversion utilities.
//!
//! Used by pull-based and push-based aggregate, filter, and sort operators
//! to avoid duplicating comparison logic across six different files.

use std::cmp::Ordering;
use std::collections::HashMap;

use grafeo_common::types::Value;

use super::sort::{NullOrder, SortDirection};

/// Converts a value to `f64` for numeric aggregations.
///
/// Supports RDF values stored as strings by attempting numeric parsing.
pub fn value_to_f64(value: &Value) -> Option<f64> {
    match value {
        Value::Int64(i) => Some(*i as f64),
        Value::Float64(f) => Some(*f),
        // RDF stores numeric literals as strings - try to parse them
        Value::String(s) => s.parse::<f64>().ok(),
        _ => None,
    }
}

/// Compares two values with partial ordering (returns `None` for incomparable types).
///
/// Handles cross-type comparisons between Int64/Float64/String, including
/// RDF numeric strings that need parsing before comparison.
pub fn compare_values(a: &Value, b: &Value) -> Option<Ordering> {
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
        (Value::Timestamp(a), Value::Timestamp(b)) => Some(a.cmp(b)),
        (Value::Date(a), Value::Date(b)) => Some(a.cmp(b)),
        (Value::Time(a), Value::Time(b)) => Some(a.cmp(b)),
        _ => None,
    }
}

/// Compares two values for `ORDER BY`: a total order over every value, the
/// orderability of openCypher.
///
/// Values of different types order by type, ascending: maps, lists, paths,
/// vectors, zoned datetimes, datetimes, dates, zoned times, local times,
/// durations, bytes, strings, booleans, counters, numbers, then null. A sort
/// key's own nulls do not get here: they go first or last as its
/// `NULLS FIRST` / `NULLS LAST` says (see [`compare_sort_values`]); null
/// inside a list or map is larger than any other value.
///
/// Within a type: numbers numerically, integers and floats exactly, with NaN
/// after positive infinity; strings by code point; `false` before `true`;
/// temporal values in time order; durations by months, then days, then
/// nanoseconds; lists and paths element by element, a prefix first; maps by
/// size, then their keys, then their values; vectors and bytes element by
/// element; counters by their value.
#[must_use]
pub fn compare_values_total(a: &Value, b: &Value) -> Ordering {
    let by_type = type_rank(a).cmp(&type_rank(b));
    if by_type.is_ne() {
        return by_type;
    }
    match (a, b) {
        (Value::Bool(a), Value::Bool(b)) => a.cmp(b),
        (Value::Int64(a), Value::Int64(b)) => a.cmp(b),
        (Value::Float64(a), Value::Float64(b)) => compare_floats(*a, *b),
        (Value::Int64(a), Value::Float64(b)) => compare_int_float(*a, *b),
        (Value::Float64(a), Value::Int64(b)) => compare_int_float(*b, *a).reverse(),
        (Value::String(a), Value::String(b)) => a.cmp(b),
        (Value::Bytes(a), Value::Bytes(b)) => a.cmp(b),
        (Value::Timestamp(a), Value::Timestamp(b)) => a.cmp(b),
        (Value::Date(a), Value::Date(b)) => a.cmp(b),
        (Value::Time(a), Value::Time(b)) => a.cmp(b),
        (Value::ZonedDatetime(a), Value::ZonedDatetime(b)) => a.cmp(b),
        (Value::Duration(a), Value::Duration(b)) => {
            (a.months(), a.days(), a.nanos()).cmp(&(b.months(), b.days(), b.nanos()))
        }
        (Value::List(a), Value::List(b)) => compare_sequences(a.iter(), b.iter()),
        (
            Value::Path {
                nodes: a_nodes,
                edges: a_edges,
            },
            Value::Path {
                nodes: b_nodes,
                edges: b_edges,
            },
        ) => compare_sequences(
            path_elements(a_nodes, a_edges),
            path_elements(b_nodes, b_edges),
        ),
        (Value::Map(a), Value::Map(b)) => a
            .len()
            .cmp(&b.len())
            .then_with(|| a.keys().cmp(b.keys()))
            .then_with(|| compare_sequences(a.values(), b.values())),
        (Value::Vector(a), Value::Vector(b)) => a
            .iter()
            .zip(b.iter())
            .map(|(x, y)| compare_floats(f64::from(*x), f64::from(*y)))
            .find(|order| order.is_ne())
            .unwrap_or_else(|| a.len().cmp(&b.len())),
        (Value::GCounter(_) | Value::OnCounter { .. }, _) => {
            counter_value(a).cmp(&counter_value(b))
        }
        // Values of the same rank not matched above are equal: nulls, and
        // values of a type added later.
        _ => Ordering::Equal,
    }
}

/// The position of a value's type in the order across types.
fn type_rank(value: &Value) -> u8 {
    match value {
        Value::Map(_) => 0,
        Value::List(_) => 1,
        Value::Path { .. } => 2,
        Value::Vector(_) => 3,
        Value::ZonedDatetime(_) => 4,
        Value::Timestamp(_) => 5,
        Value::Date(_) => 6,
        Value::Time(time) if time.offset_seconds().is_some() => 7,
        Value::Time(_) => 8,
        Value::Duration(_) => 9,
        Value::Bytes(_) => 10,
        Value::String(_) => 11,
        Value::Bool(_) => 12,
        Value::GCounter(_) | Value::OnCounter { .. } => 13,
        Value::Int64(_) | Value::Float64(_) => 15,
        Value::Null => 16,
        // A type added later goes before the numbers: openCypher keeps types
        // it does not define out from after them.
        _ => 14,
    }
}

/// Compares two floats numerically, with NaN after every other number and
/// equal to itself; `-0.0` equals `0.0`. A total order, so floats can be
/// sorted with it (`partial_cmp` with NaN as equal is not: a sort may panic).
pub(crate) fn compare_floats(a: f64, b: f64) -> Ordering {
    match (a.is_nan(), b.is_nan()) {
        (true, true) => Ordering::Equal,
        (true, false) => Ordering::Greater,
        (false, true) => Ordering::Less,
        // Without NaN, `partial_cmp` always has an answer.
        (false, false) => a.partial_cmp(&b).unwrap_or(Ordering::Equal),
    }
}

/// Compares an integer with a float exactly. Converting the integer to a
/// float rounds above 2^53, which would make `2^53 + 1` equal to the float
/// `2^53` while the integers `2^53` and `2^53 + 1` differ. NaN is after every
/// integer.
fn compare_int_float(int: i64, float: f64) -> Ordering {
    // 2^63, the smallest float above every i64.
    const BEYOND_I64: f64 = 9_223_372_036_854_775_808.0;
    if float.is_nan() || float >= BEYOND_I64 {
        return Ordering::Less;
    }
    if float < -BEYOND_I64 {
        return Ordering::Greater;
    }
    let whole = float.trunc();
    #[allow(
        clippy::cast_possible_truncation,
        reason = "`whole` is an integer in [-2^63, 2^63), checked above, so the cast is exact"
    )]
    let whole_int = whole as i64;
    // On the same whole part, the float's fraction decides.
    int.cmp(&whole_int)
        .then_with(|| whole.partial_cmp(&float).unwrap_or(Ordering::Equal))
}

/// Compares two sequences element by element; a prefix comes first.
fn compare_sequences<'a>(
    a: impl IntoIterator<Item = &'a Value>,
    b: impl IntoIterator<Item = &'a Value>,
) -> Ordering {
    let mut b = b.into_iter();
    for x in a {
        let Some(y) = b.next() else {
            return Ordering::Greater;
        };
        let order = compare_values_total(x, y);
        if order.is_ne() {
            return order;
        }
    }
    if b.next().is_some() {
        Ordering::Less
    } else {
        Ordering::Equal
    }
}

/// The nodes and edges of a path, alternating from its first node.
fn path_elements<'a>(nodes: &'a [Value], edges: &'a [Value]) -> impl Iterator<Item = &'a Value> {
    nodes
        .iter()
        .enumerate()
        .flat_map(move |(index, node)| std::iter::once(node).chain(edges.get(index)))
}

/// The value of a counter.
fn counter_value(value: &Value) -> i128 {
    let total = |counts: &HashMap<String, u64>| {
        i128::try_from(
            counts
                .values()
                .map(|&count| u128::from(count))
                .sum::<u128>(),
        )
        .unwrap_or(i128::MAX)
    };
    match value {
        Value::GCounter(counts) => total(counts),
        Value::OnCounter { pos, neg } => total(pos).saturating_sub(total(neg)),
        _ => 0,
    }
}

/// Orders the values of one sort key: nulls (and missing values) first or
/// last as `null_order` says, in either direction, and the other values by
/// [`compare_values_total`] in `direction`.
///
/// Used by the sort and top-K operators for `ORDER BY ... [ASC | DESC]
/// [NULLS FIRST | NULLS LAST]`.
#[must_use]
pub fn compare_sort_values(
    a: Option<&Value>,
    b: Option<&Value>,
    direction: SortDirection,
    null_order: NullOrder,
) -> Ordering {
    order_by(
        a,
        b,
        direction == SortDirection::Descending,
        null_order == NullOrder::NullsFirst,
    )
}

/// [`compare_sort_values`] with the direction and the null order as flags, for
/// the sorts that have sort keys of their own.
pub(crate) fn order_by(
    a: Option<&Value>,
    b: Option<&Value>,
    descending: bool,
    nulls_first: bool,
) -> Ordering {
    let nulls = if nulls_first {
        Ordering::Less
    } else {
        Ordering::Greater
    };
    match (
        a.filter(|value| !value.is_null()),
        b.filter(|value| !value.is_null()),
    ) {
        (None, None) => Ordering::Equal,
        (None, Some(_)) => nulls,
        (Some(_), None) => nulls.reverse(),
        (Some(a), Some(b)) => {
            let order = compare_values_total(a, b);
            if descending { order.reverse() } else { order }
        }
    }
}

/// Returns `true` if `new` is less than `current` in the order of ORDER BY
/// ([`compare_values_total`]), as MIN compares values of every type.
///
/// Returns `true` when `current` is `None` (first value always wins).
pub fn is_less_than(current: &Option<Value>, new: &Value) -> bool {
    current
        .as_ref()
        .is_none_or(|current| compare_values_total(new, current).is_lt())
}

/// Returns `true` if `new` is greater than `current` in the order of ORDER
/// BY ([`compare_values_total`]), as MAX compares values of every type.
///
/// Returns `true` when `current` is `None` (first value always wins).
pub fn is_greater_than(current: &Option<Value>, new: &Value) -> bool {
    current
        .as_ref()
        .is_none_or(|current| compare_values_total(new, current).is_gt())
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;
    use std::sync::Arc;

    use grafeo_common::types::{Date, Duration, PropertyKey, Time, Timestamp, ZonedDatetime};

    use super::*;

    #[test]
    fn value_to_f64_int() {
        assert_eq!(value_to_f64(&Value::Int64(42)), Some(42.0));
    }

    #[test]
    fn value_to_f64_float() {
        assert_eq!(value_to_f64(&Value::Float64(2.72)), Some(2.72));
    }

    #[test]
    fn value_to_f64_numeric_string() {
        assert_eq!(value_to_f64(&Value::String("2.5".into())), Some(2.5));
    }

    #[test]
    fn value_to_f64_non_numeric_string() {
        assert_eq!(value_to_f64(&Value::String("abc".into())), None);
    }

    #[test]
    fn value_to_f64_null() {
        assert_eq!(value_to_f64(&Value::Null), None);
    }

    #[test]
    fn compare_same_type_int() {
        assert_eq!(
            compare_values(&Value::Int64(1), &Value::Int64(2)),
            Some(Ordering::Less)
        );
    }

    #[test]
    fn compare_cross_type_int_float() {
        assert_eq!(
            compare_values(&Value::Int64(2), &Value::Float64(2.0)),
            Some(Ordering::Equal)
        );
    }

    #[test]
    fn compare_rdf_numeric_strings() {
        assert_eq!(
            compare_values(&Value::String("10".into()), &Value::String("9".into())),
            Some(Ordering::Greater)
        );
    }

    #[test]
    fn compare_incomparable() {
        assert_eq!(compare_values(&Value::Bool(true), &Value::Int64(1)), None);
    }

    #[test]
    fn is_less_than_none_always_true() {
        assert!(is_less_than(&None, &Value::Int64(5)));
    }

    #[test]
    fn is_less_than_smaller() {
        assert!(is_less_than(&Some(Value::Int64(10)), &Value::Int64(5)));
    }

    #[test]
    fn is_less_than_larger() {
        assert!(!is_less_than(&Some(Value::Int64(3)), &Value::Int64(5)));
    }

    #[test]
    fn is_greater_than_none_always_true() {
        assert!(is_greater_than(&None, &Value::Int64(5)));
    }

    #[test]
    fn is_greater_than_larger() {
        assert!(is_greater_than(&Some(Value::Int64(3)), &Value::Int64(5)));
    }

    #[test]
    fn is_greater_than_smaller() {
        assert!(!is_greater_than(&Some(Value::Int64(10)), &Value::Int64(5)));
    }

    fn list(values: Vec<Value>) -> Value {
        Value::List(values.into())
    }

    fn map(entries: &[(&str, Value)]) -> Value {
        let map: BTreeMap<PropertyKey, Value> = entries
            .iter()
            .map(|(key, value)| (PropertyKey::new(*key), value.clone()))
            .collect();
        Value::Map(Arc::new(map))
    }

    fn time(hour: u32) -> Time {
        Time::from_hms(hour, 0, 0).unwrap()
    }

    /// One value of each type, in ascending order.
    fn one_of_each_type() -> Vec<Value> {
        vec![
            map(&[("k", Value::Int64(1))]),
            list(vec![Value::Int64(1)]),
            Value::Path {
                nodes: vec![Value::Int64(0), Value::Int64(1)].into(),
                edges: vec![Value::Int64(0)].into(),
            },
            Value::Vector(vec![1.0_f32, 2.0].into()),
            Value::ZonedDatetime(ZonedDatetime::from_timestamp_offset(
                Timestamp::from_secs(0),
                3600,
            )),
            Value::Timestamp(Timestamp::from_secs(0)),
            Value::Date(Date::from_ymd(2024, 1, 15).unwrap()),
            Value::Time(time(12).with_offset(3600)),
            Value::Time(time(12)),
            Value::Duration(Duration::new(1, 0, 0)),
            Value::Bytes(vec![1_u8, 2].into()),
            Value::String("a".into()),
            Value::Bool(false),
            Value::GCounter(Arc::new(HashMap::from([("r".to_string(), 3_u64)]))),
            Value::Int64(1),
            Value::Null,
        ]
    }

    /// Sorts `values` with `compare_values_total`; also checks that the
    /// result is in order, pair by pair.
    fn sorted(mut values: Vec<Value>) -> Vec<Value> {
        values.sort_by(compare_values_total);
        for pair in values.windows(2) {
            assert_ne!(
                compare_values_total(&pair[0], &pair[1]),
                Ordering::Greater,
                "{:?} before {:?}",
                pair[0],
                pair[1]
            );
        }
        values
    }

    #[test]
    fn values_of_different_types_follow_the_opencypher_order() {
        let ascending = one_of_each_type();
        for rotation in 0..ascending.len() {
            let mut shuffled = ascending.clone();
            shuffled.rotate_left(rotation);
            shuffled.reverse();
            assert_eq!(sorted(shuffled), ascending, "rotation {rotation}");
        }
    }

    #[test]
    fn integers_and_floats_compare_exactly() {
        let two_53 = 9_007_199_254_740_992_i64;
        let cases = [
            (
                Value::Int64(two_53),
                Value::Float64(2f64.powi(53)),
                Ordering::Equal,
            ),
            (
                Value::Int64(two_53 + 1),
                Value::Float64(2f64.powi(53)),
                Ordering::Greater,
            ),
            (Value::Int64(2), Value::Float64(2.5), Ordering::Less),
            (Value::Int64(3), Value::Float64(2.5), Ordering::Greater),
            (Value::Int64(-2), Value::Float64(-2.5), Ordering::Greater),
            (Value::Int64(-3), Value::Float64(-2.5), Ordering::Less),
            (Value::Int64(0), Value::Float64(-0.0), Ordering::Equal),
            (Value::Float64(-0.0), Value::Float64(0.0), Ordering::Equal),
            (
                Value::Int64(i64::MAX),
                Value::Float64(2f64.powi(63)),
                Ordering::Less,
            ),
            (
                Value::Int64(i64::MIN),
                Value::Float64(-(2f64.powi(63))),
                Ordering::Equal,
            ),
            (
                Value::Int64(i64::MAX),
                Value::Float64(f64::INFINITY),
                Ordering::Less,
            ),
            (
                Value::Int64(i64::MIN),
                Value::Float64(f64::NEG_INFINITY),
                Ordering::Greater,
            ),
        ];
        for (a, b, expected) in cases {
            assert_eq!(compare_values_total(&a, &b), expected, "{a:?} vs {b:?}");
            assert_eq!(
                compare_values_total(&b, &a),
                expected.reverse(),
                "{b:?} vs {a:?}"
            );
        }
    }

    #[test]
    fn nan_is_the_largest_number_and_equals_itself() {
        let nan = Value::Float64(f64::NAN);
        assert_eq!(
            compare_values_total(&nan, &Value::Float64(f64::INFINITY)),
            Ordering::Greater
        );
        assert_eq!(
            compare_values_total(&nan, &Value::Int64(i64::MAX)),
            Ordering::Greater
        );
        assert_eq!(compare_values_total(&nan, &nan), Ordering::Equal);
        assert_eq!(
            sorted(vec![
                Value::Int64(1),
                nan.clone(),
                Value::Int64(-1),
                Value::Float64(f64::INFINITY),
                Value::Float64(f64::NEG_INFINITY),
            ])
            .iter()
            .map(|value| format!("{value:?}"))
            .collect::<Vec<_>>(),
            [
                "Float64(-inf)",
                "Int64(-1)",
                "Int64(1)",
                "Float64(inf)",
                "Float64(NaN)"
            ]
        );
    }

    /// The openCypher examples: element by element, a missing element
    /// before any value (null too), null after any value.
    #[test]
    fn lists_compare_element_by_element() {
        let cases = [
            (
                vec![Value::Int64(1)],
                vec![Value::Int64(1), Value::Int64(0)],
            ),
            (vec![Value::Int64(1)], vec![Value::Int64(1), Value::Null]),
            (
                vec![
                    Value::Int64(1),
                    Value::String("foo".into()),
                    Value::Int64(3),
                ],
                vec![
                    Value::Int64(1),
                    Value::Int64(2),
                    Value::String("bar".into()),
                ],
            ),
            (
                vec![Value::Int64(1), Value::Int64(2)],
                vec![Value::Null, Value::Int64(1)],
            ),
            (
                vec![Value::Null, Value::Int64(1)],
                vec![Value::Null, Value::Int64(2)],
            ),
            (vec![], vec![Value::Null]),
        ];
        for (smaller, larger) in cases {
            let (smaller, larger) = (list(smaller), list(larger));
            assert_eq!(
                compare_values_total(&smaller, &larger),
                Ordering::Less,
                "{smaller:?} < {larger:?}"
            );
            assert_eq!(compare_values_total(&larger, &smaller), Ordering::Greater);
        }
    }

    #[test]
    fn maps_compare_by_size_then_keys_then_values() {
        let cases = [
            (
                map(&[("z", Value::Int64(9))]),
                map(&[("a", Value::Int64(1)), ("b", Value::Int64(1))]),
            ),
            (
                map(&[("a", Value::Int64(2))]),
                map(&[("b", Value::Int64(1))]),
            ),
            (
                map(&[("a", Value::Int64(1))]),
                map(&[("a", Value::Int64(2))]),
            ),
            (map(&[("a", Value::Int64(1))]), map(&[("a", Value::Null)])),
        ];
        for (smaller, larger) in cases {
            assert_eq!(
                compare_values_total(&smaller, &larger),
                Ordering::Less,
                "{smaller:?} < {larger:?}"
            );
            assert_eq!(compare_values_total(&larger, &smaller), Ordering::Greater);
        }
    }

    #[test]
    fn paths_compare_node_edge_node() {
        let path = |ids: &[i64]| Value::Path {
            nodes: ids.iter().step_by(2).map(|&id| Value::Int64(id)).collect(),
            edges: ids
                .iter()
                .skip(1)
                .step_by(2)
                .map(|&id| Value::Int64(id))
                .collect(),
        };
        // The openCypher example: n1 r1 n3 before n1 r2 n2, decided by the edge.
        assert_eq!(
            compare_values_total(&path(&[1, 1, 3]), &path(&[1, 2, 2])),
            Ordering::Less
        );
        assert_eq!(
            compare_values_total(&path(&[1]), &path(&[1, 1, 3])),
            Ordering::Less
        );
    }

    #[test]
    fn temporal_values_order_in_time_within_their_type() {
        assert_eq!(
            compare_values_total(
                &Value::Duration(Duration::new(1, 0, 0)),
                &Value::Duration(Duration::new(0, 40, 0))
            ),
            Ordering::Greater,
            "months before days"
        );
        assert_eq!(
            compare_values_total(
                &Value::Duration(Duration::new(0, 1, 5)),
                &Value::Duration(Duration::new(0, 1, 7))
            ),
            Ordering::Less
        );
        // A zoned time before every local time, whatever the clock says.
        assert_eq!(
            compare_values_total(&Value::Time(time(23).with_offset(0)), &Value::Time(time(1))),
            Ordering::Less
        );
        assert_eq!(
            compare_values_total(
                &Value::Time(time(9).with_offset(3600)),
                &Value::Time(time(9).with_offset(0))
            ),
            Ordering::Less,
            "09:00+01:00 is 08:00 UTC"
        );
    }

    /// The order is total on a mix of every type and the edge cases of each:
    /// antisymmetric, transitive, and `sort_by` gives a sorted result.
    /// Comparing values of different types as equal broke transitivity, and
    /// `slice::sort_by` may panic on such a comparator.
    #[test]
    fn the_order_is_total_on_mixed_values() {
        let mut pool = one_of_each_type();
        pool.extend([
            Value::Int64(-3),
            Value::Int64(9_007_199_254_740_993),
            Value::Float64(9_007_199_254_740_992.0),
            Value::Float64(2.5),
            Value::Float64(-0.0),
            Value::Float64(f64::NAN),
            Value::Float64(f64::INFINITY),
            Value::String(String::new().into()),
            Value::String("b".into()),
            Value::Bool(true),
            list(vec![]),
            list(vec![Value::Null]),
            list(vec![Value::Int64(1), Value::Null]),
            list(vec![Value::String("x".into())]),
            map(&[]),
            map(&[("a", Value::Null)]),
            Value::Vector(vec![f32::NAN].into()),
            Value::Bytes(vec![].into()),
            Value::OnCounter {
                pos: Arc::new(HashMap::from([("r".to_string(), 1_u64)])),
                neg: Arc::new(HashMap::from([("r".to_string(), 5_u64)])),
            },
        ]);
        for a in &pool {
            for b in &pool {
                let order = compare_values_total(a, b);
                assert_eq!(
                    compare_values_total(b, a),
                    order.reverse(),
                    "{a:?} vs {b:?}"
                );
                for c in &pool {
                    if order.is_le() && compare_values_total(b, c).is_le() {
                        assert!(
                            compare_values_total(a, c).is_le(),
                            "{a:?} <= {b:?} <= {c:?}"
                        );
                    }
                }
            }
        }
        // A long mix in a scrambled order sorts without panicking.
        let mut long: Vec<Value> = (0..2000_u64)
            .map(|i| pool[usize::try_from(i * 7919 % 41).unwrap() % pool.len()].clone())
            .collect();
        long.rotate_left(17);
        sorted(long);
    }

    #[test]
    fn sort_values_put_nulls_where_the_key_says_in_both_directions() {
        let (one, two) = (Value::Int64(1), Value::Int64(2));
        let null = Value::Null;
        for direction in [SortDirection::Ascending, SortDirection::Descending] {
            assert_eq!(
                compare_sort_values(Some(&null), Some(&one), direction, NullOrder::NullsFirst),
                Ordering::Less,
                "{direction:?} NULLS FIRST"
            );
            assert_eq!(
                compare_sort_values(Some(&null), Some(&one), direction, NullOrder::NullsLast),
                Ordering::Greater,
                "{direction:?} NULLS LAST"
            );
            assert_eq!(
                compare_sort_values(None, Some(&null), direction, NullOrder::NullsLast),
                Ordering::Equal,
                "a missing value is null"
            );
        }
        assert_eq!(
            compare_sort_values(
                Some(&one),
                Some(&two),
                SortDirection::Descending,
                NullOrder::NullsLast
            ),
            Ordering::Greater
        );
        assert_eq!(
            compare_sort_values(
                Some(&one),
                Some(&two),
                SortDirection::Ascending,
                NullOrder::NullsLast
            ),
            Ordering::Less
        );
    }
}
