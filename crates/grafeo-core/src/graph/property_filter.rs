//! The property filters of the search calls: a value a filter names, or a
//! map of `$`-prefixed operators (`$gt`, `$gte`, `$lt`, `$lte`, `$ne`, `$in`,
//! `$nin`, `$contains`), as the engine's `vector_search`, `text_search` and
//! `hybrid_search` take them.

use std::cmp::Ordering;

use grafeo_common::types::Value;

/// Whether a node's value `node_value` matches the filter `filter_value`:
/// with a map of `$`-prefixed operator keys, every operator holds (an
/// unknown operator never does); with any other value, the two are equal.
#[must_use]
pub fn matches_filter(node_value: &Value, filter_value: &Value) -> bool {
    match filter_value {
        Value::Map(ops) if is_operator_map(filter_value) => {
            ops.iter().all(|(op_key, op_val)| match op_key.as_str() {
                "$gt" => compare_values(node_value, op_val) == Some(Ordering::Greater),
                "$gte" => matches!(
                    compare_values(node_value, op_val),
                    Some(Ordering::Greater | Ordering::Equal)
                ),
                "$lt" => compare_values(node_value, op_val) == Some(Ordering::Less),
                "$lte" => matches!(
                    compare_values(node_value, op_val),
                    Some(Ordering::Less | Ordering::Equal)
                ),
                "$ne" => node_value != op_val,
                "$in" => match op_val {
                    Value::List(items) => items.iter().any(|v| v == node_value),
                    _ => false,
                },
                "$nin" => match op_val {
                    Value::List(items) => !items.iter().any(|v| v == node_value),
                    _ => true,
                },
                "$contains" => match (node_value, op_val) {
                    (Value::String(a), Value::String(b)) => a.contains(b.as_str()),
                    _ => false,
                },
                _ => false,
            })
        }
        _ => node_value == filter_value,
    }
}

/// The operators a filter map may hold.
const OPERATORS: [&str; 8] = [
    "$gt",
    "$gte",
    "$lt",
    "$lte",
    "$ne",
    "$in",
    "$nin",
    "$contains",
];

/// Whether `filter_value` is a map of `$`-prefixed operators, not a value to
/// equal.
#[must_use]
pub fn is_operator_map(filter_value: &Value) -> bool {
    matches!(filter_value, Value::Map(ops) if ops.keys().any(|k| k.as_str().starts_with('$')))
}

/// Why the filter `filter_value` on the property `key` is refused, or `None`
/// when it is well formed: a null value (no value equals null; `$ne` keeps
/// the nodes without the property), an unknown operator, `$in` or `$nin`
/// without a list, `$contains` without a string. A search checks its filters
/// once, before it runs.
#[must_use]
pub fn filter_problem(key: &str, filter_value: &Value) -> Option<String> {
    let null = || {
        Some(format!(
            "the filter on `{key}` compares with null, which no value equals; to keep the nodes without `{key}`, use `$ne`"
        ))
    };
    let Value::Map(ops) = filter_value else {
        return if filter_value.is_null() { null() } else { None };
    };
    if !is_operator_map(filter_value) {
        return None;
    }
    for (op, operand) in ops.iter() {
        let op = op.as_str();
        if !OPERATORS.contains(&op) {
            return Some(format!(
                "the filter on `{key}` uses `{op}`, which is not an operator (they are {})",
                OPERATORS.join(", ")
            ));
        }
        match (op, operand) {
            (_, Value::Null) => return null(),
            ("$in" | "$nin", Value::List(items)) if items.iter().any(Value::is_null) => {
                return null();
            }
            ("$in" | "$nin", Value::List(_)) | ("$contains", Value::String(_)) => {}
            ("$in" | "$nin", _) => {
                return Some(format!("the filter on `{key}`: `{op}` takes a list"));
            }
            ("$contains", _) => {
                return Some(format!("the filter on `{key}`: `$contains` takes a string"));
            }
            _ => {}
        }
    }
    None
}

/// Whether the filter `filter_value` keeps a node without the property: only
/// a map whose every operator is `$ne` or `$nin` does ("not equal to true"
/// holds for a missing flag); equality and the other operators need a value.
#[must_use]
pub fn keeps_missing(filter_value: &Value) -> bool {
    match filter_value {
        Value::Map(ops) if is_operator_map(filter_value) => {
            ops.keys().all(|op| matches!(op.as_str(), "$ne" | "$nin"))
        }
        _ => false,
    }
}

/// Orders two values for a filter (numbers across their types).
fn compare_values(a: &Value, b: &Value) -> Option<Ordering> {
    match (a, b) {
        (Value::Decimal(_), _) | (_, Value::Decimal(_)) => a.compare_decimal(b),
        (Value::Int64(a), Value::Int64(b)) => Some(a.cmp(b)),
        (Value::Float64(a), Value::Float64(b)) => a.partial_cmp(b),
        (Value::String(a), Value::String(b)) => Some(a.cmp(b)),
        (Value::Bool(a), Value::Bool(b)) => Some(a.cmp(b)),
        // An integer and a float compare exactly, as a filter does.
        (Value::Int64(a), Value::Float64(b)) => {
            crate::execution::operators::value_utils::partial_cmp_int_float(*a, *b)
        }
        (Value::Float64(a), Value::Int64(b)) => {
            crate::execution::operators::value_utils::partial_cmp_int_float(*b, *a)
                .map(Ordering::reverse)
        }
        (Value::Timestamp(a), Value::Timestamp(b)) => Some(a.cmp(b)),
        (Value::Date(a), Value::Date(b)) => Some(a.cmp(b)),
        (Value::Time(a), Value::Time(b)) => Some(a.cmp(b)),
        // Zoned datetimes, also against a timestamp: by their instant.
        _ => a.compare_instants(b),
    }
}

#[cfg(test)]
mod tests {
    use grafeo_common::types::{PropertyKey, Value};

    use super::{filter_problem, keeps_missing, matches_filter};

    fn operators(pairs: &[(&str, Value)]) -> Value {
        Value::Map(std::sync::Arc::new(
            pairs
                .iter()
                .map(|(key, value)| (PropertyKey::new(*key), value.clone()))
                .collect(),
        ))
    }

    /// A plain value is an equality filter; operators compare numbers across
    /// their types, and an unknown operator matches nothing.
    #[test]
    fn filters_match_by_equality_and_by_their_operators() {
        assert!(matches_filter(&Value::from(19), &Value::from(19)));
        assert!(!matches_filter(&Value::from(19), &Value::from(3)));
        let between = operators(&[("$gte", Value::from(3)), ("$lt", Value::from(19.5))]);
        assert!(matches_filter(&Value::from(19), &between));
        assert!(!matches_filter(&Value::from(20), &between));
        let cities = operators(&[(
            "$in",
            Value::List(vec![Value::from("Amsterdam"), Value::from("Berlin")].into()),
        )]);
        assert!(matches_filter(&Value::from("Berlin"), &cities));
        assert!(!matches_filter(&Value::from("Paris"), &cities));
        assert!(matches_filter(
            &Value::from("Prague castle"),
            &operators(&[("$contains", Value::from("castle"))])
        ));
        assert!(!matches_filter(
            &Value::from(1),
            &operators(&[("$near", Value::from(1))])
        ));
    }

    /// Only `$ne` and `$nin`, alone or together, keep a node without the
    /// property.
    #[test]
    fn only_ne_and_nin_keep_a_node_without_the_value() {
        let yes = Value::Bool(true);
        let list = Value::List(vec![yes.clone()].into());
        assert!(keeps_missing(&operators(&[("$ne", yes.clone())])));
        assert!(keeps_missing(&operators(&[("$nin", list.clone())])));
        assert!(keeps_missing(&operators(&[
            ("$ne", yes.clone()),
            ("$nin", list.clone())
        ])));
        assert!(!keeps_missing(&operators(&[
            ("$ne", yes.clone()),
            ("$in", list)
        ])));
        assert!(!keeps_missing(&operators(&[("$gt", Value::from(1))])));
        assert!(!keeps_missing(&yes));
    }

    /// Nulls and malformed operators are refused with a reason naming the
    /// key, and `$ne` or the operator; well-formed filters pass.
    #[test]
    fn nulls_and_malformed_operators_are_refused_by_name() {
        let refused = |value: &Value, also: &str| {
            let problem = filter_problem("deleted", value).expect("refused");
            assert!(
                problem.contains("deleted") && problem.contains(also),
                "{problem}"
            );
        };
        let null = Value::Null;
        refused(&null, "$ne");
        refused(&operators(&[("$ne", null.clone())]), "$ne");
        refused(&operators(&[("$gt", null.clone())]), "$ne");
        refused(
            &operators(&[("$in", Value::List(vec![null.clone()].into()))]),
            "$ne",
        );
        refused(
            &operators(&[("$nin", Value::List(vec![Value::from(1), null].into()))]),
            "$ne",
        );
        refused(&operators(&[("$near", Value::from(1))]), "$near");
        refused(&operators(&[("$in", Value::from(1))]), "$in");
        refused(&operators(&[("$nin", Value::from("x"))]), "$nin");
        refused(&operators(&[("$contains", Value::from(5))]), "$contains");
        for fine in [
            Value::Bool(false),
            operators(&[("$ne", Value::Bool(true))]),
            operators(&[("$gte", Value::from(3)), ("$lt", Value::from(19.5))]),
            operators(&[("$nin", Value::List(vec![Value::Bool(true)].into()))]),
            operators(&[("$contains", Value::from("castle"))]),
        ] {
            assert_eq!(filter_problem("deleted", &fine), None, "{fine:?}");
        }
    }
}
