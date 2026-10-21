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
        Value::Map(ops) if ops.keys().any(|k| k.as_str().starts_with('$')) => {
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

    use super::matches_filter;

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
}
