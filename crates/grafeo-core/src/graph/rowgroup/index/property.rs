//! The index on one node property: the nodes that hold each value, and the
//! values `=` can find equal to a key that are not among the forms the key
//! names itself (#535).
//!
//! `=` is lenient across kinds (see [`HashKey::for_equality`]): a number
//! equals a string it parses as (`'042' = 42`), an integer, a float and a
//! decimal of the same value are equal, and lists and maps compare element by
//! element. Numbers compare exactly (IEEE for floats: `-0.0 = 0.0`, NaN
//! equals nothing). The index finds values by exact value, so a lookup by `=`
//! probes the forms a key names (the key, and for a number its integer, its
//! float, its decimal and its canonical strings), and the values no such
//! probe finds (a number spelled another way, such as `'042'`, or a list)
//! wait in a side table by their `=` class, which is empty for a property
//! that holds none of them.

use dashmap::DashMap;
use dashmap::mapref::entry::Entry;
use grafeo_common::types::{Decimal, HashableValue, NodeId, Value};
use grafeo_common::utils::hash::FxHashSet;

use crate::execution::operators::HashKey;

/// The index of one node property.
#[derive(Debug, Default)]
pub(crate) struct PropertyIndex {
    /// The nodes that hold each value.
    values: DashMap<HashableValue, FxHashSet<NodeId>>,
    /// The values of `values` that no probe of a key finds (see
    /// [`is_found_by_probes`]), by their `=` class. A value is added and
    /// removed here under the lock of its entry in `values`.
    others_by_class: DashMap<HashKey, FxHashSet<HashableValue>>,
}

impl PropertyIndex {
    /// Adds `node` under `value`.
    pub(crate) fn insert(&self, value: HashableValue, node: NodeId) {
        match self.values.entry(value) {
            Entry::Occupied(mut entry) => {
                entry.get_mut().insert(node);
            }
            Entry::Vacant(entry) => {
                if let Some(class) = other_class(entry.key().inner()) {
                    self.others_by_class
                        .entry(class)
                        .or_default()
                        .insert(entry.key().clone());
                }
                let mut nodes = FxHashSet::default();
                nodes.insert(node);
                entry.insert(nodes);
            }
        }
    }

    /// Removes `node` from `value`, dropping the value when no node holds it.
    pub(crate) fn remove(&self, value: &HashableValue, node: NodeId) {
        if let Some(mut nodes) = self.values.get_mut(value) {
            nodes.remove(&node);
        }
        // Checked again under the shard lock: between releasing the bucket and
        // removing it, another writer may have added a node to it.
        self.values.remove_if(value, |value, nodes| {
            let empty = nodes.is_empty();
            if empty && let Some(class) = other_class(value.inner()) {
                if let Some(mut others) = self.others_by_class.get_mut(&class) {
                    others.remove(value);
                }
                self.others_by_class
                    .remove_if(&class, |_, others| others.is_empty());
            }
            empty
        });
    }

    /// The nodes that hold exactly `value`.
    pub(crate) fn nodes(&self, value: &HashableValue) -> Vec<NodeId> {
        self.values
            .get(value)
            .map(|nodes| nodes.iter().copied().collect())
            .unwrap_or_default()
    }

    /// The nodes whose value `=` may find equal to `key`: every one it finds
    /// equal, and maybe others, for a filter to decide.
    pub(crate) fn nodes_maybe_equal(&self, key: &Value) -> Vec<NodeId> {
        let Some(class) = HashKey::for_equality(key) else {
            return Vec::new(); // NULL equals nothing
        };
        let mut found = FxHashSet::default();
        for probe in probes(key) {
            if let Some(nodes) = self.values.get(&probe) {
                found.extend(nodes.iter().copied());
            }
        }
        let others: Vec<HashableValue> = self
            .others_by_class
            .get(&class)
            .map(|others| others.iter().cloned().collect())
            .unwrap_or_default();
        for value in &others {
            if let Some(nodes) = self.values.get(value) {
                found.extend(nodes.iter().copied());
            }
        }
        let mut found: Vec<NodeId> = found.into_iter().collect();
        found.sort_unstable();
        found
    }

    /// How many distinct values the index holds.
    pub(crate) fn len(&self) -> usize {
        self.values.len()
    }
}

/// Which values a lookup of one key by `=` finds through an index: a node
/// [`PropertyIndex::nodes_maybe_equal`] returns for the key holds one of
/// them. A lookup that reads a candidate's value again (at a reader's view)
/// keeps the candidate when this finds the value, so it keeps what the index
/// would find for the value the reader sees.
pub(crate) struct MaybeEqual {
    class: HashKey,
    probes: Vec<HashableValue>,
}

impl MaybeEqual {
    /// The values found for `key`; `None` for a key `=` finds nothing equal
    /// to (NULL).
    pub(crate) fn new(key: &Value) -> Option<Self> {
        Some(Self {
            class: HashKey::for_equality(key)?,
            probes: probes(key),
        })
    }

    /// Whether the lookup finds `value`: a probe of the key, or a value no
    /// probe finds in the key's `=` class.
    pub(crate) fn finds(&self, value: &Value) -> bool {
        let value = HashableValue::new(value.clone());
        self.probes.contains(&value)
            || other_class(value.inner()).is_some_and(|class| class == self.class)
    }
}

/// The values a key is looked up as: the key itself; for an integer, its
/// decimal, its text and, when a float has exactly its value, that float's
/// forms; for a float, its decimal and its forms (see [`push_float_forms`]);
/// for a decimal, its float's forms and, when whole, its integer and that
/// integer's text; for a string, the integer, decimal and float it reads as.
fn probes(key: &Value) -> Vec<HashableValue> {
    let mut probes = vec![HashableValue::new(key.clone())];
    // A decimal equals an integer of its value and a float whose shortest
    // text is its value (see `Value::compare_decimal`); a decimal probe
    // finds a stored decimal at any scale.
    match key {
        Value::Int64(i) => push_integer_forms(&mut probes, *i),
        Value::Float64(f) => {
            if let Ok(d) = Decimal::from_f64(*f) {
                probes.push(HashableValue::new(Value::Decimal(d)));
            }
            push_float_forms(&mut probes, *f);
        }
        Value::Decimal(d) => {
            push_float_forms(&mut probes, d.to_f64());
            if d.floor() == *d
                && let Some(integer) = d.trunc_to_i64()
            {
                push_integer_forms(&mut probes, integer);
            }
        }
        // A numeric string equals the decimal of the number it reads as:
        // decimal text exactly, other number text (`'4.2e1'`) as its float.
        Value::String(s) => {
            let decimal = Decimal::parse(s).ok().or_else(|| {
                s.parse::<f64>()
                    .ok()
                    .and_then(|f| Decimal::from_f64(f).ok())
            });
            if let Some(d) = decimal {
                probes.push(HashableValue::new(Value::Decimal(d)));
            }
            if let Ok(integer) = s.parse::<i64>() {
                push_integer_forms(&mut probes, integer);
            }
            if let Ok(number) = s.parse::<f64>() {
                push_float_forms(&mut probes, number);
            }
        }
        _ => {}
    }
    probes
}

/// The forms of an integer: itself, its decimal and its text, and the forms
/// of the float of exactly its value, if there is one.
fn push_integer_forms(probes: &mut Vec<HashableValue>, integer: i64) {
    probes.push(HashableValue::new(Value::Int64(integer)));
    probes.push(HashableValue::new(Value::Decimal(Decimal::from_i64(
        integer,
    ))));
    probes.push(HashableValue::new(Value::from(integer.to_string())));
    #[allow(
        clippy::cast_precision_loss,
        reason = "the float is kept only when it is exactly the integer"
    )]
    let float = integer as f64;
    if crate::execution::operators::value_utils::int_equals_float(integer, float) {
        push_float_forms(probes, float);
    }
}

/// The forms of a float `=` may find equal: the float (both zeros for a
/// zero) and its shortest text, and for a whole float in the `i64` range its
/// integer and that integer's text. NaN equals nothing, so it has no forms.
fn push_float_forms(probes: &mut Vec<HashableValue>, float: f64) {
    // 2^63 as f64 is exact; every integral float in [-2^63, 2^63) fits i64.
    const LIMIT: f64 = 9_223_372_036_854_775_808.0;
    if float.is_nan() {
        return;
    }
    if float == 0.0 {
        probes.push(HashableValue::new(Value::Float64(0.0)));
        probes.push(HashableValue::new(Value::Float64(-0.0)));
    } else {
        probes.push(HashableValue::new(Value::Float64(float)));
    }
    probes.push(HashableValue::new(Value::from(float.to_string())));
    if float.fract() == 0.0 && (-LIMIT..LIMIT).contains(&float) {
        #[allow(
            clippy::cast_possible_truncation,
            reason = "float is integral and within [-2^63, 2^63), checked above, so the cast is exact"
        )]
        let integer = float as i64;
        probes.push(HashableValue::new(Value::Int64(integer)));
        probes.push(HashableValue::new(Value::from(integer.to_string())));
    }
}

/// Whether the probes of every key `=` finds equal to `value` find `value`
/// itself.
fn is_found_by_probes(value: &Value) -> bool {
    match value {
        // Integers, floats and decimals are among the forms of every number
        // equal to them; NaN equals nothing.
        Value::Int64(_) | Value::Float64(_) | Value::Decimal(_) => true,
        // A number is found in its canonical text (`'42'`, `'0.3'`, `'-0'`
        // for the float, `'0'` for the integer); other spellings (`'042'`,
        // `'42.0'`, `'4.2e1'`) wait in the side table. A string that is no
        // number equals only itself.
        Value::String(s) => match (s.parse::<i64>(), s.parse::<f64>()) {
            (Ok(integer), _) => integer.to_string() == s.as_str(),
            (Err(_), Ok(n)) if !n.is_nan() => n.to_string() == s.as_str(),
            _ => true,
        },
        // Elements compare by `=` too.
        Value::List(_) | Value::Map(_) | Value::Path { .. } | Value::Vector(_) => false,
        // Equal by their own equality, which their hash agrees with.
        _ => true,
    }
}

/// The `=` class of `value` when no probe finds it (see
/// [`is_found_by_probes`]).
fn other_class(value: &Value) -> Option<HashKey> {
    if is_found_by_probes(value) {
        None
    } else {
        HashKey::for_equality(value)
    }
}

impl Clone for PropertyIndex {
    fn clone(&self) -> Self {
        Self {
            values: self.values.clone(),
            others_by_class: self.others_by_class.clone(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn index(values: &[Value]) -> PropertyIndex {
        let index = PropertyIndex::default();
        for (n, value) in values.iter().enumerate() {
            index.insert(HashableValue::new(value.clone()), NodeId::new(n as u64));
        }
        index
    }

    fn found(index: &PropertyIndex, key: Value) -> Vec<u64> {
        index
            .nodes_maybe_equal(&key)
            .into_iter()
            .map(|id| id.as_u64())
            .collect()
    }

    #[test]
    fn numbers_are_found_in_every_spelling() {
        let index = index(&[
            Value::Int64(42),
            Value::Float64(42.0),
            Value::from("42"),
            Value::from("042"),
            Value::from("+42"),
            Value::from("42.0"),
            Value::from("abc"),
            Value::Int64(19),
        ]);
        let all = vec![0, 1, 2, 3, 4, 5];
        assert_eq!(found(&index, Value::Int64(42)), all);
        assert_eq!(found(&index, Value::Float64(42.0)), all);
        assert_eq!(found(&index, Value::from("042")), all);
        assert_eq!(found(&index, Value::from("abc")), vec![6]);
        assert_eq!(found(&index, Value::Null), Vec::<u64>::new());
    }

    #[test]
    fn a_decimal_is_found_by_the_strings_that_read_as_it() {
        let decimal = |text: &str| Value::Decimal(Decimal::parse(text).unwrap());
        let index = index(&[decimal("42.0"), decimal("0.3"), Value::from("abc")]);
        for key in ["42", "042", "+42", "42.00", "4.2e1"] {
            assert_eq!(found(&index, Value::from(key)), vec![0], "{key}");
        }
        assert_eq!(found(&index, Value::from("0.30")), vec![1]);
        assert_eq!(found(&index, Value::from("abc")), vec![2]);
    }

    #[test]
    fn close_numbers_do_not_meet() {
        let index = index(&[
            Value::Float64(0.1 + 0.2),
            Value::Float64(0.3),
            Value::Int64(1),
            Value::Float64(0.999_999_999_999_999_9),
            Value::Int64(88),
        ]);
        // 0.1 + 0.2 is 0.30000000000000004, not 0.3.
        assert_eq!(found(&index, Value::Float64(0.3)), vec![1]);
        assert_eq!(found(&index, Value::Float64(0.1 + 0.2)), vec![0]);
        assert_eq!(found(&index, Value::Int64(1)), vec![2]);
        assert_eq!(found(&index, Value::Float64(1.0)), vec![2]);
        assert!(index.others_by_class.is_empty());
    }

    #[test]
    fn zero_meets_negative_zero_and_nan_meets_nothing() {
        let index = index(&[
            Value::Float64(0.0),
            Value::Float64(-0.0),
            Value::Int64(0),
            Value::Float64(f64::NAN),
            Value::Float64(1e-300),
        ]);
        assert_eq!(found(&index, Value::Float64(0.0)), vec![0, 1, 2]);
        assert_eq!(found(&index, Value::Float64(-0.0)), vec![0, 1, 2]);
        assert_eq!(found(&index, Value::Int64(0)), vec![0, 1, 2]);
        assert_eq!(found(&index, Value::Float64(1e-300)), vec![4]);
        // The key itself is a probe, so the stored NaN is a candidate the
        // filter then refuses; no other number finds it.
        assert!(!found(&index, Value::Int64(0)).contains(&3));
    }

    #[test]
    fn large_integers_meet_only_a_float_of_exactly_their_value() {
        let above = (1_i64 << 53) + 1;
        let index = index(&[
            Value::Int64(above),
            Value::Float64(9_007_199_254_740_992.0),
            Value::Int64(1 << 53),
            Value::from(above.to_string()),
        ]);
        // 2^53 + 1 has no float of its value.
        assert_eq!(found(&index, Value::Int64(above)), vec![0, 3]);
        assert_eq!(
            found(&index, Value::Float64(9_007_199_254_740_992.0)),
            vec![1, 2]
        );
        assert_eq!(found(&index, Value::Int64(1 << 53)), vec![1, 2]);
        assert!(index.others_by_class.is_empty());
    }

    #[test]
    fn small_integers_are_found_by_every_key_equal_to_them() {
        // A property of 0 and 1 flags keeps no side table, and a key finds
        // only its own flag.
        let index = index(&[Value::Int64(0), Value::Int64(1), Value::Int64(1)]);
        assert!(index.others_by_class.is_empty());
        assert_eq!(found(&index, Value::Int64(1)), vec![1, 2]);
        assert_eq!(found(&index, Value::Int64(0)), vec![0]);
        assert_eq!(found(&index, Value::Float64(1.0)), vec![1, 2]);
        assert_eq!(found(&index, Value::from("1")), vec![1, 2]);
        assert_eq!(
            found(&index, Value::Float64(0.999_999_999_999_999_9)),
            Vec::<u64>::new()
        );
        assert_eq!(found(&index, Value::Float64(1e-300)), Vec::<u64>::new());
    }

    #[test]
    fn a_removed_value_leaves_the_side_table() {
        let index = index(&[Value::from("042"), Value::from("42")]);
        assert_eq!(found(&index, Value::Int64(42)), vec![0, 1]);
        index.remove(&HashableValue::new(Value::from("042")), NodeId::new(0));
        assert_eq!(found(&index, Value::Int64(42)), vec![1]);
        assert!(
            index.others_by_class.is_empty(),
            "no value no probe finds is left"
        );
    }

    #[test]
    fn values_the_probes_find_need_no_side_table() {
        for value in [
            Value::Int64(42),
            Value::Int64(0),
            Value::Int64(1),
            Value::Int64(-1 << 53),
            Value::Int64((1 << 53) + 1),
            Value::Int64(i64::MIN),
            Value::Float64(42.5),
            Value::Float64(0.3),
            Value::Float64(-0.0),
            Value::Float64(f64::NAN),
            Value::Float64(f64::INFINITY),
            Value::from("42"),
            Value::from("42.5"),
            Value::from("0.3"),
            Value::from("abc"),
            Value::from("NaN"),
            Value::Bool(true),
        ] {
            assert!(is_found_by_probes(&value), "{value:?}");
        }
        for value in [
            Value::from("042"),
            Value::from("42.0"),
            Value::from("0.30"),
            Value::from("4.2e1"),
            Value::from("-0"),
            Value::List(vec![Value::Int64(42)].into()),
        ] {
            assert!(!is_found_by_probes(&value), "{value:?}");
        }
    }

    /// `MaybeEqual` finds a value exactly when the index's lookup of the key
    /// returns a node holding it.
    #[test]
    fn maybe_equal_finds_what_the_lookup_returns() {
        let values = [
            Value::Int64(42),
            Value::Float64(42.0),
            Value::from("42"),
            Value::from("042"),
            Value::from("42.0"),
            Value::Decimal(Decimal::parse("42.00").unwrap()),
            Value::Float64(0.1 + 0.2),
            Value::Float64(0.3),
            Value::Int64(1),
            Value::from("abc"),
            Value::List(vec![Value::Int64(42)].into()),
            Value::Bool(true),
        ];
        let index = index(&values);
        for key in values.iter().chain(&[Value::Null, Value::from("4.2e1")]) {
            let found = found(&index, key.clone());
            for (n, value) in values.iter().enumerate() {
                let finds = MaybeEqual::new(key).is_some_and(|test| test.finds(value));
                assert_eq!(finds, found.contains(&(n as u64)), "{key:?} and {value:?}");
            }
        }
    }
}
