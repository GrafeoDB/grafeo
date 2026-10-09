//! The index on one node property: the nodes that hold each value, and the
//! values `=` can find equal to a key that are not among the forms the key
//! names itself (#535).
//!
//! `=` is lenient (see [`HashKey::for_equality`]): a number equals a string
//! it parses as (`'042' = 42`), numbers compare within `f64::EPSILON`, and
//! lists and maps compare element by element. The index finds values by
//! exact value, so a lookup by `=` probes the forms a key names (the key, and
//! for a number its integer, its float and its decimal string), and the
//! values no such probe finds (a number spelled another way, a number below
//! 2 in magnitude, a list) wait in a side table by their `=` class, which is
//! empty for a property that holds none of them.

use dashmap::DashMap;
use dashmap::mapref::entry::Entry;
use grafeo_common::types::{HashableValue, NodeId, Value};
use grafeo_common::utils::hash::FxHashSet;

use crate::execution::operators::HashKey;

/// Below this magnitude distinct numbers can be within `f64::EPSILON` of
/// each other, so `=` finds one equal to several (see
/// [`HashKey::for_equality`]); from it up, only to the same number.
const EXACT_FROM: f64 = 2.0;

/// The largest magnitude up to which every integer is exactly a float.
const EXACT_INTEGERS: f64 = 9_007_199_254_740_992.0;

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
    pub(super) fn insert(&self, value: HashableValue, node: NodeId) {
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
    pub(super) fn remove(&self, value: &HashableValue, node: NodeId) {
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
    pub(super) fn nodes(&self, value: &HashableValue) -> Vec<NodeId> {
        self.values
            .get(value)
            .map(|nodes| nodes.iter().copied().collect())
            .unwrap_or_default()
    }

    /// The nodes whose value `=` may find equal to `key`: every one it finds
    /// equal, and maybe others, for a filter to decide.
    pub(super) fn nodes_maybe_equal(&self, key: &Value) -> Vec<NodeId> {
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
    pub(super) fn len(&self) -> usize {
        self.values.len()
    }
}

/// The number `=` reads `value` as, if any: an integer or a float, or a
/// string that parses as one (not NaN).
fn number(value: &Value) -> Option<f64> {
    match value {
        // `=` compares an integer with a float as this `f64`.
        Value::Int64(i) => Some(*i as f64),
        Value::Float64(f) => Some(*f),
        Value::String(s) => s.parse::<f64>().ok().filter(|n| !n.is_nan()),
        _ => None,
    }
}

/// The values a key is looked up as: the key itself; for a number, the
/// integers within `f64::EPSILON` of it (and, for a string, the integer it
/// parses as); and for a number from 2 up in magnitude, its float and its
/// decimal string.
fn probes(key: &Value) -> Vec<HashableValue> {
    let mut probes = vec![HashableValue::new(key.clone())];
    if let Value::String(s) = key
        && let Ok(integer) = s.parse::<i64>()
    {
        probes.push(HashableValue::new(Value::Int64(integer)));
    }
    let Some(n) = number(key).filter(|n| n.is_finite()) else {
        return probes;
    };
    for near in [n.floor(), n.ceil()] {
        // The decimal string of an integral float is its digits, so it
        // parses as the integer.
        if (near - n).abs() < f64::EPSILON
            && near.abs() <= EXACT_INTEGERS
            && let Ok(integer) = near.to_string().parse::<i64>()
        {
            probes.push(HashableValue::new(Value::Int64(integer)));
        }
    }
    if n.abs() >= EXACT_FROM {
        probes.push(HashableValue::new(Value::Float64(n)));
        probes.push(HashableValue::new(Value::from(n.to_string())));
    }
    probes
}

/// Whether the probes of every key `=` finds equal to `value` find `value`
/// itself.
fn is_found_by_probes(value: &Value) -> bool {
    match value {
        // Every key `=` finds equal to an integer probes it, up to 2^53:
        // beyond, the integer shares its float with other integers.
        Value::Int64(i) => i.unsigned_abs() <= 1 << 53,
        // A float below 2 is within EPSILON of others; NaN and the
        // infinities equal no number.
        Value::Float64(f) => !f.is_finite() || f.abs() >= EXACT_FROM,
        // A number in its own decimal spelling is a probe of its class; a
        // string that is no number equals only itself.
        Value::String(s) => match s.parse::<f64>() {
            Ok(n) if n.is_finite() => n.abs() >= EXACT_FROM && n.to_string() == s.as_str(),
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
    fn numbers_below_two_meet_their_close_neighbours() {
        let index = index(&[
            Value::Float64(0.1 + 0.2),
            Value::Float64(0.3),
            Value::Int64(1),
            Value::Float64(0.999_999_999_999_999_9),
            Value::Int64(88),
        ]);
        // 0.1 + 0.2 is within EPSILON of 0.3: both found, 88 not.
        let small = found(&index, Value::Float64(0.3));
        assert!(small.contains(&0) && small.contains(&1), "{small:?}");
        assert!(!small.contains(&4));
        let one = found(&index, Value::Int64(1));
        assert!(one.contains(&2) && one.contains(&3), "{one:?}");
    }

    #[test]
    fn small_integers_are_found_by_every_key_equal_to_them() {
        // A property of 0 and 1 flags keeps no side table, and a key finds
        // only its own flag.
        let index = index(&[Value::Int64(0), Value::Int64(1), Value::Int64(1)]);
        assert!(index.others_by_class.is_empty());
        assert_eq!(found(&index, Value::Int64(1)), vec![1, 2]);
        assert_eq!(found(&index, Value::Int64(0)), vec![0]);
        assert_eq!(
            found(&index, Value::Float64(0.999_999_999_999_999_9)),
            vec![1, 2]
        );
        assert_eq!(found(&index, Value::from("1")), vec![1, 2]);
        assert_eq!(found(&index, Value::Float64(1e-300)), vec![0]);
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
            // A flag of 0 or 1: the integers near a key are probes.
            Value::Int64(0),
            Value::Int64(1),
            Value::Int64(-1 << 53),
            Value::Float64(42.5),
            Value::from("42"),
            Value::from("42.5"),
            Value::from("abc"),
            Value::Bool(true),
        ] {
            assert!(is_found_by_probes(&value), "{value:?}");
        }
        for value in [
            Value::Int64((1 << 53) + 1),
            Value::Float64(0.3),
            Value::from("042"),
            Value::from("42.0"),
            Value::from("0.3"),
            Value::List(vec![Value::Int64(42)].into()),
        ] {
            assert!(!is_found_by_probes(&value), "{value:?}");
        }
    }
}
