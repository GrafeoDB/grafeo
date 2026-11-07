//! The index on one node property: the nodes that hold each value, and the
//! values `=` can find equal to a key that are not among the forms the key
//! names itself (#535).
//!
//! `=` reads numbers by value (see [`HashKey::for_equality`]): an integer, a
//! float and a decimal of the same value are equal, and lists and maps
//! compare element by element. Two values of different kinds are never equal:
//! a string is not the number it spells. Numbers compare exactly (IEEE for
//! floats: `-0.0 = 0.0`, NaN equals nothing). The index finds values by exact
//! value, so a lookup by `=` probes the forms a key names (the key, and for a
//! number its integer, its float and its decimal), and the values no such
//! probe finds (a list or a map of numbers) wait in a side table by their `=`
//! class, which is empty for a property that holds none of them.

use std::sync::atomic::{AtomicU32, AtomicUsize, Ordering};

use dashmap::DashMap;
use dashmap::mapref::entry::Entry;
use grafeo_common::types::{Decimal, HashableValue, NodeId, Value};
use grafeo_common::utils::hash::FxHashSet;

use crate::execution::operators::HashKey;
use crate::execution::operators::value_utils::ValueKinds;

/// The index of one node property.
#[derive(Debug, Default)]
pub(crate) struct PropertyIndex {
    /// The nodes that hold each value.
    values: DashMap<HashableValue, FxHashSet<NodeId>>,
    /// The values of `values` that no probe of a key finds (see
    /// [`is_found_by_probes`]), by their `=` class. A value is added and
    /// removed here under the lock of its entry in `values`.
    others_by_class: DashMap<HashKey, FxHashSet<HashableValue>>,
    /// The kinds of every value it has held (see [`ValueKinds`]): one `OR` per insert, never
    /// cleared, so it errs toward a read of the values themselves after a delete. Built again with
    /// the index when a database opens, as the index is.
    kinds: AtomicU32,
    /// The heap bytes the index holds by [`entry_bytes`], kept as its entries change (see
    /// [`counted_bytes`](Self::counted_bytes)).
    counted: AtomicUsize,
}

/// The bytes an entry takes besides its value's payload and its set of ids: its value and set,
/// and a word of the map's own.
const ENTRY_BYTES: usize = std::mem::size_of::<HashableValue>()
    + std::mem::size_of::<FxHashSet<NodeId>>()
    + std::mem::size_of::<usize>();

/// The bytes a class of `others_by_class` takes besides its set's table: its key and set, and a
/// word of the map's own.
const CLASS_BYTES: usize = std::mem::size_of::<HashKey>()
    + std::mem::size_of::<FxHashSet<HashableValue>>()
    + std::mem::size_of::<usize>();

/// The heap bytes of the entry of `value` with the ids `nodes`: the one formula of the index's
/// size, which its walk and its count both read. A set is measured by its table's allocation, not
/// its capacity: an erase that leaves a tombstone lowers the capacity and a later insert into it
/// raises it again, while the allocation stays. `DashMap::capacity` is not read: it locks every
/// shard.
fn entry_bytes(value: &HashableValue, nodes: &FxHashSet<NodeId>) -> usize {
    ENTRY_BYTES + value.inner().estimated_size_bytes() + nodes.allocation_size()
}

/// The heap bytes of a class of `others_by_class` with the values `others` (clones of keys of
/// `values`, whose payload those count).
fn class_bytes(others: &FxHashSet<HashableValue>) -> usize {
    CLASS_BYTES + others.allocation_size()
}

impl PropertyIndex {
    /// The heap bytes the index holds, as its entries' changes left them: one load.
    pub(crate) fn counted_bytes(&self) -> usize {
        self.counted.load(Ordering::Relaxed)
    }

    /// The heap bytes the index holds, by a walk of every entry and class: the oracle of
    /// [`counted_bytes`](Self::counted_bytes).
    pub(crate) fn walk_bytes(&self) -> usize {
        self.values
            .iter()
            .map(|entry| entry_bytes(entry.key(), entry.value()))
            .sum::<usize>()
            + self
                .others_by_class
                .iter()
                .map(|class| class_bytes(class.value()))
                .sum::<usize>()
    }

    /// Adds `bytes` to the count, takes `freed` off.
    fn count(&self, bytes: usize, freed: usize) {
        if bytes == freed {
            return;
        }
        // A count never below its parts: saturating, so a drift cannot wrap into a huge figure.
        let _ = self
            .counted
            .try_update(Ordering::Relaxed, Ordering::Relaxed, |counted| {
                debug_assert!(counted + bytes >= freed, "a property index count drifted");
                Some((counted + bytes).saturating_sub(freed))
            });
    }

    /// Adds `node` under `value`.
    pub(crate) fn insert(&self, value: HashableValue, node: NodeId) {
        self.kinds
            .fetch_or(ValueKinds::of(value.inner()).bits(), Ordering::Relaxed);
        let (bytes, freed) = match self.values.entry(value) {
            Entry::Occupied(mut entry) => {
                let nodes = entry.get_mut();
                let before = nodes.allocation_size();
                nodes.insert(node);
                (nodes.allocation_size(), before)
            }
            Entry::Vacant(entry) => {
                let (mut bytes, mut freed) = (0, 0);
                if let Some(class) = other_class(entry.key().inner()) {
                    let mut others = self.others_by_class.entry(class).or_default();
                    // A new class's set is empty: its bytes before are the class's own.
                    freed = if others.is_empty() {
                        0
                    } else {
                        class_bytes(&others)
                    };
                    others.insert(entry.key().clone());
                    bytes = class_bytes(&others);
                }
                let mut nodes = FxHashSet::default();
                nodes.insert(node);
                bytes += entry_bytes(entry.key(), &nodes);
                entry.insert(nodes);
                (bytes, freed)
            }
        };
        self.count(bytes, freed);
    }

    /// Removes `node` from `value`, dropping the value when no node holds it.
    pub(crate) fn remove(&self, value: &HashableValue, node: NodeId) {
        // An erase keeps the set's table: nothing to count until the entry goes.
        if let Some(mut nodes) = self.values.get_mut(value) {
            nodes.remove(&node);
        }
        // Checked again under the shard lock: between releasing the bucket and
        // removing it, another writer may have added a node to it.
        let mut freed = 0;
        let removed = self.values.remove_if(value, |value, nodes| {
            let empty = nodes.is_empty();
            if empty && let Some(class) = other_class(value.inner()) {
                if let Some(mut others) = self.others_by_class.get_mut(&class) {
                    others.remove(value);
                }
                if let Some((_, others)) = self
                    .others_by_class
                    .remove_if(&class, |_, others| others.is_empty())
                {
                    freed += class_bytes(&others);
                }
            }
            empty
        });
        if let Some((value, nodes)) = removed {
            freed += entry_bytes(&value, &nodes);
        }
        self.count(0, freed);
    }

    /// The kinds of every value it has held, deleted ones too.
    pub(crate) fn kinds(&self) -> ValueKinds {
        ValueKinds::from_bits(self.kinds.load(Ordering::Relaxed))
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

    /// Whether [`nodes_maybe_equal`](Self::nodes_maybe_equal) finds at most
    /// `limit` nodes for `key`, stopping once it is sure: a node under two
    /// values counts twice, so it may answer no for a key just within the limit.
    pub(crate) fn candidates_within(&self, key: &Value, limit: usize) -> bool {
        let Some(class) = HashKey::for_equality(key) else {
            return true; // NULL equals nothing
        };
        let probes = probes(key);
        let mut count = 0;
        let mut within = |value: &HashableValue| {
            count += self.values.get(value).map_or(0, |nodes| nodes.len());
            count <= limit
        };
        probes
            .iter()
            .enumerate()
            .all(|(index, value)| probes[..index].contains(value) || within(value))
            && self.others_by_class.get(&class).is_none_or(|others| {
                others
                    .iter()
                    .all(|value| probes.contains(value) || within(value))
            })
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
#[derive(Debug)]
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

    /// Adds the integers and the bits of the floats a lookup of `key`, an
    /// integer or a float, finds: what [`integers`](Self::integers) and
    /// [`floats`](Self::floats) of its own lookup list, without the decimals
    /// a lookup of a column of other values needs. A long `IN` over a column
    /// of numbers builds none of those.
    pub(crate) fn number_forms(key: &Value, integers: &mut Vec<i64>, floats: &mut Vec<u64>) {
        let float_forms = |float: f64, integers: &mut Vec<i64>, floats: &mut Vec<u64>| {
            if float.is_nan() {
                return;
            }
            if float == 0.0 {
                floats.push(0.0_f64.to_bits());
                floats.push((-0.0_f64).to_bits());
            } else {
                floats.push(float.to_bits());
            }
            integers.extend(whole_integer(float));
        };
        match key {
            Value::Int64(integer) => {
                integers.push(*integer);
                if let Some(float) = exact_float(*integer) {
                    float_forms(float, integers, floats);
                }
            }
            Value::Float64(float) => {
                floats.push(float.to_bits());
                float_forms(*float, integers, floats);
            }
            _ => {}
        }
    }

    /// The integers it finds. Every integer is found by a probe or not at
    /// all (see [`is_found_by_probes`]), so an integer is found if and only
    /// if it is one of these.
    pub(crate) fn integers(&self) -> impl Iterator<Item = i64> + '_ {
        self.probes
            .iter()
            .filter_map(|probe| probe.inner().as_int64())
    }

    /// The floats it finds. Every float is found by a probe or not at all
    /// (see [`is_found_by_probes`]), so a float of the same bits as one of
    /// these is found, and no other.
    pub(crate) fn floats(&self) -> impl Iterator<Item = f64> + '_ {
        self.probes
            .iter()
            .filter_map(|probe| probe.inner().as_float64())
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
/// decimal and, when a float has exactly its value, that float's forms; for a
/// float, its decimal and its forms (see [`push_float_forms`]); for a decimal,
/// its float's forms and, when whole, its integer. A string is looked up as
/// itself: it equals no number.
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
        _ => {}
    }
    probes
}

/// The forms of an integer: itself and its decimal, and the forms of the
/// float of exactly its value, if there is one.
fn push_integer_forms(probes: &mut Vec<HashableValue>, integer: i64) {
    probes.push(HashableValue::new(Value::Int64(integer)));
    probes.push(HashableValue::new(Value::Decimal(Decimal::from_i64(
        integer,
    ))));
    if let Some(float) = exact_float(integer) {
        push_float_forms(probes, float);
    }
}

/// The float of exactly the value of `integer`, if there is one.
fn exact_float(integer: i64) -> Option<f64> {
    #[allow(
        clippy::cast_precision_loss,
        reason = "the float is kept only when it is exactly the integer"
    )]
    let float = integer as f64;
    crate::execution::operators::value_utils::int_equals_float(integer, float).then_some(float)
}

/// The integer of a whole float in the `i64` range.
fn whole_integer(float: f64) -> Option<i64> {
    // 2^63 as f64 is exact; every integral float in [-2^63, 2^63) fits i64.
    const LIMIT: f64 = 9_223_372_036_854_775_808.0;
    if float.fract() == 0.0 && (-LIMIT..LIMIT).contains(&float) {
        #[allow(
            clippy::cast_possible_truncation,
            reason = "float is integral and within [-2^63, 2^63), checked above, so the cast is exact"
        )]
        let integer = float as i64;
        return Some(integer);
    }
    None
}

/// The forms of a float `=` may find equal: the float (both zeros for a
/// zero), and for a whole float in the `i64` range its integer. NaN equals
/// nothing, so it has no forms.
fn push_float_forms(probes: &mut Vec<HashableValue>, float: f64) {
    if float.is_nan() {
        return;
    }
    if float == 0.0 {
        probes.push(HashableValue::new(Value::Float64(0.0)));
        probes.push(HashableValue::new(Value::Float64(-0.0)));
    } else {
        probes.push(HashableValue::new(Value::Float64(float)));
    }
    if let Some(integer) = whole_integer(float) {
        probes.push(HashableValue::new(Value::Int64(integer)));
    }
}

/// Whether the probes of every key `=` finds equal to `value` find `value`
/// itself.
fn is_found_by_probes(value: &Value) -> bool {
    match value {
        // Integers, floats and decimals are among the forms of every number
        // equal to them; NaN equals nothing.
        Value::Int64(_) | Value::Float64(_) | Value::Decimal(_) => true,
        // A string equals only itself.
        Value::String(_) => true,
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
            kinds: AtomicU32::new(self.kinds.load(Ordering::Relaxed)),
            // A clone of a set keeps its capacity: the same bytes.
            counted: AtomicUsize::new(self.counted_bytes()),
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

    /// Ids taken out of a set and put back, and values that come and go (lists among them, which
    /// wait in a class of their own), leave the index's count equal to its walk after every step:
    /// a set's erase leaves a tombstone, which lowers its capacity but not its table.
    #[test]
    fn the_count_follows_ids_that_come_and_go() {
        let index = PropertyIndex::default();
        let counted = |index: &PropertyIndex, what: &str| {
            assert_eq!(index.counted_bytes(), index.walk_bytes(), "{what}");
        };
        let shared = HashableValue::new(Value::from("Amsterdam"));
        for id in 0..40 {
            index.insert(shared.clone(), NodeId::new(id));
        }
        counted(&index, "forty ids under one value");
        for round in 0..5_u64 {
            for id in (0..40).filter(|id| id % 3 == round % 3) {
                index.remove(&shared, NodeId::new(id));
            }
            counted(&index, &format!("round {round}: removed"));
            for id in (0..40).filter(|id| id % 3 == round % 3) {
                index.insert(shared.clone(), NodeId::new(id + 100 * round));
            }
            counted(&index, &format!("round {round}: put back"));
        }
        let list = |n: u64| {
            HashableValue::new(Value::List(
                vec![Value::Int64(i64::try_from(n).unwrap())].into(),
            ))
        };
        for n in 0..20 {
            index.insert(list(n), NodeId::new(1_000 + n));
        }
        counted(&index, "lists, in their class");
        for n in (0..20).step_by(2) {
            index.remove(&list(n), NodeId::new(1_000 + n));
        }
        counted(&index, "half the lists gone");
        for n in 0..20 {
            index.remove(&list(n), NodeId::new(1_000 + n));
        }
        counted(&index, "every list gone, and its class");
    }

    fn found(index: &PropertyIndex, key: Value) -> Vec<u64> {
        index
            .nodes_maybe_equal(&key)
            .into_iter()
            .map(|id| id.as_u64())
            .collect()
    }

    /// A number finds the numbers of its value, a string its own text: no string equals a number,
    /// whatever it spells.
    #[test]
    fn a_number_finds_numbers_and_a_string_its_text() {
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
        assert_eq!(found(&index, Value::Int64(42)), vec![0, 1]);
        assert_eq!(found(&index, Value::Float64(42.0)), vec![0, 1]);
        assert_eq!(found(&index, Value::from("042")), vec![3]);
        assert_eq!(found(&index, Value::from("42")), vec![2]);
        assert_eq!(found(&index, Value::from("42.0")), vec![5]);
        assert_eq!(found(&index, Value::from("abc")), vec![6]);
        assert_eq!(found(&index, Value::Null), Vec::<u64>::new());
    }

    #[test]
    fn a_decimal_is_found_by_numbers_not_by_text() {
        let decimal = |text: &str| Value::Decimal(Decimal::parse(text).unwrap());
        let index = index(&[decimal("42.0"), decimal("0.3"), Value::from("abc")]);
        for key in ["42", "042", "+42", "42.00", "4.2e1", "0.30"] {
            assert_eq!(found(&index, Value::from(key)), Vec::<u64>::new(), "{key}");
        }
        assert_eq!(found(&index, Value::Int64(42)), vec![0]);
        assert_eq!(found(&index, Value::Float64(42.0)), vec![0]);
        assert_eq!(found(&index, decimal("0.30")), vec![1]);
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
        // 2^53 + 1 has no float of its value, and its text is no number.
        assert_eq!(found(&index, Value::Int64(above)), vec![0]);
        assert_eq!(found(&index, Value::from(above.to_string())), vec![3]);
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
        assert_eq!(found(&index, Value::from("1")), Vec::<u64>::new());
        assert_eq!(
            found(&index, Value::Float64(0.999_999_999_999_999_9)),
            Vec::<u64>::new()
        );
        assert_eq!(found(&index, Value::Float64(1e-300)), Vec::<u64>::new());
    }

    #[test]
    fn a_removed_value_leaves_the_side_table() {
        let index = index(&[Value::from("042"), Value::from("42")]);
        assert_eq!(found(&index, Value::Int64(42)), Vec::<u64>::new());
        assert_eq!(found(&index, Value::from("042")), vec![0]);
        index.remove(&HashableValue::new(Value::from("042")), NodeId::new(0));
        assert_eq!(found(&index, Value::from("042")), Vec::<u64>::new());
        assert_eq!(found(&index, Value::from("42")), vec![1]);
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
            // A string is found by its own text, whatever number it spells.
            Value::from("042"),
            Value::from("42.0"),
            Value::from("0.30"),
            Value::from("4.2e1"),
            Value::from("-0"),
        ] {
            assert!(is_found_by_probes(&value), "{value:?}");
        }
        let list = Value::List(vec![Value::Int64(42)].into());
        assert!(!is_found_by_probes(&list), "{list:?}");
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

    /// A value reached both as a probe and in the side table counts once, or a key within the cap
    /// loses its seek and every edge is listed.
    #[test]
    fn a_value_counts_once_however_it_is_reached() {
        let values: Vec<Value> = std::iter::repeat_n(Value::Int64(42), 600).collect();
        let index = index(&values);
        assert_eq!(found(&index, Value::Int64(42)).len(), 600);
        assert!(index.candidates_within(&Value::Int64(42), 600), "the int");
        assert!(
            index.candidates_within(&Value::Float64(42.0), 600),
            "a float of it"
        );
        let strings: Vec<Value> = std::iter::repeat_n(Value::from("42"), 600).collect();
        let index = self::tests::index(&strings);
        assert_eq!(found(&index, Value::from("42")).len(), 600);
        // A number finds no string, whatever it spells (GA04).
        assert_eq!(found(&index, Value::Int64(42)), Vec::<u64>::new());
        assert!(
            index.candidates_within(&Value::Int64(42), 600),
            "a number, which finds none"
        );
        assert!(
            index.candidates_within(&Value::from("42"), 600),
            "the string"
        );
    }
}
