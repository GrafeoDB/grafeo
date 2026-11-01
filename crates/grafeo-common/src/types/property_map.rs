//! Compact property storage for nodes and edges.
//!
//! [`PropertyMap`] replaces `BTreeMap<PropertyKey, Value>` on user-facing
//! [`Node`](crate::graph::lpg::Node) and [`Edge`](crate::graph::lpg::Edge)
//! types.  For the typical case of 1-4 properties the data lives entirely
//! inline (no heap allocation), and lookups use a fast linear scan instead
//! of tree traversal.

use std::collections::BTreeMap;
use std::fmt;

use smallvec::SmallVec;

use super::value::{PropertyKey, Value};

/// Inline capacity: entries stored on the stack without allocation.
///
/// Covers the common case of 1-4 properties per entity.  Beyond this the
/// backing `SmallVec` spills to the heap but remains a flat array, so
/// iteration is still cache-friendly.
const INLINE_CAP: usize = 4;

/// A compact, ordered property map.
///
/// Behaves like a `BTreeMap<PropertyKey, Value>` but stores small maps
/// inline (up to 4 entries) and uses linear scan for lookups.  For the
/// sizes typical in graph workloads this is both faster and more
/// memory-efficient than a tree.
#[derive(Clone, PartialEq)]
pub struct PropertyMap {
    entries: SmallVec<[(PropertyKey, Value); INLINE_CAP]>,
}

impl PropertyMap {
    /// Creates an empty property map.
    #[inline]
    #[must_use]
    pub fn new() -> Self {
        Self {
            entries: SmallVec::new(),
        }
    }

    /// Creates a property map with pre-allocated capacity.
    #[inline]
    #[must_use]
    pub fn with_capacity(cap: usize) -> Self {
        Self {
            entries: SmallVec::with_capacity(cap),
        }
    }

    /// Returns the number of properties.
    #[inline]
    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Returns `true` if the map contains no properties.
    #[inline]
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Looks up a property by key.
    #[must_use]
    pub fn get(&self, key: &PropertyKey) -> Option<&Value> {
        self.entries.iter().find(|(k, _)| k == key).map(|(_, v)| v)
    }

    /// Returns `true` if the map contains the given key.
    #[must_use]
    pub fn contains_key(&self, key: &PropertyKey) -> bool {
        self.entries.iter().any(|(k, _)| k == key)
    }

    /// Inserts a property, replacing any existing value for the same key.
    ///
    /// Returns the previous value if the key was already present.
    pub fn insert(&mut self, key: PropertyKey, value: Value) -> Option<Value> {
        for entry in &mut self.entries {
            if entry.0 == key {
                let old = std::mem::replace(&mut entry.1, value);
                return Some(old);
            }
        }
        self.entries.push((key, value));
        None
    }

    /// A map of `entries` whose keys are known to be distinct (the columns
    /// of one stored row), without the check for an earlier entry of the
    /// same key that [`insert`](Self::insert) makes: building a node of
    /// 30 properties so is linear instead of quadratic. The entries keep
    /// their order.
    #[must_use]
    pub fn from_distinct(entries: impl IntoIterator<Item = (PropertyKey, Value)>) -> Self {
        let entries: SmallVec<[(PropertyKey, Value); INLINE_CAP]> = entries.into_iter().collect();
        debug_assert!(
            entries
                .iter()
                .enumerate()
                .all(|(i, (key, _))| entries[..i].iter().all(|(earlier, _)| earlier != key)),
            "PropertyMap::from_distinct with a repeated key"
        );
        Self { entries }
    }

    /// Removes a property by key, returning its value if present.
    pub fn remove(&mut self, key: &PropertyKey) -> Option<Value> {
        if let Some(pos) = self.entries.iter().position(|(k, _)| k == key) {
            Some(self.entries.swap_remove(pos).1)
        } else {
            None
        }
    }

    /// Iterates over `(key, value)` pairs.
    pub fn iter(&self) -> impl Iterator<Item = (&PropertyKey, &Value)> {
        self.entries.iter().map(|(k, v)| (k, v))
    }

    /// Converts to a `BTreeMap` (e.g. for serialization into `Value::Map`).
    #[must_use]
    pub fn to_btree_map(&self) -> BTreeMap<PropertyKey, Value> {
        self.entries.iter().cloned().collect()
    }
}

impl Default for PropertyMap {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Debug for PropertyMap {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_map()
            .entries(self.entries.iter().map(|(k, v)| (k, v)))
            .finish()
    }
}

impl FromIterator<(PropertyKey, Value)> for PropertyMap {
    fn from_iter<I: IntoIterator<Item = (PropertyKey, Value)>>(iter: I) -> Self {
        let iter = iter.into_iter();
        let (lower, _) = iter.size_hint();
        let mut map = Self::with_capacity(lower);
        for (k, v) in iter {
            map.insert(k, v);
        }
        map
    }
}

impl IntoIterator for PropertyMap {
    type Item = (PropertyKey, Value);
    type IntoIter = smallvec::IntoIter<[(PropertyKey, Value); INLINE_CAP]>;

    fn into_iter(self) -> Self::IntoIter {
        self.entries.into_iter()
    }
}

impl<'a> IntoIterator for &'a PropertyMap {
    type Item = (&'a PropertyKey, &'a Value);
    type IntoIter = std::iter::Map<
        std::slice::Iter<'a, (PropertyKey, Value)>,
        fn(&'a (PropertyKey, Value)) -> (&'a PropertyKey, &'a Value),
    >;

    fn into_iter(self) -> Self::IntoIter {
        self.entries.iter().map(|(k, v)| (k, v))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn insert_and_get() {
        let mut map = PropertyMap::new();
        map.insert(PropertyKey::new("name"), Value::from("Alix"));
        map.insert(PropertyKey::new("age"), Value::from(30i64));

        assert_eq!(map.len(), 2);
        assert_eq!(
            map.get(&PropertyKey::new("name")).and_then(Value::as_str),
            Some("Alix")
        );
        assert_eq!(
            map.get(&PropertyKey::new("age")).and_then(Value::as_int64),
            Some(30)
        );
    }

    #[test]
    fn insert_replaces_existing() {
        let mut map = PropertyMap::new();
        map.insert(PropertyKey::new("x"), Value::from(1i64));
        let old = map.insert(PropertyKey::new("x"), Value::from(2i64));
        assert_eq!(old, Some(Value::from(1i64)));
        assert_eq!(map.len(), 1);
        assert_eq!(
            map.get(&PropertyKey::new("x")).and_then(Value::as_int64),
            Some(2)
        );
    }

    #[test]
    fn remove() {
        let mut map = PropertyMap::new();
        map.insert(PropertyKey::new("a"), Value::from(1i64));
        map.insert(PropertyKey::new("b"), Value::from(2i64));

        let removed = map.remove(&PropertyKey::new("a"));
        assert_eq!(removed, Some(Value::from(1i64)));
        assert_eq!(map.len(), 1);
        assert!(map.get(&PropertyKey::new("a")).is_none());
        assert!(map.get(&PropertyKey::new("b")).is_some());
    }

    #[test]
    fn contains_key() {
        let mut map = PropertyMap::new();
        map.insert(PropertyKey::new("x"), Value::Null);
        assert!(map.contains_key(&PropertyKey::new("x")));
        assert!(!map.contains_key(&PropertyKey::new("y")));
    }

    #[test]
    fn from_iterator() {
        let pairs = vec![
            (PropertyKey::new("a"), Value::from(1i64)),
            (PropertyKey::new("b"), Value::from(2i64)),
        ];
        let map: PropertyMap = pairs.into_iter().collect();
        assert_eq!(map.len(), 2);
    }

    #[test]
    fn into_iterator() {
        let mut map = PropertyMap::new();
        map.insert(PropertyKey::new("x"), Value::from(1i64));
        map.insert(PropertyKey::new("y"), Value::from(2i64));

        let pairs: Vec<_> = map.into_iter().collect();
        assert_eq!(pairs.len(), 2);
    }

    #[test]
    fn empty() {
        let map = PropertyMap::new();
        assert!(map.is_empty());
        assert_eq!(map.len(), 0);
        assert!(map.get(&PropertyKey::new("x")).is_none());
    }

    #[test]
    fn inline_capacity() {
        // 4 entries should stay inline (no heap allocation)
        let mut map = PropertyMap::new();
        for i in 0..4 {
            map.insert(PropertyKey::new(format!("k{i}")), Value::from(i as i64));
        }
        assert_eq!(map.len(), 4);
        assert!(!map.entries.spilled());
    }

    #[test]
    fn spills_to_heap_beyond_capacity() {
        let mut map = PropertyMap::new();
        for i in 0..10 {
            map.insert(PropertyKey::new(format!("k{i}")), Value::from(i as i64));
        }
        assert_eq!(map.len(), 10);
        assert!(map.entries.spilled());
        // All entries still accessible
        for i in 0..10 {
            assert!(map.contains_key(&PropertyKey::new(format!("k{i}"))));
        }
    }

    #[test]
    fn debug_format() {
        let mut map = PropertyMap::new();
        map.insert(PropertyKey::new("name"), Value::from("Alix"));
        let dbg = format!("{map:?}");
        assert!(dbg.contains("name"));
    }

    #[test]
    fn from_empty_iterator() {
        let map: PropertyMap = std::iter::empty().collect();
        assert!(map.is_empty());
    }

    #[test]
    fn ref_iterator() {
        let mut map = PropertyMap::new();
        map.insert(PropertyKey::new("a"), Value::from(1i64));
        map.insert(PropertyKey::new("b"), Value::from(2i64));
        let pairs: Vec<_> = (&map).into_iter().collect();
        assert_eq!(pairs.len(), 2);
    }

    #[test]
    fn default_is_empty() {
        let map = PropertyMap::default();
        assert!(map.is_empty());
    }

    #[test]
    fn remove_nonexistent() {
        let mut map = PropertyMap::new();
        assert_eq!(map.remove(&PropertyKey::new("missing")), None);
    }

    #[test]
    fn to_btree_map_round_trip() {
        let mut map = PropertyMap::new();
        map.insert(PropertyKey::new("x"), Value::from(1i64));
        map.insert(PropertyKey::new("y"), Value::from(2i64));
        let btree = map.to_btree_map();
        assert_eq!(btree.len(), 2);
        assert_eq!(btree[&PropertyKey::new("x")], Value::from(1i64));
    }
}
