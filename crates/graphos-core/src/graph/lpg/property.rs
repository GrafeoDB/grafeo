//! Property storage for the LPG model.
//!
//! This module provides columnar property storage optimized for
//! efficient scanning and filtering. Includes zone map support for
//! predicate pushdown and data skipping.

use crate::index::zone_map::ZoneMapEntry;
use graphos_common::types::{EdgeId, NodeId, PropertyKey, Value};
use graphos_common::utils::hash::FxHashMap;
use parking_lot::RwLock;
use std::cmp::Ordering;
use std::hash::Hash;
use std::marker::PhantomData;

/// Comparison operator for zone map predicate checks.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompareOp {
    /// Equal to value.
    Eq,
    /// Not equal to value.
    Ne,
    /// Less than value.
    Lt,
    /// Less than or equal to value.
    Le,
    /// Greater than value.
    Gt,
    /// Greater than or equal to value.
    Ge,
}

/// Trait for entity IDs that can be used as property storage keys.
pub trait EntityId: Copy + Eq + Hash + 'static {}

impl EntityId for NodeId {}
impl EntityId for EdgeId {}

/// Columnar property storage.
///
/// Properties are stored in a columnar format where each property key
/// has its own column. This enables efficient filtering and scanning
/// of specific properties across many entities.
///
/// Generic over the entity ID type (NodeId or EdgeId).
pub struct PropertyStorage<Id: EntityId = NodeId> {
    /// Map from property key to column.
    columns: RwLock<FxHashMap<PropertyKey, PropertyColumn<Id>>>,
    _marker: PhantomData<Id>,
}

impl<Id: EntityId> PropertyStorage<Id> {
    /// Creates a new property storage.
    #[must_use]
    pub fn new() -> Self {
        Self {
            columns: RwLock::new(FxHashMap::default()),
            _marker: PhantomData,
        }
    }

    /// Sets a property value for an entity.
    pub fn set(&self, id: Id, key: PropertyKey, value: Value) {
        let mut columns = self.columns.write();
        columns
            .entry(key)
            .or_insert_with(PropertyColumn::new)
            .set(id, value);
    }

    /// Gets a property value for an entity.
    #[must_use]
    pub fn get(&self, id: Id, key: &PropertyKey) -> Option<Value> {
        let columns = self.columns.read();
        columns.get(key).and_then(|col| col.get(id))
    }

    /// Removes a property value for an entity.
    pub fn remove(&self, id: Id, key: &PropertyKey) -> Option<Value> {
        let mut columns = self.columns.write();
        columns.get_mut(key).and_then(|col| col.remove(id))
    }

    /// Removes all properties for an entity.
    pub fn remove_all(&self, id: Id) {
        let mut columns = self.columns.write();
        for col in columns.values_mut() {
            col.remove(id);
        }
    }

    /// Gets all properties for an entity.
    #[must_use]
    pub fn get_all(&self, id: Id) -> FxHashMap<PropertyKey, Value> {
        let columns = self.columns.read();
        let mut result = FxHashMap::default();
        for (key, col) in columns.iter() {
            if let Some(value) = col.get(id) {
                result.insert(key.clone(), value);
            }
        }
        result
    }

    /// Returns the number of property columns.
    #[must_use]
    pub fn column_count(&self) -> usize {
        self.columns.read().len()
    }

    /// Returns the keys of all columns.
    #[must_use]
    pub fn keys(&self) -> Vec<PropertyKey> {
        self.columns.read().keys().cloned().collect()
    }

    /// Gets a column by key for bulk access.
    #[must_use]
    pub fn column(&self, key: &PropertyKey) -> Option<PropertyColumnRef<'_, Id>> {
        let columns = self.columns.read();
        if columns.contains_key(key) {
            Some(PropertyColumnRef {
                _guard: columns,
                key: key.clone(),
                _marker: PhantomData,
            })
        } else {
            None
        }
    }

    /// Checks if a predicate on a property might match any values.
    ///
    /// Returns `true` if the property column might contain matching values,
    /// `false` if it definitely doesn't. Returns `true` if the property doesn't exist.
    #[must_use]
    pub fn might_match(&self, key: &PropertyKey, op: CompareOp, value: &Value) -> bool {
        let columns = self.columns.read();
        columns
            .get(key)
            .map(|col| col.might_match(op, value))
            .unwrap_or(true) // No column = assume might match (conservative)
    }

    /// Gets the zone map for a property column.
    #[must_use]
    pub fn zone_map(&self, key: &PropertyKey) -> Option<ZoneMapEntry> {
        let columns = self.columns.read();
        columns.get(key).map(|col| col.zone_map().clone())
    }

    /// Rebuilds zone maps for all columns (call after bulk removes).
    pub fn rebuild_zone_maps(&self) {
        let mut columns = self.columns.write();
        for col in columns.values_mut() {
            col.rebuild_zone_map();
        }
    }
}

impl<Id: EntityId> Default for PropertyStorage<Id> {
    fn default() -> Self {
        Self::new()
    }
}

/// A single property column with zone map tracking.
///
/// Stores values for a specific property key across all entities.
/// Maintains zone map metadata (min/max/null_count) for predicate pushdown.
pub struct PropertyColumn<Id: EntityId = NodeId> {
    /// Sparse storage: entity ID -> value.
    /// For dense properties, this could be optimized to a flat vector.
    values: FxHashMap<Id, Value>,
    /// Zone map tracking min/max/null_count for predicate pushdown.
    zone_map: ZoneMapEntry,
    /// Whether zone map needs rebuild (after removes).
    zone_map_dirty: bool,
}

impl<Id: EntityId> PropertyColumn<Id> {
    /// Creates a new empty column.
    #[must_use]
    pub fn new() -> Self {
        Self {
            values: FxHashMap::default(),
            zone_map: ZoneMapEntry::new(),
            zone_map_dirty: false,
        }
    }

    /// Sets a value for an entity.
    pub fn set(&mut self, id: Id, value: Value) {
        // Update zone map incrementally
        self.update_zone_map_on_insert(&value);
        self.values.insert(id, value);
    }

    /// Updates zone map when inserting a value.
    fn update_zone_map_on_insert(&mut self, value: &Value) {
        self.zone_map.row_count += 1;

        if matches!(value, Value::Null) {
            self.zone_map.null_count += 1;
            return;
        }

        // Update min
        match &self.zone_map.min {
            None => self.zone_map.min = Some(value.clone()),
            Some(current) => {
                if compare_values(value, current) == Some(Ordering::Less) {
                    self.zone_map.min = Some(value.clone());
                }
            }
        }

        // Update max
        match &self.zone_map.max {
            None => self.zone_map.max = Some(value.clone()),
            Some(current) => {
                if compare_values(value, current) == Some(Ordering::Greater) {
                    self.zone_map.max = Some(value.clone());
                }
            }
        }
    }

    /// Gets a value for an entity.
    #[must_use]
    pub fn get(&self, id: Id) -> Option<Value> {
        self.values.get(&id).cloned()
    }

    /// Removes a value for an entity.
    pub fn remove(&mut self, id: Id) -> Option<Value> {
        let removed = self.values.remove(&id);
        if removed.is_some() {
            // Mark zone map as dirty - would need full rebuild for accurate min/max
            self.zone_map_dirty = true;
        }
        removed
    }

    /// Returns the number of values in this column.
    #[must_use]
    #[allow(dead_code)]
    pub fn len(&self) -> usize {
        self.values.len()
    }

    /// Returns true if this column is empty.
    #[must_use]
    #[allow(dead_code)]
    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }

    /// Iterates over all (id, value) pairs.
    #[allow(dead_code)]
    pub fn iter(&self) -> impl Iterator<Item = (Id, &Value)> {
        self.values.iter().map(|(&id, v)| (id, v))
    }

    /// Returns the zone map for this column.
    #[must_use]
    pub fn zone_map(&self) -> &ZoneMapEntry {
        &self.zone_map
    }

    /// Checks if a predicate might match any values in this column.
    ///
    /// Returns `true` if the column might contain matching values,
    /// `false` if it definitely doesn't (allowing the column to be skipped).
    #[must_use]
    pub fn might_match(&self, op: CompareOp, value: &Value) -> bool {
        if self.zone_map_dirty {
            // Conservative: can't skip if zone map is stale
            return true;
        }

        match op {
            CompareOp::Eq => self.zone_map.might_contain_equal(value),
            CompareOp::Ne => {
                // Can only skip if all values are equal to the value
                // (which means min == max == value)
                match (&self.zone_map.min, &self.zone_map.max) {
                    (Some(min), Some(max)) => {
                        !(compare_values(min, value) == Some(Ordering::Equal)
                            && compare_values(max, value) == Some(Ordering::Equal))
                    }
                    _ => true,
                }
            }
            CompareOp::Lt => self.zone_map.might_contain_less_than(value, false),
            CompareOp::Le => self.zone_map.might_contain_less_than(value, true),
            CompareOp::Gt => self.zone_map.might_contain_greater_than(value, false),
            CompareOp::Ge => self.zone_map.might_contain_greater_than(value, true),
        }
    }

    /// Rebuilds zone map from current values.
    pub fn rebuild_zone_map(&mut self) {
        let mut zone_map = ZoneMapEntry::new();

        for value in self.values.values() {
            zone_map.row_count += 1;

            if matches!(value, Value::Null) {
                zone_map.null_count += 1;
                continue;
            }

            // Update min
            match &zone_map.min {
                None => zone_map.min = Some(value.clone()),
                Some(current) => {
                    if compare_values(value, current) == Some(Ordering::Less) {
                        zone_map.min = Some(value.clone());
                    }
                }
            }

            // Update max
            match &zone_map.max {
                None => zone_map.max = Some(value.clone()),
                Some(current) => {
                    if compare_values(value, current) == Some(Ordering::Greater) {
                        zone_map.max = Some(value.clone());
                    }
                }
            }
        }

        self.zone_map = zone_map;
        self.zone_map_dirty = false;
    }
}

/// Compares two values for ordering.
fn compare_values(a: &Value, b: &Value) -> Option<Ordering> {
    match (a, b) {
        (Value::Int64(a), Value::Int64(b)) => Some(a.cmp(b)),
        (Value::Float64(a), Value::Float64(b)) => a.partial_cmp(b),
        (Value::String(a), Value::String(b)) => Some(a.cmp(b)),
        (Value::Bool(a), Value::Bool(b)) => Some(a.cmp(b)),
        (Value::Int64(a), Value::Float64(b)) => (*a as f64).partial_cmp(b),
        (Value::Float64(a), Value::Int64(b)) => a.partial_cmp(&(*b as f64)),
        _ => None,
    }
}

impl<Id: EntityId> Default for PropertyColumn<Id> {
    fn default() -> Self {
        Self::new()
    }
}

/// A reference to a property column for bulk access.
pub struct PropertyColumnRef<'a, Id: EntityId = NodeId> {
    _guard: parking_lot::RwLockReadGuard<'a, FxHashMap<PropertyKey, PropertyColumn<Id>>>,
    #[allow(dead_code)]
    key: PropertyKey,
    _marker: PhantomData<Id>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_property_storage_basic() {
        let storage = PropertyStorage::new();

        let node1 = NodeId::new(1);
        let node2 = NodeId::new(2);
        let name_key = PropertyKey::new("name");
        let age_key = PropertyKey::new("age");

        storage.set(node1, name_key.clone(), "Alice".into());
        storage.set(node1, age_key.clone(), 30i64.into());
        storage.set(node2, name_key.clone(), "Bob".into());

        assert_eq!(
            storage.get(node1, &name_key),
            Some(Value::String("Alice".into()))
        );
        assert_eq!(storage.get(node1, &age_key), Some(Value::Int64(30)));
        assert_eq!(
            storage.get(node2, &name_key),
            Some(Value::String("Bob".into()))
        );
        assert!(storage.get(node2, &age_key).is_none());
    }

    #[test]
    fn test_property_storage_remove() {
        let storage = PropertyStorage::new();

        let node = NodeId::new(1);
        let key = PropertyKey::new("name");

        storage.set(node, key.clone(), "Alice".into());
        assert!(storage.get(node, &key).is_some());

        let removed = storage.remove(node, &key);
        assert!(removed.is_some());
        assert!(storage.get(node, &key).is_none());
    }

    #[test]
    fn test_property_storage_get_all() {
        let storage = PropertyStorage::new();

        let node = NodeId::new(1);
        storage.set(node, PropertyKey::new("name"), "Alice".into());
        storage.set(node, PropertyKey::new("age"), 30i64.into());
        storage.set(node, PropertyKey::new("active"), true.into());

        let props = storage.get_all(node);
        assert_eq!(props.len(), 3);
    }

    #[test]
    fn test_property_storage_remove_all() {
        let storage = PropertyStorage::new();

        let node = NodeId::new(1);
        storage.set(node, PropertyKey::new("name"), "Alice".into());
        storage.set(node, PropertyKey::new("age"), 30i64.into());

        storage.remove_all(node);

        assert!(storage.get(node, &PropertyKey::new("name")).is_none());
        assert!(storage.get(node, &PropertyKey::new("age")).is_none());
    }

    #[test]
    fn test_property_column() {
        let mut col = PropertyColumn::new();

        col.set(NodeId::new(1), "Alice".into());
        col.set(NodeId::new(2), "Bob".into());

        assert_eq!(col.len(), 2);
        assert!(!col.is_empty());

        assert_eq!(col.get(NodeId::new(1)), Some(Value::String("Alice".into())));

        col.remove(NodeId::new(1));
        assert!(col.get(NodeId::new(1)).is_none());
        assert_eq!(col.len(), 1);
    }
}
