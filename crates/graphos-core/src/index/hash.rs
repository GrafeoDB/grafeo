//! Hash index for primary key lookups.

use graphos_common::types::NodeId;
use graphos_common::utils::hash::FxHashMap;
use parking_lot::RwLock;
use std::hash::Hash;

/// A concurrent hash index for fast key lookups.
///
/// This index provides O(1) average-case lookup, insertion, and deletion.
pub struct HashIndex<K: Hash + Eq, V: Copy> {
    /// The underlying hash map.
    map: RwLock<FxHashMap<K, V>>,
}

impl<K: Hash + Eq, V: Copy> HashIndex<K, V> {
    /// Creates a new empty hash index.
    #[must_use]
    pub fn new() -> Self {
        Self {
            map: RwLock::new(FxHashMap::default()),
        }
    }

    /// Creates a new hash index with the given capacity.
    #[must_use]
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            map: RwLock::new(FxHashMap::with_capacity_and_hasher(
                capacity,
                Default::default(),
            )),
        }
    }

    /// Inserts a key-value pair into the index.
    ///
    /// Returns the previous value if the key was already present.
    pub fn insert(&self, key: K, value: V) -> Option<V> {
        self.map.write().insert(key, value)
    }

    /// Gets the value for a key.
    pub fn get(&self, key: &K) -> Option<V> {
        self.map.read().get(key).copied()
    }

    /// Removes a key from the index.
    ///
    /// Returns the value if the key was present.
    pub fn remove(&self, key: &K) -> Option<V> {
        self.map.write().remove(key)
    }

    /// Checks if a key exists in the index.
    pub fn contains(&self, key: &K) -> bool {
        self.map.read().contains_key(key)
    }

    /// Returns the number of entries in the index.
    pub fn len(&self) -> usize {
        self.map.read().len()
    }

    /// Returns true if the index is empty.
    pub fn is_empty(&self) -> bool {
        self.map.read().is_empty()
    }

    /// Clears all entries from the index.
    pub fn clear(&self) {
        self.map.write().clear();
    }
}

impl<K: Hash + Eq, V: Copy> Default for HashIndex<K, V> {
    fn default() -> Self {
        Self::new()
    }
}

/// A hash index from string keys to NodeIds.
pub type StringKeyIndex = HashIndex<String, NodeId>;

/// A hash index from NodeIds to NodeIds.
pub type NodeIdIndex = HashIndex<NodeId, NodeId>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hash_index_basic() {
        let index: HashIndex<u64, NodeId> = HashIndex::new();

        index.insert(1, NodeId::new(100));
        index.insert(2, NodeId::new(200));

        assert_eq!(index.get(&1), Some(NodeId::new(100)));
        assert_eq!(index.get(&2), Some(NodeId::new(200)));
        assert_eq!(index.get(&3), None);
    }

    #[test]
    fn test_hash_index_update() {
        let index: HashIndex<u64, NodeId> = HashIndex::new();

        index.insert(1, NodeId::new(100));
        let old = index.insert(1, NodeId::new(200));

        assert_eq!(old, Some(NodeId::new(100)));
        assert_eq!(index.get(&1), Some(NodeId::new(200)));
    }

    #[test]
    fn test_hash_index_remove() {
        let index: HashIndex<u64, NodeId> = HashIndex::new();

        index.insert(1, NodeId::new(100));
        assert!(index.contains(&1));

        let removed = index.remove(&1);
        assert_eq!(removed, Some(NodeId::new(100)));
        assert!(!index.contains(&1));
    }

    #[test]
    fn test_hash_index_len() {
        let index: HashIndex<u64, NodeId> = HashIndex::new();

        assert!(index.is_empty());
        assert_eq!(index.len(), 0);

        index.insert(1, NodeId::new(100));
        index.insert(2, NodeId::new(200));

        assert!(!index.is_empty());
        assert_eq!(index.len(), 2);
    }

    #[test]
    fn test_hash_index_clear() {
        let index: HashIndex<u64, NodeId> = HashIndex::new();

        index.insert(1, NodeId::new(100));
        index.insert(2, NodeId::new(200));

        index.clear();

        assert!(index.is_empty());
        assert_eq!(index.get(&1), None);
    }

    #[test]
    fn test_string_key_index() {
        let index: StringKeyIndex = HashIndex::new();

        index.insert("alice".to_string(), NodeId::new(1));
        index.insert("bob".to_string(), NodeId::new(2));

        assert_eq!(index.get(&"alice".to_string()), Some(NodeId::new(1)));
        assert_eq!(index.get(&"bob".to_string()), Some(NodeId::new(2)));
    }
}
