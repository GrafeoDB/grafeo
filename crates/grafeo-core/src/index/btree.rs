//! BTree index for ordered data and range queries.
//!
//! Use this when you need to answer range queries like `age > 30 AND age < 50`
//! or when you need min/max values. O(log n) lookups but efficient range scans.

use grafeo_common::types::NodeId;
use parking_lot::RwLock;
use std::collections::BTreeMap;
use std::ops::RangeBounds;

/// A thread-safe BTree index for range queries.
///
/// Unlike [`HashIndex`](super::HashIndex), this maintains keys in sorted order,
/// making it perfect for range queries and finding min/max values.
///
/// # Example
///
/// ```
/// use grafeo_core::index::BTreeIndex;
/// use grafeo_common::types::NodeId;
///
/// let index: BTreeIndex<i64, NodeId> = BTreeIndex::new();
/// index.insert(25, NodeId::new(1));  // age 25
/// index.insert(35, NodeId::new(2));  // age 35
/// index.insert(45, NodeId::new(3));  // age 45
///
/// // Find all people between 30 and 50
/// let results = index.range(30..50);
/// assert_eq!(results.len(), 2);  // ages 35 and 45
/// ```
pub struct BTreeIndex<K: Ord, V: Copy> {
    /// The underlying BTree map.
    map: RwLock<BTreeMap<K, V>>,
}

impl<K: Ord + Clone, V: Copy> BTreeIndex<K, V> {
    /// Creates a new empty BTree index.
    #[must_use]
    pub fn new() -> Self {
        Self {
            map: RwLock::new(BTreeMap::new()),
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

    /// Returns all values in the given range.
    pub fn range<R: RangeBounds<K>>(&self, range: R) -> Vec<(K, V)> {
        self.map
            .read()
            .range(range)
            .map(|(k, v)| (k.clone(), *v))
            .collect()
    }

    /// Returns the minimum key-value pair.
    pub fn min(&self) -> Option<(K, V)> {
        self.map
            .read()
            .first_key_value()
            .map(|(k, v)| (k.clone(), *v))
    }

    /// Returns the maximum key-value pair.
    pub fn max(&self) -> Option<(K, V)> {
        self.map
            .read()
            .last_key_value()
            .map(|(k, v)| (k.clone(), *v))
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

impl<K: Ord + Clone, V: Copy> Default for BTreeIndex<K, V> {
    fn default() -> Self {
        Self::new()
    }
}

/// A BTree index from i64 keys to NodeIds.
pub type Int64Index = BTreeIndex<i64, NodeId>;

/// A BTree index from f64 keys to NodeIds.
/// Note: f64 doesn't implement Ord, so we use a wrapper.
pub type Float64Index = BTreeIndex<OrderedFloat, NodeId>;

/// A wrapper around f64 that implements Ord for use in BTreeIndex.
///
/// Since f64 doesn't implement Ord (due to NaN), we need this wrapper.
/// NaN values are treated as equal to each other.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct OrderedFloat(pub f64);

impl Eq for OrderedFloat {}

impl PartialOrd for OrderedFloat {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for OrderedFloat {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.0
            .partial_cmp(&other.0)
            .unwrap_or(std::cmp::Ordering::Equal)
    }
}

impl From<f64> for OrderedFloat {
    fn from(f: f64) -> Self {
        Self(f)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_btree_basic() {
        let index: Int64Index = BTreeIndex::new();

        index.insert(10, NodeId::new(100));
        index.insert(20, NodeId::new(200));
        index.insert(30, NodeId::new(300));

        assert_eq!(index.get(&10), Some(NodeId::new(100)));
        assert_eq!(index.get(&20), Some(NodeId::new(200)));
        assert_eq!(index.get(&15), None);
    }

    #[test]
    fn test_btree_range() {
        let index: Int64Index = BTreeIndex::new();

        index.insert(10, NodeId::new(100));
        index.insert(20, NodeId::new(200));
        index.insert(30, NodeId::new(300));
        index.insert(40, NodeId::new(400));

        let range = index.range(15..=35);
        assert_eq!(range.len(), 2);
        assert!(range.contains(&(20, NodeId::new(200))));
        assert!(range.contains(&(30, NodeId::new(300))));
    }

    #[test]
    fn test_btree_min_max() {
        let index: Int64Index = BTreeIndex::new();

        assert!(index.min().is_none());
        assert!(index.max().is_none());

        index.insert(20, NodeId::new(200));
        index.insert(10, NodeId::new(100));
        index.insert(30, NodeId::new(300));

        assert_eq!(index.min(), Some((10, NodeId::new(100))));
        assert_eq!(index.max(), Some((30, NodeId::new(300))));
    }

    #[test]
    fn test_btree_remove() {
        let index: Int64Index = BTreeIndex::new();

        index.insert(10, NodeId::new(100));
        index.insert(20, NodeId::new(200));

        let removed = index.remove(&10);
        assert_eq!(removed, Some(NodeId::new(100)));
        assert!(!index.contains(&10));
        assert_eq!(index.len(), 1);
    }

    #[test]
    fn test_float64_index() {
        let index: Float64Index = BTreeIndex::new();

        index.insert(1.5.into(), NodeId::new(100));
        index.insert(2.5.into(), NodeId::new(200));
        index.insert(3.5.into(), NodeId::new(300));

        let range = index.range(OrderedFloat(1.0)..OrderedFloat(3.0));
        assert_eq!(range.len(), 2);
    }
}
