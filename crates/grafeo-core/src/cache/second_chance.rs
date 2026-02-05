//! Second-chance (clock) LRU cache implementation.
//!
//! This cache uses the second-chance algorithm (also known as clock algorithm)
//! for eviction. Each entry has an "accessed" flag that is set on every access
//! and cleared during eviction scans. Entries that have been accessed get a
//! "second chance" before eviction.
//!
//! The key advantage over traditional LRU is that access marking is lock-free
//! (using atomic bools), reducing contention on the hot read path.

use std::collections::VecDeque;
use std::hash::Hash;
use std::sync::atomic::{AtomicBool, Ordering};

use hashbrown::HashMap;

/// LRU cache with second-chance eviction algorithm.
///
/// Access marking is lock-free (atomic bool), reducing contention
/// on the hot read path. Only eviction requires exclusive access.
///
/// # Example
///
/// ```
/// use grafeo_core::cache::SecondChanceLru;
///
/// let mut cache = SecondChanceLru::new(3);
/// cache.insert("a", 1);
/// cache.insert("b", 2);
/// cache.insert("c", 3);
///
/// // Access "a" to give it a second chance
/// let _ = cache.get(&"a");
///
/// // Insert "d" - should evict "b" (not accessed), not "a"
/// cache.insert("d", 4);
///
/// assert!(cache.get(&"a").is_some());
/// assert!(cache.get(&"b").is_none()); // evicted
/// ```
pub struct SecondChanceLru<K, V> {
    /// Map from key to (value, accessed_flag).
    cache: HashMap<K, (V, AtomicBool)>,
    /// Eviction order queue (clock hand).
    queue: VecDeque<K>,
    /// Maximum cache entries.
    capacity: usize,
}

impl<K: Hash + Eq + Clone, V> SecondChanceLru<K, V> {
    /// Creates a new cache with the given capacity.
    ///
    /// The cache will hold at most `capacity` entries. When full, the
    /// second-chance algorithm determines which entry to evict.
    #[must_use]
    pub fn new(capacity: usize) -> Self {
        Self {
            cache: HashMap::with_capacity(capacity),
            queue: VecDeque::with_capacity(capacity),
            capacity,
        }
    }

    /// Gets a value, marking it as accessed (lock-free flag set).
    ///
    /// Returns `None` if the key is not in the cache.
    #[must_use]
    pub fn get(&self, key: &K) -> Option<&V> {
        self.cache.get(key).map(|(val, accessed)| {
            // Mark as accessed - atomic, no lock needed
            accessed.store(true, Ordering::Relaxed);
            val
        })
    }

    /// Gets a mutable reference to a value, marking it as accessed.
    ///
    /// Returns `None` if the key is not in the cache.
    pub fn get_mut(&mut self, key: &K) -> Option<&mut V> {
        self.cache.get_mut(key).map(|(val, accessed)| {
            accessed.store(true, Ordering::Relaxed);
            val
        })
    }

    /// Checks if the cache contains the given key.
    ///
    /// This does NOT mark the entry as accessed.
    #[must_use]
    pub fn contains_key(&self, key: &K) -> bool {
        self.cache.contains_key(key)
    }

    /// Inserts a value, evicting if at capacity.
    ///
    /// If the key already exists, the value is updated and marked as accessed.
    /// If the cache is at capacity and this is a new key, one entry is evicted
    /// using the second-chance algorithm.
    pub fn insert(&mut self, key: K, value: V) {
        // If key exists, update in place
        if let Some((existing, accessed)) = self.cache.get_mut(&key) {
            *existing = value;
            accessed.store(true, Ordering::Relaxed);
            return;
        }

        // Need to insert new entry - evict if at capacity
        if self.cache.len() >= self.capacity {
            self.evict_one();
        }

        // New entries start with accessed=false; only get() marks as accessed
        self.cache
            .insert(key.clone(), (value, AtomicBool::new(false)));
        self.queue.push_back(key);
    }

    /// Removes a specific key from the cache.
    ///
    /// Returns the value if it was present.
    pub fn remove(&mut self, key: &K) -> Option<V> {
        self.cache.remove(key).map(|(v, _)| v)
    }

    /// Evicts one item using the second-chance algorithm.
    ///
    /// Scans entries in FIFO order. If an entry's accessed flag is set,
    /// clears it and gives the entry a "second chance" by moving it to
    /// the back of the queue. Otherwise, evicts the entry.
    fn evict_one(&mut self) {
        // Limit iterations to prevent infinite loop if all entries are accessed
        let max_iterations = self.queue.len() * 2;

        for _ in 0..max_iterations {
            if let Some(key) = self.queue.pop_front() {
                if let Some((_, accessed)) = self.cache.get(&key) {
                    if accessed.swap(false, Ordering::Relaxed) {
                        // Was accessed - give second chance
                        self.queue.push_back(key);
                    } else {
                        // Not accessed - evict
                        self.cache.remove(&key);
                        return;
                    }
                }
                // Key not in cache (shouldn't happen, but handle gracefully)
            } else {
                return; // Queue is empty
            }
        }

        // All entries accessed - evict the first one anyway
        if let Some(key) = self.queue.pop_front() {
            self.cache.remove(&key);
        }
    }

    /// Returns the number of entries in the cache.
    #[must_use]
    pub fn len(&self) -> usize {
        self.cache.len()
    }

    /// Returns true if the cache is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.cache.is_empty()
    }

    /// Clears all entries from the cache.
    pub fn clear(&mut self) {
        self.cache.clear();
        self.queue.clear();
    }

    /// Returns the maximum capacity of the cache.
    #[must_use]
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Returns an iterator over the keys in the cache.
    ///
    /// The order is unspecified and does not reflect access patterns.
    pub fn keys(&self) -> impl Iterator<Item = &K> {
        self.cache.keys()
    }

    /// Returns an iterator over the values in the cache.
    ///
    /// The order is unspecified. Values are not marked as accessed.
    pub fn values(&self) -> impl Iterator<Item = &V> {
        self.cache.values().map(|(v, _)| v)
    }

    /// Returns an iterator over key-value pairs in the cache.
    ///
    /// The order is unspecified. Values are not marked as accessed.
    pub fn iter(&self) -> impl Iterator<Item = (&K, &V)> {
        self.cache.iter().map(|(k, (v, _))| (k, v))
    }
}

impl<K: Hash + Eq + Clone, V> Default for SecondChanceLru<K, V> {
    fn default() -> Self {
        Self::new(16)
    }
}

impl<K: Hash + Eq + Clone, V: Clone> Clone for SecondChanceLru<K, V> {
    fn clone(&self) -> Self {
        let mut cache = HashMap::with_capacity(self.capacity);
        for (k, (v, accessed)) in &self.cache {
            cache.insert(
                k.clone(),
                (v.clone(), AtomicBool::new(accessed.load(Ordering::Relaxed))),
            );
        }
        Self {
            cache,
            queue: self.queue.clone(),
            capacity: self.capacity,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_operations() {
        let mut cache = SecondChanceLru::new(3);

        cache.insert("a", 1);
        cache.insert("b", 2);
        cache.insert("c", 3);

        assert_eq!(cache.len(), 3);
        assert_eq!(*cache.get(&"a").unwrap(), 1);
        assert_eq!(*cache.get(&"b").unwrap(), 2);
        assert_eq!(*cache.get(&"c").unwrap(), 3);
        assert!(cache.get(&"d").is_none());
    }

    #[test]
    fn test_second_chance_eviction() {
        let mut cache = SecondChanceLru::new(3);

        cache.insert("a", 1);
        cache.insert("b", 2);
        cache.insert("c", 3);

        // Access "a" to give it second chance
        let _ = cache.get(&"a");

        // Insert "d" - should evict "b" (oldest non-accessed), not "a"
        cache.insert("d", 4);

        assert!(
            cache.get(&"a").is_some(),
            "a should survive (second chance)"
        );
        assert!(cache.get(&"b").is_none(), "b should be evicted");
        assert!(cache.get(&"c").is_some(), "c should survive");
        assert!(cache.get(&"d").is_some(), "d was just inserted");
    }

    #[test]
    fn test_capacity_respected() {
        let mut cache = SecondChanceLru::new(2);

        cache.insert("a", 1);
        cache.insert("b", 2);
        cache.insert("c", 3);

        assert_eq!(cache.len(), 2);
        assert_eq!(cache.capacity(), 2);
    }

    #[test]
    fn test_update_existing() {
        let mut cache = SecondChanceLru::new(2);

        cache.insert("a", 1);
        cache.insert("a", 2);

        assert_eq!(cache.len(), 1);
        assert_eq!(*cache.get(&"a").unwrap(), 2);
    }

    #[test]
    fn test_remove() {
        let mut cache = SecondChanceLru::new(3);

        cache.insert("a", 1);
        cache.insert("b", 2);

        let removed = cache.remove(&"a");
        assert_eq!(removed, Some(1));
        assert!(cache.get(&"a").is_none());
        assert_eq!(cache.len(), 1);
    }

    #[test]
    fn test_clear() {
        let mut cache = SecondChanceLru::new(3);

        cache.insert("a", 1);
        cache.insert("b", 2);
        cache.clear();

        assert!(cache.is_empty());
        assert_eq!(cache.len(), 0);
    }

    #[test]
    fn test_get_mut() {
        let mut cache = SecondChanceLru::new(3);

        cache.insert("a", 1);

        if let Some(val) = cache.get_mut(&"a") {
            *val = 10;
        }

        assert_eq!(*cache.get(&"a").unwrap(), 10);
    }

    #[test]
    fn test_contains_key() {
        let mut cache = SecondChanceLru::new(3);

        cache.insert("a", 1);

        assert!(cache.contains_key(&"a"));
        assert!(!cache.contains_key(&"b"));
    }

    #[test]
    fn test_all_accessed_eviction() {
        let mut cache = SecondChanceLru::new(2);

        cache.insert("a", 1);
        cache.insert("b", 2);

        // Access both entries
        let _ = cache.get(&"a");
        let _ = cache.get(&"b");

        // Insert new entry - must evict one even though both are accessed
        cache.insert("c", 3);

        assert_eq!(cache.len(), 2);
        // One of a or b should be evicted, c should be present
        assert!(cache.get(&"c").is_some());
    }

    #[test]
    fn test_iterators() {
        let mut cache = SecondChanceLru::new(3);

        cache.insert("a", 1);
        cache.insert("b", 2);

        let keys: Vec<_> = cache.keys().collect();
        assert_eq!(keys.len(), 2);

        let values: Vec<_> = cache.values().collect();
        assert_eq!(values.len(), 2);

        let pairs: Vec<_> = cache.iter().collect();
        assert_eq!(pairs.len(), 2);
    }

    #[test]
    fn test_clone() {
        let mut cache = SecondChanceLru::new(3);

        cache.insert("a", 1);
        cache.insert("b", 2);
        let _ = cache.get(&"a"); // Mark "a" as accessed

        let cloned = cache.clone();

        assert_eq!(cloned.len(), 2);
        assert_eq!(*cloned.get(&"a").unwrap(), 1);
        assert_eq!(*cloned.get(&"b").unwrap(), 2);
    }
}
