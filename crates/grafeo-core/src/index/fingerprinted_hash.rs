//! Hash index with fingerprint-based fast rejection.
//!
//! Use this when key comparison is expensive (e.g., string keys) and you want
//! to avoid unnecessary comparisons. The fingerprint check rejects ~99.99% of
//! non-matches without comparing full keys.
//!
//! # When to Use
//!
//! | Scenario | Use `HashIndex` | Use `FingerprintedHashIndex` |
//! |----------|-----------------|------------------------------|
//! | Integer keys | ✅ | ❌ (overhead not worth it) |
//! | Short string keys | ✅ | Maybe |
//! | Long string keys | ❌ | ✅ |
//! | Disk-backed storage | ❌ | ✅ (avoids I/O) |
//! | High collision rate | ❌ | ✅ |
//!
//! # Example
//!
//! ```
//! use grafeo_core::index::FingerprintedHashIndex;
//! use grafeo_common::types::NodeId;
//!
//! let index: FingerprintedHashIndex<String, NodeId> = FingerprintedHashIndex::new();
//!
//! index.insert("alice@example.com".to_string(), NodeId::new(1));
//! index.insert("bob@example.com".to_string(), NodeId::new(2));
//!
//! assert_eq!(index.get(&"alice@example.com".to_string()), Some(NodeId::new(1)));
//! assert_eq!(index.get(&"missing@example.com".to_string()), None);
//!
//! // Check rejection statistics
//! let stats = index.stats();
//! println!("Rejection rate: {:.2}%", stats.rejection_rate() * 100.0);
//! ```

use std::hash::Hash;
use std::sync::atomic::{AtomicU64, Ordering};

use grafeo_common::utils::hash::hash_one;
use parking_lot::RwLock;

use super::fingerprint::{FingerprintBucket, FingerprintStats};

/// Default number of shards (power of 2 for efficient modulo).
const DEFAULT_SHARD_COUNT: usize = 64;

/// Atomic version of FingerprintStats for concurrent access.
#[derive(Debug, Default)]
pub struct AtomicFingerprintStats {
    lookups: AtomicU64,
    fingerprint_rejections: AtomicU64,
    full_comparisons: AtomicU64,
}

impl AtomicFingerprintStats {
    /// Creates new empty stats.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Records a lookup operation.
    #[inline]
    pub fn record_lookup(&self) {
        self.lookups.fetch_add(1, Ordering::Relaxed);
    }

    /// Records a fingerprint rejection (fast path).
    #[inline]
    pub fn record_rejection(&self) {
        self.fingerprint_rejections.fetch_add(1, Ordering::Relaxed);
    }

    /// Records a full key comparison (slow path).
    #[inline]
    pub fn record_comparison(&self) {
        self.full_comparisons.fetch_add(1, Ordering::Relaxed);
    }

    /// Returns a snapshot of the current statistics.
    #[must_use]
    pub fn snapshot(&self) -> FingerprintStats {
        FingerprintStats {
            lookups: self.lookups.load(Ordering::Relaxed),
            fingerprint_rejections: self.fingerprint_rejections.load(Ordering::Relaxed),
            full_comparisons: self.full_comparisons.load(Ordering::Relaxed),
        }
    }

    /// Resets all counters to zero.
    pub fn reset(&self) {
        self.lookups.store(0, Ordering::Relaxed);
        self.fingerprint_rejections.store(0, Ordering::Relaxed);
        self.full_comparisons.store(0, Ordering::Relaxed);
    }
}

/// A sharded hash index with fingerprint-based fast rejection.
///
/// This index uses fingerprints (derived from the hash) to quickly reject
/// non-matching entries without expensive full key comparisons. It's particularly
/// useful when:
///
/// - Key comparison is expensive (strings, large structs)
/// - You have high collision rates in buckets
/// - You want to avoid disk I/O for disk-backed indices
///
/// # Concurrency
///
/// The index is sharded into 64 independent buckets, each protected by its own
/// `RwLock`. This provides good concurrent read throughput similar to `DashMap`.
///
/// # Performance
///
/// - Fast path (fingerprint mismatch): No key comparison needed
/// - Slow path (fingerprint match): Full key comparison
/// - Expected rejection rate: ~99.99% for random keys
pub struct FingerprintedHashIndex<K, V> {
    /// Sharded buckets for concurrent access.
    shards: Vec<RwLock<FingerprintBucket<K, V>>>,
    /// Number of shards (power of 2).
    shard_count: usize,
    /// Mask for efficient shard selection.
    shard_mask: usize,
    /// Statistics for observability.
    stats: AtomicFingerprintStats,
}

impl<K: Hash + Eq, V: Copy> FingerprintedHashIndex<K, V> {
    /// Creates a new empty index with default shard count (64).
    #[must_use]
    pub fn new() -> Self {
        Self::with_shard_count(DEFAULT_SHARD_COUNT)
    }

    /// Creates a new index with the specified number of shards.
    ///
    /// The shard count is rounded up to the next power of 2.
    #[must_use]
    pub fn with_shard_count(shard_count: usize) -> Self {
        // Round up to power of 2
        let shard_count = shard_count.next_power_of_two();
        let shard_mask = shard_count - 1;

        let shards = (0..shard_count)
            .map(|_| RwLock::new(FingerprintBucket::new()))
            .collect();

        Self {
            shards,
            shard_count,
            shard_mask,
            stats: AtomicFingerprintStats::new(),
        }
    }

    /// Creates a new index with pre-allocated capacity per shard.
    #[must_use]
    pub fn with_capacity(capacity_per_shard: usize) -> Self {
        let shard_count = DEFAULT_SHARD_COUNT;
        let shard_mask = shard_count - 1;

        let shards = (0..shard_count)
            .map(|_| RwLock::new(FingerprintBucket::with_capacity(capacity_per_shard)))
            .collect();

        Self {
            shards,
            shard_count,
            shard_mask,
            stats: AtomicFingerprintStats::new(),
        }
    }

    /// Returns the shard index for a given hash.
    #[inline]
    fn shard_index(&self, hash: u64) -> usize {
        (hash as usize) & self.shard_mask
    }

    /// Inserts a key-value pair into the index.
    ///
    /// If the key already exists, the value is updated and the old value is returned.
    pub fn insert(&self, key: K, value: V) -> Option<V> {
        let hash = hash_one(&key);
        let shard_idx = self.shard_index(hash);

        let mut shard = self.shards[shard_idx].write();

        // Check if key exists and get old value
        let old_value = shard.get(&key, hash).copied();

        // Insert (updates if exists)
        shard.insert(key, hash, value);

        old_value
    }

    /// Gets the value for a key, using fingerprint for fast rejection.
    ///
    /// This is the key optimization: most non-matching entries are
    /// rejected by fingerprint comparison alone.
    #[must_use]
    pub fn get(&self, key: &K) -> Option<V> {
        let hash = hash_one(key);
        let shard_idx = self.shard_index(hash);

        self.stats.record_lookup();

        let shard = self.shards[shard_idx].read();
        self.get_with_stats(&shard, key, hash)
    }

    /// Internal get that tracks statistics.
    fn get_with_stats(&self, shard: &FingerprintBucket<K, V>, key: &K, hash: u64) -> Option<V> {
        // Manual iteration to track stats
        let fp = super::fingerprint::fingerprint(hash);

        for (k, entry) in shard.iter_entries() {
            if entry.fingerprint != fp {
                self.stats.record_rejection();
                continue;
            }

            self.stats.record_comparison();
            if k == key {
                return Some(entry.value);
            }
        }

        None
    }

    /// Removes a key from the index.
    ///
    /// Returns the value if the key was present.
    pub fn remove(&self, key: &K) -> Option<V> {
        let hash = hash_one(key);
        let shard_idx = self.shard_index(hash);

        let mut shard = self.shards[shard_idx].write();
        shard.remove(key, hash)
    }

    /// Checks if a key exists in the index.
    #[must_use]
    pub fn contains(&self, key: &K) -> bool {
        self.get(key).is_some()
    }

    /// Returns the total number of entries across all shards.
    #[must_use]
    pub fn len(&self) -> usize {
        self.shards.iter().map(|s| s.read().len()).sum()
    }

    /// Returns true if the index is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.shards.iter().all(|s| s.read().is_empty())
    }

    /// Clears all entries from the index.
    pub fn clear(&self) {
        for shard in &self.shards {
            *shard.write() = FingerprintBucket::new();
        }
        self.stats.reset();
    }

    /// Returns current fingerprint statistics.
    ///
    /// Use this to monitor the effectiveness of fingerprinting.
    /// A high rejection rate (>95%) indicates fingerprinting is working well.
    #[must_use]
    pub fn stats(&self) -> FingerprintStats {
        self.stats.snapshot()
    }

    /// Resets the statistics counters.
    pub fn reset_stats(&self) {
        self.stats.reset();
    }

    /// Returns the number of shards.
    #[must_use]
    pub fn shard_count(&self) -> usize {
        self.shard_count
    }
}

impl<K: Hash + Eq, V: Copy> Default for FingerprintedHashIndex<K, V> {
    fn default() -> Self {
        Self::new()
    }
}

// Implement iteration support
impl<K: Hash + Eq + Clone, V: Copy> FingerprintedHashIndex<K, V> {
    /// Iterates over all key-value pairs in the index.
    ///
    /// Note: This acquires read locks on all shards sequentially.
    /// For large indices, consider using `for_each_shard` instead.
    pub fn iter(&self) -> impl Iterator<Item = (K, V)> + '_ {
        self.shards.iter().flat_map(|shard| {
            let guard = shard.read();
            guard
                .iter()
                .map(|(k, v)| (k.clone(), *v))
                .collect::<Vec<_>>()
        })
    }

    /// Applies a function to each shard, allowing parallel processing.
    ///
    /// The function receives a read guard to the shard's bucket.
    pub fn for_each_shard<F>(&self, mut f: F)
    where
        F: FnMut(usize, &FingerprintBucket<K, V>),
    {
        for (idx, shard) in self.shards.iter().enumerate() {
            f(idx, &shard.read());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use grafeo_common::types::NodeId;

    #[test]
    fn test_basic_operations() {
        let index: FingerprintedHashIndex<String, NodeId> = FingerprintedHashIndex::new();

        // Insert
        assert!(index.insert("alice".to_string(), NodeId::new(1)).is_none());
        assert!(index.insert("bob".to_string(), NodeId::new(2)).is_none());

        // Get
        assert_eq!(index.get(&"alice".to_string()), Some(NodeId::new(1)));
        assert_eq!(index.get(&"bob".to_string()), Some(NodeId::new(2)));
        assert_eq!(index.get(&"charlie".to_string()), None);

        // Length
        assert_eq!(index.len(), 2);
        assert!(!index.is_empty());
    }

    #[test]
    fn test_update() {
        let index: FingerprintedHashIndex<String, NodeId> = FingerprintedHashIndex::new();

        index.insert("alice".to_string(), NodeId::new(1));
        let old = index.insert("alice".to_string(), NodeId::new(100));

        assert_eq!(old, Some(NodeId::new(1)));
        assert_eq!(index.get(&"alice".to_string()), Some(NodeId::new(100)));
        assert_eq!(index.len(), 1);
    }

    #[test]
    fn test_remove() {
        let index: FingerprintedHashIndex<String, NodeId> = FingerprintedHashIndex::new();

        index.insert("alice".to_string(), NodeId::new(1));
        assert!(index.contains(&"alice".to_string()));

        let removed = index.remove(&"alice".to_string());
        assert_eq!(removed, Some(NodeId::new(1)));
        assert!(!index.contains(&"alice".to_string()));
    }

    #[test]
    fn test_clear() {
        let index: FingerprintedHashIndex<String, NodeId> = FingerprintedHashIndex::new();

        index.insert("alice".to_string(), NodeId::new(1));
        index.insert("bob".to_string(), NodeId::new(2));

        index.clear();

        assert!(index.is_empty());
        assert_eq!(index.get(&"alice".to_string()), None);
    }

    #[test]
    fn test_stats() {
        let index: FingerprintedHashIndex<String, NodeId> = FingerprintedHashIndex::new();

        // Insert some keys
        for i in 0..100 {
            index.insert(format!("key_{}", i), NodeId::new(i));
        }

        // Do some lookups (mix of hits and misses)
        for i in 0..50 {
            let _ = index.get(&format!("key_{}", i)); // Hit
        }
        for i in 100..150 {
            let _ = index.get(&format!("key_{}", i)); // Miss
        }

        let stats = index.stats();
        assert_eq!(stats.lookups, 100);

        // Most lookups should result in either rejection or comparison
        assert!(stats.fingerprint_rejections + stats.full_comparisons > 0);
    }

    #[test]
    fn test_shard_count() {
        let index: FingerprintedHashIndex<u64, u64> = FingerprintedHashIndex::with_shard_count(16);
        assert_eq!(index.shard_count(), 16);

        // Non-power-of-2 should round up
        let index: FingerprintedHashIndex<u64, u64> = FingerprintedHashIndex::with_shard_count(17);
        assert_eq!(index.shard_count(), 32);
    }

    #[test]
    fn test_concurrent_access() {
        use std::sync::Arc;
        use std::thread;

        let index: Arc<FingerprintedHashIndex<u64, u64>> =
            Arc::new(FingerprintedHashIndex::new());

        // Spawn multiple writers
        let mut handles = vec![];
        for t in 0..4 {
            let index = Arc::clone(&index);
            handles.push(thread::spawn(move || {
                for i in 0..1000 {
                    let key = t * 1000 + i;
                    index.insert(key, key * 2);
                }
            }));
        }

        // Spawn multiple readers
        for _ in 0..4 {
            let index = Arc::clone(&index);
            handles.push(thread::spawn(move || {
                for _ in 0..1000 {
                    let _ = index.get(&42);
                    let _ = index.get(&9999);
                }
            }));
        }

        for handle in handles {
            handle.join().unwrap();
        }

        // All writes should have succeeded
        assert_eq!(index.len(), 4000);
    }

    #[test]
    fn test_iteration() {
        let index: FingerprintedHashIndex<u64, u64> = FingerprintedHashIndex::new();

        for i in 0..10 {
            index.insert(i, i * 10);
        }

        let items: Vec<_> = index.iter().collect();
        assert_eq!(items.len(), 10);

        // Check all values are present (order not guaranteed)
        for i in 0..10 {
            assert!(items.iter().any(|(k, v)| *k == i && *v == i * 10));
        }
    }

    #[test]
    fn test_fingerprint_effectiveness() {
        // Insert keys that would hash to the same shard
        let index: FingerprintedHashIndex<String, u64> =
            FingerprintedHashIndex::with_shard_count(1); // Force all to same shard

        for i in 0..100 {
            index.insert(format!("key_{}", i), i);
        }

        // Reset stats and do lookups for non-existent keys
        index.reset_stats();

        for i in 100..200 {
            let _ = index.get(&format!("key_{}", i));
        }

        let stats = index.stats();

        // With 100 entries and 100 misses, we should see many rejections
        // Each miss iterates through entries until no match
        assert!(stats.fingerprint_rejections > 0, "Expected some fingerprint rejections");

        // Rejection rate should be high (most entries rejected by fingerprint)
        if stats.fingerprint_rejections + stats.full_comparisons > 0 {
            let rate = stats.rejection_rate();
            // We expect high rejection rate since fingerprints are diverse
            assert!(
                rate > 0.5,
                "Expected rejection rate > 50%, got {:.2}%",
                rate * 100.0
            );
        }
    }
}
