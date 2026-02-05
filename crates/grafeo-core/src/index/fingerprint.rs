//! Fingerprinting utilities for fast hash index lookups.
//!
//! Fingerprints are compact hash-derived values used to quickly reject
//! non-matching entries without expensive full key comparisons.
//!
//! # How It Works
//!
//! When storing entries in a hash bucket:
//! 1. Compute the full hash of the key
//! 2. Use lower bits for bucket selection
//! 3. Use upper bits as the "fingerprint"
//!
//! On lookup:
//! 1. Compute hash, select bucket
//! 2. For each entry, compare fingerprints first (cheap)
//! 3. Only if fingerprint matches, compare full keys (expensive)
//!
//! This gives ~99.99% rejection rate for non-matches with 48-bit fingerprints,
//! significantly reducing disk I/O and CPU time for hash lookups.

use std::hash::Hash;

use grafeo_common::utils::hash::hash_one;

/// Compute fingerprint from hash (use upper bits, lower bits for bucket).
///
/// The fingerprint uses the upper 48 bits of the hash, giving a false
/// positive rate of approximately 1/2^48 (~0.000000000003%).
#[inline]
#[must_use]
pub fn fingerprint(hash: u64) -> u64 {
    hash >> 16
}

/// Compute fingerprint directly from a hashable key.
#[inline]
#[must_use]
pub fn fingerprint_of<K: Hash>(key: &K) -> u64 {
    fingerprint(hash_one(key))
}

/// Hash entry with fingerprint for fast rejection.
///
/// Stores a fingerprint alongside the value, enabling quick miss detection
/// without comparing full keys.
#[derive(Debug, Clone, Copy)]
pub struct FingerprintEntry<V> {
    /// 8-byte fingerprint from hash upper bits.
    pub fingerprint: u64,
    /// The actual value.
    pub value: V,
}

impl<V> FingerprintEntry<V> {
    /// Creates a new entry with fingerprint computed from the hash.
    #[inline]
    pub fn new(hash: u64, value: V) -> Self {
        Self {
            fingerprint: fingerprint(hash),
            value,
        }
    }

    /// Quick check before full comparison.
    ///
    /// Returns `true` if the fingerprints match, meaning a full key
    /// comparison is needed. Returns `false` if this definitely isn't
    /// a match.
    #[inline]
    #[must_use]
    pub fn matches_fingerprint(&self, hash: u64) -> bool {
        self.fingerprint == fingerprint(hash)
    }
}

/// A bucket with fingerprint-based filtering for hash collisions.
///
/// When multiple keys hash to the same bucket, fingerprints help
/// quickly identify which entry (if any) matches without comparing
/// full keys.
///
/// # Example
///
/// ```
/// use grafeo_core::index::fingerprint::FingerprintBucket;
/// use grafeo_common::utils::hash::hash_one;
///
/// let mut bucket = FingerprintBucket::new();
///
/// let key1 = "alice";
/// let hash1 = hash_one(&key1);
/// bucket.insert(key1, hash1, 1);
///
/// let key2 = "bob";
/// let hash2 = hash_one(&key2);
/// bucket.insert(key2, hash2, 2);
///
/// assert_eq!(bucket.get(&"alice", hash_one(&"alice")), Some(&1));
/// assert_eq!(bucket.get(&"charlie", hash_one(&"charlie")), None);
/// ```
#[derive(Debug, Clone)]
pub struct FingerprintBucket<K, V> {
    entries: Vec<(K, FingerprintEntry<V>)>,
}

impl<K: Eq, V> FingerprintBucket<K, V> {
    /// Creates a new empty bucket.
    #[must_use]
    pub fn new() -> Self {
        Self {
            entries: Vec::new(),
        }
    }

    /// Creates a bucket with pre-allocated capacity.
    #[must_use]
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            entries: Vec::with_capacity(capacity),
        }
    }

    /// Inserts a key-value pair with its hash.
    pub fn insert(&mut self, key: K, hash: u64, value: V) {
        // Check if key already exists
        let fp = fingerprint(hash);
        for (k, entry) in &mut self.entries {
            if entry.fingerprint == fp && k == &key {
                entry.value = value;
                return;
            }
        }
        self.entries.push((key, FingerprintEntry::new(hash, value)));
    }

    /// Gets a value, using fingerprint for fast rejection.
    ///
    /// This is the key optimization: most non-matching entries are
    /// rejected by fingerprint comparison alone, without comparing keys.
    #[must_use]
    pub fn get(&self, key: &K, hash: u64) -> Option<&V> {
        let fp = fingerprint(hash);
        for (k, entry) in &self.entries {
            // Fast path: fingerprint mismatch = definitely not a match
            if entry.fingerprint != fp {
                continue;
            }
            // Slow path: fingerprint matches, verify full key
            if k == key {
                return Some(&entry.value);
            }
        }
        None
    }

    /// Gets a mutable reference to a value.
    pub fn get_mut(&mut self, key: &K, hash: u64) -> Option<&mut V> {
        let fp = fingerprint(hash);
        for (k, entry) in &mut self.entries {
            if entry.fingerprint != fp {
                continue;
            }
            if k == key {
                return Some(&mut entry.value);
            }
        }
        None
    }

    /// Removes a key from the bucket.
    pub fn remove(&mut self, key: &K, hash: u64) -> Option<V> {
        let fp = fingerprint(hash);
        let mut idx = None;
        for (i, (k, entry)) in self.entries.iter().enumerate() {
            if entry.fingerprint == fp && k == key {
                idx = Some(i);
                break;
            }
        }
        idx.map(|i| self.entries.swap_remove(i).1.value)
    }

    /// Checks if the bucket contains the key.
    #[must_use]
    pub fn contains(&self, key: &K, hash: u64) -> bool {
        self.get(key, hash).is_some()
    }

    /// Returns the number of entries in this bucket.
    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Returns true if the bucket is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Iterates over all entries in the bucket (key and value only).
    pub fn iter(&self) -> impl Iterator<Item = (&K, &V)> {
        self.entries.iter().map(|(k, e)| (k, &e.value))
    }

    /// Iterates over all entries with their fingerprint metadata.
    ///
    /// Use this when you need access to the fingerprint for statistics tracking.
    pub fn iter_entries(&self) -> impl Iterator<Item = (&K, &FingerprintEntry<V>)> {
        self.entries.iter().map(|(k, e)| (k, e))
    }
}

impl<K: Eq, V> Default for FingerprintBucket<K, V> {
    fn default() -> Self {
        Self::new()
    }
}

/// Statistics about fingerprint effectiveness.
///
/// Use this to measure how well fingerprinting is working for your workload.
#[derive(Debug, Clone, Copy, Default)]
pub struct FingerprintStats {
    /// Total lookup operations.
    pub lookups: u64,
    /// Entries rejected by fingerprint alone.
    pub fingerprint_rejections: u64,
    /// Entries that required full key comparison.
    pub full_comparisons: u64,
}

impl FingerprintStats {
    /// Creates new empty stats.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Records a fingerprint rejection (fast path).
    pub fn record_rejection(&mut self) {
        self.fingerprint_rejections += 1;
    }

    /// Records a full comparison (slow path).
    pub fn record_comparison(&mut self) {
        self.full_comparisons += 1;
    }

    /// Records a lookup operation.
    pub fn record_lookup(&mut self) {
        self.lookups += 1;
    }

    /// Returns the rejection rate (0.0 to 1.0).
    ///
    /// Higher is better - means more entries rejected without full comparison.
    #[must_use]
    pub fn rejection_rate(&self) -> f64 {
        let total = self.fingerprint_rejections + self.full_comparisons;
        if total == 0 {
            0.0
        } else {
            self.fingerprint_rejections as f64 / total as f64
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fingerprint_basic() {
        let hash1: u64 = 0x1234_5678_9ABC_DEF0;
        let hash2: u64 = 0x1234_5678_9ABC_1234; // Same upper 48 bits (only lower 16 differ)

        // Same upper 48 bits = same fingerprint (lower 16 bits are discarded)
        assert_eq!(fingerprint(hash1), fingerprint(hash2));

        let hash3: u64 = 0xFFFF_5678_9ABC_DEF0;
        // Different upper 48 bits = different fingerprint
        assert_ne!(fingerprint(hash1), fingerprint(hash3));
    }

    #[test]
    fn test_fingerprint_entry() {
        let entry = FingerprintEntry::new(0x1234_5678_9ABC_DEF0, 42);

        // Same upper 48 bits should match (lower 16 differ)
        assert!(entry.matches_fingerprint(0x1234_5678_9ABC_0000));

        // Different upper 48 bits should not match
        assert!(!entry.matches_fingerprint(0xFFFF_5678_9ABC_DEF0));
    }

    #[test]
    fn test_bucket_basic() {
        let mut bucket: FingerprintBucket<&str, i32> = FingerprintBucket::new();

        let hash_a = hash_one(&"a");
        let hash_b = hash_one(&"b");

        bucket.insert("a", hash_a, 1);
        bucket.insert("b", hash_b, 2);

        assert_eq!(bucket.get(&"a", hash_a), Some(&1));
        assert_eq!(bucket.get(&"b", hash_b), Some(&2));
        assert_eq!(bucket.get(&"c", hash_one(&"c")), None);
    }

    #[test]
    fn test_bucket_update() {
        let mut bucket: FingerprintBucket<&str, i32> = FingerprintBucket::new();

        let hash_a = hash_one(&"a");
        bucket.insert("a", hash_a, 1);
        bucket.insert("a", hash_a, 2);

        assert_eq!(bucket.len(), 1);
        assert_eq!(bucket.get(&"a", hash_a), Some(&2));
    }

    #[test]
    fn test_bucket_remove() {
        let mut bucket: FingerprintBucket<&str, i32> = FingerprintBucket::new();

        let hash_a = hash_one(&"a");
        bucket.insert("a", hash_a, 1);

        assert!(bucket.contains(&"a", hash_a));
        let removed = bucket.remove(&"a", hash_a);
        assert_eq!(removed, Some(1));
        assert!(!bucket.contains(&"a", hash_a));
    }

    #[test]
    fn test_bucket_get_mut() {
        let mut bucket: FingerprintBucket<&str, i32> = FingerprintBucket::new();

        let hash_a = hash_one(&"a");
        bucket.insert("a", hash_a, 1);

        if let Some(val) = bucket.get_mut(&"a", hash_a) {
            *val = 10;
        }

        assert_eq!(bucket.get(&"a", hash_a), Some(&10));
    }

    #[test]
    fn test_stats() {
        let mut stats = FingerprintStats::new();

        stats.record_lookup();
        stats.record_rejection();
        stats.record_rejection();
        stats.record_comparison();

        assert_eq!(stats.lookups, 1);
        assert_eq!(stats.fingerprint_rejections, 2);
        assert_eq!(stats.full_comparisons, 1);

        // 2 rejections out of 3 total = 66.67%
        assert!((stats.rejection_rate() - 0.6667).abs() < 0.01);
    }
}
