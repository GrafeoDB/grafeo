//! Fast, non-cryptographic hashing for internal use.
//!
//! Use [`FxHashMap`] and [`FxHashSet`] instead of the standard library versions
//! for better performance. For hashes that need to be stable across runs
//! (like in the WAL), use [`stable_hash()`].

use ahash::AHasher;
use std::hash::{BuildHasher, Hasher};
use std::sync::OnceLock;

/// A fast hasher based on aHash.
///
/// This hasher is optimized for speed rather than cryptographic security.
/// It's suitable for hash tables, bloom filters, and other internal uses.
pub type FxHasher = AHasher;

/// A fast hash builder using aHash.
pub type FxBuildHasher = ahash::RandomState;

/// A `HashMap` using fast hashing.
pub type FxHashMap<K, V> = hashbrown::HashMap<K, V, FxBuildHasher>;

/// A `HashSet` using fast hashing.
pub type FxHashSet<T> = hashbrown::HashSet<T, FxBuildHasher>;

/// Static `RandomState` used for consistent hashing within a program run.
static HASH_STATE: OnceLock<ahash::RandomState> = OnceLock::new();

fn get_hash_state() -> &'static ahash::RandomState {
    HASH_STATE.get_or_init(ahash::RandomState::new)
}

/// Computes a 64-bit hash of the given value.
///
/// The hash is consistent within a single program run but may vary between runs.
#[inline]
#[must_use]
pub fn hash_one<T: std::hash::Hash>(value: &T) -> u64 {
    get_hash_state().hash_one(value)
}

/// Computes a stable hash of the given bytes.
///
/// Unlike `hash_one`, this function produces the same hash for the same
/// input across different runs of the program. Use this for persistent
/// hashing (e.g., in WAL entries).
#[inline]
#[must_use]
pub fn stable_hash(bytes: &[u8]) -> u64 {
    // Use a simple, stable hash function
    let mut hash: u64 = 0xcbf29ce4_84222325; // FNV offset basis
    for &byte in bytes {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(0x100000001b3); // FNV prime
    }
    hash
}

/// Combines two hash values.
///
/// This is useful for combining hashes of multiple fields.
#[inline]
#[must_use]
pub const fn combine_hashes(h1: u64, h2: u64) -> u64 {
    // Based on boost::hash_combine
    h1 ^ (h2
        .wrapping_add(0x9e3779b9)
        .wrapping_add(h1 << 6)
        .wrapping_add(h1 >> 2))
}

/// A hash builder that produces consistent hashes.
///
/// Use this when you need deterministic hashing (e.g., for testing or
/// persistent storage).
#[derive(Clone, Default)]
pub struct StableHashBuilder;

impl BuildHasher for StableHashBuilder {
    type Hasher = StableHasher;

    fn build_hasher(&self) -> Self::Hasher {
        StableHasher::new()
    }
}

/// A stable hasher using FNV-1a.
pub struct StableHasher {
    state: u64,
}

impl StableHasher {
    /// Creates a new stable hasher.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            state: 0xcbf29ce4_84222325, // FNV offset basis
        }
    }
}

impl Default for StableHasher {
    fn default() -> Self {
        Self::new()
    }
}

impl Hasher for StableHasher {
    #[inline]
    fn write(&mut self, bytes: &[u8]) {
        for &byte in bytes {
            self.state ^= u64::from(byte);
            self.state = self.state.wrapping_mul(0x100000001b3);
        }
    }

    #[inline]
    fn finish(&self) -> u64 {
        self.state
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hash_one() {
        let h1 = hash_one(&42u64);
        let h2 = hash_one(&42u64);
        let h3 = hash_one(&43u64);

        // Same value should produce same hash (within the same run)
        assert_eq!(h1, h2);
        // Different values should (very likely) produce different hashes
        assert_ne!(h1, h3);
    }

    #[test]
    fn test_stable_hash() {
        let h1 = stable_hash(b"hello");
        let h2 = stable_hash(b"hello");
        let h3 = stable_hash(b"world");

        assert_eq!(h1, h2);
        assert_ne!(h1, h3);

        // These values should be consistent across runs
        // (hard-coded expected values for verification)
        assert_eq!(stable_hash(b""), 0xcbf29ce4_84222325);
    }

    #[test]
    fn test_combine_hashes() {
        let h1 = 123456u64;
        let h2 = 789012u64;

        let combined = combine_hashes(h1, h2);

        // Combining should be deterministic
        assert_eq!(combined, combine_hashes(h1, h2));

        // Order matters
        assert_ne!(combine_hashes(h1, h2), combine_hashes(h2, h1));
    }

    #[test]
    fn test_stable_hasher() {
        let mut hasher = StableHasher::new();
        hasher.write(b"hello");
        let h1 = hasher.finish();

        let mut hasher = StableHasher::new();
        hasher.write(b"hello");
        let h2 = hasher.finish();

        assert_eq!(h1, h2);
    }

    #[test]
    fn test_fx_hashmap() {
        let mut map: FxHashMap<u64, String> = FxHashMap::default();
        map.insert(1, "one".to_string());
        map.insert(2, "two".to_string());

        assert_eq!(map.get(&1), Some(&"one".to_string()));
        assert_eq!(map.get(&2), Some(&"two".to_string()));
        assert_eq!(map.get(&3), None);
    }

    #[test]
    fn test_fx_hashset() {
        let mut set: FxHashSet<u64> = FxHashSet::default();
        set.insert(1);
        set.insert(2);

        assert!(set.contains(&1));
        assert!(set.contains(&2));
        assert!(!set.contains(&3));
    }
}
