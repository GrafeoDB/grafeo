//! Standard collection type aliases for Grafeo.
//!
//! Use these instead of direct HashMap/HashSet to allow future optimization
//! and ensure consistent hashing across the codebase.
//!
//! # Type Aliases
//!
//! | Type | Use Case |
//! |------|----------|
//! | [`GrafeoMap`] | Single-threaded hash map |
//! | [`GrafeoSet`] | Single-threaded hash set |
//! | [`GrafeoConcurrentMap`] | Multi-threaded hash map |
//! | [`GrafeoConcurrentSet`] | Multi-threaded hash set |
//! | [`GrafeoIndexMap`] | Insertion-order preserving map |
//! | [`GrafeoIndexSet`] | Insertion-order preserving set |
//!
//! # Example
//!
//! ```rust
//! use grafeo_common::collections::{GrafeoMap, GrafeoSet};
//!
//! let mut map: GrafeoMap<String, i32> = GrafeoMap::default();
//! map.insert("key".to_string(), 42);
//!
//! let mut set: GrafeoSet<i32> = GrafeoSet::default();
//! set.insert(1);
//! ```

use rustc_hash::FxBuildHasher;

/// Standard HashMap with FxHash (fast, non-cryptographic).
///
/// FxHash is optimized for small keys and provides excellent performance
/// for integer and string keys common in graph databases.
pub type GrafeoMap<K, V> = hashbrown::HashMap<K, V, FxBuildHasher>;

/// Standard HashSet with FxHash.
pub type GrafeoSet<T> = hashbrown::HashSet<T, FxBuildHasher>;

/// Concurrent HashMap for multi-threaded access.
///
/// Uses fine-grained locking for high concurrent throughput.
/// Prefer this over `Arc<Mutex<HashMap>>` for shared mutable state.
pub type GrafeoConcurrentMap<K, V> = dashmap::DashMap<K, V, FxBuildHasher>;

/// Concurrent HashSet for multi-threaded access.
pub type GrafeoConcurrentSet<T> = dashmap::DashSet<T, FxBuildHasher>;

/// Ordered map preserving insertion order.
///
/// Useful when iteration order matters (e.g., property serialization).
pub type GrafeoIndexMap<K, V> = indexmap::IndexMap<K, V, FxBuildHasher>;

/// Ordered set preserving insertion order.
pub type GrafeoIndexSet<T> = indexmap::IndexSet<T, FxBuildHasher>;

/// Create a new empty [`GrafeoMap`].
#[inline]
#[must_use]
pub fn grafeo_map<K, V>() -> GrafeoMap<K, V> {
    GrafeoMap::with_hasher(FxBuildHasher)
}

/// Create a new [`GrafeoMap`] with the specified capacity.
#[inline]
#[must_use]
pub fn grafeo_map_with_capacity<K, V>(capacity: usize) -> GrafeoMap<K, V> {
    GrafeoMap::with_capacity_and_hasher(capacity, FxBuildHasher)
}

/// Create a new empty [`GrafeoSet`].
#[inline]
#[must_use]
pub fn grafeo_set<T>() -> GrafeoSet<T> {
    GrafeoSet::with_hasher(FxBuildHasher)
}

/// Create a new [`GrafeoSet`] with the specified capacity.
#[inline]
#[must_use]
pub fn grafeo_set_with_capacity<T>(capacity: usize) -> GrafeoSet<T> {
    GrafeoSet::with_capacity_and_hasher(capacity, FxBuildHasher)
}

/// Create a new empty [`GrafeoConcurrentMap`].
#[inline]
#[must_use]
pub fn grafeo_concurrent_map<K, V>() -> GrafeoConcurrentMap<K, V>
where
    K: Eq + std::hash::Hash,
{
    GrafeoConcurrentMap::with_hasher(FxBuildHasher)
}

/// Create a new [`GrafeoConcurrentMap`] with the specified capacity.
#[inline]
#[must_use]
pub fn grafeo_concurrent_map_with_capacity<K, V>(capacity: usize) -> GrafeoConcurrentMap<K, V>
where
    K: Eq + std::hash::Hash,
{
    GrafeoConcurrentMap::with_capacity_and_hasher(capacity, FxBuildHasher)
}

/// Create a new empty [`GrafeoIndexMap`].
#[inline]
#[must_use]
pub fn grafeo_index_map<K, V>() -> GrafeoIndexMap<K, V> {
    GrafeoIndexMap::with_hasher(FxBuildHasher)
}

/// Create a new [`GrafeoIndexMap`] with the specified capacity.
#[inline]
#[must_use]
pub fn grafeo_index_map_with_capacity<K, V>(capacity: usize) -> GrafeoIndexMap<K, V> {
    GrafeoIndexMap::with_capacity_and_hasher(capacity, FxBuildHasher)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_grafeo_map() {
        let mut map = grafeo_map::<String, i32>();
        map.insert("key".to_string(), 42);
        assert_eq!(map.get("key"), Some(&42));
    }

    #[test]
    fn test_grafeo_set() {
        let mut set = grafeo_set::<i32>();
        set.insert(1);
        set.insert(2);
        assert!(set.contains(&1));
        assert!(!set.contains(&3));
    }

    #[test]
    fn test_grafeo_concurrent_map() {
        let map = grafeo_concurrent_map::<String, i32>();
        map.insert("key".to_string(), 42);
        assert_eq!(*map.get("key").unwrap(), 42);
    }

    #[test]
    fn test_grafeo_index_map_preserves_order() {
        let mut map = grafeo_index_map::<&str, i32>();
        map.insert("c", 3);
        map.insert("a", 1);
        map.insert("b", 2);

        let keys: Vec<_> = map.keys().copied().collect();
        assert_eq!(keys, vec!["c", "a", "b"]);
    }
}
