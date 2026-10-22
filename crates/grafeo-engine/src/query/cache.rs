//! Query cache for parsed and planned queries.
//!
//! This module provides an LRU cache for query plans to avoid repeated
//! parsing and optimization of frequently executed queries.
//!
//! ## Cache Levels
//!
//! - **Parsed cache**: Caches logical plans after translation (language-specific parsing)
//! - **Optimized cache**: Caches logical plans after optimization
//!
//! ## Usage
//!
//! ```no_run
//! use grafeo_engine::query::cache::{QueryCache, CacheKey};
//! use grafeo_engine::query::processor::QueryLanguage;
//! use grafeo_engine::query::plan::{LogicalPlan, LogicalOperator};
//!
//! let cache = QueryCache::new(1000);
//! let cache_key = CacheKey::new("MATCH (n) RETURN n", QueryLanguage::Gql);
//!
//! // Check cache first
//! if let Some(plan) = cache.get_optimized(&cache_key) {
//!     // use cached plan
//! }
//!
//! // Parse and optimize, then cache
//! let plan = LogicalPlan::new(LogicalOperator::Empty);
//! cache.put_optimized(cache_key, plan);
//! ```

use parking_lot::Mutex;
use std::collections::HashMap;
use std::hash::Hash;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

use crate::query::plan::LogicalPlan;
use crate::query::processor::QueryLanguage;

/// Cache key combining query text, language, and active graph.
#[derive(Clone, Eq, PartialEq, Hash)]
pub struct CacheKey {
    /// The query string (normalized).
    query: String,
    /// The query language.
    language: QueryLanguage,
    /// Active graph name (`None` = default graph).
    graph: Option<String>,
}

impl CacheKey {
    /// Creates a new cache key for the default graph.
    #[must_use]
    pub fn new(query: impl Into<String>, language: QueryLanguage) -> Self {
        Self {
            query: normalize_query(&query.into()),
            language,
            graph: None,
        }
    }

    /// Creates a cache key scoped to a specific graph.
    #[must_use]
    pub fn with_graph(
        query: impl Into<String>,
        language: QueryLanguage,
        graph: Option<String>,
    ) -> Self {
        Self {
            query: normalize_query(&query.into()),
            language,
            graph,
        }
    }

    /// Returns the query string.
    #[must_use]
    pub fn query(&self) -> &str {
        &self.query
    }

    /// Returns the query language.
    #[must_use]
    pub fn language(&self) -> QueryLanguage {
        self.language
    }
}

/// Normalizes a query string for caching.
///
/// Removes extra whitespace and normalizes case for keywords.
fn normalize_query(query: &str) -> String {
    // Simple normalization: collapse whitespace
    query.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// Entry in the cache with metadata.
struct CacheEntry<T> {
    /// The cached value.
    value: T,
    /// Number of times this entry was accessed.
    access_count: u64,
    /// Last access time (not available on WASM).
    #[cfg(not(target_arch = "wasm32"))]
    last_accessed: Instant,
}

impl<T: Clone> CacheEntry<T> {
    fn new(value: T) -> Self {
        Self {
            value,
            access_count: 0,
            #[cfg(not(target_arch = "wasm32"))]
            last_accessed: Instant::now(),
        }
    }

    fn access(&mut self) -> T {
        self.access_count += 1;
        #[cfg(not(target_arch = "wasm32"))]
        {
            self.last_accessed = Instant::now();
        }
        self.value.clone()
    }
}

/// LRU cache implementation.
struct LruCache<K, V> {
    /// The cache storage.
    entries: HashMap<K, CacheEntry<V>>,
    /// Maximum number of entries.
    capacity: usize,
    /// Order of access (for LRU eviction).
    access_order: Vec<K>,
}

impl<K: Clone + Eq + Hash, V: Clone> LruCache<K, V> {
    fn new(capacity: usize) -> Self {
        Self {
            entries: HashMap::with_capacity(capacity),
            capacity,
            access_order: Vec::with_capacity(capacity),
        }
    }

    fn get(&mut self, key: &K) -> Option<V> {
        if let Some(entry) = self.entries.get_mut(key) {
            // Move to end of access order (most recently used)
            if let Some(pos) = self.access_order.iter().position(|k| k == key) {
                self.access_order.remove(pos);
            }
            self.access_order.push(key.clone());
            Some(entry.access())
        } else {
            None
        }
    }

    fn put(&mut self, key: K, value: V) {
        // Evict if at capacity
        if self.entries.len() >= self.capacity && !self.entries.contains_key(&key) {
            self.evict_lru();
        }

        // Remove from current position in access order
        if let Some(pos) = self.access_order.iter().position(|k| k == &key) {
            self.access_order.remove(pos);
        }

        // Add to end (most recently used)
        self.access_order.push(key.clone());
        self.entries.insert(key, CacheEntry::new(value));
    }

    fn evict_lru(&mut self) {
        if let Some(key) = self.access_order.first().cloned() {
            self.access_order.remove(0);
            self.entries.remove(&key);
        }
    }

    fn clear(&mut self) {
        self.entries.clear();
        self.access_order.clear();
    }

    fn len(&self) -> usize {
        self.entries.len()
    }

    fn remove(&mut self, key: &K) -> Option<V> {
        if let Some(pos) = self.access_order.iter().position(|k| k == key) {
            self.access_order.remove(pos);
        }
        self.entries.remove(key).map(|e| e.value)
    }

    /// Estimates heap memory used by this cache (map buckets + access order vec).
    fn heap_memory_bytes(&self) -> usize {
        let entry_size = std::mem::size_of::<K>() + std::mem::size_of::<CacheEntry<V>>() + 1;
        let map_bytes = self.entries.capacity() * entry_size;
        let vec_bytes = self.access_order.capacity() * std::mem::size_of::<K>();
        map_bytes + vec_bytes
    }
}

/// Query cache for parsed and optimized plans.
pub struct QueryCache {
    /// Cache for parsed (translated) logical plans.
    parsed_cache: Mutex<LruCache<CacheKey, LogicalPlan>>,
    /// Cache for optimized logical plans.
    optimized_cache: Mutex<LruCache<CacheKey, LogicalPlan>>,
    /// Cache hit count for parsed plans.
    parsed_hits: AtomicU64,
    /// Cache miss count for parsed plans.
    parsed_misses: AtomicU64,
    /// Cache hit count for optimized plans.
    optimized_hits: AtomicU64,
    /// Cache miss count for optimized plans.
    optimized_misses: AtomicU64,
    /// Number of times the cache was invalidated (cleared due to DDL).
    invalidations: AtomicU64,
    /// Whether caching is enabled.
    enabled: bool,
}

impl QueryCache {
    /// Creates a new query cache with the specified capacity.
    ///
    /// The capacity is shared between parsed and optimized caches
    /// (each gets half the capacity).
    #[must_use]
    pub fn new(capacity: usize) -> Self {
        let half_capacity = capacity / 2;
        Self {
            parsed_cache: Mutex::new(LruCache::new(half_capacity.max(1))),
            optimized_cache: Mutex::new(LruCache::new(half_capacity.max(1))),
            parsed_hits: AtomicU64::new(0),
            parsed_misses: AtomicU64::new(0),
            optimized_hits: AtomicU64::new(0),
            optimized_misses: AtomicU64::new(0),
            invalidations: AtomicU64::new(0),
            enabled: true,
        }
    }

    /// Creates a disabled cache (for testing or when caching is not desired).
    #[must_use]
    pub fn disabled() -> Self {
        Self {
            parsed_cache: Mutex::new(LruCache::new(0)),
            optimized_cache: Mutex::new(LruCache::new(0)),
            parsed_hits: AtomicU64::new(0),
            parsed_misses: AtomicU64::new(0),
            optimized_hits: AtomicU64::new(0),
            optimized_misses: AtomicU64::new(0),
            invalidations: AtomicU64::new(0),
            enabled: false,
        }
    }

    /// Returns whether caching is enabled.
    #[must_use]
    pub fn is_enabled(&self) -> bool {
        self.enabled
    }

    /// Gets a parsed plan from the cache.
    pub fn get_parsed(&self, key: &CacheKey) -> Option<LogicalPlan> {
        if !self.enabled {
            return None;
        }

        let result = self.parsed_cache.lock().get(key);
        if result.is_some() {
            self.parsed_hits.fetch_add(1, Ordering::Relaxed);
        } else {
            self.parsed_misses.fetch_add(1, Ordering::Relaxed);
        }
        result
    }

    /// Puts a parsed plan into the cache.
    pub fn put_parsed(&self, key: CacheKey, plan: LogicalPlan) {
        if !self.enabled {
            return;
        }
        self.parsed_cache.lock().put(key, plan);
    }

    /// Gets an optimized plan from the cache.
    pub fn get_optimized(&self, key: &CacheKey) -> Option<LogicalPlan> {
        if !self.enabled {
            return None;
        }

        let result = self.optimized_cache.lock().get(key);
        if result.is_some() {
            self.optimized_hits.fetch_add(1, Ordering::Relaxed);
        } else {
            self.optimized_misses.fetch_add(1, Ordering::Relaxed);
        }
        result
    }

    /// Puts an optimized plan into the cache.
    pub fn put_optimized(&self, key: CacheKey, plan: LogicalPlan) {
        if !self.enabled {
            return;
        }
        self.optimized_cache.lock().put(key, plan);
    }

    /// Invalidates a specific query from both caches.
    pub fn invalidate(&self, key: &CacheKey) {
        self.parsed_cache.lock().remove(key);
        self.optimized_cache.lock().remove(key);
    }

    /// Clears all cached entries and increments the invalidation counter
    /// (only when the cache was non-empty).
    pub fn clear(&self) {
        let had_entries =
            self.parsed_cache.lock().len() > 0 || self.optimized_cache.lock().len() > 0;
        self.parsed_cache.lock().clear();
        self.optimized_cache.lock().clear();
        if had_entries {
            self.invalidations.fetch_add(1, Ordering::Relaxed);
        }
    }

    /// Returns cache statistics.
    #[must_use]
    pub fn stats(&self) -> CacheStats {
        CacheStats {
            parsed_size: self.parsed_cache.lock().len(),
            optimized_size: self.optimized_cache.lock().len(),
            parsed_hits: self.parsed_hits.load(Ordering::Relaxed),
            parsed_misses: self.parsed_misses.load(Ordering::Relaxed),
            optimized_hits: self.optimized_hits.load(Ordering::Relaxed),
            optimized_misses: self.optimized_misses.load(Ordering::Relaxed),
            invalidations: self.invalidations.load(Ordering::Relaxed),
        }
    }

    /// Estimates heap memory used by both caches.
    #[must_use]
    pub fn heap_memory_bytes(&self) -> (usize, usize, usize) {
        let parsed = self.parsed_cache.lock();
        let optimized = self.optimized_cache.lock();
        let parsed_bytes = parsed.heap_memory_bytes();
        let optimized_bytes = optimized.heap_memory_bytes();
        let count = parsed.len() + optimized.len();
        (parsed_bytes, optimized_bytes, count)
    }

    /// Resets hit/miss counters and invalidation counter.
    pub fn reset_stats(&self) {
        self.parsed_hits.store(0, Ordering::Relaxed);
        self.parsed_misses.store(0, Ordering::Relaxed);
        self.optimized_hits.store(0, Ordering::Relaxed);
        self.optimized_misses.store(0, Ordering::Relaxed);
        self.invalidations.store(0, Ordering::Relaxed);
    }
}

impl Default for QueryCache {
    fn default() -> Self {
        // Default capacity of 1000 queries
        Self::new(1000)
    }
}

/// Cache statistics.
#[derive(Debug, Clone)]
pub struct CacheStats {
    /// Number of entries in parsed cache.
    pub parsed_size: usize,
    /// Number of entries in optimized cache.
    pub optimized_size: usize,
    /// Number of parsed cache hits.
    pub parsed_hits: u64,
    /// Number of parsed cache misses.
    pub parsed_misses: u64,
    /// Number of optimized cache hits.
    pub optimized_hits: u64,
    /// Number of optimized cache misses.
    pub optimized_misses: u64,
    /// Number of times the cache was invalidated (cleared due to DDL).
    pub invalidations: u64,
}

impl CacheStats {
    /// Returns the hit rate for parsed cache (0.0 to 1.0).
    #[must_use]
    pub fn parsed_hit_rate(&self) -> f64 {
        let total = self.parsed_hits + self.parsed_misses;
        if total == 0 {
            0.0
        } else {
            self.parsed_hits as f64 / total as f64
        }
    }

    /// Returns the hit rate for optimized cache (0.0 to 1.0).
    #[must_use]
    pub fn optimized_hit_rate(&self) -> f64 {
        let total = self.optimized_hits + self.optimized_misses;
        if total == 0 {
            0.0
        } else {
            self.optimized_hits as f64 / total as f64
        }
    }

    /// Returns the total cache size.
    #[must_use]
    pub fn total_size(&self) -> usize {
        self.parsed_size + self.optimized_size
    }

    /// Returns the total hit rate.
    #[must_use]
    pub fn total_hit_rate(&self) -> f64 {
        let total_hits = self.parsed_hits + self.optimized_hits;
        let total_misses = self.parsed_misses + self.optimized_misses;
        let total = total_hits + total_misses;
        if total == 0 {
            0.0
        } else {
            total_hits as f64 / total as f64
        }
    }
}

/// A caching wrapper for the query processor.
///
/// This type wraps a query processor and adds caching capabilities.
/// Use this for production deployments where query caching is beneficial.
pub struct CachingQueryProcessor<P> {
    /// The underlying processor.
    processor: P,
    /// The query cache.
    cache: QueryCache,
}

impl<P> CachingQueryProcessor<P> {
    /// Creates a new caching processor.
    pub fn new(processor: P, cache: QueryCache) -> Self {
        Self { processor, cache }
    }

    /// Creates a new caching processor with default cache settings.
    pub fn with_default_cache(processor: P) -> Self {
        Self::new(processor, QueryCache::default())
    }

    /// Returns a reference to the cache.
    #[must_use]
    pub fn cache(&self) -> &QueryCache {
        &self.cache
    }

    /// Returns a reference to the underlying processor.
    #[must_use]
    pub fn processor(&self) -> &P {
        &self.processor
    }

    /// Returns cache statistics.
    #[must_use]
    pub fn stats(&self) -> CacheStats {
        self.cache.stats()
    }

    /// Clears the cache.
    pub fn clear_cache(&self) {
        self.cache.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(feature = "gql")]
    fn test_language() -> QueryLanguage {
        QueryLanguage::Gql
    }

    #[cfg(not(feature = "gql"))]
    fn test_language() -> QueryLanguage {
        // Fallback for tests without gql feature
        #[cfg(feature = "cypher")]
        return QueryLanguage::Cypher;
        #[cfg(feature = "sparql")]
        return QueryLanguage::Sparql;
    }

    #[test]
    fn test_cache_key_normalization() {
        let key1 = CacheKey::new("MATCH  (n)  RETURN n", test_language());
        let key2 = CacheKey::new("MATCH (n) RETURN n", test_language());

        // Both should normalize to the same key
        assert_eq!(key1.query(), key2.query());
    }

    #[test]
    fn test_cache_basic_operations() {
        let cache = QueryCache::new(10);
        let key = CacheKey::new("MATCH (n) RETURN n", test_language());

        // Create a simple logical plan for testing
        use crate::query::plan::{LogicalOperator, LogicalPlan};
        let plan = LogicalPlan::new(LogicalOperator::Empty);

        // Initially empty
        assert!(cache.get_parsed(&key).is_none());

        // Put and get
        cache.put_parsed(key.clone(), plan.clone());
        assert!(cache.get_parsed(&key).is_some());

        // Stats
        let stats = cache.stats();
        assert_eq!(stats.parsed_size, 1);
        assert_eq!(stats.parsed_hits, 1);
        assert_eq!(stats.parsed_misses, 1);
    }

    #[test]
    fn test_cache_lru_eviction() {
        let cache = QueryCache::new(4); // 2 entries per cache level

        use crate::query::plan::{LogicalOperator, LogicalPlan};

        // Add 3 entries to parsed cache (capacity is 2)
        for i in 0..3 {
            let key = CacheKey::new(format!("QUERY {}", i), test_language());
            cache.put_parsed(key, LogicalPlan::new(LogicalOperator::Empty));
        }

        // First entry should be evicted
        let key0 = CacheKey::new("QUERY 0", test_language());
        assert!(cache.get_parsed(&key0).is_none());

        // Entry 1 and 2 should still be present
        let key1 = CacheKey::new("QUERY 1", test_language());
        let key2 = CacheKey::new("QUERY 2", test_language());
        assert!(cache.get_parsed(&key1).is_some());
        assert!(cache.get_parsed(&key2).is_some());
    }

    #[test]
    fn test_cache_invalidation() {
        let cache = QueryCache::new(10);
        let key = CacheKey::new("MATCH (n) RETURN n", test_language());

        use crate::query::plan::{LogicalOperator, LogicalPlan};
        let plan = LogicalPlan::new(LogicalOperator::Empty);

        cache.put_parsed(key.clone(), plan.clone());
        cache.put_optimized(key.clone(), plan);

        assert!(cache.get_parsed(&key).is_some());
        assert!(cache.get_optimized(&key).is_some());

        // Invalidate
        cache.invalidate(&key);

        // Clear stats from previous gets
        cache.reset_stats();

        assert!(cache.get_parsed(&key).is_none());
        assert!(cache.get_optimized(&key).is_none());
    }

    #[test]
    fn test_cache_disabled() {
        let cache = QueryCache::disabled();
        let key = CacheKey::new("MATCH (n) RETURN n", test_language());

        use crate::query::plan::{LogicalOperator, LogicalPlan};
        let plan = LogicalPlan::new(LogicalOperator::Empty);

        // Should not store anything
        cache.put_parsed(key.clone(), plan);
        assert!(cache.get_parsed(&key).is_none());

        // Stats should be zero
        let stats = cache.stats();
        assert_eq!(stats.parsed_size, 0);
    }

    #[test]
    fn test_cache_stats() {
        let cache = QueryCache::new(10);

        use crate::query::plan::{LogicalOperator, LogicalPlan};

        let key1 = CacheKey::new("QUERY 1", test_language());
        let key2 = CacheKey::new("QUERY 2", test_language());
        let plan = LogicalPlan::new(LogicalOperator::Empty);

        // Miss
        cache.get_optimized(&key1);

        // Put and hit
        cache.put_optimized(key1.clone(), plan);
        cache.get_optimized(&key1);
        cache.get_optimized(&key1);

        // Another miss
        cache.get_optimized(&key2);

        let stats = cache.stats();
        assert_eq!(stats.optimized_hits, 2);
        assert_eq!(stats.optimized_misses, 2);
        assert!((stats.optimized_hit_rate() - 0.5).abs() < 0.01);
    }
}
