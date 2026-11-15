//! The plan cache: parsed and optimized plans, shared by a database's
//! sessions, so a repeated statement skips translation and optimization.
//!
//! - **One bound:** both stages share one byte bound and one recency order
//!   ([`QueryCache::with_limit`]); a plan larger than the whole bound runs
//!   and is not stored.
//! - **Shared plans:** a hit hands out the cached plan as an `Arc`
//!   ([`CachedPlan`]), never a copy.
//! - **One generation:** the database's planning generation moves at a
//!   committed change of the schema, a restore, `clear_plan_cache` and a
//!   large move of the counts the planner reads. The cache empties once it
//!   sees a newer one, and a plan planned under an older one is not stored,
//!   so a plan made before such a change never outlives it.
//!
//! ## Usage
//!
//! ```no_run
//! use grafeo_engine::query::cache::{CacheKey, QueryCache, Stage};
//! use grafeo_engine::query::processor::QueryLanguage;
//! use grafeo_engine::query::plan::{LogicalOperator, LogicalPlan};
//!
//! let cache = QueryCache::with_limit(4 << 20);
//! let key = CacheKey::new("MATCH (n) RETURN n", QueryLanguage::Gql);
//!
//! let plan = match cache.get(&key, Stage::Optimized) {
//!     Some(cached) => cached,
//!     None => {
//!         // Read the generation before planning, so a change meanwhile
//!         // keeps the plan out of the cache.
//!         let generation = cache.generation();
//!         let plan = LogicalPlan::new(LogicalOperator::Empty);
//!         cache.put(key, Stage::Optimized, plan, generation)
//!     }
//! };
//! # let _ = plan.plan();
//! ```

use parking_lot::Mutex;
use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use crate::query::plan::LogicalPlan;
use crate::query::processor::QueryLanguage;

/// Cache key combining query text, language, and the session's current
/// schema and graph: what a plan is made for besides the text.
#[derive(Clone, Eq, PartialEq, Hash)]
pub struct CacheKey {
    /// The query string (normalized).
    query: String,
    /// The query language.
    language: QueryLanguage,
    /// Current schema (`None` = default schema): graph names resolve in it,
    /// and the optimizer reads the statistics of the store they name.
    schema: Option<String>,
    /// Active graph name (`None` = default graph).
    graph: Option<String>,
    /// Whether the plan keeps the statement's parameters, to be filled in on
    /// each call (a prepared plan), or holds no parameter.
    prepared: bool,
    /// The planner rules the plan is made with (see
    /// [`PlannerRules`](crate::query::optimizer::PlannerRules)).
    rules: crate::query::optimizer::PlannerRules,
}

impl CacheKey {
    /// Creates a new cache key for the default graph.
    #[must_use]
    pub fn new(query: impl Into<String>, language: QueryLanguage) -> Self {
        Self {
            query: normalize_query(&query.into()),
            language,
            schema: None,
            graph: None,
            prepared: false,
            rules: crate::query::optimizer::PlannerRules::all(),
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
            schema: None,
            graph,
            prepared: false,
            rules: crate::query::optimizer::PlannerRules::all(),
        }
    }

    /// Creates a cache key for a session whose current schema is `schema`
    /// and current graph `graph` (`None`: the defaults).
    #[must_use]
    pub fn in_scope(
        query: impl Into<String>,
        language: QueryLanguage,
        schema: Option<String>,
        graph: Option<String>,
    ) -> Self {
        Self {
            query: normalize_query(&query.into()),
            language,
            schema,
            graph,
            prepared: false,
            rules: crate::query::optimizer::PlannerRules::all(),
        }
    }

    /// The key of the same statement planned with `rules`.
    #[must_use]
    pub fn with_rules(self, rules: crate::query::optimizer::PlannerRules) -> Self {
        Self { rules, ..self }
    }

    /// The key of the same statement's prepared plan: optimized once with
    /// its parameters in, which each call fills with its values.
    #[must_use]
    pub fn prepared(&self) -> Self {
        Self {
            prepared: true,
            ..self.clone()
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

/// The text a statement is cached by: the whole text but the whitespace
/// around it.
///
/// Whitespace inside it is kept: inside a string, a quoted name or a
/// comment it is part of the statement (`RETURN 'a  b'` is not
/// `RETURN 'a b'`), and telling those apart from whitespace between tokens
/// would take each language's lexer. Two texts that differ only in the
/// spacing between tokens are planned apart, which costs a miss, never a
/// wrong answer.
fn normalize_query(query: &str) -> String {
    query.trim().to_string()
}

/// The stage a cached plan is at: as the statement's language translated it,
/// or optimized too. One text is cached at each stage apart.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Stage {
    /// Translated, not yet optimized (a plan whose parameters a call fills).
    Parsed,
    /// Optimized, ready to run.
    Optimized,
}

impl Stage {
    /// The stage's place in a key's pair of slots.
    const fn at(self) -> usize {
        match self {
            Self::Parsed => 0,
            Self::Optimized => 1,
        }
    }
}

/// A plan the cache holds, shared by every statement that hits it: a hit
/// hands out the same plan, never a copy.
#[derive(Debug)]
pub struct CachedPlan {
    plan: LogicalPlan,
    /// What the entry is charged: the plan's deep estimate, its key's text
    /// and the cache's bookkeeping.
    bytes: usize,
}

impl std::ops::Deref for CachedPlan {
    type Target = LogicalPlan;

    fn deref(&self) -> &LogicalPlan {
        &self.plan
    }
}

impl CachedPlan {
    /// The plan. A caller that changes it (a parameter filled in) clones it.
    #[must_use]
    pub fn plan(&self) -> &LogicalPlan {
        &self.plan
    }

    /// What the cache charges the entry, in bytes.
    #[must_use]
    pub fn bytes(&self) -> usize {
        self.bytes
    }
}

/// No slot: the end of the recency list.
const NONE: usize = usize::MAX;

/// One cached entry, linked in the recency order.
struct Slot<K> {
    key: K,
    stage: Stage,
    plan: Arc<CachedPlan>,
    /// The next newer slot, `NONE` for the newest.
    newer: usize,
    /// The next older slot, `NONE` for the oldest.
    older: usize,
}

/// The entries of both stages in one recency order, bounded in bytes: a map
/// from a key to its slot at each stage, into a slab of slots linked from the
/// newest to the oldest, so a hit, a put and an eviction are each O(1), and a
/// lookup borrows its key.
///
/// Generic over its key, which is a [`CacheKey`]: a build without a query
/// language has no key value (its [`QueryLanguage`] has no variant), and a
/// generic body is not linted as unreachable there.
struct Lru<K> {
    index: HashMap<K, [usize; 2]>,
    slots: Vec<Option<Slot<K>>>,
    free: Vec<usize>,
    newest: Option<usize>,
    oldest: Option<usize>,
    /// What the entries are charged together.
    held_bytes: usize,
}

impl<K> Default for Lru<K> {
    fn default() -> Self {
        Self {
            index: HashMap::new(),
            slots: Vec::new(),
            free: Vec::new(),
            newest: None,
            oldest: None,
            held_bytes: 0,
        }
    }
}

impl<K: std::hash::Hash + Eq + Clone> Lru<K> {
    fn slot(&self, at: usize) -> &Slot<K> {
        self.slots[at].as_ref().expect("a linked slot is filled")
    }

    fn slot_mut(&mut self, at: usize) -> &mut Slot<K> {
        self.slots[at].as_mut().expect("a linked slot is filled")
    }

    /// Takes the slot at `at` out of the recency list.
    fn unlink(&mut self, at: usize) {
        let (newer, older) = {
            let slot = self.slot(at);
            (slot.newer, slot.older)
        };
        match newer {
            NONE => self.newest = (older != NONE).then_some(older),
            newer => self.slot_mut(newer).older = older,
        }
        match older {
            NONE => self.oldest = (newer != NONE).then_some(newer),
            older => self.slot_mut(older).newer = newer,
        }
    }

    /// Puts the slot at `at` at the newest end.
    fn link_newest(&mut self, at: usize) {
        let previous = self.newest.unwrap_or(NONE);
        {
            let slot = self.slot_mut(at);
            slot.newer = NONE;
            slot.older = previous;
        }
        if previous != NONE {
            self.slot_mut(previous).newer = at;
        }
        self.newest = Some(at);
        if self.oldest.is_none() {
            self.oldest = Some(at);
        }
    }

    fn find(&self, key: &K, stage: Stage) -> Option<usize> {
        let at = self.index.get(key)?[stage.at()];
        (at != NONE).then_some(at)
    }

    fn get(&mut self, key: &K, stage: Stage) -> Option<Arc<CachedPlan>> {
        let at = self.find(key, stage)?;
        if self.newest != Some(at) {
            self.unlink(at);
            self.link_newest(at);
        }
        Some(Arc::clone(&self.slot(at).plan))
    }

    /// Takes the entry at `at` out, returning its plan.
    fn remove_at(&mut self, at: usize) -> Arc<CachedPlan> {
        self.unlink(at);
        let slot = self.slots[at].take().expect("a linked slot is filled");
        if let Some(pair) = self.index.get_mut(&slot.key) {
            pair[slot.stage.at()] = NONE;
            if *pair == [NONE; 2] {
                self.index.remove(&slot.key);
            }
        }
        self.free.push(at);
        self.held_bytes -= slot.plan.bytes;
        slot.plan
    }

    fn remove(&mut self, key: &K, stage: Stage) -> Option<Arc<CachedPlan>> {
        let at = self.find(key, stage)?;
        Some(self.remove_at(at))
    }

    /// Stores `plan` under `key` as the newest entry, evicting the oldest
    /// until it fits in `limit`. The caller checked that it fits alone.
    fn put(&mut self, key: K, stage: Stage, plan: Arc<CachedPlan>, limit: usize) {
        // An entry the key held goes first: its bytes do not count twice.
        self.remove(&key, stage);
        while self.held_bytes + plan.bytes > limit
            && let Some(oldest) = self.oldest
        {
            self.remove_at(oldest);
        }
        self.held_bytes += plan.bytes;
        let slot = Slot {
            key: key.clone(),
            stage,
            plan,
            newer: NONE,
            older: NONE,
        };
        let at = match self.free.pop() {
            Some(at) => {
                self.slots[at] = Some(slot);
                at
            }
            None => {
                self.slots.push(Some(slot));
                self.slots.len() - 1
            }
        };
        self.index.entry(key).or_insert([NONE; 2])[stage.at()] = at;
        self.link_newest(at);
    }

    fn clear(&mut self) {
        *self = Self::default();
    }

    fn len(&self) -> usize {
        self.slots.len() - self.free.len()
    }

    /// The bytes and the count of each stage's entries.
    fn by_stage(&self) -> (usize, usize, usize) {
        let mut parsed = 0;
        let mut optimized = 0;
        for slot in self.slots.iter().flatten() {
            match slot.stage {
                Stage::Parsed => parsed += slot.plan.bytes,
                Stage::Optimized => optimized += slot.plan.bytes,
            }
        }
        (parsed, optimized, self.len())
    }
}

/// What the cache's bookkeeping takes per entry besides the plan and the
/// key's text: its slot, the map's entry and the key's own value.
const ENTRY_OVERHEAD: usize = std::mem::size_of::<Option<Slot<CacheKey>>>()
    + std::mem::size_of::<(CacheKey, [usize; 2])>()
    + std::mem::size_of::<CachedPlan>();

impl CacheKey {
    /// The bytes the key's text takes (a map key holds it once, a slot once).
    fn text_bytes(&self) -> usize {
        2 * (self.query.capacity()
            + self.schema.as_ref().map_or(0, String::capacity)
            + self.graph.as_ref().map_or(0, String::capacity))
    }
}

/// The plan cache: parsed and optimized plans under one byte bound and one
/// recency order, shared by every session of a database.
///
/// A hit returns the cached plan as an `Arc`: the lock is held only to find
/// it and move it to the newest end. A put stores a plan only when it was
/// planned under the cache's generation, so a plan built before a change of
/// the schema is never kept after it.
pub struct QueryCache {
    lru: Mutex<Lru<CacheKey>>,
    /// The byte bound; 0 turns the cache off.
    limit: usize,
    /// The planning generation the entries were planned under.
    generation: AtomicU64,
    /// The planning generation now: the database's, which its changes and
    /// its stores move ([`advance`](Self::advance)).
    source: Arc<AtomicU64>,
    parsed_hits: AtomicU64,
    parsed_misses: AtomicU64,
    optimized_hits: AtomicU64,
    optimized_misses: AtomicU64,
    /// How many times a clear moved the generation.
    invalidations: AtomicU64,
    /// How many plans were not stored because they exceed the whole limit.
    oversized: AtomicU64,
    /// The pause before each put a test can hold (`hold_puts`).
    #[cfg(feature = "testing-plan-cache")]
    put_hold: Arc<PutHold>,
}

impl QueryCache {
    /// A cache that holds at most `limit` bytes of plans, both stages
    /// together; 0 turns it off.
    #[must_use]
    pub fn with_limit(limit: usize) -> Self {
        Self {
            lru: Mutex::new(Lru::default()),
            limit,
            generation: AtomicU64::new(0),
            source: Arc::new(AtomicU64::new(0)),
            parsed_hits: AtomicU64::new(0),
            parsed_misses: AtomicU64::new(0),
            optimized_hits: AtomicU64::new(0),
            optimized_misses: AtomicU64::new(0),
            invalidations: AtomicU64::new(0),
            oversized: AtomicU64::new(0),
            #[cfg(feature = "testing-plan-cache")]
            put_hold: Arc::default(),
        }
    }

    /// A cache that stores nothing: every statement plans.
    #[must_use]
    pub fn disabled() -> Self {
        Self::with_limit(0)
    }

    /// Whether the cache stores plans.
    #[must_use]
    pub fn is_enabled(&self) -> bool {
        self.limit > 0
    }

    /// The byte bound in force (0: off).
    #[must_use]
    pub fn limit(&self) -> usize {
        self.limit
    }

    /// The planning generation now: a statement reads it before it plans
    /// and passes it to [`put`](Self::put).
    #[must_use]
    pub fn generation(&self) -> u64 {
        self.source.load(Ordering::Acquire)
    }

    /// The planning generation as an atomic a store moves (see
    /// `RowGroupStore::attach_planning_generation`).
    #[must_use]
    pub fn planning_generation(&self) -> &Arc<AtomicU64> {
        &self.source
    }

    /// Moves the planning generation and empties the cache: a change that
    /// any plan may have been made without (a committed change of the
    /// schema, a restore, `clear_plan_cache`).
    pub fn advance(&self) {
        let next = self.source.fetch_add(1, Ordering::AcqRel) + 1;
        self.clear(next);
    }

    /// Empties the cache when the planning generation moved past the one its
    /// entries were planned under (a store's statistics moved it): one atomic
    /// load otherwise.
    fn catch_up(&self) {
        let now = self.source.load(Ordering::Acquire);
        if now > self.generation.load(Ordering::Acquire) {
            self.clear(now);
        }
    }

    /// The plan cached for `key` at `stage`, as the newest entry from now on.
    pub fn get(&self, key: &CacheKey, stage: Stage) -> Option<Arc<CachedPlan>> {
        if !self.is_enabled() {
            return None;
        }
        self.catch_up();
        let found = self.lru.lock().get(key, stage);
        let (hits, misses) = match stage {
            Stage::Parsed => (&self.parsed_hits, &self.parsed_misses),
            Stage::Optimized => (&self.optimized_hits, &self.optimized_misses),
        };
        if found.is_some() {
            hits.fetch_add(1, Ordering::Relaxed);
        } else {
            misses.fetch_add(1, Ordering::Relaxed);
        }
        found
    }

    /// Stores `plan`, planned under `generation`, for `key` at `stage`, and
    /// returns it shared. The entry's bytes are estimated here, outside the
    /// lock. It is not stored when the cache is off, when it was planned
    /// under an older generation than the planning generation now (a change,
    /// or a store's statistics, moved it while it was planned), or when it
    /// alone exceeds the limit (counted in [`CacheStats::oversized`]); the
    /// oldest entries make room for it otherwise.
    pub fn put(
        &self,
        key: CacheKey,
        stage: Stage,
        plan: LogicalPlan,
        generation: u64,
    ) -> Arc<CachedPlan> {
        let bytes = if self.is_enabled() {
            plan.estimated_bytes()
                .saturating_add(key.text_bytes())
                .saturating_add(ENTRY_OVERHEAD)
        } else {
            0
        };
        let cached = Arc::new(CachedPlan { plan, bytes });
        if !self.is_enabled() {
            return cached;
        }
        if bytes > self.limit {
            self.oversized.fetch_add(1, Ordering::Relaxed);
            return cached;
        }
        #[cfg(feature = "testing-plan-cache")]
        self.put_hold.pause();
        // Entries of an older generation go first, so the new one is not
        // cleared with them at the next lookup.
        self.catch_up();
        let mut lru = self.lru.lock();
        // Read under the lock, as a clear moves it: no put of an older
        // generation lands after the clear. A store moves it without the
        // lock: a put that lands just before is cleared at the next lookup.
        if generation < self.source.load(Ordering::Acquire) {
            return cached;
        }
        lru.put(key, stage, Arc::clone(&cached), self.limit);
        cached
    }

    /// Takes the plans of `key` out of both stages.
    pub fn invalidate(&self, key: &CacheKey) {
        let mut lru = self.lru.lock();
        lru.remove(key, Stage::Parsed);
        lru.remove(key, Stage::Optimized);
    }

    /// Empties the cache and moves its generation to `generation`, when that
    /// is newer (the planning generation too): a plan planned under an older
    /// generation is not stored from now on. Each move counts one
    /// invalidation.
    pub fn clear(&self, generation: u64) {
        let mut lru = self.lru.lock();
        self.source.fetch_max(generation, Ordering::AcqRel);
        let previous = self.generation.fetch_max(generation, Ordering::AcqRel);
        if generation > previous {
            self.invalidations.fetch_add(1, Ordering::Relaxed);
        }
        lru.clear();
    }

    /// What the entries are charged together, in bytes.
    #[must_use]
    pub fn held_bytes(&self) -> usize {
        self.lru.lock().held_bytes
    }

    /// The cache's counters.
    #[must_use]
    pub fn stats(&self) -> CacheStats {
        self.catch_up();
        let (parsed_size, optimized_size) = {
            let lru = self.lru.lock();
            let parsed = lru
                .index
                .values()
                .filter(|pair| pair[Stage::Parsed.at()] != NONE)
                .count();
            (parsed, lru.len() - parsed)
        };
        CacheStats {
            parsed_size,
            optimized_size,
            parsed_hits: self.parsed_hits.load(Ordering::Relaxed),
            parsed_misses: self.parsed_misses.load(Ordering::Relaxed),
            optimized_hits: self.optimized_hits.load(Ordering::Relaxed),
            optimized_misses: self.optimized_misses.load(Ordering::Relaxed),
            invalidations: self.invalidations.load(Ordering::Relaxed),
            generation: self.generation.load(Ordering::Acquire),
            session_hits: 0,
            oversized: self.oversized.load(Ordering::Relaxed),
        }
    }

    /// The bytes the parsed and the optimized entries are charged, and how
    /// many entries there are: the deep estimates the entries were stored
    /// with.
    #[must_use]
    pub fn heap_memory_bytes(&self) -> (usize, usize, usize) {
        self.lru.lock().by_stage()
    }

    /// Holds every put from now on until the returned guard drops: a
    /// statement that planned waits before it stores its plan, holding no
    /// lock (feature `testing-plan-cache`).
    #[cfg(feature = "testing-plan-cache")]
    #[must_use]
    pub fn hold_puts(&self) -> PlanPutHold {
        self.put_hold.state.lock().held = true;
        PlanPutHold {
            hold: Arc::clone(&self.put_hold),
        }
    }

    /// Resets the hit, miss, invalidation and oversized counters.
    pub fn reset_stats(&self) {
        self.parsed_hits.store(0, Ordering::Relaxed);
        self.parsed_misses.store(0, Ordering::Relaxed);
        self.optimized_hits.store(0, Ordering::Relaxed);
        self.optimized_misses.store(0, Ordering::Relaxed);
        self.invalidations.store(0, Ordering::Relaxed);
        self.oversized.store(0, Ordering::Relaxed);
    }
}

impl Default for QueryCache {
    /// A cache of [`DEFAULT_LIMIT`] bytes, for a cache made outside a
    /// database (which sizes its own from the memory budget).
    fn default() -> Self {
        Self::with_limit(DEFAULT_LIMIT)
    }
}

/// The pause a test holds before each put (feature `testing-plan-cache`).
#[cfg(feature = "testing-plan-cache")]
#[derive(Default)]
struct PutHold {
    state: Mutex<PutHoldState>,
    changed: parking_lot::Condvar,
}

#[cfg(feature = "testing-plan-cache")]
#[derive(Default)]
struct PutHoldState {
    held: bool,
    waiting: usize,
}

#[cfg(feature = "testing-plan-cache")]
impl PutHold {
    /// Waits while the puts are held.
    fn pause(&self) {
        let mut state = self.state.lock();
        if !state.held {
            return;
        }
        state.waiting += 1;
        self.changed.notify_all();
        while state.held {
            self.changed.wait(&mut state);
        }
        state.waiting -= 1;
    }
}

/// A hold on the plan cache's puts: while it lives, a statement that planned
/// waits before its put; dropping it lets them go on (feature
/// `testing-plan-cache`).
#[cfg(feature = "testing-plan-cache")]
pub struct PlanPutHold {
    hold: Arc<PutHold>,
}

#[cfg(feature = "testing-plan-cache")]
impl PlanPutHold {
    /// Waits until a statement waits at its put, at most `timeout`: whether
    /// one does.
    #[must_use]
    pub fn wait_for_held(&self, timeout: std::time::Duration) -> bool {
        // A timeout past what an instant holds waits without one.
        let deadline = std::time::Instant::now().checked_add(timeout);
        let mut state = self.hold.state.lock();
        while state.waiting == 0 {
            match deadline {
                Some(deadline) => {
                    if self
                        .hold
                        .changed
                        .wait_until(&mut state, deadline)
                        .timed_out()
                    {
                        return state.waiting > 0;
                    }
                }
                None => self.hold.changed.wait(&mut state),
            }
        }
        true
    }
}

#[cfg(feature = "testing-plan-cache")]
impl Drop for PlanPutHold {
    fn drop(&mut self) {
        self.hold.state.lock().held = false;
        self.hold.changed.notify_all();
    }
}

/// The byte bound of a cache made outside a database: the smallest bound a
/// database's default gives.
pub const DEFAULT_LIMIT: usize = 4 * 1024 * 1024;

/// Cache statistics.
#[derive(Debug, Clone)]
#[non_exhaustive]
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
    /// How many times a clear moved the cache's generation.
    pub invalidations: u64,
    /// The planning generation the entries were planned under.
    pub generation: u64,
    /// Hits a session's own cache answered without the shared one.
    pub session_hits: u64,
    /// Plans not stored because they alone exceed the limit.
    pub oversized: u64,
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

    /// Clears the cache, moving it to its next generation.
    pub fn clear_cache(&self) {
        self.cache.advance();
    }
}

// A cache key names a query language: without one there is nothing to cache.
#[cfg(all(test, query_language))]
mod tests {
    use super::*;

    #[cfg(feature = "gql")]
    fn test_language() -> QueryLanguage {
        QueryLanguage::Gql
    }

    #[cfg(not(feature = "gql"))]
    fn test_language() -> QueryLanguage {
        // Without gql, the first language the build has.
        let languages: &[QueryLanguage] = &[
            #[cfg(feature = "cypher")]
            QueryLanguage::Cypher,
            #[cfg(feature = "gremlin")]
            QueryLanguage::Gremlin,
            #[cfg(feature = "graphql")]
            QueryLanguage::GraphQL,
            #[cfg(feature = "sql-pgq")]
            QueryLanguage::SqlPgq,
            #[cfg(feature = "sparql")]
            QueryLanguage::Sparql,
        ];
        languages[0]
    }

    #[test]
    fn test_cache_key_normalization() {
        // The whitespace around a statement goes; inside, a literal's own
        // spacing is the statement's, so the text is kept as written.
        let key1 = CacheKey::new("  MATCH (n) RETURN n ", test_language());
        let key2 = CacheKey::new("MATCH (n) RETURN n", test_language());
        assert_eq!(key1.query(), key2.query());
        let spaced = CacheKey::new("RETURN 'a  b'", test_language());
        let single = CacheKey::new("RETURN 'a b'", test_language());
        assert_eq!(spaced.query(), "RETURN 'a  b'");
        assert_ne!(spaced.query(), single.query());
    }

    use crate::query::plan::{LogicalOperator, LogicalPlan};

    fn plan() -> LogicalPlan {
        LogicalPlan::new(LogicalOperator::Empty)
    }

    #[test]
    fn test_cache_basic_operations() {
        let cache = QueryCache::with_limit(1 << 20);
        let key = CacheKey::new("MATCH (n) RETURN n", test_language());
        assert!(cache.get(&key, Stage::Parsed).is_none());
        cache.put(key.clone(), Stage::Parsed, plan(), 0);
        assert!(cache.get(&key, Stage::Parsed).is_some());
        let stats = cache.stats();
        assert_eq!(
            (stats.parsed_size, stats.parsed_hits, stats.parsed_misses),
            (1, 1, 1)
        );
    }

    #[test]
    fn test_cache_invalidation() {
        let cache = QueryCache::with_limit(1 << 20);
        let key = CacheKey::new("MATCH (n) RETURN n", test_language());
        cache.put(key.clone(), Stage::Parsed, plan(), 0);
        cache.put(key.clone(), Stage::Optimized, plan(), 0);
        cache.invalidate(&key);
        assert!(cache.get(&key, Stage::Parsed).is_none());
        assert!(cache.get(&key, Stage::Optimized).is_none());
        assert_eq!(cache.held_bytes(), 0);
    }

    #[test]
    fn test_cache_disabled() {
        let cache = QueryCache::disabled();
        let key = CacheKey::new("MATCH (n) RETURN n", test_language());
        cache.put(key.clone(), Stage::Parsed, plan(), 0);
        assert!(cache.get(&key, Stage::Parsed).is_none());
        assert_eq!(cache.stats().parsed_size, 0);
        assert_eq!(cache.held_bytes(), 0);
    }

    #[test]
    fn test_cache_stats() {
        let cache = QueryCache::with_limit(1 << 20);
        let key1 = CacheKey::new("QUERY 1", test_language());
        let key2 = CacheKey::new("QUERY 2", test_language());
        cache.get(&key1, Stage::Optimized);
        cache.put(key1.clone(), Stage::Optimized, plan(), 0);
        cache.get(&key1, Stage::Optimized);
        cache.get(&key1, Stage::Optimized);
        cache.get(&key2, Stage::Optimized);
        let stats = cache.stats();
        assert_eq!((stats.optimized_hits, stats.optimized_misses), (2, 2));
        assert!((stats.optimized_hit_rate() - 0.5).abs() < 0.01);
    }

    /// A put under a key it holds replaces the entry: its bytes count once.
    #[test]
    fn a_put_replaces_the_entry_of_its_key() {
        let cache = QueryCache::with_limit(1 << 20);
        let key = CacheKey::new("QUERY 1", test_language());
        cache.put(key.clone(), Stage::Optimized, plan(), 0);
        let once = cache.held_bytes();
        cache.put(key.clone(), Stage::Optimized, plan(), 0);
        assert_eq!(cache.held_bytes(), once);
        assert_eq!(cache.stats().optimized_size, 1);
    }

    /// The deep estimate grows with what a plan owns: a long literal costs
    /// its length.
    #[test]
    fn a_plan_is_charged_for_its_literal() {
        use crate::query::plan::{LogicalExpression, ProjectOp, Projection};
        use grafeo_common::types::Value;

        let literal = |len: usize| {
            LogicalPlan::new(LogicalOperator::Project(ProjectOp {
                projections: vec![Projection {
                    expression: LogicalExpression::Literal(Value::from("x".repeat(len))),
                    alias: Some("s".to_string()),
                }],
                input: Box::new(LogicalOperator::Empty),
                pass_through_input: false,
            }))
        };
        let short = literal(8).estimated_bytes();
        let long = literal(64 * 1024).estimated_bytes();
        assert!(long >= short + 64 * 1024 - 8, "{short} then {long}");
    }
}
