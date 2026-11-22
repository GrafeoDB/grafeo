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
//! - **A cache per session:** in front of the shared one, a session keeps a
//!   hash and a weak reference per plan it ran, so a repeat takes no shared
//!   lock, and the session keeps no plan alive.
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

use grafeo_common::utils::hash::{FxBuildHasher, FxHashMap};
use parking_lot::Mutex;
use std::borrow::Borrow;
use std::collections::HashMap;
use std::hash::BuildHasher;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Weak};

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

    /// The planner rules the plan is made with.
    #[must_use]
    pub fn rules(&self) -> crate::query::optimizer::PlannerRules {
        self.rules
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
pub struct CachedPlan {
    plan: LogicalPlan,
    /// The physical shape a prepared plan's calls open, made on its first call.
    #[cfg(prepared_plans)]
    recipe: std::sync::OnceLock<
        std::result::Result<
            crate::query::planner::lpg::recipe::Recipe,
            crate::query::planner::lpg::recipe::Uncovered,
        >,
    >,
    /// What the entry is charged: the plan's deep estimate, its key's text
    /// and the cache's bookkeeping.
    bytes: usize,
    /// The key and stage the plan is cached under: a session's cache, which
    /// holds only a hash of them, compares them at a hit.
    key: CacheKey,
    stage: Stage,
    /// Whether a session's own cache hit the plan since the shared cache
    /// last passed over it: such an entry gets a second chance at eviction,
    /// as a session hit does not touch the shared recency order.
    used: AtomicBool,
}

impl std::fmt::Debug for CachedPlan {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CachedPlan")
            .field("plan", &self.plan)
            .field("bytes", &self.bytes)
            .field("stage", &self.stage)
            .finish_non_exhaustive()
    }
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

    /// The recipe the calls of this prepared plan open, or the kind it holds that no recipe
    /// covers: made on first use from `inputs`, which every call of one cache key and database
    /// gives alike (read only then), then shared by every call and session.
    #[cfg(prepared_plans)]
    pub(crate) fn recipe(
        &self,
        inputs: impl FnOnce() -> crate::query::planner::lpg::aggregate::RouteInputs,
    ) -> &std::result::Result<
        crate::query::planner::lpg::recipe::Recipe,
        crate::query::planner::lpg::recipe::Uncovered,
    > {
        self.recipe
            .get_or_init(|| crate::query::planner::lpg::recipe::recipe_of(&self.plan, inputs()))
    }

    /// What the cache charges the entry, in bytes.
    #[must_use]
    pub fn bytes(&self) -> usize {
        self.bytes
    }

    /// Whether the plan is the one cached for `key` at `stage`.
    fn is_for(&self, key: &CacheKey, stage: Stage) -> bool {
        self.stage == stage && self.key == *key
    }

    /// Marks the plan used by a session hit. It writes only when the flag is
    /// clear, so sessions hitting one plan do not keep writing its line; the
    /// flag only steers eviction, so `Relaxed` is enough.
    fn mark_used(&self) {
        if !self.used.load(Ordering::Relaxed) {
            self.used.store(true, Ordering::Relaxed);
        }
    }
}

/// No slot: the end of the recency list.
const NONE: usize = usize::MAX;

/// One cached entry, linked in the recency order; its key and stage are the
/// plan's.
struct Slot {
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
/// Generic over its map's key, which is a [`CacheKey`]: a build without a
/// query language has no key value (its [`QueryLanguage`] has no variant),
/// and a generic body is not linted as unreachable there.
struct Lru<K> {
    index: HashMap<K, [usize; 2]>,
    slots: Vec<Option<Slot>>,
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

/// The map key an entry is found by, made from its plan's key (generic, as
/// [`Lru`] is).
trait KeyOf {
    fn key_of(key: &CacheKey) -> Self;
}

impl KeyOf for CacheKey {
    fn key_of(key: &CacheKey) -> Self {
        key.clone()
    }
}

impl<K: std::hash::Hash + Eq + Borrow<CacheKey> + KeyOf> Lru<K> {
    fn slot(&self, at: usize) -> &Slot {
        self.slots[at].as_ref().expect("a linked slot is filled")
    }

    fn slot_mut(&mut self, at: usize) -> &mut Slot {
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

    fn find(&self, key: &CacheKey, stage: Stage) -> Option<usize> {
        let at = self.index.get(key)?[stage.at()];
        (at != NONE).then_some(at)
    }

    fn get(&mut self, key: &CacheKey, stage: Stage) -> Option<Arc<CachedPlan>> {
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
        let key = &slot.plan.key;
        if let Some(pair) = self.index.get_mut(key) {
            pair[slot.plan.stage.at()] = NONE;
            if *pair == [NONE; 2] {
                self.index.remove(key);
            }
        }
        self.free.push(at);
        self.held_bytes -= slot.plan.bytes;
        slot.plan
    }

    fn remove(&mut self, key: &CacheKey, stage: Stage) -> Option<Arc<CachedPlan>> {
        let at = self.find(key, stage)?;
        Some(self.remove_at(at))
    }

    /// Stores `plan` under its key and stage as the newest entry, evicting
    /// the oldest until it fits in `limit`. The caller checked that it fits
    /// alone. An oldest entry a session hit since the last pass (its `used`
    /// flag) has the flag cleared and moves to the newest end instead, at
    /// most once per entry in one put, so the put ends even when every entry
    /// is flagged.
    fn put(&mut self, plan: Arc<CachedPlan>, limit: usize) {
        let stage = plan.stage;
        // An entry the key held goes first: its bytes do not count twice.
        self.remove(&plan.key, stage);
        let mut second_chances = self.len();
        while self.held_bytes + plan.bytes > limit
            && let Some(oldest) = self.oldest
        {
            if second_chances > 0 && self.slot(oldest).plan.used.swap(false, Ordering::Relaxed) {
                second_chances -= 1;
                self.unlink(oldest);
                self.link_newest(oldest);
            } else {
                self.remove_at(oldest);
            }
        }
        self.held_bytes += plan.bytes;
        let key = K::key_of(&plan.key);
        let slot = Slot {
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
            match slot.plan.stage {
                Stage::Parsed => parsed += slot.plan.bytes,
                Stage::Optimized => optimized += slot.plan.bytes,
            }
        }
        (parsed, optimized, self.len())
    }
}

/// What the cache's bookkeeping takes per entry besides the plan and the
/// key's text: its slot, the map's entry and the key's own value.
const ENTRY_OVERHEAD: usize = std::mem::size_of::<Option<Slot>>()
    + std::mem::size_of::<(CacheKey, [usize; 2])>()
    + std::mem::size_of::<CachedPlan>();

impl CacheKey {
    /// The bytes the key's text takes (a map key holds it once, the plan
    /// once).
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
    /// How many lookups a session's own cache answered.
    session_hits: AtomicU64,
    /// The bytes of the open sessions' maps together.
    session_bytes: AtomicUsize,
    /// Hashes a key and stage for a session's map: one seed per database.
    session_keys: FxBuildHasher,
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
            session_hits: AtomicU64::new(0),
            session_bytes: AtomicUsize::new(0),
            session_keys: FxBuildHasher::default(),
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
            grafeo_common::work::count(grafeo_common::work::Work::PlanCacheHit, 1);
        } else {
            misses.fetch_add(1, Ordering::Relaxed);
            grafeo_common::work::count(grafeo_common::work::Work::PlanCacheMiss, 1);
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
        let cached = Arc::new(CachedPlan {
            plan,
            #[cfg(prepared_plans)]
            recipe: std::sync::OnceLock::new(),
            bytes,
            key,
            stage,
            used: AtomicBool::new(false),
        });
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
        lru.put(Arc::clone(&cached), self.limit);
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
            session_hits: self.session_hits.load(Ordering::Relaxed),
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

    /// The bytes the open sessions' own caches take: their maps, which hold
    /// a hash and a weak reference per plan, never a plan (each plan is
    /// counted once, in the shared entries).
    #[must_use]
    pub fn session_bytes(&self) -> usize {
        self.session_bytes.load(Ordering::Relaxed)
    }

    /// Moves a session's counted table bytes from `from` to `to`.
    fn move_session_bytes(&self, from: usize, to: usize) {
        if to > from {
            self.session_bytes.fetch_add(to - from, Ordering::Relaxed);
        } else {
            self.session_bytes.fetch_sub(from - to, Ordering::Relaxed);
        }
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

    /// Resets the hit (the sessions' too), miss, invalidation and oversized
    /// counters.
    pub fn reset_stats(&self) {
        self.session_hits.store(0, Ordering::Relaxed);
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

/// The share of the plan cache's limit that a session's own cache, and a
/// transaction's private cache, may take: a sixteenth. It moves with the one
/// setting, and a limit of 0 turns both off.
pub(crate) const SESSION_SHARE: usize = 16;

/// A session's own cache, in front of the shared one: a repeat in the
/// session takes no shared lock.
///
/// It maps a hash of each key and stage the session ran to a weak reference
/// to the shared cache's plan, so it keeps neither a plan nor a statement's
/// text alive: once the shared cache evicts or clears a plan, the plan is
/// freed, and the session's next lookup of it misses. A hit compares the
/// plan's own key and stage, so a hash that collides is a miss. The map
/// empties at its next lookup once the planning generation moved, and its
/// table stays within a sixteenth of the shared limit.
pub struct SessionPlans {
    shared: Arc<QueryCache>,
    /// The bytes the map may take; 0 turns it off.
    bound: usize,
    map: Mutex<SessionMap>,
}

/// A session's entries, all found or stored under one planning generation.
#[derive(Default)]
struct SessionMap {
    generation: u64,
    entries: FxHashMap<u64, Weak<CachedPlan>>,
    /// The table's bytes, as the shared cache counts them for the session.
    bytes: usize,
}

impl SessionPlans {
    /// An empty cache in front of `shared`.
    #[must_use]
    pub fn new(shared: Arc<QueryCache>) -> Self {
        Self {
            bound: shared.limit / SESSION_SHARE,
            shared,
            map: Mutex::new(SessionMap::default()),
        }
    }

    /// Turns the cache off: a session of one call would drop its entries
    /// unused.
    pub(crate) fn turn_off(&mut self) {
        self.bound = 0;
    }

    /// The shared cache it stands in front of.
    #[must_use]
    pub fn shared(&self) -> &QueryCache {
        &self.shared
    }

    /// The hash the map keeps for `key` at `stage`.
    fn hash(&self, key: &CacheKey, stage: Stage) -> u64 {
        self.shared.session_keys.hash_one((key, stage))
    }

    /// The plan cached for `key` at `stage`: the session's own entry while
    /// its plan is cached, else the shared cache's, remembered for the next
    /// lookup.
    pub fn get(&self, key: &CacheKey, stage: Stage) -> Option<Arc<CachedPlan>> {
        if self.bound == 0 {
            return self.shared.get(key, stage);
        }
        let hash = self.hash(key, stage);
        let now = self.shared.generation();
        {
            let mut map = self.map.lock();
            if map.generation != now {
                map.entries.clear();
                map.generation = now;
            }
            if let Some(weak) = map.entries.get(&hash) {
                match weak.upgrade() {
                    Some(plan) if plan.is_for(key, stage) => {
                        plan.mark_used();
                        self.shared.session_hits.fetch_add(1, Ordering::Relaxed);
                        // A hit as the shared cache counts one: each lookup counts once.
                        grafeo_common::work::count(grafeo_common::work::Work::PlanCacheHit, 1);
                        return Some(plan);
                    }
                    // Evicted, cleared, or another key's plan: a miss.
                    _ => {
                        map.entries.remove(&hash);
                    }
                }
            }
        }
        let found = self.shared.get(key, stage)?;
        self.remember(hash, &found, now);
        Some(found)
    }

    /// Stores `plan`, planned under `generation`, in the shared cache (see
    /// [`QueryCache::put`]), and remembers it for the session.
    pub fn put(
        &self,
        key: CacheKey,
        stage: Stage,
        plan: LogicalPlan,
        generation: u64,
    ) -> Arc<CachedPlan> {
        if self.bound == 0 {
            return self.shared.put(key, stage, plan, generation);
        }
        let hash = self.hash(&key, stage);
        let cached = self.shared.put(key, stage, plan, generation);
        self.remember(hash, &cached, generation);
        cached
    }

    /// Remembers `plan`, found or planned under `generation`, under `hash`,
    /// while the map is at that generation: no plan of an older one enters
    /// it. A colliding entry is replaced. An insert that grows the table past
    /// the bound makes room in the table it had instead (see [`make_room`]).
    fn remember(&self, hash: u64, plan: &Arc<CachedPlan>, generation: u64) {
        let mut map = self.map.lock();
        if map.generation != generation {
            return;
        }
        let capacity = map.entries.capacity();
        map.entries.insert(hash, Arc::downgrade(plan));
        if map.entries.allocation_size() > self.bound {
            make_room(&mut map.entries, hash, capacity);
        }
        let bytes = map.entries.allocation_size();
        self.shared.move_session_bytes(map.bytes, bytes);
        map.bytes = bytes;
    }
}

/// Takes `entries`, which the insert of `kept` grew past its bound, back to
/// the table of `capacity` entries it had: the entries whose plans are gone
/// go first, then others, in the table's order, until an eighth of it is
/// free, so the next inserts fit without another pass (one pass per eighth of
/// the table's inserts). `kept` stays. With a capacity of 0, not even the
/// smallest table fits, and nothing is kept.
fn make_room(entries: &mut FxHashMap<u64, Weak<CachedPlan>>, kept: u64, capacity: usize) {
    if capacity == 0 {
        *entries = FxHashMap::default();
        return;
    }
    entries.retain(|_, weak| weak.strong_count() > 0);
    let mut evict = entries.len().saturating_sub(capacity - capacity / 8);
    entries.retain(|hash, _| {
        if evict == 0 || *hash == kept {
            return true;
        }
        evict -= 1;
        false
    });
    entries.shrink_to(capacity);
}

impl Drop for SessionPlans {
    fn drop(&mut self) {
        let map = self.map.get_mut();
        self.shared.move_session_bytes(map.bytes, 0);
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

    /// A session's map keeps only a hash per key: a plan found under a hash
    /// that collides is another key's, so the lookup misses, and a put under
    /// that hash replaces the entry.
    #[test]
    fn a_colliding_session_entry_is_a_miss() {
        let shared = Arc::new(QueryCache::with_limit(1 << 20));
        let session = SessionPlans::new(Arc::clone(&shared));
        let gus = CacheKey::new("QUERY gus", test_language());
        let mia = CacheKey::new("QUERY mia", test_language());
        let generation = shared.generation();
        let gus_plan = session.put(gus.clone(), Stage::Optimized, plan(), generation);
        // Mia's hash leads to Gus's plan, as a collision would.
        let hash = session.hash(&mia, Stage::Optimized);
        session
            .map
            .lock()
            .entries
            .insert(hash, Arc::downgrade(&gus_plan));
        assert!(session.get(&mia, Stage::Optimized).is_none());
        assert_eq!(shared.stats().session_hits, 0);

        let mia_plan = session.put(mia.clone(), Stage::Optimized, plan(), generation);
        let found = session.get(&mia, Stage::Optimized).expect("Mia's plan");
        assert!(Arc::ptr_eq(&found, &mia_plan), "the put replaced the entry");
        assert_eq!(shared.stats().session_hits, 1);
        let found = session.get(&gus, Stage::Optimized).expect("Gus's plan");
        assert!(Arc::ptr_eq(&found, &gus_plan));
    }

    /// The bytes the table of a session's map allocates once it holds
    /// `capacity` entries.
    fn table_bytes(capacity: usize) -> usize {
        let mut entries: FxHashMap<u64, Weak<CachedPlan>> = FxHashMap::default();
        entries.reserve(capacity);
        entries.allocation_size()
    }

    /// A session that runs more statements than its map's bound fits keeps
    /// most of them: an insert that would grow the table past the bound
    /// evicts an eighth of it instead, the newest entry kept, and the bytes
    /// counted are the table's as allocated.
    #[test]
    fn a_session_map_past_its_bound_keeps_most_entries() {
        let bound = table_bytes(28);
        let shared = Arc::new(QueryCache::with_limit(bound * SESSION_SHARE));
        let session = SessionPlans::new(Arc::clone(&shared));
        // Held here, so every entry's plan stays alive whatever the shared
        // cache evicts: the room is made by the bound alone.
        let mut plans = Vec::new();
        let mut most = 0;
        for i in 0..200 {
            let key = CacheKey::new(format!("QUERY {i}"), test_language());
            plans.push(session.put(key.clone(), Stage::Optimized, plan(), shared.generation()));
            {
                let map = session.map.lock();
                let allocated = map.entries.allocation_size();
                assert_eq!(shared.session_bytes(), allocated, "put {i}: as allocated");
                assert!(allocated <= bound, "put {i}: {allocated} over {bound}");
                most = most.max(map.entries.len());
                assert!(
                    map.entries.len() >= most - most / 8,
                    "put {i}: {} entries kept of {most}",
                    map.entries.len()
                );
            }
            let hits = shared.stats().session_hits;
            assert!(session.get(&key, Stage::Optimized).is_some());
            assert_eq!(shared.stats().session_hits, hits + 1, "put {i} is kept");
        }
        assert!(most >= 28, "the table filled up to the bound: {most}");
        drop(session);
        assert_eq!(shared.session_bytes(), 0, "the session's bytes go with it");
    }

    /// A bound below the smallest table keeps nothing: every lookup goes to
    /// the shared cache, and no byte is counted.
    #[test]
    fn a_bound_below_the_smallest_table_keeps_nothing() {
        let bound = table_bytes(1) - 1;
        let shared = Arc::new(QueryCache::with_limit(bound * SESSION_SHARE));
        let session = SessionPlans::new(Arc::clone(&shared));
        let key = CacheKey::new("QUERY gus", test_language());
        let _plan = session.put(key.clone(), Stage::Optimized, plan(), shared.generation());
        assert!(
            session.get(&key, Stage::Optimized).is_some(),
            "a shared hit"
        );
        let stats = shared.stats();
        assert_eq!((stats.session_hits, stats.optimized_hits), (0, 1));
        assert_eq!(shared.session_bytes(), 0);
        assert_eq!(session.map.lock().entries.allocation_size(), 0);
    }

    /// Every entry marked used by a session hit, then a put: each gets its
    /// second chance once, and the put still evicts (the oldest, as without
    /// the marks) and keeps the bound.
    #[test]
    fn a_put_over_entries_all_marked_used_still_evicts() {
        let key = |text: &str| CacheKey::new(text, test_language());
        let entry = QueryCache::with_limit(1 << 20)
            .put(key("query-a"), Stage::Optimized, plan(), 0)
            .bytes();
        let cache = Arc::new(QueryCache::with_limit(3 * entry));
        for text in ["query-a", "query-b", "query-c"] {
            cache
                .put(key(text), Stage::Optimized, plan(), 0)
                .mark_used();
        }
        let (done, ended) = std::sync::mpsc::channel();
        let putting = Arc::clone(&cache);
        std::thread::spawn(move || {
            putting.put(key("query-d"), Stage::Optimized, plan(), 0);
            done.send(()).expect("the test waits");
        });
        ended
            .recv_timeout(std::time::Duration::from_secs(10))
            .expect("the put ends");
        assert!(cache.held_bytes() <= 3 * entry);
        let holds = |text: &str| cache.get(&key(text), Stage::Optimized).is_some();
        assert!(
            !holds("query-a"),
            "the oldest goes once each had its chance"
        );
        assert!(holds("query-b") && holds("query-c") && holds("query-d"));
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
