//! The plan cache's bounds and freshness (#674, the plan-cache half).
//!
//! - **Bound:** the cache is bounded in bytes by `Config::with_plan_cache_limit`; `0` turns it
//!   off. The default is a share of the memory budget, and an explicit value is capped at the
//!   budget. One bound and one recency order hold both stages (parsed and optimized), and every
//!   put leaves `held_bytes <= limit`.
//! - **Plans and the report:** a plan's bytes are a deep estimate, reported in
//!   `memory_usage().caches`. A plan larger than the whole limit runs and is not stored. A hit
//!   hands out a shared plan, never a clone.
//! - **Generation:** one planning generation, moved by committed DDL, a restore,
//!   `clear_plan_cache` and a large statistics change, drops every older plan. A plan built
//!   under an older generation is returned but never stored.
//! - **Statistics:** a count that doubles or halves, by at least 1,000 rows, since its anchor
//!   moves the generation.
//! - **Session cache:** a per-session cache sits in front of the shared one.
//!
//! ```bash
//! cargo test -p grafeo-engine --all-features --test main plan_cache_bounds::
//! ```

#![cfg(feature = "gql")]

use grafeo_common::types::Value;
use grafeo_engine::config::PlanCacheLimit;
use grafeo_engine::{Config, GrafeoDB};

const KIB: usize = 1024;
const MIB: usize = 1024 * KIB;
const GIB: usize = 1024 * MIB;

/// Shared-cache hits so far. One-shot calls (`db.execute`) each take a fresh session, so no
/// session cache stands in front of these.
fn hits(db: &GrafeoDB) -> u64 {
    db.query_cache().stats().optimized_hits
}

/// Whether `query` ran on a cached plan.
fn cached(db: &GrafeoDB, query: &str) -> bool {
    let before = hits(db);
    db.execute(query).unwrap();
    hits(db) > before
}

fn with_limit(limit: usize) -> GrafeoDB {
    GrafeoDB::with_config(Config::in_memory().with_plan_cache_limit(limit)).unwrap()
}

/// The bytes both stages hold, as the report gives them.
fn held(db: &GrafeoDB) -> usize {
    let caches = db.memory_usage().caches;
    caches.parsed_plan_cache_bytes + caches.optimized_plan_cache_bytes
}

/// A one-row query with a string literal of `len` bytes.
fn literal(len: usize) -> String {
    format!("RETURN '{}' AS s", "x".repeat(len))
}

/// The setting reads back, and the bound in force follows the budget. The default's figures
/// are option (a) of the design's question to the user, a sixty-fourth of the budget from 4 to
/// 64 MiB: pinned once the answer arrives (`DEFAULT_SHARE`, `FLOOR`, `CAP`).
#[test]
fn the_plan_cache_limit_follows_its_default_and_setting() {
    const FLOOR: usize = 4 * MIB;
    const CAP: usize = 64 * MIB;
    const DEFAULT_SHARE: usize = 64;
    assert_eq!(
        Config::in_memory().plan_cache_limit(),
        PlanCacheLimit::Default
    );
    assert_eq!(
        Config::in_memory()
            .with_plan_cache_limit(8 * MIB)
            .plan_cache_limit(),
        PlanCacheLimit::Bytes(8 * MIB)
    );
    let budget = |bytes: usize| Config::in_memory().with_memory_limit(bytes);
    for (case, config, bound) in [
        (
            "a share of 768 MiB",
            budget(768 * MIB),
            768 * MIB / DEFAULT_SHARE,
        ),
        ("a share of 1 GiB", budget(GIB), GIB / DEFAULT_SHARE),
        ("the floor under 64 MiB", budget(64 * MIB), FLOOR),
        ("the cap over 8 GiB", budget(8 * GIB), CAP),
        (
            "an explicit 8 MiB",
            budget(GIB).with_plan_cache_limit(8 * MIB),
            8 * MIB,
        ),
        (
            "an explicit limit above the budget",
            budget(32 * MIB).with_plan_cache_limit(GIB),
            32 * MIB,
        ),
        ("0, off", budget(GIB).with_plan_cache_limit(0), 0),
    ] {
        let db = GrafeoDB::with_config(config).unwrap();
        assert_eq!(
            db.memory_usage().caches.plan_cache_limit_bytes,
            bound,
            "{case}"
        );
    }
}

/// `0` turns both caches off: every statement plans, nothing is stored or counted.
#[test]
fn a_limit_of_zero_caches_nothing() {
    let db = with_limit(0);
    db.execute("INSERT (:Person {name: 'Alix'})").unwrap();
    let query = "MATCH (p:Person) WHERE p.name = $name RETURN p.name";
    for _ in 0..3 {
        assert!(!cached(&db, "MATCH (p:Person) RETURN p.name"));
        let mut params = std::collections::HashMap::new();
        params.insert("name".to_string(), Value::from("Alix"));
        let rows = db.execute_with_params(query, params).unwrap();
        assert_eq!(rows.rows().len(), 1);
    }
    let stats = db.query_cache().stats();
    assert_eq!((stats.parsed_hits, stats.optimized_hits), (0, 0));
    let caches = db.memory_usage().caches;
    assert_eq!(caches.total_bytes, 0);
    assert_eq!(caches.cached_plan_count, 0);
}

/// Statements of every size, most of them fitting: after each one the cache holds at most its
/// limit, and the statement just run is the one it keeps.
#[test]
fn the_cache_stays_within_its_limit() {
    let limit = 64 * KIB;
    let db = with_limit(limit);
    for i in 0..400 {
        let query = format!("RETURN {i} AS i, '{}' AS s", "x".repeat(i * 7 % 3000));
        db.execute(&query).unwrap();
        assert!(held(&db) <= limit, "statement {i}: {} bytes", held(&db));
        assert!(cached(&db, &query), "statement {i} fits, so it is kept");
    }
    assert_eq!(db.memory_usage().caches.plan_cache_limit_bytes, limit);
}

/// The estimate is deep: a plan holding a 64 KiB string adds at least 64 KiB to the report
/// (today's estimate counts the map's buckets and nothing a plan owns).
#[test]
fn the_report_counts_what_a_plan_holds() {
    let db = GrafeoDB::new_in_memory();
    db.execute("RETURN 1 AS one").unwrap();
    let before = held(&db);
    db.execute(&literal(64 * KIB)).unwrap();
    let after = held(&db);
    assert!(
        after >= before + 64 * KIB,
        "a 64 KiB plan added {} bytes",
        after.saturating_sub(before)
    );
    let caches = db.memory_usage().caches;
    assert_eq!(caches.cached_plan_count, 2);
    assert_eq!(
        caches.total_bytes,
        caches.parsed_plan_cache_bytes
            + caches.optimized_plan_cache_bytes
            + caches.session_plan_cache_bytes
    );
}

/// A plan larger than the whole limit runs and is not stored, the next run plans again, and
/// the refusal counts in `oversized`. What the cache held stays.
#[test]
fn a_plan_over_the_limit_runs_and_is_not_stored() {
    let limit = 16 * KIB;
    let db = with_limit(limit);
    let small = "RETURN 1 AS one";
    db.execute(small).unwrap();
    let big = literal(32 * KIB);
    for _ in 0..2 {
        let rows = db.execute(&big).unwrap().rows().to_vec();
        assert_eq!(rows, vec![vec![Value::from("x".repeat(32 * KIB))]]);
    }
    assert!(!cached(&db, &big), "an oversized plan is never a hit");
    assert!(db.query_cache().stats().oversized >= 1);
    assert!(held(&db) <= limit);
    assert!(cached(&db, small), "an oversized put evicts nothing");
}

/// The cache's own API: one recency order over both stages, the byte bound, oversized plans,
/// stale puts, the clear to a newer generation and shared hits.
///
/// API names are the design's (section 4): `QueryCache::with_limit`, `Stage`, `put` with a
/// generation, `get` returning `Arc<CachedPlan>`, `clear` to a generation, `held_bytes`.
mod cache {
    use std::sync::Arc;

    use grafeo_engine::query::cache::{CacheKey, QueryCache, Stage};
    use grafeo_engine::query::plan::{LogicalOperator, LogicalPlan};
    use grafeo_engine::query::processor::QueryLanguage;

    fn key(text: &str) -> CacheKey {
        CacheKey::new(text, QueryLanguage::Gql)
    }

    fn plan() -> LogicalPlan {
        LogicalPlan::new(LogicalOperator::Empty)
    }

    /// The bytes an entry under `text` and `stage` is charged, read from a roomy cache.
    fn bytes_of(text: &str, stage: Stage) -> usize {
        let cache = QueryCache::with_limit(1 << 30);
        cache.put(key(text), stage, plan(), 0);
        cache.get(&key(text), stage).unwrap().bytes()
    }

    fn holds(cache: &QueryCache, text: &str, stage: Stage) -> bool {
        cache.get(&key(text), stage).is_some()
    }

    /// Three entries fill the cache; a touch makes the first the most recent, so the next put
    /// evicts the second, across stages, and the one after evicts the third. A check is a get,
    /// which touches too, so every check comes after the last put.
    #[test]
    fn eviction_follows_one_recency_order_over_both_stages() {
        let entry = bytes_of("query-a", Stage::Parsed);
        assert_eq!(
            entry,
            bytes_of("query-b", Stage::Optimized),
            "keys of one length and one plan are charged alike in both stages"
        );
        let cache = QueryCache::with_limit(3 * entry);
        cache.put(key("query-a"), Stage::Parsed, plan(), 0);
        cache.put(key("query-b"), Stage::Optimized, plan(), 0);
        cache.put(key("query-c"), Stage::Parsed, plan(), 0);
        assert_eq!(cache.held_bytes(), 3 * entry);
        assert!(holds(&cache, "query-a", Stage::Parsed), "a touch");

        // Oldest first: b, c, a. The put of d evicts b, leaving c, a, d; the put of e evicts c.
        cache.put(key("query-d"), Stage::Optimized, plan(), 0);
        cache.put(key("query-e"), Stage::Parsed, plan(), 0);
        assert!(cache.held_bytes() <= 3 * entry);
        assert!(
            !holds(&cache, "query-b", Stage::Optimized),
            "b was the oldest at the put of d"
        );
        assert!(
            !holds(&cache, "query-c", Stage::Parsed),
            "c was the oldest at the put of e"
        );
        assert!(holds(&cache, "query-a", Stage::Parsed), "the touch kept a");
        assert!(holds(&cache, "query-d", Stage::Optimized));
        assert!(holds(&cache, "query-e", Stage::Parsed));
    }

    /// The same text in both stages is two entries.
    #[test]
    fn a_text_in_both_stages_is_two_entries() {
        let cache = QueryCache::with_limit(1 << 20);
        cache.put(key("query-a"), Stage::Parsed, plan(), 0);
        assert!(!holds(&cache, "query-a", Stage::Optimized));
        cache.put(key("query-a"), Stage::Optimized, plan(), 0);
        assert!(holds(&cache, "query-a", Stage::Parsed));
        assert!(holds(&cache, "query-a", Stage::Optimized));
    }

    /// Entries of many sizes: every put leaves at most the limit held.
    #[test]
    fn every_put_stays_within_the_limit() {
        let limit = 16 * 1024;
        let cache = QueryCache::with_limit(limit);
        for i in 0..2_000 {
            let text = format!("q{i}-{}", "x".repeat(i * 13 % 900));
            let stage = if i % 3 == 0 {
                Stage::Parsed
            } else {
                Stage::Optimized
            };
            cache.put(key(&text), stage, plan(), 0);
            assert!(
                cache.held_bytes() <= limit,
                "put {i}: {}",
                cache.held_bytes()
            );
            assert!(holds(&cache, &text, stage), "put {i} fits, so it is kept");
        }
    }

    /// An entry larger than the limit is not stored and evicts nothing.
    #[test]
    fn an_oversized_entry_is_counted_and_not_stored() {
        let small = bytes_of("query-a", Stage::Optimized);
        let cache = QueryCache::with_limit(small * 4);
        cache.put(key("query-a"), Stage::Optimized, plan(), 0);
        let big = "x".repeat(small * 8);
        cache.put(key(&big), Stage::Optimized, plan(), 0);
        assert!(!holds(&cache, &big, Stage::Optimized));
        assert!(holds(&cache, "query-a", Stage::Optimized));
        assert_eq!(cache.stats().oversized, 1);
        assert_eq!(cache.held_bytes(), small);
    }

    /// A clear to a newer generation empties the cache once (one invalidation), and a put
    /// planned under the older generation afterwards is dropped.
    #[test]
    fn a_put_from_an_older_generation_is_dropped() {
        let cache = QueryCache::with_limit(1 << 20);
        cache.put(key("query-a"), Stage::Optimized, plan(), 0);
        cache.clear(1);
        let stats = cache.stats();
        assert_eq!((stats.generation, stats.invalidations), (1, 1));
        assert_eq!(cache.held_bytes(), 0);

        cache.put(key("query-b"), Stage::Optimized, plan(), 0);
        assert!(
            !holds(&cache, "query-b", Stage::Optimized),
            "planned before the clear"
        );
        assert_eq!(cache.held_bytes(), 0);
        cache.put(key("query-b"), Stage::Optimized, plan(), 1);
        assert!(holds(&cache, "query-b", Stage::Optimized));
    }

    /// Two hits hand out one plan.
    #[test]
    fn two_hits_share_one_plan() {
        let cache = QueryCache::with_limit(1 << 20);
        cache.put(key("query-a"), Stage::Optimized, plan(), 0);
        let first = cache.get(&key("query-a"), Stage::Optimized).unwrap();
        let second = cache.get(&key("query-a"), Stage::Optimized).unwrap();
        assert!(Arc::ptr_eq(&first, &second));
    }
}
