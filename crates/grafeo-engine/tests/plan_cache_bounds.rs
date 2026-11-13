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

/// The plan `EXPLAIN query` shows, as it renders with the indexes there now: `[index: ...]` comes
/// from the schema at the render, so the text cannot tell a stale cached plan from a fresh one.
fn explain(db: &GrafeoDB, query: &str) -> String {
    let rows = db
        .execute(&format!("EXPLAIN {query}"))
        .unwrap()
        .rows()
        .to_vec();
    match &rows[0][0] {
        Value::String(text) => text.to_string(),
        other => panic!("EXPLAIN `{query}` returned {other:?}"),
    }
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

/// The planning generation: each of its four sources makes the next run plan again; what is
/// rolled back moves nothing; a schema change queued in an open transaction moves it at commit.
mod generation {
    use std::collections::HashMap;

    use super::*;

    const QUERY: &str = "MATCH (p:Person) WHERE p.name = 'Alix' RETURN p.name AS name";

    fn generation(db: &GrafeoDB) -> u64 {
        db.query_cache().stats().generation
    }

    fn invalidations(db: &GrafeoDB) -> u64 {
        db.query_cache().stats().invalidations
    }

    /// 2,000 people, Alix among them.
    fn people() -> GrafeoDB {
        let db = GrafeoDB::new_in_memory();
        db.execute("INSERT (:Person {name: 'Alix'})").unwrap();
        db.execute(
            "UNWIND range(1, 1999) AS i INSERT (:Person {name: 'Gus' || CAST(i AS STRING)})",
        )
        .unwrap();
        db
    }

    fn alix(db: &GrafeoDB) -> Vec<Vec<Value>> {
        db.execute(QUERY).unwrap().rows().to_vec()
    }

    /// Committed DDL, a restore, `clear_plan_cache` and a statistics move: each moves the
    /// generation, counts one invalidation, and the next run plans again with the same answer.
    #[test]
    fn each_source_moves_the_generation_and_the_next_run_replans() {
        type Source = fn(&GrafeoDB);
        let sources: [(&str, Source); 4] = [
            ("a committed index", |db| {
                db.create_property_index("name").unwrap();
            }),
            ("a restore", |db| {
                let snapshot = db.export_snapshot().unwrap();
                db.restore_snapshot(&snapshot).unwrap();
            }),
            ("clear_plan_cache", GrafeoDB::clear_plan_cache),
            ("a label from 0 to 5,000", |db| {
                db.execute("UNWIND range(1, 5000) AS i INSERT (:Admin {i: i})")
                    .unwrap();
            }),
        ];
        let db = people();
        for (source, apply) in sources {
            db.execute(QUERY).unwrap();
            assert!(cached(&db, QUERY), "{source}: warm");
            let (moved, cleared) = (generation(&db), invalidations(&db));
            apply(&db);
            assert!(generation(&db) > moved, "{source}: the generation moves");
            assert!(!cached(&db, QUERY), "{source}: the next run plans again");
            assert_eq!(
                invalidations(&db),
                cleared + 1,
                "{source}: one invalidation"
            );
            assert_eq!(alix(&db), vec![vec![Value::from("Alix")]], "{source}");
        }
        assert!(explain(&db, QUERY).contains("[index:"), "the index is used");
    }

    /// A bulk insert and an index, each rolled back: the cached plan stays, and the
    /// generation does not move.
    #[test]
    fn what_is_rolled_back_moves_nothing() {
        let db = people();
        db.execute(QUERY).unwrap();
        let moved = generation(&db);
        let mut session = db.session();
        session.begin_transaction().unwrap();
        session
            .execute("UNWIND range(1, 5000) AS i INSERT (:Person {name: 'Vincent'})")
            .unwrap();
        session.rollback().unwrap();
        assert!(cached(&db, QUERY), "a rolled-back insert");

        session.begin_transaction().unwrap();
        session
            .execute("CREATE INDEX person_name FOR (p:Person) ON (p.name)")
            .unwrap();
        session.rollback().unwrap();
        assert!(cached(&db, QUERY), "a rolled-back index");
        assert_eq!(generation(&db), moved);
        assert!(!explain(&db, QUERY).contains("[index:"));
    }

    /// An index queued in an open transaction leaves every other session on the cached plan;
    /// at its commit the generation moves and the next run uses the index.
    #[test]
    fn a_schema_change_moves_the_generation_at_its_commit() {
        let db = people();
        explain(&db, QUERY);
        db.execute(QUERY).unwrap();
        let moved = generation(&db);
        let mut session = db.session();
        session.begin_transaction().unwrap();
        session
            .execute("CREATE INDEX person_name FOR (p:Person) ON (p.name)")
            .unwrap();
        assert!(cached(&db, QUERY), "queued, not committed");
        assert_eq!(generation(&db), moved);
        session.commit().unwrap();
        assert!(generation(&db) > moved);
        assert!(!cached(&db, QUERY));
        assert!(explain(&db, QUERY).contains("[index:"));
    }

    /// A statement with values keeps its prepared plan until the generation moves, then
    /// plans again on its next call, by a statistics move or an index.
    #[test]
    fn a_prepared_plan_replans_after_a_generation_move() {
        let db = people();
        let query = "MATCH (p:Person) WHERE p.name = $name RETURN p.name AS name";
        let call = |db: &GrafeoDB| {
            let before = hits(db);
            let mut params = HashMap::new();
            params.insert("name".to_string(), Value::from("Alix"));
            let rows = db
                .execute_with_params(query, params)
                .unwrap()
                .rows()
                .to_vec();
            assert_eq!(rows, vec![vec![Value::from("Alix")]]);
            hits(db) > before
        };
        call(&db);
        assert!(call(&db), "the prepared plan is cached");
        db.execute("UNWIND range(1, 5000) AS i INSERT (:Admin {i: i})")
            .unwrap();
        assert!(!call(&db), "after a statistics move");
        assert!(call(&db));
        db.create_property_index("name").unwrap();
        assert!(!call(&db), "after an index");
    }

    /// One generation for the database: an index on one graph replans another graph's plans.
    #[test]
    fn an_index_on_one_graph_replans_every_graph() {
        let db = GrafeoDB::new_in_memory();
        let setup = db.session();
        setup.execute("CREATE GRAPH g1").unwrap();
        setup.execute("CREATE GRAPH g2").unwrap();
        let in_graph = |name: &str| {
            let session = db.session();
            session.execute(&format!("USE GRAPH {name}")).unwrap();
            session
        };
        in_graph("g2")
            .execute("INSERT (:Person {name: 'Jules'})")
            .unwrap();
        let run_in_g2 = || {
            let before = hits(&db);
            let rows = in_graph("g2")
                .execute("MATCH (p:Person) RETURN p.name AS name")
                .unwrap()
                .rows()
                .to_vec();
            assert_eq!(rows, vec![vec![Value::from("Jules")]]);
            hits(&db) > before
        };
        run_in_g2();
        assert!(run_in_g2(), "cached in g2");
        in_graph("g1")
            .execute("CREATE INDEX person_name FOR (p:Person) ON (p.name)")
            .unwrap();
        assert!(!run_in_g2(), "an index on g1 replans g2's plans");
    }
}

/// Large statistics changes: a count that doubles or halves since its anchor, by at least
/// `PLAN_STATISTICS_MIN_ROWS = 1_000` rows, moves the generation at its commit, so the next run
/// plans again. The counts are total nodes and edges, each label and each edge type. Every insert
/// or delete below is one commit. A first bulk of City nodes (or RAIL edges) sets the total's
/// anchor apart, so only the count under test moves.
///
/// The thresholds are pinned through hits and misses; which plan the optimizer then picks is its
/// own business. `a_plan_made_on_an_empty_graph_turns_after_a_bulk_load` shows the issue's case
/// end to end, with a choice the optimizer makes from the counts (the far-end turn). A label
/// scan's choice of label is no oracle: the physical planner makes it from the live counts on
/// every run (`planner/lpg/scan.rs`).
mod statistics {
    use super::*;

    const QUERY: &str = "MATCH (p:Person) RETURN count(p) AS n";

    fn insert(db: &GrafeoDB, label: &str, from: usize, to: usize) {
        db.execute(&format!(
            "UNWIND range({from}, {to}) AS i INSERT (:{label} {{i: i}})"
        ))
        .unwrap();
    }

    /// The query runs and counts `people`; whether it ran on a cached plan.
    fn run(db: &GrafeoDB, people: i64) -> bool {
        let before = hits(db);
        let rows = db.execute(QUERY).unwrap().rows().to_vec();
        assert_eq!(rows, vec![vec![Value::Int64(people)]]);
        hits(db) > before
    }

    /// From an empty label: 999 Person keep the plan, the 1,000th replans.
    #[test]
    fn from_an_empty_label_a_thousand_rows_replan() {
        let db = GrafeoDB::new_in_memory();
        insert(&db, "City", 1, 5000);
        run(&db, 0);
        assert!(run(&db, 0), "cached");

        insert(&db, "Person", 1, 999);
        assert!(run(&db, 999), "999 rows keep the plan");

        insert(&db, "Person", 1000, 1000);
        assert!(!run(&db, 1000), "the 1,000th row replans");
        assert!(run(&db, 1000));
    }

    /// From 10,000 going up: 19,999 keep the plan, 20,000 replan.
    #[test]
    fn from_ten_thousand_going_up_twice_the_anchor_replans() {
        let db = GrafeoDB::new_in_memory();
        insert(&db, "City", 1, 30_000);
        insert(&db, "Person", 1, 10_000);
        run(&db, 10_000);
        assert!(run(&db, 10_000), "cached");

        insert(&db, "Person", 10_001, 19_999);
        assert!(run(&db, 19_999), "19,999 keep the plan");

        insert(&db, "Person", 20_000, 20_000);
        assert!(!run(&db, 20_000), "20,000 replan");
    }

    /// From 10,000 going down: 5,001 keep the plan, 5,000 replan.
    #[test]
    fn from_ten_thousand_going_down_half_the_anchor_replans() {
        let db = GrafeoDB::new_in_memory();
        insert(&db, "City", 1, 30_000);
        insert(&db, "Person", 1, 10_000);
        run(&db, 10_000);
        assert!(run(&db, 10_000), "cached");

        db.execute("MATCH (p:Person) WHERE p.i <= 4999 DELETE p")
            .unwrap();
        assert!(run(&db, 5_001), "5,001 keep the plan");

        db.execute("MATCH (p:Person) WHERE p.i = 5000 DELETE p")
            .unwrap();
        assert!(!run(&db, 5_000), "5,000 replan");
    }

    /// The anchor sits where the last move happened: a count that crosses back and forth near it
    /// never moves the generation again.
    #[test]
    fn a_count_going_back_and_forth_moves_once() {
        let db = GrafeoDB::new_in_memory();
        insert(&db, "City", 1, 5000);
        insert(&db, "Person", 1, 1000);
        run(&db, 1000);
        assert!(run(&db, 1000), "cached");
        for round in 0..3 {
            insert(&db, "Person", 1001, 1999);
            assert!(run(&db, 1999), "round {round}: up to 1,999");
            db.execute("MATCH (p:Person) WHERE p.i > 1000 DELETE p")
                .unwrap();
            assert!(run(&db, 1000), "round {round}: back to 1,000");
        }
    }

    /// A move in an edge type's count replans too.
    #[test]
    fn an_edge_type_count_moves_the_generation() {
        let db = GrafeoDB::new_in_memory();
        insert(&db, "City", 1, 5000);
        db.execute("MATCH (c:City) INSERT (c)-[:RAIL]->(c)")
            .unwrap();
        db.execute("MATCH (c:City) WHERE c.i <= 10 INSERT (c)-[:ROAD]->(c)")
            .unwrap();
        let query = "MATCH (a:City)-[:ROAD]->(b:City) RETURN count(*) AS n";
        db.execute(query).unwrap();
        assert!(cached(&db, query));
        db.execute("MATCH (c:City) WHERE c.i > 10 AND c.i <= 2010 INSERT (c)-[:ROAD]->(c)")
            .unwrap();
        assert!(!cached(&db, query), "ROAD from 10 to 2,010");
    }

    /// The issue's case: a plan made on an empty graph is made again after a bulk load. On an
    /// empty graph the hop from `a:Start` to `m:Far {k: 7}` starts at its source. Once 2,000
    /// starts and 5,000 far nodes are loaded, with no schema change in between, the optimizer
    /// turns it to start at the far end through its index (`far_end_turn.rs`,
    /// `a_hub_start_found_by_a_scan_turns`).
    #[test]
    fn a_plan_made_on_an_empty_graph_turns_after_a_bulk_load() {
        let db = GrafeoDB::new_in_memory();
        db.execute("CREATE INDEX by_far FOR (n:Far) ON (n.k)")
            .unwrap();
        let query = "MATCH (a:Start)-[:R]->(m:Far {k: 7}) WHERE a.s = 1 RETURN a.s AS s, m.k AS k";
        let plan = explain(&db, query);
        assert!(
            !plan.contains("Expand (m)"),
            "on an empty graph the hop starts at its source: {plan}"
        );

        db.execute("UNWIND range(1, 2000) AS i INSERT (:Start {s: i})")
            .unwrap();
        db.execute("UNWIND range(1, 5000) AS i INSERT (:Far {k: i})")
            .unwrap();
        db.execute("MATCH (a:Start {s: 1}), (m:Far) INSERT (a)-[:R]->(m)")
            .unwrap();
        let rows = db.execute(query).unwrap().rows().to_vec();
        assert_eq!(rows, vec![vec![Value::Int64(1), Value::Int64(7)]]);
        let plan = explain(&db, query);
        assert!(
            plan.contains("Expand (m)"),
            "after the load the hop starts at the far end: {plan}"
        );
    }
}

/// A plan built before a schema change is never stored after it.
mod race {
    use std::sync::atomic::{AtomicBool, Ordering};

    use super::*;

    const QUERY: &str = "MATCH (p:Person) WHERE p.name = 'Mia' RETURN p.name AS name";

    fn people() -> GrafeoDB {
        let db = GrafeoDB::new_in_memory();
        db.execute(
            "UNWIND range(1, 2000) AS i INSERT (:Person {name: 'Butch' || CAST(i AS STRING)})",
        )
        .unwrap();
        db.execute("INSERT (:Person {name: 'Mia'})").unwrap();
        db
    }

    /// Deterministic: a statement pauses between its optimize and its put while another
    /// session creates an index. Its plan, made under the generation before the index, runs but
    /// is not stored: the next run misses and plans again, and the one after hits. The paused
    /// statement is the query itself, so its put and the checks share one cache key (an `EXPLAIN`
    /// of it is another key). The plan's text is no oracle: `EXPLAIN` marks `[index: ...]` from
    /// the indexes there when it renders, after the pause; the cache's hits and misses are.
    ///
    /// `hold_plan_puts()` (feature `testing-plan-cache`) returns a guard: while it lives, a
    /// statement that planned waits before its put; `wait_for_held(timeout)` returns once one
    /// waits; dropping the guard lets it go on. The paused statement holds no commit lock, so the
    /// index commits meanwhile.
    #[cfg(feature = "testing-plan-cache")]
    #[test]
    fn a_plan_from_before_an_index_is_not_stored_after_it() {
        let db = people();
        std::thread::scope(|scope| {
            let hold = db.hold_plan_puts();
            let paused = scope.spawn(|| db.execute(QUERY).unwrap().rows().to_vec());
            assert!(
                hold.wait_for_held(std::time::Duration::from_secs(10)),
                "the statement reaches its put"
            );
            db.create_property_index("name").unwrap();
            drop(hold);
            let answered = paused.join().unwrap();
            assert_eq!(
                answered,
                [vec![Value::from("Mia")]],
                "the paused statement answered with its own plan"
            );
        });
        assert!(
            !cached(&db, QUERY),
            "nothing of the old generation was kept"
        );
        assert!(cached(&db, QUERY));
    }

    /// A plan made before a statistics move is not held after it: a statement pauses at its
    /// put while 5,000 direct inserts more than double the label's 2,001 nodes (past the
    /// 1,000-row floor), which moves only the live planning generation. The put must compare with that, not with the entries'
    /// generation, which catches up only at the next lookup. Read with `held_bytes()` straight
    /// away, since a lookup or `stats()` would catch up first and hide a stale entry.
    #[cfg(feature = "testing-plan-cache")]
    #[test]
    fn a_plan_from_before_a_statistics_move_is_not_held() {
        let db = people();
        std::thread::scope(|scope| {
            let hold = db.hold_plan_puts();
            let paused = scope.spawn(|| db.execute(QUERY).unwrap().rows().to_vec());
            assert!(
                hold.wait_for_held(std::time::Duration::from_secs(10)),
                "the statement reaches its put"
            );
            // Through the direct API, which plans nothing: a statement would wait at its own put.
            for i in 0..5_000 {
                db.create_node_with_props(&["Person"], [("name", Value::from(format!("Hans{i}")))])
                    .unwrap();
            }
            drop(hold);
            assert_eq!(
                paused.join().unwrap(),
                [vec![Value::from("Mia")]],
                "the paused statement answered with its own plan"
            );
        });
        assert_eq!(
            db.query_cache().held_bytes(),
            0,
            "nothing planned before the move is held, read with no lookup first"
        );
    }

    /// The net: one thread runs the EXPLAIN in a loop while the other creates and drops the
    /// index 100 times; each change moves the planning generation at its commit, so no plan of
    /// the generation before it is served after it, and the query answers right throughout.
    #[test]
    fn a_plan_matches_the_schema_after_every_change() {
        let db = people();
        let stop = AtomicBool::new(false);
        std::thread::scope(|scope| {
            scope.spawn(|| {
                while !stop.load(Ordering::Relaxed) {
                    explain(&db, QUERY);
                }
            });
            for round in 0..100 {
                let indexed = round % 2 == 0;
                let generation = db.query_cache().stats().generation;
                if indexed {
                    db.create_property_index("name").unwrap();
                } else {
                    assert!(db.drop_property_index("name").unwrap());
                }
                assert!(
                    db.query_cache().stats().generation > generation,
                    "round {round}, index {indexed}: the change moved the generation"
                );
                let rows = db.execute(QUERY).unwrap();
                assert_eq!(
                    rows.rows(),
                    [vec![Value::from("Mia")]],
                    "round {round}, index {indexed}"
                );
            }
            stop.store(true, Ordering::Relaxed);
        });
    }
}
