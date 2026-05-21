//! Integration tests for the heap-based top-K rewrite added in
//! `query/planner/lpg/project.rs::try_heap_topk_rewrite`.
//!
//! These tests exercise end-to-end via `session.execute()`, asserting result
//! correctness on the cases the rewrite fires for and the cases that should
//! fall through.
//!
//! Direct string-match verification that the heap rewrite fired isn't
//! possible from outside the engine: EXPLAIN walks the logical tree (which
//! the rewrite leaves unchanged: the fusion is physical-only, see the
//! `try_topk_rewrite` doc), and PROFILE gates the rewrite off so its output
//! never names TopK either. PROFILE is still useful for the *negative*
//! direction (test 18 confirms the gate works by asserting Sort + Limit
//! both run with timings); silent fall-through under non-PROFILE mode is
//! the regression class caught by the §2.9 e2e benchmark in `benches/topk_e2e.rs`.

#![cfg(feature = "lpg")]

use grafeo_common::types::Value;
use grafeo_engine::GrafeoDB;

/// Inserts `n` `:Item` nodes with property `r` set to a deterministic
/// pseudo-random `Int64` derived from the index. Returns the DB.
///
/// One INSERT per row: the GQL/Cypher INSERT path doesn't accept
/// arithmetic expressions in property sources, so a single
/// `UNWIND range(...) INSERT (:Item {r: i * k % m})` shape is rejected.
/// Tests use small `n`, so the per-row cost is acceptable here.
fn seed_items(n: usize) -> GrafeoDB {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    for i in 0..n {
        let r = seed_value(i as u64);
        session
            .execute(&format!("INSERT (:Item {{id: {i}, r: {r}}})"))
            .unwrap();
    }
    db
}

/// Replays the seed formula in Rust so tests can compute expected values
/// without re-running the database.
fn seed_value(i: u64) -> i64 {
    // reason: deterministic pseudo-random in [0, 1_000_000), fits in i64.
    #[allow(clippy::cast_possible_wrap)]
    let v = (i.wrapping_mul(2_654_435_761) % 1_000_000) as i64;
    v
}

/// Returns the EXPLAIN plan text for `query` against `db`.
fn explain(db: &GrafeoDB, query: &str) -> String {
    let session = db.session();
    let result = session
        .execute(&format!("EXPLAIN {query}"))
        .expect("EXPLAIN should not fail");
    match &result.rows()[0][0] {
        Value::String(s) => s.to_string(),
        other => panic!("EXPLAIN should return String, got {other:?}"),
    }
}

/// Returns the PROFILE plan text for `query` against `db`.
fn profile(db: &GrafeoDB, query: &str) -> String {
    let session = db.session();
    let result = session
        .execute(&format!("PROFILE {query}"))
        .expect("PROFILE should not panic");
    match &result.rows()[0][0] {
        Value::String(s) => s.to_string(),
        other => panic!("PROFILE should return String, got {other:?}"),
    }
}

#[cfg(feature = "vector-index")]
#[test]
fn cypher_vector_topk_still_fires_first() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    // Embeddings spread by angle so cosine similarity to [1,0,0] is monotone:
    // id=0 has [1,9,0] (mostly y-axis, low sim), id=9 has [10,0,0] (x-axis, sim=1).
    for i in 0..10 {
        // reason: fixture indices stay below i64::MAX.
        #[allow(clippy::cast_possible_wrap)]
        let x = (i + 1) as i64;
        #[allow(clippy::cast_possible_wrap)]
        let y = (9 - i) as i64;
        session
            .execute(&format!(
                "INSERT (:Doc {{id: {i}, embedding: [{x}.0, {y}.0, 0.0]}})"
            ))
            .unwrap();
    }
    db.create_vector_index(
        "Doc",
        "embedding",
        Some(3),
        Some("cosine"),
        None,
        None,
        None,
    )
    .unwrap();

    let result = session
        .execute(
            "MATCH (d:Doc) RETURN d.id \
             ORDER BY cosine_similarity(d.embedding, [1.0, 0.0, 0.0]) DESC LIMIT 3",
        )
        .unwrap();
    assert_eq!(result.row_count(), 3);

    let top_id = match &result.rows()[0][0] {
        Value::Int64(i) => *i,
        other => panic!("expected Int64 id, got {other:?}"),
    };
    assert_eq!(top_id, 9, "id=9 has embedding closest to [1,0,0]");
}

#[test]
fn cypher_order_by_after_optional_match_uses_topk() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    // 10 :Person nodes; even-indexed ones have an outgoing :KNOWS to a :Friend.
    for i in 0..10 {
        session
            .execute(&format!("INSERT (:Person {{id: {i}, r: {i}}})"))
            .unwrap();
        if i % 2 == 0 {
            session
                .execute(&format!(
                    "MATCH (p:Person {{id: {i}}}) INSERT (p)-[:KNOWS]->(:Friend {{tag: {i}}})"
                ))
                .unwrap();
        }
    }

    let result = session
        .execute(
            "MATCH (p:Person) OPTIONAL MATCH (p)-[:KNOWS]->(f:Friend) \
             RETURN p.id, f.tag ORDER BY p.r DESC LIMIT 3",
        )
        .unwrap();
    assert_eq!(result.row_count(), 3);
    // Top 3 by p.r DESC: ids 9, 8, 7 (9 has no :Friend, so f.tag is Null).
    assert_eq!(result.rows()[0][0], Value::Int64(9));
    assert_eq!(result.rows()[1][0], Value::Int64(8));
    assert_eq!(result.rows()[2][0], Value::Int64(7));
}

#[test]
fn cypher_order_by_with_filter_uses_topk() {
    let db = seed_items(88);
    let session = db.session();

    // WHERE filter sits below Sort; the rewrite plans sort.input as-is and
    // wraps with TopK, so the filter still pushes via the existing path.
    let result = session
        .execute("MATCH (n:Item) WHERE n.r > 100000 RETURN n.r ORDER BY n.r DESC LIMIT 5")
        .unwrap();

    // Compute expected: 88 ids, take r > 100_000, sort DESC, take 5.
    let mut expected: Vec<i64> = (0..88_u64)
        .map(seed_value)
        .filter(|r| *r > 100_000)
        .collect();
    expected.sort_unstable_by(|a, b| b.cmp(a));
    expected.truncate(5);

    let actual: Vec<i64> = result
        .rows()
        .iter()
        .map(|row| match &row[0] {
            Value::Int64(r) => *r,
            other => panic!("expected Int64, got {other:?}"),
        })
        .collect();

    assert_eq!(actual, expected, "filter + sort + top-K result mismatch");
    for r in &actual {
        assert!(*r > 100_000, "filter should be honoured: r={r}");
    }
}

#[test]
fn cypher_order_by_aggregate_alias_falls_through() {
    let db = seed_items(19);
    // ORDER BY uses the aggregate alias `c`. plan_sort needs the augmenting
    // projection path; the heap rewrite must defer.
    let session = db.session();
    let result = session
        .execute("MATCH (n:Item) RETURN n.id, count(*) AS c ORDER BY c DESC LIMIT 5")
        .unwrap();
    // 19 distinct ids, each with count 1 after the GROUP BY n.id implicit
    // grouping; LIMIT picks 5 from 19.
    assert_eq!(result.row_count(), 5);
    for row in result.rows() {
        assert_eq!(row[1], Value::Int64(1), "every group has count 1");
    }
}

#[test]
fn cypher_skip_limit_falls_through() {
    let db = seed_items(19);
    let session = db.session();
    let result = session
        .execute("MATCH (n:Item) RETURN n.r ORDER BY n.r DESC SKIP 5 LIMIT 5")
        .unwrap();
    assert_eq!(result.row_count(), 5);

    // Plan should be Limit over Skip over Sort (separate operators); the
    // rewrite only fires when limit.input is Sort directly.
    let plan = explain(
        &db,
        "MATCH (n:Item) RETURN n.r ORDER BY n.r DESC SKIP 5 LIMIT 5",
    );
    assert!(
        plan.contains("Skip"),
        "Plan should contain Skip operator:\n{plan}"
    );
}

#[test]
fn cypher_order_by_limit_unfused_under_profile() {
    let db = seed_items(19);
    let plan = profile(&db, "MATCH (n:Item) RETURN n.r ORDER BY n.r DESC LIMIT 5");

    // PROFILE should not panic, and the unfused path should run: both Sort
    // and Limit operators must be visible.
    assert!(
        plan.contains("Sort"),
        "PROFILE under heap-rewrite-disabled should show Sort:\n{plan}"
    );
    assert!(
        plan.contains("Limit"),
        "PROFILE under heap-rewrite-disabled should show Limit:\n{plan}"
    );
    assert!(
        plan.contains("rows="),
        "PROFILE should report row counts:\n{plan}"
    );
    // Defensive: confirm the rewrite did NOT fire under PROFILE.
    assert!(
        !plan.contains("TopK"),
        "PROFILE must not show TopK; rewrite should be gated by !profiling:\n{plan}"
    );
}

#[test]
fn cypher_order_by_limit_uses_topk() {
    let db = seed_items(88);
    let session = db.session();

    let result = session
        .execute("MATCH (n:Item) RETURN n.r ORDER BY n.r DESC LIMIT 5")
        .unwrap();

    assert_eq!(result.row_count(), 5);

    // Returned values should be the 5 highest `r` in DESC order.
    let mut all: Vec<i64> = (0..88_u64).map(seed_value).collect();
    all.sort_unstable_by(|a, b| b.cmp(a));
    let expected_top5: Vec<Value> = all.iter().take(5).map(|&v| Value::Int64(v)).collect();
    let actual_top5: Vec<Value> = result.rows().iter().map(|row| row[0].clone()).collect();
    assert_eq!(actual_top5, expected_top5);
}

// Regression test for issue #335: ORDER BY + LIMIT on a full-node RETURN
// was returning raw NodeIds instead of resolved maps. The heap top-K probe
// inside try_heap_topk_rewrite mutated scalar_columns as a side effect,
// causing the unfused re-plan of the same Return subtree to skip NodeResolve.
#[test]
fn order_by_limit_node_return_yields_map() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    session
        .execute("INSERT (:Article {title: 'A1', body: 'rust database internals'})")
        .unwrap();

    // Each variant must return Value::Map, not a raw integer NodeId.
    let result = session.execute("MATCH (n:Article) RETURN n").unwrap();
    assert!(result.rows()[0][0].as_map().is_some(), "bare RETURN n");

    let result = session
        .execute("MATCH (n:Article) RETURN n LIMIT 50")
        .unwrap();
    assert!(result.rows()[0][0].as_map().is_some(), "RETURN n LIMIT 50");

    let result = session
        .execute("MATCH (n:Article) RETURN n ORDER BY n.title")
        .unwrap();
    assert!(
        result.rows()[0][0].as_map().is_some(),
        "RETURN n ORDER BY n.title"
    );

    let result = session
        .execute("MATCH (n:Article) RETURN n ORDER BY n.title LIMIT 50")
        .unwrap();
    assert!(
        result.rows()[0][0].as_map().is_some(),
        "RETURN n ORDER BY n.title LIMIT 50: col 0 = {:?}",
        result.rows()[0][0]
    );
}
