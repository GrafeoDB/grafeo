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

use grafeo_common::types::{PropertyKey, Value};
use grafeo_engine::GrafeoDB;

/// Inserts `n` `:Item` nodes with property `r` set to a deterministic
/// pseudo-random `Int64` derived from the index. Returns the DB.
fn seed_items(n: usize) -> GrafeoDB {
    let db = GrafeoDB::new_in_memory();
    db.session()
        .execute(&format!(
            "FOR i IN range(1, {n}) INSERT (:Item {{id: i - 1, r: (i - 1) * 2654435761 % 1000000}})"
        ))
        .unwrap();
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

// ORDER BY keys that wrap a RETURN-dropped variable inside a slice, list
// comprehension or list predicate must still take the augmenting projection,
// both on the plain Sort path and the ORDER BY ... LIMIT top-K path.
#[test]
fn order_by_wrapped_dropped_variable_sorts_correctly() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    session
        .execute("INSERT (:Seq {id: 0, s: [1, 2, 3]})")
        .unwrap();
    session.execute("INSERT (:Seq {id: 1, s: [1]})").unwrap();
    session.execute("INSERT (:Seq {id: 2, s: [1, 2]})").unwrap();

    let ids = |query: &str| -> Vec<Value> {
        let result = session
            .execute(query)
            .unwrap_or_else(|e| panic!("{query}: {e}"));
        result.rows().iter().map(|row| row[0].clone()).collect()
    };
    let expected =
        |order: &[i64]| -> Vec<Value> { order.iter().map(|&i| Value::Int64(i)).collect() };

    for limit in ["", " LIMIT 3"] {
        // Slice: remaining lengths are 2, 0, 1.
        assert_eq!(
            ids(&format!(
                "MATCH (n:Seq) RETURN n.id AS id ORDER BY size(n.s[1..]){limit}"
            )),
            expected(&[1, 2, 0]),
            "slice key{limit}"
        );
        // List comprehension: sizes 3, 1, 2.
        assert_eq!(
            ids(&format!(
                "MATCH (n:Seq) RETURN n.id AS id ORDER BY size([x IN n.s WHERE x > 0 | x * 2]) DESC{limit}"
            )),
            expected(&[0, 2, 1]),
            "list comprehension key{limit}"
        );
        // List predicate: only node 0 contains 3; node 2 is excluded to avoid ties.
        assert_eq!(
            ids(&format!(
                "MATCH (n:Seq) WHERE n.id < 2 RETURN n.id AS id ORDER BY any(x IN n.s WHERE x = 3) DESC{limit}"
            )),
            expected(&[0, 1]),
            "list predicate key{limit}"
        );
    }
}

// The same for keys wrapping a dropped variable in `reduce` or a subquery,
// which the variable collector used to skip.
#[cfg(feature = "cypher")]
#[test]
fn order_by_reduce_or_subquery_over_dropped_variable_sorts_correctly() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    session
        .execute(
            "INSERT (a:Seq {id: 0, s: [1, 2, 3]}), (b:Seq {id: 1, s: [1]}), \
                    (c:Seq {id: 2, s: [1, 2]}), (t:Target), \
                    (a)-[:R]->(t), (c)-[:R]->(t), (c)-[:R]->(t)",
        )
        .unwrap();

    let ids = |query: &str| -> Vec<Value> {
        let result = session
            .execute_cypher(query)
            .unwrap_or_else(|e| panic!("{query}: {e}"));
        result.rows().iter().map(|row| row[0].clone()).collect()
    };
    let expected =
        |order: &[i64]| -> Vec<Value> { order.iter().map(|&i| Value::Int64(i)).collect() };

    for limit in ["", " LIMIT 3"] {
        // reduce: sums are 6, 1, 3.
        assert_eq!(
            ids(&format!(
                "MATCH (n:Seq) RETURN n.id AS id ORDER BY reduce(acc = 0, x IN n.s | acc + x){limit}"
            )),
            expected(&[1, 2, 0]),
            "reduce key{limit}"
        );
        // COUNT subquery: out-degrees are 1, 0, 2.
        assert_eq!(
            ids(&format!(
                "MATCH (n:Seq) RETURN n.id AS id ORDER BY COUNT {{ MATCH (n)-->() }} DESC{limit}"
            )),
            expected(&[2, 0, 1]),
            "COUNT subquery key{limit}"
        );
    }
}

// Issues #335 and #347: `RETURN n ORDER BY <key> LIMIT k` must return `n` as a
// resolved map for every sort-key shape (property, function call, CASE, binary).
// The heap top-K rewrite used to plan the input speculatively and leave planner
// state behind when it bailed out, turning `n` into a raw NodeId.

#[cfg(feature = "text-index")]
#[test]
fn order_by_limit_text_score_key_yields_map() {
    // Issue #347: a function-call sort key (text_score).
    use std::collections::HashMap;

    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    session
        .execute("INSERT (:Article {title: 'A1', body: 'rust database internals'})")
        .unwrap();
    db.create_text_index("Article", "body").expect("index");

    let params = HashMap::from([("sub".to_string(), Value::from("database"))]);

    // Without LIMIT: already worked, asserting it stays correct.
    let result = session
        .execute_with_params(
            "MATCH (s:Article) RETURN s, text_score(s.body, $sub) \
             ORDER BY text_score(s.body, $sub) DESC",
            params.clone(),
        )
        .unwrap();
    assert!(
        result.rows()[0][0].as_map().is_some(),
        "without LIMIT: expected Map, got {:?}",
        result.rows()[0][0]
    );

    // With LIMIT: the failing case in #347.
    let result = session
        .execute_with_params(
            "MATCH (s:Article) RETURN s, text_score(s.body, $sub) \
             ORDER BY text_score(s.body, $sub) DESC LIMIT 50",
            params.clone(),
        )
        .unwrap();
    assert!(
        result.rows()[0][0].as_map().is_some(),
        "with LIMIT: expected Map, got {:?}",
        result.rows()[0][0]
    );
}

#[test]
fn order_by_limit_case_key_yields_map() {
    // CASE sort key over a variable that is also returned whole.
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    session
        .execute("INSERT (:Article {title: 'A1', tier: 1})")
        .unwrap();
    session
        .execute("INSERT (:Article {title: 'A2', tier: 2})")
        .unwrap();

    let result = session
        .execute(
            "MATCH (n:Article) RETURN n \
             ORDER BY CASE n.tier WHEN 1 THEN 0 ELSE 1 END LIMIT 50",
        )
        .unwrap();

    assert_eq!(result.row_count(), 2);
    let titles: Vec<Value> = result
        .rows()
        .iter()
        .map(|row| {
            let map = row[0]
                .as_map()
                .unwrap_or_else(|| panic!("expected Map, got {:?}", row[0]));
            map.get(&PropertyKey::new("title")).cloned().expect("title")
        })
        .collect();
    assert_eq!(titles, vec![Value::from("A1"), Value::from("A2")]);
}

#[test]
fn order_by_limit_binary_key_yields_map() {
    // Binary expression sort key, variable in Return.
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    session.execute("INSERT (:Item {a: 3, b: 5})").unwrap();
    session.execute("INSERT (:Item {a: 1, b: 2})").unwrap();

    let result = session
        .execute("MATCH (n:Item) RETURN n ORDER BY n.a + n.b DESC LIMIT 50")
        .unwrap();

    assert_eq!(result.row_count(), 2);
    let sums: Vec<Value> = result
        .rows()
        .iter()
        .map(|row| {
            let map = row[0]
                .as_map()
                .unwrap_or_else(|| panic!("expected Map, got {:?}", row[0]));
            map.get(&PropertyKey::new("a")).cloned().expect("a")
        })
        .collect();
    // a + b is 8 for a = 3 and 3 for a = 1; DESC puts a = 3 first.
    assert_eq!(sums, vec![Value::Int64(3), Value::Int64(1)]);
}
