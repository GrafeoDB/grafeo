//! An unbounded `WALK` has no hop cap (#678, item 11): on a graph without a
//! cycle it returns every walk, also one longer than `min + 100` hops, where
//! it used to stop. On a cycle, where walks never end, it fails with an error
//! that says how to bound it, once it reaches the query's memory budget; it
//! neither runs forever nor stops silently. TRAIL, SIMPLE and ACYCLIC end on
//! their own and are unchanged.

#![cfg(feature = "gql")]

use grafeo_common::types::Value;
use grafeo_engine::{Config, GrafeoDB};

/// 150 `:N` nodes, `i` 0 to 149, in a chain of `:NEXT` edges.
fn chain(db: &GrafeoDB) {
    db.execute("FOR i IN range(0, 149) INSERT (:N {i: i})")
        .unwrap();
    db.execute("MATCH (a:N), (b:N) WHERE b.i = a.i + 1 INSERT (a)-[:NEXT]->(b)")
        .unwrap();
}

fn value(db: &GrafeoDB, query: &str) -> Value {
    let result = db.execute(query).unwrap_or_else(|e| panic!("{query}: {e}"));
    assert_eq!(result.rows().len(), 1, "{query}");
    result.rows()[0][0].clone()
}

#[test]
fn an_unbounded_walk_without_a_cycle_returns_every_walk() {
    let db = GrafeoDB::new_in_memory();
    chain(&db);
    for query in [
        "MATCH (a:N {i: 0})-[*]->(b) RETURN count(*)",
        "MATCH (a:N {i: 0})-[:NEXT]->{1,}(b) RETURN count(*)",
        "MATCH (a:N {i: 0})-[:NEXT]->+(b) RETURN count(*)",
        "MATCH WALK (a:N {i: 0})-[*]->(b) RETURN count(*)",
    ] {
        assert_eq!(value(&db, query), Value::Int64(149), "{query}");
    }
    // The longest walk has 149 hops, past min + 100.
    assert_eq!(
        value(&db, "MATCH p = (a:N {i: 0})-[*]->(b) RETURN max(length(p))"),
        Value::Int64(149)
    );
    assert_eq!(
        value(&db, "MATCH (a:N {i: 0})-[*]->(b) RETURN max(b.i)"),
        Value::Int64(149)
    );
    // From every node: 149 + 148 + ... + 1 walks.
    assert_eq!(
        value(&db, "MATCH (a:N)-[*]->(b) RETURN count(*)"),
        Value::Int64(149 * 150 / 2)
    );
    // `{0,}` adds the walk of no hops.
    assert_eq!(
        value(&db, "MATCH (a:N {i: 0})-[:NEXT]->{0,}(b) RETURN count(*)"),
        Value::Int64(150)
    );
}

/// A ring of three: 0 -> 1 -> 2 -> 0.
fn ring(db: &GrafeoDB) {
    db.execute(
        "INSERT (a:N {i: 0}), (b:N {i: 1}), (c:N {i: 2}), \
         (a)-[:NEXT]->(b), (b)-[:NEXT]->(c), (c)-[:NEXT]->(a)",
    )
    .unwrap();
}

#[test]
fn trail_simple_and_acyclic_on_a_cycle_are_unchanged() {
    let db = GrafeoDB::new_in_memory();
    ring(&db);
    for (mode, rows) in [("TRAIL", 3), ("SIMPLE", 3), ("ACYCLIC", 2)] {
        let query = format!("MATCH {mode} (a:N {{i: 0}})-[*]->(b) RETURN count(*)");
        assert_eq!(value(&db, &query), Value::Int64(rows), "{query}");
    }
}

/// The error of an unbounded walk around a cycle: it names the fix.
fn assert_unbounded_walk_error(db: &GrafeoDB, query: &str) {
    let error = db
        .execute(query)
        .expect_err(&format!("{query}: an unbounded walk around a cycle"));
    let message = error.to_string();
    assert!(
        message.contains("bound the quantifier, or use TRAIL, ACYCLIC or ANY SHORTEST"),
        "{query}: {message}"
    );
}

/// Two cycles through one node: the walks double with every lap, so the
/// search reaches its budget (a quarter of the memory limit) quickly.
#[test]
fn an_unbounded_walk_around_two_cycles_fails_at_its_budget() {
    let db = GrafeoDB::with_config(Config::in_memory().with_memory_limit(4 * 1024 * 1024)).unwrap();
    db.execute(
        "INSERT (hub:N {i: 0}), (a:N {i: 1}), (b:N {i: 2}), \
         (hub)-[:NEXT]->(a), (a)-[:NEXT]->(hub), (hub)-[:NEXT]->(b), (b)-[:NEXT]->(hub)",
    )
    .unwrap();
    for query in [
        "MATCH (s:N {i: 0})-[*]->(t) RETURN count(*)",
        "MATCH (s:N {i: 0})-[:NEXT]->{1,}(t) RETURN count(*)",
        "MATCH p = (s:N {i: 0})-[*]->(t) RETURN count(p)",
    ] {
        assert_unbounded_walk_error(&db, query);
    }
    // Bounded, it answers.
    assert_eq!(
        value(&db, "MATCH (s:N {i: 0})-[:NEXT]->{1,4}(t) RETURN count(*)"),
        Value::Int64(2 + 2 + 4 + 4)
    );
}

/// One cycle: a walk around it holds one path at a time, longer each lap,
/// so the budget is reached only because a path's length counts against it.
/// The query fails with the error, not at its timeout. (A memory limit of
/// 256 KiB keeps it short: `count(p)` builds every path, each longer than
/// the last.)
#[test]
fn an_unbounded_walk_around_one_cycle_fails_at_its_budget() {
    let db = GrafeoDB::with_config(Config::in_memory().with_memory_limit(256 * 1024)).unwrap();
    ring(&db);
    for query in [
        "MATCH (a:N {i: 0})-[*]->(b) RETURN count(*)",
        "MATCH p = (a:N {i: 0})-[*]->(b) RETURN count(p)",
    ] {
        assert_unbounded_walk_error(&db, query);
    }
}

/// Around a cycle an unbounded walk still streams: a query that asks for a
/// few rows gets them, also past the old cap of min + 100 hops, and stops.
#[test]
fn an_unbounded_walk_around_a_cycle_streams_the_rows_a_limit_asks_for() {
    let db = GrafeoDB::with_config(Config::in_memory().with_memory_limit(256 * 1024)).unwrap();
    ring(&db);
    let result = db
        .execute("MATCH p = (a:N {i: 0})-[*]->(b) RETURN length(p) AS hops, b.i AS i LIMIT 150")
        .unwrap();
    let rows: Vec<(i64, i64)> = result
        .rows()
        .iter()
        .map(|row| (row[0].as_int64().unwrap(), row[1].as_int64().unwrap()))
        .collect();
    // Shortest first: hop h ends at node h mod 3.
    let expected: Vec<(i64, i64)> = (1..=150).map(|hops| (hops, hops % 3)).collect();
    assert_eq!(rows, expected);
}
