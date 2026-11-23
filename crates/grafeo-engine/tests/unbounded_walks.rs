//! An unbounded `WALK` (#678, item 11) is refused in GQL since 0.6.4: a quantifier without an
//! upper bound needs a path mode or a selector (ISO/IEC 39075:2024 16.4, syntax rule 14), so
//! `-[*]->`, `{1,}`, `+` and `{0,}` in WALK mode fail with `42001` before they run, on a cycle or
//! not. TRAIL, SIMPLE and ACYCLIC end on their own: on a chain a trail is every walk, also one
//! longer than `min + 100` hops (the old cap), and on a cycle they are unchanged. A bounded walk
//! still runs, and still streams the rows a `LIMIT` asks for.

#![cfg(feature = "gql")]

use grafeo_common::types::Value;
use grafeo_common::utils::gqlstatus::GqlStatus;
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

/// `query` is refused before it runs: `42001`, naming a path mode and a selector.
fn assert_refused(db: &GrafeoDB, query: &str) {
    let error = db.execute(query).expect_err(query);
    assert_eq!(
        GqlStatus::from(&error).as_str(),
        "42001",
        "{query}: {error}"
    );
    let message = error.to_string();
    assert!(
        message.contains("TRAIL") && message.contains("ANY SHORTEST"),
        "{query}: {message}"
    );
}

#[test]
fn an_unbounded_walk_is_refused_and_a_trail_returns_every_path_of_a_chain() {
    let db = GrafeoDB::new_in_memory();
    chain(&db);
    for query in [
        "MATCH (a:N {i: 0})-[*]->(b) RETURN count(*)",
        "MATCH (a:N {i: 0})-[:NEXT]->{1,}(b) RETURN count(*)",
        "MATCH (a:N {i: 0})-[:NEXT]->+(b) RETURN count(*)",
        "MATCH WALK (a:N {i: 0})-[*]->(b) RETURN count(*)",
        "MATCH (a:N {i: 0})-[:NEXT]->{0,}(b) RETURN count(*)",
    ] {
        assert_refused(&db, query);
    }
    for query in [
        "MATCH TRAIL (a:N {i: 0})-[*]->(b) RETURN count(*)",
        "MATCH TRAIL (a:N {i: 0})-[:NEXT]->{1,}(b) RETURN count(*)",
        "MATCH TRAIL (a:N {i: 0})-[:NEXT]->+(b) RETURN count(*)",
    ] {
        assert_eq!(value(&db, query), Value::Int64(149), "{query}");
    }
    // The longest trail has 149 hops, past min + 100.
    assert_eq!(
        value(
            &db,
            "MATCH p = TRAIL (a:N {i: 0})-[*]->(b) RETURN max(length(p))"
        ),
        Value::Int64(149)
    );
    assert_eq!(
        value(&db, "MATCH TRAIL (a:N {i: 0})-[*]->(b) RETURN max(b.i)"),
        Value::Int64(149)
    );
    // From every node: 149 + 148 + ... + 1 trails.
    assert_eq!(
        value(&db, "MATCH TRAIL (a:N)-[*]->(b) RETURN count(*)"),
        Value::Int64(149 * 150 / 2)
    );
    // `{0,}` adds the trail of no hops.
    assert_eq!(
        value(
            &db,
            "MATCH TRAIL (a:N {i: 0})-[:NEXT]->{0,}(b) RETURN count(*)"
        ),
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

/// Two cycles through one node, where the walks double with every lap: an unbounded walk is
/// refused at once, and a bounded one answers.
#[test]
fn an_unbounded_walk_around_two_cycles_is_refused() {
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
        assert_refused(&db, query);
    }
    // Bounded, it answers.
    assert_eq!(
        value(&db, "MATCH (s:N {i: 0})-[:NEXT]->{1,4}(t) RETURN count(*)"),
        Value::Int64(2 + 2 + 4 + 4)
    );
}

/// One cycle, where an unbounded walk never ends: refused at once.
#[test]
fn an_unbounded_walk_around_one_cycle_is_refused() {
    let db = GrafeoDB::with_config(Config::in_memory().with_memory_limit(256 * 1024)).unwrap();
    ring(&db);
    for query in [
        "MATCH (a:N {i: 0})-[*]->(b) RETURN count(*)",
        "MATCH p = (a:N {i: 0})-[*]->(b) RETURN count(p)",
    ] {
        assert_refused(&db, query);
    }
}

/// Around a cycle a bounded walk streams: a query that asks for a few rows gets them, also past
/// the old cap of min + 100 hops, and stops.
#[test]
fn a_bounded_walk_around_a_cycle_streams_the_rows_a_limit_asks_for() {
    let db = GrafeoDB::with_config(Config::in_memory().with_memory_limit(256 * 1024)).unwrap();
    ring(&db);
    let result = db
        .execute(
            "MATCH p = (a:N {i: 0})-[*1..200]->(b) RETURN length(p) AS hops, b.i AS i LIMIT 150",
        )
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
