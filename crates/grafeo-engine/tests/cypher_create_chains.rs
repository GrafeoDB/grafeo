//! A run of consecutive Cypher CREATE clauses, with no other clause between
//! them, is one create operation holding their patterns in order, so a long
//! chain of them does not reach the plan depth limit (#798, openCypher TCK
//! Create4). It creates what the clauses one after the other would, with the
//! same counters, and a pattern failing in the middle writes nothing. A
//! CREATE that reads the graph (`EXISTS`, `COUNT`, a pattern comprehension)
//! reads what the CREATE before it created, so neither joins the other.
//!
//! ```bash
//! cargo test -p grafeo-engine --all-features --test main cypher_create_chains::
//! ```

#![cfg(feature = "cypher")]

use grafeo_common::types::Value;
use grafeo_engine::GrafeoDB;
use grafeo_engine::database::WriteCounters;

fn count(db: &GrafeoDB, query: &str) -> i64 {
    db.execute_cypher(query)
        .unwrap_or_else(|error| panic!("`{query}` failed: {error}"))
        .scalar()
        .unwrap()
}

fn counters(db: &GrafeoDB, query: &str) -> WriteCounters {
    db.execute_cypher(query)
        .unwrap_or_else(|error| panic!("`{query}` failed: {error}"))
        .counters
}

/// The create operations of `EXPLAIN query` (`Create (a:N), ...`).
fn create_operators(db: &GrafeoDB, query: &str) -> usize {
    let result = db.execute_cypher(&format!("EXPLAIN {query}")).unwrap();
    let Value::String(text) = &result.rows()[0][0] else {
        panic!("`EXPLAIN {query}` returned {:?}", result.rows());
    };
    text.lines()
        .filter(|line| line.trim_start().starts_with("Create "))
        .count()
}

/// `size` CREATE clauses of one node each, the first one bound to `a`.
fn chained_creates(size: usize) -> String {
    (0..size)
        .map(|i| match i {
            0 => "CREATE (a:N {i: 0})".to_string(),
            i => format!("CREATE (:N {{i: {i}}})"),
        })
        .collect::<Vec<_>>()
        .join(" ")
}

#[test]
fn two_hundred_chained_creates_run() {
    let db = GrafeoDB::new_in_memory();
    let query = format!("{} RETURN a.i", chained_creates(200));
    assert_eq!(count(&db, &query), 0);
    assert_eq!(count(&db, "MATCH (n:N) RETURN count(n)"), 200);
    assert_eq!(
        count(&db, "MATCH (n:N) RETURN sum(n.i)"),
        (0..200).sum::<i64>()
    );
    assert_eq!(create_operators(&db, &query), 1, "one create operation");
}

#[test]
fn one_create_of_two_hundred_patterns_runs() {
    let db = GrafeoDB::new_in_memory();
    let patterns = (0..200)
        .map(|i| format!("(:N {{i: {i}}})"))
        .collect::<Vec<_>>()
        .join(", ");
    let query = format!("CREATE {patterns}");
    let c = counters(&db, &query);
    assert_eq!((c.nodes_created, c.properties_set), (200, 200));
    assert_eq!(count(&db, "MATCH (n:N) RETURN count(n)"), 200);
    assert_eq!(create_operators(&db, &query), 1, "one create operation");
}

#[test]
fn a_later_create_reads_an_earlier_ones_property() {
    let db = GrafeoDB::new_in_memory();
    db.execute_cypher("CREATE (a {x: 1}) CREATE (b {y: a.x})")
        .unwrap();
    assert_eq!(count(&db, "MATCH (b) WHERE b.y IS NOT NULL RETURN b.y"), 1);
}

#[test]
fn a_chain_creates_for_each_row_in_order() {
    let db = GrafeoDB::new_in_memory();
    db.execute_cypher(
        "UNWIND [1, 2, 3] AS i CREATE (a:A {i: i}) CREATE (b:B {i: a.i * 10}) \
         CREATE (a)-[:R]->(b)",
    )
    .unwrap();
    let rows = db
        .execute_cypher("MATCH (a:A)-[:R]->(b:B) RETURN a.i, b.i ORDER BY a.i")
        .unwrap()
        .rows()
        .to_vec();
    assert_eq!(
        rows,
        vec![
            vec![Value::Int64(1), Value::Int64(10)],
            vec![Value::Int64(2), Value::Int64(20)],
            vec![Value::Int64(3), Value::Int64(30)],
        ]
    );
}

#[test]
fn chained_creates_count_as_one_create() {
    let chained = "CREATE (a:A {x: 1}) CREATE (b:B {y: 2}) CREATE (a)-[:R {z: 3}]->(b)";
    let single = "CREATE (a:A {x: 1}), (b:B {y: 2}), (a)-[:R {z: 3}]->(b)";
    let of = |query: &str| {
        let c = counters(&GrafeoDB::new_in_memory(), query);
        (
            c.nodes_created,
            c.edges_created,
            c.properties_set,
            c.labels_added,
        )
    };
    assert_eq!(of(chained), (2, 1, 3, 2));
    assert_eq!(of(chained), of(single));
}

#[test]
fn a_create_reading_the_graph_is_not_joined() {
    // The second CREATE reads the :A nodes the first creates, so it stays an
    // operator of its own, as two clauses one after the other are. (A pattern
    // comprehension in a property map is refused before it gets here.)
    for read in [
        "COUNT { (h)-->(:A) }",
        "CASE WHEN EXISTS { (h)-->(:A) } THEN 1 ELSE 0 END",
    ] {
        let db = GrafeoDB::new_in_memory();
        db.execute_cypher("CREATE (:H)").unwrap();
        let query = format!(
            "MATCH (h:H) UNWIND [1, 2] AS i CREATE (h)-[:R]->(:A {{i: i}}) CREATE (:B {{n: {read}}})"
        );
        assert_eq!(create_operators(&db, &query), 2, "`{query}`");
        db.execute_cypher(&query)
            .unwrap_or_else(|error| panic!("`{query}` failed: {error}"));
        assert_eq!(count(&db, "MATCH (b:B) RETURN count(b)"), 2, "`{query}`");
    }
}

#[test]
fn a_create_after_one_reading_the_graph_is_not_joined() {
    let db = GrafeoDB::new_in_memory();
    let query = "CREATE (a {n: COUNT { MATCH (x)-->() }}) CREATE (b) CREATE (c)";
    assert_eq!(create_operators(&db, query), 2);
}

#[test]
fn a_clause_between_creates_stops_the_chain() {
    let db = GrafeoDB::new_in_memory();
    assert_eq!(
        create_operators(&db, "CREATE (a) WITH a CREATE (b) CREATE (c)"),
        2
    );
    assert_eq!(
        create_operators(
            &db,
            "CREATE (a) CALL { CREATE (x) CREATE (y) } CREATE (b) CREATE (c)"
        ),
        3
    );
}

#[test]
fn a_pattern_failing_in_the_middle_writes_nothing() {
    let db = GrafeoDB::new_in_memory();
    // The second row fails in the second CREATE, after the first row and the
    // second row's first CREATE wrote.
    let failed = db
        .execute_cypher("UNWIND [1, 0] AS i CREATE (:A {i: i}) CREATE (:B {x: 1 / i}) CREATE (:C)");
    assert!(failed.is_err(), "division by zero fails: {failed:?}");
    assert_eq!(count(&db, "MATCH (n) RETURN count(n)"), 0);
}
