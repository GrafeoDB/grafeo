//! A subquery or volatile conjunct in WHERE stays where it is written: the
//! optimizer does not push it below the expand or projection that binds what
//! it reads, and does not change how often it runs.
//!
//! ```bash
//! cargo test -p grafeo-engine --all-features --test subquery_filters
//! ```

#![cfg(all(feature = "lpg", feature = "gql"))]

use grafeo_common::types::Value;
use grafeo_engine::GrafeoDB;

/// Alix knows Gus (25, lives in Amsterdam) and Vincent (30, lives nowhere).
fn db() -> GrafeoDB {
    let db = GrafeoDB::new_in_memory();
    db.execute(
        "INSERT (alix:Person {name: 'Alix', age: 40}), \
         (gus:Person {name: 'Gus', age: 25}), \
         (vincent:Person {name: 'Vincent', age: 30}), \
         (amsterdam:City {name: 'Amsterdam'}), \
         (alix)-[:KNOWS]->(gus), (alix)-[:KNOWS]->(vincent), \
         (gus)-[:LIVES_IN]->(amsterdam)",
    )
    .unwrap();
    db
}

fn names(db: &GrafeoDB, query: &str) -> Vec<Value> {
    let mut names: Vec<Value> = db
        .execute(query)
        .unwrap()
        .rows()
        .iter()
        .map(|row| row[0].clone())
        .collect();
    names.sort_by_key(|value| format!("{value:?}"));
    names
}

const FRIENDS: &str = "MATCH (a:Person {name: 'Alix'})-[:KNOWS]->(b)";
const LIVES: &str = "MATCH (b)-[:LIVES_IN]->(:City)";

#[test]
fn a_subquery_conjunct_sees_the_expanded_node() {
    let db = db();
    for (condition, expected) in [
        (format!("EXISTS {{ {LIVES} }} AND b.age > 0"), "Gus"),
        (format!("NOT EXISTS {{ {LIVES} }} AND b.age > 0"), "Vincent"),
        (format!("COUNT {{ {LIVES} }} > 0 AND b.age > 0"), "Gus"),
        (format!("b.age > 0 AND EXISTS {{ {LIVES} }}"), "Gus"),
        // The subquery alone was pushed below the expand as well.
        (format!("EXISTS {{ {LIVES} }}"), "Gus"),
    ] {
        let query = format!("{FRIENDS} WHERE {condition} RETURN b.name");
        assert_eq!(names(&db, &query), [Value::from(expected)], "{query}");
    }
}

#[test]
fn a_subquery_conjunct_sees_a_projected_alias() {
    let db = db();
    let query = "MATCH (p:Person) WITH p AS friend \
                 WHERE EXISTS { MATCH (friend)-[:LIVES_IN]->(:City) } RETURN friend.name";
    assert_eq!(names(&db, query), [Value::from("Gus")]);
}

#[cfg(feature = "cypher")]
#[test]
fn cypher_subquery_conjuncts_see_the_expanded_node() {
    let db = db();
    let session = db.session();
    for (condition, expected) in [
        (format!("EXISTS {{ {LIVES} }} AND b.age > 0"), "Gus"),
        (format!("NOT EXISTS {{ {LIVES} }} AND b.age > 0"), "Vincent"),
        (format!("COUNT {{ {LIVES} }} > 0 AND b.age > 0"), "Gus"),
    ] {
        let query = format!("{FRIENDS} WHERE {condition} RETURN b.name");
        let rows = session.execute_cypher(&query).unwrap();
        let found: Vec<Value> = rows.rows().iter().map(|row| row[0].clone()).collect();
        assert_eq!(found, [Value::from(expected)], "{query}");
    }
}

/// A volatile conjunct runs once per row of the pattern it is written after,
/// not once per row of an earlier one: it stays above the expand.
#[test]
fn a_volatile_conjunct_stays_above_the_expand() {
    let db = db();
    let plan = db
        .execute(&format!(
            "EXPLAIN {FRIENDS} WHERE rand() < 2.0 AND a.age > 0 RETURN b.name"
        ))
        .unwrap();
    let plan: Vec<String> = plan
        .rows()
        .iter()
        .flat_map(|row| match &row[0] {
            Value::String(text) => text.lines().map(str::to_string).collect::<Vec<_>>(),
            other => panic!("expected plan text, got {other:?}"),
        })
        .collect();
    let line = |needle: &str| {
        plan.iter()
            .position(|line| line.contains(needle))
            .unwrap_or_else(|| panic!("no {needle} in {plan:#?}"))
    };
    assert!(line("rand") < line("Expand"), "{plan:#?}");
}
