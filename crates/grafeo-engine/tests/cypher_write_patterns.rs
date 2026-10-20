//! Cypher patterns and operators as openCypher reads them: a relationship
//! with arrows both ways is either direction, a created relationship needs
//! one type and a direction, and the string, list and null predicates bind
//! tighter than the comparisons.
//!
//! ```bash
//! cargo test -p grafeo-engine --all-features --test cypher_write_patterns
//! ```

#![cfg(all(feature = "lpg", feature = "gql", feature = "cypher"))]

use grafeo_common::types::Value;
use grafeo_engine::GrafeoDB;

fn cypher(db: &GrafeoDB, query: &str) -> Vec<Vec<Value>> {
    db.execute_cypher(query)
        .unwrap_or_else(|e| panic!("`{query}`: {e}"))
        .rows()
        .to_vec()
}

fn int(value: i64) -> Value {
    Value::Int64(value)
}

/// `(a)<-[:T]->(b)` and `(a)<-->(b)` point both ways: either direction, as
/// `(a)-[:T]-(b)`. They matched outgoing only.
#[test]
fn arrows_both_ways_match_either_direction() {
    let db = GrafeoDB::new_in_memory();
    db.execute("INSERT (:A {n: 1})-[:T]->(:B {n: 2})").unwrap();
    for query in [
        "MATCH (a)<-[:T]->(b) RETURN a.n, b.n ORDER BY a.n",
        "MATCH (a)<-[]->(b) RETURN a.n, b.n ORDER BY a.n",
        "MATCH (a)<-->(b) RETURN a.n, b.n ORDER BY a.n",
        "MATCH (a)-[:T]-(b) RETURN a.n, b.n ORDER BY a.n",
    ] {
        assert_eq!(
            cypher(&db, query),
            [[int(1), int(2)], [int(2), int(1)]],
            "`{query}`"
        );
    }
}

/// A created relationship needs exactly one type and a direction: without a
/// type it got `RELATED`, and with arrows both ways it was created outgoing.
#[test]
fn a_created_relationship_needs_a_type_and_a_direction() {
    for (query, message) in [
        ("CREATE (a)-[]->(b)", "type"),
        ("CREATE (a)-->(b)", "type"),
        ("CREATE (a)<-[:T]->(b)", "direction"),
        ("CREATE (a)-[:T]-(b)", "direction"),
    ] {
        let db = GrafeoDB::new_in_memory();
        let error = db.execute_cypher(query).expect_err(query).to_string();
        assert!(error.contains(message), "`{query}`: {error}");
        assert_eq!(
            cypher(&db, "MATCH (n) RETURN count(n)"),
            [[int(0)]],
            "`{query}` wrote nothing"
        );
    }
}

/// `IN`, `STARTS WITH`, `ENDS WITH`, `CONTAINS`, `=~` and `IS NULL` bind
/// tighter than `=` and `<` (openCypher 9): `1 = 2 IN [false]` is
/// `1 = (2 IN [false])`, false; read left to right it was `(1 = 2) IN
/// [false]`, true.
#[test]
fn predicates_bind_tighter_than_comparisons() {
    let db = GrafeoDB::new_in_memory();
    for (query, expected) in [
        ("RETURN 1 = 2 IN [false] AS v", Value::Bool(false)),
        ("RETURN 2 IN [2] = true AS v", Value::Bool(true)),
        (
            "RETURN false = 'ab' STARTS WITH 'b' AS v",
            Value::Bool(true),
        ),
        ("RETURN false = 'ab' ENDS WITH 'a' AS v", Value::Bool(true)),
        ("RETURN false = 'ab' CONTAINS 'c' AS v", Value::Bool(true)),
        ("RETURN false = 'ab' =~ 'c' AS v", Value::Bool(true)),
        ("RETURN false = null IS NULL AS v", Value::Bool(false)),
        // A chain of comparisons stays one: `1 < 2 AND 2 = true`.
        ("RETURN 1 < 2 = true AS v", Value::Bool(false)),
    ] {
        assert_eq!(cypher(&db, query), [[expected]], "`{query}`");
    }
}
