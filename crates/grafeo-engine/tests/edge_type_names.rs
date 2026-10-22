//! A pattern's edge types match the stored types regardless of ASCII case:
//! the planner resolves the names a pattern gives to the names the store
//! holds, once per plan, and the store matches those exactly. Every operator
//! that walks typed edges goes through it.
//!
//! ```bash
//! cargo test -p grafeo-engine --all-features --test edge_type_names
//! ```

use grafeo_engine::GrafeoDB;

fn graph() -> GrafeoDB {
    let db = GrafeoDB::new_in_memory();
    db.execute(
        "INSERT (:Person {name: 'Alix'})-[:KNOWS]->(:Person {name: 'Gus'})-[:KNOWS]->\
         (:Person {name: 'Vincent'})",
    )
    .unwrap();
    db
}

fn count(db: &GrafeoDB, query: &str) -> String {
    let result = db.execute(query).unwrap();
    result.rows()[0][0].to_string()
}

#[test]
fn an_expand_matches_the_stored_type_in_any_case() {
    let db = graph();
    for query in [
        "MATCH ()-[:KNOWS]->() RETURN count(*)",
        "MATCH ()-[:knows]->() RETURN count(*)",
        "MATCH ()-[:Knows]->() RETURN count(*)",
        "MATCH ()<-[:knows]-() RETURN count(*)",
    ] {
        assert_eq!(count(&db, query), "2", "{query}");
    }
    assert_eq!(count(&db, "MATCH ()-[:LIKES]->() RETURN count(*)"), "0");
}

#[test]
fn a_variable_length_path_matches_in_any_case() {
    let db = graph();
    assert_eq!(
        count(
            &db,
            "MATCH (:Person {name: 'Alix'})-[:knows*1..2]->(p) RETURN count(p)"
        ),
        "2"
    );
}

#[test]
fn a_shortest_path_matches_in_any_case() {
    let db = graph();
    assert_eq!(
        count(
            &db,
            "MATCH p = SHORTEST 1 (:Person {name: 'Alix'})-[:knows]->+(:Person {name: 'Vincent'}) \
             RETURN count(p)"
        ),
        "1"
    );
}

#[test]
fn a_subquery_pattern_matches_in_any_case() {
    let db = graph();
    assert_eq!(
        count(
            &db,
            "MATCH (p:Person) WHERE EXISTS { MATCH (p)-[:knows]->() } RETURN count(p)"
        ),
        "2"
    );
    assert_eq!(
        count(
            &db,
            "MATCH (p:Person {name: 'Gus'}) RETURN COUNT { MATCH (p)-[:Knows]-() }"
        ),
        "2"
    );
}

/// Two stored types that differ only in case both match a name in either
/// case.
#[test]
fn types_that_differ_in_case_both_match() {
    let db = graph();
    db.execute(
        "MATCH (a:Person {name: 'Vincent'}), (b:Person {name: 'Alix'}) INSERT (a)-[:knows]->(b)",
    )
    .unwrap();
    assert_eq!(count(&db, "MATCH ()-[:KNOWS]->() RETURN count(*)"), "3");
    assert_eq!(count(&db, "MATCH ()-[:knows]->() RETURN count(*)"), "3");
}
