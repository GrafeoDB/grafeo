//! `SET x = e` and `SET x += e`, with `e` a node or a relationship, copy `e`'s
//! properties onto `x` (`=` replaces `x`'s properties, `+=` adds to them), as
//! openCypher has it (TCK 2024.3 Merge6 [6], Merge7 [4]). The node or
//! relationship reached the write as its id, which is no map, so nothing was
//! set.
//!
//! ```bash
//! cargo test -p grafeo-engine --all-features --test main set_properties_from_an_entity::
//! ```

#![cfg(all(feature = "gql", feature = "cypher"))]

use grafeo_common::types::Value;
use grafeo_engine::GrafeoDB;

/// `(:A {name: 'A', n: 1})-[:R {w: 7}]->(:B {name: 'B', keep: true})`.
fn database() -> GrafeoDB {
    let db = GrafeoDB::new_in_memory();
    db.execute_cypher("CREATE (:A {name: 'A', n: 1})-[:R {w: 7}]->(:B {name: 'B', keep: true})")
        .unwrap();
    db
}

/// The properties of what `query` returns as `x`, as `key->value` texts, sorted.
fn properties(db: &GrafeoDB, query: &str) -> Vec<String> {
    let rows = db
        .execute_cypher(query)
        .unwrap_or_else(|error| panic!("{query}: {error}"))
        .rows()
        .to_vec();
    let [row] = rows.as_slice() else {
        panic!("{query}: {rows:?}");
    };
    let Value::Map(map) = &row[0] else {
        panic!("{query}: {row:?}");
    };
    let mut found: Vec<String> = map
        .iter()
        .map(|(key, value)| format!("{}->{value}", key.as_str()))
        .collect();
    found.sort();
    found
}

fn run(db: &GrafeoDB, language: &str, query: &str) {
    match language {
        "gql" => db.execute(query),
        _ => db.execute_cypher(query),
    }
    .unwrap_or_else(|error| panic!("{language}: {query}: {error}"));
}

/// `=` replaces the target's properties with the source's, `+=` adds them, in
/// every direction between nodes and relationships.
#[test]
fn set_copies_the_properties_of_a_node_or_relationship() {
    for language in ["gql", "cypher"] {
        for (write, read, expected) in [
            (
                "MATCH (a:A)-[r:R]->() SET r = a",
                "MATCH ()-[r:R]->() RETURN properties(r) AS x",
                vec!["n->1", r#"name->"A""#],
            ),
            (
                "MATCH (a:A)-[r:R]->() SET r += a",
                "MATCH ()-[r:R]->() RETURN properties(r) AS x",
                vec!["n->1", r#"name->"A""#, "w->7"],
            ),
            (
                "MATCH (a:A)-[r:R]->(b:B) SET b = r",
                "MATCH (b:B) RETURN properties(b) AS x",
                vec!["w->7"],
            ),
            (
                "MATCH (a:A)-[r:R]->(b:B) SET b += a",
                "MATCH (b:B) RETURN properties(b) AS x",
                vec!["keep->true", "n->1", r#"name->"A""#],
            ),
        ] {
            let db = database();
            run(&db, language, write);
            assert_eq!(properties(&db, read), expected, "{language}: {write}");
        }
    }
}

/// The TCK's MERGE forms: `ON CREATE SET r = a` and `ON MATCH SET r = a`.
#[test]
fn merge_sets_copy_the_properties_of_a_node() {
    let db = GrafeoDB::new_in_memory();
    db.execute_cypher("CREATE (:A {name: 'A'}), (:B {name: 'B'})")
        .unwrap();
    let result = db
        .execute_cypher(
            "MATCH (a {name: 'A'}), (b {name: 'B'}) MERGE (a)-[r:TYPE]->(b) ON CREATE SET r = a",
        )
        .unwrap();
    assert_eq!(result.counters.properties_set, 1);
    assert_eq!(
        properties(&db, "MATCH ()-[r:TYPE]->() RETURN properties(r) AS x"),
        [r#"name->"A""#]
    );

    let db = GrafeoDB::new_in_memory();
    db.execute_cypher(
        "CREATE (a:A {name: 'A'}), (b:B {name: 'B'}) CREATE (a)-[:TYPE {name: 'bar'}]->(b)",
    )
    .unwrap();
    db.execute_cypher(
        "MATCH (a {name: 'A'}), (b {name: 'B'}) MERGE (a)-[r:TYPE]->(b) ON MATCH SET r = a",
    )
    .unwrap();
    assert_eq!(
        properties(&db, "MATCH ()-[r:TYPE]->() RETURN properties(r) AS x"),
        [r#"name->"A""#]
    );
}

/// A map in a variable is still set as a map.
#[test]
fn a_map_variable_is_still_a_map() {
    let db = database();
    db.execute_cypher("MATCH (a:A) WITH a, {z: 9} AS m SET a = m")
        .unwrap();
    assert_eq!(
        properties(&db, "MATCH (a:A) RETURN properties(a) AS x"),
        ["z->9"]
    );
}
