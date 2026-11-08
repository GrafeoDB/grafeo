//! A MERGE whose pattern holds a null entry matches what it inserted: `{k: null}` in a MERGE
//! matches an element without `k` (or with a null `k`), as an insert stores no null, so running
//! the same MERGE again finds the element instead of inserting another. This holds for a null
//! literal, a null parameter and a computed null, on nodes and on edges, in a path MERGE in GQL
//! and in Cypher, and in a Cypher node MERGE that sets as a clause. A MATCH keeps the equality
//! of its property map: `MATCH (n {k: null})` finds nothing.
//!
//! ```bash
//! cargo test -p grafeo-engine --all-features --test main merge_null_entries::
//! ```

#![cfg(all(feature = "gql", feature = "cypher"))]

use std::collections::HashMap;

use grafeo_common::types::Value;
use grafeo_engine::GrafeoDB;

/// The nodes and the edges of `db`.
fn counts(db: &GrafeoDB) -> (i64, i64) {
    let count = |query: &str| match db.execute(query).unwrap().rows()[0][0] {
        Value::Int64(count) => count,
        ref other => panic!("a count, got {other:?}"),
    };
    (
        count("MATCH (n) RETURN count(n)"),
        count("MATCH ()-[e]->() RETURN count(e)"),
    )
}

fn null_param(name: &str) -> HashMap<String, Value> {
    HashMap::from([(name.to_string(), Value::Null)])
}

/// How one MERGE runs.
#[derive(Clone, Copy, Debug)]
enum Run {
    Gql,
    GqlWithNull,
    Cypher,
    CypherWithNull,
}

fn run(db: &GrafeoDB, how: Run, query: &str) {
    match how {
        Run::Gql => db.execute(query),
        Run::GqlWithNull => db.execute_with_params(query, null_param("c")),
        Run::Cypher => db.execute_cypher(query),
        Run::CypherWithNull => db.execute_cypher_with_params(query, null_param("c")),
    }
    .unwrap_or_else(|error| panic!("{how:?} {query}: {error}"));
}

/// Each MERGE, run twice: the second run finds what the first inserted, so the counts after it
/// equal the counts after the first. `setup` runs once before.
#[test]
fn a_merge_with_a_null_entry_finds_what_it_inserted() {
    let path_null =
        "MERGE (a:City {name: 'Amsterdam', code: null})-[:LOCATED_IN]->(c:Country {name: 'NL'})";
    let path_param =
        "MERGE (a:City {name: 'Amsterdam', code: $c})-[:LOCATED_IN]->(c:Country {name: 'NL'})";
    let path_computed = "MERGE (a:City {name: 'Amsterdam', code: CASE WHEN 1 = 2 THEN 'x' END})-[:LOCATED_IN]->(c:Country {name: 'NL'})";
    let edge_null = "MATCH (a:City {name: 'Utrecht'}), (b:Country {name: 'NL'}) \
                     MERGE (a)-[:LOCATED_IN {since: null}]->(b)";
    let edge_param = "MATCH (a:City {name: 'Utrecht'}), (b:Country {name: 'NL'}) \
                      MERGE (a)-[:LOCATED_IN {since: $c}]->(b)";
    let bound = "INSERT (:City {name: 'Utrecht'}), (:Country {name: 'NL'})";
    let cypher_bound = "CREATE (:City {name: 'Utrecht'}), (:Country {name: 'NL'})";
    let node_as_a_clause = "MERGE (n:City {name: 'Amsterdam', code: $c}) ON CREATE SET n = {name: 'Amsterdam', made: true}";
    let cases: [(Run, Option<&str>, &str); 12] = [
        (Run::Gql, None, path_null),
        (Run::GqlWithNull, None, path_param),
        (Run::Gql, None, path_computed),
        (Run::Gql, Some(bound), edge_null),
        (Run::GqlWithNull, Some(bound), edge_param),
        (Run::Cypher, None, path_null),
        (Run::CypherWithNull, None, path_param),
        (Run::Cypher, None, path_computed),
        (Run::Cypher, Some(cypher_bound), edge_null),
        (Run::CypherWithNull, Some(cypher_bound), edge_param),
        (Run::CypherWithNull, None, node_as_a_clause),
        (
            Run::GqlWithNull,
            None,
            "MERGE (n:City {name: 'Amsterdam', code: $c})",
        ),
    ];
    let mut wrong = Vec::new();
    for (how, setup, query) in cases {
        let db = GrafeoDB::new_in_memory();
        if let Some(setup) = setup {
            match how {
                Run::Cypher | Run::CypherWithNull => db.execute_cypher(setup).unwrap(),
                Run::Gql | Run::GqlWithNull => db.execute(setup).unwrap(),
            };
        }
        run(&db, how, query);
        let first = counts(&db);
        run(&db, how, query);
        let second = counts(&db);
        if first != second {
            wrong.push(format!(
                "{how:?} {query}: {first:?} after one run, {second:?} after two"
            ));
        }
    }
    assert!(wrong.is_empty(), "{}", wrong.join("\n"));
}

/// The match branch takes an element whose entry is null or absent, and not one whose entry
/// holds a value: a MERGE with `code: null` next to a city with a code inserts its own city.
#[test]
fn a_null_entry_does_not_match_an_element_with_a_value() {
    for (how, query) in [
        (
            Run::Gql,
            "MERGE (a:City {name: 'Amsterdam', code: null})-[:LOCATED_IN]->(c:Country {name: 'NL'})",
        ),
        (
            Run::CypherWithNull,
            "MERGE (a:City {name: 'Amsterdam', code: $c})-[:LOCATED_IN]->(c:Country {name: 'NL'})",
        ),
    ] {
        let db = GrafeoDB::new_in_memory();
        db.execute("INSERT (:City {name: 'Amsterdam', code: 'AMS'})-[:LOCATED_IN]->(:Country {name: 'NL'})")
            .unwrap();
        run(&db, how, query);
        assert_eq!(counts(&db), (4, 2), "{how:?} {query}");
        let with_code = db
            .execute("MATCH (a:City) WHERE a.code IS NOT NULL RETURN count(a)")
            .unwrap();
        assert_eq!(with_code.rows()[0][0], Value::Int64(1), "{how:?}");
    }
}

/// A MATCH keeps the equality of its property map: a null entry finds nothing, there and in
/// the rows a MERGE matched before.
#[test]
fn a_match_with_a_null_entry_still_finds_nothing() {
    let db = GrafeoDB::new_in_memory();
    db.execute(
        "MERGE (a:City {name: 'Amsterdam', code: null})-[:LOCATED_IN]->(c:Country {name: 'NL'})",
    )
    .unwrap();
    let found = db
        .execute("MATCH (a:City {code: null}) RETURN count(a)")
        .unwrap();
    assert_eq!(found.rows()[0][0], Value::Int64(0));
}
