//! A SET, REMOVE or DELETE whose target is null (a node or relationship an
//! OPTIONAL MATCH did not find) is a no-op for that row: the statement
//! succeeds, changes nothing for the row, counts nothing, and passes the row
//! on (ISO/IEC 39075 13.3 GR 8, 13.4 GR 4, 13.5 GR 4; openCypher TCK 2024.3
//! Set1 [8], Set3 [8], Set4 [5], Set5 [1], Remove1 [5] [6], Remove2 [5],
//! Delete1 [4] [5] [6], Delete2 [2] [4], Delete3 [2]). It failed with
//! GRAFEO-V006 "expected a node, found null".
//!
//! ```bash
//! cargo test -p grafeo-engine --all-features --test main null_target_writes::
//! ```

#![cfg(all(feature = "gql", feature = "cypher"))]

use grafeo_common::types::Value;
use grafeo_engine::GrafeoDB;
use grafeo_engine::database::{QueryResult, WriteCounters};

fn run(db: &GrafeoDB, language: &str, query: &str) -> QueryResult {
    match language {
        "gql" => db.execute(query),
        "cypher" => db.execute_cypher(query),
        other => panic!("unknown language {other}"),
    }
    .unwrap_or_else(|error| panic!("{language}: {query}: {error}"))
}

fn counted(c: &WriteCounters) -> [u64; 7] {
    [
        c.nodes_created,
        c.nodes_deleted,
        c.edges_created,
        c.edges_deleted,
        c.properties_set,
        c.labels_added,
        c.labels_removed,
    ]
}

/// `(:A {k: 1})-[:R]->(:B)` and `(:A {k: 2})`.
fn database() -> GrafeoDB {
    let db = GrafeoDB::new_in_memory();
    db.execute_cypher("CREATE (:A {k: 1})-[:R]->(:B), (:A {k: 2})")
        .unwrap();
    db
}

/// What the store holds, to compare before and after a no-op.
fn contents(db: &GrafeoDB) -> Vec<Vec<Value>> {
    db.execute_cypher(
        "MATCH (n) OPTIONAL MATCH (n)-[r]->() \
         RETURN labels(n) AS l, properties(n) AS p, type(r) AS t ORDER BY n.k, t",
    )
    .unwrap()
    .rows()
    .to_vec()
}

/// Each write on a null target, in each language that has its syntax: the
/// statement succeeds, its row comes back with the null, nothing is counted
/// and the store is unchanged.
#[test]
fn a_write_on_a_null_target_is_a_no_op() {
    let both = &["gql", "cypher"][..];
    let cypher = &["cypher"][..];
    for (languages, query, returns_row) in [
        (
            both,
            "OPTIONAL MATCH (a:Missing) SET a.num = 42 RETURN a",
            true,
        ),
        (both, "OPTIONAL MATCH (a:Missing) SET a:L RETURN a", true),
        (
            both,
            "OPTIONAL MATCH (a:Missing) SET a = {num: 42} RETURN a",
            true,
        ),
        (
            cypher,
            "OPTIONAL MATCH (a:Missing) SET a += {num: 42} RETURN a",
            true,
        ),
        (
            both,
            "OPTIONAL MATCH (a:Missing) REMOVE a.num RETURN a",
            true,
        ),
        (both, "OPTIONAL MATCH (a:Missing) REMOVE a:L RETURN a", true),
        (both, "OPTIONAL MATCH (a:Missing) DELETE a RETURN a", true),
        (
            both,
            "OPTIONAL MATCH (a:Missing) DETACH DELETE a RETURN a",
            true,
        ),
        (
            cypher,
            "OPTIONAL MATCH ()-[r:Missing]->() DELETE r RETURN r",
            true,
        ),
        (
            cypher,
            "OPTIONAL MATCH p = (:Missing)-->() DETACH DELETE p",
            false,
        ),
    ] {
        for language in languages {
            let db = database();
            let before = contents(&db);
            let result = run(&db, language, query);
            assert_eq!(counted(&result.counters), [0; 7], "{language}: {query}");
            if returns_row {
                assert_eq!(result.rows(), &[vec![Value::Null]], "{language}: {query}");
            }
            assert_eq!(contents(&db), before, "{language}: {query}");
        }
    }
}

/// Rows with a null target and rows with a real one in one statement: the
/// real ones are written, the null ones pass through.
#[test]
fn null_and_real_targets_in_one_statement() {
    for language in ["gql", "cypher"] {
        let db = database();
        let result = run(
            &db,
            language,
            "MATCH (a:A) OPTIONAL MATCH (a)-[:R]->(b) SET b.seen = true, a.seen = true \
             RETURN a.k AS k, b IS NULL AS missing ORDER BY k",
        );
        assert_eq!(
            result.rows(),
            &[
                vec![Value::Int64(1), Value::Bool(false)],
                vec![Value::Int64(2), Value::Bool(true)],
            ],
            "{language}"
        );
        assert_eq!(result.counters.properties_set, 3, "{language}");
        let seen = run(&db, "cypher", "MATCH (n {seen: true}) RETURN count(n) AS c");
        assert_eq!(seen.rows(), &[vec![Value::Int64(3)]], "{language}");

        let result = run(
            &db,
            language,
            "MATCH (a:A) OPTIONAL MATCH (a)-[r:R]->() DELETE r RETURN a.k AS k ORDER BY k",
        );
        assert_eq!(
            result.rows(),
            &[vec![Value::Int64(1)], vec![Value::Int64(2)]],
            "{language}"
        );
        assert_eq!(result.counters.edges_deleted, 1, "{language}");
    }
}

/// A null target beside a real one in the same SET writes the real one.
#[test]
fn a_null_target_beside_a_real_one() {
    for language in ["gql", "cypher"] {
        let db = database();
        let result = run(
            &db,
            language,
            "MATCH (a:A {k: 1}) OPTIONAL MATCH (x:Missing) SET x.v = 1, a.v = 1 RETURN a.v AS v",
        );
        assert_eq!(result.rows(), &[vec![Value::Int64(1)]], "{language}");
        assert_eq!(result.counters.properties_set, 1, "{language}");
    }
}

/// A write that needs its target still fails on a null one: the endpoint of a
/// relationship a CREATE makes.
#[test]
fn a_create_from_a_null_endpoint_still_fails() {
    let db = database();
    let error = db
        .execute_cypher("OPTIONAL MATCH (a:Missing) CREATE (a)-[:R]->(:C)")
        .expect_err("a relationship needs both endpoints");
    assert!(error.to_string().contains("null"), "{error}");
}

/// A node deleted twice in one statement is deleted once: the second
/// reference is invalidated, which ISO 13.5 GR 4 reads as null.
#[test]
fn a_node_deleted_twice_in_one_statement_is_deleted_once() {
    for (language, query) in [
        ("cypher", "MATCH (a:A {k: 2}) DELETE a, a"),
        ("gql", "MATCH (a:A {k: 2}) DELETE a, a"),
        ("cypher", "MATCH (a:A {k: 2}), (b:A {k: 2}) DELETE a, b"),
        ("cypher", "MATCH (a:A {k: 2}) DELETE a WITH a DELETE a"),
    ] {
        let db = database();
        let result = run(&db, language, query);
        assert_eq!(result.counters.nodes_deleted, 1, "{language}: {query}");
    }
}

/// A map assignment of null (`SET a = null`, `SET a += null`, or a null
/// parameter) changes nothing and counts nothing: `=` with null does not
/// remove the properties. It pins that the write was already a no-op.
#[test]
fn a_null_map_assignment_changes_and_counts_nothing() {
    let params = || std::collections::HashMap::from([("m".to_string(), Value::Null)]);
    for (language, query) in [
        ("gql", "MATCH (a:A {k: 1}) SET a = null"),
        ("cypher", "MATCH (a:A {k: 1}) SET a = null"),
        ("cypher", "MATCH (a:A {k: 1}) SET a += null"),
        ("gql", "MATCH (a:A {k: 1}) SET a = $m"),
        ("gql", "MATCH (a:A {k: 1}) SET a += $m"),
        ("cypher", "MATCH (a:A {k: 1}) SET a = $m"),
        ("cypher", "MATCH (a:A {k: 1}) SET a += $m"),
        ("cypher", "MATCH ()-[r:R]->() SET r = $m"),
    ] {
        let db = database();
        let before = contents(&db);
        let result = match language {
            "gql" => db.execute_with_params(query, params()),
            _ => db.execute_cypher_with_params(query, params()),
        }
        .unwrap_or_else(|error| panic!("{language}: {query}: {error}"));
        assert_eq!(counted(&result.counters), [0; 7], "{language}: {query}");
        assert_eq!(contents(&db), before, "{language}: {query}");
    }
}

/// A map assignment of a value that is neither a map nor null (`SET a += 5`,
/// a string, a list, a boolean, written or as a parameter, also in a MERGE)
/// raises invalid value type (22G12) on a node and a relationship, in both
/// languages, and changes nothing. It succeeded and set nothing.
#[test]
fn a_map_assignment_of_a_non_map_raises() {
    let params = || std::collections::HashMap::from([("m".to_string(), Value::Int64(5))]);
    for (language, query) in [
        ("cypher", "MATCH (a:A {k: 1}) SET a += 5"),
        ("cypher", "MATCH (a:A {k: 1}) SET a = 'x'"),
        ("cypher", "MATCH (a:A {k: 1}) SET a += [1, 2]"),
        ("cypher", "MATCH (a:A {k: 1}) SET a = true"),
        ("cypher", "MATCH (a:A {k: 1}) SET a += $m"),
        ("cypher", "MATCH (a:A {k: 1}) SET a = $m"),
        ("cypher", "MATCH ()-[r:R]->() SET r += $m"),
        ("cypher", "MATCH ()-[r:R]->() SET r = 'x'"),
        ("cypher", "MERGE (a:A {k: 1}) ON MATCH SET a += $m"),
        ("gql", "MATCH (a:A {k: 1}) SET a = 5"),
        ("gql", "MATCH (a:A {k: 1}) SET a = $m"),
        ("gql", "MATCH (a:A {k: 1}) SET a += $m"),
        ("gql", "MATCH ()-[r:R]->() SET r = $m"),
    ] {
        let db = database();
        let before = contents(&db);
        let error = match language {
            "gql" => db.execute_with_params(query, params()),
            _ => db.execute_cypher_with_params(query, params()),
        }
        .expect_err(query);
        let status = grafeo_common::utils::gqlstatus::GqlStatus::from(&error);
        assert_eq!(status.as_str(), "22G12", "{language}: {query}: {error}");
        assert!(
            error
                .to_string()
                .contains("SET of all properties takes a map or null, not a value of type"),
            "{language}: {query}: {error}"
        );
        assert_eq!(contents(&db), before, "{language}: {query}");
    }
}
