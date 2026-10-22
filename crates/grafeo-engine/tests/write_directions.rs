//! A relationship written by Cypher `CREATE` or by `MERGE` (Cypher and GQL)
//! points the way its arrow does: `(a)<-[:T]-(b)` is a relationship from `b`
//! to `a` (openCypher 9, "CREATE" and "MERGE"). A left arrow used to be stored
//! from `a` to `b`, so a later read in the written direction missed it and a
//! read the other way round found it; GQL `INSERT` was right.
//!
//! An undirected relationship is an error in a Cypher `CREATE` (a stored
//! relationship has a direction); in a `MERGE` it matches a relationship
//! either way round, and creates one from left to right when none matches.
//!
//! ```bash
//! cargo test -p grafeo-engine --all-features --test write_directions
//! ```

#![cfg(all(feature = "lpg", feature = "gql", feature = "cypher"))]

use grafeo_common::types::Value;
use grafeo_engine::{GrafeoDB, Session};

#[derive(Debug, Clone, Copy)]
enum Language {
    Gql,
    Cypher,
}

fn run(session: &Session, language: Language, query: &str) -> Vec<Vec<Value>> {
    let result = match language {
        Language::Gql => session.execute(query),
        Language::Cypher => session.execute_cypher(query),
    };
    result
        .unwrap_or_else(|error| panic!("{language:?} `{query}` failed: {error}"))
        .rows()
        .to_vec()
}

fn cypher(session: &Session, query: &str) -> Vec<Vec<Value>> {
    run(session, Language::Cypher, query)
}

fn text(value: &Value) -> String {
    match value {
        Value::String(text) => text.to_string(),
        other => format!("{other:?}"),
    }
}

/// The relationships of type `edge_type` as (source name, target name), read
/// in their stored direction, sorted.
fn edges(session: &Session, edge_type: &str) -> Vec<(String, String)> {
    let query = format!("MATCH (s)-[:{edge_type}]->(t) RETURN s.name, t.name");
    let mut pairs: Vec<(String, String)> = run(session, Language::Gql, &query)
        .iter()
        .map(|row| (text(&row[0]), text(&row[1])))
        .collect();
    pairs.sort();
    pairs
}

fn pairs(expected: &[(&str, &str)]) -> Vec<(String, String)> {
    expected
        .iter()
        .map(|(source, target)| ((*source).to_string(), (*target).to_string()))
        .collect()
}

fn alix_and_gus(session: &Session) {
    run(
        session,
        Language::Gql,
        "INSERT (:Person {name: 'Alix'}), (:Person {name: 'Gus'})",
    );
}

// ---------------------------------------------------------------------------
// Cypher CREATE
// ---------------------------------------------------------------------------

#[test]
fn a_left_arrow_in_create_points_from_the_right_node_to_the_left_one() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    cypher(
        &session,
        "CREATE (:Person {name: 'Alix'})<-[:KNOWS]-(:Person {name: 'Gus'})",
    );
    assert_eq!(edges(&session, "KNOWS"), pairs(&[("Gus", "Alix")]));
}

#[test]
fn a_left_arrow_from_a_bound_node_is_read_back_the_way_it_was_written() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    cypher(&session, "CREATE (:Person {name: 'Alix'})");
    cypher(
        &session,
        "MATCH (a:Person {name: 'Alix'}) CREATE (a)<-[:KNOWS]-(:Person {name: 'Gus'})",
    );
    assert_eq!(
        cypher(
            &session,
            "MATCH (:Person {name: 'Alix'})<-[:KNOWS]-(x) RETURN x.name"
        ),
        [vec![Value::from("Gus")]],
        "the written direction finds the relationship"
    );
    assert_eq!(
        cypher(
            &session,
            "MATCH (:Person {name: 'Alix'})-[:KNOWS]->(x) RETURN count(x)"
        ),
        [vec![Value::Int64(0)]],
        "the other direction does not"
    );
}

#[test]
fn a_left_arrow_to_a_bound_node_points_from_it() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    cypher(&session, "CREATE (:Person {name: 'Gus'})");
    cypher(
        &session,
        "MATCH (g:Person {name: 'Gus'}) CREATE (:Person {name: 'Alix'})<-[:KNOWS]-(g)",
    );
    assert_eq!(edges(&session, "KNOWS"), pairs(&[("Gus", "Alix")]));
}

#[test]
fn each_relationship_of_a_chain_points_its_own_way() {
    for (query, expected) in [
        (
            "CREATE (:Person {name: 'Alix'})-[:KNOWS]->(:Person {name: 'Gus'})<-[:KNOWS]-(:Person {name: 'Vincent'})",
            [("Alix", "Gus"), ("Vincent", "Gus")],
        ),
        (
            "CREATE (:Person {name: 'Alix'})<-[:KNOWS]-(:Person {name: 'Gus'})-[:KNOWS]->(:Person {name: 'Vincent'})",
            [("Gus", "Alix"), ("Gus", "Vincent")],
        ),
        (
            "CREATE (:Person {name: 'Alix'})<-[:KNOWS]-(:Person {name: 'Gus'})<-[:KNOWS]-(:Person {name: 'Vincent'})",
            [("Gus", "Alix"), ("Vincent", "Gus")],
        ),
    ] {
        let db = GrafeoDB::new_in_memory();
        let session = db.session();
        cypher(&session, query);
        assert_eq!(edges(&session, "KNOWS"), pairs(&expected), "{query}");
    }
}

#[test]
fn a_left_arrow_keeps_its_variable_and_properties() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    assert_eq!(
        cypher(
            &session,
            "CREATE (:Person {name: 'Alix'})<-[r:KNOWS {since: 2019}]-(:Person {name: 'Gus'}) \
             RETURN type(r), r.since"
        ),
        [vec![Value::from("KNOWS"), Value::Int64(2019)]]
    );
    assert_eq!(
        cypher(
            &session,
            "MATCH (s)-[r:KNOWS]->(t) RETURN s.name, t.name, r.since"
        ),
        [vec![
            Value::from("Gus"),
            Value::from("Alix"),
            Value::Int64(2019)
        ]]
    );
}

#[test]
fn a_left_arrow_between_nodes_of_earlier_patterns_of_the_create() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    cypher(
        &session,
        "CREATE (a:Person {name: 'Alix'}), (b:Person {name: 'Gus'}), (a)<-[:KNOWS]-(b)",
    );
    assert_eq!(edges(&session, "KNOWS"), pairs(&[("Gus", "Alix")]));
}

#[test]
fn an_undirected_relationship_in_create_is_an_error_that_writes_nothing() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    let error = session
        .execute_cypher("CREATE (:Person {name: 'Alix'})-[:KNOWS]-(:Person {name: 'Gus'})")
        .expect_err("a relationship without a direction cannot be created");
    assert!(
        error.to_string().contains("direction"),
        "the error names the missing direction: {error}"
    );
    assert_eq!(
        cypher(&session, "MATCH (n) RETURN count(n)"),
        [vec![Value::Int64(0)]],
        "the failed CREATE wrote no node"
    );
}

#[test]
fn gql_insert_with_a_left_arrow_points_the_same_way() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    run(
        &session,
        Language::Gql,
        "INSERT (:Person {name: 'Alix'})<-[:KNOWS]-(:Person {name: 'Gus'})",
    );
    assert_eq!(edges(&session, "KNOWS"), pairs(&[("Gus", "Alix")]));
}

// ---------------------------------------------------------------------------
// MERGE (Cypher and GQL)
// ---------------------------------------------------------------------------

const LANGUAGES: [Language; 2] = [Language::Gql, Language::Cypher];

#[test]
fn a_left_arrow_in_merge_creates_the_relationship_from_the_right_node() {
    for language in LANGUAGES {
        let db = GrafeoDB::new_in_memory();
        let session = db.session();
        alix_and_gus(&session);
        run(
            &session,
            language,
            "MATCH (a:Person {name: 'Alix'}), (b:Person {name: 'Gus'}) MERGE (a)<-[:KNOWS]-(b)",
        );
        assert_eq!(
            edges(&session, "KNOWS"),
            pairs(&[("Gus", "Alix")]),
            "{language:?}"
        );
    }
}

#[test]
fn a_left_arrow_in_merge_matches_only_a_relationship_pointing_that_way() {
    for language in LANGUAGES {
        // Gus to Alix exists: the MERGE matches it and writes ON MATCH.
        let db = GrafeoDB::new_in_memory();
        let session = db.session();
        alix_and_gus(&session);
        run(
            &session,
            Language::Gql,
            "MATCH (a:Person {name: 'Alix'}), (b:Person {name: 'Gus'}) INSERT (b)-[:KNOWS {w: 3}]->(a)",
        );
        run(
            &session,
            language,
            "MATCH (a:Person {name: 'Alix'}), (b:Person {name: 'Gus'}) \
             MERGE (a)<-[r:KNOWS]-(b) ON MATCH SET r.w = 19",
        );
        assert_eq!(
            run(
                &session,
                Language::Gql,
                "MATCH (s)-[r:KNOWS]->(t) RETURN s.name, t.name, r.w"
            ),
            [vec![
                Value::from("Gus"),
                Value::from("Alix"),
                Value::Int64(19)
            ]],
            "{language:?}: the existing relationship matched"
        );

        // Only Alix to Gus exists: the MERGE creates Gus to Alix beside it.
        let db = GrafeoDB::new_in_memory();
        let session = db.session();
        alix_and_gus(&session);
        run(
            &session,
            Language::Gql,
            "MATCH (a:Person {name: 'Alix'}), (b:Person {name: 'Gus'}) INSERT (a)-[:KNOWS]->(b)",
        );
        run(
            &session,
            language,
            "MATCH (a:Person {name: 'Alix'}), (b:Person {name: 'Gus'}) MERGE (a)<-[:KNOWS]-(b)",
        );
        assert_eq!(
            edges(&session, "KNOWS"),
            pairs(&[("Alix", "Gus"), ("Gus", "Alix")]),
            "{language:?}: the relationship the other way round did not match"
        );
    }
}

#[test]
fn a_left_arrow_in_merge_with_nodes_it_merges_too() {
    for language in LANGUAGES {
        let db = GrafeoDB::new_in_memory();
        let session = db.session();
        run(
            &session,
            language,
            "MERGE (a:Person {name: 'Alix'})<-[:KNOWS]-(b:Person {name: 'Gus'})",
        );
        run(
            &session,
            language,
            "MERGE (a:Person {name: 'Alix'})<-[:KNOWS]-(b:Person {name: 'Gus'})",
        );
        assert_eq!(
            edges(&session, "KNOWS"),
            pairs(&[("Gus", "Alix")]),
            "{language:?}: created once, then matched"
        );
    }
}

#[test]
fn an_undirected_merge_matches_a_relationship_either_way_round() {
    for language in LANGUAGES {
        let db = GrafeoDB::new_in_memory();
        let session = db.session();
        alix_and_gus(&session);
        run(
            &session,
            Language::Gql,
            "MATCH (a:Person {name: 'Alix'}), (b:Person {name: 'Gus'}) INSERT (b)-[:KNOWS {w: 3}]->(a)",
        );
        assert_eq!(
            run(
                &session,
                language,
                "MATCH (a:Person {name: 'Alix'}), (b:Person {name: 'Gus'}) \
                 MERGE (a)-[r:KNOWS]-(b) ON MATCH SET r.w = 88 RETURN r.w"
            ),
            [vec![Value::Int64(88)]],
            "{language:?}: the relationship from Gus to Alix matched"
        );
        assert_eq!(
            edges(&session, "KNOWS"),
            pairs(&[("Gus", "Alix")]),
            "{language:?}: nothing was created"
        );
    }
}

#[test]
fn an_undirected_merge_without_a_match_creates_from_left_to_right() {
    for language in LANGUAGES {
        let db = GrafeoDB::new_in_memory();
        let session = db.session();
        alix_and_gus(&session);
        for _ in 0..2 {
            run(
                &session,
                language,
                "MATCH (a:Person {name: 'Alix'}), (b:Person {name: 'Gus'}) MERGE (a)-[:KNOWS]-(b)",
            );
        }
        assert_eq!(
            edges(&session, "KNOWS"),
            pairs(&[("Alix", "Gus")]),
            "{language:?}: created once, from left to right, then matched"
        );
    }
}

#[test]
fn an_undirected_merge_binds_each_relationship_between_the_nodes() {
    for language in LANGUAGES {
        let db = GrafeoDB::new_in_memory();
        let session = db.session();
        alix_and_gus(&session);
        run(
            &session,
            Language::Gql,
            "MATCH (a:Person {name: 'Alix'}), (b:Person {name: 'Gus'}) \
             INSERT (a)-[:KNOWS {w: 3}]->(b), (b)-[:KNOWS {w: 19}]->(a)",
        );
        let mut weights = run(
            &session,
            language,
            "MATCH (a:Person {name: 'Alix'}), (b:Person {name: 'Gus'}) \
             MERGE (a)-[r:KNOWS]-(b) RETURN r.w",
        );
        weights.sort_by_key(|row| match row[0] {
            Value::Int64(w) => w,
            _ => i64::MIN,
        });
        assert_eq!(
            weights,
            [vec![Value::Int64(3)], vec![Value::Int64(19)]],
            "{language:?}: both relationships matched, each once"
        );
    }
}

#[test]
fn an_undirected_merge_binds_a_self_loop_once() {
    for language in LANGUAGES {
        let db = GrafeoDB::new_in_memory();
        let session = db.session();
        run(&session, Language::Gql, "INSERT (:Person {name: 'Alix'})");
        run(
            &session,
            Language::Gql,
            "MATCH (a:Person {name: 'Alix'}) INSERT (a)-[:KNOWS {w: 3}]->(a)",
        );
        assert_eq!(
            run(
                &session,
                language,
                "MATCH (a:Person {name: 'Alix'}) MERGE (a)-[r:KNOWS]-(a) RETURN r.w",
            ),
            [vec![Value::Int64(3)]],
            "{language:?}: the loop is one relationship, matched once"
        );
    }
}
