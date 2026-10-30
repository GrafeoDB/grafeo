//! A property map on an unlabelled node with an `id` index finds its node by
//! the index, also when label checks on that node sit between the map and
//! the scan (the LDBC Interactive short query 4 shape, `MATCH (m {id: $id})
//! WHERE m:Post OR m:Comment`): it reads one node, not every node. The checks
//! run on the node the index finds, in their order, so a check that would
//! fail on other nodes does not run for them. A check that reads another
//! variable, or a lookup after a write in the same statement, keeps its
//! rows (no lookup is promised there). In GQL and Cypher.
//!
//! ```bash
//! cargo test -p grafeo-engine --all-features --test index_lookup_through_label_checks
//! ```

#![cfg(all(feature = "gql", feature = "cypher"))]

use std::collections::HashMap;

use grafeo_common::types::Value;
use grafeo_engine::GrafeoDB;

const LANGUAGES: [&str; 2] = ["gql", "cypher"];

/// Posts `id` 0 to 199 and Comments 200 to 399, each with `zero: 0` except
/// Post 7 and Comment 250 (`zero: 1`); Forums 400 to 499; one Tag, 1000.
/// Post 7 has `lang: 'nl'`. An `id` index.
fn database() -> GrafeoDB {
    let db = GrafeoDB::new_in_memory();
    db.create_property_index("id").unwrap();
    db.execute("UNWIND range(0, 199) AS i INSERT (:Post {id: i, zero: 0, lang: 'en'})")
        .unwrap();
    db.execute("UNWIND range(200, 399) AS i INSERT (:Comment {id: i, zero: 0})")
        .unwrap();
    db.execute("UNWIND range(400, 499) AS i INSERT (:Forum {id: i, title: 'Amsterdam'})")
        .unwrap();
    db.execute("INSERT (:Tag {id: 1000, zero: 1})").unwrap();
    db.execute("MATCH (m:Post {id: 7}) SET m.zero = 1, m.lang = 'nl'")
        .unwrap();
    db.execute("MATCH (m:Comment {id: 250}) SET m.zero = 1")
        .unwrap();
    db
}

const IS4: &str = "MATCH (m {id: $id}) WHERE m:Post OR m:Comment RETURN m.id AS id";

fn run(db: &GrafeoDB, language: &str, query: &str, id: i64) -> (Vec<Vec<Value>>, Option<u64>) {
    let params = HashMap::from([("id".to_string(), Value::Int64(id))]);
    let result = db
        .execute_language(query, language, Some(params))
        .unwrap_or_else(|error| panic!("{language} `{query}` id {id}: {error}"));
    (result.rows().to_vec(), result.rows_scanned())
}

fn plan(db: &GrafeoDB, language: &str, query: &str) -> String {
    let query = format!("EXPLAIN {}", query.replace("$id", "7"));
    match &db.execute_language(&query, language, None).unwrap().rows()[0][0] {
        Value::String(text) => text.to_string(),
        other => panic!("{language}: {other:?}"),
    }
}

fn one(id: i64) -> Vec<Vec<Value>> {
    vec![vec![Value::Int64(id)]]
}

/// A Post and a Comment by id: one row each, one node read, and EXPLAIN
/// shows the index.
#[test]
fn the_is4_shape_reads_one_node() {
    let db = database();
    for language in LANGUAGES {
        for id in [7, 250] {
            let (rows, scanned) = run(&db, language, IS4, id);
            assert_eq!(rows, one(id), "{language} id {id}");
            assert_eq!(scanned, Some(1), "{language} id {id}: one node read");
        }
        let explained = plan(&db, language, IS4);
        assert!(explained.contains("[index: id]"), "{language}: {explained}");
    }
}

/// An id the index finds on a node with neither label (a Tag, a Forum), and
/// an id no node has: no row, at most the one node read.
#[test]
fn an_id_on_a_node_with_neither_label() {
    let db = database();
    for language in LANGUAGES {
        for id in [1000, 450, 5000] {
            let (rows, scanned) = run(&db, language, IS4, id);
            assert!(rows.is_empty(), "{language} id {id}: {rows:?}");
            assert!(
                scanned.unwrap_or(0) <= 1,
                "{language} id {id}: {scanned:?} nodes read"
            );
        }
    }
}

/// The check divides by `zero`, 0 on every Post and Comment but 7 and 250:
/// it runs only on the node the index finds, so it does not fail.
#[test]
fn the_checks_run_only_on_the_node_the_index_finds() {
    let db = database();
    let query =
        "MATCH (m {id: $id}) WHERE (m:Post OR m:Comment) AND 10 / m.zero > 0 RETURN m.id AS id";
    for language in LANGUAGES {
        for id in [7, 250] {
            let (rows, scanned) = run(&db, language, query, id);
            assert_eq!(rows, one(id), "{language} id {id}");
            assert_eq!(scanned, Some(1), "{language} id {id}");
        }
        // The Tag has `zero: 1` and neither label: no row, no error.
        let (rows, _) = run(&db, language, query, 1000);
        assert!(rows.is_empty(), "{language}: {rows:?}");
    }
}

/// Two checks on the node: the label OR and a property.
#[test]
fn two_checks_on_the_node() {
    let db = database();
    let query =
        "MATCH (m {id: $id}) WHERE (m:Post OR m:Comment) AND m.lang = 'nl' RETURN m.id AS id";
    for language in LANGUAGES {
        let (rows, scanned) = run(&db, language, query, 7);
        assert_eq!(rows, one(7), "{language}");
        assert_eq!(scanned, Some(1), "{language}");
        let (rows, _) = run(&db, language, query, 8);
        assert!(rows.is_empty(), "{language}: Post 8 is in English");
        let explained = plan(&db, language, query);
        assert!(explained.contains("[index: id]"), "{language}: {explained}");
    }
}

/// After a write in the same statement the lookup sees the new node (the
/// plan cannot look it up before the write).
#[test]
fn after_a_write_in_the_same_statement() {
    for (language, query) in [
        (
            "gql",
            "INSERT (:Post {id: 9999}) WITH 1 AS one \
             MATCH (m {id: 9999}) WHERE m:Post OR m:Comment RETURN m.id AS id",
        ),
        (
            "cypher",
            "CREATE (:Post {id: 9999}) WITH 1 AS one \
             MATCH (m {id: 9999}) WHERE m:Post OR m:Comment RETURN m.id AS id",
        ),
    ] {
        let db = database();
        let rows = db
            .execute_language(query, language, None)
            .unwrap_or_else(|error| panic!("{language}: {error}"))
            .rows()
            .to_vec();
        assert_eq!(rows, one(9999), "{language}");
    }
}

/// A check that reads another variable (`f`) is not a check on `m` alone:
/// the rows are those of checking it for every pair.
#[test]
fn a_check_that_reads_another_variable() {
    let db = database();
    let query = "MATCH (f:Forum {id: 400}), (m {id: $id}) \
                 WHERE m:Comment OR f.title = $title RETURN m.id AS id";
    for language in LANGUAGES {
        for (id, title, expected) in [
            (7, "Amsterdam", one(7)),
            (7, "Berlin", vec![]),
            (250, "Berlin", one(250)),
        ] {
            let params = HashMap::from([
                ("id".to_string(), Value::Int64(id)),
                ("title".to_string(), Value::from(title)),
            ]);
            let rows = db
                .execute_language(query, language, Some(params))
                .unwrap_or_else(|error| panic!("{language}: {error}"))
                .rows()
                .to_vec();
            assert_eq!(rows, expected, "{language} id {id} title {title}");
        }
    }
}
