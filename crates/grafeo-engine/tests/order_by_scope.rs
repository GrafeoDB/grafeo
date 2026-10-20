//! What an ORDER BY reads after a WITH or an aggregating RETURN, in GQL as
//! in Cypher (openCypher 9, ORDER BY). After a WITH that neither aggregates
//! nor is DISTINCT it reads the WITH's input as well (`WITH a.name AS name
//! ORDER BY a.age`; GQL failed with "Undefined variable"). After an
//! aggregating RETURN or WITH a key that repeats a returned expression reads
//! its column (GQL refused `ORDER BY max(a.age)` and failed on `ORDER BY
//! a.name` with an internal error), and a key that reads a variable the
//! clause drops is "Undefined variable" (an internal error in GQL). With
//! DISTINCT, ORDER BY reads only what the clause returns (it returned
//! duplicates).
//!
//! ```bash
//! cargo test -p grafeo-engine --all-features --test order_by_scope
//! ```

#![cfg(all(feature = "lpg", feature = "gql", feature = "cypher"))]

use grafeo_common::types::Value;
use grafeo_common::utils::error::{Error, QueryErrorKind};
use grafeo_engine::GrafeoDB;

const LANGUAGES: [&str; 2] = ["gql", "cypher"];

/// Alix 30, Gus 25, Mia 41 and a second Gus, 19.
fn people() -> GrafeoDB {
    let db = GrafeoDB::new_in_memory();
    db.execute(
        "INSERT (:P {name: 'Alix', age: 30}), (:P {name: 'Gus', age: 25}), \
         (:P {name: 'Mia', age: 41}), (:P {name: 'Gus', age: 19})",
    )
    .unwrap();
    db
}

fn rows(db: &GrafeoDB, language: &str, query: &str) -> Vec<Vec<Value>> {
    db.execute_language(query, language, None)
        .unwrap_or_else(|e| panic!("{language} `{query}`: {e}"))
        .rows()
        .to_vec()
}

fn names(values: &[&str]) -> Vec<Vec<Value>> {
    values.iter().map(|name| vec![Value::from(*name)]).collect()
}

/// Asserts a semantic error whose message holds `says`.
fn refused(db: &GrafeoDB, language: &str, query: &str, says: &str) {
    match db.execute_language(query, language, None) {
        Err(Error::Query(error)) => {
            assert_eq!(error.kind, QueryErrorKind::Semantic, "{language} `{query}`");
            assert!(
                error.message.contains(says),
                "{language} `{query}`: {}",
                error.message
            );
        }
        other => panic!("{language} `{query}`: expected a semantic error, got {other:?}"),
    }
}

/// A WITH that neither aggregates nor is DISTINCT lets its ORDER BY read its
/// input, also with a WHERE and a LIMIT after it; the variable is gone
/// again after the sort.
#[test]
fn an_order_by_after_a_plain_with_reads_its_input() {
    let db = people();
    for language in LANGUAGES {
        assert_eq!(
            rows(
                &db,
                language,
                "MATCH (a:P) WITH a.name AS name ORDER BY a.age RETURN name"
            ),
            names(&["Gus", "Gus", "Alix", "Mia"]),
            "{language}"
        );
        assert_eq!(
            rows(
                &db,
                language,
                "MATCH (a:P) WITH a.name AS name ORDER BY a.age DESC LIMIT 2 RETURN name"
            ),
            names(&["Mia", "Alix"]),
            "{language}: with a LIMIT"
        );
        assert_eq!(
            rows(
                &db,
                language,
                "MATCH (a:P) WITH a.name AS name WHERE name <> 'Mia' ORDER BY a.age RETURN name"
            ),
            names(&["Gus", "Gus", "Alix"]),
            "{language}: with a WHERE"
        );
        refused(
            &db,
            language,
            "MATCH (a:P) WITH a.name AS name ORDER BY a.age RETURN a.age AS age",
            "'a'",
        );
    }
}

/// After an aggregating RETURN a key that repeats a returned expression, a
/// grouping key or an aggregate, also inside a larger key, reads its column.
#[test]
fn an_order_by_after_an_aggregating_return_reads_its_columns() {
    let db = people();
    for language in LANGUAGES {
        let by_max = [
            [Value::from("Gus"), Value::Int64(25)],
            [Value::from("Alix"), Value::Int64(30)],
            [Value::from("Mia"), Value::Int64(41)],
        ];
        assert_eq!(
            rows(
                &db,
                language,
                "MATCH (a:P) RETURN a.name, max(a.age) ORDER BY max(a.age)"
            ),
            by_max,
            "{language}: unaliased"
        );
        assert_eq!(
            rows(
                &db,
                language,
                "MATCH (a:P) RETURN a.name AS name, max(a.age) AS m ORDER BY -max(a.age) DESC"
            ),
            by_max,
            "{language}: inside a key"
        );
        assert_eq!(
            rows(
                &db,
                language,
                "MATCH (a:P) RETURN a.name AS name, count(*) AS c ORDER BY a.name DESC"
            ),
            [
                [Value::from("Mia"), Value::Int64(1)],
                [Value::from("Gus"), Value::Int64(2)],
                [Value::from("Alix"), Value::Int64(1)],
            ],
            "{language}: a grouping key"
        );
        refused(
            &db,
            language,
            "MATCH (a:P) RETURN count(*) AS c ORDER BY a.age",
            "Undefined variable 'a': after an aggregating RETURN",
        );
        refused(
            &db,
            language,
            "MATCH (a:P) RETURN a.name AS name, count(*) AS c ORDER BY min(a.age)",
            "is not an aggregate the RETURN returns",
        );
    }
}

/// The ORDER BY of an aggregating WITH reads its columns, and of a DISTINCT
/// one too.
#[test]
fn an_order_by_after_an_aggregating_or_distinct_with_reads_its_columns() {
    let db = people();
    for language in LANGUAGES {
        assert_eq!(
            rows(
                &db,
                language,
                "MATCH (a:P) WITH a.name AS name, max(a.age) AS m ORDER BY max(a.age) DESC \
                 RETURN name"
            ),
            names(&["Mia", "Alix", "Gus"]),
            "{language}"
        );
        refused(
            &db,
            language,
            "MATCH (a:P) WITH a.name AS name, count(*) AS c ORDER BY a.age RETURN name",
            "Undefined variable 'a': after an aggregating WITH",
        );
        refused(
            &db,
            language,
            "MATCH (a:P) WITH DISTINCT a.name AS name ORDER BY a.age RETURN name",
            "ORDER BY reads 'a', which the WITH DISTINCT does not return",
        );
    }
}

/// With DISTINCT, ORDER BY reads only what the RETURN returns: the two
/// Guses are one row, with two ages, so sorting by the age is refused (it
/// returned both, openCypher TCK ReturnOrderBy2 [13]). A key that repeats a
/// returned expression reads its column.
#[test]
fn an_order_by_after_return_distinct_reads_its_columns() {
    let db = people();
    for language in LANGUAGES {
        refused(
            &db,
            language,
            "MATCH (a:P) RETURN DISTINCT a.name AS name ORDER BY a.age",
            "ORDER BY reads 'a', which the RETURN DISTINCT does not return",
        );
        assert_eq!(
            rows(
                &db,
                language,
                "MATCH (a:P) RETURN DISTINCT a.name AS name ORDER BY a.name DESC"
            ),
            names(&["Mia", "Gus", "Alix"]),
            "{language}: a returned expression"
        );
        assert_eq!(
            rows(
                &db,
                language,
                "MATCH (a:P) RETURN DISTINCT a.name AS name ORDER BY name"
            ),
            names(&["Alix", "Gus", "Mia"]),
            "{language}: an alias"
        );
    }
}

/// In a CALL body too: an aggregating RETURN, and a WITH whose import a key
/// reads.
#[test]
fn an_order_by_in_a_call_body() {
    let db = people();
    for language in LANGUAGES {
        assert_eq!(
            rows(
                &db,
                language,
                "CALL { MATCH (a:P) RETURN a.name AS name, max(a.age) AS m ORDER BY max(a.age) \
                 LIMIT 1 } RETURN name"
            ),
            names(&["Gus"]),
            "{language}"
        );
    }
    assert_eq!(
        rows(
            &db,
            "gql",
            "MATCH (a:P) CALL (a) { WITH a.name AS name ORDER BY a.age RETURN name } \
             RETURN name ORDER BY name"
        ),
        names(&["Alix", "Gus", "Gus", "Mia"]),
        "an import the WITH passes on"
    );
}
