//! A procedure call inside a query runs once for each row before it, as
//! openCypher and ISO GQL define it: its rows are joined with every incoming
//! row, which stays in scope. A Cypher `CALL proc() YIELD x` after a `MATCH`
//! or an `UNWIND` dropped its input (one row per procedure row, the earlier
//! variables gone), and GQL did not parse a `CALL ... YIELD` followed by a
//! `RETURN` inside a query.
//!
//! ```bash
//! cargo test -p grafeo-engine --all-features --test procedure_call_in_a_query
//! ```

#![cfg(all(
    feature = "lpg",
    feature = "gql",
    feature = "cypher",
    feature = "algos"
))]

use grafeo_common::types::Value;
use grafeo_common::utils::error::{Error, QueryErrorKind};
use grafeo_engine::GrafeoDB;

/// Alix knows Gus; Mia knows nobody. Returns the database and Alix's id.
fn people() -> (GrafeoDB, i64) {
    let db = GrafeoDB::new_in_memory();
    db.execute(
        "INSERT (:Person {name: 'Alix'})-[:KNOWS]->(:Person {name: 'Gus'}), \
         (:Person {name: 'Mia'})",
    )
    .unwrap();
    let alix = match &db
        .execute("MATCH (p:Person {name: 'Alix'}) RETURN id(p)")
        .unwrap()
        .rows()[0][0]
    {
        Value::Int64(id) => *id,
        other => panic!("id() returned {other:?}"),
    };
    (db, alix)
}

fn rows(db: &GrafeoDB, language: &str, query: &str) -> Vec<Vec<Value>> {
    db.execute_language(query, language, None)
        .unwrap_or_else(|e| panic!("{language} `{query}`: {e}"))
        .rows()
        .to_vec()
}

/// Each of the three people runs the BFS from Alix, which reaches Alix and
/// Gus: six rows.
#[test]
fn a_call_after_match_runs_once_per_row() {
    let (db, alix) = people();
    for language in ["cypher", "gql"] {
        assert_eq!(
            rows(
                &db,
                language,
                &format!(
                    "MATCH (p:Person) CALL grafeo.bfs({alix}) YIELD node_id RETURN count(*) AS c"
                )
            ),
            [[Value::Int64(6)]],
            "{language}"
        );
    }
}

/// The variables bound before the call stay in scope after it.
#[test]
fn the_earlier_variables_stay_in_scope() {
    let (db, _) = people();
    for language in ["cypher", "gql"] {
        assert_eq!(
            rows(
                &db,
                language,
                "UNWIND [1, 2] AS x CALL db.labels() YIELD label RETURN x, label ORDER BY x"
            ),
            [
                [Value::Int64(1), Value::from("Person")],
                [Value::Int64(2), Value::from("Person")]
            ],
            "{language}"
        );
        assert_eq!(
            rows(
                &db,
                language,
                "MATCH (a:Person {name: 'Alix'}) CALL db.labels() YIELD label \
                 RETURN a.name AS a, label"
            ),
            [[Value::from("Alix"), Value::from("Person")]],
            "{language}"
        );
    }
}

/// A `WHERE` after the `YIELD` filters the call's rows, per incoming row.
#[test]
fn a_where_after_yield_filters_the_calls_rows() {
    let (db, alix) = people();
    let gus = rows(&db, "gql", "MATCH (p:Person {name: 'Gus'}) RETURN id(p)")[0][0].clone();
    for language in ["cypher", "gql"] {
        assert_eq!(
            rows(
                &db,
                language,
                &format!(
                    "MATCH (p:Person) CALL grafeo.bfs({alix}) YIELD node_id, depth \
                     WHERE depth = 1 RETURN p.name AS name, node_id ORDER BY name"
                )
            ),
            [
                [Value::from("Alix"), gus.clone()],
                [Value::from("Gus"), gus.clone()],
                [Value::from("Mia"), gus.clone()]
            ],
            "{language}: each person, with Gus at depth 1"
        );
    }
}

/// A call inside a subquery, and a call as the whole statement, work as
/// before.
#[test]
fn calls_in_a_subquery_and_as_a_statement() {
    let (db, _) = people();
    assert_eq!(
        rows(
            &db,
            "gql",
            "CALL { CALL db.labels() YIELD label RETURN count(label) AS label_count } \
             RETURN label_count"
        ),
        [[Value::Int64(1)]]
    );
    for language in ["cypher", "gql"] {
        assert_eq!(
            rows(&db, language, "CALL db.labels() YIELD label RETURN label"),
            [[Value::from("Person")]],
            "{language}"
        );
    }
}

/// An argument that reads a row stays an error naming the variable: the
/// call's arguments are constants, a limit of Grafeo's (unsupported, not a
/// mistake in the query).
#[test]
fn an_argument_that_reads_the_row_is_an_error() {
    let (db, _) = people();
    for language in ["cypher", "gql"] {
        match db.execute_language(
            "UNWIND [0] AS x CALL grafeo.bfs(x) YIELD node_id RETURN node_id",
            language,
            None,
        ) {
            Err(Error::Query(error)) => {
                assert_eq!(
                    error.kind,
                    QueryErrorKind::Unsupported,
                    "{language}: {error}"
                );
                assert!(error.message.contains("'x'"), "{language}: {error}");
            }
            other => panic!("{language}: expected an unsupported error, got {other:?}"),
        }
    }
}
