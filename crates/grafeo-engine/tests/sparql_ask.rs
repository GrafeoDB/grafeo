//! `ASK` answers with one row holding one Boolean, in the column `boolean`:
//! the name the SPARQL 1.1 Query Results JSON Format gives an ASK result
//! (section 4.2, "boolean"), true when the pattern has a solution and false
//! when it has none, on every path that runs a query.
//!
//! ```bash
//! cargo test -p grafeo-engine --all-features --test sparql_ask
//! ```

#![cfg(all(feature = "sparql", feature = "triple-store"))]

use grafeo_common::types::Value;
use grafeo_engine::config::GraphModel;
use grafeo_engine::database::QueryResult;
use grafeo_engine::{Config, GrafeoDB};

fn loaded() -> GrafeoDB {
    let db = GrafeoDB::with_config(Config::in_memory().with_graph_model(GraphModel::Rdf)).unwrap();
    db.execute_sparql(
        "INSERT DATA { <http://example.org/alix> <http://example.org/knows> <http://example.org/gus> . \
         <http://example.org/alix> <http://example.org/age> 30 . \
         GRAPH <http://example.org/paris> { <http://example.org/vincent> <http://example.org/knows> <http://example.org/mia> } }",
    )
    .unwrap();
    db
}

/// The answer of an ASK result: its one row's one Boolean, in `boolean`.
#[track_caller]
fn answer(result: &QueryResult) -> bool {
    assert_eq!(result.columns, ["boolean"], "{result:?}");
    assert_eq!(result.row_count(), 1, "{result:?}");
    match &result.rows()[0][..] {
        [Value::Bool(answer)] => *answer,
        row => panic!("an ASK row is one Boolean: {row:?}"),
    }
}

/// True for a pattern with a solution, false for one without, whatever the
/// pattern (bound, with variables, a filter, an optional part, a named
/// graph, a dataset clause), through a session and through `execute_sparql`.
#[test]
fn ask_answers_with_one_boolean_row() {
    let db = loaded();
    let cases = [
        (
            "ASK { <http://example.org/alix> <http://example.org/knows> <http://example.org/gus> }",
            true,
        ),
        (
            "ASK { <http://example.org/alix> <http://example.org/knows> <http://example.org/mia> }",
            false,
        ),
        ("ASK { ?s <http://example.org/knows> ?o }", true),
        ("ASK { ?s <http://example.org/likes> ?o }", false),
        (
            "ASK { ?s <http://example.org/age> ?age FILTER (?age > 25) }",
            true,
        ),
        (
            "ASK { ?s <http://example.org/age> ?age FILTER (?age > 40) }",
            false,
        ),
        (
            "ASK { <http://example.org/alix> <http://example.org/knows> ?o \
             OPTIONAL { ?o <http://example.org/age> ?age } }",
            true,
        ),
        (
            "ASK { GRAPH <http://example.org/paris> { ?s ?p ?o } }",
            true,
        ),
        (
            "ASK { GRAPH <http://example.org/berlin> { ?s ?p ?o } }",
            false,
        ),
        (
            "ASK FROM <http://example.org/paris> { <http://example.org/vincent> ?p ?o }",
            true,
        ),
        (
            "ASK FROM <http://example.org/paris> { <http://example.org/alix> ?p ?o }",
            false,
        ),
        (
            "ASK { <http://example.org/alix> <http://example.org/knows>+ <http://example.org/gus> }",
            true,
        ),
    ];
    for (query, expected) in cases {
        assert_eq!(
            answer(&db.execute_sparql(query).unwrap()),
            expected,
            "{query}"
        );
        assert_eq!(
            answer(&db.session().execute_sparql(query).unwrap()),
            expected,
            "{query} in a session"
        );
    }
}

/// An empty store answers false; a transaction answers from what it sees,
/// its own writes included.
#[test]
fn ask_answers_from_what_the_reader_sees() {
    let db = GrafeoDB::with_config(Config::in_memory().with_graph_model(GraphModel::Rdf)).unwrap();
    let ask = "ASK { ?s ?p ?o }";
    assert!(!answer(&db.execute_sparql(ask).unwrap()), "an empty store");

    let mut session = db.session();
    session.begin_transaction().unwrap();
    session
        .execute_sparql(
            "INSERT DATA { <http://example.org/jules> <http://example.org/knows> <http://example.org/butch> }",
        )
        .unwrap();
    assert!(
        answer(&session.execute_sparql(ask).unwrap()),
        "the transaction's own write"
    );
    assert!(
        !answer(&db.execute_sparql(ask).unwrap()),
        "another reader, before the commit"
    );
    session.commit().unwrap();
    assert!(answer(&db.execute_sparql(ask).unwrap()), "after the commit");
}
