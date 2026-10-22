//! The kind of result each SPARQL query form returns: rows for SELECT and
//! updates, one boolean for ASK, a graph for CONSTRUCT; EXPLAIN is rows
//! whatever the form.
//!
//! ```bash
//! cargo test -p grafeo-engine --all-features --test sparql_result_kind
//! ```

#![cfg(all(feature = "sparql", feature = "triple-store"))]

use grafeo_common::types::Value;
use grafeo_engine::GrafeoDB;
use grafeo_engine::database::ResultKind;

fn fruit_db() -> GrafeoDB {
    let db = GrafeoDB::new_in_memory();
    db.execute_sparql(
        r#"INSERT DATA {
            <http://example.org/apple> <http://www.w3.org/2004/02/skos/core#prefLabel> "Apple"@en .
        }"#,
    )
    .unwrap();
    db
}

fn boolean(db: &GrafeoDB, query: &str) -> bool {
    let result = db.execute_sparql(query).unwrap();
    assert_eq!(result.kind, ResultKind::Boolean, "{query}");
    assert_eq!(result.columns, vec!["boolean"], "{query}");
    assert_eq!(result.row_count(), 1, "{query}");
    match &result.rows()[0][0] {
        Value::Bool(found) => *found,
        other => panic!("ASK returned {other:?} for {query}"),
    }
}

#[test]
fn select_and_updates_return_rows() {
    let db = fruit_db();
    let select = db.execute_sparql("SELECT ?s WHERE { ?s ?p ?o }").unwrap();
    assert_eq!(select.kind, ResultKind::Rows);
    let update = db
        .execute_sparql("INSERT DATA { <http://example.org/pear> <http://example.org/p> 1 }")
        .unwrap();
    assert_eq!(update.kind, ResultKind::Rows);
}

#[test]
fn construct_returns_a_graph() {
    let db = fruit_db();
    let construct = db
        .execute_sparql("CONSTRUCT { ?s ?p ?o } WHERE { ?s ?p ?o }")
        .unwrap();
    assert_eq!(construct.kind, ResultKind::Graph);
    assert_eq!(construct.columns, vec!["subject", "predicate", "object"]);
}

#[test]
fn ask_tells_whether_the_pattern_matches() {
    let db = fruit_db();
    assert!(boolean(&db, r#"ASK { ?s ?p "Apple"@en }"#));
    assert!(!boolean(&db, r#"ASK { ?s ?p "Pear"@en }"#));
    assert!(!boolean(&GrafeoDB::new_in_memory(), "ASK { ?s ?p ?o }"));
}

#[test]
fn ask_through_a_session_is_a_boolean_too() {
    let db = fruit_db();
    let session = db.session();
    let result = session
        .execute_sparql(r#"ASK { ?s ?p "Apple"@en }"#)
        .unwrap();
    assert_eq!(result.kind, ResultKind::Boolean);
    assert_eq!(result.rows()[0][0], Value::Bool(true));
}

#[test]
fn explain_returns_rows_whatever_the_query_form() {
    let db = fruit_db();
    for query in [
        "EXPLAIN ASK { ?s ?p ?o }",
        "EXPLAIN CONSTRUCT { ?s ?p ?o } WHERE { ?s ?p ?o }",
    ] {
        let result = db.execute_sparql(query).unwrap();
        assert_eq!(result.kind, ResultKind::Rows, "{query}");
        assert_ne!(result.columns, vec!["boolean"], "{query}");
    }
}

#[test]
fn other_languages_return_rows() {
    let db = GrafeoDB::new_in_memory();
    let result = db.execute("RETURN 1 AS one").unwrap();
    assert_eq!(result.kind, ResultKind::Rows);
}
