//! A sign directly before a number is part of the number (SPARQL 1.1 grammar [148] to [154]):
//! a triple pattern's object `+1` is the literal `"+1"^^xsd:integer`, which matches only that
//! term, and a number keeps its sign in its lexical form. Updates match the same solutions.
//!
//! ```bash
//! cargo test -p grafeo-engine --all-features --test main sparql_signed_numbers::
//! ```

#![cfg(all(feature = "sparql", feature = "triple-store"))]

use grafeo_engine::config::GraphModel;
use grafeo_engine::{Config, GrafeoDB};

fn rdf() -> GrafeoDB {
    GrafeoDB::with_config(Config::in_memory().with_graph_model(GraphModel::Rdf)).unwrap()
}

/// Alix with the number `1` and an `urn:p` to Jules, who has `1` too.
fn loaded() -> GrafeoDB {
    let db = rdf();
    db.execute_sparql(
        "INSERT DATA { <urn:alix> <urn:p> 1 . <urn:alix> <urn:p> <urn:jules> . \
         <urn:jules> <urn:p> 1 }",
    )
    .unwrap();
    db
}

fn subjects(db: &GrafeoDB, query: &str) -> Vec<String> {
    let mut rows: Vec<String> = db
        .execute_sparql(query)
        .unwrap()
        .rows()
        .iter()
        .map(|row| format!("{:?}", row[0]))
        .collect();
    rows.sort();
    rows
}

fn count(db: &GrafeoDB) -> usize {
    db.execute_sparql("SELECT ?s ?p ?o WHERE { ?s ?p ?o }")
        .unwrap()
        .row_count()
}

/// An object `+1` or `-1` written after the predicate is one signed number, with or without
/// white space before the sign, and matches no stored `1`.
#[test]
fn a_signed_object_matches_no_other_term() {
    let db = loaded();
    for query in [
        "SELECT ?s WHERE { ?s <urn:p> +1 }",
        "SELECT ?s WHERE { ?s <urn:p>+1 }",
        "SELECT ?s WHERE { ?s <urn:p> -1 }",
    ] {
        assert!(subjects(&db, query).is_empty(), "{query}");
    }
    assert_eq!(subjects(&db, "SELECT ?s WHERE { ?s <urn:p> 1 }").len(), 2);
}

/// A signed number is its own term, so a `VALUES` entry `+1` joins no stored `1`, while `=`
/// compares the values.
#[test]
fn equality_compares_a_signed_number_by_value() {
    let db = loaded();
    assert_eq!(
        subjects(&db, "SELECT ?s WHERE { VALUES ?o { +1 } ?s <urn:p> ?o }"),
        Vec::<String>::new()
    );
    assert_eq!(
        subjects(&db, "SELECT ?s WHERE { ?s <urn:p> ?o FILTER(?o = +1) }").len(),
        2
    );
}

/// A delete whose pattern has a signed object removes nothing when no triple holds that term.
#[test]
fn a_delete_with_a_signed_object_removes_no_other_term() {
    let db = loaded();
    let before = count(&db);
    for update in [
        "DELETE { ?s <urn:p> ?o } WHERE { ?s <urn:p> +1 . ?s <urn:p> ?o }",
        "DELETE { ?s <urn:p> ?o } WHERE { ?s <urn:p>+1 . ?s <urn:p> ?o }",
        "DELETE { ?s <urn:p> ?o } WHERE { ?s <urn:p> -1 . ?s <urn:p> ?o }",
    ] {
        db.execute_sparql(update).unwrap();
        assert_eq!(count(&db), before, "{update}");
    }
}

/// A stored `+1` is its own term: a pattern with `+1` finds it and a delete removes it alone.
#[test]
fn a_signed_object_matches_its_own_term() {
    let db = loaded();
    db.execute_sparql("INSERT DATA { <urn:gus> <urn:p> +1 }")
        .unwrap();
    assert_eq!(
        subjects(&db, "SELECT ?s WHERE { ?s <urn:p> +1 }"),
        vec!["RdfTerm(<urn:gus>)".to_string()]
    );
    let before = count(&db);
    db.execute_sparql("DELETE WHERE { ?s <urn:p> +1 }").unwrap();
    assert_eq!(count(&db), before - 1);
    assert_eq!(subjects(&db, "SELECT ?s WHERE { ?s <urn:p> 1 }").len(), 2);
}

/// A signed number keeps its sign in its lexical form, and after an operand it is a sum or a
/// difference.
#[test]
fn a_signed_number_keeps_its_sign_and_adds_after_an_operand() {
    let db = rdf();
    let value = |query: &str| format!("{:?}", db.execute_sparql(query).unwrap().rows()[0][0]);
    let integer =
        |text: &str| format!("RdfTerm(\"{text}\"^^<http://www.w3.org/2001/XMLSchema#integer>)");
    for (query, expected) in [
        (
            "SELECT (STR(+05) AS ?y) WHERE {}",
            "RdfTerm(\"+05\")".to_string(),
        ),
        (
            "SELECT (STR(-05) AS ?y) WHERE {}",
            "RdfTerm(\"-05\")".to_string(),
        ),
        ("SELECT (3 -1 AS ?y) WHERE {}", integer("2")),
        ("SELECT (3 +1 AS ?y) WHERE {}", integer("4")),
        ("SELECT (3 - 1 AS ?y) WHERE {}", integer("2")),
    ] {
        assert_eq!(value(query), expected, "{query}");
    }
}
