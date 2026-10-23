//! The SPARQL string tests (`CONTAINS`, `STRSTARTS`, `STRENDS`, `REGEX`,
//! `STRLEN`) read their arguments where they are: an inlined short string's
//! bytes, a stored literal's text and tag, a constant of the query. Each kind
//! of argument gives the answer SPARQL 1.1 section 17.4.3 gives, and a type
//! error (a number, an IRI, an incompatible tag) drops the row.
//!
//! ```bash
//! cargo test -p grafeo-engine --all-features --test sparql_string_tests_on_term_ids
//! ```

#![cfg(all(feature = "sparql", feature = "triple-store"))]

use grafeo_common::types::Value;
use grafeo_engine::{Config, GrafeoDB, GraphModel};

/// One value per subject: `Kraków` is 7 bytes (an inlined short string),
/// `Amsterdam` is stored in the dictionary, then a tagged literal, a typed
/// `xsd:string`, an integer and an IRI.
const DATA: &str = r#"PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>
INSERT DATA {
    <http://example.org/short> <http://example.org/v> "Kraków" .
    <http://example.org/stored> <http://example.org/v> "Amsterdam" .
    <http://example.org/tagged> <http://example.org/v> "Prague"@en .
    <http://example.org/typed> <http://example.org/v> "Berlin"^^xsd:string .
    <http://example.org/number> <http://example.org/v> 42 .
    <http://example.org/iri> <http://example.org/v> <http://example.org/Paris> .
}"#;

/// No subject passes.
const NONE: [&str; 0] = [];

fn db() -> GrafeoDB {
    let db = GrafeoDB::with_config(Config::in_memory().with_graph_model(GraphModel::Rdf)).unwrap();
    db.execute_sparql(DATA).unwrap();
    db
}

/// The local names of the subjects whose `?v` passes `filter`, sorted.
fn passing(db: &GrafeoDB, filter: &str) -> Vec<String> {
    let result = db
        .execute_sparql(&format!(
            "SELECT ?s WHERE {{ ?s <http://example.org/v> ?v FILTER({filter}) }}"
        ))
        .unwrap_or_else(|error| panic!("{filter}: {error}"));
    let mut names: Vec<String> = result
        .rows()
        .iter()
        .map(|row| match &row[0] {
            Value::RdfTerm(term) => term
                .to_string()
                .trim_matches(|c| c == '<' || c == '>')
                .trim_start_matches("http://example.org/")
                .to_string(),
            other => panic!("{filter}: a subject is an IRI, not {other:?}"),
        })
        .collect();
    names.sort();
    names
}

#[test]
fn every_kind_of_string_argument_is_read() {
    let db = db();
    assert_eq!(passing(&db, r#"CONTAINS(?v, "ak")"#), ["short"]);
    assert_eq!(passing(&db, r#"CONTAINS(?v, "ster")"#), ["stored"]);
    assert_eq!(passing(&db, r#"CONTAINS(?v, "rag")"#), ["tagged"]);
    assert_eq!(passing(&db, r#"CONTAINS(?v, "erl")"#), ["typed"]);
    assert_eq!(passing(&db, r#"STRSTARTS(?v, "Kra")"#), ["short"]);
    assert_eq!(passing(&db, r#"STRENDS(?v, "dam")"#), ["stored"]);
    assert_eq!(passing(&db, r#"STRENDS(?v, "ków")"#), ["short"]);
    assert_eq!(passing(&db, r#"REGEX(?v, "^K.*w$")"#), ["short"]);
    assert_eq!(passing(&db, r#"REGEX(?v, "^p", "i")"#), ["tagged"]);
    // "Kraków" is 6 characters in 7 bytes, as "Prague" and "Berlin" are 6.
    assert_eq!(passing(&db, "STRLEN(?v) = 6"), ["short", "tagged", "typed"]);
    assert_eq!(passing(&db, "STRLEN(?v) = 9"), ["stored"]);
}

/// The empty string is in every string; the type errors drop their rows.
#[test]
fn a_number_or_an_iri_is_a_type_error() {
    let db = db();
    assert_eq!(
        passing(&db, r#"CONTAINS(?v, "")"#),
        ["short", "stored", "tagged", "typed"]
    );
    assert_eq!(
        passing(&db, r#"!CONTAINS(?v, "zzz")"#),
        ["short", "stored", "tagged", "typed"],
        "an error stays an error under !"
    );
    assert_eq!(passing(&db, r#"REGEX(?v, "4")"#), NONE);
    assert_eq!(passing(&db, "STRLEN(?v) = 2"), NONE);
}

/// Argument compatibility (17.4.3.1.1): a tagged first argument takes an
/// untagged second one or the same tag; an untagged one takes no tag.
#[test]
fn tags_must_be_compatible() {
    let db = db();
    assert_eq!(passing(&db, r#"CONTAINS(?v, "ag"@en)"#), ["tagged"]);
    assert_eq!(passing(&db, r#"CONTAINS(?v, "ag"@EN)"#), ["tagged"]);
    assert_eq!(passing(&db, r#"CONTAINS(?v, "ag"@fr)"#), NONE);
    assert_eq!(
        passing(&db, r#"CONTAINS(?v, "ak"@pl)"#),
        NONE,
        "an untagged first argument takes no tagged second one"
    );
    assert_eq!(
        passing(&db, r#"STRSTARTS("Kraków"@pl, ?v)"#),
        ["short"],
        "a constant first, the inlined short string second"
    );
}
