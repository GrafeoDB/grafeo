//! A `\u` or `\U` escape in a SPARQL IRI stands for its code point (SPARQL
//! 1.1, section 19.2), so `<http://example.org/Krak\u00F3w>` and
//! `<http://example.org/Kraków>` are one IRI: in INSERT DATA, in a pattern,
//! in a FILTER, and against a triple a Turtle file loaded (whose loader
//! decodes the same escapes).
//!
//! ```bash
//! cargo test -p grafeo-engine --all-features --test main sparql_iri_escapes::
//! ```

#![cfg(all(feature = "sparql", feature = "triple-store"))]

use grafeo_common::types::Value;
use grafeo_common::types::rdf::Term;
use grafeo_engine::{GrafeoDB, RdfFormat};

/// `text` with every `~` a backslash, so an escape reads as written.
fn escaped(text: &str) -> String {
    text.replace('~', "\\")
}

const PLAIN: &str = "<http://example.org/Kraków>";
const ESCAPED: &str = "<http://example.org/Krak~u00F3w>";

/// The one-column rows of `query`, each term as N-Triples.
fn column(db: &GrafeoDB, query: &str) -> Vec<String> {
    let mut rows: Vec<String> = db
        .execute_sparql(query)
        .unwrap_or_else(|error| panic!("{query}: {error}"))
        .rows()
        .iter()
        .map(|row| match &row[0] {
            Value::RdfTerm(term) => term.to_ntriples(),
            other => panic!("{query} returned {other:?}, not a term"),
        })
        .collect();
    rows.sort();
    rows
}

/// What `iri` (either spelling) finds in `db`: the names of its pattern, of
/// a FILTER on it, and its subjects.
fn found_by(db: &GrafeoDB, iri: &str) -> [Vec<String>; 3] {
    let iri = escaped(iri);
    [
        column(
            db,
            &format!("SELECT ?name WHERE {{ {iri} <http://example.org/name> ?name }}"),
        ),
        column(
            db,
            &format!(
                "SELECT ?name WHERE {{ ?city <http://example.org/name> ?name FILTER(?city = {iri}) }}"
            ),
        ),
        column(
            db,
            "SELECT ?city WHERE { ?city <http://example.org/name> ?name }",
        ),
    ]
}

/// What both spellings find in a database holding the city once.
fn expected() -> [Vec<String>; 3] {
    [
        vec![r#""Kraków""#.to_string()],
        vec![r#""Kraków""#.to_string()],
        vec![PLAIN.to_string()],
    ]
}

/// A city inserted with the escaped spelling is found by the plain one, and
/// by the escaped one.
#[test]
fn an_escaped_insert_is_found_by_the_plain_iri() {
    let db = GrafeoDB::new_in_memory();
    db.execute_sparql(&escaped(&format!(
        "INSERT DATA {{ {ESCAPED} <http://example.org/name> 'Kraków' }}"
    )))
    .unwrap();
    assert_eq!(found_by(&db, PLAIN), expected(), "the plain spelling");
    assert_eq!(found_by(&db, ESCAPED), expected(), "the escaped spelling");
}

/// A city inserted with the plain spelling is found by the escaped one, and
/// an insert of the other spelling adds nothing.
#[test]
fn a_plain_insert_is_found_by_the_escaped_iri() {
    let db = GrafeoDB::new_in_memory();
    db.execute_sparql(&format!(
        "INSERT DATA {{ {PLAIN} <http://example.org/name> 'Kraków' }}"
    ))
    .unwrap();
    assert_eq!(found_by(&db, ESCAPED), expected(), "the escaped spelling");

    db.execute_sparql(&escaped(&format!(
        "INSERT DATA {{ {ESCAPED} <http://example.org/name> 'Kraków' }}"
    )))
    .unwrap();
    assert_eq!(found_by(&db, PLAIN), expected(), "one triple, not two");
}

/// A Turtle file that writes the city either way is found by both SPARQL
/// spellings.
#[test]
fn a_turtle_loaded_iri_is_found_by_both_spellings() {
    let dir = tempfile::tempdir().unwrap();
    for (name, iri) in [("plain.ttl", PLAIN), ("escaped.ttl", ESCAPED)] {
        let path = dir.path().join(name);
        std::fs::write(
            &path,
            escaped(&format!("{iri} <http://example.org/name> \"Kraków\" .\n")),
        )
        .unwrap();
        let db = GrafeoDB::new_in_memory();
        assert_eq!(db.load_rdf(&path, Some(RdfFormat::Turtle)).unwrap(), 1);
        assert_eq!(
            found_by(&db, PLAIN),
            expected(),
            "{name}: the plain spelling"
        );
        assert_eq!(
            found_by(&db, ESCAPED),
            expected(),
            "{name}: the escaped spelling"
        );
    }
}

/// The plain and the escaped spelling of an IRI are one term to the store
/// too, so the decoded IRI is the one stored.
#[test]
fn the_stored_iri_is_decoded() {
    let db = GrafeoDB::new_in_memory();
    db.execute_sparql(&escaped(
        "INSERT DATA { <http://example.org/gus> <http://example.org/lives> <http://example.org/Krak~u00F3w> }",
    ))
    .unwrap();
    let cities = column(
        &db,
        "SELECT ?city WHERE { <http://example.org/gus> <http://example.org/lives> ?city }",
    );
    assert_eq!(cities, vec![PLAIN.to_string()]);
    assert_eq!(
        Term::iri("http://example.org/Kraków").to_ntriples(),
        PLAIN,
        "the plain spelling is the term's own"
    );
}

/// A string literal decodes the same escapes (it always did): the name
/// written escaped is the plain one.
#[test]
fn a_string_escape_is_its_code_point_too() {
    let db = GrafeoDB::new_in_memory();
    db.execute_sparql(&escaped(
        "INSERT DATA { <http://example.org/mia> <http://example.org/name> 'Krak~u00F3w ~U0001F6B2' }",
    ))
    .unwrap();
    let names = column(
        &db,
        "SELECT ?name WHERE { <http://example.org/mia> <http://example.org/name> ?name }",
    );
    assert_eq!(names, vec!["\"Kraków \u{1F6B2}\"".to_string()]);
}
