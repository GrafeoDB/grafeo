//! A SKOS vocabulary round-trips through SPARQL: the terms a CONSTRUCT
//! returns, written back as N-Triples into another database, are the triples
//! that were loaded, byte for byte, also after a reopen. Each literal keeps
//! its lexical form (a non-canonical `"019"`, a decimal's scale), its
//! language tag (with a region) and its datatype (a custom one too).
//!
//! ```bash
//! cargo test -p grafeo-engine --all-features --test sparql_term_round_trip
//! ```

#![cfg(all(feature = "sparql", feature = "triple-store", feature = "grafeo-file"))]

use std::collections::BTreeSet;

use grafeo_common::types::Value;
use grafeo_engine::database::ResultKind;
use grafeo_engine::{Config, GrafeoDB};

/// The vocabulary as N-Triples, one triple per line, with no blank node (a
/// blank node's label is not kept across stores).
const SKOS: &str = r#"<http://ex.org/fruit> <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> <http://www.w3.org/2004/02/skos/core#Concept> .
<http://ex.org/fruit> <http://www.w3.org/2004/02/skos/core#prefLabel> "Fruit"@en .
<http://ex.org/fruit> <http://www.w3.org/2004/02/skos/core#prefLabel> "Fruit"@fr .
<http://ex.org/fruit> <http://www.w3.org/2004/02/skos/core#prefLabel> "Fruit"@nl-BE .
<http://ex.org/apple> <http://www.w3.org/2004/02/skos/core#broader> <http://ex.org/fruit> .
<http://ex.org/apple> <http://www.w3.org/2004/02/skos/core#prefLabel> "Apple"@en .
<http://ex.org/apple> <http://www.w3.org/2004/02/skos/core#prefLabel> "Pomme"@fr .
<http://ex.org/apple> <http://www.w3.org/2004/02/skos/core#altLabel> "Kraków \"red\"\nrenet" .
<http://ex.org/apple> <http://www.w3.org/2004/02/skos/core#notation> "A-019"^^<http://ex.org/code> .
<http://ex.org/apple> <http://ex.org/rank> "019"^^<http://www.w3.org/2001/XMLSchema#integer> .
<http://ex.org/apple> <http://ex.org/price> "1.50"^^<http://www.w3.org/2001/XMLSchema#decimal> .
<http://ex.org/apple> <http://ex.org/seasonal> "true"^^<http://www.w3.org/2001/XMLSchema#boolean> .
<http://ex.org/apple> <http://ex.org/picked> "2026-01-15"^^<http://www.w3.org/2001/XMLSchema#date> .
<http://ex.org/apple> <http://ex.org/sold> "2026-01-15T10:30:00+01:00"^^<http://www.w3.org/2001/XMLSchema#dateTime> .
<http://ex.org/apple> <http://www.w3.org/2004/02/skos/core#exactMatch> <mailto:apple@ex.org> .
<http://ex.org/apple> <http://ex.org/note> "Apple"^^<http://www.w3.org/2001/XMLSchema#string> ."#;

/// The N-Triples lines of the database, from a CONSTRUCT of every triple.
fn triples(db: &GrafeoDB) -> BTreeSet<String> {
    let result = db
        .execute_sparql("CONSTRUCT { ?s ?p ?o } WHERE { ?s ?p ?o }")
        .unwrap();
    assert_eq!(result.kind, ResultKind::Graph);
    result
        .rows()
        .iter()
        .map(|row| {
            let terms: Vec<String> = row
                .iter()
                .map(|value| match value {
                    Value::RdfTerm(term) => term.to_ntriples(),
                    other => panic!("CONSTRUCT returned {other:?}, not a term"),
                })
                .collect();
            format!("{} .", terms.join(" "))
        })
        .collect()
}

/// What the vocabulary reads as: `xsd:string` is a simple literal in RDF
/// 1.1, so it comes back without its datatype, and a language tag in
/// lowercase, as RDF 1.2 keeps it.
fn expected() -> BTreeSet<String> {
    SKOS.lines()
        .map(|line| {
            line.replace("^^<http://www.w3.org/2001/XMLSchema#string>", "")
                .replace("@nl-BE", "@nl-be")
        })
        .collect()
}

fn load(db: &GrafeoDB, ntriples: &str) {
    db.execute_sparql(&format!("INSERT DATA {{\n{ntriples}\n}}"))
        .unwrap();
}

#[test]
fn a_construct_returns_the_triples_that_were_loaded() {
    let db = GrafeoDB::new_in_memory();
    load(&db, SKOS);
    assert_eq!(triples(&db), expected());
}

#[test]
fn construct_results_written_back_are_the_same_triples() {
    let source = GrafeoDB::new_in_memory();
    load(&source, SKOS);
    let copy = GrafeoDB::new_in_memory();
    let lines: Vec<String> = triples(&source).into_iter().collect();
    load(&copy, &lines.join("\n"));
    assert_eq!(triples(&copy), expected());
    // Each term is the same term: a pattern of the exact literal finds it.
    let found = copy
        .execute_sparql(
            r#"ASK {
                <http://ex.org/apple> <http://ex.org/rank> "019"^^<http://www.w3.org/2001/XMLSchema#integer> ;
                    <http://www.w3.org/2004/02/skos/core#prefLabel> "Pomme"@fr ;
                    <http://www.w3.org/2004/02/skos/core#notation> "A-019"^^<http://ex.org/code>
            }"#,
        )
        .unwrap();
    assert_eq!(found.rows()[0][0], Value::Bool(true));
}

#[test]
fn the_vocabulary_round_trips_through_a_reopen() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("skos.grafeo");
    {
        let db = GrafeoDB::with_config(Config::persistent(&path)).unwrap();
        load(&db, SKOS);
        db.close().unwrap();
    }
    let db = GrafeoDB::with_config(Config::persistent(&path)).unwrap();
    assert_eq!(triples(&db), expected());
}
