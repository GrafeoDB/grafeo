//! A triple whose subject is a literal, or whose predicate is a blank node or
//! a literal, is no RDF triple. Every way in refuses it, in debug and in
//! release builds alike (a `debug_assert` in `Triple::new` panicked in debug
//! builds and nothing checked in release builds, which stored it):
//!
//! - SPARQL `INSERT DATA` fails and stores nothing;
//! - a SPARQL `INSERT` template whose variable binds a literal subject (or a
//!   blank or literal predicate) skips that triple and stores the others, as
//!   SPARQL 1.1 Update says;
//! - `GrafeoDB::batch_insert_rdf` fails and stores nothing of the batch;
//! - `RdfStore::insert` does not store it.
//!
//! A database file of valid triples still loads (the section loader builds
//! its triples unchecked, so files written before keep loading).
//!
//! ```bash
//! cargo test -p grafeo-engine --all-features --test rdf_invalid_triples
//! cargo test -p grafeo-engine --all-features --release --test rdf_invalid_triples
//! ```

#![cfg(all(feature = "sparql", feature = "triple-store"))]

use std::panic::{AssertUnwindSafe, catch_unwind};

use grafeo_core::graph::rdf::{Term, Triple};
use grafeo_engine::GrafeoDB;

const ALIX: &str = "http://ex.org/alix";
const GUS: &str = "http://ex.org/gus";
const KNOWS: &str = "http://ex.org/knows";

/// Runs `f`; a panic comes back as an error with its message, so a test
/// reports every case that fails: a refused triple is an error or a skip,
/// never a panic.
fn run<T>(f: impl FnOnce() -> T) -> Result<T, String> {
    catch_unwind(AssertUnwindSafe(f)).map_err(|payload| {
        let message = payload
            .downcast_ref::<&str>()
            .map(ToString::to_string)
            .or_else(|| payload.downcast_ref::<String>().cloned())
            .unwrap_or_default();
        format!("panicked ({message}), where it is refused without a panic")
    })
}

/// Fails the test with every failure of its cases, one a line.
fn assert_no_failures(failures: &[String]) {
    assert!(failures.is_empty(), "\n{}", failures.join("\n"));
}

/// The triples of the default graph as `[s, p, o]` debug strings, sorted.
fn default_graph(db: &GrafeoDB) -> Vec<Vec<String>> {
    let mut rows: Vec<Vec<String>> = db
        .execute_sparql("SELECT ?s ?p ?o WHERE { ?s ?p ?o }")
        .unwrap()
        .rows()
        .iter()
        .map(|row| row.iter().map(|value| format!("{value:?}")).collect())
        .collect();
    rows.sort();
    rows
}

/// The three invalid shapes, built unchecked: a literal subject, a blank
/// predicate, a literal predicate.
fn invalid_triples() -> Vec<(&'static str, Triple)> {
    vec![
        (
            "a literal subject",
            Triple::new_unchecked(Term::literal("Alix"), Term::iri(KNOWS), Term::iri(GUS)),
        ),
        (
            "a blank predicate",
            Triple::new_unchecked(Term::iri(ALIX), Term::blank("knows"), Term::iri(GUS)),
        ),
        (
            "a literal predicate",
            Triple::new_unchecked(Term::iri(ALIX), Term::literal("knows"), Term::iri(GUS)),
        ),
    ]
}

fn valid_triple() -> Triple {
    Triple::new(Term::iri(ALIX), Term::iri(KNOWS), Term::iri(GUS))
}

// ── SPARQL ──────────────────────────────────────────────────────────────

/// `INSERT DATA` of a triple with a literal subject, a blank predicate or a
/// literal predicate fails (a parse error is fine) and stores nothing, also
/// not the valid triple next to it.
#[test]
fn sparql_insert_data_of_an_invalid_triple_fails_and_stores_nothing() {
    let mut failures = Vec::new();
    for (shape, update) in [
        (
            "a literal subject",
            r#"INSERT DATA { <http://ex.org/alix> <http://ex.org/knows> <http://ex.org/gus> .
                             "Alix" <http://ex.org/knows> <http://ex.org/gus> . }"#,
        ),
        (
            "a typed literal subject",
            r#"INSERT DATA { <http://ex.org/alix> <http://ex.org/knows> <http://ex.org/gus> .
                             "3"^^<http://www.w3.org/2001/XMLSchema#integer>
                                 <http://ex.org/knows> <http://ex.org/gus> . }"#,
        ),
        (
            "a blank predicate",
            r"INSERT DATA { <http://ex.org/alix> <http://ex.org/knows> <http://ex.org/gus> .
                            <http://ex.org/alix> _:knows <http://ex.org/gus> . }",
        ),
        (
            "a literal predicate",
            r#"INSERT DATA { <http://ex.org/alix> <http://ex.org/knows> <http://ex.org/gus> .
                             <http://ex.org/alix> "knows" <http://ex.org/gus> . }"#,
        ),
    ] {
        let db = GrafeoDB::new_in_memory();
        match run(|| db.execute_sparql(update)) {
            Err(panic) => failures.push(format!("{shape}: {panic}")),
            Ok(Ok(_)) => failures.push(format!("{shape}: succeeded, where it fails")),
            Ok(Err(_)) => {}
        }
        if !db.rdf_store().is_empty() {
            failures.push(format!("{shape}: stored {:?}", default_graph(&db)));
        }
    }
    assert_no_failures(&failures);
}

/// An `INSERT` template whose subject variable binds a literal skips that
/// solution's triple without an error (SPARQL 1.1 Update, 3.1.3: a template
/// that yields an illegal triple inserts nothing for it) and inserts the
/// others; so does a predicate variable bound to a literal or a blank node.
#[test]
fn sparql_insert_template_skips_an_invalid_triple_and_inserts_the_others() {
    let db = GrafeoDB::new_in_memory();
    let mut failures = Vec::new();
    for update in [
        r#"INSERT { ?s <http://ex.org/city> <http://ex.org/amsterdam> }
           WHERE { VALUES ?s { <http://ex.org/alix> "Gus" } }"#,
        r#"INSERT { ?s <http://ex.org/city> <http://ex.org/berlin> }
           WHERE { BIND("Vincent" AS ?s) }"#,
        r#"INSERT { <http://ex.org/jules> ?p <http://ex.org/paris> }
           WHERE { VALUES ?p { "city" <http://ex.org/city> } }"#,
        r"INSERT { <http://ex.org/mia> ?p <http://ex.org/paris> }
          WHERE { BIND(BNODE() AS ?p) }",
        r#"DELETE { <http://ex.org/nobody> <http://ex.org/city> <http://ex.org/nowhere> }
           INSERT { ?s <http://ex.org/city> <http://ex.org/paris> }
           WHERE { BIND("Mia" AS ?s) }"#,
    ] {
        match run(|| db.execute_sparql(update)) {
            Err(panic) => failures.push(format!("{update}: {panic}")),
            Ok(Err(error)) => failures.push(format!("{update}: {error}, where it skips")),
            Ok(Ok(_)) => {}
        }
    }
    let iri = |name: &str| Term::iri(format!("http://ex.org/{name}"));
    let store = db.rdf_store();
    for (subject, city) in [("alix", "amsterdam"), ("jules", "paris")] {
        let triple = Triple::new(iri(subject), iri("city"), iri(city));
        if !store.contains(&triple) {
            failures.push(format!("{triple} is missing"));
        }
    }
    if store.len() != 2 {
        failures.push(format!(
            "only Alix in Amsterdam and Jules in Paris, where it holds {:?}",
            default_graph(&db)
        ));
    }
    assert_no_failures(&failures);
}

// ── The Rust API ────────────────────────────────────────────────────────

/// `batch_insert_rdf` refuses a batch that holds an invalid triple with an
/// error and stores none of it, in debug and release builds.
#[test]
fn batch_insert_rdf_refuses_an_invalid_triple() {
    let mut failures = Vec::new();
    for (shape, invalid) in invalid_triples() {
        let db = GrafeoDB::new_in_memory();
        match run(|| db.batch_insert_rdf(vec![valid_triple(), invalid])) {
            Err(panic) => failures.push(format!("{shape}: {panic}")),
            Ok(Ok(inserted)) => failures.push(format!(
                "{shape}: inserted {inserted} triples, where it fails"
            )),
            Ok(Err(_)) => {}
        }
        if !db.rdf_store().is_empty() {
            failures.push(format!("{shape}: stored {:?}", default_graph(&db)));
        }
    }
    assert_no_failures(&failures);
}

/// `RdfStore::insert` does not store an invalid triple (it reports the
/// refusal; whether as `false` or an error is the implementation's choice,
/// this test only reads the store).
#[test]
fn rdf_store_insert_does_not_store_an_invalid_triple() {
    let mut failures = Vec::new();
    for (shape, invalid) in invalid_triples() {
        let db = GrafeoDB::new_in_memory();
        let store = db.rdf_store();
        if let Err(panic) = run(|| {
            let _ = store.insert(invalid.clone());
        }) {
            failures.push(format!("{shape}: {panic}"));
        }
        if store.contains(&invalid) || !store.is_empty() {
            failures.push(format!("{shape}: stored, where it is refused"));
        }
    }
    assert_no_failures(&failures);
}

// ── Controls: valid triples go in everywhere ────────────────────────────

/// Valid triples insert through `INSERT DATA`, a template, the batch insert
/// and the store, objects that are literals or blank nodes included.
#[test]
fn valid_triples_insert_through_every_path() {
    let db = GrafeoDB::new_in_memory();
    db.execute_sparql(
        r#"INSERT DATA { <http://ex.org/alix> <http://ex.org/name> "Alix" .
                         _:gus <http://ex.org/knows> _:vincent . }"#,
    )
    .unwrap();
    db.execute_sparql(
        r"INSERT { ?s <http://ex.org/city> <http://ex.org/amsterdam> }
          WHERE { ?s <http://ex.org/name> ?name }",
    )
    .unwrap();
    assert_eq!(db.batch_insert_rdf(vec![valid_triple()]).unwrap(), 1);
    assert!(db.rdf_store().insert(Triple::new(
        Term::iri(GUS),
        Term::iri(KNOWS),
        Term::literal("Mia")
    )));
    assert_eq!(db.rdf_store().len(), 5, "{:?}", default_graph(&db));
}

/// A database file of valid triples reopens with them all: the section
/// loader keeps loading what was written.
#[cfg(all(feature = "wal", feature = "grafeo-file"))]
#[test]
fn a_file_of_valid_triples_reopens() {
    use grafeo_engine::Config;

    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("valid.grafeo");
    let written = {
        let db = GrafeoDB::with_config(Config::persistent(&path)).unwrap();
        db.execute_sparql(
            r#"INSERT DATA { <http://ex.org/alix> <http://ex.org/name> "Alix" .
                             <http://ex.org/alix> <http://ex.org/knows> _:gus .
                             _:gus <http://ex.org/city> <http://ex.org/berlin> . }"#,
        )
        .unwrap();
        db.batch_insert_rdf(vec![valid_triple()]).unwrap();
        let written = default_graph(&db);
        db.close().unwrap();
        written
    };
    assert_eq!(written.len(), 4);
    let db = GrafeoDB::with_config(Config::persistent(&path)).unwrap();
    assert_eq!(db.rdf_store().len(), 4);
    assert_eq!(default_graph(&db).len(), 4);
    db.close().unwrap();
}
