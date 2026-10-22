//! An RDF term stored as an LPG property (a SPARQL result written back to a
//! node) survives every way a database is saved and read back: a checkpoint
//! and a reopen, a read-only open, a crash with WAL replay, and a snapshot
//! export and import. Each term keeps its kind, its lexical form, and its
//! language tag or datatype.
//!
//! ```bash
//! cargo test -p grafeo-engine --all-features --test rdf_term_properties
//! ```

#![cfg(all(feature = "gql", feature = "grafeo-file", feature = "wal"))]

mod common;

use std::path::Path;
use std::sync::Arc;

use grafeo_common::types::Value;
use grafeo_common::types::rdf::Term;
use grafeo_engine::{Config, GrafeoDB};

fn open(path: &Path) -> GrafeoDB {
    GrafeoDB::with_config(Config::persistent(path)).unwrap()
}

/// One term of each kind, with the text a lossy encoding would change:
/// non-ASCII, quotes and a line break, an empty language tag's neighbour,
/// a custom datatype, and a blank node label with a space.
fn terms() -> Vec<Term> {
    vec![
        Term::iri("http://example.org/vocab/Kraków"),
        Term::blank("b0 x"),
        Term::literal("Gus \"de bus\"\nAmsterdam"),
        Term::lang_literal("Pomme", "fr"),
        Term::lang_literal("Apple", "en-GB"),
        Term::typed_literal("2026-01-15", "http://www.w3.org/2001/XMLSchema#date"),
        Term::typed_literal("01", "http://www.w3.org/2001/XMLSchema#integer"),
        Term::typed_literal("Praha", "http://example.org/types/city"),
    ]
}

fn write_terms(db: &GrafeoDB) {
    for (n, term) in terms().into_iter().enumerate() {
        let node = db.create_node(&["Concept"]).unwrap();
        db.set_node_property(node, "n", Value::Int64(i64::try_from(n).unwrap()))
            .unwrap();
        db.set_node_property(node, "term", Value::RdfTerm(Arc::new(term)))
            .unwrap();
    }
}

fn assert_terms(db: &GrafeoDB) {
    let result = db
        .execute("MATCH (c:Concept) RETURN c.term ORDER BY c.n")
        .unwrap();
    let stored: Vec<Value> = result.rows().iter().map(|row| row[0].clone()).collect();
    let expected: Vec<Value> = terms()
        .into_iter()
        .map(|term| Value::RdfTerm(Arc::new(term)))
        .collect();
    assert_eq!(stored, expected);
    // An RDF term is no string: a filter on the lexical form finds nothing
    let result = db
        .execute("MATCH (c:Concept) WHERE c.term = 'Pomme' RETURN count(*)")
        .unwrap();
    assert_eq!(result.rows()[0][0], Value::Int64(0));
}

#[test]
fn rdf_terms_survive_a_checkpoint_and_a_reopen() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("terms.grafeo");
    let db = open(&path);
    write_terms(&db);
    db.close().unwrap();
    drop(db);

    let db = open(&path);
    assert_terms(&db);
    db.close().unwrap();
    drop(db);

    let db = GrafeoDB::with_config(Config::read_only(&path)).unwrap();
    assert_terms(&db);
}

#[test]
fn rdf_terms_survive_a_crash_and_a_wal_replay() {
    let (_dir, db) = common::replay::reopened_after_crash(
        "rdf_terms_survive_a_crash_and_a_wal_replay",
        open,
        write_terms,
    );
    assert_terms(&db);
}

#[test]
fn rdf_terms_survive_a_snapshot() {
    let db = GrafeoDB::new_in_memory();
    write_terms(&db);
    let snapshot = db.export_snapshot().unwrap();
    let imported = GrafeoDB::import_snapshot(&snapshot).unwrap();
    assert_terms(&imported);
}
