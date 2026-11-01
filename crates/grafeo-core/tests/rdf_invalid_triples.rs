//! The N-Triples and Turtle loaders refuse a triple whose subject is a
//! literal, or whose predicate is a blank node or a literal, with an error,
//! in debug and release builds alike, and load nothing of the document. The
//! N-Triples loader built such a line with `Triple::new`, whose
//! `debug_assert` panicked in debug builds; release builds stored it.
//!
//! ```bash
//! cargo test -p grafeo-core --all-features --test rdf_invalid_triples
//! cargo test -p grafeo-core --all-features --release --test rdf_invalid_triples
//! ```

#![cfg(feature = "triple-store")]

use std::io::Cursor;
use std::panic::{AssertUnwindSafe, catch_unwind};

use grafeo_core::graph::rdf::{RdfStore, Term, Triple};

/// A valid line before the invalid one: the load refuses the whole document.
const VALID_LINE: &str = "<http://ex.org/alix> <http://ex.org/knows> <http://ex.org/gus> .";

/// Lines with a literal subject, a blank predicate or a literal predicate.
const INVALID_LINES: &[(&str, &str)] = &[
    (
        "a literal subject",
        r#""Alix" <http://ex.org/knows> <http://ex.org/gus> ."#,
    ),
    (
        "a language-tagged literal subject",
        r#""Alix"@nl <http://ex.org/knows> <http://ex.org/gus> ."#,
    ),
    (
        "a typed literal subject",
        r#""3"^^<http://www.w3.org/2001/XMLSchema#integer> <http://ex.org/knows> <http://ex.org/gus> ."#,
    ),
    (
        "a blank predicate",
        "<http://ex.org/alix> _:knows <http://ex.org/gus> .",
    ),
    (
        "a literal predicate",
        r#"<http://ex.org/alix> "knows" <http://ex.org/gus> ."#,
    ),
];

/// Runs `f`; a panic comes back as an error with its message, so a test
/// reports every case that fails: an invalid triple is an error, never a
/// panic.
fn run<T>(f: impl FnOnce() -> T) -> Result<T, String> {
    catch_unwind(AssertUnwindSafe(f)).map_err(|payload| {
        let message = payload
            .downcast_ref::<&str>()
            .map(ToString::to_string)
            .or_else(|| payload.downcast_ref::<String>().cloned())
            .unwrap_or_default();
        format!("panicked ({message}), where it fails with an error")
    })
}

/// Fails the test with every failure of its cases, one a line.
fn assert_no_failures(failures: &[String]) {
    assert!(failures.is_empty(), "\n{}", failures.join("\n"));
}

#[test]
fn load_ntriples_refuses_an_invalid_triple_with_an_error() {
    let mut failures = Vec::new();
    for (shape, line) in INVALID_LINES {
        let store = RdfStore::new();
        let document = format!("{VALID_LINE}\n{line}\n");
        match run(|| store.load_ntriples(Cursor::new(document.as_bytes()))) {
            Err(panic) => failures.push(format!("{shape}: {panic}")),
            Ok(Ok(loaded)) => failures.push(format!(
                "{shape}: loaded {} triples, where it fails",
                loaded.triple_count
            )),
            Ok(Err(error)) => {
                let error = error.to_string();
                if !error.contains("line 2") {
                    failures.push(format!("{shape}: the error names no line 2: {error}"));
                }
            }
        }
        if !store.is_empty() {
            failures.push(format!("{shape}: {} triples stored", store.len()));
        }
    }
    assert_no_failures(&failures);
}

/// Turtle's grammar has no literal subject and no blank or literal
/// predicate: each is a parse error. (The streaming loader may have
/// inserted the batches before the error, so only its error is checked.)
#[test]
fn turtle_loaders_refuse_an_invalid_triple_with_an_error() {
    let mut failures = Vec::new();
    for (shape, line) in INVALID_LINES {
        let document = format!("{VALID_LINE}\n{line}\n");
        let store = RdfStore::new();
        match run(|| store.load_turtle(&document)) {
            Err(panic) => failures.push(format!("load_turtle, {shape}: {panic}")),
            Ok(Ok(_)) => failures.push(format!("load_turtle, {shape}: succeeded")),
            Ok(Err(_)) => {}
        }
        if !store.is_empty() {
            failures.push(format!("load_turtle, {shape}: {} stored", store.len()));
        }

        let store = RdfStore::new();
        match run(|| store.load_turtle_streaming(&document, 1)) {
            Err(panic) => failures.push(format!("load_turtle_streaming, {shape}: {panic}")),
            Ok(Ok(_)) => failures.push(format!("load_turtle_streaming, {shape}: succeeded")),
            Ok(Err(_)) => {}
        }
    }
    assert_no_failures(&failures);
}

/// Control: valid lines load, objects that are literals or blank nodes and
/// blank subjects included.
#[test]
fn valid_documents_load() {
    let document = format!(
        "{VALID_LINE}\n\
         <http://ex.org/alix> <http://ex.org/name> \"Alix\"@nl .\n\
         _:gus <http://ex.org/city> <http://ex.org/amsterdam> .\n\
         <http://ex.org/alix> <http://ex.org/knows> _:gus .\n"
    );
    let store = RdfStore::new();
    let loaded = store
        .load_ntriples(Cursor::new(document.as_bytes()))
        .unwrap();
    assert_eq!(loaded.triple_count, 4);
    assert!(store.contains(&Triple::new(
        Term::iri("http://ex.org/alix"),
        Term::iri("http://ex.org/knows"),
        Term::iri("http://ex.org/gus"),
    )));

    let store = RdfStore::new();
    assert_eq!(store.load_turtle(&document).unwrap().triple_count, 4);
    let store = RdfStore::new();
    assert_eq!(store.load_turtle_streaming(&document, 1).unwrap(), 4);
}
