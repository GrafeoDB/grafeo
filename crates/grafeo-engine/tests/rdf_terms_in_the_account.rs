//! RDF triples and their terms in the memory account (#718's item 19, the term-text count).
//!
//! The memory limit grants writes by `memory_usage()`'s account, so what the RDF store holds must
//! be in it: each triple's own allocation (its `Arc` header with it), the text of every IRI, blank
//! node and literal (its value, its language, a datatype of its own), once per distinct term, and
//! the text of a term held inline in the triple. The oracle is the allocator: this binary counts
//! every byte the process allocates and frees, as the requested layout sizes (an `Arc<str>`'s
//! padded layout, not a size class of the system allocator).
//!
//! Each kind of term is loaded alone, 20,000 triples of it with one predicate (so the predicate's
//! text is held once, not once per triple), then the steps that free what the load made: a load
//! into a named graph, its `CLEAR`, a load into it again, its `DROP`, and a delete of the default
//! graph's triples. After each step the allocator and the report moved by the same bytes, within
//! a fixed margin (a statement's plan, a dictionary's next room), and the account equals its walk.
//!
//! ```bash
//! cargo test -p grafeo-engine --all-features --test rdf_terms_in_the_account -- --test-threads 1
//! ```

#![cfg(all(feature = "sparql", feature = "triple-store"))]
#![expect(
    unsafe_code,
    reason = "GlobalAlloc is an unsafe trait; this one forwards every call to System"
)]
#![allow(
    deprecated,
    reason = "the account's walk is read from the RDF store, which no session shows"
)]

use std::alloc::{GlobalAlloc, Layout, System};
use std::fmt::Write;
use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};

use grafeo_engine::{GrafeoDB, RdfFormat};

static ALLOCATED: AtomicUsize = AtomicUsize::new(0);
static FREED: AtomicUsize = AtomicUsize::new(0);

/// Forwards to [`System`] and counts the bytes of each call.
struct Counting;

// SAFETY: every method forwards to `System` with the caller's arguments, so `System`'s
// guarantees are this allocator's; the counters are only read by the tests.
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        // SAFETY: the caller's contract for `alloc`, passed on.
        let pointer = unsafe { System.alloc(layout) };
        if !pointer.is_null() {
            ALLOCATED.fetch_add(layout.size(), Ordering::Relaxed);
        }
        pointer
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        // SAFETY: the caller's contract for `alloc_zeroed`, passed on.
        let pointer = unsafe { System.alloc_zeroed(layout) };
        if !pointer.is_null() {
            ALLOCATED.fetch_add(layout.size(), Ordering::Relaxed);
        }
        pointer
    }

    unsafe fn dealloc(&self, pointer: *mut u8, layout: Layout) {
        // SAFETY: the caller's contract for `dealloc`, passed on.
        unsafe { System.dealloc(pointer, layout) };
        FREED.fetch_add(layout.size(), Ordering::Relaxed);
    }

    unsafe fn realloc(&self, pointer: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        // SAFETY: the caller's contract for `realloc`, passed on.
        let moved = unsafe { System.realloc(pointer, layout, new_size) };
        if !moved.is_null() {
            ALLOCATED.fetch_add(new_size, Ordering::Relaxed);
            FREED.fetch_add(layout.size(), Ordering::Relaxed);
        }
        moved
    }
}

#[global_allocator]
static GLOBAL: Counting = Counting;

/// One measurement at a time: another test's allocations would land in its difference.
static SERIAL: Mutex<()> = Mutex::new(());

/// Starts what the process sets up once, at its first load, and no database holds (the thread
/// pool a parallel index build runs on, lazy statics): a load into a database of its own, which
/// it then drops. Every test calls it before its first measured step, so none counts that setup,
/// whichever test runs first or alone.
fn warm_up() {
    static ONCE: std::sync::Once = std::sync::Once::new();
    ONCE.call_once(|| {
        let db = GrafeoDB::new_in_memory();
        db.load_rdf_text(&text_of(kinds()[0].1, 0), RdfFormat::NTriples, None)
            .unwrap();
        drop(db);
    });
}

/// The bytes the process holds now.
fn held() -> isize {
    let allocated = ALLOCATED.load(Ordering::Relaxed);
    let freed = FREED.load(Ordering::Relaxed);
    isize::try_from(allocated).unwrap() - isize::try_from(freed).unwrap()
}

/// The triples of each load.
const TRIPLES: usize = 20_000;

/// What the allocator and the report may differ by over a step: the few kilobytes a statement
/// keeps (its cached plan, a dictionary's next room). A term text left out of the count, or one
/// per triple held twice, shows here as tens of bytes per triple, far above it.
const MARGIN: isize = 64 * 1024;

/// The XML Schema datatypes' namespace.
const XSD: &str = "http://www.w3.org/2001/XMLSchema#";

/// The named graph the steps load into, clear and drop.
const NAMED: &str = "http://ex.org/g";

/// One kind of term: its name, and the N-Triples line of triple `i`. Every line has the one
/// predicate `<http://ex.org/p>`.
type Kind = (&'static str, fn(usize) -> String);

fn kinds() -> [Kind; 9] {
    [
        ("IRIs", |i| {
            format!("<http://ex.org/s{i}> <http://ex.org/p> <http://ex.org/o{i}> .")
        }),
        ("blank nodes", |i| {
            format!("_:s{i} <http://ex.org/p> _:o{i} .")
        }),
        ("plain literals", |i| {
            format!("<http://ex.org/s{i}> <http://ex.org/p> \"a canal house, number {i}\" .")
        }),
        ("language-tagged literals", |i| {
            format!("<http://ex.org/s{i}> <http://ex.org/p> \"een grachtenpand, nummer {i}\"@nl .")
        }),
        ("literals of a shared datatype held inline", |i| {
            format!("<http://ex.org/s{i}> <http://ex.org/p> \"{i}\"^^<{XSD}integer> .")
        }),
        ("literals of a shared datatype not inlined", |i| {
            format!("<http://ex.org/s{i}> <http://ex.org/p> \"{i}.5e0\"^^<{XSD}double> .")
        }),
        ("literals of a datatype of their own", |i| {
            format!("<http://ex.org/s{i}> <http://ex.org/p> \"v{i}\"^^<http://ex.org/datatype> .")
        }),
        ("short strings held inline", |i| {
            format!("<http://ex.org/s{i}> <http://ex.org/p> \"s{}\" .", i % 1000)
        }),
        ("one object repeated over every triple", |i| {
            format!("<http://ex.org/s{i}> <http://ex.org/p> <http://ex.org/Amsterdam> .")
        }),
    ]
}

/// The N-Triples text of `TRIPLES` triples of `kind`, numbered from `from`.
fn text_of(kind: fn(usize) -> String, from: usize) -> String {
    let mut text = String::new();
    for i in from..from + TRIPLES {
        writeln!(text, "{}", kind(i)).unwrap();
    }
    text
}

/// What `step` moves: by the allocator, and in the report's total. The report is read before
/// each count, so its own allocations fall outside the difference.
fn moved(db: &GrafeoDB, step: impl FnOnce()) -> (isize, isize) {
    let report_before = isize::try_from(db.memory_usage().total_bytes).unwrap();
    let before = held();
    step();
    let after = held();
    let report_after = isize::try_from(db.memory_usage().total_bytes).unwrap();
    (after - before, report_after - report_before)
}

/// Each kind of term, alone: its load, a load into a named graph, the `CLEAR` of that graph, a
/// load into it again, its `DROP`, and a delete of the default graph's triples. After each the
/// allocator and the report moved alike, and the account equals its walk.
#[test]
fn every_kind_of_term_is_counted_as_the_allocator_holds_it() {
    let _serial = SERIAL
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    warm_up();
    let mut wrong = Vec::new();
    for (name, kind) in kinds() {
        let db = GrafeoDB::new_in_memory();
        // The text is the test's, made before each count, so no step holds it.
        let first = text_of(kind, 0);
        let second = text_of(kind, TRIPLES);
        let third = text_of(kind, 2 * TRIPLES);
        let steps: [(&str, Box<dyn FnOnce()>); 6] = [
            (
                "the load",
                Box::new(|| {
                    db.load_rdf_text(&first, RdfFormat::NTriples, None).unwrap();
                }),
            ),
            (
                "a load into a named graph",
                Box::new(|| {
                    db.load_rdf_text(&second, RdfFormat::NTriples, Some(NAMED))
                        .unwrap();
                }),
            ),
            (
                "the named graph's CLEAR",
                Box::new(|| {
                    db.execute_sparql(&format!("CLEAR GRAPH <{NAMED}>"))
                        .unwrap();
                }),
            ),
            (
                "a load into it again",
                Box::new(|| {
                    db.load_rdf_text(&third, RdfFormat::NTriples, Some(NAMED))
                        .unwrap();
                }),
            ),
            (
                "the named graph's DROP",
                Box::new(|| {
                    db.execute_sparql(&format!("DROP GRAPH <{NAMED}>")).unwrap();
                }),
            ),
            (
                "a delete of the default graph's triples",
                Box::new(|| {
                    db.execute_sparql("DELETE WHERE { ?s ?p ?o }").unwrap();
                }),
            ),
        ];
        for (step, run) in steps {
            let (allocated, reported) = moved(&db, run);
            if (allocated - reported).abs() > MARGIN {
                wrong.push(format!(
                    "{name}, {step}: {allocated} B by the allocator, {reported} B in the report"
                ));
            }
            let mismatches = db.rdf_store().account_mismatches();
            if !mismatches.is_empty() {
                wrong.push(format!(
                    "{name}, {step}: the account differs from its walk: {mismatches:?}"
                ));
            }
        }
        let left = db
            .execute_sparql("SELECT ?s WHERE { { ?s ?p ?o } UNION { GRAPH ?g { ?s ?p ?o } } }")
            .unwrap()
            .row_count();
        assert_eq!(left, 0, "{name}: every triple is gone");
    }
    assert!(wrong.is_empty(), "{}", wrong.join("\n"));
}

// ------------------------------------------------------------------ each string held once

/// The data pointers of the strings `term` holds: an IRI's or a blank node's text, a literal's
/// value, datatype and language.
fn text_pointers(term: &grafeo_core::graph::rdf::Term) -> Vec<*const u8> {
    use grafeo_core::graph::rdf::Term;
    match term {
        Term::Iri(iri) => vec![iri.as_str().as_ptr()],
        Term::BlankNode(node) => vec![node.id().as_ptr()],
        Term::Literal(literal) => {
            let mut pointers = vec![literal.value().as_ptr(), literal.datatype().as_ptr()];
            pointers.extend(literal.language().map(str::as_ptr));
            pointers
        }
        other => panic!("a term kind the test does not know: {other:?}"),
    }
}

/// Every stored triple of `db`'s default graph holds the dictionary's own strings for each of
/// its terms the dictionary holds (an inlined term has no entry, and holds its own): the same
/// allocation, not a copy of it. Returns how many triples it checked.
fn assert_strings_are_held_by_the_dictionary(db: &GrafeoDB, what: &str) -> usize {
    use grafeo_core::graph::rdf::{LatestUse, RdfView};

    let store = db.rdf_store();
    let dictionary = store.dictionary();
    let triples = store.triples(&RdfView::latest(LatestUse::Test));
    for triple in &triples {
        for term in [triple.subject(), triple.predicate(), triple.object()] {
            let Some(id) = dictionary.lookup(term) else {
                continue;
            };
            let Some(entry) = dictionary.term(id) else {
                // An inlined term: no entry to share.
                continue;
            };
            assert_eq!(
                text_pointers(term),
                text_pointers(&entry),
                "{what}: {term:?} holds a copy of the dictionary's strings"
            );
        }
    }
    triples.len()
}

/// The triples of the repeated-term cases: each subject its own, one predicate and one object
/// IRI for all, and a language-tagged literal of one value, so most of each triple's text repeats.
fn repeated(i: usize) -> grafeo_core::graph::rdf::Triple {
    use grafeo_core::graph::rdf::{Literal, Term, Triple};
    let object = if i.is_multiple_of(2) {
        Term::iri("http://ex.org/Amsterdam")
    } else {
        Term::Literal(Literal::with_language("een grachtenpand", "nl"))
    };
    Triple::new(
        Term::iri(format!("http://ex.org/s{i}")),
        Term::iri("http://ex.org/p"),
        object,
    )
}

/// Repeated terms, written by an insert: each stored triple holds the dictionary's strings, and
/// the allocator and the report agree, so a repeated term's text is held and counted once.
#[test]
fn repeated_terms_by_insert_are_held_once() {
    let _serial = SERIAL
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    warm_up();
    let db = GrafeoDB::new_in_memory();
    // The input is built and freed inside the step: built outside, its free would land in it.
    let (allocated, reported) = moved(&db, || {
        db.batch_insert_rdf((0..TRIPLES).map(repeated).collect::<Vec<_>>())
            .unwrap();
    });
    assert!(
        (allocated - reported).abs() <= MARGIN,
        "{allocated} B by the allocator, {reported} B in the report"
    );
    assert_eq!(
        assert_strings_are_held_by_the_dictionary(&db, "an insert"),
        TRIPLES
    );
}

/// Repeated terms, written by a load of N-Triples text: as by an insert.
#[test]
fn repeated_terms_by_load_are_held_once() {
    let _serial = SERIAL
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    warm_up();
    let db = GrafeoDB::new_in_memory();
    let mut text = String::new();
    for i in 0..TRIPLES {
        let object = if i.is_multiple_of(2) {
            "<http://ex.org/Amsterdam>"
        } else {
            "\"een grachtenpand\"@nl"
        };
        writeln!(text, "<http://ex.org/s{i}> <http://ex.org/p> {object} .").unwrap();
    }
    let (allocated, reported) = moved(&db, || {
        db.load_rdf_text(&text, RdfFormat::NTriples, None).unwrap();
    });
    assert!(
        (allocated - reported).abs() <= MARGIN,
        "{allocated} B by the allocator, {reported} B in the report"
    );
    assert_eq!(
        assert_strings_are_held_by_the_dictionary(&db, "a load"),
        TRIPLES
    );
}

/// Repeated terms after a reopen, which loads the triples from the file's sections: each stored
/// triple holds the dictionary's strings, and the reopened store's account equals its walk.
#[cfg(feature = "grafeo-file")]
#[test]
fn repeated_terms_after_a_reopen_are_held_once() {
    let _serial = SERIAL
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    warm_up();
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("terms.grafeo");
    {
        let db = GrafeoDB::open(&path).unwrap();
        db.batch_insert_rdf((0..TRIPLES).map(repeated)).unwrap();
        db.close().unwrap();
    }
    let db = GrafeoDB::open(&path).unwrap();
    assert_eq!(
        assert_strings_are_held_by_the_dictionary(&db, "after a reopen"),
        TRIPLES
    );
    assert_eq!(db.rdf_store().account_mismatches(), Vec::new());
    db.close().unwrap();
}
