//! RDF terms in the database file: the RDF section stores the dataset's term
//! ids with a term table (version 4), and the ring section refers to that
//! table. Every term comes back exact after a checkpoint, a reopen, WAL
//! replay and a crash; a term a query makes (a BIND result, a fresh blank
//! node) is written as a term of the dataset; a 0.6.0-beta.1 file (version 3
//! sections, N-Triples strings) is rewritten by its first read-write open,
//! which keeps a copy as `<path>.pre-0.6.0`.
//!
//! The golden fixture `fixtures/rdf/0.6.0/terms.grafeo` pins the format (see
//! the README there); `fixtures/rdf/0.6.0-beta.1/triples.grafeo` is a file of
//! the version before.
//!
//! ```bash
//! cargo test -p grafeo-engine --all-features --test rdf_term_ids_on_disk
//! ```

#![cfg(all(
    feature = "sparql",
    feature = "triple-store",
    feature = "wal",
    feature = "grafeo-file",
    not(miri)
))]

use std::path::{Path, PathBuf};

use grafeo_common::storage::section::SectionType;
use grafeo_common::testing::child_process;
use grafeo_common::types::Value;
use grafeo_engine::{Config, GrafeoDB};

const SCENARIO_VAR: &str = "GRAFEO_RDF_TERM_IDS_SCENARIO";
const PATH_VAR: &str = "GRAFEO_RDF_TERM_IDS_PATH";
/// The exit code of a child whose `close()` ran to completion.
const CLOSE_COMPLETED: i32 = 3;

/// The golden fixture, written by [`write_terms`].
const GOLDEN: &str = "tests/fixtures/rdf/0.6.0/terms.grafeo";
/// A 0.6.0-beta.1 file: version 3 RDF and ring sections.
const BETA: &str = "tests/fixtures/rdf/0.6.0-beta.1/triples.grafeo";

/// Every kind of term: each literal the ids inline (an integer, a decimal, a
/// boolean, a date, dateTimes with `Z` and without an offset, a short
/// string), a non-canonical integer, a double, a long string, a tagged
/// literal, a literal of another datatype, a blank node, in the default
/// graph and a named one.
const TERMS: &str = r#"PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>
INSERT DATA {
    <http://example.org/alix> <http://example.org/age> 30 ;
        <http://example.org/rank> "019"^^xsd:integer ;
        <http://example.org/price> 1.50 ;
        <http://example.org/active> true ;
        <http://example.org/born> "1996-01-15"^^xsd:date ;
        <http://example.org/seen> "2026-01-15T10:30:00Z"^^xsd:dateTime ;
        <http://example.org/local> "2026-01-15T10:30:00.5"^^xsd:dateTime ;
        <http://example.org/name> "Alix" ;
        <http://example.org/bio> "Alix lives in Amsterdam and knows Gus." ;
        <http://example.org/label> "Pomme"@fr ;
        <http://example.org/code> "x"^^<http://example.org/dt> ;
        <http://example.org/weight> "1.5e0"^^xsd:double ;
        <http://example.org/knows> _:gus .
    GRAPH <http://example.org/trips> {
        _:gus <http://example.org/visited> <http://example.org/prague> ;
            <http://example.org/when> "2026-03-01"^^xsd:date ;
            <http://example.org/note> "Kraków \"next\""@pl .
    }
}"#;

/// More terms after a checkpoint (the WAL holds them): new dictionary terms
/// and literals the first write has, and terms a query makes.
const MORE: &str = r#"PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>
INSERT DATA {
    <http://example.org/gus> <http://example.org/age> 30 ;
        <http://example.org/name> "Gus" ;
        <http://example.org/rank> "0019"^^xsd:integer ;
        <http://example.org/label> "Poire"@fr .
}"#;

/// Terms a query makes, written back: a BIND result and a fresh blank node.
const MADE: &str = r#"INSERT {
    ?s <http://example.org/greeting> ?greeting ;
        <http://example.org/card> ?card .
} WHERE {
    ?s <http://example.org/name> ?name .
    BIND(CONCAT("Hello, ", ?name) AS ?greeting)
    BIND(BNODE() AS ?card)
}"#;

/// Deletes a triple whose literal no other triple has.
const DELETE: &str = r#"DELETE DATA {
    <http://example.org/alix> <http://example.org/bio> "Alix lives in Amsterdam and knows Gus." .
}"#;

fn open(path: &Path) -> GrafeoDB {
    GrafeoDB::with_config(Config::persistent(path)).unwrap()
}

fn run(db: &GrafeoDB, updates: &[&str]) {
    for update in updates {
        db.execute_sparql(update).unwrap();
    }
}

/// Every triple of every graph, as `graph subject predicate object` in
/// N-Triples (`default` for the default graph), sorted; a blank node's
/// label as it is.
fn every_triple(db: &GrafeoDB) -> Vec<String> {
    let result = db
        .execute_sparql("SELECT ?g ?s ?p ?o WHERE { { ?s ?p ?o } UNION { GRAPH ?g { ?s ?p ?o } } }")
        .unwrap();
    let mut lines: Vec<String> = result
        .rows()
        .iter()
        .map(|row| {
            row.iter()
                .map(|value| match value {
                    Value::RdfTerm(term) => term.to_ntriples(),
                    Value::Null => "default".to_string(),
                    other => panic!("a SPARQL result holds terms, not {other:?}"),
                })
                .collect::<Vec<_>>()
                .join(" ")
        })
        .collect();
    lines.sort();
    lines
}

/// The triples of `updates`, as an in-memory database holds them.
fn expected(updates: &[&str]) -> Vec<String> {
    let db = GrafeoDB::new_in_memory();
    run(&db, updates);
    every_triple(&db)
}

/// The triples, with the label of each blank node replaced by its order of
/// first appearance: a query's fresh blank nodes have labels of their own.
fn without_blank_labels(lines: Vec<String>) -> Vec<String> {
    let mut labels: Vec<String> = Vec::new();
    let mut lines: Vec<String> = lines
        .into_iter()
        .map(|line| {
            line.split(' ')
                .map(|part| match part.strip_prefix("_:") {
                    Some(label) => {
                        let index = labels
                            .iter()
                            .position(|known| known == label)
                            .unwrap_or_else(|| {
                                labels.push(label.to_string());
                                labels.len() - 1
                            });
                        format!("_:b{index}")
                    }
                    None => part.to_string(),
                })
                .collect::<Vec<_>>()
                .join(" ")
        })
        .collect();
    lines.sort();
    lines
}

/// The versions of the RDF and ring sections of the file at `path`.
fn section_versions(path: &Path) -> Vec<(SectionType, u8)> {
    let manager = grafeo_storage::file::GrafeoFileManager::open_read_only(path, None).unwrap();
    let versions = manager
        .read_image(|image| {
            Ok([SectionType::RdfStore, SectionType::RdfRing]
                .into_iter()
                .filter_map(|section| {
                    image
                        .section_source(section)
                        .map(|source| (section, source.section_version()))
                })
                .collect())
        })
        .unwrap();
    manager.close().unwrap();
    versions
}

/// The chunks of the RDF and ring sections of the file at `path`, with
/// their bytes.
fn rdf_chunks(path: &Path) -> Vec<(SectionType, Vec<u8>)> {
    let manager = grafeo_storage::file::GrafeoFileManager::open_read_only(path, None).unwrap();
    let chunks = manager
        .read_image(|image| {
            let mut chunks = Vec::new();
            for section in [SectionType::RdfStore, SectionType::RdfRing] {
                if let Some(source) = image.section_source(section) {
                    for index in 0..source.chunks().len() {
                        chunks.push((section, source.fetch(index)?.to_vec()));
                    }
                }
            }
            Ok(chunks)
        })
        .unwrap();
    manager.close().unwrap();
    chunks
}

fn with_suffix(path: &Path, suffix: &str) -> PathBuf {
    let mut name = path.as_os_str().to_owned();
    name.push(suffix);
    PathBuf::from(name)
}

/// A copy of fixture `fixture` in a new directory, and the path of the copy.
fn copy_of(fixture: &str) -> (tempfile::TempDir, PathBuf) {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("database.grafeo");
    std::fs::copy(Path::new(env!("CARGO_MANIFEST_DIR")).join(fixture), &path).unwrap();
    (dir, path)
}

// ---------------------------------------------------------------------------
// The golden fixture
// ---------------------------------------------------------------------------

/// Writes the golden fixture's database at `path`: [`TERMS`], the ring over
/// the triples, `close()`.
#[cfg(feature = "ring-index")]
fn write_terms(path: &Path) {
    let db = open(path);
    run(&db, &[TERMS]);
    db.rdf_store().rebuild_ring();
    db.close().unwrap();
}

/// Writes the golden fixture again at the path `GRAFEO_WRITE_RDF_TERMS_FIXTURE`
/// names; a no-op without it.
#[cfg(feature = "ring-index")]
#[test]
fn write_the_golden_fixture() {
    if let Some(path) = std::env::var_os("GRAFEO_WRITE_RDF_TERMS_FIXTURE") {
        write_terms(Path::new(&path));
    }
}

/// The golden fixture holds every kind of term, exactly, in version 4 RDF and
/// ring sections; and the same steps write the same bytes, so a change of
/// the format shows up here (write the fixture again only with a format
/// change, which a release names). A blank node's label names the query that
/// made it by a count of the process's SPARQL queries, so the steps run in a
/// child process, as they did for the fixture: `_:gus` is `_:q0_gus` in both.
#[cfg(feature = "ring-index")]
#[test]
fn the_golden_fixture_pins_the_format() {
    let (_dir, path) = copy_of(GOLDEN);
    assert_eq!(
        section_versions(&path),
        [(SectionType::RdfStore, 4), (SectionType::RdfRing, 4)]
    );
    let golden = rdf_chunks(&path);
    let db = GrafeoDB::with_config(Config::read_only(&path)).unwrap();
    let triples = every_triple(&db);
    assert!(
        triples.iter().any(|line| line.contains("_:q0_gus")),
        "the blank node keeps its label"
    );
    assert_eq!(
        without_blank_labels(triples),
        without_blank_labels(expected(&[TERMS]))
    );
    assert!(db.rdf_store().ring().is_some(), "the ring is read");
    drop(db);

    let dir = tempfile::tempdir().unwrap();
    let written = dir.path().join("terms.grafeo");
    assert_eq!(in_child("write_terms", &written), 0);
    assert!(
        rdf_chunks(&written) == golden,
        "the same steps write the same RDF and ring chunks"
    );
}

// ---------------------------------------------------------------------------
// Checkpoint, reopen, replay, crash
// ---------------------------------------------------------------------------

/// Every term comes back after a checkpoint and a reopen, also after a
/// delete that leaves a term to no triple; the terms a query made are
/// written as the dataset's.
#[test]
fn every_term_survives_a_checkpoint_and_a_reopen() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("terms.grafeo");
    let db = open(&path);
    run(&db, &[TERMS, MORE, MADE]);
    let before = every_triple(&db);
    assert_eq!(
        without_blank_labels(before.clone()),
        without_blank_labels(expected(&[TERMS, MORE, MADE]))
    );
    db.wal_checkpoint().unwrap();
    run(&db, &[DELETE]);
    db.close().unwrap();
    drop(db);

    assert_eq!(
        section_versions(&path)[0],
        (SectionType::RdfStore, 4),
        "the file holds term ids"
    );
    let db = open(&path);
    let after = every_triple(&db);
    let mut want = before;
    want.retain(|line| !line.contains("Alix lives in Amsterdam"));
    assert_eq!(after, want, "the blank nodes keep their labels too");
    // A term the reopened dataset holds answers a pattern of its own.
    let rank = db
        .execute_sparql(
            "ASK { <http://example.org/gus> <http://example.org/rank> \
             \"0019\"^^<http://www.w3.org/2001/XMLSchema#integer> }",
        )
        .unwrap();
    assert_eq!(rank.rows()[0][0], Value::Bool(true));
    db.close().unwrap();
}

/// Runs `scenario` of [`crash_child`] in a child process on the database at
/// `path`, and returns its exit code.
fn in_child(scenario: &str, path: &Path) -> i32 {
    let status = child_process::run(
        std::process::Command::new(std::env::current_exe().unwrap())
            .args(["--exact", "crash_child", "--nocapture"])
            .env(SCENARIO_VAR, scenario)
            .env(PATH_VAR, path),
    )
    .unwrap();
    status
        .code()
        .unwrap_or_else(|| panic!("scenario {scenario}: the child was killed: {status}"))
}

/// Child-process entry for [`in_child`]; a no-op when run directly. Each
/// scenario but `write_terms` ends in a crash: an exit without `close()` and
/// without destructors.
#[test]
fn crash_child() {
    let Ok(scenario) = std::env::var(SCENARIO_VAR) else {
        return;
    };
    let path = PathBuf::from(std::env::var_os(PATH_VAR).unwrap());
    match scenario.as_str() {
        // The golden fixture's steps, with a clean close (not a crash).
        #[cfg(feature = "ring-index")]
        "write_terms" => write_terms(&path),
        // The file holds the first write, the WAL the rest.
        "checkpoint_then_more" => {
            let db = open(&path);
            run(&db, &[TERMS]);
            db.wal_checkpoint().unwrap();
            run(&db, &[MORE, MADE, DELETE]);
            // Exits while `db` is alive: dropping it would close it.
            std::process::exit(0);
        }
        // A crash at point N of the `close()` that follows the same writes.
        // The database stays outside the closure, so the injected panic
        // does not drop (and so close) it.
        #[cfg(feature = "testing-crash-injection")]
        other if other.starts_with("close:") => {
            use grafeo_common::testing::crash::{CrashResult, with_crash_at};
            let point: u64 = other["close:".len()..].parse().unwrap();
            let db = open(&path);
            run(&db, &[TERMS]);
            db.wal_checkpoint().unwrap();
            run(&db, &[MORE, MADE, DELETE]);
            let borrowed = std::panic::AssertUnwindSafe(&db);
            if let CrashResult::Completed(closed) = with_crash_at(point, move || borrowed.close()) {
                closed.unwrap();
                std::process::exit(CLOSE_COMPLETED);
            }
        }
        // A crash at point N of the first read-write open of a beta.1 file:
        // keeping its copy, then the checkpoint that rewrites its sections.
        #[cfg(feature = "testing-crash-injection")]
        other if other.starts_with("upgrade:") => {
            use grafeo_common::testing::crash::{CrashResult, with_crash_at};
            let point: u64 = other["upgrade:".len()..].parse().unwrap();
            // The open's checkpoint runs once the database exists: a panic
            // unwinding from it would drop (and so close) the database. The
            // crash exits at the panic instead.
            std::panic::set_hook(Box::new(|info| {
                eprintln!("crashed: {info}");
                std::process::exit(0);
            }));
            let opened = with_crash_at(point, || GrafeoDB::with_config(Config::persistent(&path)));
            if let CrashResult::Completed(db) = opened {
                let db = db.unwrap();
                db.close().unwrap();
                std::process::exit(CLOSE_COMPLETED);
            }
        }
        other => panic!("unknown scenario {other}"),
    }
    std::process::exit(0);
}

/// The triples after the writes of the crash scenarios.
fn after_the_writes(db: &GrafeoDB) -> Vec<String> {
    without_blank_labels(every_triple(db))
}

fn expected_after_the_writes() -> Vec<String> {
    without_blank_labels(expected(&[TERMS, MORE, MADE, DELETE]))
}

/// A crash after a checkpoint: the file holds term ids, the WAL the later
/// writes as terms; the reopen replays them onto the file's ids, and every
/// term comes back.
#[test]
fn every_term_survives_a_crash_and_the_replay() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("terms.grafeo");
    assert_eq!(in_child("checkpoint_then_more", &path), 0);
    assert!(
        with_suffix(&path, ".wal").exists(),
        "the crash left the sidecar WAL"
    );
    let db = open(&path);
    assert_eq!(after_the_writes(&db), expected_after_the_writes());
    db.close().unwrap();
    drop(db);
    let db = open(&path);
    assert_eq!(
        after_the_writes(&db),
        expected_after_the_writes(),
        "and after a clean close"
    );
    db.close().unwrap();
}

/// A crash at every point of the `close()` that writes the term table and
/// the ids: the reopen finds every term, whichever image or WAL it reads.
#[cfg(feature = "testing-crash-injection")]
#[test]
fn every_term_survives_a_crash_at_any_point_of_a_checkpoint() {
    for point in 1.. {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("terms.grafeo");
        let code = in_child(&format!("close:{point}"), &path);
        let db = open(&path);
        assert_eq!(
            after_the_writes(&db),
            expected_after_the_writes(),
            "after a crash at point {point} of close()"
        );
        db.close().unwrap();
        if code == CLOSE_COMPLETED {
            assert!(point > 1, "close() has crash points");
            break;
        }
        assert_eq!(code, 0, "point {point}");
    }
}

// ---------------------------------------------------------------------------
// A 0.6.0-beta.1 file
// ---------------------------------------------------------------------------

/// What the beta.1 fixture holds (see its README): Alix knows Gus in the
/// default graph, Gus visited Prague in `trips`.
fn beta_triples() -> Vec<String> {
    let mut lines = vec![
        "default <http://example.org/alix> <http://example.org/knows> <http://example.org/gus>"
            .to_string(),
        "<http://example.org/trips> <http://example.org/gus> <http://example.org/visited> \
         <http://example.org/prague>"
            .to_string(),
    ];
    lines.sort();
    lines
}

/// A read-write open rewrites the RDF sections of a beta.1 file in version
/// 4 and keeps the file it found, byte for byte, as `<path>.pre-0.6.0`; the
/// next open neither copies nor rewrites again.
#[test]
fn a_beta_file_is_rewritten_and_its_copy_kept() {
    let (_dir, path) = copy_of(BETA);
    let original = std::fs::read(&path).unwrap();
    assert!(
        section_versions(&path)
            .iter()
            .all(|(_, version)| *version == 3),
        "the fixture is of 0.6.0-beta.1"
    );
    let db = open(&path);
    assert_eq!(every_triple(&db), beta_triples());
    #[cfg(feature = "ring-index")]
    assert!(db.rdf_store().ring().is_some(), "the ring is read");
    db.close().unwrap();
    drop(db);

    let kept = with_suffix(&path, ".pre-0.6.0");
    assert_eq!(
        std::fs::read(&kept).unwrap(),
        original,
        "the kept copy is the old file"
    );
    assert!(
        section_versions(&path)
            .iter()
            .all(|(_, version)| *version == 4),
        "the open wrote the current version: {:?}",
        section_versions(&path)
    );
    let db = open(&path);
    assert_eq!(every_triple(&db), beta_triples());
    db.close().unwrap();
    drop(db);
    assert_eq!(
        std::fs::read(&kept).unwrap(),
        original,
        "the copy is never replaced"
    );
}

/// A read-only open reads a beta.1 file as it is: no copy, no rewrite.
#[test]
fn a_read_only_open_leaves_a_beta_file_alone() {
    let (_dir, path) = copy_of(BETA);
    let original = std::fs::read(&path).unwrap();
    let db = GrafeoDB::with_config(Config::read_only(&path)).unwrap();
    assert_eq!(every_triple(&db), beta_triples());
    drop(db);
    assert_eq!(std::fs::read(&path).unwrap(), original);
    assert!(!with_suffix(&path, ".pre-0.6.0").exists());
}

/// A crash at any point of the first open of a beta.1 file leaves either the
/// file as it was (with a whole copy or none) or the rewritten one with its
/// copy: the next open finishes, and the kept copy is the old file.
#[cfg(feature = "testing-crash-injection")]
#[test]
fn a_crash_while_a_beta_file_is_rewritten_is_finished_by_the_next_open() {
    for point in 1.. {
        let (_dir, path) = copy_of(BETA);
        let original = std::fs::read(&path).unwrap();
        let code = in_child(&format!("upgrade:{point}"), &path);
        assert!(
            !with_suffix(&path, ".pre-0.6.0.partial").exists()
                || !with_suffix(&path, ".pre-0.6.0").exists(),
            "point {point}: a partial copy never stands next to a whole one"
        );
        let db = open(&path);
        assert_eq!(
            every_triple(&db),
            beta_triples(),
            "after a crash at point {point}"
        );
        db.close().unwrap();
        drop(db);
        assert_eq!(
            std::fs::read(with_suffix(&path, ".pre-0.6.0")).unwrap(),
            original,
            "point {point}: the kept copy is the old file"
        );
        assert!(
            section_versions(&path)
                .iter()
                .all(|(_, version)| *version == 4)
        );
        if code == CLOSE_COMPLETED {
            assert!(point > 1, "the open has crash points");
            break;
        }
        assert_eq!(code, 0, "point {point}");
    }
}
