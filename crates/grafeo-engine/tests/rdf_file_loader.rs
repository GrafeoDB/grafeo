//! `GrafeoDB::load_rdf(path, format)` (0.6.1): a Turtle or N-Triples file into
//! the default graph as one bulk write. Each load is checked against the same
//! triples written by hand as N-Triples and inserted with INSERT DATA:
//!
//! - Turtle with prefixes (both spellings), `a`, lists, blank nodes, language
//!   tags, typed and numeric literals, long strings and escapes;
//! - N-Triples with comments, blank lines and escapes;
//! - the format by extension in any case, a named format over the extension,
//!   an extension it cannot tell, the names `RdfFormat::from_name` takes;
//! - a line that does not parse inserts nothing and names its line;
//! - duplicates and triples already there count as nothing new;
//! - a file it cannot read, or that is not UTF-8;
//! - a persistent database: closed and reopened, and crashed after the load
//!   returned (the WAL replays it);
//! - a load while another session's transaction is open.
//!
//! (A read-only or closed database refuses before reading the file: see
//! `read_only_writes.rs` and `writes_after_close.rs`.)
//!
//! ```bash
//! cargo test -p grafeo-engine --all-features --test rdf_file_loader
//! ```

#![cfg(all(feature = "sparql", feature = "triple-store"))]

use std::path::Path;

use grafeo_common::types::Value;
use grafeo_engine::{GrafeoDB, RdfFormat};

/// Every triple of the default graph, sorted, with blank node labels left
/// out (two loads label them differently).
fn triples(db: &GrafeoDB) -> Vec<String> {
    use grafeo_common::types::rdf::Term;
    let mut rows: Vec<String> = db
        .execute_sparql("SELECT ?s ?p ?o WHERE { ?s ?p ?o }")
        .unwrap()
        .rows()
        .iter()
        .map(|row| {
            row.iter()
                .map(|value| match value {
                    Value::RdfTerm(term) => match term.as_ref() {
                        Term::BlankNode(_) => "_:".to_string(),
                        other => format!("{other:?}"),
                    },
                    other => format!("{other:?}"),
                })
                .collect::<Vec<_>>()
                .join(" ")
        })
        .collect();
    rows.sort();
    rows
}

/// A database with `ntriples` inserted through INSERT DATA.
fn inserted(ntriples: &str) -> GrafeoDB {
    let db = GrafeoDB::new_in_memory();
    db.execute_sparql(&format!("INSERT DATA {{ {ntriples} }}"))
        .unwrap();
    db
}

fn write(dir: &Path, name: &str, text: &str) -> std::path::PathBuf {
    let path = dir.join(name);
    std::fs::write(&path, text).unwrap();
    path
}

const TURTLE: &str = r#"@prefix ex: <http://ex.org/> .
@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .
PREFIX foaf: <http://xmlns.com/foaf/0.1/>

ex:alix a foaf:Person ;
    foaf:name "Alix" , "Alix"@nl ;
    ex:age 30 ;
    ex:height 1.72 ;
    ex:score 1.5e2 ;
    ex:active true ;
    ex:born "1994-05-01"^^xsd:date ;
    ex:likes ( ex:jazz ex:chess ) ;
    ex:address [ ex:city "Amsterdam" ; ex:zip "1011" ] ;
    ex:knows _:gus .
_:gus foaf:name """Gus
the second""" .
ex:mia ex:note "a \"quoted\" word" .
"#;

/// The 19 triples of [`TURTLE`], by hand.
const TURTLE_AS_NTRIPLES: &str = r#"
<http://ex.org/alix> <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> <http://xmlns.com/foaf/0.1/Person> .
<http://ex.org/alix> <http://xmlns.com/foaf/0.1/name> "Alix" .
<http://ex.org/alix> <http://xmlns.com/foaf/0.1/name> "Alix"@nl .
<http://ex.org/alix> <http://ex.org/age> "30"^^<http://www.w3.org/2001/XMLSchema#integer> .
<http://ex.org/alix> <http://ex.org/height> "1.72"^^<http://www.w3.org/2001/XMLSchema#decimal> .
<http://ex.org/alix> <http://ex.org/score> "1.5e2"^^<http://www.w3.org/2001/XMLSchema#double> .
<http://ex.org/alix> <http://ex.org/active> "true"^^<http://www.w3.org/2001/XMLSchema#boolean> .
<http://ex.org/alix> <http://ex.org/born> "1994-05-01"^^<http://www.w3.org/2001/XMLSchema#date> .
<http://ex.org/alix> <http://ex.org/likes> _:l1 .
_:l1 <http://www.w3.org/1999/02/22-rdf-syntax-ns#first> <http://ex.org/jazz> .
_:l1 <http://www.w3.org/1999/02/22-rdf-syntax-ns#rest> _:l2 .
_:l2 <http://www.w3.org/1999/02/22-rdf-syntax-ns#first> <http://ex.org/chess> .
_:l2 <http://www.w3.org/1999/02/22-rdf-syntax-ns#rest> <http://www.w3.org/1999/02/22-rdf-syntax-ns#nil> .
<http://ex.org/alix> <http://ex.org/address> _:a .
_:a <http://ex.org/city> "Amsterdam" .
_:a <http://ex.org/zip> "1011" .
<http://ex.org/alix> <http://ex.org/knows> _:gus .
_:gus <http://xmlns.com/foaf/0.1/name> "Gus\nthe second" .
<http://ex.org/mia> <http://ex.org/note> "a \"quoted\" word" .
"#;

const NTRIPLES: &str = r#"# people of Amsterdam
<http://ex.org/alix> <http://ex.org/name> "Alix" .

   # an indented comment
<http://ex.org/alix> <http://ex.org/city> "Amsterdam"@en .
_:b1 <http://ex.org/note> "café \"x\"\n" .
<http://ex.org/alix> <http://ex.org/age> "30"^^<http://www.w3.org/2001/XMLSchema#integer> .
"#;

#[test]
fn turtle_holds_what_its_triples_insert() {
    let dir = tempfile::tempdir().unwrap();
    let db = GrafeoDB::new_in_memory();
    let new = db
        .load_rdf(write(dir.path(), "people.ttl", TURTLE), None)
        .unwrap();
    assert_eq!(new, 19);
    assert_eq!(db.rdf_store().len(), 19);
    assert_eq!(triples(&db), triples(&inserted(TURTLE_AS_NTRIPLES)));
    // The same triples as an N-Triples file.
    let again = GrafeoDB::new_in_memory();
    again
        .load_rdf(write(dir.path(), "people.nt", TURTLE_AS_NTRIPLES), None)
        .unwrap();
    assert_eq!(triples(&again), triples(&db));
}

#[test]
fn ntriples_with_comments_blank_lines_and_escapes() {
    let dir = tempfile::tempdir().unwrap();
    let db = GrafeoDB::new_in_memory();
    let new = db
        .load_rdf(write(dir.path(), "people.nt", NTRIPLES), None)
        .unwrap();
    assert_eq!(new, 4);
    let without_comments: String = NTRIPLES
        .lines()
        .filter(|line| !line.trim().is_empty() && !line.trim().starts_with('#'))
        .collect::<Vec<_>>()
        .join("\n");
    assert_eq!(triples(&db), triples(&inserted(&without_comments)));
    let note = db
        .execute_sparql("SELECT (STR(?n) AS ?text) WHERE { ?b <http://ex.org/note> ?n }")
        .unwrap();
    let text = match &note.rows()[0][0] {
        Value::RdfTerm(term) => match term.as_ref() {
            grafeo_common::types::rdf::Term::Literal(literal) => literal.value().to_string(),
            other => panic!("{other:?}"),
        },
        other => panic!("{other:?}"),
    };
    assert_eq!(text, "caf\u{e9} \"x\"\n", "the escapes read");
}

#[test]
fn the_format_by_extension_or_by_name() {
    let dir = tempfile::tempdir().unwrap();
    let one = "<http://ex.org/alix> <http://ex.org/city> \"Amsterdam\" .\n";
    let prefixed = "@prefix ex: <http://ex.org/> .\nex:gus ex:city \"Berlin\" .\n";
    for (name, text, format) in [
        ("upper.TTL", prefixed, None),
        ("mixed.Nt", one, None),
        ("turtle.rdf", prefixed, Some(RdfFormat::Turtle)),
        ("plain", one, Some(RdfFormat::NTriples)),
        // A named format wins over the extension: Turtle in a .nt file.
        ("prefixed.nt", prefixed, Some(RdfFormat::Turtle)),
    ] {
        let db = GrafeoDB::new_in_memory();
        assert_eq!(
            db.load_rdf(write(dir.path(), name, text), format).unwrap(),
            1,
            "{name}"
        );
    }
    for (name, text) in [("data.rdf", one), ("noextension", one), ("data.json", one)] {
        let db = GrafeoDB::new_in_memory();
        let error = db
            .load_rdf(write(dir.path(), name, text), None)
            .expect_err(name)
            .to_string();
        assert!(error.contains("name the format"), "{name}: {error}");
        assert_eq!(db.rdf_store().len(), 0, "{name}");
    }
    // Turtle in a .nt file without a named format is no N-Triples.
    let db = GrafeoDB::new_in_memory();
    assert!(
        db.load_rdf(write(dir.path(), "turtle.nt", prefixed), None)
            .is_err()
    );
    assert_eq!(db.rdf_store().len(), 0);

    for (name, format) in [
        ("turtle", Some(RdfFormat::Turtle)),
        ("TTL", Some(RdfFormat::Turtle)),
        ("ntriples", Some(RdfFormat::NTriples)),
        ("N-Triples", Some(RdfFormat::NTriples)),
        ("nt", Some(RdfFormat::NTriples)),
        ("rdfxml", None),
        ("", None),
    ] {
        assert_eq!(RdfFormat::from_name(name), format, "{name}");
    }
    assert_eq!(
        RdfFormat::of_path(Path::new("a/b.Ttl")),
        Some(RdfFormat::Turtle)
    );
    assert_eq!(RdfFormat::of_path(Path::new("a/b.ttl.gz")), None);
}

/// A line that does not parse: nothing of the file is inserted, and the
/// error names the line.
#[test]
fn a_line_that_does_not_parse_inserts_nothing() {
    let dir = tempfile::tempdir().unwrap();
    let db = GrafeoDB::new_in_memory();
    let bad = concat!(
        "<http://ex.org/alix> <http://ex.org/city> \"Amsterdam\" .\n",
        "<http://ex.org/gus> <http://ex.org/city> \"Berlin\" .\n",
        "<http://ex.org/mia> <http://ex.org/city> Prague .\n",
        "<http://ex.org/vincent> <http://ex.org/city> \"Paris\" .\n",
    );
    let error = db
        .load_rdf(write(dir.path(), "bad.nt", bad), None)
        .unwrap_err()
        .to_string();
    assert!(error.contains('3'), "the third line: {error}");
    assert_eq!(db.rdf_store().len(), 0);

    let bad_turtle =
        "@prefix ex: <http://ex.org/> .\nex:alix ex:city \"Amsterdam\" .\nex:gus ex:city .\n";
    assert!(
        db.load_rdf(write(dir.path(), "bad.ttl", bad_turtle), None)
            .is_err()
    );
    assert_eq!(db.rdf_store().len(), 0);
}

/// Duplicates in the file, and triples already in the store, are nothing
/// new; a second load of the file adds nothing.
#[test]
fn duplicates_and_triples_already_there_count_as_nothing_new() {
    let dir = tempfile::tempdir().unwrap();
    let db = inserted("<http://ex.org/alix> <http://ex.org/city> \"Amsterdam\" .");
    let file = write(
        dir.path(),
        "cities.nt",
        concat!(
            "<http://ex.org/alix> <http://ex.org/city> \"Amsterdam\" .\n",
            "<http://ex.org/gus> <http://ex.org/city> \"Berlin\" .\n",
            "<http://ex.org/gus> <http://ex.org/city> \"Berlin\" .\n",
            "<http://ex.org/mia> <http://ex.org/city> \"Prague\" .\n",
        ),
    );
    assert_eq!(db.load_rdf(&file, None).unwrap(), 2, "Gus and Mia");
    assert_eq!(db.load_rdf(&file, None).unwrap(), 0);
    assert_eq!(db.rdf_store().len(), 3);
}

#[test]
fn a_file_it_cannot_read() {
    let dir = tempfile::tempdir().unwrap();
    let db = GrafeoDB::new_in_memory();
    let missing = db
        .load_rdf(dir.path().join("missing.nt"), None)
        .unwrap_err()
        .to_string();
    assert!(missing.contains("cannot read"), "{missing}");
    let latin1 = dir.path().join("latin1.nt");
    std::fs::write(
        &latin1,
        b"<http://ex.org/alix> <http://ex.org/note> \"caf\xe9\" .\n",
    )
    .unwrap();
    let error = db.load_rdf(&latin1, None).unwrap_err().to_string();
    assert!(error.contains("cannot read"), "not UTF-8: {error}");
    assert_eq!(db.rdf_store().len(), 0);
}

#[cfg(feature = "grafeo-file")]
#[test]
fn a_persistent_database_keeps_the_load() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("people.grafeo");
    let file = write(dir.path(), "people.ttl", TURTLE);
    {
        let db = GrafeoDB::open(&path).unwrap();
        assert_eq!(db.load_rdf(&file, None).unwrap(), 19);
        db.close().unwrap();
    }
    let db = GrafeoDB::open(&path).unwrap();
    assert_eq!(db.rdf_store().len(), 19);
    assert_eq!(triples(&db), triples(&inserted(TURTLE_AS_NTRIPLES)));
    db.close().unwrap();
}

/// A load while another session's transaction is open: the load completes,
/// and the transaction then commits its own triple beside it.
#[test]
fn a_load_beside_an_open_transaction() {
    let dir = tempfile::tempdir().unwrap();
    let db = GrafeoDB::new_in_memory();
    let mut session = db.session();
    session.begin_transaction().unwrap();
    session
        .execute_sparql("INSERT DATA { <http://ex.org/jules> <http://ex.org/city> \"Paris\" . }")
        .unwrap();
    assert_eq!(
        db.load_rdf(write(dir.path(), "people.nt", NTRIPLES), None)
            .unwrap(),
        4
    );
    session.commit().unwrap();
    assert_eq!(db.rdf_store().len(), 5);
}

/// A crash after the load returned: the reopen replays the WAL and holds
/// the file's triples.
#[cfg(all(feature = "wal", feature = "grafeo-file"))]
mod crash {
    use std::path::PathBuf;

    use grafeo_common::testing::child_process;
    use grafeo_engine::GrafeoDB;

    /// Tells the child process where its database and file are.
    const PATH_VAR: &str = "GRAFEO_RDF_FILE_LOADER_CRASH_DIR";

    #[test]
    fn a_load_that_returned_survives_a_crash() {
        if let Some(dir) = std::env::var_os(PATH_VAR) {
            let dir = PathBuf::from(dir);
            let db = GrafeoDB::open(dir.join("people.grafeo")).unwrap();
            let file = super::write(&dir, "people.ttl", super::TURTLE);
            assert_eq!(db.load_rdf(&file, None).unwrap(), 19);
            // Crash: no close(), no checkpoint, no destructors.
            std::process::exit(0);
        }
        let dir = tempfile::tempdir().unwrap();
        let status = child_process::run(
            std::process::Command::new(std::env::current_exe().unwrap())
                .args([
                    "--exact",
                    "crash::a_load_that_returned_survives_a_crash",
                    "--nocapture",
                ])
                .env(PATH_VAR, dir.path()),
        )
        .unwrap();
        assert!(status.success(), "the child process failed");
        assert!(
            dir.path().join("people.grafeo.wal").exists(),
            "the crash left the WAL for the reopen to replay"
        );
        let db = GrafeoDB::open(dir.path().join("people.grafeo")).unwrap();
        assert_eq!(db.rdf_store().len(), 19);
        assert_eq!(
            super::triples(&db),
            super::triples(&super::inserted(super::TURTLE_AS_NTRIPLES))
        );
        db.close().unwrap();
    }
}
