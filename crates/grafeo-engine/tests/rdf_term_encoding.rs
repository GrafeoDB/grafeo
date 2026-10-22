//! RDF terms with non-ASCII text survive every way a database keeps them: a
//! checkpoint (the RDF section in chunks), the WAL replay after a crash, a
//! snapshot and a copy in memory. 0.5.x read terms back byte by byte, so
//! "Kraków" came back mangled and a query for it found nothing.
//!
//! ```bash
//! cargo test -p grafeo-engine --all-features --test rdf_term_encoding
//! ```

#![cfg(all(feature = "sparql", feature = "triple-store"))]

#[cfg(all(feature = "wal", feature = "grafeo-file"))]
mod common;

use grafeo_common::types::Value;
use grafeo_engine::{Config, GrafeoDB, GraphModel};

/// Triples with non-ASCII literals, language tags and an IRI, in the default
/// graph and a named graph.
const INSERT: &str = r#"INSERT DATA {
    <http://example.org/krakow> <http://example.org/name> "Kraków" .
    <http://example.org/krakow> <http://example.org/name> "Krakau"@de .
    <http://example.org/amsterdam> <http://example.org/name> "Ámsterdam"@es .
    <http://example.org/amsterdam> <http://example.org/name> "阿姆斯特丹"@zh .
    <http://example.org/amsterdam> <http://example.org/motto> "🚲 naar Amsterdam" .
    <http://example.org/Kraków> <http://example.org/population> "883193" .
    GRAPH <http://example.org/trips> {
        <http://example.org/mia> <http://example.org/visited> "Kraków" .
    }
}"#;

/// Finds every term of [`INSERT`] by its value.
fn assert_found(db: &GrafeoDB, how: &str) {
    let session = db.session();
    let rows = |query: &str| {
        session
            .execute_sparql(query)
            .unwrap_or_else(|error| panic!("{how}: {query}: {error}"))
            .rows()
            .to_vec()
    };
    for pattern in [
        r#"?s <http://example.org/name> "Kraków""#,
        r#"?s <http://example.org/name> "Krakau"@de"#,
        r#"?s <http://example.org/name> "Ámsterdam"@es"#,
        r#"?s <http://example.org/name> "阿姆斯特丹"@zh"#,
        r#"?s <http://example.org/motto> "🚲 naar Amsterdam""#,
        r#"?s <http://example.org/population> "883193""#,
        r#"GRAPH <http://example.org/trips> { ?s <http://example.org/visited> "Kraków" }"#,
    ] {
        let found = rows(&format!("SELECT ?s WHERE {{ {pattern} }}"));
        assert_eq!(found.len(), 1, "{how}: {pattern}");
    }
    let motto =
        rows("SELECT ?m WHERE { <http://example.org/amsterdam> <http://example.org/motto> ?m }");
    assert_eq!(motto, [vec![Value::from("🚲 naar Amsterdam")]], "{how}");
    let population =
        rows("SELECT ?p WHERE { <http://example.org/Kraków> <http://example.org/population> ?p }");
    assert_eq!(population, [vec![Value::from("883193")]], "{how}");
}

fn rdf_in_memory() -> GrafeoDB {
    GrafeoDB::with_config(Config::in_memory().with_graph_model(GraphModel::Rdf)).unwrap()
}

#[cfg(feature = "grafeo-file")]
fn rdf_file(path: &std::path::Path) -> GrafeoDB {
    GrafeoDB::with_config(rdf_file_config(path)).unwrap()
}

#[cfg(feature = "grafeo-file")]
fn rdf_file_config(path: &std::path::Path) -> Config {
    use grafeo_engine::config::StorageFormat;

    Config::persistent(path)
        .with_graph_model(GraphModel::Rdf)
        .with_storage_format(StorageFormat::Auto)
}

/// A close writes the RDF section in chunks (tiny caps make many) and the
/// reopen reads every term back.
#[cfg(feature = "grafeo-file")]
#[test]
fn non_ascii_terms_survive_a_close_and_reopen() {
    use grafeo_common::storage::{ChunkCaps, SectionType};
    use grafeo_common::testing::chunk_caps::with_chunk_caps;

    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("db.grafeo");
    {
        let db = rdf_file(&path);
        db.session().execute_sparql(INSERT).unwrap();
        let tiny = ChunkCaps {
            max_rows: 2,
            max_bytes: 256,
        };
        with_chunk_caps(tiny, || db.close()).unwrap();
    }
    let db = rdf_file(&path);
    let (version, chunks) = db
        .file_manager()
        .expect("a database file")
        .read_image(|image| {
            let source = image
                .section_source(SectionType::RdfStore)
                .expect("an RDF section");
            Ok((source.section_version(), source.chunks().len()))
        })
        .unwrap();
    assert_eq!(version, 3, "the RDF section in chunks");
    // In row groups of 2: the metadata chunk, then 3 chunks (subject,
    // predicate, object) for each of the 3 ranges of the 6 triples of the
    // default graph and the 1 range of the named graph.
    assert_eq!(chunks, 1 + 3 * (3 + 1), "the tiny caps cut the triples");
    assert_found(&db, "after a reopen");
}

/// The WAL replay after a crash reads every term back.
#[cfg(all(feature = "wal", feature = "grafeo-file"))]
#[test]
fn non_ascii_terms_survive_a_crash() {
    let (_dir, db) =
        common::replay::reopened_after_crash("non_ascii_terms_survive_a_crash", rdf_file, |db| {
            db.session().execute_sparql(INSERT).unwrap();
        });
    assert_found(&db, "after the WAL replay");
}

/// A WAL record whose term does not parse fails the open, naming the record,
/// where 0.5.x replayed the WAL without the triple. The record is damaged
/// after the crash: the IRI loses its closing '>' and keeps its length, and
/// its frame gets a matching CRC, so only the term is wrong.
#[cfg(all(feature = "wal", feature = "grafeo-file"))]
#[test]
fn a_wal_record_whose_term_does_not_parse_fails_the_open() {
    use std::cell::RefCell;

    let refused = RefCell::new(None);
    let open = |path: &std::path::Path| {
        // In the child the WAL does not exist yet; in the parent the crash
        // left it.
        if damage_wal_term(path) == 0 {
            return rdf_file(path);
        }
        match GrafeoDB::with_config(rdf_file_config(path)) {
            Ok(db) => db,
            Err(error) => {
                *refused.borrow_mut() = Some(error.to_string());
                rdf_in_memory()
            }
        }
    };
    let (_dir, _db) = common::replay::reopened_after_crash(
        "a_wal_record_whose_term_does_not_parse_fails_the_open",
        open,
        |db| {
            db.session()
                .execute_sparql(
                    r#"INSERT DATA { <http://example.org/alix> <http://example.org/name> "Alix" . }"#,
                )
                .unwrap();
        },
    );
    let error = refused
        .into_inner()
        .expect("the open with a damaged WAL record failed");
    assert!(
        error.contains("WAL record InsertRdfTriple in the default graph, subject")
            && error.contains("an IRI without its closing '>'"),
        "{error}"
    );
}

/// Damages the WAL next to the database at `path`: the IRI
/// `<http://example.org/alix>` loses its closing '>' in every frame that holds
/// it, and the frame's CRC is recomputed. Returns the frames changed.
#[cfg(all(feature = "wal", feature = "grafeo-file"))]
fn damage_wal_term(path: &std::path::Path) -> usize {
    let iri = b"<http://example.org/alix>";
    let Ok(logs) = std::fs::read_dir(format!("{}.wal", path.display())) else {
        return 0;
    };
    let mut changed = 0;
    for log in logs {
        let log = log.unwrap().path();
        let mut bytes = std::fs::read(&log).unwrap();
        // Frames: [length u32][data][crc32 of the data u32].
        let mut at = 0;
        while at + 4 <= bytes.len() {
            let length = u32::from_le_bytes(bytes[at..at + 4].try_into().unwrap());
            let data = at + 4..at + 4 + usize::try_from(length).unwrap();
            if data.end + 4 > bytes.len() {
                break;
            }
            if let Some(found) = bytes[data.clone()]
                .windows(iri.len())
                .position(|window| window == iri)
            {
                bytes[data.start + found + iri.len() - 1] = b' ';
                let crc = crc32fast::hash(&bytes[data.clone()]);
                bytes[data.end..data.end + 4].copy_from_slice(&crc.to_le_bytes());
                changed += 1;
            }
            at = data.end + 4;
        }
        std::fs::write(&log, bytes).unwrap();
    }
    changed
}

#[test]
fn non_ascii_terms_survive_a_snapshot_and_a_copy() {
    let db = rdf_in_memory();
    db.session().execute_sparql(INSERT).unwrap();
    let bytes = db.export_snapshot().unwrap();
    assert_found(
        &GrafeoDB::import_snapshot(&bytes).unwrap(),
        "after a snapshot import",
    );
    let restored = rdf_in_memory();
    restored.restore_snapshot(&bytes).unwrap();
    assert_found(&restored, "after a snapshot restore");
    assert_found(&db.to_memory().unwrap(), "in a copy in memory");
}

/// A snapshot whose term does not parse (here a damaged IRI) is refused,
/// naming the triple, where 0.5.x dropped the triple; a restore of it leaves
/// the database as it was.
#[test]
fn a_snapshot_with_a_term_that_does_not_parse_changes_nothing() {
    let db = rdf_in_memory();
    db.session()
        .execute_sparql(
            r#"INSERT DATA { <http://example.org/alix> <http://example.org/name> "Alix" . }"#,
        )
        .unwrap();
    let mut bytes = db.export_snapshot().unwrap();
    // The IRI loses its closing '>' and keeps its length, so the snapshot
    // still decodes.
    let iri = b"<http://example.org/alix>";
    let at = bytes
        .windows(iri.len())
        .position(|window| window == iri)
        .expect("the IRI in the snapshot");
    bytes[at + iri.len() - 1] = b' ';

    let error = match GrafeoDB::import_snapshot(&bytes) {
        Ok(_) => panic!("a damaged term was imported"),
        Err(error) => error.to_string(),
    };
    assert!(
        error.contains("snapshot RDF triple 0 of the default graph, subject")
            && error.contains("an IRI without its closing '>'"),
        "{error}"
    );

    let target = rdf_in_memory();
    target
        .session()
        .execute_sparql(
            r#"INSERT DATA { <http://example.org/gus> <http://example.org/name> "Gus" . }"#,
        )
        .unwrap();
    assert!(target.restore_snapshot(&bytes).is_err());
    let names = target
        .session()
        .execute_sparql("SELECT ?name WHERE { ?s <http://example.org/name> ?name }")
        .unwrap();
    assert_eq!(
        names.rows(),
        [vec![Value::from("Gus")]],
        "the refused restore changed nothing"
    );
}
