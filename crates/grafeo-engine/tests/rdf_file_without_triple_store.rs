//! A database with RDF triples in a build without the `triple-store` feature
//! (the engine's default build, the default and `lpg` profiles of the `grafeo`
//! crate, the CLI).
//!
//! A database file holds its triples in its `RdfStore` section and the Ring
//! index over them in its `RdfRing` section; its sidecar WAL holds the RDF
//! changes since the last checkpoint as records, a 0.5.x WAL directory holds
//! all of them as records, and a 0.5.x container v1 file in its snapshot.
//! A build without `triple-store` cannot read any of these. It used to open
//! the database without its triples, and its next checkpoint wrote the file
//! without them, losing them for good. So such a build refuses the database:
//! a read-write open, a read-only open and `open_in_memory` fail with an
//! error that names the file and the feature, and change nothing on disk.
//!
//! The fixture (`fixtures/rdf/`, see its README) is written by a build with
//! `triple-store`; the tests with the feature check that it is the database
//! the README describes, and write it again on request.
//!
//! ```bash
//! cargo test -p grafeo-engine --no-default-features --features lpg,gql,grafeo-file \
//!     --test rdf_file_without_triple_store
//! cargo test -p grafeo-engine --no-default-features --features lpg,gql,wal,grafeo-file \
//!     --test rdf_file_without_triple_store
//! cargo test -p grafeo-engine --all-features --test rdf_file_without_triple_store
//! ```

#![cfg(all(feature = "lpg", feature = "gql", feature = "grafeo-file", not(miri)))]

#[cfg(not(feature = "triple-store"))]
#[path = "common/legacy_file.rs"]
mod legacy_file;

use std::path::{Path, PathBuf};

use grafeo_common::storage::section::SectionType;
use grafeo_common::types::Value;
use grafeo_engine::{Config, GrafeoDB};

/// The fixture, written by a build with `triple-store` (see its README).
const FIXTURE: &str = "tests/fixtures/rdf/0.6.0-dev/triples.grafeo";

/// A copy of the fixture, and the path of the copy.
fn fixture() -> (tempfile::TempDir, PathBuf) {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("triples.grafeo");
    std::fs::copy(Path::new(env!("CARGO_MANIFEST_DIR")).join(FIXTURE), &path).unwrap();
    (dir, path)
}

/// The names of the people, sorted.
fn people(db: &GrafeoDB) -> Vec<Value> {
    db.execute("MATCH (p:Person) RETURN p.name AS name ORDER BY name")
        .unwrap()
        .rows()
        .iter()
        .map(|row| row[0].clone())
        .collect()
}

/// The RDF sections of the file at `path`.
fn rdf_sections(path: &Path) -> Vec<SectionType> {
    let manager = grafeo_storage::file::GrafeoFileManager::open_read_only(path, None).unwrap();
    let found = manager
        .read_image(|image| {
            Ok([SectionType::RdfStore, SectionType::RdfRing]
                .into_iter()
                .filter(|section_type| image.section_source(*section_type).is_some())
                .collect::<Vec<_>>())
        })
        .unwrap();
    manager.close().unwrap();
    found
}

/// The triples of `store` (the default graph, or a named graph), as
/// N-Triples terms, sorted.
#[cfg(feature = "triple-store")]
fn stored(store: &grafeo_core::graph::rdf::RdfStore) -> Vec<[String; 3]> {
    let mut triples: Vec<[String; 3]> = store
        .triples()
        .iter()
        .map(|triple| {
            [
                triple.subject().to_string(),
                triple.predicate().to_string(),
                triple.object().to_string(),
            ]
        })
        .collect();
    triples.sort();
    triples
}

/// `<http://example.org/s> <http://example.org/p> <http://example.org/o>`.
#[cfg(feature = "triple-store")]
fn triple(s: &str, p: &str, o: &str) -> [String; 3] {
    [s, p, o].map(|name| format!("<http://example.org/{name}>"))
}

/// The triples of the default graph and of the named graph `trips` of `db`.
#[cfg(feature = "triple-store")]
fn triples(db: &GrafeoDB) -> (Vec<[String; 3]>, Vec<[String; 3]>) {
    let trips = db
        .rdf_store()
        .graph("http://example.org/trips")
        .map(|graph| stored(&graph))
        .unwrap_or_default();
    (stored(db.rdf_store()), trips)
}

/// What the fixture's RDF graphs hold: Alix knows Gus in the default graph,
/// Gus visited Prague in `trips`.
#[cfg(feature = "triple-store")]
fn every_triple() -> (Vec<[String; 3]>, Vec<[String; 3]>) {
    (
        vec![triple("alix", "knows", "gus")],
        vec![triple("gus", "visited", "prague")],
    )
}

/// Writes the fixture's database at `path`, as its README describes.
#[cfg(all(feature = "triple-store", feature = "sparql", feature = "ring-index"))]
fn write_triples(path: &Path) {
    let db = GrafeoDB::with_config(Config::persistent(path)).unwrap();
    db.execute("INSERT (:Person {name: 'Alix'})-[:KNOWS]->(:Person {name: 'Gus'})")
        .unwrap();
    db.execute_sparql(
        "INSERT DATA { <http://example.org/alix> <http://example.org/knows> \
         <http://example.org/gus> . GRAPH <http://example.org/trips> { \
         <http://example.org/gus> <http://example.org/visited> <http://example.org/prague> . } }",
    )
    .unwrap();
    db.rdf_store().rebuild_ring();
    db.close().unwrap();
}

/// Writes the fixture again at the path `GRAFEO_WRITE_RDF_FIXTURE` names; a
/// no-op without it.
#[cfg(all(feature = "triple-store", feature = "sparql", feature = "ring-index"))]
#[test]
fn write_the_fixture() {
    if let Some(path) = std::env::var_os("GRAFEO_WRITE_RDF_FIXTURE") {
        write_triples(Path::new(&path));
    }
}

/// The fixture is the database its README describes: a build with
/// `triple-store` reads its triples, in the default graph and in a named
/// graph, with the Ring index over them, and its people. A change of the
/// file format shows up here first: write the fixture again.
#[cfg(feature = "triple-store")]
#[test]
fn the_fixture_holds_rdf_triples() {
    let (_dir, path) = fixture();
    assert_eq!(
        rdf_sections(&path),
        [SectionType::RdfStore, SectionType::RdfRing],
        "the file holds the triples and the Ring index"
    );
    let db = GrafeoDB::with_config(Config::persistent(&path)).unwrap();
    assert_eq!(triples(&db), every_triple());
    #[cfg(feature = "ring-index")]
    assert!(db.rdf_store().ring().is_some(), "the Ring index is loaded");
    assert_eq!(people(&db), [Value::from("Alix"), Value::from("Gus")]);
    db.close().unwrap();

    // The same steps write the same database.
    #[cfg(all(feature = "sparql", feature = "ring-index"))]
    {
        let dir = tempfile::tempdir().unwrap();
        let written = dir.path().join("triples.grafeo");
        write_triples(&written);
        assert_eq!(
            rdf_sections(&written),
            [SectionType::RdfStore, SectionType::RdfRing]
        );
        let db = GrafeoDB::with_config(Config::persistent(&written)).unwrap();
        assert_eq!(triples(&db), every_triple());
        assert_eq!(people(&db), [Value::from("Alix"), Value::from("Gus")]);
        db.close().unwrap();
    }
}

/// A 0.5.x container v1 file holds its database as one snapshot blob. This
/// one holds a single RDF triple and nothing else, encoded by hand as 0.5.x
/// encoded its snapshot (bincode of the snapshot of `export_snapshot`,
/// version 4): every list empty but the default graph's triples.
fn snapshot_with_a_triple() -> Vec<u8> {
    let mut blob = vec![4, 0, 0, 0, 1];
    for term in [
        "<http://example.org/alix>",
        "<http://example.org/knows>",
        "<http://example.org/gus>",
    ] {
        blob.push(u8::try_from(term.len()).unwrap());
        blob.extend_from_slice(term.as_bytes());
    }
    // The RDF named graphs, the six lists of the schema, the three lists of
    // the indexes, and the epoch.
    blob.extend_from_slice(&[0; 11]);
    blob
}

/// Writes at `path` a 0.5.x container v1 file whose snapshot is `snapshot`.
fn write_0_5_snapshot_file(path: &Path, snapshot: &[u8]) {
    use std::io::{Seek, SeekFrom, Write};

    use grafeo_storage::file::format::DATA_OFFSET;
    use grafeo_storage::file::header::{write_db_header, write_file_header};
    use grafeo_storage::file::{DbHeader, FileHeader};

    let mut file = std::fs::File::create(path).unwrap();
    write_file_header(&mut file, &FileHeader::new()).unwrap();
    write_db_header(&mut file, 0, &DbHeader::EMPTY).unwrap();
    write_db_header(
        &mut file,
        1,
        &DbHeader {
            iteration: 1,
            checksum: crc32fast::hash(snapshot),
            snapshot_length: u64::try_from(snapshot.len()).unwrap(),
            epoch: 1,
            ..DbHeader::EMPTY
        },
    )
    .unwrap();
    file.seek(SeekFrom::Start(DATA_OFFSET)).unwrap();
    file.write_all(snapshot).unwrap();
}

/// The snapshot [`snapshot_with_a_triple`] encodes by hand is one 0.5.x
/// wrote: `import_snapshot` reads it, and a read-only open of a container v1
/// file around it reads the triple.
#[cfg(feature = "triple-store")]
#[test]
fn the_hand_encoded_snapshot_holds_a_triple() {
    let expected = vec![triple("alix", "knows", "gus")];
    let imported = GrafeoDB::import_snapshot(&snapshot_with_a_triple()).unwrap();
    assert_eq!(stored(imported.rdf_store()), expected, "imported");

    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("triples.grafeo");
    write_0_5_snapshot_file(&path, &snapshot_with_a_triple());
    let db = GrafeoDB::open_read_only(&path).unwrap();
    assert_eq!(stored(db.rdf_store()), expected, "read from the 0.5.x file");
}

/// An open of a database file, as a build without `triple-store` has it.
#[cfg(not(feature = "triple-store"))]
type Open = fn(&Path) -> grafeo_common::utils::error::Result<GrafeoDB>;

/// Every open of an existing database file: read-write, read-only and,
/// with the `wal` feature, `open_in_memory` (which reads the file as a
/// read-only open does).
#[cfg(not(feature = "triple-store"))]
fn opens() -> Vec<(&'static str, Open)> {
    #[cfg_attr(
        not(feature = "wal"),
        expect(unused_mut, reason = "only the `wal` feature adds `open_in_memory`")
    )]
    let mut opens: Vec<(&'static str, Open)> = vec![
        ("read-write", |path| {
            GrafeoDB::with_config(Config::persistent(path))
        }),
        ("read-only", |path| GrafeoDB::open_read_only(path)),
    ];
    #[cfg(feature = "wal")]
    opens.push(("in-memory", |path| GrafeoDB::open_in_memory(path)));
    opens
}

/// Every file and directory under `root`, with the bytes of each file.
#[cfg(not(feature = "triple-store"))]
fn files(root: &Path) -> std::collections::BTreeMap<PathBuf, Option<Vec<u8>>> {
    let mut found = std::collections::BTreeMap::new();
    let mut pending = vec![root.to_path_buf()];
    while let Some(dir) = pending.pop() {
        for entry in std::fs::read_dir(&dir).unwrap() {
            let path = entry.unwrap().path();
            let relative = path.strip_prefix(root).unwrap().to_path_buf();
            if path.is_dir() {
                found.insert(relative, None);
                pending.push(path);
            } else {
                found.insert(relative, Some(std::fs::read(&path).unwrap()));
            }
        }
    }
    found
}

/// Checks that every open of the database at `path` fails with an error
/// that names `named` (the file, or its WAL), says where it holds the
/// triples (`held`) and names the feature, and that every file in `dir` is
/// still what it was before.
#[cfg(not(feature = "triple-store"))]
fn assert_refused(dir: &Path, path: &Path, named: &Path, held: &str) {
    let before = files(dir);
    for (kind, open) in opens() {
        let error = match open(path) {
            Ok(db) => panic!(
                "the {kind} open of {} succeeded without `triple-store`, and serves the \
                 people {:?} without their triples",
                path.display(),
                people(&db)
            ),
            Err(error) => error.to_string(),
        };
        assert!(
            error.contains(&format!("{} holds", named.display()))
                && error.contains(&format!("RDF triples ({held}"))
                && error.contains("only a build with the `triple-store` feature can read"),
            "the {kind} error names {}, where the triples are ({held}) and the feature: \
             {error}",
            named.display()
        );
        assert!(
            files(dir) == before,
            "the refused {kind} open of {} changes nothing on disk",
            path.display()
        );
    }
}

/// A build without `triple-store` refuses the fixture, whose triples and
/// Ring index are in their sections, with an error that names the file,
/// the sections and the feature, on a read-write open, a read-only open and
/// `open_in_memory`, and every file stays as it was, byte for byte.
#[cfg(not(feature = "triple-store"))]
#[test]
fn a_build_without_triple_store_refuses_a_file_with_triples() {
    let (dir, path) = fixture();
    assert_eq!(
        rdf_sections(&path),
        [SectionType::RdfStore, SectionType::RdfRing],
        "the fixture holds the triples and the Ring index"
    );
    assert_refused(dir.path(), &path, &path, "sections RdfStore, RdfRing)");
}

/// The released 0.5.x files closed cleanly hold their triples in their
/// `RdfStore` section: a build without `triple-store` neither reads them
/// (read-only, `open_in_memory`) nor migrates them (read-write). A file
/// with only the Ring index (`RdfRing`, which mirrors the triples) is
/// refused as well.
#[cfg(not(feature = "triple-store"))]
#[test]
fn a_build_without_triple_store_refuses_a_0_5_file_with_triples() {
    for version in ["0.5.43", "0.5.44"] {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("closed.grafeo");
        std::fs::copy(legacy_file::released(version, "closed.grafeo"), &path).unwrap();
        assert_refused(dir.path(), &path, &path, "sections RdfStore");
    }

    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("ring.grafeo");
    legacy_file::write_0_5_file(
        &path,
        "0.5.43",
        |section_type| section_type != SectionType::RdfStore,
        &[(SectionType::RdfRing, b"RdfRing of Vincent".to_vec())],
    );
    assert_refused(dir.path(), &path, &path, "sections RdfRing)");
}

/// A 0.5.x container v1 file holds its triples in its snapshot: a build
/// without `triple-store` neither reads nor migrates it.
#[cfg(not(feature = "triple-store"))]
#[test]
fn a_build_without_triple_store_refuses_a_0_5_snapshot_with_triples() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("snapshot.grafeo");
    write_0_5_snapshot_file(&path, &snapshot_with_a_triple());
    assert_refused(dir.path(), &path, &path, "the snapshot of a 0.5.x file)");
}

/// The released 0.5.x WAL directories hold their triples only as records of
/// their WAL: a build without `triple-store` neither reads nor migrates
/// them, and names the WAL.
#[cfg(all(feature = "wal", not(feature = "triple-store")))]
#[test]
fn a_build_without_triple_store_refuses_a_0_5_wal_with_triples() {
    for version in ["0.5.43", "0.5.44"] {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("directory");
        copy_dir(&legacy_file::released(version, "directory"), &path);
        assert_refused(dir.path(), &path, &path.join("wal"), "WAL records)");
    }
}

/// A 0.6 file whose sidecar WAL holds RDF records (a writer with
/// `triple-store` exited without `close()`): a build without the feature
/// refuses it, naming the WAL, and the WAL stays, so a build with the
/// feature still replays the triples.
#[cfg(all(feature = "wal", not(feature = "triple-store")))]
#[test]
fn a_build_without_triple_store_refuses_a_sidecar_wal_with_triples() {
    use grafeo_common::types::TransactionId;
    use grafeo_storage::wal::{WalManager, WalRecord};

    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("people.grafeo");
    let db = GrafeoDB::with_config(Config::persistent(&path)).unwrap();
    db.execute("INSERT (:Person {name: 'Alix'})").unwrap();
    db.close().unwrap();
    drop(db);
    let mut wal = path.as_os_str().to_owned();
    wal.push(".wal");
    let wal = PathBuf::from(wal);
    {
        let log = WalManager::open(&wal).unwrap();
        log.log(&WalRecord::InsertRdfTriple {
            subject: "<http://example.org/alix>".to_string(),
            predicate: "<http://example.org/knows>".to_string(),
            object: "<http://example.org/gus>".to_string(),
            graph: None,
        })
        .unwrap();
        log.log(&WalRecord::TransactionCommit {
            transaction_id: TransactionId::new(19),
        })
        .unwrap();
        log.sync().unwrap();
    }
    assert_refused(dir.path(), &path, &wal, "WAL records)");
}

#[cfg(all(feature = "wal", not(feature = "triple-store")))]
fn copy_dir(from: &Path, to: &Path) {
    std::fs::create_dir_all(to).unwrap();
    for entry in std::fs::read_dir(from).unwrap() {
        let path = entry.unwrap().path();
        let target = to.join(path.file_name().unwrap());
        if path.is_dir() {
            copy_dir(&path, &target);
        } else {
            std::fs::copy(&path, &target).unwrap();
        }
    }
}
