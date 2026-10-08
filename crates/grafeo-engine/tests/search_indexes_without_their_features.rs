//! A database with vector and text indexes in a build without the
//! `vector-index` or `text-index` feature (the engine's default build, the
//! `lpg` and `rdf` profiles of the `grafeo` crate, the CLI).
//!
//! The catalog of a database file holds the definition of each vector index
//! (its label, property, dimensions, metric and HNSW parameters) and of each
//! text index, and the default graph's indexes are also in their own
//! sections (`VectorStore`, `TextIndex`), which only mirror the data. A build
//! without the feature cannot build such an index, and used to drop its
//! definition: its next checkpoint wrote the catalog without it, so a build
//! with the feature found no index after that (searches failed with "No
//! vector index found"). So such a build refuses the database: a read-write
//! open, a read-only open and `open_in_memory` fail with an error that names
//! the file, the indexes and the feature, and change nothing on disk.
//!
//! The fixture (`fixtures/search-indexes/`, see its README) is written by a
//! build with both features; the tests with them check that it is the
//! database the README describes, and write it again on request.
//!
//! ```bash
//! cargo test -p grafeo-engine --no-default-features --features lpg,gql,grafeo-file \
//!     --test search_indexes_without_their_features
//! cargo test -p grafeo-engine --all-features --test search_indexes_without_their_features
//! ```

#![cfg(all(feature = "lpg", feature = "gql", feature = "grafeo-file", not(miri)))]

#[cfg(not(all(feature = "vector-index", feature = "text-index")))]
#[path = "common/legacy_file.rs"]
mod legacy_file;

use std::path::{Path, PathBuf};

use grafeo_common::storage::section::SectionType;
use grafeo_common::types::Value;
use grafeo_engine::{Config, GrafeoDB};

/// The fixture, written by a build with both features (see its README).
const FIXTURE: &str = "tests/fixtures/search-indexes/0.6.0-dev/documents.grafeo";

/// A copy of the fixture, and the path of the copy.
fn fixture() -> (tempfile::TempDir, PathBuf) {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("documents.grafeo");
    std::fs::copy(Path::new(env!("CARGO_MANIFEST_DIR")).join(FIXTURE), &path).unwrap();
    (dir, path)
}

/// The titles of the documents, sorted.
fn titles(db: &GrafeoDB) -> Vec<Value> {
    db.execute("MATCH (d:Document) RETURN d.title AS title ORDER BY title")
        .unwrap()
        .rows()
        .iter()
        .map(|row| row[0].clone())
        .collect()
}

/// The index sections of the file at `path`.
fn index_sections(path: &Path) -> Vec<SectionType> {
    let manager = grafeo_storage::file::GrafeoFileManager::open_read_only(path, None).unwrap();
    let found = manager
        .read_image(|image| {
            Ok([SectionType::VectorStore, SectionType::TextIndex]
                .into_iter()
                .filter(|section_type| image.section_source(*section_type).is_some())
                .collect::<Vec<_>>())
        })
        .unwrap();
    manager.close().unwrap();
    found
}

/// Writes the fixture's database at `path`, as its README describes.
#[cfg(all(feature = "vector-index", feature = "text-index"))]
fn write_documents(path: &Path) {
    let db = GrafeoDB::with_config(Config::persistent(path)).unwrap();
    db.execute(
        "INSERT (:Document {title: 'Canals', content: 'boats on the canals of Amsterdam', \
         embedding: vector([3.0, 19.0, 88.0])}), \
         (:Document {title: 'Bridges', content: 'bridges over the river in Prague', \
         embedding: vector([88.0, 19.0, 3.0])})",
    )
    .unwrap();
    db.create_vector_index(
        "Document",
        "embedding",
        Some(3),
        Some("euclidean"),
        Some(19),
        Some(88),
        None,
    )
    .unwrap();
    db.create_text_index("Document", "content").unwrap();
    // A vector index of a named graph: only the catalog holds it.
    db.execute("CREATE GRAPH trips").unwrap();
    let trips = db.graph("trips").unwrap();
    trips
        .execute("INSERT (:Stop {name: 'Berlin', position: vector([3.0, 19.0])})")
        .unwrap();
    trips
        .execute("CREATE VECTOR INDEX stop_position ON :Stop(position)")
        .unwrap();
    db.close().unwrap();
}

/// Writes the fixture again at the path `GRAFEO_WRITE_SEARCH_INDEXES_FIXTURE`
/// names; a no-op without it.
#[cfg(all(feature = "vector-index", feature = "text-index"))]
#[test]
fn write_the_fixture() {
    if let Some(path) = std::env::var_os("GRAFEO_WRITE_SEARCH_INDEXES_FIXTURE") {
        write_documents(Path::new(&path));
    }
}

/// Checks that `db` holds the fixture's documents and its three indexes,
/// with their configuration.
#[cfg(all(feature = "vector-index", feature = "text-index"))]
fn assert_indexed(db: &GrafeoDB) {
    assert_eq!(
        titles(db),
        [Value::from("Bridges"), Value::from("Canals")],
        "the documents"
    );
    let title = |node| {
        db.execute(&format!(
            "MATCH (d:Document) WHERE id(d) = {} RETURN d.title",
            node
        ))
        .unwrap()
        .rows()[0][0]
            .clone()
    };
    let nearest = db
        .vector_search("Document", "embedding", &[3.0, 19.0, 80.0], 1, None, None)
        .unwrap_or_else(|error| panic!("the vector index of the documents: {error}"));
    assert_eq!(
        nearest
            .iter()
            .map(|(node, _)| title(node.as_u64()))
            .collect::<Vec<_>>(),
        [Value::from("Canals")],
        "the nearest document"
    );
    let config = db
        .graph_store()
        .vector_index_config("Document", "embedding")
        .expect("the vector index of the documents");
    assert_eq!(
        (
            config.dimensions,
            config.metric.name(),
            config.m,
            config.ef_construction
        ),
        (3, "euclidean", 19, 88),
        "the vector index keeps its configuration"
    );
    let matches = db
        .text_search("Document", "content", "canals", 3)
        .unwrap_or_else(|error| panic!("the text index of the documents: {error}"));
    assert_eq!(
        matches
            .iter()
            .map(|(node, _)| title(node.as_u64()))
            .collect::<Vec<_>>(),
        [Value::from("Canals")],
        "the documents about canals"
    );
    assert!(
        db.graph("trips")
            .unwrap()
            .graph_store()
            .unwrap()
            .has_vector_index("Stop", "position"),
        "the vector index of the named graph"
    );
}

/// The fixture is the database its README describes: a build with both
/// features reads its documents and finds its three indexes with their
/// configuration. A change of the file format shows up here first: write
/// the fixture again.
#[cfg(all(feature = "vector-index", feature = "text-index"))]
#[test]
fn the_fixture_holds_vector_and_text_indexes() {
    let (_dir, path) = fixture();
    assert_eq!(
        index_sections(&path),
        [SectionType::VectorStore, SectionType::TextIndex],
        "the file holds the default graph's index sections"
    );
    let db = GrafeoDB::with_config(Config::persistent(&path)).unwrap();
    assert_indexed(&db);
    db.close().unwrap();

    // The same steps write the same database.
    let dir = tempfile::tempdir().unwrap();
    let written = dir.path().join("documents.grafeo");
    write_documents(&written);
    let db = GrafeoDB::with_config(Config::persistent(&written)).unwrap();
    assert_indexed(&db);
    db.close().unwrap();
}

/// An open of a database file, as a build without one of the features has
/// it.
#[cfg(not(all(feature = "vector-index", feature = "text-index")))]
type Open = fn(&Path) -> grafeo_common::utils::error::Result<GrafeoDB>;

/// Every open of an existing database file: read-write, read-only and,
/// with the `wal` feature, `open_in_memory` (which reads the file as a
/// read-only open does).
#[cfg(not(all(feature = "vector-index", feature = "text-index")))]
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
#[cfg(not(all(feature = "vector-index", feature = "text-index")))]
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

/// What the error of a refused open says about the indexes this build
/// cannot build: `vector` and `text` list them (`on :Label(property)`, ...).
#[cfg(not(all(feature = "vector-index", feature = "text-index")))]
fn expected_parts(vector: &str, text: &str) -> Vec<String> {
    let mut parts = Vec::new();
    if !cfg!(feature = "vector-index") {
        parts.push(format!(
            "vector indexes ({vector}), data that only a build with the `vector-index` feature \
             can read"
        ));
    }
    if !cfg!(feature = "text-index") {
        parts.push(format!(
            "text indexes ({text}), data that only a build with the `text-index` feature can read"
        ));
    }
    parts
}

/// Checks that every open of the database at `path` fails with an error
/// that names the file and holds each of `parts`, and that every file in
/// `dir` is still what it was before.
#[cfg(not(all(feature = "vector-index", feature = "text-index")))]
fn assert_refused(dir: &Path, path: &Path, parts: &[String]) {
    let before = files(dir);
    for (kind, open) in opens() {
        let error = match open(path) {
            Ok(db) => panic!(
                "the {kind} open of {} succeeded, and serves the documents {:?} without the \
                 indexes it cannot build",
                path.display(),
                titles(&db)
            ),
            Err(error) => error.to_string(),
        };
        assert!(
            error.contains(&format!("{} holds", path.display()))
                && parts.iter().all(|part| error.contains(part.as_str())),
            "the {kind} error names the file and {parts:?}: {error}"
        );
        assert!(
            files(dir) == before,
            "the refused {kind} open of {} changes nothing on disk",
            path.display()
        );
    }
}

/// A build without `vector-index` or `text-index` refuses the fixture, with
/// an error that names the file, the indexes it cannot build (the named
/// graph's too, whose definition only the catalog holds) and the feature,
/// on a read-write open, a read-only open and `open_in_memory`, and every
/// file stays as it was, byte for byte.
#[cfg(not(all(feature = "vector-index", feature = "text-index")))]
#[test]
fn a_build_without_the_features_refuses_a_file_with_search_indexes() {
    let (dir, path) = fixture();
    assert_eq!(
        index_sections(&path),
        [SectionType::VectorStore, SectionType::TextIndex],
        "the fixture holds the default graph's index sections"
    );
    assert_refused(
        dir.path(),
        &path,
        &expected_parts(
            "on :Document(embedding), :Stop(position) in graph trips",
            "on :Document(content)",
        ),
    );
}

/// The released 0.5.44 file closed cleanly defines a vector and a text
/// index in its catalog, and holds them in their sections: a build without
/// the features neither reads nor migrates it, and neither a file whose
/// catalog alone defines them. The triples of the released file are left
/// out, so the refusal is about the indexes.
#[cfg(not(all(feature = "vector-index", feature = "text-index")))]
#[test]
fn a_build_without_the_features_refuses_a_0_5_file_with_search_indexes() {
    let parts = expected_parts("on :Document(embedding)", "on :Document(content)");
    for left_out in [
        &[SectionType::RdfStore][..],
        &[
            SectionType::RdfStore,
            SectionType::VectorStore,
            SectionType::TextIndex,
        ],
    ] {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("closed.grafeo");
        legacy_file::write_0_5_file(
            &path,
            "0.5.44",
            |section_type| !left_out.contains(&section_type),
            &[],
        );
        assert_refused(dir.path(), &path, &parts);
    }
}
