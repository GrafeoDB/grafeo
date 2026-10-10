//! A snapshot with RDF triples and vector and text indexes, imported or
//! restored by a build without the `triple-store`, `vector-index` or
//! `text-index` feature (the engine's default build, the WASM `edge`
//! profile, the `lpg` profile of the `grafeo` crate).
//!
//! A snapshot of `export_snapshot` holds the triples of every RDF graph and
//! the definition of each vector index (its label, property, dimensions,
//! metric and HNSW parameters) and text index of the default graph. A build
//! without `triple-store` cannot hold the triples, and one without
//! `vector-index` or `text-index` cannot build such an index:
//! `import_snapshot` and `restore_snapshot` used to leave them out without a
//! word, so the database lacked them, and a snapshot exported from it again
//! lost them for good. So such a build refuses the snapshot with an error
//! that names what it holds and the feature: `import_snapshot` creates no
//! database, and `restore_snapshot` leaves the database as it was.
//!
//! The fixture (`fixtures/snapshots/`, see its README) is exported by a
//! build with the three features; the test with them checks that it is the
//! snapshot the README describes, and writes it again on request.
//!
//! ```bash
//! cargo test -p grafeo-engine --no-default-features --features lpg,gql,grafeo-file \
//!     --test snapshot_without_its_features
//! cargo test -p grafeo-engine --all-features --test snapshot_without_its_features
//! ```

#![cfg(all(feature = "lpg", feature = "gql", not(miri)))]

use std::path::Path;

use grafeo_common::types::Value;
use grafeo_engine::GrafeoDB;

/// The fixture, exported by a build with the three features (see its
/// README).
const FIXTURE: &str = "tests/fixtures/snapshots/0.6.0-dev/documents.snapshot";

/// The bytes of the fixture.
fn fixture() -> Vec<u8> {
    std::fs::read(Path::new(env!("CARGO_MANIFEST_DIR")).join(FIXTURE)).unwrap()
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

/// Exports the fixture's snapshot, as its README describes.
#[cfg(all(
    feature = "triple-store",
    feature = "vector-index",
    feature = "text-index"
))]
fn export_documents() -> Vec<u8> {
    use grafeo_core::graph::rdf::{Term, Triple};

    let db = GrafeoDB::new_in_memory();
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
    let iri = |name: &str| Term::iri(format!("http://example.org/{name}"));
    db.batch_insert_rdf([Triple::new(iri("alix"), iri("knows"), iri("gus"))])
        .unwrap();
    db.rdf_store()
        .graph_or_create("http://example.org/trips")
        .insert(Triple::new(iri("gus"), iri("visited"), iri("prague")));
    db.export_snapshot().unwrap()
}

/// Writes the fixture again at the path `GRAFEO_WRITE_SNAPSHOT_FIXTURE`
/// names; a no-op without it.
#[cfg(all(
    feature = "triple-store",
    feature = "vector-index",
    feature = "text-index"
))]
#[test]
fn write_the_fixture() {
    if let Some(path) = std::env::var_os("GRAFEO_WRITE_SNAPSHOT_FIXTURE") {
        std::fs::write(path, export_documents()).unwrap();
    }
}

/// The triples of `store` (the default graph, or a named graph), as
/// N-Triples terms, sorted.
#[cfg(all(
    feature = "triple-store",
    feature = "vector-index",
    feature = "text-index"
))]
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

/// Checks that `db` holds the fixture's documents with their two indexes
/// and their configuration, and its triples: Alix knows Gus in the default
/// graph, Gus visited Prague in `trips`.
#[cfg(all(
    feature = "triple-store",
    feature = "vector-index",
    feature = "text-index"
))]
fn assert_documents(db: &GrafeoDB) {
    assert_eq!(
        titles(db),
        [Value::from("Bridges"), Value::from("Canals")],
        "the documents"
    );
    let title = |node: grafeo_common::types::NodeId| {
        db.execute(&format!(
            "MATCH (d:Document) WHERE id(d) = {} RETURN d.title",
            node.as_u64()
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
            .map(|(node, _)| title(*node))
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
        .text_search("Document", "content", "canals", 3, None)
        .unwrap_or_else(|error| panic!("the text index of the documents: {error}"));
    assert_eq!(
        matches
            .iter()
            .map(|(node, _)| title(*node))
            .collect::<Vec<_>>(),
        [Value::from("Canals")],
        "the documents about canals"
    );
    let triple =
        |s: &str, p: &str, o: &str| [s, p, o].map(|name| format!("<http://example.org/{name}>"));
    assert_eq!(
        stored(db.rdf_store()),
        [triple("alix", "knows", "gus")],
        "the triples of the default graph"
    );
    assert_eq!(
        db.rdf_store()
            .graph("http://example.org/trips")
            .map(|graph| stored(&graph)),
        Some(vec![triple("gus", "visited", "prague")]),
        "the triples of the named graph"
    );
}

/// The fixture is the snapshot its README describes: a build with the three
/// features imports and restores it with its triples and its two indexes,
/// with their configuration. A change of the snapshot format shows up here
/// first: write the fixture again.
#[cfg(all(
    feature = "triple-store",
    feature = "vector-index",
    feature = "text-index"
))]
#[test]
fn the_fixture_holds_triples_and_search_indexes() {
    let imported = GrafeoDB::import_snapshot(&fixture()).unwrap();
    assert_documents(&imported);

    let restored = GrafeoDB::new_in_memory();
    restored
        .execute("INSERT (:Person {name: 'Vincent'})")
        .unwrap();
    restored.restore_snapshot(&fixture()).unwrap();
    assert_documents(&restored);

    // The same steps export the same snapshot.
    assert_documents(&GrafeoDB::import_snapshot(&export_documents()).unwrap());
}

/// What the error of a refused snapshot says about the data this build
/// cannot read, and the features this build has, which the error does not
/// name.
#[cfg(not(all(
    feature = "triple-store",
    feature = "vector-index",
    feature = "text-index"
)))]
fn expected_parts() -> (Vec<&'static str>, Vec<&'static str>) {
    let mut parts = Vec::new();
    let mut in_build = Vec::new();
    if cfg!(feature = "triple-store") {
        in_build.push("`triple-store`");
    } else {
        parts.push(
            "RDF triples (1 in the default graph, 1 in graph http://example.org/trips), data that \
             only a build with the `triple-store` feature can read",
        );
    }
    if cfg!(feature = "vector-index") {
        in_build.push("`vector-index`");
    } else {
        parts.push(
            "vector indexes (on :Document(embedding)), data that only a build with the \
             `vector-index` feature can read",
        );
    }
    if cfg!(feature = "text-index") {
        in_build.push("`text-index`");
    } else {
        parts.push(
            "text indexes (on :Document(content)), data that only a build with the `text-index` \
             feature can read",
        );
    }
    (parts, in_build)
}

/// Checks that `error` refuses the fixture: it says what the snapshot holds
/// that this build cannot read, and the feature of each, names none of the
/// features this build has, and tells to `action` the snapshot with a
/// build that has them.
#[cfg(not(all(
    feature = "triple-store",
    feature = "vector-index",
    feature = "text-index"
)))]
fn assert_refusal(error: &str, action: &str) {
    let (parts, in_build) = expected_parts();
    assert!(
        error.contains("the snapshot holds ")
            && parts.iter().all(|part| error.contains(part))
            && error.contains(&format!(": {action} it with ")),
        "the error says what the snapshot holds, the features and to {action} it with a build \
         that has them, {parts:?}: {error}"
    );
    assert!(
        in_build.iter().all(|feature| !error.contains(feature)),
        "the error names none of the features {in_build:?} of this build: {error}"
    );
}

/// A build without one of the features refuses to import the fixture: it
/// creates no database, where it used to create one without the triples or
/// the indexes.
#[cfg(not(all(
    feature = "triple-store",
    feature = "vector-index",
    feature = "text-index"
)))]
#[test]
fn a_build_without_the_features_refuses_to_import_the_snapshot() {
    let error = match GrafeoDB::import_snapshot(&fixture()) {
        Ok(db) => panic!(
            "the import succeeded, and serves the documents {:?} without what this build cannot \
             read",
            titles(&db)
        ),
        Err(error) => error.to_string(),
    };
    assert_refusal(&error, "import");
}

/// A build without one of the features refuses to restore the fixture, and
/// the database stays as it was: its people, its edge and everything else a
/// snapshot of it holds.
#[cfg(not(all(
    feature = "triple-store",
    feature = "vector-index",
    feature = "text-index"
)))]
#[test]
fn a_build_without_the_features_refuses_to_restore_the_snapshot() {
    let db = GrafeoDB::new_in_memory();
    db.execute(
        "INSERT (:Person {name: 'Vincent', city: 'Paris'})-[:KNOWS]->(:Person {name: 'Mia'})",
    )
    .unwrap();
    let before = db.export_snapshot().unwrap();

    let error = match db.restore_snapshot(&fixture()) {
        Ok(()) => panic!(
            "the restore succeeded, and the database serves the documents {:?} without what this \
             build cannot read",
            titles(&db)
        ),
        Err(error) => error.to_string(),
    };
    assert_refusal(&error, "restore");

    let people: Vec<Vec<Value>> = db
        .execute(
            "MATCH (a:Person)-[:KNOWS]->(b:Person) RETURN a.name, a.city, b.name ORDER BY a.name",
        )
        .unwrap()
        .rows()
        .to_vec();
    assert_eq!(
        people,
        [vec![
            Value::from("Vincent"),
            Value::from("Paris"),
            Value::from("Mia")
        ]],
        "the people the refused restore leaves"
    );
    assert!(titles(&db).is_empty(), "no document is restored");
    assert!(
        db.export_snapshot().unwrap() == before,
        "the refused restore leaves the database as it was"
    );
}
