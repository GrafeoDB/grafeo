//! `compact()` keeps the indexes made before it.
//!
//! The property, text and vector indexes defined before it (through the API
//! or through DDL, a vector index with its whole configuration) stay: they
//! find the nodes written before and after `compact()`, across
//! `recompact()`, another `compact()`, a checkpoint and a reopen. The indexes
//! of named graphs stay too, and so do the constraints that read an index (a
//! vector index fixes the size of its vectors, UNIQUE looks a value up in a
//! property index).
//!
//! ```bash
//! cargo test -p grafeo-engine --all-features --test compact_keeps_indexes
//! ```

#![cfg(all(
    feature = "lpg",
    feature = "gql",
    feature = "text-index",
    feature = "vector-index"
))]

use grafeo_common::types::{NodeId, Value};
use grafeo_core::index::vector::{DistanceMetric, QuantizationType};
use grafeo_engine::GrafeoDB;

/// The people compacted: Alix, Gus and Mia, each with a city in their bio
/// and a two-dimensional embedding; two notes for the quantized index.
/// Gus's embedding lies far from the others, so no one is reached only
/// through him in the HNSW graph, which a delete does not repair (#600).
const PEOPLE: &str = "INSERT (:Person {name: 'Alix', bio: 'canals of Amsterdam', \
                      embedding: vector([3.0, 3.0])}), \
                      (:Person {name: 'Gus', bio: 'bridges of Prague', \
                      embedding: vector([88.0, 88.0])}), \
                      (:Person {name: 'Mia', bio: 'museums of Paris', \
                      embedding: vector([19.0, 19.0])}), \
                      (:Note {embedding: vector([3.0, 19.0])}), \
                      (:Note {embedding: vector([19.0, 3.0])})";

/// Vincent, written after `compact()`: his bio has canals too, and his
/// embedding is the second nearest to the origin.
const VINCENT: &str = "INSERT (:Person {name: 'Vincent', bio: 'canals of Berlin', \
                       embedding: vector([3.0, 19.0])})";

/// The HNSW links per node the API index is given.
const LINKS: usize = 19;

/// The construction beam width the API index is given.
const BEAM: usize = 88;

/// An in-memory database holding [`PEOPLE`].
fn people() -> GrafeoDB {
    let db = GrafeoDB::new_in_memory();
    db.execute(PEOPLE).unwrap();
    db
}

/// A property index on `name`, a text index on `:Person(bio)`, a Euclidean
/// vector index on `:Person(embedding)` with its own HNSW parameters, and a
/// scalar-quantized one on `:Note(embedding)`, all through the API.
fn index_through_the_api(db: &GrafeoDB) {
    db.create_property_index("name").unwrap();
    db.create_text_index("Person", "bio").unwrap();
    db.create_vector_index(
        "Person",
        "embedding",
        Some(2),
        Some("euclidean"),
        Some(LINKS),
        Some(BEAM),
        None,
    )
    .unwrap();
    db.create_vector_index("Note", "embedding", None, None, None, None, Some("scalar"))
        .unwrap();
}

/// The indexes of [`index_through_the_api`] on `:Person`, through DDL.
const PERSON_INDEXES: [&str; 3] = [
    "CREATE INDEX person_name FOR (p:Person) ON (p.name)",
    "CREATE INDEX person_bio FOR (p:Person) ON (p.bio) USING TEXT",
    "CREATE VECTOR INDEX person_embedding ON :Person(embedding) DIMENSION 2 \
     METRIC 'euclidean'",
];

/// Runs [`PERSON_INDEXES`] through `execute`, a database's or a graph's.
fn index_through_ddl<T, E: std::fmt::Display>(execute: impl Fn(&str) -> Result<T, E>) {
    for statement in PERSON_INDEXES {
        if let Err(error) = execute(statement) {
            panic!("{statement}: {error}");
        }
    }
}

/// The names of the nodes `hits` found, sorted.
fn names_of(db: &GrafeoDB, hits: impl IntoIterator<Item = NodeId>) -> Vec<String> {
    let mut names: Vec<String> = hits
        .into_iter()
        .map(|id| {
            match db
                .get_node(id)
                .and_then(|node| node.get_property("name").cloned())
            {
                Some(Value::String(name)) => name.to_string(),
                other => panic!("node {id:?} has no name: {other:?}"),
            }
        })
        .collect();
    names.sort();
    names
}

/// The people the text index finds for `word` in their bio.
fn bios_with(db: &GrafeoDB, word: &str, stage: &str) -> Vec<String> {
    let hits = db
        .text_search("Person", "bio", word, 19, None)
        .unwrap_or_else(|error| panic!("{stage}: {error}"));
    names_of(db, hits.into_iter().map(|(id, _)| id))
}

/// The people the vector index finds nearest to the origin, nearest first.
fn nearest_to_the_origin(db: &GrafeoDB, stage: &str) -> Vec<String> {
    db.vector_search("Person", "embedding", &[0.0, 0.0], 19, None, None)
        .unwrap_or_else(|error| panic!("{stage}: {error}"))
        .into_iter()
        .map(|(id, _)| names_of(db, [id]).remove(0))
        .collect()
}

/// The people the property index finds by `name`.
fn named(db: &GrafeoDB, name: &str) -> Vec<String> {
    names_of(db, db.find_nodes_by_property("name", &Value::from(name)))
}

/// The indexes on `:Person` work in `db` and find `everyone`, whose bios
/// with canals are `canals`, nearest to the origin first: the property index
/// on `name`, the text index on `bio` and the Euclidean vector index on
/// `embedding`.
fn assert_person_indexes(db: &GrafeoDB, everyone: &[&str], canals: &[&str], stage: &str) {
    assert!(db.has_property_index("name"), "{stage}: the property index");
    for name in everyone {
        assert_eq!(named(db, name), [*name], "{stage}: the property index");
    }
    let lookup = format!("MATCH (p:Person {{name: '{}'}}) RETURN p.name", everyone[0]);
    let plan = db.execute(&format!("EXPLAIN {lookup}")).unwrap();
    assert!(
        matches!(&plan.rows()[0][0], Value::String(plan) if plan.contains("[index: name]")),
        "{stage}: the lookup plans the property index: {:?}",
        plan.rows()
    );
    assert_eq!(
        db.execute(&lookup).unwrap().rows(),
        [[Value::from(everyone[0])]],
        "{stage}: the lookup"
    );
    assert_eq!(
        bios_with(db, "canals", stage),
        canals,
        "{stage}: the text index"
    );
    assert_eq!(
        nearest_to_the_origin(db, stage),
        everyone,
        "{stage}: the vector index"
    );
    let config = db
        .graph_store()
        .vector_index_config("Person", "embedding")
        .unwrap_or_else(|| panic!("{stage}: no vector index config"));
    assert_eq!(
        (config.dimensions, config.metric),
        (2, DistanceMetric::Euclidean),
        "{stage}: the vector index's dimensions and metric"
    );
}

/// The API indexes keep the configuration they were given: the HNSW
/// parameters of `:Person(embedding)` and the quantization of
/// `:Note(embedding)`, whose search finds both notes.
fn assert_api_configuration(db: &GrafeoDB, stage: &str) {
    let config = db
        .graph_store()
        .vector_index_config("Person", "embedding")
        .unwrap_or_else(|| panic!("{stage}: no vector index config"));
    assert_eq!(
        (config.m, config.ef_construction),
        (LINKS, BEAM),
        "{stage}: the HNSW parameters"
    );
    let notes = db
        .store()
        .get_vector_index("Note", "embedding")
        .unwrap_or_else(|| panic!("{stage}: no quantized index"));
    assert_eq!(
        notes.quantization_type(),
        Some(QuantizationType::Scalar),
        "{stage}: the quantization"
    );
    assert_eq!(
        db.vector_search("Note", "embedding", &[3.0, 19.0], 3, None, None)
            .unwrap_or_else(|error| panic!("{stage}: {error}"))
            .len(),
        2,
        "{stage}: the quantized index finds both notes"
    );
}

/// The indexes made through the API before `compact()` find the compacted
/// people, and Vincent, written after it.
#[test]
fn api_indexes_made_before_compact_stay_after_it() {
    let mut db = people();
    index_through_the_api(&db);
    db.compact().unwrap();
    assert_person_indexes(&db, &["Alix", "Mia", "Gus"], &["Alix"], "after compact");
    assert_api_configuration(&db, "after compact");

    db.execute(VINCENT).unwrap();
    assert_person_indexes(
        &db,
        &["Alix", "Vincent", "Mia", "Gus"],
        &["Alix", "Vincent"],
        "after a write",
    );
}

/// The indexes made through DDL before `compact()` stay, and a query plans
/// a text search over the text index.
#[test]
fn ddl_indexes_made_before_compact_stay_after_it() {
    let mut db = people();
    index_through_ddl(|statement| db.execute(statement));
    db.compact().unwrap();
    db.execute(VINCENT).unwrap();
    assert_person_indexes(
        &db,
        &["Alix", "Vincent", "Mia", "Gus"],
        &["Alix", "Vincent"],
        "after compact",
    );

    let search = "MATCH (p:Person) WHERE text_score(p.bio, 'canals') > 0.0 \
                  RETURN p.name AS name ORDER BY name";
    let plan = db.execute(&format!("PROFILE {search}")).unwrap();
    assert!(
        matches!(&plan.rows()[0][0], Value::String(plan) if plan.contains("TextScan")),
        "the query searches the text index: {:?}",
        plan.rows()
    );
    assert_eq!(
        db.execute(search).unwrap().rows(),
        [[Value::from("Alix")], [Value::from("Vincent")]]
    );
}

/// A named graph's indexes, made through DDL and the API before
/// `compact()`, stay in it.
#[test]
fn named_graph_indexes_stay_after_compact() {
    let mut db = people();
    db.create_graph("model").unwrap();
    {
        let model = db.graph("model").unwrap();
        model.execute(PEOPLE).unwrap();
        index_through_ddl(|statement| model.execute(statement));
        model
            .session()
            .unwrap()
            .create_property_index("bio")
            .unwrap();
    }
    db.compact().unwrap();
    let model = db.graph("model").unwrap();
    model.execute(VINCENT).unwrap();

    let session = model.session().unwrap();
    assert!(session.has_property_index("name"), "the DDL property index");
    assert!(session.has_property_index("bio"), "the API property index");
    assert!(
        !db.has_property_index("bio"),
        "the named graph's index stays in it"
    );
    assert_eq!(
        session
            .find_nodes_by_property("name", &Value::from("Vincent"))
            .len(),
        1,
        "the property index finds Vincent"
    );
    let store = model.graph_store().unwrap();
    assert_eq!(
        store.text_search("Person", "bio", "canals", 19).len(),
        2,
        "the text index finds Alix and Vincent"
    );
    assert!(
        store.has_vector_index("Person", "embedding"),
        "the vector index"
    );
    assert_eq!(
        store
            .vector_search(
                Some("Person"),
                "embedding",
                &[0.0, 0.0],
                19,
                DistanceMetric::Euclidean,
            )
            .len(),
        4,
        "the vector index finds everyone"
    );
}

/// The constraints that read an index hold after `compact()`: the vector
/// index refuses a vector of another size, and UNIQUE, which looks the value
/// up in the property index, refuses the name of a compacted person and of
/// one written after `compact()`.
#[test]
fn constraints_that_read_an_index_hold_after_compact() {
    let mut db = people();
    index_through_the_api(&db);
    db.execute("CREATE CONSTRAINT person_name FOR (p:Person) ON (p.name) UNIQUE")
        .unwrap();
    db.compact().unwrap();
    db.execute(VINCENT).unwrap();

    let error = db
        .execute("INSERT (:Person {name: 'Jules', embedding: vector([3.0, 19.0, 88.0])})")
        .expect_err("a vector of three dimensions in a two-dimensional index");
    assert!(error.to_string().contains("dimensions"), "{error}");
    for name in ["Gus", "Vincent"] {
        let error = db
            .execute(&format!("INSERT (:Person {{name: '{name}'}})"))
            .expect_err("a second person with a name taken");
        assert!(error.to_string().contains("UNIQUE"), "{name}: {error}");
    }
    assert_eq!(named(&db, "Jules"), Vec::<String>::new());
}

/// The indexes stay across writes after `compact()`, `recompact()` and a
/// second `compact()`, and Gus, deleted before it, leaves the text and
/// vector indexes.
#[test]
fn indexes_stay_across_recompact_and_another_compact() {
    let mut db = people();
    index_through_the_api(&db);
    db.compact().unwrap();
    db.execute(VINCENT).unwrap();

    let everyone = ["Alix", "Vincent", "Mia", "Gus"];
    assert_person_indexes(&db, &everyone, &["Alix", "Vincent"], "after a write");

    db.execute("MATCH (m:Person {name: 'Mia'}) SET m.bio = 'canals of Paris'")
        .unwrap();
    db.compact().unwrap();
    assert_person_indexes(
        &db,
        &everyone,
        &["Alix", "Mia", "Vincent"],
        "after recompact",
    );

    db.execute("MATCH (g:Person {name: 'Gus'}) DELETE g")
        .unwrap();
    db.compact().unwrap();
    assert_person_indexes(
        &db,
        &["Alix", "Vincent", "Mia"],
        &["Alix", "Mia", "Vincent"],
        "after a second compact",
    );
    assert_eq!(
        bios_with(&db, "Prague", "after a second compact"),
        Vec::<String>::new(),
        "Gus left the text index"
    );
    assert_api_configuration(&db, "after a second compact");
}

#[cfg(all(feature = "wal", feature = "grafeo-file"))]
mod file {
    use std::path::{Path, PathBuf};
    use std::process::Command;

    use grafeo_common::testing::child_process;

    use super::*;

    /// A database at `path` holding [`PEOPLE`], indexed through the API and
    /// compacted, with Vincent written after `compact()`.
    fn compacted_file(path: &Path) -> GrafeoDB {
        let mut db = GrafeoDB::open(path).unwrap();
        db.execute(PEOPLE).unwrap();
        index_through_the_api(&db);
        db.compact().unwrap();
        db.execute(VINCENT).unwrap();
        db
    }

    /// The indexes of [`compacted_file`] work in `db`.
    fn assert_compacted_file(db: &GrafeoDB, stage: &str) {
        assert_person_indexes(
            db,
            &["Alix", "Vincent", "Mia", "Gus"],
            &["Alix", "Vincent"],
            stage,
        );
        assert_api_configuration(db, stage);
    }

    /// The indexes survive a close (which checkpoints) and a reopen, and a
    /// second one after a write and a `compact()` of the reopened database.
    #[test]
    fn indexes_made_before_compact_survive_a_reopen() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("people.grafeo");
        compacted_file(&path).close().unwrap();

        let mut db = GrafeoDB::open(&path).unwrap();
        assert_compacted_file(&db, "after the reopen");
        db.execute(
            "INSERT (:Person {name: 'Jules', bio: 'canals of Prague', \
                    embedding: vector([88.0, 3.0])})",
        )
        .unwrap();
        db.compact().unwrap();
        db.close().unwrap();

        let db = GrafeoDB::open(&path).unwrap();
        let stage = "after a compact of the reopened database and a reopen";
        assert_eq!(
            bios_with(&db, "canals", stage),
            ["Alix", "Jules", "Vincent"],
            "{stage}"
        );
        assert_eq!(named(&db, "Jules"), ["Jules"], "{stage}");
        assert_api_configuration(&db, stage);
        db.close().unwrap();
    }

    /// The database path the child process works on.
    const PATH_VAR: &str = "GRAFEO_COMPACT_KEEPS_INDEXES_PATH";
    /// Exit code of a child that reached its end.
    const EXITED: i32 = 19;

    /// The WAL next to the database file.
    fn sidecar_wal(path: &Path) -> PathBuf {
        let mut sidecar = path.as_os_str().to_owned();
        sidecar.push(".wal");
        PathBuf::from(sidecar)
    }

    /// Child-process entry for
    /// [`indexes_made_before_compact_survive_a_checkpoint_and_a_crash`]; a
    /// no-op when run directly. It indexes, compacts, writes and
    /// checkpoints, then exits without `close()`, like a crash.
    #[test]
    fn compact_keeps_indexes_child() {
        let Some(path) = std::env::var_os(PATH_VAR) else {
            return;
        };
        let db = compacted_file(&PathBuf::from(path));
        db.wal_checkpoint().unwrap();
        std::process::exit(EXITED);
    }

    /// A checkpoint after `compact()`, then a crash: the reopened file has
    /// the indexes.
    #[test]
    fn indexes_made_before_compact_survive_a_checkpoint_and_a_crash() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("people.grafeo");
        let output = child_process::output(
            Command::new(std::env::current_exe().unwrap())
                .args([
                    "--exact",
                    "file::compact_keeps_indexes_child",
                    "--nocapture",
                ])
                .env(PATH_VAR, &path),
        )
        .unwrap();
        assert_eq!(
            output.status.code(),
            Some(EXITED),
            "the child exited early:\n{}",
            String::from_utf8_lossy(&output.stderr)
        );
        assert!(
            sidecar_wal(&path).exists(),
            "the child left its WAL, as a crash does"
        );
        let db = GrafeoDB::open(&path).unwrap();
        assert_compacted_file(&db, "after the crash");
        db.close().unwrap();
    }
}
