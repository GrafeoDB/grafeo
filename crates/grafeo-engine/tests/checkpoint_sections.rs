//! Every checkpoint writes the database's complete state.
//!
//! A checkpoint writes a new `.grafeo` container that holds only the sections
//! it was given, so a checkpoint that leaves a section out drops it from the
//! file. Compacted databases lost their catalog (schema) and RDF data this way,
//! and the periodic timer kept writing the pre-compaction store.
//!
//! ```bash
//! cargo test -p grafeo-engine --features full --test checkpoint_sections
//! ```

#![cfg(all(
    feature = "compact-store",
    feature = "lpg",
    feature = "grafeo-file",
    feature = "gql"
))]

use std::time::{Duration, Instant};

use grafeo_common::storage::SectionType;
use grafeo_common::types::Value;
use grafeo_engine::config::StorageFormat;
use grafeo_engine::{Config, GrafeoDB};

fn config(path: &std::path::Path) -> Config {
    Config::persistent(path).with_storage_format(StorageFormat::SingleFile)
}

fn single_value(db: &GrafeoDB, query: &str) -> Value {
    db.session().execute(query).unwrap().rows()[0][0].clone()
}

#[test]
fn a_compacted_database_keeps_its_schema() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("db.grafeo");
    {
        let mut db = GrafeoDB::with_config(config(&path)).unwrap();
        db.session()
            .execute("CREATE NODE TYPE Person (name STRING)")
            .unwrap();
        db.session()
            .execute("INSERT (:Person {name: 'Alix'}), (:Person {name: 'Gus'})")
            .unwrap();
        db.compact().unwrap();
        db.close().unwrap();
    }

    let db = GrafeoDB::with_config(config(&path)).unwrap();
    assert_eq!(
        single_value(&db, "MATCH (p:Person) RETURN count(p)"),
        Value::Int64(2)
    );
    let types = db.session().execute("SHOW NODE TYPES").unwrap();
    assert_eq!(types.rows().len(), 1, "the node type survives: {types:?}");
}

/// The file keeps the RDF section with its triples. (Querying the triples
/// after `compact()` is a separate problem: they are hidden in memory already.)
#[test]
#[cfg(feature = "sparql")]
fn a_compacted_database_keeps_its_rdf_section() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("db.grafeo");
    {
        let mut db = GrafeoDB::with_config(config(&path)).unwrap();
        db.session()
            .execute_sparql("INSERT DATA { <http://ex/alix> <http://ex/knows> <http://ex/gus> }")
            .unwrap();
        db.session()
            .execute("INSERT (:Person {name: 'Alix'})")
            .unwrap();
        db.compact().unwrap();
        db.close().unwrap();
    }

    let db = GrafeoDB::with_config(config(&path)).unwrap();
    let directory = db
        .file_manager()
        .unwrap()
        .read_section_directory()
        .unwrap()
        .unwrap();
    assert!(directory.find(SectionType::RdfStore).is_some());
    // Read the store itself: SPARQL over a compacted database does not see
    // RDF data yet, a separate problem.
    assert_eq!(db.rdf_store().len(), 1, "the triple is in the section");
}

/// The header of a compacted database's checkpoint counts the compacted
/// base as well as the overlay.
#[test]
fn a_compacted_checkpoint_counts_the_base() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("db.grafeo");
    let mut db = GrafeoDB::with_config(config(&path)).unwrap();
    db.session()
        .execute("INSERT (:Person {name: 'Alix'})-[:KNOWS]->(:Person {name: 'Gus'})")
        .unwrap();
    db.compact().unwrap();
    db.session()
        .execute("INSERT (:Person {name: 'Vincent'})")
        .unwrap();
    db.wal_checkpoint().unwrap();

    let header = db.file_manager().unwrap().active_header();
    assert_eq!((header.node_count, header.edge_count), (3, 1));
    db.close().unwrap();
}

/// The timer captured the store when the database opened, so after
/// `compact()` its checkpoints wrote the old store without the compacted
/// base. A crash after such a checkpoint lost the base: the WAL files that
/// held its data were deleted.
#[test]
fn the_checkpoint_timer_writes_the_compacted_base() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("db.grafeo");
    let mut db =
        GrafeoDB::with_config(config(&path).with_checkpoint_interval(Duration::from_millis(100)))
            .unwrap();
    db.session()
        .execute("INSERT (:Person {name: 'Alix'})-[:KNOWS]->(:Person {name: 'Gus'})")
        .unwrap();
    db.compact().unwrap();

    let fm = std::sync::Arc::clone(db.file_manager().unwrap());
    let compacted_at = fm.active_header().iteration;
    let deadline = Instant::now() + Duration::from_secs(10);
    while fm.active_header().iteration <= compacted_at {
        assert!(Instant::now() < deadline, "the timer did not checkpoint");
        std::thread::sleep(Duration::from_millis(20));
    }

    let directory = fm.read_section_directory().unwrap().unwrap();
    assert!(
        directory.find(SectionType::CompactStore).is_some(),
        "the timer's checkpoint contains the compacted base"
    );
    assert!(directory.find(SectionType::Catalog).is_some());
    db.close().unwrap();
}
