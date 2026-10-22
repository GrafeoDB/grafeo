//! What only the built-in store has (indexes, search, the database file and
//! its copies, bulk imports, property history) answers the unsupported
//! error on a database on a backend of its own (`GrafeoDB::with_store`),
//! never a panic, and leaves the graph as it was: the queries and the direct
//! API go on working on the backend.
//!
//! ```bash
//! cargo test -p grafeo-engine --all-features --test backend_built_in_only
//! ```

#![cfg(feature = "gql")]

use std::sync::Arc;

use grafeo_common::types::Value;
use grafeo_common::utils::error::{Error, ErrorCode};
use grafeo_core::graph::GraphBackend;
use grafeo_core::graph::rowgroup::RowGroupStore;
use grafeo_engine::{Config, GrafeoDB};

/// A database on a backend of its own with Alix and Gus, who know each
/// other.
fn on_backend() -> GrafeoDB {
    let db = GrafeoDB::with_store(
        Arc::new(RowGroupStore::new()) as Arc<dyn GraphBackend>,
        Config::in_memory(),
    )
    .unwrap();
    db.execute(
        "INSERT (:Person {name: 'Alix', bio: 'graph databases'})-[:KNOWS]->\
         (:Person {name: 'Gus', bio: 'row groups'})",
    )
    .unwrap();
    db
}

/// The unsupported error that names the built-in store.
fn assert_needs_built_in<T>(outcome: Result<T, Error>, call: &str) {
    let Err(error) = outcome else {
        panic!("{call} succeeded on a backend of its own");
    };
    assert_eq!(
        error.error_code(),
        ErrorCode::QueryUnsupported,
        "{call}: {error}"
    );
    assert!(
        error.to_string().contains("built-in store"),
        "{call}: {error}"
    );
}

/// The names of the people, in order: the graph is untouched.
fn people(db: &GrafeoDB) -> Vec<Value> {
    db.execute("MATCH (p:Person) RETURN p.name ORDER BY p.name")
        .unwrap()
        .rows()
        .iter()
        .map(|row| row[0].clone())
        .collect()
}

/// A property index lives in the built-in store: creating or dropping one
/// is an error and asking for one is `false`, while a lookup by value reads
/// the backend's own rows.
#[test]
fn a_property_index_needs_the_built_in_store() {
    let db = on_backend();
    assert_needs_built_in(db.create_property_index("name"), "create_property_index");
    assert_needs_built_in(db.drop_property_index("name"), "drop_property_index");
    assert!(!db.has_property_index("name"));
    let found = db
        .find_nodes_by_property("name", &Value::from("Alix"))
        .unwrap();
    assert_eq!(found.len(), 1, "{found:?}");
    assert_eq!(
        db.get_node(found[0])
            .unwrap()
            .and_then(|alix| alix.get_property("name").cloned()),
        Some(Value::from("Alix"))
    );
    assert_eq!(people(&db), [Value::from("Alix"), Value::from("Gus")]);
}

/// A vector index and a text index live in the built-in store, and so does
/// every search over them.
#[cfg(all(feature = "vector-index", feature = "text-index"))]
#[test]
fn search_needs_the_built_in_store() {
    let db = on_backend();
    assert_needs_built_in(
        db.create_vector_index("Person", "embedding", Some(2), None, None, None, None),
        "create_vector_index",
    );
    assert_needs_built_in(
        db.vector_search("Person", "embedding", &[1.0, 0.0], 1, None, None),
        "vector_search",
    );
    assert_needs_built_in(db.create_text_index("Person", "bio"), "create_text_index");
    assert_needs_built_in(
        db.text_search("Person", "bio", "graph", 1, None),
        "text_search",
    );
    assert_eq!(people(&db), [Value::from("Alix"), Value::from("Gus")]);
}

/// The database file, its snapshots, an in-memory copy and the iterators
/// over the built-in store's rows have nothing to read on a backend of its
/// own; restoring a snapshot into it is refused as well.
#[cfg(all(feature = "wal", feature = "grafeo-file"))]
#[test]
fn the_database_file_and_its_copies_need_the_built_in_store() {
    let db = on_backend();
    let dir = tempfile::tempdir().unwrap();
    assert_needs_built_in(db.save(dir.path().join("trips.grafeo")), "save");
    assert!(
        !dir.path().join("trips.grafeo").exists(),
        "a failed save leaves no file"
    );
    assert_needs_built_in(db.to_memory().map(drop), "to_memory");
    assert_needs_built_in(db.export_snapshot(), "export_snapshot");
    assert_needs_built_in(db.iter_nodes().map(|nodes| nodes.count()), "iter_nodes");
    assert_needs_built_in(db.iter_edges().map(|edges| edges.count()), "iter_edges");
    assert_needs_built_in(db.store(), "store");
    // The rows are the backend's, which the engine does not measure.
    assert_eq!(db.memory_usage().store.total_bytes, 0);

    let snapshot = GrafeoDB::new_in_memory().export_snapshot().unwrap();
    assert_needs_built_in(db.restore_snapshot(&snapshot), "restore_snapshot");
    assert_eq!(people(&db), [Value::from("Alix"), Value::from("Gus")]);
}

/// A bulk import writes the built-in store's rows: it is refused before it
/// writes anything.
#[test]
fn a_bulk_import_needs_the_built_in_store() {
    let db = on_backend();
    assert_needs_built_in(
        db.import_tsv_str("1\t2\n2\t3\n", "ROAD", true),
        "import_tsv_str",
    );
    let dir = tempfile::tempdir().unwrap();
    let matrix = dir.path().join("roads.mtx");
    std::fs::write(
        &matrix,
        "%%MatrixMarket matrix coordinate pattern general\n3 3 2\n1 2\n2 3\n",
    )
    .unwrap();
    assert_needs_built_in(db.import_mmio(&matrix, "ROAD"), "import_mmio");
    assert_eq!(
        db.execute("MATCH ()-[r:ROAD]->() RETURN count(r)")
            .unwrap()
            .rows(),
        [[Value::Int64(0)]]
    );
}

/// The direct API reads and writes the backend; only the property history,
/// which the built-in store keeps, is the unsupported error.
#[test]
fn the_direct_api_works_on_the_backend_but_its_history_needs_the_built_in_store() {
    let db = on_backend();
    let vincent = db
        .create_node_with_props(&["Person"], [("name", Value::from("Vincent"))])
        .unwrap();
    db.set_node_property(vincent, "city", Value::from("Amsterdam"))
        .unwrap();
    let node = db.get_node(vincent).unwrap().expect("Vincent is there");
    assert_eq!(
        node.get_property("city").cloned(),
        Some(Value::from("Amsterdam"))
    );
    assert_eq!(
        people(&db),
        [
            Value::from("Alix"),
            Value::from("Gus"),
            Value::from("Vincent")
        ]
    );

    let error = db
        .get_node_property_at_epoch(vincent, "city", db.current_epoch())
        .expect_err("a backend of its own keeps no property history");
    assert_eq!(error.error_code(), ErrorCode::QueryUnsupported, "{error}");
    let error = db
        .get_node_property_history(vincent, "city")
        .expect_err("a backend of its own keeps no property history");
    assert_eq!(error.error_code(), ErrorCode::QueryUnsupported, "{error}");
}

/// A backup copies the built-in store and follows its log: a backend with a
/// path, which logs its commits to a WAL of its own, refuses a full backup
/// and an incremental one, and writes nothing into the backup directory.
#[cfg(all(feature = "wal", feature = "grafeo-file"))]
#[test]
fn a_backup_needs_the_built_in_store() {
    let dir = tempfile::tempdir().unwrap();
    let db = GrafeoDB::with_store(
        Arc::new(RowGroupStore::new()) as Arc<dyn GraphBackend>,
        Config::persistent(dir.path().join("trips")),
    )
    .unwrap();
    db.execute("INSERT (:Person {name: 'Jules'})").unwrap();
    let backups = dir.path().join("backups");
    std::fs::create_dir_all(&backups).unwrap();
    assert_needs_built_in(db.backup_full(&backups), "backup_full");
    assert_needs_built_in(db.backup_incremental(&backups), "backup_incremental");
    assert_eq!(std::fs::read_dir(&backups).unwrap().count(), 0);
    assert_eq!(people(&db), [Value::from("Jules")]);
}
