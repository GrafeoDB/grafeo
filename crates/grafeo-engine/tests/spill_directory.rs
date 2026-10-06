//! A statement that spills nothing touches no spill directory (#565): every
//! GQL statement on a file database created and removed `<file>.spill/query_<n>`
//! (about 0.25 ms each) and left `<file>.spill` next to the database.

#![cfg(all(
    feature = "lpg",
    feature = "gql",
    feature = "grafeo-file",
    feature = "spill"
))]

use std::collections::HashMap;

use grafeo_common::types::{PropertyKey, Value};
use grafeo_engine::GrafeoDB;

#[test]
fn statements_that_spill_nothing_leave_no_spill_directory() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("amsterdam.grafeo");
    let spill = dir.path().join("amsterdam.grafeo.spill");

    let db = GrafeoDB::open(&path).unwrap();
    db.create_property_index("name").unwrap();
    db.execute("INSERT (:City {name: 'Amsterdam'}), (:City {name: 'Berlin'})")
        .unwrap();
    db.execute("MATCH (c:City) RETURN c.name ORDER BY c.name")
        .unwrap();
    db.execute_with_params(
        "MATCH (c:City {name: $name}) SET c.visited = true",
        HashMap::from([("name".to_string(), Value::from("Prague"))]),
    )
    .unwrap();
    let row = HashMap::from([
        (PropertyKey::new("name"), Value::from("Paris")),
        (PropertyKey::new("country"), Value::from("France")),
    ]);
    db.upsert_nodes(&["City"], "name", vec![row], false)
        .unwrap();
    assert!(!spill.exists(), "a statement created {}", spill.display());

    db.close().unwrap();
    assert!(!spill.exists(), "close left {}", spill.display());
}

/// A clean close of a database that ran nothing leaves only the file.
#[test]
fn opening_and_closing_leaves_no_spill_directory() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("berlin.grafeo");
    let db = GrafeoDB::open(&path).unwrap();
    db.close().unwrap();
    drop(db);
    assert!(!dir.path().join("berlin.grafeo.spill").exists());
}

/// A read-only open writes nothing beside the database, also when it runs
/// queries.
#[test]
fn a_read_only_open_creates_no_spill_directory() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("prague.grafeo");
    let spill = dir.path().join("prague.grafeo.spill");
    {
        let db = GrafeoDB::open(&path).unwrap();
        db.execute("INSERT (:City {name: 'Prague'})").unwrap();
        db.close().unwrap();
    }
    assert!(!spill.exists());

    let db = GrafeoDB::open_read_only(&path).unwrap();
    let rows = db.execute("MATCH (c:City) RETURN c.name").unwrap();
    assert_eq!(rows.rows().len(), 1);
    #[cfg(feature = "cypher")]
    db.execute_cypher("MATCH (c:City) RETURN c.name").unwrap();
    assert!(
        !spill.exists(),
        "a read-only query created {}",
        spill.display()
    );
    drop(db);
    assert!(!spill.exists());
}

/// A query of a read-only open that spills writes into the open's own
/// directory in the system temp directory, never beside the database, and
/// that directory goes with the database.
#[test]
fn a_read_only_open_spills_into_the_temp_directory() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("barcelona.grafeo");
    {
        let db = GrafeoDB::open(&path).unwrap();
        db.execute("INSERT (:City {name: 'Barcelona'})").unwrap();
        db.close().unwrap();
    }

    let db = GrafeoDB::open_read_only(&path).unwrap();
    let spill = db
        .buffer_manager()
        .config()
        .spill_path
        .clone()
        .expect("a read-only open can spill");
    assert_eq!(spill.parent(), Some(std::env::temp_dir().as_path()));
    let name = spill.file_name().unwrap().to_string_lossy().into_owned();
    assert!(
        name.starts_with(&format!("grafeo-barcelona.grafeo-{}-", std::process::id())),
        "{name}"
    );
    // What a spilling query leaves once its own subdirectory is gone.
    std::fs::create_dir_all(&spill).unwrap();
    drop(db);
    assert!(!spill.exists(), "the temp directory went with the database");
    assert!(!dir.path().join("barcelona.grafeo.spill").exists());
}
