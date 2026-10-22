//! Snapshots of version 5 hold the catalog as the records a database file
//! holds (#551): everything `SHOW` lists comes back after an export and an
//! import or a restore, with what version 4 lost (named constraints, index
//! names, the options of a text index, the indexes of a named graph). The
//! stored configuration stays with the file. Version 4 still imports.
//!
//! ```bash
//! cargo test -p grafeo-engine --all-features --test snapshot_v5
//! ```

#![cfg(all(feature = "gql", feature = "vector-index", feature = "text-index"))]
#![allow(
    missing_docs,
    reason = "a test crate has no public API: each test documents itself"
)]

use grafeo_common::types::Value;
use grafeo_engine::GrafeoDB;

/// The schema of `catalog_v2.rs` (types with defaults, a parent type,
/// endpoints, graph types, a typed graph, named and unnamed constraints,
/// property, vector and text indexes with names, a schema, a procedure),
/// plus a text index with options and an index in a named graph.
const SCHEMA: &[&str] = &[
    "CREATE NODE TYPE City (name STRING NOT NULL, country STRING DEFAULT 'NL')",
    "CREATE NODE TYPE Capital EXTENDS City (since INT64)",
    "CREATE EDGE TYPE ROUTE CONNECTING (City) TO (City) (km INT64 DEFAULT 88)",
    "CREATE GRAPH TYPE travel (NODE TYPE City, EDGE TYPE ROUTE)",
    "CREATE CONSTRAINT person_email FOR (p:Person) ON (p.email) UNIQUE",
    "CREATE CONSTRAINT FOR (p:Person) ON (p.name) NOT NULL",
    "CREATE INDEX person_name FOR (p:Person) ON (p.name)",
    "INSERT (:Document {title: 'Canals', content: 'boats on the canals of Amsterdam', \
     embedding: vector([3.0, 19.0, 88.0])}), \
     (:Document {title: 'Bridges', content: 'bridges over the river in Prague', \
     embedding: vector([88.0, 19.0, 3.0])})",
    "CREATE VECTOR INDEX doc_embedding ON :Document(embedding)",
    "CREATE INDEX doc_content FOR (d:Document) ON (d.content) USING TEXT",
    "INSERT (:Note {body: 'Berlin and Paris'})",
    "CREATE INDEX notes FOR (n:Note) ON (n.body) USING TEXT {k1: 1.5, b: 0.5, tokenizer: 'standard'}",
    "CREATE GRAPH trips TYPED travel",
    "CREATE SCHEMA archive",
    "CREATE PROCEDURE capitals() RETURNS (name STRING) AS { MATCH (c:Capital) RETURN c.name AS name }",
];

/// The `SHOW` statements whose output a snapshot must keep.
const SHOWN: &[&str] = &[
    "SHOW NODE TYPES",
    "SHOW EDGE TYPES",
    "SHOW GRAPH TYPES",
    "SHOW CONSTRAINTS",
    "SHOW INDEXES",
    "SHOW GRAPHS",
    "SHOW SCHEMAS",
];

/// Every row of every statement of [`SHOWN`], one line each, the rows of a
/// statement sorted (`SHOW INDEXES` lists them in hash-map order).
fn shown(db: &GrafeoDB) -> Vec<String> {
    SHOWN
        .iter()
        .flat_map(|query| {
            let mut lines: Vec<String> = db
                .execute(query)
                .unwrap_or_else(|error| panic!("{query}: {error}"))
                .rows()
                .iter()
                .map(|row| format!("{query}: {row:?}"))
                .collect();
            lines.sort();
            lines
        })
        .collect()
}

fn with_schema() -> GrafeoDB {
    let db = GrafeoDB::new_in_memory();
    for statement in SCHEMA {
        db.execute(statement)
            .unwrap_or_else(|error| panic!("{statement}: {error}"));
    }
    db
}

/// An export, then an import, keeps everything `SHOW` lists, and the
/// imported database exports the same bytes again.
#[test]
fn an_import_keeps_the_whole_catalog() {
    let db = with_schema();
    let bytes = db.export_snapshot().unwrap();
    assert_eq!(bytes[0], 5, "a snapshot of version 5");
    let imported = GrafeoDB::import_snapshot(&bytes).unwrap();
    assert_eq!(shown(&imported), shown(&db));
    assert_eq!(
        imported.export_snapshot().unwrap(),
        bytes,
        "the import holds what the export held"
    );
}

/// The export of the same data gives the same bytes: named graphs came in
/// the order of a hash map, which an import builds anew, so an imported
/// database exported other bytes than the one it was imported from.
#[test]
fn the_export_of_the_same_data_gives_the_same_bytes() {
    let db = GrafeoDB::new_in_memory();
    for city in [
        "amsterdam",
        "berlin",
        "paris",
        "prague",
        "barcelona",
        "vienna",
        "lisbon",
        "oslo",
    ] {
        db.execute(&format!("CREATE GRAPH {city}")).unwrap();
        let session = db.session();
        session.use_graph(city);
        session
            .execute(&format!("INSERT (:City {{name: '{city}'}})"))
            .unwrap();
    }
    let bytes = db.export_snapshot().unwrap();
    for round in 0..3 {
        let imported = GrafeoDB::import_snapshot(&bytes).unwrap();
        assert_eq!(imported.export_snapshot().unwrap(), bytes, "round {round}");
    }
}

/// The definitions version 4 lost do their work after an import: the named
/// unique constraint refuses a duplicate, the named index can be dropped by
/// its name, and the text index with options searches.
#[test]
fn what_version_4_lost_works_after_an_import() {
    let db = with_schema();
    let imported = GrafeoDB::import_snapshot(&db.export_snapshot().unwrap()).unwrap();

    imported
        .execute("INSERT (:Person {name: 'Alix', email: 'alix@example.org'})")
        .unwrap();
    assert!(
        imported
            .execute("INSERT (:Person {name: 'Gus', email: 'alix@example.org'})")
            .is_err(),
        "the unique constraint person_email holds"
    );
    imported.execute("DROP INDEX person_name").unwrap();

    let found = imported
        .text_search("Note", "body", "paris", 3, None)
        .unwrap();
    assert_eq!(found.len(), 1, "the text index with options was built");
}

/// A restore into a database whose catalog holds the same definitions
/// already applies the snapshot's catalog over it, and keeps the rest.
#[test]
fn a_restore_applies_the_catalog_over_what_is_there() {
    let db = with_schema();
    let bytes = db.export_snapshot().unwrap();
    let target = with_schema();
    target.restore_snapshot(&bytes).unwrap();
    assert_eq!(shown(&target), shown(&db));
    assert_eq!(
        target
            .execute("MATCH (d:Document) RETURN count(d)")
            .unwrap()
            .rows()[0][0],
        Value::Int64(2),
        "the data is the snapshot's, not doubled"
    );
}

/// The stored configuration stays with the database file: a snapshot of a
/// file with a memory limit does not carry it into a file saved from the
/// import.
#[cfg(all(feature = "wal", feature = "grafeo-file"))]
#[test]
fn a_snapshot_leaves_the_stored_configuration_out() {
    let dir = tempfile::tempdir().unwrap();
    let source = GrafeoDB::with_config(
        grafeo_engine::Config::persistent(dir.path().join("prague.grafeo"))
            .with_memory_limit(88 * 1024 * 1024),
    )
    .unwrap();
    source.execute("INSERT (:City {name: 'Prague'})").unwrap();
    let imported = GrafeoDB::import_snapshot(&source.export_snapshot().unwrap()).unwrap();
    let copy = dir.path().join("copy.grafeo");
    imported.save(&copy).unwrap();
    let reopened = GrafeoDB::open(&copy).unwrap();
    assert_eq!(reopened.config().memory_limit(), None);
    assert_eq!(reopened.node_count().unwrap(), 1);
}

/// A version 4 snapshot still restores.
#[test]
fn a_v4_snapshot_still_restores() {
    let bytes = include_bytes!("fixtures/snapshot_v4.bin");
    let db = with_schema();
    db.restore_snapshot(bytes).unwrap();
    assert_eq!((db.node_count().unwrap(), db.edge_count().unwrap()), (3, 2));
}
