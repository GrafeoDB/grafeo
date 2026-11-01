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
/// already gives the snapshot's catalog, and its data once.
#[test]
fn a_restore_over_the_same_catalog_gives_the_snapshots() {
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

/// The database a snapshot of another catalog is taken of: a node type of
/// its own, data a unique and a NOT NULL constraint on `Person` would
/// refuse (two people with one email, one without a name), the documents
/// and notes the indexes of [`SCHEMA`] index, and an untyped graph `trips`.
const OTHER_CATALOG: &[&str] = &[
    "CREATE NODE TYPE Place (name STRING)",
    "INSERT (:Person {name: 'Vincent', email: 'shared@example.org'}), \
     (:Person {name: 'Jules', email: 'shared@example.org'}), \
     (:Person {email: 'mia@example.org'})",
    "INSERT (:Document {title: 'Canals', content: 'boats on the canals of Paris', \
     embedding: vector([3.0, 19.0, 88.0])})",
    "INSERT (:Note {body: 'Berlin and Amsterdam'})",
    "CREATE GRAPH trips",
];

fn run(db: &GrafeoDB, statements: &[&str]) {
    for statement in statements {
        db.execute(statement)
            .unwrap_or_else(|error| panic!("{statement}: {error}"));
    }
}

fn other_catalog() -> GrafeoDB {
    let db = GrafeoDB::new_in_memory();
    run(&db, OTHER_CATALOG);
    db
}

/// A database with the whole catalog of [`SCHEMA`], restored from a snapshot
/// of [`OTHER_CATALOG`], and the database the snapshot was taken of.
fn restored_over_schema() -> (GrafeoDB, GrafeoDB) {
    let source = other_catalog();
    let target = with_schema();
    target
        .restore_snapshot(&source.export_snapshot().unwrap())
        .expect("the restore succeeds");
    (target, source)
}

/// The rows of `query` on `db`, one line each, sorted.
fn rows(db: &GrafeoDB, query: &str) -> Vec<String> {
    let mut lines: Vec<String> = db
        .execute(query)
        .unwrap_or_else(|error| panic!("{query}: {error}"))
        .rows()
        .iter()
        .map(|row| format!("{row:?}"))
        .collect();
    lines.sort();
    lines
}

/// The error of a vector search for the documents of [`SCHEMA`]'s vector
/// index, or `None` when the search runs.
fn vector_search_error(db: &GrafeoDB) -> Option<String> {
    db.vector_search("Document", "embedding", &[3.0, 19.0, 88.0], 1, None, None)
        .err()
        .map(|error| error.to_string())
}

/// A restore replaces the catalog: the restored database lists what the
/// snapshot's database lists, and nothing of the catalog it had before.
#[test]
fn a_restore_replaces_the_catalog() {
    let (target, source) = restored_over_schema();
    assert_eq!(shown(&target), shown(&source));
}

/// A node type the snapshot does not have is gone after a restore.
#[test]
fn a_restore_drops_the_node_types_the_snapshot_has_not() {
    let (target, source) = restored_over_schema();
    assert_eq!(
        rows(&target, "SHOW NODE TYPES"),
        rows(&source, "SHOW NODE TYPES")
    );
}

/// An edge type the snapshot does not have is gone after a restore.
#[test]
fn a_restore_drops_the_edge_types_the_snapshot_has_not() {
    let (target, source) = restored_over_schema();
    assert_eq!(
        rows(&target, "SHOW EDGE TYPES"),
        rows(&source, "SHOW EDGE TYPES")
    );
}

/// A graph type the snapshot does not have is gone after a restore, and so
/// is the binding of a graph the snapshot holds untyped: `trips` takes a
/// `Person`, which the graph type `travel` has not.
#[test]
fn a_restore_drops_the_graph_types_and_bindings_the_snapshot_has_not() {
    let (target, source) = restored_over_schema();
    assert_eq!(
        rows(&target, "SHOW GRAPH TYPES"),
        rows(&source, "SHOW GRAPH TYPES")
    );
    target
        .graph("trips")
        .unwrap()
        .execute("INSERT (:Person {name: 'Gus'})")
        .expect("trips is untyped in the snapshot");
}

/// A constraint the snapshot does not have is gone after a restore: the
/// writes the snapshot's database takes, the restored one takes too.
#[test]
fn a_restore_drops_the_constraints_the_snapshot_has_not() {
    let (target, source) = restored_over_schema();
    assert_eq!(
        rows(&target, "SHOW CONSTRAINTS"),
        rows(&source, "SHOW CONSTRAINTS")
    );
    target
        .execute("INSERT (:Person {name: 'Gus', email: 'shared@example.org'})")
        .expect("no unique constraint on Person.email");
    target
        .execute("INSERT (:Person {email: 'gus@example.org'})")
        .expect("no NOT NULL constraint on Person.name");
}

/// Data a unique constraint the snapshot never had would refuse restores
/// whole, and the constraint does not come back to refuse the next write
/// of the same kind.
#[test]
fn data_a_constraint_the_snapshot_never_had_would_refuse_restores() {
    let (target, _source) = restored_over_schema();
    assert_eq!(
        target
            .execute("MATCH (p:Person {email: 'shared@example.org'}) RETURN count(p)")
            .unwrap()
            .rows()[0][0],
        Value::Int64(2),
        "both people with the shared email are restored"
    );
    target
        .execute("INSERT (:Person {name: 'Mia', email: 'shared@example.org'})")
        .expect("the snapshot's database takes a third person with the email");
}

/// A procedure the snapshot does not have is gone after a restore.
#[test]
fn a_restore_drops_the_procedures_the_snapshot_has_not() {
    let (target, _source) = restored_over_schema();
    assert!(
        target
            .execute("CALL capitals() YIELD name RETURN name")
            .is_err(),
        "the procedure capitals is not in the snapshot"
    );
}

/// A property and a text index the snapshot does not have are gone after a
/// restore.
#[test]
fn a_restore_drops_the_property_and_text_indexes_the_snapshot_has_not() {
    let (target, source) = restored_over_schema();
    assert_eq!(rows(&target, "SHOW INDEXES"), rows(&source, "SHOW INDEXES"));
    assert!(
        target.execute("DROP INDEX person_name").is_err(),
        "the index person_name is not in the snapshot"
    );
    assert!(
        target
            .text_search("Document", "content", "canals", 1, None)
            .is_err(),
        "the text index on Document.content is not in the snapshot"
    );
}

/// A vector index the snapshot does not have is gone after a restore
/// (`SHOW INDEXES` leaves vector indexes out).
#[test]
fn a_restore_drops_the_vector_indexes_the_snapshot_has_not() {
    let (target, _source) = restored_over_schema();
    let error = vector_search_error(&target);
    assert!(
        error
            .as_deref()
            .is_some_and(|error| error.contains("there is no vector index")),
        "the vector index on Document.embedding is not in the snapshot: {error:?}"
    );
}

/// A schema the snapshot does not have is gone after a restore.
#[test]
fn a_restore_drops_the_schemas_the_snapshot_has_not() {
    let (target, source) = restored_over_schema();
    assert_eq!(rows(&target, "SHOW SCHEMAS"), rows(&source, "SHOW SCHEMAS"));
}

/// Control: what the snapshot holds is there after a restore into a
/// database with another catalog, and works.
#[test]
fn a_restore_keeps_every_definition_the_snapshot_holds() {
    let source = with_schema();
    let target = other_catalog();
    target
        .restore_snapshot(&source.export_snapshot().unwrap())
        .unwrap();
    let restored = shown(&target);
    for line in shown(&source) {
        assert!(restored.contains(&line), "{line} is restored");
    }
    assert_eq!(vector_search_error(&target), None);
    assert_eq!(
        target
            .text_search("Note", "body", "paris", 3, None)
            .unwrap()
            .len(),
        1
    );
    target
        .execute("CALL capitals() YIELD name RETURN name")
        .expect("the procedure capitals is restored");
}

/// A restore into a database file replaces its catalog for good: after a
/// close and an open, the file lists what the snapshot's database lists,
/// and nothing of the catalog it had before.
#[cfg(all(feature = "wal", feature = "grafeo-file"))]
#[test]
fn a_restored_file_reopens_with_the_snapshots_catalog_alone() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("amsterdam.grafeo");
    let source = other_catalog();
    {
        let db = GrafeoDB::with_config(grafeo_engine::Config::persistent(&path)).unwrap();
        run(&db, SCHEMA);
        db.restore_snapshot(&source.export_snapshot().unwrap())
            .unwrap();
        db.close().unwrap();
    }
    let reopened = GrafeoDB::open(&path).unwrap();
    assert_eq!(shown(&reopened), shown(&source));
    let error = vector_search_error(&reopened);
    assert!(
        error
            .as_deref()
            .is_some_and(|error| error.contains("there is no vector index")),
        "the vector index on Document.embedding is not in the snapshot: {error:?}"
    );
    assert!(
        reopened
            .execute("CALL capitals() YIELD name RETURN name")
            .is_err(),
        "the procedure capitals is not in the snapshot"
    );
    reopened
        .execute("INSERT (:Person {name: 'Gus', email: 'shared@example.org'})")
        .expect("no unique constraint on Person.email");
    reopened
        .graph("trips")
        .unwrap()
        .execute("INSERT (:Person {name: 'Gus'})")
        .expect("trips is untyped in the snapshot");
}

/// Control: a restore into a database file keeps what the snapshot holds
/// after a close and an open.
#[cfg(all(feature = "wal", feature = "grafeo-file"))]
#[test]
fn a_restored_file_reopens_with_every_definition_the_snapshot_holds() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("berlin.grafeo");
    let source = with_schema();
    {
        let db = GrafeoDB::with_config(grafeo_engine::Config::persistent(&path)).unwrap();
        run(&db, OTHER_CATALOG);
        db.restore_snapshot(&source.export_snapshot().unwrap())
            .unwrap();
        db.close().unwrap();
    }
    let reopened = GrafeoDB::open(&path).unwrap();
    let restored = shown(&reopened);
    for line in shown(&source) {
        assert!(restored.contains(&line), "{line} is restored");
    }
    assert_eq!(vector_search_error(&reopened), None);
    reopened
        .execute("CALL capitals() YIELD name RETURN name")
        .expect("the procedure capitals is restored");
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
