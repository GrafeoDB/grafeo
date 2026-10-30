//! A catalog record of a snapshot that does not apply fails the import or
//! the restore, naming the record, instead of being left out: in version 5,
//! a graph bound to a graph type the catalog does not define (the database's
//! data without its type); in version 4, a graph type defined twice. A
//! binding a dropped graph type left in a version 4 snapshot still imports:
//! 0.5.x wrote those. The snapshots are written by hand (bincode of the
//! snapshot with no rows), with the catalog records a database never writes.
//! A restore that fails on such a record leaves the database as it was.
//!
//! ```bash
//! cargo test -p grafeo-engine --features full --test snapshot_catalog_errors
//! ```

#![cfg(feature = "gql")]
#![allow(
    missing_docs,
    reason = "a test crate has no public API: each test documents itself"
)]

use grafeo_common::storage::CatalogRecord;
use grafeo_common::storage::catalog_record::{GraphBindingRecord, GraphTypeRecord, SchemaRecord};
use grafeo_engine::GrafeoDB;

/// Appends `value` as bincode's standard configuration writes a length or
/// an integer: one byte below 251, else a marker and the little-endian
/// bytes.
fn varint(out: &mut Vec<u8>, value: u64) {
    match value {
        0..=250 => out.push(u8::try_from(value).unwrap()),
        251..=0xFFFF => {
            out.push(251);
            out.extend_from_slice(&u16::try_from(value).unwrap().to_le_bytes());
        }
        0x1_0000..=0xFFFF_FFFF => {
            out.push(252);
            out.extend_from_slice(&u32::try_from(value).unwrap().to_le_bytes());
        }
        _ => {
            out.push(253);
            out.extend_from_slice(&value.to_le_bytes());
        }
    }
}

fn string(out: &mut Vec<u8>, value: &str) {
    varint(out, value.len() as u64);
    out.extend_from_slice(value.as_bytes());
}

/// The rows of a snapshot with none: nodes, edges, named graphs, RDF
/// triples and RDF named graphs, each an empty list.
const NO_ROWS: [u8; 5] = [0; 5];

/// A snapshot of version 5 with no rows, whose catalog is `records`.
fn snapshot_v5(records: &[CatalogRecord]) -> Vec<u8> {
    let mut catalog = Vec::new();
    for record in records {
        record.encode_framed(&mut catalog).unwrap();
    }
    let mut bytes = vec![5];
    bytes.extend_from_slice(&NO_ROWS);
    varint(&mut bytes, catalog.len() as u64);
    bytes.extend_from_slice(&catalog);
    varint(&mut bytes, 0);
    bytes
}

/// A snapshot of version 4 with no rows, whose schema holds open graph
/// types of the names `graph_types` (a name given twice is defined twice),
/// the namespaces `schemas` and the graph type bindings `bindings`, and
/// nothing else.
fn snapshot_v4(graph_types: &[&str], schemas: &[&str], bindings: &[(&str, &str)]) -> Vec<u8> {
    let mut bytes = vec![4];
    bytes.extend_from_slice(&NO_ROWS);
    // Node types, edge types.
    bytes.extend_from_slice(&[0, 0]);
    varint(&mut bytes, graph_types.len() as u64);
    for name in graph_types {
        // The name, no node types, no edge types, open.
        string(&mut bytes, name);
        bytes.extend_from_slice(&[0, 0, 1]);
    }
    // Procedures.
    bytes.push(0);
    varint(&mut bytes, schemas.len() as u64);
    for schema in schemas {
        string(&mut bytes, schema);
    }
    varint(&mut bytes, bindings.len() as u64);
    for (graph, graph_type) in bindings {
        string(&mut bytes, graph);
        string(&mut bytes, graph_type);
    }
    // Property, vector and text indexes; the epoch.
    bytes.extend_from_slice(&[0, 0, 0, 0]);
    bytes
}

fn travel() -> CatalogRecord {
    CatalogRecord::GraphType(GraphTypeRecord {
        name: "travel".into(),
        node_types: Vec::new(),
        edge_types: Vec::new(),
        open: true,
    })
}

fn binding(graph_type: &str) -> CatalogRecord {
    CatalogRecord::GraphBinding(GraphBindingRecord {
        graph: "trips".into(),
        graph_type: graph_type.into(),
    })
}

fn rows(db: &GrafeoDB, query: &str) -> Vec<String> {
    db.execute(query)
        .unwrap_or_else(|error| panic!("{query}: {error}"))
        .rows()
        .iter()
        .map(|row| format!("{row:?}"))
        .collect()
}

/// Asserts `outcome` is an error naming the binding of `trips` to the graph
/// type `missing`.
fn assert_names_the_binding<T>(outcome: grafeo_common::utils::error::Result<T>) {
    let error = outcome
        .err()
        .expect("a binding to a graph type the catalog does not define fails")
        .to_string();
    assert!(
        error.contains("trips") && error.contains("missing"),
        "the error names the binding: {error}"
    );
}

/// Control: a version 5 snapshot written by hand with a graph type and a
/// binding to it imports, and lists both.
#[test]
fn a_v5_snapshot_written_by_hand_imports() {
    let schema = CatalogRecord::Schema(SchemaRecord {
        name: "archive".into(),
    });
    let db = GrafeoDB::import_snapshot(&snapshot_v5(&[schema, travel(), binding("travel")]))
        .expect("the hand-written snapshot imports");
    assert_eq!(rows(&db, "SHOW GRAPH TYPES").len(), 1);
    assert!(
        rows(&db, "SHOW SCHEMAS")
            .iter()
            .any(|row| row.contains("archive")),
        "{:?}",
        rows(&db, "SHOW SCHEMAS")
    );
}

/// A version 5 snapshot that binds a graph to a graph type its catalog does
/// not define fails the import, naming the record.
#[test]
fn a_v5_binding_to_a_missing_graph_type_fails_the_import() {
    assert_names_the_binding(GrafeoDB::import_snapshot(&snapshot_v5(&[binding(
        "missing",
    )])));
}

/// The same snapshot fails a restore.
#[test]
fn a_v5_binding_to_a_missing_graph_type_fails_the_restore() {
    let db = GrafeoDB::new_in_memory();
    db.execute("INSERT (:City {name: 'Amsterdam'})").unwrap();
    assert_names_the_binding(db.restore_snapshot(&snapshot_v5(&[binding("missing")])));
}

/// Control: a version 4 snapshot written by hand with a graph type, a
/// binding to it and a schema imports, and lists them.
#[test]
fn a_v4_snapshot_written_by_hand_imports() {
    let db = GrafeoDB::import_snapshot(&snapshot_v4(
        &["travel"],
        &["archive"],
        &[("trips", "travel")],
    ))
    .expect("the hand-written snapshot imports");
    assert_eq!(rows(&db, "SHOW GRAPH TYPES").len(), 1);
    assert!(
        rows(&db, "SHOW SCHEMAS")
            .iter()
            .any(|row| row.contains("archive")),
        "{:?}",
        rows(&db, "SHOW SCHEMAS")
    );
}

/// Control: `DROP GRAPH TYPE` leaves the bindings to the type, and 0.5.x
/// exported them in version 4 snapshots (the catalog's version 2 writer
/// leaves them out since): such a snapshot still imports.
#[test]
fn a_v4_binding_a_dropped_graph_type_left_still_imports() {
    let db = GrafeoDB::import_snapshot(&snapshot_v4(&[], &[], &[("trips", "dropped")]))
        .expect("a binding a dropped graph type left imports");
    assert_eq!(rows(&db, "SHOW GRAPH TYPES"), Vec::<String>::new());
}

/// Asserts `outcome` is an error naming the graph type `travel`.
fn assert_names_travel<T>(outcome: grafeo_common::utils::error::Result<T>) {
    let error = outcome
        .err()
        .expect("a graph type defined twice fails")
        .to_string();
    assert!(error.contains("travel"), "the error names it: {error}");
}

/// A version 4 snapshot (whose schema `restore_schema_from_snapshot` loads,
/// as it loads a 0.5.x file's) that defines a graph type twice fails the
/// import, naming it: the second definition is left out today.
#[test]
fn a_v4_graph_type_defined_twice_fails_the_import() {
    assert_names_travel(GrafeoDB::import_snapshot(&snapshot_v4(
        &["travel", "travel"],
        &[],
        &[],
    )));
}

/// The same snapshot fails a restore.
#[test]
fn a_v4_graph_type_defined_twice_fails_the_restore() {
    let db = GrafeoDB::new_in_memory();
    db.execute("INSERT (:City {name: 'Amsterdam'})").unwrap();
    assert_names_travel(db.restore_snapshot(&snapshot_v4(&["travel", "travel"], &[], &[])));
}

/// The database a failed restore must leave as it was: data in the root and
/// in a typed named graph, node, edge and graph types, a unique constraint,
/// an index, a schema and a procedure (and an RDF triple in a build with
/// SPARQL).
fn filled(db: GrafeoDB) -> GrafeoDB {
    for statement in [
        "CREATE NODE TYPE City (name STRING NOT NULL)",
        "CREATE EDGE TYPE ROUTE CONNECTING (City) TO (City) (km INT64)",
        "CREATE GRAPH TYPE travel (NODE TYPE City, EDGE TYPE ROUTE)",
        "CREATE GRAPH trips TYPED travel",
        "CREATE SCHEMA archive",
        "CREATE CONSTRAINT person_email FOR (p:Person) ON (p.email) UNIQUE",
        "CREATE INDEX person_name FOR (p:Person) ON (p.name)",
        "CREATE PROCEDURE people() RETURNS (name STRING) AS \
         { MATCH (p:Person) RETURN p.name AS name }",
        "INSERT (:Person {name: 'Alix', email: 'alix@example.org'}) \
         -[:KNOWS {since: 2015}]-> (:Person {name: 'Gus', email: 'gus@example.org'})",
    ] {
        db.execute(statement)
            .unwrap_or_else(|error| panic!("{statement}: {error}"));
    }
    let session = db.session();
    session.use_graph("trips");
    session
        .execute(
            "INSERT (:City {name: 'Amsterdam'}) -[:ROUTE {km: 650}]-> (:City {name: 'Berlin'})",
        )
        .unwrap();
    #[cfg(all(feature = "sparql", feature = "triple-store"))]
    db.execute_sparql(
        "INSERT DATA { <http://example.org/alix> <http://example.org/knows> \
         <http://example.org/gus> }",
    )
    .unwrap();
    db
}

/// Every row of `queries` on `db` (in the graph `graph` when given), one
/// line each, the rows of a query sorted; a query that fails gives one line
/// with its error, so two states compare in full.
fn lines(db: &GrafeoDB, graph: Option<&str>, queries: &[&str]) -> Vec<String> {
    let session = db.session();
    if let Some(graph) = graph {
        session.use_graph(graph);
    }
    queries
        .iter()
        .flat_map(|query| match session.execute(query) {
            Ok(result) => {
                let mut lines: Vec<String> = result
                    .rows()
                    .iter()
                    .map(|row| format!("{query}: {row:?}"))
                    .collect();
                lines.sort();
                lines
            }
            Err(error) => vec![format!("{query}: fails: {error}")],
        })
        .collect()
}

/// What `SHOW` lists, the nodes and edges of the root and of `trips`, the
/// procedure's rows and the RDF triples: a readable picture of the state.
fn state(db: &GrafeoDB) -> Vec<String> {
    const DATA: &[&str] = &[
        "MATCH (n) RETURN labels(n) AS labels, properties(n) AS properties",
        "MATCH (a)-[r]->(b) RETURN a.name, type(r), properties(r), b.name",
    ];
    let mut state = lines(
        db,
        None,
        &[
            "SHOW NODE TYPES",
            "SHOW EDGE TYPES",
            "SHOW GRAPH TYPES",
            "SHOW CONSTRAINTS",
            "SHOW INDEXES",
            "SHOW GRAPHS",
            "SHOW SCHEMAS",
            "CALL people() YIELD name RETURN name",
        ],
    );
    state.extend(lines(db, None, DATA));
    state.extend(lines(db, Some("trips"), DATA));
    #[cfg(all(feature = "sparql", feature = "triple-store"))]
    state.extend(
        db.execute_sparql("SELECT ?s ?p ?o WHERE { ?s ?p ?o }")
            .unwrap()
            .rows()
            .iter()
            .map(|row| format!("rdf: {row:?}")),
    );
    state
}

/// Snapshots whose restore fails in its catalog, after the rows and some of
/// the records would apply: each defines a schema and a graph type the
/// database does not hold, then a record that fails.
fn failing_snapshots() -> [(&'static str, Vec<u8>); 2] {
    let elsewhere = CatalogRecord::Schema(SchemaRecord {
        name: "elsewhere".into(),
    });
    let voyage = CatalogRecord::GraphType(GraphTypeRecord {
        name: "voyage".into(),
        node_types: Vec::new(),
        edge_types: Vec::new(),
        open: true,
    });
    [
        (
            "version 5, a binding to a missing graph type",
            snapshot_v5(&[elsewhere, voyage, binding("missing")]),
        ),
        (
            "version 4, a graph type defined twice",
            snapshot_v4(&["voyage", "voyage"], &["elsewhere"], &[]),
        ),
    ]
}

/// A restore that fails leaves the database as it was: its data, every
/// graph, type, constraint, index, schema and procedure, nothing of the
/// snapshot, and the same bytes on export. Before, the restore emptied the
/// stores and the catalog first, so a record failing after that left the
/// database empty or half restored.
#[test]
fn a_failed_restore_leaves_the_database_as_it_was() {
    let mut failures = Vec::new();
    for (name, snapshot) in failing_snapshots() {
        let db = filled(GrafeoDB::new_in_memory());
        let before = state(&db);
        let bytes = db.export_snapshot().unwrap();
        let Err(error) = db.restore_snapshot(&snapshot) else {
            failures.push(format!("{name}: the restore succeeded"));
            continue;
        };
        let after = state(&db);
        if after != before {
            failures.push(format!(
                "{name} ({error}): the state changed\n  before: {before:#?}\n  after: {after:#?}"
            ));
            continue;
        }
        if db.export_snapshot().unwrap() != bytes {
            failures.push(format!("{name}: the export differs"));
        }
        // The definitions still work: the unique constraint refuses a
        // duplicate, and the database takes writes.
        if db
            .execute("INSERT (:Person {name: 'Vincent', email: 'alix@example.org'})")
            .is_ok()
        {
            failures.push(format!("{name}: the unique constraint is gone"));
        }
        db.execute("INSERT (:Person {name: 'Jules', email: 'jules@example.org'})")
            .unwrap_or_else(|error| panic!("{name}: a write after the failed restore: {error}"));
    }
    assert!(failures.is_empty(), "\n{}", failures.join("\n\n"));
}

/// Control: the same database restores its own export, and keeps its state.
#[test]
fn a_restore_of_its_own_export_keeps_the_state() {
    let db = filled(GrafeoDB::new_in_memory());
    let before = state(&db);
    db.restore_snapshot(&db.export_snapshot().unwrap()).unwrap();
    assert_eq!(state(&db), before);
}

/// A persistent database after a failed restore: a commit, a close and a
/// reopen give the state before the restore plus the commit. The close
/// checkpoints what the database holds, so a half-cleared database would
/// write itself to the file.
#[cfg(all(feature = "wal", feature = "grafeo-file"))]
#[test]
fn a_failed_restore_of_a_persistent_database_survives_a_reopen() {
    const MIA: &str = "INSERT (:Person {name: 'Mia', email: 'mia@example.org'})";
    // The state to reopen: the filled database plus Mia, without a restore.
    let control = filled(GrafeoDB::new_in_memory());
    control.execute(MIA).unwrap();
    let expected = state(&control);

    let mut failures = Vec::new();
    for (name, snapshot) in failing_snapshots() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("prague.grafeo");
        let db = filled(GrafeoDB::open(&path).unwrap());
        if db.restore_snapshot(&snapshot).is_ok() {
            failures.push(format!("{name}: the restore succeeded"));
            continue;
        }
        db.execute(MIA)
            .unwrap_or_else(|error| panic!("{name}: a write after the failed restore: {error}"));
        db.close().unwrap();
        drop(db);

        let reopened = GrafeoDB::open(&path).unwrap();
        let state = state(&reopened);
        if state != expected {
            failures.push(format!(
                "{name}: the reopen differs\n  expected: {expected:#?}\n  reopened: {state:#?}"
            ));
        }
        reopened.close().unwrap();
    }
    assert!(failures.is_empty(), "\n{}", failures.join("\n\n"));
}

/// A node type and an edge type whose records carry a CHECK constraint, the
/// record kind the format keeps though no version supports CHECK any more.
fn types_with_a_check() -> [(&'static str, CatalogRecord); 2] {
    use grafeo_common::storage::catalog_record::{
        EdgeTypeRecord, NodeTypeRecord, TypeConstraintRecord,
    };

    [
        (
            "City",
            CatalogRecord::NodeType(NodeTypeRecord {
                name: "City".into(),
                properties: Vec::new(),
                constraints: vec![TypeConstraintRecord::Check {
                    name: Some("populated".into()),
                    expression: "population > 3".into(),
                }],
                parent_types: Vec::new(),
                key_labels: Vec::new(),
            }),
        ),
        (
            "ROAD",
            CatalogRecord::EdgeType(EdgeTypeRecord {
                name: "ROAD".into(),
                properties: Vec::new(),
                constraints: vec![TypeConstraintRecord::Check {
                    name: None,
                    expression: "km > 0".into(),
                }],
                endpoints: Vec::new(),
                key_labels: Vec::new(),
            }),
        ),
    ]
}

/// A snapshot whose node or edge type has a CHECK constraint fails the
/// import and the restore with a serialization error naming the type, the
/// expression and the lack of support; the restore leaves the database as
/// it was.
#[test]
fn a_check_constraint_record_fails_the_import_and_the_restore() {
    let mut failures = Vec::new();
    for (type_name, record) in types_with_a_check() {
        let snapshot = snapshot_v5(&[record]);
        let check = |what: &str, outcome: grafeo_common::utils::error::Result<()>| match outcome {
            Ok(()) => Some(format!("{type_name}, {what}: accepted")),
            Err(error) => {
                let message = error.to_string();
                let named = message.contains(type_name)
                    && message.contains("CHECK")
                    && message.contains("does not support");
                (!named).then(|| format!("{type_name}, {what}: {message}"))
            }
        };
        failures.extend(check(
            "import",
            GrafeoDB::import_snapshot(&snapshot).map(|_| ()),
        ));
        let db = GrafeoDB::new_in_memory();
        db.execute("INSERT (:City {name: 'Amsterdam'})").unwrap();
        failures.extend(check("restore", db.restore_snapshot(&snapshot)));
        if rows(&db, "MATCH (c:City) RETURN c.name") != rows(&db, "RETURN 'Amsterdam' AS n") {
            failures.push(format!("{type_name}: the refused restore changed the data"));
        }
    }
    assert!(failures.is_empty(), "\n{}", failures.join("\n"));
}
