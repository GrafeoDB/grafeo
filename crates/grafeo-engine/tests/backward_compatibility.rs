//! Snapshot format stability tests.
//!
//! These tests verify that snapshots produced by the current code can be
//! round-tripped (export then import) without data loss. If the format
//! changes, the fixture must be regenerated.

use grafeo_common::types::Value;
use grafeo_engine::GrafeoDB;

/// Generate a pinned v3 snapshot at runtime (the fixture is not committed
/// because format changes are tracked via this test, not via binary files).
fn generate_fixture() -> Vec<u8> {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();

    // Nodes
    session
        .execute("INSERT (:Person {name: 'Alix', age: 30})")
        .unwrap();
    session
        .execute("INSERT (:Person:Employee {name: 'Gus', age: 25})")
        .unwrap();
    session
        .execute("INSERT (:Company {name: 'Acme Corp'})")
        .unwrap();

    // Edges
    session
        .execute(
            "MATCH (a:Person {name: 'Alix'}), (b:Person {name: 'Gus'}) \
             INSERT (a)-[:KNOWS {since: 2020}]->(b)",
        )
        .unwrap();
    session
        .execute(
            "MATCH (g:Employee {name: 'Gus'}), (c:Company {name: 'Acme Corp'}) \
             INSERT (g)-[:WORKS_AT {role: 'Engineer'}]->(c)",
        )
        .unwrap();

    // Schema (DDL)
    session
        .execute("CREATE NODE TYPE Person (name STRING NOT NULL, age INT64)")
        .unwrap();

    db.export_snapshot().unwrap()
}

#[test]
fn snapshot_round_trip_preserves_node_count() {
    let snapshot = generate_fixture();
    let db = GrafeoDB::import_snapshot(&snapshot).unwrap();
    assert_eq!(db.node_count(), 3);
}

#[test]
fn snapshot_round_trip_preserves_edge_count() {
    let snapshot = generate_fixture();
    let db = GrafeoDB::import_snapshot(&snapshot).unwrap();
    assert_eq!(db.edge_count(), 2);
}

#[test]
fn snapshot_round_trip_preserves_labels() {
    let snapshot = generate_fixture();
    let db = GrafeoDB::import_snapshot(&snapshot).unwrap();
    let session = db.session();

    let persons = session
        .execute("MATCH (p:Person) RETURN p.name ORDER BY p.name")
        .unwrap();
    assert_eq!(persons.rows().len(), 2);
    assert_eq!(persons.rows()[0][0], Value::String("Alix".into()));
    assert_eq!(persons.rows()[1][0], Value::String("Gus".into()));

    let companies = session.execute("MATCH (c:Company) RETURN c.name").unwrap();
    assert_eq!(companies.rows().len(), 1);
    assert_eq!(companies.rows()[0][0], Value::String("Acme Corp".into()));
}

#[test]
fn snapshot_round_trip_preserves_node_properties() {
    let snapshot = generate_fixture();
    let db = GrafeoDB::import_snapshot(&snapshot).unwrap();
    let session = db.session();

    let result = session
        .execute("MATCH (p:Person) WHERE p.name = 'Alix' RETURN p.age")
        .unwrap();
    assert_eq!(result.rows().len(), 1);
    assert_eq!(result.rows()[0][0], Value::Int64(30));
}

#[test]
fn snapshot_round_trip_preserves_edge_properties() {
    let snapshot = generate_fixture();
    let db = GrafeoDB::import_snapshot(&snapshot).unwrap();
    let session = db.session();

    let knows = session
        .execute("MATCH ()-[e:KNOWS]->() RETURN e.since")
        .unwrap();
    assert_eq!(knows.rows().len(), 1);
    assert_eq!(knows.rows()[0][0], Value::Int64(2020));

    let works = session
        .execute("MATCH ()-[e:WORKS_AT]->() RETURN e.role")
        .unwrap();
    assert_eq!(works.rows().len(), 1);
    assert_eq!(works.rows()[0][0], Value::String("Engineer".into()));
}

#[test]
fn snapshot_round_trip_preserves_multi_labels() {
    let snapshot = generate_fixture();
    let db = GrafeoDB::import_snapshot(&snapshot).unwrap();
    let session = db.session();

    let employees = session.execute("MATCH (e:Employee) RETURN e.name").unwrap();
    assert_eq!(employees.rows().len(), 1);
    assert_eq!(employees.rows()[0][0], Value::String("Gus".into()));
}

#[test]
fn snapshot_round_trip_preserves_schema() {
    let snapshot = generate_fixture();
    let db = GrafeoDB::import_snapshot(&snapshot).unwrap();
    let session = db.session();

    let result = session.execute("SHOW NODE TYPES").unwrap();
    let mut type_names: Vec<String> = result
        .rows()
        .iter()
        .filter_map(|r| match &r[0] {
            Value::String(s) => Some(s.to_string()),
            _ => None,
        })
        .collect();
    type_names.sort();
    assert!(
        type_names.contains(&"Person".to_string()),
        "Person type missing after import: {type_names:?}"
    );
}

#[test]
fn snapshot_double_round_trip() {
    // Export -> import -> re-export -> re-import: verify data integrity.
    let snapshot = generate_fixture();
    let db = GrafeoDB::import_snapshot(&snapshot).unwrap();
    let re_exported = db.export_snapshot().unwrap();

    let db2 = GrafeoDB::import_snapshot(&re_exported).unwrap();
    assert_eq!(db2.node_count(), 3);
    assert_eq!(db2.edge_count(), 2);

    let session = db2.session();
    let result = session
        .execute("MATCH (p:Person) WHERE p.name = 'Alix' RETURN p.age")
        .unwrap();
    assert_eq!(result.rows()[0][0], Value::Int64(30));
}
