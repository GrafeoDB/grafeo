//! Integration tests for snapshot export/import.

use grafeo_engine::GrafeoDB;

#[test]
fn export_import_empty_database() {
    let db = GrafeoDB::new_in_memory();
    let bytes = db.export_snapshot().unwrap();
    let restored = GrafeoDB::import_snapshot(&bytes).unwrap();
    assert_eq!(restored.node_count(), 0);
    assert_eq!(restored.edge_count(), 0);
}

#[test]
fn export_import_preserves_nodes() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    session
        .execute("INSERT (:Person {name: 'Alice', age: 30})")
        .unwrap();
    session
        .execute("INSERT (:Person {name: 'Bob', age: 25})")
        .unwrap();

    let bytes = db.export_snapshot().unwrap();
    let restored = GrafeoDB::import_snapshot(&bytes).unwrap();

    assert_eq!(restored.node_count(), 2);

    let session2 = restored.session();
    let result = session2
        .execute("MATCH (p:Person) RETURN p.name ORDER BY p.name")
        .unwrap();
    assert_eq!(result.rows.len(), 2);
}

#[test]
fn export_import_preserves_edges() {
    let db = GrafeoDB::new_in_memory();
    let alice = db.create_node(&["Person"]);
    db.set_node_property(alice, "name", "Alice".into());
    let bob = db.create_node(&["Person"]);
    db.set_node_property(bob, "name", "Bob".into());
    db.create_edge(alice, bob, "KNOWS");

    let bytes = db.export_snapshot().unwrap();
    let restored = GrafeoDB::import_snapshot(&bytes).unwrap();

    assert_eq!(restored.node_count(), 2);
    assert_eq!(restored.edge_count(), 1);

    let session2 = restored.session();
    let result = session2
        .execute("MATCH (a)-[:KNOWS]->(b) RETURN a.name, b.name")
        .unwrap();
    assert_eq!(result.rows.len(), 1);
}

#[test]
fn export_import_preserves_properties() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    session
        .execute("INSERT (:Item {name: 'Widget', price: 9.99, active: true})")
        .unwrap();

    let bytes = db.export_snapshot().unwrap();
    let restored = GrafeoDB::import_snapshot(&bytes).unwrap();

    let session2 = restored.session();
    let result = session2
        .execute("MATCH (i:Item) RETURN i.name, i.price, i.active")
        .unwrap();
    assert_eq!(result.rows.len(), 1);
}

#[test]
fn import_rejects_invalid_data() {
    let result = GrafeoDB::import_snapshot(b"not a valid snapshot");
    assert!(result.is_err());
}

#[test]
fn snapshot_round_trip_schema() {
    let db = GrafeoDB::new_in_memory();
    let alice = db.create_node(&["Person"]);
    db.set_node_property(alice, "name", "Alice".into());
    let bob = db.create_node(&["Person"]);
    db.set_node_property(bob, "name", "Bob".into());
    db.create_edge(alice, bob, "KNOWS");

    let schema_before = db.schema();
    let bytes = db.export_snapshot().unwrap();
    let restored = GrafeoDB::import_snapshot(&bytes).unwrap();
    let schema_after = restored.schema();

    // Both schemas should report the same label/edge info
    let fmt_before = format!("{schema_before:?}");
    let fmt_after = format!("{schema_after:?}");
    assert_eq!(fmt_before, fmt_after);
}
