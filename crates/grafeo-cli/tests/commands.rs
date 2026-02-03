//! Integration tests for CLI commands.

use std::path::Path;
use tempfile::TempDir;

/// Helper to create a test database.
fn create_test_db(dir: &Path) -> grafeo_engine::GrafeoDB {
    let db = grafeo_engine::GrafeoDB::open(dir).expect("Failed to create test database");

    // Add some test data
    let n1 = db.create_node(&["Person"]);
    let n2 = db.create_node(&["Person"]);
    let n3 = db.create_node(&["Company"]);

    db.set_node_property(n1, "name", "Alice".into());
    db.set_node_property(n2, "name", "Bob".into());
    db.set_node_property(n3, "name", "Acme Corp".into());

    db.create_edge(n1, n2, "KNOWS");
    db.create_edge(n1, n3, "WORKS_AT");

    db
}

#[test]
fn test_database_can_be_opened() {
    let temp_dir = TempDir::new().expect("create temp dir");
    let db_path = temp_dir.path().join("test.grafeo");

    let db = grafeo_engine::GrafeoDB::open(&db_path).expect("create db");
    drop(db);

    // Reopen to verify persistence
    let db2 = grafeo_engine::GrafeoDB::open(&db_path).expect("reopen db");
    let info = db2.info();
    assert!(info.is_persistent);
}

#[test]
fn test_database_info() {
    let temp_dir = TempDir::new().expect("create temp dir");
    let db_path = temp_dir.path().join("test.grafeo");

    let db = create_test_db(&db_path);
    let info = db.info();

    assert_eq!(info.node_count, 3);
    assert_eq!(info.edge_count, 2);
    assert!(info.is_persistent);
}

#[test]
fn test_database_stats() {
    let temp_dir = TempDir::new().expect("create temp dir");
    let db_path = temp_dir.path().join("test.grafeo");

    let db = create_test_db(&db_path);
    let stats = db.detailed_stats();

    assert_eq!(stats.node_count, 3);
    assert_eq!(stats.edge_count, 2);
    assert_eq!(stats.label_count, 2); // Person, Company
    assert_eq!(stats.edge_type_count, 2); // KNOWS, WORKS_AT
    assert!(stats.property_key_count >= 1); // name
    // Note: memory_bytes may be 0 depending on implementation
}

#[test]
fn test_query_execution() {
    let temp_dir = TempDir::new().expect("create temp dir");
    let db_path = temp_dir.path().join("test.grafeo");

    let db = create_test_db(&db_path);

    // Test a simple query
    let result = db
        .execute("MATCH (n:Person) RETURN n.name")
        .expect("execute query");
    assert_eq!(result.row_count(), 2);
}

#[test]
fn test_in_memory_database() {
    let db = grafeo_engine::GrafeoDB::new_in_memory();
    let info = db.info();

    assert!(!info.is_persistent);
    assert_eq!(info.node_count, 0);
    assert_eq!(info.edge_count, 0);
}

#[test]
fn test_node_and_edge_creation() {
    let db = grafeo_engine::GrafeoDB::new_in_memory();

    let n1 = db.create_node(&["Test"]);
    let n2 = db.create_node(&["Test"]);
    let e1 = db.create_edge(n1, n2, "LINKS");

    let info = db.info();
    assert_eq!(info.node_count, 2);
    assert_eq!(info.edge_count, 1);

    // Verify edge exists
    let edge = db.get_edge(e1);
    assert!(edge.is_some());
}
