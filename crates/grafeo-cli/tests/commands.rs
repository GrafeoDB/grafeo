//! Integration tests for CLI commands.

use std::path::Path;
use tempfile::TempDir;

/// Helper to create a test database.
fn create_test_db(dir: &Path) -> grafeo_engine::GrafeoDB {
    let db = grafeo_engine::GrafeoDB::open(dir).expect("Failed to create test database");

    // Add some test data
    let n1 = db.create_node(&["Person"]).unwrap();
    let n2 = db.create_node(&["Person"]).unwrap();
    let n3 = db.create_node(&["Company"]).unwrap();

    db.set_node_property(n1, "name", "Alix".into()).unwrap();
    db.set_node_property(n2, "name", "Gus".into()).unwrap();
    db.set_node_property(n3, "name", "Acme Corp".into())
        .unwrap();

    db.create_edge(n1, n2, "KNOWS").unwrap();
    db.create_edge(n1, n3, "WORKS_AT").unwrap();

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

    let n1 = db.create_node(&["Test"]).unwrap();
    let n2 = db.create_node(&["Test"]).unwrap();
    let e1 = db.create_edge(n1, n2, "LINKS").unwrap();

    let info = db.info();
    assert_eq!(info.node_count, 2);
    assert_eq!(info.edge_count, 1);

    // Verify edge exists
    let edge = db.get_edge(e1);
    assert!(edge.is_some());
}

#[test]
fn test_validate_passes_on_good_database() {
    let temp_dir = TempDir::new().expect("create temp dir");
    let db_path = temp_dir.path().join("test.grafeo");

    let db = create_test_db(&db_path);
    let result = db.validate();

    assert!(result.errors.is_empty(), "expected no validation errors");
}

#[test]
fn test_schema_inspection() {
    let temp_dir = TempDir::new().expect("create temp dir");
    let db_path = temp_dir.path().join("test.grafeo");

    let db = create_test_db(&db_path);
    let schema = db.schema();

    match schema {
        grafeo_engine::SchemaInfo::Lpg(lpg) => {
            let label_names: Vec<&str> = lpg.labels.iter().map(|l| l.name.as_str()).collect();
            assert!(label_names.contains(&"Person"));
            assert!(label_names.contains(&"Company"));

            let edge_names: Vec<&str> = lpg.edge_types.iter().map(|e| e.name.as_str()).collect();
            assert!(edge_names.contains(&"KNOWS"));
            assert!(edge_names.contains(&"WORKS_AT"));

            assert!(lpg.property_keys.contains(&"name".to_string()));
        }
        other => panic!("expected LPG schema, got {other:?}"),
    }
}

#[test]
fn test_admin_service_trait() {
    use grafeo_engine::AdminService;

    let db = grafeo_engine::GrafeoDB::new_in_memory();
    let n1 = db.create_node(&["Test"]).unwrap();
    db.set_node_property(n1, "key", "value".into()).unwrap();

    // AdminService methods should work
    let info = AdminService::info(&db);
    assert_eq!(info.node_count, 1);

    let stats = AdminService::detailed_stats(&db);
    assert_eq!(stats.node_count, 1);

    let schema = AdminService::schema(&db);
    assert!(matches!(schema, grafeo_engine::SchemaInfo::Lpg(_)));

    let validation = AdminService::validate(&db);
    assert!(validation.errors.is_empty());
}

#[test]
fn test_query_with_parameters() {
    let temp_dir = TempDir::new().expect("create temp dir");
    let db_path = temp_dir.path().join("test.grafeo");

    let db = create_test_db(&db_path);

    // Test parameterized query
    let mut params = std::collections::HashMap::new();
    params.insert(
        "name".to_string(),
        grafeo_common::types::Value::from("Alix"),
    );

    let session = db.session();
    let result = session
        .execute_with_params("MATCH (n {name: $name}) RETURN n.name", params)
        .expect("execute parameterized query");
    assert_eq!(result.row_count(), 1);
}

#[test]
fn test_init_creates_database() {
    let temp_dir = TempDir::new().expect("create temp dir");
    let db_path = temp_dir.path().join("new.grafeo");

    let db = grafeo_engine::GrafeoDB::open(&db_path).expect("create db");
    let info = db.info();
    assert_eq!(info.node_count, 0);
    assert_eq!(info.edge_count, 0);
    assert!(info.is_persistent);
}
