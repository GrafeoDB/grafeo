//! Tests for schema DDL coverage: CREATE/ALTER/DROP types, procedures, namespaces.
//!
//! Targets: catalog/mod.rs (36.75%), session.rs DDL paths, parser.rs schema syntax
//!
//! ```bash
//! cargo test -p grafeo-engine --test coverage_schema_ddl
//! ```

use grafeo_common::types::Value;
use grafeo_engine::GrafeoDB;

// ---------------------------------------------------------------------------
// CREATE / DROP NODE TYPE
// ---------------------------------------------------------------------------

#[test]
fn test_create_node_type_and_drop() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    session
        .execute("CREATE NODE TYPE Vehicle (make STRING NOT NULL, year INTEGER)")
        .unwrap();
    session
        .execute("INSERT (:Vehicle {make: 'Volvo', year: 2024})")
        .unwrap();
    session.execute("DROP NODE TYPE Vehicle").unwrap();
}

#[test]
fn test_create_node_type_duplicate_fails() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    session
        .execute("CREATE NODE TYPE Gadget (name STRING)")
        .unwrap();
    let dup = session.execute("CREATE NODE TYPE Gadget (name STRING)");
    assert!(dup.is_err(), "duplicate type should fail");
}

#[test]
fn test_create_or_replace_node_type() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    session
        .execute("CREATE NODE TYPE Widget (name STRING)")
        .unwrap();
    session
        .execute("CREATE OR REPLACE NODE TYPE Widget (name STRING, color STRING)")
        .unwrap();
}

#[test]
fn test_drop_nonexistent_node_type_fails() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    let result = session.execute("DROP NODE TYPE Nonexistent");
    let err = result.unwrap_err().to_string();
    assert!(
        err.contains("Nonexistent") || err.contains("not found") || err.contains("does not exist"),
        "error should name the missing type, got: {err}"
    );
}

// ---------------------------------------------------------------------------
// CREATE / DROP EDGE TYPE
// ---------------------------------------------------------------------------

#[test]
fn test_create_edge_type_and_drop() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    session
        .execute("CREATE EDGE TYPE SUPPLIES (quantity INTEGER)")
        .unwrap();
    session.execute("DROP EDGE TYPE SUPPLIES").unwrap();
}

#[test]
fn test_create_or_replace_edge_type() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    session
        .execute("CREATE EDGE TYPE RATES (stars INTEGER)")
        .unwrap();
    session
        .execute("CREATE OR REPLACE EDGE TYPE RATES (stars INTEGER, comment STRING)")
        .unwrap();
}

// ---------------------------------------------------------------------------
// ALTER NODE TYPE: ADD / DROP PROPERTY
// ---------------------------------------------------------------------------

#[test]
fn test_alter_node_type_add_and_drop_property() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    session
        .execute("CREATE NODE TYPE Sensor (id INTEGER NOT NULL)")
        .unwrap();
    session
        .execute("ALTER NODE TYPE Sensor ADD PROPERTY location STRING")
        .unwrap();
    session
        .execute("ALTER NODE TYPE Sensor DROP PROPERTY location")
        .unwrap();
}

// ---------------------------------------------------------------------------
// ALTER EDGE TYPE: ADD / DROP PROPERTY
// ---------------------------------------------------------------------------

#[test]
fn test_alter_edge_type_add_and_drop_property() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    session
        .execute("CREATE EDGE TYPE MONITORS (interval INTEGER)")
        .unwrap();
    session
        .execute("ALTER EDGE TYPE MONITORS ADD PROPERTY threshold FLOAT")
        .unwrap();
    session
        .execute("ALTER EDGE TYPE MONITORS DROP PROPERTY threshold")
        .unwrap();
}

// ---------------------------------------------------------------------------
// ALTER GRAPH TYPE: ADD / DROP NODE/EDGE TYPE
// ---------------------------------------------------------------------------

#[test]
fn test_alter_graph_type_add_drop_members() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    session
        .execute("CREATE NODE TYPE Device (serial STRING)")
        .unwrap();
    session.execute("CREATE EDGE TYPE CONNECTS").unwrap();
    session
        .execute("CREATE GRAPH TYPE iot_network (NODE TYPE Device)")
        .unwrap();
    session
        .execute("ALTER GRAPH TYPE iot_network ADD EDGE TYPE CONNECTS")
        .unwrap();
    session
        .execute("ALTER GRAPH TYPE iot_network DROP EDGE TYPE CONNECTS")
        .unwrap();
}

// ---------------------------------------------------------------------------
// DROP GRAPH TYPE
// ---------------------------------------------------------------------------

#[test]
fn test_drop_graph_type() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    session
        .execute("CREATE GRAPH TYPE temp_type (NODE TYPE Temp (v INTEGER))")
        .unwrap();
    session.execute("DROP GRAPH TYPE temp_type").unwrap();
    session
        .execute("CREATE GRAPH TYPE temp_type (NODE TYPE Temp2 (v INTEGER))")
        .unwrap();
}

// ---------------------------------------------------------------------------
// Schema namespaces (CREATE/DROP SCHEMA)
// ---------------------------------------------------------------------------

#[test]
fn test_create_and_drop_schema_namespace() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    session.execute("CREATE SCHEMA analytics").unwrap();
    session.execute("DROP SCHEMA analytics").unwrap();
}

#[test]
fn test_create_duplicate_schema_fails() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    session.execute("CREATE SCHEMA reporting").unwrap();
    let dup = session.execute("CREATE SCHEMA reporting");
    let err = dup.unwrap_err().to_string();
    assert!(
        err.contains("reporting") || err.contains("exists") || err.contains("duplicate"),
        "error should mention duplicate schema, got: {err}"
    );
}

// ---------------------------------------------------------------------------
// Stored procedures (CREATE/DROP PROCEDURE)
// ---------------------------------------------------------------------------

#[test]
fn test_create_and_drop_procedure() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    session
        .execute(
            "CREATE PROCEDURE get_adults() RETURNS (name STRING) AS { \
             MATCH (p:Person) WHERE p.age >= 18 RETURN p.name AS name \
             }",
        )
        .unwrap();
    session.execute("DROP PROCEDURE get_adults").unwrap();
}

/// A pattern in a procedure body that comes back to a node it bound is a
/// cycle there too: the body skipped that check (it is planned without the
/// session's optimizer), so every path of three edges counted.
#[test]
fn test_procedure_body_closes_cycles() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    session
        .execute(
            "INSERT (alix:Person {name: 'Alix'}), (gus:Person {name: 'Gus'}), \
             (vincent:Person {name: 'Vincent'}), (jules:Person {name: 'Jules'}), \
             (mia:Person {name: 'Mia'}), (alix)-[:KNOWS]->(gus), (gus)-[:KNOWS]->(vincent), \
             (vincent)-[:KNOWS]->(alix), (jules)-[:KNOWS]->(mia), (alix)-[:KNOWS]->(jules)",
        )
        .unwrap();
    session
        .execute(
            "CREATE PROCEDURE triangles() RETURNS (name STRING) AS { \
             MATCH (a:Person)-[:KNOWS]->(b)-[:KNOWS]->(c)-[:KNOWS]->(a) RETURN a.name AS name }",
        )
        .unwrap();
    let result = session
        .execute("CALL triangles() YIELD name RETURN name ORDER BY name")
        .unwrap();
    let names: Vec<Value> = result.rows().iter().map(|row| row[0].clone()).collect();
    assert_eq!(
        names,
        [
            Value::from("Alix"),
            Value::from("Gus"),
            Value::from("Vincent")
        ]
    );
}

#[test]
fn test_create_or_replace_procedure() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    session
        .execute(
            "CREATE PROCEDURE greet() RETURNS (msg STRING) AS { MATCH (n) RETURN count(n) AS cnt }",
        )
        .unwrap();
    session
        .execute("CREATE OR REPLACE PROCEDURE greet() RETURNS (msg STRING) AS { MATCH (n) RETURN count(n) AS cnt }")
        .unwrap();
}

// ---------------------------------------------------------------------------
// Type inheritance (EXTENDS)
// ---------------------------------------------------------------------------

#[test]
fn test_node_type_with_not_null_property() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    session
        .execute("CREATE NODE TYPE Account (owner STRING NOT NULL, balance INTEGER)")
        .unwrap();
    // Insert with required property
    session
        .execute("INSERT (:Account {owner: 'Alix', balance: 1000})")
        .unwrap();
    // Verify the type is tracked
    session.execute("DROP NODE TYPE Account").unwrap();
}

#[test]
fn test_not_null_constraint_rejects_missing_property() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    session
        .execute("CREATE NODE TYPE Account (owner STRING NOT NULL, balance INTEGER)")
        .unwrap();
    // Insert without the NOT NULL property should fail
    let result = session.execute("INSERT (:Account {balance: 500})");
    let err = result.unwrap_err();
    let msg = err.to_string();
    assert!(
        msg.contains("constraint") || msg.contains("NOT NULL") || msg.contains("owner"),
        "Expected constraint violation error, got: {msg}"
    );
}

#[test]
fn test_not_null_constraint_allows_all_properties_present() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    session
        .execute("CREATE NODE TYPE Invoice (number INTEGER NOT NULL, total FLOAT NOT NULL)")
        .unwrap();
    // Both NOT NULL properties present: should succeed
    session
        .execute("INSERT (:Invoice {number: 42, total: 99.95})")
        .unwrap();
    let result = session
        .execute("MATCH (i:Invoice) RETURN i.number, i.total")
        .unwrap();
    assert_eq!(result.rows().len(), 1);
}

// ---------------------------------------------------------------------------
// Full lifecycle: type -> graph type -> bind -> insert
// ---------------------------------------------------------------------------

#[test]
fn test_full_schema_lifecycle() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();

    // Create types
    session
        .execute("CREATE NODE TYPE Employee (name STRING NOT NULL)")
        .unwrap();
    session.execute("CREATE EDGE TYPE REPORTS_TO").unwrap();

    // Create graph type referencing them
    session
        .execute("CREATE GRAPH TYPE org_chart (NODE TYPE Employee, EDGE TYPE REPORTS_TO)")
        .unwrap();

    // Bind a graph to the type
    session
        .execute("CREATE GRAPH hr_graph TYPED org_chart")
        .unwrap();

    // Clean up
    session.execute("DROP GRAPH TYPE org_chart").unwrap();
    session.execute("DROP NODE TYPE Employee").unwrap();
    session.execute("DROP EDGE TYPE REPORTS_TO").unwrap();
}
