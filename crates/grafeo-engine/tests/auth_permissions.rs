//! Integration tests for role-based access control in sessions and the
//! database projection API.

use grafeo_engine::GrafeoDB;
use grafeo_engine::auth::{Identity, Role};

// ── Session identity creation ────────────────────────────────────

#[test]
fn session_with_identity_admin_can_write() {
    let db = GrafeoDB::new_in_memory();
    let identity = Identity::new("alix", [Role::Admin]);
    let session = db.session_with_identity(identity);

    let result = session.execute("INSERT (:Person {name: 'Alix'})");
    assert!(result.is_ok(), "Admin identity should be able to write");
    assert_eq!(db.node_count(), 1);
}

#[test]
fn session_with_identity_readwrite_can_write() {
    let db = GrafeoDB::new_in_memory();
    let identity = Identity::new("gus", [Role::ReadWrite]);
    let session = db.session_with_identity(identity);

    let result = session.execute("INSERT (:Person {name: 'Gus'})");
    assert!(result.is_ok(), "ReadWrite identity should be able to write");
    assert_eq!(db.node_count(), 1);
}

#[test]
fn session_with_identity_readonly_can_read() {
    let db = GrafeoDB::new_in_memory();

    // Seed some data
    let admin = db.session();
    admin.execute("INSERT (:Person {name: 'Alix'})").unwrap();

    let identity = Identity::new("gus", [Role::ReadOnly]);
    let session = db.session_with_identity(identity);

    let result = session.execute("MATCH (p:Person) RETURN p.name");
    assert!(result.is_ok(), "ReadOnly identity should be able to read");
}

#[test]
fn session_with_identity_readonly_cannot_write() {
    let db = GrafeoDB::new_in_memory();
    let identity = Identity::new("alix", [Role::ReadOnly]);
    let session = db.session_with_identity(identity);

    let result = session.execute("INSERT (:Person {name: 'Alix'})");
    assert!(
        result.is_err(),
        "ReadOnly identity should not be able to write"
    );
    let err_msg = result.unwrap_err().to_string();
    assert!(
        err_msg.contains("permission denied") || err_msg.contains("read-only"),
        "Error should indicate permission denial, got: {err_msg}"
    );
}

// ── session_with_role convenience ────────────────────────────────

#[test]
fn session_with_role_readonly() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session_with_role(Role::ReadOnly);

    let result = session.execute("INSERT (:City {name: 'Amsterdam'})");
    assert!(result.is_err(), "ReadOnly role should not allow writes");
}

#[test]
fn session_with_role_readwrite() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session_with_role(Role::ReadWrite);

    let result = session.execute("INSERT (:City {name: 'Berlin'})");
    assert!(result.is_ok(), "ReadWrite role should allow writes");
    assert_eq!(db.node_count(), 1);
}

#[test]
fn session_with_role_admin() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session_with_role(Role::Admin);

    let result = session.execute("INSERT (:City {name: 'Paris'})");
    assert!(result.is_ok(), "Admin role should allow writes");
}

// ── Parameterized queries with permissions ───────────────────────

#[test]
fn readonly_session_parameterized_read_succeeds() {
    let db = GrafeoDB::new_in_memory();

    // Seed data with an admin session
    let admin = db.session();
    admin.execute("INSERT (:Person {name: 'Alix'})").unwrap();

    let session = db.session_with_role(Role::ReadOnly);
    let params = std::collections::HashMap::from([(
        "name".to_string(),
        grafeo_common::types::Value::from("Alix"),
    )]);

    let result = session.execute_with_params(
        "MATCH (p:Person) WHERE p.name = $name RETURN p.name",
        params,
    );
    assert!(
        result.is_ok(),
        "ReadOnly session should execute parameterized reads: {:?}",
        result.err()
    );
}

#[test]
fn readonly_session_parameterized_write_fails() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session_with_role(Role::ReadOnly);
    let params = std::collections::HashMap::from([(
        "name".to_string(),
        grafeo_common::types::Value::from("Gus"),
    )]);

    let result = session.execute_with_params("INSERT (:Person {name: $name})", params);
    assert!(
        result.is_err(),
        "ReadOnly session should reject parameterized writes"
    );
    let err_msg = result.unwrap_err().to_string();
    assert!(
        err_msg.contains("permission denied") || err_msg.contains("read-only"),
        "Error should mention permission denial, got: {err_msg}"
    );
}

#[test]
fn readwrite_session_parameterized_write_succeeds() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session_with_role(Role::ReadWrite);
    let params = std::collections::HashMap::from([(
        "name".to_string(),
        grafeo_common::types::Value::from("Alix"),
    )]);

    let result = session.execute_with_params("INSERT (:Person {name: $name})", params);
    assert!(
        result.is_ok(),
        "ReadWrite session should accept parameterized writes: {:?}",
        result.err()
    );
    assert_eq!(db.node_count(), 1);
}

// ── Projection API ──────────────────────────────────────────────

#[test]
fn create_and_list_projections() {
    let db = GrafeoDB::new_in_memory();

    // Seed some data
    let admin = db.session();
    admin
        .execute("INSERT (:Person {name: 'Alix'})-[:LIVES_IN]->(:City {name: 'Amsterdam'})")
        .unwrap();

    let spec = grafeo_core::graph::ProjectionSpec::new()
        .with_node_labels(["Person", "City"])
        .with_edge_types(["LIVES_IN"]);

    assert!(
        db.create_projection("social", spec),
        "first creation should succeed"
    );

    let names = db.list_projections();
    assert_eq!(names.len(), 1);
    assert!(names.contains(&"social".to_string()));
}

#[test]
fn create_projection_duplicate_returns_false() {
    let db = GrafeoDB::new_in_memory();

    let spec1 = grafeo_core::graph::ProjectionSpec::new().with_node_labels(["Person"]);
    let spec2 = grafeo_core::graph::ProjectionSpec::new().with_node_labels(["City"]);

    assert!(db.create_projection("proj", spec1));
    assert!(
        !db.create_projection("proj", spec2),
        "duplicate name should return false"
    );
}

#[test]
fn drop_projection_existing() {
    let db = GrafeoDB::new_in_memory();

    let spec = grafeo_core::graph::ProjectionSpec::new().with_node_labels(["Person"]);
    db.create_projection("temp", spec);

    assert!(
        db.drop_projection("temp"),
        "dropping existing projection should return true"
    );
    assert!(
        db.list_projections().is_empty(),
        "{:?}",
        db.list_projections()
    );
}

#[test]
fn drop_projection_nonexistent() {
    let db = GrafeoDB::new_in_memory();
    assert!(
        !db.drop_projection("nonexistent"),
        "dropping nonexistent should return false"
    );
}

#[test]
fn get_projection_by_name() {
    let db = GrafeoDB::new_in_memory();

    // Seed data
    let admin = db.session();
    admin.execute("INSERT (:Person {name: 'Gus'})").unwrap();

    let spec = grafeo_core::graph::ProjectionSpec::new().with_node_labels(["Person"]);
    db.create_projection("people", spec);

    let proj = db.projection("people");
    assert!(proj.is_some(), "projection should be retrievable");

    assert!(db.projection("nonexistent").is_none());
}

// ── SPARQL permission checks ────────────────────────────────────

#[cfg(all(feature = "sparql", feature = "triple-store"))]
#[test]
fn sparql_select_with_readonly_succeeds() {
    let db = GrafeoDB::new_in_memory();

    // Seed some RDF data via admin session
    let admin = db.session();
    let _ = admin.execute_sparql(
        "INSERT DATA { <http://example.org/alix> <http://example.org/name> \"Alix\" . }",
    );

    let session = db.session_with_role(Role::ReadOnly);
    let result = session.execute_sparql("SELECT ?s ?o WHERE { ?s <http://example.org/name> ?o }");
    assert!(
        result.is_ok(),
        "ReadOnly should execute SPARQL SELECT: {:?}",
        result.err()
    );
}

#[cfg(all(feature = "sparql", feature = "triple-store"))]
#[test]
fn sparql_insert_with_readonly_fails() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session_with_role(Role::ReadOnly);

    let result = session.execute_sparql(
        "INSERT DATA { <http://example.org/gus> <http://example.org/name> \"Gus\" . }",
    );
    assert!(result.is_err(), "ReadOnly should not execute SPARQL INSERT");
    let err_msg = result.unwrap_err().to_string();
    assert!(
        err_msg.contains("permission denied") || err_msg.contains("read-only"),
        "Error should mention permission denial, got: {err_msg}"
    );
}

// ── SPARQL permission checks on RDF-model database ─────────────

#[cfg(all(feature = "sparql", feature = "triple-store"))]
#[test]
fn sparql_select_on_rdf_database_with_readonly_succeeds() {
    use grafeo_engine::config::{Config, GraphModel};

    let db = GrafeoDB::with_config(Config::in_memory().with_graph_model(GraphModel::Rdf)).unwrap();

    // Seed triples via admin session
    let admin = db.session();
    admin
        .execute_sparql(
            "INSERT DATA { <http://example.org/alix> <http://example.org/name> \"Alix\" . }",
        )
        .unwrap();

    let session = db.session_with_role(Role::ReadOnly);
    let result = session.execute_sparql("SELECT ?s ?o WHERE { ?s <http://example.org/name> ?o }");
    assert!(
        result.is_ok(),
        "ReadOnly should execute SPARQL SELECT on RDF database: {:?}",
        result.err()
    );
}

#[cfg(all(feature = "sparql", feature = "triple-store"))]
#[test]
fn sparql_insert_on_rdf_database_with_readonly_fails() {
    use grafeo_engine::config::{Config, GraphModel};

    let db = GrafeoDB::with_config(Config::in_memory().with_graph_model(GraphModel::Rdf)).unwrap();
    let session = db.session_with_role(Role::ReadOnly);

    let result = session.execute_sparql(
        "INSERT DATA { <http://example.org/gus> <http://example.org/name> \"Gus\" . }",
    );
    assert!(
        result.is_err(),
        "ReadOnly should not execute SPARQL INSERT on RDF database"
    );
    let err_msg = result.unwrap_err().to_string();
    assert!(
        err_msg.contains("permission denied") || err_msg.contains("read-only"),
        "Error should mention permission denial, got: {err_msg}"
    );
}

// ── GQL with_params permission checks (restricted identity path) ─

#[cfg(feature = "gql")]
#[test]
fn gql_with_params_read_with_readonly_succeeds() {
    let db = GrafeoDB::new_in_memory();

    // Seed data with admin session
    let admin = db.session();
    admin.execute("INSERT (:Person {name: 'Alix'})").unwrap();

    let session = db.session_with_role(Role::ReadOnly);
    let params = std::collections::HashMap::from([(
        "name".to_string(),
        grafeo_common::types::Value::from("Alix"),
    )]);

    let result = session.execute_with_params(
        "MATCH (p:Person) WHERE p.name = $name RETURN p.name",
        params,
    );
    assert!(
        result.is_ok(),
        "ReadOnly should execute parameterized GQL reads: {:?}",
        result.err()
    );
}

#[cfg(feature = "gql")]
#[test]
fn gql_with_params_write_with_readonly_fails() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session_with_role(Role::ReadOnly);
    let params = std::collections::HashMap::from([(
        "name".to_string(),
        grafeo_common::types::Value::from("Vincent"),
    )]);

    let result = session.execute_with_params("INSERT (:Person {name: $name})", params);
    assert!(
        result.is_err(),
        "ReadOnly should reject parameterized GQL writes"
    );
    let err_msg = result.unwrap_err().to_string();
    assert!(
        err_msg.contains("permission denied") || err_msg.contains("read-only"),
        "Error should mention permission denial, got: {err_msg}"
    );
}

// ── Cypher via execute_language with restricted identity ────────

#[cfg(feature = "cypher")]
#[test]
fn cypher_execute_language_write_with_readonly_fails() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session_with_role(Role::ReadOnly);
    let params = std::collections::HashMap::from([(
        "name".to_string(),
        grafeo_common::types::Value::from("Jules"),
    )]);

    let result =
        session.execute_language("CREATE (n:Person {name: $name})", "cypher", Some(params));
    assert!(
        result.is_err(),
        "ReadOnly should reject Cypher writes via execute_language"
    );
    let err_msg = result.unwrap_err().to_string();
    assert!(
        err_msg.contains("permission denied") || err_msg.contains("read-only"),
        "Error should mention permission denial, got: {err_msg}"
    );
}

// ── Per-graph grants through the Rust API ──────────────────────

/// `use_graph` selects a graph without the grant check `USE GRAPH` makes:
/// statements and direct writes check the grant themselves, reads needing
/// any grant for the graph and writes a read-write one. `EXPLAIN` needs what
/// the statement it shows would need.
#[test]
fn per_graph_grants_hold_after_use_graph() {
    use grafeo_engine::auth::Grant;

    let db = GrafeoDB::new_in_memory();
    for graph in ["open", "readonly", "closed"] {
        db.create_graph(graph).unwrap();
    }
    let identity = Identity::new("gus", [Role::ReadWrite]).with_grants([
        Grant::new("open", Role::ReadWrite),
        Grant::new("readonly", Role::ReadOnly),
    ]);
    let session = db.session_with_identity(identity);
    let denied = |result: String| assert!(result.contains("permission denied"), "{result}");

    session.use_graph("closed");
    denied(
        session
            .execute("MATCH (n) RETURN n")
            .unwrap_err()
            .to_string(),
    );
    denied(session.execute("INSERT (:X)").unwrap_err().to_string());
    denied(session.create_node(&["X"]).unwrap_err().to_string());
    denied(
        session
            .execute("EXPLAIN MATCH (n) RETURN n")
            .unwrap_err()
            .to_string(),
    );
    #[cfg(feature = "cypher")]
    denied(
        session
            .execute_cypher("EXPLAIN MATCH (n) RETURN n")
            .unwrap_err()
            .to_string(),
    );
    let Err(error) = session.execute_streaming("MATCH (n) RETURN n") else {
        panic!("a stream of a graph without a grant");
    };
    denied(error.to_string());

    session.use_graph("readonly");
    session.execute("MATCH (n) RETURN count(n)").unwrap();
    denied(session.execute("INSERT (:X)").unwrap_err().to_string());
    denied(session.create_node(&["X"]).unwrap_err().to_string());
    session.execute("EXPLAIN MATCH (n) RETURN n").unwrap();
    assert!(session.execute_streaming("MATCH (n) RETURN n").is_ok());
    denied(
        session
            .execute("EXPLAIN INSERT (:X)")
            .unwrap_err()
            .to_string(),
    );

    session.use_graph("open");
    session.execute("INSERT (:X)").unwrap();
    session.create_node(&["X"]).unwrap();
    session.execute("EXPLAIN INSERT (:X)").unwrap();

    let admin = db.session();
    for (graph, expected) in [("open", 2), ("readonly", 0), ("closed", 0)] {
        admin.use_graph(graph);
        let count = admin.execute("MATCH (n) RETURN count(n)").unwrap();
        assert_eq!(
            count.rows()[0][0],
            grafeo_common::types::Value::Int64(expected),
            "{graph}"
        );
    }
}
