//! Integration tests verifying that CDC records session-driven mutations.
//!
//! Before the `CdcGraphStore` decorator, only direct CRUD API calls
//! (`db.create_node()`, `db.set_node_property()`) generated CDC events.
//! Session mutations via `session.execute("INSERT ...")` bypassed CDC entirely.
// Test IDs originate as u64 counters stored in i64; roundtrip is lossless
#![allow(clippy::cast_sign_loss)]
//!
//! These tests verify the decorator correctly buffers events during mutations,
//! flushes them on commit, and discards them on rollback.
//!
//! ```bash
//! cargo test --features "full" -p grafeo-engine --test cdc_session_mutations
//! ```

#![cfg(all(feature = "cdc", feature = "gql"))]

use std::collections::HashMap;

use grafeo_common::types::Value;
use grafeo_engine::cdc::{ChangeKind, EntityId};
use grafeo_engine::{Config, GrafeoDB};

fn db() -> GrafeoDB {
    GrafeoDB::with_config(Config::in_memory().with_cdc()).unwrap()
}

// ============================================================================
// Basic session mutations generate CDC events
// ============================================================================

#[test]
fn insert_through_session_generates_create_event() {
    let db = db();
    let session = db.session();
    session
        .execute("INSERT (:Person {name: 'Alix', age: 30})")
        .unwrap();

    // Find the node ID
    let result = session
        .execute("MATCH (n:Person {name: 'Alix'}) RETURN id(n) AS nid")
        .unwrap();
    assert_eq!(result.row_count(), 1, "Node should exist after INSERT");

    let node_id = match &result.rows()[0][0] {
        grafeo_common::types::Value::Int64(id) => grafeo_common::types::NodeId::new(*id as u64),
        other => panic!("Expected Int64 node ID, got: {other:?}"),
    };

    // Check CDC recorded the creation
    let history = db.history(node_id).unwrap();
    assert!(
        !history.is_empty(),
        "CDC should record session INSERT, got 0 events"
    );
    assert!(
        history.iter().any(|e| e.kind == ChangeKind::Create),
        "Should contain a Create event for the session INSERT"
    );
}

#[test]
fn set_through_session_generates_update_event() {
    let db = db();
    let session = db.session();
    session.execute("INSERT (:Person {name: 'Alix'})").unwrap();

    let result = session
        .execute("MATCH (n:Person {name: 'Alix'}) RETURN id(n)")
        .unwrap();
    let node_id = match &result.rows()[0][0] {
        grafeo_common::types::Value::Int64(id) => grafeo_common::types::NodeId::new(*id as u64),
        other => panic!("Expected Int64, got: {other:?}"),
    };

    // Now SET a property through session
    session
        .execute("MATCH (n:Person {name: 'Alix'}) SET n.city = 'Amsterdam'")
        .unwrap();

    let history = db.history(node_id).unwrap();
    let update_count = history
        .iter()
        .filter(|e| e.kind == ChangeKind::Update)
        .count();
    assert!(
        update_count >= 1,
        "Should have at least 1 Update event from SET, got {update_count}"
    );
}

#[test]
fn delete_through_session_generates_delete_event() {
    let db = db();
    let session = db.session();
    session.execute("INSERT (:Person {name: 'Alix'})").unwrap();

    let result = session
        .execute("MATCH (n:Person {name: 'Alix'}) RETURN id(n)")
        .unwrap();
    let node_id = match &result.rows()[0][0] {
        grafeo_common::types::Value::Int64(id) => grafeo_common::types::NodeId::new(*id as u64),
        other => panic!("Expected Int64, got: {other:?}"),
    };

    session
        .execute("MATCH (n:Person {name: 'Alix'}) DELETE n")
        .unwrap();

    let history = db.history(node_id).unwrap();
    assert!(
        history.iter().any(|e| e.kind == ChangeKind::Delete),
        "Should contain a Delete event from session DELETE"
    );
}

// ============================================================================
// Transaction semantics: rollback discards CDC events
// ============================================================================

#[test]
fn rollback_discards_cdc_events() {
    let db = db();
    let mut session = db.session();

    session.begin_transaction().unwrap();
    session.execute("INSERT (:Person {name: 'Gus'})").unwrap();
    session.rollback().unwrap();

    // After rollback, there should be no nodes and no CDC events
    let result = session
        .execute("MATCH (n:Person) RETURN count(n) AS cnt")
        .unwrap();
    assert_eq!(
        result.rows()[0][0],
        grafeo_common::types::Value::Int64(0),
        "Rolled-back node should not exist"
    );

    // Check that no CDC events leaked
    let changes = db
        .changes_between(
            grafeo_common::types::EpochId::new(0),
            grafeo_common::types::EpochId::new(u64::MAX),
        )
        .unwrap();
    assert!(
        changes.is_empty(),
        "Rolled-back transaction should produce 0 CDC events, got {}",
        changes.len()
    );
}

#[test]
fn multi_statement_transaction_flushes_on_commit() {
    let db = db();
    let mut session = db.session();

    session.begin_transaction().unwrap();
    session.execute("INSERT (:Person {name: 'Alix'})").unwrap();
    session.execute("INSERT (:Person {name: 'Gus'})").unwrap();

    // Before commit: check that CDC log has no events yet
    let pre_commit_changes = db
        .changes_between(
            grafeo_common::types::EpochId::new(0),
            grafeo_common::types::EpochId::new(u64::MAX),
        )
        .unwrap();
    assert!(
        pre_commit_changes.is_empty(),
        "CDC events should not appear before commit, got {}",
        pre_commit_changes.len()
    );

    session.commit().unwrap();

    // After commit: CDC log should have events
    let post_commit_changes = db
        .changes_between(
            grafeo_common::types::EpochId::new(0),
            grafeo_common::types::EpochId::new(u64::MAX),
        )
        .unwrap();
    let create_count = post_commit_changes
        .iter()
        .filter(|e| e.kind == ChangeKind::Create)
        .count();
    assert!(
        create_count >= 2,
        "Should have at least 2 Create events after commit, got {create_count}"
    );
}

// ============================================================================
// Savepoint rollback truncates CDC buffer
// ============================================================================

#[test]
fn savepoint_rollback_discards_post_savepoint_events() {
    let db = db();
    let mut session = db.session();

    session.begin_transaction().unwrap();
    session.execute("INSERT (:Person {name: 'Alix'})").unwrap();
    session.execute("SAVEPOINT sp1").unwrap();
    session.execute("INSERT (:Person {name: 'Gus'})").unwrap();
    session.execute("ROLLBACK TO SAVEPOINT sp1").unwrap();
    session.commit().unwrap();

    // Only Alix should exist, Gus was rolled back
    let result = session
        .execute("MATCH (n:Person) RETURN n.name ORDER BY n.name")
        .unwrap();
    assert_eq!(
        result.row_count(),
        1,
        "Only Alix should exist after savepoint rollback"
    );

    // CDC should only have events for Alix, not Gus
    let changes = db
        .changes_between(
            grafeo_common::types::EpochId::new(0),
            grafeo_common::types::EpochId::new(u64::MAX),
        )
        .unwrap();
    let create_count = changes
        .iter()
        .filter(|e| e.kind == ChangeKind::Create && matches!(e.entity_id, EntityId::Node(_)))
        .count();
    assert_eq!(
        create_count, 1,
        "Should have exactly 1 Create node event (Alix only), got {create_count}"
    );
}

// ============================================================================
// Edge creation/deletion through session
// ============================================================================

#[test]
fn edge_creation_through_session_generates_cdc() {
    let db = db();
    let session = db.session();
    session
        .execute("INSERT (:Person {name: 'Alix'})-[:KNOWS]->(:Person {name: 'Gus'})")
        .unwrap();

    let changes = db
        .changes_between(
            grafeo_common::types::EpochId::new(0),
            grafeo_common::types::EpochId::new(u64::MAX),
        )
        .unwrap();

    let node_creates = changes
        .iter()
        .filter(|e| e.kind == ChangeKind::Create && matches!(e.entity_id, EntityId::Node(_)))
        .count();
    let edge_creates = changes
        .iter()
        .filter(|e| e.kind == ChangeKind::Create && matches!(e.entity_id, EntityId::Edge(_)))
        .count();

    assert!(
        node_creates >= 2,
        "Should have at least 2 node Create events, got {node_creates}"
    );
    assert!(
        edge_creates >= 1,
        "Should have at least 1 edge Create event, got {edge_creates}"
    );
}

// ============================================================================
// Auto-commit mode (single INSERT without explicit transaction)
// ============================================================================

#[test]
fn auto_commit_insert_generates_cdc() {
    let db = db();
    let session = db.session();

    // Single statement without explicit transaction uses auto-commit
    session
        .execute("INSERT (:Person {name: 'Vincent'})")
        .unwrap();

    let changes = db
        .changes_between(
            grafeo_common::types::EpochId::new(0),
            grafeo_common::types::EpochId::new(u64::MAX),
        )
        .unwrap();
    assert!(
        !changes.is_empty(),
        "Auto-commit INSERT should generate CDC events"
    );
    assert!(
        changes.iter().any(|e| e.kind == ChangeKind::Create),
        "Should contain a Create event"
    );
}

// ============================================================================
// Edge deletion through session generates CDC
// ============================================================================

#[test]
fn edge_deletion_through_session_generates_cdc() {
    let db = db();
    let session = db.session();
    session
        .execute("INSERT (:Person {name: 'Alix'})-[:KNOWS]->(:Person {name: 'Gus'})")
        .unwrap();

    // Delete the edge
    session
        .execute("MATCH (:Person {name: 'Alix'})-[r:KNOWS]->(:Person {name: 'Gus'}) DELETE r")
        .unwrap();

    let changes = db
        .changes_between(
            grafeo_common::types::EpochId::new(0),
            grafeo_common::types::EpochId::new(u64::MAX),
        )
        .unwrap();

    let edge_deletes = changes
        .iter()
        .filter(|e| e.kind == ChangeKind::Delete && matches!(e.entity_id, EntityId::Edge(_)))
        .count();
    assert!(
        edge_deletes >= 1,
        "Should have at least 1 edge Delete event, got {edge_deletes}"
    );
}

// ============================================================================
// Property removal through session generates CDC
// ============================================================================

#[test]
fn remove_property_through_session_generates_cdc() {
    let db = db();
    let session = db.session();
    session
        .execute("INSERT (:Person {name: 'Alix', city: 'Amsterdam'})")
        .unwrap();

    // Remove a property
    session
        .execute("MATCH (n:Person {name: 'Alix'}) SET n.city = NULL")
        .unwrap();

    let result = session
        .execute("MATCH (n:Person {name: 'Alix'}) RETURN id(n)")
        .unwrap();
    let node_id = match &result.rows()[0][0] {
        grafeo_common::types::Value::Int64(id) => grafeo_common::types::NodeId::new(*id as u64),
        other => panic!("Expected Int64, got: {other:?}"),
    };

    let history = db.history(node_id).unwrap();
    let update_count = history
        .iter()
        .filter(|e| e.kind == ChangeKind::Update)
        .count();
    assert!(
        update_count >= 1,
        "Should have at least 1 Update event for property removal, got {update_count}"
    );
}

// ============================================================================
// Label mutation through session generates CDC
// ============================================================================

#[test]
fn set_label_through_session_generates_cdc() {
    let db = db();
    let session = db.session();
    session.execute("INSERT (:Person {name: 'Alix'})").unwrap();

    let result = session
        .execute("MATCH (n:Person {name: 'Alix'}) RETURN id(n)")
        .unwrap();
    let node_id = match &result.rows()[0][0] {
        grafeo_common::types::Value::Int64(id) => grafeo_common::types::NodeId::new(*id as u64),
        other => panic!("Expected Int64, got: {other:?}"),
    };

    // Add a label
    session
        .execute("MATCH (n:Person {name: 'Alix'}) SET n:Employee")
        .unwrap();

    let history = db.history(node_id).unwrap();
    // Should have Create + at least one Update (for label and possibly SET)
    assert!(
        history.len() >= 2,
        "Should have at least 2 events after label SET, got {}",
        history.len()
    );
}

// ============================================================================
// Edge property mutation through session generates CDC
// ============================================================================

#[test]
fn set_edge_property_through_session_generates_cdc() {
    let db = db();
    let session = db.session();
    session
        .execute("INSERT (:Person {name: 'Alix'})-[:KNOWS {since: 2020}]->(:Person {name: 'Gus'})")
        .unwrap();

    // Update edge property
    session
        .execute(
            "MATCH (:Person {name: 'Alix'})-[r:KNOWS]->(:Person {name: 'Gus'}) SET r.since = 2025",
        )
        .unwrap();

    let changes = db
        .changes_between(
            grafeo_common::types::EpochId::new(0),
            grafeo_common::types::EpochId::new(u64::MAX),
        )
        .unwrap();

    let edge_updates = changes
        .iter()
        .filter(|e| e.kind == ChangeKind::Update && matches!(e.entity_id, EntityId::Edge(_)))
        .count();
    assert!(
        edge_updates >= 1,
        "Should have at least 1 edge Update event, got {edge_updates}"
    );
}

// ============================================================================
// Node deletion with edges through session
// ============================================================================

#[test]
fn detach_delete_through_session_generates_cdc() {
    let db = db();
    let session = db.session();
    session
        .execute("INSERT (:Person {name: 'Alix'})-[:KNOWS]->(:Person {name: 'Gus'})")
        .unwrap();

    // DETACH DELETE removes node and its edges
    session
        .execute("MATCH (n:Person {name: 'Alix'}) DETACH DELETE n")
        .unwrap();

    let changes = db
        .changes_between(
            grafeo_common::types::EpochId::new(0),
            grafeo_common::types::EpochId::new(u64::MAX),
        )
        .unwrap();

    let node_deletes = changes
        .iter()
        .filter(|e| e.kind == ChangeKind::Delete && matches!(e.entity_id, EntityId::Node(_)))
        .count();
    let edge_deletes = changes
        .iter()
        .filter(|e| e.kind == ChangeKind::Delete && matches!(e.entity_id, EntityId::Edge(_)))
        .count();
    assert!(
        node_deletes >= 1,
        "Should have at least 1 node Delete from DETACH DELETE, got {node_deletes}"
    );
    assert!(
        edge_deletes >= 1,
        "Should have at least 1 edge Delete from DETACH DELETE, got {edge_deletes}"
    );
}

// ============================================================================
// Multiple property updates in single transaction
// ============================================================================

#[test]
fn multiple_property_updates_in_transaction_generate_cdc() {
    let db = db();
    db.execute("INSERT (:Person {name: 'Alix', age: 30})")
        .unwrap();
    let mut session = db.session();

    session.begin_transaction().unwrap();
    session
        .execute("MATCH (n:Person {name: 'Alix'}) SET n.age = 31, n.city = 'Amsterdam'")
        .unwrap();
    session.commit().unwrap();

    let changes = db
        .changes_between(
            grafeo_common::types::EpochId::new(0),
            grafeo_common::types::EpochId::new(u64::MAX),
        )
        .unwrap();

    let mut updates: Vec<_> = changes
        .iter()
        .filter(|e| e.kind == ChangeKind::Update)
        .map(|e| (e.before.clone(), e.after.clone()))
        .collect();
    updates.sort_by_key(|(_, after)| format!("{after:?}"));
    let single = |key: &str, value: Value| Some(HashMap::from([(key.to_string(), value)]));
    assert_eq!(
        updates,
        [
            (
                single("age", Value::Int64(30)),
                single("age", Value::Int64(31))
            ),
            (None, single("city", Value::from("Amsterdam"))),
        ]
    );
}

/// A node created and then changed in one transaction gets one create event
/// that shows it as the transaction left it.
#[test]
fn updates_to_a_node_created_in_the_same_transaction_fold_into_its_create() {
    let db = db();
    let mut session = db.session();

    session.begin_transaction().unwrap();
    session
        .execute("INSERT (:Person {name: 'Alix', age: 30})")
        .unwrap();
    session
        .execute("MATCH (n:Person {name: 'Alix'}) SET n.age = 31, n.city = 'Amsterdam', n:Admin")
        .unwrap();
    session.commit().unwrap();

    let changes = db
        .changes_between(
            grafeo_common::types::EpochId::new(0),
            grafeo_common::types::EpochId::new(u64::MAX),
        )
        .unwrap();

    assert_eq!(changes.len(), 1, "{changes:?}");
    let create = &changes[0];
    assert_eq!(create.kind, ChangeKind::Create);
    let mut labels = create.labels.clone().unwrap_or_default();
    labels.sort();
    assert_eq!(labels, ["Admin", "Person"]);
    assert_eq!(create.before_labels, None);
    assert_eq!(
        create.after,
        Some(HashMap::from([
            ("name".to_string(), Value::from("Alix")),
            ("age".to_string(), Value::Int64(31)),
            ("city".to_string(), Value::from("Amsterdam")),
        ]))
    );
}

/// The fold stays within one graph. A transaction creates a node in the
/// default graph and one in a named graph, which number their nodes alike,
/// and changes the second: each node gets its own create event, and only the
/// second shows the change.
#[test]
fn creates_in_two_graphs_fold_separately() {
    let db = db();
    db.create_graph("g").unwrap();
    let mut session = db.session();

    session.begin_transaction().unwrap();
    session.execute("INSERT (:InDefault {a: 1})").unwrap();
    session.use_graph("g");
    session.execute("INSERT (:InG {b: 1})").unwrap();
    session.execute("MATCH (n:InG) SET n.c = 99").unwrap();
    session.commit().unwrap();

    let changes = db
        .changes_between(
            grafeo_common::types::EpochId::new(0),
            grafeo_common::types::EpochId::new(u64::MAX),
        )
        .unwrap();
    let creates: Vec<_> = changes
        .iter()
        .filter(|e| e.kind == ChangeKind::Create)
        .map(|e| (e.labels.clone().unwrap_or_default(), e.after.clone()))
        .collect();
    assert_eq!(changes.len(), 2, "{changes:?}");
    assert!(
        creates.contains(&(
            vec!["InDefault".to_string()],
            Some(HashMap::from([("a".to_string(), Value::Int64(1))]))
        )),
        "{changes:?}"
    );
    assert!(
        creates.contains(&(
            vec!["InG".to_string()],
            Some(HashMap::from([
                ("b".to_string(), Value::Int64(1)),
                ("c".to_string(), Value::Int64(99)),
            ]))
        )),
        "{changes:?}"
    );
}

/// Create and delete events say what was created or deleted: a node's labels,
/// an edge's type and endpoints, and on a delete the last properties.
#[test]
fn events_describe_the_entity() {
    let db = db();
    let alix = db
        .create_node_with_props(&["Graph", "File"], [("id", Value::from("a"))])
        .unwrap();
    let gus = db
        .create_node_with_props(&["Graph", "Concept"], [("id", Value::from("b"))])
        .unwrap();
    let edge = db
        .create_edge_with_props(alix, gus, "REFERENCES", [("id", Value::from("e1"))])
        .unwrap();
    assert!(db.remove_node_label(alix, "File").unwrap());
    assert!(db.delete_edge(edge).unwrap());
    assert!(db.delete_node(gus).unwrap());

    let changes = db
        .changes_between(
            grafeo_common::types::EpochId::new(0),
            grafeo_common::types::EpochId::new(u64::MAX),
        )
        .unwrap();
    let find = |entity: EntityId, kind: ChangeKind| {
        changes
            .iter()
            .find(|e| e.entity_id == entity && e.kind == kind)
            .unwrap_or_else(|| panic!("no {kind:?} event for {entity:?} in {changes:?}"))
    };
    let sorted = |labels: &Option<Vec<String>>| {
        let mut labels = labels.clone().unwrap_or_default();
        labels.sort();
        labels
    };
    let endpoints = |e: &grafeo_engine::cdc::ChangeEvent| (e.edge_type.clone(), e.src_id, e.dst_id);
    let expected_edge = (
        Some("REFERENCES".to_string()),
        Some(alix.as_u64()),
        Some(gus.as_u64()),
    );
    let only_id = |id: &str| Some(HashMap::from([("id".to_string(), Value::from(id))]));

    let gus_created = find(EntityId::Node(gus), ChangeKind::Create);
    assert_eq!(sorted(&gus_created.labels), ["Concept", "Graph"]);

    let edge_created = find(EntityId::Edge(edge), ChangeKind::Create);
    assert_eq!(endpoints(edge_created), expected_edge);

    let label_removed = find(EntityId::Node(alix), ChangeKind::Update);
    assert_eq!(sorted(&label_removed.before_labels), ["File", "Graph"]);
    assert_eq!(sorted(&label_removed.labels), ["Graph"]);

    let edge_deleted = find(EntityId::Edge(edge), ChangeKind::Delete);
    assert_eq!(endpoints(edge_deleted), expected_edge);
    assert_eq!(edge_deleted.before, only_id("e1"));

    let gus_deleted = find(EntityId::Node(gus), ChangeKind::Delete);
    assert_eq!(sorted(&gus_deleted.labels), ["Concept", "Graph"]);
    assert_eq!(gus_deleted.before, only_id("b"));
}
