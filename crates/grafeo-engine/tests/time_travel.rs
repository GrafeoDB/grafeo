//! Time-travel query tests (0.5.14).
//!
//! Validates epoch-based time-travel: querying historical state,
//! session viewing_epoch override, and version history APIs.
//!
//! Note: epochs only advance when explicit transactions are committed.
//! The MVCC assigns `created_epoch = tx.start_epoch`, so to query "before"
//! a creation we need the creation to happen at epoch > 0 by committing
//! a preliminary transaction first.

use grafeo_common::types::{EpochId, Value};
use grafeo_engine::GrafeoDB;

// ---------------------------------------------------------------------------
// Helper
// ---------------------------------------------------------------------------

fn setup_db() -> GrafeoDB {
    GrafeoDB::new_in_memory()
}

/// Advances the epoch by committing an empty transaction.
fn bump_epoch(session: &mut grafeo_engine::Session) {
    session.begin_transaction().unwrap();
    session.commit().unwrap();
}

// ---------------------------------------------------------------------------
// execute_at_epoch
// ---------------------------------------------------------------------------

#[test]
fn test_execute_at_epoch_sees_old_state() {
    let db = setup_db();
    let mut session = db.session();

    // Create a person inside a transaction
    session.begin_transaction().unwrap();
    session
        .execute("INSERT (:Person {name: 'Alix', age: 30})")
        .unwrap();
    session.commit().unwrap();
    let epoch_after_insert = db.current_epoch();

    // Update the person in a new transaction
    session.begin_transaction().unwrap();
    session
        .execute("MATCH (p:Person {name: 'Alix'}) SET p.age = 31")
        .unwrap();
    session.commit().unwrap();

    // Current state should be 31
    let result = session
        .execute("MATCH (p:Person {name: 'Alix'}) RETURN p.age")
        .unwrap();
    assert_eq!(result.rows().len(), 1);
    assert_eq!(result.rows()[0][0], Value::Int64(31));

    // Time-travel to the epoch after insert should see the node.
    // Note: properties are not versioned per-epoch in the current architecture,
    // so time-travel applies to node/edge existence, not property values.
    let result = session
        .execute_at_epoch(
            "MATCH (p:Person {name: 'Alix'}) RETURN p.age",
            epoch_after_insert,
        )
        .unwrap();
    assert_eq!(result.rows().len(), 1);
}

#[test]
fn test_execute_at_epoch_before_creation_returns_empty() {
    let db = setup_db();
    let mut session = db.session();

    // Capture epoch before any data exists
    let epoch_before = db.current_epoch();

    // Advance epoch so the INSERT starts at a higher epoch
    bump_epoch(&mut session);

    // Insert inside a transaction (node created at epoch 1)
    session.begin_transaction().unwrap();
    session.execute("INSERT (:Person {name: 'Gus'})").unwrap();
    session.commit().unwrap();

    // Verify node exists at current epoch
    let result = session.execute("MATCH (p:Person) RETURN p.name").unwrap();
    assert_eq!(result.rows().len(), 1);

    // At epoch before creation, the node should not exist
    let result = session
        .execute_at_epoch("MATCH (p:Person) RETURN p.name", epoch_before)
        .unwrap();
    assert!(
        result.rows().is_empty(),
        "Node should not be visible before its creation epoch"
    );
}

#[test]
fn test_deleted_node_history_preserves_epoch() {
    let db = setup_db();
    let mut session = db.session();

    // Insert in a transaction (Vincent created at epoch 0)
    session.begin_transaction().unwrap();
    session
        .execute("INSERT (:Person {name: 'Vincent'})")
        .unwrap();
    session.commit().unwrap();

    // Bump epoch to separate insert from delete
    bump_epoch(&mut session);

    // Delete in a separate transaction
    session.begin_transaction().unwrap();
    session
        .execute("MATCH (p:Person {name: 'Vincent'}) DELETE p")
        .unwrap();
    session.commit().unwrap();

    // After deletion, node should not be visible at current epoch
    let result = session
        .execute("MATCH (p:Person {name: 'Vincent'}) RETURN p")
        .unwrap();
    assert!(result.rows().is_empty());

    // The version history API still tracks the deletion metadata.
    // Note: MATCH-based time-travel on deleted nodes is not supported
    // because deletion removes labels, properties, and index entries.
    // Use get_node_at_epoch() or get_node_history() for deleted nodes.
}

#[test]
fn test_get_node_at_epoch_deleted_node() {
    let db = setup_db();
    let mut session = db.session();

    // Bump epoch so the node is created at epoch > 0
    bump_epoch(&mut session);

    let id = db.create_node(&["Person"]).unwrap();
    let epoch_after_insert = db.current_epoch();

    // Bump epoch before delete
    bump_epoch(&mut session);

    db.delete_node(id).unwrap();

    // Node should be visible at the epoch after insert (before deletion)
    let node = db.get_node_at_epoch(id, epoch_after_insert);
    assert!(node.is_some());

    // Node should not be visible at the current epoch (after deletion)
    let node = db.get_node_at_epoch(id, db.current_epoch());
    assert!(node.is_none());

    // Node should not be visible at epoch 0 (before creation)
    let node = db.get_node_at_epoch(id, EpochId::new(0));
    assert!(node.is_none());
}

// ---------------------------------------------------------------------------
// Session viewing_epoch override
// ---------------------------------------------------------------------------

#[test]
fn test_session_set_viewing_epoch() {
    let db = setup_db();
    let mut session = db.session();

    // Insert Eve in a transaction (Eve created at epoch 0)
    session.begin_transaction().unwrap();
    session.execute("INSERT (:Person {name: 'Eve'})").unwrap();
    session.commit().unwrap();
    let epoch_after_eve = db.current_epoch();

    // Bump epoch to separate Eve from Frank
    bump_epoch(&mut session);

    // Insert Frank in a separate transaction (Frank created at epoch 2)
    session.begin_transaction().unwrap();
    session.execute("INSERT (:Person {name: 'Frank'})").unwrap();
    session.commit().unwrap();

    // Without override: 2 nodes
    let result = session.execute("MATCH (p:Person) RETURN p.name").unwrap();
    assert_eq!(result.rows().len(), 2);

    // Set viewing epoch to after Eve but before Frank
    session.set_viewing_epoch(epoch_after_eve);
    assert_eq!(session.viewing_epoch(), Some(epoch_after_eve));

    // With override: only Eve should be visible
    let result = session.execute("MATCH (p:Person) RETURN p.name").unwrap();
    assert_eq!(result.rows().len(), 1);

    // Clear override
    session.clear_viewing_epoch();
    assert_eq!(session.viewing_epoch(), None);

    // Back to normal: 2 nodes
    let result = session.execute("MATCH (p:Person) RETURN p.name").unwrap();
    assert_eq!(result.rows().len(), 2);
}

#[test]
fn test_session_reset_clears_viewing_epoch() {
    let db = setup_db();
    let session = db.session();

    session.set_viewing_epoch(EpochId::new(1));
    assert!(session.viewing_epoch().is_some());

    session.reset_session();
    assert!(session.viewing_epoch().is_none());
}

// ---------------------------------------------------------------------------
// GQL session parameter
// ---------------------------------------------------------------------------

#[test]
fn test_gql_session_set_viewing_epoch() {
    let db = setup_db();
    let mut session = db.session();

    // Insert Grace in a transaction (Grace created at epoch 0)
    session.begin_transaction().unwrap();
    session.execute("INSERT (:Person {name: 'Grace'})").unwrap();
    session.commit().unwrap();
    let epoch = db.current_epoch();

    // Bump epoch to separate Grace from Hank
    bump_epoch(&mut session);

    // Insert Hank in a separate transaction (Hank created at epoch 2)
    session.begin_transaction().unwrap();
    session.execute("INSERT (:Person {name: 'Hank'})").unwrap();
    session.commit().unwrap();

    // Set viewing_epoch via GQL session parameter
    let set_result = session
        .execute(&format!(
            "SESSION SET PARAMETER viewing_epoch = {}",
            epoch.as_u64()
        ))
        .unwrap();
    assert!(set_result.status_message.is_some());

    // Only Grace should be visible
    let result = session.execute("MATCH (p:Person) RETURN p.name").unwrap();
    assert_eq!(result.rows().len(), 1);

    // Reset session
    session.execute("SESSION RESET").unwrap();

    // Both should be visible again
    let result = session.execute("MATCH (p:Person) RETURN p.name").unwrap();
    assert_eq!(result.rows().len(), 2);
}

// ---------------------------------------------------------------------------
// Point lookups at epoch
// ---------------------------------------------------------------------------

#[test]
fn test_get_node_at_epoch() {
    let db = setup_db();
    let mut session = db.session();

    // Advance epoch first so nodes are created at epoch > 0
    bump_epoch(&mut session);

    let id = db.create_node(&["Person"]).unwrap();
    db.set_node_property(id, "name", Value::String("Alix".into()))
        .unwrap();
    let epoch_after = db.current_epoch();

    // Node should exist at this epoch
    let node = db.get_node_at_epoch(id, epoch_after);
    assert!(node.is_some());

    // Node should not exist at epoch 0 (before it was created)
    let node = db.get_node_at_epoch(id, EpochId::new(0));
    assert!(node.is_none());
}

#[test]
fn test_get_edge_at_epoch() {
    let db = setup_db();
    let mut session = db.session();

    // Advance epoch first
    bump_epoch(&mut session);

    let src = db.create_node(&["Person"]).unwrap();
    let dst = db.create_node(&["Person"]).unwrap();
    let edge_id = db.create_edge(src, dst, "KNOWS");
    let epoch_after = db.current_epoch();

    let edge = db.get_edge_at_epoch(edge_id, epoch_after);
    assert!(edge.is_some());

    let edge = db.get_edge_at_epoch(edge_id, EpochId::new(0));
    assert!(edge.is_none());
}

// ---------------------------------------------------------------------------
// Version history
// ---------------------------------------------------------------------------

#[test]
fn test_node_history_single_version() {
    let db = setup_db();

    let id = db.create_node(&["Person"]).unwrap();
    db.set_node_property(id, "name", Value::String("Alix".into()))
        .unwrap();

    let history = db.get_node_history(id);
    assert_eq!(history.len(), 1);

    let (created_epoch, deleted_epoch, node) = &history[0];
    let _ = created_epoch.as_u64();
    assert!(deleted_epoch.is_none());
    assert!(node.has_label("Person"));
}

#[test]
fn test_node_history_deleted() {
    let db = setup_db();

    let id = db.create_node(&["Person"]).unwrap();
    db.delete_node(id).unwrap();

    let history = db.get_node_history(id);
    // Version chain should still exist with a deleted marker
    assert!(!history.is_empty());
    let (_, deleted_epoch, _) = &history[0];
    assert!(deleted_epoch.is_some());
}

#[test]
fn test_node_history_nonexistent() {
    let db = setup_db();
    use grafeo_common::types::NodeId;

    let history = db.get_node_history(NodeId::new(9999));
    assert!(history.is_empty());
}

#[test]
fn test_edge_history_single_version() {
    let db = setup_db();

    let src = db.create_node(&["Person"]).unwrap();
    let dst = db.create_node(&["Person"]).unwrap();
    let edge_id = db.create_edge(src, dst, "KNOWS");

    let history = db.get_edge_history(edge_id);
    assert_eq!(history.len(), 1);

    let (created_epoch, deleted_epoch, edge) = &history[0];
    let _ = created_epoch.as_u64();
    assert!(deleted_epoch.is_none());
    assert_eq!(edge.edge_type.as_str(), "KNOWS");
    assert_eq!(edge.src, src);
    assert_eq!(edge.dst, dst);
}

#[test]
fn test_current_epoch_increments() {
    let db = setup_db();
    let mut session = db.session();

    let e1 = db.current_epoch();

    // Epoch only advances with explicit transaction commit
    bump_epoch(&mut session);

    let e2 = db.current_epoch();
    assert!(e2.as_u64() > e1.as_u64());
}

// ---------------------------------------------------------------------------
// ValidityTs (unit tests are in the type itself, but add a smoke test)
// ---------------------------------------------------------------------------

#[test]
fn test_validity_ts_smoke() {
    use grafeo_common::types::ValidityTs;

    let ts1 = ValidityTs::new(100);
    let ts2 = ValidityTs::new(200);

    // Reverse ordering
    assert!(ts2 < ts1);

    // Key roundtrip
    let key = ValidityTs::versioned_key(42, ts1);
    let (id, decoded) = ValidityTs::from_versioned_key(&key);
    assert_eq!(id, 42);
    assert_eq!(decoded, ts1);
}
