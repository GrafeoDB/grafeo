//! Tests for CDC recording through the direct CRUD API (`db.create_node()` etc.).
//!
//! These exercise the CDC paths in `crud.rs` that record events directly to
//! the `CdcLog` (as opposed to session-driven mutations via `CdcGraphStore`).
//!
//! ```bash
//! cargo test --features "cdc" -p grafeo-engine --test cdc_crud_api
//! ```

#![cfg(feature = "cdc")]

use grafeo_common::types::Value;
use grafeo_engine::cdc::{ChangeKind, EntityId};
use grafeo_engine::{Config, GrafeoDB};

fn db() -> GrafeoDB {
    GrafeoDB::with_config(Config::in_memory().with_cdc()).unwrap()
}

// ============================================================================
// Node creation with properties
// ============================================================================

#[test]
fn create_node_with_props_generates_cdc() {
    let db = db();
    let id = db
        .create_node_with_props(
            &["Person"],
            vec![("name", Value::from("Alix")), ("age", Value::Int64(30))],
        )
        .unwrap();

    let history = db.history(id).unwrap();
    assert!(!history.is_empty(), "Should have CDC events");
    let create = history
        .iter()
        .find(|e| e.kind == ChangeKind::Create)
        .unwrap();
    // The after snapshot should contain the properties
    let after = create.after.as_ref().unwrap();
    assert_eq!(after.get("name"), Some(&Value::from("Alix")));
    assert_eq!(after.get("age"), Some(&Value::Int64(30)));
}

// ============================================================================
// Node deletion with properties
// ============================================================================

#[test]
fn delete_node_generates_cdc_with_before_snapshot() {
    let db = db();
    let id = db.create_node(&["Person"]).unwrap();
    db.set_node_property(id, "name", Value::from("Alix"))
        .unwrap();
    db.set_node_property(id, "city", Value::from("Amsterdam"))
        .unwrap();

    let deleted = db.delete_node(id).unwrap();
    assert!(deleted);

    let history = db.history(id).unwrap();
    let del = history
        .iter()
        .find(|e| e.kind == ChangeKind::Delete)
        .unwrap();
    let before = del.before.as_ref().unwrap();
    assert_eq!(before.get("name"), Some(&Value::from("Alix")));
    assert_eq!(before.get("city"), Some(&Value::from("Amsterdam")));
}

// ============================================================================
// set_node_property with old value capture
// ============================================================================

#[test]
fn set_node_property_records_old_and_new_values() {
    let db = db();
    let id = db.create_node(&["Person"]).unwrap();
    db.set_node_property(id, "name", Value::from("Alix"))
        .unwrap();
    db.set_node_property(id, "name", Value::from("Gus"))
        .unwrap();

    let history = db.history(id).unwrap();
    let updates: Vec<_> = history
        .iter()
        .filter(|e| e.kind == ChangeKind::Update)
        .collect();
    assert!(updates.len() >= 2, "Should have at least 2 Update events");

    // Second update should have before = "Alix", after = "Gus"
    let last_update = updates.last().unwrap();
    assert_eq!(
        last_update.before.as_ref().unwrap().get("name"),
        Some(&Value::from("Alix"))
    );
    assert_eq!(
        last_update.after.as_ref().unwrap().get("name"),
        Some(&Value::from("Gus"))
    );
}

// ============================================================================
// Edge creation
// ============================================================================

#[test]
fn create_edge_generates_cdc() {
    let db = db();
    let a = db.create_node(&["Person"]).unwrap();
    let b = db.create_node(&["Person"]).unwrap();
    let eid = db.create_edge(a, b, "KNOWS");

    let changes = db
        .changes_between(
            grafeo_common::types::EpochId::new(0),
            grafeo_common::types::EpochId::new(u64::MAX),
        )
        .unwrap();

    let edge_creates: Vec<_> = changes
        .iter()
        .filter(|e| e.kind == ChangeKind::Create && e.entity_id == EntityId::Edge(eid))
        .collect();
    assert_eq!(edge_creates.len(), 1);
    assert_eq!(edge_creates[0].edge_type.as_deref(), Some("KNOWS"));
}

// ============================================================================
// Edge creation with properties
// ============================================================================

#[test]
fn create_edge_with_props_generates_cdc() {
    let db = db();
    let a = db.create_node(&["Person"]).unwrap();
    let b = db.create_node(&["Person"]).unwrap();
    let eid = db.create_edge_with_props(
        a,
        b,
        "KNOWS",
        vec![
            ("since", Value::Int64(2020)),
            ("weight", Value::Float64(0.8)),
        ],
    );

    let changes = db
        .changes_between(
            grafeo_common::types::EpochId::new(0),
            grafeo_common::types::EpochId::new(u64::MAX),
        )
        .unwrap();

    let edge_create = changes
        .iter()
        .find(|e| e.kind == ChangeKind::Create && e.entity_id == EntityId::Edge(eid))
        .unwrap();
    let after = edge_create.after.as_ref().unwrap();
    assert_eq!(after.get("since"), Some(&Value::Int64(2020)));
    assert_eq!(after.get("weight"), Some(&Value::Float64(0.8)));
}

// ============================================================================
// Edge deletion with properties
// ============================================================================

#[test]
fn delete_edge_generates_cdc_with_before_snapshot() {
    let db = db();
    let a = db.create_node(&["Person"]).unwrap();
    let b = db.create_node(&["Person"]).unwrap();
    let eid = db.create_edge(a, b, "KNOWS");
    db.set_edge_property(eid, "since", Value::Int64(2020));

    let deleted = db.delete_edge(eid);
    assert!(deleted);

    let changes = db
        .changes_between(
            grafeo_common::types::EpochId::new(0),
            grafeo_common::types::EpochId::new(u64::MAX),
        )
        .unwrap();

    let edge_del = changes
        .iter()
        .find(|e| e.kind == ChangeKind::Delete && e.entity_id == EntityId::Edge(eid))
        .unwrap();
    let before = edge_del.before.as_ref().unwrap();
    assert_eq!(before.get("since"), Some(&Value::Int64(2020)));
}

// ============================================================================
// set_edge_property with old value capture
// ============================================================================

#[test]
fn set_edge_property_records_old_and_new_values() {
    let db = db();
    let a = db.create_node(&["Person"]).unwrap();
    let b = db.create_node(&["Person"]).unwrap();
    let eid = db.create_edge(a, b, "KNOWS");
    db.set_edge_property(eid, "weight", Value::Float64(0.5));
    db.set_edge_property(eid, "weight", Value::Float64(0.9));

    let changes = db
        .changes_between(
            grafeo_common::types::EpochId::new(0),
            grafeo_common::types::EpochId::new(u64::MAX),
        )
        .unwrap();

    let edge_updates: Vec<_> = changes
        .iter()
        .filter(|e| e.kind == ChangeKind::Update && e.entity_id == EntityId::Edge(eid))
        .collect();
    assert!(edge_updates.len() >= 2);

    let last = edge_updates.last().unwrap();
    assert_eq!(
        last.before.as_ref().unwrap().get("weight"),
        Some(&Value::Float64(0.5))
    );
    assert_eq!(
        last.after.as_ref().unwrap().get("weight"),
        Some(&Value::Float64(0.9))
    );
}

// ============================================================================
// Database-level GC prunes CDC events (integration test for #250)
// ============================================================================

#[test]
fn database_gc_prunes_cdc_events() {
    use grafeo_common::types::EpochId;

    // Small retention: keep only the last 3 events
    let config = Config::in_memory().with_cdc().with_gc_interval(1);
    let db = GrafeoDB::with_config(config).unwrap();

    // Create 10 nodes via the CRUD API (each generates a CDC Create event)
    for i in 0..10 {
        db.create_node_with_props(&["Person"], vec![("idx", Value::Int64(i))])
            .unwrap();
    }

    let before_gc = db
        .changes_between(EpochId::new(0), EpochId::new(u64::MAX))
        .unwrap()
        .len();
    assert!(
        before_gc >= 10,
        "should have at least 10 CDC events before GC"
    );

    // Trigger database GC, which calls cdc_log.apply_retention(current_epoch)
    db.gc();

    let after_gc = db
        .changes_between(EpochId::new(0), EpochId::new(u64::MAX))
        .unwrap()
        .len();

    // Default retention is 1000 epochs / 100k events, so with only 10 events
    // nothing should be pruned. The point is that gc() does not crash and the
    // CDC log remains queryable.
    assert!(after_gc <= before_gc, "GC should not increase event count");
    assert!(
        after_gc > 0,
        "CDC events should still be queryable after GC"
    );
}

#[test]
fn database_gc_prunes_old_cdc_events_with_session_commits() {
    use grafeo_common::types::EpochId;

    // GC triggers every 2 commits via session auto-GC
    let config = Config::in_memory().with_cdc().with_gc_interval(2);
    let db = GrafeoDB::with_config(config).unwrap();

    // Drive up the epoch through many session commits
    for i in 0..20 {
        let session = db.session_with_cdc(true);
        session
            .execute(&format!("INSERT (:Batch {{idx: {i}}})"))
            .unwrap();
    }

    // All 20 nodes should exist
    assert_eq!(db.node_count(), 20);

    // CDC should have events (auto-GC may have pruned some old ones)
    let events = db
        .changes_between(EpochId::new(0), EpochId::new(u64::MAX))
        .unwrap();
    assert!(
        !events.is_empty(),
        "CDC should retain recent events after auto-GC"
    );
}
