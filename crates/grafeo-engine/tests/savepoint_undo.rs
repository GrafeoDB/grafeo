//! Tests for savepoint interaction with the transaction's change set.
//!
//! Verifies that rollback_to_savepoint correctly undoes property and label
//! mutations made after the savepoint, while preserving earlier changes.

use grafeo_common::types::Value;
use grafeo_engine::GrafeoDB;

#[test]
fn test_savepoint_rolls_back_set_property() {
    let db = GrafeoDB::new_in_memory();
    let mut session = db.session();

    session
        .execute("INSERT (:Account {owner: 'Alix', balance: 1000})")
        .unwrap();

    session.begin_transaction().unwrap();

    // First mutation: within the transaction, before the savepoint
    session
        .execute("MATCH (a:Account {owner: 'Alix'}) SET a.balance = 2000")
        .unwrap();

    session.savepoint("sp1").unwrap();

    // Second mutation: after the savepoint
    session
        .execute("MATCH (a:Account {owner: 'Alix'}) SET a.balance = 9999")
        .unwrap();

    // Rollback to savepoint: should undo the second SET but keep the first
    session.rollback_to_savepoint("sp1").unwrap();

    let result = session
        .execute("MATCH (a:Account {owner: 'Alix'}) RETURN a.balance")
        .unwrap();
    assert_eq!(
        result.rows()[0][0],
        Value::Int64(2000),
        "balance should be 2000 (pre-savepoint value), not 9999"
    );

    // Commit the transaction: the pre-savepoint change should persist
    session.commit().unwrap();

    let result = session
        .execute("MATCH (a:Account {owner: 'Alix'}) RETURN a.balance")
        .unwrap();
    assert_eq!(result.rows()[0][0], Value::Int64(2000));
}

#[test]
fn test_savepoint_rolls_back_new_property() {
    let db = GrafeoDB::new_in_memory();
    let mut session = db.session();

    session.execute("INSERT (:Person {name: 'Gus'})").unwrap();

    session.begin_transaction().unwrap();
    session.savepoint("sp1").unwrap();

    // Add a new property after the savepoint
    session
        .execute("MATCH (p:Person {name: 'Gus'}) SET p.status = 'active'")
        .unwrap();

    session.rollback_to_savepoint("sp1").unwrap();

    // Property should not exist
    let result = session
        .execute("MATCH (p:Person {name: 'Gus'}) RETURN p.status")
        .unwrap();
    assert_eq!(
        result.rows()[0][0],
        Value::Null,
        "status property should not exist after savepoint rollback"
    );

    session.commit().unwrap();
}

#[test]
fn test_savepoint_rolls_back_label_add() {
    let db = GrafeoDB::new_in_memory();
    let mut session = db.session();

    session
        .execute("INSERT (:Person {name: 'Vincent'})")
        .unwrap();

    session.begin_transaction().unwrap();
    session.savepoint("sp1").unwrap();

    // Add a label after the savepoint
    session
        .execute("MATCH (p:Person {name: 'Vincent'}) SET p:VIP")
        .unwrap();

    // Verify label exists
    let result = session
        .execute("MATCH (p:VIP {name: 'Vincent'}) RETURN p.name")
        .unwrap();
    assert_eq!(result.row_count(), 1);

    session.rollback_to_savepoint("sp1").unwrap();

    // Label should be gone
    let result = session
        .execute("MATCH (p:VIP {name: 'Vincent'}) RETURN p.name")
        .unwrap();
    assert_eq!(
        result.row_count(),
        0,
        "VIP label should be removed after savepoint rollback"
    );

    // Original label should still exist
    let result = session
        .execute("MATCH (p:Person {name: 'Vincent'}) RETURN p.name")
        .unwrap();
    assert_eq!(result.row_count(), 1);

    session.commit().unwrap();
}

#[test]
fn test_savepoint_preserves_pre_savepoint_changes() {
    let db = GrafeoDB::new_in_memory();
    let mut session = db.session();

    session
        .execute("INSERT (:Person {name: 'Jules', age: 30})")
        .unwrap();

    session.begin_transaction().unwrap();

    // Change before savepoint
    session
        .execute("MATCH (p:Person {name: 'Jules'}) SET p.age = 40")
        .unwrap();
    session
        .execute("MATCH (p:Person {name: 'Jules'}) SET p:Senior")
        .unwrap();

    session.savepoint("sp1").unwrap();

    // Changes after savepoint
    session
        .execute("MATCH (p:Person {name: 'Jules'}) SET p.age = 99")
        .unwrap();

    session.rollback_to_savepoint("sp1").unwrap();

    // Pre-savepoint changes should remain
    let result = session
        .execute("MATCH (p:Person {name: 'Jules'}) RETURN p.age")
        .unwrap();
    assert_eq!(
        result.rows()[0][0],
        Value::Int64(40),
        "age should be 40 (pre-savepoint value)"
    );

    let result = session
        .execute("MATCH (p:Senior {name: 'Jules'}) RETURN p.name")
        .unwrap();
    assert_eq!(
        result.row_count(),
        1,
        "Senior label from before savepoint should remain"
    );

    session.commit().unwrap();
}

#[test]
fn test_full_rollback_after_savepoint_undoes_everything() {
    let db = GrafeoDB::new_in_memory();
    let mut session = db.session();

    session
        .execute("INSERT (:Account {owner: 'Mia', balance: 500})")
        .unwrap();

    session.begin_transaction().unwrap();

    session
        .execute("MATCH (a:Account {owner: 'Mia'}) SET a.balance = 1000")
        .unwrap();

    session.savepoint("sp1").unwrap();

    session
        .execute("MATCH (a:Account {owner: 'Mia'}) SET a.balance = 2000")
        .unwrap();

    // Full rollback (not savepoint): should undo everything
    session.rollback().unwrap();

    let result = session
        .execute("MATCH (a:Account {owner: 'Mia'}) RETURN a.balance")
        .unwrap();
    assert_eq!(
        result.rows()[0][0],
        Value::Int64(500),
        "balance should be restored to original 500 after full rollback"
    );
}

// ============================================================================
// Savepoint rollback with labels
// ============================================================================

#[test]
fn test_savepoint_rollback_preserves_earlier_label() {
    let db = GrafeoDB::new_in_memory();
    let mut session = db.session();

    session.execute("INSERT (:Base {name: 'test'})").unwrap();

    session.begin_transaction().unwrap();

    // Mutation before savepoint
    session.execute("MATCH (n:Base) SET n:First").unwrap();

    session.savepoint("sp1").unwrap();

    // Mutation after savepoint
    session.execute("MATCH (n:Base) SET n:Second").unwrap();

    let during = session.execute("MATCH (n:Second) RETURN n.name").unwrap();
    assert_eq!(during.rows().len(), 1);

    // Rollback to savepoint: Second removed, First preserved
    session.rollback_to_savepoint("sp1").unwrap();

    let first_check = session.execute("MATCH (n:First) RETURN n.name").unwrap();
    assert_eq!(
        first_check.rows().len(),
        1,
        "First label should survive savepoint rollback"
    );

    let second_check = session.execute("MATCH (n:Second) RETURN n.name").unwrap();
    assert!(
        second_check.rows().is_empty(),
        "Second label should be rolled back"
    );
}
