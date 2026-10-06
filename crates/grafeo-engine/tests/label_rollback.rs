//! Tests for label mutations in transactions.
//!
//! Verifies that ADD/REMOVE label operations are correctly undone
//! when a transaction is rolled back, and that they reach a node
//! created in the same transaction.

use grafeo_common::types::Value;
use grafeo_engine::GrafeoDB;

#[test]
fn test_add_label_rollback_removes_label() {
    let db = GrafeoDB::new_in_memory();
    let mut session = db.session();

    session.execute("INSERT (:Person {name: 'Alix'})").unwrap();

    // Verify initial labels
    let result = session
        .execute("MATCH (p:Person {name: 'Alix'}) RETURN labels(p)")
        .unwrap();
    assert_eq!(result.row_count(), 1);

    // Begin transaction, add a label, then rollback
    session.begin_transaction().unwrap();
    session
        .execute("MATCH (p:Person {name: 'Alix'}) SET p:Employee")
        .unwrap();

    // Verify label was added within transaction
    let result = session
        .execute("MATCH (p:Employee {name: 'Alix'}) RETURN p.name")
        .unwrap();
    assert_eq!(
        result.row_count(),
        1,
        "should find node with Employee label in tx"
    );

    session.rollback().unwrap();

    // Label should be gone after rollback
    let result = session
        .execute("MATCH (p:Employee {name: 'Alix'}) RETURN p.name")
        .unwrap();
    assert_eq!(
        result.row_count(),
        0,
        "Employee label should not exist after rollback"
    );

    // Original label should still be there
    let result = session
        .execute("MATCH (p:Person {name: 'Alix'}) RETURN p.name")
        .unwrap();
    assert_eq!(result.row_count(), 1, "Person label should still exist");
}

#[test]
fn test_remove_label_rollback_restores_label() {
    let db = GrafeoDB::new_in_memory();
    let mut session = db.session();

    session
        .execute("INSERT (:Person:Employee {name: 'Gus'})")
        .unwrap();

    // Verify both labels exist
    let result = session
        .execute("MATCH (p:Employee {name: 'Gus'}) RETURN p.name")
        .unwrap();
    assert_eq!(result.row_count(), 1);

    // Begin transaction, remove a label, then rollback
    session.begin_transaction().unwrap();
    session
        .execute("MATCH (p:Person {name: 'Gus'}) REMOVE p:Employee")
        .unwrap();

    // Verify label was removed within transaction
    let result = session
        .execute("MATCH (p:Employee {name: 'Gus'}) RETURN p.name")
        .unwrap();
    assert_eq!(result.row_count(), 0, "Employee label should be gone in tx");

    session.rollback().unwrap();

    // Label should be restored after rollback
    let result = session
        .execute("MATCH (p:Employee {name: 'Gus'}) RETURN p.name")
        .unwrap();
    assert_eq!(
        result.row_count(),
        1,
        "Employee label should be restored after rollback"
    );
}

#[test]
fn test_add_label_committed_stays() {
    let db = GrafeoDB::new_in_memory();
    let mut session = db.session();

    session
        .execute("INSERT (:Person {name: 'Vincent'})")
        .unwrap();

    // Add label in a committed transaction
    session.begin_transaction().unwrap();
    session
        .execute("MATCH (p:Person {name: 'Vincent'}) SET p:VIP")
        .unwrap();
    session.commit().unwrap();

    // Label should persist
    let result = session
        .execute("MATCH (p:VIP {name: 'Vincent'}) RETURN p.name")
        .unwrap();
    assert_eq!(
        result.row_count(),
        1,
        "VIP label should persist after commit"
    );
}

// ============================================================================
// Label undo via transaction rollback: covers property_ops.rs LabelAdded/LabelRemoved
// ============================================================================

#[test]
fn test_label_add_undo_on_transaction_rollback() {
    let db = GrafeoDB::new_in_memory();
    let mut session = db.session();

    session
        .execute("INSERT (:Animal {species: 'Cat'})")
        .unwrap();

    session.begin_transaction().unwrap();
    session.execute("MATCH (n:Animal) SET n:Pet").unwrap();

    let during = session.execute("MATCH (n:Pet) RETURN n.species").unwrap();
    assert_eq!(during.rows().len(), 1);

    session.rollback().unwrap();

    let after = session.execute("MATCH (n:Pet) RETURN n.species").unwrap();
    assert!(
        after.rows().is_empty(),
        "Label should be removed after rollback"
    );
}

#[test]
fn test_label_remove_undo_on_transaction_rollback() {
    let db = GrafeoDB::new_in_memory();
    let mut session = db.session();

    session
        .execute("INSERT (:Animal:Pet {species: 'Dog'})")
        .unwrap();

    session.begin_transaction().unwrap();
    session.execute("MATCH (n:Animal) REMOVE n:Pet").unwrap();

    let during = session.execute("MATCH (n:Pet) RETURN n.species").unwrap();
    assert!(during.rows().is_empty(), "{:?}", during.rows());

    session.rollback().unwrap();

    let after = session.execute("MATCH (n:Pet) RETURN n.species").unwrap();
    assert_eq!(
        after.rows().len(),
        1,
        "Label should be restored after rollback"
    );
}

// ============================================================================
// Label changes on a node created in the same transaction
// ============================================================================

/// The labels of the node named `name`, sorted.
fn labels_of(session: &grafeo_engine::Session, name: &str) -> Vec<Value> {
    let result = session
        .execute(&format!(
            "MATCH (p {{name: '{name}'}}) UNWIND labels(p) AS label RETURN label ORDER BY label"
        ))
        .unwrap();
    result.rows().iter().map(|row| row[0].clone()).collect()
}

/// `SET p:Label` on a node the transaction created sticks. Without the
/// `temporal` feature (the Python and Node.js packages) the label was
/// dropped, as the store looked for the node among the committed ones.
#[test]
fn a_label_set_on_a_node_created_in_the_same_transaction_is_committed() {
    let db = GrafeoDB::new_in_memory();
    let mut session = db.session();

    session.begin_transaction().unwrap();
    session.execute("INSERT (:Person {name: 'Alix'})").unwrap();
    session
        .execute("MATCH (p:Person {name: 'Alix'}) SET p:Admin")
        .unwrap();
    assert_eq!(
        labels_of(&session, "Alix"),
        [Value::from("Admin"), Value::from("Person")],
        "the transaction sees the label it set"
    );
    session.commit().unwrap();

    assert_eq!(
        labels_of(&session, "Alix"),
        [Value::from("Admin"), Value::from("Person")]
    );
    let admins = session.execute("MATCH (p:Admin) RETURN p.name").unwrap();
    assert_eq!(admins.rows(), [vec![Value::from("Alix")]]);
}

/// `REMOVE p:Label` on a node the transaction created sticks too.
#[test]
fn a_label_removed_from_a_node_created_in_the_same_transaction_stays_removed() {
    let db = GrafeoDB::new_in_memory();
    let mut session = db.session();

    session.begin_transaction().unwrap();
    session
        .execute("INSERT (:Person:Guest {name: 'Gus'})")
        .unwrap();
    session
        .execute("MATCH (p:Person {name: 'Gus'}) REMOVE p:Guest")
        .unwrap();
    session.commit().unwrap();

    assert_eq!(labels_of(&session, "Gus"), [Value::from("Person")]);
    let guests = session.execute("MATCH (p:Guest) RETURN p.name").unwrap();
    assert_eq!(guests.row_count(), 0, "{:?}", guests.rows());
}

/// A statement outside a transaction runs in one of its own, so a statement
/// that creates a node and labels it took the same path and lost the label.
#[test]
fn one_statement_that_creates_and_labels_a_node_keeps_the_label() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();

    session
        .execute("INSERT (p:Person {name: 'Vincent'}) SET p:New")
        .unwrap();

    assert_eq!(
        labels_of(&session, "Vincent"),
        [Value::from("New"), Value::from("Person")]
    );
}

/// A rolled-back transaction leaves nothing of a node it created and
/// labeled: no node, and no entry under either label.
#[test]
fn a_rollback_drops_a_created_node_with_the_labels_set_on_it() {
    let db = GrafeoDB::new_in_memory();
    let mut session = db.session();

    session.begin_transaction().unwrap();
    session.execute("INSERT (:Person {name: 'Mia'})").unwrap();
    session
        .execute("MATCH (p:Person {name: 'Mia'}) SET p:Admin")
        .unwrap();
    session.rollback().unwrap();

    for query in ["MATCH (p:Person) RETURN p", "MATCH (p:Admin) RETURN p"] {
        let result = session.execute(query).unwrap();
        assert_eq!(result.row_count(), 0, "{query}: {:?}", result.rows());
    }
}
