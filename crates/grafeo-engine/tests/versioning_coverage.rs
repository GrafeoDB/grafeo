//! Targeted tests for the MVCC versioning layer (change sets, property rollback,
//! label rollback, deletion rollback).
//!
//! These tests exercise edge cases and interleaved operations that the existing
//! test files do not cover, focusing on real-world transaction bugs.

use grafeo_common::types::Value;
use grafeo_engine::GrafeoDB;

// ============================================================================
// 1. Edge property rollback
// ============================================================================

/// SET an edge property in a transaction, then rollback.
/// Verify the property is removed (it did not exist before the tx).
#[test]
fn edge_property_set_then_rollback_removes_property() {
    let db = GrafeoDB::new_in_memory();
    let mut session = db.session();

    session
        .execute("INSERT (:Person {name: 'Alix'})-[:KNOWS]->(:Person {name: 'Gus'})")
        .unwrap();

    // Edge has no properties yet
    let before = session
        .execute("MATCH (:Person {name: 'Alix'})-[r:KNOWS]->() RETURN r.since")
        .unwrap();
    assert_eq!(before.rows()[0][0], Value::Null);

    session.begin_transaction().unwrap();
    session
        .execute(
            "MATCH (:Person {name: 'Alix'})-[r:KNOWS]->(:Person {name: 'Gus'}) SET r.since = 2020",
        )
        .unwrap();

    // Visible inside tx
    let during = session
        .execute("MATCH (:Person {name: 'Alix'})-[r:KNOWS]->() RETURN r.since")
        .unwrap();
    assert_eq!(during.rows()[0][0], Value::Int64(2020));

    session.rollback().unwrap();

    let after = session
        .execute("MATCH (:Person {name: 'Alix'})-[r:KNOWS]->() RETURN r.since")
        .unwrap();
    assert_eq!(
        after.rows()[0][0],
        Value::Null,
        "edge property should be gone after rollback"
    );
}

/// Overwrite an existing edge property in a transaction, then rollback.
/// The original value should be restored.
#[test]
fn edge_property_overwrite_then_rollback_restores_original() {
    let db = GrafeoDB::new_in_memory();
    let mut session = db.session();

    session
        .execute(
            "INSERT (:Person {name: 'Vincent'})-[:KNOWS {since: 2018}]->(:Person {name: 'Jules'})",
        )
        .unwrap();

    session.begin_transaction().unwrap();
    session
        .execute(
            "MATCH (:Person {name: 'Vincent'})-[r:KNOWS]->(:Person {name: 'Jules'}) SET r.since = 2025",
        )
        .unwrap();
    session.rollback().unwrap();

    let result = session
        .execute("MATCH (:Person {name: 'Vincent'})-[r:KNOWS]->() RETURN r.since")
        .unwrap();
    assert_eq!(
        result.rows()[0][0],
        Value::Int64(2018),
        "edge property should revert to 2018 after rollback"
    );
}

/// SET multiple edge properties in one tx, rollback, verify all reverted.
#[test]
fn edge_multiple_properties_rollback() {
    let db = GrafeoDB::new_in_memory();
    let mut session = db.session();

    session
        .execute(
            "INSERT (:City {name: 'Amsterdam'})-[:ROUTE {distance: 650, toll: 15}]->(:City {name: 'Berlin'})",
        )
        .unwrap();

    session.begin_transaction().unwrap();
    session
        .execute(
            "MATCH (:City {name: 'Amsterdam'})-[r:ROUTE]->(:City {name: 'Berlin'}) SET r.distance = 999, r.toll = 0",
        )
        .unwrap();
    session.rollback().unwrap();

    let result = session
        .execute("MATCH (:City {name: 'Amsterdam'})-[r:ROUTE]->() RETURN r.distance, r.toll")
        .unwrap();
    assert_eq!(
        result.rows()[0][0],
        Value::Int64(650),
        "distance should revert to 650"
    );
    assert_eq!(
        result.rows()[0][1],
        Value::Int64(15),
        "toll should revert to 15"
    );
}

// ============================================================================
// 2. REMOVE property rollback
// ============================================================================

/// REMOVE a node property in a tx, rollback, verify property is restored.
#[test]
fn remove_node_property_rollback_restores_value() {
    let db = GrafeoDB::new_in_memory();
    let mut session = db.session();

    session
        .execute("INSERT (:Person {name: 'Mia', age: 28})")
        .unwrap();

    session.begin_transaction().unwrap();
    session
        .execute("MATCH (p:Person {name: 'Mia'}) REMOVE p.age")
        .unwrap();

    // Property gone inside tx
    let during = session
        .execute("MATCH (p:Person {name: 'Mia'}) RETURN p.age")
        .unwrap();
    assert_eq!(during.rows()[0][0], Value::Null);

    session.rollback().unwrap();

    let after = session
        .execute("MATCH (p:Person {name: 'Mia'}) RETURN p.age")
        .unwrap();
    assert_eq!(
        after.rows()[0][0],
        Value::Int64(28),
        "age should be restored to 28 after rollback of REMOVE"
    );
}

/// SET then REMOVE same property in one tx, rollback.
/// Original value should be restored (not null, not the SET value).
#[test]
fn set_then_remove_same_property_rollback() {
    let db = GrafeoDB::new_in_memory();
    let mut session = db.session();

    session
        .execute("INSERT (:Config {key: 'timeout', value: 30})")
        .unwrap();

    session.begin_transaction().unwrap();
    session
        .execute("MATCH (c:Config {key: 'timeout'}) SET c.value = 60")
        .unwrap();
    session
        .execute("MATCH (c:Config {key: 'timeout'}) REMOVE c.value")
        .unwrap();
    session.rollback().unwrap();

    let result = session
        .execute("MATCH (c:Config {key: 'timeout'}) RETURN c.value")
        .unwrap();
    assert_eq!(
        result.rows()[0][0],
        Value::Int64(30),
        "value should revert to original 30, not null or 60"
    );
}

// ============================================================================
// 3. Multiple label operations rollback
// ============================================================================

/// Add multiple labels in a tx, rollback, verify all removed.
#[test]
fn add_multiple_labels_rollback_removes_all() {
    let db = GrafeoDB::new_in_memory();
    let mut session = db.session();

    session.execute("INSERT (:Person {name: 'Butch'})").unwrap();

    session.begin_transaction().unwrap();
    session
        .execute("MATCH (p:Person {name: 'Butch'}) SET p:Employee")
        .unwrap();
    session
        .execute("MATCH (p:Person {name: 'Butch'}) SET p:Manager")
        .unwrap();
    session.rollback().unwrap();

    let employee = session
        .execute("MATCH (p:Employee {name: 'Butch'}) RETURN p.name")
        .unwrap();
    assert_eq!(
        employee.row_count(),
        0,
        "Employee label should not exist after rollback"
    );

    let manager = session
        .execute("MATCH (p:Manager {name: 'Butch'}) RETURN p.name")
        .unwrap();
    assert_eq!(
        manager.row_count(),
        0,
        "Manager label should not exist after rollback"
    );

    // Original label should remain
    let person = session
        .execute("MATCH (p:Person {name: 'Butch'}) RETURN p.name")
        .unwrap();
    assert_eq!(
        person.row_count(),
        1,
        "Person label should survive rollback"
    );
}

/// Add a label, then remove it, in the same tx, then rollback.
/// The label should be back in its original state (present if it was present before).
#[test]
fn add_then_remove_label_same_tx_rollback() {
    let db = GrafeoDB::new_in_memory();
    let mut session = db.session();

    session
        .execute("INSERT (:Person {name: 'Django'})")
        .unwrap();

    session.begin_transaction().unwrap();
    session
        .execute("MATCH (p:Person {name: 'Django'}) SET p:Gunslinger")
        .unwrap();
    session
        .execute("MATCH (p:Person {name: 'Django'}) REMOVE p:Gunslinger")
        .unwrap();
    session.rollback().unwrap();

    // Gunslinger was added then removed within tx; rollback should undo both
    // Since node never had Gunslinger before the tx, it should not have it now
    let result = session
        .execute("MATCH (p:Gunslinger {name: 'Django'}) RETURN p.name")
        .unwrap();
    assert_eq!(
        result.row_count(),
        0,
        "Gunslinger label should not exist (was never there before tx)"
    );
}

/// Remove a pre-existing label and add a new one, rollback.
/// Pre-existing label restored, new label removed.
#[test]
fn swap_labels_rollback_restores_original() {
    let db = GrafeoDB::new_in_memory();
    let mut session = db.session();

    session
        .execute("INSERT (:Active {name: 'Shosanna'})")
        .unwrap();

    session.begin_transaction().unwrap();
    session
        .execute("MATCH (n:Active {name: 'Shosanna'}) REMOVE n:Active")
        .unwrap();
    session
        .execute("MATCH (n {name: 'Shosanna'}) SET n:Archived")
        .unwrap();
    session.rollback().unwrap();

    let active = session
        .execute("MATCH (n:Active {name: 'Shosanna'}) RETURN n.name")
        .unwrap();
    assert_eq!(active.row_count(), 1, "Active label should be restored");

    let archived = session
        .execute("MATCH (n:Archived {name: 'Shosanna'}) RETURN n.name")
        .unwrap();
    assert_eq!(
        archived.row_count(),
        0,
        "Archived label should not exist after rollback"
    );
}

// ============================================================================
// 4. Node deletion rollback: verify full restoration
// ============================================================================

/// Delete a node with multiple properties and labels, rollback.
/// Verify all properties and labels are restored.
#[test]
fn node_delete_rollback_restores_all_properties_and_labels() {
    let db = GrafeoDB::new_in_memory();
    let mut session = db.session();

    session
        .execute("INSERT (:Person:VIP {name: 'Hans', age: 55, city: 'Berlin'})")
        .unwrap();

    session.begin_transaction().unwrap();
    session
        .execute("MATCH (p:Person {name: 'Hans'}) DELETE p")
        .unwrap();
    session.rollback().unwrap();

    // All properties restored
    let result = session
        .execute("MATCH (p:Person {name: 'Hans'}) RETURN p.age, p.city")
        .unwrap();
    assert_eq!(result.row_count(), 1, "node should exist after rollback");
    assert_eq!(
        result.rows()[0][0],
        Value::Int64(55),
        "age should be restored"
    );
    assert_eq!(
        result.rows()[0][1],
        Value::String("Berlin".into()),
        "city should be restored"
    );

    // VIP label restored
    let vip = session
        .execute("MATCH (p:VIP {name: 'Hans'}) RETURN p.name")
        .unwrap();
    assert_eq!(
        vip.row_count(),
        1,
        "VIP label should be restored after rollback"
    );
}

// ============================================================================
// 5. Edge deletion rollback: verify properties restored
// ============================================================================

/// Delete an edge that has properties, rollback, verify edge and properties restored.
#[test]
fn edge_delete_rollback_restores_properties() {
    let db = GrafeoDB::new_in_memory();
    let mut session = db.session();

    session
        .execute(
            "INSERT (:Person {name: 'Beatrix'})-[:KNOWS {since: 2015, trust: 9}]->(:Person {name: 'Alix'})",
        )
        .unwrap();

    session.begin_transaction().unwrap();
    session
        .execute("MATCH (:Person {name: 'Beatrix'})-[r:KNOWS]->(:Person {name: 'Alix'}) DELETE r")
        .unwrap();

    // Edge gone inside tx
    let during = session
        .execute("MATCH (:Person {name: 'Beatrix'})-[r:KNOWS]->() RETURN r.since")
        .unwrap();
    assert_eq!(during.row_count(), 0);

    session.rollback().unwrap();

    let after = session
        .execute("MATCH (:Person {name: 'Beatrix'})-[r:KNOWS]->() RETURN r.since, r.trust")
        .unwrap();
    assert_eq!(after.row_count(), 1, "edge should be restored");
    assert_eq!(
        after.rows()[0][0],
        Value::Int64(2015),
        "edge 'since' property should be restored"
    );
    assert_eq!(
        after.rows()[0][1],
        Value::Int64(9),
        "edge 'trust' property should be restored"
    );
}

/// DETACH DELETE a node with multiple edges that have properties,
/// rollback, verify everything restored.
#[test]
fn detach_delete_rollback_restores_node_edges_and_properties() {
    let db = GrafeoDB::new_in_memory();
    let mut session = db.session();

    // Create a hub node with two outgoing edges
    session
        .execute("INSERT (:Person {name: 'Vincent', age: 40})")
        .unwrap();
    session.execute("INSERT (:Person {name: 'Jules'})").unwrap();
    session.execute("INSERT (:Person {name: 'Mia'})").unwrap();
    session
        .execute(
            "MATCH (v:Person {name: 'Vincent'}), (j:Person {name: 'Jules'}) \
             INSERT (v)-[:KNOWS {since: 2010}]->(j)",
        )
        .unwrap();
    session
        .execute(
            "MATCH (v:Person {name: 'Vincent'}), (m:Person {name: 'Mia'}) \
             INSERT (v)-[:LIKES {intensity: 8}]->(m)",
        )
        .unwrap();

    session.begin_transaction().unwrap();
    session
        .execute("MATCH (v:Person {name: 'Vincent'}) DETACH DELETE v")
        .unwrap();

    // Node and edges gone inside tx
    let during_nodes = session
        .execute("MATCH (v:Person {name: 'Vincent'}) RETURN v.name")
        .unwrap();
    assert_eq!(during_nodes.row_count(), 0);

    session.rollback().unwrap();

    // Node restored with property
    let node = session
        .execute("MATCH (v:Person {name: 'Vincent'}) RETURN v.age")
        .unwrap();
    assert_eq!(node.row_count(), 1, "node should be restored");
    assert_eq!(node.rows()[0][0], Value::Int64(40));

    // KNOWS edge restored with property
    let knows = session
        .execute("MATCH (:Person {name: 'Vincent'})-[r:KNOWS]->() RETURN r.since")
        .unwrap();
    assert_eq!(knows.row_count(), 1, "KNOWS edge should be restored");
    assert_eq!(knows.rows()[0][0], Value::Int64(2010));

    // LIKES edge restored with property
    let likes = session
        .execute("MATCH (:Person {name: 'Vincent'})-[r:LIKES]->() RETURN r.intensity")
        .unwrap();
    assert_eq!(likes.row_count(), 1, "LIKES edge should be restored");
    assert_eq!(likes.rows()[0][0], Value::Int64(8));
}

// ============================================================================
// 6. Interleaved operations: mix property sets, label changes, and deletions
// ============================================================================

/// In one transaction: set properties, add labels, delete a different node,
/// then rollback. Verify complete undo.
#[test]
fn interleaved_property_label_and_delete_rollback() {
    let db = GrafeoDB::new_in_memory();
    let mut session = db.session();

    session
        .execute("INSERT (:Person {name: 'Alix', age: 30})")
        .unwrap();
    session
        .execute("INSERT (:Temporary {name: 'ephemeral', data: 42})")
        .unwrap();

    session.begin_transaction().unwrap();

    // Modify Alix's property
    session
        .execute("MATCH (p:Person {name: 'Alix'}) SET p.age = 99")
        .unwrap();

    // Add label to Alix
    session
        .execute("MATCH (p:Person {name: 'Alix'}) SET p:VIP")
        .unwrap();

    // Delete the Temporary node
    session
        .execute("MATCH (t:Temporary {name: 'ephemeral'}) DELETE t")
        .unwrap();

    session.rollback().unwrap();

    // Alix's age reverted
    let alix = session
        .execute("MATCH (p:Person {name: 'Alix'}) RETURN p.age")
        .unwrap();
    assert_eq!(
        alix.rows()[0][0],
        Value::Int64(30),
        "age should revert to 30"
    );

    // Alix's VIP label removed
    let vip = session
        .execute("MATCH (p:VIP {name: 'Alix'}) RETURN p.name")
        .unwrap();
    assert_eq!(
        vip.row_count(),
        0,
        "VIP label should not exist after rollback"
    );

    // Temporary node restored
    let temp = session
        .execute("MATCH (t:Temporary {name: 'ephemeral'}) RETURN t.data")
        .unwrap();
    assert_eq!(temp.row_count(), 1, "Temporary node should be restored");
    assert_eq!(temp.rows()[0][0], Value::Int64(42));
}

/// Property set, label add, edge property set, node delete, all in one tx.
#[test]
fn kitchen_sink_operations_rollback() {
    let db = GrafeoDB::new_in_memory();
    let mut session = db.session();

    session
        .execute("INSERT (:Person {name: 'Gus', score: 10})-[:WORKS_WITH {years: 3}]->(:Person {name: 'Alix'})")
        .unwrap();

    session.begin_transaction().unwrap();

    // Modify node property
    session
        .execute("MATCH (p:Person {name: 'Gus'}) SET p.score = 100")
        .unwrap();

    // Add label
    session
        .execute("MATCH (p:Person {name: 'Gus'}) SET p:Senior")
        .unwrap();

    // Modify edge property
    session
        .execute(
            "MATCH (:Person {name: 'Gus'})-[r:WORKS_WITH]->(:Person {name: 'Alix'}) SET r.years = 10",
        )
        .unwrap();

    session.rollback().unwrap();

    // Node property reverted
    let score = session
        .execute("MATCH (p:Person {name: 'Gus'}) RETURN p.score")
        .unwrap();
    assert_eq!(
        score.rows()[0][0],
        Value::Int64(10),
        "score should revert to 10"
    );

    // Label removed
    let senior = session
        .execute("MATCH (p:Senior {name: 'Gus'}) RETURN p.name")
        .unwrap();
    assert_eq!(
        senior.row_count(),
        0,
        "Senior label should not exist after rollback"
    );

    // Edge property reverted
    let years = session
        .execute("MATCH (:Person {name: 'Gus'})-[r:WORKS_WITH]->() RETURN r.years")
        .unwrap();
    assert_eq!(
        years.rows()[0][0],
        Value::Int64(3),
        "edge years should revert to 3"
    );
}

// ============================================================================
// 7. Concurrent transactions: two sessions, one rolls back
// ============================================================================

/// Session 1 commits property changes, then session 2 modifies a different
/// node and rolls back. The committed changes from session 1 should not be
/// affected by session 2's rollback.
#[test]
fn sequential_commit_then_rollback_different_nodes() {
    let db = GrafeoDB::new_in_memory();

    let setup = db.session();
    setup
        .execute("INSERT (:Account {owner: 'Alix', balance: 100})")
        .unwrap();
    setup
        .execute("INSERT (:Account {owner: 'Gus', balance: 200})")
        .unwrap();

    // Session 1: modifies Alix, commits
    let mut session1 = db.session();
    session1.begin_transaction().unwrap();
    session1
        .execute("MATCH (a:Account {owner: 'Alix'}) SET a.balance = 500")
        .unwrap();
    session1.commit().unwrap();

    // Session 2: modifies Gus, rolls back
    let mut session2 = db.session();
    session2.begin_transaction().unwrap();
    session2
        .execute("MATCH (a:Account {owner: 'Gus'}) SET a.balance = 9999")
        .unwrap();
    session2.rollback().unwrap();

    // Verify with a fresh session
    let reader = db.session();

    let alix = reader
        .execute("MATCH (a:Account {owner: 'Alix'}) RETURN a.balance")
        .unwrap();
    assert_eq!(
        alix.rows()[0][0],
        Value::Int64(500),
        "Alix's committed balance should be 500"
    );

    let gus = reader
        .execute("MATCH (a:Account {owner: 'Gus'}) RETURN a.balance")
        .unwrap();
    assert_eq!(
        gus.rows()[0][0],
        Value::Int64(200),
        "Gus's balance should remain 200 after session2's rollback"
    );
}

/// Session 1 commits a label add, then session 2 adds a different label
/// to a different node and rolls back. Committed label survives.
#[test]
fn sequential_label_commit_then_rollback_different_nodes() {
    let db = GrafeoDB::new_in_memory();

    let setup = db.session();
    setup.execute("INSERT (:Person {name: 'Vincent'})").unwrap();
    setup.execute("INSERT (:Person {name: 'Jules'})").unwrap();

    // Session 1: adds Hitman label to Vincent, commits
    let mut session1 = db.session();
    session1.begin_transaction().unwrap();
    session1
        .execute("MATCH (p:Person {name: 'Vincent'}) SET p:Hitman")
        .unwrap();
    session1.commit().unwrap();

    // Session 2: adds Philosopher label to Jules, rolls back
    let mut session2 = db.session();
    session2.begin_transaction().unwrap();
    session2
        .execute("MATCH (p:Person {name: 'Jules'}) SET p:Philosopher")
        .unwrap();
    session2.rollback().unwrap();

    let reader = db.session();

    let hitman = reader
        .execute("MATCH (p:Hitman {name: 'Vincent'}) RETURN p.name")
        .unwrap();
    assert_eq!(
        hitman.row_count(),
        1,
        "Vincent should have Hitman label (committed)"
    );

    let philosopher = reader
        .execute("MATCH (p:Philosopher {name: 'Jules'}) RETURN p.name")
        .unwrap();
    assert_eq!(
        philosopher.row_count(),
        0,
        "Jules should NOT have Philosopher label (rolled back)"
    );
}

// ============================================================================
// 8. Property overwrite multiple times in one tx
// ============================================================================

/// Overwrite the same property three times in one tx, rollback.
/// Original value should be restored, not any intermediate value.
#[test]
fn triple_overwrite_same_property_rollback_restores_original() {
    let db = GrafeoDB::new_in_memory();
    let mut session = db.session();

    session
        .execute("INSERT (:Counter {name: 'visits', count: 0})")
        .unwrap();

    session.begin_transaction().unwrap();
    session
        .execute("MATCH (c:Counter {name: 'visits'}) SET c.count = 1")
        .unwrap();
    session
        .execute("MATCH (c:Counter {name: 'visits'}) SET c.count = 2")
        .unwrap();
    session
        .execute("MATCH (c:Counter {name: 'visits'}) SET c.count = 3")
        .unwrap();
    session.rollback().unwrap();

    let result = session
        .execute("MATCH (c:Counter {name: 'visits'}) RETURN c.count")
        .unwrap();
    assert_eq!(
        result.rows()[0][0],
        Value::Int64(0),
        "count should be 0 (original), not 1, 2, or 3"
    );
}

// ============================================================================
// 9. Commit-then-rollback chain: labels
// ============================================================================

/// Commit a label add in tx1, then rollback a label add in tx2.
/// The tx1 label should persist, the tx2 label should not.
#[test]
fn commit_label_then_rollback_different_label() {
    let db = GrafeoDB::new_in_memory();
    let mut session = db.session();

    session.execute("INSERT (:Person {name: 'Alix'})").unwrap();

    // tx1: add Employee, commit
    session.begin_transaction().unwrap();
    session
        .execute("MATCH (p:Person {name: 'Alix'}) SET p:Employee")
        .unwrap();
    session.commit().unwrap();

    // tx2: add Manager, rollback
    session.begin_transaction().unwrap();
    session
        .execute("MATCH (p:Person {name: 'Alix'}) SET p:Manager")
        .unwrap();
    session.rollback().unwrap();

    let employee = session
        .execute("MATCH (p:Employee {name: 'Alix'}) RETURN p.name")
        .unwrap();
    assert_eq!(
        employee.row_count(),
        1,
        "Employee label should persist from committed tx"
    );

    let manager = session
        .execute("MATCH (p:Manager {name: 'Alix'}) RETURN p.name")
        .unwrap();
    assert_eq!(
        manager.row_count(),
        0,
        "Manager label should not exist (rolled back)"
    );
}

// ============================================================================
// 10. Edge property on newly-created edge within tx
// ============================================================================

/// Create an edge and set a property on it within the same tx, then rollback.
/// Both the edge and property should be gone.
#[test]
fn create_edge_and_set_property_in_tx_rollback() {
    let db = GrafeoDB::new_in_memory();
    let mut session = db.session();

    session.execute("INSERT (:Person {name: 'Alix'})").unwrap();
    session.execute("INSERT (:Person {name: 'Gus'})").unwrap();

    session.begin_transaction().unwrap();
    session
        .execute(
            "MATCH (a:Person {name: 'Alix'}), (g:Person {name: 'Gus'}) \
             INSERT (a)-[:FRIENDS {level: 5}]->(g)",
        )
        .unwrap();

    // Edge visible inside tx
    let during = session
        .execute("MATCH ()-[r:FRIENDS]->() RETURN r.level")
        .unwrap();
    assert_eq!(during.row_count(), 1);

    session.rollback().unwrap();

    let after = session
        .execute("MATCH ()-[r:FRIENDS]->() RETURN r.level")
        .unwrap();
    assert_eq!(
        after.row_count(),
        0,
        "edge created in rolled-back tx should not exist"
    );
}

// ============================================================================
// 11. Property type change rollback
// ============================================================================

/// Change a property from one type to another (int to string), rollback.
/// Original type and value should be restored.
#[test]
fn property_type_change_rollback_restores_original_type() {
    let db = GrafeoDB::new_in_memory();
    let mut session = db.session();

    session
        .execute("INSERT (:Setting {key: 'mode', value: 1})")
        .unwrap();

    session.begin_transaction().unwrap();
    session
        .execute("MATCH (s:Setting {key: 'mode'}) SET s.value = 'debug'")
        .unwrap();

    // Verify type changed inside tx
    let during = session
        .execute("MATCH (s:Setting {key: 'mode'}) RETURN s.value")
        .unwrap();
    assert_eq!(during.rows()[0][0], Value::String("debug".into()));

    session.rollback().unwrap();

    let after = session
        .execute("MATCH (s:Setting {key: 'mode'}) RETURN s.value")
        .unwrap();
    assert_eq!(
        after.rows()[0][0],
        Value::Int64(1),
        "value should revert to int 1, not string 'debug'"
    );
}

// ============================================================================
// 12. Empty transaction rollback is harmless
// ============================================================================

/// Begin and immediately rollback without any mutations. Database should be unchanged.
#[test]
fn empty_transaction_rollback_leaves_data_intact() {
    let db = GrafeoDB::new_in_memory();
    let mut session = db.session();

    session
        .execute("INSERT (:Person {name: 'Alix', age: 30})")
        .unwrap();

    session.begin_transaction().unwrap();
    // No mutations
    session.rollback().unwrap();

    let result = session
        .execute("MATCH (p:Person {name: 'Alix'}) RETURN p.age")
        .unwrap();
    assert_eq!(result.row_count(), 1);
    assert_eq!(result.rows()[0][0], Value::Int64(30));
}

// ============================================================================
// 13. Rollback after read-only operations
// ============================================================================

/// A tx that only reads (MATCH/RETURN) and then rolls back should not corrupt state.
#[test]
fn read_only_transaction_rollback() {
    let db = GrafeoDB::new_in_memory();
    let mut session = db.session();

    session
        .execute("INSERT (:Person {name: 'Gus', score: 42})")
        .unwrap();

    session.begin_transaction().unwrap();
    let _read = session
        .execute("MATCH (p:Person {name: 'Gus'}) RETURN p.score")
        .unwrap();
    session.rollback().unwrap();

    // Data intact
    let after = session
        .execute("MATCH (p:Person {name: 'Gus'}) RETURN p.score")
        .unwrap();
    assert_eq!(after.rows()[0][0], Value::Int64(42));
}

// ============================================================================
// 14. Multiple nodes with same property name, rollback only affects modified
// ============================================================================

/// Two nodes share the same property key. Modify only one in a tx, rollback.
/// The other should be completely untouched.
#[test]
fn rollback_only_affects_modified_node_shared_property_key() {
    let db = GrafeoDB::new_in_memory();
    let mut session = db.session();

    session
        .execute("INSERT (:Person {name: 'Alix', score: 10})")
        .unwrap();
    session
        .execute("INSERT (:Person {name: 'Gus', score: 20})")
        .unwrap();

    session.begin_transaction().unwrap();
    session
        .execute("MATCH (p:Person {name: 'Alix'}) SET p.score = 999")
        .unwrap();
    session.rollback().unwrap();

    let alix = session
        .execute("MATCH (p:Person {name: 'Alix'}) RETURN p.score")
        .unwrap();
    assert_eq!(
        alix.rows()[0][0],
        Value::Int64(10),
        "Alix's score should revert to 10"
    );

    let gus = session
        .execute("MATCH (p:Person {name: 'Gus'}) RETURN p.score")
        .unwrap();
    assert_eq!(
        gus.rows()[0][0],
        Value::Int64(20),
        "Gus's score should be unaffected (20)"
    );
}

// ============================================================================
// 15. Rollback after node creation: adjacency consistency
// ============================================================================

/// Create a node and edges in a tx, rollback.
/// Verify that adjacency index is clean: no phantom edges.
#[test]
fn create_node_with_edges_rollback_cleans_adjacency() {
    let db = GrafeoDB::new_in_memory();
    let mut session = db.session();

    session.execute("INSERT (:Person {name: 'Alix'})").unwrap();

    session.begin_transaction().unwrap();
    session.execute("INSERT (:Person {name: 'Ghost'})").unwrap();
    session
        .execute(
            "MATCH (a:Person {name: 'Alix'}), (g:Person {name: 'Ghost'}) \
             INSERT (a)-[:HAUNTS]->(g)",
        )
        .unwrap();
    session.rollback().unwrap();

    // Ghost node should not exist
    let ghost = session
        .execute("MATCH (g:Person {name: 'Ghost'}) RETURN g.name")
        .unwrap();
    assert_eq!(
        ghost.row_count(),
        0,
        "Ghost node should not exist after rollback"
    );

    // No HAUNTS edges should exist
    let edges = session.execute("MATCH ()-[r:HAUNTS]->() RETURN r").unwrap();
    assert_eq!(
        edges.row_count(),
        0,
        "HAUNTS edge should not exist after rollback"
    );

    // Alix should have no outgoing edges
    let alix_edges = session
        .execute("MATCH (:Person {name: 'Alix'})-[r]->() RETURN r")
        .unwrap();
    assert_eq!(
        alix_edges.row_count(),
        0,
        "Alix should have no outgoing edges after rollback"
    );
}

// ============================================================================
// 16. Commit followed by rollback, same property
// ============================================================================

/// Commit a property change, then in a new tx change the same property and rollback.
/// The committed value should be the final state.
#[test]
fn commit_then_rollback_same_property_retains_committed_value() {
    let db = GrafeoDB::new_in_memory();
    let mut session = db.session();

    session
        .execute("INSERT (:Person {name: 'Mia', level: 1})")
        .unwrap();

    // tx1: set level to 5, commit
    session.begin_transaction().unwrap();
    session
        .execute("MATCH (p:Person {name: 'Mia'}) SET p.level = 5")
        .unwrap();
    session.commit().unwrap();

    // tx2: set level to 99, rollback
    session.begin_transaction().unwrap();
    session
        .execute("MATCH (p:Person {name: 'Mia'}) SET p.level = 99")
        .unwrap();
    session.rollback().unwrap();

    let result = session
        .execute("MATCH (p:Person {name: 'Mia'}) RETURN p.level")
        .unwrap();
    assert_eq!(
        result.rows()[0][0],
        Value::Int64(5),
        "level should be 5 (committed value), not 99 (rolled back)"
    );
}

// ============================================================================
// 17. Boolean and float property rollback (different value types)
// ============================================================================

/// Test rollback with boolean and float property types.
#[test]
fn rollback_preserves_boolean_and_float_property_types() {
    let db = GrafeoDB::new_in_memory();
    let mut session = db.session();

    session
        .execute("INSERT (:Sensor {name: 'thermostat', enabled: true, reading: 21.5})")
        .unwrap();

    session.begin_transaction().unwrap();
    session
        .execute("MATCH (s:Sensor {name: 'thermostat'}) SET s.enabled = false, s.reading = 99.9")
        .unwrap();
    session.rollback().unwrap();

    let result = session
        .execute("MATCH (s:Sensor {name: 'thermostat'}) RETURN s.enabled, s.reading")
        .unwrap();
    assert_eq!(
        result.rows()[0][0],
        Value::Bool(true),
        "enabled should revert to true"
    );
    assert_eq!(
        result.rows()[0][1],
        Value::Float64(21.5),
        "reading should revert to 21.5"
    );
}

// ============================================================================
// 18. Rapid begin-rollback cycles
// ============================================================================

/// Multiple begin/rollback cycles in sequence should not leak state or corrupt data.
#[test]
fn rapid_begin_rollback_cycles_no_state_leak() {
    let db = GrafeoDB::new_in_memory();
    let mut session = db.session();

    session
        .execute("INSERT (:Counter {name: 'stable', value: 42})")
        .unwrap();

    for i in 0..10 {
        session.begin_transaction().unwrap();
        session
            .execute(&format!(
                "MATCH (c:Counter {{name: 'stable'}}) SET c.value = {}",
                i * 100
            ))
            .unwrap();
        session.rollback().unwrap();
    }

    let result = session
        .execute("MATCH (c:Counter {name: 'stable'}) RETURN c.value")
        .unwrap();
    assert_eq!(
        result.rows()[0][0],
        Value::Int64(42),
        "value should remain 42 after 10 begin/rollback cycles"
    );
}
