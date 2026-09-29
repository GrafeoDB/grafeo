//! Every write goes through the same checks: schema constraints and
//! write-conflict detection hold for MERGE and for entities the transaction
//! itself created, as they do for INSERT and SET.

use grafeo_engine::GrafeoDB;

/// MERGE did not record its writes, so two transactions could both update
/// the same node through `ON MATCH SET` without a write-write conflict.
#[test]
fn merge_writes_are_checked_for_conflicts() {
    let db = GrafeoDB::new_in_memory();
    db.execute("INSERT (:Person {name: 'Alix', visits: 0})")
        .unwrap();

    let mut first = db.session();
    let mut second = db.session();
    first.begin_transaction().unwrap();
    second.begin_transaction().unwrap();
    first
        .execute("MERGE (n:Person {name: 'Alix'}) ON MATCH SET n.visits = 1")
        .unwrap();
    let err = second
        .execute("MERGE (n:Person {name: 'Alix'}) ON MATCH SET n.visits = 2")
        .unwrap_err()
        .to_string();
    assert!(err.to_lowercase().contains("conflict"), "got: {err}");
}

/// A SET on a node created earlier in the same transaction skipped the
/// constraint checks: the check looked the node up outside the transaction
/// and did not find it.
#[test]
fn set_on_a_node_created_in_the_transaction_is_checked() {
    let db = GrafeoDB::new_in_memory();
    db.execute("CREATE CONSTRAINT person_name FOR (n:Person) ON (n.name) NOT NULL")
        .unwrap();

    let mut session = db.session();
    session.begin_transaction().unwrap();
    session.execute("INSERT (:Person {name: 'Alix'})").unwrap();
    let err = session
        .execute("MATCH (n:Person) SET n.name = NULL")
        .unwrap_err()
        .to_string();
    assert!(err.contains("NOT NULL"), "got: {err}");
}

/// The same holds for a label added to a node the transaction created.
#[test]
fn label_on_a_node_created_in_the_transaction_is_checked() {
    let db = GrafeoDB::new_in_memory();
    db.execute("CREATE CONSTRAINT person_email FOR (n:Person) ON (n.email) UNIQUE")
        .unwrap();
    db.execute("INSERT (:Person {name: 'Alix', email: 'alix@example.org'})")
        .unwrap();

    let mut session = db.session();
    session.begin_transaction().unwrap();
    session
        .execute("INSERT (:Guest {name: 'Gus', email: 'alix@example.org'})")
        .unwrap();
    let err = session
        .execute("MATCH (n:Guest) SET n:Person")
        .unwrap_err()
        .to_string();
    assert!(err.to_lowercase().contains("constraint"), "got: {err}");
}
