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

// ── A failed statement inside a transaction ──────────────────────

/// A database whose `Doc` nodes must have an integer `id` and a string `tag`.
fn typed_docs() -> GrafeoDB {
    let db = GrafeoDB::new_in_memory();
    db.execute("CREATE NODE TYPE Doc (id INTEGER, tag STRING)")
        .unwrap();
    db
}

/// The sorted `id`s of every `Doc`.
fn doc_ids(db: &GrafeoDB) -> Vec<grafeo_common::types::Value> {
    let rows = db
        .execute("MATCH (d:Doc) RETURN d.id ORDER BY d.id")
        .unwrap();
    rows.rows().iter().map(|row| row[0].clone()).collect()
}

fn ids(values: &[i64]) -> Vec<grafeo_common::types::Value> {
    values
        .iter()
        .map(|v| grafeo_common::types::Value::Int64(*v))
        .collect()
}

/// A statement that fails on its second row is undone completely, and the
/// transaction goes on: its other statements still commit.
#[test]
fn a_failed_statement_in_a_transaction_leaves_nothing() {
    let db = typed_docs();
    let mut session = db.session();
    session.begin_transaction().unwrap();
    session.execute("INSERT (:Doc {id: 1})").unwrap();
    session
        .execute("UNWIND [5, 'x'] AS v INSERT (:Doc {id: v})")
        .unwrap_err();
    session.execute("INSERT (:Doc {id: 2})").unwrap();
    session.commit().unwrap();

    assert_eq!(doc_ids(&db), ids(&[1, 2]));
}

/// A MERGE whose `ON CREATE SET` fails does not leave the node it created.
#[test]
fn a_failed_merge_in_a_transaction_leaves_nothing() {
    let db = typed_docs();
    let mut session = db.session();
    session.begin_transaction().unwrap();
    session
        .execute("MERGE (d:Doc {id: 1}) ON CREATE SET d.tag = d.id + 1")
        .unwrap_err();
    session.commit().unwrap();

    assert_eq!(doc_ids(&db), ids(&[]));
}

/// A direct batch that fails on a later row inside a transaction leaves none
/// of its rows.
#[test]
fn a_failed_direct_batch_in_a_transaction_leaves_nothing() {
    use grafeo_common::types::{PropertyKey, Value};
    use std::collections::HashMap;

    let db = typed_docs();
    let mut session = db.session();
    session.begin_transaction().unwrap();
    session
        .batch_create_nodes_with_props(
            "Doc",
            vec![
                HashMap::from([(PropertyKey::new("id"), Value::Int64(3))]),
                HashMap::from([(PropertyKey::new("id"), Value::from("x"))]),
            ],
        )
        .unwrap_err();
    session.commit().unwrap();

    assert_eq!(doc_ids(&db), ids(&[]));
}

/// The undone statement's WAL records are dropped too: a reopen replays only
/// what committed.
#[cfg(feature = "wal")]
#[test]
fn a_failed_statement_is_not_replayed() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("docs");
    {
        let db = GrafeoDB::open(&path).unwrap();
        db.execute("CREATE NODE TYPE Doc (id INTEGER, tag STRING)")
            .unwrap();
        let mut session = db.session();
        session.begin_transaction().unwrap();
        session.execute("INSERT (:Doc {id: 1})").unwrap();
        session
            .execute("UNWIND [5, 'x'] AS v INSERT (:Doc {id: v})")
            .unwrap_err();
        session.commit().unwrap();
        db.close().unwrap();
    }

    assert_eq!(doc_ids(&GrafeoDB::open(&path).unwrap()), ids(&[1]));
}
