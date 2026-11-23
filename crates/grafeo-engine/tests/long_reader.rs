//! #673 part A: a transaction left open (a long reader) changes neither what conflicts nor what
//! the transaction manager keeps. After 10,000 commits behind an open transaction, a write it
//! makes to what one of them changed fails at its commit with `GRAFEO-T001`, a write to
//! anything else commits, and a finished transaction leaves the manager whatever else is open.
//!
//! What a commit costs with readers open is measured by the Callgrind gate (mode
//! `update_with_reader`), not here.
//!
//! ```bash
//! cargo test -p grafeo-engine --all-features --test main long_reader::
//! ```

#![cfg(feature = "gql")]

use std::collections::HashMap;

use grafeo_common::types::Value;
use grafeo_common::utils::error::{Error, ErrorCode};
use grafeo_engine::GrafeoDB;
use grafeo_engine::transaction::{TransactionManager, TransactionState};

/// Commits behind the reader.
const COMMITS: i64 = 10_000;

fn people() -> GrafeoDB {
    let db = GrafeoDB::new_in_memory();
    db.execute(
        "INSERT (:Person {name: 'Alix', bio: 'Amsterdam'}), (:Person {name: 'Gus', bio: 'Berlin'}), \
         (:Person {name: 'Vincent', n: 0}), (:Person {name: 'Mia'}), (:Person {name: 'Butch'})",
    )
    .unwrap();
    db
}

/// `COMMITS` auto-commit updates of `name`'s `n`, each a commit of its own.
fn updates(db: &GrafeoDB, name: &str) {
    for i in 0..COMMITS {
        db.execute_with_params(
            "MATCH (p:Person) WHERE p.name = $name SET p.n = $i",
            HashMap::from([
                ("name".to_string(), Value::from(name)),
                ("i".to_string(), Value::Int64(i)),
            ]),
        )
        .unwrap();
    }
}

fn is_conflict(error: &Error) -> bool {
    error.error_code() == ErrorCode::TransactionConflict
}

fn bio(db: &GrafeoDB, name: &str) -> Value {
    db.execute(&format!(
        "MATCH (p:Person) WHERE p.name = '{name}' RETURN p.bio AS b"
    ))
    .unwrap()
    .rows()[0][0]
        .clone()
}

/// A reader that read Alix, then 10,000 updates of Alix by others: the reader's own write of
/// Alix fails at its commit, and Alix keeps the committed value.
#[test]
fn a_write_after_10000_commits_of_the_same_node_conflicts() {
    let db = people();
    let mut reader = db.session();
    reader.begin_transaction().unwrap();
    reader
        .execute("MATCH (p:Person {name: 'Alix'}) RETURN p.bio")
        .unwrap();
    updates(&db, "Alix");
    reader
        .execute("MATCH (p:Person {name: 'Alix'}) SET p.bio = 'Utrecht'")
        .unwrap();
    let error = reader
        .commit()
        .expect_err("Alix was written after it began");
    assert!(is_conflict(&error), "{error}");
    assert_eq!(bio(&db, "Alix"), Value::from("Amsterdam"));
}

/// 10,000 updates of Gus behind the reader do not touch Alix: its write of Alix commits.
#[test]
fn a_write_after_10000_commits_of_other_nodes_commits() {
    let db = people();
    let mut reader = db.session();
    reader.begin_transaction().unwrap();
    reader
        .execute("MATCH (p:Person {name: 'Alix'}) RETURN p.bio")
        .unwrap();
    updates(&db, "Gus");
    reader
        .execute("MATCH (p:Person {name: 'Alix'}) SET p.bio = 'Utrecht'")
        .unwrap();
    reader.commit().unwrap();
    assert_eq!(bio(&db, "Alix"), Value::from("Utrecht"));
}

/// An edge to a node deleted after the reader began, and a delete of a node another commit
/// linked to after it began, each behind 10,000 unrelated commits: both commits fail.
#[test]
fn endpoint_and_delete_conflicts_hold_behind_10000_commits() {
    let db = people();
    let mut linker = db.session();
    linker.begin_transaction().unwrap();
    linker.execute("MATCH (p:Person) RETURN count(p)").unwrap();
    let mut deleter = db.session();
    deleter.begin_transaction().unwrap();
    deleter.execute("MATCH (p:Person) RETURN count(p)").unwrap();

    updates(&db, "Vincent");
    db.execute("MATCH (m:Person {name: 'Mia'}) DELETE m")
        .unwrap();
    db.execute(
        "MATCH (v:Person {name: 'Vincent'}), (b:Person {name: 'Butch'}) INSERT (v)-[:KNOWS]->(b)",
    )
    .unwrap();

    linker
        .execute(
            "MATCH (a:Person {name: 'Alix'}), (m:Person {name: 'Mia'}) INSERT (a)-[:KNOWS]->(m)",
        )
        .unwrap();
    let error = linker.commit().expect_err("Mia was deleted after it began");
    assert!(is_conflict(&error), "{error}");

    // The delete fails, at its statement (the edge is there now) or at its commit: no
    // committed edge ends at a deleted node.
    let outcome = deleter
        .execute("MATCH (b:Person {name: 'Butch'}) DELETE b")
        .and_then(|_| deleter.commit());
    assert!(
        outcome.is_err(),
        "an edge to Butch was committed after it began"
    );
    let butch = db
        .execute("MATCH (b:Person {name: 'Butch'}) RETURN count(b) AS n")
        .unwrap();
    assert_eq!(butch.rows()[0][0], Value::Int64(1), "Butch stays");

    let edges = db
        .execute("MATCH (:Person)-[k:KNOWS]->(:Person) RETURN count(k) AS n")
        .unwrap();
    assert_eq!(edges.rows()[0][0], Value::Int64(1), "only Vincent's edge");
}

/// A unique key taken by a commit behind 10,000 others cannot be taken again by a transaction
/// that began before it: its insert or its commit fails with a write conflict, and one node
/// holds the key.
#[test]
fn a_key_taken_behind_10000_commits_is_taken() {
    let db = people();
    db.execute("CREATE CONSTRAINT person_email FOR (p:Person) ON (p.email) UNIQUE")
        .unwrap();
    let mut reader = db.session();
    reader.begin_transaction().unwrap();
    reader.execute("MATCH (p:Person) RETURN count(p)").unwrap();
    updates(&db, "Vincent");
    db.execute("INSERT (:Person {name: 'Jules', email: 'jules@example.org'})")
        .unwrap();
    let outcome = reader
        .execute("INSERT (:Person {name: 'Hans', email: 'jules@example.org'})")
        .and_then(|_| reader.commit());
    let error = outcome.expect_err("the key was taken after it began");
    assert!(is_conflict(&error), "{error}");
    let holders = db
        .execute("MATCH (p:Person {email: 'jules@example.org'}) RETURN p.name AS n")
        .unwrap();
    assert_eq!(holders.rows(), [vec![Value::from("Jules")]]);
}

/// A3 through the manager's public calls: a transaction that commits or aborts leaves it at
/// once, also while another stays open, so `state` knows nothing of it and `gc` finds nothing.
#[test]
fn a_finished_transaction_leaves_the_manager_while_another_is_open() {
    let manager = TransactionManager::new();
    let reader = manager.begin();
    let committed = manager.begin();
    manager
        .record_write(committed, grafeo_common::types::NodeId::new(1))
        .unwrap();
    manager.commit(committed).unwrap();
    let aborted = manager.begin();
    manager.abort(aborted).unwrap();

    assert_eq!(manager.state(committed), None, "published");
    assert_eq!(manager.state(aborted), None, "aborted");
    assert_eq!(manager.state(reader), Some(TransactionState::Active));
    assert_eq!(manager.gc(), 0, "nothing left to drop");
    manager.commit(reader).unwrap();
    assert_eq!(manager.state(reader), None);
}
