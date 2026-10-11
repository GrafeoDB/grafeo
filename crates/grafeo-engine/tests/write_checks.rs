//! Every write goes through the same checks: schema constraints and
//! write-conflict detection hold for MERGE and for entities the transaction
//! itself created, as they do for INSERT and SET.

use grafeo_engine::GrafeoDB;

#[cfg(all(feature = "wal", feature = "grafeo-file"))]
mod common;

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

/// A MERGE whose `ON CREATE SET` fails on the new edge does not leave the
/// edge it created.
#[test]
fn a_failed_edge_merge_in_a_transaction_leaves_nothing() {
    let db = typed_docs();
    db.execute("CREATE EDGE TYPE CITES (since INTEGER, note STRING)")
        .unwrap();
    db.execute("INSERT (:Doc {id: 1}), (:Doc {id: 2})").unwrap();
    let mut session = db.session();
    session.begin_transaction().unwrap();
    session
        .execute(
            "MATCH (a:Doc {id: 1}), (b:Doc {id: 2}) \
             MERGE (a)-[r:CITES {since: 2020}]->(b) ON CREATE SET r.note = r.since + 1",
        )
        .unwrap_err();
    session.commit().unwrap();

    let edges = db
        .execute("MATCH ()-[r:CITES]->() RETURN count(r)")
        .unwrap();
    assert_eq!(edges.rows()[0][0], grafeo_common::types::Value::Int64(0));
    assert_eq!(doc_ids(&db), ids(&[1, 2]));
}

/// The undone statement's WAL records are dropped too: a reopen replays only
/// what committed. The writes run in a child process that exits without
/// `close()`, so nothing is checkpointed and the reopen replays the WAL.
#[cfg(all(feature = "wal", feature = "grafeo-file"))]
#[test]
fn a_failed_statement_is_not_replayed() {
    let (_dir, db) = common::replay::reopened_after_crash(
        "a_failed_statement_is_not_replayed",
        |path| GrafeoDB::open(path).unwrap(),
        |db| {
            db.execute("CREATE NODE TYPE Doc (id INTEGER, tag STRING)")
                .unwrap();
            let mut session = db.session();
            session.begin_transaction().unwrap();
            session.execute("INSERT (:Doc {id: 1})").unwrap();
            session
                .execute("UNWIND [5, 'x'] AS v INSERT (:Doc {id: v})")
                .unwrap_err();
            session.commit().unwrap();
        },
    );

    assert_eq!(doc_ids(&db), ids(&[1]));
}

// ── A failed statement with auto-commit off, no transaction open (#536) ──

/// The auto-commit setting (deprecated) no longer changes how writes run:
/// the same writes, with the setting on and off, succeed and fail alike,
/// leave the same graph, and commit the same number of times (each statement
/// and call on its own, a failed one not at all), and no transaction stays
/// open.
#[test]
#[expect(deprecated, reason = "the deprecated setting is what this tests")]
fn the_auto_commit_setting_no_longer_changes_how_writes_run() {
    use grafeo_common::types::{PropertyKey, Value};
    use std::collections::HashMap;

    let outcome = |auto_commit: bool| {
        let db = typed_docs();
        let mut session = db.session();
        session.set_auto_commit(auto_commit);
        let doc = |id: Value| HashMap::from([(PropertyKey::new("id"), id)]);
        let succeeded = [
            session.execute("INSERT (:Doc {id: 1})").is_ok(),
            session
                .execute("UNWIND [5, 'x'] AS v INSERT (:Doc {id: v})")
                .is_ok(),
            session
                .batch_create_nodes_with_props(
                    "Doc",
                    vec![doc(Value::Int64(3)), doc(Value::from("x"))],
                )
                .is_ok(),
            session
                .create_node_with_props(&["Doc"], [("id", Value::Int64(19))])
                .is_ok(),
            session
                .execute("MATCH (d:Doc {id: 1}) SET d.tag = 'Paris'")
                .is_ok(),
        ];
        assert!(!session.in_transaction(), "auto-commit {auto_commit}");
        (succeeded, doc_ids(&db), db.current_epoch())
    };
    let on = outcome(true);
    assert_eq!(on.0, [true, false, false, true, true]);
    assert_eq!(on.1, ids(&[1, 19]));
    assert_eq!(
        outcome(false),
        on,
        "auto-commit off runs the writes as on does"
    );
}

/// With auto-commit off and no transaction open, each write statement is a
/// transaction of its own: one that fails on its second row leaves nothing,
/// and the statements around it are committed, each on its own.
#[test]
#[expect(
    deprecated,
    reason = "auto-commit off is what #536 is about: the setting no longer changes how writes run"
)]
fn a_failed_statement_with_auto_commit_off_leaves_nothing() {
    let db = typed_docs();
    let mut session = db.session();
    session.set_auto_commit(false);
    session.execute("INSERT (:Doc {id: 1})").unwrap();
    session
        .execute("UNWIND [5, 'x'] AS v INSERT (:Doc {id: v})")
        .unwrap_err();
    session.execute("INSERT (:Doc {id: 2})").unwrap();

    assert!(!session.in_transaction(), "no transaction stays open");
    assert_eq!(
        doc_ids(&db),
        ids(&[1, 2]),
        "another session sees both statements that succeeded, without a commit"
    );
}

/// A MERGE whose `ON CREATE SET` fails does not leave the node it created.
#[test]
#[expect(
    deprecated,
    reason = "auto-commit off is what #536 is about: the setting no longer changes how writes run"
)]
fn a_failed_merge_with_auto_commit_off_leaves_nothing() {
    let db = typed_docs();
    let mut session = db.session();
    session.set_auto_commit(false);
    session
        .execute("MERGE (d:Doc {id: 1}) ON CREATE SET d.tag = d.id + 1")
        .unwrap_err();

    assert_eq!(doc_ids(&db), ids(&[]));
}

/// A MERGE whose `ON CREATE SET` fails on the new edge does not leave the
/// edge it created.
#[test]
#[expect(
    deprecated,
    reason = "auto-commit off is what #536 is about: the setting no longer changes how writes run"
)]
fn a_failed_edge_merge_with_auto_commit_off_leaves_nothing() {
    let db = typed_docs();
    db.execute("CREATE EDGE TYPE CITES (since INTEGER, note STRING)")
        .unwrap();
    db.execute("INSERT (:Doc {id: 1}), (:Doc {id: 2})").unwrap();
    let mut session = db.session();
    session.set_auto_commit(false);
    session
        .execute(
            "MATCH (a:Doc {id: 1}), (b:Doc {id: 2}) \
             MERGE (a)-[r:CITES {since: 2020}]->(b) ON CREATE SET r.note = r.since + 1",
        )
        .unwrap_err();

    let edges = db
        .execute("MATCH ()-[r:CITES]->() RETURN count(r)")
        .unwrap();
    assert_eq!(edges.rows()[0][0], grafeo_common::types::Value::Int64(0));
    assert_eq!(doc_ids(&db), ids(&[1, 2]));
}

/// The session's batch calls with auto-commit off: a batch whose later row
/// fails leaves none of its rows.
#[test]
#[expect(
    deprecated,
    reason = "auto-commit off is what #536 is about: the setting no longer changes how writes run"
)]
fn a_failed_direct_batch_with_auto_commit_off_leaves_nothing() {
    use grafeo_common::types::{PropertyKey, Value};
    use std::collections::HashMap;

    let db = typed_docs();
    let mut session = db.session();
    session.set_auto_commit(false);
    session
        .batch_create_nodes_with_props(
            "Doc",
            vec![
                HashMap::from([(PropertyKey::new("id"), Value::Int64(3))]),
                HashMap::from([(PropertyKey::new("id"), Value::from("x"))]),
            ],
        )
        .unwrap_err();
    assert_eq!(doc_ids(&db), ids(&[]));
}

/// `batch_create_nodes` with auto-commit off: a vector of another size than
/// the property's vector index breaks the batch, and its first vector is
/// undone with it.
#[cfg(feature = "vector-index")]
#[test]
#[expect(
    deprecated,
    reason = "auto-commit off is what #536 is about: the setting no longer changes how writes run"
)]
fn a_failed_vector_batch_with_auto_commit_off_leaves_nothing() {
    let db = GrafeoDB::new_in_memory();
    db.create_vector_index(
        "Point",
        "embedding",
        Some(3),
        Some("cosine"),
        None,
        None,
        None,
    )
    .unwrap();
    let mut session = db.session();
    session.set_auto_commit(false);
    session
        .batch_create_nodes(
            "Point",
            "embedding",
            vec![vec![0.3, 0.19, 0.88], vec![0.3, 0.19]],
        )
        .unwrap_err();
    let points = db.execute("MATCH (p:Point) RETURN count(p)").unwrap();
    assert_eq!(
        points.rows()[0][0],
        grafeo_common::types::Value::Int64(0),
        "the batch's first vector is undone with the failing one"
    );
}

/// A write statement with auto-commit off claims what it writes like any
/// transaction: it fails with a write conflict on a node an open
/// transaction changed first, and changes nothing.
#[test]
#[expect(
    deprecated,
    reason = "auto-commit off is what #536 is about: the setting no longer changes how writes run"
)]
fn a_statement_with_auto_commit_off_conflicts_with_an_open_transaction() {
    let db = GrafeoDB::new_in_memory();
    db.execute("INSERT (:Person {name: 'Alix', city: 'Amsterdam'})")
        .unwrap();
    let mut open = db.session();
    open.begin_transaction().unwrap();
    open.execute("MATCH (p:Person) SET p.city = 'Berlin'")
        .unwrap();

    let mut manual = db.session();
    manual.set_auto_commit(false);
    let error = manual
        .execute("MATCH (p:Person) SET p.city = 'Paris'")
        .unwrap_err()
        .to_string();
    assert!(error.to_lowercase().contains("conflict"), "got: {error}");

    open.commit().unwrap();
    let city = db.execute("MATCH (p:Person) RETURN p.city").unwrap();
    assert_eq!(
        city.rows()[0][0],
        grafeo_common::types::Value::from("Berlin")
    );
}

/// The failed statement leaves nothing in the WAL either, and each
/// statement that succeeded is replayed: the writes run in a child process
/// that exits without `close()`, so the reopen replays the WAL.
#[cfg(all(feature = "wal", feature = "grafeo-file"))]
#[test]
#[expect(
    deprecated,
    reason = "auto-commit off is what #536 is about: the setting no longer changes how writes run"
)]
fn a_failed_statement_with_auto_commit_off_is_not_replayed() {
    let (_dir, db) = common::replay::reopened_after_crash(
        "a_failed_statement_with_auto_commit_off_is_not_replayed",
        |path| GrafeoDB::open(path).unwrap(),
        |db| {
            db.execute("CREATE NODE TYPE Doc (id INTEGER, tag STRING)")
                .unwrap();
            let mut session = db.session();
            session.set_auto_commit(false);
            session.execute("INSERT (:Doc {id: 1})").unwrap();
            session
                .execute("UNWIND [5, 'x'] AS v INSERT (:Doc {id: v})")
                .unwrap_err();
            session.execute("INSERT (:Doc {id: 2})").unwrap();
        },
    );

    assert_eq!(doc_ids(&db), ids(&[1, 2]));
}
