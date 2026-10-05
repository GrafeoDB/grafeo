//! A closed database takes no more writes.
//!
//! `close()` writes a final checkpoint and then removes the WAL. A commit, or
//! a write outside a transaction, that ran after that checkpoint was written
//! only to the WAL that `close()` removed (or, without the `wal` feature,
//! nowhere): it returned success and was gone after a reopen. From the moment
//! `close()` of a persistent database starts, commits and writes fail with an
//! error saying the database is closed; a commit already in progress
//! completes first and is in the final checkpoint. Reads still work, and an
//! in-memory database, which has nothing to persist, still takes writes after
//! `close()`.
//!
//! ```bash
//! cargo test -p grafeo-engine --all-features --test writes_after_close
//! cargo test -p grafeo-engine --no-default-features --features lpg,gql,grafeo-file --test writes_after_close
//! ```

#![cfg(all(feature = "lpg", feature = "gql", feature = "grafeo-file"))]

use std::path::Path;

use grafeo_common::types::Value;
use grafeo_common::utils::error::{Error, TransactionError};
use grafeo_engine::{Config, GrafeoDB};

#[cfg(feature = "testing-statement-injection")]
#[path = "common/started.rs"]
mod started;

#[cfg(feature = "testing-statement-injection")]
use started::Started;

/// A read-write open of `path` (`GrafeoDB::open` needs the `wal` feature).
fn open(path: &Path) -> GrafeoDB {
    GrafeoDB::with_config(Config::persistent(path)).unwrap()
}

/// The names of the people in `db`, sorted.
fn people(db: &GrafeoDB) -> Vec<Value> {
    db.execute("MATCH (p:Person) RETURN p.name AS name ORDER BY name")
        .unwrap()
        .rows()
        .iter()
        .map(|row| row[0].clone())
        .collect()
}

/// A database at `path` holding Alix, closed, so the file holds her.
fn database_with_alix(path: &Path) {
    let db = open(path);
    db.execute("INSERT (:Person {name: 'Alix'})").unwrap();
    db.close().unwrap();
}

/// Checks that `error` says the database is closed.
fn assert_closed(error: &impl std::fmt::Display) {
    let message = error.to_string();
    assert!(message.contains("database is closed"), "{message}");
}

/// The people a fresh open of `path` shows.
fn people_after_reopen(path: &Path) -> Vec<Value> {
    let db = open(path);
    let names = people(&db);
    db.close().unwrap();
    names
}

/// A transaction begun before `close()` and committed after it fails, is
/// rolled back, and is not in the file.
#[test]
fn a_transaction_committed_after_close_fails_and_is_not_in_the_file() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("amsterdam.grafeo");
    database_with_alix(&path);

    let db = open(&path);
    let mut session = db.session();
    session.begin_transaction().unwrap();
    session.execute("INSERT (:Person {name: 'Gus'})").unwrap();
    db.close().unwrap();

    let error = session.commit().expect_err("a commit after close() fails");
    assert!(
        matches!(error, Error::Transaction(TransactionError::DatabaseClosed)),
        "a typed error: {error:?}"
    );
    assert_eq!(error.error_code().as_str(), "GRAFEO-T007");
    assert_closed(&error);
    assert_eq!(
        people(&db),
        vec![Value::from("Alix")],
        "the refused commit is rolled back, and reads still work"
    );
    drop(session);
    drop(db);
    assert_eq!(people_after_reopen(&path), vec![Value::from("Alix")]);
}

/// A statement that commits on its own after `close()` fails.
#[test]
fn a_statement_after_close_fails_and_is_not_in_the_file() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("berlin.grafeo");
    database_with_alix(&path);

    let db = open(&path);
    db.close().unwrap();

    let error = db
        .execute("INSERT (:Person {name: 'Gus'})")
        .expect_err("a write after close() fails");
    assert_closed(&error);
    assert_eq!(people(&db), vec![Value::from("Alix")]);
    drop(db);
    assert_eq!(people_after_reopen(&path), vec![Value::from("Alix")]);
}

/// Writes outside a transaction (the direct API) after `close()` fail and
/// change nothing.
#[test]
fn a_direct_write_after_close_fails_and_changes_nothing() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("paris.grafeo");
    database_with_alix(&path);

    let db = open(&path);
    let alix = db
        .execute("MATCH (p:Person) RETURN id(p) AS id")
        .unwrap()
        .rows()[0][0]
        .clone();
    let Value::Int64(alix) = alix else {
        panic!("id(p) is an integer, got {alix:?}");
    };
    let alix = grafeo_common::types::NodeId::new(u64::try_from(alix).unwrap());
    db.close().unwrap();

    assert_closed(&db.create_node(&["Person"]).expect_err("create_node fails"));
    assert_closed(
        &db.set_node_property(alix, "city", Value::from("Paris"))
            .expect_err("set_node_property fails"),
    );
    assert_closed(&db.delete_node(alix).expect_err("delete_node fails"));
    assert_eq!(db.node_count(), 1, "nothing was created or deleted");
    assert_eq!(
        db.get_node(alix)
            .and_then(|node| node.get_property("city").cloned()),
        None,
        "the property was not set"
    );
    drop(db);
    assert_eq!(people_after_reopen(&path), vec![Value::from("Alix")]);
}

/// Schema changes and graph commands take effect at once, outside any
/// commit: after `close()` they fail too, and read-only schema statements
/// still work.
#[test]
fn schema_changes_and_graph_commands_after_close_fail() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("rotterdam.grafeo");
    database_with_alix(&path);

    let db = open(&path);
    db.close().unwrap();

    assert_closed(
        &db.execute("CREATE CONSTRAINT person_name FOR (p:Person) ON (p.name) UNIQUE")
            .expect_err("a schema change after close() fails"),
    );
    assert_closed(
        &db.execute("CREATE GRAPH berlin")
            .expect_err("a graph command after close() fails"),
    );
    let constraints = db.execute("SHOW CONSTRAINTS").unwrap();
    assert!(
        constraints.rows().is_empty(),
        "SHOW still works and lists no constraint: {:?}",
        constraints.rows()
    );
    drop(db);

    let reopened = open(&path);
    let constraints = reopened.execute("SHOW CONSTRAINTS").unwrap();
    assert!(
        constraints.rows().is_empty(),
        "the reopened file holds no constraint: {:?}",
        constraints.rows()
    );
    let graphs = reopened.execute("SHOW GRAPHS").unwrap();
    assert!(
        !graphs
            .rows()
            .iter()
            .any(|row| row.contains(&Value::from("berlin"))),
        "the graph was not created: {:?}",
        graphs.rows()
    );
    // The check above can see a graph: an open database creates one.
    reopened.execute("CREATE GRAPH berlin").unwrap();
    let graphs = reopened.execute("SHOW GRAPHS").unwrap();
    assert!(
        graphs
            .rows()
            .iter()
            .any(|row| row.contains(&Value::from("berlin"))),
        "{:?}",
        graphs.rows()
    );
    reopened.close().unwrap();
}

/// A statement inside a transaction begun before `close()` fails at once,
/// before it writes.
#[test]
fn a_statement_in_a_transaction_begun_before_close_fails() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("utrecht.grafeo");
    database_with_alix(&path);

    let db = open(&path);
    let mut session = db.session();
    session.begin_transaction().unwrap();
    db.close().unwrap();

    assert_closed(
        &session
            .execute("INSERT (:Person {name: 'Gus'})")
            .expect_err("a write after close() fails"),
    );
    assert_eq!(
        session
            .execute("MATCH (p:Person) RETURN p.name AS name")
            .unwrap()
            .rows()
            .len(),
        1,
        "the transaction still reads, and holds no write"
    );
    drop(session);
    drop(db);
    assert_eq!(people_after_reopen(&path), vec![Value::from("Alix")]);
}

/// An in-memory database has nothing to persist: `close()` leaves it
/// working, for writes too.
#[test]
fn an_in_memory_database_still_takes_writes_after_close() {
    let db = GrafeoDB::new_in_memory();
    db.execute("INSERT (:Person {name: 'Alix'})").unwrap();
    db.close().unwrap();

    db.execute("INSERT (:Person {name: 'Gus'})").unwrap();
    db.create_node(&["City"]).unwrap();
    assert_eq!(people(&db), vec![Value::from("Alix"), Value::from("Gus")]);
    assert_eq!(db.node_count(), 3);
}

/// The race `close()` used to lose: work that waits for the final checkpoint
/// (which holds commits off) and runs once it is written. A transaction that
/// wrote before `close()` and commits meanwhile fails, instead of landing in
/// the WAL that `close()` then removes. (A statement that starts once
/// `close()` began fails at once, before it writes.)
#[cfg(feature = "testing-statement-injection")]
#[test]
fn a_commit_waiting_for_the_final_checkpoint_fails() {
    use std::sync::{Arc, mpsc};

    use grafeo_common::testing::commit_hook::during_next_checkpoint;

    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("prague.grafeo");
    database_with_alix(&path);

    let db = Arc::new(open(&path));
    let mut session = db.session();
    session.begin_transaction().unwrap();
    session.execute("INSERT (:Person {name: 'Gus'})").unwrap();
    let (sender, started) = mpsc::channel();
    during_next_checkpoint(move || {
        let commit = Started::spawn(move || session.commit().map_err(|e| e.to_string()));
        let finished = commit.finishes_briefly();
        sender.send((commit, finished)).unwrap();
    });
    db.close().unwrap();

    let (commit, finished) = started.recv().expect("close() checkpointed");
    assert!(!finished, "the commit waits for the final checkpoint");
    assert_closed(&commit.join().expect_err("the commit fails"));
    drop(db);
    assert_eq!(people_after_reopen(&path), vec![Value::from("Alix")]);
}

/// A direct write (no session, no transaction) that waits for the final
/// checkpoint fails once it is written, and changes nothing.
#[cfg(feature = "testing-statement-injection")]
#[test]
fn a_direct_write_waiting_for_the_final_checkpoint_fails() {
    use std::sync::{Arc, mpsc};

    use grafeo_common::testing::commit_hook::during_next_checkpoint;

    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("delft.grafeo");
    database_with_alix(&path);

    let db = Arc::new(open(&path));
    let writer = Arc::clone(&db);
    let (sender, started) = mpsc::channel();
    during_next_checkpoint(move || {
        let write = Started::spawn(move || {
            writer
                .create_node(&["Person"])
                .map(|_| ())
                .map_err(|e| e.to_string())
        });
        let finished = write.finishes_briefly();
        sender.send((write, finished)).unwrap();
    });
    db.close().unwrap();

    let (write, finished) = started.recv().expect("close() checkpointed");
    assert!(!finished, "the write waits for the final checkpoint");
    assert_closed(&write.join().expect_err("the write fails"));
    assert_eq!(db.node_count(), 1, "nothing was created");
    drop(db);
    assert_eq!(people_after_reopen(&path), vec![Value::from("Alix")]);
}

/// A schema change or a graph command that waits for the final checkpoint
/// (they hold commits off for the whole statement) fails once it is written,
/// instead of changing the catalog after the last image and logging to the WAL
/// `close()` removes.
#[cfg(feature = "testing-statement-injection")]
#[test]
fn a_schema_change_waiting_for_the_final_checkpoint_fails() {
    use std::sync::{Arc, mpsc};

    use grafeo_common::testing::commit_hook::during_next_checkpoint;

    for (city, statement) in [
        (
            "leiden",
            "CREATE CONSTRAINT person_name FOR (p:Person) ON (p.name) UNIQUE",
        ),
        ("haarlem", "CREATE GRAPH berlin"),
    ] {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join(format!("{city}.grafeo"));
        database_with_alix(&path);

        let db = Arc::new(open(&path));
        let writer = Arc::clone(&db);
        let (sender, started) = mpsc::channel();
        during_next_checkpoint(move || {
            let write = Started::spawn(move || {
                writer
                    .execute(statement)
                    .map(|_| ())
                    .map_err(|e| e.to_string())
            });
            let finished = write.finishes_briefly();
            sender.send((write, finished)).unwrap();
        });
        db.close().unwrap();

        let (write, finished) = started.recv().expect("close() checkpointed");
        assert!(!finished, "{statement}: waits for the final checkpoint");
        assert_closed(&write.join().expect_err("the statement fails"));
        drop(db);
        let reopened = open(&path);
        assert!(
            reopened
                .execute("SHOW CONSTRAINTS")
                .unwrap()
                .rows()
                .is_empty(),
            "{statement}: no constraint in the file"
        );
        assert!(
            reopened.graph("berlin").is_err(),
            "{statement}: no graph in the file"
        );
        reopened.close().unwrap();
    }
}

/// A commit in progress when `close()` starts completes first and is in the
/// final checkpoint.
#[cfg(feature = "testing-statement-injection")]
#[test]
fn close_waits_for_a_commit_in_progress_and_writes_it() {
    use std::sync::{Arc, mpsc};

    use grafeo_common::testing::commit_hook::after_next_commit_stamped;

    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("barcelona.grafeo");
    database_with_alix(&path);

    let db = Arc::new(open(&path));
    let mut session = db.session();
    session.begin_transaction().unwrap();
    session.execute("INSERT (:Person {name: 'Gus'})").unwrap();
    let closer = Arc::clone(&db);
    let (sender, started) = mpsc::channel();
    after_next_commit_stamped(move || {
        let close = Started::spawn(move || closer.close().map_err(|e| e.to_string()));
        let finished = close.finishes_briefly();
        sender.send((close, finished)).unwrap();
    });
    session.commit().unwrap();

    let (close, finished) = started.recv().expect("the commit ran the hook");
    assert!(!finished, "close() waits for the commit in progress");
    close.join().unwrap();
    drop(session);
    drop(db);
    assert_eq!(
        people_after_reopen(&path),
        vec![Value::from("Alix"), Value::from("Gus")]
    );
}
