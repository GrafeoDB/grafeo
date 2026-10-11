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
#[path = "common/image.rs"]
mod image;
#[cfg(feature = "testing-statement-injection")]
#[path = "common/started.rs"]
mod started;

#[cfg(feature = "testing-statement-injection")]
use image::image_holds;
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

/// Every file and directory under `dir` (spill directories left out), with
/// the bytes of each file, sorted.
fn files_in(dir: &Path) -> Vec<(std::path::PathBuf, Option<Vec<u8>>)> {
    let mut found = Vec::new();
    let mut pending = vec![dir.to_path_buf()];
    while let Some(dir) = pending.pop() {
        for entry in std::fs::read_dir(&dir).unwrap() {
            let path = entry.unwrap().path();
            if path.to_string_lossy().contains(".spill") {
                continue;
            }
            if path.is_dir() {
                found.push((path.clone(), None));
                pending.push(path);
            } else {
                let bytes = std::fs::read(&path).unwrap();
                found.push((path, Some(bytes)));
            }
        }
    }
    found.sort();
    found
}

/// Checks that `result` failed with the typed database-closed error.
#[track_caller]
fn assert_closed_error<T>(call: &str, result: grafeo_common::utils::error::Result<T>) {
    match result {
        Ok(_) => panic!("{call} after close() succeeded"),
        Err(error) => assert!(
            matches!(error, Error::Transaction(TransactionError::DatabaseClosed)),
            "{call}: the database-closed error, got {error:?}"
        ),
    }
}

/// Checkpoints, backups, saves and compaction after `close()` fail with the
/// database-closed error and write nothing: `close()` released the file,
/// which another handle may have opened and written since.
#[test]
fn persisting_after_close_fails_and_writes_nothing() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("prague.grafeo");
    database_with_alix(&path);

    let db = open(&path);
    db.close().unwrap();
    let before = files_in(dir.path());

    assert_closed_error("wal_checkpoint", db.wal_checkpoint());
    #[cfg(feature = "wal")]
    assert_closed_error("save", db.save(dir.path().join("copy.grafeo")));
    #[cfg(all(feature = "wal", feature = "grafeo-file"))]
    {
        assert_closed_error("backup_full", db.backup_full(&dir.path().join("backups")));
        assert_closed_error(
            "backup_incremental",
            db.backup_incremental(&dir.path().join("backups")),
        );
    }
    // `compact()` takes `&mut self`.
    let db = {
        let mut db = db;
        assert_closed_error("compact", db.compact());
        db
    };
    assert!(
        files_in(dir.path()) == before,
        "nothing was written: {:?}",
        files_in(dir.path())
            .iter()
            .map(|(file, _)| file)
            .collect::<Vec<_>>()
    );
    assert_eq!(people(&db), vec![Value::from("Alix")], "reads still work");
}

/// Without a WAL (every persistent database in a build without the `wal`
/// feature) the file holds the only copy: a checkpoint from a handle that was
/// closed would overwrite what a later handle wrote with its stale image. It
/// fails instead, and so does `compact()`, which would restart the periodic
/// checkpoint; the later handle's commit stays.
#[test]
fn a_closed_handle_overwrites_nothing_a_later_handle_wrote() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("paris.grafeo");
    let config = || {
        let mut config = Config::persistent(&path)
            .with_checkpoint_interval(std::time::Duration::from_millis(50));
        config.wal_enabled = false;
        config
    };

    let first = GrafeoDB::with_config(config()).unwrap();
    first.execute("INSERT (:Person {name: 'Alix'})").unwrap();
    first.close().unwrap();
    {
        let second = GrafeoDB::with_config(config()).unwrap();
        second.execute("INSERT (:Person {name: 'Gus'})").unwrap();
        second.close().unwrap();
    }

    assert_closed_error("wal_checkpoint", first.wal_checkpoint());
    {
        // `compact()` takes `&mut self`.
        let mut first = first;
        assert_closed_error("compact", first.compact());
    }

    let db = GrafeoDB::with_config(config()).unwrap();
    assert_eq!(
        people(&db),
        vec![Value::from("Alix"), Value::from("Gus")],
        "the later handle's commit is in the file"
    );
    db.close().unwrap();
}

/// SPARQL updates after `close()` fail like every other write, through each
/// path (a statement outside a transaction, `GrafeoDB::execute_sparql`, and a
/// transaction begun before `close()`), also for the embedded admin identity;
/// they used to land in memory and in the WAL `close()` removed. Queries
/// still run.
#[cfg(all(feature = "sparql", feature = "triple-store"))]
#[test]
fn sparql_updates_after_close_fail_and_are_not_in_the_file() {
    const ALIX: &str = r#"INSERT DATA { <http://ex.org/alix> <http://ex.org/city> "Amsterdam" . }"#;
    const GUS: &str = r#"INSERT DATA { <http://ex.org/gus> <http://ex.org/city> "Berlin" . }"#;

    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("berlin.grafeo");
    let db = open(&path);
    db.execute_language(ALIX, "sparql", None).unwrap();
    let mut session = db.session();
    session.begin_transaction().unwrap();
    db.close().unwrap();

    assert_closed_error(
        "execute_language sparql",
        db.execute_language(GUS, "sparql", None),
    );
    assert_closed_error("GrafeoDB::execute_sparql", db.execute_sparql(GUS));
    assert_closed_error(
        "Session::execute_sparql in a transaction",
        session.execute_sparql(GUS),
    );
    assert_eq!(db.rdf_store().len(), 1, "no triple was added in memory");
    assert_eq!(
        db.execute_sparql("SELECT ?s WHERE { ?s ?p ?o }")
            .unwrap()
            .rows()
            .len(),
        1,
        "queries still run"
    );
    drop(session);
    drop(db);

    let reopened = open(&path);
    assert_eq!(
        reopened.rdf_store().len(),
        1,
        "the file holds Alix's triple"
    );
    reopened.close().unwrap();
}

/// The direct graph and index calls after `close()` fail with the
/// database-closed error and change nothing: they used to report success,
/// and a reopen did not show the change.
#[test]
fn direct_graph_and_index_calls_after_close_fail() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("amsterdam.grafeo");
    {
        let db = open(&path);
        db.execute("INSERT (:Person {name: 'Alix'})").unwrap();
        db.create_graph("prague").unwrap();
        db.create_property_index("name").unwrap();
        db.close().unwrap();
    }
    let indexed = |db: &GrafeoDB| (db.has_property_index("name"), db.has_property_index("city"));

    let db = open(&path);
    db.close().unwrap();
    assert_closed_error("create_graph", db.create_graph("berlin"));
    assert_closed_error("drop_graph", db.drop_graph("prague"));
    assert_closed_error("create_property_index", db.create_property_index("city"));
    assert_closed_error("drop_property_index", db.drop_property_index("name"));
    #[cfg(feature = "vector-index")]
    {
        assert_closed_error(
            "create_vector_index",
            db.create_vector_index("Person", "embedding", Some(3), None, None, None, None),
        );
        assert_closed_error(
            "drop_vector_index",
            db.drop_vector_index("Person", "embedding"),
        );
        assert_closed_error(
            "rebuild_vector_index",
            db.rebuild_vector_index("Person", "embedding"),
        );
    }
    #[cfg(feature = "text-index")]
    {
        assert_closed_error("create_text_index", db.create_text_index("Person", "name"));
        assert_closed_error("drop_text_index", db.drop_text_index("Person", "name"));
        assert_closed_error(
            "rebuild_text_index",
            db.rebuild_text_index("Person", "name"),
        );
    }
    assert_eq!(
        db.list_graphs(),
        vec!["prague".to_string()],
        "no graph changed"
    );
    assert_eq!(indexed(&db), (true, false), "no index changed");
    drop(db);

    let reopened = open(&path);
    assert_eq!(reopened.list_graphs(), vec!["prague".to_string()]);
    assert_eq!(indexed(&reopened), (true, false));
    reopened.close().unwrap();
}

/// `CREATE PROJECTION` and `DROP PROJECTION` change only the session's
/// projections (nothing is logged or persisted): they still work after
/// `close()`, as reads do.
#[test]
fn projections_still_work_after_close() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("barcelona.grafeo");
    database_with_alix(&path);
    let db = open(&path);
    db.close().unwrap();

    let session = db.session();
    session
        .execute("CREATE PROJECTION social LABELS (Person)")
        .unwrap_or_else(|error| panic!("CREATE PROJECTION after close(): {error}"));
    assert_eq!(
        session.execute("SHOW PROJECTIONS").unwrap().rows().to_vec(),
        vec![vec![Value::from("social")]]
    );
    session
        .execute("DROP PROJECTION social")
        .unwrap_or_else(|error| panic!("DROP PROJECTION after close(): {error}"));
}

/// A graph command refused for another reason says so after `close()`: the
/// read-only and grant checks come before commits are held off (which fails
/// once the database is closed), so a read-only transaction gets the
/// read-only error and a session without a grant the permission error.
#[test]
fn a_graph_command_refused_for_another_reason_says_so_after_close() {
    use grafeo_engine::auth::{Grant, Identity, Role};

    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("amsterdam.grafeo");
    database_with_alix(&path);
    let db = open(&path);
    db.close().unwrap();

    let read_only = db.session();
    read_only.execute("START TRANSACTION READ ONLY").unwrap();
    let error = read_only
        .execute("CREATE GRAPH berlin")
        .expect_err("a graph command in a read-only transaction fails");
    assert!(
        matches!(error, Error::Transaction(TransactionError::ReadOnly)),
        "the read-only error: {error:?}"
    );

    let without_grant = db.session_with_identity(
        Identity::new("gus", [Role::ReadWrite]).with_grants([Grant::new("paris", Role::ReadWrite)]),
    );
    let error = without_grant
        .execute("CREATE GRAPH berlin")
        .expect_err("a graph command without a grant fails");
    assert!(
        error.to_string().contains("permission denied"),
        "the permission error: {error}"
    );
}

/// The schema statements and graph commands the hold tests run, with what
/// each leaves in the database.
#[cfg(feature = "testing-statement-injection")]
const HELD_STATEMENTS: [(&str, &str); 2] = [
    (
        "CREATE CONSTRAINT person_name_unique FOR (p:Person) ON (p.name) UNIQUE",
        "person_name_unique",
    ),
    ("CREATE GRAPH barcelona", "barcelona"),
];

/// A schema statement or a graph command holds commits off for its whole
/// run, so a `close()` that starts while it holds them (from a hook right
/// after the hold is taken, before it changes anything) waits for it, and
/// the final checkpoint holds its change: it is in the reopened database.
#[cfg(feature = "testing-statement-injection")]
#[test]
fn close_waits_for_a_schema_change_holding_commits_off() {
    use std::sync::{Arc, mpsc};

    use grafeo_common::testing::commit_hook::during_next_held_change;

    for (statement, made) in HELD_STATEMENTS {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("amsterdam.grafeo");
        database_with_alix(&path);

        let db = Arc::new(open(&path));
        let closer = Arc::clone(&db);
        let (sender, started) = mpsc::channel();
        during_next_held_change(move || {
            let close = Started::spawn(move || closer.close().map_err(|e| e.to_string()));
            let finished = close.finishes_briefly();
            sender.send((close, finished)).unwrap();
        });
        db.execute(statement)
            .unwrap_or_else(|error| panic!("{statement}: {error}"));

        // The hook runs inside the statement, so its message is there now.
        let (close, finished) = started.try_recv().expect("the statement ran the hook");
        assert!(!finished, "{statement}: close() waits for the statement");
        close.join().unwrap();
        drop(db);

        let reopened = open(&path);
        assert!(
            reopened
                .execute("SHOW CONSTRAINTS")
                .unwrap()
                .rows()
                .iter()
                .any(|row| row.contains(&Value::from(made)))
                || reopened.list_graphs().contains(&made.to_string()),
            "{statement}: the change is in the file"
        );
        reopened.close().unwrap();
    }
}

/// A checkpoint that starts while a schema statement or graph command holds
/// commits off waits for it, and its image holds the whole change.
#[cfg(feature = "testing-statement-injection")]
#[test]
fn a_checkpoint_waits_for_a_schema_change_and_holds_all_of_it() {
    use std::sync::{Arc, mpsc};

    use grafeo_common::testing::commit_hook::during_next_held_change;

    for (statement, made) in HELD_STATEMENTS {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("berlin.grafeo");
        database_with_alix(&path);

        let db = Arc::new(open(&path));
        let checkpointer = Arc::clone(&db);
        let (sender, started) = mpsc::channel();
        during_next_held_change(move || {
            let checkpoint =
                Started::spawn(move || checkpointer.wal_checkpoint().map_err(|e| e.to_string()));
            let finished = checkpoint.finishes_briefly();
            sender.send((checkpoint, finished)).unwrap();
        });
        db.execute(statement)
            .unwrap_or_else(|error| panic!("{statement}: {error}"));

        // The hook runs inside the statement, so its message is there now.
        let (checkpoint, finished) = started.try_recv().expect("the statement ran the hook");
        assert!(
            !finished,
            "{statement}: the checkpoint waits for the statement"
        );
        checkpoint.join().unwrap();
        assert!(
            image_holds(&db, made),
            "{statement}: the checkpoint image holds the change"
        );
        db.close().unwrap();
    }
}

/// A direct write holds commits off from its epoch until it is published, so
/// a `close()` that starts in between (from a hook once its epoch has moved,
/// before it writes) waits for it, and the final checkpoint holds the write.
#[cfg(feature = "testing-statement-injection")]
#[test]
fn close_waits_for_a_direct_write_holding_commits_off() {
    use std::sync::{Arc, mpsc};

    use grafeo_common::testing::commit_hook::during_next_held_change;

    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("paris.grafeo");
    database_with_alix(&path);

    let db = Arc::new(open(&path));
    let closer = Arc::clone(&db);
    let (sender, started) = mpsc::channel();
    during_next_held_change(move || {
        let close = Started::spawn(move || closer.close().map_err(|e| e.to_string()));
        let finished = close.finishes_briefly();
        sender.send((close, finished)).unwrap();
    });
    db.create_node_with_props(&["Person"], [("name", Value::from("Gus"))])
        .unwrap();

    // The hook runs inside the write, so its message is there now.
    let (close, finished) = started.try_recv().expect("the write ran the hook");
    assert!(!finished, "close() waits for the direct write");
    close.join().unwrap();
    drop(db);
    assert_eq!(
        people_after_reopen(&path),
        vec![Value::from("Alix"), Value::from("Gus")]
    );
}

/// A checkpoint that starts while a direct write holds commits off waits for
/// it: the checkpoint header counts the written node.
#[cfg(feature = "testing-statement-injection")]
#[test]
fn a_checkpoint_waits_for_a_direct_write_and_counts_it() {
    use std::sync::{Arc, mpsc};

    use grafeo_common::testing::commit_hook::during_next_held_change;

    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("prague.grafeo");
    database_with_alix(&path);

    let db = Arc::new(open(&path));
    let checkpointer = Arc::clone(&db);
    let (sender, started) = mpsc::channel();
    during_next_held_change(move || {
        let checkpoint =
            Started::spawn(move || checkpointer.wal_checkpoint().map_err(|e| e.to_string()));
        let finished = checkpoint.finishes_briefly();
        sender.send((checkpoint, finished)).unwrap();
    });
    db.create_node_with_props(&["Person"], [("name", Value::from("Gus"))])
        .unwrap();

    // The hook runs inside the write, so its message is there now.
    let (checkpoint, finished) = started.try_recv().expect("the write ran the hook");
    assert!(!finished, "the checkpoint waits for the direct write");
    checkpoint.join().unwrap();
    assert_eq!(
        db.file_manager().unwrap().active_header().node_count,
        2,
        "the checkpoint header counts Alix and Gus"
    );
    db.close().unwrap();
}

/// Bulk imports, RDF batch inserts and snapshot restores after `close()`
/// fail with the database-closed error and change nothing (what they changed
/// after the final checkpoint would be lost: the close removes the WAL).
#[test]
fn imports_and_restores_after_close_fail() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("barcelona.grafeo");
    database_with_alix(&path);
    let snapshot = {
        let other = GrafeoDB::new_in_memory();
        other.execute("INSERT (:Person {name: 'Gus'})").unwrap();
        other.export_snapshot().unwrap()
    };

    let db = open(&path);
    db.close().unwrap();
    assert_closed_error(
        "import_tsv_str",
        db.import_tsv_str("3\t19\n19\t88\n", "KNOWS", true),
    );
    assert_closed_error("restore_snapshot", db.restore_snapshot(&snapshot));
    #[cfg(feature = "triple-store")]
    {
        use grafeo_core::graph::rdf::{Term, Triple};
        let triple = Triple::new(
            Term::iri("http://ex.org/alix"),
            Term::iri("http://ex.org/city"),
            Term::literal("Amsterdam"),
        );
        assert_closed_error("batch_insert_rdf", db.batch_insert_rdf([triple]));
        let tsv = dir.path().join("edges.tsv");
        std::fs::write(&tsv, "3\t19\n").unwrap();
        assert_closed_error(
            "import_tsv_rdf",
            db.import_tsv_rdf(&tsv, "http://ex.org/knows", "http://ex.org/"),
        );
        assert_eq!(db.rdf_store().len(), 0, "no triple was added");
    }
    assert_eq!(people(&db), vec![Value::from("Alix")], "nothing changed");
    assert_eq!(db.node_count(), 1);
}

/// After `close()` an RDF batch insert refuses before it pulls the caller's
/// iterator (which may parse or compute the triples), and an import before it
/// opens its file or parses its data: a refused call does no work, and a
/// missing file or malformed data still gets the database-closed error.
#[test]
fn imports_after_close_fail_before_reading_their_input() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("amsterdam.grafeo");
    database_with_alix(&path);
    let missing = dir.path().join("missing.tsv");
    assert!(!missing.exists(), "the input file does not exist");

    let db = open(&path);
    db.close().unwrap();
    #[cfg(feature = "triple-store")]
    {
        use grafeo_core::graph::rdf::{Term, Triple};
        let pulled = std::cell::Cell::new(false);
        let triples = std::iter::once_with(|| {
            pulled.set(true);
            Triple::new(
                Term::iri("http://ex.org/gus"),
                Term::iri("http://ex.org/city"),
                Term::literal("Berlin"),
            )
        });
        assert_closed_error("batch_insert_rdf", db.batch_insert_rdf(triples));
        assert!(
            !pulled.get(),
            "the refused batch insert pulled the caller's iterator"
        );
        assert_closed_error(
            "import_tsv_rdf of a missing file",
            db.import_tsv_rdf(&missing, "http://ex.org/knows", "http://ex.org/"),
        );
        assert_eq!(db.rdf_store().len(), 0, "no triple was added");
    }
    assert_closed_error(
        "import_tsv of a missing file",
        db.import_tsv(&missing, "KNOWS", true),
    );
    assert_closed_error(
        "import_mmio of a missing file",
        db.import_mmio(dir.path().join("missing.mtx"), "KNOWS"),
    );
    assert_closed_error(
        "import_tsv_str of malformed data",
        db.import_tsv_str("Alix\tGus\n", "KNOWS", true),
    );
    assert_eq!(people(&db), vec![Value::from("Alix")], "nothing changed");
}

/// A bulk import or an RDF batch insert, by name, given a directory for its
/// input file, and how many items it adds (see [`imported`]): three nodes
/// and two edges besides Alix, or two triples.
type Import = (
    &'static str,
    fn(&GrafeoDB, &Path) -> grafeo_common::utils::error::Result<()>,
    usize,
);

/// The bulk imports and the RDF batch insert.
fn imports() -> Vec<Import> {
    let imports: Vec<Import> = vec![
        (
            "import_tsv_str",
            |db, _| {
                db.import_tsv_str("3\t19\n19\t88\n", "KNOWS", true)
                    .map(drop)
            },
            5,
        ),
        (
            "import_mmio",
            |db, dir| {
                let mtx = dir.join("edges.mtx");
                std::fs::write(
                    &mtx,
                    "%%MatrixMarket matrix coordinate real general\n88 88 2\n3 19 1.0\n19 88 1.0\n",
                )
                .unwrap();
                db.import_mmio(&mtx, "KNOWS").map(drop)
            },
            5,
        ),
    ];
    #[cfg(feature = "triple-store")]
    let imports = imports.into_iter().chain::<[Import; 2]>([
        (
            "import_tsv_rdf",
            |db, dir| {
                let tsv = dir.join("edges.tsv");
                std::fs::write(&tsv, "3\t19\n19\t88\n").unwrap();
                db.import_tsv_rdf(&tsv, "http://ex.org/knows", "http://ex.org/")
                    .map(drop)
            },
            2,
        ),
        (
            "batch_insert_rdf",
            |db, _| {
                use grafeo_core::graph::rdf::{Term, Triple};
                let city = |name: &str, city: &str| {
                    Triple::new(
                        Term::iri(format!("http://ex.org/{name}")),
                        Term::iri("http://ex.org/city"),
                        Term::literal(city),
                    )
                };
                db.batch_insert_rdf([city("gus", "Berlin"), city("mia", "Prague")])
                    .map(drop)
            },
            2,
        ),
    ]);
    imports.into_iter().collect()
}

/// The items an import added to `db`, which holds Alix besides: its nodes
/// and edges, and its triples.
fn imported(db: &GrafeoDB) -> usize {
    let items = db.node_count() - 1 + db.edge_count();
    #[cfg(feature = "triple-store")]
    let items = items + db.rdf_store().len();
    items
}

/// A read-only handle never takes a snapshot restore, an import or an RDF
/// batch insert (nothing would persist them): while open each refuses with
/// the read-only error, and after its `close()` (which releases the file)
/// with the database-closed error. Before, each changed the handle's store.
#[test]
fn a_read_only_handle_takes_no_restore_or_import_open_or_closed() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("prague.grafeo");
    database_with_alix(&path);
    let snapshot = {
        let other = GrafeoDB::new_in_memory();
        other.execute("INSERT (:Person {name: 'Gus'})").unwrap();
        other.export_snapshot().unwrap()
    };
    // Each call, run against the handle.
    type Call<'a> = Box<dyn Fn(&GrafeoDB) -> grafeo_common::utils::error::Result<()> + 'a>;
    let mut calls: Vec<(&str, Call<'_>)> = vec![(
        "restore_snapshot",
        Box::new(|db| db.restore_snapshot(&snapshot)),
    )];
    let inputs = dir.path();
    for (name, call, _) in imports() {
        calls.push((name, Box::new(move |db| call(db, inputs))));
    }

    let mut failures = Vec::new();
    for (name, call) in &calls {
        for closed in [false, true] {
            let db = GrafeoDB::open_read_only(&path).unwrap();
            if closed {
                db.close().unwrap();
            }
            let outcome = call(&db);
            match outcome {
                Ok(()) => failures.push(format!("{name}, closed {closed}: it succeeded")),
                Err(Error::Transaction(TransactionError::ReadOnly)) if !closed => {}
                Err(Error::Transaction(TransactionError::DatabaseClosed)) if closed => {}
                Err(error) => {
                    failures.push(format!("{name}, closed {closed}: another error: {error:?}"));
                }
            }
            let seen = people(&db);
            if seen != vec![Value::from("Alix")] || imported(&db) != 0 {
                failures.push(format!(
                    "{name}, closed {closed}: the handle shows {seen:?} and {} more items",
                    imported(&db)
                ));
            }
        }
    }
    assert!(failures.is_empty(), "{failures:#?}");
}

/// An import holds commits off while it changes the store, so a `close()`
/// started from a hook inside one (once it holds them, before it changes
/// anything) waits for it, and the final checkpoint holds all of the import:
/// the reopened database shows every node and edge, or triple, it added.
#[cfg(feature = "testing-statement-injection")]
#[test]
fn close_waits_for_an_import_holding_commits_off() {
    use std::sync::{Arc, mpsc};

    use grafeo_common::testing::commit_hook::during_next_held_change;

    // Every call is checked, and every failure listed at the end.
    let mut failures = Vec::new();
    for (name, call, adds) in imports() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("amsterdam.grafeo");
        database_with_alix(&path);
        let db = Arc::new(open(&path));
        let closer = Arc::clone(&db);
        let (sender, started) = mpsc::channel();
        during_next_held_change(move || {
            let close = Started::spawn(move || closer.close().map_err(|e| e.to_string()));
            let finished = close.finishes_briefly();
            sender.send((close, finished)).unwrap();
        });
        call(&db, dir.path()).unwrap_or_else(|error| panic!("{name}: {error}"));

        // The hook runs inside the import, so its message is there now.
        let Ok((close, finished)) = started.try_recv() else {
            failures.push(format!(
                "{name}: the import never held commits off (the hook did not run)"
            ));
            db.close().unwrap();
            continue;
        };
        if finished {
            failures.push(format!("{name}: close() did not wait for the import"));
        }
        close.join().unwrap();
        drop(db);

        let reopened = open(&path);
        let items = imported(&reopened);
        if items != adds {
            failures.push(format!(
                "{name}: the reopened file holds {items} of the {adds} imported items"
            ));
        }
        reopened.close().unwrap();
    }
    assert!(failures.is_empty(), "{failures:#?}");
}

/// Runs the SPARQL update that adds Gus's triple outside a transaction,
/// through a session statement (`execute_language`) or through
/// `GrafeoDB::execute_sparql`.
#[cfg(all(
    feature = "testing-statement-injection",
    feature = "sparql",
    feature = "triple-store"
))]
fn add_gus_with_sparql(db: &GrafeoDB, through_session: bool) {
    const GUS: &str = r#"INSERT DATA { <http://ex.org/gus> <http://ex.org/city> "Berlin" . }"#;
    let result = if through_session {
        db.execute_language(GUS, "sparql", None)
    } else {
        db.execute_sparql(GUS)
    };
    result.unwrap_or_else(|error| panic!("session {through_session}: {error}"));
}

/// A database at `path` holding Alix's triple, open.
#[cfg(all(
    feature = "testing-statement-injection",
    feature = "sparql",
    feature = "triple-store"
))]
fn rdf_database_with_alix(path: &Path) -> GrafeoDB {
    let db = open(path);
    db.execute_sparql(r#"INSERT DATA { <http://ex.org/alix> <http://ex.org/city> "Amsterdam" . }"#)
        .unwrap();
    db
}

/// A SPARQL update outside a transaction runs in a transaction of its own,
/// whose commit applies its triples and logs them: a `close()` started from a
/// hook inside that commit (once the triples are applied, before they are
/// logged) waits for it, and the final checkpoint holds the triple (#414).
/// Through a session and through `GrafeoDB::execute_sparql`.
#[cfg(all(
    feature = "testing-statement-injection",
    feature = "sparql",
    feature = "triple-store"
))]
#[test]
fn close_waits_for_a_sparql_update_holding_commits_off() {
    use std::sync::{Arc, mpsc};

    use grafeo_common::testing::commit_hook::after_next_commit_stamped;

    for through_session in [true, false] {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("berlin.grafeo");
        let db = Arc::new(rdf_database_with_alix(&path));
        let closer = Arc::clone(&db);
        let (sender, started) = mpsc::channel();
        after_next_commit_stamped(move || {
            let close = Started::spawn(move || closer.close().map_err(|e| e.to_string()));
            let finished = close.finishes_briefly();
            sender.send((close, finished)).unwrap();
        });
        add_gus_with_sparql(&db, through_session);

        // The hook runs inside the update's commit, so its message is there now.
        let (close, finished) = started.try_recv().expect("the update ran the hook");
        assert!(
            !finished,
            "session {through_session}: close() waits for the update"
        );
        close.join().unwrap();
        drop(db);

        let reopened = open(&path);
        assert_eq!(
            reopened.rdf_store().len(),
            2,
            "session {through_session}: the file holds Alix's and Gus's triples"
        );
        reopened.close().unwrap();
    }
}

/// A checkpoint that starts while a SPARQL update commits (its triples
/// applied, not yet logged) waits for the commit, and its image holds the
/// update's triple.
#[cfg(all(
    feature = "testing-statement-injection",
    feature = "sparql",
    feature = "triple-store"
))]
#[test]
fn a_checkpoint_waits_for_a_sparql_update_and_holds_all_of_it() {
    use std::sync::{Arc, mpsc};

    use grafeo_common::testing::commit_hook::after_next_commit_stamped;

    for through_session in [true, false] {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("paris.grafeo");
        let db = Arc::new(rdf_database_with_alix(&path));
        let checkpointer = Arc::clone(&db);
        let (sender, started) = mpsc::channel();
        after_next_commit_stamped(move || {
            let checkpoint =
                Started::spawn(move || checkpointer.wal_checkpoint().map_err(|e| e.to_string()));
            let finished = checkpoint.finishes_briefly();
            sender.send((checkpoint, finished)).unwrap();
        });
        add_gus_with_sparql(&db, through_session);

        // The hook runs inside the update's commit, so its message is there now.
        let (checkpoint, finished) = started.try_recv().expect("the update ran the hook");
        assert!(
            !finished,
            "session {through_session}: the checkpoint waits for the update"
        );
        checkpoint.join().unwrap();
        assert!(
            image_holds(&db, "http://ex.org/gus"),
            "session {through_session}: the checkpoint image holds Gus's triple"
        );
        db.close().unwrap();
    }
}

/// A direct graph or index call for the hold tests: what to set up first, the
/// call, whether a database shows its change, and a name its checkpoint image
/// holds once it ran (for a call that creates something).
#[cfg(feature = "testing-statement-injection")]
struct DirectCall {
    name: &'static str,
    setup: fn(&GrafeoDB),
    call: fn(&GrafeoDB),
    done: fn(&GrafeoDB) -> bool,
    image_needle: Option<&'static str>,
}

/// The direct graph and index calls that change the store outside any commit.
#[cfg(feature = "testing-statement-injection")]
fn direct_calls() -> Vec<DirectCall> {
    let calls = vec![
        DirectCall {
            name: "create_graph",
            setup: |_| {},
            call: |db| {
                db.create_graph("barcelona").unwrap();
            },
            done: |db| db.list_graphs().contains(&"barcelona".to_string()),
            image_needle: Some("barcelona"),
        },
        DirectCall {
            name: "drop_graph",
            setup: |db| {
                db.create_graph("prague").unwrap();
            },
            call: |db| {
                db.drop_graph("prague").unwrap();
            },
            done: |db| !db.list_graphs().contains(&"prague".to_string()),
            image_needle: None,
        },
        DirectCall {
            name: "create_property_index",
            setup: |_| {},
            call: |db| db.create_property_index("city").unwrap(),
            done: |db| db.has_property_index("city"),
            image_needle: Some("city"),
        },
        DirectCall {
            name: "drop_property_index",
            setup: |db| db.create_property_index("name").unwrap(),
            call: |db| {
                db.drop_property_index("name").unwrap();
            },
            done: |db| !db.has_property_index("name"),
            image_needle: None,
        },
    ];
    #[cfg(feature = "vector-index")]
    let calls = calls.into_iter().chain([
        DirectCall {
            name: "create_vector_index",
            setup: |_| {},
            call: |db| {
                db.create_vector_index("Person", "vec3", Some(3), None, None, None, None)
                    .unwrap();
            },
            done: |db| db.store().get_vector_index("Person", "vec3").is_some(),
            image_needle: Some("vec3"),
        },
        DirectCall {
            name: "drop_vector_index",
            setup: |db| {
                db.create_vector_index("Person", "vec3", Some(3), None, None, None, None)
                    .unwrap();
            },
            call: |db| {
                db.drop_vector_index("Person", "vec3").unwrap();
            },
            done: |db| db.store().get_vector_index("Person", "vec3").is_none(),
            image_needle: None,
        },
    ]);
    #[cfg(feature = "text-index")]
    let calls = calls.into_iter().chain([
        DirectCall {
            name: "create_text_index",
            setup: |_| {},
            call: |db| db.create_text_index("Person", "bio").unwrap(),
            done: |db| db.store().get_text_index("Person", "bio").is_some(),
            image_needle: Some("bio"),
        },
        DirectCall {
            name: "drop_text_index",
            setup: |db| db.create_text_index("Person", "bio").unwrap(),
            call: |db| {
                db.drop_text_index("Person", "bio").unwrap();
            },
            done: |db| db.store().get_text_index("Person", "bio").is_none(),
            image_needle: None,
        },
    ]);
    calls.into_iter().collect()
}

/// The direct graph and index calls hold commits off while they change the
/// store, so a `close()` started from a hook inside one (once it holds them,
/// before it changes anything) waits for it, and the final checkpoint holds
/// the change: the reopened database shows it.
#[cfg(feature = "testing-statement-injection")]
#[test]
fn close_waits_for_a_direct_graph_or_index_call_holding_commits_off() {
    use std::sync::{Arc, mpsc};

    use grafeo_common::testing::commit_hook::during_next_held_change;

    // Every call is checked, and every failure listed at the end.
    let mut failures = Vec::new();
    for call in direct_calls() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("amsterdam.grafeo");
        database_with_alix(&path);
        let db = Arc::new(open(&path));
        (call.setup)(&db);
        let closer = Arc::clone(&db);
        let (sender, started) = mpsc::channel();
        during_next_held_change(move || {
            let close = Started::spawn(move || closer.close().map_err(|e| e.to_string()));
            let finished = close.finishes_briefly();
            sender.send((close, finished)).unwrap();
        });
        (call.call)(&db);

        // The hook runs inside the call, so its message is there now.
        let (close, finished) = started
            .try_recv()
            .unwrap_or_else(|_| panic!("{}: the call ran the hook", call.name));
        if finished {
            failures.push(format!("{}: close() did not wait for the call", call.name));
        }
        close.join().unwrap();
        drop(db);

        let reopened = open(&path);
        if !(call.done)(&reopened) {
            failures.push(format!("{}: the reopened file lacks the change", call.name));
        }
        reopened.close().unwrap();
    }
    assert!(failures.is_empty(), "{failures:#?}");
}

/// A checkpoint that starts while a direct graph or index call holds commits
/// off waits for it, and its image holds what the call created.
#[cfg(feature = "testing-statement-injection")]
#[test]
fn a_checkpoint_waits_for_a_direct_graph_or_index_call_and_holds_all_of_it() {
    use std::sync::{Arc, mpsc};

    use grafeo_common::testing::commit_hook::during_next_held_change;

    // Every call is checked, and every failure listed at the end.
    let mut failures = Vec::new();
    for call in direct_calls() {
        let Some(needle) = call.image_needle else {
            continue;
        };
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("berlin.grafeo");
        database_with_alix(&path);
        let db = Arc::new(open(&path));
        (call.setup)(&db);
        let checkpointer = Arc::clone(&db);
        let (sender, started) = mpsc::channel();
        during_next_held_change(move || {
            let checkpoint =
                Started::spawn(move || checkpointer.wal_checkpoint().map_err(|e| e.to_string()));
            let finished = checkpoint.finishes_briefly();
            sender.send((checkpoint, finished)).unwrap();
        });
        (call.call)(&db);

        // The hook runs inside the call, so its message is there now.
        let (checkpoint, finished) = started
            .try_recv()
            .unwrap_or_else(|_| panic!("{}: the call ran the hook", call.name));
        if finished {
            failures.push(format!(
                "{}: the checkpoint did not wait for the call",
                call.name
            ));
        }
        checkpoint.join().unwrap();
        if !image_holds(&db, needle) {
            failures.push(format!(
                "{}: the checkpoint image lacks {needle}",
                call.name
            ));
        }
        db.close().unwrap();
    }
    assert!(failures.is_empty(), "{failures:#?}");
}
