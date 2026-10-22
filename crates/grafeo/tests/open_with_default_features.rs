//! `GrafeoDB::open` with the facade's default features (#379): a dependency on
//! `grafeo` without a feature list opens a database file, and what it writes
//! is there after a reopen, for writing and read-only. Without the `wal`
//! feature (not in the defaults) a commit reaches the file at a checkpoint,
//! so a process that exits without one loses the commits since; with it, the
//! reopen replays them.
//!
//! ```bash
//! cargo test -p grafeo --test open_with_default_features
//! ```

#![cfg(feature = "grafeo-file")]

use std::path::{Path, PathBuf};

use grafeo::GrafeoDB;
use grafeo_common::testing::child_process;
use grafeo_common::types::Value;

/// Tells [`exit_without_closing_child`] where its database is.
const PATH_VAR: &str = "GRAFEO_OPEN_DEFAULT_FEATURES_PATH";

/// A new directory for one test's files.
fn test_dir(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("grafeo-{name}-{}", std::process::id()));
    // A run that failed leaves its files behind.
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

/// The names of the people in `db`, sorted.
fn people(db: &GrafeoDB) -> Vec<Value> {
    db.execute("MATCH (p:Person) RETURN p.name ORDER BY p.name")
        .unwrap()
        .rows()
        .iter()
        .map(|row| row[0].clone())
        .collect()
}

#[test]
fn open_keeps_what_it_wrote_across_a_reopen() {
    let dir = test_dir("open-default");
    let path = dir.join("amsterdam.grafeo");

    let db = GrafeoDB::open(&path).unwrap();
    db.execute("INSERT (:Person {name: 'Alix'})").unwrap();
    drop(db);

    let db = GrafeoDB::open(&path).unwrap();
    assert_eq!(db.node_count().unwrap(), 1);
    assert_eq!(people(&db), [Value::from("Alix")]);
    db.execute("INSERT (:Person {name: 'Gus'})").unwrap();
    db.close().unwrap();

    let db = GrafeoDB::open_read_only(&path).unwrap();
    assert_eq!(db.node_count().unwrap(), 2);
    assert_eq!(people(&db), [Value::from("Alix"), Value::from("Gus")]);
    drop(db);
    std::fs::remove_dir_all(&dir).unwrap();
}

/// Child-process entry for [`a_process_that_exits_keeps_what_it_checkpointed`];
/// a no-op when run directly. Commits Alix, checkpoints, commits Gus and exits
/// without closing.
#[test]
fn exit_without_closing_child() {
    let Some(path) = std::env::var_os(PATH_VAR) else {
        return;
    };
    let db = GrafeoDB::open(Path::new(&path)).unwrap();
    db.execute("INSERT (:Person {name: 'Alix'})").unwrap();
    db.wal_checkpoint().unwrap();
    db.execute("INSERT (:Person {name: 'Gus'})").unwrap();
    // A crash: no close(), no checkpoint, no destructors.
    std::process::exit(0);
}

#[test]
fn a_process_that_exits_keeps_what_it_checkpointed() {
    let dir = test_dir("open-default-exit");
    let path = dir.join("berlin.grafeo");
    let status = child_process::run(
        std::process::Command::new(std::env::current_exe().unwrap())
            .args(["--exact", "exit_without_closing_child", "--nocapture"])
            .env(PATH_VAR, &path),
    )
    .unwrap();
    assert!(status.success(), "the child process failed");

    let expected: &[&str] = if cfg!(feature = "wal") {
        &["Alix", "Gus"]
    } else {
        &["Alix"]
    };
    let expected: Vec<Value> = expected.iter().map(|name| Value::from(*name)).collect();
    let db = GrafeoDB::open_read_only(&path).unwrap();
    assert_eq!(people(&db), expected, "read-only");
    drop(db);
    let db = GrafeoDB::open(&path).unwrap();
    assert_eq!(people(&db), expected);
    db.close().unwrap();
    std::fs::remove_dir_all(&dir).unwrap();
}
