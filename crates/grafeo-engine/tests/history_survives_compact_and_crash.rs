//! History kept forever survives `compact()` and a crash before `close()` (#712).
//!
//! A child process writes values with history kept forever, overwrites one and deletes a
//! node, runs `compact()`, and exits with the database still open, so no clean close runs.
//! The parent checks the WAL is still there, reopens the file and reads at the epoch before
//! the overwrite and the delete: the old value and the deleted node are there, as after a
//! clean close (the control).
//!
//! ```bash
//! cargo test -p grafeo-engine --all-features --test main history_survives_compact_and_crash::
//! ```

#![cfg(all(feature = "gql", feature = "wal", feature = "grafeo-file"))]

use std::path::{Path, PathBuf};

use grafeo_common::testing::child_process;
use grafeo_common::types::{EpochId, NodeId, Value};
use grafeo_engine::config::{DurabilityMode, HistoryRetention};
use grafeo_engine::{Config, GrafeoDB};

/// The database path the child works on.
const PATH_VAR: &str = "GRAFEO_HISTORY_COMPACT_CRASH_PATH";
/// How the child ends: `crash` (exit with the database open) or `close`.
const END_VAR: &str = "GRAFEO_HISTORY_COMPACT_CRASH_END";
/// Exit code of a child that reached its end.
const EXITED: i32 = 23;

fn open(path: &Path) -> GrafeoDB {
    GrafeoDB::with_config(
        Config::persistent(path)
            .with_wal_durability(DurabilityMode::Sync)
            .with_history_retention(HistoryRetention::Forever),
    )
    .unwrap()
}

fn sidecar_wal(path: &Path) -> PathBuf {
    let mut sidecar = path.as_os_str().to_owned();
    sidecar.push(".wal");
    PathBuf::from(sidecar)
}

/// What the child wrote: the node it updated, the node it deleted, and the epoch both were
/// as first written. Printed by the child, read by the parent.
struct Written {
    kept: NodeId,
    deleted: NodeId,
    before: EpochId,
}

/// Child-process entry for [`run_child`]; a no-op when run directly.
#[test]
fn history_compact_child() {
    let (Some(path), Ok(end)) = (std::env::var_os(PATH_VAR), std::env::var(END_VAR)) else {
        return;
    };
    let db = open(Path::new(&path));
    let kept = db.create_node(&["City"]).unwrap();
    db.set_node_property(kept, "name", Value::from("Amsterdam"))
        .unwrap();
    let deleted = db.create_node(&["City"]).unwrap();
    db.set_node_property(deleted, "name", Value::from("Utrecht"))
        .unwrap();
    let before = db.current_epoch();
    db.set_node_property(kept, "name", Value::from("Haarlem"))
        .unwrap();
    db.execute("MATCH (c:City {name: 'Utrecht'}) DETACH DELETE c")
        .unwrap();
    // `compact()` takes `&mut self`.
    let mut db = db;
    db.compact().unwrap();
    println!(
        "WRITTEN {} {} {}",
        kept.as_u64(),
        deleted.as_u64(),
        before.as_u64()
    );
    if end == "close" {
        db.close().unwrap();
    }
    std::process::exit(EXITED);
}

/// Runs the child on a fresh database at `path`, ending it by `end`, and returns what it
/// wrote.
fn run_child(path: &Path, end: &str) -> Written {
    let output = child_process::output(
        grafeo_common::child_test!("history_compact_child")
            .env(PATH_VAR, path)
            .env(END_VAR, end),
    )
    .unwrap();
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert_eq!(
        output.status.code(),
        Some(EXITED),
        "the child ({end}) exited early:\n{stdout}\n{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let line = stdout
        .lines()
        .find_map(|line| line.strip_prefix("WRITTEN "))
        .expect("the child printed what it wrote");
    let numbers: Vec<u64> = line
        .split(' ')
        .map(|number| number.parse().unwrap())
        .collect();
    Written {
        kept: NodeId::new(numbers[0]),
        deleted: NodeId::new(numbers[1]),
        before: EpochId::new(numbers[2]),
    }
}

/// Reopens `path` and checks the history the child wrote is there.
fn assert_history(path: &Path, written: &Written, end: &str) {
    let db = open(path);
    assert_eq!(
        db.get_node_property_at_epoch(written.kept, "name", db.current_epoch())
            .unwrap(),
        Some(Value::from("Haarlem")),
        "{end}: the latest value"
    );
    assert_eq!(
        db.get_node_property_at_epoch(written.kept, "name", written.before)
            .unwrap(),
        Some(Value::from("Amsterdam")),
        "{end}: the value before the overwrite"
    );
    assert!(
        db.get_node(written.deleted).unwrap().is_none(),
        "{end}: the deleted node is gone now"
    );
    assert!(
        db.get_node_at_epoch(written.deleted, written.before)
            .unwrap()
            .is_some(),
        "{end}: the deleted node at the epoch before the delete"
    );
    db.close().unwrap();
}

/// The control: a clean close after `compact()` keeps the history.
#[test]
fn history_survives_compact_and_close() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("history.grafeo");
    let written = run_child(&path, "close");
    assert!(!sidecar_wal(&path).exists(), "close: the WAL is removed");
    assert_history(&path, &written, "close");
}

/// A crash after `compact()`, before `close()`, keeps the history.
#[test]
fn history_survives_compact_and_a_crash() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("history.grafeo");
    let written = run_child(&path, "crash");
    assert!(
        sidecar_wal(&path).exists(),
        "crash: the child closed its database, no WAL is left"
    );
    assert_history(&path, &written, "crash");
}
