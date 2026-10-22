//! An incremental backup chain survives `close()` and a crash.
//!
//! The backup cursor and the WAL segments an incremental backup reads live
//! in the sidecar WAL directory. `close()` removed that directory, so after
//! a restart every incremental backup failed with "no backup cursor found"
//! until the next full backup. Now, once a backup chain exists (the sidecar
//! holds a cursor), `close()` keeps the segments from the cursor's log
//! position on (the checkpoint already keeps them) and the cursor; the next
//! incremental backup takes them, and the next checkpoint lets them go. A
//! database that never takes a backup still loses its sidecar at `close()`.
//!
//! ```bash
//! cargo test -p grafeo-engine --all-features --test backup_chain_across_close
//! ```

#![cfg(all(
    feature = "wal",
    feature = "grafeo-file",
    feature = "lpg",
    feature = "gql"
))]

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::process::Command;

use grafeo_common::testing::child_process;
use grafeo_common::types::{EpochId, Value};
use grafeo_engine::GrafeoDB;
use grafeo_engine::database::backup::read_backup_cursor;
use grafeo_storage::wal::parse_segment_file_name;

const STEP_VAR: &str = "GRAFEO_BACKUP_CHAIN_STEP";
const PATH_VAR: &str = "GRAFEO_BACKUP_CHAIN_PATH";

fn backups(path: &Path) -> PathBuf {
    path.with_file_name("backups")
}

fn sidecar(path: &Path) -> PathBuf {
    let mut name = path.as_os_str().to_owned();
    name.push(".wal");
    PathBuf::from(name)
}

fn names(db: &GrafeoDB) -> Vec<Value> {
    db.execute("MATCH (p:Person) RETURN p.name AS name ORDER BY name")
        .unwrap()
        .rows()
        .iter()
        .map(|row| row[0].clone())
        .collect()
}

fn strings(values: &[&str]) -> Vec<Value> {
    values.iter().map(|value| Value::from(*value)).collect()
}

/// The people of the database restored from `backups` to `epoch`.
fn restored(backups: &Path, epoch: EpochId) -> Vec<Value> {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("restored.grafeo");
    GrafeoDB::restore_to_epoch(backups, epoch, &path).unwrap();
    let db = GrafeoDB::open(&path).unwrap();
    let found = names(&db);
    db.close().unwrap();
    found
}

/// Every file under `dir` with its bytes.
fn files(dir: &Path) -> BTreeMap<PathBuf, Vec<u8>> {
    let mut found = BTreeMap::new();
    for entry in std::fs::read_dir(dir).unwrap() {
        let path = entry.unwrap().path();
        if path.is_file() {
            found.insert(path.clone(), std::fs::read(&path).unwrap());
        }
    }
    found
}

/// The first log positions of the segments in the sidecar WAL of `path`.
fn segments(path: &Path) -> Vec<u64> {
    let mut found: Vec<u64> = std::fs::read_dir(sidecar(path))
        .unwrap()
        .filter_map(|entry| parse_segment_file_name(&entry.unwrap().file_name().to_string_lossy()))
        .collect();
    found.sort_unstable();
    found
}

/// Runs `step` on the database at `path` in a child process, which opens it,
/// does the step and closes it (or exits without closing, for `crash`).
fn in_child(step: &str, path: &Path) {
    let status = child_process::run(
        Command::new(std::env::current_exe().unwrap())
            .args(["--exact", "step_child", "--nocapture"])
            .env(STEP_VAR, step)
            .env(PATH_VAR, path),
    )
    .unwrap();
    assert!(status.success(), "step {step} failed");
}

/// Child-process entry for [`in_child`]; a no-op when run directly.
#[test]
fn step_child() {
    let Ok(step) = std::env::var(STEP_VAR) else {
        return;
    };
    let path = PathBuf::from(std::env::var_os(PATH_VAR).unwrap());
    let db = GrafeoDB::open(&path).unwrap();
    match step.as_str() {
        "insert_alix" => {
            db.execute("INSERT (:Person {name: 'Alix'})").unwrap();
        }
        "full" => {
            db.backup_full(&backups(&path)).unwrap();
        }
        "insert_gus" => {
            db.execute("INSERT (:Person {name: 'Gus'})").unwrap();
        }
        "incremental" => {
            db.backup_incremental(&backups(&path)).unwrap();
        }
        "insert_vincent_and_crash" => {
            db.execute("INSERT (:Person {name: 'Vincent'})").unwrap();
            // Crash: no close(), no checkpoint.
            std::process::exit(0);
        }
        other => panic!("unknown step {other}"),
    }
    db.close().unwrap();
}

/// The reporter's sequence, each step in its own process (as the CLI and a
/// restarted server run them): a full backup, a write, an incremental
/// backup. Each backup restores exactly the people of its epoch.
#[test]
fn an_incremental_backup_works_across_processes() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("db.grafeo");
    for step in ["insert_alix", "full", "insert_gus", "incremental"] {
        in_child(step, &path);
    }
    let manifest = GrafeoDB::read_backup_manifest(&backups(&path))
        .unwrap()
        .unwrap();
    let epochs: Vec<EpochId> = manifest.segments.iter().map(|s| s.end_epoch).collect();
    assert_eq!(epochs.len(), 2, "a full and an incremental backup");
    assert_eq!(restored(&backups(&path), epochs[0]), strings(&["Alix"]));
    assert_eq!(
        restored(&backups(&path), epochs[1]),
        strings(&["Alix", "Gus"])
    );
}

/// A crash after a close kept the chain's segments: the reopen replays only
/// what the file does not hold (every person once), and the incremental
/// backup after it holds the commits since the full backup, the crashed
/// process's among them.
#[test]
fn a_crash_while_segments_are_kept_replays_each_commit_once() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("db.grafeo");
    for step in [
        "insert_alix",
        "full",
        "insert_gus",
        "insert_vincent_and_crash",
    ] {
        in_child(step, &path);
    }
    let db = GrafeoDB::open(&path).unwrap();
    assert_eq!(names(&db), strings(&["Alix", "Gus", "Vincent"]));
    assert_eq!(
        db.execute("MATCH (p:Person) RETURN count(p), count(DISTINCT id(p))")
            .unwrap()
            .rows(),
        [[Value::Int64(3), Value::Int64(3)]]
    );
    let epoch = db.backup_incremental(&backups(&path)).unwrap().end_epoch;
    db.close().unwrap();
    assert_eq!(
        restored(&backups(&path), epoch),
        strings(&["Alix", "Gus", "Vincent"])
    );
}

/// Between two incremental backups the kept WAL holds only what was written
/// since the last one: each close lets the backed-up segments go, so the
/// sidecar does not grow with the number of rounds.
#[test]
fn the_kept_wal_is_bounded_by_the_writes_since_the_last_backup() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("db.grafeo");
    let backups = backups(&path);
    let db = GrafeoDB::open(&path).unwrap();
    db.backup_full(&backups).unwrap();
    db.close().unwrap();
    drop(db);

    let mut sizes = Vec::new();
    for round in 0..6_i64 {
        let db = GrafeoDB::open(&path).unwrap();
        db.execute(&format!(
            "UNWIND range(1, 200) AS i INSERT (:Person {{name: 'p' || CAST({round} * 1000 + i AS STRING)}})"
        ))
        .unwrap();
        db.backup_incremental(&backups).unwrap();
        db.close().unwrap();
        drop(db);
        let cursor = read_backup_cursor(&sidecar(&path)).unwrap().unwrap();
        let from = cursor.wal_lsn.unwrap();
        assert!(
            segments(&path).iter().all(|&first| first >= from),
            "round {round}: only segments from the cursor at {from} are kept: {:?}",
            segments(&path)
        );
        let bytes: u64 = files(&sidecar(&path))
            .values()
            .map(|bytes| bytes.len() as u64)
            .sum();
        sizes.push(bytes);
    }
    assert!(
        sizes.iter().all(|&size| size <= sizes[0] * 2),
        "the kept WAL stays the size of one round: {sizes:?}"
    );
    let db = GrafeoDB::open(&path).unwrap();
    assert_eq!(
        db.execute("MATCH (p:Person) RETURN count(p)")
            .unwrap()
            .rows(),
        [[Value::Int64(1200)]]
    );
    db.close().unwrap();
}

/// A read-only open of a database whose close kept the chain's segments
/// changes no file of the sidecar, and the next incremental backup works.
#[test]
fn a_read_only_open_leaves_the_kept_segments_alone() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("db.grafeo");
    for step in ["insert_alix", "full", "insert_gus"] {
        in_child(step, &path);
    }
    let before = (files(&sidecar(&path)), std::fs::read(&path).unwrap());
    let db = GrafeoDB::open_read_only(&path).unwrap();
    assert_eq!(names(&db), strings(&["Alix", "Gus"]));
    db.close().unwrap();
    drop(db);
    assert!(
        (files(&sidecar(&path)), std::fs::read(&path).unwrap()) == before,
        "a read-only open writes nothing"
    );
    in_child("incremental", &path);
    let manifest = GrafeoDB::read_backup_manifest(&backups(&path))
        .unwrap()
        .unwrap();
    let last = manifest.segments.last().unwrap().end_epoch;
    assert_eq!(restored(&backups(&path), last), strings(&["Alix", "Gus"]));
}

/// A new full backup starts the chain over: its cursor moves to its own log
/// position, and the next close lets the older segments go.
#[test]
fn a_full_backup_lets_the_older_segments_go() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("db.grafeo");
    for step in ["insert_alix", "full", "insert_gus", "full"] {
        in_child(step, &path);
    }
    let cursor = read_backup_cursor(&sidecar(&path)).unwrap().unwrap();
    let from = cursor.wal_lsn.unwrap();
    assert!(
        segments(&path).iter().all(|&first| first >= from),
        "only segments from the second full backup's cursor at {from}: {:?}",
        segments(&path)
    );
    let manifest = GrafeoDB::read_backup_manifest(&backups(&path))
        .unwrap()
        .unwrap();
    let latest = manifest.latest_full().unwrap().end_epoch;
    assert_eq!(restored(&backups(&path), latest), strings(&["Alix", "Gus"]));
}

/// A database that never took a backup loses its sidecar at `close()`, as
/// before, and keeps its data.
#[test]
fn without_backups_close_removes_the_sidecar() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("db.grafeo");
    for step in ["insert_alix", "insert_gus"] {
        in_child(step, &path);
        assert!(!sidecar(&path).exists(), "after {step}: no sidecar");
    }
    let db = GrafeoDB::open_read_only(&path).unwrap();
    assert_eq!(names(&db), strings(&["Alix", "Gus"]));
}
