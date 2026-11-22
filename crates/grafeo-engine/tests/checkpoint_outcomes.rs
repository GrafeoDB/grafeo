//! A checkpoint's outcome is decided by its header. A failure before the header write stores
//! nothing; one from the header write up to the header's sync leaves the outcome unknown until
//! a reopen, so the file manager takes no more checkpoints and the caller keeps its new state;
//! one after the sync (the trim) fails nothing. `set_cdc`, `set_stored_config` and
//! `restore_snapshot` agree with what a reopen finds at every failure point, and a database
//! whose checkpoint outcome is unknown closes, and drops, without a checkpoint and keeps its WAL.
//!
//! ```bash
//! cargo test -p grafeo-engine --all-features --test main checkpoint_outcomes::
//! ```

#![cfg(all(
    feature = "cdc",
    feature = "gql",
    feature = "wal",
    feature = "testing-crash-injection"
))]

use std::path::{Path, PathBuf};
use std::time::Duration;

use grafeo_common::testing::child_process;
use grafeo_common::testing::crash::{fired_failure, with_failure_at};
use grafeo_common::utils::error::{Error, StorageError, TransactionError};
use grafeo_engine::config::DurabilityMode;
use grafeo_engine::{Config, GrafeoDB};
use grafeo_storage::wal::list_wal_directory;

const MIB: usize = 1024 * 1024;
/// Whether `error` is the typed outcome-unknown error (GRAFEO-S008): an earlier checkpoint
/// failed between its header write and the header's sync, or this one did.
fn is_outcome_unknown(error: &Error) -> bool {
    matches!(error, Error::Storage(storage)
        if matches!(**storage, StorageError::CheckpointOutcomeUnknown(_)))
}

fn sidecar(path: &Path) -> PathBuf {
    PathBuf::from(format!("{}.wal", path.display()))
}

/// Batch durability: no background flusher writes to the WAL while a test compares it.
fn database(path: &Path) -> GrafeoDB {
    GrafeoDB::with_config(
        Config::persistent(path)
            .with_wal_durability(DurabilityMode::batch(Duration::from_millis(100), 1000)),
    )
    .unwrap()
}

fn write_batch(db: &GrafeoDB, from: i64) {
    db.execute(&format!(
        "UNWIND range({from}, {}) AS i INSERT (:Person {{i: i}})",
        from + 99
    ))
    .unwrap();
}

fn count(db: &GrafeoDB, label: &str) -> usize {
    db.execute(&format!("MATCH (n:{label}) RETURN n"))
        .unwrap()
        .rows()
        .len()
}

fn cursor(path: &Path) -> PathBuf {
    sidecar(path).join("cdc_cursor_default.json")
}

fn cursor_lsn(path: &Path) -> u64 {
    let text = std::fs::read_to_string(cursor(path)).unwrap();
    let json: serde_json::Value = serde_json::from_str(&text).unwrap();
    json["lsn"].as_u64().unwrap()
}

fn first_kept(path: &Path) -> u64 {
    list_wal_directory(&sidecar(path)).unwrap().segments[0].0
}

/// The WAL segments of `path`'s sidecar.
fn segments(path: &Path) -> Vec<PathBuf> {
    list_wal_directory(&sidecar(path))
        .unwrap()
        .segments
        .into_iter()
        .map(|(_, segment)| segment)
        .collect()
}

/// Calls `call` on a new database from `prepare` at each failure count until the failure lands
/// on `point`; returns the database, its path (in `dir`) and the result.
fn failing_at<T>(
    dir: &Path,
    point: &str,
    prepare: impl Fn(&Path) -> GrafeoDB,
    call: impl Fn(&GrafeoDB) -> T,
) -> (GrafeoDB, PathBuf, T) {
    for count in 1..=256 {
        let path = dir.join(format!("db{count}.grafeo"));
        let db = prepare(&path);
        let result = with_failure_at(count, || call(&db));
        if fired_failure().as_deref() == Some(point) {
            return (db, path, result);
        }
        assert!(fired_failure().is_some(), "no failure landed at {point}");
        // A failure that left the WAL or the checkpoints refusing makes the close say so.
        drop(db.close());
    }
    panic!("no failure landed at {point}")
}

/// A database with change data capture stored on and a few checkpoints of log behind its cursor.
fn cdc_on_with_log(path: &Path) -> GrafeoDB {
    let db = database(path);
    db.set_cdc(true).unwrap();
    for round in 0..2 {
        write_batch(&db, round * 100);
        db.wal_checkpoint().unwrap();
    }
    write_batch(&db, 1_000);
    db
}

/// A trim that fails after the header's sync fails no toggle: `set_cdc(false)` returns Ok, and
/// a reopen has the setting off.
#[test]
fn a_failed_trim_after_the_header_sync_fails_no_toggle() {
    for point in ["checkpoint:before_trim", "checkpoint:trim"] {
        let dir = tempfile::tempdir().unwrap();
        let (db, path, result) =
            failing_at(dir.path(), point, cdc_on_with_log, |db| db.set_cdc(false));
        result.unwrap_or_else(|error| panic!("{point}: {error}"));
        assert!(!db.is_cdc_enabled(), "{point}");
        db.close().unwrap();
        drop(db);
        let db = database(&path);
        assert!(!db.is_cdc_enabled(), "{point}: the reopen has it off");
        assert!(!cursor(&path).exists(), "{point}");
        db.close().unwrap();
    }
}

/// A failure between the header write and its sync leaves the outcome unknown: `set_cdc(true)`
/// returns the unknown-outcome I/O error and keeps capture on in memory; every later checkpoint
/// and `set_cdc` is refused until a reopen. `close()` (or `Drop`, without `close()`) then runs
/// no checkpoint and removes no WAL, and the reopen finds one of the two images whole, with the
/// WAL replayed on top. The database is built outside the injection closure.
#[test]
fn a_failure_before_the_header_sync_poisons_the_checkpoints_until_a_reopen() {
    for point in ["checkpoint:after_header", "checkpoint:header_sync"] {
        for ending in ["close", "drop"] {
            let what = format!("{point}, {ending}");
            let dir = tempfile::tempdir().unwrap();
            let prepare = |path: &Path| {
                let db = database(path);
                write_batch(&db, 0);
                db
            };
            let (db, path, result) = failing_at(dir.path(), point, prepare, |db| db.set_cdc(true));
            let error = result.expect_err(&what);
            assert!(is_outcome_unknown(&error), "{what}: {error:?}");
            assert_eq!(error.error_code().as_str(), "GRAFEO-S008", "{what}");
            assert!(!error.error_code().is_retryable(), "{what}");
            assert!(
                db.is_cdc_enabled(),
                "{what}: the new setting is kept in memory"
            );

            let snapshot = {
                let cities = GrafeoDB::new_in_memory();
                cities.execute("INSERT (:City {i: 1})").unwrap();
                cities.export_snapshot().unwrap()
            };
            let segments_before_refusals = segments(&path);
            for refused in [
                db.wal_checkpoint().map(drop),
                db.set_cdc(false),
                db.set_stored_config(&Config::in_memory().with_memory_limit(48 * MIB)),
                db.restore_snapshot(&snapshot),
            ] {
                let error = refused.expect_err(&what);
                assert!(is_outcome_unknown(&error), "{what}: {error:?}");
            }
            assert_eq!(
                segments(&path),
                segments_before_refusals,
                "{what}: a refused checkpoint starts no WAL segment"
            );
            assert!(
                db.is_cdc_enabled(),
                "{what}: a refused set_cdc changes nothing"
            );
            assert_eq!(
                (count(&db, "Person"), count(&db, "City")),
                (100, 0),
                "{what}: a refused restore replaces nothing"
            );

            write_batch(&db, 1_000);
            let file_before = dir.path().join("before.grafeo");
            db.file_manager().unwrap().copy_to(&file_before).unwrap();
            let segments_before = segments(&path);
            if ending == "close" {
                let error = db.close().expect_err(&what);
                assert!(is_outcome_unknown(&error), "{what}: {error:?}");
                let after = db.execute("MATCH (n) RETURN n").map(drop).expect_err(&what);
                assert_eq!(
                    after.error_code().as_str(),
                    "GRAFEO-T007",
                    "{what}: the close closed the database: {after}"
                );
            }
            drop(db);
            assert_eq!(
                std::fs::read(&path).unwrap(),
                std::fs::read(&file_before).unwrap(),
                "{what}: no checkpoint ran"
            );
            assert!(
                segments_before.iter().all(|segment| segment.exists()),
                "{what}: the WAL is kept"
            );

            let db = database(&path);
            assert_eq!(count(&db, "Person"), 200, "{what}: every commit, once");
            if db.is_cdc_enabled() {
                assert!(cursor(&path).exists(), "{what}");
                assert!(first_kept(&path) <= cursor_lsn(&path), "{what}");
            } else {
                assert!(!cursor(&path).exists(), "{what}");
            }
            db.close().unwrap();
        }
    }
}

/// A checkpoint refused after an unknown outcome starts no WAL segment: commits made since
/// stay in the segment they were written to.
#[test]
fn a_refused_checkpoint_starts_no_wal_segment() {
    let dir = tempfile::tempdir().unwrap();
    let prepare = |path: &Path| {
        let db = database(path);
        write_batch(&db, 0);
        db
    };
    let (db, path, result) = failing_at(dir.path(), "checkpoint:after_header", prepare, |db| {
        db.set_cdc(true)
    });
    assert!(is_outcome_unknown(&result.expect_err("unknown")));
    write_batch(&db, 1_000);
    let before = segments(&path);
    let refused = db.wal_checkpoint().expect_err("refused");
    assert!(is_outcome_unknown(&refused), "{refused:?}");
    assert_eq!(segments(&path), before, "no new segment");
    drop(db.close());
}

/// A database without a WAL refuses commits once a checkpoint's outcome is unknown: a commit
/// would be durable only at a checkpoint, and none can run until a reopen. Reads go on, a copy
/// holds the state in memory, and the reopen shows the last image whole.
#[test]
fn a_database_without_a_wal_refuses_commits_after_an_unknown_outcome() {
    let dir = tempfile::tempdir().unwrap();
    let prepare = |path: &Path| {
        let db = GrafeoDB::with_config(Config::persistent(path).without_wal()).unwrap();
        write_batch(&db, 0);
        db.wal_checkpoint().unwrap();
        db
    };
    let (db, path, result) = failing_at(dir.path(), "checkpoint:after_header", prepare, |db| {
        db.set_stored_config(&Config::in_memory().with_memory_limit(48 * MIB))
    });
    assert!(is_outcome_unknown(&result.expect_err("unknown")));

    let refused = db
        .execute("INSERT (:Person {i: 9000})")
        .map(drop)
        .expect_err("a commit is refused");
    assert!(is_outcome_unknown(&refused), "{refused:?}");
    let refused = db
        .create_node(&["Person"])
        .map(drop)
        .expect_err("a direct write is refused");
    assert!(is_outcome_unknown(&refused), "{refused:?}");
    assert_eq!(count(&db, "Person"), 100, "reads go on");

    let saved = dir.path().join("saved.grafeo");
    db.save(&saved).unwrap();
    drop(db.close());
    drop(db);
    let copy = GrafeoDB::with_config(Config::persistent(&saved).without_wal()).unwrap();
    assert_eq!(
        count(&copy, "Person"),
        100,
        "the copy opens with the state in memory"
    );
    assert_eq!(
        copy.memory_limit(),
        Some(48 * MIB),
        "and the call's new value"
    );
    copy.close().unwrap();
    let source = GrafeoDB::with_config(Config::persistent(&path).without_wal()).unwrap();
    assert_eq!(count(&source, "Person"), 100, "the last image, whole");
    source.close().unwrap();
}

/// A restore whose checkpoint fails, before its header (the outcome known) or after it (the
/// outcome unknown), refuses commits until a reopen: a commit on top of the restored state in
/// memory would replay onto the previous image. The reopen gives one of the two states, whole.
#[test]
fn a_failed_restore_refuses_commits_until_a_reopen() {
    for point in ["checkpoint:after_chunks", "checkpoint:after_header"] {
        let dir = tempfile::tempdir().unwrap();
        let snapshot = {
            let cities = GrafeoDB::new_in_memory();
            cities
                .execute("UNWIND range(1, 3) AS i INSERT (:City {i: i})")
                .unwrap();
            cities.export_snapshot().unwrap()
        };
        let prepare = |path: &Path| {
            let db = database(path);
            write_batch(&db, 0);
            db
        };
        let (db, path, result) = failing_at(dir.path(), point, prepare, |db| {
            db.restore_snapshot(&snapshot)
        });
        result.expect_err(point);
        let refused = db
            .execute("INSERT (:Person {i: 9000})")
            .map(drop)
            .expect_err(point);
        if point == "checkpoint:after_header" {
            assert!(is_outcome_unknown(&refused), "{point}: {refused:?}");
        } else {
            // The failure is known: GRAFEO-T008, naming the restore.
            assert!(
                matches!(
                    refused,
                    Error::Transaction(TransactionError::IncompleteCommit)
                ),
                "{point}: {refused:?}"
            );
            assert_eq!(refused.error_code().as_str(), "GRAFEO-T008");
            assert!(
                refused.to_string().contains("restore"),
                "{point}: {refused}"
            );
        }
        drop(db.close());
        drop(db);
        let db = database(&path);
        let state = (count(&db, "Person"), count(&db, "City"));
        assert!(state == (100, 0) || state == (0, 3), "{point}: {state:?}");
        db.close().unwrap();
    }
}

/// The file and the WAL of `path` as a crash right now would leave them, at `copy`: the manager's
/// file through its handle (locked on Windows) and the sidecar's files.
fn crash_copy(db: &GrafeoDB, path: &Path, copy: &Path) {
    db.file_manager().unwrap().copy_to(copy).unwrap();
    let to = sidecar(copy);
    std::fs::create_dir_all(&to).unwrap();
    for entry in std::fs::read_dir(sidecar(path)).unwrap() {
        let entry = entry.unwrap();
        std::fs::copy(entry.path(), to.join(entry.file_name())).unwrap();
    }
}

/// A graph drop is refused as a commit is: on a database without a WAL after an unknown outcome
/// (S008) and after a restore whose checkpoint failed (T008). The graph is still there after the
/// reopen, as a crash would leave the database.
#[test]
fn a_graph_drop_is_refused_as_a_commit_is() {
    // Without a WAL, after an unknown outcome.
    let dir = tempfile::tempdir().unwrap();
    let prepare = |path: &Path| {
        let db = GrafeoDB::with_config(Config::persistent(path).without_wal()).unwrap();
        db.create_graph("berlin").unwrap();
        write_batch(&db, 0);
        db.wal_checkpoint().unwrap();
        db
    };
    let (db, path, result) = failing_at(dir.path(), "checkpoint:after_header", prepare, |db| {
        db.set_stored_config(&Config::in_memory().with_memory_limit(48 * MIB))
    });
    assert!(is_outcome_unknown(&result.expect_err("unknown")));
    let refused = db.drop_graph("berlin").expect_err("the drop is refused");
    assert!(is_outcome_unknown(&refused), "{refused:?}");
    let refused = db
        .execute("DROP GRAPH berlin")
        .map(drop)
        .expect_err("DROP GRAPH is refused");
    assert!(is_outcome_unknown(&refused), "{refused:?}");
    drop(db.close());
    drop(db);
    let db = GrafeoDB::with_config(Config::persistent(&path).without_wal()).unwrap();
    assert!(
        db.list_graphs().contains(&"berlin".to_string()),
        "the graph stays"
    );
    db.close().unwrap();

    // With a WAL, after a restore whose checkpoint failed.
    let dir = tempfile::tempdir().unwrap();
    let snapshot = {
        let cities = GrafeoDB::new_in_memory();
        cities.execute("INSERT (:City {i: 1})").unwrap();
        cities.export_snapshot().unwrap()
    };
    let prepare = |path: &Path| {
        let db = database(path);
        db.create_graph("berlin").unwrap();
        write_batch(&db, 0);
        db
    };
    let (db, path, result) = failing_at(dir.path(), "checkpoint:after_chunks", prepare, |db| {
        db.restore_snapshot(&snapshot)
    });
    result.expect_err("the restore failed");
    let refused = db.drop_graph("berlin").expect_err("the drop is refused");
    assert!(
        matches!(
            refused,
            Error::Transaction(TransactionError::IncompleteCommit)
        ),
        "{refused:?}"
    );
    let crashed = dir.path().join("crashed.grafeo");
    crash_copy(&db, &path, &crashed);
    drop(db.close());
    drop(db);
    let db = database(&crashed);
    assert!(
        db.list_graphs().contains(&"berlin".to_string()),
        "no drop was logged onto the previous image"
    );
    assert_eq!((count(&db, "Person"), count(&db, "City")), (100, 0));
    db.close().unwrap();
}

/// A commit from another session while a restore's checkpoint fails waits for the restore and is
/// then refused: none goes on between the restored stores and the outcome, so none replays onto
/// the previous image. The reopen, as a crash leaves the database, is the previous state whole.
#[cfg(feature = "testing-statement-injection")]
#[test]
fn a_commit_during_a_failing_restore_is_refused() {
    use std::sync::Arc;
    use std::thread::JoinHandle;

    use grafeo_common::testing::commit_hook::during_next_checkpoint_write;

    type Writer =
        Arc<parking_lot::Mutex<Option<JoinHandle<grafeo_common::utils::error::Result<()>>>>>;

    let snapshot = {
        let cities = GrafeoDB::new_in_memory();
        cities.execute("INSERT (:City {i: 1})").unwrap();
        cities.export_snapshot().unwrap()
    };
    for at in 1..=64 {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("db.grafeo");
        let db = Arc::new(database(&path));
        write_batch(&db, 0);
        let handle: Writer = Arc::default();
        let (slot, writer) = (Arc::clone(&handle), Arc::clone(&db));
        during_next_checkpoint_write(move || {
            *slot.lock() = Some(std::thread::spawn(move || {
                writer.execute("INSERT (:Person {i: 7777})").map(drop)
            }));
            // Long enough for a commit that is not held off to complete.
            std::thread::sleep(Duration::from_millis(300));
        });
        let result = with_failure_at(at, || db.restore_snapshot(&snapshot));
        let writer = handle.lock().take();
        if fired_failure().as_deref() != Some("checkpoint:after_chunks") {
            assert!(
                fired_failure().is_some(),
                "no failure landed after the hook"
            );
            if let Some(writer) = writer {
                drop(writer.join());
            }
            drop(db.close());
            continue;
        }
        result.expect_err("the restore failed");
        let committed = writer.expect("the hook ran").join().unwrap();
        let refused = committed.expect_err("a commit during a failing restore is refused");
        assert!(
            matches!(
                refused,
                Error::Transaction(TransactionError::IncompleteCommit)
            ),
            "{refused:?}"
        );
        let crashed = dir.path().join("crashed.grafeo");
        crash_copy(&db, &path, &crashed);
        drop(db.close());
        let reopened = database(&crashed);
        assert_eq!(
            (count(&reopened, "Person"), count(&reopened, "City")),
            (100, 0),
            "the previous state, whole"
        );
        reopened.close().unwrap();
        return;
    }
    panic!("no failure landed after the hook")
}

/// After an unknown outcome the automatic checkpoints stop, as after a commit that did not
/// complete: no checkpoint can succeed until a reopen, so the timer does not retry.
#[test]
fn the_automatic_checkpoints_stop_after_an_unknown_outcome() {
    let dir = tempfile::tempdir().unwrap();
    let prepare = |path: &Path| {
        let db = GrafeoDB::with_config(
            Config::persistent(path)
                .with_wal_durability(DurabilityMode::batch(Duration::from_millis(100), 1000))
                .with_checkpoint_interval(Duration::from_millis(50)),
        )
        .unwrap();
        write_batch(&db, 0);
        db
    };
    let (db, _, result) = failing_at(dir.path(), "checkpoint:after_header", prepare, |db| {
        db.set_cdc(true)
    });
    assert!(
        is_outcome_unknown(&result.expect_err("the outcome is unknown")),
        "the outcome is unknown"
    );
    write_batch(&db, 1_000);
    let deadline = std::time::Instant::now() + Duration::from_secs(10);
    while !db.checkpoint_status().stopped && std::time::Instant::now() < deadline {
        std::thread::sleep(Duration::from_millis(20));
    }
    let status = db.checkpoint_status();
    assert!(status.stopped, "the timer stopped: {status:?}");
    drop(db.close());
}

/// A database whose checkpoint outcome is unknown (through `set_cdc(true)`) refuses
/// `set_stored_config`, and a full backup taken next publishes the stored configuration the
/// database has, not the refused one: the backup, restored and opened, and the source,
/// reopened, both lack the refused memory limit.
#[test]
fn a_refused_stored_config_is_not_published_by_a_backup() {
    let dir = tempfile::tempdir().unwrap();
    let prepare = |path: &Path| {
        let db = database(path);
        write_batch(&db, 0);
        db
    };
    let (db, path, result) = failing_at(dir.path(), "checkpoint:after_header", prepare, |db| {
        db.set_cdc(true)
    });
    assert!(
        is_outcome_unknown(&result.expect_err("the outcome is unknown")),
        "the outcome is unknown"
    );
    db.set_stored_config(&Config::in_memory().with_memory_limit(48 * MIB))
        .expect_err("refused after an unknown outcome");
    let backups = dir.path().join("backups");
    let full = db.backup_full(&backups).unwrap();
    drop(db.close());
    drop(db);

    let restored = dir.path().join("restored.grafeo");
    GrafeoDB::restore_to_epoch(&backups, full.end_epoch, &restored).unwrap();
    let copy = database(&restored);
    assert_ne!(copy.memory_limit(), Some(48 * MIB), "the backup");
    copy.close().unwrap();
    let source = database(&path);
    assert_ne!(source.memory_limit(), Some(48 * MIB), "the source");
    source.close().unwrap();
}

/// `set_stored_config` whose checkpoint outcome is unknown keeps the new configuration in
/// memory, and a copy saved next holds that state and opens cleanly.
#[test]
fn a_copy_after_an_unknown_stored_config_holds_the_state_in_memory() {
    let dir = tempfile::tempdir().unwrap();
    let prepare = |path: &Path| {
        let db = database(path);
        write_batch(&db, 0);
        db
    };
    let (db, path, result) = failing_at(dir.path(), "checkpoint:after_header", prepare, |db| {
        db.set_stored_config(&Config::in_memory().with_memory_limit(48 * MIB))
    });
    assert!(
        is_outcome_unknown(&result.expect_err("the outcome is unknown")),
        "the outcome is unknown"
    );
    let saved = dir.path().join("saved.grafeo");
    db.save(&saved).unwrap();
    drop(db.close());
    drop(db);

    let copy = database(&saved);
    assert_eq!(copy.memory_limit(), Some(48 * MIB), "the call's new value");
    assert_eq!(count(&copy, "Person"), 100, "every commit");
    copy.close().unwrap();
    let source = database(&path);
    assert_eq!(count(&source, "Person"), 100, "the source reopens whole");
    source.close().unwrap();
}

// ---------------------------------------------------------------------------
// Every failure count, as a crash right after the call leaves it
// ---------------------------------------------------------------------------

mod sweep {
    use super::*;

    const PATH_VAR: &str = "GRAFEO_OUTCOME_PATH";
    const OP_VAR: &str = "GRAFEO_OUTCOME_OP";
    const COUNT_VAR: &str = "GRAFEO_OUTCOME_AT";

    /// What the child's call returned.
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    enum Outcome {
        Ok,
        /// GRAFEO-S007: the setting is stored off, its cursor's removal failed.
        CursorPending,
        /// The checkpoint's outcome is unknown until the reopen.
        Unknown,
        /// Nothing was stored.
        Failed,
    }

    struct Run {
        fired: bool,
        outcome: Outcome,
        /// The rows before the call.
        persons: usize,
        /// Whether the batch the child writes after the call committed.
        batch_ok: bool,
        /// The rows the child read after that batch.
        after: usize,
        /// The cursor's log position before the call, if there was one.
        cursor_before: Option<u64>,
    }

    fn run_child(path: &Path, op: &str, count: u64) -> Run {
        let output = child_process::output(
            grafeo_common::child_test!("outcome_child")
                .env(PATH_VAR, path)
                .env(OP_VAR, op)
                .env(COUNT_VAR, count.to_string()),
        )
        .unwrap();
        let stdout = String::from_utf8_lossy(&output.stdout).into_owned();
        let line = |key: &str| {
            stdout
                .lines()
                .find_map(|line| line.strip_prefix(key))
                .unwrap_or_else(|| panic!("count {count}: no {key} in {stdout}"))
                .to_string()
        };
        let outcome = match line("outcome: ").as_str() {
            "ok" => Outcome::Ok,
            "cursor pending" => Outcome::CursorPending,
            "unknown" => Outcome::Unknown,
            _ => Outcome::Failed,
        };
        Run {
            fired: line("fired: ") != "none",
            outcome,
            persons: line("persons: ").parse().unwrap(),
            batch_ok: line("batch: ") == "ok",
            after: line("after: ").parse().unwrap(),
            cursor_before: line("cursor: ").parse().ok(),
        }
    }

    /// The copy of the file the child takes before its call.
    fn before(path: &Path) -> PathBuf {
        PathBuf::from(format!("{}.before", path.display()))
    }

    /// Runs `op` in a child at every failure count, each on a database from `prepare`, the child
    /// exiting without a close; reopens and calls `check` with the outcome. After an unknown
    /// outcome the reopen finds the new image (its header is in the page cache), so the previous
    /// image is reached too: the file as it was before the call, with the WAL the exit left, is
    /// opened and handed to `previous` (when given). Returns the outcomes seen.
    fn every_count(
        op: &str,
        prepare: impl Fn(&Path),
        check: impl Fn(&GrafeoDB, &Path, &Run, u64),
        previous: Option<&dyn Fn(&GrafeoDB, &Path, &Run, u64)>,
    ) -> Vec<Outcome> {
        let mut seen = Vec::new();
        for count in 1..=512 {
            let dir = tempfile::tempdir().unwrap();
            let path = dir.path().join("db.grafeo");
            prepare(&path);
            let run = run_child(&path, op, count);
            assert!(
                sidecar(&path).exists(),
                "{op} {count}: the exit left the WAL"
            );
            let previous_image = dir.path().join("previous.grafeo");
            if run.outcome == Outcome::Unknown && previous.is_some() {
                std::fs::copy(before(&path), &previous_image).unwrap();
                copy_dir(&sidecar(&path), &sidecar(&previous_image));
            }
            let db = database(&path);
            check(&db, &path, &run, count);
            db.close().unwrap();
            if let Some(previous) = previous
                && run.outcome == Outcome::Unknown
            {
                let db = database(&previous_image);
                previous(&db, &previous_image, &run, count);
                db.close().unwrap();
            }
            if !run.fired {
                assert_eq!(run.outcome, Outcome::Ok, "{op}: the call without a failure");
                println!("{op}: {} failure counts, {seen:?}", seen.len());
                return seen;
            }
            seen.push(run.outcome);
        }
        panic!("{op}: more failure points than counts")
    }

    fn copy_dir(from: &Path, to: &Path) {
        std::fs::create_dir_all(to).unwrap();
        for entry in std::fs::read_dir(from).unwrap() {
            let entry = entry.unwrap();
            std::fs::copy(entry.path(), to.join(entry.file_name())).unwrap();
        }
    }

    /// Checks the cursor against the setting: on, a cursor with the log from it kept; off, none.
    fn assert_cursor_follows(db: &GrafeoDB, path: &Path, what: &str) {
        if db.is_cdc_enabled() {
            assert!(cursor(path).exists(), "{what}: on without a cursor");
            assert!(
                first_kept(path) <= cursor_lsn(path),
                "{what}: the log from the cursor"
            );
        } else {
            assert!(!cursor(path).exists(), "{what}: off with a cursor");
        }
    }

    /// The rows after the child's batch: the batch commits after every call but one whose
    /// failure stopped the WAL, and the reopen has it.
    fn assert_after_batch(db: &GrafeoDB, run: &Run, what: &str) {
        if matches!(run.outcome, Outcome::Ok | Outcome::Unknown) {
            assert!(run.batch_ok, "{what}: the batch after the call commits");
        }
        let expected = run.persons + if run.batch_ok { 100 } else { 0 };
        assert_eq!(run.after, expected, "{what}: the child's count");
        assert_eq!(
            count_persons(db),
            expected,
            "{what}: every commit, the batch's too"
        );
    }

    fn assert_saw_every_kind(op: &str, seen: &[Outcome]) {
        for kind in [Outcome::Ok, Outcome::Unknown, Outcome::Failed] {
            assert!(seen.contains(&kind), "{op}: no {kind:?} among {seen:?}");
        }
    }

    #[test]
    fn set_cdc_off_agrees_with_the_reopen_at_every_failure() {
        let seen = every_count(
            "cdc_off",
            |path| {
                let db = cdc_on_with_log(path);
                db.close().unwrap();
            },
            |db, path, run, count| {
                let outcome = run.outcome;
                let what = format!("off {count}, {outcome:?}");
                assert_after_batch(db, run, &what);
                match outcome {
                    Outcome::Ok | Outcome::CursorPending => {
                        assert!(!db.is_cdc_enabled(), "{what}");
                    }
                    Outcome::Failed => assert!(db.is_cdc_enabled(), "{what}"),
                    Outcome::Unknown => {}
                }
                assert_cursor_follows(db, path, &what);
            },
            Some(&|db, path, run, count| {
                let what = format!("off {count}, the previous image");
                assert_after_batch(db, run, &what);
                assert!(db.is_cdc_enabled(), "{what}: on, as before the call");
                assert_cursor_follows(db, path, &what);
                assert_eq!(
                    Some(cursor_lsn(path)),
                    run.cursor_before,
                    "{what}: the cursor the call kept, where it was"
                );
            }),
        );
        assert_saw_every_kind("set_cdc(false)", &seen);
        assert!(seen.contains(&Outcome::CursorPending), "{seen:?}");
    }

    #[test]
    fn set_cdc_on_agrees_with_the_reopen_at_every_failure() {
        let seen = every_count(
            "cdc_on",
            |path| {
                let db = database(path);
                write_batch(&db, 0);
                db.close().unwrap();
            },
            |db, path, run, count| {
                let outcome = run.outcome;
                let what = format!("on {count}, {outcome:?}");
                assert_after_batch(db, run, &what);
                match outcome {
                    Outcome::Ok => assert!(db.is_cdc_enabled(), "{what}"),
                    Outcome::Failed => assert!(!db.is_cdc_enabled(), "{what}"),
                    Outcome::Unknown => {}
                    Outcome::CursorPending => panic!("{what}: on has no cursor step after"),
                }
                assert_cursor_follows(db, path, &what);
            },
            Some(&|db, path, run, count| {
                let what = format!("on {count}, the previous image");
                assert_after_batch(db, run, &what);
                assert!(!db.is_cdc_enabled(), "{what}: off, as before the call");
                assert_cursor_follows(db, path, &what);
            }),
        );
        assert_saw_every_kind("set_cdc(true)", &seen);
    }

    #[test]
    fn set_stored_config_agrees_with_the_reopen_at_every_failure() {
        let seen = every_count(
            "memory_limit",
            |path| {
                let db = database(path);
                write_batch(&db, 0);
                db.close().unwrap();
            },
            |db, _, run, count| {
                let outcome = run.outcome;
                let what = format!("memory limit {count}, {outcome:?}");
                assert_after_batch(db, run, &what);
                let stored = db.memory_limit() == Some(48 * MIB);
                match outcome {
                    Outcome::Ok => assert!(stored, "{what}"),
                    Outcome::Failed => assert!(!stored, "{what}"),
                    Outcome::Unknown => {}
                    Outcome::CursorPending => panic!("{what}"),
                }
            },
            Some(&|db, _, run, count| {
                let what = format!("memory limit {count}, the previous image");
                assert_after_batch(db, run, &what);
                assert_ne!(db.memory_limit(), Some(48 * MIB), "{what}: the old setting");
            }),
        );
        assert_saw_every_kind("set_stored_config", &seen);
    }

    #[test]
    fn restore_agrees_with_the_reopen_at_every_failure() {
        let seen = every_count(
            "restore",
            |path| {
                let db = database(path);
                write_batch(&db, 0);
                db.close().unwrap();
            },
            |db, _, run, at| {
                let outcome = run.outcome;
                let what = format!("restore {at}, {outcome:?}");
                let state = (count_persons(db), count(db, "City"));
                // The batch after the call commits only on top of a restore that is stored.
                let restored = state == (if run.batch_ok { 100 } else { 0 }, 3);
                let kept = state == (run.persons, 0);
                match outcome {
                    Outcome::Ok => assert!(run.batch_ok && restored, "{what}: {state:?}"),
                    Outcome::Failed | Outcome::Unknown => {
                        assert!(!run.batch_ok, "{what}: a failed restore refuses commits");
                        assert!(restored || kept, "{what}: one state, whole: {state:?}");
                    }
                    Outcome::CursorPending => panic!("{what}"),
                }
            },
            Some(&|db, _, run, at| {
                let what = format!("restore {at}, the previous image");
                assert!(!run.batch_ok, "{what}: a failed restore refuses commits");
                assert_eq!(
                    (count_persons(db), count(db, "City")),
                    (run.persons, 0),
                    "{what}: the state before the restore, whole"
                );
            }),
        );
        assert_saw_every_kind("restore_snapshot", &seen);
    }

    fn count_persons(db: &GrafeoDB) -> usize {
        count(db, "Person")
    }

    /// Child-process entry for the sweeps; a no-op when run directly. It opens the database,
    /// writes a batch, and runs the call with the failure armed; the database is open before
    /// the failure is armed, and the process exits without closing it.
    #[test]
    fn outcome_child() {
        let Ok(path) = std::env::var(PATH_VAR) else {
            return;
        };
        let path = PathBuf::from(path);
        let op = std::env::var(OP_VAR).unwrap();
        let at: u64 = std::env::var(COUNT_VAR).unwrap().parse().unwrap();
        let snapshot = {
            let cities = GrafeoDB::new_in_memory();
            cities
                .execute("UNWIND range(1, 3) AS i INSERT (:City {i: i})")
                .unwrap();
            cities.export_snapshot().unwrap()
        };
        let db = database(&path);
        write_batch(&db, 5_000);
        db.file_manager().unwrap().copy_to(&before(&path)).unwrap();
        println!("persons: {}", count_persons(&db));
        match std::fs::read_to_string(cursor(&path)) {
            Ok(_) => println!("cursor: {}", cursor_lsn(&path)),
            Err(_) => println!("cursor: none"),
        }
        let result = with_failure_at(at, || match op.as_str() {
            "cdc_off" => db.set_cdc(false),
            "cdc_on" => db.set_cdc(true),
            "memory_limit" => {
                db.set_stored_config(&Config::in_memory().with_memory_limit(48 * MIB))
            }
            "restore" => db.restore_snapshot(&snapshot),
            other => panic!("unknown op {other}"),
        });
        println!(
            "fired: {}",
            fired_failure().unwrap_or_else(|| "none".into())
        );
        let outcome = match &result {
            Ok(()) => "ok",
            Err(error) if error.error_code().as_str() == "GRAFEO-S007" => "cursor pending",
            Err(error) if is_outcome_unknown(error) => "unknown",
            Err(_) => "failed",
        };
        println!("outcome: {outcome}");
        let batch = db.execute("UNWIND range(9000, 9099) AS i INSERT (:Person {i: i})");
        println!("batch: {}", if batch.is_ok() { "ok" } else { "refused" });
        println!("after: {}", count_persons(&db));
        std::mem::forget(db);
        std::process::exit(0);
    }
}
