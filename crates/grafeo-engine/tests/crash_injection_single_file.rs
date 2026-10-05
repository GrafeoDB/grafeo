//! Crash injection tests for the single-file `.grafeo` format.
//!
//! Two kinds of tests:
//!
//! - In process (the tests named `a_panic_...`): crash injection panics at a
//!   point of `close()` or of a checkpoint, and the panic is caught. Dropping
//!   the database then closes it again, which completes the checkpoint, so the
//!   reopen never starts from the state a crash would leave. These tests check
//!   that a panic during close or a checkpoint leaves a usable database that
//!   holds every write.
//! - In a child process that exits at the crash point without running any
//!   destructor (`wal_disabled_crash_during_checkpoint_recovers` here, and the
//!   sweeps `crash_at_every_step_of_the_first_checkpoint` and
//!   `crash_at_every_step_of_a_later_checkpoint` in `checkpoint_replay.rs`):
//!   these reopen from the files a crash leaves, and so test recovery,
//!   including the sidecar WAL replay.
//!
//! Requires both `grafeo-file` and `testing-crash-injection` features.

#![cfg(all(feature = "grafeo-file", feature = "testing-crash-injection"))]

use std::panic::AssertUnwindSafe;

use grafeo_common::testing::child_process;
use grafeo_common::testing::crash::{CrashResult, with_crash_at};
use grafeo_common::types::Value;
use grafeo_engine::{Config, GrafeoDB};

/// Helper: extract sorted string values from column 0 of query result rows.
fn extract_strings(rows: &[Vec<Value>]) -> Vec<String> {
    let mut names: Vec<String> = rows
        .iter()
        .filter_map(|r| match &r[0] {
            Value::String(s) => Some(s.to_string()),
            _ => None,
        })
        .collect();
    names.sort();
    names
}

/// Helper: build the sidecar WAL path for a given `.grafeo` file.
fn sidecar_wal_path(path: &std::path::Path) -> std::path::PathBuf {
    let mut p = path.as_os_str().to_owned();
    p.push(".wal");
    std::path::PathBuf::from(p)
}

/// The crash points of a checkpoint with the WAL enabled, in the order
/// `wal_checkpoint()` and `close()` reach them: the flush rotates the WAL,
/// the file manager writes the new image (its chunks, a sync, the header, the
/// trim), then the flush marks the WAL.
const CHECKPOINT_POINTS: [&str; 8] = [
    "flush:before_serialize",
    "flush:after_rotate",
    "checkpoint:after_chunks",
    "checkpoint:after_data_sync",
    "checkpoint:after_header",
    "checkpoint:before_trim",
    "flush:after_write",
    "flush:after_mark_checkpoint",
];

/// The crash points of `close()` with the WAL enabled: the checkpoint's, then
/// the removal of the sidecar WAL.
const CLOSE_POINTS: [&str; 9] = [
    CHECKPOINT_POINTS[0],
    CHECKPOINT_POINTS[1],
    CHECKPOINT_POINTS[2],
    CHECKPOINT_POINTS[3],
    CHECKPOINT_POINTS[4],
    CHECKPOINT_POINTS[5],
    CHECKPOINT_POINTS[6],
    CHECKPOINT_POINTS[7],
    "close:before_remove_sidecar_wal",
];

/// How many crash points `points` lists.
fn count(points: &[&str]) -> u64 {
    u64::try_from(points.len()).unwrap()
}

/// The crash point at `count` (1-based) of `points`, or "no crash" past the last.
fn point_name(points: &[&'static str], count: u64) -> &'static str {
    usize::try_from(count - 1)
        .ok()
        .and_then(|index| points.get(index))
        .copied()
        .unwrap_or("no crash")
}

/// Assert that at least the main file or the sidecar WAL exists, so recovery
/// is possible. If neither exists, the crash destroyed all data.
fn assert_recoverable(path: &std::path::Path, context: &str) {
    let wal = sidecar_wal_path(path);
    assert!(
        path.exists() || wal.exists(),
        "{context}: neither the main file ({}) nor the sidecar WAL ({}) exist, \
         crash destroyed all data",
        path.display(),
        wal.display(),
    );
}

// =========================================================================
// Crash during checkpoint_to_file (close path)
// =========================================================================

/// A panic at each point of `close()` (see `CLOSE_POINTS`), and a run past
/// the last one, leaves a database that reopens with every write. In process,
/// so the drop after the panic completes the close: this does not test
/// recovery from a crashed state (see the module docs).
#[test]
fn a_panic_at_each_point_of_close_leaves_a_usable_database() {
    for crash_point in 1..=count(&CLOSE_POINTS) + 1 {
        let point = point_name(&CLOSE_POINTS, crash_point);
        let dir = tempfile::TempDir::new().unwrap();
        let path = dir.path().join("crash_test.grafeo");

        // Phase 1: Create and populate
        {
            let db = GrafeoDB::with_config(Config::persistent(&path)).unwrap();
            let session = db.session();
            session.execute("INSERT (:Person {name: 'Alix'})").unwrap();
            session.execute("INSERT (:Person {name: 'Gus'})").unwrap();
            session
                .execute(
                    "MATCH (a:Person {name: 'Alix'}), (b:Person {name: 'Gus'}) \
                     INSERT (a)-[:KNOWS]->(b)",
                )
                .unwrap();

            // Crash during close
            let db = AssertUnwindSafe(db);
            let result = with_crash_at(crash_point, move || {
                let _ = db.close();
            });

            // Every point crashes; the run past the last one completes, with
            // the data in the .grafeo file. (After a crash, dropping the
            // database closes it again.)
            let crashed = matches!(result, CrashResult::Crashed);
            assert_eq!(
                crashed,
                crash_point <= count(&CLOSE_POINTS),
                "crash_point={crash_point} ({point}): close() has {} crash points",
                CLOSE_POINTS.len()
            );
        }

        // Phase 2: Reopen and verify every write is there
        assert_recoverable(&path, &format!("crash_point={crash_point} ({point})"));

        let db = GrafeoDB::open(&path).unwrap();
        let session = db.session();

        let result = session.execute("MATCH (p:Person) RETURN p.name").unwrap();
        let names = extract_strings(result.rows());
        assert_eq!(
            names,
            vec!["Alix", "Gus"],
            "crash_point={crash_point} ({point}): data lost after crash"
        );

        assert_eq!(
            db.edge_count(),
            1,
            "crash_point={crash_point} ({point}): edge lost after crash"
        );

        db.close().unwrap();
    }
}

// =========================================================================
// Crash during explicit wal_checkpoint
// =========================================================================

/// A panic at each point of an explicit `wal_checkpoint()` (see
/// `CHECKPOINT_POINTS`), and a run past the last one, leaves a database that
/// reopens with every write. In process, so dropping the database closes it
/// and completes a checkpoint: this does not test recovery from a crashed
/// state (see the module docs).
#[test]
fn a_panic_at_each_point_of_a_checkpoint_leaves_a_usable_database() {
    for crash_point in 1..=count(&CHECKPOINT_POINTS) + 1 {
        let point = point_name(&CHECKPOINT_POINTS, crash_point);
        let dir = tempfile::TempDir::new().unwrap();
        let path = dir.path().join("wal_crash.grafeo");

        // Create and populate
        let db = GrafeoDB::with_config(Config::persistent(&path)).unwrap();
        let session = db.session();
        session
            .execute("INSERT (:City {name: 'Amsterdam'})")
            .unwrap();
        session.execute("INSERT (:City {name: 'Berlin'})").unwrap();

        // Crash during explicit checkpoint (db stays open)
        let db_ref = AssertUnwindSafe(&db);
        let result = with_crash_at(crash_point, move || {
            let _ = db_ref.wal_checkpoint();
        });

        // Every point crashes (the in-memory data stays); the run past the
        // last one completes the checkpoint.
        let crashed = matches!(result, CrashResult::Crashed);
        assert_eq!(
            crashed,
            crash_point <= count(&CHECKPOINT_POINTS),
            "crash_point={crash_point} ({point}): a checkpoint has {} crash points",
            CHECKPOINT_POINTS.len()
        );

        // The drop closes the database, which completes a checkpoint.
        drop(db);

        // Reopen: every write is there
        assert_recoverable(&path, &format!("crash_point={crash_point} ({point})"));

        let db2 = GrafeoDB::open(&path).unwrap();
        let session2 = db2.session();

        let result = session2.execute("MATCH (c:City) RETURN c.name").unwrap();
        let names = extract_strings(result.rows());
        assert_eq!(
            names,
            vec!["Amsterdam", "Berlin"],
            "crash_point={crash_point} ({point}): data lost after checkpoint crash"
        );

        db2.close().unwrap();
    }
}

// =========================================================================
// Crash after first checkpoint, then more writes
// =========================================================================

/// A panic at the first point of `close()` after a checkpoint and more writes
/// leaves a database that reopens with the writes from before and after the
/// checkpoint. In process, so the drop after the panic completes the close
/// (see the module docs).
#[test]
fn a_panic_in_close_after_a_checkpoint_and_new_writes_leaves_a_usable_database() {
    let dir = tempfile::TempDir::new().unwrap();
    let path = dir.path().join("incremental.grafeo");

    // Phase 1: Create, populate, and successfully checkpoint
    {
        let db = GrafeoDB::with_config(Config::persistent(&path)).unwrap();
        let session = db.session();
        session.execute("INSERT (:Person {name: 'Alix'})").unwrap();
        db.wal_checkpoint().unwrap();

        // Phase 2: Add more data
        session.execute("INSERT (:Person {name: 'Gus'})").unwrap();

        // Phase 3: Crash during close (after more writes)
        let db = AssertUnwindSafe(db);
        let _result = with_crash_at(1, move || {
            let _ = db.close();
        });
    }

    // Reopen: the drop after the panic completed the close, so both writes
    // are in the file.
    let db = GrafeoDB::open(&path).unwrap();
    let session = db.session();

    let result = session.execute("MATCH (p:Person) RETURN p.name").unwrap();
    let names = extract_strings(result.rows());
    assert_eq!(
        names,
        vec!["Alix", "Gus"],
        "the writes before and after the checkpoint"
    );

    db.close().unwrap();
}

// =========================================================================
// Multiple checkpoint-crash-recover cycles
// =========================================================================

/// Five sessions, every other one closed with a panic at the second point of
/// `close()`, leave a database with the node of every session. In process, so
/// the drop after each panic completes the close (see the module docs).
#[test]
fn repeated_panics_in_close_leave_a_usable_database() {
    let dir = tempfile::TempDir::new().unwrap();
    let path = dir.path().join("cycles.grafeo");

    let people = ["Alix", "Gus", "Vincent", "Jules", "Mia"];

    for (i, name) in people.iter().enumerate() {
        // Open (or create on first iteration)
        let db = if i == 0 {
            GrafeoDB::with_config(Config::persistent(&path)).unwrap()
        } else {
            GrafeoDB::open(&path).unwrap()
        };

        let session = db.session();
        session
            .execute(&format!("INSERT (:Person {{name: '{name}'}})"))
            .unwrap();

        // Alternate between clean close and crash
        if i % 2 == 0 {
            db.close().unwrap();
        } else {
            let db = AssertUnwindSafe(db);
            let _result = with_crash_at(2, move || {
                let _ = db.close();
            });
        }
    }

    // Final verification: the drop after each panic completed that close.
    let db = GrafeoDB::open(&path).unwrap();
    assert_eq!(
        db.node_count(),
        people.len(),
        "the node of every session, also of those closed with a panic"
    );
    db.close().unwrap();
}

// =========================================================================
// Crash during sidecar WAL removal (close:before_remove_sidecar_wal)
// =========================================================================

/// A panic between writing the final checkpoint and removing the sidecar WAL
/// leaves a database that reopens with every write, and whose next clean
/// close removes the sidecar WAL. In process, so the drop after the panic
/// completes the close (see the module docs); a crash at this point leaves
/// the checkpoint and the sidecar WAL, whose replay the child-process sweeps
/// in `checkpoint_replay.rs` cover.
#[test]
fn a_panic_before_the_sidecar_wal_removal_leaves_a_usable_database() {
    let dir = tempfile::TempDir::new().unwrap();
    let path = dir.path().join("sidecar_crash.grafeo");

    // Phase 1: Populate, then crash exactly before remove_sidecar_wal
    {
        let db = GrafeoDB::with_config(Config::persistent(&path)).unwrap();
        let session = db.session();
        session
            .execute("INSERT (:Person {name: 'Django'})")
            .unwrap();
        session
            .execute("INSERT (:Person {name: 'Beatrix'})")
            .unwrap();

        // The last point of close(), close:before_remove_sidecar_wal: the
        // checkpoint is written, the sidecar WAL is not removed yet.
        let db = AssertUnwindSafe(db);
        let result = with_crash_at(count(&CLOSE_POINTS), move || {
            let _ = db.close();
        });
        assert!(
            matches!(result, CrashResult::Crashed),
            "close() reaches close:before_remove_sidecar_wal"
        );
    }

    // Phase 2: Reopen with both nodes
    assert_recoverable(&path, "crash before sidecar WAL removal");

    let db = GrafeoDB::open(&path).unwrap();
    let count = db.node_count();
    assert_eq!(
        count, 2,
        "both nodes must survive crash before sidecar removal"
    );

    let result = db
        .session()
        .execute("MATCH (p:Person) RETURN p.name ORDER BY p.name")
        .unwrap();
    let names = extract_strings(result.rows());
    assert!(
        names.contains(&"Beatrix".to_string()),
        "Beatrix missing after crash"
    );
    assert!(
        names.contains(&"Django".to_string()),
        "Django missing after crash"
    );

    // Proper close must now clean up the sidecar WAL
    db.close().unwrap();

    let wal_path = sidecar_wal_path(&path);
    assert!(
        !wal_path.exists(),
        "sidecar WAL must be removed after the second (clean) close"
    );
}

// =========================================================================
// WAL-disabled single-file crash injection tests
// =========================================================================

/// Helper: build a WAL-disabled persistent config for a `.grafeo` path.
fn wal_disabled_config(path: &std::path::Path) -> Config {
    Config {
        wal_enabled: false,
        ..Config::persistent(path)
    }
}

/// With WAL disabled, a clean close triggers `checkpoint_to_file` which writes
/// the snapshot. On reopen the data should be fully intact.
#[test]
fn wal_disabled_checkpoint_preserves_data() {
    let dir = tempfile::TempDir::new().unwrap();
    let path = dir.path().join("wal_off_persist.grafeo");

    // Phase 1: Create, populate, close cleanly (checkpoint on close)
    {
        let db = GrafeoDB::with_config(wal_disabled_config(&path)).unwrap();
        let session = db.session();
        session.execute("INSERT (:Person {name: 'Alix'})").unwrap();
        session.execute("INSERT (:Person {name: 'Gus'})").unwrap();
        session
            .execute(
                "MATCH (a:Person {name: 'Alix'}), (b:Person {name: 'Gus'}) \
                 INSERT (a)-[:KNOWS]->(b)",
            )
            .unwrap();

        // No sidecar WAL should exist
        assert!(
            !sidecar_wal_path(&path).exists(),
            "no sidecar WAL should be created when WAL is disabled"
        );

        db.close().unwrap();
    }

    // Phase 2: Reopen with WAL disabled and verify everything survived
    assert!(
        path.exists(),
        "checkpoint must have written the .grafeo file"
    );

    let db = GrafeoDB::with_config(wal_disabled_config(&path)).unwrap();
    let session = db.session();

    let result = session.execute("MATCH (p:Person) RETURN p.name").unwrap();
    let names = extract_strings(result.rows());
    assert_eq!(
        names,
        vec!["Alix", "Gus"],
        "both nodes must survive close-reopen with WAL disabled"
    );

    assert_eq!(
        db.edge_count(),
        1,
        "edge must survive close-reopen with WAL disabled"
    );

    db.close().unwrap();
}

/// With WAL disabled, crashing during `checkpoint_to_file` on close means there
/// is no sidecar WAL to replay. If a prior successful checkpoint exists, data
/// from that checkpoint should survive. On a first write with no prior
/// checkpoint, data may be lost entirely.
///
/// This test does two rounds:
///   1. Write initial data, close cleanly (successful checkpoint).
///   2. Write more data, crash during the close checkpoint.
///
/// On reopen, the first-round data must survive. The sweep covers every
/// crash point of the close (see `WAL_DISABLED_CLOSE_POINTS`), including
/// those of writing the new image next to the old one (#418): until the new
/// image's header is written the reopen finds the round-1 image, from then on
/// the new one.
#[test]
fn wal_disabled_crash_during_checkpoint_recovers() {
    for crash_point in 1..=count(&WAL_DISABLED_CLOSE_POINTS) + 1 {
        let point = point_name(&WAL_DISABLED_CLOSE_POINTS, crash_point);
        let dir = tempfile::TempDir::new().unwrap();
        let path = dir.path().join("wal_off_crash.grafeo");

        // Round 1: Create, populate, close cleanly to establish a valid checkpoint
        {
            let db = GrafeoDB::with_config(wal_disabled_config(&path)).unwrap();
            let session = db.session();
            session
                .execute("INSERT (:Person {name: 'Vincent'})")
                .unwrap();
            session.execute("INSERT (:Person {name: 'Jules'})").unwrap();
            db.close().unwrap();
        }

        // Round 2: Reopen, add more data, crash during close checkpoint. In a
        // child process: caught in this one, the panic would drop the
        // database, whose `Drop` closes it again and finishes the checkpoint.
        let completed = close_in_child(crash_point, &path);
        assert_eq!(
            completed,
            crash_point > count(&WAL_DISABLED_CLOSE_POINTS),
            "crash_point={crash_point} ({point}): close() without a WAL has {} crash points",
            WAL_DISABLED_CLOSE_POINTS.len()
        );

        // Reopen: the .grafeo file should still have a valid snapshot
        // from round 1, so at least those 2 nodes must survive.
        assert!(
            path.exists(),
            "crash_point={crash_point} ({point}): .grafeo file must exist from round-1 checkpoint"
        );

        let db = GrafeoDB::with_config(wal_disabled_config(&path)).unwrap();
        let session = db.session();

        let result = session.execute("MATCH (p:Person) RETURN p.name").unwrap();
        let names = extract_strings(result.rows());

        // Round-1 data must survive
        assert!(
            names.contains(&"Vincent".to_string()),
            "crash_point={crash_point} ({point}): Vincent missing after crash"
        );
        assert!(
            names.contains(&"Jules".to_string()),
            "crash_point={crash_point} ({point}): Jules missing after crash"
        );

        // Round-2 data (Mia) is in the new image, which the reopen finds once
        // its header was written (the process exit keeps written pages, so
        // this checks the order of the writes, not their durability).
        assert_eq!(
            names.contains(&"Mia".to_string()),
            crash_point >= NEW_HEADER_WRITTEN,
            "crash_point={crash_point} ({point}): Mia is in the file from \
             checkpoint:after_header on: {names:?}"
        );
        assert!(
            names.len() <= 3,
            "crash_point={crash_point} ({point}): {names:?}"
        );

        db.close().unwrap();
    }
}

/// The crash points of `close()` with the WAL disabled: the checkpoint's
/// without the WAL mark, then the removal of the (absent) sidecar WAL.
const WAL_DISABLED_CLOSE_POINTS: [&str; 8] = [
    "flush:before_serialize",
    "flush:after_rotate",
    "checkpoint:after_chunks",
    "checkpoint:after_data_sync",
    "checkpoint:after_header",
    "checkpoint:before_trim",
    "flush:after_write",
    "close:before_remove_sidecar_wal",
];

/// The first point of `WAL_DISABLED_CLOSE_POINTS` (1-based) at which the new
/// image's database header is written: `checkpoint:after_header`.
const NEW_HEADER_WRITTEN: u64 = 5;

const CHILD_POINT_VAR: &str = "GRAFEO_CRASH_SINGLE_FILE_POINT";
const CHILD_PATH_VAR: &str = "GRAFEO_CRASH_SINGLE_FILE_PATH";
/// Exit code of a child whose `close()` crashed.
const CRASHED: i32 = 3;

/// Reopens the WAL-disabled database at `path` in a child process, adds a
/// node and crashes at `crash_point` inside `close()`. Returns whether the
/// close completed.
fn close_in_child(crash_point: u64, path: &std::path::Path) -> bool {
    let status = child_process::run(
        std::process::Command::new(std::env::current_exe().unwrap())
            .args(["--exact", "close_child", "--nocapture"])
            .env(CHILD_POINT_VAR, crash_point.to_string())
            .env(CHILD_PATH_VAR, path),
    )
    .unwrap();
    match status.code() {
        Some(0) => true,
        Some(CRASHED) => false,
        other => panic!("crash_point={crash_point}: child failed with {other:?}"),
    }
}

/// Child-process entry for [`close_in_child`]; a no-op when run directly.
#[test]
fn close_child() {
    let (Ok(point), Some(path)) = (
        std::env::var(CHILD_POINT_VAR),
        std::env::var_os(CHILD_PATH_VAR),
    ) else {
        return;
    };
    let db = GrafeoDB::with_config(wal_disabled_config(std::path::Path::new(&path))).unwrap();
    db.session()
        .execute("INSERT (:Person {name: 'Mia'})")
        .unwrap();
    let target = AssertUnwindSafe(&db);
    let result = with_crash_at(point.parse().unwrap(), move || target.close());
    // Exit without running destructors, like a crash.
    match result {
        CrashResult::Completed(closed) => {
            closed.unwrap();
            std::process::exit(0);
        }
        _ => std::process::exit(CRASHED),
    }
}

/// With WAL disabled, uncommitted transaction data should not be persisted.
/// If the process crashes (or simply drops) before committing, the
/// checkpoint-on-close only captures committed state.
#[test]
fn wal_disabled_uncommitted_data_lost_on_crash() {
    let dir = tempfile::TempDir::new().unwrap();
    let path = dir.path().join("wal_off_uncommitted.grafeo");

    // Phase 1: Write committed data, close cleanly
    {
        let db = GrafeoDB::with_config(wal_disabled_config(&path)).unwrap();
        let session = db.session();
        session.execute("INSERT (:Person {name: 'Butch'})").unwrap();
        db.close().unwrap();
    }

    // Phase 2: Reopen, start an explicit transaction, write data, crash
    // without committing
    {
        let db = GrafeoDB::with_config(wal_disabled_config(&path)).unwrap();
        let mut session = db.session();

        // Begin explicit transaction (auto-commit is off)
        session.begin_transaction().unwrap();
        session
            .execute("INSERT (:Person {name: 'Django'})")
            .unwrap();
        // Do NOT commit: simulate a crash while the transaction is open

        let db = AssertUnwindSafe(db);
        let _result = with_crash_at(1, move || {
            let _ = db.close();
        });
    }

    // Phase 3: Reopen and verify only the committed data (Butch) is present.
    // Django was never committed, so it must not appear.
    assert!(
        path.exists(),
        ".grafeo file must exist from phase-1 checkpoint"
    );

    let db = GrafeoDB::with_config(wal_disabled_config(&path)).unwrap();
    let session = db.session();

    let result = session.execute("MATCH (p:Person) RETURN p.name").unwrap();
    let names = extract_strings(result.rows());

    assert!(
        names.contains(&"Butch".to_string()),
        "committed data (Butch) must survive"
    );
    assert!(
        !names.contains(&"Django".to_string()),
        "uncommitted data (Django) must NOT be present after crash"
    );
    assert_eq!(db.node_count(), 1, "only the committed node should exist");

    db.close().unwrap();
}
