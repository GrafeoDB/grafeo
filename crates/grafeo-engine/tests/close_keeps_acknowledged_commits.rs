//! A commit racing `close()` is either refused or kept (#712): every commit acknowledged before
//! or during `close()` is in the database after a reopen, whether `close()` completed or the
//! process crashed inside it at any of its crash points.
//!
//! A child process commits a node per key from a writer thread, printing each acknowledged key,
//! while its main thread closes the database with a crash armed at its `n`th crash point; the
//! crash exits the process from a panic hook, so no `Drop` of the database runs (Lessons
//! Learned, Testing 5). The parent reopens the file and finds every acknowledged key. The sweep
//! runs `n` upwards until a `close()` completes without reaching it.
//!
//! ```bash
//! cargo test -p grafeo-engine --all-features --test main close_keeps_acknowledged_commits::
//! ```

#![cfg(all(
    feature = "gql",
    feature = "wal",
    feature = "grafeo-file",
    feature = "testing-crash-injection"
))]

use std::collections::BTreeSet;
use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::{AtomicI64, Ordering};

use grafeo_common::testing::child_process;
use grafeo_common::types::Value;
use grafeo_engine::config::DurabilityMode;
use grafeo_engine::{Config, GrafeoDB};

/// The database path the child works on.
const PATH_VAR: &str = "GRAFEO_CLOSE_ACKNOWLEDGED_PATH";
/// The crash point the child arms, counted from 1.
const CRASH_VAR: &str = "GRAFEO_CLOSE_ACKNOWLEDGED_CRASH_AT";
/// Exit code of a child whose `close()` crashed.
const CRASHED: i32 = 29;
/// Exit code of a child whose `close()` completed.
const CLOSED: i32 = 31;
/// Exit code of a child that panicked for another reason than the injected crash.
const FAILED: i32 = 37;
/// Commits before the close starts, so the close has a checkpoint to write.
const BEFORE: i64 = 50;
/// The most crash points the sweep tries before it fails.
const MOST_POINTS: u64 = 64;

fn open(path: &Path) -> GrafeoDB {
    GrafeoDB::with_config(Config::persistent(path).with_wal_durability(DurabilityMode::Sync))
        .unwrap()
}

/// Commits `:N {k: key}`; whether it was acknowledged.
fn commit(db: &GrafeoDB, key: i64) -> bool {
    db.execute_with_params(
        "INSERT (:N {k: $k})",
        [("k".to_string(), Value::Int64(key))].into_iter().collect(),
    )
    .is_ok()
}

/// Child-process entry for [`run_child`]; a no-op when run directly.
#[test]
fn close_acknowledged_child() {
    let (Some(path), Ok(crash_at)) = (std::env::var_os(PATH_VAR), std::env::var(CRASH_VAR)) else {
        return;
    };
    let crash_at: u64 = crash_at.parse().unwrap();
    std::panic::set_hook(Box::new(|info| {
        // Only the injected crash counts as one: another panic (a failing unwrap, a join) is a
        // failure of the child, which the parent reports.
        let injected = info
            .payload()
            .downcast_ref::<String>()
            .map(String::as_str)
            .or_else(|| info.payload().downcast_ref::<&str>().copied())
            .is_some_and(|message| message.contains("crash injection at"));
        eprintln!("{}: {info}", if injected { "crashed" } else { "failed" });
        std::process::exit(if injected { CRASHED } else { FAILED });
    }));
    let db = Arc::new(open(Path::new(&path)));
    for key in 0..BEFORE {
        assert!(commit(&db, key));
        println!("ACK {key}");
    }
    let racing = Arc::new(AtomicI64::new(0));
    let writer = {
        let db = Arc::clone(&db);
        let racing = Arc::clone(&racing);
        std::thread::spawn(move || {
            for key in BEFORE.. {
                if !commit(&db, key) {
                    return;
                }
                println!("ACK {key}");
                racing.fetch_add(1, Ordering::Release);
            }
        })
    };
    // The writer is committing when the close starts.
    while racing.load(Ordering::Acquire) < 5 {
        std::thread::yield_now();
    }
    grafeo_common::testing::crash::enable_crash_at(crash_at);
    db.close().unwrap();
    grafeo_common::testing::crash::disable_crash();
    writer.join().unwrap();
    std::process::exit(CLOSED);
}

/// Runs the child with a crash at point `crash_at` on the database at `path`; whether its
/// `close()` completed, and the keys it acknowledged.
fn run_child(path: &Path, crash_at: u64) -> (bool, BTreeSet<i64>) {
    let output = child_process::output(
        grafeo_common::child_test!("close_acknowledged_child")
            .env(PATH_VAR, path)
            .env(CRASH_VAR, crash_at.to_string()),
    )
    .unwrap();
    let stdout = String::from_utf8_lossy(&output.stdout);
    let closed = match output.status.code() {
        Some(CLOSED) => true,
        Some(CRASHED) => false,
        other => panic!(
            "crash point {crash_at}: the child exited with {other:?}:\n{}",
            String::from_utf8_lossy(&output.stderr)
        ),
    };
    let acknowledged = stdout
        .lines()
        .filter_map(|line| line.strip_prefix("ACK "))
        .map(|key| key.parse().unwrap())
        .collect();
    (closed, acknowledged)
}

/// Every key `db` holds.
fn keys(db: &GrafeoDB) -> BTreeSet<i64> {
    db.execute("MATCH (n:N) RETURN n.k")
        .unwrap()
        .rows()
        .iter()
        .map(|row| row[0].as_int64().unwrap())
        .collect()
}

/// At every crash point of `close()`, and when it completes, the reopen holds every key the
/// child acknowledged.
#[test]
fn every_acknowledged_commit_survives_a_racing_close() {
    for crash_at in 1..=MOST_POINTS {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("racing.grafeo");
        let (closed, acknowledged) = run_child(&path, crash_at);
        assert!(
            acknowledged.len() >= usize::try_from(BEFORE).unwrap(),
            "crash point {crash_at}: the commits before the close were acknowledged"
        );
        let db = open(&path);
        let found = keys(&db);
        let missing: Vec<_> = acknowledged.difference(&found).collect();
        assert!(
            missing.is_empty(),
            "crash point {crash_at} ({}): acknowledged commits missing after the reopen: {missing:?}",
            if closed { "closed" } else { "crashed" }
        );
        db.close().unwrap();
        if closed {
            assert!(
                crash_at > 1,
                "close() passed no crash point: the sweep crashed nothing"
            );
            eprintln!("close() completed past {} crash points", crash_at - 1);
            return;
        }
    }
    panic!("close() never completed within {MOST_POINTS} crash points");
}
