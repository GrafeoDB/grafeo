//! A commit completes before anything that comes after it (#548).
//!
//! A commit assigns its epoch, then writes its versions, CDC events and WAL
//! records. These tests run work on another thread from inside a commit,
//! right after its epoch is assigned (the `testing-statement-injection`
//! commit hook), and check that the work lands after the commit, never in the
//! middle of it. The WAL tests crash a child process (it exits without
//! `close()`, so nothing is checkpointed) and reopen, so the WAL is replayed.
//!
//! ```bash
//! cargo test -p grafeo-engine --all-features --test commit_completion
//! ```

#![cfg(feature = "testing-statement-injection")]

use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, mpsc};
use std::thread::JoinHandle;
use std::time::Duration;

use grafeo_common::testing::commit_hook::after_next_commit_epoch;
use grafeo_common::types::{NodeId, Value};
use grafeo_engine::GrafeoDB;

/// Work started from inside a commit, joined after it.
struct DuringCommit<T>(Arc<Mutex<Option<JoinHandle<T>>>>);

impl<T> DuringCommit<T> {
    fn join(self) -> T {
        let handle = self
            .0
            .lock()
            .unwrap()
            .take()
            .expect("the commit did not run the hook");
        handle
            .join()
            .expect("the work started during the commit panicked")
    }
}

/// Starts `work` on another thread from inside the next commit on this
/// thread, right after its epoch is assigned, and gives it time to finish
/// there: work that does not wait for the commit runs in the middle of it.
fn during_next_commit<T: Send + 'static>(
    work: impl FnOnce() -> T + Send + 'static,
) -> DuringCommit<T> {
    let slot = Arc::new(Mutex::new(None));
    let handle_slot = Arc::clone(&slot);
    after_next_commit_epoch(move || {
        let (done, finished) = mpsc::channel();
        let handle = std::thread::spawn(move || {
            let result = work();
            let _ = done.send(());
            result
        });
        let _ = finished.recv_timeout(Duration::from_millis(300));
        *handle_slot.lock().unwrap() = Some(handle);
    });
    DuringCommit(slot)
}

fn by(db: &GrafeoDB, node: NodeId) -> Option<Value> {
    db.get_node(node)?.get_property("by").cloned()
}

/// A transaction that sets `by` on `hub` and starts `work` (given the
/// database and the hub) from inside its commit. Returns the work's result.
fn commit_with<T: Send + 'static>(
    db: &Arc<GrafeoDB>,
    hub: NodeId,
    work: impl FnOnce(Arc<GrafeoDB>, NodeId) -> T + Send + 'static,
) -> T {
    let mut session = db.session();
    session.begin_transaction().unwrap();
    session
        .set_node_property(hub, "by", Value::from("transaction"))
        .unwrap();
    let during = {
        let db = Arc::clone(db);
        during_next_commit(move || work(db, hub))
    };
    session.commit().unwrap();
    during.join()
}

/// A direct write that arrives while a transaction commits lands after the
/// commit: the committed value holds at the commit's epoch and the direct
/// write's value is the latest.
#[cfg(feature = "temporal")]
#[test]
fn a_direct_write_during_a_commit_lands_after_it() {
    let db = Arc::new(GrafeoDB::new_in_memory());
    let hub = db.create_node(&["Hub"]).unwrap();
    let commit_epoch = grafeo_common::types::EpochId::new(db.current_epoch().as_u64() + 1);
    commit_with(&db, hub, |db, hub| {
        db.set_node_property(hub, "by", Value::from("direct"))
    })
    .unwrap();

    assert_eq!(
        db.get_node_property_at_epoch(hub, "by", commit_epoch),
        Some(Value::from("transaction"))
    );
    assert_eq!(by(&db, hub), Some(Value::from("direct")));
}

/// A transaction that begins while another commits sees what that commit
/// created.
#[test]
fn a_transaction_that_begins_during_a_commit_sees_it() {
    let db = Arc::new(GrafeoDB::new_in_memory());
    let mut session = db.session();
    session.begin_transaction().unwrap();
    let doc = session.create_node(&["Doc"]).unwrap();
    let reader = {
        let db = Arc::clone(&db);
        during_next_commit(move || {
            let mut reader = db.session();
            reader.begin_transaction().unwrap();
            let seen = reader.get_node(doc).is_some();
            reader.commit().unwrap();
            seen
        })
    };
    session.commit().unwrap();

    assert!(reader.join(), "the new transaction did not see the commit");
}

/// An open transaction that writes what a committing one wrote gets a write
/// conflict, and its rollback leaves the committed value alone.
#[test]
fn a_write_during_a_commit_to_what_it_wrote_conflicts() {
    let db = Arc::new(GrafeoDB::new_in_memory());
    let hub = db.create_node(&["Hub"]).unwrap();
    let mut other = db.session();
    other.begin_transaction().unwrap();
    let written = commit_with(&db, hub, move |_, hub| {
        let result = other.set_node_property(hub, "by", Value::from("other"));
        other.rollback().unwrap();
        result.map_err(|error| error.to_string())
    });

    let error = written.unwrap_err();
    assert!(error.to_lowercase().contains("conflict"), "got: {error}");
    assert_eq!(by(&db, hub), Some(Value::from("transaction")));
}

// ---------------------------------------------------------------------------
// WAL order: a crashed child process, then a reopen that replays the WAL
// ---------------------------------------------------------------------------

#[cfg(feature = "wal")]
mod wal {
    use super::*;
    use grafeo_common::testing::child_process;
    use grafeo_engine::Config;
    use grafeo_engine::config::StorageFormat;

    const SCENARIO_VAR: &str = "GRAFEO_COMMIT_COMPLETION_SCENARIO";
    const PATH_VAR: &str = "GRAFEO_COMMIT_COMPLETION_PATH";
    const FORMAT_VAR: &str = "GRAFEO_COMMIT_COMPLETION_FORMAT";

    fn formats(dir: &Path) -> Vec<(&'static str, PathBuf)> {
        let mut formats = vec![("wal-directory", dir.join("dir-db"))];
        #[cfg(feature = "grafeo-file")]
        formats.push(("single-file", dir.join("single.grafeo")));
        formats
    }

    fn open(path: &Path, format: &str) -> GrafeoDB {
        let format = match format {
            "wal-directory" => StorageFormat::WalDirectory,
            "single-file" => StorageFormat::SingleFile,
            other => panic!("unknown format {other}"),
        };
        GrafeoDB::with_config(Config::persistent(path).with_storage_format(format)).unwrap()
    }

    /// Runs `scenario` in a child process that exits without closing the
    /// database, then reopens it and returns the hub's `by` (the hub is the
    /// first node the scenario creates).
    fn by_after_crash(scenario: &str, path: &Path, format: &str) -> Option<Value> {
        let status = child_process::run(
            std::process::Command::new(std::env::current_exe().unwrap())
                .args(["--exact", "wal::crash_child", "--nocapture"])
                .env(SCENARIO_VAR, scenario)
                .env(PATH_VAR, path)
                .env(FORMAT_VAR, format),
        )
        .unwrap();
        assert!(status.success(), "{format}: scenario {scenario} failed");
        let db = open(path, format);
        let hub = db.execute("MATCH (h:Hub) RETURN id(h)").unwrap().rows()[0][0].clone();
        let Value::Int64(hub) = hub else {
            panic!("no hub: {hub:?}");
        };
        by(&db, NodeId::new(u64::try_from(hub).unwrap()))
    }

    /// Child-process entry for [`by_after_crash`]; a no-op when run directly.
    #[test]
    fn crash_child() {
        let Ok(scenario) = std::env::var(SCENARIO_VAR) else {
            return;
        };
        let path = PathBuf::from(std::env::var_os(PATH_VAR).unwrap());
        let db = Arc::new(open(&path, &std::env::var(FORMAT_VAR).unwrap()));
        let hub = db.create_node(&["Hub"]).unwrap();
        match scenario.as_str() {
            "direct_write" => {
                commit_with(&db, hub, |db, hub| {
                    db.set_node_property(hub, "by", Value::from("later"))
                })
                .unwrap();
            }
            "transaction" => {
                commit_with(&db, hub, |db, hub| {
                    let mut later = db.session();
                    later.begin_transaction().unwrap();
                    later
                        .set_node_property(hub, "by", Value::from("later"))
                        .unwrap();
                    later.commit().unwrap();
                });
            }
            other => panic!("unknown scenario {other}"),
        }
        // Crash: no close(), no destructors.
        std::process::exit(0);
    }

    /// A direct write that arrives while a transaction commits is logged
    /// after the commit, so replay ends with the direct write's value.
    #[test]
    fn a_direct_write_during_a_commit_is_replayed_after_it() {
        let dir = tempfile::tempdir().unwrap();
        for (format, path) in formats(dir.path()) {
            assert_eq!(
                by_after_crash("direct_write", &path, format),
                Some(Value::from("later")),
                "{format}"
            );
        }
    }

    /// A transaction that begins and commits while another commits is
    /// logged after it, so replay ends with its value.
    #[test]
    fn a_transaction_during_a_commit_is_replayed_after_it() {
        let dir = tempfile::tempdir().unwrap();
        for (format, path) in formats(dir.path()) {
            assert_eq!(
                by_after_crash("transaction", &path, format),
                Some(Value::from("later")),
                "{format}"
            );
        }
    }
}
