//! A commit completes before anything that comes after it (#548), and work
//! that starts while it is in progress sees none of it.
//!
//! A commit assigns its epoch and writes its WAL group, then stamps its
//! versions, reports its CDC events and publishes its epoch, in epoch order.
//! These tests run work on another thread from inside a commit, right after
//! its epoch is assigned or once its versions are stamped (the
//! `testing-statement-injection` commit hooks): the work does not wait for
//! the commit, reads at the published epoch, which holds none of it, and
//! conflicts on what the commit wrote; work that starts after `commit()`
//! returns sees it. The WAL tests crash a child process (it exits without
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

use grafeo_common::testing::commit_hook::{after_next_commit_epoch, after_next_commit_stamped};
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

/// Where in the next commit the work starts.
#[derive(Clone, Copy)]
enum Moment {
    /// Right after the commit epoch is assigned, before anything is stamped.
    EpochAssigned,
    /// Once the commit's versions are stamped, before it is complete.
    Stamped,
}

/// How long the commit waits for the work it started.
#[derive(Clone, Copy)]
enum Wait {
    /// Until the work is done, for work that never waits for the commit: it
    /// always runs in the middle of the commit.
    UntilDone,
    /// 300 ms, for work that should wait for the commit: if it does not, it
    /// has that long to land in the middle of the commit.
    Briefly,
}

/// Starts `work` on another thread from inside the next commit on this
/// thread, at `moment`, and waits for it as `wait` says.
fn during_next_commit_at<T: Send + 'static>(
    moment: Moment,
    wait: Wait,
    work: impl FnOnce() -> T + Send + 'static,
) -> DuringCommit<T> {
    let slot = Arc::new(Mutex::new(None));
    let handle_slot = Arc::clone(&slot);
    let hook = move || {
        let (started, running) = mpsc::channel();
        let (done, finished) = mpsc::channel();
        let handle = std::thread::spawn(move || {
            let _ = started.send(());
            let result = work();
            let _ = done.send(());
            result
        });
        // Time the wait from the moment the worker runs, so a worker that is
        // scheduled late cannot slip past the commit unnoticed.
        running
            .recv_timeout(Duration::from_secs(10))
            .expect("the worker did not start");
        match wait {
            Wait::UntilDone => finished
                .recv_timeout(Duration::from_secs(10))
                .expect("the work did not finish during the commit"),
            Wait::Briefly => {
                let _ = finished.recv_timeout(Duration::from_millis(300));
            }
        }
        *handle_slot.lock().unwrap() = Some(handle);
    };
    match moment {
        Moment::EpochAssigned => after_next_commit_epoch(hook),
        Moment::Stamped => after_next_commit_stamped(hook),
    }
    DuringCommit(slot)
}

/// [`during_next_commit_at`] right after the commit epoch is assigned, for
/// work that should wait for the commit.
fn during_next_commit<T: Send + 'static>(
    work: impl FnOnce() -> T + Send + 'static,
) -> DuringCommit<T> {
    during_next_commit_at(Moment::EpochAssigned, Wait::Briefly, work)
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

/// A direct write that arrives while a transaction commits, to what the
/// commit wrote, conflicts (`GRAFEO-T001`, retryable) and changes nothing;
/// once `commit()` returned, the same write lands after the commit: the
/// committed value holds at the commit's epoch and the direct write's value
/// is the latest.
#[test]
fn a_direct_write_during_a_commit_conflicts_and_after_it_lands() {
    // History kept: the test reads the commit's epoch after a later write.
    let db = Arc::new(
        GrafeoDB::with_config(
            grafeo_engine::Config::in_memory()
                .with_history_retention(grafeo_engine::HistoryRetention::Forever),
        )
        .unwrap(),
    );
    let hub = db.create_node(&["Hub"]).unwrap();
    let commit_epoch = grafeo_common::types::EpochId::new(db.current_epoch().as_u64() + 1);
    let during = commit_with(&db, hub, |db, hub| {
        db.set_node_property(hub, "by", Value::from("direct"))
            .map_err(|error| error.to_string())
    });
    let error = during.unwrap_err();
    assert!(error.contains("GRAFEO-T001"), "{error}");
    assert_eq!(by(&db, hub), Some(Value::from("transaction")));

    db.set_node_property(hub, "by", Value::from("direct"))
        .unwrap();
    assert_eq!(
        db.get_node_property_at_epoch(hub, "by", commit_epoch)
            .unwrap(),
        Some(Value::from("transaction"))
    );
    assert_eq!(by(&db, hub), Some(Value::from("direct")));
}

/// A transaction that begins while another commits does not wait for it and
/// does not see it (it reads at the published epoch); one that begins once
/// `commit()` returned sees what the commit created.
#[test]
fn a_transaction_during_a_commit_does_not_see_it_and_one_after_it_does() {
    let db = Arc::new(GrafeoDB::new_in_memory());
    let mut session = db.session();
    session.begin_transaction().unwrap();
    let doc = session.create_node(&["Doc"]).unwrap();
    let sees = move |db: &GrafeoDB| {
        let mut reader = db.session();
        reader.begin_transaction().unwrap();
        let seen = reader.get_node(doc).is_some();
        // A commit, even of nothing, takes the next epoch and so waits for
        // the commit in progress: a read rolls back.
        reader.rollback().unwrap();
        seen
    };
    let reader = {
        let db = Arc::clone(&db);
        during_next_commit_at(Moment::Stamped, Wait::UntilDone, move || sees(&db))
    };
    session.commit().unwrap();

    assert!(!reader.join(), "a transaction during the commit saw it");
    assert!(sees(&db), "a transaction after the commit did not see it");
}

/// A read outside a transaction does not see a commit before it is complete,
/// even once the commit's versions are stamped.
#[test]
fn a_plain_read_does_not_see_a_commit_before_it_completes() {
    let db = Arc::new(GrafeoDB::new_in_memory());
    let count =
        |db: &GrafeoDB| db.execute("MATCH (d:Doc) RETURN count(d)").unwrap().rows()[0][0].clone();
    let mut session = db.session();
    session.begin_transaction().unwrap();
    session.execute("INSERT (:Doc {id: 1})").unwrap();
    let read = {
        let db = Arc::clone(&db);
        during_next_commit_at(Moment::Stamped, Wait::UntilDone, move || count(&db))
    };
    session.commit().unwrap();

    assert_eq!(read.join(), Value::Int64(0));
    assert_eq!(count(&db), Value::Int64(1));
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

#[cfg(all(feature = "wal", feature = "grafeo-file"))]
mod wal {
    use super::*;
    use grafeo_common::testing::child_process;
    use grafeo_engine::Config;
    use grafeo_engine::config::StorageFormat;

    const SCENARIO_VAR: &str = "GRAFEO_COMMIT_COMPLETION_SCENARIO";
    const PATH_VAR: &str = "GRAFEO_COMMIT_COMPLETION_PATH";

    fn open(path: &Path) -> GrafeoDB {
        GrafeoDB::with_config(Config::persistent(path).with_storage_format(StorageFormat::Auto))
            .unwrap()
    }

    /// Runs `scenario` in a child process that exits without closing the
    /// database, then reopens it and returns the `by` of the hub and of the
    /// other node (the scenario creates both first).
    fn by_after_crash(scenario: &str, path: &Path) -> (Option<Value>, Option<Value>) {
        let status = child_process::run(
            std::process::Command::new(std::env::current_exe().unwrap())
                .args(["--exact", "wal::crash_child", "--nocapture"])
                .env(SCENARIO_VAR, scenario)
                .env(PATH_VAR, path),
        )
        .unwrap();
        assert!(status.success(), "scenario {scenario} failed");
        let db = open(path);
        let by_of = |label: &str| {
            let id = db
                .execute(&format!("MATCH (n:{label}) RETURN id(n)"))
                .unwrap()
                .rows()[0][0]
                .clone();
            let Value::Int64(id) = id else {
                panic!("no {label}: {id:?}");
            };
            by(&db, NodeId::new(u64::try_from(id).unwrap()))
        };
        (by_of("Hub"), by_of("Other"))
    }

    /// Child-process entry for [`by_after_crash`]; a no-op when run directly.
    #[test]
    fn crash_child() {
        let Ok(scenario) = std::env::var(SCENARIO_VAR) else {
            return;
        };
        let path = PathBuf::from(std::env::var_os(PATH_VAR).unwrap());
        let db = Arc::new(open(&path));
        let hub = db.create_node(&["Hub"]).unwrap();
        let other = db.create_node(&["Other"]).unwrap();
        match scenario.as_str() {
            "direct_write" => {
                commit_with(&db, hub, move |db, _| {
                    db.set_node_property(other, "by", Value::from("later"))
                })
                .unwrap();
            }
            "transaction" => {
                commit_with(&db, hub, move |db, _| {
                    let mut later = db.session();
                    later.begin_transaction().unwrap();
                    later
                        .set_node_property(other, "by", Value::from("later"))
                        .unwrap();
                    later.commit().unwrap();
                });
            }
            other => panic!("unknown scenario {other}"),
        }
        // Crash: no close(), no destructors.
        std::process::exit(0);
    }

    /// A direct write that commits while a transaction commits (another
    /// node) is logged as its own group, so replay has both.
    #[test]
    fn a_direct_write_during_a_commit_is_replayed_with_it() {
        let dir = tempfile::tempdir().unwrap();
        assert_eq!(
            by_after_crash("direct_write", &dir.path().join("single.grafeo")),
            (Some(Value::from("transaction")), Some(Value::from("later")))
        );
    }

    /// A transaction that begins and commits while another commits is
    /// logged after it, so replay has both.
    #[test]
    fn a_transaction_during_a_commit_is_replayed_with_it() {
        let dir = tempfile::tempdir().unwrap();
        assert_eq!(
            by_after_crash("transaction", &dir.path().join("single.grafeo")),
            (Some(Value::from("transaction")), Some(Value::from("later")))
        );
    }
}
