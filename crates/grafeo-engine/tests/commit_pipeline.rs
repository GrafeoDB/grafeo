//! Commits are pipelined (#498): a commit writes its WAL group under the
//! commit lock and releases it; it syncs and publishes outside it, in epoch
//! order. So the next commit writes its group while one syncs, one fsync
//! covers every group written up to it, and a transaction begins without
//! waiting for a commit in progress, at the published epoch, which holds no
//! part of one.
//!
//! The overlap each test needs is pinned with the commit hooks of
//! `testing-statement-injection`: a hook runs inside a commit, once its group
//! is written and the commit lock released ("written"), or once it is stamped
//! in its publish turn ("stamped"); work that must not wait gets ten seconds,
//! work that must wait 300 ms to finish anyway.
//!
//! ```bash
//! cargo test -p grafeo-engine --all-features --test commit_pipeline
//! ```

#![cfg(all(
    feature = "testing-statement-injection",
    feature = "testing-crash-injection",
    feature = "gql",
    feature = "wal",
    feature = "grafeo-file"
))]

use std::path::Path;
use std::sync::{Arc, mpsc};
use std::time::Duration;

use grafeo_common::testing::commit_hook::{after_next_commit_stamped, after_next_commit_written};
use grafeo_common::testing::crash::with_failure_at;
use grafeo_common::types::Value;
use grafeo_common::utils::error::{Error, TransactionError};
use grafeo_engine::config::DurabilityMode;
use grafeo_engine::{Config, GrafeoDB};

#[path = "common/started.rs"]
mod started;

use started::Started;

/// How long work that must not wait gets.
const PROMPTLY: Duration = Duration::from_secs(10);

fn open(path: &Path) -> Arc<GrafeoDB> {
    Arc::new(
        GrafeoDB::with_config(Config::persistent(path).with_wal_durability(DurabilityMode::Sync))
            .unwrap(),
    )
}

/// The names of the people `db` holds, sorted.
fn people(db: &GrafeoDB) -> Vec<Value> {
    db.execute("MATCH (p:Person) RETURN p.name ORDER BY p.name")
        .unwrap()
        .rows()
        .iter()
        .map(|row| row[0].clone())
        .collect()
}

fn names(names: &[&str]) -> Vec<Value> {
    names.iter().map(|name| Value::from(*name)).collect()
}

/// Commits `name` in a transaction of its own.
fn commit_person(db: &GrafeoDB, name: &str) -> grafeo_common::utils::error::Result<()> {
    let mut session = db.session();
    session.begin_transaction()?;
    session.execute(&format!("INSERT (:Person {{name: '{name}'}})"))?;
    session.commit()
}

/// Commits `name` on another thread, which reports once its commit is
/// written (its group in the log, the commit lock released), then waits
/// there until `release` is sent (or dropped).
fn commit_held_after_written(
    db: &Arc<GrafeoDB>,
    name: &'static str,
) -> (
    Started<grafeo_common::utils::error::Result<()>>,
    mpsc::Receiver<()>,
    mpsc::Sender<()>,
) {
    let (written, is_written) = mpsc::channel();
    let (release, released) = mpsc::channel::<()>();
    let db = Arc::clone(db);
    let started = Started::spawn(move || {
        after_next_commit_written(move || {
            let _ = written.send(());
            let _ = released.recv();
        });
        commit_person(&db, name)
    });
    (started, is_written, release)
}

/// Commits `name` on another thread, which reports once its commit is
/// written (its group in the log, the commit lock released).
fn commit_on_another_thread(
    db: &Arc<GrafeoDB>,
    name: &'static str,
) -> (
    Started<grafeo_common::utils::error::Result<()>>,
    mpsc::Receiver<()>,
) {
    let (written, is_written) = mpsc::channel();
    let db = Arc::clone(db);
    let started = Started::spawn(move || {
        after_next_commit_written(move || {
            let _ = written.send(());
        });
        commit_person(&db, name)
    });
    (started, is_written)
}

/// Two commits share one fsync, in `Sync` mode: the second writes its group
/// while the first is written but not synced, the first's fsync covers both,
/// and the second publishes after the first, at the next epoch, each seeing
/// its own commit once `commit()` returns.
#[test]
fn two_commits_share_one_fsync_and_publish_in_epoch_order() {
    let dir = tempfile::tempdir().unwrap();
    let db = open(&dir.path().join("amsterdam.grafeo"));
    let wal = Arc::clone(db.wal().expect("a persistent database has a WAL"));
    let before = wal.sync_count();
    let epoch_before = db.current_epoch();

    let second = Arc::new(std::sync::Mutex::new(None));
    let second_slot = Arc::clone(&second);
    let shared = Arc::clone(&db);
    after_next_commit_written(move || {
        // Alix is written and not synced: Gus writes his group meanwhile.
        let (gus, gus_written) = commit_on_another_thread(&shared, "Gus");
        gus_written
            .recv_timeout(PROMPTLY)
            .expect("Gus writes his group while Alix's is not synced");
        *second_slot.lock().unwrap() = Some(gus);
    });
    commit_person(&db, "Alix").unwrap();
    assert!(
        people(&db).contains(&Value::from("Alix")),
        "Alix's session sees her commit once it returns"
    );
    let gus = second.lock().unwrap().take().expect("Gus committed");
    gus.join().unwrap();

    assert_eq!(people(&db), names(&["Alix", "Gus"]));
    assert_eq!(
        db.current_epoch().as_u64(),
        epoch_before.as_u64() + 2,
        "two commits, two epochs"
    );
    assert_eq!(
        wal.sync_count() - before,
        1,
        "Alix's fsync covered Gus's group too"
    );
}

/// A transaction begins while a commit is written and not yet published,
/// without waiting for it, and its snapshot holds none of it.
#[test]
fn a_transaction_begins_during_a_commit_and_sees_none_of_it() {
    let dir = tempfile::tempdir().unwrap();
    let db = open(&dir.path().join("berlin.grafeo"));
    let (sender, received) = mpsc::channel();
    let shared = Arc::clone(&db);
    after_next_commit_stamped(move || {
        // Inside Alix and Mia's publish turn: stamped, not published. The
        // reader runs to its end meanwhile (it waits for nothing).
        let (read, has_read) = mpsc::channel();
        let reader = std::thread::spawn(move || {
            let mut session = shared.session();
            session.begin_transaction().unwrap();
            let in_transaction = session
                .execute("MATCH (p:Person) RETURN count(p)")
                .unwrap()
                .rows()[0][0]
                .clone();
            session.rollback().unwrap();
            let _ = read.send((people(&shared).len(), in_transaction));
        });
        sender.send(has_read.recv_timeout(PROMPTLY)).unwrap();
        reader.join().unwrap();
    });
    let mut session = db.session();
    session.begin_transaction().unwrap();
    session
        .execute("INSERT (:Person {name: 'Alix'}), (:Person {name: 'Mia'})")
        .unwrap();
    session.commit().unwrap();
    assert_eq!(
        received.recv().unwrap(),
        Ok((0, Value::Int64(0))),
        "a snapshot taken during the commit holds no part of it"
    );
    assert_eq!(people(&db), names(&["Alix", "Mia"]));
}

/// The begin of [`a_transaction_begins_during_a_commit_and_sees_none_of_it`]
/// does not wait for the commit: it finishes while the commit holds its
/// publish turn.
#[test]
fn begin_does_not_wait_for_a_commit_in_progress() {
    let dir = tempfile::tempdir().unwrap();
    let db = open(&dir.path().join("paris.grafeo"));
    let (sender, received) = mpsc::channel();
    let shared = Arc::clone(&db);
    after_next_commit_stamped(move || {
        let (began, has_begun) = mpsc::channel();
        let begin = std::thread::spawn(move || {
            let mut session = shared.session();
            let _ = began.send(session.begin_transaction().is_ok());
        });
        sender.send(has_begun.recv_timeout(PROMPTLY)).unwrap();
        begin.join().unwrap();
    });
    commit_person(&db, "Vincent").unwrap();
    assert_eq!(
        received.recv().unwrap(),
        Ok(true),
        "the begin finished while the commit was in its turn"
    );
}

/// A checkpoint started while a commit is written and not yet published
/// waits for it, then holds it whole: a reopen from the checkpoint has it.
#[test]
fn a_checkpoint_started_between_written_and_published_waits_for_the_commit() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("prague.grafeo");
    let db = open(&path);
    let (sender, received) = mpsc::channel();
    let shared = Arc::clone(&db);
    after_next_commit_written(move || {
        let checkpoint = Started::spawn(move || shared.wal_checkpoint().map_err(|e| e.to_string()));
        let finished = checkpoint.finishes_briefly();
        sender.send((checkpoint, finished)).unwrap();
    });
    commit_person(&db, "Jules").unwrap();
    let (checkpoint, finished) = received.recv().unwrap();
    assert!(!finished, "the checkpoint waits for the commit to publish");
    checkpoint.join().unwrap();
    let header = db.file_manager().unwrap().active_header();
    assert_eq!(
        header.epoch,
        db.current_epoch().as_u64(),
        "the checkpoint holds the commit"
    );
    drop(db);
}

/// A sync that fails fails every commit whose group it covered, with
/// `GRAFEO-T010` (they may be durable), and publishes none of them; a
/// reopen replays both groups, which the log holds.
#[test]
fn a_failed_sync_fails_every_commit_it_covered_and_publishes_none() {
    for count in 1..=8 {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("barcelona.grafeo");
        let db = open(&path);
        let second = Arc::new(std::sync::Mutex::new(None));
        let second_slot = Arc::clone(&second);
        let shared = Arc::clone(&db);
        after_next_commit_written(move || {
            // Mia's group is written after Butch's and waits there, so
            // Butch's sync is the one that covers both.
            let (mia, mia_written, release) = commit_held_after_written(&shared, "Mia");
            mia_written
                .recv_timeout(PROMPTLY)
                .expect("Mia writes her group");
            *second_slot.lock().unwrap() = Some((mia, release));
        });
        let first = with_failure_at(count, || commit_person(&db, "Butch"));
        let Err(error) = first else {
            panic!("the commit succeeded with failure {count} armed");
        };
        let mia = second.lock().unwrap().take();
        if !error.to_string().contains("wal:sync") {
            // Another failure point came first: try the next one.
            if let Some((mia, release)) = mia {
                drop(release);
                let _ = mia.join();
            }
            continue;
        }
        let (mia, release) = mia.expect("Mia committed");
        drop(release);
        let mia = mia.join();
        for (who, error) in [("Butch", Some(error)), ("Mia", mia.err())] {
            let error = error.unwrap_or_else(|| panic!("{who}'s commit succeeded"));
            assert!(
                matches!(
                    error,
                    Error::Transaction(TransactionError::CommitOutcomeUnknown(_))
                ),
                "{who}: {error}"
            );
        }
        assert!(people(&db).is_empty(), "neither commit is published");
        assert!(db.close().is_err(), "close reports the failed commits");
        drop(db);
        assert_eq!(
            people(&open(&path)),
            names(&["Butch", "Mia"]),
            "the reopen replays both groups"
        );
        return;
    }
    panic!("no armed failure hit the commit's sync");
}
