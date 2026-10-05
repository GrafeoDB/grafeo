//! Checkpoints never capture part of a commit.
//!
//! A checkpoint holds commits off while it builds and writes its image: it
//! waits for a commit in progress (between its epoch and its completion) to
//! complete, and no commit, transaction start or write outside a transaction
//! runs until the image is written. These tests start a checkpoint from
//! inside a commit, and commits from inside a checkpoint, with the
//! `testing-statement-injection` hooks, and check what the image holds. Work
//! that should wait gets 300 ms to finish in the middle; the child-process
//! tests exit without `close()`, so a reopen reads the image and what is left
//! of the WAL.
//!
//! ```bash
//! cargo test -p grafeo-engine --all-features --test checkpoint_during_commit
//! ```

#![cfg(all(
    feature = "testing-statement-injection",
    feature = "lpg",
    feature = "gql",
    feature = "wal",
    feature = "grafeo-file"
))]

use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::{Arc, mpsc};

use grafeo_common::testing::child_process;
use grafeo_common::testing::commit_hook::{
    after_next_commit_epoch, after_next_commit_stamped, during_next_checkpoint,
};
use grafeo_common::types::Value;
use grafeo_engine::GrafeoDB;

#[path = "common/started.rs"]
mod started;

use started::Started;

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
    let db = GrafeoDB::open(path).unwrap();
    db.execute("INSERT (:Person {name: 'Alix'})").unwrap();
    db.close().unwrap();
}

/// A checkpoint started inside a commit that then fails (after its versions
/// are stamped) waits for the commit, gets its error, and writes nothing: a
/// reopen shows the database without the commit.
#[test]
fn a_checkpoint_during_a_failing_commit_gets_its_error_and_writes_nothing() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("amsterdam.grafeo");
    database_with_alix(&path);
    let checkpointed = std::fs::read(&path).unwrap();

    let db = Arc::new(GrafeoDB::open(&path).unwrap());
    let mut session = db.session();
    session.begin_transaction().unwrap();
    session.execute("INSERT (:Person {name: 'Gus'})").unwrap();
    let (sender, started) = mpsc::channel();
    let checkpointer = Arc::clone(&db);
    after_next_commit_stamped(move || {
        let checkpoint =
            Started::spawn(move || checkpointer.wal_checkpoint().map_err(|e| e.to_string()));
        let finished = checkpoint.finishes_briefly();
        sender.send((checkpoint, finished)).unwrap();
        panic!("injected: the commit stops after stamping");
    });
    let unwound = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| session.commit()));
    assert!(unwound.is_err(), "the commit panicked");

    let (checkpoint, finished) = started.recv().expect("the commit ran the hook");
    assert!(!finished, "the checkpoint waits for the commit in progress");
    let error = checkpoint
        .join()
        .expect_err("a checkpoint after a failed commit fails");
    assert!(error.contains("did not complete"), "{error}");

    drop(session);
    assert!(db.close().is_err(), "close reports the failed commit");
    drop(db);
    assert!(
        std::fs::read(&path).unwrap() == checkpointed,
        "the file is byte for byte the last checkpoint"
    );
    let db = GrafeoDB::open(&path).unwrap();
    assert_eq!(people(&db), [Value::from("Alix")]);
}

/// A checkpoint started inside a commit that completes waits for it, and
/// its image holds the whole commit: the header's epoch and counts are the
/// commit's.
#[test]
fn a_checkpoint_during_a_commit_waits_for_it_and_holds_all_of_it() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("berlin.grafeo");
    database_with_alix(&path);

    let db = Arc::new(GrafeoDB::open(&path).unwrap());
    let mut session = db.session();
    session.begin_transaction().unwrap();
    session
        .execute("MATCH (a:Person {name: 'Alix'}) INSERT (a)-[:KNOWS]->(:Person {name: 'Gus'})")
        .unwrap();
    let (sender, started) = mpsc::channel();
    let checkpointer = Arc::clone(&db);
    after_next_commit_stamped(move || {
        let checkpoint =
            Started::spawn(move || checkpointer.wal_checkpoint().map_err(|e| e.to_string()));
        let finished = checkpoint.finishes_briefly();
        sender.send((checkpoint, finished)).unwrap();
    });
    session.commit().unwrap();

    let (checkpoint, finished) = started.recv().expect("the commit ran the hook");
    assert!(
        !finished,
        "the checkpoint does not finish in the middle of the commit"
    );
    checkpoint.join().expect("the checkpoint succeeds");
    let header = db.file_manager().unwrap().active_header();
    assert_eq!(
        (header.epoch, header.node_count, header.edge_count),
        (db.current_epoch().as_u64(), 2, 1),
        "the image is at the commit's epoch and holds Gus and the edge"
    );
    drop(session);
    db.close().unwrap();
}

/// Inside a checkpoint, while it holds commits off: a commit, a transaction
/// start and a direct write outside a transaction, each started from
/// another thread, finish only once the checkpoint is done, and its image
/// holds none of them.
#[test]
fn commits_wait_for_a_checkpoint_and_stay_out_of_its_image() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("paris.grafeo");
    database_with_alix(&path);
    let db = Arc::new(GrafeoDB::open(&path).unwrap());

    // Runs `work` from inside a checkpoint; returns whether it finished in
    // the middle, and the work.
    let during_checkpoint = |work: Box<dyn FnOnce() -> Result<(), String> + Send>| {
        let (sender, started) = mpsc::channel();
        during_next_checkpoint(move || {
            let work = Started::spawn(work);
            let finished = work.finishes_briefly();
            sender.send((work, finished)).unwrap();
        });
        db.wal_checkpoint().unwrap();
        let (work, finished) = started.recv().expect("the checkpoint ran the hook");
        let header = db.file_manager().unwrap().active_header();
        (finished, work.join(), header)
    };

    // A commit of a transaction that began before the checkpoint.
    let (go, commit_now) = mpsc::channel::<()>();
    let (ready, began) = mpsc::channel();
    let writer = Arc::clone(&db);
    let transaction = std::thread::spawn(move || {
        let mut session = writer.session();
        session.begin_transaction().unwrap();
        session.execute("INSERT (:Person {name: 'Gus'})").unwrap();
        ready.send(()).unwrap();
        commit_now.recv().unwrap();
        session.commit().map_err(|e| e.to_string())
    });
    began.recv().unwrap();
    let (finished, committed, header) = during_checkpoint(Box::new(move || {
        go.send(()).unwrap();
        transaction.join().unwrap()
    }));
    assert!(!finished, "the commit waits for the checkpoint");
    committed.expect("the commit succeeds after the checkpoint");
    assert_eq!(header.node_count, 1, "the image holds Alix only");

    // A transaction start.
    let reader = Arc::clone(&db);
    let (finished, began, _) = during_checkpoint(Box::new(move || {
        let mut session = reader.session();
        session.begin_transaction().map_err(|e| e.to_string())
    }));
    assert!(!finished, "a transaction starts after the checkpoint");
    began.expect("the transaction starts after the checkpoint");

    // A direct write outside a transaction.
    let writer = Arc::clone(&db);
    let (finished, written, header) = during_checkpoint(Box::new(move || {
        writer
            .create_node_with_props(&["Person"], [("name", Value::from("Vincent"))])
            .map(drop)
            .map_err(|e| e.to_string())
    }));
    assert!(!finished, "the direct write waits for the checkpoint");
    written.expect("the direct write succeeds after the checkpoint");
    assert_eq!(header.node_count, 2, "the image holds Alix and Gus only");
    assert_eq!(
        db.node_count(),
        3,
        "Vincent is written after the checkpoint"
    );
    db.close().unwrap();
}

// =========================================================================
// Child processes: exit without close, reopen
// =========================================================================

/// The database path a child process works on.
const PATH_VAR: &str = "GRAFEO_CHECKPOINT_DURING_COMMIT_PATH";
/// Which child: "failing" or "graphs".
const CHILD_VAR: &str = "GRAFEO_CHECKPOINT_DURING_COMMIT_CHILD";
/// Exit code of a child that reached its end.
const EXITED: i32 = 19;

/// Runs the child `which` on the database at `path`.
fn run_child(which: &str, path: &Path) {
    let output = child_process::output(
        Command::new(std::env::current_exe().unwrap())
            .args(["--exact", "checkpoint_child", "--nocapture"])
            .env(CHILD_VAR, which)
            .env(PATH_VAR, path),
    )
    .unwrap();
    assert_eq!(
        output.status.code(),
        Some(EXITED),
        "the child {which} exited early:\n{}",
        String::from_utf8_lossy(&output.stderr)
    );
}

/// Child-process entry for [`run_child`]; a no-op when run directly.
#[test]
fn checkpoint_child() {
    let (Ok(which), Some(path)) = (std::env::var(CHILD_VAR), std::env::var_os(PATH_VAR)) else {
        return;
    };
    let db = Arc::new(GrafeoDB::open(PathBuf::from(path)).unwrap());
    let mut session = db.session();
    session.begin_transaction().unwrap();
    session.execute("INSERT (:Person {name: 'Gus'})").unwrap();
    let checkpointer = Arc::clone(&db);
    match which.as_str() {
        // The process ends inside the commit, before its WAL records are
        // written, with a checkpoint started meanwhile.
        "failing" => {
            after_next_commit_stamped(move || {
                let checkpoint = Started::spawn(move || checkpointer.wal_checkpoint().is_ok());
                let _ = checkpoint.finishes_briefly();
                std::process::exit(EXITED);
            });
            let _ = session.commit();
        }
        // A commit over the default graph and a named graph, with a
        // checkpoint started right after its epoch; the process ends after
        // both, without close.
        "graphs" => {
            session.use_graph("trips");
            session.execute("INSERT (:City {name: 'Prague'})").unwrap();
            session.reset_graph();
            let (sender, started) = mpsc::channel();
            after_next_commit_epoch(move || {
                let checkpoint = Started::spawn(move || checkpointer.wal_checkpoint().is_ok());
                let finished = checkpoint.finishes_briefly();
                sender.send((checkpoint, finished)).unwrap();
            });
            session.commit().unwrap();
            let (checkpoint, finished) = started.recv().unwrap();
            assert!(checkpoint.join(), "the checkpoint succeeds");
            assert!(
                !finished,
                "the checkpoint does not finish in the middle of the commit"
            );
            std::process::exit(EXITED);
        }
        other => panic!("unknown child {other}"),
    }
    unreachable!("the child exits from the commit");
}

/// A process that ends inside a commit, with a checkpoint started in it,
/// leaves no part of the commit: the checkpoint waited for the commit, which
/// never completed.
#[test]
fn a_crash_inside_a_commit_with_a_checkpoint_started_leaves_none_of_it() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("prague.grafeo");
    database_with_alix(&path);
    run_child("failing", &path);
    let db = GrafeoDB::open(&path).unwrap();
    assert_eq!(
        people(&db),
        [Value::from("Alix")],
        "the reopened database has nothing of the commit"
    );
}

/// A commit over the default graph and a named graph, with a checkpoint
/// started right after its epoch: the image holds all of the commit or none
/// of it (here all: the checkpoint waits for the commit), and the WAL the
/// checkpoint truncated no longer holds it.
#[test]
fn a_checkpoint_holds_a_commit_over_two_graphs_whole() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("barcelona.grafeo");
    {
        let db = GrafeoDB::open(&path).unwrap();
        db.execute("INSERT (:Person {name: 'Alix'})").unwrap();
        db.create_graph("trips").unwrap();
        db.graph("trips")
            .unwrap()
            .execute("INSERT (:City {name: 'Paris'})")
            .unwrap();
        db.close().unwrap();
    }
    run_child("graphs", &path);

    let db = GrafeoDB::open(&path).unwrap();
    let cities: Vec<Value> = db
        .graph("trips")
        .unwrap()
        .execute("MATCH (c:City) RETURN c.name AS name ORDER BY name")
        .unwrap()
        .rows()
        .iter()
        .map(|row| row[0].clone())
        .collect();
    assert_eq!(
        (people(&db), cities),
        (
            vec![Value::from("Alix"), Value::from("Gus")],
            vec![Value::from("Paris"), Value::from("Prague")]
        ),
        "the reopened database holds the whole commit, in both graphs"
    );
}
