//! Checkpoints never capture part of a commit, or part of a change of an
//! open transaction.
//!
//! A checkpoint holds commits off while it builds and writes its image: it
//! waits for a commit in progress (between its epoch and its completion) to
//! complete, and no commit, transaction start or write outside a transaction
//! runs until the image is written. It also freezes the store: the writes of
//! open transactions and their rollbacks wait too, so the image is read from
//! a store and change logs that do not move. These tests start a checkpoint
//! from inside a commit, and commits, writes and rollbacks from inside a
//! checkpoint, with the `testing-statement-injection` hooks, and check what
//! the image holds. Work that should wait gets 300 ms to finish in the
//! middle; the child-process tests exit without `close()`, so a reopen reads
//! the image and what is left of the WAL.
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

use grafeo_common::storage::ChunkCaps;
use grafeo_common::testing::child_process;
use grafeo_common::testing::chunk_caps::with_chunk_caps;
use grafeo_common::testing::commit_hook::{
    after_next_commit_epoch, after_next_commit_stamped, during_next_checkpoint,
};
use grafeo_common::types::Value;
use grafeo_engine::{GrafeoDB, Session};

#[path = "common/image.rs"]
mod image;
#[path = "common/started.rs"]
mod started;

use image::image_holds;
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
/// reopen replays the commit's complete WAL group, written before stamping.
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
    assert_eq!(
        people(&db),
        [Value::from("Alix"), Value::from("Gus")],
        "the reopen replays the stamped commit's complete WAL group"
    );
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

/// Inside a checkpoint, while it holds commits off: the writes of
/// transactions that began before it (one that wrote before the checkpoint,
/// one that did not, one through the direct API), a rollback and a rollback
/// to a savepoint, each started from another thread, finish only once the
/// image is written, and then succeed.
#[test]
fn writes_and_rollbacks_of_open_transactions_wait_for_a_checkpoint() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("amsterdam.grafeo");
    database_with_alix(&path);
    let db = Arc::new(GrafeoDB::open(&path).unwrap());

    // Each work hands its session back: dropping a session whose
    // transaction is open rolls it back, which waits for the checkpoint too.
    type Work = Box<dyn FnOnce() -> (Result<(), String>, Session) + Send>;
    let begun = || {
        let mut session = db.session();
        session.begin_transaction().unwrap();
        session
    };
    let mut works: Vec<(&str, Work)> = Vec::new();

    let first_write = begun();
    works.push((
        "the first write of a transaction",
        Box::new(move || {
            let written = first_write
                .execute("MATCH (a:Person {name: 'Alix'}) SET a.city = 'Berlin'")
                .map(drop)
                .map_err(|e| e.to_string());
            (written, first_write)
        }),
    ));

    let second_write = begun();
    second_write
        .execute("INSERT (:Person {name: 'Gus'})")
        .unwrap();
    works.push((
        "a later write of a transaction that wrote before",
        Box::new(move || {
            let written = second_write
                .execute("MATCH (g:Person {name: 'Gus'}) SET g.city = 'Paris', g:Traveller")
                .map(drop)
                .map_err(|e| e.to_string());
            (written, second_write)
        }),
    ));

    let direct_write = begun();
    works.push((
        "a direct write in a transaction",
        Box::new(move || {
            let written = direct_write
                .create_node_with_props(&["Person"], [("name", Value::from("Jules"))])
                .map(drop)
                .map_err(|e| e.to_string());
            (written, direct_write)
        }),
    ));

    let mut rolled_back = begun();
    rolled_back
        .execute("INSERT (:Person {name: 'Vincent'})")
        .unwrap();
    works.push((
        "a rollback",
        Box::new(move || {
            let undone = rolled_back.rollback().map_err(|e| e.to_string());
            (undone, rolled_back)
        }),
    ));

    let to_savepoint = begun();
    to_savepoint.savepoint("before_mia").unwrap();
    to_savepoint
        .execute("INSERT (:Person {name: 'Mia'})")
        .unwrap();
    works.push((
        "a rollback to a savepoint",
        Box::new(move || {
            let undone = to_savepoint
                .rollback_to_savepoint("before_mia")
                .map_err(|e| e.to_string());
            (undone, to_savepoint)
        }),
    ));

    let (sender, started) = mpsc::channel();
    during_next_checkpoint(move || {
        let works: Vec<_> = works
            .into_iter()
            .map(|(name, work)| (name, Started::spawn(work)))
            .collect();
        let finished: Vec<_> = works
            .iter()
            .map(|(name, work)| (*name, work.finishes_briefly()))
            .collect();
        sender.send((works, finished)).unwrap();
    });
    db.wal_checkpoint().unwrap();

    let (works, finished) = started.recv().expect("the checkpoint ran the hook");
    for (name, finished) in finished {
        assert!(!finished, "{name} waits for the checkpoint");
    }
    let mut sessions = Vec::new();
    for (name, work) in works {
        let (result, session) = work.join();
        result.unwrap_or_else(|error| panic!("{name} succeeds after the checkpoint: {error}"));
        sessions.push(session);
    }
    drop(sessions);
    db.close().unwrap();
}

// =========================================================================
// Child processes: exit without close, reopen
// =========================================================================

/// The database path a child process works on.
const PATH_VAR: &str = "GRAFEO_CHECKPOINT_DURING_COMMIT_PATH";
/// Which child: "failing", "graphs", "open", "frozen" or "busy".
const CHILD_VAR: &str = "GRAFEO_CHECKPOINT_DURING_COMMIT_CHILD";
/// Exit code of a child that reached its end.
const EXITED: i32 = 19;
/// Exit code of the "frozen" child when a write or the rollback finished
/// inside the checkpoint, before its image was written.
const FINISHED_INSIDE: i32 = 88;

/// Runs the child `which` on the database at `path`.
fn run_child(which: &str, path: &Path) {
    let output = run_child_output(which, path);
    assert_eq!(
        output.status.code(),
        Some(EXITED),
        "the child {which} exited early:\n{}",
        String::from_utf8_lossy(&output.stderr)
    );
}

/// Runs the child `which` on the database at `path` and returns how it
/// ended.
fn run_child_output(which: &str, path: &Path) -> std::process::Output {
    child_process::output(
        Command::new(std::env::current_exe().unwrap())
            .args(["--exact", "checkpoint_child", "--nocapture"])
            .env(CHILD_VAR, which)
            .env(PATH_VAR, path),
    )
    .unwrap()
}

/// The sidecar WAL of the database file at `path`, which a process that ends
/// without `close()` leaves behind.
fn sidecar_wal(path: &Path) -> PathBuf {
    let mut sidecar = path.as_os_str().to_owned();
    sidecar.push(".wal");
    PathBuf::from(sidecar)
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
            after_next_commit_epoch(move || {
                let checkpoint = Started::spawn(move || checkpointer.wal_checkpoint().is_ok());
                assert!(
                    !checkpoint.finishes_briefly(),
                    "the checkpoint waits for the commit before its WAL records"
                );
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
        // The open transaction also adds a label, a property key and an
        // edge type, then a checkpoint runs (tiny chunks, so the names come
        // as rows are written) and the process ends without a commit.
        "open" => {
            session
                .execute("MATCH (a:Person {name: 'Alix'}) SET a:Explorer, a.city = 'Berlin'")
                .unwrap();
            session
                .execute(
                    "MATCH (a:Person {name: 'Alix'}), (g:Person {name: 'Gus'}) \
                     INSERT (a)-[:VISITED {year: 2019}]->(g)",
                )
                .unwrap();
            let tiny = ChunkCaps {
                max_rows: 2,
                max_bytes: 128,
            };
            with_chunk_caps(tiny, || checkpointer.wal_checkpoint()).unwrap();
            std::process::exit(EXITED);
        }
        // Inside a checkpoint, the open transaction (it created Gus) rolls
        // back, and another one, begun before the checkpoint, deletes
        // Vincent and moves Alix to Berlin, each from another thread; the
        // process ends once the checkpoint and both are done, with the
        // second transaction open. Exits with `FINISHED_INSIDE` when one
        // finished before the image was written.
        "frozen" => {
            let mut writer = db.session();
            writer.begin_transaction().unwrap();
            let (sender, started) = mpsc::channel();
            during_next_checkpoint(move || {
                let mut rolled_back = session;
                let rollback =
                    Started::spawn(move || rolled_back.rollback().map_err(|e| e.to_string()));
                let write = Started::spawn(move || {
                    let written = writer
                        .execute("MATCH (v:Person {name: 'Vincent'}) DETACH DELETE v")
                        .and_then(|_| {
                            writer.execute("MATCH (a:Person {name: 'Alix'}) SET a.city = 'Berlin'")
                        })
                        .map(drop)
                        .map_err(|e| e.to_string());
                    // The transaction stays open until the process ends.
                    (written, writer)
                });
                let rollback_finished = rollback.finishes_briefly();
                let write_finished = write.finishes_briefly();
                sender
                    .send((rollback, write, rollback_finished || write_finished))
                    .unwrap();
            });
            checkpointer.wal_checkpoint().unwrap();
            let (rollback, write, finished_inside) = started.recv().unwrap();
            rollback.join().expect("the rollback succeeds");
            let (written, _open) = write.join();
            written.expect("the writes succeed");
            std::process::exit(if finished_inside {
                FINISHED_INSIDE
            } else {
                EXITED
            });
        }
        // Transactions commit, roll back or stay open on three threads while
        // checkpoints with tiny chunks run; the process ends without close.
        "busy" => {
            drop(session);
            let stop = Arc::new(std::sync::atomic::AtomicBool::new(false));
            let workers: Vec<_> = (0..3u64)
                .map(|worker| {
                    let db = Arc::clone(&db);
                    let stop = Arc::clone(&stop);
                    std::thread::spawn(move || busy_worker(&db, &stop, worker))
                })
                .collect();
            let tiny = ChunkCaps {
                max_rows: 3,
                max_bytes: 160,
            };
            // At least three checkpoints, however long each takes on a busy
            // machine, and checkpoints for at least 600 ms.
            let start = std::time::Instant::now();
            let mut checkpoints = 0;
            while checkpoints < 3 || start.elapsed() < std::time::Duration::from_millis(600) {
                with_chunk_caps(tiny, || checkpointer.wal_checkpoint()).unwrap();
                checkpoints += 1;
            }
            stop.store(true, std::sync::atomic::Ordering::Relaxed);
            for worker in workers {
                worker.join().unwrap();
            }
            std::process::exit(EXITED);
        }
        other => panic!("unknown child {other}"),
    }
    unreachable!("the child exits from the commit");
}

/// One thread of the "busy" child: transactions that add nodes with new
/// labels and keys, set labels, delete nodes and add edges of new types,
/// and then commit, roll back or stay open for a moment before rolling back.
fn busy_worker(db: &GrafeoDB, stop: &std::sync::atomic::AtomicBool, worker: u64) {
    let mut i = worker * 1_000_000;
    while !stop.load(std::sync::atomic::Ordering::Relaxed) {
        i += 1;
        let mut session = db.session();
        if session.begin_transaction().is_err() {
            continue;
        }
        for statement in [
            format!(
                "INSERT (:L{} {{k{}: {i}, name: 'Gus {i}'}})",
                i % 13,
                i % 17
            ),
            format!(
                "MATCH (n:Person) WHERE n.n = {} SET n:M{}, n.k{} = 'v{i}'",
                i % 40,
                i % 11,
                i % 19
            ),
            format!(
                "MATCH (n:Person) WHERE n.n = {} DETACH DELETE n",
                (i * 7) % 40
            ),
            format!(
                "MATCH (a:Person), (b:Person) WHERE a.n = {} AND b.n = {} \
                 INSERT (a)-[:T{} {{w: {i}}}]->(b)",
                i % 40,
                (i + 3) % 40,
                i % 9
            ),
        ] {
            let _ = session.execute(&statement);
        }
        match i % 3 {
            0 => {
                let _ = session.commit();
            }
            1 => {
                let _ = session.rollback();
            }
            _ => {
                std::thread::sleep(std::time::Duration::from_millis(3));
                let _ = session.rollback();
            }
        }
    }
}

/// A checkpoint while a transaction is open (it created a node, a label, a
/// property key and an edge type) writes a file that opens after the
/// process ends, with the committed data and none of the open
/// transaction's changes: without `temporal` the store keeps no versions and
/// changes committed nodes in place, and the image holds them as they were
/// before the transaction all the same (#412).
#[test]
fn a_checkpoint_during_an_open_transaction_writes_a_file_that_reopens() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("amsterdam.grafeo");
    database_with_alix(&path);
    run_child("open", &path);
    let db = GrafeoDB::open(&path).unwrap();
    assert_eq!(
        people(&db),
        [Value::from("Alix")],
        "Gus, whom the open transaction created, is not in the file"
    );
    assert_eq!(
        db.execute("MATCH ()-[r]->() RETURN count(r) AS c")
            .unwrap()
            .rows()[0][0],
        Value::Int64(0),
        "the open transaction's edge is not in the file"
    );
    let alix = db
        .execute("MATCH (a:Person) RETURN labels(a) AS labels, a.city AS city")
        .unwrap()
        .rows()[0]
        .clone();
    assert_eq!(
        alix,
        [Value::List(vec![Value::from("Person")].into()), Value::Null],
        "the open transaction's label and value are not in the file"
    );
    db.close().unwrap();
}

/// Inserts Alix, in Amsterdam, who knows Vincent.
fn insert_alix_and_vincent(db: &GrafeoDB) {
    db.execute(
        "INSERT (:Person {name: 'Alix', city: 'Amsterdam'})-[:KNOWS]->(:Person {name: 'Vincent'})",
    )
    .unwrap();
}

/// Runs the "frozen" child on the database at `path` (Alix, in Amsterdam,
/// who knows Vincent) and checks the file it leaves: the image holds the
/// committed state, as the deletion of Vincent, the move of Alix and the
/// rollback waited for it, and the open transaction's changes made after it
/// are not in the WAL.
fn check_the_frozen_child(path: &Path) {
    let output = run_child_output("frozen", path);
    assert!(
        sidecar_wal(path).exists(),
        "the child ended without close(): the reopen recovers from its WAL"
    );
    let db = GrafeoDB::open(path).unwrap();
    assert_eq!(
        people(&db),
        [Value::from("Alix"), Value::from("Vincent")],
        "Vincent, whom the open transaction deleted, is in the file, Gus, whom the rolled \
         back one created, is not"
    );
    let knows = db
        .execute(
            "MATCH (a:Person {name: 'Alix'})-[:KNOWS]->(v:Person) \
             RETURN a.city AS city, v.name AS name",
        )
        .unwrap();
    assert_eq!(
        knows.rows().to_vec(),
        vec![vec![Value::from("Amsterdam"), Value::from("Vincent")]],
        "Alix is in Amsterdam and knows Vincent, as committed"
    );
    db.close().unwrap();
    assert_eq!(
        output.status.code(),
        Some(EXITED),
        "the writes and the rollback finish only once the image is written \
         (exit code {FINISHED_INSIDE}: one finished inside the checkpoint):\n{}",
        String::from_utf8_lossy(&output.stderr)
    );
}

/// A checkpoint freezes the store: an open transaction's writes (a DETACH
/// DELETE and a SET) and another one's rollback, started inside it, wait for
/// its image, which holds the committed state; the process then ends
/// without close, and the reopened file holds that state.
#[test]
fn writes_and_a_rollback_inside_a_checkpoint_wait_for_its_image() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("berlin.grafeo");
    let db = GrafeoDB::open(&path).unwrap();
    insert_alix_and_vincent(&db);
    db.close().unwrap();
    check_the_frozen_child(&path);
}

/// As [`writes_and_a_rollback_inside_a_checkpoint_wait_for_its_image`], on a
/// compacted database: Vincent and Alix are in the compacted base, which the
/// deletion and the SET (a promotion into the overlay) change.
#[cfg(feature = "compact-store")]
#[test]
fn writes_inside_a_checkpoint_of_a_compacted_database_wait_for_its_image() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("prague.grafeo");
    let mut db = GrafeoDB::open(&path).unwrap();
    insert_alix_and_vincent(&db);
    db.compact().unwrap();
    db.close().unwrap();
    check_the_frozen_child(&path);
}

/// Checkpoints with tiny chunks among transactions that commit, roll back
/// and stay open, on three threads, then a process end without close: the
/// file always opens.
#[test]
fn checkpoints_among_open_transactions_always_leave_a_file_that_opens() {
    for round in 0..3 {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("busy.grafeo");
        {
            let db = GrafeoDB::open(&path).unwrap();
            for n in 0..40 {
                db.execute(&format!("INSERT (:Person {{name: 'Alix {n}', n: {n}}})"))
                    .unwrap();
            }
            db.execute("MATCH (a:Person), (b:Person) WHERE a.n + 1 = b.n INSERT (a)-[:KNOWS]->(b)")
                .unwrap();
            db.close().unwrap();
        }
        run_child("busy", &path);
        let db = GrafeoDB::open(&path)
            .unwrap_or_else(|error| panic!("round {round}: the file does not open: {error}"));
        db.execute("MATCH (n) RETURN count(n) AS c").unwrap();
        db.close().unwrap();
    }
}

/// A process that ends inside a commit before its WAL records are written,
/// with a checkpoint started in it, leaves no part of the commit: the
/// checkpoint waited for the commit, which never completed.
#[test]
fn a_crash_inside_a_commit_with_a_checkpoint_started_leaves_none_of_it() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("prague.grafeo");
    database_with_alix(&path);
    let checkpointed = std::fs::read(&path).unwrap();
    run_child("failing", &path);
    assert_eq!(
        std::fs::read(&path).unwrap(),
        checkpointed,
        "the waiting checkpoint wrote no image before the crash"
    );
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

/// Schema changes and graph commands take effect at once and log their own
/// WAL group, outside any commit: they hold commits off for the whole
/// statement, so a checkpoint in progress finishes first and its image holds
/// none of them (no other checkpoint runs here, so the file holds that image
/// until the last one below), and once it is written they run.
#[test]
fn schema_changes_and_graph_commands_wait_for_a_checkpoint() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("rotterdam.grafeo");
    database_with_alix(&path);

    let db = Arc::new(GrafeoDB::open(&path).unwrap());
    let statements = [
        (
            "CREATE CONSTRAINT person_name FOR (p:Person) ON (p.name) UNIQUE",
            "person_name",
        ),
        ("CREATE GRAPH berlin", "berlin"),
    ];
    for (statement, made) in statements {
        let writer = Arc::clone(&db);
        let (sender, started) = mpsc::channel();
        during_next_checkpoint(move || {
            let write = Started::spawn(move || {
                writer
                    .execute(statement)
                    .map(|_| ())
                    .map_err(|e| e.to_string())
            });
            let finished = write.finishes_briefly();
            sender.send((write, finished)).unwrap();
        });
        db.wal_checkpoint().unwrap();
        let (write, finished) = started.recv().expect("the checkpoint ran the hook");
        assert!(
            !finished,
            "{statement}: waits for the checkpoint in progress"
        );
        write
            .join()
            .unwrap_or_else(|error| panic!("{statement}: runs after the checkpoint: {error}"));
        assert!(
            !image_holds(&db, made),
            "{statement}: the image of the checkpoint it waited for holds none of it"
        );
    }
    let constraints = db.execute("SHOW CONSTRAINTS").unwrap();
    assert_eq!(constraints.rows().len(), 1, "{:?}", constraints.rows());
    assert!(db.graph("berlin").is_ok(), "the graph exists");
    db.wal_checkpoint().unwrap();
    for (statement, made) in statements {
        assert!(
            image_holds(&db, made),
            "{statement}: the next checkpoint's image holds it"
        );
    }
}
