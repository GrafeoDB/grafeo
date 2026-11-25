//! A snapshot restore replaces the whole graph, so it runs only while no other
//! work is open: a transaction (read-only too), a result stream, a node or
//! edge iterator, a snapshot or AS OF view, or a running query, from any
//! session, the caller's own included. With any of them open,
//! `restore_snapshot` fails with `GRAFEO-T004` naming what is open, changes
//! nothing, and the open work reads the graph it started on to its end. Once
//! the work ends, the restore runs; work that starts during it waits, then
//! sees the restored graph whole.
//!
//! ```bash
//! cargo test -p grafeo-engine --all-features --test main restore_under_open_work::
//! ```

#![cfg(feature = "gql")]

use std::sync::Arc;
use std::sync::mpsc;
use std::time::Duration;

use grafeo_common::types::Value;
use grafeo_common::utils::error::{Error, ErrorCode};
use grafeo_engine::{Config, GrafeoDB, HistoryRetention};

/// How long a test may run: a deadlock fails it instead of hanging the run.
const DEADLINE: Duration = Duration::from_secs(120);

/// Runs `test` on a thread of its own and fails once it passes
/// [`DEADLINE`]: a restore that waits for itself is a failure, not a hang.
fn bounded(test: impl FnOnce() + Send + 'static) {
    let (done, finished) = mpsc::channel();
    let runner = std::thread::spawn(move || {
        test();
        let _ = done.send(());
    });
    match finished.recv_timeout(DEADLINE) {
        Ok(()) => runner.join().unwrap(),
        // The test panicked: raise its panic.
        Err(mpsc::RecvTimeoutError::Disconnected) => {
            std::panic::resume_unwind(runner.join().unwrap_err())
        }
        Err(mpsc::RecvTimeoutError::Timeout) => {
            panic!("did not finish within {DEADLINE:?}: a deadlock")
        }
    }
}

/// People in the database before the restore: more than two chunks of a scan.
const OLD: usize = 5_000;

/// The people in the snapshot the restore loads.
const RESTORED: [&str; 3] = ["Hans", "Jules", "Mia"];

/// A database of [`OLD`] people, keeping its history (for the AS OF view).
fn old_graph() -> GrafeoDB {
    let db = GrafeoDB::with_config(
        Config::in_memory().with_history_retention(HistoryRetention::Forever),
    )
    .unwrap();
    db.execute(&format!(
        "UNWIND range(1, {OLD}) AS i INSERT (:Person {{name: 'Vincent', i: i}})"
    ))
    .unwrap();
    db
}

/// A snapshot of a graph of the [`RESTORED`] people.
fn snapshot() -> Vec<u8> {
    let source = GrafeoDB::new_in_memory();
    for name in RESTORED {
        source
            .execute(&format!("INSERT (:Person {{name: '{name}'}})"))
            .unwrap();
    }
    source.export_snapshot().unwrap()
}

/// How many people `db` holds.
fn people(db: &GrafeoDB) -> usize {
    match &db
        .execute("MATCH (p:Person) RETURN count(p.name)")
        .unwrap()
        .rows()[0][0]
    {
        Value::Int64(count) => usize::try_from(*count).unwrap(),
        other => panic!("a count, got {other:?}"),
    }
}

/// The names in `db`, sorted.
fn names(db: &GrafeoDB) -> Vec<String> {
    db.execute("MATCH (p:Person) RETURN p.name ORDER BY p.name")
        .unwrap()
        .rows()
        .iter()
        .map(|row| match &row[0] {
            Value::String(name) => name.to_string(),
            other => panic!("a name, got {other:?}"),
        })
        .collect()
}

/// `outcome` is the T004 refusal of a restore, naming `open`.
fn assert_refused(outcome: Result<(), Error>, open: &str) {
    let error = outcome.expect_err("the restore runs only with no other work open");
    let text = error.to_string();
    assert_eq!(
        error.error_code(),
        ErrorCode::TransactionInvalidState,
        "{text}"
    );
    assert!(
        text.contains("cannot restore a snapshot while") && text.contains(open),
        "names {open}: {text}"
    );
}

/// A transaction that wrote is open across the restore: the restore is
/// refused, the transaction commits as if no restore was asked, and the
/// graph is its own.
#[test]
fn an_open_writing_transaction_refuses_the_restore_and_commits_normally() {
    bounded(|| {
        let db = old_graph();
        let snapshot = snapshot();
        let mut writer = db.session();
        writer.begin_transaction().unwrap();
        writer.execute("INSERT (:Person {name: 'Butch'})").unwrap();

        assert_refused(db.restore_snapshot(&snapshot), "1 transaction");
        writer.commit().unwrap();
        assert_eq!(people(&db), OLD + 1);

        db.restore_snapshot(&snapshot).unwrap();
        assert_eq!(names(&db), RESTORED);
    });
}

/// The same on a database file: the reopen holds the committed graph.
#[cfg(all(feature = "wal", feature = "grafeo-file"))]
#[test]
fn a_transaction_open_across_a_refused_restore_commits_and_reopens() {
    bounded(|| {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("people.grafeo");
        let db = GrafeoDB::open(&path).unwrap();
        db.execute("INSERT (:Person {name: 'Vincent'}), (:Person {name: 'Jules'})")
            .unwrap();
        let snapshot = snapshot();
        let mut writer = db.session();
        writer.begin_transaction().unwrap();
        writer.execute("INSERT (:Person {name: 'Butch'})").unwrap();
        assert_refused(db.restore_snapshot(&snapshot), "1 transaction");
        writer.commit().unwrap();
        drop(writer);
        db.close().unwrap();

        let db = GrafeoDB::open(&path).unwrap();
        assert_eq!(names(&db), ["Butch", "Jules", "Vincent"]);
    });
}

/// A read-only transaction is open work too, and reads the old graph after
/// the refusal.
#[test]
fn an_open_read_only_transaction_refuses_the_restore() {
    bounded(|| {
        let db = old_graph();
        let snapshot = snapshot();
        let mut reader = db.session();
        reader.begin_transaction().unwrap();
        assert_eq!(
            reader
                .execute("MATCH (p:Person) RETURN count(p.name)")
                .unwrap()
                .rows()[0][0],
            Value::Int64(i64::try_from(OLD).unwrap())
        );

        assert_refused(db.restore_snapshot(&snapshot), "1 transaction");
        assert_eq!(
            reader
                .execute("MATCH (p:Person) RETURN count(p.name)")
                .unwrap()
                .rows()[0][0],
            Value::Int64(i64::try_from(OLD).unwrap())
        );
        reader.rollback().unwrap();
        db.restore_snapshot(&snapshot).unwrap();
        assert_eq!(names(&db), RESTORED);
    });
}

/// A result stream over a scan of several chunks is open work: it reads the
/// old graph to its end, and the restore runs once it drops.
#[test]
fn an_open_stream_refuses_the_restore_and_reads_to_its_end() {
    bounded(|| {
        let db = old_graph();
        let snapshot = snapshot();
        let mut stream = db.execute_streaming("MATCH (p:Person) RETURN p.i").unwrap();
        let mut rows = stream
            .next_chunk()
            .unwrap()
            .expect("a first chunk")
            .row_count();
        assert!(rows < OLD, "a stream of several chunks");

        assert_refused(db.restore_snapshot(&snapshot), "1 read");
        while let Some(chunk) = stream.next_chunk().unwrap() {
            rows += chunk.row_count();
        }
        assert_eq!(rows, OLD, "the stream reads the old graph whole");
        drop(stream);

        db.restore_snapshot(&snapshot).unwrap();
        assert_eq!(names(&db), RESTORED);
    });
}

/// A node iterator is open work: it reads the old graph to its end.
#[test]
fn an_open_node_iterator_refuses_the_restore() {
    bounded(|| {
        let db = old_graph();
        let snapshot = snapshot();
        let mut nodes = db.iter_nodes().unwrap();
        assert!(nodes.next().is_some());

        assert_refused(db.restore_snapshot(&snapshot), "1 read");
        assert_eq!(1 + nodes.count(), OLD, "the iterator reads the old graph");

        db.restore_snapshot(&snapshot).unwrap();
        assert_eq!(names(&db), RESTORED);
    });
}

/// A snapshot and an AS OF view read one state while they live: each is
/// open work.
#[test]
fn an_open_snapshot_or_view_refuses_the_restore() {
    bounded(|| {
        let db = old_graph();
        let snapshot = snapshot();
        let then = db.current_epoch();

        let held = db.snapshot().unwrap();
        assert_refused(db.restore_snapshot(&snapshot), "1 read");
        drop(held);

        let view = db.as_of(then).unwrap();
        assert_refused(db.restore_snapshot(&snapshot), "1 read");
        assert_eq!(
            view.execute("MATCH (p:Person) RETURN count(p.name)")
                .unwrap()
                .rows()[0][0],
            Value::Int64(i64::try_from(OLD).unwrap())
        );
        drop(view);

        db.restore_snapshot(&snapshot).unwrap();
        assert_eq!(names(&db), RESTORED);
    });
}

/// The refusal names everything open: transactions and reads apart.
#[test]
fn the_refusal_counts_the_transactions_and_the_reads() {
    bounded(|| {
        let db = old_graph();
        let snapshot = snapshot();
        let mut first = db.session();
        first.begin_transaction().unwrap();
        let mut second = db.session();
        second.begin_transaction().unwrap();
        let nodes = db.iter_nodes().unwrap();
        assert_refused(
            db.restore_snapshot(&snapshot),
            "2 transactions and 1 read are open",
        );
        drop(nodes);
        first.rollback().unwrap();
        second.rollback().unwrap();
        db.restore_snapshot(&snapshot).unwrap();
    });
}

/// The caller's own open iterator is refused, never waited on: the restore
/// returns.
#[test]
fn the_callers_own_iterator_is_refused_not_waited_on() {
    bounded(|| {
        let db = Arc::new(old_graph());
        let snapshot = snapshot();
        let (sent, outcome) = mpsc::channel();
        let caller = Arc::clone(&db);
        std::thread::spawn(move || {
            let nodes = caller.iter_nodes().unwrap();
            let refused = caller.restore_snapshot(&snapshot);
            drop(nodes);
            sent.send(refused).unwrap();
        });
        let refused = outcome
            .recv_timeout(Duration::from_secs(60))
            .expect("the restore returns: it does not wait for its own caller");
        assert_refused(refused, "1 read");
        assert_eq!(people(&db), OLD);
    });
}

/// With nothing open the restore runs as before, on a new database and on
/// one with data.
#[test]
fn with_nothing_open_the_restore_runs() {
    bounded(|| {
        let snapshot = snapshot();
        let fresh = GrafeoDB::new_in_memory();
        fresh.restore_snapshot(&snapshot).unwrap();
        assert_eq!(names(&fresh), RESTORED);

        let db = old_graph();
        db.restore_snapshot(&snapshot).unwrap();
        assert_eq!(names(&db), RESTORED);
    });
}

/// Work another thread starts while restores run sees one graph whole, the
/// old or a restored one, never a mix: each count is the old graph's or the
/// snapshot's.
#[test]
fn work_started_during_a_restore_sees_one_graph_whole() {
    bounded(|| {
        let db = Arc::new(old_graph());
        let snapshot = snapshot();
        let stop = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let reader = {
            let db = Arc::clone(&db);
            let stop = Arc::clone(&stop);
            std::thread::spawn(move || {
                let mut counts = Vec::new();
                while !stop.load(std::sync::atomic::Ordering::Acquire) {
                    let mut session = db.session();
                    session.begin_transaction().unwrap();
                    let rows = session
                        .execute("MATCH (p:Person) RETURN count(p.name)")
                        .unwrap();
                    counts.push(rows.rows()[0][0].clone());
                    session.rollback().unwrap();
                }
                counts
            })
        };
        let mut restored = 0;
        while restored < 5 {
            // Refused while the reader's transaction is open: tried again.
            match db.restore_snapshot(&snapshot) {
                Ok(()) => restored += 1,
                Err(error) => assert_eq!(
                    error.error_code(),
                    ErrorCode::TransactionInvalidState,
                    "{error}"
                ),
            }
        }
        stop.store(true, std::sync::atomic::Ordering::Release);
        let whole = [
            Value::Int64(i64::try_from(OLD).unwrap()),
            Value::Int64(i64::try_from(RESTORED.len()).unwrap()),
        ];
        for count in reader.join().unwrap() {
            assert!(whole.contains(&count), "a mix of two graphs: {count:?}");
        }
        assert_eq!(names(&db), RESTORED);
    });
}

/// A restore of a database file runs its own checkpoint, with history off and
/// on (two ways to pin the image it writes): it returns, and the reopen holds
/// the restored graph.
#[cfg(all(feature = "wal", feature = "grafeo-file"))]
#[test]
fn a_restore_of_a_database_file_runs_and_reopens_restored() {
    bounded(|| {
        for history in [HistoryRetention::Off, HistoryRetention::Forever] {
            let dir = tempfile::tempdir().unwrap();
            let path = dir.path().join("people.grafeo");
            let db = GrafeoDB::with_config(
                Config::persistent(path.clone()).with_history_retention(history),
            )
            .unwrap();
            db.execute("INSERT (:Person {name: 'Vincent'}), (:Person {name: 'Butch'})")
                .unwrap();
            db.restore_snapshot(&snapshot()).unwrap();
            assert_eq!(names(&db), RESTORED, "{history:?}");
            db.close().unwrap();

            let db = GrafeoDB::open(&path).unwrap();
            assert_eq!(names(&db), RESTORED, "{history:?} after the reopen");
        }
    });
}

/// What every direct read of what a restore replaces answers on `db`: the
/// counts, a node and an edge with their labels, edges and histories, a
/// lookup by value, the property keys, the indexes, the graphs and the
/// graph type warnings (some answer alike on both graphs: they still pin).
fn direct_reads(db: &GrafeoDB) -> Vec<String> {
    (0..DIRECT_READS)
        .map(|read| direct_read(db, read))
        .collect()
}

/// Each direct read four other threads make while restores run answers as
/// the graph before a restore or the one after it, never a mix: a restore
/// waits for the direct reads in progress, and one that starts during its
/// refill waits for its end; no restore starves behind them.
#[test]
fn direct_reads_during_restores_see_one_graph_whole() {
    bounded(|| {
        // Two graphs that differ in every read: Vincents who know each other,
        // with an index and a named graph, against the restored people.
        let source = old_graph();
        source
            .execute("MATCH (a:Person {i: 1}), (b:Person {i: 2}) INSERT (a)-[:KNOWS]->(b)")
            .unwrap();
        source.create_property_index("name").unwrap();
        source.create_graph("trips").unwrap();
        let first = source.export_snapshot().unwrap();
        let second = snapshot();

        let db = Arc::new(GrafeoDB::new_in_memory());
        db.restore_snapshot(&first).unwrap();
        let before = direct_reads(&db);
        db.restore_snapshot(&second).unwrap();
        let after = direct_reads(&db);
        assert_ne!(before, after, "the reads tell the two graphs apart");

        let stop = Arc::new(std::sync::atomic::AtomicBool::new(false));
        // Four readers at once: no restore may starve behind them.
        let readers: Vec<_> = (0..4)
            .map(|_| {
                let db = Arc::clone(&db);
                let stop = Arc::clone(&stop);
                std::thread::spawn(move || {
                    let mut seen = Vec::new();
                    while !stop.load(std::sync::atomic::Ordering::Acquire) {
                        // Each read alone: a restore may run between two of them.
                        for (read, answer) in direct_reads(&db).into_iter().enumerate() {
                            seen.push((read, answer));
                        }
                    }
                    seen
                })
            })
            .collect();
        let mut slowest = Duration::ZERO;
        let mut restored = 0;
        while restored < 20 {
            let snapshot = if restored % 2 == 0 { &first } else { &second };
            let started = std::time::Instant::now();
            match db.restore_snapshot(snapshot) {
                Ok(()) => restored += 1,
                Err(error) => assert_eq!(
                    error.error_code(),
                    ErrorCode::TransactionInvalidState,
                    "{error}"
                ),
            }
            slowest = slowest.max(started.elapsed());
        }
        stop.store(true, std::sync::atomic::Ordering::Release);
        assert!(
            slowest < Duration::from_secs(10),
            "a restore waited {slowest:?} behind the direct reads"
        );
        for reader in readers {
            for (read, answer) in reader.join().unwrap() {
                assert!(
                    answer == before[read] || answer == after[read],
                    "read {read} saw a mix of two graphs: {answer}"
                );
            }
        }
    });
}

/// A copy that starts after the restore's first count, while the restore
/// waits for commits, pins with commits held and reads on after it lets them
/// go: the restore's final count, with commits held, finds its read and
/// refuses, the copy reads the old graph whole, and a restore after it runs.
#[cfg(all(feature = "testing-statement-injection", feature = "grafeo-file"))]
#[test]
fn a_copy_started_between_the_two_counts_refuses_the_restore() {
    use grafeo_common::testing::commit_hook;
    bounded(|| {
        let db = Arc::new(old_graph());
        let snapshot = snapshot();
        let (copy_reading, wait_for_copy) = mpsc::channel::<()>();
        let (restore_done, wait_for_restore) = mpsc::channel::<()>();
        let (copied, copy) = mpsc::channel();
        let source = Arc::clone(&db);
        commit_hook::during_next_restore_window(move || {
            std::thread::spawn(move || {
                // Runs once the copy has let commits go on, its read pinned.
                commit_hook::during_next_checkpoint_write(move || {
                    copy_reading.send(()).unwrap();
                    wait_for_restore.recv().unwrap();
                });
                copied
                    .send(source.to_memory().map(|copy| people(&copy)))
                    .unwrap();
            });
            wait_for_copy.recv().unwrap();
        });

        assert_refused(db.restore_snapshot(&snapshot), "1 read");
        restore_done.send(()).unwrap();
        assert_eq!(
            copy.recv().unwrap().unwrap(),
            OLD,
            "the copy reads the old graph whole"
        );
        assert_eq!(people(&db), OLD, "the refused restore changed nothing");

        db.restore_snapshot(&snapshot).unwrap();
        assert_eq!(names(&db), RESTORED);
    });
}

/// The direct reads of what a restore replaces, one by one (see
/// [`direct_reads`]): each pins, so one that starts inside a restore waits
/// for its end.
const DIRECT_READS: usize = 16;

/// Read `read` of [`direct_reads`] alone.
fn direct_read(db: &GrafeoDB, read: usize) -> String {
    use grafeo_common::types::{EdgeId, NodeId};
    use grafeo_engine::Direction;
    let first = NodeId::new(0);
    match read {
        0 => format!("{:?}", db.node_count()),
        1 => format!("{:?}", db.edge_count()),
        2 => format!("{:?}", db.property_key_count()),
        3 => format!(
            "{:?}",
            db.get_node(first)
                .map(|node| node.map(|node| node.get_property("name").cloned()))
        ),
        4 => format!(
            "{:?}",
            db.get_edge(EdgeId::new(0))
                .map(|edge| edge.map(|edge| edge.edge_type))
        ),
        5 => format!(
            "{:?}",
            db.get_node_edges(first, Direction::Outgoing, &[])
                .map(|edges| edges.len())
        ),
        6 => format!("{:?}", db.get_node_labels(first)),
        7 => format!(
            "{:?}",
            db.get_node_history(first).map(|versions| versions.len())
        ),
        8 => format!(
            "{:?}",
            db.get_edge_history(EdgeId::new(0))
                .map(|versions| versions.len())
        ),
        9 => format!(
            "{:?}",
            db.get_node_property_history(first, "name")
                .map(|versions| versions.len())
        ),
        10 => format!(
            "{:?}",
            db.get_all_node_property_history(first)
                .map(|keys| keys.len())
        ),
        11 => format!(
            "{:?}",
            db.find_nodes_by_property("name", &Value::from("Vincent"))
                .map(|found| found.len())
        ),
        12 => format!("{:?}", db.has_property_index("name")),
        13 => format!("{:?}", db.list_indexes().len()),
        14 => format!("{:?}", db.list_graphs()),
        15 => format!("{:?}", db.graph_type_scope_warnings().len()),
        _ => unreachable!("{DIRECT_READS} reads"),
    }
}

/// Each direct read that starts inside a restore, once it holds the direct
/// reads off and before it replaces anything, waits for the restore and
/// answers as the restored graph: a read the replace gate did not hold off
/// would answer at once, as the old graph.
#[cfg(feature = "testing-statement-injection")]
#[test]
fn each_direct_read_started_inside_a_restore_sees_the_restored_graph() {
    use grafeo_common::testing::commit_hook;
    bounded(|| {
        let source = old_graph();
        source
            .execute("MATCH (a:Person {i: 1}), (b:Person {i: 2}) INSERT (a)-[:KNOWS]->(b)")
            .unwrap();
        source.create_property_index("name").unwrap();
        source.create_graph("trips").unwrap();
        let first = source.export_snapshot().unwrap();
        let second = snapshot();

        let db = Arc::new(GrafeoDB::new_in_memory());
        // The same restores before each baseline: a restore keeps the
        // property key dictionary, so the keys count the earlier graphs too.
        db.restore_snapshot(&first).unwrap();
        db.restore_snapshot(&second).unwrap();
        let restored: Vec<String> = (0..DIRECT_READS)
            .map(|read| direct_read(&db, read))
            .collect();
        db.restore_snapshot(&first).unwrap();
        let old: Vec<String> = (0..DIRECT_READS)
            .map(|read| direct_read(&db, read))
            .collect();

        let (answered, answers) = mpsc::channel();
        let readers = Arc::clone(&db);
        commit_hook::during_next_restore_refill(move || {
            for read in 0..DIRECT_READS {
                let db = Arc::clone(&readers);
                let answered = answered.clone();
                std::thread::spawn(move || {
                    answered.send((read, direct_read(&db, read))).unwrap();
                });
            }
            // Time for a read the gate does not hold off to answer before the refill.
            std::thread::sleep(Duration::from_millis(300));
        });
        db.restore_snapshot(&second).unwrap();

        for _ in 0..DIRECT_READS {
            let (read, answer) = answers.recv().unwrap();
            assert_eq!(
                answer, restored[read],
                "read {read} answered before the restore (the old graph's: {})",
                old[read]
            );
        }
    });
}
