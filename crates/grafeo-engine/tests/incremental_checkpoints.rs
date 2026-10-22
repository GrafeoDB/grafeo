//! Incremental checkpoints without the write freeze (H3; #430, #449).
//!
//! A checkpoint writes only the chunks that changed since the image before
//! it, the new image pointing at the unchanged ones where they are, and lets
//! commits go on while it writes. These tests count what checkpoints write
//! (the file manager's last checkpoint statistics), check that the image of
//! a checkpoint is the committed state at its epoch, across reopens, with
//! history kept and with writers busy meanwhile, and crash a checkpoint at
//! each of its points while a commit lands in the middle of it.
//!
//! ```bash
//! cargo test -p grafeo-engine --all-features --test incremental_checkpoints
//! ```

#![cfg(all(feature = "gql", feature = "wal", feature = "grafeo-file"))]

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

use grafeo_common::types::{EpochId, NodeId, Value};
use grafeo_engine::config::HistoryRetention;
use grafeo_engine::{Config, GrafeoDB};
use grafeo_storage::file::CheckpointStats;

/// People `0..count`, each with a number and a city: more than 65,536 fill
/// two row groups of the node table.
fn insert_people(db: &GrafeoDB, count: u64) {
    db.execute(&format!(
        "UNWIND range(0, {}) AS i INSERT (:Person {{i: i, city: 'Amsterdam'}})",
        count - 1
    ))
    .unwrap();
}

/// What the last checkpoint of `db` wrote and reused.
fn last_checkpoint(db: &GrafeoDB) -> CheckpointStats {
    db.file_manager()
        .unwrap()
        .last_checkpoint_stats()
        .expect("a checkpoint ran")
}

/// The single value `query` returns.
fn single(db: &GrafeoDB, query: &str) -> Value {
    db.execute(query).unwrap().rows()[0][0].clone()
}

/// After one value set in a large database, a checkpoint writes the chunk of
/// that value's column and row group and the LPG section's metadata, and
/// reuses every other chunk (the catalog's too, which did not change); the
/// file grows by about what it wrote, and a reopen reads the change. The
/// reopened database follows the image it loaded: its next checkpoint is
/// incremental too.
#[test]
fn one_value_set_writes_its_chunk_alone() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("amsterdam.grafeo");
    let db = GrafeoDB::open(&path).unwrap();
    insert_people(&db, 70_000);
    db.wal_checkpoint().unwrap();
    // Every chunk of the people is new; the catalog's are those the new
    // file was created with.
    let full = last_checkpoint(&db);
    assert!(full.chunks_written >= 6, "{full:?}");

    db.execute("MATCH (p:Person {i: 69000}) SET p.city = 'Berlin'")
        .unwrap();
    let size = db.file_manager().unwrap().file_size().unwrap();
    db.wal_checkpoint().unwrap();
    let incremental = last_checkpoint(&db);
    assert_eq!(
        incremental.chunks_written, 2,
        "the city chunk of the second row group and the metadata: {incremental:?}"
    );
    assert_eq!(
        incremental.chunks_reused,
        full.chunks_written + full.chunks_reused - 2
    );
    assert!(
        incremental.bytes_written * 10 < full.bytes_written,
        "{incremental:?} against {full:?}"
    );
    let grown = db
        .file_manager()
        .unwrap()
        .file_size()
        .unwrap()
        .saturating_sub(size);
    assert!(
        grown <= incremental.bytes_written + 4 * 4096,
        "the file grew by {grown} bytes for {incremental:?}"
    );
    db.close().unwrap();

    let db = GrafeoDB::open(&path).unwrap();
    assert_eq!(
        single(&db, "MATCH (p:Person {i: 69000}) RETURN p.city"),
        Value::from("Berlin")
    );
    assert_eq!(
        single(&db, "MATCH (p:Person) RETURN count(p)"),
        Value::Int64(70_000)
    );
    db.execute("MATCH (p:Person {i: 5}) SET p.city = 'Paris'")
        .unwrap();
    db.wal_checkpoint().unwrap();
    let after_reopen = last_checkpoint(&db);
    assert_eq!(after_reopen.chunks_written, 2, "{after_reopen:?}");
    db.close().unwrap();
    let db = GrafeoDB::open(&path).unwrap();
    assert_eq!(
        single(&db, "MATCH (p:Person {i: 5}) RETURN p.city"),
        Value::from("Paris")
    );
    assert_eq!(
        single(&db, "MATCH (p:Person {i: 69000}) RETURN p.city"),
        Value::from("Berlin")
    );
    db.close().unwrap();
}

/// A checkpoint with nothing changed writes no data chunk; creates, deletes
/// and edges write their row groups; every reopen reads what was committed.
#[test]
fn every_kind_of_change_reads_back_after_incremental_checkpoints() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("berlin.grafeo");
    let db = GrafeoDB::open(&path).unwrap();
    insert_people(&db, 70_000);
    db.wal_checkpoint().unwrap();
    db.wal_checkpoint().unwrap();
    let idle = last_checkpoint(&db);
    assert_eq!(idle.chunks_written, 1, "the metadata alone: {idle:?}");

    for statement in [
        "MATCH (a:Person {i: 3}), (b:Person {i: 69999}) INSERT (a)-[:KNOWS {since: 1994}]->(b)",
        "MATCH (p:Person {i: 69998}) DETACH DELETE p",
        "INSERT (:Person:Boxer {i: 70000, city: 'Prague'})",
        "MATCH (p:Person {i: 7}) SET p:Boxer",
        "MATCH (p:Person {i: 8}) REMOVE p.city",
    ] {
        db.execute(statement).unwrap();
        db.wal_checkpoint().unwrap();
        assert!(
            last_checkpoint(&db).chunks_reused > 0,
            "{statement}: {:?}",
            last_checkpoint(&db)
        );
    }
    db.close().unwrap();

    let db = GrafeoDB::open(&path).unwrap();
    assert_eq!(
        single(&db, "MATCH (p:Person) RETURN count(p)"),
        Value::Int64(70_000)
    );
    assert_eq!(
        single(&db, "MATCH (b:Boxer) RETURN count(b)"),
        Value::Int64(2)
    );
    assert_eq!(
        single(
            &db,
            "MATCH (:Person {i: 3})-[k:KNOWS]->(b:Person) RETURN b.i * 10000 + k.since"
        ),
        Value::Int64(69999 * 10000 + 1994)
    );
    assert_eq!(
        single(&db, "MATCH (p:Person {i: 8}) RETURN p.city"),
        Value::Null
    );
    assert_eq!(
        single(&db, "MATCH (p:Person {i: 69998}) RETURN count(p)"),
        Value::Int64(0)
    );
    db.close().unwrap();
}

/// With history kept, a checkpoint reuses the row groups that did not
/// change, history chunks included, and a reopen reads every epoch of the
/// window as before.
#[test]
fn history_survives_reused_row_groups() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("prague.grafeo");
    let config = || Config::persistent(&path).with_history_retention(HistoryRetention::Forever);
    let db = GrafeoDB::with_config(config()).unwrap();
    insert_people(&db, 70_000);
    let first = NodeId::new(0);
    let late = NodeId::new(69_000);
    db.set_node_property(first, "city", Value::from("Berlin"))
        .unwrap();
    let berlin = db.current_epoch();
    db.set_node_property(first, "city", Value::from("Paris"))
        .unwrap();
    let paris = db.current_epoch();
    db.wal_checkpoint().unwrap();
    db.set_node_property(late, "city", Value::from("Prague"))
        .unwrap();
    let prague = db.current_epoch();
    db.wal_checkpoint().unwrap();
    let incremental = last_checkpoint(&db);
    assert!(
        incremental.chunks_reused > 0,
        "the first row group is reused: {incremental:?}"
    );
    db.close().unwrap();

    let db = GrafeoDB::with_config(config()).unwrap();
    let city = |id: NodeId, epoch: EpochId| {
        db.get_node_at_epoch(id, epoch)
            .unwrap()
            .and_then(|node| node.get_property("city").cloned())
    };
    assert_eq!(city(first, berlin), Some(Value::from("Berlin")));
    assert_eq!(city(first, paris), Some(Value::from("Paris")));
    assert_eq!(city(late, paris), Some(Value::from("Amsterdam")));
    assert_eq!(city(late, prague), Some(Value::from("Prague")));
    db.close().unwrap();
}

/// Writers commit while checkpoints run; every commit is there after a
/// reopen, once.
#[test]
fn checkpoints_during_concurrent_writers_lose_nothing() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("barcelona.grafeo");
    let db = Arc::new(GrafeoDB::open(&path).unwrap());
    insert_people(&db, 70_000);
    db.wal_checkpoint().unwrap();
    let stop = Arc::new(AtomicBool::new(false));
    let written = Arc::new(AtomicU64::new(0));
    let writers: Vec<_> = (0..3_u64)
        .map(|writer| {
            let (db, stop, written) = (Arc::clone(&db), Arc::clone(&stop), Arc::clone(&written));
            std::thread::spawn(move || {
                let mut n = 0_u64;
                while !stop.load(Ordering::Acquire) {
                    let id = (writer * 20_000 + n * 7) % 70_000;
                    db.execute(&format!(
                        "MATCH (p:Person {{i: {id}}}) SET p.visits = {n} \
                         INSERT (:Visit {{writer: {writer}, n: {n}}})"
                    ))
                    .unwrap();
                    n += 1;
                    written.fetch_add(1, Ordering::AcqRel);
                }
            })
        })
        .collect();
    // Commits no longer wait for a checkpoint, so five checkpoints can end
    // before a writer commits once: each one starts after the writers
    // committed more, so every checkpoint overlaps writes.
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(300);
    for _ in 0..5 {
        let before = written.load(Ordering::Acquire);
        while written.load(Ordering::Acquire) == before {
            assert!(
                std::time::Instant::now() < deadline,
                "the writers stopped committing"
            );
            std::thread::yield_now();
        }
        db.wal_checkpoint().unwrap();
    }
    stop.store(true, Ordering::Release);
    for writer in writers {
        writer.join().unwrap();
    }
    let visits = written.load(Ordering::Acquire);
    assert!(visits > 0, "the writers wrote");
    let db = Arc::into_inner(db).unwrap();
    db.close().unwrap();

    let db = GrafeoDB::open(&path).unwrap();
    assert_eq!(
        single(&db, "MATCH (v:Visit) RETURN count(v)"),
        Value::Int64(i64::try_from(visits).unwrap())
    );
    assert_eq!(
        single(&db, "MATCH (p:Person) RETURN count(p)"),
        Value::Int64(70_000)
    );
    db.close().unwrap();
}

/// A checkpoint that panics at any of its points (in the process, so the
/// database lives on) takes the marks of what changed with it: the next
/// checkpoint writes every chunk, and nothing written before the panic is
/// lost, also once the WAL the next checkpoint covers is gone. Points 2 to 4
/// lie inside the WAL's segment rotation, after which an in-process panic
/// leaves a segment the next rotation cannot create again; the reopen
/// sweeps of `crash_injection_single_file.rs` cover them.
#[cfg(feature = "testing-crash-injection")]
#[test]
fn a_checkpoint_that_panics_leaves_the_next_one_whole() {
    for point in std::iter::once(1).chain(5..=14_u64) {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("panic.grafeo");
        let db = GrafeoDB::open(&path).unwrap();
        insert_people(&db, 1_000);
        db.wal_checkpoint().unwrap();
        // A value alone: the row group keeps its row count, so nothing but
        // the marks tells it changed.
        db.execute("MATCH (p:Person {i: 7}) SET p.city = 'Berlin'")
            .unwrap();
        let target = std::panic::AssertUnwindSafe(&db);
        let _ = grafeo_common::testing::crash::with_crash_at(point, move || {
            let _ = target.wal_checkpoint();
        });
        db.wal_checkpoint().unwrap();
        db.close().unwrap();
        // Only the image: the close removed the WAL.
        let db = GrafeoDB::open(&path).unwrap();
        assert_eq!(
            single(&db, "MATCH (p:Person {i: 7}) RETURN p.city"),
            Value::from("Berlin"),
            "point {point}"
        );
        assert_eq!(
            single(&db, "MATCH (p:Person) RETURN count(p)"),
            Value::Int64(1_000),
            "point {point}"
        );
        db.close().unwrap();
    }
}
/// Opens the database at `path` and returns its people's numbers, sorted.
#[cfg(all(
    feature = "testing-crash-injection",
    feature = "testing-statement-injection"
))]
fn numbers(path: &std::path::Path) -> Vec<i64> {
    let db = GrafeoDB::open(path).unwrap();
    let numbers = db
        .execute("MATCH (p:Person) RETURN p.i AS i ORDER BY i")
        .unwrap()
        .rows()
        .iter()
        .map(|row| match row[0] {
            Value::Int64(i) => i,
            ref other => panic!("{other:?}"),
        })
        .collect();
    db.close().unwrap();
    numbers
}

/// A crash at each point of an incremental checkpoint, with a commit landing
/// while the checkpoint writes its image: the reopen holds every commit,
/// whichever image the crash left (the database is built outside the crash,
/// and the child process exits without closing it).
#[cfg(all(
    feature = "testing-crash-injection",
    feature = "testing-statement-injection"
))]
mod crash {
    use std::path::PathBuf;

    use grafeo_common::testing::child_process;
    use grafeo_engine::config::HistoryRetention;
    use grafeo_engine::{Config, GrafeoDB};

    use super::numbers;

    const POINT_VAR: &str = "GRAFEO_INCREMENTAL_CRASH_POINT";
    const PATH_VAR: &str = "GRAFEO_INCREMENTAL_CRASH_PATH";
    const HISTORY_VAR: &str = "GRAFEO_INCREMENTAL_CRASH_HISTORY";
    /// Exit code of a child whose checkpoint ran to its end.
    const COMPLETED: i32 = 7;
    /// What the child prints once the commit in the middle of the checkpoint
    /// is done: the crash came after it.
    const WROTE_DURING: &str = "wrote during the checkpoint";
    /// The people: more than a row group of the node table, so the second
    /// checkpoint reuses the second one.
    const PEOPLE: i64 = 70_000;

    /// The configuration of the database at `path`, keeping history or not.
    fn config(path: PathBuf, history: bool) -> Config {
        let config = Config::persistent(path);
        if history {
            config.with_history_retention(HistoryRetention::Forever)
        } else {
            config
        }
    }

    /// Child-process entry; a no-op when run directly.
    #[test]
    fn crash_child() {
        let (Ok(point), Some(path)) = (std::env::var(POINT_VAR), std::env::var_os(PATH_VAR)) else {
            return;
        };
        let point: u64 = point.parse().unwrap();
        let history = std::env::var(HISTORY_VAR).is_ok_and(|value| value == "1");
        let db = std::sync::Arc::new(
            GrafeoDB::with_config(config(PathBuf::from(path), history)).unwrap(),
        );
        db.execute(&format!(
            "UNWIND range(0, {}) AS i INSERT (:Person {{i: i}})",
            PEOPLE - 1
        ))
        .unwrap();
        db.wal_checkpoint().unwrap();
        db.execute("MATCH (p:Person {i: 5}) SET p.i = -5").unwrap();
        let writer = std::sync::Arc::clone(&db);
        grafeo_common::testing::commit_hook::during_next_checkpoint_write(move || {
            // On a thread of its own: its crash points are not counted.
            std::thread::spawn(move || {
                writer
                    .execute(&format!("INSERT (:Person {{i: {PEOPLE}}})"))
                    .unwrap();
            })
            .join()
            .unwrap();
            println!("{WROTE_DURING}");
        });
        let target = std::panic::AssertUnwindSafe(&db);
        let result =
            grafeo_common::testing::crash::with_crash_at(point, move || target.wal_checkpoint());
        if let grafeo_common::testing::crash::CrashResult::Completed(checkpoint) = result {
            checkpoint.unwrap();
            std::process::exit(COMPLETED);
        }
        // Crash: no close(), no destructors.
        std::process::exit(0);
    }

    /// A crash at each point of an incremental checkpoint, with history kept
    /// (`history`) or not: the reopen holds every commit, the one in the
    /// middle of the checkpoint once it was done.
    fn crash_sweep(history: bool) {
        let mut before: Vec<i64> = (0..PEOPLE).filter(|i| *i != 5).collect();
        before.insert(0, -5);
        let mut during = before.clone();
        during.push(PEOPLE);
        let (mut completed, mut crashed_after_the_commit) = (false, false);
        for point in 1..=20_u64 {
            let dir = tempfile::tempdir().unwrap();
            let path = dir.path().join("crash.grafeo");
            let output = child_process::output(
                std::process::Command::new(std::env::current_exe().unwrap())
                    .args(["--exact", "crash::crash_child", "--nocapture"])
                    .env(POINT_VAR, point.to_string())
                    .env(PATH_VAR, &path)
                    .env(HISTORY_VAR, if history { "1" } else { "0" }),
            )
            .unwrap();
            let code = output.status.code();
            assert!(
                matches!(code, Some(0 | COMPLETED)),
                "history {history}, point {point}: the child failed:\n{}",
                String::from_utf8_lossy(&output.stderr)
            );
            completed |= code == Some(COMPLETED);
            let wrote = String::from_utf8_lossy(&output.stdout).contains(WROTE_DURING);
            crashed_after_the_commit |= wrote && code == Some(0);
            let expected = if wrote { &during } else { &before };
            assert!(
                numbers(&path) == *expected,
                "history {history}, crash point {point}: the numbers differ"
            );
        }
        assert!(completed, "the last points let the checkpoint finish");
        assert!(
            crashed_after_the_commit,
            "some crashes come after the commit in the middle"
        );
    }

    #[test]
    fn a_crash_at_every_point_of_an_incremental_checkpoint_keeps_every_commit() {
        crash_sweep(false);
    }

    #[test]
    fn a_crash_at_every_point_of_an_incremental_checkpoint_with_history_keeps_every_commit() {
        crash_sweep(true);
    }
}
