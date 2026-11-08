//! Automatic checkpoints by interval and by WAL size, set at open or on an
//! open database, and their status (#426).
//!
//! ```bash
//! cargo test -p grafeo-engine --all-features --test main checkpoint_triggers::
//! ```

#![cfg(all(feature = "gql", feature = "wal", feature = "grafeo-file", not(miri)))]

use std::collections::HashMap;
use std::path::Path;
use std::time::{Duration, Instant};

use grafeo_common::types::Value;
use grafeo_engine::admin::CheckpointTriggers;
use grafeo_engine::config::DEFAULT_CHECKPOINT_WAL_SIZE;
use grafeo_engine::{Config, GrafeoDB};

/// The bytes of WAL a reopen would replay.
fn wal_bytes(db: &GrafeoDB) -> usize {
    db.wal_status().size_bytes
}

/// Waits up to ten seconds (a loaded machine runs the checkpointer late).
fn wait_until(what: &str, condition: impl Fn() -> bool) {
    let deadline = Instant::now() + Duration::from_secs(10);
    while !condition() {
        assert!(Instant::now() < deadline, "never: {what}");
        std::thread::sleep(Duration::from_millis(20));
    }
}

/// Inserts `count` people, so the WAL grows.
fn insert(db: &GrafeoDB, count: usize) {
    for i in 0..count {
        db.execute(&format!(
            "INSERT (:Person {{name: 'Vincent {i}', city: 'Amsterdam'}})"
        ))
        .unwrap();
    }
}

/// Commits one node holding 64 KiB of text, so the WAL grows fast.
fn blob(db: &GrafeoDB) {
    db.execute_with_params(
        "INSERT (:Blob {b: $b})",
        HashMap::from([("b".to_string(), Value::from("x".repeat(64 * 1024).as_str()))]),
    )
    .unwrap();
}

fn open(path: &Path, config: impl FnOnce(Config) -> Config) -> GrafeoDB {
    GrafeoDB::with_config(config(Config::persistent(path))).unwrap()
}

/// With both triggers off (`without_checkpoint_wal_size()` and no interval) nothing checkpoints
/// by itself: the WAL keeps growing.
#[test]
fn without_triggers_the_wal_is_checkpointed_only_when_asked() {
    let dir = tempfile::tempdir().unwrap();
    let db = open(&dir.path().join("amsterdam.grafeo"), |config| {
        config.without_checkpoint_wal_size()
    });
    assert_eq!(db.checkpoint_status().triggers, CheckpointTriggers::none());
    insert(&db, 3);
    let before = wal_bytes(&db);
    assert!(before > 0);
    std::thread::sleep(Duration::from_millis(500));
    assert_eq!(wal_bytes(&db), before);
    db.wal_checkpoint().unwrap();
    assert_eq!(wal_bytes(&db), 0);
}

/// A WAL past the size set at open is checkpointed by itself.
#[test]
fn a_wal_size_set_at_open_checkpoints_once_the_wal_passes_it() {
    let dir = tempfile::tempdir().unwrap();
    let limit = 1024 * 1024;
    let db = open(&dir.path().join("berlin.grafeo"), |config| {
        config.with_checkpoint_wal_size(limit)
    });
    assert_eq!(db.checkpoint_status().triggers.wal_bytes, Some(limit));

    insert(&db, 3);
    std::thread::sleep(Duration::from_millis(400));
    assert!(wal_bytes(&db) > 0, "a small WAL stays");

    while wal_bytes(&db) < usize::try_from(limit).unwrap() {
        blob(&db);
    }
    wait_until("the checkpoint", || {
        wal_bytes(&db) < usize::try_from(limit).unwrap()
    });
    let status = db.checkpoint_status();
    assert_eq!((status.failures, status.stopped), (0, false));
    assert!(status.last_checkpoint.is_some());
    db.close().unwrap();
}

/// An interval set at open checkpoints each time it passes.
#[test]
fn an_interval_set_at_open_checkpoints_when_it_passes() {
    let dir = tempfile::tempdir().unwrap();
    let db = open(&dir.path().join("paris.grafeo"), |config| {
        config.with_checkpoint_interval(Duration::from_millis(300))
    });
    insert(&db, 3);
    wait_until("the first checkpoint", || wal_bytes(&db) == 0);
    insert(&db, 3);
    wait_until("the second checkpoint", || wal_bytes(&db) == 0);
}

/// The triggers change on an open database, and turn off again.
#[test]
fn the_triggers_change_on_an_open_database() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("prague.grafeo");
    let db = open(&path, |config| config);
    insert(&db, 3);

    let triggers = CheckpointTriggers::none().with_interval(Duration::from_millis(300));
    db.set_checkpoint_triggers(triggers).unwrap();
    assert_eq!(db.checkpoint_status().triggers, triggers);
    wait_until("the checkpoint", || wal_bytes(&db) == 0);

    db.set_checkpoint_triggers(CheckpointTriggers::none())
        .unwrap();
    insert(&db, 3);
    let before = wal_bytes(&db);
    std::thread::sleep(Duration::from_millis(800));
    assert_eq!(wal_bytes(&db), before, "off again");
    db.close().unwrap();
    drop(db);

    // The file does not store them: the next open has the default.
    let db = open(&path, |config| config);
    assert_eq!(
        db.checkpoint_status().triggers,
        CheckpointTriggers::none().with_wal_bytes(DEFAULT_CHECKPOINT_WAL_SIZE)
    );
}

/// The setter refuses what never checkpoints, and a zero trigger.
#[test]
fn the_setter_refuses_what_never_checkpoints() {
    // At the 1 MiB floor, so each refusal below is for its own reason.
    let triggers = CheckpointTriggers::none().with_wal_bytes(1024 * 1024);
    assert!(
        GrafeoDB::new_in_memory()
            .set_checkpoint_triggers(triggers)
            .is_err(),
        "in memory"
    );

    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("barcelona.grafeo");
    let db = open(&path, |config| config);
    for zero in [
        CheckpointTriggers::none().with_interval(Duration::ZERO),
        CheckpointTriggers::none().with_wal_bytes(0),
    ] {
        assert!(db.set_checkpoint_triggers(zero).is_err(), "{zero:?}");
    }
    db.close().unwrap();
    assert!(db.set_checkpoint_triggers(triggers).is_err(), "closed");
    drop(db);

    let reader = GrafeoDB::with_config(Config::read_only(&path)).unwrap();
    assert!(
        reader.set_checkpoint_triggers(triggers).is_err(),
        "read-only"
    );
    // A read-only open runs no checkpointer, whatever its config says.
    drop(reader);
    let reader = GrafeoDB::with_config(
        Config::read_only(&path).with_checkpoint_interval(Duration::from_millis(100)),
    )
    .unwrap();
    std::thread::sleep(Duration::from_millis(400));
    assert_eq!(reader.checkpoint_status().failures, 0);
}

const DIRECT_PATH: &str = "GRAFEO_CHECKPOINT_TRIGGERS_DIRECT_PATH";
const DIRECT_CASE: &str = "GRAFEO_CHECKPOINT_TRIGGERS_DIRECT_CASE";

/// The walless child's exit code when no interval checkpoint came after the writes.
const NO_INTERVAL_CHECKPOINT: i32 = 4;

/// Writes through `case`'s API on a database without a WAL whose interval is short, waits well
/// past the interval, and ends with no close, as a crash does.
#[test]
fn walless_interval_child() {
    let (Some(path), Some(case)) = (std::env::var_os(DIRECT_PATH), std::env::var_os(DIRECT_CASE))
    else {
        return;
    };
    let db = GrafeoDB::with_config(
        Config::persistent(Path::new(&path))
            .without_wal()
            .with_checkpoint_interval(Duration::from_millis(200)),
    )
    .unwrap();
    let now = || {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs()
    };
    match case.to_str().unwrap() {
        "direct" => {
            let alix = db.create_node(&["Person"]).unwrap();
            let gus = db.create_node(&["Person"]).unwrap();
            db.set_node_property(alix, "name", Value::from("Alix"))
                .unwrap();
            db.set_node_property(gus, "name", Value::from("Gus"))
                .unwrap();
            db.create_edge(alix, gus, "KNOWS").unwrap();
        }
        "statement" => {
            db.execute("INSERT (:Person {name: 'Alix'})-[:KNOWS]->(:Person {name: 'Gus'})")
                .unwrap();
        }
        other => panic!("no case {other}"),
    }
    // Checkpoints are stamped in whole seconds: one stamped after the second in which the writes
    // ended came after all of them. A marker write each round keeps the database changed, so the
    // interval keeps checkpointing until one is that late; then exits without closing: a crash.
    let written = now();
    let deadline = std::time::Instant::now() + Duration::from_secs(30);
    while db
        .checkpoint_status()
        .last_checkpoint
        .is_none_or(|stamp| stamp <= written)
    {
        if std::time::Instant::now() > deadline {
            std::process::exit(NO_INTERVAL_CHECKPOINT);
        }
        db.create_node(&["Marker"]).unwrap();
        std::thread::sleep(Duration::from_millis(50));
    }
    std::process::exit(0);
}

/// Without a WAL the interval checkpoint is what keeps the writes: direct API writes mark the
/// database changed as statements do, so the next interval checkpoints them and a crash after
/// it keeps them.
#[test]
fn an_interval_checkpoints_direct_writes_without_a_wal() {
    for case in ["statement", "direct"] {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("walless.grafeo");
        let mut child = grafeo_common::child_test!("walless_interval_child");
        child.env(DIRECT_PATH, &path).env(DIRECT_CASE, case);
        let status = grafeo_common::testing::child_process::run(&mut child).unwrap();
        match status.code() {
            Some(0) => {}
            Some(NO_INTERVAL_CHECKPOINT) => {
                panic!("{case}: no interval checkpoint within 30 s of the writes")
            }
            _ => panic!("{case}: the child failed: {status}"),
        }
        let db = GrafeoDB::with_config(Config::persistent(&path).without_wal()).unwrap();
        let knows = db
            .execute("MATCH (a:Person {name: 'Alix'})-[:KNOWS]->(g:Person {name: 'Gus'}) RETURN g")
            .unwrap()
            .rows()
            .len();
        assert_eq!(knows, 1, "{case}: the writes survived the crash");
        db.close().unwrap();
    }
}
