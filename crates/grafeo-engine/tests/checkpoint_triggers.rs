//! Automatic checkpoints by interval and by WAL size, set at open or on an
//! open database, and their status (#426).
//!
//! ```bash
//! cargo test -p grafeo-engine --all-features --test checkpoint_triggers
//! ```

#![cfg(all(
    feature = "lpg",
    feature = "gql",
    feature = "wal",
    feature = "grafeo-file",
    not(miri)
))]

use std::path::Path;
use std::time::{Duration, Instant};

use grafeo_engine::admin::CheckpointTriggers;
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

fn open(path: &Path, config: impl FnOnce(Config) -> Config) -> GrafeoDB {
    GrafeoDB::with_config(config(Config::persistent(path))).unwrap()
}

/// Without triggers nothing checkpoints by itself: the WAL keeps growing.
#[test]
fn without_triggers_the_wal_is_checkpointed_only_when_asked() {
    let dir = tempfile::tempdir().unwrap();
    let db = open(&dir.path().join("amsterdam.grafeo"), |config| config);
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
    let limit = 16 * 1024;
    let db = open(&dir.path().join("berlin.grafeo"), |config| {
        config.with_checkpoint_wal_size(limit)
    });
    assert_eq!(db.checkpoint_status().triggers.wal_bytes, Some(limit));

    insert(&db, 3);
    std::thread::sleep(Duration::from_millis(400));
    assert!(wal_bytes(&db) > 0, "a small WAL stays");

    while wal_bytes(&db) < usize::try_from(limit).unwrap() {
        insert(&db, 50);
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

    // The file does not store them: the next open has none.
    let db = open(&path, |config| config);
    assert_eq!(db.checkpoint_status().triggers, CheckpointTriggers::none());
}

/// The setter refuses what never checkpoints, and a zero trigger.
#[test]
fn the_setter_refuses_what_never_checkpoints() {
    let triggers = CheckpointTriggers::none().with_wal_bytes(1024);
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
