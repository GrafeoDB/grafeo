//! Change timestamps rise across reopens, WAL or not (#679, design behaviour 7): the checkpoint
//! stores the change log's last timestamp with the CDC setting (catalog kind 14), or with the
//! setting off when the open captured changes without a stored setting (`Config::with_cdc()`),
//! and an open starts the clock past it and past what the WAL it replays reserved. Each test runs its first
//! open with the clock an hour ahead of the wall clock, as if the clock was set back since.
//!
//! ```bash
//! cargo test -p grafeo-engine --all-features --test main cdc_clock_floor::
//! ```

#![cfg(all(
    feature = "cdc",
    feature = "gql",
    feature = "wal",
    feature = "grafeo-file"
))]

use std::path::{Path, PathBuf};
use std::sync::Arc;

use grafeo_common::types::{HlcTimestamp, Value};
use grafeo_core::graph::GraphBackend;
use grafeo_core::graph::rowgroup::RowGroupStore;
use grafeo_engine::cdc::EntityId;
use grafeo_engine::{Config, GrafeoDB};

const HOUR_MS: u64 = 3_600_000;

fn sidecar(path: &Path) -> PathBuf {
    let mut name = path.as_os_str().to_owned();
    name.push(".wal");
    PathBuf::from(name)
}

/// Runs the change log's clock an hour ahead of the wall clock.
fn run_ahead(db: &GrafeoDB) {
    let ahead = HlcTimestamp::now().physical_ms() + HOUR_MS;
    db.cdc_log().clock().update(HlcTimestamp::new(ahead, 0));
}

/// The timestamp of a new node's create event.
fn next_timestamp(db: &GrafeoDB) -> HlcTimestamp {
    let id = db
        .create_node_with_props(&["Person"], [("name", Value::from("Butch"))])
        .unwrap();
    db.cdc_log().history(EntityId::Node(id))[0].timestamp
}

/// Ahead of the wall clock by more than half an hour: the clock went on past the stored one.
fn is_ahead(timestamp: HlcTimestamp) -> bool {
    timestamp.physical_ms() > HlcTimestamp::now().physical_ms() + HOUR_MS / 2
}

/// A checkpoint and a clean close leave no WAL to replay: the clock goes on past the timestamp
/// the checkpoint stored with the setting.
#[test]
fn the_clock_after_a_reopen_without_a_wal_starts_past_the_stored_one() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("people.grafeo");
    {
        let db = GrafeoDB::open(&path).unwrap();
        db.set_cdc(true).unwrap();
        run_ahead(&db);
        let before = next_timestamp(&db);
        assert!(is_ahead(before));
        // Off: the cursor goes, so the close removes the WAL.
        db.set_cdc(false).unwrap();
        db.close().unwrap();
    }
    assert!(!sidecar(&path).exists(), "no WAL is left to replay");
    let db = GrafeoDB::open(&path).unwrap();
    db.set_cdc(true).unwrap();
    assert!(
        is_ahead(next_timestamp(&db)),
        "the clock went back below the stored timestamp"
    );
    db.close().unwrap();
}

/// An open that captured changes with `Config::with_cdc()` alone, and stored no setting, keeps
/// its timestamps rising too: its checkpoint stores the clock with the setting off, and the next
/// open with `with_cdc()` starts past it, with no WAL left to replay.
#[test]
fn the_clock_after_a_reopen_of_a_with_cdc_open_starts_past_its_timestamps() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("people.grafeo");
    let open = || GrafeoDB::with_config(Config::persistent(&path).with_cdc()).unwrap();
    {
        let db = open();
        run_ahead(&db);
        assert!(is_ahead(next_timestamp(&db)));
        db.close().unwrap();
    }
    assert!(!sidecar(&path).exists(), "no WAL is left to replay");
    let db = open();
    assert!(
        is_ahead(next_timestamp(&db)),
        "the clock went back below the timestamps the last open handed out"
    );
    db.close().unwrap();
}

/// A file whose setting is stored off, by `set_cdc(false)` or by the checkpoint of an open that
/// captured with `Config::with_cdc()` alone, captures again when opened with `with_cdc()`: the
/// open takes the config's setting or the stored one. Opened without it, it captures nothing.
#[test]
fn a_setting_stored_off_lets_with_cdc_turn_capture_on() {
    let dir = tempfile::tempdir().unwrap();
    for (name, stored_off) in [("set off", true), ("captured", false)] {
        let path = dir
            .path()
            .join(format!("{}.grafeo", name.replace(' ', "_")));
        {
            let db = if stored_off {
                let db = GrafeoDB::open(&path).unwrap();
                db.set_cdc(true).unwrap();
                next_timestamp(&db);
                db.set_cdc(false).unwrap();
                db
            } else {
                let db = GrafeoDB::with_config(Config::persistent(&path).with_cdc()).unwrap();
                next_timestamp(&db);
                db
            };
            db.close().unwrap();
        }
        let on = GrafeoDB::with_config(Config::persistent(&path).with_cdc()).unwrap();
        assert!(on.is_cdc_enabled(), "{name}: with_cdc() turns capture on");
        let id = on.create_node(&["Person"]).unwrap();
        assert_eq!(
            on.cdc_log().history(EntityId::Node(id)).len(),
            1,
            "{name}: the create is captured"
        );
        on.close().unwrap();
        drop(on);
        let off = GrafeoDB::open(&path).unwrap();
        assert!(!off.is_cdc_enabled(), "{name}: stored off, no with_cdc()");
        off.close().unwrap();
    }
}

/// A database on a backend of its own replays its WAL at the open: the clock goes on past what
/// the replayed groups reserved.
#[test]
fn the_clock_after_a_backend_wal_replay_starts_past_it() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("backend.grafeo");
    let open = || {
        GrafeoDB::with_store(
            Arc::new(RowGroupStore::new()) as Arc<dyn GraphBackend>,
            Config::persistent(&path).with_cdc(),
        )
        .unwrap()
    };
    {
        let db = open();
        run_ahead(&db);
        assert!(is_ahead(next_timestamp(&db)));
        db.close().unwrap();
    }
    assert!(
        sidecar(&path).exists(),
        "a backend that keeps nothing keeps its log"
    );
    let db = open();
    assert!(
        is_ahead(next_timestamp(&db)),
        "the clock went back below the replayed timestamps"
    );
    db.close().unwrap();
}

/// A replay applies consecutive creates in one chunk (#811) and still counts each of them: after a
/// crash, the clock goes on past the last timestamp the group reserved, not just past its first.
/// Both replay paths: the store's and a backend's of its own.
mod crash {
    use super::*;
    use grafeo_common::testing::child_process;

    const PATH_VAR: &str = "GRAFEO_CDC_CLOCK_CRASH_PATH";
    const LAST_VAR: &str = "GRAFEO_CDC_CLOCK_CRASH_LAST";
    const BACKEND_VAR: &str = "GRAFEO_CDC_CLOCK_CRASH_BACKEND";

    fn open(path: &Path, backend: bool) -> GrafeoDB {
        let config = Config::persistent(path).with_cdc();
        if backend {
            GrafeoDB::with_store(
                Arc::new(RowGroupStore::new()) as Arc<dyn GraphBackend>,
                config,
            )
        } else {
            GrafeoDB::with_config(config)
        }
        .unwrap()
    }

    /// A child commits three creates in one statement with its clock run ahead, writes the last
    /// of their timestamps to a file and exits without closing; the reopened database's next
    /// timestamp is past it.
    fn the_clock_after_a_crash_starts_past_the_last_batched_create(backend: bool) {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("people.grafeo");
        let last_file = dir.path().join("last");
        let mut command = grafeo_common::child_test!("batched_creates_child");
        command.env(PATH_VAR, &path).env(LAST_VAR, &last_file);
        if backend {
            command.env(BACKEND_VAR, "1");
        }
        let status = child_process::run(&mut command).unwrap();
        assert!(status.success(), "the child failed");
        assert!(sidecar(&path).exists(), "the crash left the WAL");
        let last = HlcTimestamp::from_u64(
            std::fs::read_to_string(&last_file)
                .unwrap()
                .trim()
                .parse()
                .unwrap(),
        );
        let db = open(&path, backend);
        let next = next_timestamp(&db);
        assert!(next > last, "{next:?} after the batch's last {last:?}");
        db.close().unwrap();
    }

    #[test]
    fn the_store_replay_counts_each_batched_create() {
        the_clock_after_a_crash_starts_past_the_last_batched_create(false);
    }

    #[test]
    fn a_backend_replay_counts_each_batched_create() {
        the_clock_after_a_crash_starts_past_the_last_batched_create(true);
    }

    /// Child-process entry for the tests above; a no-op when run directly.
    #[test]
    fn batched_creates_child() {
        let Some(path) = std::env::var_os(PATH_VAR) else {
            return;
        };
        let db = open(Path::new(&path), std::env::var_os(BACKEND_VAR).is_some());
        run_ahead(&db);
        db.execute(
            "INSERT (:Person {name: 'Vincent'}), (:Person {name: 'Jules'}), (:Person {name: 'Mia'})",
        )
        .unwrap();
        let epoch = db.current_epoch();
        let events = db.changes_between(epoch, epoch).unwrap();
        assert_eq!(events.len(), 3, "one group of three creates");
        let last = events.iter().map(|event| event.timestamp).max().unwrap();
        std::fs::write(
            std::env::var_os(LAST_VAR).unwrap(),
            last.as_u64().to_string(),
        )
        .unwrap();
        // Exit without closing: no checkpoint, the WAL stays for the replay.
        std::process::exit(0);
    }
}
