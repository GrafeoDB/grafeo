//! The CDC setting is stored with the database (#679): `set_cdc(enabled)` writes catalog kind 14
//! and checkpoints, so `is_cdc_enabled()` after a close and a reopen returns what it was. It is
//! fallible: a refused checkpoint leaves the setting as it was, and a read-only open cannot change
//! it. An in-memory database toggles as before, with nothing stored. `Config::with_cdc()` at the
//! open of a file whose stored setting is off applies to that open only. `set_cdc_enabled`, the
//! old infallible setter, is deprecated naming `set_cdc` and still turns capture on and off.
//!
//! ```bash
//! cargo test -p grafeo-engine --all-features --test main cdc_setting::
//! ```

#![cfg(all(feature = "cdc", feature = "gql", feature = "grafeo-file"))]

use std::path::Path;

use grafeo_common::types::EpochId;
use grafeo_engine::{Config, GrafeoDB};

fn file(dir: &tempfile::TempDir) -> std::path::PathBuf {
    dir.path().join("amsterdam.grafeo")
}

fn reopened(path: &Path) -> GrafeoDB {
    GrafeoDB::open(path).unwrap()
}

fn events(db: &GrafeoDB) -> usize {
    db.changes_between(EpochId::new(0), db.current_epoch())
        .unwrap()
        .len()
}

#[test]
fn the_setting_survives_a_reopen_both_ways() {
    let dir = tempfile::tempdir().unwrap();
    let path = file(&dir);
    let db = GrafeoDB::open(&path).unwrap();
    assert!(!db.is_cdc_enabled(), "off by default");
    db.set_cdc(true).unwrap();
    db.close().unwrap();
    drop(db);

    let db = reopened(&path);
    assert!(db.is_cdc_enabled(), "on after the reopen");
    db.execute("INSERT (:Person {name: 'Alix'})").unwrap();
    assert_eq!(events(&db), 1, "capture runs after the reopen");
    db.set_cdc(false).unwrap();
    db.close().unwrap();
    drop(db);

    let db = reopened(&path);
    assert!(!db.is_cdc_enabled(), "off after the reopen");
    db.close().unwrap();
}

#[test]
fn a_database_never_set_stays_off_across_a_reopen() {
    let dir = tempfile::tempdir().unwrap();
    let path = file(&dir);
    let db = GrafeoDB::open(&path).unwrap();
    db.execute("INSERT (:Person {name: 'Gus'})").unwrap();
    db.close().unwrap();
    drop(db);
    assert!(!reopened(&path).is_cdc_enabled());
}

#[test]
fn setting_it_twice_is_harmless() {
    let dir = tempfile::tempdir().unwrap();
    let path = file(&dir);
    let db = GrafeoDB::open(&path).unwrap();
    db.set_cdc(true).unwrap();
    db.set_cdc(true).unwrap();
    db.close().unwrap();
    drop(db);
    assert!(reopened(&path).is_cdc_enabled());
}

#[test]
fn an_in_memory_database_toggles_and_stores_nothing() {
    let db = GrafeoDB::new_in_memory();
    db.set_cdc(true).unwrap();
    assert!(db.is_cdc_enabled());
    db.execute("INSERT (:Person {name: 'Mia'})").unwrap();
    assert_eq!(events(&db), 1);
    db.set_cdc(false).unwrap();
    assert!(!db.is_cdc_enabled());
}

#[test]
fn a_read_only_open_cannot_change_it() {
    let dir = tempfile::tempdir().unwrap();
    let path = file(&dir);
    let db = GrafeoDB::open(&path).unwrap();
    db.execute("INSERT (:Person {name: 'Vincent'})").unwrap();
    db.close().unwrap();
    drop(db);

    let reader = GrafeoDB::open_read_only(&path).unwrap();
    assert!(
        reader.set_cdc(true).is_err(),
        "a read-only open refuses the change"
    );
    assert!(!reader.is_cdc_enabled());
    drop(reader);
    assert!(!reopened(&path).is_cdc_enabled(), "nothing was stored");
}

/// `with_cdc()` over a stored "off" turns capture on for that open only.
#[test]
fn with_cdc_at_the_open_applies_to_that_open_only() {
    let dir = tempfile::tempdir().unwrap();
    let path = file(&dir);
    let db = GrafeoDB::open(&path).unwrap();
    db.close().unwrap();
    drop(db);

    let db = GrafeoDB::with_config(Config::persistent(&path).with_cdc()).unwrap();
    assert!(db.is_cdc_enabled(), "on for this open");
    db.execute("INSERT (:Person {name: 'Jules'})").unwrap();
    assert_eq!(events(&db), 1);
    db.close().unwrap();
    drop(db);
    assert!(
        !reopened(&path).is_cdc_enabled(),
        "the stored setting stays off"
    );
}

/// The old setter still turns capture on and off.
#[test]
#[allow(
    deprecated,
    reason = "the deprecated setter must keep working until 0.7.0"
)]
fn the_deprecated_setter_still_works() {
    let db = GrafeoDB::new_in_memory();
    db.set_cdc_enabled(true);
    assert!(db.is_cdc_enabled());
    db.set_cdc_enabled(false);
    assert!(!db.is_cdc_enabled());
}

/// A checkpoint without room refuses the change: the setting and the stored record stay as they
/// were, now and after a reopen.
#[cfg(feature = "testing-crash-injection")]
#[test]
fn a_refused_checkpoint_leaves_the_setting_as_it_was() {
    use grafeo_storage::file::space::with_free_space;

    let dir = tempfile::tempdir().unwrap();
    let path = file(&dir);
    let db = GrafeoDB::open(&path).unwrap();
    db.execute("UNWIND range(1, 2000) AS i INSERT (:Person {i: i})")
        .unwrap();
    let refused = with_free_space(Some(4_321), || db.set_cdc(true));
    assert!(refused.is_err(), "no room for the checkpoint");
    assert!(!db.is_cdc_enabled(), "the setting goes back");
    db.close().unwrap();
    drop(db);
    assert!(
        !reopened(&path).is_cdc_enabled(),
        "and so does the stored record"
    );
}

/// A database over a store without a file (`with_read_store`) is read-only: the change is
/// refused and capture stays off.
#[test]
fn a_read_store_without_a_file_cannot_change_it() {
    use grafeo_core::graph::GraphStore;
    use grafeo_core::graph::rowgroup::RowGroupStore;

    let db = GrafeoDB::with_read_store(
        std::sync::Arc::new(RowGroupStore::new()) as std::sync::Arc<dyn GraphStore>,
        Config::in_memory(),
    )
    .unwrap();
    assert!(db.set_cdc(true).is_err(), "the read-only store refuses it");
    assert!(!db.is_cdc_enabled(), "capture stays off");
}

fn on_backend(config: Config) -> GrafeoDB {
    use grafeo_core::graph::GraphBackend;
    use grafeo_core::graph::rowgroup::RowGroupStore;

    GrafeoDB::with_store(
        std::sync::Arc::new(RowGroupStore::new()) as std::sync::Arc<dyn GraphBackend>,
        config,
    )
    .unwrap()
}

/// A configuration conflict: GRAFEO-V001, with `needle` in its message.
fn assert_config_conflict(result: grafeo_common::utils::error::Result<()>, needle: &str) {
    let error = result.expect_err("a configuration conflict");
    assert_eq!(error.error_code().as_str(), "GRAFEO-V001", "{error}");
    assert!(error.to_string().contains(needle), "{needle} in: {error}");
}

/// A database on a backend of its own has no file to store the setting in: `set_cdc` is refused
/// both ways as a configuration conflict (`ConfigError::CdcWithoutFile`, GRAFEO-V001) naming
/// `Config::with_cdc()`, and the setting in effect stays as it was.
#[test]
fn a_backend_without_a_file_refuses_the_change() {
    for (config, before) in [
        (Config::in_memory(), false),
        (Config::in_memory().with_cdc(), true),
    ] {
        let db = on_backend(config);
        assert_eq!(db.is_cdc_enabled(), before);
        assert_config_conflict(db.set_cdc(!before), "Config::with_cdc()");
        assert_eq!(db.is_cdc_enabled(), before, "the setting is unchanged");
    }
}

/// The remedy a refusal of a stored change feed without a WAL names.
const WITHOUT_WAL: &str = "open it with the WAL";

/// `set_cdc(true)` on a database without a WAL is refused (`ConfigError::CdcWithoutWal`,
/// GRAFEO-V001): nothing would log the feed. The setting stays off, now and after a reopen.
#[test]
fn turning_cdc_on_without_a_wal_is_refused() {
    let dir = tempfile::tempdir().unwrap();
    let path = file(&dir);
    GrafeoDB::open(&path).unwrap().close().unwrap();
    let db = GrafeoDB::with_config(Config::persistent(&path).without_wal()).unwrap();
    assert_config_conflict(db.set_cdc(true), WITHOUT_WAL);
    assert!(!db.is_cdc_enabled());
    db.set_cdc(false).unwrap();
    db.close().unwrap();
    drop(db);
    assert!(!reopened(&path).is_cdc_enabled());
}

/// The bytes of every file under `dir`, by path.
fn snapshot(dir: &Path) -> std::collections::BTreeMap<std::path::PathBuf, Vec<u8>> {
    let mut files = std::collections::BTreeMap::new();
    let mut pending = vec![dir.to_path_buf()];
    while let Some(next) = pending.pop() {
        for entry in std::fs::read_dir(&next).unwrap() {
            let path = entry.unwrap().path();
            if path.is_dir() {
                pending.push(path);
            } else {
                files.insert(path.clone(), std::fs::read(&path).unwrap());
            }
        }
    }
    files
}

/// A writable open without a WAL while CDC is stored on is refused before it writes anything
/// (`ConfigError::CdcWithoutWal`, GRAFEO-V001): the database files are as they were. A
/// read-only open without a WAL writes nothing, so it is allowed.
#[test]
fn a_writable_open_without_a_wal_is_refused_while_cdc_is_stored_on() {
    let dir = tempfile::tempdir().unwrap();
    let path = file(&dir);
    let db = GrafeoDB::open(&path).unwrap();
    db.set_cdc(true).unwrap();
    db.execute("INSERT (:Person {name: 'Butch'})").unwrap();
    db.close().unwrap();
    drop(db);
    let before = snapshot(dir.path());

    let refused = GrafeoDB::with_config(Config::persistent(&path).without_wal()).map(drop);
    assert_config_conflict(refused, WITHOUT_WAL);
    assert_eq!(snapshot(dir.path()), before, "the open wrote nothing");

    let reader = GrafeoDB::open_read_only(&path).unwrap();
    assert!(reader.is_cdc_enabled(), "a read-only open is allowed");
    drop(reader);
    assert_eq!(snapshot(dir.path()), before);
}

/// `Config::with_cdc()` is how a database on a backend of its own turns capture on.
#[test]
fn with_cdc_over_a_backend_turns_capture_on() {
    let db = on_backend(Config::in_memory().with_cdc());
    assert!(db.is_cdc_enabled());
    db.execute("INSERT (:Person {name: 'Gus'})").unwrap();
    assert_eq!(events(&db), 1);
}

// ---------------------------------------------------------------------------
// A crash after the change: the setting was stored by its own checkpoint
// ---------------------------------------------------------------------------

#[cfg(feature = "wal")]
mod crash {
    use super::*;
    use grafeo_common::testing::child_process;

    const PATH_VAR: &str = "GRAFEO_CDC_SETTING_CRASH_PATH";
    const ENABLED_VAR: &str = "GRAFEO_CDC_SETTING_CRASH_ENABLED";

    fn sidecar(path: &Path) -> std::path::PathBuf {
        std::path::PathBuf::from(format!("{}.wal", path.display()))
    }

    /// Runs `set_cdc(enabled)` in a child process that exits without closing the database, then
    /// checks the crash left the WAL in place (no close ran) and returns the reopened setting.
    fn after_a_crash(path: &Path, enabled: bool) -> bool {
        let output = child_process::output(
            grafeo_common::child_test!("set_cdc_then_crash_child")
                .env(PATH_VAR, path)
                .env(ENABLED_VAR, enabled.to_string()),
        )
        .unwrap();
        let stdout = String::from_utf8_lossy(&output.stdout);
        assert!(stdout.contains("set_cdc: Ok(())"), "the child: {stdout}");
        assert!(
            sidecar(path).exists(),
            "a crash leaves the WAL, which a close would remove or trim"
        );
        let db = reopened(path);
        let on = db.is_cdc_enabled();
        db.close().unwrap();
        on
    }

    #[test]
    fn turning_it_on_survives_a_crash_without_close() {
        let dir = tempfile::tempdir().unwrap();
        let path = file(&dir);
        let db = GrafeoDB::open(&path).unwrap();
        db.execute("INSERT (:Person {name: 'Alix'})").unwrap();
        db.close().unwrap();
        drop(db);
        assert!(after_a_crash(&path, true), "on after the crash");
    }

    #[test]
    fn turning_it_off_survives_a_crash_without_close() {
        let dir = tempfile::tempdir().unwrap();
        let path = file(&dir);
        let db = GrafeoDB::open(&path).unwrap();
        db.set_cdc(true).unwrap();
        db.execute("INSERT (:Person {name: 'Vincent'})").unwrap();
        db.close().unwrap();
        drop(db);
        assert!(!after_a_crash(&path, false), "off after the crash");
    }

    /// Child-process entry for the tests above; a no-op when run directly. It exits without
    /// dropping the database, so nothing closes it.
    #[test]
    fn set_cdc_then_crash_child() {
        let Ok(path) = std::env::var(PATH_VAR) else {
            return;
        };
        let enabled: bool = std::env::var(ENABLED_VAR).unwrap().parse().unwrap();
        let db = GrafeoDB::open(&path).unwrap();
        let result = db.set_cdc(enabled);
        println!("set_cdc: {result:?}");
        std::mem::forget(db);
        std::process::exit(0);
    }
}
