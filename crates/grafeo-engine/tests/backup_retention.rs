//! The WAL a backup cursor keeps is bounded (0.6.2, #673 item 4). A checkpoint keeps the log from
//! the backup cursor on, so the next incremental backup finds it; when incremental backups stop,
//! that log grew until the disk was full. Now it keeps at most `wal_retention_limit` bytes beyond
//! the checkpoint (1 GiB by default), releasing the oldest segments past it, and an unreadable
//! cursor is held to the same bound. The next incremental backup after a cut fails with
//! `GRAFEO-S005` (`BackupChainBroken`), naming the limit, and a full backup starts a new chain.
//! `release_backup_cursor` drops the cursor on purpose: the next backup must be a full one.
//!
//! Every refusal of an incremental backup that a new full backup fixes is S005; without a chain
//! (no manifest, or one without a full backup) it is `InvalidValue`, "run a full backup first".
//! The class test lists each refusal of `do_backup_incremental` and checks that the list matches
//! the function.
//!
//! The names this file assumes, for renaming where the implementation differs:
//! `Config::with_wal_retention_limit(bytes)`, `Config::wal_retention_limit()`,
//! `grafeo_engine::config::DEFAULT_WAL_RETENTION_LIMIT`, `GrafeoDB::release_backup_cursor()`, and
//! the crash point `flush:after_retention_release`.
//!
//! ```bash
//! cargo test -p grafeo-engine --all-features --test main backup_retention::
//! ```

#![cfg(all(feature = "gql", feature = "wal", feature = "grafeo-file"))]

use std::path::{Path, PathBuf};

use grafeo_common::types::{EpochId, Value};
use grafeo_common::utils::error::Error;
use grafeo_engine::config::DEFAULT_WAL_RETENTION_LIMIT;
use grafeo_engine::{Config, GrafeoDB};
use grafeo_storage::wal::{BACKUP_CURSOR_FILE, list_wal_directory};

/// A small bound, so a few checkpoints pass it.
const LIMIT: u64 = 256 * 1024;

fn sidecar(path: &Path) -> PathBuf {
    PathBuf::from(format!("{}.wal", path.display()))
}

fn database(path: &Path, limit: Option<u64>) -> GrafeoDB {
    let config = Config::persistent(path);
    let config = match limit {
        Some(bytes) => config.with_wal_retention_limit(bytes),
        None => config,
    };
    GrafeoDB::with_config(config).unwrap()
}

/// About 32 KiB of WAL: a hundred people, each with a 300-byte note, numbered from `from`.
fn write_batch(db: &GrafeoDB, from: i64) {
    db.session()
        .execute(&format!(
            "UNWIND range({from}, {}) AS i INSERT (:Person {{i: i, note: '{}'}})",
            from + 99,
            "x".repeat(300)
        ))
        .unwrap();
}

/// `rounds` batches, each followed by a checkpoint.
fn batches_with_checkpoints(db: &GrafeoDB, rounds: i64) -> i64 {
    for round in 0..rounds {
        write_batch(db, round * 100);
        db.wal_checkpoint().unwrap();
    }
    rounds * 100
}

fn people(db: &GrafeoDB) -> i64 {
    match db
        .session()
        .execute("MATCH (p:Person) RETURN count(p) AS c")
        .unwrap()
        .rows()[0][0]
    {
        Value::Int64(count) => count,
        ref other => panic!("{other:?}"),
    }
}

/// The WAL's segments, (first LSN, length), in log order.
fn segments(path: &Path) -> Vec<(u64, u64)> {
    list_wal_directory(&sidecar(path))
        .unwrap()
        .segments
        .iter()
        .map(|(lsn, segment)| (*lsn, std::fs::metadata(segment).unwrap().len()))
        .collect()
}

/// The bytes of every segment but the newest, which the checkpoint keeps for the cursor.
fn kept_behind(path: &Path) -> u64 {
    let all = segments(path);
    all[..all.len().saturating_sub(1)]
        .iter()
        .map(|(_, length)| length)
        .sum()
}

fn is_chain_broken(error: &Error) -> bool {
    error.error_code().as_str() == "GRAFEO-S005"
}

#[test]
fn the_default_limit_is_one_gib() {
    assert_eq!(DEFAULT_WAL_RETENTION_LIMIT, 1 << 30);
    let dir = tempfile::tempdir().unwrap();
    let config = Config::persistent(dir.path().join("db.grafeo"));
    assert_eq!(config.wal_retention_limit(), DEFAULT_WAL_RETENTION_LIMIT);
    assert_eq!(
        Config::persistent(dir.path().join("db.grafeo"))
            .with_wal_retention_limit(LIMIT)
            .wal_retention_limit(),
        LIMIT
    );
}

/// Backups stop, checkpoints go on: the WAL kept for the cursor stops at the limit, where the
/// default would keep it all.
#[test]
fn the_wal_kept_for_a_stopped_backup_stops_at_the_limit() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("db.grafeo");
    let db = database(&path, Some(LIMIT));
    db.backup_full(&dir.path().join("backups")).unwrap();
    batches_with_checkpoints(&db, 30);
    let kept = kept_behind(&path);
    assert!(
        kept <= LIMIT,
        "{kept} bytes kept for the cursor, limit {LIMIT}"
    );
    db.close().unwrap();

    let unbounded = tempfile::tempdir().unwrap();
    let path = unbounded.path().join("db.grafeo");
    let db = database(&path, None);
    db.backup_full(&unbounded.path().join("backups")).unwrap();
    batches_with_checkpoints(&db, 30);
    assert!(
        kept_behind(&path) > LIMIT,
        "the test passes the limit: the default keeps {} bytes",
        kept_behind(&path)
    );
    db.close().unwrap();
}

/// After the limit cut the log, the next incremental backup fails with S005 naming the limit; a
/// full backup then starts a new chain, which the next incremental continues and a restore reads
/// whole.
#[test]
fn a_cut_chain_fails_with_s005_and_a_full_backup_starts_a_new_one() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("db.grafeo");
    let backups = dir.path().join("backups");
    let db = database(&path, Some(LIMIT));
    db.backup_full(&backups).unwrap();
    let written = batches_with_checkpoints(&db, 30);
    let error = db.backup_incremental(&backups).unwrap_err();
    assert!(is_chain_broken(&error), "{error}");
    assert!(
        error.to_string().contains(&LIMIT.to_string()),
        "names the limit: {error}"
    );

    let fresh = dir.path().join("fresh-backups");
    db.backup_full(&fresh).unwrap();
    write_batch(&db, written);
    db.backup_incremental(&fresh).unwrap();
    db.close().unwrap();
    let restored = dir.path().join("restored.grafeo");
    GrafeoDB::restore_to_epoch(&fresh, EpochId::new(u64::MAX), &restored).unwrap();
    let db = GrafeoDB::open(&restored).unwrap();
    assert_eq!(people(&db), written + 100);
    db.close().unwrap();
}

/// Within the limit the guarantee holds: an incremental backup after several checkpoints copies
/// every commit since the full one.
#[test]
fn within_the_limit_an_incremental_after_checkpoints_restores_every_commit() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("db.grafeo");
    let backups = dir.path().join("backups");
    let db = database(&path, None);
    db.backup_full(&backups).unwrap();
    let written = batches_with_checkpoints(&db, 10);
    db.backup_incremental(&backups).unwrap();
    db.close().unwrap();
    let restored = dir.path().join("restored.grafeo");
    GrafeoDB::restore_to_epoch(&backups, EpochId::new(u64::MAX), &restored).unwrap();
    let db = GrafeoDB::open(&restored).unwrap();
    assert_eq!(people(&db), written);
    db.close().unwrap();
}

/// Releasing the cursor: the next checkpoint keeps no log for it, the next incremental backup is
/// refused with S005, and a full backup then starts a chain the incrementals continue.
#[test]
fn a_released_cursor_keeps_no_log_and_asks_for_a_full_backup() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("db.grafeo");
    let backups = dir.path().join("backups");
    let db = database(&path, None);
    db.backup_full(&backups).unwrap();
    let written = batches_with_checkpoints(&db, 5);
    assert!(kept_behind(&path) > 0, "the cursor keeps log");
    assert!(db.release_backup_cursor().unwrap(), "a cursor was released");
    assert!(!db.release_backup_cursor().unwrap(), "none is left");
    db.wal_checkpoint().unwrap();
    assert_eq!(
        kept_behind(&path),
        0,
        "no log is kept for a released cursor"
    );
    let error = db.backup_incremental(&backups).unwrap_err();
    assert!(is_chain_broken(&error), "{error}");
    assert!(
        error.to_string().contains("cursor"),
        "names the cause: {error}"
    );
    db.backup_full(&backups).unwrap();
    write_batch(&db, written);
    db.backup_incremental(&backups).unwrap();
    db.close().unwrap();
}

/// An unreadable cursor is held to the limit too: the checkpoint cannot know where it points, and
/// before kept the whole log.
#[test]
fn an_unreadable_cursor_is_held_to_the_limit() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("db.grafeo");
    let db = database(&path, Some(LIMIT));
    db.backup_full(&dir.path().join("backups")).unwrap();
    std::fs::write(sidecar(&path).join(BACKUP_CURSOR_FILE), [b'{', 3, 19]).unwrap();
    batches_with_checkpoints(&db, 30);
    let kept = kept_behind(&path);
    assert!(
        kept <= LIMIT,
        "{kept} bytes kept for an unreadable cursor, limit {LIMIT}"
    );
    db.close().unwrap();
}

// The refusals of an incremental backup, one case each.

/// A refusal of `do_backup_incremental`: its name, the setup that makes it, and its code.
struct Refusal {
    name: &'static str,
    /// Whether a new full backup fixes it: S005; otherwise `InvalidValue`.
    chain_broken: bool,
    /// Words of its own message, so the case is known to meet its own site.
    cause: &'static str,
    /// Builds the database and the backup directory into `dir` so that the next incremental
    /// backup meets this refusal: the database, the backup directory.
    arrange: fn(&Path) -> (GrafeoDB, PathBuf),
}

fn with_full_backup(dir: &Path) -> (GrafeoDB, PathBuf) {
    let path = dir.join("db.grafeo");
    let backups = dir.join("backups");
    let db = database(&path, None);
    write_batch(&db, 0);
    db.backup_full(&backups).unwrap();
    write_batch(&db, 100);
    (db, backups)
}

fn edit_json(file: &Path, edit: impl FnOnce(&mut serde_json::Value)) {
    let mut value: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(file).unwrap()).unwrap();
    edit(&mut value);
    std::fs::write(file, serde_json::to_string(&value).unwrap()).unwrap();
}

const REFUSALS: &[Refusal] = &[
    Refusal {
        name: "no manifest",
        cause: "no backup manifest found",
        chain_broken: false,
        arrange: |dir| {
            let db = database(&dir.join("db.grafeo"), None);
            write_batch(&db, 0);
            (db, dir.join("backups"))
        },
    },
    Refusal {
        name: "a manifest without a full backup",
        cause: "no full backup in manifest",
        chain_broken: false,
        arrange: |dir| {
            let (db, backups) = with_full_backup(dir);
            edit_json(&backups.join("backup_manifest.json"), |manifest| {
                for segment in manifest["segments"].as_array_mut().unwrap() {
                    segment["kind"] = "Incremental".into();
                }
            });
            (db, backups)
        },
    },
    Refusal {
        name: "no backup cursor (released)",
        cause: "has no backup cursor",
        chain_broken: true,
        arrange: |dir| {
            let (db, backups) = with_full_backup(dir);
            assert!(db.release_backup_cursor().unwrap());
            (db, backups)
        },
    },
    Refusal {
        name: "a cursor of the format before 0.6.0",
        cause: "of the WAL format before 0.6.0",
        chain_broken: true,
        arrange: |dir| {
            let (db, backups) = with_full_backup(dir);
            let cursor = sidecar(&dir.join("db.grafeo")).join(BACKUP_CURSOR_FILE);
            edit_json(&cursor, |cursor| {
                cursor.as_object_mut().unwrap().remove("wal_lsn");
            });
            (db, backups)
        },
    },
    Refusal {
        name: "a chain that ends before the cursor",
        cause: "further along the log",
        chain_broken: true,
        arrange: |dir| {
            let (db, backups) = with_full_backup(dir);
            // A full backup elsewhere moves the cursor past this chain's end.
            db.backup_full(&dir.join("elsewhere")).unwrap();
            write_batch(&db, 200);
            (db, backups)
        },
    },
    Refusal {
        name: "the log past the retention limit, cut in this open",
        chain_broken: true,
        cause: "passed the retention limit",
        arrange: |dir| {
            let path = dir.join("db.grafeo");
            let backups = dir.join("backups");
            let db = database(&path, Some(LIMIT));
            db.backup_full(&backups).unwrap();
            batches_with_checkpoints(&db, 30);
            (db, backups)
        },
    },
    Refusal {
        name: "the segment the chain ended at, removed before this open",
        chain_broken: true,
        cause: "the WAL segment at log position",
        arrange: |dir| {
            let path = dir.join("db.grafeo");
            let backups = dir.join("backups");
            let db = database(&path, Some(LIMIT));
            db.backup_full(&backups).unwrap();
            batches_with_checkpoints(&db, 30);
            db.close().unwrap();
            drop(db);
            // A reopen: no record of this open's cut is left, only the segments are gone.
            (database(&path, Some(LIMIT)), backups)
        },
    },
    Refusal {
        name: "a gap in the log (the listing, and the groups read)",
        cause: "the WAL has a gap",
        chain_broken: true,
        arrange: |dir| {
            let path = dir.join("db.grafeo");
            let backups = dir.join("backups");
            let db = database(&path, None);
            db.backup_full(&backups).unwrap();
            batches_with_checkpoints(&db, 4);
            let all = list_wal_directory(&sidecar(&path)).unwrap().segments;
            assert!(all.len() >= 3, "{all:?}");
            std::fs::remove_file(&all[1].1).unwrap();
            (db, backups)
        },
    },
];

/// Each refusal gives its code: S005 where a new full backup fixes it, `InvalidValue` where there
/// is no chain yet, and its own message; and a full backup then lets an incremental one through.
#[test]
fn every_refusal_of_an_incremental_backup_has_its_code() {
    let mut wrong = Vec::new();
    for refusal in REFUSALS {
        let dir = tempfile::tempdir().unwrap();
        let (db, backups) = (refusal.arrange)(dir.path());
        match db.backup_incremental(&backups) {
            Ok(segment) => wrong.push(format!("{}: taken: {segment:?}", refusal.name)),
            Err(error) => {
                let code = if is_chain_broken(&error) {
                    "S005"
                } else if matches!(error, Error::InvalidValue(_)) {
                    "InvalidValue"
                } else {
                    "another error"
                };
                let want = if refusal.chain_broken {
                    "S005"
                } else {
                    "InvalidValue"
                };
                if code != want {
                    wrong.push(format!("{}: {code}, want {want}: {error}", refusal.name));
                } else if !error.to_string().contains(refusal.cause) {
                    wrong.push(format!(
                        "{}: another site's refusal (no `{}`): {error}",
                        refusal.name, refusal.cause
                    ));
                }
            }
        }
        db.backup_full(&backups)
            .unwrap_or_else(|error| panic!("{}: a full backup: {error}", refusal.name));
        write_batch(&db, 10_000);
        if let Err(error) = db.backup_incremental(&backups) {
            wrong.push(format!(
                "{}: an incremental after a full backup: {error}",
                refusal.name
            ));
        }
        db.close().unwrap();
    }
    assert!(wrong.is_empty(), "{}", wrong.join("\n"));
}

/// The list above is the function's: `do_backup_incremental` refuses at as many sites as the
/// list has cases, the gap's two sites sharing one case. A new refusal fails here until it has a
/// case and a code.
#[test]
fn the_refusal_list_matches_do_backup_incremental() {
    // A Windows checkout may hold the file with CRLF line ends.
    let source = include_str!("../src/database/backup.rs").replace("\r\n", "\n");
    let start = source
        .find("fn do_backup_incremental(")
        .expect("do_backup_incremental");
    let body = &source[start..];
    let end = body.find("\n}\n").expect("its end");
    let body = &body[..end];
    let sites =
        body.matches("Error::InvalidValue(").count() + body.matches("chain_broken(").count();
    assert_eq!(
        sites,
        REFUSALS.len() + 1,
        "do_backup_incremental refuses at {sites} sites; the list has {} cases (the gap's two \
         sites share one)",
        REFUSALS.len()
    );
}

#[cfg(feature = "testing-crash-injection")]
mod crash {
    use std::panic::{AssertUnwindSafe, catch_unwind};

    use grafeo_common::testing::crash::{disable_crash, enable_crash_at};

    use super::*;

    const PATH_VAR: &str = "GRAFEO_RETENTION_PATH";
    const COUNT_VAR: &str = "GRAFEO_RETENTION_COUNT";
    const POINT: &str = "flush:after_retention_release";

    /// A database whose next checkpoint releases segments past the limit: a full backup, then
    /// 29 batches with checkpoints, then one more batch.
    fn past_the_limit(path: &Path) -> GrafeoDB {
        let db = database(path, Some(LIMIT));
        db.backup_full(&path.with_extension("backups")).unwrap();
        batches_with_checkpoints(&db, 29);
        write_batch(&db, 2_900);
        db
    }

    fn panic_at(count: u64, f: impl FnOnce()) -> Option<String> {
        enable_crash_at(count);
        let outcome = catch_unwind(AssertUnwindSafe(f));
        disable_crash();
        outcome.err().map(|payload| {
            payload
                .downcast_ref::<String>()
                .cloned()
                .unwrap_or_default()
        })
    }

    /// The count at which that checkpoint reaches the release point.
    fn count_of_the_release() -> u64 {
        for count in 1..=64 {
            let dir = tempfile::tempdir().unwrap();
            let db = past_the_limit(&dir.path().join("probe.grafeo"));
            match panic_at(count, || {
                let _ = db.wal_checkpoint();
            }) {
                Some(message) if message.ends_with(POINT) => return count,
                Some(_) => {}
                None => break,
            }
        }
        panic!("a checkpoint past the limit never reaches {POINT}")
    }

    /// The child: the checkpoint that releases segments panics at the release point, and the
    /// process ends as a crash does, the database still open.
    #[test]
    fn retention_child() {
        let Some(path) = std::env::var_os(PATH_VAR) else {
            return;
        };
        let count = std::env::var(COUNT_VAR).unwrap().parse().unwrap();
        let path = PathBuf::from(path);
        let db = past_the_limit(&path);
        let fired = panic_at(count, || {
            let _ = db.wal_checkpoint();
        });
        assert!(
            fired
                .as_deref()
                .is_some_and(|message| message.ends_with(POINT)),
            "{fired:?}"
        );
        assert!(db.wal().is_some());
        std::process::exit(0);
    }

    /// A crash between the release of segments and its warning leaves a database that opens with
    /// every commit, checkpoints, and takes a full backup; the cut chain is refused with S005.
    #[test]
    fn a_crash_after_the_release_leaves_the_database_whole() {
        let count = count_of_the_release();
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("db.grafeo");
        let mut child = grafeo_common::child_test!("retention_child");
        child.env(PATH_VAR, &path).env(COUNT_VAR, count.to_string());
        let status = grafeo_common::testing::child_process::run(&mut child).unwrap();
        assert!(status.success(), "the child failed");
        let db = GrafeoDB::open(&path).unwrap();
        assert_eq!(people(&db), 3_000);
        db.wal_checkpoint().unwrap();
        let error = db
            .backup_incremental(&path.with_extension("backups"))
            .unwrap_err();
        assert!(is_chain_broken(&error), "{error}");
        db.backup_full(&dir.path().join("fresh")).unwrap();
        db.close().unwrap();
    }
}
