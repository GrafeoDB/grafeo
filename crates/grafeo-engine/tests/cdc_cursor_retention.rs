//! A CDC consumer cursor keeps the WAL it has not read (#679). `set_cdc(true)` writes
//! `cdc_cursor_default.json` in the WAL directory at its checkpoint's position, `set_cdc(false)`
//! removes it. Every checkpoint keeps the WAL from the lowest cursor on, backup and CDC cursors
//! alike, under the one limit `Config::with_wal_retention_limit`: within it a checkpoint and a
//! close keep the cursor's segments, past it the oldest go. An unreadable cursor is held to the
//! limit. A cursor's name is `[a-z0-9_-]`, 1 to 64 characters, and the WAL directory's scan knows
//! the file. A cursor left while CDC is stored off keeps no log, refuses no open, and goes at the
//! next open or `set_cdc(false)`; its failed removal is GRAFEO-S007. After a crash anywhere in a
//! toggle, the reopen's cursor follows the stored setting, losing nothing; a failed release of
//! covered log after the image is active does not undo a toggle. The feed's truncation error is
//! GRAFEO-S006.
//!
//! ```bash
//! cargo test -p grafeo-engine --all-features --test main cdc_cursor_retention::
//! ```

#![cfg(all(feature = "cdc", feature = "gql", feature = "wal"))]

use std::path::{Path, PathBuf};
use std::time::Duration;

use grafeo_common::utils::error::{Error, StorageError};
use grafeo_engine::config::DurabilityMode;
use grafeo_engine::{Config, GrafeoDB};
use grafeo_storage::wal::{
    CdcCursorName, ScanOptions, WalScan, is_cdc_cursor_file, list_wal_directory,
};

/// A small bound, so a few checkpoints pass it.
const LIMIT: u64 = 256 * 1024;
const CURSOR_FILE: &str = "cdc_cursor_default.json";

fn sidecar(path: &Path) -> PathBuf {
    PathBuf::from(format!("{}.wal", path.display()))
}

/// Batch durability, set here so a change of the default cannot move it: it runs no background
/// flusher, whose sync marker could land between the cursor a toggle writes and its checkpoint
/// (harmless for the feed, but the tests compare the two positions).
fn database(path: &Path) -> GrafeoDB {
    GrafeoDB::with_config(
        Config::persistent(path)
            .with_wal_retention_limit(LIMIT)
            .with_wal_durability(DurabilityMode::batch(Duration::from_millis(100), 1000)),
    )
    .unwrap()
}

/// About 32 KiB of WAL: a hundred people, each with a 300-byte note.
fn write_batch(db: &GrafeoDB, from: i64) {
    db.execute(&format!(
        "UNWIND range({from}, {}) AS i INSERT (:Person {{i: i, note: '{}'}})",
        from + 99,
        "x".repeat(300)
    ))
    .unwrap();
}

fn batches_with_checkpoints(db: &GrafeoDB, rounds: i64) {
    for round in 0..rounds {
        write_batch(db, round * 100);
        db.wal_checkpoint().unwrap();
    }
}

/// The bytes of every segment but the newest: what checkpoints kept behind.
fn kept_behind(path: &Path) -> u64 {
    let segments = list_wal_directory(&sidecar(path)).unwrap().segments;
    segments[..segments.len().saturating_sub(1)]
        .iter()
        .map(|(_, segment)| std::fs::metadata(segment).unwrap().len())
        .sum()
}

fn cursor(path: &Path) -> PathBuf {
    sidecar(path).join(CURSOR_FILE)
}

/// The log position the default cursor holds.
fn cursor_lsn(path: &Path) -> u64 {
    let text = std::fs::read_to_string(cursor(path)).unwrap();
    let json: serde_json::Value = serde_json::from_str(&text).unwrap();
    json["lsn"]
        .as_u64()
        .unwrap_or_else(|| panic!("no lsn: {text}"))
}

/// The first log position the WAL directory still holds.
fn first_kept(path: &Path) -> u64 {
    list_wal_directory(&sidecar(path)).unwrap().segments[0].0
}

// ---------------------------------------------------------------------------
// The name
// ---------------------------------------------------------------------------

#[test]
fn a_cursor_name_is_lowercase_letters_digits_dash_and_underscore() {
    for good in ["default", "a", "a-b_1", &"z".repeat(64)] {
        let name = CdcCursorName::new(good).unwrap_or_else(|error| panic!("{good}: {error}"));
        assert_eq!(name.file_name(), format!("cdc_cursor_{good}.json"));
        assert!(is_cdc_cursor_file(&name.file_name()), "{good}");
    }
    for bad in ["", &"z".repeat(65), "../x", "A", "a.b", "a b", "a/b", "é"] {
        assert!(CdcCursorName::new(bad).is_err(), "{bad:?} is refused");
    }
    assert_eq!(CdcCursorName::DEFAULT, "default");
    for not_a_cursor in [
        "cdc_cursor_.json",
        "cdc_cursor_A.json",
        "cdc_cursor_default.json.tmp",
        "cdc_cursor_default",
        "backup_cursor.json",
    ] {
        assert!(!is_cdc_cursor_file(not_a_cursor), "{not_a_cursor}");
    }
}

// ---------------------------------------------------------------------------
// The file and the setting
// ---------------------------------------------------------------------------

#[test]
fn turning_cdc_on_writes_the_cursor_and_off_removes_it() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("db.grafeo");
    let db = database(&path);
    assert!(!cursor(&path).exists());
    db.set_cdc(true).unwrap();
    assert!(cursor(&path).exists(), "on writes the cursor");
    let text = std::fs::read_to_string(cursor(&path)).unwrap();
    let json: serde_json::Value = serde_json::from_str(&text).unwrap();
    assert!(json["lsn"].is_u64() && json["epoch"].is_u64(), "{text}");
    let listing = list_wal_directory(&sidecar(&path)).unwrap();
    assert!(
        listing.others.iter().any(|p| p.ends_with(CURSOR_FILE)),
        "the cursor is a non-segment entry: {:?}",
        listing.others
    );
    let fm = db.file_manager().unwrap();
    let scan = WalScan::open(
        sidecar(&path),
        ScanOptions::new(fm.database_id(), fm.active_header().checkpoint_lsn),
    )
    .unwrap()
    .finish()
    .unwrap();
    assert!(
        !scan.unknown_files.iter().any(|p| p.ends_with(CURSOR_FILE)),
        "the WAL directory's scan knows the cursor file: {:?}",
        scan.unknown_files
    );
    db.set_cdc(false).unwrap();
    assert!(!cursor(&path).exists(), "off removes it");
    db.close().unwrap();
}

// ---------------------------------------------------------------------------
// Retention
// ---------------------------------------------------------------------------

/// Without a cursor a checkpoint releases the log behind it; within the limit a cursor keeps it,
/// through a close too.
#[test]
fn within_the_limit_the_cursor_keeps_its_segments_through_a_close() {
    let dir = tempfile::tempdir().unwrap();
    let without = dir.path().join("without.grafeo");
    let db = database(&without);
    batches_with_checkpoints(&db, 4);
    assert_eq!(kept_behind(&without), 0, "nothing kept without a cursor");
    db.close().unwrap();

    let with = dir.path().join("with.grafeo");
    let db = database(&with);
    db.set_cdc(true).unwrap();
    batches_with_checkpoints(&db, 4);
    let kept = kept_behind(&with);
    assert!(
        kept > 0 && kept <= LIMIT,
        "{kept} bytes kept for the cursor"
    );
    db.close().unwrap();
    drop(db);
    assert!(
        sidecar(&with).exists(),
        "the close keeps the WAL for the cursor"
    );
    assert!(cursor(&with).exists());
    assert!(kept_behind(&with) >= kept, "the close released none of it");
}

#[test]
fn past_the_limit_the_oldest_segments_go() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("db.grafeo");
    let db = database(&path);
    db.set_cdc(true).unwrap();
    batches_with_checkpoints(&db, 30);
    let kept = kept_behind(&path);
    assert!(kept > 0, "the cursor still keeps some");
    assert!(kept <= LIMIT, "{kept} bytes kept, limit {LIMIT}");
    assert!(
        cursor_lsn(&path) < first_kept(&path),
        "the cursor at {} lies below the first segment kept, {}",
        cursor_lsn(&path),
        first_kept(&path)
    );
    db.close().unwrap();
}

/// A backup cursor and the CDC cursor together keep from the lower one.
#[test]
fn a_backup_cursor_and_the_cdc_cursor_keep_from_the_lower() {
    let dir = tempfile::tempdir().unwrap();
    let both = dir.path().join("both.grafeo");
    let db = database(&both);
    db.set_cdc(true).unwrap();
    batches_with_checkpoints(&db, 2);
    db.backup_full(&dir.path().join("backups")).unwrap();
    batches_with_checkpoints(&db, 2);
    let kept_both = kept_behind(&both);
    db.close().unwrap();

    let backup_only = dir.path().join("backup.grafeo");
    let db = database(&backup_only);
    batches_with_checkpoints(&db, 2);
    db.backup_full(&dir.path().join("backups2")).unwrap();
    batches_with_checkpoints(&db, 2);
    let kept_backup = kept_behind(&backup_only);
    db.close().unwrap();
    assert!(
        kept_both > kept_backup,
        "the older CDC cursor keeps more: {kept_both} against {kept_backup}"
    );
}

#[test]
fn an_unreadable_cdc_cursor_is_held_to_the_limit() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("db.grafeo");
    let db = database(&path);
    db.set_cdc(true).unwrap();
    std::fs::write(cursor(&path), [b'{', 3, 19]).unwrap();
    batches_with_checkpoints(&db, 30);
    let kept = kept_behind(&path);
    assert!(
        kept > 0 && kept <= LIMIT,
        "{kept} bytes kept for an unreadable cursor, limit {LIMIT}"
    );
    db.close().unwrap();
}

/// Turning CDC on while it is on leaves the cursor where it is: the changes after it stay
/// unread, and a checkpoint keeps their segments.
#[test]
fn turning_cdc_on_again_leaves_the_cursor_alone() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("db.grafeo");
    let db = database(&path);
    db.set_cdc(true).unwrap();
    let at = cursor_lsn(&path);
    batches_with_checkpoints(&db, 2);
    db.set_cdc(true).unwrap();
    assert_eq!(cursor_lsn(&path), at, "the cursor did not move");
    write_batch(&db, 10_000);
    db.wal_checkpoint().unwrap();
    assert!(
        first_kept(&path) <= at,
        "the segments from the cursor at {at} are kept, the first is {}",
        first_kept(&path)
    );
    db.close().unwrap();
}

/// Turning CDC on while it is on writes the cursor when it is missing (removed by hand, or lost
/// to a failed step): the repair needs no reopen. It starts at that call's checkpoint.
#[test]
fn turning_cdc_on_again_writes_a_missing_cursor() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("db.grafeo");
    let db = database(&path);
    db.set_cdc(true).unwrap();
    batches_with_checkpoints(&db, 2);
    std::fs::remove_file(cursor(&path)).unwrap();
    db.set_cdc(true).unwrap();
    assert!(cursor(&path).exists(), "the cursor is back");
    assert_eq!(
        cursor_lsn(&path),
        db.file_manager().unwrap().active_header().checkpoint_lsn,
        "it starts at the checkpoint of the call that wrote it"
    );
    db.close().unwrap();
}

/// A checkpoint without room refuses `set_cdc(true)`: the cursor it wrote first goes with the
/// setting, so no cursor keeps the log for a feed that is off.
#[cfg(feature = "testing-crash-injection")]
#[test]
fn a_refused_checkpoint_leaves_no_cursor() {
    use grafeo_storage::file::space::with_free_space;

    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("db.grafeo");
    let db = database(&path);
    write_batch(&db, 0);
    assert!(with_free_space(Some(4_321), || db.set_cdc(true)).is_err());
    assert!(!db.is_cdc_enabled());
    assert!(!cursor(&path).exists(), "no cursor for a change not stored");
    db.close().unwrap();
}

fn without_wal(path: &Path) -> grafeo_common::utils::error::Result<GrafeoDB> {
    GrafeoDB::with_config(
        Config::persistent(path)
            .with_wal_retention_limit(LIMIT)
            .without_wal(),
    )
}

/// The sequence that left a gap in the feed (writes through an open without a WAL, then an open
/// with one) stops at its first step: the open without a WAL is refused while CDC is stored on,
/// so the next open with a WAL has the cursor where it was and the log from it, and every change
/// after the cursor is in that log.
#[test]
fn writes_without_a_wal_cannot_leave_a_gap_in_the_feed() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("db.grafeo");
    let db = database(&path);
    db.set_cdc(true).unwrap();
    let at = cursor_lsn(&path);
    batches_with_checkpoints(&db, 2);
    write_batch(&db, 10_000);
    db.close().unwrap();
    drop(db);
    assert!(first_kept(&path) <= at, "the close kept the unread log");

    let refused = without_wal(&path).map(drop).unwrap_err();
    assert_eq!(refused.error_code().as_str(), "GRAFEO-V001", "{refused}");

    let db = database(&path);
    assert_eq!(cursor_lsn(&path), at, "the cursor is where it was");
    assert!(
        first_kept(&path) <= at,
        "the log from the cursor at {at} is kept, the first is {}",
        first_kept(&path)
    );
    assert_eq!(
        db.execute("MATCH (p:Person) RETURN p")
            .unwrap()
            .rows()
            .len(),
        300,
        "every change after the cursor came through an open with the WAL"
    );
    db.close().unwrap();
}

/// Leaves the default cursor in the sidecar of `path`, as a removal that failed does.
fn leave_a_cursor(path: &Path) {
    std::fs::create_dir_all(sidecar(path)).unwrap();
    std::fs::write(cursor(path), br#"{"lsn": 0, "epoch": 0}"#).unwrap();
}

/// A cursor left behind while CDC is stored off refuses no open: an open without a WAL opens
/// (and retires the sidecar, the cursor with it), and an open with the WAL removes it.
#[test]
fn a_cursor_left_while_cdc_is_off_refuses_no_open() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("db.grafeo");
    let db = database(&path);
    write_batch(&db, 0);
    db.close().unwrap();
    drop(db);

    leave_a_cursor(&path);
    let db = without_wal(&path).expect("a cursor of a setting stored off refuses nothing");
    assert!(!db.is_cdc_enabled());
    db.close().unwrap();
    drop(db);
    assert!(!cursor(&path).exists(), "the open without a WAL retired it");

    leave_a_cursor(&path);
    let db = database(&path);
    assert!(!cursor(&path).exists(), "the open with the WAL removed it");
    db.close().unwrap();
}

/// Turning CDC on after a failed removal left the cursor of the earlier feed writes a new one
/// at that call's checkpoint: the old cursor's position belongs to a feed that was off since.
#[cfg(feature = "testing-crash-injection")]
#[test]
fn turning_cdc_on_after_a_failed_removal_starts_at_the_new_checkpoint() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("db.grafeo");
    let (db, _) = turned_off_with_the_removal_failing(&path);
    let old = cursor_lsn(&path);
    write_batch(&db, 10_000);
    db.set_cdc(true).unwrap();
    let checkpoint = db.file_manager().unwrap().active_header().checkpoint_lsn;
    assert!(
        checkpoint > old,
        "the new checkpoint is past the old cursor"
    );
    assert_eq!(
        cursor_lsn(&path),
        checkpoint,
        "the cursor starts at the checkpoint of the call that turned CDC on"
    );
    db.close().unwrap();
}

/// A cursor write that fails leaves the setting as it was: off now and after a reopen, with no
/// cursor (the injected failure is the cursor file's sync).
#[cfg(feature = "testing-crash-injection")]
#[test]
fn a_failed_cursor_write_leaves_the_setting_as_it_was() {
    use grafeo_common::testing::crash::with_failure_at;

    const POINT: &str = "injected failure at: cdc:cursor_sync";
    let mut landed = false;
    for count in 1..=64 {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("db.grafeo");
        let db = database(&path);
        write_batch(&db, 0);
        let result = with_failure_at(count, || db.set_cdc(true));
        match result {
            Ok(()) => break,
            Err(error) if error.to_string().contains(POINT) => {
                landed = true;
                assert!(
                    matches!(error, Error::Io(_)),
                    "a failed write is an I/O error: {error:?}"
                );
                assert!(
                    !sidecar(&path).join(format!("{CURSOR_FILE}.tmp")).exists(),
                    "the half-written cursor is removed"
                );
                assert!(!db.is_cdc_enabled(), "the setting goes back");
                db.close().unwrap();
                drop(db);
                let db = GrafeoDB::open(&path).unwrap();
                assert!(!db.is_cdc_enabled(), "and so does the stored record");
                assert!(
                    !cursor(&path).exists(),
                    "no cursor for a setting that is off"
                );
                db.close().unwrap();
                break;
            }
            Err(_) => {}
        }
    }
    assert!(landed, "no failure landed at {POINT}");
}

/// A database with CDC on and a cursor holding a few checkpoints of log, turned off with the
/// cursor's removal failing (the injected failure is the removal itself): the setting is stored
/// off, the cursor is still there, and the error is returned.
#[cfg(feature = "testing-crash-injection")]
fn turned_off_with_the_removal_failing(path: &Path) -> (GrafeoDB, Error) {
    use grafeo_common::testing::crash::with_failure_at;

    const POINT: &str = "injected failure at: cdc:cursor_remove";
    for count in 1..=64 {
        let db = database(path);
        db.set_cdc(true).unwrap();
        batches_with_checkpoints(&db, 2);
        match with_failure_at(count, || db.set_cdc(false)) {
            Err(error) if error.to_string().contains(POINT) => return (db, error),
            // A failure elsewhere: before the image nothing is stored, after it (a segment
            // removal) the call still succeeds.
            Err(_) | Ok(()) => {}
        }
        // A failure that left the WAL or the checkpoints refusing makes the close say so.
        drop(db.close());
        drop(db);
        std::fs::remove_file(path).unwrap();
        if sidecar(path).exists() {
            std::fs::remove_dir_all(sidecar(path)).unwrap();
        }
    }
    panic!("no failure landed at {POINT}")
}

/// A cursor left by a failed removal holds no WAL: with CDC stored off, the next checkpoint
/// releases the log, as with CDC off.
#[cfg(feature = "testing-crash-injection")]
#[test]
fn a_cursor_left_while_cdc_is_stored_off_holds_no_wal() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("db.grafeo");
    let (db, _) = turned_off_with_the_removal_failing(&path);
    assert!(!db.is_cdc_enabled(), "the setting is stored off");
    assert!(cursor(&path).exists(), "the removal failed");
    write_batch(&db, 10_000);
    db.wal_checkpoint().unwrap();
    assert_eq!(
        kept_behind(&path),
        0,
        "the cursor of a setting stored off keeps nothing"
    );
    db.close().unwrap();
}

/// The failed removal is GRAFEO-S007: the setting is stored (off), only its cursor step failed.
/// The close keeps no WAL for the cursor, and the reopen has neither (after a crash instead, the
/// open removes the cursor: `crash::a_crashed_toggle_off_reopens_with_the_cursor_following_the_setting`).
#[cfg(feature = "testing-crash-injection")]
#[test]
fn a_failed_removal_is_s007_and_the_next_open_completes_it() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("db.grafeo");
    let (db, error) = turned_off_with_the_removal_failing(&path);
    assert!(
        matches!(&error, Error::Storage(storage)
            if matches!(**storage, StorageError::CdcCursorPending(_))),
        "{error:?}"
    );
    assert_eq!(error.error_code().as_str(), "GRAFEO-S007", "{error}");
    assert!(!db.is_cdc_enabled(), "the setting is stored off");
    db.close().unwrap();
    drop(db);
    assert!(
        !sidecar(&path).exists(),
        "the close keeps no WAL for the cursor of a setting stored off"
    );
    let db = GrafeoDB::open(&path).unwrap();
    assert!(!db.is_cdc_enabled(), "off after the reopen");
    assert!(!cursor(&path).exists(), "the open removed the cursor");
    db.close().unwrap();
}

/// Turning CDC off again while it is off retries the removal that failed.
#[cfg(feature = "testing-crash-injection")]
#[test]
fn turning_cdc_off_again_removes_a_cursor_left_behind() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("db.grafeo");
    let (db, _) = turned_off_with_the_removal_failing(&path);
    assert!(cursor(&path).exists());
    db.set_cdc(false).unwrap();
    assert!(!cursor(&path).exists(), "the second call removed it");
    assert!(!db.is_cdc_enabled());
    db.close().unwrap();
}

/// Off removes the cursor, and the next checkpoint releases what it kept.
#[test]
fn off_releases_the_kept_log_at_the_next_checkpoint() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("db.grafeo");
    let db = database(&path);
    db.set_cdc(true).unwrap();
    batches_with_checkpoints(&db, 4);
    assert!(kept_behind(&path) > 0);
    db.set_cdc(false).unwrap();
    write_batch(&db, 10_000);
    db.wal_checkpoint().unwrap();
    assert_eq!(
        kept_behind(&path),
        0,
        "nothing kept once the cursor is gone"
    );
    db.close().unwrap();
}

/// `set_cdc(false)` whose checkpoint fails after it let commits go on leaves capture on for
/// those commits: the setting goes back to on, and a commit made while the image was written
/// has its event.
#[cfg(all(
    feature = "testing-crash-injection",
    feature = "testing-statement-injection"
))]
#[test]
fn a_failed_turn_off_captures_the_commits_made_during_its_checkpoint() {
    use std::sync::Arc;

    use grafeo_common::testing::commit_hook::during_next_checkpoint_write;
    use grafeo_common::testing::crash::{fired_failure, with_failure_at};
    use grafeo_common::types::EpochId;

    for count in 1..=64 {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("db.grafeo");
        let db = Arc::new(database(&path));
        db.set_cdc(true).unwrap();
        let writer = Arc::clone(&db);
        // A failure before the write leaves the hook armed for the close's checkpoint, which
        // refuses the commit: only the landing count needs it.
        during_next_checkpoint_write(move || {
            std::thread::spawn(move || {
                let _ = writer
                    .session()
                    .execute("INSERT (:Person {name: 'Vincent'})");
            })
            .join()
            .unwrap();
        });
        let result = with_failure_at(count, || db.set_cdc(false));
        let fired = fired_failure();
        if fired.as_deref() == Some("checkpoint:after_chunks") {
            assert!(result.is_err(), "the checkpoint failed before its header");
            assert!(db.is_cdc_enabled(), "the setting goes back to on");
            let events = db
                .changes_between(EpochId::new(0), EpochId::new(u64::MAX))
                .unwrap();
            assert_eq!(
                events.len(),
                1,
                "the commit during the failed checkpoint is captured: {events:?}"
            );
            db.close().unwrap();
            return;
        }
        assert!(
            fired.is_some(),
            "no failure landed after the commits went on"
        );
        // A failure that left the WAL refusing makes the close say so.
        drop(db.close());
    }
    panic!("no failure landed after the commits went on")
}

/// An open that finds a cursor beside a setting stored off and cannot remove it opens anyway:
/// the cursor holds no log, and a later open or `set_cdc(false)` removes it.
#[cfg(feature = "testing-crash-injection")]
#[test]
fn an_open_that_cannot_remove_a_leftover_cursor_opens() {
    use grafeo_common::testing::crash::{fired_failure, with_failure_at};

    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("db.grafeo");
    let db = database(&path);
    write_batch(&db, 0);
    db.close().unwrap();
    drop(db);
    for count in 1..=64 {
        leave_a_cursor(&path);
        let opened = with_failure_at(count, || GrafeoDB::open(&path));
        if fired_failure().as_deref() == Some("cdc:cursor_remove") {
            let db = opened.expect("a cursor that cannot be removed fails no open");
            assert!(!db.is_cdc_enabled());
            assert!(cursor(&path).exists(), "the removal failed");
            db.set_cdc(false).unwrap();
            assert!(!cursor(&path).exists(), "set_cdc(false) removes it");
            db.close().unwrap();
            return;
        }
        assert!(
            fired_failure().is_some(),
            "no failure landed on the removal"
        );
        if let Ok(db) = opened {
            db.close().unwrap();
        }
    }
    panic!("no failure landed on the removal")
}

// ---------------------------------------------------------------------------
// Crashes and failures of a toggle
// ---------------------------------------------------------------------------

#[cfg(feature = "testing-crash-injection")]
mod crash {
    use super::*;
    use grafeo_common::testing::child_process;

    const PATH_VAR: &str = "GRAFEO_CDC_TOGGLE_PATH";
    const OP_VAR: &str = "GRAFEO_CDC_TOGGLE_OP";
    const COUNT_VAR: &str = "GRAFEO_CDC_TOGGLE_AT";
    const POINT: &str = "cdc:after_toggle_checkpoint";
    const MIB: usize = 1024 * 1024;

    /// Runs `op` in a child process that exits without closing, with a crash or a failure armed
    /// at `count`; returns what it printed.
    fn child(path: &Path, op: &str, count: u64) -> String {
        let output = child_process::output(
            grafeo_common::child_test!("toggle_child")
                .env(PATH_VAR, path)
                .env(OP_VAR, op)
                .env(COUNT_VAR, count.to_string()),
        )
        .unwrap();
        String::from_utf8_lossy(&output.stdout).into_owned()
    }

    /// The cursor follows the stored setting after a reopen: on, a cursor at or below the
    /// image's checkpoint (so the feed misses nothing); off, no cursor.
    fn assert_the_cursor_follows_the_setting(db: &GrafeoDB, path: &Path, count: u64) {
        if db.is_cdc_enabled() {
            let checkpoint = db.file_manager().unwrap().active_header().checkpoint_lsn;
            assert!(cursor(path).exists(), "count {count}: on without a cursor");
            assert!(
                cursor_lsn(path) <= checkpoint,
                "count {count}: the cursor at {} is past the checkpoint at {checkpoint}",
                cursor_lsn(path)
            );
            assert!(
                first_kept(path) <= cursor_lsn(path),
                "count {count}: the log from the cursor at {} is gone, the first kept is {}",
                cursor_lsn(path),
                first_kept(path)
            );
        } else {
            assert!(!cursor(path).exists(), "count {count}: off with a cursor");
        }
    }

    /// Crashes `op` at every count of crash points in turn and reopens: at each one the cursor
    /// follows the stored setting and no row is lost. `prepare` builds the database the child
    /// opens. Returns whether a crash landed between the toggle's checkpoint and its cursor step.
    fn crash_at_every_point(op: &str, prepare: impl Fn(&Path)) -> bool {
        let mut landed = false;
        for count in 1..=256 {
            let dir = tempfile::tempdir().unwrap();
            let path = dir.path().join("db.grafeo");
            prepare(&path);
            let stdout = child(&path, op, count);
            let completed = stdout.contains("completed");
            let at_point = stdout.contains(POINT);
            landed |= at_point;
            if !completed {
                assert!(
                    sidecar(&path).exists(),
                    "count {count}: the crash left the WAL, as a real one does"
                );
            }
            let db = GrafeoDB::open(&path).unwrap();
            assert_the_cursor_follows_the_setting(&db, &path, count);
            if at_point {
                assert_eq!(
                    db.is_cdc_enabled(),
                    op == "on",
                    "the checkpoint before the point stored the setting"
                );
            }
            let rows = db
                .execute("MATCH (p:Person) RETURN p")
                .unwrap()
                .rows()
                .len();
            assert_eq!(rows, 100, "count {count}: nothing lost");
            db.close().unwrap();
            if completed {
                return landed;
            }
        }
        panic!("the toggle never completed")
    }

    /// Crashes `set_cdc(true)` at each crash point in turn. A crash after the toggle's checkpoint
    /// reopens with the setting on and a cursor at the image's checkpoint.
    #[test]
    fn a_crashed_toggle_on_reopens_with_the_cursor_following_the_setting() {
        let landed = crash_at_every_point("on", |path| {
            let db = database(path);
            write_batch(&db, 0);
            db.close().unwrap();
        });
        assert!(landed, "no crash landed at {POINT}");
    }

    /// Crashes `set_cdc(false)` at each crash point in turn. A crash after the toggle's checkpoint,
    /// before the cursor goes, reopens with the setting off and no cursor.
    #[test]
    fn a_crashed_toggle_off_reopens_with_the_cursor_following_the_setting() {
        let landed = crash_at_every_point("off", |path| {
            let db = database(path);
            db.set_cdc(true).unwrap();
            write_batch(&db, 0);
            db.close().unwrap();
        });
        assert!(landed, "no crash landed at {POINT}");
    }

    /// A reopen writes a cursor removed by hand while CDC is stored on, at the image's checkpoint.
    #[test]
    fn a_reopen_writes_a_cursor_removed_by_hand() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("db.grafeo");
        let db = database(&path);
        db.set_cdc(true).unwrap();
        batches_with_checkpoints(&db, 2);
        db.close().unwrap();
        drop(db);
        std::fs::remove_file(cursor(&path)).unwrap();
        let db = database(&path);
        assert!(cursor(&path).exists(), "the reopen wrote the cursor");
        assert_eq!(
            cursor_lsn(&path),
            db.file_manager().unwrap().active_header().checkpoint_lsn,
            "it starts at the image's checkpoint"
        );
        db.close().unwrap();
    }

    /// Fails `op` at each failure point in turn in a child that exits without closing, until the
    /// failure lands on the removal of a WAL segment the checkpoint covers, after the image is
    /// active. Returns the child's result and the reopened database.
    fn failed_at_the_segment_removal(
        path: &Path,
        op: &str,
        prepare: impl Fn(&Path),
    ) -> (String, GrafeoDB) {
        for count in 1..=256 {
            if path.exists() {
                std::fs::remove_file(path).unwrap();
            }
            if sidecar(path).exists() {
                std::fs::remove_dir_all(sidecar(path)).unwrap();
            }
            prepare(path);
            let stdout = child(path, op, count);
            let result = stdout
                .lines()
                .find_map(|line| line.strip_prefix("result: "))
                .unwrap_or_else(|| panic!("count {count}: {stdout}"))
                .to_string();
            let fired = stdout
                .lines()
                .find_map(|line| line.strip_prefix("fired: "))
                .unwrap_or_else(|| panic!("count {count}: {stdout}"));
            if fired == "wal:remove_segment" {
                return (result, GrafeoDB::open(path).unwrap());
            }
            assert!(
                fired != "none",
                "count {count}: no failure landed on the segment removal"
            );
        }
        panic!("no failure landed on the segment removal")
    }

    /// Fills `path` with a database whose CDC cursor keeps a few checkpoints of log.
    fn on_with_kept_log(path: &Path) {
        let db = database(path);
        db.set_cdc(true).unwrap();
        batches_with_checkpoints(&db, 2);
        db.close().unwrap();
    }

    /// A segment removal that fails after the toggle's image is active does not undo the toggle:
    /// `set_cdc(false)` returns `Ok`, and the reopen after a crash has it off, as it said.
    #[test]
    fn a_failed_segment_removal_after_the_image_agrees_with_set_cdc() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("db.grafeo");
        let (result, db) = failed_at_the_segment_removal(&path, "off_failing", on_with_kept_log);
        assert_eq!(result, "Ok", "the toggle is stored");
        assert!(!db.is_cdc_enabled(), "the reopen agrees with the call");
        assert!(!cursor(&path).exists());
        db.close().unwrap();
    }

    /// The same for a stored configuration: `set_stored_config` returns `Ok`, and the reopen
    /// after a crash has the setting.
    #[test]
    fn a_failed_segment_removal_after_the_image_agrees_with_set_stored_config() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("db.grafeo");
        let (result, db) = failed_at_the_segment_removal(&path, "memory_limit", |path| {
            let db = database(path);
            write_batch(&db, 0);
            db.close().unwrap();
        });
        assert_eq!(result, "Ok", "the setting is stored");
        assert_eq!(
            db.memory_limit(),
            Some(48 * MIB),
            "the reopen agrees with the call"
        );
        db.close().unwrap();
    }

    /// Child-process entry for the tests above; a no-op when run directly. The database is open
    /// before anything is armed, and the panic hook exits the process, so nothing closes it.
    /// `on` and `off` crash at the count; `off_failing` and `memory_limit` write a batch first,
    /// then fail at the count and print the result and the log kept behind.
    #[test]
    fn toggle_child() {
        use grafeo_common::testing::crash::{disable_crash, enable_crash_at, with_failure_at};

        let Ok(path) = std::env::var(PATH_VAR) else {
            return;
        };
        let path = PathBuf::from(path);
        let op = std::env::var(OP_VAR).unwrap();
        let count: u64 = std::env::var(COUNT_VAR).unwrap().parse().unwrap();
        let db = database(&path);
        std::panic::set_hook(Box::new(|info| {
            println!("{info}");
            std::process::exit(3);
        }));
        if op == "on" || op == "off" {
            enable_crash_at(count);
            let result = db.set_cdc(op == "on");
            disable_crash();
            println!("completed: {result:?}");
        } else {
            write_batch(&db, 1_000);
            let result = with_failure_at(count, || {
                if op == "off_failing" {
                    db.set_cdc(false)
                } else {
                    db.set_stored_config(&Config::in_memory().with_memory_limit(48 * MIB))
                }
            });
            match result {
                Ok(()) => println!("result: Ok"),
                Err(error) => println!("result: {error}"),
            }
            println!(
                "fired: {}",
                grafeo_common::testing::crash::fired_failure().unwrap_or_else(|| "none".into())
            );
        }
        std::mem::forget(db);
        std::process::exit(0);
    }
}

// ---------------------------------------------------------------------------
// The feed's truncation error
// ---------------------------------------------------------------------------

#[test]
fn the_truncated_feed_error_is_s006_and_names_the_first_position() {
    let error = Error::from(StorageError::CdcFeedTruncated {
        first_available: 4_242,
    });
    assert_eq!(error.error_code().as_str(), "GRAFEO-S006");
    assert!(error.to_string().contains("4242"), "{error}");
}
