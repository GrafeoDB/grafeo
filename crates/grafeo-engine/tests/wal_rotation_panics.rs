//! A segment name in the WAL directory is only ever a published segment (0.6.2, #725). A rotation
//! makes the new segment under `<name>.tmp`, syncs its header and renames it into place, so a
//! panic while a checkpoint rotates the WAL leaves at most a temp file: the next checkpoint on the
//! same handle succeeds, writes go on, and every commit is there on the next open. A writable open
//! removes stale temp files and a read-only one leaves them. A 0.6.1 leftover (a segment with no
//! complete frame whose name lies strictly inside the previous segment's frame range) is a proven
//! unused rotation: the open reads past it, and a writable open removes it. Before, the half-made
//! segment blocked every later checkpoint until a reopen, and the reopen refused the log once
//! groups had landed past the leftover's name.
//!
//! The database is built outside the closure that panics, which only borrows it: a database the
//! closure owned would be dropped while unwinding, and its `Drop` closes it, so the test would see
//! a clean close and not the state the panic leaves.
//!
//! ```bash
//! cargo test -p grafeo-engine --all-features --test main wal_rotation_panics::
//! ```

#![cfg(all(feature = "grafeo-file", feature = "testing-crash-injection"))]

use std::panic::{AssertUnwindSafe, catch_unwind};
use std::path::{Path, PathBuf};

use grafeo_common::testing::crash::{disable_crash, enable_crash_at};
use grafeo_common::types::Value;
use grafeo_engine::{Config, GrafeoDB};
use grafeo_storage::wal::{
    SEGMENT_HEADER_BYTES, is_segment_temp_file_name, list_wal_directory, segment_file_name,
    segment_temp_file_name,
};

/// The crash points of a checkpoint's WAL rotation, in their order.
const ROTATION: [&str; 4] = [
    "wal:after_segment_create",
    "wal:after_segment_header",
    "wal:after_segment_rename",
    "wal:before_dir_sync",
];

/// Runs `f` with a panic armed at the `count`th crash point: the point it fired at, if any.
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

/// A file database with Amsterdam and Berlin committed.
fn database(path: &Path) -> GrafeoDB {
    let db = GrafeoDB::with_config(Config::persistent(path)).unwrap();
    insert(&db, "Amsterdam");
    insert(&db, "Berlin");
    db
}

fn insert(db: &GrafeoDB, city: &str) {
    db.session()
        .execute(&format!("INSERT (:City {{name: '{city}'}})"))
        .unwrap();
}

fn cities(db: &GrafeoDB) -> Vec<String> {
    let result = db
        .session()
        .execute("MATCH (c:City) RETURN c.name")
        .unwrap();
    let mut names: Vec<String> = result
        .rows()
        .iter()
        .filter_map(|row| match &row[0] {
            Value::String(name) => Some(name.to_string()),
            _ => None,
        })
        .collect();
    names.sort();
    names
}

fn sidecar(path: &Path) -> PathBuf {
    PathBuf::from(format!("{}.wal", path.display()))
}

/// The temp files in the sidecar WAL directory.
fn temps(path: &Path) -> Vec<String> {
    let Ok(entries) = std::fs::read_dir(sidecar(path)) else {
        return Vec::new();
    };
    let mut found: Vec<String> = entries
        .map(|entry| entry.unwrap().file_name().to_string_lossy().into_owned())
        .filter(|name| is_segment_temp_file_name(name))
        .collect();
    found.sort();
    found
}

const PATH_VAR: &str = "GRAFEO_ROTATION_PATH";
const MODE_VAR: &str = "GRAFEO_ROTATION_MODE";

/// The child: builds a database at the path it is given and ends as a crash does (no close, no
/// destructors), so its files are what a crash leaves; an open database's file cannot be read by
/// another handle on Windows, so the parent cannot copy it. `panic:<count>` panics at that crash
/// point of a checkpoint and commits Lisbon after it; `frames` commits three cities after a
/// checkpoint, so the newest segment holds their frames.
#[test]
fn rotation_child() {
    let Some(path) = std::env::var_os(PATH_VAR) else {
        return;
    };
    let mode = std::env::var(MODE_VAR).unwrap();
    let path = PathBuf::from(path);
    // The database lives in this scope until the exit: dropped, it would close (a checkpoint,
    // then the WAL removed), and the parent would see a clean close, not a crash.
    let db = if let Some(count) = mode.strip_prefix("panic:") {
        let db = database(&path);
        let fired = panic_at(count.parse().unwrap(), || {
            let _ = db.wal_checkpoint();
        });
        assert!(fired.is_some(), "count {count} reached no crash point");
        insert(&db, "Lisbon");
        db
    } else {
        let db = GrafeoDB::open(&path).unwrap();
        db.wal_checkpoint().unwrap();
        for city in ["Amsterdam", "Berlin", "Lisbon"] {
            insert(&db, city);
        }
        db
    };
    assert!(db.wal().is_some(), "the database still holds its WAL");
    assert!(sidecar(&path).exists(), "the crash leaves the WAL");
    std::process::exit(0);
}

/// Runs the child in `mode` at `path` and waits for it.
fn crash_child(path: &Path, mode: &str) {
    let mut child = grafeo_common::child_test!("rotation_child");
    child.env(PATH_VAR, path).env(MODE_VAR, mode);
    let status = grafeo_common::testing::child_process::run(&mut child).unwrap();
    assert!(status.success(), "the child failed in {mode}");
}

/// The crash points a checkpoint of a fresh database reaches, in order, with their counts.
fn checkpoint_points() -> Vec<(u64, String)> {
    let mut points = Vec::new();
    for count in 1..=64 {
        let dir = tempfile::tempdir().unwrap();
        let db = database(&dir.path().join("probe.grafeo"));
        match panic_at(count, || {
            let _ = db.wal_checkpoint();
        }) {
            Some(message) => points.push((count, message)),
            None => return points,
        }
    }
    panic!("a checkpoint ran past 64 crash points: {points:?}")
}

/// The checkpoint reaches the rotation's points in the ruled order.
#[test]
fn a_checkpoint_rotates_through_the_ruled_points_in_order() {
    let fired: Vec<String> = checkpoint_points()
        .into_iter()
        .map(|(_, message)| message)
        .collect();
    let positions: Vec<usize> = ROTATION
        .iter()
        .map(|point| {
            fired
                .iter()
                .position(|message| message.ends_with(point))
                .unwrap_or_else(|| panic!("never reached {point}: {fired:?}"))
        })
        .collect();
    assert!(positions.is_sorted(), "{fired:?}");
}

/// A panic at each crash point of a checkpoint, with no reopen: the next checkpoint on the same
/// handle succeeds, a write after it and another checkpoint too, and a close and a reopen find
/// every commit with no temp file left. With and without a commit between the panic and the next
/// checkpoint (with one, groups land past where the new segment was to start).
#[test]
fn a_panic_in_a_checkpoint_leaves_the_next_one_free_without_a_reopen() {
    for (count, point) in checkpoint_points() {
        for commit_between in [false, true] {
            let context = format!("{point}, a commit between: {commit_between}");
            let dir = tempfile::tempdir().unwrap();
            let path = dir.path().join("rotation.grafeo");
            let db = database(&path);
            let fired = panic_at(count, || {
                let _ = db.wal_checkpoint();
            });
            assert_eq!(fired.as_deref(), Some(point.as_str()), "{context}");
            let mut want = vec!["Amsterdam", "Berlin"];
            if commit_between {
                insert(&db, "Lisbon");
                want.push("Lisbon");
            }
            db.wal_checkpoint().unwrap_or_else(|error| {
                panic!("{context}: the checkpoint after the panic: {error}")
            });
            insert(&db, "Porto");
            want.push("Porto");
            want.sort_unstable();
            db.wal_checkpoint()
                .unwrap_or_else(|error| panic!("{context}: a later checkpoint: {error}"));
            assert_eq!(cities(&db), want, "{context}: on the open handle");
            db.close()
                .unwrap_or_else(|error| panic!("{context}: close: {error}"));
            let db =
                GrafeoDB::open(&path).unwrap_or_else(|error| panic!("{context}: reopen: {error}"));
            assert_eq!(cities(&db), want, "{context}: after a reopen");
            assert!(temps(&path).is_empty(), "{context}: {:?}", temps(&path));
            db.close().unwrap();
        }
    }
}

/// The same with a reopen right after the panic, with a commit between: the files a process
/// leaves when it ends right then (a child that panics, commits Lisbon and exits without a close)
/// hold no segment name without its whole header, open with no salvage and every commit, and the
/// reopened database checkpoints and takes writes. The class over every crash point of a
/// checkpoint, the rotation's included.
#[test]
fn a_panic_in_a_checkpoint_leaves_files_that_reopen_whole() {
    for (count, point) in checkpoint_points() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("rotation.grafeo");
        crash_child(&path, &format!("panic:{count}"));
        // The guarantee that makes refusing an emptied segment safe: no crash of a 0.6.2 writer
        // leaves a segment name without its whole header, so this open needs no salvage.
        for (lsn, segment) in list_wal_directory(&sidecar(&path)).unwrap().segments {
            let length = std::fs::metadata(&segment).unwrap().len();
            assert!(
                length >= SEGMENT_HEADER_BYTES as u64,
                "{point}: the segment at {lsn} holds {length} bytes, less than its header"
            );
        }
        let db = GrafeoDB::open(&path).unwrap_or_else(|error| panic!("{point}: reopen: {error}"));
        assert_eq!(cities(&db), ["Amsterdam", "Berlin", "Lisbon"], "{point}");
        assert!(
            temps(&path).is_empty(),
            "{point}: the writable open removed the temp"
        );
        insert(&db, "Porto");
        db.wal_checkpoint()
            .unwrap_or_else(|error| panic!("{point}: a checkpoint after the reopen: {error}"));
        db.close().unwrap();
    }
}

/// A database's files as a crash leaves them, with Amsterdam, Berlin and Lisbon committed after
/// the last checkpoint, so the newest WAL segment holds their frames: (the path, the newest
/// segment's first LSN).
fn crashed_with_frames(into: &Path) -> (PathBuf, u64) {
    let copied = into.join("leftover.grafeo");
    // Made and closed first, as the WAL class test does: a database the child only opens writes
    // its commits to the sidecar WAL, which the leftover goes into.
    GrafeoDB::open(&copied).unwrap().close().unwrap();
    crash_child(&copied, "frames");
    let newest = list_wal_directory(&sidecar(&copied))
        .unwrap()
        .segments
        .last()
        .map(|(lsn, _)| *lsn)
        .unwrap();
    (copied, newest)
}

/// A stale temp file: a read-only open ignores it and leaves it, a writable open removes it; both
/// read every commit.
#[test]
fn a_stale_temp_is_left_by_a_read_only_open_and_removed_by_a_writable_one() {
    let into = tempfile::tempdir().unwrap();
    let (path, newest) = crashed_with_frames(into.path());
    let temp = sidecar(&path).join(segment_temp_file_name(newest + 1));
    std::fs::write(&temp, b"never published").unwrap();
    let reader = GrafeoDB::open_read_only(&path).unwrap();
    assert_eq!(cities(&reader), ["Amsterdam", "Berlin", "Lisbon"]);
    drop(reader);
    assert!(temp.exists(), "a read-only open leaves the temp");
    let db = GrafeoDB::open(&path).unwrap();
    assert_eq!(cities(&db), ["Amsterdam", "Berlin", "Lisbon"]);
    assert!(!temp.exists(), "a writable open removes the temp");
    db.close().unwrap();
}

/// A 0.6.1 leftover: a blank file whose name lies strictly inside the newest segment's frames. A
/// read-only open reads every commit past it, and a writable open does and removes it.
#[test]
fn a_leftover_inside_a_segments_frames_opens_and_is_removed() {
    let into = tempfile::tempdir().unwrap();
    let (path, newest) = crashed_with_frames(into.path());
    let leftover = sidecar(&path).join(segment_file_name(newest + 1));
    std::fs::write(&leftover, b"").unwrap();
    let reader =
        GrafeoDB::open_read_only(&path).unwrap_or_else(|error| panic!("a read-only open: {error}"));
    assert_eq!(cities(&reader), ["Amsterdam", "Berlin", "Lisbon"]);
    drop(reader);
    let db = GrafeoDB::open(&path).unwrap_or_else(|error| panic!("a writable open: {error}"));
    assert_eq!(cities(&db), ["Amsterdam", "Berlin", "Lisbon"]);
    assert!(!leftover.exists(), "a writable open removes the leftover");
    db.close().unwrap();
}
