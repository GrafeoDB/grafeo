//! A checkpoint checks the free space before it writes (0.6.2, #726). The space needed is the
//! file's length (the copy-on-write worst case) plus the checkpoint's headroom, one WAL segment
//! (every checkpoint ends with a rotation, so the WAL must still have room for a segment after it).
//! A checkpoint that needs more than the disk has free fails with `InsufficientSpace { path,
//! needed, available }`, an I/O error of kind `StorageFull`, writes nothing, poisons nothing, and
//! leaves the previous checkpoint and the WAL as they were: the database goes on and a later
//! checkpoint with room succeeds. A probe that fails (a filesystem without the call) skips the
//! check. A write that fails after the check passed and before the new header (the disk filled in
//! between) cuts the file back to its length before the checkpoint, so the WAL keeps its room.
//!
//! The free space comes from `with_free_space`, gated on `testing-crash-injection` as `maybe_fail`
//! is: inside its closure the space query on this thread answers the given number, or fails for
//! `None`.
//!
//! ```bash
//! cargo test -p grafeo-engine --all-features --test main checkpoint_free_space::
//! ```

#![cfg(all(feature = "grafeo-file", feature = "testing-crash-injection"))]

use std::path::{Path, PathBuf};

use grafeo_common::testing::crash::with_failure_at;
use grafeo_common::types::Value;
use grafeo_common::utils::error::Error;
use grafeo_engine::{Config, GrafeoDB};
use grafeo_storage::file::InsufficientSpace;
use grafeo_storage::file::space::with_free_space;
use grafeo_storage::wal::DEFAULT_SEGMENT_BYTES;

/// Free bytes no checkpoint of the test database fits in.
const TOO_LITTLE: u64 = 4_321;
/// Room for any checkpoint of these tests.
const PLENTY: u64 = 1 << 40;

/// A file database with 2,000 people, so its checkpoint needs far more than `TOO_LITTLE`.
fn database(path: &Path) -> GrafeoDB {
    let db = GrafeoDB::with_config(Config::persistent(path)).unwrap();
    db.session()
        .execute("UNWIND range(1, 2000) AS i INSERT (:Person {i: i, name: 'Alix ' + toString(i)})")
        .unwrap();
    db
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

fn sidecar_wal(path: &Path) -> PathBuf {
    let mut wal = path.as_os_str().to_owned();
    wal.push(".wal");
    PathBuf::from(wal)
}

/// The names and lengths of the files in `dir`, recursively.
fn listing(dir: &Path) -> Vec<(String, u64)> {
    let mut files = Vec::new();
    let mut pending = vec![dir.to_path_buf()];
    while let Some(next) = pending.pop() {
        for entry in std::fs::read_dir(&next).unwrap() {
            let entry = entry.unwrap();
            let path = entry.path();
            if path.is_dir() {
                pending.push(path);
            } else {
                let name = path.strip_prefix(dir).unwrap().display().to_string();
                // From the file itself: on Windows a listing's size is stale while the file
                // is open.
                files.push((name, std::fs::metadata(&path).unwrap().len()));
            }
        }
    }
    files.sort();
    files
}

/// The refusal a checkpoint error carries: an `Error::Io` of kind `StorageFull` whose inner
/// error is `InsufficientSpace`.
fn refusal(error: &Error) -> &InsufficientSpace {
    let Error::Io(io) = error else {
        panic!("an I/O error: {error:?}")
    };
    assert_eq!(io.kind(), std::io::ErrorKind::StorageFull, "{error:?}");
    io.get_ref()
        .and_then(|inner| inner.downcast_ref::<InsufficientSpace>())
        .unwrap_or_else(|| panic!("InsufficientSpace inside: {error:?}"))
}

/// A refused checkpoint names the file, the free bytes and the bytes needed: the file's length
/// plus one WAL segment, which the message says.
#[test]
fn a_checkpoint_without_room_names_the_file_and_both_numbers() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("space.grafeo");
    let db = database(&path);
    let length = std::fs::metadata(&path).unwrap().len();
    let error = with_free_space(Some(TOO_LITTLE), || db.wal_checkpoint()).unwrap_err();
    let space = refusal(&error);
    assert!(space.path.ends_with("space.grafeo"), "{:?}", space.path);
    assert_eq!(space.available, TOO_LITTLE);
    assert_eq!(
        space.needed,
        length + DEFAULT_SEGMENT_BYTES,
        "the file's length and one WAL segment"
    );
    let message = error.to_string();
    for number in [space.available, space.needed] {
        assert!(message.contains(&number.to_string()), "{number}: {message}");
    }
    assert!(
        message.to_lowercase().contains("wal"),
        "says the WAL's headroom is in the figure: {message}"
    );
    db.close().unwrap();
}

const PATH_VAR: &str = "GRAFEO_FREE_SPACE_PATH";

/// The length and modification time of a file: what can be read of a database's file while it is
/// open (on Windows its contents cannot be, by any other handle).
fn stamp(path: &Path) -> (u64, std::time::SystemTime) {
    let metadata = std::fs::metadata(path).unwrap();
    (metadata.len(), metadata.modified().unwrap())
}

/// The child: a database with a checkpoint and 500 commits after it, then a checkpoint refused for
/// space, which must leave the file's length and time and the directory as they were; then it ends
/// as a crash does (no close), so the parent opens what the refusal left.
#[test]
fn refused_checkpoint_child() {
    let Some(path) = std::env::var_os(PATH_VAR) else {
        return;
    };
    let path = PathBuf::from(path);
    let db = database(&path);
    db.wal_checkpoint().unwrap();
    db.session()
        .execute("UNWIND range(2001, 2500) AS i INSERT (:Person {i: i})")
        .unwrap();
    let file = stamp(&path);
    let files = listing(path.parent().unwrap());
    with_free_space(Some(TOO_LITTLE), || db.wal_checkpoint()).unwrap_err();
    assert_eq!(stamp(&path), file, "the file is as it was");
    assert_eq!(
        listing(path.parent().unwrap()),
        files,
        "no partial file left"
    );
    std::process::exit(0);
}

/// A refused checkpoint writes nothing (the child checks the file and the directory), and a
/// crash right after it recovers every write from the previous checkpoint and the WAL.
#[test]
fn a_refused_checkpoint_leaves_the_previous_one_and_the_wal() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("space.grafeo");
    let mut child = grafeo_common::child_test!("refused_checkpoint_child");
    child.env(PATH_VAR, &path);
    let status = grafeo_common::testing::child_process::run(&mut child).unwrap();
    assert!(status.success(), "the child failed");
    let recovered = GrafeoDB::open(&path).unwrap();
    assert_eq!(
        people(&recovered),
        2500,
        "the WAL holds what the file does not"
    );
    recovered.close().unwrap();
}

/// After a refused checkpoint the database answers and takes writes (nothing was poisoned), and
/// the next checkpoint with room succeeds and holds everything.
#[test]
fn the_database_goes_on_and_a_later_checkpoint_with_room_succeeds() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("space.grafeo");
    let db = database(&path);
    with_free_space(Some(TOO_LITTLE), || db.wal_checkpoint()).unwrap_err();
    assert_eq!(people(&db), 2000);
    db.session()
        .execute("INSERT (:Person {i: 2001, name: 'Gus'})")
        .unwrap();
    db.wal_checkpoint().unwrap();
    db.close().unwrap();
    let db = GrafeoDB::open(&path).unwrap();
    assert_eq!(people(&db), 2001);
    db.close().unwrap();
}

/// The check is exact: one byte less than the needed figure is refused, exactly that many is
/// written.
#[test]
fn the_check_is_exact_at_the_space_needed() {
    let dir = tempfile::tempdir().unwrap();
    let db = database(&dir.path().join("space.grafeo"));
    let error = with_free_space(Some(TOO_LITTLE), || db.wal_checkpoint()).unwrap_err();
    let needed = refusal(&error).needed;
    let short = with_free_space(Some(needed - 1), || db.wal_checkpoint()).unwrap_err();
    assert_eq!(
        refusal(&short).needed,
        needed,
        "the same figure, one byte short"
    );
    with_free_space(Some(needed), || db.wal_checkpoint())
        .unwrap_or_else(|error| panic!("exactly {needed}: {error}"));
    db.close().unwrap();
}

/// A probe that fails skips the check: the checkpoint goes ahead and is written.
#[test]
fn a_failing_probe_skips_the_check() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("space.grafeo");
    let db = database(&path);
    with_free_space(None, || db.wal_checkpoint())
        .unwrap_or_else(|error| panic!("a failing probe refused the checkpoint: {error}"));
    db.close().unwrap();
    let db = GrafeoDB::open(&path).unwrap();
    assert_eq!(people(&db), 2000);
    db.close().unwrap();
}

/// A database with a checkpoint and 2,000 commits after it, at a fresh path in `dir`.
fn with_commits_after_a_checkpoint(dir: &Path) -> (GrafeoDB, PathBuf) {
    let path = dir.join("space.grafeo");
    let db = database(&path);
    db.wal_checkpoint().unwrap();
    db.session()
        .execute("UNWIND range(2001, 4000) AS i INSERT (:Person {i: i, name: 'Mia'})")
        .unwrap();
    (db, path)
}

/// A write that fails after the check passed and before the new header, as when the disk fills in
/// between: the file is cut back to its length before the checkpoint, so the WAL keeps its room,
/// and the database goes on. Each injected failure gets a fresh database, since a failure earlier
/// in the checkpoint (the WAL's sync) poisons the writer, as it should.
#[test]
fn a_write_failing_before_the_header_cuts_the_file_back() {
    for point in ["checkpoint:after_chunks", "checkpoint:after_data_sync"] {
        let mut fired = None;
        for count in 1..=16 {
            let dir = tempfile::tempdir().unwrap();
            let (db, path) = with_commits_after_a_checkpoint(dir.path());
            let length = std::fs::metadata(&path).unwrap().len();
            // Room on any host: the injected failure, not the real disk, stops the checkpoint.
            match with_free_space(Some(PLENTY), || {
                with_failure_at(count, || db.wal_checkpoint())
            }) {
                Err(error) if error.to_string().contains(point) => {
                    fired = Some((dir, db, path, length));
                    break;
                }
                Err(_) => {}
                Ok(()) => break,
            }
        }
        let Some((_dir, db, path, length)) = fired else {
            panic!("{point}: no checkpoint failed there");
        };
        assert_eq!(
            std::fs::metadata(&path).unwrap().len(),
            length,
            "{point}: the file is cut back to its length"
        );
        assert_eq!(people(&db), 4000, "{point}");
        db.wal_checkpoint()
            .unwrap_or_else(|error| panic!("{point}: the next checkpoint: {error}"));
        db.close().unwrap();
        let db = GrafeoDB::open(&path).unwrap();
        assert_eq!(people(&db), 4000, "{point}");
        db.close().unwrap();
    }
}

/// A close without room for its checkpoint keeps the WAL: the file is as it was, the sidecar WAL
/// is still there, and a reopen with room finds every write.
#[test]
fn a_close_without_room_keeps_every_write_in_the_wal() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("space.grafeo");
    let db = database(&path);
    db.wal_checkpoint().unwrap();
    db.session()
        .execute("UNWIND range(2001, 2300) AS i INSERT (:Person {i: i})")
        .unwrap();
    let file = stamp(&path);
    with_free_space(Some(TOO_LITTLE), || {
        let _ = db.close();
    });
    assert_eq!(stamp(&path), file, "the file is as it was");
    assert!(sidecar_wal(&path).exists(), "the WAL is kept");
    drop(db);
    let db = GrafeoDB::open(&path).unwrap();
    assert_eq!(people(&db), 2300);
    db.close().unwrap();
}
