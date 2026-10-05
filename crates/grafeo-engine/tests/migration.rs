//! Migration of 0.5.x databases to the 0.6 format on a read-write open.
//!
//! A read-write open of a database written by 0.5.x (a file, or a WAL
//! directory) writes the 0.6 image to `<path>.migrating`, moves the old file
//! (and its sidecar WAL) or the old directory to `<path>.pre-0.6`, and renames
//! the image into place, all under the lock file `<path>.migrate.lock`. These
//! tests crash a migration at each step in a child process, build each state a
//! crash can leave, hold the lock from another process, and open each
//! half-migrated state read-only.
//!
//! ```bash
//! cargo test -p grafeo-engine --all-features --test migration
//! ```

#![cfg(all(
    feature = "lpg",
    feature = "gql",
    feature = "wal",
    feature = "grafeo-file"
))]

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{Duration, Instant};

use grafeo_common::testing::child_process;
use grafeo_common::types::Value;
use grafeo_engine::GrafeoDB;
use grafeo_storage::file::detect::{OnDisk, detect};

#[path = "common/unlistable.rs"]
mod unlistable;

/// The 0.5.44 database whose second session is only in its sidecar WAL, so a
/// migration has a file and a sidecar WAL to keep.
fn fixture() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/released/0.5.44/unflushed.grafeo")
}

/// `<path><suffix>`, next to the database file.
fn with_suffix(path: &Path, suffix: &str) -> PathBuf {
    let mut name = path.as_os_str().to_owned();
    name.push(suffix);
    PathBuf::from(name)
}

fn copy(from: &Path, to: &Path) {
    if from.is_dir() {
        std::fs::create_dir_all(to).unwrap();
        for entry in std::fs::read_dir(from).unwrap() {
            let entry = entry.unwrap();
            copy(&entry.path(), &to.join(entry.file_name()));
        }
    } else {
        std::fs::copy(from, to).unwrap();
    }
}

/// An entry of [`files`].
#[derive(Debug, PartialEq, Eq)]
enum Entry {
    Directory,
    File(Vec<u8>),
    /// A migration lock file: only its presence counts, because Windows refuses
    /// to read a file another process holds locked.
    LockFile,
}

/// Every file and directory under `root`, with the bytes of each file. Leaves out
/// the spill directories queries create next to a database (scratch space, not
/// part of it).
fn files(root: &Path) -> BTreeMap<PathBuf, Entry> {
    let mut found = BTreeMap::new();
    let mut pending = vec![root.to_path_buf()];
    while let Some(dir) = pending.pop() {
        for entry in std::fs::read_dir(&dir).unwrap() {
            let path = entry.unwrap().path();
            let relative = path.strip_prefix(root).unwrap().to_path_buf();
            let name = relative.to_string_lossy().into_owned();
            if relative
                .components()
                .next()
                .is_some_and(|first| first.as_os_str().to_string_lossy().ends_with(".spill"))
            {
                continue;
            }
            if path.is_dir() {
                found.insert(relative, Entry::Directory);
                pending.push(path);
            } else if name.ends_with(".migrate.lock") {
                found.insert(relative, Entry::LockFile);
            } else {
                found.insert(relative, Entry::File(std::fs::read(&path).unwrap()));
            }
        }
    }
    found
}

/// Copies the fixture file to `to` and its sidecar WAL to `<to>.wal`.
fn copy_fixture(to: &Path) {
    copy(&fixture(), to);
    copy(&with_suffix(&fixture(), ".wal"), &with_suffix(to, ".wal"));
}

/// Writes a 0.6 database holding a person for each of `people` to `to`.
fn write_v3(to: &Path, people: &[&str]) {
    let db = GrafeoDB::new_in_memory();
    for name in people {
        db.execute(&format!("INSERT (:Person {{name: '{name}'}})"))
            .unwrap();
    }
    db.save(to).unwrap();
}

/// What a database holds, as queries see it.
#[derive(Debug, PartialEq)]
struct Contents {
    /// The label sets of the default graph, with their node counts.
    nodes: Vec<Vec<Value>>,
    /// The edge types of the default graph, with their counts.
    edges: Vec<Vec<Value>>,
    /// Every person with their properties.
    people: Vec<Vec<Value>>,
    /// The named graphs, with the names of their nodes.
    graphs: Vec<(String, Vec<Vec<Value>>)>,
}

fn rows(db: &GrafeoDB, query: &str) -> Vec<Vec<Value>> {
    db.execute(query)
        .unwrap_or_else(|error| panic!("{query}: {error}"))
        .rows()
        .to_vec()
}

fn contents(db: &GrafeoDB) -> Contents {
    let mut graphs: Vec<(String, Vec<Vec<Value>>)> = db
        .list_graphs()
        .into_iter()
        .map(|name| {
            let nodes = db
                .graph(&name)
                .unwrap()
                .execute("MATCH (n) RETURN n.name AS name ORDER BY name")
                .unwrap()
                .rows()
                .to_vec();
            (name, nodes)
        })
        .collect();
    graphs.sort_by(|a, b| a.0.cmp(&b.0));
    Contents {
        nodes: rows(
            db,
            "MATCH (n) RETURN labels(n) AS l, count(*) AS c ORDER BY l",
        ),
        edges: rows(
            db,
            "MATCH ()-[r]->() RETURN type(r) AS t, count(*) AS c ORDER BY t",
        ),
        people: rows(
            db,
            "MATCH (p:Person) RETURN p.name AS name, p.email, p.age, p.score, p.tags \
             ORDER BY name",
        ),
        graphs,
    }
}

/// The names of the people in `db`, sorted.
fn people(db: &GrafeoDB) -> Vec<Value> {
    rows(db, "MATCH (p:Person) RETURN p.name AS name ORDER BY name")
        .into_iter()
        .map(|row| row[0].clone())
        .collect()
}

fn names(people: &[&str]) -> Vec<Value> {
    people.iter().map(|name| Value::from(*name)).collect()
}

/// What the fixture holds: a read-only open of a copy, which reads the 0.5.x
/// file and replays its sidecar WAL without migrating it.
fn fixture_contents() -> Contents {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("db.grafeo");
    copy_fixture(&path);
    let db = GrafeoDB::open_read_only(&path).unwrap();
    let found = contents(&db);
    db.close().unwrap();
    assert!(
        found.people.len() > 1 && !found.graphs.is_empty(),
        "the fixture holds people and named graphs: {found:?}"
    );
    found
}

/// The error of a read-write open of `path`.
fn open_error(path: &Path) -> String {
    match GrafeoDB::open(path) {
        Ok(_) => panic!("the open of {} succeeded", path.display()),
        Err(error) => error.to_string(),
    }
}

/// The error of a read-only open of `path`.
fn read_only_error(path: &Path) -> String {
    match GrafeoDB::open_read_only(path) {
        Ok(_) => panic!("the read-only open of {} succeeded", path.display()),
        Err(error) => error.to_string(),
    }
}

/// The error of `open_in_memory` of `path`.
fn in_memory_error(path: &Path) -> String {
    match GrafeoDB::open_in_memory(path) {
        Ok(_) => panic!("open_in_memory of {} succeeded", path.display()),
        Err(error) => error.to_string(),
    }
}

/// Fails if a side file of a migration is left next to `path`.
fn assert_no_leftovers(path: &Path, context: &str) {
    for leftover in [".migrating", ".migrating.creating", ".migrate.lock"] {
        assert!(
            !with_suffix(path, leftover).exists(),
            "{context}: <path>{leftover} is left behind"
        );
    }
}

/// The 0.5.x files a migration keeps, to compare the kept copies with.
struct Kept {
    /// The database file or WAL directory, kept as `<path>.pre-0.6`.
    file: PathBuf,
    /// The sidecar WAL, kept as `<path>.pre-0.6.wal`.
    wal: Option<PathBuf>,
    /// The pending checkpoint image, kept as `<path>.pre-0.6.checkpoint`.
    checkpoint: Option<PathBuf>,
}

impl Kept {
    /// Where a test puts the database in `dir`: `db.grafeo` for a file, `db`
    /// for a WAL directory (0.5.x created a directory only for a path without
    /// the `.grafeo` extension, unless told otherwise).
    fn path_in(&self, dir: &Path) -> PathBuf {
        dir.join(if self.file.is_dir() {
            "db"
        } else {
            "db.grafeo"
        })
    }
}

/// What a migration of the fixture keeps: the file and its sidecar WAL.
fn fixture_kept() -> Kept {
    Kept {
        file: fixture(),
        wal: Some(with_suffix(&fixture(), ".wal")),
        checkpoint: None,
    }
}

/// The 0.5.44 WAL-directory database: its `wal/` and the `LOCK` file 0.5.44
/// creates. A migration keeps the whole directory.
fn directory() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/released/0.5.44/directory")
}

/// What a migration of the WAL-directory fixture keeps: the directory.
fn directory_kept() -> Kept {
    Kept {
        file: directory(),
        wal: None,
        checkpoint: None,
    }
}

/// The same for the 0.5.43 WAL directory, which has no `LOCK` file (0.5.43
/// took no lock): the migration creates one while it holds the directory, and
/// removes it before the directory moves.
fn directory_0_5_43_kept() -> Kept {
    Kept {
        file: Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures/released/0.5.43/directory"),
        wal: None,
        checkpoint: None,
    }
}

/// A 0.5.44 file whose last checkpoint left its new image pending: the file is
/// `0.5.44/closed.grafeo`, the image `<p>.checkpoint` is `0.5.43/closed.grafeo`
/// (another database, so tests can tell them apart).
fn pending_kept() -> Kept {
    let released = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/released");
    Kept {
        file: released.join("0.5.44/closed.grafeo"),
        wal: None,
        checkpoint: Some(released.join("0.5.43/closed.grafeo")),
    }
}

/// Puts the 0.5.x files of `kept` at `path`, under their 0.5.x names.
fn arrange_kept(kept: &Kept, path: &Path) {
    copy(&kept.file, path);
    if let Some(wal) = &kept.wal {
        copy(wal, &with_suffix(path, ".wal"));
    }
    if let Some(image) = &kept.checkpoint {
        copy(image, &with_suffix(path, ".checkpoint"));
    }
}

/// What a read-only open of the 0.5.x files of `kept` finds.
fn kept_contents(kept: &Kept) -> Contents {
    let dir = tempfile::tempdir().unwrap();
    let path = kept.path_in(dir.path());
    arrange_kept(kept, &path);
    let db = GrafeoDB::open_read_only(&path).unwrap();
    let found = contents(&db);
    db.close().unwrap();
    found
}

/// Checks that the kept copies next to `path` hold the files of `kept` byte for
/// byte (every file of a kept directory, and no other), that nothing is kept
/// that `kept` does not have, and that no 0.5.x checkpoint image is left under
/// the database's name.
fn assert_kept(path: &Path, kept: &Kept, context: &str) {
    assert_kept_with(path, kept, context, false);
}

/// [`assert_kept`], where `lock_may_remain` allows a kept directory one more
/// file than the original: an empty `LOCK`, which a crash leaves when it stops
/// a migration that created it to lock a directory without one.
fn assert_kept_with(path: &Path, kept: &Kept, context: &str, lock_may_remain: bool) {
    let kept_file = with_suffix(path, ".pre-0.6");
    if kept.file.is_dir() {
        let mut found = if kept_file.is_dir() {
            files(&kept_file)
        } else {
            BTreeMap::new()
        };
        if lock_may_remain
            && !kept.file.join("LOCK").exists()
            && found.get(Path::new("LOCK")) == Some(&Entry::File(Vec::new()))
        {
            found.remove(Path::new("LOCK"));
        }
        assert!(
            kept_file.is_dir() && found == files(&kept.file),
            "{context}: <path>.pre-0.6/ holds the files of the 0.5.x directory, byte for byte: \
             kept {:?}, original {:?}",
            kept_file
                .is_dir()
                .then(|| files(&kept_file).into_keys().collect::<Vec<_>>()),
            files(&kept.file).into_keys().collect::<Vec<_>>()
        );
    } else {
        assert!(
            std::fs::read(&kept_file).unwrap() == std::fs::read(&kept.file).unwrap(),
            "{context}: <path>.pre-0.6 holds the bytes of the 0.5.x file"
        );
    }
    let kept_wal = with_suffix(path, ".pre-0.6.wal");
    match &kept.wal {
        Some(wal) => {
            assert!(
                kept_wal.is_dir(),
                "{context}: the 0.5.x sidecar WAL is kept as <path>.pre-0.6.wal"
            );
            assert!(
                files(&kept_wal) == files(wal),
                "{context}: <path>.pre-0.6.wal holds the files of the 0.5.x sidecar WAL"
            );
        }
        None => assert!(
            !kept_wal.exists(),
            "{context}: there was no sidecar WAL to keep"
        ),
    }
    let kept_image = with_suffix(path, ".pre-0.6.checkpoint");
    match &kept.checkpoint {
        Some(image) => assert!(
            kept_image.exists()
                && std::fs::read(&kept_image).unwrap() == std::fs::read(image).unwrap(),
            "{context}: <path>.pre-0.6.checkpoint holds the bytes of the pending image"
        ),
        None => assert!(
            !kept_image.exists(),
            "{context}: there was no pending checkpoint image to keep"
        ),
    }
    assert!(
        !with_suffix(path, ".checkpoint").exists(),
        "{context}: no 0.5.x checkpoint image is left under the database's name"
    );
}

/// Opens `path` read-write and checks that it holds the `expected` data in the
/// 0.6 format, that the files of `kept` are kept, and that nothing of the
/// migration is left.
fn assert_migrated(path: &Path, expected: &Contents, kept: &Kept, context: &str) {
    assert_migrated_with(path, expected, kept, context, false);
}

/// [`assert_migrated`], with the `lock_may_remain` of [`assert_kept_with`].
fn assert_migrated_with(
    path: &Path,
    expected: &Contents,
    kept: &Kept,
    context: &str,
    lock_may_remain: bool,
) {
    let db = GrafeoDB::open(path).unwrap_or_else(|error| panic!("{context}: {error}"));
    assert_eq!(
        contents(&db),
        *expected,
        "{context}: the database holds the 0.5.x data"
    );
    db.close().unwrap();
    drop(db);
    assert_eq!(
        detect(path).unwrap(),
        OnDisk::Current,
        "{context}: the database file is in the 0.6 format"
    );
    assert_kept_with(path, kept, context, lock_may_remain);
    assert_no_leftovers(path, context);
}

// =========================================================================
// Crashes
// =========================================================================

/// The crash points a migrating open of the fixture reaches, in order: writing
/// the image (creating the file, then its checkpoint), then the steps of the
/// migration, with a point after each rename (named by its target).
#[cfg(feature = "testing-crash-injection")]
const MIGRATION_POINTS: [&str; 11] = [
    "create:after_write",
    "checkpoint:after_chunks",
    "checkpoint:after_data_sync",
    "checkpoint:after_header",
    "checkpoint:before_trim",
    "migrate:after_image",
    "migrate:renamed:.pre-0.6",
    "migrate:renamed:.pre-0.6.wal",
    "migrate:after_old",
    "migrate:renamed:database",
    "migrate:after_new",
];

/// The same for a file with a pending checkpoint image and no sidecar WAL
/// ([`pending_kept`]): the image moves instead of the WAL.
#[cfg(feature = "testing-crash-injection")]
const PENDING_MIGRATION_POINTS: [&str; 11] = [
    "create:after_write",
    "checkpoint:after_chunks",
    "checkpoint:after_data_sync",
    "checkpoint:after_header",
    "checkpoint:before_trim",
    "migrate:after_image",
    "migrate:renamed:.pre-0.6",
    "migrate:renamed:.pre-0.6.checkpoint",
    "migrate:after_old",
    "migrate:renamed:database",
    "migrate:after_new",
];

/// The same for a WAL directory ([`directory_kept`]): it has no side files,
/// and the whole directory moves to `<p>.pre-0.6`.
#[cfg(feature = "testing-crash-injection")]
const DIRECTORY_MIGRATION_POINTS: [&str; 10] = [
    "create:after_write",
    "checkpoint:after_chunks",
    "checkpoint:after_data_sync",
    "checkpoint:after_header",
    "checkpoint:before_trim",
    "migrate:after_image",
    "migrate:renamed:.pre-0.6",
    "migrate:after_old",
    "migrate:renamed:database",
    "migrate:after_new",
];

#[cfg(feature = "testing-crash-injection")]
const CRASH_POINT_VAR: &str = "GRAFEO_MIGRATION_CRASH_POINT";
/// The database path a child process works on.
const PATH_VAR: &str = "GRAFEO_MIGRATION_PATH";
/// Exit code of a child whose open crashed.
#[cfg(feature = "testing-crash-injection")]
const CRASHED: i32 = 3;
/// Exit code of a child whose open completed.
#[cfg(feature = "testing-crash-injection")]
const OPENED: i32 = 4;

/// Opens the 0.5.x database at `path` read-write in a child process that crashes
/// at the `crash_point`-th crash point. Returns the crash point's name, or `None`
/// if the open completed.
#[cfg(feature = "testing-crash-injection")]
fn open_in_child(crash_point: usize, path: &Path) -> Option<String> {
    let output = child_process::output(
        Command::new(std::env::current_exe().unwrap())
            .args(["--exact", "migrate_child", "--nocapture"])
            .env(CRASH_POINT_VAR, crash_point.to_string())
            .env(PATH_VAR, path),
    )
    .unwrap();
    let stderr = String::from_utf8_lossy(&output.stderr);
    match output.status.code() {
        Some(OPENED) => None,
        Some(CRASHED) => {
            let point = stderr
                .lines()
                .find_map(|line| line.split_once("crash injection at: "))
                .map(|(_, point)| point.trim().to_string());
            Some(point.unwrap_or_else(|| panic!("the child names no crash point:\n{stderr}")))
        }
        other => panic!("crash_point={crash_point}: the child exited with {other:?}:\n{stderr}"),
    }
}

/// Child-process entry for [`open_in_child`]; a no-op when run directly.
#[test]
#[cfg(feature = "testing-crash-injection")]
fn migrate_child() {
    use grafeo_common::testing::crash::{CrashResult, with_crash_at};

    let (Ok(point), Some(path)) = (std::env::var(CRASH_POINT_VAR), std::env::var_os(PATH_VAR))
    else {
        return;
    };
    let path = PathBuf::from(path);
    let result = with_crash_at(point.parse().unwrap(), move || GrafeoDB::open(&path));
    // Exit without running destructors, like a crash.
    match result {
        CrashResult::Completed(Ok(_db)) => std::process::exit(OPENED),
        CrashResult::Completed(Err(error)) => panic!("the migrating open failed: {error}"),
        _ => std::process::exit(CRASHED),
    }
}

/// For each crash point of a read-write open of the files `arrange` puts at a
/// path, and once past the last: runs the open in a child process that crashes
/// there, then checks that an open in this process finds the 0.5.x files of
/// `kept` migrated ([`assert_migrated`]). Checks that the open reaches exactly
/// `points`, in order, and then completes.
///
/// A crash before the old database moved (`migrate:renamed:.pre-0.6`) may
/// leave an empty `LOCK` in a directory that had none: the migration created
/// it to lock the directory, and only removes it when it is not stopped. The
/// kept directory may hold it ([`assert_kept_with`]); one kept after a later
/// crash, or after none, is the original exactly.
#[cfg(feature = "testing-crash-injection")]
fn sweep_crashes(arrange: impl Fn(&Path), points: &[&str], kept: &Kept) {
    let expected = kept_contents(kept);
    let moved_at = points
        .iter()
        .position(|point| *point == "migrate:renamed:.pre-0.6");
    let mut reached = Vec::new();
    for crash_point in 1..=points.len() + 1 {
        let dir = tempfile::tempdir().unwrap();
        let path = kept.path_in(dir.path());
        arrange(&path);

        let crashed_at = open_in_child(crash_point, &path);
        let context = format!(
            "crash_point={crash_point} ({})",
            crashed_at.as_deref().unwrap_or("no crash")
        );
        if crashed_at.is_some() {
            assert!(
                with_suffix(&path, ".migrate.lock").exists(),
                "{context}: a crash inside the migration leaves its lock file"
            );
        }
        let before_the_move = crashed_at.is_some() && moved_at.is_some_and(|at| crash_point <= at);
        assert_migrated_with(&path, &expected, kept, &context, before_the_move);
        reached.push(crashed_at);
    }

    let mut expected_points: Vec<Option<String>> = points
        .iter()
        .map(|point| Some((*point).to_string()))
        .collect();
    expected_points.push(None);
    assert_eq!(
        reached, expected_points,
        "the open reaches these crash points, then completes"
    );
}

/// A crash at any point of a migration, in another process, leaves files the
/// next read-write open migrates from: afterwards the database holds all the
/// data, the old files are kept byte for byte, and nothing else is left. A
/// crash point after each rename pins their order: the 0.5.x side files (the
/// sidecar WAL of the fixture, the pending checkpoint image of the second
/// arrangement) are out of the way before the image becomes the database. A
/// WAL directory goes through the same steps, as one rename of the directory.
#[test]
#[cfg(feature = "testing-crash-injection")]
fn a_crash_at_each_migration_step_recovers() {
    let fixture = fixture_kept();
    sweep_crashes(
        |path| arrange_kept(&fixture, path),
        &MIGRATION_POINTS,
        &fixture,
    );
    let pending = pending_kept();
    sweep_crashes(
        |path| arrange_kept(&pending, path),
        &PENDING_MIGRATION_POINTS,
        &pending,
    );
    let directory = directory_kept();
    sweep_crashes(
        |path| arrange_kept(&directory, path),
        &DIRECTORY_MIGRATION_POINTS,
        &directory,
    );
    let without_lock = directory_0_5_43_kept();
    sweep_crashes(
        |path| arrange_kept(&without_lock, path),
        &DIRECTORY_MIGRATION_POINTS,
        &without_lock,
    );
}

/// The crash points of an open that finishes a migration cut off between the
/// move of the old file and the move of its sidecar WAL.
#[cfg(feature = "testing-crash-injection")]
const FINISH_POINTS: [&str; 2] = ["migrate:renamed:.pre-0.6.wal", "migrate:renamed:database"];

/// The same with a pending checkpoint image instead of a sidecar WAL.
#[cfg(feature = "testing-crash-injection")]
const PENDING_FINISH_POINTS: [&str; 2] = [
    "migrate:renamed:.pre-0.6.checkpoint",
    "migrate:renamed:database",
];

/// Writes the image a migration of the 0.5.x files of `kept` writes, to `to`.
#[cfg(feature = "testing-crash-injection")]
fn write_migrated_image(kept: &Kept, to: &Path) {
    let dir = tempfile::tempdir().unwrap();
    let source = dir.path().join("source.grafeo");
    arrange_kept(kept, &source);
    let db = GrafeoDB::open_read_only(&source).unwrap();
    db.save(to).unwrap();
    db.close().unwrap();
}

/// The state a migration cut off between its renames leaves: the file moved to
/// `<p>.pre-0.6`, its side files still under their 0.5.x names, the complete
/// image at `<p>.migrating`, and the lock file a crash leaves.
#[cfg(feature = "testing-crash-injection")]
fn arrange_between_renames(kept: &Kept, image: &Path, path: &Path) {
    copy(&kept.file, &with_suffix(path, ".pre-0.6"));
    if let Some(wal) = &kept.wal {
        copy(wal, &with_suffix(path, ".wal"));
    }
    if let Some(checkpoint) = &kept.checkpoint {
        copy(checkpoint, &with_suffix(path, ".checkpoint"));
    }
    copy(image, &with_suffix(path, ".migrating"));
    std::fs::write(with_suffix(path, ".migrate.lock"), b"").unwrap();
}

/// The same for a WAL directory, which has no side files to move.
#[cfg(feature = "testing-crash-injection")]
const DIRECTORY_FINISH_POINTS: [&str; 1] = ["migrate:renamed:database"];

/// Finishing a migration cut off between its renames survives a crash at each
/// of its own renames: the side files (sidecar WAL, pending checkpoint image)
/// move to their kept names before the image becomes the database, so the 0.6
/// database never replays the 0.5.x WAL, and no 0.5.x image is left under its
/// name. A WAL directory moved to `<p>.pre-0.6/` is finished the same way.
#[test]
#[cfg(feature = "testing-crash-injection")]
fn a_crash_while_finishing_a_migration_recovers() {
    let images = tempfile::tempdir().unwrap();
    for (index, (kept, points)) in [
        (fixture_kept(), &FINISH_POINTS[..]),
        (pending_kept(), &PENDING_FINISH_POINTS[..]),
        (directory_kept(), &DIRECTORY_FINISH_POINTS[..]),
    ]
    .into_iter()
    .enumerate()
    {
        let image = images.path().join(format!("image-{index}.grafeo"));
        write_migrated_image(&kept, &image);
        sweep_crashes(
            |path| arrange_between_renames(&kept, &image, path),
            points,
            &kept,
        );
    }
}

// =========================================================================
// Interrupted migrations
// =========================================================================

/// Each state a crash can leave is finished by the next read-write open, from
/// the files present.
#[test]
fn interrupted_migrations_resume_from_the_files_present() {
    let expected = fixture_contents();

    // `<p>` (0.5.x) and `<p>.migrating`: the image may be incomplete, so it goes
    // and the migration runs again.
    {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("db.grafeo");
        copy_fixture(&path);
        write_v3(&with_suffix(&path, ".migrating"), &["Vincent"]);
        assert_migrated(&path, &expected, &fixture_kept(), "0.5.x file and an image");
    }

    // `<p>.pre-0.6` and `<p>.migrating`, no `<p>`: the old database was moved
    // after the image was complete, so the image goes into place.
    {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("db.grafeo");
        copy(&fixture(), &with_suffix(&path, ".pre-0.6"));
        write_v3(&with_suffix(&path, ".migrating"), &["Vincent", "Mia"]);
        let db = GrafeoDB::open(&path).unwrap();
        assert_eq!(
            people(&db),
            names(&["Mia", "Vincent"]),
            "kept copy and an image: the database is the image"
        );
        db.close().unwrap();
        drop(db);
        assert!(
            std::fs::read(with_suffix(&path, ".pre-0.6")).unwrap()
                == std::fs::read(fixture()).unwrap(),
            "kept copy and an image: the kept copy stays as it is"
        );
        assert_no_leftovers(&path, "kept copy and an image");
    }

    // The same, with the 0.5.x sidecar WAL still at `<p>.wal` (the moves of the
    // old file and its WAL are two renames): the WAL is kept with the old file,
    // never replayed into the image.
    {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("db.grafeo");
        copy(&fixture(), &with_suffix(&path, ".pre-0.6"));
        copy(
            &with_suffix(&fixture(), ".wal"),
            &with_suffix(&path, ".wal"),
        );
        write_v3(&with_suffix(&path, ".migrating"), &["Vincent", "Mia"]);
        let db = GrafeoDB::open(&path).unwrap();
        assert_eq!(
            people(&db),
            names(&["Mia", "Vincent"]),
            "kept copy, 0.5.x WAL and an image: the 0.5.x WAL is not replayed"
        );
        db.close().unwrap();
        drop(db);
        assert_kept(&path, &fixture_kept(), "kept copy, 0.5.x WAL and an image");
        assert_no_leftovers(&path, "kept copy, 0.5.x WAL and an image");
    }

    // `<p>.migrating` only: the old database is missing, so nothing is guessed.
    {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("db.grafeo");
        let migrating = with_suffix(&path, ".migrating");
        write_v3(&migrating, &["Vincent"]);
        let before = files(dir.path());
        let error = open_error(&path);
        assert!(
            error.contains(&migrating.display().to_string()) && error.contains("missing"),
            "only an image: the error names the image and says the old database is \
             missing: {error}"
        );
        assert!(
            files(dir.path()) == before,
            "only an image: the failed open changes nothing"
        );
    }

    // `<p>` (0.6) and `<p>.pre-0.6`: a finished migration, nothing to do.
    {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("db.grafeo");
        write_v3(&path, &["Vincent"]);
        copy(&fixture(), &with_suffix(&path, ".pre-0.6"));
        let db = GrafeoDB::open(&path).unwrap();
        assert_eq!(
            people(&db),
            names(&["Vincent"]),
            "a 0.6 file and a kept copy: the database is the 0.6 file"
        );
        db.close().unwrap();
        drop(db);
        assert!(
            std::fs::read(with_suffix(&path, ".pre-0.6")).unwrap()
                == std::fs::read(fixture()).unwrap(),
            "a 0.6 file and a kept copy: the kept copy stays as it is"
        );
        assert_no_leftovers(&path, "a 0.6 file and a kept copy");
    }
}

/// A power loss can keep a later rename of a migration and lose an earlier one:
/// the sidecar WAL (or the pending checkpoint image) under its kept name while
/// the 0.5.x file is still `<p>`, next to the image. The next read-write open
/// moves the side file back and migrates again, so the migrated database holds
/// the side file's changes, and the files are kept as after any migration.
#[test]
fn a_side_file_kept_before_its_database_file_moved_is_moved_back() {
    for (kept, side, kept_side) in [
        (fixture_kept(), ".wal", ".pre-0.6.wal"),
        (pending_kept(), ".checkpoint", ".pre-0.6.checkpoint"),
    ] {
        let expected = kept_contents(&kept);
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("db.grafeo");
        arrange_kept(&kept, &path);
        std::fs::rename(with_suffix(&path, side), with_suffix(&path, kept_side)).unwrap();
        write_v3(&with_suffix(&path, ".migrating"), &["Vincent"]);
        std::fs::write(with_suffix(&path, ".migrate.lock"), b"").unwrap();
        assert_migrated(
            &path,
            &expected,
            &kept,
            &format!("<p>{kept_side} without <p>.pre-0.6"),
        );
    }
}

/// In the same state a read-only open (and `open_in_memory`) would read the
/// 0.5.x file without its sidecar WAL or pending checkpoint image: it fails,
/// saying a read-write open finishes the migration, and changes nothing.
#[test]
fn a_read_only_open_refuses_a_side_file_kept_before_its_database_file_moved() {
    for (kept, side, kept_side) in [
        (fixture_kept(), ".wal", ".pre-0.6.wal"),
        (pending_kept(), ".checkpoint", ".pre-0.6.checkpoint"),
    ] {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("db.grafeo");
        arrange_kept(&kept, &path);
        std::fs::rename(with_suffix(&path, side), with_suffix(&path, kept_side)).unwrap();
        write_v3(&with_suffix(&path, ".migrating"), &["Vincent"]);
        let before = files(dir.path());
        for (open, error) in [
            ("read-only", read_only_error(&path)),
            ("in-memory", in_memory_error(&path)),
        ] {
            assert!(
                error.contains(&with_suffix(&path, kept_side).display().to_string())
                    && error.contains("a read-write open finishes the migration"),
                "<p>{kept_side}: the {open} error names the kept file and says a read-write \
                 open finishes the migration: {error}"
            );
            assert!(
                files(dir.path()) == before,
                "<p>{kept_side}: the {open} open changes nothing"
            );
        }
    }
}

/// The same leftovers next to a 0.6 `<p>` (a stale image and a kept WAL of an
/// earlier migration) are no part of a migration: a read-only open reads the
/// 0.6 file.
#[test]
fn a_read_only_open_of_a_0_6_file_ignores_a_stale_image_and_a_kept_wal() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("db.grafeo");
    write_v3(&path, &["Vincent"]);
    write_v3(&with_suffix(&path, ".migrating"), &["Mia"]);
    copy(
        &with_suffix(&fixture(), ".wal"),
        &with_suffix(&path, ".pre-0.6.wal"),
    );
    let db = GrafeoDB::open_read_only(&path).unwrap();
    assert_eq!(people(&db), names(&["Vincent"]), "the 0.6 file is read");
    db.close().unwrap();
}

/// A database that was migrated but whose 0.6 file is missing (only a kept
/// copy is left) is never recreated as an empty database: every open fails with
/// an error naming the kept copy, and nothing is created.
#[test]
fn a_missing_database_next_to_a_kept_copy_is_never_created() {
    for kept_suffix in [".pre-0.6", ".pre-0.6.wal"] {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("db.grafeo");
        let kept = with_suffix(&path, kept_suffix);
        if kept_suffix == ".pre-0.6" {
            copy(&fixture(), &kept);
        } else {
            copy(&with_suffix(&fixture(), ".wal"), &kept);
        }
        let before = files(dir.path());
        for (open, error) in [
            ("read-write", open_error(&path)),
            ("read-only", read_only_error(&path)),
            ("in-memory", in_memory_error(&path)),
        ] {
            assert!(
                error.contains(&kept.display().to_string()) && error.contains("missing"),
                "only {kept_suffix}: the {open} error names the kept copy and says the \
                 database file is missing: {error}"
            );
            assert!(
                files(dir.path()) == before,
                "only {kept_suffix}: the {open} open creates nothing"
            );
        }
    }
}

/// An open of a migration cut between its renames (the old file moved, the image
/// not yet in place) waits for the process still holding the migration lock,
/// then opens the database that process installed. This pins
/// `finish_interrupted` taking the lock when it finds a lock file; the race of
/// an open that checked before the lock file existed is
/// [`a_create_never_races_a_migration_between_its_renames`].
#[test]
fn an_open_of_a_migration_cut_between_its_renames_waits_for_the_process_holding_the_lock() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("db.grafeo");
    copy(&fixture(), &with_suffix(&path, ".pre-0.6"));
    copy(
        &with_suffix(&fixture(), ".wal"),
        &with_suffix(&path, ".pre-0.6.wal"),
    );
    write_v3(&with_suffix(&path, ".migrating"), &["Vincent", "Mia"]);

    let holder = hold_migrate_lock(&path, LockMode::Install);
    let releaser = holder.release_after(Duration::from_secs(1));
    let db = GrafeoDB::open(&path)
        .unwrap_or_else(|error| panic!("the open after the migration finished: {error}"));
    releaser.join().unwrap();
    holder.join();
    assert_eq!(
        people(&db),
        names(&["Mia", "Vincent"]),
        "the open found the migrated database, not a new empty one"
    );
    db.close().unwrap();
    drop(db);
    assert_kept(
        &path,
        &fixture_kept(),
        "a migration cut between its renames",
    );
    assert_no_leftovers(&path, "a migration cut between its renames");
}

/// A read-only open (and `open_in_memory`) never changes a half-migrated
/// database: it reads a 0.5.x file that is still in place, and otherwise fails
/// with an error that says what to do.
#[test]
fn a_read_only_open_never_finishes_a_migration() {
    let expected = fixture_contents();

    // `<p>` (0.5.x) and `<p>.migrating`: the 0.5.x file is read as it is.
    {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("db.grafeo");
        copy_fixture(&path);
        write_v3(&with_suffix(&path, ".migrating"), &["Vincent"]);
        let before = files(dir.path());
        let db = GrafeoDB::open_read_only(&path).unwrap();
        assert_eq!(
            contents(&db),
            expected,
            "0.5.x file and an image: the read-only open reads the 0.5.x file"
        );
        db.close().unwrap();
        drop(db);
        let db = GrafeoDB::open_in_memory(&path).unwrap();
        assert_eq!(
            contents(&db),
            expected,
            "0.5.x file and an image: open_in_memory reads the 0.5.x file"
        );
        drop(db);
        assert!(
            files(dir.path()) == before,
            "0.5.x file and an image: the read-only open and open_in_memory change nothing"
        );
    }

    // `<p>.pre-0.6` and `<p>.migrating`, no `<p>`: only a read-write open
    // finishes the migration.
    {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("db.grafeo");
        copy(&fixture(), &with_suffix(&path, ".pre-0.6"));
        write_v3(&with_suffix(&path, ".migrating"), &["Vincent"]);
        let before = files(dir.path());
        for (open, error) in [
            ("read-only", read_only_error(&path)),
            ("in-memory", in_memory_error(&path)),
        ] {
            assert!(
                error.contains("read-write open"),
                "kept copy and an image: the {open} error says a read-write open \
                 finishes the migration: {error}"
            );
        }
        assert!(
            files(dir.path()) == before,
            "kept copy and an image: the read-only open and open_in_memory change nothing"
        );
    }

    // `<p>.migrating` only: the error names the image.
    {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("db.grafeo");
        let migrating = with_suffix(&path, ".migrating");
        write_v3(&migrating, &["Vincent"]);
        let before = files(dir.path());
        for (open, error) in [
            ("read-only", read_only_error(&path)),
            ("in-memory", in_memory_error(&path)),
        ] {
            assert!(
                error.contains(&migrating.display().to_string()) && error.contains("missing"),
                "only an image: the {open} error names the image and says the old \
                 database is missing: {error}"
            );
        }
        assert!(
            files(dir.path()) == before,
            "only an image: the read-only open and open_in_memory change nothing"
        );
    }
}

/// A corrupt 0.5.x file fails the migration with an error that says it
/// happened while migrating the file and keeps the cause, and changes nothing.
#[test]
fn a_corrupt_0_5_file_fails_the_migration_with_context() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("db.grafeo");
    copy(
        &Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/released/0.5.44/closed.grafeo"),
        &path,
    );
    // A byte inside the first section (0.5.x section data starts at 16 KiB).
    let mut bytes = std::fs::read(&path).unwrap();
    bytes[16 * 1024 + 19] ^= 0xFF;
    std::fs::write(&path, &bytes).unwrap();
    let before = files(dir.path());

    let error = open_error(&path);
    assert!(
        error.contains("migrating") && error.contains(&path.display().to_string()),
        "the error says it happened while migrating the file: {error}"
    );
    assert!(
        error.contains("CRC mismatch"),
        "the error keeps its cause: {error}"
    );
    assert!(
        files(dir.path()) == before,
        "the failed migration changes nothing"
    );
}

/// Two processes that open the same 0.5.x file at once migrate it once: one
/// migrates while the other waits for the migration lock, then opens the
/// migrated database. The kept copy is the 0.5.x file, byte for byte. This pins
/// `migrate` looking at the files again once it holds the lock: without that,
/// the second process would migrate the 0.6 file and fail on the kept copy.
#[test]
fn two_processes_opening_a_0_5_file_at_once_migrate_it_once() {
    let expected = fixture_contents();
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("db.grafeo");
    copy_fixture(&path);

    let go = dir.path().join("go");
    let openers: Vec<_> = ["first", "second"]
        .into_iter()
        .map(|name| {
            let ready = dir.path().join(format!("{name}-ready"));
            let (child_path, child_go, child_ready) = (path.clone(), go.clone(), ready.clone());
            let opener = std::thread::spawn(move || {
                child_process::output(
                    Command::new(std::env::current_exe().unwrap())
                        .args(["--exact", "open_child", "--nocapture"])
                        .env(PATH_VAR, &child_path)
                        .env(OPEN_GO_VAR, &child_go)
                        .env(LOCK_READY_VAR, &child_ready),
                )
                .unwrap()
            });
            (ready, opener)
        })
        .collect();
    let deadline = Instant::now() + Duration::from_secs(60);
    while !openers.iter().all(|(ready, _)| ready.exists()) {
        assert!(
            Instant::now() < deadline,
            "the child processes never started"
        );
        std::thread::sleep(Duration::from_millis(10));
    }
    std::fs::write(&go, b"go").unwrap();
    for (_, opener) in openers {
        let output = opener.join().unwrap();
        assert!(
            output.status.code() == Some(OPENED_AND_CLOSED),
            "each process opens the database: {:?}\n{}",
            output.status,
            String::from_utf8_lossy(&output.stderr)
        );
    }
    std::fs::remove_file(&go).unwrap();
    for name in ["first", "second"] {
        std::fs::remove_file(dir.path().join(format!("{name}-ready"))).unwrap();
    }
    assert!(
        !with_suffix(&path, ".pre-0.6.pre-0.6").exists(),
        "nothing migrated the kept copy"
    );
    assert_migrated(&path, &expected, &fixture_kept(), "two processes at once");
}

/// Waits for this file before [`open_child`] opens its database.
const OPEN_GO_VAR: &str = "GRAFEO_MIGRATION_OPEN_GO";
/// Exit code of an [`open_child`] that opened and closed its database.
const OPENED_AND_CLOSED: i32 = 5;

/// Child-process entry for [`two_processes_opening_a_0_5_file_at_once_migrate_it_once`];
/// a no-op when run directly. Signals that it runs, waits for the go file, then
/// opens the database read-write and closes it. While the other process has the
/// migrated database open, it finds it locked and tries again.
#[test]
fn open_child() {
    let (Some(path), Some(go), Some(ready)) = (
        std::env::var_os(PATH_VAR),
        std::env::var_os(OPEN_GO_VAR),
        std::env::var_os(LOCK_READY_VAR),
    ) else {
        return;
    };
    let (path, go) = (PathBuf::from(path), PathBuf::from(go));
    std::fs::write(&ready, b"ready").unwrap();
    let deadline = Instant::now() + Duration::from_secs(60);
    while !go.exists() {
        assert!(Instant::now() < deadline, "the go file never came");
        std::thread::sleep(Duration::from_millis(1));
    }
    // The format is given: Windows refuses to read a locked file, so `Auto`
    // could not tell a single file from its first bytes while the other
    // process has it open.
    let config = grafeo_engine::Config::persistent(&path)
        .with_storage_format(grafeo_engine::config::StorageFormat::SingleFile);
    loop {
        match GrafeoDB::with_config(config.clone()) {
            Ok(db) => {
                db.close().unwrap();
                drop(db);
                std::process::exit(OPENED_AND_CLOSED);
            }
            // The other process has the migrated database open (on Windows a
            // read of the locked file fails with "locked a portion").
            Err(error) if error.to_string().contains("lock") && Instant::now() < deadline => {
                std::thread::sleep(Duration::from_millis(20));
            }
            Err(error) => panic!("the open failed: {error}"),
        }
    }
}

/// Marks a [`race_child`] run.
#[cfg(feature = "testing-crash-injection")]
const RACE_VAR: &str = "GRAFEO_MIGRATION_RACE";

/// Starts [`race_child`] on `path`, pausing at `pause_point` with the flag
/// `flag` (see `grafeo_common::testing::pause`). The thread returns the
/// child's output.
#[cfg(feature = "testing-crash-injection")]
fn start_race_child(
    path: &Path,
    pause_point: &str,
    flag: &Path,
) -> std::thread::JoinHandle<std::process::Output> {
    use grafeo_common::testing::pause::{FLAG_VAR, POINT_VAR};

    let (path, pause_point, flag) = (
        path.to_path_buf(),
        pause_point.to_string(),
        flag.to_path_buf(),
    );
    std::thread::spawn(move || {
        child_process::output(
            Command::new(std::env::current_exe().unwrap())
                .args(["--exact", "race_child", "--nocapture"])
                .env(RACE_VAR, "1")
                .env(PATH_VAR, &path)
                .env(POINT_VAR, &pause_point)
                .env(FLAG_VAR, &flag),
        )
        .unwrap()
    })
}

/// Waits until `file` exists; fails if `child` exited first.
#[cfg(feature = "testing-crash-injection")]
fn wait_for(file: &Path, child: &std::thread::JoinHandle<std::process::Output>) {
    let deadline = Instant::now() + Duration::from_secs(60);
    while !file.exists() {
        assert!(
            !child.is_finished(),
            "the child process exited before {} appeared",
            file.display()
        );
        assert!(
            Instant::now() < deadline,
            "{} never appeared",
            file.display()
        );
        std::thread::sleep(Duration::from_millis(5));
    }
}

/// Child-process entry for [`a_create_never_races_a_migration_between_its_renames`];
/// a no-op when run directly. Opens the database read-write with the default
/// format detection (pausing where the environment says), prints its node
/// count, and closes it. While the other process has the database open, it
/// finds it locked and tries again.
#[test]
#[cfg(feature = "testing-crash-injection")]
fn race_child() {
    let (Some(_), Some(path)) = (std::env::var_os(RACE_VAR), std::env::var_os(PATH_VAR)) else {
        return;
    };
    let path = PathBuf::from(path);
    let deadline = Instant::now() + Duration::from_secs(60);
    loop {
        match GrafeoDB::open(&path) {
            Ok(db) => {
                println!("nodes={}", db.node_count());
                db.close().unwrap();
                drop(db);
                std::process::exit(OPENED_AND_CLOSED);
            }
            Err(error) if error.to_string().contains("lock") && Instant::now() < deadline => {
                std::thread::sleep(Duration::from_millis(20));
            }
            Err(error) => panic!("the open failed: {error}"),
        }
    }
}

/// An open that checked `<p>` before a migration in another process moved it
/// never creates a database at `<p>` while that migration is between its
/// renames: it waits for the migration lock and opens the migrated database.
/// Deterministic, with pause points: the opener pauses right after its check
/// (the 0.5.x file is there), the migrator pauses after moving the old files
/// (holding the lock, `<p>` missing), then the opener goes on, then the
/// migrator. Both for a `.grafeo` path and for a path without an extension
/// (where the extension once decided what a missing path became).
///
/// The outcome is decided by the lock and by the checks at the end: an opener
/// that created a database at `<p>` prints its own node count (0, not the
/// fixture's), or makes the migrator's last rename fail (on Windows), so the
/// migrator does not exit cleanly. The 500 ms before the
/// migrator resumes only give such an opener time to show the race early, in
/// the `!path.exists()` check; a correct opener is still waiting for the lock
/// then, however long the pause.
#[test]
#[cfg(feature = "testing-crash-injection")]
fn a_create_never_races_a_migration_between_its_renames() {
    let expected = fixture_contents();
    let expected_nodes = {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("db.grafeo");
        copy_fixture(&path);
        let db = GrafeoDB::open_read_only(&path).unwrap();
        let nodes = db.node_count();
        db.close().unwrap();
        nodes
    };
    for name in ["db.grafeo", "db"] {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join(name);
        copy_fixture(&path);
        let opener_flag = dir.path().join("opener");
        let migrator_flag = dir.path().join("migrator");

        let opener = start_race_child(&path, "open:after_check", &opener_flag);
        wait_for(&with_suffix(&opener_flag, ".paused"), &opener);
        let migrator = start_race_child(&path, "migrate:after_old", &migrator_flag);
        wait_for(&with_suffix(&migrator_flag, ".paused"), &migrator);
        assert!(
            !path.exists() && with_suffix(&path, ".migrating").exists(),
            "{name}: the migrator is between its renames"
        );

        std::fs::write(with_suffix(&opener_flag, ".resume"), b"go").unwrap();
        std::thread::sleep(Duration::from_millis(500));
        assert!(
            !opener.is_finished(),
            "{name}: the opener waits for the migration lock"
        );
        assert!(
            !path.exists(),
            "{name}: the opener created nothing at the path while the migration was \
             between its renames"
        );
        std::fs::write(with_suffix(&migrator_flag, ".resume"), b"go").unwrap();

        for (role, child) in [("migrator", migrator), ("opener", opener)] {
            let output = child.join().unwrap();
            let stdout = String::from_utf8_lossy(&output.stdout);
            assert!(
                output.status.code() == Some(OPENED_AND_CLOSED),
                "{name}: the {role} opens the database: {:?}\n{stdout}\n{}",
                output.status,
                String::from_utf8_lossy(&output.stderr)
            );
            assert!(
                stdout.contains(&format!("nodes={expected_nodes}")),
                "{name}: the {role} opened the migrated database ({expected_nodes} nodes), \
                 not a new one:\n{stdout}"
            );
        }
        assert!(
            path.is_file(),
            "{name}: the database is a file, not a directory"
        );
        assert_migrated(&path, &expected, &fixture_kept(), name);
    }
}

/// Marks a [`read_child`] run, and says how it opens: `read-only` or `in-memory`.
#[cfg(feature = "testing-crash-injection")]
const READ_VAR: &str = "GRAFEO_MIGRATION_READ";
/// Exit code of a [`read_child`] whose open failed.
#[cfg(feature = "testing-crash-injection")]
const READ_FAILED: i32 = 6;

/// Child-process entry for
/// [`a_read_of_a_0_5_database_that_another_process_migrates_meanwhile_fails`];
/// a no-op when run directly. Opens the database read-only or in memory
/// (pausing where the environment says) and prints its node count or the
/// error.
#[test]
#[cfg(feature = "testing-crash-injection")]
fn read_child() {
    let (Ok(mode), Some(path)) = (std::env::var(READ_VAR), std::env::var_os(PATH_VAR)) else {
        return;
    };
    let opened = if mode == "in-memory" {
        GrafeoDB::open_in_memory(&path)
    } else {
        GrafeoDB::open_read_only(&path)
    };
    match opened {
        Ok(db) => {
            println!("nodes={}", db.node_count());
            drop(db);
            std::process::exit(OPENED_AND_CLOSED);
        }
        Err(error) => {
            println!("error={error}");
            std::process::exit(READ_FAILED);
        }
    }
}

/// A read-only open (or `open_in_memory`) of a 0.5.x database takes no lock,
/// so another process can migrate the database while it is read: here the
/// reader pauses right after it listed the WAL files (of the directory, or the
/// sidecar WAL of the file), and this process migrates the database before
/// the reader goes on to read them. The reader then fails with an error that
/// says what happened, instead of opening a database without the WAL's part.
#[test]
#[cfg(feature = "testing-crash-injection")]
fn a_read_of_a_0_5_database_that_another_process_migrates_meanwhile_fails() {
    use grafeo_common::testing::pause::{FLAG_VAR, POINT_VAR};

    for (kept, mode) in [
        (directory_kept(), "read-only"),
        (directory_kept(), "in-memory"),
        (fixture_kept(), "read-only"),
    ] {
        let dir = tempfile::tempdir().unwrap();
        let path = kept.path_in(dir.path());
        arrange_kept(&kept, &path);
        let name = format!("{} {mode}", path.file_name().unwrap().to_string_lossy());
        let flag = dir.path().join("reader");

        let (child_path, child_flag, child_mode) = (path.clone(), flag.clone(), mode.to_string());
        let reader = std::thread::spawn(move || {
            child_process::output(
                Command::new(std::env::current_exe().unwrap())
                    .args(["--exact", "read_child", "--nocapture"])
                    .env(READ_VAR, &child_mode)
                    .env(PATH_VAR, &child_path)
                    .env(POINT_VAR, "wal_recovery:after_listing")
                    .env(FLAG_VAR, &child_flag),
            )
            .unwrap()
        });
        wait_for(&with_suffix(&flag, ".paused"), &reader);

        let db = GrafeoDB::open(&path).unwrap_or_else(|error| panic!("{name}: {error}"));
        db.close().unwrap();
        drop(db);
        assert!(
            path.is_file(),
            "{name}: the database was migrated meanwhile"
        );
        std::fs::write(with_suffix(&flag, ".resume"), b"go").unwrap();

        let output = reader.join().unwrap();
        let stdout = String::from_utf8_lossy(&output.stdout);
        assert!(
            output.status.code() == Some(READ_FAILED)
                && stdout.contains(
                    "was migrated to the 0.6 format by another process while it was read"
                ),
            "{name}: the reader fails, saying the database was migrated while it was read: \
             {:?}\n{stdout}\n{}",
            output.status,
            String::from_utf8_lossy(&output.stderr)
        );
    }
}

/// An I/O error while migrating (here injected while the image is written)
/// keeps its variant and kind, carries one error code, and says it happened
/// while migrating the file; nothing changes.
#[test]
#[cfg(feature = "testing-crash-injection")]
fn an_io_error_while_migrating_keeps_its_variant_and_says_where() {
    use grafeo_common::testing::crash::with_failure_at;
    use grafeo_common::utils::error::Error;

    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("db.grafeo");
    copy_fixture(&path);
    let before = files(dir.path());

    let Err(error) = with_failure_at(1, || GrafeoDB::open(&path)) else {
        panic!("the migration succeeded despite the injected failure");
    };
    match &error {
        Error::Io(io) => assert_eq!(
            io.kind(),
            std::io::ErrorKind::Other,
            "the I/O error keeps its kind"
        ),
        other => panic!("the I/O error stays an I/O error, got {other:?}"),
    }
    let message = error.to_string();
    assert!(
        message.contains("migrating")
            && message.contains(&path.display().to_string())
            && message.contains("injected failure at: checkpoint:after_chunks"),
        "the error says where and keeps its cause: {message}"
    );
    assert_eq!(
        message.matches("GRAFEO-").count(),
        1,
        "the error carries one code: {message}"
    );
    assert!(
        files(dir.path()) == before,
        "the failed migration changes nothing"
    );
}

/// A migration never replaces a kept copy: with `<p>.pre-0.6` already there (for
/// example a 0.5.x file restored from an earlier migration's copy), the open
/// fails before it writes anything.
#[test]
fn an_existing_kept_copy_is_never_replaced() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("db.grafeo");
    copy_fixture(&path);
    let kept = with_suffix(&path, ".pre-0.6");
    write_v3(&kept, &["Gus"]);
    let before = files(dir.path());
    let error = open_error(&path);
    assert!(
        error.contains(&kept.display().to_string()),
        "the error names the kept copy: {error}"
    );
    assert!(
        files(dir.path()) == before,
        "the failed migration changes nothing"
    );
}

/// A checkpoint image a 0.5.44 checkpoint left pending (`<p>.checkpoint`) holds
/// the database: the migration reads it, and keeps it as
/// `<p>.pre-0.6.checkpoint` next to the kept file, byte for byte.
#[test]
fn a_pending_0_5_44_checkpoint_image_is_migrated_and_kept() {
    let pending = pending_kept();
    let file_only = Kept {
        checkpoint: None,
        ..pending_kept()
    };
    // `SHOW INDEXES` tells the two apart: 0.5.43 files hold no index
    // definitions. A read-only database refuses it, so it runs on a copy.
    let indexes = |db: &GrafeoDB| {
        let copy = db.is_read_only().then(|| db.to_memory().unwrap());
        rows(copy.as_ref().unwrap_or(db), "SHOW INDEXES")
            .into_iter()
            .map(|row| row[0].clone())
            .collect::<Vec<Value>>()
    };
    let read_only_indexes = |kept: &Kept| {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("db.grafeo");
        arrange_kept(kept, &path);
        let db = GrafeoDB::open_read_only(&path).unwrap();
        let found = indexes(&db);
        db.close().unwrap();
        found
    };
    let expected_indexes = read_only_indexes(&pending);
    assert_ne!(
        expected_indexes,
        read_only_indexes(&file_only),
        "the pending image and the file hold different indexes"
    );

    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("db.grafeo");
    arrange_kept(&pending, &path);
    let db = GrafeoDB::open(&path).unwrap();
    assert_eq!(
        indexes(&db),
        expected_indexes,
        "the migrated database holds the pending image's indexes, not the file's"
    );
    db.close().unwrap();
    drop(db);
    assert_migrated(
        &path,
        &kept_contents(&pending),
        &pending,
        "a pending checkpoint image",
    );
}

/// While another handle holds the 0.5.x file locked for writing, as a 0.5.x
/// process does, the migration fails with a "locked" error and changes nothing.
#[test]
fn a_0_5_writer_holding_the_file_makes_the_migration_fail() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("db.grafeo");
    copy_fixture(&path);
    let before = files(dir.path());

    let writer = std::fs::OpenOptions::new()
        .read(true)
        .write(true)
        .open(&path)
        .unwrap();
    writer.lock().unwrap();
    // The format is given: Windows refuses to read a locked file, so `Auto`
    // could not tell a single file from its first bytes.
    let opened = GrafeoDB::with_config(
        grafeo_engine::Config::persistent(&path)
            .with_storage_format(grafeo_engine::config::StorageFormat::SingleFile),
    );
    writer.unlock().unwrap();
    drop(writer);

    let error = match opened {
        Ok(_) => panic!("the migration ran while a writer held the file"),
        Err(error) => error.to_string(),
    };
    assert!(
        error.contains("locked"),
        "the error says the file is locked: {error}"
    );
    assert!(
        files(dir.path()) == before,
        "the failed migration changes nothing"
    );
}

// =========================================================================
// WAL directories
// =========================================================================

/// Copies the WAL-directory fixture to a new directory; returns it and the
/// path of the copy.
fn copy_directory() -> (tempfile::TempDir, PathBuf) {
    let dir = tempfile::tempdir().unwrap();
    let path = directory_kept().path_in(dir.path());
    copy(&directory(), &path);
    (dir, path)
}

/// A read-write open of a 0.5.x WAL directory migrates it to a single file at
/// the same path, whatever the configured storage format: an existing path
/// opens by what it is. The deprecated `WalDirectory` format included.
#[test]
#[allow(
    deprecated,
    reason = "pins what the deprecated `StorageFormat::WalDirectory` still does until 0.7.0"
)]
fn a_wal_directory_migrates_whatever_the_configured_format() {
    use grafeo_engine::config::StorageFormat;

    let expected = kept_contents(&directory_kept());
    for format in [
        StorageFormat::Auto,
        StorageFormat::SingleFile,
        StorageFormat::WalDirectory,
    ] {
        let (_dir, path) = copy_directory();
        let db = GrafeoDB::with_config(
            grafeo_engine::Config::persistent(&path).with_storage_format(format),
        )
        .unwrap_or_else(|error| panic!("{format}: {error}"));
        assert_eq!(contents(&db), expected, "{format}: the migrated data");
        db.close().unwrap();
        drop(db);
        assert!(path.is_file(), "{format}: the database is a single file");
        assert_migrated(&path, &expected, &directory_kept(), &format.to_string());
    }
}

/// Each state a crash can leave in the migration of a WAL directory is
/// finished by the next read-write open, as for a file.
#[test]
fn interrupted_directory_migrations_resume_from_the_files_present() {
    let expected = kept_contents(&directory_kept());

    // `<p>/` (0.5.x) and `<p>.migrating`: the image may be incomplete, so it
    // goes and the migration runs again.
    {
        let (_dir, path) = copy_directory();
        write_v3(&with_suffix(&path, ".migrating"), &["Vincent"]);
        assert_migrated(
            &path,
            &expected,
            &directory_kept(),
            "0.5.x directory and an image",
        );
    }

    // `<p>.pre-0.6/` and `<p>.migrating`, no `<p>`: the directory was moved
    // after the image was complete, so the image goes into place.
    {
        let dir = tempfile::tempdir().unwrap();
        let path = directory_kept().path_in(dir.path());
        copy(&directory(), &with_suffix(&path, ".pre-0.6"));
        write_v3(&with_suffix(&path, ".migrating"), &["Vincent", "Mia"]);
        let db = GrafeoDB::open(&path).unwrap();
        assert_eq!(
            people(&db),
            names(&["Mia", "Vincent"]),
            "kept directory and an image: the database is the image"
        );
        db.close().unwrap();
        drop(db);
        assert_kept(&path, &directory_kept(), "kept directory and an image");
        assert_no_leftovers(&path, "kept directory and an image");
    }
}

/// A read-only open (and `open_in_memory`) never finishes a half-migrated WAL
/// directory: it reads a directory still in place (next to a stale image) as
/// it is, and refuses one already moved to `<p>.pre-0.6/`, saying a read-write
/// open finishes the migration. Neither changes a file.
#[test]
fn a_read_only_open_never_finishes_a_directory_migration() {
    let expected = kept_contents(&directory_kept());

    {
        let (dir, path) = copy_directory();
        write_v3(&with_suffix(&path, ".migrating"), &["Vincent"]);
        let before = files(dir.path());
        let db = GrafeoDB::open_read_only(&path).unwrap();
        assert_eq!(
            contents(&db),
            expected,
            "0.5.x directory and an image: the read-only open reads the directory"
        );
        db.close().unwrap();
        drop(db);
        let db = GrafeoDB::open_in_memory(&path).unwrap();
        assert_eq!(
            contents(&db),
            expected,
            "0.5.x directory and an image: open_in_memory reads the directory"
        );
        drop(db);
        assert!(
            files(dir.path()) == before,
            "0.5.x directory and an image: the read-only open and open_in_memory change nothing"
        );
    }

    {
        let dir = tempfile::tempdir().unwrap();
        let path = directory_kept().path_in(dir.path());
        copy(&directory(), &with_suffix(&path, ".pre-0.6"));
        write_v3(&with_suffix(&path, ".migrating"), &["Vincent"]);
        let before = files(dir.path());
        for (open, error) in [
            ("read-only", read_only_error(&path)),
            ("in-memory", in_memory_error(&path)),
        ] {
            assert!(
                error.contains("read-write open"),
                "kept directory and an image: the {open} error says a read-write open \
                 finishes the migration: {error}"
            );
        }
        assert!(
            files(dir.path()) == before,
            "kept directory and an image: the read-only open and open_in_memory change nothing"
        );
    }
}

/// A migration never replaces a kept copy, also for a WAL directory: with
/// `<p>.pre-0.6` already there, the open fails before it writes anything.
#[test]
fn an_existing_kept_copy_is_never_replaced_by_a_directory() {
    let (dir, path) = copy_directory();
    let kept = with_suffix(&path, ".pre-0.6");
    write_v3(&kept, &["Gus"]);
    let before = files(dir.path());
    let error = open_error(&path);
    assert!(
        error.contains(&kept.display().to_string()),
        "the error names the kept copy: {error}"
    );
    assert!(
        files(dir.path()) == before,
        "the failed migration changes nothing"
    );
}

/// While a 0.5.44 process has a WAL directory open for writing, it holds the
/// directory's `LOCK` file: the migration fails with its "locked" error and
/// changes nothing. A read-only open takes no lock (there is no shared one),
/// so it reads the directory meanwhile.
#[test]
fn a_0_5_44_writer_holding_the_directory_makes_the_migration_fail() {
    let expected = kept_contents(&directory_kept());
    let (dir, path) = copy_directory();
    let before = files(dir.path());

    let writer = grafeo_storage::lock::DirectoryLock::acquire(&path).unwrap();
    let opened = GrafeoDB::open(&path);
    let read = GrafeoDB::open_read_only(&path).map(|db| {
        let found = contents(&db);
        db.close().unwrap();
        found
    });
    drop(writer);

    let error = match opened {
        Ok(_) => panic!("the migration ran while a 0.5.44 writer held the directory"),
        Err(error) => error.to_string(),
    };
    assert!(
        error.contains("locked") && error.contains(&path.display().to_string()),
        "the error says the directory is locked, and which: {error}"
    );
    assert_eq!(
        read.unwrap_or_else(|error| panic!("the read-only open failed: {error}")),
        expected,
        "the read-only open reads the directory while the writer holds it"
    );
    assert!(
        files(dir.path()) == before,
        "the failed migration changes nothing"
    );
}

/// A directory that is not a 0.5.x database (no `wal/` inside) is never
/// migrated: every open fails with an error naming it, and nothing in it or
/// next to it changes.
#[test]
fn a_directory_that_is_not_a_database_is_refused_and_untouched() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("db");
    std::fs::create_dir(&path).unwrap();
    std::fs::write(path.join("notes.txt"), "Vincent and Mia went to Paris").unwrap();
    let before = files(dir.path());

    for (open, error) in [
        ("read-write", open_error(&path)),
        ("read-only", read_only_error(&path)),
        ("in-memory", in_memory_error(&path)),
    ] {
        assert!(
            error.contains(&path.display().to_string()) && error.contains("not a 0.5.x"),
            "the {open} error names the directory and says it is not a 0.5.x database: {error}"
        );
        assert!(
            files(dir.path()) == before,
            "the refused {open} open changes nothing"
        );
    }
}

/// A 0.5.x directory whose `wal/` exists but cannot be listed is never read as
/// empty: the read-write open fails (naming the directory) before anything
/// moves, so no empty database replaces it, and the read-only open and
/// `open_in_memory` fail too.
#[test]
fn a_directory_whose_wal_cannot_be_listed_is_refused_and_untouched() {
    let (dir, path) = copy_directory();
    let before = files(dir.path());
    let outcomes = {
        let Some(_denied) = unlistable::Unlistable::deny(&path.join("wal")) else {
            return;
        };
        [
            ("read-write", GrafeoDB::open(&path).map(drop)),
            ("read-only", GrafeoDB::open_read_only(&path).map(drop)),
            ("in-memory", GrafeoDB::open_in_memory(&path).map(drop)),
        ]
    };
    for (open, outcome) in outcomes {
        let error = match outcome {
            Ok(()) => panic!("the {open} open read a WAL it cannot list"),
            Err(error) => error.to_string(),
        };
        assert!(
            error.contains(&path.join("wal").display().to_string()),
            "the {open} error names the WAL directory: {error}"
        );
    }
    assert!(path.is_dir(), "the 0.5.x directory is still the database");
    assert!(
        files(dir.path()) == before,
        "the refused opens changed nothing: no image, no kept copy"
    );
}

/// A directory path written with a trailing separator (`<p>/`, and `<p>\` on
/// Windows), a natural way to name a WAL directory, migrates as `<p>`: the
/// single file at `<p>` and the kept directory `<p>.pre-0.6/` next to it, never
/// side files inside the directory being moved. A read-only open through such
/// a path reads it in place.
#[test]
fn a_directory_path_with_a_trailing_separator_migrates_as_the_directory() {
    let expected = kept_contents(&directory_kept());
    let mut separators = vec!["/"];
    if cfg!(windows) {
        separators.push("\\");
    }
    for separator in separators {
        let (_dir, path) = copy_directory();
        let written = with_suffix(&path, separator);

        let db = GrafeoDB::open_read_only(&written)
            .unwrap_or_else(|error| panic!("{separator}: read-only: {error}"));
        assert_eq!(contents(&db), expected, "{separator}: read in place");
        db.close().unwrap();
        drop(db);

        let db = GrafeoDB::open(&written).unwrap_or_else(|error| panic!("{separator}: {error}"));
        assert_eq!(contents(&db), expected, "{separator}: the migrated data");
        // Compared as strings: `Path` equality ignores a trailing separator.
        assert_eq!(
            db.path().map(Path::as_os_str),
            Some(written.as_os_str()),
            "{separator}: the database reports the path as it was given"
        );
        // The path as given no longer exists once `<p>` is a file (`<p>/` is
        // not a directory): disk usage goes by the database's files.
        let file_bytes = usize::try_from(std::fs::metadata(&path).unwrap().len()).unwrap();
        let disk_bytes = db.detailed_stats().disk_bytes;
        assert!(
            disk_bytes.is_some_and(|bytes| bytes >= file_bytes),
            "{separator}: disk usage {disk_bytes:?} counts the migrated file ({file_bytes} bytes)"
        );
        db.close().unwrap();
        drop(db);
        assert!(path.is_file(), "{separator}: <p> is the migrated file");
        assert_migrated(&path, &expected, &directory_kept(), separator);
    }
}

/// A path that ends in `..` names the directory it leads to: `<p>/wal/..`
/// migrates `<p>`, with its side files next to it, never inside it.
#[test]
fn a_directory_path_ending_in_parent_migrates_the_directory_it_names() {
    let expected = kept_contents(&directory_kept());
    let (_dir, path) = copy_directory();
    let written = path.join("wal").join("..");
    let db = GrafeoDB::open(&written).unwrap_or_else(|error| panic!("{error}"));
    assert_eq!(contents(&db), expected, "the migrated data");
    db.close().unwrap();
    drop(db);
    assert!(path.is_file(), "<p> is the migrated file");
    assert_migrated(&path, &expected, &directory_kept(), "<p>/wal/..");
}

/// Marks a [`current_directory_child`] run.
const CURRENT_DIRECTORY_VAR: &str = "GRAFEO_MIGRATION_CURRENT_DIRECTORY";

/// Child-process entry for
/// [`a_directory_that_is_the_current_directory_is_read_but_not_migrated`]; a
/// no-op when run directly. Runs with the 0.5.x directory as its current
/// directory, opens `.` read-write (printing the error) and read-only
/// (printing the node count).
#[test]
fn current_directory_child() {
    if std::env::var_os(CURRENT_DIRECTORY_VAR).is_none() {
        return;
    }
    match GrafeoDB::open(".") {
        Ok(_) => println!("read-write=opened"),
        Err(error) => println!("read-write={error}"),
    }
    match GrafeoDB::open_read_only(".") {
        Ok(db) => println!("read-only={}", db.node_count()),
        Err(error) => println!("read-only={error}"),
    }
}

/// A 0.5.x directory opened as `.` from inside it (as `grafeo info .` does) is
/// read in place by a read-only open. A read-write open refuses to migrate it:
/// it is the process's current directory, which cannot move (on Windows not
/// at all, elsewhere the process would go on inside the kept copy). The error
/// says to open it by its name from outside, nothing changes, and no side file
/// goes inside the directory.
#[test]
fn a_directory_that_is_the_current_directory_is_read_but_not_migrated() {
    let expected_nodes = {
        let db = GrafeoDB::open_read_only(directory()).unwrap();
        let nodes = db.node_count();
        db.close().unwrap();
        nodes
    };
    let (dir, path) = copy_directory();
    let before = files(dir.path());
    let output = child_process::output(
        Command::new(std::env::current_exe().unwrap())
            .args(["--exact", "current_directory_child", "--nocapture"])
            .env(CURRENT_DIRECTORY_VAR, "1")
            .current_dir(&path),
    )
    .unwrap();
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        output.status.success(),
        "the child ran: {stdout}\n{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let read_write = stdout
        .lines()
        .find_map(|line| line.strip_prefix("read-write="))
        .unwrap_or_else(|| panic!("the child printed no read-write outcome:\n{stdout}"));
    assert!(
        read_write.contains("current directory")
            && read_write.contains(&format!("{}db", std::path::MAIN_SEPARATOR)),
        "the read-write open refuses, naming the directory and why: {read_write}"
    );
    assert!(
        stdout.contains(&format!("read-only={expected_nodes}")),
        "the read-only open reads the directory in place:\n{stdout}"
    );
    assert!(
        files(dir.path()) == before,
        "nothing changed, inside the directory or next to it"
    );
}

/// On Windows a directory cannot be renamed while any process has a file in it
/// open. The migration then fails before anything moved, and its error says
/// why and what to do (not only "access is denied"). Elsewhere an open file
/// does not keep the directory from moving, and the migration succeeds.
#[test]
fn a_directory_that_cannot_move_says_why() {
    let expected = kept_contents(&directory_kept());
    let (dir, path) = copy_directory();
    let before = files(dir.path());
    let reader = std::fs::File::open(path.join("wal/wal_00000000.log")).unwrap();
    let opened = GrafeoDB::open(&path);
    if cfg!(windows) {
        drop(reader);
        let error = match opened {
            Ok(_) => panic!("a directory with an open file in it was renamed on Windows"),
            Err(error) => error.to_string(),
        };
        assert!(
            error.contains("another process may have a file in it open")
                && error.contains("a directory of its own"),
            "the error says why the directory cannot move and what to do: {error}"
        );
        assert_eq!(
            error.matches("GRAFEO-").count(),
            1,
            "the error carries one code: {error}"
        );
        assert!(
            files(dir.path()) == before,
            "the failed migration changes nothing"
        );
    } else {
        let db = opened.unwrap();
        drop(reader);
        assert_eq!(contents(&db), expected, "the open file did not keep it");
        db.close().unwrap();
        drop(db);
        assert_migrated(&path, &expected, &directory_kept(), "an open file inside");
    }
}

// =========================================================================
// What a migration logs
// =========================================================================

/// The info events `f` emits on this thread, as `tracing` delivers them.
///
/// The subscriber is the global default of this test binary and keeps the
/// events of the threads that ask for them. A scoped subscriber
/// (`with_default`) is not reliable while other tests run: as long as it is
/// the only one registered, `tracing` decides whether a log statement is
/// enabled from the thread that reaches it first, which may be another
/// test's, and remembers "never".
#[cfg(feature = "tracing")]
fn info_events(f: impl FnOnce()) -> Vec<String> {
    use std::cell::RefCell;
    use std::sync::Once;

    use tracing::field::{Field, Visit};
    use tracing::span::{Attributes, Id, Record};
    use tracing::{Event, Level, Metadata, Subscriber};

    thread_local! {
        /// The messages of this thread's info events, while it captures them.
        static CAPTURED: RefCell<Option<Vec<String>>> = const { RefCell::new(None) };
    }

    /// Keeps the message of every info event of a capturing thread.
    struct Capture;

    /// Reads the message field of an event.
    struct Message(String);

    impl Visit for Message {
        fn record_debug(&mut self, field: &Field, value: &dyn std::fmt::Debug) {
            if field.name() == "message" {
                self.0 = format!("{value:?}");
            }
        }
    }

    impl Subscriber for Capture {
        fn enabled(&self, metadata: &Metadata<'_>) -> bool {
            *metadata.level() == Level::INFO
        }

        fn new_span(&self, _span: &Attributes<'_>) -> Id {
            Id::from_u64(1)
        }

        fn record(&self, _span: &Id, _values: &Record<'_>) {}

        fn record_follows_from(&self, _span: &Id, _follows: &Id) {}

        fn event(&self, event: &Event<'_>) {
            CAPTURED.with(|captured| {
                if let Some(events) = captured.borrow_mut().as_mut() {
                    let mut message = Message(String::new());
                    event.record(&mut message);
                    events.push(message.0);
                }
            });
        }

        fn enter(&self, _span: &Id) {}

        fn exit(&self, _span: &Id) {}
    }

    static INSTALL: Once = Once::new();
    INSTALL.call_once(|| {
        tracing::subscriber::set_global_default(Capture)
            .expect("no other global subscriber in this test binary");
    });
    CAPTURED.with(|captured| *captured.borrow_mut() = Some(Vec::new()));
    f();
    CAPTURED.with(|captured| captured.borrow_mut().take().unwrap_or_default())
}

/// A migration logs the counts of the 0.5.x database before it writes the
/// image, and the counts the written image records once it is in place.
#[cfg(feature = "tracing")]
#[test]
fn a_migration_logs_the_counts_before_and_after() {
    let expected_nodes = {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("db.grafeo");
        copy_fixture(&path);
        let db = GrafeoDB::open_read_only(&path).unwrap();
        let nodes = db.node_count();
        db.close().unwrap();
        nodes
    };
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("db.grafeo");
    copy_fixture(&path);

    let events = info_events(|| {
        let db = GrafeoDB::open(&path).unwrap();
        db.close().unwrap();
    });
    let nodes = format!("{expected_nodes} nodes");
    assert!(
        events
            .iter()
            .any(|event| event.starts_with("migrating") && event.contains(&nodes)),
        "the counts before the migration ({nodes}) are logged: {events:#?}"
    );
    assert!(
        events
            .iter()
            .any(|event| event.starts_with("migrated") && event.contains(&nodes)),
        "the counts of the migrated file ({nodes}) are logged: {events:#?}"
    );
}

/// A read-only open of a 0.5.x file says that a read-write open will migrate
/// it, and that it is read in place meanwhile.
#[cfg(feature = "tracing")]
#[test]
fn a_read_only_open_of_a_0_5_file_logs_that_it_will_be_migrated() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("db.grafeo");
    copy_fixture(&path);

    let events = info_events(|| {
        let db = GrafeoDB::open_read_only(&path).unwrap();
        db.close().unwrap();
    });
    assert!(
        events
            .iter()
            .any(|event| event.contains(&path.display().to_string())
                && event.contains("a read-write open migrates it")),
        "the read-only open says the file will be migrated: {events:#?}"
    );
    assert!(
        !with_suffix(&path, ".pre-0.6").exists(),
        "the read-only open migrated nothing"
    );
}

// =========================================================================
// The migration lock
// =========================================================================

const LOCK_READY_VAR: &str = "GRAFEO_MIGRATION_LOCK_READY";
const LOCK_RELEASE_VAR: &str = "GRAFEO_MIGRATION_LOCK_RELEASE";
const LOCK_INSTALL_VAR: &str = "GRAFEO_MIGRATION_LOCK_INSTALL";

/// What the child process of [`hold_migrate_lock`] stands in for.
#[derive(Clone, Copy, PartialEq, Eq)]
enum LockMode {
    /// A migration that has not changed any file yet: before it lets go, it
    /// checks that the 0.5.x file is still in place and nothing was migrated.
    Hold,
    /// A migration between its renames (the old file moved, the image at
    /// `<path>.migrating`): before it lets go, it checks that `<path>` is still
    /// missing (nothing created it meanwhile) and renames the image into place.
    Install,
}

/// A child process holding `<path>.migrate.lock`, see [`hold_migrate_lock`].
struct LockHolder {
    /// Ends when the child has exited.
    thread: std::thread::JoinHandle<()>,
    /// The child lets go of the lock once this file exists.
    release: PathBuf,
}

impl LockHolder {
    /// Lets the child go on: it checks the files and exits, which releases the
    /// lock.
    fn release(&self) {
        std::fs::write(&self.release, b"release").unwrap();
    }

    /// Releases the lock after `delay`, from another thread, so this one can
    /// wait for the lock meanwhile.
    fn release_after(&self, delay: Duration) -> std::thread::JoinHandle<()> {
        let release = self.release.clone();
        std::thread::spawn(move || {
            std::thread::sleep(delay);
            std::fs::write(&release, b"release").unwrap();
        })
    }

    /// Waits until the child has exited, and fails if it failed.
    fn join(self) {
        self.thread.join().unwrap();
    }
}

/// Holds `<path>.migrate.lock` from a child process, like a migration running
/// in another process, until [`LockHolder::release`] (so how long it holds the
/// lock never depends on the runner's speed). Returns once the child holds the
/// lock.
fn hold_migrate_lock(path: &Path, mode: LockMode) -> LockHolder {
    let ready = path.parent().unwrap().join("lock-holder-ready");
    let release = path.parent().unwrap().join("lock-holder-release");
    let child_path = path.to_path_buf();
    let (child_ready, child_release) = (ready.clone(), release.clone());
    let thread = std::thread::spawn(move || {
        let mut command = Command::new(std::env::current_exe().unwrap());
        command
            .args(["--exact", "lock_holder_child", "--nocapture"])
            .env(PATH_VAR, &child_path)
            .env(LOCK_READY_VAR, &child_ready)
            .env(LOCK_RELEASE_VAR, &child_release);
        if mode == LockMode::Install {
            command.env(LOCK_INSTALL_VAR, "1");
        }
        let status = child_process::run(&mut command).unwrap();
        assert!(status.success(), "the lock holder failed with {status}");
    });
    let deadline = Instant::now() + Duration::from_secs(60);
    while !ready.exists() {
        assert!(
            Instant::now() < deadline && !thread.is_finished(),
            "the child process never took the lock"
        );
        std::thread::sleep(Duration::from_millis(10));
    }
    LockHolder { thread, release }
}

/// Child-process entry for [`hold_migrate_lock`]; a no-op when run directly.
/// Before it lets go, it checks that nothing changed the files meanwhile (see
/// [`LockMode`]).
#[test]
fn lock_holder_child() {
    let (Some(path), Some(ready), Some(release)) = (
        std::env::var_os(PATH_VAR),
        std::env::var_os(LOCK_READY_VAR),
        std::env::var_os(LOCK_RELEASE_VAR),
    ) else {
        return;
    };
    let (path, release) = (PathBuf::from(path), PathBuf::from(release));
    let lock = std::fs::OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .truncate(false)
        .open(with_suffix(&path, ".migrate.lock"))
        .unwrap();
    lock.lock().unwrap();
    std::fs::write(&ready, b"locked").unwrap();
    let deadline = Instant::now() + Duration::from_secs(60);
    while !release.exists() {
        assert!(Instant::now() < deadline, "the release file never came");
        std::thread::sleep(Duration::from_millis(5));
    }
    if std::env::var_os(LOCK_INSTALL_VAR).is_some() {
        assert!(
            !path.exists(),
            "nothing creates the database while a migration is between its renames"
        );
        std::fs::rename(with_suffix(&path, ".migrating"), &path).unwrap();
        // Exits holding the lock until here and leaves the lock file.
        return;
    }
    assert_eq!(
        detect(&path).unwrap(),
        OnDisk::LegacyFile,
        "nothing migrates the database while another process holds the lock"
    );
    assert!(
        !with_suffix(&path, ".migrating").exists() && !with_suffix(&path, ".pre-0.6").exists(),
        "no migration starts while another process holds the lock"
    );
    // Exits holding the lock until here and leaves the lock file, as a
    // migration that crashed would.
}

/// A read-write open waits while another process holds the migration lock, and
/// migrates once it is released.
#[test]
fn a_second_process_waits_for_the_migration() {
    let expected = fixture_contents();
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("db.grafeo");
    copy_fixture(&path);

    let hold = Duration::from_secs(1);
    let holder = hold_migrate_lock(&path, LockMode::Hold);
    let started = Instant::now();
    let releaser = holder.release_after(hold);
    let db = GrafeoDB::open(&path)
        .unwrap_or_else(|error| panic!("the open after the lock was released: {error}"));
    let waited = started.elapsed();
    releaser.join().unwrap();
    holder.join();
    assert!(
        waited >= hold,
        "the open waited until the lock was released after {hold:?}, it took only {waited:?}"
    );
    assert_eq!(
        contents(&db),
        expected,
        "the database holds the fixture's data"
    );
    db.close().unwrap();
    drop(db);
    assert_migrated(
        &path,
        &expected,
        &fixture_kept(),
        "after waiting for the lock",
    );
}

/// A read-write open gives up after waiting about five seconds for the
/// migration lock, with an error that says a migration is running, and changes
/// nothing; once the lock is free, the open migrates. The child holds the lock
/// until the open has given up.
#[test]
fn a_migration_lock_held_too_long_fails_the_open() {
    let expected = fixture_contents();
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("db.grafeo");
    copy_fixture(&path);

    let holder = hold_migrate_lock(&path, LockMode::Hold);
    let before = files(dir.path());
    let started = Instant::now();
    let error = open_error(&path);
    let waited = started.elapsed();
    assert!(
        error.contains("migration is running"),
        "the error says a migration is running: {error}"
    );
    assert!(
        waited >= Duration::from_secs(5),
        "the open waited five seconds before it gave up, not {waited:?}"
    );
    assert!(
        files(dir.path()) == before,
        "the open that gave up changes nothing"
    );
    holder.release();
    holder.join();

    assert_migrated(
        &path,
        &expected,
        &fixture_kept(),
        "after the lock holder exited",
    );
}
