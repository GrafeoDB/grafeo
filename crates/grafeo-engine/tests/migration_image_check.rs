//! The migration of a 0.5.x database checks its 0.6 image against the old
//! database before it moves anything.
//!
//! `migration::migrate` writes the image of the old database to
//! `<path>.migrating`, reads the image's header back, and used to only log
//! its node and edge counts: they were never compared with the old
//! database's, so an image that lost rows (a section writer that dropped
//! some) would be installed, with only the kept `.pre-0.6` copy left to limit
//! the damage. A migration whose image holds other node or edge counts than
//! the old database fails, names the difference, removes the image and
//! leaves the old files as they were: no `.pre-0.6` copy, no rename.
//!
//! Nothing in a sound build writes such an image, so the test changes the
//! image under a running migration, with a pause point (see
//! `grafeo_common::testing::pause`) that the check needs:
//!
//! - **`migrate:image_written`** (to add): in `migration::migrate`, right
//!   after `old.write_image_with(&migrating, &keys)` has written and closed
//!   the image, before the image is read back for the check. The test
//!   replaces `<path>.migrating` there with an image holding one node (or
//!   one edge) fewer, then lets the migration go on.
//!
//! Until that point exists the migration never pauses, and
//! [`a_migration_whose_image_lost_rows_fails_and_keeps_the_old_file`] fails
//! saying so (and, when the migration then installed its image, that too).
//!
//! ```bash
//! cargo test -p grafeo-engine --features full,testing-crash-injection \
//!     --test migration_image_check
//! ```

#![cfg(all(
    feature = "gql",
    feature = "wal",
    feature = "grafeo-file",
    feature = "triple-store",
    feature = "vector-index",
    feature = "text-index",
    not(miri)
))]

use std::path::{Path, PathBuf};

use grafeo_engine::GrafeoDB;
use grafeo_storage::file::detect::{OnDisk, detect};

/// The 0.5.44 database closed cleanly: a container v2 file, no sidecar WAL.
fn fixture() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/released/0.5.44/closed.grafeo")
}

/// `<path><suffix>`, next to the database file.
fn with_suffix(path: &Path, suffix: &str) -> PathBuf {
    let mut name = path.as_os_str().to_owned();
    name.push(suffix);
    PathBuf::from(name)
}

/// The node and edge counts of the database at `path`, read-only.
fn counts(path: &Path) -> (usize, usize) {
    let db = GrafeoDB::open_read_only(path).unwrap();
    let counts = (db.node_count().unwrap(), db.edge_count().unwrap());
    db.close().unwrap();
    counts
}

/// The names of the files and directories in `dir`, sorted, without the
/// spill directories 0.6 creates next to a database (scratch space).
#[cfg(feature = "testing-crash-injection")]
fn names(dir: &Path) -> Vec<String> {
    let mut names: Vec<String> = std::fs::read_dir(dir)
        .unwrap()
        .map(|entry| entry.unwrap().path())
        .filter(|path| {
            path.extension()
                .is_none_or(|extension| extension != "spill")
        })
        .map(|path| path.file_name().unwrap().to_string_lossy().into_owned())
        .collect();
    names.sort();
    names
}

/// Control: the fixture migrates with its data: the migrated file holds the
/// node and edge counts of the old database, and the old file is kept as
/// `<path>.pre-0.6` byte for byte.
#[test]
fn a_sound_migration_keeps_the_counts_of_the_old_database() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("db.grafeo");
    std::fs::copy(fixture(), &path).unwrap();
    let before = std::fs::read(&path).unwrap();
    let old = counts(&path);
    assert!(old.0 > 1 && old.1 > 1, "the fixture holds data: {old:?}");

    let db = GrafeoDB::open(&path).unwrap();
    assert_eq!(
        (db.node_count().unwrap(), db.edge_count().unwrap()),
        old,
        "the migrated database"
    );
    db.close().unwrap();
    drop(db);
    assert_eq!(detect(&path).unwrap(), OnDisk::Current);
    assert_eq!(
        std::fs::read(with_suffix(&path, ".pre-0.6")).unwrap(),
        before,
        "the 0.5.x file is kept as it was"
    );
    assert_eq!(counts(&path), old, "the migrated file reopened");
}

/// The whole numbers in `text`, as written.
#[cfg(feature = "testing-crash-injection")]
fn numbers(text: &str) -> Vec<String> {
    text.split(|c: char| !c.is_ascii_digit())
        .filter(|part| !part.is_empty())
        .map(str::to_string)
        .collect()
}

/// The pause point the check needs (see the module docs).
#[cfg(feature = "testing-crash-injection")]
const IMAGE_WRITTEN: &str = "migrate:image_written";

/// Marks an [`image_check_child`] run, and names the database it opens.
#[cfg(feature = "testing-crash-injection")]
const PATH_VAR: &str = "GRAFEO_IMAGE_CHECK_PATH";
/// Exit code of an [`image_check_child`] whose open succeeded.
#[cfg(feature = "testing-crash-injection")]
const OPENED: i32 = 5;
/// Exit code of an [`image_check_child`] whose open failed.
#[cfg(feature = "testing-crash-injection")]
const FAILED: i32 = 6;

/// Child-process entry for
/// [`a_migration_whose_image_lost_rows_fails_and_keeps_the_old_file`]; a
/// no-op when run directly. Opens the database read-write (pausing where the
/// environment says) and prints its counts, or the error between
/// `error<<` and `>>error`.
#[cfg(feature = "testing-crash-injection")]
#[test]
fn image_check_child() {
    let Some(path) = std::env::var_os(PATH_VAR) else {
        return;
    };
    match GrafeoDB::open(PathBuf::from(path)) {
        Ok(db) => {
            println!(
                "nodes={} edges={}",
                db.node_count().unwrap(),
                db.edge_count().unwrap()
            );
            db.close().unwrap();
            drop(db);
            std::process::exit(OPENED);
        }
        Err(error) => {
            println!("error<<{error}>>error");
            std::process::exit(FAILED);
        }
    }
}

/// A migration whose image holds one node fewer, or one edge fewer, than the
/// old database fails with an error that names the difference (the kind and
/// both counts), removes the image and leaves the old file as it was: no
/// `.pre-0.6` copy, nothing renamed. The image is replaced at the pause point
/// `migrate:image_written` (see the module docs) with a 0.6 file of the old
/// database without Mia (a node with no edges), or without Alix's `KNOWS`
/// edge.
#[cfg(feature = "testing-crash-injection")]
#[test]
fn a_migration_whose_image_lost_rows_fails_and_keeps_the_old_file() {
    use std::process::Command;
    use std::time::{Duration, Instant};

    use grafeo_common::testing::child_process;
    use grafeo_common::testing::pause::{FLAG_VAR, POINT_VAR};

    for (case, change, kind) in [
        (
            "a node fewer",
            "MATCH (p:Person {name: 'Mia'}) DELETE p",
            "node",
        ),
        (
            "an edge fewer",
            "MATCH (:Person {name: 'Alix'})-[k:KNOWS]->() DELETE k",
            "edge",
        ),
    ] {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("db.grafeo");
        std::fs::copy(fixture(), &path).unwrap();
        let before = std::fs::read(&path).unwrap();
        let old = counts(&path);

        // The image to put in place of the migration's: the old database
        // with one row less, as a 0.6 file.
        let lossy = dir.path().join("lossy");
        std::fs::create_dir(&lossy).unwrap();
        let replacement = lossy.join("image.grafeo");
        {
            let copy = dir.path().join("copy.grafeo");
            std::fs::copy(fixture(), &copy).unwrap();
            let db = GrafeoDB::open_in_memory(&copy).unwrap();
            db.execute(change).unwrap();
            db.save(&replacement).unwrap();
            db.close().unwrap();
            drop(db);
            std::fs::remove_file(&copy).unwrap();
        }
        let image = counts(&replacement);
        let (old_count, image_count) = if kind == "node" {
            (old.0, image.0)
        } else {
            (old.1, image.1)
        };
        assert_eq!(
            image_count + 1,
            old_count,
            "{case}: the replacement has one {kind} fewer: {old:?} and {image:?}"
        );
        let header = {
            let manager =
                grafeo_storage::file::GrafeoFileManager::open_read_only(&replacement, None)
                    .unwrap();
            let header = manager.active_header();
            let counts = if kind == "node" {
                header.node_count
            } else {
                header.edge_count
            };
            manager.close().unwrap();
            usize::try_from(counts).unwrap()
        };
        let before_files = names(dir.path());

        let flag = dir.path().join("flag");
        let child = {
            let (path, flag) = (path.clone(), flag.clone());
            std::thread::spawn(move || {
                child_process::output(
                    Command::new(std::env::current_exe().unwrap())
                        .args(["--exact", "image_check_child", "--nocapture"])
                        .env(PATH_VAR, &path)
                        .env(POINT_VAR, IMAGE_WRITTEN)
                        .env(FLAG_VAR, &flag),
                )
                .unwrap()
            })
        };
        let paused = with_suffix(&flag, ".paused");
        let deadline = Instant::now() + Duration::from_secs(60);
        while !paused.exists() {
            if child.is_finished() {
                let output = child.join().unwrap();
                let installed = detect(&path).ok() == Some(OnDisk::Current)
                    && with_suffix(&path, ".pre-0.6").exists();
                panic!(
                    "{case}: the migration never paused at `{IMAGE_WRITTEN}`, the point this \
                     test replaces the image at (see the module docs); it exited with {:?}{}\n\
                     {}",
                    output.status,
                    if installed {
                        " and installed its image"
                    } else {
                        ""
                    },
                    String::from_utf8_lossy(&output.stdout)
                );
            }
            assert!(
                Instant::now() < deadline,
                "{case}: {} never appeared",
                paused.display()
            );
            std::thread::sleep(Duration::from_millis(5));
        }
        let migrating = with_suffix(&path, ".migrating");
        assert!(
            migrating.is_file(),
            "{case}: the image is written when the migration pauses"
        );
        std::fs::copy(&replacement, &migrating).unwrap();
        std::fs::write(with_suffix(&flag, ".resume"), b"go").unwrap();

        let output = child.join().unwrap();
        let stdout = String::from_utf8_lossy(&output.stdout).into_owned();
        assert_eq!(
            output.status.code(),
            Some(FAILED),
            "{case}: the migration of an image with one {kind} fewer fails: {:?}\n{stdout}\n{}",
            output.status,
            String::from_utf8_lossy(&output.stderr)
        );
        let error = stdout
            .split_once("error<<")
            .and_then(|(_, rest)| rest.split_once(">>error"))
            .map_or_else(
                || panic!("{case}: the child printed its error: {stdout}"),
                |(error, _)| error.to_string(),
            );
        // The numbers of the error, without those of the paths it names.
        let found = numbers(&error.replace(&dir.path().display().to_string(), ""));
        let named = |count: usize| found.contains(&count.to_string());
        let header_old = old_count + header - image_count;
        assert!(
            error.contains(kind)
                && ((named(old_count) && named(image_count))
                    || (named(header_old) && named(header))),
            "{case}: the error names the {kind} counts of the old database ({old_count}) and of \
             the image ({image_count}): {error}"
        );

        assert_eq!(
            std::fs::read(&path).unwrap(),
            before,
            "{case}: the old file is unchanged"
        );
        assert_eq!(
            detect(&path).unwrap(),
            OnDisk::LegacyFile,
            "{case}: still the 0.5.x file"
        );
        let mut expected_files = before_files.clone();
        expected_files.extend(["flag.paused".to_string(), "flag.resume".to_string()]);
        expected_files.sort();
        assert_eq!(
            names(dir.path()),
            expected_files,
            "{case}: no `.pre-0.6` copy, no image and no lock file left"
        );
    }
}
