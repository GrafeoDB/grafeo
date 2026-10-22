//! Online full backups and copies (#362): a full backup, `save` and
//! `to_memory` write the database's state at an epoch E from a snapshot, as
//! a checkpoint takes it, and hold commits off only while they take it.
//! Commits go on while they write the graph, and the copy holds exactly E.
//!
//! The commits run from inside the copy's write with the
//! `testing-statement-injection` hook `during_next_checkpoint_write` (the
//! LPG section's writer runs it for a copy as for a checkpoint); work that
//! should not wait gets 300 ms to finish.
//!
//! ```bash
//! cargo test -p grafeo-engine --all-features --test online_backup
//! ```

#![cfg(all(
    feature = "testing-statement-injection",
    feature = "testing-crash-injection",
    feature = "lpg",
    feature = "gql",
    feature = "wal",
    feature = "grafeo-file"
))]

use std::panic::AssertUnwindSafe;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::{Arc, mpsc};

use grafeo_common::testing::child_process;
use grafeo_common::testing::commit_hook::during_next_checkpoint_write;
use grafeo_common::testing::crash::{CrashResult, with_crash_at, with_failure_at};
use grafeo_common::types::{EpochId, Value};
use grafeo_engine::GrafeoDB;

#[path = "common/started.rs"]
mod started;

use started::Started;

/// The names of the people in `db`, sorted.
fn people(db: &GrafeoDB) -> Vec<Value> {
    db.execute("MATCH (p:Person) RETURN p.name AS name ORDER BY name")
        .unwrap()
        .rows()
        .iter()
        .map(|row| row[0].clone())
        .collect()
}

fn names(names: &[&str]) -> Vec<Value> {
    names.iter().map(|name| Value::from(*name)).collect()
}

/// A database at `path` with Alix, open, with nothing checkpointed since she
/// was written (she is in the WAL only).
fn open_with_alix(path: &Path) -> Arc<GrafeoDB> {
    let db = GrafeoDB::open(path).unwrap();
    db.execute("INSERT (:Person {name: 'Alix'})").unwrap();
    Arc::new(db)
}

/// Arms the copy's write to run, from other threads, a direct write, a
/// transaction's commit and an autocommit query, each writing one of
/// `writers`: returns what they report once the copy's caller ran it.
fn commit_during_the_next_copy(
    db: &Arc<GrafeoDB>,
    writers: [&'static str; 3],
) -> mpsc::Receiver<(Vec<Started<Result<(), String>>>, Vec<bool>)> {
    let (direct, transaction, query) = (Arc::clone(db), Arc::clone(db), Arc::clone(db));
    let (sender, started) = mpsc::channel();
    during_next_checkpoint_write(move || {
        let works = vec![
            Started::spawn(move || {
                direct
                    .create_node_with_props(&["Person"], [("name", Value::from(writers[0]))])
                    .map(drop)
                    .map_err(|e| e.to_string())
            }),
            Started::spawn(move || {
                let mut session = transaction.session();
                session.begin_transaction().map_err(|e| e.to_string())?;
                session
                    .execute(&format!("INSERT (:Person {{name: '{}'}})", writers[1]))
                    .map_err(|e| e.to_string())?;
                session.commit().map_err(|e| e.to_string())
            }),
            Started::spawn(move || {
                query
                    .execute(&format!("INSERT (:Person {{name: '{}'}})", writers[2]))
                    .map(drop)
                    .map_err(|e| e.to_string())
            }),
        ];
        let finished: Vec<bool> = works.iter().map(Started::finishes_briefly).collect();
        sender.send((works, finished)).unwrap();
    });
    started
}

/// Joins the writes `commit_during_the_next_copy` started, checking that
/// none of them waited for the copy.
fn none_waited(started: &mpsc::Receiver<(Vec<Started<Result<(), String>>>, Vec<bool>)>) {
    let (works, finished) = started.recv().expect("the copy ran the hook");
    assert_eq!(finished, [true, true, true], "no writer waits for the copy");
    for work in works {
        work.join().expect("the write succeeds");
    }
}

/// While a full backup writes its image, a direct write, a transaction's
/// commit and an autocommit query from other threads finish without waiting
/// for it; the backup holds exactly the state at its epoch (Alix), which a
/// restore to that epoch yields, and the database has every commit.
#[test]
fn commits_go_on_while_a_full_backup_writes_its_image() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("amsterdam.grafeo");
    let backups = dir.path().join("backups");
    let db = open_with_alix(&path);
    let epoch = db.current_epoch();

    let started = commit_during_the_next_copy(&db, ["Gus", "Vincent", "Jules"]);
    let segment = db.backup_full(&backups).unwrap();
    none_waited(&started);
    assert_eq!(
        segment.end_epoch, epoch,
        "the backup is of the epoch it began at"
    );
    assert!(
        !backups
            .join(format!("{}.partial", segment.filename))
            .exists(),
        "the temporary image is renamed"
    );

    let restored = dir.path().join("restored.grafeo");
    GrafeoDB::restore_to_epoch(&backups, epoch, &restored).unwrap();
    let image = GrafeoDB::open(&restored).unwrap();
    assert_eq!(
        people(&image),
        names(&["Alix"]),
        "the backup holds exactly E"
    );
    image.close().unwrap();

    assert_eq!(people(&db), names(&["Alix", "Gus", "Jules", "Vincent"]));
    Arc::into_inner(db).unwrap().close().unwrap();
}

/// The commits that went on while the full backup wrote are in the next
/// incremental backup: a restore to the latest epoch is the image and the
/// WAL segments after it, with every commit; a restore to the full backup's
/// epoch has its state only.
#[test]
fn a_full_backup_written_during_commits_chains_to_the_incrementals() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("berlin.grafeo");
    let backups = dir.path().join("backups");
    let db = open_with_alix(&path);
    let full_epoch = db.current_epoch();

    let started = commit_during_the_next_copy(&db, ["Gus", "Vincent", "Jules"]);
    db.backup_full(&backups).unwrap();
    none_waited(&started);
    db.execute("INSERT (:Person {name: 'Mia'})").unwrap();
    let incremental = db.backup_incremental(&backups).unwrap();
    assert_eq!(
        incremental.start_epoch,
        EpochId::new(full_epoch.as_u64() + 1),
        "the incremental starts after the full backup's epoch"
    );
    let latest = db.current_epoch();

    let at_latest = dir.path().join("latest.grafeo");
    GrafeoDB::restore_to_epoch(&backups, latest, &at_latest).unwrap();
    let image = GrafeoDB::open(&at_latest).unwrap();
    assert_eq!(
        people(&image),
        names(&["Alix", "Gus", "Jules", "Mia", "Vincent"]),
        "the full image and the segments after it hold every commit"
    );
    image.close().unwrap();

    let at_full = dir.path().join("full.grafeo");
    GrafeoDB::restore_to_epoch(&backups, full_epoch, &at_full).unwrap();
    let image = GrafeoDB::open(&at_full).unwrap();
    assert_eq!(people(&image), names(&["Alix"]));
    image.close().unwrap();
    Arc::into_inner(db).unwrap().close().unwrap();
}

/// A full backup leaves the database's own file and its next checkpoint as
/// they were: the backup is no checkpoint (the file still holds the image of
/// the last one), and a checkpoint after it writes and reopens as usual.
#[test]
fn a_full_backup_leaves_the_database_file_alone() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("paris.grafeo");
    let db = open_with_alix(&path);
    db.wal_checkpoint().unwrap();
    db.execute("INSERT (:Person {name: 'Gus'})").unwrap();
    let before = db.file_manager().unwrap().active_header();

    db.backup_full(&dir.path().join("backups")).unwrap();
    let after = db.file_manager().unwrap().active_header();
    assert_eq!(
        (after.iteration, after.epoch),
        (before.iteration, before.epoch),
        "the backup wrote no image into the database's file"
    );

    db.execute("INSERT (:Person {name: 'Vincent'})").unwrap();
    db.wal_checkpoint().unwrap();
    let db = Arc::into_inner(db).unwrap();
    db.close().unwrap();
    let db = GrafeoDB::open(&path).unwrap();
    assert_eq!(people(&db), names(&["Alix", "Gus", "Vincent"]));
    db.close().unwrap();
}

/// A full backup, a save and an in-memory copy taken after one value set in
/// a large database leave the next checkpoint as incremental as it would have
/// been: it writes that value's chunk and the LPG section's metadata, and
/// reuses every other chunk, as the copies take no marks. A reopen reads the
/// change, and so do the backup and the copies.
#[test]
fn a_copy_leaves_the_next_checkpoint_incremental() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("rotterdam.grafeo");
    let backups = dir.path().join("backups");
    let saved = dir.path().join("saved.grafeo");
    let db = GrafeoDB::open(&path).unwrap();
    db.execute("UNWIND range(0, 69999) AS i INSERT (:Person {i: i, city: 'Amsterdam'})")
        .unwrap();
    db.wal_checkpoint().unwrap();
    let full = db.file_manager().unwrap().last_checkpoint_stats().unwrap();

    db.execute("MATCH (p:Person {i: 69000}) SET p.city = 'Berlin'")
        .unwrap();
    db.backup_full(&backups).unwrap();
    db.save(&saved).unwrap();
    let in_memory = db.to_memory().unwrap();
    db.wal_checkpoint().unwrap();
    let incremental = db.file_manager().unwrap().last_checkpoint_stats().unwrap();
    assert_eq!(
        incremental.chunks_written, 2,
        "the city chunk of the second row group and the metadata: {incremental:?}"
    );
    assert_eq!(
        incremental.chunks_reused,
        full.chunks_written + full.chunks_reused - 2
    );
    let epoch = db.current_epoch();
    db.close().unwrap();

    let berlin = "MATCH (p:Person) WHERE p.city = 'Berlin' RETURN p.i";
    let cities = |db: &GrafeoDB| -> (Value, Value) {
        (
            db.execute(berlin).unwrap().rows()[0][0].clone(),
            db.execute("MATCH (p:Person) RETURN count(p)")
                .unwrap()
                .rows()[0][0]
                .clone(),
        )
    };
    let expected = (Value::Int64(69_000), Value::Int64(70_000));
    assert_eq!(cities(&in_memory), expected, "the in-memory copy");
    let restored = dir.path().join("restored.grafeo");
    GrafeoDB::restore_to_epoch(&backups, epoch, &restored).unwrap();
    for (name, file) in [
        ("the database", &path),
        ("the save", &saved),
        ("the backup", &restored),
    ] {
        let db = GrafeoDB::open(file).unwrap();
        assert_eq!(cities(&db), expected, "{name}, reopened");
        db.close().unwrap();
    }
}

/// The crash point a child process of
/// [`a_crash_during_a_full_backup_leaves_the_database_and_the_backups_whole`]
/// crashes at.
const CRASH_POINT_VAR: &str = "GRAFEO_ONLINE_BACKUP_CRASH_POINT";
/// The directory with the child's database (`prague.grafeo`) and its backups.
const DIR_VAR: &str = "GRAFEO_ONLINE_BACKUP_DIR";
/// Exit code of a child whose backup crashed.
const CRASHED: i32 = 3;
/// Exit code of a child whose backup completed.
const BACKED_UP: i32 = 4;

/// Opens the database in `dir`, commits Gus (to the WAL only) and takes a full
/// backup to `dir/backups`, in a child process that crashes at the
/// `crash_point`-th crash point of the backup. Returns the crash point's name,
/// or `None` if the backup completed.
fn back_up_in_child(crash_point: usize, dir: &Path) -> Option<String> {
    let output = child_process::output(
        Command::new(std::env::current_exe().unwrap())
            .args(["--exact", "back_up_child", "--nocapture"])
            .env(CRASH_POINT_VAR, crash_point.to_string())
            .env(DIR_VAR, dir),
    )
    .unwrap();
    let stderr = String::from_utf8_lossy(&output.stderr);
    match output.status.code() {
        Some(BACKED_UP) => None,
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

/// Child-process entry for [`back_up_in_child`]; a no-op when run directly.
/// The database is built outside the crash closure, and the child exits
/// without running destructors, as a crash does (Lessons, Testing 5).
#[test]
fn back_up_child() {
    let (Ok(point), Some(dir)) = (std::env::var(CRASH_POINT_VAR), std::env::var_os(DIR_VAR)) else {
        return;
    };
    let dir = PathBuf::from(dir);
    let db = GrafeoDB::open(dir.join("prague.grafeo")).unwrap();
    db.execute("INSERT (:Person {name: 'Gus'})").unwrap();
    let backups = dir.join("backups");
    let db = AssertUnwindSafe(&db);
    match with_crash_at(point.parse().unwrap(), move || db.backup_full(&backups)) {
        CrashResult::Completed(Ok(_)) => std::process::exit(BACKED_UP),
        CrashResult::Completed(Err(error)) => panic!("the backup failed: {error}"),
        _ => std::process::exit(CRASHED),
    }
}

/// The files in `backups` that end in `.partial`.
fn partial_files(backups: &Path) -> Vec<String> {
    std::fs::read_dir(backups)
        .unwrap()
        .map(|entry| entry.unwrap().file_name().to_string_lossy().into_owned())
        .filter(|name| name.ends_with(".partial"))
        .collect()
}

/// Restores `backups` to each epoch and checks the people of the restored
/// database, opened for writing and read-only.
fn assert_restores(backups: &Path, restores: &[(EpochId, Vec<Value>)], context: &str) {
    for (at, expected) in restores {
        let dir = tempfile::tempdir().unwrap();
        let restored = dir.path().join("restored.grafeo");
        GrafeoDB::restore_to_epoch(backups, *at, &restored)
            .unwrap_or_else(|e| panic!("{context}: a restore to {at:?}: {e}"));
        let db = GrafeoDB::open(&restored).unwrap();
        assert_eq!(&people(&db), expected, "{context}: a restore to {at:?}");
        db.close().unwrap();
        let db = GrafeoDB::open_read_only(&restored).unwrap();
        assert_eq!(
            &people(&db),
            expected,
            "{context}: a restore to {at:?}, read-only"
        );
    }
}

/// A crash at any point of a full backup, in another process, leaves the
/// database whole (the reopen has every commit, Gus from the WAL) and the
/// backups as they were or with the whole new backup: the manifest lists an
/// image only once it is complete, and an image under its final name opens
/// with exactly the state of its epoch. The next full backup to the same
/// directory works, past what the crash left there, and the incremental
/// backups chain from it: a restore to each epoch yields exactly its state.
#[test]
fn a_crash_during_a_full_backup_leaves_the_database_and_the_backups_whole() {
    let mut reached = Vec::new();
    for crash_point in 1.. {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("prague.grafeo");
        let backups = dir.path().join("backups");
        let db = GrafeoDB::open(&path).unwrap();
        db.execute("INSERT (:Person {name: 'Alix'})").unwrap();
        db.backup_full(&backups).unwrap();
        let alix = db.current_epoch();
        db.close().unwrap();

        let crashed_at = back_up_in_child(crash_point, dir.path());
        let context = format!(
            "crash_point={crash_point} ({})",
            crashed_at.as_deref().unwrap_or("no crash")
        );
        let image = backups.join("backup_full_0001.grafeo");
        if image.exists() {
            let copy = dir.path().join("image.grafeo");
            std::fs::copy(&image, &copy).unwrap();
            let backup = GrafeoDB::open(&copy).unwrap();
            assert_eq!(
                people(&backup),
                names(&["Alix", "Gus"]),
                "{context}: an image under its name is whole"
            );
            backup.close().unwrap();
        }
        assert_restores(&backups, &[(alix, names(&["Alix"]))], &context);

        let db = GrafeoDB::open(&path).unwrap();
        assert_eq!(
            people(&db),
            names(&["Alix", "Gus"]),
            "{context}: the database keeps every commit"
        );
        let gus = db.current_epoch();
        db.backup_full(&backups)
            .unwrap_or_else(|e| panic!("{context}: the next backup works: {e}"));
        assert_eq!(partial_files(&backups), Vec::<String>::new(), "{context}");
        db.execute("INSERT (:Person {name: 'Vincent'})").unwrap();
        db.backup_incremental(&backups).unwrap();
        let vincent = db.current_epoch();
        db.close().unwrap();
        assert_restores(
            &backups,
            &[
                (alix, names(&["Alix"])),
                (gus, names(&["Alix", "Gus"])),
                (vincent, names(&["Alix", "Gus", "Vincent"])),
            ],
            &context,
        );

        reached.push(crashed_at.clone());
        if crashed_at.is_none() {
            break;
        }
    }

    // The steps that make the backup safe, in order: the snapshot, commits
    // released, the image written, renamed into place, then its manifest
    // entry.
    let steps: Vec<&str> = reached
        .iter()
        .flatten()
        .map(String::as_str)
        .filter(|point| point.starts_with("copy:") || point.starts_with("backup:"))
        .collect();
    assert_eq!(
        steps,
        [
            "copy:after_snapshot",
            "copy:after_release",
            "copy:after_write",
            "copy:before_rename",
            "backup:after_image",
            "backup:after_manifest",
        ],
        "{reached:?}"
    );
}

/// An I/O error at any point of a full backup fails it and removes what it
/// wrote. A failure of the database file's writes leaves the database working:
/// it takes commits, the next full backup to the same directory holds exactly
/// its epoch, and the incremental backups chain from it; so does a failed
/// write of the WAL segment the backup starts. A failed sync of the WAL (a segment or its
/// directory) stops the log instead, as any failed WAL sync does: writes then fail with
/// `GRAFEO-T011`, and the reopen has every commit.
#[test]
fn a_failed_full_backup_leaves_the_database_working() {
    let mut file_failures = 0;
    for count in 1.. {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("brno.grafeo");
        let backups = dir.path().join("backups");
        let db = GrafeoDB::open(&path).unwrap();
        db.execute("INSERT (:Person {name: 'Alix'})").unwrap();
        let error = match with_failure_at(count, || db.backup_full(&backups)) {
            Ok(_) => break,
            Err(error) => error.to_string(),
        };
        let context = format!("failure {count}: {error}");
        assert!(error.contains("injected failure at: "), "{context}");
        assert_eq!(partial_files(&backups), Vec::<String>::new(), "{context}");

        if ["wal:sync", "wal:dir_sync"]
            .iter()
            .any(|point| error.ends_with(&format!("injected failure at: {point}")))
        {
            let refused = db
                .execute("INSERT (:Person {name: 'Gus'})")
                .unwrap_err()
                .to_string();
            assert!(refused.contains("GRAFEO-T011"), "{context}: {refused}");
            drop(db);
            let db = GrafeoDB::open(&path).unwrap();
            assert_eq!(people(&db), names(&["Alix"]), "{context}: the reopen");
            db.close().unwrap();
            continue;
        }
        file_failures += 1;
        db.execute("INSERT (:Person {name: 'Gus'})").unwrap();
        let gus = db.current_epoch();
        db.backup_full(&backups)
            .unwrap_or_else(|e| panic!("{context}: the next backup works: {e}"));
        db.execute("INSERT (:Person {name: 'Vincent'})").unwrap();
        db.backup_incremental(&backups).unwrap();
        let vincent = db.current_epoch();
        db.close().unwrap();
        assert_restores(
            &backups,
            &[
                (gus, names(&["Alix", "Gus"])),
                (vincent, names(&["Alix", "Gus", "Vincent"])),
            ],
            &context,
        );
        let db = GrafeoDB::open(&path).unwrap();
        assert_eq!(
            people(&db),
            names(&["Alix", "Gus", "Vincent"]),
            "{context}: the database after the failure"
        );
        db.close().unwrap();
    }
    assert!(
        file_failures >= 4,
        "the backup reaches the failure points of the image's checkpoint"
    );
}

/// The database a child of [`a_read_only_full_backup_holds_what_it_sees`]
/// writes and leaves without closing.
const LEAVE_WAL_PATH_VAR: &str = "GRAFEO_ONLINE_BACKUP_LEAVE_WAL_PATH";

/// Child-process entry for [`a_read_only_full_backup_holds_what_it_sees`]; a
/// no-op when run directly. Commits Alix, takes a full backup, commits Gus
/// and exits without closing, so Gus is in the WAL only.
#[test]
fn leave_commits_in_the_wal_child() {
    let Some(path) = std::env::var_os(LEAVE_WAL_PATH_VAR) else {
        return;
    };
    let path = PathBuf::from(path);
    let db = GrafeoDB::open(&path).unwrap();
    db.execute("INSERT (:Person {name: 'Alix'})").unwrap();
    db.backup_full(&path.with_file_name("backups")).unwrap();
    db.execute("INSERT (:Person {name: 'Gus'})").unwrap();
    std::process::exit(0);
}

/// A writer leaves Gus in the WAL (it exits without closing). A read-only open
/// sees him, and its full backup holds what it sees: a restore to its epoch
/// has Gus. The writer that opens next commits Vincent and takes an
/// incremental backup: a restore to its epoch has all three, the read-only
/// full backup in the chain or not.
#[test]
fn a_read_only_full_backup_holds_what_it_sees() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("ghent.grafeo");
    let backups = dir.path().join("backups");
    let output = child_process::output(
        Command::new(std::env::current_exe().unwrap())
            .args(["--exact", "leave_commits_in_the_wal_child", "--nocapture"])
            .env(LEAVE_WAL_PATH_VAR, &path),
    )
    .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );

    let db = GrafeoDB::open_read_only(&path).unwrap();
    assert_eq!(people(&db), names(&["Alix", "Gus"]), "the read-only open");
    db.backup_full(&backups).unwrap();
    let gus = db.current_epoch();
    drop(db);
    assert_restores(&backups, &[(gus, names(&["Alix", "Gus"]))], "read-only");

    let db = GrafeoDB::open(&path).unwrap();
    db.execute("INSERT (:Person {name: 'Vincent'})").unwrap();
    db.backup_incremental(&backups).unwrap();
    let vincent = db.current_epoch();
    db.close().unwrap();
    assert_restores(
        &backups,
        &[
            (gus, names(&["Alix", "Gus"])),
            (vincent, names(&["Alix", "Gus", "Vincent"])),
        ],
        "after the writer",
    );
}

/// `save` writes the state at its epoch with commits going on: the writers
/// do not wait, and the saved file holds Alix only.
#[test]
fn commits_go_on_while_a_save_writes() {
    let dir = tempfile::tempdir().unwrap();
    let db = open_with_alix(&dir.path().join("lisbon.grafeo"));
    let saved = dir.path().join("saved.grafeo");

    let started = commit_during_the_next_copy(&db, ["Gus", "Vincent", "Jules"]);
    db.save(&saved).unwrap();
    none_waited(&started);
    assert!(!dir.path().join("saved.grafeo.partial").exists());

    let copy = GrafeoDB::open(&saved).unwrap();
    assert_eq!(
        people(&copy),
        names(&["Alix"]),
        "the save holds its epoch only"
    );
    copy.close().unwrap();
    Arc::into_inner(db).unwrap().close().unwrap();
}

/// `to_memory` copies the state at its epoch with commits going on.
#[test]
fn commits_go_on_while_to_memory_copies() {
    let dir = tempfile::tempdir().unwrap();
    let db = open_with_alix(&dir.path().join("vienna.grafeo"));

    let started = commit_during_the_next_copy(&db, ["Gus", "Vincent", "Jules"]);
    let copy = db.to_memory().unwrap();
    none_waited(&started);
    assert_eq!(
        people(&copy),
        names(&["Alix"]),
        "the copy holds its epoch only"
    );
    Arc::into_inner(db).unwrap().close().unwrap();
}

/// `export_snapshot` reads the graphs at its epoch with commits going on:
/// the writers do not wait, and the snapshot holds Alix only.
#[test]
fn commits_go_on_while_a_snapshot_is_exported() {
    let dir = tempfile::tempdir().unwrap();
    let db = open_with_alix(&dir.path().join("madrid.grafeo"));

    let started = commit_during_the_next_copy(&db, ["Gus", "Vincent", "Jules"]);
    let snapshot = db.export_snapshot().unwrap();
    none_waited(&started);
    let copy = GrafeoDB::import_snapshot(&snapshot).unwrap();
    assert_eq!(
        people(&copy),
        names(&["Alix"]),
        "the snapshot holds its epoch only"
    );
    assert_eq!(people(&db), names(&["Alix", "Gus", "Jules", "Vincent"]));
    Arc::into_inner(db).unwrap().close().unwrap();
}
