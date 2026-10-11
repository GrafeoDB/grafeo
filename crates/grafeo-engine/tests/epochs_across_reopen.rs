//! The epoch continues across a reopen: a database never opens below the last
//! epoch its file or WAL holds. A checkpoint writes the database epoch into the
//! file's database header, and every commit logs it in the WAL (`EpochAdvance`)
//! after its records; an open continues from the highest of the two. A 0.5.x
//! database is migrated at the highest epoch its header, catalog and WAL name.
//!
//! Time travel addresses history by epoch, and a backup chain names its
//! segments by epoch: both lose their meaning when a reopen starts the epochs
//! over, as 0.5.44 did at every reopen (without `temporal`).
//!
//! Crashes run in a child process that exits without `close()`. Run without
//! and with `temporal` (which stores the epoch in the LPG section too, and
//! counts the replayed commits); the 0.5.x fixtures need the features of
//! `full` and `compact-store`:
//!
//! ```bash
//! cargo test -p grafeo-engine --features full,compact-store,testing-crash-injection --test epochs_across_reopen
//! cargo test -p grafeo-engine --all-features --test epochs_across_reopen
//! ```

#![cfg(all(feature = "wal", feature = "grafeo-file", feature = "lpg"))]

use std::path::{Path, PathBuf};

use grafeo_common::testing::child_process;
use grafeo_common::types::EpochId;
use grafeo_engine::GrafeoDB;
use grafeo_engine::database::backup::BackupKind;

const SCENARIO_VAR: &str = "GRAFEO_EPOCHS_ACROSS_REOPEN_SCENARIO";
const PATH_VAR: &str = "GRAFEO_EPOCHS_ACROSS_REOPEN_PATH";

/// What a child prints before its last line: the epoch it reached.
const EPOCH_PREFIX: &str = "reached epoch ";

/// Inserts a person named `name` in a statement of its own (one commit).
fn insert(db: &GrafeoDB, name: &str) {
    db.execute(&format!("INSERT (:Person {{name: '{name}'}})"))
        .unwrap_or_else(|e| panic!("insert {name}: {e}"));
}

/// The names of the people in `db`, sorted.
fn names(db: &GrafeoDB) -> Vec<String> {
    db.execute("MATCH (p:Person) RETURN p.name ORDER BY p.name")
        .expect("read the names")
        .rows()
        .iter()
        .map(|row| match &row[0] {
            grafeo_common::types::Value::String(name) => name.to_string(),
            other => panic!("a name is a string, got {other:?}"),
        })
        .collect()
}

fn people(list: &[&str]) -> Vec<String> {
    list.iter().map(ToString::to_string).collect()
}

fn sidecar_wal(path: &Path) -> PathBuf {
    let mut sidecar = path.as_os_str().to_owned();
    sidecar.push(".wal");
    PathBuf::from(sidecar)
}

/// What a child prints when the checkpoint it was to crash in completed.
#[cfg(feature = "testing-crash-injection")]
const CHECKPOINT_COMPLETED: &str = "the checkpoint completed";

/// Runs `scenario` in a child process that exits without closing the
/// database, like a crash, and returns the epoch it reached.
fn crash_after(scenario: &str, path: &Path) -> EpochId {
    run_child(scenario, path).0
}

/// [`crash_after`], also returning what the child printed.
fn run_child(scenario: &str, path: &Path) -> (EpochId, String) {
    let output = child_process::output(
        std::process::Command::new(std::env::current_exe().unwrap())
            .args(["--exact", "crash_child", "--nocapture"])
            .env(SCENARIO_VAR, scenario)
            .env(PATH_VAR, path),
    )
    .unwrap();
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        output.status.success(),
        "scenario {scenario} failed: {stdout}{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let epoch = stdout
        .lines()
        .find_map(|line| line.strip_prefix(EPOCH_PREFIX))
        .unwrap_or_else(|| panic!("scenario {scenario} printed no epoch: {stdout}"));
    (
        EpochId::new(epoch.trim().parse().unwrap()),
        stdout.to_string(),
    )
}

/// Child-process entry for [`crash_after`]; a no-op when run directly.
#[test]
fn crash_child() {
    let Ok(scenario) = std::env::var(SCENARIO_VAR) else {
        return;
    };
    let path = PathBuf::from(std::env::var_os(PATH_VAR).unwrap());
    let db = GrafeoDB::open(&path).unwrap();
    match scenario.as_str() {
        // Every commit is only in the WAL.
        "commits" => {
            for name in ["Alix", "Gus", "Vincent"] {
                insert(&db, name);
            }
        }
        // The file holds the first commits, the WAL the later ones.
        "checkpoint_then_commits" => {
            insert(&db, "Alix");
            insert(&db, "Gus");
            db.wal_checkpoint().unwrap();
            insert(&db, "Vincent");
            insert(&db, "Mia");
        }
        // A reopen replays the WAL, commits once more and crashes again.
        "reopen_then_commit" => insert(&db, "Jules"),
        // A full and an incremental backup, then a crash: the backup cursor
        // stays in the WAL directory for the next incremental backup.
        "backups" => {
            let backups = backup_dir(&path);
            insert(&db, "Alix");
            db.backup_full(&backups).unwrap();
            insert(&db, "Gus");
            insert(&db, "Vincent");
            db.backup_incremental(&backups).unwrap();
        }
        // Crash at injection point N of a checkpoint after an earlier one.
        #[cfg(feature = "testing-crash-injection")]
        other if other.starts_with("checkpoint:") => {
            let point: u64 = other["checkpoint:".len()..].parse().unwrap();
            insert(&db, "Alix");
            insert(&db, "Gus");
            db.wal_checkpoint().unwrap();
            insert(&db, "Vincent");
            println!("{EPOCH_PREFIX}{}", db.current_epoch().as_u64());
            let target = std::panic::AssertUnwindSafe(&db);
            let result = grafeo_common::testing::crash::with_crash_at(point, move || {
                target.wal_checkpoint()
            });
            if let grafeo_common::testing::crash::CrashResult::Completed(checkpoint) = result {
                checkpoint.unwrap();
                println!("{CHECKPOINT_COMPLETED}");
            }
            // Crash: no close(), no destructors.
            std::process::exit(0);
        }
        other => panic!("unknown scenario {other}"),
    }
    println!("{EPOCH_PREFIX}{}", db.current_epoch().as_u64());
    // Crash: no close(), no destructors.
    std::process::exit(0);
}

fn backup_dir(path: &Path) -> PathBuf {
    path.with_extension("backups")
}

// ── Close and reopen, checkpoints ─────────────────────────────────

#[test]
fn the_epoch_continues_after_close_and_reopen() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("people.grafeo");

    let db = GrafeoDB::open(&path).unwrap();
    for name in ["Alix", "Gus", "Vincent"] {
        insert(&db, name);
    }
    let last = db.current_epoch();
    assert_eq!(last, EpochId::new(3), "three commits, three epochs");
    db.close().unwrap();

    let db = GrafeoDB::open(&path).unwrap();
    assert_eq!(
        db.current_epoch(),
        last,
        "the reopen continues at the last epoch"
    );
    insert(&db, "Mia");
    assert_eq!(
        db.current_epoch(),
        EpochId::new(last.as_u64() + 1),
        "the next commit takes the next epoch"
    );
    let next = db.current_epoch();
    db.close().unwrap();

    let db = GrafeoDB::open(&path).unwrap();
    assert_eq!(db.current_epoch(), next, "and so does every later reopen");
    assert_eq!(names(&db), people(&["Alix", "Gus", "Mia", "Vincent"]));
    db.close().unwrap();
}

#[test]
fn a_checkpoint_records_the_epoch_in_the_database_header() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("people.grafeo");

    let db = GrafeoDB::open(&path).unwrap();
    insert(&db, "Alix");
    insert(&db, "Gus");
    db.wal_checkpoint().unwrap();
    let header = db.file_manager().unwrap().active_header();
    assert_eq!(
        EpochId::new(header.epoch),
        db.current_epoch(),
        "the header holds the epoch of the checkpoint"
    );
    let checkpointed = db.current_epoch();
    insert(&db, "Vincent");
    let last = db.current_epoch();
    db.close().unwrap();

    let db = GrafeoDB::open(&path).unwrap();
    let header = db.file_manager().unwrap().active_header();
    assert_eq!(
        EpochId::new(header.epoch),
        last,
        "close() checkpoints at the last epoch"
    );
    assert!(last > checkpointed);
    assert_eq!(db.current_epoch(), last);
    db.close().unwrap();
}

// ── Crash and WAL replay ──────────────────────────────────────────

#[test]
fn the_epoch_continues_after_a_crash_replays_the_wal() {
    for scenario in ["commits", "checkpoint_then_commits"] {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("people.grafeo");
        let reached = crash_after(scenario, &path);
        assert!(
            sidecar_wal(&path).exists(),
            "{scenario}: the crash leaves the WAL to replay"
        );

        let db = GrafeoDB::open(&path).unwrap();
        assert_eq!(
            db.current_epoch(),
            reached,
            "{scenario}: the replay continues at the epoch of the last commit"
        );
        insert(&db, "Jules");
        assert_eq!(db.current_epoch(), EpochId::new(reached.as_u64() + 1));
        db.close().unwrap();
    }
}

#[test]
fn every_crash_and_replay_continues_the_epochs() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("people.grafeo");
    let mut reached = crash_after("checkpoint_then_commits", &path);
    for round in 0..3 {
        let next = crash_after("reopen_then_commit", &path);
        assert_eq!(
            next,
            EpochId::new(reached.as_u64() + 1),
            "round {round}: the commit after the replay takes the next epoch"
        );
        reached = next;
    }
    assert!(sidecar_wal(&path).exists(), "no WAL left to replay");

    let db = GrafeoDB::open(&path).unwrap();
    assert_eq!(db.current_epoch(), reached);
    db.close().unwrap();
}

/// A crash at every injection point of a checkpoint: whether the new image
/// or only the old one with the WAL survives, the reopen continues at the
/// last committed epoch.
#[cfg(feature = "testing-crash-injection")]
#[test]
fn a_crash_during_a_checkpoint_keeps_the_epoch() {
    let mut crashes = 0;
    // More points than a checkpoint has, so the last runs complete.
    for point in 1..=12 {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("people.grafeo");
        let scenario = format!("checkpoint:{point}");
        let (reached, stdout) = run_child(&scenario, &path);
        if !stdout.contains(CHECKPOINT_COMPLETED) {
            crashes += 1;
        }
        assert!(
            sidecar_wal(&path).exists(),
            "point {point}: the crash leaves the WAL"
        );

        let db = GrafeoDB::open(&path).unwrap();
        // With `temporal`, replay also advances the epoch at the commit
        // markers it replays over a new image that holds them already (a
        // crash after the image, before the WAL is marked): the reopen can
        // continue above the last committed epoch, never below it.
        #[cfg(feature = "temporal")]
        assert!(
            db.current_epoch() >= reached,
            "point {point}: the reopen continues at {}, below the last committed epoch {}",
            db.current_epoch().as_u64(),
            reached.as_u64()
        );
        #[cfg(not(feature = "temporal"))]
        assert_eq!(
            db.current_epoch(),
            reached,
            "point {point}: the reopen continues at the last committed epoch"
        );
        assert_eq!(names(&db), people(&["Alix", "Gus", "Vincent"]));
        db.close().unwrap();
    }
    assert!(
        (1..12).contains(&crashes),
        "{crashes} of 12 runs crashed: the sweep covers every point of a checkpoint"
    );
}

// ── Read-only and in-memory opens ─────────────────────────────────

#[test]
fn a_read_only_open_reads_at_the_last_epoch() {
    let dir = tempfile::tempdir().unwrap();

    let closed = dir.path().join("closed.grafeo");
    let db = GrafeoDB::open(&closed).unwrap();
    insert(&db, "Alix");
    insert(&db, "Gus");
    let last = db.current_epoch();
    db.close().unwrap();
    let db = GrafeoDB::open_read_only(&closed).unwrap();
    assert_eq!(db.current_epoch(), last, "a closed database");
    db.close().unwrap();

    let crashed = dir.path().join("crashed.grafeo");
    let reached = crash_after("checkpoint_then_commits", &crashed);
    let db = GrafeoDB::open_read_only(&crashed).unwrap();
    assert_eq!(
        db.current_epoch(),
        reached,
        "a crashed database, its WAL replayed in memory"
    );
    db.close().unwrap();
}

#[test]
fn open_in_memory_and_to_memory_keep_the_epoch() {
    let dir = tempfile::tempdir().unwrap();

    let closed = dir.path().join("closed.grafeo");
    let db = GrafeoDB::open(&closed).unwrap();
    insert(&db, "Alix");
    insert(&db, "Gus");
    let last = db.current_epoch();
    let copy = db.to_memory().unwrap();
    assert_eq!(copy.current_epoch(), last, "to_memory");
    db.close().unwrap();
    let copy = GrafeoDB::open_in_memory(&closed).unwrap();
    assert_eq!(
        copy.current_epoch(),
        last,
        "open_in_memory of a closed file"
    );
    insert(&copy, "Mia");
    assert_eq!(copy.current_epoch(), EpochId::new(last.as_u64() + 1));

    let crashed = dir.path().join("crashed.grafeo");
    let reached = crash_after("checkpoint_then_commits", &crashed);
    let copy = GrafeoDB::open_in_memory(&crashed).unwrap();
    assert_eq!(
        copy.current_epoch(),
        reached,
        "open_in_memory of a crashed file, its WAL replayed"
    );
}

/// `restore_snapshot` clears the store and loads an older snapshot: the
/// database's epoch goes on, and a checkpoint never writes a lower one.
#[test]
fn restoring_a_snapshot_does_not_lower_the_epoch() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("people.grafeo");

    let db = GrafeoDB::open(&path).unwrap();
    insert(&db, "Alix");
    let snapshot = db.export_snapshot().unwrap();
    for name in ["Gus", "Vincent", "Mia"] {
        insert(&db, name);
    }
    let last = db.current_epoch();
    db.restore_snapshot(&snapshot).unwrap();
    assert_eq!(names(&db), people(&["Alix"]));
    assert!(db.current_epoch() >= last, "the restore keeps the epoch");
    db.close().unwrap();

    let db = GrafeoDB::open(&path).unwrap();
    assert!(
        db.current_epoch() >= last,
        "the reopen continues at or above epoch {}, got {}",
        last.as_u64(),
        db.current_epoch().as_u64()
    );
    assert_eq!(names(&db), people(&["Alix"]));
    db.close().unwrap();
}

// ── Backups ───────────────────────────────────────────────────────

/// The kind and epochs of every segment of the manifest in `backups`.
fn segments(backups: &Path) -> Vec<(BackupKind, u64, u64)> {
    GrafeoDB::read_backup_manifest(backups)
        .unwrap()
        .expect("a manifest")
        .segments
        .iter()
        .map(|s| (s.kind, s.start_epoch.as_u64(), s.end_epoch.as_u64()))
        .collect()
}

/// Every segment ends after the one before it, and an incremental segment
/// starts right after it: the epochs of the chain never repeat.
fn assert_strictly_increasing(chain: &[(BackupKind, u64, u64)]) {
    for pair in chain.windows(2) {
        let (_, _, previous_end) = pair[0];
        let (kind, start, end) = pair[1];
        assert!(
            end > previous_end,
            "{chain:?}: a segment ends after the one before"
        );
        if kind == BackupKind::Incremental {
            assert_eq!(
                start,
                previous_end + 1,
                "{chain:?}: an incremental segment starts after the one before"
            );
            assert!(start <= end, "{chain:?}: a segment starts before it ends");
        }
    }
}

/// The names in the database restored from `backups` at `epoch`.
fn restored_names(backups: &Path, epoch: u64, dir: &Path) -> Vec<String> {
    let path = dir.join(format!("restored_at_{epoch}.grafeo"));
    GrafeoDB::restore_to_epoch(backups, EpochId::new(epoch), &path)
        .unwrap_or_else(|e| panic!("restore to epoch {epoch}: {e}"));
    let db = GrafeoDB::open(&path).unwrap();
    assert_eq!(
        db.current_epoch(),
        EpochId::new(epoch),
        "the restored database opens at the epoch it was restored to"
    );
    let found = names(&db);
    db.close().unwrap();
    found
}

/// A close removes the WAL and with it the backup cursor, so the chain goes
/// on with a full backup after the reopen; its epochs go on too, and a
/// restore to an epoch before the reopen finds the data of then.
#[test]
fn a_backup_chain_goes_on_across_a_reopen() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("people.grafeo");
    let backups = dir.path().join("backups");

    let db = GrafeoDB::open(&path).unwrap();
    insert(&db, "Alix");
    let full = db.backup_full(&backups).unwrap();
    insert(&db, "Gus");
    let incremental = db.backup_incremental(&backups).unwrap();
    db.close().unwrap();

    let db = GrafeoDB::open(&path).unwrap();
    insert(&db, "Vincent");
    let second_full = db.backup_full(&backups).unwrap();
    insert(&db, "Mia");
    let second_incremental = db.backup_incremental(&backups).unwrap();
    db.close().unwrap();

    let chain = segments(&backups);
    assert_eq!(chain.len(), 4);
    assert_strictly_increasing(&chain);

    assert_eq!(
        restored_names(&backups, full.end_epoch.as_u64(), dir.path()),
        people(&["Alix"])
    );
    assert_eq!(
        restored_names(&backups, incremental.end_epoch.as_u64(), dir.path()),
        people(&["Alix", "Gus"]),
        "the last epoch before the reopen"
    );
    assert_eq!(
        restored_names(&backups, second_full.end_epoch.as_u64(), dir.path()),
        people(&["Alix", "Gus", "Vincent"])
    );
    assert_eq!(
        restored_names(&backups, second_incremental.end_epoch.as_u64(), dir.path()),
        people(&["Alix", "Gus", "Mia", "Vincent"])
    );
}

/// After a crash the backup cursor is still in the WAL directory: the next
/// incremental backup continues the chain right after the last one.
#[test]
fn a_backup_chain_goes_on_across_a_crash() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("people.grafeo");
    let backups = backup_dir(&path);
    let reached = crash_after("backups", &path);

    let db = GrafeoDB::open(&path).unwrap();
    assert_eq!(db.current_epoch(), reached);
    insert(&db, "Mia");
    db.backup_incremental(&backups).unwrap();
    db.close().unwrap();

    let chain = segments(&backups);
    assert_eq!(chain.len(), 3);
    assert_strictly_increasing(&chain);
    let (_, _, before_the_crash) = chain[1];
    let (_, _, after_the_crash) = chain[2];
    assert_eq!(
        restored_names(&backups, before_the_crash, dir.path()),
        people(&["Alix", "Gus", "Vincent"])
    );
    assert_eq!(
        restored_names(&backups, after_the_crash, dir.path()),
        people(&["Alix", "Gus", "Mia", "Vincent"])
    );
}

/// A restore to each epoch of a chain opens at that epoch, with the data of
/// then, and the restored database goes on from it.
#[test]
fn a_restore_to_an_epoch_opens_at_that_epoch() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("people.grafeo");
    let backups = dir.path().join("backups");

    let db = GrafeoDB::open(&path).unwrap();
    insert(&db, "Alix");
    let full = db.backup_full(&backups).unwrap();
    insert(&db, "Gus");
    insert(&db, "Vincent");
    let incremental = db.backup_incremental(&backups).unwrap();
    db.close().unwrap();

    let start = full.end_epoch.as_u64();
    let end = incremental.end_epoch.as_u64();
    assert_eq!(end, start + 2, "one epoch per commit");
    assert_eq!(
        restored_names(&backups, start, dir.path()),
        people(&["Alix"])
    );
    assert_eq!(
        restored_names(&backups, start + 1, dir.path()),
        people(&["Alix", "Gus"])
    );
    assert_eq!(
        restored_names(&backups, end, dir.path()),
        people(&["Alix", "Gus", "Vincent"])
    );

    let restored = dir.path().join(format!("restored_at_{start}.grafeo"));
    let db = GrafeoDB::open(&restored).unwrap();
    insert(&db, "Mia");
    assert_eq!(db.current_epoch(), EpochId::new(start + 1));
    db.close().unwrap();
}

// ── Databases written by 0.5.x ────────────────────────────────────

/// The released fixtures hold triples and search indexes, which only these
/// features read.
#[cfg(all(
    feature = "triple-store",
    feature = "vector-index",
    feature = "text-index"
))]
mod released {
    use super::*;

    /// A released 0.5.x fixture, copied with its sidecar WAL (see
    /// `fixtures/released/`).
    fn copy_released(version: &str, name: &str, to: &Path) -> PathBuf {
        let source = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures/released")
            .join(version);
        for entry in std::fs::read_dir(&source).unwrap() {
            let entry = entry.unwrap();
            if entry.file_name().to_string_lossy().starts_with(name) {
                copy_tree(&entry.path(), &to.join(entry.file_name()));
            }
        }
        to.join(name)
    }

    fn copy_tree(from: &Path, to: &Path) {
        if from.is_dir() {
            std::fs::create_dir_all(to).unwrap();
            for entry in std::fs::read_dir(from).unwrap() {
                let entry = entry.unwrap();
                copy_tree(&entry.path(), &to.join(entry.file_name()));
            }
        } else {
            std::fs::copy(from, to).unwrap();
        }
    }

    /// The highest epoch each released fixture names: a file's database header
    /// and catalog hold the epochs of its last checkpoint (the root store's and
    /// the transaction manager's, which 0.5.43 let run ahead), and its WAL logs
    /// the epochs of the commits after it (0.5.x started them over at every
    /// open, so a WAL can name lower epochs than the header). The migration
    /// continues from the highest.
    const RELEASED: [(&str, &str, u64); 7] = [
        // Header 7, catalog 8, no WAL.
        ("0.5.43", "closed.grafeo", 8),
        // Header 7, catalog 9, the WAL logs epochs 1 to 8 of the second process.
        ("0.5.43", "unflushed.grafeo", 9),
        // No header, the WAL logs epochs 1 to 9, then 1 to 8.
        ("0.5.43", "directory", 9),
        // Header and catalog 8.
        ("0.5.44", "closed.grafeo", 8),
        // Header and catalog 9, the WAL logs epochs 1 to 8 of the second process.
        ("0.5.44", "unflushed.grafeo", 9),
        ("0.5.44", "directory", 9),
        ("0.5.44", "compacted.grafeo", 17),
    ];

    /// `compacted.grafeo` also needs `compact-store`.
    #[test]
    fn a_migrated_database_continues_at_the_highest_epoch_0_5_stored() {
        for (version, name, highest) in RELEASED {
            if name == "compacted.grafeo" && !cfg!(feature = "compact-store") {
                continue;
            }
            let dir = tempfile::tempdir().unwrap();
            let path = copy_released(version, name, dir.path());

            let db = GrafeoDB::open_read_only(&path).unwrap();
            assert!(
                db.current_epoch() >= EpochId::new(highest),
                "{version}/{name}: read in place at {}, below epoch {highest}",
                db.current_epoch().as_u64()
            );
            db.close().unwrap();

            let db = GrafeoDB::open(&path).unwrap();
            let migrated = db.current_epoch();
            assert!(
                migrated >= EpochId::new(highest),
                "{version}/{name}: migrated at {}, below epoch {highest}",
                migrated.as_u64()
            );
            insert(&db, "Butch");
            let next = db.current_epoch();
            assert_eq!(next, EpochId::new(migrated.as_u64() + 1));
            db.close().unwrap();

            let db = GrafeoDB::open(&path).unwrap();
            assert_eq!(
                db.current_epoch(),
                next,
                "{version}/{name}: the migrated file keeps its epoch"
            );
            db.close().unwrap();
        }
    }

    /// The epoch of a 0.5.x database header counts on its own: here it is
    /// above the catalog's (8), as after a 0.5.x process whose transaction
    /// manager fell behind its store.
    #[test]
    fn a_migration_continues_at_the_epoch_of_the_0_5_database_header() {
        use grafeo_storage::file::header::{active_db_header, read_db_headers, write_db_header};

        let dir = tempfile::tempdir().unwrap();
        let path = copy_released("0.5.44", "closed.grafeo", dir.path());
        {
            let mut file = std::fs::OpenOptions::new()
                .read(true)
                .write(true)
                .open(&path)
                .unwrap();
            let (first, second) = read_db_headers(&mut file).unwrap();
            let (slot, mut header) = active_db_header(&first, &second);
            assert_eq!(header.epoch, 8, "the fixture's header");
            header.epoch = 88;
            write_db_header(&mut file, slot, &header).unwrap();
        }

        let db = GrafeoDB::open_read_only(&path).unwrap();
        assert!(
            db.current_epoch() >= EpochId::new(88),
            "read in place at {}",
            db.current_epoch().as_u64()
        );
        db.close().unwrap();

        let db = GrafeoDB::open(&path).unwrap();
        assert!(
            db.current_epoch() >= EpochId::new(88),
            "migrated at {}",
            db.current_epoch().as_u64()
        );
        db.close().unwrap();
    }
}
