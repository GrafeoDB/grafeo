//! Integration tests for incremental backup and point-in-time restore.
//!
//! Covers: full backup, incremental backup, restore to epoch, and the
//! backup chain model.
//!
//! ```bash
//! cargo test -p grafeo-engine --features full --test backup_restore
//! ```

#![cfg(all(feature = "wal", feature = "grafeo-file", feature = "lpg"))]

use std::path::{Path, PathBuf};

use grafeo_common::types::{EpochId, Value};
use grafeo_engine::GrafeoDB;
use grafeo_engine::database::backup::BackupKind;

// ── Full backup roundtrip ─────────────────────────────────────────

#[test]
fn full_backup_and_restore_to_epoch() {
    let dir = tempfile::tempdir().expect("create temp dir");
    let db_path = dir.path().join("source.grafeo");
    let backup_dir = dir.path().join("backups");
    let restore_path = dir.path().join("restored.grafeo");

    // Create and populate
    {
        let db = GrafeoDB::open(&db_path).expect("open");
        let session = db.session();
        session
            .execute("INSERT (:Person {name: 'Alix', batch: 1})")
            .expect("insert");
        session
            .execute("INSERT (:Person {name: 'Gus', batch: 1})")
            .expect("insert");
        db.close().expect("close");
    }

    // Take a full backup
    let db = GrafeoDB::open(&db_path).expect("reopen");
    let segment = db.backup_full(&backup_dir).expect("full backup");
    assert_eq!(segment.start_epoch, EpochId::new(0));
    assert!(segment.size_bytes > 0);

    let current_epoch = segment.end_epoch;
    db.close().expect("close");

    // Restore to the full backup epoch
    GrafeoDB::restore_to_epoch(&backup_dir, current_epoch, &restore_path)
        .expect("restore to epoch");

    // Verify restored data
    let restored = GrafeoDB::open(&restore_path).expect("open restored");
    assert_eq!(restored.node_count(), 2, "restored should have 2 nodes");
    let session = restored.session();
    let result = session
        .execute("MATCH (n:Person) RETURN n.name ORDER BY n.name")
        .unwrap();
    assert_eq!(result.rows().len(), 2);
    restored.close().expect("close");
}

// ── Full + incremental backup cycle ───────────────────────────────

#[test]
fn incremental_backup_captures_new_data() {
    let dir = tempfile::tempdir().expect("create temp dir");
    let db_path = dir.path().join("incr.grafeo");
    let backup_dir = dir.path().join("backups");

    let db = GrafeoDB::open(&db_path).expect("open");

    // Initial data
    let session = db.session();
    session
        .execute("INSERT (:Person {name: 'Alix'})")
        .expect("insert");

    // Full backup
    let full = db.backup_full(&backup_dir).expect("full backup");
    assert!(full.size_bytes > 0);

    // Add more data after the full backup
    session
        .execute("INSERT (:Person {name: 'Gus'})")
        .expect("insert");
    session
        .execute("INSERT (:Person {name: 'Vincent'})")
        .expect("insert");

    // Force a WAL rotation so incremental has new files to capture
    db.wal().expect("WAL").rotate().expect("rotate");
    session
        .execute("INSERT (:Person {name: 'Jules'})")
        .expect("insert");

    // Incremental backup
    let incr = db
        .backup_incremental(&backup_dir)
        .expect("incremental backup");
    assert!(incr.size_bytes > 0);
    assert!(incr.start_epoch > full.end_epoch);

    db.close().expect("close");

    // Verify manifest has both segments
    let manifest = GrafeoDB::read_backup_manifest(&backup_dir)
        .expect("read manifest")
        .expect("manifest exists");
    assert_eq!(manifest.segments.len(), 2);
}

// ── Backup cursor tracking ────────────────────────────────────────

#[test]
fn backup_cursor_updated_after_full_backup() {
    let dir = tempfile::tempdir().expect("create temp dir");
    let db_path = dir.path().join("cursor.grafeo");
    let backup_dir = dir.path().join("backups");

    let db = GrafeoDB::open(&db_path).expect("open");

    // Use a session to advance the epoch beyond 0
    let session = db.session();
    session.execute("INSERT (:Test {val: 1})").expect("insert");

    assert!(
        db.backup_cursor().is_none(),
        "no cursor before first backup"
    );

    db.backup_full(&backup_dir).expect("full backup");

    let cursor = db
        .backup_cursor()
        .expect("cursor should exist after backup");
    assert!(
        cursor.backed_up_epoch.as_u64() > 0,
        "epoch should be > 0 after session commit"
    );

    db.close().expect("close");
}

// ── Backup manifest metadata ──────────────────────────────────────

#[test]
fn backup_manifest_tracks_segments() {
    let dir = tempfile::tempdir().expect("create temp dir");
    let db_path = dir.path().join("meta.grafeo");
    let backup_dir = dir.path().join("backups");

    let db = GrafeoDB::open(&db_path).expect("open");
    db.create_node(&["Test"]).unwrap();

    let segment = db.backup_full(&backup_dir).expect("full backup");

    let manifest = GrafeoDB::read_backup_manifest(&backup_dir)
        .unwrap()
        .unwrap();
    assert_eq!(manifest.segments.len(), 1);
    assert_eq!(manifest.segments[0].filename, segment.filename);
    assert_eq!(manifest.epoch_range().unwrap().1, segment.end_epoch);

    db.close().expect("close");
}

// ── Error cases (all platforms) ───────────────────────────────────

#[test]
fn incremental_without_full_fails() {
    let dir = tempfile::tempdir().expect("create temp dir");
    let db_path = dir.path().join("nofull.grafeo");
    let backup_dir = dir.path().join("backups");

    let db = GrafeoDB::open(&db_path).expect("open");
    db.create_node(&["Test"]).unwrap();

    let result = db.backup_incremental(&backup_dir);
    assert!(result.is_err(), "incremental without full should fail");

    db.close().expect("close");
}

#[test]
fn restore_nonexistent_backup_fails() {
    let dir = tempfile::tempdir().expect("create temp dir");
    let backup_dir = dir.path().join("empty_backups");
    let restore_path = dir.path().join("restored.grafeo");

    let result = GrafeoDB::restore_to_epoch(&backup_dir, EpochId::new(100), &restore_path);
    assert!(result.is_err(), "restore from empty dir should fail");
}

// ── Bug regression tests ─────────────────────────────────────────

/// Regression: backup_full() on a read-only database should succeed.
///
/// The on-disk `.grafeo` file is already a valid snapshot, so there is
/// nothing to flush. Previously, backup_full() unconditionally called
/// checkpoint_to_file() which rejects writes on read-only file managers.
#[test]
fn backup_full_on_read_only_database() {
    let dir = tempfile::tempdir().expect("create temp dir");
    let db_path = dir.path().join("readonly_backup.grafeo");
    let backup_dir = dir.path().join("backups");

    // Create and populate a database, then close it so the file is complete.
    {
        let db = GrafeoDB::open(&db_path).expect("open");
        let session = db.session();
        session
            .execute("INSERT (:Person {name: 'Alix'})")
            .expect("insert");
        session
            .execute("INSERT (:Person {name: 'Gus'})")
            .expect("insert");
        db.close().expect("close");
    }

    // Re-open in read-only mode and take a full backup.
    let db = GrafeoDB::open_read_only(&db_path).expect("open read-only");
    let segment = db
        .backup_full(&backup_dir)
        .expect("backup_full on read-only should succeed");

    assert_eq!(segment.start_epoch, EpochId::new(0));
    assert!(segment.size_bytes > 0, "backup file should not be empty");

    // Verify the backup is a valid database by restoring and querying.
    let restore_path = dir.path().join("restored.grafeo");
    GrafeoDB::restore_to_epoch(&backup_dir, segment.end_epoch, &restore_path)
        .expect("restore should succeed");

    let restored = GrafeoDB::open(&restore_path).expect("open restored");
    assert_eq!(restored.node_count(), 2, "restored should have 2 nodes");
    restored.close().expect("close");
}

/// Regression: backup_full() must work on Windows where the .grafeo file
/// is held open with an exclusive lock.
///
/// Previously, do_backup_full() used std::fs::copy() which tries to open
/// the source file with a new handle. On Windows, that fails because the
/// GrafeoFileManager already holds an exclusive lock.
#[test]
fn backup_full_works_while_database_is_open() {
    let dir = tempfile::tempdir().expect("create temp dir");
    let db_path = dir.path().join("open_backup.grafeo");
    let backup_dir = dir.path().join("backups");

    let db = GrafeoDB::open(&db_path).expect("open");
    let session = db.session();
    session
        .execute("INSERT (:Person {name: 'Alix'})")
        .expect("insert");
    session
        .execute("INSERT (:Person {name: 'Gus'})")
        .expect("insert");

    // This should succeed on ALL platforms, including Windows.
    let segment = db
        .backup_full(&backup_dir)
        .expect("backup_full on open database should work on all platforms");

    assert_eq!(segment.start_epoch, EpochId::new(0));
    assert!(segment.size_bytes > 0);

    db.close().expect("close");

    // Verify the backup is restorable.
    let restore_path = dir.path().join("restored.grafeo");
    GrafeoDB::restore_to_epoch(&backup_dir, segment.end_epoch, &restore_path)
        .expect("restore should succeed");

    let restored = GrafeoDB::open(&restore_path).expect("open restored");
    assert_eq!(restored.node_count(), 2, "restored should have 2 nodes");
    restored.close().expect("close");
}

/// Two full backups into the same directory produce two distinct segments
/// in the manifest. The second backup does not overwrite the first.
#[test]
fn backup_full_twice_produces_two_segments() {
    let dir = tempfile::tempdir().expect("create temp dir");
    let db_path = dir.path().join("double.grafeo");
    let backup_dir = dir.path().join("backups");

    let db = GrafeoDB::open(&db_path).expect("open");
    let session = db.session();
    session
        .execute("INSERT (:Person {name: 'Alix'})")
        .expect("insert");

    let seg1 = db.backup_full(&backup_dir).expect("first backup");
    assert_eq!(seg1.filename, "backup_full_0000.grafeo");

    // Add more data between backups
    session
        .execute("INSERT (:Person {name: 'Gus'})")
        .expect("insert");

    let seg2 = db.backup_full(&backup_dir).expect("second backup");
    assert_eq!(seg2.filename, "backup_full_0001.grafeo");
    assert!(seg2.end_epoch >= seg1.end_epoch);

    let manifest = GrafeoDB::read_backup_manifest(&backup_dir)
        .unwrap()
        .unwrap();
    assert_eq!(manifest.segments.len(), 2);

    // Restore from the second backup (latest state)
    let restore_path = dir.path().join("restored.grafeo");
    GrafeoDB::restore_to_epoch(&backup_dir, seg2.end_epoch, &restore_path)
        .expect("restore should succeed");

    let restored = GrafeoDB::open(&restore_path).expect("open restored");
    assert_eq!(restored.node_count(), 2);
    restored.close().expect("close");

    db.close().expect("close");
}

/// Regression for GrafeoDB/grafeo#267: incremental backup must succeed
/// after a full backup without requiring a manual WAL rotation.
///
/// Previously, `do_backup_full` stored the active log file's sequence in the
/// cursor but did not rotate, so writes that landed in the same file were
/// invisible to incremental (which skips `seq <= cursor.log_sequence`).
#[test]
fn incremental_backup_works_without_manual_rotation() {
    let dir = tempfile::tempdir().expect("create temp dir");
    let db_path = dir.path().join("issue267.grafeo");
    let backup_dir = dir.path().join("backups");

    let db = GrafeoDB::open(&db_path).expect("open");
    let session = db.session();

    // Seed data and take a full backup
    session
        .execute("INSERT (:Person {name: 'Alix'})")
        .expect("insert");
    let full = db.backup_full(&backup_dir).expect("full backup");

    // Insert more data WITHOUT manually rotating the WAL
    session
        .execute("INSERT (:Person {name: 'Gus'})")
        .expect("insert");
    session
        .execute("INSERT (:Person {name: 'Vincent'})")
        .expect("insert");

    // Incremental must succeed (no manual wal.rotate() call)
    let incr = db
        .backup_incremental(&backup_dir)
        .expect("incremental backup should work without manual WAL rotation");
    assert!(incr.size_bytes > 0);
    assert!(incr.start_epoch > full.end_epoch);

    db.close().expect("close");
}

/// An incremental backup with nothing new to back up fails without touching
/// the WAL: polling for backups must not leave an empty log file each time.
#[test]
fn incremental_backup_without_new_records_leaves_the_wal_alone() {
    let dir = tempfile::tempdir().expect("create temp dir");
    let db_path = dir.path().join("idle.grafeo");
    let backup_dir = dir.path().join("backups");
    let wal_dir = dir.path().join("idle.grafeo.wal");
    let log_files = || {
        std::fs::read_dir(&wal_dir)
            .expect("read WAL directory")
            .filter(|entry| {
                entry
                    .as_ref()
                    .is_ok_and(|entry| entry.path().extension().is_some_and(|ext| ext == "log"))
            })
            .count()
    };

    let db = GrafeoDB::open(&db_path).expect("open");
    db.session()
        .execute("INSERT (:Person {name: 'Alix'})")
        .expect("insert");
    db.backup_full(&backup_dir).expect("full backup");
    let before = log_files();

    for _ in 0..3 {
        let err = db
            .backup_incremental(&backup_dir)
            .expect_err("nothing to back up");
        assert!(err.to_string().contains("no new WAL records"), "{err}");
    }
    assert_eq!(log_files(), before);

    // The next write is still backed up.
    db.session()
        .execute("INSERT (:Person {name: 'Gus'})")
        .expect("insert");
    db.backup_incremental(&backup_dir)
        .expect("incremental after a write");
    db.close().expect("close");
}

/// Regression for GrafeoDB/grafeo#267: two consecutive incremental backups
/// with writes between them must both succeed.
///
/// This tests the same boundary condition in `do_backup_incremental` itself:
/// the cursor it writes must not block the next incremental from seeing new
/// WAL data.
#[test]
fn multiple_incremental_backups_in_sequence() {
    let dir = tempfile::tempdir().expect("create temp dir");
    let db_path = dir.path().join("multi_incr.grafeo");
    let backup_dir = dir.path().join("backups");

    let db = GrafeoDB::open(&db_path).expect("open");
    let session = db.session();

    // Seed + full backup
    session
        .execute("INSERT (:Person {name: 'Alix'})")
        .expect("insert");
    db.backup_full(&backup_dir).expect("full backup");

    // First incremental
    session
        .execute("INSERT (:Person {name: 'Gus'})")
        .expect("insert");
    let incr1 = db
        .backup_incremental(&backup_dir)
        .expect("first incremental");

    // Second incremental (no manual rotation between them)
    session
        .execute("INSERT (:Person {name: 'Vincent'})")
        .expect("insert");
    let incr2 = db
        .backup_incremental(&backup_dir)
        .expect("second incremental should also succeed");
    assert!(incr2.start_epoch > incr1.start_epoch);

    let manifest = GrafeoDB::read_backup_manifest(&backup_dir)
        .unwrap()
        .unwrap();
    assert_eq!(
        manifest.segments.len(),
        3,
        "should have 1 full + 2 incremental segments"
    );

    db.close().expect("close");
}

// ── Backups while other sessions write ────────────────────────────

/// Records written while a backup ran belonged to neither the backup nor the
/// next one: a full backup copied the file of its checkpoint and then took
/// the WAL file active at that point as backed up, and an incremental backup
/// read the active WAL file and rotated only afterwards. A restore then
/// missed those records.
#[test]
fn backups_during_writes_lose_nothing() {
    use std::sync::Arc;

    const NODES: i64 = 3000;

    let dir = tempfile::tempdir().expect("create temp dir");
    let db_path = dir.path().join("busy.grafeo");
    let backup_dir = dir.path().join("backups");
    let restore_path = dir.path().join("restored.grafeo");

    let db = Arc::new(GrafeoDB::open(&db_path).expect("open"));
    db.session()
        .execute("INSERT (:N {i: -1})")
        .expect("seed insert");

    let writer = {
        let db = Arc::clone(&db);
        std::thread::spawn(move || {
            let session = db.session();
            for i in 0..NODES {
                session
                    .execute(&format!("INSERT (:N {{i: {i}}})"))
                    .expect("insert");
            }
        })
    };

    db.backup_full(&backup_dir).expect("full backup");
    // Until the writer is done, or has panicked: `join()` below reports that.
    while !writer.is_finished() {
        // "no new WAL records" is fine while the writer is between inserts
        let _ = db.backup_incremental(&backup_dir);
    }
    writer.join().expect("writer thread");
    // One more write, so the final incremental has content and its epoch is
    // past every insert above.
    db.session()
        .execute("INSERT (:N {i: -2})")
        .expect("last insert");
    let last = db
        .backup_incremental(&backup_dir)
        .expect("final incremental");

    GrafeoDB::restore_to_epoch(&backup_dir, last.end_epoch, &restore_path).expect("restore");
    let restored = GrafeoDB::open(&restore_path).expect("open restored");
    let count = restored
        .session()
        .execute("MATCH (n:N) RETURN count(n) AS c")
        .expect("count")
        .rows()[0][0]
        .clone();
    assert_eq!(count, grafeo_common::types::Value::Int64(NODES + 2));
    restored.close().expect("close restored");
    db.close().expect("close");
}

// ── Backups taken by 0.5.x ────────────────────────────────────────

/// A copy of the backup chain that 0.5.44 wrote (`fixtures/backups/0.5.44`,
/// see its README): a full backup at epoch 3 and incremental backups of
/// epoch 4 and of epochs 5 to 7, under a bincode manifest. The copy keeps the
/// committed files as the release wrote them; the manifest is committed as
/// `backup_manifest.bincode` (text hooks would append a newline to a `.json`)
/// and copied back under its real name.
fn backups_of_0_5_44(dir: &Path) -> PathBuf {
    let source = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/backups/0.5.44");
    let backups = dir.join("backups");
    std::fs::create_dir_all(&backups).expect("create backup directory");
    for entry in std::fs::read_dir(&source).expect("read fixture") {
        let entry = entry.expect("fixture entry");
        let name = if entry.file_name() == "backup_manifest.bincode" {
            "backup_manifest.json".into()
        } else {
            entry.file_name()
        };
        std::fs::copy(entry.path(), backups.join(name)).expect("copy fixture");
    }
    backups
}

/// The people (name and city, by name) and the `KNOWS` edges (from, to and
/// since, by since) of the database at `path`.
fn people_and_friendships(path: &Path) -> (Vec<Vec<Value>>, Vec<Vec<Value>>) {
    let db = GrafeoDB::open(path).unwrap_or_else(|e| panic!("open {}: {e}", path.display()));
    let rows = |query: &str| {
        db.execute(query)
            .unwrap_or_else(|e| panic!("{query}: {e}"))
            .rows()
            .to_vec()
    };
    let people = rows("MATCH (p:Person) RETURN p.name, p.city ORDER BY p.name");
    let friendships = rows(
        "MATCH (a:Person)-[k:KNOWS]->(b:Person) RETURN a.name, b.name, k.since ORDER BY k.since",
    );
    db.close().expect("close");
    (people, friendships)
}

fn person(name: &str, city: &str) -> Vec<Value> {
    vec![Value::from(name), Value::from(city)]
}

fn knows(from: &str, to: &str, since: i64) -> Vec<Value> {
    vec![Value::from(from), Value::from(to), Value::Int64(since)]
}

/// The kind, file name and epochs of every segment in the manifest.
fn segments(backups: &Path) -> Vec<(BackupKind, String, u64, u64)> {
    GrafeoDB::read_backup_manifest(backups)
        .expect("read manifest")
        .expect("a manifest")
        .segments
        .iter()
        .map(|s| {
            (
                s.kind,
                s.filename.clone(),
                s.start_epoch.as_u64(),
                s.end_epoch.as_u64(),
            )
        })
        .collect()
}

/// The segments of the 0.5.44 chain, as its manifest lists them.
fn segments_of_0_5_44() -> Vec<(BackupKind, String, u64, u64)> {
    vec![
        (
            BackupKind::Full,
            "backup_full_0000.grafeo".to_string(),
            0,
            3,
        ),
        (
            BackupKind::Incremental,
            "backup_incr_0001.wal".to_string(),
            4,
            4,
        ),
        (
            BackupKind::Incremental,
            "backup_incr_0002.wal".to_string(),
            5,
            7,
        ),
    ]
}

/// A backup chain taken by 0.5.44, under its bincode manifest, restores to
/// each epoch it covers: the full backup alone, and with its incremental
/// segments replayed up to the epoch.
#[test]
fn a_backup_chain_taken_by_0_5_44_still_restores() {
    let dir = tempfile::tempdir().expect("create temp dir");
    let backups = backups_of_0_5_44(dir.path());
    assert_eq!(segments(&backups), segments_of_0_5_44());

    let restore = |epoch: u64| {
        let path = dir.path().join(format!("restored_at_{epoch}.grafeo"));
        GrafeoDB::restore_to_epoch(&backups, EpochId::new(epoch), &path)
            .unwrap_or_else(|e| panic!("restore to epoch {epoch}: {e}"));
        people_and_friendships(&path)
    };

    assert_eq!(
        restore(3),
        (
            vec![person("Alix", "Amsterdam"), person("Gus", "Berlin")],
            vec![knows("Alix", "Gus", 2019)]
        ),
        "the full backup alone"
    );
    assert_eq!(
        restore(4),
        (
            vec![
                person("Alix", "Amsterdam"),
                person("Gus", "Berlin"),
                person("Vincent", "Paris")
            ],
            vec![knows("Alix", "Gus", 2019)]
        ),
        "with the first incremental backup"
    );
    assert_eq!(
        restore(7),
        (
            vec![
                person("Alix", "Amsterdam"),
                person("Gus", "Barcelona"),
                person("Mia", "Prague"),
                person("Vincent", "Paris")
            ],
            vec![knows("Mia", "Alix", 1988), knows("Alix", "Gus", 2019)]
        ),
        "with both incremental backups"
    );
}

/// After an upgrade, the database restored from a 0.5.44 chain goes on
/// backing up into the same directory: the manifest is rewritten as JSON with
/// the 0.5.44 segments kept, and both the old and the new segments restore.
#[test]
fn a_backup_directory_of_0_5_44_takes_new_backups() {
    let dir = tempfile::tempdir().expect("create temp dir");
    let backups = backups_of_0_5_44(dir.path());
    let db_path = dir.path().join("upgraded.grafeo");
    GrafeoDB::restore_to_epoch(&backups, EpochId::new(7), &db_path).expect("restore to epoch 7");

    let db = GrafeoDB::open(&db_path).expect("open the restored 0.5.x database");
    db.execute("INSERT (:Person {name: 'Jules', city: 'Amsterdam'})")
        .expect("insert Jules");
    let full = db.backup_full(&backups).expect("full backup");
    db.execute("INSERT (:Person {name: 'Butch', city: 'Berlin'})")
        .expect("insert Butch");
    let incremental = db.backup_incremental(&backups).expect("incremental backup");
    db.close().expect("close");

    let manifest = std::fs::read(backups.join("backup_manifest.json")).expect("read manifest");
    let json: serde_json::Value =
        serde_json::from_slice(&manifest).expect("the rewritten manifest is JSON");
    assert_eq!(json["version"], 2, "{json}");

    let mut expected = segments_of_0_5_44();
    expected.push((
        BackupKind::Full,
        "backup_full_0003.grafeo".to_string(),
        0,
        full.end_epoch.as_u64(),
    ));
    expected.push((
        BackupKind::Incremental,
        "backup_incr_0004.wal".to_string(),
        incremental.start_epoch.as_u64(),
        incremental.end_epoch.as_u64(),
    ));
    assert_eq!(
        segments(&backups),
        expected,
        "the 0.5.44 segments are kept, the new ones appended"
    );

    let restore = |epoch: EpochId| {
        let path = dir
            .path()
            .join(format!("restored_at_{}.grafeo", epoch.as_u64()));
        GrafeoDB::restore_to_epoch(&backups, epoch, &path)
            .unwrap_or_else(|e| panic!("restore to epoch {epoch:?}: {e}"));
        people_and_friendships(&path).0
    };
    assert_eq!(
        restore(EpochId::new(4)),
        vec![
            person("Alix", "Amsterdam"),
            person("Gus", "Berlin"),
            person("Vincent", "Paris")
        ],
        "a 0.5.44 segment still restores from the rewritten manifest"
    );
    assert_eq!(
        restore(incremental.end_epoch),
        vec![
            person("Alix", "Amsterdam"),
            person("Butch", "Berlin"),
            person("Gus", "Barcelona"),
            person("Jules", "Amsterdam"),
            person("Mia", "Prague"),
            person("Vincent", "Paris")
        ],
        "the new segments restore"
    );
}
