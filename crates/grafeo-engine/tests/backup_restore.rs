//! Integration tests for incremental backup and point-in-time restore.
//!
//! Covers: full backup, incremental backup, restore to epoch, and the
//! backup chain model.
//!
//! ```bash
//! cargo test -p grafeo-engine --features full --test backup_restore
//! ```

#![cfg(all(feature = "wal", feature = "grafeo-file", feature = "lpg"))]

use grafeo_common::types::EpochId;
use grafeo_engine::GrafeoDB;

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
