//! Caller controls for durable full and incremental backup publication.
//!
//! Injected I/O failures check ordering and retry behavior; these tests do not
//! simulate loss of the operating system's page cache or prove power-loss safety.

#![cfg(all(
    feature = "gql",
    feature = "wal",
    feature = "grafeo-file",
    feature = "testing-crash-injection"
))]
#![allow(
    missing_docs,
    reason = "integration tests exercise public backup callers"
)]

use std::path::Path;

use grafeo_common::types::{EpochId, Value};
use grafeo_engine::GrafeoDB;

/// The names of the people in the database restored from `backups` to
/// `epoch`.
fn restored_people(backups: &Path, epoch: EpochId) -> Vec<Value> {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("restored.grafeo");
    GrafeoDB::restore_to_epoch(backups, epoch, &path).unwrap();
    let db = GrafeoDB::open(&path).unwrap();
    let names = db
        .execute("MATCH (p:Person) RETURN p.name AS name ORDER BY name")
        .unwrap()
        .rows()
        .iter()
        .map(|row| row[0].clone())
        .collect();
    db.close().unwrap();
    names
}

fn names(values: &[&str]) -> Vec<Value> {
    values.iter().map(|value| Value::from(*value)).collect()
}

mod durable_publication {
    use super::*;
    use grafeo_common::testing::crash::with_failure_at;

    #[derive(Clone, Copy, Debug)]
    enum Backup {
        Full,
        Incremental,
    }

    fn assert_people(db: &GrafeoDB, expected: &[&str]) {
        let result = db
            .execute("MATCH (p:Person) RETURN p.name AS name ORDER BY name")
            .expect("query committed people");
        let actual: Vec<_> = result.rows().iter().map(|row| row[0].clone()).collect();
        assert_eq!(actual, names(expected));
    }

    fn assert_restore_reopens(backups: &Path, epoch: EpochId) {
        let dir = tempfile::tempdir().expect("create restore directory");
        let path = dir.path().join("restored.grafeo");
        GrafeoDB::restore_to_epoch(backups, epoch, &path).expect("restore retried backup");
        for _ in 0..2 {
            let restored = GrafeoDB::open(&path).expect("open restored database");
            assert_people(&restored, &["Base", "Committed", "Retry"]);
            restored.close().expect("close restored database");
        }
    }

    /// Select a named failure rather than assuming a fixed position among the
    /// WAL's own I/O hooks. Every count starts from a fresh, identical fixture.
    fn assert_publication_failure_is_retryable(backup: Backup, phase: &str) {
        let injected_phase = format!("injected failure at: {phase}");
        for count in 1..=64 {
            let dir = tempfile::tempdir().expect("create source directory");
            let db_path = dir.path().join("source.grafeo");
            let backups = dir.path().join("backups");
            let db = GrafeoDB::open(&db_path).expect("open source database");
            db.execute("INSERT (:Person {name: 'Base'})")
                .expect("commit base row");
            let full = db.backup_full(&backups).expect("take initial full backup");
            let manifest_path = backups.join("backup_manifest.json");
            let cursor_path = db
                .wal()
                .expect("source has a WAL")
                .dir()
                .join("backup_cursor.meta");
            let prior_manifest = std::fs::read(&manifest_path).expect("read prior manifest");
            let prior_cursor = std::fs::read(&cursor_path).expect("read prior cursor");
            let prior_epoch = db
                .backup_cursor()
                .expect("initial backup published a cursor")
                .backed_up_epoch;
            assert_eq!(prior_epoch, full.end_epoch);
            db.execute("INSERT (:Person {name: 'Committed'})")
                .expect("commit row before failing backup");

            let result = with_failure_at(count, || match backup {
                Backup::Full => db.backup_full(&backups),
                Backup::Incremental => db.backup_incremental(&backups),
            });
            let Err(error) = result else {
                panic!(
                    "{backup:?} backup completed at failure count {count} without reaching \
                     the required durability barrier {phase}"
                );
            };
            let message = error.to_string();
            if !message.contains(&injected_phase) {
                assert!(
                    message.contains("injected failure at:"),
                    "unexpected error while selecting {phase} at count {count}: {error}"
                );
                // An earlier WAL fault can make that fixture unavailable. It
                // must not count as exercising the requested backup barrier.
                let _ = db.close();
                continue;
            }

            let manifest = GrafeoDB::read_backup_manifest(&backups)
                .expect("published manifest remains readable")
                .expect("prior full backup remains listed");
            let manifest_end = manifest
                .epoch_range()
                .expect("published manifest contains a full backup")
                .1;
            let cursor = db
                .backup_cursor()
                .expect("published cursor remains readable");
            assert!(
                cursor.backed_up_epoch <= manifest_end,
                "{phase}: cursor {:?} advanced beyond manifest {:?}",
                cursor.backed_up_epoch,
                manifest_end
            );
            if matches!(
                phase,
                "backup:manifest_directory_sync"
                    | "backup:cursor_sync"
                    | "backup:cursor_directory_sync"
            ) {
                assert!(
                    manifest_end > prior_epoch,
                    "{phase}: manifest rename must publish the newer epoch"
                );
                assert_eq!(
                    manifest.segments.len(),
                    2,
                    "{phase}: exactly one segment must follow the initial full backup"
                );
            }
            if phase == "backup:cursor_directory_sync" {
                assert_eq!(
                    cursor.backed_up_epoch, manifest_end,
                    "cursor rename must be visible before its directory sync"
                );
            }
            if matches!(phase, "backup:incremental_sync" | "backup:manifest_sync") {
                assert_eq!(
                    std::fs::read(&manifest_path).expect("read unchanged manifest"),
                    prior_manifest,
                    "{phase}: failure before manifest publication changed it"
                );
            }
            if phase != "backup:cursor_directory_sync" {
                assert_eq!(
                    std::fs::read(&cursor_path).expect("read unchanged cursor"),
                    prior_cursor,
                    "{phase}: cursor advanced before its publication barrier"
                );
                assert_eq!(cursor.backed_up_epoch, prior_epoch, "{phase}");
            }
            assert_eq!(
                restored_people(&backups, full.end_epoch),
                names(&["Base"]),
                "{phase}: the previously completed backup remains restorable"
            );
            // Checkpoint pruning must still retain the WAL needed from the
            // last published cursor, including after a failed backup.
            db.wal_checkpoint().expect("checkpoint after failed backup");
            // These rows were committed successfully; only their backup failed.
            assert_people(&db, &["Base", "Committed"]);
            db.execute("INSERT (:Person {name: 'Retry'})")
                .expect("commit a fresh row after backup failure");
            let retried = match backup {
                Backup::Full => db.backup_full(&backups),
                Backup::Incremental => db.backup_incremental(&backups),
            }
            .expect("retry backup without injection");
            assert_eq!(
                db.backup_cursor()
                    .expect("retry published cursor")
                    .backed_up_epoch,
                retried.end_epoch
            );
            db.close().expect("close source after successful retry");
            assert_restore_reopens(&backups, retried.end_epoch);
            return;
        }
        panic!("{backup:?} backup never reached named durability barrier {phase} in 64 attempts");
    }

    #[test]
    fn incremental_payload_sync_failure_keeps_the_backup_retryable() {
        assert_publication_failure_is_retryable(Backup::Incremental, "backup:incremental_sync");
    }

    #[test]
    fn incremental_manifest_sync_failure_keeps_the_backup_retryable() {
        assert_publication_failure_is_retryable(Backup::Incremental, "backup:manifest_sync");
    }

    #[cfg(unix)]
    #[test]
    fn incremental_manifest_directory_sync_failure_keeps_the_backup_retryable() {
        assert_publication_failure_is_retryable(
            Backup::Incremental,
            "backup:manifest_directory_sync",
        );
    }

    #[test]
    fn incremental_cursor_sync_failure_keeps_the_backup_retryable() {
        assert_publication_failure_is_retryable(Backup::Incremental, "backup:cursor_sync");
    }

    #[cfg(unix)]
    #[test]
    fn incremental_cursor_directory_sync_failure_keeps_the_backup_retryable() {
        assert_publication_failure_is_retryable(
            Backup::Incremental,
            "backup:cursor_directory_sync",
        );
    }

    #[test]
    fn full_manifest_sync_failure_keeps_the_backup_retryable() {
        assert_publication_failure_is_retryable(Backup::Full, "backup:manifest_sync");
    }

    #[cfg(unix)]
    #[test]
    fn full_manifest_directory_sync_failure_keeps_the_backup_retryable() {
        assert_publication_failure_is_retryable(Backup::Full, "backup:manifest_directory_sync");
    }

    #[test]
    fn full_cursor_sync_failure_keeps_the_backup_retryable() {
        assert_publication_failure_is_retryable(Backup::Full, "backup:cursor_sync");
    }

    #[cfg(unix)]
    #[test]
    fn full_cursor_directory_sync_failure_keeps_the_backup_retryable() {
        assert_publication_failure_is_retryable(Backup::Full, "backup:cursor_directory_sync");
    }

    /// The second sweep replays the selected first failure on each fresh
    /// fixture, then independently selects a failure in the retry. The retry
    /// must sync the directory chain even though create_dir_all now sees it.
    #[cfg(unix)]
    fn nested_directory_failure_count(replay_first: Option<u64>) -> u64 {
        let phase = "backup:directory_sync";
        let injected_phase = format!("injected failure at: {phase}");
        for count in 1..=64 {
            let dir = tempfile::tempdir().expect("create source directory");
            let db_path = dir.path().join("source.grafeo");
            let backups = dir.path().join("new").join("nested").join("backups");
            let db = GrafeoDB::open(&db_path).expect("open source database");
            db.execute("INSERT (:Person {name: 'Base'})")
                .expect("commit base row");
            db.execute("INSERT (:Person {name: 'Committed'})")
                .expect("commit second row");
            assert!(!backups.exists(), "fixture starts without the directory");

            if let Some(first_count) = replay_first {
                let error = with_failure_at(first_count, || db.backup_full(&backups))
                    .expect_err("replay the selected first directory-sync failure");
                assert!(error.to_string().contains(&injected_phase), "{error}");
                assert!(backups.is_dir(), "first attempt created the directories");
                assert!(
                    !backups.join("backup_manifest.json").exists(),
                    "first directory-sync failure must precede manifest publication"
                );
            }

            let Err(error) = with_failure_at(count, || db.backup_full(&backups)) else {
                panic!(
                    "backup completed at count {count}, after first failure {replay_first:?}, \
                     without the required directory barrier {phase}"
                );
            };
            let message = error.to_string();
            if !message.contains(&injected_phase) {
                assert!(
                    message.contains("injected failure at:"),
                    "unexpected error selecting {phase} at count {count}: {error}"
                );
                let _ = db.close();
                continue;
            }
            assert!(backups.is_dir(), "failed attempt created the directories");
            assert!(
                !backups.join("backup_manifest.json").exists(),
                "directory-sync failure must precede manifest publication"
            );
            assert!(db.backup_cursor().is_none(), "no backup has completed");

            if replay_first.is_some() {
                db.wal_checkpoint()
                    .expect("checkpoint after both directory-sync failures");
                db.execute("INSERT (:Person {name: 'Retry'})")
                    .expect("commit a fresh row after directory-sync failures");
                let full = db.backup_full(&backups).expect("retry full backup");
                db.close().expect("close source after successful retry");
                assert_restore_reopens(&backups, full.end_epoch);
            } else {
                db.close()
                    .expect("close source after first selected failure");
            }
            return count;
        }
        panic!("never reached {phase} after first failure {replay_first:?} in 64 attempts");
    }

    #[cfg(unix)]
    #[test]
    fn a_nested_backup_directory_is_synced_again_after_a_failed_publication() {
        let first_count = nested_directory_failure_count(None);
        nested_directory_failure_count(Some(first_count));
    }
}
