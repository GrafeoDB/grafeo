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
            if let Some(chain_end) = manifest.segments.last().and_then(|last| last.end_wal_lsn) {
                assert!(
                    cursor.wal_lsn.expect("current-format cursor has an LSN") <= chain_end,
                    "{phase}: cursor advanced beyond the manifest's log position"
                );
                if phase == "backup:cursor_directory_sync" {
                    assert_eq!(cursor.wal_lsn, Some(chain_end));
                }
            }
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
            if matches!(
                phase,
                "backup:incremental_sync"
                    | "backup:incremental_directory_sync"
                    | "backup:manifest_sync"
            ) {
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
            if let Some(end) = retried.end_wal_lsn {
                assert_eq!(db.backup_cursor().unwrap().wal_lsn, Some(end));
            }
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

    #[cfg(unix)]
    #[test]
    fn incremental_payload_directory_sync_failure_keeps_the_backup_retryable() {
        assert_publication_failure_is_retryable(
            Backup::Incremental,
            "backup:incremental_directory_sync",
        );
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

    /// Count actual barriers in fresh fixtures so earlier WAL failures cannot
    /// make a later backup look as though it skipped a directory sync.
    #[cfg(unix)]
    fn assert_existing_directory_has_one_parent_barrier(backup: Backup) {
        let mut directory_failures = 0;
        for count in 1..=64 {
            let dir = tempfile::tempdir().unwrap();
            let db = GrafeoDB::open(dir.path().join("source.grafeo")).unwrap();
            let backups = dir.path().join("backups");
            db.execute("INSERT (:Person {name: 'Base'})").unwrap();
            let full = db.backup_full(&backups).unwrap();
            let manifest_path = backups.join("backup_manifest.json");
            let cursor_path = db.wal().unwrap().dir().join("backup_cursor.meta");
            let prior_manifest = std::fs::read(&manifest_path).unwrap();
            let prior_cursor = std::fs::read(&cursor_path).unwrap();
            db.execute("INSERT (:Person {name: 'Committed'})").unwrap();
            db.execute("INSERT (:Person {name: 'Retry'})").unwrap();
            let result = with_failure_at(count, || match backup {
                Backup::Full => db.backup_full(&backups),
                Backup::Incremental => db.backup_incremental(&backups),
            });
            match result {
                Err(error) => {
                    let message = error.to_string();
                    assert!(
                        message.contains("injected failure at:"),
                        "{backup:?}: {error}"
                    );
                    if message.ends_with("injected failure at: backup:directory_sync") {
                        directory_failures += 1;
                        assert_eq!(std::fs::read(&manifest_path).unwrap(), prior_manifest);
                        assert_eq!(std::fs::read(&cursor_path).unwrap(), prior_cursor);
                        assert_eq!(db.backup_cursor().unwrap().backed_up_epoch, full.end_epoch);
                    }
                    let _ = db.close();
                }
                Ok(segment) => {
                    assert_eq!(
                        directory_failures, 1,
                        "{backup:?}: an existing backup directory needs exactly one parent barrier"
                    );
                    let cursor = db.backup_cursor().unwrap();
                    assert_eq!(cursor.backed_up_epoch, segment.end_epoch);
                    if let Some(end) = segment.end_wal_lsn {
                        assert_eq!(cursor.wal_lsn, Some(end));
                    }
                    db.close().unwrap();
                    assert_restore_reopens(&backups, segment.end_epoch);
                    return;
                }
            }
        }
        panic!("{backup:?}: existing-directory failure sweep exceeded 64 attempts");
    }

    #[cfg(unix)]
    #[test]
    fn an_existing_backup_directory_needs_one_parent_barrier_per_backup() {
        for backup in [Backup::Full, Backup::Incremental] {
            assert_existing_directory_has_one_parent_barrier(backup);
        }
    }

    /// Replay the first failure after creating `new`, then select the retry's
    /// barrier that repairs that existing directory entry before proceeding.
    #[cfg(unix)]
    fn nested_directory_failure_count(replay_first: Option<u64>) -> u64 {
        let phase = if replay_first.is_some() {
            "backup:directory_sync"
        } else {
            "backup:new_directory_sync"
        };
        let injected_phase = format!("injected failure at: {phase}");
        for count in 1..=64 {
            let dir = tempfile::tempdir().expect("create source directory");
            let db_path = dir.path().join("source.grafeo");
            let new_directory = dir.path().join("new");
            let backups = new_directory.join("nested").join("backups");
            let db = GrafeoDB::open(&db_path).expect("open source database");
            db.execute("INSERT (:Person {name: 'Base'})").unwrap();
            db.execute("INSERT (:Person {name: 'Committed'})").unwrap();
            assert!(!backups.exists(), "fixture starts without the directory");

            if let Some(first_count) = replay_first {
                let error = with_failure_at(first_count, || db.backup_full(&backups))
                    .expect_err("replay the first newly-created-directory failure");
                assert!(
                    error
                        .to_string()
                        .ends_with("injected failure at: backup:new_directory_sync"),
                    "{error}"
                );
                assert!(
                    new_directory.is_dir(),
                    "first attempt created its first ancestor"
                );
                assert!(
                    !new_directory.join("nested").exists(),
                    "the first new directory is synced before creating its child"
                );
                assert!(!backups.join("backup_manifest.json").exists());
                assert!(db.backup_cursor().is_none());
            }

            let Err(error) = with_failure_at(count, || db.backup_full(&backups)) else {
                panic!(
                    "backup completed at count {count}, after {replay_first:?}, without {phase}"
                );
            };
            let message = error.to_string();
            if !message.contains(&injected_phase) {
                assert!(
                    message.contains("injected failure at:"),
                    "unexpected error selecting {phase}: {error}"
                );
                let _ = db.close();
                continue;
            }
            assert!(
                new_directory.is_dir(),
                "the created ancestor survives for retry"
            );
            assert!(
                !new_directory.join("nested").exists(),
                "repair the first directory before creating its child"
            );
            assert!(
                !backups.join("backup_manifest.json").exists(),
                "ancestor publication must precede the manifest"
            );
            assert!(db.backup_cursor().is_none(), "no backup has completed");
            assert!(!db.wal().unwrap().dir().join("backup_cursor.meta").exists());

            if replay_first.is_some() {
                db.wal_checkpoint()
                    .expect("checkpoint after both directory-sync failures");
                db.execute("INSERT (:Person {name: 'Retry'})").unwrap();
                let full = db.backup_full(&backups).expect("retry full backup");
                db.close().unwrap();
                assert_restore_reopens(&backups, full.end_epoch);
            } else {
                db.close().unwrap();
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

// Process exits at publication boundaries, not a simulation of power loss.
#[cfg(all(
    feature = "gql",
    feature = "wal",
    feature = "grafeo-file",
    feature = "testing-crash-injection"
))]
mod incremental_process_crash {
    use std::panic::AssertUnwindSafe;
    use std::path::Path;
    use std::process::Command;

    use grafeo_common::testing::child_process;
    use grafeo_common::testing::crash::{CrashResult, with_crash_at};
    use grafeo_common::types::{EpochId, Value};
    use grafeo_engine::GrafeoDB;
    use grafeo_engine::database::backup::read_backup_cursor;

    const POINT_VAR: &str = "GRAFEO_INCREMENTAL_BACKUP_CRASH_POINT";
    const DIR_VAR: &str = "GRAFEO_INCREMENTAL_BACKUP_CRASH_DIR";
    const CRASHED: i32 = 3;
    const BACKED_UP: i32 = 4;
    const PHASES: [&str; 3] = [
        "backup:after_incremental_file",
        "backup:after_incremental_manifest",
        "backup:after_incremental_cursor",
    ];

    fn assert_state(db: &GrafeoDB, count: usize) {
        let people = [
            vec![Value::from("Alix"), Value::from("Amsterdam")],
            vec![Value::from("Gus"), Value::from("Barcelona")],
            vec![Value::from("Vincent"), Value::from("Paris")],
        ];
        let edges = [
            vec![Value::from("Alix"), Value::from("Gus"), Value::Int64(2019)],
            vec![
                Value::from("Gus"),
                Value::from("Vincent"),
                Value::Int64(2020),
            ],
        ];
        assert_eq!(
            db.execute("MATCH (p:Person) RETURN p.name, p.city ORDER BY p.name")
                .unwrap()
                .rows(),
            &people[..count],
        );
        assert_eq!(
            db.execute("MATCH (a:Person)-[k:KNOWS]->(b:Person) RETURN a.name, b.name, k.since ORDER BY k.since")
                .unwrap()
                .rows(),
            &edges[..count - 1],
        );
    }

    fn assert_restores(backups: &Path, epoch: EpochId, count: usize) {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("restored.grafeo");
        GrafeoDB::restore_to_epoch(backups, epoch, &path).unwrap();
        for _ in 0..2 {
            let db = GrafeoDB::open(&path).unwrap();
            assert_state(&db, count);
            db.close().unwrap();
        }
    }

    fn back_up_in_child(point: u64, directory: &Path) -> Option<String> {
        let output = child_process::output(
            Command::new(std::env::current_exe().unwrap())
                .args([
                    "--exact",
                    "incremental_process_crash::incremental_backup_child",
                    "--nocapture",
                ])
                .env(POINT_VAR, point.to_string())
                .env(DIR_VAR, directory),
        )
        .unwrap();
        let stderr = String::from_utf8_lossy(&output.stderr);
        match output.status.code() {
            Some(BACKED_UP) => None,
            Some(CRASHED) => Some(
                stderr
                    .lines()
                    .find_map(|line| line.split_once("crash injection at: "))
                    .map_or_else(
                        || panic!("child did not name its crash: {stderr}"),
                        |(_, phase)| phase.trim().to_owned(),
                    ),
            ),
            code => panic!("point={point}: child exited with {code:?}: {stderr}"),
        }
    }

    /// Invoked only by the bounded parent sweep; no database is dropped after
    /// injection, so recovery sees an interrupted process, not a clean close.
    /// Like the full-backup child, it does nothing without the parent's inputs.
    #[test]
    fn incremental_backup_child() {
        let (Ok(point), Some(directory)) = (std::env::var(POINT_VAR), std::env::var_os(DIR_VAR))
        else {
            return;
        };
        let point = point.parse().unwrap();
        let directory = std::path::PathBuf::from(directory);
        let db = GrafeoDB::open(directory.join("source.grafeo")).unwrap();
        let mut session = db.session();
        session.begin_transaction().unwrap();
        session
            .execute("INSERT (:Person {name: 'Gus', city: 'Barcelona'})")
            .unwrap();
        session
            .execute("MATCH (a:Person {name: 'Alix'}), (b:Person {name: 'Gus'}) INSERT (a)-[:KNOWS {since: 2019}]->(b)")
            .unwrap();
        session.commit().unwrap();
        drop(session);
        let backups = directory.join("backups");
        let borrowed_db = AssertUnwindSafe(&db);
        match with_crash_at(point, move || borrowed_db.backup_incremental(&backups)) {
            CrashResult::Completed(Ok(_)) => std::process::exit(BACKED_UP),
            CrashResult::Completed(Err(error)) => panic!("incremental backup failed: {error}"),
            _ => std::process::exit(CRASHED),
        }
    }

    /// Publication may leave an unlisted file or a manifest ahead of the
    /// cursor. Neither can lose committed data when checkpointing and retrying.
    #[test]
    fn a_crash_during_an_incremental_backup_keeps_the_chain_and_cursor_recoverable() {
        let mut reached = Vec::new();
        for point in 1..=64 {
            let directory = tempfile::tempdir().unwrap();
            let path = directory.path().join("source.grafeo");
            let backups = directory.path().join("backups");
            let db = GrafeoDB::open(&path).unwrap();
            db.execute("INSERT (:Person {name: 'Alix', city: 'Amsterdam'})")
                .unwrap();
            let full = db.backup_full(&backups).unwrap();
            let wal_dir = db.wal().unwrap().dir().to_path_buf();
            let cursor_path = wal_dir.join("backup_cursor.meta");
            let manifest_path = backups.join("backup_manifest.json");
            let prior_cursor = std::fs::read(&cursor_path).unwrap();
            let prior_manifest = std::fs::read(&manifest_path).unwrap();
            db.close().unwrap();

            let phase = back_up_in_child(point, directory.path());
            let manifest = GrafeoDB::read_backup_manifest(&backups).unwrap().unwrap();
            let cursor = read_backup_cursor(&wal_dir).unwrap().unwrap();
            let last = manifest.segments.last().unwrap();
            let gus = EpochId::new(full.end_epoch.as_u64() + 1);
            // WAL rotation has its own earlier crash points. They leave the
            // previous manifest and cursor in place, before publication starts.
            let manifest_published = phase.is_none()
                || matches!(phase.as_deref(), Some(name) if name == PHASES[1] || name == PHASES[2]);
            let cursor_published = phase.is_none() || phase.as_deref() == Some(PHASES[2]);
            assert_eq!(
                last.end_epoch,
                if manifest_published {
                    gus
                } else {
                    full.end_epoch
                },
                "point={point}, phase={phase:?}"
            );
            assert_eq!(
                cursor.backed_up_epoch,
                if cursor_published {
                    gus
                } else {
                    full.end_epoch
                },
                "point={point}, phase={phase:?}"
            );
            assert!(cursor.backed_up_epoch <= last.end_epoch);
            assert!(cursor.wal_lsn.unwrap() <= last.end_wal_lsn.unwrap());
            if !manifest_published {
                assert_eq!(std::fs::read(&manifest_path).unwrap(), prior_manifest);
            }
            if cursor_published {
                assert_eq!(cursor.wal_lsn, last.end_wal_lsn);
            } else {
                assert_eq!(std::fs::read(&cursor_path).unwrap(), prior_cursor);
                if manifest_published {
                    assert!(cursor.wal_lsn.unwrap() < last.end_wal_lsn.unwrap());
                }
            }
            assert_restores(&backups, full.end_epoch, 1);
            if manifest_published {
                assert_restores(&backups, gus, 2);
            }

            let db = GrafeoDB::open(&path).unwrap();
            assert_eq!(db.current_epoch(), gus);
            assert_state(&db, 2);
            // Checkpoint must retain the WAL needed by the last acknowledged
            // cursor, including when the manifest has advanced without it.
            let before_checkpoint = std::fs::read(&cursor_path).unwrap();
            db.wal_checkpoint().unwrap();
            assert_eq!(std::fs::read(&cursor_path).unwrap(), before_checkpoint);
            let mut session = db.session();
            session.begin_transaction().unwrap();
            session
                .execute("INSERT (:Person {name: 'Vincent', city: 'Paris'})")
                .unwrap();
            session
                .execute("MATCH (a:Person {name: 'Gus'}), (b:Person {name: 'Vincent'}) INSERT (a)-[:KNOWS {since: 2020}]->(b)")
                .unwrap();
            session.commit().unwrap();
            drop(session);
            let vincent = db.current_epoch();
            let retried = db.backup_incremental(&backups).unwrap();
            assert_eq!(
                retried.start_epoch.as_u64(),
                cursor.backed_up_epoch.as_u64() + 1
            );
            assert_eq!(retried.end_epoch, vincent);
            let manifest = GrafeoDB::read_backup_manifest(&backups).unwrap().unwrap();
            let last = manifest.segments.last().unwrap();
            let cursor = db.backup_cursor().unwrap();
            assert_eq!(cursor.backed_up_epoch, vincent);
            assert_eq!(cursor.wal_lsn, last.end_wal_lsn);
            if phase.as_deref() == Some(PHASES[1]) {
                assert_eq!(manifest.segments.len(), 3);
                assert_eq!(
                    retried.start_epoch, gus,
                    "retry must cover the unpublished cursor range"
                );
            }
            db.close().unwrap();
            for (epoch, count) in [(full.end_epoch, 1), (gus, 2), (vincent, 3)] {
                assert_restores(&backups, epoch, count);
            }
            for _ in 0..2 {
                let db = GrafeoDB::open(&path).unwrap();
                assert_state(&db, 3);
                db.close().unwrap();
            }
            if let Some(phase) = phase {
                if phase.starts_with("backup:") {
                    reached.push(phase);
                } else {
                    assert!(phase.starts_with("wal:"), "unexpected crash point: {phase}");
                }
            } else {
                assert_eq!(
                    reached, PHASES,
                    "the sweep must exercise all three publication boundaries"
                );
                return;
            }
        }
        panic!("incremental crash sweep exceeded 64 child processes: {reached:?}");
    }
}
