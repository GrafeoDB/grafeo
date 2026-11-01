//! Errors keep their kind and code (0.6.1): a unique-key conflict between two
//! transactions is one retryable write conflict, not a conflict wrapped in a
//! conflict; a procedure argument that fails to evaluate keeps its data
//! exception and names the argument; a damaged incremental backup is
//! corruption naming its file, not an internal error; and an incremental
//! backup with nothing new since the last one is a successful no-op that
//! leaves the backup chain as it was. The tests pin kinds, codes and GQLSTATUS,
//! not the wording.
//!
//! ```bash
//! cargo test -p grafeo-engine --all-features --test error_kinds_of_conflicts_arguments_and_backups
//! ```

#![cfg(feature = "gql")]

use grafeo_common::utils::error::{Error, ErrorCode, TransactionError};
use grafeo_common::utils::gqlstatus::GqlStatus;
use grafeo_engine::GrafeoDB;

#[test]
fn a_unique_key_conflict_is_one_write_conflict() {
    let db = GrafeoDB::new_in_memory();
    db.execute("CREATE CONSTRAINT memory_key FOR (m:Memory) ON (m.key) UNIQUE")
        .unwrap();
    let mut first = db.session();
    first.begin_transaction().unwrap();
    first.execute("INSERT (:Memory {key: 'Alix'})").unwrap();
    let mut second = db.session();
    second.begin_transaction().unwrap();
    let error = second
        .execute("INSERT (:Memory {key: 'Alix'})")
        .expect_err("the key is claimed by the first transaction");
    assert!(
        matches!(
            error,
            Error::Transaction(TransactionError::WriteConflict(_))
        ),
        "{error:?}"
    );
    assert_eq!(
        error.error_code(),
        ErrorCode::TransactionConflict,
        "{error}"
    );
    assert!(error.error_code().is_retryable());
    let Error::Transaction(TransactionError::WriteConflict(detail)) = &error else {
        unreachable!()
    };
    // The detail is the conflict itself, not the text of another conflict.
    assert!(!detail.contains("GRAFEO-T001"), "{detail}");
    let text = error.to_string().to_lowercase();
    assert_eq!(text.matches("write conflict").count(), 1, "{error}");
    second.rollback().unwrap();
    first.commit().unwrap();
}

#[cfg(feature = "algos")]
mod procedure_arguments {
    use super::*;

    fn refused(db: &GrafeoDB, query: &str) -> Error {
        db.execute(query).expect_err(query)
    }

    #[test]
    fn a_division_by_zero_in_an_argument_keeps_its_status_and_names_the_argument() {
        let db = GrafeoDB::new_in_memory();
        db.execute("INSERT (:P {name: 'Alix'})-[:KNOWS]->(:P {name: 'Gus'})")
            .unwrap();
        let error = refused(&db, "CALL grafeo.pagerank({damping: 1 / 0})");
        assert_eq!(
            GqlStatus::from(&error),
            GqlStatus::DATA_DIVISION_BY_ZERO,
            "{error}"
        );
        assert!(error.to_string().contains("damping"), "{error}");
    }

    #[test]
    fn a_failed_cast_in_an_argument_keeps_its_status_and_names_the_argument() {
        let db = GrafeoDB::new_in_memory();
        db.execute("INSERT (:P {name: 'Alix'})-[:KNOWS]->(:P {name: 'Gus'})")
            .unwrap();
        let error = refused(
            &db,
            "CALL grafeo.pagerank({damping: CAST('Alix' AS FLOAT64)})",
        );
        assert_eq!(
            GqlStatus::from(&error),
            GqlStatus::DATA_INVALID_CHARACTER_VALUE_FOR_CAST,
            "{error}"
        );
        assert!(error.to_string().contains("damping"), "{error}");
    }

    #[test]
    fn a_parameter_argument_that_divides_by_zero_keeps_its_status() {
        let db = GrafeoDB::new_in_memory();
        db.execute("INSERT (:P {name: 'Alix'})-[:KNOWS]->(:P {name: 'Gus'})")
            .unwrap();
        let params = std::collections::HashMap::from([(
            "zero".to_string(),
            grafeo_common::types::Value::Int64(0),
        )]);
        let error = db
            .execute_with_params("CALL grafeo.pagerank({damping: 1 / $zero})", params)
            .expect_err("1 / $zero");
        assert_eq!(
            GqlStatus::from(&error),
            GqlStatus::DATA_DIVISION_BY_ZERO,
            "{error}"
        );
        assert!(error.to_string().contains("damping"), "{error}");
    }
}

#[cfg(all(feature = "wal", feature = "grafeo-file"))]
mod backups {
    use std::path::Path;

    use super::*;
    use grafeo_common::types::{EpochId, Value};

    fn people(path: &Path) -> Vec<Value> {
        let db = GrafeoDB::open(path).unwrap();
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

    fn listing(dir: &Path) -> Vec<(String, Vec<u8>)> {
        let mut files: Vec<(String, Vec<u8>)> = std::fs::read_dir(dir)
            .unwrap()
            .map(|entry| entry.unwrap().path())
            .filter(|path| path.is_file())
            .map(|path| {
                (
                    path.file_name().unwrap().to_string_lossy().into_owned(),
                    std::fs::read(&path).unwrap(),
                )
            })
            .collect();
        files.sort();
        files
    }

    /// `segment` is the newest segment of the chain in `backups`.
    fn is_newest(backups: &Path, segment: &grafeo_engine::database::backup::BackupSegment) {
        let manifest = GrafeoDB::read_backup_manifest(backups).unwrap().unwrap();
        let newest = manifest.segments.last().unwrap();
        assert_eq!(
            (&segment.filename, segment.end_epoch, segment.checksum),
            (&newest.filename, newest.end_epoch, newest.checksum),
            "the newest segment of the chain"
        );
    }

    #[test]
    fn an_incremental_backup_with_nothing_new_is_a_no_op() {
        let dir = tempfile::tempdir().unwrap();
        let source = dir.path().join("source.grafeo");
        let backups = dir.path().join("backups");
        let db = GrafeoDB::open(&source).unwrap();
        db.execute("INSERT (:Person {name: 'Alix'})").unwrap();
        let full = db.backup_full(&backups).unwrap();
        let before = listing(&backups);

        // Nothing written since the full backup: success, the backup
        // directory is byte for byte what it was, and the call returns the
        // chain's newest segment, the full backup.
        let same = db
            .backup_incremental(&backups)
            .expect("an incremental backup with nothing new is no error");
        assert_eq!(listing(&backups), before, "the no-op wrote nothing");
        assert_eq!(same.filename, full.filename);
        is_newest(&backups, &same);
        let again = db
            .backup_incremental(&backups)
            .expect("and again, still nothing new");
        assert_eq!(listing(&backups), before);
        assert_eq!(again.filename, full.filename);

        // The chain goes on: a write, an incremental, and the restore holds it.
        db.execute("INSERT (:Person {name: 'Gus'})").unwrap();
        let segment = db.backup_incremental(&backups).unwrap();
        is_newest(&backups, &segment);
        // A no-op after an incremental returns that incremental.
        let after = listing(&backups);
        let same = db.backup_incremental(&backups).unwrap();
        assert_eq!(listing(&backups), after, "the no-op wrote nothing");
        assert_eq!(
            (&same.filename, same.end_epoch, same.checksum),
            (&segment.filename, segment.end_epoch, segment.checksum)
        );
        is_newest(&backups, &same);
        db.close().unwrap();
        let restored = dir.path().join("restored.grafeo");
        GrafeoDB::restore_to_epoch(&backups, segment.end_epoch, &restored).unwrap();
        assert_eq!(people(&restored), [Value::from("Alix"), Value::from("Gus")]);
        let latest = dir.path().join("latest.grafeo");
        GrafeoDB::restore_to_epoch(&backups, EpochId::new(u64::MAX), &latest).unwrap();
        assert_eq!(people(&latest), [Value::from("Alix"), Value::from("Gus")]);
    }

    #[test]
    fn a_damaged_incremental_backup_is_corruption_naming_its_file() {
        let dir = tempfile::tempdir().unwrap();
        let source = dir.path().join("source.grafeo");
        let backups = dir.path().join("backups");
        let db = GrafeoDB::open(&source).unwrap();
        db.execute("INSERT (:Person {name: 'Alix'})").unwrap();
        db.backup_full(&backups).unwrap();
        db.execute("INSERT (:Person {name: 'Gus'})").unwrap();
        let segment = db.backup_incremental(&backups).unwrap();
        db.close().unwrap();

        let path = backups.join(&segment.filename);
        let mut bytes = std::fs::read(&path).unwrap();
        let middle = bytes.len() / 2;
        bytes[middle] ^= 0x40;
        std::fs::write(&path, &bytes).unwrap();

        let restored = dir.path().join("restored.grafeo");
        let error = GrafeoDB::restore_to_epoch(&backups, segment.end_epoch, &restored)
            .expect_err("a damaged incremental backup is refused");
        assert!(matches!(error, Error::Corruption(_)), "{error:?}");
        assert_eq!(error.error_code(), ErrorCode::StorageCorrupted, "{error}");
        assert!(error.to_string().contains(&segment.filename), "{error}");
        assert!(!restored.exists(), "a refused restore writes no database");
    }

    /// The WAL directory of the database at `path`.
    fn wal_dir(path: &Path) -> std::path::PathBuf {
        let mut sidecar = path.as_os_str().to_owned();
        sidecar.push(".wal");
        std::path::PathBuf::from(sidecar)
    }

    #[test]
    fn a_removed_wal_segment_breaks_the_chain_with_its_own_code() {
        let dir = tempfile::tempdir().unwrap();
        let source = dir.path().join("source.grafeo");
        let backups = dir.path().join("backups");
        let db = GrafeoDB::open(&source).unwrap();
        db.execute("INSERT (:Person {name: 'Alix'})").unwrap();
        db.backup_full(&backups).unwrap();
        db.execute("INSERT (:Person {name: 'Gus'})").unwrap();

        // Remove the segment the next incremental backup starts at.
        let from = db
            .backup_cursor()
            .and_then(|cursor| cursor.wal_lsn)
            .expect("the full backup left a cursor");
        let segment = wal_dir(&source).join(grafeo_storage::wal::segment_file_name(from));
        assert!(segment.exists(), "{}", segment.display());
        std::fs::remove_file(&segment).unwrap();
        let before = listing(&backups);

        let error = db
            .backup_incremental(&backups)
            .expect_err("the chain cannot go on without the segment");
        assert_eq!(error.error_code().as_str(), "GRAFEO-S005", "{error}");
        assert!(!error.error_code().is_retryable(), "{error}");
        assert!(
            error.to_string().to_lowercase().contains("full backup"),
            "the message says to take a new full backup: {error}"
        );
        assert_eq!(listing(&backups), before, "a refused backup writes nothing");

        // A new full backup starts a chain that works again.
        db.backup_full(&backups).unwrap();
        db.execute("INSERT (:Person {name: 'Vincent'})").unwrap();
        let incremental = db.backup_incremental(&backups).unwrap();
        db.close().unwrap();
        let restored = dir.path().join("restored.grafeo");
        GrafeoDB::restore_to_epoch(&backups, incremental.end_epoch, &restored).unwrap();
        assert_eq!(
            people(&restored),
            [
                Value::from("Alix"),
                Value::from("Gus"),
                Value::from("Vincent")
            ]
        );
    }
}
