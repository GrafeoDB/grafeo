//! WAL-directory databases have no snapshot file: the WAL is the only copy of
//! the data (#419). `wal_checkpoint()` must not write checkpoint metadata or
//! delete WAL files there, and recovery must replay every WAL file even when
//! an older version left a `checkpoint.meta` behind.
//!
//! ```bash
//! cargo test -p grafeo-engine --features full --test wal_directory_checkpoint
//! ```

#![allow(missing_docs)]

#[cfg(feature = "wal")]
mod tests {
    use grafeo_common::types::{EpochId, NodeId, TransactionId, Value};
    use grafeo_engine::config::StorageFormat;
    use grafeo_engine::{Config, GrafeoDB};
    use grafeo_storage::wal::{WalConfig, WalManager, WalRecord};
    use std::path::{Path, PathBuf};

    fn dir_config(path: &Path) -> Config {
        Config::persistent(path).with_storage_format(StorageFormat::WalDirectory)
    }

    fn wal_files(wal_dir: &Path) -> Vec<PathBuf> {
        let mut files: Vec<PathBuf> = std::fs::read_dir(wal_dir)
            .unwrap()
            .map(|entry| entry.unwrap().path())
            .filter(|path| path.extension().is_some_and(|ext| ext == "log"))
            .collect();
        files.sort();
        files
    }

    fn names(db: &GrafeoDB, label: &str) -> Vec<Value> {
        let result = db
            .session()
            .execute(&format!("MATCH (n:{label}) RETURN n.name ORDER BY n.name"))
            .unwrap();
        result.rows().iter().map(|row| row[0].clone()).collect()
    }

    fn count(db: &GrafeoDB, label: &str) -> Value {
        let result = db
            .session()
            .execute(&format!("MATCH (n:{label}) RETURN count(n)"))
            .unwrap();
        result.rows()[0][0].clone()
    }

    #[test]
    fn wal_checkpoint_only_syncs_in_directory_mode() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("db");
        let wal_dir = path.join("wal");

        {
            let db = GrafeoDB::with_config(dir_config(&path)).unwrap();
            db.session()
                .execute("INSERT (:Person {name: 'Alix'}), (:Person {name: 'Gus'})")
                .unwrap();
            let files_before = wal_files(&wal_dir);

            db.wal_checkpoint().unwrap();

            assert!(
                !wal_dir.join("checkpoint.meta").exists(),
                "a checkpoint record makes recovery skip older WAL files"
            );
            assert_eq!(wal_files(&wal_dir), files_before, "no WAL file is deleted");

            db.session()
                .execute("INSERT (:Person {name: 'Vincent'})")
                .unwrap();
            db.close().unwrap();
        }

        let db = GrafeoDB::with_config(dir_config(&path)).unwrap();
        assert_eq!(
            names(&db, "Person"),
            vec![
                Value::String("Alix".into()),
                Value::String("Gus".into()),
                Value::String("Vincent".into()),
            ]
        );
        db.close().unwrap();
    }

    /// An older version wrote `checkpoint.meta` in directory mode. Recovery
    /// must ignore it, otherwise every WAL file below its sequence is skipped.
    #[test]
    fn recovery_ignores_checkpoint_metadata_in_directory_mode() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("db");
        let wal_dir = path.join("wal");

        {
            // Tiny log files so the committed nodes span several rotated files.
            let wal = WalManager::with_config(
                &wal_dir,
                WalConfig {
                    max_log_size: 100,
                    ..WalConfig::default()
                },
            )
            .unwrap();
            for i in 0..10u64 {
                wal.log(&WalRecord::CreateNode {
                    id: NodeId::new(i),
                    labels: vec!["Rotated".to_string()],
                })
                .unwrap();
                wal.log(&WalRecord::TransactionCommit {
                    transaction_id: TransactionId::new(i + 1),
                })
                .unwrap();
            }
            // Epoch 0 keeps every file on disk while the metadata still points
            // at the latest sequence, like a database whose truncation had not
            // caught up yet.
            wal.checkpoint(TransactionId::new(10), EpochId::new(0))
                .unwrap();
            wal.sync().unwrap();
        }
        assert!(wal_dir.join("checkpoint.meta").exists());
        assert!(wal_files(&wal_dir).len() > 2, "the WAL must have rotated");

        let db = GrafeoDB::with_config(dir_config(&path)).unwrap();
        assert_eq!(count(&db, "Rotated"), Value::Int64(10));
        db.close().unwrap();
    }
}
