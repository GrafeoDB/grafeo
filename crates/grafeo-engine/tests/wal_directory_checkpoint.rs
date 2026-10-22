//! A 0.5.x WAL-directory database has no snapshot file: its WAL is the only
//! copy of the data (#419). Loading one replays every WAL file even when an
//! older version left a `checkpoint.meta` behind, which would otherwise make
//! recovery skip the files below its sequence. Since 0.6 a WAL directory is
//! only loaded (read in place, or migrated to a single file), never written.
//!
//! ```bash
//! cargo test -p grafeo-engine --features full --test wal_directory_checkpoint
//! ```

#![cfg(all(
    feature = "lpg",
    feature = "gql",
    feature = "wal",
    feature = "grafeo-file"
))]

use std::path::{Path, PathBuf};

use grafeo_common::types::{EpochId, NodeId, TransactionId, Value};
use grafeo_engine::GrafeoDB;
use grafeo_storage::wal::{WalConfig, WalManager, WalRecord};

fn wal_files(wal_dir: &Path) -> Vec<PathBuf> {
    let mut files: Vec<PathBuf> = std::fs::read_dir(wal_dir)
        .unwrap()
        .map(|entry| entry.unwrap().path())
        .filter(|path| path.extension().is_some_and(|ext| ext == "log"))
        .collect();
    files.sort();
    files
}

fn count(db: &GrafeoDB, label: &str) -> Value {
    db.execute(&format!("MATCH (n:{label}) RETURN count(n)"))
        .unwrap()
        .rows()[0][0]
        .clone()
}

/// An older version wrote `checkpoint.meta` in directory mode. Loading the
/// directory ignores it: a read-only open, and the read-write open that
/// migrates it, find every committed node, and so does the migrated file.
#[test]
fn loading_a_0_5_directory_ignores_its_checkpoint_metadata() {
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

    let db = GrafeoDB::open_read_only(&path).unwrap();
    assert_eq!(count(&db, "Rotated"), Value::Int64(10), "read-only");
    db.close().unwrap();
    drop(db);

    let db = GrafeoDB::open(&path).unwrap();
    assert_eq!(count(&db, "Rotated"), Value::Int64(10), "migrated");
    db.close().unwrap();
    drop(db);
    let db = GrafeoDB::open(&path).unwrap();
    assert_eq!(
        count(&db, "Rotated"),
        Value::Int64(10),
        "the migrated file reopened"
    );
    db.close().unwrap();
}
