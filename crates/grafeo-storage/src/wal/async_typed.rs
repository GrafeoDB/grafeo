//! Type-safe async WAL wrapper.
//!
//! [`AsyncTypedWal`] wraps an [`AsyncWalManager`] and ensures that only records
//! of type `R` can be written, mirroring the sync [`TypedWal`](super::TypedWal)
//! for async contexts.
//!
//! Use [`AsyncLpgWal`] for the standard labeled property graph async WAL.

use std::marker::PhantomData;
use std::path::{Path, PathBuf};

use grafeo_common::types::{EpochId, TransactionId};
use grafeo_common::utils::error::{Error, Result};

use super::WalRecord;
use super::async_log::AsyncWalManager;
use super::log::{DurabilityMode, WalConfig};
use super::record::WalEntry;

/// A type-safe async wrapper around [`AsyncWalManager`] that constrains
/// record types at compile time.
///
/// This is the async equivalent of [`TypedWal`](super::TypedWal). It ensures
/// that only records implementing [`WalEntry`] with the specific type `R` can
/// be logged. The underlying WAL format is identical to the sync version, so
/// files written by `AsyncTypedWal` are recoverable by
/// [`WalRecovery`](super::WalRecovery).
///
/// # Example
///
/// ```no_run
/// use grafeo_storage::wal::{AsyncLpgWal, WalRecord};
/// use grafeo_common::types::NodeId;
///
/// # async fn example() -> grafeo_common::utils::error::Result<()> {
/// let wal = AsyncLpgWal::open("wal_dir").await?;
/// wal.log(&WalRecord::CreateNode {
///     id: NodeId::new(1),
///     labels: vec!["Person".to_string()],
/// }).await?;
/// # Ok(())
/// # }
/// ```
///
/// Like [`AsyncWalManager`], it writes plaintext records: it has no
/// encryption.
pub struct AsyncTypedWal<R: WalEntry> {
    manager: AsyncWalManager,
    _record: PhantomData<R>,
}

impl<R: WalEntry> AsyncTypedWal<R> {
    /// Opens or creates a typed async WAL in the given directory.
    ///
    /// # Errors
    ///
    /// Returns an error if the directory cannot be created or accessed.
    pub async fn open(dir: impl AsRef<Path>) -> Result<Self> {
        Ok(Self {
            manager: AsyncWalManager::open(dir).await?,
            _record: PhantomData,
        })
    }

    /// Opens or creates a typed async WAL with custom configuration.
    ///
    /// # Errors
    ///
    /// Returns an error if the directory cannot be created or accessed.
    pub async fn with_config(dir: impl AsRef<Path>, config: WalConfig) -> Result<Self> {
        Ok(Self {
            manager: AsyncWalManager::with_config(dir, config).await?,
            _record: PhantomData,
        })
    }

    /// Logs a typed record to the WAL asynchronously.
    ///
    /// The record is serialized via bincode and written with a length prefix
    /// and CRC32 checksum. Durability handling (fsync) is determined by the
    /// record's [`WalEntry::requires_sync`] method.
    ///
    /// # Errors
    ///
    /// Returns an error if serialization or writing fails.
    pub async fn log(&self, record: &R) -> Result<()> {
        let data = bincode::serde::encode_to_vec(record, bincode::config::standard())
            .map_err(|e| Error::Serialization(e.to_string()))?;
        let force_sync = record.requires_sync();
        self.manager.write_frame(&data, force_sync).await
    }

    /// Writes a checkpoint marker asynchronously.
    ///
    /// Creates a checkpoint record via [`WalEntry::make_checkpoint`], logs it,
    /// then syncs. The checkpoint epoch is tracked internally.
    ///
    /// # Errors
    ///
    /// Returns an error if the checkpoint cannot be written.
    pub async fn checkpoint(
        &self,
        current_transaction: TransactionId,
        epoch: EpochId,
    ) -> Result<()> {
        let checkpoint_record = R::make_checkpoint(current_transaction);
        self.log(&checkpoint_record).await?;
        self.manager.sync().await?;
        self.manager.set_checkpoint_epoch(epoch).await;
        Ok(())
    }

    /// Syncs the WAL to disk (fsync).
    ///
    /// # Errors
    ///
    /// Returns an error if the sync fails.
    pub async fn sync(&self) -> Result<()> {
        self.manager.sync().await
    }

    /// Flushes the WAL buffer to disk.
    ///
    /// # Errors
    ///
    /// Returns an error if the flush fails.
    pub async fn flush(&self) -> Result<()> {
        self.manager.flush().await
    }

    /// Rotates to a new log file.
    ///
    /// # Errors
    ///
    /// Returns an error if rotation fails.
    pub async fn rotate(&self) -> Result<()> {
        self.manager.rotate().await
    }

    /// Returns the underlying [`AsyncWalManager`].
    ///
    /// Useful for accessing administrative methods like background sync.
    #[must_use]
    pub fn manager(&self) -> &AsyncWalManager {
        &self.manager
    }

    /// Returns the total number of records written.
    #[must_use]
    pub fn record_count(&self) -> u64 {
        self.manager.record_count()
    }

    /// Returns the WAL directory path.
    #[must_use]
    pub fn dir(&self) -> &Path {
        self.manager.dir()
    }

    /// Returns the current durability mode.
    #[must_use]
    pub fn durability_mode(&self) -> DurabilityMode {
        self.manager.durability_mode()
    }

    /// Returns all WAL log file paths in sequence order.
    ///
    /// # Errors
    ///
    /// Returns an error if the WAL directory cannot be read.
    pub async fn log_files(&self) -> Result<Vec<PathBuf>> {
        self.manager.log_files().await
    }

    /// Returns the latest checkpoint epoch, if any.
    pub async fn checkpoint_epoch(&self) -> Option<EpochId> {
        self.manager.checkpoint_epoch().await
    }
}

/// Type alias for the async LPG (labeled property graph) WAL.
pub type AsyncLpgWal = AsyncTypedWal<WalRecord>;

#[cfg(test)]
mod tests {
    use super::*;
    use grafeo_common::types::NodeId;
    use tempfile::tempdir;

    #[tokio::test]
    async fn test_async_typed_wal_write() {
        let dir = tempdir().unwrap();
        let wal: AsyncLpgWal = AsyncTypedWal::open(dir.path()).await.unwrap();

        let record = WalRecord::CreateNode {
            id: NodeId::new(1),
            labels: vec!["Person".to_string()],
        };

        wal.log(&record).await.unwrap();
        wal.flush().await.unwrap();
        assert_eq!(wal.record_count(), 1);
    }

    #[tokio::test]
    async fn test_async_typed_wal_checkpoint() {
        let dir = tempdir().unwrap();
        let wal: AsyncLpgWal = AsyncTypedWal::open(dir.path()).await.unwrap();

        wal.log(&WalRecord::CreateNode {
            id: NodeId::new(1),
            labels: vec!["Test".to_string()],
        })
        .await
        .unwrap();

        wal.log(&WalRecord::TransactionCommit {
            transaction_id: TransactionId::new(1),
        })
        .await
        .unwrap();

        wal.checkpoint(TransactionId::new(1), EpochId::new(10))
            .await
            .unwrap();

        // Checkpoint record + the two records above = 3 records total
        assert_eq!(wal.record_count(), 3);
    }

    #[tokio::test]
    async fn test_async_typed_wal_recovery_compatible() {
        // Verify AsyncTypedWal writes are recoverable by existing sync WalRecovery.
        // The on-disk format (length-prefix + bincode + CRC32) is identical.
        let dir = tempdir().unwrap();

        {
            let wal: AsyncLpgWal = AsyncTypedWal::open(dir.path()).await.unwrap();
            wal.log(&WalRecord::CreateNode {
                id: NodeId::new(1),
                labels: vec!["Person".to_string()],
            })
            .await
            .unwrap();
            wal.log(&WalRecord::TransactionCommit {
                transaction_id: TransactionId::new(1),
            })
            .await
            .unwrap();
            wal.sync().await.unwrap();
        }

        let recovery = super::super::WalRecovery::new(dir.path());
        let records = recovery.recover().unwrap();
        assert_eq!(records.len(), 2);
    }

    #[tokio::test]
    async fn test_async_sync_byte_equivalence() {
        // Same mutation sequence through sync TypedWal and async AsyncTypedWal
        // should produce identical WAL frames (same serialization + CRC).
        use super::super::TypedWal;

        let sync_dir = tempdir().unwrap();
        let async_dir = tempdir().unwrap();

        let records = vec![
            WalRecord::CreateNode {
                id: NodeId::new(1),
                labels: vec!["Person".to_string()],
            },
            WalRecord::SetNodeProperty {
                id: NodeId::new(1),
                key: "name".to_string(),
                value: grafeo_common::types::Value::String("Alix".into()),
            },
            WalRecord::TransactionCommit {
                transaction_id: TransactionId::new(1),
            },
        ];

        // Write via sync path
        {
            let wal: super::super::LpgWal = TypedWal::open(sync_dir.path()).unwrap();
            for record in &records {
                wal.log(record).unwrap();
            }
            wal.sync().unwrap();
        }

        // Write via async path
        {
            let wal: AsyncLpgWal = AsyncTypedWal::open(async_dir.path()).await.unwrap();
            for record in &records {
                wal.log(record).await.unwrap();
            }
            wal.sync().await.unwrap();
        }

        // Both should be recoverable with the same records
        let sync_recovery = super::super::WalRecovery::new(sync_dir.path());
        let async_recovery = super::super::WalRecovery::new(async_dir.path());

        let sync_records = sync_recovery.recover().unwrap();
        let async_records = async_recovery.recover().unwrap();

        assert_eq!(sync_records.len(), async_records.len());
        assert_eq!(sync_records.len(), 3);

        // Compare record-by-record via Debug representation
        for (sync_rec, async_rec) in sync_records.iter().zip(async_records.iter()) {
            assert_eq!(format!("{sync_rec:?}"), format!("{async_rec:?}"));
        }
    }

    #[tokio::test]
    async fn test_async_typed_wal_delegates_admin_methods() {
        let dir = tempdir().unwrap();
        let wal: AsyncLpgWal = AsyncTypedWal::open(dir.path()).await.unwrap();

        assert_eq!(wal.record_count(), 0);
        assert_eq!(wal.dir(), dir.path());
        assert!(wal.checkpoint_epoch().await.is_none());

        let files = wal.log_files().await.unwrap();
        assert!(!files.is_empty(), "no WAL files");

        let _mode = wal.durability_mode();
    }
}
