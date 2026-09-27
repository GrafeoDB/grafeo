//! Type-safe WAL wrapper.
//!
//! [`TypedWal`] wraps a [`WalManager`] and ensures that only records of type `R`
//! can be written. This prevents accidentally mixing record types (e.g., LPG
//! and RDF) in the same WAL instance.
//!
//! Use [`LpgWal`] for the standard labeled property graph WAL.

use std::marker::PhantomData;
use std::path::{Path, PathBuf};

use grafeo_common::types::EpochId;
use grafeo_common::types::TransactionId;
use grafeo_common::utils::error::{Error, Result};

use super::WalRecord;
use super::log::{CheckpointMetadata, DurabilityMode, WalConfig, WalManager};
use super::record::WalEntry;

/// A type-safe wrapper around [`WalManager`] that constrains record types
/// at compile time.
///
/// `TypedWal<R>` ensures that only records implementing [`WalEntry`] with
/// the specific type `R` can be logged. This prevents accidentally writing
/// the wrong record type to a WAL instance.
///
/// # Example
///
/// ```no_run
/// use grafeo_storage::wal::{LpgWal, WalRecord};
/// use grafeo_common::types::NodeId;
///
/// # fn main() -> grafeo_common::utils::error::Result<()> {
/// let wal = LpgWal::open("wal_dir")?;
/// wal.log(&WalRecord::CreateNode {
///     id: NodeId::new(1),
///     labels: vec!["Person".to_string()],
/// })?;
/// # Ok(())
/// # }
/// ```
pub struct TypedWal<R: WalEntry> {
    manager: WalManager,
    _record: PhantomData<R>,
}

impl<R: WalEntry> TypedWal<R> {
    /// Opens or creates a typed WAL in the given directory.
    ///
    /// # Errors
    ///
    /// Returns an error if the directory cannot be created or accessed.
    pub fn open(dir: impl AsRef<Path>) -> Result<Self> {
        Ok(Self {
            manager: WalManager::open(dir)?,
            _record: PhantomData,
        })
    }

    /// Opens or creates a typed WAL with custom configuration.
    ///
    /// # Errors
    ///
    /// Returns an error if the directory cannot be created or accessed.
    pub fn with_config(dir: impl AsRef<Path>, config: WalConfig) -> Result<Self> {
        Ok(Self {
            manager: WalManager::with_config(dir, config)?,
            _record: PhantomData,
        })
    }

    /// Logs a typed record to the WAL.
    ///
    /// The record is serialized via bincode and written with a length prefix
    /// and CRC32 checksum. Durability handling (fsync) is determined by the
    /// record's [`WalEntry::requires_sync`] method.
    ///
    /// # Errors
    ///
    /// Returns an error if serialization or writing fails.
    pub fn log(&self, record: &R) -> Result<()> {
        let data = bincode::serde::encode_to_vec(record, bincode::config::standard())
            .map_err(|e| Error::Serialization(e.to_string()))?;
        let force_sync = record.requires_sync();
        self.manager.write_frame(&data, force_sync)
    }

    /// Logs typed records as one contiguous group.
    ///
    /// No other writer's records can land between them, and the group is
    /// synced once if any record requires it (see [`WalEntry::requires_sync`]).
    ///
    /// # Errors
    ///
    /// Returns an error if serialization or writing fails.
    pub fn log_batch(&self, records: &[R]) -> Result<()> {
        let frames = records
            .iter()
            .map(|record| {
                bincode::serde::encode_to_vec(record, bincode::config::standard())
                    .map_err(|e| Error::Serialization(e.to_string()))
            })
            .collect::<Result<Vec<_>>>()?;
        let frame_refs: Vec<&[u8]> = frames.iter().map(Vec::as_slice).collect();
        let force_sync = records.iter().any(WalEntry::requires_sync);
        self.manager.write_frames(&frame_refs, force_sync)
    }

    /// Writes a checkpoint marker and persists checkpoint metadata.
    ///
    /// Creates a checkpoint record via [`WalEntry::make_checkpoint`], logs it,
    /// then syncs and writes the checkpoint metadata file.
    ///
    /// # Errors
    ///
    /// Returns an error if the checkpoint cannot be written.
    pub fn checkpoint(&self, current_transaction: TransactionId, epoch: EpochId) -> Result<()> {
        let checkpoint_record = R::make_checkpoint(current_transaction);
        self.log(&checkpoint_record)?;
        self.manager.complete_checkpoint(current_transaction, epoch)
    }

    /// Syncs the WAL to disk (fsync).
    ///
    /// # Errors
    ///
    /// Returns an error if the sync fails.
    pub fn sync(&self) -> Result<()> {
        self.manager.sync()
    }

    /// Flushes the WAL buffer to disk.
    ///
    /// # Errors
    ///
    /// Returns an error if the flush fails.
    pub fn flush(&self) -> Result<()> {
        self.manager.flush()
    }

    /// Rotates to a new log file.
    ///
    /// # Errors
    ///
    /// Returns an error if rotation fails.
    pub fn rotate(&self) -> Result<()> {
        self.manager.rotate()
    }

    /// Closes the active log file, releasing its file handle.
    ///
    /// This allows the WAL directory to be safely removed on Windows,
    /// where open file handles prevent directory deletion.
    pub fn close_active_log(&self) {
        self.manager.close_active_log();
    }

    /// Returns the underlying [`WalManager`].
    ///
    /// Useful for accessing administrative methods or for passing to
    /// [`AdaptiveFlusher`](super::AdaptiveFlusher).
    #[must_use]
    pub fn manager(&self) -> &WalManager {
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

    /// Returns the total size of all WAL files in bytes.
    #[must_use]
    pub fn size_bytes(&self) -> usize {
        self.manager.size_bytes()
    }

    /// Returns the timestamp of the last checkpoint (Unix epoch seconds), if any.
    #[must_use]
    pub fn last_checkpoint_timestamp(&self) -> Option<u64> {
        self.manager.last_checkpoint_timestamp()
    }

    /// Returns the latest checkpoint epoch, if any.
    #[must_use]
    pub fn checkpoint_epoch(&self) -> Option<EpochId> {
        self.manager.checkpoint_epoch()
    }

    /// Returns all WAL log file paths in sequence order.
    ///
    /// # Errors
    ///
    /// Returns an error if the WAL directory cannot be read.
    pub fn log_files(&self) -> Result<Vec<PathBuf>> {
        self.manager.log_files()
    }

    /// Reads checkpoint metadata from disk.
    ///
    /// # Errors
    ///
    /// Returns an error if the metadata file cannot be read or deserialized.
    pub fn read_checkpoint_metadata(&self) -> Result<Option<CheckpointMetadata>> {
        self.manager.read_checkpoint_metadata()
    }

    /// Returns the path to the active WAL file.
    #[must_use]
    pub fn path(&self) -> PathBuf {
        self.manager.path()
    }

    /// Returns the current WAL log sequence number.
    #[must_use]
    pub fn current_sequence(&self) -> u64 {
        self.manager.current_sequence()
    }
}

/// Type alias for the LPG (labeled property graph) WAL.
pub type LpgWal = TypedWal<WalRecord>;

impl TypedWal<WalRecord> {
    /// Seals a torn tail left by a crash, so a later commit marker cannot
    /// commit the torn records on replay.
    ///
    /// Starts a new log file (new records appended after a partially written
    /// frame would be unreadable) and writes an abort marker there, which makes
    /// recovery discard the records that no marker closed.
    ///
    /// Call this at open, before logging anything, when recovery reported a
    /// torn tail.
    ///
    /// # Errors
    ///
    /// Returns an error if the new log file or the marker cannot be written.
    pub fn seal_torn_tail(&self) -> Result<()> {
        self.manager.rotate()?;
        self.manager.log(&WalRecord::TransactionAbort {
            transaction_id: TransactionId::INVALID,
        })?;
        self.manager.sync()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use grafeo_common::types::NodeId;
    use tempfile::tempdir;

    #[test]
    fn test_typed_wal_write() {
        let dir = tempdir().unwrap();
        let wal: LpgWal = TypedWal::open(dir.path()).unwrap();

        let record = WalRecord::CreateNode {
            id: NodeId::new(1),
            labels: vec!["Person".to_string()],
        };

        wal.log(&record).unwrap();
        wal.flush().unwrap();
        assert_eq!(wal.record_count(), 1);
    }

    #[test]
    fn test_typed_wal_checkpoint() {
        let dir = tempdir().unwrap();
        let wal: LpgWal = TypedWal::open(dir.path()).unwrap();

        wal.log(&WalRecord::CreateNode {
            id: NodeId::new(1),
            labels: vec!["Test".to_string()],
        })
        .unwrap();

        wal.log(&WalRecord::TransactionCommit {
            transaction_id: TransactionId::new(1),
        })
        .unwrap();

        wal.checkpoint(TransactionId::new(1), EpochId::new(10))
            .unwrap();
        assert_eq!(wal.checkpoint_epoch(), Some(EpochId::new(10)));
    }

    #[test]
    fn test_typed_wal_recovery_compatible() {
        // Verify TypedWal writes are recoverable by existing WalRecovery
        let dir = tempdir().unwrap();

        {
            let wal: LpgWal = TypedWal::open(dir.path()).unwrap();
            wal.log(&WalRecord::CreateNode {
                id: NodeId::new(1),
                labels: vec!["Person".to_string()],
            })
            .unwrap();
            wal.log(&WalRecord::TransactionCommit {
                transaction_id: TransactionId::new(1),
            })
            .unwrap();
            wal.sync().unwrap();
        }

        let recovery = super::super::WalRecovery::new(dir.path());
        let records = recovery.recover().unwrap();
        assert_eq!(records.len(), 2);
    }

    #[test]
    fn test_typed_wal_delegates_admin_methods() {
        let dir = tempdir().unwrap();
        let wal: LpgWal = TypedWal::open(dir.path()).unwrap();

        // Verify delegation works
        assert_eq!(wal.record_count(), 0);
        assert_eq!(wal.dir(), dir.path());
        assert!(wal.size_bytes() > 0 || wal.size_bytes() == 0);
        assert!(wal.checkpoint_epoch().is_none());
        assert!(wal.last_checkpoint_timestamp().is_none());

        let files = wal.log_files().unwrap();
        assert!(!files.is_empty());

        let _path = wal.path();
        let _mode = wal.durability_mode();
    }

    fn log_files_with_data(dir: &Path) -> usize {
        std::fs::read_dir(dir)
            .unwrap()
            .map(|e| e.unwrap().path())
            .filter(|p| p.extension().is_some_and(|ext| ext == "log"))
            .filter(|p| std::fs::metadata(p).unwrap().len() > 0)
            .count()
    }

    #[test]
    fn test_log_batch_groups_do_not_interleave() {
        use super::super::WalRecovery;
        use std::sync::Arc;

        /// Records per group, and the same count as an id offset.
        const GROUP: usize = 5;
        const GROUP_IDS: u64 = 5;
        let dir = tempdir().unwrap();
        {
            let wal: Arc<LpgWal> = Arc::new(TypedWal::open(dir.path()).unwrap());
            let writers: Vec<_> = (0..4u64)
                .map(|writer| {
                    let wal = Arc::clone(&wal);
                    std::thread::spawn(move || {
                        for group in 0..50u64 {
                            let base = (writer * 1000 + group) * 10;
                            let mut records: Vec<WalRecord> = (0..GROUP_IDS)
                                .map(|k| WalRecord::CreateNode {
                                    id: NodeId::new(base + k),
                                    labels: vec!["N".to_string()],
                                })
                                .collect();
                            records.push(WalRecord::TransactionCommit {
                                transaction_id: TransactionId::new(base),
                            });
                            wal.log_batch(&records).unwrap();
                        }
                    })
                })
                .collect();
            for writer in writers {
                writer.join().unwrap();
            }
            wal.sync().unwrap();
        }

        // Every commit marker must directly follow its own group's records.
        let records = WalRecovery::new(dir.path()).recover().unwrap();
        assert_eq!(records.len(), 4 * 50 * (GROUP + 1));
        for chunk in records.chunks(GROUP + 1) {
            let WalRecord::TransactionCommit { transaction_id } = chunk[GROUP] else {
                panic!("group not closed by its commit marker: {chunk:?}");
            };
            let ids: Vec<u64> = chunk[..GROUP]
                .iter()
                .map(|r| match r {
                    WalRecord::CreateNode { id, .. } => id.as_u64(),
                    other => panic!("unexpected record {other:?}"),
                })
                .collect();
            let base = transaction_id.as_u64();
            assert_eq!(ids, (base..base + GROUP_IDS).collect::<Vec<_>>());
        }
    }

    #[test]
    fn test_log_batch_rotates_only_between_groups() {
        let dir = tempdir().unwrap();
        let wal: LpgWal = TypedWal::with_config(
            dir.path(),
            WalConfig {
                max_log_size: 100,
                ..WalConfig::default()
            },
        )
        .unwrap();
        let group: Vec<WalRecord> = (0..10)
            .map(|i| WalRecord::CreateNode {
                id: NodeId::new(i),
                labels: vec!["Rotation".to_string()],
            })
            .chain(std::iter::once(WalRecord::TransactionCommit {
                transaction_id: TransactionId::new(1),
            }))
            .collect();

        // Each group is far over the size limit, yet stays in one file.
        wal.log_batch(&group).unwrap();
        assert_eq!(log_files_with_data(dir.path()), 1);
        wal.log_batch(&group).unwrap();
        assert_eq!(log_files_with_data(dir.path()), 2);
    }
}
