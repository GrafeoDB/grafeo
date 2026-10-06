//! WAL recovery.

use super::record::WalEntry;
use super::{CheckpointMetadata, WalCipher, WalManager, WalRecord};
use grafeo_common::utils::error::{Error, Result, StorageError};
use grafeo_common::{grafeo_debug, grafeo_info, grafeo_warn};
use std::fs::File;
use std::io::{BufReader, Read};
use std::path::Path;
#[cfg(feature = "encryption")]
use std::sync::Arc;

/// Name of the checkpoint metadata file.
const CHECKPOINT_METADATA_FILE: &str = "checkpoint.meta";

/// Committed records from a recovery pass, and whether the log ends torn.
#[derive(Debug)]
pub struct RecoveredWal<R = WalRecord> {
    /// Records of committed transactions, plus checkpoint and metadata
    /// records, in log order.
    pub records: Vec<R>,
    /// Whether the log ends with records that no commit or abort marker
    /// closes, or with bytes that do not form a complete record. Such a tail
    /// must be sealed (see `LpgWal::seal_torn_tail`) before anything new is
    /// logged, otherwise the next commit marker would commit the torn records.
    pub torn_tail: bool,
}

/// Handles WAL recovery after a crash.
pub struct WalRecovery {
    /// Directory containing WAL files.
    dir: std::path::PathBuf,
    /// Encryptor for decrypting WAL records (None = unencrypted).
    #[cfg(feature = "encryption")]
    encryptor: Option<Arc<grafeo_common::encryption::PageEncryptor>>,
}

impl WalRecovery {
    /// Creates a new recovery handler for the given WAL directory.
    pub fn new(dir: impl AsRef<Path>) -> Self {
        Self {
            dir: dir.as_ref().to_path_buf(),
            #[cfg(feature = "encryption")]
            encryptor: None,
        }
    }

    /// Creates a recovery handler for the given WAL directory that decrypts
    /// its records with `cipher`; `None` reads plaintext records.
    #[must_use]
    pub fn with_cipher(dir: impl AsRef<Path>, cipher: Option<WalCipher>) -> Self {
        #[cfg(not(feature = "encryption"))]
        if let Some(cipher) = cipher {
            match cipher {}
        }
        Self {
            dir: dir.as_ref().to_path_buf(),
            #[cfg(feature = "encryption")]
            encryptor: cipher.map(Arc::new),
        }
    }

    /// Sets the encryptor for decrypting WAL records during recovery.
    #[cfg(feature = "encryption")]
    pub fn set_encryptor(&mut self, encryptor: grafeo_common::encryption::PageEncryptor) {
        self.encryptor = Some(Arc::new(encryptor));
    }

    /// Creates a recovery handler from a WAL manager, which decrypts with
    /// the manager's encryptor, if it has one.
    #[must_use]
    pub fn from_wal(wal: &WalManager) -> Self {
        Self {
            dir: wal.dir().to_path_buf(),
            #[cfg(feature = "encryption")]
            encryptor: wal.encryptor().cloned(),
        }
    }

    /// Reads checkpoint metadata if it exists.
    ///
    /// Returns `None` if no checkpoint metadata is found.
    ///
    /// # Errors
    ///
    /// Returns an error if the metadata file cannot be read or deserialized.
    pub fn read_checkpoint_metadata(&self) -> Result<Option<CheckpointMetadata>> {
        let metadata_path = self.dir.join(CHECKPOINT_METADATA_FILE);

        if !metadata_path.exists() {
            return Ok(None);
        }

        let file = File::open(&metadata_path)?;
        let mut reader = BufReader::new(file);
        let mut data = Vec::new();
        reader.read_to_end(&mut data)?;

        let (metadata, _): (CheckpointMetadata, _) =
            bincode::serde::decode_from_slice(&data, bincode::config::standard())
                .map_err(|e| Error::Serialization(e.to_string()))?;

        Ok(Some(metadata))
    }

    /// Returns the checkpoint metadata, if any.
    ///
    /// This is useful for determining whether to perform a full or
    /// incremental recovery.
    #[must_use]
    pub fn checkpoint(&self) -> Option<CheckpointMetadata> {
        self.read_checkpoint_metadata().ok().flatten()
    }

    /// Recovers committed records from all WAL files.
    ///
    /// Returns only records that were part of committed transactions.
    /// If checkpoint metadata exists, only replays files from the
    /// checkpoint sequence onwards.
    ///
    /// # Errors
    ///
    /// Returns an error if recovery fails.
    pub fn recover(&self) -> Result<Vec<WalRecord>> {
        self.recover_as::<WalRecord>()
    }

    /// Recovers committed records of a specific type from all WAL files.
    ///
    /// This is the generic version of [`recover`](Self::recover). Use it
    /// when recovering a WAL that stores a custom record type.
    ///
    /// # Errors
    ///
    /// Returns an error if recovery fails.
    pub fn recover_as<R: WalEntry>(&self) -> Result<Vec<R>> {
        let checkpoint = self.read_checkpoint_metadata()?;
        Ok(self.recover_internal_as::<R>(checkpoint)?.records)
    }

    /// Recovers committed records from every WAL file, ignoring checkpoint
    /// metadata.
    ///
    /// For WAL-directory databases, where the WAL is the only copy of the data
    /// and no snapshot covers the files a checkpoint would skip. An existing
    /// `checkpoint.meta` (written by older versions) is reported with a warning.
    ///
    /// # Errors
    ///
    /// Returns an error if recovery fails.
    pub fn recover_all(&self) -> Result<Vec<WalRecord>> {
        Ok(self.recover_all_with_tail()?.records)
    }

    /// Like [`recover_all`](Self::recover_all), and also reports whether the
    /// log ends with a torn tail.
    ///
    /// # Errors
    ///
    /// Returns an error if recovery fails.
    pub fn recover_all_with_tail(&self) -> Result<RecoveredWal> {
        if self.dir.join(CHECKPOINT_METADATA_FILE).exists() {
            grafeo_warn!(
                "Ignoring {:?}: the WAL is the only copy of the data, replaying every WAL file",
                self.dir.join(CHECKPOINT_METADATA_FILE)
            );
        }
        self.recover_internal_as::<WalRecord>(None)
    }

    /// Like [`recover`](Self::recover), and also reports whether the log
    /// ends with a torn tail.
    ///
    /// # Errors
    ///
    /// Returns an error if recovery fails.
    pub fn recover_with_tail(&self) -> Result<RecoveredWal> {
        let checkpoint = self.read_checkpoint_metadata()?;
        self.recover_internal_as::<WalRecord>(checkpoint)
    }

    /// Recovers committed records up to and including the given epoch.
    ///
    /// Returns only records belonging to transactions committed at or before
    /// `max_epoch`. Records from the first transaction committed after
    /// `max_epoch` are excluded.
    ///
    /// The WAL commit sequence is: `[data records] [TxCommit] [EpochAdvance]`.
    /// When an `EpochAdvance { epoch }` where `epoch > max_epoch` is seen, we
    /// discard the preceding transaction's records (everything since the last
    /// `EpochAdvance` that was within range).
    ///
    /// Used by point-in-time recovery to restore a database to a specific epoch.
    ///
    /// # Errors
    ///
    /// Returns an error if recovery fails.
    pub fn recover_until_epoch(
        &self,
        max_epoch: grafeo_common::types::EpochId,
    ) -> Result<Vec<WalRecord>> {
        let all_records = self.recover()?;
        let mut committed = Vec::new();
        let mut pending = Vec::new();

        for record in all_records {
            if let WalRecord::EpochAdvance { epoch } = &record {
                if *epoch > max_epoch {
                    // This epoch's transaction is beyond our target:
                    // discard the pending records and stop.
                    break;
                }
                // Epoch is within range: flush pending into committed
                committed.append(&mut pending);
                committed.push(record);
            } else {
                pending.push(record);
            }
        }

        // Any remaining pending records lack a confirming EpochAdvance
        // within range, so they belong to a transaction beyond max_epoch
        // (or an incomplete transaction) and are intentionally dropped.

        Ok(committed)
    }

    fn recover_internal_as<R: WalEntry>(
        &self,
        checkpoint: Option<CheckpointMetadata>,
    ) -> Result<RecoveredWal<R>> {
        let mut current_tx_records = Vec::new();
        let mut committed_records = Vec::new();
        // Whether the last file read ended exactly at a record boundary.
        let mut last_file_clean = true;

        // Get all log files in order
        let log_files = self.get_log_files()?;
        // Lets a test move the directory between the listing and the reads.
        grafeo_common::testing::pause::maybe_pause("wal_recovery:after_listing");

        // Determine the minimum sequence number to process
        let min_sequence = checkpoint.as_ref().map_or(0, |cp| cp.log_sequence);

        if checkpoint.is_some() {
            grafeo_info!(
                "Recovering from checkpoint at epoch {:?}, starting from log sequence {}",
                checkpoint.as_ref().map(|c| c.epoch),
                min_sequence
            );
        }

        // Read log files in sequence, skipping those before checkpoint
        for log_file in log_files {
            // Extract sequence number from filename
            let sequence = Self::sequence_from_path(&log_file).unwrap_or(0);

            // Skip files that are completely before the checkpoint
            // We include the checkpoint sequence file because it may contain
            // records after the checkpoint record itself
            if sequence < min_sequence {
                grafeo_debug!(
                    "Skipping log file {:?} (sequence {} < checkpoint {})",
                    log_file,
                    sequence,
                    min_sequence
                );
                continue;
            }

            let file = match File::open(&log_file) {
                Ok(f) => f,
                Err(e) if e.kind() == std::io::ErrorKind::NotFound => continue,
                Err(e) => return Err(e.into()),
            };
            let file_len = file.metadata()?.len();
            let mut reader = BufReader::new(file);
            let mut position = 0u64;

            // Read all records from this file
            loop {
                match self.read_record_as::<R>(&mut reader) {
                    Ok(Some((record, frame_len))) => {
                        position += frame_len;
                        if record.is_commit() {
                            committed_records.append(&mut current_tx_records);
                            committed_records.push(record);
                        } else if record.is_abort() {
                            current_tx_records.clear();
                        } else if record.is_checkpoint() || record.is_metadata() {
                            // Checkpoint and metadata records (e.g. EpochAdvance)
                            // are not part of any transaction: always include them.
                            committed_records.push(record);
                        } else {
                            current_tx_records.push(record);
                        }
                    }
                    Ok(None) => {
                        // EOF, or a partial length prefix at the end.
                        last_file_clean = position >= file_len;
                        break;
                    }
                    Err(e) => {
                        // Log corruption - stop reading this file but continue
                        // with remaining files (best-effort recovery)
                        grafeo_warn!("WAL corruption detected in {:?}: {}", log_file, e);
                        last_file_clean = false;
                        break;
                    }
                }
            }
        }

        // Uncommitted records in current_tx_records are discarded
        let torn_tail = !current_tx_records.is_empty() || !last_file_clean;
        if torn_tail {
            grafeo_warn!(
                "WAL in {:?} ends with an incomplete transaction ({} records without a commit marker{})",
                self.dir,
                current_tx_records.len(),
                if last_file_clean {
                    ""
                } else {
                    ", followed by an incomplete record"
                }
            );
        }

        Ok(RecoveredWal {
            records: committed_records,
            torn_tail,
        })
    }

    /// Extracts the sequence number from a WAL log file path.
    fn sequence_from_path(path: &Path) -> Option<u64> {
        path.file_stem()
            .and_then(|s| s.to_str())
            .and_then(|s| s.strip_prefix("wal_"))
            .and_then(|s| s.parse().ok())
    }

    /// The WAL files of the directory, sorted by sequence. A missing
    /// directory holds none; one that cannot be listed is an error, never
    /// "no files": recovering nothing would lose the records it holds.
    fn get_log_files(&self) -> Result<Vec<std::path::PathBuf>> {
        let cannot_list = |error: std::io::Error| {
            Error::Io(std::io::Error::new(
                error.kind(),
                format!(
                    "cannot list the WAL directory {}: {error}",
                    self.dir.display()
                ),
            ))
        };
        let entries = match std::fs::read_dir(&self.dir) {
            Ok(entries) => entries,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(Vec::new()),
            Err(error) => return Err(cannot_list(error)),
        };

        let mut files = Vec::new();
        for entry in entries {
            let path = entry.map_err(cannot_list)?.path();
            if path.extension().is_some_and(|ext| ext == "log") {
                files.push(path);
            }
        }

        // Sort by filename (which includes sequence number)
        files.sort();

        Ok(files)
    }

    /// Reads one record, returning it with its frame length in bytes.
    fn read_record_as<R: WalEntry>(
        &self,
        reader: &mut BufReader<File>,
    ) -> Result<Option<(R, u64)>> {
        // Read length prefix
        let mut len_buf = [0u8; 4];
        match reader.read_exact(&mut len_buf) {
            Ok(()) => {}
            Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => return Ok(None),
            Err(e) => return Err(e.into()),
        }
        let len = u32::from_le_bytes(len_buf) as usize;

        // Read payload (either encrypted or plaintext+checksum)
        #[cfg(feature = "encryption")]
        let data = if let Some(ref enc) = self.encryptor {
            // Encrypted frame: payload is nonce(12) || ciphertext || tag(16)
            let mut encrypted = vec![0u8; len];
            reader.read_exact(&mut encrypted)?;
            let aad = b"grafeo-wal";
            enc.decrypt(&encrypted, aad).map_err(|_| {
                Error::Storage(StorageError::Corruption(
                    "WAL decryption failed: wrong key or corrupted record".to_string(),
                ))
            })?
        } else {
            // Plaintext frame: data + crc32
            let mut data = vec![0u8; len];
            reader.read_exact(&mut data)?;

            let mut checksum_buf = [0u8; 4];
            reader.read_exact(&mut checksum_buf)?;
            let stored_checksum = u32::from_le_bytes(checksum_buf);
            let computed_checksum = crc32fast::hash(&data);
            if stored_checksum != computed_checksum {
                return Err(Error::Storage(StorageError::Corruption(
                    "WAL checksum mismatch".to_string(),
                )));
            }
            data
        };

        #[cfg(not(feature = "encryption"))]
        let data = {
            let mut data = vec![0u8; len];
            reader.read_exact(&mut data)?;

            let mut checksum_buf = [0u8; 4];
            reader.read_exact(&mut checksum_buf)?;
            let stored_checksum = u32::from_le_bytes(checksum_buf);
            let computed_checksum = crc32fast::hash(&data);
            if stored_checksum != computed_checksum {
                return Err(Error::Storage(StorageError::Corruption(
                    "WAL checksum mismatch".to_string(),
                )));
            }
            data
        };

        // Deserialize
        let (record, _): (R, _) =
            bincode::serde::decode_from_slice(&data, bincode::config::standard())
                .map_err(|e| Error::Serialization(e.to_string()))?;

        // Length prefix + payload, plus the CRC32 for plaintext frames
        // (an encrypted payload carries its own authentication tag).
        #[cfg(feature = "encryption")]
        let checksum_len = if self.encryptor.is_some() { 0 } else { 4 };
        #[cfg(not(feature = "encryption"))]
        let checksum_len = 4;
        Ok(Some((record, 4 + len as u64 + checksum_len)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use grafeo_common::types::{EpochId, NodeId, TransactionId};
    use tempfile::tempdir;

    #[test]
    fn test_recovery_committed() {
        let dir = tempdir().unwrap();

        // Write some records
        {
            let wal = WalManager::open(dir.path()).unwrap();

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

        // Recover
        let recovery = WalRecovery::new(dir.path());
        let records = recovery.recover().unwrap();

        assert_eq!(records.len(), 2);
    }

    #[test]
    fn test_recovery_uncommitted() {
        let dir = tempdir().unwrap();

        // Write some records without commit
        {
            let wal = WalManager::open(dir.path()).unwrap();

            wal.log(&WalRecord::CreateNode {
                id: NodeId::new(1),
                labels: vec!["Person".to_string()],
            })
            .unwrap();

            // No commit!
            wal.sync().unwrap();
        }

        // Recover
        let recovery = WalRecovery::new(dir.path());
        let records = recovery.recover().unwrap();

        // Uncommitted records should be discarded
        assert_eq!(records.len(), 0);
    }

    /// Denies the listing of a directory to everyone until dropped, which
    /// allows it again, also when the test fails in between, so the temporary
    /// directory can be removed.
    struct Unlisted<'a>(&'a Path);

    impl<'a> Unlisted<'a> {
        fn deny(dir: &'a Path) -> Self {
            assert!(
                set_listing(dir, false),
                "cannot deny listing {}",
                dir.display()
            );
            Self(dir)
        }
    }

    impl Drop for Unlisted<'_> {
        /// Reports a failure instead of panicking: a panic in a drop while a
        /// failed test unwinds would abort the test binary.
        fn drop(&mut self) {
            if !set_listing(self.0, true) {
                eprintln!("cannot allow listing {} again", self.0.display());
            }
        }
    }

    /// Denies (`allowed` false) or allows again the listing of `dir` to
    /// everyone: `icacls` on Windows, the mode on Unix. Returns whether it
    /// worked.
    fn set_listing(dir: &Path, allowed: bool) -> bool {
        #[cfg(windows)]
        {
            let mut command = std::process::Command::new("icacls");
            command.arg(dir);
            if allowed {
                command.args(["/remove:d", "*S-1-1-0"]);
            } else {
                command.args(["/deny", "*S-1-1-0:(RD)"]);
            }
            command.status().is_ok_and(|status| status.success())
        }
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let mode = if allowed { 0o700 } else { 0o300 };
            std::fs::set_permissions(dir, std::fs::Permissions::from_mode(mode)).is_ok()
        }
    }

    /// A WAL directory that exists but cannot be listed fails the recovery,
    /// naming the directory, instead of recovering nothing: the records it
    /// holds would be lost to whatever the caller does next (a migration
    /// writes an empty image, an open replays nothing). A missing directory
    /// still recovers nothing.
    #[test]
    fn a_wal_directory_that_cannot_be_listed_fails_the_recovery() {
        let dir = tempdir().unwrap();
        let wal_dir = dir.path().join("wal");
        {
            let wal = WalManager::open(&wal_dir).unwrap();
            wal.log(&WalRecord::CreateNode {
                id: NodeId::new(3),
                labels: vec!["Person".to_string()],
            })
            .unwrap();
            wal.log(&WalRecord::TransactionCommit {
                transaction_id: TransactionId::new(19),
            })
            .unwrap();
            wal.sync().unwrap();
        }

        let (listed, recovered) = {
            let _denied = Unlisted::deny(&wal_dir);
            (
                std::fs::read_dir(&wal_dir).map(|_| ()),
                WalRecovery::new(&wal_dir).recover_all_with_tail(),
            )
        };
        if listed.is_ok() {
            eprintln!("skipped: the WAL directory can still be listed (permissions do not apply)");
            return;
        }

        let error = recovered
            .expect_err("a WAL directory that cannot be listed is an error")
            .to_string();
        assert!(
            error.contains(&wal_dir.display().to_string()),
            "the error names the WAL directory: {error}"
        );
        assert_eq!(
            WalRecovery::new(&wal_dir).recover().unwrap().len(),
            2,
            "listed again, the WAL recovers its records"
        );
        assert!(
            WalRecovery::new(dir.path().join("missing"))
                .recover()
                .unwrap()
                .is_empty(),
            "a missing WAL directory recovers nothing"
        );
    }

    #[test]
    fn test_recovery_multiple_files() {
        let dir = tempdir().unwrap();

        // Write records across multiple files
        {
            let config = super::super::WalConfig {
                max_log_size: 100, // Force rotation
                ..Default::default()
            };
            let wal = WalManager::with_config(dir.path(), config).unwrap();

            // First transaction
            for i in 0..5 {
                wal.log(&WalRecord::CreateNode {
                    id: NodeId::new(i),
                    labels: vec!["Test".to_string()],
                })
                .unwrap();
            }
            wal.log(&WalRecord::TransactionCommit {
                transaction_id: TransactionId::new(1),
            })
            .unwrap();

            // Second transaction
            for i in 5..10 {
                wal.log(&WalRecord::CreateNode {
                    id: NodeId::new(i),
                    labels: vec!["Test".to_string()],
                })
                .unwrap();
            }
            wal.log(&WalRecord::TransactionCommit {
                transaction_id: TransactionId::new(2),
            })
            .unwrap();

            wal.sync().unwrap();
        }

        // Recover
        let recovery = WalRecovery::new(dir.path());
        let records = recovery.recover().unwrap();

        // Should have 10 CreateNode + 2 TransactionCommit
        assert_eq!(records.len(), 12);
    }

    #[test]
    fn test_checkpoint_metadata() {
        use grafeo_common::types::EpochId;

        let dir = tempdir().unwrap();

        // Write records and create a checkpoint
        {
            let wal = WalManager::open(dir.path()).unwrap();

            // First transaction
            wal.log(&WalRecord::CreateNode {
                id: NodeId::new(1),
                labels: vec!["Test".to_string()],
            })
            .unwrap();
            wal.log(&WalRecord::TransactionCommit {
                transaction_id: TransactionId::new(1),
            })
            .unwrap();

            // Create checkpoint
            wal.checkpoint(TransactionId::new(1), EpochId::new(10))
                .unwrap();

            // Second transaction after checkpoint
            wal.log(&WalRecord::CreateNode {
                id: NodeId::new(2),
                labels: vec!["Test".to_string()],
            })
            .unwrap();
            wal.log(&WalRecord::TransactionCommit {
                transaction_id: TransactionId::new(2),
            })
            .unwrap();

            wal.sync().unwrap();
        }

        // Verify checkpoint metadata was written
        let recovery = WalRecovery::new(dir.path());
        let checkpoint = recovery.checkpoint();
        assert!(checkpoint.is_some(), "Checkpoint metadata should exist");

        let cp = checkpoint.unwrap();
        assert_eq!(cp.epoch.as_u64(), 10);
        assert_eq!(cp.transaction_id.as_u64(), 1);
    }

    #[test]
    fn test_recovery_from_checkpoint() {
        use super::super::WalConfig;
        use grafeo_common::types::EpochId;

        let dir = tempdir().unwrap();

        // Write records across multiple log files with checkpoint
        {
            let config = WalConfig {
                max_log_size: 100, // Force rotation
                ..Default::default()
            };
            let wal = WalManager::with_config(dir.path(), config).unwrap();

            // First batch of records (should end up in early log files)
            for i in 0..5 {
                wal.log(&WalRecord::CreateNode {
                    id: NodeId::new(i),
                    labels: vec!["Before".to_string()],
                })
                .unwrap();
            }
            wal.log(&WalRecord::TransactionCommit {
                transaction_id: TransactionId::new(1),
            })
            .unwrap();

            // Create checkpoint
            wal.checkpoint(TransactionId::new(1), EpochId::new(100))
                .unwrap();

            // Second batch after checkpoint
            for i in 100..103 {
                wal.log(&WalRecord::CreateNode {
                    id: NodeId::new(i),
                    labels: vec!["After".to_string()],
                })
                .unwrap();
            }
            wal.log(&WalRecord::TransactionCommit {
                transaction_id: TransactionId::new(2),
            })
            .unwrap();

            wal.sync().unwrap();
        }

        // Recovery should use checkpoint metadata to skip old files
        let recovery = WalRecovery::new(dir.path());
        let records = recovery.recover().unwrap();

        // We should get all committed records (checkpoint metadata is used for optimization)
        // The number depends on how many log files were skipped
        assert!(!records.is_empty(), "Should recover some records");
    }

    #[test]
    fn test_recover_all_ignores_checkpoint_metadata() {
        use super::super::WalConfig;
        use grafeo_common::types::EpochId;

        let dir = tempdir().unwrap();
        {
            let config = WalConfig {
                max_log_size: 100, // Force rotation
                ..Default::default()
            };
            let wal = WalManager::with_config(dir.path(), config).unwrap();
            for i in 0..5 {
                wal.log(&WalRecord::CreateNode {
                    id: NodeId::new(i),
                    labels: vec!["Before".to_string()],
                })
                .unwrap();
                wal.log(&WalRecord::TransactionCommit {
                    transaction_id: TransactionId::new(i + 1),
                })
                .unwrap();
            }
            // Epoch 0 keeps all files on disk; the metadata still points at
            // the latest sequence.
            wal.checkpoint(TransactionId::new(5), EpochId::new(0))
                .unwrap();
        }

        let recovery = WalRecovery::new(dir.path());
        let node_ids = |records: &[WalRecord]| -> Vec<u64> {
            records
                .iter()
                .filter_map(|r| match r {
                    WalRecord::CreateNode { id, .. } => Some(id.as_u64()),
                    _ => None,
                })
                .collect()
        };
        assert!(
            node_ids(&recovery.recover().unwrap()).len() < 5,
            "checkpoint-aware recovery skips the rotated files"
        );
        assert_eq!(
            node_ids(&recovery.recover_all().unwrap()),
            vec![0, 1, 2, 3, 4]
        );
    }

    fn create(id: u64) -> WalRecord {
        WalRecord::CreateNode {
            id: NodeId::new(id),
            labels: vec!["N".to_string()],
        }
    }

    fn commit(tx: u64) -> WalRecord {
        WalRecord::TransactionCommit {
            transaction_id: TransactionId::new(tx),
        }
    }

    fn created_ids(records: &[WalRecord]) -> Vec<u64> {
        records
            .iter()
            .filter_map(|r| match r {
                WalRecord::CreateNode { id, .. } => Some(id.as_u64()),
                _ => None,
            })
            .collect()
    }

    fn only_log_file(dir: &Path) -> std::path::PathBuf {
        let mut files: Vec<_> = std::fs::read_dir(dir)
            .unwrap()
            .map(|e| e.unwrap().path())
            .filter(|p| p.extension().is_some_and(|ext| ext == "log"))
            .collect();
        assert_eq!(files.len(), 1);
        files.pop().unwrap()
    }

    #[test]
    fn test_torn_tail_detection() {
        // Every committed: clean.
        let dir = tempdir().unwrap();
        {
            let wal = WalManager::open(dir.path()).unwrap();
            wal.log_batch(&[create(1), commit(1)]).unwrap();
        }
        let recovered = WalRecovery::new(dir.path()).recover_with_tail().unwrap();
        assert!(!recovered.torn_tail);
        assert_eq!(created_ids(&recovered.records), vec![1]);

        // Records without a closing marker at the end: torn.
        {
            let wal = WalManager::open(dir.path()).unwrap();
            wal.log(&create(2)).unwrap();
        }
        let recovered = WalRecovery::new(dir.path()).recover_with_tail().unwrap();
        assert!(recovered.torn_tail);
        assert_eq!(created_ids(&recovered.records), vec![1]);

        // An abort closes them: clean again.
        {
            let wal = WalManager::open(dir.path()).unwrap();
            wal.log(&WalRecord::TransactionAbort {
                transaction_id: TransactionId::new(2),
            })
            .unwrap();
        }
        let recovered = WalRecovery::new(dir.path())
            .recover_all_with_tail()
            .unwrap();
        assert!(!recovered.torn_tail);
        assert_eq!(created_ids(&recovered.records), vec![1]);
    }

    #[test]
    fn test_partial_frame_is_a_torn_tail() {
        // A crash mid-write leaves bytes that are not a complete record, even
        // when every complete record is committed.
        for garbage in [&[7u8, 0][..], &[40, 0, 0, 0, 1, 2, 3][..]] {
            let dir = tempdir().unwrap();
            {
                let wal = WalManager::open(dir.path()).unwrap();
                wal.log_batch(&[create(1), commit(1)]).unwrap();
            }
            use std::io::Write;
            std::fs::OpenOptions::new()
                .append(true)
                .open(only_log_file(dir.path()))
                .unwrap()
                .write_all(garbage)
                .unwrap();

            let recovered = WalRecovery::new(dir.path()).recover_with_tail().unwrap();
            assert!(recovered.torn_tail, "garbage {garbage:?}");
            assert_eq!(created_ids(&recovered.records), vec![1]);
        }
    }

    #[test]
    fn test_sealed_torn_tail_is_not_committed_later() {
        use super::super::LpgWal;

        let dir = tempdir().unwrap();
        {
            let wal = LpgWal::open(dir.path()).unwrap();
            wal.log_batch(&[create(1), commit(1)]).unwrap();
            wal.log(&create(2)).unwrap(); // crash before the commit marker
        }

        let recovered = WalRecovery::new(dir.path()).recover_with_tail().unwrap();
        assert!(recovered.torn_tail);
        {
            let wal = LpgWal::open(dir.path()).unwrap();
            wal.seal_torn_tail().unwrap();
            wal.log_batch(&[create(3), commit(3)]).unwrap();
        }

        let recovered = WalRecovery::new(dir.path()).recover_with_tail().unwrap();
        assert!(!recovered.torn_tail);
        assert_eq!(created_ids(&recovered.records), vec![1, 3]);
    }

    #[test]
    fn test_seal_after_partial_frame_keeps_new_records_readable() {
        use super::super::LpgWal;
        use std::io::Write;

        let dir = tempdir().unwrap();
        {
            let wal = LpgWal::open(dir.path()).unwrap();
            wal.log_batch(&[create(1), commit(1)]).unwrap();
        }
        std::fs::OpenOptions::new()
            .append(true)
            .open(only_log_file(dir.path()))
            .unwrap()
            .write_all(&[40, 0, 0, 0, 1, 2, 3])
            .unwrap();

        assert!(
            WalRecovery::new(dir.path())
                .recover_with_tail()
                .unwrap()
                .torn_tail
        );
        {
            // Records appended after the partial frame in the same file would
            // be unreadable: sealing moves on to a new file.
            let wal = LpgWal::open(dir.path()).unwrap();
            wal.seal_torn_tail().unwrap();
            wal.log_batch(&[create(2), commit(2)]).unwrap();
        }

        let recovered = WalRecovery::new(dir.path()).recover_with_tail().unwrap();
        assert_eq!(created_ids(&recovered.records), vec![1, 2]);
    }

    #[test]
    fn test_recover_as_generic() {
        let dir = tempdir().unwrap();

        // Write records using WalManager
        {
            let wal = WalManager::open(dir.path()).unwrap();

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

        // Recover using the generic method
        let recovery = WalRecovery::new(dir.path());
        let records: Vec<WalRecord> = recovery.recover_as().unwrap();

        assert_eq!(records.len(), 2);

        // Verify the records are correct via WalEntry trait methods
        assert!(!records[0].is_commit());
        assert!(records[1].is_commit());
    }

    #[test]
    fn test_recovery_truncated_wal_mid_record() {
        let dir = tempdir().unwrap();

        // Write valid records first
        {
            let wal = WalManager::open(dir.path()).unwrap();
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

        // Find the WAL file and append a truncated record (length prefix only, no data)
        let wal_files: Vec<_> = std::fs::read_dir(dir.path())
            .unwrap()
            .filter_map(|e| {
                let e = e.ok()?;
                if e.path().extension().is_some_and(|ext| ext == "log") {
                    Some(e.path())
                } else {
                    None
                }
            })
            .collect();
        assert!(!wal_files.is_empty(), "wal_files is empty");

        // Append a partial record: just a length prefix, then truncate
        use std::io::Write;
        let mut f = std::fs::OpenOptions::new()
            .append(true)
            .open(&wal_files[0])
            .unwrap();
        f.write_all(&100u32.to_le_bytes()).unwrap(); // length=100 but no data follows

        // Recovery should still return the committed records (best-effort)
        let recovery = WalRecovery::new(dir.path());
        let records = recovery.recover().unwrap();
        assert_eq!(
            records.len(),
            2,
            "committed records before truncation should be recovered"
        );
    }

    #[test]
    fn test_recovery_corrupted_checksum() {
        let dir = tempdir().unwrap();

        // Write valid records
        {
            let wal = WalManager::open(dir.path()).unwrap();
            wal.log(&WalRecord::CreateNode {
                id: NodeId::new(1),
                labels: vec!["First".to_string()],
            })
            .unwrap();
            wal.log(&WalRecord::TransactionCommit {
                transaction_id: TransactionId::new(1),
            })
            .unwrap();
            wal.sync().unwrap();
        }

        // Find the WAL file and corrupt a byte in the data section
        let wal_files: Vec<_> = std::fs::read_dir(dir.path())
            .unwrap()
            .filter_map(|e| {
                let e = e.ok()?;
                if e.path().extension().is_some_and(|ext| ext == "log") {
                    Some(e.path())
                } else {
                    None
                }
            })
            .collect();
        assert!(!wal_files.is_empty(), "wal_files is empty");

        let mut data = std::fs::read(&wal_files[0]).unwrap();
        // Flip a byte in the middle of the data (after the 4-byte length prefix)
        if data.len() > 8 {
            data[6] ^= 0xFF;
        }
        std::fs::write(&wal_files[0], &data).unwrap();

        // Recovery should handle corruption gracefully (not panic)
        let recovery = WalRecovery::new(dir.path());
        let result = recovery.recover();
        // Should either succeed with fewer records or return an error
        match result {
            Ok(records) => {
                // Best-effort: may have recovered 0 records due to corruption
                assert!(records.len() <= 2);
            }
            Err(_) => {
                // Also acceptable: corruption detection as error
            }
        }
    }

    #[test]
    fn test_recovery_empty_wal_file() {
        let dir = tempdir().unwrap();

        // Create an empty WAL file
        std::fs::write(dir.path().join("wal_00000000.log"), []).unwrap();

        let recovery = WalRecovery::new(dir.path());
        let records = recovery.recover().unwrap();
        assert_eq!(records.len(), 0, "empty WAL should produce no records");
    }

    // ── recover_until_epoch tests ─────────────────────────────────────

    /// Helper: writes a committed transaction with an EpochAdvance marker,
    /// matching the pattern used by the engine session commit path.
    fn write_tx_with_epoch(wal: &WalManager, node_id: u64, tx_id: u64, epoch: u64) {
        wal.log(&WalRecord::CreateNode {
            id: NodeId::new(node_id),
            labels: vec![format!("N{node_id}")],
        })
        .unwrap();
        wal.log(&WalRecord::TransactionCommit {
            transaction_id: TransactionId::new(tx_id),
        })
        .unwrap();
        wal.log(&WalRecord::EpochAdvance {
            epoch: EpochId::new(epoch),
        })
        .unwrap();
    }

    #[test]
    fn test_recover_until_epoch_includes_target() {
        let dir = tempdir().unwrap();
        {
            let wal = WalManager::open(dir.path()).unwrap();
            write_tx_with_epoch(&wal, 1, 1, 5);
            write_tx_with_epoch(&wal, 2, 2, 10);
            write_tx_with_epoch(&wal, 3, 3, 15);
            wal.sync().unwrap();
        }

        let recovery = WalRecovery::new(dir.path());
        let records = recovery.recover_until_epoch(EpochId::new(10)).unwrap();

        // Should include tx1 (epoch 5) and tx2 (epoch 10), each producing
        // CreateNode + TxCommit + EpochAdvance = 3 records per tx.
        assert_eq!(records.len(), 6, "should include epochs 5 and 10");

        // Verify no records from epoch 15
        assert!(
            !records.iter().any(
                |r| matches!(r, WalRecord::EpochAdvance { epoch } if *epoch > EpochId::new(10))
            ),
            "should not include EpochAdvance beyond target"
        );
    }

    #[test]
    fn test_recover_until_epoch_excludes_next_epoch() {
        let dir = tempdir().unwrap();
        {
            let wal = WalManager::open(dir.path()).unwrap();
            write_tx_with_epoch(&wal, 1, 1, 5);
            write_tx_with_epoch(&wal, 2, 2, 6);
            wal.sync().unwrap();
        }

        let recovery = WalRecovery::new(dir.path());
        let records = recovery.recover_until_epoch(EpochId::new(5)).unwrap();

        // Only tx1 (epoch 5): CreateNode + TxCommit + EpochAdvance = 3
        assert_eq!(records.len(), 3, "should exclude epoch 6 transaction");

        // Verify the CreateNode is for node 1 only
        let nodes: Vec<_> = records
            .iter()
            .filter_map(|r| match r {
                WalRecord::CreateNode { id, .. } => Some(id.as_u64()),
                _ => None,
            })
            .collect();
        assert_eq!(nodes, vec![1], "only node from epoch 5 should be present");
    }

    #[test]
    fn test_recover_until_epoch_zero_returns_empty() {
        let dir = tempdir().unwrap();
        {
            let wal = WalManager::open(dir.path()).unwrap();
            write_tx_with_epoch(&wal, 1, 1, 5);
            wal.sync().unwrap();
        }

        let recovery = WalRecovery::new(dir.path());
        let records = recovery.recover_until_epoch(EpochId::new(0)).unwrap();

        // Epoch 5 > 0, so all records should be excluded
        assert!(
            records.is_empty(),
            "no records should be at or below epoch 0"
        );
    }

    #[test]
    fn test_recover_until_epoch_beyond_max_returns_all() {
        let dir = tempdir().unwrap();
        {
            let wal = WalManager::open(dir.path()).unwrap();
            write_tx_with_epoch(&wal, 1, 1, 5);
            write_tx_with_epoch(&wal, 2, 2, 10);
            wal.sync().unwrap();
        }

        let recovery = WalRecovery::new(dir.path());
        let records = recovery.recover_until_epoch(EpochId::new(999)).unwrap();

        // Both transactions are within range: 6 records total
        assert_eq!(records.len(), 6, "all records should be included");
    }

    #[test]
    fn test_recover_until_epoch_empty_wal() {
        let dir = tempdir().unwrap();
        std::fs::create_dir_all(dir.path()).unwrap();

        let recovery = WalRecovery::new(dir.path());
        let records = recovery.recover_until_epoch(EpochId::new(100)).unwrap();
        assert!(records.is_empty());
    }
}

/// Crash injection tests for WAL recovery.
///
/// These tests verify that WAL recovery produces a consistent state after
/// simulated crashes at every crash point in the write path. The three crash
/// points are:
/// - `wal_before_write`: before writing length prefix + data + checksum
/// - `wal_after_write`: after writing data but before durability handling
/// - `wal_before_flush`: before fsync on TransactionCommit in Sync mode
///
/// Run with:
/// ```bash
/// cargo test -p grafeo-adapters --features "wal,testing-crash-injection" -- crash
/// ```
#[cfg(all(test, feature = "testing-crash-injection"))]
mod crash_tests {
    use super::*;
    use grafeo_common::testing::crash::{CrashResult, with_crash_at};
    use grafeo_common::types::{EpochId, NodeId, TransactionId, Value};
    use tempfile::tempdir;

    /// Helper: Sync durability config so all three crash points are reachable.
    fn sync_config() -> super::super::WalConfig {
        super::super::WalConfig {
            durability: super::super::DurabilityMode::Sync,
            ..Default::default()
        }
    }

    /// Crash at `wal_before_write`: no record bytes reach disk.
    /// Recovery should only return previously committed data.
    #[test]
    fn test_crash_before_write_discards_record() {
        let dir = tempdir().unwrap();
        let path = dir.path().to_path_buf();

        // Seed one committed transaction
        {
            let wal = WalManager::with_config(&path, sync_config()).unwrap();
            wal.log(&WalRecord::CreateNode {
                id: NodeId::new(1),
                labels: vec!["Committed".into()],
            })
            .unwrap();
            wal.log(&WalRecord::TransactionCommit {
                transaction_id: TransactionId::new(1),
            })
            .unwrap();
        }

        // Crash at the first crash point (wal_before_write)
        let p = path.clone();
        let result = with_crash_at(1, move || {
            let wal = WalManager::with_config(&p, sync_config()).unwrap();
            wal.log(&WalRecord::CreateNode {
                id: NodeId::new(2),
                labels: vec!["Lost".into()],
            })
            .unwrap();
        });
        assert!(matches!(result, CrashResult::Crashed));

        // Only the first committed tx should survive
        let recovery = WalRecovery::new(&path);
        let records = recovery.recover().unwrap();
        assert_eq!(records.len(), 2, "CreateNode(1) + TransactionCommit(1)");
    }

    /// Crash at `wal_after_write`: data may be in BufWriter but no commit
    /// marker. Recovery should discard the uncommitted record.
    #[test]
    fn test_crash_after_write_uncommitted_discarded() {
        let dir = tempdir().unwrap();
        let path = dir.path().to_path_buf();

        // For a non-commit record the crash points are:
        //   1 = wal_before_write, 2 = wal_after_write
        let p = path.clone();
        let result = with_crash_at(2, move || {
            let wal = WalManager::with_config(&p, sync_config()).unwrap();
            wal.log(&WalRecord::CreateNode {
                id: NodeId::new(1),
                labels: vec!["Partial".into()],
            })
            .unwrap();
        });
        assert!(matches!(result, CrashResult::Crashed));

        // No committed tx ⇒ recovery returns nothing
        let recovery = WalRecovery::new(&path);
        let records = recovery.recover().unwrap();
        assert_eq!(records.len(), 0, "Uncommitted records must be discarded");
    }

    /// Two committed transactions, then crash during the third.
    /// Recovery should preserve exactly the first two.
    #[test]
    fn test_crash_preserves_prior_committed_transactions() {
        let dir = tempdir().unwrap();
        let path = dir.path().to_path_buf();

        // Commit two transactions
        {
            let wal = WalManager::with_config(&path, sync_config()).unwrap();
            wal.log(&WalRecord::CreateNode {
                id: NodeId::new(1),
                labels: vec!["T1".into()],
            })
            .unwrap();
            wal.log(&WalRecord::TransactionCommit {
                transaction_id: TransactionId::new(1),
            })
            .unwrap();
            wal.log(&WalRecord::CreateNode {
                id: NodeId::new(2),
                labels: vec!["T2".into()],
            })
            .unwrap();
            wal.log(&WalRecord::TransactionCommit {
                transaction_id: TransactionId::new(2),
            })
            .unwrap();
        }

        // Third transaction crashes immediately
        let p = path.clone();
        let result = with_crash_at(1, move || {
            let wal = WalManager::with_config(&p, sync_config()).unwrap();
            wal.log(&WalRecord::CreateNode {
                id: NodeId::new(3),
                labels: vec!["T3".into()],
            })
            .unwrap();
        });
        assert!(matches!(result, CrashResult::Crashed));

        // Both committed txs intact, third discarded
        let recovery = WalRecovery::new(&path);
        let records = recovery.recover().unwrap();
        assert_eq!(records.len(), 4, "2 CreateNode + 2 TransactionCommit");
    }

    /// Crash during checkpoint: committed data must still be recoverable.
    #[test]
    fn test_crash_during_checkpoint_preserves_data() {
        for crash_at in 1..15 {
            let dir = tempdir().unwrap();
            let path = dir.path().to_path_buf();

            // Seed committed data
            {
                let wal = WalManager::with_config(&path, sync_config()).unwrap();
                wal.log(&WalRecord::CreateNode {
                    id: NodeId::new(1),
                    labels: vec!["A".into()],
                })
                .unwrap();
                wal.log(&WalRecord::TransactionCommit {
                    transaction_id: TransactionId::new(1),
                })
                .unwrap();
            }

            // Crash during checkpoint
            let p = path.clone();
            let _result = with_crash_at(crash_at, move || {
                let wal = WalManager::with_config(&p, sync_config()).unwrap();
                wal.checkpoint(TransactionId::new(1), EpochId::new(10))
                    .unwrap();
            });

            // Committed data must survive regardless of checkpoint outcome
            let recovery = WalRecovery::new(&path);
            let records = recovery.recover().unwrap();
            assert!(
                !records.is_empty(),
                "crash_at={crash_at}: committed data must survive checkpoint crash"
            );
        }
    }

    /// Crash with rotated log files: recovery should span all files.
    #[test]
    fn test_crash_with_log_rotation() {
        let dir = tempdir().unwrap();
        let path = dir.path().to_path_buf();

        // Write enough to trigger rotation
        {
            let config = super::super::WalConfig {
                durability: super::super::DurabilityMode::Sync,
                max_log_size: 100, // force rotation
                ..Default::default()
            };
            let wal = WalManager::with_config(&path, config).unwrap();
            for i in 0..5 {
                wal.log(&WalRecord::CreateNode {
                    id: NodeId::new(i),
                    labels: vec!["Rotated".into()],
                })
                .unwrap();
            }
            wal.log(&WalRecord::TransactionCommit {
                transaction_id: TransactionId::new(1),
            })
            .unwrap();
        }

        // Crash during additional write
        let p = path.clone();
        let result = with_crash_at(1, move || {
            let config = super::super::WalConfig {
                durability: super::super::DurabilityMode::Sync,
                max_log_size: 100,
                ..Default::default()
            };
            let wal = WalManager::with_config(&p, config).unwrap();
            wal.log(&WalRecord::CreateNode {
                id: NodeId::new(99),
                labels: vec!["Crash".into()],
            })
            .unwrap();
        });
        assert!(matches!(result, CrashResult::Crashed));

        // All committed data across rotated files should survive
        let recovery = WalRecovery::new(&path);
        let records = recovery.recover().unwrap();
        assert_eq!(records.len(), 6, "5 CreateNode + 1 TransactionCommit");
    }

    /// Exhaustive sweep: crash at every possible point during a multi-record
    /// transaction and verify recovery invariants.
    ///
    /// Invariants checked:
    /// 1. Previously committed transactions always survive
    /// 2. Recovery output never contains partial (uncommitted) transactions
    #[test]
    fn test_crash_sweep_all_points() {
        for crash_at in 1..20 {
            let dir = tempdir().unwrap();
            let path = dir.path().to_path_buf();

            // Seed one committed transaction
            {
                let wal = WalManager::with_config(&path, sync_config()).unwrap();
                wal.log(&WalRecord::CreateNode {
                    id: NodeId::new(1),
                    labels: vec!["Base".into()],
                })
                .unwrap();
                wal.log(&WalRecord::TransactionCommit {
                    transaction_id: TransactionId::new(1),
                })
                .unwrap();
            }

            // Attempt a second transaction with crash injection
            let p = path.clone();
            let result = with_crash_at(crash_at, move || {
                let wal = WalManager::with_config(&p, sync_config()).unwrap();
                wal.log(&WalRecord::CreateNode {
                    id: NodeId::new(100),
                    labels: vec!["New".into()],
                })
                .unwrap();
                wal.log(&WalRecord::SetNodeProperty {
                    id: NodeId::new(100),
                    key: "name".into(),
                    value: Value::String("test".into()),
                })
                .unwrap();
                wal.log(&WalRecord::TransactionCommit {
                    transaction_id: TransactionId::new(2),
                })
                .unwrap();
            });

            // Verify recovery invariants
            let recovery = WalRecovery::new(&path);
            let records = recovery.recover().unwrap();

            // Invariant 1: base committed tx always survives
            assert!(
                records.len() >= 2,
                "crash_at={crash_at}: base tx must survive, got {} records",
                records.len()
            );

            // Invariant 2: no partial transactions in output
            let mut pending = 0usize;
            for record in &records {
                match record {
                    WalRecord::TransactionCommit { .. }
                    | WalRecord::TransactionAbort { .. }
                    | WalRecord::Checkpoint { .. } => pending = 0,
                    _ => pending += 1,
                }
            }
            assert_eq!(
                pending, 0,
                "crash_at={crash_at}: recovery must not output partial transactions"
            );

            // If the operation completed, the second tx should also be present
            if matches!(result, CrashResult::Completed(())) {
                assert!(
                    records.len() >= 5,
                    "crash_at={crash_at}: completed run should include second tx"
                );
            }
        }
    }

    /// Aborted transactions are not recovered even without a crash.
    /// Verifies that TransactionAbort correctly discards pending records.
    #[test]
    fn test_abort_then_crash_discards_aborted_tx() {
        let dir = tempdir().unwrap();
        let path = dir.path().to_path_buf();

        {
            let wal = WalManager::with_config(&path, sync_config()).unwrap();
            // Committed tx
            wal.log(&WalRecord::CreateNode {
                id: NodeId::new(1),
                labels: vec!["Keep".into()],
            })
            .unwrap();
            wal.log(&WalRecord::TransactionCommit {
                transaction_id: TransactionId::new(1),
            })
            .unwrap();
            // Aborted tx
            wal.log(&WalRecord::CreateNode {
                id: NodeId::new(2),
                labels: vec!["Discard".into()],
            })
            .unwrap();
            wal.log(&WalRecord::TransactionAbort {
                transaction_id: TransactionId::new(2),
            })
            .unwrap();
        }

        // Crash during a third transaction
        let p = path.clone();
        let result = with_crash_at(1, move || {
            let wal = WalManager::with_config(&p, sync_config()).unwrap();
            wal.log(&WalRecord::CreateNode {
                id: NodeId::new(3),
                labels: vec!["Also lost".into()],
            })
            .unwrap();
        });
        assert!(matches!(result, CrashResult::Crashed));

        let recovery = WalRecovery::new(&path);
        let records = recovery.recover().unwrap();
        // Only the committed tx (2 records)
        assert_eq!(
            records.len(),
            2,
            "Aborted + crashed records should both be discarded"
        );
    }

    /// A recovery made from an encrypted WAL manager decrypts with its key.
    #[cfg(all(feature = "encryption", not(miri)))]
    #[test]
    fn recovery_from_an_encrypted_wal_keeps_its_cipher() {
        use grafeo_common::encryption::KeyChain;

        let dir = tempdir().unwrap();
        let chain = KeyChain::new([3; 32]);
        let wal = WalManager::with_config_and_cipher(
            dir.path(),
            super::super::WalConfig::default(),
            Some(chain.encryptor_for("grafeo-wal", &19u128.to_le_bytes())),
        )
        .unwrap();
        wal.log(&WalRecord::CreateNode {
            id: NodeId::new(88),
            labels: vec!["Person".to_string()],
        })
        .unwrap();
        wal.log(&WalRecord::TransactionCommit {
            transaction_id: TransactionId::new(3),
        })
        .unwrap();
        wal.flush().unwrap();

        let records = WalRecovery::from_wal(&wal).recover().unwrap();
        assert_eq!(records.len(), 2, "both records decrypt: {records:?}");
        assert!(
            WalRecovery::new(dir.path()).recover().unwrap().is_empty(),
            "without the cipher nothing reads"
        );
    }

    #[cfg(all(feature = "encryption", not(miri)))]
    #[test]
    fn test_encrypted_wal_roundtrip() {
        use grafeo_common::encryption::{KEY_SIZE, KeyChain};

        let dir = tempdir().unwrap();
        let key = [42u8; KEY_SIZE];
        let chain = KeyChain::new(key);

        // Write encrypted records
        {
            let mut wal = WalManager::open(dir.path()).unwrap();
            wal.set_encryptor(chain.encryptor_for("grafeo-wal", &0u64.to_be_bytes()));

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

        // Recover with same key
        let mut recovery = WalRecovery::new(dir.path());
        recovery.set_encryptor(chain.encryptor_for("grafeo-wal", &0u64.to_be_bytes()));
        let records = recovery.recover().unwrap();
        assert_eq!(records.len(), 2, "should recover both encrypted records");
    }

    #[cfg(all(feature = "encryption", not(miri)))]
    #[test]
    fn test_encrypted_wal_wrong_key_fails() {
        use grafeo_common::encryption::{KEY_SIZE, KeyChain};

        let dir = tempdir().unwrap();
        let key = [42u8; KEY_SIZE];
        let chain = KeyChain::new(key);

        // Write with key A
        {
            let mut wal = WalManager::open(dir.path()).unwrap();
            wal.set_encryptor(chain.encryptor_for("grafeo-wal", &0u64.to_be_bytes()));

            wal.log(&WalRecord::CreateNode {
                id: NodeId::new(1),
                labels: vec!["Test".to_string()],
            })
            .unwrap();

            wal.log(&WalRecord::TransactionCommit {
                transaction_id: TransactionId::new(1),
            })
            .unwrap();

            wal.sync().unwrap();
        }

        // Try recovery with wrong key B: decryption fails, treated as corruption,
        // best-effort recovery returns no usable records
        let wrong_key = [99u8; KEY_SIZE];
        let wrong_chain = KeyChain::new(wrong_key);
        let mut recovery = WalRecovery::new(dir.path());
        recovery.set_encryptor(wrong_chain.encryptor_for("grafeo-wal", &0u64.to_be_bytes()));

        let result = recovery.recover();
        assert!(
            result.is_ok(),
            "recovery should succeed (best-effort), got: {result:?}"
        );
        let records = result.unwrap();
        assert!(
            records.is_empty(),
            "wrong key should produce no recoverable records, got {}",
            records.len()
        );
    }
}
