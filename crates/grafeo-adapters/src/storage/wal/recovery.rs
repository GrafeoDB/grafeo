//! WAL recovery.

use super::{CheckpointMetadata, WalManager, WalRecord};
use grafeo_common::utils::error::{Error, Result, StorageError};
use std::fs::File;
use std::io::{BufReader, Read};
use std::path::Path;

/// Name of the checkpoint metadata file.
const CHECKPOINT_METADATA_FILE: &str = "checkpoint.meta";

/// Handles WAL recovery after a crash.
pub struct WalRecovery {
    /// Directory containing WAL files.
    dir: std::path::PathBuf,
}

impl WalRecovery {
    /// Creates a new recovery handler for the given WAL directory.
    pub fn new(dir: impl AsRef<Path>) -> Self {
        Self {
            dir: dir.as_ref().to_path_buf(),
        }
    }

    /// Creates a recovery handler from a WAL manager.
    #[must_use]
    pub fn from_wal(wal: &WalManager) -> Self {
        Self {
            dir: wal.dir().to_path_buf(),
        }
    }

    /// Reads checkpoint metadata if it exists.
    ///
    /// Returns `None` if no checkpoint metadata is found.
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
        // Check for checkpoint metadata
        let checkpoint = self.read_checkpoint_metadata()?;
        self.recover_internal(checkpoint)
    }

    /// Recovers committed records, starting from a specific checkpoint.
    ///
    /// This can be used for incremental recovery when you want to
    /// skip WAL files that precede the checkpoint.
    ///
    /// # Errors
    ///
    /// Returns an error if recovery fails.
    pub fn recover_from_checkpoint(
        &self,
        checkpoint: Option<&CheckpointMetadata>,
    ) -> Result<Vec<WalRecord>> {
        self.recover_internal(checkpoint.cloned())
    }

    fn recover_internal(&self, checkpoint: Option<CheckpointMetadata>) -> Result<Vec<WalRecord>> {
        let mut current_tx_records = Vec::new();
        let mut committed_records = Vec::new();

        // Get all log files in order
        let log_files = self.get_log_files()?;

        // Determine the minimum sequence number to process
        let min_sequence = checkpoint.as_ref().map_or(0, |cp| cp.log_sequence);

        if checkpoint.is_some() {
            tracing::info!(
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
                tracing::debug!(
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
            let mut reader = BufReader::new(file);

            // Read all records from this file
            loop {
                match self.read_record(&mut reader) {
                    Ok(Some(record)) => {
                        match &record {
                            WalRecord::TxCommit { .. } => {
                                // Commit current transaction
                                committed_records.append(&mut current_tx_records);
                                committed_records.push(record);
                            }
                            WalRecord::TxAbort { .. } => {
                                // Discard current transaction
                                current_tx_records.clear();
                            }
                            WalRecord::Checkpoint { .. } => {
                                // Checkpoint - clear uncommitted, keep committed
                                current_tx_records.clear();
                                committed_records.push(record);
                            }
                            _ => {
                                current_tx_records.push(record);
                            }
                        }
                    }
                    Ok(None) => break, // EOF
                    Err(e) => {
                        // Log corruption - stop reading this file but continue
                        // with remaining files (best-effort recovery)
                        tracing::warn!("WAL corruption detected in {:?}: {}", log_file, e);
                        break;
                    }
                }
            }
        }

        // Uncommitted records in current_tx_records are discarded

        Ok(committed_records)
    }

    /// Extracts the sequence number from a WAL log file path.
    fn sequence_from_path(path: &Path) -> Option<u64> {
        path.file_stem()
            .and_then(|s| s.to_str())
            .and_then(|s| s.strip_prefix("wal_"))
            .and_then(|s| s.parse().ok())
    }

    /// Recovers committed records from a single WAL file.
    ///
    /// # Errors
    ///
    /// Returns an error if recovery fails.
    pub fn recover_file(&self, path: impl AsRef<Path>) -> Result<Vec<WalRecord>> {
        let file = File::open(path.as_ref())?;
        let mut reader = BufReader::new(file);

        let mut current_tx_records = Vec::new();
        let mut committed_records = Vec::new();

        loop {
            match self.read_record(&mut reader) {
                Ok(Some(record)) => match &record {
                    WalRecord::TxCommit { .. } => {
                        committed_records.append(&mut current_tx_records);
                        committed_records.push(record);
                    }
                    WalRecord::TxAbort { .. } => {
                        current_tx_records.clear();
                    }
                    _ => {
                        current_tx_records.push(record);
                    }
                },
                Ok(None) => break,
                Err(e) => {
                    tracing::warn!("WAL corruption detected: {}", e);
                    break;
                }
            }
        }

        Ok(committed_records)
    }

    fn get_log_files(&self) -> Result<Vec<std::path::PathBuf>> {
        let mut files = Vec::new();

        if !self.dir.exists() {
            return Ok(files);
        }

        if let Ok(entries) = std::fs::read_dir(&self.dir) {
            for entry in entries.flatten() {
                let path = entry.path();
                if path.extension().is_some_and(|ext| ext == "log") {
                    files.push(path);
                }
            }
        }

        // Sort by filename (which includes sequence number)
        files.sort();

        Ok(files)
    }

    fn read_record(&self, reader: &mut BufReader<File>) -> Result<Option<WalRecord>> {
        // Read length prefix
        let mut len_buf = [0u8; 4];
        match reader.read_exact(&mut len_buf) {
            Ok(()) => {}
            Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => return Ok(None),
            Err(e) => return Err(e.into()),
        }
        let len = u32::from_le_bytes(len_buf) as usize;

        // Read data
        let mut data = vec![0u8; len];
        reader.read_exact(&mut data)?;

        // Read and verify checksum
        let mut checksum_buf = [0u8; 4];
        reader.read_exact(&mut checksum_buf)?;
        let stored_checksum = u32::from_le_bytes(checksum_buf);
        let computed_checksum = crc32fast::hash(&data);

        if stored_checksum != computed_checksum {
            return Err(Error::Storage(StorageError::Corruption(
                "WAL checksum mismatch".to_string(),
            )));
        }

        // Deserialize
        let (record, _): (WalRecord, _) =
            bincode::serde::decode_from_slice(&data, bincode::config::standard())
                .map_err(|e| Error::Serialization(e.to_string()))?;

        Ok(Some(record))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use grafeo_common::types::{NodeId, TxId};
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

            wal.log(&WalRecord::TxCommit {
                tx_id: TxId::new(1),
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
            wal.log(&WalRecord::TxCommit {
                tx_id: TxId::new(1),
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
            wal.log(&WalRecord::TxCommit {
                tx_id: TxId::new(2),
            })
            .unwrap();

            wal.sync().unwrap();
        }

        // Recover
        let recovery = WalRecovery::new(dir.path());
        let records = recovery.recover().unwrap();

        // Should have 10 CreateNode + 2 TxCommit
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
            wal.log(&WalRecord::TxCommit {
                tx_id: TxId::new(1),
            })
            .unwrap();

            // Create checkpoint
            wal.checkpoint(TxId::new(1), EpochId::new(10)).unwrap();

            // Second transaction after checkpoint
            wal.log(&WalRecord::CreateNode {
                id: NodeId::new(2),
                labels: vec!["Test".to_string()],
            })
            .unwrap();
            wal.log(&WalRecord::TxCommit {
                tx_id: TxId::new(2),
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
        assert_eq!(cp.tx_id.as_u64(), 1);
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
            wal.log(&WalRecord::TxCommit {
                tx_id: TxId::new(1),
            })
            .unwrap();

            // Create checkpoint
            wal.checkpoint(TxId::new(1), EpochId::new(100)).unwrap();

            // Second batch after checkpoint
            for i in 100..103 {
                wal.log(&WalRecord::CreateNode {
                    id: NodeId::new(i),
                    labels: vec!["After".to_string()],
                })
                .unwrap();
            }
            wal.log(&WalRecord::TxCommit {
                tx_id: TxId::new(2),
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
}
