//! WAL recovery.

use super::WalRecord;
use graphos_common::utils::error::{Error, Result, StorageError};
use std::fs::File;
use std::io::{BufReader, Read};
use std::path::Path;

/// Handles WAL recovery after a crash.
pub struct WalRecovery {
    /// Path to the WAL file.
    path: std::path::PathBuf,
}

impl WalRecovery {
    /// Creates a new recovery handler for the given WAL file.
    pub fn new(path: impl AsRef<Path>) -> Self {
        Self {
            path: path.as_ref().to_path_buf(),
        }
    }

    /// Recovers committed records from the WAL.
    ///
    /// Returns only records that were part of committed transactions.
    ///
    /// # Errors
    ///
    /// Returns an error if recovery fails.
    pub fn recover(&self) -> Result<Vec<WalRecord>> {
        let file = File::open(&self.path)?;
        let mut reader = BufReader::new(file);

        let mut current_tx_records = Vec::new();
        let mut committed_records = Vec::new();

        // Read all records
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
                        _ => {
                            current_tx_records.push(record);
                        }
                    }
                }
                Ok(None) => break, // EOF
                Err(e) => {
                    // Log corruption - stop reading
                    tracing::warn!("WAL corruption detected: {}", e);
                    break;
                }
            }
        }

        // Uncommitted records in current_tx_records are discarded

        Ok(committed_records)
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
    use crate::storage::wal::WalManager;
    use graphos_common::types::{NodeId, TxId};
    use tempfile::tempdir;

    #[test]
    fn test_recovery_committed() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.wal");

        // Write some records
        {
            let wal = WalManager::open(&path).unwrap();

            wal.log(&WalRecord::CreateNode {
                id: NodeId::new(1),
                labels: vec!["Person".to_string()],
            })
            .unwrap();

            wal.log(&WalRecord::TxCommit {
                tx_id: TxId::new(1),
            })
            .unwrap();

            wal.flush().unwrap();
        }

        // Recover
        let recovery = WalRecovery::new(&path);
        let records = recovery.recover().unwrap();

        assert_eq!(records.len(), 2);
    }

    #[test]
    fn test_recovery_uncommitted() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.wal");

        // Write some records without commit
        {
            let wal = WalManager::open(&path).unwrap();

            wal.log(&WalRecord::CreateNode {
                id: NodeId::new(1),
                labels: vec!["Person".to_string()],
            })
            .unwrap();

            // No commit!
            wal.flush().unwrap();
        }

        // Recover
        let recovery = WalRecovery::new(&path);
        let records = recovery.recover().unwrap();

        // Uncommitted records should be discarded
        assert_eq!(records.len(), 0);
    }
}
