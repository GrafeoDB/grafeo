//! WAL log file management.

use super::WalRecord;
use graphos_common::utils::error::{Error, Result, StorageError};
use parking_lot::Mutex;
use std::fs::{File, OpenOptions};
use std::io::{BufReader, BufWriter, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

/// Manages the Write-Ahead Log.
pub struct WalManager {
    /// Path to the WAL file.
    path: PathBuf,
    /// Writer for appending records.
    writer: Mutex<Option<BufWriter<File>>>,
    /// Number of records written.
    record_count: Mutex<u64>,
}

impl WalManager {
    /// Opens or creates a WAL file at the given path.
    ///
    /// # Errors
    ///
    /// Returns an error if the file cannot be opened or created.
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref().to_path_buf();

        let file = OpenOptions::new()
            .create(true)
            .read(true)
            .append(true)
            .open(&path)?;

        let writer = BufWriter::new(file);

        Ok(Self {
            path,
            writer: Mutex::new(Some(writer)),
            record_count: Mutex::new(0),
        })
    }

    /// Logs a record to the WAL.
    ///
    /// # Errors
    ///
    /// Returns an error if the record cannot be written.
    pub fn log(&self, record: &WalRecord) -> Result<()> {
        let mut guard = self.writer.lock();
        let writer = guard
            .as_mut()
            .ok_or_else(|| Error::Internal("WAL writer not available".to_string()))?;

        // Serialize the record
        let data = bincode::serde::encode_to_vec(record, bincode::config::standard())
            .map_err(|e| Error::Serialization(e.to_string()))?;

        // Write length prefix
        let len = data.len() as u32;
        writer.write_all(&len.to_le_bytes())?;

        // Write data
        writer.write_all(&data)?;

        // Write checksum
        let checksum = crc32fast::hash(&data);
        writer.write_all(&checksum.to_le_bytes())?;

        *self.record_count.lock() += 1;

        Ok(())
    }

    /// Flushes the WAL to disk.
    ///
    /// # Errors
    ///
    /// Returns an error if the flush fails.
    pub fn flush(&self) -> Result<()> {
        let mut guard = self.writer.lock();
        if let Some(writer) = guard.as_mut() {
            writer.flush()?;
        }
        Ok(())
    }

    /// Syncs the WAL to disk (fsync).
    ///
    /// # Errors
    ///
    /// Returns an error if the sync fails.
    pub fn sync(&self) -> Result<()> {
        let mut guard = self.writer.lock();
        if let Some(writer) = guard.as_mut() {
            writer.flush()?;
            writer.get_ref().sync_all()?;
        }
        Ok(())
    }

    /// Returns the number of records written.
    #[must_use]
    pub fn record_count(&self) -> u64 {
        *self.record_count.lock()
    }

    /// Returns the path to the WAL file.
    #[must_use]
    pub fn path(&self) -> &Path {
        &self.path
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use graphos_common::types::NodeId;
    use tempfile::tempdir;

    #[test]
    fn test_wal_write() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.wal");

        let wal = WalManager::open(&path).unwrap();

        let record = WalRecord::CreateNode {
            id: NodeId::new(1),
            labels: vec!["Person".to_string()],
        };

        wal.log(&record).unwrap();
        wal.flush().unwrap();

        assert_eq!(wal.record_count(), 1);
    }
}
