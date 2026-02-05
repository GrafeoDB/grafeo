//! WAL log file management.

use super::WalRecord;
use grafeo_common::types::{EpochId, TxId};
use grafeo_common::utils::error::{Error, Result};
use parking_lot::Mutex;
use serde::{Deserialize, Serialize};
use std::fs::{self, File, OpenOptions};
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

/// Checkpoint metadata stored in a separate file.
///
/// This file is written atomically (via rename) during checkpoint and read
/// during recovery to determine which WAL files can be skipped.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CheckpointMetadata {
    /// The epoch at which the checkpoint was taken.
    pub epoch: EpochId,
    /// The log sequence number at the time of checkpoint.
    pub log_sequence: u64,
    /// Timestamp of the checkpoint (milliseconds since UNIX epoch).
    pub timestamp_ms: u64,
    /// Transaction ID at checkpoint.
    pub tx_id: TxId,
}

/// Name of the checkpoint metadata file.
const CHECKPOINT_METADATA_FILE: &str = "checkpoint.meta";

/// Durability mode for the WAL.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DurabilityMode {
    /// Sync (fsync) after every commit for maximum durability.
    /// Slowest but safest.
    Sync,
    /// Batch sync - fsync periodically (e.g., every N ms or N records).
    /// Good balance of performance and durability.
    Batch {
        /// Maximum time between syncs in milliseconds.
        max_delay_ms: u64,
        /// Maximum records between syncs.
        max_records: u64,
    },
    /// Adaptive sync - background thread adjusts timing based on flush duration.
    ///
    /// Unlike `Batch` which checks thresholds inline, `Adaptive` spawns a
    /// dedicated flusher thread that maintains consistent flush cadence
    /// regardless of disk speed. Use [`AdaptiveFlusher`](super::AdaptiveFlusher)
    /// to manage the background thread.
    ///
    /// The WAL itself only buffers writes; the flusher thread handles syncing.
    Adaptive {
        /// Target interval between flushes in milliseconds.
        /// The flusher adjusts wait times to maintain this cadence.
        target_interval_ms: u64,
    },
    /// No sync - rely on OS buffer flushing.
    /// Fastest but may lose recent data on crash.
    NoSync,
}

impl Default for DurabilityMode {
    fn default() -> Self {
        Self::Batch {
            max_delay_ms: 100,
            max_records: 1000,
        }
    }
}

/// Configuration for the WAL manager.
#[derive(Debug, Clone)]
pub struct WalConfig {
    /// Durability mode.
    pub durability: DurabilityMode,
    /// Maximum log file size before rotation (in bytes).
    pub max_log_size: u64,
    /// Whether to enable compression.
    pub compression: bool,
}

impl Default for WalConfig {
    fn default() -> Self {
        Self {
            durability: DurabilityMode::default(),
            max_log_size: 64 * 1024 * 1024, // 64 MB
            compression: false,
        }
    }
}

/// State for a single log file.
struct LogFile {
    /// File handle.
    writer: BufWriter<File>,
    /// Current size in bytes.
    size: u64,
    /// File path.
    path: PathBuf,
    /// Sequence number (for log file ordering during recovery).
    #[allow(dead_code)]
    sequence: u64,
}

/// Manages the Write-Ahead Log with rotation, checkpointing, and durability modes.
pub struct WalManager {
    /// Directory for WAL files.
    dir: PathBuf,
    /// Configuration.
    config: WalConfig,
    /// Active log file.
    active_log: Mutex<Option<LogFile>>,
    /// Total number of records written across all log files.
    total_record_count: AtomicU64,
    /// Records since last sync (for batch mode).
    records_since_sync: AtomicU64,
    /// Time of last sync (for batch mode).
    last_sync: Mutex<Instant>,
    /// Current log sequence number.
    current_sequence: AtomicU64,
    /// Latest checkpoint epoch.
    checkpoint_epoch: Mutex<Option<EpochId>>,
}

impl WalManager {
    /// Opens or creates a WAL in the given directory.
    ///
    /// # Errors
    ///
    /// Returns an error if the directory cannot be created or accessed.
    pub fn open(dir: impl AsRef<Path>) -> Result<Self> {
        Self::with_config(dir, WalConfig::default())
    }

    /// Opens or creates a WAL with custom configuration.
    ///
    /// # Errors
    ///
    /// Returns an error if the directory cannot be created or accessed.
    pub fn with_config(dir: impl AsRef<Path>, config: WalConfig) -> Result<Self> {
        let dir = dir.as_ref().to_path_buf();
        fs::create_dir_all(&dir)?;

        // Find the highest existing sequence number
        let mut max_sequence = 0u64;
        if let Ok(entries) = fs::read_dir(&dir) {
            for entry in entries.flatten() {
                if let Some(name) = entry.file_name().to_str() {
                    if let Some(seq_str) = name
                        .strip_prefix("wal_")
                        .and_then(|s| s.strip_suffix(".log"))
                    {
                        if let Ok(seq) = seq_str.parse::<u64>() {
                            max_sequence = max_sequence.max(seq);
                        }
                    }
                }
            }
        }

        let manager = Self {
            dir,
            config,
            active_log: Mutex::new(None),
            total_record_count: AtomicU64::new(0),
            records_since_sync: AtomicU64::new(0),
            last_sync: Mutex::new(Instant::now()),
            current_sequence: AtomicU64::new(max_sequence),
            checkpoint_epoch: Mutex::new(None),
        };

        // Open or create the active log
        manager.ensure_active_log()?;

        Ok(manager)
    }

    /// Logs a record to the WAL.
    ///
    /// # Errors
    ///
    /// Returns an error if the record cannot be written.
    pub fn log(&self, record: &WalRecord) -> Result<()> {
        self.ensure_active_log()?;

        let mut guard = self.active_log.lock();
        let log_file = guard
            .as_mut()
            .ok_or_else(|| Error::Internal("WAL writer not available".to_string()))?;

        // Serialize the record
        let data = bincode::serde::encode_to_vec(record, bincode::config::standard())
            .map_err(|e| Error::Serialization(e.to_string()))?;

        // Write length prefix
        let len = data.len() as u32;
        log_file.writer.write_all(&len.to_le_bytes())?;

        // Write data
        log_file.writer.write_all(&data)?;

        // Write checksum
        let checksum = crc32fast::hash(&data);
        log_file.writer.write_all(&checksum.to_le_bytes())?;

        // Update size tracking
        let record_size = 4 + data.len() as u64 + 4; // length + data + checksum
        log_file.size += record_size;

        self.total_record_count.fetch_add(1, Ordering::Relaxed);
        self.records_since_sync.fetch_add(1, Ordering::Relaxed);

        // Check if we need to rotate
        let needs_rotation = log_file.size >= self.config.max_log_size;

        // Handle durability mode
        match &self.config.durability {
            DurabilityMode::Sync => {
                // Sync on every commit record
                if matches!(record, WalRecord::TxCommit { .. }) {
                    log_file.writer.flush()?;
                    log_file.writer.get_ref().sync_all()?;
                    self.records_since_sync.store(0, Ordering::Relaxed);
                    *self.last_sync.lock() = Instant::now();
                }
            }
            DurabilityMode::Batch {
                max_delay_ms,
                max_records,
            } => {
                let records = self.records_since_sync.load(Ordering::Relaxed);
                let elapsed = self.last_sync.lock().elapsed();

                if records >= *max_records || elapsed >= Duration::from_millis(*max_delay_ms) {
                    log_file.writer.flush()?;
                    log_file.writer.get_ref().sync_all()?;
                    self.records_since_sync.store(0, Ordering::Relaxed);
                    *self.last_sync.lock() = Instant::now();
                }
            }
            DurabilityMode::Adaptive { .. } => {
                // Adaptive mode: just flush buffer, background thread handles sync
                // The AdaptiveFlusher calls sync() periodically with self-tuning timing
                log_file.writer.flush()?;
            }
            DurabilityMode::NoSync => {
                // Just flush buffer, no sync
                log_file.writer.flush()?;
            }
        }

        drop(guard);

        // Rotate if needed
        if needs_rotation {
            self.rotate()?;
        }

        Ok(())
    }

    /// Writes a checkpoint marker and persists checkpoint metadata.
    ///
    /// The checkpoint metadata is written atomically to a separate file,
    /// allowing recovery to skip WAL files that precede the checkpoint.
    ///
    /// # Errors
    ///
    /// Returns an error if the checkpoint cannot be written.
    pub fn checkpoint(&self, current_tx: TxId, epoch: EpochId) -> Result<()> {
        // Write checkpoint record to WAL
        self.log(&WalRecord::Checkpoint { tx_id: current_tx })?;

        // Force sync on checkpoint
        self.sync()?;

        // Get current log sequence
        let log_sequence = self.current_sequence.load(Ordering::SeqCst);

        // Get current timestamp
        let timestamp_ms = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_millis() as u64)
            .unwrap_or(0);

        // Create checkpoint metadata
        let metadata = CheckpointMetadata {
            epoch,
            log_sequence,
            timestamp_ms,
            tx_id: current_tx,
        };

        // Write checkpoint metadata atomically
        self.write_checkpoint_metadata(&metadata)?;

        // Update in-memory checkpoint epoch
        *self.checkpoint_epoch.lock() = Some(epoch);

        // Optionally truncate old logs
        self.truncate_old_logs()?;

        Ok(())
    }

    /// Writes checkpoint metadata to disk atomically.
    ///
    /// Uses a write-to-temp-then-rename pattern for atomicity.
    fn write_checkpoint_metadata(&self, metadata: &CheckpointMetadata) -> Result<()> {
        let metadata_path = self.dir.join(CHECKPOINT_METADATA_FILE);
        let temp_path = self.dir.join(format!("{}.tmp", CHECKPOINT_METADATA_FILE));

        // Serialize metadata
        let data = bincode::serde::encode_to_vec(metadata, bincode::config::standard())
            .map_err(|e| Error::Serialization(e.to_string()))?;

        // Write to temp file
        let mut file = File::create(&temp_path)?;
        file.write_all(&data)?;
        file.sync_all()?;
        drop(file);

        // Atomic rename
        fs::rename(&temp_path, &metadata_path)?;

        Ok(())
    }

    /// Reads checkpoint metadata from disk.
    ///
    /// Returns `None` if no checkpoint metadata exists.
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

    /// Rotates to a new log file.
    ///
    /// # Errors
    ///
    /// Returns an error if rotation fails.
    pub fn rotate(&self) -> Result<()> {
        let new_sequence = self.current_sequence.fetch_add(1, Ordering::SeqCst) + 1;
        let new_path = self.log_path(new_sequence);

        let file = OpenOptions::new()
            .create(true)
            .read(true)
            .append(true)
            .open(&new_path)?;

        let new_log = LogFile {
            writer: BufWriter::new(file),
            size: 0,
            path: new_path,
            sequence: new_sequence,
        };

        // Replace active log
        let mut guard = self.active_log.lock();
        if let Some(old_log) = guard.take() {
            // Ensure old log is flushed
            drop(old_log);
        }
        *guard = Some(new_log);

        Ok(())
    }

    /// Flushes the WAL buffer to disk.
    ///
    /// # Errors
    ///
    /// Returns an error if the flush fails.
    pub fn flush(&self) -> Result<()> {
        let mut guard = self.active_log.lock();
        if let Some(log_file) = guard.as_mut() {
            log_file.writer.flush()?;
        }
        Ok(())
    }

    /// Syncs the WAL to disk (fsync).
    ///
    /// # Errors
    ///
    /// Returns an error if the sync fails.
    pub fn sync(&self) -> Result<()> {
        let mut guard = self.active_log.lock();
        if let Some(log_file) = guard.as_mut() {
            log_file.writer.flush()?;
            log_file.writer.get_ref().sync_all()?;
        }
        self.records_since_sync.store(0, Ordering::Relaxed);
        *self.last_sync.lock() = Instant::now();
        Ok(())
    }

    /// Returns the total number of records written.
    #[must_use]
    pub fn record_count(&self) -> u64 {
        self.total_record_count.load(Ordering::Relaxed)
    }

    /// Returns the WAL directory path.
    #[must_use]
    pub fn dir(&self) -> &Path {
        &self.dir
    }

    /// Returns the current durability mode.
    #[must_use]
    pub fn durability_mode(&self) -> DurabilityMode {
        self.config.durability
    }

    /// Returns all WAL log file paths in sequence order.
    pub fn log_files(&self) -> Result<Vec<PathBuf>> {
        let mut files = Vec::new();

        if let Ok(entries) = fs::read_dir(&self.dir) {
            for entry in entries.flatten() {
                let path = entry.path();
                if path.extension().is_some_and(|ext| ext == "log") {
                    files.push(path);
                }
            }
        }

        // Sort by sequence number
        files.sort_by(|a, b| {
            let seq_a = Self::sequence_from_path(a).unwrap_or(0);
            let seq_b = Self::sequence_from_path(b).unwrap_or(0);
            seq_a.cmp(&seq_b)
        });

        Ok(files)
    }

    /// Returns the latest checkpoint epoch, if any.
    #[must_use]
    pub fn checkpoint_epoch(&self) -> Option<EpochId> {
        *self.checkpoint_epoch.lock()
    }

    /// Returns the total size of all WAL files in bytes.
    #[must_use]
    pub fn size_bytes(&self) -> usize {
        let mut total = 0usize;
        if let Ok(files) = self.log_files() {
            for file in files {
                if let Ok(metadata) = fs::metadata(&file) {
                    total += metadata.len() as usize;
                }
            }
        }
        // Also include checkpoint metadata file
        let metadata_path = self.dir.join(CHECKPOINT_METADATA_FILE);
        if let Ok(metadata) = fs::metadata(&metadata_path) {
            total += metadata.len() as usize;
        }
        total
    }

    /// Returns the timestamp of the last checkpoint (Unix epoch seconds), if any.
    #[must_use]
    pub fn last_checkpoint_timestamp(&self) -> Option<u64> {
        if let Ok(Some(metadata)) = self.read_checkpoint_metadata() {
            // Convert milliseconds to seconds
            Some(metadata.timestamp_ms / 1000)
        } else {
            None
        }
    }

    // === Private methods ===

    fn ensure_active_log(&self) -> Result<()> {
        let mut guard = self.active_log.lock();
        if guard.is_none() {
            let sequence = self.current_sequence.load(Ordering::Relaxed);
            let path = self.log_path(sequence);

            let file = OpenOptions::new()
                .create(true)
                .read(true)
                .append(true)
                .open(&path)?;

            let size = file.metadata()?.len();

            *guard = Some(LogFile {
                writer: BufWriter::new(file),
                size,
                path,
                sequence,
            });
        }
        Ok(())
    }

    fn log_path(&self, sequence: u64) -> PathBuf {
        self.dir.join(format!("wal_{:08}.log", sequence))
    }

    fn sequence_from_path(path: &Path) -> Option<u64> {
        path.file_stem()
            .and_then(|s| s.to_str())
            .and_then(|s| s.strip_prefix("wal_"))
            .and_then(|s| s.parse().ok())
    }

    fn truncate_old_logs(&self) -> Result<()> {
        let checkpoint = match *self.checkpoint_epoch.lock() {
            Some(e) => e,
            None => return Ok(()),
        };

        // Keep logs that might still be needed
        // For now, keep the two most recent logs after checkpoint
        let files = self.log_files()?;
        let current_seq = self.current_sequence.load(Ordering::Relaxed);

        for file in files {
            if let Some(seq) = Self::sequence_from_path(&file) {
                // Keep the last 2 log files before current
                if seq + 2 < current_seq {
                    // Only delete if we have a checkpoint after this log
                    if checkpoint.as_u64() > seq {
                        let _ = fs::remove_file(&file);
                    }
                }
            }
        }

        Ok(())
    }
}

// Backward compatibility - single-file API
impl WalManager {
    /// Opens a single WAL file (legacy API).
    ///
    /// # Errors
    ///
    /// Returns an error if the file cannot be opened.
    pub fn open_file(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let dir = path.parent().unwrap_or(Path::new("."));
        let manager = Self::open(dir)?;
        Ok(manager)
    }

    /// Returns the path to the active WAL file.
    #[must_use]
    pub fn path(&self) -> PathBuf {
        let guard = self.active_log.lock();
        guard
            .as_ref()
            .map(|l| l.path.clone())
            .unwrap_or_else(|| self.log_path(0))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use grafeo_common::types::NodeId;
    use tempfile::tempdir;

    #[test]
    fn test_wal_write() {
        let dir = tempdir().unwrap();

        let wal = WalManager::open(dir.path()).unwrap();

        let record = WalRecord::CreateNode {
            id: NodeId::new(1),
            labels: vec!["Person".to_string()],
        };

        wal.log(&record).unwrap();
        wal.flush().unwrap();

        assert_eq!(wal.record_count(), 1);
    }

    #[test]
    fn test_wal_rotation() {
        let dir = tempdir().unwrap();

        // Small max size to force rotation
        let config = WalConfig {
            max_log_size: 100,
            ..Default::default()
        };

        let wal = WalManager::with_config(dir.path(), config).unwrap();

        // Write enough records to trigger rotation
        for i in 0..10 {
            let record = WalRecord::CreateNode {
                id: NodeId::new(i),
                labels: vec!["Person".to_string()],
            };
            wal.log(&record).unwrap();
        }

        wal.flush().unwrap();

        // Should have multiple log files
        let files = wal.log_files().unwrap();
        assert!(
            files.len() > 1,
            "Expected multiple log files after rotation"
        );
    }

    #[test]
    fn test_durability_modes() {
        let dir = tempdir().unwrap();

        // Test Sync mode
        let config = WalConfig {
            durability: DurabilityMode::Sync,
            ..Default::default()
        };
        let wal = WalManager::with_config(dir.path().join("sync"), config).unwrap();
        wal.log(&WalRecord::TxCommit {
            tx_id: TxId::new(1),
        })
        .unwrap();

        // Test NoSync mode
        let config = WalConfig {
            durability: DurabilityMode::NoSync,
            ..Default::default()
        };
        let wal = WalManager::with_config(dir.path().join("nosync"), config).unwrap();
        wal.log(&WalRecord::CreateNode {
            id: NodeId::new(1),
            labels: vec![],
        })
        .unwrap();

        // Test Batch mode
        let config = WalConfig {
            durability: DurabilityMode::Batch {
                max_delay_ms: 10,
                max_records: 5,
            },
            ..Default::default()
        };
        let wal = WalManager::with_config(dir.path().join("batch"), config).unwrap();
        for i in 0..10 {
            wal.log(&WalRecord::CreateNode {
                id: NodeId::new(i),
                labels: vec![],
            })
            .unwrap();
        }

        // Test Adaptive mode (just buffer flush, no inline sync)
        let config = WalConfig {
            durability: DurabilityMode::Adaptive {
                target_interval_ms: 100,
            },
            ..Default::default()
        };
        let wal = WalManager::with_config(dir.path().join("adaptive"), config).unwrap();
        for i in 0..10 {
            wal.log(&WalRecord::CreateNode {
                id: NodeId::new(i),
                labels: vec![],
            })
            .unwrap();
        }
        // Manually sync since no flusher thread in this test
        wal.sync().unwrap();
    }

    #[test]
    fn test_checkpoint() {
        let dir = tempdir().unwrap();

        let wal = WalManager::open(dir.path()).unwrap();

        // Write some records
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

        assert_eq!(wal.checkpoint_epoch(), Some(EpochId::new(10)));
    }
}
