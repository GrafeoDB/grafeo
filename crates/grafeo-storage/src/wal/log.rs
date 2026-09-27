//! WAL log file management.

use super::WalRecord;
use grafeo_common::types::{EpochId, TransactionId};
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
    pub transaction_id: TransactionId,
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
    /// Encryptor for WAL records (None = unencrypted).
    #[cfg(feature = "encryption")]
    encryptor: Option<grafeo_common::encryption::PageEncryptor>,
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
                if let Some(name) = entry.file_name().to_str()
                    && let Some(seq_str) = name
                        .strip_prefix("wal_")
                        .and_then(|s| s.strip_suffix(".log"))
                    && let Ok(seq) = seq_str.parse::<u64>()
                {
                    max_sequence = max_sequence.max(seq);
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
            #[cfg(feature = "encryption")]
            encryptor: None,
        };

        // Open or create the active log
        manager.ensure_active_log()?;

        Ok(manager)
    }

    /// Sets the encryptor for WAL record encryption.
    ///
    /// When set, all written records are encrypted with AES-256-GCM and the
    /// GCM authentication tag replaces the CRC32 checksum. The nonce is derived
    /// from the WAL sequence number.
    #[cfg(feature = "encryption")]
    pub fn set_encryptor(&mut self, encryptor: grafeo_common::encryption::PageEncryptor) {
        self.encryptor = Some(encryptor);
    }

    /// Returns whether encryption is active.
    #[cfg(feature = "encryption")]
    #[must_use]
    pub fn is_encrypted(&self) -> bool {
        self.encryptor.is_some()
    }

    /// Logs a record to the WAL.
    ///
    /// # Errors
    ///
    /// Returns an error if the record cannot be written.
    pub fn log(&self, record: &WalRecord) -> Result<()> {
        let data = bincode::serde::encode_to_vec(record, bincode::config::standard())
            .map_err(|e| Error::Serialization(e.to_string()))?;
        let force_sync = matches!(record, WalRecord::TransactionCommit { .. });
        self.write_frame(&data, force_sync)
    }

    /// Logs records as one contiguous group: no other writer's records can
    /// land between them.
    ///
    /// # Errors
    ///
    /// Returns an error if a record cannot be serialized or written.
    pub fn log_batch(&self, records: &[WalRecord]) -> Result<()> {
        let frames = records
            .iter()
            .map(|record| {
                bincode::serde::encode_to_vec(record, bincode::config::standard())
                    .map_err(|e| Error::Serialization(e.to_string()))
            })
            .collect::<Result<Vec<_>>>()?;
        let frame_refs: Vec<&[u8]> = frames.iter().map(Vec::as_slice).collect();
        let force_sync = records
            .iter()
            .any(|record| matches!(record, WalRecord::TransactionCommit { .. }));
        self.write_frames(&frame_refs, force_sync)
    }

    /// Writes a pre-serialized frame to the active WAL log.
    ///
    /// Frame format: `[length: u32 LE][data: bytes][crc32: u32 LE]`.
    /// Handles durability mode (sync/batch/adaptive/nosync) and log rotation.
    ///
    /// `force_sync` controls whether an fsync is performed in Sync durability
    /// mode. Callers typically set this to `true` for commit markers.
    pub(crate) fn write_frame(&self, data: &[u8], force_sync: bool) -> Result<()> {
        self.write_frames(&[data], force_sync)
    }

    /// Writes pre-serialized frames as one contiguous group.
    ///
    /// All frames are written while holding the active-log lock, so frames
    /// from other writers cannot interleave with them, and rotation only
    /// happens after the whole group. Durability handling runs once for the
    /// group, as for a single frame.
    pub(crate) fn write_frames(&self, frames: &[&[u8]], force_sync: bool) -> Result<()> {
        use grafeo_common::testing::crash::maybe_crash;

        self.ensure_active_log()?;

        // Phase 1: write frame data and flush buffer while holding the lock.
        // Determine whether an fsync is needed, and if so clone the file handle
        // so we can release the lock before the (potentially slow) sync_all().
        let (needs_rotation, sync_file, synced_records) = {
            let mut guard = self.active_log.lock();
            let log_file = guard
                .as_mut()
                .ok_or_else(|| Error::Internal("WAL writer not available".to_string()))?;

            for &data in frames {
                maybe_crash("wal_before_write");

                // Encrypt or write plaintext depending on encryption configuration.
                // Encrypted frame: [len:4][nonce(12) || ciphertext || tag(16)]
                // Plaintext frame:  [len:4][data][crc32:4]
                #[cfg(feature = "encryption")]
                let (frame_data, record_size) = if let Some(ref enc) = self.encryptor {
                    let file_seq = self.current_sequence.load(Ordering::Relaxed);
                    // Use the file byte offset as the nonce counter, not the ephemeral
                    // record count. The byte offset survives restarts (file is append-only)
                    // and is unique per record within a file. Combined with the file sequence,
                    // this guarantees nonce uniqueness even after crash + restart.
                    //
                    // The nonce high word is 4 bytes, so the file sequence must fit in u32.
                    // With one rotation per ~64 MB of WAL, this allows ~256 exabytes of
                    // total WAL writes before exhaustion, which is effectively unlimited.
                    let seq_u32 = u32::try_from(file_seq).map_err(|_| {
                        Error::Internal(
                            "WAL file sequence exceeds u32::MAX: encryption nonce space exhausted"
                                .to_string(),
                        )
                    })?;
                    let byte_offset = log_file.size;
                    let nonce = grafeo_common::encryption::build_nonce(seq_u32, byte_offset);
                    let aad = b"grafeo-wal";
                    let encrypted = enc
                        .encrypt(data, &nonce, aad)
                        .map_err(|e| Error::Internal(format!("WAL encryption failed: {e}")))?;
                    // reason: WAL wire format uses u32 length prefix; individual records are well under 4 GiB
                    #[allow(clippy::cast_possible_truncation)]
                    let len = encrypted.len() as u32;
                    log_file.writer.write_all(&len.to_le_bytes())?;
                    log_file.writer.write_all(&encrypted)?;
                    let size = 4 + encrypted.len() as u64;
                    (true, size)
                } else {
                    (false, 0u64)
                };

                #[cfg(feature = "encryption")]
                if !frame_data {
                    // reason: WAL wire format uses u32 length prefix; individual records are well under 4 GiB
                    #[allow(clippy::cast_possible_truncation)]
                    let len = data.len() as u32;
                    log_file.writer.write_all(&len.to_le_bytes())?;
                    log_file.writer.write_all(data)?;
                    let checksum = crc32fast::hash(data);
                    log_file.writer.write_all(&checksum.to_le_bytes())?;
                }

                #[cfg(not(feature = "encryption"))]
                {
                    // Write length prefix
                    // reason: WAL wire format uses u32 length prefix; individual records are well under 4 GiB
                    #[allow(clippy::cast_possible_truncation)]
                    let len = data.len() as u32;
                    log_file.writer.write_all(&len.to_le_bytes())?;

                    // Write data
                    log_file.writer.write_all(data)?;

                    // Write checksum
                    let checksum = crc32fast::hash(data);
                    log_file.writer.write_all(&checksum.to_le_bytes())?;
                }

                maybe_crash("wal_after_write");

                // Update size tracking
                #[cfg(feature = "encryption")]
                let record_size = if frame_data {
                    record_size
                } else {
                    4 + data.len() as u64 + 4
                };
                #[cfg(not(feature = "encryption"))]
                let record_size = 4 + data.len() as u64 + 4; // length + data + checksum
                log_file.size += record_size;

                self.total_record_count.fetch_add(1, Ordering::Relaxed);
                self.records_since_sync.fetch_add(1, Ordering::Relaxed);
            }

            let needs_rotation = log_file.size >= self.config.max_log_size;

            // Decide whether we need to fsync based on durability mode.
            // Always flush the BufWriter so data reaches the OS page cache.
            let needs_sync = match &self.config.durability {
                DurabilityMode::Sync => {
                    if force_sync {
                        maybe_crash("wal_before_flush");
                    }
                    force_sync
                }
                DurabilityMode::Batch {
                    max_delay_ms,
                    max_records,
                } => {
                    let records = self.records_since_sync.load(Ordering::Relaxed);
                    let elapsed = self.last_sync.lock().elapsed();
                    records >= *max_records || elapsed >= Duration::from_millis(*max_delay_ms)
                }
                DurabilityMode::Adaptive { .. } | DurabilityMode::NoSync => false,
            };

            // Flush the BufWriter while holding the lock (pushes data to OS).
            log_file.writer.flush()?;

            // Snapshot the record count while holding the lock so we can
            // subtract exactly this amount after sync, preserving any
            // concurrent increments that arrive between lock release and sync.
            let synced_records = if needs_sync {
                self.records_since_sync.load(Ordering::Relaxed)
            } else {
                0
            };

            // Clone the file handle for out-of-lock sync if needed.
            let sync_file = if needs_sync {
                Some(log_file.writer.get_ref().try_clone()?)
            } else {
                None
            };

            (needs_rotation, sync_file, synced_records)
            // guard dropped here: active_log lock released
        };

        // Phase 2: fsync outside the lock so other threads can write concurrently.
        if let Some(file) = sync_file {
            file.sync_all()?;
            self.records_since_sync
                .fetch_sub(synced_records, Ordering::Relaxed);
            *self.last_sync.lock() = Instant::now();
        }

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
    pub fn checkpoint(&self, current_transaction: TransactionId, epoch: EpochId) -> Result<()> {
        self.log(&WalRecord::Checkpoint {
            transaction_id: current_transaction,
        })?;
        self.complete_checkpoint(current_transaction, epoch)
    }

    /// Completes a checkpoint after the checkpoint record has been written.
    ///
    /// Syncs the WAL, writes checkpoint metadata atomically, updates the
    /// in-memory epoch, and truncates old log files.
    pub(crate) fn complete_checkpoint(
        &self,
        transaction_id: TransactionId,
        epoch: EpochId,
    ) -> Result<()> {
        // Ordering guarantee: fsync all WAL data before writing checkpoint
        // metadata. This ensures that on recovery, any WAL entries referenced
        // by the checkpoint metadata are durable on disk. Without this barrier,
        // a crash between metadata write and WAL sync could cause recovery to
        // skip replaying un-synced WAL records.
        self.sync()?;

        // Get current log sequence
        let log_sequence = self.current_sequence.load(Ordering::SeqCst);

        // Get current timestamp
        let timestamp_ms = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            // reason: millis since UNIX epoch fits in u64 for ~585 million years
            .map_or(0, |d| {
                // reason: value is bounded by format constraints
                #[allow(clippy::cast_possible_truncation)]
                let ms = d.as_millis() as u64;
                ms
            });

        // Create checkpoint metadata
        let metadata = CheckpointMetadata {
            epoch,
            log_sequence,
            timestamp_ms,
            transaction_id,
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
        };

        // Replace active log, syncing the old one to ensure durability
        let mut guard = self.active_log.lock();
        if let Some(mut old_log) = guard.take() {
            old_log.writer.flush()?;
            old_log.writer.get_ref().sync_all()?;
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
        // Flush buffer and clone handle while holding the lock, then sync outside.
        let sync_file = {
            let mut guard = self.active_log.lock();
            if let Some(log_file) = guard.as_mut() {
                log_file.writer.flush()?;
                Some(log_file.writer.get_ref().try_clone()?)
            } else {
                None
            }
        };
        if let Some(file) = sync_file {
            file.sync_all()?;
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

    /// Returns the current WAL log sequence number.
    ///
    /// Each log file has a sequence number embedded in its name
    /// (`wal_XXXXXXXX.log`). This returns the sequence of the active log file.
    #[must_use]
    pub fn current_sequence(&self) -> u64 {
        self.current_sequence.load(Ordering::Relaxed)
    }

    /// Returns the current durability mode.
    #[must_use]
    pub fn durability_mode(&self) -> DurabilityMode {
        self.config.durability
    }

    /// Returns all WAL log file paths in sequence order.
    ///
    /// # Errors
    ///
    /// Returns an error if the WAL directory cannot be read.
    pub fn log_files(&self) -> Result<Vec<PathBuf>> {
        let mut files = Vec::new();

        for entry in fs::read_dir(&self.dir)?.flatten() {
            let path = entry.path();
            if path.extension().is_some_and(|ext| ext == "log") {
                files.push(path);
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
                    // reason: WAL files are capped at max_log_size (default 64 MiB), fits in usize on all targets
                    #[allow(clippy::cast_possible_truncation)]
                    let file_len = metadata.len() as usize;
                    total += file_len;
                }
            }
        }
        // Also include checkpoint metadata file
        let metadata_path = self.dir.join(CHECKPOINT_METADATA_FILE);
        if let Ok(metadata) = fs::metadata(&metadata_path) {
            // reason: checkpoint metadata file is a small fixed-size struct, fits in usize
            #[allow(clippy::cast_possible_truncation)]
            let meta_len = metadata.len() as usize;
            total += meta_len;
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

    /// Closes the active log file, releasing its file handle.
    ///
    /// This allows the WAL directory to be safely removed on Windows,
    /// where open file handles prevent directory deletion. A new log file
    /// will be created automatically on the next write.
    pub fn close_active_log(&self) {
        let mut guard = self.active_log.lock();
        // Dropping the LogFile closes the BufWriter and underlying File
        *guard = None;
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
        let Some(checkpoint) = *self.checkpoint_epoch.lock() else {
            return Ok(());
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
            .map_or_else(|| self.log_path(0), |l| l.path.clone())
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
        wal.log(&WalRecord::TransactionCommit {
            transaction_id: TransactionId::new(1),
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

        wal.log(&WalRecord::TransactionCommit {
            transaction_id: TransactionId::new(1),
        })
        .unwrap();

        // Create checkpoint
        wal.checkpoint(TransactionId::new(1), EpochId::new(10))
            .unwrap();

        assert_eq!(wal.checkpoint_epoch(), Some(EpochId::new(10)));
    }
}
