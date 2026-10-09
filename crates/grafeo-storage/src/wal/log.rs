//! WAL log file management.

use super::{GroupError, WalEntry, WalError, WalRecord};
use grafeo_common::types::{EpochId, TransactionId};
use grafeo_common::utils::error::{Error, Result};
use parking_lot::Mutex;
use serde::{Deserialize, Serialize};
use std::fs::{self, File, OpenOptions};
use std::io::{BufReader, BufWriter, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
#[cfg(feature = "encryption")]
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
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
    /// An uncertain group or failed tail repair prevents further appends.
    unavailable: AtomicBool,
    /// The first failure, retained for subsequent rejected writes.
    unavailable_reason: Mutex<Option<String>>,
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
    /// Encryptor for WAL records (None = unencrypted), shared with the
    /// recoveries made from this manager.
    #[cfg(feature = "encryption")]
    encryptor: Option<Arc<grafeo_common::encryption::PageEncryptor>>,
}

/// Encrypts and decrypts WAL records: see
/// [`WalManager::with_config_and_cipher`] and
/// [`WalRecovery::with_cipher`](super::WalRecovery::with_cipher).
#[cfg(feature = "encryption")]
pub type WalCipher = grafeo_common::encryption::PageEncryptor;

/// Placeholder without the `encryption` feature: it has no values, so a
/// cipher can never be supplied.
#[cfg(not(feature = "encryption"))]
#[derive(Debug)]
pub enum WalCipher {}

/// The length prefix of a frame whose payload is `length` bytes.
fn frame_length(length: usize) -> Result<u32> {
    u32::try_from(length).map_err(|_| {
        Error::Internal(format!(
            "a WAL record of {length} bytes does not fit a frame, whose length prefix \
             allows at most {} bytes",
            u32::MAX
        ))
    })
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
        Self::with_config_and_cipher(dir, config, None)
    }

    /// Opens or creates a WAL with custom configuration that encrypts every
    /// record it writes with `cipher` (see [`set_encryptor`](Self::set_encryptor)
    /// for the frame format); `None` writes plaintext.
    ///
    /// The cipher is in place before the manager exists, so no record can be
    /// written without it.
    ///
    /// # Errors
    ///
    /// Returns an error if the directory cannot be created or accessed.
    pub fn with_config_and_cipher(
        dir: impl AsRef<Path>,
        config: WalConfig,
        cipher: Option<WalCipher>,
    ) -> Result<Self> {
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
            unavailable: AtomicBool::new(false),
            unavailable_reason: Mutex::new(None),
            total_record_count: AtomicU64::new(0),
            records_since_sync: AtomicU64::new(0),
            last_sync: Mutex::new(Instant::now()),
            current_sequence: AtomicU64::new(max_sequence),
            checkpoint_epoch: Mutex::new(None),
            #[cfg(feature = "encryption")]
            encryptor: cipher.map(Arc::new),
        };
        #[cfg(not(feature = "encryption"))]
        if let Some(cipher) = cipher {
            match cipher {}
        }

        // Open or create the active log
        manager.ensure_active_log()?;

        Ok(manager)
    }

    /// Sets the encryptor for WAL record encryption.
    ///
    /// When set, all written records are encrypted with AES-256-GCM and the
    /// GCM authentication tag replaces the CRC32 checksum. Each record gets a
    /// random nonce, stored in front of its ciphertext, so a log that starts
    /// over under the same key never repeats one.
    #[cfg(feature = "encryption")]
    pub fn set_encryptor(&mut self, encryptor: grafeo_common::encryption::PageEncryptor) {
        self.encryptor = Some(Arc::new(encryptor));
    }

    /// Returns whether encryption is active.
    #[cfg(feature = "encryption")]
    #[must_use]
    pub fn is_encrypted(&self) -> bool {
        self.encryptor.is_some()
    }

    /// The encryptor of this WAL, for a recovery that reads it.
    #[cfg(feature = "encryption")]
    pub(crate) fn encryptor(&self) -> Option<&Arc<grafeo_common::encryption::PageEncryptor>> {
        self.encryptor.as_ref()
    }

    /// The encrypted payload of a frame (`nonce || ciphertext || tag`), or
    /// `None` for a WAL without an encryptor.
    ///
    /// Each record gets a random nonce: the key of a database's WAL stays the
    /// same for its whole life, while the log starts over at file 0, offset 0
    /// whenever a clean close removes the sidecar WAL, so a nonce built from
    /// the file sequence and offset would repeat under that key.
    #[cfg(feature = "encryption")]
    fn encrypt(&self, data: &[u8]) -> Result<Option<Vec<u8>>> {
        let Some(encryptor) = &self.encryptor else {
            return Ok(None);
        };
        let nonce = grafeo_common::encryption::random_nonce();
        encryptor
            .encrypt(data, &nonce, b"grafeo-wal")
            .map(Some)
            .map_err(|e| Error::Internal(format!("WAL encryption failed: {e}")))
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
        self.write_frame(&data, force_sync, record.is_commit())
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
        let commit_frame = records.iter().position(WalEntry::is_commit);
        self.write_frames(&frame_refs, force_sync, commit_frame)
    }

    /// Writes a pre-serialized frame to the active WAL log.
    ///
    /// Frame format: `[length: u32 LE][data: bytes][crc32: u32 LE]`.
    /// Handles durability mode (sync/batch/adaptive/nosync) and log rotation.
    ///
    /// `force_sync` controls whether an fsync is performed in Sync durability
    /// mode. Callers typically set this to `true` for commit markers.
    pub(crate) fn write_frame(&self, data: &[u8], force_sync: bool, is_commit: bool) -> Result<()> {
        self.write_frames(&[data], force_sync, is_commit.then_some(0))
    }

    /// Writes pre-serialized frames as one contiguous group.
    ///
    /// All frames are written while holding the active-log lock, so frames
    /// from other writers cannot interleave with them, and rotation only
    /// happens after the whole group. Durability handling runs once for the
    /// group, as for a single frame.
    pub(crate) fn write_frames(
        &self,
        frames: &[&[u8]],
        force_sync: bool,
        commit_frame: Option<usize>,
    ) -> Result<()> {
        use grafeo_common::testing::crash::{maybe_crash, maybe_fail};

        self.check_available()?;
        // Keep these two frontend failure calls stable: before any group
        // bytes, and after the group has actually been synced.
        if let Err(error) = maybe_fail("wal_before_group_write") {
            return Err(Self::not_written(self.path(), error));
        }

        // Validate every length before any frame of this group is written.
        #[cfg(feature = "encryption")]
        let overhead = self
            .encryptor
            .as_ref()
            .map_or(0, |_| grafeo_common::encryption::ENCRYPTION_OVERHEAD);
        #[cfg(not(feature = "encryption"))]
        let overhead = 0;
        for data in frames {
            frame_length(data.len().saturating_add(overhead))?;
        }
        if let Err(error) = self.ensure_active_log() {
            // An existing fence must not be turned into an abortable error.
            self.check_available()?;
            return Err(Self::not_written(self.path(), error));
        }

        let mut guard = self.active_log.lock();
        self.check_available()?;
        let log_file = guard
            .as_mut()
            .ok_or_else(|| Error::Internal("WAL writer not available".to_string()))?;
        let start_size = log_file.size;
        let path = log_file.path.clone();
        let mut bytes_attempted = false;
        let mut marker_attempted = false;
        let mut added_records = 0;

        // Each successful preceding group flushed its buffer, so size is
        // the exact file boundary to restore under this same lock.
        let written: Result<bool> = (|| {
            for (frame_index, &data) in frames.iter().enumerate() {
                maybe_crash("wal_before_write");
                #[cfg(feature = "encryption")]
                let encrypted = self.encrypt(data)?;
                #[cfg(not(feature = "encryption"))]
                let encrypted: Option<Vec<u8>> = None;

                // A failed write may have accepted the marker's bytes even
                // when it returned Err. Enter uncertainty before trying it.
                marker_attempted |= commit_frame == Some(frame_index);
                #[cfg(all(test, feature = "testing-crash-injection"))]
                legacy_commit_tests::maybe_fail(
                    "wal_group_frame_write",
                    frame_index,
                    log_file.writer.get_ref(),
                )?;
                bytes_attempted = true;
                let record_size = match encrypted {
                    Some(encrypted) => {
                        let length = frame_length(encrypted.len())?;
                        log_file.writer.write_all(&length.to_le_bytes())?;
                        log_file.writer.write_all(&encrypted)?;
                        #[cfg(all(test, feature = "testing-crash-injection"))]
                        legacy_commit_tests::maybe_fail(
                            "wal_group_payload_write",
                            frame_index,
                            log_file.writer.get_ref(),
                        )?;
                        4 + u64::from(length)
                    }
                    None => {
                        let length = frame_length(data.len())?;
                        log_file.writer.write_all(&length.to_le_bytes())?;
                        log_file.writer.write_all(data)?;
                        #[cfg(all(test, feature = "testing-crash-injection"))]
                        legacy_commit_tests::maybe_fail(
                            "wal_group_payload_write",
                            frame_index,
                            log_file.writer.get_ref(),
                        )?;
                        log_file
                            .writer
                            .write_all(&crc32fast::hash(data).to_le_bytes())?;
                        4 + u64::from(length) + 4
                    }
                };
                maybe_crash("wal_after_write");
                log_file.size += record_size;
                self.total_record_count.fetch_add(1, Ordering::Relaxed);
                self.records_since_sync.fetch_add(1, Ordering::Relaxed);
                added_records += 1;
            }

            let needs_rotation = log_file.size >= self.config.max_log_size;
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
                    self.records_since_sync.load(Ordering::Relaxed) >= *max_records
                        || self.last_sync.lock().elapsed() >= Duration::from_millis(*max_delay_ms)
                }
                DurabilityMode::Adaptive { .. } | DurabilityMode::NoSync => false,
            };
            log_file.writer.flush()?;
            if needs_sync {
                let synced_records = self.records_since_sync.load(Ordering::Relaxed);
                // Hold writer serialization through sync and acknowledgement:
                // a failure must fence the log before another group starts.
                log_file.writer.get_ref().sync_all()?;
                self.subtract_pending_records(synced_records);
                *self.last_sync.lock() = Instant::now();
                maybe_fail("wal_after_group_sync")?;
            }
            Ok(needs_rotation)
        })();
        let written = written.and_then(|needs_rotation| {
            if needs_rotation {
                #[cfg(all(test, feature = "testing-crash-injection"))]
                if let Some(log_file) = guard.as_ref() {
                    legacy_commit_tests::maybe_fail(
                        "wal_group_rotate",
                        0,
                        log_file.writer.get_ref(),
                    )?;
                }
                self.rotate_under_lock(&mut guard)?;
            }
            Ok(())
        });
        match written {
            Ok(()) => Ok(()),
            Err(error) if marker_attempted => Err(self.unknown(path, error)),
            Err(error) => {
                if bytes_attempted
                    && let Err(repair) = self.restore_group(&mut guard, start_size, added_records)
                {
                    return Err(self.make_unavailable(format!(
                        "{error}; the pre-marker WAL tail could not be restored: {repair}"
                    )));
                }
                Err(Self::not_written(path, error))
            }
        }
    }

    /// Restores a proven pre-marker tail without flushing rejected bytes.
    /// The caller holds active_log throughout removal and replacement.
    fn restore_group(
        &self,
        active: &mut Option<LogFile>,
        start_size: u64,
        added_records: u64,
    ) -> Result<()> {
        let log_file = active
            .take()
            .ok_or_else(|| Error::Internal("WAL writer not available for repair".to_string()))?;
        let (mut file, pending) = log_file.writer.into_parts();
        // into_parts, unlike into_inner or Drop, never tries to flush.
        let pending = pending
            .map_err(|_| Error::Internal("WAL buffer panicked and cannot be reused".to_string()))?;
        drop(pending);
        #[cfg(all(test, feature = "testing-crash-injection"))]
        legacy_commit_tests::maybe_fail("wal_group_repair_truncate", 0, &file)?;
        file.set_len(start_size)?;
        file.seek(SeekFrom::Start(start_size))?;
        #[cfg(all(test, feature = "testing-crash-injection"))]
        legacy_commit_tests::maybe_fail("wal_group_repair_sync", 0, &file)?;
        file.sync_all()?;
        *active = Some(LogFile {
            writer: BufWriter::new(file),
            size: start_size,
            path: log_file.path,
        });
        self.total_record_count
            .fetch_sub(added_records, Ordering::Relaxed);
        self.subtract_pending_records(added_records);
        Ok(())
    }

    fn subtract_pending_records(&self, records: u64) {
        // A background sync may have cleared this count in the meantime.
        self.records_since_sync
            .update(Ordering::Relaxed, Ordering::Relaxed, |pending| {
                pending.saturating_sub(records)
            });
    }

    fn not_written(path: PathBuf, error: Error) -> Error {
        match error {
            Error::Io(source) => GroupError::NotWritten(WalError::io(path, source)).into(),
            // Preserve validation/serialization classes before any marker.
            other => other,
        }
    }

    fn unknown(&self, path: PathBuf, error: Error) -> Error {
        self.poison(error.to_string());
        GroupError::OutcomeUnknown(WalError::injected(path, error)).into()
    }

    fn poison(&self, reason: String) -> String {
        let mut first = self.unavailable_reason.lock();
        let first = first.get_or_insert(reason).clone();
        self.unavailable.store(true, Ordering::Release);
        first
    }

    fn make_unavailable(&self, reason: String) -> Error {
        GroupError::Unavailable {
            reason: self.poison(reason),
        }
        .into()
    }

    fn check_available(&self) -> Result<()> {
        if self.unavailable.load(Ordering::Acquire) {
            return Err(GroupError::Unavailable {
                reason: self
                    .unavailable_reason
                    .lock()
                    .clone()
                    .unwrap_or_else(|| "an earlier WAL group did not complete".to_string()),
            }
            .into());
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

    /// Records that a checkpoint image holds everything logged in files with a
    /// sequence below `log_sequence`, so recovery starts at that file.
    ///
    /// Call this only after the image is durable. The metadata is written
    /// atomically (temporary file and rename). Unlike
    /// [`checkpoint`](Self::checkpoint), it logs no record and deletes no file;
    /// use [`remove_files_before`](Self::remove_files_before) for that.
    ///
    /// # Errors
    ///
    /// Returns an error if the metadata cannot be written.
    pub fn mark_checkpoint(
        &self,
        log_sequence: u64,
        epoch: EpochId,
        transaction_id: TransactionId,
    ) -> Result<()> {
        let timestamp_ms = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            // reason: millis since UNIX epoch fits in u64 for ~585 million years
            .map_or(0, |d| {
                #[allow(clippy::cast_possible_truncation)]
                let ms = d.as_millis() as u64;
                ms
            });
        self.write_checkpoint_metadata(&CheckpointMetadata {
            epoch,
            log_sequence,
            timestamp_ms,
            transaction_id,
        })?;
        *self.checkpoint_epoch.lock() = Some(epoch);
        Ok(())
    }

    /// Deletes the log files with a sequence below `sequence`, never the
    /// active file. Returns how many files were deleted.
    ///
    /// # Errors
    ///
    /// Returns an error if the directory cannot be listed or a file cannot be
    /// deleted.
    pub fn remove_files_before(&self, sequence: u64) -> Result<usize> {
        let active = self.current_sequence.load(Ordering::SeqCst);
        let mut removed = 0;
        for file in self.log_files()? {
            if let Some(seq) = Self::sequence_from_path(&file)
                && seq < sequence
                && seq != active
            {
                fs::remove_file(&file)?;
                removed += 1;
            }
        }
        Ok(removed)
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
        let mut active = self.active_log.lock();
        self.check_available()?;
        self.rotate_under_lock(&mut active)
            .map_err(|error| self.make_unavailable(error.to_string()))
    }

    /// The caller holds active_log until a rotation failure is classified.
    fn rotate_under_lock(&self, active: &mut Option<LogFile>) -> Result<()> {
        self.check_available()?;
        if let Some(old_log) = active.as_mut() {
            old_log.writer.flush()?;
            old_log.writer.get_ref().sync_all()?;
        }
        let new_sequence = self.current_sequence.load(Ordering::SeqCst) + 1;
        let new_path = self.log_path(new_sequence);
        let file = OpenOptions::new()
            .create(true)
            .read(true)
            .append(true)
            .open(&new_path)?;
        let size = file.metadata()?.len();
        // Keep the old handle and sequence intact if opening the new file
        // fails; a pre-marker group can still restore its original tail.
        *active = Some(LogFile {
            writer: BufWriter::new(file),
            size,
            path: new_path,
        });
        self.current_sequence.store(new_sequence, Ordering::SeqCst);
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
        self.check_available()?;
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

    /// A log that starts over under the same key, as the sidecar WAL of an
    /// encrypted database does after a clean close removed it, never repeats
    /// a nonce: the same record at the same position of the first log file
    /// gets a different nonce the second time.
    #[cfg(all(feature = "encryption", not(miri)))]
    #[test]
    fn a_log_started_over_under_one_key_never_repeats_a_nonce() {
        use grafeo_common::encryption::{KeyChain, NONCE_SIZE};

        let dir = tempdir().unwrap();
        let wal_dir = dir.path().join("amsterdam.grafeo.wal");
        let chain = KeyChain::new([3; 32]);
        let record = WalRecord::CreateNode {
            id: NodeId::new(19),
            labels: vec!["Person".to_string()],
        };
        let mut nonces = Vec::new();
        for _ in 0..2 {
            {
                let mut wal = WalManager::open(&wal_dir).unwrap();
                wal.set_encryptor(chain.encryptor_for("grafeo-wal", &88u128.to_le_bytes()));
                wal.log(&record).unwrap();
                wal.flush().unwrap();
            }
            let log = fs::read(wal_dir.join("wal_00000000.log")).unwrap();
            // Frame: length (4 bytes), then the nonce.
            nonces.push(log[4..4 + NONCE_SIZE].to_vec());
            fs::remove_dir_all(&wal_dir).unwrap();
        }
        assert_ne!(
            nonces[0], nonces[1],
            "the log started over reused the nonce of its first record"
        );
    }
}

#[cfg(all(test, feature = "testing-crash-injection"))]
mod legacy_commit_tests {
    use super::*;
    use crate::wal::{GroupError, LpgWal, WalRecovery};
    use grafeo_common::testing::crash::with_failure_at;
    use grafeo_common::types::NodeId;
    use std::cell::RefCell;
    use std::collections::VecDeque;
    use tempfile::tempdir;

    type Fault = (&'static str, usize);
    type Hit = (&'static str, u64);

    thread_local! {
        static FAULTS: RefCell<VecDeque<Fault>> = const { RefCell::new(VecDeque::new()) };
        static HITS: RefCell<Vec<Hit>> = const { RefCell::new(Vec::new()) };
    }

    /// Unit-only named faults do not consume the two frontend failure calls.
    pub(super) fn maybe_fail(point: &'static str, frame: usize, file: &File) -> Result<()> {
        let selected = FAULTS.with(|faults| {
            let mut faults = faults.borrow_mut();
            if faults.front() == Some(&(point, frame)) {
                faults.pop_front();
                true
            } else {
                false
            }
        });
        if !selected {
            return Ok(());
        }
        HITS.with(|hits| {
            hits.borrow_mut()
                .push((point, file.metadata().unwrap().len()));
        });
        with_failure_at(1, || grafeo_common::testing::crash::maybe_fail(point))
    }

    fn with_faults<T>(faults: &[Fault], body: impl FnOnce() -> T) -> (T, Vec<Hit>) {
        struct Reset;
        impl Drop for Reset {
            fn drop(&mut self) {
                FAULTS.with(|faults| faults.borrow_mut().clear());
                HITS.with(|hits| hits.borrow_mut().clear());
            }
        }
        FAULTS.with(|pending| pending.borrow_mut().extend(faults.iter().copied()));
        let _reset = Reset;
        let result = body();
        let hits = HITS.with(|hits| std::mem::take(&mut *hits.borrow_mut()));
        (result, hits)
    }

    fn open_sync(path: &Path) -> LpgWal {
        LpgWal::with_config(
            path,
            WalConfig {
                durability: DurabilityMode::Sync,
                ..WalConfig::default()
            },
        )
        .unwrap()
    }

    fn records(id: u64, large: bool) -> Vec<WalRecord> {
        vec![
            WalRecord::CreateNode {
                id: NodeId::new(id),
                labels: vec![if large {
                    "x".repeat(20_000)
                } else {
                    "Probe".to_owned()
                }],
            },
            WalRecord::TransactionCommit {
                transaction_id: TransactionId::new(id + 10),
            },
            WalRecord::EpochAdvance {
                epoch: EpochId::new(id + 1),
            },
        ]
    }

    fn disposition(error: &Error) -> &GroupError {
        let Error::Io(source) = error else {
            panic!("expected typed WAL I/O error, got {error:?}");
        };
        source
            .get_ref()
            .and_then(|source| source.downcast_ref::<GroupError>())
            .unwrap_or_else(|| panic!("WAL error lost its group disposition: {error:?}"))
    }

    fn recovered_ids(path: &Path) -> Vec<u64> {
        WalRecovery::new(path)
            .recover()
            .unwrap()
            .into_iter()
            .filter_map(|record| match record {
                WalRecord::CreateNode { id, .. } => Some(id.as_u64()),
                _ => None,
            })
            .collect()
    }

    #[test]
    fn legacy_commit_no_bytes_failure_is_abortable_and_reusable() {
        let dir = tempdir().unwrap();
        let wal = open_sync(dir.path());
        wal.log_batch(&records(1, false)).unwrap();
        let before = fs::read(wal.path()).unwrap();
        let error = with_failure_at(1, || wal.log_batch(&records(2, false))).unwrap_err();
        assert!(
            error.to_string().contains("wal_before_group_write"),
            "{error}"
        );
        assert!(matches!(disposition(&error), GroupError::NotWritten(_)));
        assert_eq!(fs::read(wal.path()).unwrap(), before);
        wal.log_batch(&records(3, false)).unwrap();
        drop(wal);
        assert_eq!(recovered_ids(dir.path()), vec![1, 3]);
    }

    #[test]
    fn legacy_commit_partial_payload_is_removed_before_next_marker() {
        let dir = tempdir().unwrap();
        let wal = open_sync(dir.path());
        wal.log_batch(&records(1, false)).unwrap();
        let before = fs::read(wal.path()).unwrap();
        let count = wal.record_count();
        let (result, hits) = with_faults(&[("wal_group_payload_write", 0)], || {
            wal.log_batch(&records(2, true))
        });
        let error = result.unwrap_err();
        assert_eq!(hits.len(), 1);
        assert!(
            hits[0].1 > before.len() as u64,
            "no partial bytes reached the file"
        );
        assert!(
            error.to_string().contains("wal_group_payload_write"),
            "{error}"
        );
        assert!(matches!(disposition(&error), GroupError::NotWritten(_)));
        assert_eq!(fs::read(wal.path()).unwrap(), before);
        assert_eq!(wal.record_count(), count);
        wal.log_batch(&records(3, false)).unwrap();
        drop(wal);
        assert_eq!(recovered_ids(dir.path()), vec![1, 3]);
    }

    #[test]
    fn legacy_commit_failed_tail_repair_makes_writer_unavailable() {
        for point in ["wal_group_repair_truncate", "wal_group_repair_sync"] {
            let dir = tempdir().unwrap();
            let wal = open_sync(dir.path());
            wal.log_batch(&records(1, false)).unwrap();
            let (result, hits) = with_faults(&[("wal_group_payload_write", 0), (point, 0)], || {
                wal.log_batch(&records(2, true))
            });
            let error = result.unwrap_err();
            assert_eq!(
                hits.iter().map(|hit| hit.0).collect::<Vec<_>>(),
                vec!["wal_group_payload_write", point]
            );
            assert!(error.to_string().contains(point), "{error}");
            assert!(matches!(
                disposition(&error),
                GroupError::Unavailable { .. }
            ));
            assert!(matches!(
                disposition(&wal.log_batch(&records(3, false)).unwrap_err()),
                GroupError::Unavailable { .. }
            ));
            assert!(matches!(
                disposition(&wal.manager().rotate().unwrap_err()),
                GroupError::Unavailable { .. }
            ));
            wal.close_active_log();
            assert!(matches!(
                disposition(&wal.log_batch(&records(4, false)).unwrap_err()),
                GroupError::Unavailable { .. }
            ));
            drop(wal);
            assert_eq!(recovered_ids(dir.path()), vec![1]);
        }
    }

    #[test]
    fn legacy_commit_marker_attempt_fences_without_repair() {
        let dir = tempdir().unwrap();
        let wal = open_sync(dir.path());
        wal.log_batch(&records(1, false)).unwrap();
        let count = wal.record_count();
        let (result, hits) = with_faults(&[("wal_group_frame_write", 1)], || {
            wal.log_batch(&records(2, false))
        });
        let error = result.unwrap_err();
        assert_eq!(hits.len(), 1);
        assert!(
            error.to_string().contains("wal_group_frame_write"),
            "{error}"
        );
        assert!(matches!(disposition(&error), GroupError::OutcomeUnknown(_)));
        assert_eq!(
            wal.record_count(),
            count + 1,
            "the data frame was not repaired away"
        );
        assert!(matches!(
            disposition(&wal.log_batch(&records(3, false)).unwrap_err()),
            GroupError::Unavailable { .. }
        ));
        drop(wal);
        assert_eq!(recovered_ids(dir.path()), vec![1]);
    }

    #[test]
    fn legacy_commit_epoch_frame_failure_is_outcome_unknown() {
        let dir = tempdir().unwrap();
        let wal = open_sync(dir.path());
        wal.log_batch(&records(1, false)).unwrap();
        let (result, hits) = with_faults(&[("wal_group_frame_write", 2)], || {
            wal.log_batch(&records(2, false))
        });
        let error = result.unwrap_err();
        assert_eq!(hits.len(), 1);
        assert!(matches!(disposition(&error), GroupError::OutcomeUnknown(_)));
        assert!(matches!(
            disposition(&wal.log_batch(&records(3, false)).unwrap_err()),
            GroupError::Unavailable { .. }
        ));
        drop(wal);
        assert_eq!(recovered_ids(dir.path()), vec![1, 2]);
    }

    #[test]
    fn legacy_commit_lost_synced_acknowledgement_is_replayed() {
        let dir = tempdir().unwrap();
        let wal = open_sync(dir.path());
        wal.log_batch(&records(1, false)).unwrap();
        let error = with_failure_at(2, || wal.log_batch(&records(2, false))).unwrap_err();
        assert!(
            error.to_string().contains("wal_after_group_sync"),
            "{error}"
        );
        assert!(matches!(disposition(&error), GroupError::OutcomeUnknown(_)));
        let before = fs::read(wal.path()).unwrap();
        assert!(matches!(
            disposition(&wal.log_batch(&records(3, false)).unwrap_err()),
            GroupError::Unavailable { .. }
        ));
        assert_eq!(fs::read(wal.path()).unwrap(), before);
        drop(wal);
        assert_eq!(recovered_ids(dir.path()), vec![1, 2]);
    }

    #[test]
    fn legacy_commit_rotation_failure_preserves_synced_group() {
        let dir = tempdir().unwrap();
        let wal = LpgWal::with_config(
            dir.path(),
            WalConfig {
                durability: DurabilityMode::Sync,
                max_log_size: 1,
                ..WalConfig::default()
            },
        )
        .unwrap();
        let (result, hits) = with_faults(&[("wal_group_rotate", 0)], || {
            wal.log_batch(&records(1, false))
        });
        let error = result.unwrap_err();
        assert_eq!(hits.len(), 1);
        assert!(error.to_string().contains("wal_group_rotate"), "{error}");
        assert!(matches!(disposition(&error), GroupError::OutcomeUnknown(_)));
        assert!(matches!(
            disposition(&wal.log_batch(&records(2, false)).unwrap_err()),
            GroupError::Unavailable { .. }
        ));
        drop(wal);
        assert_eq!(recovered_ids(dir.path()), vec![1]);
    }

    #[test]
    fn legacy_commit_success_keeps_existing_frame_bytes_and_replay() {
        let dir = tempdir().unwrap();
        let wal = open_sync(dir.path());
        let group = records(1, false);
        wal.log_batch(&group).unwrap();
        let mut expected = Vec::new();
        for record in &group {
            let payload =
                bincode::serde::encode_to_vec(record, bincode::config::standard()).unwrap();
            expected.extend_from_slice(&u32::try_from(payload.len()).unwrap().to_le_bytes());
            expected.extend_from_slice(&payload);
            expected.extend_from_slice(&crc32fast::hash(&payload).to_le_bytes());
        }
        assert_eq!(fs::read(wal.path()).unwrap(), expected);
        drop(wal);
        assert_eq!(recovered_ids(dir.path()), vec![1]);
    }
}
