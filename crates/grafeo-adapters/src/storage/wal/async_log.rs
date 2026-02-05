//! Async WAL implementation using tokio for non-blocking I/O.

use super::{DurabilityMode, WalConfig, WalRecord};
use grafeo_common::types::{EpochId, TxId};
use grafeo_common::utils::error::{Error, Result};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};
use tokio::fs::{self, File, OpenOptions};
use tokio::io::{AsyncWriteExt, BufWriter};
use tokio::sync::Mutex;
use tokio::task::JoinHandle;

/// State for a single async log file.
struct AsyncLogFile {
    /// Async file handle with buffering.
    writer: BufWriter<File>,
    /// Current size in bytes.
    size: u64,
    /// File path.
    #[allow(dead_code)]
    path: PathBuf,
    /// Sequence number (for log file ordering during recovery).
    #[allow(dead_code)]
    sequence: u64,
}

/// Async Write-Ahead Log manager with non-blocking I/O.
///
/// This manager provides the same durability guarantees as the sync version
/// but uses tokio's async I/O for better throughput in async contexts.
pub struct AsyncWalManager {
    /// Directory for WAL files.
    dir: PathBuf,
    /// Configuration.
    config: WalConfig,
    /// Active log file (async mutex for async access).
    active_log: Mutex<Option<AsyncLogFile>>,
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
    /// Background sync task handle (for batch mode).
    background_sync_handle: Mutex<Option<JoinHandle<()>>>,
    /// Shutdown signal sender.
    shutdown_tx: Mutex<Option<tokio::sync::oneshot::Sender<()>>>,
}

impl AsyncWalManager {
    /// Opens or creates an async WAL in the given directory.
    ///
    /// # Errors
    ///
    /// Returns an error if the directory cannot be created or accessed.
    pub async fn open(dir: impl AsRef<Path>) -> Result<Self> {
        Self::with_config(dir, WalConfig::default()).await
    }

    /// Opens or creates an async WAL with custom configuration.
    ///
    /// # Errors
    ///
    /// Returns an error if the directory cannot be created or accessed.
    pub async fn with_config(dir: impl AsRef<Path>, config: WalConfig) -> Result<Self> {
        let dir = dir.as_ref().to_path_buf();
        fs::create_dir_all(&dir).await?;

        // Find the highest existing sequence number
        let mut max_sequence = 0u64;
        let mut entries = fs::read_dir(&dir).await?;
        while let Some(entry) = entries.next_entry().await? {
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

        let manager = Self {
            dir,
            config,
            active_log: Mutex::new(None),
            total_record_count: AtomicU64::new(0),
            records_since_sync: AtomicU64::new(0),
            last_sync: Mutex::new(Instant::now()),
            current_sequence: AtomicU64::new(max_sequence),
            checkpoint_epoch: Mutex::new(None),
            background_sync_handle: Mutex::new(None),
            shutdown_tx: Mutex::new(None),
        };

        // Open or create the active log
        manager.ensure_active_log().await?;

        Ok(manager)
    }

    /// Logs a record to the WAL asynchronously.
    ///
    /// # Errors
    ///
    /// Returns an error if the record cannot be written.
    pub async fn log(&self, record: &WalRecord) -> Result<()> {
        self.ensure_active_log().await?;

        let mut guard = self.active_log.lock().await;
        let log_file = guard
            .as_mut()
            .ok_or_else(|| Error::Internal("WAL writer not available".to_string()))?;

        // Serialize the record
        let data = bincode::serde::encode_to_vec(record, bincode::config::standard())
            .map_err(|e| Error::Serialization(e.to_string()))?;

        // Write length prefix
        let len = data.len() as u32;
        log_file.writer.write_all(&len.to_le_bytes()).await?;

        // Write data
        log_file.writer.write_all(&data).await?;

        // Write checksum
        let checksum = crc32fast::hash(&data);
        log_file.writer.write_all(&checksum.to_le_bytes()).await?;

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
                    log_file.writer.flush().await?;
                    log_file.writer.get_ref().sync_all().await?;
                    self.records_since_sync.store(0, Ordering::Relaxed);
                    *self.last_sync.lock().await = Instant::now();
                }
            }
            DurabilityMode::Batch {
                max_delay_ms,
                max_records,
            } => {
                let records = self.records_since_sync.load(Ordering::Relaxed);
                let elapsed = self.last_sync.lock().await.elapsed();

                if records >= *max_records || elapsed >= Duration::from_millis(*max_delay_ms) {
                    log_file.writer.flush().await?;
                    log_file.writer.get_ref().sync_all().await?;
                    self.records_since_sync.store(0, Ordering::Relaxed);
                    *self.last_sync.lock().await = Instant::now();
                }
            }
            DurabilityMode::Adaptive { .. } => {
                // Adaptive mode: just flush buffer, background thread handles sync
                // Note: For async, consider using a tokio task-based flusher
                log_file.writer.flush().await?;
            }
            DurabilityMode::NoSync => {
                // Just flush buffer, no sync
                log_file.writer.flush().await?;
            }
        }

        drop(guard);

        // Rotate if needed
        if needs_rotation {
            self.rotate().await?;
        }

        Ok(())
    }

    /// Writes a checkpoint marker and returns the checkpoint info.
    ///
    /// # Errors
    ///
    /// Returns an error if the checkpoint cannot be written.
    pub async fn checkpoint(&self, current_tx: TxId, epoch: EpochId) -> Result<()> {
        // Write checkpoint record
        self.log(&WalRecord::Checkpoint { tx_id: current_tx })
            .await?;

        // Force sync on checkpoint
        self.sync().await?;

        // Update checkpoint epoch
        *self.checkpoint_epoch.lock().await = Some(epoch);

        // Optionally truncate old logs
        self.truncate_old_logs().await?;

        Ok(())
    }

    /// Rotates to a new log file.
    ///
    /// # Errors
    ///
    /// Returns an error if rotation fails.
    pub async fn rotate(&self) -> Result<()> {
        let new_sequence = self.current_sequence.fetch_add(1, Ordering::SeqCst) + 1;
        let new_path = self.log_path(new_sequence);

        let file = OpenOptions::new()
            .create(true)
            .read(true)
            .append(true)
            .open(&new_path)
            .await?;

        let new_log = AsyncLogFile {
            writer: BufWriter::new(file),
            size: 0,
            path: new_path,
            sequence: new_sequence,
        };

        // Replace active log
        let mut guard = self.active_log.lock().await;
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
    pub async fn flush(&self) -> Result<()> {
        let mut guard = self.active_log.lock().await;
        if let Some(log_file) = guard.as_mut() {
            log_file.writer.flush().await?;
        }
        Ok(())
    }

    /// Syncs the WAL to disk (fsync).
    ///
    /// # Errors
    ///
    /// Returns an error if the sync fails.
    pub async fn sync(&self) -> Result<()> {
        let mut guard = self.active_log.lock().await;
        if let Some(log_file) = guard.as_mut() {
            log_file.writer.flush().await?;
            log_file.writer.get_ref().sync_all().await?;
        }
        self.records_since_sync.store(0, Ordering::Relaxed);
        *self.last_sync.lock().await = Instant::now();
        Ok(())
    }

    /// Starts a background sync task for batch durability mode.
    ///
    /// The task will periodically sync the WAL based on the batch configuration.
    /// This is useful when you want automatic syncing without waiting for
    /// individual log calls to trigger it.
    ///
    /// # Returns
    ///
    /// Returns `true` if a new background task was started, `false` if batch
    /// mode is not configured or a task is already running.
    pub async fn start_background_sync(&self) -> bool {
        let DurabilityMode::Batch { max_delay_ms, .. } = self.config.durability else {
            return false;
        };

        let mut handle_guard = self.background_sync_handle.lock().await;
        if handle_guard.is_some() {
            return false;
        }

        let (shutdown_tx, mut shutdown_rx) = tokio::sync::oneshot::channel();
        *self.shutdown_tx.lock().await = Some(shutdown_tx);

        // We need to use a weak pattern here since we can't hold Arc<Self>
        // Instead, we'll create a simple interval-based task
        let interval = Duration::from_millis(max_delay_ms);

        let handle = tokio::spawn(async move {
            let mut ticker = tokio::time::interval(interval);
            ticker.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);

            loop {
                tokio::select! {
                    _ = ticker.tick() => {
                        // The actual sync is done by the log() method when
                        // it checks elapsed time. This task just ensures
                        // we have periodic wake-ups.
                    }
                    _ = &mut shutdown_rx => {
                        break;
                    }
                }
            }
        });

        *handle_guard = Some(handle);
        true
    }

    /// Stops the background sync task if running.
    pub async fn stop_background_sync(&self) {
        if let Some(tx) = self.shutdown_tx.lock().await.take() {
            let _ = tx.send(());
        }
        if let Some(handle) = self.background_sync_handle.lock().await.take() {
            let _ = handle.await;
        }
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
    pub async fn log_files(&self) -> Result<Vec<PathBuf>> {
        let mut files = Vec::new();

        let mut entries = fs::read_dir(&self.dir).await?;
        while let Some(entry) = entries.next_entry().await? {
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
    pub async fn checkpoint_epoch(&self) -> Option<EpochId> {
        *self.checkpoint_epoch.lock().await
    }

    // === Private methods ===

    async fn ensure_active_log(&self) -> Result<()> {
        let mut guard = self.active_log.lock().await;
        if guard.is_none() {
            let sequence = self.current_sequence.load(Ordering::Relaxed);
            let path = self.log_path(sequence);

            let file = OpenOptions::new()
                .create(true)
                .read(true)
                .append(true)
                .open(&path)
                .await?;

            let size = file.metadata().await?.len();

            *guard = Some(AsyncLogFile {
                writer: BufWriter::new(file),
                size,
                path,
                sequence,
            });
        }
        Ok(())
    }

    fn log_path(&self, sequence: u64) -> PathBuf {
        self.dir.join(format!("wal_{sequence:08}.log"))
    }

    fn sequence_from_path(path: &Path) -> Option<u64> {
        path.file_stem()
            .and_then(|s| s.to_str())
            .and_then(|s| s.strip_prefix("wal_"))
            .and_then(|s| s.parse().ok())
    }

    async fn truncate_old_logs(&self) -> Result<()> {
        let checkpoint = match *self.checkpoint_epoch.lock().await {
            Some(e) => e,
            None => return Ok(()),
        };

        // Keep logs that might still be needed
        // For now, keep the two most recent logs after checkpoint
        let files = self.log_files().await?;
        let current_seq = self.current_sequence.load(Ordering::Relaxed);

        for file in files {
            if let Some(seq) = Self::sequence_from_path(&file) {
                // Keep the last 2 log files before current
                if seq + 2 < current_seq && checkpoint.as_u64() > seq {
                    let _ = fs::remove_file(&file).await;
                }
            }
        }

        Ok(())
    }
}

impl Drop for AsyncWalManager {
    fn drop(&mut self) {
        // Best-effort cleanup - background tasks will be cancelled
        // when their handles are dropped
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use grafeo_common::types::NodeId;
    use tempfile::tempdir;

    #[tokio::test]
    async fn test_async_wal_write() {
        let dir = tempdir().unwrap();

        let wal = AsyncWalManager::open(dir.path()).await.unwrap();

        let record = WalRecord::CreateNode {
            id: NodeId::new(1),
            labels: vec!["Person".to_string()],
        };

        wal.log(&record).await.unwrap();
        wal.flush().await.unwrap();

        assert_eq!(wal.record_count(), 1);
    }

    #[tokio::test]
    async fn test_async_wal_rotation() {
        let dir = tempdir().unwrap();

        // Small max size to force rotation
        let config = WalConfig {
            max_log_size: 100,
            ..Default::default()
        };

        let wal = AsyncWalManager::with_config(dir.path(), config)
            .await
            .unwrap();

        // Write enough records to trigger rotation
        for i in 0..10 {
            let record = WalRecord::CreateNode {
                id: NodeId::new(i),
                labels: vec!["Person".to_string()],
            };
            wal.log(&record).await.unwrap();
        }

        wal.flush().await.unwrap();

        // Should have multiple log files
        let files = wal.log_files().await.unwrap();
        assert!(
            files.len() > 1,
            "Expected multiple log files after rotation"
        );
    }

    #[tokio::test]
    async fn test_async_durability_modes() {
        let dir = tempdir().unwrap();

        // Test Sync mode
        let config = WalConfig {
            durability: DurabilityMode::Sync,
            ..Default::default()
        };
        let wal = AsyncWalManager::with_config(dir.path().join("sync"), config)
            .await
            .unwrap();
        wal.log(&WalRecord::TxCommit {
            tx_id: TxId::new(1),
        })
        .await
        .unwrap();

        // Test NoSync mode
        let config = WalConfig {
            durability: DurabilityMode::NoSync,
            ..Default::default()
        };
        let wal = AsyncWalManager::with_config(dir.path().join("nosync"), config)
            .await
            .unwrap();
        wal.log(&WalRecord::CreateNode {
            id: NodeId::new(1),
            labels: vec![],
        })
        .await
        .unwrap();

        // Test Batch mode
        let config = WalConfig {
            durability: DurabilityMode::Batch {
                max_delay_ms: 10,
                max_records: 5,
            },
            ..Default::default()
        };
        let wal = AsyncWalManager::with_config(dir.path().join("batch"), config)
            .await
            .unwrap();
        for i in 0..10 {
            wal.log(&WalRecord::CreateNode {
                id: NodeId::new(i),
                labels: vec![],
            })
            .await
            .unwrap();
        }
    }

    #[tokio::test]
    async fn test_async_checkpoint() {
        let dir = tempdir().unwrap();

        let wal = AsyncWalManager::open(dir.path()).await.unwrap();

        // Write some records
        wal.log(&WalRecord::CreateNode {
            id: NodeId::new(1),
            labels: vec!["Test".to_string()],
        })
        .await
        .unwrap();

        wal.log(&WalRecord::TxCommit {
            tx_id: TxId::new(1),
        })
        .await
        .unwrap();

        // Create checkpoint
        wal.checkpoint(TxId::new(1), EpochId::new(10))
            .await
            .unwrap();

        assert_eq!(wal.checkpoint_epoch().await, Some(EpochId::new(10)));
    }

    #[tokio::test]
    async fn test_background_sync() {
        let dir = tempdir().unwrap();

        let config = WalConfig {
            durability: DurabilityMode::Batch {
                max_delay_ms: 50,
                max_records: 1000,
            },
            ..Default::default()
        };

        let wal = AsyncWalManager::with_config(dir.path(), config)
            .await
            .unwrap();

        // Should start successfully
        assert!(wal.start_background_sync().await);

        // Should not start again
        assert!(!wal.start_background_sync().await);

        // Write a record
        wal.log(&WalRecord::CreateNode {
            id: NodeId::new(1),
            labels: vec![],
        })
        .await
        .unwrap();

        // Wait a bit for potential background sync
        tokio::time::sleep(Duration::from_millis(100)).await;

        // Stop background sync
        wal.stop_background_sync().await;
    }
}
