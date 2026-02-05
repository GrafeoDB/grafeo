//! Adaptive WAL flusher with self-tuning timing.
//!
//! The flusher maintains a consistent flush cadence by adjusting wait times
//! based on actual flush duration. This prevents latency spikes when disk
//! operations take longer than expected.
//!
//! # Example
//!
//! With a 100ms target interval:
//! - If flush takes 20ms → wait 80ms before next flush
//! - If flush takes 80ms → wait 20ms before next flush
//! - If flush takes 120ms → flush immediately (already past target)
//!
//! This ensures the average interval stays close to the target regardless
//! of disk speed variations.

use std::sync::Arc;
use std::sync::mpsc::{self, RecvTimeoutError};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use super::WalManager;

/// Statistics tracked by the adaptive flusher.
#[derive(Debug, Clone, Copy, Default)]
pub struct FlusherStats {
    /// Total number of flushes performed.
    pub flush_count: u64,
    /// Total time spent flushing (microseconds).
    pub total_flush_time_us: u64,
    /// Maximum flush duration seen (microseconds).
    pub max_flush_time_us: u64,
    /// Number of times flush exceeded target interval.
    pub exceeded_target_count: u64,
}

impl FlusherStats {
    /// Returns the average flush duration in microseconds.
    #[must_use]
    pub fn avg_flush_time_us(&self) -> u64 {
        if self.flush_count == 0 {
            0
        } else {
            self.total_flush_time_us / self.flush_count
        }
    }
}

/// Adaptive WAL flusher that adjusts timing based on flush duration.
///
/// Spawns a background thread that periodically syncs the WAL to disk.
/// The wait time between flushes adapts to maintain a consistent cadence:
///
/// ```text
/// wait_time = target_interval - last_flush_duration
/// ```
///
/// This self-tuning approach:
/// - Prevents thundering herd problems when disk is slow
/// - Maintains consistent flush frequency regardless of disk speed
/// - Provides graceful shutdown with final flush guarantee
pub struct AdaptiveFlusher {
    /// Target interval between flushes.
    target_interval: Duration,
    /// Channel to signal shutdown (sends ack channel back).
    shutdown_tx: Option<mpsc::Sender<mpsc::Sender<FlusherStats>>>,
    /// Background thread handle.
    handle: Option<JoinHandle<()>>,
}

impl AdaptiveFlusher {
    /// Creates and starts a new adaptive flusher.
    ///
    /// The flusher will sync the WAL at approximately `target_interval_ms`
    /// intervals, adjusting for actual flush duration.
    ///
    /// # Arguments
    ///
    /// * `wal` - The WAL manager to flush
    /// * `target_interval_ms` - Target interval between flushes in milliseconds
    #[must_use]
    pub fn new(wal: Arc<WalManager>, target_interval_ms: u64) -> Self {
        let target_interval = Duration::from_millis(target_interval_ms);
        let (shutdown_tx, shutdown_rx) = mpsc::channel();

        let handle = thread::Builder::new()
            .name("grafeo-wal-flusher".to_string())
            .spawn(move || {
                Self::flusher_loop(wal, target_interval, shutdown_rx);
            })
            .expect("Failed to spawn WAL flusher thread");

        Self {
            target_interval,
            shutdown_tx: Some(shutdown_tx),
            handle: Some(handle),
        }
    }

    /// Returns the target flush interval.
    #[must_use]
    pub fn target_interval(&self) -> Duration {
        self.target_interval
    }

    /// Gracefully shuts down the flusher, performing a final flush.
    ///
    /// Returns statistics about the flusher's operation.
    ///
    /// # Errors
    ///
    /// Returns an error if the shutdown signal cannot be sent or acknowledged.
    pub fn shutdown(&mut self) -> Result<FlusherStats, String> {
        let stats = if let Some(tx) = self.shutdown_tx.take() {
            let (ack_tx, ack_rx) = mpsc::channel();
            tx.send(ack_tx)
                .map_err(|e| format!("Failed to send shutdown signal: {e}"))?;
            ack_rx
                .recv()
                .map_err(|e| format!("Failed to receive shutdown acknowledgment: {e}"))?
        } else {
            FlusherStats::default()
        };

        if let Some(handle) = self.handle.take() {
            handle
                .join()
                .map_err(|_| "Flusher thread panicked".to_string())?;
        }

        Ok(stats)
    }

    /// The main flusher loop running in the background thread.
    fn flusher_loop(
        wal: Arc<WalManager>,
        target_interval: Duration,
        shutdown_rx: mpsc::Receiver<mpsc::Sender<FlusherStats>>,
    ) {
        let mut last_flush_duration = Duration::ZERO;
        let mut stats = FlusherStats::default();

        loop {
            // Adaptive timeout: account for how long the last flush took
            let timeout = target_interval.saturating_sub(last_flush_duration);

            match shutdown_rx.recv_timeout(timeout) {
                Ok(ack_tx) => {
                    // Graceful shutdown requested - do final flush
                    if let Err(e) = wal.sync() {
                        tracing::warn!("Final WAL flush failed: {e}");
                    }
                    // Send stats back to acknowledge shutdown
                    let _ = ack_tx.send(stats);
                    return;
                }
                Err(RecvTimeoutError::Timeout) => {
                    // Time to flush
                    let start = Instant::now();

                    if let Err(e) = wal.sync() {
                        tracing::warn!("WAL flush failed: {e}");
                        // Still update timing to avoid spin loop on persistent errors
                        last_flush_duration = Duration::from_millis(10);
                        continue;
                    }

                    last_flush_duration = start.elapsed();

                    // Update statistics
                    stats.flush_count += 1;
                    let flush_us = last_flush_duration.as_micros() as u64;
                    stats.total_flush_time_us += flush_us;
                    stats.max_flush_time_us = stats.max_flush_time_us.max(flush_us);

                    if last_flush_duration > target_interval {
                        stats.exceeded_target_count += 1;
                        tracing::debug!(
                            "WAL flush took {:?}, exceeds target {:?}",
                            last_flush_duration,
                            target_interval
                        );
                    }
                }
                Err(RecvTimeoutError::Disconnected) => {
                    // Channel closed without shutdown signal - exit gracefully
                    tracing::debug!("Flusher shutdown channel disconnected");
                    return;
                }
            }
        }
    }
}

impl Drop for AdaptiveFlusher {
    fn drop(&mut self) {
        if self.shutdown_tx.is_some() {
            if let Err(e) = self.shutdown() {
                tracing::warn!("Error during flusher drop: {e}");
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn test_adaptive_flusher_basic() {
        let dir = tempdir().unwrap();
        let wal = Arc::new(WalManager::open(dir.path()).unwrap());

        // Start flusher with 50ms target
        let mut flusher = AdaptiveFlusher::new(Arc::clone(&wal), 50);

        // Let it run for a bit (500ms gives plenty of margin for CI)
        thread::sleep(Duration::from_millis(500));

        // Shutdown and get stats
        let stats = flusher.shutdown().unwrap();

        // Should have done at least 2 flushes in 500ms with 50ms target
        assert!(
            stats.flush_count >= 2,
            "Expected at least 2 flushes, got {}",
            stats.flush_count
        );
    }

    #[test]
    fn test_adaptive_flusher_shutdown() {
        let dir = tempdir().unwrap();
        let wal = Arc::new(WalManager::open(dir.path()).unwrap());

        let mut flusher = AdaptiveFlusher::new(Arc::clone(&wal), 100);

        // Immediate shutdown should work
        let stats = flusher.shutdown().unwrap();
        assert!(stats.flush_count <= 2); // At most one or two flushes
    }

    #[test]
    fn test_adaptive_flusher_target_interval() {
        let dir = tempdir().unwrap();
        let wal = Arc::new(WalManager::open(dir.path()).unwrap());

        let flusher = AdaptiveFlusher::new(Arc::clone(&wal), 75);
        assert_eq!(flusher.target_interval(), Duration::from_millis(75));
    }

    #[test]
    fn test_flusher_stats() {
        let stats = FlusherStats {
            flush_count: 10,
            total_flush_time_us: 5000, // 5ms total
            max_flush_time_us: 1000,   // 1ms max
            ..Default::default()
        };

        assert_eq!(stats.avg_flush_time_us(), 500); // 0.5ms average
    }

    #[test]
    fn test_adaptive_flusher_drop() {
        let dir = tempdir().unwrap();
        let wal = Arc::new(WalManager::open(dir.path()).unwrap());

        // Create flusher and let it drop naturally
        {
            let _flusher = AdaptiveFlusher::new(Arc::clone(&wal), 50);
            thread::sleep(Duration::from_millis(100));
            // Drop should trigger shutdown
        }

        // WAL should still be usable after flusher is dropped
        assert!(wal.flush().is_ok());
    }
}
