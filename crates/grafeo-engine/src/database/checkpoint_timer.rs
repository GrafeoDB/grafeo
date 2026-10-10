//! Periodic checkpoint timer for automatic durability.
//!
//! When [`Config::checkpoint_interval`](crate::Config::checkpoint_interval) is set,
//! the engine spawns a background thread that periodically checkpoints the
//! database to the `.grafeo` container. This bounds the WAL size and limits data
//! loss to at most one interval on crash.
//!
//! The timer polls a shutdown flag in short intervals (100 ms) so `close()`
//! completes promptly without blocking for the full checkpoint interval.

#[cfg(feature = "grafeo-file")]
use std::sync::Arc;
#[cfg(feature = "grafeo-file")]
use std::sync::atomic::{AtomicBool, Ordering};
#[cfg(feature = "grafeo-file")]
use std::time::Duration;

#[cfg(feature = "grafeo-file")]
use grafeo_common::utils::error::Result;
#[cfg(feature = "grafeo-file")]
use grafeo_storage::file::GrafeoFileManager;

#[cfg(feature = "grafeo-file")]
use super::sections::CheckpointSources;

/// How often the timer thread checks the shutdown flag.
#[cfg(feature = "grafeo-file")]
const POLL_INTERVAL: Duration = Duration::from_millis(100);

/// Background checkpoint timer.
///
/// Spawns a thread that periodically triggers a unified flush. The thread
/// exits cleanly when [`stop`](Self::stop) is called (from `GrafeoDB::close()`).
#[cfg(feature = "grafeo-file")]
pub(super) struct CheckpointTimer {
    /// Shutdown signal: set to true to stop the timer thread.
    shutdown: Arc<AtomicBool>,
    /// Thread handle (taken on stop).
    handle: Option<std::thread::JoinHandle<()>>,
}

#[cfg(feature = "grafeo-file")]
impl CheckpointTimer {
    /// Starts the checkpoint timer.
    ///
    /// The background thread wakes every `interval` and checkpoints the
    /// database state in `sources`.
    pub(super) fn start(
        interval: Duration,
        file_manager: Arc<GrafeoFileManager>,
        sources: CheckpointSources,
        #[cfg(feature = "wal")] wal: Option<Arc<grafeo_storage::wal::LpgWal>>,
    ) -> Self {
        let shutdown = Arc::new(AtomicBool::new(false));
        let shutdown_clone = Arc::clone(&shutdown);

        let handle = std::thread::Builder::new()
            .name("grafeo-checkpoint".to_string())
            .spawn(move || {
                Self::run(
                    &shutdown_clone,
                    interval,
                    &file_manager,
                    &sources,
                    #[cfg(feature = "wal")]
                    wal.as_deref(),
                );
            })
            .expect("failed to spawn checkpoint timer thread");

        Self {
            shutdown,
            handle: Some(handle),
        }
    }

    /// Signals the timer thread to stop and waits for it to exit.
    ///
    /// Returns within ~100 ms regardless of the checkpoint interval.
    pub(super) fn stop(&mut self) {
        self.shutdown.store(true, Ordering::Release);
        if let Some(handle) = self.handle.take() {
            let _ = handle.join();
        }
    }

    /// Timer loop: sleep in short increments, checkpoint when the interval
    /// elapses, exit when shutdown is signaled.
    fn run(
        shutdown: &AtomicBool,
        interval: Duration,
        file_manager: &GrafeoFileManager,
        sources: &CheckpointSources,
        #[cfg(feature = "wal")] wal: Option<&grafeo_storage::wal::LpgWal>,
    ) {
        let mut elapsed = Duration::ZERO;

        loop {
            std::thread::sleep(POLL_INTERVAL);

            if shutdown.load(Ordering::Acquire) {
                break;
            }

            elapsed += POLL_INTERVAL;
            if elapsed < interval {
                continue;
            }
            elapsed = Duration::ZERO;

            // Attempt checkpoint (errors are logged, not propagated)
            if let Err(e) = Self::try_checkpoint(
                file_manager,
                sources,
                #[cfg(feature = "wal")]
                wal,
            ) {
                // After a commit that did not complete, no checkpoint can
                // ever succeed (it would write the commit's stamped part):
                // say so once and stop.
                if sources.transaction_manager.has_incomplete_commit() {
                    grafeo_common::grafeo_error!(
                        "periodic checkpoints stop: {e}; the file keeps its last checkpoint"
                    );
                    break;
                }
                eprintln!("periodic checkpoint failed: {e}");
            }
        }
    }

    /// Runs a single checkpoint cycle.
    fn try_checkpoint(
        file_manager: &GrafeoFileManager,
        sources: &CheckpointSources,
        #[cfg(feature = "wal")] wal: Option<&grafeo_storage::wal::LpgWal>,
    ) -> Result<()> {
        super::flush::flush(
            file_manager,
            sources,
            #[cfg(feature = "wal")]
            wal,
        )
        .map(|_| ())
    }
}

#[cfg(feature = "grafeo-file")]
impl Drop for CheckpointTimer {
    fn drop(&mut self) {
        self.stop();
    }
}

#[cfg(test)]
#[cfg(feature = "grafeo-file")]
mod tests {
    use super::*;
    use crate::catalog::Catalog;
    use crate::transaction::TransactionManager;
    use grafeo_core::graph::lpg::LpgStore;
    use std::time::Instant;

    fn sources(store: &Arc<LpgStore>) -> CheckpointSources {
        CheckpointSources {
            store: Some(Arc::clone(store)),
            catalog: Arc::new(Catalog::new()),
            transaction_manager: Arc::new(TransactionManager::new()),
            #[cfg(feature = "triple-store")]
            rdf_store: Arc::new(grafeo_core::graph::rdf::RdfStore::new()),
        }
    }

    fn start(
        interval: Duration,
        fm: &Arc<GrafeoFileManager>,
        store: &Arc<LpgStore>,
    ) -> CheckpointTimer {
        CheckpointTimer::start(
            interval,
            Arc::clone(fm),
            sources(store),
            #[cfg(feature = "wal")]
            None,
        )
    }

    #[test]
    fn timer_stops_promptly() {
        let store = Arc::new(LpgStore::new().unwrap());
        let dir = tempfile::TempDir::new().unwrap();
        let fm = Arc::new(
            GrafeoFileManager::create(dir.path().join("timer_test.grafeo"), None).unwrap(),
        );

        // Long interval
        let mut timer = start(Duration::from_mins(1), &fm, &store);

        let started = Instant::now();
        timer.stop();
        let elapsed = started.elapsed();

        // Should stop within a few poll cycles, not 60 seconds
        assert!(
            elapsed < Duration::from_secs(2),
            "stop() took {elapsed:?}, expected < 2s"
        );
    }

    /// Waits until a checkpoint moved the file's header past `iteration`,
    /// for up to ten seconds (a loaded machine may run the timer late).
    fn wait_for_a_checkpoint(fm: &GrafeoFileManager, iteration: u64) {
        let deadline = std::time::Instant::now() + Duration::from_secs(10);
        while fm.active_header().iteration == iteration {
            assert!(
                std::time::Instant::now() < deadline,
                "the timer never checkpointed"
            );
            std::thread::sleep(Duration::from_millis(10));
        }
    }

    #[test]
    fn timer_checkpoints_on_interval() {
        let store = Arc::new(LpgStore::new().unwrap());
        let dir = tempfile::TempDir::new().unwrap();
        let fm = Arc::new(
            GrafeoFileManager::create(dir.path().join("interval_test.grafeo"), None).unwrap(),
        );

        // Add some data so sections have content
        store.create_node(&["Test"]);

        // Short interval for testing
        let created = fm.active_header().iteration;
        let mut timer = start(Duration::from_millis(200), &fm, &store);

        // Wait for the first checkpoint, however loaded the machine is.
        wait_for_a_checkpoint(&fm, created);
        timer.stop();

        // Verify that a checkpoint happened (iteration > 0)
        let header = fm.active_header();
        assert!(
            header.iteration > 0,
            "expected at least one checkpoint, got iteration={}",
            header.iteration
        );
        assert_eq!(header.node_count, 1);
    }

    #[test]
    fn timer_runs_on_an_empty_database() {
        let store = Arc::new(LpgStore::new().unwrap());
        let dir = tempfile::TempDir::new().unwrap();
        let fm = Arc::new(
            GrafeoFileManager::create(dir.path().join("clean_test.grafeo"), None).unwrap(),
        );

        let created = fm.active_header().iteration;
        let mut timer = start(Duration::from_millis(200), &fm, &store);

        wait_for_a_checkpoint(&fm, created);
        timer.stop();

        let header = fm.active_header();
        assert!(
            header.iteration > created,
            "expected at least one checkpoint, got iteration={}",
            header.iteration
        );
        assert_eq!(header.node_count, 0);
    }
}
