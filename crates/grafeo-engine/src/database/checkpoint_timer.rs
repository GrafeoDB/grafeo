//! Automatic checkpoints (#426).
//!
//! When a database has [`CheckpointTriggers`](crate::admin::CheckpointTriggers)
//! (from [`Config::with_checkpoint_interval`](crate::Config::with_checkpoint_interval),
//! [`Config::with_checkpoint_wal_size`](crate::Config::with_checkpoint_wal_size)
//! or `GrafeoDB::set_checkpoint_triggers`), a background thread checkpoints
//! it to the `.grafeo` file: when the interval has passed since its last
//! checkpoint, or once the WAL holds more than the given bytes since the last
//! checkpoint. This bounds the WAL, and with it the time a reopen replays.
//!
//! The thread polls in short steps (100 ms), so `close()` stops it promptly
//! and a WAL past its size is checkpointed within one step. A failed
//! checkpoint is logged, counted in the [`Failures`] the database reports
//! (`GrafeoDB::checkpoint_status`), and retried after a pause that doubles
//! from one second to five minutes; the WAL keeps the commits meanwhile.

#[cfg(feature = "grafeo-file")]
use std::sync::Arc;
#[cfg(feature = "grafeo-file")]
use std::sync::atomic::{AtomicBool, Ordering};
#[cfg(feature = "grafeo-file")]
use std::time::{Duration, Instant};

#[cfg(feature = "grafeo-file")]
use grafeo_common::utils::error::Result;

#[cfg(feature = "grafeo-file")]
use crate::admin::CheckpointTriggers;

/// How often the thread checks the shutdown flag and the triggers.
#[cfg(feature = "grafeo-file")]
const POLL_INTERVAL: Duration = Duration::from_millis(100);

/// The first pause before a failed checkpoint is retried.
#[cfg(feature = "grafeo-file")]
const FIRST_RETRY: Duration = Duration::from_secs(1);

/// The longest pause before a failed checkpoint is retried.
#[cfg(feature = "grafeo-file")]
const LAST_RETRY: Duration = Duration::from_mins(5);

/// What the automatic checkpoints of a database have met since the last one
/// that succeeded, shared by the thread and the database.
#[cfg(feature = "grafeo-file")]
#[derive(Debug, Default)]
pub(super) struct Failures {
    state: parking_lot::Mutex<FailureState>,
}

#[cfg(feature = "grafeo-file")]
#[derive(Debug, Clone, Default)]
pub(super) struct FailureState {
    /// Failed checkpoints since the last one that succeeded.
    pub failures: u64,
    /// The error of the last one that failed.
    pub last_error: Option<String>,
    /// When it failed (Unix epoch seconds).
    pub last_failure: Option<u64>,
    /// Whether the thread stopped, as no checkpoint can succeed any more.
    pub stopped: bool,
}

#[cfg(feature = "grafeo-file")]
impl Failures {
    /// What they are now.
    pub(super) fn snapshot(&self) -> FailureState {
        self.state.lock().clone()
    }

    fn succeeded(&self) {
        let mut state = self.state.lock();
        state.failures = 0;
    }

    fn failed(&self, error: String, stopped: bool) {
        let mut state = self.state.lock();
        state.failures += 1;
        state.last_error = Some(error);
        state.last_failure = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .ok()
            .map(|since| since.as_secs());
        state.stopped |= stopped;
    }
}

/// What the thread runs and reads: a checkpoint, the bytes the WAL holds
/// since the last checkpoint, and whether no checkpoint can succeed any more.
#[cfg(feature = "grafeo-file")]
pub(super) struct Job {
    /// Runs one checkpoint.
    pub checkpoint: Box<dyn Fn() -> Result<()> + Send>,
    /// The bytes the WAL holds since the last checkpoint (0 without one).
    pub wal_bytes: Box<dyn Fn() -> u64 + Send>,
    /// Whether no checkpoint can succeed any more (after a commit that did
    /// not complete).
    pub hopeless: Box<dyn Fn() -> bool + Send>,
}

/// The background checkpointer.
///
/// The thread exits when [`stop`](Self::stop) is called (from
/// `GrafeoDB::close()`, or when the triggers change).
#[cfg(feature = "grafeo-file")]
pub(super) struct CheckpointTimer {
    /// Shutdown signal: set to true to stop the thread.
    shutdown: Arc<AtomicBool>,
    /// Thread handle (taken on stop).
    handle: Option<std::thread::JoinHandle<()>>,
}

#[cfg(feature = "grafeo-file")]
impl CheckpointTimer {
    /// Starts the checkpointer for `triggers`, at least one of which is set.
    pub(super) fn start(triggers: CheckpointTriggers, job: Job, failures: Arc<Failures>) -> Self {
        let shutdown = Arc::new(AtomicBool::new(false));
        let shutdown_clone = Arc::clone(&shutdown);

        let handle = std::thread::Builder::new()
            .name("grafeo-checkpoint".to_string())
            .spawn(move || {
                Self::run(&shutdown_clone, triggers, &job, &failures);
            })
            .expect("failed to spawn checkpoint timer thread");

        Self {
            shutdown,
            handle: Some(handle),
        }
    }

    /// Signals the thread to stop and waits for it to exit.
    ///
    /// Returns within ~100 ms regardless of the triggers, unless a
    /// checkpoint is running.
    pub(super) fn stop(&mut self) {
        self.shutdown.store(true, Ordering::Release);
        if let Some(handle) = self.handle.take() {
            let _ = handle.join();
        }
    }

    /// The loop: sleep in short steps, checkpoint when a trigger fires,
    /// exit when shutdown is signaled.
    fn run(shutdown: &AtomicBool, triggers: CheckpointTriggers, job: &Job, failures: &Failures) {
        let mut last = Instant::now();
        let mut retry: Option<(Instant, Duration)> = None;

        loop {
            std::thread::sleep(POLL_INTERVAL);

            if shutdown.load(Ordering::Acquire) {
                break;
            }

            let due = triggers
                .interval
                .is_some_and(|interval| last.elapsed() >= interval)
                || triggers
                    .wal_bytes
                    .is_some_and(|limit| (job.wal_bytes)() >= limit);
            if !due || retry.is_some_and(|(at, _)| Instant::now() < at) {
                continue;
            }

            last = Instant::now();
            match (job.checkpoint)() {
                Ok(()) => {
                    failures.succeeded();
                    retry = None;
                }
                Err(error) => {
                    // After a commit that did not complete, no checkpoint can
                    // ever succeed (it would write the commit's stamped part):
                    // say so once and stop.
                    if (job.hopeless)() {
                        grafeo_common::grafeo_error!(
                            "automatic checkpoints stop: {error}; the file keeps its last checkpoint"
                        );
                        failures.failed(error.to_string(), true);
                        break;
                    }
                    let pause = retry.map_or(FIRST_RETRY, |(_, pause)| (pause * 2).min(LAST_RETRY));
                    grafeo_common::grafeo_warn!(
                        "automatic checkpoint failed, retried in {}s: {error}",
                        pause.as_secs()
                    );
                    failures.failed(error.to_string(), false);
                    retry = Some((Instant::now() + pause, pause));
                }
            }
        }
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
    use grafeo_common::utils::error::Error;
    use std::sync::atomic::AtomicU64;

    /// A job that counts its checkpoints, fails the first `failing` of them,
    /// and reports `wal` bytes until a checkpoint succeeds.
    fn job(
        checkpoints: &Arc<AtomicU64>,
        failing: u64,
        wal: &Arc<AtomicU64>,
        hopeless: bool,
    ) -> Job {
        let (count, bytes) = (Arc::clone(checkpoints), Arc::clone(wal));
        let after = Arc::clone(wal);
        Job {
            checkpoint: Box::new(move || {
                let n = count.fetch_add(1, Ordering::SeqCst) + 1;
                if n <= failing {
                    return Err(Error::Internal(format!("injected failure {n}")));
                }
                after.store(0, Ordering::SeqCst);
                Ok(())
            }),
            wal_bytes: Box::new(move || bytes.load(Ordering::SeqCst)),
            hopeless: Box::new(move || hopeless),
        }
    }

    /// Waits up to ten seconds (a loaded machine runs the thread late).
    fn wait_until(what: &str, condition: impl Fn() -> bool) {
        let deadline = Instant::now() + Duration::from_secs(10);
        while !condition() {
            assert!(Instant::now() < deadline, "never: {what}");
            std::thread::sleep(Duration::from_millis(10));
        }
    }

    #[test]
    fn the_timer_stops_promptly() {
        let (checkpoints, wal) = (Arc::default(), Arc::default());
        let mut timer = CheckpointTimer::start(
            CheckpointTriggers::none().with_interval(Duration::from_mins(1)),
            job(&checkpoints, 0, &wal, false),
            Arc::default(),
        );
        let started = Instant::now();
        timer.stop();
        assert!(
            started.elapsed() < Duration::from_secs(2),
            "stop() took {:?}",
            started.elapsed()
        );
        assert_eq!(checkpoints.load(Ordering::SeqCst), 0);
    }

    #[test]
    fn an_interval_checkpoints_each_time_it_passes() {
        let (checkpoints, wal) = (Arc::new(AtomicU64::new(0)), Arc::default());
        let mut timer = CheckpointTimer::start(
            CheckpointTriggers::none().with_interval(Duration::from_millis(200)),
            job(&checkpoints, 0, &wal, false),
            Arc::default(),
        );
        wait_until("two checkpoints", || {
            checkpoints.load(Ordering::SeqCst) >= 2
        });
        timer.stop();
    }

    /// A WAL past the size is checkpointed; below it, nothing happens.
    #[test]
    fn a_wal_size_checkpoints_once_the_wal_passes_it() {
        let (checkpoints, wal) = (Arc::new(AtomicU64::new(0)), Arc::new(AtomicU64::new(88)));
        let mut timer = CheckpointTimer::start(
            CheckpointTriggers::none().with_wal_bytes(1024),
            job(&checkpoints, 0, &wal, false),
            Arc::default(),
        );
        std::thread::sleep(Duration::from_millis(400));
        assert_eq!(
            checkpoints.load(Ordering::SeqCst),
            0,
            "88 bytes are below it"
        );
        wal.store(1024, Ordering::SeqCst);
        wait_until("a checkpoint", || checkpoints.load(Ordering::SeqCst) == 1);
        std::thread::sleep(Duration::from_millis(400));
        assert_eq!(
            checkpoints.load(Ordering::SeqCst),
            1,
            "the checkpoint emptied the WAL"
        );
        timer.stop();
    }

    /// A failed checkpoint is counted with its error and retried after a
    /// pause, not at every step; the next success clears the count.
    #[test]
    fn a_failed_checkpoint_is_counted_and_retried_after_a_pause() {
        let (checkpoints, wal) = (Arc::new(AtomicU64::new(0)), Arc::new(AtomicU64::new(4096)));
        let failures = Arc::new(Failures::default());
        let mut timer = CheckpointTimer::start(
            CheckpointTriggers::none().with_wal_bytes(1024),
            job(&checkpoints, 1, &wal, false),
            Arc::clone(&failures),
        );
        wait_until("the failure", || failures.snapshot().failures == 1);
        let failed = failures.snapshot();
        assert_eq!(
            failed.last_error.as_deref(),
            Some("GRAFEO-X001: Internal error: injected failure 1")
        );
        assert!(failed.last_failure.is_some() && !failed.stopped);
        std::thread::sleep(Duration::from_millis(500));
        assert_eq!(
            checkpoints.load(Ordering::SeqCst),
            1,
            "no retry before the pause"
        );

        wait_until("the retry", || checkpoints.load(Ordering::SeqCst) == 2);
        wait_until("the count cleared", || failures.snapshot().failures == 0);
        assert_eq!(
            failures.snapshot().last_error.as_deref(),
            Some("GRAFEO-X001: Internal error: injected failure 1"),
            "the last error stays for the caller to read"
        );
        timer.stop();
    }

    /// When no checkpoint can succeed any more, the thread says so and stops.
    #[test]
    fn a_hopeless_failure_stops_the_checkpoints() {
        let (checkpoints, wal) = (Arc::new(AtomicU64::new(0)), Arc::new(AtomicU64::new(4096)));
        let failures = Arc::new(Failures::default());
        let mut timer = CheckpointTimer::start(
            CheckpointTriggers::none().with_wal_bytes(1024),
            job(&checkpoints, u64::MAX, &wal, true),
            Arc::clone(&failures),
        );
        wait_until("the stop", || failures.snapshot().stopped);
        std::thread::sleep(Duration::from_millis(1500));
        assert_eq!(checkpoints.load(Ordering::SeqCst), 1, "never retried");
        timer.stop();
    }
}
