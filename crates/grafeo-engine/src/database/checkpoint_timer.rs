//! Automatic checkpoints (#426; event-driven since 0.6.2).
//!
//! When a database has [`CheckpointTriggers`](crate::admin::CheckpointTriggers)
//! (from [`Config::with_checkpoint_wal_size`](crate::Config::with_checkpoint_wal_size),
//! 256 MiB by default for a file database,
//! [`Config::with_checkpoint_interval`](crate::Config::with_checkpoint_interval)
//! or `GrafeoDB::set_checkpoint_triggers`), a background thread checkpoints
//! it to the `.grafeo` file: once the WAL holds the given bytes past the last
//! checkpoint, or when the interval has passed since its last checkpoint.
//! This bounds the WAL on disk, and with it the time a reopen replays.
//!
//! The thread does nothing while nothing happens. It parks on a condition
//! variable, with the interval as the timeout when one is set. Only a commit
//! changes the size of the WAL, so the size check is the WAL's: the thread
//! arms a mark (the last checkpoint's position plus the size), and the first
//! group that ends past it signals the thread once, after the WAL's lock is
//! released ([`Wal::arm_wake`](grafeo_storage::wal::Wal::arm_wake)). The
//! checkpoint runs on the thread, never on the committing one. `stop()`
//! signals too, so `close()` returns at once.
//!
//! A failed checkpoint is logged and counted in the [`Failures`] the
//! database reports (`GrafeoDB::checkpoint_status`); the WAL keeps the
//! commits meanwhile. The size trigger tries again once the WAL has grown one
//! more segment past the failure, so a full disk with no writes costs
//! nothing. The interval trigger tries again after a pause that doubles from
//! one second to five minutes. A success resets both.

#[cfg(feature = "grafeo-file")]
use std::sync::Arc;
#[cfg(feature = "grafeo-file")]
use std::sync::atomic::{AtomicU64, Ordering};
#[cfg(feature = "grafeo-file")]
use std::time::{Duration, Instant};

#[cfg(feature = "grafeo-file")]
use grafeo_common::utils::error::Result;
#[cfg(feature = "grafeo-file")]
use parking_lot::{Condvar, Mutex};

#[cfg(feature = "grafeo-file")]
use crate::admin::CheckpointTriggers;

/// The first pause before a failed interval checkpoint is tried again.
#[cfg(feature = "grafeo-file")]
const FIRST_RETRY: Duration = Duration::from_secs(1);

/// The longest pause before a failed interval checkpoint is tried again.
#[cfg(feature = "grafeo-file")]
const LAST_RETRY: Duration = Duration::from_mins(5);

/// What the automatic checkpoints of a database have met since the last one
/// that succeeded, shared by the thread and the database.
#[cfg(feature = "grafeo-file")]
#[derive(Debug, Default)]
pub(super) struct Failures {
    state: Mutex<FailureState>,
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

/// The log the size trigger watches: the WAL, and where the last checkpoint
/// ended in it.
#[cfg(feature = "grafeo-file")]
pub(super) trait WatchedLog: Send {
    /// The log position the last checkpoint covers.
    fn checkpoint_lsn(&self) -> u64;
    /// The end of the log.
    fn end_lsn(&self) -> u64;
    /// Arms the mark `lsn` (see [`Wal::arm_wake`](grafeo_storage::wal::Wal::arm_wake)):
    /// `true` when the log already ends at or past it.
    fn arm_wake(&self, lsn: u64) -> bool;
    /// Sets what a crossing calls, or none, and disarms the mark with none.
    fn set_waker(&self, waker: Option<Arc<dyn Fn() + Send + Sync>>);
}

/// What the thread runs and watches.
#[cfg(feature = "grafeo-file")]
pub(super) struct Job {
    /// Runs one checkpoint.
    pub checkpoint: Box<dyn Fn() -> Result<()> + Send>,
    /// Whether no checkpoint can succeed any more (after a commit that did
    /// not complete).
    pub hopeless: Box<dyn Fn() -> bool + Send>,
    /// The WAL the size trigger watches; `None` without one, where the size
    /// trigger never fires.
    pub log: Option<Box<dyn WatchedLog>>,
    /// The size a WAL segment grows to: after a failed checkpoint the size
    /// trigger waits for this much more log.
    pub segment_bytes: u64,
    /// Without a WAL: whether a commit with changes came after the last
    /// checkpoint. An interval tick without one writes nothing, so a
    /// read-only workload never checkpoints. `None` with a WAL, whose
    /// position tells.
    pub changed: Option<Box<dyn Fn() -> bool + Send>>,
}

/// What wakes the thread, under the lock the thread checks it with: a wake
/// is never lost between the check and the wait.
#[cfg(feature = "grafeo-file")]
#[derive(Debug, Default)]
struct Wake {
    /// A commit took the WAL past the mark.
    signaled: bool,
    /// `stop()` was called.
    shutdown: bool,
}

/// The thread's condition variable and what it waits for.
#[cfg(feature = "grafeo-file")]
#[derive(Debug, Default)]
pub(super) struct Signal {
    wake: Mutex<Wake>,
    condvar: Condvar,
}

#[cfg(feature = "grafeo-file")]
impl Signal {
    /// Wakes the thread for a crossing of the mark.
    fn notify(&self) {
        self.wake.lock().signaled = true;
        self.condvar.notify_one();
    }

    /// Wakes the thread to exit.
    fn shut_down(&self) {
        self.wake.lock().shutdown = true;
        self.condvar.notify_one();
    }

    fn is_shut_down(&self) -> bool {
        self.wake.lock().shutdown
    }
}

/// What the thread did, for the test hook `GrafeoDB::checkpoint_thread_counts`.
#[cfg(feature = "grafeo-file")]
#[derive(Debug, Default)]
pub(super) struct Counts {
    starts: AtomicU64,
    wakes: AtomicU64,
    attempts: AtomicU64,
}

#[cfg(all(feature = "grafeo-file", feature = "testing-crash-injection"))]
impl Counts {
    pub(super) fn snapshot(&self) -> crate::admin::CheckpointThreadCounts {
        crate::admin::CheckpointThreadCounts {
            starts: self.starts.load(Ordering::Acquire),
            wakes: self.wakes.load(Ordering::Acquire),
            attempts: self.attempts.load(Ordering::Acquire),
        }
    }
}

/// The hold of `GrafeoDB::hold_checkpoint_rearm`: while held, the thread
/// waits after a checkpoint, before it arms its next mark.
#[cfg(all(feature = "grafeo-file", feature = "testing-crash-injection"))]
#[derive(Debug, Default)]
pub(crate) struct RearmGate {
    held: Mutex<bool>,
    condvar: Condvar,
}

#[cfg(all(feature = "grafeo-file", feature = "testing-crash-injection"))]
impl RearmGate {
    pub(super) fn hold(&self) {
        *self.held.lock() = true;
    }

    pub(crate) fn release(&self) {
        *self.held.lock() = false;
        self.condvar.notify_all();
    }

    /// Waits while held, or until the thread is to exit.
    fn wait_open(&self, signal: &Signal) {
        let mut held = self.held.lock();
        while *held && !signal.is_shut_down() {
            self.condvar.wait_for(&mut held, Duration::from_millis(10));
        }
    }
}

/// What the thread shares with the database besides its job.
#[cfg(feature = "grafeo-file")]
pub(super) struct Shared {
    pub failures: Arc<Failures>,
    pub counts: Arc<Counts>,
    #[cfg(feature = "testing-crash-injection")]
    pub gate: Arc<RearmGate>,
}

/// The background checkpointer, which starts its thread the first time it
/// has work.
///
/// A process that writes less than the WAL size never runs the thread (a
/// thread's mere start moves the allocator off its single-thread path): the
/// timer arms the WAL mark and waits. The commit whose group crosses the mark
/// starts the thread through the WAL's waker, and that checkpoint is the
/// thread's first job; a log already past the mark when the timer is armed
/// starts it at once. With an interval, the first committed change starts it:
/// its WAL group crosses a mark armed just past the log's end, or, without a
/// WAL, the transaction manager's first-commit hook calls
/// [`starter`](Self::starter); changes already in the log since the last
/// checkpoint start it at once.
///
/// The thread exits when [`stop`](Self::stop) is called (from
/// `GrafeoDB::close()`, or when the triggers change), or after a failure no
/// checkpoint can get past.
#[cfg(feature = "grafeo-file")]
pub(super) struct CheckpointTimer {
    lazy: Arc<Lazy>,
}

/// What a timer holds before, while and after its thread runs.
#[cfg(feature = "grafeo-file")]
enum Phase {
    /// No work yet: what the thread will run.
    Waiting(Job, Shared),
    /// The thread runs.
    Running(std::thread::JoinHandle<()>),
    /// Stopped, or the thread ended.
    Stopped,
}

/// The timer's state, which its starters share.
#[cfg(feature = "grafeo-file")]
struct Lazy {
    signal: Arc<Signal>,
    triggers: CheckpointTriggers,
    /// The phase, behind the timer's own lock: a start (from a commit's
    /// waker) never waits on the database's timer lock, which a restart holds
    /// while it joins a thread that may be checkpointing.
    phase: Mutex<Phase>,
}

#[cfg(feature = "grafeo-file")]
impl Lazy {
    /// A commit crossed the mark: a waiting timer starts its thread, counting
    /// the start as its first wake; a running thread is signaled.
    fn wake(self: &Arc<Self>) {
        let phase = self.phase.lock();
        if matches!(*phase, Phase::Running(..)) {
            drop(phase);
            self.signal.notify();
        } else {
            self.start_locked(phase, true);
        }
    }

    /// Starts the thread if it waits for work and the timer is not stopped;
    /// nothing otherwise. It spawns the thread and returns: it never waits for
    /// a checkpoint.
    fn start(self: &Arc<Self>) {
        let phase = self.phase.lock();
        self.start_locked(phase, false);
    }

    /// [`start`](Self::start) with the phase locked; `woken` counts the start
    /// as a wake.
    fn start_locked(self: &Arc<Self>, mut phase: parking_lot::MutexGuard<'_, Phase>, woken: bool) {
        if self.signal.is_shut_down() || !matches!(*phase, Phase::Waiting(..)) {
            return;
        }
        let Phase::Waiting(job, shared) = std::mem::replace(&mut *phase, Phase::Stopped) else {
            unreachable!("matched above")
        };
        let signal = Arc::clone(&self.signal);
        let triggers = self.triggers;
        shared.counts.starts.fetch_add(1, Ordering::AcqRel);
        let handle = std::thread::Builder::new()
            .name("grafeo-checkpoint".to_string())
            .spawn(move || {
                if woken {
                    shared.counts.wakes.fetch_add(1, Ordering::AcqRel);
                }
                let run = Run {
                    signal: &signal,
                    triggers,
                    job: &job,
                    shared: &shared,
                };
                run.run();
                // A thread that ends leaves no waker for the WAL to call.
                if let Some(log) = &job.log {
                    log.set_waker(None);
                }
            })
            .expect("failed to spawn checkpoint timer thread");
        *phase = Phase::Running(handle);
    }
}

#[cfg(feature = "grafeo-file")]
impl CheckpointTimer {
    /// Arms the checkpointer for `triggers`, at least one of which can fire:
    /// an interval, or a WAL size with `job.log` set. No thread starts unless
    /// work already waits: a log past the size mark, or, with an interval,
    /// the log past the last checkpoint's position. Without a log, the
    /// caller starts it ([`start`](Self::start)) when a change waits.
    pub(super) fn arm(triggers: CheckpointTriggers, job: Job, shared: Shared) -> Self {
        let signal = Arc::new(Signal::default());
        let lazy = Arc::new(Lazy {
            signal,
            triggers,
            phase: Mutex::new(Phase::Waiting(job, shared)),
        });
        // The mark is armed with the timer already waiting and its phase
        // locked: a commit that crosses it meanwhile calls `wake`, which
        // takes the lock after this and finds the timer waiting (the WAL
        // calls its waker with no lock of its own held), so no crossing is
        // lost.
        let phase = lazy.phase.lock();
        let Phase::Waiting(job, _) = &*phase else {
            unreachable!("set above")
        };
        // The mark that starts the thread: the size mark, or, with an
        // interval, just past the log's end, so the first commit starts it.
        let mark = job.log.as_deref().and_then(|log| {
            let size = triggers
                .wal_bytes
                .map(|limit| log.checkpoint_lsn().saturating_add(limit));
            let first_commit = triggers.interval.map(|_| log.end_lsn().saturating_add(1));
            size.into_iter().chain(first_commit).min()
        });
        if let (Some(log), Some(_)) = (&job.log, mark) {
            // Weak: the WAL outlives the timer; the timer's stop clears it.
            let woken = Arc::downgrade(&lazy);
            log.set_waker(Some(Arc::new(move || {
                if let Some(lazy) = woken.upgrade() {
                    lazy.wake();
                }
            })));
        }
        // Work already waiting starts the thread at once: a log past the size
        // mark, or, with an interval, changes since the last checkpoint.
        let unflushed = triggers.interval.is_some()
            && job
                .log
                .as_deref()
                .is_some_and(|log| log.end_lsn() > log.checkpoint_lsn());
        let crossed = match (&job.log, mark) {
            (Some(log), Some(mark)) => log.arm_wake(mark),
            _ => false,
        };
        #[cfg(feature = "testing-statement-injection")]
        grafeo_common::testing::commit_hook::run_during_checkpoint_arm();
        if crossed || unflushed {
            lazy.start_locked(phase, false);
        } else {
            drop(phase);
        }
        Self { lazy }
    }

    /// Starts the thread if it still waits for work.
    pub(super) fn start(&self) {
        self.lazy.start();
    }

    /// What starts the thread on the first committed change of a database
    /// without a WAL, for the transaction manager's first-commit hook.
    pub(super) fn starter(&self) -> Arc<dyn Fn() + Send + Sync> {
        let lazy = Arc::downgrade(&self.lazy);
        Arc::new(move || {
            if let Some(lazy) = lazy.upgrade() {
                lazy.start();
            }
        })
    }

    /// Whether the thread was started (and did not end), for tests.
    #[cfg(test)]
    pub(super) fn is_running(&self) -> bool {
        matches!(*self.lazy.phase.lock(), Phase::Running(..))
    }

    /// Stops the timer: a waiting one clears the WAL's waker, a running thread
    /// is signaled and joined (at once, unless a checkpoint is running).
    pub(super) fn stop(&mut self) {
        self.lazy.signal.shut_down();
        let phase = std::mem::replace(&mut *self.lazy.phase.lock(), Phase::Stopped);
        match phase {
            Phase::Waiting(job, _) => {
                if let Some(log) = &job.log {
                    log.set_waker(None);
                }
            }
            // A thread that panicked has stopped as well.
            Phase::Running(handle) => {
                let _ = handle.join();
            }
            Phase::Stopped => {}
        }
    }
}

#[cfg(feature = "grafeo-file")]
impl Drop for CheckpointTimer {
    fn drop(&mut self) {
        self.stop();
    }
}

/// One run of the thread.
#[cfg(feature = "grafeo-file")]
struct Run<'run> {
    signal: &'run Signal,
    triggers: CheckpointTriggers,
    job: &'run Job,
    shared: &'run Shared,
}

/// Where the thread stands between two checkpoints.
#[cfg(feature = "grafeo-file")]
struct State {
    /// The armed mark of the size trigger (`u64::MAX` without one).
    mark: u64,
    /// When the interval trigger fires next.
    interval_at: Option<Instant>,
    /// The pause after the last failed checkpoint, for the interval trigger.
    pause: Option<Duration>,
}

#[cfg(feature = "grafeo-file")]
impl Run<'_> {
    /// The WAL the size trigger watches, with its size: `None` when it is
    /// off or the database has no WAL.
    fn size_trigger(&self) -> Option<(&dyn WatchedLog, u64)> {
        Some((self.job.log.as_deref()?, self.triggers.wal_bytes?))
    }

    /// Arms the size trigger's mark `lsn`: whether the log is already past
    /// it, so the next checkpoint runs without a wait.
    fn arm(&self, state: &mut State, lsn: u64) -> bool {
        match self.size_trigger() {
            Some((log, _)) => {
                state.mark = lsn;
                log.arm_wake(lsn)
            }
            None => false,
        }
    }

    /// Parks until a commit takes the WAL past the mark or the interval
    /// passes: `false` when the thread is to exit.
    fn wait(&self, state: &State) -> bool {
        let mut wake = self.signal.wake.lock();
        loop {
            if wake.shutdown {
                return false;
            }
            if wake.signaled {
                wake.signaled = false;
                return true;
            }
            match state.interval_at {
                Some(at) if Instant::now() >= at => return true,
                Some(at) => {
                    self.signal.condvar.wait_until(&mut wake, at);
                }
                None => self.signal.condvar.wait(&mut wake),
            }
        }
    }

    /// When the interval next fires after `now`, at least `pause` from it:
    /// `None` without an interval, or for one past what an `Instant` holds
    /// (it never fires).
    fn next_interval(&self, now: Instant, pause: Duration) -> Option<Instant> {
        self.triggers
            .interval
            .and_then(|interval| now.checked_add(interval.max(pause)))
    }

    /// Whether a trigger is due: a signal can come from a mark armed before
    /// the last checkpoint, which the log has not reached since.
    fn due(&self, state: &State) -> bool {
        state.interval_at.is_some_and(|at| Instant::now() >= at)
            || self
                .size_trigger()
                .is_some_and(|(log, _)| log.end_lsn() >= state.mark)
    }

    fn run(&self) {
        let mut state = State {
            mark: u64::MAX,
            interval_at: self.next_interval(Instant::now(), Duration::ZERO),
            pause: None,
        };
        // A WAL already past the mark (one an open replayed and kept) is
        // checkpointed without waiting for a commit.
        let mut pending = match self.size_trigger() {
            Some((log, limit)) => {
                let mark = log.checkpoint_lsn().saturating_add(limit);
                self.arm(&mut state, mark)
            }
            None => false,
        };

        loop {
            if !pending {
                if !self.wait(&state) {
                    return;
                }
                self.shared.counts.wakes.fetch_add(1, Ordering::AcqRel);
                if !self.due(&state) {
                    continue;
                }
                // An interval tick with no change to write writes nothing.
                if self.job.changed.as_ref().is_some_and(|changed| !changed()) {
                    state.interval_at = self.next_interval(Instant::now(), Duration::ZERO);
                    continue;
                }
            }

            self.shared.counts.attempts.fetch_add(1, Ordering::AcqRel);
            let result = (self.job.checkpoint)();
            let now = Instant::now();
            let next_mark = match result {
                Ok(()) => {
                    self.shared.failures.succeeded();
                    state.pause = None;
                    state.interval_at = self.next_interval(now, Duration::ZERO);
                    self.size_trigger()
                        .map(|(log, limit)| log.checkpoint_lsn().saturating_add(limit))
                }
                Err(error) => {
                    // After a commit that did not complete, no checkpoint can
                    // ever succeed (it would write the commit's stamped part),
                    // and the database takes no more writes: say so once and
                    // stop.
                    if (self.job.hopeless)() {
                        grafeo_common::grafeo_error!(
                            "automatic checkpoints stop: {error}; the file keeps its last checkpoint"
                        );
                        self.shared.failures.failed(error.to_string(), true);
                        return;
                    }
                    let pause = state
                        .pause
                        .map_or(FIRST_RETRY, |pause| (pause * 2).min(LAST_RETRY));
                    state.pause = Some(pause);
                    state.interval_at = self.next_interval(now, pause);
                    let retry = match (self.size_trigger().is_some(), state.interval_at.is_some()) {
                        (true, true) => format!(
                            "tried again once the WAL grows by one more segment, or in {}s",
                            pause.as_secs()
                        ),
                        (true, false) => {
                            "tried again once the WAL grows by one more segment".to_string()
                        }
                        (false, _) => format!("tried again in {}s", pause.as_secs()),
                    };
                    grafeo_common::grafeo_warn!("automatic checkpoint failed, {retry}: {error}");
                    self.shared.failures.failed(error.to_string(), false);
                    self.size_trigger()
                        .map(|(log, _)| log.end_lsn().saturating_add(self.job.segment_bytes))
                }
            };

            #[cfg(feature = "testing-crash-injection")]
            self.shared.gate.wait_open(self.signal);
            pending = next_mark.is_some_and(|mark| self.arm(&mut state, mark));
        }
    }
}

#[cfg(test)]
#[cfg(feature = "grafeo-file")]
mod tests {
    use super::*;
    use grafeo_common::utils::error::Error;

    /// A WAL that grows when told to, and calls its waker as the WAL does:
    /// once for the first growth past the armed mark.
    #[derive(Default)]
    struct FakeLog {
        state: Mutex<FakeState>,
    }

    #[derive(Default)]
    struct FakeState {
        end: u64,
        checkpoint: u64,
        mark: Option<u64>,
        waker: Option<Arc<dyn Fn() + Send + Sync>>,
    }

    impl FakeLog {
        fn grow(&self, bytes: u64) {
            let waker = {
                let mut state = self.state.lock();
                state.end += bytes;
                match state.mark {
                    Some(mark) if state.end >= mark => {
                        state.mark = None;
                        state.waker.clone()
                    }
                    _ => None,
                }
            };
            if let Some(waker) = waker {
                waker();
            }
        }

        /// A checkpoint: the log is covered to its end.
        fn checkpointed(&self) {
            let mut state = self.state.lock();
            state.checkpoint = state.end;
        }

        fn end(&self) -> u64 {
            self.state.lock().end
        }
    }

    impl WatchedLog for Arc<FakeLog> {
        fn checkpoint_lsn(&self) -> u64 {
            self.state.lock().checkpoint
        }

        fn end_lsn(&self) -> u64 {
            self.end()
        }

        fn arm_wake(&self, lsn: u64) -> bool {
            let mut state = self.state.lock();
            let past = state.end >= lsn;
            state.mark = (!past).then_some(lsn);
            past
        }

        fn set_waker(&self, waker: Option<Arc<dyn Fn() + Send + Sync>>) {
            let mut state = self.state.lock();
            if waker.is_none() {
                state.mark = None;
            }
            state.waker = waker;
        }
    }

    const LIMIT: u64 = 1024;
    const SEGMENT: u64 = 512;

    /// A job over `log` that counts its checkpoints and fails while `failing`
    /// says so.
    fn job(log: &Arc<FakeLog>, failing: &Arc<Mutex<bool>>, hopeless: bool) -> Job {
        let (checkpointed, failing) = (Arc::clone(log), Arc::clone(failing));
        Job {
            checkpoint: Box::new(move || {
                if *failing.lock() {
                    return Err(Error::Internal("injected failure".to_string()));
                }
                checkpointed.checkpointed();
                Ok(())
            }),
            hopeless: Box::new(move || hopeless),
            log: Some(Box::new(Arc::clone(log))),
            segment_bytes: SEGMENT,
            changed: None,
        }
    }

    fn shared() -> Shared {
        Shared {
            failures: Arc::default(),
            counts: Arc::default(),
            #[cfg(feature = "testing-crash-injection")]
            gate: Arc::default(),
        }
    }

    fn attempts(shared: &Shared) -> u64 {
        shared.counts.attempts.load(Ordering::SeqCst)
    }

    fn wakes(shared: &Shared) -> u64 {
        shared.counts.wakes.load(Ordering::SeqCst)
    }

    /// Starts a timer over `log` and returns it with what it shares.
    fn start(
        triggers: CheckpointTriggers,
        log: &Arc<FakeLog>,
        failing: &Arc<Mutex<bool>>,
        hopeless: bool,
    ) -> (CheckpointTimer, Arc<Shared>) {
        let shared = Arc::new(shared());
        let thread_shared = Shared {
            failures: Arc::clone(&shared.failures),
            counts: Arc::clone(&shared.counts),
            #[cfg(feature = "testing-crash-injection")]
            gate: Arc::clone(&shared.gate),
        };
        let timer = CheckpointTimer::arm(triggers, job(log, failing, hopeless), thread_shared);
        (timer, shared)
    }

    /// Waits up to ten seconds (a loaded machine runs the thread late).
    fn wait_until(what: &str, condition: impl Fn() -> bool) {
        let deadline = Instant::now() + Duration::from_secs(10);
        while !condition() {
            assert!(Instant::now() < deadline, "never: {what}");
            std::thread::sleep(Duration::from_millis(5));
        }
    }

    #[test]
    fn the_timer_stops_at_once_with_the_thread_parked() {
        let (log, failing) = (Arc::default(), Arc::default());
        for triggers in [
            CheckpointTriggers::none().with_interval(Duration::from_hours(1)),
            CheckpointTriggers::none().with_wal_bytes(LIMIT),
        ] {
            let (mut timer, shared) = start(triggers, &log, &failing, false);
            std::thread::sleep(Duration::from_millis(50));
            let started = Instant::now();
            timer.stop();
            assert!(
                started.elapsed() < Duration::from_secs(1),
                "stop() took {:?}",
                started.elapsed()
            );
            assert_eq!((wakes(&shared), attempts(&shared)), (0, 0));
        }
    }

    /// A timer armed below its mark starts no thread; the crossing starts it,
    /// counting one wake, and a log past the mark when armed starts it at
    /// once, counting none.
    #[test]
    fn the_thread_starts_the_first_time_it_has_work() {
        let (log, failing): (Arc<FakeLog>, _) = (Arc::default(), Arc::default());
        let (mut timer, shared) = start(
            CheckpointTriggers::none().with_wal_bytes(LIMIT),
            &log,
            &failing,
            false,
        );
        assert!(!timer.is_running(), "no work, no thread");
        log.grow(LIMIT);
        wait_until("the checkpoint", || attempts(&shared) == 1);
        assert!(timer.is_running());
        assert_eq!(wakes(&shared), 1, "the start is the crossing's wake");
        timer.stop();

        let past: Arc<FakeLog> = Arc::default();
        past.grow(2 * LIMIT);
        let (mut timer, shared) = start(
            CheckpointTriggers::none().with_wal_bytes(LIMIT),
            &past,
            &failing,
            false,
        );
        wait_until("the checkpoint at the arm", || attempts(&shared) == 1);
        assert_eq!(wakes(&shared), 0);
        timer.stop();
    }

    /// The commit hook's starter starts a waiting timer once; a stopped timer
    /// stays stopped.
    #[test]
    fn the_starter_starts_once_and_never_after_a_stop() {
        let (log, failing): (Arc<FakeLog>, _) = (Arc::default(), Arc::default());
        let (mut timer, shared) = start(
            CheckpointTriggers::none().with_interval(Duration::from_millis(50)),
            &log,
            &failing,
            false,
        );
        let starter = timer.starter();
        starter();
        starter();
        wait_until("an interval checkpoint", || attempts(&shared) >= 1);
        timer.stop();
        let after = attempts(&shared);
        starter();
        std::thread::sleep(Duration::from_millis(200));
        assert_eq!(attempts(&shared), after, "stopped");
    }

    #[test]
    fn an_interval_checkpoints_each_time_it_passes() {
        let (log, failing) = (Arc::default(), Arc::default());
        let (mut timer, shared) = start(
            CheckpointTriggers::none().with_interval(Duration::from_millis(100)),
            &log,
            &failing,
            false,
        );
        std::thread::sleep(Duration::from_millis(250));
        assert_eq!(attempts(&shared), 0, "no thread before the first commit");
        // The first committed change starts the interval.
        log.grow(1);
        wait_until("two checkpoints", || attempts(&shared) >= 2);
        timer.stop();
    }

    /// Growth below the mark wakes nothing; the growth that crosses it wakes
    /// the thread once, which checkpoints and arms the next mark.
    #[test]
    fn the_growth_past_the_mark_wakes_the_thread_once() {
        let (log, failing): (Arc<FakeLog>, _) = (Arc::default(), Arc::default());
        let (mut timer, shared) = start(
            CheckpointTriggers::none().with_wal_bytes(LIMIT),
            &log,
            &failing,
            false,
        );
        log.grow(LIMIT - 1);
        std::thread::sleep(Duration::from_millis(100));
        assert_eq!((wakes(&shared), attempts(&shared)), (0, 0), "below");
        log.grow(1);
        wait_until("the checkpoint", || attempts(&shared) == 1);
        assert_eq!(wakes(&shared), 1);
        log.grow(LIMIT - 1);
        std::thread::sleep(Duration::from_millis(100));
        assert_eq!(attempts(&shared), 1, "below the next mark");
        timer.stop();
    }

    /// A log past the mark when the thread starts is checkpointed without a
    /// wake.
    #[test]
    fn a_log_past_the_mark_at_start_is_checkpointed_without_a_wake() {
        let (log, failing): (Arc<FakeLog>, _) = (Arc::default(), Arc::default());
        log.grow(2 * LIMIT);
        let (mut timer, shared) = start(
            CheckpointTriggers::none().with_wal_bytes(LIMIT),
            &log,
            &failing,
            false,
        );
        wait_until("the checkpoint", || attempts(&shared) == 1);
        assert_eq!(wakes(&shared), 0);
        timer.stop();
    }

    /// A failed size checkpoint is counted and tried again only once the log
    /// has grown one more segment past the failure; a success clears the
    /// count.
    #[test]
    fn a_failed_size_checkpoint_waits_for_one_more_segment() {
        let (log, failing): (Arc<FakeLog>, Arc<Mutex<bool>>) =
            (Arc::default(), Arc::new(Mutex::new(true)));
        let (mut timer, shared) = start(
            CheckpointTriggers::none().with_wal_bytes(LIMIT),
            &log,
            &failing,
            false,
        );
        log.grow(LIMIT);
        wait_until("the failure", || shared.failures.snapshot().failures == 1);
        let failed = shared.failures.snapshot();
        assert_eq!(
            failed.last_error.as_deref(),
            Some("GRAFEO-X001: Internal error: injected failure")
        );
        assert!(failed.last_failure.is_some() && !failed.stopped);
        std::thread::sleep(Duration::from_millis(1200));
        assert_eq!(attempts(&shared), 1, "no try while nothing is written");
        log.grow(SEGMENT - 1);
        std::thread::sleep(Duration::from_millis(100));
        assert_eq!(attempts(&shared), 1, "less than a segment");

        *failing.lock() = false;
        log.grow(1);
        wait_until("the retry", || attempts(&shared) == 2);
        wait_until("the count cleared", || {
            shared.failures.snapshot().failures == 0
        });
        assert_eq!(
            shared.failures.snapshot().last_error.as_deref(),
            Some("GRAFEO-X001: Internal error: injected failure"),
            "the last error stays for the caller to read"
        );
        timer.stop();
    }

    /// A failed interval checkpoint is tried again after a pause, not at
    /// every interval.
    #[test]
    fn a_failed_interval_checkpoint_backs_off_in_time() {
        let (log, failing): (Arc<FakeLog>, Arc<Mutex<bool>>) =
            (Arc::default(), Arc::new(Mutex::new(true)));
        let (mut timer, shared) = start(
            CheckpointTriggers::none().with_interval(Duration::from_millis(50)),
            &log,
            &failing,
            false,
        );
        log.grow(1);
        wait_until("the failure", || attempts(&shared) == 1);
        std::thread::sleep(Duration::from_millis(500));
        assert_eq!(attempts(&shared), 1, "no try before the pause");
        wait_until("the try after the pause", || attempts(&shared) == 2);
        timer.stop();
    }

    /// When no checkpoint can succeed any more, the thread says so and stops,
    /// and the log calls no waker after it.
    #[test]
    fn a_hopeless_failure_stops_the_checkpoints() {
        let (log, failing): (Arc<FakeLog>, Arc<Mutex<bool>>) =
            (Arc::default(), Arc::new(Mutex::new(true)));
        let (mut timer, shared) = start(
            CheckpointTriggers::none().with_wal_bytes(LIMIT),
            &log,
            &failing,
            true,
        );
        log.grow(LIMIT);
        wait_until("the stop", || shared.failures.snapshot().stopped);
        wait_until("the waker cleared", || log.state.lock().waker.is_none());
        log.grow(4 * LIMIT);
        std::thread::sleep(Duration::from_millis(200));
        assert_eq!(attempts(&shared), 1, "never tried again");
        timer.stop();
    }
}
