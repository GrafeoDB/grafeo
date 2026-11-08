//! The deadline of the running query, which the long loops of the operators
//! check.
//!
//! The executor checks the query timeout between the chunks a query returns,
//! but an operator that reads its whole input before it returns a chunk (an
//! aggregate, a sort, the build side of a join, a filter no row passes), or
//! that makes many rows of a few (an `UNWIND`, a cross product), can run far
//! past the deadline inside one call. Such loops call [`check`] once per chunk
//! they read or make, so a query stops within about one chunk of work past its
//! deadline, whatever the shape of its plan.
//!
//! The deadline belongs to the thread that runs the operators: the executor
//! sets it with [`enter`] for as long as it pulls from the plan, so no
//! operator needs a handle to it, and a worker thread takes it over with
//! [`current`] and [`enter`].
//!
//! A loop that counts its own work, such as a graph algorithm's, checks it
//! with a [`Ticker`]: a copy of the deadline and a counter, which reads the
//! clock on the first unit of work and every [`CLOCK_EVERY`]th, so the cost
//! of a unit is one increment. A worker thread gets its ticker from the
//! [`Handoff`] the query's thread took before it split the work.
//!
//! ```
//! use grafeo_common::clock::Instant;
//! use grafeo_core::execution::deadline;
//!
//! assert!(deadline::check().is_ok());
//! {
//!     let _scope = deadline::enter(Some(Instant::now()));
//!     assert!(deadline::check().is_err());
//! }
//! assert!(deadline::check().is_ok());
//! ```

use std::cell::Cell;

use grafeo_common::clock::Instant;

use super::operators::OperatorError;

thread_local! {
    /// The deadline of the query this thread runs, if it has one.
    static DEADLINE: Cell<Option<Instant>> = const { Cell::new(None) };
}

/// Sets the running query's deadline on this thread until the returned scope
/// drops, which restores the one before.
///
/// A scope inside another keeps the earlier deadline of the two: a statement
/// that runs inside a query (a procedure's) cannot outlive that query.
pub fn enter(deadline: Option<Instant>) -> DeadlineScope {
    let previous = DEADLINE.with(Cell::get);
    let deadline = match (previous, deadline) {
        (Some(outer), Some(inner)) => Some(outer.min(inner)),
        (outer, inner) => outer.or(inner),
    };
    DEADLINE.with(|cell| cell.set(deadline));
    DeadlineScope { previous }
}

/// The running query's deadline on this thread, for a thread that works for
/// it (see [`enter`]).
#[must_use]
pub fn current() -> Option<Instant> {
    DEADLINE.with(Cell::get)
}

/// Fails with [`OperatorError::Timeout`] once the running query's deadline
/// has passed; succeeds when it has none.
///
/// # Errors
///
/// [`OperatorError::Timeout`] when the deadline has passed.
#[inline]
pub fn check() -> Result<(), OperatorError> {
    match DEADLINE.with(Cell::get) {
        Some(deadline) if Instant::now() >= deadline => Err(OperatorError::Timeout),
        _ => Ok(()),
    }
}

/// The deadline [`enter`] set, until it drops.
#[must_use = "the deadline holds only while the scope lives"]
pub struct DeadlineScope {
    /// The deadline before [`enter`], restored on drop.
    previous: Option<Instant>,
}

impl Drop for DeadlineScope {
    fn drop(&mut self) {
        DEADLINE.with(|cell| cell.set(self.previous));
    }
}

/// How many units of work a [`Ticker`] counts per read of the clock.
pub const CLOCK_EVERY: u32 = 1_024;

/// What a worker thread takes over from the thread that runs the query, to build its
/// [`Ticker`]: the deadline, and the clock reads the test probe still allows (see
/// `expire_after_polls`). Taken with [`handoff`] before the work is split.
#[derive(Debug, Clone, Copy)]
pub struct Handoff {
    deadline: Option<Instant>,
    #[cfg(feature = "testing-deadline-probe")]
    polls: Option<u32>,
}

/// This thread's deadline and probe, for a worker thread (see [`Handoff`]).
#[must_use]
pub fn handoff() -> Handoff {
    Handoff {
        deadline: current(),
        #[cfg(feature = "testing-deadline-probe")]
        polls: PROBE.with(Cell::get),
    }
}

/// A counted check of a deadline, for a loop that counts its own units of work (an edge
/// relaxed, a state visited, a merge tried): [`tick`](Self::tick) costs one increment, and reads
/// the clock on the first unit and every [`CLOCK_EVERY`]th. One ticker serves a whole run; a
/// fresh one per pass of a loop would read the clock at every pass.
#[derive(Debug, Clone)]
pub struct Ticker {
    deadline: Option<Instant>,
    /// The units of work counted so far.
    work: u32,
    /// The clock reads left before a test's deadline passes (see `expire_after_polls`).
    #[cfg(any(test, feature = "testing-deadline-probe"))]
    polls_left: Option<u32>,
}

impl Default for Ticker {
    fn default() -> Self {
        Self::new()
    }
}

impl Ticker {
    /// A ticker of this thread's deadline (see [`current`]).
    #[must_use]
    pub fn new() -> Self {
        Self::with_handoff(handoff())
    }

    /// A ticker of `deadline`, given apart from this thread's.
    #[must_use]
    pub fn with_deadline(deadline: Option<Instant>) -> Self {
        Self::with_handoff(Handoff {
            deadline,
            #[cfg(feature = "testing-deadline-probe")]
            polls: PROBE.with(Cell::get),
        })
    }

    /// A worker thread's ticker of the deadline the query's thread handed over.
    #[must_use]
    pub fn with_handoff(handoff: Handoff) -> Self {
        Self {
            deadline: handoff.deadline,
            work: 0,
            #[cfg(feature = "testing-deadline-probe")]
            polls_left: handoff.polls,
            #[cfg(all(test, not(feature = "testing-deadline-probe")))]
            polls_left: None,
        }
    }

    /// The deadline it checks.
    #[must_use]
    pub fn deadline(&self) -> Option<Instant> {
        self.deadline
    }

    /// Counts a unit of work, reading the clock on the first and every [`CLOCK_EVERY`]th.
    ///
    /// # Errors
    ///
    /// [`OperatorError::Timeout`] once the deadline has passed.
    #[inline]
    pub fn tick(&mut self) -> Result<(), OperatorError> {
        let reads_the_clock = self.work.is_multiple_of(CLOCK_EVERY);
        self.work = self.work.wrapping_add(1);
        if reads_the_clock {
            self.check()
        } else {
            Ok(())
        }
    }

    /// Reads the clock now, for a wait or a retry, which may take any time.
    ///
    /// # Errors
    ///
    /// [`OperatorError::Timeout`] once the deadline has passed.
    pub fn check(&mut self) -> Result<(), OperatorError> {
        #[cfg(any(test, feature = "testing-deadline-probe"))]
        if let Some(left) = &mut self.polls_left {
            if *left == 0 {
                return Err(OperatorError::Timeout);
            }
            *left -= 1;
        }
        match self.deadline {
            Some(deadline) if Instant::now() >= deadline => Err(OperatorError::Timeout),
            _ => Ok(()),
        }
    }

    /// Lets the deadline pass after `polls` reads of the clock, whether or not there is one: a
    /// test's deadline.
    #[cfg(any(test, feature = "testing-deadline-probe"))]
    pub fn expire_after_polls(&mut self, polls: u32) {
        self.polls_left = Some(polls);
    }
}

#[cfg(feature = "testing-deadline-probe")]
thread_local! {
    /// The clock reads a test's deadline allows each ticker made on this thread.
    static PROBE: Cell<Option<u32>> = const { Cell::new(None) };
}

/// Until the returned scope drops, every [`Ticker`] made on this thread, and every one a worker
/// builds from a [`Handoff`] taken meanwhile, counts its deadline as passed after `polls` reads
/// of the clock, whether or not a deadline is set: 0 fails at the first unit of work, 1 after
/// [`CLOCK_EVERY`] units. Each ticker counts its own reads. A scope inside another replaces it
/// until it drops. A test feature, in no default and no profile.
#[cfg(feature = "testing-deadline-probe")]
pub fn expire_after_polls(polls: u32) -> ProbeScope {
    let previous = PROBE.with(|cell| cell.replace(Some(polls)));
    ProbeScope { previous }
}

/// The probe [`expire_after_polls`] set, until it drops.
#[cfg(feature = "testing-deadline-probe")]
#[must_use = "the probe holds only while the scope lives"]
pub struct ProbeScope {
    /// The probe before, restored on drop.
    previous: Option<u32>,
}

#[cfg(feature = "testing-deadline-probe")]
impl Drop for ProbeScope {
    fn drop(&mut self) {
        PROBE.with(|cell| cell.set(self.previous));
    }
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use super::*;

    #[test]
    fn no_deadline_never_times_out() {
        assert!(current().is_none());
        assert!(check().is_ok());
        let _scope = enter(None);
        assert!(check().is_ok());
    }

    #[test]
    fn a_passed_deadline_times_out_until_its_scope_drops() {
        {
            let _scope = enter(Some(Instant::now()));
            assert!(matches!(check(), Err(OperatorError::Timeout)));
        }
        assert!(current().is_none());
        assert!(check().is_ok());
    }

    #[test]
    fn a_future_deadline_does_not_time_out() {
        let _scope = enter(Some(Instant::now() + Duration::from_secs(3600)));
        assert!(check().is_ok());
    }

    /// An inner scope keeps the earlier deadline, and its drop restores the
    /// outer one.
    #[test]
    fn nested_scopes_keep_the_earlier_deadline() {
        let soon = Instant::now() + Duration::from_secs(60);
        let later = soon + Duration::from_secs(60);
        let _outer = enter(Some(soon));
        {
            let _inner = enter(Some(later));
            assert_eq!(current(), Some(soon));
        }
        {
            let _inner = enter(None);
            assert_eq!(current(), Some(soon));
        }
        let earlier = Instant::now();
        {
            let _inner = enter(Some(earlier));
            assert_eq!(current(), Some(earlier));
            assert!(check().is_err());
        }
        assert_eq!(current(), Some(soon));
    }

    /// Another thread has its own deadline, until it takes over this one.
    #[test]
    fn a_worker_thread_takes_the_deadline_over() {
        let _scope = enter(Some(Instant::now()));
        let deadline = current();
        std::thread::scope(|threads| {
            threads.spawn(|| assert!(check().is_ok()));
            threads.spawn(move || {
                let _scope = enter(deadline);
                assert!(check().is_err());
            });
        });
    }

    /// A ticker reads the clock on its first unit of work and then on every [`CLOCK_EVERY`]th:
    /// a deadline that passes after one read lets the first [`CLOCK_EVERY`] units run and stops
    /// the next; after none it stops the first.
    #[test]
    fn a_ticker_reads_the_clock_every_clock_every_units() {
        let mut ticker = Ticker::with_deadline(None);
        ticker.expire_after_polls(1);
        for unit in 1..=CLOCK_EVERY {
            assert!(ticker.tick().is_ok(), "unit {unit}");
        }
        assert!(matches!(ticker.tick(), Err(OperatorError::Timeout)));

        let mut ticker = Ticker::with_deadline(None);
        ticker.expire_after_polls(0);
        assert!(matches!(ticker.tick(), Err(OperatorError::Timeout)));

        // `check` reads the clock each time, and counts as a read.
        let mut ticker = Ticker::with_deadline(None);
        ticker.expire_after_polls(2);
        assert!(ticker.check().is_ok());
        assert!(ticker.check().is_ok());
        assert!(ticker.check().is_err());
    }

    /// A real deadline stops a ticker at its next read of the clock, and none never does.
    #[test]
    fn a_ticker_meets_a_real_deadline_at_its_next_read() {
        let mut passed = Ticker::with_deadline(Some(Instant::now()));
        assert!(matches!(passed.tick(), Err(OperatorError::Timeout)));
        let mut unbounded = Ticker::with_deadline(None);
        for _ in 0..3 * CLOCK_EVERY {
            assert!(unbounded.tick().is_ok());
        }
        let mut future = Ticker::with_deadline(Some(Instant::now() + Duration::from_secs(3600)));
        for _ in 0..3 * CLOCK_EVERY {
            assert!(future.tick().is_ok());
        }
        // A ticker of this thread's deadline takes the one `enter` set.
        let _scope = enter(Some(Instant::now()));
        assert!(Ticker::new().tick().is_err());
    }

    /// The probe reaches every ticker made under it, with or without a deadline, and a worker's
    /// through the handoff; a scope inside another replaces it until it drops.
    #[cfg(feature = "testing-deadline-probe")]
    #[test]
    fn the_probe_reaches_every_ticker_and_its_workers() {
        assert!(Ticker::new().tick().is_ok());
        {
            let _probe = expire_after_polls(0);
            assert!(Ticker::new().tick().is_err());
            assert!(Ticker::with_deadline(None).tick().is_err());
            let handoff = handoff();
            std::thread::scope(|threads| {
                threads.spawn(move || {
                    assert!(Ticker::with_handoff(handoff).tick().is_err());
                    // A worker's own thread holds no probe.
                    assert!(Ticker::new().tick().is_ok());
                });
            });
            {
                let _inner = expire_after_polls(1);
                let mut ticker = Ticker::new();
                for _ in 0..CLOCK_EVERY {
                    assert!(ticker.tick().is_ok());
                }
                assert!(ticker.tick().is_err());
            }
            // The outer probe again.
            assert!(Ticker::new().tick().is_err());
        }
        assert!(Ticker::new().tick().is_ok());
    }
}
