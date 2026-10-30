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
}
