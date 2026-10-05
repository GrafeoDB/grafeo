//! Crash injection for testing recovery paths.
//!
//! When the `testing-crash-injection` feature is enabled, [`maybe_crash`]
//! counts down a **thread-local** counter and panics when it reaches zero.
//! Tests use [`with_crash_at`] to run a closure that crashes at a
//! deterministic point, then verify that recovery produces a consistent state.
//!
//! [`maybe_fail`] is the same for errors: placed where a real I/O error could
//! occur, it returns an error at the count [`with_failure_at`] arms, so tests
//! can check how callers handle a failure there.
//!
//! Thread-local storage ensures that concurrent tests never interfere with
//! each other; only the thread that calls [`enable_crash_at`] (or
//! [`with_failure_at`]) is affected.
//!
//! When the feature is **disabled**, all functions compile to no-ops with zero
//! runtime overhead.
//!
//! An injected crash is a panic, and it unwinds: a database the closure owns
//! when it fires is dropped on the way out, and its `Drop` closes it (a
//! checkpoint, then the WAL removed), so what a test inspects afterwards is a
//! clean close, not the crash. A database outside the closure, which it only
//! borrows, stays open. A child process that stands in for a crash must not
//! unwind a database the closure owns: exit inside a panic hook
//! (`std::panic::set_hook` that prints the panic and calls
//! `std::process::exit`), or keep the database outside the closure and exit
//! without dropping it.
//!
//! # Example
//!
//! ```ignore
//! use grafeo_common::testing::crash::{with_crash_at, CrashResult};
//!
//! for point in 1..20 {
//!     let result = with_crash_at(point, || {
//!         // operations that call maybe_crash() internally
//!     });
//!     match result {
//!         CrashResult::Completed(value) => { /* ran to completion */ }
//!         CrashResult::Crashed => { /* verify recovery */ }
//!     }
//! }
//! ```

#[cfg(feature = "testing-crash-injection")]
mod inner {
    use std::cell::Cell;

    use crate::utils::error::{Error, Result};

    thread_local! {
        static CRASH_COUNTER: Cell<u64> = const { Cell::new(u64::MAX) };
        static CRASH_ENABLED: Cell<bool> = const { Cell::new(false) };
    }

    /// Conditionally panic when the crash counter reaches zero.
    ///
    /// Insert this at interesting recovery boundaries (before/after WAL
    /// writes, flushes, checkpoints). When crash injection is disabled,
    /// this compiles to nothing.
    ///
    /// Uses thread-local state so concurrent tests don't interfere.
    ///
    /// # Panics
    ///
    /// Panics (intentionally) when crash injection is enabled and the counter reaches zero.
    #[inline]
    pub fn maybe_crash(point: &'static str) {
        CRASH_ENABLED.with(|enabled| {
            if !enabled.get() {
                return;
            }
            CRASH_COUNTER.with(|counter| {
                let prev = counter.get();
                counter.set(prev.wrapping_sub(1));
                assert!(prev != 1, "crash injection at: {point}");
            });
        });
    }

    /// Enable crash injection to fire after `count` calls to [`maybe_crash`].
    ///
    /// Only affects the calling thread.
    pub fn enable_crash_at(count: u64) {
        CRASH_COUNTER.with(|c| c.set(count));
        CRASH_ENABLED.with(|e| e.set(true));
    }

    /// Disable crash injection (reset to no-op behavior).
    ///
    /// Only affects the calling thread.
    pub fn disable_crash() {
        CRASH_ENABLED.with(|e| e.set(false));
        CRASH_COUNTER.with(|c| c.set(u64::MAX));
    }

    thread_local! {
        static FAILURE_COUNTER: Cell<u64> = const { Cell::new(u64::MAX) };
        static FAILURE_ENABLED: Cell<bool> = const { Cell::new(false) };
    }

    /// Returns an error when the failure counter reaches zero.
    ///
    /// Insert this where a real I/O error could occur, next to
    /// [`maybe_crash`], so tests can make an operation fail there and check
    /// what the caller does with the error. It counts its own calls, not
    /// those of [`maybe_crash`], and fires once per [`with_failure_at`].
    ///
    /// # Errors
    ///
    /// Returns an I/O error naming `point` when injection is armed and the
    /// counter reaches zero.
    #[inline]
    pub fn maybe_fail(point: &str) -> Result<()> {
        if !FAILURE_ENABLED.with(Cell::get) {
            return Ok(());
        }
        let previous = FAILURE_COUNTER.with(|counter| {
            let previous = counter.get();
            counter.set(previous.wrapping_sub(1));
            previous
        });
        if previous == 1 {
            return Err(Error::Io(std::io::Error::other(format!(
                "injected failure at: {point}"
            ))));
        }
        Ok(())
    }

    /// Runs `f` with failure injection armed: the `count`-th call to
    /// [`maybe_fail`] on this thread returns an error. Disarmed when `f`
    /// returns or panics.
    pub fn with_failure_at<T>(count: u64, f: impl FnOnce() -> T) -> T {
        /// Disarms failure injection when dropped, also during a panic.
        struct Disarm;
        impl Drop for Disarm {
            fn drop(&mut self) {
                FAILURE_ENABLED.with(|enabled| enabled.set(false));
                FAILURE_COUNTER.with(|counter| counter.set(u64::MAX));
            }
        }

        FAILURE_COUNTER.with(|counter| counter.set(count));
        FAILURE_ENABLED.with(|enabled| enabled.set(true));
        let _disarm = Disarm;
        f()
    }
}

#[cfg(not(feature = "testing-crash-injection"))]
mod inner {
    use crate::utils::error::Result;

    /// No-op when crash injection is disabled.
    #[inline(always)]
    pub fn maybe_crash(_point: &'static str) {}

    /// No-op when crash injection is disabled.
    pub fn enable_crash_at(_count: u64) {}

    /// No-op when crash injection is disabled.
    pub fn disable_crash() {}

    /// Always `Ok` when crash injection is disabled.
    ///
    /// # Errors
    ///
    /// Never returns an error without the `testing-crash-injection` feature.
    #[inline]
    pub fn maybe_fail(_point: &str) -> Result<()> {
        Ok(())
    }

    /// Runs `f`: without the feature nothing is armed.
    pub fn with_failure_at<T>(_count: u64, f: impl FnOnce() -> T) -> T {
        f()
    }
}

pub use inner::*;

/// Outcome of a crash-injected run.
#[non_exhaustive]
pub enum CrashResult<T> {
    /// The closure completed without crashing.
    Completed(T),
    /// A crash was injected (panic caught).
    Crashed,
}

/// Run `f` with crash injection armed to fire after `crash_after` calls to
/// [`maybe_crash`]. Returns [`CrashResult::Crashed`] if the injected panic
/// was caught, or [`CrashResult::Completed`] with the return value otherwise.
///
/// Crash injection is automatically disabled after the closure returns
/// (whether normally or via panic).
pub fn with_crash_at<F, T>(crash_after: u64, f: F) -> CrashResult<T>
where
    F: FnOnce() -> T + std::panic::UnwindSafe,
{
    enable_crash_at(crash_after);
    let result = std::panic::catch_unwind(f);
    disable_crash();

    match result {
        Ok(value) => CrashResult::Completed(value),
        Err(_) => CrashResult::Crashed,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[cfg(feature = "testing-crash-injection")]
    fn crash_at_exact_count() {
        let result = with_crash_at(3, || {
            maybe_crash("point_1");
            maybe_crash("point_2");
            maybe_crash("point_3"); // should crash here
            42 // should not reach
        });
        assert!(matches!(result, CrashResult::Crashed));
    }

    #[test]
    fn completes_when_count_exceeds_calls() {
        let result = with_crash_at(100, || {
            maybe_crash("a");
            maybe_crash("b");
            42
        });
        match result {
            CrashResult::Completed(v) => assert_eq!(v, 42),
            CrashResult::Crashed => panic!("should not crash"),
        }
    }

    #[test]
    fn disabled_by_default() {
        // Without enabling, maybe_crash is a no-op
        maybe_crash("should_not_crash");
    }

    #[test]
    #[cfg(feature = "testing-crash-injection")]
    fn a_failure_is_injected_once_at_the_armed_count() {
        let outcomes = with_failure_at(2, || {
            [
                maybe_fail("point_1"),
                maybe_fail("point_2"),
                maybe_fail("point_3"),
            ]
        });
        assert!(outcomes[0].is_ok(), "the first call passes");
        let error = outcomes[1].as_ref().unwrap_err().to_string();
        assert!(
            error.contains("injected failure at: point_2"),
            "the error names the point: {error}"
        );
        assert!(outcomes[2].is_ok(), "the failure fires once");
    }

    #[test]
    #[cfg(feature = "testing-crash-injection")]
    fn crash_points_do_not_count_toward_a_failure() {
        let outcome = with_failure_at(1, || {
            maybe_crash("crash_point");
            maybe_fail("failure_point")
        });
        let error = outcome.unwrap_err().to_string();
        assert!(error.contains("failure_point"), "{error}");
    }

    #[test]
    fn no_failure_is_injected_outside_with_failure_at() {
        assert!(maybe_fail("unarmed").is_ok());
        with_failure_at(3, || {});
        assert!(
            maybe_fail("after").is_ok(),
            "disarmed once the closure returns"
        );
    }

    #[test]
    #[cfg(feature = "testing-crash-injection")]
    fn a_panic_in_the_closure_disarms_the_failure() {
        let result = std::panic::catch_unwind(|| with_failure_at(1, || panic!("Vincent")));
        assert!(result.is_err());
        assert!(maybe_fail("after the panic").is_ok());
    }

    #[test]
    #[cfg(not(feature = "testing-crash-injection"))]
    fn without_the_feature_no_failure_is_injected() {
        assert!(with_failure_at(1, || maybe_fail("point")).is_ok());
    }

    #[test]
    fn disable_resets_state() {
        enable_crash_at(2);
        disable_crash();
        // After disable, crash should not fire
        maybe_crash("a");
        maybe_crash("b");
        maybe_crash("c");
    }
}
