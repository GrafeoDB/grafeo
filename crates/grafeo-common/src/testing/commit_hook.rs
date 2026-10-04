//! Runs a test's code inside a commit, right after the commit epoch is
//! assigned and before the commit's versions, events and WAL records are
//! written, to check what other threads can do while a commit completes.
//!
//! With the `testing-statement-injection` feature,
//! [`run_after_commit_epoch`] runs the hook that [`after_next_commit_epoch`]
//! armed on the calling thread, once. The hook is thread-local, so only a
//! commit on the thread that armed it runs it. Without the feature both
//! functions are no-ops.

#[cfg(feature = "testing-statement-injection")]
mod inner {
    use std::cell::RefCell;

    type Hook = Box<dyn FnOnce()>;

    thread_local! {
        static HOOK: RefCell<Option<Hook>> = const { RefCell::new(None) };
    }

    /// Arms `hook` to run once, inside the next commit on this thread, right
    /// after its commit epoch is assigned.
    pub fn after_next_commit_epoch(hook: impl FnOnce() + 'static) {
        HOOK.with(|slot| *slot.borrow_mut() = Some(Box::new(hook)));
    }

    /// Runs the hook armed on this thread, if any. Called by the engine's
    /// commit.
    #[inline]
    pub fn run_after_commit_epoch() {
        if let Some(hook) = HOOK.with(|slot| slot.borrow_mut().take()) {
            hook();
        }
    }
}

#[cfg(not(feature = "testing-statement-injection"))]
mod inner {
    /// No-op when injection is disabled.
    pub fn after_next_commit_epoch(_hook: impl FnOnce() + 'static) {}

    /// No-op when injection is disabled.
    #[inline]
    pub fn run_after_commit_epoch() {}
}

pub use inner::*;

#[cfg(all(test, feature = "testing-statement-injection"))]
mod tests {
    use super::*;
    use std::cell::Cell;
    use std::rc::Rc;

    #[test]
    fn the_hook_runs_once() {
        let runs = Rc::new(Cell::new(0));
        let counter = Rc::clone(&runs);
        after_next_commit_epoch(move || counter.set(counter.get() + 1));
        run_after_commit_epoch();
        run_after_commit_epoch();
        assert_eq!(runs.get(), 1);
    }

    #[test]
    fn the_hook_belongs_to_the_thread_that_armed_it() {
        let ran = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
        let flag = std::sync::Arc::clone(&ran);
        after_next_commit_epoch(move || flag.store(true, std::sync::atomic::Ordering::SeqCst));
        std::thread::spawn(run_after_commit_epoch).join().unwrap();
        assert!(!ran.load(std::sync::atomic::Ordering::SeqCst));
        run_after_commit_epoch();
        assert!(ran.load(std::sync::atomic::Ordering::SeqCst));
    }
}
