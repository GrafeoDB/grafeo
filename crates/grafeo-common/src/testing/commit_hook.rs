//! Runs a test's code inside a commit or a checkpoint, to check what other
//! threads can do meanwhile: right after a commit's epoch is assigned
//! ([`after_next_commit_epoch`]), once its versions are stamped but before
//! its events and WAL records are written ([`after_next_commit_stamped`]),
//! once its WAL records are written but before the commit is complete
//! ([`after_next_commit_logged`]), or inside a checkpoint while it holds
//! commits off, before it writes its image ([`during_next_checkpoint`]).
//! [`checkpoints_started`] counts the checkpoints started on a database
//! file, from any thread (the engine calls [`count_checkpoint`]).
//!
//! With the `testing-statement-injection` feature, the `run_*` functions run
//! the hook armed on the calling thread, once. Hooks are thread-local, so only
//! a commit or checkpoint on the thread that armed one runs it. Without the
//! feature all functions are no-ops (and nothing is counted).

#[cfg(feature = "testing-statement-injection")]
mod inner {
    use std::cell::RefCell;
    use std::path::{Path, PathBuf};
    use std::sync::Mutex;

    type Hook = Box<dyn FnOnce()>;

    thread_local! {
        static AFTER_EPOCH: RefCell<Option<Hook>> = const { RefCell::new(None) };
        static AFTER_STAMPED: RefCell<Option<Hook>> = const { RefCell::new(None) };
        static AFTER_LOGGED: RefCell<Option<Hook>> = const { RefCell::new(None) };
        static DURING_CHECKPOINT: RefCell<Option<Hook>> = const { RefCell::new(None) };
    }

    /// The checkpoints started on each database file.
    static CHECKPOINTS: Mutex<Vec<(PathBuf, usize)>> = Mutex::new(Vec::new());

    /// Arms `hook` to run once, inside the next commit on this thread, right
    /// after its commit epoch is assigned.
    pub fn after_next_commit_epoch(hook: impl FnOnce() + 'static) {
        AFTER_EPOCH.with(|slot| *slot.borrow_mut() = Some(Box::new(hook)));
    }

    /// Arms `hook` to run once, inside the next commit on this thread, once
    /// its versions are stamped and before the commit is complete.
    pub fn after_next_commit_stamped(hook: impl FnOnce() + 'static) {
        AFTER_STAMPED.with(|slot| *slot.borrow_mut() = Some(Box::new(hook)));
    }

    /// Arms `hook` to run once, inside the next commit on this thread, once
    /// its events and WAL records are written and before the commit is
    /// complete.
    pub fn after_next_commit_logged(hook: impl FnOnce() + 'static) {
        AFTER_LOGGED.with(|slot| *slot.borrow_mut() = Some(Box::new(hook)));
    }

    /// Arms `hook` to run once, inside the next checkpoint on this thread,
    /// while it holds commits off and before it writes its image.
    pub fn during_next_checkpoint(hook: impl FnOnce() + 'static) {
        DURING_CHECKPOINT.with(|slot| *slot.borrow_mut() = Some(Box::new(hook)));
    }

    /// Runs the [`after_next_commit_epoch`] hook armed on this thread, if
    /// any. Called by the engine's commit.
    #[inline]
    pub fn run_after_commit_epoch() {
        if let Some(hook) = AFTER_EPOCH.with(|slot| slot.borrow_mut().take()) {
            hook();
        }
    }

    /// Runs the [`after_next_commit_stamped`] hook armed on this thread, if
    /// any. Called by the engine's commit.
    #[inline]
    pub fn run_after_commit_stamped() {
        if let Some(hook) = AFTER_STAMPED.with(|slot| slot.borrow_mut().take()) {
            hook();
        }
    }

    /// Runs the [`after_next_commit_logged`] hook armed on this thread, if
    /// any. Called by the engine's commit.
    #[inline]
    pub fn run_after_commit_logged() {
        if let Some(hook) = AFTER_LOGGED.with(|slot| slot.borrow_mut().take()) {
            hook();
        }
    }

    /// Counts a checkpoint started on the database file at `path` (see
    /// [`checkpoints_started`]). Called by the engine's checkpoint first.
    pub fn count_checkpoint(path: &Path) {
        let mut started = CHECKPOINTS.lock().unwrap_or_else(|e| e.into_inner());
        match started.iter_mut().find(|(file, _)| file == path) {
            Some((_, count)) => *count += 1,
            None => started.push((path.to_path_buf(), 1)),
        }
    }

    /// Runs the [`during_next_checkpoint`] hook armed on this thread, if
    /// any. Called by the engine's checkpoint once it holds commits off.
    #[inline]
    pub fn run_during_checkpoint() {
        if let Some(hook) = DURING_CHECKPOINT.with(|slot| slot.borrow_mut().take()) {
            hook();
        }
    }

    /// The checkpoints started on the database file at `path`, from any
    /// thread, also those that failed.
    #[must_use]
    pub fn checkpoints_started(path: &Path) -> usize {
        CHECKPOINTS
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .iter()
            .find(|(file, _)| file == path)
            .map_or(0, |(_, count)| *count)
    }
}

#[cfg(not(feature = "testing-statement-injection"))]
mod inner {
    use std::path::Path;

    /// No-op when injection is disabled.
    pub fn after_next_commit_epoch(_hook: impl FnOnce() + 'static) {}

    /// No-op when injection is disabled.
    #[inline]
    pub fn run_after_commit_epoch() {}

    /// No-op when injection is disabled.
    pub fn after_next_commit_stamped(_hook: impl FnOnce() + 'static) {}

    /// No-op when injection is disabled.
    #[inline]
    pub fn run_after_commit_stamped() {}

    /// No-op when injection is disabled.
    pub fn after_next_commit_logged(_hook: impl FnOnce() + 'static) {}

    /// No-op when injection is disabled.
    #[inline]
    pub fn run_after_commit_logged() {}

    /// No-op when injection is disabled.
    pub fn during_next_checkpoint(_hook: impl FnOnce() + 'static) {}

    /// No-op when injection is disabled.
    pub fn count_checkpoint(_path: &Path) {}

    /// No-op when injection is disabled.
    #[inline]
    pub fn run_during_checkpoint() {}

    /// Always 0 when injection is disabled.
    #[must_use]
    pub fn checkpoints_started(_path: &Path) -> usize {
        0
    }
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
