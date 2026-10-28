//! Runs a test's code inside a commit or a checkpoint, to check what other
//! threads can do meanwhile: right after a commit's epoch is assigned
//! ([`after_next_commit_epoch`]), once its WAL group is written and the
//! commit lock released but before it is synced and published
//! ([`after_next_commit_written`]), once its versions are stamped but before
//! its events are reported ([`after_next_commit_stamped`]), once its events
//! are reported but before the commit is complete
//! ([`after_next_commit_logged`]), inside a checkpoint while it holds
//! commits off, before it writes its image ([`during_next_checkpoint`]),
//! inside a checkpoint once it let commits go on again and wrote its first
//! row group's chunks, while it writes the rest
//! ([`during_next_checkpoint_write`]), or
//! inside a change outside any commit (a schema statement, a graph command, an
//! RDF update, a direct write, a direct graph or index call, an import or an
//! RDF batch insert) once it holds commits off and before it changes anything
//! ([`during_next_held_change`]), or after a backend write has been applied
//! and recorded ([`after_next_data_write`]). [`fail_next_change_record`]
//! makes the recording of a direct write's change fail.
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
    type WriteHook =
        Box<dyn FnOnce() -> Result<(), crate::testing::statement_failure::InjectedFailure>>;

    thread_local! {
        static AFTER_EPOCH: RefCell<Option<Hook>> = const { RefCell::new(None) };
        static AFTER_DATA_WRITE: RefCell<Option<WriteHook>> = const { RefCell::new(None) };
        static FAIL_RECORD: RefCell<bool> = const { RefCell::new(false) };
        static AFTER_WRITTEN: RefCell<Option<Hook>> = const { RefCell::new(None) };
        static AFTER_STAMPED: RefCell<Option<Hook>> = const { RefCell::new(None) };
        static AFTER_LOGGED: RefCell<Option<Hook>> = const { RefCell::new(None) };
        static AFTER_BACKUP_STATE_READ: RefCell<Option<Hook>> = const { RefCell::new(None) };
        static BEFORE_BACKUP_WAIT: RefCell<Option<Hook>> = const { RefCell::new(None) };
        static DURING_CHECKPOINT: RefCell<Option<Hook>> = const { RefCell::new(None) };
        static DURING_CHECKPOINT_WRITE: RefCell<Option<Hook>> = const { RefCell::new(None) };
        static DURING_HELD_CHANGE: RefCell<Option<Hook>> = const { RefCell::new(None) };
    }

    /// Arms a one-shot hook after the next changed backend write is recorded
    /// on this thread. Returning an error or panicking exercises the caller's
    /// cleanup after a completed write, before the rest of its operation.
    pub fn after_next_data_write(
        hook: impl FnOnce() -> Result<(), crate::testing::statement_failure::InjectedFailure> + 'static,
    ) {
        AFTER_DATA_WRITE.with(|slot| *slot.borrow_mut() = Some(Box::new(hook)));
    }

    /// Runs the data-write hook armed on this thread, if any.
    ///
    /// # Errors
    ///
    /// Returns the armed hook's injected failure.
    #[inline]
    pub fn run_after_data_write() -> Result<(), crate::testing::statement_failure::InjectedFailure>
    {
        match AFTER_DATA_WRITE.with(|slot| slot.borrow_mut().take()) {
            Some(hook) => hook(),
            None => Ok(()),
        }
    }

    /// Makes the next change a direct write records on this thread fail to
    /// be recorded, after the write is applied: the failure that poisons the
    /// database, since the log would miss the write.
    pub fn fail_next_change_record() {
        FAIL_RECORD.with(|slot| *slot.borrow_mut() = true);
    }

    /// Whether a recording of a change on this thread was armed to fail
    /// ([`fail_next_change_record`]), once.
    #[inline]
    #[must_use]
    pub fn run_change_record() -> bool {
        FAIL_RECORD.with(|slot| std::mem::take(&mut *slot.borrow_mut()))
    }

    /// The checkpoints started on each database file.
    static CHECKPOINTS: Mutex<Vec<(PathBuf, usize)>> = Mutex::new(Vec::new());

    /// Arms `hook` to run once, inside the next commit on this thread, right
    /// after its commit epoch is assigned.
    pub fn after_next_commit_epoch(hook: impl FnOnce() + 'static) {
        AFTER_EPOCH.with(|slot| *slot.borrow_mut() = Some(Box::new(hook)));
    }

    /// Arms `hook` to run once, inside the next commit on this thread, once
    /// its WAL group is written and the commit lock released, before the
    /// commit is synced and published.
    pub fn after_next_commit_written(hook: impl FnOnce() + 'static) {
        AFTER_WRITTEN.with(|slot| *slot.borrow_mut() = Some(Box::new(hook)));
    }

    /// Runs the [`after_next_commit_written`] hook armed on this thread, if
    /// any. Called by the engine's commit.
    #[inline]
    pub fn run_after_commit_written() {
        if let Some(hook) = AFTER_WRITTEN.with(|slot| slot.borrow_mut().take()) {
            hook();
        }
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

    /// Arms `hook` to run once after the next backup on this thread reads
    /// the state and selects the file it will write, before that write.
    pub fn after_next_backup_state_read(hook: impl FnOnce() + 'static) {
        AFTER_BACKUP_STATE_READ.with(|slot| *slot.borrow_mut() = Some(Box::new(hook)));
    }

    /// Runs the backup state-read hook armed on this thread, if any.
    #[inline]
    pub fn run_after_backup_state_read() {
        if let Some(hook) = AFTER_BACKUP_STATE_READ.with(|slot| slot.borrow_mut().take()) {
            hook();
        }
    }

    /// Arms `hook` to run once before the next backup on this thread waits
    /// for another backup on the same database to finish.
    pub fn before_next_backup_wait(hook: impl FnOnce() + 'static) {
        BEFORE_BACKUP_WAIT.with(|slot| *slot.borrow_mut() = Some(Box::new(hook)));
    }

    /// Runs the backup wait hook after contention, before waiting.
    #[inline]
    pub fn run_before_backup_wait() {
        if let Some(hook) = BEFORE_BACKUP_WAIT.with(|slot| slot.borrow_mut().take()) {
            hook();
        }
    }

    /// Arms `hook` to run once, inside the next checkpoint on this thread,
    /// while it holds commits off and before it writes its image.
    pub fn during_next_checkpoint(hook: impl FnOnce() + 'static) {
        DURING_CHECKPOINT.with(|slot| *slot.borrow_mut() = Some(Box::new(hook)));
    }

    /// Arms `hook` to run once, inside the next checkpoint on this thread,
    /// once it let commits go on again (its snapshot is taken) and wrote the
    /// chunks of its first row group, before it writes the others.
    pub fn during_next_checkpoint_write(hook: impl FnOnce() + 'static) {
        DURING_CHECKPOINT_WRITE.with(|slot| *slot.borrow_mut() = Some(Box::new(hook)));
    }

    /// Runs the [`during_next_checkpoint_write`] hook armed on this thread,
    /// if any. Called by the LPG section's writer after it handed a row
    /// group's chunks to the file, holding no lock of the store.
    #[inline]
    pub fn run_during_checkpoint_write() {
        if let Some(hook) = DURING_CHECKPOINT_WRITE.with(|slot| slot.borrow_mut().take()) {
            hook();
        }
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

    /// Arms `hook` to run once, inside the next change outside any commit on
    /// this thread (a schema statement, a graph command, an RDF update, a
    /// direct write, a direct graph or index call, an import or an RDF batch
    /// insert), once it holds commits off and before it changes anything.
    pub fn during_next_held_change(hook: impl FnOnce() + 'static) {
        DURING_HELD_CHANGE.with(|slot| *slot.borrow_mut() = Some(Box::new(hook)));
    }

    /// Runs the [`during_next_held_change`] hook armed on this thread, if
    /// any. Called by the engine once such a change holds commits off.
    #[inline]
    pub fn run_during_held_change() {
        if let Some(hook) = DURING_HELD_CHANGE.with(|slot| slot.borrow_mut().take()) {
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
    pub fn after_next_data_write(
        _hook: impl FnOnce() -> Result<(), crate::testing::statement_failure::InjectedFailure> + 'static,
    ) {
    }

    /// No-op when injection is disabled.
    ///
    /// # Errors
    ///
    /// Always succeeds without the test feature.
    #[inline]
    pub fn run_after_data_write() -> Result<(), crate::testing::statement_failure::InjectedFailure>
    {
        Ok(())
    }

    /// No-op when injection is disabled.
    pub fn fail_next_change_record() {}

    /// Always `false` when injection is disabled.
    #[inline]
    #[must_use]
    pub fn run_change_record() -> bool {
        false
    }

    /// No-op when injection is disabled.
    pub fn after_next_commit_epoch(_hook: impl FnOnce() + 'static) {}

    /// No-op when injection is disabled.
    #[inline]
    pub fn run_after_commit_epoch() {}

    /// No-op when injection is disabled.
    pub fn after_next_commit_written(_hook: impl FnOnce() + 'static) {}

    /// No-op when injection is disabled.
    #[inline]
    pub fn run_after_commit_written() {}

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
    pub fn after_next_backup_state_read(_hook: impl FnOnce() + 'static) {}

    /// No-op when injection is disabled.
    #[inline]
    pub fn run_after_backup_state_read() {}

    /// No-op when injection is disabled.
    pub fn before_next_backup_wait(_hook: impl FnOnce() + 'static) {}

    /// No-op when injection is disabled.
    #[inline]
    pub fn run_before_backup_wait() {}

    /// No-op when injection is disabled.
    pub fn during_next_checkpoint(_hook: impl FnOnce() + 'static) {}

    /// No-op when injection is disabled.
    pub fn during_next_held_change(_hook: impl FnOnce() + 'static) {}

    /// No-op when injection is disabled.
    pub fn during_next_checkpoint_write(_hook: impl FnOnce() + 'static) {}

    /// No-op when injection is disabled.
    #[inline]
    pub fn run_during_checkpoint_write() {}

    /// No-op when injection is disabled.
    #[inline]
    pub fn run_during_held_change() {}

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
