//! Child processes for tests, started so they never hold a database lock.
//!
//! Crash tests run part of a test in a child process that exits without
//! closing its database, and lock tests open a database from a second
//! process. On Unix a starting child holds a copy of every file the test
//! process has open, including the lock files of databases that other tests
//! hold, until it runs its own program. A test that closes a database and
//! opens it again at that moment finds it "locked by another process".
//!
//! [`run`] and [`output`] start a child only while no database lock is being
//! taken, and the storage layer takes its locks inside [`lock_acquisition`],
//! which waits while a child starts. Outside tests nothing starts children
//! this way, so the guard is never contended.

use std::io;
use std::process::{Command, ExitStatus, Output, Stdio};
use std::sync::{PoisonError, RwLock, RwLockReadGuard};

/// Held for writing while a child starts, for reading while a lock is taken.
static CHILD_START: RwLock<()> = RwLock::new(());

/// Starts `command` and waits for it, like [`Command::status`].
///
/// # Errors
///
/// Returns the error of starting the child or of waiting for it.
pub fn run(command: &mut Command) -> io::Result<ExitStatus> {
    let mut child = {
        let _starting = CHILD_START.write().unwrap_or_else(PoisonError::into_inner);
        // `spawn` reports a failed exec, so it returns only once the child
        // runs its own program, which closes the copies (Rust opens files
        // close-on-exec).
        command.spawn()?
    };
    child.wait()
}

/// Starts `command` with its output captured and waits for it, like
/// [`Command::output`].
///
/// # Errors
///
/// Returns the error of starting the child or of waiting for it.
pub fn output(command: &mut Command) -> io::Result<Output> {
    let child = {
        let _starting = CHILD_START.write().unwrap_or_else(PoisonError::into_inner);
        command
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()?
    };
    child.wait_with_output()
}

/// Holds off [`run`] and [`output`] while the caller takes a database lock,
/// so no starting child holds a copy of the file being locked.
pub fn lock_acquisition() -> RwLockReadGuard<'static, ()> {
    CHILD_START.read().unwrap_or_else(PoisonError::into_inner)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn list_tests() -> Command {
        let mut command = Command::new(std::env::current_exe().unwrap());
        command
            .arg("--list")
            .stdout(Stdio::null())
            .stderr(Stdio::null());
        command
    }

    #[test]
    #[cfg_attr(miri, ignore = "Miri cannot start child processes")]
    fn run_returns_the_exit_status() {
        assert!(run(&mut list_tests()).unwrap().success());
        let failing = run(list_tests().arg("--no-such-option")).unwrap();
        assert!(!failing.success());
    }

    #[test]
    #[cfg_attr(miri, ignore = "Miri cannot start child processes")]
    fn output_captures_stdout() {
        let output = output(&mut list_tests()).unwrap();
        assert!(output.status.success());
        let stdout = String::from_utf8_lossy(&output.stdout);
        assert!(
            stdout.contains("child_process::tests::output_captures_stdout"),
            "{stdout}"
        );
    }
}
