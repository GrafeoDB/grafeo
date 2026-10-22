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
//! taken, and the storage layer takes its locks through [`take_lock`], which
//! waits while a child starts. Outside tests nothing starts children this
//! way, so the guard is never contended and a held lock fails at once.
//!
//! A child that runs longer than [`DEADLINE`] (or the deadline given to
//! [`run_with_deadline`] or [`output_with_deadline`]) is killed and the call
//! fails, so a hung child fails its test instead of blocking the run.

use std::io::{self, Read};
use std::process::{Child, Command, ExitStatus, Output, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{PoisonError, RwLock};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

/// How long [`run`] and [`output`] let a child run before they kill it.
pub const DEADLINE: Duration = Duration::from_secs(120);

/// How often a waiting call checks whether its child has exited.
const POLL_INTERVAL: Duration = Duration::from_millis(5);

/// Held for writing while a child starts, for reading while a lock is taken.
static CHILD_START: RwLock<()> = RwLock::new(());

/// Whether this process has started a child with [`run`] or [`output`].
static CHILDREN_STARTED: AtomicBool = AtomicBool::new(false);

/// How long [`take_lock`] tries a held lock again in a process that starts
/// children.
const RETRY_WINDOW: Duration = Duration::from_millis(250);

/// Starts `command` and waits for it, like [`Command::status`], for at most
/// [`DEADLINE`].
///
/// # Errors
///
/// Returns the error of starting the child or of waiting for it, and an
/// error of kind [`io::ErrorKind::TimedOut`] naming the command when the
/// child ran past the deadline (it is killed).
pub fn run(command: &mut Command) -> io::Result<ExitStatus> {
    run_with_deadline(command, DEADLINE)
}

/// [`run`] with its own deadline, for a child that needs longer (or, in a
/// test of the deadline, less).
///
/// # Errors
///
/// As [`run`].
pub fn run_with_deadline(command: &mut Command, deadline: Duration) -> io::Result<ExitStatus> {
    let mut child = start(command)?;
    wait_until(&mut child, command, deadline)
}

/// Starts `command` with its output captured and waits for it, like
/// [`Command::output`], for at most [`DEADLINE`].
///
/// # Errors
///
/// As [`run`], and the error of reading the output of the child.
pub fn output(command: &mut Command) -> io::Result<Output> {
    output_with_deadline(command, DEADLINE)
}

/// [`output`] with its own deadline.
///
/// # Errors
///
/// As [`output`].
pub fn output_with_deadline(command: &mut Command, deadline: Duration) -> io::Result<Output> {
    command
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    let mut child = start(command)?;
    // Read both pipes while waiting: a child that fills a pipe nobody reads
    // blocks, and would only end at the deadline.
    let stdout = read_to_end(child.stdout.take());
    let stderr = read_to_end(child.stderr.take());
    let status = wait_until(&mut child, command, deadline)?;
    Ok(Output {
        status,
        stdout: joined(stdout)?,
        stderr: joined(stderr)?,
    })
}

/// Starts `command` while no database lock is being taken.
fn start(command: &mut Command) -> io::Result<Child> {
    let _starting = CHILD_START.write().unwrap_or_else(PoisonError::into_inner);
    // `spawn` reports a failed exec, so it returns only once the child runs
    // its own program, which closes the copies (Rust opens files
    // close-on-exec), give or take the moment `take_lock` allows for.
    let child = command.spawn()?;
    CHILDREN_STARTED.store(true, Ordering::Relaxed);
    Ok(child)
}

/// Waits for `child`, started from `command`, for at most `deadline`, and
/// kills it once the deadline has passed.
fn wait_until(child: &mut Child, command: &Command, deadline: Duration) -> io::Result<ExitStatus> {
    let until = Instant::now() + deadline;
    loop {
        if let Some(status) = child.try_wait()? {
            return Ok(status);
        }
        if Instant::now() >= until {
            if let Err(error) = child.kill() {
                // It exited after the last check: its status stands.
                return child.try_wait()?.ok_or(error);
            }
            child.wait()?;
            return Err(io::Error::new(
                io::ErrorKind::TimedOut,
                format!(
                    "the child process {command:?} ran longer than {deadline:?} and was killed"
                ),
            ));
        }
        std::thread::sleep(POLL_INTERVAL);
    }
}

/// Reads `pipe` to its end on another thread.
fn read_to_end(pipe: Option<impl Read + Send + 'static>) -> JoinHandle<io::Result<Vec<u8>>> {
    std::thread::spawn(move || {
        let mut bytes = Vec::new();
        if let Some(mut pipe) = pipe {
            pipe.read_to_end(&mut bytes)?;
        }
        Ok(bytes)
    })
}

/// The bytes a [`read_to_end`] thread read.
fn joined(reader: JoinHandle<io::Result<Vec<u8>>>) -> io::Result<Vec<u8>> {
    reader
        .join()
        .map_err(|_| io::Error::other("the thread reading the output of the child panicked"))?
}

/// Takes a database lock with `try_lock` while no child starts, so no
/// starting child holds a copy of the file being locked.
///
/// On Linux the kernel lets the parent of a starting child go on just before
/// it closes the child's close-on-exec files, so for a moment after [`run`]
/// or [`output`] started a child, that child can still hold the lock of a
/// database a test has just closed. In a process that started children this
/// way, a lock that `try_lock` finds held (an error `held` accepts) is
/// therefore tried again for up to 250 ms before the error stands. Other
/// errors, and every error in other processes, are returned at once.
///
/// # Errors
///
/// Returns the last error of `try_lock`.
pub fn take_lock<T, E>(
    mut try_lock: impl FnMut() -> Result<T, E>,
    held: impl Fn(&E) -> bool,
) -> Result<T, E> {
    let _no_child_start = CHILD_START.read().unwrap_or_else(PoisonError::into_inner);
    let deadline = CHILDREN_STARTED
        .load(Ordering::Relaxed)
        .then(|| Instant::now() + RETRY_WINDOW);
    loop {
        match (try_lock(), deadline) {
            (Err(error), Some(deadline)) if held(&error) && Instant::now() < deadline => {
                std::thread::sleep(Duration::from_millis(2));
            }
            (result, _) => return result,
        }
    }
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

    /// Tells [`sleeping_child`] to sleep.
    const SLEEP_VAR: &str = "GRAFEO_CHILD_PROCESS_SLEEP";

    /// Child-process entry for [`a_child_past_its_deadline_is_killed`]: sleeps
    /// far past the deadline; a no-op when run directly.
    #[test]
    fn sleeping_child() {
        if std::env::var_os(SLEEP_VAR).is_some() {
            std::thread::sleep(Duration::from_secs(60));
        }
    }

    fn sleeper() -> Command {
        let mut command = Command::new(std::env::current_exe().unwrap());
        command
            .args([
                "--exact",
                "testing::child_process::tests::sleeping_child",
                "--nocapture",
            ])
            .env(SLEEP_VAR, "1")
            .stdout(Stdio::null())
            .stderr(Stdio::null());
        command
    }

    /// A child that runs past its deadline is killed, and the call fails with
    /// an error naming the test the child runs, long before the child would
    /// have ended.
    #[test]
    #[cfg_attr(miri, ignore = "Miri cannot start child processes")]
    fn a_child_past_its_deadline_is_killed() {
        let short = Duration::from_millis(300);
        let started = Instant::now();
        let error = run_with_deadline(&mut sleeper(), short).unwrap_err();
        assert_eq!(error.kind(), io::ErrorKind::TimedOut, "{error}");
        assert!(error.to_string().contains("sleeping_child"), "{error}");
        assert!(
            started.elapsed() < Duration::from_secs(30),
            "run waited {:?}",
            started.elapsed()
        );

        let started = Instant::now();
        let error = output_with_deadline(&mut sleeper(), short).unwrap_err();
        assert_eq!(error.kind(), io::ErrorKind::TimedOut, "{error}");
        assert!(error.to_string().contains("sleeping_child"), "{error}");
        assert!(
            started.elapsed() < Duration::from_secs(30),
            "output waited {:?}",
            started.elapsed()
        );
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

    /// A lock released a moment after the first try, as by a starting child
    /// that still holds a copy of it, is taken once it is free; a lock that
    /// stays held fails after the window.
    #[test]
    #[cfg_attr(miri, ignore = "Miri cannot start child processes")]
    fn a_lock_released_a_moment_later_is_taken() {
        assert!(run(&mut list_tests()).unwrap().success());
        let path = std::env::temp_dir().join(format!("grafeo-take-lock-{}", std::process::id()));
        let open = || {
            std::fs::OpenOptions::new()
                .read(true)
                .write(true)
                .create(true)
                .truncate(false)
                .open(&path)
                .unwrap()
        };

        let held = open();
        held.lock().unwrap();
        let release = std::thread::spawn(move || {
            std::thread::sleep(Duration::from_millis(20));
            drop(held);
        });
        let held =
            |error: &std::fs::TryLockError| matches!(error, std::fs::TryLockError::WouldBlock);
        let ours = open();
        assert!(take_lock(|| ours.try_lock(), held).is_ok());
        release.join().unwrap();

        let started = Instant::now();
        assert!(take_lock(|| open().try_lock(), held).is_err());
        assert!(started.elapsed() >= RETRY_WINDOW);

        // An error other than a held lock is returned at once.
        let started = Instant::now();
        assert_eq!(
            take_lock(|| Err::<(), _>("broken"), |_| false),
            Err("broken")
        );
        assert!(started.elapsed() < RETRY_WINDOW);
        drop(ours);
        std::fs::remove_file(&path).unwrap();
    }
}
