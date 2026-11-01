//! Exclusive lock for 0.5.x WAL-directory databases.
//!
//! A `.grafeo` file locks itself. A 0.5.x WAL-directory database has no single
//! file to lock, so 0.5.44 held an exclusive lock on a `LOCK` file in the
//! database directory for as long as the database was open for writing, and
//! [`DirectoryLock`] takes that same lock: 0.6 holds it while it reads such a
//! directory to migrate it to a single file, until right before the directory
//! moves, so that read and a 0.5.44 writer exclude each other. 0.6 never opens
//! a directory for writing.

use std::fs::{self, File, OpenOptions};
use std::path::{Path, PathBuf};

use grafeo_common::testing::child_process;
use grafeo_common::utils::error::{Error, Result};

/// Name of the lock file inside the database directory.
pub const LOCK_FILE_NAME: &str = "LOCK";

/// An exclusive lock on a database directory, released on drop.
#[derive(Debug)]
pub struct DirectoryLock {
    /// Held open for the lifetime of the lock; the OS lock lives on it.
    _file: File,
    path: PathBuf,
}

impl DirectoryLock {
    /// Creates `dir` if needed and locks it exclusively.
    ///
    /// # Errors
    ///
    /// Returns an error if another handle holds the lock, or if the directory
    /// or lock file cannot be created.
    pub fn acquire(dir: impl AsRef<Path>) -> Result<Self> {
        let dir = dir.as_ref();
        fs::create_dir_all(dir)?;
        let path = dir.join(LOCK_FILE_NAME);
        let file = try_lock_file(&path, || {
            format!("database is locked by another process: {}", dir.display())
        })?;
        Ok(Self { _file: file, path })
    }

    /// Path of the lock file.
    #[must_use]
    pub fn path(&self) -> &Path {
        &self.path
    }
}

/// Locks the file at `path` exclusively, without waiting, and creates it
/// empty if it is not there. The lock lives on the returned handle and ends
/// when it is dropped (or the process exits); the file itself stays, so the
/// next caller locks the same file.
///
/// # Errors
///
/// [`Error::Internal`] with the message `busy` gives when another handle
/// holds the lock, [`Error::Io`] when the file cannot be opened (a missing
/// directory: [`std::io::ErrorKind::NotFound`]) or locked.
pub fn try_lock_file(path: &Path, busy: impl FnOnce() -> String) -> Result<File> {
    let file = OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .truncate(false)
        .open(path)?;
    child_process::take_lock(
        || file.try_lock(),
        |e| matches!(e, std::fs::TryLockError::WouldBlock),
    )
    .map_err(|e| match e {
        std::fs::TryLockError::WouldBlock => Error::Internal(busy()),
        std::fs::TryLockError::Error(e) => Error::Io(e),
    })?;
    Ok(file)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn second_acquire_fails_until_release() {
        let dir = tempfile::tempdir().unwrap();
        let db_dir = dir.path().join("db");

        let first = DirectoryLock::acquire(&db_dir).unwrap();
        assert_eq!(first.path(), db_dir.join(LOCK_FILE_NAME));

        let err = DirectoryLock::acquire(&db_dir).unwrap_err();
        assert!(err.to_string().contains("locked"), "{err}");

        drop(first);
        DirectoryLock::acquire(&db_dir).unwrap();
    }

    #[test]
    fn a_locked_file_is_busy_until_its_handle_drops_and_stays() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("busy.lock");

        let first = try_lock_file(&path, || "busy".to_string()).unwrap();
        let err = try_lock_file(&path, || "busy".to_string()).unwrap_err();
        assert!(
            matches!(&err, Error::Internal(message) if message == "busy"),
            "{err}"
        );

        drop(first);
        assert!(path.exists(), "the lock file stays for the next caller");
        try_lock_file(&path, || "busy".to_string()).unwrap();

        let missing = dir.path().join("missing").join("busy.lock");
        let err = try_lock_file(&missing, || "busy".to_string()).unwrap_err();
        assert!(
            matches!(&err, Error::Io(e) if e.kind() == std::io::ErrorKind::NotFound),
            "{err}"
        );
    }

    #[test]
    fn stale_lock_file_does_not_block() {
        // A LOCK file left behind by a crashed process holds no OS lock.
        let dir = tempfile::tempdir().unwrap();
        fs::write(dir.path().join(LOCK_FILE_NAME), b"").unwrap();
        DirectoryLock::acquire(dir.path()).unwrap();
    }
}
