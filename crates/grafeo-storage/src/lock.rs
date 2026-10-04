//! Exclusive lock for WAL-directory databases.
//!
//! A `.grafeo` file locks itself. A WAL-directory database has no single file
//! to lock, so [`DirectoryLock`] holds an exclusive lock on a `LOCK` file in
//! the database directory for as long as the database is open for writing.
//! A second open, from this process or another one, fails instead of later
//! overwriting the first one's data.

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
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(&path)?;

        child_process::take_lock(
            || file.try_lock(),
            |e| matches!(e, std::fs::TryLockError::WouldBlock),
        )
        .map_err(|e| match e {
            std::fs::TryLockError::WouldBlock => Error::Internal(format!(
                "database is locked by another process: {}",
                dir.display()
            )),
            std::fs::TryLockError::Error(e) => Error::Io(e),
        })?;

        Ok(Self { _file: file, path })
    }

    /// Path of the lock file.
    #[must_use]
    pub fn path(&self) -> &Path {
        &self.path
    }
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
    fn stale_lock_file_does_not_block() {
        // A LOCK file left behind by a crashed process holds no OS lock.
        let dir = tempfile::tempdir().unwrap();
        fs::write(dir.path().join(LOCK_FILE_NAME), b"").unwrap();
        DirectoryLock::acquire(dir.path()).unwrap();
    }
}
