//! Who holds the lock of a database file: this process or another one.
//!
//! A database file is locked while a manager has it open (see
//! [`GrafeoFileManager`](crate::file::GrafeoFileManager)). Every manager
//! registers its file here while it lives, so when an open finds the file
//! locked, [`locked_error`] can say whether a handle of this process holds it
//! (close that one first) or another process does.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Mutex, PoisonError};

use grafeo_common::utils::error::{Error, StorageError};

/// The files the managers of this process have open, with how many managers
/// have each one.
static OPEN_HERE: Mutex<Option<HashMap<PathBuf, usize>>> = Mutex::new(None);

/// The key of `path` in [`OPEN_HERE`]: its canonical form, or the path as
/// given when it cannot be resolved.
fn key(path: &Path) -> PathBuf {
    std::fs::canonicalize(path).unwrap_or_else(|_| path.to_path_buf())
}

/// A file a manager of this process has open, registered until it drops.
#[derive(Debug)]
pub(crate) struct OpenHere(PathBuf);

impl OpenHere {
    /// Registers `path` as open in this process.
    pub(crate) fn register(path: &Path) -> Self {
        let key = key(path);
        let mut open = OPEN_HERE.lock().unwrap_or_else(PoisonError::into_inner);
        *open
            .get_or_insert_with(HashMap::new)
            .entry(key.clone())
            .or_insert(0) += 1;
        Self(key)
    }
}

impl Drop for OpenHere {
    fn drop(&mut self) {
        let mut open = OPEN_HERE.lock().unwrap_or_else(PoisonError::into_inner);
        if let Some(open) = open.as_mut()
            && let Some(count) = open.get_mut(&self.0)
        {
            *count -= 1;
            if *count == 0 {
                open.remove(&self.0);
            }
        }
    }
}

/// Whether a manager of this process has `path` open.
fn open_here(path: &Path) -> bool {
    let key = key(path);
    OPEN_HERE
        .lock()
        .unwrap_or_else(PoisonError::into_inner)
        .as_ref()
        .is_some_and(|open| open.contains_key(&key))
}

/// The error of an open that found the database file at `path` locked: a
/// handle of this process holds it, or another process does.
pub(crate) fn locked_error(path: &Path) -> Error {
    let message = if open_here(path) {
        format!(
            "the database at {} is already open in this process: close the other handle first",
            path.display()
        )
    } else {
        format!(
            "the database at {} is held by another process: close it there first",
            path.display()
        )
    };
    Error::from(StorageError::Locked(message))
}
