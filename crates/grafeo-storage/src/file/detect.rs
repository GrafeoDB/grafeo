//! Detecting what a database path holds, and the side-file names of a migration.
//!
//! Opening a path must tell a container v3 file from a file written by 0.5.x
//! and from a 0.5.x WAL-directory database. [`detect`] reads at most eight
//! bytes and never writes or locks anything.

use std::fs::{self, File};
use std::io::Read;
use std::path::{Path, PathBuf};

use grafeo_common::utils::error::{Error, Result};

use super::format::MAGIC;

/// What lives at a database path.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OnDisk {
    /// Nothing exists at the path.
    Missing,
    /// A container v3 file (0.6.x).
    Current,
    /// A single file written by 0.5.x (container v1 or v2).
    LegacyFile,
    /// A 0.5.x WAL-directory database (a directory holding `wal/`).
    WalDirectory,
    /// Something else: another version, a foreign file, or an unrelated directory.
    Unknown,
}

/// Classifies `path` without writing or locking anything.
///
/// Byte 4 of a file is `0x01` for a 0.5.x bincode header and `3` (followed by
/// three zero bytes) for container v3. A directory holding `wal/` is a WAL
/// directory. Anything that is neither a regular file nor a directory (a
/// FIFO, a socket, a device) is [`OnDisk::Unknown`] without being opened.
///
/// # Errors
///
/// Returns an error for I/O failures other than "not found". On Windows a
/// file another process holds open as a database cannot be read (its lock
/// covers the bytes): that error says the database file is locked by another
/// process, as the file manager's does.
pub fn detect(path: &Path) -> Result<OnDisk> {
    let io_error = |source: std::io::Error| {
        if is_lock_violation(&source) {
            return Error::Internal(format!(
                "database file is locked by another process: {}",
                path.display()
            ));
        }
        Error::Internal(format!(
            "cannot inspect database path {}: {source}",
            path.display()
        ))
    };
    let metadata = match fs::metadata(path) {
        Ok(metadata) => metadata,
        Err(source) if source.kind() == std::io::ErrorKind::NotFound => {
            return Ok(OnDisk::Missing);
        }
        Err(source) => return Err(io_error(source)),
    };
    if metadata.is_dir() {
        return match fs::metadata(path.join("wal")) {
            Ok(wal) if wal.is_dir() => Ok(OnDisk::WalDirectory),
            Ok(_) => Ok(OnDisk::Unknown),
            Err(source) if source.kind() == std::io::ErrorKind::NotFound => Ok(OnDisk::Unknown),
            Err(source) => Err(io_error(source)),
        };
    }
    // Anything else that is not a regular file (a FIFO, a socket, a device)
    // is never opened: opening a FIFO blocks until a writer comes.
    if !metadata.is_file() {
        return Ok(OnDisk::Unknown);
    }

    let mut prefix = [0u8; 8];
    let mut read = 0;
    let mut file = File::open(path).map_err(io_error)?;
    while read < prefix.len() {
        match file.read(&mut prefix[read..]) {
            Ok(0) => break,
            Ok(count) => read += count,
            Err(source) if source.kind() == std::io::ErrorKind::Interrupted => {}
            Err(source) => return Err(io_error(source)),
        }
    }
    if read < prefix.len() || prefix[..4] != MAGIC {
        return Ok(OnDisk::Unknown);
    }
    Ok(match prefix[4..8] {
        [3, 0, 0, 0] => OnDisk::Current,
        [1, ..] => OnDisk::LegacyFile,
        _ => OnDisk::Unknown,
    })
}

/// Whether `error` is a read refused because another handle locked the bytes:
/// `ERROR_LOCK_VIOLATION` (33) on Windows, whose file locks are mandatory.
/// Unix locks are advisory and never fail a read.
fn is_lock_violation(error: &std::io::Error) -> bool {
    /// `ERROR_LOCK_VIOLATION`: another process has locked a portion of the file.
    const ERROR_LOCK_VIOLATION: i32 = 33;
    cfg!(windows) && error.raw_os_error() == Some(ERROR_LOCK_VIOLATION)
}

fn with_suffix(path: &Path, suffix: &str) -> PathBuf {
    let mut name = path.as_os_str().to_owned();
    name.push(suffix);
    PathBuf::from(name)
}

/// The kept copy of the old database: `<path>.pre-0.6`.
#[must_use]
pub fn pre_06_path(path: &Path) -> PathBuf {
    with_suffix(path, ".pre-0.6")
}

/// The in-progress migrated image: `<path>.migrating`.
#[must_use]
pub fn migrating_path(path: &Path) -> PathBuf {
    with_suffix(path, ".migrating")
}

/// The migration lock file: `<path>.migrate.lock`.
#[must_use]
pub fn migrate_lock_path(path: &Path) -> PathBuf {
    with_suffix(path, ".migrate.lock")
}

/// The sidecar WAL directory of a database file: `<path>.wal`.
#[must_use]
pub fn sidecar_wal_path(path: &Path) -> PathBuf {
    with_suffix(path, ".wal")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::file::v3::header::FileHeaderV3;

    fn fixture(version: &str, name: &str) -> PathBuf {
        PathBuf::from(format!(
            "{}/../grafeo-engine/tests/fixtures/released/{version}/{name}",
            env!("CARGO_MANIFEST_DIR")
        ))
    }

    #[test]
    fn released_files_are_legacy_files() {
        for version in ["0.5.43", "0.5.44"] {
            assert_eq!(
                detect(&fixture(version, "closed.grafeo")).unwrap(),
                OnDisk::LegacyFile,
                "{version}"
            );
        }
    }

    #[test]
    fn released_directories_are_wal_directories() {
        for version in ["0.5.43", "0.5.44"] {
            assert_eq!(
                detect(&fixture(version, "directory")).unwrap(),
                OnDisk::WalDirectory,
                "{version}"
            );
        }
    }

    #[test]
    fn a_v3_header_is_current() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("amsterdam.grafeo");
        fs::write(&path, FileHeaderV3::new(false).encode()).unwrap();
        assert_eq!(detect(&path).unwrap(), OnDisk::Current);
    }

    #[test]
    fn a_missing_path_is_missing() {
        let dir = tempfile::tempdir().unwrap();
        assert_eq!(detect(&dir.path().join("berlin")).unwrap(), OnDisk::Missing);
    }

    #[test]
    fn foreign_and_short_files_are_unknown() {
        let dir = tempfile::tempdir().unwrap();
        let text = dir.path().join("notes.txt");
        fs::write(&text, "Vincent and Jules went to Paris").unwrap();
        assert_eq!(detect(&text).unwrap(), OnDisk::Unknown);

        let short = dir.path().join("short.grafeo");
        fs::write(&short, b"GRAF\x01").unwrap();
        assert_eq!(detect(&short).unwrap(), OnDisk::Unknown);

        let other_version = dir.path().join("other.grafeo");
        fs::write(&other_version, b"GRAF\x09\x00\x00\x00").unwrap();
        assert_eq!(detect(&other_version).unwrap(), OnDisk::Unknown);

        let empty = dir.path().join("empty.grafeo");
        fs::write(&empty, b"").unwrap();
        assert_eq!(detect(&empty).unwrap(), OnDisk::Unknown);
    }

    /// While a manager holds the database open, Windows refuses reads of its
    /// bytes from other handles (a lock violation, os error 33): `detect`
    /// says the file is locked, as the manager does. Elsewhere the lock does
    /// not block reads, and the file is detected.
    #[test]
    fn a_database_held_open_elsewhere_is_reported_as_locked() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("berlin.grafeo");
        let manager = crate::file::GrafeoFileManager::create(&path, None).unwrap();
        let detected = detect(&path);
        drop(manager);
        if cfg!(windows) {
            let error = detected
                .expect_err("Windows refuses to read the locked bytes")
                .to_string();
            assert!(
                error.contains("database file is locked by another process")
                    && error.contains("berlin.grafeo"),
                "the error says the database is locked, and which: {error}"
            );
        } else {
            assert_eq!(detected.unwrap(), OnDisk::Current);
        }
    }

    /// A FIFO at the path is neither a database file nor a directory: it is
    /// unknown, decided from its metadata, never opened (opening a FIFO blocks
    /// until a writer comes).
    #[cfg(unix)]
    #[test]
    fn a_fifo_is_unknown_and_never_opened() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("paris.grafeo");
        match std::process::Command::new("mkfifo").arg(&path).status() {
            Ok(status) if status.success() => {}
            outcome => {
                eprintln!("skipped: `mkfifo` is not available to create a FIFO ({outcome:?})");
                return;
            }
        }
        let (sender, receiver) = std::sync::mpsc::channel();
        let fifo = path.clone();
        std::thread::spawn(move || sender.send(detect(&fifo).map_err(|e| e.to_string())));
        let detected = receiver
            .recv_timeout(std::time::Duration::from_secs(5))
            .expect("detect returns without waiting for a writer to the FIFO");
        assert_eq!(detected, Ok(OnDisk::Unknown));
    }

    #[test]
    fn directories_need_a_wal_subdirectory() {
        let dir = tempfile::tempdir().unwrap();
        assert_eq!(detect(dir.path()).unwrap(), OnDisk::Unknown);
        fs::create_dir(dir.path().join("wal")).unwrap();
        assert_eq!(detect(dir.path()).unwrap(), OnDisk::WalDirectory);
    }

    #[test]
    fn side_file_names_append_to_the_full_path() {
        let with_extension = Path::new("data/gus.grafeo");
        assert_eq!(
            sidecar_wal_path(with_extension),
            Path::new("data/gus.grafeo.wal")
        );
        assert_eq!(
            pre_06_path(with_extension),
            Path::new("data/gus.grafeo.pre-0.6")
        );
        assert_eq!(
            migrating_path(with_extension),
            Path::new("data/gus.grafeo.migrating")
        );
        assert_eq!(
            migrate_lock_path(with_extension),
            Path::new("data/gus.grafeo.migrate.lock")
        );
        let bare = Path::new("mydb");
        assert_eq!(sidecar_wal_path(bare), Path::new("mydb.wal"));
        assert_eq!(pre_06_path(bare), Path::new("mydb.pre-0.6"));
    }
}
