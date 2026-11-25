//! The cursor of a change data capture consumer (#679): where the consumer
//! reads the durable feed from, as the file `cdc_cursor_<name>.json` in the
//! WAL directory (see [`CdcCursorName`]). `default` is the only consumer
//! today: [`GrafeoDB::set_cdc`](crate::GrafeoDB::set_cdc) writes its cursor
//! at the log position of the checkpoint that stores the setting on (and
//! leaves one that is there alone), and removes it once the setting is
//! stored off. Every checkpoint keeps the WAL from the lowest cursor on,
//! within the retention limit, as it keeps it for the backup cursor
//! (`wal_retention`).

use std::path::{Path, PathBuf};

use grafeo_common::types::EpochId;
use grafeo_common::utils::error::{Error, Result};
use grafeo_storage::wal::{CdcCursorName, is_cdc_cursor_file};

use super::wal_retention::{CursorPosition, WalRetention};

/// Where a consumer stands in the durable feed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub(crate) struct CdcCursor {
    /// The log position the consumer reads from: the groups before it are
    /// acknowledged.
    pub lsn: u64,
    /// The epoch of the last acknowledged group (0 before any).
    pub epoch: EpochId,
}

/// The path of the default consumer's cursor in `wal_dir`.
fn default_path(wal_dir: &Path) -> PathBuf {
    wal_dir.join(CdcCursorName::default().file_name())
}

/// Whether `wal_dir` holds the default consumer's cursor. One that cannot
/// be looked at counts as there: it is never written over.
pub(super) fn default_exists(wal_dir: &Path) -> bool {
    default_path(wal_dir).try_exists().unwrap_or(true)
}

/// The default consumer's cursor in `wal_dir`.
///
/// # Errors
///
/// As [`read`]; a missing cursor is an error too.
pub(super) fn read_default(wal_dir: &Path) -> Result<CdcCursor> {
    read(&default_path(wal_dir))
}

/// Moves the default consumer's cursor to `cursor`: written to a temporary
/// file, synced, then renamed into place and the directory synced, so a
/// crash leaves the old cursor or the new one. The crash points name each
/// step.
///
/// # Errors
///
/// The cursor cannot be written, synced or renamed.
pub(super) fn move_default(wal_dir: &Path, cursor: CdcCursor) -> Result<()> {
    use grafeo_common::testing::crash::{maybe_crash, maybe_fail};
    use std::io::Write as _;

    let data = serde_json::to_vec_pretty(&cursor)
        .map_err(|e| Error::Internal(format!("failed to serialize the CDC cursor: {e}")))?;
    let path = default_path(wal_dir);
    let mut temp = path.as_os_str().to_owned();
    temp.push(".tmp");
    let temp = PathBuf::from(temp);
    let io = |step: &str, shown: &Path, e: std::io::Error| {
        Error::Io(std::io::Error::new(
            e.kind(),
            format!("failed to {step} the CDC cursor {}: {e}", shown.display()),
        ))
    };
    maybe_crash("cdc:before_cursor_write");
    let written = (|| {
        let mut file = std::fs::File::create(&temp).map_err(|e| io("create", &temp, e))?;
        file.write_all(&data).map_err(|e| io("write", &temp, e))?;
        maybe_fail("cdc:cursor_sync")?;
        file.sync_all().map_err(|e| io("sync", &temp, e))?;
        drop(file);
        maybe_crash("cdc:after_cursor_sync");
        std::fs::rename(&temp, &path).map_err(|e| io("finalize", &path, e))
    })();
    if let Err(error) = written {
        let _ = std::fs::remove_file(&temp);
        return Err(error);
    }
    maybe_crash("cdc:after_cursor_rename");
    super::backup::sync_backup_entry(&path, "cdc:cursor_directory_sync")
}

/// The cursor in the file `path`.
///
/// # Errors
///
/// The file cannot be read, or [`Error::Corruption`] naming it when it does
/// not parse.
fn read(path: &Path) -> Result<CdcCursor> {
    let data = std::fs::read(path).map_err(|e| {
        Error::Internal(format!(
            "failed to read the CDC cursor {}: {e}",
            path.display()
        ))
    })?;
    serde_json::from_slice(&data)
        .map_err(|e| Error::corruption(format!("the CDC cursor does not parse: {e}")).in_file(path))
}

/// The positions of the CDC cursors among `entries`, the non-segment entries
/// of a WAL directory, for a checkpoint's retention: an unreadable cursor is
/// held to the limit, with a warning once per open.
pub(super) fn positions(entries: &[PathBuf], retention: &WalRetention) -> Vec<CursorPosition> {
    entries
        .iter()
        .filter(|path| {
            path.file_name()
                .and_then(|name| name.to_str())
                .is_some_and(is_cdc_cursor_file)
        })
        .map(|path| match read(path) {
            Ok(cursor) => CursorPosition::At(cursor.lsn),
            Err(error) => {
                if retention.first_cdc_unreadable() {
                    grafeo_common::grafeo_warn!(
                        "cannot read the CDC cursor {} ({error}): checkpoints keep the WAL for \
                         it up to the retention limit of {} bytes",
                        path.display(),
                        retention.limit
                    );
                }
                CursorPosition::Unreadable
            }
        })
        .collect()
}

/// Whether `wal_dir` holds a CDC cursor.
pub(super) fn any_in(wal_dir: &Path) -> bool {
    // A directory that cannot be listed may hold one: it is kept.
    std::fs::read_dir(wal_dir).map_or(wal_dir.exists(), |entries| {
        entries
            .flatten()
            .any(|entry| entry.file_name().to_str().is_some_and(is_cdc_cursor_file))
    })
}

/// Writes the default consumer's cursor at log position `lsn`, before any
/// group acknowledged: synced, then renamed into place, so a crash leaves
/// the old file or the new one.
///
/// # Errors
///
/// The cursor cannot be written or synced.
pub(super) fn write_default(wal_dir: &Path, lsn: u64) -> Result<()> {
    let cursor = CdcCursor {
        lsn,
        epoch: EpochId::new(0),
    };
    let data = serde_json::to_vec_pretty(&cursor)
        .map_err(|e| Error::Internal(format!("failed to serialize the CDC cursor: {e}")))?;
    super::backup::write_metadata(
        &default_path(wal_dir),
        &data,
        "CDC cursor",
        "cdc:cursor_sync",
        "cdc:cursor_directory_sync",
    )
}

/// Removes the default consumer's cursor, if there is one, and makes the
/// removal durable: the next checkpoint keeps no WAL for it.
///
/// # Errors
///
/// The cursor exists but cannot be removed, or the directory cannot be synced.
pub(super) fn remove_default(wal_dir: &Path) -> Result<()> {
    let path = default_path(wal_dir);
    if !path.exists() {
        return Ok(());
    }
    grafeo_common::testing::crash::maybe_fail("cdc:cursor_remove")?;
    match std::fs::remove_file(&path) {
        Ok(()) => super::backup::sync_backup_entry(&path, "cdc:cursor_remove_sync"),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(error) => Err(Error::Internal(format!(
            "failed to remove the CDC cursor {}: {error}",
            path.display()
        ))),
    }
}

/// Brings the default cursor of `wal_dir` in line with the stored setting
/// at an open: with change data capture `on`, a missing cursor is written at
/// `checkpoint_lsn`, the image's; off, a cursor left behind goes. One that
/// cannot be removed only warns: with the setting off it keeps no log, and
/// a later open or `set_cdc(false)` removes it.
///
/// # Errors
///
/// The cursor cannot be written.
pub(super) fn follow_setting(wal_dir: &Path, on: bool, checkpoint_lsn: u64) -> Result<()> {
    match (on, default_exists(wal_dir)) {
        (true, false) => write_default(wal_dir, checkpoint_lsn),
        (false, true) => {
            if let Err(error) = remove_default(wal_dir) {
                grafeo_common::grafeo_warn!(
                    "change data capture is stored off, but the cursor left in {} could not be \
                     removed ({error}): it keeps no log, and the next open or \
                     set_cdc(false) removes it",
                    wal_dir.display()
                );
            }
            Ok(())
        }
        _ => Ok(()),
    }
}
