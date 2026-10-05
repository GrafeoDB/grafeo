//! Migration of database files written by 0.5.x to the 0.6 file format.
//!
//! A read-write open of a 0.5.x file (container v1 or v2) migrates it before
//! the normal open. [`migrate`] does, under an exclusive lock on
//! `<path>.migrate.lock`:
//!
//! 1. Take a shared lock on the old file, which fails while a 0.5.x process
//!    has it open for writing and keeps one from opening it until it has
//!    moved. Open the old database read-only, with the 0.5.x reader (its
//!    sidecar WAL included), write its complete state as a 0.6 image to
//!    `<path>.migrating`, and open the image read-only to check it (its counts
//!    are logged once it is in place).
//! 2. Move the old file to `<path>.pre-0.6`, its sidecar WAL `<path>.wal` to
//!    `<path>.pre-0.6.wal` and a checkpoint image 0.5.44 left pending,
//!    `<path>.checkpoint`, to `<path>.pre-0.6.checkpoint`, syncing the
//!    directory after each move: a power loss keeps the first moves, never a
//!    later one without an earlier one.
//! 3. Rename `<path>.migrating` to `<path>` and sync the directory.
//!
//! The old files are kept byte for byte. To go back to 0.5.x, move the 0.6
//! file `<path>` and its sidecar WAL `<path>.wal` aside (they hold what was
//! written since the migration, and 0.5.x would otherwise replay the 0.6
//! WAL), then rename the kept files back. A migration never
//! replaces a kept copy: while one of the `.pre-0.6` names is taken, it fails
//! before it writes anything.
//!
//! With `Config::encryption` set, the image is encrypted with the keys the key
//! chain derives for its new database id. The kept files stay unencrypted
//! (0.5.x never encrypted them), and a warning names each of them.
//!
//! A 0.5.x sidecar WAL holds changes the file does not. A build without the
//! `wal` feature cannot replay it, so it refuses to read (and so to migrate)
//! a 0.5.x database whose sidecar WAL holds files.
//!
//! ## What a failure leaves
//!
//! - An error in step 1, or in the first rename of step 2, leaves the old
//!   database exactly as it was, and no `<path>.migrating` (an incomplete image
//!   is removed).
//! - An error later in step 2 or in step 3 leaves the old files partly or fully
//!   moved and a complete image: a state the next read-write open finishes, as
//!   after a crash.
//! - A crash leaves one of these states, which [`finish_interrupted`] (or, for
//!   a missing `<path>`, [`lock_if_missing`]) handles on the next read-write
//!   open:
//!
//! | Files present | Next read-write open |
//! | --- | --- |
//! | `<path>` (0.5.x) and `<path>.migrating` | removes the image, which may be incomplete, and migrates again |
//! | `<path>` (0.5.x), `<path>.migrating` and a side file under its kept name (`<path>.pre-0.6.wal` or `<path>.pre-0.6.checkpoint`), no `<path>.pre-0.6` | moves the side file back to its 0.5.x name and syncs, then as above (left by a power loss where the side file's move reached the disk and the database file's did not) |
//! | `<path>.pre-0.6` and `<path>.migrating`, no `<path>` | moves a 0.5.x sidecar WAL or checkpoint image still under the old names to the kept names, syncing after each move, then renames the image to `<path>` |
//! | `<path>.migrating`, no `<path>` and no `<path>.pre-0.6` | fails with an error naming the files: the old database is missing, and nothing is guessed |
//! | `<path>` (0.6) and `<path>.pre-0.6` | nothing to do |
//!
//! A crash also leaves the lock file behind, which the next open removes.
//!
//! Two states are never the result of a migration, and are not acted on: a
//! `<path>.migrating` next to a 0.6 `<path>` is left alone with a warning, and
//! a kept copy without `<path>` and without an image means the migrated file
//! was lost, so every open fails instead of creating an empty database there.
//!
//! A migration in another process holds the lock while `<path>` is missing
//! between its renames. A read-write open therefore decides what to do with
//! the files ([`decide_read_write`]) without the lock only while `<path>`
//! exists: a 0.5.x file is migrated under the lock (which looks at the files
//! again), and a 0.6 file is never moved. Whenever `<path>` turns out missing,
//! the open takes the lock, looks again, and creates a database only while it
//! holds it (the lock is released once the new file, or the directory of a new
//! WAL-directory database, exists).
//!
//! A read-only open (and `open_in_memory`) never migrates and never changes
//! these files: it reads a 0.5.x `<path>` in place, and fails for a state
//! without `<path>` and for a 0.5.x `<path>` whose side file a cut-off
//! migration moved to its kept name ([`check_read_only`]).
//!
//! ## The lock
//!
//! A process takes the lock with a retry every 50 ms for up to five seconds,
//! then fails with "database locked: a migration is running". The holder
//! removes the lock file when it is done and then marks its handle (writes to
//! the removed file), so a process that opened the file before it was removed
//! sees the mark once it gets the lock, and tries again with a new file.

use std::fs::{self, File, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use grafeo_common::testing::child_process;
use grafeo_common::testing::crash::maybe_crash;
use grafeo_common::testing::pause::maybe_pause;
use grafeo_common::utils::error::{Error, Result, StorageError};
use grafeo_common::{grafeo_info, grafeo_warn};
use grafeo_storage::file::GrafeoFileManager;
use grafeo_storage::file::detect::{
    OnDisk, detect, migrate_lock_path, migrating_path, pre_06_path, sidecar_wal_path,
};

use super::GrafeoDB;
use super::encryption::DatabaseKeys;
use crate::config::Config;

/// How long an open waits for a migration another process is running.
const LOCK_WAIT: Duration = Duration::from_secs(5);

/// How often it tries the lock again meanwhile.
const LOCK_RETRY: Duration = Duration::from_millis(50);

/// What the holder writes to the lock file after removing it.
const RELEASED: &[u8] = b"released";

/// The crash point after the image is renamed to the database's name.
const INSTALLED: &str = "migrate:renamed:database";

/// Finishes or undoes a migration a crash cut off, from the files present (the
/// table in the module docs), under the migration lock.
///
/// Does nothing, and takes no lock, when neither `<path>.migrating` nor a lock
/// file exists.
///
/// # Errors
///
/// Returns an error if another process holds the migration lock for longer
/// than the wait, if only `<path>.migrating` is left (the old database is
/// missing), or if a file cannot be inspected, removed or renamed.
pub(super) fn finish_interrupted(path: &Path) -> Result<()> {
    if !exists(&migrating_path(path))? && !exists(&migrate_lock_path(path))? {
        return Ok(());
    }
    let _lock = MigrateLock::acquire(path)?;
    resolve(path)
}

/// Decides what a read-write open finds at `path`: whether it is a single file
/// (`single_file` tells from the path and the configured format), what is on
/// disk, and, when a database is to be created at the missing `path`, the
/// migration lock to hold until it exists.
///
/// First finishes a migration a crash cut off ([`finish_interrupted`]). The
/// files are then looked at without the lock while `path` exists; whenever it
/// turns out missing (a migration in another process may be between its
/// renames), the lock is taken ([`lock_if_missing`]) and the files are looked
/// at again, until `path` exists or the decision is taken holding the lock.
///
/// # Errors
///
/// The errors of [`finish_interrupted`] and [`lock_if_missing`], or a file
/// that cannot be inspected.
pub(super) fn decide_read_write(
    path: &Path,
    single_file: impl Fn(&Path) -> bool,
) -> Result<(bool, OnDisk, Option<MigrateLock>)> {
    finish_interrupted(path)?;
    let mut create_lock = lock_if_missing(path)?;
    maybe_pause("open:after_check");
    loop {
        let is_single_file = single_file(path);
        let on_disk = detect(path)?;
        if on_disk != OnDisk::Missing || create_lock.is_some() {
            return Ok((is_single_file, on_disk, create_lock));
        }
        create_lock = lock_if_missing(path)?;
    }
}

/// Settles a missing database path under the migration lock before a
/// read-write open creates a database there.
///
/// A migration in another process holds the lock while `<path>` is missing
/// between its renames, so under the lock this finishes a cut-off migration
/// (as [`finish_interrupted`] does) and looks again. Returns the lock, to be
/// held while the database is created, when `<path>` is still missing and
/// nothing of a migration is next to it (the directory of `path` is created
/// for the lock file first); `None` when `<path>` exists (now).
///
/// # Errors
///
/// Returns an error if another process holds the migration lock for longer
/// than the wait, if only `<path>.migrating` is left, if a kept copy of a
/// migration is left without its 0.6 file (the database was migrated and its
/// new file is missing: nothing is created there), or if a file or the
/// directory cannot be inspected, created or renamed.
pub(super) fn lock_if_missing(path: &Path) -> Result<Option<MigrateLock>> {
    if exists(path)? {
        return Ok(None);
    }
    if let Some(parent) = path.parent()
        && !parent.as_os_str().is_empty()
    {
        fs::create_dir_all(parent).map_err(|error| {
            Error::Internal(format!("cannot create {}: {error}", parent.display()))
        })?;
    }
    let lock = MigrateLock::acquire(path)?;
    resolve(path)?;
    if exists(path)? {
        return Ok(None);
    }
    refuse_kept_copy_without_database(path)?;
    Ok(Some(lock))
}

/// Migrates the 0.5.x database at `path` to the 0.6 format: a read-only open,
/// the 0.6 image written to `<path>.migrating`, then the renames (see the
/// module docs). Nothing to do if, once the lock is held, `<path>` is no
/// longer a 0.5.x file: another process migrated it meanwhile.
///
/// `config` is the configuration of the open that migrates; the read-only
/// open of the old database uses its memory limit and spill path, and the
/// image is encrypted with its key chain, if any.
///
/// # Errors
///
/// Returns an error if another process holds the migration lock for longer
/// than the wait, a kept copy (`<path>.pre-0.6` and its side files) already
/// exists, the old database cannot be read (for example while a 0.5.x process
/// holds it, or in a build without the `wal` feature when its sidecar WAL
/// holds files), or the image cannot be written or a rename fails.
pub(super) fn migrate(path: &Path, config: &Config) -> Result<()> {
    let _lock = MigrateLock::acquire(path)?;
    // While this open waited for the lock, another process may have migrated
    // the file, or crashed while it did.
    resolve(path)?;
    if detect(path)? != OnDisk::LegacyFile {
        return Ok(());
    }

    let moves = kept_names(path);
    for (_, kept, _) in &moves {
        if exists(kept)? {
            return Err(Error::Internal(format!(
                "cannot migrate {} to the 0.6 format: {} already exists, and a migration \
                 never replaces a kept copy; move it away and open the database again",
                path.display(),
                kept.display()
            )));
        }
    }

    let failed = |error: Error| with_migration_context(path, error);
    // A 0.5.x process that opened the file for writing after the read-only
    // open would write to the kept copy: a shared lock keeps it out until the
    // old file is moved, and makes the migration fail while one holds it.
    let _old_file_lock = lock_shared(path)?;
    let migrating = migrating_path(path);
    // The old database is read without a key (a 0.5.x file is never
    // encrypted); the image is encrypted when the open has one, and then no
    // spill file may hold the old data in plaintext meanwhile.
    let keys = DatabaseKeys::from_config(config);
    let mut old_config = Config::read_only(path);
    old_config.memory_limit = config.memory_limit;
    old_config.spill_path.clone_from(&config.spill_path);
    let old = GrafeoDB::with_config_and_spill(old_config, !keys.is_encrypted()).map_err(failed)?;
    old.write_image_with(&migrating, &keys).map_err(failed)?;
    grafeo_info!(
        "migrating {} to the 0.6 format: {} nodes, {} edges, {} named graphs, {} indexes \
         (as SHOW INDEXES lists them), {} constraints",
        path.display(),
        old.node_count(),
        old.edge_count(),
        old.list_graphs().len(),
        old.catalog.index_count(),
        old.catalog.constraints().len()
    );
    drop(old);
    // What the written image records, read back from it before the old files
    // move (which also checks its directory and key): an image that does not
    // open is never installed.
    let (written_nodes, written_edges) = match image_counts(&migrating, &keys) {
        Ok(counts) => counts,
        Err(error) => {
            remove_after_failure(&migrating);
            return Err(failed(error));
        }
    };
    maybe_crash("migrate:after_image");

    // Up to the first rename the old database is untouched: on failure the
    // image goes, so the database is as it was.
    let [(file, kept_file, file_point), side_files @ ..] = &moves;
    if let Err(error) = rename(file, kept_file, file_point) {
        remove_after_failure(&migrating);
        return Err(error);
    }
    // The move of the old file is durable before a side file moves: a power
    // loss that kept a side file's move without it would leave a 0.5.x
    // `<path>` without its sidecar WAL.
    sync_parent_dir(path)?;
    // The side files go before the image comes: a 0.6 database next to the
    // 0.5.x sidecar WAL would replay it.
    move_side_files(path, side_files)?;
    maybe_crash("migrate:after_old");
    maybe_pause("migrate:after_old");

    rename(&migrating, path, INSTALLED)?;
    sync_parent_dir(path)?;
    maybe_crash("migrate:after_new");
    grafeo_info!(
        "migrated {} to the 0.6 format: its database header records {} nodes and {} edges; \
         the 0.5.x database is kept as {}",
        path.display(),
        written_nodes,
        written_edges,
        kept_file.display()
    );
    if keys.is_encrypted() {
        // Every kept file that exists (one that cannot be inspected is named
        // too): each holds 0.5.x data in plaintext.
        let kept_files: Vec<String> = moves
            .iter()
            .filter(|(_, kept, _)| kept.try_exists().unwrap_or(true))
            .map(|(_, kept, _)| kept.display().to_string())
            .collect();
        grafeo_warn!(
            "{} is encrypted now, but the 0.5.x files the migration kept are not: {}; \
             remove them once you no longer need to go back to 0.5.x",
            path.display(),
            kept_files.join(", ")
        );
    }
    Ok(())
}

/// The node and edge counts the active database header of the image at
/// `image` records, opened read-only with the keys it was written with.
fn image_counts(image: &Path, keys: &DatabaseKeys) -> Result<(u64, u64)> {
    let manager =
        GrafeoFileManager::open_read_only_with_cipher_for(image, |id| keys.container_cipher(id))?;
    let header = manager.active_header();
    Ok((header.node_count, header.edge_count))
}

/// Refuses a read-only open of a database whose migration was cut off after
/// the old file or one of its side files was moved away (only a read-write
/// open finishes it), or whose migrated file is missing next to its kept copy.
///
/// # Errors
///
/// Returns an error if `<path>` is missing and `<path>.migrating` or a kept
/// copy exists, if a 0.5.x `<path>` exists next to `<path>.migrating` while
/// one of its side files is under its kept name (the 0.5.x database would be
/// read without it, see [`move_side_files_back`]), or a file cannot be
/// inspected.
pub(super) fn check_read_only(path: &Path) -> Result<()> {
    if exists(path)? {
        // Only a 0.5.x `<path>` can be part of a cut-off migration (as in
        // `resolve`): next to a 0.6 file, the leftovers are left alone.
        if !exists(&migrating_path(path))? || detect(path)? != OnDisk::LegacyFile {
            return Ok(());
        }
        let [(_, kept_file, _), side_files @ ..] = &kept_names(path);
        if exists(kept_file)? {
            return Ok(());
        }
        for (_, kept, _) in side_files {
            if exists(kept)? {
                return Err(Error::Internal(format!(
                    "{} cannot be read: its migration to the 0.6 format was interrupted after \
                     {} was moved away from it; a read-write open finishes the migration",
                    path.display(),
                    kept.display()
                )));
            }
        }
        return Ok(());
    }
    if !exists(&migrating_path(path))? {
        return refuse_kept_copy_without_database(path);
    }
    let kept = pre_06_path(path);
    if exists(&kept)? {
        return Err(Error::Internal(format!(
            "{} is missing: its migration to the 0.6 format was interrupted after the \
             0.5.x database was moved to {}; a read-write open finishes the migration",
            path.display(),
            kept.display()
        )));
    }
    Err(missing_old_database(path))
}

/// Refuses to read the 0.5.x database at `path` when its sidecar WAL holds
/// files and the build cannot replay them (`can_replay` is false without the
/// `wal` feature): reading the file alone would lose the WAL's changes.
///
/// # Errors
///
/// Returns an error naming the sidecar WAL in that case, or if it cannot be
/// inspected.
pub(super) fn refuse_unreplayable_wal(path: &Path, can_replay: bool) -> Result<()> {
    let wal = sidecar_wal_path(path);
    if can_replay || !GrafeoDB::holds_files(&wal)? {
        return Ok(());
    }
    Err(Error::Internal(format!(
        "{} has a 0.5.x sidecar WAL ({}) with changes only a build with the `wal` \
         feature can replay: open or migrate it with such a build",
        path.display(),
        wal.display()
    )))
}

/// Acts on the files a cut-off migration left (the table in the module docs).
/// The caller holds the migration lock.
fn resolve(path: &Path) -> Result<()> {
    let migrating = migrating_path(path);
    if !exists(&migrating)? {
        return Ok(());
    }
    match detect(path)? {
        // The image may be incomplete: the migration runs again, with the
        // side files back under their 0.5.x names.
        OnDisk::LegacyFile => {
            move_side_files_back(path)?;
            remove_file(&migrating)
        }
        // The old database was moved after the image was complete. Its side
        // files go before the image comes, as in `migrate`. The move of the
        // old file may not be durable yet (an error, not a crash, may have
        // stopped the migration in this process): it is made so first.
        OnDisk::Missing if exists(&pre_06_path(path))? => {
            sync_parent_dir(path)?;
            let [_, side_files @ ..] = &kept_names(path);
            move_side_files(path, side_files)?;
            rename(&migrating, path, INSTALLED)?;
            sync_parent_dir(path)
        }
        OnDisk::Missing => Err(missing_old_database(path)),
        // `<path>` is the database; the image is no part of a migration state.
        _ => {
            grafeo_warn!(
                "{} is not part of a migration (the database {} is not a 0.5.x file) and is \
                 left alone; remove it if it is not needed",
                migrating.display(),
                path.display()
            );
            Ok(())
        }
    }
}

/// Moves each 0.5.x side file of `side_files` (the sidecar WAL and a pending
/// checkpoint image, with their kept names and crash points, from
/// [`kept_names`]) that exists to its kept name, syncing the directory of
/// `path` after each move, so a power loss keeps the first moves and never a
/// later one without an earlier one. The caller has moved the database file
/// and synced the directory.
///
/// # Errors
///
/// Returns an error if a side file and its kept name both exist, or a file
/// cannot be inspected, renamed or synced.
fn move_side_files(path: &Path, side_files: &[(PathBuf, PathBuf, &'static str)]) -> Result<()> {
    for (from, to, point) in side_files {
        if !exists(from)? {
            continue;
        }
        if exists(to)? {
            return Err(Error::Internal(format!(
                "cannot finish the migration of {}: both {} and {} exist",
                path.display(),
                from.display(),
                to.display()
            )));
        }
        rename(from, to, point)?;
        sync_parent_dir(path)?;
    }
    Ok(())
}

/// Moves the 0.5.x side files (sidecar WAL, pending checkpoint image) found
/// under their kept names back to their 0.5.x names while the database file is
/// still `<path>`: a power loss kept their move and lost the earlier move of
/// the database file. The 0.5.x database is then whole again for the
/// migration to run again. Nothing moves while `<path>.pre-0.6` exists (a kept
/// copy that is no part of this state, which `migrate` refuses to replace).
///
/// # Errors
///
/// Returns an error if a side file exists under both names, or a file cannot
/// be inspected, renamed or synced.
fn move_side_files_back(path: &Path) -> Result<()> {
    let [(_, kept_file, _), side_files @ ..] = &kept_names(path);
    if exists(kept_file)? {
        return Ok(());
    }
    let mut moved = false;
    for (original, kept, _) in side_files {
        if !exists(kept)? {
            continue;
        }
        if exists(original)? {
            return Err(Error::Internal(format!(
                "cannot finish the migration of {}: both {} and {} exist",
                path.display(),
                original.display(),
                kept.display()
            )));
        }
        fs::rename(kept, original).map_err(|error| {
            Error::Internal(format!(
                "cannot move {} back to {}: {error}",
                kept.display(),
                original.display()
            ))
        })?;
        grafeo_warn!(
            "{} was moved to {} by a migration whose move of {} was lost; it is moved back \
             and the migration runs again",
            original.display(),
            kept.display(),
            path.display()
        );
        moved = true;
    }
    if moved {
        sync_parent_dir(path)?;
    }
    Ok(())
}

/// Adds that `error` happened while migrating `path` to its message, keeping its
/// variant: an I/O error keeps its kind, a message-carrying error gets the
/// context in front of its message, and an error without a message is
/// returned as it is.
fn with_migration_context(path: &Path, error: Error) -> Error {
    let context = |message: &dyn std::fmt::Display| {
        format!(
            "error while migrating {} to the 0.6 format: {message}",
            path.display()
        )
    };
    match error {
        Error::Io(source) => Error::Io(std::io::Error::new(source.kind(), context(&source))),
        Error::Internal(message) => Error::Internal(context(&message)),
        Error::Serialization(message) => Error::Serialization(context(&message)),
        Error::InvalidValue(message) => Error::InvalidValue(context(&message)),
        Error::Storage(StorageError::Corruption(message)) => {
            Error::Storage(StorageError::Corruption(context(&message)))
        }
        Error::Storage(StorageError::InvalidWalEntry(message)) => {
            Error::Storage(StorageError::InvalidWalEntry(context(&message)))
        }
        Error::Storage(StorageError::RecoveryFailed(message)) => {
            Error::Storage(StorageError::RecoveryFailed(context(&message)))
        }
        Error::Storage(StorageError::CheckpointFailed(message)) => {
            Error::Storage(StorageError::CheckpointFailed(context(&message)))
        }
        other => other,
    }
}

/// Refuses a missing `<path>` next to a kept copy of a migration: the
/// database was migrated and its 0.6 file is missing, so an empty database
/// there would hide it.
fn refuse_kept_copy_without_database(path: &Path) -> Result<()> {
    for (_, kept, _) in kept_names(path) {
        if exists(&kept)? {
            return Err(Error::Internal(format!(
                "cannot open {}: the database was migrated to the 0.6 format and its \
                 0.5.x files are kept (found {}), but its 0.6 file {} is missing; \
                 nothing was created",
                path.display(),
                kept.display(),
                path.display()
            )));
        }
    }
    Ok(())
}

/// The error for an image without the old database it was migrated from.
fn missing_old_database(path: &Path) -> Error {
    Error::Internal(format!(
        "cannot open {}: {} holds an image of an interrupted migration to the 0.6 \
         format, but the 0.5.x database (neither {} nor {}) is missing, so the \
         migration cannot be finished; nothing was changed",
        path.display(),
        migrating_path(path).display(),
        path.display(),
        pre_06_path(path).display()
    ))
}

/// The 0.5.x files of a database, the names they are kept under, and the
/// crash point after each rename: the file, its sidecar WAL and a pending
/// checkpoint image.
fn kept_names(path: &Path) -> [(PathBuf, PathBuf, &'static str); 3] {
    let kept = pre_06_path(path);
    [
        (path.to_path_buf(), kept.clone(), "migrate:renamed:.pre-0.6"),
        (
            sidecar_wal_path(path),
            sidecar_wal_path(&kept),
            "migrate:renamed:.pre-0.6.wal",
        ),
        (
            with_suffix(path, ".checkpoint"),
            with_suffix(&kept, ".checkpoint"),
            "migrate:renamed:.pre-0.6.checkpoint",
        ),
    ]
}

fn with_suffix(path: &Path, suffix: &str) -> PathBuf {
    let mut name = path.as_os_str().to_owned();
    name.push(suffix);
    PathBuf::from(name)
}

fn exists(path: &Path) -> Result<bool> {
    path.try_exists()
        .map_err(|error| Error::Internal(format!("cannot inspect {}: {error}", path.display())))
}

/// Renames `from` to `to`, then passes the crash point `point`, so a test can
/// crash after each rename and pin their order.
fn rename(from: &Path, to: &Path, point: &'static str) -> Result<()> {
    fs::rename(from, to).map_err(|error| {
        Error::Internal(format!(
            "cannot rename {} to {}: {error}",
            from.display(),
            to.display()
        ))
    })?;
    maybe_crash(point);
    Ok(())
}

fn remove_file(path: &Path) -> Result<()> {
    fs::remove_file(path)
        .map_err(|error| Error::Internal(format!("cannot remove {}: {error}", path.display())))
}

/// A shared lock on the 0.5.x database file, released when dropped. A 0.5.x
/// writer's exclusive lock and this one exclude each other.
struct SharedLock(File);

impl Drop for SharedLock {
    fn drop(&mut self) {
        let _ = self.0.unlock();
    }
}

/// Opens the 0.5.x file at `path` under a shared lock.
fn lock_shared(path: &Path) -> Result<SharedLock> {
    let file = File::open(path)
        .map_err(|error| Error::Internal(format!("cannot open {}: {error}", path.display())))?;
    child_process::take_lock(
        || file.try_lock_shared(),
        |error| matches!(error, std::fs::TryLockError::WouldBlock),
    )
    .map_err(|error| match error {
        std::fs::TryLockError::WouldBlock => Error::Internal(format!(
            "cannot migrate {}: the database file is locked by another process",
            path.display()
        )),
        std::fs::TryLockError::Error(error) => {
            Error::Internal(format!("cannot lock {}: {error}", path.display()))
        }
    })?;
    Ok(SharedLock(file))
}

/// Removes the image of a migration that failed. Best effort: the error that
/// matters is the one returned, and the next open removes a leftover image.
fn remove_after_failure(migrating: &Path) {
    if let Err(error) = fs::remove_file(migrating) {
        grafeo_warn!(
            "cannot remove the image of the failed migration {}: {error}",
            migrating.display()
        );
    }
}

/// Makes the renames in the directory holding `path` durable, as the storage
/// crate does after creating a file. Windows has no directory handles to
/// sync; its directory changes are metadata-journaled.
fn sync_parent_dir(path: &Path) -> Result<()> {
    #[cfg(unix)]
    if let Some(parent) = path.parent() {
        let parent = if parent.as_os_str().is_empty() {
            Path::new(".")
        } else {
            parent
        };
        File::open(parent)
            .and_then(|directory| directory.sync_all())
            .map_err(|error| {
                Error::Internal(format!("cannot sync {}: {error}", parent.display()))
            })?;
    }
    #[cfg(not(unix))]
    let _ = path;
    Ok(())
}

/// The exclusive lock on `<path>.migrate.lock`, held while the files of a
/// migration change. Dropping it removes the lock file and releases the lock.
pub(super) struct MigrateLock {
    /// The locked lock file.
    file: File,
    /// Where the lock file is.
    path: PathBuf,
}

/// One try to take the migration lock.
enum Attempt {
    Taken(MigrateLock),
    /// Another process holds the lock, or just released it.
    Held,
    /// Windows refuses to open a file that is being removed; anywhere else
    /// this is an error.
    Denied(std::io::Error),
}

impl MigrateLock {
    /// Takes the migration lock of the database at `database`, trying again
    /// every 50 ms while another process holds it, for up to five seconds.
    fn acquire(database: &Path) -> Result<Self> {
        let path = migrate_lock_path(database);
        let deadline = Instant::now() + LOCK_WAIT;
        loop {
            match Self::try_acquire(&path)? {
                Attempt::Taken(lock) => return Ok(lock),
                _ if Instant::now() < deadline => std::thread::sleep(LOCK_RETRY),
                Attempt::Denied(error) => {
                    return Err(Error::Internal(format!(
                        "cannot open the migration lock file {}: {error}",
                        path.display()
                    )));
                }
                Attempt::Held => {
                    return Err(Error::Internal(format!(
                        "database locked: a migration is running ({} is held by another \
                         process); open the database again once it has finished",
                        path.display()
                    )));
                }
            }
        }
    }

    fn try_acquire(path: &Path) -> Result<Attempt> {
        let file = match OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(path)
        {
            Ok(file) => file,
            Err(error) if cfg!(windows) && error.kind() == std::io::ErrorKind::PermissionDenied => {
                return Ok(Attempt::Denied(error));
            }
            Err(error) => {
                return Err(Error::Internal(format!(
                    "cannot open the migration lock file {}: {error}",
                    path.display()
                )));
            }
        };
        match child_process::take_lock(
            || file.try_lock(),
            |error| matches!(error, std::fs::TryLockError::WouldBlock),
        ) {
            Ok(()) => {}
            Err(std::fs::TryLockError::WouldBlock) => return Ok(Attempt::Held),
            Err(std::fs::TryLockError::Error(error)) => {
                return Err(Error::Internal(format!(
                    "cannot lock the migration lock file {}: {error}",
                    path.display()
                )));
            }
        }
        // A holder writes to the file only after removing it: this handle
        // is to a file that is no longer at `path`.
        let length = file.metadata().map_err(|error| {
            Error::Internal(format!(
                "cannot inspect the migration lock file {}: {error}",
                path.display()
            ))
        })?;
        if length.len() > 0 {
            return Ok(Attempt::Held);
        }
        Ok(Attempt::Taken(Self {
            file,
            path: path.to_path_buf(),
        }))
    }
}

impl Drop for MigrateLock {
    fn drop(&mut self) {
        // A panic stands in for a crash (crash injection does this), which
        // leaves the lock file: the next open removes it.
        if std::thread::panicking() {
            return;
        }
        match fs::remove_file(&self.path) {
            Ok(()) => {
                // A process waiting on this handle sees the mark once it gets
                // the lock, and tries again with a new file.
                if let Err(error) = self.file.write_all(RELEASED) {
                    grafeo_warn!(
                        "cannot mark the removed migration lock file {}: {error}",
                        self.path.display()
                    );
                }
            }
            Err(error) => grafeo_warn!(
                "cannot remove the migration lock file {}: {error}",
                self.path.display()
            ),
        }
        // Dropping the file releases the lock.
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The refusal a build without the `wal` feature applies, tested here with
    /// `can_replay` false: a sidecar WAL with files refuses, an empty or
    /// missing one does not, and a build that replays never refuses.
    #[test]
    fn a_sidecar_wal_with_files_needs_a_build_that_replays_it() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("amsterdam.grafeo");
        let wal = sidecar_wal_path(&path);

        refuse_unreplayable_wal(&path, false).expect("no sidecar WAL: nothing to replay");
        fs::create_dir(&wal).unwrap();
        refuse_unreplayable_wal(&path, false).expect("an empty sidecar WAL: nothing to replay");

        fs::write(wal.join("wal_00000000.log"), b"Gus").unwrap();
        let error = refuse_unreplayable_wal(&path, false)
            .expect_err("a sidecar WAL with files needs a replay")
            .to_string();
        assert!(
            error.contains(&wal.display().to_string()) && error.contains("`wal` feature"),
            "the error names the WAL and the feature: {error}"
        );
        refuse_unreplayable_wal(&path, true).expect("a build with the `wal` feature replays it");
    }

    /// The migration context keeps an error's variant (and an I/O error's kind)
    /// and puts the context in front of its message.
    #[test]
    fn the_migration_context_keeps_the_variant() {
        let path = Path::new("berlin.grafeo");
        let context = "error while migrating berlin.grafeo to the 0.6 format: ";

        let io = with_migration_context(
            path,
            Error::Io(std::io::Error::new(
                std::io::ErrorKind::StorageFull,
                "no space left",
            )),
        );
        match &io {
            Error::Io(source) => {
                assert_eq!(source.kind(), std::io::ErrorKind::StorageFull);
                assert_eq!(source.to_string(), format!("{context}no space left"));
            }
            other => panic!("an I/O error stays an I/O error, got {other:?}"),
        }

        let internal = with_migration_context(path, Error::Internal("CRC mismatch".into()));
        assert!(
            matches!(&internal, Error::Internal(message) if *message == format!("{context}CRC mismatch")),
            "{internal:?}"
        );
        let corrupt = with_migration_context(
            path,
            Error::Storage(StorageError::Corruption("torn page".into())),
        );
        assert!(
            matches!(
                &corrupt,
                Error::Storage(StorageError::Corruption(message))
                    if *message == format!("{context}torn page")
            ),
            "{corrupt:?}"
        );
        let full = with_migration_context(path, Error::Storage(StorageError::Full));
        assert!(
            matches!(full, Error::Storage(StorageError::Full)),
            "an error without a message stays as it is: {full:?}"
        );
    }
}
