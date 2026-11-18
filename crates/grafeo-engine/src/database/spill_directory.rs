//! Where an open spills, and who removes what (#594).
//!
//! | Open | Spill path (query and section spills) | Vector cache |
//! | --- | --- | --- |
//! | explicit [`Config::spill_path`](crate::config::Config::spill_path) | the spill path | `<spill_path>/grafeo-<name>-<pid>-<random>/` |
//! | read-write file database | `<file>.spill` | `<file>.spill/cache/` |
//! | read-only file database | `<temp>/grafeo-<name>-<pid>-<random>.spill`, with its `owner.lock` | the same directory |
//! | in-memory without a spill path, encrypted | none | none |
//!
//! Several databases may share an explicit spill path, so each open's vector
//! cache gets a directory of its own there, and nothing removes another
//! open's directory. `<file>.spill/cache/` belongs to the read-write open
//! that holds the database's exclusive lock: it removes what a previous open
//! left there, and after its `close()` (which releases the lock) it writes
//! nothing more into it. A read-only open writes nothing beside the
//! database: its directory in the system temp directory is created at open,
//! exclusively (with a random name, and readable only by its user on Unix),
//! so another user of a shared temp directory cannot have made it first;
//! when it cannot be created (a read-only or missing temp directory), the
//! open warns and spills nothing. It holds an exclusive lock on the
//! directory's `owner.lock` while the directory lives, and the operating
//! system drops the lock when the process dies. So once per process, the
//! first read-only open, once it holds its own directory and before the spill
//! account measures the free space, removes the directories of dead owners in
//! the temp directory: those of its own user whose lock it can take (#450).
//! Directories without an `owner.lock` (of 0.6.3 and earlier, or on a file
//! system without file locks) stay, with one warning. The other directories are created only
//! when something spills. Each is removed
//! (when empty) once the last spilled column reading from it is gone.
//! `<file>.spill/kept/` is not spilled data: it holds the old spill files of
//! an older build that an open kept instead of folding them in completely
//! (see `legacy_spill`), and nothing here reads or removes it.

use std::fs::File;
use std::path::{Path, PathBuf};
use std::sync::Arc;
#[cfg(all(feature = "vector-index", feature = "wal"))]
use std::sync::atomic::AtomicBool;
use std::sync::atomic::{AtomicU64, Ordering};

/// How many names a read-only open tries for its temp directory before it
/// gives up: each is random, so a clash means someone made them on purpose.
const NAME_ATTEMPTS: usize = 16;

/// The file in a read-only open's temp directory whose exclusive lock the
/// open holds while the directory lives: a directory whose lock a sweep can
/// take belongs to a process that is gone (#450).
const OWNER_LOCK: &str = "owner.lock";

/// A directory an open spills into, removed when the last user lets go of
/// it (only when it is empty: what is still in it was not ours to delete).
pub(crate) struct SpillDirectory {
    path: PathBuf,
    /// Also remove the parent when that leaves it empty: `<file>.spill`
    /// around `cache/`.
    remove_empty_parent: bool,
    /// The locked `owner.lock` of a read-only open's temp directory, held for
    /// the directory's whole life (#450).
    owner: Option<File>,
    /// Set by `close()`: nothing more is written here (see the module docs).
    #[cfg(all(feature = "vector-index", feature = "wal"))]
    closed: AtomicBool,
}

impl SpillDirectory {
    fn new(path: PathBuf, remove_empty_parent: bool) -> Self {
        Self {
            path,
            remove_empty_parent,
            owner: None,
            #[cfg(all(feature = "vector-index", feature = "wal"))]
            closed: AtomicBool::new(false),
        }
    }

    /// A read-only open's temp directory at `path`, whose `owner.lock` it
    /// holds locked in `owner`.
    fn owned(path: PathBuf, owner: File) -> Self {
        let mut directory = Self::new(path, false);
        directory.owner = Some(owner);
        directory
    }

    /// A directory at `path` that no other directory removes (tests).
    #[cfg(all(test, feature = "vector-index", feature = "wal"))]
    pub(crate) fn at(path: PathBuf) -> Self {
        Self::new(path, false)
    }
}

// What the vector cache uses (a build without it only removes directories).
#[cfg(all(feature = "vector-index", feature = "wal"))]
impl SpillDirectory {
    /// The directory's path.
    pub(crate) fn path(&self) -> &Path {
        &self.path
    }

    /// Creates the directory (and its parents) if they are missing.
    ///
    /// # Errors
    ///
    /// Returns an error once the database is closed, or the error of
    /// creating the directory.
    pub(crate) fn create(&self) -> std::io::Result<()> {
        if self.is_closed() {
            return Err(std::io::Error::other(
                "the database is closed: nothing more is spilled",
            ));
        }
        std::fs::create_dir_all(&self.path)
    }

    /// Stops all further writes into the directory: the database closed.
    pub(crate) fn close_for_writes(&self) {
        self.closed.store(true, Ordering::Release);
    }

    /// Whether the database closed (see [`close_for_writes`](Self::close_for_writes)).
    pub(crate) fn is_closed(&self) -> bool {
        self.closed.load(Ordering::Acquire)
    }
}

impl Drop for SpillDirectory {
    fn drop(&mut self) {
        if let Some(owner) = self.owner.take() {
            // An emptied directory loses its lock file while still locked, so
            // no sweep finds it unlocked. One with something left keeps it,
            // unlocked: the next sweep removes it as a dead owner's.
            if only_the_lock_is_left(&self.path) {
                let _ = std::fs::remove_file(self.path.join(OWNER_LOCK));
            }
            drop(owner);
        }
        // Non-recursive: fails, and keeps the directory, while anything is
        // left in it.
        if std::fs::remove_dir(&self.path).is_ok()
            && self.remove_empty_parent
            && let Some(parent) = self.path.parent()
        {
            let _ = std::fs::remove_dir(parent);
        }
    }
}

/// The spill directories of one open (see the module docs).
pub(crate) struct SpillLayout {
    /// The spill path of the buffer manager.
    pub(crate) root: Option<PathBuf>,
    /// Held by the database while it is the database's own (read-only).
    pub(crate) root_guard: Option<Arc<SpillDirectory>>,
    /// Where vector columns spill to.
    #[cfg(all(feature = "vector-index", feature = "wal"))]
    pub(crate) vector_cache: Option<Arc<SpillDirectory>>,
    /// A cache a previous read-write open left behind, removed at open.
    pub(crate) stale_cache: Option<PathBuf>,
}

impl SpillLayout {
    /// The layout of an open with an explicit `spill_path`, or one derived
    /// from `database` (`None` for an in-memory database, or one that must
    /// not spill). A read-only open's directory is created here, in the
    /// system temp directory, which only that open asks for: on wasm32 the
    /// call panics, and an in-memory database never makes it.
    pub(crate) fn for_open(
        spill_path: Option<&Path>,
        database: Option<&Path>,
        read_only: bool,
    ) -> Self {
        Self::for_open_in(std::env::temp_dir, spill_path, database, read_only)
    }

    /// [`for_open`](Self::for_open) with the directory a read-only open
    /// creates its own in, asked for only by that open. When that fails (the temp directory is missing or
    /// read-only, or every name tried was taken), the open warns and spills
    /// nothing, as an in-memory database without a spill path: it never
    /// falls back to a directory it did not create.
    fn for_open_in(
        temp_parent: impl FnOnce() -> PathBuf,
        spill_path: Option<&Path>,
        database: Option<&Path>,
        read_only: bool,
    ) -> Self {
        let name = database.and_then(Path::file_name).map_or_else(
            || "memory".to_string(),
            |name| name.to_string_lossy().into_owned(),
        );
        if let Some(spill_path) = spill_path {
            return Self {
                root: Some(spill_path.to_path_buf()),
                root_guard: None,
                #[cfg(all(feature = "vector-index", feature = "wal"))]
                vector_cache: Some(Arc::new(SpillDirectory::new(
                    spill_path.join(unique_name(&name, "")),
                    false,
                ))),
                stale_cache: None,
            };
        }
        let derived = database.and_then(|database| {
            let parent = database.parent()?;
            let name = database.file_name()?.to_str()?;
            Some(parent.join(format!("{name}.spill")))
        });
        let Some(derived) = derived else {
            return Self::none();
        };
        if read_only {
            // Nothing in this build spills: no directory.
            if !cfg!(any(
                feature = "spill",
                all(feature = "vector-index", feature = "wal")
            )) {
                return Self::none();
            }
            let temp_parent = temp_parent();
            let (created, owner) =
                match create_owned_dir(&temp_parent, || unique_name(&name, ".spill")) {
                    Ok(created) => created,
                    Err(error) => {
                        grafeo_common::grafeo_warn!(
                            "the read-only open of {} spills nothing: it cannot create its spill \
                         directory in {}: {error}",
                            database.map_or_else(String::new, |path| path.display().to_string()),
                            temp_parent.display()
                        );
                        return Self::none();
                    }
                };
            // With its own directory held, and before the account measures
            // the free space: dead owners' space is free again.
            sweep_dead_once(&temp_parent, &created);
            let own = Arc::new(match owner {
                Some(owner) => SpillDirectory::owned(created, owner),
                None => SpillDirectory::new(created, false),
            });
            return Self {
                root: Some(own.path.clone()),
                root_guard: Some(Arc::clone(&own)),
                #[cfg(all(feature = "vector-index", feature = "wal"))]
                vector_cache: Some(Arc::clone(&own)),
                stale_cache: None,
            };
        }
        // A read-write open makes `<file>.spill` a directory; a link there is
        // not one it left (in a shared directory another user can plant one),
        // and spilling through it would write into whatever it points at. The
        // open spills nothing instead. Read-only opens never reach this: they
        // spill to their own private temp directory above.
        if is_link(&derived) {
            grafeo_common::grafeo_warn!(
                "the spill directory {} is a link, not a directory an open made: this open \
                 spills nothing",
                derived.display()
            );
            return Self::none();
        }
        let cache = derived.join("cache");
        Self {
            root: Some(derived),
            root_guard: None,
            #[cfg(all(feature = "vector-index", feature = "wal"))]
            vector_cache: Some(Arc::new(SpillDirectory::new(cache.clone(), true))),
            stale_cache: Some(cache),
        }
    }

    fn none() -> Self {
        Self {
            root: None,
            root_guard: None,
            #[cfg(all(feature = "vector-index", feature = "wal"))]
            vector_cache: None,
            stale_cache: None,
        }
    }
}

/// Creates a directory in `parent` exclusively, under the first free name
/// `next_name` gives: the last component must not exist (a directory or a
/// link someone made first is never used), and on Unix only its user may
/// enter it. After [`NAME_ATTEMPTS`] taken names it gives up.
fn create_private_dir(
    parent: &Path,
    mut next_name: impl FnMut() -> String,
) -> std::io::Result<PathBuf> {
    #[cfg(unix)]
    let builder = {
        use std::os::unix::fs::DirBuilderExt;
        let mut builder = std::fs::DirBuilder::new();
        builder.mode(0o700);
        builder
    };
    #[cfg(not(unix))]
    let builder = std::fs::DirBuilder::new();
    for _ in 0..NAME_ATTEMPTS {
        let path = parent.join(next_name());
        match builder.create(&path) {
            Ok(()) => return Ok(path),
            Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {}
            Err(error) => return Err(error),
        }
    }
    Err(std::io::Error::new(
        std::io::ErrorKind::AlreadyExists,
        format!(
            "no free name for a spill directory in {} after {NAME_ATTEMPTS} attempts",
            parent.display()
        ),
    ))
}

/// Creates a read-only open's directory in `parent` (see
/// [`create_private_dir`]) and takes its owner lock: the directory, and the
/// locked `owner.lock` in it. A sweep that took the directory between its
/// creation and the lock costs one more name. On a file system without file
/// locks the directory has no `owner.lock` (`None`), as in 0.6.3: it spills,
/// and no sweep removes it after a crash.
fn create_owned_dir(
    parent: &Path,
    mut next_name: impl FnMut() -> String,
) -> std::io::Result<(PathBuf, Option<File>)> {
    for _ in 0..NAME_ATTEMPTS {
        let created = create_private_dir(parent, &mut next_name)?;
        match take_owner(&created) {
            Ok(Owner::Held(owner)) => return Ok((created, Some(owner))),
            // A sweep had it: what it left of the directory goes (its handle
            // is closed by now), and the next name is tried.
            Ok(Owner::Lost) => {
                let _ = std::fs::remove_dir(&created);
            }
            Ok(Owner::NoLocks(error)) => {
                let _ = std::fs::remove_file(created.join(OWNER_LOCK));
                grafeo_common::grafeo_warn!(
                    "the file system of {} has no file locks ({error}): the spill directory {} \
                     is not removed by a later open if this process crashes",
                    parent.display(),
                    created.display()
                );
                return Ok((created, None));
            }
            Err(error) => {
                let _ = std::fs::remove_file(created.join(OWNER_LOCK));
                let _ = std::fs::remove_dir(&created);
                return Err(error);
            }
        }
    }
    Err(std::io::Error::new(
        std::io::ErrorKind::AlreadyExists,
        format!(
            "no spill directory in {} kept its owner lock after {NAME_ATTEMPTS} attempts",
            parent.display()
        ),
    ))
}

/// What a creator got of its directory's owner lock.
#[derive(Debug)]
enum Owner {
    /// The locked `owner.lock`.
    Held(File),
    /// A sweep took the directory first: it holds the lock, or removed the
    /// directory once ours was taken.
    Lost,
    /// The file system takes no file locks.
    NoLocks(std::io::Error),
}

/// Creates `dir`'s `owner.lock` and locks it.
fn take_owner(dir: &Path) -> std::io::Result<Owner> {
    let lock = dir.join(OWNER_LOCK);
    let owner = match std::fs::OpenOptions::new()
        .read(true)
        .write(true)
        .create_new(true)
        .open(&lock)
    {
        Ok(owner) => owner,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(Owner::Lost),
        Err(error) => return Err(error),
    };
    Ok(lock_owner(&lock, owner))
}

/// Locks `owner`, the open handle of the `owner.lock` at `lock` its open just
/// created.
fn lock_owner(lock: &Path, owner: File) -> Owner {
    match owner.try_lock() {
        Ok(()) => {}
        Err(std::fs::TryLockError::WouldBlock) => return Owner::Lost,
        Err(std::fs::TryLockError::Error(error)) => return Owner::NoLocks(error),
    }
    // Locked, but a sweep may have removed the directory just before.
    if lock.exists() {
        Owner::Held(owner)
    } else {
        Owner::Lost
    }
}

/// Whether nothing but `owner.lock` is in `dir`.
fn only_the_lock_is_left(dir: &Path) -> bool {
    std::fs::read_dir(dir).is_ok_and(|entries| {
        entries
            .flatten()
            .all(|entry| entry.file_name() == OWNER_LOCK)
    })
}

/// Who owns a file: its user on Unix. Elsewhere (Windows) the temp directory
/// is the user's own, and there is nothing to compare.
#[cfg(unix)]
type User = u32;
#[cfg(not(unix))]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct User;

/// The user owning the file `metadata` describes (see [`User`]).
fn user_of(metadata: &std::fs::Metadata) -> User {
    #[cfg(unix)]
    {
        std::os::unix::fs::MetadataExt::uid(metadata)
    }
    #[cfg(not(unix))]
    {
        let _ = metadata;
        User
    }
}

/// What a sweep of a temp directory did (#450).
#[derive(Debug, Default, PartialEq, Eq)]
struct Swept {
    /// Directories of dead owners, removed.
    removed: usize,
    /// Directories without an `owner.lock` (0.6.3 and earlier), kept.
    unowned: usize,
    /// Their bytes.
    unowned_bytes: u64,
}

/// [`sweep_dead`] of `parent` by the user owning `own` (the open's own
/// directory, which it holds and the sweep skips), once per process for each
/// parent, with one warning about the directories it kept for want of an
/// owner lock.
fn sweep_dead_once(parent: &Path, own: &Path) {
    static SWEPT: parking_lot::Mutex<Vec<PathBuf>> = parking_lot::Mutex::new(Vec::new());
    {
        let mut swept = SWEPT.lock();
        if swept.iter().any(|done| done == parent) {
            return;
        }
        swept.push(parent.to_path_buf());
    }
    let Ok(metadata) = std::fs::symlink_metadata(own) else {
        return;
    };
    let swept = sweep_dead(parent, own, user_of(&metadata));
    if swept.unowned > 0 {
        grafeo_common::grafeo_warn!(
            "{} spill directories without an owner lock ({} bytes) are in {}: \
             `grafeo-<name>-<pid>-<random>.spill`, made by read-only opens of Grafeo 0.6.3 or \
             earlier (or on a file system without file locks); they can be deleted once no \
             process of those versions runs",
            swept.unowned,
            swept.unowned_bytes,
            parent.display()
        );
    }
}

/// Removes the read-only opens' directories in `parent` whose owner is gone:
/// those named as [`unique_name`] names them, owned by `user`, whose
/// `owner.lock` this sweep can lock; `own` (the sweeping open's directory) is
/// skipped. Links, other names, files and other users' entries are left
/// alone; so is a directory whose lock is held, cannot be opened, or is
/// missing (see [`Swept::unowned`]).
///
/// Only the user's own entries: in a shared temp directory (`/tmp`, sticky)
/// nobody else can rename or replace them, so the path the removal works on
/// stays the directory checked here, never a link swapped in after the check
/// (a non-sticky shared temp directory gives no such guarantee).
fn sweep_dead(parent: &Path, own: &Path, user: User) -> Swept {
    let mut swept = Swept::default();
    let Ok(entries) = std::fs::read_dir(parent) else {
        return swept;
    };
    for entry in entries.flatten() {
        let dir = entry.path();
        // `DirEntry::metadata` does not follow a link: a link named like ours
        // is no directory, and stays.
        let is_ours = entry.file_name().to_str().is_some_and(is_spill_name)
            && dir != own
            && entry
                .metadata()
                .is_ok_and(|metadata| metadata.is_dir() && user_of(&metadata) == user);
        if !is_ours {
            continue;
        }
        let owner = match std::fs::OpenOptions::new()
            .read(true)
            .write(true)
            .open(dir.join(OWNER_LOCK))
        {
            Ok(owner) => owner,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                swept.unowned += 1;
                swept.unowned_bytes += tree_size(&dir);
                continue;
            }
            Err(_) => continue,
        };
        // Held: the owner lives (or another process sweeps it now).
        if owner.try_lock().is_err() {
            continue;
        }
        match remove_owned(&dir, owner) {
            Ok(()) => swept.removed += 1,
            Err(error) => grafeo_common::grafeo_warn!(
                "could not remove the spill directory {} of a process that is gone: {error}",
                dir.display()
            ),
        }
    }
    swept
}

/// Whether `name` is one [`unique_name`] gives a read-only open's directory:
/// `grafeo-<name>-<pid>-<16 hex digits>.spill`.
fn is_spill_name(name: &str) -> bool {
    let Some(rest) = name
        .strip_prefix("grafeo-")
        .and_then(|rest| rest.strip_suffix(".spill"))
    else {
        return false;
    };
    let Some((rest, random)) = rest.rsplit_once('-') else {
        return false;
    };
    let Some((database, pid)) = rest.rsplit_once('-') else {
        return false;
    };
    !database.is_empty()
        && !pid.is_empty()
        && pid.bytes().all(|byte| byte.is_ascii_digit())
        && random.len() == 16
        && random.bytes().all(|byte| byte.is_ascii_hexdigit())
}

/// Removes a dead owner's directory, holding its lock (`owner`) until only
/// `owner.lock` is left: its contents (`remove_dir_all` follows no link in
/// them), then `owner.lock`, then the directory.
fn remove_owned(dir: &Path, owner: File) -> std::io::Result<()> {
    for entry in std::fs::read_dir(dir)?.flatten() {
        if entry.file_name() == OWNER_LOCK {
            continue;
        }
        let path = entry.path();
        if entry.file_type()?.is_dir() {
            std::fs::remove_dir_all(&path)?;
        } else {
            // A link to a directory is removed as a directory on Windows.
            std::fs::remove_file(&path).or_else(|_| std::fs::remove_dir(&path))?;
        }
    }
    // While locked: on Windows the file goes once the handle closes.
    std::fs::remove_file(dir.join(OWNER_LOCK))?;
    drop(owner);
    std::fs::remove_dir(dir)
}

/// The bytes of the files under `dir`, links not followed.
fn tree_size(dir: &Path) -> u64 {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return 0;
    };
    entries
        .flatten()
        .map(|entry| match entry.file_type() {
            Ok(kind) if kind.is_dir() => tree_size(&entry.path()),
            Ok(kind) if kind.is_file() => entry.metadata().map_or(0, |metadata| metadata.len()),
            _ => 0,
        })
        .sum()
}

/// Removes the vector cache a previous read-write open (or a crash) left in
/// `cache`, and its `<file>.spill` parent when that leaves it empty. The
/// caller holds the database's exclusive lock. A file that cannot be removed
/// (another process still maps it) stays: names never repeat, so nothing
/// mistakes it for one of this open's files.
pub(crate) fn remove_stale_cache(cache: &Path) {
    // An open makes `cache` a directory: a link there (a symbolic link, or a
    // junction on Windows) is not one it left, and reading through it would
    // delete the files it points at. The link itself goes, never followed.
    if is_link(cache) {
        // A link to a directory is removed as a directory on Windows.
        if let Err(error) = std::fs::remove_file(cache).or_else(|_| std::fs::remove_dir(cache)) {
            grafeo_common::grafeo_warn!(
                "could not remove the link {} at the spill cache: {error}",
                cache.display()
            );
        }
        return;
    }
    let Ok(entries) = std::fs::read_dir(cache) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        let removed = if entry.file_type().is_ok_and(|kind| kind.is_dir()) {
            std::fs::remove_dir_all(&path)
        } else {
            std::fs::remove_file(&path)
        };
        if let Err(error) = removed {
            grafeo_common::grafeo_warn!(
                "could not remove {} from the stale spill cache: {error}",
                path.display()
            );
        }
    }
    if std::fs::remove_dir(cache).is_ok()
        && let Some(parent) = cache.parent()
    {
        let _ = std::fs::remove_dir(parent);
    }
}

/// Removes what earlier opens left in a read-write open's own spill
/// directory (`<file>.spill`, under the exclusive lock the caller holds):
/// the vector cache ([`remove_stale_cache`]) and the directory of every
/// query that spilled when it crashed (`query_*`). Nothing for another
/// layout: a shared spill path holds other opens' directories, and a
/// read-only open's is new. Runs before the spill account measures the free
/// space, so leftovers neither stay nor shrink the default cap.
pub(crate) fn remove_leftovers(layout: &SpillLayout) {
    let Some(cache) = &layout.stale_cache else {
        return;
    };
    if let Some(root) = cache.parent() {
        // An open makes `<file>.spill` a directory: a link there is not one
        // it left (in a shared directory another user can plant one), and
        // reading through it would delete what it points at.
        if is_link(root) {
            grafeo_common::grafeo_warn!(
                "the spill directory {} is a link, not a directory an open made: nothing in it \
                 is removed",
                root.display()
            );
            return;
        }
        remove_query_directories(root);
    }
    remove_stale_cache(cache);
}

/// Whether `path` is a link (a symbolic link, or a junction on Windows,
/// which the standard library reports as one), never followed.
fn is_link(path: &Path) -> bool {
    std::fs::symlink_metadata(path).is_ok_and(|metadata| metadata.file_type().is_symlink())
}

/// Removes the `query_*` directories under `root` (a link there goes, not
/// followed).
fn remove_query_directories(root: &Path) {
    let Ok(entries) = std::fs::read_dir(root) else {
        return;
    };
    for entry in entries.flatten() {
        if !entry.file_name().to_string_lossy().starts_with("query_") {
            continue;
        }
        let path = entry.path();
        let removed = match entry.file_type() {
            Ok(kind) if kind.is_symlink() => {
                std::fs::remove_file(&path).or_else(|_| std::fs::remove_dir(&path))
            }
            Ok(kind) if kind.is_dir() => std::fs::remove_dir_all(&path),
            _ => continue,
        };
        if let Err(error) = removed {
            grafeo_common::grafeo_warn!(
                "could not remove the crashed query's spill directory {}: {error}",
                path.display()
            );
        }
    }
}

/// `grafeo-<name>-<pid>-<random><suffix>`: the process id keeps processes
/// apart, the random part opens of one process (and guessers).
fn unique_name(name: &str, suffix: &str) -> String {
    format!(
        "grafeo-{name}-{}-{:016x}{suffix}",
        std::process::id(),
        random_u64()
    )
}

/// A random number from what std has: a fresh `RandomState` (randomly seeded
/// per process, varied per call), hashing the time and a counter. Names need
/// it to be unpredictable and unique, not secret: the exclusive creation is
/// the protection.
pub(crate) fn random_u64() -> u64 {
    use std::hash::{BuildHasher, Hasher};
    static NEXT: AtomicU64 = AtomicU64::new(0);
    let mut hasher = std::collections::hash_map::RandomState::new().build_hasher();
    hasher.write_u64(NEXT.fetch_add(1, Ordering::Relaxed));
    hasher.write_u128(
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_or(0, |elapsed| elapsed.as_nanos()),
    );
    hasher.write_u32(std::process::id());
    hasher.finish()
}

#[cfg(test)]
#[cfg(all(feature = "vector-index", feature = "wal"))]
mod tests {
    use super::*;

    #[test]
    fn an_explicit_spill_path_gets_a_directory_per_open() {
        let shared = Path::new("shared");
        let first = SpillLayout::for_open(Some(shared), Some(Path::new("db/paris.grafeo")), false);
        let second = SpillLayout::for_open(Some(shared), None, true);
        let again = SpillLayout::for_open(Some(shared), Some(Path::new("db/paris.grafeo")), false);
        assert_eq!(first.root.as_deref(), Some(shared));
        let cache =
            |layout: &SpillLayout| layout.vector_cache.as_ref().unwrap().path().to_path_buf();
        let pid = std::process::id();
        let name = |path: PathBuf| path.file_name().unwrap().to_string_lossy().into_owned();
        assert!(
            name(cache(&first)).starts_with(&format!("grafeo-paris.grafeo-{pid}-")),
            "{:?}",
            cache(&first)
        );
        assert!(name(cache(&second)).starts_with(&format!("grafeo-memory-{pid}-")));
        assert_ne!(cache(&first), cache(&again), "a new name per open");
        assert_eq!(cache(&first).parent(), Some(shared));
        assert!(first.stale_cache.is_none() && second.stale_cache.is_none());
    }

    #[test]
    fn a_read_write_file_database_spills_beside_its_file() {
        let layout = SpillLayout::for_open(None, Some(Path::new("db/berlin.grafeo")), false);
        let spill = Path::new("db").join("berlin.grafeo.spill");
        assert_eq!(layout.root.as_deref(), Some(spill.as_path()));
        assert_eq!(
            layout.vector_cache.as_ref().unwrap().path(),
            spill.join("cache")
        );
        assert_eq!(layout.stale_cache, Some(spill.join("cache")));
        assert!(layout.root_guard.is_none());
    }

    /// A read-only open's temp directory exists from the open on, made by it,
    /// and goes with the layout.
    #[test]
    fn a_read_only_file_database_spills_to_its_own_temp_directory() {
        let layout = SpillLayout::for_open(None, Some(Path::new("db/prague.grafeo")), true);
        let root = layout.root.clone().unwrap();
        assert_eq!(root.parent(), Some(std::env::temp_dir().as_path()));
        assert!(root.is_dir());
        assert!(
            root.file_name()
                .unwrap()
                .to_string_lossy()
                .ends_with(".spill")
        );
        assert_eq!(layout.vector_cache.as_ref().unwrap().path(), root);
        assert_eq!(layout.root_guard.as_ref().unwrap().path(), root);
        assert!(layout.stale_cache.is_none());
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let mode = std::fs::metadata(&root).unwrap().permissions().mode();
            assert_eq!(mode & 0o777, 0o700, "{mode:o}");
        }
        drop(layout);
        assert!(!root.exists());
    }

    /// Only a read-only open of a file database without a spill path asks for
    /// the system temp directory: on wasm32 that call panics ("no filesystem
    /// on this platform"), so an in-memory database must never make it.
    #[test]
    fn only_a_read_only_file_open_asks_for_the_temp_directory() {
        let no_temp = || -> PathBuf { panic!("this open needs no temp directory") };
        let database = Some(Path::new("db/amsterdam.grafeo"));
        for read_only in [false, true] {
            let layout = SpillLayout::for_open_in(no_temp, None, None, read_only);
            assert!(
                layout.root.is_none(),
                "an in-memory database spills nowhere"
            );
        }
        let layout = SpillLayout::for_open_in(no_temp, None, database, false);
        assert_eq!(
            layout.root.as_deref(),
            Some(Path::new("db/amsterdam.grafeo.spill"))
        );
        let explicit = Path::new("berlin-spill");
        let layout = SpillLayout::for_open_in(no_temp, Some(explicit), database, true);
        assert_eq!(layout.root.as_deref(), Some(explicit));
    }

    /// A read-only open that cannot create its directory (here the temp
    /// directory is missing) spills nothing instead of failing the open, and
    /// creates nothing.
    #[test]
    fn a_read_only_open_without_a_usable_temp_directory_spills_nothing() {
        let dir = tempfile::tempdir().unwrap();
        let missing = dir.path().join("no-temp");
        let layout = SpillLayout::for_open_in(
            || missing.clone(),
            None,
            Some(Path::new("db/amsterdam.grafeo")),
            true,
        );
        assert!(layout.root.is_none());
        assert!(layout.root_guard.is_none());
        assert!(layout.vector_cache.is_none());
        assert!(layout.stale_cache.is_none());
        assert!(!missing.exists());
    }

    /// The temp directory is never one that already exists: a name taken
    /// first (by a directory or a file) is skipped, and when every try is
    /// taken the creation fails (and the open spills nothing).
    #[test]
    fn a_private_directory_is_created_exclusively() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::create_dir(dir.path().join("taken")).unwrap();
        std::fs::write(dir.path().join("also-taken"), b"").unwrap();
        let mut names = ["taken", "also-taken", "free"]
            .into_iter()
            .map(String::from);
        let created = create_private_dir(dir.path(), || names.next().unwrap()).unwrap();
        assert_eq!(created, dir.path().join("free"));

        let all_taken = create_private_dir(dir.path(), || "taken".to_string());
        assert_eq!(
            all_taken.unwrap_err().kind(),
            std::io::ErrorKind::AlreadyExists
        );
        assert!(
            create_private_dir(dir.path(), || unique_name("barcelona.grafeo", ".spill"))
                .unwrap()
                .is_dir()
        );
    }

    /// Makes `link` a link to the directory `target`. Windows creates symbolic
    /// links only with the privilege to (or in developer mode); a junction,
    /// which directory reads follow as well, needs none.
    fn link_directory(target: &Path, link: &Path) -> std::io::Result<()> {
        #[cfg(unix)]
        let linked = std::os::unix::fs::symlink(target, link);
        #[cfg(windows)]
        let linked = std::os::windows::fs::symlink_dir(target, link).or_else(|_| {
            let made = std::process::Command::new("cmd")
                .arg("/C")
                .arg("mklink")
                .arg("/J")
                .arg(link)
                .arg(target)
                .output()?;
            if made.status.success() {
                Ok(())
            } else {
                Err(std::io::Error::other(
                    String::from_utf8_lossy(&made.stderr).into_owned(),
                ))
            }
        });
        #[cfg(not(any(unix, windows)))]
        let linked: std::io::Result<()> = Err(std::io::Error::other("no symbolic links"));
        linked
    }

    /// A link planted at `<file>.spill` (another user of a shared directory
    /// can make one) is not followed: the open takes no spill root at all,
    /// so nothing spills through the link and the removal removes no
    /// `query_*` directory and no `cache/` from the directory it points at.
    #[test]
    fn the_leftover_removal_does_not_follow_a_linked_spill_directory() {
        let dir = tempfile::tempdir().unwrap();
        let victim = dir.path().join("victim");
        std::fs::create_dir_all(victim.join("query_7")).unwrap();
        std::fs::create_dir_all(victim.join("cache")).unwrap();
        std::fs::write(victim.join("query_7").join("vincent.bin"), b"keep").unwrap();
        std::fs::write(victim.join("cache").join("jules.bin"), b"keep").unwrap();
        let database = dir.path().join("amsterdam.grafeo");
        let spill = dir.path().join("amsterdam.grafeo.spill");
        if let Err(error) = link_directory(&victim, &spill) {
            eprintln!("skipped: this platform or user cannot create a directory link: {error}");
            return;
        }
        assert!(is_link(&spill), "the spill path is a link");

        let layout = SpillLayout::for_open(None, Some(&database), false);
        assert!(
            layout.root.is_none(),
            "a linked spill root is refused, not followed: {:?}",
            layout.root
        );
        assert!(layout.stale_cache.is_none());
        remove_leftovers(&layout);
        drop(layout);
        assert_eq!(
            std::fs::read(victim.join("query_7").join("vincent.bin")).unwrap(),
            b"keep"
        );
        assert_eq!(
            std::fs::read(victim.join("cache").join("jules.bin")).unwrap(),
            b"keep"
        );
        assert!(is_link(&spill), "the link is left as it was");
    }

    #[test]
    fn nothing_spills_without_a_path() {
        let layout = SpillLayout::for_open(None, None, false);
        assert!(layout.root.is_none() && layout.vector_cache.is_none());
    }

    /// The directory goes once its last user lets go of it, and takes an
    /// emptied `<file>.spill` with it; one with something left in it stays.
    #[test]
    fn a_spill_directory_is_removed_when_empty() {
        let dir = tempfile::tempdir().unwrap();
        let spill = dir.path().join("barcelona.grafeo.spill");
        let cache = Arc::new(SpillDirectory::new(spill.join("cache"), true));
        cache.create().unwrap();
        let user = Arc::clone(&cache);
        drop(cache);
        assert!(spill.join("cache").exists(), "still in use");
        drop(user);
        assert!(!spill.exists());

        let kept = SpillDirectory::new(spill.join("cache"), true);
        kept.create().unwrap();
        std::fs::write(spill.join("vectors_legacy.bin"), b"").unwrap();
        drop(kept);
        assert!(!spill.join("cache").exists());
        assert!(spill.exists(), "a legacy file is not ours to delete");
    }

    /// After `close()` nothing more is created in the directory.
    #[test]
    fn a_closed_directory_takes_no_more_writes() {
        let dir = tempfile::tempdir().unwrap();
        let cache = SpillDirectory::new(dir.path().join("cache"), false);
        cache.close_for_writes();
        assert!(cache.create().is_err());
        assert!(!dir.path().join("cache").exists());
    }

    /// The stale-cache removal takes what it can and leaves the rest, the
    /// 0.5.x files beside `cache/` included.
    #[test]
    fn the_stale_cache_removal_leaves_everything_outside_the_cache() {
        let dir = tempfile::tempdir().unwrap();
        let spill = dir.path().join("amsterdam.grafeo.spill");
        let cache = spill.join("cache");
        std::fs::create_dir_all(cache.join("nested")).unwrap();
        std::fs::write(cache.join("vectors_1_0_0.bin"), b"").unwrap();
        std::fs::write(spill.join("vectors_Item%3Aembedding.bin"), b"old").unwrap();

        remove_stale_cache(&cache);
        assert!(!cache.exists());
        assert_eq!(
            std::fs::read(spill.join("vectors_Item%3Aembedding.bin")).unwrap(),
            b"old"
        );
    }

    /// A symbolic link at the cache path is not followed: the stale-cache
    /// removal takes the link away and leaves every file it points at.
    #[test]
    fn the_stale_cache_removal_does_not_follow_a_link() {
        let dir = tempfile::tempdir().unwrap();
        let spill = dir.path().join("amsterdam.grafeo.spill");
        let cache = spill.join("cache");
        let elsewhere = dir.path().join("elsewhere");
        std::fs::create_dir_all(&spill).unwrap();
        std::fs::create_dir_all(elsewhere.join("nested")).unwrap();
        std::fs::write(elsewhere.join("gus.txt"), b"keep").unwrap();
        std::fs::write(elsewhere.join("nested").join("mia.txt"), b"keep").unwrap();
        if let Err(error) = link_directory(&elsewhere, &cache) {
            eprintln!("skipped: this platform or user cannot create a directory link: {error}");
            return;
        }
        assert!(
            std::fs::symlink_metadata(&cache)
                .unwrap()
                .file_type()
                .is_symlink(),
            "the cache path is a link"
        );

        remove_stale_cache(&cache);
        assert_eq!(std::fs::read(elsewhere.join("gus.txt")).unwrap(), b"keep");
        assert_eq!(
            std::fs::read(elsewhere.join("nested").join("mia.txt")).unwrap(),
            b"keep"
        );
        assert!(
            std::fs::symlink_metadata(&cache).is_err(),
            "the link itself is gone"
        );
    }

    /// The stale-cache removal goes on past an entry it cannot delete (a file
    /// another process holds on Windows, a directory without write permission
    /// on Unix): the rest of the cache still goes. The entry sorts first, so
    /// on a file system that lists by name it is met before the others.
    #[test]
    fn the_stale_cache_removal_goes_on_past_what_it_cannot_delete() {
        let dir = tempfile::tempdir().unwrap();
        let cache = dir.path().join("amsterdam.grafeo.spill").join("cache");
        std::fs::create_dir_all(&cache).unwrap();
        let held = cache.join("0-held");
        #[cfg(windows)]
        let _handle = {
            use std::os::windows::fs::OpenOptionsExt;
            std::fs::write(&held, b"3").unwrap();
            // No sharing at all: the removal cannot open it for deletion.
            std::fs::OpenOptions::new()
                .read(true)
                .share_mode(0)
                .open(&held)
                .unwrap()
        };
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            std::fs::create_dir(&held).unwrap();
            std::fs::write(held.join("vectors_3_0_0.bin"), b"3").unwrap();
            std::fs::set_permissions(&held, std::fs::Permissions::from_mode(0o500)).unwrap();
        }
        for name in [
            "vectors_1_19_0.bin",
            "vectors_1_88_0.bin",
            "vectors_1_319_0.bin",
        ] {
            std::fs::write(cache.join(name), b"19").unwrap();
        }

        remove_stale_cache(&cache);
        let left: Vec<std::ffi::OsString> = std::fs::read_dir(&cache)
            .map_or_default(|entries| entries.flatten().map(|entry| entry.file_name()).collect());
        assert!(left.iter().all(|name| name == "0-held"), "{left:?}");

        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            if held.exists() {
                std::fs::set_permissions(&held, std::fs::Permissions::from_mode(0o700)).unwrap();
            }
        }
    }

    // -----------------------------------------------------------------------------------------
    // Dead owners' directories in the temp directory (#450).
    // -----------------------------------------------------------------------------------------

    /// A directory named as a read-only open names its own, in `parent`, with a file and a
    /// nested `query_*` directory in it, and with an `owner.lock` nobody holds when
    /// `owner_lock` (without one, as 0.6.3 and earlier made them).
    fn dead_dir(parent: &Path, database: &str, owner_lock: bool) -> PathBuf {
        let dir = parent.join(unique_name(database, ".spill"));
        std::fs::create_dir_all(dir.join("query_3")).unwrap();
        std::fs::write(dir.join("columns_1_0_0.bin"), b"vincent").unwrap();
        std::fs::write(dir.join("query_3").join("run_0.bin"), b"jules").unwrap();
        if owner_lock {
            std::fs::write(dir.join(OWNER_LOCK), b"").unwrap();
        }
        dir
    }

    /// Takes `dir`'s owner lock through a handle of its own, as a live owner holds it: std's
    /// file locks conflict between two handles of one process too.
    fn hold(dir: &Path) -> File {
        let owner = std::fs::OpenOptions::new()
            .read(true)
            .write(true)
            .open(dir.join(OWNER_LOCK))
            .unwrap();
        owner.try_lock().unwrap();
        owner
    }

    /// Rows 1 to 3 of the matrix: a dead owner's directory goes, contents included; a held
    /// one stays untouched; names that are not ours stay, whatever they hold.
    #[test]
    fn the_sweep_removes_only_the_directories_of_dead_owners() {
        let temp = tempfile::tempdir().unwrap();
        let dead = dead_dir(temp.path(), "amsterdam.grafeo", true);
        let live = dead_dir(temp.path(), "berlin.grafeo", true);
        let _live_owner = hold(&live);
        let others = [
            "grafeo-x.spill",
            "grafeo-a-12-zz.spill",
            "grafeo-paris.grafeo-12-0123456789abcdeg.spill",
            "other",
        ];
        for other in others {
            std::fs::create_dir(temp.path().join(other)).unwrap();
            std::fs::write(temp.path().join(other).join(OWNER_LOCK), b"").unwrap();
        }
        let file_like_ours = temp.path().join(unique_name("prague.grafeo", ".spill"));
        std::fs::write(&file_like_ours, b"mia").unwrap();

        let swept = sweep_dead(temp.path(), &temp.path().join("own"), me(temp.path()));
        assert_eq!(
            swept,
            Swept {
                removed: 1,
                unowned: 0,
                unowned_bytes: 0
            }
        );
        assert!(
            !dead.exists(),
            "the dead owner's directory goes, contents included"
        );
        assert_eq!(
            std::fs::read(live.join("columns_1_0_0.bin")).unwrap(),
            b"vincent",
            "a live owner's directory is untouched"
        );
        assert!(live.join(OWNER_LOCK).exists());
        for other in others {
            assert!(
                temp.path().join(other).join(OWNER_LOCK).exists(),
                "{other} is not ours"
            );
        }
        assert_eq!(std::fs::read(&file_like_ours).unwrap(), b"mia");
    }

    /// Row 4: a link named like ours, pointing at a directory with an `owner.lock` nobody
    /// holds, is neither followed nor removed.
    #[test]
    fn the_sweep_leaves_a_link_named_like_ours() {
        let temp = tempfile::tempdir().unwrap();
        let target = temp.path().join("elsewhere");
        std::fs::create_dir(&target).unwrap();
        std::fs::write(target.join(OWNER_LOCK), b"").unwrap();
        std::fs::write(target.join("gus.txt"), b"keep").unwrap();
        let link = temp.path().join(unique_name("oslo.grafeo", ".spill"));
        if let Err(error) = link_directory(&target, &link) {
            eprintln!("skipped: this platform or user cannot create a directory link: {error}");
            return;
        }
        assert_eq!(
            sweep_dead(temp.path(), &temp.path().join("own"), me(temp.path())).removed,
            0
        );
        assert_eq!(std::fs::read(target.join("gus.txt")).unwrap(), b"keep");
        assert!(target.join(OWNER_LOCK).exists());
        assert!(is_link(&link), "the link is left as it was");
    }

    /// Inside a dead owner's directory, a link to a directory outside it
    /// goes as a link; what it points at stays.
    #[test]
    fn the_sweep_follows_no_link_inside_a_dead_owners_directory() {
        let temp = tempfile::tempdir().unwrap();
        let outside = temp.path().join("outside");
        std::fs::create_dir_all(outside.join("nested")).unwrap();
        std::fs::write(outside.join("nested").join("butch.txt"), b"keep").unwrap();
        let dead = dead_dir(temp.path(), "lisbon.grafeo", true);
        if let Err(error) = link_directory(&outside, &dead.join("query_9")) {
            eprintln!("skipped: this platform or user cannot create a directory link: {error}");
            return;
        }
        assert_eq!(
            sweep_dead(temp.path(), &temp.path().join("own"), me(temp.path())).removed,
            1
        );
        assert!(!dead.exists());
        assert_eq!(
            std::fs::read(outside.join("nested").join("butch.txt")).unwrap(),
            b"keep"
        );
    }

    /// Row 5: a directory without an `owner.lock` (0.6.3 and earlier) is kept, and counted
    /// with its bytes for the warning.
    #[test]
    fn the_sweep_keeps_a_directory_without_an_owner_lock() {
        let temp = tempfile::tempdir().unwrap();
        let old = dead_dir(temp.path(), "vienna.grafeo", false);
        let swept = sweep_dead(temp.path(), &temp.path().join("own"), me(temp.path()));
        assert_eq!(
            swept,
            Swept {
                removed: 0,
                unowned: 1,
                unowned_bytes: 12
            },
            "\"vincent\" and \"jules\""
        );
        assert_eq!(
            std::fs::read(old.join("columns_1_0_0.bin")).unwrap(),
            b"vincent"
        );
    }

    /// For Windows as for Unix: the sweep removes a dead owner's
    /// directory whole, `owner.lock` last, while it holds the lock it took.
    #[test]
    fn a_dead_owners_directory_goes_whole_its_lock_file_last() {
        let temp = tempfile::tempdir().unwrap();
        let dead = dead_dir(temp.path(), "madrid.grafeo", true);
        let owner = std::fs::OpenOptions::new()
            .read(true)
            .write(true)
            .open(dead.join(OWNER_LOCK))
            .unwrap();
        owner.try_lock().unwrap();
        remove_owned(&dead, owner).expect("the removal, holding the lock");
        assert!(!dead.exists());
    }

    /// The user owning `dir` (the test's own temp directory).
    fn me(dir: &Path) -> User {
        user_of(&std::fs::symlink_metadata(dir).unwrap())
    }

    /// Unix: a dead owner's directory of another user is kept (the predicate, given another
    /// uid: CI cannot make a file of another user), so no entry anyone else could swap for a
    /// link is ever removed through its path.
    #[cfg(unix)]
    #[test]
    fn the_sweep_keeps_another_users_directory() {
        let temp = tempfile::tempdir().unwrap();
        let dead = dead_dir(temp.path(), "cairo.grafeo", true);
        let someone_else = me(temp.path()).wrapping_add(1);
        let swept = sweep_dead(temp.path(), &temp.path().join("own"), someone_else);
        assert_eq!(swept.removed, 0);
        assert!(dead.join("columns_1_0_0.bin").exists());
        assert_eq!(
            sweep_dead(temp.path(), &temp.path().join("own"), me(temp.path())).removed,
            1,
            "the positive control: its own user sweeps it"
        );
    }

    /// The sweeping open's own directory is skipped, whatever its lock.
    #[test]
    fn the_sweep_skips_the_sweeping_opens_directory() {
        let temp = tempfile::tempdir().unwrap();
        let own = dead_dir(temp.path(), "dublin.grafeo", true);
        assert_eq!(sweep_dead(temp.path(), &own, me(temp.path())).removed, 0);
        assert!(own.join(OWNER_LOCK).exists());
    }

    /// A directory dropped with something left in it keeps its `owner.lock`, unlocked, and the
    /// next sweep removes it as a dead owner's; an emptied one goes whole.
    #[test]
    fn a_directory_left_with_files_is_swept_later() {
        let temp = tempfile::tempdir().unwrap();
        let (left, owner) =
            create_owned_dir(temp.path(), || unique_name("geneva.grafeo", ".spill")).unwrap();
        std::fs::write(left.join("columns_2_0_0.bin"), b"shosanna").unwrap();
        drop(SpillDirectory::owned(left.clone(), owner.unwrap()));
        assert!(
            left.join(OWNER_LOCK).exists(),
            "the lock file stays, unlocked"
        );
        assert_eq!(
            sweep_dead(temp.path(), &temp.path().join("own"), me(temp.path())),
            Swept {
                removed: 1,
                unowned: 0,
                unowned_bytes: 0
            }
        );
        assert!(!left.exists());

        let (emptied, owner) =
            create_owned_dir(temp.path(), || unique_name("geneva.grafeo", ".spill")).unwrap();
        drop(SpillDirectory::owned(emptied.clone(), owner.unwrap()));
        assert!(!emptied.exists());
    }

    /// Rows 6 and 7: a read-only open sweeps its temp directory once per process: its own
    /// directory, made after the sweep, holds its lock, and a second open in the same process
    /// does not sweep again (a dead directory made between the two stays).
    #[test]
    fn a_read_only_open_sweeps_its_temp_directory_once() {
        let temp = tempfile::tempdir().unwrap();
        let parent = temp.path().to_path_buf();
        let dead = dead_dir(&parent, "rome.grafeo", true);
        let first = SpillLayout::for_open_in(
            || parent.clone(),
            None,
            Some(Path::new("db/rome.grafeo")),
            true,
        );
        assert!(!dead.exists(), "the first read-only open swept it");
        let own = first.root.clone().unwrap();
        assert!(own.join(OWNER_LOCK).exists());
        assert!(
            std::fs::OpenOptions::new()
                .read(true)
                .write(true)
                .open(own.join(OWNER_LOCK))
                .unwrap()
                .try_lock()
                .is_err(),
            "the open holds its own lock"
        );

        let later = dead_dir(&parent, "rome.grafeo", true);
        let second = SpillLayout::for_open_in(
            || parent.clone(),
            None,
            Some(Path::new("db/rome.grafeo")),
            true,
        );
        assert!(later.exists(), "no second sweep in this process");
        assert!(
            own.exists(),
            "the first open's directory is never swept while it lives"
        );
        drop(first);
        assert!(
            !own.exists(),
            "a clean close removes its directory, owner.lock included"
        );
        drop(second);
    }

    /// Row 8: the creator's race with a sweep. It created `owner.lock` and has not locked it
    /// yet: a sweep that locked it first, or removed the directory already (Unix: a removed
    /// directory with an open file in it), makes the creator give the name up.
    #[test]
    fn a_creator_that_loses_its_directory_to_a_sweep_gives_the_name_up() {
        let temp = tempfile::tempdir().unwrap();
        let dir = temp.path().join(unique_name("athens.grafeo", ".spill"));
        std::fs::create_dir(&dir).unwrap();
        let creator = std::fs::File::create_new(dir.join(OWNER_LOCK)).unwrap();
        let sweeper = hold(&dir);
        assert!(matches!(
            lock_owner(&dir.join(OWNER_LOCK), creator),
            Owner::Lost
        ));
        drop(sweeper);

        #[cfg(unix)]
        {
            let dir = temp.path().join(unique_name("athens.grafeo", ".spill"));
            std::fs::create_dir(&dir).unwrap();
            let creator = std::fs::File::create_new(dir.join(OWNER_LOCK)).unwrap();
            std::fs::remove_dir_all(&dir).unwrap();
            assert!(matches!(
                lock_owner(&dir.join(OWNER_LOCK), creator),
                Owner::Lost
            ));
        }

        let (created, owner) =
            create_owned_dir(temp.path(), || unique_name("athens.grafeo", ".spill")).unwrap();
        assert!(owner.is_some(), "this file system takes locks");
        assert!(
            created.join(OWNER_LOCK).exists(),
            "an undisturbed creator keeps its name"
        );
    }

    /// Row 9, end to end: a read-only open whose process dies without a close leaves its
    /// directory, and the next process's first read-only open removes it.
    #[test]
    fn a_crashed_read_only_open_leaves_its_directory_to_the_next_sweep() {
        let temp = tempfile::tempdir().unwrap();
        let status = grafeo_common::testing::child_process::run(
            grafeo_common::child_test!("a_read_only_open_that_dies_without_closing")
                .env(CRASH_ROOT, temp.path()),
        )
        .unwrap();
        assert!(status.success(), "the child ran: {status}");
        let left: Vec<PathBuf> = std::fs::read_dir(temp.path())
            .unwrap()
            .flatten()
            .map(|entry| entry.path())
            .collect();
        assert_eq!(
            left.len(),
            1,
            "the dead process left its directory: {left:?}"
        );
        assert!(left[0].join("columns_1_0_0.bin").exists());
        assert!(left[0].join(OWNER_LOCK).exists());

        let parent = temp.path().to_path_buf();
        let next = SpillLayout::for_open_in(
            || parent.clone(),
            None,
            Some(Path::new("db/oslo.grafeo")),
            true,
        );
        assert!(!left[0].exists(), "the next read-only open swept it");
        drop(next);
    }

    /// Where the crash child makes its read-only open's directory.
    const CRASH_ROOT: &str = "GRAFEO_TEST_DEAD_SPILL_ROOT";

    /// The child of the test above: a read-only open that spills and dies, its destructors
    /// never run. Run alone, it does nothing.
    #[test]
    fn a_read_only_open_that_dies_without_closing() {
        let Some(root) = std::env::var_os(CRASH_ROOT).map(PathBuf::from) else {
            return;
        };
        let layout = SpillLayout::for_open_in(
            || root.clone(),
            None,
            Some(Path::new("db/oslo.grafeo")),
            true,
        );
        std::fs::write(layout.root.unwrap().join("columns_1_0_0.bin"), b"hans").unwrap();
        std::process::exit(0);
    }
}
