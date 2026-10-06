//! Where an open spills, and who removes what (#594).
//!
//! | Open | Spill path (query and section spills) | Vector cache |
//! | --- | --- | --- |
//! | explicit [`Config::spill_path`](crate::config::Config::spill_path) | the spill path | `<spill_path>/grafeo-<name>-<pid>-<random>/` |
//! | read-write file database | `<file>.spill` | `<file>.spill/cache/` |
//! | read-only file database | `<temp>/grafeo-<name>-<pid>-<random>.spill` | the same directory |
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
//! open warns and spills nothing. The other directories are created only
//! when something spills. Each is removed
//! (when empty) once the last spilled column reading from it is gone.
//! `<file>.spill/kept/` is not spilled data: it holds the old spill files of
//! an older build that an open kept instead of folding them in completely
//! (see `legacy_spill`), and nothing here reads or removes it.

use std::path::{Path, PathBuf};
use std::sync::Arc;
#[cfg(all(
    feature = "lpg",
    feature = "vector-index",
    feature = "mmap",
    not(feature = "temporal")
))]
use std::sync::atomic::AtomicBool;
use std::sync::atomic::{AtomicU64, Ordering};

/// How many names a read-only open tries for its temp directory before it
/// gives up: each is random, so a clash means someone made them on purpose.
const NAME_ATTEMPTS: usize = 16;

/// A directory an open spills into, removed when the last user lets go of
/// it (only when it is empty: what is still in it was not ours to delete).
pub(crate) struct SpillDirectory {
    path: PathBuf,
    /// Also remove the parent when that leaves it empty: `<file>.spill`
    /// around `cache/`.
    remove_empty_parent: bool,
    /// Set by `close()`: nothing more is written here (see the module docs).
    #[cfg(all(
        feature = "lpg",
        feature = "vector-index",
        feature = "mmap",
        not(feature = "temporal")
    ))]
    closed: AtomicBool,
}

impl SpillDirectory {
    fn new(path: PathBuf, remove_empty_parent: bool) -> Self {
        Self {
            path,
            remove_empty_parent,
            #[cfg(all(
                feature = "lpg",
                feature = "vector-index",
                feature = "mmap",
                not(feature = "temporal")
            ))]
            closed: AtomicBool::new(false),
        }
    }
}

// What the vector cache uses (a build without it only removes directories).
#[cfg(all(
    feature = "lpg",
    feature = "vector-index",
    feature = "mmap",
    not(feature = "temporal")
))]
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
    #[cfg(all(
        feature = "lpg",
        feature = "vector-index",
        feature = "mmap",
        not(feature = "temporal")
    ))]
    pub(crate) vector_cache: Option<Arc<SpillDirectory>>,
    /// A cache a previous read-write open left behind, removed at open.
    pub(crate) stale_cache: Option<PathBuf>,
}

impl SpillLayout {
    /// The layout of an open with an explicit `spill_path`, or one derived
    /// from `database` (`None` for an in-memory database, or one that must
    /// not spill). A read-only open's directory is created here, in the
    /// system temp directory.
    pub(crate) fn for_open(
        spill_path: Option<&Path>,
        database: Option<&Path>,
        read_only: bool,
    ) -> Self {
        Self::for_open_in(&std::env::temp_dir(), spill_path, database, read_only)
    }

    /// [`for_open`](Self::for_open) with the directory a read-only open
    /// creates its own in. When that fails (the temp directory is missing or
    /// read-only, or every name tried was taken), the open warns and spills
    /// nothing, as an in-memory database without a spill path: it never
    /// falls back to a directory it did not create.
    fn for_open_in(
        temp_parent: &Path,
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
                #[cfg(all(
                    feature = "lpg",
                    feature = "vector-index",
                    feature = "mmap",
                    not(feature = "temporal")
                ))]
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
                all(
                    feature = "lpg",
                    feature = "vector-index",
                    feature = "mmap",
                    not(feature = "temporal")
                )
            )) {
                return Self::none();
            }
            let created = match create_private_dir(temp_parent, || unique_name(&name, ".spill")) {
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
            let own = Arc::new(SpillDirectory::new(created, false));
            return Self {
                root: Some(own.path.clone()),
                root_guard: Some(Arc::clone(&own)),
                #[cfg(all(
                    feature = "lpg",
                    feature = "vector-index",
                    feature = "mmap",
                    not(feature = "temporal")
                ))]
                vector_cache: Some(Arc::clone(&own)),
                stale_cache: None,
            };
        }
        let cache = derived.join("cache");
        Self {
            root: Some(derived),
            root_guard: None,
            #[cfg(all(
                feature = "lpg",
                feature = "vector-index",
                feature = "mmap",
                not(feature = "temporal")
            ))]
            vector_cache: Some(Arc::new(SpillDirectory::new(cache.clone(), true))),
            stale_cache: Some(cache),
        }
    }

    fn none() -> Self {
        Self {
            root: None,
            root_guard: None,
            #[cfg(all(
                feature = "lpg",
                feature = "vector-index",
                feature = "mmap",
                not(feature = "temporal")
            ))]
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

/// Removes the vector cache a previous read-write open (or a crash) left in
/// `cache`, and its `<file>.spill` parent when that leaves it empty. The
/// caller holds the database's exclusive lock. A file that cannot be removed
/// (another process still maps it) stays: names never repeat, so nothing
/// mistakes it for one of this open's files.
pub(crate) fn remove_stale_cache(cache: &Path) {
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
#[cfg(all(
    feature = "lpg",
    feature = "vector-index",
    feature = "mmap",
    not(feature = "temporal")
))]
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

    /// A read-only open that cannot create its directory (here the temp
    /// directory is missing) spills nothing instead of failing the open, and
    /// creates nothing.
    #[test]
    fn a_read_only_open_without_a_usable_temp_directory_spills_nothing() {
        let dir = tempfile::tempdir().unwrap();
        let missing = dir.path().join("no-temp");
        let layout =
            SpillLayout::for_open_in(&missing, None, Some(Path::new("db/amsterdam.grafeo")), true);
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
            .map(|entries| entries.flatten().map(|entry| entry.file_name()).collect())
            .unwrap_or_default();
        assert!(left.iter().all(|name| name == "0-held"), "{left:?}");

        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            if held.exists() {
                std::fs::set_permissions(&held, std::fs::Permissions::from_mode(0o700)).unwrap();
            }
        }
    }
}
