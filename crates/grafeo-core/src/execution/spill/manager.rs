//! Spill manager for file lifecycle management.

use super::file::SpillFile;
use parking_lot::Mutex;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

/// Manages spill file lifecycle for out-of-core processing.
///
/// The manager handles:
/// - Creating unique spill files with prefixes
/// - Tracking total bytes spilled to disk
/// - Automatic cleanup of all spill files on drop
///
/// By default the spill directory itself outlives the manager (the caller
/// owns it). Per-query callers that pass a unique throwaway directory should
/// chain [`with_owned_dir`](Self::with_owned_dir) so `Drop` also removes the
/// directory once its files are gone.
pub struct SpillManager {
    /// Directory for spill files.
    spill_dir: PathBuf,
    /// Counter for unique file IDs.
    next_file_id: AtomicU64,
    /// Active spill file paths for cleanup.
    active_files: Mutex<Vec<PathBuf>>,
    /// Total bytes currently spilled to disk.
    total_spilled_bytes: AtomicU64,
    /// Whether `Drop` should remove `spill_dir` itself (non-recursive).
    owns_dir: bool,
    /// Whether `spill_dir` is known to exist (checked with the first spill
    /// file).
    dir_ready: AtomicBool,
    /// Whether this manager created `spill_dir` (on its first spill file);
    /// a directory that was already there is never removed.
    dir_created: AtomicBool,
}

impl SpillManager {
    /// Creates a new spill manager with the given directory.
    ///
    /// Touches no disk: the directory is created with the first spill file
    /// ([`create_file`](Self::create_file)), so a query that spills nothing
    /// costs no filesystem calls. The directory is *not* removed on drop
    /// unless [`with_owned_dir`](Self::with_owned_dir) is chained on the
    /// result.
    ///
    /// # Errors
    ///
    /// Never fails today; the `Result` stays for callers that handle a
    /// directory error.
    pub fn new(spill_dir: impl Into<PathBuf>) -> std::io::Result<Self> {
        Ok(Self {
            spill_dir: spill_dir.into(),
            next_file_id: AtomicU64::new(0),
            active_files: Mutex::new(Vec::new()),
            total_spilled_bytes: AtomicU64::new(0),
            owns_dir: false,
            dir_ready: AtomicBool::new(false),
            dir_created: AtomicBool::new(false),
        })
    }

    /// Marks the spill directory as owned by this manager so that `Drop`
    /// removes it (non-recursive) after spill files are cleaned up, if the
    /// manager created it.
    ///
    /// Use for per-query spill subdirectories (e.g. `<base>/query_<id>/`)
    /// where leaving the empty directory behind would accumulate over time.
    /// The removal is best-effort: if anything unexpected is left in the
    /// directory, `remove_dir` fails and the directory is preserved.
    #[must_use]
    pub fn with_owned_dir(mut self) -> Self {
        self.owns_dir = true;
        self
    }

    /// Creates a new spill manager using a system temp directory.
    ///
    /// # Errors
    ///
    /// Returns an error if the temp directory cannot be created.
    pub fn with_temp_dir() -> std::io::Result<Self> {
        let temp_dir = std::env::temp_dir().join("grafeo_spill");
        Self::new(temp_dir)
    }

    /// Returns the spill directory path.
    #[must_use]
    pub fn spill_dir(&self) -> &Path {
        &self.spill_dir
    }

    /// Creates a new spill file with the given prefix, and the spill
    /// directory with the first one.
    ///
    /// The file name format is: `{prefix}_{file_id}.spill`
    ///
    /// # Errors
    ///
    /// Returns an error if the directory or the file cannot be created.
    pub fn create_file(&self, prefix: &str) -> std::io::Result<SpillFile> {
        if !self.dir_ready.load(Ordering::Acquire) {
            if let Some(parent) = self.spill_dir.parent() {
                std::fs::create_dir_all(parent)?;
            }
            // Only the call that creates the directory marks it as this
            // manager's: one that existed before, or that another thread of
            // this manager created first, is `AlreadyExists` here.
            match std::fs::create_dir(&self.spill_dir) {
                Ok(()) => self.dir_created.store(true, Ordering::Release),
                Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {}
                Err(error) => return Err(error),
            }
            self.dir_ready.store(true, Ordering::Release);
        }
        let file_id = self.next_file_id.fetch_add(1, Ordering::Relaxed);
        let file_name = format!("{prefix}_{file_id}.spill");
        let file_path = self.spill_dir.join(file_name);

        // Track the file for cleanup
        self.active_files.lock().push(file_path.clone());

        SpillFile::new(file_path)
    }

    /// Registers bytes spilled to disk.
    ///
    /// Called by SpillFile when writing completes.
    pub fn register_spilled_bytes(&self, bytes: u64) {
        self.total_spilled_bytes.fetch_add(bytes, Ordering::Relaxed);
    }

    /// Unregisters bytes when a spill file is deleted.
    ///
    /// Called by SpillFile on deletion.
    pub fn unregister_spilled_bytes(&self, bytes: u64) {
        self.total_spilled_bytes.fetch_sub(bytes, Ordering::Relaxed);
    }

    /// Removes a file path from tracking (called when file is deleted).
    pub fn unregister_file(&self, path: &Path) {
        let mut files = self.active_files.lock();
        files.retain(|p| p != path);
    }

    /// Returns total bytes currently spilled to disk.
    #[must_use]
    pub fn spilled_bytes(&self) -> u64 {
        self.total_spilled_bytes.load(Ordering::Relaxed)
    }

    /// Returns the number of active spill files.
    #[must_use]
    pub fn active_file_count(&self) -> usize {
        self.active_files.lock().len()
    }

    /// Cleans up all spill files.
    ///
    /// This is called automatically on drop, but can be called manually.
    ///
    /// # Errors
    ///
    /// Returns an error if any file cannot be deleted (continues trying others).
    pub fn cleanup(&self) -> std::io::Result<()> {
        let files = std::mem::take(&mut *self.active_files.lock());
        let mut last_error = None;

        for path in files {
            if let Err(e) = std::fs::remove_file(&path) {
                // Ignore "not found" errors (file may have been deleted already)
                if e.kind() != std::io::ErrorKind::NotFound {
                    last_error = Some(e);
                }
            }
        }

        self.total_spilled_bytes.store(0, Ordering::Relaxed);

        match last_error {
            Some(e) => Err(e),
            None => Ok(()),
        }
    }
}

impl Drop for SpillManager {
    fn drop(&mut self) {
        // Best-effort cleanup on drop
        let _ = self.cleanup();
        if self.owns_dir && self.dir_created.load(Ordering::Acquire) {
            // Non-recursive remove_dir: succeeds only if cleanup left the
            // directory empty. If something else (a stray file, a subdir we
            // didn't track) is in there, the directory is preserved.
            let _ = std::fs::remove_dir(&self.spill_dir);
        }
    }
}

impl std::fmt::Debug for SpillManager {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SpillManager")
            .field("spill_dir", &self.spill_dir)
            .field("active_files", &self.active_file_count())
            .field("spilled_bytes", &self.spilled_bytes())
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_manager_creation() {
        let temp_dir = TempDir::new().unwrap();
        let manager = SpillManager::new(temp_dir.path()).unwrap();

        assert_eq!(manager.spilled_bytes(), 0);
        assert_eq!(manager.active_file_count(), 0);
        assert_eq!(manager.spill_dir(), temp_dir.path());
    }

    #[test]
    fn test_create_spill_file() {
        let temp_dir = TempDir::new().unwrap();
        let manager = SpillManager::new(temp_dir.path()).unwrap();

        let file1 = manager.create_file("sort").unwrap();
        let file2 = manager.create_file("sort").unwrap();
        let file3 = manager.create_file("agg").unwrap();

        assert_eq!(manager.active_file_count(), 3);

        // File names should be unique
        assert_ne!(file1.path(), file2.path());
        assert!(file1.path().to_str().unwrap().contains("sort_0"));
        assert!(file2.path().to_str().unwrap().contains("sort_1"));
        assert!(file3.path().to_str().unwrap().contains("agg_2"));
    }

    #[test]
    fn test_cleanup() {
        let temp_dir = TempDir::new().unwrap();
        let manager = SpillManager::new(temp_dir.path()).unwrap();

        // Create some files
        let _file1 = manager.create_file("test").unwrap();
        let _file2 = manager.create_file("test").unwrap();
        assert_eq!(manager.active_file_count(), 2);

        // Cleanup should remove all files
        manager.cleanup().unwrap();
        assert_eq!(manager.active_file_count(), 0);
    }

    #[test]
    fn test_spilled_bytes_tracking() {
        let temp_dir = TempDir::new().unwrap();
        let manager = SpillManager::new(temp_dir.path()).unwrap();

        manager.register_spilled_bytes(1000);
        manager.register_spilled_bytes(500);
        assert_eq!(manager.spilled_bytes(), 1500);

        manager.unregister_spilled_bytes(300);
        assert_eq!(manager.spilled_bytes(), 1200);
    }

    #[test]
    fn test_cleanup_on_drop() {
        let temp_dir = TempDir::new().unwrap();
        let temp_path = temp_dir.path().to_path_buf();

        let file_path = {
            let manager = SpillManager::new(&temp_path).unwrap();
            let file = manager.create_file("test").unwrap();
            file.path().to_path_buf()
        };

        // After manager is dropped, the file should be cleaned up
        assert!(!file_path.exists());
    }

    #[test]
    fn unowned_dir_survives_drop() {
        // Default behavior: caller owns the directory, manager leaves it
        // alone on drop. Per-query subdirs should opt into ownership; shared
        // / caller-managed dirs should not.
        let temp_dir = TempDir::new().unwrap();
        let dir_path = temp_dir.path().join("shared_spill");

        {
            let manager = SpillManager::new(&dir_path).unwrap();
            let _file = manager.create_file("sort").unwrap();
        }

        assert!(
            dir_path.exists(),
            "default SpillManager must not remove its directory on drop"
        );
    }

    #[test]
    fn the_directory_is_created_by_the_first_spill_file() {
        // A query that spills nothing must not touch the disk (#565): every
        // statement on a file database created and removed its directory.
        let temp_dir = TempDir::new().unwrap();
        let query_dir = temp_dir.path().join("base").join("query_7");

        let manager = SpillManager::new(&query_dir).unwrap().with_owned_dir();
        assert!(!query_dir.exists(), "new must not create the directory");
        assert!(!temp_dir.path().join("base").exists());

        let file = manager.create_file("sort").unwrap();
        assert!(query_dir.is_dir());
        assert!(file.path().starts_with(&query_dir));
        let _second = manager.create_file("sort").unwrap();
        assert_eq!(manager.active_file_count(), 2);
    }

    #[test]
    fn an_owned_directory_never_created_is_left_alone() {
        // Dropping a manager that never spilled removes nothing, not even a
        // directory of the same name that someone else created meanwhile.
        let temp_dir = TempDir::new().unwrap();
        let query_dir = temp_dir.path().join("query_8");
        {
            let _manager = SpillManager::new(&query_dir).unwrap().with_owned_dir();
            std::fs::create_dir(&query_dir).unwrap();
        }
        assert!(query_dir.exists());
    }

    #[test]
    fn an_owned_directory_that_already_existed_survives_a_spill() {
        // The manager removes only a directory its own first spill file
        // created: one that was there before is someone else's, empty or not.
        let temp_dir = TempDir::new().unwrap();
        let query_dir = temp_dir.path().join("query_19");
        std::fs::create_dir(&query_dir).unwrap();
        {
            let manager = SpillManager::new(&query_dir).unwrap().with_owned_dir();
            let _file = manager.create_file("sort").unwrap();
            let _second = manager.create_file("sort").unwrap();
        }
        assert!(
            query_dir.is_dir(),
            "the directory that existed before the spill is left in place"
        );
    }

    #[test]
    fn owned_dir_removed_after_files_cleaned_on_drop() {
        // Regression test for the per-query spill leak (#323 follow-up):
        // session-created `<base>/query_<id>/` subdirs accumulated empty
        // because Drop only removed files, not the directory itself.
        let temp_dir = TempDir::new().unwrap();
        let query_dir = temp_dir.path().join("query_42");

        {
            let manager = SpillManager::new(&query_dir).unwrap().with_owned_dir();
            let _file = manager.create_file("sort").unwrap();
            assert!(query_dir.exists());
        }

        assert!(
            !query_dir.exists(),
            "with_owned_dir manager must remove its empty directory on drop"
        );
    }

    /// A spill directory given as one relative component (`spill`) has an
    /// empty parent, which `create_dir_all` takes as already there: the first
    /// spill file creates the directory in the working directory.
    #[test]
    fn a_one_component_relative_directory_spills() {
        /// Removes the directory even when an assert fails, so a failing run
        /// leaves nothing in the working directory.
        struct RemoveOnDrop(PathBuf);
        impl Drop for RemoveOnDrop {
            fn drop(&mut self) {
                if self.0.exists()
                    && let Err(error) = std::fs::remove_dir_all(&self.0)
                {
                    eprintln!("cannot remove {}: {error}", self.0.display());
                }
            }
        }

        let name = format!("grafeo_spill_relative_{}", std::process::id());
        let relative = PathBuf::from(&name);
        assert_eq!(relative.parent(), Some(Path::new("")));
        let _cleanup = RemoveOnDrop(relative.clone());
        {
            let manager = SpillManager::new(&relative).unwrap().with_owned_dir();
            let file = manager.create_file("sort").unwrap();
            assert!(relative.is_dir(), "the first spill file creates {name}");
            assert!(file.path().starts_with(&relative));
        }
        assert!(
            !relative.exists(),
            "the owned directory is removed with the manager"
        );
    }

    #[test]
    fn owned_dir_preserved_when_unexpected_contents_remain() {
        // remove_dir is non-recursive on purpose: if something the manager
        // did not track is sitting in the directory, the directory survives
        // rather than being silently deleted.
        let temp_dir = TempDir::new().unwrap();
        let query_dir = temp_dir.path().join("query_with_extra");

        let stray_file = {
            let manager = SpillManager::new(&query_dir).unwrap().with_owned_dir();
            let _file = manager.create_file("sort").unwrap();
            let stray = query_dir.join("not_tracked.dat");
            std::fs::write(&stray, b"keep me").unwrap();
            stray
        };

        assert!(
            query_dir.exists(),
            "directory with untracked content must not be removed"
        );
        assert!(
            stray_file.exists(),
            "untracked file must not be touched by SpillManager Drop"
        );
    }
}
