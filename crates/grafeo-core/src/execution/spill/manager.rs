//! Spill manager for file lifecycle management.

use super::account::{SpillAccount, SpillOwner, SpillReservation};
use super::file::SpillFile;
use parking_lot::Mutex;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

/// The bytes a spill file reserves at a time: its first ones when it is
/// created, then as its writes pass what it holds (less where the cap
/// leaves less).
pub const SPILL_FILE_STEP: u64 = 1 << 20;

/// The files a manager removes at its cleanup, shared with the files: one
/// still on disk when it drops (not deleted, or its removal failed) leaves
/// its reservation here, next to its path, until the cleanup removes it.
#[derive(Default)]
pub(super) struct ActiveFiles(Mutex<Vec<ActiveFile>>);

/// A file of [`ActiveFiles`], with the reservation it left.
struct ActiveFile {
    path: PathBuf,
    reservation: Option<SpillReservation>,
}

impl ActiveFiles {
    /// Tracks `path`, a new file.
    fn track(&self, path: PathBuf) {
        self.0.lock().push(ActiveFile {
            path,
            reservation: None,
        });
    }

    /// `path` was removed: no longer tracked.
    pub(super) fn removed(&self, path: &Path) {
        self.0.lock().retain(|file| file.path != path);
    }

    /// `path` stays on disk: its reservation is kept with it. A path already
    /// gone (a cleanup removed it while its file was still open) is no
    /// longer tracked, and its reservation goes back.
    pub(super) fn keep(&self, path: &Path, reservation: SpillReservation) {
        // The check runs under the lock: a cleanup that removes the file
        // between the check and the append would leave a reservation tracked
        // for a path that no longer exists.
        let mut files = self.0.lock();
        if std::fs::symlink_metadata(path)
            .is_err_and(|error| error.kind() == std::io::ErrorKind::NotFound)
        {
            drop(files);
            self.removed(path);
            drop(reservation);
            return;
        }
        match files.iter_mut().find(|file| file.path == path) {
            Some(file) => file.reservation = Some(reservation),
            None => files.push(ActiveFile {
                path: path.to_path_buf(),
                reservation: Some(reservation),
            }),
        }
    }

    /// The files tracked.
    fn len(&self) -> usize {
        self.0.lock().len()
    }

    /// Removes every file (one already gone counts as removed), giving back
    /// the reservations of those removed. One that cannot be removed stays
    /// tracked, with its reservation; the last error is returned.
    fn remove_all(&self) -> std::io::Result<()> {
        let files = std::mem::take(&mut *self.0.lock());
        let mut last_error = None;
        let mut kept = Vec::new();
        for file in files {
            match std::fs::remove_file(&file.path) {
                Ok(()) => {}
                Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
                Err(error) => {
                    last_error = Some(error);
                    kept.push(file);
                }
            }
        }
        if !kept.is_empty() {
            self.0.lock().append(&mut kept);
        }
        last_error.map_or(Ok(()), Err)
    }
}

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
    /// Active spill files for cleanup, shared with the files.
    active_files: Arc<ActiveFiles>,
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
    /// The account its files reserve their bytes in (`None`: unbounded).
    account: Option<Arc<SpillAccount>>,
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
            active_files: Arc::new(ActiveFiles::default()),
            total_spilled_bytes: AtomicU64::new(0),
            owns_dir: false,
            dir_ready: AtomicBool::new(false),
            dir_created: AtomicBool::new(false),
            account: None,
        })
    }

    /// Reserves the bytes of its files in `account` before they are
    /// written: a write past the account's cap or room is refused.
    #[must_use]
    pub fn with_account(mut self, account: Arc<SpillAccount>) -> Self {
        self.account = Some(account);
        self
    }

    /// The account its files reserve in.
    #[must_use]
    pub fn account(&self) -> Option<&Arc<SpillAccount>> {
        self.account.as_ref()
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

    /// Creates a new spill file of `owner` with the given prefix, and the
    /// spill directory with the first one. With an account, the file first
    /// reserves its first [`SPILL_FILE_STEP`] bytes there (probing the free
    /// space; less where the cap or the room leaves less, at least a byte),
    /// and holds the reservation as it grows.
    ///
    /// The file name format is: `{prefix}_{file_id}.spill`
    ///
    /// # Errors
    ///
    /// Returns an error if the directory or the file cannot be created, or
    /// the account refuses (an error of kind `StorageFull` carrying the
    /// refusal; nothing is created then).
    pub fn create_file(&self, owner: SpillOwner, prefix: &str) -> std::io::Result<SpillFile> {
        let reservation = match &self.account {
            Some(account) => Some(
                account
                    .reserve_up_to(owner, SPILL_FILE_STEP, 1, true)
                    .map_err(super::refused)?,
            ),
            None => None,
        };
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
        self.active_files.track(file_path.clone());

        let file = SpillFile::new(file_path)?.tracked_in(Arc::clone(&self.active_files));
        Ok(match reservation {
            Some(reservation) => file.with_reservation(reservation),
            None => file,
        })
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
        self.active_files.removed(path);
    }

    /// Returns total bytes currently spilled to disk.
    #[must_use]
    pub fn spilled_bytes(&self) -> u64 {
        self.total_spilled_bytes.load(Ordering::Relaxed)
    }

    /// Returns the number of active spill files.
    #[must_use]
    pub fn active_file_count(&self) -> usize {
        self.active_files.len()
    }

    /// Cleans up all spill files.
    ///
    /// This is called automatically on drop, but can be called manually.
    /// A file it removes gives back the reservation it left (see
    /// [`SpillFile::delete`]); one it cannot remove stays tracked, with its
    /// reservation, for the next cleanup.
    ///
    /// # Errors
    ///
    /// Returns an error if any file cannot be deleted (continues trying others).
    pub fn cleanup(&self) -> std::io::Result<()> {
        let removed = self.active_files.remove_all();
        self.total_spilled_bytes.store(0, Ordering::Relaxed);
        removed
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

        let file1 = manager.create_file(SpillOwner::Sort, "sort").unwrap();
        let file2 = manager.create_file(SpillOwner::Sort, "sort").unwrap();
        let file3 = manager.create_file(SpillOwner::Sort, "agg").unwrap();

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
        let _file1 = manager.create_file(SpillOwner::Sort, "test").unwrap();
        let _file2 = manager.create_file(SpillOwner::Sort, "test").unwrap();
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
            let file = manager.create_file(SpillOwner::Sort, "test").unwrap();
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
            let _file = manager.create_file(SpillOwner::Sort, "sort").unwrap();
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

        let file = manager.create_file(SpillOwner::Sort, "sort").unwrap();
        assert!(query_dir.is_dir());
        assert!(file.path().starts_with(&query_dir));
        let _second = manager.create_file(SpillOwner::Sort, "sort").unwrap();
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
            let _file = manager.create_file(SpillOwner::Sort, "sort").unwrap();
            let _second = manager.create_file(SpillOwner::Sort, "sort").unwrap();
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
            let _file = manager.create_file(SpillOwner::Sort, "sort").unwrap();
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
        // Process id, clock and a counter: no other run or test picks it.
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_or(0, |elapsed| elapsed.as_nanos());
        let name = format!(
            "grafeo_spill_relative_{}_{nanos}_{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        );
        spills_into_a_new_relative_directory(&name);
    }

    /// Spills into `name`, a one-component directory in the working
    /// directory that the manager must create. Its creation is exclusive
    /// (`create_dir`), so a directory that was already there fails the test
    /// and is left as it was: only a directory this test created is removed,
    /// also when an assert fails.
    fn spills_into_a_new_relative_directory(name: &str) {
        /// Removes the directory on drop, if this test created it.
        struct RemoveIfCreated {
            path: PathBuf,
            created: bool,
        }
        impl Drop for RemoveIfCreated {
            fn drop(&mut self) {
                if self.created
                    && self.path.exists()
                    && let Err(error) = std::fs::remove_dir_all(&self.path)
                {
                    eprintln!("cannot remove {}: {error}", self.path.display());
                }
            }
        }

        let relative = PathBuf::from(name);
        assert_eq!(relative.parent(), Some(Path::new("")));
        let mut cleanup = RemoveIfCreated {
            path: relative.clone(),
            created: false,
        };
        {
            let manager = SpillManager::new(&relative).unwrap().with_owned_dir();
            let file = manager.create_file(SpillOwner::Sort, "sort").unwrap();
            cleanup.created = manager.dir_created.load(Ordering::Acquire);
            assert!(
                cleanup.created,
                "the first spill file creates {name}, which was not there before"
            );
            assert!(relative.is_dir());
            assert!(file.path().starts_with(&relative));
        }
        assert!(
            !relative.exists(),
            "the owned directory is removed with the manager"
        );
    }

    /// A file whose removal fails keeps its bytes in the account (they are
    /// still on disk) until the manager's cleanup removes it.
    #[test]
    fn a_failed_removal_keeps_the_reservation_until_cleanup() {
        use crate::execution::spill::SpillCap;

        let temp_dir = TempDir::new().unwrap();
        let query_dir = temp_dir.path().join("query_3");
        let account = SpillAccount::new(SpillCap::Unlimited, None);
        let manager = SpillManager::new(&query_dir)
            .unwrap()
            .with_account(Arc::clone(&account));
        let mut file = manager.create_file(SpillOwner::Sort, "sort").unwrap();
        file.write_all(&[7; 4096]).unwrap();
        file.finish_write().unwrap();
        assert_eq!(account.used(), 4096);
        let path = file.path().to_path_buf();

        // No sharing at all on Windows, no write permission on the
        // directory on Unix: the removal fails.
        #[cfg(windows)]
        let held = {
            use std::os::windows::fs::OpenOptionsExt;
            std::fs::OpenOptions::new()
                .read(true)
                .share_mode(0)
                .open(&path)
                .unwrap()
        };
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            std::fs::set_permissions(&query_dir, std::fs::Permissions::from_mode(0o500)).unwrap();
        }
        let removed = file.delete();
        #[cfg(windows)]
        drop(held);
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            std::fs::set_permissions(&query_dir, std::fs::Permissions::from_mode(0o700)).unwrap();
        }
        if removed.is_ok() {
            eprintln!("skipped: this platform or user removed a file it should not have");
            return;
        }
        assert!(path.exists());
        assert_eq!(account.used(), 4096, "the file is still on disk");
        assert_eq!(manager.active_file_count(), 1);

        manager.cleanup().unwrap();
        assert!(!path.exists());
        assert_eq!(account.used(), 0, "the cleanup removed it");
        assert_eq!(manager.active_file_count(), 0);
    }

    /// A file dropped without a delete stays on disk until the manager's
    /// cleanup, and so do its written bytes in the account.
    #[test]
    fn a_dropped_file_keeps_its_written_bytes_until_cleanup() {
        use crate::execution::spill::SpillCap;

        let temp_dir = TempDir::new().unwrap();
        let account = SpillAccount::new(SpillCap::Unlimited, None);
        let manager = SpillManager::new(temp_dir.path().join("query_5"))
            .unwrap()
            .with_account(Arc::clone(&account));
        let mut file = manager
            .create_file(SpillOwner::Aggregate, "partition")
            .unwrap();
        file.write_all(&[5; 1000]).unwrap();
        assert_eq!(account.used(), SPILL_FILE_STEP, "a step ahead");
        let path = file.path().to_path_buf();
        drop(file);
        assert!(path.exists());
        assert_eq!(account.used(), 1000, "what it wrote, not its step");

        manager.cleanup().unwrap();
        assert!(!path.exists());
        assert_eq!(account.used(), 0);
    }

    /// A deleted file leaves the manager's list and gives its bytes back.
    #[test]
    fn a_deleted_file_is_no_longer_tracked() {
        use crate::execution::spill::SpillCap;

        let temp_dir = TempDir::new().unwrap();
        let account = SpillAccount::new(SpillCap::Unlimited, None);
        let manager = SpillManager::new(temp_dir.path())
            .unwrap()
            .with_account(Arc::clone(&account));
        let mut file = manager.create_file(SpillOwner::Sort, "sort").unwrap();
        let _other = manager.create_file(SpillOwner::Sort, "sort").unwrap();
        file.write_all(b"mia").unwrap();
        file.delete().unwrap();
        assert_eq!(manager.active_file_count(), 1);
        assert_eq!(account.used(), SPILL_FILE_STEP, "the other file's step");
    }

    /// A cleanup that removes the file of a live handle: the handle's delete
    /// finds it gone, which counts as removed, and its bytes go back.
    #[test]
    fn a_delete_after_a_cleanup_removed_the_file_gives_its_bytes_back() {
        use crate::execution::spill::SpillCap;

        let temp_dir = TempDir::new().unwrap();
        let account = SpillAccount::new(SpillCap::Unlimited, None);
        let manager = SpillManager::new(temp_dir.path().join("query_7"))
            .unwrap()
            .with_account(Arc::clone(&account));
        let mut file = manager.create_file(SpillOwner::Sort, "sort").unwrap();
        file.write_all(&[3; 2048]).unwrap();
        file.finish_write().unwrap();
        let path = file.path().to_path_buf();

        manager.cleanup().unwrap();
        assert!(!path.exists());
        file.delete()
            .expect("a file already gone counts as removed");
        assert_eq!(account.used(), 0, "nothing of it is on disk");
        assert_eq!(manager.active_file_count(), 0);
    }

    /// A cleanup that removes the file of a live handle: the handle dropped
    /// after it hands nothing to the manager, and its bytes go back.
    #[test]
    fn a_drop_after_a_cleanup_removed_the_file_gives_its_bytes_back() {
        use crate::execution::spill::SpillCap;

        let temp_dir = TempDir::new().unwrap();
        let account = SpillAccount::new(SpillCap::Unlimited, None);
        let manager = SpillManager::new(temp_dir.path().join("query_8"))
            .unwrap()
            .with_account(Arc::clone(&account));
        let mut file = manager
            .create_file(SpillOwner::Aggregate, "partition")
            .unwrap();
        file.write_all(&[4; 1000]).unwrap();
        let path = file.path().to_path_buf();

        manager.cleanup().unwrap();
        assert!(!path.exists());
        drop(file);
        assert_eq!(account.used(), 0, "nothing of it is on disk");
        assert_eq!(manager.active_file_count(), 0, "not tracked again");
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
            let _file = manager.create_file(SpillOwner::Sort, "sort").unwrap();
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
