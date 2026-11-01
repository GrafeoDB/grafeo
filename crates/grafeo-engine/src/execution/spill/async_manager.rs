//! Async spill manager for file lifecycle management using tokio.

use super::async_file::AsyncSpillFile;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use tokio::sync::Mutex;

/// Async manager for spill file lifecycle for out-of-core processing.
///
/// The async manager handles:
/// - Creating unique spill files with prefixes using async I/O
/// - Tracking total bytes spilled to disk
/// - Automatic cleanup of all spill files on drop
///
/// By default the spill directory itself outlives the manager (the caller
/// owns it). Per-query callers that pass a unique throwaway directory should
/// chain [`with_owned_dir`](Self::with_owned_dir) so `Drop` also removes the
/// directory once its files are gone.
///
/// Code that denies deprecated items no longer builds with it:
///
/// ```compile_fail
/// #![deny(deprecated)]
/// let _size = size_of::<grafeo_engine::execution::spill::AsyncSpillManager>();
/// ```
#[deprecated(
    since = "0.6.1",
    note = "the engine spills through `grafeo_core::execution::spill::SpillManager`, use it instead; removed in 0.7.0 with the `async-storage` feature"
)]
pub struct AsyncSpillManager {
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
}

impl AsyncSpillManager {
    /// Creates a new async spill manager with the given directory.
    ///
    /// Creates the directory if it doesn't exist. The directory is *not*
    /// removed on drop unless [`with_owned_dir`](Self::with_owned_dir) is
    /// chained on the result.
    ///
    /// # Errors
    ///
    /// Returns an error if the directory cannot be created.
    pub async fn new(spill_dir: impl Into<PathBuf>) -> std::io::Result<Self> {
        let spill_dir = spill_dir.into();
        tokio::fs::create_dir_all(&spill_dir).await?;

        Ok(Self {
            spill_dir,
            next_file_id: AtomicU64::new(0),
            active_files: Mutex::new(Vec::new()),
            total_spilled_bytes: AtomicU64::new(0),
            owns_dir: false,
        })
    }

    /// Marks the spill directory as owned by this manager so that `Drop`
    /// removes it (non-recursive) after spill files are cleaned up.
    ///
    /// Use for per-query spill subdirectories where leaving the empty
    /// directory behind would accumulate over time. The removal is
    /// best-effort: if anything unexpected is left in the directory,
    /// `remove_dir` fails and the directory is preserved.
    #[must_use]
    pub fn with_owned_dir(mut self) -> Self {
        self.owns_dir = true;
        self
    }

    /// Creates a new async spill manager using a system temp directory.
    ///
    /// # Errors
    ///
    /// Returns an error if the temp directory cannot be created.
    pub async fn with_temp_dir() -> std::io::Result<Self> {
        let temp_dir = std::env::temp_dir().join("grafeo_spill");
        Self::new(temp_dir).await
    }

    /// Returns the spill directory path.
    #[must_use]
    pub fn spill_dir(&self) -> &Path {
        &self.spill_dir
    }

    /// Creates a new async spill file with the given prefix.
    ///
    /// The file name format is: `{prefix}_{file_id}.spill`
    ///
    /// # Errors
    ///
    /// Returns an error if the file cannot be created.
    pub async fn create_file(&self, prefix: &str) -> std::io::Result<AsyncSpillFile> {
        let file_id = self.next_file_id.fetch_add(1, Ordering::Relaxed);
        let file_name = format!("{prefix}_{file_id}.spill");
        let file_path = self.spill_dir.join(file_name);

        // Track the file for cleanup
        self.active_files.lock().await.push(file_path.clone());

        AsyncSpillFile::new(file_path).await
    }

    /// Registers bytes spilled to disk.
    ///
    /// Called by AsyncSpillFile when writing completes.
    pub fn register_spilled_bytes(&self, bytes: u64) {
        self.total_spilled_bytes.fetch_add(bytes, Ordering::Relaxed);
    }

    /// Unregisters bytes when a spill file is deleted.
    ///
    /// Called by AsyncSpillFile on deletion.
    pub fn unregister_spilled_bytes(&self, bytes: u64) {
        self.total_spilled_bytes.fetch_sub(bytes, Ordering::Relaxed);
    }

    /// Removes a file path from tracking (called when file is deleted).
    pub async fn unregister_file(&self, path: &Path) {
        let mut files = self.active_files.lock().await;
        files.retain(|p| p != path);
    }

    /// Returns total bytes currently spilled to disk.
    #[must_use]
    pub fn spilled_bytes(&self) -> u64 {
        self.total_spilled_bytes.load(Ordering::Relaxed)
    }

    /// Returns the number of active spill files.
    pub async fn active_file_count(&self) -> usize {
        self.active_files.lock().await.len()
    }

    /// Cleans up all spill files asynchronously.
    ///
    /// This can be called manually for async cleanup.
    ///
    /// # Errors
    ///
    /// Returns an error if any file cannot be deleted (continues trying others).
    pub async fn cleanup(&self) -> std::io::Result<()> {
        let files = std::mem::take(&mut *self.active_files.lock().await);
        let mut last_error = None;

        for path in files {
            if let Err(e) = tokio::fs::remove_file(&path).await {
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

    /// Synchronous cleanup for use in Drop.
    ///
    /// This is a best-effort cleanup that blocks on file deletion.
    fn cleanup_sync(&self) {
        // Use std::fs for synchronous cleanup in Drop
        if let Ok(mut files) = self.active_files.try_lock() {
            let files_to_remove = std::mem::take(&mut *files);
            for path in files_to_remove {
                let _ = std::fs::remove_file(&path);
            }
        }
    }
}

impl Drop for AsyncSpillManager {
    fn drop(&mut self) {
        // Best-effort synchronous cleanup on drop
        self.cleanup_sync();
        if self.owns_dir {
            // Non-recursive remove_dir: succeeds only if cleanup_sync left
            // the directory empty. Anything untracked keeps the directory.
            let _ = std::fs::remove_dir(&self.spill_dir);
        }
    }
}

impl std::fmt::Debug for AsyncSpillManager {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AsyncSpillManager")
            .field("spill_dir", &self.spill_dir)
            .field("spilled_bytes", &self.spilled_bytes())
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[tokio::test]
    async fn test_async_manager_creation() {
        let temp_dir = TempDir::new().unwrap();
        let manager = AsyncSpillManager::new(temp_dir.path()).await.unwrap();

        assert_eq!(manager.spilled_bytes(), 0);
        assert_eq!(manager.active_file_count().await, 0);
        assert_eq!(manager.spill_dir(), temp_dir.path());
    }

    #[tokio::test]
    async fn test_async_create_spill_file() {
        let temp_dir = TempDir::new().unwrap();
        let manager = AsyncSpillManager::new(temp_dir.path()).await.unwrap();

        let file1 = manager.create_file("sort").await.unwrap();
        let file2 = manager.create_file("sort").await.unwrap();
        let file3 = manager.create_file("agg").await.unwrap();

        assert_eq!(manager.active_file_count().await, 3);

        // File names should be unique
        assert_ne!(file1.path(), file2.path());
        assert!(file1.path().to_str().unwrap().contains("sort_0"));
        assert!(file2.path().to_str().unwrap().contains("sort_1"));
        assert!(file3.path().to_str().unwrap().contains("agg_2"));
    }

    #[tokio::test]
    async fn test_async_cleanup() {
        let temp_dir = TempDir::new().unwrap();
        let manager = AsyncSpillManager::new(temp_dir.path()).await.unwrap();

        // Create some files
        let _file1 = manager.create_file("test").await.unwrap();
        let _file2 = manager.create_file("test").await.unwrap();
        assert_eq!(manager.active_file_count().await, 2);

        // Cleanup should remove all files
        manager.cleanup().await.unwrap();
        assert_eq!(manager.active_file_count().await, 0);
    }

    #[tokio::test]
    async fn test_async_spilled_bytes_tracking() {
        let temp_dir = TempDir::new().unwrap();
        let manager = AsyncSpillManager::new(temp_dir.path()).await.unwrap();

        manager.register_spilled_bytes(1000);
        manager.register_spilled_bytes(500);
        assert_eq!(manager.spilled_bytes(), 1500);

        manager.unregister_spilled_bytes(300);
        assert_eq!(manager.spilled_bytes(), 1200);
    }

    #[tokio::test]
    async fn test_async_cleanup_on_drop() {
        let temp_dir = TempDir::new().unwrap();
        let temp_path = temp_dir.path().to_path_buf();

        let file_path = {
            let manager = AsyncSpillManager::new(&temp_path).await.unwrap();
            let file = manager.create_file("test").await.unwrap();
            file.path().to_path_buf()
        };

        // After manager is dropped, the file should be cleaned up
        assert!(!file_path.exists());
    }
}
