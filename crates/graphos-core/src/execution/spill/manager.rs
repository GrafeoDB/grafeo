//! Spill manager for file lifecycle management.

use super::file::SpillFile;
use parking_lot::Mutex;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

/// Manages spill file lifecycle for out-of-core processing.
///
/// The manager handles:
/// - Creating unique spill files with prefixes
/// - Tracking total bytes spilled to disk
/// - Automatic cleanup of all spill files on drop
pub struct SpillManager {
    /// Directory for spill files.
    spill_dir: PathBuf,
    /// Counter for unique file IDs.
    next_file_id: AtomicU64,
    /// Active spill file paths for cleanup.
    active_files: Mutex<Vec<PathBuf>>,
    /// Total bytes currently spilled to disk.
    total_spilled_bytes: AtomicU64,
}

impl SpillManager {
    /// Creates a new spill manager with the given directory.
    ///
    /// Creates the directory if it doesn't exist.
    ///
    /// # Errors
    ///
    /// Returns an error if the directory cannot be created.
    pub fn new(spill_dir: impl Into<PathBuf>) -> std::io::Result<Self> {
        let spill_dir = spill_dir.into();
        std::fs::create_dir_all(&spill_dir)?;

        Ok(Self {
            spill_dir,
            next_file_id: AtomicU64::new(0),
            active_files: Mutex::new(Vec::new()),
            total_spilled_bytes: AtomicU64::new(0),
        })
    }

    /// Creates a new spill manager using a system temp directory.
    ///
    /// # Errors
    ///
    /// Returns an error if the temp directory cannot be created.
    pub fn with_temp_dir() -> std::io::Result<Self> {
        let temp_dir = std::env::temp_dir().join("graphos_spill");
        Self::new(temp_dir)
    }

    /// Returns the spill directory path.
    #[must_use]
    pub fn spill_dir(&self) -> &Path {
        &self.spill_dir
    }

    /// Creates a new spill file with the given prefix.
    ///
    /// The file name format is: `{prefix}_{file_id}.spill`
    ///
    /// # Errors
    ///
    /// Returns an error if the file cannot be created.
    pub fn create_file(&self, prefix: &str) -> std::io::Result<SpillFile> {
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
}
