//! Vector storage backends for different memory/performance tradeoffs.
//!
//! This module provides storage backends for vector data:
//!
//! | Backend | Memory Usage | Access Speed | Best For |
//! |---------|--------------|--------------|----------|
//! | [`RamStorage`] | High | Fastest | Small-medium datasets |
//! | [`MmapStorage`] | Low | Fast | Large datasets |
//!
//! # Example
//!
//! ```ignore
//! use grafeo_core::index::vector::storage::{MmapStorage, VectorStorage};
//! use grafeo_common::types::NodeId;
//!
//! // Create memory-mapped storage for 384-dimensional vectors
//! let storage = MmapStorage::create("vectors.bin", 384)?;
//!
//! // Store vectors
//! storage.insert(NodeId::new(1), &embedding)?;
//!
//! // Retrieve vectors (zero-copy when possible)
//! let vec = storage.get(NodeId::new(1))?;
//! ```

use grafeo_common::types::NodeId;
use parking_lot::RwLock;
use std::collections::HashMap;
#[cfg(feature = "mmap")]
use std::fs::{File, OpenOptions};
use std::io;
#[cfg(feature = "mmap")]
use std::io::{Read, Seek, SeekFrom, Write};
#[cfg(feature = "mmap")]
use std::path::{Path, PathBuf};
use std::sync::Arc;

// ============================================================================
// Storage Backend Enum
// ============================================================================

/// Storage backend configuration for vector data.
#[derive(Debug, Clone)]
pub enum StorageBackend {
    /// In-memory storage (fastest, highest memory usage).
    Ram,
    /// Memory-mapped file storage (low memory, fast access).
    #[cfg(feature = "mmap")]
    Mmap {
        /// Path to the storage file.
        path: PathBuf,
    },
}

impl Default for StorageBackend {
    fn default() -> Self {
        Self::Ram
    }
}

// ============================================================================
// Vector Storage Trait
// ============================================================================

/// Trait for vector storage backends.
pub trait VectorStorage: Send + Sync {
    /// Inserts a vector with the given ID.
    fn insert(&self, id: NodeId, vector: &[f32]) -> io::Result<()>;

    /// Retrieves a vector by ID.
    fn get(&self, id: NodeId) -> Option<Arc<[f32]>>;

    /// Checks if a vector exists.
    fn contains(&self, id: NodeId) -> bool;

    /// Removes a vector by ID.
    fn remove(&self, id: NodeId) -> bool;

    /// Returns the number of stored vectors.
    fn len(&self) -> usize;

    /// Returns true if storage is empty.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Returns the dimensions of stored vectors.
    fn dimensions(&self) -> usize;

    /// Returns estimated memory usage in bytes.
    fn memory_usage(&self) -> usize;

    /// Flushes any pending writes to disk (for persistent backends).
    fn flush(&self) -> io::Result<()>;
}

// ============================================================================
// RAM Storage (In-Memory)
// ============================================================================

/// In-memory vector storage using a HashMap.
///
/// Provides fastest access but highest memory usage.
/// Suitable for small to medium datasets that fit in RAM.
#[derive(Debug)]
pub struct RamStorage {
    vectors: RwLock<HashMap<NodeId, Arc<[f32]>>>,
    dimensions: usize,
}

impl RamStorage {
    /// Creates a new in-memory storage for vectors of the given dimensions.
    #[must_use]
    pub fn new(dimensions: usize) -> Self {
        Self {
            vectors: RwLock::new(HashMap::new()),
            dimensions,
        }
    }

    /// Creates storage with pre-allocated capacity.
    #[must_use]
    pub fn with_capacity(dimensions: usize, capacity: usize) -> Self {
        Self {
            vectors: RwLock::new(HashMap::with_capacity(capacity)),
            dimensions,
        }
    }

    /// Returns an iterator over all stored vectors.
    pub fn iter(&self) -> impl Iterator<Item = (NodeId, Arc<[f32]>)> + '_ {
        let guard = self.vectors.read();
        guard
            .iter()
            .map(|(&id, vec)| (id, Arc::clone(vec)))
            .collect::<Vec<_>>()
            .into_iter()
    }
}

impl VectorStorage for RamStorage {
    fn insert(&self, id: NodeId, vector: &[f32]) -> io::Result<()> {
        debug_assert_eq!(
            vector.len(),
            self.dimensions,
            "Vector dimension mismatch: expected {}, got {}",
            self.dimensions,
            vector.len()
        );
        let arc: Arc<[f32]> = vector.into();
        self.vectors.write().insert(id, arc);
        Ok(())
    }

    fn get(&self, id: NodeId) -> Option<Arc<[f32]>> {
        self.vectors.read().get(&id).cloned()
    }

    fn contains(&self, id: NodeId) -> bool {
        self.vectors.read().contains_key(&id)
    }

    fn remove(&self, id: NodeId) -> bool {
        self.vectors.write().remove(&id).is_some()
    }

    fn len(&self) -> usize {
        self.vectors.read().len()
    }

    fn dimensions(&self) -> usize {
        self.dimensions
    }

    fn memory_usage(&self) -> usize {
        let count = self.len();
        // Approximate: vector data + HashMap overhead
        count * self.dimensions * 4 + count * 64
    }

    fn flush(&self) -> io::Result<()> {
        // No-op for RAM storage
        Ok(())
    }
}

// ============================================================================
// Memory-Mapped Storage (requires filesystem - not available in WASM)
// ============================================================================

#[cfg(feature = "mmap")]
/// File header for mmap storage.
const MMAP_HEADER_SIZE: usize = 64;
#[cfg(feature = "mmap")]
const MMAP_MAGIC: [u8; 8] = *b"GRAFVEC1";

#[cfg(feature = "mmap")]
/// Memory-mapped vector storage backed by a file.
///
/// Vectors are stored in a binary file format:
/// - Header: magic, dimensions, count, version
/// - Index: NodeId -> offset mapping
/// - Data: packed f32 vectors
///
/// This provides low memory usage while maintaining fast access through
/// the operating system's page cache.
pub struct MmapStorage {
    /// Path to the storage file.
    path: PathBuf,
    /// Vector dimensions.
    dimensions: usize,
    /// Index mapping NodeId to file offset.
    index: RwLock<HashMap<NodeId, u64>>,
    /// File handle for reading/writing.
    file: RwLock<File>,
    /// Current write position in the data section.
    write_offset: RwLock<u64>,
    /// In-memory cache for frequently accessed vectors.
    cache: RwLock<HashMap<NodeId, Arc<[f32]>>>,
    /// Maximum cache size in entries.
    cache_limit: usize,
}

#[cfg(feature = "mmap")]
impl MmapStorage {
    /// Creates a new memory-mapped storage file.
    ///
    /// # Arguments
    ///
    /// * `path` - Path to the storage file (will be created/overwritten)
    /// * `dimensions` - Number of dimensions per vector
    pub fn create<P: AsRef<Path>>(path: P, dimensions: usize) -> io::Result<Self> {
        let path = path.as_ref().to_path_buf();

        let mut file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(true)
            .open(&path)?;

        // Write header
        let mut header = [0u8; MMAP_HEADER_SIZE];
        header[0..8].copy_from_slice(&MMAP_MAGIC);
        header[8..16].copy_from_slice(&(dimensions as u64).to_le_bytes());
        header[16..24].copy_from_slice(&0u64.to_le_bytes()); // count (initially 0)
        header[24..32].copy_from_slice(&1u64.to_le_bytes()); // version
        file.write_all(&header)?;
        file.flush()?;

        Ok(Self {
            path,
            dimensions,
            index: RwLock::new(HashMap::new()),
            file: RwLock::new(file),
            write_offset: RwLock::new(MMAP_HEADER_SIZE as u64),
            cache: RwLock::new(HashMap::new()),
            cache_limit: 10000,
        })
    }

    /// Opens an existing memory-mapped storage file.
    pub fn open<P: AsRef<Path>>(path: P) -> io::Result<Self> {
        let path = path.as_ref().to_path_buf();

        let mut file = OpenOptions::new().read(true).write(true).open(&path)?;

        // Read and validate header
        let mut header = [0u8; MMAP_HEADER_SIZE];
        file.read_exact(&mut header)?;

        if header[0..8] != MMAP_MAGIC {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "Invalid mmap storage file magic",
            ));
        }

        let dimensions = u64::from_le_bytes(header[8..16].try_into().unwrap()) as usize;
        let count = u64::from_le_bytes(header[16..24].try_into().unwrap()) as usize;

        // Read index (stored after header)
        let mut index = HashMap::with_capacity(count);
        let bytes_per_vector = dimensions * 4;

        // Calculate positions based on sequential storage
        for i in 0..count {
            let offset = MMAP_HEADER_SIZE as u64 + (i as u64) * (8 + bytes_per_vector as u64);
            file.seek(SeekFrom::Start(offset))?;

            let mut id_bytes = [0u8; 8];
            file.read_exact(&mut id_bytes)?;
            let id = NodeId::new(u64::from_le_bytes(id_bytes));

            index.insert(id, offset + 8);
        }

        let write_offset = MMAP_HEADER_SIZE as u64 + (count as u64) * (8 + bytes_per_vector as u64);

        Ok(Self {
            path,
            dimensions,
            index: RwLock::new(index),
            file: RwLock::new(file),
            write_offset: RwLock::new(write_offset),
            cache: RwLock::new(HashMap::new()),
            cache_limit: 10000,
        })
    }

    /// Sets the maximum cache size in entries.
    #[must_use]
    pub fn with_cache_limit(mut self, limit: usize) -> Self {
        self.cache_limit = limit;
        self
    }

    /// Returns the file path.
    #[must_use]
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Returns the file size in bytes.
    pub fn file_size(&self) -> io::Result<u64> {
        self.file.read().metadata().map(|m| m.len())
    }

    /// Clears the in-memory cache.
    pub fn clear_cache(&self) {
        self.cache.write().clear();
    }

    /// Updates the count in the file header.
    fn update_header_count(&self) -> io::Result<()> {
        let count = self.index.read().len() as u64;
        let mut file = self.file.write();
        file.seek(SeekFrom::Start(16))?;
        file.write_all(&count.to_le_bytes())?;
        Ok(())
    }
}

#[cfg(feature = "mmap")]
impl VectorStorage for MmapStorage {
    fn insert(&self, id: NodeId, vector: &[f32]) -> io::Result<()> {
        debug_assert_eq!(
            vector.len(),
            self.dimensions,
            "Vector dimension mismatch: expected {}, got {}",
            self.dimensions,
            vector.len()
        );

        let mut file = self.file.write();
        let mut offset = self.write_offset.write();

        // Seek to write position
        file.seek(SeekFrom::Start(*offset))?;

        // Write NodeId
        file.write_all(&id.as_u64().to_le_bytes())?;

        // Write vector data
        let bytes: Vec<u8> = vector.iter().flat_map(|f| f.to_le_bytes()).collect();
        file.write_all(&bytes)?;

        // Update index
        self.index.write().insert(id, *offset + 8);

        // Update write offset
        *offset += 8 + (self.dimensions * 4) as u64;

        // Update cache
        let mut cache = self.cache.write();
        if cache.len() >= self.cache_limit {
            // Simple eviction: clear half the cache
            let keys: Vec<_> = cache.keys().take(cache.len() / 2).copied().collect();
            for key in keys {
                cache.remove(&key);
            }
        }
        let arc: Arc<[f32]> = vector.into();
        cache.insert(id, arc);

        drop(file);
        drop(offset);
        self.update_header_count()
    }

    fn get(&self, id: NodeId) -> Option<Arc<[f32]>> {
        // Check cache first
        if let Some(vec) = self.cache.read().get(&id) {
            return Some(Arc::clone(vec));
        }

        // Read from file
        let offset = *self.index.read().get(&id)?;

        let mut file = self.file.write();
        if file.seek(SeekFrom::Start(offset)).is_err() {
            return None;
        }

        let mut bytes = vec![0u8; self.dimensions * 4];
        if file.read_exact(&mut bytes).is_err() {
            return None;
        }

        // Convert bytes to f32
        let vector: Vec<f32> = bytes
            .chunks_exact(4)
            .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
            .collect();

        let arc: Arc<[f32]> = vector.into();

        // Update cache
        let mut cache = self.cache.write();
        if cache.len() < self.cache_limit {
            cache.insert(id, Arc::clone(&arc));
        }

        Some(arc)
    }

    fn contains(&self, id: NodeId) -> bool {
        self.index.read().contains_key(&id)
    }

    fn remove(&self, id: NodeId) -> bool {
        // Remove from index and cache (data remains in file as "deleted")
        self.cache.write().remove(&id);
        self.index.write().remove(&id).is_some()
    }

    fn len(&self) -> usize {
        self.index.read().len()
    }

    fn dimensions(&self) -> usize {
        self.dimensions
    }

    fn memory_usage(&self) -> usize {
        // Index + cache
        let index_size = self.index.read().len() * 24; // NodeId + offset + overhead
        let cache_size = self.cache.read().len() * (self.dimensions * 4 + 64);
        index_size + cache_size
    }

    fn flush(&self) -> io::Result<()> {
        self.file.write().flush()
    }
}

#[cfg(feature = "mmap")]
impl std::fmt::Debug for MmapStorage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MmapStorage")
            .field("path", &self.path)
            .field("dimensions", &self.dimensions)
            .field("count", &self.len())
            .field("cache_size", &self.cache.read().len())
            .finish()
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ram_storage_basic() {
        let storage = RamStorage::new(4);

        storage
            .insert(NodeId::new(1), &[1.0, 2.0, 3.0, 4.0])
            .unwrap();
        storage
            .insert(NodeId::new(2), &[5.0, 6.0, 7.0, 8.0])
            .unwrap();

        assert_eq!(storage.len(), 2);
        assert!(storage.contains(NodeId::new(1)));
        assert!(storage.contains(NodeId::new(2)));
        assert!(!storage.contains(NodeId::new(3)));

        let vec1 = storage.get(NodeId::new(1)).unwrap();
        assert_eq!(&*vec1, &[1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_ram_storage_remove() {
        let storage = RamStorage::new(2);

        storage.insert(NodeId::new(1), &[1.0, 2.0]).unwrap();
        assert!(storage.contains(NodeId::new(1)));

        assert!(storage.remove(NodeId::new(1)));
        assert!(!storage.contains(NodeId::new(1)));
        assert!(!storage.remove(NodeId::new(1)));
    }

    #[test]
    fn test_ram_storage_memory_usage() {
        let storage = RamStorage::new(384);

        assert_eq!(storage.memory_usage(), 0);

        for i in 0..10 {
            storage.insert(NodeId::new(i + 1), &vec![0.1; 384]).unwrap();
        }

        // Should have significant memory usage
        assert!(storage.memory_usage() > 10 * 384 * 4);
    }

    #[test]
    fn test_mmap_storage_create_and_open() {
        let temp_dir = std::env::temp_dir();
        let path = temp_dir.join("test_mmap_storage.bin");

        // Clean up if exists
        let _ = std::fs::remove_file(&path);

        // Create new storage
        {
            let storage = MmapStorage::create(&path, 4).unwrap();
            storage
                .insert(NodeId::new(1), &[1.0, 2.0, 3.0, 4.0])
                .unwrap();
            storage
                .insert(NodeId::new(2), &[5.0, 6.0, 7.0, 8.0])
                .unwrap();
            storage.flush().unwrap();

            assert_eq!(storage.len(), 2);
        }

        // Reopen and verify
        {
            let storage = MmapStorage::open(&path).unwrap();
            assert_eq!(storage.len(), 2);
            assert_eq!(storage.dimensions(), 4);

            let vec1 = storage.get(NodeId::new(1)).unwrap();
            assert_eq!(&*vec1, &[1.0, 2.0, 3.0, 4.0]);

            let vec2 = storage.get(NodeId::new(2)).unwrap();
            assert_eq!(&*vec2, &[5.0, 6.0, 7.0, 8.0]);
        }

        // Clean up
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_mmap_storage_cache() {
        let temp_dir = std::env::temp_dir();
        let path = temp_dir.join("test_mmap_cache.bin");
        let _ = std::fs::remove_file(&path);

        let storage = MmapStorage::create(&path, 4).unwrap().with_cache_limit(2);

        storage
            .insert(NodeId::new(1), &[1.0, 2.0, 3.0, 4.0])
            .unwrap();
        storage
            .insert(NodeId::new(2), &[5.0, 6.0, 7.0, 8.0])
            .unwrap();

        // Both should be in cache
        assert!(storage.cache.read().contains_key(&NodeId::new(1)));
        assert!(storage.cache.read().contains_key(&NodeId::new(2)));

        // Insert more, triggering eviction
        storage
            .insert(NodeId::new(3), &[9.0, 10.0, 11.0, 12.0])
            .unwrap();

        // Cache should have been cleared partially
        assert!(storage.cache.read().len() <= 2);

        // But all vectors should still be retrievable
        assert!(storage.get(NodeId::new(1)).is_some());
        assert!(storage.get(NodeId::new(2)).is_some());
        assert!(storage.get(NodeId::new(3)).is_some());

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_mmap_storage_remove() {
        let temp_dir = std::env::temp_dir();
        let path = temp_dir.join("test_mmap_remove.bin");
        let _ = std::fs::remove_file(&path);

        let storage = MmapStorage::create(&path, 2).unwrap();

        storage.insert(NodeId::new(1), &[1.0, 2.0]).unwrap();
        assert!(storage.contains(NodeId::new(1)));

        assert!(storage.remove(NodeId::new(1)));
        assert!(!storage.contains(NodeId::new(1)));
        assert!(storage.get(NodeId::new(1)).is_none());

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_storage_backend_default() {
        let backend = StorageBackend::default();
        assert!(matches!(backend, StorageBackend::Ram));
    }

    #[test]
    fn test_mmap_storage_invalid_magic() {
        let temp_dir = std::env::temp_dir();
        let path = temp_dir.join("test_mmap_invalid.bin");
        let _ = std::fs::remove_file(&path);

        // Write invalid file (must be at least MMAP_HEADER_SIZE bytes)
        let invalid_data = [0u8; MMAP_HEADER_SIZE];
        std::fs::write(&path, invalid_data).unwrap();

        // Should fail to open due to invalid magic
        let result = MmapStorage::open(&path);
        assert!(result.is_err());

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_mmap_storage_clear_cache() {
        let temp_dir = std::env::temp_dir();
        let path = temp_dir.join("test_mmap_clear_cache.bin");
        let _ = std::fs::remove_file(&path);

        let storage = MmapStorage::create(&path, 4).unwrap();
        storage
            .insert(NodeId::new(1), &[1.0, 2.0, 3.0, 4.0])
            .unwrap();

        assert!(storage.cache.read().contains_key(&NodeId::new(1)));

        storage.clear_cache();
        assert!(storage.cache.read().is_empty());

        // Should still retrieve from file
        let vec = storage.get(NodeId::new(1)).unwrap();
        assert_eq!(&*vec, &[1.0, 2.0, 3.0, 4.0]);

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_mmap_storage_file_size() {
        let temp_dir = std::env::temp_dir();
        let path = temp_dir.join("test_mmap_file_size.bin");
        let _ = std::fs::remove_file(&path);

        let storage = MmapStorage::create(&path, 4).unwrap();
        let initial_size = storage.file_size().unwrap();
        assert_eq!(initial_size, MMAP_HEADER_SIZE as u64);

        storage
            .insert(NodeId::new(1), &[1.0, 2.0, 3.0, 4.0])
            .unwrap();
        storage.flush().unwrap();

        let after_insert = storage.file_size().unwrap();
        // Header + NodeId (8) + 4 floats (16) = 32 + 24 = 56
        assert!(after_insert > initial_size);

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_ram_storage_concurrent_access() {
        use std::sync::Arc;
        use std::thread;

        let storage = Arc::new(RamStorage::new(4));

        let handles: Vec<_> = (0..10)
            .map(|i| {
                let storage = Arc::clone(&storage);
                thread::spawn(move || {
                    for j in 0..100 {
                        let id = NodeId::new((i * 100 + j) as u64);
                        storage.insert(id, &[i as f32, j as f32, 0.0, 0.0]).unwrap();
                    }
                })
            })
            .collect();

        for handle in handles {
            handle.join().unwrap();
        }

        assert_eq!(storage.len(), 1000);
    }

    #[test]
    fn test_ram_storage_is_empty() {
        let storage = RamStorage::new(4);
        assert!(storage.is_empty());

        storage
            .insert(NodeId::new(1), &[1.0, 2.0, 3.0, 4.0])
            .unwrap();
        assert!(!storage.is_empty());
    }

    #[test]
    fn test_ram_storage_with_capacity() {
        let storage = RamStorage::with_capacity(4, 100);

        assert_eq!(storage.dimensions(), 4);
        assert!(storage.is_empty());

        // Should be able to insert without reallocation up to capacity
        for i in 0..100 {
            storage
                .insert(NodeId::new(i), &[i as f32, 0.0, 0.0, 0.0])
                .unwrap();
        }

        assert_eq!(storage.len(), 100);
    }

    #[test]
    fn test_ram_storage_iter() {
        let storage = RamStorage::new(2);

        storage.insert(NodeId::new(1), &[1.0, 2.0]).unwrap();
        storage.insert(NodeId::new(2), &[3.0, 4.0]).unwrap();
        storage.insert(NodeId::new(3), &[5.0, 6.0]).unwrap();

        let items: Vec<_> = storage.iter().collect();
        assert_eq!(items.len(), 3);

        // Verify all IDs are present (order not guaranteed)
        let ids: Vec<_> = items.iter().map(|(id, _)| id.0).collect();
        assert!(ids.contains(&1));
        assert!(ids.contains(&2));
        assert!(ids.contains(&3));

        // Verify vector content
        for (id, vec) in items {
            match id.0 {
                1 => assert_eq!(&*vec, &[1.0, 2.0]),
                2 => assert_eq!(&*vec, &[3.0, 4.0]),
                3 => assert_eq!(&*vec, &[5.0, 6.0]),
                _ => panic!("Unexpected ID: {}", id.0),
            }
        }
    }

    #[test]
    fn test_ram_storage_flush() {
        let storage = RamStorage::new(4);
        storage
            .insert(NodeId::new(1), &[1.0, 2.0, 3.0, 4.0])
            .unwrap();

        // Flush should succeed (no-op for RAM storage)
        assert!(storage.flush().is_ok());
    }

    #[test]
    fn test_mmap_storage_is_empty() {
        let temp_dir = std::env::temp_dir();
        let path = temp_dir.join("test_mmap_is_empty.bin");
        let _ = std::fs::remove_file(&path);

        let storage = MmapStorage::create(&path, 4).unwrap();
        assert!(storage.is_empty());

        storage
            .insert(NodeId::new(1), &[1.0, 2.0, 3.0, 4.0])
            .unwrap();
        assert!(!storage.is_empty());

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_mmap_storage_memory_usage() {
        let temp_dir = std::env::temp_dir();
        let path = temp_dir.join("test_mmap_memory.bin");
        let _ = std::fs::remove_file(&path);

        let storage = MmapStorage::create(&path, 4).unwrap();
        let initial_usage = storage.memory_usage();

        // Insert and read to populate cache
        storage
            .insert(NodeId::new(1), &[1.0, 2.0, 3.0, 4.0])
            .unwrap();
        let _ = storage.get(NodeId::new(1));

        let after_usage = storage.memory_usage();
        assert!(after_usage >= initial_usage);

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    #[cfg(debug_assertions)]
    #[should_panic(expected = "Vector dimension mismatch")]
    fn test_mmap_storage_dimension_mismatch() {
        let temp_dir = std::env::temp_dir();
        let path = temp_dir.join("test_mmap_dim_mismatch.bin");
        let _ = std::fs::remove_file(&path);

        let storage = MmapStorage::create(&path, 4).unwrap();

        // Inserting wrong dimensions triggers debug assertion panic
        let _ = storage.insert(NodeId::new(1), &[1.0, 2.0]); // Only 2 dims

        let _ = std::fs::remove_file(&path);
    }
}
