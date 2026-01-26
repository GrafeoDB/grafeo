//! Epoch-based arena allocator for structural sharing.
//!
//! The arena allocator is the core memory management strategy for Graphos.
//! It provides:
//! - Fast bump-pointer allocation
//! - Epoch-based versioning for MVCC
//! - Bulk deallocation when epochs become unreachable

use crate::types::EpochId;
use parking_lot::RwLock;
use std::alloc::{alloc, dealloc, Layout};
use std::cell::UnsafeCell;
use std::ptr::NonNull;
use std::sync::atomic::{AtomicUsize, Ordering};

/// Default chunk size for arena allocations (1 MB).
const DEFAULT_CHUNK_SIZE: usize = 1024 * 1024;

/// A memory chunk in the arena.
struct Chunk {
    /// Pointer to the start of the chunk.
    ptr: NonNull<u8>,
    /// Total capacity of the chunk.
    capacity: usize,
    /// Current allocation offset.
    offset: AtomicUsize,
}

impl Chunk {
    /// Creates a new chunk with the given capacity.
    fn new(capacity: usize) -> Self {
        let layout = Layout::from_size_align(capacity, 16).expect("Invalid layout");
        // SAFETY: We're allocating a valid layout
        let ptr = unsafe { alloc(layout) };
        let ptr = NonNull::new(ptr).expect("Allocation failed");

        Self {
            ptr,
            capacity,
            offset: AtomicUsize::new(0),
        }
    }

    /// Tries to allocate `size` bytes with the given alignment.
    /// Returns None if there's not enough space.
    fn try_alloc(&self, size: usize, align: usize) -> Option<NonNull<u8>> {
        loop {
            let current = self.offset.load(Ordering::Relaxed);

            // Calculate aligned offset
            let aligned = (current + align - 1) & !(align - 1);
            let new_offset = aligned + size;

            if new_offset > self.capacity {
                return None;
            }

            // Try to reserve the space
            match self.offset.compare_exchange_weak(
                current,
                new_offset,
                Ordering::AcqRel,
                Ordering::Relaxed,
            ) {
                Ok(_) => {
                    // SAFETY: We've reserved this range exclusively
                    let ptr = unsafe { self.ptr.as_ptr().add(aligned) };
                    return NonNull::new(ptr);
                }
                Err(_) => continue, // Retry
            }
        }
    }

    /// Returns the amount of memory used in this chunk.
    fn used(&self) -> usize {
        self.offset.load(Ordering::Relaxed)
    }

    /// Returns the remaining capacity in this chunk.
    fn remaining(&self) -> usize {
        self.capacity - self.used()
    }
}

impl Drop for Chunk {
    fn drop(&mut self) {
        let layout = Layout::from_size_align(self.capacity, 16).expect("Invalid layout");
        // SAFETY: We allocated this memory with the same layout
        unsafe { dealloc(self.ptr.as_ptr(), layout) };
    }
}

// SAFETY: Chunk uses atomic operations for thread-safe allocation
unsafe impl Send for Chunk {}
unsafe impl Sync for Chunk {}

/// An epoch-aware arena allocator.
///
/// The arena allocates memory in large chunks and uses bump-pointer allocation
/// for fast allocation. Each arena is associated with an epoch, enabling
/// structural sharing for MVCC.
pub struct Arena {
    /// The epoch this arena belongs to.
    epoch: EpochId,
    /// List of memory chunks.
    chunks: RwLock<Vec<Chunk>>,
    /// Default chunk size for new allocations.
    chunk_size: usize,
    /// Total bytes allocated.
    total_allocated: AtomicUsize,
}

impl Arena {
    /// Creates a new arena for the given epoch.
    #[must_use]
    pub fn new(epoch: EpochId) -> Self {
        Self::with_chunk_size(epoch, DEFAULT_CHUNK_SIZE)
    }

    /// Creates a new arena with a custom chunk size.
    #[must_use]
    pub fn with_chunk_size(epoch: EpochId, chunk_size: usize) -> Self {
        let initial_chunk = Chunk::new(chunk_size);
        Self {
            epoch,
            chunks: RwLock::new(vec![initial_chunk]),
            chunk_size,
            total_allocated: AtomicUsize::new(chunk_size),
        }
    }

    /// Returns the epoch this arena belongs to.
    #[must_use]
    pub fn epoch(&self) -> EpochId {
        self.epoch
    }

    /// Allocates `size` bytes with the given alignment.
    ///
    /// # Panics
    ///
    /// Panics if allocation fails (out of memory).
    pub fn alloc(&self, size: usize, align: usize) -> NonNull<u8> {
        // First try to allocate from existing chunks
        {
            let chunks = self.chunks.read();
            for chunk in chunks.iter().rev() {
                if let Some(ptr) = chunk.try_alloc(size, align) {
                    return ptr;
                }
            }
        }

        // Need a new chunk
        self.alloc_new_chunk(size, align)
    }

    /// Allocates a value of type T.
    pub fn alloc_value<T>(&self, value: T) -> &mut T {
        let ptr = self.alloc(std::mem::size_of::<T>(), std::mem::align_of::<T>());
        // SAFETY: We've allocated the correct size and alignment
        unsafe {
            let typed_ptr = ptr.as_ptr() as *mut T;
            typed_ptr.write(value);
            &mut *typed_ptr
        }
    }

    /// Allocates a slice of values.
    pub fn alloc_slice<T: Copy>(&self, values: &[T]) -> &mut [T] {
        if values.is_empty() {
            return &mut [];
        }

        let size = std::mem::size_of::<T>() * values.len();
        let align = std::mem::align_of::<T>();
        let ptr = self.alloc(size, align);

        // SAFETY: We've allocated the correct size and alignment
        unsafe {
            let typed_ptr = ptr.as_ptr() as *mut T;
            std::ptr::copy_nonoverlapping(values.as_ptr(), typed_ptr, values.len());
            std::slice::from_raw_parts_mut(typed_ptr, values.len())
        }
    }

    /// Allocates a new chunk and performs the allocation.
    fn alloc_new_chunk(&self, size: usize, align: usize) -> NonNull<u8> {
        let chunk_size = self.chunk_size.max(size + align);
        let chunk = Chunk::new(chunk_size);

        self.total_allocated
            .fetch_add(chunk_size, Ordering::Relaxed);

        let ptr = chunk
            .try_alloc(size, align)
            .expect("Fresh chunk should have space");

        let mut chunks = self.chunks.write();
        chunks.push(chunk);

        ptr
    }

    /// Returns the total memory allocated by this arena.
    #[must_use]
    pub fn total_allocated(&self) -> usize {
        self.total_allocated.load(Ordering::Relaxed)
    }

    /// Returns the total memory used (not just allocated capacity).
    #[must_use]
    pub fn total_used(&self) -> usize {
        let chunks = self.chunks.read();
        chunks.iter().map(Chunk::used).sum()
    }

    /// Returns statistics about this arena.
    #[must_use]
    pub fn stats(&self) -> ArenaStats {
        let chunks = self.chunks.read();
        ArenaStats {
            epoch: self.epoch,
            chunk_count: chunks.len(),
            total_allocated: self.total_allocated.load(Ordering::Relaxed),
            total_used: chunks.iter().map(Chunk::used).sum(),
        }
    }
}

/// Statistics about an arena.
#[derive(Debug, Clone)]
pub struct ArenaStats {
    /// The epoch this arena belongs to.
    pub epoch: EpochId,
    /// Number of chunks allocated.
    pub chunk_count: usize,
    /// Total bytes allocated.
    pub total_allocated: usize,
    /// Total bytes used.
    pub total_used: usize,
}

/// The global arena allocator managing multiple epochs.
pub struct ArenaAllocator {
    /// Map of epochs to arenas.
    arenas: RwLock<hashbrown::HashMap<EpochId, Arena>>,
    /// Current epoch.
    current_epoch: AtomicUsize,
    /// Default chunk size.
    chunk_size: usize,
}

impl ArenaAllocator {
    /// Creates a new arena allocator.
    #[must_use]
    pub fn new() -> Self {
        Self::with_chunk_size(DEFAULT_CHUNK_SIZE)
    }

    /// Creates a new arena allocator with a custom chunk size.
    #[must_use]
    pub fn with_chunk_size(chunk_size: usize) -> Self {
        let allocator = Self {
            arenas: RwLock::new(hashbrown::HashMap::new()),
            current_epoch: AtomicUsize::new(0),
            chunk_size,
        };

        // Create the initial epoch
        let epoch = EpochId::INITIAL;
        allocator
            .arenas
            .write()
            .insert(epoch, Arena::with_chunk_size(epoch, chunk_size));

        allocator
    }

    /// Returns the current epoch.
    #[must_use]
    pub fn current_epoch(&self) -> EpochId {
        EpochId::new(self.current_epoch.load(Ordering::Acquire) as u64)
    }

    /// Creates a new epoch and returns its ID.
    pub fn new_epoch(&self) -> EpochId {
        let new_id = self.current_epoch.fetch_add(1, Ordering::AcqRel) as u64 + 1;
        let epoch = EpochId::new(new_id);

        let arena = Arena::with_chunk_size(epoch, self.chunk_size);
        self.arenas.write().insert(epoch, arena);

        epoch
    }

    /// Gets the arena for a specific epoch.
    ///
    /// # Panics
    ///
    /// Panics if the epoch doesn't exist.
    pub fn arena(&self, epoch: EpochId) -> impl std::ops::Deref<Target = Arena> + '_ {
        parking_lot::RwLockReadGuard::map(self.arenas.read(), |arenas| {
            arenas.get(&epoch).expect("Epoch should exist")
        })
    }

    /// Allocates in the current epoch.
    pub fn alloc(&self, size: usize, align: usize) -> NonNull<u8> {
        let epoch = self.current_epoch();
        let arenas = self.arenas.read();
        arenas.get(&epoch).expect("Current epoch exists").alloc(size, align)
    }

    /// Drops an epoch, freeing all its memory.
    ///
    /// This should only be called when no readers are using this epoch.
    pub fn drop_epoch(&self, epoch: EpochId) {
        self.arenas.write().remove(&epoch);
    }

    /// Returns total memory allocated across all epochs.
    #[must_use]
    pub fn total_allocated(&self) -> usize {
        self.arenas
            .read()
            .values()
            .map(Arena::total_allocated)
            .sum()
    }
}

impl Default for ArenaAllocator {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_arena_basic_allocation() {
        let arena = Arena::new(EpochId::INITIAL);

        // Allocate some bytes
        let ptr1 = arena.alloc(100, 8);
        let ptr2 = arena.alloc(200, 8);

        // Pointers should be different
        assert_ne!(ptr1.as_ptr(), ptr2.as_ptr());
    }

    #[test]
    fn test_arena_value_allocation() {
        let arena = Arena::new(EpochId::INITIAL);

        let value = arena.alloc_value(42u64);
        assert_eq!(*value, 42);

        *value = 100;
        assert_eq!(*value, 100);
    }

    #[test]
    fn test_arena_slice_allocation() {
        let arena = Arena::new(EpochId::INITIAL);

        let slice = arena.alloc_slice(&[1u32, 2, 3, 4, 5]);
        assert_eq!(slice, &[1, 2, 3, 4, 5]);

        slice[0] = 10;
        assert_eq!(slice[0], 10);
    }

    #[test]
    fn test_arena_large_allocation() {
        let arena = Arena::with_chunk_size(EpochId::INITIAL, 1024);

        // Allocate something larger than the chunk size
        let ptr = arena.alloc(2048, 8);
        assert!(!ptr.as_ptr().is_null());

        // Should have created a new chunk
        assert!(arena.stats().chunk_count >= 2);
    }

    #[test]
    fn test_arena_allocator_epochs() {
        let allocator = ArenaAllocator::new();

        let epoch0 = allocator.current_epoch();
        assert_eq!(epoch0, EpochId::INITIAL);

        let epoch1 = allocator.new_epoch();
        assert_eq!(epoch1, EpochId::new(1));

        let epoch2 = allocator.new_epoch();
        assert_eq!(epoch2, EpochId::new(2));

        // Current epoch should be the latest
        assert_eq!(allocator.current_epoch(), epoch2);
    }

    #[test]
    fn test_arena_allocator_allocation() {
        let allocator = ArenaAllocator::new();

        let ptr1 = allocator.alloc(100, 8);
        let ptr2 = allocator.alloc(100, 8);

        assert_ne!(ptr1.as_ptr(), ptr2.as_ptr());
    }

    #[test]
    fn test_arena_drop_epoch() {
        let allocator = ArenaAllocator::new();

        let initial_mem = allocator.total_allocated();

        let epoch1 = allocator.new_epoch();
        // Allocate some memory in the new epoch
        {
            let arena = allocator.arena(epoch1);
            arena.alloc(10000, 8);
        }

        let after_alloc = allocator.total_allocated();
        assert!(after_alloc > initial_mem);

        // Drop the epoch
        allocator.drop_epoch(epoch1);

        // Memory should decrease
        let after_drop = allocator.total_allocated();
        assert!(after_drop < after_alloc);
    }

    #[test]
    fn test_arena_stats() {
        let arena = Arena::with_chunk_size(EpochId::new(5), 4096);

        let stats = arena.stats();
        assert_eq!(stats.epoch, EpochId::new(5));
        assert_eq!(stats.chunk_count, 1);
        assert_eq!(stats.total_allocated, 4096);
        assert_eq!(stats.total_used, 0);

        arena.alloc(100, 8);
        let stats = arena.stats();
        assert!(stats.total_used >= 100);
    }
}
