//! Epoch-based arena allocator for MVCC.
//!
//! This is how Grafeo manages memory for versioned data. Each epoch gets its
//! own arena, and when all readers from an old epoch finish, we free the whole
//! thing at once. Much faster than tracking individual allocations.
//!
//! Use [`ArenaAllocator`] to manage multiple epochs, or [`Arena`] directly
//! if you're working with a single epoch.

// Arena allocators require unsafe code for memory management
#![allow(unsafe_code)]

use std::alloc::{Layout, alloc, dealloc};
use std::ptr::NonNull;
use std::sync::atomic::{AtomicUsize, Ordering};

use parking_lot::RwLock;

use crate::types::EpochId;

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
        self.try_alloc_with_offset(size, align).map(|(_, ptr)| ptr)
    }

    /// Tries to allocate `size` bytes with the given alignment.
    /// Returns (offset, ptr) where offset is the aligned offset within this chunk.
    /// Returns None if there's not enough space.
    fn try_alloc_with_offset(&self, size: usize, align: usize) -> Option<(u32, NonNull<u8>)> {
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
                    return Some((aligned as u32, NonNull::new(ptr)?));
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
    #[allow(dead_code)]
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

/// A single epoch's memory arena.
///
/// Allocates by bumping a pointer forward - extremely fast. You can't free
/// individual allocations; instead, drop the whole arena when the epoch
/// is no longer needed.
///
/// Thread-safe: multiple threads can allocate concurrently using atomics.
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

    /// Allocates a value and returns its offset within the primary chunk.
    ///
    /// This is used by tiered storage to store values in the arena and track
    /// their locations via compact u32 offsets in `HotVersionRef`.
    ///
    /// # Panics
    ///
    /// Panics if allocation would require a new chunk. Ensure chunk size is
    /// large enough for your use case.
    #[cfg(feature = "tiered-storage")]
    pub fn alloc_value_with_offset<T>(&self, value: T) -> (u32, &mut T) {
        let size = std::mem::size_of::<T>();
        let align = std::mem::align_of::<T>();

        // Try to allocate in the first chunk to get a stable offset
        let chunks = self.chunks.read();
        let chunk = chunks
            .first()
            .expect("Arena should have at least one chunk");

        let (offset, ptr) = chunk
            .try_alloc_with_offset(size, align)
            .expect("Allocation would create new chunk - increase chunk size");

        // SAFETY: We've allocated the correct size and alignment
        unsafe {
            let typed_ptr = ptr.as_ptr().cast::<T>();
            typed_ptr.write(value);
            (offset, &mut *typed_ptr)
        }
    }

    /// Reads a value at the given offset in the primary chunk.
    ///
    /// # Safety
    ///
    /// - The offset must have been returned by a previous `alloc_value_with_offset` call
    /// - The type T must match what was stored at that offset
    /// - The arena must not have been dropped
    #[cfg(feature = "tiered-storage")]
    pub unsafe fn read_at<T>(&self, offset: u32) -> &T {
        let chunks = self.chunks.read();
        let chunk = chunks
            .first()
            .expect("Arena should have at least one chunk");

        debug_assert!(
            (offset as usize) + std::mem::size_of::<T>() <= chunk.used(),
            "read_at: offset {} + size_of::<{}>() = {} exceeds chunk used bytes {}",
            offset,
            std::any::type_name::<T>(),
            (offset as usize) + std::mem::size_of::<T>(),
            chunk.used()
        );
        debug_assert!(
            (offset as usize).is_multiple_of(std::mem::align_of::<T>()),
            "read_at: offset {} is not aligned for {} (alignment {})",
            offset,
            std::any::type_name::<T>(),
            std::mem::align_of::<T>()
        );

        // SAFETY: Caller guarantees offset is valid and T matches stored type
        unsafe {
            let ptr = chunk.ptr.as_ptr().add(offset as usize).cast::<T>();
            &*ptr
        }
    }

    /// Reads a value mutably at the given offset in the primary chunk.
    ///
    /// # Safety
    ///
    /// - The offset must have been returned by a previous `alloc_value_with_offset` call
    /// - The type T must match what was stored at that offset
    /// - The arena must not have been dropped
    /// - No other references to this value may exist
    #[cfg(feature = "tiered-storage")]
    pub unsafe fn read_at_mut<T>(&self, offset: u32) -> &mut T {
        let chunks = self.chunks.read();
        let chunk = chunks
            .first()
            .expect("Arena should have at least one chunk");

        debug_assert!(
            (offset as usize) + std::mem::size_of::<T>() <= chunk.capacity,
            "read_at_mut: offset {} + size_of::<{}>() = {} exceeds chunk capacity {}",
            offset,
            std::any::type_name::<T>(),
            (offset as usize) + std::mem::size_of::<T>(),
            chunk.capacity
        );
        debug_assert!(
            (offset as usize).is_multiple_of(std::mem::align_of::<T>()),
            "read_at_mut: offset {} is not aligned for {} (alignment {})",
            offset,
            std::any::type_name::<T>(),
            std::mem::align_of::<T>()
        );

        // SAFETY: Caller guarantees offset is valid, T matches, and no aliasing
        unsafe {
            let ptr = chunk.ptr.as_ptr().add(offset as usize).cast::<T>();
            &mut *ptr
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

/// Manages arenas across multiple epochs.
///
/// Use this to create new epochs, allocate in the current epoch, and
/// clean up old epochs when they're no longer needed.
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

    /// Ensures an arena exists for the given epoch, creating it if necessary.
    /// Returns whether a new arena was created.
    #[cfg(feature = "tiered-storage")]
    pub fn ensure_epoch(&self, epoch: EpochId) -> bool {
        // Fast path: check if epoch already exists
        {
            let arenas = self.arenas.read();
            if arenas.contains_key(&epoch) {
                return false;
            }
        }

        // Slow path: create the epoch
        let mut arenas = self.arenas.write();
        // Double-check after acquiring write lock
        if arenas.contains_key(&epoch) {
            return false;
        }

        let arena = Arena::with_chunk_size(epoch, self.chunk_size);
        arenas.insert(epoch, arena);
        true
    }

    /// Gets or creates an arena for a specific epoch.
    #[cfg(feature = "tiered-storage")]
    pub fn arena_or_create(&self, epoch: EpochId) -> impl std::ops::Deref<Target = Arena> + '_ {
        self.ensure_epoch(epoch);
        self.arena(epoch)
    }

    /// Allocates in the current epoch.
    pub fn alloc(&self, size: usize, align: usize) -> NonNull<u8> {
        let epoch = self.current_epoch();
        let arenas = self.arenas.read();
        arenas
            .get(&epoch)
            .expect("Current epoch exists")
            .alloc(size, align)
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
        let _ptr = arena.alloc(2048, 8);

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

#[cfg(all(test, feature = "tiered-storage"))]
mod tiered_storage_tests {
    use super::*;

    #[test]
    fn test_alloc_value_with_offset_basic() {
        let arena = Arena::with_chunk_size(EpochId::INITIAL, 4096);

        let (offset1, val1) = arena.alloc_value_with_offset(42u64);
        let (offset2, val2) = arena.alloc_value_with_offset(100u64);

        // First allocation should be at offset 0 (aligned)
        assert_eq!(offset1, 0);
        // Second allocation should be after the first
        assert!(offset2 > offset1);
        assert!(offset2 >= std::mem::size_of::<u64>() as u32);

        // Values should be correct
        assert_eq!(*val1, 42);
        assert_eq!(*val2, 100);

        // Mutation should work
        *val1 = 999;
        assert_eq!(*val1, 999);
    }

    #[test]
    fn test_read_at_basic() {
        let arena = Arena::with_chunk_size(EpochId::INITIAL, 4096);

        let (offset, _) = arena.alloc_value_with_offset(12345u64);

        // Read it back
        let value: &u64 = unsafe { arena.read_at(offset) };
        assert_eq!(*value, 12345);
    }

    #[test]
    fn test_read_at_mut_basic() {
        let arena = Arena::with_chunk_size(EpochId::INITIAL, 4096);

        let (offset, _) = arena.alloc_value_with_offset(42u64);

        // Read and modify
        let value: &mut u64 = unsafe { arena.read_at_mut(offset) };
        assert_eq!(*value, 42);
        *value = 100;

        // Verify modification persisted
        let value: &u64 = unsafe { arena.read_at(offset) };
        assert_eq!(*value, 100);
    }

    #[test]
    fn test_alloc_value_with_offset_struct() {
        #[derive(Debug, Clone, PartialEq)]
        struct TestNode {
            id: u64,
            name: [u8; 32],
            value: i32,
        }

        let arena = Arena::with_chunk_size(EpochId::INITIAL, 4096);

        let node = TestNode {
            id: 12345,
            name: [b'A'; 32],
            value: -999,
        };

        let (offset, stored) = arena.alloc_value_with_offset(node.clone());
        assert_eq!(stored.id, 12345);
        assert_eq!(stored.value, -999);

        // Read it back
        let read: &TestNode = unsafe { arena.read_at(offset) };
        assert_eq!(read.id, node.id);
        assert_eq!(read.name, node.name);
        assert_eq!(read.value, node.value);
    }

    #[test]
    fn test_alloc_value_with_offset_alignment() {
        let arena = Arena::with_chunk_size(EpochId::INITIAL, 4096);

        // Allocate a byte first to potentially misalign
        let (offset1, _) = arena.alloc_value_with_offset(1u8);
        assert_eq!(offset1, 0);

        // Now allocate a u64 which requires 8-byte alignment
        let (offset2, val) = arena.alloc_value_with_offset(42u64);

        // offset2 should be 8-byte aligned
        assert_eq!(offset2 % 8, 0);
        assert_eq!(*val, 42);
    }

    #[test]
    fn test_alloc_value_with_offset_multiple() {
        let arena = Arena::with_chunk_size(EpochId::INITIAL, 4096);

        let mut offsets = Vec::new();
        for i in 0..100u64 {
            let (offset, val) = arena.alloc_value_with_offset(i);
            offsets.push(offset);
            assert_eq!(*val, i);
        }

        // All offsets should be unique and in ascending order
        for window in offsets.windows(2) {
            assert!(window[0] < window[1]);
        }

        // Read all values back
        for (i, offset) in offsets.iter().enumerate() {
            let val: &u64 = unsafe { arena.read_at(*offset) };
            assert_eq!(*val, i as u64);
        }
    }

    #[test]
    fn test_arena_allocator_with_offset() {
        let allocator = ArenaAllocator::with_chunk_size(4096);

        let epoch = allocator.current_epoch();
        let arena = allocator.arena(epoch);

        let (offset, val) = arena.alloc_value_with_offset(42u64);
        assert_eq!(*val, 42);

        let read: &u64 = unsafe { arena.read_at(offset) };
        assert_eq!(*read, 42);
    }

    #[test]
    #[cfg(debug_assertions)]
    #[should_panic(expected = "exceeds chunk used bytes")]
    fn test_read_at_out_of_bounds() {
        let arena = Arena::with_chunk_size(EpochId::INITIAL, 4096);
        let (_offset, _) = arena.alloc_value_with_offset(42u64);

        // Read way past the allocated region — should panic in debug
        unsafe {
            let _: &u64 = arena.read_at(4000);
        }
    }

    #[test]
    #[cfg(debug_assertions)]
    #[should_panic(expected = "is not aligned")]
    fn test_read_at_misaligned() {
        let arena = Arena::with_chunk_size(EpochId::INITIAL, 4096);
        // Allocate a u8 at offset 0
        let (_offset, _) = arena.alloc_value_with_offset(0xFFu8);
        // Also allocate some bytes so offset 1 is within used range
        let _ = arena.alloc_value_with_offset(0u64);

        // Try to read a u64 at offset 1 (misaligned for u64)
        unsafe {
            let _: &u64 = arena.read_at(1);
        }
    }

    #[test]
    #[cfg(not(miri))] // parking_lot uses integer-to-pointer casts incompatible with Miri strict provenance
    fn test_concurrent_read_stress() {
        use std::sync::Arc;

        let arena = Arc::new(Arena::with_chunk_size(EpochId::INITIAL, 1024 * 1024));
        let num_threads = 8;
        let values_per_thread = 1000;

        // Each thread allocates values and records offsets
        let mut all_offsets = Vec::new();
        for t in 0..num_threads {
            let base = (t * values_per_thread) as u64;
            let mut offsets = Vec::with_capacity(values_per_thread);
            for i in 0..values_per_thread as u64 {
                let (offset, _) = arena.alloc_value_with_offset(base + i);
                offsets.push(offset);
            }
            all_offsets.push(offsets);
        }

        // Now read all values back concurrently from multiple threads
        let mut handles = Vec::new();
        for (t, offsets) in all_offsets.into_iter().enumerate() {
            let arena = Arc::clone(&arena);
            let base = (t * values_per_thread) as u64;
            handles.push(std::thread::spawn(move || {
                for (i, offset) in offsets.iter().enumerate() {
                    let val: &u64 = unsafe { arena.read_at(*offset) };
                    assert_eq!(*val, base + i as u64);
                }
            }));
        }

        for handle in handles {
            handle.join().expect("Thread panicked");
        }
    }

    #[test]
    fn test_multi_type_interleaved() {
        #[derive(Debug, Clone, PartialEq)]
        #[repr(C)]
        struct Record {
            id: u64,
            flags: u32,
            weight: f32,
        }

        let arena = Arena::with_chunk_size(EpochId::INITIAL, 4096);

        // Interleave different types
        let (off_u8, _) = arena.alloc_value_with_offset(0xAAu8);
        let (off_u32, _) = arena.alloc_value_with_offset(0xBBBBu32);
        let (off_u64, _) = arena.alloc_value_with_offset(0xCCCCCCCCu64);
        let (off_rec, _) = arena.alloc_value_with_offset(Record {
            id: 42,
            flags: 0xFF,
            weight: 3.14,
        });

        // Read them all back
        unsafe {
            assert_eq!(*arena.read_at::<u8>(off_u8), 0xAA);
            assert_eq!(*arena.read_at::<u32>(off_u32), 0xBBBB);
            assert_eq!(*arena.read_at::<u64>(off_u64), 0xCCCCCCCC);

            let rec: &Record = arena.read_at(off_rec);
            assert_eq!(rec.id, 42);
            assert_eq!(rec.flags, 0xFF);
            assert!((rec.weight - 3.14).abs() < 0.001);
        }
    }
}
