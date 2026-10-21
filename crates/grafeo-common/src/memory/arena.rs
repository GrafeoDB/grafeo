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
use std::fmt;
use std::ptr::NonNull;
use std::sync::atomic::{AtomicUsize, Ordering};

use parking_lot::RwLock;

use crate::types::EpochId;

/// Default chunk size for arena allocations (1 MB).
const DEFAULT_CHUNK_SIZE: usize = 1024 * 1024;

/// Errors from arena allocation operations.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub enum AllocError {
    /// The system allocator returned null (out of memory).
    OutOfMemory,
    /// The requested epoch does not exist.
    EpochNotFound(EpochId),
    /// Arena chunk has insufficient space for the allocation.
    InsufficientSpace,
    /// Alignment must be a non-zero power of two.
    InvalidAlignment(usize),
    /// Every `u32` id of the kind `what` (a named graph's, say) was given
    /// out: ids are never reused, so no more can be created.
    IdsExhausted {
        /// What the ids number, for the message.
        what: &'static str,
    },
}

impl fmt::Display for AllocError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::OutOfMemory => write!(f, "arena allocation failed: out of memory"),
            Self::EpochNotFound(id) => write!(f, "epoch {id} not found in arena allocator"),
            Self::InsufficientSpace => {
                write!(f, "arena chunk has insufficient space for allocation")
            }
            Self::InvalidAlignment(align) => {
                write!(f, "alignment must be a non-zero power of two, got {align}")
            }
            Self::IdsExhausted { what } => {
                write!(
                    f,
                    "every {what} id was given out, so no {what} can be created"
                )
            }
        }
    }
}

impl std::error::Error for AllocError {}

impl From<AllocError> for crate::Error {
    fn from(e: AllocError) -> Self {
        match e {
            AllocError::OutOfMemory | AllocError::InsufficientSpace => {
                crate::Error::Storage(crate::utils::error::StorageError::Full)
            }
            AllocError::EpochNotFound(id) => {
                crate::Error::Internal(format!("epoch {id} not found in arena allocator"))
            }
            AllocError::InvalidAlignment(align) => crate::Error::Internal(format!(
                "alignment must be a non-zero power of two, got {align}"
            )),
            AllocError::IdsExhausted { .. } => crate::Error::InvalidValue(e.to_string()),
        }
    }
}

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
    ///
    /// # Errors
    ///
    /// Returns `AllocError::OutOfMemory` if the system allocator fails.
    fn new(capacity: usize) -> Result<Self, AllocError> {
        if capacity > u32::MAX as usize {
            return Err(AllocError::OutOfMemory);
        }
        let layout = Layout::from_size_align(capacity, 16).map_err(|_| AllocError::OutOfMemory)?;
        // SAFETY: We're allocating a valid layout
        let ptr = unsafe { alloc(layout) };
        let ptr = NonNull::new(ptr).ok_or(AllocError::OutOfMemory)?;

        Ok(Self {
            ptr,
            capacity,
            offset: AtomicUsize::new(0),
        })
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
        // Alignment must be a power of two; checked_sub handles align == 0 below,
        // but non-power-of-two values produce invalid bitmasks silently.
        debug_assert!(
            align.is_power_of_two(),
            "alignment must be a power of two, got {align}"
        );
        let base_addr = self.ptr.as_ptr() as usize;
        loop {
            let current = self.offset.load(Ordering::Relaxed);

            // Align the absolute address (base + offset), then convert back to an
            // offset. This is correct for any requested alignment, even when it
            // exceeds the chunk's own 16-byte alignment.
            let align_mask = align.checked_sub(1)?;
            let current_addr = base_addr.checked_add(current)?;
            let aligned_addr = current_addr.checked_add(align_mask)? & !align_mask;
            let aligned = aligned_addr - base_addr;
            let new_offset = aligned.checked_add(size)?;

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
                    // reason: aligned <= capacity, and chunk sizes are well under u32::MAX (4 GiB)
                    #[allow(clippy::cast_possible_truncation)]
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
    ///
    /// # Errors
    ///
    /// Returns `AllocError::OutOfMemory` if the initial chunk allocation fails.
    pub fn new(epoch: EpochId) -> Result<Self, AllocError> {
        Self::with_chunk_size(epoch, DEFAULT_CHUNK_SIZE)
    }

    /// Creates a new arena with a custom chunk size.
    ///
    /// # Errors
    ///
    /// Returns `AllocError::OutOfMemory` if the initial chunk allocation fails.
    pub fn with_chunk_size(epoch: EpochId, chunk_size: usize) -> Result<Self, AllocError> {
        let initial_chunk = Chunk::new(chunk_size)?;
        Ok(Self {
            epoch,
            chunks: RwLock::new(vec![initial_chunk]),
            chunk_size,
            total_allocated: AtomicUsize::new(chunk_size),
        })
    }

    /// Returns the epoch this arena belongs to.
    #[must_use]
    pub fn epoch(&self) -> EpochId {
        self.epoch
    }

    /// Allocates `size` bytes with the given alignment.
    ///
    /// # Errors
    ///
    /// Returns `AllocError::OutOfMemory` if a new chunk is needed and
    /// the system allocator fails.
    pub fn alloc(&self, size: usize, align: usize) -> Result<NonNull<u8>, AllocError> {
        if align == 0 || !align.is_power_of_two() {
            return Err(AllocError::InvalidAlignment(align));
        }
        // First try to allocate from existing chunks
        {
            let chunks = self.chunks.read();
            for chunk in chunks.iter().rev() {
                if let Some(ptr) = chunk.try_alloc(size, align) {
                    return Ok(ptr);
                }
            }
        }

        // Need a new chunk
        self.alloc_new_chunk(size, align)
    }

    /// Allocates a value of type T.
    ///
    /// # Errors
    ///
    /// Returns `AllocError::OutOfMemory` if allocation fails.
    pub fn alloc_value<T>(&self, value: T) -> Result<&mut T, AllocError> {
        let ptr = self.alloc(std::mem::size_of::<T>(), std::mem::align_of::<T>())?;
        // SAFETY: We've allocated the correct size and alignment
        Ok(unsafe {
            let typed_ptr = ptr.as_ptr() as *mut T;
            typed_ptr.write(value);
            &mut *typed_ptr
        })
    }

    /// Allocates a slice of values.
    ///
    /// # Errors
    ///
    /// Returns `AllocError::OutOfMemory` if allocation fails.
    pub fn alloc_slice<T: Copy>(&self, values: &[T]) -> Result<&mut [T], AllocError> {
        if values.is_empty() {
            return Ok(&mut []);
        }

        let size = std::mem::size_of::<T>()
            .checked_mul(values.len())
            .ok_or(AllocError::OutOfMemory)?;
        let align = std::mem::align_of::<T>();
        let ptr = self.alloc(size, align)?;

        // SAFETY: We've allocated the correct size and alignment
        Ok(unsafe {
            let typed_ptr = ptr.as_ptr() as *mut T;
            std::ptr::copy_nonoverlapping(values.as_ptr(), typed_ptr, values.len());
            std::slice::from_raw_parts_mut(typed_ptr, values.len())
        })
    }

    /// Allocates a new chunk and performs the allocation.
    fn alloc_new_chunk(&self, size: usize, align: usize) -> Result<NonNull<u8>, AllocError> {
        let required = size.checked_add(align).ok_or(AllocError::OutOfMemory)?;
        let chunk_size = self.chunk_size.max(required);
        let chunk = Chunk::new(chunk_size)?;

        self.total_allocated
            .fetch_add(chunk_size, Ordering::Relaxed);

        // The chunk was sized to fit this allocation, so this cannot fail.
        let ptr = chunk
            .try_alloc(size, align)
            .expect("fresh chunk sized to fit");

        let mut chunks = self.chunks.write();
        chunks.push(chunk);

        Ok(ptr)
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
    ///
    /// # Errors
    ///
    /// Returns `AllocError::OutOfMemory` if the initial arena allocation fails.
    pub fn new() -> Result<Self, AllocError> {
        Self::with_chunk_size(DEFAULT_CHUNK_SIZE)
    }

    /// Creates a new arena allocator with a custom chunk size.
    ///
    /// # Errors
    ///
    /// Returns `AllocError::OutOfMemory` if the initial arena allocation fails.
    pub fn with_chunk_size(chunk_size: usize) -> Result<Self, AllocError> {
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
            .insert(epoch, Arena::with_chunk_size(epoch, chunk_size)?);

        Ok(allocator)
    }

    /// Returns the current epoch.
    #[must_use]
    pub fn current_epoch(&self) -> EpochId {
        EpochId::new(self.current_epoch.load(Ordering::Acquire) as u64)
    }

    /// Creates a new epoch and returns its ID.
    ///
    /// # Errors
    ///
    /// Returns `AllocError::OutOfMemory` if the arena allocation fails.
    pub fn new_epoch(&self) -> Result<EpochId, AllocError> {
        let new_id = self.current_epoch.fetch_add(1, Ordering::AcqRel) as u64 + 1;
        let epoch = EpochId::new(new_id);

        let arena = Arena::with_chunk_size(epoch, self.chunk_size)?;
        self.arenas.write().insert(epoch, arena);

        Ok(epoch)
    }

    /// Gets the arena for a specific epoch.
    ///
    /// # Errors
    ///
    /// Returns `AllocError::EpochNotFound` if the epoch doesn't exist.
    pub fn arena(
        &self,
        epoch: EpochId,
    ) -> Result<impl std::ops::Deref<Target = Arena> + '_, AllocError> {
        let arenas = self.arenas.read();
        if !arenas.contains_key(&epoch) {
            return Err(AllocError::EpochNotFound(epoch));
        }
        Ok(parking_lot::RwLockReadGuard::map(arenas, |arenas| {
            &arenas[&epoch]
        }))
    }

    /// Allocates in the current epoch.
    ///
    /// # Errors
    ///
    /// Returns `AllocError` if allocation fails.
    ///
    /// # Panics
    ///
    /// Panics if the current epoch has no arena (should never happen in normal use).
    pub fn alloc(&self, size: usize, align: usize) -> Result<NonNull<u8>, AllocError> {
        let epoch = self.current_epoch();
        let arenas = self.arenas.read();
        arenas
            .get(&epoch)
            .expect("current epoch always exists")
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
    /// Creates a default arena allocator.
    ///
    /// # Panics
    ///
    /// Panics if the initial arena allocation fails (out of memory).
    fn default() -> Self {
        Self::new().expect("failed to allocate default arena")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_arena_basic_allocation() {
        let arena = Arena::new(EpochId::INITIAL).unwrap();

        // Allocate some bytes
        let ptr1 = arena.alloc(100, 8).unwrap();
        let ptr2 = arena.alloc(200, 8).unwrap();

        // Pointers should be different
        assert_ne!(ptr1.as_ptr(), ptr2.as_ptr());
    }

    #[test]
    fn test_arena_value_allocation() {
        let arena = Arena::new(EpochId::INITIAL).unwrap();

        let value = arena.alloc_value(42u64).unwrap();
        assert_eq!(*value, 42);

        *value = 100;
        assert_eq!(*value, 100);
    }

    #[test]
    fn test_arena_slice_allocation() {
        let arena = Arena::new(EpochId::INITIAL).unwrap();

        let slice = arena.alloc_slice(&[1u32, 2, 3, 4, 5]).unwrap();
        assert_eq!(slice, &[1, 2, 3, 4, 5]);

        slice[0] = 10;
        assert_eq!(slice[0], 10);
    }

    #[test]
    fn test_arena_large_allocation() {
        let arena = Arena::with_chunk_size(EpochId::INITIAL, 1024).unwrap();

        // Allocate something larger than the chunk size
        let _ptr = arena.alloc(2048, 8).unwrap();

        // Should have created a new chunk
        assert!(arena.stats().chunk_count >= 2);
    }

    #[test]
    fn test_arena_allocator_epochs() {
        let allocator = ArenaAllocator::new().unwrap();

        let epoch0 = allocator.current_epoch();
        assert_eq!(epoch0, EpochId::INITIAL);

        let epoch1 = allocator.new_epoch().unwrap();
        assert_eq!(epoch1, EpochId::new(1));

        let epoch2 = allocator.new_epoch().unwrap();
        assert_eq!(epoch2, EpochId::new(2));

        // Current epoch should be the latest
        assert_eq!(allocator.current_epoch(), epoch2);
    }

    #[test]
    fn test_arena_allocator_allocation() {
        let allocator = ArenaAllocator::new().unwrap();

        let ptr1 = allocator.alloc(100, 8).unwrap();
        let ptr2 = allocator.alloc(100, 8).unwrap();

        assert_ne!(ptr1.as_ptr(), ptr2.as_ptr());
    }

    #[test]
    fn test_arena_drop_epoch() {
        let allocator = ArenaAllocator::new().unwrap();

        let initial_mem = allocator.total_allocated();

        let epoch1 = allocator.new_epoch().unwrap();
        // Allocate some memory in the new epoch
        {
            let arena = allocator.arena(epoch1).unwrap();
            arena.alloc(10000, 8).unwrap();
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
        let arena = Arena::with_chunk_size(EpochId::new(5), 4096).unwrap();

        let stats = arena.stats();
        assert_eq!(stats.epoch, EpochId::new(5));
        assert_eq!(stats.chunk_count, 1);
        assert_eq!(stats.total_allocated, 4096);
        assert_eq!(stats.total_used, 0);

        arena.alloc(100, 8).unwrap();
        let stats = arena.stats();
        assert!(stats.total_used >= 100);
    }
}
