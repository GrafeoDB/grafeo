//! Bump allocator for temporary allocations.
//!
//! When you need fast allocations that all get freed together (like during
//! a single query), this is your friend. Just keep allocating, then call
//! `reset()` when you're done. Wraps `bumpalo` for the heavy lifting.

use std::cell::Cell;
use std::ptr::NonNull;

/// Allocates by bumping a pointer - fast, but no individual frees.
///
/// Keep allocating throughout an operation, then call [`reset()`](Self::reset)
/// to free everything at once. Not thread-safe - use one per thread.
///
/// # Examples
///
/// ```
/// use grafeo_common::memory::BumpAllocator;
///
/// let mut bump = BumpAllocator::new();
///
/// // Allocate a bunch of stuff
/// let a = bump.alloc(42u64);
/// let b = bump.alloc_str("hello");
///
/// // Use them...
/// assert_eq!(*a, 42);
///
/// // Free everything at once
/// bump.reset();
/// ```
pub struct BumpAllocator {
    /// The underlying bumpalo allocator.
    inner: bumpalo::Bump,
    /// Number of allocations made.
    allocation_count: Cell<usize>,
}

impl BumpAllocator {
    /// Creates a new bump allocator.
    #[must_use]
    pub fn new() -> Self {
        Self {
            inner: bumpalo::Bump::new(),
            allocation_count: Cell::new(0),
        }
    }

    /// Creates a new bump allocator with the given initial capacity.
    #[must_use]
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            inner: bumpalo::Bump::with_capacity(capacity),
            allocation_count: Cell::new(0),
        }
    }

    /// Allocates memory for a value of type T and initializes it.
    #[inline]
    pub fn alloc<T>(&self, value: T) -> &mut T {
        self.allocation_count.set(self.allocation_count.get() + 1);
        self.inner.alloc(value)
    }

    /// Allocates memory for a slice and copies the values.
    #[inline]
    pub fn alloc_slice_copy<T: Copy>(&self, values: &[T]) -> &mut [T] {
        self.allocation_count.set(self.allocation_count.get() + 1);
        self.inner.alloc_slice_copy(values)
    }

    /// Allocates memory for a slice and clones the values.
    #[inline]
    pub fn alloc_slice_clone<T: Clone>(&self, values: &[T]) -> &mut [T] {
        self.allocation_count.set(self.allocation_count.get() + 1);
        self.inner.alloc_slice_clone(values)
    }

    /// Allocates memory for a string and returns a mutable reference.
    #[inline]
    pub fn alloc_str(&self, s: &str) -> &mut str {
        self.allocation_count.set(self.allocation_count.get() + 1);
        self.inner.alloc_str(s)
    }

    /// Allocates raw bytes with the given layout.
    #[inline]
    pub fn alloc_layout(&self, layout: std::alloc::Layout) -> NonNull<u8> {
        self.allocation_count.set(self.allocation_count.get() + 1);
        self.inner.alloc_layout(layout)
    }

    /// Resets the allocator, freeing all allocated memory.
    ///
    /// This is very fast - it just resets the internal pointer.
    #[inline]
    pub fn reset(&mut self) {
        self.inner.reset();
        self.allocation_count.set(0);
    }

    /// Returns the number of bytes currently allocated.
    #[must_use]
    #[inline]
    pub fn allocated_bytes(&self) -> usize {
        self.inner.allocated_bytes()
    }

    /// Returns the number of allocations made.
    #[must_use]
    #[inline]
    pub fn allocation_count(&self) -> usize {
        self.allocation_count.get()
    }

    /// Returns the number of chunks in use.
    #[must_use]
    #[inline]
    pub fn chunk_count(&mut self) -> usize {
        self.inner.iter_allocated_chunks().count()
    }
}

impl Default for BumpAllocator {
    fn default() -> Self {
        Self::new()
    }
}

/// Tracks allocations within a specific scope.
///
/// Wraps a [`BumpAllocator`] and tracks how much was allocated in this scope.
/// Note: bumpalo doesn't support partial reset, so memory is only freed when
/// the parent allocator is reset.
pub struct ScopedBump<'a> {
    bump: &'a mut BumpAllocator,
    start_bytes: usize,
}

impl<'a> ScopedBump<'a> {
    /// Creates a new scoped allocator.
    pub fn new(bump: &'a mut BumpAllocator) -> Self {
        let start_bytes = bump.allocated_bytes();
        Self { bump, start_bytes }
    }

    /// Allocates a value.
    #[inline]
    pub fn alloc<T>(&self, value: T) -> &mut T {
        self.bump.alloc(value)
    }

    /// Returns the number of bytes allocated in this scope.
    #[must_use]
    pub fn scope_allocated_bytes(&self) -> usize {
        self.bump.allocated_bytes() - self.start_bytes
    }
}

impl Drop for ScopedBump<'_> {
    fn drop(&mut self) {
        // Note: bumpalo doesn't support partial reset, so this is a no-op.
        // The memory will be freed when the parent BumpAllocator is reset.
        // This type is mainly for tracking scope-level allocations.
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bump_basic_allocation() {
        let bump = BumpAllocator::new();

        let a = bump.alloc(42u64);
        assert_eq!(*a, 42);

        let b = bump.alloc(String::from("hello"));
        assert_eq!(b.as_str(), "hello");
    }

    #[test]
    fn test_bump_slice_allocation() {
        let bump = BumpAllocator::new();

        let slice = bump.alloc_slice_copy(&[1, 2, 3, 4, 5]);
        assert_eq!(slice, &[1, 2, 3, 4, 5]);

        slice[0] = 10;
        assert_eq!(slice[0], 10);
    }

    #[test]
    fn test_bump_string_allocation() {
        let bump = BumpAllocator::new();

        let s = bump.alloc_str("hello world");
        assert_eq!(s, "hello world");
    }

    #[test]
    fn test_bump_reset() {
        let mut bump = BumpAllocator::new();

        for _ in 0..100 {
            bump.alloc(42u64);
        }

        let bytes_before = bump.allocated_bytes();
        assert!(bytes_before > 0);
        assert_eq!(bump.allocation_count(), 100);

        bump.reset();

        // After reset, allocation count should be 0
        assert_eq!(bump.allocation_count(), 0);
    }

    #[test]
    fn test_bump_with_capacity() {
        let bump = BumpAllocator::with_capacity(1024);
        assert_eq!(bump.allocation_count(), 0);

        // Allocate less than capacity
        for _ in 0..10 {
            bump.alloc(42u64);
        }

        assert_eq!(bump.allocation_count(), 10);
    }

    #[test]
    fn test_scoped_bump() {
        let mut bump = BumpAllocator::new();

        bump.alloc(1u64);
        let outer_allocs = bump.allocation_count();

        {
            let scope = ScopedBump::new(&mut bump);
            scope.alloc(2u64);
            scope.alloc(3u64);

            // Note: bumpalo's allocated_bytes() tracks chunk-level memory, not individual
            // allocations. It may not increase for small allocations that fit in existing chunks.
            // We verify functionality through allocation_count instead.
        }

        // Parent bump should have all allocations (outer + 2 from scope)
        assert_eq!(bump.allocation_count(), outer_allocs + 2);
    }

    #[test]
    fn test_bump_many_small_allocations() {
        let bump = BumpAllocator::new();

        for i in 0u64..10_000u64 {
            let v = bump.alloc(i);
            assert_eq!(*v, i);
        }

        assert_eq!(bump.allocation_count(), 10_000);
    }
}
