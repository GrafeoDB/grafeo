//! Memory grant RAII wrapper for automatic resource release.

use super::region::MemoryRegion;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

/// Trait for releasing memory grants.
///
/// This allows the MemoryGrant to release memory without directly
/// depending on the full BufferManager type.
pub trait GrantReleaser: Send + Sync {
    /// Releases memory back to the manager.
    fn release(&self, size: usize, region: MemoryRegion);

    /// Tries to allocate additional memory (for resize operations).
    fn try_allocate_raw(&self, size: usize, region: MemoryRegion) -> bool;
}

/// RAII wrapper for memory allocations.
///
/// Automatically releases memory back to the `BufferManager` when dropped.
/// Use `consume()` to transfer ownership without releasing.
pub struct MemoryGrant {
    /// Reference to the releaser (BufferManager).
    releaser: Arc<dyn GrantReleaser>,
    /// Size of this grant in bytes.
    size: AtomicUsize,
    /// Memory region for this grant.
    region: MemoryRegion,
    /// Whether this grant has been consumed (transferred).
    consumed: bool,
}

impl MemoryGrant {
    /// Creates a new memory grant.
    pub(crate) fn new(releaser: Arc<dyn GrantReleaser>, size: usize, region: MemoryRegion) -> Self {
        Self {
            releaser,
            size: AtomicUsize::new(size),
            region,
            consumed: false,
        }
    }

    /// Returns the size of this grant in bytes.
    #[must_use]
    pub fn size(&self) -> usize {
        self.size.load(Ordering::Relaxed)
    }

    /// Returns the memory region of this grant.
    #[must_use]
    pub fn region(&self) -> MemoryRegion {
        self.region
    }

    /// Attempts to resize the grant.
    ///
    /// Returns `true` if the resize succeeded, `false` if more memory
    /// could not be allocated.
    pub fn resize(&mut self, new_size: usize) -> bool {
        let current = self.size.load(Ordering::Relaxed);

        if new_size > current {
            // Need more memory - try to allocate the difference
            let diff = new_size - current;
            if self.releaser.try_allocate_raw(diff, self.region) {
                self.size.store(new_size, Ordering::Relaxed);
                true
            } else {
                false
            }
        } else if new_size < current {
            // Releasing memory
            let diff = current - new_size;
            self.releaser.release(diff, self.region);
            self.size.store(new_size, Ordering::Relaxed);
            true
        } else {
            // Same size, nothing to do
            true
        }
    }

    /// Splits off a portion of this grant into a new grant.
    ///
    /// Returns `None` if the requested amount exceeds the current size.
    pub fn split(&mut self, amount: usize) -> Option<MemoryGrant> {
        let current = self.size.load(Ordering::Relaxed);
        if amount > current {
            return None;
        }

        self.size.store(current - amount, Ordering::Relaxed);
        Some(MemoryGrant {
            releaser: Arc::clone(&self.releaser),
            size: AtomicUsize::new(amount),
            region: self.region,
            consumed: false,
        })
    }

    /// Merges another grant into this one.
    ///
    /// Both grants must be for the same region.
    ///
    /// # Panics
    ///
    /// Panics if the grants are for different regions.
    pub fn merge(&mut self, other: MemoryGrant) {
        assert_eq!(
            self.region, other.region,
            "Cannot merge grants from different regions"
        );

        let other_size = other.consume();
        let current = self.size.load(Ordering::Relaxed);
        self.size.store(current + other_size, Ordering::Relaxed);
    }

    /// Consumes this grant without releasing memory.
    ///
    /// Use this to transfer ownership of the memory to another owner.
    /// The caller is responsible for eventually releasing the memory.
    pub fn consume(mut self) -> usize {
        self.consumed = true;
        self.size.load(Ordering::Relaxed)
    }

    /// Returns whether this grant has been consumed.
    #[must_use]
    pub fn is_consumed(&self) -> bool {
        self.consumed
    }

    /// Returns whether this grant is empty (size == 0).
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.size.load(Ordering::Relaxed) == 0
    }
}

impl Drop for MemoryGrant {
    fn drop(&mut self) {
        if !self.consumed {
            let size = self.size.load(Ordering::Relaxed);
            if size > 0 {
                self.releaser.release(size, self.region);
            }
        }
    }
}

impl std::fmt::Debug for MemoryGrant {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MemoryGrant")
            .field("size", &self.size.load(Ordering::Relaxed))
            .field("region", &self.region)
            .field("consumed", &self.consumed)
            .finish()
    }
}

/// A collection of memory grants that can be managed together.
#[derive(Default)]
pub struct CompositeGrant {
    grants: Vec<MemoryGrant>,
}

impl CompositeGrant {
    /// Creates a new empty composite grant.
    #[must_use]
    pub fn new() -> Self {
        Self { grants: Vec::new() }
    }

    /// Adds a grant to the collection.
    pub fn add(&mut self, grant: MemoryGrant) {
        self.grants.push(grant);
    }

    /// Returns the total size of all grants.
    #[must_use]
    pub fn total_size(&self) -> usize {
        self.grants.iter().map(MemoryGrant::size).sum()
    }

    /// Returns the number of grants.
    #[must_use]
    pub fn len(&self) -> usize {
        self.grants.len()
    }

    /// Returns whether the collection is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.grants.is_empty()
    }

    /// Consumes all grants and returns the total size.
    pub fn consume_all(self) -> usize {
        self.grants.into_iter().map(MemoryGrant::consume).sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicUsize;

    struct MockReleaser {
        released: AtomicUsize,
        allocated: AtomicUsize,
    }

    impl MockReleaser {
        fn new() -> Arc<Self> {
            Arc::new(Self {
                released: AtomicUsize::new(0),
                allocated: AtomicUsize::new(0),
            })
        }
    }

    impl GrantReleaser for MockReleaser {
        fn release(&self, size: usize, _region: MemoryRegion) {
            self.released.fetch_add(size, Ordering::Relaxed);
        }

        fn try_allocate_raw(&self, size: usize, _region: MemoryRegion) -> bool {
            self.allocated.fetch_add(size, Ordering::Relaxed);
            true
        }
    }

    #[test]
    fn test_grant_drop_releases_memory() {
        let releaser = MockReleaser::new();

        {
            let _grant = MemoryGrant::new(
                Arc::clone(&releaser) as Arc<dyn GrantReleaser>,
                1024,
                MemoryRegion::ExecutionBuffers,
            );
            assert_eq!(releaser.released.load(Ordering::Relaxed), 0);
        }

        // After drop, memory should be released
        assert_eq!(releaser.released.load(Ordering::Relaxed), 1024);
    }

    #[test]
    fn test_grant_consume_no_release() {
        let releaser = MockReleaser::new();

        let grant = MemoryGrant::new(
            Arc::clone(&releaser) as Arc<dyn GrantReleaser>,
            1024,
            MemoryRegion::ExecutionBuffers,
        );

        let size = grant.consume();
        assert_eq!(size, 1024);

        // No release should happen
        assert_eq!(releaser.released.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn test_grant_resize_grow() {
        let releaser = MockReleaser::new();

        let mut grant = MemoryGrant::new(
            Arc::clone(&releaser) as Arc<dyn GrantReleaser>,
            1024,
            MemoryRegion::ExecutionBuffers,
        );

        assert!(grant.resize(2048));
        assert_eq!(grant.size(), 2048);
        assert_eq!(releaser.allocated.load(Ordering::Relaxed), 1024);
    }

    #[test]
    fn test_grant_resize_shrink() {
        let releaser = MockReleaser::new();

        let mut grant = MemoryGrant::new(
            Arc::clone(&releaser) as Arc<dyn GrantReleaser>,
            1024,
            MemoryRegion::ExecutionBuffers,
        );

        assert!(grant.resize(512));
        assert_eq!(grant.size(), 512);
        assert_eq!(releaser.released.load(Ordering::Relaxed), 512);
    }

    #[test]
    fn test_grant_split() {
        let releaser = MockReleaser::new();

        let mut grant = MemoryGrant::new(
            Arc::clone(&releaser) as Arc<dyn GrantReleaser>,
            1000,
            MemoryRegion::ExecutionBuffers,
        );

        let split = grant.split(400).unwrap();
        assert_eq!(grant.size(), 600);
        assert_eq!(split.size(), 400);

        // Cannot split more than available
        assert!(grant.split(1000).is_none());
    }

    #[test]
    fn test_grant_merge() {
        let releaser = MockReleaser::new();

        let mut grant1 = MemoryGrant::new(
            Arc::clone(&releaser) as Arc<dyn GrantReleaser>,
            600,
            MemoryRegion::ExecutionBuffers,
        );

        let grant2 = MemoryGrant::new(
            Arc::clone(&releaser) as Arc<dyn GrantReleaser>,
            400,
            MemoryRegion::ExecutionBuffers,
        );

        grant1.merge(grant2);
        assert_eq!(grant1.size(), 1000);

        // grant2 was consumed during merge, no release
        assert_eq!(releaser.released.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn test_composite_grant() {
        let releaser = MockReleaser::new();

        let mut composite = CompositeGrant::new();
        assert!(composite.is_empty());

        composite.add(MemoryGrant::new(
            Arc::clone(&releaser) as Arc<dyn GrantReleaser>,
            100,
            MemoryRegion::ExecutionBuffers,
        ));
        composite.add(MemoryGrant::new(
            Arc::clone(&releaser) as Arc<dyn GrantReleaser>,
            200,
            MemoryRegion::ExecutionBuffers,
        ));

        assert_eq!(composite.len(), 2);
        assert_eq!(composite.total_size(), 300);

        let total = composite.consume_all();
        assert_eq!(total, 300);

        // No release since all were consumed
        assert_eq!(releaser.released.load(Ordering::Relaxed), 0);
    }
}
