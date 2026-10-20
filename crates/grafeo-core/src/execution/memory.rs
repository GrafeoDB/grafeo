//! Execution memory context for memory-aware query execution.

use grafeo_common::memory::buffer::{BufferManager, MemoryGrant, MemoryRegion, PressureLevel};
use std::sync::Arc;

/// Default chunk size for execution buffers.
pub const DEFAULT_CHUNK_SIZE: usize = 2048;

/// Chunk size under moderate memory pressure.
pub const MODERATE_PRESSURE_CHUNK_SIZE: usize = 1024;

/// Chunk size under high memory pressure.
pub const HIGH_PRESSURE_CHUNK_SIZE: usize = 512;

/// Chunk size under critical memory pressure.
pub const CRITICAL_PRESSURE_CHUNK_SIZE: usize = 256;

/// Execution context with memory awareness.
///
/// This context provides memory allocation for query execution operators
/// and adjusts chunk sizes based on memory pressure.
pub struct ExecutionMemoryContext {
    /// Reference to the buffer manager.
    manager: Arc<BufferManager>,
    /// Total bytes allocated for this execution context.
    allocated: usize,
    /// Grants held by this context.
    grants: Vec<MemoryGrant>,
}

impl ExecutionMemoryContext {
    /// Creates a new execution memory context.
    #[must_use]
    pub fn new(manager: Arc<BufferManager>) -> Self {
        Self {
            manager,
            allocated: 0,
            grants: Vec::new(),
        }
    }

    /// Requests memory for execution buffers.
    ///
    /// Returns `None` if the allocation cannot be satisfied.
    pub fn allocate(&mut self, size: usize) -> Option<MemoryGrant> {
        let grant = self
            .manager
            .try_allocate(size, MemoryRegion::ExecutionBuffers)?;
        self.allocated += size;
        Some(grant)
    }

    /// Allocates and stores a grant internally.
    ///
    /// The grant will be released when this context is dropped.
    pub fn allocate_tracked(&mut self, size: usize) -> bool {
        if let Some(grant) = self
            .manager
            .try_allocate(size, MemoryRegion::ExecutionBuffers)
        {
            self.allocated += size;
            self.grants.push(grant);
            true
        } else {
            false
        }
    }

    /// Returns the current pressure level.
    #[must_use]
    pub fn pressure_level(&self) -> PressureLevel {
        self.manager.pressure_level()
    }

    /// Returns whether chunk size should be reduced due to memory pressure.
    #[must_use]
    pub fn should_reduce_chunk_size(&self) -> bool {
        matches!(
            self.pressure_level(),
            PressureLevel::High | PressureLevel::Critical
        )
    }

    /// Computes adjusted chunk size based on memory pressure.
    #[must_use]
    pub fn adjusted_chunk_size(&self, requested: usize) -> usize {
        match self.pressure_level() {
            PressureLevel::Normal => requested,
            PressureLevel::Moderate => requested.min(MODERATE_PRESSURE_CHUNK_SIZE),
            PressureLevel::High => requested.min(HIGH_PRESSURE_CHUNK_SIZE),
            PressureLevel::Critical => requested.min(CRITICAL_PRESSURE_CHUNK_SIZE),
            _ => requested.min(CRITICAL_PRESSURE_CHUNK_SIZE),
        }
    }

    /// Returns the optimal chunk size for the current memory state.
    #[must_use]
    pub fn optimal_chunk_size(&self) -> usize {
        self.adjusted_chunk_size(DEFAULT_CHUNK_SIZE)
    }

    /// Returns total bytes allocated through this context.
    #[must_use]
    pub fn total_allocated(&self) -> usize {
        self.allocated
    }

    /// Returns the buffer manager.
    #[must_use]
    pub fn manager(&self) -> &Arc<BufferManager> {
        &self.manager
    }

    /// Releases all tracked grants.
    pub fn release_all(&mut self) {
        self.grants.clear();
        self.allocated = 0;
    }
}

impl Drop for ExecutionMemoryContext {
    fn drop(&mut self) {
        // Grants are automatically released when dropped
        self.grants.clear();
    }
}

/// Builder for execution memory contexts with pre-allocation.
pub struct ExecutionMemoryContextBuilder {
    manager: Arc<BufferManager>,
    initial_allocation: usize,
}

impl ExecutionMemoryContextBuilder {
    /// Creates a new builder with the given buffer manager.
    #[must_use]
    pub fn new(manager: Arc<BufferManager>) -> Self {
        Self {
            manager,
            initial_allocation: 0,
        }
    }

    /// Sets the initial allocation size.
    #[must_use]
    pub fn with_initial_allocation(mut self, size: usize) -> Self {
        self.initial_allocation = size;
        self
    }

    /// Builds the execution memory context.
    ///
    /// Returns `None` if the initial allocation cannot be satisfied.
    pub fn build(self) -> Option<ExecutionMemoryContext> {
        let mut ctx = ExecutionMemoryContext::new(self.manager);

        if self.initial_allocation > 0 && !ctx.allocate_tracked(self.initial_allocation) {
            return None;
        }

        Some(ctx)
    }
}

/// Memory context passed to pipeline operators for pressure-aware spill decisions.
///
/// Lightweight and `Clone`-able (two `Arc` references). Operators that support
/// memory-aware spilling receive this at construction time. It provides access
/// to the `BufferManager` for pressure queries and consumer registration, and
/// to a per-query `SpillManager` for creating spill files.
#[cfg(feature = "spill")]
#[derive(Clone)]
pub struct OperatorMemoryContext {
    /// The buffer manager for pressure queries and consumer registration.
    buffer_manager: Arc<BufferManager>,
    /// Per-query spill manager for creating spill files.
    spill_manager: Arc<super::spill::SpillManager>,
}

#[cfg(feature = "spill")]
impl OperatorMemoryContext {
    /// Creates a new operator memory context.
    #[must_use]
    pub fn new(
        buffer_manager: Arc<BufferManager>,
        spill_manager: Arc<super::spill::SpillManager>,
    ) -> Self {
        Self {
            buffer_manager,
            spill_manager,
        }
    }

    /// Returns the current pressure level.
    #[must_use]
    pub fn pressure_level(&self) -> PressureLevel {
        self.buffer_manager.pressure_level()
    }

    /// Returns whether system pressure warrants spilling (High or Critical).
    #[must_use]
    pub fn should_spill(&self) -> bool {
        self.pressure_level().should_spill()
    }

    /// Registers a memory consumer with the buffer manager.
    pub fn register_consumer(
        &self,
        consumer: Arc<dyn grafeo_common::memory::buffer::MemoryConsumer>,
    ) {
        self.buffer_manager.register_consumer(consumer);
    }

    /// Unregisters a memory consumer by name.
    pub fn unregister_consumer(&self, name: &str) {
        self.buffer_manager.unregister_consumer(name);
    }

    /// Returns a reference to the buffer manager.
    #[must_use]
    pub fn buffer_manager(&self) -> &Arc<BufferManager> {
        &self.buffer_manager
    }

    /// Returns a reference to the spill manager.
    #[must_use]
    pub fn spill_manager(&self) -> &Arc<super::spill::SpillManager> {
        &self.spill_manager
    }
}

/// Reservation carried with grouped state or the output that owns its buffers.
///
/// This owner never detaches a grant from its releaser. Splitting transfers
/// existing admission; merging is permitted only for the same buffer manager.
pub struct GroupedMemoryOwner {
    grant: MemoryGrant,
    manager: Arc<BufferManager>,
}

impl GroupedMemoryOwner {
    /// Admits bytes before the caller allocates their backing storage.
    ///
    /// # Errors
    ///
    /// Returns the existing resource-limit error if admission is refused.
    pub fn try_new(
        manager: Arc<BufferManager>,
        bytes: usize,
    ) -> Result<Self, super::operators::OperatorError> {
        let grant = manager
            .try_allocate(bytes, MemoryRegion::ExecutionBuffers)
            .ok_or_else(grouped_memory_limit)?;
        Ok(Self { grant, manager })
    }

    /// Returns this owner's reserved byte count.
    #[must_use]
    pub fn size(&self) -> usize {
        self.grant.size()
    }

    /// Returns the manager that owns this reservation.
    #[must_use]
    pub fn manager(&self) -> &Arc<BufferManager> {
        &self.manager
    }

    /// Admits additional bytes before growing owned storage.
    ///
    /// # Errors
    ///
    /// Overflow or refused admission leaves the original reservation intact.
    pub fn try_grow(&mut self, additional: usize) -> Result<(), super::operators::OperatorError> {
        let total = self
            .size()
            .checked_add(additional)
            .ok_or_else(grouped_memory_limit)?;
        self.try_resize(total)
    }

    /// Transfers already admitted bytes without allocating another grant.
    ///
    /// # Errors
    ///
    /// Returns an invariant error when the requested bytes are not owned.
    pub fn split(&mut self, bytes: usize) -> Result<Self, super::operators::OperatorError> {
        let grant = self.grant.split(bytes).ok_or_else(|| {
            super::operators::OperatorError::Internal(
                "grouped memory reservation cannot transfer unowned bytes".to_string(),
            )
        })?;
        Ok(Self {
            grant,
            manager: Arc::clone(&self.manager),
        })
    }

    /// Combines reservations from the same manager.
    ///
    /// # Errors
    ///
    /// Returns the error and the unmerged owner on a manager mismatch or
    /// overflow, preserving the caller's authority over its live buffers.
    pub fn try_merge(
        &mut self,
        other: Self,
    ) -> Result<(), (super::operators::OperatorError, Self)> {
        if !Arc::ptr_eq(&self.manager, &other.manager)
            || self.size().checked_add(other.size()).is_none()
        {
            return Err((
                super::operators::OperatorError::Internal(
                    "grouped memory reservations must belong to the same manager".to_string(),
                ),
                other,
            ));
        }
        self.grant.merge(other.grant);
        Ok(())
    }

    pub(crate) fn try_resize(
        &mut self,
        bytes: usize,
    ) -> Result<(), super::operators::OperatorError> {
        if self.grant.resize(bytes) {
            Ok(())
        } else {
            Err(grouped_memory_limit())
        }
    }
}

pub(crate) fn grouped_memory_limit() -> super::operators::OperatorError {
    super::operators::OperatorError::LimitExceeded(
        "grouped query exceeds its memory limit; increase the memory budget or reduce the grouped result".to_string(),
    )
}

#[cfg(feature = "spill")]
pub(crate) fn checked_grouped_bytes(
    count: usize,
    width: usize,
) -> Result<usize, super::operators::OperatorError> {
    count.checked_mul(width).ok_or_else(grouped_memory_limit)
}

/// Bounds a fresh std/hashbrown table allocation for the pinned Rust 1.99
/// layout. The data area is followed by one control byte per bucket and a
/// 16-byte control group. Charge the old table separately during rehashing.
#[cfg(feature = "spill")]
pub(crate) fn grouped_hash_table_bytes<T>(
    entries: usize,
) -> Result<usize, super::operators::OperatorError> {
    if entries == 0 {
        return Ok(0);
    }
    // Small-table minima depend on element size in hashbrown 0.17.1;
    // sixteen buckets bounds each case without assuming an older layout.
    let buckets = entries
        .checked_mul(8)
        .and_then(|n| n.checked_add(6))
        .map(|n| n / 7)
        .and_then(usize::checked_next_power_of_two)
        .ok_or_else(grouped_memory_limit)?
        .max(16);
    let alignment = std::mem::align_of::<T>().max(16);
    let bytes = checked_grouped_bytes(buckets, std::mem::size_of::<T>())?;
    let aligned = bytes
        .checked_add(alignment - 1)
        .map(|n| n & !(alignment - 1))
        .ok_or_else(grouped_memory_limit)?;
    aligned
        .checked_add(buckets)
        .and_then(|n| n.checked_add(16))
        .ok_or_else(grouped_memory_limit)
}

#[cfg(feature = "spill")]
pub(crate) fn grouped_io_error(error: std::io::Error) -> super::operators::OperatorError {
    if error
        .get_ref()
        .is_some_and(|inner| inner.is::<super::operators::OperatorError>())
    {
        // The type was checked above; retain its resource/timeout category.
        let inner = error.into_inner().expect("checked an inner operator error");
        *inner
            .downcast::<super::operators::OperatorError>()
            .expect("checked operator error type")
    } else {
        super::operators::OperatorError::from(grafeo_common::utils::error::Error::Io(error))
    }
}

impl std::fmt::Debug for GroupedMemoryOwner {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GroupedMemoryOwner")
            .field("bytes", &self.size())
            .finish_non_exhaustive()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use grafeo_common::memory::buffer::BufferManagerConfig;

    #[test]
    fn test_execution_context_creation() {
        let manager = BufferManager::with_budget(1024 * 1024);
        let ctx = ExecutionMemoryContext::new(manager);

        assert_eq!(ctx.total_allocated(), 0);
        assert_eq!(ctx.pressure_level(), PressureLevel::Normal);
    }

    #[test]
    fn test_execution_context_allocation() {
        let manager = BufferManager::with_budget(1024 * 1024);
        let mut ctx = ExecutionMemoryContext::new(manager);

        let grant = ctx.allocate(1024);
        assert!(grant.is_some());
        assert_eq!(ctx.total_allocated(), 1024);
    }

    #[test]
    fn test_execution_context_tracked_allocation() {
        let manager = BufferManager::with_budget(1024 * 1024);
        let mut ctx = ExecutionMemoryContext::new(manager);

        assert!(ctx.allocate_tracked(1024));
        assert_eq!(ctx.total_allocated(), 1024);

        ctx.release_all();
        assert_eq!(ctx.total_allocated(), 0);
    }

    #[test]
    fn test_adjusted_chunk_size_normal() {
        let manager = BufferManager::with_budget(1024 * 1024);
        let ctx = ExecutionMemoryContext::new(manager);

        assert_eq!(ctx.adjusted_chunk_size(2048), 2048);
        assert_eq!(ctx.optimal_chunk_size(), 2048);
    }

    #[test]
    fn test_adjusted_chunk_size_under_pressure() {
        let config = BufferManagerConfig {
            budget: 1000,
            soft_limit_fraction: 0.70,
            evict_limit_fraction: 0.85,
            hard_limit_fraction: 0.95,
            background_eviction: false,
            spill_path: None,
        };
        let manager = BufferManager::new(config);

        // Allocate to reach high pressure (>85%)
        let _g = manager.try_allocate(860, MemoryRegion::ExecutionBuffers);

        let ctx = ExecutionMemoryContext::new(manager);
        assert_eq!(ctx.pressure_level(), PressureLevel::High);
        assert_eq!(ctx.adjusted_chunk_size(2048), HIGH_PRESSURE_CHUNK_SIZE);
        assert!(ctx.should_reduce_chunk_size());
    }

    #[test]
    fn test_builder() {
        let manager = BufferManager::with_budget(1024 * 1024);

        let ctx = ExecutionMemoryContextBuilder::new(manager)
            .with_initial_allocation(4096)
            .build();

        assert!(ctx.is_some());
        let ctx = ctx.unwrap();
        assert_eq!(ctx.total_allocated(), 4096);
    }

    #[test]
    fn test_builder_insufficient_memory() {
        let manager = BufferManager::with_budget(1000);

        // Try to allocate more than available
        let ctx = ExecutionMemoryContextBuilder::new(manager)
            .with_initial_allocation(10000)
            .build();

        assert!(ctx.is_none());
    }
}
