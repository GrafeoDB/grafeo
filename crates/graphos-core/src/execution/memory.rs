//! Execution memory context for memory-aware query execution.

use graphos_common::memory::buffer::{
    BufferManager, MemoryGrant, MemoryRegion, PressureLevel,
};
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
        let grant = self.manager.try_allocate(size, MemoryRegion::ExecutionBuffers)?;
        self.allocated += size;
        Some(grant)
    }

    /// Allocates and stores a grant internally.
    ///
    /// The grant will be released when this context is dropped.
    pub fn allocate_tracked(&mut self, size: usize) -> bool {
        if let Some(grant) = self.manager.try_allocate(size, MemoryRegion::ExecutionBuffers) {
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

#[cfg(test)]
mod tests {
    use super::*;
    use graphos_common::memory::buffer::BufferManagerConfig;

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
