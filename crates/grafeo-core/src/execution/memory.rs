//! Execution memory context for memory-aware query execution.

use grafeo_common::memory::buffer::{BufferManager, MemoryGrant, MemoryRegion, PressureLevel};
use std::sync::Arc;

// The statement memory of the operators: one statement's cap and the grants
// under it with spilling, nothing to count without it (the operators hold the
// same type in either build).
#[cfg(feature = "spill")]
mod budget;
#[cfg(not(feature = "spill"))]
mod no_budget;
mod scope;

#[cfg(feature = "spill")]
pub use budget::{
    ConsumerRegistration, OperatorGrant, OperatorMemory, OperatorMemoryContext, QueryMemory,
    StatementMemory, capture, current,
};
#[cfg(not(feature = "spill"))]
pub use no_budget::{OperatorMemory, StatementMemory, capture};
pub use scope::{MemoryScope, enter, in_scope, statement};

#[cfg(all(test, feature = "spill"))]
mod operator_memory_tests;

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

/// What a materializing operator that keeps whole chunks counts for `chunk`:
/// a value slot per row and column, the rows a selection hides included, and
/// what the string and generic columns hold on the heap (see
/// [`heap_bytes`](super::vector::ValueVector::heap_bytes)): one pass over
/// those columns, none over flat ones.
#[must_use]
pub fn chunk_bytes(chunk: &super::chunk::DataChunk) -> usize {
    chunk
        .total_row_count()
        .saturating_mul(chunk.column_count().max(1))
        .saturating_mul(std::mem::size_of::<grafeo_common::types::Value>())
        .saturating_add(std::mem::size_of::<super::chunk::DataChunk>())
        .saturating_add(chunk_heap_bytes(chunk))
}

/// What the string and generic columns of `chunk` hold on the heap, all rows
/// (see [`heap_bytes`](super::vector::ValueVector::heap_bytes)); 0 for a chunk
/// of flat columns, without reading their entries.
#[must_use]
pub fn chunk_heap_bytes(chunk: &super::chunk::DataChunk) -> usize {
    chunk
        .columns()
        .iter()
        .filter(|column| column.holds_heap())
        .map(super::vector::ValueVector::heap_bytes)
        .fold(0, usize::saturating_add)
}

/// The columns of a chunk whose values can hold heap memory (string and
/// generic ones, see [`holds_heap`](super::vector::ValueVector::holds_heap)),
/// picked once per chunk: what a hash table's kept keys hold is read from
/// these alone, and not at all when every key column is flat.
pub struct HeapColumns<'a>(smallvec::SmallVec<[&'a super::vector::ValueVector; 4]>);

impl<'a> HeapColumns<'a> {
    /// The heap-holding columns of `chunk` among `columns` (all of its
    /// columns with `None`).
    #[must_use]
    pub fn of(chunk: &'a super::chunk::DataChunk, columns: Option<&[usize]>) -> Self {
        let picked = match columns {
            Some(columns) => columns
                .iter()
                .filter_map(|&index| chunk.column(index))
                .filter(|column| column.holds_heap())
                .collect(),
            None => chunk
                .columns()
                .iter()
                .filter(|column| column.holds_heap())
                .collect(),
        };
        Self(picked)
    }

    /// No columns: for an operator that counts nothing, so it reads no
    /// values.
    #[must_use]
    pub fn none() -> Self {
        Self(smallvec::SmallVec::new())
    }

    /// The heap-holding key columns of `chunk` when `memory` counts, none
    /// otherwise (see [`none`](Self::none)).
    #[must_use]
    pub fn counted(
        memory: &OperatorMemory,
        chunk: &'a super::chunk::DataChunk,
        columns: Option<&[usize]>,
    ) -> Self {
        if memory.counts() {
            Self::of(chunk, columns)
        } else {
            Self::none()
        }
    }

    /// Whether no column holds heap memory: then a row holds none.
    #[must_use]
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    /// What the values of `row` in these columns hold on the heap.
    #[must_use]
    pub fn row_bytes(&self, row: usize) -> usize {
        self.0
            .iter()
            .map(|column| column.heap_bytes_at(row))
            .fold(0, usize::saturating_add)
    }

    /// What the values of `rows` in these columns hold on the heap; no read
    /// when no column holds any.
    #[must_use]
    pub fn rows_bytes(&self, rows: impl Iterator<Item = usize>) -> usize {
        if self.is_empty() {
            return 0;
        }
        rows.map(|row| self.row_bytes(row))
            .fold(0, usize::saturating_add)
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
