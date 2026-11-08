//! Execution memory context for memory-aware query execution.

use grafeo_common::memory::buffer::{BufferManager, MemoryGrant, MemoryRegion, PressureLevel};
#[cfg(feature = "spill")]
use grafeo_common::utils::error::{Error, LimitBound};
#[cfg(feature = "spill")]
use std::cell::RefCell;
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
    /// Per-query spill manager for creating spill files (`None`: this open
    /// has no spill path, so its operators cannot spill).
    spill_manager: Option<Arc<super::spill::SpillManager>>,
    /// What one statement's operators may hold: the query memory limit, never
    /// above the buffer manager's budget.
    query_limit: usize,
}

#[cfg(feature = "spill")]
impl OperatorMemoryContext {
    /// Creates a new operator memory context.
    #[must_use]
    pub fn new(
        buffer_manager: Arc<BufferManager>,
        spill_manager: Option<Arc<super::spill::SpillManager>>,
    ) -> Self {
        let query_limit = buffer_manager.budget();
        Self {
            buffer_manager,
            spill_manager,
            query_limit,
        }
    }

    /// Sets the query memory limit this context bounds its operators by.
    #[must_use]
    pub fn with_query_limit(mut self, limit: usize) -> Self {
        self.query_limit = limit.min(self.buffer_manager.budget());
        self
    }

    /// The query memory limit: what one statement's operators may hold.
    #[must_use]
    pub fn query_limit(&self) -> usize {
        self.query_limit
    }

    /// Grows `grant` to cover `counted` bytes, bounded by the query memory
    /// limit: over it the refusal names the limit and its setting. The grant
    /// itself stays under the buffer manager's budget.
    ///
    /// # Errors
    ///
    /// [`LimitBound::QueryMemoryLimit`] when `counted` passes the query
    /// limit, or the buffer manager's refusal when the budget itself is out.
    pub fn grow(&self, grant: &mut Option<MemoryGrant>, counted: usize) -> Result<(), Error> {
        if counted > self.query_limit {
            return Err(self
                .buffer_manager
                .refusal(LimitBound::QueryMemoryLimit, counted)
                .into());
        }
        let step = self.buffer_manager.grant_step();
        // Round up to the grant step, but never past the query limit: the
        // reservation must not exceed the configured bound.
        let wanted = counted
            .checked_next_multiple_of(step)
            .unwrap_or(usize::MAX)
            .min(self.query_limit);
        let held = grant.as_ref().map_or(0, MemoryGrant::size);
        if held < wanted {
            let grant = grant.get_or_insert_with(|| {
                self.buffer_manager
                    .empty_grant(MemoryRegion::ExecutionBuffers)
            });
            if !grant.try_grow(wanted - held) {
                let refusal = self.buffer_manager.refusal(LimitBound::MemoryLimit, wanted);
                return Err(refusal.into());
            }
        }
        Ok(())
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

    /// Returns the spill manager, when this open has a spill path.
    #[must_use]
    pub fn spill_manager(&self) -> Option<&Arc<super::spill::SpillManager>> {
        self.spill_manager.as_ref()
    }
}

#[cfg(feature = "spill")]
thread_local! {
    /// The operator memory context of the statement this thread plans and
    /// converts, if it has one (see [`enter`]).
    static OPERATOR_MEMORY: RefCell<Option<OperatorMemoryContext>> = const { RefCell::new(None) };
}

/// Sets the running statement's operator memory context on this thread until
/// the returned scope drops, which restores the one before. Operators built
/// while the scope lives capture it at construction (see [`current`]).
#[cfg(feature = "spill")]
pub fn enter(ctx: OperatorMemoryContext) -> MemoryScope {
    let previous = OPERATOR_MEMORY.with(|cell| cell.replace(Some(ctx)));
    MemoryScope {
        previous,
        _not_send: std::marker::PhantomData,
    }
}

/// The running statement's operator memory context on this thread, for an
/// operator built while its statement is planned (see [`enter`]).
#[cfg(feature = "spill")]
#[must_use]
pub fn current() -> Option<OperatorMemoryContext> {
    OPERATOR_MEMORY.with(|cell| cell.borrow().clone())
}

/// What an operator built now captures: the ambient scope's context (the
/// statement being planned and converted), or `None` when the site passes the
/// explicit unbudgeted marker (`unbudgeted`) because it really has no
/// statement (the dead `QueryProcessor` entry point, deleted in 0.6.3). In
/// debug builds a materializing operator built with neither fails loudly
/// instead of silently getting no budget (patch 9, #751).
#[cfg(feature = "spill")]
pub fn capture(unbudgeted: bool) -> Option<OperatorMemoryContext> {
    let current = current();
    debug_assert!(
        unbudgeted || current.is_some(),
        "a materializing operator built outside a statement's planning scope"
    );
    current
}

/// The operator memory context [`enter`] set, until it drops.
#[cfg(feature = "spill")]
#[must_use = "the context holds only while the scope lives"]
pub struct MemoryScope {
    /// The context before [`enter`], restored on drop.
    previous: Option<OperatorMemoryContext>,
    /// Thread-local state restores on drop, so the scope must not move across
    /// threads.
    _not_send: std::marker::PhantomData<*const ()>,
}

#[cfg(feature = "spill")]
impl Drop for MemoryScope {
    fn drop(&mut self) {
        OPERATOR_MEMORY.with(|cell| cell.replace(self.previous.take()));
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
