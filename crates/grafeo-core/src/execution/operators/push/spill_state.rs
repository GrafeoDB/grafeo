//! Shared state for memory-aware spill operators.
//!
//! Provides the bridge between the `MemoryConsumer` trait (which uses `&self`
//! for eviction callbacks) and `PushOperator` (which uses `&mut self` for
//! data processing). The cooperative model works as follows:
//!
//! 1. `OperatorConsumerAdapter` is registered with the `BufferManager`.
//! 2. When memory pressure triggers eviction, the adapter sets an atomic flag.
//! 3. On the operator's next `push()` call, it checks the flag and spills.

use grafeo_common::memory::buffer::{MemoryConsumer, MemoryRegion, SpillError, priorities};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

/// Shared state between a spill-capable operator and its `MemoryConsumer` adapter.
///
/// Uses lock-free atomics so the eviction thread (which calls `evict()` via
/// `&self`) can signal the execution thread (which calls `push()` via `&mut self`)
/// without mutex contention on the hot path.
pub(crate) struct OperatorSpillState {
    /// Name for debugging/logging and consumer registration.
    name: String,
    /// Approximate memory usage in bytes, updated by the operator after each push.
    usage: AtomicUsize,
    /// Set by the consumer adapter when BufferManager requests eviction.
    eviction_requested: AtomicBool,
    /// Target bytes the BufferManager wants freed (informational).
    eviction_target: AtomicUsize,
}

impl OperatorSpillState {
    /// Creates a new spill state with the given consumer name.
    pub(crate) fn new(name: String) -> Self {
        Self {
            name,
            usage: AtomicUsize::new(0),
            eviction_requested: AtomicBool::new(false),
            eviction_target: AtomicUsize::new(0),
        }
    }

    /// Updates the approximate memory usage (called by the operator on each push).
    pub(crate) fn set_usage(&self, bytes: usize) {
        self.usage.store(bytes, Ordering::Relaxed);
    }

    /// Returns the current approximate memory usage in bytes.
    pub(crate) fn usage(&self) -> usize {
        self.usage.load(Ordering::Relaxed)
    }

    /// Requests eviction of the given number of bytes (called by the consumer adapter).
    pub(crate) fn request_eviction(&self, target_bytes: usize) {
        self.eviction_target.store(target_bytes, Ordering::Relaxed);
        self.eviction_requested.store(true, Ordering::Release);
    }

    /// Checks and clears the eviction request flag.
    ///
    /// Returns `Some(target_bytes)` if eviction was requested, `None` otherwise.
    /// The flag is cleared atomically so each request is processed exactly once.
    pub(crate) fn take_eviction_request(&self) -> Option<usize> {
        if self.eviction_requested.swap(false, Ordering::AcqRel) {
            Some(self.eviction_target.load(Ordering::Relaxed))
        } else {
            None
        }
    }

    /// Returns the consumer name.
    pub(crate) fn name(&self) -> &str {
        &self.name
    }
}

/// Adapts an `OperatorSpillState` into a `MemoryConsumer` for BufferManager registration.
///
/// When the BufferManager calls `evict()` or `spill()`, this adapter sets the
/// eviction flag on the shared state and returns 0 (deferred eviction). The
/// actual spilling happens in the operator's next `push()` call.
pub(crate) struct OperatorConsumerAdapter {
    state: Arc<OperatorSpillState>,
}

impl OperatorConsumerAdapter {
    /// Creates a new adapter wrapping the given shared state.
    pub(crate) fn new(state: Arc<OperatorSpillState>) -> Self {
        Self { state }
    }
}

impl MemoryConsumer for OperatorConsumerAdapter {
    fn name(&self) -> &str {
        self.state.name()
    }

    fn memory_usage(&self) -> usize {
        self.state.usage()
    }

    fn eviction_priority(&self) -> u8 {
        priorities::EXECUTION_BUFFERS
    }

    fn region(&self) -> MemoryRegion {
        MemoryRegion::ExecutionBuffers
    }

    fn evict(&self, target_bytes: usize) -> usize {
        // Deferred eviction: set the flag, actual spilling happens on next push().
        self.state.request_eviction(target_bytes);
        0
    }

    fn can_evict(&self) -> bool {
        true
    }

    fn can_spill(&self) -> bool {
        true
    }

    fn spill(&self, target_bytes: usize) -> Result<usize, SpillError> {
        // Deferred: signal the operator to spill on its next push().
        self.state.request_eviction(target_bytes);
        Ok(0)
    }

    fn current_tier(&self) -> grafeo_common::memory::buffer::StorageTier {
        // Operator-internal state lives in heap buffers managed by the
        // operator itself; the buffer manager only sees an aggregate
        // memory_usage. We surface that as `InMemory`/`Uninitialized` and
        // let `reload_eligible` skip these consumers (operators do not
        // expose a reload entry point).
        if self.memory_usage() == 0 {
            grafeo_common::memory::buffer::StorageTier::Uninitialized
        } else {
            grafeo_common::memory::buffer::StorageTier::InMemory
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_spill_state_usage_tracking() {
        let state = OperatorSpillState::new("test_sort".to_string());
        assert_eq!(state.usage(), 0);

        state.set_usage(1024);
        assert_eq!(state.usage(), 1024);

        state.set_usage(0);
        assert_eq!(state.usage(), 0);
    }

    #[test]
    fn test_spill_state_eviction_request() {
        let state = OperatorSpillState::new("test_agg".to_string());

        // No request initially
        assert!(state.take_eviction_request().is_none());

        // Request eviction
        state.request_eviction(4096);
        assert_eq!(state.take_eviction_request(), Some(4096));

        // Flag is cleared after take
        assert!(state.take_eviction_request().is_none());
    }

    #[test]
    fn test_spill_state_multiple_requests_last_wins() {
        let state = OperatorSpillState::new("test".to_string());

        state.request_eviction(1000);
        state.request_eviction(2000);

        // Last target wins (but flag is set either way)
        let target = state.take_eviction_request();
        assert!(target.is_some());
        assert_eq!(target.unwrap(), 2000);
    }

    #[test]
    fn test_consumer_adapter_reports_correct_metadata() {
        let state = Arc::new(OperatorSpillState::new("sort_op_1".to_string()));
        state.set_usage(8192);

        let adapter = OperatorConsumerAdapter::new(Arc::clone(&state));

        assert_eq!(adapter.name(), "sort_op_1");
        assert_eq!(adapter.memory_usage(), 8192);
        assert_eq!(adapter.eviction_priority(), priorities::EXECUTION_BUFFERS);
        assert_eq!(adapter.region(), MemoryRegion::ExecutionBuffers);
        assert!(adapter.can_spill());
    }

    #[test]
    fn test_consumer_adapter_evict_sets_flag() {
        let state = Arc::new(OperatorSpillState::new("agg_op".to_string()));
        let adapter = OperatorConsumerAdapter::new(Arc::clone(&state));

        // evict() returns 0 (deferred) but sets the flag
        let freed = adapter.evict(4096);
        assert_eq!(freed, 0);
        assert_eq!(state.take_eviction_request(), Some(4096));
    }

    #[test]
    fn test_consumer_adapter_spill_sets_flag() {
        let state = Arc::new(OperatorSpillState::new("sort_op".to_string()));
        let adapter = OperatorConsumerAdapter::new(Arc::clone(&state));

        let result = adapter.spill(2048);
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), 0);
        assert_eq!(state.take_eviction_request(), Some(2048));
    }
}
