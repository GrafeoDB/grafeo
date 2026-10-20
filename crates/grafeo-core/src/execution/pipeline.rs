//! Push-based execution pipeline.
//!
//! This module provides push-based execution primitives where data flows
//! forward through operators via `push()` calls, enabling better parallelism
//! and cache utilization compared to pull-based execution.

use std::time::Instant;

use super::chunk::DataChunk;
use super::operators::OperatorError;

/// Hint for preferred chunk size.
///
/// Operators can provide hints to optimize chunk sizing for their workload.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum ChunkSizeHint {
    /// Use default chunk size (2048 tuples).
    Default,
    /// Use small chunks (256-512 tuples) for LIMIT or high selectivity.
    Small,
    /// Use large chunks (4096 tuples) for full scans.
    Large,
    /// Use exact chunk size.
    Exact(usize),
    /// Use at most this many tuples (for LIMIT).
    AtMost(usize),
}

impl Default for ChunkSizeHint {
    fn default() -> Self {
        Self::Default
    }
}

/// Default chunk size in tuples.
pub const DEFAULT_CHUNK_SIZE: usize = 2048;

/// Small chunk size for high selectivity or LIMIT.
pub const SMALL_CHUNK_SIZE: usize = 512;

/// Large chunk size for full scans.
pub const LARGE_CHUNK_SIZE: usize = 4096;

/// Source of data chunks for a pipeline.
///
/// Sources produce chunks of data that flow through the pipeline.
pub trait Source: Send + Sync {
    /// Produce the next chunk of data.
    ///
    /// Returns `Ok(Some(chunk))` if data is available, `Ok(None)` if exhausted.
    ///
    /// # Errors
    ///
    /// Returns `Err` if the source fails to produce data.
    fn next_chunk(&mut self, chunk_size: usize) -> Result<Option<DataChunk>, OperatorError>;

    /// Reset the source to its initial state.
    fn reset(&mut self);

    /// Name of this source for debugging.
    fn name(&self) -> &'static str;
}

/// Sink that receives output from operators.
///
/// Sinks consume data chunks produced by the pipeline.
pub trait Sink: Send + Sync {
    /// Consume a chunk of data.
    ///
    /// Returns `Ok(true)` to continue, `Ok(false)` to signal early termination.
    ///
    /// # Errors
    ///
    /// Returns `Err` if the sink fails to process the chunk.
    fn consume(&mut self, chunk: DataChunk) -> Result<bool, OperatorError>;

    /// Called when all input has been processed.
    ///
    /// # Errors
    ///
    /// Returns `Err` if finalization fails.
    fn finalize(&mut self) -> Result<(), OperatorError>;

    /// Name of this sink for debugging.
    fn name(&self) -> &'static str;

    /// Converts this boxed sink into `Box<dyn Any>` for type-based dispatch.
    fn into_any(self: Box<Self>) -> Box<dyn std::any::Any>;
}

/// Push-based operator trait.
///
/// Unlike pull-based operators that return data on `next()` calls,
/// push-based operators receive data via `push()` and forward results
/// to a downstream sink.
pub trait PushOperator: Send + Sync {
    /// Process an incoming chunk and push results to the sink.
    ///
    /// Returns `Ok(true)` to continue processing, `Ok(false)` for early termination.
    ///
    /// # Errors
    ///
    /// Returns `Err` if the operator or sink fails during processing.
    fn push(&mut self, chunk: DataChunk, sink: &mut dyn Sink) -> Result<bool, OperatorError>;

    /// Called when all input has been processed.
    ///
    /// Pipeline breakers (Sort, Aggregate, etc.) emit their results here.
    ///
    /// # Errors
    ///
    /// Returns `Err` if finalization or downstream sink consumption fails.
    fn finalize(&mut self, sink: &mut dyn Sink) -> Result<(), OperatorError>;

    /// Hint for preferred chunk size.
    fn preferred_chunk_size(&self) -> ChunkSizeHint {
        ChunkSizeHint::Default
    }

    /// Name of this operator for debugging.
    fn name(&self) -> &'static str;
}

/// Execution pipeline connecting source, operators, and sink.
pub struct Pipeline {
    source: Box<dyn Source>,
    operators: Vec<Box<dyn PushOperator>>,
    sink: Box<dyn Sink>,
    /// Optional wall-clock deadline after which execution is aborted.
    deadline: Option<Instant>,
}

impl Pipeline {
    /// Create a new pipeline.
    pub fn new(
        source: Box<dyn Source>,
        operators: Vec<Box<dyn PushOperator>>,
        sink: Box<dyn Sink>,
    ) -> Self {
        Self {
            source,
            operators,
            sink,
            deadline: None,
        }
    }

    /// Create a simple pipeline with just source and sink.
    pub fn simple(source: Box<dyn Source>, sink: Box<dyn Sink>) -> Self {
        Self {
            source,
            operators: Vec::new(),
            sink,
            deadline: None,
        }
    }

    /// Add an operator to the pipeline.
    #[must_use]
    pub fn with_operator(mut self, op: Box<dyn PushOperator>) -> Self {
        self.operators.push(op);
        self
    }

    /// Sets a wall-clock deadline for pipeline execution.
    ///
    /// When set, the pipeline checks between chunks whether the deadline has
    /// been exceeded and aborts with a timeout error if so.
    #[must_use]
    pub fn with_deadline(mut self, deadline: Option<Instant>) -> Self {
        self.deadline = deadline;
        self
    }

    /// Sets the deadline on an already-constructed pipeline.
    pub fn set_deadline(&mut self, deadline: Option<Instant>) {
        self.deadline = deadline;
    }

    /// Checks whether the deadline has been exceeded.
    ///
    /// On WASM targets this is a no-op because `Instant` is not available.
    fn check_deadline(&self) -> Result<(), OperatorError> {
        #[cfg(not(target_arch = "wasm32"))]
        if let Some(deadline) = self.deadline
            && Instant::now() >= deadline
        {
            return Err(OperatorError::Timeout);
        }
        Ok(())
    }

    /// Consumes the pipeline and returns the sink.
    ///
    /// Call this after [`execute()`](Self::execute) to retrieve collected results
    /// from the sink. Useful for extracting chunks from a [`CollectorSink`](super::sink::CollectorSink)
    /// or [`ChunkCollector`].
    pub fn into_sink(self) -> Box<dyn Sink> {
        self.sink
    }

    /// Execute the pipeline.
    ///
    /// # Deadline behavior
    ///
    /// The deadline is checked **between** chunk iterations, not during source reads.
    /// A slow source read (e.g., a complex scan or join) can exceed the deadline
    /// before cancellation is evaluated. This matches the pull-based executor's
    /// behavior and is an accepted trade-off: cooperative cancellation between
    /// chunks avoids the complexity of interrupt-based cancellation within operators.
    ///
    /// # Errors
    ///
    /// Returns `Err` if any source, operator, or sink fails during execution,
    /// or if the configured deadline is exceeded between chunks.
    pub fn execute(&mut self) -> Result<(), OperatorError> {
        let chunk_size = self.compute_chunk_size();

        // Process all chunks from source
        while let Some(chunk) = self.source.next_chunk(chunk_size)? {
            self.check_deadline()?;

            if !self.push_through(chunk)? {
                // Early termination requested
                break;
            }
        }

        // Finalize all operators (important for pipeline breakers)
        self.finalize_all()
    }

    /// Compute optimal chunk size from operator hints.
    fn compute_chunk_size(&self) -> usize {
        let mut size = DEFAULT_CHUNK_SIZE;

        for op in &self.operators {
            match op.preferred_chunk_size() {
                ChunkSizeHint::Default => {}
                ChunkSizeHint::Small => size = size.min(SMALL_CHUNK_SIZE),
                ChunkSizeHint::Large => size = size.max(LARGE_CHUNK_SIZE),
                ChunkSizeHint::Exact(s) => return s,
                ChunkSizeHint::AtMost(s) => size = size.min(s),
            }
        }

        size
    }

    /// Push a chunk through the operator chain.
    fn push_through(&mut self, chunk: DataChunk) -> Result<bool, OperatorError> {
        if self.operators.is_empty() {
            // No operators, push directly to sink
            return self.sink.consume(chunk);
        }

        // Build a chain: operators push to each other, final one pushes to sink
        let mut current_chunk = chunk;
        let num_operators = self.operators.len();

        for i in 0..num_operators {
            let is_last = i == num_operators - 1;

            if is_last {
                // Last operator pushes to the real sink
                return self.operators[i].push(current_chunk, &mut *self.sink);
            }

            // Intermediate operators collect output
            let mut collector = ChunkCollector::new();
            let continue_processing = self.operators[i].push(current_chunk, &mut collector)?;

            if !continue_processing || collector.is_empty() {
                return Ok(continue_processing);
            }

            // Merge collected chunks for next operator
            current_chunk = collector.into_single_chunk();
        }

        Ok(true)
    }

    /// Finalize in source order, forwarding each batch through the remaining
    /// operators immediately so downstream stop/error reaches its producer.
    fn finalize_all(&mut self) -> Result<(), OperatorError> {
        for index in 0..self.operators.len() {
            self.check_deadline()?;
            let (_, pending) = self.operators.split_at_mut(index);
            let (operator, remaining) = pending.split_first_mut().expect("index is in bounds");
            let mut forwarding = FinalizeForwarder {
                operators: remaining,
                sink: &mut *self.sink,
                deadline: self.deadline,
                stopped: false,
            };
            operator.finalize(&mut forwarding)?;
            // Stop this producer immediately, but still flush downstream
            // operators: LIMIT may have left accepted rows in a buffered sort
            // or aggregate before it refused the next producer batch.
        }
        self.check_deadline()?;
        self.sink.finalize()
    }
}

/// This borrowed adapter lives only during one finalization call.
struct FinalizeForwarder<'a> {
    operators: &'a mut [Box<dyn PushOperator>],
    sink: &'a mut dyn Sink,
    deadline: Option<Instant>,
    stopped: bool,
}
impl Sink for FinalizeForwarder<'_> {
    fn consume(&mut self, chunk: DataChunk) -> Result<bool, OperatorError> {
        if self.stopped {
            return Ok(false);
        }
        if self
            .deadline
            .is_some_and(|deadline| Instant::now() >= deadline)
        {
            return Err(OperatorError::Timeout);
        }
        let continued = if let Some((operator, remaining)) = self.operators.split_first_mut() {
            let mut forwarding = FinalizeForwarder {
                operators: remaining,
                sink: self.sink,
                deadline: self.deadline,
                stopped: false,
            };
            let continued = operator.push(chunk, &mut forwarding)?;
            continued && !forwarding.stopped
        } else {
            self.sink.consume(chunk)?
        };
        self.stopped = !continued;
        Ok(continued)
    }
    fn finalize(&mut self) -> Result<(), OperatorError> {
        Ok(())
    }
    fn name(&self) -> &'static str {
        "FinalizeForwarder"
    }
    fn into_any(self: Box<Self>) -> Box<dyn std::any::Any> {
        unreachable!("borrowed finalization adapter cannot be detached")
    }
}

/// Collects chunks from operators for intermediate processing.
pub struct ChunkCollector {
    chunks: Vec<DataChunk>,
    #[cfg(feature = "spill")]
    chunks_owner: Option<super::memory::GroupedMemoryOwner>,
}

impl ChunkCollector {
    /// Create a new chunk collector.
    pub fn new() -> Self {
        Self {
            chunks: Vec::new(),
            #[cfg(feature = "spill")]
            chunks_owner: None,
        }
    }

    /// Check if collector has any chunks.
    pub fn is_empty(&self) -> bool {
        self.chunks.is_empty()
    }

    /// Get total row count across all chunks.
    pub fn row_count(&self) -> usize {
        self.chunks.iter().map(DataChunk::len).sum()
    }

    /// Convert to a vector of chunks.
    pub fn into_chunks(self) -> Vec<DataChunk> {
        self.chunks
    }

    /// Transfers chunks together with the reservation for their outer container.
    ///
    /// Keep the owner until the vector storage has been released. Each chunk
    /// separately retains its data reservation.
    #[cfg(feature = "spill")]
    pub fn into_accounted_chunks(
        self,
    ) -> (Vec<DataChunk>, Option<super::memory::GroupedMemoryOwner>) {
        (self.chunks, self.chunks_owner)
    }

    #[cfg(feature = "spill")]
    fn admit_chunk_slot(&mut self, chunk: &mut DataChunk) -> Result<(), OperatorError> {
        use super::memory::{checked_grouped_bytes, grouped_memory_limit};
        let manager = chunk.owner_mut().map(|owner| owner.manager().clone());
        if let Some(manager) = manager {
            if let Some(owner) = &self.chunks_owner {
                if !std::sync::Arc::ptr_eq(owner.manager(), &manager) {
                    return Err(OperatorError::Internal(
                        "collected chunks must belong to the same memory manager".to_string(),
                    ));
                }
            } else {
                let mut owner = chunk
                    .owner_mut()
                    .expect("accounted chunk retains its owner")
                    .split(0)?;
                owner.try_grow(checked_grouped_bytes(
                    self.chunks.capacity(),
                    size_of::<DataChunk>(),
                )?)?;
                self.chunks_owner = Some(owner);
            }
        }
        let Some(owner) = &mut self.chunks_owner else {
            return Ok(());
        };
        if self.chunks.len() < self.chunks.capacity() {
            return Ok(());
        }
        let old_bytes = owner.size();
        let capacity = self
            .chunks
            .capacity()
            .checked_mul(2)
            .map(|n| n.max(4))
            .ok_or_else(grouped_memory_limit)?;
        owner.try_grow(checked_grouped_bytes(capacity, size_of::<DataChunk>())?)?;
        if self
            .chunks
            .try_reserve_exact(capacity - self.chunks.len())
            .is_err()
        {
            owner.try_resize(old_bytes)?;
            return Err(grouped_memory_limit());
        }
        owner.try_resize(checked_grouped_bytes(
            self.chunks.capacity(),
            size_of::<DataChunk>(),
        )?)?;
        Ok(())
    }

    /// Merge all chunks into a single chunk.
    ///
    /// # Panics
    ///
    /// Panics if internal invariants are violated (single-element vec is unexpectedly empty).
    pub fn into_single_chunk(self) -> DataChunk {
        if self.chunks.is_empty() {
            return DataChunk::empty();
        }
        if self.chunks.len() == 1 {
            // Invariant: self.chunks.len() == 1 guarantees exactly one element
            return self
                .chunks
                .into_iter()
                .next()
                .expect("chunks has exactly one element: checked on previous line");
        }

        // Concatenate all chunks
        DataChunk::concat(&self.chunks)
    }
}

impl Default for ChunkCollector {
    fn default() -> Self {
        Self::new()
    }
}

impl Sink for ChunkCollector {
    fn consume(&mut self, chunk: DataChunk) -> Result<bool, OperatorError> {
        #[cfg(feature = "spill")]
        let mut chunk = chunk;
        if !chunk.is_empty() {
            #[cfg(feature = "spill")]
            self.admit_chunk_slot(&mut chunk)?;
            self.chunks.push(chunk);
        }
        Ok(true)
    }

    fn finalize(&mut self) -> Result<(), OperatorError> {
        Ok(())
    }

    fn name(&self) -> &'static str {
        "ChunkCollector"
    }

    fn into_any(self: Box<Self>) -> Box<dyn std::any::Any> {
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::execution::vector::ValueVector;
    use grafeo_common::types::Value;

    /// Test source that produces a fixed number of chunks.
    struct TestSource {
        remaining: usize,
        values_per_chunk: usize,
    }

    impl TestSource {
        fn new(num_chunks: usize, values_per_chunk: usize) -> Self {
            Self {
                remaining: num_chunks,
                values_per_chunk,
            }
        }
    }

    impl Source for TestSource {
        fn next_chunk(&mut self, _chunk_size: usize) -> Result<Option<DataChunk>, OperatorError> {
            if self.remaining == 0 {
                return Ok(None);
            }
            self.remaining -= 1;

            // Create a chunk with integer values
            // reason: test chunk size is small, fits i64
            #[allow(clippy::cast_possible_wrap)]
            let values: Vec<Value> = (0..self.values_per_chunk)
                .map(|i| Value::Int64(i as i64))
                .collect();
            let vector = ValueVector::from_values(&values);
            let chunk = DataChunk::new(vec![vector]);
            Ok(Some(chunk))
        }

        fn reset(&mut self) {}

        fn name(&self) -> &'static str {
            "TestSource"
        }
    }

    /// Test sink that collects all chunks.
    struct TestSink {
        chunks: Vec<DataChunk>,
        finalized: bool,
    }

    impl TestSink {
        fn new() -> Self {
            Self {
                chunks: Vec::new(),
                finalized: false,
            }
        }
    }

    impl Sink for TestSink {
        fn consume(&mut self, chunk: DataChunk) -> Result<bool, OperatorError> {
            self.chunks.push(chunk);
            Ok(true)
        }

        fn finalize(&mut self) -> Result<(), OperatorError> {
            self.finalized = true;
            Ok(())
        }

        fn name(&self) -> &'static str {
            "TestSink"
        }

        fn into_any(self: Box<Self>) -> Box<dyn std::any::Any> {
            self
        }
    }

    /// Pass-through operator for testing.
    struct PassThroughOperator;

    impl PushOperator for PassThroughOperator {
        fn push(&mut self, chunk: DataChunk, sink: &mut dyn Sink) -> Result<bool, OperatorError> {
            sink.consume(chunk)
        }

        fn finalize(&mut self, _sink: &mut dyn Sink) -> Result<(), OperatorError> {
            Ok(())
        }

        fn name(&self) -> &'static str {
            "PassThrough"
        }
    }

    #[test]
    fn test_simple_pipeline() {
        let source = Box::new(TestSource::new(3, 10));
        let sink = Box::new(TestSink::new());

        let mut pipeline = Pipeline::simple(source, sink);
        pipeline.execute().unwrap();

        // Access sink through downcast (in real code we'd use a different pattern)
        // For this test, we verify execution completed without error
    }

    #[test]
    fn test_pipeline_with_operator() {
        let source = Box::new(TestSource::new(2, 5));
        let sink = Box::new(TestSink::new());

        let mut pipeline =
            Pipeline::simple(source, sink).with_operator(Box::new(PassThroughOperator));

        pipeline.execute().unwrap();
    }

    #[test]
    fn test_chunk_collector() {
        let mut collector = ChunkCollector::new();
        assert!(collector.is_empty());

        let values: Vec<Value> = vec![Value::Int64(1), Value::Int64(2)];
        let vector = ValueVector::from_values(&values);
        let chunk = DataChunk::new(vec![vector]);

        collector.consume(chunk).unwrap();
        assert!(!collector.is_empty());
        assert_eq!(collector.row_count(), 2);

        let merged = collector.into_single_chunk();
        assert_eq!(merged.len(), 2);
    }

    #[test]
    fn test_pipeline_deadline_expired() {
        use std::time::{Duration, Instant};

        let source = Box::new(TestSource::new(10, 5));
        let sink = Box::new(TestSink::new());

        // Set a deadline that has already passed
        let expired = Instant::now().checked_sub(Duration::from_secs(1)).unwrap();
        let mut pipeline = Pipeline::simple(source, sink).with_deadline(Some(expired));

        let result = pipeline.execute();
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(
            err.to_string().contains("Query exceeded timeout"),
            "Expected timeout error, got: {err}"
        );
    }

    #[test]
    fn test_pipeline_no_deadline() {
        let source = Box::new(TestSource::new(3, 5));
        let sink = Box::new(TestSink::new());

        // No deadline should execute normally
        let mut pipeline = Pipeline::simple(source, sink).with_deadline(None);
        pipeline.execute().unwrap();
    }

    #[test]
    fn test_pipeline_set_deadline() {
        use std::time::{Duration, Instant};

        let source = Box::new(TestSource::new(10, 5));
        let sink = Box::new(TestSink::new());

        let mut pipeline = Pipeline::simple(source, sink);

        // Set deadline after construction
        let expired = Instant::now().checked_sub(Duration::from_secs(1)).unwrap();
        pipeline.set_deadline(Some(expired));

        let result = pipeline.execute();
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(
            err.to_string().contains("Query exceeded timeout"),
            "Expected timeout error, got: {err}"
        );
    }

    #[test]
    fn test_chunk_size_hints() {
        assert_eq!(ChunkSizeHint::default(), ChunkSizeHint::Default);

        let source = Box::new(TestSource::new(1, 10));
        let sink = Box::new(TestSink::new());

        // Test with small hint operator
        struct SmallHintOp;
        impl PushOperator for SmallHintOp {
            fn push(
                &mut self,
                chunk: DataChunk,
                sink: &mut dyn Sink,
            ) -> Result<bool, OperatorError> {
                sink.consume(chunk)
            }
            fn finalize(&mut self, _sink: &mut dyn Sink) -> Result<(), OperatorError> {
                Ok(())
            }
            fn preferred_chunk_size(&self) -> ChunkSizeHint {
                ChunkSizeHint::Small
            }
            fn name(&self) -> &'static str {
                "SmallHint"
            }
        }

        let pipeline = Pipeline::simple(source, sink).with_operator(Box::new(SmallHintOp));

        let computed_size = pipeline.compute_chunk_size();
        assert!(computed_size <= SMALL_CHUNK_SIZE);
    }
    struct FinalizeBatches(std::sync::Arc<std::sync::atomic::AtomicUsize>);
    impl PushOperator for FinalizeBatches {
        fn push(&mut self, _chunk: DataChunk, _sink: &mut dyn Sink) -> Result<bool, OperatorError> {
            Ok(true)
        }
        fn finalize(&mut self, sink: &mut dyn Sink) -> Result<(), OperatorError> {
            for value in [1, 2] {
                self.0.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                let chunk = DataChunk::new(vec![ValueVector::from_values(&[Value::Int64(value)])]);
                if !sink.consume(chunk)? {
                    break;
                }
            }
            Ok(())
        }
        fn name(&self) -> &'static str {
            "FinalizeBatches"
        }
    }

    #[test]
    fn limit_stops_before_second_drain_batch() {
        let produced = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let mut pipeline = Pipeline::new(
            Box::new(TestSource::new(0, 0)),
            vec![
                Box::new(FinalizeBatches(produced.clone())),
                Box::new(crate::execution::operators::push::LimitPushOperator::new(1)),
            ],
            Box::new(ChunkCollector::new()),
        );
        pipeline.execute().unwrap();
        assert_eq!(produced.load(std::sync::atomic::Ordering::SeqCst), 1);
        let sink = pipeline
            .into_sink()
            .into_any()
            .downcast::<ChunkCollector>()
            .unwrap();
        assert_eq!(sink.row_count(), 1);
    }

    #[test]
    fn limit_stop_still_finalizes_downstream_sort() {
        let produced = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let mut pipeline = Pipeline::new(
            Box::new(TestSource::new(0, 0)),
            vec![
                Box::new(FinalizeBatches(produced.clone())),
                Box::new(crate::execution::operators::push::LimitPushOperator::new(1)),
                Box::new(crate::execution::operators::push::SortPushOperator::ascending(0)),
            ],
            Box::new(ChunkCollector::new()),
        );
        pipeline.execute().unwrap();
        assert_eq!(produced.load(std::sync::atomic::Ordering::SeqCst), 1);
        let sink = pipeline
            .into_sink()
            .into_any()
            .downcast::<ChunkCollector>()
            .unwrap();
        assert_eq!(sink.row_count(), 1);
        let chunks = sink.into_chunks();
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].column_count(), 1);
        assert_eq!(
            chunks[0].column(0).unwrap().get_value(0),
            Some(Value::Int64(1))
        );
    }

    #[test]
    fn deadline_prevents_empty_source_finalization() {
        let produced = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let mut pipeline = Pipeline::new(
            Box::new(TestSource::new(0, 0)),
            vec![Box::new(FinalizeBatches(produced.clone()))],
            Box::new(ChunkCollector::new()),
        )
        .with_deadline(Some(Instant::now()));
        assert!(matches!(pipeline.execute(), Err(OperatorError::Timeout)));
        assert_eq!(produced.load(std::sync::atomic::Ordering::SeqCst), 0);
    }
    #[test]
    #[cfg(feature = "spill")]
    fn owned_collector_denies_container_growth_before_retaining_chunk() {
        use crate::execution::memory::GroupedMemoryOwner;
        use grafeo_common::memory::buffer::{BufferManager, BufferManagerConfig, MemoryRegion};
        use grafeo_common::types::LogicalType;
        let budget = 1024 * 1024;
        let manager = BufferManager::new(BufferManagerConfig {
            budget,
            hard_limit_fraction: 1.0,
            ..BufferManagerConfig::default()
        });
        let owner = GroupedMemoryOwner::try_new(manager.clone(), 0).unwrap();
        let mut chunk = DataChunk::try_with_owner(&[LogicalType::Int64], 1, owner).unwrap();
        chunk.try_push_owned_value(0, Value::Int64(7)).unwrap();
        chunk.set_count(1);
        let held = manager
            .try_allocate(budget - manager.allocated(), MemoryRegion::ExecutionBuffers)
            .unwrap();
        let mut collector = ChunkCollector::new();
        assert!(matches!(
            collector.consume(chunk),
            Err(OperatorError::LimitExceeded(_))
        ));
        assert!(collector.is_empty());
        drop(collector);
        assert_eq!(manager.allocated(), held.size());
        drop(held);
        assert_eq!(manager.allocated(), 0);
    }

    #[test]
    #[cfg(feature = "spill")]
    fn owned_collector_handoff_keeps_container_and_chunk_grants() {
        use crate::execution::memory::GroupedMemoryOwner;
        use grafeo_common::memory::buffer::{BufferManager, BufferManagerConfig};
        use grafeo_common::types::LogicalType;
        let manager = BufferManager::new(BufferManagerConfig::default());
        let mut collector = ChunkCollector::new();
        for value in 0..5 {
            let owner = GroupedMemoryOwner::try_new(manager.clone(), 0).unwrap();
            let mut chunk = DataChunk::try_with_owner(&[LogicalType::Int64], 1, owner).unwrap();
            chunk.try_push_owned_value(0, Value::Int64(value)).unwrap();
            chunk.set_count(1);
            collector.consume(chunk).unwrap();
        }
        let retained = manager.allocated();
        let (chunks, owner) = collector.into_accounted_chunks();
        assert_eq!(chunks.len(), 5);
        assert_eq!(manager.allocated(), retained);
        let container_bytes = owner.as_ref().unwrap().size();
        assert_eq!(container_bytes, chunks.capacity() * size_of::<DataChunk>());
        drop(chunks);
        assert_eq!(manager.allocated(), container_bytes);
        drop(owner);
        assert_eq!(manager.allocated(), 0);
    }

    #[test]
    #[cfg(feature = "spill")]
    fn owned_collector_rejects_foreign_manager_without_losing_retained_chunk() {
        use crate::execution::memory::GroupedMemoryOwner;
        use grafeo_common::memory::buffer::{BufferManager, BufferManagerConfig};
        use grafeo_common::types::LogicalType;
        let first = BufferManager::new(BufferManagerConfig::default());
        let second = BufferManager::new(BufferManagerConfig::default());
        let mut collector = ChunkCollector::new();
        for manager in [&first, &second] {
            let owner = GroupedMemoryOwner::try_new(manager.clone(), 0).unwrap();
            let mut chunk = DataChunk::try_with_owner(&[LogicalType::Int64], 1, owner).unwrap();
            chunk.try_push_owned_value(0, Value::Int64(7)).unwrap();
            chunk.set_count(1);
            let result = collector.consume(chunk);
            if std::sync::Arc::ptr_eq(manager, &first) {
                assert!(result.is_ok());
            } else {
                assert!(matches!(result, Err(OperatorError::Internal(_))));
            }
        }
        assert_eq!(collector.row_count(), 1);
        assert!(first.allocated() > 0);
        assert_eq!(second.allocated(), 0);
        drop(collector);
        assert_eq!(first.allocated(), 0);
    }
}
