//! Parallel pipeline execution with morsel-driven scheduling.
//!
//! Executes push-based pipelines in parallel using work-stealing schedulers
//! and per-worker operator instances.

use super::morsel::{DEFAULT_MORSEL_SIZE, compute_morsel_size};
use super::scheduler::MorselScheduler;
use super::source::ParallelSource;
use crate::execution::chunk::DataChunk;
use crate::execution::operators::OperatorError;
use crate::execution::pipeline::{ChunkCollector, DEFAULT_CHUNK_SIZE, PushOperator, Sink};
use grafeo_common::memory::buffer::PressureLevel;
use parking_lot::Mutex;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::thread;

/// Factory for creating per-worker operator chains.
///
/// Each worker needs its own operator instances since operators may have
/// internal state (e.g., aggregation accumulators).
pub trait OperatorChainFactory: Send + Sync {
    /// Creates a new operator chain for a worker.
    ///
    /// Each call should return fresh operator instances.
    fn create_chain(&self) -> Vec<Box<dyn PushOperator>>;

    /// Returns whether the chain contains pipeline breakers.
    ///
    /// Pipeline breakers (Sort, Aggregate, Distinct) require a merge phase.
    fn has_pipeline_breakers(&self) -> bool;

    /// Returns the number of operators in the chain.
    fn chain_length(&self) -> usize;
}

/// Simple factory that clones a prototype chain.
pub struct CloneableOperatorFactory {
    /// Factory functions for each operator.
    factories: Vec<Box<dyn Fn() -> Box<dyn PushOperator> + Send + Sync>>,
    /// Whether chain has pipeline breakers.
    has_breakers: bool,
}

impl CloneableOperatorFactory {
    /// Creates a new factory.
    pub fn new() -> Self {
        Self {
            factories: Vec::new(),
            has_breakers: false,
        }
    }

    /// Adds an operator factory.
    #[must_use]
    pub fn with_operator<F>(mut self, factory: F) -> Self
    where
        F: Fn() -> Box<dyn PushOperator> + Send + Sync + 'static,
    {
        self.factories.push(Box::new(factory));
        self
    }

    /// Marks that the chain has pipeline breakers.
    #[must_use]
    pub fn with_pipeline_breakers(mut self) -> Self {
        self.has_breakers = true;
        self
    }
}

impl Default for CloneableOperatorFactory {
    fn default() -> Self {
        Self::new()
    }
}

impl OperatorChainFactory for CloneableOperatorFactory {
    fn create_chain(&self) -> Vec<Box<dyn PushOperator>> {
        self.factories.iter().map(|f| f()).collect()
    }

    fn has_pipeline_breakers(&self) -> bool {
        self.has_breakers
    }

    fn chain_length(&self) -> usize {
        self.factories.len()
    }
}

/// Result of parallel pipeline execution.
pub struct ParallelPipelineResult {
    /// Output chunks from all workers.
    pub chunks: Vec<DataChunk>,
    /// Number of workers used.
    pub num_workers: usize,
    /// Total morsels processed.
    pub morsels_processed: usize,
    /// Total rows processed.
    pub rows_processed: usize,
}

/// Configuration for parallel pipeline execution.
#[derive(Debug, Clone)]
pub struct ParallelPipelineConfig {
    /// Number of worker threads.
    pub num_workers: usize,
    /// Base morsel size (adjusted for memory pressure).
    pub morsel_size: usize,
    /// Chunk size for processing within morsels.
    pub chunk_size: usize,
    /// Whether to preserve output ordering.
    pub preserve_order: bool,
    /// Memory pressure level (affects morsel sizing).
    pub pressure_level: PressureLevel,
}

impl Default for ParallelPipelineConfig {
    fn default() -> Self {
        Self {
            num_workers: thread::available_parallelism()
                .map(|n| n.get())
                .unwrap_or(4),
            morsel_size: DEFAULT_MORSEL_SIZE,
            chunk_size: DEFAULT_CHUNK_SIZE,
            preserve_order: false,
            pressure_level: PressureLevel::Normal,
        }
    }
}

impl ParallelPipelineConfig {
    /// Creates config for testing with limited workers.
    #[must_use]
    pub fn for_testing() -> Self {
        Self {
            num_workers: 2,
            ..Default::default()
        }
    }

    /// Sets the number of workers.
    #[must_use]
    pub fn with_workers(mut self, n: usize) -> Self {
        self.num_workers = n.max(1);
        self
    }

    /// Sets memory pressure level.
    #[must_use]
    pub fn with_pressure(mut self, level: PressureLevel) -> Self {
        self.pressure_level = level;
        self
    }

    /// Returns effective morsel size based on pressure.
    #[must_use]
    pub fn effective_morsel_size(&self) -> usize {
        compute_morsel_size(self.pressure_level)
    }
}

/// Parallel execution pipeline.
///
/// Distributes work across multiple threads using morsel-driven scheduling.
pub struct ParallelPipeline {
    /// Source of data (must support parallel partitioning).
    source: Arc<dyn ParallelSource>,
    /// Factory for creating operator chains.
    operator_factory: Arc<dyn OperatorChainFactory>,
    /// Configuration.
    config: ParallelPipelineConfig,
}

impl ParallelPipeline {
    /// Creates a new parallel pipeline.
    pub fn new(
        source: Arc<dyn ParallelSource>,
        operator_factory: Arc<dyn OperatorChainFactory>,
        config: ParallelPipelineConfig,
    ) -> Self {
        Self {
            source,
            operator_factory,
            config,
        }
    }

    /// Creates a simple parallel pipeline with default config.
    pub fn simple(
        source: Arc<dyn ParallelSource>,
        operator_factory: Arc<dyn OperatorChainFactory>,
    ) -> Self {
        Self::new(source, operator_factory, ParallelPipelineConfig::default())
    }

    /// Executes the pipeline and returns results.
    pub fn execute(&self) -> Result<ParallelPipelineResult, OperatorError> {
        let morsel_size = self.config.effective_morsel_size();
        let morsels = self.source.generate_morsels(morsel_size, 0);

        if morsels.is_empty() {
            return Ok(ParallelPipelineResult {
                chunks: Vec::new(),
                num_workers: self.config.num_workers,
                morsels_processed: 0,
                rows_processed: 0,
            });
        }

        // Create scheduler and submit morsels
        let scheduler = Arc::new(MorselScheduler::new(self.config.num_workers));
        let total_morsels = morsels.len();
        scheduler.submit_batch(morsels);
        scheduler.finish_submission();

        // Shared results collector
        let results = Arc::new(Mutex::new(Vec::new()));
        let rows_processed = Arc::new(AtomicUsize::new(0));
        let errors: Arc<Mutex<Option<OperatorError>>> = Arc::new(Mutex::new(None));

        // Spawn workers
        thread::scope(|s| {
            for worker_id in 0..self.config.num_workers {
                let scheduler = Arc::clone(&scheduler);
                let source = Arc::clone(&self.source);
                let factory = Arc::clone(&self.operator_factory);
                let results = Arc::clone(&results);
                let rows_processed = Arc::clone(&rows_processed);
                let errors = Arc::clone(&errors);
                let chunk_size = self.config.chunk_size;

                s.spawn(move || {
                    if let Err(e) = Self::worker_loop(
                        worker_id,
                        scheduler,
                        source,
                        factory,
                        results,
                        rows_processed,
                        chunk_size,
                    ) {
                        let mut guard = errors.lock();
                        if guard.is_none() {
                            *guard = Some(e);
                        }
                    }
                });
            }
        });

        // Check for errors
        if let Some(e) = errors.lock().take() {
            return Err(e);
        }

        let chunks = match Arc::try_unwrap(results) {
            Ok(mutex) => mutex.into_inner(),
            Err(arc) => arc.lock().clone(),
        };

        Ok(ParallelPipelineResult {
            chunks,
            num_workers: self.config.num_workers,
            morsels_processed: total_morsels,
            rows_processed: rows_processed.load(Ordering::Relaxed),
        })
    }

    /// Worker loop: process morsels until done.
    fn worker_loop(
        _worker_id: usize,
        scheduler: Arc<MorselScheduler>,
        source: Arc<dyn ParallelSource>,
        factory: Arc<dyn OperatorChainFactory>,
        results: Arc<Mutex<Vec<DataChunk>>>,
        rows_processed: Arc<AtomicUsize>,
        chunk_size: usize,
    ) -> Result<(), OperatorError> {
        use super::scheduler::WorkerHandle;

        // Create worker handle (registers with scheduler for work-stealing)
        let handle = WorkerHandle::new(scheduler);

        // Create per-worker operator chain
        let mut operators = factory.create_chain();
        let mut local_sink = CollectorSink::new();

        // Process morsels
        while let Some(morsel) = handle.get_work() {
            let mut partition = source.create_partition(&morsel);
            let mut morsel_rows = 0;

            // Process chunks within morsel
            while let Some(chunk) = partition.next_chunk(chunk_size)? {
                morsel_rows += chunk.len();
                Self::push_through_chain(&mut operators, chunk, &mut local_sink)?;
            }

            rows_processed.fetch_add(morsel_rows, Ordering::Relaxed);
            handle.complete_morsel();
        }

        // Finalize operators (important for pipeline breakers)
        Self::finalize_chain(&mut operators, &mut local_sink)?;

        // Collect results
        let chunks = local_sink.into_chunks();
        if !chunks.is_empty() {
            results.lock().extend(chunks);
        }

        Ok(())
    }

    /// Pushes a chunk through the operator chain.
    fn push_through_chain(
        operators: &mut [Box<dyn PushOperator>],
        chunk: DataChunk,
        sink: &mut dyn Sink,
    ) -> Result<bool, OperatorError> {
        if operators.is_empty() {
            return sink.consume(chunk);
        }

        let num_operators = operators.len();
        let mut current_chunk = chunk;

        for i in 0..num_operators {
            let is_last = i == num_operators - 1;

            if is_last {
                return operators[i].push(current_chunk, sink);
            }

            // Intermediate: collect output
            let mut collector = ChunkCollector::new();
            let continue_processing = operators[i].push(current_chunk, &mut collector)?;

            if !continue_processing || collector.is_empty() {
                return Ok(continue_processing);
            }

            current_chunk = collector.into_single_chunk();
        }

        Ok(true)
    }

    /// Finalizes all operators in the chain.
    fn finalize_chain(
        operators: &mut [Box<dyn PushOperator>],
        sink: &mut dyn Sink,
    ) -> Result<(), OperatorError> {
        if operators.is_empty() {
            return sink.finalize();
        }

        let num_operators = operators.len();

        for i in 0..num_operators {
            let is_last = i == num_operators - 1;

            if is_last {
                operators[i].finalize(sink)?;
            } else {
                // Collect finalize output and push through remaining operators
                let mut collector = ChunkCollector::new();
                operators[i].finalize(&mut collector)?;

                // Push through remaining operators
                for chunk in collector.into_chunks() {
                    Self::push_through_from_index(operators, i + 1, chunk, sink)?;
                }
            }
        }

        sink.finalize()
    }

    /// Pushes a chunk through operators starting at index.
    fn push_through_from_index(
        operators: &mut [Box<dyn PushOperator>],
        start: usize,
        chunk: DataChunk,
        sink: &mut dyn Sink,
    ) -> Result<bool, OperatorError> {
        let num_operators = operators.len();
        let mut current_chunk = chunk;

        for i in start..num_operators {
            let is_last = i == num_operators - 1;

            if is_last {
                return operators[i].push(current_chunk, sink);
            }

            let mut collector = ChunkCollector::new();
            let continue_processing = operators[i].push(current_chunk, &mut collector)?;

            if !continue_processing || collector.is_empty() {
                return Ok(continue_processing);
            }

            current_chunk = collector.into_single_chunk();
        }

        sink.consume(current_chunk)
    }
}

/// Collector sink that accumulates chunks.
#[derive(Default)]
pub struct CollectorSink {
    chunks: Vec<DataChunk>,
}

impl CollectorSink {
    /// Creates a new collector sink.
    pub fn new() -> Self {
        Self { chunks: Vec::new() }
    }

    /// Returns collected chunks.
    pub fn into_chunks(self) -> Vec<DataChunk> {
        self.chunks
    }

    /// Returns number of chunks collected.
    pub fn len(&self) -> usize {
        self.chunks.len()
    }

    /// Returns whether no chunks collected.
    pub fn is_empty(&self) -> bool {
        self.chunks.is_empty()
    }

    /// Returns total row count.
    pub fn row_count(&self) -> usize {
        self.chunks.iter().map(DataChunk::len).sum()
    }
}

impl Sink for CollectorSink {
    fn consume(&mut self, chunk: DataChunk) -> Result<bool, OperatorError> {
        if !chunk.is_empty() {
            self.chunks.push(chunk);
        }
        Ok(true)
    }

    fn finalize(&mut self) -> Result<(), OperatorError> {
        Ok(())
    }

    fn name(&self) -> &'static str {
        "ParallelCollectorSink"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::execution::parallel::source::RangeSource;
    use crate::execution::vector::ValueVector;
    use grafeo_common::types::Value;

    /// Pass-through operator for testing.
    struct PassThroughOp;

    impl PushOperator for PassThroughOp {
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

    /// Filter operator that keeps only even numbers.
    struct EvenFilterOp;

    impl PushOperator for EvenFilterOp {
        fn push(&mut self, chunk: DataChunk, sink: &mut dyn Sink) -> Result<bool, OperatorError> {
            let col = chunk
                .column(0)
                .ok_or_else(|| OperatorError::Execution("Missing column".to_string()))?;

            let mut filtered = ValueVector::new();
            for i in 0..chunk.len() {
                if let Some(Value::Int64(v)) = col.get(i)
                    && v % 2 == 0
                {
                    filtered.push(Value::Int64(v));
                }
            }

            if !filtered.is_empty() {
                sink.consume(DataChunk::new(vec![filtered]))?;
            }
            Ok(true)
        }

        fn finalize(&mut self, _sink: &mut dyn Sink) -> Result<(), OperatorError> {
            Ok(())
        }

        fn name(&self) -> &'static str {
            "EvenFilter"
        }
    }

    #[test]
    fn test_parallel_pipeline_creation() {
        let source = Arc::new(RangeSource::new(1000));
        let factory = Arc::new(CloneableOperatorFactory::new());
        let config = ParallelPipelineConfig::for_testing();

        let pipeline = ParallelPipeline::new(source, factory, config);
        assert_eq!(pipeline.config.num_workers, 2);
    }

    #[test]
    fn test_parallel_pipeline_empty_source() {
        let source = Arc::new(RangeSource::new(0));
        let factory = Arc::new(CloneableOperatorFactory::new());

        let pipeline = ParallelPipeline::simple(source, factory);
        let result = pipeline.execute().unwrap();

        assert!(result.chunks.is_empty());
        assert_eq!(result.morsels_processed, 0);
        assert_eq!(result.rows_processed, 0);
    }

    #[test]
    fn test_parallel_pipeline_passthrough() {
        let source = Arc::new(RangeSource::new(100));
        let factory =
            Arc::new(CloneableOperatorFactory::new().with_operator(|| Box::new(PassThroughOp)));
        let config = ParallelPipelineConfig::for_testing();

        let pipeline = ParallelPipeline::new(source, factory, config);
        let result = pipeline.execute().unwrap();

        // Should process all 100 rows
        let total_rows: usize = result.chunks.iter().map(DataChunk::len).sum();
        assert_eq!(total_rows, 100);
        assert_eq!(result.rows_processed, 100);
    }

    #[test]
    fn test_parallel_pipeline_filter() {
        let source = Arc::new(RangeSource::new(100));
        let factory =
            Arc::new(CloneableOperatorFactory::new().with_operator(|| Box::new(EvenFilterOp)));
        let config = ParallelPipelineConfig::for_testing();

        let pipeline = ParallelPipeline::new(source, factory, config);
        let result = pipeline.execute().unwrap();

        // Should have 50 even numbers (0, 2, 4, ..., 98)
        let total_rows: usize = result.chunks.iter().map(DataChunk::len).sum();
        assert_eq!(total_rows, 50);
    }

    #[test]
    fn test_parallel_pipeline_multiple_workers() {
        let source = Arc::new(RangeSource::new(10000));
        let factory = Arc::new(CloneableOperatorFactory::new());
        let config = ParallelPipelineConfig::default().with_workers(4);

        let pipeline = ParallelPipeline::new(source, factory, config);
        let result = pipeline.execute().unwrap();

        let total_rows: usize = result.chunks.iter().map(DataChunk::len).sum();
        assert_eq!(total_rows, 10000);
        assert_eq!(result.num_workers, 4);
    }

    #[test]
    fn test_parallel_pipeline_under_pressure() {
        let source = Arc::new(RangeSource::new(10000));
        let factory = Arc::new(CloneableOperatorFactory::new());
        let config = ParallelPipelineConfig::for_testing().with_pressure(PressureLevel::High);

        let pipeline = ParallelPipeline::new(source, factory, config);
        let result = pipeline.execute().unwrap();

        // More morsels under pressure due to smaller size
        let total_rows: usize = result.chunks.iter().map(DataChunk::len).sum();
        assert_eq!(total_rows, 10000);
    }

    #[test]
    fn test_cloneable_operator_factory() {
        let factory = CloneableOperatorFactory::new()
            .with_operator(|| Box::new(PassThroughOp))
            .with_operator(|| Box::new(EvenFilterOp))
            .with_pipeline_breakers();

        assert_eq!(factory.chain_length(), 2);
        assert!(factory.has_pipeline_breakers());

        let chain = factory.create_chain();
        assert_eq!(chain.len(), 2);
    }

    #[test]
    fn test_collector_sink() {
        let mut sink = CollectorSink::new();
        assert!(sink.is_empty());

        let values = vec![Value::Int64(1), Value::Int64(2), Value::Int64(3)];
        let chunk = DataChunk::new(vec![ValueVector::from_values(&values)]);

        sink.consume(chunk).unwrap();
        assert!(!sink.is_empty());
        assert_eq!(sink.len(), 1);
        assert_eq!(sink.row_count(), 3);

        let chunks = sink.into_chunks();
        assert_eq!(chunks.len(), 1);
    }

    #[test]
    fn test_pipeline_config() {
        let config = ParallelPipelineConfig::default()
            .with_workers(8)
            .with_pressure(PressureLevel::Moderate);

        assert_eq!(config.num_workers, 8);
        assert_eq!(config.pressure_level, PressureLevel::Moderate);
        assert!(config.effective_morsel_size() < DEFAULT_MORSEL_SIZE);
    }
}
