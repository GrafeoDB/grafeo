//! Push-based execution pipeline.
//!
//! This module provides push-based execution primitives where data flows
//! forward through operators via `push()` calls, enabling better parallelism
//! and cache utilization compared to pull-based execution.

use grafeo_common::clock::Instant;

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
    /// The deadline is checked between the source's chunks, and it is the
    /// running query's [`deadline`](super::deadline) while the pipeline runs,
    /// so the operators under the source check it in their own loops too (an
    /// aggregate or a join that reads its whole input inside one source
    /// read). A query stops within about one chunk of work past it.
    ///
    /// # Errors
    ///
    /// Returns `Err` if any source, operator, or sink fails during execution,
    /// or [`OperatorError::Timeout`] once the deadline has passed.
    pub fn execute(&mut self) -> Result<(), OperatorError> {
        let _deadline = super::deadline::enter(self.deadline);
        let chunk_size = self.compute_chunk_size();

        // Process all chunks from source
        while let Some(chunk) = self.source.next_chunk(chunk_size)? {
            super::deadline::check()?;

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
        push_chain(&mut self.operators, &mut *self.sink, chunk)
    }

    /// Finalizes the operators, then the sink (see [`finalize_chain`]).
    fn finalize_all(&mut self) -> Result<(), OperatorError> {
        finalize_chain(&mut self.operators, &mut self.sink)
    }
}

/// Pushes `chunk` through `operators`, each into the next and the last into
/// `sink`. `Ok(false)` once one of them stops: an operator that stops still
/// hands on the rows it emitted with the stop (a LIMIT reaching its count),
/// and they go through the rest first.
pub(crate) fn push_chain(
    operators: &mut [Box<dyn PushOperator>],
    sink: &mut dyn Sink,
    chunk: DataChunk,
) -> Result<bool, OperatorError> {
    let Some((last, before)) = operators.split_last_mut() else {
        return sink.consume(chunk);
    };
    let mut stopped = false;
    let mut current = chunk;
    for operator in before {
        // One push's output, bounded by that push.
        let mut collector = ChunkCollector::new();
        stopped |= !operator.push(current, &mut collector)?;
        if collector.is_empty() {
            return Ok(!stopped);
        }
        current = collector.into_single_chunk();
    }
    Ok(last.push(current, sink)? && !stopped)
}

/// Finalizes `operators` in order, then `sink`. What a breaker emits as it
/// finalizes goes through the operators after it a chunk as it comes, so a
/// breaker's output is never held whole; once they stop, the breaker is told
/// to stop and is sent nothing more. Each later operator is still finalized
/// once, in order: a stop means no more input, not "skip your own output".
pub(crate) fn finalize_chain(
    operators: &mut Vec<Box<dyn PushOperator>>,
    sink: &mut Box<dyn Sink>,
) -> Result<(), OperatorError> {
    for i in 0..operators.len() {
        if i + 1 == operators.len() {
            operators[i].finalize(&mut **sink)?;
            break;
        }
        // The rest of the chain, owned by the forwarding sink while operator
        // `i` finalizes (a sink is `'static`), and taken back, error or not.
        let mut forward = Forwarding {
            operators: operators.split_off(i + 1),
            sink: std::mem::replace(sink, Box::new(super::sink::NullSink)),
            stopped: false,
        };
        let finalized = operators[i].finalize(&mut forward);
        operators.append(&mut forward.operators);
        *sink = forward.sink;
        finalized?;
    }
    sink.finalize()
}

/// The sink a breaker finalizes into: each chunk goes through the rest of
/// the chain at once. Once the rest has stopped it takes nothing more.
struct Forwarding {
    operators: Vec<Box<dyn PushOperator>>,
    sink: Box<dyn Sink>,
    stopped: bool,
}

impl Sink for Forwarding {
    fn consume(&mut self, chunk: DataChunk) -> Result<bool, OperatorError> {
        if self.stopped {
            return Ok(false);
        }
        // A breaker's output is the query's work too: past the deadline it
        // stops within a chunk.
        super::deadline::check()?;
        let go_on = push_chain(&mut self.operators, &mut *self.sink, chunk)?;
        self.stopped = !go_on;
        Ok(go_on)
    }

    fn finalize(&mut self) -> Result<(), OperatorError> {
        Ok(())
    }

    fn name(&self) -> &'static str {
        "Forwarding"
    }

    fn into_any(self: Box<Self>) -> Box<dyn std::any::Any> {
        self
    }
}

/// Collects chunks from operators for intermediate processing.
pub struct ChunkCollector {
    chunks: Vec<DataChunk>,
}

impl ChunkCollector {
    /// Create a new chunk collector.
    pub fn new() -> Self {
        Self { chunks: Vec::new() }
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
        if !chunk.is_empty() {
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

/// Stub operators and a sink that record what a finalize sends where: the
/// pull pipeline's tests here and the parallel pipeline's use them.
#[cfg(test)]
pub(crate) mod finalize_stubs {
    use super::{DataChunk, OperatorError, PushOperator, Sink};
    use crate::execution::vector::ValueVector;
    use grafeo_common::types::Value;
    use std::sync::{Arc, Mutex};

    /// What happened, in order: every operator's finalize, every chunk a
    /// breaker emits, every chunk a limit gets, the sink's finalize.
    pub(crate) type Log = Arc<Mutex<Vec<String>>>;

    pub(crate) fn note(log: &Log, event: String) {
        log.lock().unwrap().push(event);
    }

    /// How many entries of `log` start with `prefix`.
    pub(crate) fn count(log: &Log, prefix: &str) -> usize {
        log.lock()
            .unwrap()
            .iter()
            .filter(|event| event.starts_with(prefix))
            .count()
    }

    /// The finalize entries of `log`, in order.
    pub(crate) fn finalizes(log: &Log) -> Vec<String> {
        log.lock()
            .unwrap()
            .iter()
            .filter(|event| event.starts_with("finalize"))
            .cloned()
            .collect()
    }

    pub(crate) fn one_row(value: i64) -> DataChunk {
        DataChunk::new(vec![ValueVector::from_values(&[Value::Int64(value)])])
    }

    /// The first column of every chunk, in order.
    pub(crate) fn values(chunks: &[DataChunk]) -> Vec<i64> {
        chunks
            .iter()
            .flat_map(|chunk| {
                chunk.selected_indices().map(|row| {
                    match chunk.column(0).and_then(|column| column.get_value(row)) {
                        Some(Value::Int64(value)) => value,
                        other => panic!("not an integer: {other:?}"),
                    }
                })
            })
            .collect()
    }

    /// A breaker: keeps no input, and its finalize emits `chunks` one-row
    /// chunks (0, 1, ...). One that `honours` the flag stops at the first
    /// `Ok(false)`; one that does not keeps emitting.
    pub(crate) struct Emitter {
        pub(crate) name: &'static str,
        pub(crate) chunks: i64,
        pub(crate) honours: bool,
        pub(crate) log: Log,
    }

    impl PushOperator for Emitter {
        fn push(&mut self, _chunk: DataChunk, _sink: &mut dyn Sink) -> Result<bool, OperatorError> {
            Ok(true)
        }

        fn finalize(&mut self, sink: &mut dyn Sink) -> Result<(), OperatorError> {
            note(&self.log, format!("finalize {}", self.name));
            for value in 0..self.chunks {
                note(&self.log, format!("{} emits {value}", self.name));
                if !sink.consume(one_row(value))? && self.honours {
                    break;
                }
            }
            Ok(())
        }

        fn name(&self) -> &'static str {
            self.name
        }
    }

    /// A limit of `chunks` chunks: forwards the first ones and says
    /// `Ok(false)` once it has them; forwards nothing after.
    pub(crate) struct StopAfter {
        pub(crate) name: &'static str,
        pub(crate) chunks: usize,
        pub(crate) seen: usize,
        pub(crate) log: Log,
    }

    impl PushOperator for StopAfter {
        fn push(&mut self, chunk: DataChunk, sink: &mut dyn Sink) -> Result<bool, OperatorError> {
            self.seen += 1;
            note(&self.log, format!("{} gets {}", self.name, self.seen));
            if self.seen <= self.chunks {
                sink.consume(chunk)?;
            }
            Ok(self.seen < self.chunks)
        }

        fn finalize(&mut self, _sink: &mut dyn Sink) -> Result<(), OperatorError> {
            note(&self.log, format!("finalize {}", self.name));
            Ok(())
        }

        fn name(&self) -> &'static str {
            self.name
        }
    }

    /// Forwards every chunk and returns what its sink says.
    pub(crate) struct Pass {
        pub(crate) name: &'static str,
        pub(crate) log: Log,
    }

    impl PushOperator for Pass {
        fn push(&mut self, chunk: DataChunk, sink: &mut dyn Sink) -> Result<bool, OperatorError> {
            sink.consume(chunk)
        }

        fn finalize(&mut self, _sink: &mut dyn Sink) -> Result<(), OperatorError> {
            note(&self.log, format!("finalize {}", self.name));
            Ok(())
        }

        fn name(&self) -> &'static str {
            self.name
        }
    }

    /// A breaker that keeps what it gets and emits it at its finalize.
    pub(crate) struct Hold {
        pub(crate) name: &'static str,
        pub(crate) held: Vec<DataChunk>,
        pub(crate) log: Log,
    }

    impl PushOperator for Hold {
        fn push(&mut self, chunk: DataChunk, _sink: &mut dyn Sink) -> Result<bool, OperatorError> {
            self.held.push(chunk);
            Ok(true)
        }

        fn finalize(&mut self, sink: &mut dyn Sink) -> Result<(), OperatorError> {
            note(&self.log, format!("finalize {}", self.name));
            for chunk in std::mem::take(&mut self.held) {
                sink.consume(chunk)?;
            }
            Ok(())
        }

        fn name(&self) -> &'static str {
            self.name
        }
    }

    /// The pipeline's sink: keeps every chunk, and says `Ok(false)` from
    /// its `stop_after`-th chunk on.
    pub(crate) struct Recording {
        pub(crate) chunks: Arc<Mutex<Vec<DataChunk>>>,
        pub(crate) stop_after: Option<usize>,
        pub(crate) log: Log,
    }

    impl Sink for Recording {
        fn consume(&mut self, chunk: DataChunk) -> Result<bool, OperatorError> {
            let mut chunks = self.chunks.lock().unwrap();
            chunks.push(chunk);
            Ok(self.stop_after.is_none_or(|stop| chunks.len() < stop))
        }

        fn finalize(&mut self) -> Result<(), OperatorError> {
            note(&self.log, "finalize sink".to_string());
            Ok(())
        }

        fn name(&self) -> &'static str {
            "Recording"
        }

        fn into_any(self: Box<Self>) -> Box<dyn std::any::Any> {
            self
        }
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

    mod streamed_finalize {
        use super::super::finalize_stubs::{
            Emitter, Hold, Log, Pass, Recording, StopAfter, count, finalizes, values,
        };
        use super::*;
        use std::sync::{Arc, Mutex};

        /// Runs one input chunk through `operators` into a recording sink
        /// that stops from its `stop_after`-th chunk on: the rows it got.
        fn run(
            operators: Vec<Box<dyn PushOperator>>,
            stop_after: Option<usize>,
            log: &Log,
        ) -> Vec<i64> {
            let chunks = Arc::new(Mutex::new(Vec::new()));
            let sink = Recording {
                chunks: Arc::clone(&chunks),
                stop_after,
                log: Arc::clone(log),
            };
            let mut pipeline =
                Pipeline::new(Box::new(TestSource::new(1, 1)), operators, Box::new(sink));
            pipeline.execute().unwrap();
            values(&chunks.lock().unwrap())
        }

        fn emitter(log: &Log, honours: bool) -> Box<dyn PushOperator> {
            Box::new(Emitter {
                name: "A",
                chunks: 5,
                honours,
                log: Arc::clone(log),
            })
        }

        fn stop_after_one(log: &Log) -> Box<dyn PushOperator> {
            Box::new(StopAfter {
                name: "B",
                chunks: 1,
                seen: 0,
                log: Arc::clone(log),
            })
        }

        fn pass(log: &Log) -> Box<dyn PushOperator> {
            Box::new(Pass {
                name: "P",
                log: Arc::clone(log),
            })
        }

        /// A stop during a breaker's finalize reaches it at once: from the
        /// operator right after it, from one further on through a pass, and
        /// from the pipeline's sink through a pass. A breaker that honours
        /// the flag then emits nothing more: one chunk of its five.
        #[test]
        fn a_stop_after_a_breaker_stops_its_finalize() {
            let log = Log::default();
            assert_eq!(
                run(vec![emitter(&log, true), stop_after_one(&log)], None, &log),
                [0]
            );
            assert_eq!(count(&log, "A emits"), 1, "from the next operator: {log:?}");

            let log = Log::default();
            let operators = vec![emitter(&log, true), pass(&log), stop_after_one(&log)];
            assert_eq!(run(operators, None, &log), [0]);
            assert_eq!(count(&log, "A emits"), 1, "from two on: {log:?}");

            let log = Log::default();
            assert_eq!(
                run(vec![emitter(&log, true), pass(&log)], Some(1), &log),
                [0]
            );
            assert_eq!(count(&log, "A emits"), 1, "from the sink: {log:?}");
        }

        /// A stop means no more input, not "skip your finalize": every
        /// operator after the breaker is finalized once, in order, and a
        /// later breaker still emits what it holds.
        #[test]
        fn every_operator_after_a_stop_is_finalized_once_in_order() {
            let log = Log::default();
            let hold = Box::new(Hold {
                name: "H",
                held: Vec::new(),
                log: Arc::clone(&log),
            });
            let rows = run(
                vec![emitter(&log, true), stop_after_one(&log), hold],
                None,
                &log,
            );
            assert_eq!(rows, [0], "the later breaker emits what it holds");
            assert_eq!(
                finalizes(&log),
                ["finalize A", "finalize B", "finalize H", "finalize sink"]
            );
        }

        /// A breaker that ignores the flag keeps emitting, and nothing of
        /// it is delivered after the stop: the limit gets one chunk.
        #[test]
        fn a_breaker_that_ignores_the_stop_delivers_nothing_after_it() {
            let log = Log::default();
            assert_eq!(
                run(vec![emitter(&log, false), stop_after_one(&log)], None, &log),
                [0]
            );
            assert_eq!(count(&log, "A emits"), 5, "it ignores the flag");
            assert_eq!(
                count(&log, "B gets"),
                1,
                "delivered after the stop: {log:?}"
            );

            let log = Log::default();
            assert_eq!(
                run(vec![emitter(&log, false), pass(&log)], Some(1), &log),
                [0]
            );
        }

        /// A breaker that emits a chunk every 5 ms, 200 in all, and checks
        /// no deadline itself: when each chunk went out.
        struct Slow(Arc<Mutex<Vec<std::time::Instant>>>);

        impl PushOperator for Slow {
            fn push(
                &mut self,
                _chunk: DataChunk,
                _sink: &mut dyn Sink,
            ) -> Result<bool, OperatorError> {
                Ok(true)
            }

            fn finalize(&mut self, sink: &mut dyn Sink) -> Result<(), OperatorError> {
                for value in 0..200 {
                    std::thread::sleep(std::time::Duration::from_millis(5));
                    self.0.lock().unwrap().push(std::time::Instant::now());
                    if !sink.consume(super::super::finalize_stubs::one_row(value))? {
                        break;
                    }
                }
                Ok(())
            }

            fn name(&self) -> &'static str {
                "Slow"
            }
        }

        /// A deadline that passes during a breaker's finalize stops the
        /// pipeline with a timeout within about one chunk: at most two of
        /// the breaker's chunks go out at or past it.
        #[test]
        fn a_deadline_during_a_finalize_stops_it_within_a_chunk() {
            let log = Log::default();
            let emitted = Arc::new(Mutex::new(Vec::new()));
            let deadline = std::time::Instant::now() + std::time::Duration::from_millis(100);
            let sink = Recording {
                chunks: Arc::new(Mutex::new(Vec::new())),
                stop_after: None,
                log: Arc::clone(&log),
            };
            let mut pipeline = Pipeline::new(
                Box::new(TestSource::new(1, 1)),
                vec![Box::new(Slow(Arc::clone(&emitted))), pass(&log)],
                Box::new(sink),
            )
            .with_deadline(Some(deadline));
            let result = pipeline.execute();
            let emitted = emitted.lock().unwrap();
            let late = emitted.iter().filter(|at| **at >= deadline).count();
            assert!(
                matches!(result, Err(OperatorError::Timeout)),
                "{result:?} after {} chunks",
                emitted.len()
            );
            assert!(late <= 2, "{late} chunks went out past the deadline");
        }
    }
}
