//! Execution primitives for vectorized query processing.
//!
//! This module provides the core data structures for vectorized execution:
//!
//! - [`chunk`] - DataChunk for batched tuple processing
//! - [`vector`] - ValueVector for columnar storage
//! - [`selection`] - SelectionVector for filtering
//! - [`operators`] - Physical operators (scan, filter, project, join)
//! - [`pipeline`] - Push-based execution pipeline
//! - [`sink`] - Common sink implementations
//! - [`memory`] - Memory-aware execution context
//! - [`parallel`] - Morsel-driven parallel execution
//! - [`spill`] - Transparent spilling for out-of-core processing

pub mod chunk;
pub mod memory;
pub mod operators;
pub mod parallel;
pub mod pipeline;
pub mod selection;
pub mod sink;
pub mod source;
pub mod spill;
pub mod vector;

pub use chunk::DataChunk;
pub use memory::{ExecutionMemoryContext, ExecutionMemoryContextBuilder};
pub use parallel::{
    CloneableOperatorFactory, MorselScheduler, ParallelPipeline, ParallelPipelineConfig,
    ParallelSource, RangeSource,
};
pub use pipeline::{ChunkCollector, ChunkSizeHint, Pipeline, PushOperator, Sink, Source};
pub use selection::SelectionVector;
pub use sink::{CollectorSink, CountingSink, LimitingSink, MaterializingSink, NullSink};
pub use source::{ChunkSource, EmptySource, GeneratorSource, OperatorSource, VectorSource};
pub use spill::{SpillFile, SpillFileReader, SpillManager};
pub use vector::ValueVector;
