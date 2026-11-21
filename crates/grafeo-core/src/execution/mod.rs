//! Vectorized query execution engine.
//!
//! Grafeo uses vectorized processing - instead of one row at a time, we process
//! batches of ~1024 rows. This unlocks SIMD and keeps the CPU busy.
//!
//! | Module | Purpose |
//! | ------ | ------- |
//! | [`chunk`] | Batched rows (DataChunk = multiple columns) |
//! | [`vector`] | Single column of values |
//! | [`factorized_vector`] | Factorized vectors for avoiding Cartesian products |
//! | [`factorized_chunk`] | Multi-level factorized chunks |
//! | [`selection`] | Bitmap for filtering without copying |
//! | [`operators`] | Physical operators (scan, filter, join, etc.) |
//! | [`pipeline`] | Push-based execution (data flows through operators) |
//! | [`parallel`] | Morsel-driven parallelism |
//! | [`spill`] | Disk spilling when memory is tight |
//! | [`collector`] | Generic collector pattern for parallel aggregation |
//! | [`deadline`] | The query deadline the long loops of the operators check |
//!
//! The execution model is push-based: sources push data through a pipeline of
//! operators until it reaches a sink.

pub mod chunk;
pub mod chunk_state;
pub mod collector;
pub mod deadline;
pub mod factorized_chunk;
pub mod factorized_iter;
pub mod factorized_vector;
pub mod memory;
pub mod operators;
#[cfg(feature = "parallel")]
pub mod parallel;
pub(crate) mod path_budget;
pub(crate) mod path_queue;
pub mod pipeline;
pub mod pipeline_convert;
pub mod profile;
pub mod selection;
pub mod sink;
pub mod source;
// Always built: its account bounds the vector columns' spill too; the
// operators' spill files need the `spill` feature.
pub mod spill;
pub mod vector;

pub use chunk::{ChunkZoneHints, DataChunk};
pub use collector::{
    Collector, CollectorStats, CountCollector, LimitCollector, MaterializeCollector,
    PartitionCollector, StatsCollector,
};
pub use memory::{
    ConsumerRegistration, OperatorGrant, OperatorMemoryContext, QueryMemory,
    current as current_operator_memory,
};
pub use memory::{ExecutionMemoryContext, ExecutionMemoryContextBuilder};
pub use memory::{
    MemoryScope, OperatorMemory, StatementMemory, enter as enter_operator_memory,
    in_scope as in_operator_memory_scope, statement as current_statement_memory,
};
pub use operators::PATH_QUEUE_ENTRY_BYTES;
/// The bytes one segment of a followed path holds, for capacity accounting
/// and tests.
pub use operators::PATH_SEGMENT_BYTES;
#[cfg(feature = "parallel")]
pub use parallel::{
    CloneableOperatorFactory, MorselScheduler, ParallelPipeline, ParallelPipelineConfig,
    ParallelSource, RangeSource,
};
/// The entries a path search's queue reservation counts beyond its doubled
/// capacity, for capacity accounting and tests.
pub use path_budget::RESERVE_SLACK_ENTRIES as PATH_RESERVE_SLACK_ENTRIES;
/// The bytes of one block of a path search's queue, for capacity accounting
/// and tests.
pub use path_queue::BLOCK_BYTES as PATH_QUEUE_BLOCK_BYTES;
/// The bytes of one slot of a path search queue's block index, for capacity
/// accounting and tests.
pub use path_queue::INDEX_SLOT_BYTES as PATH_QUEUE_INDEX_SLOT_BYTES;
/// The path search budget from which every block of a path search's queue is
/// full size, for capacity accounting and tests.
pub use path_queue::SMALL_BUDGET_BYTES as PATH_QUEUE_SMALL_BUDGET_BYTES;
pub use pipeline::{ChunkCollector, ChunkSizeHint, Pipeline, PushOperator, Sink, Source};
pub use profile::{ProfileStats, ProfiledOperator, SharedProfileStats};
pub use selection::SelectionVector;
pub use sink::{CollectorSink, CountingSink, LimitingSink, MaterializingSink, NullSink};
pub use source::{ChunkSource, EmptySource, GeneratorSource, OperatorSource, VectorSource};
#[cfg(feature = "spill")]
pub use spill::{SpillFile, SpillFileReader, SpillManager};
pub use vector::ValueVector;

// Factorized execution types
pub use chunk_state::{ChunkState, FactorizationState, FactorizedSelection, LevelSelection};
pub use factorized_chunk::{ChunkVariant, FactorizationLevel, FactorizedChunk};
pub use factorized_iter::{PrecomputedIter, RowIndices, RowView, StreamingIter};
pub use factorized_vector::{FactorizedState, FactorizedVector, UnflatMetadata};
pub use operators::{FactorizedData, FlatDataWrapper};
