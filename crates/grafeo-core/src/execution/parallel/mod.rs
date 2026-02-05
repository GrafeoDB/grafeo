//! Morsel-driven parallel execution engine.
//!
//! This module provides parallel query execution using morsel-driven scheduling
//! with work-stealing. Workers process data chunks (morsels) independently,
//! enabling linear scaling on multi-core systems.
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────┐
//! │                    ParallelPipeline                              │
//! │  ┌─────────────────────────────────────────────────────────────┐ │
//! │  │                MorselScheduler                               │ │
//! │  │   ┌─────────┐   ┌──────────────┐   ┌──────────────────┐    │ │
//! │  │   │ Global  │   │ Work-Stealing│   │ Local Queues     │    │ │
//! │  │   │ Queue   │   │ Deques       │   │ (per worker)     │    │ │
//! │  │   └─────────┘   └──────────────┘   └──────────────────┘    │ │
//! │  └─────────────────────────────────────────────────────────────┘ │
//! │                                                                  │
//! │  Workers: [Thread 0] [Thread 1] [Thread 2] ... [Thread N-1]     │
//! │           Each has its own operator chain instance               │
//! │                                                                  │
//! │  Morsels: 64K row units distributed across workers               │
//! └─────────────────────────────────────────────────────────────────┘
//! ```
//!
//! # Key Concepts
//!
//! - **Morsel**: A unit of work (typically 64K rows) that a worker processes
//! - **Work-Stealing**: Workers steal morsels from others when their queue is empty
//! - **Per-Worker Pipelines**: Each worker has its own operator chain instances
//! - **Pipeline Breakers**: Operators like Sort/Aggregate that need merge phase
//!
//! # Example
//!
//! ```ignore
//! use grafeo_core::execution::parallel::{
//!     ParallelPipeline, ParallelPipelineConfig, CloneableOperatorFactory, RangeSource
//! };
//! use std::sync::Arc;
//!
//! // Create a parallel source
//! let source = Arc::new(RangeSource::new(1_000_000));
//!
//! // Create operator factory (each worker gets its own operators)
//! let factory = Arc::new(
//!     CloneableOperatorFactory::new()
//!         .with_operator(|| Box::new(FilterOp::new()))
//! );
//!
//! // Configure and execute
//! let config = ParallelPipelineConfig::default().with_workers(4);
//! let pipeline = ParallelPipeline::new(source, factory, config);
//! let result = pipeline.execute().unwrap();
//!
//! println!("Processed {} rows", result.rows_processed);
//! ```

pub mod fold;
mod merge;
mod morsel;
mod pipeline;
mod scheduler;
mod source;

// Re-export main types
pub use fold::{
    Mergeable, fold_reduce, fold_reduce_with, parallel_count, parallel_max, parallel_min,
    parallel_partition, parallel_stats, parallel_sum, parallel_sum_i64, parallel_try_collect,
};
pub use merge::{
    MergeableAccumulator, MergeableOperator, SortKey, concat_parallel_results,
    merge_distinct_results, merge_sorted_chunks, merge_sorted_runs, rows_to_chunks,
};
pub use morsel::{
    CRITICAL_PRESSURE_MORSEL_SIZE, DEFAULT_MORSEL_SIZE, HIGH_PRESSURE_MORSEL_SIZE, MIN_MORSEL_SIZE,
    MODERATE_PRESSURE_MORSEL_SIZE, Morsel, compute_morsel_size, compute_morsel_size_with_base,
    generate_adaptive_morsels, generate_morsels,
};
pub use pipeline::{
    CloneableOperatorFactory, CollectorSink, OperatorChainFactory, ParallelPipeline,
    ParallelPipelineConfig, ParallelPipelineResult,
};
pub use scheduler::{MorselScheduler, NumaConfig, NumaNode, WorkerHandle};
#[cfg(feature = "rdf")]
pub use source::ParallelTripleScanSource;
pub use source::{ParallelChunkSource, ParallelSource, ParallelVectorSource, RangeSource};
