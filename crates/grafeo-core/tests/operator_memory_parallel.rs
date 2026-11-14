//! A parallel pipeline's workers build and run their operators in the
//! statement's memory (#674): every worker takes its grants under the one
//! statement cap, so a materializing parallel pipeline past the query memory
//! limit is refused, and passes under a large one.
//!
//! ```bash
//! cargo test -p grafeo-core --all-features --test main operator_memory_parallel::
//! ```

#![cfg(all(feature = "spill", feature = "parallel"))]

use std::sync::Arc;

use grafeo_common::memory::buffer::BufferManager;
use grafeo_common::utils::error::LimitBound;
use grafeo_core::execution::memory::{self, OperatorMemoryContext, StatementMemory};
use grafeo_core::execution::operators::{DistinctPushOperator, OperatorError};
use grafeo_core::execution::{
    CloneableOperatorFactory, ParallelPipeline, ParallelPipelineConfig, RangeSource,
};

const KIB: usize = 1 << 10;
const MIB: usize = 1 << 20;

/// A distinct over 200,000 numbers, run by four workers in the statement of
/// `ctx`.
fn distinct_in(ctx: &OperatorMemoryContext) -> Result<usize, OperatorError> {
    let factory = CloneableOperatorFactory::new()
        .with_operator(|| Box::new(DistinctPushOperator::new()))
        .with_pipeline_breakers();
    let pipeline = ParallelPipeline::new(
        Arc::new(RangeSource::new(200_000)),
        Arc::new(factory),
        ParallelPipelineConfig::for_testing().with_workers(4),
    );
    let _scope = memory::enter(StatementMemory::Budgeted(ctx.clone()));
    pipeline.execute().map(|result| result.rows_processed)
}

fn statement(limit: usize) -> OperatorMemoryContext {
    OperatorMemoryContext::new(BufferManager::with_budget(256 * MIB), None).with_query_limit(limit)
}

#[test]
fn a_parallel_distinct_past_the_query_limit_is_refused() {
    let small = statement(64 * KIB);
    let error = distinct_in(&small).expect_err("refused at the query memory limit");
    let OperatorError::Wrapped(inner) = &error else {
        panic!("not a refusal: {error}");
    };
    assert_eq!(inner.error_code().as_str(), "GRAFEO-S001", "{error}");
    assert_eq!(
        inner.limit_refusal().map(|refusal| refusal.bound()),
        Some(LimitBound::QueryMemoryLimit),
        "{error}"
    );
    assert_eq!(
        small.query().used(),
        0,
        "the workers gave back what they held"
    );

    let large = statement(256 * MIB);
    assert_eq!(distinct_in(&large).unwrap(), 200_000);
    assert_eq!(
        large.query().used(),
        0,
        "the workers gave back what they held"
    );
}
