//! Physical operators that actually execute queries.
//!
//! These are the building blocks of query execution. The optimizer picks which
//! operators to use and how to wire them together.
//!
//! **Graph operators:**
//! - [`ScanOperator`] - Read nodes/edges from storage
//! - [`ExpandOperator`] - Traverse edges (the core of graph queries)
//! - [`VariableLengthExpandOperator`] - Paths of variable length
//! - [`ShortestPathOperator`] - Find shortest paths
//!
//! **Relational operators:**
//! - [`FilterOperator`] - Apply predicates
//! - [`ProjectOperator`] - Select/transform columns
//! - [`HashJoinOperator`] - Efficient equi-joins
//! - [`HashAggregateOperator`] - Group by with aggregation
//! - [`SortOperator`] - Order results
//! - [`LimitOperator`] - SKIP and LIMIT
//!
//! The [`push`] submodule has push-based variants for pipeline execution.

mod aggregate;
mod distinct;
mod expand;
mod factorized_aggregate;
mod factorized_expand;
mod factorized_filter;
mod filter;
mod join;
mod limit;
mod merge;
mod mutation;
mod project;
pub mod push;
mod scan;
mod shortest_path;
pub mod single_row;
mod sort;
mod union;
mod unwind;
mod variable_length_expand;

pub use aggregate::{
    AggregateExpr, AggregateFunction, HashAggregateOperator, SimpleAggregateOperator,
};
pub use distinct::DistinctOperator;
pub use expand::ExpandOperator;
pub use factorized_aggregate::{
    FactorizedAggregate, FactorizedAggregateOperator, FactorizedOperator,
};
pub use factorized_expand::{
    ExpandStep, FactorizedExpandChain, FactorizedExpandOperator, FactorizedResult,
    LazyFactorizedChainOperator,
};
pub use factorized_filter::{
    AndPredicate, ColumnPredicate, CompareOp as FactorizedCompareOp, FactorizedFilterOperator,
    FactorizedPredicate, OrPredicate, PropertyPredicate,
};
pub use filter::{
    BinaryFilterOp, ExpressionPredicate, FilterExpression, FilterOperator, Predicate, UnaryFilterOp,
};
pub use join::{
    EqualityCondition, HashJoinOperator, HashKey, JoinCondition, JoinType, NestedLoopJoinOperator,
};
pub use limit::{LimitOperator, LimitSkipOperator, SkipOperator};
pub use merge::MergeOperator;
pub use mutation::{
    AddLabelOperator, CreateEdgeOperator, CreateNodeOperator, DeleteEdgeOperator,
    DeleteNodeOperator, PropertySource, RemoveLabelOperator, SetPropertyOperator,
};
pub use project::{ProjectExpr, ProjectOperator};
pub use push::{
    AggregatePushOperator, DistinctMaterializingOperator, DistinctPushOperator, FilterPushOperator,
    LimitPushOperator, ProjectPushOperator, SkipLimitPushOperator, SkipPushOperator,
    SortPushOperator, SpillableAggregatePushOperator, SpillableSortPushOperator,
};
pub use scan::ScanOperator;
pub use shortest_path::ShortestPathOperator;
pub use sort::{NullOrder, SortDirection, SortKey, SortOperator};
pub use union::UnionOperator;
pub use unwind::UnwindOperator;
pub use variable_length_expand::VariableLengthExpandOperator;

use thiserror::Error;

use super::DataChunk;
use super::chunk_state::ChunkState;
use super::factorized_chunk::FactorizedChunk;

/// Result of executing an operator.
pub type OperatorResult = Result<Option<DataChunk>, OperatorError>;

// ============================================================================
// Factorized Data Traits
// ============================================================================

/// Trait for data that can be in factorized or flat form.
///
/// This provides a common interface for operators that need to handle both
/// representations without caring which is used. Inspired by LadybugDB's
/// unified data model.
///
/// # Example
///
/// ```ignore
/// fn process_data(data: &dyn FactorizedData) {
///     if data.is_factorized() {
///         // Handle factorized path
///         let chunk = data.as_factorized().unwrap();
///         // ... use factorized chunk directly
///     } else {
///         // Handle flat path
///         let chunk = data.flatten();
///         // ... process flat chunk
///     }
/// }
/// ```
pub trait FactorizedData: Send + Sync {
    /// Returns the chunk state (factorization status, cached data).
    fn chunk_state(&self) -> &ChunkState;

    /// Returns the logical row count (considering selection).
    fn logical_row_count(&self) -> usize;

    /// Returns the physical size (actual stored values).
    fn physical_size(&self) -> usize;

    /// Returns true if this data is factorized (multi-level).
    fn is_factorized(&self) -> bool;

    /// Flattens to a DataChunk (materializes if factorized).
    fn flatten(&self) -> DataChunk;

    /// Returns as FactorizedChunk if factorized, None if flat.
    fn as_factorized(&self) -> Option<&FactorizedChunk>;

    /// Returns as DataChunk if flat, None if factorized.
    fn as_flat(&self) -> Option<&DataChunk>;
}

/// Wrapper to treat a flat DataChunk as FactorizedData.
///
/// This enables uniform handling of flat and factorized data in operators.
pub struct FlatDataWrapper {
    chunk: DataChunk,
    state: ChunkState,
}

impl FlatDataWrapper {
    /// Creates a new wrapper around a flat DataChunk.
    #[must_use]
    pub fn new(chunk: DataChunk) -> Self {
        let state = ChunkState::flat(chunk.row_count());
        Self { chunk, state }
    }

    /// Returns the underlying DataChunk.
    #[must_use]
    pub fn into_inner(self) -> DataChunk {
        self.chunk
    }
}

impl FactorizedData for FlatDataWrapper {
    fn chunk_state(&self) -> &ChunkState {
        &self.state
    }

    fn logical_row_count(&self) -> usize {
        self.chunk.row_count()
    }

    fn physical_size(&self) -> usize {
        self.chunk.row_count() * self.chunk.column_count()
    }

    fn is_factorized(&self) -> bool {
        false
    }

    fn flatten(&self) -> DataChunk {
        self.chunk.clone()
    }

    fn as_factorized(&self) -> Option<&FactorizedChunk> {
        None
    }

    fn as_flat(&self) -> Option<&DataChunk> {
        Some(&self.chunk)
    }
}

/// Error during operator execution.
#[derive(Error, Debug, Clone)]
pub enum OperatorError {
    /// Type mismatch during execution.
    #[error("type mismatch: expected {expected}, found {found}")]
    TypeMismatch {
        /// Expected type name.
        expected: String,
        /// Found type name.
        found: String,
    },
    /// Column not found.
    #[error("column not found: {0}")]
    ColumnNotFound(String),
    /// Execution error.
    #[error("execution error: {0}")]
    Execution(String),
}

/// The core trait for pull-based operators.
///
/// Call [`next()`](Self::next) repeatedly until it returns `None`. Each call
/// returns a batch of rows (a DataChunk) or an error.
pub trait Operator: Send + Sync {
    /// Pulls the next batch of data. Returns `None` when exhausted.
    fn next(&mut self) -> OperatorResult;

    /// Resets to initial state so you can iterate again.
    fn reset(&mut self);

    /// Returns a name for debugging/explain output.
    fn name(&self) -> &'static str;
}
