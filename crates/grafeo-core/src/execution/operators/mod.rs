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

pub mod accumulator;
mod aggregate;
mod apply;
mod distinct;
mod eager;
mod expand;
mod factorized_aggregate;
mod factorized_expand;
mod factorized_filter;
mod filter;
mod functions;
mod horizontal_aggregate;
mod join;
mod leapfrog_join;
mod limit;
mod load_data;
mod map_collect;
mod merge;
mod mutation;
mod node_seek;
mod parameter_scan;
mod project;
pub mod push;
mod range_scan;
mod scan;
#[cfg(feature = "text-index")]
mod scan_text;
#[cfg(feature = "vector-index")]
mod scan_vector;
mod set_ops;
mod shortest_path;
mod shuffle;
pub mod single_row;
mod sort;
pub mod top_k;
mod union;
mod unwind;
pub mod value_utils;
mod variable_length_expand;
mod vector_join;
mod writer;

pub use accumulator::{AggregateExpr, AggregateFunction, HashableValue};
pub use aggregate::{HashAggregateOperator, SimpleAggregateOperator};
pub use apply::ApplyOperator;
pub use distinct::DistinctOperator;
pub use eager::EagerOperator;
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
    BinaryFilterOp, ExpressionPredicate, FilterExpression, FilterOperator, LazyValue,
    ListPredicateKind, Predicate, SessionContext, UnaryFilterOp,
};
pub use functions::{
    Arity, FunctionSupport, REGEX_SUPPORT, function_names, function_support, regex_pattern_error,
};
pub use horizontal_aggregate::{EntityKind, HorizontalAggregateOperator};
pub use join::{
    EqualityCondition, HashJoinOperator, HashKey, JoinCondition, JoinType, JoinedRowCondition,
    NestedLoopJoinOperator,
};
pub use leapfrog_join::LeapfrogJoinOperator;
pub use limit::{LimitOperator, LimitSkipOperator, SkipOperator};
pub use load_data::{LoadDataFormat, LoadDataOperator};
pub use map_collect::MapCollectOperator;
pub use merge::{MergeConfig, MergeOperator, MergeRelationshipConfig, MergeRelationshipOperator};
pub use mutation::{
    AddLabelOperator, ConstraintValidator, CreateEdgeOperator, CreateNodeOperator, CreateOperator,
    CreateStep, DeleteEdgeOperator, DeleteNodeOperator, PropertySource, RemoveLabelOperator,
    SetPropertyOperator,
};
pub use node_seek::{NodeSeekOperator, SeekKey};
pub use parameter_scan::{ParameterScanOperator, ParameterState};
pub use project::{EntityValue, ProjectExpr, ProjectOperator};
pub use push::{
    AggregatePushOperator, DistinctMaterializingOperator, DistinctPushOperator, FilterPushOperator,
    LimitPushOperator, ProjectPushOperator, SkipLimitPushOperator, SkipPushOperator,
    SortPushOperator,
};
#[cfg(feature = "spill")]
pub use push::{SpillableAggregatePushOperator, SpillableSortPushOperator};
pub use range_scan::RangeScanOperator;
pub use scan::ScanOperator;
#[cfg(feature = "text-index")]
pub use scan_text::TextScanOperator;
#[cfg(feature = "vector-index")]
pub use scan_vector::VectorScanOperator;
pub use set_ops::{ExceptOperator, IntersectOperator, OtherwiseOperator};
pub use shortest_path::{PathSelection as ExecutionPathSelection, ShortestPathOperator};
pub use shuffle::ShuffleOperator;
pub use single_row::{EmptyOperator, NodeListOperator, SingleRowOperator};
pub use sort::{NullOrder, SortDirection, SortKey, SortOperator};
pub use top_k::TopKOperator;
pub use union::UnionOperator;
pub use unwind::UnwindOperator;
pub use variable_length_expand::{
    DEFAULT_PATH_SEARCH_BUDGET, PathMode as ExecutionPathMode, VariableLengthExpandOperator,
};
pub use vector_join::VectorJoinOperator;
pub use writer::{GraphWriter, WriteCounter, WriteCounters};

use std::sync::Arc;

use grafeo_common::types::{EdgeId, NodeId, TransactionId};
use thiserror::Error;

use super::DataChunk;
use super::chunk_state::ChunkState;
use super::factorized_chunk::FactorizedChunk;

/// A store change of an open transaction in progress (see
/// [`WriteTracker::write_in_progress`]): a shared hold on the database's
/// write freeze, which a checkpoint or a copy of the store holds exclusively
/// while it reads the store. Released when dropped.
///
/// Not reentrant: a thread that holds one must not ask for another, since a
/// checkpoint waiting for the first would block the second request, and the
/// thread would wait for itself.
pub type WriteInProgress<'a> = parking_lot::RwLockReadGuard<'a, ()>;

/// Trait for recording write operations during query execution.
///
/// This bridges `grafeo-core` mutation operators (which perform writes) with
/// `grafeo-engine`'s `TransactionManager` (which tracks write sets for conflict
/// detection, and freezes the store for checkpoints). The trait lives in
/// `grafeo-core` to avoid circular dependencies.
pub trait WriteTracker: Send + Sync {
    /// Marks a store change of the writing transaction as in progress, for
    /// as long as the returned guard lives: no checkpoint or copy of the
    /// store starts reading the store meanwhile, and one that is reading it
    /// is waited for, so none sees part of a change or of the change log
    /// entry that undoes it.
    ///
    /// [`GraphWriter`] takes it once per write method, around its store
    /// changes, and never twice on one thread (see [`WriteInProgress`]).
    fn write_in_progress(&self) -> WriteInProgress<'_>;

    /// Records that a node was written (created or modified; a delete is
    /// recorded with [`record_node_delete`](Self::record_node_delete)).
    ///
    /// # Errors
    ///
    /// Returns `Err` if a write-write conflict is detected (first-writer-wins).
    fn record_node_write(
        &self,
        transaction_id: TransactionId,
        node_id: NodeId,
    ) -> Result<(), OperatorError>;

    /// Records that a node is deleted: a write, as
    /// [`record_node_write`](Self::record_node_write) records one, that also
    /// conflicts with another transaction's claim on the node as an endpoint
    /// of an edge it creates (see
    /// [`record_edge_endpoints`](Self::record_edge_endpoints)).
    ///
    /// # Errors
    ///
    /// Returns `Err` if another open transaction wrote the node or claimed it
    /// (first-writer-wins).
    fn record_node_delete(
        &self,
        transaction_id: TransactionId,
        node_id: NodeId,
    ) -> Result<(), OperatorError>;

    /// Claims the endpoints of an edge the transaction is about to create:
    /// another transaction that deletes one of them conflicts with this one,
    /// the later of the two failing (first-writer-wins), and so does one that
    /// deleted one and committed after this transaction began, at this
    /// transaction's commit. Unlike a write, a claim does not conflict with
    /// another transaction's writes of the node or claims on it.
    ///
    /// # Errors
    ///
    /// Returns `Err` if another open transaction deletes either endpoint.
    fn record_edge_endpoints(
        &self,
        transaction_id: TransactionId,
        src: NodeId,
        dst: NodeId,
    ) -> Result<(), OperatorError>;

    /// Records that an edge was written (created, deleted, or modified).
    ///
    /// # Errors
    ///
    /// Returns `Err` if a write-write conflict is detected (first-writer-wins).
    fn record_edge_write(
        &self,
        transaction_id: TransactionId,
        edge_id: EdgeId,
    ) -> Result<(), OperatorError>;
}

/// Type alias for a shared write tracker.
pub type SharedWriteTracker = Arc<dyn WriteTracker>;

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
/// ```rust
/// use grafeo_core::execution::operators::FactorizedData;
///
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
#[non_exhaustive]
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
    /// Schema constraint violation during a write operation.
    #[error("constraint violation: {0}")]
    ConstraintViolation(String),
    /// A value the statement gave cannot be used (a user mistake, such as a
    /// query vector of another size than the index's), reported as an
    /// invalid value rather than an execution failure.
    #[error("invalid value: {0}")]
    InvalidValue(String),
    /// Write-write conflict detected (first-writer-wins).
    #[error("write conflict: {0}")]
    WriteConflict(String),
    /// The query would need more of a resource than it may use, such as
    /// the memory of a path search; the message says what to change.
    #[error("{0}")]
    LimitExceeded(String),
}

/// The core trait for pull-based operators.
///
/// Call [`next()`](Self::next) repeatedly until it returns `None`. Each call
/// returns a batch of rows (a DataChunk) or an error.
pub trait Operator: Send + Sync {
    /// Pulls the next batch of data. Returns `None` when exhausted.
    ///
    /// # Errors
    ///
    /// Returns `Err` if the operator encounters a runtime error.
    fn next(&mut self) -> OperatorResult;

    /// Resets to initial state so you can iterate again.
    fn reset(&mut self);

    /// Returns a name for debugging/explain output.
    fn name(&self) -> &'static str;

    /// Converts this boxed operator into `Box<dyn Any>` for type-based dispatch.
    ///
    /// Used by the pipeline converter to decompose pull-based operator trees
    /// into push-based pipelines via downcasting to concrete types.
    fn into_any(self: Box<Self>) -> Box<dyn std::any::Any + Send>;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::execution::vector::ValueVector;
    use grafeo_common::types::LogicalType;

    fn create_test_chunk() -> DataChunk {
        let mut col = ValueVector::with_type(LogicalType::Int64);
        col.push_int64(1);
        col.push_int64(2);
        col.push_int64(3);
        DataChunk::new(vec![col])
    }

    #[test]
    fn test_flat_data_wrapper_new() {
        let chunk = create_test_chunk();
        let wrapper = FlatDataWrapper::new(chunk);

        assert!(!wrapper.is_factorized());
        assert_eq!(wrapper.logical_row_count(), 3);
    }

    #[test]
    fn test_flat_data_wrapper_into_inner() {
        let chunk = create_test_chunk();
        let wrapper = FlatDataWrapper::new(chunk);

        let inner = wrapper.into_inner();
        assert_eq!(inner.row_count(), 3);
    }

    #[test]
    fn test_flat_data_wrapper_chunk_state() {
        let chunk = create_test_chunk();
        let wrapper = FlatDataWrapper::new(chunk);

        let state = wrapper.chunk_state();
        assert!(state.is_flat());
        assert_eq!(state.logical_row_count(), 3);
    }

    #[test]
    fn test_flat_data_wrapper_physical_size() {
        let mut col1 = ValueVector::with_type(LogicalType::Int64);
        col1.push_int64(1);
        col1.push_int64(2);

        let mut col2 = ValueVector::with_type(LogicalType::String);
        col2.push_string("a");
        col2.push_string("b");

        let chunk = DataChunk::new(vec![col1, col2]);
        let wrapper = FlatDataWrapper::new(chunk);

        // 2 rows * 2 columns = 4
        assert_eq!(wrapper.physical_size(), 4);
    }

    #[test]
    fn test_flat_data_wrapper_flatten() {
        let chunk = create_test_chunk();
        let wrapper = FlatDataWrapper::new(chunk);

        let flattened = wrapper.flatten();
        assert_eq!(flattened.row_count(), 3);
        assert_eq!(flattened.column(0).unwrap().get_int64(0), Some(1));
    }

    #[test]
    fn test_flat_data_wrapper_as_factorized() {
        let chunk = create_test_chunk();
        let wrapper = FlatDataWrapper::new(chunk);

        assert!(wrapper.as_factorized().is_none());
    }

    #[test]
    fn test_flat_data_wrapper_as_flat() {
        let chunk = create_test_chunk();
        let wrapper = FlatDataWrapper::new(chunk);

        let flat = wrapper.as_flat();
        assert!(flat.is_some());
        assert_eq!(flat.unwrap().row_count(), 3);
    }

    #[test]
    fn test_operator_error_type_mismatch() {
        let err = OperatorError::TypeMismatch {
            expected: "Int64".to_string(),
            found: "String".to_string(),
        };

        let msg = format!("{err}");
        assert!(msg.contains("type mismatch"));
        assert!(msg.contains("Int64"));
        assert!(msg.contains("String"));
    }

    #[test]
    fn test_operator_error_column_not_found() {
        let err = OperatorError::ColumnNotFound("missing_col".to_string());

        let msg = format!("{err}");
        assert!(msg.contains("column not found"));
        assert!(msg.contains("missing_col"));
    }

    #[test]
    fn test_operator_error_execution() {
        let err = OperatorError::Execution("something went wrong".to_string());

        let msg = format!("{err}");
        assert!(msg.contains("execution error"));
        assert!(msg.contains("something went wrong"));
    }

    #[test]
    fn test_operator_error_debug() {
        let err = OperatorError::TypeMismatch {
            expected: "Int64".to_string(),
            found: "String".to_string(),
        };

        let debug = format!("{err:?}");
        assert!(debug.contains("TypeMismatch"));
    }

    #[test]
    fn test_operator_error_clone() {
        let err1 = OperatorError::ColumnNotFound("col".to_string());
        let err2 = err1.clone();

        assert_eq!(format!("{err1}"), format!("{err2}"));
    }
}
