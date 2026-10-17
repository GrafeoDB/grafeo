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
pub use writer::{GraphWriter, NewEdge, Recording, WriteCounter, WriteCounters, WriteTarget};

use std::ops::Range;

use grafeo_common::change::{Before, DataOp, PendingVersion, Table};
use grafeo_common::types::{EdgeId, NodeId};
use thiserror::Error;

use crate::graph::apply::Writer;

use super::DataChunk;
use super::chunk_state::ChunkState;
use super::factorized_chunk::FactorizedChunk;

/// A store change of an open transaction in progress (see
/// [`WriteClaims::write_in_progress`]): a shared hold on the database's
/// write freeze, which a checkpoint or a copy of the store holds exclusively
/// while it reads the store. Released when dropped.
///
/// Not reentrant: a thread that holds one must not ask for another, since a
/// checkpoint waiting for the first would block the second request, and the
/// thread would wait for itself.
pub type WriteInProgress<'a> = parking_lot::RwLockReadGuard<'a, ()>;

/// What a transaction claims before it changes the store, for write-conflict
/// detection: first writer wins between open transactions, and at commit
/// against the transactions that committed after it began.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WriteClaim {
    /// A write of a node: its properties or labels.
    Node(NodeId),
    /// The delete of a node: a write, which also conflicts with another
    /// transaction's claim on the node as an endpoint of an edge it creates.
    NodeDelete(NodeId),
    /// A write of an edge: its create, delete or properties.
    Edge(EdgeId),
    /// The endpoints of an edge the transaction creates: they conflict only
    /// with another transaction's delete of either node.
    Endpoints(NodeId, NodeId),
}

/// The claims and the write freeze of a transaction that writes: the
/// bridge between the writer in this crate and the engine's transaction
/// manager (first-writer-wins conflict detection, and the freeze a
/// checkpoint takes).
pub trait WriteClaims: Send + Sync {
    /// Claims what the next store change writes, before the store changes.
    ///
    /// # Errors
    ///
    /// Returns [`OperatorError::WriteConflict`] when another open
    /// transaction holds a claim that conflicts with it (first writer wins).
    fn claim(&self, claim: WriteClaim) -> Result<(), OperatorError>;

    /// Marks a store change of the transaction as in progress, for as long
    /// as the returned guard lives: no checkpoint or copy of the store starts
    /// reading the store meanwhile, and one that is reading it is waited for,
    /// so none sees a change without its entry in the change set. `None` for
    /// a writer that holds checkpoints off itself (an immediate write holds
    /// commits off for its whole run).
    ///
    /// [`GraphWriter`] takes it once per write method, around its store
    /// changes and their records, and never twice on one thread (see
    /// [`WriteInProgress`]).
    fn write_in_progress(&self) -> Option<WriteInProgress<'_>>;
}

/// Where a [`GraphWriter`] records what it changes: one transaction's
/// changes in one graph (an entry of its change set per write), with the
/// claims and the write freeze that go with them.
///
/// The bridge between the writer in this crate and the engine's transaction
/// layer (the change set and the transaction manager's claims), so the
/// writer records every write it applies the same way, whichever path made
/// it: a statement, the direct API, a batch.
pub trait ChangeRecorder: WriteClaims {
    /// The transaction writing, as the store's change target takes it.
    fn writer(&self) -> Writer;

    /// Records a write the store applied: `op` with what it replaced
    /// (`before`) and what it did to the transaction's pending version.
    /// Called while the write's freeze guard is held, right after the store
    /// changed, once per change.
    ///
    /// # Errors
    ///
    /// Returns an error when the change set refuses the entry: a broken
    /// invariant (the store reported a before-image the op does not have),
    /// after which the store holds a change no undo knows of; the recorder
    /// makes sure no commit follows.
    fn record(
        &self,
        op: DataOp,
        before: Before,
        version: PendingVersion,
    ) -> Result<(), OperatorError>;

    /// Whether this recorder takes bulk writes (see
    /// [`GraphWriter::create_nodes`] and [`GraphWriter::create_edges`]), and
    /// what it does with their rows. `None`, the default, for one that takes
    /// none: its writer creates row by row, recording each.
    fn bulk(&self) -> Option<BulkRows> {
        None
    }

    /// Records that a bulk write reserved `ids` in `table` of the recorder's
    /// graph, before it applies the first row: one entry for the range, which
    /// a rollback undoes and the commit stamps as a range, whatever its size.
    ///
    /// # Errors
    ///
    /// Returns an error when the change set refuses the range (a broken
    /// invariant), and always for a recorder that takes no bulk writes (the
    /// default).
    fn record_bulk(&self, table: Table, ids: Range<u64>) -> Result<(), OperatorError> {
        let _ = (table, ids);
        Err(OperatorError::Execution(
            "this change recorder takes no bulk writes".to_string(),
        ))
    }

    /// Keeps `rows`, the creates a bulk write applied in the range it
    /// recorded last, with that range for the commit's log and change data
    /// capture ([`BulkRows::Keep`]).
    ///
    /// # Errors
    ///
    /// Returns an error when the change set refuses the rows (a broken
    /// invariant: the store holds rows the log would miss; the recorder
    /// makes sure no commit follows), and always for a recorder that takes
    /// no bulk writes (the default).
    fn record_bulk_rows(&self, rows: Vec<DataOp>) -> Result<(), OperatorError> {
        let _ = rows;
        Err(OperatorError::Execution(
            "this change recorder takes no bulk writes".to_string(),
        ))
    }
}

/// What a [`ChangeRecorder`] that takes bulk writes does with their rows.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BulkRows {
    /// Drops each row once it is applied: neither the log nor change data
    /// capture reads them (a database in memory without change data
    /// capture).
    Drop,
    /// Keeps the rows for the commit's log and change data capture (see
    /// [`ChangeRecorder::record_bulk_rows`]).
    Keep,
}

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
    /// The query ran past its deadline (a retryable timeout, not a failure).
    #[error("Query exceeded timeout")]
    Timeout,
    /// A broken invariant of the engine: a bug in Grafeo, not a mistake in
    /// the statement or its data. [`Execution`](Self::Execution) is the
    /// statement's failure.
    #[error("{0}")]
    Internal(String),
    /// An error raised below the operator (a store that refuses a write, a
    /// procedure, a statement run inside the operator), kept as it is so its
    /// code reaches the caller.
    #[error(transparent)]
    Wrapped(Box<grafeo_common::utils::error::Error>),
}

impl From<grafeo_common::utils::error::Error> for OperatorError {
    fn from(error: grafeo_common::utils::error::Error) -> Self {
        Self::Wrapped(Box::new(error))
    }
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
