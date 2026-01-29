//! Physical operators for query execution.
//!
//! This module provides the physical operators that form the execution tree:
//!
//! - Scan: Read nodes/edges from storage
//! - Expand: Traverse edges from nodes
//! - Filter: Apply predicates to filter rows
//! - Project: Select and transform columns
//! - Join: Hash join and nested loop join
//! - Aggregate: Group by and aggregation functions
//! - Sort: Order results by columns
//! - Limit: Limit the number of results
//!
//! The `push` submodule contains push-based operator implementations.

mod aggregate;
mod distinct;
mod expand;
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

/// Result of executing an operator.
pub type OperatorResult = Result<Option<DataChunk>, OperatorError>;

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

/// Trait for physical operators.
pub trait Operator: Send + Sync {
    /// Returns the next chunk of data, or None if exhausted.
    fn next(&mut self) -> OperatorResult;

    /// Resets the operator to its initial state.
    fn reset(&mut self);

    /// Returns the name of this operator for debugging.
    fn name(&self) -> &'static str;
}
