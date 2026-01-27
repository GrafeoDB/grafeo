//! Push-based operator implementations.
//!
//! These operators implement the PushOperator trait for push-based execution,
//! where data flows forward through the pipeline via `push()` calls.
//!
//! ## Non-blocking operators
//! - `FilterPushOperator` - Filters rows based on predicates
//! - `ProjectPushOperator` - Evaluates expressions to produce columns
//! - `LimitPushOperator` - Limits output rows (enables early termination)
//!
//! ## Pipeline breakers
//! - `SortPushOperator` - Buffers all input, produces sorted output
//! - `AggregatePushOperator` - Groups and aggregates, produces in finalize
//! - `DistinctPushOperator` - Filters duplicates (incremental)
//! - `DistinctMaterializingOperator` - Filters duplicates (materializing)

mod aggregate;
mod distinct;
mod filter;
mod limit;
mod project;
mod sort;

pub use aggregate::{AggregateExpr, AggregateFunction, AggregatePushOperator, SpillableAggregatePushOperator, DEFAULT_AGGREGATE_SPILL_THRESHOLD};
pub use distinct::{DistinctMaterializingOperator, DistinctPushOperator};
pub use filter::{
    AndPredicate, ColumnPredicate, CompareOp, FilterPredicate, FilterPushOperator, NotNullPredicate,
    OrPredicate,
};
pub use limit::{LimitPushOperator, SkipLimitPushOperator, SkipPushOperator};
pub use project::{
    ArithOp, BinaryExpr, ColumnExpr, ConstantExpr, ProjectExpression, ProjectPushOperator,
};
pub use sort::{NullOrder, SortDirection, SortKey, SortPushOperator, SpillableSortPushOperator, DEFAULT_SPILL_THRESHOLD};
