//! The core types you'll work with in Grafeo.
//!
//! Most of these are re-exported from the main `grafeo` crate, so you rarely
//! need to import from here directly.
//!
//! - **IDs**: [`NodeId`], [`EdgeId`] - handles to graph elements
//! - **Values**: [`Value`] - the dynamic type for properties
//! - **Keys**: [`PropertyKey`] - interned property names
//! - **Time**: [`Timestamp`] - for temporal properties

mod id;
mod logical_type;
mod timestamp;
mod value;

pub use id::{EdgeId, EdgeTypeId, EpochId, IndexId, LabelId, NodeId, PropertyKeyId, TxId};
pub use logical_type::LogicalType;
pub use timestamp::Timestamp;
pub use value::{HashableValue, OrderableValue, OrderedFloat64, PropertyKey, Value};
