//! The core types you'll work with in Grafeo.
//!
//! Most of these are re-exported from the main `grafeo` crate, so you rarely
//! need to import from here directly.
//!
//! - **IDs**: [`NodeId`], [`EdgeId`] - handles to graph elements
//! - **Values**: [`Value`] - the dynamic type for properties
//! - **Keys**: [`PropertyKey`] - interned property names
//! - **Time**: [`Timestamp`] - for temporal properties

mod date;
mod duration;
pub mod hlc;
mod id;
mod logical_type;
mod property_map;
mod temporal_component;
mod time;
mod timestamp;
mod validity;
mod value;
mod zoned_datetime;

pub use date::Date;
pub use duration::Duration;
pub use hlc::{HlcClock, HlcTimestamp};
pub use id::{EdgeId, EdgeTypeId, EpochId, IndexId, LabelId, NodeId, PropertyKeyId, TransactionId};
pub use logical_type::LogicalType;
pub use property_map::PropertyMap;
pub use time::Time;
pub use timestamp::Timestamp;
pub use validity::ValidityTs;
pub use value::{HashableValue, OrderableValue, OrderedFloat64, PropertyKey, Value};
pub use zoned_datetime::ZonedDatetime;

// Re-export ArcStr so downstream crates don't need a direct arcstr dependency.
pub use arcstr::ArcStr;
