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
mod decimal;
mod duration;
pub mod hlc;
mod id;
mod logical_type;
mod property_map;
mod property_type;
pub mod rdf;
mod rdf_order;
mod read_view;
mod temporal_comparison;
mod temporal_component;
mod term_id;
mod time;
mod timestamp;
mod validity;
mod value;
mod value_type;
mod zoned_datetime;

pub use date::Date;
pub use decimal::{Decimal, DecimalError, DecimalType, MAX_DECIMAL_DIGITS};
pub use duration::Duration;
pub use hlc::{HlcClock, HlcTimestamp};
pub use id::{EdgeId, EdgeTypeId, EpochId, IndexId, LabelId, NodeId, PropertyKeyId, TransactionId};
pub use logical_type::LogicalType;
pub use property_map::PropertyMap;
pub use property_type::{PropertyDataType, PropertyTypeTooDeep, exact_float};
pub use rdf_order::{SparqlOrderKey, sparql_order};
pub use read_view::ReadView;
pub use temporal_comparison::TemporalComparison;
pub use term_id::{TermId, TermIdKind};
pub use time::Time;
pub use timestamp::Timestamp;
pub use validity::ValidityTs;
pub use value::{
    HashableValue, NumberTextError, OrderableValue, OrderedFloat64, PropertyKey, Value,
};
pub use value_type::ValueType;
pub use zoned_datetime::ZonedDatetime;

// Re-export ArcStr so downstream crates don't need a direct arcstr dependency.
pub use arcstr::ArcStr;
