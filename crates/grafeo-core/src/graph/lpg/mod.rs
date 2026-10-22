//! Labeled Property Graph (LPG) storage.
//!
//! This is Grafeo's primary graph model - the same model used by Neo4j,
//! TigerGraph, and most modern graph databases. If you're used to working
//! with nodes, relationships, and properties, you're in the right place.
//!
//! ## What you get
//!
//! - **Nodes** with labels (like "Person", "Company") and properties (like "name", "age")
//! - **Edges** that connect nodes, with types (like "KNOWS", "WORKS_AT") and their own properties
//! - **Indexes** that make lookups fast
//!
//! Start with [`LpgStore`] - that's where everything lives.

#[cfg(feature = "lpg")]
pub(crate) mod block;
#[cfg(feature = "lpg")]
mod chunked;
#[cfg(feature = "lpg")]
mod committed;
#[cfg(feature = "lpg")]
pub(crate) mod dictionary;
mod edge;
mod node;
pub mod overlay;
mod property;
#[cfg(feature = "lpg")]
pub mod section;
#[cfg(feature = "lpg")]
mod store;

// Types are always available (used by GraphStore trait and RDF adapter)
pub use edge::{Edge, EdgeFlags, EdgeRecord};
pub use node::{Node, NodeFlags, NodeRecord};
#[cfg(not(feature = "temporal"))]
pub use property::ColumnBacking;
#[cfg(not(feature = "temporal"))]
pub use property::DecodedBlock;
#[cfg(all(test, feature = "lpg", not(feature = "temporal")))]
pub(crate) use property::test_backing;
pub use property::{CompareOp, PropertyStorage};

// Store and section require the lpg feature
#[cfg(feature = "lpg")]
pub use section::LpgStoreSection;
#[cfg(feature = "lpg")]
pub use store::{LpgStore, OpenChangesByGraph};
