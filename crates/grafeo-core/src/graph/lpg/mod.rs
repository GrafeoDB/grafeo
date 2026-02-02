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

mod edge;
mod node;
mod property;
mod store;

pub use edge::{Edge, EdgeFlags, EdgeRecord};
pub use node::{Node, NodeFlags, NodeRecord};
pub use property::{CompareOp, PropertyStorage};
pub use store::LpgStore;
