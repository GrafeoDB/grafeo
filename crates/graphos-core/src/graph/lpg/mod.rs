//! Labeled Property Graph (LPG) model.
//!
//! The LPG model is the default graph model in Graphos. It supports:
//! - Nodes with labels and properties
//! - Directed edges with types and properties
//! - Multiple labels per node
//! - Efficient property storage

mod edge;
mod node;
mod property;
mod store;

pub use edge::{Edge, EdgeRecord};
pub use node::{Node, NodeRecord};
pub use property::PropertyStorage;
pub use store::LpgStore;
