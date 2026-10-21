//! Labeled Property Graph (LPG) types, and the readers of the LPG sections of
//! earlier releases.
//!
//! This is Grafeo's primary graph model - the same model used by Neo4j,
//! TigerGraph, and most modern graph databases. If you're used to working
//! with nodes, relationships, and properties, you're in the right place.
//!
//! ## What you get
//!
//! - **Nodes** with labels (like "Person", "Company") and properties (like "name", "age")
//! - **Edges** that connect nodes, with types (like "KNOWS", "WORKS_AT") and their own properties
//!
//! A read returns them as [`Node`] and [`Edge`]. The graphs themselves live
//! in the row-group store (`graph::rowgroup`, with the `lpg` feature).
//!
//! The readers of the LPG sections of 0.5.x (the block layout) and of
//! 0.6.0-beta.1 (version 3, in chunks) load a file of those releases into
//! the row-group store through `graph::load::LoadTarget`: no build writes
//! those sections any more.

#[cfg(feature = "lpg")]
pub(crate) mod block;
#[cfg(feature = "lpg")]
mod chunked;
mod compare;
#[cfg(feature = "lpg")]
pub(crate) mod dictionary;
mod edge;
mod node;
#[cfg(feature = "lpg")]
mod section;

pub use compare::CompareOp;
pub use edge::Edge;
pub use node::Node;

#[cfg(feature = "lpg")]
pub(crate) use chunked::{
    LPG_SECTION_VERSION as LPG_SECTION_VERSION_3, read_lpg_chunks as read_version_3_chunks,
};
#[cfg(feature = "lpg")]
pub(crate) use section::load_blocks as load_0_5_blocks;
#[cfg(feature = "lpg")]
pub use section::load_versions;
