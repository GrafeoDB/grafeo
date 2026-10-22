//! Graph model implementations.
//!
//! Pick your graph model:
//!
//! | Model | When to use | Example use case |
//! | ----- | ----------- | ---------------- |
//! | [`lpg`] | Most apps (default) | Social networks, fraud detection |
// The feature-gated modules are linked only where they are built: rustdoc
// fails on a link to a module that is not there.
#![cfg_attr(
    feature = "triple-store",
    doc = "| [`rdf`] | Knowledge graphs | Ontologies, linked data (feature-gated: `triple-store`) |"
)]
#![cfg_attr(
    not(feature = "triple-store"),
    doc = "| `rdf` | Knowledge graphs | Ontologies, linked data (feature-gated: `triple-store`) |"
)]
//!
//! These are separate implementations with no abstraction overhead - you get
//! the full performance of whichever model you choose.
//!
//! `compact` (with `lpg`) is not a model: it reads the compacted base of a
//! 0.5.x database file, so that opening the file folds it into the LPG store.

pub mod apply;
/// The conformance suite of the stores behind the change set.
#[cfg(all(test, feature = "lpg"))]
mod conformance;
pub mod lpg;
pub mod projection;
/// The row-group store (workstream H): not used by the engine until H2d.
#[cfg(feature = "lpg")]
#[doc(hidden)]
pub mod rowgroup;
pub mod traits;

#[cfg(feature = "lpg")]
pub mod compact;

#[cfg(feature = "triple-store")]
pub mod rdf;

pub use projection::{GraphProjection, ProjectionSpec};
pub use traits::{GraphStore, GraphStoreMut, GraphStoreSearch, NullGraphStore};

/// Controls which edges to follow during traversal.
///
/// Most graph operations need to specify direction. Use [`Outgoing`](Self::Outgoing)
/// when you care about relationships *from* a node, [`Incoming`](Self::Incoming) for
/// relationships *to* a node, and [`Both`](Self::Both) when direction doesn't matter.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum Direction {
    /// Follow outgoing edges (A)-\[r\]->(B) from A's perspective.
    Outgoing,
    /// Follow incoming edges (A)<-\[r\]-(B) from A's perspective.
    Incoming,
    /// Follow edges in either direction - treat the graph as undirected.
    Both,
}

impl Direction {
    /// Flips the direction - outgoing becomes incoming and vice versa.
    ///
    /// Useful when traversing backward along a path.
    #[must_use]
    pub const fn reverse(self) -> Self {
        match self {
            Direction::Outgoing => Direction::Incoming,
            Direction::Incoming => Direction::Outgoing,
            Direction::Both => Direction::Both,
        }
    }
}
