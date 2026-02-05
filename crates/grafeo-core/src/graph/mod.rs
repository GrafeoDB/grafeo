//! Graph model implementations.
//!
//! Pick your graph model:
//!
//! | Model | When to use | Example use case |
//! | ----- | ----------- | ---------------- |
//! | [`lpg`] | Most apps (default) | Social networks, fraud detection |
//! | [`rdf`] | Knowledge graphs | Ontologies, linked data (feature-gated) |
//!
//! These are separate implementations with no abstraction overhead - you get
//! the full performance of whichever model you choose.

pub mod lpg;

#[cfg(feature = "rdf")]
pub mod rdf;

/// Controls which edges to follow during traversal.
///
/// Most graph operations need to specify direction. Use [`Outgoing`](Self::Outgoing)
/// when you care about relationships *from* a node, [`Incoming`](Self::Incoming) for
/// relationships *to* a node, and [`Both`](Self::Both) when direction doesn't matter.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
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
