//! Graph model implementations.
//!
//! Graphos supports two graph models:
//!
//! - [`lpg`] - Labeled Property Graph (default)
//! - [`rdf`] - RDF Triple Store (optional, feature-gated)
//!
//! These are completely separate implementations with no abstraction overhead.

pub mod lpg;

#[cfg(feature = "rdf")]
pub mod rdf;

/// Direction of edge traversal.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Direction {
    /// Follow outgoing edges (from source to destination).
    Outgoing,
    /// Follow incoming edges (from destination to source).
    Incoming,
    /// Follow edges in either direction.
    Both,
}

impl Direction {
    /// Returns the opposite direction.
    #[must_use]
    pub const fn reverse(self) -> Self {
        match self {
            Direction::Outgoing => Direction::Incoming,
            Direction::Incoming => Direction::Outgoing,
            Direction::Both => Direction::Both,
        }
    }
}
