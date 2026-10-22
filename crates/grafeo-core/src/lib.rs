//! # grafeo-core
//!
//! The core data structures behind Grafeo. You'll find graph storage, indexes,
//! and the execution engine here.
//!
//! Most users don't need this crate directly - use `grafeo` or `grafeo-engine`
//! instead. But if you're building algorithms or need low-level access, this
//! is where the action is.
//!
//! ## Modules
//!
//! - [`cache`] - Caching utilities: second-chance LRU
//! - [`graph`] - Graph storage: LPG (labeled property graph) and RDF triple stores
//! - [`index`] - Fast lookups: hash, B-tree, adjacency lists, tries
//! - [`execution`] - Query execution: data chunks, vectors, operators
//! - [`statistics`] - Cardinality estimates for the query optimizer
//! - [`codec`] - Compression: dictionary encoding, bit-packing, delta encoding

#![deny(unsafe_code)]

pub mod cache;
pub mod codec;
pub mod execution;
pub mod graph;
pub mod index;
pub mod statistics;
pub mod testing;

// Re-export the types you'll use most often
pub use codec::{DictionaryBuilder, DictionaryEncoding};
#[cfg(feature = "lpg")]
pub use graph::lpg::LpgStore;
pub use graph::lpg::{Edge, Node};
pub use index::adjacency::ChunkedAdjacency;
pub use statistics::{ColumnStatistics, Histogram, LabelStatistics, Statistics};
