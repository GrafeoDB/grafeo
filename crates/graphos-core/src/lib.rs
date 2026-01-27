//! # graphos-core
//!
//! Core layer for Graphos: graph models, index structures, and execution primitives.
//!
//! This crate provides the fundamental data structures for storing and querying
//! graph data. It depends only on `graphos-common`.
//!
//! ## Modules
//!
//! - [`graph`] - Graph model implementations (LPG, RDF)
//! - [`index`] - Index structures (Hash, BTree, Chunked Adjacency, Trie)
//! - [`execution`] - Execution primitives (DataChunk, ValueVector, Operators)
//! - [`statistics`] - Statistics collection for query optimization
//! - [`storage`] - Storage utilities (Dictionary encoding, compression)

#![warn(missing_docs)]
#![warn(clippy::all)]
#![warn(clippy::pedantic)]

pub mod execution;
pub mod graph;
pub mod index;
pub mod statistics;
pub mod storage;

// Re-export commonly used types
pub use graph::lpg::{Edge, LpgStore, Node};
pub use index::adjacency::ChunkedAdjacency;
pub use statistics::{ColumnStatistics, Histogram, LabelStatistics, Statistics};
pub use storage::{DictionaryBuilder, DictionaryEncoding};
