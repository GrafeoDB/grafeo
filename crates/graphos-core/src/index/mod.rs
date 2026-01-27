//! Index structures for efficient graph queries.
//!
//! This module provides various index structures:
//!
//! - [`adjacency`] - Chunked adjacency lists with delta buffers
//! - [`hash`] - Hash index for primary key lookups
//! - [`btree`] - BTree index for range queries
//! - [`trie`] - Trie index for WCOJ (Worst-Case Optimal Joins)
//! - [`zone_map`] - Zone maps for data skipping during scans

pub mod adjacency;
pub mod btree;
pub mod hash;
pub mod trie;
pub mod zone_map;

pub use adjacency::ChunkedAdjacency;
pub use btree::BTreeIndex;
pub use hash::HashIndex;
pub use zone_map::{BloomFilter, ZoneMapBuilder, ZoneMapEntry, ZoneMapIndex};
