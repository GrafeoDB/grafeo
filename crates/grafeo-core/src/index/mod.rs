//! Index structures that make queries fast.
//!
//! Pick the right index for your access pattern:
//!
//! | Index | Best for | Complexity |
//! | ----- | -------- | ---------- |
//! | [`adjacency`] | Traversing neighbors | O(degree) |
//! | [`hash`] | Point lookups by exact value | O(1) average |
//! | [`btree`] | Range queries like `age > 30` | O(log n) |
//! | [`trie`] | Multi-way joins | Worst-case optimal |
//! | [`zone_map`] | Skipping chunks during scans | O(1) per chunk |
//! | [`ring`] | RDF triples (3x space reduction) | O(log σ) |
//!
//! Most queries use `adjacency` for traversals and `hash` or `btree` for filtering.
//! For RDF workloads, the `ring` index provides significant space savings.

pub mod adjacency;
pub mod btree;
pub mod hash;
#[cfg(feature = "ring-index")]
pub mod ring;
pub mod trie;
pub mod zone_map;

pub use adjacency::ChunkedAdjacency;
pub use btree::BTreeIndex;
pub use hash::HashIndex;
#[cfg(feature = "ring-index")]
pub use ring::{LeapfrogRing, RingIterator, SuccinctPermutation, TripleRing};
pub use zone_map::{BloomFilter, ZoneMapBuilder, ZoneMapEntry, ZoneMapIndex};
