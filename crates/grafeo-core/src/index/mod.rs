//! Index structures that make queries fast.
//!
//! Pick the right index for your access pattern:
//!
//! | Index | Best for | Complexity |
//! | ----- | -------- | ---------- |
//! | [`adjacency`] | Traversing neighbors | O(degree) |
//! | [`hash`] | Point lookups by exact value | O(1) average |
//! | [`trie`] | Multi-way joins | Worst-case optimal |
//! | [`zone_map`] | Skipping chunks during scans | O(1) per chunk |
//! | `ring` | RDF triples (3x space reduction, feature-gated: `ring-index`) | O(log σ) |
//! | [`vector`] | Similarity search (k-NN) | O(n) brute-force, O(log n) HNSW |
//! | [`text`] | Full-text search (BM25 scoring) | O(terms × postings) |
//!
//! Most queries use `adjacency` for traversals and `hash` for filtering.
//! For RDF workloads, the `ring` index provides significant space savings.
//! For AI/ML workloads, the `vector` module provides similarity search capabilities.

pub mod adjacency;
pub mod hash;
#[cfg(feature = "ring-index")]
pub mod ring;
pub mod text;
pub mod trie;
pub mod vector;
pub mod zone_map;

pub use adjacency::ChunkedAdjacency;
pub use hash::HashIndex;
#[cfg(feature = "ring-index")]
pub use ring::{LeapfrogRing, RingIterator, SuccinctPermutation, TripleRing};
#[cfg(feature = "text-index")]
pub use text::{BM25Config, InvertedIndex, SimpleTokenizer, Tokenizer};
#[cfg(feature = "hybrid-search")]
pub use text::{FusionMethod, fuse_results};
pub use vector::{
    DistanceMetric, VectorConfig, batch_distances, brute_force_knn, brute_force_knn_filtered,
    compute_distance, cosine_distance, cosine_similarity, dot_product, euclidean_distance,
    euclidean_distance_squared, l2_norm, manhattan_distance, normalize,
};
#[cfg(feature = "vector-index")]
pub use vector::{HnswConfig, HnswIndex, VectorIndexKind};
pub use zone_map::{BloomFilter, ZoneMapBuilder, ZoneMapEntry, ZoneMapIndex};
