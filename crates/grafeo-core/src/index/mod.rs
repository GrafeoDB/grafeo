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
//! | [`fingerprint`] | Fast miss detection in buckets | O(1) per entry |
//! | [`vector`] | Similarity search (k-NN) | O(n) brute-force, O(log n) HNSW |
//!
//! Most queries use `adjacency` for traversals and `hash` or `btree` for filtering.
//! For RDF workloads, the `ring` index provides significant space savings.
//! For AI/ML workloads, the `vector` module provides similarity search capabilities.

pub mod adjacency;
pub mod btree;
pub mod fingerprint;
pub mod fingerprinted_hash;
pub mod hash;
#[cfg(feature = "ring-index")]
pub mod ring;
pub mod trie;
pub mod vector;
pub mod zone_map;

pub use adjacency::ChunkedAdjacency;
pub use btree::BTreeIndex;
pub use fingerprint::{FingerprintBucket, FingerprintEntry, FingerprintStats};
pub use fingerprinted_hash::{AtomicFingerprintStats, FingerprintedHashIndex};
pub use hash::HashIndex;
#[cfg(feature = "ring-index")]
pub use ring::{LeapfrogRing, RingIterator, SuccinctPermutation, TripleRing};
pub use vector::{
    DistanceMetric, VectorConfig, batch_distances, brute_force_knn, brute_force_knn_filtered,
    compute_distance, cosine_distance, cosine_similarity, dot_product, euclidean_distance,
    euclidean_distance_squared, l2_norm, manhattan_distance, normalize,
};
#[cfg(feature = "vector-index")]
pub use vector::{HnswConfig, HnswIndex};
pub use zone_map::{BloomFilter, ZoneMapBuilder, ZoneMapEntry, ZoneMapIndex};
