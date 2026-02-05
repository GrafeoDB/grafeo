//! Vector similarity search support.
//!
//! This module provides infrastructure for storing and searching vector embeddings,
//! enabling AI/ML use cases like RAG, semantic search, and recommendations.
//!
//! # Distance Metrics
//!
//! Choose the metric based on your embedding type:
//!
//! | Metric | Best For | Range |
//! |--------|----------|-------|
//! | [`Cosine`](DistanceMetric::Cosine) | Normalized embeddings (text) | [0, 2] |
//! | [`Euclidean`](DistanceMetric::Euclidean) | Raw embeddings | [0, inf) |
//! | [`DotProduct`](DistanceMetric::DotProduct) | Max inner product search | (-inf, inf) |
//! | [`Manhattan`](DistanceMetric::Manhattan) | Outlier-resistant | [0, inf) |
//!
//! # Index Types
//!
//! | Index | Complexity | Use Case |
//! |-------|------------|----------|
//! | [`brute_force_knn`] | O(n) | Small datasets, exact results |
//! | [`HnswIndex`] | O(log n) | Large datasets, approximate results |
//!
//! # Example
//!
//! ```
//! use grafeo_core::index::vector::{compute_distance, DistanceMetric, brute_force_knn};
//! use grafeo_common::types::NodeId;
//!
//! // Compute distance between two vectors
//! let query = [0.1f32, 0.2, 0.3];
//! let doc1 = [0.1f32, 0.2, 0.35];
//! let doc2 = [0.5f32, 0.6, 0.7];
//!
//! let dist1 = compute_distance(&query, &doc1, DistanceMetric::Cosine);
//! let dist2 = compute_distance(&query, &doc2, DistanceMetric::Cosine);
//!
//! // doc1 is more similar (smaller distance)
//! assert!(dist1 < dist2);
//!
//! // Brute-force k-NN search
//! let vectors = vec![
//!     (NodeId::new(1), doc1.as_slice()),
//!     (NodeId::new(2), doc2.as_slice()),
//! ];
//!
//! let results = brute_force_knn(vectors.into_iter(), &query, 1, DistanceMetric::Cosine);
//! assert_eq!(results[0].0, NodeId::new(1)); // doc1 is closest
//! ```
//!
//! # HNSW Index (requires `vector-index` feature)
//!
//! For larger datasets, use the HNSW approximate nearest neighbor index:
//!
//! ```ignore
//! use grafeo_core::index::vector::{HnswIndex, HnswConfig, DistanceMetric};
//! use grafeo_common::types::NodeId;
//!
//! let config = HnswConfig::new(384, DistanceMetric::Cosine);
//! let index = HnswIndex::new(config);
//!
//! // Insert vectors
//! index.insert(NodeId::new(1), &embedding);
//!
//! // Search (O(log n))
//! let results = index.search(&query, 10);
//! ```

mod distance;
pub mod quantization;
mod simd;
pub mod storage;
pub mod zone_map;

#[cfg(feature = "vector-index")]
mod config;
#[cfg(feature = "vector-index")]
mod hnsw;
#[cfg(feature = "vector-index")]
mod quantized_hnsw;

pub use distance::{
    DistanceMetric, compute_distance, cosine_distance, cosine_similarity, dot_product,
    euclidean_distance, euclidean_distance_squared, l2_norm, manhattan_distance, normalize,
    simd_support,
};
pub use quantization::{BinaryQuantizer, ProductQuantizer, QuantizationType, ScalarQuantizer};
pub use storage::{MmapStorage, RamStorage, StorageBackend, VectorStorage};
pub use zone_map::VectorZoneMap;

#[cfg(feature = "vector-index")]
pub use config::HnswConfig;
#[cfg(feature = "vector-index")]
pub use hnsw::HnswIndex;
#[cfg(feature = "vector-index")]
pub use quantized_hnsw::QuantizedHnswIndex;

use grafeo_common::types::NodeId;

/// Configuration for vector search operations.
#[derive(Debug, Clone)]
pub struct VectorConfig {
    /// Expected vector dimensions (for validation).
    pub dimensions: usize,
    /// Distance metric for similarity computation.
    pub metric: DistanceMetric,
}

impl VectorConfig {
    /// Creates a new vector configuration.
    #[must_use]
    pub const fn new(dimensions: usize, metric: DistanceMetric) -> Self {
        Self { dimensions, metric }
    }

    /// Creates a configuration for cosine similarity with the given dimensions.
    #[must_use]
    pub const fn cosine(dimensions: usize) -> Self {
        Self::new(dimensions, DistanceMetric::Cosine)
    }

    /// Creates a configuration for Euclidean distance with the given dimensions.
    #[must_use]
    pub const fn euclidean(dimensions: usize) -> Self {
        Self::new(dimensions, DistanceMetric::Euclidean)
    }
}

impl Default for VectorConfig {
    fn default() -> Self {
        Self {
            dimensions: 384, // Common embedding size (MiniLM, etc.)
            metric: DistanceMetric::default(),
        }
    }
}

/// Performs brute-force k-nearest neighbor search.
///
/// This is O(n) where n is the number of vectors. Use this for:
/// - Small datasets (< 10K vectors)
/// - Baseline comparisons
/// - Exact nearest neighbor search
///
/// For larger datasets, use an approximate index like HNSW.
///
/// # Arguments
///
/// * `vectors` - Iterator of (id, vector) pairs to search
/// * `query` - The query vector
/// * `k` - Number of nearest neighbors to return
/// * `metric` - Distance metric to use
///
/// # Returns
///
/// Vector of (id, distance) pairs sorted by distance (ascending).
///
/// # Example
///
/// ```
/// use grafeo_core::index::vector::{brute_force_knn, DistanceMetric};
/// use grafeo_common::types::NodeId;
///
/// let vectors = vec![
///     (NodeId::new(1), [0.1f32, 0.2, 0.3].as_slice()),
///     (NodeId::new(2), [0.4f32, 0.5, 0.6].as_slice()),
///     (NodeId::new(3), [0.7f32, 0.8, 0.9].as_slice()),
/// ];
///
/// let query = [0.15f32, 0.25, 0.35];
/// let results = brute_force_knn(vectors.into_iter(), &query, 2, DistanceMetric::Euclidean);
///
/// assert_eq!(results.len(), 2);
/// assert_eq!(results[0].0, NodeId::new(1)); // Closest
/// ```
pub fn brute_force_knn<'a, I>(
    vectors: I,
    query: &[f32],
    k: usize,
    metric: DistanceMetric,
) -> Vec<(NodeId, f32)>
where
    I: Iterator<Item = (NodeId, &'a [f32])>,
{
    let mut results: Vec<(NodeId, f32)> = vectors
        .map(|(id, vec)| (id, compute_distance(query, vec, metric)))
        .collect();

    // Sort by distance (ascending)
    results.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

    // Truncate to k
    results.truncate(k);
    results
}

/// Performs brute-force k-nearest neighbor search with a filter predicate.
///
/// Only considers vectors where the predicate returns true.
///
/// # Arguments
///
/// * `vectors` - Iterator of (id, vector) pairs to search
/// * `query` - The query vector
/// * `k` - Number of nearest neighbors to return
/// * `metric` - Distance metric to use
/// * `predicate` - Filter function; only vectors where this returns true are considered
///
/// # Returns
///
/// Vector of (id, distance) pairs sorted by distance (ascending).
pub fn brute_force_knn_filtered<'a, I, F>(
    vectors: I,
    query: &[f32],
    k: usize,
    metric: DistanceMetric,
    predicate: F,
) -> Vec<(NodeId, f32)>
where
    I: Iterator<Item = (NodeId, &'a [f32])>,
    F: Fn(NodeId) -> bool,
{
    let mut results: Vec<(NodeId, f32)> = vectors
        .filter(|(id, _)| predicate(*id))
        .map(|(id, vec)| (id, compute_distance(query, vec, metric)))
        .collect();

    results.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
    results.truncate(k);
    results
}

/// Computes the distance between a query and multiple vectors in batch.
///
/// More efficient than computing distances one by one for large batches.
///
/// # Returns
///
/// Vector of (id, distance) pairs in the same order as input.
pub fn batch_distances<'a, I>(
    vectors: I,
    query: &[f32],
    metric: DistanceMetric,
) -> Vec<(NodeId, f32)>
where
    I: Iterator<Item = (NodeId, &'a [f32])>,
{
    vectors
        .map(|(id, vec)| (id, compute_distance(query, vec, metric)))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vector_config_default() {
        let config = VectorConfig::default();
        assert_eq!(config.dimensions, 384);
        assert_eq!(config.metric, DistanceMetric::Cosine);
    }

    #[test]
    fn test_vector_config_constructors() {
        let cosine = VectorConfig::cosine(768);
        assert_eq!(cosine.dimensions, 768);
        assert_eq!(cosine.metric, DistanceMetric::Cosine);

        let euclidean = VectorConfig::euclidean(1536);
        assert_eq!(euclidean.dimensions, 1536);
        assert_eq!(euclidean.metric, DistanceMetric::Euclidean);
    }

    #[test]
    fn test_brute_force_knn() {
        let vectors = vec![
            (NodeId::new(1), [0.0f32, 0.0, 0.0].as_slice()),
            (NodeId::new(2), [1.0f32, 0.0, 0.0].as_slice()),
            (NodeId::new(3), [2.0f32, 0.0, 0.0].as_slice()),
            (NodeId::new(4), [3.0f32, 0.0, 0.0].as_slice()),
        ];

        let query = [0.5f32, 0.0, 0.0];
        let results = brute_force_knn(vectors.into_iter(), &query, 2, DistanceMetric::Euclidean);

        assert_eq!(results.len(), 2);
        // Closest should be node 1 (dist 0.5) or node 2 (dist 0.5)
        assert!(results[0].0 == NodeId::new(1) || results[0].0 == NodeId::new(2));
    }

    #[test]
    fn test_brute_force_knn_empty() {
        let vectors: Vec<(NodeId, &[f32])> = vec![];
        let query = [0.0f32, 0.0];
        let results = brute_force_knn(vectors.into_iter(), &query, 10, DistanceMetric::Cosine);
        assert!(results.is_empty());
    }

    #[test]
    fn test_brute_force_knn_k_larger_than_n() {
        let vectors = vec![
            (NodeId::new(1), [0.0f32, 0.0].as_slice()),
            (NodeId::new(2), [1.0f32, 0.0].as_slice()),
        ];

        let query = [0.0f32, 0.0];
        let results = brute_force_knn(vectors.into_iter(), &query, 10, DistanceMetric::Euclidean);

        // Should return all 2 vectors, not 10
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn test_brute_force_knn_filtered() {
        let vectors = vec![
            (NodeId::new(1), [0.0f32, 0.0].as_slice()),
            (NodeId::new(2), [1.0f32, 0.0].as_slice()),
            (NodeId::new(3), [2.0f32, 0.0].as_slice()),
        ];

        let query = [0.0f32, 0.0];

        // Only consider even IDs
        let results = brute_force_knn_filtered(
            vectors.into_iter(),
            &query,
            10,
            DistanceMetric::Euclidean,
            |id| id.as_u64() % 2 == 0,
        );

        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0, NodeId::new(2));
    }

    #[test]
    fn test_batch_distances() {
        let vectors = vec![
            (NodeId::new(1), [0.0f32, 0.0].as_slice()),
            (NodeId::new(2), [3.0f32, 4.0].as_slice()),
        ];

        let query = [0.0f32, 0.0];
        let results = batch_distances(vectors.into_iter(), &query, DistanceMetric::Euclidean);

        assert_eq!(results.len(), 2);
        assert_eq!(results[0].0, NodeId::new(1));
        assert!((results[0].1 - 0.0).abs() < 0.001);
        assert_eq!(results[1].0, NodeId::new(2));
        assert!((results[1].1 - 5.0).abs() < 0.001); // 3-4-5 triangle
    }
}
