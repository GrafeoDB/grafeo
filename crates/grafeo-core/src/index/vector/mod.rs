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
//! ```no_run
//! # #[cfg(feature = "vector-index")]
//! # {
//! use grafeo_core::index::vector::{HnswIndex, HnswConfig, DistanceMetric, VectorAccessor};
//! use grafeo_common::types::NodeId;
//! use std::sync::Arc;
//! use std::collections::HashMap;
//!
//! let config = HnswConfig::new(384, DistanceMetric::Cosine);
//! let index = HnswIndex::new(config);
//!
//! // Build an accessor backed by a HashMap
//! let mut map: HashMap<NodeId, Arc<[f32]>> = HashMap::new();
//! let embedding: Arc<[f32]> = vec![0.1f32; 384].into();
//! map.insert(NodeId::new(1), embedding.clone());
//! let accessor = |id: NodeId| -> Option<Arc<[f32]>> { map.get(&id).cloned() };
//!
//! // Insert vectors (requires accessor for neighbor lookups)
//! index.insert(NodeId::new(1), &embedding, &accessor);
//!
//! // Search (O(log n))
//! let query = vec![0.15f32; 384];
//! let results = index.search(&query, 10, &accessor);
//! # }
//! ```

mod accessor;
mod distance;
mod mmr;
pub mod quantization;
mod simd;
pub mod storage;
pub mod zone_map;

#[cfg(feature = "vector-index")]
mod config;
#[cfg(feature = "vector-index")]
mod hnsw;
#[cfg(feature = "vector-index")]
pub mod paged_topology;
#[cfg(feature = "vector-index")]
mod quantized_hnsw;
#[cfg(feature = "vector-index")]
pub mod section;

pub use accessor::{
    PropertyVectorAccessor, SpillableVectorAccessor, VectorAccessor, VectorAccessorKind,
};
pub use distance::{
    DistanceMetric, compute_distance, cosine_distance, cosine_similarity, dot_product,
    euclidean_distance, euclidean_distance_squared, l2_norm, manhattan_distance, normalize,
    simd_support,
};
pub use mmr::mmr_select;
pub use quantization::{BinaryQuantizer, ProductQuantizer, QuantizationType, ScalarQuantizer};
#[cfg(feature = "mmap")]
pub use storage::MmapStorage;
pub use storage::{RamStorage, StorageBackend, VectorStorage};
pub use zone_map::VectorZoneMap;

#[cfg(feature = "vector-index")]
pub use config::HnswConfig;
#[cfg(feature = "vector-index")]
pub use hnsw::HnswIndex;
#[cfg(feature = "vector-index")]
pub use quantized_hnsw::QuantizedHnswIndex;
#[cfg(feature = "vector-index")]
pub use section::VectorStoreSection;
// VectorIndexKind is defined below in this file (not in a sub-module).

use grafeo_common::types::NodeId;
#[cfg(feature = "vector-index")]
use std::collections::HashSet;

// ── VectorIndexKind ────────────────────────────────────────────────

/// Unified enum for vector indexes stored in the LPG store.
///
/// Wraps either a plain [`HnswIndex`] or a [`QuantizedHnswIndex`],
/// allowing the store and engine to handle both through a single type.
#[cfg(feature = "vector-index")]
pub enum VectorIndexKind {
    /// Standard full-precision HNSW index.
    Hnsw(HnswIndex),
    /// Quantized HNSW index (scalar, binary, or product quantization).
    Quantized(QuantizedHnswIndex),
}

#[cfg(feature = "vector-index")]
impl VectorIndexKind {
    /// Returns the HNSW configuration.
    #[must_use]
    pub fn config(&self) -> &HnswConfig {
        match self {
            Self::Hnsw(idx) => idx.config(),
            Self::Quantized(idx) => idx.config(),
        }
    }

    /// Returns the number of vectors in the index.
    #[must_use]
    pub fn len(&self) -> usize {
        match self {
            Self::Hnsw(idx) => idx.len(),
            Self::Quantized(idx) => idx.len(),
        }
    }

    /// Returns true if the index is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        match self {
            Self::Hnsw(idx) => idx.is_empty(),
            Self::Quantized(idx) => idx.is_empty(),
        }
    }

    /// Returns true if the index contains the given ID.
    #[must_use]
    pub fn contains(&self, id: NodeId) -> bool {
        match self {
            Self::Hnsw(idx) => idx.contains(id),
            Self::Quantized(idx) => idx.contains(id),
        }
    }

    /// Removes a vector from the index.
    pub fn remove(&self, id: NodeId) -> bool {
        match self {
            Self::Hnsw(idx) => idx.remove(id),
            Self::Quantized(idx) => idx.remove(id),
        }
    }

    /// Inserts a vector into the index.
    ///
    /// Both variants use `accessor` for topology neighbor distances. Quantized
    /// retains codes (not a dual full-f32 store after training/rehydrate).
    pub fn insert(&self, id: NodeId, vector: &[f32], accessor: &impl VectorAccessor) {
        match self {
            Self::Hnsw(idx) => idx.insert(id, vector, accessor),
            Self::Quantized(idx) => idx.insert(id, vector, accessor),
        }
    }

    /// Searches for the k nearest neighbors.
    #[must_use]
    pub fn search(
        &self,
        query: &[f32],
        k: usize,
        accessor: &impl VectorAccessor,
    ) -> Vec<(NodeId, f32)> {
        match self {
            Self::Hnsw(idx) => idx.search(query, k, accessor),
            Self::Quantized(idx) => idx.search(query, k, accessor),
        }
    }

    /// Searches with a custom ef (beam width) parameter.
    #[must_use]
    pub fn search_with_ef(
        &self,
        query: &[f32],
        k: usize,
        ef: usize,
        accessor: &impl VectorAccessor,
    ) -> Vec<(NodeId, f32)> {
        match self {
            Self::Hnsw(idx) => idx.search_with_ef(query, k, ef, accessor),
            Self::Quantized(idx) => idx.search_with_ef(query, k, ef, accessor),
        }
    }

    /// Searches with an allowlist filter.
    #[must_use]
    pub fn search_with_filter(
        &self,
        query: &[f32],
        k: usize,
        allowlist: &HashSet<NodeId>,
        accessor: &impl VectorAccessor,
    ) -> Vec<(NodeId, f32)> {
        match self {
            Self::Hnsw(idx) => idx.search_with_filter(query, k, allowlist, accessor),
            Self::Quantized(idx) => idx.search_with_filter(query, k, allowlist, accessor),
        }
    }

    /// Searches with a custom ef and an allowlist filter.
    #[must_use]
    pub fn search_with_ef_and_filter(
        &self,
        query: &[f32],
        k: usize,
        ef: usize,
        allowlist: &HashSet<NodeId>,
        accessor: &impl VectorAccessor,
    ) -> Vec<(NodeId, f32)> {
        match self {
            Self::Hnsw(idx) => idx.search_with_ef_and_filter(query, k, ef, allowlist, accessor),
            Self::Quantized(idx) => idx.search_with_ef_and_filter(query, k, ef, allowlist, accessor),
        }
    }

    /// Batch search for multiple queries.
    #[must_use]
    pub fn batch_search(
        &self,
        queries: &[Vec<f32>],
        k: usize,
        accessor: &impl VectorAccessor,
    ) -> Vec<Vec<(NodeId, f32)>> {
        match self {
            Self::Hnsw(idx) => idx.batch_search(queries, k, accessor),
            Self::Quantized(idx) => idx.batch_search(queries, k, accessor),
        }
    }

    /// Batch search with custom ef for multiple queries.
    #[must_use]
    pub fn batch_search_with_ef(
        &self,
        queries: &[Vec<f32>],
        k: usize,
        ef: usize,
        accessor: &impl VectorAccessor,
    ) -> Vec<Vec<(NodeId, f32)>> {
        match self {
            Self::Hnsw(idx) => idx.batch_search_with_ef(queries, k, ef, accessor),
            Self::Quantized(idx) => idx.batch_search_with_ef(queries, k, ef, accessor),
        }
    }

    /// Batch search with an allowlist filter for multiple queries.
    #[must_use]
    pub fn batch_search_with_filter(
        &self,
        queries: &[Vec<f32>],
        k: usize,
        allowlist: &HashSet<NodeId>,
        accessor: &impl VectorAccessor,
    ) -> Vec<Vec<(NodeId, f32)>> {
        match self {
            Self::Hnsw(idx) => idx.batch_search_with_filter(queries, k, allowlist, accessor),
            Self::Quantized(idx) => idx.batch_search_with_filter(queries, k, allowlist, accessor),
        }
    }

    /// Batch search with custom ef and an allowlist filter.
    #[must_use]
    pub fn batch_search_with_ef_and_filter(
        &self,
        queries: &[Vec<f32>],
        k: usize,
        ef: usize,
        allowlist: &HashSet<NodeId>,
        accessor: &impl VectorAccessor,
    ) -> Vec<Vec<(NodeId, f32)>> {
        match self {
            Self::Hnsw(idx) => {
                idx.batch_search_with_ef_and_filter(queries, k, ef, allowlist, accessor)
            }
            Self::Quantized(idx) => idx.batch_search_with_ef_and_filter(queries, k, ef, allowlist, accessor),
        }
    }

    /// Snapshot the HNSW topology for serialization.
    #[must_use]
    pub fn snapshot_topology(&self) -> (Option<NodeId>, usize, Vec<(NodeId, Vec<Vec<NodeId>>)>) {
        match self {
            Self::Hnsw(idx) => idx.snapshot_topology(),
            Self::Quantized(idx) => idx.snapshot_topology(),
        }
    }

    /// Restore topology from a snapshot.
    pub fn restore_topology(
        &self,
        entry_point: Option<NodeId>,
        max_level: usize,
        node_data: Vec<(NodeId, Vec<Vec<NodeId>>)>,
    ) {
        match self {
            Self::Hnsw(idx) => idx.restore_topology(entry_point, max_level, node_data),
            Self::Quantized(idx) => idx.restore_topology(entry_point, max_level, node_data),
        }
    }

    /// Rehydrate quantized payloads without topology rebuild.
    ///
    /// No-op for plain [`Hnsw`] shells (search uses LPG property accessor).
    pub fn rehydrate_payloads_from_vectors(
        &self,
        vectors: impl IntoIterator<Item = (NodeId, Vec<f32>)>,
    ) {
        match self {
            Self::Hnsw(_) => {
                // Plain HNSW keeps embeddings in LPG; nothing to rehydrate.
                let _ = vectors.into_iter().count();
            }
            Self::Quantized(idx) => idx.rehydrate_payloads_from_vectors(vectors),
        }
    }

    /// Returns estimated heap memory in bytes.
    #[must_use]
    pub fn heap_memory_bytes(&self) -> usize {
        match self {
            Self::Hnsw(idx) => idx.heap_memory_bytes(),
            Self::Quantized(idx) => idx.heap_memory_bytes(),
        }
    }

    /// Returns the quantization type, if this is a quantized index.
    ///
    /// Note: plain `Hnsw` returns `None` (not `Some(QuantizationType::None)`).
    /// Public DB inspect APIs map missing-index vs plain-mode separately.
    #[must_use]
    pub fn quantization_type(&self) -> Option<QuantizationType> {
        match self {
            Self::Hnsw(_) => None,
            Self::Quantized(idx) => Some(idx.quantization_type()),
        }
    }

    /// Returns a reference to the inner `HnswIndex`, if this is a plain HNSW variant.
    #[must_use]
    pub fn as_hnsw(&self) -> Option<&HnswIndex> {
        match self {
            Self::Hnsw(idx) => Some(idx),
            Self::Quantized(_) => None,
        }
    }

    /// Returns a reference to the inner `QuantizedHnswIndex`, if quantized.
    #[must_use]
    pub fn as_quantized(&self) -> Option<&QuantizedHnswIndex> {
        match self {
            Self::Hnsw(_) => None,
            Self::Quantized(idx) => Some(idx),
        }
    }
}

#[cfg(feature = "vector-index")]
impl From<HnswIndex> for VectorIndexKind {
    fn from(idx: HnswIndex) -> Self {
        Self::Hnsw(idx)
    }
}

#[cfg(feature = "vector-index")]
impl From<QuantizedHnswIndex> for VectorIndexKind {
    fn from(idx: QuantizedHnswIndex) -> Self {
        Self::Quantized(idx)
    }
}

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

    // ── VectorIndexKind Quantized dispatch ────────────────────────────

    #[cfg(feature = "vector-index")]
    mod vector_index_kind_tests {
        use super::super::*;
        use std::collections::HashSet;

        /// Minimal accessor that always returns None (quantized indexes
        /// store vectors internally so the accessor is unused).
        struct NoopAccessor;
        impl VectorAccessor for NoopAccessor {
            fn get_vector(&self, _id: NodeId) -> Option<std::sync::Arc<[f32]>> {
                None
            }
        }

        fn build_quantized_kind(n: usize) -> VectorIndexKind {
            let config = HnswConfig::new(4, DistanceMetric::Euclidean);
            let q = QuantizedHnswIndex::new(config, QuantizationType::Scalar);
            for i in 0..n {
                let vec: Vec<f32> = (0..4)
                    .map(|j| ((i * 4 + j) as f32) / (n * 4) as f32)
                    .collect();
                q.test_insert(NodeId::new(i as u64 + 1), &vec);
            }
            VectorIndexKind::Quantized(q)
        }

        #[test]
        fn quantized_kind_basic_ops() {
            let kind = build_quantized_kind(20);
            assert_eq!(kind.len(), 20);
            assert!(!kind.is_empty());
            assert!(kind.contains(NodeId::new(1)));
            assert!(!kind.contains(NodeId::new(999)));
        }

        #[test]
        fn quantized_kind_insert_and_search() {
            let kind = build_quantized_kind(30);
            let query = vec![0.5, 0.5, 0.0, 0.0];
            let results = kind.search(&query, 3, &NoopAccessor);
            assert_eq!(results.len(), 3);
        }

        #[test]
        fn quantized_kind_search_with_ef() {
            let kind = build_quantized_kind(30);
            let query = vec![0.5, 0.5, 0.0, 0.0];
            let results = kind.search_with_ef(&query, 3, 50, &NoopAccessor);
            assert_eq!(results.len(), 3);
        }

        #[test]
        fn quantized_kind_search_with_filter() {
            let kind = build_quantized_kind(30);
            let allowlist: HashSet<NodeId> = (1..=10).map(NodeId::new).collect();
            let query = vec![0.1, 0.1, 0.0, 0.0];
            let results = kind.search_with_filter(&query, 5, &allowlist, &NoopAccessor);
            assert!(!results.is_empty());
            for (id, _) in &results {
                assert!(allowlist.contains(id));
            }
        }

        #[test]
        fn quantized_kind_search_with_ef_and_filter() {
            let kind = build_quantized_kind(30);
            let allowlist: HashSet<NodeId> = (5..=15).map(NodeId::new).collect();
            let query = vec![0.3, 0.3, 0.0, 0.0];
            let results = kind.search_with_ef_and_filter(&query, 3, 50, &allowlist, &NoopAccessor);
            for (id, _) in &results {
                assert!(allowlist.contains(id));
            }
        }

        #[test]
        fn quantized_kind_batch_search() {
            let kind = build_quantized_kind(30);
            let queries = vec![vec![0.1, 0.0, 0.0, 0.0], vec![0.9, 0.9, 0.0, 0.0]];
            let results = kind.batch_search(&queries, 2, &NoopAccessor);
            assert_eq!(results.len(), 2);
        }

        #[test]
        fn quantized_kind_batch_search_with_ef() {
            let kind = build_quantized_kind(30);
            let queries = vec![vec![0.1, 0.0, 0.0, 0.0]];
            let results = kind.batch_search_with_ef(&queries, 2, 50, &NoopAccessor);
            assert_eq!(results.len(), 1);
        }

        #[test]
        fn quantized_kind_batch_search_with_filter() {
            let kind = build_quantized_kind(30);
            let allowlist: HashSet<NodeId> = (1..=10).map(NodeId::new).collect();
            let queries = vec![vec![0.1, 0.0, 0.0, 0.0]];
            let results = kind.batch_search_with_filter(&queries, 5, &allowlist, &NoopAccessor);
            assert_eq!(results.len(), 1);
            for (id, _) in &results[0] {
                assert!(allowlist.contains(id));
            }
        }

        #[test]
        fn quantized_kind_batch_search_with_ef_and_filter() {
            let kind = build_quantized_kind(30);
            let allowlist: HashSet<NodeId> = (1..=15).map(NodeId::new).collect();
            let queries = vec![vec![0.2, 0.0, 0.0, 0.0]];
            let results =
                kind.batch_search_with_ef_and_filter(&queries, 3, 50, &allowlist, &NoopAccessor);
            assert_eq!(results.len(), 1);
            for (id, _) in &results[0] {
                assert!(allowlist.contains(id));
            }
        }

        #[test]
        fn quantized_kind_remove() {
            let kind = build_quantized_kind(5);
            assert!(kind.remove(NodeId::new(1)));
            assert_eq!(kind.len(), 4);
            assert!(!kind.contains(NodeId::new(1)));
        }

        #[test]
        // reason: test indices 0..10 are non-negative
        #[allow(clippy::cast_sign_loss)]
        fn quantized_kind_snapshot_restore() {
            let kind = build_quantized_kind(10);
            let (entry, level, nodes) = kind.snapshot_topology();

            let config = HnswConfig::new(4, DistanceMetric::Euclidean);
            let kind2 = VectorIndexKind::Quantized(QuantizedHnswIndex::new(
                config,
                QuantizationType::Scalar,
            ));
            for i in 0..10 {
                let vec: Vec<f32> = (0..4).map(|j| ((i * 4 + j) as f32) / 40.0).collect();
                kind2.insert(NodeId::new(i as u64 + 1), &vec, &NoopAccessor);
            }
            kind2.restore_topology(entry, level, nodes);
            assert_eq!(kind2.len(), 10);
        }

        #[test]
        fn quantized_kind_heap_memory() {
            let kind = build_quantized_kind(10);
            assert!(kind.heap_memory_bytes() > 0);
        }

        #[test]
        fn quantized_kind_type_accessors() {
            let kind = build_quantized_kind(1);
            assert!(kind.as_quantized().is_some());
            assert!(kind.as_hnsw().is_none());
            assert_eq!(kind.quantization_type(), Some(QuantizationType::Scalar));
        }

        #[test]
        fn from_trait_impls() {
            let config = HnswConfig::new(4, DistanceMetric::Euclidean);

            let hnsw = HnswIndex::new(config.clone());
            let kind: VectorIndexKind = hnsw.into();
            assert!(kind.as_hnsw().is_some());

            let quantized = QuantizedHnswIndex::new(config, QuantizationType::Binary);
            let kind2: VectorIndexKind = quantized.into();
            assert!(kind2.as_quantized().is_some());
            assert_eq!(kind2.quantization_type(), Some(QuantizationType::Binary));
        }
    }
}
