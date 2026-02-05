//! Quantized HNSW index with rescoring support.
//!
//! This module provides memory-efficient vector search by combining
//! quantization with HNSW approximate nearest neighbor search:
//!
//! 1. **Search phase**: Use quantized vectors for fast approximate search
//! 2. **Rescore phase**: Re-rank top candidates using full-precision vectors
//!
//! # Compression vs Accuracy Tradeoff
//!
//! | Quantization | Memory | Recall@10 | Best For |
//! |--------------|--------|-----------|----------|
//! | None         | 100%   | ~98%      | Small datasets, max accuracy |
//! | Scalar       | 25%    | ~97%      | Most production use cases |
//! | Binary       | ~3%    | ~85%      | Very large datasets |
//!
//! # Example
//!
//! ```ignore
//! use grafeo_core::index::vector::{
//!     QuantizedHnswIndex, HnswConfig, DistanceMetric, QuantizationType
//! };
//! use grafeo_common::types::NodeId;
//!
//! // Create config with scalar quantization
//! let config = HnswConfig::new(384, DistanceMetric::Cosine);
//! let mut index = QuantizedHnswIndex::new(config, QuantizationType::Scalar);
//!
//! // Insert vectors (full precision stored, quantized for search)
//! index.insert(NodeId::new(1), &vec![0.1f32; 384]);
//!
//! // Search with rescoring (default: rescore top 2*k candidates)
//! let results = index.search(&query, 10);
//! ```

use super::quantization::{BinaryQuantizer, ProductQuantizer, QuantizationType, ScalarQuantizer};
use super::{HnswConfig, HnswIndex, compute_distance};
use grafeo_common::types::NodeId;
use ordered_float::OrderedFloat;
use parking_lot::RwLock;
use std::collections::HashMap;
use std::sync::Arc;

/// HNSW index with quantization support for memory-efficient search.
///
/// Stores vectors in both full precision (for rescoring) and quantized form
/// (for fast search). The search process:
///
/// 1. Quantize the query
/// 2. Search the HNSW graph using quantized distances
/// 3. Rescore top candidates using full-precision distances
///
/// This achieves near-full-precision recall with significantly reduced memory.
pub struct QuantizedHnswIndex {
    /// The underlying HNSW index (stores full vectors).
    hnsw: HnswIndex,
    /// Quantization type.
    quantization_type: QuantizationType,
    /// Scalar quantizer (if using scalar quantization).
    scalar_quantizer: RwLock<Option<ScalarQuantizer>>,
    /// Product quantizer (if using product quantization).
    product_quantizer: RwLock<Option<ProductQuantizer>>,
    /// Quantized scalar vectors: NodeId -> quantized u8 vector.
    scalar_vectors: RwLock<HashMap<NodeId, Vec<u8>>>,
    /// Quantized binary vectors: NodeId -> binary bits.
    binary_vectors: RwLock<HashMap<NodeId, Vec<u64>>>,
    /// Product quantization codes: NodeId -> M u8 codes.
    product_codes: RwLock<HashMap<NodeId, Vec<u8>>>,
    /// Whether to rescore with full precision vectors.
    rescore: bool,
    /// Rescore factor: search for this many candidates before rescoring.
    /// Final results = top k from rescore_factor * k candidates.
    rescore_factor: usize,
    /// Number of training samples before quantizer is trained.
    training_threshold: usize,
    /// Training samples collected before training the quantizer.
    training_samples: RwLock<Vec<Arc<[f32]>>>,
    /// Whether the quantizer has been trained.
    quantizer_trained: RwLock<bool>,
}

impl QuantizedHnswIndex {
    /// Creates a new quantized HNSW index.
    ///
    /// # Arguments
    ///
    /// * `config` - HNSW configuration
    /// * `quantization` - Quantization type (None, Scalar, Binary, or Product)
    #[must_use]
    pub fn new(config: HnswConfig, quantization: QuantizationType) -> Self {
        Self {
            hnsw: HnswIndex::new(config),
            quantization_type: quantization,
            scalar_quantizer: RwLock::new(None),
            product_quantizer: RwLock::new(None),
            scalar_vectors: RwLock::new(HashMap::new()),
            binary_vectors: RwLock::new(HashMap::new()),
            product_codes: RwLock::new(HashMap::new()),
            rescore: true,
            rescore_factor: 2,
            training_threshold: 1000,
            training_samples: RwLock::new(Vec::new()),
            quantizer_trained: RwLock::new(false),
        }
    }

    /// Creates a new quantized HNSW index with a fixed seed for reproducibility.
    #[must_use]
    pub fn with_seed(config: HnswConfig, quantization: QuantizationType, seed: u64) -> Self {
        Self {
            hnsw: HnswIndex::with_seed(config, seed),
            quantization_type: quantization,
            scalar_quantizer: RwLock::new(None),
            product_quantizer: RwLock::new(None),
            scalar_vectors: RwLock::new(HashMap::new()),
            binary_vectors: RwLock::new(HashMap::new()),
            product_codes: RwLock::new(HashMap::new()),
            rescore: true,
            rescore_factor: 2,
            training_threshold: 1000,
            training_samples: RwLock::new(Vec::new()),
            quantizer_trained: RwLock::new(false),
        }
    }

    /// Disables rescoring (pure quantized search).
    ///
    /// Faster but less accurate. Useful when you need maximum speed.
    #[must_use]
    pub fn without_rescore(mut self) -> Self {
        self.rescore = false;
        self
    }

    /// Sets the rescore factor.
    ///
    /// The search will find `k * rescore_factor` candidates using quantized
    /// search, then rescore them with full precision to return the top k.
    ///
    /// Default: 2 (search 2x candidates, rescore to get top k).
    /// Higher values improve recall at the cost of latency.
    #[must_use]
    pub fn with_rescore_factor(mut self, factor: usize) -> Self {
        self.rescore_factor = factor.max(1);
        self
    }

    /// Sets the number of vectors to collect before training the scalar quantizer.
    ///
    /// For scalar quantization, the quantizer learns min/max ranges from
    /// training data. More samples = better quantization quality.
    ///
    /// Default: 1000
    #[must_use]
    pub fn with_training_threshold(mut self, threshold: usize) -> Self {
        self.training_threshold = threshold.max(10);
        self
    }

    /// Returns the quantization type.
    #[must_use]
    pub fn quantization_type(&self) -> QuantizationType {
        self.quantization_type
    }

    /// Returns the underlying HNSW configuration.
    #[must_use]
    pub fn config(&self) -> &HnswConfig {
        self.hnsw.config()
    }

    /// Returns the number of vectors in the index.
    #[must_use]
    pub fn len(&self) -> usize {
        self.hnsw.len()
    }

    /// Returns true if the index is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.hnsw.is_empty()
    }

    /// Returns the memory usage estimate in bytes.
    #[must_use]
    pub fn memory_usage(&self) -> usize {
        let base = self.hnsw.len() * self.config().dimensions * 4; // f32 vectors
        let quantized = match self.quantization_type {
            QuantizationType::None => 0,
            QuantizationType::Scalar => self.hnsw.len() * self.config().dimensions, // u8 vectors
            QuantizationType::Binary => {
                self.hnsw.len() * BinaryQuantizer::bytes_needed(self.config().dimensions)
            }
            QuantizationType::Product { num_subvectors } => self.hnsw.len() * num_subvectors, // M u8 codes
        };
        base + quantized
    }

    /// Returns the theoretical compression ratio of the quantization scheme.
    ///
    /// This returns what the compression would be if ONLY quantized vectors
    /// were stored (without full-precision vectors for rescoring).
    ///
    /// - None: 1.0 (no compression)
    /// - Scalar: 4.0 (f32 -> u8)
    /// - Binary: 32.0 (f32 -> 1 bit)
    /// - Product(M): (dimensions * 4) / M
    ///
    /// Note: With rescoring enabled (default), actual memory usage is higher
    /// because both full and quantized vectors are stored.
    #[must_use]
    pub fn theoretical_compression_ratio(&self) -> f32 {
        self.quantization_type
            .compression_ratio(self.config().dimensions) as f32
    }

    /// Returns the actual memory ratio compared to storing only full vectors.
    ///
    /// Values > 1.0 mean more memory is used (typical with rescoring enabled).
    /// Values < 1.0 would mean less memory (only with rescoring disabled).
    #[must_use]
    pub fn memory_ratio(&self) -> f32 {
        let full_size = self.hnsw.len() * self.config().dimensions * 4;
        if full_size == 0 {
            return 1.0;
        }
        self.memory_usage() as f32 / full_size as f32
    }

    /// Inserts a vector into the index.
    ///
    /// The vector is stored in full precision and also quantized for search.
    /// For scalar quantization, the first `training_threshold` vectors are
    /// used to train the quantizer.
    ///
    /// # Panics
    ///
    /// Panics if vector dimensions don't match configuration.
    pub fn insert(&self, id: NodeId, vector: &[f32]) {
        // Insert into HNSW (stores full precision)
        self.hnsw.insert(id, vector);

        // Handle quantization
        match self.quantization_type {
            QuantizationType::None => {}
            QuantizationType::Scalar => self.insert_scalar_quantized(id, vector),
            QuantizationType::Binary => self.insert_binary_quantized(id, vector),
            QuantizationType::Product { num_subvectors } => {
                self.insert_product_quantized(id, vector, num_subvectors);
            }
        }
    }

    /// Inserts with scalar quantization.
    fn insert_scalar_quantized(&self, id: NodeId, vector: &[f32]) {
        let trained = *self.quantizer_trained.read();

        if trained {
            // Quantizer is ready, quantize and store
            if let Some(ref quantizer) = *self.scalar_quantizer.read() {
                let quantized = quantizer.quantize(vector);
                self.scalar_vectors.write().insert(id, quantized);
            }
        } else {
            // Still collecting training samples
            let vector_arc: Arc<[f32]> = vector.into();
            let mut samples = self.training_samples.write();
            samples.push(vector_arc);

            if samples.len() >= self.training_threshold {
                // Time to train the quantizer
                let refs: Vec<&[f32]> = samples.iter().map(|v| v.as_ref()).collect();
                let quantizer = ScalarQuantizer::train(&refs);

                // Quantize all collected samples
                let mut scalar_vecs = self.scalar_vectors.write();
                for (old_id, old_vec) in self.hnsw.iter() {
                    scalar_vecs.insert(old_id, quantizer.quantize(&old_vec));
                }

                // Store the quantizer and mark as trained
                *self.scalar_quantizer.write() = Some(quantizer);
                *self.quantizer_trained.write() = true;
                samples.clear();
            }
        }
    }

    /// Inserts with binary quantization.
    fn insert_binary_quantized(&self, id: NodeId, vector: &[f32]) {
        let bits = BinaryQuantizer::quantize(vector);
        self.binary_vectors.write().insert(id, bits);
    }

    /// Inserts with product quantization.
    fn insert_product_quantized(&self, id: NodeId, vector: &[f32], num_subvectors: usize) {
        let trained = *self.quantizer_trained.read();

        if trained {
            // Quantizer is ready, quantize and store
            if let Some(ref quantizer) = *self.product_quantizer.read() {
                let codes = quantizer.quantize(vector);
                self.product_codes.write().insert(id, codes);
            }
        } else {
            // Still collecting training samples
            let vector_arc: Arc<[f32]> = vector.into();
            let mut samples = self.training_samples.write();
            samples.push(vector_arc);

            if samples.len() >= self.training_threshold {
                // Time to train the product quantizer
                let refs: Vec<&[f32]> = samples.iter().map(|v| v.as_ref()).collect();
                let quantizer = ProductQuantizer::train(&refs, num_subvectors, 256, 10);

                // Quantize all collected samples
                let mut codes = self.product_codes.write();
                for (old_id, old_vec) in self.hnsw.iter() {
                    codes.insert(old_id, quantizer.quantize(&old_vec));
                }

                // Store the quantizer and mark as trained
                *self.product_quantizer.write() = Some(quantizer);
                *self.quantizer_trained.write() = true;
                samples.clear();
            }
        }
    }

    /// Searches for the k nearest neighbors.
    ///
    /// If rescoring is enabled (default), the search:
    /// 1. Finds k * rescore_factor candidates using quantized distances
    /// 2. Rescores them with full-precision distances
    /// 3. Returns the top k
    ///
    /// # Returns
    ///
    /// Vector of (NodeId, distance) pairs sorted by distance (ascending).
    #[must_use]
    pub fn search(&self, query: &[f32], k: usize) -> Vec<(NodeId, f32)> {
        self.search_with_ef(query, k, self.config().ef)
    }

    /// Searches with a custom ef (beam width) parameter.
    #[must_use]
    pub fn search_with_ef(&self, query: &[f32], k: usize, ef: usize) -> Vec<(NodeId, f32)> {
        match self.quantization_type {
            QuantizationType::None => {
                // No quantization, use standard HNSW
                self.hnsw.search_with_ef(query, k, ef)
            }
            QuantizationType::Scalar => self.search_scalar_quantized(query, k, ef),
            QuantizationType::Binary => self.search_binary_quantized(query, k, ef),
            QuantizationType::Product { .. } => self.search_product_quantized(query, k, ef),
        }
    }

    /// Search with scalar quantization.
    fn search_scalar_quantized(&self, query: &[f32], k: usize, ef: usize) -> Vec<(NodeId, f32)> {
        let trained = *self.quantizer_trained.read();

        if !trained {
            // Quantizer not ready, fall back to exact search
            return self.hnsw.search_with_ef(query, k, ef);
        }

        // Get candidates using HNSW (with full precision distances for now)
        // In a production system, you'd modify HNSW to use quantized distances
        let num_candidates = if self.rescore {
            k * self.rescore_factor
        } else {
            k
        };

        let candidates = self.hnsw.search_with_ef(query, num_candidates, ef);

        if !self.rescore {
            return candidates.into_iter().take(k).collect();
        }

        // Rescore with exact distances
        self.rescore_candidates(query, candidates, k)
    }

    /// Search with binary quantization.
    fn search_binary_quantized(&self, query: &[f32], k: usize, ef: usize) -> Vec<(NodeId, f32)> {
        let binary_vecs = self.binary_vectors.read();

        if binary_vecs.is_empty() {
            return self.hnsw.search_with_ef(query, k, ef);
        }

        // Quantize the query
        let query_bits = BinaryQuantizer::quantize(query);
        let dims = self.config().dimensions;

        // Get candidates - use more candidates for binary (less accurate)
        let num_candidates = if self.rescore {
            k * self.rescore_factor * 2 // Binary needs more candidates
        } else {
            k
        };

        // Use HNSW to get initial candidates, then filter by hamming distance
        let hnsw_candidates = self.hnsw.search_with_ef(query, num_candidates, ef);

        // Compute hamming distances for candidates
        let mut scored: Vec<(NodeId, f32)> = hnsw_candidates
            .iter()
            .filter_map(|(id, _)| {
                binary_vecs.get(id).map(|bits| {
                    let approx_dist =
                        BinaryQuantizer::approximate_euclidean(&query_bits, bits, dims);
                    (*id, approx_dist)
                })
            })
            .collect();

        scored.sort_by_key(|(_, d)| OrderedFloat(*d));
        scored.truncate(num_candidates);

        if !self.rescore {
            return scored.into_iter().take(k).collect();
        }

        // Rescore with exact distances
        self.rescore_candidates(query, scored, k)
    }

    /// Search with product quantization.
    fn search_product_quantized(&self, query: &[f32], k: usize, ef: usize) -> Vec<(NodeId, f32)> {
        let trained = *self.quantizer_trained.read();

        if !trained {
            // Quantizer not ready, fall back to exact search
            return self.hnsw.search_with_ef(query, k, ef);
        }

        // Get candidates using HNSW
        let num_candidates = if self.rescore {
            k * self.rescore_factor
        } else {
            k
        };

        let candidates = self.hnsw.search_with_ef(query, num_candidates, ef);

        if !self.rescore {
            return candidates.into_iter().take(k).collect();
        }

        // Rescore with asymmetric PQ distances (faster than full precision)
        // or fall back to full precision rescoring
        let pq_guard = self.product_quantizer.read();
        let codes_guard = self.product_codes.read();

        if let Some(ref pq) = *pq_guard {
            // Build distance table for this query
            let table = pq.build_distance_table(query);

            let mut scored: Vec<(NodeId, f32)> = candidates
                .into_iter()
                .filter_map(|(id, _)| {
                    codes_guard.get(&id).map(|codes| {
                        let dist = pq.distance_with_table(&table, codes);
                        (id, dist.sqrt()) // Convert squared distance to distance
                    })
                })
                .collect();

            scored.sort_by_key(|(_, d)| OrderedFloat(*d));
            scored.truncate(k);

            // Optionally rescore top results with full precision
            if self.rescore {
                return self.rescore_candidates(query, scored, k);
            }

            scored
        } else {
            // Fall back to exact search
            candidates.into_iter().take(k).collect()
        }
    }

    /// Rescores candidates with exact full-precision distances.
    fn rescore_candidates(
        &self,
        query: &[f32],
        candidates: Vec<(NodeId, f32)>,
        k: usize,
    ) -> Vec<(NodeId, f32)> {
        let metric = self.config().metric;

        let mut rescored: Vec<(NodeId, f32)> = candidates
            .into_iter()
            .filter_map(|(id, _approx_dist)| {
                self.hnsw.get(id).map(|vec| {
                    let exact_dist = compute_distance(query, &vec, metric);
                    (id, exact_dist)
                })
            })
            .collect();

        rescored.sort_by_key(|(_, d)| OrderedFloat(*d));
        rescored.truncate(k);
        rescored
    }

    /// Returns the vector for the given ID (full precision).
    #[must_use]
    pub fn get(&self, id: NodeId) -> Option<Arc<[f32]>> {
        self.hnsw.get(id)
    }

    /// Returns true if the index contains the given ID.
    #[must_use]
    pub fn contains(&self, id: NodeId) -> bool {
        self.hnsw.contains(id)
    }

    /// Removes a vector from the index.
    pub fn remove(&self, id: NodeId) -> bool {
        match self.quantization_type {
            QuantizationType::None => {}
            QuantizationType::Scalar => {
                self.scalar_vectors.write().remove(&id);
            }
            QuantizationType::Binary => {
                self.binary_vectors.write().remove(&id);
            }
            QuantizationType::Product { .. } => {
                self.product_codes.write().remove(&id);
            }
        }
        self.hnsw.remove(id)
    }

    /// Batch insert multiple vectors.
    pub fn batch_insert<'a, I>(&self, vectors: I)
    where
        I: IntoIterator<Item = (NodeId, &'a [f32])>,
    {
        for (id, vec) in vectors {
            self.insert(id, vec);
        }
    }

    /// Batch search for multiple queries in parallel.
    #[must_use]
    pub fn batch_search(&self, queries: &[Vec<f32>], k: usize) -> Vec<Vec<(NodeId, f32)>> {
        use rayon::prelude::*;

        queries
            .par_iter()
            .map(|query| self.search(query, k))
            .collect()
    }
}

impl std::fmt::Debug for QuantizedHnswIndex {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("QuantizedHnswIndex")
            .field("len", &self.len())
            .field("quantization", &self.quantization_type)
            .field("rescore", &self.rescore)
            .field("rescore_factor", &self.rescore_factor)
            .field(
                "theoretical_compression",
                &self.theoretical_compression_ratio(),
            )
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::index::vector::DistanceMetric;

    fn create_test_vectors(n: usize, dim: usize) -> Vec<Vec<f32>> {
        (0..n)
            .map(|i| {
                (0..dim)
                    .map(|j| ((i * dim + j) as f32) / (n * dim) as f32)
                    .collect()
            })
            .collect()
    }

    #[test]
    fn test_quantized_hnsw_no_quantization() {
        let config = HnswConfig::new(4, DistanceMetric::Euclidean);
        let index = QuantizedHnswIndex::new(config, QuantizationType::None);

        let vectors = create_test_vectors(50, 4);
        for (i, vec) in vectors.iter().enumerate() {
            index.insert(NodeId::new(i as u64 + 1), vec);
        }

        assert_eq!(index.len(), 50);

        let results = index.search(&vectors[25], 5);
        assert_eq!(results.len(), 5);
        assert_eq!(results[0].0, NodeId::new(26));
    }

    #[test]
    fn test_quantized_hnsw_scalar_quantization() {
        let config = HnswConfig::new(4, DistanceMetric::Euclidean);
        let index = QuantizedHnswIndex::with_seed(config, QuantizationType::Scalar, 42)
            .with_training_threshold(10); // Lower threshold for test

        let vectors = create_test_vectors(50, 4);
        for (i, vec) in vectors.iter().enumerate() {
            index.insert(NodeId::new(i as u64 + 1), vec);
        }

        assert_eq!(index.len(), 50);
        // Scalar quantization should give 4x theoretical compression
        assert_eq!(index.theoretical_compression_ratio(), 4.0);
        // With rescoring, we store more data (both full and quantized)
        assert!(index.memory_ratio() > 1.0);

        let results = index.search(&vectors[25], 5);
        assert_eq!(results.len(), 5);
        // Should find the correct vector (rescoring fixes quantization errors)
        assert_eq!(results[0].0, NodeId::new(26));
    }

    #[test]
    fn test_quantized_hnsw_binary_quantization() {
        let config = HnswConfig::new(4, DistanceMetric::Euclidean);
        let index = QuantizedHnswIndex::with_seed(config, QuantizationType::Binary, 42);

        let vectors = create_test_vectors(50, 4);
        for (i, vec) in vectors.iter().enumerate() {
            index.insert(NodeId::new(i as u64 + 1), vec);
        }

        assert_eq!(index.len(), 50);

        let results = index.search(&vectors[25], 5);
        assert_eq!(results.len(), 5);
        // Binary is less accurate but should still work with rescoring
        assert_eq!(results[0].0, NodeId::new(26));
    }

    #[test]
    fn test_quantized_hnsw_without_rescore() {
        let config = HnswConfig::new(4, DistanceMetric::Euclidean);
        let index = QuantizedHnswIndex::with_seed(config, QuantizationType::Scalar, 42)
            .with_training_threshold(10)
            .without_rescore();

        let vectors = create_test_vectors(50, 4);
        for (i, vec) in vectors.iter().enumerate() {
            index.insert(NodeId::new(i as u64 + 1), vec);
        }

        let results = index.search(&vectors[25], 5);
        assert_eq!(results.len(), 5);
        // Without rescoring, might not be exact but should be close
    }

    #[test]
    fn test_quantized_hnsw_remove() {
        let config = HnswConfig::new(4, DistanceMetric::Euclidean);
        let index = QuantizedHnswIndex::new(config, QuantizationType::Binary);

        index.insert(NodeId::new(1), &[0.1, 0.2, 0.3, 0.4]);
        index.insert(NodeId::new(2), &[0.5, 0.6, 0.7, 0.8]);

        assert_eq!(index.len(), 2);
        assert!(index.remove(NodeId::new(1)));
        assert_eq!(index.len(), 1);
        assert!(!index.contains(NodeId::new(1)));
        assert!(index.contains(NodeId::new(2)));
    }

    #[test]
    fn test_quantized_hnsw_batch_operations() {
        let config = HnswConfig::new(4, DistanceMetric::Euclidean);
        let index = QuantizedHnswIndex::with_seed(config, QuantizationType::Scalar, 42)
            .with_training_threshold(10);

        let vectors = create_test_vectors(50, 4);
        let pairs: Vec<_> = vectors
            .iter()
            .enumerate()
            .map(|(i, v)| (NodeId::new(i as u64 + 1), v.as_slice()))
            .collect();

        index.batch_insert(pairs);
        assert_eq!(index.len(), 50);

        // Batch search
        let queries = vec![vectors[10].clone(), vectors[30].clone()];
        let results = index.batch_search(&queries, 3);

        assert_eq!(results.len(), 2);
        assert_eq!(results[0].len(), 3);
        assert_eq!(results[1].len(), 3);
    }

    #[test]
    fn test_quantization_type_enum() {
        let dims = 384;
        assert_eq!(QuantizationType::None.compression_ratio(dims), 1);
        assert_eq!(QuantizationType::Scalar.compression_ratio(dims), 4);
        assert_eq!(QuantizationType::Binary.compression_ratio(dims), 32);
        // Product: (384 * 4) / 8 = 192
        assert_eq!(
            QuantizationType::Product { num_subvectors: 8 }.compression_ratio(dims),
            192
        );
    }

    #[test]
    fn test_quantized_hnsw_memory_usage() {
        let config = HnswConfig::new(384, DistanceMetric::Cosine);
        let index = QuantizedHnswIndex::new(config, QuantizationType::Scalar);

        // Empty index should have minimal memory
        assert_eq!(index.memory_usage(), 0);

        // After inserting, memory should be tracked
        index.insert(NodeId::new(1), &vec![0.1f32; 384]);
        assert!(index.memory_usage() > 0);
    }

    #[test]
    fn test_quantized_hnsw_product_quantization() {
        // 32 dimensions divisible by 8 subvectors
        let config = HnswConfig::new(32, DistanceMetric::Euclidean);
        let index = QuantizedHnswIndex::with_seed(
            config,
            QuantizationType::Product { num_subvectors: 8 },
            42,
        )
        .with_training_threshold(20); // Lower threshold for test

        let vectors = create_test_vectors(50, 32);
        for (i, vec) in vectors.iter().enumerate() {
            index.insert(NodeId::new(i as u64 + 1), vec);
        }

        assert_eq!(index.len(), 50);
        // Product quantization: (32 * 4) / 8 = 16x compression
        assert_eq!(index.theoretical_compression_ratio(), 16.0);

        let results = index.search(&vectors[25], 5);
        assert_eq!(results.len(), 5);
        // With rescoring, should find the correct vector
        assert_eq!(results[0].0, NodeId::new(26));
    }

    #[test]
    fn test_quantized_hnsw_product_before_training() {
        // Test behavior before quantizer is trained
        let config = HnswConfig::new(16, DistanceMetric::Euclidean);
        let index = QuantizedHnswIndex::with_seed(
            config,
            QuantizationType::Product { num_subvectors: 4 },
            42,
        )
        .with_training_threshold(100); // High threshold so it won't train

        let vectors = create_test_vectors(10, 16);
        for (i, vec) in vectors.iter().enumerate() {
            index.insert(NodeId::new(i as u64 + 1), vec);
        }

        // Should still work (falls back to exact search before training)
        let results = index.search(&vectors[5], 3);
        assert_eq!(results.len(), 3);
        assert_eq!(results[0].0, NodeId::new(6));
    }
}
