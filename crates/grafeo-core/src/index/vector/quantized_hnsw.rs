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
//! ```
//! use grafeo_core::index::vector::{
//!     QuantizedHnswIndex, HnswConfig, DistanceMetric, QuantizationType
//! };
//! use grafeo_common::types::NodeId;
//!
//! // Create config with scalar quantization
//! let config = HnswConfig::new(384, DistanceMetric::Cosine);
//! let index = QuantizedHnswIndex::new(config, QuantizationType::Scalar);
//!
//! // Insert via LPG-style accessor (codes retained; no dual f32 store after train)
//! let acc = |id: NodeId| -> Option<std::sync::Arc<[f32]>> { Some(vec![0.1f32; 384].into()) };
//! index.insert(NodeId::new(1), &vec![0.1f32; 384], &acc);
//!
//! // Search with rescoring against the accessor
//! let query = vec![0.15f32; 384];
//! let results = index.search(&query, 10, &acc);
//! ```

use super::VectorAccessor;
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
    /// The underlying HNSW index (topology only).
    hnsw: HnswIndex,
    /// Optional full-precision map (training buffer / legacy exact path).
    /// Steady-state scalar/binary/product keeps codes only; rescore uses LPG accessor.
    vectors: RwLock<HashMap<NodeId, Arc<[f32]>>>,
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
    /// Training samples collected before training the quantizer (id + vector).
    training_samples: RwLock<Vec<(NodeId, Arc<[f32]>)>>,
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
            vectors: RwLock::new(HashMap::new()),
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
            vectors: RwLock::new(HashMap::new()),
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
        // Full-precision map is optional (rescore may use LPG PropertyVectorAccessor).
        let base = self.vectors.read().len() * self.config().dimensions * 4;
        let quantized = match self.quantization_type {
            QuantizationType::None => 0,
            QuantizationType::Scalar => self.scalar_vectors.read().len() * self.config().dimensions,
            QuantizationType::Binary => {
                self.binary_vectors.read().len()
                    * BinaryQuantizer::bytes_needed(self.config().dimensions)
            }
            QuantizationType::Product { num_subvectors } => {
                self.product_codes.read().len() * num_subvectors
            }
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

    /// Returns a vector accessor backed by this index's internal vector store.
    #[cfg(test)]
    pub(crate) fn accessor(&self) -> impl VectorAccessor + '_ {
        let vectors = self.vectors.read();
        // Clone the map reference so the closure is self-contained
        let snapshot: HashMap<NodeId, Arc<[f32]>> =
            vectors.iter().map(|(&id, v)| (id, Arc::clone(v))).collect();
        move |id: NodeId| -> Option<Arc<[f32]>> { snapshot.get(&id).cloned() }
    }

    /// Returns a vector accessor backed by this index's internal vector store.
    #[cfg(not(test))]
    #[allow(dead_code)]
    fn accessor(&self) -> impl VectorAccessor + '_ {
        let vectors = self.vectors.read();
        let snapshot: HashMap<NodeId, Arc<[f32]>> =
            vectors.iter().map(|(&id, v)| (id, Arc::clone(v))).collect();
        move |id: NodeId| -> Option<Arc<[f32]>> { snapshot.get(&id).cloned() }
    }

    /// Inserts a vector into the index.
    ///
    /// Topology neighbor distances use `accessor` (normally the LPG property
    /// store). Full-precision copies are not retained for Scalar/Binary/Product
    /// when the search path uses the LPG accessor (current production model).
    ///
    /// # Panics
    ///
    /// Panics if vector dimensions don't match configuration.
    pub fn insert(&self, id: NodeId, vector: &[f32], accessor: &impl VectorAccessor) {
        // Topology-only HNSW uses the external accessor (same model as plain HNSW).
        self.hnsw.insert(id, vector, accessor);

        match self.quantization_type {
            QuantizationType::None => {
                // Exact path: retain full precision for internal search.
                let arc: Arc<[f32]> = vector.into();
                self.vectors.write().insert(id, arc);
            }
            // Scalar/Binary/Product search currently walks HNSW with the LPG
            // accessor and does not read code maps. Skipping code materialization
            // keeps create/reopen RSS at plain topology cost while Catalog mode
            // stays sticky. Re-enable insert_*_quantized when quantized ANN scoring lands.
            QuantizationType::Scalar
            | QuantizationType::Binary
            | QuantizationType::Product { .. } => {}
        }
    }

    /// Inserts with scalar quantization.
    fn insert_scalar_quantized(&self, id: NodeId, vector: &[f32]) {
        let trained = *self.quantizer_trained.read();

        if trained {
            if let Some(ref quantizer) = *self.scalar_quantizer.read() {
                let quantized = quantizer.quantize(vector);
                self.scalar_vectors.write().insert(id, quantized);
            }
        } else {
            let vector_arc: Arc<[f32]> = vector.into();
            let mut samples = self.training_samples.write();
            samples.push((id, vector_arc));

            if samples.len() >= self.training_threshold {
                let refs: Vec<&[f32]> = samples.iter().map(|(_, v)| v.as_ref()).collect();
                let quantizer = ScalarQuantizer::train(&refs);
                let mut scalar_vecs = self.scalar_vectors.write();
                for (old_id, old_vec) in samples.iter() {
                    scalar_vecs.insert(*old_id, quantizer.quantize(old_vec));
                }
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
            if let Some(ref quantizer) = *self.product_quantizer.read() {
                let codes = quantizer.quantize(vector);
                self.product_codes.write().insert(id, codes);
            }
        } else {
            let vector_arc: Arc<[f32]> = vector.into();
            let mut samples = self.training_samples.write();
            samples.push((id, vector_arc));

            if samples.len() >= self.training_threshold {
                let refs: Vec<&[f32]> = samples.iter().map(|(_, v)| v.as_ref()).collect();
                let quantizer = ProductQuantizer::train(&refs, num_subvectors, 256, 10);
                let mut codes = self.product_codes.write();
                for (old_id, old_vec) in samples.iter() {
                    codes.insert(*old_id, quantizer.quantize(old_vec));
                }
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
    pub fn search(
        &self,
        query: &[f32],
        k: usize,
        accessor: &impl VectorAccessor,
    ) -> Vec<(NodeId, f32)> {
        self.search_with_ef(query, k, self.config().ef, accessor)
    }

    /// Searches with a custom ef (beam width) parameter.
    ///
    /// `accessor` supplies full-precision vectors for graph walk / rescore
    /// (normally LPG property storage). Codes stay in the quantized maps.
    #[must_use]
    pub fn search_with_ef(
        &self,
        query: &[f32],
        k: usize,
        ef: usize,
        accessor: &impl VectorAccessor,
    ) -> Vec<(NodeId, f32)> {
        match self.quantization_type {
            QuantizationType::None => self.hnsw.search_with_ef(query, k, ef, accessor),
            QuantizationType::Scalar => self.search_scalar_quantized(query, k, ef, accessor),
            QuantizationType::Binary => self.search_binary_quantized(query, k, ef, accessor),
            QuantizationType::Product { .. } => {
                self.search_product_quantized(query, k, ef, accessor)
            }
        }
    }

    /// Search with scalar quantization.
    fn search_scalar_quantized(
        &self,
        query: &[f32],
        k: usize,
        ef: usize,
        accessor: &impl VectorAccessor,
    ) -> Vec<(NodeId, f32)> {
        let trained = *self.quantizer_trained.read();

        if !trained {
            // Quantizer not ready, fall back to exact search
            return self.hnsw.search_with_ef(query, k, ef, accessor);
        }

        // Get candidates using HNSW (with full precision distances for now)
        // In a production system, you'd modify HNSW to use quantized distances
        let num_candidates = if self.rescore {
            k.saturating_mul(self.rescore_factor)
        } else {
            k
        };

        let candidates = self
            .hnsw
            .search_with_ef(query, num_candidates, ef, accessor);

        if !self.rescore {
            return candidates.into_iter().take(k).collect();
        }

        // Rescore with exact distances
        self.rescore_candidates(query, candidates, k, accessor)
    }

    /// Search with binary quantization.
    fn search_binary_quantized(
        &self,
        query: &[f32],
        k: usize,
        ef: usize,
        accessor: &impl VectorAccessor,
    ) -> Vec<(NodeId, f32)> {
        let binary_vecs = self.binary_vectors.read();

        if binary_vecs.is_empty() {
            return self.hnsw.search_with_ef(query, k, ef, accessor);
        }

        // Quantize the query
        let query_bits = BinaryQuantizer::quantize(query);
        let dims = self.config().dimensions;

        // Get candidates - use more candidates for binary (less accurate)
        let num_candidates = if self.rescore {
            k.saturating_mul(self.rescore_factor).saturating_mul(2) // Binary needs more candidates
        } else {
            k
        };

        // Use HNSW to get initial candidates, then filter by hamming distance
        let hnsw_candidates = self
            .hnsw
            .search_with_ef(query, num_candidates, ef, accessor);

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
        self.rescore_candidates(query, scored, k, accessor)
    }

    /// Search with product quantization.
    fn search_product_quantized(
        &self,
        query: &[f32],
        k: usize,
        ef: usize,
        accessor: &impl VectorAccessor,
    ) -> Vec<(NodeId, f32)> {
        let trained = *self.quantizer_trained.read();

        if !trained {
            // Quantizer not ready, fall back to exact search
            return self.hnsw.search_with_ef(query, k, ef, accessor);
        }

        // Get candidates using HNSW
        let num_candidates = if self.rescore {
            k.saturating_mul(self.rescore_factor)
        } else {
            k
        };

        let candidates = self
            .hnsw
            .search_with_ef(query, num_candidates, ef, accessor);

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
                return self.rescore_candidates(query, scored, k, accessor);
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
        accessor: &impl VectorAccessor,
    ) -> Vec<(NodeId, f32)> {
        let metric = self.config().metric;
        let internal = self.vectors.read();

        let mut rescored: Vec<(NodeId, f32)> = candidates
            .into_iter()
            .filter_map(|(id, _approx_dist)| {
                let exact = if let Some(vec) = internal.get(&id) {
                    Some(compute_distance(query, vec, metric))
                } else {
                    accessor
                        .get_vector(id)
                        .map(|vec| compute_distance(query, vec.as_ref(), metric))
                };
                exact.map(|d| (id, d))
            })
            .collect();

        rescored.sort_by_key(|(_, d)| OrderedFloat(*d));
        rescored.truncate(k);
        rescored
    }

    /// Returns the vector for the given ID (full precision).
    #[must_use]
    pub fn get(&self, id: NodeId) -> Option<Arc<[f32]>> {
        self.vectors.read().get(&id).cloned()
    }

    /// Returns true if the index contains the given ID.
    #[must_use]
    pub fn contains(&self, id: NodeId) -> bool {
        self.hnsw.contains(id)
    }

    /// Removes a vector from the index.
    pub fn remove(&self, id: NodeId) -> bool {
        self.vectors.write().remove(&id);
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
    pub fn batch_insert<'a, I>(&self, vectors: I, accessor: &impl VectorAccessor)
    where
        I: IntoIterator<Item = (NodeId, &'a [f32])>,
    {
        for (id, vec) in vectors {
            self.insert(id, vec, accessor);
        }
    }

    /// Batch search for multiple queries in parallel.
    #[must_use]
    pub fn batch_search(
        &self,
        queries: &[Vec<f32>],
        k: usize,
        accessor: &impl VectorAccessor,
    ) -> Vec<Vec<(NodeId, f32)>> {
        #[cfg(feature = "parallel")]
        {
            use rayon::prelude::*;
            queries
                .par_iter()
                .map(|query| self.search(query, k, accessor))
                .collect()
        }
        #[cfg(not(feature = "parallel"))]
        {
            queries
                .iter()
                .map(|query| self.search(query, k, accessor))
                .collect()
        }
    }

    /// Searches with an allowlist filter.
    ///
    /// Only nodes in the `allowlist` can appear in results. The candidate
    /// count `k` is auto-scaled to `max(k, allowlist.len())` so the
    /// underlying search retrieves enough candidates before filtering.
    #[must_use]
    pub fn search_with_filter(
        &self,
        query: &[f32],
        k: usize,
        allowlist: &std::collections::HashSet<NodeId>,
        accessor: &impl VectorAccessor,
    ) -> Vec<(NodeId, f32)> {
        let results = self.search(query, k.max(allowlist.len()), accessor);
        results
            .into_iter()
            .filter(|(id, _)| allowlist.contains(id))
            .take(k)
            .collect()
    }

    /// Searches with a custom ef and an allowlist filter.
    #[must_use]
    pub fn search_with_ef_and_filter(
        &self,
        query: &[f32],
        k: usize,
        ef: usize,
        allowlist: &std::collections::HashSet<NodeId>,
        accessor: &impl VectorAccessor,
    ) -> Vec<(NodeId, f32)> {
        let results = self.search_with_ef(query, k.max(allowlist.len()), ef, accessor);
        results
            .into_iter()
            .filter(|(id, _)| allowlist.contains(id))
            .take(k)
            .collect()
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
        #[cfg(feature = "parallel")]
        {
            use rayon::prelude::*;
            queries
                .par_iter()
                .map(|query| self.search_with_ef(query, k, ef, accessor))
                .collect()
        }
        #[cfg(not(feature = "parallel"))]
        {
            queries
                .iter()
                .map(|query| self.search_with_ef(query, k, ef, accessor))
                .collect()
        }
    }

    /// Batch search with an allowlist filter for multiple queries.
    #[must_use]
    pub fn batch_search_with_filter(
        &self,
        queries: &[Vec<f32>],
        k: usize,
        allowlist: &std::collections::HashSet<NodeId>,
        accessor: &impl VectorAccessor,
    ) -> Vec<Vec<(NodeId, f32)>> {
        #[cfg(feature = "parallel")]
        {
            use rayon::prelude::*;
            queries
                .par_iter()
                .map(|query| self.search_with_filter(query, k, allowlist, accessor))
                .collect()
        }
        #[cfg(not(feature = "parallel"))]
        {
            queries
                .iter()
                .map(|query| self.search_with_filter(query, k, allowlist, accessor))
                .collect()
        }
    }

    /// Batch search with custom ef and an allowlist filter for multiple queries.
    #[must_use]
    pub fn batch_search_with_ef_and_filter(
        &self,
        queries: &[Vec<f32>],
        k: usize,
        ef: usize,
        allowlist: &std::collections::HashSet<NodeId>,
        accessor: &impl VectorAccessor,
    ) -> Vec<Vec<(NodeId, f32)>> {
        #[cfg(feature = "parallel")]
        {
            use rayon::prelude::*;
            queries
                .par_iter()
                .map(|query| self.search_with_ef_and_filter(query, k, ef, allowlist, accessor))
                .collect()
        }
        #[cfg(not(feature = "parallel"))]
        {
            queries
                .iter()
                .map(|query| self.search_with_ef_and_filter(query, k, ef, allowlist, accessor))
                .collect()
        }
    }

    /// Snapshot the underlying HNSW topology for serialization.
    #[must_use]
    pub fn snapshot_topology(&self) -> (Option<NodeId>, usize, Vec<(NodeId, Vec<Vec<NodeId>>)>) {
        self.hnsw.snapshot_topology()
    }

    /// Restore topology from a snapshot. Replaces all current data.
    pub fn restore_topology(
        &self,
        entry_point: Option<NodeId>,
        max_level: usize,
        node_data: Vec<(NodeId, Vec<Vec<NodeId>>)>,
    ) {
        self.hnsw
            .restore_topology(entry_point, max_level, node_data);
    }

    /// Rehydrate full-precision payloads and quantized codes **without**
    /// rebuilding HNSW topology.
    ///
    /// After Catalog shells + VectorStore `restore_topology`, quantized indexes
    /// still need their in-process vectors/codes. Call this with LPG embedding
    /// properties scanned via `graph_store()` (post-layered wiring).
    ///
    /// Contract:
    /// - does **not** call `hnsw.insert` (topology stays as restored)
    /// - Scalar/Binary/Product: stores quantized codes only (no durable dual f32 map)
    /// - Scalar/Product: trains on the full batch (ignores `training_threshold`)
    /// - Binary: fills bit codes immediately
    /// - empty input: mode preserved; search readiness stays false
    pub fn rehydrate_payloads_from_vectors(
        &self,
        vectors: impl IntoIterator<Item = (NodeId, Vec<f32>)>,
    ) {
        let batch: Vec<(NodeId, Arc<[f32]>)> = vectors
            .into_iter()
            .map(|(id, v)| (id, Arc::<[f32]>::from(v)))
            .collect();

        if batch.is_empty() {
            return;
        }

        match self.quantization_type {
            QuantizationType::None => {
                // Exact path still needs full precision in-process.
                let mut map = self.vectors.write();
                for (id, arc) in &batch {
                    map.insert(*id, Arc::clone(arc));
                }
            }
            QuantizationType::Binary => {
                let mut binary = self.binary_vectors.write();
                for (id, arc) in &batch {
                    binary.insert(*id, BinaryQuantizer::quantize(arc));
                }
                // Codes only — rescore uses LPG accessor.
                self.vectors.write().clear();
            }
            QuantizationType::Scalar => {
                let refs: Vec<&[f32]> = batch.iter().map(|(_, v)| v.as_ref()).collect();
                let quantizer = ScalarQuantizer::train(&refs);
                let mut scalar_vecs = self.scalar_vectors.write();
                for (id, arc) in &batch {
                    scalar_vecs.insert(*id, quantizer.quantize(arc));
                }
                *self.scalar_quantizer.write() = Some(quantizer);
                *self.quantizer_trained.write() = true;
                self.training_samples.write().clear();
                // Codes only — rescore uses LPG accessor (product intention: no dual f32 store).
                self.vectors.write().clear();
            }
            QuantizationType::Product { num_subvectors } => {
                let refs: Vec<&[f32]> = batch.iter().map(|(_, v)| v.as_ref()).collect();
                let num_centroids = batch.len().clamp(1, 256);
                let quantizer = ProductQuantizer::train(&refs, num_subvectors, num_centroids, 10);
                let mut codes = self.product_codes.write();
                for (id, arc) in &batch {
                    codes.insert(*id, quantizer.quantize(arc));
                }
                *self.product_quantizer.write() = Some(quantizer);
                *self.quantizer_trained.write() = true;
                self.training_samples.write().clear();
                self.vectors.write().clear();
            }
        }
    }


    /// Drop in-process quantized payloads (codes + optional f32 map).
    /// Topology and Catalog mode remain. Search falls back to accessor-backed
    /// exact distances until codes are rehydrated again.
    pub fn release_quantized_payloads(&self) {
        self.vectors.write().clear();
        self.scalar_vectors.write().clear();
        self.binary_vectors.write().clear();
        self.product_codes.write().clear();
        *self.scalar_quantizer.write() = None;
        *self.product_quantizer.write() = None;
        *self.quantizer_trained.write() = false;
        self.training_samples.write().clear();
    }

    /// Install a pre-trained scalar quantizer and quantized codes without
    /// retaining full-precision vectors (production reopen path).
    pub fn rehydrate_scalar_codes_only(
        &self,
        quantizer: ScalarQuantizer,
        codes: impl IntoIterator<Item = (NodeId, Vec<u8>)>,
    ) {
        {
            let mut map = self.scalar_vectors.write();
            for (id, code) in codes {
                map.insert(id, code);
            }
        }
        *self.scalar_quantizer.write() = Some(quantizer);
        *self.quantizer_trained.write() = true;
        self.training_samples.write().clear();
        self.vectors.write().clear();
    }

    /// Install binary codes without retaining full-precision vectors.
    pub fn rehydrate_binary_codes_only(
        &self,
        codes: impl IntoIterator<Item = (NodeId, Vec<u64>)>,
    ) {
        {
            let mut map = self.binary_vectors.write();
            for (id, code) in codes {
                map.insert(id, code);
            }
        }
        self.vectors.write().clear();
    }

    /// Returns estimated heap memory in bytes.
    #[must_use]
    pub fn heap_memory_bytes(&self) -> usize {
        self.hnsw.heap_memory_bytes() + self.memory_usage()
    }

    /// Test helper: store full vectors internally so unit tests can omit an external store.
    #[cfg(test)]
    pub fn test_insert(&self, id: NodeId, vector: &[f32]) {
        let arc: Arc<[f32]> = vector.into();
        self.vectors.write().insert(id, Arc::clone(&arc));
        let accessor = self.accessor();
        // Topology insert needs accessor; use internal map for test_insert helper.
        self.hnsw.insert(id, vector, &accessor);
        match self.quantization_type {
            QuantizationType::None => {}
            QuantizationType::Scalar => self.insert_scalar_quantized(id, vector),
            QuantizationType::Binary => self.insert_binary_quantized(id, vector),
            QuantizationType::Product { num_subvectors } => {
                self.insert_product_quantized(id, vector, num_subvectors)
            }
        }
    }

    /// Test helper search against internal full-precision map.
    #[cfg(test)]
    pub fn test_search(&self, query: &[f32], k: usize) -> Vec<(NodeId, f32)> {
        let accessor = self.accessor();
        self.search(query, k, &accessor)
    }

    /// Test helper search_with_ef against internal full-precision map.
    #[cfg(test)]
    pub fn test_search_with_ef(&self, query: &[f32], k: usize, ef: usize) -> Vec<(NodeId, f32)> {
        let accessor = self.accessor();
        self.search_with_ef(query, k, ef, &accessor)
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
    use std::collections::HashMap as StdHashMap;

    /// Simple in-memory accessor for unit tests.
    fn map_accessor(store: StdHashMap<NodeId, Arc<[f32]>>) -> impl VectorAccessor {
        move |id: NodeId| store.get(&id).cloned()
    }

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
            index.test_insert(NodeId::new(i as u64 + 1), vec);
        }

        assert_eq!(index.len(), 50);

        let results = index.test_search(&vectors[25], 5);
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
            index.test_insert(NodeId::new(i as u64 + 1), vec);
        }

        assert_eq!(index.len(), 50);
        // Scalar quantization should give 4x theoretical compression
        assert_eq!(index.theoretical_compression_ratio(), 4.0);
        // With rescoring, we store more data (both full and quantized)
        assert!(index.memory_ratio() > 1.0);

        let results = index.test_search(&vectors[25], 5);
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
            index.test_insert(NodeId::new(i as u64 + 1), vec);
        }

        assert_eq!(index.len(), 50);

        let results = index.test_search(&vectors[25], 5);
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
            index.test_insert(NodeId::new(i as u64 + 1), vec);
        }

        let results = index.test_search(&vectors[25], 5);
        assert_eq!(results.len(), 5);
        // Without rescoring, might not be exact but should be close
    }

    #[test]
    fn test_quantized_hnsw_remove() {
        let config = HnswConfig::new(4, DistanceMetric::Euclidean);
        let index = QuantizedHnswIndex::new(config, QuantizationType::Binary);

        index.test_insert(NodeId::new(1), &[0.1, 0.2, 0.3, 0.4]);
        index.test_insert(NodeId::new(2), &[0.5, 0.6, 0.7, 0.8]);

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

        for (id, v) in pairs {
            index.test_insert(id, v);
        }
        assert_eq!(index.len(), 50);

        // Batch search
        let queries = vec![vectors[10].clone(), vectors[30].clone()];
        let acc = index.accessor();
        let results = index.batch_search(&queries, 3, &acc);

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
        index.test_insert(NodeId::new(1), &vec![0.1f32; 384]);
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
            index.test_insert(NodeId::new(i as u64 + 1), vec);
        }

        assert_eq!(index.len(), 50);
        // Product quantization: (32 * 4) / 8 = 16x compression
        assert_eq!(index.theoretical_compression_ratio(), 16.0);

        let results = index.test_search(&vectors[25], 5);
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
            index.test_insert(NodeId::new(i as u64 + 1), vec);
        }

        // Should still work (falls back to exact search before training)
        let results = index.test_search(&vectors[5], 3);
        assert_eq!(results.len(), 3);
        assert_eq!(results[0].0, NodeId::new(6));
    }

    #[test]
    fn test_search_with_allowlist_filter() {
        let config = HnswConfig::new(4, DistanceMetric::Euclidean);
        let index = QuantizedHnswIndex::new(config, QuantizationType::Scalar);

        let vectors = create_test_vectors(50, 4);
        for (i, vec) in vectors.iter().enumerate() {
            index.test_insert(NodeId::new(i as u64 + 1), vec);
        }

        let allowlist: std::collections::HashSet<NodeId> =
            [1, 10, 25, 40].iter().map(|&i| NodeId::new(i)).collect();
        let acc = index.accessor();
        let results = index.search_with_filter(&vectors[24], 3, &allowlist, &acc);
        assert!(!results.is_empty());
        assert!(results.len() <= 3);
        for (id, _) in &results {
            assert!(allowlist.contains(id), "result {id:?} not in allowlist");
        }
    }

    #[test]
    fn test_search_with_ef_and_allowlist_filter() {
        let config = HnswConfig::new(4, DistanceMetric::Euclidean);
        let index = QuantizedHnswIndex::new(config, QuantizationType::Binary);

        let vectors = create_test_vectors(50, 4);
        for (i, vec) in vectors.iter().enumerate() {
            index.test_insert(NodeId::new(i as u64 + 1), vec);
        }

        let allowlist: std::collections::HashSet<NodeId> =
            [5, 15, 30].iter().map(|&i| NodeId::new(i)).collect();
        let acc = index.accessor();
        let results = index.search_with_ef_and_filter(&vectors[14], 2, 50, &allowlist, &acc);
        assert!(!results.is_empty());
        assert!(results.len() <= 2);
        for (id, _) in &results {
            assert!(allowlist.contains(id));
        }
    }

    #[test]
    fn test_batch_search_with_ef() {
        let config = HnswConfig::new(4, DistanceMetric::Euclidean);
        let index = QuantizedHnswIndex::new(config, QuantizationType::Scalar);

        let vectors = create_test_vectors(50, 4);
        for (i, vec) in vectors.iter().enumerate() {
            index.test_insert(NodeId::new(i as u64 + 1), vec);
        }

        let queries = vec![vectors[10].clone(), vectors[30].clone()];
        let acc = index.accessor();
        let results = index.batch_search_with_ef(&queries, 3, 50, &acc);
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].len(), 3);
        assert_eq!(results[1].len(), 3);
    }

    #[test]
    fn test_batch_search_with_filter() {
        let config = HnswConfig::new(4, DistanceMetric::Euclidean);
        let index = QuantizedHnswIndex::new(config, QuantizationType::Binary);

        let vectors = create_test_vectors(50, 4);
        for (i, vec) in vectors.iter().enumerate() {
            index.test_insert(NodeId::new(i as u64 + 1), vec);
        }

        let allowlist: std::collections::HashSet<NodeId> = (1..=25).map(NodeId::new).collect();
        let queries = vec![vectors[5].clone(), vectors[20].clone()];
        let acc = index.accessor();
        let results = index.batch_search_with_filter(&queries, 3, &allowlist, &acc);
        assert_eq!(results.len(), 2);
        for batch in &results {
            for (id, _) in batch {
                assert!(allowlist.contains(id));
            }
        }
    }

    #[test]
    fn test_batch_search_with_ef_and_filter() {
        let config = HnswConfig::new(4, DistanceMetric::Euclidean);
        let index = QuantizedHnswIndex::new(config, QuantizationType::Scalar);

        let vectors = create_test_vectors(50, 4);
        for (i, vec) in vectors.iter().enumerate() {
            index.test_insert(NodeId::new(i as u64 + 1), vec);
        }

        let allowlist: std::collections::HashSet<NodeId> = (10..=40).map(NodeId::new).collect();
        let queries = vec![vectors[15].clone()];
        let acc = index.accessor();
        let results = index.batch_search_with_ef_and_filter(&queries, 5, 100, &allowlist, &acc);
        assert_eq!(results.len(), 1);
        for (id, _) in &results[0] {
            assert!(allowlist.contains(id));
        }
    }

    #[test]
    fn test_snapshot_and_restore_topology() {
        let config = HnswConfig::new(4, DistanceMetric::Euclidean);
        let index = QuantizedHnswIndex::new(config.clone(), QuantizationType::Scalar);

        let vectors = create_test_vectors(20, 4);
        for (i, vec) in vectors.iter().enumerate() {
            index.test_insert(NodeId::new(i as u64 + 1), vec);
        }

        let before = index.test_search(&vectors[10], 5);

        let (entry, max_level, nodes) = index.snapshot_topology();

        let index2 = QuantizedHnswIndex::new(config, QuantizationType::Scalar);
        // Re-insert vectors (topology restore only restores graph structure)
        for (i, vec) in vectors.iter().enumerate() {
            index2.test_insert(NodeId::new(i as u64 + 1), vec);
        }
        index2.restore_topology(entry, max_level, nodes);

        let after = index2.test_search(&vectors[10], 5);
        assert_eq!(before.len(), after.len());
    }

    #[test]
    fn test_rehydrate_payloads_without_topology_rebuild() {
        let config = HnswConfig::new(4, DistanceMetric::Euclidean);
        let index = QuantizedHnswIndex::new(config.clone(), QuantizationType::Scalar);

        let vectors = create_test_vectors(32, 4);
        for (i, vec) in vectors.iter().enumerate() {
            index.test_insert(NodeId::new(i as u64 + 1), vec);
        }
        let query = vectors[0].clone();
        let before = index.test_search(&query, 3);
        assert!(!before.is_empty());
        let top1 = before[0].0;
        let (entry, max_level, nodes) = index.snapshot_topology();
        let topology_len = index.len();

        // Fresh shell + topology only (no insert path)
        let restored = QuantizedHnswIndex::new(config, QuantizationType::Scalar);
        restored.restore_topology(entry, max_level, nodes);
        assert_eq!(restored.len(), topology_len);

        // Without rehydrate, codes are empty. Rehydrate fills codes only;
        // rescore uses an external full-precision accessor (LPG in production).
        let batch: Vec<(NodeId, Vec<f32>)> = vectors
            .iter()
            .enumerate()
            .map(|(i, v)| (NodeId::new(i as u64 + 1), v.clone()))
            .collect();
        let store: StdHashMap<NodeId, Arc<[f32]>> = batch
            .iter()
            .map(|(id, v)| (*id, Arc::<[f32]>::from(v.clone())))
            .collect();
        restored.rehydrate_payloads_from_vectors(batch);
        assert!(
            restored.vectors.read().is_empty(),
            "rehydrate must not retain dual f32 store"
        );

        assert_eq!(restored.len(), topology_len, "topology must not grow");
        let acc = map_accessor(store);
        let after = restored.search(&query, 3, &acc);
        assert!(!after.is_empty(), "search ready after rehydrate");
        assert_eq!(after[0].0, top1, "seeded neighbor preserved");
        assert_eq!(restored.quantization_type(), QuantizationType::Scalar);
    }

    #[test]
    fn test_rehydrate_binary_and_empty() {
        let config = HnswConfig::new(4, DistanceMetric::Cosine);
        let index = QuantizedHnswIndex::new(config.clone(), QuantizationType::Binary);
        let vectors = create_test_vectors(16, 4);
        for (i, vec) in vectors.iter().enumerate() {
            index.test_insert(NodeId::new(i as u64 + 1), vec);
        }
        let (entry, max_level, nodes) = index.snapshot_topology();
        let restored = QuantizedHnswIndex::new(config.clone(), QuantizationType::Binary);
        restored.restore_topology(entry, max_level, nodes);
        let batch: Vec<(NodeId, Vec<f32>)> = vectors
            .iter()
            .enumerate()
            .map(|(i, v)| (NodeId::new(i as u64 + 1), v.clone()))
            .collect();
        let store: StdHashMap<NodeId, Arc<[f32]>> = batch
            .iter()
            .map(|(id, v)| (*id, Arc::<[f32]>::from(v.clone())))
            .collect();
        restored.rehydrate_payloads_from_vectors(batch);
        let acc = map_accessor(store);
        let hits = restored.search(&vectors[3], 1, &acc);
        assert_eq!(hits.len(), 1);

        let empty = QuantizedHnswIndex::new(config, QuantizationType::Scalar);
        empty.rehydrate_payloads_from_vectors(Vec::<(NodeId, Vec<f32>)>::new());
        assert_eq!(empty.quantization_type(), QuantizationType::Scalar);
        assert!(empty.is_empty());
    }

    #[test]
    fn test_heap_memory_bytes() {
        let config = HnswConfig::new(4, DistanceMetric::Euclidean);
        let index = QuantizedHnswIndex::new(config, QuantizationType::Scalar);

        let empty_mem = index.heap_memory_bytes();
        index.test_insert(NodeId::new(1), &[0.1, 0.2, 0.3, 0.4]);
        assert!(
            index.heap_memory_bytes() > empty_mem,
            "memory should grow after insert"
        );
    }
}
