//! Quantized HNSW index with rescoring support.
//!
//! This module provides memory-efficient vector search by combining
//! quantization with HNSW approximate nearest neighbor search:
//!
//! 1. **Search phase**: Use quantized vectors for fast approximate search
//! 2. **Rescore phase**: Re-rank top candidates using full-precision vectors
//!
//! The index holds the quantized codes and the topology only: full-precision
//! vectors are read through a [`VectorAccessor`], as [`HnswIndex`] reads
//! them, so each vector is kept once, by whoever owns the data.
//!
//! # Compression vs Accuracy Tradeoff
//!
//! | Quantization | Codes per vector | Recall@10 | Best For |
//! |--------------|------------------|-----------|----------|
//! | None         | none             | ~98%      | Small datasets, max accuracy |
//! | Scalar       | 25% of the vector | ~97%     | Most production use cases |
//! | Binary       | ~3% of the vector | ~85%     | Very large datasets |
//!
//! # Example
//!
//! ```
//! use grafeo_core::index::vector::{
//!     QuantizedHnswIndex, HnswConfig, DistanceMetric, QuantizationType
//! };
//! use grafeo_common::types::NodeId;
//! use std::collections::HashMap;
//! use std::sync::Arc;
//!
//! // Create config with scalar quantization
//! let config = HnswConfig::new(384, DistanceMetric::Cosine);
//! let index = QuantizedHnswIndex::new(config, QuantizationType::Scalar);
//!
//! // The vectors live outside the index; the accessor reads them
//! let mut vectors: HashMap<NodeId, Arc<[f32]>> = HashMap::new();
//! vectors.insert(NodeId::new(1), vec![0.1f32; 384].into());
//! let accessor = |id: NodeId| vectors.get(&id).cloned();
//!
//! index.insert(NodeId::new(1), &vec![0.1f32; 384], &accessor);
//!
//! // Search with rescoring (default: rescore top 2*k candidates)
//! let query = vec![0.15f32; 384];
//! let results = index.search(&query, 10, &accessor);
//! ```

use super::hnsw::{infallible, no_visit};
use super::quantization::{BinaryQuantizer, ProductQuantizer, QuantizationType, ScalarQuantizer};
use super::{BrokenLink, HnswConfig, HnswIndex, compute_distance};
use super::{TopologyVisitor, VectorAccessor};
use grafeo_common::types::NodeId;
use grafeo_common::utils::error::{Error, Result};
use ordered_float::OrderedFloat;
use parking_lot::RwLock;
use std::collections::HashMap;
use std::sync::Arc;

/// A quantized index's quantizer as a saved index holds it (see
/// [`QuantizedHnswIndex::saved_quantizer`]).
#[derive(Debug, Clone, PartialEq)]
pub enum SavedQuantizer {
    /// Nothing to fit: no quantization, or binary quantization.
    Stateless,
    /// Not fitted yet: the ids of the training samples, in insert order.
    Unfitted(Vec<NodeId>),
    /// A fitted scalar quantizer.
    Scalar(ScalarQuantizer),
    /// A fitted product quantizer.
    Product(ProductQuantizer),
}

/// HNSW index with quantization support for memory-efficient search.
///
/// Keeps each member's quantized code and the HNSW topology; the
/// full-precision vectors are read through the accessor each call is given.
/// The search process:
///
/// 1. Search the HNSW graph for candidates
/// 2. Score them with the quantized codes (binary and product)
/// 3. Rescore the top candidates with full-precision distances
pub struct QuantizedHnswIndex {
    /// The underlying HNSW index (topology only).
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
    /// The ids of the vectors inserted before the quantizer is trained, in
    /// insert order: ids, not vectors (#712). Training reads the vectors of
    /// the ones still in the index through the accessor.
    training_samples: RwLock<Vec<NodeId>>,
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
        Self::over(HnswIndex::new(config), quantization)
    }

    /// Creates a new quantized HNSW index with a fixed seed for reproducibility.
    #[must_use]
    pub fn with_seed(config: HnswConfig, quantization: QuantizationType, seed: u64) -> Self {
        Self::over(HnswIndex::with_seed(config, seed), quantization)
    }

    /// The index over an empty `hnsw`.
    fn over(hnsw: HnswIndex, quantization: QuantizationType) -> Self {
        Self {
            hnsw,
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

    /// The underlying HNSW index: the topology and its members.
    #[must_use]
    pub fn hnsw(&self) -> &HnswIndex {
        &self.hnsw
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

    /// The bytes of one member's code, or `None` while the index keeps no
    /// codes (no quantization, or a quantizer not fitted yet).
    #[must_use]
    pub fn code_length(&self) -> Option<usize> {
        let dimensions = self.config().dimensions;
        match self.quantization_type {
            QuantizationType::None => None,
            QuantizationType::Scalar => (*self.quantizer_trained.read()).then_some(dimensions),
            QuantizationType::Binary => Some(BinaryQuantizer::bytes_needed(dimensions)),
            QuantizationType::Product { num_subvectors } => {
                (*self.quantizer_trained.read()).then_some(num_subvectors)
            }
        }
    }

    /// Returns the memory usage estimate in bytes: the members' codes.
    #[must_use]
    pub fn memory_usage(&self) -> usize {
        self.hnsw.len() * self.code_length().unwrap_or(0)
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
    #[must_use]
    pub fn theoretical_compression_ratio(&self) -> f32 {
        self.quantization_type
            .compression_ratio(self.config().dimensions) as f32
    }

    /// Returns the size of the codes compared to the full-precision vectors
    /// they stand for (which the index does not keep): below 1.0 once the
    /// quantizer is fitted, 0.0 while it keeps no codes.
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
    /// The vector is quantized for search; `accessor` reads the vectors of
    /// the other members (this one's is `vector`, stored or not). For scalar
    /// and product quantization, the first `training_threshold` vectors are
    /// used to train the quantizer.
    ///
    /// # Panics
    ///
    /// Panics if vector dimensions don't match configuration.
    pub fn insert(&self, id: NodeId, vector: &[f32], accessor: &impl VectorAccessor) {
        self.hnsw.insert(id, vector, accessor);

        // A fit trains on and codes this node by the vector given.
        let newest = |member: NodeId| {
            if member == id {
                Some(Arc::from(vector))
            } else {
                accessor.get_vector(member)
            }
        };
        match self.quantization_type {
            QuantizationType::None => {}
            QuantizationType::Scalar => self.insert_scalar_quantized(id, vector, &newest),
            QuantizationType::Binary => self.insert_binary_quantized(id, vector),
            QuantizationType::Product { num_subvectors } => {
                self.insert_product_quantized(id, vector, num_subvectors, &newest);
            }
        }
    }

    /// Takes `id` in as a training sample and returns the vectors to train
    /// on once enough are live: at the threshold, the samples no longer in
    /// the index and the repeats of an id (its first place kept) go, and
    /// the rest train only when they still reach it, in insert order, with
    /// each id's current vector. The caller holds the samples' lock through
    /// the training.
    fn live_training_set(
        &self,
        samples: &mut Vec<NodeId>,
        id: NodeId,
        accessor: &impl VectorAccessor,
    ) -> Option<Vec<Arc<[f32]>>> {
        samples.push(id);
        if samples.len() < self.training_threshold {
            return None;
        }
        let mut seen = std::collections::HashSet::with_capacity(samples.len());
        samples.retain(|sample| self.hnsw.contains(*sample) && seen.insert(*sample));
        if samples.len() < self.training_threshold {
            return None;
        }
        let set: Vec<Arc<[f32]>> = samples
            .iter()
            .filter_map(|sample| accessor.get_vector(*sample))
            .collect();
        (set.len() >= self.training_threshold).then_some(set)
    }

    /// Every member's vector, through `accessor`, for the codes of a fit.
    fn member_vectors(&self, accessor: &impl VectorAccessor) -> Vec<(NodeId, Arc<[f32]>)> {
        self.hnsw
            .member_ids()
            .into_iter()
            .filter_map(|id| accessor.get_vector(id).map(|vector| (id, vector)))
            .collect()
    }

    /// Inserts with scalar quantization.
    fn insert_scalar_quantized(&self, id: NodeId, vector: &[f32], accessor: &impl VectorAccessor) {
        let trained = *self.quantizer_trained.read();

        if trained {
            // Quantizer is ready, quantize and store
            if let Some(ref quantizer) = *self.scalar_quantizer.read() {
                let quantized = quantizer.quantize(vector);
                self.scalar_vectors.write().insert(id, quantized);
            }
        } else {
            // Still collecting training samples
            let mut samples = self.training_samples.write();
            if let Some(set) = self.live_training_set(&mut samples, id, accessor) {
                // Time to train the quantizer
                let refs: Vec<&[f32]> = set.iter().map(AsRef::as_ref).collect();
                let quantizer = ScalarQuantizer::train(&refs);

                // Code every member
                let mut scalar_vecs = self.scalar_vectors.write();
                for (member, member_vector) in self.member_vectors(accessor) {
                    scalar_vecs.insert(member, quantizer.quantize(&member_vector));
                }

                // Store the quantizer and mark as trained
                *self.scalar_quantizer.write() = Some(quantizer);
                *self.quantizer_trained.write() = true;
                *samples = Vec::new();
            }
        }
    }

    /// Inserts with binary quantization.
    fn insert_binary_quantized(&self, id: NodeId, vector: &[f32]) {
        let bits = BinaryQuantizer::quantize(vector);
        self.binary_vectors.write().insert(id, bits);
    }

    /// Inserts with product quantization.
    fn insert_product_quantized(
        &self,
        id: NodeId,
        vector: &[f32],
        num_subvectors: usize,
        accessor: &impl VectorAccessor,
    ) {
        let trained = *self.quantizer_trained.read();

        if trained {
            // Quantizer is ready, quantize and store
            if let Some(ref quantizer) = *self.product_quantizer.read() {
                let codes = quantizer.quantize(vector);
                self.product_codes.write().insert(id, codes);
            }
        } else {
            // Still collecting training samples
            let mut samples = self.training_samples.write();
            if let Some(set) = self.live_training_set(&mut samples, id, accessor) {
                // Time to train the product quantizer
                let refs: Vec<&[f32]> = set.iter().map(AsRef::as_ref).collect();
                let quantizer = ProductQuantizer::train(&refs, num_subvectors, 256, 10);

                // Code every member
                let mut codes = self.product_codes.write();
                for (member, member_vector) in self.member_vectors(accessor) {
                    codes.insert(member, quantizer.quantize(&member_vector));
                }

                // Store the quantizer and mark as trained
                *self.product_quantizer.write() = Some(quantizer);
                *self.quantizer_trained.write() = true;
                *samples = Vec::new();
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
    /// A query of another size than the configured `dimensions` finds
    /// nothing (see [`HnswIndex::search`]).
    #[must_use]
    pub fn search_with_ef(
        &self,
        query: &[f32],
        k: usize,
        ef: usize,
        accessor: &impl VectorAccessor,
    ) -> Vec<(NodeId, f32)> {
        infallible(self.search_visiting(query, k, ef, None, accessor, &mut no_visit))
    }

    /// The search behind every `search` method: the `k` nearest nodes to `query` with beam width
    /// `ef`, among `allowlist` when given, calling `visit` before each node the graph search
    /// expands (see [`HnswIndex::search_visiting`]). The first error of `visit` ends the search,
    /// which returns it.
    ///
    /// # Errors
    ///
    /// The first error of `visit`.
    pub fn search_visiting<E>(
        &self,
        query: &[f32],
        k: usize,
        ef: usize,
        allowlist: Option<&std::collections::HashSet<NodeId>>,
        accessor: &impl VectorAccessor,
        visit: &mut impl FnMut() -> std::result::Result<(), E>,
    ) -> std::result::Result<Vec<(NodeId, f32)>, E> {
        if query.len() != self.config().dimensions {
            return Ok(Vec::new());
        }
        // The allowlist goes into the graph search, which collects only
        // allowed nodes as it walks the whole graph: filtering the nearest
        // nodes overall afterwards found fewer than `k`, or none, when the
        // allowed ones lay farther out. Rescoring then only reorders allowed
        // candidates.
        match self.quantization_type {
            QuantizationType::None => {
                // No quantization, use standard HNSW
                self.hnsw
                    .search_visiting(query, k, ef, allowlist, accessor, visit)
            }
            QuantizationType::Scalar => {
                self.search_scalar_quantized(query, k, ef, allowlist, accessor, visit)
            }
            QuantizationType::Binary => {
                self.search_binary_quantized(query, k, ef, allowlist, accessor, visit)
            }
            QuantizationType::Product { .. } => {
                self.search_product_quantized(query, k, ef, allowlist, accessor, visit)
            }
        }
    }

    /// Search with scalar quantization.
    fn search_scalar_quantized<E>(
        &self,
        query: &[f32],
        k: usize,
        ef: usize,
        allowlist: Option<&std::collections::HashSet<NodeId>>,
        accessor: &impl VectorAccessor,
        visit: &mut impl FnMut() -> std::result::Result<(), E>,
    ) -> std::result::Result<Vec<(NodeId, f32)>, E> {
        let trained = *self.quantizer_trained.read();

        if !trained {
            // Quantizer not ready, fall back to exact search
            return self
                .hnsw
                .search_visiting(query, k, ef, allowlist, accessor, visit);
        }

        // Get candidates using HNSW (with full precision distances for now)
        // In a production system, you'd modify HNSW to use quantized distances
        let num_candidates = if self.rescore {
            k.saturating_mul(self.rescore_factor)
        } else {
            k
        };

        let candidates =
            self.hnsw
                .search_visiting(query, num_candidates, ef, allowlist, accessor, visit)?;

        if !self.rescore {
            return Ok(candidates.into_iter().take(k).collect());
        }

        // Rescore with exact distances
        Ok(self.rescore_candidates(query, candidates, k, accessor))
    }

    /// Search with binary quantization.
    fn search_binary_quantized<E>(
        &self,
        query: &[f32],
        k: usize,
        ef: usize,
        allowlist: Option<&std::collections::HashSet<NodeId>>,
        accessor: &impl VectorAccessor,
        visit: &mut impl FnMut() -> std::result::Result<(), E>,
    ) -> std::result::Result<Vec<(NodeId, f32)>, E> {
        let binary_vecs = self.binary_vectors.read();

        if binary_vecs.is_empty() {
            return self
                .hnsw
                .search_visiting(query, k, ef, allowlist, accessor, visit);
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
        let hnsw_candidates =
            self.hnsw
                .search_visiting(query, num_candidates, ef, allowlist, accessor, visit)?;

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

        // Stable: equal approximate distances keep the exact order the HNSW
        // search gave them (nearest first, ties by id)
        scored.sort_by_key(|(_, d)| OrderedFloat(*d));
        scored.truncate(num_candidates);

        if !self.rescore {
            return Ok(scored.into_iter().take(k).collect());
        }

        // Rescore with exact distances
        Ok(self.rescore_candidates(query, scored, k, accessor))
    }

    /// Search with product quantization.
    fn search_product_quantized<E>(
        &self,
        query: &[f32],
        k: usize,
        ef: usize,
        allowlist: Option<&std::collections::HashSet<NodeId>>,
        accessor: &impl VectorAccessor,
        visit: &mut impl FnMut() -> std::result::Result<(), E>,
    ) -> std::result::Result<Vec<(NodeId, f32)>, E> {
        let trained = *self.quantizer_trained.read();

        if !trained {
            // Quantizer not ready, fall back to exact search
            return self
                .hnsw
                .search_visiting(query, k, ef, allowlist, accessor, visit);
        }

        // Get candidates using HNSW
        let num_candidates = if self.rescore {
            k.saturating_mul(self.rescore_factor)
        } else {
            k
        };

        let candidates =
            self.hnsw
                .search_visiting(query, num_candidates, ef, allowlist, accessor, visit)?;

        if !self.rescore {
            return Ok(candidates.into_iter().take(k).collect());
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

            // Stable: equal approximate distances keep the exact order the
            // HNSW search gave them (nearest first, ties by id)
            scored.sort_by_key(|(_, d)| OrderedFloat(*d));
            scored.truncate(k);

            // Optionally rescore top results with full precision
            if self.rescore {
                return Ok(self.rescore_candidates(query, scored, k, accessor));
            }

            Ok(scored)
        } else {
            // Fall back to exact search
            Ok(candidates.into_iter().take(k).collect())
        }
    }

    /// Rescores candidates with exact full-precision distances, the vectors
    /// read through `accessor`.
    fn rescore_candidates(
        &self,
        query: &[f32],
        candidates: Vec<(NodeId, f32)>,
        k: usize,
        accessor: &impl VectorAccessor,
    ) -> Vec<(NodeId, f32)> {
        let metric = self.config().metric;

        let mut rescored: Vec<(NodeId, f32)> = candidates
            .into_iter()
            .filter_map(|(id, _approx_dist)| {
                accessor.get_vector(id).map(|vec| {
                    let exact_dist = compute_distance(query, &vec, metric);
                    (id, exact_dist)
                })
            })
            .collect();

        rescored.sort_by_key(|(id, d)| (OrderedFloat(*d), *id));
        rescored.truncate(k);
        rescored
    }

    /// Returns true if the index contains the given ID.
    #[must_use]
    pub fn contains(&self, id: NodeId) -> bool {
        self.hnsw.contains(id)
    }

    /// Removes a vector from the index, mending the links of the nodes that
    /// led to it with the vectors `accessor` reads (see
    /// [`HnswIndex::remove`]).
    pub fn remove(&self, id: NodeId, accessor: &impl VectorAccessor) -> bool {
        let removed = self.hnsw.remove(id, accessor);
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
        removed
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
        crate::index::vector::batch::map(queries, |query| self.search(query, k, accessor))
    }

    /// Searches with an allowlist filter.
    ///
    /// Only nodes in the `allowlist` can appear in results: the graph search
    /// walks the whole graph and collects only allowed nodes, with its beam
    /// width scaled by how few nodes the allowlist keeps, as
    /// [`HnswIndex::search_with_filter`] does (see [`HnswIndex::filtered_ef`]).
    #[must_use]
    pub fn search_with_filter(
        &self,
        query: &[f32],
        k: usize,
        allowlist: &std::collections::HashSet<NodeId>,
        accessor: &impl VectorAccessor,
    ) -> Vec<(NodeId, f32)> {
        let ef = self.filtered_ef(k, allowlist);
        infallible(self.search_visiting(query, k, ef, Some(allowlist), accessor, &mut no_visit))
    }

    /// The beam width of a search among `allowlist` without one given: the
    /// configured `ef` scaled up by how few nodes the allowlist keeps (see
    /// [`HnswIndex::filtered_ef`]).
    #[must_use]
    pub fn filtered_ef(&self, k: usize, allowlist: &std::collections::HashSet<NodeId>) -> usize {
        self.hnsw.filtered_ef(k, allowlist)
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
        infallible(self.search_visiting(query, k, ef, Some(allowlist), accessor, &mut no_visit))
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
        crate::index::vector::batch::map(queries, |query| {
            self.search_with_ef(query, k, ef, accessor)
        })
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
        crate::index::vector::batch::map(queries, |query| {
            self.search_with_filter(query, k, allowlist, accessor)
        })
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
        crate::index::vector::batch::map(queries, |query| {
            self.search_with_ef_and_filter(query, k, ef, allowlist, accessor)
        })
    }

    /// The quantizer as a saved index holds it: fitted, the ids of its
    /// training samples so far, or nothing to fit.
    #[must_use]
    pub fn saved_quantizer(&self) -> SavedQuantizer {
        let samples = self.training_samples.read();
        let trained = *self.quantizer_trained.read();
        match self.quantization_type {
            QuantizationType::None | QuantizationType::Binary => SavedQuantizer::Stateless,
            QuantizationType::Scalar | QuantizationType::Product { .. } if !trained => {
                SavedQuantizer::Unfitted(samples.clone())
            }
            QuantizationType::Scalar => self.scalar_quantizer.read().clone().map_or_else(
                || SavedQuantizer::Unfitted(Vec::new()),
                SavedQuantizer::Scalar,
            ),
            QuantizationType::Product { .. } => self.product_quantizer.read().clone().map_or_else(
                || SavedQuantizer::Unfitted(Vec::new()),
                SavedQuantizer::Product,
            ),
        }
    }

    /// Restores the quantizer [`saved_quantizer`](Self::saved_quantizer)
    /// gave, before the codes ([`restore_code`](Self::restore_code)).
    ///
    /// # Errors
    ///
    /// [`Error::Corruption`] when it is not of this index's quantization, or
    /// a fitted quantizer is for other dimensions or another number of
    /// subvectors.
    pub fn restore_quantizer(&self, saved: SavedQuantizer) -> Result<()> {
        let dimensions = self.config().dimensions;
        let refused = |what: &str| {
            Err(Error::corruption(format!(
                "the saved quantizer of a {:?} index {what}",
                self.quantization_type
            )))
        };
        let mut samples = self.training_samples.write();
        match (self.quantization_type, saved) {
            (QuantizationType::None | QuantizationType::Binary, SavedQuantizer::Stateless) => {}
            (
                QuantizationType::Scalar | QuantizationType::Product { .. },
                SavedQuantizer::Unfitted(ids),
            ) => {
                *samples = ids;
                *self.quantizer_trained.write() = false;
            }
            (QuantizationType::Scalar, SavedQuantizer::Scalar(quantizer)) => {
                if quantizer.dimensions() != dimensions {
                    return refused(&format!(
                        "has {} dimensions, where the index has {dimensions}",
                        quantizer.dimensions()
                    ));
                }
                *self.scalar_quantizer.write() = Some(quantizer);
                *self.quantizer_trained.write() = true;
                samples.clear();
            }
            (QuantizationType::Product { num_subvectors }, SavedQuantizer::Product(quantizer)) => {
                if quantizer.dimensions() != dimensions
                    || quantizer.num_subvectors() != num_subvectors
                {
                    return refused(&format!(
                        "has {} dimensions in {} subvectors, where the index has {dimensions} \
                         in {num_subvectors}",
                        quantizer.dimensions(),
                        quantizer.num_subvectors()
                    ));
                }
                *self.product_quantizer.write() = Some(quantizer);
                *self.quantizer_trained.write() = true;
                samples.clear();
            }
            (_, other) => return refused(&format!("is {other:?}")),
        }
        Ok(())
    }

    /// Member `id`'s code, as [`code_length`](Self::code_length) bytes (the
    /// binary bits as little-endian words): `None` while the index keeps no
    /// codes, or when `id` has none.
    #[must_use]
    pub fn code_of(&self, id: NodeId) -> Option<Vec<u8>> {
        match self.quantization_type {
            QuantizationType::None => None,
            QuantizationType::Scalar => self.scalar_vectors.read().get(&id).cloned(),
            QuantizationType::Binary => self
                .binary_vectors
                .read()
                .get(&id)
                .map(|words| words.iter().flat_map(|word| word.to_le_bytes()).collect()),
            QuantizationType::Product { .. } => self.product_codes.read().get(&id).cloned(),
        }
    }

    /// Restores member `id`'s code as [`code_of`](Self::code_of) gave it.
    ///
    /// # Errors
    ///
    /// [`Error::Corruption`] when the index keeps no codes or `code` is not
    /// [`code_length`](Self::code_length) bytes.
    pub fn restore_code(&self, id: NodeId, code: &[u8]) -> Result<()> {
        let length = self.code_length();
        if length != Some(code.len()) {
            return Err(Error::corruption(format!(
                "a saved code of {} bytes for node {} of a {:?} index, which keeps codes of {}",
                code.len(),
                id.as_u64(),
                self.quantization_type,
                length.map_or_else(|| "no bytes".to_string(), |bytes| format!("{bytes} bytes"))
            )));
        }
        match self.quantization_type {
            QuantizationType::None => {}
            QuantizationType::Scalar => {
                self.scalar_vectors.write().insert(id, code.to_vec());
            }
            QuantizationType::Binary => {
                let words = code
                    .as_chunks::<8>()
                    .0
                    .iter()
                    .map(|word| u64::from_le_bytes(*word))
                    .collect();
                self.binary_vectors.write().insert(id, words);
            }
            QuantizationType::Product { .. } => {
                self.product_codes.write().insert(id, code.to_vec());
            }
        }
        Ok(())
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

    /// Hands the topology to `visitor` one node at a time (see
    /// [`HnswIndex::visit_topology`]).
    ///
    /// # Errors
    ///
    /// Returns the first error `visitor` returns.
    pub fn visit_topology(&self, visitor: &mut dyn TopologyVisitor) -> Result<()> {
        self.hnsw.visit_topology(visitor)
    }

    /// Starts restoring a topology node by node (see
    /// [`HnswIndex::begin_restore`]). The codes and the quantizer are not
    /// touched.
    pub fn begin_restore(&self, entry_point: Option<NodeId>, max_level: usize, node_count: usize) {
        self.hnsw.begin_restore(entry_point, max_level, node_count);
    }

    /// Restores one node of a topology begun with
    /// [`begin_restore`](Self::begin_restore) (see
    /// [`HnswIndex::restore_node`]).
    pub fn restore_node(&self, id: NodeId, layers: Vec<Vec<NodeId>>) {
        self.hnsw.restore_node(id, layers);
    }

    /// The first neighbor reference that breaks the rules of the topology
    /// (see [`HnswIndex::first_broken_link`]).
    pub(crate) fn first_broken_link(&self) -> Option<BrokenLink> {
        self.hnsw.first_broken_link()
    }

    /// Returns estimated heap memory in bytes: the topology's walk and the
    /// codes.
    #[must_use]
    pub fn heap_memory_bytes(&self) -> usize {
        self.hnsw.heap_memory_bytes() + self.memory_usage()
    }

    /// [`heap_memory_bytes`](Self::heap_memory_bytes), the topology as its
    /// writes keep it: without a walk.
    #[must_use]
    pub fn counted_bytes(&self) -> usize {
        self.hnsw.counted_bytes() + self.memory_usage()
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

    /// The vectors the tests' indexes read, kept as a store keeps them.
    #[derive(Default)]
    struct Store(RwLock<HashMap<NodeId, Arc<[f32]>>>);

    impl VectorAccessor for Store {
        fn get_vector(&self, id: NodeId) -> Option<Arc<[f32]>> {
            self.0.read().get(&id).cloned()
        }
    }

    impl Store {
        /// Stores `vector` as `id`'s.
        fn put(&self, id: NodeId, vector: &[f32]) {
            self.0.write().insert(id, vector.into());
        }

        /// Stores `vector` as `id`'s and inserts it into `index`.
        fn insert(&self, index: &QuantizedHnswIndex, id: NodeId, vector: &[f32]) {
            self.put(id, vector);
            index.insert(id, vector, self);
        }

        /// Takes `id` out of `index` while its vector is in place, then out
        /// of the store.
        fn remove(&self, index: &QuantizedHnswIndex, id: NodeId) -> bool {
            let removed = index.remove(id, self);
            self.0.write().remove(&id);
            removed
        }
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
        let store = Store::default();
        let config = HnswConfig::new(4, DistanceMetric::Euclidean);
        let index = QuantizedHnswIndex::new(config, QuantizationType::None);

        let vectors = create_test_vectors(50, 4);
        for (i, vec) in vectors.iter().enumerate() {
            store.insert(&index, NodeId::new(i as u64 + 1), vec);
        }

        assert_eq!(index.len(), 50);

        let results = index.search(&vectors[25], 5, &store);
        assert_eq!(results.len(), 5);
        assert_eq!(results[0].0, NodeId::new(26));
    }

    #[test]
    fn test_quantized_hnsw_scalar_quantization() {
        let store = Store::default();
        let config = HnswConfig::new(4, DistanceMetric::Euclidean);
        let index = QuantizedHnswIndex::with_seed(config, QuantizationType::Scalar, 42)
            .with_training_threshold(10); // Lower threshold for test

        let vectors = create_test_vectors(50, 4);
        for (i, vec) in vectors.iter().enumerate() {
            store.insert(&index, NodeId::new(i as u64 + 1), vec);
        }

        assert_eq!(index.len(), 50);
        // Scalar quantization should give 4x theoretical compression
        assert_eq!(index.theoretical_compression_ratio(), 4.0);
        // The index keeps the codes only: a byte per value of 4 bytes
        assert_eq!(index.memory_ratio(), 0.25);

        let results = index.search(&vectors[25], 5, &store);
        assert_eq!(results.len(), 5);
        // Should find the correct vector (rescoring fixes quantization errors)
        assert_eq!(results[0].0, NodeId::new(26));
    }

    #[test]
    fn test_quantized_hnsw_binary_quantization() {
        let store = Store::default();
        let config = HnswConfig::new(4, DistanceMetric::Euclidean);
        let index = QuantizedHnswIndex::with_seed(config, QuantizationType::Binary, 42);

        let vectors = create_test_vectors(50, 4);
        for (i, vec) in vectors.iter().enumerate() {
            store.insert(&index, NodeId::new(i as u64 + 1), vec);
        }

        assert_eq!(index.len(), 50);

        let results = index.search(&vectors[25], 5, &store);
        assert_eq!(results.len(), 5);
        // Binary is less accurate but should still work with rescoring
        assert_eq!(results[0].0, NodeId::new(26));
    }

    #[test]
    fn test_quantized_hnsw_without_rescore() {
        let store = Store::default();
        let config = HnswConfig::new(4, DistanceMetric::Euclidean);
        let index = QuantizedHnswIndex::with_seed(config, QuantizationType::Scalar, 42)
            .with_training_threshold(10)
            .without_rescore();

        let vectors = create_test_vectors(50, 4);
        for (i, vec) in vectors.iter().enumerate() {
            store.insert(&index, NodeId::new(i as u64 + 1), vec);
        }

        let results = index.search(&vectors[25], 5, &store);
        assert_eq!(results.len(), 5);
        // Without rescoring, might not be exact but should be close
    }

    /// #593: a query of another size than the index's finds nothing, with
    /// every quantization (trained or not), and no search panics.
    #[test]
    fn a_query_of_another_size_finds_nothing_with_every_quantization() {
        let store = Store::default();
        for quantization in [
            QuantizationType::None,
            QuantizationType::Scalar,
            QuantizationType::Binary,
            QuantizationType::Product { num_subvectors: 4 },
        ] {
            let config = HnswConfig::new(8, DistanceMetric::Euclidean);
            let index =
                QuantizedHnswIndex::with_seed(config, quantization, 3).with_training_threshold(19);
            let vectors = create_test_vectors(88, 8);
            for (i, vec) in vectors.iter().enumerate() {
                store.insert(&index, NodeId::new(i as u64 + 1), vec);
            }
            let allowlist: std::collections::HashSet<NodeId> = (1..=88).map(NodeId::new).collect();
            for query in [vec![0.3; 3], vec![0.3; 19], vec![]] {
                assert!(
                    index.search(&query, 3, &store).is_empty(),
                    "{quantization:?}"
                );
                assert!(
                    index
                        .search_with_filter(&query, 3, &allowlist, &store)
                        .is_empty(),
                    "{quantization:?}"
                );
                assert!(
                    index.batch_search(std::slice::from_ref(&query), 3, &store)[0].is_empty(),
                    "{quantization:?}"
                );
            }
            assert_eq!(
                index.search(&vectors[19], 1, &store)[0].0,
                NodeId::new(20),
                "{quantization:?}"
            );
        }
    }

    #[test]
    fn test_quantized_hnsw_remove() {
        let store = Store::default();
        let config = HnswConfig::new(4, DistanceMetric::Euclidean);
        let index = QuantizedHnswIndex::new(config, QuantizationType::Binary);

        store.insert(&index, NodeId::new(1), &[0.1, 0.2, 0.3, 0.4]);
        store.insert(&index, NodeId::new(2), &[0.5, 0.6, 0.7, 0.8]);

        assert_eq!(index.len(), 2);
        assert!(store.remove(&index, NodeId::new(1)));
        assert_eq!(index.len(), 1);
        assert!(!index.contains(NodeId::new(1)));
        assert!(index.contains(NodeId::new(2)));
    }

    #[test]
    fn test_quantized_hnsw_batch_operations() {
        let store = Store::default();
        let config = HnswConfig::new(4, DistanceMetric::Euclidean);
        let index = QuantizedHnswIndex::with_seed(config, QuantizationType::Scalar, 42)
            .with_training_threshold(10);

        let vectors = create_test_vectors(50, 4);
        let pairs: Vec<_> = vectors
            .iter()
            .enumerate()
            .map(|(i, v)| (NodeId::new(i as u64 + 1), v.as_slice()))
            .collect();

        for &(id, vector) in &pairs {
            store.put(id, vector);
        }
        index.batch_insert(pairs, &store);
        assert_eq!(index.len(), 50);

        // Batch search
        let queries = vec![vectors[10].clone(), vectors[30].clone()];
        let results = index.batch_search(&queries, 3, &store);

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
    fn a_fit_codes_the_inserted_node_by_its_given_vector() {
        for quantization in [
            QuantizationType::Scalar,
            QuantizationType::Product { num_subvectors: 2 },
        ] {
            let store = Store::default();
            let config = HnswConfig::new(4, DistanceMetric::Euclidean);
            let index =
                QuantizedHnswIndex::with_seed(config, quantization, 3).with_training_threshold(10);
            let vectors = create_test_vectors(10, 4);
            for (i, vector) in vectors.iter().enumerate().take(9) {
                store.insert(&index, NodeId::new(i as u64), vector);
            }
            // The tenth reaches the threshold before the store holds it.
            index.insert(NodeId::new(9), &vectors[9], &store);
            assert_eq!(
                index.memory_usage(),
                10 * index.code_length().expect("fitted")
            );
            assert!(index.code_of(NodeId::new(9)).is_some(), "{quantization:?}");
        }
    }

    #[test]
    fn test_quantized_hnsw_memory_usage() {
        let store = Store::default();
        let config = HnswConfig::new(384, DistanceMetric::Cosine);
        let index =
            QuantizedHnswIndex::new(config, QuantizationType::Scalar).with_training_threshold(10);
        assert_eq!(index.memory_usage(), 0);

        // Before the fit the index keeps no codes, and never a vector copy.
        store.insert(&index, NodeId::new(1), &vec![0.1f32; 384]);
        assert_eq!(index.memory_usage(), 0);

        // From the fit on, a byte per dimension for every member.
        for (i, vector) in create_test_vectors(11, 384).iter().enumerate().skip(1) {
            store.insert(&index, NodeId::new(i as u64 + 1), vector);
        }
        assert_eq!(index.memory_usage(), 11 * 384);
    }

    #[test]
    fn test_quantized_hnsw_product_quantization() {
        let store = Store::default();
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
            store.insert(&index, NodeId::new(i as u64 + 1), vec);
        }

        assert_eq!(index.len(), 50);
        // Product quantization: (32 * 4) / 8 = 16x compression
        assert_eq!(index.theoretical_compression_ratio(), 16.0);

        let results = index.search(&vectors[25], 5, &store);
        assert_eq!(results.len(), 5);
        // With rescoring, should find the correct vector
        assert_eq!(results[0].0, NodeId::new(26));
    }

    #[test]
    fn test_quantized_hnsw_product_before_training() {
        let store = Store::default();
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
            store.insert(&index, NodeId::new(i as u64 + 1), vec);
        }

        // Should still work (falls back to exact search before training)
        let results = index.search(&vectors[5], 3, &store);
        assert_eq!(results.len(), 3);
        assert_eq!(results[0].0, NodeId::new(6));
    }

    #[test]
    fn test_search_with_allowlist_filter() {
        let store = Store::default();
        let config = HnswConfig::new(4, DistanceMetric::Euclidean);
        let index = QuantizedHnswIndex::new(config, QuantizationType::Scalar);

        let vectors = create_test_vectors(50, 4);
        for (i, vec) in vectors.iter().enumerate() {
            store.insert(&index, NodeId::new(i as u64 + 1), vec);
        }

        let allowlist: std::collections::HashSet<NodeId> =
            [1, 10, 25, 40].iter().map(|&i| NodeId::new(i)).collect();
        let results = index.search_with_filter(&vectors[24], 3, &allowlist, &store);
        assert!(!results.is_empty(), "results is empty");
        assert!(results.len() <= 3);
        for (id, _) in &results {
            assert!(allowlist.contains(id), "result {id:?} not in allowlist");
        }
    }

    #[test]
    fn test_search_with_ef_and_allowlist_filter() {
        let store = Store::default();
        let config = HnswConfig::new(4, DistanceMetric::Euclidean);
        let index = QuantizedHnswIndex::new(config, QuantizationType::Binary);

        let vectors = create_test_vectors(50, 4);
        for (i, vec) in vectors.iter().enumerate() {
            store.insert(&index, NodeId::new(i as u64 + 1), vec);
        }

        let allowlist: std::collections::HashSet<NodeId> =
            [5, 15, 30].iter().map(|&i| NodeId::new(i)).collect();
        let results = index.search_with_ef_and_filter(&vectors[14], 2, 50, &allowlist, &store);
        assert!(!results.is_empty(), "results is empty");
        assert!(results.len() <= 2);
        for (id, _) in &results {
            assert!(allowlist.contains(id));
        }
    }

    #[test]
    fn test_batch_search_with_ef() {
        let store = Store::default();
        let config = HnswConfig::new(4, DistanceMetric::Euclidean);
        let index = QuantizedHnswIndex::new(config, QuantizationType::Scalar);

        let vectors = create_test_vectors(50, 4);
        for (i, vec) in vectors.iter().enumerate() {
            store.insert(&index, NodeId::new(i as u64 + 1), vec);
        }

        let queries = vec![vectors[10].clone(), vectors[30].clone()];
        let results = index.batch_search_with_ef(&queries, 3, 50, &store);
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].len(), 3);
        assert_eq!(results[1].len(), 3);
    }

    #[test]
    fn test_batch_search_with_filter() {
        let store = Store::default();
        let config = HnswConfig::new(4, DistanceMetric::Euclidean);
        let index = QuantizedHnswIndex::new(config, QuantizationType::Binary);

        let vectors = create_test_vectors(50, 4);
        for (i, vec) in vectors.iter().enumerate() {
            store.insert(&index, NodeId::new(i as u64 + 1), vec);
        }

        let allowlist: std::collections::HashSet<NodeId> = (1..=25).map(NodeId::new).collect();
        let queries = vec![vectors[5].clone(), vectors[20].clone()];
        let results = index.batch_search_with_filter(&queries, 3, &allowlist, &store);
        assert_eq!(results.len(), 2);
        for batch in &results {
            for (id, _) in batch {
                assert!(allowlist.contains(id));
            }
        }
    }

    #[test]
    fn test_batch_search_with_ef_and_filter() {
        let store = Store::default();
        let config = HnswConfig::new(4, DistanceMetric::Euclidean);
        let index = QuantizedHnswIndex::new(config, QuantizationType::Scalar);

        let vectors = create_test_vectors(50, 4);
        for (i, vec) in vectors.iter().enumerate() {
            store.insert(&index, NodeId::new(i as u64 + 1), vec);
        }

        let allowlist: std::collections::HashSet<NodeId> = (10..=40).map(NodeId::new).collect();
        let queries = vec![vectors[15].clone()];
        let results = index.batch_search_with_ef_and_filter(&queries, 5, 100, &allowlist, &store);
        assert_eq!(results.len(), 1);
        for (id, _) in &results[0] {
            assert!(allowlist.contains(id));
        }
    }

    #[test]
    fn test_snapshot_and_restore_topology() {
        let store = Store::default();
        let config = HnswConfig::new(4, DistanceMetric::Euclidean);
        let index = QuantizedHnswIndex::new(config.clone(), QuantizationType::Scalar);

        let vectors = create_test_vectors(20, 4);
        for (i, vec) in vectors.iter().enumerate() {
            store.insert(&index, NodeId::new(i as u64 + 1), vec);
        }

        let before = index.search(&vectors[10], 5, &store);

        let (entry, max_level, nodes) = index.snapshot_topology();

        let index2 = QuantizedHnswIndex::new(config, QuantizationType::Scalar);
        // Re-insert vectors (topology restore only restores graph structure)
        for (i, vec) in vectors.iter().enumerate() {
            store.insert(&index2, NodeId::new(i as u64 + 1), vec);
        }
        index2.restore_topology(entry, max_level, nodes);

        let after = index2.search(&vectors[10], 5, &store);
        assert_eq!(before.len(), after.len());
    }

    #[test]
    fn test_heap_memory_bytes() {
        let store = Store::default();
        let config = HnswConfig::new(4, DistanceMetric::Euclidean);
        let index = QuantizedHnswIndex::new(config, QuantizationType::Scalar);

        let empty_mem = index.heap_memory_bytes();
        store.insert(&index, NodeId::new(1), &[0.1, 0.2, 0.3, 0.4]);
        assert!(
            index.heap_memory_bytes() > empty_mem,
            "memory should grow after insert"
        );
    }

    /// A quantizer trains on the samples still in the index alone, in their insert order, once
    /// enough of them are live (#712): a vector removed before training, or replaced by a new
    /// insert of its id, does not train it, and the samples hold ids, so a removed vector is
    /// freed with its entry. Scalar and product quantization.
    #[test]
    fn training_waits_for_live_samples_and_uses_them_alone_in_insert_order() {
        let store = Store::default();
        let vector = |id: u64| -> Vec<f32> {
            (0..4u64)
                .map(|dimension| (id * 4 + dimension) as f32 / 10.0)
                .collect()
        };
        let extreme = vec![100.0_f32; 4];
        let renewed = vec![0.5_f32, 0.25, 0.75, 0.125];
        let probe = [0.3_f32, 0.9, 1.7, 2.2];
        for quantization in [
            QuantizationType::Scalar,
            QuantizationType::Product { num_subvectors: 2 },
        ] {
            let index = QuantizedHnswIndex::with_seed(
                HnswConfig::new(4, DistanceMetric::Euclidean),
                quantization,
                42,
            )
            .with_training_threshold(10);
            for id in 1..=9 {
                let inserted = if id == 2 { extreme.clone() } else { vector(id) };
                store.insert(&index, NodeId::new(id), &inserted);
            }
            assert!(
                store.remove(&index, NodeId::new(2)),
                "{quantization:?}: the extreme one"
            );
            assert!(store.remove(&index, NodeId::new(5)));
            // The tenth sample, a new insert of id 1: seven distinct live ids.
            store.insert(&index, NodeId::new(1), &renewed);
            assert!(
                !*index.quantizer_trained.read(),
                "{quantization:?}: ten samples, seven live"
            );
            store.insert(&index, NodeId::new(10), &vector(10));
            store.insert(&index, NodeId::new(11), &vector(11));
            assert!(
                !*index.quantizer_trained.read(),
                "{quantization:?}: nine live"
            );
            store.insert(&index, NodeId::new(12), &vector(12));
            assert!(
                *index.quantizer_trained.read(),
                "{quantization:?}: ten live"
            );
            assert!(
                index.training_samples.read().is_empty(),
                "{quantization:?}: the samples go once trained"
            );

            let live: Vec<Vec<f32>> = [1, 3, 4, 6, 7, 8, 9, 10, 11, 12]
                .into_iter()
                .map(|id| if id == 1 { renewed.clone() } else { vector(id) })
                .collect();
            let refs: Vec<&[f32]> = live.iter().map(Vec::as_slice).collect();
            match quantization {
                QuantizationType::Scalar => {
                    assert_eq!(
                        index
                            .scalar_quantizer
                            .read()
                            .as_ref()
                            .expect("trained")
                            .quantize(&probe),
                        ScalarQuantizer::train(&refs).quantize(&probe),
                        "scalar: trained on the live samples in order"
                    );
                    assert_eq!(
                        index.scalar_vectors.read().len(),
                        10,
                        "a code per live vector"
                    );
                }
                QuantizationType::Product { num_subvectors } => {
                    assert_eq!(
                        index
                            .product_quantizer
                            .read()
                            .as_ref()
                            .expect("trained")
                            .quantize(&probe),
                        ProductQuantizer::train(&refs, num_subvectors, 256, 10).quantize(&probe),
                        "product: trained on the live samples in order"
                    );
                    assert_eq!(
                        index.product_codes.read().len(),
                        10,
                        "a code per live vector"
                    );
                }
                _ => unreachable!("two quantizations"),
            }
        }
    }

    /// The index keeps no copy of the vectors (#711): rescoring reads them
    /// through the accessor it is given, so a vector the store changed is
    /// measured as the store holds it, and the index's memory is its codes.
    #[test]
    fn a_quantized_index_reads_its_vectors_through_the_accessor() {
        for quantization in [
            QuantizationType::Scalar,
            QuantizationType::Binary,
            QuantizationType::Product { num_subvectors: 2 },
        ] {
            let store = Store::default();
            let index = QuantizedHnswIndex::with_seed(
                HnswConfig::new(4, DistanceMetric::Euclidean),
                quantization,
                42,
            )
            .with_training_threshold(10);
            let vectors = create_test_vectors(40, 4);
            for (i, vec) in vectors.iter().enumerate() {
                store.insert(&index, NodeId::new(i as u64 + 1), vec);
            }
            let code = index.code_length().expect("fitted");
            assert_eq!(index.memory_usage(), 40 * code, "{quantization:?}");

            let query = &vectors[10];
            let moved = [0.11_f32, 0.0, 0.0, 0.0];
            store.put(NodeId::new(11), &moved);
            let found = index.search(query, 40, &store);
            let distance = found
                .iter()
                .find(|(id, _)| *id == NodeId::new(11))
                .map(|&(_, distance)| distance);
            assert_eq!(
                distance,
                Some(compute_distance(query, &moved, DistanceMetric::Euclidean)),
                "{quantization:?}: rescored by the stored vector"
            );
        }
    }
}
