//! Vector quantization algorithms for memory-efficient storage.
//!
//! Quantization reduces vector precision for memory savings:
//!
//! | Method  | Compression | Accuracy | Speed    | Use Case                    |
//! |---------|-------------|----------|----------|----------------------------|
//! | Scalar  | 4x          | ~97%     | Fast     | Default for most datasets   |
//! | Binary  | 32x         | ~80%     | Fastest  | Very large datasets         |
//!
//! # Scalar Quantization
//!
//! Converts f32 values to u8 by learning min/max ranges per dimension:
//!
//! ```ignore
//! use grafeo_core::index::vector::quantization::ScalarQuantizer;
//!
//! let vectors: Vec<Vec<f32>> = get_training_vectors();
//! let quantizer = ScalarQuantizer::train(&vectors);
//!
//! // Quantize: f32 -> u8 (4x compression)
//! let original = vec![0.1f32, 0.5, 0.9];
//! let quantized = quantizer.quantize(&original);
//!
//! // Compute distance in quantized space (approximate)
//! let dist = quantizer.distance_u8(&quantized, &other_quantized);
//! ```
//!
//! # Binary Quantization
//!
//! Converts f32 values to bits (sign only), enabling hamming distance:
//!
//! ```ignore
//! use grafeo_core::index::vector::quantization::BinaryQuantizer;
//!
//! let v = vec![0.1f32, -0.5, 0.0, 0.9];
//! let bits = BinaryQuantizer::quantize(&v);
//!
//! // Hamming distance (count differing bits)
//! let dist = BinaryQuantizer::hamming_distance(&bits, &other_bits);
//! ```

use serde::{Deserialize, Serialize};

// ============================================================================
// Quantization Type
// ============================================================================

/// Quantization strategy for vector storage.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
pub enum QuantizationType {
    /// No quantization - full f32 precision.
    #[default]
    None,
    /// Scalar quantization: f32 -> u8 (4x compression, ~97% accuracy).
    Scalar,
    /// Binary quantization: f32 -> 1 bit (32x compression, ~80% accuracy).
    Binary,
    /// Product quantization: f32 -> M u8 codes (8-32x compression, ~90% accuracy).
    Product {
        /// Number of subvectors (typically 8, 16, 32, 64).
        num_subvectors: usize,
    },
}

impl QuantizationType {
    /// Returns the compression ratio (memory reduction factor).
    ///
    /// For Product quantization, this depends on dimensions and num_subvectors.
    /// The ratio is approximate: dimensions * 4 / num_subvectors.
    #[must_use]
    pub fn compression_ratio(&self, dimensions: usize) -> usize {
        match self {
            Self::None => 1,
            Self::Scalar => 4, // f32 (4 bytes) -> u8 (1 byte)
            Self::Binary => 32, // f32 (4 bytes) -> 1 bit (0.125 bytes)
            Self::Product { num_subvectors } => {
                // Original: dimensions * 4 bytes (f32)
                // Compressed: num_subvectors bytes (u8 codes)
                // Ratio: (dimensions * 4) / num_subvectors
                let m = (*num_subvectors).max(1);
                (dimensions * 4) / m
            }
        }
    }

    /// Returns the name of the quantization type.
    #[must_use]
    pub fn name(&self) -> &'static str {
        match self {
            Self::None => "none",
            Self::Scalar => "scalar",
            Self::Binary => "binary",
            Self::Product { .. } => "product",
        }
    }

    /// Parses from string (case-insensitive).
    #[must_use]
    pub fn from_str(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "none" | "full" | "f32" => Some(Self::None),
            "scalar" | "sq" | "u8" | "int8" => Some(Self::Scalar),
            "binary" | "bin" | "bit" | "1bit" => Some(Self::Binary),
            "product" | "pq" => Some(Self::Product { num_subvectors: 8 }),
            s if s.starts_with("pq") => {
                // Parse "pq8", "pq16", etc.
                s[2..].parse().ok().map(|n| Self::Product { num_subvectors: n })
            }
            _ => None,
        }
    }

    /// Returns true if this quantization type requires training.
    #[must_use]
    pub const fn requires_training(&self) -> bool {
        matches!(self, Self::Scalar | Self::Product { .. })
    }
}

// ============================================================================
// Scalar Quantization
// ============================================================================

/// Scalar quantizer: f32 -> u8 with per-dimension min/max scaling.
///
/// Training learns the min/max value for each dimension, then quantizes
/// values to [0, 255] range. This achieves 4x compression with typically
/// >97% recall retention.
///
/// # Example
///
/// ```
/// use grafeo_core::index::vector::quantization::ScalarQuantizer;
///
/// // Training vectors
/// let vectors = vec![
///     vec![0.0f32, 0.5, 1.0],
///     vec![0.2, 0.3, 0.8],
///     vec![0.1, 0.6, 0.9],
/// ];
///
/// // Train quantizer
/// let refs: Vec<&[f32]> = vectors.iter().map(|v| v.as_slice()).collect();
/// let quantizer = ScalarQuantizer::train(&refs);
///
/// // Quantize a vector
/// let quantized = quantizer.quantize(&[0.1, 0.4, 0.85]);
/// assert_eq!(quantized.len(), 3);
///
/// // Compute approximate distance
/// let q2 = quantizer.quantize(&[0.15, 0.45, 0.9]);
/// let dist = quantizer.distance_squared_u8(&quantized, &q2);
/// assert!(dist < 1000.0);
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScalarQuantizer {
    /// Minimum value per dimension.
    min: Vec<f32>,
    /// Scale factor per dimension: 255 / (max - min).
    scale: Vec<f32>,
    /// Inverse scale for distance computation: (max - min) / 255.
    inv_scale: Vec<f32>,
    /// Number of dimensions.
    dimensions: usize,
}

impl ScalarQuantizer {
    /// Trains a scalar quantizer from sample vectors.
    ///
    /// Learns the min/max value per dimension from the training data.
    /// The more representative the training data, the better the quantization.
    ///
    /// # Arguments
    ///
    /// * `vectors` - Training vectors (should be representative of the dataset)
    ///
    /// # Panics
    ///
    /// Panics if `vectors` is empty or if vectors have different dimensions.
    #[must_use]
    pub fn train(vectors: &[&[f32]]) -> Self {
        assert!(!vectors.is_empty(), "Cannot train on empty vector set");

        let dimensions = vectors[0].len();
        assert!(
            vectors.iter().all(|v| v.len() == dimensions),
            "All training vectors must have the same dimensions"
        );

        // Find min/max per dimension
        let mut min = vec![f32::INFINITY; dimensions];
        let mut max = vec![f32::NEG_INFINITY; dimensions];

        for vec in vectors {
            for (i, &v) in vec.iter().enumerate() {
                min[i] = min[i].min(v);
                max[i] = max[i].max(v);
            }
        }

        // Compute scale factors (avoid division by zero)
        let (scale, inv_scale): (Vec<f32>, Vec<f32>) = min
            .iter()
            .zip(&max)
            .map(|(&mn, &mx)| {
                let range = mx - mn;
                if range.abs() < f32::EPSILON {
                    // All values are the same, use 1.0 as scale
                    (1.0, 1.0)
                } else {
                    (255.0 / range, range / 255.0)
                }
            })
            .unzip();

        Self {
            min,
            scale,
            inv_scale,
            dimensions,
        }
    }

    /// Creates a quantizer with explicit ranges (useful for testing).
    #[must_use]
    pub fn with_ranges(min: Vec<f32>, max: Vec<f32>) -> Self {
        let dimensions = min.len();
        assert_eq!(min.len(), max.len(), "Min and max must have same length");

        let (scale, inv_scale): (Vec<f32>, Vec<f32>) = min
            .iter()
            .zip(&max)
            .map(|(&mn, &mx)| {
                let range = mx - mn;
                if range.abs() < f32::EPSILON {
                    (1.0, 1.0)
                } else {
                    (255.0 / range, range / 255.0)
                }
            })
            .unzip();

        Self {
            min,
            scale,
            inv_scale,
            dimensions,
        }
    }

    /// Returns the number of dimensions.
    #[must_use]
    pub fn dimensions(&self) -> usize {
        self.dimensions
    }

    /// Returns the min values per dimension.
    #[must_use]
    pub fn min_values(&self) -> &[f32] {
        &self.min
    }

    /// Quantizes an f32 vector to u8.
    ///
    /// Values are clamped to the learned [min, max] range.
    #[must_use]
    pub fn quantize(&self, vector: &[f32]) -> Vec<u8> {
        debug_assert_eq!(
            vector.len(),
            self.dimensions,
            "Vector dimension mismatch: expected {}, got {}",
            self.dimensions,
            vector.len()
        );

        vector
            .iter()
            .enumerate()
            .map(|(i, &v)| {
                let normalized = (v - self.min[i]) * self.scale[i];
                normalized.clamp(0.0, 255.0) as u8
            })
            .collect()
    }

    /// Quantizes multiple vectors in batch.
    #[must_use]
    pub fn quantize_batch(&self, vectors: &[&[f32]]) -> Vec<Vec<u8>> {
        vectors.iter().map(|v| self.quantize(v)).collect()
    }

    /// Dequantizes a u8 vector back to f32 (approximate).
    #[must_use]
    pub fn dequantize(&self, quantized: &[u8]) -> Vec<f32> {
        debug_assert_eq!(quantized.len(), self.dimensions);

        quantized
            .iter()
            .enumerate()
            .map(|(i, &q)| self.min[i] + (q as f32) * self.inv_scale[i])
            .collect()
    }

    /// Computes squared Euclidean distance between quantized vectors.
    ///
    /// This is an approximation that works well for ranking nearest neighbors.
    /// The returned distance is scaled back to the original space.
    #[must_use]
    pub fn distance_squared_u8(&self, a: &[u8], b: &[u8]) -> f32 {
        debug_assert_eq!(a.len(), self.dimensions);
        debug_assert_eq!(b.len(), self.dimensions);

        // Compute in quantized space, then scale
        let mut sum = 0.0f32;
        for i in 0..a.len() {
            let diff = (a[i] as f32) - (b[i] as f32);
            sum += diff * diff * self.inv_scale[i] * self.inv_scale[i];
        }
        sum
    }

    /// Computes Euclidean distance between quantized vectors.
    #[must_use]
    #[inline]
    pub fn distance_u8(&self, a: &[u8], b: &[u8]) -> f32 {
        self.distance_squared_u8(a, b).sqrt()
    }

    /// Computes approximate cosine distance using quantized vectors.
    ///
    /// This is less accurate than exact computation but much faster.
    #[must_use]
    pub fn cosine_distance_u8(&self, a: &[u8], b: &[u8]) -> f32 {
        debug_assert_eq!(a.len(), self.dimensions);
        debug_assert_eq!(b.len(), self.dimensions);

        let mut dot = 0.0f32;
        let mut norm_a = 0.0f32;
        let mut norm_b = 0.0f32;

        for i in 0..a.len() {
            // Dequantize on the fly
            let va = self.min[i] + (a[i] as f32) * self.inv_scale[i];
            let vb = self.min[i] + (b[i] as f32) * self.inv_scale[i];

            dot += va * vb;
            norm_a += va * va;
            norm_b += vb * vb;
        }

        let denom = (norm_a * norm_b).sqrt();
        if denom < f32::EPSILON {
            1.0 // Maximum distance for zero vectors
        } else {
            1.0 - (dot / denom)
        }
    }

    /// Computes distance between a f32 query and a quantized vector.
    ///
    /// This is useful for search where we keep the query in full precision.
    #[must_use]
    pub fn asymmetric_distance_squared(&self, query: &[f32], quantized: &[u8]) -> f32 {
        debug_assert_eq!(query.len(), self.dimensions);
        debug_assert_eq!(quantized.len(), self.dimensions);

        let mut sum = 0.0f32;
        for i in 0..query.len() {
            // Dequantize the stored vector
            let dequant = self.min[i] + (quantized[i] as f32) * self.inv_scale[i];
            let diff = query[i] - dequant;
            sum += diff * diff;
        }
        sum
    }

    /// Computes asymmetric Euclidean distance.
    #[must_use]
    #[inline]
    pub fn asymmetric_distance(&self, query: &[f32], quantized: &[u8]) -> f32 {
        self.asymmetric_distance_squared(query, quantized).sqrt()
    }
}

// ============================================================================
// Binary Quantization
// ============================================================================

/// Binary quantizer: f32 -> 1 bit (sign only).
///
/// Provides extreme compression (32x) at the cost of accuracy (~80% recall).
/// Uses hamming distance for fast comparison. Best used with rescoring.
///
/// # Example
///
/// ```
/// use grafeo_core::index::vector::quantization::BinaryQuantizer;
///
/// let v1 = vec![0.5f32, -0.3, 0.0, 0.8, -0.1, 0.2, -0.4, 0.9];
/// let v2 = vec![0.4f32, -0.2, 0.1, 0.7, -0.2, 0.3, -0.3, 0.8];
///
/// let bits1 = BinaryQuantizer::quantize(&v1);
/// let bits2 = BinaryQuantizer::quantize(&v2);
///
/// let dist = BinaryQuantizer::hamming_distance(&bits1, &bits2);
/// // Vectors are similar, so hamming distance should be low
/// assert!(dist < 4);
/// ```
pub struct BinaryQuantizer;

impl BinaryQuantizer {
    /// Quantizes f32 vector to binary (sign bits packed in u64).
    ///
    /// Each f32 becomes 1 bit: 1 if >= 0, 0 if < 0.
    /// Bits are packed into u64 words (64 dimensions per word).
    #[must_use]
    pub fn quantize(vector: &[f32]) -> Vec<u64> {
        let num_words = (vector.len() + 63) / 64;
        let mut result = vec![0u64; num_words];

        for (i, &v) in vector.iter().enumerate() {
            if v >= 0.0 {
                result[i / 64] |= 1u64 << (i % 64);
            }
        }

        result
    }

    /// Quantizes multiple vectors in batch.
    #[must_use]
    pub fn quantize_batch(vectors: &[&[f32]]) -> Vec<Vec<u64>> {
        vectors.iter().map(|v| Self::quantize(v)).collect()
    }

    /// Computes hamming distance between binary vectors.
    ///
    /// Counts the number of differing bits. Lower = more similar.
    #[must_use]
    pub fn hamming_distance(a: &[u64], b: &[u64]) -> u32 {
        debug_assert_eq!(a.len(), b.len(), "Binary vectors must have same length");

        a.iter()
            .zip(b)
            .map(|(&x, &y)| (x ^ y).count_ones())
            .sum()
    }

    /// Computes normalized hamming distance (0.0 to 1.0).
    ///
    /// Returns the fraction of bits that differ.
    #[must_use]
    pub fn hamming_distance_normalized(a: &[u64], b: &[u64], dimensions: usize) -> f32 {
        let hamming = Self::hamming_distance(a, b);
        hamming as f32 / dimensions as f32
    }

    /// Estimates Euclidean distance from hamming distance.
    ///
    /// Uses an empirical approximation: d_euclidean ≈ sqrt(2 * hamming / dim).
    /// This is a rough estimate suitable for initial filtering.
    #[must_use]
    pub fn approximate_euclidean(a: &[u64], b: &[u64], dimensions: usize) -> f32 {
        let hamming = Self::hamming_distance(a, b);
        // Empirical approximation: assume values are roughly unit-normalized
        (2.0 * hamming as f32 / dimensions as f32).sqrt()
    }

    /// Returns the number of u64 words needed for the given dimensions.
    #[must_use]
    pub const fn words_needed(dimensions: usize) -> usize {
        (dimensions + 63) / 64
    }

    /// Returns the memory footprint in bytes for quantized storage.
    #[must_use]
    pub const fn bytes_needed(dimensions: usize) -> usize {
        Self::words_needed(dimensions) * 8
    }
}

// ============================================================================
// Product Quantization
// ============================================================================

/// Product quantizer: splits vectors into M subvectors, quantizes each to K centroids.
///
/// Product Quantization (PQ) provides excellent compression (8-32x) with ~90% recall.
/// It works by:
/// 1. Dividing vectors into M subvectors
/// 2. Learning K centroids (typically 256) for each subvector via k-means
/// 3. Storing each vector as M u8 codes (indices into centroid tables)
///
/// Distance computation uses asymmetric distance tables (ADC) for efficiency.
///
/// # Example
///
/// ```
/// use grafeo_core::index::vector::quantization::ProductQuantizer;
///
/// // Training vectors (16 dimensions, split into 4 subvectors)
/// let vectors: Vec<Vec<f32>> = (0..50)
///     .map(|i| (0..16).map(|j| (i + j) as f32 * 0.1).collect())
///     .collect();
/// let refs: Vec<&[f32]> = vectors.iter().map(|v| v.as_slice()).collect();
///
/// // Train quantizer with 4 subvectors, 8 centroids each
/// let quantizer = ProductQuantizer::train(&refs, 4, 8, 5);
///
/// // Quantize a vector to 4 u8 codes
/// let query = &vectors[0];
/// let codes = quantizer.quantize(query);
/// assert_eq!(codes.len(), 4);
///
/// // Each code is an index into the centroid table (0-7)
/// assert!(codes.iter().all(|&c| c < 8));
///
/// // Reconstruct approximate vector from codes
/// let reconstructed = quantizer.reconstruct(&codes);
/// assert_eq!(reconstructed.len(), 16);
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProductQuantizer {
    /// Number of subvectors (M).
    num_subvectors: usize,
    /// Number of centroids per subvector (K, typically 256 for u8 codes).
    num_centroids: usize,
    /// Dimensions per subvector.
    subvector_dim: usize,
    /// Total dimensions.
    dimensions: usize,
    /// Centroids: [M][K][subvector_dim] flattened to [M * K * subvector_dim].
    centroids: Vec<f32>,
}

impl ProductQuantizer {
    /// Trains a product quantizer from sample vectors using k-means clustering.
    ///
    /// # Arguments
    ///
    /// * `vectors` - Training vectors (should be representative of the dataset)
    /// * `num_subvectors` - Number of subvectors (M), must divide dimensions evenly
    /// * `num_centroids` - Number of centroids per subvector (K), typically 256
    /// * `iterations` - Number of k-means iterations (10-20 is usually sufficient)
    ///
    /// # Panics
    ///
    /// Panics if vectors is empty, dimensions not divisible by num_subvectors,
    /// or num_centroids > 256.
    #[must_use]
    pub fn train(
        vectors: &[&[f32]],
        num_subvectors: usize,
        num_centroids: usize,
        iterations: usize,
    ) -> Self {
        assert!(!vectors.is_empty(), "Cannot train on empty vector set");
        assert!(num_centroids <= 256, "num_centroids must be <= 256 for u8 codes");
        assert!(num_subvectors > 0, "num_subvectors must be > 0");

        let dimensions = vectors[0].len();
        assert!(
            dimensions.is_multiple_of(num_subvectors),
            "dimensions ({dimensions}) must be divisible by num_subvectors ({num_subvectors})"
        );
        assert!(
            vectors.iter().all(|v| v.len() == dimensions),
            "All training vectors must have the same dimensions"
        );

        let subvector_dim = dimensions / num_subvectors;

        // Train centroids for each subvector independently
        let mut centroids = Vec::with_capacity(num_subvectors * num_centroids * subvector_dim);

        for m in 0..num_subvectors {
            // Extract subvectors for this partition
            let subvectors: Vec<Vec<f32>> = vectors
                .iter()
                .map(|v| {
                    let start = m * subvector_dim;
                    let end = start + subvector_dim;
                    v[start..end].to_vec()
                })
                .collect();

            // Run k-means on this partition
            let partition_centroids =
                Self::kmeans(&subvectors, num_centroids, subvector_dim, iterations);

            centroids.extend(partition_centroids);
        }

        Self {
            num_subvectors,
            num_centroids,
            subvector_dim,
            dimensions,
            centroids,
        }
    }

    /// Simple k-means clustering implementation.
    fn kmeans(
        vectors: &[Vec<f32>],
        k: usize,
        dims: usize,
        iterations: usize,
    ) -> Vec<f32> {
        let n = vectors.len();

        // Initialize centroids using k-means++ style (first k vectors or random sampling)
        let actual_k = k.min(n);
        let mut centroids: Vec<f32> = if actual_k == n {
            vectors.iter().flat_map(|v| v.iter().copied()).collect()
        } else {
            // Take evenly spaced samples
            let step = n / actual_k;
            (0..actual_k)
                .flat_map(|i| vectors[i * step].iter().copied())
                .collect()
        };

        // Pad with zeros if we don't have enough training vectors
        if actual_k < k {
            centroids.resize(k * dims, 0.0);
        }

        let mut assignments = vec![0usize; n];
        let mut counts = vec![0usize; k];

        for _ in 0..iterations {
            // Assignment step: find nearest centroid for each vector
            for (i, vec) in vectors.iter().enumerate() {
                let mut best_dist = f32::INFINITY;
                let mut best_k = 0;

                for j in 0..k {
                    let centroid_start = j * dims;
                    let dist: f32 = vec
                        .iter()
                        .enumerate()
                        .map(|(d, &v)| {
                            let diff = v - centroids[centroid_start + d];
                            diff * diff
                        })
                        .sum();

                    if dist < best_dist {
                        best_dist = dist;
                        best_k = j;
                    }
                }

                assignments[i] = best_k;
            }

            // Update step: recompute centroids as mean of assigned vectors
            centroids.fill(0.0);
            counts.fill(0);

            for (i, vec) in vectors.iter().enumerate() {
                let k_idx = assignments[i];
                let centroid_start = k_idx * dims;
                counts[k_idx] += 1;

                for (d, &v) in vec.iter().enumerate() {
                    centroids[centroid_start + d] += v;
                }
            }

            // Divide by counts to get means
            for j in 0..k {
                if counts[j] > 0 {
                    let centroid_start = j * dims;
                    let count = counts[j] as f32;
                    for d in 0..dims {
                        centroids[centroid_start + d] /= count;
                    }
                }
            }
        }

        centroids
    }

    /// Creates a product quantizer with explicit centroids (for testing/loading).
    #[must_use]
    pub fn with_centroids(
        num_subvectors: usize,
        num_centroids: usize,
        dimensions: usize,
        centroids: Vec<f32>,
    ) -> Self {
        let subvector_dim = dimensions / num_subvectors;
        assert_eq!(
            centroids.len(),
            num_subvectors * num_centroids * subvector_dim,
            "Invalid centroid count"
        );

        Self {
            num_subvectors,
            num_centroids,
            subvector_dim,
            dimensions,
            centroids,
        }
    }

    /// Returns the number of subvectors (M).
    #[must_use]
    pub fn num_subvectors(&self) -> usize {
        self.num_subvectors
    }

    /// Returns the number of centroids per subvector (K).
    #[must_use]
    pub fn num_centroids(&self) -> usize {
        self.num_centroids
    }

    /// Returns the total dimensions.
    #[must_use]
    pub fn dimensions(&self) -> usize {
        self.dimensions
    }

    /// Returns the dimensions per subvector.
    #[must_use]
    pub fn subvector_dim(&self) -> usize {
        self.subvector_dim
    }

    /// Returns the memory footprint in bytes for a quantized vector.
    #[must_use]
    pub fn code_size(&self) -> usize {
        self.num_subvectors // M u8 codes
    }

    /// Returns the compression ratio compared to f32 storage.
    #[must_use]
    pub fn compression_ratio(&self) -> usize {
        // Original: dimensions * 4 bytes
        // Compressed: num_subvectors bytes
        (self.dimensions * 4) / self.num_subvectors
    }

    /// Quantizes a vector to M u8 codes.
    ///
    /// Each code is the index of the nearest centroid for that subvector.
    #[must_use]
    pub fn quantize(&self, vector: &[f32]) -> Vec<u8> {
        debug_assert_eq!(
            vector.len(),
            self.dimensions,
            "Vector dimension mismatch: expected {}, got {}",
            self.dimensions,
            vector.len()
        );

        let mut codes = Vec::with_capacity(self.num_subvectors);

        for m in 0..self.num_subvectors {
            let subvec_start = m * self.subvector_dim;
            let subvec = &vector[subvec_start..subvec_start + self.subvector_dim];

            // Find nearest centroid for this subvector
            let mut best_dist = f32::INFINITY;
            let mut best_k = 0u8;

            for k in 0..self.num_centroids {
                let centroid_start =
                    (m * self.num_centroids + k) * self.subvector_dim;
                let dist: f32 = subvec
                    .iter()
                    .enumerate()
                    .map(|(d, &v)| {
                        let diff = v - self.centroids[centroid_start + d];
                        diff * diff
                    })
                    .sum();

                if dist < best_dist {
                    best_dist = dist;
                    best_k = k as u8;
                }
            }

            codes.push(best_k);
        }

        codes
    }

    /// Quantizes multiple vectors in batch.
    #[must_use]
    pub fn quantize_batch(&self, vectors: &[&[f32]]) -> Vec<Vec<u8>> {
        vectors.iter().map(|v| self.quantize(v)).collect()
    }

    /// Builds asymmetric distance table for a query vector.
    ///
    /// Returns a table of shape [M][K] containing the squared distance
    /// from each query subvector to each centroid. This allows O(M) distance
    /// computation for quantized vectors via table lookups.
    #[must_use]
    pub fn build_distance_table(&self, query: &[f32]) -> Vec<f32> {
        debug_assert_eq!(query.len(), self.dimensions);

        let mut table = Vec::with_capacity(self.num_subvectors * self.num_centroids);

        for m in 0..self.num_subvectors {
            let query_start = m * self.subvector_dim;
            let query_subvec = &query[query_start..query_start + self.subvector_dim];

            for k in 0..self.num_centroids {
                let centroid_start =
                    (m * self.num_centroids + k) * self.subvector_dim;

                let dist: f32 = query_subvec
                    .iter()
                    .enumerate()
                    .map(|(d, &v)| {
                        let diff = v - self.centroids[centroid_start + d];
                        diff * diff
                    })
                    .sum();

                table.push(dist);
            }
        }

        table
    }

    /// Computes asymmetric squared distance using a precomputed distance table.
    ///
    /// This is O(M) - just M table lookups and additions.
    #[must_use]
    #[inline]
    pub fn distance_with_table(&self, table: &[f32], codes: &[u8]) -> f32 {
        debug_assert_eq!(codes.len(), self.num_subvectors);
        debug_assert_eq!(table.len(), self.num_subvectors * self.num_centroids);

        codes
            .iter()
            .enumerate()
            .map(|(m, &code)| table[m * self.num_centroids + code as usize])
            .sum()
    }

    /// Computes asymmetric squared distance from query to quantized vector.
    ///
    /// This builds the distance table on the fly - use `build_distance_table`
    /// and `distance_with_table` for batch queries.
    #[must_use]
    pub fn asymmetric_distance_squared(&self, query: &[f32], codes: &[u8]) -> f32 {
        let table = self.build_distance_table(query);
        self.distance_with_table(&table, codes)
    }

    /// Computes asymmetric distance (Euclidean).
    #[must_use]
    #[inline]
    pub fn asymmetric_distance(&self, query: &[f32], codes: &[u8]) -> f32 {
        self.asymmetric_distance_squared(query, codes).sqrt()
    }

    /// Reconstructs an approximate vector from codes.
    ///
    /// Returns the concatenated centroids for the given codes.
    #[must_use]
    pub fn reconstruct(&self, codes: &[u8]) -> Vec<f32> {
        debug_assert_eq!(codes.len(), self.num_subvectors);

        let mut result = Vec::with_capacity(self.dimensions);

        for (m, &code) in codes.iter().enumerate() {
            let centroid_start =
                (m * self.num_centroids + code as usize) * self.subvector_dim;
            result.extend_from_slice(
                &self.centroids[centroid_start..centroid_start + self.subvector_dim],
            );
        }

        result
    }

    /// Returns the centroid vectors for a specific subvector partition.
    #[must_use]
    pub fn get_partition_centroids(&self, partition: usize) -> Vec<&[f32]> {
        assert!(partition < self.num_subvectors);

        (0..self.num_centroids)
            .map(|k| {
                let start = (partition * self.num_centroids + k) * self.subvector_dim;
                &self.centroids[start..start + self.subvector_dim]
            })
            .collect()
    }
}

// ============================================================================
// SIMD-Accelerated Hamming Distance
// ============================================================================

/// Computes hamming distance with SIMD acceleration (if available).
///
/// On x86_64 with popcnt instruction, this is significantly faster than
/// the scalar implementation.
#[cfg(target_arch = "x86_64")]
#[must_use]
pub fn hamming_distance_simd(a: &[u64], b: &[u64]) -> u32 {
    // Use popcnt instruction if available (almost all modern CPUs)
    a.iter()
        .zip(b)
        .map(|(&x, &y)| {
            let xor = x ^ y;
            // Safety: popcnt is available on virtually all x86_64 CPUs since Nehalem (2008).
            // This is a well-understood CPU intrinsic with no memory safety implications.
            #[allow(unsafe_code)]
            unsafe {
                std::arch::x86_64::_popcnt64(xor as i64) as u32
            }
        })
        .sum()
}

/// Fallback scalar implementation.
#[cfg(not(target_arch = "x86_64"))]
#[must_use]
pub fn hamming_distance_simd(a: &[u64], b: &[u64]) -> u32 {
    BinaryQuantizer::hamming_distance(a, b)
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quantization_type_compression_ratio() {
        // Use 384 dimensions (common embedding size)
        let dims = 384;
        assert_eq!(QuantizationType::None.compression_ratio(dims), 1);
        assert_eq!(QuantizationType::Scalar.compression_ratio(dims), 4);
        assert_eq!(QuantizationType::Binary.compression_ratio(dims), 32);

        // Product quantization: (384 * 4) / 8 = 192x compression
        let pq8 = QuantizationType::Product { num_subvectors: 8 };
        assert_eq!(pq8.compression_ratio(dims), 192);

        // PQ16: (384 * 4) / 16 = 96x compression
        let pq16 = QuantizationType::Product { num_subvectors: 16 };
        assert_eq!(pq16.compression_ratio(dims), 96);
    }

    #[test]
    fn test_quantization_type_from_str() {
        assert_eq!(QuantizationType::from_str("none"), Some(QuantizationType::None));
        assert_eq!(QuantizationType::from_str("scalar"), Some(QuantizationType::Scalar));
        assert_eq!(QuantizationType::from_str("SQ"), Some(QuantizationType::Scalar));
        assert_eq!(QuantizationType::from_str("binary"), Some(QuantizationType::Binary));
        assert_eq!(QuantizationType::from_str("bit"), Some(QuantizationType::Binary));
        assert_eq!(QuantizationType::from_str("invalid"), None);
    }

    // ========================================================================
    // Scalar Quantization Tests
    // ========================================================================

    #[test]
    fn test_scalar_quantizer_train() {
        let vectors = [
            vec![0.0f32, 0.5, 1.0],
            vec![0.2, 0.3, 0.8],
            vec![0.1, 0.6, 0.9],
        ];
        let refs: Vec<&[f32]> = vectors.iter().map(|v| v.as_slice()).collect();

        let quantizer = ScalarQuantizer::train(&refs);

        assert_eq!(quantizer.dimensions(), 3);
        assert_eq!(quantizer.min_values()[0], 0.0);
        assert_eq!(quantizer.min_values()[1], 0.3);
        assert_eq!(quantizer.min_values()[2], 0.8);
    }

    #[test]
    fn test_scalar_quantizer_quantize() {
        let quantizer = ScalarQuantizer::with_ranges(vec![0.0, 0.0], vec![1.0, 1.0]);

        // Min value should quantize to 0
        let q_min = quantizer.quantize(&[0.0, 0.0]);
        assert_eq!(q_min, vec![0, 0]);

        // Max value should quantize to 255
        let q_max = quantizer.quantize(&[1.0, 1.0]);
        assert_eq!(q_max, vec![255, 255]);

        // Middle value should quantize to ~127
        let q_mid = quantizer.quantize(&[0.5, 0.5]);
        assert!(q_mid[0] >= 126 && q_mid[0] <= 128);
    }

    #[test]
    fn test_scalar_quantizer_dequantize() {
        let quantizer = ScalarQuantizer::with_ranges(vec![0.0], vec![1.0]);

        let original = [0.5f32];
        let quantized = quantizer.quantize(&original);
        let dequantized = quantizer.dequantize(&quantized);

        // Should be close to original (within quantization error)
        assert!((original[0] - dequantized[0]).abs() < 0.01);
    }

    #[test]
    fn test_scalar_quantizer_distance() {
        let quantizer = ScalarQuantizer::with_ranges(vec![0.0, 0.0], vec![1.0, 1.0]);

        let a = quantizer.quantize(&[0.0, 0.0]);
        let b = quantizer.quantize(&[1.0, 0.0]);

        let dist = quantizer.distance_u8(&a, &b);
        // Should be approximately 1.0 (the Euclidean distance in original space)
        assert!((dist - 1.0).abs() < 0.1);
    }

    #[test]
    fn test_scalar_quantizer_asymmetric_distance() {
        let quantizer = ScalarQuantizer::with_ranges(vec![0.0, 0.0], vec![1.0, 1.0]);

        let query = [0.0f32, 0.0];
        let stored = quantizer.quantize(&[1.0, 0.0]);

        let dist = quantizer.asymmetric_distance(&query, &stored);
        assert!((dist - 1.0).abs() < 0.1);
    }

    #[test]
    fn test_scalar_quantizer_cosine_distance() {
        let quantizer = ScalarQuantizer::with_ranges(vec![-1.0, -1.0], vec![1.0, 1.0]);

        // Orthogonal vectors
        let a = quantizer.quantize(&[1.0, 0.0]);
        let b = quantizer.quantize(&[0.0, 1.0]);

        let dist = quantizer.cosine_distance_u8(&a, &b);
        // Cosine distance of orthogonal vectors = 1.0
        assert!((dist - 1.0).abs() < 0.1);
    }

    #[test]
    #[should_panic(expected = "Cannot train on empty vector set")]
    fn test_scalar_quantizer_empty_training() {
        let vectors: Vec<&[f32]> = vec![];
        let _ = ScalarQuantizer::train(&vectors);
    }

    // ========================================================================
    // Binary Quantization Tests
    // ========================================================================

    #[test]
    fn test_binary_quantizer_quantize() {
        let v = vec![0.5f32, -0.3, 0.0, 0.8];
        let bits = BinaryQuantizer::quantize(&v);

        assert_eq!(bits.len(), 1); // 4 dims fit in 1 u64

        // Check individual bits: 0.5 >= 0 (1), -0.3 < 0 (0), 0.0 >= 0 (1), 0.8 >= 0 (1)
        // Expected bits (LSB first): 1, 0, 1, 1 = 0b1101 = 13
        assert_eq!(bits[0] & 0xF, 0b1101);
    }

    #[test]
    fn test_binary_quantizer_hamming_distance() {
        let v1 = vec![1.0f32, 1.0, 1.0, 1.0]; // All positive: 1111
        let v2 = vec![1.0f32, -1.0, 1.0, -1.0]; // Mixed: 1010

        let bits1 = BinaryQuantizer::quantize(&v1);
        let bits2 = BinaryQuantizer::quantize(&v2);

        let dist = BinaryQuantizer::hamming_distance(&bits1, &bits2);
        assert_eq!(dist, 2); // Two bits differ
    }

    #[test]
    fn test_binary_quantizer_identical_vectors() {
        let v = vec![0.1f32, -0.2, 0.3, -0.4, 0.5];
        let bits = BinaryQuantizer::quantize(&v);

        let dist = BinaryQuantizer::hamming_distance(&bits, &bits);
        assert_eq!(dist, 0);
    }

    #[test]
    fn test_binary_quantizer_opposite_vectors() {
        let v1 = vec![1.0f32; 64];
        let v2 = vec![-1.0f32; 64];

        let bits1 = BinaryQuantizer::quantize(&v1);
        let bits2 = BinaryQuantizer::quantize(&v2);

        let dist = BinaryQuantizer::hamming_distance(&bits1, &bits2);
        assert_eq!(dist, 64); // All bits differ
    }

    #[test]
    fn test_binary_quantizer_large_vector() {
        let v: Vec<f32> = (0..1000).map(|i| if i % 2 == 0 { 1.0 } else { -1.0 }).collect();
        let bits = BinaryQuantizer::quantize(&v);

        // 1000 dims needs ceil(1000/64) = 16 words
        assert_eq!(bits.len(), 16);
    }

    #[test]
    fn test_binary_quantizer_normalized_distance() {
        let v1 = vec![1.0f32; 100];
        let v2 = vec![-1.0f32; 100];

        let bits1 = BinaryQuantizer::quantize(&v1);
        let bits2 = BinaryQuantizer::quantize(&v2);

        let norm_dist = BinaryQuantizer::hamming_distance_normalized(&bits1, &bits2, 100);
        assert!((norm_dist - 1.0).abs() < 0.01); // All bits differ
    }

    #[test]
    fn test_binary_quantizer_words_needed() {
        assert_eq!(BinaryQuantizer::words_needed(1), 1);
        assert_eq!(BinaryQuantizer::words_needed(64), 1);
        assert_eq!(BinaryQuantizer::words_needed(65), 2);
        assert_eq!(BinaryQuantizer::words_needed(128), 2);
        assert_eq!(BinaryQuantizer::words_needed(1536), 24); // OpenAI embedding size
    }

    #[test]
    fn test_binary_quantizer_bytes_needed() {
        // Each u64 is 8 bytes
        assert_eq!(BinaryQuantizer::bytes_needed(64), 8);
        assert_eq!(BinaryQuantizer::bytes_needed(128), 16);
        assert_eq!(BinaryQuantizer::bytes_needed(1536), 192); // vs 6144 for f32
    }

    // ========================================================================
    // SIMD Tests
    // ========================================================================

    #[test]
    fn test_hamming_distance_simd() {
        let a = vec![0xFFFF_FFFF_FFFF_FFFFu64, 0x0000_0000_0000_0000];
        let b = vec![0x0000_0000_0000_0000u64, 0xFFFF_FFFF_FFFF_FFFF];

        let dist = hamming_distance_simd(&a, &b);
        assert_eq!(dist, 128); // All 128 bits differ
    }

    // ========================================================================
    // Product Quantization Tests
    // ========================================================================

    #[test]
    fn test_quantization_type_product_from_str() {
        // Basic PQ parsing
        assert_eq!(
            QuantizationType::from_str("pq"),
            Some(QuantizationType::Product { num_subvectors: 8 })
        );
        assert_eq!(
            QuantizationType::from_str("product"),
            Some(QuantizationType::Product { num_subvectors: 8 })
        );

        // PQ with specific subvector count
        assert_eq!(
            QuantizationType::from_str("pq8"),
            Some(QuantizationType::Product { num_subvectors: 8 })
        );
        assert_eq!(
            QuantizationType::from_str("pq16"),
            Some(QuantizationType::Product { num_subvectors: 16 })
        );
        assert_eq!(
            QuantizationType::from_str("pq32"),
            Some(QuantizationType::Product { num_subvectors: 32 })
        );
    }

    #[test]
    fn test_quantization_type_requires_training() {
        assert!(!QuantizationType::None.requires_training());
        assert!(QuantizationType::Scalar.requires_training());
        assert!(!QuantizationType::Binary.requires_training());
        assert!(QuantizationType::Product { num_subvectors: 8 }.requires_training());
    }

    #[test]
    fn test_product_quantizer_train() {
        // Create 100 training vectors with 16 dimensions
        let vectors: Vec<Vec<f32>> = (0..100)
            .map(|i| (0..16).map(|j| ((i * j) as f32 * 0.01).sin()).collect())
            .collect();
        let refs: Vec<&[f32]> = vectors.iter().map(|v| v.as_slice()).collect();

        // Train with 4 subvectors (16/4 = 4 dims each), 8 centroids
        let pq = ProductQuantizer::train(&refs, 4, 8, 5);

        assert_eq!(pq.num_subvectors(), 4);
        assert_eq!(pq.num_centroids(), 8);
        assert_eq!(pq.dimensions(), 16);
        assert_eq!(pq.subvector_dim(), 4);
        assert_eq!(pq.code_size(), 4);
    }

    #[test]
    fn test_product_quantizer_quantize() {
        let vectors: Vec<Vec<f32>> = (0..50)
            .map(|i| (0..8).map(|j| ((i * j) as f32 * 0.1).cos()).collect())
            .collect();
        let refs: Vec<&[f32]> = vectors.iter().map(|v| v.as_slice()).collect();

        let pq = ProductQuantizer::train(&refs, 2, 16, 3);

        // Quantize a vector
        let codes = pq.quantize(&vectors[0]);
        assert_eq!(codes.len(), 2);

        // All codes should be < num_centroids
        for &code in &codes {
            assert!(code < 16);
        }
    }

    #[test]
    fn test_product_quantizer_reconstruct() {
        let vectors: Vec<Vec<f32>> = (0..50)
            .map(|i| (0..12).map(|j| (i + j) as f32 * 0.05).collect())
            .collect();
        let refs: Vec<&[f32]> = vectors.iter().map(|v| v.as_slice()).collect();

        let pq = ProductQuantizer::train(&refs, 3, 8, 5);

        // Quantize and reconstruct
        let original = &vectors[10];
        let codes = pq.quantize(original);
        let reconstructed = pq.reconstruct(&codes);

        assert_eq!(reconstructed.len(), 12);

        // Reconstructed should be somewhat close to original (not exact due to quantization)
        let error: f32 = original
            .iter()
            .zip(&reconstructed)
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f32>()
            .sqrt();

        // Error should be bounded (not zero, but reasonable)
        assert!(error < 2.0, "Reconstruction error too high: {error}");
    }

    #[test]
    fn test_product_quantizer_asymmetric_distance() {
        let vectors: Vec<Vec<f32>> = (0..100)
            .map(|i| (0..32).map(|j| ((i * j) as f32 * 0.01).sin()).collect())
            .collect();
        let refs: Vec<&[f32]> = vectors.iter().map(|v| v.as_slice()).collect();

        let pq = ProductQuantizer::train(&refs, 8, 32, 5);

        // Distance to self should be small
        let query = &vectors[0];
        let codes = pq.quantize(query);
        let self_dist = pq.asymmetric_distance(query, &codes);
        assert!(self_dist < 1.0, "Self-distance too high: {self_dist}");

        // Distance to different vector should be larger
        let other_codes = pq.quantize(&vectors[50]);
        let other_dist = pq.asymmetric_distance(query, &other_codes);
        assert!(other_dist > self_dist, "Other vector should be farther");
    }

    #[test]
    fn test_product_quantizer_distance_table() {
        let vectors: Vec<Vec<f32>> = (0..50)
            .map(|i| (0..16).map(|j| (i + j) as f32 * 0.02).collect())
            .collect();
        let refs: Vec<&[f32]> = vectors.iter().map(|v| v.as_slice()).collect();

        let pq = ProductQuantizer::train(&refs, 4, 8, 3);

        let query = &vectors[0];
        let table = pq.build_distance_table(query);

        // Table should have M * K entries
        assert_eq!(table.len(), 4 * 8);

        // Distance via table should match direct computation
        let codes = pq.quantize(&vectors[5]);
        let dist_direct = pq.asymmetric_distance_squared(query, &codes);
        let dist_table = pq.distance_with_table(&table, &codes);

        assert!((dist_direct - dist_table).abs() < 0.001);
    }

    #[test]
    fn test_product_quantizer_batch() {
        let vectors: Vec<Vec<f32>> = (0..20)
            .map(|i| (0..8).map(|j| (i + j) as f32).collect())
            .collect();
        let refs: Vec<&[f32]> = vectors.iter().map(|v| v.as_slice()).collect();

        let pq = ProductQuantizer::train(&refs, 2, 4, 2);

        let batch_codes = pq.quantize_batch(&refs[0..5]);
        assert_eq!(batch_codes.len(), 5);

        for codes in &batch_codes {
            assert_eq!(codes.len(), 2);
        }
    }

    #[test]
    fn test_product_quantizer_compression_ratio() {
        let vectors: Vec<Vec<f32>> = (0..50)
            .map(|i| (0..384).map(|j| ((i * j) as f32).sin()).collect())
            .collect();
        let refs: Vec<&[f32]> = vectors.iter().map(|v| v.as_slice()).collect();

        // PQ8: 384 dims split into 8 subvectors
        let pq8 = ProductQuantizer::train(&refs, 8, 256, 3);
        assert_eq!(pq8.compression_ratio(), 192); // (384 * 4) / 8 = 192

        // PQ48: 384 dims split into 48 subvectors (8 dims each)
        let pq48 = ProductQuantizer::train(&refs, 48, 256, 3);
        assert_eq!(pq48.compression_ratio(), 32); // (384 * 4) / 48 = 32
    }

    #[test]
    #[should_panic(expected = "dimensions (15) must be divisible by num_subvectors (4)")]
    fn test_product_quantizer_invalid_dimensions() {
        let vectors: Vec<Vec<f32>> = (0..10)
            .map(|i| (0..15).map(|j| (i + j) as f32).collect())
            .collect();
        let refs: Vec<&[f32]> = vectors.iter().map(|v| v.as_slice()).collect();

        // 15 is not divisible by 4
        let _ = ProductQuantizer::train(&refs, 4, 8, 3);
    }

    #[test]
    fn test_product_quantizer_get_partition_centroids() {
        let vectors: Vec<Vec<f32>> = (0..30)
            .map(|i| (0..8).map(|j| (i + j) as f32 * 0.1).collect())
            .collect();
        let refs: Vec<&[f32]> = vectors.iter().map(|v| v.as_slice()).collect();

        let pq = ProductQuantizer::train(&refs, 2, 4, 3);

        // Get centroids for first partition
        let centroids = pq.get_partition_centroids(0);
        assert_eq!(centroids.len(), 4); // 4 centroids
        assert_eq!(centroids[0].len(), 4); // 4 dims per subvector (8/2)
    }
}
