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
}

impl QuantizationType {
    /// Returns the compression ratio (memory reduction factor).
    #[must_use]
    pub const fn compression_ratio(&self) -> usize {
        match self {
            Self::None => 1,
            Self::Scalar => 4,  // f32 (4 bytes) -> u8 (1 byte)
            Self::Binary => 32, // f32 (4 bytes) -> 1 bit (0.125 bytes)
        }
    }

    /// Returns the name of the quantization type.
    #[must_use]
    pub const fn name(&self) -> &'static str {
        match self {
            Self::None => "none",
            Self::Scalar => "scalar",
            Self::Binary => "binary",
        }
    }

    /// Parses from string (case-insensitive).
    #[must_use]
    pub fn from_str(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "none" | "full" | "f32" => Some(Self::None),
            "scalar" | "sq" | "u8" | "int8" => Some(Self::Scalar),
            "binary" | "bin" | "bit" | "1bit" => Some(Self::Binary),
            _ => None,
        }
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
        assert_eq!(QuantizationType::None.compression_ratio(), 1);
        assert_eq!(QuantizationType::Scalar.compression_ratio(), 4);
        assert_eq!(QuantizationType::Binary.compression_ratio(), 32);
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
        let vectors = vec![
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
}
