//! Python bindings for vector quantization.
//!
//! Provides Python access to scalar and product quantization for
//! memory-efficient vector storage.

use grafeo_core::index::vector::{
    BinaryQuantizer, ProductQuantizer, QuantizationType, ScalarQuantizer,
};
use pyo3::prelude::*;

// ============================================================================
// Quantization Type
// ============================================================================

/// Quantization type for vectors.
///
/// - "none": Full f32 precision (no compression)
/// - "scalar": f32 -> u8, 4x compression, ~97% accuracy
/// - "binary": f32 -> 1 bit, 32x compression, ~80% accuracy
/// - "product" or "pq": Product quantization, 8-32x compression, ~90% accuracy
///
/// Example:
///     quant_type = grafeo.QuantizationType.from_str("scalar")
///     print(quant_type.name())  # "scalar"
///     print(quant_type.compression_ratio(384))  # 4
#[pyclass(name = "QuantizationType")]
#[derive(Clone)]
pub struct PyQuantizationType {
    inner: QuantizationType,
}

#[pymethods]
impl PyQuantizationType {
    /// Creates a None (no compression) quantization type.
    #[staticmethod]
    fn none() -> Self {
        Self {
            inner: QuantizationType::None,
        }
    }

    /// Creates a Scalar quantization type (4x compression).
    #[staticmethod]
    fn scalar() -> Self {
        Self {
            inner: QuantizationType::Scalar,
        }
    }

    /// Creates a Binary quantization type (32x compression).
    #[staticmethod]
    fn binary() -> Self {
        Self {
            inner: QuantizationType::Binary,
        }
    }

    /// Creates a Product quantization type with the given number of subvectors.
    ///
    /// Args:
    ///     num_subvectors: Number of subvectors (typically 8, 16, 32, 64).
    ///                     Must divide the vector dimensions evenly.
    #[staticmethod]
    fn product(num_subvectors: usize) -> Self {
        Self {
            inner: QuantizationType::Product { num_subvectors },
        }
    }

    /// Parses a quantization type from string.
    ///
    /// Valid values: "none", "scalar", "binary", "product", "pq", "pq8", "pq16", etc.
    ///
    /// Returns None if the string is invalid.
    #[staticmethod]
    fn from_str(s: &str) -> Option<Self> {
        QuantizationType::from_str(s).map(|inner| Self { inner })
    }

    /// Returns the name of this quantization type.
    fn name(&self) -> &'static str {
        self.inner.name()
    }

    /// Returns the compression ratio for the given dimensions.
    ///
    /// For product quantization, this depends on the number of dimensions.
    fn compression_ratio(&self, dimensions: usize) -> usize {
        self.inner.compression_ratio(dimensions)
    }

    /// Returns True if this quantization type requires training.
    fn requires_training(&self) -> bool {
        self.inner.requires_training()
    }

    fn __repr__(&self) -> String {
        format!("QuantizationType.{}()", self.inner.name())
    }

    fn __str__(&self) -> String {
        self.inner.name().to_string()
    }
}

// ============================================================================
// Scalar Quantizer
// ============================================================================

/// Scalar quantizer: converts f32 vectors to u8 with per-dimension scaling.
///
/// Achieves 4x compression with typically >97% accuracy retention.
///
/// Example:
///     # Train on sample vectors
///     vectors = [[0.1, 0.5, 0.9], [0.2, 0.3, 0.8], [0.0, 0.6, 1.0]]
///     quantizer = grafeo.ScalarQuantizer.train(vectors)
///
///     # Quantize a vector
///     quantized = quantizer.quantize([0.15, 0.45, 0.85])
///     print(quantized)  # [38, 115, 217] (u8 values)
///
///     # Compute distance
///     q2 = quantizer.quantize([0.2, 0.5, 0.9])
///     dist = quantizer.distance(quantized, q2)
#[pyclass(name = "ScalarQuantizer")]
pub struct PyScalarQuantizer {
    inner: ScalarQuantizer,
}

#[pymethods]
impl PyScalarQuantizer {
    /// Trains a scalar quantizer from sample vectors.
    ///
    /// Args:
    ///     vectors: List of vectors (list of lists or numpy arrays).
    ///
    /// Returns:
    ///     A trained ScalarQuantizer.
    #[staticmethod]
    fn train(vectors: Vec<Vec<f32>>) -> PyResult<Self> {
        if vectors.is_empty() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "Cannot train on empty vector set",
            ));
        }

        let refs: Vec<&[f32]> = vectors.iter().map(|v| v.as_slice()).collect();
        let inner = ScalarQuantizer::train(&refs);

        Ok(Self { inner })
    }

    /// Creates a quantizer with explicit min/max ranges per dimension.
    ///
    /// Args:
    ///     min_values: Minimum value per dimension.
    ///     max_values: Maximum value per dimension.
    #[staticmethod]
    fn with_ranges(min_values: Vec<f32>, max_values: Vec<f32>) -> PyResult<Self> {
        if min_values.len() != max_values.len() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "min_values and max_values must have same length",
            ));
        }

        Ok(Self {
            inner: ScalarQuantizer::with_ranges(min_values, max_values),
        })
    }

    /// Returns the number of dimensions.
    fn dimensions(&self) -> usize {
        self.inner.dimensions()
    }

    /// Quantizes an f32 vector to u8.
    fn quantize(&self, vector: Vec<f32>) -> Vec<u8> {
        self.inner.quantize(&vector)
    }

    /// Quantizes multiple vectors in batch.
    fn quantize_batch(&self, vectors: Vec<Vec<f32>>) -> Vec<Vec<u8>> {
        let refs: Vec<&[f32]> = vectors.iter().map(|v| v.as_slice()).collect();
        self.inner.quantize_batch(&refs)
    }

    /// Dequantizes a u8 vector back to f32 (approximate).
    fn dequantize(&self, quantized: Vec<u8>) -> Vec<f32> {
        self.inner.dequantize(&quantized)
    }

    /// Computes Euclidean distance between quantized vectors.
    fn distance(&self, a: Vec<u8>, b: Vec<u8>) -> f32 {
        self.inner.distance_u8(&a, &b)
    }

    /// Computes squared Euclidean distance between quantized vectors.
    fn distance_squared(&self, a: Vec<u8>, b: Vec<u8>) -> f32 {
        self.inner.distance_squared_u8(&a, &b)
    }

    /// Computes cosine distance between quantized vectors.
    fn cosine_distance(&self, a: Vec<u8>, b: Vec<u8>) -> f32 {
        self.inner.cosine_distance_u8(&a, &b)
    }

    /// Computes asymmetric distance (f32 query vs u8 stored).
    fn asymmetric_distance(&self, query: Vec<f32>, quantized: Vec<u8>) -> f32 {
        self.inner.asymmetric_distance(&query, &quantized)
    }

    fn __repr__(&self) -> String {
        format!("ScalarQuantizer(dimensions={})", self.dimensions())
    }
}

// ============================================================================
// Product Quantizer
// ============================================================================

/// Product quantizer: splits vectors into subvectors and quantizes each.
///
/// Achieves 8-32x compression with ~90% accuracy. Best for large datasets
/// where memory is constrained.
///
/// Example:
///     # Generate training vectors (384 dimensions)
///     import random
///     vectors = [[random.random() for _ in range(384)] for _ in range(1000)]
///
///     # Train with 8 subvectors, 256 centroids
///     quantizer = grafeo.ProductQuantizer.train(vectors, num_subvectors=8)
///
///     # Quantize a vector to 8 u8 codes
///     codes = quantizer.quantize(vectors[0])
///     print(len(codes))  # 8
///
///     # Compute distance
///     dist = quantizer.asymmetric_distance(vectors[0], codes)
#[pyclass(name = "ProductQuantizer")]
pub struct PyProductQuantizer {
    inner: ProductQuantizer,
}

#[pymethods]
impl PyProductQuantizer {
    /// Trains a product quantizer from sample vectors.
    ///
    /// Args:
    ///     vectors: Training vectors (list of lists).
    ///     num_subvectors: Number of subvectors (M). Must divide dimensions evenly.
    ///                     Typical values: 8, 16, 32, 64.
    ///     num_centroids: Centroids per subvector (K). Default 256 for u8 codes.
    ///     iterations: K-means iterations. Default 10.
    ///
    /// Returns:
    ///     A trained ProductQuantizer.
    #[staticmethod]
    #[pyo3(signature = (vectors, num_subvectors, num_centroids=256, iterations=10))]
    fn train(
        vectors: Vec<Vec<f32>>,
        num_subvectors: usize,
        num_centroids: usize,
        iterations: usize,
    ) -> PyResult<Self> {
        if vectors.is_empty() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "Cannot train on empty vector set",
            ));
        }

        if num_centroids > 256 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "num_centroids must be <= 256 for u8 codes",
            ));
        }

        let dims = vectors[0].len();
        if !dims.is_multiple_of(num_subvectors) {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "dimensions ({}) must be divisible by num_subvectors ({})",
                dims, num_subvectors
            )));
        }

        let refs: Vec<&[f32]> = vectors.iter().map(|v| v.as_slice()).collect();
        let inner = ProductQuantizer::train(&refs, num_subvectors, num_centroids, iterations);

        Ok(Self { inner })
    }

    /// Returns the number of subvectors (M).
    fn num_subvectors(&self) -> usize {
        self.inner.num_subvectors()
    }

    /// Returns the number of centroids per subvector (K).
    fn num_centroids(&self) -> usize {
        self.inner.num_centroids()
    }

    /// Returns the total dimensions.
    fn dimensions(&self) -> usize {
        self.inner.dimensions()
    }

    /// Returns the dimensions per subvector.
    fn subvector_dim(&self) -> usize {
        self.inner.subvector_dim()
    }

    /// Returns the size of quantized codes in bytes.
    fn code_size(&self) -> usize {
        self.inner.code_size()
    }

    /// Returns the compression ratio compared to f32 storage.
    fn compression_ratio(&self) -> usize {
        self.inner.compression_ratio()
    }

    /// Quantizes a vector to M u8 codes.
    fn quantize(&self, vector: Vec<f32>) -> Vec<u8> {
        self.inner.quantize(&vector)
    }

    /// Quantizes multiple vectors in batch.
    fn quantize_batch(&self, vectors: Vec<Vec<f32>>) -> Vec<Vec<u8>> {
        let refs: Vec<&[f32]> = vectors.iter().map(|v| v.as_slice()).collect();
        self.inner.quantize_batch(&refs)
    }

    /// Reconstructs an approximate vector from codes.
    fn reconstruct(&self, codes: Vec<u8>) -> Vec<f32> {
        self.inner.reconstruct(&codes)
    }

    /// Computes asymmetric distance from query to quantized vector.
    fn asymmetric_distance(&self, query: Vec<f32>, codes: Vec<u8>) -> f32 {
        self.inner.asymmetric_distance(&query, &codes)
    }

    /// Computes asymmetric squared distance.
    fn asymmetric_distance_squared(&self, query: Vec<f32>, codes: Vec<u8>) -> f32 {
        self.inner.asymmetric_distance_squared(&query, &codes)
    }

    /// Builds a distance table for efficient batch distance computation.
    ///
    /// Returns a flattened table that can be used with distance_with_table().
    fn build_distance_table(&self, query: Vec<f32>) -> Vec<f32> {
        self.inner.build_distance_table(&query)
    }

    /// Computes distance using a precomputed table (fast for batch queries).
    fn distance_with_table(&self, table: Vec<f32>, codes: Vec<u8>) -> f32 {
        self.inner.distance_with_table(&table, &codes).sqrt()
    }

    fn __repr__(&self) -> String {
        format!(
            "ProductQuantizer(dimensions={}, subvectors={}, centroids={}, compression={}x)",
            self.dimensions(),
            self.num_subvectors(),
            self.num_centroids(),
            self.compression_ratio()
        )
    }
}

// ============================================================================
// Binary Quantizer
// ============================================================================

/// Binary quantizer: converts f32 vectors to sign bits.
///
/// Achieves 32x compression with ~80% accuracy. Extremely fast hamming distance.
/// Best used with rescoring.
///
/// Example:
///     # Quantize vectors to binary
///     bits1 = grafeo.BinaryQuantizer.quantize([0.5, -0.3, 0.0, 0.8])
///     bits2 = grafeo.BinaryQuantizer.quantize([0.4, -0.2, 0.1, 0.7])
///
///     # Compute hamming distance
///     dist = grafeo.BinaryQuantizer.hamming_distance(bits1, bits2)
#[pyclass(name = "BinaryQuantizer")]
pub struct PyBinaryQuantizer;

#[pymethods]
impl PyBinaryQuantizer {
    /// Quantizes a vector to binary (sign bits packed in u64).
    #[staticmethod]
    fn quantize(vector: Vec<f32>) -> Vec<u64> {
        BinaryQuantizer::quantize(&vector)
    }

    /// Quantizes multiple vectors in batch.
    #[staticmethod]
    fn quantize_batch(vectors: Vec<Vec<f32>>) -> Vec<Vec<u64>> {
        let refs: Vec<&[f32]> = vectors.iter().map(|v| v.as_slice()).collect();
        BinaryQuantizer::quantize_batch(&refs)
    }

    /// Computes hamming distance between binary vectors.
    #[staticmethod]
    fn hamming_distance(a: Vec<u64>, b: Vec<u64>) -> u32 {
        BinaryQuantizer::hamming_distance(&a, &b)
    }

    /// Computes normalized hamming distance (0.0 to 1.0).
    #[staticmethod]
    fn hamming_distance_normalized(a: Vec<u64>, b: Vec<u64>, dimensions: usize) -> f32 {
        BinaryQuantizer::hamming_distance_normalized(&a, &b, dimensions)
    }

    /// Estimates Euclidean distance from hamming distance.
    #[staticmethod]
    fn approximate_euclidean(a: Vec<u64>, b: Vec<u64>, dimensions: usize) -> f32 {
        BinaryQuantizer::approximate_euclidean(&a, &b, dimensions)
    }

    /// Returns the number of u64 words needed for the given dimensions.
    #[staticmethod]
    fn words_needed(dimensions: usize) -> usize {
        BinaryQuantizer::words_needed(dimensions)
    }

    /// Returns the memory footprint in bytes for quantized storage.
    #[staticmethod]
    fn bytes_needed(dimensions: usize) -> usize {
        BinaryQuantizer::bytes_needed(dimensions)
    }
}

// ============================================================================
// Module Registration
// ============================================================================

/// Registers quantization types in the grafeo module.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyQuantizationType>()?;
    m.add_class::<PyScalarQuantizer>()?;
    m.add_class::<PyProductQuantizer>()?;
    m.add_class::<PyBinaryQuantizer>()?;
    Ok(())
}
