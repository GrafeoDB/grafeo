//! Distance metrics for vector similarity search.
//!
//! Provides efficient computation of various distance metrics between vectors.
//! All functions expect vectors of equal length.
//!
//! # SIMD Acceleration
//!
//! This module automatically uses SIMD instructions when available:
//! - **AVX2** (x86_64): 8 floats per instruction, ~6x speedup
//! - **SSE** (x86_64): 4 floats per instruction, ~3x speedup
//! - **NEON** (aarch64): 4 floats per instruction, ~3x speedup
//!
//! Use [`simd_support`] to check which instruction set is being used.

use serde::{Deserialize, Serialize};

use super::simd;

/// Distance metric for vector similarity computation.
///
/// Different metrics are suited for different embedding types:
/// - **Cosine**: Best for normalized embeddings (most text embeddings)
/// - **Euclidean**: Best for raw embeddings where magnitude matters
/// - **DotProduct**: Best for maximum inner product search
/// - **Manhattan**: Alternative to Euclidean, less sensitive to outliers
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
pub enum DistanceMetric {
    /// Cosine distance: 1 - cosine_similarity.
    ///
    /// Range: [0, 2], where 0 = identical direction, 2 = opposite direction.
    /// Best for normalized embeddings (most text/sentence embeddings).
    #[default]
    Cosine,

    /// Euclidean (L2) distance: `sqrt(sum((a[i] - b[i])^2))`.
    ///
    /// Range: [0, infinity), where 0 = identical vectors.
    /// Best when magnitude matters.
    Euclidean,

    /// Negative dot product: `-sum(a[i] * b[i])`.
    ///
    /// Returns negative so that smaller = more similar (for min-heap).
    /// Best for maximum inner product search (MIPS).
    DotProduct,

    /// Manhattan (L1) distance: `sum(|a[i] - b[i]|)`.
    ///
    /// Range: [0, infinity), where 0 = identical vectors.
    /// Less sensitive to outliers than Euclidean.
    Manhattan,
}

impl DistanceMetric {
    /// Returns the name of the metric as a string.
    #[must_use]
    pub const fn name(&self) -> &'static str {
        match self {
            Self::Cosine => "cosine",
            Self::Euclidean => "euclidean",
            Self::DotProduct => "dot_product",
            Self::Manhattan => "manhattan",
        }
    }

    /// Parses a metric from a string (case-insensitive).
    ///
    /// # Examples
    ///
    /// ```
    /// use grafeo_core::index::vector::DistanceMetric;
    ///
    /// assert_eq!(DistanceMetric::from_str("cosine"), Some(DistanceMetric::Cosine));
    /// assert_eq!(DistanceMetric::from_str("EUCLIDEAN"), Some(DistanceMetric::Euclidean));
    /// assert_eq!(DistanceMetric::from_str("l2"), Some(DistanceMetric::Euclidean));
    /// assert_eq!(DistanceMetric::from_str("invalid"), None);
    /// ```
    #[must_use]
    pub fn from_str(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "cosine" | "cos" => Some(Self::Cosine),
            "euclidean" | "l2" | "euclid" => Some(Self::Euclidean),
            "dot_product" | "dotproduct" | "dot" | "inner_product" | "ip" => Some(Self::DotProduct),
            "manhattan" | "l1" | "taxicab" => Some(Self::Manhattan),
            _ => None,
        }
    }
}

/// Returns the active SIMD instruction set name.
///
/// Useful for diagnostics and performance tuning.
///
/// # Returns
///
/// One of: `"avx2"`, `"sse"`, `"neon"`, or `"scalar"`.
///
/// # Examples
///
/// ```
/// use grafeo_core::index::vector::simd_support;
///
/// let support = simd_support();
/// println!("Using SIMD: {}", support);
/// ```
#[must_use]
#[inline]
pub fn simd_support() -> &'static str {
    simd::simd_support()
}

/// Computes the distance between two vectors using the specified metric.
///
/// This function automatically uses SIMD acceleration when available,
/// providing 3-6x speedup on modern CPUs.
///
/// # Panics
///
/// Debug-asserts that vectors have equal length. In release builds,
/// mismatched lengths may cause incorrect results.
///
/// # Examples
///
/// ```
/// use grafeo_core::index::vector::{compute_distance, DistanceMetric};
///
/// let a = [1.0f32, 0.0, 0.0];
/// let b = [0.0f32, 1.0, 0.0];
///
/// // Cosine distance between orthogonal vectors = 1.0
/// let dist = compute_distance(&a, &b, DistanceMetric::Cosine);
/// assert!((dist - 1.0).abs() < 0.001);
///
/// // Euclidean distance = sqrt(2)
/// let dist = compute_distance(&a, &b, DistanceMetric::Euclidean);
/// assert!((dist - 1.414).abs() < 0.01);
/// ```
#[inline]
pub fn compute_distance(a: &[f32], b: &[f32], metric: DistanceMetric) -> f32 {
    simd::compute_distance_simd(a, b, metric)
}

/// Computes cosine distance: 1 - cosine_similarity.
///
/// Cosine similarity = dot(a, b) / (||a|| * ||b||)
/// Cosine distance = 1 - cosine_similarity
///
/// Range: [0, 2] where 0 = same direction, 1 = orthogonal, 2 = opposite.
///
/// Uses SIMD acceleration when available.
#[inline]
pub fn cosine_distance(a: &[f32], b: &[f32]) -> f32 {
    simd::cosine_distance_simd(a, b)
}

/// Computes cosine similarity: dot(a, b) / (||a|| * ||b||).
///
/// Range: [-1, 1] where 1 = same direction, 0 = orthogonal, -1 = opposite.
#[inline]
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    1.0 - cosine_distance(a, b)
}

/// Computes Euclidean (L2) distance: `sqrt(sum((a[i] - b[i])^2))`.
///
/// Uses SIMD acceleration when available.
#[inline]
pub fn euclidean_distance(a: &[f32], b: &[f32]) -> f32 {
    simd::euclidean_distance_simd(a, b)
}

/// Computes squared Euclidean distance: `sum((a[i] - b[i])^2)`.
///
/// Use this when you only need to compare distances (avoids sqrt).
/// Uses SIMD acceleration when available.
#[inline]
pub fn euclidean_distance_squared(a: &[f32], b: &[f32]) -> f32 {
    simd::euclidean_distance_squared_simd(a, b)
}

/// Computes dot product: `sum(a[i] * b[i])`.
///
/// Uses SIMD acceleration when available.
#[inline]
pub fn dot_product(a: &[f32], b: &[f32]) -> f32 {
    simd::dot_product_simd(a, b)
}

/// Computes Manhattan (L1) distance: `sum(|a[i] - b[i]|)`.
///
/// Uses SIMD acceleration when available.
#[inline]
pub fn manhattan_distance(a: &[f32], b: &[f32]) -> f32 {
    simd::manhattan_distance_simd(a, b)
}

/// Normalizes a vector to unit length (L2 norm = 1).
///
/// Returns the original magnitude. If magnitude is zero, returns 0.0
/// and leaves the vector unchanged.
#[inline]
pub fn normalize(v: &mut [f32]) -> f32 {
    let mut norm = 0.0f32;
    for &x in v.iter() {
        norm += x * x;
    }
    let norm = norm.sqrt();

    if norm > f32::EPSILON {
        for x in v.iter_mut() {
            *x /= norm;
        }
    }

    norm
}

/// Computes the L2 norm (magnitude) of a vector.
#[inline]
pub fn l2_norm(v: &[f32]) -> f32 {
    let mut sum = 0.0f32;
    for &x in v {
        sum += x * x;
    }
    sum.sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPSILON: f32 = 1e-5;

    fn approx_eq(a: f32, b: f32) -> bool {
        (a - b).abs() < EPSILON
    }

    #[test]
    fn test_cosine_distance_identical() {
        let a = [1.0f32, 2.0, 3.0];
        let b = [1.0f32, 2.0, 3.0];
        assert!(approx_eq(cosine_distance(&a, &b), 0.0));
    }

    #[test]
    fn test_cosine_distance_orthogonal() {
        let a = [1.0f32, 0.0, 0.0];
        let b = [0.0f32, 1.0, 0.0];
        assert!(approx_eq(cosine_distance(&a, &b), 1.0));
    }

    #[test]
    fn test_cosine_distance_opposite() {
        let a = [1.0f32, 0.0, 0.0];
        let b = [-1.0f32, 0.0, 0.0];
        assert!(approx_eq(cosine_distance(&a, &b), 2.0));
    }

    #[test]
    fn test_euclidean_distance_identical() {
        let a = [1.0f32, 2.0, 3.0];
        let b = [1.0f32, 2.0, 3.0];
        assert!(approx_eq(euclidean_distance(&a, &b), 0.0));
    }

    #[test]
    fn test_euclidean_distance_unit_vectors() {
        let a = [1.0f32, 0.0, 0.0];
        let b = [0.0f32, 1.0, 0.0];
        assert!(approx_eq(euclidean_distance(&a, &b), 2.0f32.sqrt()));
    }

    #[test]
    fn test_euclidean_distance_3_4_5() {
        let a = [0.0f32, 0.0];
        let b = [3.0f32, 4.0];
        assert!(approx_eq(euclidean_distance(&a, &b), 5.0));
    }

    #[test]
    fn test_dot_product() {
        let a = [1.0f32, 2.0, 3.0];
        let b = [4.0f32, 5.0, 6.0];
        // 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32
        assert!(approx_eq(dot_product(&a, &b), 32.0));
    }

    #[test]
    fn test_manhattan_distance() {
        let a = [1.0f32, 2.0, 3.0];
        let b = [4.0f32, 6.0, 3.0];
        // |1-4| + |2-6| + |3-3| = 3 + 4 + 0 = 7
        assert!(approx_eq(manhattan_distance(&a, &b), 7.0));
    }

    #[test]
    fn test_normalize() {
        let mut v = [3.0f32, 4.0];
        let orig_norm = normalize(&mut v);
        assert!(approx_eq(orig_norm, 5.0));
        assert!(approx_eq(v[0], 0.6));
        assert!(approx_eq(v[1], 0.8));
        assert!(approx_eq(l2_norm(&v), 1.0));
    }

    #[test]
    fn test_normalize_zero_vector() {
        let mut v = [0.0f32, 0.0, 0.0];
        let norm = normalize(&mut v);
        assert!(approx_eq(norm, 0.0));
        // Vector should remain unchanged
        assert!(approx_eq(v[0], 0.0));
    }

    #[test]
    fn test_compute_distance_dispatch() {
        let a = [1.0f32, 0.0];
        let b = [0.0f32, 1.0];

        let cos = compute_distance(&a, &b, DistanceMetric::Cosine);
        let euc = compute_distance(&a, &b, DistanceMetric::Euclidean);
        let man = compute_distance(&a, &b, DistanceMetric::Manhattan);

        assert!(approx_eq(cos, 1.0)); // Orthogonal
        assert!(approx_eq(euc, 2.0f32.sqrt()));
        assert!(approx_eq(man, 2.0));
    }

    #[test]
    fn test_metric_from_str() {
        assert_eq!(
            DistanceMetric::from_str("cosine"),
            Some(DistanceMetric::Cosine)
        );
        assert_eq!(
            DistanceMetric::from_str("COSINE"),
            Some(DistanceMetric::Cosine)
        );
        assert_eq!(
            DistanceMetric::from_str("cos"),
            Some(DistanceMetric::Cosine)
        );

        assert_eq!(
            DistanceMetric::from_str("euclidean"),
            Some(DistanceMetric::Euclidean)
        );
        assert_eq!(
            DistanceMetric::from_str("l2"),
            Some(DistanceMetric::Euclidean)
        );

        assert_eq!(
            DistanceMetric::from_str("dot_product"),
            Some(DistanceMetric::DotProduct)
        );
        assert_eq!(
            DistanceMetric::from_str("ip"),
            Some(DistanceMetric::DotProduct)
        );

        assert_eq!(
            DistanceMetric::from_str("manhattan"),
            Some(DistanceMetric::Manhattan)
        );
        assert_eq!(
            DistanceMetric::from_str("l1"),
            Some(DistanceMetric::Manhattan)
        );

        assert_eq!(DistanceMetric::from_str("invalid"), None);
    }

    #[test]
    fn test_metric_name() {
        assert_eq!(DistanceMetric::Cosine.name(), "cosine");
        assert_eq!(DistanceMetric::Euclidean.name(), "euclidean");
        assert_eq!(DistanceMetric::DotProduct.name(), "dot_product");
        assert_eq!(DistanceMetric::Manhattan.name(), "manhattan");
    }

    #[test]
    fn test_high_dimensional() {
        // Test with 384-dim vectors (common embedding size)
        let a: Vec<f32> = (0..384).map(|i| (i as f32) / 384.0).collect();
        let b: Vec<f32> = (0..384).map(|i| ((383 - i) as f32) / 384.0).collect();

        let cos = cosine_distance(&a, &b);
        let euc = euclidean_distance(&a, &b);

        // Just verify they produce reasonable values
        assert!((0.0..=2.0).contains(&cos));
        assert!(euc >= 0.0);
    }

    // ── Edge case tests ─────────────────────────────────────────────

    #[test]
    fn test_single_dimension() {
        let a = [5.0f32];
        let b = [3.0f32];
        assert!(approx_eq(euclidean_distance(&a, &b), 2.0));
        assert!(approx_eq(manhattan_distance(&a, &b), 2.0));
    }

    #[test]
    fn test_zero_vectors_euclidean() {
        let a = [0.0f32, 0.0, 0.0];
        let b = [0.0f32, 0.0, 0.0];
        assert!(approx_eq(euclidean_distance(&a, &b), 0.0));
    }

    #[test]
    fn test_zero_vectors_cosine() {
        let a = [0.0f32, 0.0, 0.0];
        let b = [0.0f32, 0.0, 0.0];
        let d = cosine_distance(&a, &b);
        // Zero vectors have undefined cosine; should not panic
        assert!(!d.is_nan() || d.is_nan()); // Just verify no panic
    }

    #[test]
    fn test_one_zero_vector_cosine() {
        let a = [1.0f32, 0.0, 0.0];
        let b = [0.0f32, 0.0, 0.0];
        let d = cosine_distance(&a, &b);
        // Should not panic; result depends on implementation
        assert!(d.is_finite() || d.is_nan());
    }

    #[test]
    fn test_identical_vectors_all_metrics() {
        let v = [0.5f32, -0.3, 0.8, 1.2];
        assert!(approx_eq(cosine_distance(&v, &v), 0.0));
        assert!(approx_eq(euclidean_distance(&v, &v), 0.0));
        assert!(approx_eq(manhattan_distance(&v, &v), 0.0));
    }

    #[test]
    fn test_negative_values() {
        let a = [-1.0f32, -2.0, -3.0];
        let b = [-1.0f32, -2.0, -3.0];
        assert!(approx_eq(cosine_distance(&a, &b), 0.0));
        assert!(approx_eq(euclidean_distance(&a, &b), 0.0));
    }

    #[test]
    fn test_dot_product_orthogonal() {
        let a = [1.0f32, 0.0];
        let b = [0.0f32, 1.0];
        assert!(approx_eq(dot_product(&a, &b), 0.0));
    }

    #[test]
    fn test_dot_product_negative() {
        let a = [1.0f32, 0.0];
        let b = [-1.0f32, 0.0];
        assert!(approx_eq(dot_product(&a, &b), -1.0));
    }

    #[test]
    fn test_manhattan_single_axis_diff() {
        let a = [0.0f32, 0.0, 0.0];
        let b = [0.0f32, 5.0, 0.0];
        assert!(approx_eq(manhattan_distance(&a, &b), 5.0));
    }

    #[test]
    fn test_cosine_similarity_range() {
        // Cosine similarity should be in [-1, 1], distance in [0, 2]
        let a = [0.3f32, 0.7, -0.2];
        let b = [0.6f32, -0.1, 0.9];
        let d = cosine_distance(&a, &b);
        assert!((0.0 - EPSILON..=2.0 + EPSILON).contains(&d));
    }

    #[test]
    fn test_normalize_already_normalized() {
        let mut v = [0.6f32, 0.8]; // Already unit length
        let norm = normalize(&mut v);
        assert!(approx_eq(norm, 1.0));
        assert!(approx_eq(l2_norm(&v), 1.0));
    }

    #[test]
    fn test_normalize_single_element() {
        let mut v = [7.0f32];
        normalize(&mut v);
        assert!(approx_eq(v[0], 1.0));
    }

    #[test]
    fn test_large_values() {
        let a = [1e10f32, 1e10, 1e10];
        let b = [1e10f32, 1e10, 1e10];
        assert!(approx_eq(euclidean_distance(&a, &b), 0.0));
        assert!(approx_eq(cosine_distance(&a, &b), 0.0));
    }

    #[test]
    fn test_very_small_values() {
        let a = [1e-10f32, 1e-10];
        let b = [1e-10f32, 1e-10];
        assert!(approx_eq(euclidean_distance(&a, &b), 0.0));
    }

    #[test]
    fn test_compute_distance_dot_product() {
        let a = [1.0f32, 2.0, 3.0];
        let b = [4.0f32, 5.0, 6.0];
        let d = compute_distance(&a, &b, DistanceMetric::DotProduct);
        // dot_product returns negative for sorting: -32.0
        assert!(approx_eq(d, -32.0));
    }
}
