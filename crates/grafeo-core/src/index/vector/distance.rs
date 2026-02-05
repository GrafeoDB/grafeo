//! Distance metrics for vector similarity search.
//!
//! Provides efficient computation of various distance metrics between vectors.
//! All functions expect vectors of equal length.

use serde::{Deserialize, Serialize};

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

/// Computes the distance between two vectors using the specified metric.
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
    debug_assert_eq!(a.len(), b.len(), "Vector dimensions must match");

    match metric {
        DistanceMetric::Cosine => cosine_distance(a, b),
        DistanceMetric::Euclidean => euclidean_distance(a, b),
        DistanceMetric::DotProduct => negative_dot_product(a, b),
        DistanceMetric::Manhattan => manhattan_distance(a, b),
    }
}

/// Computes cosine distance: 1 - cosine_similarity.
///
/// Cosine similarity = dot(a, b) / (||a|| * ||b||)
/// Cosine distance = 1 - cosine_similarity
///
/// Range: [0, 2] where 0 = same direction, 1 = orthogonal, 2 = opposite.
#[inline]
pub fn cosine_distance(a: &[f32], b: &[f32]) -> f32 {
    let mut dot = 0.0f32;
    let mut norm_a = 0.0f32;
    let mut norm_b = 0.0f32;

    for i in 0..a.len() {
        dot += a[i] * b[i];
        norm_a += a[i] * a[i];
        norm_b += b[i] * b[i];
    }

    let denom = (norm_a.sqrt() * norm_b.sqrt()) + f32::EPSILON;
    1.0 - (dot / denom)
}

/// Computes cosine similarity: dot(a, b) / (||a|| * ||b||).
///
/// Range: [-1, 1] where 1 = same direction, 0 = orthogonal, -1 = opposite.
#[inline]
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    1.0 - cosine_distance(a, b)
}

/// Computes Euclidean (L2) distance: `sqrt(sum((a[i] - b[i])^2))`.
#[inline]
pub fn euclidean_distance(a: &[f32], b: &[f32]) -> f32 {
    euclidean_distance_squared(a, b).sqrt()
}

/// Computes squared Euclidean distance: `sum((a[i] - b[i])^2)`.
///
/// Use this when you only need to compare distances (avoids sqrt).
#[inline]
pub fn euclidean_distance_squared(a: &[f32], b: &[f32]) -> f32 {
    let mut sum = 0.0f32;
    for i in 0..a.len() {
        let diff = a[i] - b[i];
        sum += diff * diff;
    }
    sum
}

/// Computes dot product: `sum(a[i] * b[i])`.
#[inline]
pub fn dot_product(a: &[f32], b: &[f32]) -> f32 {
    let mut sum = 0.0f32;
    for i in 0..a.len() {
        sum += a[i] * b[i];
    }
    sum
}

/// Computes negative dot product for max inner product search.
///
/// Returns negative so smaller values = more similar (compatible with min-heap).
#[inline]
fn negative_dot_product(a: &[f32], b: &[f32]) -> f32 {
    -dot_product(a, b)
}

/// Computes Manhattan (L1) distance: `sum(|a[i] - b[i]|)`.
#[inline]
pub fn manhattan_distance(a: &[f32], b: &[f32]) -> f32 {
    let mut sum = 0.0f32;
    for i in 0..a.len() {
        sum += (a[i] - b[i]).abs();
    }
    sum
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
        assert!(cos >= 0.0 && cos <= 2.0);
        assert!(euc >= 0.0);
    }
}
