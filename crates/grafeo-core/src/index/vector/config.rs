//! Configuration types for HNSW vector index.

use super::DistanceMetric;

/// Configuration for HNSW (Hierarchical Navigable Small World) index.
///
/// HNSW is an approximate nearest neighbor algorithm that builds a multi-layer
/// navigable small world graph. It provides logarithmic search complexity with
/// high recall (typically >95%).
///
/// # Parameters
///
/// - **M**: Number of bi-directional links per node. Higher M means better
///   recall but more memory and slower construction. Typical: 12-48.
/// - **M_max**: Maximum links at layer 0 (usually 2*M).
/// - **ef_construction**: Search beam width during construction. Higher values
///   give better index quality at the cost of construction time. Typical: 100-500.
/// - **ef**: Search beam width during queries. Higher values give better recall
///   at the cost of query latency. Typical: 50-200.
///
/// # Example
///
/// ```
/// use grafeo_core::index::vector::{HnswConfig, DistanceMetric};
///
/// // Configuration for OpenAI embeddings (1536 dimensions)
/// let config = HnswConfig::new(1536, DistanceMetric::Cosine)
///     .with_m(16)
///     .with_ef_construction(200)
///     .with_ef(100);
///
/// assert_eq!(config.dimensions, 1536);
/// assert_eq!(config.m, 16);
/// assert_eq!(config.m_max, 32); // Automatically set to 2*M
/// ```
#[derive(Debug, Clone)]
pub struct HnswConfig {
    /// Vector dimensions (e.g., 384, 768, 1536).
    pub dimensions: usize,

    /// Distance metric for similarity computation.
    pub metric: DistanceMetric,

    /// Number of bi-directional links per node at layers > 0.
    ///
    /// Higher M = better recall, more memory, slower construction.
    /// Typical values: 12-48. Default: 16.
    pub m: usize,

    /// Maximum number of links per node at layer 0.
    ///
    /// Usually set to 2*M. Default: 32.
    pub m_max: usize,

    /// Search beam width during index construction.
    ///
    /// Higher values = better index quality, slower construction.
    /// Typical values: 100-500. Default: 128.
    pub ef_construction: usize,

    /// Search beam width during queries.
    ///
    /// Higher values = better recall, higher latency.
    /// Typical values: 50-200. Default: 50.
    pub ef: usize,

    /// Normalization factor for level selection (1/ln(M)).
    ///
    /// Controls the probability distribution of node levels.
    /// Computed automatically from M.
    pub ml: f64,

    /// Relaxation parameter for diversity-aware neighbor selection.
    ///
    /// Controls how aggressively the heuristic prunes neighbors that are
    /// "covered" by already-selected neighbors. Based on the Vamana/DiskANN
    /// algorithm's robust pruning.
    ///
    /// - **1.0** (default): Standard HNSW heuristic. A candidate is rejected
    ///   if any selected neighbor is closer to it than the query is.
    /// - **>1.0** (e.g., 1.2): Relaxed - allows some longer-range edges to
    ///   survive, improving navigability and recall at the cost of slightly
    ///   more edges.
    ///
    /// Typical values: 1.0-1.4.
    pub alpha: f32,
}

impl HnswConfig {
    /// Creates a new HNSW configuration with the given dimensions and metric.
    ///
    /// Uses default values for M, ef_construction, and ef.
    #[must_use]
    pub fn new(dimensions: usize, metric: DistanceMetric) -> Self {
        let m = 16;
        Self {
            dimensions,
            metric,
            m,
            m_max: m * 2,
            ef_construction: 128,
            ef: 50,
            ml: 1.0 / (m as f64).ln(),
            alpha: 1.0,
        }
    }

    /// Creates a configuration optimized for high recall.
    ///
    /// Uses higher M and ef values for better accuracy at the cost of
    /// memory and latency.
    #[must_use]
    pub fn high_recall(dimensions: usize, metric: DistanceMetric) -> Self {
        let m = 32;
        Self {
            dimensions,
            metric,
            m,
            m_max: m * 2,
            ef_construction: 256,
            ef: 100,
            ml: 1.0 / (m as f64).ln(),
            alpha: 1.2,
        }
    }

    /// Creates a configuration optimized for speed.
    ///
    /// Uses lower M and ef values for faster queries at the cost of recall.
    #[must_use]
    pub fn fast(dimensions: usize, metric: DistanceMetric) -> Self {
        let m = 12;
        Self {
            dimensions,
            metric,
            m,
            m_max: m * 2,
            ef_construction: 100,
            ef: 32,
            ml: 1.0 / (m as f64).ln(),
            alpha: 1.0,
        }
    }

    /// Sets the number of bi-directional links per node (M).
    #[must_use]
    pub fn with_m(mut self, m: usize) -> Self {
        self.m = m;
        self.m_max = m * 2;
        self.ml = 1.0 / (m as f64).ln();
        self
    }

    /// Sets the maximum links at layer 0 (M_max).
    #[must_use]
    pub fn with_m_max(mut self, m_max: usize) -> Self {
        self.m_max = m_max;
        self
    }

    /// Sets the construction beam width (ef_construction).
    #[must_use]
    pub fn with_ef_construction(mut self, ef_construction: usize) -> Self {
        self.ef_construction = ef_construction;
        self
    }

    /// Sets the query beam width (ef).
    #[must_use]
    pub fn with_ef(mut self, ef: usize) -> Self {
        self.ef = ef;
        self
    }

    /// Sets the diversity pruning relaxation parameter (alpha).
    ///
    /// - 1.0: Standard HNSW heuristic (default)
    /// - >1.0: Relaxed pruning, more long-range edges, better recall
    #[must_use]
    pub fn with_alpha(mut self, alpha: f32) -> Self {
        self.alpha = alpha;
        self
    }

    /// Returns the maximum number of layers expected for a given number of nodes.
    ///
    /// This is a rough estimate based on the probability distribution.
    #[must_use]
    pub fn expected_max_level(&self, n: usize) -> usize {
        if n == 0 {
            return 0;
        }
        ((n as f64).ln() * self.ml).ceil() as usize
    }
}

impl Default for HnswConfig {
    fn default() -> Self {
        Self::new(384, DistanceMetric::Cosine)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hnsw_config_new() {
        let config = HnswConfig::new(768, DistanceMetric::Euclidean);

        assert_eq!(config.dimensions, 768);
        assert_eq!(config.metric, DistanceMetric::Euclidean);
        assert_eq!(config.m, 16);
        assert_eq!(config.m_max, 32);
        assert_eq!(config.ef_construction, 128);
        assert_eq!(config.ef, 50);
    }

    #[test]
    fn test_hnsw_config_builder() {
        let config = HnswConfig::new(1536, DistanceMetric::Cosine)
            .with_m(24)
            .with_ef_construction(200)
            .with_ef(100);

        assert_eq!(config.m, 24);
        assert_eq!(config.m_max, 48);
        assert_eq!(config.ef_construction, 200);
        assert_eq!(config.ef, 100);
    }

    #[test]
    fn test_hnsw_config_high_recall_preset() {
        let config = HnswConfig::high_recall(384, DistanceMetric::Cosine);

        assert_eq!(config.dimensions, 384);
        assert_eq!(config.metric, DistanceMetric::Cosine);
        assert_eq!(config.m, 32);
        assert_eq!(config.m_max, 64); // 2 * m
        assert_eq!(config.ef_construction, 256);
        assert_eq!(config.ef, 100);
        // Verify ml is computed correctly
        assert!((config.ml - 1.0 / (32.0_f64).ln()).abs() < 1e-10);
    }

    #[test]
    fn test_hnsw_config_fast_preset() {
        let config = HnswConfig::fast(384, DistanceMetric::Euclidean);

        assert_eq!(config.dimensions, 384);
        assert_eq!(config.metric, DistanceMetric::Euclidean);
        assert_eq!(config.m, 12);
        assert_eq!(config.m_max, 24); // 2 * m
        assert_eq!(config.ef_construction, 100);
        assert_eq!(config.ef, 32);
        // Verify ml is computed correctly
        assert!((config.ml - 1.0 / (12.0_f64).ln()).abs() < 1e-10);
    }

    #[test]
    fn test_preset_ordering() {
        // High recall should have higher values than default
        let default = HnswConfig::default();
        let high_recall = HnswConfig::high_recall(384, DistanceMetric::Cosine);

        assert!(high_recall.m > default.m);
        assert!(high_recall.ef_construction > default.ef_construction);
        assert!(high_recall.ef > default.ef);

        // Fast should have lower values than default
        let fast = HnswConfig::fast(384, DistanceMetric::Cosine);

        assert!(fast.m < default.m);
        assert!(fast.ef_construction < default.ef_construction);
        assert!(fast.ef < default.ef);
    }

    #[test]
    fn test_with_m_updates_derived_fields() {
        let config = HnswConfig::new(384, DistanceMetric::Cosine).with_m(24);

        assert_eq!(config.m, 24);
        assert_eq!(config.m_max, 48); // Auto-updated to 2*m
        assert!((config.ml - 1.0 / (24.0_f64).ln()).abs() < 1e-10); // ml updated
    }

    #[test]
    fn test_with_m_max_override() {
        // Can override m_max independently
        let config = HnswConfig::new(384, DistanceMetric::Cosine)
            .with_m(16)
            .with_m_max(64);

        assert_eq!(config.m, 16);
        assert_eq!(config.m_max, 64); // Overridden, not 2*m
    }

    #[test]
    fn test_expected_max_level() {
        let config = HnswConfig::new(384, DistanceMetric::Cosine);

        // Empty index
        assert_eq!(config.expected_max_level(0), 0);

        // Small index
        let level_100 = config.expected_max_level(100);
        assert!(level_100 > 0 && level_100 < 10);

        // Large index
        let level_1m = config.expected_max_level(1_000_000);
        assert!(level_1m > level_100);
    }
}
