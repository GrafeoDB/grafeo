//! Database configuration.

use std::path::PathBuf;

/// Database configuration.
#[derive(Debug, Clone)]
#[allow(clippy::struct_excessive_bools)] // Config structs naturally have many boolean flags
pub struct Config {
    /// Path to the database directory (None for in-memory only).
    pub path: Option<PathBuf>,

    /// Memory limit in bytes (None for unlimited).
    pub memory_limit: Option<usize>,

    /// Path for spilling data to disk under memory pressure.
    pub spill_path: Option<PathBuf>,

    /// Number of worker threads for query execution.
    pub threads: usize,

    /// Whether to enable WAL for durability.
    pub wal_enabled: bool,

    /// WAL flush interval in milliseconds.
    pub wal_flush_interval_ms: u64,

    /// Whether to maintain backward edges.
    pub backward_edges: bool,

    /// Whether to enable query logging.
    pub query_logging: bool,

    /// Adaptive execution configuration.
    pub adaptive: AdaptiveConfig,

    /// Whether to use factorized execution for multi-hop queries.
    ///
    /// When enabled, consecutive MATCH expansions are executed using factorized
    /// representation which avoids Cartesian product materialization. This provides
    /// 5-100x speedup for multi-hop queries with high fan-out.
    ///
    /// Enabled by default.
    pub factorized_execution: bool,
}

/// Configuration for adaptive query execution.
///
/// Adaptive execution monitors actual row counts during query processing and
/// can trigger re-optimization when estimates are significantly wrong.
#[derive(Debug, Clone)]
pub struct AdaptiveConfig {
    /// Whether adaptive execution is enabled.
    pub enabled: bool,

    /// Deviation threshold that triggers re-optimization.
    ///
    /// A value of 3.0 means re-optimization is triggered when actual cardinality
    /// is more than 3x or less than 1/3x the estimated value.
    pub threshold: f64,

    /// Minimum number of rows before considering re-optimization.
    ///
    /// Helps avoid thrashing on small result sets.
    pub min_rows: u64,

    /// Maximum number of re-optimizations allowed per query.
    pub max_reoptimizations: usize,
}

impl Default for AdaptiveConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            threshold: 3.0,
            min_rows: 1000,
            max_reoptimizations: 3,
        }
    }
}

impl AdaptiveConfig {
    /// Creates a disabled adaptive config.
    #[must_use]
    pub fn disabled() -> Self {
        Self {
            enabled: false,
            ..Default::default()
        }
    }

    /// Sets the deviation threshold.
    #[must_use]
    pub fn with_threshold(mut self, threshold: f64) -> Self {
        self.threshold = threshold;
        self
    }

    /// Sets the minimum rows before re-optimization.
    #[must_use]
    pub fn with_min_rows(mut self, min_rows: u64) -> Self {
        self.min_rows = min_rows;
        self
    }

    /// Sets the maximum number of re-optimizations.
    #[must_use]
    pub fn with_max_reoptimizations(mut self, max: usize) -> Self {
        self.max_reoptimizations = max;
        self
    }
}

impl Default for Config {
    fn default() -> Self {
        Self {
            path: None,
            memory_limit: None,
            spill_path: None,
            threads: num_cpus::get(),
            wal_enabled: true,
            wal_flush_interval_ms: 100,
            backward_edges: true,
            query_logging: false,
            adaptive: AdaptiveConfig::default(),
            factorized_execution: true,
        }
    }
}

impl Config {
    /// Creates a new configuration for an in-memory database.
    #[must_use]
    pub fn in_memory() -> Self {
        Self {
            path: None,
            wal_enabled: false,
            ..Default::default()
        }
    }

    /// Creates a new configuration for a persistent database.
    #[must_use]
    pub fn persistent(path: impl Into<PathBuf>) -> Self {
        Self {
            path: Some(path.into()),
            wal_enabled: true,
            ..Default::default()
        }
    }

    /// Sets the memory limit.
    #[must_use]
    pub fn with_memory_limit(mut self, limit: usize) -> Self {
        self.memory_limit = Some(limit);
        self
    }

    /// Sets the number of worker threads.
    #[must_use]
    pub fn with_threads(mut self, threads: usize) -> Self {
        self.threads = threads;
        self
    }

    /// Disables backward edges.
    #[must_use]
    pub fn without_backward_edges(mut self) -> Self {
        self.backward_edges = false;
        self
    }

    /// Enables query logging.
    #[must_use]
    pub fn with_query_logging(mut self) -> Self {
        self.query_logging = true;
        self
    }

    /// Sets the memory budget as a fraction of system RAM.
    #[must_use]
    pub fn with_memory_fraction(mut self, fraction: f64) -> Self {
        use grafeo_common::memory::buffer::BufferManagerConfig;
        let system_memory = BufferManagerConfig::detect_system_memory();
        self.memory_limit = Some((system_memory as f64 * fraction) as usize);
        self
    }

    /// Sets the spill directory for out-of-core processing.
    #[must_use]
    pub fn with_spill_path(mut self, path: impl Into<PathBuf>) -> Self {
        self.spill_path = Some(path.into());
        self
    }

    /// Sets the adaptive execution configuration.
    #[must_use]
    pub fn with_adaptive(mut self, adaptive: AdaptiveConfig) -> Self {
        self.adaptive = adaptive;
        self
    }

    /// Disables adaptive execution.
    #[must_use]
    pub fn without_adaptive(mut self) -> Self {
        self.adaptive.enabled = false;
        self
    }

    /// Disables factorized execution for multi-hop queries.
    ///
    /// This reverts to the traditional flat execution model where each expansion
    /// creates a full Cartesian product. Only use this if you encounter issues
    /// with factorized execution.
    #[must_use]
    pub fn without_factorized_execution(mut self) -> Self {
        self.factorized_execution = false;
        self
    }
}

/// Helper function to get CPU count (fallback implementation).
mod num_cpus {
    pub fn get() -> usize {
        std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(4)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_default() {
        let config = Config::default();
        assert!(config.path.is_none());
        assert!(config.memory_limit.is_none());
        assert!(config.spill_path.is_none());
        assert!(config.threads > 0);
        assert!(config.wal_enabled);
        assert_eq!(config.wal_flush_interval_ms, 100);
        assert!(config.backward_edges);
        assert!(!config.query_logging);
        assert!(config.factorized_execution);
    }

    #[test]
    fn test_config_in_memory() {
        let config = Config::in_memory();
        assert!(config.path.is_none());
        assert!(!config.wal_enabled);
        assert!(config.backward_edges);
    }

    #[test]
    fn test_config_persistent() {
        let config = Config::persistent("/tmp/test_db");
        assert_eq!(
            config.path.as_deref(),
            Some(std::path::Path::new("/tmp/test_db"))
        );
        assert!(config.wal_enabled);
    }

    #[test]
    fn test_config_with_memory_limit() {
        let config = Config::in_memory().with_memory_limit(1024 * 1024);
        assert_eq!(config.memory_limit, Some(1024 * 1024));
    }

    #[test]
    fn test_config_with_threads() {
        let config = Config::in_memory().with_threads(8);
        assert_eq!(config.threads, 8);
    }

    #[test]
    fn test_config_without_backward_edges() {
        let config = Config::in_memory().without_backward_edges();
        assert!(!config.backward_edges);
    }

    #[test]
    fn test_config_with_query_logging() {
        let config = Config::in_memory().with_query_logging();
        assert!(config.query_logging);
    }

    #[test]
    fn test_config_with_spill_path() {
        let config = Config::in_memory().with_spill_path("/tmp/spill");
        assert_eq!(
            config.spill_path.as_deref(),
            Some(std::path::Path::new("/tmp/spill"))
        );
    }

    #[test]
    fn test_config_with_memory_fraction() {
        let config = Config::in_memory().with_memory_fraction(0.5);
        assert!(config.memory_limit.is_some());
        assert!(config.memory_limit.unwrap() > 0);
    }

    #[test]
    fn test_config_with_adaptive() {
        let adaptive = AdaptiveConfig::default().with_threshold(5.0);
        let config = Config::in_memory().with_adaptive(adaptive);
        assert!((config.adaptive.threshold - 5.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_config_without_adaptive() {
        let config = Config::in_memory().without_adaptive();
        assert!(!config.adaptive.enabled);
    }

    #[test]
    fn test_config_without_factorized_execution() {
        let config = Config::in_memory().without_factorized_execution();
        assert!(!config.factorized_execution);
    }

    #[test]
    fn test_config_builder_chaining() {
        let config = Config::persistent("/tmp/db")
            .with_memory_limit(512 * 1024 * 1024)
            .with_threads(4)
            .with_query_logging()
            .without_backward_edges()
            .with_spill_path("/tmp/spill");

        assert!(config.path.is_some());
        assert_eq!(config.memory_limit, Some(512 * 1024 * 1024));
        assert_eq!(config.threads, 4);
        assert!(config.query_logging);
        assert!(!config.backward_edges);
        assert!(config.spill_path.is_some());
    }

    #[test]
    fn test_adaptive_config_default() {
        let config = AdaptiveConfig::default();
        assert!(config.enabled);
        assert!((config.threshold - 3.0).abs() < f64::EPSILON);
        assert_eq!(config.min_rows, 1000);
        assert_eq!(config.max_reoptimizations, 3);
    }

    #[test]
    fn test_adaptive_config_disabled() {
        let config = AdaptiveConfig::disabled();
        assert!(!config.enabled);
    }

    #[test]
    fn test_adaptive_config_with_threshold() {
        let config = AdaptiveConfig::default().with_threshold(10.0);
        assert!((config.threshold - 10.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_adaptive_config_with_min_rows() {
        let config = AdaptiveConfig::default().with_min_rows(500);
        assert_eq!(config.min_rows, 500);
    }

    #[test]
    fn test_adaptive_config_with_max_reoptimizations() {
        let config = AdaptiveConfig::default().with_max_reoptimizations(5);
        assert_eq!(config.max_reoptimizations, 5);
    }

    #[test]
    fn test_adaptive_config_builder_chaining() {
        let config = AdaptiveConfig::default()
            .with_threshold(2.0)
            .with_min_rows(100)
            .with_max_reoptimizations(10);
        assert!((config.threshold - 2.0).abs() < f64::EPSILON);
        assert_eq!(config.min_rows, 100);
        assert_eq!(config.max_reoptimizations, 10);
    }
}
