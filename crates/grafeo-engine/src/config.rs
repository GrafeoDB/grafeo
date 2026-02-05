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
