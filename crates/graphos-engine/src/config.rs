//! Database configuration.

use std::path::PathBuf;

/// Database configuration.
#[derive(Debug, Clone)]
pub struct Config {
    /// Path to the database directory (None for in-memory only).
    pub path: Option<PathBuf>,

    /// Memory limit in bytes (None for unlimited).
    pub memory_limit: Option<usize>,

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
}

impl Default for Config {
    fn default() -> Self {
        Self {
            path: None,
            memory_limit: None,
            threads: num_cpus::get(),
            wal_enabled: true,
            wal_flush_interval_ms: 100,
            backward_edges: true,
            query_logging: false,
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
}

/// Helper function to get CPU count (fallback implementation).
mod num_cpus {
    pub fn get() -> usize {
        std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(4)
    }
}
