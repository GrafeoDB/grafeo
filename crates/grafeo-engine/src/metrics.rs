//! Lightweight, lock-free metrics for query, transaction, and session tracking.
//!
//! All counters use relaxed atomics so recording a metric is a single
//! atomic increment with no contention. The [`MetricsRegistry`] is designed
//! to be wrapped in an `Arc` and shared across sessions.
//!
//! Gate everything behind `#[cfg(feature = "metrics")]` so the types
//! compile away to nothing when the feature is disabled.

use std::sync::atomic::{AtomicI64, AtomicU64, Ordering};

// ---------------------------------------------------------------------------
// Histogram
// ---------------------------------------------------------------------------

/// Bucket boundaries for latency histograms (in milliseconds).
const LATENCY_BUCKETS: &[f64] = &[
    0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 25.0, 50.0, 100.0, 250.0, 500.0, 1000.0, 5000.0, 10000.0,
];

/// A lock-free histogram backed by fixed bucket boundaries.
///
/// Each observation is placed in the first bucket whose boundary is
/// greater than or equal to the value. An overflow bucket catches
/// values above the last boundary.
pub struct AtomicHistogram {
    boundaries: &'static [f64],
    /// One counter per bucket + one overflow bucket at the end.
    buckets: Box<[AtomicU64]>,
    /// Running sum stored as `f64` bits (updated via CAS loop).
    sum: AtomicU64,
    /// Total number of observations.
    count: AtomicU64,
}

impl AtomicHistogram {
    /// Creates a new histogram with the given bucket boundaries.
    ///
    /// `boundaries` must be sorted in ascending order. One extra overflow
    /// bucket is allocated for values exceeding the last boundary.
    #[must_use]
    pub fn new(boundaries: &'static [f64]) -> Self {
        let bucket_count = boundaries.len() + 1; // +1 for overflow
        let buckets: Vec<AtomicU64> = (0..bucket_count).map(|_| AtomicU64::new(0)).collect();
        Self {
            boundaries,
            buckets: buckets.into_boxed_slice(),
            sum: AtomicU64::new(0),
            count: AtomicU64::new(0),
        }
    }

    /// Records a single observation.
    pub fn observe(&self, value: f64) {
        // Find the correct bucket.
        let idx = self
            .boundaries
            .iter()
            .position(|&b| value <= b)
            .unwrap_or(self.boundaries.len()); // overflow bucket
        self.buckets[idx].fetch_add(1, Ordering::Relaxed);

        // Update sum via CAS loop (relaxed is fine for metrics).
        loop {
            let old_bits = self.sum.load(Ordering::Relaxed);
            let old_sum = f64::from_bits(old_bits);
            let new_sum = old_sum + value;
            if self
                .sum
                .compare_exchange_weak(
                    old_bits,
                    new_sum.to_bits(),
                    Ordering::Relaxed,
                    Ordering::Relaxed,
                )
                .is_ok()
            {
                break;
            }
        }

        self.count.fetch_add(1, Ordering::Relaxed);
    }

    /// Estimates the given percentile (0.0 ..= 1.0) using linear
    /// interpolation within the matching bucket.
    #[must_use]
    pub fn percentile(&self, p: f64) -> f64 {
        let total = self.count.load(Ordering::Relaxed);
        if total == 0 {
            return 0.0;
        }

        // reason: p is in 0..1 and total is non-negative, so product is always non-negative
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        let target = (p * total as f64).ceil() as u64;
        let mut cumulative: u64 = 0;

        for (i, bucket) in self.buckets.iter().enumerate() {
            cumulative += bucket.load(Ordering::Relaxed);
            if cumulative >= target {
                // Return the upper bound of this bucket.
                return if i < self.boundaries.len() {
                    self.boundaries[i]
                } else {
                    // Overflow bucket: return the last boundary as a lower-bound estimate.
                    *self.boundaries.last().unwrap_or(&0.0)
                };
            }
        }

        *self.boundaries.last().unwrap_or(&0.0)
    }

    /// Returns the arithmetic mean of all observations.
    #[must_use]
    pub fn mean(&self) -> f64 {
        let total = self.count.load(Ordering::Relaxed);
        if total == 0 {
            return 0.0;
        }
        let sum = f64::from_bits(self.sum.load(Ordering::Relaxed));
        sum / total as f64
    }

    /// Resets all buckets, the sum, and the count to zero.
    pub fn reset(&self) {
        for bucket in &*self.buckets {
            bucket.store(0, Ordering::Relaxed);
        }
        self.sum.store(0, Ordering::Relaxed);
        self.count.store(0, Ordering::Relaxed);
    }

    /// Takes a point-in-time snapshot of the histogram state.
    #[must_use]
    pub fn snapshot(&self) -> HistogramSnapshot {
        let bucket_counts: Vec<u64> = self
            .buckets
            .iter()
            .map(|b| b.load(Ordering::Relaxed))
            .collect();
        HistogramSnapshot {
            boundaries: self.boundaries.to_vec(),
            bucket_counts,
            sum: f64::from_bits(self.sum.load(Ordering::Relaxed)),
            count: self.count.load(Ordering::Relaxed),
        }
    }
}

/// A serializable point-in-time snapshot of an [`AtomicHistogram`].
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct HistogramSnapshot {
    /// Bucket boundaries (same as the source histogram).
    pub boundaries: Vec<f64>,
    /// Observation count for each bucket (len = boundaries.len() + 1).
    pub bucket_counts: Vec<u64>,
    /// Sum of all observed values.
    pub sum: f64,
    /// Total number of observations.
    pub count: u64,
}

// ---------------------------------------------------------------------------
// Language counters
// ---------------------------------------------------------------------------

/// Per-language query counters.
pub struct LanguageCounters {
    pub(crate) gql: AtomicU64,
    pub(crate) cypher: AtomicU64,
    pub(crate) sparql: AtomicU64,
    pub(crate) gremlin: AtomicU64,
    pub(crate) graphql: AtomicU64,
    pub(crate) sql_pgq: AtomicU64,
}

impl LanguageCounters {
    /// Creates a new set of language counters, all starting at zero.
    #[must_use]
    fn new() -> Self {
        Self {
            gql: AtomicU64::new(0),
            cypher: AtomicU64::new(0),
            sparql: AtomicU64::new(0),
            gremlin: AtomicU64::new(0),
            graphql: AtomicU64::new(0),
            sql_pgq: AtomicU64::new(0),
        }
    }

    /// Increments the counter for the given language string.
    ///
    /// Unrecognized language names are silently ignored.
    pub fn increment(&self, language: &str) {
        match language {
            "gql" => {
                self.gql.fetch_add(1, Ordering::Relaxed);
            }
            "cypher" => {
                self.cypher.fetch_add(1, Ordering::Relaxed);
            }
            "sparql" => {
                self.sparql.fetch_add(1, Ordering::Relaxed);
            }
            "gremlin" => {
                self.gremlin.fetch_add(1, Ordering::Relaxed);
            }
            "graphql" | "graphql-rdf" => {
                self.graphql.fetch_add(1, Ordering::Relaxed);
            }
            "sql" | "sql-pgq" => {
                self.sql_pgq.fetch_add(1, Ordering::Relaxed);
            }
            _ => {}
        }
    }

    /// Takes a point-in-time snapshot.
    #[must_use]
    pub fn snapshot(&self) -> LanguageSnapshot {
        LanguageSnapshot {
            gql: self.gql.load(Ordering::Relaxed),
            cypher: self.cypher.load(Ordering::Relaxed),
            sparql: self.sparql.load(Ordering::Relaxed),
            gremlin: self.gremlin.load(Ordering::Relaxed),
            graphql: self.graphql.load(Ordering::Relaxed),
            sql_pgq: self.sql_pgq.load(Ordering::Relaxed),
        }
    }

    /// Resets all language counters to zero.
    fn reset(&self) {
        self.gql.store(0, Ordering::Relaxed);
        self.cypher.store(0, Ordering::Relaxed);
        self.sparql.store(0, Ordering::Relaxed);
        self.gremlin.store(0, Ordering::Relaxed);
        self.graphql.store(0, Ordering::Relaxed);
        self.sql_pgq.store(0, Ordering::Relaxed);
    }
}

/// A serializable snapshot of per-language query counts.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct LanguageSnapshot {
    /// GQL (ISO) queries executed.
    pub gql: u64,
    /// Cypher queries executed.
    pub cypher: u64,
    /// SPARQL queries executed.
    pub sparql: u64,
    /// Gremlin queries executed.
    pub gremlin: u64,
    /// GraphQL queries executed.
    pub graphql: u64,
    /// SQL/PGQ queries executed.
    pub sql_pgq: u64,
}

// ---------------------------------------------------------------------------
// Registry
// ---------------------------------------------------------------------------

/// Central metrics registry shared across all sessions.
///
/// Every field uses atomic operations so recording is lock-free.
/// Call [`snapshot()`](Self::snapshot) to get a serializable view.
pub struct MetricsRegistry {
    // -- Query --
    pub(crate) query_count: AtomicU64,
    pub(crate) query_errors: AtomicU64,
    pub(crate) query_timeouts: AtomicU64,
    pub(crate) query_latency: AtomicHistogram,
    pub(crate) query_count_by_language: LanguageCounters,
    pub(crate) rows_returned: AtomicU64,
    pub(crate) rows_scanned: AtomicU64,

    // -- Transaction --
    pub(crate) tx_active: AtomicI64,
    pub(crate) tx_committed: AtomicU64,
    pub(crate) tx_rolled_back: AtomicU64,
    pub(crate) tx_conflicts: AtomicU64,
    pub(crate) tx_duration: AtomicHistogram,

    // -- Session --
    pub(crate) session_active: AtomicI64,
    pub(crate) session_created: AtomicU64,

    // -- GC --
    pub(crate) gc_runs: AtomicU64,
    pub(crate) gc_versions_reclaimed: AtomicU64,
    pub(crate) gc_duration: AtomicHistogram,

    // -- WAL --
    pub(crate) wal_records_written: AtomicU64,
    pub(crate) wal_flush_duration: AtomicHistogram,
}

impl MetricsRegistry {
    /// Creates a new registry with all counters at zero.
    #[must_use]
    pub fn new() -> Self {
        Self {
            query_count: AtomicU64::new(0),
            query_errors: AtomicU64::new(0),
            query_timeouts: AtomicU64::new(0),
            query_latency: AtomicHistogram::new(LATENCY_BUCKETS),
            query_count_by_language: LanguageCounters::new(),
            rows_returned: AtomicU64::new(0),
            rows_scanned: AtomicU64::new(0),

            tx_active: AtomicI64::new(0),
            tx_committed: AtomicU64::new(0),
            tx_rolled_back: AtomicU64::new(0),
            tx_conflicts: AtomicU64::new(0),
            tx_duration: AtomicHistogram::new(LATENCY_BUCKETS),

            session_active: AtomicI64::new(0),
            session_created: AtomicU64::new(0),

            gc_runs: AtomicU64::new(0),
            gc_versions_reclaimed: AtomicU64::new(0),
            gc_duration: AtomicHistogram::new(LATENCY_BUCKETS),

            wal_records_written: AtomicU64::new(0),
            wal_flush_duration: AtomicHistogram::new(LATENCY_BUCKETS),
        }
    }

    /// Takes a point-in-time snapshot of every metric.
    #[must_use]
    pub fn snapshot(&self) -> MetricsSnapshot {
        let lang = self.query_count_by_language.snapshot();
        MetricsSnapshot {
            query_count: self.query_count.load(Ordering::Relaxed),
            query_errors: self.query_errors.load(Ordering::Relaxed),
            query_timeouts: self.query_timeouts.load(Ordering::Relaxed),
            query_latency_p50_ms: self.query_latency.percentile(0.50),
            query_latency_p99_ms: self.query_latency.percentile(0.99),
            query_latency_mean_ms: self.query_latency.mean(),
            rows_returned: self.rows_returned.load(Ordering::Relaxed),
            rows_scanned: self.rows_scanned.load(Ordering::Relaxed),
            queries_gql: lang.gql,
            queries_cypher: lang.cypher,
            queries_sparql: lang.sparql,
            queries_gremlin: lang.gremlin,
            queries_graphql: lang.graphql,
            queries_sql_pgq: lang.sql_pgq,
            tx_active: self.tx_active.load(Ordering::Relaxed),
            tx_committed: self.tx_committed.load(Ordering::Relaxed),
            tx_rolled_back: self.tx_rolled_back.load(Ordering::Relaxed),
            tx_conflicts: self.tx_conflicts.load(Ordering::Relaxed),
            tx_duration_p50_ms: self.tx_duration.percentile(0.50),
            tx_duration_p99_ms: self.tx_duration.percentile(0.99),
            tx_duration_mean_ms: self.tx_duration.mean(),
            session_active: self.session_active.load(Ordering::Relaxed),
            session_created: self.session_created.load(Ordering::Relaxed),
            gc_runs: self.gc_runs.load(Ordering::Relaxed),
            gc_versions_reclaimed: self.gc_versions_reclaimed.load(Ordering::Relaxed),
            gc_duration_mean_ms: self.gc_duration.mean(),
            wal_records_written: self.wal_records_written.load(Ordering::Relaxed),
            wal_flush_duration_mean_ms: self.wal_flush_duration.mean(),
            cache_hits: 0,
            cache_misses: 0,
            cache_size: 0,
            cache_invalidations: 0,
        }
    }

    /// Takes a snapshot with cache statistics merged in.
    ///
    /// Call this instead of [`snapshot()`](Self::snapshot) when you have
    /// access to the query cache stats.
    #[must_use]
    pub fn snapshot_with_cache(
        &self,
        cache_hits: u64,
        cache_misses: u64,
        cache_size: usize,
        cache_invalidations: u64,
    ) -> MetricsSnapshot {
        let mut snap = self.snapshot();
        snap.cache_hits = cache_hits;
        snap.cache_misses = cache_misses;
        snap.cache_size = cache_size;
        snap.cache_invalidations = cache_invalidations;
        snap
    }

    /// Renders all metrics in Prometheus text exposition format.
    ///
    /// Returns a string ready to be served from an HTTP `/metrics` endpoint.
    #[must_use]
    pub fn to_prometheus(&self) -> String {
        use std::fmt::Write;
        let mut out = String::with_capacity(4096);

        // Helper: counter
        macro_rules! counter {
            ($name:expr, $help:expr, $value:expr) => {
                let _ = writeln!(out, "# HELP {} {}", $name, $help);
                let _ = writeln!(out, "# TYPE {} counter", $name);
                let _ = writeln!(out, "{} {}", $name, $value);
            };
        }

        // Helper: gauge
        macro_rules! gauge {
            ($name:expr, $help:expr, $value:expr) => {
                let _ = writeln!(out, "# HELP {} {}", $name, $help);
                let _ = writeln!(out, "# TYPE {} gauge", $name);
                let _ = writeln!(out, "{} {}", $name, $value);
            };
        }

        // Query metrics
        counter!(
            "grafeo_query_count",
            "Total queries executed.",
            self.query_count.load(Ordering::Relaxed)
        );
        counter!(
            "grafeo_query_errors",
            "Queries that returned an error.",
            self.query_errors.load(Ordering::Relaxed)
        );
        counter!(
            "grafeo_query_timeouts",
            "Queries cancelled by timeout.",
            self.query_timeouts.load(Ordering::Relaxed)
        );
        counter!(
            "grafeo_query_rows_returned",
            "Cumulative rows returned.",
            self.rows_returned.load(Ordering::Relaxed)
        );
        counter!(
            "grafeo_query_rows_scanned",
            "Cumulative rows scanned.",
            self.rows_scanned.load(Ordering::Relaxed)
        );

        // Query latency histogram
        Self::write_histogram(
            &mut out,
            "grafeo_query_latency_ms",
            "Query latency in milliseconds.",
            &self.query_latency,
        );

        // Per-language counters
        let lang = self.query_count_by_language.snapshot();
        let _ = writeln!(
            out,
            "# HELP grafeo_query_count_by_language Queries executed per language."
        );
        let _ = writeln!(out, "# TYPE grafeo_query_count_by_language counter");
        let _ = writeln!(
            out,
            "grafeo_query_count_by_language{{language=\"gql\"}} {}",
            lang.gql
        );
        let _ = writeln!(
            out,
            "grafeo_query_count_by_language{{language=\"cypher\"}} {}",
            lang.cypher
        );
        let _ = writeln!(
            out,
            "grafeo_query_count_by_language{{language=\"sparql\"}} {}",
            lang.sparql
        );
        let _ = writeln!(
            out,
            "grafeo_query_count_by_language{{language=\"gremlin\"}} {}",
            lang.gremlin
        );
        let _ = writeln!(
            out,
            "grafeo_query_count_by_language{{language=\"graphql\"}} {}",
            lang.graphql
        );
        let _ = writeln!(
            out,
            "grafeo_query_count_by_language{{language=\"sql_pgq\"}} {}",
            lang.sql_pgq
        );

        // Transaction metrics
        gauge!(
            "grafeo_tx_active",
            "Currently active transactions.",
            self.tx_active.load(Ordering::Relaxed)
        );
        counter!(
            "grafeo_tx_committed",
            "Total transactions committed.",
            self.tx_committed.load(Ordering::Relaxed)
        );
        counter!(
            "grafeo_tx_rolled_back",
            "Total transactions rolled back.",
            self.tx_rolled_back.load(Ordering::Relaxed)
        );
        counter!(
            "grafeo_tx_conflicts",
            "Write-write conflicts detected.",
            self.tx_conflicts.load(Ordering::Relaxed)
        );
        Self::write_histogram(
            &mut out,
            "grafeo_tx_duration_ms",
            "Transaction duration in milliseconds.",
            &self.tx_duration,
        );

        // Session metrics
        gauge!(
            "grafeo_session_active",
            "Currently active sessions.",
            self.session_active.load(Ordering::Relaxed)
        );
        counter!(
            "grafeo_session_created",
            "Total sessions created.",
            self.session_created.load(Ordering::Relaxed)
        );

        // GC metrics
        counter!(
            "grafeo_gc_runs",
            "Total garbage collection runs.",
            self.gc_runs.load(Ordering::Relaxed)
        );
        counter!(
            "grafeo_gc_versions_reclaimed",
            "MVCC versions reclaimed by GC.",
            self.gc_versions_reclaimed.load(Ordering::Relaxed)
        );
        Self::write_histogram(
            &mut out,
            "grafeo_gc_duration_ms",
            "GC pass duration in milliseconds.",
            &self.gc_duration,
        );

        // WAL metrics
        counter!(
            "grafeo_wal_records_written",
            "Total WAL records written.",
            self.wal_records_written.load(Ordering::Relaxed)
        );
        Self::write_histogram(
            &mut out,
            "grafeo_wal_flush_duration_ms",
            "WAL flush duration in milliseconds.",
            &self.wal_flush_duration,
        );

        out
    }

    /// Writes a histogram in Prometheus text format.
    fn write_histogram(out: &mut String, name: &str, help: &str, histogram: &AtomicHistogram) {
        use std::fmt::Write;
        let snap = histogram.snapshot();

        let _ = writeln!(out, "# HELP {name} {help}");
        let _ = writeln!(out, "# TYPE {name} histogram");

        let mut cumulative: u64 = 0;
        for (i, &boundary) in snap.boundaries.iter().enumerate() {
            cumulative += snap.bucket_counts[i];
            let _ = writeln!(out, "{name}_bucket{{le=\"{boundary}\"}} {cumulative}");
        }
        // Overflow bucket (+Inf)
        cumulative += snap.bucket_counts[snap.boundaries.len()];
        let _ = writeln!(out, "{name}_bucket{{le=\"+Inf\"}} {cumulative}");
        let _ = writeln!(out, "{name}_sum {}", snap.sum);
        let _ = writeln!(out, "{name}_count {}", snap.count);
    }

    /// Resets every counter and histogram to zero.
    pub fn reset(&self) {
        self.query_count.store(0, Ordering::Relaxed);
        self.query_errors.store(0, Ordering::Relaxed);
        self.query_timeouts.store(0, Ordering::Relaxed);
        self.query_latency.reset();
        self.query_count_by_language.reset();
        self.rows_returned.store(0, Ordering::Relaxed);
        self.rows_scanned.store(0, Ordering::Relaxed);

        self.tx_active.store(0, Ordering::Relaxed);
        self.tx_committed.store(0, Ordering::Relaxed);
        self.tx_rolled_back.store(0, Ordering::Relaxed);
        self.tx_conflicts.store(0, Ordering::Relaxed);
        self.tx_duration.reset();

        self.session_active.store(0, Ordering::Relaxed);
        self.session_created.store(0, Ordering::Relaxed);

        self.gc_runs.store(0, Ordering::Relaxed);
        self.gc_versions_reclaimed.store(0, Ordering::Relaxed);
        self.gc_duration.reset();

        self.wal_records_written.store(0, Ordering::Relaxed);
        self.wal_flush_duration.reset();
    }
}

impl Default for MetricsRegistry {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Snapshot
// ---------------------------------------------------------------------------

/// A serializable point-in-time view of all metrics.
///
/// Read, not built, outside this crate: later releases may add metrics.
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
#[non_exhaustive]
pub struct MetricsSnapshot {
    // -- Query --
    /// Total number of queries executed.
    pub query_count: u64,
    /// Number of queries that returned an error.
    pub query_errors: u64,
    /// Number of queries that timed out.
    pub query_timeouts: u64,
    /// 50th percentile query latency in milliseconds.
    pub query_latency_p50_ms: f64,
    /// 99th percentile query latency in milliseconds.
    pub query_latency_p99_ms: f64,
    /// Mean query latency in milliseconds.
    pub query_latency_mean_ms: f64,
    /// Total rows returned across all queries.
    pub rows_returned: u64,
    /// Total rows scanned across all queries.
    pub rows_scanned: u64,
    /// GQL queries executed.
    pub queries_gql: u64,
    /// Cypher queries executed.
    pub queries_cypher: u64,
    /// SPARQL queries executed.
    pub queries_sparql: u64,
    /// Gremlin queries executed.
    pub queries_gremlin: u64,
    /// GraphQL queries executed.
    pub queries_graphql: u64,
    /// SQL/PGQ queries executed.
    pub queries_sql_pgq: u64,

    // -- Transaction --
    /// Currently active (open) transactions.
    pub tx_active: i64,
    /// Total transactions committed.
    pub tx_committed: u64,
    /// Total transactions rolled back.
    pub tx_rolled_back: u64,
    /// Total transaction conflicts detected.
    pub tx_conflicts: u64,
    /// 50th percentile transaction duration in milliseconds.
    pub tx_duration_p50_ms: f64,
    /// 99th percentile transaction duration in milliseconds.
    pub tx_duration_p99_ms: f64,
    /// Mean transaction duration in milliseconds.
    pub tx_duration_mean_ms: f64,

    // -- Session --
    /// Currently active sessions.
    pub session_active: i64,
    /// Total sessions created.
    pub session_created: u64,

    // -- GC --
    /// Total garbage collection runs.
    pub gc_runs: u64,
    /// Total MVCC versions reclaimed by GC.
    pub gc_versions_reclaimed: u64,
    /// Mean GC pass duration in milliseconds.
    pub gc_duration_mean_ms: f64,

    // -- WAL --
    /// Total WAL records written.
    pub wal_records_written: u64,
    /// Mean WAL flush duration in milliseconds.
    pub wal_flush_duration_mean_ms: f64,

    // -- Cache --
    /// Total plan cache hits (parsed + optimized).
    pub cache_hits: u64,
    /// Total plan cache misses (parsed + optimized).
    pub cache_misses: u64,
    /// Current number of cached plans.
    pub cache_size: usize,
    /// Number of times the cache was invalidated (cleared due to DDL).
    pub cache_invalidations: u64,
}

// ---------------------------------------------------------------------------
// Macro
// ---------------------------------------------------------------------------

/// Records a metric on an `Option<Arc<MetricsRegistry>>`.
///
/// Compiles to a no-op when the `metrics` feature is disabled.
///
/// # Variants
///
/// - `record_metric!(reg, field, inc)` : fetch_add(1)
/// - `record_metric!(reg, field, dec)` : fetch_sub(1)
/// - `record_metric!(reg, field, add $value)` : fetch_add($value)
/// - `record_metric!(reg, field, observe $value)` : histogram observe
macro_rules! record_metric {
    ($registry:expr, $field:ident, inc) => {
        #[cfg(feature = "metrics")]
        if let Some(ref reg) = $registry {
            reg.$field
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        }
    };
    ($registry:expr, $field:ident, dec) => {
        #[cfg(feature = "metrics")]
        if let Some(ref reg) = $registry {
            reg.$field
                .fetch_sub(1, std::sync::atomic::Ordering::Relaxed);
        }
    };
    ($registry:expr, $field:ident, add $value:expr) => {
        #[cfg(feature = "metrics")]
        if let Some(ref reg) = $registry {
            reg.$field
                .fetch_add($value, std::sync::atomic::Ordering::Relaxed);
        }
    };
    ($registry:expr, $field:ident, observe $value:expr) => {
        #[cfg(feature = "metrics")]
        if let Some(ref reg) = $registry {
            reg.$field.observe($value);
        }
    };
}

pub(crate) use record_metric;

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn histogram_bucket_assignment() {
        let h = AtomicHistogram::new(LATENCY_BUCKETS);

        // Value exactly on a boundary goes into that bucket.
        h.observe(1.0);
        // Bucket index for 1.0: boundaries[3] == 1.0, so index 3.
        assert_eq!(h.buckets[3].load(Ordering::Relaxed), 1);

        // Value between boundaries.
        h.observe(0.3);
        // 0.3 > 0.25 (boundaries[1]) and <= 0.5 (boundaries[2]), so index 2.
        assert_eq!(h.buckets[2].load(Ordering::Relaxed), 1);

        // Value above all boundaries goes into overflow.
        h.observe(99999.0);
        let overflow_idx = LATENCY_BUCKETS.len();
        assert_eq!(h.buckets[overflow_idx].load(Ordering::Relaxed), 1);
    }

    #[test]
    fn histogram_percentile_accuracy() {
        let h = AtomicHistogram::new(LATENCY_BUCKETS);

        // Insert 100 observations at 1.0 ms.
        for _ in 0..100 {
            h.observe(1.0);
        }

        // p50 and p99 should both return the 1.0 bucket boundary.
        let p50 = h.percentile(0.50);
        assert!(
            (p50 - 1.0).abs() < f64::EPSILON,
            "expected p50 ~ 1.0, got {p50}"
        );

        let p99 = h.percentile(0.99);
        assert!(
            (p99 - 1.0).abs() < f64::EPSILON,
            "expected p99 ~ 1.0, got {p99}"
        );
    }

    #[test]
    fn histogram_mean() {
        let h = AtomicHistogram::new(LATENCY_BUCKETS);
        h.observe(2.0);
        h.observe(4.0);
        h.observe(6.0);

        let mean = h.mean();
        assert!(
            (mean - 4.0).abs() < f64::EPSILON,
            "expected mean 4.0, got {mean}"
        );
    }

    #[test]
    fn histogram_reset() {
        let h = AtomicHistogram::new(LATENCY_BUCKETS);
        h.observe(5.0);
        h.observe(10.0);
        assert_eq!(h.count.load(Ordering::Relaxed), 2);

        h.reset();

        assert_eq!(h.count.load(Ordering::Relaxed), 0);
        assert!((h.mean()).abs() < f64::EPSILON);
        for b in &h.buckets {
            assert_eq!(b.load(Ordering::Relaxed), 0);
        }
    }

    #[test]
    fn histogram_empty_percentile_and_mean() {
        let h = AtomicHistogram::new(LATENCY_BUCKETS);
        assert!((h.percentile(0.5)).abs() < f64::EPSILON);
        assert!((h.mean()).abs() < f64::EPSILON);
    }

    #[test]
    fn metrics_snapshot_serde_roundtrip() {
        let registry = MetricsRegistry::new();
        registry.query_count.store(42, Ordering::Relaxed);
        registry.query_errors.store(3, Ordering::Relaxed);
        registry.tx_committed.store(10, Ordering::Relaxed);
        registry.session_created.store(5, Ordering::Relaxed);

        let snapshot = registry.snapshot();
        let json = serde_json::to_string(&snapshot).expect("serialize");
        let deserialized: MetricsSnapshot = serde_json::from_str(&json).expect("deserialize");

        assert_eq!(deserialized.query_count, 42);
        assert_eq!(deserialized.query_errors, 3);
        assert_eq!(deserialized.tx_committed, 10);
        assert_eq!(deserialized.session_created, 5);
    }

    #[test]
    fn language_counters_increment() {
        let lc = LanguageCounters::new();
        lc.increment("gql");
        lc.increment("gql");
        lc.increment("cypher");
        lc.increment("sparql");
        lc.increment("gremlin");
        lc.increment("graphql");
        lc.increment("graphql-rdf");
        lc.increment("sql-pgq");
        lc.increment("sql");
        lc.increment("unknown_lang"); // should be ignored

        let snap = lc.snapshot();
        assert_eq!(snap.gql, 2);
        assert_eq!(snap.cypher, 1);
        assert_eq!(snap.sparql, 1);
        assert_eq!(snap.gremlin, 1);
        assert_eq!(snap.graphql, 2); // graphql + graphql-rdf
        assert_eq!(snap.sql_pgq, 2); // sql-pgq + sql
    }

    #[test]
    fn registry_reset() {
        let registry = MetricsRegistry::new();
        registry.query_count.fetch_add(10, Ordering::Relaxed);
        registry.tx_committed.fetch_add(5, Ordering::Relaxed);
        registry.session_active.fetch_add(3, Ordering::Relaxed);
        registry.query_latency.observe(42.0);
        registry.tx_duration.observe(10.0);
        registry.gc_runs.fetch_add(2, Ordering::Relaxed);

        registry.reset();

        let snap = registry.snapshot();
        assert_eq!(snap.query_count, 0);
        assert_eq!(snap.tx_committed, 0);
        assert_eq!(snap.session_active, 0);
        assert!((snap.query_latency_mean_ms).abs() < f64::EPSILON);
        assert!((snap.tx_duration_mean_ms).abs() < f64::EPSILON);
        assert_eq!(snap.gc_runs, 0);
    }

    #[test]
    fn record_metric_macro_with_some_registry() {
        let registry: Option<std::sync::Arc<MetricsRegistry>> =
            Some(std::sync::Arc::new(MetricsRegistry::new()));

        record_metric!(registry, query_count, inc);
        record_metric!(registry, query_count, inc);
        record_metric!(registry, query_errors, inc);
        record_metric!(registry, tx_active, inc);
        record_metric!(registry, tx_active, dec);
        record_metric!(registry, rows_returned, add 42u64);
        record_metric!(registry, query_latency, observe 5.0);

        let reg = registry.as_ref().unwrap();
        assert_eq!(reg.query_count.load(Ordering::Relaxed), 2);
        assert_eq!(reg.query_errors.load(Ordering::Relaxed), 1);
        assert_eq!(reg.tx_active.load(Ordering::Relaxed), 0);
        assert_eq!(reg.rows_returned.load(Ordering::Relaxed), 42);
        assert_eq!(reg.query_latency.count.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn record_metric_macro_with_none_registry() {
        let registry: Option<std::sync::Arc<MetricsRegistry>> = None;

        // Should not panic.
        record_metric!(registry, query_count, inc);
        record_metric!(registry, tx_active, dec);
        record_metric!(registry, rows_returned, add 10u64);
        record_metric!(registry, query_latency, observe 1.0);
    }

    #[test]
    fn histogram_snapshot_captures_state() {
        let h = AtomicHistogram::new(LATENCY_BUCKETS);
        h.observe(1.0);
        h.observe(5.0);
        h.observe(100.0);

        let snap = h.snapshot();
        assert_eq!(snap.count, 3);
        assert!((snap.sum - 106.0).abs() < f64::EPSILON);
        assert_eq!(snap.boundaries.len(), LATENCY_BUCKETS.len());
        assert_eq!(snap.bucket_counts.len(), LATENCY_BUCKETS.len() + 1);

        // Bucket for 1.0 (index 3) should have 1 observation
        assert_eq!(snap.bucket_counts[3], 1);
        // Bucket for 5.0 (index 5) should have 1 observation
        assert_eq!(snap.bucket_counts[5], 1);
        // Bucket for 100.0 (index 9) should have 1 observation
        assert_eq!(snap.bucket_counts[9], 1);
    }

    #[test]
    fn histogram_snapshot_serde_roundtrip() {
        let h = AtomicHistogram::new(LATENCY_BUCKETS);
        h.observe(2.5);
        h.observe(50.0);

        let snap = h.snapshot();
        let json = serde_json::to_string(&snap).expect("serialize");
        let deserialized: HistogramSnapshot = serde_json::from_str(&json).expect("deserialize");

        assert_eq!(deserialized.count, snap.count);
        assert!((deserialized.sum - snap.sum).abs() < f64::EPSILON);
        assert_eq!(deserialized.bucket_counts, snap.bucket_counts);
    }

    #[test]
    fn registry_default_is_zeroed() {
        let registry = MetricsRegistry::default();
        let snap = registry.snapshot();
        assert_eq!(snap.query_count, 0);
        assert_eq!(snap.tx_committed, 0);
        assert_eq!(snap.session_created, 0);
        assert_eq!(snap.gc_runs, 0);
        assert!((snap.query_latency_mean_ms).abs() < f64::EPSILON);
    }

    #[test]
    fn registry_snapshot_captures_all_fields() {
        let registry = MetricsRegistry::new();
        registry.query_count.fetch_add(100, Ordering::Relaxed);
        registry.query_errors.fetch_add(5, Ordering::Relaxed);
        registry.query_timeouts.fetch_add(2, Ordering::Relaxed);
        registry.rows_returned.fetch_add(1000, Ordering::Relaxed);
        registry.rows_scanned.fetch_add(5000, Ordering::Relaxed);
        registry.tx_active.fetch_add(3, Ordering::Relaxed);
        registry.tx_committed.fetch_add(50, Ordering::Relaxed);
        registry.tx_rolled_back.fetch_add(2, Ordering::Relaxed);
        registry.tx_conflicts.fetch_add(1, Ordering::Relaxed);
        registry.session_active.fetch_add(4, Ordering::Relaxed);
        registry.session_created.fetch_add(10, Ordering::Relaxed);
        registry.gc_runs.fetch_add(7, Ordering::Relaxed);
        registry.query_latency.observe(10.0);
        registry.tx_duration.observe(5.0);
        registry.query_count_by_language.increment("gql");
        registry.query_count_by_language.increment("cypher");
        registry.query_count_by_language.increment("sparql");
        registry.query_count_by_language.increment("gremlin");
        registry.query_count_by_language.increment("graphql");
        registry.query_count_by_language.increment("sql-pgq");

        let snap = registry.snapshot();
        assert_eq!(snap.query_count, 100);
        assert_eq!(snap.query_errors, 5);
        assert_eq!(snap.query_timeouts, 2);
        assert_eq!(snap.rows_returned, 1000);
        assert_eq!(snap.rows_scanned, 5000);
        assert_eq!(snap.tx_active, 3);
        assert_eq!(snap.tx_committed, 50);
        assert_eq!(snap.tx_rolled_back, 2);
        assert_eq!(snap.tx_conflicts, 1);
        assert_eq!(snap.session_active, 4);
        assert_eq!(snap.session_created, 10);
        assert_eq!(snap.gc_runs, 7);
        assert_eq!(snap.queries_gql, 1);
        assert_eq!(snap.queries_cypher, 1);
        assert_eq!(snap.queries_sparql, 1);
        assert_eq!(snap.queries_gremlin, 1);
        assert_eq!(snap.queries_graphql, 1);
        assert_eq!(snap.queries_sql_pgq, 1);
        assert!((snap.query_latency_mean_ms - 10.0).abs() < f64::EPSILON);
        assert!((snap.tx_duration_mean_ms - 5.0).abs() < f64::EPSILON);
    }

    #[test]
    fn histogram_overflow_percentile() {
        let h = AtomicHistogram::new(LATENCY_BUCKETS);
        // All observations above the highest boundary (10000.0)
        h.observe(20000.0);
        h.observe(50000.0);

        // p50 should return the last boundary as a lower-bound estimate
        let p50 = h.percentile(0.50);
        assert!(
            (p50 - 10000.0).abs() < f64::EPSILON,
            "overflow bucket should return last boundary, got {p50}"
        );
    }

    #[test]
    fn prometheus_output_format() {
        let registry = MetricsRegistry::new();
        registry.query_count.fetch_add(42, Ordering::Relaxed);
        registry.query_errors.fetch_add(3, Ordering::Relaxed);
        registry.tx_committed.fetch_add(10, Ordering::Relaxed);
        registry.session_created.fetch_add(5, Ordering::Relaxed);
        registry.gc_runs.fetch_add(2, Ordering::Relaxed);
        registry.query_latency.observe(1.0);
        registry.query_latency.observe(5.0);
        registry.query_count_by_language.increment("gql");

        let output = registry.to_prometheus();

        // Counters
        assert!(output.contains("# TYPE grafeo_query_count counter"));
        assert!(output.contains("grafeo_query_count 42"));
        assert!(output.contains("grafeo_query_errors 3"));
        assert!(output.contains("grafeo_tx_committed 10"));
        assert!(output.contains("grafeo_session_created 5"));
        assert!(output.contains("grafeo_gc_runs 2"));

        // Gauges
        assert!(output.contains("# TYPE grafeo_tx_active gauge"));
        assert!(output.contains("# TYPE grafeo_session_active gauge"));

        // Histogram
        assert!(output.contains("# TYPE grafeo_query_latency_ms histogram"));
        assert!(output.contains("grafeo_query_latency_ms_bucket{le=\"+Inf\"} 2"));
        assert!(output.contains("grafeo_query_latency_ms_count 2"));

        // Per-language
        assert!(output.contains("grafeo_query_count_by_language{language=\"gql\"} 1"));
    }

    #[test]
    fn snapshot_with_cache_merges_stats() {
        let registry = MetricsRegistry::new();
        registry.query_count.fetch_add(10, Ordering::Relaxed);

        let snap = registry.snapshot_with_cache(100, 20, 50, 3);
        assert_eq!(snap.query_count, 10);
        assert_eq!(snap.cache_hits, 100);
        assert_eq!(snap.cache_misses, 20);
        assert_eq!(snap.cache_size, 50);
        assert_eq!(snap.cache_invalidations, 3);
    }

    #[test]
    fn language_counters_reset() {
        let lc = LanguageCounters::new();
        lc.increment("gql");
        lc.increment("cypher");
        lc.increment("sparql");

        lc.reset();

        let snap = lc.snapshot();
        assert_eq!(snap.gql, 0);
        assert_eq!(snap.cypher, 0);
        assert_eq!(snap.sparql, 0);
        assert_eq!(snap.gremlin, 0);
        assert_eq!(snap.graphql, 0);
        assert_eq!(snap.sql_pgq, 0);
    }
}
