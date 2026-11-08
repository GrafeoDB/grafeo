//! Hierarchical memory usage breakdown for the database.
//!
//! Store-level types (`StoreMemory`, `IndexMemory`, etc.) live in grafeo-common.
//! This module defines the top-level `MemoryUsage` aggregate and engine-specific
//! types (`CacheMemory`, `BufferManagerMemory`, `RdfMemory`, `CdcMemory`,
//! `SpillMemory`).
//!
//! The breakdowns are read, not built, outside this crate: later releases may
//! add fields. [`GrafeoDB::memory_usage`](crate::GrafeoDB::memory_usage)
//! returns one; another crate that needs its own starts from
//! `MemoryUsage::default()` and sets the fields it fills.
//!
//! # Stability: Experimental
//!
//! The call `GrafeoDB::memory_usage` is stable; the fields of these types,
//! and what each one counts, follow the storage engine and may change in any
//! release, with a CHANGELOG entry: see <https://grafeo.dev/stability/>.

pub use grafeo_common::memory::usage::{
    IndexMemory, MvccMemory, NamedMemory, StoreMemory, StringPoolMemory,
};
use serde::{Deserialize, Serialize};

/// Hierarchical memory usage breakdown for the entire database.
///
/// # Stability: Experimental
///
/// Its fields, and what each one counts, may change in any release (see the
/// [module documentation](crate::memory_usage)).
///
/// ```
/// use grafeo_engine::MemoryUsage;
///
/// let mut usage = MemoryUsage::default();
/// usage.store.nodes_bytes = 88;
/// usage.caches.total_bytes = 19;
/// usage.compute_total();
/// assert_eq!(usage.total_bytes, 19);
/// ```
///
/// ```compile_fail,E0639
/// use grafeo_engine::MemoryUsage;
///
/// let usage = MemoryUsage {
///     total_bytes: 88,
///     ..MemoryUsage::default()
/// };
/// ```
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[non_exhaustive]
pub struct MemoryUsage {
    /// Total estimated memory usage in bytes.
    pub total_bytes: usize,
    /// The memory limit in force, in bytes: the one set
    /// ([`Config::with_memory_limit`](crate::config::Config::with_memory_limit))
    /// or the default share of the detected memory. A write past it is
    /// refused with `GRAFEO-S001`.
    #[serde(default)]
    pub limit_bytes: usize,
    /// What one statement's operators may hold, in bytes: the query memory
    /// limit set ([`Config::with_query_memory_limit`](crate::config::Config::with_query_memory_limit)),
    /// never above `limit_bytes`; the global limit when none is set.
    #[serde(default)]
    pub query_limit_bytes: usize,
    /// Graph storage (nodes, edges, properties).
    pub store: StoreMemory,
    /// Index structures.
    pub indexes: IndexMemory,
    /// MVCC versioning overhead.
    pub mvcc: MvccMemory,
    /// Caches (query plans, etc.).
    pub caches: CacheMemory,
    /// String interning (ArcStr label/type registries).
    pub string_pool: StringPoolMemory,
    /// Buffer manager tracked allocations.
    pub buffer_manager: BufferManagerMemory,
    /// RDF triple store (only populated when the `triple-store` feature is enabled).
    #[serde(default, skip_serializing_if = "RdfMemory::is_empty")]
    pub rdf: RdfMemory,
    /// Change data capture log (only populated when the `cdc` feature is enabled).
    #[serde(default, skip_serializing_if = "CdcMemory::is_empty")]
    pub cdc: CdcMemory,
    /// What the open has spilled to disk, against its spill limit. Disk, not
    /// memory: not part of `total_bytes`.
    #[serde(default)]
    pub spill: SpillMemory,
}

/// What an open has spilled to disk, by who spilled it, and its cap
/// ([`Config::with_spill_limit`](crate::config::Config::with_spill_limit)).
///
/// ```compile_fail,E0639
/// use grafeo_engine::memory_usage::SpillMemory;
///
/// let spill = SpillMemory {
///     total_bytes: 88,
///     ..SpillMemory::default()
/// };
/// ```
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[non_exhaustive]
pub struct SpillMemory {
    /// The bytes spilled now, all spills together.
    pub total_bytes: u64,
    /// The spill limit; `None` when unlimited (the room for the next
    /// checkpoint still bounds), `Some(0)` for an open that spills nothing
    /// (no spill directory: in memory without a spill path, or encrypted).
    pub limit_bytes: Option<u64>,
    /// Sorts' runs.
    pub sort_bytes: u64,
    /// Aggregates' partitions.
    pub aggregate_bytes: u64,
    /// Vector indexes' spilled embedding columns.
    pub vector_column_bytes: u64,
}

impl MemoryUsage {
    /// Recomputes `total_bytes` from child totals, each byte once: the
    /// components (store, indexes, MVCC, caches, string pool, RDF, CDC), and
    /// of the buffer manager only what no component reports (the execution
    /// buffers but the CDC log's grant, and the spill staging). Its graph
    /// storage and index buffer regions hold the grants that cover the
    /// components, so they are not added again.
    pub fn compute_total(&mut self) {
        self.total_bytes = self.store.total_bytes
            + self.indexes.total_bytes
            + self.mvcc.total_bytes
            + self.caches.total_bytes
            + self.string_pool.total_bytes
            + self.rdf.total_bytes
            + self.cdc.total_bytes
            + self
                .buffer_manager
                .execution_buffers_bytes
                .saturating_sub(self.cdc.granted_bytes)
            + self.buffer_manager.spill_staging_bytes;
    }
}

/// Cache memory usage.
///
/// ```compile_fail,E0639
/// use grafeo_engine::memory_usage::CacheMemory;
///
/// let caches = CacheMemory {
///     cached_plan_count: 3,
///     ..CacheMemory::default()
/// };
/// ```
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[non_exhaustive]
pub struct CacheMemory {
    /// Total cache memory.
    pub total_bytes: usize,
    /// Parsed plan cache.
    pub parsed_plan_cache_bytes: usize,
    /// Optimized plan cache.
    pub optimized_plan_cache_bytes: usize,
    /// Number of cached plans (parsed + optimized).
    pub cached_plan_count: usize,
}

impl CacheMemory {
    /// Recomputes `total_bytes` from child values.
    pub fn compute_total(&mut self) {
        self.total_bytes = self.parsed_plan_cache_bytes + self.optimized_plan_cache_bytes;
    }
}

/// RDF triple store memory breakdown.
///
/// Default is empty (all zeros) when the `triple-store` feature is disabled,
/// so users on LPG-only builds see no RDF line in the hierarchical report.
///
/// ```compile_fail,E0639
/// use grafeo_engine::memory_usage::RdfMemory;
///
/// let rdf = RdfMemory {
///     triple_count: 88,
///     ..RdfMemory::default()
/// };
/// ```
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[non_exhaustive]
pub struct RdfMemory {
    /// Total estimated RDF memory in bytes.
    pub total_bytes: usize,
    /// Number of triples across the default graph and any named graphs.
    pub triple_count: usize,
    /// Primary triple set and all six index maps (subject, predicate, object, SP, PO, OS).
    pub triples_and_indexes_bytes: usize,
    /// Cached term dictionary bytes (None when no cache is warm).
    pub term_dictionary_bytes: usize,
    /// Cached Ring index bytes (only populated when `ring-index` is enabled).
    pub ring_index_bytes: usize,
    /// Named graphs in the default store (does not include nested graph memory,
    /// which is summed into `triples_and_indexes_bytes`).
    pub named_graph_count: usize,
}

impl RdfMemory {
    /// True when no RDF memory is reported. Used by `skip_serializing_if` so
    /// LPG-only builds don't emit an empty `rdf` block in JSON.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.total_bytes == 0 && self.triple_count == 0
    }

    /// Recomputes `total_bytes` from child values.
    pub fn compute_total(&mut self) {
        self.total_bytes =
            self.triples_and_indexes_bytes + self.term_dictionary_bytes + self.ring_index_bytes;
    }
}

/// CDC log memory breakdown.
///
/// Default is empty when the `cdc` feature is disabled.
///
/// ```compile_fail,E0639
/// use grafeo_engine::memory_usage::CdcMemory;
///
/// let cdc = CdcMemory {
///     event_count: 19,
///     ..CdcMemory::default()
/// };
/// ```
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[non_exhaustive]
pub struct CdcMemory {
    /// Total estimated CDC memory in bytes.
    pub total_bytes: usize,
    /// Number of entities with at least one recorded event.
    pub entity_count: usize,
    /// Total number of recorded change events across all entities.
    pub event_count: usize,
    /// The CDC log's grant, part of the buffer manager's execution buffers:
    /// [`MemoryUsage::compute_total`] counts the log once, as
    /// `total_bytes`, and leaves the grant out of the execution buffers.
    #[serde(default)]
    pub granted_bytes: usize,
}

impl CdcMemory {
    /// True when no CDC memory is reported.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.event_count == 0 && self.total_bytes == 0
    }
}

/// Buffer manager tracked allocations.
///
/// ```compile_fail,E0639
/// use grafeo_engine::memory_usage::BufferManagerMemory;
///
/// let buffer_manager = BufferManagerMemory {
///     budget_bytes: 88,
///     ..BufferManagerMemory::default()
/// };
/// ```
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[non_exhaustive]
pub struct BufferManagerMemory {
    /// Budget configured for the buffer manager.
    pub budget_bytes: usize,
    /// Currently allocated via grants.
    pub allocated_bytes: usize,
    /// The line past which a reservation is refused: the memory limit.
    #[serde(default)]
    pub hard_limit_bytes: usize,
    /// Graph storage region.
    pub graph_storage_bytes: usize,
    /// Index buffers region.
    pub index_buffers_bytes: usize,
    /// Execution buffers region.
    pub execution_buffers_bytes: usize,
    /// Spill staging region.
    pub spill_staging_bytes: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_memory_usage_is_zero() {
        let usage = MemoryUsage::default();
        assert_eq!(usage.total_bytes, 0);
        assert_eq!(usage.store.total_bytes, 0);
        assert_eq!(usage.indexes.total_bytes, 0);
        assert_eq!(usage.mvcc.total_bytes, 0);
        assert_eq!(usage.caches.total_bytes, 0);
        assert_eq!(usage.string_pool.total_bytes, 0);
        assert_eq!(usage.buffer_manager.allocated_bytes, 0);
    }

    #[test]
    fn compute_total_sums_children() {
        // The grafeo-common parts are `#[non_exhaustive]` here: set field by
        // field.
        let mut usage = MemoryUsage {
            caches: CacheMemory {
                total_bytes: 30,
                ..Default::default()
            },
            buffer_manager: BufferManagerMemory {
                // The grants of the components: not counted again.
                allocated_bytes: 1_000,
                graph_storage_bytes: 900,
                execution_buffers_bytes: 15,
                spill_staging_bytes: 5,
                ..Default::default()
            },
            rdf: RdfMemory {
                total_bytes: 500,
                triple_count: 10,
                triples_and_indexes_bytes: 500,
                ..Default::default()
            },
            cdc: CdcMemory {
                total_bytes: 40,
                event_count: 3,
                entity_count: 2,
                granted_bytes: 0,
            },
            ..Default::default()
        };
        usage.store.total_bytes = 100;
        usage.indexes.total_bytes = 200;
        usage.mvcc.total_bytes = 50;
        usage.string_pool.total_bytes = 10;
        usage.compute_total();
        assert_eq!(usage.total_bytes, 950);
    }

    #[test]
    fn the_cdc_grant_in_the_execution_buffers_is_counted_once() {
        // The log counts 40 bytes under a grant of 64 in the execution
        // buffers, beside 15 bytes of other execution buffers.
        let mut usage = MemoryUsage {
            buffer_manager: BufferManagerMemory {
                allocated_bytes: 79,
                execution_buffers_bytes: 79,
                ..Default::default()
            },
            cdc: CdcMemory {
                total_bytes: 40,
                event_count: 3,
                entity_count: 2,
                granted_bytes: 64,
            },
            ..Default::default()
        };
        usage.compute_total();
        assert_eq!(usage.total_bytes, 40 + 15);

        // CDC off: no grant, the execution buffers count whole.
        usage.cdc = CdcMemory::default();
        usage.buffer_manager.execution_buffers_bytes = 15;
        usage.compute_total();
        assert_eq!(usage.total_bytes, 15);
    }

    #[test]
    fn rdf_and_cdc_default_is_empty() {
        let rdf = RdfMemory::default();
        assert!(rdf.is_empty());
        let cdc = CdcMemory::default();
        assert!(cdc.is_empty());
    }

    #[test]
    fn rdf_compute_total_sums_children() {
        let mut rdf = RdfMemory {
            triples_and_indexes_bytes: 100,
            term_dictionary_bytes: 50,
            ring_index_bytes: 25,
            ..Default::default()
        };
        rdf.compute_total();
        assert_eq!(rdf.total_bytes, 175);
    }

    #[test]
    fn serde_roundtrip() {
        let mut usage = MemoryUsage::default();
        usage.store.nodes_bytes = 1024;
        usage.indexes.vector_indexes.push(NamedMemory {
            name: "vec_idx".to_string(),
            bytes: 512,
            item_count: 100,
        });
        usage.mvcc.average_chain_depth = 1.5;

        let json = serde_json::to_string(&usage).unwrap();
        let deserialized: MemoryUsage = serde_json::from_str(&json).unwrap();

        assert_eq!(deserialized.store.nodes_bytes, 1024);
        assert_eq!(deserialized.indexes.vector_indexes.len(), 1);
        assert_eq!(deserialized.indexes.vector_indexes[0].name, "vec_idx");
        assert!((deserialized.mvcc.average_chain_depth - 1.5).abs() < f64::EPSILON);
    }
}
