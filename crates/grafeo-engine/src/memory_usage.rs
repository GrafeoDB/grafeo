//! Hierarchical memory usage breakdown for the database.
//!
//! Store-level types (`StoreMemory`, `IndexMemory`, etc.) live in grafeo-common.
//! This module defines the top-level `MemoryUsage` aggregate and engine-specific
//! types (`CacheMemory`, `BufferManagerMemory`, `RdfMemory`, `CdcMemory`).

pub use grafeo_common::memory::usage::{
    AdjacencyCapacityMemory, IndexMemory, LpgResidencyMemory, MvccMemory, NamedMemory,
    PropertyColumnMemory, PropertyStorageMemory, StoreMemory, StringPoolMemory,
};
use serde::{Deserialize, Serialize};

/// Hierarchical memory usage breakdown for the entire database.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct MemoryUsage {
    /// Total estimated memory usage in bytes.
    pub total_bytes: usize,
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
}

impl MemoryUsage {
    /// Recomputes `total_bytes` from child totals.
    pub fn compute_total(&mut self) {
        self.total_bytes = self.store.total_bytes
            + self.indexes.total_bytes
            + self.mvcc.total_bytes
            + self.caches.total_bytes
            + self.string_pool.total_bytes
            + self.buffer_manager.allocated_bytes
            + self.rdf.total_bytes
            + self.cdc.total_bytes;
    }
}

/// Cache memory usage.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
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
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
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
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CdcMemory {
    /// Total estimated CDC memory in bytes.
    pub total_bytes: usize,
    /// Number of entities with at least one recorded event.
    pub entity_count: usize,
    /// Total number of recorded change events across all entities.
    pub event_count: usize,
}

impl CdcMemory {
    /// True when no CDC memory is reported.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.event_count == 0 && self.total_bytes == 0
    }
}

/// Buffer manager tracked allocations.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct BufferManagerMemory {
    /// Budget configured for the buffer manager.
    pub budget_bytes: usize,
    /// Currently allocated via grants.
    pub allocated_bytes: usize,
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
        let mut usage = MemoryUsage {
            store: StoreMemory {
                total_bytes: 100,
                ..Default::default()
            },
            indexes: IndexMemory {
                total_bytes: 200,
                ..Default::default()
            },
            mvcc: MvccMemory {
                total_bytes: 50,
                ..Default::default()
            },
            caches: CacheMemory {
                total_bytes: 30,
                ..Default::default()
            },
            string_pool: StringPoolMemory {
                total_bytes: 10,
                ..Default::default()
            },
            buffer_manager: BufferManagerMemory {
                allocated_bytes: 20,
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
            },
            ..Default::default()
        };
        usage.compute_total();
        assert_eq!(usage.total_bytes, 950);
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
