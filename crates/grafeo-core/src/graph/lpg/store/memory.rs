//! Memory introspection for `LpgStore`.

use super::LpgStore;
use grafeo_common::memory::usage::{IndexMemory, MvccMemory, StoreMemory, StringPoolMemory};
use std::mem::size_of;

impl LpgStore {
    /// Returns a detailed memory breakdown of the store.
    ///
    /// Acquires read locks on internal structures briefly. Safe to call
    /// concurrently with queries, but sub-totals may be slightly
    /// inconsistent if mutations are in progress.
    #[must_use]
    pub fn memory_breakdown(&self) -> (StoreMemory, IndexMemory, MvccMemory, StringPoolMemory) {
        let store = self.store_memory();
        let (mvcc, _) = self.mvcc_memory();
        let indexes = self.index_memory();
        let string_pool = self.string_pool_memory();
        (store, indexes, mvcc, string_pool)
    }

    fn store_memory(&self) -> StoreMemory {
        let node_props_bytes = self.node_properties.heap_memory_bytes();
        let edge_props_bytes = self.edge_properties.heap_memory_bytes();
        let col_count = self.node_properties.column_count() + self.edge_properties.column_count();

        // Node/edge map overhead (excluding version chain internals, which go to MVCC)
        #[cfg(not(feature = "tiered-storage"))]
        let (nodes_bytes, edges_bytes) = {
            let nodes = self.nodes.read();
            let edges = self.edges.read();
            let n = nodes.capacity()
                * (size_of::<grafeo_common::types::NodeId>()
                    + size_of::<grafeo_common::mvcc::VersionChain<super::super::NodeRecord>>()
                    + 1);
            let e = edges.capacity()
                * (size_of::<grafeo_common::types::EdgeId>()
                    + size_of::<grafeo_common::mvcc::VersionChain<super::super::EdgeRecord>>()
                    + 1);
            (n, e)
        };
        #[cfg(feature = "tiered-storage")]
        let (nodes_bytes, edges_bytes) = {
            let nv = self.node_versions.read();
            let ev = self.edge_versions.read();
            let n = nv.capacity()
                * (size_of::<grafeo_common::types::NodeId>()
                    + size_of::<grafeo_common::mvcc::VersionIndex>()
                    + 1);
            let e = ev.capacity()
                * (size_of::<grafeo_common::types::EdgeId>()
                    + size_of::<grafeo_common::mvcc::VersionIndex>()
                    + 1);
            (n, e)
        };

        // `#[non_exhaustive]` in grafeo-common: built field by field.
        let mut store = StoreMemory::default();
        store.nodes_bytes = nodes_bytes;
        store.edges_bytes = edges_bytes;
        store.node_properties_bytes = node_props_bytes;
        store.edge_properties_bytes = edge_props_bytes;
        store.property_column_count = col_count;
        store.compute_total();
        store
    }

    fn mvcc_memory(&self) -> (MvccMemory, usize) {
        #[cfg(not(feature = "tiered-storage"))]
        {
            let nodes = self.nodes.read();
            let edges = self.edges.read();

            let mut node_chain_bytes = 0usize;
            let mut edge_chain_bytes = 0usize;
            let mut total_depth = 0usize;
            let mut max_depth = 0usize;
            let total_chains = nodes.len() + edges.len();

            for chain in nodes.values() {
                let depth = chain.version_count();
                node_chain_bytes += chain.heap_memory_bytes();
                total_depth += depth;
                if depth > max_depth {
                    max_depth = depth;
                }
            }
            for chain in edges.values() {
                let depth = chain.version_count();
                edge_chain_bytes += chain.heap_memory_bytes();
                total_depth += depth;
                if depth > max_depth {
                    max_depth = depth;
                }
            }

            let average_chain_depth = if total_chains > 0 {
                total_depth as f64 / total_chains as f64
            } else {
                0.0
            };

            let mut mvcc = MvccMemory::default();
            mvcc.node_version_chains_bytes = node_chain_bytes;
            mvcc.edge_version_chains_bytes = edge_chain_bytes;
            mvcc.average_chain_depth = average_chain_depth;
            mvcc.max_chain_depth = max_depth;
            mvcc.compute_total();
            (mvcc, 0)
        }

        #[cfg(feature = "tiered-storage")]
        {
            // Tiered storage uses VersionIndex (SmallVec-based) plus arena storage.
            // Approximate: count entries in version index maps.
            let nv = self.node_versions.read();
            let ev = self.edge_versions.read();
            let node_count = nv.len();
            let edge_count = ev.len();
            // Each VersionIndex is a SmallVec; estimate ~64 bytes per entry
            let node_chain_bytes = node_count * 64;
            let edge_chain_bytes = edge_count * 64;
            let total_chains = node_count + edge_count;
            let mut mvcc = MvccMemory::default();
            mvcc.node_version_chains_bytes = node_chain_bytes;
            mvcc.edge_version_chains_bytes = edge_chain_bytes;
            mvcc.average_chain_depth = if total_chains > 0 { 1.0 } else { 0.0 };
            mvcc.max_chain_depth = usize::from(total_chains > 0);
            mvcc.compute_total();
            (mvcc, 0)
        }
    }

    fn index_memory(&self) -> IndexMemory {
        let forward_bytes = self.forward_adj.heap_memory_bytes();
        let backward_bytes = self
            .backward_adj
            .as_ref()
            .map_or(0, |adj| adj.heap_memory_bytes());

        // Label index: Vec<FxHashMap<NodeId, ()>>
        let label_idx = self.label_index.read();
        let label_index_bytes: usize = label_idx
            .iter()
            .map(|map| map.capacity() * (size_of::<grafeo_common::types::NodeId>() + 1))
            .sum::<usize>()
            + label_idx.capacity()
                * size_of::<grafeo_common::utils::hash::FxHashMap<grafeo_common::types::NodeId, ()>>(
                );
        drop(label_idx);

        // Node labels
        let node_labels = self.node_labels.read();
        #[cfg(not(feature = "temporal"))]
        let node_labels_bytes = node_labels.capacity()
            * (size_of::<grafeo_common::types::NodeId>()
                + size_of::<grafeo_common::utils::hash::FxHashSet<u32>>()
                + 1)
            + node_labels
                .values()
                .map(|set| set.capacity() * (size_of::<u32>() + 1))
                .sum::<usize>();
        #[cfg(feature = "temporal")]
        let node_labels_bytes = node_labels.capacity()
            * (size_of::<grafeo_common::types::NodeId>()
                + size_of::<
                    grafeo_common::temporal::VersionLog<grafeo_common::utils::hash::FxHashSet<u32>>,
                >()
                + 1)
            + node_labels
                .values()
                .map(|log| log.len() * size_of::<grafeo_common::utils::hash::FxHashSet<u32>>())
                .sum::<usize>();
        drop(node_labels);

        // Property indexes
        let prop_indexes = self.property_indexes.read();
        let property_index_bytes: usize = prop_indexes
            .values()
            .map(|index| {
                // DashMap: approximate as capacity * entry size
                index.len()
                    * (size_of::<grafeo_common::types::HashableValue>()
                        + size_of::<
                            grafeo_common::utils::hash::FxHashSet<grafeo_common::types::NodeId>,
                        >()
                        + 32)
            })
            .sum();
        drop(prop_indexes);

        // Vector indexes
        #[cfg(feature = "vector-index")]
        let vector_indexes: Vec<grafeo_common::memory::NamedMemory> = {
            use grafeo_common::memory::NamedMemory;
            let vidx = self.vector_indexes.read();
            vidx.iter()
                .map(|(name, stored)| NamedMemory {
                    name: name.clone(),
                    bytes: stored.index.heap_memory_bytes(),
                    item_count: stored.index.len(),
                })
                .collect()
        };
        #[cfg(not(feature = "vector-index"))]
        let vector_indexes = Vec::new();

        // Text indexes
        #[cfg(feature = "text-index")]
        let text_indexes: Vec<grafeo_common::memory::NamedMemory> = {
            use grafeo_common::memory::NamedMemory;
            let tidx = self.text_indexes.read();
            tidx.iter()
                .map(|(name, idx)| {
                    let guard = idx.read();
                    NamedMemory {
                        name: name.clone(),
                        bytes: guard.heap_memory_bytes(),
                        item_count: guard.len(),
                    }
                })
                .collect()
        };
        #[cfg(not(feature = "text-index"))]
        let text_indexes = Vec::new();

        let mut indexes = IndexMemory::default();
        indexes.forward_adjacency_bytes = forward_bytes;
        indexes.backward_adjacency_bytes = backward_bytes;
        indexes.label_index_bytes = label_index_bytes;
        indexes.node_labels_bytes = node_labels_bytes;
        indexes.property_index_bytes = property_index_bytes;
        indexes.vector_indexes = vector_indexes;
        indexes.text_indexes = text_indexes;
        indexes.compute_total();
        indexes
    }

    fn string_pool_memory(&self) -> StringPoolMemory {
        let label_reg = self.label_registry.read();
        let edge_types = self.edge_types.read();

        let label_count = label_reg.len();
        let edge_type_count = edge_types.len();

        let label_registry_bytes = label_reg.heap_bytes();
        let edge_type_registry_bytes = edge_types.heap_bytes();

        let mut sp = StringPoolMemory::default();
        sp.label_registry_bytes = label_registry_bytes;
        sp.edge_type_registry_bytes = edge_type_registry_bytes;
        sp.label_count = label_count;
        sp.edge_type_count = edge_type_count;
        sp.compute_total();
        sp
    }
}
