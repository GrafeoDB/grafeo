//! Memory introspection for `LpgStore`.

use super::LpgStore;
use grafeo_common::memory::usage::{
    IndexMemory, LpgResidencyMemory, MvccMemory, StoreMemory, StringPoolMemory,
};
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

    /// Property-column / string-payload / adjacency-capacity residency detail.
    ///
    /// Closes the `memory_usage()` blind spot where map-slot estimates omitted
    /// decoded `Value` heap (especially ArcStr) and adjacency capacity waste.
    #[must_use]
    pub fn lpg_residency_detail(&self) -> LpgResidencyMemory {
        let mut detail = LpgResidencyMemory {
            node_properties: self.node_properties.memory_detail(),
            edge_properties: self.edge_properties.memory_detail(),
            forward_adjacency: self.forward_adj.capacity_memory(),
            backward_adjacency: self
                .backward_adj
                .as_ref()
                .map_or_else(Default::default, |adj| adj.capacity_memory()),
            ..Default::default()
        };
        detail.compute_total();
        detail
    }

    /// Shrinks property-column and adjacency Vec/HashMap capacities in place.
    ///
    /// Does not change logical graph contents. Intended for residency
    /// experiments after bulk open/deserialize.
    pub fn shrink_lpg_capacities(&self) {
        self.node_properties.shrink_capacities();
        self.edge_properties.shrink_capacities();
        self.forward_adj.shrink_capacities();
        if let Some(adj) = &self.backward_adj {
            adj.shrink_capacities();
        }
    }

    /// Forces dictionary/int/bool compression on all property columns.
    ///
    /// Point reads remain correct via lazy compressed `get` decode.
    pub fn force_compress_properties(&self) {
        self.node_properties.force_compress_all();
        self.edge_properties.force_compress_all();
    }

    fn store_memory(&self) -> StoreMemory {
        let node_detail = self.node_properties.memory_detail();
        let edge_detail = self.edge_properties.memory_detail();
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

        let mut store = StoreMemory {
            nodes_bytes,
            edges_bytes,
            node_properties_bytes: node_detail.total_bytes,
            edge_properties_bytes: edge_detail.total_bytes,
            property_column_count: col_count,
            node_property_map_slot_bytes: node_detail.total_map_slot_bytes,
            node_property_decoded_payload_bytes: node_detail.total_decoded_payload_bytes,
            node_property_string_payload_bytes: node_detail.total_string_payload_bytes,
            node_property_capacity_waste_bytes: node_detail.total_capacity_waste_bytes,
            edge_property_map_slot_bytes: edge_detail.total_map_slot_bytes,
            edge_property_decoded_payload_bytes: edge_detail.total_decoded_payload_bytes,
            edge_property_string_payload_bytes: edge_detail.total_string_payload_bytes,
            edge_property_capacity_waste_bytes: edge_detail.total_capacity_waste_bytes,
            ..Default::default()
        };
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

            let mut mvcc = MvccMemory {
                node_version_chains_bytes: node_chain_bytes,
                edge_version_chains_bytes: edge_chain_bytes,
                average_chain_depth,
                max_chain_depth: max_depth,
                ..Default::default()
            };
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
            let mvcc = MvccMemory {
                node_version_chains_bytes: node_chain_bytes,
                edge_version_chains_bytes: edge_chain_bytes,
                average_chain_depth: if total_chains > 0 { 1.0 } else { 0.0 },
                max_chain_depth: usize::from(total_chains > 0),
                total_bytes: node_chain_bytes + edge_chain_bytes,
            };
            (mvcc, 0)
        }
    }

    fn index_memory(&self) -> IndexMemory {
        let forward_detail = self.forward_adj.capacity_memory();
        let backward_detail = self
            .backward_adj
            .as_ref()
            .map(|adj| adj.capacity_memory());
        let forward_bytes = forward_detail.total_bytes;
        let backward_bytes = backward_detail
            .as_ref()
            .map_or(0, |d| d.total_bytes);

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
            .map(|dmap| {
                // DashMap: approximate as capacity * entry size
                dmap.len()
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
                .map(|(name, idx)| NamedMemory {
                    name: name.clone(),
                    bytes: idx.heap_memory_bytes(),
                    item_count: idx.len(),
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

        let mut indexes = IndexMemory {
            forward_adjacency_bytes: forward_bytes,
            backward_adjacency_bytes: backward_bytes,
            forward_adjacency_capacity_waste_bytes: forward_detail.capacity_waste_bytes,
            backward_adjacency_capacity_waste_bytes: backward_detail
                .as_ref()
                .map_or(0, |d| d.capacity_waste_bytes),
            label_index_bytes,
            node_labels_bytes,
            property_index_bytes,
            vector_indexes,
            text_indexes,
            ..Default::default()
        };
        indexes.compute_total();
        indexes
    }

    fn string_pool_memory(&self) -> StringPoolMemory {
        let label_reg = self.label_registry.read();
        let edge_type_to_id = self.edge_type_to_id.read();
        let id_to_edge_type = self.id_to_edge_type.read();

        let label_count = label_reg.len();
        let edge_type_count = id_to_edge_type.len();

        let label_registry_bytes = label_reg.heap_bytes();

        // Same for edge types
        let et_map_bytes = edge_type_to_id.capacity()
            * (size_of::<arcstr::ArcStr>() + size_of::<u32>() + 1)
            + edge_type_to_id.keys().map(|s| s.len()).sum::<usize>();
        let et_vec_bytes = id_to_edge_type.capacity() * size_of::<arcstr::ArcStr>()
            + id_to_edge_type.iter().map(|s| s.len()).sum::<usize>();
        let edge_type_registry_bytes = et_map_bytes + et_vec_bytes;

        let mut sp = StringPoolMemory {
            label_registry_bytes,
            edge_type_registry_bytes,
            label_count,
            edge_type_count,
            ..Default::default()
        };
        sp.compute_total();
        sp
    }
}
