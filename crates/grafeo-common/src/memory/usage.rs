//! Memory usage breakdown types for graph store components.
//!
//! These types live in grafeo-common so both grafeo-core (which implements
//! the estimations) and grafeo-engine (which aggregates them) can use them.

use serde::{Deserialize, Serialize};

/// Memory used by the graph store (nodes, edges, properties).
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct StoreMemory {
    /// Total store memory.
    pub total_bytes: usize,
    /// Node record storage (hash map buckets + `NodeRecord` data).
    pub nodes_bytes: usize,
    /// Edge record storage (hash map buckets + `EdgeRecord` data).
    pub edges_bytes: usize,
    /// Node property columns.
    pub node_properties_bytes: usize,
    /// Edge property columns.
    pub edge_properties_bytes: usize,
    /// Number of property columns (node + edge).
    pub property_column_count: usize,
    /// Hash-map slot bytes for node property columns (capacity × entry).
    #[serde(default)]
    pub node_property_map_slot_bytes: usize,
    /// Decoded `Value` payload bytes inside node property columns (strings/lists/…).
    #[serde(default)]
    pub node_property_decoded_payload_bytes: usize,
    /// String/bytes payload subset of node property decoded values.
    #[serde(default)]
    pub node_property_string_payload_bytes: usize,
    /// Unused map capacity waste in node property columns.
    #[serde(default)]
    pub node_property_capacity_waste_bytes: usize,
    /// Hash-map slot bytes for edge property columns.
    #[serde(default)]
    pub edge_property_map_slot_bytes: usize,
    /// Decoded `Value` payload bytes inside edge property columns.
    #[serde(default)]
    pub edge_property_decoded_payload_bytes: usize,
    /// String/bytes payload subset of edge property decoded values.
    #[serde(default)]
    pub edge_property_string_payload_bytes: usize,
    /// Unused map capacity waste in edge property columns.
    #[serde(default)]
    pub edge_property_capacity_waste_bytes: usize,
}

impl StoreMemory {
    /// Recomputes `total_bytes` from child values.
    pub fn compute_total(&mut self) {
        self.total_bytes = self.nodes_bytes
            + self.edges_bytes
            + self.node_properties_bytes
            + self.edge_properties_bytes;
    }
}

/// Per-column residency attribution for one property key.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PropertyColumnMemory {
    /// Property key name.
    pub key: String,
    /// Live entries in the hot map.
    pub entry_count: usize,
    /// Hot map capacity (buckets reserved).
    pub map_capacity: usize,
    /// `capacity × (Id + Value + 1)` slot estimate.
    pub map_slot_bytes: usize,
    /// Sum of [`crate::types::Value::estimated_size_bytes`] over hot values.
    pub decoded_payload_bytes: usize,
    /// String + bytes payload subset of decoded values.
    pub string_payload_bytes: usize,
    /// Compressed column backing (0 when uncompressed).
    pub compressed_bytes: usize,
    /// `(capacity − len) × entry` waste in the hot map.
    pub capacity_waste_bytes: usize,
    /// Values held only in compressed form (not in the hot map).
    pub compressed_entry_count: usize,
}

impl PropertyColumnMemory {
    /// Estimated total for this column (slots + payloads + compressed).
    #[must_use]
    pub fn total_bytes(&self) -> usize {
        self.map_slot_bytes + self.decoded_payload_bytes + self.compressed_bytes
    }
}

/// Aggregated property-storage residency (node or edge side).
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PropertyStorageMemory {
    /// Outer column-map overhead.
    pub map_overhead_bytes: usize,
    /// Per-column breakdown (sorted by `total_bytes` descending when produced).
    pub columns: Vec<PropertyColumnMemory>,
    /// Sum of column map-slot bytes.
    pub total_map_slot_bytes: usize,
    /// Sum of decoded payloads.
    pub total_decoded_payload_bytes: usize,
    /// Sum of string/bytes payloads.
    pub total_string_payload_bytes: usize,
    /// Sum of compressed backings.
    pub total_compressed_bytes: usize,
    /// Sum of hot-map capacity waste.
    pub total_capacity_waste_bytes: usize,
    /// `map_overhead + Σ column totals`.
    pub total_bytes: usize,
}

impl PropertyStorageMemory {
    /// Recomputes aggregate totals from columns + outer map overhead.
    pub fn compute_total(&mut self) {
        self.total_map_slot_bytes = self.columns.iter().map(|c| c.map_slot_bytes).sum();
        self.total_decoded_payload_bytes =
            self.columns.iter().map(|c| c.decoded_payload_bytes).sum();
        self.total_string_payload_bytes =
            self.columns.iter().map(|c| c.string_payload_bytes).sum();
        self.total_compressed_bytes = self.columns.iter().map(|c| c.compressed_bytes).sum();
        self.total_capacity_waste_bytes =
            self.columns.iter().map(|c| c.capacity_waste_bytes).sum();
        self.total_bytes = self.map_overhead_bytes
            + self.total_map_slot_bytes
            + self.total_decoded_payload_bytes
            + self.total_compressed_bytes;
    }
}

/// Adjacency list capacity vs used-byte attribution.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct AdjacencyCapacityMemory {
    /// Nodes with an adjacency list.
    pub node_count: usize,
    /// Outer list-map capacity.
    pub list_map_capacity: usize,
    /// Outer list-map overhead bytes.
    pub list_map_overhead_bytes: usize,
    /// Bytes occupied by live hot entries (destinations + edge ids used len).
    pub hot_used_bytes: usize,
    /// Bytes reserved by hot chunk Vec capacities.
    pub hot_capacity_bytes: usize,
    /// Compressed cold-chunk bytes.
    pub cold_bytes: usize,
    /// Delta / deleted / skip-index reserved bytes.
    pub aux_capacity_bytes: usize,
    /// `hot_capacity − hot_used` (+ list map slack approximated separately).
    pub capacity_waste_bytes: usize,
    /// Total estimated heap (`list_map_overhead + hot_capacity + cold + aux`).
    pub total_bytes: usize,
}

impl AdjacencyCapacityMemory {
    /// Recomputes `capacity_waste_bytes` and `total_bytes`.
    pub fn compute_total(&mut self) {
        self.capacity_waste_bytes = self.hot_capacity_bytes.saturating_sub(self.hot_used_bytes);
        self.total_bytes = self.list_map_overhead_bytes
            + self.hot_capacity_bytes
            + self.cold_bytes
            + self.aux_capacity_bytes;
    }
}

/// Full LPG residency attribution (properties + adjacency capacities).
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct LpgResidencyMemory {
    /// Node property columns.
    pub node_properties: PropertyStorageMemory,
    /// Edge property columns.
    pub edge_properties: PropertyStorageMemory,
    /// Forward adjacency capacity detail.
    pub forward_adjacency: AdjacencyCapacityMemory,
    /// Backward adjacency capacity detail (empty when disabled).
    pub backward_adjacency: AdjacencyCapacityMemory,
    /// Sum of property + adjacency totals.
    pub total_bytes: usize,
}

impl LpgResidencyMemory {
    /// Recomputes `total_bytes` from children.
    pub fn compute_total(&mut self) {
        self.node_properties.compute_total();
        self.edge_properties.compute_total();
        self.forward_adjacency.compute_total();
        self.backward_adjacency.compute_total();
        self.total_bytes = self.node_properties.total_bytes
            + self.edge_properties.total_bytes
            + self.forward_adjacency.total_bytes
            + self.backward_adjacency.total_bytes;
    }
}

/// Memory used by index structures.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct IndexMemory {
    /// Total index memory.
    pub total_bytes: usize,
    /// Forward adjacency lists.
    pub forward_adjacency_bytes: usize,
    /// Backward adjacency lists (0 if disabled).
    pub backward_adjacency_bytes: usize,
    /// Forward adjacency capacity waste (`capacity − used` on hot chunks).
    #[serde(default)]
    pub forward_adjacency_capacity_waste_bytes: usize,
    /// Backward adjacency capacity waste.
    #[serde(default)]
    pub backward_adjacency_capacity_waste_bytes: usize,
    /// Label index (label_id -> node set).
    pub label_index_bytes: usize,
    /// Node-to-labels reverse index.
    pub node_labels_bytes: usize,
    /// Property value indexes.
    pub property_index_bytes: usize,
    /// Per-index breakdown for vector indexes.
    pub vector_indexes: Vec<NamedMemory>,
    /// Per-index breakdown for text indexes.
    pub text_indexes: Vec<NamedMemory>,
}

impl IndexMemory {
    /// Recomputes `total_bytes` from child values.
    pub fn compute_total(&mut self) {
        self.total_bytes = self.forward_adjacency_bytes
            + self.backward_adjacency_bytes
            + self.label_index_bytes
            + self.node_labels_bytes
            + self.property_index_bytes
            + self.vector_indexes.iter().map(|v| v.bytes).sum::<usize>()
            + self.text_indexes.iter().map(|t| t.bytes).sum::<usize>();
    }
}

/// Memory usage for a named component (e.g., a specific index).
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct NamedMemory {
    /// Component name.
    pub name: String,
    /// Estimated heap bytes.
    pub bytes: usize,
    /// Number of items.
    pub item_count: usize,
}

/// MVCC versioning overhead.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct MvccMemory {
    /// Total MVCC overhead.
    pub total_bytes: usize,
    /// Version chain overhead for nodes.
    pub node_version_chains_bytes: usize,
    /// Version chain overhead for edges.
    pub edge_version_chains_bytes: usize,
    /// Average version chain depth.
    pub average_chain_depth: f64,
    /// Maximum version chain depth seen.
    pub max_chain_depth: usize,
}

impl MvccMemory {
    /// Recomputes `total_bytes` from child values.
    pub fn compute_total(&mut self) {
        self.total_bytes = self.node_version_chains_bytes + self.edge_version_chains_bytes;
    }
}

/// Memory used by label/edge type registries.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct StringPoolMemory {
    /// Total bytes for label/type registries.
    pub total_bytes: usize,
    /// Label registry (names + ID maps).
    pub label_registry_bytes: usize,
    /// Edge type registry (names + ID maps).
    pub edge_type_registry_bytes: usize,
    /// Number of interned labels.
    pub label_count: usize,
    /// Number of interned edge types.
    pub edge_type_count: usize,
}

impl StringPoolMemory {
    /// Recomputes `total_bytes` from child values.
    pub fn compute_total(&mut self) {
        self.total_bytes = self.label_registry_bytes + self.edge_type_registry_bytes;
    }
}
