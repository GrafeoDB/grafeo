//! Memory usage breakdown types for graph store components.
//!
//! These types live in grafeo-common so both grafeo-core (which implements
//! the estimations) and grafeo-engine (which aggregates them) can use them.
//!
//! The breakdowns are read, not built, outside this crate: later releases may
//! add fields. Another crate starts from `Default::default()` and sets the
//! fields it fills.

use serde::{Deserialize, Serialize};

/// Memory used by the graph store (nodes, edges, properties).
///
/// ```compile_fail,E0639
/// use grafeo_common::memory::usage::StoreMemory;
///
/// let store = StoreMemory {
///     nodes_bytes: 88,
///     ..StoreMemory::default()
/// };
/// ```
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[non_exhaustive]
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

/// Memory used by index structures.
///
/// ```compile_fail,E0639
/// use grafeo_common::memory::usage::IndexMemory;
///
/// let indexes = IndexMemory {
///     label_index_bytes: 88,
///     ..IndexMemory::default()
/// };
/// ```
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[non_exhaustive]
pub struct IndexMemory {
    /// Total index memory.
    pub total_bytes: usize,
    /// Forward adjacency lists.
    pub forward_adjacency_bytes: usize,
    /// Backward adjacency lists (0 if disabled).
    pub backward_adjacency_bytes: usize,
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
///
/// ```compile_fail,E0639
/// use grafeo_common::memory::usage::MvccMemory;
///
/// let mvcc = MvccMemory {
///     max_chain_depth: 3,
///     ..MvccMemory::default()
/// };
/// ```
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[non_exhaustive]
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
///
/// ```compile_fail,E0639
/// use grafeo_common::memory::usage::StringPoolMemory;
///
/// let string_pool = StringPoolMemory {
///     label_count: 19,
///     ..StringPoolMemory::default()
/// };
/// ```
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[non_exhaustive]
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
