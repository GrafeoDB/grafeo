//! Relationship table: double-indexed CSR for a single edge type.
//!
//! Stores all edges of one type with optional forward and backward CSR.
//! Edge properties are columnar, parallel to the forward CSR targets.

use arcstr::ArcStr;
use grafeo_common::types::{EdgeId, NodeId, PropertyKey, Value};
use grafeo_common::utils::hash::FxHashMap;

use super::column::ColumnCodec;
use super::csr::CsrAdjacency;
use super::id::{encode_edge_id, encode_node_id};
use super::schema::EdgeSchema;

/// A relationship table holding all edges of a single type.
///
/// Edges are stored in a forward CSR indexed by source node offset, with an
/// optional backward CSR indexed by target node offset. Edge properties are
/// stored in columnar format, parallel to the forward CSR targets array
/// (i.e. the property at index `i` corresponds to the edge at CSR position `i`).
#[derive(Debug)]
pub struct RelTable {
    /// Schema describing the edge type and connected node labels.
    schema: EdgeSchema,
    /// Forward CSR, indexed by source node offset.
    fwd: CsrAdjacency,
    /// Backward CSR, indexed by target node offset. `None` means backward
    /// traversal falls back to a full scan of the forward CSR.
    /// When present, its `edge_data` stores the corresponding forward CSR
    /// position for each backward edge.
    bwd: Option<CsrAdjacency>,
    /// Edge properties, keyed by property name, parallel to forward CSR targets.
    properties: FxHashMap<PropertyKey, ColumnCodec>,
    /// Table ID of the source node table.
    src_table_id: u16,
    /// Table ID of the destination node table.
    dst_table_id: u16,
}

impl RelTable {
    /// Creates a new relationship table.
    ///
    /// # Panics
    ///
    /// Panics if `bwd` is a non-empty CSR that has no edge data populated.
    #[must_use]
    pub fn new(
        schema: EdgeSchema,
        fwd: CsrAdjacency,
        bwd: Option<CsrAdjacency>,
        properties: FxHashMap<PropertyKey, ColumnCodec>,
        src_table_id: u16,
        dst_table_id: u16,
    ) -> Self {
        if let Some(ref b) = bwd {
            assert!(
                b.has_edge_data() || b.num_edges() == 0,
                "backward CSR must have edge_data populated"
            );
        }
        Self {
            schema,
            fwd,
            bwd,
            properties,
            src_table_id,
            dst_table_id,
        }
    }

    /// Returns the edge type name (e.g. `"KNOWS"`).
    #[must_use]
    pub fn edge_type(&self) -> &ArcStr {
        &self.schema.edge_type
    }

    /// Returns the table ID of the source node table.
    #[must_use]
    pub fn src_table_id(&self) -> u16 {
        self.src_table_id
    }

    /// Returns the table ID of the destination node table.
    #[must_use]
    pub fn dst_table_id(&self) -> u16 {
        self.dst_table_id
    }

    /// Returns the total number of edges in this table.
    #[must_use]
    pub fn num_edges(&self) -> usize {
        self.fwd.num_edges()
    }

    /// Returns all edges originating from the given source node.
    ///
    /// Each result is a `(target_NodeId, EdgeId)` pair where the `EdgeId`
    /// encodes this table's `rel_table_id` and the forward CSR position.
    #[must_use]
    pub fn edges_from_source(&self, src_offset: u32) -> Vec<(NodeId, EdgeId)> {
        let neighbors = self.fwd.neighbors(src_offset);
        let start_pos = u64::from(self.fwd.offset_of(src_offset));
        let rel_id = self.schema.rel_table_id;

        neighbors
            .iter()
            .enumerate()
            .map(|(i, &target_offset)| {
                let node_id = encode_node_id(self.dst_table_id, u64::from(target_offset));
                let edge_id = encode_edge_id(rel_id, start_pos + i as u64);
                (node_id, edge_id)
            })
            .collect()
    }

    /// Returns all edges pointing to the given target node.
    ///
    /// Returns `None` if no backward CSR is available. Each result is a
    /// `(source_NodeId, EdgeId)` pair. The `EdgeId` is derived from the
    /// *forward* CSR position for stability.
    #[must_use]
    pub fn edges_to_target(&self, dst_offset: u32) -> Option<Vec<(NodeId, EdgeId)>> {
        let bwd = self.bwd.as_ref()?;
        let bwd_start = bwd.offset_of(dst_offset) as usize;
        let source_offsets = bwd.neighbors(dst_offset);
        let rel_id = self.schema.rel_table_id;

        let results = source_offsets
            .iter()
            .enumerate()
            .filter_map(|(i, &src_offset)| {
                // O(1) lookup via edge_data stored on the backward CSR.
                // Returns None if edge_data was not populated on backward CSR.
                let fwd_pos = bwd.edge_data_at(bwd_start + i)?;
                let node_id = encode_node_id(self.src_table_id, u64::from(src_offset));
                let edge_id = encode_edge_id(rel_id, u64::from(fwd_pos));
                Some((node_id, edge_id))
            })
            .collect();

        Some(results)
    }

    /// Returns all properties for the edge at the given forward CSR position.
    #[must_use]
    pub fn get_all_edge_properties(&self, csr_position: usize) -> FxHashMap<PropertyKey, Value> {
        let mut props = FxHashMap::default();
        for (key, col) in &self.properties {
            if let Some(value) = col.get(csr_position) {
                props.insert(key.clone(), value);
            }
        }
        props
    }

    /// Returns the source [`NodeId`] for the edge at the given forward CSR position.
    #[must_use]
    pub fn source_node_id(&self, csr_position: u32) -> Option<NodeId> {
        let src_offset = self.fwd.source_for_position(csr_position)?;
        Some(encode_node_id(self.src_table_id, u64::from(src_offset)))
    }

    /// Returns the destination [`NodeId`] for the edge at the given forward CSR position.
    #[must_use]
    pub fn dest_node_id(&self, csr_position: u32) -> Option<NodeId> {
        let src = self.fwd.source_for_position(csr_position)?;
        let start = self.fwd.offset_of(src);
        let local_idx = (csr_position - start) as usize;
        let target_offset = *self.fwd.neighbors(src).get(local_idx)?;
        Some(encode_node_id(self.dst_table_id, u64::from(target_offset)))
    }
}
