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
#[derive(Debug, Clone)]
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

    /// Returns the relationship table ID (encoded into [`EdgeId`] values).
    #[must_use]
    pub fn rel_table_id(&self) -> u16 {
        self.schema.rel_table_id
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

    /// Updates the relationship table ID embedded in the schema.
    ///
    /// Used by incremental compaction when tables are re-indexed after
    /// cloning from an old compact store.
    pub(crate) fn set_rel_table_id(&mut self, rel_table_id: u16) {
        self.schema.rel_table_id = rel_table_id;
    }

    /// Updates the source node table ID.
    ///
    /// Used by incremental compaction when node tables are re-indexed.
    pub(crate) fn set_src_table_id(&mut self, src_table_id: u16) {
        self.src_table_id = src_table_id;
    }

    /// Updates the destination node table ID.
    ///
    /// Used by incremental compaction when node tables are re-indexed.
    pub(crate) fn set_dst_table_id(&mut self, dst_table_id: u16) {
        self.dst_table_id = dst_table_id;
    }

    /// Returns the total number of edges in this table.
    #[must_use]
    pub fn num_edges(&self) -> usize {
        self.fwd.num_edges()
    }

    /// Returns `true` if a backward CSR is available.
    #[must_use]
    pub fn has_backward(&self) -> bool {
        self.bwd.is_some()
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

    /// Returns the property value for a specific edge (by CSR position) and key.
    #[must_use]
    pub fn get_edge_property(&self, csr_position: usize, key: &PropertyKey) -> Option<Value> {
        self.properties.get(key)?.get(csr_position)
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

    /// Returns all property keys present in this relationship table.
    #[must_use]
    pub fn property_keys(&self) -> Vec<PropertyKey> {
        self.properties.keys().cloned().collect()
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

    /// Returns the out-degree of a source node.
    #[must_use]
    pub fn out_degree(&self, src_offset: u32) -> usize {
        self.fwd.degree(src_offset)
    }

    /// Returns the in-degree of a target node, or `None` if no backward CSR.
    #[must_use]
    pub fn in_degree(&self, dst_offset: u32) -> Option<usize> {
        self.bwd.as_ref().map(|b| b.degree(dst_offset))
    }

    /// Returns the schema for this relationship table.
    #[must_use]
    pub(crate) fn schema_ref(&self) -> &EdgeSchema {
        &self.schema
    }

    /// Returns the forward CSR adjacency.
    #[must_use]
    pub(crate) fn fwd(&self) -> &CsrAdjacency {
        &self.fwd
    }

    /// Returns the optional backward CSR adjacency.
    #[must_use]
    pub(crate) fn bwd(&self) -> &Option<CsrAdjacency> {
        &self.bwd
    }

    /// Returns the edge properties map.
    #[must_use]
    pub(crate) fn edge_properties(&self) -> &FxHashMap<PropertyKey, ColumnCodec> {
        &self.properties
    }

    /// Returns an estimate of heap memory used by the CSR structures and
    /// edge property columns in bytes.
    #[must_use]
    pub fn memory_bytes(&self) -> usize {
        let fwd_bytes = self.fwd.memory_bytes();
        let bwd_bytes = self.bwd.as_ref().map_or(0, |b| b.memory_bytes());
        let prop_bytes: usize = self.properties.values().map(|c| c.heap_bytes()).sum();
        fwd_bytes + bwd_bytes + prop_bytes
    }
}

#[cfg(test)]
mod tests {
    use super::super::id::decode_node_id;
    use super::*;

    /// Helper to create a simple test scenario.
    ///
    /// 3 source nodes (table 0), 2 target nodes (table 1), 5 edges:
    ///   src0 -> dst0, src0 -> dst1
    ///   src1 -> dst0, src1 -> dst1
    ///   src2 -> dst0
    fn make_test_rel_table() -> RelTable {
        // Forward CSR: 3 source nodes.
        let fwd_edges = vec![(0u32, 0u32), (0, 1), (1, 0), (1, 1), (2, 0)];
        let fwd = CsrAdjacency::from_sorted_edges(3, &fwd_edges);

        // Backward CSR: 2 target nodes.
        // dst0 <- src0, src1, src2
        // dst1 <- src0, src1
        let bwd_edges = vec![(0u32, 0u32), (0, 1), (0, 2), (1, 0), (1, 1)];
        let mut bwd = CsrAdjacency::from_sorted_edges(2, &bwd_edges);

        // Pre-compute bwd-to-fwd position mapping and store as edge_data.
        let mut mapping = Vec::with_capacity(bwd_edges.len());
        for &(dst, src) in &bwd_edges {
            let fwd_start = fwd.offset_of(src);
            let local_idx = fwd.neighbors(src).iter().position(|&t| t == dst).unwrap();
            mapping.push(fwd_start + local_idx as u32);
        }
        bwd.set_edge_data(mapping);

        let schema = EdgeSchema::new("LIKES", 5, "Person", "Movie", vec![]);

        RelTable::new(
            schema,
            fwd,
            Some(bwd),
            FxHashMap::default(),
            0, // src_table_id
            1, // dst_table_id
        )
    }

    #[test]
    fn test_forward_traversal() {
        let rt = make_test_rel_table();

        assert_eq!(rt.edge_type().as_str(), "LIKES");
        assert_eq!(rt.rel_table_id(), 5);
        assert_eq!(rt.num_edges(), 5);
        assert!(rt.has_backward());

        // Source 0 -> targets [0, 1]
        let edges_0 = rt.edges_from_source(0);
        assert_eq!(edges_0.len(), 2);
        let (node_id_0, _edge_id_0) = edges_0[0];
        let (table, offset) = decode_node_id(node_id_0);
        assert_eq!(table, 1); // dst_table_id
        assert_eq!(offset, 0); // target offset 0

        let (node_id_1, _edge_id_1) = edges_0[1];
        let (table, offset) = decode_node_id(node_id_1);
        assert_eq!(table, 1);
        assert_eq!(offset, 1);

        // Source 1 -> targets [0, 1]
        let edges_1 = rt.edges_from_source(1);
        assert_eq!(edges_1.len(), 2);

        // Source 2 -> targets [0]
        let edges_2 = rt.edges_from_source(2);
        assert_eq!(edges_2.len(), 1);
        let (nid, _) = edges_2[0];
        let (table, offset) = decode_node_id(nid);
        assert_eq!(table, 1);
        assert_eq!(offset, 0);
    }

    #[test]
    fn test_backward_traversal() {
        let rt = make_test_rel_table();

        // Target 0 <- sources [0, 1, 2]
        let edges_to_0 = rt.edges_to_target(0).expect("backward CSR is present");
        assert_eq!(edges_to_0.len(), 3);
        let source_offsets: Vec<u64> = edges_to_0
            .iter()
            .map(|(nid, _)| decode_node_id(*nid).1)
            .collect();
        assert_eq!(source_offsets, vec![0, 1, 2]);

        // Target 1 <- sources [0, 1]
        let edges_to_1 = rt.edges_to_target(1).expect("backward CSR is present");
        assert_eq!(edges_to_1.len(), 2);
        let source_offsets: Vec<u64> = edges_to_1
            .iter()
            .map(|(nid, _)| decode_node_id(*nid).1)
            .collect();
        assert_eq!(source_offsets, vec![0, 1]);
    }

    #[test]
    fn test_degree() {
        let rt = make_test_rel_table();

        assert_eq!(rt.out_degree(0), 2);
        assert_eq!(rt.out_degree(1), 2);
        assert_eq!(rt.out_degree(2), 1);

        assert_eq!(rt.in_degree(0), Some(3));
        assert_eq!(rt.in_degree(1), Some(2));
    }

    #[test]
    fn test_no_backward_csr() {
        let fwd_edges = vec![(0u32, 1u32)];
        let fwd = CsrAdjacency::from_sorted_edges(2, &fwd_edges);
        let schema = EdgeSchema::new("FOLLOWS", 10, "User", "User", vec![]);

        let rt = RelTable::new(schema, fwd, None, FxHashMap::default(), 0, 0);

        assert!(!rt.has_backward());
        assert_eq!(rt.edges_to_target(0), None);
        assert_eq!(rt.in_degree(0), None);
    }
}
