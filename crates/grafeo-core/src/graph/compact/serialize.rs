//! Serialization for CompactStore via bincode.
//!
//! `CompactStore` cannot derive `Serialize`/`Deserialize` directly because it
//! contains `FxHashMap` fields (non-deterministic iteration order) and
//! `Arc<Statistics>` (should be recomputed on load). Instead we use a set of
//! proxy structs that convert hashmaps to sorted vecs and drop statistics.

use arcstr::ArcStr;
use grafeo_common::types::PropertyKey;
use grafeo_common::utils::hash::FxHashMap;

use super::CompactStore;
use super::column::ColumnCodec;
use super::csr::CsrAdjacency;
use super::node_table::NodeTable;
use super::rel_table::RelTable;
use super::schema::{EdgeSchema, TableSchema};
use super::zone_map::ZoneMap;
use crate::statistics::{EdgeTypeStatistics, LabelStatistics, Statistics};

// ---------------------------------------------------------------------------
// Proxy types
// ---------------------------------------------------------------------------

/// Serializable mirror of [`NodeTable`] with sorted vecs instead of hashmaps.
#[derive(serde::Serialize, serde::Deserialize)]
struct NodeTableProxy {
    schema: TableSchema,
    /// Sorted by PropertyKey for deterministic output.
    columns: Vec<(PropertyKey, ColumnCodec)>,
    /// Sorted by PropertyKey for deterministic output.
    zone_maps: Vec<(PropertyKey, ZoneMap)>,
    len: usize,
}

impl NodeTableProxy {
    fn from_node_table(nt: &NodeTable) -> Self {
        let mut columns: Vec<_> = nt
            .columns()
            .iter()
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect();
        columns.sort_by(|a, b| a.0.as_ref().cmp(b.0.as_ref()));

        let mut zone_maps: Vec<_> = nt
            .zone_maps()
            .iter()
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect();
        zone_maps.sort_by(|a, b| a.0.as_ref().cmp(b.0.as_ref()));

        NodeTableProxy {
            schema: nt.schema().clone(),
            columns,
            zone_maps,
            len: nt.len(),
        }
    }

    fn into_node_table(self) -> NodeTable {
        let columns: FxHashMap<PropertyKey, ColumnCodec> = self.columns.into_iter().collect();
        let zone_maps: FxHashMap<PropertyKey, ZoneMap> = self.zone_maps.into_iter().collect();
        NodeTable::from_columns(self.schema, columns, zone_maps, self.len)
    }
}

/// Serializable mirror of [`RelTable`] with sorted vecs instead of hashmaps.
#[derive(serde::Serialize, serde::Deserialize)]
struct RelTableProxy {
    schema: EdgeSchema,
    fwd: CsrAdjacency,
    bwd: Option<CsrAdjacency>,
    /// Sorted by PropertyKey for deterministic output.
    properties: Vec<(PropertyKey, ColumnCodec)>,
    src_table_id: u16,
    dst_table_id: u16,
}

impl RelTableProxy {
    fn from_rel_table(rt: &RelTable) -> Self {
        let mut properties: Vec<_> = rt
            .edge_properties()
            .iter()
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect();
        properties.sort_by(|a, b| a.0.as_ref().cmp(b.0.as_ref()));

        RelTableProxy {
            schema: rt.schema_ref().clone(),
            fwd: rt.fwd().clone(),
            bwd: rt.bwd().clone(),
            properties,
            src_table_id: rt.src_table_id(),
            dst_table_id: rt.dst_table_id(),
        }
    }

    fn into_rel_table(self) -> RelTable {
        let properties: FxHashMap<PropertyKey, ColumnCodec> =
            self.properties.into_iter().collect();
        RelTable::new(
            self.schema,
            self.fwd,
            self.bwd,
            properties,
            self.src_table_id,
            self.dst_table_id,
        )
    }
}

/// Serializable mirror of [`CompactStore`].
#[derive(serde::Serialize, serde::Deserialize)]
struct CompactStoreProxy {
    node_tables_by_id: Vec<NodeTableProxy>,
    rel_tables_by_id: Vec<RelTableProxy>,
    /// Sorted by label for deterministic output.
    label_to_table_id: Vec<(ArcStr, u16)>,
    /// Sorted by edge type for deterministic output.
    edge_type_to_rel_id: Vec<(ArcStr, Vec<u16>)>,
    table_id_to_label: Vec<ArcStr>,
    rel_table_id_to_type: Vec<ArcStr>,
}

impl CompactStoreProxy {
    fn from_compact_store(store: &CompactStore) -> Self {
        let node_tables_by_id = store
            .node_tables_by_id()
            .iter()
            .map(NodeTableProxy::from_node_table)
            .collect();

        let rel_tables_by_id = store
            .rel_tables_by_id()
            .iter()
            .map(RelTableProxy::from_rel_table)
            .collect();

        let mut label_to_table_id: Vec<_> = store
            .label_to_table_id_map()
            .iter()
            .map(|(k, &v)| (k.clone(), v))
            .collect();
        label_to_table_id.sort_by(|a, b| a.0.cmp(&b.0));

        let mut edge_type_to_rel_id: Vec<_> = store
            .edge_type_to_rel_id_map()
            .iter()
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect();
        edge_type_to_rel_id.sort_by(|a, b| a.0.cmp(&b.0));

        CompactStoreProxy {
            node_tables_by_id,
            rel_tables_by_id,
            label_to_table_id,
            edge_type_to_rel_id,
            table_id_to_label: store.table_id_to_label().to_vec(),
            rel_table_id_to_type: store.rel_table_id_to_type_slice().to_vec(),
        }
    }

    fn into_compact_store(self) -> CompactStore {
        let node_tables_by_id: Vec<NodeTable> = self
            .node_tables_by_id
            .into_iter()
            .map(|p| p.into_node_table())
            .collect();

        let rel_tables_by_id: Vec<RelTable> = self
            .rel_tables_by_id
            .into_iter()
            .map(|p| p.into_rel_table())
            .collect();

        let label_to_table_id: FxHashMap<ArcStr, u16> =
            self.label_to_table_id.into_iter().collect();

        let edge_type_to_rel_id: FxHashMap<ArcStr, Vec<u16>> =
            self.edge_type_to_rel_id.into_iter().collect();

        // Recompute statistics (they are not persisted).
        let mut stats = Statistics::new();
        let mut total_nodes: u64 = 0;
        let mut total_edges: u64 = 0;

        for (idx, nt) in node_tables_by_id.iter().enumerate() {
            let count = nt.len() as u64;
            total_nodes += count;
            let label = &self.table_id_to_label[idx];
            stats.update_label(label.as_str(), LabelStatistics::new(count));
        }

        let mut edge_type_counts: FxHashMap<&str, u64> = FxHashMap::default();
        for (idx, rt) in rel_tables_by_id.iter().enumerate() {
            let count = rt.num_edges() as u64;
            total_edges += count;
            let edge_type = &self.rel_table_id_to_type[idx];
            *edge_type_counts.entry(edge_type.as_str()).or_default() += count;
        }
        for (edge_type, count) in edge_type_counts {
            stats.update_edge_type(edge_type, EdgeTypeStatistics::new(count, 0.0, 0.0));
        }

        stats.total_nodes = total_nodes;
        stats.total_edges = total_edges;

        CompactStore::new(
            node_tables_by_id,
            label_to_table_id,
            rel_tables_by_id,
            edge_type_to_rel_id,
            self.table_id_to_label,
            self.rel_table_id_to_type,
            stats,
        )
    }
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

impl CompactStore {
    /// Serializes the store to bytes using bincode.
    ///
    /// Statistics are **not** included; they will be recomputed on
    /// [`from_bytes`](Self::from_bytes).
    ///
    /// # Errors
    ///
    /// Returns `Err` if bincode serialization fails.
    pub fn to_bytes(&self) -> Result<Vec<u8>, String> {
        let proxy = CompactStoreProxy::from_compact_store(self);
        bincode::serde::encode_to_vec(&proxy, bincode::config::standard())
            .map_err(|e| format!("serialization failed: {e}"))
    }

    /// Deserializes a store from bytes produced by [`to_bytes`](Self::to_bytes).
    ///
    /// Statistics are recomputed after deserialization.
    ///
    /// # Errors
    ///
    /// Returns `Err` if the bytes are invalid or cannot be decoded.
    pub fn from_bytes(data: &[u8]) -> Result<Self, String> {
        let (proxy, _): (CompactStoreProxy, _) =
            bincode::serde::decode_from_slice(data, bincode::config::standard())
                .map_err(|e| format!("deserialization failed: {e}"))?;
        Ok(proxy.into_compact_store())
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::GraphStore;
    use crate::graph::compact::CompactStoreBuilder;

    #[test]
    fn roundtrip_empty() {
        let store = CompactStoreBuilder::new().build().unwrap();
        let bytes = store.to_bytes().unwrap();
        let restored = CompactStore::from_bytes(&bytes).unwrap();
        assert_eq!(restored.node_count(), 0);
        assert_eq!(restored.edge_count(), 0);
    }

    #[test]
    fn roundtrip_nodes() {
        let grades: Vec<u64> = (0..50).collect();
        let names: Vec<&str> = (0..50).map(|i| if i % 2 == 0 { "A" } else { "B" }).collect();
        let active: Vec<bool> = (0..50).map(|i| i % 3 == 0).collect();

        let store = CompactStoreBuilder::new()
            .node_table("Student", |b| {
                b.column_bitpacked("grade", &grades, 8)
                    .column_dict("name", &names)
                    .column_bitmap("active", &active)
            })
            .build()
            .unwrap();

        let bytes = store.to_bytes().unwrap();
        let restored = CompactStore::from_bytes(&bytes).unwrap();

        assert_eq!(restored.node_count(), 50);
        assert_eq!(restored.nodes_by_label("Student").len(), 50);

        let id = restored.nodes_by_label("Student")[0];
        let node = restored.get_node(id).unwrap();
        assert!(node
            .properties
            .contains_key(&grafeo_common::types::PropertyKey::new("grade")));
        assert!(node
            .properties
            .contains_key(&grafeo_common::types::PropertyKey::new("name")));
    }

    #[test]
    fn roundtrip_with_edges() {
        let grades: Vec<u64> = (0..10).collect();
        let levels: Vec<u64> = (0..5).collect();
        let edges: Vec<(u32, u32)> = (0..10).map(|i| (i, i % 5)).collect();

        let store = CompactStoreBuilder::new()
            .node_table("Student", |b| b.column_bitpacked("grade", &grades, 8))
            .node_table("Course", |b| b.column_bitpacked("level", &levels, 4))
            .rel_table("ENROLLED_IN", "Student", "Course", |b| {
                b.edges(edges).backward(true)
            })
            .build()
            .unwrap();

        let bytes = store.to_bytes().unwrap();
        let restored = CompactStore::from_bytes(&bytes).unwrap();

        assert_eq!(restored.node_count(), 15);
        assert_eq!(restored.edge_count(), 10);
        assert_eq!(restored.all_edge_types(), vec!["ENROLLED_IN".to_string()]);

        let student_0 = restored.nodes_by_label("Student")[0];
        let edges = restored.edges_from(student_0, crate::graph::Direction::Outgoing);
        assert!(!edges.is_empty());
    }

    #[test]
    fn roundtrip_all_column_types() {
        let ints: Vec<u64> = vec![0, 100, 200, 300];
        let strings: Vec<&str> = vec!["alpha", "beta", "gamma", "delta"];
        let bools: Vec<bool> = vec![true, false, true, false];

        let store = CompactStoreBuilder::new()
            .node_table("Mixed", |b| {
                b.column_bitpacked("score", &ints, 16)
                    .column_dict("label", &strings)
                    .column_bitmap("flag", &bools)
            })
            .build()
            .unwrap();

        let bytes = store.to_bytes().unwrap();
        let restored = CompactStore::from_bytes(&bytes).unwrap();

        let id = restored.nodes_by_label("Mixed")[2];
        let node = restored.get_node(id).unwrap();
        assert_eq!(
            node.properties
                .get(&grafeo_common::types::PropertyKey::new("score")),
            Some(&grafeo_common::types::Value::Int64(200))
        );
        assert_eq!(
            node.properties
                .get(&grafeo_common::types::PropertyKey::new("label")),
            Some(&grafeo_common::types::Value::String(arcstr::literal!("gamma")))
        );
        assert_eq!(
            node.properties
                .get(&grafeo_common::types::PropertyKey::new("flag")),
            Some(&grafeo_common::types::Value::Bool(true))
        );
    }
}
