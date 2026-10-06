//! LPG section serializer for the `.grafeo` container format.
//!
//! Implements the [`Section`] trait for LPG graph data (nodes, edges,
//! properties, named graphs). Uses the block-based binary format (v2)
//! defined in the `block` submodule for efficient serialization, CRC integrity
//! checking, and future mmap support.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use grafeo_common::storage::section::{Section, SectionType};
use grafeo_common::types::{EpochId, Value};
use grafeo_common::utils::error::Result;

use super::block::{self, BlockEdge, BlockNamedGraph, BlockNode};
use crate::graph::lpg::LpgStore;

/// Current LPG section format version (v2 = block-based).
const LPG_SECTION_VERSION: u8 = 2;

// ── Collection helpers ──────────────────────────────────────────────

/// The nodes of `store` as the section writes them, read one at a time. A
/// node record or a spilled property value that cannot be read fails the
/// checkpoint instead of leaving the file without it.
fn collect_block_nodes(store: &LpgStore) -> Result<Vec<BlockNode>> {
    // With `temporal` the section holds each property's history, read below:
    // the current values are not read.
    #[cfg(feature = "temporal")]
    let source = store
        .try_nodes_without_properties()?
        .map(Ok::<_, grafeo_common::utils::error::Error>);
    #[cfg(not(feature = "temporal"))]
    let source = store.try_nodes()?;
    let mut nodes: Vec<BlockNode> = Vec::new();
    for n in source {
        let n = n?;
        #[cfg(feature = "temporal")]
        let mut properties: Vec<(String, Vec<(EpochId, Value)>)> = store
            .node_property_history(n.id)
            .into_iter()
            .map(|(k, entries)| (k.to_string(), entries))
            .collect();

        #[cfg(not(feature = "temporal"))]
        let mut properties: Vec<(String, Vec<(EpochId, Value)>)> = n
            .properties
            .into_iter()
            .map(|(k, v)| (k.to_string(), vec![(EpochId::new(0), v)]))
            .collect();

        properties.sort_by(|(a, _), (b, _)| a.cmp(b));

        let mut labels: Vec<String> = n.labels.iter().map(|l| l.to_string()).collect();
        labels.sort();

        nodes.push(BlockNode {
            id: n.id,
            labels,
            properties,
        });
    }
    nodes.sort_by_key(|n| n.id);
    Ok(nodes)
}

fn collect_block_edges(store: &LpgStore) -> Vec<BlockEdge> {
    let mut edges: Vec<BlockEdge> = store
        .all_edges()
        .map(|e| {
            #[cfg(feature = "temporal")]
            let mut properties: Vec<(String, Vec<(EpochId, Value)>)> = store
                .edge_property_history(e.id)
                .into_iter()
                .map(|(k, entries)| (k.to_string(), entries))
                .collect();

            #[cfg(not(feature = "temporal"))]
            let mut properties: Vec<(String, Vec<(EpochId, Value)>)> = e
                .properties
                .into_iter()
                .map(|(k, v)| (k.to_string(), vec![(EpochId::new(0), v)]))
                .collect();

            properties.sort_by(|(a, _), (b, _)| a.cmp(b));

            BlockEdge {
                id: e.id,
                src: e.src,
                dst: e.dst,
                edge_type: e.edge_type.to_string(),
                properties,
            }
        })
        .collect();
    edges.sort_by_key(|e| e.id);
    edges
}

fn populate_store(store: &LpgStore, nodes: &[BlockNode], edges: &[BlockEdge]) -> Result<()> {
    for node in nodes {
        let label_refs: Vec<&str> = node.labels.iter().map(|s| s.as_str()).collect();
        store.create_node_with_id(node.id, &label_refs)?;
        for (key, entries) in &node.properties {
            #[cfg(feature = "temporal")]
            for (epoch, value) in entries {
                store.set_node_property_at_epoch(node.id, key, value.clone(), *epoch);
            }
            #[cfg(not(feature = "temporal"))]
            if let Some((_, value)) = entries.last() {
                store.set_node_property(node.id, key, value.clone());
            }
        }
    }
    for edge in edges {
        store.create_edge_with_id(edge.id, edge.src, edge.dst, &edge.edge_type)?;
        for (key, entries) in &edge.properties {
            #[cfg(feature = "temporal")]
            for (epoch, value) in entries {
                store.set_edge_property_at_epoch(edge.id, key, value.clone(), *epoch);
            }
            #[cfg(not(feature = "temporal"))]
            if let Some((_, value)) = entries.last() {
                store.set_edge_property(edge.id, key, value.clone());
            }
        }
    }
    Ok(())
}

// ── Section implementation ──────────────────────────────────────────

/// LPG store section for the `.grafeo` container.
///
/// Wraps an `Arc<LpgStore>` and implements the [`Section`] trait for
/// serialization/deserialization of LPG graph data using the block-based
/// format (v2).
pub struct LpgStoreSection {
    store: Arc<LpgStore>,
    dirty: AtomicBool,
}

impl LpgStoreSection {
    /// Create a new LPG section wrapping the given store.
    pub fn new(store: Arc<LpgStore>) -> Self {
        Self {
            store,
            dirty: AtomicBool::new(false),
        }
    }

    /// Mark this section as dirty (has unsaved changes).
    pub fn mark_dirty(&self) {
        self.dirty.store(true, Ordering::Release);
    }

    /// Access the underlying store.
    #[must_use]
    pub fn store(&self) -> &Arc<LpgStore> {
        &self.store
    }
}

impl Section for LpgStoreSection {
    fn section_type(&self) -> SectionType {
        SectionType::LpgStore
    }

    fn version(&self) -> u8 {
        LPG_SECTION_VERSION
    }

    fn serialize(&self) -> Result<Vec<u8>> {
        let nodes = collect_block_nodes(&self.store)?;
        let edges = collect_block_edges(&self.store);

        let mut named_graphs: Vec<BlockNamedGraph> = Vec::new();
        for name in self.store.graph_names() {
            if let Some(graph_store) = self.store.graph(&name) {
                named_graphs.push(BlockNamedGraph {
                    name,
                    nodes: collect_block_nodes(&graph_store)?,
                    edges: collect_block_edges(&graph_store),
                });
            }
        }

        #[cfg(feature = "temporal")]
        let epoch = self.store.current_epoch().as_u64();
        #[cfg(not(feature = "temporal"))]
        let epoch = 0u64;

        block::write_blocks(&nodes, &edges, &named_graphs, epoch)
    }

    fn deserialize(&mut self, data: &[u8]) -> Result<()> {
        let store = &self.store;

        block::read_blocks(data, &mut |nodes, edges, named_graphs, epoch| {
            populate_store(store, &nodes, &edges)?;

            #[cfg(feature = "temporal")]
            store.sync_epoch(EpochId::new(epoch));
            #[cfg(not(feature = "temporal"))]
            let _ = epoch;

            for graph in &named_graphs {
                store
                    .create_graph(&graph.name)
                    .map_err(|e| grafeo_common::utils::error::Error::Internal(e.to_string()))?;
                if let Some(graph_store) = store.graph(&graph.name) {
                    populate_store(&graph_store, &graph.nodes, &graph.edges)?;
                    #[cfg(feature = "temporal")]
                    graph_store.sync_epoch(EpochId::new(epoch));
                }
            }

            Ok(())
        })
    }

    fn is_dirty(&self) -> bool {
        self.dirty.load(Ordering::Acquire)
    }

    fn mark_clean(&self) {
        self.dirty.store(false, Ordering::Release);
    }

    fn memory_usage(&self) -> usize {
        let (store, indexes, mvcc, string_pool) = self.store.memory_breakdown();
        store.total_bytes + indexes.total_bytes + mvcc.total_bytes + string_pool.total_bytes
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use grafeo_common::types::{NodeId, PropertyKey, Value};

    /// A checkpoint of a spilled column writes its values, and fails rather
    /// than write the column without one it cannot read (#594).
    #[cfg(not(feature = "temporal"))]
    #[test]
    fn a_checkpoint_reads_spilled_values_and_fails_on_an_unreadable_one() {
        use crate::graph::lpg::property::test_backing::MemoryBacking;

        let store = Arc::new(LpgStore::new().unwrap());
        let key = PropertyKey::new("embedding");
        let alix = store.create_node_with_props(
            &["Item"],
            [("embedding", Value::Vector(vec![3.0, 19.0].into()))],
        );
        let snapshot = store.node_property_column_entries(&key).unwrap();
        let backing = MemoryBacking::of(&snapshot);
        assert!(store.spill_node_property_column(&key, backing.clone(), &snapshot));

        let section = LpgStoreSection::new(Arc::clone(&store));
        let bytes = section.serialize().unwrap();
        let copy = Arc::new(LpgStore::new().unwrap());
        LpgStoreSection::new(Arc::clone(&copy))
            .deserialize(&bytes)
            .unwrap();
        assert_eq!(
            copy.get_node_property(alix, &key),
            Some(Value::Vector(vec![3.0, 19.0].into()))
        );

        backing.fail_reads(true);
        assert!(
            section.serialize().is_err(),
            "a checkpoint without the value"
        );
    }

    #[test]
    fn lpg_section_round_trip() {
        let store = Arc::new(LpgStore::new().unwrap());
        store.create_node(&["Person"]);
        store.create_node(&["Person"]);
        let n1 = NodeId::new(1);
        let n2 = NodeId::new(2);
        store.set_node_property(n1, "name", Value::String("Alix".into()));
        store.set_node_property(n2, "name", Value::String("Gus".into()));
        store.create_edge(n1, n2, "KNOWS");

        let section = LpgStoreSection::new(Arc::clone(&store));
        let bytes = section.serialize().expect("serialize should succeed");
        assert!(!bytes.is_empty(), "bytes is empty");
        assert!(block::is_block_format(&bytes));

        // Deserialize into a fresh store
        let store2 = Arc::new(LpgStore::new().unwrap());
        let mut section2 = LpgStoreSection::new(store2);
        section2
            .deserialize(&bytes)
            .expect("deserialize should succeed");

        assert_eq!(section2.store().node_count(), 2);
        assert_eq!(section2.store().edge_count(), 1);
    }

    #[test]
    fn lpg_section_dirty_tracking() {
        let store = Arc::new(LpgStore::new().unwrap());
        let section = LpgStoreSection::new(store);

        assert!(!section.is_dirty());
        section.mark_dirty();
        assert!(section.is_dirty());
        section.mark_clean();
        assert!(!section.is_dirty());
    }

    #[test]
    fn lpg_section_type() {
        let store = Arc::new(LpgStore::new().unwrap());
        let section = LpgStoreSection::new(store);
        assert_eq!(section.section_type(), SectionType::LpgStore);
        assert_eq!(section.version(), LPG_SECTION_VERSION);
    }

    #[test]
    fn lpg_section_empty_round_trip() {
        let store = Arc::new(LpgStore::new().unwrap());
        let section = LpgStoreSection::new(Arc::clone(&store));
        let bytes = section.serialize().unwrap();

        let store2 = Arc::new(LpgStore::new().unwrap());
        let mut section2 = LpgStoreSection::new(store2);
        section2.deserialize(&bytes).unwrap();
        assert_eq!(section2.store().node_count(), 0);
        assert_eq!(section2.store().edge_count(), 0);
    }

    #[test]
    fn lpg_section_properties_preserved() {
        let store = Arc::new(LpgStore::new().unwrap());
        let n = store.create_node(&["Person"]);
        store.set_node_property(n, "name", Value::String("Alix".into()));
        store.set_node_property(n, "age", Value::Int64(30));
        store.set_node_property(n, "active", Value::Bool(true));

        let section = LpgStoreSection::new(Arc::clone(&store));
        let bytes = section.serialize().unwrap();

        let store2 = Arc::new(LpgStore::new().unwrap());
        let mut section2 = LpgStoreSection::new(Arc::clone(&store2));
        section2.deserialize(&bytes).unwrap();

        let node = store2.get_node(n).unwrap();
        let name_key: PropertyKey = "name".into();
        let age_key: PropertyKey = "age".into();
        let active_key: PropertyKey = "active".into();
        assert_eq!(
            node.properties.get(&name_key),
            Some(&Value::String("Alix".into()))
        );
        assert_eq!(node.properties.get(&age_key), Some(&Value::Int64(30)));
        assert_eq!(node.properties.get(&active_key), Some(&Value::Bool(true)));
    }

    #[test]
    fn lpg_section_named_graphs() {
        let store = Arc::new(LpgStore::new().unwrap());
        store.create_node(&["Root"]);
        store.create_graph("social").unwrap();

        if let Some(g) = store.graph("social") {
            g.create_node(&["Friend"]);
        }

        let section = LpgStoreSection::new(Arc::clone(&store));
        let bytes = section.serialize().unwrap();

        let store2 = Arc::new(LpgStore::new().unwrap());
        let mut section2 = LpgStoreSection::new(Arc::clone(&store2));
        section2.deserialize(&bytes).unwrap();

        assert_eq!(store2.node_count(), 1);
        assert!(store2.graph("social").is_some());
        assert_eq!(store2.graph("social").unwrap().node_count(), 1);
    }

    #[test]
    fn lpg_section_crc_integrity() {
        let store = Arc::new(LpgStore::new().unwrap());
        store.create_node(&["Test"]);

        let section = LpgStoreSection::new(Arc::clone(&store));
        let mut bytes = section.serialize().unwrap();

        // Corrupt a byte
        let last = bytes.len() - 1;
        bytes[last] ^= 0xFF;

        let store2 = Arc::new(LpgStore::new().unwrap());
        let mut section2 = LpgStoreSection::new(store2);
        assert!(section2.deserialize(&bytes).is_err());
    }
}
