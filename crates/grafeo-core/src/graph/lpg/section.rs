//! LPG section serializer for the `.grafeo` container format.
//!
//! Implements the [`Section`] trait for LPG graph data (nodes, edges,
//! properties, named graphs). A checkpoint streams the section as chunks
//! (version 3, see the `chunked` submodule), at most one open chunk per
//! column of the row group being written, and a load reads them one chunk at
//! a time. [`Section::serialize`] and [`Section::deserialize`] keep the
//! block-based format of 0.5.x (version 2, the `block` submodule), which
//! 0.5.x files hold (as one raw chunk) and the spill path uses.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use grafeo_common::storage::ChunkCaps;
use grafeo_common::storage::section::{
    Section, SectionSink, SectionSource, SectionType, check_version, legacy_bytes,
};
use grafeo_common::types::{EpochId, Value};
use grafeo_common::utils::error::Result;

use super::block::{self, BlockEdge, BlockNamedGraph, BlockNode};
use super::chunked::{LPG_SECTION_VERSION, read_lpg_chunks, write_lpg_chunks};
use super::store::OpenChangeSource;
use crate::graph::lpg::{LpgStore, OpenChangesByGraph};

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
/// Wraps an `Arc<LpgStore>` and implements the [`Section`] trait: it streams
/// the store as chunks (version 3) of at most the [`ChunkCaps`] it was built
/// with, and serializes and deserializes the block-based format of 0.5.x
/// (version 2).
pub struct LpgStoreSection {
    store: Arc<LpgStore>,
    dirty: AtomicBool,
    caps: ChunkCaps,
    /// What the transactions open while the section is written changed,
    /// from their change sets; `None` while no transaction is open.
    open_changes: Option<OpenChangesByGraph>,
}

impl LpgStoreSection {
    /// Create a new LPG section wrapping the given store, writing chunks of
    /// at most [`ChunkCaps::current`] (the caps a test set on this thread, or
    /// the default ones).
    pub fn new(store: Arc<LpgStore>) -> Self {
        Self::with_caps(store, ChunkCaps::current())
    }

    /// Create a new LPG section wrapping the given store, writing chunks of
    /// at most `caps`.
    pub fn with_caps(store: Arc<LpgStore>, caps: ChunkCaps) -> Self {
        Self {
            store,
            dirty: AtomicBool::new(false),
            caps,
            open_changes: None,
        }
    }

    /// Writes the committed state of what the open transactions changed
    /// from `changes`, indexed from their change sets. The caller holds
    /// their writes, rollbacks and commits until the section is written (see
    /// [`OpenChangesByGraph`]).
    #[must_use]
    pub fn with_open_changes(mut self, changes: OpenChangesByGraph) -> Self {
        self.open_changes = Some(changes);
        self
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

    /// The block layout of 0.5.x, of the store as it is now: the nodes and
    /// edges visible now with their current values (a checkpoint writes the
    /// committed state with [`write_to`](Section::write_to)).
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

    /// Streams the store's committed state as chunks of version 3: per graph
    /// the node and edge tables in row groups, then the metadata chunk. What
    /// open transactions changed is written as it was committed, from the
    /// change sets the section was given, so the caller holds their writes
    /// and rollbacks and commits for the whole write (a checkpoint's write
    /// freeze). Without them, the store is written as it is.
    fn write_to(&self, sink: &mut dyn SectionSink) -> Result<()> {
        let open = match &self.open_changes {
            Some(changes) => OpenChangeSource::ChangeSets(changes),
            None => OpenChangeSource::None,
        };
        write_lpg_chunks(&self.store, self.caps, open, sink)
    }

    /// Loads 0.5.x bytes (one raw chunk) through
    /// [`deserialize`](Section::deserialize), or the chunks of version 3 one
    /// at a time.
    ///
    /// # Errors
    ///
    /// Returns a serialization error for a section of another version, or
    /// chunks the reader refuses.
    fn read_from(&mut self, source: &dyn SectionSource) -> Result<()> {
        if let Some(bytes) = legacy_bytes(source)? {
            return self.deserialize(&bytes);
        }
        check_version(SectionType::LpgStore, source, LPG_SECTION_VERSION)?;
        read_lpg_chunks(&self.store, source)
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

    /// The section writes chunks of version 3 with the caps it was built
    /// with: those `with_caps` names, or the thread's caps when `new` ran.
    #[test]
    fn a_section_writes_chunks_with_the_caps_it_was_built_with() {
        use grafeo_common::storage::{ChunkKind, ChunkNamespace, ImageSource, MemoryImage};
        use grafeo_common::testing::chunk_caps::with_chunk_caps;

        let store = Arc::new(LpgStore::new().unwrap());
        for name in ["Alix", "Gus", "Vincent", "Mia", "Jules"] {
            let id = store.create_node(&["Person"]);
            store.set_node_property(id, "name", Value::from(name));
        }
        let tiny = ChunkCaps {
            max_rows: 2,
            max_bytes: 1 << 20,
        };
        let label_chunks = |section: &LpgStoreSection| {
            let image = MemoryImage::from_sections(&[section as &dyn Section]).unwrap();
            let source = image.section_source(SectionType::LpgStore).unwrap();
            assert_eq!(source.section_version(), 3);
            assert_eq!(source.chunks().last().unwrap().kind, ChunkKind::Meta);
            source
                .chunks()
                .iter()
                .filter(|chunk| {
                    chunk.kind == ChunkKind::Column
                        && chunk.namespace == ChunkNamespace::NodeStructure
                        && chunk.column_id == 0
                })
                .count()
        };
        let built_with = LpgStoreSection::with_caps(Arc::clone(&store), tiny);
        assert_eq!(label_chunks(&built_with), 3, "5 nodes in groups of 2");
        let built_under = with_chunk_caps(tiny, || LpgStoreSection::new(Arc::clone(&store)));
        assert_eq!(
            label_chunks(&built_under),
            3,
            "the caps of the thread that built it"
        );
        let default = LpgStoreSection::new(Arc::clone(&store));
        assert_eq!(label_chunks(&default), 1, "the default caps");
    }

    /// A store with nodes of 0, 1 and 3 labels, edges of two types, every
    /// value kind the 0.5.x block layout keeps exactly, and a named graph.
    fn sample_store() -> Arc<LpgStore> {
        use grafeo_common::types::{Date, Duration, Time};

        let store = Arc::new(LpgStore::new().unwrap());
        let alix = store.create_node(&[]);
        let gus = store.create_node(&["Person"]);
        let mia = store.create_node(&["Person", "Employee", "Traveller"]);
        let values = [
            ("name", Value::from("Alix")),
            ("age", Value::Int64(-19)),
            ("score", Value::Float64(-0.0)),
            ("active", Value::Bool(true)),
            ("photo", Value::Bytes(Arc::from(vec![3u8, 19, 88]))),
            ("born", Value::Date(Date::from_days(-3))),
            (
                "wake",
                Value::Time(Time::from_nanos(88).unwrap().with_offset(3600)),
            ),
            ("stay", Value::Duration(Duration::new(3, 19, 88))),
            (
                "stops",
                Value::List(Arc::from(vec![Value::from("Paris"), Value::Int64(3)])),
            ),
            ("embedding", Value::Vector(Arc::from(vec![3.0f32, 19.0]))),
        ];
        for (index, (key, value)) in values.into_iter().enumerate() {
            let node = [alix, gus, mia][index % 3];
            store.set_node_property(node, key, value);
        }
        let knows = store.create_edge(alix, gus, "KNOWS");
        store.set_edge_property(knows, "since", Value::Int64(2019));
        store.create_edge(gus, mia, "VISITED");
        store.create_graph("trips").unwrap();
        let trips = store.graph("trips").unwrap();
        let berlin = trips.create_node(&["City"]);
        trips.set_node_property(berlin, "name", Value::from("Berlin"));
        store
    }

    /// The labels of `node`, sorted.
    fn labels_of(node: &crate::graph::lpg::Node) -> Vec<String> {
        let mut labels: Vec<String> = node.labels.iter().map(ToString::to_string).collect();
        labels.sort();
        labels
    }

    /// Asserts `a` and `b` hold the same nodes, labels, edges and values.
    fn assert_same(a: &LpgStore, b: &LpgStore) {
        assert_eq!(a.node_ids(), b.node_ids());
        for id in a.node_ids() {
            let (x, y) = (a.get_node(id).unwrap(), b.get_node(id).unwrap());
            assert_eq!(labels_of(&x), labels_of(&y), "node {}", id.0);
            assert_eq!(
                x.properties.to_btree_map(),
                y.properties.to_btree_map(),
                "node {}",
                id.0
            );
        }
        assert_eq!(a.edge_count(), b.edge_count());
        for edge in a.all_edges() {
            let other = b.get_edge(edge.id).unwrap();
            assert_eq!(
                (edge.src, edge.dst, edge.edge_type.clone()),
                (other.src, other.dst, other.edge_type.clone())
            );
            assert_eq!(
                edge.properties.to_btree_map(),
                other.properties.to_btree_map()
            );
        }
    }

    #[test]
    fn a_0_5_lpg_section_still_loads() {
        use grafeo_common::storage::{ImageSource, MemoryImage};

        let store = sample_store();
        // The 0.5.x block layout, as one raw chunk of version 0.
        let bytes = LpgStoreSection::new(Arc::clone(&store))
            .serialize()
            .unwrap();
        let image = MemoryImage::from_raw(vec![(SectionType::LpgStore, bytes)]).unwrap();
        let back = Arc::new(LpgStore::new().unwrap());
        LpgStoreSection::new(Arc::clone(&back))
            .read_from(&*image.section_source(SectionType::LpgStore).unwrap())
            .unwrap();
        assert_same(&store, &back);
        assert_same(
            &store.graph("trips").unwrap(),
            &back.graph("trips").unwrap(),
        );
    }

    #[test]
    fn a_section_reads_back_what_it_writes() {
        use grafeo_common::storage::{ImageSource, MemoryImage};

        let store = sample_store();
        let section = LpgStoreSection::with_caps(
            Arc::clone(&store),
            ChunkCaps {
                max_rows: 2,
                max_bytes: 64,
            },
        );
        let image = MemoryImage::from_sections(&[&section as &dyn Section]).unwrap();
        let back = Arc::new(LpgStore::new().unwrap());
        LpgStoreSection::new(Arc::clone(&back))
            .read_from(&*image.section_source(SectionType::LpgStore).unwrap())
            .unwrap();
        assert_same(&store, &back);
        assert_same(
            &store.graph("trips").unwrap(),
            &back.graph("trips").unwrap(),
        );
    }

    #[test]
    fn chunks_of_another_version_are_refused_naming_it() {
        use grafeo_common::storage::{ImageSource, MemoryImage};

        let store = sample_store();
        let mut image = MemoryImage::new();
        image.begin_section(SectionType::LpgStore, 2).unwrap();
        LpgStoreSection::new(Arc::clone(&store))
            .write_to(&mut image)
            .unwrap();
        let error = LpgStoreSection::new(Arc::new(LpgStore::new().unwrap()))
            .read_from(&*image.section_source(SectionType::LpgStore).unwrap())
            .unwrap_err()
            .to_string();
        assert!(
            error.contains("LpgStore") && error.contains("version 2"),
            "{error}"
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
