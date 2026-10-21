//! The reader of the LPG section of 0.5.x: the block layout (the `block`
//! submodule), one raw chunk of the container, loaded into the row-group
//! store through [`LoadTarget`] (see
//! [`RowGroupStoreSection`](crate::graph::rowgroup::section::RowGroupStoreSection)).
//! No build writes it any more.

use grafeo_common::change::Entity;
use grafeo_common::types::{EpochId, Value};
use grafeo_common::utils::error::Result;

use super::block::{self, BlockEdge, BlockNode};
use crate::graph::load::LoadTarget;

/// Loads the properties of a node or edge of a 0.5.x section, or of a
/// snapshot: each property's versions, oldest first, the last its current
/// value.
///
/// # Errors
///
/// The error of a value the store refuses.
pub fn load_versions<S: LoadTarget>(
    store: &S,
    entity: Entity,
    properties: &[(String, Vec<(EpochId, Value)>)],
) -> Result<()> {
    for (key, entries) in properties {
        if let Some(((epoch, value), older)) = entries.split_last() {
            for (epoch, value) in older {
                store.load_old_value(entity, key, value.clone(), *epoch);
            }
            store.load_value(entity, key, value.clone(), Some(*epoch))?;
        }
    }
    Ok(())
}

fn populate_store<S: LoadTarget>(
    store: &S,
    nodes: &[BlockNode],
    edges: &[BlockEdge],
) -> Result<()> {
    for node in nodes {
        let label_refs: Vec<&str> = node.labels.iter().map(|s| s.as_str()).collect();
        store.load_node(node.id, &label_refs)?;
        load_versions(store, Entity::Node(node.id), &node.properties)?;
    }
    for edge in edges {
        // A store that leaves out an edge whose end it lacks (see
        // `LoadTarget::load_edge`) changes nothing for its values either.
        store.load_edge(edge.id, edge.src, edge.dst, &edge.edge_type)?;
        load_versions(store, Entity::Edge(edge.id), &edge.properties)?;
    }
    Ok(())
}

/// Loads the LPG section of 0.5.x, the block layout `data`, into `store`
/// and its named graphs (each created, as a 0.5.x file names each once).
///
/// # Errors
///
/// The errors of the block reader, and of a node, edge or graph the store
/// refuses.
pub(crate) fn load_blocks<S: LoadTarget>(store: &S, data: &[u8]) -> Result<()> {
    block::read_blocks(data, &mut |nodes, edges, named_graphs, epoch| {
        populate_store(store, &nodes, &edges)?;
        store.load_section_epoch(EpochId::new(epoch));
        store.load_finish();

        for graph in &named_graphs {
            let graph_store = store.load_named_graph(&graph.name).map_err(|e| {
                grafeo_common::utils::error::Error::corruption(format!(
                    "LPG section (0.5.x): graph {:?}: {e}",
                    graph.name
                ))
            })?;
            populate_store(&*graph_store, &graph.nodes, &graph.edges)?;
            graph_store.load_section_epoch(EpochId::new(epoch));
            graph_store.load_finish();
        }

        Ok(())
    })
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use grafeo_common::storage::{ImageSource, MemoryImage, Section, SectionType};
    use grafeo_common::types::Value;

    use super::*;
    use crate::graph::load::testing::assert_same_rows;
    use crate::graph::rowgroup::RowGroupStore;
    use crate::graph::rowgroup::section::RowGroupStoreSection;
    use crate::graph::{GraphStore, GraphStoreMut};

    /// The LPG section of [`sample_store`] in the 0.5.x block layout, as
    /// `LpgStore` wrote it (`tests/fixtures/lpg-v3/README.md`).
    const SAMPLE: &[u8] = include_bytes!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/lpg-v3/sample_0_5.bin"
    ));

    /// A store with nodes of 0, 1 and 3 labels, edges of two types, every
    /// value kind the 0.5.x block layout keeps exactly, and a named graph.
    fn sample_store() -> RowGroupStore {
        use grafeo_common::types::{Date, Duration, Time};

        let store = RowGroupStore::new();
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

    /// Reads the 0.5.x section `bytes` (one raw chunk) into a new root.
    fn migrate(bytes: &[u8]) -> Result<Arc<RowGroupStore>> {
        let image = MemoryImage::from_raw(vec![(SectionType::LpgStore, bytes.to_vec())])?;
        let root = Arc::new(RowGroupStore::new());
        RowGroupStoreSection::new(Arc::clone(&root))
            .read_from(&*image.section_source(SectionType::LpgStore).unwrap())?;
        Ok(root)
    }

    #[test]
    fn a_0_5_lpg_section_loads_into_a_root() {
        assert!(block::is_block_format(SAMPLE));
        let root = migrate(SAMPLE).unwrap();
        let expected = sample_store();
        assert_same_rows(&expected, &*root, "the default graph");
        assert_same_rows(
            &*expected.graph("trips").unwrap(),
            &*root.graph("trips").unwrap(),
            "graph trips",
        );
        assert_eq!(root.graph_names(), vec!["trips".to_string()]);
    }

    #[test]
    fn a_damaged_0_5_section_is_refused() {
        let mut bytes = SAMPLE.to_vec();
        let last = bytes.len() - 1;
        bytes[last] ^= 0xFF;
        assert!(migrate(&bytes).is_err(), "the checksum catches the damage");
    }

    #[test]
    fn an_empty_0_5_section_loads_nothing() {
        let bytes = block::write_blocks(&[], &[], &[], 0).unwrap();
        let root = migrate(&bytes).unwrap();
        assert_eq!((root.node_count(), root.edge_count()), (0, 0));
        assert_eq!(root.graph_count(), 0);
    }
}
