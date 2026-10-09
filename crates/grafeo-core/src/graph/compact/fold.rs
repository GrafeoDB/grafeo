//! Folds the compacted base of a database file into its LPG store.
//!
//! The file of a database compacted with `compact()` (0.5.x, and 0.6
//! development builds) holds a columnar base (the `CompactStore` section), an
//! overlay with the changes made since (the `LpgStore` section) and the base
//! nodes and edges the overlay deleted (the `OverlayDeletions` section).
//! Opening such a file folds the base into the store the overlay loaded into,
//! so the database goes on with one store, and its next checkpoint writes it
//! as a plain one. Removed in 0.7.0 with the other 0.5.x readers.

use grafeo_common::types::{EdgeId, NodeId};
use grafeo_common::utils::error::{Error, Result, StorageError};
use grafeo_common::utils::hash::FxHashSet;

use super::CompactStore;
use crate::graph::Direction;
use crate::graph::lpg::LpgStore;
use crate::graph::traits::GraphStore;

/// What [`fold_into`] created.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct Folded {
    /// Base nodes created in the store.
    pub nodes: usize,
    /// Base edges created in the store.
    pub edges: usize,
    /// Base edges left out because an endpoint is gone: a node the deletion
    /// log lists while the edge is not listed. A delete of a node deleted its
    /// edges first, so a file holds none of them unless it is damaged.
    pub dangling_edges: usize,
}

/// Creates in `store` every node and edge of `base` that `store` does not
/// hold and the deletion log (`deleted_nodes`, `deleted_edges`) does not
/// list, under its id, with its labels or its type and endpoints, and its
/// properties.
///
/// `store` holds the overlay as loaded: the nodes and edges created after
/// `compact()`, and a whole copy of each base node and edge changed since,
/// under the base's id, which is the current version of it. Every other base
/// node and edge is current in the base, unless the log lists it.
///
/// No transaction may be open in `store`: the nodes and edges are created as
/// recovery creates them, committed at the store's current epoch. The
/// indexes of `store` are built after the fold, from all the data.
///
/// # Errors
///
/// Returns an error if `base` does not keep the ids the overlay refers to
/// (every base a compacted database wrote keeps them), or if the store
/// cannot allocate a record.
pub fn fold_into(
    base: &CompactStore,
    store: &LpgStore,
    deleted_nodes: &[NodeId],
    deleted_edges: &[EdgeId],
) -> Result<Folded> {
    if !base.preserves_ids() {
        return Err(Error::Storage(StorageError::Corruption(
            "the compacted base does not keep the ids of its nodes and edges".to_string(),
        )));
    }
    let deleted_nodes: FxHashSet<NodeId> = deleted_nodes.iter().copied().collect();
    let deleted_edges: FxHashSet<EdgeId> = deleted_edges.iter().copied().collect();
    let mut folded = Folded::default();

    // The nodes the store holds: the overlay's, then the folded ones.
    let mut live: FxHashSet<NodeId> = store.node_ids().into_iter().collect();
    let base_nodes = base.node_ids();
    for &id in &base_nodes {
        if deleted_nodes.contains(&id) || live.contains(&id) {
            continue;
        }
        let Some(node) = base.get_node(id) else {
            continue;
        };
        let labels: Vec<&str> = node.labels.iter().map(|label| label.as_str()).collect();
        store.create_node_with_id(id, &labels)?;
        for (key, value) in node.properties {
            store.set_node_property(id, key.as_str(), value);
        }
        live.insert(id);
        folded.nodes += 1;
    }

    // Every base edge once: from its source, as the outgoing edges of the
    // base's nodes (also of those the overlay copied or deleted).
    for &source in &base_nodes {
        for (_, id) in base.edges_from(source, Direction::Outgoing) {
            if deleted_edges.contains(&id) || store.edge_type(id).is_some() {
                continue;
            }
            let Some(edge) = base.get_edge(id) else {
                continue;
            };
            if !live.contains(&edge.src) || !live.contains(&edge.dst) {
                folded.dangling_edges += 1;
                continue;
            }
            store.create_edge_with_id(id, edge.src, edge.dst, edge.edge_type.as_str())?;
            for (key, value) in edge.properties {
                store.set_edge_property(id, key.as_str(), value);
            }
            folded.edges += 1;
        }
    }
    Ok(folded)
}

#[cfg(test)]
mod tests {
    use grafeo_common::types::{PropertyKey, Value};

    use super::*;
    use crate::graph::compact::from_graph_store_preserving_ids;

    /// A graph and its compacted base: Alix knows Gus and Vincent, Gus knows
    /// Vincent, Alix lives in Amsterdam.
    struct Compacted {
        base: CompactStore,
        alix: NodeId,
        gus: NodeId,
        vincent: NodeId,
        amsterdam: NodeId,
        alix_knows_gus: EdgeId,
        alix_knows_vincent: EdgeId,
        gus_knows_vincent: EdgeId,
        lives_in: EdgeId,
    }

    fn compacted() -> Compacted {
        let source = LpgStore::new().unwrap();
        let person = |name: &str, age: i64| {
            let id = source.create_node(&["Person"]);
            source.set_node_property(id, "name", Value::from(name));
            source.set_node_property(id, "age", Value::Int64(age));
            id
        };
        let alix = person("Alix", 30);
        let gus = person("Gus", 25);
        let vincent = person("Vincent", 40);
        let amsterdam = source.create_node(&["City", "Capital"]);
        source.set_node_property(amsterdam, "name", Value::from("Amsterdam"));
        let alix_knows_gus = source.create_edge(alix, gus, "KNOWS");
        source.set_edge_property(alix_knows_gus, "since", Value::Int64(2019));
        let alix_knows_vincent = source.create_edge(alix, vincent, "KNOWS");
        let gus_knows_vincent = source.create_edge(gus, vincent, "KNOWS");
        let lives_in = source.create_edge(alix, amsterdam, "LIVES_IN");
        Compacted {
            base: from_graph_store_preserving_ids(&source).unwrap(),
            alix,
            gus,
            vincent,
            amsterdam,
            alix_knows_gus,
            alix_knows_vincent,
            gus_knows_vincent,
            lives_in,
        }
    }

    /// An overlay as `compact()` leaves it: empty, its ids past the base's.
    fn overlay_over(compacted: &Compacted) -> LpgStore {
        let overlay = LpgStore::new().unwrap();
        overlay.set_next_node_id(compacted.amsterdam.as_u64() + 1);
        overlay.set_next_edge_id(compacted.lives_in.as_u64() + 1);
        overlay
    }

    fn name(store: &LpgStore, id: NodeId) -> Option<Value> {
        store.get_node_property(id, &PropertyKey::from("name"))
    }

    #[test]
    fn an_untouched_base_folds_whole_under_its_ids() {
        let compacted = compacted();
        let store = overlay_over(&compacted);

        let folded = fold_into(&compacted.base, &store, &[], &[]).unwrap();

        assert_eq!(
            folded,
            Folded {
                nodes: 4,
                edges: 4,
                dangling_edges: 0
            }
        );
        assert_eq!(name(&store, compacted.alix), Some(Value::from("Alix")));
        let mut labels: Vec<String> = store
            .get_node(compacted.amsterdam)
            .unwrap()
            .labels
            .iter()
            .map(ToString::to_string)
            .collect();
        labels.sort();
        assert_eq!(
            labels,
            ["Capital", "City"],
            "both labels of a multi-label node"
        );
        let edge = store.get_edge(compacted.alix_knows_gus).unwrap();
        assert_eq!((edge.src, edge.dst), (compacted.alix, compacted.gus));
        assert_eq!(edge.edge_type.as_str(), "KNOWS");
        assert_eq!(
            store.get_edge_property(compacted.alix_knows_gus, &PropertyKey::from("since")),
            Some(Value::Int64(2019))
        );
        assert_eq!(
            store.edge_type(compacted.lives_in).as_deref(),
            Some("LIVES_IN")
        );
        assert_eq!(
            store.out_degree(compacted.alix),
            3,
            "adjacency holds the folded edges"
        );
        assert_eq!(store.nodes_by_label("Person").len(), 3);

        // A node created after the fold gets an id past the base's.
        let created = store.create_node(&["Person"]);
        assert!(created.as_u64() > compacted.amsterdam.as_u64());
    }

    #[test]
    fn the_overlay_copy_of_a_changed_base_node_wins() {
        let compacted = compacted();
        let store = overlay_over(&compacted);
        // Gus, copied to the overlay when a label and a value changed.
        store
            .create_node_with_id(compacted.gus, &["Person", "Manager"])
            .unwrap();
        store.set_node_property(compacted.gus, "name", Value::from("Gus"));
        store.set_node_property(compacted.gus, "age", Value::Int64(26));
        // The overlay's own node, created after `compact()`.
        let mia = store.create_node(&["Person"]);
        store.set_node_property(mia, "name", Value::from("Mia"));

        let folded = fold_into(&compacted.base, &store, &[], &[]).unwrap();

        assert_eq!(folded.nodes, 3, "every base node but the copied one");
        let gus = store.get_node(compacted.gus).unwrap();
        assert!(gus.labels.iter().any(|label| label.as_str() == "Manager"));
        assert_eq!(
            store.get_node_property(compacted.gus, &PropertyKey::from("age")),
            Some(Value::Int64(26)),
            "the copy's value, not the base's"
        );
        assert_eq!(name(&store, mia), Some(Value::from("Mia")));
        assert_eq!(store.node_count(), 5);
        // The base edges of the copied node are folded too.
        assert!(store.edge_type(compacted.alix_knows_gus).is_some());
        assert!(store.edge_type(compacted.gus_knows_vincent).is_some());
    }

    #[test]
    fn the_overlay_copy_of_a_changed_base_edge_wins() {
        let compacted = compacted();
        let store = overlay_over(&compacted);
        store
            .create_edge_with_id(
                compacted.alix_knows_gus,
                compacted.alix,
                compacted.gus,
                "KNOWS",
            )
            .unwrap();
        store.set_edge_property(compacted.alix_knows_gus, "since", Value::Int64(2020));

        let folded = fold_into(&compacted.base, &store, &[], &[]).unwrap();

        assert_eq!(folded.edges, 3, "every base edge but the copied one");
        assert_eq!(
            store.get_edge_property(compacted.alix_knows_gus, &PropertyKey::from("since")),
            Some(Value::Int64(2020))
        );
        assert_eq!(store.edge_count(), 4);
    }

    #[test]
    fn deleted_base_nodes_and_edges_stay_deleted() {
        let compacted = compacted();
        let store = overlay_over(&compacted);
        // Vincent detach-deleted: the node and both edges to him.
        let deleted_nodes = [compacted.vincent];
        let deleted_edges = [compacted.alix_knows_vincent, compacted.gus_knows_vincent];

        let folded = fold_into(&compacted.base, &store, &deleted_nodes, &deleted_edges).unwrap();

        assert_eq!(
            folded,
            Folded {
                nodes: 3,
                edges: 2,
                dangling_edges: 0
            }
        );
        assert!(store.get_node(compacted.vincent).is_none());
        assert!(store.edge_type(compacted.alix_knows_vincent).is_none());
        assert!(store.edge_type(compacted.gus_knows_vincent).is_none());
        assert_eq!(store.out_degree(compacted.alix), 2);
    }

    #[test]
    fn an_edge_to_a_deleted_node_the_log_misses_is_left_out() {
        let compacted = compacted();
        let store = overlay_over(&compacted);

        let folded = fold_into(
            &compacted.base,
            &store,
            &[compacted.vincent],
            &[compacted.alix_knows_vincent],
        )
        .unwrap();

        assert_eq!(folded.dangling_edges, 1, "Gus knows Vincent: no endpoint");
        assert!(store.edge_type(compacted.gus_knows_vincent).is_none());
        assert_eq!(folded.edges, 2);
    }

    #[test]
    fn a_base_without_its_ids_is_refused() {
        let source = LpgStore::new().unwrap();
        source.create_node(&["Person"]);
        let base = crate::graph::compact::from_graph_store(&source).unwrap();
        let store = LpgStore::new().unwrap();

        let err = fold_into(&base, &store, &[], &[]).unwrap_err();

        assert!(
            matches!(err, Error::Storage(StorageError::Corruption(_))),
            "{err}"
        );
        assert_eq!(store.node_count(), 0, "nothing folded");
    }
}
