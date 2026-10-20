//! The row-group store's indexes: what they hold after each write, undo and
//! collect, and what the lookups and searches at a view find.

use grafeo_common::types::{EpochId, NodeId, ReadView, Value};

use crate::graph::conformance::{Tx, people};
use crate::graph::rowgroup::RowGroupStore;
use crate::graph::traits::GraphStore;

/// The nodes `view` finds with `city`, through the index.
fn in_city(store: &RowGroupStore, city: &str, view: ReadView) -> Vec<NodeId> {
    store.find_nodes_by_property_at("city", &Value::from(city), view)
}

/// The nodes the index holds for `city` in any version.
fn held(store: &RowGroupStore, city: &str) -> Vec<NodeId> {
    store
        .find_nodes_maybe_equal("city", &Value::from(city))
        .expect("an index on city")
}

/// A committed write keeps the replaced value's entry for the readers before
/// it until `collect` at its epoch; an undone write and a transaction's
/// overwritten value leave no entry.
#[test]
fn an_index_keeps_old_values_until_collect_and_none_that_were_undone() {
    let store = RowGroupStore::new();
    let people = people(&store).unwrap();
    store.create_property_index("city");
    let before = store.current_epoch();
    assert_eq!(store.property_index_len("city"), Some(1), "Amsterdam");

    let mut tx = Tx::begin(&store, 10);
    tx.set_node(people.alix, "city", Value::from("Berlin"))
        .unwrap();
    let committed = tx.commit().unwrap();
    assert_eq!(store.property_index_len("city"), Some(2));
    assert_eq!(held(&store, "Amsterdam"), [people.alix]);
    assert_eq!(
        in_city(&store, "Amsterdam", ReadView::committed(before)),
        [people.alix]
    );
    assert_eq!(
        in_city(&store, "Amsterdam", ReadView::committed(committed)),
        []
    );
    assert_eq!(
        in_city(&store, "Berlin", ReadView::committed(committed)),
        [people.alix]
    );

    // Two writes of one transaction: the first value is no version's once
    // the second replaces it, and the rollback takes the second out.
    let mut tx = Tx::begin(&store, 11);
    tx.set_node(people.gus, "city", Value::from("Paris"))
        .unwrap();
    tx.set_node(people.gus, "city", Value::from("Prague"))
        .unwrap();
    assert_eq!(held(&store, "Paris"), [], "replaced in place");
    assert_eq!(held(&store, "Prague"), [people.gus]);
    let own = ReadView::transaction(tx.id, tx.snapshot);
    assert_eq!(in_city(&store, "Prague", own), [people.gus]);
    assert_eq!(
        in_city(&store, "Prague", ReadView::committed(committed)),
        []
    );
    tx.roll_back().unwrap();
    assert_eq!(held(&store, "Prague"), [], "undone");
    assert_eq!(store.property_index_len("city"), Some(2));

    store.collect(committed);
    assert_eq!(store.property_index_len("city"), Some(1), "Berlin alone");
    assert_eq!(held(&store, "Amsterdam"), []);
    assert_eq!(held(&store, "Berlin"), [people.alix]);
}

/// A delete keeps a node's entries for the readers before it; `collect` at
/// the delete's epoch takes them out with the node.
#[test]
fn a_deleted_nodes_entries_stay_until_collect() {
    let store = RowGroupStore::new();
    let people = people(&store).unwrap();
    store.create_property_index("name");
    let before = store.current_epoch();
    let mut tx = Tx::begin(&store, 10);
    tx.delete_edge(people.gus_mia).unwrap();
    tx.delete_node(people.mia).unwrap();
    let deleted = tx.commit().unwrap();
    let mia = Value::from("Mia");
    assert_eq!(
        store.find_nodes_by_property_at("name", &mia, ReadView::committed(before)),
        [people.mia]
    );
    assert_eq!(
        store.find_nodes_by_property_at("name", &mia, ReadView::committed(deleted)),
        []
    );
    store.collect(deleted);
    assert_eq!(
        store.find_nodes_maybe_equal("name", &mia),
        Some(Vec::new()),
        "collected with the node"
    );
    assert_eq!(store.property_index_len("name"), Some(2), "Alix and Gus");
}

/// An index made while a transaction has written a value holds both the
/// committed value and the pending one: each reader finds its own.
#[test]
fn an_index_made_during_a_transaction_holds_both_versions() {
    let store = RowGroupStore::new();
    let people = people(&store).unwrap();
    let mut tx = Tx::begin(&store, 10);
    tx.set_node(people.alix, "city", Value::from("Paris"))
        .unwrap();
    store.create_property_index("city");
    let now = ReadView::committed(store.current_epoch());
    let own = ReadView::transaction(tx.id, tx.snapshot);
    assert_eq!(in_city(&store, "Amsterdam", now), [people.alix]);
    assert_eq!(in_city(&store, "Paris", now), []);
    assert_eq!(in_city(&store, "Paris", own), [people.alix]);
    assert_eq!(in_city(&store, "Amsterdam", own), []);
    assert_eq!(
        store.find_nodes_by_property("city", &Value::from("Paris")),
        Vec::<NodeId>::new(),
        "the lookup without a view reads at the store's epoch"
    );
    let committed = tx.commit().unwrap();
    store.collect(committed);
    assert_eq!(store.property_index_len("city"), Some(1));
    assert_eq!(
        in_city(
            &store,
            "Paris",
            ReadView::committed(EpochId::new(committed.as_u64()))
        ),
        [people.alix]
    );
}

#[cfg(feature = "text-index")]
mod text {
    use std::sync::Arc;

    use grafeo_common::types::{PropertyKey, ReadView, Value};
    use parking_lot::RwLock;

    use crate::graph::conformance::{Tx, people};
    use crate::graph::rowgroup::RowGroupStore;
    use crate::graph::traits::{GraphStore, GraphStoreSearch};
    use crate::index::text::{BM25Config, InvertedIndex};

    /// The index holds each document's newest text, a deleted one until
    /// `collect`; once no chain is left it scores as an index built from
    /// the store's committed texts does.
    #[test]
    fn the_index_holds_the_newest_texts_and_drops_a_deleted_one_at_collect() {
        let store = RowGroupStore::new();
        let index = Arc::new(RwLock::new(InvertedIndex::new(BM25Config::default())));
        store.add_text_index("Person", "bio", Arc::clone(&index));
        let people = people(&store).unwrap();
        let mut tx = Tx::begin(&store, 10);
        tx.set_node(
            people.alix,
            "bio",
            Value::from("graph storage in Amsterdam"),
        )
        .unwrap();
        tx.set_node(people.gus, "bio", Value::from("graph theory"))
            .unwrap();
        tx.set_node(people.mia, "bio", Value::from("columns"))
            .unwrap();
        let written = tx.commit().unwrap();
        let mut tx = Tx::begin(&store, 11);
        tx.set_node(people.alix, "bio", Value::from("storage engines"))
            .unwrap();
        tx.delete_edge(people.gus_mia).unwrap();
        tx.delete_node(people.mia).unwrap();
        assert_eq!(index.read().len(), 3, "the newest texts, Mia's too");
        let committed = tx.commit().unwrap();
        let found = |query: &str, view: ReadView| -> Vec<u64> {
            let mut ids: Vec<u64> = store
                .text_search_at("Person", "bio", query, 10, view)
                .into_iter()
                .map(|(id, _)| id.as_u64())
                .collect();
            ids.sort_unstable();
            ids
        };
        let earlier = ReadView::committed(written);
        assert_eq!(
            found("graph", earlier),
            [people.alix.as_u64(), people.gus.as_u64()]
        );
        assert_eq!(found("columns", earlier), [people.mia.as_u64()]);
        assert_eq!(
            found("graph", ReadView::committed(committed)),
            [people.gus.as_u64()]
        );
        assert_eq!(
            found("columns", ReadView::committed(committed)),
            [] as [u64; 0]
        );

        store.collect(committed);
        assert_eq!(index.read().len(), 2, "Mia's is collected");
        let mut rebuilt = InvertedIndex::new(BM25Config::default());
        for id in store.node_ids() {
            if let Some(Value::String(text)) = store.get_node_property(id, &PropertyKey::new("bio"))
            {
                rebuilt.insert(id, &text);
            }
        }
        assert_eq!(
            store.text_search("Person", "bio", "storage graph", 10),
            rebuilt.search("storage graph", 10)
        );
    }
}

#[cfg(feature = "vector-index")]
mod vector {
    use std::sync::Arc;

    use grafeo_common::types::{ReadView, Value};

    use crate::graph::conformance::{Tx, people};
    use crate::graph::rowgroup::RowGroupStore;
    use crate::graph::traits::GraphStoreSearch;
    use crate::index::vector::{DistanceMetric, HnswConfig, HnswIndex, VectorIndexKind};

    fn vector(values: &[f32]) -> Value {
        Value::Vector(values.to_vec().into())
    }

    /// A deleted node's vector stays in the index for the readers before
    /// the delete, and `collect` takes it out; a vector of another size
    /// never goes in.
    #[test]
    fn a_deleted_nodes_vector_stays_until_collect() {
        let store = RowGroupStore::new();
        let index = Arc::new(VectorIndexKind::Hnsw(HnswIndex::new(HnswConfig::new(
            2,
            DistanceMetric::Euclidean,
        ))));
        store.add_vector_index("Person", "embedding", Arc::clone(&index));
        let people = people(&store).unwrap();
        let mut tx = Tx::begin(&store, 10);
        tx.set_node(people.alix, "embedding", vector(&[1.0, 0.0]))
            .unwrap();
        tx.set_node(people.gus, "embedding", vector(&[0.0, 1.0]))
            .unwrap();
        tx.set_node(people.mia, "embedding", vector(&[1.0, 0.0, 0.0]))
            .unwrap();
        let written = tx.commit().unwrap();
        assert!(!index.contains(people.mia), "three values: not measurable");
        let mut tx = Tx::begin(&store, 11);
        tx.delete_edge(people.alix_gus).unwrap();
        tx.delete_node(people.alix).unwrap();
        let deleted = tx.commit().unwrap();
        let nearest = |view: ReadView| -> Vec<u64> {
            store
                .vector_search_at(
                    Some("Person"),
                    "embedding",
                    &[1.0, 0.0],
                    1,
                    DistanceMetric::Euclidean,
                    view,
                )
                .into_iter()
                .map(|(id, _)| id.as_u64())
                .collect()
        };
        assert_eq!(
            nearest(ReadView::committed(written)),
            [people.alix.as_u64()]
        );
        assert_eq!(nearest(ReadView::committed(deleted)), [people.gus.as_u64()]);
        assert!(index.contains(people.alix));
        store.collect(deleted);
        assert!(!index.contains(people.alix), "collected");
        assert_eq!(index.len(), 1);
    }
}
