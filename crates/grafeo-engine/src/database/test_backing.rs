//! Spill backings for tests (#594): one whose reads fail, one that counts the
//! values copied out of it.

#[cfg(feature = "vector-index")]
use std::collections::HashMap;
use std::sync::Arc;
#[cfg(feature = "vector-index")]
use std::sync::atomic::{AtomicUsize, Ordering};

use grafeo_common::types::{NodeId, PropertyKey, Value};
use grafeo_core::graph::lpg::{ColumnBacking, LpgStore};

/// A backing holding one id whose value cannot be read; it holds no other.
pub(crate) struct Unreadable(pub(crate) NodeId);

impl ColumnBacking<NodeId> for Unreadable {
    fn get(&self, id: NodeId) -> std::io::Result<Option<Value>> {
        if id == self.0 {
            Err(std::io::Error::other("the spill file cannot be read"))
        } else {
            Ok(None)
        }
    }
    fn contains(&self, id: NodeId) -> bool {
        id == self.0
    }
    fn ids(&self) -> Vec<NodeId> {
        vec![self.0]
    }
    fn len(&self) -> usize {
        1
    }
    fn heap_bytes(&self) -> usize {
        0
    }
}

/// A backing that lends its vectors in place and counts the values `get`
/// copies out.
#[cfg(feature = "vector-index")]
pub(crate) struct Counting {
    vectors: HashMap<NodeId, Arc<[f32]>>,
    pub(crate) copies: AtomicUsize,
}

#[cfg(feature = "vector-index")]
impl Counting {
    /// A backing holding the vectors of `entries`.
    pub(crate) fn of(entries: &[(NodeId, Value)]) -> Arc<Self> {
        Arc::new(Self {
            vectors: entries
                .iter()
                .filter_map(|(id, value)| match value {
                    Value::Vector(vector) => Some((*id, Arc::clone(vector))),
                    _ => None,
                })
                .collect(),
            copies: AtomicUsize::new(0),
        })
    }

    pub(crate) fn copies(&self) -> usize {
        self.copies.load(Ordering::Relaxed)
    }
}

#[cfg(feature = "vector-index")]
impl ColumnBacking<NodeId> for Counting {
    fn get(&self, id: NodeId) -> std::io::Result<Option<Value>> {
        self.copies.fetch_add(1, Ordering::Relaxed);
        Ok(self
            .vectors
            .get(&id)
            .map(|vector| Value::Vector(vector.to_vec().into())))
    }
    fn contains(&self, id: NodeId) -> bool {
        self.vectors.contains_key(&id)
    }
    fn ids(&self) -> Vec<NodeId> {
        self.vectors.keys().copied().collect()
    }
    fn len(&self) -> usize {
        self.vectors.len()
    }
    fn heap_bytes(&self) -> usize {
        0
    }
    fn with_vector(&self, id: NodeId, f: &mut dyn FnMut(&[f32])) -> std::io::Result<bool> {
        Ok(match self.vectors.get(&id) {
            Some(vector) => {
                f(vector);
                true
            }
            None => false,
        })
    }
}

/// Spills the column `key` of `store` into `backing`, built from its
/// snapshot.
pub(crate) fn spill(store: &LpgStore, key: &str, backing: Arc<dyn ColumnBacking<NodeId>>) {
    let key = PropertyKey::new(key);
    let snapshot = store.node_property_column_entries(&key).unwrap();
    assert!(store.spill_node_property_column(&key, backing, &snapshot));
}

#[cfg(all(test, feature = "gql", feature = "vector-index"))]
mod tests {
    use super::*;
    use crate::GrafeoDB;

    /// Search through a session's store (the WAL decorator of a file
    /// database) reads a spilled column's vectors in place: the procedures
    /// and `vector_search` copy none per distance, and MMR copies only its
    /// candidates, once each (ruling (d) of #594).
    #[test]
    fn search_through_a_session_reads_spilled_vectors_in_place() {
        let dir = tempfile::tempdir().unwrap();
        let db = GrafeoDB::open(dir.path().join("paris.grafeo")).unwrap();
        let alix = db
            .create_node_with_props(
                &["Item"],
                [("embedding", Value::Vector(vec![3.0, 19.0].into()))],
            )
            .unwrap();
        for vector in [[19.0, 88.0], [88.0, 3.0], [3.19, 19.88]] {
            db.create_node_with_props(
                &["Item"],
                [("embedding", Value::Vector(vector.to_vec().into()))],
            )
            .unwrap();
        }
        db.create_vector_index("Item", "embedding", Some(2), None, None, None, None)
            .unwrap();
        let store = db.lpg_store();
        let backing = Counting::of(
            &store
                .node_property_column_entries(&PropertyKey::new("embedding"))
                .unwrap(),
        );
        spill(store, "embedding", backing.clone());

        let hits = db
            .vector_search("Item", "embedding", &[3.0, 19.0], 2, None, None)
            .unwrap();
        assert_eq!(hits[0].0, alix);
        assert_eq!(backing.copies(), 0, "vector_search copied");

        let rows = db
            .execute("CALL grafeo.search.vector('Item', 'embedding', [3.0, 19.0], 2)")
            .unwrap();
        assert_eq!(rows.rows().len(), 2);
        assert_eq!(backing.copies(), 0, "grafeo.search.vector copied");

        let rows = db
            .execute("CALL grafeo.search.mmr('Item', 'embedding', [3.0, 19.0], 2, 3, 0.5)")
            .unwrap();
        assert_eq!(rows.rows().len(), 2);
        assert!(
            backing.copies() <= 3,
            "grafeo.search.mmr copied {} vectors for 3 candidates",
            backing.copies()
        );
    }
}

/// A vector join over each store decorator reads a spilled column in place:
/// the decorators forward the borrowed read (the session test covers the WAL
/// store through queries; the join is not reachable from a query language).
#[cfg(all(
    test,
    feature = "vector-index",
    feature = "wal",
    feature = "cdc",
    feature = "grafeo-file"
))]
mod decorator_tests {
    use std::sync::Arc;

    use grafeo_common::types::{NodeId, PropertyKey, Value};
    use grafeo_core::execution::operators::{NodeListOperator, Operator, VectorJoinOperator};
    use grafeo_core::graph::lpg::LpgStore;
    use grafeo_core::graph::{GraphStoreMut, GraphStoreSearch};
    use grafeo_core::index::vector::{
        DistanceMetric, HnswConfig, HnswIndex, PropertyVectorAccessor, VectorIndexKind,
    };

    use super::{Counting, spill};

    /// Three items with spilled embeddings, and an index over them.
    fn spilled_items() -> (Arc<LpgStore>, NodeId, HnswIndex, Arc<Counting>) {
        let store = Arc::new(LpgStore::new().unwrap());
        let vectors = [[3.0, 19.0, 88.0], [19.0, 88.0, 3.0], [88.0, 3.0, 19.0]];
        let ids: Vec<NodeId> = vectors
            .iter()
            .map(|vector| {
                store.create_node_with_props(
                    &["Item"],
                    [("embedding", Value::Vector(vector.to_vec().into()))],
                )
            })
            .collect();
        let index = HnswIndex::new(HnswConfig::new(3, DistanceMetric::Euclidean));
        {
            let accessor = PropertyVectorAccessor::new(&*store, "embedding");
            for (id, vector) in ids.iter().zip(vectors) {
                index.insert(*id, &vector, &accessor);
            }
        }
        let backing = Counting::of(
            &store
                .node_property_column_entries(&PropertyKey::new("embedding"))
                .unwrap(),
        );
        spill(&store, "embedding", backing.clone());
        (store, ids[0], index, backing)
    }

    /// The rows a vector join from `left` over `store` returns.
    fn join_rows(store: Arc<dyn GraphStoreSearch>, left: NodeId, index: HnswIndex) -> usize {
        let mut join = VectorJoinOperator::with_static_query(
            Box::new(NodeListOperator::new(vec![left], 1024)),
            store,
            vec![3.0, 19.0, 88.0],
            "embedding",
            3,
            DistanceMetric::Euclidean,
        )
        .with_index(Arc::new(VectorIndexKind::Hnsw(index)));
        let mut rows = 0;
        while let Ok(Some(chunk)) = join.next() {
            rows += chunk.row_count();
        }
        rows
    }

    #[test]
    fn a_join_through_the_wal_store_reads_spilled_vectors_in_place() {
        let (store, left, index, backing) = spilled_items();
        let dir = tempfile::tempdir().unwrap();
        let wal = Arc::new(crate::transaction::wal_buffer::WalBuffer::new(Arc::new(
            grafeo_storage::wal::TypedWal::open(dir.path()).unwrap(),
        )));
        let wrapped = Arc::new(super::super::wal_store::WalGraphStore::new(store, wal));
        assert_eq!(join_rows(wrapped, left, index), 3);
        assert_eq!(backing.copies(), 0, "the join through the WAL store copied");
    }

    #[test]
    fn a_join_through_the_cdc_store_reads_spilled_vectors_in_place() {
        let (store, left, index, backing) = spilled_items();
        let wrapped = Arc::new(super::super::cdc_store::CdcGraphStore::new(
            store as Arc<dyn GraphStoreMut>,
            Arc::new(crate::cdc::CdcLog::new()),
        ));
        assert_eq!(join_rows(wrapped, left, index), 3);
        assert_eq!(backing.copies(), 0, "the join through the CDC store copied");
    }
}

/// Ordering the nodes by a key (`key=` of the algorithms, #566) reads the key
/// fallibly: a spilled value that cannot be read is a read error, never a node
/// without the key, through every store the algorithms read.
#[cfg(feature = "algos")]
mod key_order_tests {
    use std::sync::Arc;

    use grafeo_adapters::plugins::algorithms::{KeyOrderError, order_by_key};
    use grafeo_common::types::{NodeId, Value};
    use grafeo_core::graph::lpg::LpgStore;
    use grafeo_core::graph::{GraphStore, ProjectionSpec};

    use super::{Unreadable, spill};
    use crate::GrafeoDB;

    /// Alix, whose `id` is spilled and cannot be read, and Gus, whose `id`
    /// was written after the spill (so the column holds it).
    fn with_an_unreadable_key(store: &LpgStore) -> NodeId {
        let alix = store.create_node_with_props(&["Person"], [("id", Value::from("alix"))]);
        spill(store, "id", Arc::new(Unreadable(alix)));
        store.create_node_with_props(&["Person"], [("id", Value::from("gus"))]);
        alix
    }

    fn assert_read_error(store: &dyn GraphStore, through: &str) {
        let error = order_by_key(store, "id").expect_err(through);
        assert!(
            matches!(&error, KeyOrderError::Read { key, .. } if key == "id"),
            "{through}: {error}"
        );
        assert_eq!(
            error.to_string(),
            concat!(
                "the values of the key 'id' cannot be read: ",
                "GRAFEO-X003: I/O error: the spill file cannot be read"
            ),
            "{through}"
        );
    }

    #[test]
    fn a_key_that_cannot_be_read_is_a_read_error() {
        let db = GrafeoDB::new_in_memory();
        with_an_unreadable_key(db.lpg_store());
        assert_read_error(&**db.lpg_store(), "the store");
        assert_read_error(&*db.selected_graph_store().unwrap(), "the selected graph");
        assert!(
            db.create_projection("people", ProjectionSpec::new().with_node_labels(["Person"]))
                .unwrap()
        );
        assert_read_error(&*db.projection("people").unwrap(), "a projection");
    }

    /// The WAL and CDC decorators forward the fallible read.
    #[cfg(all(feature = "wal", feature = "cdc", feature = "grafeo-file"))]
    #[test]
    fn a_key_that_cannot_be_read_is_a_read_error_through_the_decorators() {
        use grafeo_core::graph::GraphStoreMut;

        let store = Arc::new(LpgStore::new().unwrap());
        with_an_unreadable_key(&store);
        let dir = tempfile::tempdir().unwrap();
        let wal = Arc::new(crate::transaction::wal_buffer::WalBuffer::new(Arc::new(
            grafeo_storage::wal::TypedWal::open(dir.path()).unwrap(),
        )));
        let wal_store = super::super::wal_store::WalGraphStore::new(Arc::clone(&store), wal);
        assert_read_error(&wal_store, "the WAL store");
        let cdc_store = super::super::cdc_store::CdcGraphStore::new(
            store as Arc<dyn GraphStoreMut>,
            Arc::new(crate::cdc::CdcLog::new()),
        );
        assert_read_error(&cdc_store, "the CDC store");
    }
}
