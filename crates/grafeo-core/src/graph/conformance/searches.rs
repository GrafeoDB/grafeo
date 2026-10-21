//! Text and vector searches at a view, on every store: each reader's
//! search finds the documents and vectors of its view (a transaction's own
//! writes found, another open one's not, a reader at an earlier epoch
//! finding those of its epoch), also through an index built from what a
//! reader at the current epoch sees and added while transactions are open,
//! as the engine adds one (#412 S7, decision Q5 A). A store takes part
//! through [`text::TextIndexed`] and [`vector::VectorIndexed`], which
//! forward to its own methods of the same names.

use grafeo_common::change::DataOp;
use grafeo_common::types::{NodeId, ReadView, Value};

use super::{Store, Tx, apply_all, labels, next_epoch, properties, reserve_node};
use crate::graph::apply::Writer;
use crate::graph::rowgroup::RowGroupStore;

/// A document with `body` and `embedding`.
fn doc(id: NodeId, body: &str, embedding: [f32; 3]) -> DataOp {
    DataOp::CreateNode {
        id,
        labels: labels(&["Doc"]),
        properties: properties(&[
            ("body", Value::from(body)),
            ("embedding", Value::Vector(embedding.to_vec().into())),
        ]),
    }
}

/// Three documents, replayed: Alix's on graphs in Amsterdam, Gus's on
/// storage engines, Mia's on graphs in Berlin, with embeddings along the
/// three axes.
fn docs<S: Store>(store: &S) -> Result<[NodeId; 3], String> {
    let ids = [
        NodeId::new(reserve_node(store)?),
        NodeId::new(reserve_node(store)?),
        NodeId::new(reserve_node(store)?),
    ];
    let ops = [
        doc(ids[0], "graph databases in Amsterdam", [1.0, 0.0, 0.0]),
        doc(ids[1], "rust storage engines", [0.0, 1.0, 0.0]),
        doc(ids[2], "graph theory in Berlin", [0.0, 0.0, 1.0]),
    ];
    apply_all(
        store,
        &ops,
        Writer::Replay {
            epoch: next_epoch(store),
        },
    )?;
    Ok(ids)
}

/// Two open transactions over [`docs`]: the writer rewrites Alix's
/// document (no `graph`, now `storage`) and moves its embedding away, the
/// other creates Jules's on graph storage near the first axis and takes
/// `Doc` off Mia's.
struct Writes<'s, S: Store> {
    writer: Tx<'s, S>,
    other: Tx<'s, S>,
    jules: NodeId,
}

impl<'s, S: Store> Writes<'s, S> {
    fn begin(store: &'s S, ids: &[NodeId; 3]) -> Result<Self, String> {
        let mut writer = Tx::begin(store, 10);
        let mut other = Tx::begin(store, 11);
        writer.set_node(ids[0], "body", Value::from("storage of columns in Paris"))?;
        writer.set_node(
            ids[0],
            "embedding",
            Value::Vector(vec![0.0, 0.0, 2.0].into()),
        )?;
        let jules = other.create_node(&["Doc"], &[])?;
        other.set_node(jules, "body", Value::from("graph storage"))?;
        other.set_node(
            jules,
            "embedding",
            Value::Vector(vec![0.9, 0.1, 0.0].into()),
        )?;
        other.remove_label(ids[2], "Doc")?;
        Ok(Self {
            writer,
            other,
            jules,
        })
    }

    /// The writer's view and the other's.
    fn views(&self) -> (ReadView, ReadView) {
        (
            ReadView::transaction(self.writer.id, self.writer.snapshot),
            ReadView::transaction(self.other.id, self.other.snapshot),
        )
    }

    /// Commits the writer and rolls the other back; returns the view at
    /// the commit.
    fn end(self) -> Result<ReadView, String> {
        let committed = ReadView::committed(self.writer.commit()?);
        self.other.roll_back()?;
        Ok(committed)
    }
}

/// Instantiates the cases of a search module for the row-group store.
macro_rules! search_suites {
    ($($case:ident),* $(,)?) => {
        /// The row-group store's cases: it answers every one at its view.
        mod rowgroup {
            use super::super::RowGroupStore;

            $crate::graph::conformance::indexes::index_cases!(
                RowGroupStore, RowGroupStore::new, &[], super; $($case),*
            );
        }
    };
}

#[cfg(feature = "text-index")]
mod text {
    use std::collections::BTreeSet;
    use std::sync::Arc;

    use grafeo_common::types::{NodeId, PropertyKey, ReadView, Value};
    use parking_lot::RwLock;

    use super::{RowGroupStore, Writes, docs};
    use crate::graph::conformance::indexes::set;
    use crate::graph::conformance::{Outcome, Store, ensure, people, same};
    use crate::index::text::{BM25Config, InvertedIndex};

    /// A store that holds text indexes for the engine.
    pub(crate) trait TextIndexed: Store {
        fn add_text_index(&self, label: &str, property: &str, index: Arc<RwLock<InvertedIndex>>);
        fn remove_text_index(&self, label: &str, property: &str) -> bool;
        fn text_index_entries(&self) -> Vec<(String, Arc<RwLock<InvertedIndex>>)>;
    }

    /// Implements [`TextIndexed`] for a store by its own methods.
    macro_rules! text_indexed {
        ($store:ty) => {
            impl TextIndexed for $store {
                fn add_text_index(
                    &self,
                    label: &str,
                    property: &str,
                    index: Arc<RwLock<InvertedIndex>>,
                ) {
                    <$store>::add_text_index(self, label, property, index);
                }
                fn remove_text_index(&self, label: &str, property: &str) -> bool {
                    <$store>::remove_text_index(self, label, property)
                }
                fn text_index_entries(&self) -> Vec<(String, Arc<RwLock<InvertedIndex>>)> {
                    <$store>::text_index_entries(self)
                }
            }
        };
    }

    text_indexed!(RowGroupStore);

    /// An empty text index.
    fn text_index() -> Arc<RwLock<InvertedIndex>> {
        Arc::new(RwLock::new(InvertedIndex::new(BM25Config::default())))
    }

    /// The documents a search of `query` finds at `view`: the top 10, and
    /// every one scoring at least 0, which must agree.
    fn searched<S: TextIndexed>(
        store: &S,
        query: &str,
        view: ReadView,
    ) -> Result<BTreeSet<u64>, String> {
        let top: Vec<NodeId> = store
            .text_search_at("Doc", "body", query, 10, view)
            .into_iter()
            .map(|(id, _)| id)
            .collect();
        let all: Vec<NodeId> = store
            .text_search_with_threshold_at("Doc", "body", query, 0.0, view)
            .into_iter()
            .map(|(id, _)| id)
            .collect();
        same(
            &set(&all),
            &set(&top),
            &format!("the threshold search of {query} at {view:?}"),
        )?;
        Ok(set(&top))
    }

    /// The searches of `graph` and `storage` at the views of [`Writes`],
    /// the scores of single documents, and the searches after the writer
    /// commits and the other rolls back.
    fn check<S: TextIndexed>(
        store: &S,
        ids: &[NodeId; 3],
        writes: Writes<'_, S>,
        before: ReadView,
    ) -> Outcome {
        let (own, others) = writes.views();
        let jules = writes.jules;
        same(
            &searched(store, "graph", before)?,
            &set(&[ids[0], ids[2]]),
            "graph before",
        )?;
        same(
            &searched(store, "storage", before)?,
            &set(&[ids[1]]),
            "storage before",
        )?;
        same(
            &searched(store, "graph", own)?,
            &set(&[ids[2]]),
            "graph for the writer",
        )?;
        same(
            &searched(store, "storage", own)?,
            &set(&[ids[0], ids[1]]),
            "storage for the writer",
        )?;
        same(
            &searched(store, "graph", others)?,
            &set(&[ids[0], jules]),
            "graph for the other",
        )?;
        same(
            &searched(store, "storage", others)?,
            &set(&[ids[1], jules]),
            "storage for the other",
        )?;
        let score = |id: NodeId, query: &str, view: ReadView| {
            store
                .score_text_at(id, "Doc", "body", query, view)
                .unwrap_or(-1.0)
        };
        ensure(
            score(ids[0], "graph", before) > 0.0 && score(ids[0], "graph", own) == 0.0,
            || "score_text_at reads Alix's document at the view".to_string(),
        )?;
        ensure(
            score(ids[2], "graph", others) == 0.0 && score(ids[2], "graph", before) > 0.0,
            || "score_text_at reads Mia's labels at the view".to_string(),
        )?;
        // A document every reader sees alike scores with the statistics of
        // now at every view.
        let now = ReadView::committed(store.current_epoch());
        same(
            &score(ids[1], "storage", before),
            &score(ids[1], "storage", now),
            "Gus's score",
        )?;

        let committed = writes.end()?;
        same(
            &searched(store, "graph", before)?,
            &set(&[ids[0], ids[2]]),
            "graph before, after the commit",
        )?;
        same(
            &searched(store, "graph", committed)?,
            &set(&[ids[2]]),
            "graph at the commit",
        )?;
        same(
            &searched(store, "storage", committed)?,
            &set(&[ids[0], ids[1]]),
            "storage at the commit",
        )
    }

    /// A text index the store keeps from its creation.
    pub(crate) fn a_text_search_answers_at_its_view<S: TextIndexed>(
        make: &dyn Fn() -> S,
    ) -> Outcome {
        let store = make();
        store.add_text_index("Doc", "body", text_index());
        let ids = docs(&store)?;
        let before = ReadView::committed(store.current_epoch());
        let writes = Writes::begin(&store, &ids)?;
        check(&store, &ids, writes, before)
    }

    /// A text index built from what a reader at the current epoch sees and
    /// added while transactions are open, as the engine adds one.
    pub(crate) fn a_text_index_added_during_a_transaction_answers_at_its_view<S: TextIndexed>(
        make: &dyn Fn() -> S,
    ) -> Outcome {
        let store = make();
        let ids = docs(&store)?;
        let before = ReadView::committed(store.current_epoch());
        let writes = Writes::begin(&store, &ids)?;
        let index = text_index();
        {
            let mut built = index.write();
            let key = PropertyKey::new("body");
            for id in store.nodes_by_label("Doc") {
                if let Some(Value::String(text)) = store.get_node_property(id, &key) {
                    built.insert(id, &text);
                }
            }
        }
        store.add_text_index("Doc", "body", index);
        check(&store, &ids, writes, before)
    }

    /// A text index is listed once added and gone once removed.
    pub(crate) fn a_text_index_is_added_listed_and_removed<S: TextIndexed>(
        make: &dyn Fn() -> S,
    ) -> Outcome {
        let store = make();
        people(&store)?;
        store.add_text_index("Person", "name", text_index());
        let keys: Vec<String> = store
            .text_index_entries()
            .into_iter()
            .map(|(key, _)| key)
            .collect();
        same(&keys, &vec!["Person:name".to_string()], "the text indexes")?;
        ensure(store.has_text_index("Person", "name"), || {
            "has_text_index".to_string()
        })?;
        ensure(
            store.remove_text_index("Person", "name")
                && !store.remove_text_index("Person", "name")
                && !store.has_text_index("Person", "name"),
            || "remove_text_index returns whether it removed".to_string(),
        )
    }

    search_suites!(
        a_text_search_answers_at_its_view,
        a_text_index_added_during_a_transaction_answers_at_its_view,
        a_text_index_is_added_listed_and_removed,
    );
}

#[cfg(feature = "vector-index")]
mod vector {
    use std::sync::Arc;

    use grafeo_common::types::{NodeId, PropertyKey, ReadView, Value};

    use super::{RowGroupStore, Writes, docs};
    use crate::graph::conformance::{Outcome, Store, ensure, people, same};
    use crate::index::vector::{
        DistanceMetric, HnswConfig, HnswIndex, VectorAccessor, VectorIndexKind,
    };

    /// A store that holds vector indexes for the engine.
    pub(crate) trait VectorIndexed: Store {
        fn add_vector_index(&self, label: &str, property: &str, index: Arc<VectorIndexKind>);
        fn remove_vector_index(&self, label: &str, property: &str) -> bool;
        fn vector_index_entries(&self) -> Vec<(String, Arc<VectorIndexKind>)>;
    }

    /// Implements [`VectorIndexed`] for a store by its own methods.
    macro_rules! vector_indexed {
        ($store:ty) => {
            impl VectorIndexed for $store {
                fn add_vector_index(
                    &self,
                    label: &str,
                    property: &str,
                    index: Arc<VectorIndexKind>,
                ) {
                    <$store>::add_vector_index(self, label, property, index);
                }
                fn remove_vector_index(&self, label: &str, property: &str) -> bool {
                    <$store>::remove_vector_index(self, label, property)
                }
                fn vector_index_entries(&self) -> Vec<(String, Arc<VectorIndexKind>)> {
                    <$store>::vector_index_entries(self)
                }
            }
        };
    }

    vector_indexed!(RowGroupStore);

    /// An empty HNSW index of 3 dimensions by Euclidean distance.
    fn vector_index() -> Arc<VectorIndexKind> {
        Arc::new(VectorIndexKind::Hnsw(HnswIndex::new(HnswConfig::new(
            3,
            DistanceMetric::Euclidean,
        ))))
    }

    /// The query of the searches: the first axis.
    const QUERY: [f32; 3] = [1.0, 0.0, 0.0];

    /// The `k` nodes nearest to [`QUERY`] at `view`, nearest first, through
    /// the index; a threshold scan must give the same nearest `k`, and a
    /// search by another metric (a scan) as many nodes as the threshold
    /// scan finds.
    fn nearest<S: VectorIndexed>(store: &S, k: usize, view: ReadView) -> Result<Vec<u64>, String> {
        let euclidean = DistanceMetric::Euclidean;
        let through: Vec<u64> = store
            .vector_search_at(Some("Doc"), "embedding", &QUERY, k, euclidean, view)
            .into_iter()
            .map(|(id, _)| id.as_u64())
            .collect();
        let within = store.vector_search_with_threshold_at(
            Some("Doc"),
            "embedding",
            &QUERY,
            100.0,
            euclidean,
            view,
        );
        let scanned: Vec<u64> = within.iter().take(k).map(|(id, _)| id.as_u64()).collect();
        same(
            &through,
            &scanned,
            &format!("the index and the threshold scan at {view:?}"),
        )?;
        let cosine = store
            .vector_search_at(
                Some("Doc"),
                "embedding",
                &QUERY,
                10,
                DistanceMetric::Cosine,
                view,
            )
            .len();
        same(
            &cosine,
            &within.len(),
            &format!("the scan of another metric at {view:?}"),
        )?;
        Ok(through)
    }

    /// The searches at the views of [`Writes`]: before them Alix's document
    /// is nearest; the writer moved it away, the other added Jules's next to
    /// it and took Mia's out of `Doc`.
    fn check<S: VectorIndexed>(
        store: &S,
        ids: &[NodeId; 3],
        writes: Writes<'_, S>,
        before: ReadView,
    ) -> Outcome {
        let (own, others) = writes.views();
        let jules = writes.jules.as_u64();
        let (alix, gus, mia) = (ids[0].as_u64(), ids[1].as_u64(), ids[2].as_u64());
        same(
            &nearest(store, 1, before)?,
            &vec![alix],
            "the nearest before",
        )?;
        same(
            &nearest(store, 10, before)?.len(),
            &3,
            "every document before",
        )?;
        same(
            &nearest(store, 3, own)?,
            &vec![gus, mia, alix],
            "the writer's nearest",
        )?;
        same(
            &nearest(store, 2, others)?,
            &vec![alix, jules],
            "the other's nearest",
        )?;
        same(
            &nearest(store, 10, others)?.len(),
            &3,
            "Mia's is no Doc for the other",
        )?;
        let committed = writes.end()?;
        same(
            &nearest(store, 1, before)?,
            &vec![alix],
            "the nearest before, after the commit",
        )?;
        same(
            &nearest(store, 3, committed)?,
            &vec![gus, mia, alix],
            "the nearest at the commit",
        )?;
        same(
            &nearest(store, 10, committed)?.len(),
            &3,
            "Jules's is rolled back",
        )
    }

    /// A vector index the store keeps from its creation.
    pub(crate) fn a_vector_search_answers_at_its_view<S: VectorIndexed>(
        make: &dyn Fn() -> S,
    ) -> Outcome {
        let store = make();
        store.add_vector_index("Doc", "embedding", vector_index());
        let ids = docs(&store)?;
        let before = ReadView::committed(store.current_epoch());
        let writes = Writes::begin(&store, &ids)?;
        check(&store, &ids, writes, before)
    }

    /// The vectors of `embedding` a build reads, at the store's epoch.
    struct Built<'s, S: Store>(&'s S);

    impl<S: Store> VectorAccessor for Built<'_, S> {
        fn get_vector(&self, id: NodeId) -> Option<Arc<[f32]>> {
            match self.0.get_node_property(id, &PropertyKey::new("embedding")) {
                Some(Value::Vector(vector)) => Some(vector),
                _ => None,
            }
        }
    }

    /// A vector index built from what a reader at the current epoch sees
    /// and added while transactions are open, as the engine adds one.
    pub(crate) fn a_vector_index_added_during_a_transaction_answers_at_its_view<
        S: VectorIndexed,
    >(
        make: &dyn Fn() -> S,
    ) -> Outcome {
        let store = make();
        let ids = docs(&store)?;
        let before = ReadView::committed(store.current_epoch());
        let writes = Writes::begin(&store, &ids)?;
        let index = vector_index();
        let built = Built(&store);
        for id in store.nodes_by_label("Doc") {
            if let Some(vector) = built.get_vector(id) {
                index.insert(id, &vector, &built);
            }
        }
        store.add_vector_index("Doc", "embedding", index);
        check(&store, &ids, writes, before)
    }

    /// A vector index is listed once added and gone once removed.
    pub(crate) fn a_vector_index_is_added_listed_and_removed<S: VectorIndexed>(
        make: &dyn Fn() -> S,
    ) -> Outcome {
        let store = make();
        people(&store)?;
        store.add_vector_index("Person", "embedding", vector_index());
        let keys: Vec<String> = store
            .vector_index_entries()
            .into_iter()
            .map(|(key, _)| key)
            .collect();
        same(
            &keys,
            &vec!["Person:embedding".to_string()],
            "the vector indexes",
        )?;
        ensure(
            store.has_vector_index("Person", "embedding")
                && store
                    .vector_index_config("Person", "embedding")
                    .is_some_and(|config| config.dimensions == 3),
            || "has_vector_index and vector_index_config".to_string(),
        )?;
        ensure(
            store.remove_vector_index("Person", "embedding")
                && !store.remove_vector_index("Person", "embedding")
                && !store.has_vector_index("Person", "embedding"),
            || "remove_vector_index returns whether it removed".to_string(),
        )
    }

    search_suites!(
        a_vector_search_answers_at_its_view,
        a_vector_index_added_during_a_transaction_answers_at_its_view,
        a_vector_index_is_added_listed_and_removed,
    );
}
