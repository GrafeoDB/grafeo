//! `text_match` and `text_score` run per row on a database built on a store
//! of its own (`GrafeoDB::with_read_store` or `with_store`) when that store
//! keeps a text index on the property: the planner asks the store
//! (`has_text_index`), as it does for the pushed-down search, and each row
//! gets the score the store answers (`score_text`), as the query's view sees
//! the node. A store without the index refuses the query, saying so (#809).
//!
//! ```bash
//! cargo test -p grafeo-engine --all-features --test main text_search_on_a_store_of_its_own::
//! ```

#![cfg(all(feature = "text-index", feature = "gql", feature = "cypher"))]

use std::collections::HashMap;
use std::sync::Arc;

use grafeo_common::types::{ArcStr, EdgeId, EpochId, NodeId, ReadView, Value};
use grafeo_common::utils::error::{ErrorCode, Result};
use grafeo_core::graph::lpg::{Edge, Node};
use grafeo_core::graph::rowgroup::RowGroupStore;
use grafeo_core::graph::{Direction, GraphBackend, GraphStore};
use grafeo_core::index::{BM25Config, InvertedIndex};
use grafeo_core::statistics::Statistics;
use grafeo_engine::{Config, GrafeoDB};
use parking_lot::RwLock;

const LANGUAGES: [&str; 2] = ["gql", "cypher"];

/// A read-only store of its own with three `Doc` nodes and a text index on
/// `Doc.body` whose scores are fixed: for 'graph', Alix scores 0.75, Mia 0.25
/// and Gus 0.
struct Shelf;

/// The documents: id, title, body and score for 'graph'.
const DOCS: [(u64, &str, &str, f64); 3] = [
    (1, "Alix", "graph databases", 0.75),
    (2, "Gus", "row groups", 0.0),
    (3, "Mia", "graph theory", 0.25),
];

impl Shelf {
    /// The fixed score of document `id` for `query`.
    fn score(id: NodeId, query: &str) -> f64 {
        DOCS.iter()
            .find(|doc| NodeId::new(doc.0) == id && query == "graph")
            .map_or(0.0, |doc| doc.3)
    }

    /// The documents scoring at least `threshold` (above 0), best first.
    fn found(query: &str, threshold: f64) -> Vec<(NodeId, f64)> {
        let mut found: Vec<(NodeId, f64)> = DOCS
            .iter()
            .map(|doc| (NodeId::new(doc.0), Self::score(NodeId::new(doc.0), query)))
            .filter(|&(_, score)| score > 0.0 && score >= threshold)
            .collect();
        found.sort_by(|a, b| b.1.total_cmp(&a.1));
        found
    }
}

impl GraphStore for Shelf {
    fn node(&self, id: NodeId, _: ReadView) -> Result<Option<Node>> {
        Ok(DOCS
            .iter()
            .find(|doc| NodeId::new(doc.0) == id)
            .map(|&(_, title, body, _)| {
                let mut node = Node::with_labels(id, ["Doc"]);
                node.set_property("title", title);
                node.set_property("body", body);
                node
            }))
    }
    fn edge(&self, _: EdgeId, _: ReadView) -> Result<Option<Edge>> {
        Ok(None)
    }
    fn edges(
        &self,
        _: NodeId,
        _: Direction,
        _: &[ArcStr],
        _: ReadView,
    ) -> Result<Vec<(NodeId, EdgeId)>> {
        Ok(Vec::new())
    }
    fn has_backward_adjacency(&self) -> bool {
        true
    }
    fn node_ids(&self, _: ReadView) -> Result<Vec<NodeId>> {
        Ok(DOCS.iter().map(|doc| NodeId::new(doc.0)).collect())
    }
    fn all_labels(&self) -> Result<Vec<String>> {
        Ok(vec!["Doc".into()])
    }
    fn all_edge_types(&self) -> Result<Vec<String>> {
        Ok(Vec::new())
    }
    fn all_property_keys(&self) -> Result<Vec<String>> {
        Ok(vec!["title".into(), "body".into()])
    }
    fn statistics(&self) -> Arc<Statistics> {
        Arc::new(Statistics::default())
    }
    fn committed_epoch(&self) -> EpochId {
        EpochId::new(1)
    }
    fn has_text_index(&self, label: &str, property: &str) -> bool {
        (label, property) == ("Doc", "body")
    }
    fn score_text(
        &self,
        id: NodeId,
        label: &str,
        property: &str,
        query: &str,
        _: ReadView,
    ) -> Result<Option<f64>> {
        Ok(self
            .has_text_index(label, property)
            .then(|| Self::score(id, query)))
    }
    fn text_search(
        &self,
        label: &str,
        property: &str,
        query: &str,
        k: usize,
        _: ReadView,
    ) -> Result<Vec<(NodeId, f64)>> {
        let mut found = Vec::new();
        if self.has_text_index(label, property) {
            found = Self::found(query, 0.0);
            found.truncate(k);
        }
        Ok(found)
    }
    fn text_search_with_threshold(
        &self,
        label: &str,
        property: &str,
        query: &str,
        threshold: f64,
        _: ReadView,
    ) -> Result<Vec<(NodeId, f64)>> {
        Ok(if self.has_text_index(label, property) {
            Self::found(query, threshold)
        } else {
            Vec::new()
        })
    }
}

/// A database built on the read-only store of its own, [`Shelf`].
fn on_the_shelf() -> GrafeoDB {
    GrafeoDB::with_read_store(Arc::new(Shelf), Config::in_memory()).unwrap()
}

/// A database on a writable store of its own with Alix ('graph databases'),
/// Gus ('row groups') and Mia ('graph theory'), the store keeping a text
/// index on `Person.<indexed>` when given.
fn on_a_store_of_its_own(indexed: Option<&str>) -> GrafeoDB {
    let store = Arc::new(RowGroupStore::new());
    if let Some(property) = indexed {
        store.add_text_index(
            "Person",
            property,
            Arc::new(RwLock::new(InvertedIndex::new(BM25Config::default()))),
        );
    }
    let db = GrafeoDB::with_store(store as Arc<dyn GraphBackend>, Config::in_memory()).unwrap();
    db.execute(
        "INSERT (:Person {name: 'Alix', bio: 'graph databases'}), \
         (:Person {name: 'Gus', bio: 'row groups'}), \
         (:Person {name: 'Mia', bio: 'graph theory'})",
    )
    .unwrap();
    db
}

/// The rows of `query`, run with `params`.
fn rows(
    db: &GrafeoDB,
    language: &str,
    query: &str,
    params: Option<HashMap<String, Value>>,
) -> Vec<Vec<Value>> {
    db.execute_language(query, language, params)
        .unwrap_or_else(|error| panic!("{language} `{query}`: {error}"))
        .rows()
        .to_vec()
}

/// The titles (or names) in the first column of `query`'s rows, in row order.
fn names(db: &GrafeoDB, language: &str, query: &str) -> Vec<Value> {
    rows(db, language, query, None)
        .into_iter()
        .map(|row| row[0].clone())
        .collect()
}

/// Each row's `text_score` is the score the store answers, not a stand-in.
#[test]
fn text_score_is_the_score_the_store_answers() {
    let db = on_the_shelf();
    for language in LANGUAGES {
        assert_eq!(
            rows(
                &db,
                language,
                "MATCH (d:Doc) RETURN d.title, text_score(d.body, 'graph') AS score \
                 ORDER BY d.title",
                None,
            ),
            DOCS.map(|doc| vec![Value::from(doc.1), Value::Float64(doc.3)]),
            "{language}"
        );
        assert_eq!(
            names(
                &db,
                language,
                "MATCH (d:Doc) RETURN d.title ORDER BY text_score(d.body, 'graph') DESC",
            ),
            [Value::from("Alix"), Value::from("Mia"), Value::from("Gus")],
            "{language}"
        );
    }
}

/// The issue's form: `text_match` under OR runs per row.
#[test]
fn text_match_runs_per_row_under_or_and_not() {
    let db = on_the_shelf();
    for language in LANGUAGES {
        assert_eq!(
            names(
                &db,
                language,
                "MATCH (d:Doc) WHERE text_match(d.body, 'graph') OR d.body = 'x' \
                 RETURN d.title ORDER BY d.title",
            ),
            [Value::from("Alix"), Value::from("Mia")],
            "{language}: OR"
        );
        assert_eq!(
            names(
                &db,
                language,
                "MATCH (d:Doc) WHERE NOT text_match(d.body, 'graph') RETURN d.title",
            ),
            [Value::from("Gus")],
            "{language}: NOT"
        );
    }
}

/// Forms the planner cannot push into a text search run per row: a scan
/// without a label, a threshold that is not a literal, and a scan with an
/// input (an UNWIND or a MATCH before it).
#[test]
fn text_search_runs_per_row_where_it_is_not_pushed_down() {
    let db = on_the_shelf();
    let alix_and_mia = [Value::from("Alix"), Value::from("Mia")];
    for language in LANGUAGES {
        assert_eq!(
            names(
                &db,
                language,
                "MATCH (d) WHERE text_match(d.body, 'graph') RETURN d.title ORDER BY d.title",
            ),
            alix_and_mia,
            "{language}: no label"
        );
        assert_eq!(
            rows(
                &db,
                language,
                "MATCH (d:Doc) WHERE text_score(d.body, 'graph') > $t RETURN d.title",
                Some(HashMap::from([("t".to_string(), Value::Float64(0.5))])),
            ),
            [[Value::from("Alix")]],
            "{language}: a parameter as the threshold"
        );
        assert_eq!(
            names(
                &db,
                language,
                "UNWIND [1] AS x MATCH (d:Doc) WHERE text_match(d.body, 'graph') \
                 RETURN d.title ORDER BY d.title",
            ),
            alix_and_mia,
            "{language}: after an UNWIND"
        );
        assert_eq!(
            names(
                &db,
                language,
                "MATCH (g:Doc {title: 'Gus'}) MATCH (d:Doc) WHERE text_match(d.body, 'graph') \
                 RETURN d.title ORDER BY d.title",
            ),
            alix_and_mia,
            "{language}: after another MATCH"
        );
    }
}

/// The pushed-down search and the per-row score agree on the same store.
#[test]
fn the_pushed_down_search_and_the_per_row_score_agree() {
    let db = on_the_shelf();
    for language in LANGUAGES {
        assert_eq!(
            names(
                &db,
                language,
                "MATCH (d:Doc) WHERE text_match(d.body, 'graph') RETURN d.title ORDER BY d.title",
            ),
            names(
                &db,
                language,
                "MATCH (d) WHERE text_match(d.body, 'graph') RETURN d.title ORDER BY d.title",
            ),
            "{language}"
        );
    }
}

/// A store without a text index on the property refuses the query, saying it
/// reports none: with no text index at all, and with one on another property.
#[test]
fn a_store_without_the_index_says_it_reports_none() {
    for indexed in [None, Some("name")] {
        let db = on_a_store_of_its_own(indexed);
        for language in LANGUAGES {
            for query in [
                "MATCH (p:Person) WHERE text_match(p.bio, 'graph') RETURN p.name",
                "MATCH (p:Person) RETURN p.name, text_score(p.bio, 'graph') AS score",
            ] {
                let error = db.execute_language(query, language, None).expect_err(query);
                let message = error.to_string();
                assert_eq!(
                    error.error_code(),
                    ErrorCode::QueryUnsupported,
                    "{indexed:?} {language} `{query}`: {message}"
                );
                assert!(
                    message.contains("the store reports none"),
                    "{indexed:?} {language} `{query}`: {message}"
                );
                assert!(
                    !message.contains("built-in store") && !message.contains("CREATE INDEX"),
                    "{indexed:?} {language} `{query}`: {message}"
                );
            }
        }
    }
}

/// The score is read as the query's view sees the node: a transaction that
/// changed Alix's text scores her by its own write, and another session by
/// the committed text.
#[test]
fn a_score_follows_the_view_of_the_query() {
    let db = on_a_store_of_its_own(Some("bio"));
    for language in LANGUAGES {
        let mut writer = db.session();
        writer.begin_transaction().unwrap();
        writer
            .execute("MATCH (p:Person {name: 'Alix'}) SET p.bio = 'row stores'")
            .unwrap();
        let query = "MATCH (p:Person {name: 'Alix'}) RETURN text_score(p.bio, 'graph') AS score";
        let score = |result: grafeo_engine::database::QueryResult| match &result.rows()[0][0] {
            Value::Float64(score) => *score,
            other => panic!("{language}: a score, got {other:?}"),
        };
        let own = score(writer.execute_language(query, language, None).unwrap());
        let other = score(
            db.session()
                .execute_language(query, language, None)
                .unwrap(),
        );
        assert!(
            own.abs() < f64::EPSILON,
            "{language}: the writer sees its own text: {own}"
        );
        assert!(
            other > 0.0,
            "{language}: another session sees the committed text: {other}"
        );
        writer.rollback().unwrap();
    }
}
