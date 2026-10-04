//! Indexes and constraints survive `to_memory()` and reopening a `.grafeo`
//! file. Both load the database from its checkpoint sections, and loading
//! puts back the property, vector and text indexes of every graph, with the
//! names that `DROP INDEX` and `DROP CONSTRAINT` use.
//!
//! ```bash
//! cargo test -p grafeo-engine --all-features --test index_persistence
//! ```

#![cfg(all(feature = "lpg", feature = "gql"))]

use grafeo_common::types::Value;
use grafeo_engine::GrafeoDB;

/// A property index, a named index and a UNIQUE constraint in the default
/// graph, and a property index in the named graph `model`.
fn build(db: &GrafeoDB) {
    db.create_property_index("id");
    db.execute("CREATE INDEX file_size FOR (n:File) ON (n.size)")
        .unwrap();
    db.execute("CREATE CONSTRAINT file_id FOR (n:File) ON (n.id) UNIQUE")
        .unwrap();
    db.execute("INSERT (:File {id: 'f0', size: 1})").unwrap();
    db.create_graph("model").unwrap();
    let model = db.graph("model").unwrap().session().unwrap();
    model.create_property_index("id");
    model.execute("INSERT (:Component {id: 'c0'})").unwrap();
}

/// Everything `build` made, working in `db`.
fn assert_built(db: &GrafeoDB) {
    assert!(db.has_property_index("id"));
    assert!(db.has_property_index("size"));
    assert_eq!(db.find_nodes_by_property("id", &Value::from("f0")).len(), 1);
    let duplicate = db.execute("INSERT (:File {id: 'f0'})").unwrap_err();
    assert!(duplicate.to_string().contains("UNIQUE"), "{duplicate}");

    let model = db.graph("model").unwrap().session().unwrap();
    assert!(model.has_property_index("id"));
    assert!(
        !model.has_property_index("size"),
        "indexes stay in their graph"
    );
    assert_eq!(
        model.find_nodes_by_property("id", &Value::from("c0")).len(),
        1
    );
}

#[test]
fn to_memory_keeps_indexes_and_constraints() {
    let db = GrafeoDB::new_in_memory();
    build(&db);
    let copy = db.to_memory().unwrap();
    assert_built(&copy);

    // The copy has the names, so it can drop its own index and constraint.
    copy.execute("DROP INDEX file_size").unwrap();
    copy.execute("DROP CONSTRAINT file_id").unwrap();
    assert!(!copy.has_property_index("size"));
    copy.execute("INSERT (:File {id: 'f0'})").unwrap();
    copy.execute("INSERT (:File {id: 'f1'})").unwrap();

    // None of that reaches the source.
    assert_built(&db);
    assert!(
        db.find_nodes_by_property("id", &Value::from("f1"))
            .is_empty(),
        "expected no nodes"
    );
}

#[cfg(feature = "grafeo-file")]
#[test]
fn reopening_a_file_keeps_indexes_and_constraints() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("db.grafeo");
    let db = GrafeoDB::open(&path).unwrap();
    build(&db);
    db.close().unwrap();

    let db = GrafeoDB::open(&path).unwrap();
    assert_built(&db);
    assert_built(&db.to_memory().unwrap());
    db.execute("DROP INDEX file_size").unwrap();
    db.close().unwrap();

    let db = GrafeoDB::open(&path).unwrap();
    assert!(
        !db.has_property_index("size"),
        "a dropped index stays dropped"
    );
    assert!(db.has_property_index("id"));
    db.close().unwrap();
}

/// 200 `Graph:File` nodes `n0`..`n199` in a chain of `T` edges, with a
/// property index on `id`, written to a `.grafeo` file and reopened.
#[cfg(feature = "grafeo-file")]
fn reopened_chain(dir: &tempfile::TempDir) -> GrafeoDB {
    let path = dir.path().join("chain.grafeo");
    let db = GrafeoDB::open(&path).unwrap();
    db.execute("UNWIND range(0, 199) AS i INSERT (:Graph:File {id: 'n' + toString(i), i: i})")
        .unwrap();
    db.execute("MATCH (a:File), (b:File) WHERE b.i = a.i + 1 INSERT (a)-[:T]->(b)")
        .unwrap();
    db.create_property_index("id");
    db.close().unwrap();
    GrafeoDB::open(&path).unwrap()
}

/// #459: on a reopened file a point lookup plus one hop seeks the property
/// index, as in memory, instead of scanning every node (0.5.43 lost the index
/// on reopen and scanned).
#[cfg(feature = "grafeo-file")]
#[test]
fn a_point_lookup_on_a_reopened_file_seeks_the_index() {
    let dir = tempfile::tempdir().unwrap();
    let db = reopened_chain(&dir);
    assert!(db.has_property_index("id"));
    for query in [
        "MATCH (s {id: 'n10'})-[:T]->(d) RETURN d.id",
        "MATCH (s:File {id: 'n10'})-[:T]->(d) RETURN d.id",
    ] {
        let profile = plan(&db.execute(&format!("PROFILE {query}")).unwrap());
        assert!(profile.contains("NodeList (s.id Eq"), "{query}: {profile}");
        assert_eq!(
            db.execute(query).unwrap().rows(),
            &[vec![Value::from("n11")]],
            "{query}"
        );
    }
    db.close().unwrap();
}

/// The first row's first column of a PROFILE: the plan.
fn plan(result: &grafeo_engine::database::QueryResult) -> String {
    match &result.rows()[0][0] {
        Value::String(plan) => plan.to_string(),
        other => panic!("expected a plan, got {other:?}"),
    }
}

#[cfg(feature = "vector-index")]
mod vector {
    use super::*;
    use grafeo_common::types::NodeId;

    const DOCS: &str = "INSERT (:Doc {id: 1, emb: vector([1.0, 0.0, 0.0])}), \
                        (:Doc {id: 2, emb: vector([0.0, 2.0, 0.0])}), \
                        (:Doc {id: 3, emb: vector([0.0, 0.0, 3.0])})";

    /// A Euclidean index with its own HNSW parameters and a quantized one.
    fn build(db: &GrafeoDB) {
        db.execute(DOCS).unwrap();
        db.execute(&DOCS.replace("Doc", "Note")).unwrap();
        db.create_vector_index(
            "Doc",
            "emb",
            None,
            Some("euclidean"),
            Some(8),
            Some(64),
            None,
        )
        .unwrap();
        db.create_vector_index("Note", "emb", None, None, None, None, Some("scalar"))
            .unwrap();
    }

    fn search(db: &GrafeoDB, label: &str) -> Vec<(NodeId, f32)> {
        db.vector_search(label, "emb", &[2.0, 0.0, 0.0], 3, None, None)
            .unwrap()
    }

    fn assert_same_indexes(copy: &GrafeoDB, source: &GrafeoDB) {
        let docs = search(copy, "Doc");
        assert_eq!(docs, search(source, "Doc"));
        assert_eq!(docs[0].1, 1.0, "the Euclidean metric came back: {docs:?}");
        assert_eq!(search(copy, "Note").len(), 3);
        assert_eq!(search(copy, "Note")[0].0, search(source, "Note")[0].0);
    }

    #[test]
    fn to_memory_keeps_vector_indexes() {
        let db = GrafeoDB::new_in_memory();
        build(&db);
        let copy = db.to_memory().unwrap();
        assert_same_indexes(&copy, &db);

        // The copy's index follows the copy's writes.
        copy.execute("INSERT (:Doc {id: 4, emb: vector([2.0, 0.0, 0.0])})")
            .unwrap();
        assert_eq!(search(&copy, "Doc")[0].1, 0.0);
        assert_eq!(search(&db, "Doc")[0].1, 1.0);
    }

    #[cfg(feature = "grafeo-file")]
    #[test]
    fn reopening_a_file_keeps_vector_indexes() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("db.grafeo");
        let source = GrafeoDB::new_in_memory();
        build(&source);
        let db = GrafeoDB::open(&path).unwrap();
        build(&db);
        db.close().unwrap();

        let db = GrafeoDB::open(&path).unwrap();
        assert_same_indexes(&db, &source);
        db.close().unwrap();
    }

    #[test]
    fn a_named_graph_keeps_its_vector_index() {
        let db = GrafeoDB::new_in_memory();
        db.create_graph("model").unwrap();
        let model = db.graph("model").unwrap();
        model.execute(DOCS).unwrap();
        model
            .execute("CREATE VECTOR INDEX doc_emb ON :Doc(emb)")
            .unwrap();

        let copy = db.to_memory().unwrap();
        let result = copy
            .graph("model")
            .unwrap()
            .execute(
                "PROFILE MATCH (d:Doc) \
                 WHERE cosine_similarity(d.emb, [1.0, 0.1, 0.0]) >= 0.9 RETURN d.id",
            )
            .unwrap();
        assert!(plan(&result).contains("VectorScan"), "{}", plan(&result));
        let found = copy
            .graph("model")
            .unwrap()
            .execute(
                "MATCH (d:Doc) WHERE cosine_similarity(d.emb, [1.0, 0.1, 0.0]) >= 0.9 RETURN d.id",
            )
            .unwrap();
        assert_eq!(found.rows(), [[Value::Int64(1)]]);
    }
}

#[cfg(feature = "text-index")]
mod text {
    use super::*;

    const ARTICLES: &str = "INSERT (:Article {id: 1, body: 'graph database engine'}), \
                            (:Article {id: 2, body: 'python web framework'})";
    const SEARCH: &str = "MATCH (a:Article) WHERE text_score(a.body, 'graph') > 0.0 RETURN a.id";

    fn assert_text_index(db: &GrafeoDB) {
        let found = db.text_search("Article", "body", "graph", 10).unwrap();
        assert_eq!(found.len(), 1);
        let result = db.execute(&format!("PROFILE {SEARCH}")).unwrap();
        assert!(plan(&result).contains("TextScan"), "{}", plan(&result));
    }

    #[test]
    fn to_memory_keeps_text_indexes() {
        let db = GrafeoDB::new_in_memory();
        db.execute(ARTICLES).unwrap();
        db.create_text_index("Article", "body").unwrap();
        let copy = db.to_memory().unwrap();
        assert_text_index(&copy);
        copy.execute("INSERT (:Article {id: 3, body: 'graph theory'})")
            .unwrap();
        assert_eq!(
            copy.text_search("Article", "body", "graph", 10)
                .unwrap()
                .len(),
            2
        );
        assert_eq!(
            db.text_search("Article", "body", "graph", 10)
                .unwrap()
                .len(),
            1
        );
    }

    #[cfg(feature = "grafeo-file")]
    #[test]
    fn reopening_a_file_keeps_text_indexes() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("db.grafeo");
        let db = GrafeoDB::open(&path).unwrap();
        db.execute(ARTICLES).unwrap();
        db.create_text_index("Article", "body").unwrap();
        db.close().unwrap();

        let db = GrafeoDB::open(&path).unwrap();
        assert_text_index(&db);
        db.close().unwrap();
    }

    #[test]
    fn a_named_graph_keeps_its_text_index() {
        let db = GrafeoDB::new_in_memory();
        db.create_graph("model").unwrap();
        let model = db.graph("model").unwrap();
        model.execute(ARTICLES).unwrap();
        model
            .execute("CREATE INDEX article_body FOR (a:Article) ON (a.body) USING TEXT")
            .unwrap();

        let copy = db.to_memory().unwrap();
        let model = copy.graph("model").unwrap();
        let result = model.execute(&format!("PROFILE {SEARCH}")).unwrap();
        assert!(plan(&result).contains("TextScan"), "{}", plan(&result));
        assert_eq!(model.execute(SEARCH).unwrap().rows(), [[Value::Int64(1)]]);
    }
}

/// A compacted database keeps its data in a compact base under an overlay:
/// the copy has both, and the indexes over them.
#[cfg(feature = "compact-store")]
#[test]
fn to_memory_copies_a_compacted_database() {
    let mut db = GrafeoDB::new_in_memory();
    db.execute("INSERT (:File {id: 'f0'}), (:File {id: 'f1'})")
        .unwrap();
    db.compact().unwrap();
    db.execute("INSERT (:File {id: 'f2'})").unwrap();
    db.create_property_index("id");

    let copy = db.to_memory().unwrap();
    let ids = copy
        .execute("MATCH (f:File) RETURN f.id ORDER BY f.id")
        .unwrap();
    assert_eq!(
        ids.rows(),
        [
            [Value::from("f0")],
            [Value::from("f1")],
            [Value::from("f2")]
        ]
    );
    assert!(copy.has_property_index("id"));
}
