//! Direct calls follow the selected graph (#677).
//!
//! After `set_current_graph` (or `USE GRAPH` in a session), reads and writes
//! through queries act on the selected graph. The direct calls must too:
//! creating and dropping a vector or text index, the vector, text and hybrid
//! searches, the edge list imports and the `CALL grafeo.search.*`
//! procedures. Each test has a control on the default graph, without a
//! selection, that passes today, so a failure is the selected graph being
//! ignored.
//!
//! The fixture puts different documents in the two graphs: Amsterdam notes in
//! the default graph and Berlin notes in `archive`, with node ids that do not
//! overlap, so every result tells which graph it came from.
//!
//! ```bash
//! cargo test -p grafeo-engine --features full --test selected_graph_direct_calls
//! ```

#![cfg(feature = "gql")]

use std::collections::BTreeSet;

use grafeo_common::types::{NodeId, Value};
use grafeo_engine::GrafeoDB;

const ARCHIVE: &str = "archive";

fn vec3(x: f32, y: f32, z: f32) -> Value {
    Value::Vector(vec![x, y, z].into())
}

/// A database with Amsterdam documents in the default graph and Berlin
/// documents in `archive`, both `:Doc` with a 3-dimensional `emb` and a
/// `body` text. No indexes.
struct Docs {
    db: GrafeoDB,
    default_docs: BTreeSet<NodeId>,
    archive_docs: BTreeSet<NodeId>,
}

fn docs() -> Docs {
    let db = GrafeoDB::new_in_memory();
    let mut default_docs = BTreeSet::new();
    for (emb, body) in [
        (vec3(1.0, 0.0, 0.0), "Amsterdam canal notes by Alix"),
        (vec3(0.9, 0.1, 0.0), "Amsterdam museum notes by Gus"),
    ] {
        let id = db
            .create_node_with_props(&["Doc"], [("emb", emb), ("body", Value::from(body))])
            .expect("a default document");
        default_docs.insert(id);
    }

    db.create_graph(ARCHIVE).expect("create archive");
    let archive = db.graph(ARCHIVE).expect("archive handle");
    // Two shelves first, so the archive's document ids are not the default
    // graph's, whether each graph counts ids on its own or not.
    for _ in 0..2 {
        archive
            .create_node_with_props(&["Shelf"], [("city", Value::from("Paris"))])
            .expect("a shelf");
    }
    let mut archive_docs = BTreeSet::new();
    for (emb, body) in [
        (vec3(0.0, 0.0, 1.0), "Berlin wall notes by Vincent"),
        (vec3(0.0, 0.1, 0.9), "Berlin park notes by Jules"),
        (vec3(0.1, 0.0, 0.9), "Berlin station notes by Mia"),
    ] {
        let id = archive
            .create_node_with_props(&["Doc"], [("emb", emb), ("body", Value::from(body))])
            .expect("an archive document");
        archive_docs.insert(id);
    }
    assert!(
        default_docs.is_disjoint(&archive_docs),
        "fixture: the two graphs' document ids overlap: {default_docs:?} and {archive_docs:?}"
    );
    Docs {
        db,
        default_docs,
        archive_docs,
    }
}

impl Docs {
    /// Indexes `emb` in the default graph, without a selection.
    #[cfg(feature = "vector-index")]
    fn default_vector_index(&self) {
        self.db
            .create_vector_index("Doc", "emb", Some(3), Some("cosine"), None, None, None)
            .expect("default vector index");
    }

    /// Indexes `body` in the default graph, without a selection.
    #[cfg(feature = "text-index")]
    fn default_text_index(&self) {
        self.db
            .create_text_index("Doc", "body")
            .expect("default text index");
    }

    /// Indexes `emb` in `archive` through a query on its handle.
    #[cfg(feature = "vector-index")]
    fn archive_vector_index(&self) {
        self.db
            .graph(ARCHIVE)
            .unwrap()
            .execute("CREATE VECTOR INDEX doc_emb ON :Doc(emb) DIMENSION 3 METRIC 'cosine'")
            .expect("archive vector index");
    }

    /// Indexes `body` in `archive` through a query on its handle.
    #[cfg(feature = "text-index")]
    fn archive_text_index(&self) {
        self.db
            .graph(ARCHIVE)
            .unwrap()
            .execute("CREATE INDEX doc_body FOR (n:Doc) ON (n.body) USING TEXT")
            .expect("archive text index");
    }

    fn select_archive(&self) {
        self.db
            .set_current_graph(Some(ARCHIVE))
            .expect("select archive");
    }

    fn select_default(&self) {
        self.db.set_current_graph(None).expect("select default");
    }
}

#[cfg(any(feature = "vector-index", feature = "text-index"))]
fn ids<S>(results: &[(NodeId, S)]) -> BTreeSet<NodeId> {
    results.iter().map(|(id, _)| *id).collect()
}

/// The node ids in column 0 of a `CALL grafeo.search.*` result.
#[cfg(any(feature = "vector-index", feature = "text-index"))]
fn called_ids(result: &grafeo_engine::database::QueryResult) -> BTreeSet<NodeId> {
    result
        .rows()
        .iter()
        .map(|row| match row[0] {
            Value::Int64(id) => NodeId::new(id.cast_unsigned()),
            ref other => panic!("node_id should be an integer, got {other:?}"),
        })
        .collect()
}

/// Asserts that a search found only `expected` documents, at least one,
/// naming the graph each wrong id belongs to.
#[cfg(any(feature = "vector-index", feature = "text-index"))]
fn assert_found_in(found: &BTreeSet<NodeId>, expected: &BTreeSet<NodeId>, docs: &Docs, what: &str) {
    let default_hits: Vec<_> = found.intersection(&docs.default_docs).collect();
    assert!(
        default_hits.is_empty(),
        "{what}: found the default graph's documents {default_hits:?} (all found: {found:?})"
    );
    assert!(
        !found.is_empty(),
        "{what}: found nothing; expected some of {expected:?}"
    );
    assert!(
        found.is_subset(expected),
        "{what}: found {found:?}, expected only ids among {expected:?}"
    );
}

/// The number of nodes of `label` in `graph`. `None` reads through a new
/// session, which starts in the database's current graph: select the
/// default graph first.
fn count_label(db: &GrafeoDB, graph: Option<&str>, label: &str) -> i64 {
    let query = format!("MATCH (n:{label}) RETURN count(n)");
    let result = match graph {
        Some(name) => db.graph(name).unwrap().execute(&query),
        None => db.session().execute(&query),
    }
    .expect("count query");
    match result.rows()[0][0] {
        Value::Int64(count) => count,
        ref other => panic!("count should be an integer, got {other:?}"),
    }
}

/// The number of edges of `edge_type` in `graph`, as [`count_label`].
fn count_edges(db: &GrafeoDB, graph: Option<&str>, edge_type: &str) -> i64 {
    let query = format!("MATCH ()-[e:{edge_type}]->() RETURN count(e)");
    let result = match graph {
        Some(name) => db.graph(name).unwrap().execute(&query),
        None => db.session().execute(&query),
    }
    .expect("edge count query");
    match result.rows()[0][0] {
        Value::Int64(count) => count,
        ref other => panic!("count should be an integer, got {other:?}"),
    }
}

// ============================================================================
// Vector index and search
// ============================================================================

#[cfg(feature = "vector-index")]
mod vector {
    use super::*;

    /// Control: on the default graph, without a selection, the index and
    /// every vector search work and find the default graph's documents.
    #[test]
    fn control_vector_calls_on_the_default_graph() {
        let docs = docs();
        docs.default_vector_index();
        let db = &docs.db;
        let query = [1.0, 0.0, 0.0];

        let found = ids(&db
            .vector_search("Doc", "emb", &query, 5, None, None)
            .unwrap());
        assert_eq!(found, docs.default_docs);

        let batch = db
            .batch_vector_search("Doc", "emb", &[query.to_vec()], 5, None, None)
            .unwrap();
        assert_eq!(ids(&batch[0]), docs.default_docs);

        let mmr = db
            .mmr_search("Doc", "emb", &query, 2, None, None, None, None)
            .unwrap();
        assert_eq!(ids(&mmr), docs.default_docs);

        assert!(db.drop_vector_index("Doc", "emb").unwrap());
        let err = db
            .vector_search("Doc", "emb", &query, 5, None, None)
            .expect_err("dropped");
        assert!(
            err.to_string().contains("there is no vector index"),
            "{err}"
        );
    }

    /// `create_vector_index` after selecting `archive` indexes archive's
    /// documents; the default graph gets no index.
    #[test]
    fn create_vector_index_indexes_the_selected_graph() {
        let docs = docs();
        docs.select_archive();
        docs.db
            .create_vector_index("Doc", "emb", Some(3), Some("cosine"), None, None, None)
            .expect("create in archive");

        let found = ids(&docs
            .db
            .vector_search("Doc", "emb", &[0.0, 0.0, 1.0], 5, None, None)
            .expect("search in archive"));
        assert_found_in(
            &found,
            &docs.archive_docs,
            &docs,
            "vector_search after create_vector_index with archive selected",
        );

        docs.select_default();
        let result = docs
            .db
            .vector_search("Doc", "emb", &[1.0, 0.0, 0.0], 5, None, None);
        match result {
            Err(err) => assert!(
                err.to_string().contains("there is no vector index"),
                "unexpected error: {err}"
            ),
            Ok(found) => panic!(
                "create_vector_index with archive selected indexed the default graph: \
                 a search there found {:?}",
                ids(&found)
            ),
        }
    }

    /// The archive has an index (made by a query in it), the default graph
    /// has none: `vector_search` with archive selected uses archive's.
    #[test]
    fn vector_search_uses_the_selected_graphs_index() {
        let docs = docs();
        docs.archive_vector_index();
        docs.select_archive();
        let found = docs
            .db
            .vector_search("Doc", "emb", &[0.0, 0.0, 1.0], 5, None, None)
            .map(|found| ids(&found));
        match found {
            Ok(found) => assert_found_in(
                &found,
                &docs.archive_docs,
                &docs,
                "vector_search with archive selected",
            ),
            Err(err) => panic!("vector_search with archive selected: {err}"),
        }
    }

    /// Both graphs have an index: `vector_search` with archive selected
    /// finds archive's documents, not the default graph's.
    #[test]
    fn vector_search_finds_the_selected_graphs_documents() {
        let docs = docs();
        docs.default_vector_index();
        docs.archive_vector_index();
        docs.select_archive();
        let found = ids(&docs
            .db
            .vector_search("Doc", "emb", &[0.0, 0.0, 1.0], 5, None, None)
            .expect("search"));
        assert_found_in(
            &found,
            &docs.archive_docs,
            &docs,
            "vector_search with archive selected",
        );
    }

    #[test]
    fn batch_vector_search_finds_the_selected_graphs_documents() {
        let docs = docs();
        docs.default_vector_index();
        docs.archive_vector_index();
        docs.select_archive();
        let batch = docs
            .db
            .batch_vector_search(
                "Doc",
                "emb",
                &[vec![0.0, 0.0, 1.0], vec![0.0, 1.0, 0.0]],
                5,
                None,
                None,
            )
            .expect("batch search");
        assert_eq!(batch.len(), 2);
        for (i, results) in batch.iter().enumerate() {
            assert_found_in(
                &ids(results),
                &docs.archive_docs,
                &docs,
                &format!("batch_vector_search query {i} with archive selected"),
            );
        }
    }

    #[test]
    fn mmr_search_finds_the_selected_graphs_documents() {
        let docs = docs();
        docs.default_vector_index();
        docs.archive_vector_index();
        docs.select_archive();
        let found = ids(&docs
            .db
            .mmr_search("Doc", "emb", &[0.0, 0.0, 1.0], 2, None, None, None, None)
            .expect("mmr search"));
        assert_found_in(
            &found,
            &docs.archive_docs,
            &docs,
            "mmr_search with archive selected",
        );
    }

    /// `drop_vector_index` with archive selected drops archive's index and
    /// leaves the default graph's.
    #[test]
    fn drop_vector_index_drops_the_selected_graphs_index() {
        let docs = docs();
        docs.default_vector_index();
        docs.archive_vector_index();
        docs.select_archive();
        assert!(
            docs.db.drop_vector_index("Doc", "emb").expect("drop"),
            "drop_vector_index with archive selected found no index"
        );

        docs.select_default();
        let found = docs
            .db
            .vector_search("Doc", "emb", &[1.0, 0.0, 0.0], 5, None, None)
            .map(|found| ids(&found));
        match found {
            Ok(found) => assert_eq!(found, docs.default_docs),
            Err(err) => panic!(
                "drop_vector_index with archive selected dropped the default graph's index: {err}"
            ),
        }

        // The archive's index is gone: a CALL in archive finds none.
        let err = docs
            .db
            .graph(ARCHIVE)
            .unwrap()
            .execute("CALL grafeo.search.vector('Doc', 'emb', [0.0, 0.0, 1.0], 5)")
            .expect_err("archive's index should be dropped");
        assert!(
            err.to_string().contains("there is no vector index"),
            "{err}"
        );
    }
}

// ============================================================================
// Text index and search
// ============================================================================

#[cfg(feature = "text-index")]
mod text {
    use super::*;

    #[test]
    fn control_text_calls_on_the_default_graph() {
        let docs = docs();
        docs.default_text_index();
        let db = &docs.db;
        let found = ids(&db.text_search("Doc", "body", "notes", 5, None).unwrap());
        assert_eq!(found, docs.default_docs);

        assert!(db.drop_text_index("Doc", "body").unwrap());
        let err = db
            .text_search("Doc", "body", "notes", 5, None)
            .expect_err("dropped");
        assert!(err.to_string().contains("there is no text index"), "{err}");
    }

    /// `create_text_index` after selecting `archive` indexes archive's
    /// documents; the default graph gets no index.
    #[test]
    fn create_text_index_indexes_the_selected_graph() {
        let docs = docs();
        docs.select_archive();
        docs.db
            .create_text_index("Doc", "body")
            .expect("create in archive");

        let found = docs
            .db
            .text_search("Doc", "body", "notes", 5, None)
            .map(|found| ids(&found));
        match found {
            Ok(found) => assert_found_in(
                &found,
                &docs.archive_docs,
                &docs,
                "text_search after create_text_index with archive selected",
            ),
            Err(err) => panic!("text_search after create_text_index in archive: {err}"),
        }

        docs.select_default();
        match docs.db.text_search("Doc", "body", "notes", 5, None) {
            Err(err) => assert!(
                err.to_string().contains("there is no text index"),
                "unexpected error: {err}"
            ),
            Ok(found) => panic!(
                "create_text_index with archive selected indexed the default graph: \
                 a search there found {:?}",
                ids(&found)
            ),
        }
    }

    /// The archive has a text index (made by a query in it), the default
    /// graph has none: `text_search` with archive selected uses archive's.
    #[test]
    fn text_search_uses_the_selected_graphs_index() {
        let docs = docs();
        docs.archive_text_index();
        docs.select_archive();
        match docs.db.text_search("Doc", "body", "Berlin", 5, None) {
            Ok(found) => assert_found_in(
                &ids(&found),
                &docs.archive_docs,
                &docs,
                "text_search with archive selected",
            ),
            Err(err) => panic!("text_search with archive selected: {err}"),
        }
    }

    /// Both graphs have a text index: a word in every document finds
    /// archive's documents with archive selected.
    #[test]
    fn text_search_finds_the_selected_graphs_documents() {
        let docs = docs();
        docs.default_text_index();
        docs.archive_text_index();
        docs.select_archive();
        let found = ids(&docs
            .db
            .text_search("Doc", "body", "notes", 5, None)
            .expect("text search"));
        assert_found_in(
            &found,
            &docs.archive_docs,
            &docs,
            "text_search with archive selected",
        );
    }

    #[test]
    fn drop_text_index_drops_the_selected_graphs_index() {
        let docs = docs();
        docs.default_text_index();
        docs.archive_text_index();
        docs.select_archive();
        assert!(
            docs.db.drop_text_index("Doc", "body").expect("drop"),
            "drop_text_index with archive selected found no index"
        );

        docs.select_default();
        match docs.db.text_search("Doc", "body", "notes", 5, None) {
            Ok(found) => assert_eq!(ids(&found), docs.default_docs),
            Err(err) => panic!(
                "drop_text_index with archive selected dropped the default graph's index: {err}"
            ),
        }

        let err = docs
            .db
            .graph(ARCHIVE)
            .unwrap()
            .execute("CALL grafeo.search.text('Doc', 'body', 'notes', 5)")
            .expect_err("archive's text index should be dropped");
        assert!(err.to_string().contains("there is no text index"), "{err}");
    }
}

// ============================================================================
// Hybrid search
// ============================================================================

#[cfg(feature = "hybrid-search")]
mod hybrid {
    use super::*;

    #[test]
    fn control_hybrid_search_on_the_default_graph() {
        let docs = docs();
        docs.default_vector_index();
        docs.default_text_index();
        let found = docs
            .db
            .hybrid_search(
                "Doc",
                "body",
                "emb",
                "notes",
                Some(&[1.0, 0.0, 0.0]),
                5,
                None,
                None,
            )
            .unwrap();
        assert_eq!(ids(&found), docs.default_docs);
    }

    /// Both graphs have both indexes: hybrid search with archive selected
    /// fuses archive's results.
    #[test]
    fn hybrid_search_finds_the_selected_graphs_documents() {
        let docs = docs();
        docs.default_vector_index();
        docs.default_text_index();
        docs.archive_vector_index();
        docs.archive_text_index();
        docs.select_archive();
        let found = docs
            .db
            .hybrid_search(
                "Doc",
                "body",
                "emb",
                "notes",
                Some(&[0.0, 0.0, 1.0]),
                5,
                None,
                None,
            )
            .expect("hybrid search");
        assert_found_in(
            &ids(&found),
            &docs.archive_docs,
            &docs,
            "hybrid_search with archive selected",
        );
    }

    /// Only archive has the indexes: hybrid search (which skips a missing
    /// index) with archive selected must not come back empty.
    #[test]
    fn hybrid_search_uses_the_selected_graphs_indexes() {
        let docs = docs();
        docs.archive_vector_index();
        docs.archive_text_index();
        docs.select_archive();
        let found = docs
            .db
            .hybrid_search(
                "Doc",
                "body",
                "emb",
                "Berlin",
                Some(&[0.0, 0.0, 1.0]),
                5,
                None,
                None,
            )
            .expect("hybrid search");
        assert_found_in(
            &ids(&found),
            &docs.archive_docs,
            &docs,
            "hybrid_search with archive selected",
        );
    }
}

// ============================================================================
// CALL grafeo.search.* in a session, a one-shot execute and a graph handle
// ============================================================================

#[cfg(feature = "vector-index")]
mod procedures {
    use super::*;

    const CALL_VECTOR: &str =
        "CALL grafeo.search.vector('Doc', 'emb', [0.0, 0.0, 1.0], 5) YIELD node_id, distance";
    const CALL_MMR: &str = "CALL grafeo.search.mmr('Doc', 'emb', [0.0, 0.0, 1.0], 2) YIELD node_id";

    /// Control: the procedures in the default graph find its documents.
    #[test]
    fn control_call_search_in_the_default_graph() {
        let docs = docs();
        docs.default_vector_index();
        let session = docs.db.session();
        let found = called_ids(
            &session
                .execute("CALL grafeo.search.vector('Doc', 'emb', [1.0, 0.0, 0.0], 5)")
                .unwrap(),
        );
        assert_eq!(found, docs.default_docs);
        let found = called_ids(
            &session
                .execute("CALL grafeo.search.mmr('Doc', 'emb', [1.0, 0.0, 0.0], 2)")
                .unwrap(),
        );
        assert_eq!(found, docs.default_docs);
    }

    /// The reported case: `USE GRAPH archive`, an index made there, then
    /// `CALL grafeo.search.vector` said there is no vector index.
    #[test]
    fn session_call_search_vector_after_use_graph() {
        let docs = docs();
        let session = docs.db.session();
        session.execute("USE GRAPH archive").expect("use archive");
        session
            .execute("CREATE VECTOR INDEX doc_emb ON :Doc(emb) DIMENSION 3 METRIC 'cosine'")
            .expect("index in archive");
        match session.execute(CALL_VECTOR) {
            Ok(result) => assert_found_in(
                &called_ids(&result),
                &docs.archive_docs,
                &docs,
                "CALL grafeo.search.vector after USE GRAPH archive",
            ),
            Err(err) => panic!("CALL grafeo.search.vector after USE GRAPH archive: {err}"),
        }
    }

    /// Both graphs have an index: the session's CALL finds archive's
    /// documents after `USE GRAPH archive`.
    #[test]
    fn session_call_search_vector_finds_the_used_graphs_documents() {
        let docs = docs();
        docs.default_vector_index();
        docs.archive_vector_index();
        let session = docs.db.session();
        session.execute("USE GRAPH archive").expect("use archive");
        let result = session.execute(CALL_VECTOR).expect("call");
        assert_found_in(
            &called_ids(&result),
            &docs.archive_docs,
            &docs,
            "CALL grafeo.search.vector after USE GRAPH archive",
        );
    }

    #[test]
    fn session_call_search_mmr_after_use_graph() {
        let docs = docs();
        docs.archive_vector_index();
        let session = docs.db.session();
        session.execute("USE GRAPH archive").expect("use archive");
        match session.execute(CALL_MMR) {
            Ok(result) => assert_found_in(
                &called_ids(&result),
                &docs.archive_docs,
                &docs,
                "CALL grafeo.search.mmr after USE GRAPH archive",
            ),
            Err(err) => panic!("CALL grafeo.search.mmr after USE GRAPH archive: {err}"),
        }
    }

    /// A one-shot `execute` after `set_current_graph` runs in archive.
    #[test]
    fn one_shot_call_search_vector_after_set_current_graph() {
        let docs = docs();
        docs.archive_vector_index();
        docs.select_archive();
        match docs.db.execute(CALL_VECTOR) {
            Ok(result) => assert_found_in(
                &called_ids(&result),
                &docs.archive_docs,
                &docs,
                "db.execute(CALL grafeo.search.vector) with archive selected",
            ),
            Err(err) => {
                panic!("db.execute(CALL grafeo.search.vector) with archive selected: {err}")
            }
        }
    }

    /// A graph handle on archive runs its CALL in archive.
    #[test]
    fn graph_handle_call_search_vector() {
        let docs = docs();
        docs.archive_vector_index();
        let archive = docs.db.graph(ARCHIVE).unwrap();
        match archive.execute(CALL_VECTOR) {
            Ok(result) => assert_found_in(
                &called_ids(&result),
                &docs.archive_docs,
                &docs,
                "archive handle CALL grafeo.search.vector",
            ),
            Err(err) => panic!("archive handle CALL grafeo.search.vector: {err}"),
        }
    }

    #[test]
    fn graph_handle_call_search_mmr() {
        let docs = docs();
        docs.default_vector_index();
        docs.archive_vector_index();
        let archive = docs.db.graph(ARCHIVE).unwrap();
        let result = archive.execute(CALL_MMR).expect("call");
        assert_found_in(
            &called_ids(&result),
            &docs.archive_docs,
            &docs,
            "archive handle CALL grafeo.search.mmr",
        );
    }
}

#[cfg(feature = "text-index")]
mod text_procedures {
    use super::*;

    const CALL_TEXT: &str =
        "CALL grafeo.search.text('Doc', 'body', 'notes', 5) YIELD node_id, score";

    #[test]
    fn control_call_search_text_in_the_default_graph() {
        let docs = docs();
        docs.default_text_index();
        let found = called_ids(&docs.db.session().execute(CALL_TEXT).unwrap());
        assert_eq!(found, docs.default_docs);
    }

    #[test]
    fn session_call_search_text_after_use_graph() {
        let docs = docs();
        docs.archive_text_index();
        let session = docs.db.session();
        session.execute("USE GRAPH archive").expect("use archive");
        match session.execute(CALL_TEXT) {
            Ok(result) => assert_found_in(
                &called_ids(&result),
                &docs.archive_docs,
                &docs,
                "CALL grafeo.search.text after USE GRAPH archive",
            ),
            Err(err) => panic!("CALL grafeo.search.text after USE GRAPH archive: {err}"),
        }
    }

    #[test]
    fn graph_handle_call_search_text() {
        let docs = docs();
        docs.default_text_index();
        docs.archive_text_index();
        let result = docs
            .db
            .graph(ARCHIVE)
            .unwrap()
            .execute(CALL_TEXT)
            .expect("call");
        assert_found_in(
            &called_ids(&result),
            &docs.archive_docs,
            &docs,
            "archive handle CALL grafeo.search.text",
        );
    }
}

// ============================================================================
// Edge list imports
// ============================================================================

mod imports {
    use super::*;

    const EDGES: &str = "# Amsterdam to Berlin to Paris\n1\t2\n2\t3\n";

    /// Control: without a selection the import lands in the default graph.
    #[test]
    fn control_import_tsv_str_into_the_default_graph() {
        let docs = docs();
        let (nodes, edges) = docs.db.import_tsv_str(EDGES, "ROAD", true).expect("import");
        assert_eq!((nodes, edges), (3, 2));
        assert_eq!(count_label(&docs.db, None, "_Imported"), 3);
        assert_eq!(count_edges(&docs.db, None, "ROAD"), 2);
        assert_eq!(count_label(&docs.db, Some(ARCHIVE), "_Imported"), 0);
    }

    fn assert_imported_into_archive(docs: &Docs, what: &str, edges: i64) {
        docs.select_default();
        let default_nodes = count_label(&docs.db, None, "_Imported");
        let default_edges = count_edges(&docs.db, None, "ROAD");
        assert_eq!(
            (default_nodes, default_edges),
            (0, 0),
            "{what} with archive selected changed the default graph"
        );
        assert_eq!(
            (
                count_label(&docs.db, None, "Doc"),
                count_label(&docs.db, Some(ARCHIVE), "Doc")
            ),
            (
                i64::try_from(docs.default_docs.len()).unwrap(),
                i64::try_from(docs.archive_docs.len()).unwrap()
            ),
            "{what}: the documents of the default graph and of archive"
        );
        assert_eq!(
            (
                count_label(&docs.db, Some(ARCHIVE), "_Imported"),
                count_edges(&docs.db, Some(ARCHIVE), "ROAD")
            ),
            (3, edges),
            "{what} with archive selected: archive's imported nodes and edges"
        );
    }

    #[test]
    fn import_tsv_str_imports_into_the_selected_graph() {
        let docs = docs();
        docs.select_archive();
        let (nodes, edges) = docs.db.import_tsv_str(EDGES, "ROAD", true).expect("import");
        assert_eq!((nodes, edges), (3, 2));
        assert_imported_into_archive(&docs, "import_tsv_str", 2);
    }

    #[test]
    fn import_tsv_imports_into_the_selected_graph() {
        let docs = docs();
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("roads.tsv");
        std::fs::write(&path, EDGES).unwrap();
        docs.select_archive();
        let (nodes, edges) = docs.db.import_tsv(&path, "ROAD", false).expect("import");
        assert_eq!((nodes, edges), (3, 4));
        assert_imported_into_archive(&docs, "import_tsv", 4);
    }

    #[test]
    fn import_mmio_imports_into_the_selected_graph() {
        let docs = docs();
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("roads.mtx");
        std::fs::write(
            &path,
            "%%MatrixMarket matrix coordinate pattern general\n% Amsterdam, Berlin, Paris\n3 3 2\n1 2\n2 3\n",
        )
        .unwrap();
        docs.select_archive();
        let (nodes, edges) = docs.db.import_mmio(&path, "ROAD").expect("import");
        assert_eq!((nodes, edges), (3, 2));
        assert_imported_into_archive(&docs, "import_mmio", 2);
    }
}

// ============================================================================
// Rebuilding a vector index, and text indexes with options
// ============================================================================

/// `rebuild_vector_index` rebuilds the selected graph's index: with only
/// `archive` indexed it succeeds there and creates nothing in the default
/// graph; with both indexed, the default graph's index still finds its own
/// documents after `archive`'s was rebuilt.
#[cfg(feature = "vector-index")]
#[test]
fn rebuild_vector_index_rebuilds_the_selected_graphs_index() {
    let docs = docs();
    docs.archive_vector_index();
    docs.select_archive();
    docs.db
        .rebuild_vector_index("Doc", "emb")
        .expect("rebuild the selected graph's index");
    let found = docs
        .db
        .vector_search("Doc", "emb", &[0.0, 0.0, 1.0], 10, None, None)
        .expect("a search of the rebuilt index");
    assert_found_in(&ids(&found), &docs.archive_docs, &docs, "after the rebuild");
    docs.select_default();
    let error = docs
        .db
        .vector_search("Doc", "emb", &[1.0, 0.0, 0.0], 10, None, None)
        .expect_err("the rebuild made no index in the default graph");
    assert!(
        error.to_string().contains("there is no vector index"),
        "{error}"
    );

    // Both graphs indexed, with different metrics: the rebuild keeps the
    // selected graph's own (euclidean), not the default graph's (cosine).
    // [0, 0, 2] is at euclidean distance 1 from archive's [0, 0, 1], and at
    // cosine distance 0.
    let docs = self::docs();
    docs.default_vector_index();
    docs.db
        .graph(ARCHIVE)
        .unwrap()
        .execute("CREATE VECTOR INDEX doc_emb ON :Doc(emb) DIMENSION 3 METRIC 'euclidean'")
        .expect("archive euclidean index");
    docs.select_archive();
    docs.db.rebuild_vector_index("Doc", "emb").unwrap();
    let nearest = docs
        .db
        .vector_search("Doc", "emb", &[0.0, 0.0, 2.0], 1, None, None)
        .unwrap();
    assert_found_in(
        &ids(&nearest),
        &docs.archive_docs,
        &docs,
        "after the rebuild",
    );
    assert!(
        (nearest[0].1 - 1.0).abs() < 1e-4,
        "the rebuilt index keeps archive's euclidean metric: distance {}",
        nearest[0].1
    );
    docs.select_default();
    let found = docs
        .db
        .vector_search("Doc", "emb", &[1.0, 0.0, 0.0], 10, None, None)
        .unwrap();
    assert_eq!(
        ids(&found),
        docs.default_docs,
        "the default graph's index is untouched"
    );
}

/// `create_text_index_with` creates the index, with its options, in the
/// selected graph, and `text_index_options` reads the selected graph's: the
/// two graphs keep different options apart.
#[cfg(feature = "text-index")]
#[test]
fn text_index_options_follow_the_selected_graph() {
    use grafeo_core::index::text::TextIndexOptions;

    let docs = docs();
    let archive_options = TextIndexOptions::new().with_k1(1.5).with_b(0.3);
    docs.select_archive();
    docs.db
        .create_text_index_with("Doc", "body", archive_options.clone())
        .expect("a text index with options in archive");
    assert_eq!(
        docs.db.text_index_options("Doc", "body"),
        Some(archive_options.clone()),
        "archive's options while archive is selected"
    );
    let found = docs
        .db
        .text_search("Doc", "body", "Berlin", 10, None)
        .unwrap();
    assert_found_in(
        &ids(&found),
        &docs.archive_docs,
        &docs,
        "text_search after create_text_index_with",
    );

    docs.select_default();
    assert_eq!(
        docs.db.text_index_options("Doc", "body"),
        None,
        "the default graph has no text index"
    );
    let default_options = TextIndexOptions::new().with_k1(0.9).with_b(0.6);
    docs.db
        .create_text_index_with("Doc", "body", default_options.clone())
        .unwrap();
    assert_eq!(
        docs.db.text_index_options("Doc", "body"),
        Some(default_options),
        "the default graph's own options"
    );
    docs.select_archive();
    assert_eq!(
        docs.db.text_index_options("Doc", "body"),
        Some(archive_options),
        "archive keeps its options"
    );
}
