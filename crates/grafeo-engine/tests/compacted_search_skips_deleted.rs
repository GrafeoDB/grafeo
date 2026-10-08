//! The searches of a compacted database leave out the compacted nodes
//! deleted since `compact()`.
//!
//! After `compact()` the text and vector indexes (made before it and carried
//! over, or made after it) hold the compacted nodes, and a delete of one
//! leaves its entries in them until the next merge of the overlay drops
//! them. Every search leaves a node out once its delete is committed:
//! `text_search`, `hybrid_search`, `vector_search` (with and without
//! filters), `batch_vector_search`, `mmr_search`, and the text and vector
//! searches of a query, also after a close and reopen. A top-k search still
//! returns k nodes when a deleted one would have been among them, and a
//! delete that is not committed yet hides nothing from the other readers.
//!
//! ```bash
//! cargo test -p grafeo-engine --all-features --test compacted_search_skips_deleted
//! ```

#![cfg(all(
    feature = "compact-store",
    feature = "lpg",
    feature = "gql",
    feature = "text-index",
    feature = "vector-index",
    feature = "hybrid-search"
))]

use std::collections::HashMap;

use grafeo_common::types::{NodeId, Value};
use grafeo_engine::GrafeoDB;

/// The people and notes compacted. Gus has the most canals in his bio, so a
/// text search for canals ranks him first, and his embedding is the nearest
/// to (88, 3); the Amsterdam note is the nearest to the origin. Gus and the
/// Amsterdam note lie apart from the others, so no one is reached only
/// through them in the HNSW graphs, which a delete does not repair (#600).
const DATA: &str = "INSERT (:Person {name: 'Alix', bio: 'canals of Amsterdam', \
                    embedding: vector([3.0, 3.0])}), \
                    (:Person {name: 'Gus', bio: 'canals canals canals of Prague', \
                    embedding: vector([88.0, 3.0])}), \
                    (:Person {name: 'Mia', bio: 'museums of Paris', \
                    embedding: vector([19.0, 19.0])}), \
                    (:Note {name: 'Amsterdam', kind: 'city', embedding: vector([3.0, 3.0])}), \
                    (:Note {name: 'Prague', kind: 'city', embedding: vector([19.0, 19.0])}), \
                    (:Note {name: 'Paris', kind: 'city', embedding: vector([88.0, 88.0])})";

/// Deletes Gus and the Amsterdam note in one committed statement.
const DELETE: &str = "MATCH (g:Person {name: 'Gus'}), (a:Note {name: 'Amsterdam'}) \
                      DETACH DELETE g, a";

/// A text index on `:Person(bio)`, a Euclidean vector index on
/// `:Person(embedding)` and a scalar-quantized Euclidean one on
/// `:Note(embedding)`, which keeps its own copy of the vectors.
fn index(db: &GrafeoDB) {
    db.create_text_index("Person", "bio").unwrap();
    db.create_vector_index(
        "Person",
        "embedding",
        Some(2),
        Some("euclidean"),
        None,
        None,
        None,
    )
    .unwrap();
    db.create_vector_index(
        "Note",
        "embedding",
        Some(2),
        Some("euclidean"),
        None,
        None,
        Some("scalar"),
    )
    .unwrap();
}

/// [`DATA`], indexed before `compact()`: the indexes are carried over.
fn indexed_then_compacted() -> GrafeoDB {
    let mut db = GrafeoDB::new_in_memory();
    db.execute(DATA).unwrap();
    index(&db);
    db.compact().unwrap();
    db
}

/// [`DATA`], compacted, then indexed: the indexes are built over the base.
fn compacted_then_indexed() -> GrafeoDB {
    let mut db = GrafeoDB::new_in_memory();
    db.execute(DATA).unwrap();
    db.compact().unwrap();
    index(&db);
    db
}

/// The name of node `id`, or a description of what `id` is when it has
/// none (a deleted node has none).
fn name_of(db: &GrafeoDB, id: NodeId) -> String {
    match db
        .get_node(id)
        .and_then(|node| node.get_property("name").cloned())
    {
        Some(Value::String(name)) => name.to_string(),
        other => format!("node {id:?} without a name ({other:?})"),
    }
}

/// The names of `hits`, in their order.
fn names<S>(db: &GrafeoDB, hits: Vec<(NodeId, S)>) -> Vec<String> {
    hits.into_iter().map(|(id, _)| name_of(db, id)).collect()
}

/// The names of `hits`, sorted.
fn sorted_names<S>(db: &GrafeoDB, hits: Vec<(NodeId, S)>) -> Vec<String> {
    let mut names = names(db, hits);
    names.sort();
    names
}

/// The first column of each row of `query`, as text.
fn column(db: &GrafeoDB, query: &str) -> Vec<String> {
    db.execute(query)
        .unwrap_or_else(|error| panic!("{query}: {error}"))
        .rows()
        .iter()
        .map(|row| match &row[0] {
            Value::String(name) => name.to_string(),
            other => format!("{other:?}"),
        })
        .collect()
}

/// The checks of one stage: every one runs, and the stage fails with the
/// list of those that did not hold.
struct Checks {
    stage: String,
    failed: Vec<String>,
}

impl Checks {
    fn new(stage: &str) -> Self {
        Self {
            stage: stage.to_string(),
            failed: Vec::new(),
        }
    }

    /// Notes a failure when `found` is not `expected`.
    fn names<E: AsRef<str>>(&mut self, what: &str, found: Vec<String>, expected: &[E]) {
        let expected: Vec<&str> = expected.iter().map(AsRef::as_ref).collect();
        if found != expected {
            self.failed
                .push(format!("{what}: found {found:?}, expected {expected:?}"));
        }
    }

    /// Panics with every failure noted.
    fn finish(self) {
        assert!(
            self.failed.is_empty(),
            "{}: {} checks failed:\n{}",
            self.stage,
            self.failed.len(),
            self.failed.join("\n")
        );
    }
}

/// Every search of `db`, in which Gus and the Amsterdam note are deleted,
/// finds only the nodes still there, and as many as it is asked for.
fn assert_searches_skip_the_deleted(db: &GrafeoDB, stage: &str) {
    let mut checks = Checks::new(stage);

    // Text: Gus ranks first for canals, so the top 1 needs a second look.
    let text =
        |query: &str, k: usize| names(db, db.text_search("Person", "bio", query, k).unwrap());
    checks.names("text_search", text("canals", 19), &["Alix"]);
    checks.names("text_search for the top 1", text("canals", 1), &["Alix"]);
    checks.names(
        "text_search for a word only Gus has",
        text("Prague", 19),
        &[] as &[&str],
    );

    // Hybrid: the text source alone, and with the vector source, whose
    // nearest is Gus.
    let hybrid = |vector: Option<&[f32]>, k: usize| {
        sorted_names(
            db,
            db.hybrid_search("Person", "bio", "embedding", "canals", vector, k, None)
                .unwrap(),
        )
    };
    checks.names("hybrid_search, text only", hybrid(None, 1), &["Alix"]);
    checks.names(
        "hybrid_search, text and vector",
        hybrid(Some(&[88.0, 3.0]), 19),
        &["Alix", "Mia"],
    );
    checks.names(
        "hybrid_search, the quantized vector index only",
        names(
            db,
            db.hybrid_search("Note", "name", "embedding", "", Some(&[0.0, 0.0]), 1, None)
                .unwrap(),
        ),
        &["Prague"],
    );

    // Vector, full precision and quantized: the deleted node is the nearest
    // to the query.
    for (label, query, nearest, both) in [
        ("Person", [88.0, 3.0], "Mia", ["Alix", "Mia"]),
        ("Note", [0.0, 0.0], "Prague", ["Paris", "Prague"]),
    ] {
        let search = |ef: Option<usize>| {
            names(
                db,
                db.vector_search(label, "embedding", &query, 1, ef, None)
                    .unwrap(),
            )
        };
        checks.names(
            &format!("vector_search of :{label}"),
            search(None),
            &[nearest],
        );
        checks.names(
            &format!("vector_search of :{label} with a beam width"),
            search(Some(88)),
            &[nearest],
        );
        let batch = db
            .batch_vector_search(
                label,
                "embedding",
                &[query.to_vec(), query.to_vec()],
                1,
                None,
                None,
            )
            .unwrap();
        for (number, hits) in batch.into_iter().enumerate() {
            checks.names(
                &format!("batch_vector_search of :{label}, query {number}"),
                names(db, hits),
                &[nearest],
            );
        }
        let diverse = |k: usize, candidates: Option<usize>| {
            sorted_names(
                db,
                db.mmr_search(
                    label,
                    "embedding",
                    &query,
                    k,
                    candidates,
                    Some(1.0),
                    None,
                    None,
                )
                .unwrap(),
            )
        };
        checks.names(
            &format!("mmr_search of :{label}"),
            diverse(1, None),
            &[nearest],
        );
        checks.names(
            &format!("mmr_search of :{label} with as many candidates as hits"),
            diverse(2, Some(2)),
            &both,
        );
    }
    let cities = HashMap::from([("kind".to_string(), Value::from("city"))]);
    checks.names(
        "vector_search with a filter",
        names(
            db,
            db.vector_search("Note", "embedding", &[0.0, 0.0], 1, None, Some(&cities))
                .unwrap(),
        ),
        &["Prague"],
    );

    // Queries: the text and vector scans of the indexes.
    checks.names(
        "a query with text_score",
        column(
            db,
            "MATCH (p:Person) WHERE text_score(p.bio, 'canals') > 0.0 \
             RETURN p.name AS name ORDER BY name",
        ),
        &["Alix"],
    );
    checks.names(
        "a query for the top text_score",
        column(
            db,
            "MATCH (p:Person) RETURN p.name \
             ORDER BY text_score(p.bio, 'canals') DESC LIMIT 1",
        ),
        &["Alix"],
    );
    checks.names(
        "a query for the nearest note",
        column(
            db,
            "MATCH (n:Note) RETURN n.name \
             ORDER BY euclidean_distance(n.embedding, [0.0, 0.0]) ASC LIMIT 1",
        ),
        &["Prague"],
    );
    checks.names(
        "a query for the notes within a distance",
        column(
            db,
            "MATCH (n:Note) WHERE euclidean_distance(n.embedding, [0.0, 0.0]) <= 188.0 \
             RETURN n.name AS name ORDER BY name",
        ),
        &["Paris", "Prague"],
    );

    // The search procedures, whose rows start with the node id.
    for (procedure, expected) in [
        (
            "CALL grafeo.search.text('Person', 'bio', 'canals', 1)",
            &["Alix"][..],
        ),
        (
            "CALL grafeo.search.vector('Note', 'embedding', [0.0, 0.0], 1)",
            &["Prague"][..],
        ),
        (
            "CALL grafeo.search.mmr('Note', 'embedding', [0.0, 0.0], 2, 2, 1.0)",
            &["Paris", "Prague"][..],
        ),
    ] {
        checks.names(procedure, called(db, procedure), expected);
    }
    checks.finish();
}

/// The names of the nodes whose ids are the first column of the rows of
/// `procedure`, sorted.
fn called(db: &GrafeoDB, procedure: &str) -> Vec<String> {
    let rows = db
        .execute(procedure)
        .unwrap_or_else(|error| panic!("{procedure}: {error}"));
    let mut names: Vec<String> = rows
        .rows()
        .iter()
        .map(|row| match &row[0] {
            Value::Int64(id) => name_of(db, NodeId::new(id.cast_unsigned())),
            other => format!("{other:?}"),
        })
        .collect();
    names.sort();
    names
}

/// The indexes made before `compact()` are carried over with the compacted
/// nodes in them: a delete after it leaves every search.
#[test]
fn searches_skip_compacted_nodes_deleted_after_compact() {
    let db = indexed_then_compacted();
    db.execute(DELETE).unwrap();
    assert_searches_skip_the_deleted(&db, "indexes carried over");
}

/// The indexes made after `compact()` are built over the compacted nodes: a
/// delete outside a transaction leaves every search.
#[test]
fn searches_of_indexes_made_after_compact_skip_deleted_compacted_nodes() {
    let db = compacted_then_indexed();
    for (label, name) in [("Person", "Gus"), ("Note", "Amsterdam")] {
        let id = db
            .find_nodes_by_property("name", &Value::from(name))
            .into_iter()
            .find(|&id| db.get_node(id).is_some_and(|node| node.has_label(label)))
            .unwrap_or_else(|| panic!("no :{label} named {name}"));
        assert!(db.delete_node(id).unwrap(), "{name} is deleted");
    }
    assert_searches_skip_the_deleted(&db, "indexes made after compact");
}

/// A delete that is not committed yet hides nothing from the other readers;
/// its rollback leaves the node found, and a committed delete hides it.
#[test]
fn an_open_delete_hides_nothing_until_it_commits() {
    let db = indexed_then_compacted();
    let canals =
        |db: &GrafeoDB| sorted_names(db, db.text_search("Person", "bio", "canals", 19).unwrap());
    let nearest_note = |db: &GrafeoDB| {
        names(
            db,
            db.vector_search("Note", "embedding", &[0.0, 0.0], 1, None, None)
                .unwrap(),
        )
    };

    let mut session = db.session();
    session.begin_transaction().unwrap();
    session.execute(DELETE).unwrap();
    assert_eq!(canals(&db), ["Alix", "Gus"], "an open delete hides nothing");
    assert_eq!(
        nearest_note(&db),
        ["Amsterdam"],
        "an open delete hides nothing"
    );
    session.rollback().unwrap();
    assert_eq!(
        canals(&db),
        ["Alix", "Gus"],
        "a rolled back delete hides nothing"
    );
    assert_eq!(
        nearest_note(&db),
        ["Amsterdam"],
        "a rolled back delete hides nothing"
    );

    session.begin_transaction().unwrap();
    session.execute(DELETE).unwrap();
    session.commit().unwrap();
    assert_eq!(canals(&db), ["Alix"], "a committed delete hides Gus");
    assert_eq!(
        nearest_note(&db),
        ["Prague"],
        "a committed delete hides the note"
    );
}

/// After a merge of the overlay the deleted nodes are gone from the base
/// and from the indexes, and the searches find what they found before it.
#[test]
fn searches_find_the_same_after_a_merge() {
    let mut db = indexed_then_compacted();
    db.execute(DELETE).unwrap();
    db.recompact().unwrap();
    assert_searches_skip_the_deleted(&db, "after recompact");
}

/// A close and reopen restores the indexes (with the compacted nodes) and
/// the log of the deleted ones: the searches skip them.
#[cfg(all(feature = "wal", feature = "grafeo-file"))]
#[test]
fn searches_skip_deleted_compacted_nodes_after_a_reopen() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("people.grafeo");
    let mut db = GrafeoDB::open(&path).unwrap();
    db.execute(DATA).unwrap();
    index(&db);
    db.compact().unwrap();
    db.execute(DELETE).unwrap();
    db.close().unwrap();

    let db = GrafeoDB::open(&path).unwrap();
    assert_searches_skip_the_deleted(&db, "after a reopen");
    db.close().unwrap();
}
