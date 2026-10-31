//! A vector index created over numeric lists converts them (#587): a numeric
//! list written before the index existed is an embedding the index must
//! find. `create_vector_index` and `CREATE VECTOR INDEX` turn every numeric
//! list of `:Label(property)` into a VECTOR, all of them or none, before the
//! index is installed; one of another size, or a list with a non-number,
//! fails the creation and changes nothing. Lists elsewhere are left alone,
//! and a reopen builds the index from vectors only, as before.

#![cfg(all(feature = "vector-index", feature = "gql"))]

use std::collections::HashMap;

use grafeo_common::types::{NodeId, Value};
use grafeo_common::utils::error::ErrorCode;
use grafeo_engine::GrafeoDB;

/// Documents written before any index: `emb` as a GQL list literal
/// (decimals), a list of integers, a list parameter of floats and a VECTOR.
fn documents(db: &GrafeoDB) {
    db.execute("INSERT (:Doc {name: 'Alix', emb: [1.0, 0.0, 0.0]})")
        .unwrap();
    db.execute("INSERT (:Doc {name: 'Gus', emb: [0, 1, 0]})")
        .unwrap();
    let mut params = HashMap::new();
    params.insert(
        "emb".to_string(),
        Value::List(
            vec![
                Value::Float64(1.0),
                Value::Float64(1.0),
                Value::Float64(0.0),
            ]
            .into(),
        ),
    );
    db.execute_with_params("INSERT (:Doc {name: 'Vincent', emb: $emb})", params)
        .unwrap();
    db.execute("INSERT (:Doc {name: 'Jules', emb: vector([0.0, 0.0, 1.0])})")
        .unwrap();
}

/// The values the index must leave alone: a list on another label, a list
/// on another property and a string on the indexed one.
fn bystanders(db: &GrafeoDB) {
    db.execute("INSERT (:Draft {name: 'Mia', emb: [1.0, 0.0, 0.0]})")
        .unwrap();
    db.execute("INSERT (:Doc {name: 'Butch', tags: [1.0, 0.0, 0.0]})")
        .unwrap();
    db.execute("INSERT (:Doc {name: 'Django', emb: 'not an embedding'})")
        .unwrap();
}

const EXPECTED: [(&str, [f32; 3]); 4] = [
    ("Alix", [1.0, 0.0, 0.0]),
    ("Gus", [0.0, 1.0, 0.0]),
    ("Vincent", [1.0, 1.0, 0.0]),
    ("Jules", [0.0, 0.0, 1.0]),
];

fn node(db: &GrafeoDB, name: &str) -> NodeId {
    let mut params = HashMap::new();
    params.insert("name".to_string(), Value::from(name));
    let id = db
        .execute_with_params("MATCH (n {name: $name}) RETURN id(n)", params)
        .unwrap()
        .scalar::<i64>()
        .unwrap();
    NodeId::new(u64::try_from(id).unwrap())
}

fn property(db: &GrafeoDB, name: &str, key: &str) -> Option<Value> {
    db.get_node(node(db, name))
        .unwrap()
        .unwrap()
        .get_property(key)
        .cloned()
}

/// Asserts that every document of [`documents`] holds its embedding as a
/// VECTOR the index finds at distance 0.
fn converted(db: &GrafeoDB, what: &str) {
    for (name, expected) in EXPECTED {
        match property(db, name, "emb") {
            Some(Value::Vector(vector)) => assert_eq!(&*vector, &expected, "{what}: {name}"),
            other => panic!("{what}: {name} holds {other:?}, not a vector"),
        }
        let id = node(db, name);
        let hits = db
            .vector_search("Doc", "emb", &expected, 10, None, None)
            .unwrap();
        let hit = hits
            .iter()
            .find(|(hit, _)| *hit == id)
            .unwrap_or_else(|| panic!("{what}: the index does not find {name}: {hits:?}"));
        assert!(hit.1.abs() < 1e-6, "{what}: {name} at {}", hit.1);
    }
}

/// Asserts that the lists of [`documents`] are lists still.
fn not_converted(db: &GrafeoDB, what: &str) {
    for name in ["Alix", "Gus", "Vincent"] {
        assert!(
            matches!(property(db, name, "emb"), Some(Value::List(_))),
            "{what}: {name} holds {:?}",
            property(db, name, "emb")
        );
    }
}

/// Asserts that the values of [`bystanders`] are as written.
fn bystanders_unchanged(db: &GrafeoDB, what: &str) {
    assert!(
        matches!(property(db, "Mia", "emb"), Some(Value::List(_))),
        "{what}: another label"
    );
    assert!(
        matches!(property(db, "Butch", "tags"), Some(Value::List(_))),
        "{what}: another property"
    );
    assert_eq!(
        property(db, "Django", "emb"),
        Some(Value::from("not an embedding")),
        "{what}: a string"
    );
}

/// Asserts that there is no vector index on `:Doc(emb)`.
fn no_index(db: &GrafeoDB, what: &str) {
    let err = db
        .vector_search("Doc", "emb", &[1.0, 0.0, 0.0], 1, None, None)
        .expect_err(&format!("{what}: no index after a failed creation"));
    assert!(
        err.to_string()
            .contains("there is no vector index on :Doc(emb)"),
        "{what}: {err}"
    );
}

#[test]
fn create_vector_index_converts_the_lists_of_its_property() {
    for dimensions in [Some(3), None] {
        let what = format!("create_vector_index, dimensions {dimensions:?}");
        let db = GrafeoDB::new_in_memory();
        documents(&db);
        bystanders(&db);
        db.create_vector_index("Doc", "emb", dimensions, Some("cosine"), None, None, None)
            .unwrap_or_else(|e| panic!("{what}: {e}"));
        converted(&db, &what);
        bystanders_unchanged(&db, &what);
        assert!(
            !db.execute("MATCH (d:Doc {name: 'Alix'}) RETURN d.emb IS TYPED LIST")
                .unwrap()
                .scalar::<bool>()
                .unwrap(),
            "{what}: a query reads the vector"
        );
    }
}

#[test]
fn create_vector_index_statement_converts_the_lists_of_its_property() {
    for statement in [
        "CREATE VECTOR INDEX doc_emb ON :Doc(emb) DIMENSION 3 METRIC 'cosine'",
        "CREATE VECTOR INDEX doc_emb ON :Doc(emb)",
    ] {
        let db = GrafeoDB::new_in_memory();
        documents(&db);
        bystanders(&db);
        db.execute(statement)
            .unwrap_or_else(|e| panic!("{statement}: {e}"));
        converted(&db, statement);
        bystanders_unchanged(&db, statement);
    }
}

/// With only lists, and no dimensions given, the dimensions come from the
/// first numeric list.
#[test]
fn the_dimensions_come_from_a_list_when_no_vector_is_there() {
    let db = GrafeoDB::new_in_memory();
    db.execute("INSERT (:Doc {name: 'Alix', emb: [1.0, 0.0]})")
        .unwrap();
    db.execute("INSERT (:Doc {name: 'Gus', emb: [0, 1]})")
        .unwrap();
    db.create_vector_index("Doc", "emb", None, None, None, None, None)
        .unwrap();
    let hits = db
        .vector_search("Doc", "emb", &[0.0, 1.0], 1, None, None)
        .unwrap();
    assert_eq!(hits.len(), 1);
    assert_eq!(hits[0].0, node(&db, "Gus"));
    let err = db
        .vector_search("Doc", "emb", &[0.0, 1.0, 0.0], 1, None, None)
        .unwrap_err();
    assert!(err.to_string().contains("expects 2"), "{err}");
}

/// A list of another size fails the creation as a vector of that size does:
/// nothing is converted and there is no index.
#[test]
fn a_list_of_another_size_fails_the_creation_and_converts_nothing() {
    for create in ["api", "statement"] {
        let db = GrafeoDB::new_in_memory();
        documents(&db);
        bystanders(&db);
        db.execute("INSERT (:Doc {name: 'Hans', emb: [1.0, 0.0]})")
            .unwrap();
        let hans = node(&db, "Hans").as_u64();
        let result = if create == "api" {
            db.create_vector_index("Doc", "emb", Some(3), None, None, None, None)
        } else {
            db.execute("CREATE VECTOR INDEX doc_emb ON :Doc(emb) DIMENSION 3")
                .map(|_| ())
        };
        let err = result.expect_err(create);
        assert!(
            err.to_string().contains(&format!(
                "Vector dimension mismatch: expected 3, found 2 on node {hans}"
            )),
            "{create}: {err}"
        );
        not_converted(&db, create);
        assert!(matches!(property(&db, "Hans", "emb"), Some(Value::List(_))));
        bystanders_unchanged(&db, create);
        no_index(&db, create);
    }
}

/// A list with an element that is not a number fails the creation, naming
/// the node and the position: nothing is converted and there is no index.
#[test]
fn a_list_with_a_non_number_fails_the_creation_and_converts_nothing() {
    for create in ["api", "statement"] {
        let db = GrafeoDB::new_in_memory();
        documents(&db);
        db.execute("INSERT (:Doc {name: 'Hans', emb: [1.0, 'x', 0.0]})")
            .unwrap();
        let hans = node(&db, "Hans").as_u64();
        let result = if create == "api" {
            db.create_vector_index("Doc", "emb", Some(3), None, None, None, None)
        } else {
            db.execute("CREATE VECTOR INDEX doc_emb ON :Doc(emb) DIMENSION 3")
                .map(|_| ())
        };
        let err = result.expect_err(create);
        assert!(
            err.to_string().contains(&format!(
                "the list of node {hans} has STRING at position 1: a vector index takes a list of numbers"
            )),
            "{create}: {err}"
        );
        not_converted(&db, create);
        no_index(&db, create);
    }
}

/// With history kept, a read at an epoch before the creation sees the list
/// as it was; one after it, the vector.
#[test]
fn a_read_before_the_creation_sees_the_list() {
    use grafeo_engine::{Config, HistoryRetention};

    let db = GrafeoDB::with_config(
        Config::in_memory().with_history_retention(HistoryRetention::Forever),
    )
    .unwrap();
    documents(&db);
    let before = db.current_epoch();
    db.create_vector_index("Doc", "emb", Some(3), None, None, None, None)
        .unwrap();
    let query = "MATCH (d:Doc {name: 'Alix'}) RETURN d.emb IS TYPED LIST";
    assert!(
        db.execute_at_epoch(query, before)
            .unwrap()
            .scalar::<bool>()
            .unwrap(),
        "the epoch before the creation"
    );
    assert!(!db.execute(query).unwrap().scalar::<bool>().unwrap());
    converted(&db, "after the creation");
}

/// An open transaction that wrote one of the nodes: the creation fails with
/// the retryable write conflict and converts nothing; once the transaction
/// is done, it succeeds.
#[test]
fn an_open_transaction_on_a_node_makes_the_creation_conflict() {
    let db = GrafeoDB::new_in_memory();
    documents(&db);
    let mut session = db.session();
    session.begin_transaction().unwrap();
    session
        .execute("MATCH (d:Doc {name: 'Gus'}) SET d.seen = true")
        .unwrap();
    let err = db
        .create_vector_index("Doc", "emb", Some(3), None, None, None, None)
        .expect_err("the open transaction wrote a node the creation converts");
    assert_eq!(err.error_code(), ErrorCode::TransactionConflict, "{err}");
    not_converted(&db, "a conflicting creation");
    no_index(&db, "a conflicting creation");
    session.commit().unwrap();
    db.create_vector_index("Doc", "emb", Some(3), None, None, None, None)
        .unwrap();
    converted(&db, "after the transaction");
}

/// In a named graph: the lists written through a graph handle are converted
/// by `CREATE VECTOR INDEX` after `USE GRAPH`; the default graph's lists of
/// the same label and property stay lists.
#[test]
fn the_creation_converts_the_lists_of_a_named_graph() {
    let db = GrafeoDB::new_in_memory();
    db.execute("INSERT (:Doc {name: 'Alix', emb: [1.0, 0.0, 0.0]})")
        .unwrap();
    assert!(db.create_graph("archive").unwrap());
    let archive = db.graph("archive").unwrap();
    archive
        .create_node_with_props(
            &["Doc"],
            [
                ("name", Value::from("Gus")),
                (
                    "emb",
                    Value::List(vec![Value::Int64(0), Value::Int64(1), Value::Int64(0)].into()),
                ),
            ],
        )
        .unwrap();
    let session = db.session();
    session.execute("USE GRAPH archive").unwrap();
    session
        .execute("CREATE VECTOR INDEX doc_emb ON :Doc(emb) DIMENSION 3")
        .unwrap();
    assert!(
        !session
            .execute("MATCH (d:Doc {name: 'Gus'}) RETURN d.emb IS TYPED LIST")
            .unwrap()
            .scalar::<bool>()
            .unwrap(),
        "the named graph's list is a vector"
    );
    // The search procedure does not reach a named graph's index (an older
    // gap, apart from this): the conversion is what this test checks.
    assert_eq!(
        session
            .execute(
                "MATCH (d:Doc) WHERE cosine_similarity(d.emb, [0.0, 1.0, 0.0]) > 0.99 RETURN d.name"
            )
            .unwrap()
            .rows(),
        &[vec![Value::from("Gus")]],
        "the named graph's vector measures"
    );
    assert!(
        db.execute("MATCH (d:Doc {name: 'Alix'}) RETURN d.emb IS TYPED LIST")
            .unwrap()
            .scalar::<bool>()
            .unwrap(),
        "the default graph is another graph"
    );
}

/// The creation is durable: after a reopen, from the WAL after a crash and
/// from a checkpoint, the values are vectors and the index finds them. A
/// reopen builds the index from vectors only: a list that reached the
/// property around the checks (a raw store write) is not converted by it.
#[cfg(feature = "wal")]
mod persistent {
    use super::*;
    use grafeo_common::testing::child_process;
    use std::path::{Path, PathBuf};
    use std::process::Command;

    const CHILD_VAR: &str = "GRAFEO_VECTOR_INDEX_OVER_LISTS_CHILD";
    const PATH_VAR: &str = "GRAFEO_VECTOR_INDEX_OVER_LISTS_PATH";
    const EXITED: i32 = 19;

    fn sidecar_wal(path: &Path) -> PathBuf {
        let mut sidecar = path.as_os_str().to_owned();
        sidecar.push(".wal");
        PathBuf::from(sidecar)
    }

    fn crash(which: &str, path: &Path) {
        let output = child_process::output(
            Command::new(std::env::current_exe().unwrap())
                .args(["--exact", "persistent::index_child", "--nocapture"])
                .env(CHILD_VAR, which)
                .env(PATH_VAR, path),
        )
        .unwrap();
        assert_eq!(
            output.status.code(),
            Some(EXITED),
            "the child {which} exited early:\n{}",
            String::from_utf8_lossy(&output.stderr)
        );
        assert!(
            sidecar_wal(path).exists(),
            "{which}: a crash leaves the WAL"
        );
    }

    /// The child processes: write the documents, create the index (and
    /// checkpoint, for "checkpoint"), exit without `close()`.
    #[test]
    fn index_child() {
        let (Ok(which), Some(path)) = (std::env::var(CHILD_VAR), std::env::var_os(PATH_VAR)) else {
            return;
        };
        let db = GrafeoDB::open(PathBuf::from(path)).unwrap();
        documents(&db);
        bystanders(&db);
        db.create_vector_index("Doc", "emb", Some(3), None, None, None, None)
            .unwrap();
        if which == "checkpoint" {
            db.wal_checkpoint().unwrap();
        }
        converted(&db, &which);
        std::process::exit(EXITED);
    }

    #[test]
    fn the_conversion_survives_a_crash() {
        for which in ["wal", "checkpoint"] {
            let dir = tempfile::tempdir().unwrap();
            let path = dir.path().join("graph.grafeo");
            crash(which, &path);
            for reopen in ["after the crash", "after a clean close"] {
                let db = GrafeoDB::open(&path).unwrap();
                let what = format!("{which}, {reopen}");
                converted(&db, &what);
                bystanders_unchanged(&db, &what);
                db.close().unwrap();
            }
        }
    }

    #[test]
    fn a_reopen_builds_the_index_from_vectors_only() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("graph.grafeo");
        let db = GrafeoDB::open(&path).unwrap();
        documents(&db);
        db.create_vector_index("Doc", "emb", Some(3), None, None, None, None)
            .unwrap();
        // Around every check: the raw store takes any value.
        let store = db.store().unwrap();
        let raw = store.create_node(&["Doc"]);
        store.set_node_property(raw, "name", Value::from("Raw"));
        store.set_node_property(
            raw,
            "emb",
            Value::List(
                vec![
                    Value::Float64(0.5),
                    Value::Float64(0.5),
                    Value::Float64(0.0),
                ]
                .into(),
            ),
        );
        db.close().unwrap();

        let db = GrafeoDB::open(&path).unwrap();
        converted(&db, "after the reopen");
        assert!(
            matches!(property(&db, "Raw", "emb"), Some(Value::List(_))),
            "the reopen converts nothing: {:?}",
            property(&db, "Raw", "emb")
        );
        let hits = db
            .vector_search("Doc", "emb", &[0.5, 0.5, 0.0], 10, None, None)
            .unwrap();
        assert!(
            hits.iter().all(|(id, _)| *id != raw),
            "the raw list is not indexed: {hits:?}"
        );
        db.close().unwrap();
    }
}

/// Inside an open transaction of the same session the statement is refused
/// when it would convert lists: an index applies at once and survives a
/// rollback, the conversion would not. With only vectors it works, as
/// before. Another session's transaction on a node gives the conflict.
#[test]
fn the_statement_inside_a_transaction_refuses_to_convert() {
    let statement = "CREATE VECTOR INDEX doc_emb ON :Doc(emb) DIMENSION 3";
    let db = GrafeoDB::new_in_memory();
    documents(&db);
    let mut session = db.session();
    session.begin_transaction().unwrap();
    let err = session
        .execute(statement)
        .expect_err("a conversion inside a transaction");
    assert_eq!(
        err.error_code(),
        ErrorCode::TransactionInvalidState,
        "{err}"
    );
    assert!(
        err.to_string()
            .contains("converts 3 lists to vectors, which cannot run inside a transaction"),
        "{err}"
    );
    session.rollback().unwrap();
    not_converted(&db, "refused inside a transaction");
    no_index(&db, "refused inside a transaction");

    // Another session's open transaction on one of the nodes.
    let mut other = db.session();
    other.begin_transaction().unwrap();
    other
        .execute("MATCH (d:Doc {name: 'Gus'}) SET d.seen = true")
        .unwrap();
    let err = db
        .session()
        .execute(statement)
        .expect_err("another session holds a node");
    assert_eq!(err.error_code(), ErrorCode::TransactionConflict, "{err}");
    not_converted(&db, "a conflicting statement");
    no_index(&db, "a conflicting statement");
    other.rollback().unwrap();

    db.session().execute(statement).unwrap();
    converted(&db, "outside a transaction");

    // Only vectors: the statement works inside a transaction.
    let db = GrafeoDB::new_in_memory();
    db.execute("INSERT (:Doc {name: 'Jules', emb: vector([0.0, 0.0, 1.0])})")
        .unwrap();
    let mut session = db.session();
    session.begin_transaction().unwrap();
    session.execute(statement).unwrap();
    session.commit().unwrap();
    let hits = db
        .vector_search("Doc", "emb", &[0.0, 0.0, 1.0], 1, None, None)
        .unwrap();
    assert_eq!(hits.len(), 1);
}
