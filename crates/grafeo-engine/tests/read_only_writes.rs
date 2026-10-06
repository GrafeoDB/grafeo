//! Nothing writes through a read-only transaction, a read-only role or a
//! read-only database: no statement in any query language, and no call of
//! the direct API (#413).
//!
//! ```bash
//! cargo test -p grafeo-engine --all-features --test read_only_writes
//! ```

use grafeo_common::types::{NodeId, PropertyKey, Value};
use grafeo_common::utils::error::Result;
use grafeo_engine::GrafeoDB;
use grafeo_engine::auth::Role;
use grafeo_engine::session::Session;
use std::collections::HashMap;

/// A database holding one `:Person` Alix with `w: 1`.
fn seeded() -> (GrafeoDB, NodeId) {
    let db = GrafeoDB::new_in_memory();
    let alix = db
        .create_node_with_props(
            &["Person"],
            [("name", Value::from("Alix")), ("w", Value::Int64(1))],
        )
        .unwrap();
    (db, alix)
}

/// Node count and every `:Person`'s name and `w`: what a rejected write
/// must leave as it was.
fn state(db: &GrafeoDB) -> (usize, Vec<Vec<Value>>) {
    let people = db
        .session()
        .execute("MATCH (p:Person) RETURN p.name, p.w ORDER BY p.name")
        .unwrap();
    (db.node_count(), people.rows().to_vec())
}

fn seeded_state() -> (usize, Vec<Vec<Value>>) {
    (1, vec![vec![Value::from("Alix"), Value::Int64(1)]])
}

fn assert_read_only<T: std::fmt::Debug>(result: Result<T>, what: &str) {
    let err = result.expect_err(what).to_string();
    assert!(err.contains("read-only"), "{what}: {err}");
}

fn assert_denied<T: std::fmt::Debug>(result: Result<T>, what: &str) {
    let err = result.expect_err(what).to_string();
    assert!(err.contains("permission denied"), "{what}: {err}");
}

/// A session inside `START TRANSACTION READ ONLY`.
fn read_only_transaction(db: &GrafeoDB) -> Session {
    let session = db.session();
    session.execute("START TRANSACTION READ ONLY").unwrap();
    session
}

/// Ends the read-only transaction and checks that the session writes again.
fn ends_and_writes_again(db: &GrafeoDB, session: &Session) {
    session.execute("ROLLBACK").unwrap();
    session.execute("INSERT (:Person {name: 'Gus'})").unwrap();
    assert_eq!(db.node_count(), 2);
}

#[cfg(feature = "cypher")]
#[test]
fn cypher_writes_fail_in_a_read_only_transaction() {
    let (db, _) = seeded();
    let session = read_only_transaction(&db);

    assert_read_only(
        session.execute_cypher("MATCH (a:Person) SET a.w = 5"),
        "SET",
    );
    assert_read_only(
        session.execute_cypher("CREATE (:Person {name: 'Gus'})"),
        "CREATE",
    );
    assert_read_only(
        session.execute_cypher("MERGE (:Person {name: 'Vincent'})"),
        "MERGE",
    );
    assert_read_only(
        session.execute_cypher("MATCH (a:Person) DETACH DELETE a"),
        "DELETE",
    );
    assert_read_only(
        session.execute_cypher_with_params(
            "MATCH (a:Person) SET a.w = $w",
            HashMap::from([("w".to_string(), Value::Int64(5))]),
        ),
        "SET with a parameter",
    );
    // Reads still work.
    let rows = session
        .execute_cypher("MATCH (a:Person) RETURN a.w")
        .unwrap();
    assert_eq!(rows.rows()[0][0], Value::Int64(1));
    session.execute("COMMIT").unwrap();

    assert_eq!(state(&db), seeded_state());
    let session = read_only_transaction(&db);
    ends_and_writes_again(&db, &session);
}

#[cfg(feature = "gremlin")]
#[test]
fn gremlin_writes_fail_in_a_read_only_transaction() {
    let (db, _) = seeded();
    let session = read_only_transaction(&db);

    assert_read_only(
        session.execute_gremlin("g.addV('Person').property('name', 'Gus')"),
        "addV",
    );
    assert_read_only(
        session.execute_gremlin("g.V().hasLabel('Person').property('w', 5)"),
        "property",
    );
    session.execute("COMMIT").unwrap();

    assert_eq!(state(&db), seeded_state());
}

#[cfg(feature = "graphql")]
#[test]
fn graphql_mutations_fail_in_a_read_only_transaction() {
    let (db, _) = seeded();
    let session = read_only_transaction(&db);

    assert_read_only(
        session.execute_graphql(r#"mutation { createPerson(name: "Gus", w: 2) { name } }"#),
        "mutation",
    );
    session.execute("COMMIT").unwrap();

    assert_eq!(state(&db), seeded_state());
}

#[cfg(all(feature = "sparql", feature = "triple-store"))]
#[test]
fn sparql_updates_fail_in_a_read_only_transaction() {
    let db = GrafeoDB::new_in_memory();
    let triples = |db: &GrafeoDB| {
        db.session()
            .execute_sparql("SELECT ?s WHERE { ?s ?p ?o }")
            .unwrap()
            .rows()
            .len()
    };
    db.session()
        .execute_sparql(r#"INSERT DATA { <http://ex.org/alix> <http://ex.org/name> "Alix" . }"#)
        .unwrap();
    let session = read_only_transaction(&db);

    assert_read_only(
        session
            .execute_sparql(r#"INSERT DATA { <http://ex.org/gus> <http://ex.org/name> "Gus" . }"#),
        "INSERT DATA",
    );
    assert_read_only(
        session.execute_sparql("DELETE WHERE { ?s ?p ?o }"),
        "DELETE WHERE",
    );
    session.execute("COMMIT").unwrap();

    assert_eq!(triples(&db), 1);
}

#[test]
fn direct_writes_fail_in_a_read_only_transaction() {
    let (db, alix) = seeded();
    let session = read_only_transaction(&db);

    assert_read_only(session.create_node(&["Person"]), "create_node");
    assert_read_only(
        session.set_node_property(alix, "w", Value::Int64(5)),
        "set_node_property",
    );
    assert_read_only(session.delete_node(alix), "delete_node");
    assert_read_only(
        session.batch_create_nodes_with_props(
            "Person",
            vec![HashMap::from([(
                PropertyKey::new("name"),
                Value::from("Gus"),
            )])],
        ),
        "batch_create_nodes_with_props",
    );
    session.execute("COMMIT").unwrap();

    assert_eq!(state(&db), seeded_state());
    let session = read_only_transaction(&db);
    ends_and_writes_again(&db, &session);
}

#[test]
fn a_read_only_role_cannot_write_directly() {
    let (db, alix) = seeded();
    let session = db.session_with_role(Role::ReadOnly);

    assert_denied(session.create_node(&["Person"]), "create_node");
    assert_denied(
        session.set_node_property(alix, "w", Value::Int64(5)),
        "set_node_property",
    );
    assert_denied(session.delete_node(alix), "delete_node");
    assert_denied(
        session.batch_create_nodes_with_props(
            "Person",
            vec![HashMap::from([(
                PropertyKey::new("name"),
                Value::from("Gus"),
            )])],
        ),
        "batch_create_nodes_with_props",
    );

    assert_eq!(state(&db), seeded_state());
}

#[cfg(feature = "grafeo-file")]
#[test]
fn a_read_only_database_rejects_direct_writes() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("people.grafeo");
    let alix = {
        let db = GrafeoDB::open(&path).unwrap();
        let alix = db
            .create_node_with_props(
                &["Person"],
                [("name", Value::from("Alix")), ("w", Value::Int64(1))],
            )
            .unwrap();
        db.close().unwrap();
        alix
    };

    let db = GrafeoDB::open_read_only(&path).unwrap();
    assert_read_only(db.create_node(&["Person"]), "create_node");
    assert_read_only(
        db.set_node_property(alix, "w", Value::Int64(5)),
        "set_node_property",
    );
    assert_read_only(db.delete_node(alix), "delete_node");
    assert_read_only(db.execute("INSERT (:Person {name: 'Gus'})"), "INSERT");

    assert_eq!(state(&db), seeded_state());
}

/// A read-only database at a new path, holding nothing.
#[cfg(all(
    feature = "grafeo-file",
    any(feature = "lpg", feature = "triple-store")
))]
fn empty_read_only_database(dir: &std::path::Path) -> GrafeoDB {
    let path = dir.join("empty.grafeo");
    GrafeoDB::with_config(grafeo_engine::Config::persistent(&path))
        .unwrap()
        .close()
        .unwrap();
    GrafeoDB::open_read_only(&path).unwrap()
}

/// Checks that `result` failed with the typed read-only error.
#[cfg(all(
    feature = "grafeo-file",
    any(feature = "lpg", feature = "triple-store")
))]
#[track_caller]
fn assert_read_only_error<T: std::fmt::Debug>(what: &str, result: Result<T>) {
    use grafeo_common::utils::error::{Error, TransactionError};
    match result {
        Err(Error::Transaction(TransactionError::ReadOnly)) => {}
        other => panic!("{what}: the read-only error, got {other:?}"),
    }
}

/// A read-only database refuses an RDF batch insert before it pulls the
/// caller's iterator, which may parse or compute the triples: a refused call
/// does no work.
#[cfg(all(feature = "grafeo-file", feature = "triple-store"))]
#[test]
fn a_read_only_database_refuses_an_rdf_batch_insert_without_pulling_its_triples() {
    use grafeo_core::graph::rdf::{Term, Triple};

    let dir = tempfile::tempdir().unwrap();
    let db = empty_read_only_database(dir.path());
    let pulled = std::cell::Cell::new(false);
    let triples = std::iter::once_with(|| {
        pulled.set(true);
        Triple::new(
            Term::iri("http://ex.org/alix"),
            Term::iri("http://ex.org/city"),
            Term::literal("Amsterdam"),
        )
    });

    assert_read_only_error("batch_insert_rdf", db.batch_insert_rdf(triples));
    assert!(
        !pulled.get(),
        "the refused batch insert pulled the caller's iterator"
    );
    assert_eq!(db.rdf_store().len(), 0, "no triple was added");
}

/// A read-only database refuses an import before it opens its file or parses
/// its data: a missing file, or malformed data, still gets the read-only
/// error.
#[cfg(all(feature = "grafeo-file", feature = "lpg"))]
#[test]
fn a_read_only_database_refuses_imports_before_reading_their_input() {
    let dir = tempfile::tempdir().unwrap();
    let db = empty_read_only_database(dir.path());
    let missing = dir.path().join("missing.tsv");
    assert!(!missing.exists(), "the input file does not exist");

    assert_read_only_error(
        "import_tsv of a missing file",
        db.import_tsv(&missing, "KNOWS", true),
    );
    assert_read_only_error(
        "import_mmio of a missing file",
        db.import_mmio(dir.path().join("missing.mtx"), "KNOWS"),
    );
    assert_read_only_error(
        "import_tsv_str of malformed data",
        db.import_tsv_str("Alix\tGus\n", "KNOWS", true),
    );
    #[cfg(feature = "triple-store")]
    assert_read_only_error(
        "import_tsv_rdf of a missing file",
        db.import_tsv_rdf(&missing, "http://ex.org/knows", "http://ex.org/"),
    );
    assert_eq!(db.node_count(), 0, "nothing was imported");
}

/// `GrafeoDB::execute_sparql` respects a read-only database, as the session
/// path does: an update fails and changes nothing, a query still runs.
#[cfg(all(feature = "grafeo-file", feature = "sparql", feature = "triple-store"))]
#[test]
fn a_read_only_database_rejects_sparql_updates() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("triples.grafeo");
    {
        let db = GrafeoDB::open(&path).unwrap();
        db.execute_sparql(r#"INSERT DATA { <http://ex.org/alix> <http://ex.org/name> "Alix" . }"#)
            .unwrap();
        db.close().unwrap();
    }

    let db = GrafeoDB::open_read_only(&path).unwrap();
    assert_read_only(
        db.execute_sparql(r#"INSERT DATA { <http://ex.org/gus> <http://ex.org/name> "Gus" . }"#),
        "GrafeoDB::execute_sparql INSERT DATA",
    );
    assert_read_only(
        db.execute_language(
            r#"INSERT DATA { <http://ex.org/gus> <http://ex.org/name> "Gus" . }"#,
            "sparql",
            None,
        ),
        "execute_language sparql INSERT DATA",
    );
    assert_eq!(db.rdf_store().len(), 1, "no triple was added");
    assert_eq!(
        db.execute_sparql("SELECT ?s WHERE { ?s ?p ?o }")
            .unwrap()
            .rows()
            .len(),
        1,
        "queries still run"
    );
}
