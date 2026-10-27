//! Rollback releases the claims acquired by the undone writes (#670), while
//! earlier writes, deletes, endpoints and unique values remain protected.

#![cfg(feature = "gql")]

use grafeo_common::types::Value;
use grafeo_common::utils::error::{Error, TransactionError};
use grafeo_engine::GrafeoDB;
use grafeo_engine::session::Session;

fn database() -> GrafeoDB {
    let db = GrafeoDB::new_in_memory();
    db.execute(
        "INSERT (:Item {name: 'keep', v: 0}), (:Item {name: 'free', v: 0}), \
         (:Item {name: 'other', v: 0})",
    )
    .unwrap();
    db
}

fn conflict(result: grafeo_common::utils::error::Result<grafeo_engine::database::QueryResult>) {
    let error = result.expect_err("the earlier claim must still conflict");
    assert!(
        matches!(
            error,
            Error::Transaction(TransactionError::WriteConflict(_))
        ),
        "{error}"
    );
}

fn reverted_write(session: &Session, operation: &str, failed_statement: bool) {
    let query = format!("MATCH (z:Item {{name: 'other'}}) {operation}");
    if failed_statement {
        // The denominator reads a row, so the failure occurs during execution.
        let error = session
            .execute(&format!("{query} RETURN 1 / (z.v - z.v)"))
            .expect_err("the write's result expression divides by zero");
        assert!(error.to_string().contains("division by zero"), "{error}");
    } else {
        session.savepoint("before").unwrap();
        session.execute(&query).unwrap();
        session.rollback_to_savepoint("before").unwrap();
    }
}

fn writes(failed_statement: bool) {
    let db = database();
    let mut first = db.session();
    first.begin_transaction().unwrap();
    first
        .execute("MATCH (n:Item {name: 'keep'}) SET n.v = 1")
        .unwrap();
    reverted_write(
        &first,
        "MATCH (n:Item {name: 'free'}) SET n.v = 2",
        failed_statement,
    );
    assert_eq!(
        first
            .execute("MATCH (n:Item {name: 'free'}) RETURN n.v")
            .unwrap()
            .rows(),
        &[vec![Value::Int64(0)]],
        "storage undo happened before the next writer"
    );

    let mut second = db.session();
    second.begin_transaction().unwrap();
    second
        .execute("MATCH (n:Item {name: 'free'}) SET n.v = 7")
        .expect("the undone write no longer claims this node");
    conflict(second.execute("MATCH (n:Item {name: 'keep'}) SET n.v = 9"));
    second.commit().unwrap();
    first.commit().unwrap();
    assert_eq!(
        db.execute("MATCH (n:Item) RETURN n.name, n.v ORDER BY n.name")
            .unwrap()
            .rows(),
        &[
            vec![Value::from("free"), Value::Int64(7)],
            vec![Value::from("keep"), Value::Int64(1)],
            vec![Value::from("other"), Value::Int64(0)],
        ]
    );
}

#[test]
fn savepoint_releases_new_write_claims() {
    writes(false);
}

#[test]
fn failed_statement_releases_new_write_claims() {
    writes(true);
}

fn deletes(failed_statement: bool) {
    let db = database();
    let mut first = db.session();
    first.begin_transaction().unwrap();
    first
        .execute("MATCH (n:Item {name: 'free'}) SET n.v = 1")
        .unwrap();
    // The earlier write on the same node must survive removal of its later
    // delete claim. An endpoint does not conflict with that earlier write.
    reverted_write(
        &first,
        "MATCH (n:Item {name: 'free'}) DELETE n",
        failed_statement,
    );
    let mut second = db.session();
    second.begin_transaction().unwrap();
    second
        .execute(
            "MATCH (a:Item {name: 'free'}), (b:Item {name: 'other'}) \
             INSERT (a)-[:LINK]->(b)",
        )
        .expect("the undone delete no longer blocks an endpoint claim");
    conflict(second.execute("MATCH (n:Item {name: 'free'}) SET n.v = 9"));
    second.commit().unwrap();
    first.commit().unwrap();
    assert_eq!(
        db.execute("MATCH (a:Item {name: 'free'})-[r:LINK]->() RETURN a.v, count(r)")
            .unwrap()
            .rows(),
        &[vec![Value::Int64(1), Value::Int64(1)]]
    );
}

#[test]
fn savepoint_releases_delete_but_preserves_earlier_write_claims() {
    deletes(false);
}

#[test]
fn failed_statement_releases_delete_but_preserves_earlier_write_claims() {
    deletes(true);
}

fn endpoints(failed_statement: bool) {
    let db = database();
    let mut first = db.session();
    first.begin_transaction().unwrap();
    first
        .execute(
            "MATCH (a:Item {name: 'keep'}), (b:Item {name: 'other'}) \
             INSERT (a)-[:KEPT]->(b)",
        )
        .unwrap();
    reverted_write(
        &first,
        "MATCH (a:Item {name: 'free'}), (b:Item {name: 'other'}) INSERT (a)-[:UNDONE]->(b)",
        failed_statement,
    );
    let mut second = db.session();
    second.begin_transaction().unwrap();
    second
        .execute("MATCH (n:Item {name: 'free'}) DELETE n")
        .expect("the undone edge no longer claims its endpoints");
    conflict(second.execute("MATCH (n:Item {name: 'keep'}) DELETE n"));
    second.commit().unwrap();
    first.commit().unwrap();
    assert_eq!(
        db.execute("MATCH ()-[r]->() RETURN type(r)")
            .unwrap()
            .rows(),
        &[vec![Value::from("KEPT")]]
    );
}

#[test]
fn savepoint_releases_only_new_endpoint_claims() {
    endpoints(false);
}

#[test]
fn failed_statement_releases_only_new_endpoint_claims() {
    endpoints(true);
}

fn keys(failed_statement: bool) {
    let db = database();
    db.execute("CREATE CONSTRAINT memory_key FOR (m:Memory) ON (m.key) UNIQUE")
        .unwrap();
    let mut first = db.session();
    first.begin_transaction().unwrap();
    first.execute("INSERT (:Memory {key: 'keep'})").unwrap();
    reverted_write(&first, "INSERT (:Memory {key: 'free'})", failed_statement);
    let mut second = db.session();
    second.begin_transaction().unwrap();
    second
        .execute("INSERT (:Memory {key: 'free'})")
        .expect("the undone node no longer claims its unique key");
    conflict(second.execute("INSERT (:Memory {key: 'keep'})"));
    second.commit().unwrap();
    first.commit().unwrap();
    assert_eq!(
        db.execute("MATCH (m:Memory) RETURN m.key ORDER BY m.key")
            .unwrap()
            .rows(),
        &[vec![Value::from("free")], vec![Value::from("keep")]]
    );
}

#[test]
fn savepoint_releases_only_new_unique_key_claims() {
    keys(false);
}

#[test]
fn failed_statement_releases_only_new_unique_key_claims() {
    keys(true);
}

#[test]
fn nested_rollbacks_release_only_their_own_claims() {
    let db = database();
    let mut first = db.session();
    first.begin_transaction().unwrap();
    first
        .execute("MATCH (n:Item {name: 'keep'}) SET n.v = 1")
        .unwrap();
    first.begin_transaction().unwrap();
    first
        .execute("MATCH (n:Item {name: 'free'}) SET n.v = 2")
        .unwrap();
    first.begin_transaction().unwrap();
    first
        .execute("MATCH (n:Item {name: 'other'}) SET n.v = 3")
        .unwrap();
    first.rollback().unwrap();
    let mut second = db.session();
    second.begin_transaction().unwrap();
    second
        .execute("MATCH (n:Item {name: 'other'}) SET n.v = 7")
        .unwrap();
    conflict(second.execute("MATCH (n:Item {name: 'free'}) SET n.v = 7"));
    first.rollback().unwrap();
    second
        .execute("MATCH (n:Item {name: 'free'}) SET n.v = 7")
        .unwrap();
    conflict(second.execute("MATCH (n:Item {name: 'keep'}) SET n.v = 7"));
    second.commit().unwrap();
    first.commit().unwrap();
    assert_eq!(
        db.execute("MATCH (n:Item) RETURN n.name, n.v ORDER BY n.name")
            .unwrap()
            .rows(),
        &[
            vec![Value::from("free"), Value::Int64(7)],
            vec![Value::from("keep"), Value::Int64(1)],
            vec![Value::from("other"), Value::Int64(7)]
        ]
    );
}

#[test]
fn savepoint_claims_keep_named_graph_identity() {
    let db = GrafeoDB::new_in_memory();
    for graph in ["left", "right"] {
        db.execute(&format!("CREATE GRAPH {graph}")).unwrap();
        let mut session = db.session();
        session.use_graph(graph);
        session.execute("INSERT (:Item {v: 0})").unwrap();
    }
    let mut first = db.session();
    let mut second = db.session();
    first.use_graph("left");
    second.use_graph("right");
    assert_eq!(
        first.execute("MATCH (n:Item) RETURN id(n)").unwrap().rows(),
        second
            .execute("MATCH (n:Item) RETURN id(n)")
            .unwrap()
            .rows(),
        "the graph-local IDs overlap"
    );
    first.begin_transaction().unwrap();
    first.execute("MATCH (n:Item) SET n.v = 1").unwrap();
    first.savepoint("before").unwrap();
    first.use_graph("right");
    first.execute("MATCH (n:Item) SET n.v = 2").unwrap();
    first.rollback_to_savepoint("before").unwrap();
    second.begin_transaction().unwrap();
    second.execute("MATCH (n:Item) SET n.v = 7").unwrap();
    second.use_graph("left");
    conflict(second.execute("MATCH (n:Item) SET n.v = 7"));
    second.commit().unwrap();
    first.commit().unwrap();
    assert_eq!(
        first.execute("MATCH (n:Item) RETURN n.v").unwrap().rows(),
        &[vec![Value::Int64(7)]]
    );
    first.use_graph("left");
    assert_eq!(
        first.execute("MATCH (n:Item) RETURN n.v").unwrap().rows(),
        &[vec![Value::Int64(1)]]
    );
}
