//! A `QueryProcessor` in a transaction writes as that transaction: it claims
//! what it writes in the graph its store holds, as a session writing that
//! graph does, so the two conflict on the same node and never on the same id
//! in another graph; and its writes are recorded in the transaction's change
//! set, which the transaction manager's commit stamps and its abort undoes.

use std::sync::Arc;

use grafeo_common::types::{EpochId, TransactionId, Value};
use grafeo_core::graph::lpg::LpgStore;

use crate::GrafeoDB;
use crate::query::{QueryLanguage, QueryProcessor};

/// A processor on `store`, the graph with storage key `graph`, writing as
/// `transaction` of `db`'s transaction manager, which reads at `epoch`.
fn processor(
    db: &GrafeoDB,
    store: Arc<LpgStore>,
    graph: Option<&str>,
    (epoch, transaction): (EpochId, TransactionId),
) -> QueryProcessor {
    let processor =
        QueryProcessor::for_lpg_with_transaction(store, Arc::clone(&db.transaction_manager))
            .with_transaction_context(epoch, transaction);
    match graph {
        Some(graph) => processor.with_graph(graph),
        None => processor,
    }
}

/// A transaction of `db`'s manager, begun now, with the epoch it reads at.
fn begin(db: &GrafeoDB) -> (EpochId, TransactionId) {
    let transaction = db.transaction_manager.begin();
    (db.transaction_manager.current_epoch(), transaction)
}

fn is_conflict(error: &grafeo_common::utils::error::Error) -> bool {
    error.to_string().to_lowercase().contains("conflict")
}

/// Paris in the graph `trips` and Amsterdam in the default graph: both are
/// node 0 of their graph.
fn two_graphs() -> GrafeoDB {
    let db = GrafeoDB::new_in_memory();
    db.execute("CREATE GRAPH trips").unwrap();
    let session = db.session();
    session.use_graph("trips");
    session
        .execute("INSERT (:City {name: 'Paris', visits: 3})")
        .unwrap();
    db.execute("INSERT (:City {name: 'Amsterdam', visits: 19})")
        .unwrap();
    db
}

/// The `visits` of the one city of the graph `session` reads.
fn visits(session: &crate::Session) -> Value {
    session
        .execute("MATCH (c:City) RETURN c.visits")
        .unwrap()
        .rows()[0][0]
        .clone()
}

#[test]
fn a_processor_conflicts_with_a_session_on_a_node_of_the_graph_it_writes() {
    let db = two_graphs();
    let trips = db.lpg_store().graph("trips").unwrap();
    let mut session = db.session();
    session.use_graph("trips");
    session.begin_transaction().unwrap();
    session.execute("MATCH (c:City) SET c.visits = 88").unwrap();

    let other = begin(&db);
    let error = processor(&db, trips, Some("trips"), other)
        .process("MATCH (c:City) SET c.visits = 4", QueryLanguage::Gql, None)
        .unwrap_err();
    assert!(
        is_conflict(&error),
        "the session wrote Paris first: {error}"
    );
    db.transaction_manager.abort(other.1).unwrap();
    session.commit().unwrap();
    assert_eq!(visits(&session), Value::Int64(88));
}

#[test]
fn a_processor_does_not_conflict_with_the_same_id_in_another_graph() {
    let db = two_graphs();
    let trips = db.lpg_store().graph("trips").unwrap();
    // The session writes Amsterdam, node 0 of the default graph.
    let mut session = db.session();
    session.begin_transaction().unwrap();
    session.execute("MATCH (c:City) SET c.visits = 20").unwrap();

    // The processor writes Paris, node 0 of `trips`, and commits.
    let other = begin(&db);
    processor(&db, trips, Some("trips"), other)
        .process("MATCH (c:City) SET c.visits = 4", QueryLanguage::Gql, None)
        .expect("Paris is another node than Amsterdam");
    db.transaction_manager.commit(other.1).unwrap();
    session.commit().unwrap();

    let reader = db.session();
    assert_eq!(visits(&reader), Value::Int64(20), "Amsterdam");
    reader.use_graph("trips");
    assert_eq!(
        visits(&reader),
        Value::Int64(4),
        "the processor's commit is stamped and visible"
    );
}

/// A processor's transaction rolled back through the transaction manager
/// leaves nothing: its writes were recorded in its change set and undone.
#[test]
fn a_processor_transaction_aborted_by_the_manager_leaves_nothing() {
    let db = two_graphs();
    let root = db.lpg_store();
    let transaction = begin(&db);
    let processor = processor(&db, root, None, transaction);
    processor
        .process(
            "MATCH (c:City) SET c.visits = 88 INSERT (:City {name: 'Berlin'})",
            QueryLanguage::Gql,
            None,
        )
        .unwrap();
    db.transaction_manager.abort(transaction.1).unwrap();

    let rows = db
        .execute("MATCH (c:City) RETURN c.name, c.visits")
        .unwrap();
    assert_eq!(
        rows.rows(),
        [[Value::from("Amsterdam"), Value::Int64(19)]],
        "the update and the insert are undone"
    );
}
