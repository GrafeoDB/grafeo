//! A reused change set carries no pending RDF triple (see
//! `recycled_change_sets.rs` for the property graph): a transaction's reads
//! lay its pending triples over the store and its commit applies them, so a
//! triple left over from the set's last transaction would show in the next
//! one's reads and be applied by its commit.

#![cfg(all(feature = "sparql", feature = "triple-store"))]

use grafeo_common::types::Value;
use grafeo_engine::config::GraphModel;
use grafeo_engine::session::Session;
use grafeo_engine::{Config, GrafeoDB};

fn rdf_db() -> GrafeoDB {
    GrafeoDB::with_config(Config::in_memory().with_graph_model(GraphModel::Rdf)).unwrap()
}

/// The objects of `<http://ex.org/s> <http://ex.org/p>`, sorted, as `session`
/// reads them.
fn objects(session: &Session) -> Vec<String> {
    let result = session
        .execute_sparql("SELECT ?o WHERE { <http://ex.org/s> <http://ex.org/p> ?o }")
        .unwrap();
    let mut objects: Vec<String> = result
        .rows()
        .iter()
        .map(|row| match &row[0] {
            Value::RdfTerm(term) => match term.as_ref() {
                grafeo_common::types::rdf::Term::Literal(literal) => literal.value().to_string(),
                other => format!("{other:?}"),
            },
            other => format!("{other:?}"),
        })
        .collect();
    objects.sort();
    objects
}

fn insert(object: &str) -> String {
    format!("INSERT DATA {{ <http://ex.org/s> <http://ex.org/p> \"{object}\" }}")
}

fn delete(object: &str) -> String {
    format!("DELETE DATA {{ <http://ex.org/s> <http://ex.org/p> \"{object}\" }}")
}

/// A rolled-back update, then transactions on the reused set: none of them
/// sees the rolled-back triple, and none commits it.
#[test]
fn a_rolled_back_update_leaves_no_pending_triple_for_the_next_transaction() {
    let db = rdf_db();
    let mut session = db.session();
    for round in 0..5 {
        session.begin_transaction().unwrap();
        session.execute_sparql(&insert("rolled back")).unwrap();
        assert!(
            objects(&session).contains(&"rolled back".to_string()),
            "round {round}: the transaction sees its own triple"
        );
        session.rollback().unwrap();

        session.begin_transaction().unwrap();
        assert!(
            !objects(&session).contains(&"rolled back".to_string()),
            "round {round}: the next transaction sees no pending triple"
        );
        session
            .execute_sparql(&insert(&format!("kept {round}")))
            .unwrap();
        session.commit().unwrap();
    }
    let expected: Vec<String> = (0..5).map(|round| format!("kept {round}")).collect();
    assert_eq!(
        objects(&db.session()),
        expected,
        "only the committed triples"
    );
}

/// A committed delete leaves a pending delete in the set it used. When the
/// triple is inserted again and later transactions reuse that set (a
/// session's, and a direct call's), the old delete must not apply again.
#[test]
fn a_committed_update_is_not_applied_again_by_a_later_transaction() {
    let db = rdf_db();
    let mut session = db.session();
    session.execute_sparql(&insert("Alix")).unwrap();
    session.begin_transaction().unwrap();
    session.execute_sparql(&delete("Alix")).unwrap();
    session.commit().unwrap();
    session.execute_sparql(&insert("Alix")).unwrap();
    assert_eq!(objects(&db.session()), ["Alix"]);

    // Direct calls that run as a private transaction, on a reused set.
    let ids = db
        .batch_create_nodes_with_props("Doc", vec![Default::default(), Default::default()])
        .unwrap();
    let hub = db.create_node(&["Hub"]).unwrap();
    db.create_edge(ids[0], hub, "LINK").unwrap();
    assert!(db.detach_delete_node(ids[0]).unwrap());
    assert_eq!(objects(&db.session()), ["Alix"], "after the direct calls");

    // Transactions that write something else, on reused sets.
    for round in 0..3 {
        session.begin_transaction().unwrap();
        session
            .execute_sparql(&insert(&format!("Gus {round}")))
            .unwrap();
        session.commit().unwrap();
        db.session()
            .execute_sparql(&format!(
                "INSERT DATA {{ <http://ex.org/other> <http://ex.org/p> \"{round}\" }}"
            ))
            .unwrap();
    }
    let mut expected = vec!["Alix".to_string()];
    expected.extend((0..3).map(|round| format!("Gus {round}")));
    assert_eq!(
        objects(&db.session()),
        expected,
        "the delete applied once, at its own commit"
    );
}

/// A committed transaction's pending triples are its own: once the set is
/// reused, the next transaction reads the store as it is, also when the
/// store changed in between without a transaction (a raw write). A pending
/// delete left over would hide a triple that is there; a pending insert
/// would show one that is gone.
#[test]
fn the_next_transaction_reads_the_store_not_the_last_ones_pending_triples() {
    use grafeo_core::graph::rdf::{Term, Triple};

    let triple = |object: &str| {
        Triple::new(
            Term::iri("http://ex.org/s"),
            Term::iri("http://ex.org/p"),
            Term::literal(object),
        )
    };
    let db = rdf_db();
    let mut session = db.session();
    session.execute_sparql(&insert("Alix")).unwrap();

    // A committed delete, then the triple back through the raw store.
    session.begin_transaction().unwrap();
    session.execute_sparql(&delete("Alix")).unwrap();
    session.commit().unwrap();
    assert!(db.rdf_store().insert(triple("Alix")));
    session.begin_transaction().unwrap();
    assert_eq!(
        objects(&session),
        ["Alix"],
        "the last transaction's delete is not pending in this one"
    );
    session.rollback().unwrap();

    // A committed insert, then the triple gone through the raw store.
    session.begin_transaction().unwrap();
    session.execute_sparql(&insert("Gus")).unwrap();
    session.commit().unwrap();
    assert!(db.rdf_store().remove(&triple("Gus")));
    session.begin_transaction().unwrap();
    assert_eq!(
        objects(&session),
        ["Alix"],
        "the last transaction's insert is not pending in this one"
    );
    session.rollback().unwrap();
}
