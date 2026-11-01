//! A named graph a transaction creates by writing to it is part of the
//! transaction: a rollback leaves no graph behind (#415).
//!
//! ```bash
//! cargo test -p grafeo-engine --all-features --test rolled_back_graphs
//! ```

#![cfg(all(feature = "gql", feature = "sparql", feature = "triple-store"))]

use grafeo_engine::{Config, GrafeoDB, GraphModel};

#[test]
fn a_rolled_back_sparql_insert_leaves_no_named_graph() {
    let db = GrafeoDB::with_config(Config::in_memory().with_graph_model(GraphModel::Rdf)).unwrap();
    let mut session = db.session();
    session.begin_transaction().unwrap();
    session
        .execute_sparql(
            "INSERT DATA { GRAPH <http://example.org/trips> { <http://example.org/alix> <http://example.org/visits> <http://example.org/berlin> } }",
        )
        .unwrap();
    session.rollback().unwrap();
    // A graph that survived the rollback makes CREATE GRAPH fail.
    db.execute_sparql("CREATE GRAPH <http://example.org/trips>")
        .expect("the rolled-back transaction left no graph behind");

    // The control: a graph a committed insert created is there.
    session.begin_transaction().unwrap();
    session
        .execute_sparql(
            "INSERT DATA { GRAPH <http://example.org/visits> { <http://example.org/gus> <http://example.org/visits> <http://example.org/paris> } }",
        )
        .unwrap();
    session.commit().unwrap();
    let error = db
        .execute_sparql("CREATE GRAPH <http://example.org/visits>")
        .expect_err("a committed insert created the graph");
    assert!(error.to_string().contains("exist"), "{error}");
}

#[test]
fn a_rolled_back_write_to_a_new_lpg_graph_leaves_no_graph() {
    let db = GrafeoDB::new_in_memory();
    let mut session = db.session();
    session.begin_transaction().unwrap();
    let wrote = session.execute("USE GRAPH trips INSERT (:City {name: 'Berlin'})");
    session.rollback().unwrap();
    assert!(
        !db.list_graphs().contains(&"trips".to_string()),
        "write {wrote:?} left {:?}",
        db.list_graphs()
    );
}
