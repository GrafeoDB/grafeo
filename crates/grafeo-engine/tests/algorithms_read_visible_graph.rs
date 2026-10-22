//! `CALL` algorithms read the committed graph: they never reach a node
//! another transaction created and has not committed, nor follow its
//! uncommitted edges, nor reach a node deleted at the store level whose edges
//! remain.

#![cfg(all(feature = "algos", feature = "lpg"))]

use std::collections::BTreeMap;

use grafeo_common::types::{NodeId, Value};
use grafeo_engine::GrafeoDB;
use grafeo_engine::session::Session;

/// Alix -> Gus -> Vincent.
fn chain() -> GrafeoDB {
    let db = GrafeoDB::new_in_memory();
    db.execute(
        "INSERT (:Person {name: 'Alix'})-[:KNOWS]->(:Person {name: 'Gus'})\
         -[:KNOWS]->(:Person {name: 'Vincent'})",
    )
    .unwrap();
    db
}

fn id_of(session: &Session, name: &str) -> i64 {
    let result = session
        .execute(&format!("MATCH (p:Person {{name: '{name}'}}) RETURN id(p)"))
        .unwrap();
    match result.rows()[0][0] {
        Value::Int64(id) => id,
        ref other => panic!("id() returned {other:?}"),
    }
}

/// BFS depth by name, from Alix.
fn bfs_from_alix(session: &Session) -> BTreeMap<String, i64> {
    let names: BTreeMap<i64, String> = session
        .execute("MATCH (p:Person) RETURN id(p), p.name")
        .unwrap()
        .rows()
        .iter()
        .map(|row| match (&row[0], &row[1]) {
            (Value::Int64(id), Value::String(name)) => (*id, name.to_string()),
            other => panic!("unexpected row {other:?}"),
        })
        .collect();
    let alix = id_of(session, "Alix");
    session
        .execute(&format!(
            "CALL grafeo.bfs({alix}) YIELD node_id, depth RETURN node_id, depth"
        ))
        .unwrap()
        .rows()
        .iter()
        .map(|row| match (&row[0], &row[1]) {
            (Value::Int64(id), Value::Int64(depth)) => (
                names
                    .get(id)
                    .cloned()
                    .unwrap_or_else(|| format!("invisible node {id}")),
                *depth,
            ),
            other => panic!("unexpected row {other:?}"),
        })
        .collect()
}

fn expected(pairs: &[(&str, i64)]) -> BTreeMap<String, i64> {
    pairs
        .iter()
        .map(|(name, depth)| ((*name).to_string(), *depth))
        .collect()
}

fn rows(session: &Session, query: &str) -> Vec<Vec<Value>> {
    session.execute(query).unwrap().rows().to_vec()
}

#[test]
fn an_algorithm_does_not_read_another_transactions_uncommitted_writes() {
    let db = chain();
    let reader = db.session();
    let pagerank =
        "CALL grafeo.pagerank() YIELD node_id, score RETURN node_id, score ORDER BY node_id";
    let components = "CALL grafeo.connected_components() YIELD node_id, component_id \
                      RETURN node_id, component_id ORDER BY node_id";
    let ranks_before = rows(&reader, pagerank);
    let components_before = rows(&reader, components);

    let mut writer = db.session();
    writer.begin_transaction().unwrap();
    writer
        .execute(
            "MATCH (a:Person {name: 'Alix'}), (v:Person {name: 'Vincent'}) \
             INSERT (a)-[:KNOWS]->(v), (v)-[:KNOWS]->(:Person {name: 'Django'})",
        )
        .unwrap();

    assert_eq!(
        bfs_from_alix(&reader),
        expected(&[("Alix", 0), ("Gus", 1), ("Vincent", 2)]),
        "the open transaction's shortcut and its new node are not committed"
    );
    assert_eq!(rows(&reader, pagerank), ranks_before);
    assert_eq!(rows(&reader, components), components_before);

    writer.commit().unwrap();
    assert_eq!(
        bfs_from_alix(&reader),
        expected(&[("Alix", 0), ("Django", 2), ("Gus", 1), ("Vincent", 1)]),
        "after the commit the algorithm reads the new edges"
    );
}

#[test]
fn an_algorithm_skips_a_node_deleted_without_its_edges() {
    let db = chain();
    let session = db.session();
    session
        .execute(
            "MATCH (a:Person {name: 'Alix'}), (v:Person {name: 'Vincent'}) \
             INSERT (a)-[:KNOWS]->(:Person {name: 'Butch'})-[:KNOWS]->(v)",
        )
        .unwrap();
    let butch = id_of(&session, "Butch");
    // The store-level delete leaves Butch's edges in place.
    assert!(
        db.store()
            .delete_node(NodeId::new(u64::try_from(butch).unwrap()))
    );

    assert_eq!(
        bfs_from_alix(&session),
        expected(&[("Alix", 0), ("Gus", 1), ("Vincent", 2)])
    );
    let centrality = rows(
        &session,
        "CALL grafeo.betweenness_centrality() YIELD node_id, centrality \
         RETURN node_id, centrality ORDER BY node_id",
    );
    assert_eq!(centrality.len(), 3, "only the three visible people");
}
