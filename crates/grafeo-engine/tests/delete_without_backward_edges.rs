//! A node with incoming edges is not deleted without its edges, also in a
//! database without backward edges (`Config::without_backward_edges`).
//!
//! 0.6.0-beta.1 and 0.5.x found a node's incoming edges there only through a
//! check that ran in debug builds: in a release build a plain `DELETE` of a
//! node with incoming edges went through and left the edges pointing at a
//! node that no longer existed. The store now finds the incoming edges
//! without the backward adjacency (a scan of the edge table) in every build,
//! so these tests hold in a release build too (`cargo test --release`).
//!
//! ```bash
//! cargo test -p grafeo-engine --all-features --test delete_without_backward_edges
//! cargo test -p grafeo-engine --all-features --release --test delete_without_backward_edges
//! ```

#![cfg(all(feature = "lpg", feature = "gql"))]

use grafeo_common::types::Value;
use grafeo_engine::{Config, Direction, GrafeoDB};

/// Alix knows Gus, in a database that keeps no backward edges.
fn alix_knows_gus() -> GrafeoDB {
    let db = GrafeoDB::with_config(Config::in_memory().without_backward_edges()).unwrap();
    db.execute("INSERT (:Person {name: 'Alix'})-[:KNOWS]->(:Person {name: 'Gus'})")
        .unwrap();
    db
}

fn count(db: &GrafeoDB, query: &str) -> i64 {
    match db.execute(query).unwrap().rows()[0][0] {
        Value::Int64(count) => count,
        ref other => panic!("a count, got {other:?}"),
    }
}

/// A plain `DELETE` of the edge's target is refused, in GQL and Cypher, and
/// leaves the node and its edge as they were.
#[test]
fn a_plain_delete_of_a_node_with_incoming_edges_is_refused() {
    let db = alix_knows_gus();
    for (language, statement) in [
        ("gql", "MATCH (g:Person {name: 'Gus'}) DELETE g"),
        ("cypher", "MATCH (g:Person {name: 'Gus'}) DELETE g"),
    ] {
        let refused = match language {
            "gql" => db.execute(statement),
            _ => db.execute_cypher(statement),
        }
        .expect_err("a node with an incoming edge is not deleted on its own");
        assert!(
            refused.to_string().contains("connected edge"),
            "{language}: {refused}"
        );
        assert_eq!(
            count(&db, "MATCH (p:Person) RETURN count(p)"),
            2,
            "{language}"
        );
        assert_eq!(
            count(&db, "MATCH (:Person)-[r:KNOWS]->(:Person) RETURN count(r)"),
            1,
            "{language}: the edge still has both ends"
        );
    }
}

/// The direct API refuses it too, and `DETACH DELETE` takes the incoming
/// edge with the node: no edge is left pointing at a deleted node.
#[test]
fn detach_delete_takes_the_incoming_edges_and_leaves_none_dangling() {
    let db = alix_knows_gus();
    let gus = db
        .execute("MATCH (g:Person {name: 'Gus'}) RETURN id(g)")
        .unwrap()
        .rows()[0][0]
        .clone();
    let Value::Int64(gus) = gus else {
        panic!("an id, got {gus:?}");
    };
    let gus = grafeo_common::types::NodeId::new(u64::try_from(gus).unwrap());
    assert!(
        db.delete_node(gus).is_err(),
        "the direct API refuses the delete of a node with an incoming edge"
    );
    assert_eq!(db.get_node_edges(gus, Direction::Incoming, &[]).len(), 1);

    db.execute("MATCH (g:Person {name: 'Gus'}) DETACH DELETE g")
        .unwrap();
    assert_eq!(count(&db, "MATCH (p:Person) RETURN count(p)"), 1);
    assert_eq!(db.edge_count(), 0, "the incoming edge went with Gus");
    assert_eq!(count(&db, "MATCH ()-[r]->() RETURN count(r)"), 0);
}
