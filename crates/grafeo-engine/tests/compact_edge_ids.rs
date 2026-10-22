//! `compact()` keeps every edge id on its own edge: reads by id, queries and
//! deletes after a compaction reach the edge the id was given to.
//!
//! ```bash
//! cargo test -p grafeo-engine --features "lpg gql" --test compact_edge_ids
//! ```

#![cfg(all(feature = "lpg", feature = "gql"))]

use grafeo_common::types::{EdgeId, NodeId, PropertyKey, Value};
use grafeo_engine::GrafeoDB;

fn rows(db: &GrafeoDB, query: &str) -> Vec<Vec<Value>> {
    db.session().execute(query).unwrap().rows().to_vec()
}

fn person(db: &GrafeoDB, name: &str) -> NodeId {
    let id = db.create_node(&["Person"]).unwrap();
    db.set_node_property(id, "name", Value::from(name)).unwrap();
    id
}

fn knows(db: &GrafeoDB, from: NodeId, to: NodeId, w: i64) -> EdgeId {
    let id = db.create_edge(from, to, "KNOWS").unwrap();
    db.set_edge_property(id, "w", Value::Int64(w)).unwrap();
    id
}

/// The endpoints and `w` of the edge `id` names.
fn edge(db: &GrafeoDB, id: EdgeId) -> (NodeId, NodeId, Option<Value>) {
    let edge = db.get_edge(id).unwrap();
    let w = edge.properties.get(&PropertyKey::new("w")).cloned();
    (edge.src, edge.dst, w)
}

/// Alix's edge to Mia is created before her edge to Gus, while Gus was
/// created before Mia: the order of a node's edges differs from the order of
/// their targets. Each id keeps its edge, its property and its delete.
#[test]
fn compact_keeps_each_edge_id_on_its_own_edge() {
    let mut db = GrafeoDB::new_in_memory();
    let alix = person(&db, "Alix");
    let gus = person(&db, "Gus");
    let mia = person(&db, "Mia");
    let to_mia = knows(&db, alix, mia, 3);
    let to_gus = knows(&db, alix, gus, 19);
    db.compact().unwrap();

    assert_eq!(edge(&db, to_mia), (alix, mia, Some(Value::Int64(3))));
    assert_eq!(edge(&db, to_gus), (alix, gus, Some(Value::Int64(19))));
    let id = |edge: EdgeId| Value::Int64(i64::try_from(edge.as_u64()).unwrap());
    assert_eq!(
        rows(
            &db,
            "MATCH (:Person {name: 'Alix'})-[r:KNOWS]->(b) RETURN id(r), b.name, r.w ORDER BY r.w"
        ),
        [
            vec![id(to_mia), Value::from("Mia"), Value::Int64(3)],
            vec![id(to_gus), Value::from("Gus"), Value::Int64(19)],
        ]
    );

    assert!(db.delete_edge(to_mia).unwrap());
    assert_eq!(
        rows(
            &db,
            "MATCH (:Person {name: 'Alix'})-[r:KNOWS]->(b) RETURN b.name, r.w"
        ),
        [vec![Value::from("Gus"), Value::Int64(19)]],
        "deleting the edge to Mia deletes that edge only"
    );
}

/// Parallel edges keep their own ids when reached from their target.
#[test]
fn compact_keeps_parallel_edges_apart_on_incoming_traversal() {
    let mut db = GrafeoDB::new_in_memory();
    let vincent = person(&db, "Vincent");
    let jules = person(&db, "Jules");
    let first = knows(&db, vincent, jules, 3);
    let second = knows(&db, vincent, jules, 88);
    db.compact().unwrap();

    let id = |edge: EdgeId| Value::Int64(i64::try_from(edge.as_u64()).unwrap());
    assert_eq!(
        rows(
            &db,
            "MATCH (:Person {name: 'Jules'})<-[r:KNOWS]-(a) RETURN id(r), r.w ORDER BY r.w"
        ),
        [
            vec![id(first), Value::Int64(3)],
            vec![id(second), Value::Int64(88)],
        ]
    );
}

/// The reproduction of #221, in Cypher as reported: two edge types from
/// one node, then `compact()`, which failed with "duplicate edge type".
#[cfg(feature = "cypher")]
const SEVERAL_EDGE_TYPES: [&str; 5] = [
    "CREATE (:A {name: 'a'})",
    "CREATE (:B {name: 'b'})",
    "CREATE (:C {name: 'c'})",
    "MATCH (a:A), (b:B) CREATE (a)-[:CALLS]->(b)",
    "MATCH (a:A), (c:C) CREATE (a)-[:USES_TYPE]->(c)",
];

/// Every edge from `a`, as `[type, target name]`.
#[cfg(feature = "cypher")]
fn edges_from_a(db: &GrafeoDB) -> Vec<Vec<Value>> {
    db.execute_cypher("MATCH (:A)-[r]->(x) RETURN type(r) AS t, x.name ORDER BY t")
        .unwrap()
        .rows()
        .to_vec()
}

#[cfg(feature = "cypher")]
fn edge_from_a(edge_type: &str, target: &str) -> Vec<Value> {
    vec![Value::from(edge_type), Value::from(target)]
}

/// #221: `compact()` of a graph with several edge types succeeds, and every
/// edge keeps its type and target; an edge of a third type written after it
/// joins them.
#[cfg(feature = "cypher")]
#[test]
fn compact_with_several_edge_types_keeps_each_edge() {
    let mut db = GrafeoDB::new_in_memory();
    for statement in SEVERAL_EDGE_TYPES {
        db.execute_cypher(statement).unwrap();
    }
    db.compact().unwrap();
    assert_eq!(
        edges_from_a(&db),
        [edge_from_a("CALLS", "b"), edge_from_a("USES_TYPE", "c")]
    );
    db.execute_cypher("MATCH (a:A), (c:C) CREATE (a)-[:IMPORTS]->(c)")
        .unwrap();
    assert_eq!(
        edges_from_a(&db),
        [
            edge_from_a("CALLS", "b"),
            edge_from_a("IMPORTS", "c"),
            edge_from_a("USES_TYPE", "c"),
        ]
    );
}

/// #221 on a database file: the compacted edges and the one written after
/// `compact()` are there after a reopen.
#[cfg(all(feature = "cypher", feature = "grafeo-file", feature = "wal"))]
#[test]
fn compact_with_several_edge_types_survives_a_reopen() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("db.grafeo");
    let mut db = GrafeoDB::open(&path).unwrap();
    for statement in SEVERAL_EDGE_TYPES {
        db.execute_cypher(statement).unwrap();
    }
    db.compact().unwrap();
    db.execute_cypher("MATCH (a:A), (c:C) CREATE (a)-[:IMPORTS]->(c)")
        .unwrap();
    db.close().unwrap();
    drop(db);

    let db = GrafeoDB::open(&path).unwrap();
    assert_eq!(
        edges_from_a(&db),
        [
            edge_from_a("CALLS", "b"),
            edge_from_a("IMPORTS", "c"),
            edge_from_a("USES_TYPE", "c"),
        ]
    );
    db.close().unwrap();
}

/// One edge type between three pairs of labels, and a second type beside
/// it: after `compact()` each edge keeps its endpoints and its weight.
#[test]
fn one_edge_type_across_label_pairs_survives_compact() {
    let mut db = GrafeoDB::new_in_memory();
    db.execute(
        "INSERT (alix:Person {name: 'Alix'}), (gus:Person {name: 'Gus'}), \
         (ams:City {name: 'Amsterdam'}), (ber:City {name: 'Berlin'}), \
         (alix)-[:LINKS {w: 1}]->(gus), (alix)-[:LINKS {w: 2}]->(ams), \
         (ams)-[:LINKS {w: 3}]->(ber), (gus)-[:LIVES_IN {w: 4}]->(ber)",
    )
    .unwrap();
    let query = "MATCH (a)-[r]->(b) \
                 RETURN r.w AS w, type(r), labels(a), a.name, labels(b), b.name ORDER BY w";
    let edge = |w: i64, edge_type: &str, from: [&str; 2], to: [&str; 2]| {
        let label = |name: &str| Value::List(std::iter::once(Value::from(name)).collect());
        vec![
            Value::Int64(w),
            Value::from(edge_type),
            label(from[0]),
            Value::from(from[1]),
            label(to[0]),
            Value::from(to[1]),
        ]
    };
    let expected = [
        edge(1, "LINKS", ["Person", "Alix"], ["Person", "Gus"]),
        edge(2, "LINKS", ["Person", "Alix"], ["City", "Amsterdam"]),
        edge(3, "LINKS", ["City", "Amsterdam"], ["City", "Berlin"]),
        edge(4, "LIVES_IN", ["Person", "Gus"], ["City", "Berlin"]),
    ];
    assert_eq!(rows(&db, query), expected, "before compact()");
    db.compact().unwrap();
    assert_eq!(rows(&db, query), expected, "after compact()");
}
