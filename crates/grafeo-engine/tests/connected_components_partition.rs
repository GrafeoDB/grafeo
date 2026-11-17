//! `connected_components` groups the nodes the view sees into the weakly connected components of
//! its edges: two nodes share a component exactly when a chain of edges joins them, in either
//! direction. The oracle is a union-find over the edges the store lists, in this file; the
//! component numbers may differ, the partition may not.
//!
//! ```bash
//! cargo test -p grafeo-engine --all-features --test main connected_components_partition::
//! ```

#![cfg(feature = "algos")]

use std::collections::{BTreeMap, BTreeSet};

use grafeo_common::types::{NodeId, Value};
use grafeo_engine::GrafeoDB;

/// The partition as sets of node ids, from the algorithm.
fn found(db: &GrafeoDB) -> BTreeSet<BTreeSet<i64>> {
    found_in(db, "")
}

/// The partition as sets of node ids, from the algorithm with `arguments` (a projection).
fn found_in(db: &GrafeoDB, arguments: &str) -> BTreeSet<BTreeSet<i64>> {
    let result = db
        .session()
        .execute(&format!(
            "CALL grafeo.connected_components({arguments}) YIELD node_id, component_id \
             RETURN node_id, component_id"
        ))
        .unwrap();
    let mut groups: BTreeMap<i64, BTreeSet<i64>> = BTreeMap::new();
    for row in result.rows() {
        let (Value::Int64(node), Value::Int64(component)) = (&row[0], &row[1]) else {
            panic!("unexpected row {row:?}");
        };
        groups.entry(*component).or_default().insert(*node);
    }
    groups.into_values().collect()
}

/// The partition as sets of node ids, by a union-find over the nodes and edges a query lists.
fn oracle(db: &GrafeoDB) -> BTreeSet<BTreeSet<i64>> {
    let session = db.session();
    let ints = |row: &[Value]| -> Vec<i64> {
        row.iter()
            .map(|value| match value {
                Value::Int64(n) => *n,
                other => panic!("an id, not {other:?}"),
            })
            .collect()
    };
    let nodes: Vec<i64> = session
        .execute("MATCH (n) RETURN id(n)")
        .unwrap()
        .rows()
        .iter()
        .map(|row| ints(row)[0])
        .collect();
    let mut parent: BTreeMap<i64, i64> = nodes.iter().map(|&n| (n, n)).collect();
    fn root(parent: &mut BTreeMap<i64, i64>, mut n: i64) -> i64 {
        while parent[&n] != n {
            n = parent[&n];
        }
        n
    }
    for row in session
        .execute("MATCH (a)-[]->(b) RETURN id(a), id(b)")
        .unwrap()
        .rows()
    {
        let ends = ints(row);
        let (a, b) = (root(&mut parent, ends[0]), root(&mut parent, ends[1]));
        if a != b {
            parent.insert(a.max(b), a.min(b));
        }
    }
    let mut groups: BTreeMap<i64, BTreeSet<i64>> = BTreeMap::new();
    for &n in &nodes {
        let r = root(&mut parent, n);
        groups.entry(r).or_default().insert(n);
    }
    groups.into_values().collect()
}

/// Nodes joined only through edges into one node, a self-loop, an isolated node, parallel edges
/// and a cycle: each component as the edges make it, whatever their direction.
#[test]
fn every_component_shape_is_one_component() {
    let db = GrafeoDB::new_in_memory();
    let n: Vec<NodeId> = (0..9).map(|_| db.create_node(&["N"]).unwrap()).collect();
    // 0 -> 1 <- 2: 0 and 2 reach each other only through 1's incoming edges.
    db.create_edge(n[0], n[1], "E").unwrap();
    db.create_edge(n[2], n[1], "E").unwrap();
    // 3 with a self-loop alone; 4 isolated.
    db.create_edge(n[3], n[3], "E").unwrap();
    // 5 -> 6 twice.
    db.create_edge(n[5], n[6], "E").unwrap();
    db.create_edge(n[5], n[6], "E").unwrap();
    // 7 -> 8 -> 7.
    db.create_edge(n[7], n[8], "E").unwrap();
    db.create_edge(n[8], n[7], "E").unwrap();
    let want = oracle(&db);
    assert_eq!(want.len(), 5, "{want:?}");
    assert_eq!(found(&db), want);
}

/// Seeded random directed graphs, sparse to dense: the partition is the union-find's.
#[test]
fn random_directed_graphs_match_the_union_find() {
    for (seed, nodes, edges) in [
        (1_u64, 300, 150),
        (2, 300, 290),
        (3, 300, 600),
        (4, 50, 1_000),
    ] {
        let db = GrafeoDB::new_in_memory();
        let ids: Vec<NodeId> = (0..nodes)
            .map(|_| db.create_node(&["N"]).unwrap())
            .collect();
        let mut state = seed.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
        let mut next = |bound: usize| {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            usize::try_from(state >> 33).unwrap() % bound
        };
        for _ in 0..edges {
            let (a, b) = (next(nodes), next(nodes));
            db.create_edge(ids[a], ids[b], "E").unwrap();
        }
        assert_eq!(
            found(&db),
            oracle(&db),
            "seed {seed}, {nodes} nodes, {edges} edges"
        );
    }
}

/// A deleted node leaves its component with its edges: the nodes it alone joined split.
#[test]
fn a_deleted_node_splits_what_it_joined() {
    let db = GrafeoDB::new_in_memory();
    let n: Vec<NodeId> = (0..4).map(|_| db.create_node(&["N"]).unwrap()).collect();
    db.create_edge(n[0], n[1], "E").unwrap();
    db.create_edge(n[2], n[1], "E").unwrap();
    db.create_edge(n[3], n[2], "E").unwrap();
    assert_eq!(found(&db).len(), 1);
    db.session()
        .execute(&format!(
            "MATCH (x) WHERE id(x) = {} DETACH DELETE x",
            n[1].as_u64()
        ))
        .unwrap();
    let want = oracle(&db);
    assert_eq!(want.len(), 2, "{want:?}");
    assert_eq!(found(&db), want);
}

/// A view that leaves a node out leaves its edges out on both sides: an edge from a node of the
/// view to it, and one from it to a node of the view, join nothing, so two nodes joined only
/// through it stay apart. The whole graph joins them.
#[test]
fn an_edge_to_or_from_a_node_the_view_leaves_out_joins_nothing() {
    use grafeo_core::graph::ProjectionSpec;

    let db = GrafeoDB::new_in_memory();
    let [a, b, c] = [0; 3].map(|_| db.create_node(&["N"]).unwrap());
    let m = db.create_node(&["M"]).unwrap();
    db.create_edge(a, m, "E").unwrap();
    db.create_edge(m, b, "E").unwrap();
    db.create_edge(c, a, "E").unwrap();
    assert_eq!(found(&db).len(), 1, "the whole graph is one component");
    assert!(
        db.create_projection(
            "only_n",
            ProjectionSpec::new()
                .with_node_labels(["N"])
                .with_edge_types(["E"]),
        )
        .unwrap()
    );
    let id = |n: NodeId| i64::try_from(n.as_u64()).unwrap();
    let want: BTreeSet<BTreeSet<i64>> = [BTreeSet::from([id(a), id(c)]), BTreeSet::from([id(b)])]
        .into_iter()
        .collect();
    assert_eq!(found_in(&db, "{projection: 'only_n'}"), want);
}
