//! PageRank, Louvain and label propagation over the nodes in the order of a
//! key give the same result, bit for bit, whatever order the nodes and edges
//! were inserted in (#566 `key=`). Without a key they follow the internal ids,
//! which the insertion order decides (#592).

#![cfg(all(feature = "lpg", feature = "algos"))]

use std::collections::BTreeMap;

use grafeo_adapters::plugins::algorithms::{
    label_propagation, label_propagation_in_order, louvain, louvain_in_order, order_by_key,
    pagerank, pagerank_in_order,
};
use grafeo_common::types::{NodeId, PropertyKey, Value};
use grafeo_core::graph::lpg::LpgStore;

/// A seeded graph shaped like a code base (modules of files of functions that
/// call each other, with a few shared hubs), as node count and edge list.
fn code_graph(modules: usize) -> (usize, Vec<(usize, usize)>) {
    let mut state: u64 = 0x9E37_79B9_7F4A_7C15;
    let mut next = |bound: usize| {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        usize::try_from(state % u64::try_from(bound).unwrap()).unwrap()
    };
    let mut nodes = 1;
    let mut edges = Vec::new();
    let mut functions_by_module: Vec<Vec<usize>> = Vec::new();
    for _ in 0..modules {
        let module = nodes;
        nodes += 1;
        edges.push((0, module));
        let mut functions = Vec::new();
        for _ in 0..(6 + next(5)) {
            let file = nodes;
            nodes += 1;
            edges.push((module, file));
            for _ in 0..(18 + next(14)) {
                edges.push((file, nodes));
                functions.push(nodes);
                nodes += 1;
            }
        }
        functions_by_module.push(functions);
    }
    let hubs: Vec<usize> = functions_by_module.iter().map(|f| f[0]).take(8).collect();
    for functions in &functions_by_module {
        for &caller in functions {
            for _ in 0..next(4) {
                let callee = if next(10) == 0 {
                    let other = &functions_by_module[next(modules)];
                    other[next(other.len())]
                } else {
                    functions[next(functions.len())]
                };
                edges.push((caller, callee));
            }
            if next(6) == 0 {
                edges.push((caller, hubs[next(hubs.len())]));
            }
        }
    }
    (nodes, edges)
}

/// The key of node `i` of the abstract graph.
fn key(i: usize) -> String {
    format!("n{i:05}")
}

/// The graph loaded into a store, nodes and edges in order or both reversed,
/// with each node's key in the property `key`.
fn load(nodes: usize, edges: &[(usize, usize)], reversed: bool) -> LpgStore {
    let store = LpgStore::new().unwrap();
    let mut ids = vec![NodeId::new(0); nodes];
    let order: Vec<usize> = if reversed {
        (0..nodes).rev().collect()
    } else {
        (0..nodes).collect()
    };
    for (count, i) in order.into_iter().enumerate() {
        if count % 1000 == 0 {
            // `tiered-storage` (on under --all-features) bounds an epoch's arena.
            store.new_epoch();
        }
        ids[i] = store.create_node_with_props(&["Node"], [("key", Value::from(key(i)))]);
    }
    let edge_order: Vec<&(usize, usize)> = if reversed {
        edges.iter().rev().collect()
    } else {
        edges.iter().collect()
    };
    for (count, &(from, to)) in edge_order.into_iter().enumerate() {
        if count % 1000 == 0 {
            store.new_epoch();
        }
        store.create_edge(ids[from], ids[to], "EDGE");
    }
    store
}

fn key_order(store: &LpgStore) -> Vec<NodeId> {
    order_by_key(store, "key")
        .unwrap()
        .into_iter()
        .map(|(node, _)| node)
        .collect()
}

/// A result keyed by the nodes' keys.
fn by_key<T>(
    store: &LpgStore,
    result: impl IntoIterator<Item = (NodeId, T)>,
) -> BTreeMap<String, T> {
    let property = PropertyKey::new("key");
    result
        .into_iter()
        .map(
            |(node, value)| match store.get_node_property(node, &property) {
                Some(Value::String(key)) => (key.to_string(), value),
                other => panic!("key of {node:?}: {other:?}"),
            },
        )
        .collect()
}

fn pagerank_bits(store: &LpgStore, keyed: bool, directed: bool) -> BTreeMap<String, u64> {
    let scores = if keyed {
        pagerank_in_order(store, &key_order(store), 0.85, 100, 1e-12, directed)
    } else {
        pagerank(store, 0.85, 100, 1e-12, directed)
    };
    by_key(
        store,
        scores
            .into_iter()
            .map(|(node, score)| (node, score.to_bits())),
    )
}

/// Louvain's communities by key, with the modularity's bits.
fn louvain_by_key(store: &LpgStore, keyed: bool) -> (BTreeMap<String, u64>, u64) {
    let result = if keyed {
        louvain_in_order(store, &key_order(store), 1.0)
    } else {
        louvain(store, 1.0)
    };
    (
        by_key(store, result.communities),
        result.modularity.to_bits(),
    )
}

fn label_propagation_by_key(store: &LpgStore, keyed: bool) -> BTreeMap<String, u64> {
    let labels = if keyed {
        label_propagation_in_order(store, &key_order(store), 100)
    } else {
        label_propagation(store, 100)
    };
    by_key(store, labels)
}

const MODULES: usize = 8;

/// In key order, PageRank (directed and undirected), Louvain and label
/// propagation return the same bits and community numbers for the graph loaded
/// forward and reversed. Without a key, PageRank's scores differ between the
/// two loads (their sums run in another order), so the test is not vacuous.
#[test]
fn results_in_key_order_do_not_depend_on_the_insertion_order() {
    let (nodes, edges) = code_graph(MODULES);
    let forward = load(nodes, &edges, false);
    let backward = load(nodes, &edges, true);

    for directed in [true, false] {
        assert_eq!(
            pagerank_bits(&forward, true, directed),
            pagerank_bits(&backward, true, directed),
            "pagerank, directed {directed}"
        );
    }
    assert_ne!(
        pagerank_bits(&forward, false, false),
        pagerank_bits(&backward, false, false),
        "without a key the insertion order shows"
    );
    assert_eq!(
        louvain_by_key(&forward, true),
        louvain_by_key(&backward, true)
    );
    assert_eq!(
        label_propagation_by_key(&forward, true),
        label_propagation_by_key(&backward, true)
    );
}

/// With keys that sort like the node ids, the key order is the id order, and
/// the results equal those without a key, bit for bit.
#[test]
fn keys_that_sort_like_ids_give_the_results_without_a_key() {
    let (nodes, edges) = code_graph(MODULES);
    let store = load(nodes, &edges, false);
    let mut ids = store.node_ids();
    ids.sort_unstable();
    assert_eq!(key_order(&store), ids);

    for directed in [true, false] {
        assert_eq!(
            pagerank_bits(&store, true, directed),
            pagerank_bits(&store, false, directed)
        );
    }
    assert_eq!(louvain_by_key(&store, true), louvain_by_key(&store, false));
    assert_eq!(
        label_propagation_by_key(&store, true),
        label_propagation_by_key(&store, false)
    );
}

/// Communities are numbered by their smallest key: two triangles, the one
/// created second holding the smaller keys, are 0 and 1 the other way round
/// with a key than without.
#[test]
fn communities_are_numbered_by_their_smallest_key() {
    let store = LpgStore::new().unwrap();
    let triangle = |keys: [&str; 3]| -> Vec<NodeId> {
        let ids: Vec<NodeId> = keys
            .iter()
            .map(|key| store.create_node_with_props(&["Node"], [("key", Value::from(*key))]))
            .collect();
        for (a, b) in [(0, 1), (1, 2), (2, 0)] {
            store.create_edge(ids[a], ids[b], "EDGE");
        }
        ids
    };
    let first = triangle(["Vincent", "Jules", "Mia"]);
    let second = triangle(["Alix", "Gus", "Butch"]);
    let order = key_order(&store);

    let plain = louvain(&store, 1.0).communities;
    let keyed = louvain_in_order(&store, &order, 1.0).communities;
    assert_eq!((plain[&first[0]], plain[&second[0]]), (0, 1));
    assert_eq!((keyed[&first[0]], keyed[&second[0]]), (1, 0));

    let plain = label_propagation(&store, 100);
    let keyed = label_propagation_in_order(&store, &order, 100);
    assert_eq!((plain[&first[0]], plain[&second[0]]), (0, 1));
    assert_eq!((keyed[&first[0]], keyed[&second[0]]), (1, 0));
}
