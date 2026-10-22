//! Graph algorithms read the graph visible at the store's epoch: the nodes of
//! `node_ids` and the visible edges between them. A store's adjacency also
//! holds edges to a node a store-level `delete_node` removed without
//! detaching it, and edges a transaction created and has not committed (to a
//! node only that transaction sees, or between two visible nodes); an
//! algorithm that followed them reported nodes outside the graph, or ran on
//! another graph. Every algorithm must give, on a store with such leftovers,
//! exactly what it gives on the same graph without them.

#![cfg(all(feature = "lpg", feature = "algos"))]

use grafeo_adapters::plugins::Parameters;
use grafeo_adapters::plugins::algorithms::{
    self, ArticulationPointsAlgorithm, BellmanFordAlgorithm, BetweennessCentralityAlgorithm,
    BfsAlgorithm, BridgesAlgorithm, ClosenessCentralityAlgorithm, ClusteringCoefficientAlgorithm,
    ConnectedComponentsAlgorithm, DegreeCentralityAlgorithm, DfsAlgorithm, DijkstraAlgorithm,
    FloydWarshallAlgorithm, GraphAlgorithm, KCoreAlgorithm, KTrussAlgorithm, KruskalAlgorithm,
    LabelPropagationAlgorithm, LouvainAlgorithm, MaxFlowAlgorithm, MinCostFlowAlgorithm,
    PageRankAlgorithm, PrimAlgorithm, SsspAlgorithm, StochasticBlockPartitionAlgorithm,
    StronglyConnectedComponentsAlgorithm, SubgraphIsomorphismAlgorithm, TopologicalSortAlgorithm,
    TotalTrianglesAlgorithm,
};
use grafeo_common::types::{NodeId, TransactionId, Value};
use grafeo_core::graph::lpg::LpgStore;

/// The people of both stores, in creation order: their ids are equal.
const ALIX: u64 = 0;
const GUS: u64 = 1;
const MIA: u64 = 3;
const JULES: u64 = 4;

/// Alix -> Gus -> Vincent -> Mia, Alix -> Jules -> Vincent, Gus -> Jules,
/// with weights. A DAG, so topological sort has an answer.
fn people() -> LpgStore {
    let store = LpgStore::new().unwrap();
    let ids: Vec<NodeId> = (0..5).map(|_| store.create_node(&["Person"])).collect();
    for (src, dst, weight) in [
        (0, 1, 3),
        (1, 2, 19),
        (2, 3, 88),
        (0, 4, 19),
        (4, 2, 3),
        (1, 4, 88),
    ] {
        let edge = store.create_edge(ids[src], ids[dst], "KNOWS");
        store.set_edge_property(edge, "weight", Value::Int64(weight));
        store.set_edge_property(edge, "cost", Value::Int64(1));
    }
    store
}

/// The same people, with what the adjacency holds beyond the visible graph:
/// Butch, deleted at the store level while his edges remain (one of them to
/// Alix, which would make a cycle), and an open transaction's new node Django
/// and its new edges, among them one from Mia back to Alix (a cycle too) and
/// a shortcut from Alix to Mia.
fn people_with_leftovers() -> LpgStore {
    let store = people();
    let node = |id: u64| NodeId::new(id);

    let butch = store.create_node(&["Person"]);
    for (src, dst) in [(node(ALIX), butch), (butch, node(MIA)), (butch, node(ALIX))] {
        let edge = store.create_edge(src, dst, "KNOWS");
        store.set_edge_property(edge, "weight", Value::Int64(1));
    }
    assert!(
        store.delete_node(butch),
        "Butch is deleted, his edges are not"
    );

    let open = TransactionId::new(88);
    let epoch = store.current_epoch();
    let django = store.create_node_versioned(&["Person"], epoch, open);
    for (src, dst) in [
        (node(GUS), django),
        (django, node(JULES)),
        (node(ALIX), node(MIA)),
        (node(MIA), node(ALIX)),
    ] {
        store.create_edge_versioned(src, dst, "KNOWS", epoch, open);
    }
    assert_eq!(
        store.node_ids(),
        (0..5).map(NodeId::new).collect::<Vec<_>>(),
        "only the five people are visible"
    );
    store
}

fn with(pairs: &[(&str, Value)]) -> Parameters {
    let mut params = Parameters::new();
    for (name, value) in pairs {
        match value {
            Value::Bool(b) => params.set_bool(*name, *b),
            Value::Int64(i) => params.set_int(*name, *i),
            Value::Float64(f) => params.set_float(*name, *f),
            Value::String(s) => params.set_string(*name, s.as_str()),
            other => panic!("unexpected parameter {other:?}"),
        }
    }
    params
}

/// Every algorithm wrapper, with the parameters it needs.
fn algorithms_under_test() -> Vec<(&'static str, Box<dyn GraphAlgorithm>, Parameters)> {
    let start = || with(&[("start", Value::Int64(0))]);
    let weighted_source = || {
        with(&[
            ("source", Value::Int64(0)),
            ("weight", Value::from("weight")),
        ])
    };
    let flow = || {
        with(&[
            ("source", Value::Int64(0)),
            ("sink", Value::Int64(3)),
            ("capacity", Value::from("weight")),
            ("cost", Value::from("cost")),
        ])
    };
    vec![
        ("pagerank", Box::new(PageRankAlgorithm), Parameters::new()),
        (
            "pagerank_undirected",
            Box::new(PageRankAlgorithm),
            with(&[("directed", Value::Bool(false))]),
        ),
        (
            "betweenness_centrality",
            Box::new(BetweennessCentralityAlgorithm),
            Parameters::new(),
        ),
        (
            "closeness_centrality",
            Box::new(ClosenessCentralityAlgorithm),
            Parameters::new(),
        ),
        (
            "degree_centrality",
            Box::new(DegreeCentralityAlgorithm),
            Parameters::new(),
        ),
        ("bfs", Box::new(BfsAlgorithm), start()),
        ("dfs", Box::new(DfsAlgorithm), start()),
        (
            "connected_components",
            Box::new(ConnectedComponentsAlgorithm),
            Parameters::new(),
        ),
        (
            "strongly_connected_components",
            Box::new(StronglyConnectedComponentsAlgorithm),
            Parameters::new(),
        ),
        (
            "topological_sort",
            Box::new(TopologicalSortAlgorithm),
            Parameters::new(),
        ),
        ("dijkstra", Box::new(DijkstraAlgorithm), weighted_source()),
        (
            "dijkstra_to_mia",
            Box::new(DijkstraAlgorithm),
            with(&[("source", Value::Int64(0)), ("target", Value::Int64(3))]),
        ),
        (
            "sssp",
            Box::new(SsspAlgorithm),
            with(&[("source", Value::from("0"))]),
        ),
        (
            "bellman_ford",
            Box::new(BellmanFordAlgorithm),
            weighted_source(),
        ),
        (
            "floyd_warshall",
            Box::new(FloydWarshallAlgorithm),
            with(&[("weight", Value::from("weight"))]),
        ),
        (
            "clustering_coefficient",
            Box::new(ClusteringCoefficientAlgorithm),
            with(&[("parallel", Value::Bool(false))]),
        ),
        (
            "total_triangles",
            Box::new(TotalTrianglesAlgorithm),
            with(&[("parallel", Value::Bool(false))]),
        ),
        (
            "label_propagation",
            Box::new(LabelPropagationAlgorithm),
            Parameters::new(),
        ),
        ("louvain", Box::new(LouvainAlgorithm), Parameters::new()),
        (
            "stochastic_block_partition",
            Box::new(StochasticBlockPartitionAlgorithm),
            with(&[("num_blocks", Value::Int64(2))]),
        ),
        (
            "kruskal",
            Box::new(KruskalAlgorithm),
            with(&[("weight", Value::from("weight"))]),
        ),
        (
            "prim",
            Box::new(PrimAlgorithm),
            with(&[("weight", Value::from("weight"))]),
        ),
        ("max_flow", Box::new(MaxFlowAlgorithm), flow()),
        ("min_cost_max_flow", Box::new(MinCostFlowAlgorithm), flow()),
        (
            "subgraph_isomorphism",
            Box::new(SubgraphIsomorphismAlgorithm),
            with(&[
                ("pattern_edges", Value::from("0-1,1-2")),
                ("pattern_nodes", Value::Int64(3)),
            ]),
        ),
        (
            "articulation_points",
            Box::new(ArticulationPointsAlgorithm),
            Parameters::new(),
        ),
        ("bridges", Box::new(BridgesAlgorithm), Parameters::new()),
        ("kcore", Box::new(KCoreAlgorithm), Parameters::new()),
        ("ktruss", Box::new(KTrussAlgorithm), Parameters::new()),
    ]
}

/// The rows of a result, in a fixed order, as text (floats exactly).
fn sorted_rows(rows: &[Vec<Value>]) -> Vec<String> {
    let mut rows: Vec<String> = rows.iter().map(|row| format!("{row:?}")).collect();
    rows.sort();
    rows
}

#[test]
fn every_algorithm_reads_only_the_visible_graph() {
    let clean = people();
    let dirty = people_with_leftovers();
    let mut differ = Vec::new();
    for (name, algorithm, params) in algorithms_under_test() {
        let expected = algorithm.execute(&clean, &params).unwrap();
        let actual = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            algorithm.execute(&dirty, &params)
        }));
        match actual {
            Ok(Ok(actual)) => {
                let (actual_rows, expected_rows) =
                    (sorted_rows(&actual.rows), sorted_rows(&expected.rows));
                if actual.columns != expected.columns || actual_rows != expected_rows {
                    differ.push(format!(
                        "{name}: {actual_rows:?} instead of {expected_rows:?}"
                    ));
                }
            }
            Ok(Err(error)) => differ.push(format!("{name}: failed with {error}")),
            Err(_) => differ.push(format!("{name}: panicked")),
        }
    }
    assert!(
        differ.is_empty(),
        "algorithms that read past the visible graph:\n{}",
        differ.join("\n")
    );
}

#[test]
fn traversals_skip_a_deleted_node_and_uncommitted_edges() {
    let dirty = people_with_leftovers();
    let alix = NodeId::new(ALIX);
    let layers = algorithms::bfs_layers(&dirty, alix);
    assert_eq!(
        layers,
        vec![
            vec![alix],
            vec![NodeId::new(GUS), NodeId::new(JULES)],
            vec![NodeId::new(2)],
            vec![NodeId::new(MIA)],
        ],
        "Mia is three hops away, Butch and Django are not reachable"
    );
    assert_eq!(algorithms::dfs(&dirty, alix).len(), 5);
    assert_eq!(
        algorithms::topological_sort(&dirty).map(|order| order.len()),
        Some(5),
        "the cycles through Butch and the uncommitted edge are not in the graph"
    );
}
