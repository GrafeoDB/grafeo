//! Graph algorithms give bit-identical results for the same graph, loaded in
//! the same order, whatever the process or thread count (#592). Each result is
//! reduced to a digest of its `(node id, value bits)` pairs in node-id order,
//! pinned below, so any change to an algorithm's output fails here and goes
//! into the CHANGELOG's "Result changes".

#![cfg(all(feature = "lpg", feature = "algos", feature = "parallel"))]

use grafeo_adapters::plugins::Parameters;
use grafeo_adapters::plugins::algorithms::{
    self, ArticulationPointsAlgorithm, BetweennessCentralityAlgorithm, BridgesAlgorithm,
    ClosenessCentralityAlgorithm, ClusteringCoefficientAlgorithm, ConnectedComponentsAlgorithm,
    DegreeCentralityAlgorithm, GraphAlgorithm, KCoreAlgorithm, KTrussAlgorithm,
    LabelPropagationAlgorithm, LouvainAlgorithm, PageRankAlgorithm,
    StronglyConnectedComponentsAlgorithm,
};
use grafeo_common::types::{NodeId, Value};
use grafeo_core::graph::lpg::LpgStore;

/// A seeded graph shaped like a code base: a repository holds modules, modules
/// hold files, files hold functions, and functions call each other, mostly
/// within their module, sometimes across, with a few shared utility hubs.
/// About `modules * 220` nodes, one connected component.
fn code_graph(modules: usize) -> LpgStore {
    let mut state: u64 = 0x9E37_79B9_7F4A_7C15;
    let mut next = |bound: usize| {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        usize::try_from(state % u64::try_from(bound).unwrap()).unwrap()
    };
    let store = LpgStore::new().unwrap();
    let repo = store.create_node(&["Repository"]);
    let mut functions_by_module: Vec<Vec<NodeId>> = Vec::new();
    for _ in 0..modules {
        // One epoch per module: with `tiered-storage` (on under the workspace's
        // --all-features), the records of one epoch must fit its arena chunk.
        // Ids, and so every result, do not depend on the epochs.
        store.new_epoch();
        let module = store.create_node(&["Module"]);
        store.create_edge(repo, module, "CONTAINS");
        let mut functions = Vec::new();
        for _ in 0..(6 + next(5)) {
            let file = store.create_node(&["File"]);
            store.create_edge(module, file, "CONTAINS");
            for _ in 0..(18 + next(14)) {
                let function = store.create_node(&["Function"]);
                store.create_edge(file, function, "CONTAINS");
                functions.push(function);
            }
        }
        functions_by_module.push(functions);
    }
    let hubs: Vec<NodeId> = functions_by_module.iter().map(|f| f[0]).take(8).collect();
    for (m, functions) in functions_by_module.iter().enumerate() {
        store.new_epoch();
        for &caller in functions {
            for _ in 0..next(4) {
                let callee = if next(10) == 0 {
                    let other = &functions_by_module[next(modules)];
                    other[next(other.len())]
                } else {
                    functions[next(functions.len())]
                };
                store.create_edge(caller, callee, "CALLS");
            }
            if next(6) == 0 {
                store.create_edge(caller, hubs[next(hubs.len())], "CALLS");
            }
        }
        let _ = m;
    }
    store
}

/// FNV-1a over the bytes, stable across platforms and processes.
fn fnv1a(bytes: &[u8]) -> u64 {
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    for byte in bytes {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x0100_0000_01b3);
    }
    hash
}

fn value_bits(value: &Value) -> u64 {
    match value {
        Value::Float64(f) => f.to_bits(),
        Value::Int64(i) => u64::from_ne_bytes(i.to_ne_bytes()),
        Value::Bool(b) => u64::from(*b),
        other => panic!("unexpected value {other:?}"),
    }
}

/// The digest of a procedure's rows, as they come (rows are in node-id order,
/// which `rows_in_node_order` checks separately).
fn rows_digest(rows: &[Vec<Value>]) -> u64 {
    let mut bytes = Vec::new();
    for row in rows {
        for value in row {
            bytes.extend_from_slice(&value_bits(value).to_le_bytes());
        }
    }
    fnv1a(&bytes)
}

fn run(algorithm: &dyn GraphAlgorithm, store: &LpgStore, params: &Parameters) -> Vec<Vec<Value>> {
    algorithm.execute(store, params).unwrap().rows
}

fn with(pairs: &[(&str, Value)]) -> Parameters {
    let mut params = Parameters::new();
    for (name, value) in pairs {
        match value {
            Value::Bool(b) => params.set_bool(*name, *b),
            Value::Int64(i) => params.set_int(*name, *i),
            Value::Float64(f) => params.set_float(*name, *f),
            other => panic!("unexpected parameter {other:?}"),
        }
    }
    params
}

/// Every procedure under test, with its parameters, on the large graph.
fn procedures() -> Vec<(&'static str, Box<dyn GraphAlgorithm>, Parameters)> {
    vec![
        ("pagerank", Box::new(PageRankAlgorithm), Parameters::new()),
        (
            "pagerank_undirected",
            Box::new(PageRankAlgorithm),
            with(&[("directed", Value::Bool(false))]),
        ),
        ("louvain", Box::new(LouvainAlgorithm), Parameters::new()),
        ("kcore", Box::new(KCoreAlgorithm), Parameters::new()),
        (
            "articulation_points",
            Box::new(ArticulationPointsAlgorithm),
            Parameters::new(),
        ),
        ("bridges", Box::new(BridgesAlgorithm), Parameters::new()),
        (
            "degree_centrality",
            Box::new(DegreeCentralityAlgorithm),
            Parameters::new(),
        ),
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
            "label_propagation",
            Box::new(LabelPropagationAlgorithm),
            Parameters::new(),
        ),
        (
            "clustering_coefficient",
            Box::new(ClusteringCoefficientAlgorithm),
            with(&[("parallel", Value::Bool(true))]),
        ),
        (
            "clustering_coefficient_sequential",
            Box::new(ClusteringCoefficientAlgorithm),
            with(&[("parallel", Value::Bool(false))]),
        ),
        ("ktruss", Box::new(KTrussAlgorithm), Parameters::new()),
    ]
}

/// Betweenness and closeness are O(V * E): pinned on a smaller graph.
fn slow_procedures() -> Vec<(&'static str, Box<dyn GraphAlgorithm>, Parameters)> {
    vec![
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
    ]
}

const LARGE_MODULES: usize = 45;
const SMALL_MODULES: usize = 4;

/// The digest of every procedure, in a fixed order.
fn all_digests() -> Vec<(&'static str, u64)> {
    let large = code_graph(LARGE_MODULES);
    let small = code_graph(SMALL_MODULES);
    let mut digests: Vec<(&'static str, u64)> = procedures()
        .iter()
        .map(|(name, algorithm, params)| {
            (*name, rows_digest(&run(algorithm.as_ref(), &large, params)))
        })
        .collect();
    digests.extend(slow_procedures().iter().map(|(name, algorithm, params)| {
        (*name, rows_digest(&run(algorithm.as_ref(), &small, params)))
    }));
    digests
}

/// The digests of the current algorithms. A change here is a result change:
/// update the digest only together with a "Result changes" CHANGELOG entry.
const GOLDEN: &[(&str, u64)] = &[
    ("pagerank", 0x3abf500dc241110f),
    ("pagerank_undirected", 0x7fca26444151927d),
    ("louvain", 0x7b5f451cbe0602dd),
    ("kcore", 0x74ded8d85df2d82c),
    ("articulation_points", 0x9115d9fcb91b55d6),
    ("bridges", 0x7ef97eaf8d068b04),
    ("degree_centrality", 0x235f9a3dc80c2d3b),
    ("connected_components", 0xaa211db929fb3293),
    ("strongly_connected_components", 0xcfb40e420e8cba43),
    ("label_propagation", 0x7466bf931f2950f6),
    // 0.6.0: clustering and k-truss read the simple graph, the self-loops of
    // recursive functions left out.
    ("clustering_coefficient", 0x2e12f21fc91f0237),
    ("clustering_coefficient_sequential", 0x2e12f21fc91f0237),
    ("ktruss", 0xa60d0757ba8bd32a),
    ("betweenness_centrality", 0x42bccf91788a07a2),
    ("closeness_centrality", 0x6474cd4c532e5eac),
];

#[test]
fn the_graph_is_one_code_base_sized_component() {
    let store = code_graph(LARGE_MODULES);
    let n = store.node_ids().len();
    assert!((9_000..=11_000).contains(&n), "{n} nodes");
    let components = algorithms::connected_components(&store);
    let distinct: std::collections::BTreeSet<u64> = components.values().copied().collect();
    assert_eq!(distinct.len(), 1);
}

#[test]
fn rows_come_in_node_id_order() {
    let store = code_graph(SMALL_MODULES);
    for (name, algorithm, params) in procedures().iter().chain(slow_procedures().iter()) {
        let rows = run(algorithm.as_ref(), &store, params);
        let keys: Vec<(i64, i64)> = rows
            .iter()
            .map(|row| {
                let first = match row[0] {
                    Value::Int64(i) => i,
                    ref other => panic!("{name}: first column {other:?}"),
                };
                // Edge results (bridges, k-truss) are ordered by (source, target).
                let second = match row.get(1) {
                    Some(Value::Int64(i)) if *name == "bridges" || *name == "ktruss" => *i,
                    _ => 0,
                };
                (first, second)
            })
            .collect();
        assert!(
            keys.windows(2).all(|w| w[0] < w[1]),
            "{name}: rows not in node-id order: {:?}",
            &keys[..keys.len().min(8)]
        );
    }
}

#[test]
fn results_match_the_pinned_digests() {
    let digests = all_digests();
    for (name, digest) in &digests {
        eprintln!("DIGEST (\"{name}\", 0x{digest:016x}),");
    }
    assert_eq!(digests.len(), GOLDEN.len(), "pin the digests printed above");
    for ((name, digest), (golden_name, golden)) in digests.iter().zip(GOLDEN) {
        assert_eq!(name, golden_name);
        assert_eq!(
            digest, golden,
            "{name}: result changed (0x{digest:016x}, pinned 0x{golden:016x})"
        );
    }
}

#[test]
fn results_do_not_depend_on_the_thread_count() {
    let in_pool = |threads: usize| {
        rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap()
            .install(all_digests)
    };
    let (one, eight) = (in_pool(1), in_pool(8));
    let differing: Vec<&str> = one
        .iter()
        .zip(&eight)
        .filter(|((_, a), (_, b))| a != b)
        .map(|((name, _), _)| *name)
        .collect();
    assert!(
        differing.is_empty(),
        "differ between 1 and 8 threads: {differing:?}"
    );
}

const CHILD_VAR: &str = "GRAFEO_ALGORITHM_DETERMINISM_CHILD";

#[test]
fn results_are_the_same_in_another_process() {
    let output = std::process::Command::new(std::env::current_exe().unwrap())
        .args(["--exact", "digests_child", "--nocapture"])
        .env(CHILD_VAR, "1")
        .output()
        .unwrap();
    assert!(output.status.success(), "child failed");
    let stdout = String::from_utf8(output.stdout).unwrap();
    let child: Vec<String> = stdout
        .lines()
        .filter_map(|line| line.strip_prefix("CHILD "))
        .map(str::to_string)
        .collect();
    let here: Vec<String> = all_digests()
        .iter()
        .map(|(name, digest)| format!("{name} {digest:016x}"))
        .collect();
    assert_eq!(child, here);
}

/// Child-process entry for `results_are_the_same_in_another_process`; a no-op
/// when run directly.
#[test]
fn digests_child() {
    if std::env::var_os(CHILD_VAR).is_none() {
        return;
    }
    for (name, digest) in all_digests() {
        println!("CHILD {name} {digest:016x}");
    }
}

/// Neumaier-compensated sum, in the order given.
fn neumaier(values: impl IntoIterator<Item = f64>) -> f64 {
    let (mut sum, mut compensation) = (0.0_f64, 0.0_f64);
    for value in values {
        let t = sum + value;
        if sum.abs() >= value.abs() {
            compensation += (sum - t) + value;
        } else {
            compensation += (value - t) + sum;
        }
        sum = t;
    }
    sum + compensation
}

/// The average clustering coefficient sums the per-node coefficients in node-id
/// order with compensation, whatever the call, the thread count or the path.
#[test]
fn the_average_clustering_coefficient_is_the_same_every_time() {
    let store = code_graph(LARGE_MODULES);
    let local = algorithms::local_clustering_coefficient(&store);
    let mut by_node: Vec<(NodeId, f64)> = local.into_iter().collect();
    by_node.sort_unstable_by_key(|&(node, _)| node);
    let n = by_node.len() as f64;
    let expected = (neumaier(by_node.iter().map(|&(_, c)| c)) / n).to_bits();

    for _ in 0..5 {
        assert_eq!(
            algorithms::global_clustering_coefficient(&store).to_bits(),
            expected
        );
        assert_eq!(
            algorithms::clustering_coefficient(&store)
                .global_coefficient
                .to_bits(),
            expected
        );
    }
    for threads in [1, 8] {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap();
        let parallel = pool.install(|| algorithms::clustering_coefficient_parallel(&store, 1));
        assert_eq!(
            parallel.global_coefficient.to_bits(),
            expected,
            "{threads} threads"
        );
    }
}

/// The Rust functions that return lists of nodes or edges return them in
/// node-id order (edges by source, then target).
#[test]
fn rust_lists_come_in_node_id_order() {
    let store = code_graph(SMALL_MODULES);
    let strictly_increasing = |ids: &[NodeId]| ids.windows(2).all(|w| w[0] < w[1]);
    let decomposition = algorithms::kcore_decomposition(&store);
    for k in 0..=decomposition.max_core {
        assert!(strictly_increasing(&decomposition.k_core(k)), "k_core({k})");
        assert!(
            strictly_increasing(&decomposition.k_shell(k)),
            "k_shell({k})"
        );
    }
    assert!(strictly_increasing(&algorithms::k_core(&store, 2)));
    let bridges = algorithms::bridges(&store);
    assert!(
        bridges.len() >= 2,
        "the graph needs bridges to check their order"
    );
    assert!(bridges.windows(2).all(|w| w[0] < w[1]), "bridges");
    let truss = algorithms::k_truss(&store, 3);
    assert!(truss.windows(2).all(|w| w[0] < w[1]), "k_truss");
}
