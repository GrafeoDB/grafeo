//! `grafeo.ktruss`, `grafeo.total_triangles` and `grafeo.subgraph_isomorphism`, registered in
//! 0.6.2, answer as the documented reading says: the simple undirected graph (direction ignored,
//! parallel edges once, self-loops no edge), each edge's truss number the largest k whose k-truss
//! holds it, each triangle once, and a pattern count of every non-induced mapping (each distinct
//! copy once per automorphism of the pattern).
//!
//! Two oracles. Brute forces in this file, written from the definitions, run on every graph; on
//! the known graphs below they are checked against networkx 3.7 (`k_truss`, `triangles` and
//! `GraphMatcher.subgraph_monomorphisms_iter`), whose answers the table holds, so no Python runs
//! here. The same brute forces agreed with networkx on 300 seeded graphs when the table was made.
//!
//! ```bash
//! cargo test -p grafeo-engine --all-features --test main structure_procedures::
//! ```

#![cfg(all(feature = "gql", feature = "cypher", feature = "algos"))]

use std::collections::{BTreeMap, BTreeSet, HashMap};

use grafeo_common::types::{NodeId, Value};
use grafeo_engine::GrafeoDB;

/// The patterns: name, `pattern_edges`, `pattern_nodes`.
const PATTERNS: [(&str, &str, usize); 6] = [
    ("an edge", "0-1", 2),
    ("a path of three", "0-1,1-2", 3),
    ("a triangle", "0-1,1-2,2-0", 3),
    ("a 4-cycle", "0-1,1-2,2-3,3-0", 4),
    ("a star of three", "0-1,0-2,0-3", 4),
    ("an edge and an isolated node", "0-1", 3),
];

/// A graph with networkx's answers: the triangle count, how many edges have each truss number,
/// and the count of each of [`PATTERNS`].
struct Known {
    name: &'static str,
    nodes: usize,
    edges: &'static [(usize, usize)],
    triangles: i64,
    truss_histogram: &'static [(i64, usize)],
    pattern_counts: [i64; 6],
}

const KNOWN: &[Known] = &[
    Known {
        name: "a clique of five",
        nodes: 5,
        edges: &[
            (0, 1),
            (0, 2),
            (0, 3),
            (0, 4),
            (1, 2),
            (1, 3),
            (1, 4),
            (2, 3),
            (2, 4),
            (3, 4),
        ],
        triangles: 10,
        truss_histogram: &[(5, 10)],
        pattern_counts: [20, 60, 60, 120, 120, 60],
    },
    Known {
        name: "two 4-cliques sharing a node",
        nodes: 7,
        edges: &[
            (0, 1),
            (0, 2),
            (0, 3),
            (1, 2),
            (1, 3),
            (2, 3),
            (3, 4),
            (3, 5),
            (3, 6),
            (4, 5),
            (4, 6),
            (5, 6),
        ],
        triangles: 8,
        truss_histogram: &[(4, 12)],
        pattern_counts: [24, 66, 48, 48, 156, 120],
    },
    Known {
        name: "a tree",
        nodes: 7,
        edges: &[(0, 1), (0, 2), (1, 3), (1, 4), (2, 5), (2, 6)],
        triangles: 0,
        truss_histogram: &[(2, 6)],
        pattern_counts: [12, 14, 0, 0, 12, 60],
    },
    Known {
        name: "both directions, a parallel edge and self-loops",
        nodes: 4,
        edges: &[
            (0, 1),
            (1, 0),
            (1, 2),
            (1, 2),
            (2, 0),
            (2, 2),
            (3, 3),
            (2, 3),
        ],
        triangles: 1,
        truss_histogram: &[(2, 1), (3, 3)],
        pattern_counts: [8, 10, 6, 0, 6, 16],
    },
    Known {
        name: "a seeded random graph",
        nodes: 18,
        edges: &[
            (1, 15),
            (2, 1),
            (3, 11),
            (3, 12),
            (4, 0),
            (4, 2),
            (4, 6),
            (4, 12),
            (4, 14),
            (4, 17),
            (5, 10),
            (6, 2),
            (6, 7),
            (7, 4),
            (7, 7),
            (7, 8),
            (7, 17),
            (8, 5),
            (8, 10),
            (8, 11),
            (8, 17),
            (9, 4),
            (10, 11),
            (11, 7),
            (11, 9),
            (11, 11),
            (11, 14),
            (13, 7),
            (16, 1),
            (16, 3),
            (17, 4),
            (17, 9),
        ],
        triangles: 8,
        truss_histogram: &[(2, 12), (3, 17)],
        pattern_counts: [58, 192, 48, 96, 696, 928],
    },
    Known {
        name: "no edges",
        nodes: 3,
        edges: &[],
        triangles: 0,
        truss_histogram: &[],
        pattern_counts: [0, 0, 0, 0, 0, 0],
    },
];

// ---------------------------------------------------------------------------
// Graphs
// ---------------------------------------------------------------------------

/// Directed edges with self-loops and repeats: each ordered pair, the loops included, with a
/// chance of `per_mille` in a thousand. The table's seeded random graph is `random_graph(0x5EED_0001,
/// 18, 120)`, which pins this generator to the one the networkx answers were made with.
fn random_graph(seed: u64, nodes: usize, per_mille: u64) -> Vec<(usize, usize)> {
    let mut state = seed;
    let mut below = |bound: u64| {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        state % bound
    };
    let mut edges = Vec::new();
    for u in 0..nodes {
        for v in 0..nodes {
            if below(1000) < per_mille {
                edges.push((u, v));
            }
        }
    }
    edges
}

/// The graphs every test runs on: the known ones and seeded random ones of up to `max_nodes`.
fn graphs(max_nodes: usize) -> Vec<(String, usize, Vec<(usize, usize)>)> {
    let mut all: Vec<_> = KNOWN
        .iter()
        .map(|known| (known.name.to_string(), known.nodes, known.edges.to_vec()))
        .collect();
    all.push(("the empty graph".to_string(), 0, Vec::new()));
    all.push(("one node".to_string(), 1, Vec::new()));
    all.push(("one self-loop".to_string(), 1, vec![(0, 0)]));
    for trial in 0..40u64 {
        let nodes = 2 + usize::try_from(trial).unwrap() % (max_nodes - 1);
        let per_mille = [80, 200, 350, 550][usize::try_from(trial % 4).unwrap()];
        all.push((
            format!("random graph {trial} ({nodes} nodes, {per_mille}/1000)"),
            nodes,
            random_graph(0xA11C_E000 + trial, nodes, per_mille),
        ));
    }
    all
}

/// The database of a graph, and the index of each node id.
fn build(nodes: usize, edges: &[(usize, usize)]) -> (GrafeoDB, HashMap<i64, usize>) {
    let db = GrafeoDB::new_in_memory();
    let ids: Vec<NodeId> = (0..nodes)
        .map(|_| db.create_node(&["V"]).unwrap())
        .collect();
    for &(u, v) in edges {
        db.create_edge(ids[u], ids[v], "E").unwrap();
    }
    let index = ids
        .iter()
        .enumerate()
        .map(|(i, id)| (i64::try_from(id.as_u64()).unwrap(), i))
        .collect();
    (db, index)
}

fn rows(db: &GrafeoDB, language: &str, query: &str) -> Vec<Vec<Value>> {
    db.execute_language(query, language, None)
        .unwrap_or_else(|error| panic!("{language} `{query}`: {error}"))
        .rows()
        .to_vec()
}

/// A call in GQL and in Cypher, the Cypher one returning the yielded columns in order.
fn both(call: &str, columns: &[&str]) -> [(&'static str, String); 2] {
    let yields = columns.join(", ");
    [
        ("gql", format!("CALL {call} YIELD {yields}")),
        (
            "cypher",
            format!("CALL {call} YIELD {yields} RETURN {yields}"),
        ),
    ]
}

fn int(value: &Value) -> i64 {
    match value {
        Value::Int64(n) => *n,
        other => panic!("an integer, not {other:?}"),
    }
}

// ---------------------------------------------------------------------------
// The brute forces, from the definitions
// ---------------------------------------------------------------------------

/// The simple undirected graph: no direction, each pair once, no self-loops.
fn adjacency(nodes: usize, edges: &[(usize, usize)]) -> Vec<BTreeSet<usize>> {
    let mut adjacent = vec![BTreeSet::new(); nodes];
    for &(u, v) in edges {
        if u != v {
            adjacent[u].insert(v);
            adjacent[v].insert(u);
        }
    }
    adjacent
}

/// The truss number of each edge `(low, high)`: the largest k from 2 for which the edge is in the
/// k-truss, the largest subgraph whose every edge closes at least k - 2 triangles inside it.
fn brute_truss(nodes: usize, edges: &[(usize, usize)]) -> BTreeMap<(usize, usize), i64> {
    let adjacent = adjacency(nodes, edges);
    let mut all = BTreeSet::new();
    for (u, neighbors) in adjacent.iter().enumerate() {
        all.extend(neighbors.iter().filter(|&&v| u < v).map(|&v| (u, v)));
    }
    let mut numbers: BTreeMap<(usize, usize), i64> = all.iter().map(|&e| (e, 2)).collect();
    let pair = |a: usize, b: usize| (a.min(b), a.max(b));
    for k in 3.. {
        let mut live = all.clone();
        loop {
            let drop: Vec<(usize, usize)> = live
                .iter()
                .copied()
                .filter(|&(u, v)| {
                    let support = (0..nodes)
                        .filter(|&w| live.contains(&pair(u, w)) && live.contains(&pair(v, w)))
                        .count();
                    support + 2 < k
                })
                .collect();
            if drop.is_empty() {
                break;
            }
            for edge in drop {
                live.remove(&edge);
            }
        }
        if live.is_empty() {
            break;
        }
        for edge in live {
            numbers.insert(edge, i64::try_from(k).unwrap());
        }
    }
    numbers
}

fn brute_triangles(nodes: usize, edges: &[(usize, usize)]) -> i64 {
    let adjacent = adjacency(nodes, edges);
    let mut count = 0;
    for a in 0..nodes {
        for b in (a + 1)..nodes {
            for c in (b + 1)..nodes {
                if adjacent[a].contains(&b) && adjacent[a].contains(&c) && adjacent[b].contains(&c)
                {
                    count += 1;
                }
            }
        }
    }
    count
}

fn parse_pattern(text: &str) -> Vec<(usize, usize)> {
    text.split(',')
        .filter(|part| !part.is_empty())
        .map(|part| {
            let (u, v) = part.split_once('-').unwrap();
            (u.parse().unwrap(), v.parse().unwrap())
        })
        .collect()
}

/// Every injective map of the pattern's nodes into the graph's that sends each pattern edge to an
/// edge (non-induced: an extra edge between the images is allowed). A pattern self-loop needs a
/// self-loop at its image (the 2026-11-06 ruling); a target self-loop the pattern lacks is allowed.
fn brute_pattern(
    nodes: usize,
    edges: &[(usize, usize)],
    pattern: &[(usize, usize)],
    pattern_nodes: usize,
) -> i64 {
    struct Target {
        adjacent: Vec<BTreeSet<usize>>,
        looped: BTreeSet<usize>,
    }
    fn extend(
        target: &Target,
        pattern: &[(usize, usize)],
        pattern_nodes: usize,
        image: &mut Vec<usize>,
    ) -> i64 {
        if image.len() == pattern_nodes {
            let maps = pattern.iter().all(|&(u, v)| {
                if u == v {
                    target.looped.contains(&image[u])
                } else {
                    target.adjacent[image[u]].contains(&image[v])
                }
            });
            return i64::from(maps);
        }
        let mut count = 0;
        for node in 0..target.adjacent.len() {
            if !image.contains(&node) {
                image.push(node);
                count += extend(target, pattern, pattern_nodes, image);
                image.pop();
            }
        }
        count
    }
    let target = Target {
        adjacent: adjacency(nodes, edges),
        looped: edges
            .iter()
            .filter(|(u, v)| u == v)
            .map(|&(u, _)| u)
            .collect(),
    };
    extend(&target, pattern, pattern_nodes, &mut Vec::new())
}

// ---------------------------------------------------------------------------
// What the procedures give
// ---------------------------------------------------------------------------

/// `grafeo.ktruss()` as `(low index, high index) -> truss number`, checking the row shape: one row
/// per edge, `source < target`, in id order.
fn ktruss(
    db: &GrafeoDB,
    index: &HashMap<i64, usize>,
    language: &str,
    query: &str,
) -> BTreeMap<(usize, usize), i64> {
    let found = rows(db, language, query);
    let ids: Vec<(i64, i64)> = found.iter().map(|r| (int(&r[0]), int(&r[1]))).collect();
    assert!(
        ids.iter().all(|(source, target)| source < target),
        "{language}: source < target on every row: {ids:?}"
    );
    assert!(
        ids.windows(2).all(|w| w[0] < w[1]),
        "{language}: rows in id order, each edge once: {ids:?}"
    );
    found
        .iter()
        .map(|row| {
            let (u, v) = (index[&int(&row[0])], index[&int(&row[1])]);
            let number = row.get(2).map_or(0, int);
            ((u.min(v), u.max(v)), number)
        })
        .collect()
}

fn one(db: &GrafeoDB, language: &str, query: &str) -> i64 {
    let found = rows(db, language, query);
    assert_eq!(found.len(), 1, "{language} `{query}`: one row");
    int(&found[0][0])
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

/// The brute forces give networkx's answers on the known graphs, and the generator still makes the
/// table's seeded graph.
#[test]
fn the_oracles_agree_with_networkx_on_the_known_graphs() {
    assert_eq!(
        random_graph(0x5EED_0001, 18, 120),
        KNOWN[4].edges,
        "the generator the table was made with"
    );
    for known in KNOWN {
        let truss = brute_truss(known.nodes, known.edges);
        let mut histogram: BTreeMap<i64, usize> = BTreeMap::new();
        for number in truss.values() {
            *histogram.entry(*number).or_default() += 1;
        }
        assert_eq!(
            histogram.into_iter().collect::<Vec<_>>(),
            known.truss_histogram,
            "{}: truss numbers",
            known.name
        );
        assert_eq!(
            brute_triangles(known.nodes, known.edges),
            known.triangles,
            "{}: triangles",
            known.name
        );
        for ((pattern, edges, pattern_nodes), want) in PATTERNS.iter().zip(known.pattern_counts) {
            assert_eq!(
                brute_pattern(
                    known.nodes,
                    known.edges,
                    &parse_pattern(edges),
                    *pattern_nodes
                ),
                want,
                "{}: {pattern}",
                known.name
            );
        }
    }
}

#[test]
fn ktruss_gives_each_edge_its_truss_number() {
    for (name, nodes, edges) in graphs(14) {
        let (db, index) = build(nodes, &edges);
        let want = brute_truss(nodes, &edges);
        for (language, query) in both("grafeo.ktruss()", &["source", "target", "truss_number"]) {
            assert_eq!(
                ktruss(&db, &index, language, &query),
                want,
                "{name}, {language}"
            );
        }
    }
}

/// `grafeo.ktruss(k)` is the decomposition's edges with a truss number of at least k, in the same
/// order, for every k from 0 to one past the largest.
#[test]
fn ktruss_at_k_is_the_decomposition_filtered_at_k() {
    for (name, nodes, edges) in graphs(12) {
        let (db, index) = build(nodes, &edges);
        let want = brute_truss(nodes, &edges);
        let largest = want.values().copied().max().unwrap_or(0);
        for k in 0..=largest + 1 {
            let filtered: BTreeMap<(usize, usize), i64> = want
                .iter()
                .filter(|&(_, &number)| number >= k)
                .map(|(&edge, _)| (edge, 0))
                .collect();
            for (language, query) in both(&format!("grafeo.ktruss({k})"), &["source", "target"]) {
                assert_eq!(
                    ktruss(&db, &index, language, &query),
                    filtered,
                    "{name}, k = {k}, {language}"
                );
            }
        }
    }
}

/// Each triangle once, on the sequential and the parallel path, on either side of the threshold.
#[test]
fn total_triangles_counts_each_triangle_once() {
    let mut all = graphs(14);
    // Past the default threshold of 50 nodes, so the default call takes the parallel path.
    for trial in 0..3u64 {
        let nodes = 80 + usize::try_from(trial).unwrap() * 20;
        all.push((
            format!("large random graph {trial}"),
            nodes,
            random_graph(0xB16_0000 + trial, nodes, 60),
        ));
    }
    for (name, nodes, edges) in all {
        let (db, _) = build(nodes, &edges);
        let want = brute_triangles(nodes, &edges);
        for call in [
            "grafeo.total_triangles()",
            "grafeo.total_triangles(false)",
            "grafeo.total_triangles(true, 0)",
            "grafeo.total_triangles(true, 1000000)",
        ] {
            for (language, query) in both(call, &["total_triangles"]) {
                assert_eq!(
                    one(&db, language, &query),
                    want,
                    "{name}, {call}, {language}"
                );
            }
        }
    }
}

/// Patterns with self-loops and a duplicate edge, for the sweep (the brute force agreed with
/// networkx's monomorphism matcher on these over 300 seeded graphs).
const LOOP_PATTERNS: [(&str, &str, usize); 5] = [
    ("a loop", "0-0", 1),
    ("a loop and an edge", "0-0,0-1", 2),
    ("a duplicate edge", "0-1,0-1", 2),
    ("two loops", "0-0,1-1", 2),
    ("a looped triangle", "0-0,0-1,1-2,2-0", 3),
];

#[test]
fn subgraph_isomorphism_counts_every_mapping() {
    for (name, nodes, edges) in graphs(9) {
        let (db, _) = build(nodes, &edges);
        for (pattern, pattern_edges, pattern_nodes) in PATTERNS.into_iter().chain(LOOP_PATTERNS) {
            let want = brute_pattern(nodes, &edges, &parse_pattern(pattern_edges), pattern_nodes);
            let call = format!("grafeo.subgraph_isomorphism('{pattern_edges}', {pattern_nodes})");
            for (language, query) in both(&call, &["count"]) {
                assert_eq!(
                    one(&db, language, &query),
                    want,
                    "{name}, {pattern}, {language}"
                );
            }
        }
    }
}

/// A triangle has six automorphisms, so the triangle pattern counts six per triangle.
#[test]
fn the_triangle_pattern_counts_six_per_triangle() {
    for (name, nodes, edges) in graphs(12) {
        let (db, _) = build(nodes, &edges);
        let triangles = both("grafeo.total_triangles()", &["total_triangles"]);
        let patterns = both("grafeo.subgraph_isomorphism('0-1,1-2,2-0', 3)", &["count"]);
        for ((language, triangles), (_, pattern)) in triangles.into_iter().zip(patterns) {
            assert_eq!(
                one(&db, language, &pattern),
                6 * one(&db, language, &triangles),
                "{name}, {language}"
            );
        }
    }
}

/// The count is non-induced: a path of three nodes maps onto a triangle six ways, although no
/// three nodes of a triangle induce a path.
#[test]
fn a_path_of_three_on_a_triangle_counts_six() {
    let (db, _) = build(3, &[(0, 1), (1, 2), (2, 0)]);
    for (language, query) in both("grafeo.subgraph_isomorphism('0-1,1-2', 3)", &["count"]) {
        assert_eq!(one(&db, language, &query), 6, "{language}");
    }
}

/// A pattern self-loop matches only a node with a self-loop (the 2026-11-06 ruling), with
/// networkx's monomorphism counts: on the triangle with a tail whose nodes 2 and 3 carry loops, on
/// a triangle with one looped node, and on a triangle with none. A target loop the pattern lacks
/// changes nothing, and a duplicate pattern edge counts once.
#[test]
fn a_pattern_self_loop_needs_a_target_self_loop() {
    let graphs: [(&str, usize, &[(usize, usize)], [i64; 6]); 3] = [
        (
            "two looped nodes",
            4,
            &[
                (0, 1),
                (1, 0),
                (1, 2),
                (1, 2),
                (2, 0),
                (2, 2),
                (3, 3),
                (2, 3),
            ],
            [2, 4, 8, 8, 2, 2],
        ),
        (
            "one looped node",
            3,
            &[(0, 1), (1, 2), (2, 0), (0, 0)],
            [1, 2, 6, 6, 0, 2],
        ),
        ("no loops", 3, &[(0, 1), (1, 2), (2, 0)], [0, 0, 6, 6, 0, 0]),
    ];
    let patterns = [
        ("0-0", 1),
        ("0-0,0-1", 2),
        ("0-1,0-1", 2),
        ("0-1", 2),
        ("0-0,1-1", 2),
        ("0-0,0-1,1-2,2-0", 3),
    ];
    for (name, nodes, edges, counts) in graphs {
        let (db, _) = build(nodes, edges);
        for ((pattern, pattern_nodes), want) in patterns.into_iter().zip(counts) {
            assert_eq!(
                brute_pattern(nodes, edges, &parse_pattern(pattern), pattern_nodes),
                want,
                "the oracle, {name}, {pattern}"
            );
            let call = format!("grafeo.subgraph_isomorphism('{pattern}', {pattern_nodes})");
            for (language, query) in both(&call, &["count"]) {
                assert_eq!(
                    one(&db, language, &query),
                    want,
                    "{name}, {pattern}, {language}"
                );
            }
        }
    }
}

/// A pattern with more nodes than the graph has no mapping, edges or not.
#[test]
fn a_pattern_larger_than_the_graph_counts_zero() {
    let (db, _) = build(3, &[(0, 1), (1, 2), (2, 0)]);
    for call in [
        "grafeo.subgraph_isomorphism('0-1,1-2,2-3,3-0', 4)",
        "grafeo.subgraph_isomorphism('0-1', 4)",
        "grafeo.subgraph_isomorphism('', 4)",
    ] {
        for (language, query) in both(call, &["count"]) {
            assert_eq!(one(&db, language, &query), 0, "{call}, {language}");
        }
    }
}

/// The three are listed with their columns, as the docs give them.
#[test]
fn the_procedures_are_listed_with_their_columns() {
    let db = GrafeoDB::new_in_memory();
    let listed: HashMap<String, String> = rows(
        &db,
        "gql",
        "CALL grafeo.procedures() YIELD name, output_columns",
    )
    .into_iter()
    .map(|row| (row[0].to_string(), row[1].to_string()))
    .collect();
    for (name, columns) in [
        ("grafeo.ktruss", &["source", "target", "truss_number"][..]),
        ("grafeo.total_triangles", &["total_triangles"][..]),
        ("grafeo.subgraph_isomorphism", &["count"][..]),
    ] {
        let found = listed
            .iter()
            .find(|(key, _)| key.contains(name))
            .unwrap_or_else(|| panic!("{name} is not listed"));
        for column in columns {
            assert!(found.1.contains(column), "{name}: {column} in {}", found.1);
        }
    }
}
