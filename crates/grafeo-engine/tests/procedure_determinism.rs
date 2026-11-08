//! Every registered procedure gives the same answer for the same graph (#592), keyed on the
//! procedure catalogue so that a new procedure fails here until it has a row. The hand-picked
//! list in `grafeo-adapters/tests/algorithm_determinism.rs` had no `topological_sort`, which gave
//! a different order on every run (0.6.1 and up to 0.6.2's step 3).
//!
//! Two properties:
//! - **The same graph, loaded the same way:** two runs, and a second database built the same way,
//!   give the very same rows, floats to the bit.
//! - **The same graph, its edges loaded in another order:** the nodes as before, so every id is
//!   the same, the edges reversed. Every answer that the node ids decide is the same, a
//!   topological order included (the smallest by node id, Kahn's algorithm with a min-heap, as
//!   networkx's `lexicographical_topological_sort`).
//! - **The same graph, loaded in another order:** node and edge insertion reversed, so every id
//!   differs. Where the answer is unique, it is the same up to the node names: a value per node, a
//!   partition up to its labels, a total. Where the algorithm breaks ties by node id or depends on
//!   the order it meets nodes (a topological order, depth-first times, label propagation, Louvain),
//!   the row says so, and the answer must still be valid.
//!
//! A cycle gives no topological order: an error or no rows, never a row of text.
//!
//! ```bash
//! cargo test -p grafeo-engine --all-features --test main procedure_determinism::
//! ```

#![cfg(all(feature = "algos", feature = "vector-index", feature = "text-index"))]

use std::collections::{BTreeMap, BTreeSet, HashMap};

use grafeo_common::types::Value;
use grafeo_engine::GrafeoDB;
use grafeo_engine::procedures::BuiltinProcedures;

/// How a procedure's answer compares across insertion orders.
enum Across {
    /// The answer is unique: every row, ids read as node names, floats to 1e-9 relative.
    Rows,
    /// As `Rows`, for an answer whose edges are undirected: each row's `source` and `target` compare
    /// as an unordered pair, since the procedure names an edge's ends by node id (`source <
    /// target`) and a load in another order gives the ends other ids.
    UndirectedRows,
    /// Only these columns are unique (a total of a flow whose edges may differ).
    Columns(&'static [&'static str]),
    /// A partition of the nodes into the column's ids: the same groups, whatever their numbers.
    Partition(&'static str),
    /// Decided by the node ids: the same while the ids are, another when they change; the reason.
    ById(&'static str),
    /// The answer depends on the order the algorithm meets nodes; the reason.
    OrderDependent(&'static str),
}

/// The row of each procedure of the catalogue; a new procedure fails here until it has one.
fn across(name: &str) -> Across {
    match name {
        "grafeo.articulation_points"
        | "grafeo.bellman_ford"
        | "grafeo.betweenness_centrality"
        | "grafeo.bfs"
        | "grafeo.bridges"
        | "grafeo.closeness_centrality"
        | "grafeo.clustering_coefficient"
        | "grafeo.degree_centrality"
        | "grafeo.dijkstra"
        | "grafeo.floyd_warshall"
        | "grafeo.kcore"
        | "grafeo.kruskal"
        | "grafeo.prim"
        | "grafeo.pagerank"
        | "grafeo.sssp"
        | "grafeo.labels"
        | "grafeo.propertyKeys"
        | "grafeo.relationshipTypes"
        | "grafeo.search.vector"
        | "grafeo.search.text"
        | "grafeo.search.mmr"
        | "grafeo.total_triangles"
        | "grafeo.subgraph_isomorphism" => Across::Rows,
        "grafeo.ktruss" => Across::UndirectedRows,
        "grafeo.max_flow" => Across::Columns(&["max_flow"]),
        "grafeo.min_cost_max_flow" => Across::Columns(&["max_flow", "total_cost"]),
        "grafeo.connected_components" | "grafeo.strongly_connected_components" => {
            Across::Partition("component_id")
        }
        "grafeo.topological_sort" => {
            Across::ById("the smallest order by node id; checked to be valid in every load")
        }
        "grafeo.dfs" => Across::OrderDependent("discovery and finish times follow the edge order"),
        "grafeo.label_propagation" | "grafeo.louvain" => {
            Across::OrderDependent("a local search: the order it visits nodes decides ties")
        }
        "grafeo.stochastic_block_partition" => {
            Across::OrderDependent("a greedy search: ties between merges go by block order")
        }
        other => panic!("`{other}` is new in the catalogue: give it a row here"),
    }
}

/// The nodes `v0` .. `v79` and a seeded list of about 300 edges `vi -> vj` (i < j, so a DAG) with
/// distinct weights `w` (a unique spanning tree), capacities `c` and costs `k`, plus documents
/// with vectors and texts that differ, indexed. `reversed` inserts the nodes and the edges in the
/// opposite order, so every id differs and so does each node's edge order; `edges_reversed`
/// reverses the edges alone.
fn graph(reversed: bool) -> (GrafeoDB, HashMap<String, Value>) {
    load(reversed, reversed)
}

fn load(nodes_reversed: bool, edges_reversed: bool) -> (GrafeoDB, HashMap<String, Value>) {
    const NODES: usize = 80;
    let mut state: u64 = 0x0D1F_FE5E_ED00_0001;
    let mut below = |bound: u64| {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        state % bound
    };
    let mut edges: Vec<(usize, usize)> = Vec::new();
    for u in 0..NODES {
        for v in (u + 1)..NODES {
            if below(1000) < 95 {
                edges.push((u, v));
            }
        }
    }
    let db = GrafeoDB::new_in_memory();
    let mut order: Vec<usize> = (0..NODES).collect();
    if nodes_reversed {
        order.reverse();
    }
    if edges_reversed {
        edges.reverse();
    }
    let mut ids = vec![None; NODES];
    for i in order {
        let id = db.create_node(&["V"]).unwrap();
        db.set_node_property(id, "name", Value::from(format!("v{i}").as_str()))
            .unwrap();
        ids[i] = Some(id);
    }
    let count = edges.len();
    for (position, (u, v)) in edges.iter().enumerate() {
        // The weight, capacity and cost follow the edge, not its place in the insertion.
        let index = if edges_reversed {
            count - 1 - position
        } else {
            position
        };
        let edge = db
            .create_edge(ids[*u].unwrap(), ids[*v].unwrap(), "E")
            .unwrap();
        for (key, value) in [
            ("w", i64::try_from(index + 1).unwrap()),
            ("c", i64::try_from(1 + index % 5).unwrap()),
            ("k", i64::try_from(1 + index % 4).unwrap()),
        ] {
            db.set_edge_property(edge, key, Value::Int64(value))
                .unwrap();
        }
    }
    let mut docs: Vec<usize> = (0..30).collect();
    if nodes_reversed {
        docs.reverse();
    }
    for i in docs {
        let x = f64::from(u32::try_from(i).unwrap()) / 30.0 + 0.001;
        // A body of its own length, so no two BM25 scores tie (a tie goes by node id).
        let body = format!("graph{}", " word".repeat(i));
        db.execute(&format!(
            "INSERT (:Doc {{name: 'doc{i}', emb: vector([{x}, {}]), body: '{body}'}})",
            1.0 - x * x
        ))
        .unwrap();
    }
    db.create_vector_index("Doc", "emb", Some(2), Some("euclidean"), None, None, None)
        .unwrap();
    db.create_text_index("Doc", "body").unwrap();
    let id = |n: usize| Value::Int64(i64::try_from(ids[n].unwrap().as_u64()).unwrap());
    let params = HashMap::from([("s".to_string(), id(0)), ("t".to_string(), id(NODES - 1))]);
    (db, params)
}

fn argument(procedure: &str, parameter: &str, required: bool) -> Option<&'static str> {
    Some(match (procedure, parameter) {
        ("grafeo.sssp", "source") => "'v0'",
        (_, "source" | "start") => "$s",
        (_, "sink") => "$t",
        (_, "weight") => "'w'",
        (_, "capacity") => "'c'",
        (_, "cost") => "'k'",
        (_, "label") => "'Doc'",
        ("grafeo.search.text", "property") => "'body'",
        (_, "property") => "'emb'",
        ("grafeo.search.text", "query") => "'graph'",
        (_, "query") => "[0.5, 0.5]",
        ("grafeo.search.mmr", "k") => "3",
        ("grafeo.search.mmr", "fetch_k") => "10",
        (_, "pattern_edges") => "'0-1,1-2'",
        (_, "pattern_nodes") => "3",
        _ if required => panic!("{procedure}: no value for the required `{parameter}`"),
        _ => return None,
    })
}

fn calls() -> Vec<(String, String)> {
    let mut found: Vec<(String, String)> = BuiltinProcedures::new()
        .list()
        .into_iter()
        // The two loads read a file and write the graph: no answer to compare, and the loader
        // and deadline tests cover them. Any other procedure, one that reaches outside too,
        // fails here until it has its arguments.
        .filter(|info| {
            !["grafeo.load_nodes_csv", "grafeo.load_edges_csv"].contains(&info.name.as_str())
        })
        .map(|info| {
            let entries: Vec<String> = info
                .parameters
                .iter()
                .filter_map(|p| {
                    argument(&info.name, &p.name, p.required).map(|v| format!("{}: {v}", p.name))
                })
                .collect();
            let call = format!("CALL {}({{{}}})", info.name, entries.join(", "));
            (info.name, call)
        })
        .collect();
    found.sort();
    assert!(found.len() > 25, "the catalogue: {}", found.len());
    found
}

/// The columns and rows of a call.
fn run(
    db: &GrafeoDB,
    call: &str,
    params: &HashMap<String, Value>,
) -> (Vec<String>, Vec<Vec<Value>>) {
    let result = db
        .execute_with_params(call, params.clone())
        .unwrap_or_else(|error| panic!("{call}: {error}"));
    (result.columns.clone(), result.rows().to_vec())
}

/// The name of each node id.
fn names(db: &GrafeoDB) -> HashMap<i64, String> {
    db.execute("MATCH (n) RETURN id(n), n.name")
        .unwrap()
        .rows()
        .iter()
        .map(|row| {
            let Value::Int64(id) = row[0] else {
                panic!("an id")
            };
            (id, row[1].as_str().unwrap().to_string())
        })
        .collect()
}

/// A value as it compares across insertion orders: a node id as its node's name.
#[derive(Debug, Clone, PartialEq, PartialOrd)]
enum Cell {
    Name(String),
    Number(f64),
    Text(String),
}

fn cells(
    columns: &[String],
    rows: &[Vec<Value>],
    names: &HashMap<i64, String>,
    keep: Option<&[&str]>,
    undirected: bool,
) -> Vec<Vec<Cell>> {
    let ids = ["node_id", "source", "target"];
    let kept: Vec<&String> = columns
        .iter()
        .filter(|column| keep.is_none_or(|keep| keep.contains(&column.as_str())))
        .collect();
    let ends = (
        kept.iter().position(|column| *column == "source"),
        kept.iter().position(|column| *column == "target"),
    );
    let mut found: Vec<Vec<Cell>> = rows
        .iter()
        .map(|row| {
            let mut cells: Vec<Cell> = columns
                .iter()
                .zip(row)
                .filter(|(column, _)| keep.is_none_or(|keep| keep.contains(&column.as_str())))
                .map(|(column, value)| match value {
                    Value::Int64(id) if ids.contains(&column.as_str()) => {
                        Cell::Name(names[id].clone())
                    }
                    Value::Int64(i) => Cell::Number(*i as f64),
                    Value::Float64(f) => Cell::Number(*f),
                    other => Cell::Text(format!("{other:?}")),
                })
                .collect();
            if let (true, (Some(source), Some(target))) = (undirected, ends)
                && cells[source].partial_cmp(&cells[target]) == Some(std::cmp::Ordering::Greater)
            {
                cells.swap(source, target);
            }
            cells
        })
        .collect();
    found.sort_by(|a, b| a.partial_cmp(b).unwrap());
    found.dedup();
    found
}

fn alike(a: &[Vec<Cell>], b: &[Vec<Cell>]) -> bool {
    a.len() == b.len()
        && a.iter().zip(b).all(|(x, y)| {
            x.len() == y.len()
                && x.iter().zip(y).all(|pair| match pair {
                    (Cell::Number(p), Cell::Number(q)) => {
                        (p - q).abs() <= 1e-9 * p.abs().max(q.abs()).max(1.0)
                    }
                    (p, q) => p == q,
                })
        })
}

/// The groups of a partition column, by node names.
fn groups(
    columns: &[String],
    rows: &[Vec<Value>],
    names: &HashMap<i64, String>,
    column: &str,
) -> BTreeSet<BTreeSet<String>> {
    let node = columns.iter().position(|c| c == "node_id").unwrap();
    let group = columns.iter().position(|c| c == column).unwrap();
    let mut by: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();
    for row in rows {
        let Value::Int64(id) = row[node] else {
            panic!("a node id")
        };
        by.entry(format!("{:?}", row[group]))
            .or_default()
            .insert(names[&id].clone());
    }
    by.into_values().collect()
}

/// The same graph loaded the same way: the very same rows, twice in one database and in a second.
#[test]
fn every_procedure_answers_the_same_for_the_same_graph() {
    let (first, params) = graph(false);
    let (second, second_params) = graph(false);
    let mut wrong = Vec::new();
    for (name, call) in calls() {
        let _ = across(&name);
        let once = format!("{:?}", run(&first, &call, &params));
        let again = format!("{:?}", run(&first, &call, &params));
        let elsewhere = format!("{:?}", run(&second, &call, &second_params));
        if once != again {
            wrong.push(format!("{name}: a second run differs"));
        }
        if once != elsewhere {
            wrong.push(format!(
                "{name}: a second database built the same way differs"
            ));
        }
    }
    assert!(wrong.is_empty(), "{}", wrong.join("\n"));
}

/// The same graph loaded in the opposite order: the unique answers agree up to the node names.
#[test]
fn every_unique_answer_survives_another_insertion_order() {
    let (forward, forward_params) = graph(false);
    let (backward, backward_params) = graph(true);
    let (forward_names, backward_names) = (names(&forward), names(&backward));
    let mut wrong = Vec::new();
    for (name, call) in calls() {
        let (columns, a) = run(&forward, &call, &forward_params);
        let (_, b) = run(&backward, &call, &backward_params);
        match across(&name) {
            Across::Rows | Across::UndirectedRows | Across::Columns(_) => {
                let keep = match across(&name) {
                    Across::Columns(keep) => Some(keep),
                    _ => None,
                };
                let undirected = matches!(across(&name), Across::UndirectedRows);
                let (x, y) = (
                    cells(&columns, &a, &forward_names, keep, undirected),
                    cells(&columns, &b, &backward_names, keep, undirected),
                );
                if !alike(&x, &y) {
                    wrong.push(format!("{name}: {x:?}\n  reversed: {y:?}"));
                }
            }
            Across::Partition(column) => {
                let (x, y) = (
                    groups(&columns, &a, &forward_names, column),
                    groups(&columns, &b, &backward_names, column),
                );
                if x != y {
                    wrong.push(format!("{name}: another partition"));
                }
            }
            Across::ById(reason) | Across::OrderDependent(reason) => {
                assert!(!reason.is_empty(), "{name}");
            }
        }
    }
    assert!(wrong.is_empty(), "{}", wrong.join("\n"));
}

/// The same nodes, the edges loaded in the opposite order: the ids are the same, so every answer
/// the ids decide is the very same, the topological order included.
#[test]
fn every_answer_survives_another_edge_order() {
    let (forward, params) = load(false, false);
    let (other, other_params) = load(false, true);
    let names = names(&forward);
    let mut wrong = Vec::new();
    for (name, call) in calls() {
        let (columns, a) = run(&forward, &call, &params);
        let (_, b) = run(&other, &call, &other_params);
        let keep = match across(&name) {
            Across::Columns(keep) => Some(keep),
            Across::OrderDependent(_) => continue,
            Across::Rows | Across::UndirectedRows | Across::Partition(_) | Across::ById(_) => None,
        };
        let same = match across(&name) {
            Across::Partition(column) => {
                groups(&columns, &a, &names, column) == groups(&columns, &b, &names, column)
            }
            undirected => {
                let undirected = matches!(undirected, Across::UndirectedRows);
                alike(
                    &cells(&columns, &a, &names, keep, undirected),
                    &cells(&columns, &b, &names, keep, undirected),
                )
            }
        };
        if !same {
            wrong.push(format!(
                "{name}: another answer with the edges in another order"
            ));
        }
    }
    assert!(wrong.is_empty(), "{}", wrong.join("\n"));
}

/// The smallest topological order by node index: Kahn's algorithm, the ready node with the
/// smallest index first.
fn smallest_order(nodes: usize, edges: &[(usize, usize)]) -> Option<Vec<usize>> {
    let mut indegree = vec![0usize; nodes];
    let mut out = vec![Vec::new(); nodes];
    for &(u, v) in edges {
        out[u].push(v);
        indegree[v] += 1;
    }
    let mut ready: std::collections::BinaryHeap<std::cmp::Reverse<usize>> = (0..nodes)
        .filter(|&v| indegree[v] == 0)
        .map(std::cmp::Reverse)
        .collect();
    let mut order = Vec::new();
    while let Some(std::cmp::Reverse(u)) = ready.pop() {
        order.push(u);
        for &v in &out[u] {
            indegree[v] -= 1;
            if indegree[v] == 0 {
                ready.push(std::cmp::Reverse(v));
            }
        }
    }
    (order.len() == nodes).then_some(order)
}

/// `grafeo.topological_sort()` on a DAG whose nodes were created in index order, as indices.
fn topological_indices(nodes: usize, edges: &[(usize, usize)]) -> Vec<usize> {
    let db = GrafeoDB::new_in_memory();
    let ids: Vec<_> = (0..nodes)
        .map(|_| db.create_node(&["V"]).unwrap())
        .collect();
    for &(u, v) in edges {
        db.create_edge(ids[u], ids[v], "E").unwrap();
    }
    let index: HashMap<i64, usize> = ids
        .iter()
        .enumerate()
        .map(|(i, id)| (i64::try_from(id.as_u64()).unwrap(), i))
        .collect();
    let (columns, rows) = run(&db, "CALL grafeo.topological_sort()", &HashMap::new());
    let node = columns.iter().position(|c| c == "node_id").unwrap();
    let place = columns.iter().position(|c| c == "order").unwrap();
    let mut placed: Vec<(i64, usize)> = rows
        .iter()
        .map(|row| {
            let (Value::Int64(id), Value::Int64(order)) = (&row[node], &row[place]) else {
                panic!("ids and places")
            };
            (*order, index[id])
        })
        .collect();
    placed.sort_unstable();
    placed.into_iter().map(|(_, i)| i).collect()
}

/// The order is networkx's `lexicographical_topological_sort` on seeded DAGs (pinned), and the
/// reference's on the class graph, its edges in either order.
#[test]
fn the_topological_order_is_the_smallest_by_node_id() {
    let pinned: [(usize, &[(usize, usize)], &[usize]); 3] = [
        (
            9,
            &[
                (0, 2),
                (1, 2),
                (3, 1),
                (3, 8),
                (4, 1),
                (4, 2),
                (4, 8),
                (5, 1),
                (5, 4),
                (5, 7),
                (6, 1),
                (6, 4),
                (6, 8),
                (7, 0),
                (7, 1),
                (7, 2),
            ],
            &[3, 5, 6, 4, 7, 0, 1, 2, 8],
        ),
        (
            9,
            &[
                (3, 5),
                (4, 0),
                (4, 1),
                (5, 2),
                (6, 2),
                (6, 7),
                (6, 8),
                (7, 2),
            ],
            &[3, 4, 0, 1, 5, 6, 7, 2, 8],
        ),
        (
            9,
            &[
                (1, 8),
                (4, 2),
                (5, 0),
                (5, 2),
                (5, 8),
                (6, 0),
                (6, 3),
                (6, 4),
                (7, 2),
                (8, 3),
            ],
            &[1, 5, 6, 0, 4, 7, 2, 8, 3],
        ),
    ];
    for (nodes, edges, want) in pinned {
        assert_eq!(
            smallest_order(nodes, edges).as_deref(),
            Some(want),
            "the reference"
        );
        assert_eq!(topological_indices(nodes, edges), want, "{edges:?}");
        let mut reversed = edges.to_vec();
        reversed.reverse();
        assert_eq!(
            topological_indices(nodes, &reversed),
            want,
            "{edges:?} reversed"
        );
    }
}

/// A cycle gives no order: an error, or no rows, never a row that says so in text.
#[test]
fn a_cycle_gives_no_topological_order() {
    let db = GrafeoDB::new_in_memory();
    let ids: Vec<_> = (0..4).map(|_| db.create_node(&["V"]).unwrap()).collect();
    for (u, v) in [(0, 1), (1, 2), (2, 0), (2, 3)] {
        db.create_edge(ids[u], ids[v], "E").unwrap();
    }
    if let Ok(result) = db.execute("CALL grafeo.topological_sort()") {
        assert!(
            result.rows().is_empty(),
            "a cycle answered rows: {:?}",
            result.rows()
        );
    }
}

/// A topological order, whichever it is, puts every edge's source before its target.
#[test]
fn every_topological_order_is_valid() {
    for reversed in [false, true] {
        let (db, params) = graph(reversed);
        let (columns, rows) = run(&db, "CALL grafeo.topological_sort()", &params);
        let node = columns.iter().position(|c| c == "node_id").unwrap();
        let place = columns.iter().position(|c| c == "order").unwrap();
        let at: HashMap<String, i64> = rows
            .iter()
            .map(|row| {
                let (Value::Int64(id), Value::Int64(order)) = (&row[node], &row[place]) else {
                    panic!("ids and places")
                };
                (format!("{id}"), *order)
            })
            .collect();
        let edges = db
            .execute("MATCH (a:V)-[:E]->(b:V) RETURN id(a), id(b)")
            .unwrap();
        for edge in edges.rows() {
            let (Value::Int64(a), Value::Int64(b)) = (&edge[0], &edge[1]) else {
                panic!("ids")
            };
            assert!(
                at[&format!("{a}")] < at[&format!("{b}")],
                "reversed = {reversed}: {a} -> {b} out of order"
            );
        }
    }
}
