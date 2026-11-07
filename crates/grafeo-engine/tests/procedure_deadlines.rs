//! Procedures stop at the query's deadline (0.6.2): every registered procedure polls one
//! `deadline::Ticker` in its work loops, so `CALL` past its deadline raises the executor's timeout
//! and never returns a partial answer. 0.6.1 had no check in any algorithm:
//! `GrafeoDB(query_timeout=0.5)` ran `min_cost_max_flow` on a negative cycle for 100 s.
//!
//! The probe (`testing-deadline-probe`): `deadline::expire_after_polls(n)` makes every ticker on the
//! thread, and every worker given a handoff taken under it, count its deadline as passed after `n`
//! clock reads, with or without a deadline. A ticker reads the clock on its first unit and then on
//! every 1,024th, so `n = 0` stops at the first unit and `n = 1` after 1,024. The count is per
//! ticker: an algorithm that starts a fresh ticker per source may finish under `n = 1` without
//! one ticker passing 1,024 units, and still be right, as a real deadline is read on each new
//! ticker's first unit. So the class test asserts what holds for every procedure: under `n = 1`
//! the timeout or the very answer of a run without one, never a partial answer.
//!
//! ```bash
//! cargo test -p grafeo-engine --all-features --test main procedure_deadlines::
//! ```

#![cfg(all(feature = "testing-deadline-probe", feature = "algos"))]

use std::collections::HashMap;
use std::time::{Duration, Instant};

use grafeo_common::types::Value;
use grafeo_common::utils::error::{Error, QueryErrorKind};
use grafeo_core::execution::deadline;
use grafeo_engine::procedures::BuiltinProcedures;
use grafeo_engine::{Config, GrafeoDB};

/// What a procedure does under the probe.
enum Row {
    /// It polls: under `n = 0` it raises the timeout.
    Polls,
    /// It does no work loop of its own; the reason.
    Exempt(&'static str),
}

/// Procedures that need not poll, each with its reason. Empty: even the schema procedures
/// (`labels`, `relationshipTypes`, `propertyKeys`) scan every node and edge for the names in use.
const EXEMPT: [(&str, &str); 0] = [];

/// The row of each procedure of the catalogue; a procedure added to the catalogue fails here
/// until it has one.
fn row(name: &str) -> Row {
    if let Some((_, reason)) = EXEMPT.iter().find(|(exempt, _)| *exempt == name) {
        return Row::Exempt(reason);
    }
    match name {
        "grafeo.labels"
        | "grafeo.relationshipTypes"
        | "grafeo.propertyKeys"
        | "grafeo.pagerank"
        | "grafeo.betweenness_centrality"
        | "grafeo.closeness_centrality"
        | "grafeo.degree_centrality"
        | "grafeo.bfs"
        | "grafeo.dfs"
        | "grafeo.connected_components"
        | "grafeo.strongly_connected_components"
        | "grafeo.topological_sort"
        | "grafeo.dijkstra"
        | "grafeo.sssp"
        | "grafeo.bellman_ford"
        | "grafeo.floyd_warshall"
        | "grafeo.clustering_coefficient"
        | "grafeo.label_propagation"
        | "grafeo.louvain"
        | "grafeo.kruskal"
        | "grafeo.prim"
        | "grafeo.max_flow"
        | "grafeo.min_cost_max_flow"
        | "grafeo.articulation_points"
        | "grafeo.bridges"
        | "grafeo.kcore"
        | "grafeo.ktruss"
        | "grafeo.total_triangles"
        | "grafeo.subgraph_isomorphism"
        | "grafeo.stochastic_block_partition"
        | "grafeo.search.vector"
        | "grafeo.search.text"
        | "grafeo.search.mmr"
        | "grafeo.load_nodes_csv"
        | "grafeo.load_edges_csv" => Row::Polls,
        other => panic!("`{other}` is new in the catalogue: give it a row here"),
    }
}

/// A DAG of 200 nodes and about 1,200 edges in a seeded order, each edge with a weight `w`, a
/// capacity `c` and a cost `k`, all positive (no negative cycle, so `min_cost_max_flow` runs), and
/// 60 documents with a vector `emb` and a text `body`, indexed. Node `i` is named `v{i}`. Every procedure does more than
/// 1,024 units of work on it. `$s` is the first node, `$t` the last.
fn graph() -> (GrafeoDB, HashMap<String, Value>) {
    let db = GrafeoDB::new_in_memory();
    let ids: Vec<_> = (0..200)
        .map(|i| {
            let id = db.create_node(&["V"]).unwrap();
            db.set_node_property(id, "name", Value::from(format!("v{i}").as_str()))
                .unwrap();
            id
        })
        .collect();
    let mut state: u64 = 0x5EED_DEAD_11E5;
    let mut below = |bound: u64| {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        state % bound
    };
    for (u, &from) in ids.iter().enumerate() {
        for &to in &ids[u + 1..] {
            if below(1000) < 60 {
                let edge = db.create_edge(from, to, "E").unwrap();
                for (key, value) in [
                    ("w", 1 + below(9)),
                    ("c", 1 + below(5)),
                    ("k", 1 + below(4)),
                ] {
                    db.set_edge_property(edge, key, Value::Int64(i64::try_from(value).unwrap()))
                        .unwrap();
                }
            }
        }
    }
    for i in 0..60 {
        let x = f64::from(i) / 60.0;
        db.execute(&format!(
            "INSERT (:Doc {{name: 'doc{i}', emb: vector([{x}, {}]), body: 'graph {}'}})",
            1.0 - x,
            if i % 2 == 0 { "database" } else { "theory" }
        ))
        .unwrap();
    }
    db.create_vector_index("Doc", "emb", Some(2), Some("euclidean"), None, None, None)
        .unwrap();
    db.create_text_index("Doc", "body").unwrap();
    let id = |n: usize| Value::Int64(i64::try_from(ids[n].as_u64()).unwrap());
    let params = HashMap::from([("s".to_string(), id(0)), ("t".to_string(), id(199))]);
    (db, params)
}

/// The quoted path of a CSV fixture the loads read, written once per process: `nodes.csv` with
/// 2,000 names (past the ticker's 1,024), `edges.csv` with an edge from each `v{i}` of the graph
/// to `v{i+1}` (so the graph stays acyclic).
fn fixture(name: &str) -> &'static str {
    static PATHS: std::sync::OnceLock<(&'static str, &'static str)> = std::sync::OnceLock::new();
    let (nodes, edges) = PATHS.get_or_init(|| {
        let dir =
            std::env::temp_dir().join(format!("grafeo-deadline-loads-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let nodes: String = std::iter::once("name\n".to_string())
            .chain((0..2_000).map(|i| format!("Loaded{i}\n")))
            .collect();
        let edges: String = std::iter::once("from,to\n".to_string())
            .chain((0..199).map(|i| format!("v{i},v{}\n", i + 1)))
            .collect();
        let quoted = |file: &str, text: &str| -> &'static str {
            let path = dir.join(file);
            std::fs::write(&path, text).unwrap();
            let path = path.to_string_lossy().replace('\\', "/");
            Box::leak(format!("'{path}'").into_boxed_str())
        };
        (quoted("nodes.csv", &nodes), quoted("edges.csv", &edges))
    });
    if name == "nodes.csv" { nodes } else { edges }
}

/// The value a call gives `parameter` of `procedure`: the required ones, and the edge properties
/// the weighted procedures read.
fn argument(procedure: &str, parameter: &str, required: bool) -> Option<&'static str> {
    Some(match (procedure, parameter) {
        // The loads read a CSV fixture: 2,000 nodes, and an edge from each `v{i}` to `v{i+1}`.
        ("grafeo.load_nodes_csv", "path") => fixture("nodes.csv"),
        ("grafeo.load_nodes_csv", "labels") => "'Loaded'",
        ("grafeo.load_edges_csv", "path") => fixture("edges.csv"),
        ("grafeo.load_edges_csv", "type") => "'LOADED'",
        ("grafeo.load_edges_csv", "source") => "{label: 'V', property: 'name', column: 'from'}",
        ("grafeo.load_edges_csv", "target") => "{label: 'V', property: 'name', column: 'to'}",
        // `sssp` names its source by the node's `name`.
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

/// Every procedure of the catalogue that stays inside the database, with its call, in the map
/// form, and its row.
fn calls() -> Vec<(String, String, Row)> {
    let mut found: Vec<(String, String, Row)> = BuiltinProcedures::new()
        .list()
        .into_iter()
        // Every procedure has its row, the loads included (they read the CSV fixture): a new
        // procedure, one that reaches outside too, fails here until it has its arguments.
        .map(|info| {
            let entries: Vec<String> = info
                .parameters
                .iter()
                .filter_map(|p| {
                    argument(&info.name, &p.name, p.required).map(|v| format!("{}: {v}", p.name))
                })
                .collect();
            let call = format!("CALL {}({{{}}})", info.name, entries.join(", "));
            let row = row(&info.name);
            (info.name, call, row)
        })
        .collect();
    found.sort_by(|a, b| a.0.cmp(&b.0));
    assert!(found.len() > 25, "the catalogue: {}", found.len());
    found
}

/// The rows of a call, as text, sorted: a deterministic answer compares alike.
fn answer(
    db: &GrafeoDB,
    call: &str,
    params: &HashMap<String, Value>,
) -> Result<Vec<String>, Error> {
    db.execute_with_params(call, params.clone()).map(|result| {
        let mut rows: Vec<String> = result.rows().iter().map(|r| format!("{r:?}")).collect();
        rows.sort();
        rows
    })
}

fn is_timeout(error: &Error) -> bool {
    matches!(error, Error::Query(query) if query.kind == QueryErrorKind::Timeout)
}

/// `n = 0`: every procedure that polls raises the timeout at its first unit; an exempt one
/// answers as without the probe.
#[test]
fn every_procedure_stops_at_its_first_poll() {
    let (db, params) = graph();
    let mut wrong = Vec::new();
    for (name, call, row) in calls() {
        let plain = answer(&db, &call, &params);
        let probed = {
            let _probe = deadline::expire_after_polls(0);
            answer(&db, &call, &params)
        };
        match (row, probed) {
            (Row::Polls, Err(error)) if is_timeout(&error) => {}
            (Row::Polls, other) => wrong.push(format!("{name}: {other:?}, want the timeout")),
            (Row::Exempt(reason), probed) => {
                if format!("{probed:?}") != format!("{plain:?}") {
                    wrong.push(format!(
                        "{name} (exempt: {reason}): {probed:?}, without the probe {plain:?}"
                    ));
                }
            }
        }
    }
    assert!(wrong.is_empty(), "{}", wrong.join("\n"));
}

/// `n = 1`: the timeout, or the very answer of a run without the probe; never a partial answer.
#[test]
fn a_later_poll_gives_the_timeout_or_the_whole_answer() {
    let (db, params) = graph();
    let mut wrong = Vec::new();
    for (name, call, _) in calls() {
        let plain = answer(&db, &call, &params).unwrap_or_else(|error| panic!("{name}: {error}"));
        let probed = {
            let _probe = deadline::expire_after_polls(1);
            answer(&db, &call, &params)
        };
        match probed {
            Err(error) if is_timeout(&error) => {}
            Ok(rows) if rows == plain => {}
            other => wrong.push(format!("{name}: {other:?}")),
        }
    }
    assert!(wrong.is_empty(), "{}", wrong.join("\n"));
}

/// One ticker that runs past 1,024 units: pagerank's iterations over 1,200 edges, and the
/// parallel triangle count, whose workers take the probe over by the handoff.
#[test]
fn a_long_loop_stops_at_its_second_read() {
    let (db, params) = graph();
    for call in [
        "CALL grafeo.pagerank({max_iterations: 100, tolerance: 0.0})",
        "CALL grafeo.total_triangles({parallel: true, parallel_threshold: 0})",
        "CALL grafeo.floyd_warshall({weight: 'w'})",
    ] {
        let _probe = deadline::expire_after_polls(1);
        let error = answer(&db, call, &params).expect_err(call);
        assert!(is_timeout(&error), "{call}: {error}");
    }
}

/// A deadline far in the future changes no answer.
#[test]
fn a_future_deadline_changes_no_answer() {
    let (db, params) = graph();
    for (name, call, _) in calls() {
        let plain = answer(&db, &call, &params).unwrap_or_else(|error| panic!("{name}: {error}"));
        let bounded = {
            let _scope = deadline::enter(Some(Instant::now() + Duration::from_secs(3_600)));
            answer(&db, &call, &params).unwrap_or_else(|error| panic!("{name}: {error}"))
        };
        assert_eq!(bounded, plain, "{name}");
    }
}

/// A procedure inside a query whose deadline is earlier keeps the earlier one: an outer deadline
/// already passed stops a call that the database's own timeout would let run.
#[test]
fn an_earlier_outer_deadline_wins() {
    let db =
        GrafeoDB::with_config(Config::in_memory().with_query_timeout(Duration::from_secs(3_600)))
            .unwrap();
    db.execute("INSERT (a:V)-[:E]->(b:V)-[:E]->(c:V)-[:E]->(a)")
        .unwrap();
    let _scope = deadline::enter(Some(Instant::now()));
    let error = db
        .execute("CALL grafeo.pagerank()")
        .expect_err("the outer deadline has passed");
    assert!(is_timeout(&error), "{error}");
}

/// A timeout leaves the session and its transaction usable: procedures only read.
#[test]
fn a_timeout_leaves_the_session_usable() {
    let (db, params) = graph();
    let mut session = db.session();
    session.begin_transaction().unwrap();
    session.execute("INSERT (:Kept {n: 1})").unwrap();
    {
        let _probe = deadline::expire_after_polls(0);
        let error = session
            .execute_with_params("CALL grafeo.pagerank()", params.clone())
            .expect_err("the probe");
        assert!(is_timeout(&error), "{error}");
    }
    assert_eq!(
        session
            .execute("MATCH (n:Kept) RETURN n.n")
            .unwrap()
            .rows()
            .len(),
        1,
        "the transaction's own write"
    );
    session.commit().unwrap();
    assert!(
        session
            .execute_with_params("CALL grafeo.pagerank()", params)
            .is_ok()
    );
}

/// End to end, a real timeout: Floyd-Warshall on 1,500 nodes (3.4e9 relaxations, seconds even in
/// release) under `query_timeout = 0.5 s` raises the timeout, well before it could finish. The
/// bound on the time is loose, so a slow runner cannot flake it.
#[test]
fn a_query_timeout_stops_floyd_warshall() {
    let db =
        GrafeoDB::with_config(Config::in_memory().with_query_timeout(Duration::from_millis(500)))
            .unwrap();
    let ids: Vec<_> = (0..1_500)
        .map(|_| db.create_node(&["V"]).unwrap())
        .collect();
    for pair in ids.windows(2) {
        db.create_edge(pair[0], pair[1], "E").unwrap();
    }
    let started = Instant::now();
    let error = db
        .execute("CALL grafeo.floyd_warshall()")
        .expect_err("past its timeout");
    assert!(is_timeout(&error), "{error}");
    assert!(
        started.elapsed() < Duration::from_secs(20),
        "stopped after {:?}",
        started.elapsed()
    );
    // The database answers the next statement.
    assert!(db.execute("MATCH (n:V) RETURN count(n)").is_ok());
}
