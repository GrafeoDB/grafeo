//! A weighted algorithm refuses an edge value it cannot use, naming the edge and the property (the
//! ruling of 2026-11-06): a value that is no number is 22G12; NaN, either infinity, and a negative
//! value where the algorithm needs a non-negative one are 22023. A missing value takes the
//! algorithm's default (a cost 0.0, every other weight 1.0), as released, and an integer or decimal
//! weight runs as its float. 0.6.1 read a string, boolean or list weight as 1.0 and ran on NaN,
//! returning NaN distances and totals as a success. The table holds every weighted procedure of the
//! catalogue; the class test holds it equal to the registry, so a new one fails until it has a row.
//! `min_cost_max_flow` also refuses a negative-cost cycle the source reaches (it never returned).
//!
//! ```bash
//! cargo test -p grafeo-engine --all-features --test main procedure_edge_weights::
//! ```

#![cfg(all(feature = "gql", feature = "algos"))]

use std::collections::HashMap;

use grafeo_common::types::Value;
use grafeo_common::utils::gqlstatus::GqlStatus;
use grafeo_engine::GrafeoDB;
use grafeo_engine::procedures::BuiltinProcedures;

const WRONG_KIND: &str = "22G12";
const INVALID_PARAMETER_VALUE: &str = "22023";

/// Procedure, the argument naming the edge property, its default when an edge has none, whether a
/// negative value is in its domain, and the call (with `$a` the first node, `$d` the last, and `'w'`
/// the property probed).
struct Weighted {
    procedure: &'static str,
    argument: &'static str,
    default: f64,
    negative: bool,
    call: &'static str,
}

const WEIGHTED: &[Weighted] = &[
    Weighted {
        procedure: "grafeo.dijkstra",
        argument: "weight",
        default: 1.0,
        negative: false,
        call: "CALL grafeo.dijkstra({source: $a, weight: 'w'})",
    },
    Weighted {
        procedure: "grafeo.sssp",
        argument: "weight",
        default: 1.0,
        negative: false,
        call: "CALL grafeo.sssp({source: 'Alix', weight: 'w'})",
    },
    Weighted {
        procedure: "grafeo.bellman_ford",
        argument: "weight",
        default: 1.0,
        negative: true,
        call: "CALL grafeo.bellman_ford({source: $a, weight: 'w'})",
    },
    Weighted {
        procedure: "grafeo.floyd_warshall",
        argument: "weight",
        default: 1.0,
        negative: true,
        call: "CALL grafeo.floyd_warshall({weight: 'w'})",
    },
    Weighted {
        procedure: "grafeo.kruskal",
        argument: "weight",
        default: 1.0,
        negative: true,
        call: "CALL grafeo.kruskal({weight: 'w'})",
    },
    Weighted {
        procedure: "grafeo.prim",
        argument: "weight",
        default: 1.0,
        negative: true,
        call: "CALL grafeo.prim({weight: 'w'})",
    },
    Weighted {
        procedure: "grafeo.max_flow",
        argument: "capacity",
        default: 1.0,
        negative: false,
        call: "CALL grafeo.max_flow({source: $a, sink: $d, capacity: 'w'})",
    },
    Weighted {
        procedure: "grafeo.min_cost_max_flow",
        argument: "capacity",
        default: 1.0,
        negative: false,
        call: "CALL grafeo.min_cost_max_flow({source: $a, sink: $d, capacity: 'w', cost: 'k'})",
    },
    Weighted {
        procedure: "grafeo.min_cost_max_flow",
        argument: "cost",
        default: 0.0,
        negative: true,
        call: "CALL grafeo.min_cost_max_flow({source: $a, sink: $d, capacity: 'c', cost: 'w'})",
    },
];

/// Alix -> Gus -> Vincent -> Mia and Alix -> Vincent: no cycle. Every edge has `w`, `c` and `k`;
/// `probe` gives the first edge (Alix -> Gus) its `w`, or none.
fn graph(probe: Option<&str>) -> (GrafeoDB, HashMap<String, Value>) {
    let db = GrafeoDB::new_in_memory();
    db.execute(
        "INSERT (a:P {name: 'Alix'}), (b:P {name: 'Gus'}), (c:P {name: 'Vincent'}), (d:P {name: 'Mia'}), \
         (a)-[:E {first: true, w: 3, c: 5, k: 1}]->(b), (b)-[:E {w: 4, c: 5, k: 1}]->(c), \
         (a)-[:E {w: 10, c: 2, k: 4}]->(c), (c)-[:E {w: 2, c: 6, k: 1}]->(d)",
    )
    .unwrap();
    match probe {
        Some(value) => {
            db.execute(&format!(
                "MATCH ()-[e:E {{first: true}}]->() SET e.w = {value}"
            ))
            .unwrap();
        }
        None => {
            db.execute("MATCH ()-[e:E {first: true}]->() REMOVE e.w")
                .unwrap();
        }
    }
    let id = |name: &str| {
        db.execute(&format!("MATCH (n:P {{name: '{name}'}}) RETURN id(n)"))
            .unwrap()
            .rows()[0][0]
            .clone()
    };
    let params = HashMap::from([("a".to_string(), id("Alix")), ("d".to_string(), id("Mia"))]);
    (db, params)
}

fn run(
    db: &GrafeoDB,
    call: &str,
    params: &HashMap<String, Value>,
) -> Result<Vec<String>, (String, String)> {
    db.execute_with_params(call, params.clone())
        .map(|result| {
            let mut rows: Vec<String> =
                result.rows().iter().map(|row| format!("{row:?}")).collect();
            rows.sort();
            rows
        })
        .map_err(|error| {
            (
                GqlStatus::from(&error).as_str().to_string(),
                error.to_string(),
            )
        })
}

/// The table names every procedure argument that names an edge property, and nothing else.
#[test]
fn every_weighted_procedure_has_a_row() {
    let mut wrong = Vec::new();
    for info in BuiltinProcedures::new().list() {
        for p in &info.parameters {
            if ["weight", "capacity", "cost"].contains(&p.name.as_str())
                && !WEIGHTED
                    .iter()
                    .any(|row| row.procedure == info.name && row.argument == p.name)
            {
                wrong.push(format!("{}.{}: no row", info.name, p.name));
            }
        }
    }
    for row in WEIGHTED {
        let listed = BuiltinProcedures::new().list().into_iter().any(|info| {
            info.name == row.procedure && info.parameters.iter().any(|p| p.name == row.argument)
        });
        if !listed {
            wrong.push(format!(
                "{}.{}: not in the registry",
                row.procedure, row.argument
            ));
        }
    }
    assert!(wrong.is_empty(), "{}", wrong.join("\n"));
}

/// Every probe on every row: the refusal names the property and the edge, the values in the domain
/// run, a missing value is the default.
#[test]
fn an_edge_value_outside_the_domain_is_refused() {
    let mut wrong = Vec::new();
    for row in WEIGHTED {
        let name = format!("{}.{}", row.procedure, row.argument);
        let refused = |value: &str, status: &str, wrong: &mut Vec<String>| {
            let (db, params) = graph(Some(value));
            match run(&db, row.call, &params) {
                Err((found, message)) if found == status => {
                    if !(message.contains("'w'")
                        || message.contains("\"w\"")
                        || message.contains(" w "))
                        || !message.contains("edge")
                    {
                        wrong.push(format!(
                            "{name}, w = {value}: names no edge and property: {message}"
                        ));
                    }
                }
                other => wrong.push(format!("{name}, w = {value}: {other:?}, want {status}")),
            }
        };
        for value in ["'5'", "'x'", "true", "[5]", "{a: 5}", "DATE '2024-01-01'"] {
            refused(value, WRONG_KIND, &mut wrong);
        }
        for (label, x) in [
            ("NaN", f64::NAN),
            ("+inf", f64::INFINITY),
            ("-inf", f64::NEG_INFINITY),
        ] {
            let (db, mut params) = graph(None);
            params.insert("x".to_string(), Value::Float64(x));
            db.execute_with_params(
                "MATCH ()-[e:E {first: true}]->() SET e.w = $x",
                params.clone(),
            )
            .unwrap();
            match run(&db, row.call, &params) {
                Err((status, message)) if status == INVALID_PARAMETER_VALUE => {
                    if !message.contains("edge") {
                        wrong.push(format!("{name}, w = {label}: names no edge: {message}"));
                    }
                }
                other => wrong.push(format!("{name}, w = {label}: {other:?}, want 22023")),
            }
        }
        let negative = graph(Some("-2"));
        match (row.negative, run(&negative.0, row.call, &negative.1)) {
            (true, Ok(_)) => {}
            (false, Err((status, _))) if status == INVALID_PARAMETER_VALUE => {}
            (allowed, other) => wrong.push(format!(
                "{name}, w = -2 (negative {}): {other:?}",
                if allowed { "allowed" } else { "refused" }
            )),
        }
        // -0.0 is zero: in every domain.
        let signed_zero = graph(Some("-0.0"));
        let zero = graph(Some("0.0"));
        let (with_signed, with_zero) = (
            run(&signed_zero.0, row.call, &signed_zero.1),
            run(&zero.0, row.call, &zero.1),
        );
        if with_signed.is_err() || with_signed != with_zero {
            wrong.push(format!(
                "{name}: -0.0 gave {with_signed:?}, 0.0 gave {with_zero:?}"
            ));
        }
        // A missing value is the default; an integer and a decimal run as their float.
        let missing = graph(None);
        let default = graph(Some(&format!("{:?}", row.default)));
        let (found, want) = (
            run(&missing.0, row.call, &missing.1),
            run(&default.0, row.call, &default.1),
        );
        if found.is_err() || found != want {
            wrong.push(format!(
                "{name}: a missing value gave {found:?}, the default {} gives {want:?}",
                row.default
            ));
        }
        let (float_db, float_params) = graph(Some("3.0"));
        let float = run(&float_db, row.call, &float_params);
        for value in ["3", "CAST(3 AS DECIMAL)"] {
            let (db, params) = graph(Some(value));
            let found = run(&db, row.call, &params);
            if found.is_err() || found.as_ref().map(Vec::len) != float.as_ref().map(Vec::len) {
                wrong.push(format!(
                    "{name}: w = {value} gave {found:?}, 3.0 gives {float:?}"
                ));
            }
        }
    }
    assert!(
        wrong.is_empty(),
        "{} wrong:\n{}",
        wrong.len(),
        wrong.join("\n")
    );
}

fn total(db: &GrafeoDB, procedure: &str) -> f64 {
    db.execute(&format!(
        "CALL grafeo.{procedure}({{weight: 'w'}}) YIELD total_weight RETURN total_weight LIMIT 1"
    ))
    .unwrap()
    .rows()[0][0]
        .as_float64()
        .unwrap()
}

/// The two spanning-tree algorithms agree on the total, with negative weights too.
#[test]
fn kruskal_and_prim_give_the_same_total() {
    for probe in ["3", "-2", "0.5"] {
        let (db, _) = graph(Some(probe));
        assert_eq!(total(&db, "kruskal"), total(&db, "prim"), "w = {probe}");
    }
}

#[test]
fn a_negative_or_infinite_capacity_is_refused() {
    const MAX_FLOW: &str = "CALL grafeo.max_flow({source: $s, sink: $t, capacity: 'c'}) \
                            YIELD max_flow RETURN max_flow";
    let (db, params) = flow_graph(
        "(s:N {name: 's'}), (t:N {name: 't'}), (s)-[:E {c: -5}]->(t), (s)-[:E {c: 3}]->(t)",
    );
    let error = db
        .execute_with_params(MAX_FLOW, params)
        .expect_err("a capacity of -5");
    assert_eq!(
        GqlStatus::from(&error).as_str(),
        INVALID_PARAMETER_VALUE,
        "{error}"
    );
    assert!(error.to_string().contains("-5"), "{error}");
    let (db, params) = flow_graph("(s:N {name: 's'}), (t:N {name: 't'}), (s)-[:E]->(t)");
    db.execute_with_params(
        "MATCH ()-[e:E]->() SET e.c = $x",
        HashMap::from([("x".to_string(), Value::Float64(f64::INFINITY))]),
    )
    .unwrap();
    let error = db
        .execute_with_params(MAX_FLOW, params)
        .expect_err("an infinite capacity");
    assert_eq!(
        GqlStatus::from(&error).as_str(),
        INVALID_PARAMETER_VALUE,
        "{error}"
    );
}

/// An infinite edge to a node another path reaches is refused, not read as "unreachable".
#[test]
fn an_infinite_edge_is_refused_in_a_shortest_path() {
    let (db, mut params) = graph(None);
    db.execute("MATCH ()-[e:E {first: true}]->() SET e.w = 3")
        .unwrap();
    params.insert("x".to_string(), Value::Float64(f64::INFINITY));
    db.execute_with_params(
        "MATCH (:P {name: 'Alix'})-[e:E]->(:P {name: 'Vincent'}) SET e.w = $x",
        params.clone(),
    )
    .unwrap();
    let error = db
        .execute_with_params("CALL grafeo.dijkstra({source: $a, weight: 'w'})", params)
        .expect_err("an infinite weight");
    assert_eq!(
        GqlStatus::from(&error).as_str(),
        INVALID_PARAMETER_VALUE,
        "{error}"
    );
}

/// With no weight on any edge, every algorithm answers as unweighted: each edge the default.
#[test]
fn no_weights_at_all_is_the_unweighted_answer() {
    for row in WEIGHTED {
        let (db, params) = graph(None);
        db.execute("MATCH ()-[e:E]->() REMOVE e.w").unwrap();
        let (with_default, default_params) = graph(Some("0"));
        with_default
            .execute(&format!("MATCH ()-[e:E]->() SET e.w = {:?}", row.default))
            .unwrap();
        assert_eq!(
            run(&db, row.call, &params),
            run(&with_default, row.call, &default_params),
            "{}.{}",
            row.procedure,
            row.argument
        );
    }
}

/// `MIN_COST` on `edges` in a thread of its own, so a run that never ends (0.6.1 on a
/// negative-cost cycle) fails the test after a bound instead of hanging the suite; the thread is
/// left to the process's exit.
fn min_cost_within_bound(
    name: &str,
    edges: &'static str,
) -> grafeo_common::utils::error::Result<Vec<Vec<Value>>> {
    let (sender, receiver) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
        let (db, params) = flow_graph(edges);
        let found = db
            .execute_with_params(MIN_COST, params)
            .map(|result| result.rows().to_vec());
        let _ = sender.send(found);
    });
    receiver
        .recv_timeout(std::time::Duration::from_secs(30))
        .unwrap_or_else(|_| panic!("{name}: still running after 30 s"))
}

fn flow_graph(edges: &str) -> (GrafeoDB, HashMap<String, Value>) {
    let db = GrafeoDB::new_in_memory();
    db.execute(&format!("INSERT {edges}")).unwrap();
    let id = |name: &str| {
        db.execute(&format!("MATCH (n:N {{name: '{name}'}}) RETURN id(n)"))
            .unwrap()
            .rows()[0][0]
            .clone()
    };
    let params = HashMap::from([("s".to_string(), id("s")), ("t".to_string(), id("t"))]);
    (db, params)
}

const MIN_COST: &str = "CALL grafeo.min_cost_max_flow({source: $s, sink: $t, capacity: 'c', cost: 'k'}) \
                        YIELD max_flow, total_cost RETURN max_flow, total_cost LIMIT 1";

/// A negative-cost cycle the source reaches is refused, naming an edge of it and the cost; one the
/// source cannot reach, a zero-cost cycle and a negative cost without a cycle run.
#[test]
fn min_cost_max_flow_refuses_a_negative_cycle_the_source_reaches() {
    for (name, edges) in [
        (
            "the repro",
            "(s:N {name: 's'}), (a:N {name: 'a'}), (b:N {name: 'b'}), (t:N {name: 't'}), \
             (s)-[:E {c: 5, k: 1}]->(a), (a)-[:E {c: 5, k: -3}]->(b), (b)-[:E {c: 5, k: -3}]->(a), \
             (b)-[:E {c: 5, k: 1}]->(t)",
        ),
        (
            "a cycle the source reaches, away from the sink",
            "(s:N {name: 's'}), (t:N {name: 't'}), (x:N {name: 'x'}), (y:N {name: 'y'}), \
             (s)-[:E {c: 5, k: 1}]->(t), (s)-[:E {c: 5, k: 1}]->(x), (x)-[:E {c: 5, k: -2}]->(y), \
             (y)-[:E {c: 5, k: -2}]->(x)",
        ),
    ] {
        let error = min_cost_within_bound(name, edges).expect_err(name);
        assert_eq!(
            GqlStatus::from(&error).as_str(),
            INVALID_PARAMETER_VALUE,
            "{name}: {error}"
        );
        assert!(
            error.to_string().contains("'k'") || error.to_string().contains("cost"),
            "{name}: names no cost property: {error}"
        );
    }
    for (name, edges, want) in [
        (
            "a negative cost without a cycle",
            "(s:N {name: 's'}), (a:N {name: 'a'}), (t:N {name: 't'}), \
             (s)-[:E {c: 5, k: 2}]->(a), (a)-[:E {c: 5, k: -1}]->(t)",
            (5.0, 5.0),
        ),
        (
            "a zero-cost cycle",
            "(s:N {name: 's'}), (a:N {name: 'a'}), (b:N {name: 'b'}), (t:N {name: 't'}), \
             (s)-[:E {c: 5, k: 1}]->(a), (a)-[:E {c: 5, k: 0}]->(b), (b)-[:E {c: 5, k: 0}]->(a), \
             (b)-[:E {c: 5, k: 1}]->(t)",
            (5.0, 10.0),
        ),
        (
            "a negative cycle the source cannot reach",
            "(s:N {name: 's'}), (t:N {name: 't'}), (x:N {name: 'x'}), (y:N {name: 'y'}), \
             (s)-[:E {c: 5, k: 1}]->(t), (x)-[:E {c: 5, k: -2}]->(y), (y)-[:E {c: 5, k: -2}]->(x)",
            (5.0, 5.0),
        ),
        (
            "no costs: each is 0",
            "(s:N {name: 's'}), (t:N {name: 't'}), (s)-[:E {c: 5}]->(t)",
            (5.0, 0.0),
        ),
    ] {
        let row = min_cost_within_bound(name, edges)
            .unwrap_or_else(|error| panic!("{name}: {error}"))[0]
            .clone();
        let found = (row[0].as_float64().unwrap(), row[1].as_float64().unwrap());
        assert_eq!(found, want, "{name}");
    }
}

/// Parallel edges each keep their own capacity and cost, and an edge in the other direction is an
/// arc of its own: networkx's `max_flow_min_cost` answers, each edge routed through a middle node of
/// its own so that the graph it reads is simple. 0.6.1 summed the capacities but kept the cost of
/// the last parallel edge: 50 and 10 for the first two (both 30), 20 for the third (18), 35 for
/// the fourth (20). The flow was right in all of them.
#[test]
fn parallel_edges_each_keep_their_cost() {
    for (name, edges, want) in [
        (
            "two parallel edges, costs 1 and 5",
            "(s:N {name: 's'}), (t:N {name: 't'}), \
             (s)-[:E {c: 5, k: 1}]->(t), (s)-[:E {c: 5, k: 5}]->(t)",
            (10.0, 30.0),
        ),
        (
            "two parallel edges, the cheap one last",
            "(s:N {name: 's'}), (t:N {name: 't'}), \
             (s)-[:E {c: 5, k: 5}]->(t), (s)-[:E {c: 5, k: 1}]->(t)",
            (10.0, 30.0),
        ),
        (
            "three parallel edges inside a path",
            "(s:N {name: 's'}), (a:N {name: 'a'}), (t:N {name: 't'}), (s)-[:E {c: 10, k: 0}]->(a), \
             (a)-[:E {c: 4, k: 1}]->(t), (a)-[:E {c: 4, k: 3}]->(t), (a)-[:E {c: 4, k: 2}]->(t)",
            (10.0, 18.0),
        ),
        (
            "parallel edges and an opposite edge",
            "(s:N {name: 's'}), (t:N {name: 't'}), (s)-[:E {c: 3, k: 2}]->(t), \
             (t)-[:E {c: 3, k: 2}]->(s), (s)-[:E {c: 2, k: 7}]->(t)",
            (5.0, 20.0),
        ),
    ] {
        let row = min_cost_within_bound(name, edges)
            .unwrap_or_else(|error| panic!("{name}: {error}"))[0]
            .clone();
        let found = (row[0].as_float64().unwrap(), row[1].as_float64().unwrap());
        assert_eq!(found, want, "{name}");
    }
}

/// An edge and its opposite: two arcs, which already answer as networkx does; pinned so the fix of
/// the parallel costs keeps them.
#[test]
fn opposite_edges_are_arcs_of_their_own() {
    for (name, edges, want) in [
        (
            "an edge back from the middle",
            "(s:N {name: 's'}), (a:N {name: 'a'}), (t:N {name: 't'}), \
             (s)-[:E {c: 5, k: 1}]->(a), (a)-[:E {c: 5, k: 1}]->(t), (a)-[:E {c: 5, k: 1}]->(s)",
            (5.0, 10.0),
        ),
        (
            "opposite edges across the middle",
            "(s:N {name: 's'}), (a:N {name: 'a'}), (b:N {name: 'b'}), (t:N {name: 't'}), \
             (s)-[:E {c: 4, k: 1}]->(a), (s)-[:E {c: 4, k: 1}]->(b), (a)-[:E {c: 4, k: 1}]->(b), \
             (b)-[:E {c: 4, k: 1}]->(a), (a)-[:E {c: 4, k: 1}]->(t), (b)-[:E {c: 4, k: 1}]->(t)",
            (8.0, 16.0),
        ),
    ] {
        let row = min_cost_within_bound(name, edges)
            .unwrap_or_else(|error| panic!("{name}: {error}"))[0]
            .clone();
        let found = (row[0].as_float64().unwrap(), row[1].as_float64().unwrap());
        assert_eq!(found, want, "{name}");
    }
}

/// A spanning tree of the undirected graph takes each pair's lightest edge, whichever its
/// direction and whichever comes first: networkx's minimum spanning tree of the simple graph that
/// keeps each pair's minimum weight. 0.6.1 kept the first edge of a pair: Kruskal gave 7 for the
/// first case (3), Kruskal and Prim both 7 for the third (3), and on the seeded multigraph Kruskal
/// 43 and Prim 49 (35).
#[test]
fn a_spanning_tree_takes_each_pairs_lightest_edge() {
    const MULTIGRAPH: &[(usize, usize, i64)] = &[
        (4, 1, 1),
        (9, 4, 14),
        (6, 8, 12),
        (2, 11, 7),
        (4, 7, 1),
        (3, 9, 15),
        (0, 1, 3),
        (11, 7, 14),
        (0, 8, 14),
        (5, 0, 7),
        (0, 9, 12),
        (10, 2, 20),
        (3, 11, 12),
        (9, 5, 19),
        (4, 10, 11),
        (9, 1, 16),
        (2, 11, 16),
        (9, 7, 6),
        (8, 5, 17),
        (3, 9, 4),
        (6, 0, 19),
        (4, 9, 9),
        (9, 5, 4),
        (5, 4, 18),
        (4, 7, 18),
        (3, 4, 5),
        (1, 9, 4),
        (11, 7, 8),
        (5, 8, 5),
        (11, 9, 4),
        (6, 2, 15),
        (10, 6, 13),
        (0, 2, 8),
        (10, 0, 8),
        (11, 0, 18),
        (7, 11, 3),
        (5, 0, 13),
        (7, 3, 2),
        (1, 4, 15),
        (8, 3, 15),
        (4, 3, 3),
        (1, 8, 17),
        (10, 2, 18),
        (8, 3, 9),
        (1, 6, 6),
        (0, 7, 11),
        (10, 6, 6),
        (9, 11, 14),
        (1, 11, 8),
        (8, 4, 14),
        (6, 4, 18),
        (9, 8, 2),
        (4, 1, 1),
        (10, 2, 11),
        (8, 11, 14),
        (3, 2, 3),
        (5, 8, 18),
    ];
    for (name, edges, want) in [
        (
            "parallel, the light one last",
            &[(0, 1, 5), (0, 1, 1), (1, 2, 2)][..],
            3.0,
        ),
        (
            "parallel, the light one first",
            &[(0, 1, 1), (0, 1, 5), (1, 2, 2)][..],
            3.0,
        ),
        (
            "opposite, the light one back",
            &[(0, 1, 5), (1, 0, 1), (1, 2, 2)][..],
            3.0,
        ),
        (
            "opposite, the light one forward",
            &[(0, 1, 1), (1, 0, 5), (1, 2, 2)][..],
            3.0,
        ),
        (
            "a seeded multigraph with parallel and opposite edges",
            MULTIGRAPH,
            35.0,
        ),
    ] {
        let db = GrafeoDB::new_in_memory();
        let nodes = edges.iter().map(|&(u, v, _)| u.max(v)).max().unwrap() + 1;
        let ids: Vec<_> = (0..nodes)
            .map(|_| db.create_node(&["V"]).unwrap())
            .collect();
        for &(u, v, w) in edges {
            let edge = db.create_edge(ids[u], ids[v], "E").unwrap();
            db.set_edge_property(edge, "w", Value::Int64(w)).unwrap();
        }
        assert_eq!(total(&db, "kruskal"), want, "{name}: kruskal");
        assert_eq!(total(&db, "prim"), want, "{name}: prim");
    }
}

/// `max_flow` on a multigraph: its edges as `(source name, target name, capacity)`.
fn max_flow_rows(edges: &[(&str, &str, i64)]) -> (f64, Vec<(String, String, f64)>) {
    let db = GrafeoDB::new_in_memory();
    let mut names: Vec<&str> = edges.iter().flat_map(|&(u, v, _)| [u, v]).collect();
    names.sort_unstable();
    names.dedup();
    let ids: HashMap<&str, _> = names
        .iter()
        .map(|&name| {
            let id = db.create_node(&["N"]).unwrap();
            db.set_node_property(id, "name", Value::from(name)).unwrap();
            (name, id)
        })
        .collect();
    for &(u, v, c) in edges {
        let edge = db.create_edge(ids[u], ids[v], "E").unwrap();
        db.set_edge_property(edge, "c", Value::Int64(c)).unwrap();
    }
    let name_of: HashMap<i64, &str> = ids
        .iter()
        .map(|(&name, id)| (i64::try_from(id.as_u64()).unwrap(), name))
        .collect();
    let id = |n: &str| Value::Int64(i64::try_from(ids[n].as_u64()).unwrap());
    let result = db
        .execute_with_params(
            "CALL grafeo.max_flow({source: $s, sink: $t, capacity: 'c'}) \
             YIELD source, target, flow, max_flow",
            HashMap::from([("s".to_string(), id("s")), ("t".to_string(), id("t"))]),
        )
        .unwrap();
    let mut total = 0.0;
    let rows = result
        .rows()
        .iter()
        .map(|row| {
            let (Value::Int64(u), Value::Int64(v)) = (&row[0], &row[1]) else {
                panic!("ids")
            };
            total = row[3].as_float64().unwrap();
            (
                name_of[u].to_string(),
                name_of[v].to_string(),
                row[2].as_float64().unwrap(),
            )
        })
        .collect();
    (total, rows)
}

/// A flow is one: conserved at every node but the source and the sink, the max flow out of the
/// source; no pair has more rows than edges, and the flows of a pair fit its edges' capacities.
fn assert_a_flow(
    name: &str,
    edges: &[(&str, &str, i64)],
    total: f64,
    rows: &[(String, String, f64)],
) {
    let mut balance: HashMap<&str, f64> = HashMap::new();
    for (u, v, f) in rows {
        assert!(*f > 0.0, "{name}: a row for an edge that carries nothing");
        *balance.entry(u.as_str()).or_default() -= f;
        *balance.entry(v.as_str()).or_default() += f;
    }
    for (node, net) in &balance {
        let want = match *node {
            "s" => -total,
            "t" => total,
            _ => 0.0,
        };
        assert!(
            (net - want).abs() < 1e-9,
            "{name}: {node} nets {net}, want {want}"
        );
    }
    let mut pairs: HashMap<(&str, &str), (Vec<f64>, Vec<f64>)> = HashMap::new();
    for &(u, v, c) in edges {
        pairs.entry((u, v)).or_default().1.push(c as f64);
    }
    for (u, v, f) in rows {
        pairs
            .get_mut(&(u.as_str(), v.as_str()))
            .unwrap_or_else(|| panic!("{name}: a row for no edge {u} -> {v}"))
            .0
            .push(*f);
    }
    for ((u, v), (mut flows, mut capacities)) in pairs {
        assert!(
            flows.len() <= capacities.len(),
            "{name}: {u} -> {v} has more rows than edges"
        );
        flows.sort_by(|a, b| b.total_cmp(a));
        capacities.sort_by(|a, b| b.total_cmp(a));
        for (f, c) in flows.iter().zip(&capacities) {
            assert!(
                f <= c,
                "{name}: {u} -> {v} carries {flows:?} over {capacities:?}"
            );
        }
    }
}

/// One row per edge that carries flow, with its own capacity, as `min_cost_max_flow` gives them;
/// the flow value is networkx's (`maximum_flow` with each edge through its own middle node).
/// 0.6.1 gave parallel edges one row with their summed flow.
#[test]
fn max_flow_gives_a_row_per_edge_that_carries_flow() {
    for (name, edges, want, rows) in [
        (
            "two parallel edges",
            &[("s", "t", 5), ("s", "t", 3)][..],
            8.0,
            Some(vec![("s", "t", 5.0), ("s", "t", 3.0)]),
        ),
        (
            "parallel edges in a path",
            &[("s", "a", 10), ("a", "t", 4), ("a", "t", 4), ("a", "t", 1)][..],
            9.0,
            Some(vec![
                ("a", "t", 4.0),
                ("a", "t", 4.0),
                ("a", "t", 1.0),
                ("s", "a", 9.0),
            ]),
        ),
        (
            "a parallel and an opposite edge",
            &[("s", "t", 3), ("t", "s", 3), ("s", "t", 2)][..],
            5.0,
            Some(vec![("s", "t", 3.0), ("s", "t", 2.0)]),
        ),
        (
            "opposite edges",
            &[("s", "a", 5), ("a", "t", 5), ("a", "s", 2)][..],
            5.0,
            None,
        ),
        (
            "opposite edges across the middle",
            &[
                ("s", "a", 4),
                ("s", "b", 4),
                ("a", "b", 3),
                ("b", "a", 2),
                ("a", "t", 3),
                ("b", "t", 5),
            ][..],
            8.0,
            None,
        ),
        (
            "a seeded multigraph",
            &[
                ("n4", "n3", 4),
                ("n0", "n4", 5),
                ("n6", "n2", 7),
                ("n7", "s", 4),
                ("s", "n5", 3),
                ("n0", "n3", 3),
                ("n2", "s", 2),
                ("n7", "s", 7),
                ("n2", "n1", 2),
                ("n3", "n5", 3),
                ("n6", "s", 2),
                ("t", "n0", 3),
                ("s", "n2", 6),
                ("n1", "s", 3),
                ("n4", "s", 5),
                ("n4", "n0", 2),
                ("t", "n4", 1),
                ("n5", "n4", 4),
                ("s", "t", 3),
                ("t", "n5", 7),
                ("n4", "n6", 5),
                ("n6", "s", 1),
                ("n5", "n3", 3),
                ("n0", "t", 1),
                ("t", "n0", 7),
                ("t", "n6", 7),
                ("n7", "t", 4),
                ("n1", "n4", 6),
                ("n5", "n4", 8),
                ("t", "n2", 9),
                ("n1", "n7", 5),
                ("n4", "n3", 7),
                ("n2", "n5", 2),
                ("n1", "n6", 3),
                ("n1", "s", 4),
                ("n0", "n5", 4),
                ("t", "n1", 9),
                ("n7", "n0", 8),
                ("n3", "t", 4),
                ("n4", "t", 8),
            ][..],
            10.0,
            None,
        ),
    ] {
        let (total, found) = max_flow_rows(edges);
        assert_eq!(total, want, "{name}: the flow value");
        assert_a_flow(name, edges, total, &found);
        if let Some(rows) = rows {
            let mut found: Vec<(String, String, f64)> = found;
            found.sort_by(|a, b| (&a.0, &a.1).cmp(&(&b.0, &b.1)).then(b.2.total_cmp(&a.2)));
            let mut want_rows: Vec<(String, String, f64)> = rows
                .into_iter()
                .map(|(u, v, f)| (u.to_string(), v.to_string(), f))
                .collect();
            want_rows.sort_by(|a, b| (&a.0, &a.1).cmp(&(&b.0, &b.1)).then(b.2.total_cmp(&a.2)));
            assert_eq!(found, want_rows, "{name}: the rows");
        }
    }
}
