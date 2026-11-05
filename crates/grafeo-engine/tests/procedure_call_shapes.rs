//! A procedure's columns follow its arguments (0.6.1): `dijkstra` to a
//! `target` gives the one path (source, target, distance, path), `kcore` with
//! a `k` the members of that core (node_id, in_k_core), a normalized
//! `degree_centrality` one score (node_id, centrality); `kruskal` and `prim`
//! give `total_weight` and `min_cost_max_flow` gives `total_cost` on every
//! row, which the plan cut before. A map key that names no parameter and a
//! positional argument past the last parameter are errors naming the
//! parameters, in GQL and Cypher alike.
//!
//! The graph: Amsterdam (0) -> Berlin (1) 650 km, Berlin -> Paris (2) 1,050,
//! Amsterdam -> Paris 500, Berlin -> Prague (3) 350, Paris -> Prague 1,030,
//! each with a capacity and a cost. Every expected value is worked out by
//! hand from these edges.
//!
//! ```bash
//! cargo test -p grafeo-engine --all-features --test main procedure_call_shapes::
//! ```

#![cfg(all(feature = "gql", feature = "cypher", feature = "algos"))]

use std::collections::HashMap;

use grafeo_common::types::Value;
use grafeo_common::utils::error::Result;
use grafeo_engine::GrafeoDB;
use grafeo_engine::database::QueryResult;

fn routes() -> GrafeoDB {
    let db = GrafeoDB::new_in_memory();
    db.execute(
        "INSERT (a:City {name: 'Amsterdam'}), (b:City {name: 'Berlin'}), \
         (c:City {name: 'Paris'}), (d:City {name: 'Prague'}), \
         (a)-[:ROUTE {km: 650, capacity: 3, cost: 2}]->(b), \
         (b)-[:ROUTE {km: 1050, capacity: 2, cost: 1}]->(c), \
         (a)-[:ROUTE {km: 500, capacity: 1, cost: 5}]->(c), \
         (b)-[:ROUTE {km: 350, capacity: 4, cost: 1}]->(d), \
         (c)-[:ROUTE {km: 1030, capacity: 2, cost: 3}]->(d)",
    )
    .unwrap();
    db
}

fn run(db: &GrafeoDB, language: &str, query: &str) -> Result<QueryResult> {
    db.execute_language(query, language, None)
}

fn ok(db: &GrafeoDB, language: &str, query: &str) -> QueryResult {
    run(db, language, query).unwrap_or_else(|error| panic!("{language} `{query}`: {error}"))
}

/// The call `call` in GQL, and in Cypher with `RETURN *` after a YIELD.
fn both(call: &str) -> [(&'static str, String); 2] {
    let cypher = if call.contains(" YIELD ") {
        format!("{call} RETURN *")
    } else {
        call.to_string()
    };
    [("gql", call.to_string()), ("cypher", cypher)]
}

/// The rows of `result` sorted, for a procedure whose row order is not part
/// of its contract.
fn sorted_rows(result: &QueryResult) -> Vec<Vec<Value>> {
    let mut rows = result.rows().to_vec();
    rows.sort_by_key(|row| format!("{row:?}"));
    rows
}

fn int(n: i64) -> Value {
    Value::Int64(n)
}

fn float(x: f64) -> Value {
    Value::Float64(x)
}

#[test]
fn dijkstra_to_a_target_gives_the_one_path() {
    let db = routes();
    // Amsterdam to Prague: over Berlin 650 + 350, over Paris 500 + 1,030.
    for call in [
        "CALL grafeo.dijkstra({source: 0, target: 3, weight: 'km'})",
        "CALL grafeo.dijkstra(0, 3, 'km')",
        "CALL grafeo.dijkstra({source: 0, target: 3, weight: 'km'}) YIELD source, target, distance, path",
    ] {
        for (language, query) in both(call) {
            let result = ok(&db, language, &query);
            assert_eq!(
                result.columns,
                ["source", "target", "distance", "path"],
                "{query}"
            );
            assert_eq!(
                result.rows(),
                [vec![
                    int(0),
                    int(3),
                    float(1000.0),
                    Value::from("0 -> 1 -> 3")
                ]],
                "{language} `{query}`"
            );
        }
    }
    // Without a weight every edge counts 1: the direct route to Paris.
    let result = ok(&db, "gql", "CALL grafeo.dijkstra(0, 2)");
    assert_eq!(
        result.rows(),
        [vec![int(0), int(2), float(1.0), Value::from("0 -> 2")]]
    );
}

#[test]
fn dijkstra_without_a_target_gives_every_distance() {
    let db = routes();
    for call in [
        "CALL grafeo.dijkstra({source: 0, weight: 'km'})",
        // A null target leaves the parameter out: the same shape.
        "CALL grafeo.dijkstra({source: 0, target: null, weight: 'km'})",
        "CALL grafeo.dijkstra({source: 0, weight: 'km'}) YIELD node_id, distance",
    ] {
        for (language, query) in both(call) {
            let result = ok(&db, language, &query);
            assert_eq!(result.columns, ["node_id", "distance"], "{query}");
            assert_eq!(
                sorted_rows(&result),
                [
                    vec![int(0), float(0.0)],
                    vec![int(1), float(650.0)],
                    vec![int(2), float(500.0)],
                    vec![int(3), float(1000.0)],
                ],
                "{language} `{query}`"
            );
        }
    }
}

#[test]
fn kcore_with_k_gives_its_members() {
    // A triangle Mia, Butch, Django (0, 1, 2) and Hans (3) on Django alone:
    // the 2-core is the triangle, the 1-core everyone, the 3-core no one. A
    // call with k gives the members only.
    let pendant = GrafeoDB::new_in_memory();
    pendant
        .execute(
            "INSERT (m:P {name: 'Mia'})-[:K]->(b:P {name: 'Butch'})-[:K]->(d:P {name: 'Django'}), \
             (d)-[:K]->(m), (:P {name: 'Hans'})-[:K]->(d)",
        )
        .unwrap();
    for (k, members) in [(1, vec![0, 1, 2, 3]), (2, vec![0, 1, 2]), (3, vec![])] {
        for (language, query) in both(&format!(
            "CALL grafeo.kcore({{k: {k}}}) YIELD node_id, in_k_core"
        )) {
            let result = ok(&pendant, language, &query);
            assert_eq!(result.columns, ["node_id", "in_k_core"]);
            assert_eq!(
                sorted_rows(&result),
                members
                    .iter()
                    .map(|&node| vec![int(node), Value::Bool(true)])
                    .collect::<Vec<_>>(),
                "{language} `{query}`"
            );
        }
    }
    for language in ["gql", "cypher"] {
        let result = ok(&pendant, language, "CALL grafeo.kcore({k: 2})");
        assert_eq!(result.columns, ["node_id", "in_k_core"], "{language}");
        assert_eq!(result.row_count(), 3, "{language}");
        assert!(result.rows().iter().all(|row| row.len() == 2), "{language}");
    }
    let result = ok(&pendant, "gql", "CALL grafeo.kcore()");
    assert_eq!(result.columns, ["node_id", "core_number", "max_core"]);
    assert_eq!(
        sorted_rows(&result),
        [
            vec![int(0), int(2), int(2)],
            vec![int(1), int(2), int(2)],
            vec![int(2), int(2), int(2)],
            vec![int(3), int(1), int(2)],
        ]
    );
    // Every city of the routes has two neighbours or more: one 2-core.
    let db = routes();
    let result = ok(&db, "gql", "CALL grafeo.kcore()");
    assert_eq!(result.columns, ["node_id", "core_number", "max_core"]);
    assert_eq!(
        sorted_rows(&result),
        (0..4)
            .map(|node| vec![int(node), int(2), int(2)])
            .collect::<Vec<_>>()
    );
    // A null k is no k.
    let result = ok(&db, "gql", "CALL grafeo.kcore({k: null})");
    assert_eq!(result.columns, ["node_id", "core_number", "max_core"]);
}

#[test]
fn a_normalized_degree_centrality_gives_one_score() {
    let db = routes();
    for (language, query) in
        both("CALL grafeo.degree_centrality({normalized: true}) YIELD node_id, centrality")
    {
        let result = ok(&db, language, &query);
        assert_eq!(result.columns, ["node_id", "centrality"]);
        // Total degree over the 3 other cities.
        assert_eq!(
            sorted_rows(&result),
            [
                vec![int(0), float(2.0 / 3.0)],
                vec![int(1), float(1.0)],
                vec![int(2), float(1.0)],
                vec![int(3), float(2.0 / 3.0)],
            ],
            "{language} `{query}`"
        );
    }
    for query in [
        "CALL grafeo.degree_centrality()",
        "CALL grafeo.degree_centrality({normalized: false})",
    ] {
        let result = ok(&db, "gql", query);
        assert_eq!(
            result.columns,
            ["node_id", "in_degree", "out_degree", "total_degree"],
            "{query}"
        );
        assert_eq!(
            sorted_rows(&result),
            [
                vec![int(0), int(0), int(2), int(2)],
                vec![int(1), int(1), int(2), int(3)],
                vec![int(2), int(2), int(1), int(3)],
                vec![int(3), int(2), int(0), int(2)],
            ],
            "{query}"
        );
    }
}

#[test]
fn spanning_trees_give_their_total_weight_on_every_row() {
    let db = routes();
    // The tree: 350 (Berlin-Prague), 500 (Amsterdam-Paris), 650
    // (Amsterdam-Berlin), 1,500 in all.
    for procedure in ["kruskal", "prim"] {
        for (language, query) in both(&format!(
            "CALL grafeo.{procedure}({{weight: 'km'}}) YIELD source, target, weight, total_weight"
        )) {
            let result = ok(&db, language, &query);
            assert_eq!(
                result.columns,
                ["source", "target", "weight", "total_weight"]
            );
            let mut weights: Vec<f64> = result
                .rows()
                .iter()
                .map(|row| match (&row[2], &row[3]) {
                    (Value::Float64(weight), Value::Float64(total)) => {
                        assert!(
                            (total - 1500.0).abs() < 1e-9,
                            "{language} `{query}`: {row:?}"
                        );
                        *weight
                    }
                    other => panic!("{language} `{query}`: {other:?}"),
                })
                .collect();
            weights.sort_by(f64::total_cmp);
            assert_eq!(weights, [350.0, 500.0, 650.0], "{language} `{query}`");
        }
        // Without YIELD the call gives every declared column, the total too.
        for language in ["gql", "cypher"] {
            let query = format!("CALL grafeo.{procedure}({{weight: 'km'}})");
            let result = ok(&db, language, &query);
            assert_eq!(
                result.columns,
                ["source", "target", "weight", "total_weight"],
                "{language} `{query}`"
            );
            for row in result.rows() {
                assert_eq!(row.len(), 4, "{language} `{query}`: {row:?}");
                assert_eq!(row[3], float(1500.0), "{language} `{query}`");
            }
        }
        // YIELD of the total alone.
        let result = ok(
            &db,
            "gql",
            &format!("CALL grafeo.{procedure}({{weight: 'km'}}) YIELD total_weight"),
        );
        assert_eq!(result.rows(), vec![vec![float(1500.0)]; 3], "{procedure}");
    }
}

#[test]
fn a_min_cost_flow_gives_its_total_cost_on_every_row() {
    let db = routes();
    for language in ["gql", "cypher"] {
        let query = "CALL grafeo.min_cost_max_flow({source: 0, sink: 3, capacity: 'capacity', cost: 'cost'})";
        let result = ok(&db, language, query);
        assert_eq!(
            result.columns,
            ["source", "target", "flow", "cost", "max_flow", "total_cost"],
            "{language}"
        );
        for row in result.rows() {
            assert_eq!(row.get(5), Some(&float(17.0)), "{language}: {row:?}");
        }
    }
    // Amsterdam sends its 4 out: 3 to Berlin (cost 2) and 1 to Paris (5);
    // Berlin passes 3 to Prague (1), Paris 1 to Prague (3): 6 + 5 + 3 + 3.
    for (language, query) in both(
        "CALL grafeo.min_cost_max_flow({source: 0, sink: 3, capacity: 'capacity', cost: 'cost'}) \
         YIELD source, target, flow, cost, max_flow, total_cost",
    ) {
        let result = ok(&db, language, &query);
        assert_eq!(
            result.columns,
            ["source", "target", "flow", "cost", "max_flow", "total_cost"]
        );
        assert_eq!(
            sorted_rows(&result),
            [
                vec![
                    int(0),
                    int(1),
                    float(3.0),
                    float(2.0),
                    float(4.0),
                    float(17.0)
                ],
                vec![
                    int(0),
                    int(2),
                    float(1.0),
                    float(5.0),
                    float(4.0),
                    float(17.0)
                ],
                vec![
                    int(1),
                    int(3),
                    float(3.0),
                    float(1.0),
                    float(4.0),
                    float(17.0)
                ],
                vec![
                    int(2),
                    int(3),
                    float(1.0),
                    float(3.0),
                    float(4.0),
                    float(17.0)
                ],
            ],
            "{language} `{query}`"
        );
    }
}

/// The columns `grafeo.procedures()` lists are those of a call with the
/// defaults, and a call without YIELD gives them.
#[test]
fn the_listed_columns_are_those_of_a_call_with_the_defaults() {
    let db = routes();
    let listed: HashMap<String, String> = ok(&db, "gql", "CALL grafeo.procedures()")
        .rows()
        .iter()
        .map(|row| match (&row[0], &row[3]) {
            (Value::String(name), Value::String(columns)) => {
                (name.to_string(), columns.to_string())
            }
            other => panic!("{other:?}"),
        })
        .collect();
    for (procedure, call) in [
        ("grafeo.kruskal", "CALL grafeo.kruskal({weight: 'km'})"),
        ("grafeo.prim", "CALL grafeo.prim({weight: 'km'})"),
        (
            "grafeo.min_cost_max_flow",
            "CALL grafeo.min_cost_max_flow({source: 0, sink: 3, capacity: 'capacity', cost: 'cost'})",
        ),
        ("grafeo.dijkstra", "CALL grafeo.dijkstra({source: 0})"),
        ("grafeo.kcore", "CALL grafeo.kcore()"),
        (
            "grafeo.degree_centrality",
            "CALL grafeo.degree_centrality()",
        ),
    ] {
        let columns = ok(&db, "gql", call).columns.join(", ");
        assert_eq!(listed[procedure], columns, "{call}");
    }
}

/// The error of `query` in `language`, which must be a semantic error.
fn semantic_error(db: &GrafeoDB, language: &str, query: &str) -> String {
    let error = run(db, language, query)
        .err()
        .unwrap_or_else(|| panic!("{language} `{query}` should fail"));
    assert_eq!(
        error.error_code().as_str(),
        "GRAFEO-Q002",
        "{language} `{query}`: {error}"
    );
    error.to_string()
}

#[test]
fn a_map_key_that_names_no_parameter_is_an_error_naming_them() {
    let db = routes();
    for (query, expected) in [
        (
            "CALL grafeo.kruskal({weightt: 'km'})",
            "Argument 'weightt' is no parameter of grafeo.kruskal (it has: weight)",
        ),
        (
            "CALL grafeo.dijkstra({source: 0, targett: 3})",
            "Argument 'targett' is no parameter of grafeo.dijkstra (it has: source, target, weight)",
        ),
        (
            "CALL grafeo.pagerank({dampingg: 0.5})",
            "Argument 'dampingg' is no parameter of grafeo.pagerank \
             (it has: damping, max_iterations, tolerance, directed)",
        ),
        (
            "CALL grafeo.connected_components({k: 1})",
            "Argument 'k' is no parameter of grafeo.connected_components (it has: none)",
        ),
    ] {
        for language in ["gql", "cypher"] {
            let message = semantic_error(&db, language, query);
            assert!(
                message.contains(expected),
                "{language} `{query}`: {message}"
            );
        }
    }
    // `projection` is taken by every procedure: it reaches the projection
    // lookup instead of failing as an unknown parameter.
    let message = semantic_error(&db, "gql", "CALL grafeo.pagerank({projection: 'nope'})");
    assert!(
        message.contains("Projection 'nope' does not exist") && !message.contains("no parameter"),
        "{message}"
    );
}

#[test]
fn a_positional_argument_past_the_last_parameter_is_an_error() {
    let db = routes();
    for (query, expected) in [
        (
            "CALL grafeo.dijkstra(0, 3, 'km', 1)",
            "grafeo.dijkstra takes at most 3 arguments, got 4 (its parameters: source, target, weight)",
        ),
        (
            "CALL grafeo.pagerank(0.85, 20, 0.000001, true, 8)",
            "grafeo.pagerank takes at most 4 arguments, got 5",
        ),
        (
            "CALL grafeo.connected_components(1)",
            "grafeo.connected_components takes at most 0 arguments, got 1 (its parameters: none)",
        ),
    ] {
        for language in ["gql", "cypher"] {
            let message = semantic_error(&db, language, query);
            assert!(
                message.contains(expected),
                "{language} `{query}`: {message}"
            );
        }
    }
    // As many as there are parameters is fine.
    ok(&db, "gql", "CALL grafeo.pagerank(0.85, 20, 0.000001, true)");
}

/// A YIELD of a column of another shape (as the plan declared before)
/// fails, listing the columns this call gives.
#[test]
fn a_yield_of_another_shapes_column_lists_the_real_ones() {
    let db = routes();
    for (call, expected) in [
        (
            "CALL grafeo.dijkstra({source: 0, target: 3}) YIELD node_id",
            "YIELD column 'node_id' is no column of the procedure (it has: source, target, distance, path)",
        ),
        (
            "CALL grafeo.kcore({k: 2}) YIELD core_number",
            "YIELD column 'core_number' is no column of the procedure (it has: node_id, in_k_core)",
        ),
        (
            "CALL grafeo.degree_centrality({normalized: true}) YIELD total_degree",
            "YIELD column 'total_degree' is no column of the procedure (it has: node_id, centrality)",
        ),
        (
            "CALL grafeo.dijkstra({source: 0}) YIELD path",
            "YIELD column 'path' is no column of the procedure (it has: node_id, distance)",
        ),
    ] {
        for (language, query) in both(call) {
            let message = semantic_error(&db, language, &query);
            assert!(
                message.contains(expected),
                "{language} `{query}`: {message}"
            );
        }
    }
}

/// Arguments in a map parameter decide the shape as a map literal does.
#[test]
fn a_map_parameter_decides_the_shape_too() {
    let db = routes();
    let mut args = std::collections::BTreeMap::new();
    args.insert("source".into(), int(0));
    args.insert("target".into(), int(3));
    args.insert("weight".into(), Value::from("km"));
    let params = HashMap::from([("args".to_string(), Value::Map(std::sync::Arc::new(args)))]);
    for language in ["gql", "cypher"] {
        let result = db
            .execute_language(
                "CALL grafeo.dijkstra($args)",
                language,
                Some(params.clone()),
            )
            .unwrap_or_else(|error| panic!("{language}: {error}"));
        assert_eq!(
            result.columns,
            ["source", "target", "distance", "path"],
            "{language}"
        );
        assert_eq!(result.rows()[0][2], float(1000.0), "{language}");
    }
}
