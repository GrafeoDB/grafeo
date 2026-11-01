//! Decimal literals must keep their numeric weights through algorithm calls (#678).

#![cfg(all(feature = "gql", feature = "algos"))]

use grafeo_common::types::Value::{self, Float64, Int64};
use grafeo_engine::GrafeoDB;

#[test]
fn decimal_weights_choose_the_shorter_route() {
    let db = GrafeoDB::new_in_memory();
    db.execute(
        "INSERT (s:Node {name: 'source'})-[:R {w: 0.2500}]->(m:Node), \
         (m)-[:R {w: CAST(0.5 AS DOUBLE)}]->(t:Node), (s)-[:R {w: 1}]->(t)",
    )
    .unwrap();

    for query in [
        "CALL grafeo.dijkstra({source: 0, weight: 'w'})",
        "CALL grafeo.sssp({source: 'source', weight: 'w'})",
    ] {
        let result = db.execute(query).unwrap();
        assert_eq!(result.columns, ["node_id", "distance"]);
        assert_eq!(
            result.rows(),
            &[
                vec![Int64(0), Float64(0.0)],
                vec![Int64(1), Float64(0.25)],
                vec![Int64(2), Float64(0.75)],
            ],
            "{query}"
        );
    }

    let result = db
        .execute("CALL grafeo.bellman_ford({source: 0, weight: 'w'})")
        .unwrap();
    assert_eq!(
        result.columns,
        ["node_id", "distance", "has_negative_cycle"]
    );
    assert_eq!(
        result.rows(),
        &[
            vec![Int64(0), Float64(0.0), Value::Bool(false)],
            vec![Int64(1), Float64(0.25), Value::Bool(false)],
            vec![Int64(2), Float64(0.75), Value::Bool(false)],
        ]
    );

    let result = db
        .execute("CALL grafeo.floyd_warshall({weight: 'w'})")
        .unwrap();
    assert_eq!(result.columns, ["source", "target", "distance"]);
    assert_eq!(
        result.rows(),
        &[
            vec![Int64(0), Int64(0), Float64(0.0)],
            vec![Int64(0), Int64(1), Float64(0.25)],
            vec![Int64(0), Int64(2), Float64(0.75)],
            vec![Int64(1), Int64(1), Float64(0.0)],
            vec![Int64(1), Int64(2), Float64(0.5)],
            vec![Int64(2), Int64(2), Float64(0.0)],
        ]
    );
}

#[test]
fn decimal_weights_choose_the_lighter_spanning_tree() {
    let db = GrafeoDB::new_in_memory();
    db.execute(
        "INSERT (s:Node)-[:R {w: 4.5}]->(m:Node), \
         (s)-[:R {w: 2}]->(t:Node), (t)-[:R {w: CAST(3 AS DOUBLE)}]->(m)",
    )
    .unwrap();

    for query in [
        "CALL grafeo.kruskal({weight: 'w'}) YIELD source, target, weight",
        "CALL grafeo.prim({weight: 'w', start: 0}) YIELD source, target, weight",
    ] {
        let result = db.execute(query).unwrap();
        assert_eq!(result.columns, ["source", "target", "weight"]);
        assert_eq!(
            result.rows(),
            &[
                vec![Int64(0), Int64(2), Float64(2.0)],
                vec![Int64(2), Int64(1), Float64(3.0)],
            ],
            "{query}"
        );
    }
}

#[test]
fn decimal_capacities_bound_the_flow() {
    let db = GrafeoDB::new_in_memory();
    db.execute(
        "INSERT (s:Node)-[:R {cap: 2.5}]->(m:Node), \
         (m)-[:R {cap: CAST(4 AS DOUBLE)}]->(t:Node), (s)-[:R {cap: 2}]->(t)",
    )
    .unwrap();

    let result = db
        .execute("CALL grafeo.max_flow({source: 0, sink: 2, capacity: 'cap'})")
        .unwrap();
    assert_eq!(result.columns, ["source", "target", "flow", "max_flow"]);
    assert_eq!(
        result.rows(),
        &[
            vec![Int64(0), Int64(1), Float64(2.5), Float64(4.5)],
            vec![Int64(0), Int64(2), Float64(2.0), Float64(4.5)],
            vec![Int64(1), Int64(2), Float64(2.5), Float64(4.5)],
        ]
    );
}

#[test]
fn decimal_costs_are_included_in_min_cost_flow() {
    let db = GrafeoDB::new_in_memory();
    db.execute(
        "INSERT (s:Node)-[:R {cap: CAST(2.5 AS DOUBLE), cost: 0.5}]->(m:Node), \
         (m)-[:R {cap: CAST(4 AS DOUBLE), cost: CAST(0.25 AS DOUBLE)}]->(t:Node), \
         (s)-[:R {cap: 2, cost: 3}]->(t)",
    )
    .unwrap();

    let result = db
        .execute(
            "CALL grafeo.min_cost_max_flow({source: 0, sink: 2, capacity: 'cap', cost: 'cost'}) YIELD source, target, flow, cost, max_flow",
        )
        .unwrap();
    assert_eq!(
        result.columns,
        ["source", "target", "flow", "cost", "max_flow"]
    );
    assert_eq!(
        result.rows(),
        &[
            vec![Int64(0), Int64(1), Float64(2.5), Float64(0.5), Float64(4.5),],
            vec![Int64(0), Int64(2), Float64(2.0), Float64(3.0), Float64(4.5),],
            vec![
                Int64(1),
                Int64(2),
                Float64(2.5),
                Float64(0.25),
                Float64(4.5),
            ],
        ]
    );
    let total_cost: f64 = result
        .rows()
        .iter()
        .map(|row| row[2].as_float64().unwrap() * row[3].as_float64().unwrap())
        .sum();
    assert_eq!(total_cost, 7.875);
}
