//! A shortest-path search holds its maps and queues in its path search budget
//! and as a grant of the statement's memory (0.6.4): `ANY SHORTEST` and
//! `shortestPath` over a wide graph, whose visited map passes a small budget,
//! fail with `GRAFEO-S001` naming the budget (before 0.6.4 they answered,
//! holding the map outside it), past a small query memory limit they fail
//! naming that limit, and the statement gives its memory back either way.
//!
//! ```bash
//! cargo test -p grafeo-engine --all-features --test main shortest_path_grants::
//! ```

#![cfg(all(feature = "gql", feature = "cypher"))]

use grafeo_common::memory::buffer::MemoryRegion;
use grafeo_common::types::Value;
use grafeo_common::utils::error::{Error, LimitBound, LimitRefusal};
use grafeo_engine::{Config, GrafeoDB};

const KIB: usize = 1024;

/// The hub's leaves: a visited map of 5,001 nodes, about 300 KiB.
const LEAVES: usize = 5_000;

/// `ANY SHORTEST` and `shortestPath` from the hub to the target.
const QUERIES: [(&str, &str); 2] = [
    (
        "gql",
        "MATCH p = ANY SHORTEST (s:Hub)-[:E]->+(t:Target) RETURN length(p) AS l",
    ),
    (
        "cypher",
        "MATCH p = shortestPath((s:Hub)-[:E*]->(t:Target)) RETURN length(p) AS l",
    ),
];

/// A hub with an edge to each of [`LEAVES`] leaves, the first of which leads
/// on to the target: the shortest path has two edges, and the search from the
/// hub visits every leaf first.
fn wide(config: Config) -> GrafeoDB {
    let db = GrafeoDB::with_config(config).unwrap();
    let hub = db.create_node(&["Hub"]).unwrap();
    let target = db.create_node(&["Target"]).unwrap();
    for leaf in 0..LEAVES {
        let node = db.create_node(&["Leaf"]).unwrap();
        db.create_edge(hub, node, "E").unwrap();
        if leaf == 0 {
            db.create_edge(node, target, "E").unwrap();
        }
    }
    db
}

/// The bytes the statements of `db` hold.
fn execution_bytes(db: &GrafeoDB) -> usize {
    db.buffer_manager().stats().region_allocated[MemoryRegion::ExecutionBuffers.index()]
}

/// The bound `error` names: `GRAFEO-S001` of a memory bound.
fn refused_by(error: &Error) -> LimitBound {
    assert_eq!(error.error_code().as_str(), "GRAFEO-S001", "{error}");
    match error.limit_refusal() {
        Some(LimitRefusal::Memory { bound, .. }) => *bound,
        other => panic!("a memory bound's refusal, got {other:?}: {error}"),
    }
}

#[test]
fn a_shortest_path_answers_within_its_budget() {
    let db = wide(Config::in_memory());
    for (language, query) in QUERIES {
        let result = db
            .execute_language(query, language, None)
            .unwrap_or_else(|error| panic!("{language}: {error}"));
        assert_eq!(result.rows(), [vec![Value::Int64(2)]], "{language}");
    }
}

#[test]
fn a_shortest_path_over_a_wide_graph_refuses_past_its_budget() {
    let db = wide(Config::in_memory().with_path_search_budget(64 * KIB));
    let before = execution_bytes(&db);
    for (language, query) in QUERIES {
        let error = db.execute_language(query, language, None).expect_err(query);
        assert_eq!(
            refused_by(&error),
            LimitBound::PathSearchBudget,
            "{language}"
        );
        let message = error.to_string();
        for named in ["A shortest path search", "64 KiB", "ALL SHORTEST"] {
            assert!(
                message.contains(named),
                "{language}: names `{named}`: {message}"
            );
        }
        assert_eq!(execution_bytes(&db), before, "{language}: given back");
    }
}

#[test]
fn a_shortest_path_refuses_past_the_query_memory_limit() {
    let db = wide(Config::in_memory().with_query_memory_limit(64 * KIB));
    let before = execution_bytes(&db);
    for (language, query) in QUERIES {
        let error = db.execute_language(query, language, None).expect_err(query);
        assert_eq!(
            refused_by(&error),
            LimitBound::QueryMemoryLimit,
            "{language}"
        );
        assert_eq!(execution_bytes(&db), before, "{language}: given back");
    }
}
