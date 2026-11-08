//! Subgraph isomorphism's `pattern_edges` refuses only malformed input, with 22023 naming the part
//! (the ruling of 2026-11-06): an endpoint at or above `pattern_nodes`, a negative endpoint, and a
//! part that does not parse. Each part and each endpoint is trimmed, so `"0-1, 1-2"` is two edges;
//! before, a spaced part was dropped and the pattern counted one edge. A self-loop, a duplicate, an
//! empty string and a trailing comma keep their meaning, and a valid pattern is never refused.
//!
//! ```bash
//! cargo test -p grafeo-adapters --all-features --test main pattern_edges_rules::
//! ```

#![cfg(feature = "algos")]

use grafeo_adapters::plugins::Parameters;
use grafeo_adapters::plugins::algorithms::{GraphAlgorithm, SubgraphIsomorphismAlgorithm};
use grafeo_common::types::{NodeId, ReadView, Value};
use grafeo_common::utils::gqlstatus::GqlStatus;
use grafeo_core::graph::rowgroup::RowGroupStore;

/// A directed triangle and a tail: Alix -> Gus -> Vincent -> Alix, Vincent -> Mia.
fn graph() -> RowGroupStore {
    let store = RowGroupStore::new();
    let ids: Vec<NodeId> = (0..4).map(|_| store.create_node(&["Person"])).collect();
    for (src, dst) in [(0, 1), (1, 2), (2, 0), (2, 3)] {
        store.create_edge(ids[src], ids[dst], "KNOWS");
    }
    store
}

fn count(store: &RowGroupStore, edges: &str, nodes: i64) -> Result<i64, (String, String)> {
    let mut params = Parameters::new();
    params.set_string("pattern_edges", edges);
    params.set_int("pattern_nodes", nodes);
    SubgraphIsomorphismAlgorithm
        .execute(store, ReadView::latest(), &params)
        .map(|result| match &result.rows[0][0] {
            Value::Int64(n) => *n,
            other => panic!("a count, not {other:?}"),
        })
        .map_err(|error| {
            (
                GqlStatus::from(&error).as_str().to_string(),
                error.to_string(),
            )
        })
}

#[test]
fn a_spaced_pattern_counts_every_edge() {
    let store = graph();
    let tight = count(&store, "0-1,1-2", 3).unwrap();
    for spaced in ["0-1, 1-2", " 0-1,1-2 ", "0 - 1 , 1 - 2", "0-1,\t1-2"] {
        assert_eq!(count(&store, spaced, 3), Ok(tight), "{spaced:?}");
    }
    // Two edges, not the one a dropped part left.
    assert_ne!(tight, count(&store, "0-1", 3).unwrap());
}

#[test]
fn an_empty_part_adds_no_edge() {
    let store = graph();
    let two = count(&store, "0-1,1-2", 3).unwrap();
    for form in ["0-1,1-2,", ",0-1,1-2", "0-1,,1-2", "0-1, ,1-2"] {
        assert_eq!(count(&store, form, 3), Ok(two), "{form:?}");
    }
    // No edges at all: accepted, as today.
    assert!(count(&store, "", 2).is_ok());
}

#[test]
fn a_malformed_part_is_refused_naming_it() {
    let store = graph();
    for (edges, nodes, part) in [
        ("0-3", 3, "0-3"),
        ("0-1,1-5", 3, "1-5"),
        ("2-0,3-0", 3, "3-0"),
        ("-1-2", 3, "-1-2"),
        ("0--1", 3, "0--1"),
        ("0-x", 3, "0-x"),
        ("0-1-2", 3, "0-1-2"),
        ("0_1", 3, "0_1"),
        ("0-1,a", 3, "a"),
        ("0-", 3, "0-"),
        ("-", 3, "-"),
        ("0-1", 0, "0-1"),
    ] {
        match count(&store, edges, nodes) {
            Err((status, message)) => {
                assert_eq!(status, "22023", "{edges:?} with {nodes} nodes: {message}");
                assert!(
                    message.contains(part) && message.contains("pattern_edges"),
                    "{edges:?}: the message names neither the part {part:?} nor the argument: {message}"
                );
            }
            Ok(found) => panic!("{edges:?} with {nodes} nodes counted {found}, want 22023"),
        }
    }
}

/// What the algorithm defines stays: a self-loop and a duplicate run, and a valid pattern is never
/// refused.
#[test]
fn a_valid_pattern_runs() {
    let store = graph();
    for (edges, nodes) in [
        ("0-1,1-2,2-0", 3),
        ("0-1", 2),
        ("1-0", 2),
        ("0-0", 1),
        ("0-1,0-1", 2),
        ("0-1,1-2,2-3", 4),
        ("2-1,1-0", 3),
    ] {
        assert!(
            count(&store, edges, nodes).is_ok(),
            "{edges:?} with {nodes} nodes: {:?}",
            count(&store, edges, nodes)
        );
    }
    // The triangle is found.
    assert!(count(&store, "0-1,1-2,2-0", 3).unwrap() > 0);
}
