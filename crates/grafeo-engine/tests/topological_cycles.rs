//! `CALL grafeo.topological_sort()` on a graph with a cycle raises 22000 (data exception) and
//! names one cycle (the ruling of 2026-11-06): after Kahn's algorithm, the walk starts at the
//! smallest leftover id and follows the smallest leftover predecessor until a node repeats, so it
//! finds a cycle even from a node downstream of one, and prints the loop in edge direction,
//! "the graph has a cycle: 3 -> 5 -> 8 -> 3", shortened past 16 nodes. 0.6.1 answered one row,
//! `'Graph contains a cycle'`, in an undocumented `error` column, so `YIELD node_id` failed with
//! "no such column". A DAG is unaffected.
//!
//! The tests read the cycle out of the message and check it: a closed walk along the graph's
//! edges, over the nodes of the cycle the walk must reach; where it starts is not pinned.
//!
//! ```bash
//! cargo test -p grafeo-engine --all-features --test main topological_cycles::
//! ```

#![cfg(feature = "algos")]

use std::collections::{BTreeSet, HashMap};

use grafeo_common::utils::gqlstatus::GqlStatus;
use grafeo_engine::GrafeoDB;

/// A graph of `nodes` nodes, created in index order, and its ids by index.
fn graph(nodes: usize, edges: &[(usize, usize)]) -> (GrafeoDB, Vec<u64>) {
    let db = GrafeoDB::new_in_memory();
    let ids: Vec<_> = (0..nodes)
        .map(|_| db.create_node(&["V"]).unwrap())
        .collect();
    for &(u, v) in edges {
        db.create_edge(ids[u], ids[v], "E").unwrap();
    }
    (db, ids.iter().map(|id| id.as_u64()).collect())
}

/// The error of `query`, as its status and its message.
fn refusal(db: &GrafeoDB, query: &str) -> (String, String) {
    let error = db.execute(query).expect_err(query);
    (
        GqlStatus::from(&error).as_str().to_string(),
        error.to_string(),
    )
}

/// The node indices of the loop the message prints, in its order, the repeated last one dropped;
/// the message must carry "the graph has a cycle: a -> b -> ... -> a".
fn printed_loop(message: &str, ids: &[u64]) -> Vec<usize> {
    let (_, tail) = message
        .split_once("the graph has a cycle: ")
        .unwrap_or_else(|| panic!("no cycle named in `{message}`"));
    let index: HashMap<u64, usize> = ids.iter().enumerate().map(|(i, id)| (*id, i)).collect();
    let printed: Vec<usize> = tail
        .split(" -> ")
        .map(|part| {
            let digits: String = part.chars().take_while(char::is_ascii_digit).collect();
            index[&digits
                .parse::<u64>()
                .unwrap_or_else(|_| panic!("`{part}` in `{message}`"))]
        })
        .collect();
    assert!(
        printed.len() >= 2,
        "a loop of one step at least: `{message}`"
    );
    assert_eq!(
        printed.first(),
        printed.last(),
        "the loop closes: `{message}`"
    );
    printed[..printed.len() - 1].to_vec()
}

/// `printed` is a closed walk along `edges`.
fn is_closed_walk(printed: &[usize], edges: &[(usize, usize)]) -> bool {
    (0..printed.len()).all(|i| edges.contains(&(printed[i], printed[(i + 1) % printed.len()])))
}

#[test]
fn a_cycle_raises_22000_naming_it() {
    let edges = [(0, 1), (1, 2), (2, 0), (2, 3)];
    let (db, ids) = graph(4, &edges);
    let (status, message) = refusal(&db, "CALL grafeo.topological_sort()");
    assert_eq!(status, "22000", "{message}");
    let printed = printed_loop(&message, &ids);
    assert!(is_closed_walk(&printed, &edges), "{message}");
    assert_eq!(
        printed.iter().copied().collect::<BTreeSet<_>>(),
        BTreeSet::from([0, 1, 2])
    );
}

/// A `YIELD` of a documented column raises the same error, not "no such column".
#[test]
fn a_yield_raises_the_same_error() {
    let edges = [(0, 1), (1, 0)];
    let (db, _) = graph(2, &edges);
    let plain = refusal(&db, "CALL grafeo.topological_sort()");
    for query in [
        "CALL grafeo.topological_sort() YIELD node_id",
        "CALL grafeo.topological_sort() YIELD node_id, order RETURN node_id, order",
    ] {
        assert_eq!(refusal(&db, query), plain, "{query}");
    }
}

#[test]
fn a_self_loop_names_its_node_twice() {
    let (db, ids) = graph(3, &[(0, 1), (2, 2)]);
    let (status, message) = refusal(&db, "CALL grafeo.topological_sort()");
    assert_eq!(status, "22000");
    let id = ids[2];
    assert!(message.contains(&format!("{id} -> {id}")), "{message}");
}

/// Two separate cycles: the walk from the smallest leftover id names the cycle it reaches, here
/// the one holding node 0.
#[test]
fn of_two_cycles_the_one_the_smallest_leftover_reaches() {
    let edges = [(0, 1), (1, 2), (2, 0), (3, 4), (4, 3)];
    let (db, ids) = graph(5, &edges);
    let (_, message) = refusal(&db, "CALL grafeo.topological_sort()");
    let printed = printed_loop(&message, &ids);
    assert!(is_closed_walk(&printed, &edges), "{message}");
    assert_eq!(
        printed.iter().copied().collect::<BTreeSet<_>>(),
        BTreeSet::from([0, 1, 2])
    );
    // With node 0 on the 2-cycle, the walk names the 2-cycle.
    let edges = [(0, 1), (1, 0), (2, 3), (3, 4), (4, 2)];
    let (db, ids) = graph(5, &edges);
    let (_, message) = refusal(&db, "CALL grafeo.topological_sort()");
    assert_eq!(
        printed_loop(&message, &ids)
            .into_iter()
            .collect::<BTreeSet<_>>(),
        BTreeSet::from([0, 1])
    );
}

/// The smallest leftover id sits downstream of the cycle and has no out-edge: a forward walk would
/// dead-end there, the backward walk reaches the cycle.
#[test]
fn a_leftover_downstream_of_the_cycle_still_finds_it() {
    let edges = [(1, 2), (2, 3), (3, 1), (3, 0)];
    let (db, ids) = graph(4, &edges);
    let (status, message) = refusal(&db, "CALL grafeo.topological_sort()");
    assert_eq!(status, "22000");
    let printed = printed_loop(&message, &ids);
    assert!(is_closed_walk(&printed, &edges), "{message}");
    assert_eq!(
        printed.iter().copied().collect::<BTreeSet<_>>(),
        BTreeSet::from([1, 2, 3])
    );
}

/// A cycle of 40 nodes is shortened, "... (40 nodes)", past its first 16.
#[test]
fn a_long_cycle_gives_a_short_message() {
    let edges: Vec<(usize, usize)> = (0..40).map(|i| (i, (i + 1) % 40)).collect();
    let (db, _) = graph(40, &edges);
    let (status, message) = refusal(&db, "CALL grafeo.topological_sort()");
    assert_eq!(status, "22000");
    assert!(message.contains("... (40 nodes)"), "{message}");
    assert!(
        message.matches(" -> ").count() <= 17,
        "at most 16 nodes printed: {message}"
    );
}

/// A DAG answers its order, the smallest by node id, as before.
#[test]
fn a_dag_is_unaffected() {
    let (db, ids) = graph(4, &[(2, 0), (0, 1), (3, 1)]);
    let result = db
        .execute(
            "CALL grafeo.topological_sort() YIELD node_id, order RETURN node_id ORDER BY order",
        )
        .unwrap();
    let order: Vec<i64> = result
        .rows()
        .iter()
        .map(|row| match &row[0] {
            grafeo_common::types::Value::Int64(id) => *id,
            other => panic!("{other:?}"),
        })
        .collect();
    let want: Vec<i64> = [2, 0, 3, 1]
        .iter()
        .map(|&i| i64::try_from(ids[i]).unwrap())
        .collect();
    assert_eq!(order, want);
}
