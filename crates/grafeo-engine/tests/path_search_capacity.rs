//! How many paths a path-following search holds under its budget (0.6.1): an
//! open path is charged its queue entry and the one segment it adds to the
//! path it extends, as segments are shared with the prefix, so a search holds
//! as many paths as 0.6.0 did (3,355,443 at the default 256 MiB, 80 bytes a
//! path). Only an unbounded walk is charged a segment per edge on top, which
//! is what stops a walk around a cycle at the budget (#678).
//!
//! The budget here comes from a memory limit (a quarter of it), so the
//! searches run in milliseconds; graph-bench's shape is a hub of 571 friends.
//!
//! ```bash
//! cargo test -p grafeo-engine --all-features --test main path_search_capacity::
//! ```

#![cfg(all(feature = "gql", feature = "cypher"))]

use std::collections::BTreeSet;
use std::time::Duration;

use grafeo_common::types::{NodeId, Value};
use grafeo_common::utils::error::{Error, Result};
use grafeo_engine::database::QueryResult;
use grafeo_engine::{Config, GrafeoDB};

/// The memory limit of these databases: their path searches get a quarter.
const MEMORY_LIMIT: usize = 32 * 1024 * 1024;
const BUDGET: usize = MEMORY_LIMIT / 4;
/// What 0.6.0 held under its default budget, scaled to `BUDGET`: 3,355,443
/// paths at 256 MiB.
const HELD_BY_0_6_0: usize = 3_355_443 * (BUDGET / (1024 * 1024)) / 256;

/// A small xorshift generator: the graph is the same on every run.
struct Random(u64);

impl Random {
    fn below(&mut self, bound: usize) -> usize {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        usize::try_from(self.0 % bound as u64).unwrap()
    }
}

/// 1,000 persons in clusters of 50, each knowing up to `per_person` others in
/// their cluster, and person 0 knowing 571: the database and its edges.
fn hub_graph(per_person: usize) -> (GrafeoDB, Vec<(usize, usize)>) {
    // These tests are about the memory budget: their paths are many on purpose, and a slow build
    // must not make them a test of the query timeout, while a bad plan still fails by name.
    let db = GrafeoDB::with_config(
        Config::in_memory()
            .with_memory_limit(MEMORY_LIMIT)
            .with_query_timeout(Duration::from_secs(600)),
    )
    .unwrap();
    let mut random = Random(0xD1B5_4A32_D192_ED03);
    let ids: Vec<NodeId> = (0..1_000_i64)
        .map(|id| {
            db.create_node_with_props(&["Person"], [("id", Value::Int64(id))])
                .unwrap()
        })
        .collect();
    let mut edges = Vec::new();
    for person in 1..1_000 {
        let base = person / 50 * 50;
        for _ in 0..per_person {
            let friend = base + random.below(50);
            if friend != person && friend != 0 {
                edges.push((person, friend));
            }
        }
    }
    let mut friends = BTreeSet::new();
    while friends.len() < 571 {
        friends.insert(1 + random.below(999));
    }
    edges.extend(friends.into_iter().map(|friend| (0, friend)));
    for &(from, to) in &edges {
        db.create_edge(ids[from], ids[to], "KNOWS").unwrap();
    }
    (db, edges)
}

/// The paths of 1 to 3 edges from `source`, undirected: trails (no edge
/// twice) when `acyclic` is false, else paths that repeat no node.
fn count_paths(edges: &[(usize, usize)], source: usize, acyclic: bool) -> usize {
    let mut incident = vec![Vec::new(); 1_000];
    for (index, &(a, b)) in edges.iter().enumerate() {
        incident[a].push((index, b));
        incident[b].push((index, a));
    }
    fn walk(
        incident: &[Vec<(usize, usize)>],
        nodes: &mut Vec<usize>,
        used: &mut Vec<usize>,
        acyclic: bool,
    ) -> usize {
        if used.len() == 3 {
            return 0;
        }
        let here = *nodes.last().unwrap();
        let mut found = 0;
        for &(edge, next) in &incident[here] {
            if used.contains(&edge) || (acyclic && nodes.contains(&next)) {
                continue;
            }
            used.push(edge);
            nodes.push(next);
            found += 1 + walk(incident, nodes, used, acyclic);
            nodes.pop();
            used.pop();
        }
        found
    }
    walk(&incident, &mut vec![source], &mut Vec::new(), acyclic)
}

fn run(db: &GrafeoDB, language: &str, query: &str) -> Result<QueryResult> {
    db.execute_language(query, language, None)
}

/// The path count a GRAFEO-Q006 budget error names: "would hold more than
/// N paths".
fn paths_held(error: &Error) -> usize {
    let message = error.to_string();
    assert!(message.contains("GRAFEO-Q006"), "{message}");
    let after = message
        .split("would hold more than ")
        .nth(1)
        .unwrap_or_else(|| panic!("no path count in {message}"));
    after
        .split_whitespace()
        .next()
        .unwrap()
        .parse()
        .unwrap_or_else(|_| panic!("no path count in {message}"))
}

const HUB_QUERIES: [(&str, &str, bool); 2] = [
    (
        "cypher",
        "MATCH p = (s:Person {id: 0})-[:KNOWS*1..3]-(f:Person) RETURN p",
        false,
    ),
    (
        "gql",
        "MATCH p = ACYCLIC (s:Person {id: 0})-[:KNOWS]-{1,3}(f:Person) RETURN p",
        true,
    ),
];

/// About 87,000 paths from the hub: more than a search charged a segment
/// per edge held under the budget (about 48,000), fewer than 0.6.0 held.
#[test]
fn a_hub_search_returns_its_paths_within_the_budget() {
    let (db, edges) = hub_graph(6);
    for (language, query, acyclic) in HUB_QUERIES {
        let want = count_paths(&edges, 0, acyclic);
        assert!(
            (60_000..HELD_BY_0_6_0).contains(&want),
            "the graph should give between 60,000 and {HELD_BY_0_6_0} paths: {want}"
        );
        let result = run(&db, language, query)
            .unwrap_or_else(|error| panic!("{language} `{query}`: {error}"));
        assert_eq!(result.row_count(), want, "{language} `{query}`");
    }
}

/// Over the budget, a search fails having held at least as many paths as
/// 0.6.0 held under the same budget.
#[test]
fn a_search_over_its_budget_holds_as_many_paths_as_0_6_0() {
    let (db, _) = hub_graph(8);
    for (language, query, _) in HUB_QUERIES {
        let error = run(&db, language, query).expect_err(query);
        let held = paths_held(&error);
        assert!(
            held >= HELD_BY_0_6_0,
            "{language} `{query}`: held {held} paths, 0.6.0 held {HELD_BY_0_6_0}: {error}"
        );
        assert!(
            error.to_string().contains("8 MiB"),
            "the budget is a quarter of the memory limit: {error}"
        );
    }
}

/// A directed 3-cycle a -> b -> c -> a, with queries stopped after
/// `timeout`.
fn cycle_with_timeout(timeout: Duration) -> GrafeoDB {
    let config = Config::in_memory()
        .with_memory_limit(MEMORY_LIMIT)
        .with_query_timeout(timeout);
    let db = GrafeoDB::with_config(config).unwrap();
    db.execute(
        "INSERT (a:Spot {name: 'a'})-[:NEXT]->(b:Spot {name: 'b'})-[:NEXT]->(c:Spot {name: 'c'}),          (c)-[:NEXT]->(a)",
    )
    .unwrap();
    db
}

fn cycle() -> GrafeoDB {
    cycle_with_timeout(Duration::from_secs(30))
}

/// An unbounded walk around a cycle never ends by itself: each path is
/// charged per edge, so the one open walk reaches the budget and the query
/// fails, it does not run on.
#[test]
fn an_unbounded_walk_around_a_cycle_fails_at_the_budget() {
    let db = cycle();
    let query = "MATCH (a:Spot {name: 'a'})-[*]->(b) RETURN count(b) AS c";
    let error = run(&db, "gql", query).expect_err(query);
    // A 3-cycle has one walk of each length: the search holds that walk
    // alone when its next edge goes over the budget.
    assert_eq!(paths_held(&error), 1, "{query}: {error}");
    assert!(error.to_string().contains("8 MiB"), "{query}: {error}");
}

/// The same walk keeping its path (a path or an edge list variable) builds
/// each row's path of thousands of edges, so it reaches the query timeout
/// before the budget; either way it ends with an error.
#[test]
fn an_unbounded_walk_keeping_its_path_ends_with_an_error() {
    let db = cycle_with_timeout(Duration::from_secs(2));
    for query in [
        "MATCH p = (a:Spot {name: 'a'})-[*]->(b) RETURN count(p) AS c",
        "MATCH (a:Spot {name: 'a'})-[e*]->(b) RETURN count(b) AS c",
    ] {
        let message = run(&db, "gql", query).expect_err(query).to_string();
        assert!(
            message.contains("GRAFEO-Q003") || message.contains("GRAFEO-Q006"),
            "{query}: {message}"
        );
    }
}

/// A bounded walk on a cycle: one walk per length one way round; both ways
/// round the walks double with each edge, and the search fails at the budget
/// instead of running on. Cypher follows trails, which end after 3 edges.
#[test]
fn a_bounded_walk_on_a_cycle_finishes_or_fails_cleanly() {
    let db = cycle();
    let rows = |language: &str, query: &str| {
        run(&db, language, query)
            .unwrap_or_else(|error| panic!("{language} `{query}`: {error}"))
            .rows()
            .to_vec()
    };
    assert_eq!(
        rows(
            "gql",
            "MATCH (a:Spot {name: 'a'})-[:NEXT]->{1,50}(b) RETURN count(b) AS c"
        ),
        [vec![Value::Int64(50)]]
    );
    assert_eq!(
        rows(
            "gql",
            "MATCH p = (a:Spot {name: 'a'})-[:NEXT]->{1,50}(b) RETURN max(length(p)) AS longest"
        ),
        [vec![Value::Int64(50)]]
    );
    assert_eq!(
        rows(
            "cypher",
            "MATCH (a:Spot {name: 'a'})-[:NEXT*1..50]->(b) RETURN count(b) AS c"
        ),
        [vec![Value::Int64(3)]]
    );
    // Both ways round: 2^50 walks of 50 edges.
    let query = "MATCH (a:Spot {name: 'a'})-[:NEXT]-{1,50}(b) RETURN count(b) AS c";
    let error = run(&db, "gql", query).expect_err(query);
    assert!(paths_held(&error) >= HELD_BY_0_6_0 / 2, "{query}: {error}");
}
