//! How many paths a path-following search holds under its budget (0.6.1): an
//! open path is charged its queue entry and the one segment it adds to the
//! path it extends, as segments are shared with the prefix, so a search holds
//! as many paths as 0.6.0 did (3,355,443 at the default 256 MiB, 80 bytes a
//! path), less what it counts that 0.6.0 did not (0.6.4: the buffered output
//! ids and the queue's blocks and their index). Only an unbounded walk is
//! charged a segment per edge on top, which is what stops a walk around a
//! cycle at the budget (#678).
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
use grafeo_common::utils::error::{Error, LimitBound, LimitRefusal, Result};
use grafeo_engine::database::QueryResult;
use grafeo_engine::{Config, GrafeoDB};

/// The memory limit of these databases: their path searches get a quarter.
const MEMORY_LIMIT: usize = 32 * 1024 * 1024;
const BUDGET: usize = MEMORY_LIMIT / 4;
/// What 0.6.0 held under its default budget, scaled to `BUDGET`: 3,355,443
/// paths at 256 MiB.
const HELD_BY_0_6_0: usize = 3_355_443 * (BUDGET / (1024 * 1024)) / 256;

/// The path ids one output chunk of the hub queries holds at most: the rows
/// of one chunk (2,048) times the ids of a path of 3 edges (4 nodes and 3
/// edges) times the id size.
const BUFFERED_IDS_BYTES: usize = 2_048 * (4 + 3) * std::mem::size_of::<NodeId>();

/// The most the overhead named in [`slack`] may take, in thousandths of
/// what 0.6.0 held.
const SLACK_CAP_PER_MILLE: usize = 25;

/// The paths' worth of bytes a search holds besides its paths, which 0.6.0
/// did not count: the buffered output ids, the queue's block index at its
/// doubled peak for the blocks 0.6.0's count needs, and three blocks (the
/// head, the tail and the spare).
fn slack() -> usize {
    let block = grafeo_core::execution::PATH_QUEUE_BLOCK_BYTES;
    let entries_a_block = block / grafeo_core::execution::PATH_QUEUE_ENTRY_BYTES;
    let slots = HELD_BY_0_6_0.div_ceil(entries_a_block).next_power_of_two();
    let index = (slots / 2 + slots + grafeo_core::execution::PATH_RESERVE_SLACK_ENTRIES)
        * grafeo_core::execution::PATH_QUEUE_INDEX_SLOT_BYTES;
    let bytes = BUFFERED_IDS_BYTES + index + 3 * block;
    // 0.6.0 held `HELD_BY_0_6_0` paths in `BUDGET` bytes.
    (bytes * HELD_BY_0_6_0).div_ceil(BUDGET)
}

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
    count_paths_up_to(edges, source, acyclic, 3)
}

/// The paths of 1 to `most` edges from `source`, as [`count_paths`].
fn count_paths_up_to(edges: &[(usize, usize)], source: usize, acyclic: bool, most: usize) -> usize {
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
        most: usize,
    ) -> usize {
        if used.len() == most {
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
            found += 1 + walk(incident, nodes, used, acyclic, most);
            nodes.pop();
            used.pop();
        }
        found
    }
    walk(&incident, &mut vec![source], &mut Vec::new(), acyclic, most)
}

/// The paths' worth of the segments a search of paths of 1 to 3 edges from
/// the hub holds besides those of its queued paths, which 0.6.0 did not
/// count: a popped path stays alive while an extension of it is queued, so
/// at most the source's path and every path of 1 or 2 edges,
/// `(1 + paths of 1 or 2 edges) * PATH_SEGMENT_BYTES`, at 0.6.0's bytes a
/// path. The graph's fan-out and depth fix it; it is no constant of the
/// code, so the cap on [`slack`] does not bound it.
fn live_segments(edges: &[(usize, usize)], acyclic: bool) -> usize {
    let popped = 1 + count_paths_up_to(edges, 0, acyclic, 2);
    (popped * grafeo_core::execution::PATH_SEGMENT_BYTES * HELD_BY_0_6_0).div_ceil(BUDGET)
}

fn run(db: &GrafeoDB, language: &str, query: &str) -> Result<QueryResult> {
    db.execute_language(query, language, None)
}

/// The path count a path search budget's refusal names: "would hold more
/// than N paths". The refusal is `GRAFEO-S001` naming the budget (#492).
fn paths_held(error: &Error) -> usize {
    let message = error.to_string();
    assert!(message.contains("GRAFEO-S001"), "{message}");
    assert_eq!(
        error.limit_refusal().map(LimitRefusal::bound),
        Some(LimitBound::PathSearchBudget),
        "{message}"
    );
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

/// Over the budget, a search fails having held as many paths as 0.6.0 held
/// under the same budget, less the bytes it counts that 0.6.0 did not: the
/// constant terms of [`slack`], which stay under 2.5% of them, and the
/// segments of the popped paths it extends ([`live_segments`]).
///
/// That last term is an upper bound, not the figure at the refusal: how many
/// paths of two edges were taken out before it follows the order of the
/// edges. Here it is 5,696 paths' worth, against about 2,700 really held, so
/// the floor lets about 3% of overhead that no term names through. The
/// counting itself is pinned to the byte by
/// `at_a_refusal_the_popped_paths_with_queued_extensions_are_counted` in the
/// variable-length expand's tests.
///
/// Per path, both queries (Cypher's trail-like variable length and GQL
/// ACYCLIC) hold a queue entry (`PATH_QUEUE_ENTRY_BYTES`, 32 B) and the
/// segment it adds (`PATH_SEGMENT_BYTES`, 48 B): the 80 B 0.6.0 counted. A
/// popped path whose extensions are still queued keeps its own segment alive,
/// 48 B more, which is why a path-following TRAIL search of two hops holds a
/// path in 128 B.
#[test]
fn a_search_over_its_budget_holds_as_many_paths_as_0_6_0() {
    let slack = slack();
    assert!(
        slack * 1000 < HELD_BY_0_6_0 * SLACK_CAP_PER_MILLE,
        "the overhead a search counts besides its paths is {slack} paths' worth, past \
         {SLACK_CAP_PER_MILLE} thousandths of the {HELD_BY_0_6_0} 0.6.0 held"
    );
    let (db, edges) = hub_graph(8);
    for (language, query, acyclic) in HUB_QUERIES {
        let error = run(&db, language, query).expect_err(query);
        let held = paths_held(&error);
        let live = live_segments(&edges, acyclic);
        assert!(
            held >= HELD_BY_0_6_0 - slack - live,
            "{language} `{query}`: held {held} paths, 0.6.0 held {HELD_BY_0_6_0}, less \
             {slack} paths' worth of overhead and {live} of live segments: {error}"
        );
        assert!(
            error.to_string().contains("8 MiB"),
            "the budget is a quarter of the memory limit: {error}"
        );
    }
}

/// What 0.6.3 charged each path a restrictive shortest-path search queued:
/// its entry (an `Arc` of its last segment and its length, 16 bytes), that
/// segment (32) and the two counts of its `Arc` (16).
const QUEUED_BY_0_6_3: usize = 64;

/// The leaves of the star below: one more than a power of two, so a queue
/// that doubles regrows at the last of them.
const STAR_LEAVES: usize = (1 << 14) + 1;

/// A hub with an edge to each of [`STAR_LEAVES`] leaves, the first of which
/// leads on to the target, its path searches within `budget`.
fn star(budget: usize) -> GrafeoDB {
    let db = GrafeoDB::with_config(Config::in_memory().with_path_search_budget(budget)).unwrap();
    let hub = db.create_node(&["Hub"]).unwrap();
    let target = db.create_node(&["Target"]).unwrap();
    for leaf in 0..STAR_LEAVES {
        let node = db.create_node(&["Leaf"]).unwrap();
        db.create_edge(hub, node, "E").unwrap();
        if leaf == 0 {
            db.create_edge(node, target, "E").unwrap();
        }
    }
    db
}

/// A restrictive shortest-path search queues as many paths as 0.6.3 did at
/// the same budget. From the hub it queues every leaf at once: 0.6.3 needed
/// 64 bytes for each (the target's path is queued after the first leaf is
/// taken out, so 16,385 at most), 1,048,640 in all, and answered within
/// them. Besides its queued paths the search holds what 0.6.3 did not count
/// ([`restrictive_slack`]), so it answers at 0.6.3's figure plus exactly
/// that, and is refused a byte under it, and a byte under 0.6.3's figure: it
/// counts every queued path.
#[test]
fn a_restrictive_shortest_path_queues_as_many_paths_as_0_6_3() {
    let needed_by_0_6_3 = STAR_LEAVES * QUEUED_BY_0_6_3;
    let slack = restrictive_slack();
    assert!(
        slack * 1000 < needed_by_0_6_3 * SLACK_CAP_PER_MILLE,
        "the overhead a search counts besides its queued paths is {slack} bytes, past \
         {SLACK_CAP_PER_MILLE} thousandths of the {needed_by_0_6_3} 0.6.3 needed"
    );
    let query = "MATCH p = ANY SHORTEST TRAIL (s:Hub)-[:E]->+(t:Target) RETURN length(p) AS l";
    let result = run(&star(needed_by_0_6_3 + slack), "gql", query)
        .unwrap_or_else(|error| panic!("within 0.6.3's {needed_by_0_6_3} bytes: {error}"));
    assert_eq!(result.rows(), [vec![Value::Int64(2)]]);
    for budget in [needed_by_0_6_3 + slack - 1, needed_by_0_6_3 - 1] {
        let error = run(&star(budget), "gql", query).expect_err(query);
        assert_eq!(
            error.limit_refusal().map(LimitRefusal::bound),
            Some(LimitBound::PathSearchBudget),
            "{budget}: {error}"
        );
    }
}

/// The bytes a restrictive shortest-path search holds at its peak besides
/// what 0.6.3 counted, once the first leaf is taken out and the target's
/// path queued: the segments of the source's path and of the target's (0.6.3
/// counted its queued paths only), the room left in the queue's blocks for
/// the leaves and the target (the first leaf's slot is not reused), and the
/// block index, at the power of two that holds every block but the tail.
fn restrictive_slack() -> usize {
    let segment = grafeo_core::execution::PATH_SEGMENT_BYTES;
    let entry = QUEUED_BY_0_6_3 - segment;
    let block = grafeo_core::execution::PATH_QUEUE_BLOCK_BYTES;
    let blocks = (STAR_LEAVES + 1).div_ceil(block / entry);
    let room = blocks * block - STAR_LEAVES * entry;
    let index =
        (blocks - 1).next_power_of_two() * grafeo_core::execution::PATH_QUEUE_INDEX_SLOT_BYTES;
    2 * segment + room + index
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

/// An unbounded walk around a cycle never ends by itself, so GQL refuses it
/// before it runs (`42001`, since 0.6.4): a quantifier without an upper bound
/// needs a path mode or a selector.
#[test]
fn an_unbounded_walk_around_a_cycle_is_refused() {
    let db = cycle();
    let query = "MATCH (a:Spot {name: 'a'})-[*]->(b) RETURN count(b) AS c";
    let error = run(&db, "gql", query).expect_err(query);
    assert_eq!(
        grafeo_common::utils::gqlstatus::GqlStatus::from(&error).as_str(),
        "42001",
        "{query}: {error}"
    );
}

/// The same walk keeping its path (a path or an edge list variable) is
/// refused at once too: before 0.6.4 it built each row's path of thousands
/// of edges and ran into the query timeout.
#[test]
fn an_unbounded_walk_keeping_its_path_is_refused_at_once() {
    let db = cycle_with_timeout(Duration::from_secs(2));
    for query in [
        "MATCH p = (a:Spot {name: 'a'})-[*]->(b) RETURN count(p) AS c",
        "MATCH (a:Spot {name: 'a'})-[e*]->(b) RETURN count(b) AS c",
    ] {
        let start = std::time::Instant::now();
        let error = run(&db, "gql", query).expect_err(query);
        assert_eq!(
            grafeo_common::utils::gqlstatus::GqlStatus::from(&error).as_str(),
            "42001",
            "{query}: {error}"
        );
        assert!(
            start.elapsed() < Duration::from_secs(1),
            "{query}: refused before it runs, not at the timeout"
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
