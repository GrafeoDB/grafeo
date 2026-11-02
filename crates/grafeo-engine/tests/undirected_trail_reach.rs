//! Distinct targets of an undirected TRAIL pattern with a minimum of at most
//! one hop (0.6.1): Cypher `(s)-[:K*1..3]-(f)` and GQL `MATCH TRAIL` with
//! `count(DISTINCT f)` run as one reachability search instead of listing every
//! trail, which ran out of its path budget from a node of high degree. Every
//! node but the source is reached over a shortest path; the source is a
//! target only over a closed trail (a self-loop, two parallel edges or a cycle
//! through it) of at most the maximum, never over the edge it left by.
//!
//! The oracle is a brute-force trail enumeration over the edge list, written
//! here, not the engine's own enumeration.
//!
//! ```bash
//! cargo test -p grafeo-engine --all-features --test main undirected_trail_reach::
//! ```

#![cfg(all(feature = "gql", feature = "cypher", feature = "sql-pgq"))]

use std::collections::{BTreeSet, HashMap};

use grafeo_common::types::{NodeId, Value};
use grafeo_engine::GrafeoDB;

/// An undirected multigraph: nodes `0..nodes`, `K` edges as written (self-loops
/// and parallel edges allowed), and `L` edges the patterns must not follow.
struct Graph {
    nodes: usize,
    k_edges: Vec<(usize, usize)>,
    l_edges: Vec<(usize, usize)>,
}

impl Graph {
    fn new(nodes: usize, k_edges: &[(usize, usize)]) -> Self {
        Self {
            nodes,
            k_edges: k_edges.to_vec(),
            l_edges: Vec::new(),
        }
    }

    /// The nodes as `:N {id: i}` and the edges, each stored in the direction
    /// written: the patterns are undirected, so it must not matter.
    fn load(&self) -> GrafeoDB {
        let db = GrafeoDB::new_in_memory();
        let ids: Vec<NodeId> = (0..self.nodes)
            .map(|index| {
                db.create_node_with_props(&["N"], [("id", int(index))])
                    .unwrap()
            })
            .collect();
        for &(from, to) in &self.k_edges {
            db.create_edge(ids[from], ids[to], "K").unwrap();
        }
        for &(from, to) in &self.l_edges {
            db.create_edge(ids[from], ids[to], "L").unwrap();
        }
        db
    }

    /// The other end of `K` edge `edge` from `node`, if it touches `node`; a
    /// self-loop is one step back to its node.
    fn step(&self, edge: usize, node: usize) -> Option<usize> {
        let (a, b) = self.k_edges[edge];
        if a == node {
            Some(b)
        } else if b == node {
            Some(a)
        } else {
            None
        }
    }

    /// Every trail from `node` that repeats none of `used` or its own edges:
    /// the end of each of `low..=high` edges (counted in `ends`, so the bag
    /// shows how many trails end where).
    fn trails_from(
        &self,
        node: usize,
        used: u64,
        length: u32,
        (low, high): (u32, u32),
        ends: &mut HashMap<usize, usize>,
    ) {
        if length >= low {
            *ends.entry(node).or_default() += 1;
        }
        if length == high {
            return;
        }
        for edge in 0..self.k_edges.len() {
            if used & (1 << edge) != 0 {
                continue;
            }
            if let Some(next) = self.step(edge, node) {
                self.trails_from(next, used | (1 << edge), length + 1, (low, high), ends);
            }
        }
    }

    /// The trails of `low..=high` edges from `source`: their ends, with the
    /// number of trails to each.
    fn trails(&self, source: usize, low: u32, high: u32) -> HashMap<usize, usize> {
        let mut ends = HashMap::new();
        self.trails_from(source, 0, 0, (low, high), &mut ends);
        ends
    }

    /// The ends of `(s)-[r:K]-(m)-[:K*low..high]-(f)`: one edge from the
    /// source, then a trail from its other end that does not repeat it.
    fn trails_after_one_edge(&self, source: usize, low: u32, high: u32) -> BTreeSet<usize> {
        let mut ends = HashMap::new();
        for edge in 0..self.k_edges.len() {
            if let Some(middle) = self.step(edge, source) {
                self.trails_from(middle, 1 << edge, 0, (low, high), &mut ends);
            }
        }
        ends.into_keys().collect()
    }
}

/// An id or a count as the INT64 a query returns.
fn int(n: usize) -> Value {
    Value::Int64(i64::try_from(n).unwrap())
}

fn params(source: usize) -> Option<HashMap<String, Value>> {
    Some(HashMap::from([("s".to_string(), int(source))]))
}

fn run(db: &GrafeoDB, language: &str, query: &str, source: usize) -> Vec<Vec<Value>> {
    db.execute_language(query, language, params(source))
        .unwrap_or_else(|error| panic!("{language} `{query}` from {source}: {error}"))
        .rows()
        .to_vec()
}

fn id(value: &Value) -> usize {
    match value {
        Value::Int64(id) => usize::try_from(*id).unwrap(),
        other => panic!("expected an id, got {other:?}"),
    }
}

/// The ids of a one-row `collect(DISTINCT f.id)` result, checked free of
/// repeats.
fn collected(rows: &[Vec<Value>]) -> BTreeSet<usize> {
    let Value::List(ids) = &rows[0][0] else {
        panic!("expected a list, got {:?}", rows[0][0]);
    };
    let set: BTreeSet<usize> = ids.iter().map(id).collect();
    assert_eq!(set.len(), ids.len(), "collect(DISTINCT) repeated an id");
    set
}

fn cypher_range(low: u32, high: u32) -> String {
    format!("*{low}..{high}")
}

/// The distinct targets of the reachability plan, Cypher.
fn cypher_targets(db: &GrafeoDB, source: usize, low: u32, high: u32) -> BTreeSet<usize> {
    let query = format!(
        "MATCH (s:N {{id: $s}})-[:K{}]-(f:N) \
         RETURN collect(DISTINCT f.id) AS ids, count(DISTINCT f) AS c",
        cypher_range(low, high)
    );
    let rows = run(db, "cypher", &query, source);
    let targets = collected(&rows);
    assert_eq!(rows[0][1], int(targets.len()), "{query}");
    targets
}

/// The same with GQL `MATCH TRAIL`.
fn gql_targets(db: &GrafeoDB, source: usize, low: u32, high: u32) -> BTreeSet<usize> {
    let query = format!(
        "MATCH TRAIL (s:N {{id: $s}})-[:K]-{{{low},{high}}}(f:N) \
         RETURN collect(DISTINCT f.id) AS ids"
    );
    collected(&run(db, "gql", &query, source))
}

fn set(ids: &[usize]) -> BTreeSet<usize> {
    ids.iter().copied().collect()
}

/// Checks the reachability plan of Cypher and GQL against the oracle, and
/// the source filter of graph-bench's form.
fn assert_targets(graph: &Graph, db: &GrafeoDB, source: usize, low: u32, high: u32) {
    let want: BTreeSet<usize> = graph.trails(source, low, high).into_keys().collect();
    let case = format!("from {source}, {low}..{high}, K edges {:?}", graph.k_edges);
    assert_eq!(cypher_targets(db, source, low, high), want, "Cypher {case}");
    assert_eq!(gql_targets(db, source, low, high), want, "GQL TRAIL {case}");
    let filtered = format!(
        "MATCH (s:N {{id: $s}})-[:K{}]-(f:N) WHERE f.id <> $s RETURN count(DISTINCT f.id) AS c",
        cypher_range(low, high)
    );
    let others = want.iter().filter(|&&target| target != source).count();
    assert_eq!(
        run(db, "cypher", &filtered, source),
        [vec![int(others)]],
        "Cypher with f.id <> $s, {case}"
    );
}

/// The small shapes, each with what it shows.
#[test]
fn the_source_is_a_target_only_over_a_closed_trail() {
    // A path a-b-c from a: out and back over one edge is no trail.
    let path = Graph::new(3, &[(0, 1), (2, 1)]);
    let db = path.load();
    assert_eq!(cypher_targets(&db, 0, 1, 3), set(&[1, 2]));
    assert_targets(&path, &db, 0, 1, 3);
    // A triangle x-y-z from x: the cycle is 3 edges long.
    let triangle = Graph::new(3, &[(0, 1), (2, 1), (0, 2)]);
    let db = triangle.load();
    assert_eq!(cypher_targets(&db, 0, 1, 2), set(&[1, 2]));
    assert_eq!(cypher_targets(&db, 0, 1, 3), set(&[0, 1, 2]));
    // Two parallel edges p=q (one each way): a closed trail of 2.
    let parallel = Graph::new(2, &[(0, 1), (1, 0)]);
    let db = parallel.load();
    assert_eq!(cypher_targets(&db, 0, 1, 1), set(&[1]));
    assert_eq!(cypher_targets(&db, 0, 1, 2), set(&[0, 1]));
    // A self-loop: a closed trail of 1.
    let looped = Graph::new(1, &[(0, 0)]);
    let db = looped.load();
    assert_eq!(cypher_targets(&db, 0, 1, 1), set(&[0]));
    assert_eq!(cypher_targets(&db, 0, 1, 3), set(&[0]));
    for (graph, high) in [(&path, 3), (&triangle, 3), (&parallel, 2), (&looped, 3)] {
        let db = graph.load();
        for source in 0..graph.nodes {
            for top in 1..=high {
                assert_targets(graph, &db, source, 1, top);
            }
        }
    }
}

#[test]
fn a_minimum_of_zero_hops_always_has_the_source() {
    let path = Graph::new(3, &[(0, 1), (2, 1)]);
    let db = path.load();
    assert_eq!(cypher_targets(&db, 0, 0, 1), set(&[0, 1]));
    assert_eq!(cypher_targets(&db, 0, 0, 2), set(&[0, 1, 2]));
    assert_eq!(gql_targets(&db, 2, 0, 1), set(&[1, 2]));
    let isolated = Graph::new(2, &[]);
    let db = isolated.load();
    assert_eq!(cypher_targets(&db, 0, 0, 3), set(&[0]));
    assert_eq!(cypher_targets(&db, 0, 1, 3), set(&[]));
}

#[test]
fn a_self_loop_beside_other_edges() {
    // s has a self-loop and an edge to a; a-b. The loop closes a trail of 1;
    // a walk s-a-s is no trail, so without the loop s would be out.
    let graph = Graph::new(3, &[(0, 0), (0, 1), (1, 2)]);
    let db = graph.load();
    assert_eq!(cypher_targets(&db, 0, 1, 1), set(&[0, 1]));
    assert_eq!(cypher_targets(&db, 0, 1, 2), set(&[0, 1, 2]));
    // From a, s's loop is no closed trail through a.
    assert_eq!(cypher_targets(&db, 1, 1, 3), set(&[0, 2]));
    // A self-loop on b does not bring s back either.
    let graph = Graph::new(3, &[(0, 1), (1, 2), (2, 2)]);
    let db = graph.load();
    assert_eq!(cypher_targets(&db, 0, 1, 4), set(&[1, 2]));
    assert_eq!(cypher_targets(&db, 2, 1, 1), set(&[1, 2]));
    for source in 0..3 {
        for high in 1..=4 {
            assert_targets(&graph, &db, source, 1, high);
        }
    }
}

#[test]
fn a_cycle_past_the_first_hop_does_not_bring_the_source_back() {
    // A lollipop s-a, a-b-c-a: from s every trail back would repeat s-a.
    let lollipop = Graph::new(4, &[(0, 1), (1, 2), (3, 2), (3, 1)]);
    let db = lollipop.load();
    assert_eq!(cypher_targets(&db, 0, 1, 4), set(&[1, 2, 3]));
    assert_eq!(gql_targets(&db, 0, 1, 4), set(&[1, 2, 3]));
    // From a, on the cycle: back over b and c in 3.
    assert_eq!(cypher_targets(&db, 1, 1, 2), set(&[0, 2, 3]));
    assert_eq!(cypher_targets(&db, 1, 1, 3), set(&[0, 1, 2, 3]));
    // A square s-a-c-b-s: the closed trail through s is 4 edges.
    let square = Graph::new(4, &[(0, 1), (1, 3), (2, 3), (0, 2)]);
    let db = square.load();
    assert_eq!(cypher_targets(&db, 0, 1, 3), set(&[1, 2, 3]));
    assert_eq!(cypher_targets(&db, 0, 1, 4), set(&[0, 1, 2, 3]));
    // A five-cycle: the source needs a maximum of 5.
    let five = Graph::new(5, &[(0, 1), (1, 2), (2, 3), (3, 4), (4, 0)]);
    let db = five.load();
    assert_eq!(cypher_targets(&db, 0, 1, 4), set(&[1, 2, 3, 4]));
    assert_eq!(cypher_targets(&db, 0, 1, 5), set(&[0, 1, 2, 3, 4]));
    for (graph, nodes) in [(&lollipop, 4), (&square, 4), (&five, 5)] {
        let db = graph.load();
        for source in 0..nodes {
            for high in 1..=5 {
                assert_targets(graph, &db, source, 1, high);
            }
        }
    }
}

#[test]
fn the_distinct_count_runs_as_one_reachability_search() {
    let db = Graph::new(3, &[(0, 1), (1, 2), (2, 0)]).load();
    for (language, query) in [
        (
            "cypher",
            "EXPLAIN MATCH (s:N {id: 0})-[:K*1..3]-(f:N) RETURN count(DISTINCT f) AS c",
        ),
        (
            "cypher",
            "EXPLAIN MATCH (s:N {id: 0})-[:K*0..3]-(f:N) RETURN count(DISTINCT f) AS c",
        ),
        (
            "gql",
            "EXPLAIN MATCH TRAIL (s:N {id: 0})-[:K]-{1,3}(f:N) RETURN count(DISTINCT f) AS c",
        ),
    ] {
        let plan = format!(
            "{:?}",
            db.execute_language(query, language, None).unwrap().rows()
        );
        assert!(
            plan.contains("[reachability"),
            "{language} `{query}`: {plan}"
        );
    }
}

/// The targets in the order the reachability plan gives them: the list of a
/// one-row `collect(DISTINCT f.id)`.
fn reach_order(db: &GrafeoDB, source: usize, low: u32, high: u32) -> Vec<usize> {
    let query = format!(
        "MATCH (s:N {{id: $s}})-[:K{}]-(f:N) RETURN collect(DISTINCT f.id) AS ids",
        cypher_range(low, high)
    );
    let rows = run(db, "cypher", &query, source);
    let Value::List(ids) = &rows[0][0] else {
        panic!("expected a list, got {:?}", rows[0][0]);
    };
    ids.iter().map(id).collect()
}

/// The targets in the order the enumeration first gives them: every trail
/// as a row, each target kept where it first appears.
fn enumeration_order(db: &GrafeoDB, source: usize, low: u32, high: u32) -> Vec<usize> {
    let query = format!(
        "MATCH (s:N {{id: $s}})-[:K{}]-(f:N) RETURN f.id AS id",
        cypher_range(low, high)
    );
    let mut seen = BTreeSet::new();
    run(db, "cypher", &query, source)
        .iter()
        .map(|row| id(&row[0]))
        .filter(|target| seen.insert(*target))
        .collect()
}

impl Graph {
    /// The edges of the shortest closed trail through `source`, if one has
    /// at most `high`.
    fn shortest_closed_trail(&self, source: usize, high: u32) -> Option<u32> {
        (1..=high).find(|&length| self.trails(source, length, length).contains_key(&source))
    }
}

/// The order of the targets: as the enumeration's when the source comes
/// back over a self-loop or two parallel edges (emitted the moment a trail
/// of the layer closes), or not at all; a source closed only by a longer
/// trail comes at the end of that layer, so there the targets are the same
/// set, possibly in another order.
fn assert_order(graph: &Graph, db: &GrafeoDB, source: usize, low: u32, high: u32) {
    let reached = reach_order(db, source, low, high);
    let enumerated = enumeration_order(db, source, low, high);
    let case = format!("from {source}, {low}..{high}, K edges {:?}", graph.k_edges);
    match graph.shortest_closed_trail(source, high) {
        Some(length) if low >= 1 && length >= 3 => {
            let as_set = |ids: &[usize]| ids.iter().copied().collect::<BTreeSet<_>>();
            assert_eq!(as_set(&reached), as_set(&enumerated), "{case}");
            assert_eq!(reached.len(), enumerated.len(), "{case}");
        }
        _ => assert_eq!(reached, enumerated, "the order, {case}"),
    }
}

/// A source back over a self-loop or parallel edges comes where the
/// enumeration first gives it; one back over a triangle may come later.
#[test]
fn the_source_comes_in_the_order_of_the_enumeration_over_short_closed_trails() {
    // s with a self-loop, s-a, a-b, s-c: the loop's trail closes in layer 1.
    let looped = Graph::new(4, &[(0, 1), (0, 0), (1, 2), (3, 0)]);
    let db = looped.load();
    assert_eq!(reach_order(&db, 0, 1, 2), enumeration_order(&db, 0, 1, 2));
    assert!(reach_order(&db, 0, 1, 2).contains(&0));
    // p=q twice, q-r, p-t: the parallel pair closes in layer 2.
    let parallel = Graph::new(4, &[(0, 1), (1, 2), (1, 0), (3, 0)]);
    let db = parallel.load();
    assert_eq!(reach_order(&db, 0, 1, 3), enumeration_order(&db, 0, 1, 3));
    assert!(reach_order(&db, 0, 1, 3).contains(&0));
    // A triangle with a tail: the source closes in 3, same targets.
    let triangle = Graph::new(4, &[(0, 1), (1, 2), (2, 0), (2, 3)]);
    let db = triangle.load();
    for (graph, db) in [
        (&looped, looped.load()),
        (&parallel, parallel.load()),
        (&triangle, db),
    ] {
        for source in 0..graph.nodes {
            for low in 0..=1 {
                for high in 1..=4 {
                    assert_order(graph, &db, source, low, high);
                }
            }
        }
    }
}

/// A small xorshift generator: the cases are the same on every run.
struct Random(u64);

impl Random {
    fn below(&mut self, bound: usize) -> usize {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        usize::try_from(self.0 % bound as u64).unwrap()
    }
}

/// Random multigraphs of 1 to 7 nodes and up to 12 `K` edges, with
/// self-loops, parallel edges (also written the other way round) and `L`
/// edges beside them.
fn random_graph(random: &mut Random) -> Graph {
    let nodes = 1 + random.below(7);
    let mut graph = Graph::new(nodes, &[]);
    for _ in 0..random.below(13) {
        let edge = if !graph.k_edges.is_empty() && random.below(5) == 0 {
            let (a, b) = graph.k_edges[random.below(graph.k_edges.len())];
            if random.below(2) == 0 { (a, b) } else { (b, a) }
        } else {
            (random.below(nodes), random.below(nodes))
        };
        graph.k_edges.push(edge);
    }
    for _ in 0..random.below(3) {
        graph
            .l_edges
            .push((random.below(nodes), random.below(nodes)));
    }
    graph
}

/// The reachability plan against the brute-force trails: every source of
/// each graph, minimums 0 to 2 (2 runs the enumeration), maximums to 4.
/// Cypher's own enumeration (every trail as a row) must give the same bag.
#[test]
fn random_multigraphs_match_the_brute_force_trails() {
    let mut random = Random(0x9E37_79B9_7F4A_7C15);
    let mut checked = 0;
    for _ in 0..120 {
        let graph = random_graph(&mut random);
        let db = graph.load();
        for source in 0..graph.nodes {
            for low in 0..=2 {
                for high in low.max(1)..=4 {
                    assert_targets(&graph, &db, source, low, high);
                    assert_order(&graph, &db, source, low, high);
                    // Every trail as a row: the bag, so the enumeration is
                    // right too.
                    let query = format!(
                        "MATCH (s:N {{id: $s}})-[:K{}]-(f:N) RETURN f.id AS id",
                        cypher_range(low, high)
                    );
                    let mut got = HashMap::new();
                    for row in run(&db, "cypher", &query, source) {
                        *got.entry(id(&row[0])).or_insert(0usize) += 1;
                    }
                    assert_eq!(
                        got,
                        graph.trails(source, low, high),
                        "trails as rows from {source}, {low}..{high}, K edges {:?}",
                        graph.k_edges
                    );
                    checked += 1;
                }
            }
        }
    }
    assert!(checked > 1_000, "only {checked} cases");
}

/// Relationship uniqueness spans the whole pattern: a variable-length part
/// after a fixed edge may not reuse that edge.
#[test]
fn random_multigraphs_after_one_fixed_edge() {
    let mut random = Random(0x2545_F491_4F6C_DD1D);
    for _ in 0..80 {
        let graph = random_graph(&mut random);
        let db = graph.load();
        for source in 0..graph.nodes {
            for low in 0..=1 {
                for high in 1..=3 {
                    let query = format!(
                        "MATCH (s:N {{id: $s}})-[r:K]-(m:N)-[:K{}]-(f:N) \
                         RETURN collect(DISTINCT f.id) AS ids",
                        cypher_range(low, high)
                    );
                    assert_eq!(
                        collected(&run(&db, "cypher", &query, source)),
                        graph.trails_after_one_edge(source, low, high),
                        "from {source}, {low}..{high}, K edges {:?}",
                        graph.k_edges
                    );
                }
            }
        }
    }
}

/// graph-bench's shape: a hub of 571 `KNOWS` edges in a clustered graph of
/// 1,000 persons who know about 70 others each. Listing every trail (millions
/// of 3 edges) ran out of the path budget (GRAFEO-Q006);
/// the search answers, and agrees with SQL/PGQ, GQL and a breadth-first
/// oracle.
#[test]
fn a_hub_of_high_degree_answers_and_matches_sql_pgq() {
    const PERSONS: usize = 1_000;
    const CLUSTER: usize = 50;
    let mut random = Random(0xD1B5_4A32_D192_ED03);
    let mut edges: Vec<(usize, usize)> = Vec::new();
    for person in 1..PERSONS {
        let base = person / CLUSTER * CLUSTER;
        for _ in 0..35 {
            let friend = base + random.below(CLUSTER);
            if friend != person && friend != 0 {
                edges.push((person, friend));
            }
        }
    }
    let mut hub_friends = BTreeSet::new();
    while hub_friends.len() < 571 {
        hub_friends.insert(1 + random.below(PERSONS - 1));
    }
    edges.extend(hub_friends.iter().map(|&friend| (0, friend)));

    let db = GrafeoDB::new_in_memory();
    let ids: Vec<NodeId> = (0..PERSONS)
        .map(|index| {
            db.create_node_with_props(&["Person"], [("id", int(index))])
                .unwrap()
        })
        .collect();
    for &(from, to) in &edges {
        db.create_edge(ids[from], ids[to], "KNOWS").unwrap();
    }

    // The oracle: persons within 3 edges of the hub, by breadth-first search.
    let mut adjacent = vec![Vec::new(); PERSONS];
    for &(a, b) in &edges {
        adjacent[a].push(b);
        adjacent[b].push(a);
    }
    let mut depth = vec![u32::MAX; PERSONS];
    depth[0] = 0;
    let mut layer = vec![0];
    for hop in 1..=3 {
        let mut next = Vec::new();
        for &node in &layer {
            for &friend in &adjacent[node] {
                if depth[friend] == u32::MAX {
                    depth[friend] = hop;
                    next.push(friend);
                }
            }
        }
        layer = next;
    }
    let within = depth.iter().filter(|&&d| (1..=3).contains(&d)).count();
    // Two friends of the hub who know each other close a triangle through it.
    let triangle = edges
        .iter()
        .any(|(a, b)| hub_friends.contains(a) && hub_friends.contains(b));
    assert!(triangle, "the data should close a triangle through the hub");

    let count = |language: &str, query: &str| -> Value {
        db.execute_language(query, language, params(0))
            .unwrap_or_else(|error| panic!("{language} `{query}`: {error}"))
            .rows()[0][0]
            .clone()
    };
    for (language, query) in [
        (
            "cypher",
            "MATCH (s:Person {id: $s})-[:KNOWS*1..3]-(f:Person) WHERE f.id <> $s \
             RETURN count(DISTINCT f.id) AS persons",
        ),
        (
            "cypher",
            "MATCH (s:Person {id: $s})-[:KNOWS*1..3]-(f:Person) WHERE f.id <> $s \
             RETURN count(DISTINCT f) AS persons",
        ),
        (
            "gql",
            "MATCH TRAIL (s:Person {id: $s})-[:KNOWS]-{1,3}(f:Person) WHERE f.id <> $s \
             RETURN count(DISTINCT f.id) AS persons",
        ),
        (
            "sql-pgq",
            "SELECT COUNT(DISTINCT fid) AS persons FROM GRAPH_TABLE (MATCH \
             (s:Person WHERE s.id = $s)-[k:KNOWS]-{1,3}(f:Person WHERE f.id <> $s) \
             COLUMNS (f.id AS fid))",
        ),
    ] {
        assert_eq!(count(language, query), int(within), "{language} `{query}`");
    }
    // Without the filter the hub counts too: the triangle is a closed trail.
    assert_eq!(
        count(
            "cypher",
            "MATCH (s:Person {id: $s})-[:KNOWS*1..3]-(f:Person) RETURN count(DISTINCT f) AS persons"
        ),
        int(within + 1)
    );
}
