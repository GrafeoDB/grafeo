//! Counts over chains of hops, answered from degrees (#714 clusters 4 and 5).
//!
//! Tests first, from the decided behavior: a count over a chain answers what enumerating the
//! chain answers, for every length, in every state of the graph, in both matching rules:
//! Cypher binds an edge once per `MATCH`, GQL may repeat it. The enumerator here walks a list of
//! edges and shares nothing with the engine.
//!
//! `GRAFEO_COUNT_SEEDS=500` sweeps more seeds; `GRAFEO_COUNT_SEED=17` runs one.

#![cfg(all(feature = "gql", feature = "cypher"))]
#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss,
    reason = "test graphs of a few thousand nodes"
)]

use std::collections::BTreeMap;
use std::fmt::Write as _;

use grafeo_common::types::{NodeId, Value};
use grafeo_engine::query::optimizer::{PlannerRule, PlannerRules};
use grafeo_engine::session::Session;
use grafeo_engine::{Config, GrafeoDB, HistoryRetention};

const FIXED_SEEDS: u64 = 30;
const REPORTED: usize = 8;

/// The most walks a rule set that enumerates is asked for: past this it tests only patience, and
/// a slow build reaches the query timeout (a graph of 15 self-loops has 14^6 at length 6). The
/// count itself is still checked on them, with every rule on.
const ENUMERATED_WALKS: i64 = 200_000;

struct Random(u64);

impl Random {
    fn new(seed: u64) -> Self {
        Self(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ 0x5851_F42D_4C95_7F2D)
    }

    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0.wrapping_mul(0x2545_F491_4F6C_DD1D) >> 11
    }

    fn below(&mut self, bound: usize) -> usize {
        usize::try_from(self.next() % bound.max(1) as u64).unwrap()
    }

    fn chance(&mut self, percent: usize) -> bool {
        self.below(100) < percent
    }

    fn pick<T: Copy>(&mut self, from: &[T]) -> T {
        from[self.below(from.len())]
    }
}

#[derive(Clone, Debug)]
struct Vertex {
    key: i64,
    labels: Vec<&'static str>,
}

#[derive(Clone, Debug)]
struct Link {
    key: i64,
    from: i64,
    to: i64,
    kind: &'static str,
}

#[derive(Clone, Debug, Default)]
struct Multigraph {
    vertices: Vec<Vertex>,
    links: Vec<Link>,
}

const LABELS: [&str; 2] = ["V", "W"];
const KINDS: [&str; 2] = ["E", "F"];

impl Multigraph {
    /// Self-loops, parallel edges, two-cycles, hubs, nodes alone, two labels and two types.
    fn random(random: &mut Random) -> Self {
        let count = 2 + random.below(8);
        let vertices = (0..count)
            .map(|key| {
                let mut labels = vec!["V"];
                if random.chance(25) {
                    labels = vec!["W"];
                } else if random.chance(15) {
                    labels.push("W");
                }
                Vertex {
                    key: key as i64,
                    labels,
                }
            })
            .collect();
        let mut graph = Self {
            vertices,
            links: Vec::new(),
        };
        let wanted = 3 + random.below(16);
        let hub = random.chance(40).then(|| random.below(count) as i64);
        let connected = 1 + random.below(count);
        while graph.links.len() < wanted {
            let pick = |random: &mut Random| {
                hub.filter(|_| random.chance(40))
                    .unwrap_or(random.below(connected) as i64)
            };
            let (from, to) = (pick(random), pick(random));
            let kind = if random.chance(80) {
                "E"
            } else {
                random.pick(&KINDS)
            };
            for _ in 0..=usize::from(random.chance(20)) {
                let key = graph.links.len() as i64;
                graph.links.push(Link {
                    key,
                    from,
                    to,
                    kind,
                });
            }
            if random.chance(10) {
                let key = graph.links.len() as i64;
                graph.links.push(Link {
                    key,
                    from,
                    to: from,
                    kind,
                });
            }
        }
        graph
    }
}

/// The fixture whose answer tells trails from walks: parallel edges, antiparallel pairs,
/// self-loops, parallel self-loops and a node alone.
fn fixture() -> Multigraph {
    let links = [
        (1, 2),
        (1, 2),
        (2, 1),
        (2, 2),
        (2, 3),
        (3, 3),
        (3, 3),
        (3, 4),
        (4, 5),
        (5, 4),
        (5, 5),
    ];
    Multigraph {
        vertices: (1..=6)
            .map(|key| Vertex {
                key,
                labels: vec!["V"],
            })
            .collect(),
        links: links
            .iter()
            .enumerate()
            .map(|(index, (from, to))| Link {
                key: index as i64,
                from: *from,
                to: *to,
                kind: "E",
            })
            .collect(),
    }
}

#[derive(Clone, Copy, PartialEq, Debug)]
enum Way {
    Out,
    In,
    Either,
}

#[derive(Clone, Debug)]
struct Hop {
    way: Way,
    kind: Option<&'static str>,
    /// The label of the node the hop ends at.
    end: Option<&'static str>,
}

#[derive(Clone, Debug)]
struct Chain {
    first: Option<&'static str>,
    hops: Vec<Hop>,
    /// Counts per first node instead of in all.
    grouped: bool,
    /// A condition on the first node's key.
    below: Option<i64>,
}

impl Chain {
    fn text(&self) -> String {
        let label = |label: Option<&str>| label.map_or(String::new(), |l| format!(":{l}"));
        let mut out = format!("MATCH (a{})", label(self.first));
        for hop in &self.hops {
            let kind = label(hop.kind);
            let end = label(hop.end);
            match hop.way {
                Way::Out => write!(out, "-[{kind}]->({end})").unwrap(),
                Way::In => write!(out, "<-[{kind}]-({end})").unwrap(),
                Way::Either => write!(out, "-[{kind}]-({end})").unwrap(),
            }
        }
        if let Some(below) = self.below {
            write!(out, " WHERE a.k < {below}").unwrap();
        }
        if self.grouped {
            out.push_str(" RETURN a.k AS k, count(*) AS c");
        } else {
            out.push_str(" RETURN count(*) AS c");
        }
        out
    }

    /// Walks the chain over the list of edges: (first node, how many), for each first node.
    fn enumerate(&self, graph: &Multigraph, edges_may_repeat: bool) -> BTreeMap<i64, i64> {
        fn walk(
            graph: &Multigraph,
            hops: &[Hop],
            at: i64,
            used: &mut Vec<i64>,
            edges_may_repeat: bool,
        ) -> i64 {
            let Some((hop, rest)) = hops.split_first() else {
                return 1;
            };
            let mut total = 0;
            for link in &graph.links {
                if hop.kind.is_some_and(|kind| kind != link.kind) {
                    continue;
                }
                let other = match hop.way {
                    Way::Out => (link.from == at).then_some(link.to),
                    Way::In => (link.to == at).then_some(link.from),
                    // A self-loop once.
                    Way::Either => {
                        if link.from == at {
                            Some(link.to)
                        } else {
                            (link.to == at).then_some(link.from)
                        }
                    }
                };
                let Some(other) = other else {
                    continue;
                };
                if !edges_may_repeat && used.contains(&link.key) {
                    continue;
                }
                let vertex = graph.vertices.iter().find(|v| v.key == other).unwrap();
                if hop.end.is_some_and(|label| !vertex.labels.contains(&label)) {
                    continue;
                }
                used.push(link.key);
                total += walk(graph, rest, other, used, edges_may_repeat);
                used.pop();
            }
            total
        }
        let mut counts = BTreeMap::new();
        for vertex in &graph.vertices {
            if self
                .first
                .is_some_and(|label| !vertex.labels.contains(&label))
            {
                continue;
            }
            if self.below.is_some_and(|below| vertex.key >= below) {
                continue;
            }
            let found = walk(
                graph,
                &self.hops,
                vertex.key,
                &mut Vec::new(),
                edges_may_repeat,
            );
            counts.insert(vertex.key, found);
        }
        counts
    }

    /// How many steps enumerating the chain can take, whatever its labels let through: a last
    /// label no node has leaves no row, and every prefix is still walked.
    fn walks_at_most(&self, graph: &Multigraph) -> i64 {
        let mut at: BTreeMap<i64, i64> = graph.vertices.iter().map(|v| (v.key, 1)).collect();
        let mut total = 0_i64;
        for hop in &self.hops {
            let mut next: BTreeMap<i64, i64> = BTreeMap::new();
            for link in &graph.links {
                if hop.kind.is_some_and(|kind| kind != link.kind) {
                    continue;
                }
                let mut step = |from: i64, to: i64| {
                    let walks = at.get(&from).copied().unwrap_or(0);
                    let entry = next.entry(to).or_insert(0);
                    *entry = entry.saturating_add(walks);
                };
                match hop.way {
                    Way::Out => step(link.from, link.to),
                    Way::In => step(link.to, link.from),
                    Way::Either => {
                        step(link.from, link.to);
                        if link.from != link.to {
                            step(link.to, link.from);
                        }
                    }
                }
            }
            total = next
                .values()
                .fold(total, |sum, walks| sum.saturating_add(*walks));
            at = next;
        }
        total
    }

    fn expected(&self, graph: &Multigraph, edges_may_repeat: bool) -> Vec<Vec<i64>> {
        let counts = self.enumerate(graph, edges_may_repeat);
        if self.grouped {
            counts
                .into_iter()
                .filter(|(_, count)| *count > 0)
                .map(|(key, count)| vec![key, count])
                .collect()
        } else {
            vec![vec![counts.values().sum()]]
        }
    }
}

fn random_chain(random: &mut Random, length: usize) -> Chain {
    let label = |random: &mut Random, percent| random.chance(percent).then(|| random.pick(&LABELS));
    let same_kind = random.chance(70);
    let same_way = random.chance(60);
    let way = |random: &mut Random| match random.below(10) {
        0..=6 => Way::Out,
        7 => Way::In,
        _ => Way::Either,
    };
    let first_way = way(random);
    let hops = (0..length)
        .map(|_| Hop {
            way: if same_way { first_way } else { way(random) },
            kind: if same_kind {
                Some("E")
            } else {
                random.chance(80).then(|| random.pick(&KINDS))
            },
            end: label(random, 35),
        })
        .collect();
    Chain {
        first: label(random, 60),
        hops,
        grouped: random.chance(35),
        below: random.chance(15).then(|| random.below(6) as i64),
    }
}

fn load(db: &GrafeoDB, graph: &Multigraph) {
    let mut ids: BTreeMap<i64, NodeId> = BTreeMap::new();
    for vertex in &graph.vertices {
        let id = db
            .create_node_with_props(&vertex.labels, [("k", Value::Int64(vertex.key))])
            .unwrap();
        ids.insert(vertex.key, id);
    }
    for link in &graph.links {
        db.create_edge_with_props(
            ids[&link.from],
            ids[&link.to],
            link.kind,
            [("k", Value::Int64(link.key))],
        )
        .unwrap();
    }
}

/// Writes for a stage: edges made, edges removed, a node removed. Applied to the model too.
fn writes(graph: &Multigraph, random: &mut Random) -> (Vec<String>, Multigraph) {
    let mut after = graph.clone();
    let mut statements = Vec::new();
    let mut next = 1_000;
    for _ in 0..=random.below(6) {
        match random.below(10) {
            0..=4 => {
                let from = after.vertices[random.below(after.vertices.len())].key;
                let to = after.vertices[random.below(after.vertices.len())].key;
                let kind = random.pick(&KINDS);
                statements.push(format!(
                    "MATCH (a {{k: {from}}}), (b {{k: {to}}}) INSERT (a)-[:{kind} {{k: {next}}}]->(b)"
                ));
                after.links.push(Link {
                    key: next,
                    from,
                    to,
                    kind,
                });
                next += 1;
            }
            5..=8 if !after.links.is_empty() => {
                let link = after.links.remove(random.below(after.links.len()));
                statements.push(format!("MATCH ()-[e {{k: {}}}]->() DELETE e", link.key));
            }
            _ if after.vertices.len() > 2 => {
                let vertex = after.vertices.remove(random.below(after.vertices.len()));
                statements.push(format!("MATCH (n {{k: {}}}) DETACH DELETE n", vertex.key));
                after
                    .links
                    .retain(|l| l.from != vertex.key && l.to != vertex.key);
            }
            _ => {}
        }
    }
    (statements, after)
}

fn answer(session: &Session, cypher: bool, query: &str) -> Result<Vec<Vec<i64>>, String> {
    let result = if cypher {
        session.execute_cypher(query)
    } else {
        session.execute(query)
    }
    .map_err(|error| error.to_string())?;
    let mut rows: Vec<Vec<i64>> = result
        .rows()
        .iter()
        .map(|row| {
            row.iter()
                .map(|value| match value {
                    Value::Int64(number) => *number,
                    other => panic!("{query}: {other:?}"),
                })
                .collect()
        })
        .collect();
    rows.sort();
    Ok(rows)
}

fn rule_sets() -> [(&'static str, PlannerRules); 3] {
    [
        ("all rules", PlannerRules::all()),
        (
            "all but FactorizedExecution",
            PlannerRules::all().with(PlannerRule::FactorizedExecution, false),
        ),
        ("no rule", PlannerRules::plain()),
    ]
}

fn check(
    session: &mut Session,
    seen: &Multigraph,
    chains: &[Chain],
    stage: &str,
    seed: u64,
    differences: &mut Vec<String>,
) {
    for chain in chains {
        let query = chain.text();
        for cypher in [false, true] {
            let expected = chain.expected(seen, !cypher);
            let walks = chain.walks_at_most(seen);
            for (name, rules) in rule_sets() {
                if walks > ENUMERATED_WALKS && name != "all rules" {
                    continue;
                }
                session.set_planner_rules(rules);
                let got = answer(session, cypher, &query);
                if got.as_ref() != Ok(&expected) && differences.len() < REPORTED {
                    differences.push(format!(
                        "seed {seed}, {stage}, {}, {name}\n  {query}\n  expected {expected:?}\n  got      {got:?}\n  vertices {:?}\n  links    {:?}\n",
                        if cypher { "Cypher" } else { "GQL" },
                        seen.vertices.iter().map(|v| (v.key, v.labels.clone())).collect::<Vec<_>>(),
                        seen.links.iter().map(|l| (l.from, l.to, l.kind)).collect::<Vec<_>>(),
                    ));
                }
            }
        }
    }
}

fn seeds() -> Vec<u64> {
    if let Some(seed) = std::env::var("GRAFEO_COUNT_SEED")
        .ok()
        .and_then(|s| s.parse().ok())
    {
        return vec![seed];
    }
    let count = std::env::var("GRAFEO_COUNT_SEEDS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(FIXED_SEEDS);
    (0..count).collect()
}

fn history() -> Config {
    Config::in_memory().with_history_retention(HistoryRetention::Forever)
}

/// Every length, against plain enumeration: self-loops, parallel edges, undirected hops, other
/// types, labels on any node, grouped or not, and every state of the graph.
#[test]
fn a_count_answers_what_enumerating_answers_for_every_length() {
    let mut differences = Vec::new();
    for seed in seeds() {
        let mut random = Random::new(seed);
        let graph = Multigraph::random(&mut random);
        // The longest length is the slow one: on every third seed, 4 chains ran 150 s.
        let chains: Vec<Chain> = (1..=6)
            .flat_map(|length| {
                let each = if length < 6 {
                    4
                } else {
                    usize::from(seed % 3 == 0)
                };
                (0..each)
                    .map(|_| random_chain(&mut random, length))
                    .collect::<Vec<_>>()
            })
            .collect();
        let (statements, after) = writes(&graph, &mut random);
        let apply = |session: &Session| {
            for statement in &statements {
                session.execute(statement).unwrap();
            }
        };
        let fresh = || {
            let db = GrafeoDB::with_config(history()).unwrap();
            load(&db, &graph);
            db
        };

        let db = fresh();
        check(
            &mut db.session(),
            &graph,
            &chains,
            "committed",
            seed,
            &mut differences,
        );

        let db = fresh();
        let mut session = db.session();
        session.begin_transaction().unwrap();
        apply(&session);
        check(
            &mut session,
            &after,
            &chains,
            "own writes",
            seed,
            &mut differences,
        );
        session.rollback().unwrap();
        check(
            &mut session,
            &graph,
            &chains,
            "after rollback",
            seed,
            &mut differences,
        );

        let db = fresh();
        let mut other = db.session();
        other.begin_transaction().unwrap();
        apply(&other);
        check(
            &mut db.session(),
            &graph,
            &chains,
            "another open transaction",
            seed,
            &mut differences,
        );
        other.rollback().unwrap();

        let db = fresh();
        let mut session = db.session();
        let epoch = db.current_epoch();
        apply(&session);
        check(
            &mut session,
            &after,
            &chains,
            "after writes and deletes",
            seed,
            &mut differences,
        );
        session.set_viewing_epoch(epoch);
        check(
            &mut session,
            &graph,
            &chains,
            "past epoch",
            seed,
            &mut differences,
        );
        session.clear_viewing_epoch();

        if differences.len() >= REPORTED {
            break;
        }
    }
    assert!(differences.is_empty(), "{}", differences.join("\n"));
}

const TWO_HOPS: &str = "MATCH (a:V)-[:E]->(b:V)-[:E]->(c:V) RETURN count(*) AS paths";

fn fixture_db() -> GrafeoDB {
    let db = GrafeoDB::with_config(history()).unwrap();
    load(&db, &fixture());
    db
}

/// The fixture's pinned answers: trails 22, walks 26.
#[test]
fn the_fixture_counts_trails_and_walks() {
    let db = fixture_db();
    let mut session = db.session();
    for (_, rules) in rule_sets() {
        session.set_planner_rules(rules);
        assert_eq!(answer(&session, true, TWO_HOPS), Ok(vec![vec![22]]));
        assert_eq!(answer(&session, false, TWO_HOPS), Ok(vec![vec![26]]));
    }
    let chain = Chain {
        first: Some("V"),
        hops: vec![
            Hop {
                way: Way::Out,
                kind: Some("E"),
                end: Some("V"),
            };
            2
        ],
        grouped: true,
        below: None,
    };
    // Per source, as the top-10 query needs it.
    let query = "MATCH (a:V)-[:E]->(b:V)-[:E]->(c:V) RETURN a.k AS k, count(*) AS c";
    assert_eq!(
        answer(&session, true, query),
        Ok(chain.expected(&fixture(), false))
    );
    assert_eq!(
        answer(&session, false, query),
        Ok(chain.expected(&fixture(), true))
    );
    assert_eq!(
        chain
            .expected(&fixture(), false)
            .iter()
            .map(|row| row[1])
            .sum::<i64>(),
        22
    );
}

fn profile(session: &Session, cypher: bool, query: &str) -> String {
    let text = format!("PROFILE {query}");
    let result = if cypher {
        session.execute_cypher(&text)
    } else {
        session.execute(&text)
    }
    .unwrap();
    result
        .rows()
        .iter()
        .map(|row| match &row[0] {
            Value::String(line) => line.to_string(),
            other => format!("{other:?}"),
        })
        .collect::<Vec<_>>()
        .join("\n")
}

/// Labels on each hop no longer keep a chain from its count, in either language, grouped by
/// the chain's first column too (A1 and A2 of the analytics brief).
#[test]
fn a_labelled_chain_reaches_the_count() {
    let db = fixture_db();
    let session = db.session();
    let mut off = db.session();
    off.set_planner_rules(PlannerRules::all().with(PlannerRule::FactorizedExecution, false));
    for query in [
        TWO_HOPS,
        "MATCH (a:V)-[:E]->(b:V)-[:E]->(c:V) RETURN a.k AS k, count(*) AS c",
        "MATCH (a:V)-[:E]->(b:V)-[:E]->(c:V) WHERE a.k < 4 RETURN a.k AS k, count(*) AS c ORDER BY c DESC, k ASC LIMIT 10",
        "MATCH (a:V)-[:E]->(b)-[:E]->(c:V) RETURN count(*)",
        "MATCH (a:V)-[:E]->(b:V)-[:E]->() RETURN count(*)",
        "MATCH (a)-[:E]->(b:V)-[:E]->(c) RETURN count(*)",
    ] {
        for cypher in [false, true] {
            let plan = profile(&session, cypher, query);
            assert!(
                plan.contains("EdgeCount"),
                "cypher {cypher}: {query}\n{plan}"
            );
            let plain = profile(&off, cypher, query);
            assert!(
                !plain.contains("EdgeCount"),
                "cypher {cypher}: {query}\n{plain}"
            );
        }
    }
}

/// A hub in the middle of a chain with 2,000 edges each way: 4 million paths, counted without
/// listing them.
#[test]
fn a_count_through_a_hub_reads_its_degrees_not_its_paths() {
    let db = GrafeoDB::new_in_memory();
    let hub = db
        .create_node_with_props(&["V"], [("k", Value::Int64(0))])
        .unwrap();
    for key in 1..=2_000_i64 {
        let near = db
            .create_node_with_props(&["V"], [("k", Value::Int64(key))])
            .unwrap();
        db.create_edge(near, hub, "E").unwrap();
        let far = db
            .create_node_with_props(&["V"], [("k", Value::Int64(-key))])
            .unwrap();
        db.create_edge(hub, far, "E").unwrap();
    }
    let session = db.session();
    for cypher in [false, true] {
        let result = if cypher {
            session.execute_cypher(TWO_HOPS)
        } else {
            session.execute(TWO_HOPS)
        }
        .unwrap();
        assert_eq!(
            result.rows()[0][0],
            Value::Int64(4_000_000),
            "cypher {cypher}"
        );
        let reads = result.rows_scanned().unwrap();
        assert!(
            reads < 40_000,
            "cypher {cypher}: read {reads} for 4,000,000 paths"
        );
    }
}

/// Lengths the sum over positions covers, each on its own hub-heavy graph; the answer checks
/// the terms of the sum (5, 15 and 52 for 3, 4 and 5 hops). A length past where it pays runs
/// the plain plan, which this also accepts.
#[test]
fn cypher_chains_of_one_type_keep_every_edge_once() {
    for seed in 0..12_u64 {
        let mut random = Random::new(seed + 500);
        let graph = Multigraph {
            vertices: (0..4)
                .map(|key| Vertex {
                    key,
                    labels: vec!["V"],
                })
                .collect(),
            links: (0..12)
                .map(|key| Link {
                    key,
                    from: random.below(4) as i64,
                    to: random.below(4) as i64,
                    kind: "E",
                })
                .collect(),
        };
        let db = GrafeoDB::new_in_memory();
        load(&db, &graph);
        let mut session = db.session();
        for length in 2..=6 {
            let chain = Chain {
                first: Some("V"),
                hops: vec![
                    Hop {
                        way: Way::Out,
                        kind: Some("E"),
                        end: None,
                    };
                    length
                ],
                grouped: false,
                below: None,
            };
            let expected = chain.expected(&graph, false);
            for (name, rules) in rule_sets() {
                session.set_planner_rules(rules);
                let got = answer(&session, true, &chain.text());
                assert_eq!(
                    got.as_ref(),
                    Ok(&expected),
                    "seed {seed}, length {length}, {name}"
                );
            }
            // The walks: the count GQL gives, which is never below the trails.
            let walks = chain.expected(&graph, true);
            session.set_planner_rules(PlannerRules::all());
            assert_eq!(answer(&session, false, &chain.text()), Ok(walks.clone()));
            assert!(walks[0][0] >= expected[0][0]);
        }
    }
}

// ---------------------------------------------------------------------------
// Two hops from the middle node: cluster 5
// ---------------------------------------------------------------------------

/// A whole graph large enough to merge its adjacency, with the edges kept as a list: self-loops,
/// parallel edges and two types, one in 10 far-ends of another label.
struct LargeGraph {
    labels: Vec<&'static str>,
    edges: Vec<(usize, usize, &'static str)>,
}

impl LargeGraph {
    fn new(nodes: usize, edges: usize) -> Self {
        let mut state = 99_u64;
        let mut next = move |bound: usize| {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            usize::try_from((state >> 33) % bound as u64).unwrap()
        };
        let labels = (0..nodes)
            .map(|index| if index % 10 == 9 { "W" } else { "V" })
            .collect();
        let mut list = Vec::with_capacity(edges);
        while list.len() < edges {
            let (from, to) = (next(nodes), next(nodes));
            let kind = if next(5) == 0 { "F" } else { "E" };
            list.push((from, to, kind));
            match next(40) {
                0 => list.push((from, from, kind)),
                1 => list.push((from, to, kind)),
                2 => list.push((to, from, kind)),
                _ => {}
            }
        }
        Self {
            labels,
            edges: list,
        }
    }

    /// Two hops by the sum over the middle node: all in-edges of `first` times all out-edges of
    /// `second`, from and to the label asked for, less the pairs that are one edge twice.
    fn two_hops(&self, first: &str, second: &str, label: Option<&str>, trails: bool) -> i64 {
        let fits = |node: usize| label.is_none_or(|label| self.labels[node] == label);
        let mut into = vec![0_i64; self.labels.len()];
        let mut out = vec![0_i64; self.labels.len()];
        let mut once = 0;
        for (from, to, kind) in &self.edges {
            if *kind == first && fits(*from) {
                into[*to] += 1;
            }
            if *kind == second && fits(*to) {
                out[*from] += 1;
            }
            // One edge as both hops: it ends where it starts.
            if first == second && *kind == first && from == to && fits(*from) {
                once += 1;
            }
        }
        let walks: i64 = (0..self.labels.len())
            .filter(|node| fits(*node))
            .map(|node| into[node] * out[node])
            .sum();
        if trails { walks - once } else { walks }
    }

    /// The same, against the edges' direction on both hops or either way.
    fn two_hops_either_way(&self, kind: &str, trails: bool) -> i64 {
        let mut both = vec![0_i64; self.labels.len()];
        for (from, to, edge_kind) in &self.edges {
            if *edge_kind == kind {
                both[*from] += 1;
                if from != to {
                    both[*to] += 1;
                }
            }
        }
        // A pair that is one edge twice: an edge from one end and back, for each end it has.
        both.iter()
            .map(|count| count * count - if trails { *count } else { 0 })
            .sum()
    }
}

fn load_large(db: &GrafeoDB, graph: &LargeGraph) -> Vec<NodeId> {
    let ids: Vec<NodeId> = graph
        .labels
        .iter()
        .map(|label| db.create_node(&[label]).unwrap())
        .collect();
    for (from, to, kind) in &graph.edges {
        db.create_edge(ids[*from], ids[*to], kind).unwrap();
    }
    ids
}

fn one_count(session: &Session, cypher: bool, query: &str) -> i64 {
    let rows = answer(session, cypher, query).unwrap_or_else(|error| panic!("{query}: {error}"));
    rows[0][0]
}

/// The whole-graph counts at each stage, against the sum over the list of edges.
fn all_two_hop_counts_agree(session: &mut Session, graph: &LargeGraph, stage: &str) {
    for (name, rules) in rule_sets() {
        session.set_planner_rules(rules);
        for (query, cypher, expected) in [
            (
                "MATCH (a)-[:E]->(b)-[:E]->(c) RETURN count(*)",
                false,
                graph.two_hops("E", "E", None, false),
            ),
            (
                "MATCH (a)-[:E]->(b)-[:E]->(c) RETURN count(*)",
                true,
                graph.two_hops("E", "E", None, true),
            ),
            (
                "MATCH (a:V)-[:E]->(b:V)-[:E]->(c:V) RETURN count(*)",
                false,
                graph.two_hops("E", "E", Some("V"), false),
            ),
            (
                "MATCH (a:V)-[:E]->(b:V)-[:E]->(c:V) RETURN count(*)",
                true,
                graph.two_hops("E", "E", Some("V"), true),
            ),
            (
                "MATCH (a)-[:E]->(b)-[:F]->(c) RETURN count(*)",
                true,
                graph.two_hops("E", "F", None, true),
            ),
            (
                "MATCH (a)-[:E]-(b)-[:E]-(c) RETURN count(*)",
                false,
                graph.two_hops_either_way("E", false),
            ),
            (
                "MATCH (a)-[:E]-(b)-[:E]-(c) RETURN count(*)",
                true,
                graph.two_hops_either_way("E", true),
            ),
        ] {
            let got = one_count(session, cypher, query);
            assert_eq!(got, expected, "{stage}, {name}, cypher {cypher}: {query}");
        }
    }
}

/// Own uncommitted edges, a rollback, deletes (so the data is no longer flat, with edges removed
/// from sorted lists), and a past epoch: the count from degrees is the count of the list.
#[test]
fn the_middle_sum_counts_what_each_state_holds() {
    let mut graph = LargeGraph::new(1_500, 6_000);
    let db = GrafeoDB::with_config(history()).unwrap();
    let ids = load_large(&db, &graph);
    let mut session = db.session();
    all_two_hop_counts_agree(&mut session, &graph, "committed");

    let committed = LargeGraph {
        labels: graph.labels.clone(),
        edges: graph.edges.clone(),
    };
    let mut watcher = db.session();
    let epoch = db.current_epoch();

    // Own new edges: on a hub, a self-loop, and between two labels. No key on an edge here: the
    // ids of the nodes name them.
    session.begin_transaction().unwrap();
    for (from, to, kind) in [
        (0, 1, "E"),
        (1, 1, "E"),
        (1, 2, "E"),
        (9, 1, "E"),
        (1, 9, "F"),
    ] {
        session
            .execute(&format!(
                "MATCH (a), (b) WHERE id(a) = {} AND id(b) = {} INSERT (a)-[:{kind}]->(b)",
                ids[from].as_u64(),
                ids[to].as_u64()
            ))
            .unwrap();
        graph.edges.push((from, to, kind));
    }
    all_two_hop_counts_agree(&mut session, &graph, "own new edges");
    all_two_hop_counts_agree(&mut watcher, &committed, "another session, own new edges");

    session.rollback().unwrap();
    all_two_hop_counts_agree(&mut session, &committed, "after the rollback");

    // Own deletes: every edge of a node, in and out, then a commit of the same.
    let victim = committed.edges[0].0;
    let statement = format!(
        "MATCH (n)-[e]-() WHERE id(n) = {} DELETE e",
        ids[victim].as_u64()
    );
    let without = LargeGraph {
        labels: committed.labels.clone(),
        edges: committed
            .edges
            .iter()
            .filter(|(from, to, _)| *from != victim && *to != victim)
            .copied()
            .collect(),
    };
    session.begin_transaction().unwrap();
    session.execute(&statement).unwrap();
    all_two_hop_counts_agree(&mut session, &without, "own deletes");
    all_two_hop_counts_agree(&mut watcher, &committed, "another session, own deletes");
    session.commit().unwrap();
    all_two_hop_counts_agree(&mut session, &without, "committed deletes");

    session.set_viewing_epoch(epoch);
    all_two_hop_counts_agree(&mut session, &committed, "past epoch");
    session.clear_viewing_epoch();
}

/// What cluster 5 is for: a whole-graph count reads each node's degrees, not its first hop's list
/// of edges. Edges outnumber the nodes twenty times here, and the bound is on the nodes.
#[test]
fn a_whole_graph_count_reads_degrees_not_lists() {
    let db = GrafeoDB::new_in_memory();
    let graph = LargeGraph::new(1_000, 20_000);
    load_large(&db, &graph);
    let session = db.session();
    for (query, cypher, expected) in [
        (
            "MATCH (a)-[:E]->(b)-[:E]->(c) RETURN count(*)",
            false,
            graph.two_hops("E", "E", None, false),
        ),
        ("MATCH (a:V)-[:E]->(b)-[:E]->(c) RETURN count(*)", false, {
            // Only the first node is labelled: a walk from a V through anything to anything.
            let first_fits = |node: usize| graph.labels[node] == "V";
            let mut into = vec![0_i64; graph.labels.len()];
            let mut out = vec![0_i64; graph.labels.len()];
            for (from, to, kind) in &graph.edges {
                if *kind == "E" {
                    out[*from] += 1;
                    if first_fits(*from) {
                        into[*to] += 1;
                    }
                }
            }
            (0..graph.labels.len()).map(|b| into[b] * out[b]).sum()
        }),
        (
            "MATCH (a)-[:E]->(b)-[:E]->(c) RETURN count(*)",
            true,
            graph.two_hops("E", "E", None, true),
        ),
    ] {
        let result = if cypher {
            session.execute_cypher(query)
        } else {
            session.execute(query)
        }
        .unwrap();
        assert_eq!(result.rows()[0][0], Value::Int64(expected), "{query}");
        // A label on the first node is not in a degree: it reads through the first hop's lists.
        if !query.contains("(a:V)") {
            let reads = result.rows_scanned().unwrap();
            let nodes = graph.labels.len() as u64;
            // A scan row and two degrees per node, and for Cypher a lookup of the self-loops.
            let per_node = if cypher { 4 } else { 3 };
            assert!(
                reads <= nodes * per_node,
                "cypher {cypher}: read {reads}: {query}"
            );
        }
    }
}

/// Too many walks for a count: a clear error, never a number that wrapped. A node with 1,000
/// self-loops has 1,000^n walks of n hops: 10^18 fits an INT64 and 10^21 does not.
#[test]
fn a_count_too_large_for_its_type_is_an_error_not_a_wrong_number() {
    let db = GrafeoDB::new_in_memory();
    let node = db.create_node(&["V"]).unwrap();
    for _ in 0..1_000 {
        db.create_edge(node, node, "E").unwrap();
    }
    let chain = |hops: usize| format!("MATCH (a:V){} RETURN count(*)", "-[:E]->()".repeat(hops));
    let session = db.session();
    assert_eq!(
        answer(&session, false, &chain(6)),
        Ok(vec![vec![1_000_000_000_000_000_000]])
    );
    let error = answer(&session, false, &chain(7)).expect_err("10^21 walks in an INT64");
    assert!(
        error.contains("numeric value out of range"),
        "not a clear error: {error}"
    );
    // By first node, the same.
    let grouped = "MATCH (a:V)-[:E]->()-[:E]->()-[:E]->()-[:E]->()-[:E]->()-[:E]->()-[:E]->() \
                   RETURN id(a), count(*)";
    let error = session
        .execute(grouped)
        .map(|_| ())
        .expect_err("10^21 walks in an INT64");
    assert!(
        error.to_string().contains("numeric value out of range"),
        "not a clear error: {error}"
    );
    // Cypher binds each edge once: a falling factorial, which pins the sum over positions on
    // numbers far from the small graphs of the oracle.
    assert_eq!(
        answer(&session, true, &chain(3)),
        Ok(vec![vec![1_000 * 999 * 998]])
    );
    assert_eq!(
        answer(&session, true, &chain(5)),
        Ok(vec![vec![1_000 * 999 * 998 * 997 * 996]])
    );
}

/// The counts of many rows add up past an INT64 though each fits: 20 first nodes with 10^18
/// walks each. Per first node they are right, and over one group or in all, an error.
#[test]
fn counts_that_add_up_past_their_type_are_an_error() {
    let db = GrafeoDB::new_in_memory();
    for key in 0..20_i64 {
        let node = db
            .create_node_with_props(&["V"], [("k", Value::Int64(key)), ("g", Value::Int64(1))])
            .unwrap();
        for _ in 0..1_000 {
            db.create_edge(node, node, "E").unwrap();
        }
    }
    let session = db.session();
    let chain = "-[:E]->()".repeat(6);
    let per_node = format!("MATCH (a:V){chain} RETURN a.k AS k, count(*) AS c");
    let rows = answer(&session, false, &per_node).unwrap();
    assert_eq!(rows.len(), 20);
    assert!(rows.iter().all(|row| row[1] == 1_000_000_000_000_000_000));
    for query in [
        format!("MATCH (a:V){chain} RETURN count(*) AS c"),
        format!("MATCH (a:V){chain} RETURN a.g AS g, count(*) AS c"),
    ] {
        let error = answer(&session, false, &query).expect_err("2 x 10^19 walks in an INT64");
        assert!(
            error.contains("numeric value out of range"),
            "not a clear error: {query}: {error}"
        );
    }
}

/// The other counted paths: a sum over a chain (the factorized aggregate) past its type is an
/// error too, not a number that stopped at the largest INT64, with the rule on and off alike.
#[test]
fn a_sum_over_a_chain_past_its_type_is_an_error() {
    let db = GrafeoDB::new_in_memory();
    let ids: Vec<NodeId> = (0..3)
        .map(|_| {
            db.create_node_with_props(&["V"], [("big", Value::Int64(5_000_000_000_000_000_000))])
                .unwrap()
        })
        .collect();
    for from in &ids {
        for to in &ids {
            db.create_edge(*from, *to, "E").unwrap();
        }
    }
    let mut session = db.session();
    let queries = [
        "MATCH (a:V)-[:E]->(b)-[:E]->(c) RETURN sum(a.big) AS s",
        "MATCH (a:V)-[:E]->(b)-[:E]->(c) RETURN sum(c.big) AS s",
        "MATCH (a:V)-[:E]->(b:V)-[:E]->(c:V) RETURN a.big AS g, sum(c.big) AS s",
    ];
    for (name, rules) in rule_sets() {
        session.set_planner_rules(rules);
        for query in queries {
            let error = session
                .execute(query)
                .map(|result| format!("{:?}", result.rows()))
                .expect_err(&format!("{name}: 27 x 5 x 10^18 in an INT64: {query}"));
            assert!(
                error.to_string().contains("numeric value out of range"),
                "{name}: not a clear error: {query}: {error}"
            );
        }
    }
}

/// A middle node with two labels: the scan reads one, the other is a check of each node.
#[test]
fn the_middle_sum_checks_every_label_of_the_middle() {
    let db = GrafeoDB::new_in_memory();
    let start = db.create_node(&["S"]).unwrap();
    let end = db.create_node(&["S"]).unwrap();
    let both = db.create_node(&["V", "W"]).unwrap();
    let one = db.create_node(&["V"]).unwrap();
    let other = db.create_node(&["W"]).unwrap();
    for middle in [both, one, other] {
        db.create_edge(start, middle, "E").unwrap();
        db.create_edge(middle, end, "E").unwrap();
    }
    let session = db.session();
    assert_eq!(
        answer(
            &session,
            true,
            "MATCH (a)-[:E]->(b:V:W)-[:E]->(c) RETURN count(*)"
        ),
        Ok(vec![vec![1]])
    );
    assert_eq!(
        answer(
            &session,
            false,
            "MATCH (a)-[:E]->(b:V&W)-[:E]->(c) RETURN count(*)"
        ),
        Ok(vec![vec![1]])
    );
    assert_eq!(
        answer(
            &session,
            false,
            "MATCH (a)-[:E]->(b:V)-[:E]->(c) RETURN count(*)"
        ),
        Ok(vec![vec![2]])
    );
}

/// Two hops from one node are no chain: a star's count is the product per node, and an order the
/// search picks must not read it as a walk of two hops.
#[test]
fn hops_from_one_node_are_not_a_chain() {
    let db = GrafeoDB::new_in_memory();
    let centre = db.create_node(&["V"]).unwrap();
    let side = db.create_node(&["V"]).unwrap();
    let mut far = Vec::new();
    for _ in 0..7 {
        far.push(db.create_node(&["W"]).unwrap());
    }
    for node in far.iter().take(3) {
        db.create_edge(centre, *node, "E").unwrap();
    }
    for node in far.iter().skip(3) {
        db.create_edge(centre, *node, "F").unwrap();
        // What a walk from the far end of E or F would find.
        db.create_edge(*node, side, "F").unwrap();
        db.create_edge(*node, side, "E").unwrap();
    }
    let mut session = db.session();
    for (name, rules) in rule_sets() {
        session.set_planner_rules(rules);
        for cypher in [false, true] {
            for query in [
                "MATCH (a:V)-[:E]->(b), (a)-[:F]->(c) RETURN count(*)",
                "MATCH (a:V)-[:F]->(b), (a)-[:E]->(c) RETURN count(*)",
                "MATCH (a:V)-[:E]->(b) MATCH (a)-[:F]->(c) RETURN count(*)",
                "MATCH (a:V)-[:E]->(b), (a:V)-[:F]->(c) RETURN count(*)",
                "MATCH (b)<-[:E]-(a:V)-[:F]->(c) RETURN count(*)",
            ] {
                assert_eq!(
                    answer(&session, cypher, query),
                    Ok(vec![vec![3 * 4]]),
                    "{name}, cypher {cypher}: {query}"
                );
            }
        }
    }
}

/// A count stops at the query timeout, however many terms its sum has: with the limit raised, a
/// count of 8 hops ran 55 s under a 30 s timeout and returned its answer. Here 6 hops, the longest
/// that is counted.
#[test]
fn a_count_stops_at_the_query_timeout() {
    let db = GrafeoDB::with_config(
        Config::in_memory().with_query_timeout(std::time::Duration::from_millis(50)),
    )
    .unwrap();
    let nodes: Vec<NodeId> = (0..2_000)
        .map(|_| db.create_node(&["V"]).unwrap())
        .collect();
    let mut state = 7_u64;
    for node in &nodes {
        for _ in 0..8 {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            let far = nodes[usize::try_from((state >> 33) % 2_000).unwrap()];
            db.create_edge(*node, far, "E").unwrap();
        }
    }
    let session = db.session();
    let query = format!("MATCH (a:V){} RETURN count(*)", "-[:E]->()".repeat(6));
    let start = std::time::Instant::now();
    let error = session
        .execute_cypher(&query)
        .expect_err("this count takes far longer than 50 ms");
    assert!(error.to_string().contains("GRAFEO-Q003"), "{error}");
    let took = start.elapsed();
    assert!(took.as_secs() < 2, "stopped {took:?} into a 50 ms limit");
}

/// Two nodes with `parallel` edges each way, so that 6 hops from the first have a known count.
fn parallel_edges_both_ways(parallel: usize) -> GrafeoDB {
    let db = GrafeoDB::new_in_memory();
    let start = db.create_node(&["V"]).unwrap();
    let other = db.create_node(&["W"]).unwrap();
    for _ in 0..parallel {
        db.create_edge(start, other, "E").unwrap();
        db.create_edge(other, start, "E").unwrap();
    }
    db
}

const SIX_HOPS: &str =
    "MATCH (a:V)-[:E]->()-[:E]->()-[:E]->()-[:E]->()-[:E]->()-[:E]->() RETURN count(*)";

/// A count too large for an integer is the checked-arithmetic error, never a wrapped number. The
/// walks of 6 hops over 2,000 parallel edges are 2,000^6, about 6.4e19; the largest integer is
/// about 9.2e18.
#[test]
fn a_count_past_the_largest_integer_is_an_error() {
    let db = parallel_edges_both_ways(2_000);
    let error = match db.session().execute(SIX_HOPS) {
        Ok(result) => panic!("a count past the largest integer: {:?}", result.rows()),
        Err(error) => error,
    };
    assert!(
        error.to_string().contains("numeric value out of range"),
        "{error}"
    );
}

/// openCypher's sum walks the edges its terms share, so over this many parallel edges it meets a
/// short timeout before it could overflow. No number is fine, a wrong one is not.
#[test]
fn an_opencypher_count_of_that_size_gives_no_wrong_number() {
    let db = GrafeoDB::with_config(
        Config::in_memory().with_query_timeout(std::time::Duration::from_secs(2)),
    )
    .unwrap();
    let start = db.create_node(&["V"]).unwrap();
    let other = db.create_node(&["W"]).unwrap();
    for _ in 0..2_000 {
        db.create_edge(start, other, "E").unwrap();
        db.create_edge(other, start, "E").unwrap();
    }
    match db.session().execute_cypher(SIX_HOPS) {
        Ok(result) => panic!(
            "(2,000 x 1,999 x 1,998)^2 is no integer: {:?}",
            result.rows()
        ),
        Err(error) => {
            let text = error.to_string();
            assert!(
                text.contains("numeric value out of range") || text.contains("GRAFEO-Q003"),
                "{text}"
            );
        }
    }
}

/// Just below the largest integer the count is exact, where a float or a wrapped step would show:
/// 1,448^6 is about 9.2e18.
#[test]
fn a_count_near_the_largest_integer_is_exact() {
    let parallel: i128 = 1_448;
    let walks = parallel.pow(6);
    assert!(
        walks <= i128::from(i64::MAX),
        "a fixture past the largest integer"
    );
    let db = parallel_edges_both_ways(parallel as usize);
    assert_eq!(
        db.session().execute(SIX_HOPS).unwrap().rows()[0][0],
        Value::Int64(walks as i64),
    );
}

/// openCypher's trails, by formula: the terms of its sum have opposite signs, so a step that
/// wrapped or rounded shows in the difference.
#[test]
fn an_opencypher_count_over_parallel_edges_is_exact() {
    for parallel in [3_i64, 7, 30] {
        let trails = (parallel * (parallel - 1) * (parallel - 2)).pow(2);
        let db = parallel_edges_both_ways(parallel as usize);
        assert_eq!(
            db.session().execute_cypher(SIX_HOPS).unwrap().rows()[0][0],
            Value::Int64(trails),
            "{parallel} parallel edges"
        );
    }
}

/// Parallel edges read alike may count as one branch of the sum, and only those: here some are of
/// another type, some run the other way, some are self-loops, so that hops over one pair of nodes
/// share some of them and not others. Every chain of up to 3 hops over every way and type, and a
/// sample of the longer ones, in both languages, against enumeration.
#[test]
fn parallel_edges_group_only_where_they_read_alike() {
    let link = |key: i64, from: i64, to: i64, kind: &'static str| Link {
        key,
        from,
        to,
        kind,
    };
    let mut links = Vec::new();
    for (from, to, kind, copies) in [
        (0, 1, "E", 3),
        (1, 0, "E", 2),
        (0, 1, "F", 2),
        (1, 0, "F", 1),
        (0, 0, "E", 2),
        (1, 1, "F", 2),
        (1, 2, "E", 2),
        (2, 0, "F", 2),
    ] {
        for _ in 0..copies {
            links.push(link(links.len() as i64, from, to, kind));
        }
    }
    let graph = Multigraph {
        vertices: (0..3)
            .map(|key| Vertex {
                key,
                labels: vec!["V"],
            })
            .collect(),
        links,
    };
    let db = GrafeoDB::new_in_memory();
    load(&db, &graph);
    let mut session = db.session();

    let mut forms = Vec::new();
    for way in [Way::Out, Way::In, Way::Either] {
        for kind in [Some("E"), Some("F"), None] {
            forms.push(Hop {
                way,
                kind,
                end: None,
            });
        }
    }
    let mut chains = Vec::new();
    let with_hops = |hops: Vec<Hop>| Chain {
        first: Some("V"),
        hops,
        grouped: false,
        below: None,
    };
    for one in &forms {
        chains.push(with_hops(vec![one.clone()]));
        for two in &forms {
            chains.push(with_hops(vec![one.clone(), two.clone()]));
            for three in &forms {
                chains.push(with_hops(vec![one.clone(), two.clone(), three.clone()]));
            }
        }
    }
    let mut random = Random::new(41);
    for length in 4..=6 {
        for _ in 0..60 {
            chains.push(with_hops(
                (0..length)
                    .map(|_| forms[random.below(forms.len())].clone())
                    .collect(),
            ));
        }
    }
    let mut differences = Vec::new();
    check(
        &mut session,
        &graph,
        &chains,
        "parallel",
        41,
        &mut differences,
    );
    assert!(
        differences.is_empty(),
        "{} differences:\n{}",
        differences.len(),
        differences.join("\n")
    );
}

/// Each edge once over a size the grouping makes possible: the terms of the sum have opposite signs
/// and pass through values larger than the answer.
#[test]
fn an_opencypher_count_near_the_largest_integer_is_exact() {
    let parallel: i128 = 1_449;
    let trails = (parallel * (parallel - 1) * (parallel - 2)).pow(2);
    assert!(
        trails <= i128::from(i64::MAX),
        "a fixture past the largest integer"
    );
    let db = parallel_edges_both_ways(parallel as usize);
    assert_eq!(
        db.session().execute_cypher(SIX_HOPS).unwrap().rows()[0][0],
        Value::Int64(trails as i64),
    );
}
