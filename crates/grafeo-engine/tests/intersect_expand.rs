//! Intersecting adjacency lists for cyclic patterns (#714 cluster 6, `PlannerRule::IntersectExpand`).
//!
//! Tests first, from the design: per input row, intersect every adjacency touching a variable,
//! galloping from the shortest list on the sorted part, with the unsorted delta on its own. Rows
//! are those of the plain plan, parallel edges and edge uniqueness included; off without
//! backward adjacency, above a write, or when the column order is not free. It does not compile
//! until the rule exists. Rows over random multigraphs are `planner_rules_results.rs`'s, which
//! picks the rule up from `PlannerRule::ALL`.

#![cfg(all(feature = "gql", feature = "cypher"))]
#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss,
    reason = "test graphs of a few thousand nodes"
)]

use std::time::Duration;

use grafeo_common::types::{NodeId, Value};
use grafeo_engine::query::optimizer::{PlannerRule, PlannerRules};
use grafeo_engine::session::Session;
use grafeo_engine::{Config, GrafeoDB};

/// The rows are what these tests read: a slow build must not make them a test of the query
/// timeout, and a bad plan must still fail a test by name, well inside the job's own limit.
const SLOW_BUILD: Duration = Duration::from_secs(600);

const TRIANGLE: &str =
    "MATCH (a:Person)-[:KNOWS]->(b:Person)-[:KNOWS]->(c:Person)-[:KNOWS]->(a) RETURN a.k, b.k, c.k";
const SQUARE: &str = "MATCH (a:Person)-[:KNOWS]->(b:Person)-[:KNOWS]->(c:Person)-[:KNOWS]->(d:Person)-[:KNOWS]->(a) \
     RETURN a.k, b.k, c.k, d.k";

const MARK: &str = "[IntersectExpand into:";

fn text(session: &Session, query: &str) -> String {
    session
        .execute(query)
        .unwrap()
        .rows()
        .iter()
        .map(|row| match &row[0] {
            Value::String(line) => line.to_string(),
            other => format!("{other:?}"),
        })
        .collect::<Vec<_>>()
        .join("\n")
}

fn explain(session: &Session, query: &str) -> String {
    text(session, &format!("EXPLAIN {query}"))
}

fn rows(session: &Session, query: &str) -> Vec<String> {
    let result = session.execute(query).unwrap();
    let mut rows: Vec<String> = result.rows().iter().map(|row| format!("{row:?}")).collect();
    rows.sort();
    rows
}

fn without(db: &GrafeoDB) -> Session {
    let mut session = db.session();
    session.set_planner_rules(PlannerRules::all().with(PlannerRule::IntersectExpand, false));
    session
}

/// `size` people with `degree` random friends each, and some hubs that know everybody.
fn people(config: Config, size: u64, degree: u64, hubs: u64) -> GrafeoDB {
    let db = GrafeoDB::with_config(config.with_query_timeout(SLOW_BUILD)).unwrap();
    let ids: Vec<NodeId> = (0..size)
        .map(|key| {
            db.create_node_with_props(&["Person"], [("k", Value::Int64(key as i64))])
                .unwrap()
        })
        .collect();
    let mut state = 11_u64;
    let mut next = |bound: u64| {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        (state >> 33) % bound
    };
    for key in 0..size {
        for _ in 0..degree {
            db.create_edge(ids[key as usize], ids[next(size) as usize], "KNOWS")
                .unwrap();
        }
    }
    for hub in 0..hubs {
        for key in 0..size {
            db.create_edge(ids[hub as usize], ids[key as usize], "KNOWS")
                .unwrap();
            db.create_edge(ids[key as usize], ids[hub as usize], "KNOWS")
                .unwrap();
        }
    }
    db
}

/// A cost model can quietly never pick the operator, so the shapes it is for are pinned.
#[test]
fn triangles_and_squares_intersect() {
    let db = people(Config::in_memory(), 100, 6, 1);
    let usual = db.session();
    let off = without(&db);
    for query in [TRIANGLE, SQUARE] {
        assert!(
            explain(&usual, query).contains(MARK),
            "{query}\n{}",
            explain(&usual, query)
        );
        assert!(!explain(&off, query).contains(MARK), "{query}");
        assert_eq!(rows(&usual, query), rows(&off, query), "{query}");
    }
    let mut plain = db.session();
    plain.set_planner_rules(PlannerRules::plain());
    assert!(!explain(&plain, TRIANGLE).contains(MARK));
}

/// The reason for it: a hub's lists are not walked for each row. The lookup alone reads every
/// edge from the hub at the middle hop; the intersection reads a fraction.
#[test]
fn a_triangle_through_hubs_reads_far_less_than_a_lookup() {
    let db = people(Config::in_memory(), 1_000, 4, 3);
    let usual = db.session();
    let off = without(&db);
    let query = "MATCH (a:Person {k: 0})-[:KNOWS]->(b:Person)-[:KNOWS]->(c:Person)-[:KNOWS]->(a) RETURN b.k, c.k";
    let (on, plain) = (usual.execute(query).unwrap(), off.execute(query).unwrap());
    assert_eq!(on.row_count(), plain.row_count());
    let (on_reads, off_reads) = (on.rows_scanned().unwrap(), plain.rows_scanned().unwrap());
    assert!(on_reads * 4 < off_reads, "{on_reads} against {off_reads}");
}

/// Parallel edges are rows of their own, as many as the plain plan has; Cypher binds each edge
/// once; a self-loop is a triangle of one node in GQL and none in Cypher, with one edge.
#[test]
fn parallel_edges_and_self_loops_give_the_rows_of_the_plain_plan() {
    let db = GrafeoDB::new_in_memory();
    db.session()
        .execute(
            "INSERT (a:Person {k: 1})-[:KNOWS]->(b:Person {k: 2}),
                    (a)-[:KNOWS]->(b),
                    (b)-[:KNOWS]->(c:Person {k: 3}),
                    (b)-[:KNOWS]->(c),
                    (b)-[:KNOWS]->(c),
                    (c)-[:KNOWS]->(a),
                    (m:Person {k: 4})-[:KNOWS]->(m),
                    (n:Person {k: 5})-[:KNOWS]->(n),
                    (n)-[:KNOWS]->(n),
                    (n)-[:KNOWS]->(n)",
        )
        .unwrap();
    let usual = db.session();
    let off = without(&db);
    assert_eq!(rows(&usual, TRIANGLE), rows(&off, TRIANGLE));
    // 2 x 3 x 1 from each of the three starts, one for Mia's loop, 27 for the three loops.
    let result = usual.execute(TRIANGLE).unwrap();
    assert_eq!(result.row_count(), 3 * 6 + 1 + 27);
    let sorted = |session: &Session| {
        let result = session.execute_cypher(TRIANGLE).unwrap();
        let mut rows: Vec<String> = result.rows().iter().map(|row| format!("{row:?}")).collect();
        rows.sort();
        rows
    };
    assert_eq!(sorted(&usual), sorted(&off));
    // The three loops bind each edge once: 3 x 2 x 1.
    assert_eq!(
        usual.execute_cypher(TRIANGLE).unwrap().row_count(),
        3 * 6 + 6
    );
}

/// Sorted lists with a delta on top, removed edges among them, a transaction's own, and a past
/// epoch: enough edges for a merge first.
#[test]
fn sorted_lists_and_deltas_read_the_same() {
    let config =
        Config::in_memory().with_history_retention(grafeo_engine::HistoryRetention::Forever);
    let db = people(config, 300, 6, 1);
    let mut usual = db.session();
    let mut off = without(&db);
    let same = |usual: &Session, off: &Session, stage: &str| {
        assert_eq!(rows(usual, TRIANGLE), rows(off, TRIANGLE), "{stage}");
    };
    same(&usual, &off, "merged");
    let epoch = db.current_epoch();
    for statement in [
        "MATCH (a:Person {k: 1}), (b:Person {k: 2}), (c:Person {k: 3}) \
         INSERT (a)-[:KNOWS]->(b), (b)-[:KNOWS]->(c), (c)-[:KNOWS]->(a)",
        "MATCH (:Person {k: 5})-[e:KNOWS]->() DELETE e",
        "MATCH (n:Person {k: 7}) DETACH DELETE n",
    ] {
        usual.execute(statement).unwrap();
        same(&usual, &off, statement);
    }
    usual.set_viewing_epoch(epoch);
    off.set_viewing_epoch(epoch);
    same(&usual, &off, "past epoch");
    usual.clear_viewing_epoch();
    off.clear_viewing_epoch();
    // Each transaction alone: the two would conflict over one edge.
    let mut own = Vec::new();
    for session in [&mut usual, &mut off] {
        session.begin_transaction().unwrap();
        session
            .execute(
                "MATCH (a:Person {k: 10}), (b:Person {k: 11}), (c:Person {k: 12}) \
                 INSERT (a)-[:KNOWS]->(b), (b)-[:KNOWS]->(c), (c)-[:KNOWS]->(a)",
            )
            .unwrap();
        session
            .execute("MATCH (:Person {k: 13})-[e:KNOWS]->() DELETE e")
            .unwrap();
        own.push(rows(session, TRIANGLE));
        session.rollback().unwrap();
    }
    assert_eq!(own[0], own[1], "own writes");
    same(&usual, &off, "after the rollback");
}

/// Where it stays off: no backward adjacency, a write below it, columns whose order shows.
#[test]
fn it_stays_off_where_it_must() {
    let db = people(Config::in_memory().without_backward_edges(), 200, 6, 0);
    assert!(!explain(&db.session(), TRIANGLE).contains(MARK));

    let db = people(Config::in_memory(), 200, 6, 0);
    let usual = db.session();
    for query in [
        "MATCH (s:Person {k: 1}) INSERT (s)-[:KNOWS]->(s) WITH s \
         MATCH (s)-[:KNOWS]->(b:Person)-[:KNOWS]->(c:Person)-[:KNOWS]->(s) RETURN b.k",
        "MATCH (a:Person)-[:KNOWS]->(b:Person)-[:KNOWS]->(c:Person)-[:KNOWS]->(a) RETURN *",
    ] {
        assert!(!explain(&usual, query).contains(MARK), "{query}");
    }
}

/// The order of the variables: pinned first, never by hash order, the same on every run.
#[test]
fn the_variable_order_is_fixed() {
    let plans: Vec<String> = (0..4)
        .map(|_| {
            let db = people(Config::in_memory(), 300, 6, 1);
            explain(&db.session(), SQUARE)
        })
        .collect();
    assert!(plans.iter().all(|plan| plan == &plans[0]), "{plans:?}");

    let db = people(Config::in_memory(), 300, 6, 1);
    db.create_property_index("k").unwrap();
    let query = "MATCH (a:Person)-[:KNOWS]->(b:Person)-[:KNOWS]->(c:Person {k: 7})-[:KNOWS]->(a) RETURN a.k, b.k";
    let plan = explain(&db.session(), query);
    let deepest = plan
        .lines()
        .rfind(|line| line.contains("NodeScan ("))
        .expect(&plan);
    assert!(deepest.contains("(c:"), "pinned first:\n{plan}");
}

// ---------------------------------------------------------------------------
// Every way the hops can run, against rows worked out by hand
// ---------------------------------------------------------------------------

/// The ways a hop can run, as a pattern and as a question about one stored edge.
#[derive(Clone, Copy, Debug)]
enum Way {
    Out,
    In,
    Either,
}

const WAYS: [Way; 3] = [Way::Out, Way::In, Way::Either];

impl Way {
    fn pattern(self, kind: &str) -> String {
        let label = if kind.is_empty() {
            String::new()
        } else {
            format!(":{kind}")
        };
        match self {
            Way::Out => format!("-[{label}]->"),
            Way::In => format!("<-[{label}]-"),
            Way::Either => format!("-[{label}]-"),
        }
    }

    /// Whether the edge goes from `near` to `far` this way. A self-loop is one edge either way.
    fn holds(self, edge: &Stored, near: u64, far: u64) -> bool {
        match self {
            Way::Out => edge.from == near && edge.to == far,
            Way::In => edge.from == far && edge.to == near,
            Way::Either => {
                (edge.from == near && edge.to == far) || (edge.from == far && edge.to == near)
            }
        }
    }
}

#[derive(Clone, Copy)]
struct Stored {
    from: u64,
    to: u64,
    knows: bool,
}

/// Random edges with self-loops and parallel edges, a tenth of another type, and hubs with
/// their lists both ways; the keys are the node positions.
fn multigraph(size: u64, edges: u64, hubs: u64, seed: u64) -> (GrafeoDB, Vec<Stored>) {
    let db = GrafeoDB::with_config(Config::in_memory().with_query_timeout(SLOW_BUILD)).unwrap();
    let ids: Vec<NodeId> = (0..size)
        .map(|key| {
            db.create_node_with_props(&["Person"], [("k", Value::Int64(key as i64))])
                .unwrap()
        })
        .collect();
    let mut state = seed;
    let mut next = move |bound: u64| {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        (state >> 33) % bound
    };
    let mut stored = Vec::new();
    let add = |db: &GrafeoDB, from: u64, to: u64, knows: bool, stored: &mut Vec<Stored>| {
        let kind = if knows { "KNOWS" } else { "LIKES" };
        db.create_edge(ids[from as usize], ids[to as usize], kind)
            .unwrap();
        stored.push(Stored { from, to, knows });
    };
    for _ in 0..edges {
        let from = next(size);
        // One in eight a self-loop, one in eight a copy of the last edge.
        let (from, to) = match next(8) {
            0 => (from, from),
            1 if !stored.is_empty() => {
                let last: &Stored = stored.last().unwrap();
                (last.from, last.to)
            }
            _ => (from, next(size)),
        };
        let knows = next(10) != 0;
        add(&db, from, to, knows, &mut stored);
    }
    // Every node has an edge with each hub, so the lists merge at the edges they would; one in
    // four is of the type the queries read, as rows through a hub go by the square of those.
    for hub in 0..hubs {
        for key in 0..size {
            let knows = next(4) == 0;
            if next(2) == 0 {
                add(&db, hub, key, knows, &mut stored);
            } else {
                add(&db, key, hub, knows, &mut stored);
            }
        }
    }
    (db, stored)
}

/// The triangle by hand: per node triple, how many edge triples close it. openCypher binds each
/// edge once, GQL does not ask that here.
fn triangles_by_hand(
    stored: &[Stored],
    size: u64,
    ways: [Way; 3],
    kind: &str,
    once_each: bool,
) -> Vec<String> {
    let edges: Vec<(usize, Stored)> = stored
        .iter()
        .copied()
        .enumerate()
        .filter(|(_, edge)| match kind {
            "" | "KNOWS|LIKES" => true,
            "KNOWS" => edge.knows,
            "LIKES" => !edge.knows,
            _ => false,
        })
        .collect();
    let mut touching: Vec<Vec<(usize, Stored)>> = vec![Vec::new(); size as usize];
    for (index, edge) in &edges {
        touching[edge.from as usize].push((*index, *edge));
        if edge.to != edge.from {
            touching[edge.to as usize].push((*index, *edge));
        }
    }
    let far_end = |edge: &Stored, near: u64| {
        if edge.from == near {
            edge.to
        } else {
            edge.from
        }
    };
    let mut rows = Vec::new();
    for a in 0..size {
        for (first, one) in &touching[a as usize] {
            let b = far_end(one, a);
            if !ways[0].holds(one, a, b) {
                continue;
            }
            for (second, two) in &touching[b as usize] {
                let c = far_end(two, b);
                if !ways[1].holds(two, b, c) {
                    continue;
                }
                if once_each && first == second {
                    continue;
                }
                for (third, three) in &touching[c as usize] {
                    if far_end(three, c) != a || !ways[2].holds(three, c, a) {
                        continue;
                    }
                    if once_each && (third == first || third == second) {
                        continue;
                    }
                    rows.push(format!(
                        "{:?}",
                        vec![
                            Value::Int64(a as i64),
                            Value::Int64(b as i64),
                            Value::Int64(c as i64)
                        ]
                    ));
                }
            }
        }
    }
    rows.sort();
    rows
}

fn every_way() -> Vec<[Way; 3]> {
    let mut all = Vec::new();
    for one in WAYS {
        for two in WAYS {
            for three in WAYS {
                all.push([one, two, three]);
            }
        }
    }
    all
}

fn check_every_way(db: &GrafeoDB, stored: &[Stored], size: u64, kinds: &[&str]) -> (usize, usize) {
    let usual = db.session();
    let off = without(db);
    let mut plain = db.session();
    plain.set_planner_rules(PlannerRules::plain());
    let (mut with_rows, mut with_rule) = (0, 0);
    for ways in every_way() {
        for kind in kinds {
            let query = format!(
                "MATCH (a:Person){}(b:Person){}(c:Person){}(a) RETURN a.k, b.k, c.k",
                ways[0].pattern(kind),
                ways[1].pattern(kind),
                ways[2].pattern(kind)
            );
            let want = triangles_by_hand(stored, size, ways, kind, false);
            with_rows += usize::from(!want.is_empty());
            with_rule += usize::from(explain(&usual, &query).contains(MARK));
            assert_eq!(rows(&usual, &query), want, "GQL, usual: {query}");
            assert_eq!(rows(&off, &query), want, "GQL, without the rule: {query}");
            assert_eq!(rows(&plain, &query), want, "GQL, plain: {query}");

            let want = triangles_by_hand(stored, size, ways, kind, true);
            let cypher = |session: &Session| {
                let result = session.execute_cypher(&query).unwrap();
                let mut rows: Vec<String> =
                    result.rows().iter().map(|row| format!("{row:?}")).collect();
                rows.sort();
                rows
            };
            assert_eq!(cypher(&usual), want, "Cypher, usual: {query}");
            assert_eq!(cypher(&off), want, "Cypher, without the rule: {query}");
            assert_eq!(cypher(&plain), want, "Cypher, plain: {query}");
        }
    }
    (with_rows, with_rule)
}

/// All 27 ways over a dense little multigraph, typed and untyped, self-loops and parallel edges,
/// in GQL and in Cypher where each edge binds once.
#[test]
fn every_way_of_the_hops_gives_the_rows_by_hand() {
    for seed in [3, 5, 9] {
        let (db, stored) = multigraph(14, 90, 0, seed);
        let (with_rows, with_rule) = check_every_way(
            &db,
            &stored,
            14,
            &["KNOWS", "", "KNOWS|LIKES", "LIKES", "NOSUCH"],
        );
        assert!(
            with_rows >= 27,
            "seed {seed}: {with_rows} forms had rows: a weak graph"
        );
        assert!(with_rule > 0, "seed {seed}: no form used the rule");
    }
}

/// The same with merged lists and hubs, where the two sides of a row differ most in length.
#[test]
fn every_way_of_the_hops_with_merged_lists_and_hubs() {
    let (db, stored) = multigraph(300, 1_800, 2, 7);
    let (with_rows, with_rule) = check_every_way(&db, &stored, 300, &["KNOWS"]);
    assert_eq!(with_rows, 27, "the hubs close a triangle every way");
    assert!(with_rule > 0, "no form used the rule");
}

/// An undirected hop keeps the rule: it reads both sides of a node, not a list of its own for
/// each row, and answers as the lookup does on a hub.
#[test]
fn an_undirected_hop_keeps_the_rule() {
    let (db, _) = multigraph(300, 1_800, 1, 11);
    let session = db.session();
    for query in [
        "MATCH (a:Person)-[:KNOWS]-(b:Person)-[:KNOWS]-(c:Person)-[:KNOWS]-(a) RETURN a.k, b.k, c.k",
        "MATCH (a:Person)-[:KNOWS]->(b:Person)-[:KNOWS]-(c:Person)-[:KNOWS]->(a) RETURN a.k, b.k, c.k",
        "MATCH (a:Person {k: 0})-[:KNOWS]-(b:Person)-[:KNOWS]-(c:Person)-[:KNOWS]-(a) RETURN b.k, c.k",
    ] {
        let plan = explain(&session, query);
        assert!(plan.contains(MARK), "{query}:\n{plan}");
    }
    let hub = "MATCH (a:Person {k: 0})-[:KNOWS]-(b:Person)-[:KNOWS]-(c:Person)-[:KNOWS]-(a) WHERE b.k < c.k RETURN count(*)";
    let usual = session.execute(hub).unwrap();
    let off = without(&db).execute(hub).unwrap();
    assert_eq!(usual.rows(), off.rows());
}

/// Sorted lists with entries in a delta on either side, and edges removed from a sorted list: the
/// hub's list merged, then new edges on some nodes only, then edges deleted, each against the
/// rows by hand. All the sorted and delta combinations come up: the hub merged and its partner
/// not, and the other way round.
#[test]
fn deltas_and_removed_edges_give_the_rows_by_hand() {
    let (db, mut stored) = multigraph(300, 1_800, 2, 13);
    let ids: Vec<NodeId> = {
        let result = db
            .session()
            .execute("MATCH (n:Person) RETURN id(n), n.k ORDER BY n.k")
            .unwrap();
        result
            .rows()
            .iter()
            .map(|row| match row[0] {
                Value::Int64(id) => NodeId::new(id as u64),
                _ => panic!("an id"),
            })
            .collect()
    };
    let (_, with_rule) = check_every_way(&db, &stored, 300, &["KNOWS"]);
    assert!(with_rule > 0, "no form used the rule");
    // New edges on top of what merged, at nodes that are hubs and nodes that are not.
    for key in 0..60_u64 {
        let (from, to) = (key * 5 % 300, (key * 7 + 1) % 300);
        db.create_edge(ids[from as usize], ids[to as usize], "KNOWS")
            .unwrap();
        stored.push(Stored {
            from,
            to,
            knows: true,
        });
        db.create_edge(ids[0], ids[to as usize], "KNOWS").unwrap();
        stored.push(Stored {
            from: 0,
            to,
            knows: true,
        });
    }
    check_every_way(&db, &stored, 300, &["KNOWS"]);
    // Edges removed from sorted lists and from the deltas.
    let session = db.session();
    for key in [3_i64, 50, 100] {
        session
            .execute(&format!(
                "MATCH (:Person {{k: {key}}})-[e:KNOWS]->() DELETE e"
            ))
            .unwrap();
        stored.retain(|edge| !(edge.knows && edge.from == key as u64));
    }
    check_every_way(&db, &stored, 300, &["KNOWS"]);
}

/// An older read view sees what stood then: rows of the rule against the plain plan, every way.
#[test]
fn an_older_view_reads_the_same_every_way() {
    let config =
        Config::in_memory().with_history_retention(grafeo_engine::HistoryRetention::Forever);
    // Enough edges for a merge. No hub: the rows through one go by its square.
    let db = people(config, 200, 6, 0);
    let epoch = db.current_epoch();
    let writer = db.session();
    writer
        .execute("MATCH (:Person {k: 5})-[e:KNOWS]->() DELETE e")
        .unwrap();
    writer
        .execute(
            "MATCH (a:Person {k: 1}), (b:Person {k: 2}), (c:Person {k: 3}) \
             INSERT (a)-[:KNOWS]->(b), (b)-[:KNOWS]->(c), (c)-[:KNOWS]->(a), (c)-[:KNOWS]->(a)",
        )
        .unwrap();
    let mut usual = db.session();
    let mut off = without(&db);
    let mut plain = db.session();
    plain.set_planner_rules(PlannerRules::plain());
    for session in [&mut usual, &mut off, &mut plain] {
        session.set_viewing_epoch(epoch);
    }
    for ways in every_way() {
        let query = format!(
            "MATCH (a:Person){}(b:Person){}(c:Person){}(a) RETURN a.k, b.k, c.k",
            ways[0].pattern("KNOWS"),
            ways[1].pattern("KNOWS"),
            ways[2].pattern("KNOWS")
        );
        let want = rows(&plain, &query);
        assert_eq!(rows(&usual, &query), want, "usual: {query}");
        assert_eq!(rows(&off, &query), want, "without the rule: {query}");
    }
}

// ---------------------------------------------------------------------------
// A short list against a hub's delta
// ---------------------------------------------------------------------------

/// A hub (key 0) with `spokes` edges each way and of two types, with its self-loops and parallel
/// edges, and `shorts` nodes (keys 1 to 3) of 1 to 3 entries each that reach the hub and its
/// spokes, with self-loops and parallel edges of their own and on a far end. Under 1,024 edges all
/// of the hub's list is in its delta; past it part of it is sorted.
fn short_against_a_hub(spokes: u64, config: Config) -> (GrafeoDB, Vec<NodeId>, Vec<Stored>) {
    let db = GrafeoDB::with_config(config.with_query_timeout(SLOW_BUILD)).unwrap();
    let size = spokes + 4;
    let ids: Vec<NodeId> = (0..size)
        .map(|key| {
            db.create_node_with_props(&["Person"], [("k", Value::Int64(key as i64))])
                .unwrap()
        })
        .collect();
    let mut stored = Vec::new();
    let mut add = |from: u64, to: u64, knows: bool| {
        let kind = if knows { "KNOWS" } else { "LIKES" };
        // Each edge says which it is, as `w`, its place in `stored`.
        db.create_edge_with_props(
            ids[from as usize],
            ids[to as usize],
            kind,
            [("w", Value::Int64(stored.len() as i64))],
        )
        .unwrap();
        stored.push(Stored { from, to, knows });
    };
    for spoke in 4..size {
        match spoke % 4 {
            0 => add(0, spoke, true),
            1 => add(spoke, 0, true),
            2 => add(0, spoke, false),
            _ => {
                add(0, spoke, true);
                add(spoke, 0, true);
            }
        }
    }
    // The hub's own: self-loops of both types, and parallel edges to a spoke.
    add(0, 0, true);
    add(0, 0, true);
    add(0, 0, false);
    add(0, 4, true);
    add(0, 4, true);
    // Short lists, of each way round: one entry, two, three, reaching the hub's neighbours.
    add(1, 4, true);
    add(2, 5, true);
    add(5, 2, true);
    add(3, 6, true);
    add(3, 7, false);
    add(8, 3, true);
    // Each short node to and from the hub.
    add(0, 1, true);
    add(2, 0, true);
    add(3, 0, true);
    add(0, 3, true);
    // A self-loop on a short node and on the far end they share with the hub.
    add(2, 2, true);
    add(4, 4, true);
    add(4, 4, false);
    // Parallel edges from a short node to a shared far end.
    add(1, 4, true);
    (db, ids, stored)
}

/// The sessions to compare: the usual, the written order with the rule (so that the hop runs as
/// written, hub on either side), and the plain plan.
fn short_list_sessions(db: &GrafeoDB) -> [(&'static str, Session); 3] {
    let usual = db.session();
    let mut fixed = db.session();
    fixed.set_planner_rules(PlannerRules::all().with(PlannerRule::JoinOrder, false));
    let mut plain = db.session();
    plain.set_planner_rules(PlannerRules::plain());
    [("usual", usual), ("written order", fixed), ("plain", plain)]
}

/// The triangles from the keys under 4, by hand, in both languages, every way on every hop. With
/// `a` the hub and `b` a short node, and the other way round, each list takes each side.
fn check_short_lists(db: &GrafeoDB, stored: &[Stored], size: u64, kinds: &[&str]) -> usize {
    let sessions = short_list_sessions(db);
    let mut used = 0;
    for ways in every_way() {
        for kind in kinds {
            let query = format!(
                "MATCH (a:Person){}(b:Person){}(c:Person){}(a) WHERE a.k < 4 RETURN a.k, b.k, c.k",
                ways[0].pattern(kind),
                ways[1].pattern(kind),
                ways[2].pattern(kind)
            );
            let keep = |rows: Vec<String>| -> Vec<String> {
                rows.into_iter()
                    .filter(|row| {
                        ["[Int64(0),", "[Int64(1),", "[Int64(2),", "[Int64(3),"]
                            .iter()
                            .any(|start| row.starts_with(start))
                    })
                    .collect()
            };
            let walks = keep(triangles_by_hand(stored, size, ways, kind, false));
            let trails = keep(triangles_by_hand(stored, size, ways, kind, true));
            for (name, session) in &sessions {
                used += usize::from(explain(session, &query).contains(MARK));
                assert_eq!(rows(session, &query), walks, "GQL, {name}: {query}");
                let result = session.execute_cypher(&query).unwrap();
                let mut got: Vec<String> =
                    result.rows().iter().map(|row| format!("{row:?}")).collect();
                got.sort();
                assert_eq!(got, trails, "Cypher, {name}: {query}");
            }
        }
    }
    used
}

/// A short list against a hub whose list is all in its delta: every way, typed, untyped, several
/// types and a type the store does not have.
#[test]
fn a_short_list_against_a_hubs_delta_gives_the_rows_by_hand() {
    let (db, _, stored) = short_against_a_hub(200, Config::in_memory());
    let used = check_short_lists(
        &db,
        &stored,
        204,
        &["KNOWS", "LIKES", "", "KNOWS|LIKES", "NOSUCH"],
    );
    assert!(used > 0, "no plan used the rule");
}

/// The same with part of the hub's list sorted and the rest in its delta.
#[test]
fn a_short_list_against_a_hubs_sorted_part_and_delta_gives_the_rows_by_hand() {
    let (db, _, stored) = short_against_a_hub(1_200, Config::in_memory());
    let used = check_short_lists(&db, &stored, 1_204, &["KNOWS", ""]);
    assert!(used > 0, "no plan used the rule");
}

/// Edges removed from the hub's delta and from a short list, by a statement, and then the rows.
#[test]
fn removed_edges_of_a_short_list_and_a_hub_give_the_rows_by_hand() {
    let (db, ids, mut stored) = short_against_a_hub(200, Config::in_memory());
    let session = db.session();
    // A spoke's edges, one of the parallel ones of the short node, and one of the hub's loops.
    for (query, gone) in [
        (
            "MATCH (:Person {k: 7})-[e]-() DELETE e",
            Box::new(|edge: &Stored| edge.from == 7 || edge.to == 7)
                as Box<dyn Fn(&Stored) -> bool>,
        ),
        (
            "MATCH (:Person {k: 3})-[e:LIKES]->() DELETE e",
            Box::new(|edge: &Stored| edge.from == 3 && !edge.knows),
        ),
        (
            "MATCH (h:Person {k: 0})-[e:LIKES]->(h) DELETE e",
            Box::new(|edge: &Stored| edge.from == 0 && edge.to == 0 && !edge.knows),
        ),
    ] {
        session.execute(query).unwrap();
        stored.retain(|edge| !gone(edge));
        check_short_lists(&db, &stored, 204, &["KNOWS", ""]);
    }
    assert_eq!(ids.len(), 204);
}

/// The triangles from the keys under 4 with which edge serves which hop, by hand.
fn triangles_with_their_edges(stored: &[Stored], size: u64, ways: [Way; 3]) -> Vec<String> {
    let edges: Vec<(usize, Stored)> = stored
        .iter()
        .copied()
        .enumerate()
        .filter(|(_, edge)| edge.knows)
        .collect();
    let mut touching: Vec<Vec<(usize, Stored)>> = vec![Vec::new(); size as usize];
    for (index, edge) in &edges {
        touching[edge.from as usize].push((*index, *edge));
        if edge.to != edge.from {
            touching[edge.to as usize].push((*index, *edge));
        }
    }
    let far_end = |edge: &Stored, near: u64| {
        if edge.from == near {
            edge.to
        } else {
            edge.from
        }
    };
    let mut rows = Vec::new();
    for a in 0..4_u64 {
        for (first, one) in &touching[a as usize] {
            let b = far_end(one, a);
            if !ways[0].holds(one, a, b) {
                continue;
            }
            for (second, two) in &touching[b as usize] {
                let c = far_end(two, b);
                if !ways[1].holds(two, b, c) {
                    continue;
                }
                for (third, three) in &touching[c as usize] {
                    if far_end(three, c) != a || !ways[2].holds(three, c, a) {
                        continue;
                    }
                    rows.push(format!(
                        "{:?}",
                        vec![
                            Value::Int64(a as i64),
                            Value::Int64(b as i64),
                            Value::Int64(c as i64),
                            Value::Int64(*first as i64),
                            Value::Int64(*second as i64),
                            Value::Int64(*third as i64),
                        ]
                    ));
                }
            }
        }
    }
    rows.sort();
    rows
}

/// Which edge serves which hop: an intersection gives a pair of edges for each common node, and
/// the path from the short list swaps them when the short list is the far hop. Rows that name
/// only nodes cannot see that, so these name each edge.
#[test]
fn each_edge_serves_its_own_hop_with_a_short_list_on_either_side() {
    for spokes in [200, 1_200] {
        let (db, _, stored) = short_against_a_hub(spokes, Config::in_memory());
        let sessions = short_list_sessions(&db);
        let mut used = 0;
        for ways in every_way() {
            let hop = |way: Way, name: &str| match way {
                Way::Out => format!("-[{name}:KNOWS]->"),
                Way::In => format!("<-[{name}:KNOWS]-"),
                Way::Either => format!("-[{name}:KNOWS]-"),
            };
            let query = format!(
                "MATCH (a:Person){}(b:Person){}(c:Person){}(a) WHERE a.k < 4 \
                 RETURN a.k, b.k, c.k, e1.w, e2.w, e3.w",
                hop(ways[0], "e1"),
                hop(ways[1], "e2"),
                hop(ways[2], "e3")
            );
            let want = triangles_with_their_edges(&stored, spokes + 4, ways);
            for (name, session) in &sessions {
                used += usize::from(explain(session, &query).contains(MARK));
                assert_eq!(
                    rows(session, &query),
                    want,
                    "{spokes} spokes, {name}: {query}"
                );
            }
        }
        assert!(used > 0, "no plan used the rule");
    }
}

/// An older view reads what stood then, every way, against the plain plan.
#[test]
fn an_older_view_of_a_short_list_and_a_hub_reads_the_same() {
    let (db, _, _) = short_against_a_hub(
        200,
        Config::in_memory().with_history_retention(grafeo_engine::HistoryRetention::Forever),
    );
    let epoch = db.current_epoch();
    let writer = db.session();
    writer
        .execute("MATCH (:Person {k: 2})-[e:KNOWS]->() DELETE e")
        .unwrap();
    writer
        .execute("MATCH (a:Person {k: 1}), (b:Person {k: 0}) INSERT (a)-[:KNOWS]->(b), (b)-[:KNOWS]->(a)")
        .unwrap();
    let mut sessions = short_list_sessions(&db);
    for (_, session) in &mut sessions {
        session.set_viewing_epoch(epoch);
    }
    for ways in every_way() {
        let query = format!(
            "MATCH (a:Person){}(b:Person){}(c:Person){}(a) WHERE a.k < 4 RETURN a.k, b.k, c.k",
            ways[0].pattern("KNOWS"),
            ways[1].pattern("KNOWS"),
            ways[2].pattern("KNOWS")
        );
        let want = rows(&sessions[2].1, &query);
        assert_eq!(rows(&sessions[0].1, &query), want, "usual: {query}");
        assert_eq!(rows(&sessions[1].1, &query), want, "written order: {query}");
    }
}

/// A transaction's own edges, in the hub's delta and on a short list, and after its rollback.
#[test]
fn a_transactions_own_edges_on_a_short_list_read_the_same() {
    let (db, _, _) = short_against_a_hub(200, Config::in_memory());
    let query = "MATCH (a:Person)-[:KNOWS]->(b:Person)-[:KNOWS]->(c:Person)-[:KNOWS]->(a) WHERE a.k < 4 RETURN a.k, b.k, c.k";
    let before = rows(&db.session(), query);
    let mut sets = short_list_sessions(&db);
    let mut seen = Vec::new();
    for (_, session) in &mut sets {
        session.begin_transaction().unwrap();
        session
            .execute("MATCH (a:Person {k: 1}), (b:Person {k: 0}) INSERT (a)-[:KNOWS]->(b), (b)-[:KNOWS]->(a)")
            .unwrap();
        seen.push(rows(session, query));
        session.rollback().unwrap();
    }
    assert_eq!(seen[0], seen[2], "usual against plain");
    assert_eq!(seen[1], seen[2], "written order against plain");
    assert_eq!(rows(&db.session(), query), before, "after the rollbacks");
}
