//! Degrees sampled at plan time for comma patterns (#714, "Degrees by sampling").
//!
//! Tests first, from the design: `GraphStore::sampled_degree(label, edge_type, direction)` reads
//! up to 256 nodes at low-discrepancy ranks, at the latest committed state, cached on the store.
//! It does not compile until that method and `degree_samples_taken` exist. Rows never depend on
//! a sample: that is `planner_rules_results.rs`'s to check, and here the answers on skewed data.

#![cfg(all(feature = "gql", feature = "cypher"))]
#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss,
    deprecated,
    reason = "test graphs of a few thousand nodes, and the raw store's sampled degrees"
)]

use std::fmt::Write as _;

use grafeo_common::types::{NodeId, Value};
use grafeo_core::graph::Direction;
use grafeo_core::graph::traits::GraphStore;
use grafeo_engine::GrafeoDB;
use grafeo_engine::query::optimizer::{PlannerRule, PlannerRules};
use grafeo_engine::session::Session;

const OUT: Direction = Direction::Outgoing;
const IN: Direction = Direction::Incoming;

fn people(db: &GrafeoDB, label: &str, count: usize) -> Vec<NodeId> {
    (0..count)
        .map(|key| {
            db.create_node_with_props(&[label], [("k", Value::Int64(key as i64))])
                .unwrap()
        })
        .collect()
}

fn sampled(db: &GrafeoDB, label: &str, kind: &str, direction: Direction) -> f64 {
    db.store()
        .unwrap()
        .sampled_degree(label, kind, direction)
        .unwrap_or_else(|| panic!("no sample for {label} {kind} {direction:?}"))
}

fn close(got: f64, want: f64, within: f64, what: &str) {
    assert!(
        (got - want).abs() <= want.abs() * within,
        "{what}: sampled {got}, true {want}"
    );
}

/// A type that fans out one way: each of 1,000 posts has one creator, each of 100 people ten
/// posts. The count of edges over a label is blind to the direction; a sample is not.
#[test]
fn the_direction_shows() {
    let db = GrafeoDB::new_in_memory();
    let posts = people(&db, "Post", 1_000);
    let persons = people(&db, "Person", 100);
    for (index, post) in posts.iter().enumerate() {
        db.create_edge(*post, persons[index % 100], "CREATED_BY")
            .unwrap();
    }
    close(
        sampled(&db, "Post", "CREATED_BY", OUT),
        1.0,
        0.0,
        "Post out",
    );
    close(sampled(&db, "Post", "CREATED_BY", IN), 0.0, 0.0, "Post in");
    close(
        sampled(&db, "Person", "CREATED_BY", IN),
        10.0,
        0.0,
        "Person in",
    );
    close(
        sampled(&db, "Person", "CREATED_BY", OUT),
        0.0,
        0.0,
        "Person out",
    );
}

/// Labels that share a type each get their own degree: posts have 2 tags and comments 6, over
/// 2,000 edges a count would share out alike.
#[test]
fn a_label_sharing_a_type_gets_its_own_degree() {
    let db = GrafeoDB::new_in_memory();
    let tags = people(&db, "Tag", 8);
    let posts = people(&db, "Post", 1_000);
    let comments = people(&db, "Comment", 1_000);
    for (index, post) in posts.iter().enumerate() {
        for step in 0..2 {
            db.create_edge(*post, tags[(index + step) % 8], "HAS_TAG")
                .unwrap();
        }
    }
    for (index, comment) in comments.iter().enumerate() {
        for step in 0..6 {
            db.create_edge(*comment, tags[(index + step) % 8], "HAS_TAG")
                .unwrap();
        }
    }
    close(sampled(&db, "Post", "HAS_TAG", OUT), 2.0, 0.0, "Post");
    close(sampled(&db, "Comment", "HAS_TAG", OUT), 6.0, 0.0, "Comment");
    close(
        sampled(&db, "Tag", "HAS_TAG", IN),
        1000.0,
        0.0,
        "Tag, sampled whole",
    );
}

/// The mean is over all sampled nodes, including those with no edge; a label of at most 256
/// nodes is read whole, so it is exact.
#[test]
fn a_small_label_is_exact_and_counts_the_nodes_without_edges() {
    let db = GrafeoDB::new_in_memory();
    let nodes = people(&db, "Person", 200);
    for node in nodes.iter().take(20) {
        for other in nodes.iter().take(5) {
            db.create_edge(*node, *other, "KNOWS").unwrap();
        }
    }
    assert_eq!(sampled(&db, "Person", "KNOWS", OUT), 100.0 / 200.0);
    assert_eq!(sampled(&db, "Person", "KNOWS", IN), 100.0 / 200.0);
}

/// A load order that repeats does not line up with the ranks: one node in every P has all the
/// edges, and the mean stays near the truth where evenly spaced ranks read 0 or twice it.
#[test]
fn a_periodic_load_does_not_line_up_with_the_sample() {
    for period in [2_usize, 4, 16, 39, 64, 128] {
        let db = GrafeoDB::new_in_memory();
        let nodes = people(&db, "Person", 32_768);
        let sink = people(&db, "Sink", 1)[0];
        for (index, node) in nodes.iter().enumerate() {
            if index % period == 0 {
                db.create_edge(*node, sink, "E").unwrap();
            }
        }
        let truth = (nodes.len().div_ceil(period)) as f64 / nodes.len() as f64;
        let got = sampled(&db, "Person", "E", OUT);
        assert!(
            got > truth * 0.4 && got < truth * 2.5,
            "period {period}: sampled {got}, true {truth}"
        );
    }
}

/// The largest hubs load first: a sample that starts at rank 0 always holds the largest, and
/// read 30 to 100 times the mean in a simulation.
#[test]
fn hubs_loaded_first_do_not_swamp_the_sample() {
    let db = GrafeoDB::new_in_memory();
    let nodes = people(&db, "Person", 10_000);
    let sink = people(&db, "Sink", 1)[0];
    let mut total = 0_u64;
    for (rank, node) in nodes.iter().enumerate() {
        let degree = (3_000 / (rank + 1)).max(1);
        total += degree as u64;
        for _ in 0..degree {
            db.create_edge(*node, sink, "E").unwrap();
        }
    }
    let truth = total as f64 / nodes.len() as f64;
    let got = sampled(&db, "Person", "E", OUT);
    assert!(got < truth * 3.0, "sampled {got}, true {truth}");
}

/// A mean over a skewed label: this reports how far off it is rather than pinning it; the
/// numbers decide if a mean holds. The answers stay right whatever the estimate.
#[test]
fn a_mean_over_a_skewed_label_is_reported_and_answers_stay_right() {
    for hubs_first in [true, false] {
        let db = GrafeoDB::new_in_memory();
        let nodes = people(&db, "Hubby", 4_000);
        let others = people(&db, "Leaf", 500);
        let mut total = 0_u64;
        for index in 0..nodes.len() {
            let rank = if hubs_first {
                index
            } else {
                // Spread the large ones through the ids, always the same way.
                (index * 1_021) % nodes.len()
            };
            let degree = 1_500 / (rank + 1);
            total += degree as u64;
            for step in 0..degree {
                db.create_edge(nodes[index], others[step % 500], "E")
                    .unwrap();
            }
        }
        let truth = total as f64 / nodes.len() as f64;
        let got = sampled(&db, "Hubby", "E", OUT);
        eprintln!(
            "skewed, hubs first {hubs_first}: sampled {got:.3}, true {truth:.3}, ratio {:.2}",
            got / truth
        );
        let query = "MATCH (a:Hubby)-[:E]->(l:Leaf), (b:Hubby)-[:E]->(l) WHERE a.k = 1 AND b.k < 3 RETURN count(*)";
        let usual = db.session();
        let mut off = db.session();
        off.set_planner_rules(PlannerRules::all().with(PlannerRule::JoinOrder, false));
        assert_eq!(
            usual.execute(query).unwrap().rows(),
            off.execute(query).unwrap().rows()
        );
    }
}

/// No uncommitted write counts: the sample is the latest committed state's, whatever an open
/// transaction holds, so the plan does not depend on which session plans first.
#[test]
fn no_uncommitted_write_counts() {
    let db = GrafeoDB::new_in_memory();
    let nodes = people(&db, "Person", 100);
    for node in &nodes {
        db.create_edge(*node, nodes[0], "KNOWS").unwrap();
    }
    let before = sampled(&db, "Person", "KNOWS", OUT);
    let mut session = db.session();
    session.begin_transaction().unwrap();
    session
        .execute("MATCH (a:Person), (b:Person {k: 1}) INSERT (a)-[:KNOWS]->(b), (a)-[:KNOWS]->(b)")
        .unwrap();
    session
        .execute("MATCH (:Person)-[e:KNOWS]->(:Person {k: 0}) DELETE e")
        .unwrap();
    session.execute("INSERT (:Person {k: 5000})").unwrap();
    assert_eq!(sampled(&db, "Person", "KNOWS", OUT), before);
    session.rollback().unwrap();
    assert_eq!(sampled(&db, "Person", "KNOWS", OUT), before);
}

/// Same data, a transaction with writes planning first on one side: the plans of the comma
/// pattern stay the same on both.
#[test]
fn the_plan_does_not_depend_on_who_plans_first() {
    let query =
        "MATCH (p:Person)-[:KNOWS]->(t:Tag), (m:Post)-[:HAS]->(t), (m)-[:BY]->(p) RETURN count(*)";
    let build = || {
        let db = GrafeoDB::new_in_memory();
        let persons = people(&db, "Person", 400);
        let tags = people(&db, "Tag", 10);
        let posts = people(&db, "Post", 400);
        for (index, post) in posts.iter().enumerate() {
            db.create_edge(*post, tags[index % 10], "HAS").unwrap();
            db.create_edge(*post, persons[index], "BY").unwrap();
            db.create_edge(persons[index], tags[(index + 1) % 10], "KNOWS")
                .unwrap();
            db.create_edge(persons[index], tags[(index + 2) % 10], "KNOWS")
                .unwrap();
        }
        db
    };
    let explain = |session: &Session| {
        format!(
            "{:?}",
            session.execute(&format!("EXPLAIN {query}")).unwrap().rows()
        )
    };
    let (quiet, busy) = (build(), build());
    let mut writer = busy.session();
    writer.begin_transaction().unwrap();
    writer
        .execute("MATCH (a:Person), (b:Tag) WHERE a.k < 300 INSERT (a)-[:KNOWS]->(b)")
        .unwrap();
    let in_transaction = explain(&writer);
    writer.rollback().unwrap();
    assert_eq!(explain(&busy.session()), explain(&quiet.session()));
    assert_eq!(in_transaction, explain(&quiet.session()));
}

/// The sampled values and the plan, printed by a child process: the ranks are integer
/// arithmetic, and a map's order changes per process. The values are pinned to the bit.
fn degrees_and_plan() -> String {
    let db = GrafeoDB::new_in_memory();
    let persons = people(&db, "Person", 3_000);
    let tags = people(&db, "Tag", 20);
    let posts = people(&db, "Post", 2_000);
    for (index, person) in persons.iter().enumerate() {
        for step in 0..=index % 7 {
            db.create_edge(*person, tags[(index * 3 + step) % 20], "KNOWS")
                .unwrap();
        }
    }
    for (index, post) in posts.iter().enumerate() {
        db.create_edge(*post, persons[index % 3_000], "BY").unwrap();
        for step in 0..=index % 3 {
            db.create_edge(*post, tags[(index + step) % 20], "HAS")
                .unwrap();
        }
    }
    let mut out = String::new();
    for (label, kind, direction) in [
        ("Person", "KNOWS", OUT),
        ("Post", "BY", OUT),
        ("Post", "HAS", OUT),
        ("Person", "BY", IN),
        ("Tag", "HAS", IN),
    ] {
        writeln!(
            out,
            "{label} {kind} {direction:?} {:016x}",
            sampled(&db, label, kind, direction).to_bits()
        )
        .unwrap();
    }
    let query =
        "MATCH (p:Person)-[:KNOWS]->(t:Tag), (m:Post)-[:HAS]->(t), (m)-[:BY]->(p) RETURN count(*)";
    let plan = db.session().execute(&format!("EXPLAIN {query}")).unwrap();
    write!(out, "{:?}", plan.rows()).unwrap();
    out
}

const CHILD: &str = "GRAFEO_SAMPLED_DEGREES_CHILD";

#[test]
fn the_sample_and_the_plan_are_the_same_in_every_process() {
    if std::env::var_os(CHILD).is_some() {
        print!("BEGIN\n{}\nEND\n", degrees_and_plan());
        return;
    }
    let here = degrees_and_plan();
    assert_eq!(degrees_and_plan(), here);
    for _ in 0..4 {
        let output = grafeo_common::testing::child_process::output(
            grafeo_common::child_test!("the_sample_and_the_plan_are_the_same_in_every_process")
                .env(CHILD, "1"),
        )
        .unwrap();
        assert!(output.status.success());
        let out = String::from_utf8_lossy(&output.stdout).to_string();
        let begin = out.find("BEGIN\n").unwrap() + "BEGIN\n".len();
        let end = out.find("\nEND").unwrap();
        assert_eq!(&out[begin..end], here);
    }
}

/// The ranks the design decides, in integer arithmetic so that every platform agrees: for i
/// from 1 to 256, the high bits of `i` times the golden ratio in 64 bits, scaled to the count.
fn ranks(count: u64) -> Vec<u64> {
    const GOLDEN: u64 = 0x9E37_79B9_7F4A_7C15;
    let mut ranks: Vec<u64> = (1..=256_u64)
        .map(|i| {
            let fraction = i.wrapping_mul(GOLDEN);
            ((u128::from(fraction) * u128::from(count)) >> 64) as u64
        })
        .collect();
    ranks.sort_unstable();
    ranks.dedup();
    ranks
}

/// The sample is exactly these nodes: each has its own number of edges, so the mean tells which
/// were read, to the bit. A change of the sequence, or of a float that rounds otherwise on
/// another platform, shows here.
#[test]
fn the_ranks_are_the_decided_ones() {
    let db = GrafeoDB::new_in_memory();
    let nodes = people(&db, "Person", 1_000);
    let sink = people(&db, "Sink", 1)[0];
    let mut weight = 1_u64;
    let mut weights = Vec::new();
    for (rank, node) in nodes.iter().enumerate() {
        weight = (weight * 31 + rank as u64) % 11;
        weights.push(weight);
        for _ in 0..weight {
            db.create_edge(*node, sink, "E").unwrap();
        }
    }
    let ranks = ranks(nodes.len() as u64);
    let sum: u64 = ranks.iter().map(|rank| weights[*rank as usize]).sum();
    let expected = sum as f64 / ranks.len() as f64;
    assert_eq!(
        sampled(&db, "Person", "E", OUT).to_bits(),
        expected.to_bits(),
        "the sample read other nodes"
    );
}

/// The planner's degree: `min(sampled, derived)`, where derived is the type's edges over the
/// label's nodes, twice that for both directions. So an over-read is bounded.
#[test]
fn the_degree_is_clamped_by_the_derived_one() {
    use grafeo_engine::query::optimizer::hop_degree;

    let db = GrafeoDB::new_in_memory();
    // The first nodes hold every edge, and golden ratio ranks may find more than their share.
    let nodes = people(&db, "Person", 2_000);
    let sink = people(&db, "Sink", 1)[0];
    for (rank, node) in nodes.iter().enumerate() {
        for _ in 0..3_000 / (rank + 1) {
            db.create_edge(*node, sink, "E").unwrap();
        }
    }
    let store = db.store().unwrap();
    let edges = db
        .session()
        .execute("MATCH (:Person)-[:E]->() RETURN count(*)")
        .unwrap();
    let Value::Int64(edges) = edges.rows()[0][0] else {
        panic!("a count");
    };
    let derived = edges as f64 / nodes.len() as f64;
    let out = hop_degree(&*store, &["Person"], &["E"], OUT);
    assert!(out <= derived, "{out} over {derived}");
    assert_eq!(out, sampled(&db, "Person", "E", OUT).min(derived));
    let both = hop_degree(&*store, &["Person"], &["E"], Direction::Both);
    assert!(both <= derived * 2.0, "{both} over {}", derived * 2.0);
}

/// `(a:A:B)` samples the smaller label; several types add up per type.
#[test]
fn several_labels_take_the_smaller_and_several_types_add_up() {
    use grafeo_engine::query::optimizer::hop_degree;

    let db = GrafeoDB::new_in_memory();
    let many = people(&db, "Many", 300);
    let both: Vec<NodeId> = (0..50)
        .map(|_| db.create_node(&["Many", "Few"]).unwrap())
        .collect();
    let sink = people(&db, "Sink", 1)[0];
    for node in &many {
        db.create_edge(*node, sink, "E").unwrap();
    }
    for node in &both {
        for _ in 0..4 {
            db.create_edge(*node, sink, "E").unwrap();
        }
        db.create_edge(*node, sink, "F").unwrap();
    }
    let store = db.store().unwrap();
    assert_eq!(hop_degree(&*store, &["Few"], &["E"], OUT), 4.0);
    assert_eq!(hop_degree(&*store, &["Many", "Few"], &["E"], OUT), 4.0);
    assert_eq!(hop_degree(&*store, &["Few", "Many"], &["E"], OUT), 4.0);
    assert_eq!(hop_degree(&*store, &["Few"], &["E", "F"], OUT), 5.0);
    assert_eq!(hop_degree(&*store, &["Few"], &["F"], OUT), 1.0);
}

/// Common queries do not pay: a plain query samples nothing, and only the join search does.
#[test]
fn a_plain_query_samples_nothing() {
    let db = GrafeoDB::new_in_memory();
    db.create_property_index("k").unwrap();
    let nodes = people(&db, "Person", 500);
    for pair in nodes.windows(2) {
        db.create_edge(pair[0], pair[1], "KNOWS").unwrap();
    }
    let session = db.session();
    let taken = || db.store().unwrap().degree_samples_taken();
    for query in [
        "MATCH (n:Person) RETURN n.k",
        "MATCH (n:Person {k: 1}) RETURN n.k",
        "MATCH (n:Person) WHERE n.k > 490 RETURN n.k",
        "MATCH (a:Person)-[:KNOWS]->(b) RETURN a.k, b.k",
        "MATCH (a:Person {k: 1})-[:KNOWS]->(b)-[:KNOWS]->(c) RETURN c.k",
        "MATCH (a:Person)-[:KNOWS]->(b)-[:KNOWS]->(a) RETURN a.k",
        "MATCH (a:Person) RETURN count(*)",
        "MATCH (a:Person)-[:KNOWS]->(b) RETURN a.k, count(*)",
        "INSERT (:Person {k: 1000})",
        "UNWIND [1, 2] AS x MATCH (a:Person {k: x}) RETURN a.k",
    ] {
        session.execute(query).unwrap();
        assert_eq!(taken(), 0, "{query} took a sample");
    }
    let comma =
        "MATCH (a:Person)-[:KNOWS]->(b), (b)-[:KNOWS]->(c), (c)-[:KNOWS]->(a) RETURN count(*)";
    session.execute(comma).unwrap();
    let after_search = taken();
    assert!(after_search > 0, "the search took no sample");
    // A second planning of another text over the same degrees reads the cache.
    session
        .execute("MATCH (a:Person)-[:KNOWS]->(b), (b)-[:KNOWS]->(c), (c)-[:KNOWS]->(a) RETURN a.k")
        .unwrap();
    assert_eq!(taken(), after_search, "the same degrees were sampled again");
}

fn add_edges(db: &GrafeoDB, nodes: &[NodeId], count: usize, kind: &str) {
    for index in 0..count {
        db.create_edge(
            nodes[index % nodes.len()],
            nodes[(index * 7) % nodes.len()],
            kind,
        )
        .unwrap();
    }
}

/// A larger label keeps its sample until a count moves by the larger of 10 percent and 16.
#[test]
fn a_large_label_resamples_after_a_large_change_only() {
    let db = GrafeoDB::new_in_memory();
    let nodes = people(&db, "Person", 2_000);
    add_edges(&db, &nodes, 2_000, "KNOWS");
    let first = sampled(&db, "Person", "KNOWS", OUT);
    let taken = || db.store().unwrap().degree_samples_taken();
    let samples = taken();

    // Under the bound of 200 edges: the cached value, to the bit.
    add_edges(&db, &nodes, 150, "KNOWS");
    assert_eq!(
        sampled(&db, "Person", "KNOWS", OUT).to_bits(),
        first.to_bits()
    );
    assert_eq!(taken(), samples);

    // Past it: a new sample that sees them.
    add_edges(&db, &nodes, 2_000, "KNOWS");
    let second = sampled(&db, "Person", "KNOWS", OUT);
    assert!(second > first * 1.5, "{first} then {second}");
    assert_eq!(taken(), samples + 1);

    // The label's count moves it too.
    people(&db, "Person", 400);
    let third = sampled(&db, "Person", "KNOWS", OUT);
    assert!(third < second, "{second} then {third}");
}

/// Over a type with few edges the bound is 16, not 10 percent of nearly nothing or a wait for
/// 10 percent of the label.
#[test]
fn a_new_type_on_a_large_label_resamples_after_sixteen_writes() {
    let db = GrafeoDB::new_in_memory();
    let nodes = people(&db, "Person", 2_000);
    assert_eq!(sampled(&db, "Person", "RARE", OUT), 0.0);
    let samples = db.store().unwrap().degree_samples_taken();
    add_edges(&db, &nodes, 15, "RARE");
    assert_eq!(sampled(&db, "Person", "RARE", OUT), 0.0);
    assert_eq!(db.store().unwrap().degree_samples_taken(), samples);
    add_edges(&db, &nodes, 2, "RARE");
    let _ = sampled(&db, "Person", "RARE", OUT);
    assert_eq!(db.store().unwrap().degree_samples_taken(), samples + 1);
}

/// A label of at most 256 nodes resamples on any change, so a small graph stays exact.
#[test]
fn a_small_label_resamples_on_any_change() {
    let db = GrafeoDB::new_in_memory();
    let nodes = people(&db, "Person", 100);
    add_edges(&db, &nodes, 100, "KNOWS");
    assert_eq!(sampled(&db, "Person", "KNOWS", OUT), 1.0);
    db.create_edge(nodes[0], nodes[1], "KNOWS").unwrap();
    assert_eq!(sampled(&db, "Person", "KNOWS", OUT), 101.0 / 100.0);
    let edge = db
        .session()
        .execute("MATCH (n:Person {k: 0})-[e:KNOWS]->() RETURN id(e) LIMIT 1")
        .unwrap();
    assert_eq!(edge.row_count(), 1);
    db.session()
        .execute("MATCH (n:Person {k: 2})-[e:KNOWS]->() DELETE e")
        .unwrap();
    assert!(sampled(&db, "Person", "KNOWS", OUT) < 101.0 / 100.0);
    people(&db, "Person", 100);
    assert!(sampled(&db, "Person", "KNOWS", OUT) < 0.6);
}

/// What is gone does not count, and no error reaches the planner: a label or a type nobody
/// has plans as usual.
#[test]
fn the_sample_reads_the_latest_committed_state_and_never_fails() {
    let db = GrafeoDB::new_in_memory();
    let nodes = people(&db, "Person", 10);
    add_edges(&db, &nodes, 10, "KNOWS");
    assert_eq!(sampled(&db, "Person", "KNOWS", OUT), 1.0);
    db.session()
        .execute("MATCH (n:Person) WHERE n.k < 5 DETACH DELETE n")
        .unwrap();
    let left = db
        .session()
        .execute("MATCH (a:Person)-[e:KNOWS]->() RETURN count(*)")
        .unwrap();
    let Value::Int64(left) = left.rows()[0][0] else {
        panic!("a count");
    };
    assert_eq!(
        sampled(&db, "Person", "KNOWS", OUT),
        left as f64 / 5.0,
        "over the 5 nodes left"
    );
    let store = db.store().unwrap();
    assert!(
        store
            .sampled_degree("NoSuchLabel", "KNOWS", OUT)
            .is_none_or(|d| d == 0.0)
    );
    assert!(
        store
            .sampled_degree("Person", "NO_SUCH_TYPE", OUT)
            .is_none_or(|d| d == 0.0)
    );
    let session = db.session();
    session
        .execute("MATCH (a:Nobody)-[:KNOWS]->(b), (b)-[:NOTHING]->(a) RETURN count(*)")
        .unwrap();
}

/// A store that cannot sample says so, and the planner keeps the derived estimate.
#[test]
fn a_store_that_cannot_sample_says_none() {
    let store = grafeo_core::graph::traits::NullGraphStore;
    assert_eq!(store.sampled_degree("Person", "KNOWS", OUT), None);
}
