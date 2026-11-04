//! Degrees of reached ends, the second stage of a comma pattern's estimates (#714).
//!
//! Tests first, from the design: `GraphStore::sampled_reached_degree(start_label, via_types,
//! via_direction, on_types, on_direction)` samples up to 256 start nodes at the first stage's
//! ranks, takes their lists as one sequence of T edges in rank order, keeps every edge when T is
//! at most 256 and otherwise the positions at 256 golden-ratio ranks of T, and answers the
//! plain mean of the next hop's count at the kept ends. A mean of 0 stands only when every edge
//! was read; on a partial sample it is `None`, so the first stage answers. It does not compile
//! until that method exists. Rows never depend on it.
//!
//! ```bash
//! cargo test -p grafeo-engine --all-features --test main reached_degrees::
//! ```

#![cfg(all(feature = "gql", feature = "cypher"))]
#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss,
    deprecated,
    reason = "test graphs of a few thousand nodes, and the raw store's sampled degrees"
)]

use grafeo_common::types::{ArcStr, NodeId, ReadView};
use grafeo_core::graph::Direction;
use grafeo_core::graph::traits::GraphStore;
use grafeo_engine::GrafeoDB;
use grafeo_engine::query::optimizer::{PlannerRule, PlannerRules};
use grafeo_engine::session::Session;

const OUT: Direction = Direction::Outgoing;
const IN: Direction = Direction::Incoming;
const GOLDEN: u64 = 0x9E37_79B9_7F4A_7C15;

fn nodes(db: &GrafeoDB, label: &str, count: usize) -> Vec<NodeId> {
    (0..count)
        .map(|_| db.create_node(&[label]).unwrap())
        .collect()
}

/// `size` golden-ratio ranks of `count`, sorted and without repeats; all of them when they fit.
fn ranks(count: u64, size: u64) -> Vec<u64> {
    if count <= size {
        return (0..count).collect();
    }
    let mut ranks: Vec<u64> = (1..=size)
        .map(|i| ((u128::from(i.wrapping_mul(GOLDEN)) * u128::from(count)) >> 64) as u64)
        .collect();
    ranks.sort_unstable();
    ranks.dedup();
    ranks
}

fn reached(
    db: &GrafeoDB,
    start: Option<&str>,
    via: &[&str],
    via_direction: Direction,
    on: &[&str],
    on_direction: Direction,
) -> Option<f64> {
    db.store()
        .unwrap()
        .sampled_reached_degree(start, via, via_direction, on, on_direction)
}

fn taken(db: &GrafeoDB) -> u64 {
    db.store().unwrap().degree_samples_taken()
}

// ---------------------------------------------------------------------------
// What it reads
// ---------------------------------------------------------------------------

/// Few starts, every edge read: the mean is exact, and an end reached twice counts twice.
#[test]
fn the_mean_is_over_the_edges_with_repeats() {
    let db = GrafeoDB::new_in_memory();
    let starts = nodes(&db, "Start", 3);
    let ends = nodes(&db, "End", 2);
    // Two edges reach the first end and one the second.
    db.create_edge(starts[0], ends[0], "VIA").unwrap();
    db.create_edge(starts[1], ends[0], "VIA").unwrap();
    db.create_edge(starts[2], ends[1], "VIA").unwrap();
    let sink = nodes(&db, "Sink", 1)[0];
    for _ in 0..9 {
        db.create_edge(ends[0], sink, "ON").unwrap();
    }
    for _ in 0..3 {
        db.create_edge(ends[1], sink, "ON").unwrap();
    }
    assert_eq!(
        reached(&db, Some("Start"), &["VIA"], OUT, &["ON"], OUT),
        Some((9.0 + 9.0 + 3.0) / 3.0)
    );
}

/// The hub shape's own numbers, whether or not the middle has a label: the sides reach hubs.
#[test]
fn a_reached_hub_shows_where_the_label_mean_hides_it() {
    let db = GrafeoDB::new_in_memory();
    let ordinary = nodes(&db, "Middle", 2_000);
    let hubs = nodes(&db, "Middle", 4);
    let left = nodes(&db, "Left", 5);
    for (index, node) in left.iter().enumerate() {
        db.create_edge(*node, hubs[index % 4], "TO").unwrap();
    }
    for hub in &hubs {
        let others = nodes(&db, "Other", 1_000);
        for other in others {
            db.create_edge(other, *hub, "TO").unwrap();
        }
    }
    for node in &ordinary {
        let other = nodes(&db, "Other", 1)[0];
        db.create_edge(other, *node, "TO").unwrap();
    }
    let first_hub_has_two_sides = 1_002.0;
    let mean = reached(&db, Some("Left"), &["TO"], OUT, &["TO"], IN).unwrap();
    assert_eq!(
        mean,
        (first_hub_has_two_sides * 2.0 + 1_001.0 * 3.0) / 5.0,
        "the mean over the five reached hubs"
    );
    let store = db.store().unwrap();
    let of_the_label = store.sampled_degree("Middle", "TO", IN).unwrap();
    assert!(of_the_label < 10.0, "the label's mean is {of_the_label}");
}

/// The way each hop runs matters: a type read the other way counts other edges.
#[test]
fn the_directions_are_the_hops_own() {
    let db = GrafeoDB::new_in_memory();
    let starts = nodes(&db, "Start", 2);
    let middle = nodes(&db, "Middle", 1)[0];
    let far = nodes(&db, "Far", 5);
    db.create_edge(starts[0], middle, "VIA").unwrap();
    db.create_edge(starts[1], middle, "VIA").unwrap();
    for node in &far {
        db.create_edge(middle, *node, "ON").unwrap();
    }
    assert_eq!(
        reached(&db, Some("Start"), &["VIA"], OUT, &["ON"], OUT),
        Some(5.0)
    );
    assert_eq!(
        reached(&db, Some("Start"), &["VIA"], OUT, &["ON"], IN),
        Some(0.0)
    );
    assert_eq!(
        reached(&db, Some("Start"), &["VIA"], OUT, &["VIA"], IN),
        Some(2.0)
    );
    assert_eq!(
        reached(&db, Some("Far"), &["ON"], IN, &["VIA"], IN),
        Some(2.0)
    );
    assert_eq!(
        reached(&db, Some("Far"), &["ON"], IN, &["ON"], OUT),
        Some(5.0)
    );
    // Both ways reach the middle and back out to the start's own edge list.
    assert_eq!(
        reached(&db, Some("Start"), &["VIA"], Direction::Both, &["ON"], OUT),
        Some(5.0)
    );
}

/// Several types on either hop add up, each edge kept once.
#[test]
fn several_types_add_up() {
    let db = GrafeoDB::new_in_memory();
    let start = nodes(&db, "Start", 1)[0];
    let ends = nodes(&db, "End", 2);
    db.create_edge(start, ends[0], "ONE").unwrap();
    db.create_edge(start, ends[1], "TWO").unwrap();
    let sink = nodes(&db, "Sink", 1)[0];
    for _ in 0..2 {
        db.create_edge(ends[0], sink, "P").unwrap();
    }
    for _ in 0..4 {
        db.create_edge(ends[1], sink, "Q").unwrap();
    }
    assert_eq!(
        reached(&db, Some("Start"), &["ONE", "TWO"], OUT, &["P", "Q"], OUT),
        Some(3.0)
    );
    assert_eq!(
        reached(&db, Some("Start"), &["ONE"], OUT, &["P", "Q"], OUT),
        Some(2.0)
    );
    assert_eq!(
        reached(&db, Some("Start"), &["TWO"], OUT, &["P"], OUT),
        Some(0.0)
    );
    // Written in another order, one entry and one answer.
    let before = taken(&db);
    assert_eq!(
        reached(&db, Some("Start"), &["TWO", "ONE"], OUT, &["Q", "P"], OUT),
        Some(3.0)
    );
    assert_eq!(
        taken(&db),
        before,
        "the order of the type lists is a second entry"
    );
}

/// With no label the starts are the ranks of all nodes.
#[test]
fn no_label_starts_from_all_nodes() {
    let db = GrafeoDB::new_in_memory();
    let everyone = nodes(&db, "Any", 1_000);
    let sink = nodes(&db, "Sink", 1)[0];
    // Each node's edge reaches itself, each with its own number of edges on, so the mean tells
    // which nodes were read. The sink has none of those.
    let mut weights = Vec::new();
    let mut weight = 1_u64;
    for (rank, node) in everyone.iter().enumerate() {
        db.create_edge(*node, *node, "VIA").unwrap();
        weight = (weight * 31 + rank as u64) % 11;
        weights.push(weight);
        for _ in 0..weight {
            db.create_edge(*node, sink, "ON").unwrap();
        }
    }
    let all = (everyone.len() + 1) as u64;
    // The sink is the last node, with no VIA edge: it adds a start and no list.
    let starts = ranks(all, 256);
    let kept: Vec<u64> = starts
        .iter()
        .copied()
        .filter(|rank| *rank < 1_000)
        .collect();
    let sum: u64 = kept.iter().map(|rank| weights[*rank as usize]).sum();
    let want = sum as f64 / kept.len() as f64;
    assert_eq!(
        reached(&db, None, &["VIA"], OUT, &["ON"], OUT).map(f64::to_bits),
        Some(want.to_bits()),
        "the sample read other nodes"
    );
}

/// The sample is exactly the decided positions: every end has its own count, so the mean tells
/// which were kept, to the bit. Starts at the first stage's ranks, their lists in rank order,
/// each in its `edges` order, 256 golden-ratio ranks of the whole sequence.
#[test]
fn the_positions_are_the_decided_ones() {
    let db = GrafeoDB::new_in_memory();
    let starts = nodes(&db, "Start", 600);
    let ends = nodes(&db, "End", 500);
    let sink = nodes(&db, "Sink", 1)[0];
    let mut state = 3_u64;
    let mut next = move |bound: u64| {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        (state >> 33) % bound
    };
    // Lists of different lengths, so that a position is not a function of the start alone.
    for start in &starts {
        for _ in 0..=next(30) {
            db.create_edge(*start, ends[next(500) as usize], "VIA")
                .unwrap();
        }
    }
    let mut count_at = vec![0_u64; 500];
    for (index, end) in ends.iter().enumerate() {
        let count = next(17);
        count_at[index] = count;
        for _ in 0..count {
            db.create_edge(*end, sink, "ON").unwrap();
        }
    }
    let store = db.store().unwrap();
    let via = vec![ArcStr::from("VIA")];
    let view = ReadView::latest();
    let mut sequence = Vec::new();
    for rank in ranks(starts.len() as u64, 256) {
        let list = store.edges(starts[rank as usize], OUT, &via, view).unwrap();
        sequence.extend(list.iter().map(|(end, _)| *end));
    }
    let total = sequence.len() as u64;
    assert!(total > 256, "the fixture must read a part: {total} edges");
    let positions = ranks(total, 256);
    let index_of = |end: NodeId| ends.iter().position(|node| *node == end).unwrap();
    let sum: u64 = positions
        .iter()
        .map(|position| count_at[index_of(sequence[*position as usize])])
        .sum();
    let want = sum as f64 / positions.len() as f64;
    assert_eq!(
        reached(&db, Some("Start"), &["VIA"], OUT, &["ON"], OUT).map(f64::to_bits),
        Some(want.to_bits()),
        "the sample kept other positions"
    );
}

/// All edges up to 256, at the boundary and one short of it.
#[test]
fn up_to_256_edges_are_all_read() {
    for total in [1, 255, 256] {
        let db = GrafeoDB::new_in_memory();
        let start = nodes(&db, "Start", 1)[0];
        let ends = nodes(&db, "End", total);
        let sink = nodes(&db, "Sink", 1)[0];
        let mut want = 0_u64;
        for (index, end) in ends.iter().enumerate() {
            db.create_edge(start, *end, "VIA").unwrap();
            for _ in 0..(index % 3) {
                db.create_edge(*end, sink, "ON").unwrap();
                want += 1;
            }
        }
        assert_eq!(
            reached(&db, Some("Start"), &["VIA"], OUT, &["ON"], OUT),
            Some(want as f64 / total as f64),
            "{total} edges"
        );
    }
}

/// A long first list takes its share by length and no more: the old budget read 1 here, where
/// half the edges reach a hub.
#[test]
fn a_long_list_first_does_not_eat_the_sample() {
    let db = GrafeoDB::new_in_memory();
    let long = db.create_node(&["Start"]).unwrap();
    let starts = nodes(&db, "Start", 250);
    let ordinary = nodes(&db, "End", 5_000);
    let hubs = nodes(&db, "End", 4);
    for end in &ordinary {
        db.create_edge(long, *end, "VIA").unwrap();
    }
    for (index, start) in starts.iter().enumerate() {
        for _ in 0..20 {
            db.create_edge(*start, hubs[index % 4], "VIA").unwrap();
        }
    }
    let sink = nodes(&db, "Sink", 1)[0];
    for hub in &hubs {
        for _ in 0..3_000 {
            db.create_edge(*hub, sink, "ON").unwrap();
        }
    }
    let mean = reached(&db, Some("Start"), &["VIA"], OUT, &["ON"], OUT).unwrap();
    assert!(
        (1_200.0..=1_800.0).contains(&mean),
        "about half of 3,000 reached a hub: {mean}"
    );
}

/// A regular load whose period divides the step of evenly stepped positions reads one end only.
/// These positions do not line up: with one end in a hundred holding every edge, 256 positions
/// meet it about 2.6 times, so they read within a factor of 3 or so of the truth wherever it sits,
/// never all of the edges or none.
#[test]
fn a_periodic_load_does_not_line_up_with_the_positions() {
    for hub_at in [0_usize, 1, 2, 50, 99] {
        let db = GrafeoDB::new_in_memory();
        let starts = nodes(&db, "Start", 256);
        let ends = nodes(&db, "End", 100);
        let mut next = 0_usize;
        for start in &starts {
            for _ in 0..400 {
                db.create_edge(*start, ends[next % 100], "VIA").unwrap();
                next += 1;
            }
        }
        let sink = nodes(&db, "Sink", 1)[0];
        for _ in 0..1_000 {
            db.create_edge(ends[hub_at], sink, "ON").unwrap();
        }
        let truth = 10.0;
        let mean = reached(&db, Some("Start"), &["VIA"], OUT, &["ON"], OUT).unwrap();
        assert!(
            (truth / 4.0..=truth * 3.0).contains(&mean),
            "hub at {hub_at}: read {mean}, true {truth}"
        );
    }
}

// ---------------------------------------------------------------------------
// A zero
// ---------------------------------------------------------------------------

/// Every edge read and none has the next hop's edges: a true 0.
/// A budget holds whatever the lists' lengths: a list past its share is counted as one read, and
/// only its first edges are read.
#[test]
fn a_spent_budget_reads_little_of_long_lists() {
    let db = GrafeoDB::new_in_memory();
    let ends = nodes(&db, "End", 64);
    for start in nodes(&db, "Start", 256) {
        for index in 0..1_000 {
            db.create_edge(start, ends[index % ends.len()], "VIA")
                .unwrap();
        }
    }
    let store = db.store().unwrap();
    let before = store.degree_sample_reads();
    let mut budget = 1_u64;
    store.sampled_reached_degree_within(Some("Start"), &["VIA"], OUT, &["ON"], OUT, &mut budget);
    let reads = store.degree_sample_reads() - before;
    assert!(reads > 0, "the sample read nothing");
    assert!(reads < 1_024, "{reads} reads of 256,000 edges");
}

#[test]
fn a_zero_stands_when_every_edge_was_read() {
    let db = GrafeoDB::new_in_memory();
    let starts = nodes(&db, "Start", 100);
    let ends = nodes(&db, "End", 10);
    for (index, start) in starts.iter().enumerate() {
        db.create_edge(*start, ends[index % 10], "VIA").unwrap();
    }
    db.create_edge(starts[0], starts[1], "ON").unwrap();
    assert_eq!(
        reached(&db, Some("Start"), &["VIA"], OUT, &["ON"], OUT),
        Some(0.0)
    );
}

/// A partial sample that reads 0 says nothing, so the first stage answers.
#[test]
fn a_zero_of_a_partial_sample_is_no_answer() {
    let db = GrafeoDB::new_in_memory();
    let start = nodes(&db, "Start", 1)[0];
    let ends = nodes(&db, "End", 2_000);
    for end in &ends {
        db.create_edge(start, *end, "VIA").unwrap();
    }
    assert_eq!(
        reached(&db, Some("Start"), &["VIA"], OUT, &["ON"], OUT),
        None
    );
}

/// The pair on medium SNB where this happens: a country's located-in edges, then work-at into
/// what they reach. A few ends hold every edge of the next hop, the kept positions miss them, and
/// the true mean is far from 0. That read is no answer, never a 0 that makes the hop look free.
#[test]
fn rare_ends_the_positions_miss_are_no_answer_not_zero() {
    let db = GrafeoDB::new_in_memory();
    let start = nodes(&db, "Start", 1)[0];
    let ends = nodes(&db, "End", 30_000);
    for end in &ends {
        db.create_edge(start, *end, "VIA").unwrap();
    }
    let store = db.store().unwrap();
    let via = vec![ArcStr::from("VIA")];
    let list = store.edges(start, OUT, &via, ReadView::latest()).unwrap();
    let kept: std::collections::BTreeSet<u64> = ranks(list.len() as u64, 256).into_iter().collect();
    let sink = nodes(&db, "Sink", 1)[0];
    let missed: Vec<NodeId> = list
        .iter()
        .enumerate()
        .filter(|(position, _)| !kept.contains(&(*position as u64)))
        .take(20)
        .map(|(_, (end, _))| *end)
        .collect();
    for end in &missed {
        for _ in 0..450 {
            db.create_edge(*end, sink, "ON").unwrap();
        }
    }
    assert_eq!(
        reached(&db, Some("Start"), &["VIA"], OUT, &["ON"], OUT),
        None
    );
}

/// Starts outside the ranks hold the only edges: no edge is reached, and that is no answer
/// because not every start was read. With every start read it is a true 0.
#[test]
fn no_edge_reached_is_a_zero_only_when_every_start_was_read() {
    let db = GrafeoDB::new_in_memory();
    let starts = nodes(&db, "Start", 2_000);
    let read: std::collections::BTreeSet<u64> = ranks(2_000, 256).into_iter().collect();
    let outside = (0..2_000).find(|rank| !read.contains(rank)).unwrap() as usize;
    let end = nodes(&db, "End", 1)[0];
    db.create_edge(starts[outside], end, "VIA").unwrap();
    assert_eq!(
        reached(&db, Some("Start"), &["VIA"], OUT, &["VIA"], IN),
        None
    );

    let small = GrafeoDB::new_in_memory();
    nodes(&small, "Start", 100);
    let others = nodes(&small, "Other", 2);
    small.create_edge(others[0], others[1], "VIA").unwrap();
    assert_eq!(
        reached(&small, Some("Start"), &["VIA"], OUT, &["VIA"], IN),
        Some(0.0)
    );
}

// ---------------------------------------------------------------------------
// The contract of the read
// ---------------------------------------------------------------------------

/// No uncommitted write counts, whoever holds it open and whichever way it goes.
#[test]
fn no_uncommitted_write_counts() {
    let db = GrafeoDB::new_in_memory();
    let starts = nodes(&db, "Start", 10);
    let middle = nodes(&db, "Middle", 2);
    for start in &starts {
        db.create_edge(*start, middle[0], "VIA").unwrap();
    }
    let sink = nodes(&db, "Sink", 1)[0];
    for _ in 0..3 {
        db.create_edge(middle[0], sink, "ON").unwrap();
    }
    let before = reached(&db, Some("Start"), &["VIA"], OUT, &["ON"], OUT);
    assert_eq!(before, Some(3.0));
    let mut session = db.session();
    session.begin_transaction().unwrap();
    session
        .execute("MATCH (s:Start), (m:Middle) WHERE id(m) <> -1 INSERT (s)-[:VIA]->(m)")
        .unwrap();
    session
        .execute("MATCH (m:Middle)-[e:ON]->() DELETE e")
        .unwrap();
    session.execute("INSERT (:Start)").unwrap();
    assert_eq!(
        reached(&db, Some("Start"), &["VIA"], OUT, &["ON"], OUT),
        before
    );
    session.rollback().unwrap();
    assert_eq!(
        reached(&db, Some("Start"), &["VIA"], OUT, &["ON"], OUT),
        before
    );
}

/// Deleted nodes and edges are not read, and what is gone answers without an error.
#[test]
fn it_reads_the_latest_committed_state_and_never_fails() {
    let db = GrafeoDB::new_in_memory();
    let starts = nodes(&db, "Start", 10);
    let middle = nodes(&db, "Middle", 2);
    let sink = nodes(&db, "Sink", 1)[0];
    for (index, start) in starts.iter().enumerate() {
        db.create_edge(*start, middle[index % 2], "VIA").unwrap();
    }
    for _ in 0..4 {
        db.create_edge(middle[0], sink, "ON").unwrap();
    }
    assert_eq!(
        reached(&db, Some("Start"), &["VIA"], OUT, &["ON"], OUT),
        Some(2.0)
    );
    db.session()
        .execute(&format!(
            "MATCH (m) WHERE id(m) = {} DETACH DELETE m",
            middle[1].as_u64()
        ))
        .unwrap();
    assert_eq!(
        reached(&db, Some("Start"), &["VIA"], OUT, &["ON"], OUT),
        Some(4.0),
        "only the edges to the middle that is left"
    );
    for answer in [
        reached(&db, Some("NoSuchLabel"), &["VIA"], OUT, &["ON"], OUT),
        reached(&db, Some("Start"), &["NO_SUCH_TYPE"], OUT, &["ON"], OUT),
        reached(&db, Some("Start"), &["VIA"], OUT, &["NO_SUCH_TYPE"], OUT),
    ] {
        assert!(answer.is_none_or(|mean| mean == 0.0), "{answer:?}");
    }
    db.session()
        .execute("MATCH (a:Nobody)-[:VIA]->(b), (c:Nobody)-[:NOTHING]->(b) RETURN count(*)")
        .unwrap();
}

/// A store that cannot sample says so, and the planner keeps the first stage.
#[test]
fn a_store_that_cannot_sample_says_none() {
    let store = grafeo_core::graph::traits::NullGraphStore;
    assert_eq!(
        store.sampled_reached_degree(Some("Start"), &["VIA"], OUT, &["ON"], OUT),
        None
    );
}

// ---------------------------------------------------------------------------
// When it reads
// ---------------------------------------------------------------------------

/// A graph with long lists for the queries below.
fn walked() -> GrafeoDB {
    let db = GrafeoDB::new_in_memory();
    db.create_property_index("k").unwrap();
    let starts = nodes(&db, "Person", 400);
    for pair in starts.windows(2) {
        db.create_edge(pair[0], pair[1], "KNOWS").unwrap();
    }
    db
}

#[test]
fn a_plain_query_reads_nothing() {
    let db = walked();
    let session = db.session();
    for query in [
        "MATCH (n:Person) RETURN id(n)",
        "MATCH (a:Person)-[:KNOWS]->(b) RETURN id(a), id(b)",
        "MATCH (a:Person)-[:KNOWS]->(b)-[:KNOWS]->(c) RETURN id(c)",
        "MATCH (a:Person)-[:KNOWS]->(b)-[:KNOWS]->(a) RETURN id(a)",
        "MATCH (a:Person)-[:KNOWS]->(b) RETURN id(a), count(*)",
        "INSERT (:Person)",
    ] {
        session.execute(query).unwrap();
        assert_eq!(taken(&db), 0, "{query} read a sample");
    }
}

/// A first plan reads, a second plan of the same hops reads nothing, by another text too.
#[test]
fn the_second_plan_reads_nothing() {
    let db = walked();
    let session = db.session();
    session
        .execute("MATCH (a:Person)-[:KNOWS]->(b), (c:Person)-[:KNOWS]->(b) RETURN id(a), id(c)")
        .unwrap();
    let first = taken(&db);
    assert!(first > 0, "the search read nothing");
    session
        .execute("MATCH (x:Person)-[:KNOWS]->(y), (z:Person)-[:KNOWS]->(y) RETURN count(*)")
        .unwrap();
    assert_eq!(taken(&db), first, "the same hops were sampled again");
}

/// An entry here reads 256 starts and most of their lists of 8, more than 2,048 reads, and a plan
/// spends at most 8,192 (a placeholder until the first-plan timing), so this pattern of many hops
/// reads part of them in a plan and the rest in later plans. Each later plan reads no more than
/// its budget, and in the end none.
#[test]
fn a_plan_spends_a_limited_budget() {
    let db = GrafeoDB::new_in_memory();
    let mut state = 21_u64;
    let mut next = move |bound: usize| {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        usize::try_from((state >> 33) % bound as u64).unwrap()
    };
    // Lists long enough that a few entries spend the budget of a plan.
    let labels = ["A", "B", "C", "D", "E", "F"];
    let groups: Vec<Vec<NodeId>> = labels
        .iter()
        .map(|label| nodes(&db, label, 2_000))
        .collect();
    for (position, group) in groups.iter().enumerate() {
        let others = &groups[(position + 1) % groups.len()];
        let kind = format!("T{position}");
        for node in group {
            for _ in 0..8 {
                db.create_edge(*node, others[next(2_000)], &kind).unwrap();
            }
        }
    }
    let query = "MATCH (a:A)-[:T0]->(b:B), (b)-[:T1]->(c:C), (c)-[:T2]->(d:D), (d)-[:T3]->(e:E), (e)-[:T4]->(f:F), (f)-[:T5]->(a) RETURN id(a)";
    let session = db.session();
    let explain = format!("EXPLAIN {query}");
    session.execute(&explain).unwrap();
    let mut before = taken(&db);
    assert!(before > 0, "the first plan read nothing");
    // The first plan also read the first stage, so only later plans show the second alone.
    let most_full_entries = 8_192 / 2_048;
    let mut last = u64::MAX;
    let mut read_later = 0;
    for round in 0..50 {
        // A new text, so that no plan cache answers: a space alone reads the cached plan.
        let session = db.session();
        session.execute(&format!("{explain} AS x{round}")).unwrap();
        let now = taken(&db);
        last = now - before;
        assert!(last <= most_full_entries, "{last} entries read by one plan");
        read_later += last;
        before = now;
        if last == 0 {
            break;
        }
    }
    assert_eq!(last, 0, "later plans go on reading");
    assert!(read_later > 0, "the budget never stopped a plan");
}

/// The same data gives the same value and the same plan in every process.
#[test]
fn the_value_is_the_same_in_every_process() {
    const CHILD: &str = "GRAFEO_REACHED_DEGREES_CHILD";
    let values = || {
        let db = GrafeoDB::new_in_memory();
        let starts = nodes(&db, "Start", 600);
        let ends = nodes(&db, "End", 300);
        let sink = nodes(&db, "Sink", 1)[0];
        for (index, start) in starts.iter().enumerate() {
            for step in 0..=(index % 7) {
                db.create_edge(*start, ends[(index * 13 + step) % 300], "VIA")
                    .unwrap();
            }
        }
        for (index, end) in ends.iter().enumerate() {
            for _ in 0..(index % 5) {
                db.create_edge(*end, sink, "ON").unwrap();
            }
        }
        let mean = reached(&db, Some("Start"), &["VIA"], OUT, &["ON"], OUT).unwrap();
        let plan = db
            .session()
            .execute(
                "EXPLAIN MATCH (a:Start)-[:VIA]->(m), (b:Start)-[:VIA]->(m) RETURN id(a), id(b)",
            )
            .unwrap();
        format!("{} {:?}", mean.to_bits(), plan.rows())
    };
    if std::env::var_os(CHILD).is_some() {
        print!("BEGIN\n{}\nEND\n", values());
        return;
    }
    let here = values();
    assert_eq!(values(), here);
    for _ in 0..3 {
        let output = grafeo_common::testing::child_process::output(
            grafeo_common::child_test!("the_value_is_the_same_in_every_process").env(CHILD, "1"),
        )
        .unwrap();
        assert!(output.status.success());
        let out = String::from_utf8_lossy(&output.stdout).to_string();
        let begin = out.find("BEGIN\n").unwrap() + "BEGIN\n".len();
        let end = out.find("\nEND").unwrap();
        assert_eq!(&out[begin..end], here);
    }
}

/// A large change resamples, and a small one on a large label does not, as with the first stage.
#[test]
fn a_large_change_resamples() {
    let db = GrafeoDB::new_in_memory();
    let starts = nodes(&db, "Start", 5_000);
    let end = nodes(&db, "End", 1)[0];
    let sink = nodes(&db, "Sink", 1)[0];
    for start in &starts {
        db.create_edge(*start, end, "VIA").unwrap();
    }
    db.create_edge(end, sink, "ON").unwrap();
    assert_eq!(
        reached(&db, Some("Start"), &["VIA"], OUT, &["ON"], OUT),
        Some(1.0)
    );
    let read = taken(&db);
    db.create_edge(end, sink, "ON").unwrap();
    assert_eq!(
        reached(&db, Some("Start"), &["VIA"], OUT, &["ON"], OUT),
        Some(1.0),
        "one edge of ON changed nothing to resample for"
    );
    assert_eq!(taken(&db), read);
    for _ in 0..5_000 {
        db.create_edge(end, sink, "ON").unwrap();
    }
    assert!(
        reached(&db, Some("Start"), &["VIA"], OUT, &["ON"], OUT).unwrap() > 1_000.0,
        "5,000 new edges of ON should have resampled"
    );
    assert!(taken(&db) > read);
}

/// The cache counts in what the database reports.
#[test]
fn the_cache_shows_in_the_memory_usage() {
    let db = GrafeoDB::new_in_memory();
    let starts = nodes(&db, "Start", 10);
    let end = nodes(&db, "End", 1)[0];
    for start in &starts {
        db.create_edge(*start, end, "VIA").unwrap();
    }
    let before = db.memory_usage().total_bytes;
    for index in 0..500 {
        reached(
            &db,
            Some("Start"),
            &["VIA"],
            OUT,
            &[&format!("ON{index}")],
            OUT,
        );
    }
    let after = db.memory_usage().total_bytes;
    assert!(
        after > before,
        "{before} bytes, then {after} with 500 entries"
    );
    // A small bound on entries: a long-running process with new types does not grow for ever.
    assert!(after - before < 1 << 20, "{} bytes", after - before);
}

// ---------------------------------------------------------------------------
// What it does to a plan
// ---------------------------------------------------------------------------

/// `Left` and `Right` reach 4 hubs, among `ordinary` plain middles, with or without a label.
fn hub_shape(ordinary: usize, label: &str) -> (GrafeoDB, String) {
    let db = GrafeoDB::new_in_memory();
    let plain = nodes(&db, "Middle", ordinary);
    let hubs = nodes(&db, "Middle", 4);
    for hub in &hubs {
        for other in nodes(&db, "Other", 3_000) {
            db.create_edge(other, *hub, "TO").unwrap();
        }
    }
    for node in &plain {
        let other = nodes(&db, "Other", 1)[0];
        db.create_edge(other, *node, "TO").unwrap();
    }
    for index in 0..5 {
        let left = db.create_node(&["Left"]).unwrap();
        db.create_edge(left, hubs[index % 4], "TO").unwrap();
        let right = db.create_node(&["Right"]).unwrap();
        db.create_edge(right, hubs[index % 4], "TO").unwrap();
    }
    let query =
        format!("MATCH (l:Left)-[:TO]->(m{label}), (r:Right)-[:TO]->(m) RETURN id(l), id(r)");
    (db, query)
}

fn sorted_rows(session: &Session, query: &str) -> Vec<String> {
    let mut rows: Vec<String> = session
        .execute(query)
        .unwrap()
        .rows()
        .iter()
        .map(|row| format!("{row:?}"))
        .collect();
    rows.sort();
    rows
}

/// The first stage walked 15,024 list entries for 7 rows on this shape (63x slower than the
/// hash join at this size, 684x at 30,000 a hub). Both sides are few, so the plan joins them,
/// with a middle that has no label, one that has only hubs, and one where hubs are a few of
/// 10,000.
#[test]
fn the_hub_shape_joins_its_sides() {
    for (name, ordinary, label) in [
        ("no label", 0, ""),
        ("label of hubs", 0, ":Middle"),
        ("no label among ordinary ones", 10_000, ""),
        ("label among ordinary ones", 10_000, ":Middle"),
    ] {
        let (db, query) = hub_shape(ordinary, label);
        let session = db.session();
        let result = session.execute(&query).unwrap();
        let reads = result.rows_scanned().unwrap();
        let plan = session.execute(&format!("EXPLAIN {query}")).unwrap();
        assert!(reads <= 100, "{name}: {reads} reads\n{:?}", plan.rows());
        let mut plain = db.session();
        plain.set_planner_rules(PlannerRules::plain());
        assert_eq!(
            sorted_rows(&session, &query),
            sorted_rows(&plain, &query),
            "{name}"
        );
    }
}

/// A many-to-few shape where the expand is right, and the sides are not few: the second stage
/// must not push a hash table of every row on it.
#[test]
fn a_shape_without_hubs_keeps_its_expand() {
    let db = GrafeoDB::new_in_memory();
    let few = nodes(&db, "Few", 10);
    let many = nodes(&db, "Many", 20_000);
    let middle = nodes(&db, "Middle", 10_000);
    for (index, node) in few.iter().enumerate() {
        db.create_edge(*node, middle[index * 100], "TO").unwrap();
    }
    for (index, node) in many.iter().enumerate() {
        db.create_edge(*node, middle[index % 10_000], "TO").unwrap();
    }
    let query = "MATCH (a:Few)-[:TO]->(m), (b:Many)-[:TO]->(m) RETURN id(a), id(b)";
    let session = db.session();
    let result = session.execute(query).unwrap();
    assert_eq!(result.row_count(), 20);
    let reads = result.rows_scanned().unwrap();
    assert!(
        reads < 1_000,
        "{reads} reads: starting at the ten rows reads a few lists\n{:?}",
        session.execute(&format!("EXPLAIN {query}")).unwrap().rows()
    );
}

/// A variable reached through two hops takes the larger estimate: `m` looks plain from `Left`
/// and is a hub from `Right`, and the plan must not walk `m`'s long list once per row.
#[test]
fn a_variable_reached_twice_takes_the_larger_estimate() {
    let db = GrafeoDB::new_in_memory();
    let plain = nodes(&db, "Middle", 200);
    let hubs = nodes(&db, "Middle", 2);
    for hub in &hubs {
        for far in nodes(&db, "Far", 5_000) {
            db.create_edge(*hub, far, "ON").unwrap();
        }
    }
    for node in &plain {
        let far = nodes(&db, "Far", 1)[0];
        db.create_edge(*node, far, "ON").unwrap();
    }
    for node in nodes(&db, "Left", 100) {
        db.create_edge(node, plain[node.as_u64() as usize % 200], "A")
            .unwrap();
    }
    for (index, node) in nodes(&db, "Right", 100).iter().enumerate() {
        db.create_edge(*node, hubs[index % 2], "B").unwrap();
    }
    let query = "MATCH (l:Left)-[:A]->(m), (r:Right)-[:B]->(m), (m)-[:ON]->(f:Far) RETURN id(l), id(r), id(f)";
    let session = db.session();
    let result = session.execute(query).unwrap();
    assert_eq!(
        result.row_count(),
        0,
        "no middle is reached from both sides"
    );
    let reads = result.rows_scanned().unwrap();
    assert!(
        reads < 5_000,
        "{reads} reads: the hubs' lists were walked\n{:?}",
        session.execute(&format!("EXPLAIN {query}")).unwrap().rows()
    );
}

/// With the search off nothing reads the second stage, and rows do not depend on it.
#[test]
fn with_the_search_off_nothing_is_read() {
    let (db, query) = hub_shape(0, "");
    let mut session = db.session();
    session.set_planner_rules(PlannerRules::all().with(PlannerRule::JoinOrder, false));
    let off_rows = sorted_rows(&session, &query);
    assert_eq!(taken(&db), 0);
    let chosen = sorted_rows(&db.session(), &query);
    assert_eq!(chosen, off_rows);
    assert!(taken(&db) > 0);
}
