//! What the planner believes when a sample or a filter says little (#714).
//!
//! Tests first, from the decided behavior: a sample that reads 0 on a label larger than the
//! sample uses the derived degree, a sample of the whole label keeps its 0, a low non-zero read
//! is not raised. `id(var)` of a bound node cannot raise, so it stays with its hop, and a range
//! on ids is estimated from the id range. Rows never depend on any of it: both are checked
//! against a plan with no rule on, and against an answer worked out by hand.
//!
//! ```bash
//! cargo test -p grafeo-engine --all-features --test main degree_estimates::
//! ```

#![cfg(all(feature = "gql", feature = "cypher"))]
#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss,
    reason = "test graphs of a few thousand nodes"
)]
#![allow(
    deprecated,
    reason = "the test reads the raw store's sampled degrees, which no session shows"
)]

use std::collections::{BTreeSet, HashMap};

use grafeo_common::types::{NodeId, Value};
use grafeo_core::graph::Direction;
use grafeo_core::graph::traits::GraphStore;
use grafeo_engine::GrafeoDB;
use grafeo_engine::query::optimizer::{Optimizer, PlannerRule, PlannerRules, hop_degree};
use grafeo_engine::query::translate_gql;
use grafeo_engine::session::Session;

const OUT: Direction = Direction::Outgoing;
const IN: Direction = Direction::Incoming;

fn to_int(id: u64) -> i64 {
    i64::try_from(id).unwrap()
}

fn nodes(db: &GrafeoDB, label: &str, count: usize) -> Vec<NodeId> {
    (0..count)
        .map(|_| db.create_node(&[label]).unwrap())
        .collect()
}

/// The ranks a sample reads, as `sampled_degrees.rs` pins them.
fn ranks(count: u64) -> BTreeSet<u64> {
    const GOLDEN: u64 = 0x9E37_79B9_7F4A_7C15;
    (1..=256_u64)
        .map(|i| {
            u64::try_from((u128::from(i.wrapping_mul(GOLDEN)) * u128::from(count)) >> 64).unwrap()
        })
        .collect()
}

/// The positions of a label of `count` nodes that a sample does not read.
fn outside_the_sample(count: usize) -> Vec<usize> {
    let read = ranks(u64::try_from(count).unwrap());
    (0..count)
        .filter(|rank| !read.contains(&u64::try_from(*rank).unwrap()))
        .collect()
}

fn sampled(db: &GrafeoDB, label: &str, kind: &str, direction: Direction) -> f64 {
    db.store()
        .unwrap()
        .sampled_degree(label, kind, direction)
        .unwrap_or_else(|| panic!("no sample for {label} {kind} {direction:?}"))
}

fn degree(db: &GrafeoDB, label: &str, kind: &str, direction: Direction) -> f64 {
    hop_degree(&*db.store().unwrap(), &[label], &[kind], direction)
}

// ---------------------------------------------------------------------------
// A sample that reads nothing
// ---------------------------------------------------------------------------

/// Every edge sits on nodes the sample does not read, so it reads 0 where the mean is 1. The
/// hop looked free, and a plan through those nodes ran 40 times slower than the written order.
#[test]
fn a_zero_sample_of_a_large_label_uses_the_derived_degree() {
    let db = GrafeoDB::new_in_memory();
    let holders = nodes(&db, "Holder", 2_000);
    let sink = nodes(&db, "Sink", 1)[0];
    let outside = outside_the_sample(holders.len());
    for rank in outside.iter().take(20) {
        for _ in 0..100 {
            db.create_edge(holders[*rank], sink, "E").unwrap();
        }
    }
    assert_eq!(
        sampled(&db, "Holder", "E", OUT).to_bits(),
        0.0_f64.to_bits()
    );
    assert_eq!(
        degree(&db, "Holder", "E", OUT),
        1.0,
        "2,000 edges over 2,000 nodes"
    );
}

/// The smallest label larger than the sample behaves the same: 257 nodes cannot all be read.
#[test]
fn the_derived_degree_starts_one_node_past_the_sample() {
    let db = GrafeoDB::new_in_memory();
    let holders = nodes(&db, "Holder", 257);
    let sink = nodes(&db, "Sink", 1)[0];
    let outside = outside_the_sample(holders.len());
    assert!(!outside.is_empty(), "256 ranks cannot cover 257 nodes");
    for _ in 0..257 {
        db.create_edge(holders[outside[0]], sink, "E").unwrap();
    }
    assert_eq!(
        sampled(&db, "Holder", "E", OUT).to_bits(),
        0.0_f64.to_bits()
    );
    assert_eq!(degree(&db, "Holder", "E", OUT), 1.0);
}

/// A sample of the whole label knows: none of its nodes has the type's edges, wherever else
/// they are, at the largest size read whole and below it.
#[test]
fn a_sample_of_the_whole_label_keeps_its_zero() {
    for size in [1, 10, 256] {
        let db = GrafeoDB::new_in_memory();
        nodes(&db, "Quiet", size);
        let others = nodes(&db, "Busy", 10);
        for _ in 0..1_000 {
            db.create_edge(others[0], others[1], "E").unwrap();
        }
        assert_eq!(sampled(&db, "Quiet", "E", OUT).to_bits(), 0.0_f64.to_bits());
        assert_eq!(
            degree(&db, "Quiet", "E", OUT).to_bits(),
            0.0_f64.to_bits(),
            "a label of {size} was read whole"
        );
    }
}

/// A low read that is not zero stays as it is: only an exact 0 is a known false reading.
#[test]
fn a_low_read_is_not_raised() {
    let db = GrafeoDB::new_in_memory();
    let holders = nodes(&db, "Holder", 2_000);
    let sink = nodes(&db, "Sink", 1)[0];
    let read = ranks(u64::try_from(holders.len()).unwrap());
    let first_read = usize::try_from(*read.iter().next().unwrap()).unwrap();
    db.create_edge(holders[first_read], sink, "E").unwrap();
    for rank in outside_the_sample(holders.len()).iter().take(10) {
        for _ in 0..1_000 {
            db.create_edge(holders[*rank], sink, "E").unwrap();
        }
    }
    let one_in_the_sample = 1.0 / read.len() as f64;
    assert_eq!(
        sampled(&db, "Holder", "E", OUT).to_bits(),
        one_in_the_sample.to_bits()
    );
    assert_eq!(
        degree(&db, "Holder", "E", OUT).to_bits(),
        one_in_the_sample.to_bits()
    );
}

/// A type with no edge at all stays 0 on a large label: there is nothing to derive from.
#[test]
fn a_type_with_no_edges_stays_at_zero() {
    let db = GrafeoDB::new_in_memory();
    let holders = nodes(&db, "Holder", 2_000);
    db.create_edge(holders[0], holders[1], "OTHER").unwrap();
    assert_eq!(
        degree(&db, "Holder", "NONE", OUT).to_bits(),
        0.0_f64.to_bits()
    );
    assert_eq!(
        degree(&db, "Holder", "NONE", IN).to_bits(),
        0.0_f64.to_bits()
    );
}

/// With a true zero the query has no rows, and finding that takes few reads whichever part the
/// plan starts from: no part lists a far side's edges for each of its rows.
#[test]
fn a_hop_with_no_edges_finds_no_rows_cheaply() {
    let db = GrafeoDB::new_in_memory();
    let big = nodes(&db, "Big", 5_000);
    let middle = nodes(&db, "Middle", 50);
    for (index, node) in big.iter().enumerate() {
        db.create_edge(*node, middle[index % 50], "E").unwrap();
    }
    let query = "MATCH (a:Big)-[:NONE]->(m:Middle), (b:Big)-[:E]->(m) RETURN id(a), id(b)";
    let result = db.session().execute(query).unwrap();
    assert_eq!(result.row_count(), 0);
    let reads = result.rows_scanned().unwrap();
    assert!(reads <= 15_000, "{reads} reads to find nothing");
}

// ---------------------------------------------------------------------------
// id() stays with its hop, and a range on ids has an estimate
// ---------------------------------------------------------------------------

/// The first label holds the lowest ids, so `id(h) < c` selects from it alone. Its nodes have
/// the same edges per node as the other side, and neither degree tells the two apart.
struct TwoSides {
    db: GrafeoDB,
    hubby: Vec<NodeId>,
    flat: Vec<NodeId>,
    leaf: Vec<NodeId>,
}

fn two_sides() -> TwoSides {
    let db = GrafeoDB::new_in_memory();
    let hubby = nodes(&db, "Hubby", 2_000);
    let flat = nodes(&db, "Flat", 2_000);
    let leaf = nodes(&db, "Leaf", 200);
    let mut state = 11_u64;
    let mut next = move |bound: usize| {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        usize::try_from((state >> 33) % bound as u64).unwrap()
    };
    for node in hubby.iter().chain(&flat) {
        for _ in 0..3 {
            let (from, to) = (*node, leaf[next(200)]);
            let kind = if hubby.contains(node) { "H" } else { "F" };
            db.create_edge(from, to, kind).unwrap();
        }
    }
    TwoSides {
        db,
        hubby,
        flat,
        leaf,
    }
}

fn explain_gql(session: &Session, query: &str) -> String {
    let result = session.execute(&format!("EXPLAIN {query}")).unwrap();
    match &result.rows()[0][0] {
        Value::String(text) => text.to_string(),
        other => format!("{other:?}"),
    }
}

fn explain_cypher(session: &Session, query: &str) -> String {
    let result = session.execute_cypher(&format!("EXPLAIN {query}")).unwrap();
    match &result.rows()[0][0] {
        Value::String(text) => text.to_string(),
        other => format!("{other:?}"),
    }
}

/// The line under the filter that reads `id(variable)`: what it filters.
fn under_the_id_filter(plan: &str, variable: &str) -> String {
    let needle = format!("id({variable})");
    let lines: Vec<&str> = plan.lines().collect();
    let at = lines
        .iter()
        .position(|line| line.contains("Filter") && line.contains(&needle))
        .unwrap_or_else(|| panic!("no filter on {needle} in\n{plan}"));
    lines[at + 1].trim().to_string()
}

/// `id(h) < c` was lifted above the whole tree, so the plan read every hop and threw the rows
/// away at the end (721 ms against 11.8 ms as written). It stays right over the scan of `h`.
#[test]
fn an_id_filter_stays_with_its_hop() {
    let fixture = two_sides();
    let top = fixture.hubby[20].as_u64();
    let session = fixture.db.session();
    let query = format!(
        "MATCH (h:Hubby)-[:H]->(t:Leaf), (f:Flat)-[:F]->(t) WHERE id(h) < {top} RETURN id(h), id(t), id(f)"
    );
    for (language, plan) in [
        ("GQL", explain_gql(&session, &query)),
        ("Cypher", explain_cypher(&session, &query)),
    ] {
        let under = under_the_id_filter(&plan, "h");
        assert!(
            under.contains("h:Hubby"),
            "{language}: the id filter is over {under}, not the scan of h, in\n{plan}"
        );
    }
}

/// A selective range makes its relation the one the plan starts from: few rows, then their
/// hops. It reads a fraction of what starting at the far end reads.
#[test]
fn a_selective_id_range_starts_the_plan() {
    let fixture = two_sides();
    let top = fixture.hubby[5].as_u64();
    let query = format!(
        "MATCH (h:Hubby)-[:H]->(t:Leaf), (f:Flat)-[:F]->(t) WHERE id(h) < {top} RETURN id(h), id(t), id(f)"
    );
    let chosen = fixture.db.session().execute(&query).unwrap();
    let everything = 2_000 * 3 * 2;
    let reads = chosen.rows_scanned().unwrap();
    assert!(
        reads < everything,
        "{reads} reads, as many as listing both sides ({everything})\n{}",
        explain_gql(&fixture.db.session(), &query)
    );
}

/// The estimate through the optimizer's own function: `id(a) < c` is c over the node count of
/// the rows it filters, clamped at all of them. Written on either side, or as `<=`.
#[test]
fn an_id_range_is_estimated_from_the_id_range() {
    let db = GrafeoDB::new_in_memory();
    nodes(&db, "V", 500);
    nodes(&db, "W", 500);
    let estimate = |condition: &str| {
        let store = db.store().unwrap();
        let optimizer = Optimizer::from_graph_store(&*store);
        let plan = translate_gql(&format!("MATCH (a:V) WHERE {condition} RETURN a")).unwrap();
        let plan = optimizer.optimize(plan).unwrap();
        optimizer.estimate_cardinality(&plan)
    };
    // 500 nodes of a label times 100 of 1,000 ids.
    for condition in ["id(a) < 100", "100 > id(a)", "id(a) <= 100"] {
        let rows = estimate(condition);
        assert!((40.0..=60.0).contains(&rows), "{condition}: {rows}");
    }
    // Past the end, all of them.
    for condition in [
        "id(a) < 1000",
        "id(a) < 1000000",
        "id(a) < 9223372036854775807",
    ] {
        let rows = estimate(condition);
        assert!((400.0..=500.0).contains(&rows), "{condition}: {rows}");
    }
    // Below the start, nearly none, and no negative or NaN estimate.
    for condition in ["id(a) < 0", "id(a) < -5", "id(a) < -9223372036854775807"] {
        let rows = estimate(condition);
        assert!((0.0..=1.0).contains(&rows), "{condition}: {rows}");
    }
}

/// A condition that is no number gets no id estimate, and planning it is no error.
#[test]
fn an_id_range_on_something_else_plans_without_an_estimate() {
    let db = GrafeoDB::new_in_memory();
    nodes(&db, "V", 100);
    let store = db.store().unwrap();
    let optimizer = Optimizer::from_graph_store(&*store);
    for condition in [
        "id(a) < 'text'",
        "id(a) < NULL",
        "id(a) < 1.5",
        "id(a) < a.k",
    ] {
        let plan = translate_gql(&format!("MATCH (a:V) WHERE {condition} RETURN a")).unwrap();
        let plan = optimizer.optimize(plan).unwrap();
        let rows = optimizer.estimate_cardinality(&plan);
        assert!((0.0..=100.0).contains(&rows), "{condition}: {rows}");
    }
}

/// Through a parameter the plan starts from the same relation as with the literal.
#[test]
fn a_parameter_is_estimated_like_the_literal() {
    let fixture = two_sides();
    let top = fixture.hubby[5].as_u64();
    let literal = format!(
        "EXPLAIN MATCH (h:Hubby)-[:H]->(t:Leaf), (f:Flat)-[:F]->(t) WHERE id(h) < {top} RETURN id(h), id(t), id(f)"
    );
    let with_parameter = "EXPLAIN MATCH (h:Hubby)-[:H]->(t:Leaf), (f:Flat)-[:F]->(t) WHERE id(h) < $top RETURN id(h), id(t), id(f)";
    let parameters: HashMap<String, Value> =
        HashMap::from([("top".to_string(), Value::Int64(to_int(top)))]);
    let text = |result: grafeo_engine::database::QueryResult| match &result.rows()[0][0] {
        Value::String(text) => text.to_string(),
        other => format!("{other:?}"),
    };
    let session = fixture.db.session();
    let as_literal = text(session.execute(&literal).unwrap());
    let as_parameter = text(
        session
            .execute_with_params(with_parameter, parameters)
            .unwrap(),
    );
    assert_eq!(
        under_the_id_filter(&as_literal, "h"),
        under_the_id_filter(&as_parameter, "h"),
        "literal:\n{as_literal}\nparameter:\n{as_parameter}"
    );
    assert!(
        under_the_id_filter(&as_parameter, "h").contains("h:Hubby"),
        "{as_parameter}"
    );
}

/// One query text, the same session, values at both ends of the range: a plan chosen for one
/// value may not suit the other, but the rows are right for each.
#[test]
fn every_value_of_a_parameter_gets_its_rows() {
    let fixture = two_sides();
    let session = fixture.db.session();
    let query = "MATCH (h:Hubby) WHERE id(h) < $top RETURN count(*)";
    let everyone = fixture.hubby.last().unwrap().as_u64() + 1;
    let first = fixture.hubby[0].as_u64();
    for (top, want) in [
        (first + 3, 3),
        (everyone, 2_000),
        (first, 0),
        (first + 1_000, 1_000),
        (first + 3, 3),
    ] {
        let parameters = HashMap::from([("top".to_string(), Value::Int64(to_int(top)))]);
        let result = session.execute_with_params(query, parameters).unwrap();
        assert_eq!(result.rows()[0][0], Value::Int64(want), "id(h) < {top}");
    }
}

// ---------------------------------------------------------------------------
// Rows at every boundary
// ---------------------------------------------------------------------------

fn sorted_rows(result: &grafeo_engine::database::QueryResult) -> Vec<String> {
    let mut rows: Vec<String> = result.rows().iter().map(|row| format!("{row:?}")).collect();
    rows.sort();
    rows
}

/// The comma query with its id condition, by hand from the edges, with an id of `-1` for the
/// values nothing compares below.
fn by_hand(edges: &[(u64, u64)], other: &[(u64, u64)], keep: impl Fn(i64) -> bool) -> Vec<String> {
    let mut rows = Vec::new();
    for (from, far) in edges {
        if !keep(to_int(*from)) {
            continue;
        }
        for (start, end) in other {
            if end == far {
                rows.push(format!(
                    "{:?}",
                    vec![
                        Value::Int64(to_int(*from)),
                        Value::Int64(to_int(*far)),
                        Value::Int64(to_int(*start))
                    ]
                ));
            }
        }
    }
    rows.sort();
    rows
}

fn every_rule_set(db: &GrafeoDB) -> Vec<(&'static str, Session)> {
    [
        ("all", PlannerRules::all()),
        ("plain", PlannerRules::plain()),
        (
            "no join order",
            PlannerRules::all().with(PlannerRule::JoinOrder, false),
        ),
    ]
    .into_iter()
    .map(|(name, rules)| {
        let mut session = db.session();
        session.set_planner_rules(rules);
        (name, session)
    })
    .collect()
}

/// Each comparison at each boundary: below 0, 0, the first and last id, past the last, and the
/// largest number there is; with the rows from the edges by hand.
fn check_boundaries(db: &GrafeoDB, edges: &[(u64, u64)], other: &[(u64, u64)], points: &[i64]) {
    type Compare = fn(i64, i64) -> bool;
    let comparisons: [(&str, Compare); 6] = [
        ("<", |x, c| x < c),
        ("<=", |x, c| x <= c),
        (">", |x, c| x > c),
        (">=", |x, c| x >= c),
        ("=", |x, c| x == c),
        ("<>", |x, c| x != c),
    ];
    for (name, session) in every_rule_set(db) {
        for point in points {
            for (operator, compare) in comparisons {
                let query = format!(
                    "MATCH (a:Hubby)-[:H]->(t:Leaf), (b:Flat)-[:F]->(t) WHERE id(a) {operator} {point} RETURN id(a), id(t), id(b)"
                );
                let want = by_hand(edges, other, |id| compare(id, *point));
                let got = sorted_rows(&session.execute(&query).unwrap());
                assert_eq!(got, want, "{name}: {query}");
                let got = sorted_rows(&session.execute_cypher(&query).unwrap());
                assert_eq!(got, want, "{name}, Cypher: {query}");
            }
        }
    }
}

/// A small graph, with the edges kept to work the rows out by hand.
fn boundary_graph() -> (GrafeoDB, Vec<NodeId>, Vec<(u64, u64)>, Vec<(u64, u64)>) {
    let db = GrafeoDB::new_in_memory();
    let left = nodes(&db, "Hubby", 30);
    let right = nodes(&db, "Flat", 30);
    let leaves = nodes(&db, "Leaf", 5);
    let mut edges = Vec::new();
    let mut other = Vec::new();
    for (index, node) in left.iter().enumerate() {
        let far = leaves[index % 5];
        db.create_edge(*node, far, "H").unwrap();
        edges.push((node.as_u64(), far.as_u64()));
    }
    for (index, node) in right.iter().enumerate() {
        let far = leaves[(index * 2) % 5];
        db.create_edge(*node, far, "F").unwrap();
        other.push((node.as_u64(), far.as_u64()));
    }
    (db, left, edges, other)
}

#[test]
fn every_comparison_at_every_boundary_gets_its_rows() {
    let (db, left, edges, other) = boundary_graph();
    let first = to_int(left[0].as_u64());
    let last = to_int(left[29].as_u64());
    check_boundaries(
        &db,
        &edges,
        &other,
        &[
            -1,
            0,
            first,
            first + 1,
            last - 1,
            last,
            last + 1,
            i64::MAX,
            i64::MIN + 1,
        ],
    );
}

/// A range of deleted ids: the id range no longer matches the node count, the estimate may be
/// off and the rows are not.
#[test]
fn a_deleted_id_range_changes_no_rows() {
    let (db, left, edges, other) = boundary_graph();
    let (kept, gone): (Vec<_>, Vec<_>) = edges
        .iter()
        .partition(|(from, _)| !(left[10].as_u64()..left[20].as_u64()).contains(from));
    let session = db.session();
    for (from, _) in gone {
        session
            .execute(&format!("MATCH (a) WHERE id(a) = {from} DETACH DELETE a"))
            .unwrap();
    }
    let first = to_int(left[0].as_u64());
    let last = to_int(left[29].as_u64());
    check_boundaries(
        &db,
        &kept,
        &other,
        &[
            first + 5,
            to_int(left[10].as_u64()),
            to_int(left[15].as_u64()),
            to_int(left[19].as_u64()),
            to_int(left[20].as_u64()),
            last + 1,
        ],
    );
}

// ---------------------------------------------------------------------------
// id() of what is not a bound node gives what the plain plan gives
// ---------------------------------------------------------------------------

/// Rule on against rule off: both succeed means the same rows; only rule off failing may let
/// rule on succeed; rule off succeeding and rule on failing is the failure.
fn same_as_plain(db: &GrafeoDB, query: &str, cypher: bool) {
    let run = |session: &Session| {
        if cypher {
            session.execute_cypher(query)
        } else {
            session.execute(query)
        }
    };
    let mut plain = db.session();
    plain.set_planner_rules(PlannerRules::plain());
    let chosen = db.session();
    match (run(&plain), run(&chosen)) {
        (Ok(plain_rows), Ok(chosen_rows)) => {
            assert_eq!(
                sorted_rows(&plain_rows),
                sorted_rows(&chosen_rows),
                "{query}"
            );
        }
        (Ok(_), Err(error)) => {
            panic!("{query}: the plain plan runs, the chosen one fails: {error}")
        }
        (Err(_), _) => {}
    }
}

#[test]
fn id_of_what_is_not_a_node_gives_what_the_plain_plan_gives() {
    let (db, left, _, _) = boundary_graph();
    let top = left[10].as_u64();
    for query in [
        // An edge variable, on the part the search moves.
        format!("MATCH (a:Hubby)-[e:H]->(t:Leaf), (b:Flat)-[:F]->(t) WHERE id(e) < {top} RETURN id(a), id(b)"),
        // An end an OPTIONAL MATCH may leave null.
        format!("MATCH (a:Hubby) OPTIONAL MATCH (a)-[:F]->(b) WITH a, b WHERE id(b) < {top} RETURN id(a), id(b)"),
        format!("MATCH (a:Hubby) OPTIONAL MATCH (a)-[:H]->(b) RETURN id(a), id(b) < {top}"),
        format!("MATCH (a:Hubby)-[:H]->(t:Leaf) OPTIONAL MATCH (b:Flat)-[:F]->(t) WHERE id(b) < {top} RETURN id(a), id(b)"),
        // A number, and a property.
        format!("UNWIND [1, 2] AS n MATCH (a:Hubby)-[:H]->(t:Leaf), (b:Flat)-[:F]->(t) WHERE id(n) < {top} RETURN id(a), id(b)"),
        format!("MATCH (a:Hubby)-[:H]->(t:Leaf), (b:Flat)-[:F]->(t) WHERE id(a.k) < {top} RETURN id(a), id(b)"),
        format!("UNWIND [1, 2] AS n WITH n WHERE id(n) < {top} RETURN n"),
        // A later MATCH, where the node was bound before the part.
        format!("MATCH (a:Hubby) WITH a MATCH (a)-[:H]->(t:Leaf), (b:Flat)-[:F]->(t) WHERE id(a) < {top} RETURN id(a), id(b)"),
        // Beside a condition that can raise, written before and after.
        format!("MATCH (a:Hubby)-[:H]->(t:Leaf), (b:Flat)-[:F]->(t) WHERE 100 / id(a) > 0 AND id(a) < {top} RETURN id(a), id(b)"),
        "MATCH (a:Hubby)-[:H]->(t:Leaf), (b:Flat)-[:F]->(t) WHERE id(a) > 0 AND 100 / id(a) > 0 RETURN id(a), id(b)".to_string(),
    ] {
        same_as_plain(&db, &query, false);
        same_as_plain(&db, &query, true);
    }
}

/// A range that holds nothing still finds nothing when the far side would raise: a check that
/// cannot raise may run first, never later than it was written.
#[test]
fn an_id_check_written_first_guards_what_follows() {
    let (db, _, _, _) = boundary_graph();
    let query = "MATCH (a:Hubby)-[:H]->(t:Leaf), (b:Flat)-[:F]->(t) WHERE id(a) < -1 AND 100 / (id(a) - id(a)) > 0 RETURN id(a)";
    let result = db.session().execute(query).unwrap();
    assert_eq!(result.row_count(), 0);
}

/// The other side's id range filters the other side.
#[test]
fn a_range_on_either_side_finds_the_same_as_by_hand() {
    let fixture = two_sides();
    let flat_top = fixture.flat[3].as_u64();
    let first_flat = fixture.flat[0].as_u64();
    let query = format!(
        "MATCH (h:Hubby)-[:H]->(t:Leaf), (f:Flat)-[:F]->(t) WHERE id(f) < {flat_top} AND id(f) >= {first_flat} RETURN id(f), id(t), id(h)"
    );
    let mut plain = fixture.db.session();
    plain.set_planner_rules(PlannerRules::plain());
    let want = sorted_rows(&plain.execute(&query).unwrap());
    assert!(!want.is_empty(), "{} leaves and no row", fixture.leaf.len());
    let session = fixture.db.session();
    assert_eq!(sorted_rows(&session.execute(&query).unwrap()), want);
    assert_eq!(sorted_rows(&session.execute_cypher(&query).unwrap()), want);
    let plan = explain_gql(&session, &query);
    assert!(
        under_the_id_filter(&plan, "f").contains("f:Flat"),
        "the id filter left the scan of f:\n{plan}"
    );
}
