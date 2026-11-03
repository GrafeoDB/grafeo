//! Where a filter that can raise runs once the join search has chosen an order (#714).
//!
//! Tests first, from the ruling: a filter on one relation stays with that relation, right above
//! its hop, wherever the search puts the relation. It stays below any later hop, above the
//! relation's own label checks, and in its written order among that relation's other filters.
//! A filter that reads two relations stays above both. A tree with a raising filter under a
//! `LIMIT` is not reordered. The written plan runs the filter on every row of the relation, so
//! a chosen plan that reads a subset of them cannot raise where the written plan did not.
//!
//! The one exception, decided by the user: when another part of the pattern is empty, the written
//! plan's hash join skips its other side and returns no rows, and a chosen plan may raise. It
//! is pinned here by name and in `planner_rules_results.rs`.
//!
//! ```bash
//! cargo test -p grafeo-engine --all-features --test main raising_filter_placement::
//! ```

#![cfg(all(feature = "gql", feature = "cypher"))]
#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss,
    reason = "test graphs of a few thousand nodes"
)]

use grafeo_common::types::{NodeId, Value};
use grafeo_engine::GrafeoDB;
use grafeo_engine::query::optimizer::{PlannerRule, PlannerRules};
use grafeo_engine::session::Session;

fn node(db: &GrafeoDB, label: &str, key: i64) -> NodeId {
    db.create_node_with_props(&[label], [("k", Value::Int64(key))])
        .unwrap()
}

fn plan_of(session: &Session, query: &str) -> String {
    let result = session.execute(&format!("EXPLAIN {query}")).unwrap();
    match &result.rows()[0][0] {
        Value::String(text) => text.to_string(),
        other => format!("{other:?}"),
    }
}

fn sorted(session: &Session, query: &str) -> Result<Vec<String>, String> {
    session
        .execute(query)
        .map(|result| {
            let mut rows: Vec<String> =
                result.rows().iter().map(|row| format!("{row:?}")).collect();
            rows.sort();
            rows
        })
        .map_err(|error| error.to_string())
}

fn written(db: &GrafeoDB) -> Session {
    let mut session = db.session();
    session.set_planner_rules(PlannerRules::all().with(PlannerRule::JoinOrder, false));
    session
}

fn plain(db: &GrafeoDB) -> Session {
    let mut session = db.session();
    session.set_planner_rules(PlannerRules::plain());
    session
}

/// The chosen plan answers as the written one does, and a plan that runs the query at all.
fn same_as_written(db: &GrafeoDB, query: &str) -> Vec<String> {
    let want = sorted(&written(db), query).unwrap_or_else(|error| panic!("written: {error}"));
    let got = sorted(&db.session(), query).unwrap_or_else(|error| {
        panic!(
            "{query}: the chosen plan fails: {error}\n{}",
            plan_of(&db.session(), query)
        )
    });
    assert_eq!(got, want, "{query}\n{}", plan_of(&db.session(), query));
    want
}

/// Line numbers of the plan's lines holding `needle`, the first first.
fn line_of(plan: &str, needle: &str) -> usize {
    plan.lines()
        .position(|line| line.contains(needle))
        .unwrap_or_else(|| panic!("no {needle} in\n{plan}"))
}

/// Many rows through each part, so that where the filter runs shows in the reads and the clock.
fn far_end_graph() -> GrafeoDB {
    let db = GrafeoDB::new_in_memory();
    let left: Vec<NodeId> = (0..2_000).map(|key| node(&db, "A", key + 1)).collect();
    let right: Vec<NodeId> = (0..2_000).map(|key| node(&db, "B", key + 1)).collect();
    let far: Vec<NodeId> = (0..200).map(|key| node(&db, "M", key + 1)).collect();
    let mut state = 5_u64;
    let mut next = move |bound: usize| {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        usize::try_from((state >> 33) % bound as u64).unwrap()
    };
    for node in &left {
        for _ in 0..3 {
            db.create_edge(*node, far[next(200)], "X").unwrap();
        }
    }
    for node in &right {
        for _ in 0..3 {
            db.create_edge(*node, far[next(200)], "Y").unwrap();
        }
    }
    db
}

// ---------------------------------------------------------------------------
// It stays with its relation
// ---------------------------------------------------------------------------

/// `lower(a.k)`-style conditions were lifted to the top of the plan, so the plan listed everything
/// first: 203 ms against 6.5 ms written. The filter sits below the other part's hop.
#[test]
fn a_raising_filter_runs_before_the_other_parts_hop() {
    let db = far_end_graph();
    for (query, variable, other_hop) in [
        (
            "MATCH (a:A)-[:X]->(m:M), (b:B)-[:Y]->(m) WHERE 100 / a.k = 100 RETURN id(a), id(b)",
            "100",
            "(b)",
        ),
        (
            "MATCH (a:A)-[:X]->(m:M), (b:B)-[:Y]->(m) WHERE 100 / b.k = 100 RETURN id(a), id(b)",
            "100",
            "(a)",
        ),
    ] {
        let session = db.session();
        let plan = plan_of(&session, query);
        let filter = line_of(&plan, variable);
        let hops_above_the_filter = plan
            .lines()
            .take(filter)
            .filter(|line| line.contains("Expand") && line.contains(other_hop))
            .count();
        let reads = session.execute(query).unwrap().rows_scanned().unwrap();
        let rows_listed_in_full = 2_000 * 3 * 3;
        assert!(reads <= rows_listed_in_full, "{reads} reads\n{plan}");
        assert!(
            hops_above_the_filter > 0,
            "the other part ran first:\n{plan}"
        );
        same_as_written(&db, query);
    }
}

/// Where the chosen order starts elsewhere, the rows that reach the filter are the relation's,
/// not the whole pattern's. Three parts in a line, written first the one with the filter.
#[test]
fn the_filter_runs_early_in_a_longer_pattern() {
    let db = GrafeoDB::new_in_memory();
    let first: Vec<NodeId> = (0..500).map(|key| node(&db, "A", key + 1)).collect();
    let one: Vec<NodeId> = (0..100).map(|key| node(&db, "M", key + 1)).collect();
    let two: Vec<NodeId> = (0..20).map(|key| node(&db, "N", key + 1)).collect();
    let last: Vec<NodeId> = (0..500).map(|key| node(&db, "B", key + 1)).collect();
    for (index, node) in first.iter().enumerate() {
        db.create_edge(*node, one[index % 100], "X").unwrap();
    }
    for (index, node) in one.iter().enumerate() {
        db.create_edge(*node, two[index % 20], "Y").unwrap();
    }
    for (index, node) in last.iter().enumerate() {
        db.create_edge(*node, two[index % 20], "Z").unwrap();
    }
    let query = "MATCH (a:A)-[:X]->(m:M), (m)-[:Y]->(n:N), (b:B)-[:Z]->(n) WHERE 100 / a.k = 100 RETURN id(a), id(b)";
    let rows = same_as_written(&db, query);
    assert_ne!(rows.len(), 0);
    let session = db.session();
    let plan = plan_of(&session, query);
    let filter = line_of(&plan, "100");
    let every_hop = plan.lines().filter(|line| line.contains("Expand")).count();
    let above = plan
        .lines()
        .take(filter)
        .filter(|line| line.contains("Expand"))
        .count();
    assert!(
        above > 0,
        "every one of {every_hop} hops ran before the filter:\n{plan}"
    );
}

/// The case the ruling names: the search may start from the second hop, whose scan is not limited
/// to the label. A node without the label has the value that raises. The filter runs above the
/// label check of `a`, so it never reads that node.
#[test]
fn a_filter_does_not_overtake_the_label_it_was_written_under() {
    let db = GrafeoDB::new_in_memory();
    let people: Vec<NodeId> = (0..20).map(|key| node(&db, "Person", key + 1)).collect();
    let robots: Vec<NodeId> = (0..20).map(|_| node(&db, "Robot", 0)).collect();
    let far: Vec<NodeId> = (0..1_000).map(|key| node(&db, "Far", key + 1)).collect();
    let few = node(&db, "Few", 1);
    // Many K edges and few L edges, so that starting from the L hop is the cheap way.
    for (index, far_node) in far.iter().enumerate() {
        db.create_edge(people[index % 20], *far_node, "K").unwrap();
    }
    for person in &people {
        db.create_edge(*person, few, "L").unwrap();
    }
    for robot in &robots {
        db.create_edge(*robot, few, "L").unwrap();
        db.create_edge(*robot, far[0], "K").unwrap();
    }
    for query in [
        "MATCH (a:Person)-[:K]->(b), (a)-[:L]->(c) WHERE 100 / a.k > 0 RETURN id(a), id(b), id(c)",
        "MATCH (a:Person)-[:K]->(b:Far), (a)-[:L]->(c:Few) WHERE 100 / a.k > 0 RETURN id(a), id(b), id(c)",
        "MATCH (c:Few)<-[:L]-(a:Person), (a)-[:K]->(b) WHERE 100 / a.k > 0 RETURN id(a), id(b), id(c)",
    ] {
        same_as_written(&db, query);
        let cypher = db.session().execute_cypher(query);
        assert!(cypher.is_ok(), "Cypher: {:?}", cypher.err());
    }
}

/// Written `(a:Person)` in a `WHERE` too: a label check written first is a check written first.
#[test]
fn a_label_check_written_in_the_where_still_guards() {
    let db = GrafeoDB::new_in_memory();
    let person = node(&db, "Person", 1);
    let robot = node(&db, "Robot", 0);
    let far = node(&db, "Far", 1);
    for start in [person, robot] {
        db.create_edge(start, far, "K").unwrap();
        db.create_edge(start, far, "L").unwrap();
    }
    let query = "MATCH (a)-[:K]->(b), (a)-[:L]->(c) WHERE a:Person AND 100 / a.k > 0 RETURN id(a), id(b), id(c)";
    assert_eq!(same_as_written(&db, query).len(), 1);
}

// ---------------------------------------------------------------------------
// The order within a relation
// ---------------------------------------------------------------------------

/// Written checks only: a guard written before a raising filter on the same relation still runs
/// first, whichever part starts the plan.
#[test]
fn a_filter_keeps_its_place_among_the_relations_filters() {
    let db = far_end_graph();
    let zero = node(&db, "A", 0);
    db.create_edge(zero, node(&db, "M", 999), "X").unwrap();
    for query in [
        "MATCH (a:A)-[:X]->(m:M), (b:B)-[:Y]->(m) WHERE a.k <> 0 AND 100 / a.k > 50 RETURN id(a), id(b)",
        "MATCH (a:A {k: 3})-[:X]->(m:M), (b:B)-[:Y]->(m) WHERE 100 / a.k > 0 RETURN id(a), id(b)",
        "MATCH (a:A)-[:X]->(m:M) WHERE a.k <> 0 AND 100 / a.k > 50 MATCH (b:B)-[:Y]->(m) RETURN id(a), id(b)",
    ] {
        same_as_written(&db, query);
    }
}

/// In the plan: the two conditions in their written order, the guard below the one that raises.
#[test]
fn the_guard_stays_under_what_it_guards_in_the_plan() {
    let db = far_end_graph();
    let query = "MATCH (a:A)-[:X]->(m:M), (b:B)-[:Y]->(m) WHERE a.k <> 7 AND 100 / a.k > 50 RETURN id(a), id(b)";
    let plan = plan_of(&db.session(), query);
    let raising = line_of(&plan, "100");
    let guard = line_of(&plan, "Ne 7");
    assert!(
        guard >= raising,
        "the guard runs after what it guards:\n{plan}"
    );
}

/// A filter that reads two relations stays above both, also one that can raise.
#[test]
fn a_filter_over_two_relations_stays_above_both() {
    let db = far_end_graph();
    for (query, near, far_side) in [
        (
            "MATCH (a:A)-[:X]->(m:M), (b:B)-[:Y]->(m) WHERE 100 / (a.k + b.k) > 0 RETURN id(a), id(b)",
            "(a)",
            "(b)",
        ),
        (
            "MATCH (a:A)-[:X]->(m:M), (b:B)-[:Y]->(m) WHERE a.k < b.k AND 100 / (a.k + b.k) > 0 RETURN id(a), id(b)",
            "(a)",
            "(b)",
        ),
    ] {
        same_as_written(&db, query);
        let plan = plan_of(&db.session(), query);
        let filter = line_of(&plan, "100");
        for side in [near, far_side] {
            let lines: Vec<&str> = plan.lines().collect();
            let below = lines[filter..].iter().any(|line| {
                (line.contains("Expand") || line.contains("Scan")) && line.contains(side)
            });
            assert!(below, "{side} is not bound under the filter:\n{plan}");
        }
    }
}

/// One relation's filter and a two-relation filter together: each at its own place.
#[test]
fn filters_of_different_reach_keep_their_own_places() {
    let db = far_end_graph();
    let query = "MATCH (a:A)-[:X]->(m:M), (b:B)-[:Y]->(m) WHERE 100 / a.k = 100 AND 100 / (a.k + b.k) > 0 RETURN id(a), id(b)";
    same_as_written(&db, query);
    let plan = plan_of(&db.session(), query);
    let two = line_of(&plan, "Add").min(line_of(&plan.replace("Plus", "Add"), "Add"));
    let lines: Vec<&str> = plan.lines().collect();
    let one = lines
        .iter()
        .rposition(|line| line.contains("Filter") && line.contains("100"))
        .unwrap();
    assert!(
        two <= one,
        "the filter over both is below the one over a:\n{plan}"
    );
}

// ---------------------------------------------------------------------------
// A LIMIT
// ---------------------------------------------------------------------------

/// Under a `LIMIT` the written plan may stop before the row that raises, so a tree with a raising
/// filter under one keeps the written order, and a query that returned rows still does.
#[test]
fn a_limit_keeps_the_written_order_when_a_filter_can_raise() {
    let db = GrafeoDB::new_in_memory();
    let first = node(&db, "A", 1);
    let far = node(&db, "M", 1);
    let other = node(&db, "B", 1);
    db.create_edge(first, far, "X").unwrap();
    db.create_edge(other, far, "Y").unwrap();
    for _ in 0..200 {
        let zero = node(&db, "A", 0);
        db.create_edge(zero, far, "X").unwrap();
    }
    for query in [
        "MATCH (a:A)-[:X]->(m:M), (b:B)-[:Y]->(m) WHERE 100 / a.k > 0 RETURN id(a), id(b) LIMIT 1",
        "MATCH (a:A)-[:X]->(m:M), (b:B)-[:Y]->(m) WHERE 100 / a.k > 0 RETURN id(a), id(b) OFFSET 0 LIMIT 1",
    ] {
        let written_session = written(&db);
        let want = plan_of(&written_session, query);
        assert_eq!(plan_of(&db.session(), query), want, "{query}");
        match (
            sorted(&written_session, query),
            sorted(&db.session(), query),
        ) {
            (Ok(_), Err(error)) => {
                panic!("{query}: the written plan runs, the chosen fails: {error}")
            }
            (Ok(left), Ok(right)) => assert_eq!(left, right, "{query}"),
            _ => {}
        }
    }
}

/// With nothing that can raise a `LIMIT` does not stop the search, and any row may come back: a
/// row of the pattern, and as many as the limit asks.
#[test]
fn a_limit_with_nothing_that_raises_may_reorder() {
    let db = far_end_graph();
    let query = "MATCH (a:A)-[:X]->(m:M), (b:B)-[:Y]->(m) RETURN id(a), id(b) LIMIT 5";
    let rows = db.session().execute(query).unwrap();
    assert_eq!(rows.row_count(), 5);
}

/// The `LIMIT` of a later clause holds too: it is what stops the rows the filter reads.
#[test]
fn a_limit_after_a_with_also_counts() {
    let db = GrafeoDB::new_in_memory();
    let first = node(&db, "A", 1);
    let far = node(&db, "M", 1);
    let other = node(&db, "B", 1);
    db.create_edge(first, far, "X").unwrap();
    db.create_edge(other, far, "Y").unwrap();
    for _ in 0..50 {
        db.create_edge(node(&db, "A", 0), far, "X").unwrap();
    }
    let query = "MATCH (a:A)-[:X]->(m:M), (b:B)-[:Y]->(m) WHERE 100 / a.k > 0 WITH a, b LIMIT 1 RETURN id(a), id(b)";
    let written_session = written(&db);
    if let Ok(left) = sorted(&written_session, query) {
        assert_eq!(sorted(&db.session(), query), Ok(left), "{query}");
    }
}

// ---------------------------------------------------------------------------
// The documented exception
// ---------------------------------------------------------------------------

/// User's decision. With another part empty the plain plan's hash join sees an empty build side
/// and never reads the part with the filter, so it returns no rows. A chosen plan may run that
/// filter first and raise. Either answer is allowed here, and nothing else: no rows, or the error
/// of that filter.
#[test]
fn an_empty_other_part_may_let_a_filter_raise() {
    let db = GrafeoDB::new_in_memory();
    let zero = node(&db, "A", 0);
    let far = node(&db, "M", 1);
    db.create_edge(zero, far, "X").unwrap();
    db.create_node(&["B"]).unwrap();
    for query in [
        "MATCH (a:A)-[:X]->(m:M), (b:B)-[:Y]->(m) WHERE 100 / a.k > 0 RETURN id(a), id(b)",
        "MATCH (b:B)-[:Y]->(m:M), (a:A)-[:X]->(m) WHERE 100 / a.k > 0 RETURN id(a), id(b)",
        "MATCH (a:A)-[:X]->(m:M), (b:Nobody)-[:Y]->(m) WHERE 100 / a.k > 0 RETURN id(a), id(b)",
    ] {
        for (name, session) in [("usual", db.session()), ("plain", plain(&db))] {
            match sorted(&session, query) {
                Ok(rows) => assert!(rows.is_empty(), "{name}: {query}: {rows:?}"),
                Err(error) => assert!(
                    error.to_lowercase().contains("division")
                        || error.to_lowercase().contains("zero"),
                    "{name}: {query}: another error: {error}"
                ),
            }
        }
    }
}

/// With no part empty it is no exception: the plan that fails is the written one, never only the
/// chosen one.
#[test]
fn with_no_part_empty_the_chosen_plan_never_fails_alone() {
    let db = GrafeoDB::new_in_memory();
    let zero = node(&db, "A", 0);
    let good = node(&db, "A", 1);
    let far = node(&db, "M", 1);
    let lonely = node(&db, "M", 2);
    db.create_edge(good, far, "X").unwrap();
    db.create_edge(zero, lonely, "X").unwrap();
    db.create_edge(node(&db, "B", 1), far, "Y").unwrap();
    let query = "MATCH (a:A)-[:X]->(m:M), (b:B)-[:Y]->(m) WHERE 100 / a.k > 0 RETURN id(a), id(b)";
    // The written plan reads every A first, so it fails, and a plan reading fewer may succeed.
    match (sorted(&written(&db), query), sorted(&db.session(), query)) {
        (Ok(_), Err(error)) => panic!("only the chosen plan fails: {error}"),
        (Ok(left), Ok(right)) => assert_eq!(left, right),
        _ => {}
    }
}

// ---------------------------------------------------------------------------
// A filter is not copied to both sides of a join
// ---------------------------------------------------------------------------

/// A label on one side only, and a node without it on the other that holds the value that raises.
fn with_a_robot() -> GrafeoDB {
    let db = GrafeoDB::new_in_memory();
    let vincent = node(&db, "Person", 1);
    let jules = node(&db, "Person", 2);
    let robot = node(&db, "Robot", 0);
    let far = node(&db, "Far", 5);
    let few = node(&db, "Few", 7);
    for start in [vincent, jules, robot] {
        db.create_edge(start, far, "K").unwrap();
        db.create_edge(start, few, "L").unwrap();
    }
    db
}

/// 0.6.1 copied the filter to the side without the label, which read the robot and raised. Written
/// either way round, it gives one answer and never raises on a node that is no Person.
#[test]
fn the_side_without_the_label_does_not_raise_written_either_way() {
    let db = with_a_robot();
    let want = vec![
        "[Int64(1), Int64(5), Int64(7)]".to_string(),
        "[Int64(2), Int64(5), Int64(7)]".to_string(),
    ];
    for query in [
        "MATCH (a:Person)-[:K]->(b), (a)-[:L]->(c) WHERE 100 / a.k > 0 RETURN a.k, b.k, c.k",
        "MATCH (a)-[:L]->(c), (a:Person)-[:K]->(b) WHERE 100 / a.k > 0 RETURN a.k, b.k, c.k",
    ] {
        for (name, session) in [
            ("usual", db.session()),
            ("written", written(&db)),
            ("plain", plain(&db)),
        ] {
            assert_eq!(sorted(&session, query), Ok(want.clone()), "{name}: {query}");
            let cypher = session
                .execute_cypher(query)
                .unwrap_or_else(|error| panic!("{name}, Cypher: {query}: {error}"));
            assert_eq!(cypher.row_count(), 2, "{name}, Cypher: {query}");
        }
    }
}

/// Checks split across the two sides: the label on one, a property on the other. Each bad node
/// fails one check and passes the other, so the filter must run above both, above the join.
#[test]
fn checks_on_two_sides_keep_the_filter_above_the_join() {
    let db = GrafeoDB::new_in_memory();
    let good = db
        .create_node_with_props(
            &["Person"],
            [("k", Value::Int64(1)), ("live", Value::Bool(true))],
        )
        .unwrap();
    // A Person that is not live, and a live node that is no Person: both raise if they get there.
    let not_live = db
        .create_node_with_props(
            &["Person"],
            [("k", Value::Int64(0)), ("live", Value::Bool(false))],
        )
        .unwrap();
    let no_person = db
        .create_node_with_props(
            &["Robot"],
            [("k", Value::Int64(0)), ("live", Value::Bool(true))],
        )
        .unwrap();
    let far = node(&db, "Far", 5);
    let few = node(&db, "Few", 7);
    for start in [good, not_live, no_person] {
        db.create_edge(start, far, "K").unwrap();
        db.create_edge(start, few, "L").unwrap();
    }
    for query in [
        "MATCH (a:Person)-[:K]->(b), (a {live: true})-[:L]->(c) WHERE 100 / a.k > 0 RETURN a.k, b.k, c.k",
        "MATCH (a {live: true})-[:L]->(c), (a:Person)-[:K]->(b) WHERE 100 / a.k > 0 RETURN a.k, b.k, c.k",
    ] {
        for (name, session) in [
            ("usual", db.session()),
            ("written", written(&db)),
            ("plain", plain(&db)),
        ] {
            assert_eq!(
                sorted(&session, query),
                Ok(vec!["[Int64(1), Int64(5), Int64(7)]".to_string()]),
                "{name}: {query}"
            );
        }
    }
}

/// With nothing that can raise the filter still narrows both sides, so no query pays for this.
#[test]
fn a_filter_that_cannot_raise_still_narrows_both_sides() {
    let db = far_end_graph();
    let query = "MATCH (a:A)-[:X]->(m:M), (a)-[:Y]->(n:M) WHERE a.k = 7 RETURN id(a)";
    let plan = plan_of(&written(&db), query);
    let filters = plan
        .lines()
        .filter(|line| line.contains("a.k Eq 7"))
        .count();
    assert!(filters >= 1, "{plan}");
    same_as_written(&db, query);
}

// ---------------------------------------------------------------------------
// Written checks guard a filter that raises (the user's "written checks only")
// ---------------------------------------------------------------------------

/// A person whose age raises, and the one the written check keeps.
fn guarded_people() -> GrafeoDB {
    let db = GrafeoDB::new_in_memory();
    db.session()
        .execute(
            "INSERT (:Person {name: 'Gus', age: 40})-[:KNOWS]->(:Person {name: 'Mia', age: 30}),
                    (:Person {name: 'Hans', age: 0})-[:KNOWS]->(:Person {name: 'Jules', age: 20})",
        )
        .unwrap();
    db
}

/// A check written before the filter guards it, written as a `FILTER` or inline, on every plan.
#[test]
fn a_written_check_guards_a_filter_on_every_plan() {
    let db = guarded_people();
    for query in [
        "MATCH (a:Person)-[:KNOWS]->(b) FILTER a.name = 'Gus' FILTER 100 / a.age > 0 RETURN b.name",
        "MATCH (a:Person {name: 'Gus'})-[:KNOWS]->(b) FILTER 100 / a.age > 0 RETURN b.name",
        "MATCH (a:Person {name: 'Gus'})-[:KNOWS]->(b) WHERE 100 / a.age > 0 RETURN b.name",
    ] {
        for (name, session) in [
            ("usual", db.session()),
            ("written", written(&db)),
            ("plain", plain(&db)),
        ] {
            assert_eq!(
                sorted(&session, query),
                Ok(vec!["[String(\"Mia\")]".to_string()]),
                "{name}: {query}"
            );
        }
    }
}

/// Decided by the user: a filter may run below an expand, so with no check written before it, a
/// node with no edges may still raise. Either answer is allowed: this pins that no other is.
#[test]
fn with_no_check_written_before_it_a_filter_may_raise() {
    let db = guarded_people();
    db.session()
        .execute("INSERT (:Person {name: 'Butch', age: 0})")
        .unwrap();
    db.session()
        .execute("MATCH (a {name: 'Hans'}) DETACH DELETE a")
        .unwrap();
    let query = "MATCH (a:Person)-[:KNOWS]->(b) FILTER 100 / a.age > 0 RETURN b.name";
    for session in [db.session(), written(&db), plain(&db)] {
        match sorted(&session, query) {
            Ok(rows) => assert_eq!(rows, vec!["[String(\"Mia\")]".to_string()]),
            Err(error) => assert!(error.contains("GRAFEO-Q006"), "{error}"),
        }
    }
}
