//! Checks over several hops of a pattern run where the hops they read are bound
//! (`PlannerRule::EarlyChecks`), not above the whole join.
//!
//! Tests first, from the ruling:
//!
//! - only a check that cannot raise moves, to the lowest point where every variable it reads is
//!   bound, and never between an intersection's two hops;
//! - one that may raise stays above, in written order;
//! - nothing moves into an optional side or past a quantified hop.
//!
//! The rows are those with the rule off, as a multiset. Rows over random multigraphs, against
//! brute force, are `planner_rules_results.rs`'s, which picks the rule up from
//! `PlannerRule::ALL`.

#![cfg(all(feature = "gql", feature = "cypher"))]
#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss,
    reason = "test graphs of a few hundred nodes"
)]

use std::time::Duration;

use grafeo_common::types::{NodeId, Value};
use grafeo_engine::query::optimizer::{PlannerRule, PlannerRules};
use grafeo_engine::session::Session;
use grafeo_engine::{Config, GrafeoDB};

/// The rows are what these tests read: a slow build must not make them a test of the query
/// timeout, and a bad plan must still fail a test by name.
const SLOW_BUILD: Duration = Duration::from_secs(600);

const MARK: &str = "[IntersectExpand into:";

/// A square from `a` over its few friends `b` and `d`, closed over the many people they
/// follow: binding `b` and `d` first and intersecting their lists for `c` is the cheaper order,
/// so the plan does not hang on a tie.
const SQUARE: &str = "MATCH (a:Person {id: 3})-[:KNOWS]-(b:Person)-[:FOLLOWS]-(c:Person)-[:FOLLOWS]-(d:Person)-[:KNOWS]-(a) \
     WHERE b.id < d.id AND c.id <> 3 RETURN b.id, c.id, d.id";

/// graph-bench's square, every hop of one type: its order follows a tie of the cost model.
const BENCH_SQUARE: &str = "MATCH (a:Person {id: 3})-[:KNOWS]-(b:Person)-[:KNOWS]-(c:Person)-[:KNOWS]-(d:Person)-[:KNOWS]-(a) \
     WHERE b.id < d.id AND c.id <> 3 RETURN b.id, c.id, d.id";

/// `size` people, each who knows `knows` others and follows `follows` others, at random:
/// self-loops and parallel edges among them.
fn people(size: u64, knows: u64, follows: u64) -> GrafeoDB {
    let db = GrafeoDB::with_config(Config::in_memory().with_query_timeout(SLOW_BUILD)).unwrap();
    let ids: Vec<NodeId> = (0..size)
        .map(|id| {
            db.create_node_with_props(&["Person"], [("id", Value::Int64(id as i64))])
                .unwrap()
        })
        .collect();
    let mut state = 17_u64;
    let mut next = |bound: u64| {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        (state >> 33) % bound
    };
    for id in 0..size {
        for (kind, count) in [("KNOWS", knows), ("FOLLOWS", follows)] {
            for _ in 0..count {
                db.create_edge(ids[id as usize], ids[next(size) as usize], kind)
                    .unwrap();
            }
        }
    }
    db
}

fn without(db: &GrafeoDB) -> Session {
    with_off(db, PlannerRule::EarlyChecks)
}

/// A session with every rule but `rule`.
fn with_off(db: &GrafeoDB, rule: PlannerRule) -> Session {
    let mut session = db.session();
    session.set_planner_rules(PlannerRules::all().with(rule, false));
    session
}

fn run(session: &Session, cypher: bool, query: &str) -> grafeo_engine::database::QueryResult {
    if cypher {
        session.execute_cypher(query)
    } else {
        session.execute(query)
    }
    .unwrap_or_else(|error| panic!("{query}: {error}"))
}

fn text(session: &Session, cypher: bool, query: &str) -> String {
    run(session, cypher, query)
        .rows()
        .iter()
        .map(|row| match &row[0] {
            Value::String(line) => line.to_string(),
            other => format!("{other:?}"),
        })
        .collect::<Vec<_>>()
        .join("\n")
}

fn explain(session: &Session, cypher: bool, query: &str) -> String {
    text(session, cypher, &format!("EXPLAIN {query}"))
}

fn rows(session: &Session, cypher: bool, query: &str) -> Vec<String> {
    let mut rows: Vec<String> = run(session, cypher, query)
        .rows()
        .iter()
        .map(|row| format!("{row:?}"))
        .collect();
    rows.sort();
    rows
}

/// The line of `plan` that holds every one of `words`, the first from the top.
fn line_of(plan: &str, words: &[&str]) -> Option<usize> {
    plan.lines()
        .position(|line| words.iter().all(|word| line.contains(word)))
}

/// The rows with the rule on are those with it off.
fn same_rows(db: &GrafeoDB, cypher: bool, query: &str) {
    let (on, off) = (db.session(), without(db));
    let rows_on = rows(&on, cypher, query);
    assert!(!rows_on.is_empty(), "the test needs rows: {query}");
    assert_eq!(rows_on, rows(&off, cypher, query), "{query}");
}

/// The plan of `query` is not the written one: the join search took the pattern.
fn searched(db: &GrafeoDB, cypher: bool, query: &str, plan: &str) {
    assert_ne!(
        plan,
        explain(&with_off(db, PlannerRule::JoinOrder), cypher, query),
        "the join search must take the pattern, cypher {cypher}"
    );
}

/// The square binds `b` and `d` before it intersects their lists for `c`: the check on the two
/// runs before the intersection, so it intersects only the pairs that pass. In Cypher the check
/// that the `a-b` and `a-d` edges differ moves with it. graph-bench's text, whose order follows
/// a tie, gives the rows of the rule off.
#[test]
fn the_square_checks_its_bound_ends_before_the_intersection() {
    let db = people(200, 4, 12);
    for cypher in [false, true] {
        let on = explain(&db.session(), cypher, SQUARE);
        let off = explain(&without(&db), cypher, SQUARE);
        for (plan, moved) in [(&on, true), (&off, false)] {
            let mark = line_of(plan, &[MARK]).unwrap_or_else(|| panic!("no intersection:\n{plan}"));
            let check = line_of(plan, &["Filter", "b.id Lt d.id"])
                .unwrap_or_else(|| panic!("no check:\n{plan}"));
            assert_eq!(check > mark, moved, "cypher {cypher}:\n{plan}");
        }
        if cypher {
            let mark = line_of(&on, &[MARK]).unwrap();
            assert!(
                on.lines()
                    .skip(mark + 1)
                    .any(|line| line.contains("Filter") && line.contains("Ne Id(")),
                "an edge check runs below the intersection:\n{on}"
            );
        }
        same_rows(&db, cypher, SQUARE);
        same_rows(&db, cypher, BENCH_SQUARE);
    }
}

/// A check that reads three hops' ends, all bound before the intersection, runs before it.
#[test]
fn a_check_over_three_hops_runs_where_they_are_bound() {
    let db = people(200, 4, 12);
    let query = "MATCH (a:Person {id: 3})-[:KNOWS]-(b:Person)-[:FOLLOWS]-(c:Person)-[:FOLLOWS]-(d:Person)-[:KNOWS]-(a) \
         WHERE a.id < b.id OR b.id < d.id RETURN b.id, c.id, d.id";
    let plan = explain(&db.session(), false, query);
    let mark = line_of(&plan, &[MARK]).unwrap_or_else(|| panic!("no intersection:\n{plan}"));
    let check = line_of(&plan, &["Filter", "a.id Lt b.id"]).expect(&plan);
    assert!(check > mark, "{plan}");
    same_rows(&db, false, query);
}

/// A check over two hops that may raise stays above the join, where it is written; the one
/// beside it that cannot raise moves.
#[test]
fn a_raising_check_stays_above_beside_one_that_moves() {
    let db = people(200, 4, 12);
    let query = "MATCH (a:Person {id: 3})-[:KNOWS]-(b:Person)-[:FOLLOWS]-(c:Person)-[:FOLLOWS]-(d:Person)-[:KNOWS]-(a) \
         WHERE b.id < d.id AND 1000 / (b.id - d.id + 1000) > 0 RETURN b.id, c.id, d.id";
    for cypher in [false, true] {
        let plan = explain(&db.session(), cypher, query);
        let mark = line_of(&plan, &[MARK]).unwrap_or_else(|| panic!("no intersection:\n{plan}"));
        let division = line_of(&plan, &["Div"]).expect(&plan);
        let check = line_of(&plan, &["Filter", "b.id Lt d.id"]).expect(&plan);
        assert!(division < mark, "the division stays above:\n{plan}");
        assert!(check > mark, "the comparison moves:\n{plan}");
        same_rows(&db, cypher, query);
    }
}

/// A check that raises on every row raises the same error with the rule on and off, while the
/// one beside it moves.
#[test]
fn a_check_that_raises_raises_alike_on_and_off() {
    let db = people(200, 4, 12);
    let query = "MATCH (a:Person {id: 3})-[:KNOWS]-(b:Person)-[:FOLLOWS]-(c:Person)-[:FOLLOWS]-(d:Person)-[:KNOWS]-(a) \
         WHERE b.id < d.id AND 1000 / (b.id * 0 + d.id * 0) > 0 RETURN b.id, c.id, d.id";
    for cypher in [false, true] {
        let failed = |session: &Session| {
            let result = if cypher {
                session.execute_cypher(query)
            } else {
                session.execute(query)
            };
            result.err().map(|error| error.to_string())
        };
        let on = failed(&db.session()).expect("it divides by zero with the rule on");
        assert_eq!(Some(on), failed(&without(&db)), "cypher {cypher}");
    }
}

/// An optional match whose condition reads both sides keeps that condition on the optional
/// join, over a pattern the join search reorders; the pattern's own checks may move inside it.
#[test]
fn an_optional_match_reading_both_sides_keeps_its_filter() {
    let db = people(200, 4, 12);
    let query = "MATCH (a:Person {id: 3})-[:KNOWS]-(b:Person), (a)-[:KNOWS]-(d:Person), \
         (b)-[:FOLLOWS]-(c:Person), (d)-[:FOLLOWS]-(c) \
         OPTIONAL MATCH (b)-[:KNOWS]-(e:Person) WHERE e.id < d.id RETURN b.id, c.id, d.id, e.id";
    for cypher in [false, true] {
        let on = explain(&db.session(), cypher, query);
        searched(&db, cypher, query, &on);
        for plan in [&on, &explain(&without(&db), cypher, query)] {
            let reading_both: Vec<&str> = plan
                .lines()
                .filter(|line| line.contains("op: Lt") || line.contains("e.id Lt"))
                .collect();
            assert!(
                reading_both.len() == 1 && reading_both[0].contains("LeftJoin"),
                "the condition stays on the optional join, cypher {cypher}:\n{plan}"
            );
        }
        same_rows(&db, cypher, query);
    }
}

/// A quantified hop in a pattern the join search takes keeps the plan of the rule off: the
/// check stays above, though the side without the quantified hop binds what it reads.
#[test]
fn a_quantified_hop_is_not_moved_past() {
    let db = people(200, 4, 12);
    let query = "MATCH (a:Person {id: 3})-[:KNOWS]-(b:Person), (b)-[:FOLLOWS]-{1,2}(c:Person), \
         (a)-[:KNOWS]-(d:Person), (d)-[:FOLLOWS]-(c) WHERE b.id < d.id RETURN b.id, c.id, d.id";
    let on = explain(&db.session(), false, query);
    searched(&db, false, query, &on);
    assert_eq!(on, explain(&without(&db), false, query));
    same_rows(&db, false, query);
}

/// A check that reads the node the near hop finds never goes between the two hops that run as
/// one intersection: above them, or, in an order that binds that node lower, below both.
#[test]
fn a_check_on_the_node_met_never_goes_between_the_hops() {
    let db = people(200, 4, 12);
    let query = "MATCH (a:Person {id: 3})-[:KNOWS]-(b:Person)-[:FOLLOWS]-(c:Person)-[:FOLLOWS]-(d:Person)-[:KNOWS]-(a) \
         WHERE b.id < c.id RETURN b.id, c.id, d.id";
    for cypher in [false, true] {
        let plan = explain(&db.session(), cypher, query);
        let mark = line_of(&plan, &[MARK]).unwrap_or_else(|| panic!("no intersection:\n{plan}"));
        let check = line_of(&plan, &["Filter", "b.id Lt c.id"]).expect(&plan);
        // The closing hop is right under the mark, the hop it closes over the next expand.
        let near = plan
            .lines()
            .enumerate()
            .skip(mark + 2)
            .find(|(_, line)| line.contains("Expand ("))
            .map(|(at, _)| at)
            .expect(&plan);
        assert!(
            !(mark < check && check < near),
            "the check sits between the hops, cypher {cypher}:\n{plan}"
        );
        same_rows(&db, cypher, query);
    }
}

/// PROFILE shows the check under the intersection. `rows_scanned` is never higher with the rule
/// on: lower on the square, whose check moves, the same on the triangle, whose check reads the
/// node it meets at.
#[test]
fn profile_shows_the_check_under_the_intersection_and_fewer_reads() {
    let db = people(200, 4, 12);
    let on = db.session();
    let off = without(&db);
    let profile = text(&on, false, &format!("PROFILE {SQUARE}"));
    let intersection = line_of(&profile, &["IntersectExpand"]).expect(&profile);
    let check = line_of(&profile, &["Filter", "b.id Lt d.id"]).expect(&profile);
    assert!(check > intersection, "{profile}");

    let reads = |session: &Session, query: &str| {
        run(session, false, query)
            .rows_scanned()
            .expect("the read counter")
    };
    let (square_on, square_off) = (reads(&on, SQUARE), reads(&off, SQUARE));
    assert!(square_on < square_off, "{square_on} against {square_off}");
    let triangle = "MATCH (a:Person {id: 3})-[:KNOWS]-(b:Person)-[:KNOWS]-(c:Person)-[:KNOWS]-(a) \
         WHERE b.id < c.id RETURN b.id, c.id";
    assert_eq!(reads(&on, triangle), reads(&off, triangle));
    same_rows(&db, false, triangle);
}

/// graph-bench's SQL/PGQ triangle and square start at the person their id pins, read by its key,
/// with the rule on and off: under SQL/PGQ a comparison of a property may raise (22G04), and the
/// placement must not cost the pinned start its place.
#[cfg(feature = "sql-pgq")]
#[test]
fn sql_cycles_keep_their_pinned_start() {
    use std::collections::HashMap;

    let db = people(200, 4, 12);
    let params = HashMap::from([("personId".to_string(), Value::Int64(3))]);
    let triangle = "SELECT count(*) AS triangles FROM GRAPH_TABLE (\
         MATCH (a:Person)-[k1:KNOWS]-(b:Person)-[k2:KNOWS]-(c:Person)-[k3:KNOWS]-(a:Person) \
         WHERE a.id = $personId AND b.id < c.id COLUMNS (c.id AS cid))";
    let square = "SELECT count(*) AS cycles FROM GRAPH_TABLE (\
         MATCH (a:Person)-[k1:KNOWS]-(b:Person)-[k2:KNOWS]-(c:Person)-[k3:KNOWS]-(d:Person)-[k4:KNOWS]-(a:Person) \
         WHERE a.id = $personId AND b.id < d.id AND c.id <> $personId COLUMNS (c.id AS cid))";
    for query in [triangle, square] {
        for session in [db.session(), without(&db)] {
            let plan: Vec<String> = session
                .execute_sql_with_params(&format!("EXPLAIN {query}"), params.clone())
                .unwrap()
                .rows()
                .iter()
                .map(|row| match &row[0] {
                    Value::String(line) => line.to_string(),
                    other => format!("{other:?}"),
                })
                .collect();
            let plan = plan.join("\n");
            let lines: Vec<&str> = plan.lines().collect();
            let pinned = lines.windows(2).any(|pair| {
                pair[0].contains("a.id Eq 3")
                    && (pair[0].contains("[key column") || pair[0].contains("[index"))
                    && pair[1].contains("NodeScan (a:Person)")
            });
            assert!(pinned, "the start at a is read by its key:\n{plan}");
        }
        let count = |session: Session| {
            session
                .execute_sql_with_params(query, params.clone())
                .unwrap()
                .rows()
                .to_vec()
        };
        assert_eq!(count(db.session()), count(without(&db)), "{query}");
    }
}
