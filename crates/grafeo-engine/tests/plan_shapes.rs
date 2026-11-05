//! The plans the planner rules promise, as `EXPLAIN` prints them.
//!
//! `planner_rules_results.rs` proves that a rule keeps the rows; this file that it fires
//! where it should and nowhere else, that the switch reaches it, and that one query gets
//! one plan in every process.

#![cfg(all(feature = "gql", feature = "cypher"))]

use std::fmt::Write as _;

use grafeo_common::types::Value;
use grafeo_engine::GrafeoDB;
use grafeo_engine::query::optimizer::{PlannerRule, PlannerRules};
use grafeo_engine::session::Session;

const PEOPLE: &str =
    "INSERT (a:Person {name: 'Alix', k: 0})-[:KNOWS]->(g:Person {name: 'Gus', k: 1}),
        (g)-[:KNOWS]->(v:Person {name: 'Vincent', k: 2}),
        (v)-[:KNOWS]->(a),
        (v)-[:LIKES]->(a),
        (g)-[:KNOWS]->(:Person {name: 'Jules', k: 3}),
        (m:Person {name: 'Mia', k: 4})-[:KNOWS]->(m),
        (a)-[:KNOWS]->(m),
        (:City {name: 'Amsterdam'})";

fn db() -> GrafeoDB {
    let db = GrafeoDB::new_in_memory();
    db.session().execute(PEOPLE).unwrap();
    db
}

fn text(result: &grafeo_engine::database::QueryResult) -> String {
    let mut out = String::new();
    for row in result.rows() {
        for value in row {
            match value {
                Value::String(line) => out.push_str(line),
                other => write!(out, "{other:?}").unwrap(),
            }
            out.push('\n');
        }
    }
    out
}

fn explain(session: &Session, query: &str) -> String {
    text(&session.execute(&format!("EXPLAIN {query}")).unwrap())
}

/// What a plan shows of a query: its logical plan and what runs.
fn shown(session: &Session, query: &str) -> String {
    let profile = text(&session.execute(&format!("PROFILE {query}")).unwrap());
    format!("{}\n{profile}", explain(session, query))
}

/// A session that plans closing hops by lookup: every rule but the intersection, which takes the
/// same shapes when it applies (`intersect_expand.rs`).
fn looking_up(db: &GrafeoDB) -> Session {
    with_rules(db, without_intersection())
}

fn without_intersection() -> PlannerRules {
    PlannerRules::all().with(PlannerRule::IntersectExpand, false)
}

fn with_rules(db: &GrafeoDB, rules: PlannerRules) -> Session {
    let mut session = db.session();
    session.set_planner_rules(rules);
    session
}

// ---------------------------------------------------------------------------
// ExpandInto
// ---------------------------------------------------------------------------

/// Hops that come back to a node their input binds, and the variable each comes back to.
const CLOSING_HOPS: [(&str, &str); 15] = [
    ("MATCH (a)-[:KNOWS]->(a) RETURN a", "a"),
    ("MATCH (a)-[:KNOWS]->(b)-[:KNOWS]->(a) RETURN a", "a"),
    (
        "MATCH (a)-[:KNOWS]->(b)-[:KNOWS]->(c)-[:KNOWS]->(a) RETURN a",
        "a",
    ),
    (
        "MATCH (a)-[:KNOWS]->(b)-[:KNOWS]->(c)-[:KNOWS]->(d)-[:KNOWS]->(a) RETURN a",
        "a",
    ),
    ("MATCH (a)-[:KNOWS]->(b)<-[:KNOWS]-(a) RETURN a", "a"),
    ("MATCH (a)-[:KNOWS]->(b)-[:KNOWS]-(a) RETURN a", "a"),
    ("MATCH (a)-[:KNOWS]->(b)-[]->(a) RETURN a", "a"),
    ("MATCH (a)-[:KNOWS]->(b)-[:KNOWS|LIKES]->(a) RETURN a", "a"),
    ("MATCH (a)-[:KNOWS]->(b)-[e:KNOWS]->(a) RETURN e", "a"),
    (
        "MATCH (a)-[:KNOWS]->(b)-[:KNOWS]->(c) WHERE c = a RETURN a",
        "a",
    ),
    (
        "MATCH (a)-[:KNOWS]->(b)-[:KNOWS]->(c) WHERE a = c RETURN a",
        "a",
    ),
    (
        "MATCH (a)-[:KNOWS]->(b)-[:KNOWS]->(c) WHERE c = a AND c.k > 1 RETURN a",
        "a",
    ),
    (
        "MATCH (a)-[:KNOWS]->(b)-[:KNOWS]->(c) FILTER c.k > 1 FILTER c = a RETURN a",
        "a",
    ),
    (
        "MATCH (a)-[:KNOWS]->(b) MATCH (b)-[:KNOWS]->(a) RETURN a",
        "a",
    ),
    (
        "MATCH (a:Person) OPTIONAL MATCH (a)-[:KNOWS]->(b)-[:KNOWS]->(a) RETURN a",
        "a",
    ),
];

#[test]
fn a_hop_back_to_a_bound_node_looks_it_up() {
    let db = db();
    let usual = looking_up(&db);
    let without = with_rules(
        &db,
        without_intersection().with(PlannerRule::ExpandInto, false),
    );
    let alone = with_rules(
        &db,
        PlannerRules::plain().with(PlannerRule::ExpandInto, true),
    );
    let plain = with_rules(&db, PlannerRules::plain());
    for (query, target) in CLOSING_HOPS {
        // The search chooses which node of the cycle it closes on.
        let _ = target;
        let hint = "[expand into:".to_string();
        assert!(
            explain(&usual, query).contains(&hint),
            "{query}\n{}",
            explain(&usual, query)
        );
        assert!(
            explain(&alone, query).contains(&hint),
            "{query}\n{}",
            explain(&alone, query)
        );
        assert!(!explain(&without, query).contains("expand into"), "{query}");
        assert!(!explain(&plain, query).contains("expand into"), "{query}");
        // The check stays: it answers for a row whose target is no node.
        assert!(explain(&usual, query).contains(" Eq "), "{query}");
    }
}

/// A target that passes through operators that keep it a node keeps the lookup: without
/// one, the query is right and only slower, so nothing else would notice.
#[test]
fn a_target_through_other_operators_keeps_the_lookup() {
    let db = db();
    let usual = looking_up(&db);
    for query in [
        "MATCH (a:Person)-[:KNOWS]->(m) WITH a, m MATCH (m)-[:KNOWS]->(a) RETURN a, m",
        "MATCH (a:Person)-[:KNOWS]->(m) WITH * MATCH (m)-[:KNOWS]->(a) RETURN a, m",
        "MATCH (p:Person)-[:KNOWS]->(m) WITH p AS a, m MATCH (m)-[:KNOWS]->(a) RETURN a, m",
        "MATCH (p:Person)-[:KNOWS]->(m) LET a = p MATCH (m)-[:KNOWS]->(a) RETURN a, m",
        "MATCH (a:Person)-[:KNOWS]->(m) WITH a, m WHERE a.k > 1 MATCH (m)-[:KNOWS]->(a) RETURN a, m",
        "MATCH (a:Person)-[:KNOWS]->(m) RETURN a AS a, m AS m NEXT MATCH (m)-[:KNOWS]->(a) RETURN a, m",
        "MATCH (m:Person), (a:Person) MATCH (m)-[:KNOWS]->(a) RETURN a, m",
        "MATCH (a:Person)-[:KNOWS]->(m) WITH a, m ORDER BY a.k MATCH (m)-[:KNOWS]->(a) RETURN a, m",
        "MATCH (a:Person)-[:KNOWS]->(m) WITH a, m ORDER BY a.k LIMIT 5 MATCH (m)-[:KNOWS]->(a) RETURN a, m",
        "MATCH (a:Person)-[:KNOWS]->(m) WITH a, m SKIP 1 MATCH (m)-[:KNOWS]->(a) RETURN a, m",
        "MATCH (a:Person)-[:KNOWS]->(m) WITH DISTINCT a, m MATCH (m)-[:KNOWS]->(a) RETURN a, m",
        "MATCH (a:Person)-[:KNOWS]->(m) WITH a, m, count(*) AS n MATCH (m)-[:KNOWS]->(a) RETURN a, m, n",
        "MATCH (a:Person) CALL { WITH a MATCH (a)-[:KNOWS]->(m) RETURN m } MATCH (m)-[:KNOWS]->(a) RETURN a, m",
        // A subquery: its start is a node of the row.
        "MATCH (a:Person) WHERE EXISTS { MATCH (a)-[:KNOWS]->(b)-[:KNOWS]->(a) } RETURN a",
        "MATCH (a:Person) WHERE NOT EXISTS { MATCH (a)-[:KNOWS]->(b)-[:KNOWS]->(c)-[:KNOWS]->(a) } RETURN a",
    ] {
        let plan = explain(&usual, query);
        assert!(plan.contains("expand into"), "{query}\n{plan}");
    }
}

/// The same query in one database, on and off and on again: the plan cache keeps them apart.
#[test]
fn the_plan_follows_the_session_rules_in_any_order() {
    let db = db();
    let (query, _) = CLOSING_HOPS[2];
    let mut session = db.session();
    for round in 0..3 {
        session.set_planner_rules(without_intersection());
        assert!(
            explain(&session, query).contains("expand into"),
            "round {round}"
        );
        session.set_planner_rules(PlannerRules::plain());
        assert!(
            !explain(&session, query).contains("expand into"),
            "round {round}"
        );
        // Running it fills the cache for the next round.
        session.execute(query).unwrap();
    }
}

/// `EXPLAIN` prints the logical plan; this looks at what really runs: the lookup reads the
/// one edge back, not the hub's list, also from a cached and a prepared plan.
#[test]
fn the_lookup_reads_the_edges_back_and_no_others() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    session
        .execute("INSERT (a:Person {name: 'Alix'})-[:KNOWS]->(h:Hub), (h)-[:KNOWS]->(a)")
        .unwrap();
    session
        .execute("MATCH (h:Hub) UNWIND range(1, 2000) AS i INSERT (h)-[:KNOWS]->(:Item)")
        .unwrap();
    let query = "MATCH (a:Person)-[:KNOWS]->(h)-[:KNOWS]->(a) RETURN a.name";
    // As written, so that the hop back runs from the hub: the search would start elsewhere.
    let written = without_intersection().with(PlannerRule::JoinOrder, false);
    let session = with_rules(&db, written);
    let without = with_rules(&db, written.with(PlannerRule::ExpandInto, false));
    let scanned = |session: &Session| {
        let result = session.execute(query).unwrap();
        assert_eq!(result.row_count(), 1);
        result.rows_scanned().unwrap()
    };
    assert!(
        scanned(&without) >= 2000,
        "the list read {}",
        scanned(&without)
    );
    // Twice: the first plans it, the second reads the cache.
    assert!(
        scanned(&session) < 50,
        "the lookup read {}",
        scanned(&session)
    );
    assert!(
        scanned(&session) < 50,
        "the cached lookup read {}",
        scanned(&session)
    );
    let parameters = std::collections::HashMap::from([("name".to_string(), Value::from("Alix"))]);
    let prepared = "MATCH (a:Person {name: $name})-[:KNOWS]->(h)-[:KNOWS]->(a) RETURN a.name";
    for _ in 0..2 {
        let result = session
            .execute_with_params(prepared, parameters.clone())
            .unwrap();
        assert_eq!(result.row_count(), 1);
        assert!(
            result.rows_scanned().unwrap() < 50,
            "{:?}",
            result.rows_scanned()
        );
    }
}

/// Shapes it must leave alone: each would give wrong rows, or says nothing about the target.
#[test]
fn a_hop_that_closes_nothing_lists_its_edges() {
    let db = db();
    let usual = db.session();
    for query in [
        "MATCH (a)-[:KNOWS]->(b)-[:KNOWS]->(c) RETURN a, b, c",
        "MATCH (a)-[:KNOWS]->(b)-[:KNOWS]->(c) WHERE c = a OR c.k = 1 RETURN a",
        "MATCH (a)-[:KNOWS]->(b)-[:KNOWS]->(c) WHERE NOT c = a RETURN a",
        "MATCH (a)-[:KNOWS]->(b)-[:KNOWS]->(c) WHERE c <> a RETURN a",
        "MATCH (a)-[:KNOWS]->(b)-[:KNOWS]->(c) WHERE (c = a) IS NULL RETURN a",
        "MATCH (a)-[:KNOWS]->(b)-[:KNOWS]->(c) WHERE c.k = a.k RETURN a",
        // A quantified hop walks more than one edge.
        "MATCH (a)-[:KNOWS]->{1,3}(a) RETURN a",
        // Below a write.
        "MATCH (a:Person {k: 0}) INSERT (a)-[:KNOWS]->(a) WITH a MATCH (a)-[:KNOWS]->(b)-[:KNOWS]->(a) RETURN b",
    ] {
        assert!(
            !explain(&usual, query).contains("expand into"),
            "{query}\n{}",
            explain(&usual, query)
        );
    }
}

// ---------------------------------------------------------------------------
// The switch reaches each rule
// ---------------------------------------------------------------------------

/// A query each rule changes, and what its plan or profile shows with the rule on.
const RULE_SHAPES: [(PlannerRule, &str, &str); 6] = [
    (
        PlannerRule::PinnedFarEnd,
        "MATCH (a:Person)-[:KNOWS]->(b:Person) WHERE id(b) = 1 RETURN a.name",
        "expand to key",
    ),
    (
        PlannerRule::SoughtEnd,
        "MATCH (a)-[:KNOWS]->(b) WHERE id(b) = 1 RETURN a.name",
        "(b)<-",
    ),
    (
        PlannerRule::FactorizedExecution,
        "MATCH (:Person)-[:KNOWS]->(:Person)-[:KNOWS]->() RETURN count(*)",
        "EdgeCount",
    ),
    (
        PlannerRule::LabelCount,
        "MATCH (n:Person) RETURN count(n)",
        "LabelCount",
    ),
    (
        PlannerRule::Reachability,
        "MATCH (a:Person {k: 0})-[:KNOWS]->{1,4}(b) RETURN DISTINCT b.name",
        "[reachability",
    ),
    (
        PlannerRule::ExpandInto,
        "MATCH (a)-[:KNOWS]->(b)-[:KNOWS]->(a) RETURN a",
        "expand into",
    ),
];

#[test]
fn turning_a_rule_off_changes_its_plan_alone() {
    let db = db();
    let mut wrong = Vec::new();
    for (rule, query, mark) in RULE_SHAPES {
        // The lookup shows only where the intersection does not take the hop.
        // The far key shows only where the hop does not start at the far end.
        let start = match rule {
            PlannerRule::ExpandInto => without_intersection(),
            PlannerRule::PinnedFarEnd => PlannerRules::all().with(PlannerRule::SoughtEnd, false),
            _ => PlannerRules::all(),
        };
        let usual = with_rules(&db, start);
        let without = with_rules(&db, start.with(rule, false));
        let alone = with_rules(&db, PlannerRules::plain().with(rule, true));
        let plain = with_rules(&db, PlannerRules::plain());
        for (name, session, expected) in [
            ("on", &usual, true),
            ("off", &without, false),
            ("alone", &alone, true),
            ("no rule", &plain, false),
        ] {
            let plan = shown(session, query);
            if plan.contains(mark) != expected {
                wrong.push(format!("{rule:?} {name}: {query}\n{plan}"));
            }
        }
    }
    assert!(wrong.is_empty(), "{}", wrong.join("\n"));
}

/// The plain plan of a chain over written labels is the chain as written, whatever the
/// statistics say: each hop in text order, from the first variable.
#[test]
fn without_join_order_the_comma_patterns_run_as_written() {
    let db = db();
    let session = db.session();
    session
        .execute("MATCH (m:Person {k: 4}) UNWIND range(1, 50) AS i INSERT (m)-[:LIKES]->(:Tag)")
        .unwrap();
    let query = "MATCH (t:Tag), (p:Person) WHERE t.k = p.k RETURN t, p";
    let written = with_rules(&db, PlannerRules::plain());
    let plan = explain(&written, query);
    let tag = plan.find("NodeScan (t:Tag)").expect(&plan);
    let person = plan.find("NodeScan (p:Person)").expect(&plan);
    // A plan prints from the root: the scan below runs first.
    assert!(person < tag, "{plan}");
}

// ---------------------------------------------------------------------------
// One plan in every process
// ---------------------------------------------------------------------------

const PROCESS_ENV: &str = "GRAFEO_PLAN_SHAPES_CHILD";

/// Every plan of the corpus: a map's order changes per process, a plan must not.
fn all_plans() -> String {
    let db = db();
    db.create_property_index("k").unwrap();
    let session = db.session();
    let mut out = String::new();
    for (query, _) in CLOSING_HOPS {
        out.push_str(&explain(&session, query));
    }
    for query in [
        "MATCH (a)-[:KNOWS]->(b), (b)-[:KNOWS]->(c), (c)-[:KNOWS]->(a) RETURN a, b, c",
        "MATCH (a), (b), (c) WHERE a.k = b.k AND b.k = c.k AND c.k = a.k RETURN a",
        "MATCH (a:Person)-[:KNOWS]->(b), (a)-[:LIKES]->(c), (b)-[:KNOWS]->(c) RETURN a",
        "MATCH (a:Person {k: 1})-[:KNOWS]->(c)-[:KNOWS]->(b:Person {k: 2}) RETURN c",
        "MATCH (a:Person) WHERE a.k > 1 AND a.name <> 'x' AND a.k < 9 RETURN a.name, a.k ORDER BY a.k",
    ] {
        out.push_str(&explain(&session, query));
    }
    #[cfg(all(feature = "sparql", feature = "triple-store"))]
    out.push_str(&sparql_plans());
    out
}

#[cfg(all(feature = "sparql", feature = "triple-store"))]
fn sparql_plans() -> String {
    let db = GrafeoDB::with_config(
        grafeo_engine::Config::in_memory().with_graph_model(grafeo_engine::GraphModel::Rdf),
    )
    .unwrap();
    let session = db.session();
    session
        .execute_sparql(
            "INSERT DATA {
               <urn:alix> <urn:knows> <urn:gus> . <urn:gus> <urn:knows> <urn:vincent> .
               <urn:vincent> <urn:knows> <urn:alix> . <urn:alix> <urn:likes> <urn:gus> .
               <urn:gus> <urn:name> \"Gus\" . <urn:alix> <urn:name> \"Alix\" .
             }",
        )
        .unwrap();
    let mut out = String::new();
    for query in [
        "SELECT * WHERE { ?a <urn:knows> ?b . ?b <urn:knows> ?c . ?c <urn:knows> ?a }",
        "SELECT * WHERE { ?a <urn:knows> ?b . ?a <urn:likes> ?b . ?a <urn:name> ?n . ?b <urn:name> ?m }",
        "SELECT * WHERE { ?a <urn:knows> ?b . ?c <urn:likes> ?d }",
    ] {
        out.push_str(&text(
            &session.execute_sparql(&format!("EXPLAIN {query}")).unwrap(),
        ));
    }
    out
}

#[test]
fn one_query_gets_one_plan_in_every_process() {
    if std::env::var_os(PROCESS_ENV).is_some() {
        print!("PLANS-BEGIN\n{}\nPLANS-END\n", all_plans());
        return;
    }
    // Also within a process.
    let here = all_plans();
    assert_eq!(all_plans(), here);
    let mut seen = Vec::new();
    for _ in 0..4 {
        let output = grafeo_common::testing::child_process::output(
            grafeo_common::child_test!("one_query_gets_one_plan_in_every_process")
                .env(PROCESS_ENV, "1"),
        )
        .unwrap();
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
        let out = String::from_utf8_lossy(&output.stdout).to_string();
        let begin = out.find("PLANS-BEGIN\n").unwrap() + "PLANS-BEGIN\n".len();
        let end = out.find("\nPLANS-END").unwrap();
        seen.push(out[begin..end].to_string());
    }
    for plans in &seen {
        assert_eq!(plans, &here);
    }
}

// ---------------------------------------------------------------------------
// With a rule off, a scan and a filter
// ---------------------------------------------------------------------------

/// A database with an index on `k` and values on `v` that no index covers.
fn indexed() -> GrafeoDB {
    let db = GrafeoDB::new_in_memory();
    db.create_property_index("k").unwrap();
    db.session()
        .execute(
            "INSERT (a:Person {name: 'Alix', k: 0, v: 30})-[:KNOWS]->(g:Person {name: 'Gus', k: 1, v: 40}),
                    (g)-[:KNOWS]->(v:Person {name: 'Vincent', k: 2, v: 50}),
                    (v)-[:KNOWS]->(a),
                    (g)-[:KNOWS]->(:Person {name: 'Jules', k: 3, v: 20}),
                    (m:Person {name: 'Mia', k: 4, v: 20})-[:KNOWS]->(m)",
        )
        .unwrap();
    db
}

/// A rule, a query it answers its own way, what `PROFILE` runs then, and the hint `EXPLAIN`
/// shows for it (empty when none).
const WAYS_IN: [(PlannerRule, &str, &str, &str); 7] = [
    (
        PlannerRule::IndexLookup,
        "MATCH (n:Person) WHERE n.k = 1 RETURN n.name",
        "NodeList",
        "[index: k]",
    ),
    (
        PlannerRule::IndexLookup,
        "MATCH (n:Person) WHERE n.k IN [1, 2] RETURN n.name",
        "NodeList",
        "[index",
    ),
    (
        PlannerRule::IndexRange,
        "MATCH (n:Person) WHERE n.k > 1 RETURN n.name",
        "RangeScan",
        "[range",
    ),
    (
        PlannerRule::KeySeek,
        "UNWIND [1, 2] AS x MATCH (n:Person {k: x}) RETURN n.name",
        "NodeSeek",
        "[index: k]",
    ),
    (
        PlannerRule::IdSeek,
        "MATCH (n) WHERE id(n) = 1 RETURN n.name",
        "NodeSeek",
        "[seek: id]",
    ),
    (
        PlannerRule::ZoneMaps,
        "MATCH (n:Person) WHERE n.v = 1000 RETURN n.name",
        "Empty",
        "",
    ),
    (
        PlannerRule::ValueJoin,
        "MATCH (a:Person), (b:Person) WHERE a.v = b.v RETURN a.name, b.name",
        "HashJoin",
        "[hash join",
    ),
];

/// The edge check stops at a node's first edge; without it the subquery lists them all. No
/// plan shows which ran, so this reads the counter.
#[test]
fn the_exists_edge_check_stops_at_the_first_edge() {
    let db = indexed();
    db.session()
        .execute("MATCH (h:Person {k: 1}) UNWIND range(1, 2000) AS i INSERT (h)-[:KNOWS]->(:Item)")
        .unwrap();
    let without = with_rules(
        &db,
        PlannerRules::all().with(PlannerRule::ExistsEdgeCheck, false),
    );
    let plain = with_rules(&db, PlannerRules::plain());
    let usual = db.session();
    for query in [
        "MATCH (a:Person) WHERE EXISTS { MATCH (a)-[:KNOWS]->() } RETURN a.name",
        "MATCH (a:Person) WHERE NOT EXISTS { MATCH (a)-[:KNOWS]->() } RETURN a.name",
        "MATCH (a:Person) RETURN a.name, EXISTS { MATCH (a)-[:KNOWS]->() } AS x",
    ] {
        let reads = |session: &Session| session.execute(query).unwrap().rows_scanned().unwrap();
        assert!(reads(&usual) < 50, "{query}: on read {}", reads(&usual));
        assert!(
            reads(&without) >= 2000,
            "{query}: off read {}",
            reads(&without)
        );
        assert!(
            reads(&plain) >= 2000,
            "{query}: no rule read {}",
            reads(&plain)
        );
        assert_eq!(answer(&without, query), answer(&usual, query), "{query}");
        assert_eq!(answer(&plain, query), answer(&usual, query), "{query}");
    }
}

/// Operators no plan may run with every rule off, by the rule they stand behind.
const NOT_PLAIN: [&str; 9] = [
    "NodeList",
    "RangeScan",
    "NodeSeek",
    "EdgeCount",
    "LabelCount",
    "FactorizedAggregate",
    "LazyFactorizedChain",
    "Leapfrog",
    "[reachability",
];

fn answer(session: &Session, query: &str) -> Vec<String> {
    let result = session.execute(query).unwrap();
    let mut rows: Vec<String> = result.rows().iter().map(|row| format!("{row:?}")).collect();
    rows.sort();
    rows
}

/// With the rule on it runs its own way; off, it runs a scan and a filter with the same rows,
/// and `EXPLAIN` no longer says otherwise. Off in a plan with every other rule on, and in one
/// with no rule at all.
#[test]
fn a_rule_off_falls_back_and_explain_agrees() {
    let db = indexed();
    let usual = db.session();
    let plain = with_rules(&db, PlannerRules::plain());
    let mut wrong = Vec::new();
    for (rule, query, runs, hint) in WAYS_IN {
        let without = with_rules(
            &db,
            PlannerRules::all()
                .with(rule, false)
                .with(PlannerRule::KeyColumnLookup, false),
        );
        let on = text(&usual.execute(&format!("PROFILE {query}")).unwrap());
        if !on.contains(runs) {
            wrong.push(format!("{rule:?} on does not run {runs}: {query}\n{on}"));
        }
        if !hint.is_empty() && !explain(&usual, query).contains(hint) {
            wrong.push(format!(
                "{rule:?} on does not show {hint}: {query}\n{}",
                explain(&usual, query)
            ));
        }
        for (name, session) in [("off", &without), ("no rule", &plain)] {
            let off = text(&session.execute(&format!("PROFILE {query}")).unwrap());
            if off.contains(runs) {
                wrong.push(format!("{rule:?} {name} still runs {runs}: {query}\n{off}"));
            }
            let plan = explain(session, query);
            if !hint.is_empty() && plan.contains(hint) {
                wrong.push(format!(
                    "{rule:?} {name} still shows {hint}: {query}\n{plan}"
                ));
            }
            if answer(session, query) != answer(&usual, query) {
                wrong.push(format!("{rule:?} {name} changes the rows: {query}"));
            }
        }
    }
    assert!(wrong.is_empty(), "{}", wrong.join("\n"));
}

/// With no rule at all, none of the shortcuts runs on any of these queries.
#[test]
fn with_no_rule_no_shortcut_runs() {
    let db = indexed();
    let plain = with_rules(&db, PlannerRules::plain());
    let mut wrong = Vec::new();
    let mut queries: Vec<&str> = WAYS_IN.iter().map(|way| way.1).collect();
    queries.extend([
        "MATCH (n) RETURN count(*)",
        "MATCH (n:Person) RETURN count(*)",
        "MATCH (a)-[e]->(b) RETURN count(*)",
        "MATCH (a:Person)-[:KNOWS]->(b) RETURN a.name, count(*)",
        "MATCH (a:Person)-[:KNOWS]->() RETURN a.name, count(*)",
        "MATCH (a:Person)-[:KNOWS]->(b)-[:KNOWS]->(c) RETURN count(*)",
        "MATCH (a:Person)-[:KNOWS]->(b)-[:KNOWS]->() RETURN count(*)",
        "MATCH (a:Person)-[:KNOWS]->{1,3}(b) RETURN DISTINCT b.name",
        "MATCH (a)-[:KNOWS]->(b) WHERE id(b) = 1 RETURN a.name",
        "MATCH (a)-[:KNOWS]->(b)-[:KNOWS]->(a) RETURN a.name",
        "MATCH (a:Person {k: 1})-[:KNOWS]->(c)-[:KNOWS]->(b:Person {k: 2}) RETURN c.name",
        "MATCH (a:Person) WHERE EXISTS { MATCH (a)-[:KNOWS]->()-[:KNOWS]->() } RETURN a.name",
        "MATCH (a:Person) WHERE NOT EXISTS { MATCH (a)-[:KNOWS]->() } RETURN a.name",
        "MATCH (a:Person) RETURN a.name, COUNT { MATCH (a)-[:KNOWS]->() } AS n",
        "MATCH (a:Person) RETURN a.name, EXISTS { MATCH (a)-[:KNOWS]->() } AS n",
    ]);
    for query in queries {
        let profile = text(&plain.execute(&format!("PROFILE {query}")).unwrap());
        let shown = format!("{}\n{profile}", explain(&plain, query));
        for word in NOT_PLAIN {
            if shown.contains(word) {
                wrong.push(format!("{word} with no rule: {query}\n{shown}"));
            }
        }
        for hint in ["[index", "[range", "[seek", "[hash join", "[expand"] {
            if shown.contains(hint) {
                wrong.push(format!("{hint} with no rule: {query}\n{shown}"));
            }
        }
    }
    assert!(wrong.is_empty(), "{}", wrong.join("\n"));
}

/// The lookup gives parallel edges in the order the list does, row for row, also when it reads
/// the shorter list at the far end: a hub's long list, a target's short one, several edges
/// between the two, before and after the adjacency merges.
#[test]
fn the_lookup_keeps_the_order_of_the_list() {
    let db = GrafeoDB::new_in_memory();
    let hub = db.create_node(&["Hub"]).unwrap();
    let target = db
        .create_node_with_props(&["Target"], [("k", Value::Int64(0))])
        .unwrap();
    let mut key = 0_i64;
    let mut edge = |from, to, kind: &str| {
        key += 1;
        db.create_edge_with_props(from, to, kind, [("k", Value::Int64(key))])
            .unwrap();
    };
    edge(target, hub, "BACK");
    for _ in 0..3 {
        edge(hub, target, "E");
    }
    let query = "MATCH (t:Target)-[:BACK]->(h:Hub)-[e:E]->(t) RETURN e.k";
    let written = without_intersection().with(PlannerRule::JoinOrder, false);
    let lookup = with_rules(&db, written);
    let list = with_rules(&db, written.with(PlannerRule::ExpandInto, false));
    let rows = |session: &Session| -> Vec<String> {
        session
            .execute(query)
            .unwrap()
            .rows()
            .iter()
            .map(|row| format!("{row:?}"))
            .collect()
    };
    assert!(explain(&lookup, query).contains("expand into"));
    assert_eq!(rows(&lookup), rows(&list), "the delta, both lists short");
    // Now the hub's list is long, and the target's the short one.
    for _ in 0..1_500 {
        let other = db.create_node(&["Other"]).unwrap();
        edge(hub, other, "E");
    }
    for _ in 0..3 {
        edge(hub, target, "E");
    }
    assert_eq!(rows(&lookup).len(), 6);
    assert_eq!(rows(&lookup), rows(&list), "the hub's list longer");
}
