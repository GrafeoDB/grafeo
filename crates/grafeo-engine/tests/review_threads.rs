//! What review of the 0.6.2 pull request claimed, each against the running code: those that hold
//! fail until fixed, and those that do not stay as what they show.
//!
//! ```bash
//! cargo test -p grafeo-engine --all-features --test main review_threads::
//! ```

use std::sync::mpsc;
use std::time::Duration;

use grafeo_common::types::{ArcStr, ReadView};
use grafeo_core::graph::GraphStore;
use grafeo_engine::query::optimizer::{PlannerRule, PlannerRules};
use grafeo_engine::session::Session;
use grafeo_engine::{Config, GrafeoDB, GraphModel};

fn run_all(session: &Session, statements: &[&str]) {
    for statement in statements {
        session
            .execute(statement)
            .unwrap_or_else(|error| panic!("{statement}: {error}"));
    }
}

fn single_i64(session: &Session, query: &str) -> i64 {
    session.execute(query).unwrap().rows()[0][0]
        .as_int64()
        .unwrap()
}

// ---------------------------------------------------------------------------
// chain_counter.rs:443: repeated inequalities
// ---------------------------------------------------------------------------

/// Repeating an inequality of two edges says nothing new: the same count, at once. Over the
/// subsets of the written pairs that would take for ever, or overflow the shift, once the chain
/// counter reads them. Each form, in both languages.
fn repeated_inequality_answers(repeats: usize) {
    for (language, pair) in [
        ("cypher", "id(e1) <> id(e2)"),
        ("cypher", "e1 <> e2"),
        ("cypher", "NOT id(e1) = id(e2)"),
        ("gql", "id(e1) <> id(e2)"),
        ("gql", "e1 <> e2"),
        ("gql", "element_id(e1) <> element_id(e2)"),
    ] {
        let (sender, receiver) = mpsc::channel();
        std::thread::spawn(move || {
            let db = GrafeoDB::new_in_memory();
            let session = db.session();
            run_all(
                &session,
                &["INSERT (:P {k: 1})-[:K]->(:P {k: 2})-[:K]->(:P {k: 3})"],
            );
            let run = |query: &str| {
                let result = if language == "cypher" {
                    session.execute_cypher(query)
                } else {
                    session.execute(query)
                };
                result
                    .map(|r| r.rows()[0][0].as_int64().unwrap())
                    .map_err(|e| e.to_string())
            };
            let once = run(&format!(
                "MATCH (a)-[e1:K]->(b)-[e2:K]->(c) WHERE {pair} RETURN count(*) AS n"
            ));
            let many = run(&format!(
                "MATCH (a)-[e1:K]->(b)-[e2:K]->(c) WHERE {} RETURN count(*) AS n",
                vec![pair; repeats].join(" AND ")
            ));
            sender.send((once, many)).ok();
        });
        let (once, many) = receiver
            .recv_timeout(Duration::from_secs(20))
            .unwrap_or_else(|_| {
                panic!("{language}, {pair}, {repeats} repeats: no answer: a panic, or the subsets")
            });
        assert_eq!(many, once, "{language}, {pair}, {repeats} repeats");
    }
}

#[test]
fn a_repeated_inequality_of_edges_counts_like_one() {
    repeated_inequality_answers(2);
    repeated_inequality_answers(12);
}

#[test]
fn many_repeats_of_an_inequality_of_edges_answer_at_once() {
    repeated_inequality_answers(40);
}

#[test]
fn more_repeats_of_an_inequality_than_a_shift_holds_answer() {
    repeated_inequality_answers(70);
}

// ---------------------------------------------------------------------------
// standalone.rs:316: RDF graph claims
// ---------------------------------------------------------------------------

const G: &str = "http://example.org/g";
const H: &str = "http://example.org/h";

fn rdf_db() -> GrafeoDB {
    let db = GrafeoDB::new_in_memory();
    fill(&db);
    db
}

fn fill(db: &GrafeoDB) {
    let session = db.session();
    for statement in [
        format!("CREATE GRAPH <{G}>"),
        format!("CREATE GRAPH <{H}>"),
        "INSERT DATA { GRAPH <http://example.org/third> { <http://example.org/jules> <http://example.org/knows> <http://example.org/gus> } }".to_string(),
        format!("INSERT DATA {{ GRAPH <{G}> {{ <http://example.org/alix> <http://example.org/knows> <http://example.org/gus> }} }}"),
        format!("INSERT DATA {{ GRAPH <{H}> {{ <http://example.org/gus> <http://example.org/knows> <http://example.org/mia> }} }}"),
        "INSERT DATA { <http://example.org/vincent> <http://example.org/knows> <http://example.org/mia> }".to_string(),
    ] {
        session.execute_sparql(&statement).unwrap();
    }
}

/// One claim for all queued RDF graph operations: a second transaction's
/// operation fails at its statement, with `GRAFEO-T001`, whichever graphs the two name.
#[test]
fn two_transactions_with_graph_operations_conflict_at_the_second_statement() {
    let new = "http://example.org/new";
    for (one, other) in [
        ("CLEAR ALL".to_string(), format!("DROP GRAPH <{G}>")),
        (format!("DROP GRAPH <{G}>"), "CLEAR ALL".to_string()),
        (format!("CLEAR GRAPH <{G}>"), format!("DROP GRAPH <{G}>")),
        (format!("CLEAR GRAPH <{G}>"), format!("CLEAR GRAPH <{H}>")),
        (
            format!("CREATE GRAPH <{new}>"),
            format!("COPY <{G}> TO <{new}>"),
        ),
        (format!("ADD <{G}> TO <{H}>"), "CLEAR DEFAULT".to_string()),
    ] {
        let db = rdf_db();
        let mut first = db.session();
        let mut second = db.session();
        first.begin_transaction().unwrap();
        second.begin_transaction().unwrap();
        first.execute_sparql(&one).unwrap();
        let error = second
            .execute_sparql(&other)
            .expect_err(&format!("{one} is open, then {other}"));
        assert!(
            error.to_string().contains("GRAFEO-T001"),
            "{one}, then {other}: {error}"
        );
        second.rollback().unwrap();
        first.commit().unwrap();
    }
}

/// A standalone operation meets an open transaction's the same way.
#[test]
fn a_standalone_graph_operation_meets_an_open_transaction() {
    let db = rdf_db();
    let mut open = db.session();
    open.begin_transaction().unwrap();
    open.execute_sparql(&format!("CLEAR GRAPH <{G}>")).unwrap();
    let error = db
        .session()
        .execute_sparql(&format!("DROP GRAPH <{H}>"))
        .expect_err("a graph operation is open");
    assert!(error.to_string().contains("GRAFEO-T001"), "{error}");
    open.commit().unwrap();
}

/// What the gap costs when both commit: a graph op that fails after it was logged poisons the
/// database. Each pair of operations, in both orders, then a write must still commit.
#[test]
fn two_graph_ops_that_both_commit_do_not_poison_the_database() {
    let new = "http://example.org/new";
    let ops = [
        "CLEAR ALL".to_string(),
        "DROP ALL".to_string(),
        "CLEAR DEFAULT".to_string(),
        "DROP NAMED".to_string(),
        format!("DROP GRAPH <{G}>"),
        format!("CLEAR GRAPH <{G}>"),
        format!("CREATE GRAPH <{new}>"),
        format!("COPY <{G}> TO <{new}>"),
        format!("MOVE <{G}> TO <{new}>"),
        format!("ADD <{G}> TO <{H}>"),
        format!("MOVE <{H}> TO <{G}>"),
    ];
    for one in &ops {
        for other in &ops {
            let db = rdf_db();
            let mut first = db.session();
            let mut second = db.session();
            first.begin_transaction().unwrap();
            second.begin_transaction().unwrap();
            let _ = first.execute_sparql(one);
            let _ = second.execute_sparql(other);
            let committed_first = first.commit().map_err(|error| error.to_string());
            let committed_second = second.commit().map_err(|error| error.to_string());
            let after = db.session();
            let write = after.execute_sparql(
                "INSERT DATA { <http://example.org/mia> <http://example.org/knows> <http://example.org/gus> }",
            );
            assert!(
                write.is_ok(),
                "{one} then {other}: {committed_first:?}, {committed_second:?}: \
                 a later write fails: {write:?}"
            );
        }
    }
}

const POISON_PATH: &str = "GRAFEO_REVIEW_THREADS_POISON_PATH";
const POISON_PAIR: &str = "GRAFEO_REVIEW_THREADS_POISON_PAIR";
const POISON_RDF: &str = "GRAFEO_REVIEW_THREADS_POISON_RDF";

fn poison_ops() -> Vec<String> {
    let new = "http://example.org/new";
    vec![
        "CLEAR ALL".to_string(),
        "DROP ALL".to_string(),
        "DROP NAMED".to_string(),
        format!("DROP GRAPH <{G}>"),
        format!("CREATE GRAPH <{new}>"),
        format!("MOVE <{G}> TO <{new}>"),
        format!("COPY <{G}> TO <{H}>"),
        format!("MOVE <{H}> TO <{G}>"),
    ]
}

/// Both operations commit from two transactions, then the process ends without a close.
#[test]
fn poison_child() {
    let Some(path) = std::env::var_os(POISON_PATH) else {
        return;
    };
    let pair: Vec<usize> = std::env::var(POISON_PAIR)
        .unwrap()
        .split(',')
        .map(|n| n.parse().unwrap())
        .collect();
    let ops = poison_ops();
    let mut config = Config::persistent(std::path::PathBuf::from(path));
    if std::env::var_os(POISON_RDF).is_some() {
        config = config.with_graph_model(GraphModel::Rdf);
    }
    let db = GrafeoDB::with_config(config).unwrap();
    let setup = db.session();
    for statement in [
        format!("CREATE GRAPH <{G}>"),
        format!("CREATE GRAPH <{H}>"),
        format!(
            "INSERT DATA {{ GRAPH <{G}> {{ <http://example.org/alix> <http://example.org/knows> <http://example.org/gus> }} }}"
        ),
        format!(
            "INSERT DATA {{ GRAPH <{H}> {{ <http://example.org/gus> <http://example.org/knows> <http://example.org/mia> }} }}"
        ),
    ] {
        setup.execute_sparql(&statement).unwrap();
    }
    let mut first = db.session();
    let mut second = db.session();
    first.begin_transaction().unwrap();
    second.begin_transaction().unwrap();
    let _ = first.execute_sparql(&ops[pair[0]]);
    let _ = second.execute_sparql(&ops[pair[1]]);
    let one = first.commit().map_err(|error| error.to_string());
    let other = second.commit().map_err(|error| error.to_string());
    let after = db.session().execute_sparql(
        "INSERT DATA { <http://example.org/mia> <http://example.org/knows> <http://example.org/gus> }",
    );
    assert!(one.is_ok() || other.is_ok(), "{pair:?}: {one:?} {other:?}");
    after.unwrap_or_else(|error| panic!("{pair:?}: a write after both: {error}"));
    std::process::exit(0);
}

/// The log of two graph operations that both committed replays, and the database takes writes:
/// in a database of either graph model.
#[test]
fn the_log_of_two_committed_graph_ops_replays() {
    let count = poison_ops().len();
    for rdf in [false, true] {
        for one in 0..count {
            for other in 0..count {
                let dir = tempfile::tempdir().unwrap();
                let path = dir.path().join("poison.grafeo");
                let mut child = grafeo_common::child_test!("poison_child");
                child
                    .env(POISON_PATH, &path)
                    .env(POISON_PAIR, format!("{one},{other}"));
                if rdf {
                    child.env(POISON_RDF, "1");
                }
                let status = grafeo_common::testing::child_process::run(&mut child).unwrap();
                assert!(status.success(), "{one},{other}: the child");
                let mut config = Config::persistent(&path);
                if rdf {
                    config = config.with_graph_model(GraphModel::Rdf);
                }
                let db = GrafeoDB::with_config(config)
                    .unwrap_or_else(|error| panic!("{rdf} {one},{other}: the reopen: {error}"));
                db.session()
                    .execute_sparql(
                        "INSERT DATA { <http://example.org/butch> <http://example.org/knows> <http://example.org/mia> }",
                    )
                    .unwrap_or_else(|error| panic!("{rdf} {one},{other}: a write after replay: {error}"));
                db.close().unwrap();
            }
        }
    }
}

/// In an RDF database too, a rollback keeps what the operation would have taken.
#[test]
fn a_rolled_back_drop_in_an_rdf_database_keeps_the_graph() {
    let db = GrafeoDB::with_config(Config::in_memory().with_graph_model(GraphModel::Rdf)).unwrap();
    fill(&db);
    let mut session = db.session();
    session.begin_transaction().unwrap();
    session
        .execute_sparql(&format!("DROP GRAPH <{G}>"))
        .unwrap();
    session.rollback().unwrap();
    let rows = db
        .session()
        .execute_sparql(&format!("SELECT ?s WHERE {{ GRAPH <{G}> {{ ?s ?p ?o }} }}"))
        .unwrap()
        .row_count();
    assert_eq!(rows, 1, "a rollback of a DROP GRAPH");
}

// ---------------------------------------------------------------------------
// transaction/rdf.rs:426: a failed op leaves its first claim
// ---------------------------------------------------------------------------

/// A statement that is refused holds nothing: once the holder is gone, a third transaction may
/// do what the refused one's first claim would have stopped.
#[test]
fn a_refused_graph_op_leaves_no_claim() {
    let db = rdf_db();
    let third = "http://example.org/third";
    let mut holder = db.session();
    let mut refused = db.session();
    let mut later = db.session();
    holder.begin_transaction().unwrap();
    refused.begin_transaction().unwrap();
    later.begin_transaction().unwrap();
    holder
        .execute_sparql(&format!("COPY <{G}> TO <{H}>"))
        .unwrap();
    assert!(
        refused
            .execute_sparql(&format!("COPY <{third}> TO <{H}>"))
            .is_err(),
        "H is claimed"
    );
    holder.rollback().unwrap();
    later
        .execute_sparql(&format!("COPY <{third}> TO <{G}>"))
        .unwrap_or_else(|error| panic!("the refused statement holds nothing: {error}"));
}

// ---------------------------------------------------------------------------
// session/pending_schema.rs: back on a graph that is gone
// ---------------------------------------------------------------------------

#[test]
fn a_rollback_of_a_graph_made_chosen_and_dropped_leaves_the_session_usable() {
    let db = GrafeoDB::new_in_memory();
    let mut session = db.session();
    run_all(&session, &["INSERT (:Person {name: 'Gus'})"]);
    session.begin_transaction().unwrap();
    run_all(
        &session,
        &[
            "CREATE GRAPH g",
            "USE GRAPH g",
            "INSERT (:Person)",
            "DROP GRAPH g",
        ],
    );
    session.rollback().unwrap();
    assert_eq!(
        single_i64(&session, "MATCH (p:Person) RETURN count(p)"),
        1,
        "back on the default graph"
    );
    session.execute("INSERT (:Person {name: 'Mia'})").unwrap();
}

// ---------------------------------------------------------------------------
// optimizer/mod.rs:2212: a label check's other arguments
// ---------------------------------------------------------------------------

#[test]
fn a_label_check_with_a_raising_argument_stays_behind_its_guard() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    run_all(
        &session,
        &["INSERT (:T {k: 0})-[:R]->(:U), (:T {k: 2})-[:R]->(:U)"],
    );
    let query =
        "MATCH (n:T)-[:R]->(u) FILTER n.k <> 0 FILTER hasLabel(n, toString(100 / n.k)) RETURN n.k";
    let answer = session.execute(query);
    answer.unwrap_or_else(|error| panic!("the guard was written first: {error}"));
}

// ---------------------------------------------------------------------------
// planner/lpg/seek.rs:88: the second key when the first is turned off
// ---------------------------------------------------------------------------

/// A key from the rows, as a constant key is looked up while planning, with no seek in any case.
/// The rules choose between the two seeks, and each answers alike.
#[test]
fn with_id_seeks_off_a_key_conjunct_still_seeks() {
    let db = GrafeoDB::new_in_memory();
    db.create_property_index("k").unwrap();
    let mut session = db.session();
    run_all(&session, &["INSERT (:T {k: 1}), (:T {k: 2})"]);
    let query = "UNWIND [1] AS x MATCH (n:T) WHERE id(n) = 0 AND n.k = x RETURN n.k";
    for (rules, chosen) in [
        (PlannerRules::all(), Some("[seek: id]")),
        (
            PlannerRules::all().with(PlannerRule::IdSeek, false),
            Some("[index: k]"),
        ),
        (
            PlannerRules::all()
                .with(PlannerRule::IdSeek, false)
                .with(PlannerRule::KeySeek, false),
            None,
        ),
    ] {
        session.set_planner_rules(rules);
        let plan = session.execute(&format!("EXPLAIN {query}")).unwrap().rows()[0][0]
            .as_str()
            .unwrap()
            .to_string();
        match chosen {
            Some(seek) => assert!(plan.contains(seek), "{seek}: {plan}"),
            None => assert!(
                !plan.contains("[seek: id]") && !plan.contains("[index: k]"),
                "{plan}"
            ),
        }
        assert_eq!(single_i64(&session, query), 1, "{plan}");
    }
}

// ---------------------------------------------------------------------------
// database/admin.rs:387: a schema-qualified type over the root store
// ---------------------------------------------------------------------------

#[test]
fn a_schema_qualified_type_over_a_schema_session_default_graph_warns() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    run_all(
        &session,
        &[
            "CREATE SCHEMA archive",
            "SESSION SET SCHEMA archive",
            "CREATE GRAPH TYPE atlas (NODE TYPE City (name STRING))",
            "CREATE NODE TYPE Museum (name STRING)",
            "INSERT (:City {size: 1})",
        ],
    );
    let warnings: Vec<String> = db
        .validate()
        .unwrap()
        .warnings
        .into_iter()
        .filter(|w| w.code == "GRAPH_TYPE_NOT_BOUND")
        .map(|w| w.message)
        .collect();
    assert_eq!(warnings.len(), 1, "{warnings:?}");
    assert!(warnings[0].contains("City"), "{warnings:?}");
}

// ---------------------------------------------------------------------------
// rowgroup/read.rs:686: the order of edges_between
// ---------------------------------------------------------------------------

#[test]
fn edges_between_keeps_the_order_edges_lists() {
    let db = GrafeoDB::new_in_memory();
    let hub = db.create_node(&["Hub"]).unwrap();
    let small = db.create_node(&["Small"]).unwrap();
    let mut ids = Vec::new();
    for round in 0..40 {
        ids.push(db.create_edge(hub, small, "K").unwrap());
        for _ in 0..(round + 2) {
            let other = db.create_node(&["Other"]).unwrap();
            db.create_edge(hub, other, "K").unwrap();
        }
        ids.push(db.create_edge(small, hub, "K").unwrap());
    }
    let snapshot = db.snapshot().unwrap();
    let store = snapshot.store();
    let k = vec![ArcStr::from("K")];
    let view = ReadView::latest();
    for (from, to, direction) in [
        (hub, small, grafeo_core::graph::Direction::Outgoing),
        (hub, small, grafeo_core::graph::Direction::Incoming),
        (small, hub, grafeo_core::graph::Direction::Outgoing),
        (small, hub, grafeo_core::graph::Direction::Incoming),
    ] {
        let listed: Vec<_> = store
            .edges(from, direction, &k, view)
            .unwrap()
            .into_iter()
            .filter(|(end, _)| *end == to)
            .map(|(_, edge)| edge)
            .collect();
        assert_eq!(
            store.edges_between(from, to, direction, &k, view).unwrap(),
            listed,
            "{direction:?}"
        );
    }
}

// ---------------------------------------------------------------------------
// chain_counter.rs:76 and edge_count.rs:175: reads counted for a read not made
// ---------------------------------------------------------------------------

/// A middle node with no edge back reads one degree, one with an edge back two.
#[test]
fn a_middle_node_with_no_way_back_counts_one_read() {
    let query = "MATCH (:A)-[:K]->(m:M)-[:L]->(:B) RETURN count(*)";
    let scanned = |statements: &[&str]| {
        let db = GrafeoDB::new_in_memory();
        let session = db.session();
        run_all(&session, statements);
        session.execute(query).unwrap().rows_scanned().unwrap()
    };
    // Five middle nodes either way, and one edge of type K so the plan is the same.
    let no_way_back = scanned(&[
        "INSERT (:M), (:M), (:M), (:M), (:M)",
        "INSERT (:A)-[:K]->(:C)",
    ]);
    let one_way_back = scanned(&[
        "INSERT (:A)-[:K]->(:M), (:A)-[:K]->(:M), (:A)-[:K]->(:M), (:A)-[:K]->(:M), (:A)-[:K]->(:M)",
    ]);
    assert!(
        no_way_back < one_way_back,
        "{no_way_back} against {one_way_back}"
    );
}

// ---------------------------------------------------------------------------
// rowgroup/diagnostics.rs:270: what the samples keep, in both totals
// ---------------------------------------------------------------------------

/// `memory_usage` shows the samples through its breakdown, and `heap_bytes` must agree. A named
/// graph's samples count as well.
#[test]
#[allow(deprecated, reason = "`heap_bytes` has no route but the store itself")]
fn the_memory_total_holds_what_the_samples_keep() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    run_all(
        &session,
        &[
            "INSERT (:A)-[:K]->(:B)-[:L]->(:C)",
            "CREATE GRAPH other",
            "USE GRAPH other",
            "INSERT (:A)-[:K]->(:B)-[:L]->(:C)",
        ],
    );
    let store = db.store().unwrap();
    let total = store.heap_bytes();
    let usage = db.memory_usage().total_bytes;
    for index in 0..200 {
        let on = format!("ON{index}");
        store.sampled_reached_degree(
            Some("A"),
            &["K"],
            grafeo_core::graph::Direction::Outgoing,
            &[on.as_str()],
            grafeo_core::graph::Direction::Outgoing,
        );
    }
    assert!(db.memory_usage().total_bytes > usage, "the breakdown");
    assert!(store.heap_bytes() > total, "heap_bytes");
}

// ---------------------------------------------------------------------------
// A closing self-loop: no intersection with or without backward edges, the rows alike
// ---------------------------------------------------------------------------

#[test]
fn a_closing_self_loop_answers_alike_without_backward_edges() {
    for config in [
        Config::in_memory(),
        Config::in_memory().without_backward_edges(),
    ] {
        let db = GrafeoDB::with_config(config).unwrap();
        let session = db.session();
        run_all(
            &session,
            &[
                "INSERT (:A {k: 1})-[:K]->(b:B {k: 2}), (b)-[:K]->(b)",
                "INSERT (:A {k: 3})-[:K]->(:B {k: 4})",
            ],
        );
        assert_eq!(
            single_i64(&session, "MATCH (a:A)-[:K]->(b:B)-[:K]->(b) RETURN a.k"),
            1
        );
    }
}

// ---------------------------------------------------------------------------
// Found while checking join_order_shapes.rs:550: booleans have no order, in some plans
// ---------------------------------------------------------------------------

/// False is below true (ISO/IEC 39075 19.3, General Rule 2, b, g), and a boolean takes its place in
/// GQL's order of kinds against another: after a string, before a number. A range read of a property
/// once found them while an expression gave null, so `n.p <= true` kept rows alone and lost them
/// behind anything else: a wrong count, the same query written in another order. Every plan agrees.
#[test]
fn a_comparison_of_booleans_answers_alike_in_every_plan() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    run_all(
        &session,
        &["INSERT (:T {p: true}), (:T {p: false}), (:T {p: 1}), (:A {k: 1})"],
    );
    let names = |query: &str| {
        let mut rows: Vec<String> = session
            .execute(query)
            .unwrap_or_else(|error| panic!("{query}: {error}"))
            .rows()
            .iter()
            .map(|row| format!("{:?}", row[0]))
            .collect();
        rows.sort();
        rows
    };
    for (condition, want) in [
        ("n.p <= true", vec!["Bool(false)", "Bool(true)"]),
        ("n.p >= true", vec!["Bool(true)", "Int64(1)"]),
        ("n.p < true", vec!["Bool(false)"]),
        ("n.p > false", vec!["Bool(true)", "Int64(1)"]),
        (
            "n.p >= false",
            vec!["Bool(false)", "Bool(true)", "Int64(1)"],
        ),
        ("true >= n.p", vec!["Bool(false)", "Bool(true)"]),
        ("n.p > true", vec!["Int64(1)"]),
    ] {
        for query in [
            format!("MATCH (n:T) WHERE {condition} RETURN n.p"),
            format!("MATCH (a:A), (n:T) WHERE {condition} RETURN n.p"),
            format!("MATCH (n:T), (a:A) WHERE {condition} RETURN n.p"),
            format!("MATCH (a:A) MATCH (n:T) WHERE {condition} RETURN n.p"),
            format!("UNWIND [1] AS x MATCH (n:T) WHERE {condition} RETURN n.p"),
        ] {
            assert_eq!(names(&query), want, "{query}");
        }
    }
    // Null stays unknown: no row, and null as a value.
    for condition in ["n.p <= null", "n.p > null", "null >= n.p"] {
        for query in [
            format!("MATCH (n:T) WHERE {condition} RETURN n.p"),
            format!("MATCH (a:A), (n:T) WHERE {condition} RETURN n.p"),
        ] {
            assert_eq!(names(&query), Vec::<String>::new(), "{query}");
        }
    }
    for expression in ["true < null", "null <= false", "null > true"] {
        let query = format!("RETURN {expression} AS v");
        assert!(
            session.execute(&query).unwrap().rows()[0][0].is_null(),
            "{query}"
        );
    }
    for (expression, want) in [
        ("false < true", true),
        ("true > false", true),
        ("true <= true", true),
        ("true >= true", true),
        ("false >= true", false),
        ("true < false", false),
        // A string, a boolean and a number, in GQL's order of kinds.
        ("true < 1", true),
        ("1 < true", false),
        ("true >= 'x'", true),
        ("'x' < false", true),
    ] {
        let query = format!("RETURN {expression} AS v");
        assert_eq!(
            session.execute(&query).unwrap().rows()[0][0].as_bool(),
            Some(want),
            "{query}"
        );
    }
}

// ---------------------------------------------------------------------------
// A queued graph operation that would not apply
// ---------------------------------------------------------------------------

const X: &str = "http://example.org/x";

/// A standalone write made the graph meanwhile: the commit's `CREATE GRAPH` would not apply, so it
/// fails before anything is logged. The database stays writable, and the graph stays as written.
#[test]
fn a_graph_op_that_no_longer_applies_fails_its_commit() {
    let db = GrafeoDB::new_in_memory();
    let mut open = db.session();
    open.begin_transaction().unwrap();
    open.execute_sparql(&format!("CREATE GRAPH <{X}>")).unwrap();
    db.session()
        .execute_sparql(&format!(
            "INSERT DATA {{ GRAPH <{X}> {{ <http://example.org/gus> <http://example.org/knows> <http://example.org/mia> }} }}"
        ))
        .unwrap();
    let error = open.commit().expect_err("the graph is there now");
    assert!(error.to_string().contains("GRAFEO-T001"), "{error}");
    let after = db.session();
    after
        .execute_sparql("INSERT DATA { <http://example.org/mia> <http://example.org/knows> <http://example.org/gus> }")
        .unwrap_or_else(|error| panic!("a write after the refused commit: {error}"));
    after.execute("INSERT (:Person {name: 'Vincent'})").unwrap();
    let rows = after
        .execute_sparql(&format!("SELECT ?s WHERE {{ GRAPH <{X}> {{ ?s ?p ?o }} }}"))
        .unwrap()
        .row_count();
    assert_eq!(rows, 1);
}

/// The graph went away meanwhile: a `DROP GRAPH` queued against it. Either it no longer applies and
/// the commit says so, or it has nothing to do; never a broken database.
#[test]
fn a_drop_of_a_graph_emptied_meanwhile_leaves_the_database_writable() {
    let db = GrafeoDB::new_in_memory();
    let triple = "<http://example.org/gus> <http://example.org/knows> <http://example.org/mia>";
    let setup = db.session();
    setup
        .execute_sparql(&format!("INSERT DATA {{ GRAPH <{X}> {{ {triple} }} }}"))
        .unwrap();
    let mut open = db.session();
    open.begin_transaction().unwrap();
    open.execute_sparql(&format!("DROP GRAPH <{X}>")).unwrap();
    setup
        .execute_sparql(&format!("DELETE DATA {{ GRAPH <{X}> {{ {triple} }} }}"))
        .unwrap();
    if let Err(error) = open.commit() {
        assert!(error.to_string().contains("GRAFEO-T001"), "{error}");
    }
    setup
        .execute_sparql(&format!("INSERT DATA {{ {triple} }}"))
        .unwrap_or_else(|error| panic!("a write after the commit: {error}"));
}

const REFUSED_PATH: &str = "GRAFEO_REVIEW_THREADS_REFUSED_PATH";

/// The commit is refused, then the process ends without a close.
#[test]
fn refused_commit_child() {
    let Some(path) = std::env::var_os(REFUSED_PATH) else {
        return;
    };
    let db = GrafeoDB::open(std::path::PathBuf::from(path)).unwrap();
    let mut open = db.session();
    open.begin_transaction().unwrap();
    open.execute_sparql(&format!("CREATE GRAPH <{X}>")).unwrap();
    db.session()
        .execute_sparql(&format!(
            "INSERT DATA {{ GRAPH <{X}> {{ <http://example.org/gus> <http://example.org/knows> <http://example.org/mia> }} }}"
        ))
        .unwrap();
    open.commit().expect_err("the graph is there now");
    db.session()
        .execute("INSERT (:Person {name: 'Vincent'})")
        .unwrap();
    std::process::exit(0);
}

/// Nothing in the log trips a replay, and what committed around the refusal is there.
#[test]
fn a_refused_graph_op_leaves_a_log_that_replays() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("refused.grafeo");
    let status = grafeo_common::testing::child_process::run(
        grafeo_common::child_test!("refused_commit_child").env(REFUSED_PATH, &path),
    )
    .unwrap();
    assert!(status.success(), "the child process failed");
    let db = GrafeoDB::open(&path).unwrap();
    let session = db.session();
    assert_eq!(single_i64(&session, "MATCH (p:Person) RETURN count(p)"), 1);
    let rows = session
        .execute_sparql(&format!("SELECT ?s WHERE {{ GRAPH <{X}> {{ ?s ?p ?o }} }}"))
        .unwrap()
        .row_count();
    assert_eq!(rows, 1);
    session.execute("INSERT (:Person {name: 'Mia'})").unwrap();
    drop(session);
    db.close().unwrap();
}

// ---------------------------------------------------------------------------
// The check of endpoints, where it must not refuse
// ---------------------------------------------------------------------------

/// The check reads the graphs the rule holds in: a graph of other edges does not stop a graph type
/// of endpoints, or its replacement.
#[test]
fn endpoints_are_not_checked_over_a_graph_they_do_not_bind() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    run_all(
        &session,
        &[
            "CREATE GRAPH plain",
            "USE GRAPH plain",
            "INSERT (:Town {name: 'Delft'})-[:ROAD]->(:Town {name: 'Leiden'})",
            "USE GRAPH default",
            "CREATE GRAPH TYPE atlas { (:City {name STRING})-[:ROAD]->(:City) }",
            "CREATE OR REPLACE GRAPH TYPE atlas { (:City {name STRING})-[:ROAD]->(:City) }",
            "CREATE GRAPH maps TYPED atlas",
        ],
    );
    assert_eq!(
        single_i64(
            &session,
            "USE GRAPH plain MATCH ()-[r:ROAD]->() RETURN count(r)"
        ),
        1
    );
}

const ENDPOINTS_PATH: &str = "GRAFEO_REVIEW_THREADS_ENDPOINTS_PATH";

/// A rule on endpoints the data met, then edges that meet it, then a crash: the replay checks
/// nothing it would refuse.
#[test]
fn endpoints_child() {
    let Some(path) = std::env::var_os(ENDPOINTS_PATH) else {
        return;
    };
    let db = GrafeoDB::open(std::path::PathBuf::from(path)).unwrap();
    let session = db.session();
    run_all(
        &session,
        &[
            "INSERT (:City)-[:ROAD]->(:City)",
            "CREATE EDGE TYPE ROAD CONNECTING (City) TO (City)",
            "INSERT (:City)-[:ROAD]->(:City)",
        ],
    );
    std::process::exit(0);
}

#[test]
fn endpoints_over_edges_replay() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("endpoints.grafeo");
    let status = grafeo_common::testing::child_process::run(
        grafeo_common::child_test!("endpoints_child").env(ENDPOINTS_PATH, &path),
    )
    .unwrap();
    assert!(status.success(), "the child process failed");
    let db = GrafeoDB::open(&path).unwrap();
    let session = db.session();
    assert_eq!(
        single_i64(&session, "MATCH ()-[r:ROAD]->() RETURN count(r)"),
        2
    );
    assert!(
        session.execute("INSERT (:Town)-[:ROAD]->(:City)").is_err(),
        "the rule came back"
    );
    drop(session);
    db.close().unwrap();
}

/// `stmt` in a transaction, whether the statement or the commit is what refuses it.
fn refused_in_a_transaction(db: &GrafeoDB, stmt: &str) -> bool {
    let mut session = db.session();
    session.begin_transaction().unwrap();
    if session.execute(stmt).is_err() {
        session.rollback().unwrap();
        return true;
    }
    session.commit().is_err()
}

/// Endpoints alone are a rule to check: edges that meet them commit, and edges that do not refuse
/// it, with no rule left behind.
#[test]
fn endpoints_with_no_properties_are_checked_over_the_edges_there() {
    let stmt = "CREATE EDGE TYPE ROAD CONNECTING (City) TO (City)";

    let db = GrafeoDB::new_in_memory();
    run_all(&db.session(), &["INSERT (:City)-[:ROAD]->(:City)"]);
    assert!(!refused_in_a_transaction(&db, stmt), "the edges meet them");
    assert!(
        db.session()
            .execute("INSERT (:Town)-[:ROAD]->(:City)")
            .is_err(),
        "the rule holds after the commit"
    );

    let db = GrafeoDB::new_in_memory();
    run_all(&db.session(), &["INSERT (:Town)-[:ROAD]->(:City)"]);
    assert!(refused_in_a_transaction(&db, stmt), "an edge breaks them");
    run_all(&db.session(), &["INSERT (:Town)-[:ROAD]->(:City)"]);
}

/// The second drop of one graph in one transaction: refused when written or at the commit, never
/// logged, as the log would not replay.
#[test]
fn a_graph_dropped_twice_in_a_transaction_leaves_the_database_writable() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("twice.grafeo");
    let triple = "<http://example.org/gus> <http://example.org/knows> <http://example.org/mia>";
    {
        let db = GrafeoDB::open(&path).unwrap();
        db.session()
            .execute_sparql(&format!("INSERT DATA {{ GRAPH <{X}> {{ {triple} }} }}"))
            .unwrap();
        let mut session = db.session();
        session.begin_transaction().unwrap();
        session
            .execute_sparql(&format!("DROP GRAPH <{X}>"))
            .unwrap();
        if session.execute_sparql(&format!("DROP GRAPH <{X}>")).is_ok() {
            let error = session.commit().expect_err("the second drop has no graph");
            assert!(error.to_string().contains("GRAFEO-T001"), "{error}");
        } else {
            session.rollback().unwrap();
        }
        db.session()
            .execute_sparql(&format!("INSERT DATA {{ {triple} }}"))
            .unwrap_or_else(|error| panic!("a write after: {error}"));
        db.close().unwrap();
    }
    let db = GrafeoDB::open(&path).unwrap();
    let rows = db
        .session()
        .execute_sparql(&format!("SELECT ?s WHERE {{ GRAPH <{X}> {{ ?s ?p ?o }} }}"))
        .unwrap()
        .row_count();
    assert_eq!(rows, 1, "nothing of the refused transaction applied");
    db.close().unwrap();
}

// ---------------------------------------------------------------------------
// session/pending_schema.rs: what a commit leaves of the graphs its transaction made
// ---------------------------------------------------------------------------

/// A graph an earlier, committed transaction made is no graph of the next one's: when another
/// session drops it, an empty transaction's rollback must not move the session to the default
/// graph. The next statement says the graph is gone, as with no transaction in between.
#[test]
fn a_graph_made_earlier_and_dropped_elsewhere_is_reported_after_a_rollback() {
    let db = GrafeoDB::new_in_memory();
    run_all(&db.session(), &["INSERT (:Person {name: 'Gus'})"]);
    for ending in ["rollback", "nothing"] {
        let mut session = db.session();
        session.begin_transaction().unwrap();
        run_all(&session, &["CREATE GRAPH g", "USE GRAPH g"]);
        session.commit().unwrap();
        db.session().execute("DROP GRAPH g").unwrap();
        if ending == "rollback" {
            session.begin_transaction().unwrap();
            session.rollback().unwrap();
        }
        for statement in [
            "MATCH (n) RETURN count(n)",
            "INSERT (:Person {name: 'Mia'})",
        ] {
            let error = session
                .execute(statement)
                .expect_err("the session's graph is gone");
            assert!(
                error.to_string().contains("Graph 'g' does not exist"),
                "after {ending}, {statement}: {error}"
            );
        }
        assert_eq!(
            single_i64(&db.session(), "MATCH (p:Person) RETURN count(p)"),
            1,
            "after {ending}: nothing written to the default graph"
        );
        // It can still leave.
        run_all(&session, &["USE GRAPH default"]);
        assert_eq!(single_i64(&session, "MATCH (p:Person) RETURN count(p)"), 1);
    }
}
