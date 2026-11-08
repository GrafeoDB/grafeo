//! Starting a hop at the end an index serves, when the source has a label (key-lookup design,
//! "The turn's rule").
//!
//! `(a:Start)-[:R]->(m:Far {k: $v})` may read `m` through the index and walk back. The label on `a`
//! then moves from a scan to a check, so most of these tests are about what a check must keep doing.
//! Every test compares the rows with those the plain plan gives and with what the model says, and
//! the rows come before the plan, so a wrong answer is reported as one.
//!
//! The graph has many `:Start` nodes and few far nodes per key, so the rule has a clear winner
//! whatever its constants are, and `cost_rule` fixes the counts where the winner is the point.
//!
//! ```bash
//! cargo test -p grafeo-engine --all-features --test main far_end_turn::
//! ```

use std::collections::HashMap;

use grafeo_common::types::Value;
use grafeo_engine::query::optimizer::{PlannerRule, PlannerRules};
use grafeo_engine::session::Session;
use grafeo_engine::{Config, GrafeoDB};

fn run_all(session: &Session, statements: &[&str]) {
    for statement in statements {
        session
            .execute(statement)
            .unwrap_or_else(|error| panic!("{statement}: {error}"));
    }
}

/// The crowd, and the nodes the tests look at:
///
/// - Alix, Gus and Vincent hold `:Start` (Vincent also `:Other`) and reach the far node `k: 7`;
///   Jules lacks `:Start` and reaches it too.
/// - Far nodes `k: 9` are two, reached from Alix and Gus.
/// - Far `k: 8` has two parallel edges from Alix, and Mia, of `:Start` and `:Far`, a self-loop.
/// - `:S` goes from far to source, so the same pattern can be written the other way.
const GRAPH: &[&str] = &[
    "UNWIND range(1, 5000) AS i INSERT (a:Start {s: i + 1000}), (f:Far {k: i + 1000}), (a)-[:R]->(f), (f)-[:S]->(a), (a)-[:L]->(a)",
    "INSERT (a:Start {s: 1, name: 'Alix', age: 30, zero: 1}), (g:Start {s: 2, name: 'Gus', age: 40, zero: 1}),
            (v:Start:Other {s: 3, name: 'Vincent', age: 50, zero: 1}), (j:Other {s: 4, name: 'Jules', age: 60, zero: 0}),
            (m:Far {k: 7}), (e:Far {k: 8}), (n1:Far {k: 9}), (n2:Far {k: 9}),
            (mia:Start:Far {s: 5, name: 'Mia', k: 70, age: 20, zero: 1}),
            (a)-[:R {w: 1}]->(m), (g)-[:R {w: 2}]->(m), (v)-[:R {w: 3}]->(m), (j)-[:R {w: 4}]->(m),
            (a)-[:R {w: 5}]->(e), (a)-[:R {w: 6}]->(e),
            (a)-[:R {w: 7}]->(n1), (g)-[:R {w: 8}]->(n2),
            (m)-[:S {w: 9}]->(a), (m)-[:S {w: 10}]->(j),
            (mia)-[:L {w: 11}]->(mia)",
    "CREATE INDEX by_far FOR (n:Far) ON (n.k)",
];

fn in_memory() -> GrafeoDB {
    let db = GrafeoDB::new_in_memory();
    run_all(&db.session(), GRAPH);
    db
}

/// `check` on the graph as written and as read back from a file.
fn hot_and_reopened(mut check: impl FnMut(&str, &GrafeoDB)) {
    check("hot", &in_memory());
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("turn.grafeo");
    {
        let db = GrafeoDB::open(&path).unwrap();
        run_all(&db.session(), GRAPH);
        db.close().unwrap();
    }
    let db = GrafeoDB::open(&path).unwrap();
    check("reopened", &db);
    db.close().unwrap();
}

/// The language a form is written in.
#[derive(Clone, Copy, Debug)]
enum Language {
    Gql,
    Cypher,
}

fn execute(
    session: &Session,
    language: Language,
    query: &str,
    params: &HashMap<String, Value>,
) -> grafeo_common::utils::error::Result<grafeo_engine::database::QueryResult> {
    match language {
        Language::Gql => session.execute_with_params(query, params.clone()),
        Language::Cypher => session.execute_cypher_with_params(query, params.clone()),
    }
}

fn plan_of(
    session: &Session,
    language: Language,
    query: &str,
    params: &HashMap<String, Value>,
) -> String {
    let result = execute(session, language, &format!("EXPLAIN {query}"), params)
        .unwrap_or_else(|error| panic!("EXPLAIN {query}: {error}"));
    result.rows()[0][0].as_str().unwrap().to_string()
}

/// The rows as text, each cell its debug form, sorted: no order is owed without `ORDER BY`.
fn rows_of(
    session: &Session,
    language: Language,
    query: &str,
    params: &HashMap<String, Value>,
) -> Vec<String> {
    let result = execute(session, language, query, params)
        .unwrap_or_else(|error| panic!("{language:?} {query}: {error}"));
    let mut rows: Vec<String> = result.rows().iter().map(|row| format!("{row:?}")).collect();
    rows.sort();
    rows
}

fn text(value: &str) -> Value {
    Value::from(value)
}

/// The rows a model gives for `wanted`, in the form [`rows_of`] prints them.
fn expected(wanted: &[Vec<Value>]) -> Vec<String> {
    let mut rows: Vec<String> = wanted.iter().map(|row| format!("{row:?}")).collect();
    rows.sort();
    rows
}

fn no_params() -> HashMap<String, Value> {
    HashMap::new()
}

/// A session with the rule off.
fn unturned(db: &GrafeoDB) -> Session {
    let mut session = db.session();
    session.set_planner_rules(PlannerRules::all().with(PlannerRule::SoughtEnd, false));
    session
}

/// A session with every rule off.
fn plain(db: &GrafeoDB) -> Session {
    let mut session = db.session();
    session.set_planner_rules(PlannerRules::plain());
    session
}

/// Whether the plan starts at `far` and walks back to the source.
fn starts_at(plan: &str, far: &str) -> bool {
    plan.contains(&format!("Expand ({far})"))
}

/// Rows right under every rule set; then the plan turned, with its index read.
fn rows_and_turn(
    db: &GrafeoDB,
    stage: &str,
    language: Language,
    query: &str,
    wanted: &[Vec<Value>],
) {
    let params = no_params();
    let wanted = expected(wanted);
    for (name, session) in [
        ("all rules", db.session()),
        ("rule off", unturned(db)),
        ("plain", plain(db)),
    ] {
        assert_eq!(
            rows_of(&session, language, query, &params),
            wanted,
            "{stage}, {name}, {language:?}: {query}"
        );
    }
    let plan = plan_of(&db.session(), language, query, &params);
    assert!(
        starts_at(&plan, "m") && plan.contains("[index: k]"),
        "{stage}, {language:?}: {query} starts at the far end\n{plan}"
    );
}

// ---------------------------------------------------------------------------
// A source's label still counts
// ---------------------------------------------------------------------------

/// Jules has the edge and lacks `:Start`: the label moved from a scan to a check, and still decides.
#[test]
fn a_source_without_its_label_is_not_returned() {
    let wanted = [vec![text("Alix")], vec![text("Gus")], vec![text("Vincent")]];
    hot_and_reopened(|stage, db| {
        for (language, query) in [
            (
                Language::Gql,
                "MATCH (a:Start)-[:R]->(m:Far {k: 7}) RETURN a.name",
            ),
            (
                Language::Cypher,
                "MATCH (a:Start)-[:R]->(m:Far {k: 7}) RETURN a.name",
            ),
            (
                Language::Gql,
                "MATCH (a:Start)-[:R]->(m:Far) WHERE m.k = 7 RETURN a.name",
            ),
            (
                Language::Cypher,
                "MATCH (a:Start)-[:R]->(m:Far) WHERE m.k = 7 RETURN a.name",
            ),
            (
                Language::Gql,
                "MATCH (a)-[:R]->(m:Far {k: 7}) WHERE a:Start RETURN a.name",
            ),
            (
                Language::Cypher,
                "MATCH (a)-[:R]->(m) WHERE m:Far AND a:Start AND m.k = 7 RETURN a.name",
            ),
        ] {
            rows_and_turn(db, stage, language, query, &wanted);
        }
    });
}

#[test]
fn a_source_with_two_labels() {
    let wanted = [vec![text("Vincent")]];
    hot_and_reopened(|stage, db| {
        for (language, query) in [
            (
                Language::Cypher,
                "MATCH (a:Start:Other)-[:R]->(m:Far {k: 7}) RETURN a.name",
            ),
            (
                Language::Gql,
                "MATCH (a:Start&Other)-[:R]->(m:Far {k: 7}) RETURN a.name",
            ),
            (
                Language::Gql,
                "MATCH (a:Start)-[:R]->(m:Far {k: 7}) WHERE a:Other RETURN a.name",
            ),
        ] {
            rows_and_turn(db, stage, language, query, &wanted);
        }
    });
}

/// A union on the source is a check like one label, and a label nobody has gives no row.
#[test]
fn a_source_label_union_and_a_label_no_node_has() {
    let db = in_memory();
    let union = "MATCH (a)-[:R]->(m:Far {k: 7}) WHERE a:Start OR a:Other RETURN a.name";
    rows_and_turn(
        &db,
        "hot",
        Language::Gql,
        union,
        &[
            vec![text("Alix")],
            vec![text("Gus")],
            vec![text("Vincent")],
            vec![text("Jules")],
        ],
    );
    let nobody = "MATCH (a:Nobody)-[:R]->(m:Far {k: 7}) RETURN a.name";
    for session in [db.session(), unturned(&db), plain(&db)] {
        assert_eq!(
            rows_of(&session, Language::Gql, nobody, &no_params()),
            Vec::<String>::new()
        );
    }
}

// ---------------------------------------------------------------------------
// The hop itself
// ---------------------------------------------------------------------------

#[test]
fn each_direction_finds_what_it_finds_without_the_turn() {
    hot_and_reopened(|stage, db| {
        rows_and_turn(
            db,
            stage,
            Language::Gql,
            "MATCH (a:Start)<-[:S]-(m:Far {k: 7}) RETURN a.name",
            &[vec![text("Alix")]],
        );
        // Undirected: both types' edges, Alix twice for her two edges of either way. A hop of no named
        // type has no count to compare, so it stays at the source: `key_holders` has its rows.
        rows_and_turn(
            db,
            stage,
            Language::Gql,
            "MATCH (a:Start)-[r:R|S]-(m:Far {k: 7}) RETURN a.name, r.w",
            &[
                vec![text("Alix"), Value::Int64(1)],
                vec![text("Gus"), Value::Int64(2)],
                vec![text("Vincent"), Value::Int64(3)],
                vec![text("Alix"), Value::Int64(9)],
            ],
        );
        rows_and_turn(
            db,
            stage,
            Language::Cypher,
            "MATCH (m:Far {k: 7})<-[:R]-(a:Start) RETURN a.name",
            &[vec![text("Alix")], vec![text("Gus")], vec![text("Vincent")]],
        );
    });
}

/// A self-loop gives one row, walked either way and undirected. `m` and `a` are one node, which holds
/// the source's label and the far label.
#[test]
fn a_self_loop_gives_one_row() {
    let wanted = [vec![text("Mia"), Value::Int64(11)]];
    hot_and_reopened(|stage, db| {
        for query in [
            "MATCH (a:Start)-[r:L]->(m:Far {k: 70}) RETURN a.name, r.w",
            "MATCH (a:Start)<-[r:L]-(m:Far {k: 70}) RETURN a.name, r.w",
            "MATCH (a:Start)-[r:L]-(m:Far {k: 70}) RETURN a.name, r.w",
        ] {
            rows_and_turn(db, stage, Language::Gql, query, &wanted);
        }
    });
}

#[test]
fn parallel_edges_each_give_a_row() {
    hot_and_reopened(|stage, db| {
        rows_and_turn(
            db,
            stage,
            Language::Gql,
            "MATCH (a:Start)-[r:R]->(m:Far {k: 8}) RETURN a.name, r.w",
            &[
                vec![text("Alix"), Value::Int64(5)],
                vec![text("Alix"), Value::Int64(6)],
            ],
        );
    });
}

#[test]
fn the_edge_variable_and_a_filter_on_the_edge() {
    hot_and_reopened(|stage, db| {
        rows_and_turn(
            db,
            stage,
            Language::Gql,
            "MATCH (a:Start)-[r:R]->(m:Far {k: 7}) WHERE r.w = 2 RETURN a.name, r.w",
            &[vec![text("Gus"), Value::Int64(2)]],
        );
        rows_and_turn(
            db,
            stage,
            Language::Cypher,
            "MATCH (a:Start)-[r:R {w: 4}]->(m:Far {k: 7}) RETURN a.name",
            &[],
        );
    });
}

/// Filters on both ends, each on the right side of the turn, and an unrelated column kept.
#[test]
fn filters_on_both_ends_hold_after_the_turn() {
    rows_and_turn(
        &in_memory(),
        "hot",
        Language::Gql,
        "MATCH (a:Start)-[:R]->(m:Far {k: 7}) WHERE a.age > 35 AND m.k < 100 RETURN a.name, m.k",
        &[
            vec![text("Gus"), Value::Int64(7)],
            vec![text("Vincent"), Value::Int64(7)],
        ],
    );
}

// ---------------------------------------------------------------------------
// The key
// ---------------------------------------------------------------------------

#[test]
fn a_key_several_nodes_hold_none_hold_and_a_null_key() {
    hot_and_reopened(|stage, db| {
        rows_and_turn(
            db,
            stage,
            Language::Gql,
            "MATCH (a:Start)-[:R]->(m:Far {k: 9}) RETURN a.name",
            &[vec![text("Alix")], vec![text("Gus")]],
        );
        rows_and_turn(
            db,
            stage,
            Language::Gql,
            "MATCH (a:Start)-[:R]->(m:Far {k: 123456}) RETURN a.name",
            &[],
        );
        let query = "MATCH (a:Start)-[:R]->(m:Far {k: $v}) RETURN a.name";
        let params = HashMap::from([("v".to_string(), Value::Null)]);
        for session in [db.session(), unturned(db), plain(db)] {
            assert_eq!(
                rows_of(&session, Language::Gql, query, &params),
                Vec::<String>::new(),
                "{stage}: a NULL key finds nothing"
            );
        }
    });
}

/// A key the index does not read alike is found as `=` finds it: one type's key against another's.
/// A float of the value equals the integer 7; a string equals no number (GA04).
#[test]
fn a_key_of_another_kind_finds_what_equals_finds() {
    for (key, count) in [("7.0", 3), ("'7'", 0), ("'07'", 0)] {
        let db = in_memory();
        let query = format!("MATCH (a:Start)-[:R]->(m:Far) WHERE m.k = {key} RETURN a.name");
        let wanted = rows_of(&plain(&db), Language::Gql, &query, &no_params());
        assert_eq!(wanted.len(), count, "{key} against 7");
        assert_eq!(
            rows_of(&db.session(), Language::Gql, &query, &no_params()),
            wanted,
            "{key}"
        );
        assert_eq!(
            rows_of(&unturned(&db), Language::Gql, &query, &no_params()),
            wanted,
            "{key}"
        );
    }
}

/// The logical plan is cached across parameter values, so the plan the first value chose serves the
/// others: a rare key, a common key, an absent key and NULL, in each order.
#[test]
fn a_cached_plan_serves_other_values() {
    let db = in_memory();
    run_all(
        &db.session(),
        &["UNWIND range(1, 300) AS i INSERT (:Start {s: i, name: 'c'})-[:R]->(:Far {k: 100})"],
    );
    let query = "MATCH (a:Start)-[:R]->(m:Far {k: $v}) RETURN count(a) AS rows";
    let values = [
        (Value::Int64(7), 3),
        (Value::Int64(100), 300),
        (Value::Int64(123_456), 0),
        (Value::Null, 0),
    ];
    let count = |session: &Session, value: &Value| -> i64 {
        let params = HashMap::from([("v".to_string(), value.clone())]);
        execute(session, Language::Gql, query, &params)
            .unwrap()
            .rows()[0][0]
            .as_int64()
            .unwrap()
    };
    for first in 0..values.len() {
        let session = db.session();
        for turn in 0..values.len() * 2 {
            let (value, wanted) = &values[(first + turn) % values.len()];
            assert_eq!(
                count(&session, value),
                *wanted,
                "first {first}, turn {turn}: {value:?}"
            );
        }
    }
    for (value, wanted) in &values {
        assert_eq!(count(&plain(&db), value), *wanted, "plain: {value:?}");
    }
}

// ---------------------------------------------------------------------------
// Input rows
// ---------------------------------------------------------------------------

/// Each input row gets its own lookup: repeated keys keep their rows, NULL finds nothing.
#[test]
fn unwind_keeps_the_multiplicity_of_its_keys() {
    let query =
        "UNWIND [7, 7, 8, 123456, null] AS x MATCH (a:Start)-[:R]->(m:Far {k: x}) RETURN x, a.name";
    let row = |x: i64, name: &str| vec![Value::Int64(x), text(name)];
    let wanted = [
        row(7, "Alix"),
        row(7, "Gus"),
        row(7, "Vincent"),
        row(7, "Alix"),
        row(7, "Gus"),
        row(7, "Vincent"),
        row(8, "Alix"),
        row(8, "Alix"),
    ];
    hot_and_reopened(|stage, db| {
        rows_and_turn(db, stage, Language::Gql, query, &wanted);
        rows_and_turn(db, stage, Language::Cypher, query, &wanted);
    });
}

#[test]
fn unwind_of_nothing_gives_nothing() {
    let db = in_memory();
    for query in [
        "UNWIND [] AS x MATCH (a:Start)-[:R]->(m:Far {k: x}) RETURN x, a.name",
        "UNWIND [null] AS x MATCH (a:Start)-[:R]->(m:Far {k: x}) RETURN x, a.name",
    ] {
        for session in [db.session(), unturned(&db), plain(&db)] {
            assert_eq!(
                rows_of(&session, Language::Gql, query, &no_params()),
                Vec::<String>::new(),
                "{query}"
            );
        }
    }
}

// ---------------------------------------------------------------------------
// Versions
// ---------------------------------------------------------------------------

const AT_SEVEN: &str = "MATCH (a:Start)-[r:R]->(m:Far {k: 7}) RETURN a.name, r.w";

/// The rows at seven as a session sees them, under the rules, which must agree.
fn at_seven(sessions: &[&Session], stage: &str) -> Vec<String> {
    let first = rows_of(sessions[0], Language::Gql, AT_SEVEN, &no_params());
    for session in &sessions[1..] {
        assert_eq!(
            rows_of(session, Language::Gql, AT_SEVEN, &no_params()),
            first,
            "{stage}"
        );
    }
    first
}

fn seven(rows: &[(&str, i64)]) -> Vec<String> {
    expected(
        &rows
            .iter()
            .map(|(name, weight)| vec![text(name), Value::Int64(*weight)])
            .collect::<Vec<_>>(),
    )
}

/// An old snapshot sees the label, the edge and the far key as they were.
#[test]
fn an_old_snapshot_reads_what_it_saw() {
    let db = in_memory();
    let mut turned = db.session();
    let mut off = unturned(&db);
    turned.begin_transaction().unwrap();
    off.begin_transaction().unwrap();
    let before = seven(&[("Alix", 1), ("Gus", 2), ("Vincent", 3)]);
    assert_eq!(at_seven(&[&turned, &off], "before"), before);
    run_all(
        &db.session(),
        &[
            // A label removed from a source, a label added to one that lacked it, an edge deleted
            // and an edge added.
            "MATCH (a {name: 'Vincent'}) REMOVE a:Start",
            "MATCH (j {name: 'Jules'}) SET j:Start",
            "MATCH ({name: 'Gus'})-[r:R {w: 2}]->() DELETE r",
            "MATCH (a {name: 'Alix'}), (m:Far {k: 7}) INSERT (a)-[:R {w: 20}]->(m)",
        ],
    );
    assert_eq!(at_seven(&[&turned, &off], "the snapshot"), before);
    turned.rollback().unwrap();
    off.rollback().unwrap();
    assert_eq!(
        at_seven(&[&db.session(), &unturned(&db), &plain(&db)], "now"),
        seven(&[("Alix", 1), ("Alix", 20), ("Jules", 4)])
    );
}

/// The far key's value moved: an old snapshot finds the node by the old value, a new reader by the new.
#[test]
fn a_far_key_changed_after_a_snapshot() {
    let db = in_memory();
    let mut old = db.session();
    let mut old_off = unturned(&db);
    old.begin_transaction().unwrap();
    old_off.begin_transaction().unwrap();
    assert_eq!(at_seven(&[&old], "before").len(), 3);
    assert_eq!(at_seven(&[&old_off], "before").len(), 3);
    run_all(&db.session(), &["MATCH (m:Far {k: 7}) SET m.k = 77"]);
    assert_eq!(at_seven(&[&old], "the snapshot").len(), 3);
    assert_eq!(at_seven(&[&old_off], "the snapshot").len(), 3);
    old.rollback().unwrap();
    old_off.rollback().unwrap();
    assert_eq!(
        at_seven(&[&db.session(), &plain(&db)], "now"),
        Vec::<String>::new()
    );
    let moved = "MATCH (a:Start)-[r:R]->(m:Far {k: 77}) RETURN a.name, r.w";
    assert_eq!(
        rows_of(&db.session(), Language::Gql, moved, &no_params()),
        seven(&[("Alix", 1), ("Gus", 2), ("Vincent", 3)])
    );
}

#[test]
fn a_past_epoch_reads_what_it_saw() {
    let db = GrafeoDB::with_config(
        Config::in_memory().with_history_retention(grafeo_engine::HistoryRetention::Forever),
    )
    .unwrap();
    run_all(&db.session(), GRAPH);
    let epoch = db.current_epoch();
    run_all(
        &db.session(),
        &[
            "MATCH (a {name: 'Vincent'}) REMOVE a:Start",
            "MATCH ({name: 'Gus'})-[r:R {w: 2}]->() DELETE r",
            "MATCH (j {name: 'Jules'}) SET j:Start",
        ],
    );
    let then = db.session();
    let then_off = unturned(&db);
    then.set_viewing_epoch(epoch);
    then_off.set_viewing_epoch(epoch);
    assert_eq!(
        at_seven(&[&then, &then_off], "the epoch"),
        seven(&[("Alix", 1), ("Gus", 2), ("Vincent", 3)])
    );
    then.clear_viewing_epoch();
    assert_eq!(
        at_seven(&[&then], "now"),
        seven(&[("Alix", 1), ("Jules", 4)])
    );
}

/// A transaction sees its own edge, label and deletion, and a rollback takes them back.
#[test]
fn own_writes_are_seen_and_a_rollback_takes_them_back() {
    let db = in_memory();
    let mut session = db.session();
    session.begin_transaction().unwrap();
    run_all(
        &session,
        &[
            "MATCH (j {name: 'Jules'}) SET j:Start",
            "MATCH (a {name: 'Alix'}), (m:Far {k: 7}) INSERT (a)-[:R {w: 30}]->(m)",
            "MATCH ({name: 'Gus'})-[r:R {w: 2}]->() DELETE r",
            "MATCH (a {name: 'Vincent'}) REMOVE a:Start",
        ],
    );
    let own = seven(&[("Alix", 1), ("Alix", 30), ("Jules", 4)]);
    assert_eq!(at_seven(&[&session], "own writes"), own);
    assert_eq!(
        at_seven(&[&db.session(), &plain(&db)], "another reader"),
        seven(&[("Alix", 1), ("Gus", 2), ("Vincent", 3)])
    );
    session.rollback().unwrap();
    assert_eq!(
        at_seven(&[&session, &plain(&db)], "rolled back"),
        seven(&[("Alix", 1), ("Gus", 2), ("Vincent", 3)])
    );
}

/// A far node of an uncommitted write is found through the index, and one a transaction deleted is not.
#[test]
fn own_far_nodes_are_found_through_the_index() {
    let db = in_memory();
    let mut session = db.session();
    session.begin_transaction().unwrap();
    run_all(
        &session,
        &[
            "MATCH (a {name: 'Alix'}) INSERT (a)-[:R {w: 40}]->(:Far {k: 400})",
            "MATCH (m:Far {k: 8}) DETACH DELETE m",
        ],
    );
    let found = |query: &str| rows_of(&session, Language::Gql, query, &no_params());
    assert_eq!(
        found("MATCH (a:Start)-[r:R]->(m:Far {k: 400}) RETURN a.name, r.w"),
        seven(&[("Alix", 40)])
    );
    assert_eq!(
        found("MATCH (a:Start)-[r:R]->(m:Far {k: 8}) RETURN a.name, r.w"),
        Vec::<String>::new()
    );
    session.commit().unwrap();
    let query = "MATCH (a:Start)-[r:R]->(m:Far {k: 400}) RETURN a.name, r.w";
    assert_eq!(
        rows_of(&plain(&db), Language::Gql, query, &no_params()),
        seven(&[("Alix", 40)])
    );
}

// ---------------------------------------------------------------------------
// Shapes that must not turn
// ---------------------------------------------------------------------------

/// Rows as the plain plan's, and the plan walks from the source.
fn stays_at_the_source(db: &GrafeoDB, language: Language, query: &str) {
    let params = no_params();
    assert_eq!(
        rows_of(&db.session(), language, query, &params),
        rows_of(&plain(db), language, query, &params),
        "{query}"
    );
    let plan = plan_of(&db.session(), language, query, &params);
    assert!(
        !starts_at(&plan, "m"),
        "{query} stays at the source\n{plan}"
    );
}

#[test]
fn a_variable_length_hop_stays() {
    let db = in_memory();
    stays_at_the_source(
        &db,
        Language::Gql,
        "MATCH (a:Start)-[:R*1..2]->(m:Far {k: 7}) RETURN a.name",
    );
    stays_at_the_source(
        &db,
        Language::Cypher,
        "MATCH (a:Start)-[:R*1..2]->(m:Far {k: 7}) RETURN a.name",
    );
}

#[test]
fn a_named_path_stays() {
    let db = in_memory();
    stays_at_the_source(
        &db,
        Language::Gql,
        "MATCH p = (a:Start)-[:R]->(m:Far {k: 7}) RETURN a.name, length(p)",
    );
    stays_at_the_source(
        &db,
        Language::Cypher,
        "MATCH p = (a:Start)-[:R]->(m:Far {k: 7}) RETURN a.name, length(p)",
    );
}

#[test]
fn an_input_that_writes_stays() {
    let db = in_memory();
    let query =
        "INSERT (:Audit {at: 1}) WITH 1 AS x MATCH (a:Start)-[:R]->(m:Far {k: 7}) RETURN a.name";
    let plan = plan_of(&db.session(), Language::Gql, query, &no_params());
    assert!(!starts_at(&plan, "m"), "{plan}");
    assert_eq!(
        rows_of(&db.session(), Language::Gql, query, &no_params()),
        expected(&[vec![text("Alix")], vec![text("Gus")], vec![text("Vincent")]])
    );
}

#[test]
fn a_source_already_bound_stays() {
    let db = in_memory();
    stays_at_the_source(
        &db,
        Language::Gql,
        "MATCH (a:Start {s: 1}) MATCH (a)-[:R]->(m:Far {k: 7}) RETURN a.name",
    );
    stays_at_the_source(
        &db,
        Language::Gql,
        "MATCH (a:Start) WITH a MATCH (a)-[:R]->(m:Far {k: 7}) RETURN a.name",
    );
}

/// A store that keeps no backward edges has nothing to walk back on, for an outgoing hop.
#[test]
fn no_backward_edges_keeps_an_outgoing_hop() {
    let db = GrafeoDB::with_config(Config::in_memory().without_backward_edges()).unwrap();
    run_all(&db.session(), GRAPH);
    stays_at_the_source(
        &db,
        Language::Gql,
        "MATCH (a:Start)-[:R]->(m:Far {k: 7}) RETURN a.name",
    );
}

/// Without an index that serves the far end there is nothing to start from.
#[test]
fn no_serving_index_keeps_the_source() {
    let db = GrafeoDB::new_in_memory();
    run_all(&db.session(), &GRAPH[..2]);
    stays_at_the_source(
        &db,
        Language::Gql,
        "MATCH (a:Start)-[:R]->(m:Far {k: 7}) RETURN a.name",
    );
    run_all(
        &db.session(),
        &["CREATE INDEX on_other FOR (n:Other) ON (n.k)"],
    );
    stays_at_the_source(
        &db,
        Language::Gql,
        "MATCH (a:Start)-[:R]->(m:Far {k: 7}) RETURN a.name",
    );
}

// ---------------------------------------------------------------------------
// Checks that may raise
// ---------------------------------------------------------------------------

/// Whether a query fails, and its rows when it does not.
fn outcome(session: &Session, query: &str) -> Result<Vec<String>, ()> {
    match session.execute(query) {
        Ok(result) => {
            let mut rows: Vec<String> =
                result.rows().iter().map(|row| format!("{row:?}")).collect();
            rows.sort();
            Ok(rows)
        }
        Err(_) => Err(()),
    }
}

/// A check that may raise is one the turn must not move: it fails or holds as the plain plan's does.
/// Jules's `zero` is 0, and only the crowd's rows have no `zero` at all.
#[test]
fn a_check_that_may_raise_decides_as_without_the_turn() {
    let db = in_memory();
    for query in [
        "MATCH (a:Start)-[:R]->(m:Far {k: 7}) WHERE a.age / a.zero > 0 RETURN a.name",
        "MATCH (a:Other)-[:R]->(m:Far {k: 7}) WHERE a.age / a.zero > 0 RETURN a.name",
        "MATCH (a)-[:R]->(m:Far {k: 7}) WHERE a.age / a.zero > 0 RETURN a.name",
    ] {
        assert_eq!(
            outcome(&db.session(), query),
            outcome(&plain(&db), query),
            "{query}"
        );
    }
}

/// A written check guards the one after it, with the turn as without.
#[test]
fn a_written_check_still_guards_the_next() {
    let db = in_memory();
    let query = "MATCH (a:Other)-[:R]->(m:Far {k: 7}) WHERE a.zero <> 0 AND a.age / a.zero > 0 RETURN a.name";
    let wanted = expected(&[vec![text("Vincent")]]);
    for session in [db.session(), unturned(&db), plain(&db)] {
        assert_eq!(outcome(&session, query), Ok(wanted.clone()));
    }
}

// ---------------------------------------------------------------------------
// LIMIT and SKIP
// ---------------------------------------------------------------------------

/// With an order the rows are those of the plain plan; without one, the count.
#[test]
fn limit_and_skip_give_the_rows_or_the_count_they_owe() {
    let db = in_memory();
    for query in [
        "MATCH (a:Start)-[:R]->(m:Far {k: 7}) RETURN a.name ORDER BY a.name LIMIT 2",
        "MATCH (a:Start)-[:R]->(m:Far {k: 7}) RETURN a.name ORDER BY a.name DESC SKIP 1",
        "MATCH (a:Start)-[:R]->(m:Far {k: 7}) RETURN a.name ORDER BY a.name SKIP 1 LIMIT 1",
    ] {
        let wanted = db
            .session()
            .execute(query)
            .map(|result| format!("{:?}", result.rows()))
            .unwrap();
        for session in [unturned(&db), plain(&db)] {
            let plain_rows = session
                .execute(query)
                .map(|result| format!("{:?}", result.rows()))
                .unwrap();
            assert_eq!(wanted, plain_rows, "{query}: order and rows");
        }
    }
    for (query, count) in [
        (
            "MATCH (a:Start)-[:R]->(m:Far {k: 7}) RETURN a.name LIMIT 2",
            2,
        ),
        (
            "MATCH (a:Start)-[:R]->(m:Far {k: 7}) RETURN a.name SKIP 1",
            2,
        ),
        (
            "MATCH (a:Start)-[:R]->(m:Far {k: 7}) RETURN a.name LIMIT 10",
            3,
        ),
    ] {
        for session in [db.session(), plain(&db)] {
            assert_eq!(
                session.execute(query).unwrap().row_count(),
                count,
                "{query}"
            );
        }
    }
}

// ---------------------------------------------------------------------------
// The cost rule, by plan shape with fixed counts
// ---------------------------------------------------------------------------

mod cost_rule {
    use super::*;

    fn plan(db: &GrafeoDB, query: &str) -> String {
        plan_of(&db.session(), Language::Gql, query, &no_params())
    }

    /// The rows are the plain plan's in every form, so only the start differs.
    fn same_rows(db: &GrafeoDB, query: &str) {
        assert_eq!(
            rows_of(&db.session(), Language::Gql, query, &no_params()),
            rows_of(&plain(db), Language::Gql, query, &no_params()),
            "{query}"
        );
    }

    /// One start, sought through its own index, and a far key one node holds: 9.5 microseconds at the
    /// source against a lookup and a walk back.
    #[test]
    fn one_pinned_start_and_a_unique_far_key_stay_at_the_source() {
        let db = GrafeoDB::new_in_memory();
        run_all(
            &db.session(),
            &[
                "UNWIND range(1, 2000) AS i INSERT (:Start {s: i})-[:R]->(:Far {k: i})",
                "CREATE INDEX by_start FOR (n:Start) ON (n.s)",
                "CREATE INDEX by_far FOR (n:Far) ON (n.k)",
            ],
        );
        let query = "MATCH (a:Start {s: 7})-[:R]->(m:Far {k: 7}) RETURN a.s, m.k";
        let plan = plan(&db, query);
        assert!(!starts_at(&plan, "m"), "{plan}");
        same_rows(&db, query);
    }

    /// One hub among many starts, and no index on the source: the source's scan and the hub's list
    /// both go.
    #[test]
    fn a_hub_start_found_by_a_scan_turns() {
        let db = GrafeoDB::new_in_memory();
        run_all(
            &db.session(),
            &[
                "UNWIND range(1, 2000) AS i INSERT (:Start {s: i})",
                "UNWIND range(1, 5000) AS i INSERT (:Far {k: i})",
                "MATCH (a:Start {s: 1}), (m:Far) INSERT (a)-[:R]->(m)",
                "CREATE INDEX by_far FOR (n:Far) ON (n.k)",
            ],
        );
        let query = "MATCH (a:Start)-[:R]->(m:Far {k: 7}) WHERE a.s = 1 RETURN a.s, m.k";
        same_rows(&db, query);
        let plan = plan(&db, query);
        assert!(starts_at(&plan, "m"), "{plan}");
    }

    /// Fewer starts than the sample takes: it reads every one, the hub among them, so the rule has a
    /// reading. The variant that shows why the larger form has none.
    #[test]
    fn a_hub_among_few_starts_is_seen_by_the_sample() {
        let db = GrafeoDB::new_in_memory();
        run_all(
            &db.session(),
            &[
                "UNWIND range(1, 200) AS i INSERT (:Start {s: i})",
                "UNWIND range(1, 5000) AS i INSERT (:Far {k: i})",
                "MATCH (a:Start {s: 1}), (m:Far) INSERT (a)-[:R]->(m)",
                "CREATE INDEX by_far FOR (n:Far) ON (n.k)",
            ],
        );
        let query = "MATCH (a:Start)-[:R]->(m:Far {k: 7}) WHERE a.s = 1 RETURN a.s, m.k";
        same_rows(&db, query);
        let plan = plan(&db, query);
        assert!(starts_at(&plan, "m"), "{plan}");
    }

    /// A hub among 2,000 starts, whose edges reach `far` as the statements say.
    fn hub_among_many(statements: &[&str]) -> GrafeoDB {
        let db = GrafeoDB::new_in_memory();
        run_all(
            &db.session(),
            &[
                "UNWIND range(1, 2000) AS i INSERT (:Start {s: i})",
                "UNWIND range(1, 5000) AS i INSERT (:Far {k: i})",
                "CREATE INDEX by_far FOR (n:Far) ON (n.k)",
            ],
        );
        run_all(&db.session(), statements);
        db
    }

    const SCANNED_HUB: &str = "MATCH (a:Start)-[:R]->(m:Far {k: $v}) WHERE a.s = 1 RETURN m.k";

    /// The value may name the far hub, whose walk back is all its edges: the source stays.
    fn stays_for_each_value(db: &GrafeoDB, values: &[i64]) {
        for value in values {
            let params = HashMap::from([("v".to_string(), Value::Int64(*value))]);
            assert_eq!(
                rows_of(&db.session(), Language::Gql, SCANNED_HUB, &params),
                rows_of(&plain(db), Language::Gql, SCANNED_HUB, &params),
                "{value}"
            );
            let plan = plan_of(&db.session(), Language::Gql, SCANNED_HUB, &params);
            assert!(!starts_at(&plan, "m"), "{value}: {plan}");
        }
    }

    /// Every edge of the hub start goes to one far hub, by parallel edges.
    #[test]
    fn a_hub_start_to_one_far_hub_stays() {
        let db = hub_among_many(&[
            "MATCH (a:Start {s: 1}), (m:Far {k: 1}) UNWIND range(1, 5000) AS i INSERT (a)-[:R]->(m)",
        ]);
        stays_for_each_value(&db, &[1, 2]);
    }

    /// Half to the far hub and half to distinct far nodes.
    #[test]
    fn half_to_a_far_hub_and_half_distinct_stays() {
        let db = hub_among_many(&[
            "MATCH (a:Start {s: 1}), (m:Far {k: 1}) UNWIND range(1, 2500) AS i INSERT (a)-[:R]->(m)",
            "MATCH (a:Start {s: 1}), (m:Far) WHERE m.k > 2500 INSERT (a)-[:R]->(m)",
        ]);
        stays_for_each_value(&db, &[1, 3000]);
    }

    /// Edges to another label are ones the far nodes do not account for: they keep the source.
    #[test]
    fn edges_to_another_label_stay() {
        let db = hub_among_many(&[
            "UNWIND range(1, 5000) AS i INSERT (:Other {o: i})",
            "MATCH (a:Start {s: 1}), (m:Other) INSERT (a)-[:R]->(m)",
            "MATCH (a:Start {s: 1}), (m:Far {k: 7}) INSERT (a)-[:R]->(m)",
        ]);
        stays_for_each_value(&db, &[7]);
    }

    /// Another type's edges on the far nodes, even a hub of them, change nothing: the lengths read are
    /// the hop's own type's.
    #[test]
    fn another_types_edges_on_the_far_nodes_change_nothing() {
        let db = hub_among_many(&[
            "MATCH (a:Start {s: 1}), (m:Far) INSERT (a)-[:R]->(m)",
            "MATCH (a:Start {s: 2}), (m:Far) INSERT (a)-[:S]->(m), (m)-[:S]->(a)",
            "MATCH (a:Start {s: 3}), (m:Far {k: 7}) UNWIND range(1, 5000) AS i INSERT (a)-[:S]->(m)",
        ]);
        let params = HashMap::from([("v".to_string(), Value::Int64(7))]);
        assert_eq!(
            rows_of(&db.session(), Language::Gql, SCANNED_HUB, &params),
            rows_of(&plain(&db), Language::Gql, SCANNED_HUB, &params),
        );
        let plan = plan_of(&db.session(), Language::Gql, SCANNED_HUB, &params);
        assert!(starts_at(&plan, "m"), "{plan}");
    }

    /// Undirected, each edge counts in the lists of both its ends. The rows are the walk's whichever
    /// start it takes, also where the far key is the hub's own far node.
    #[test]
    fn an_undirected_hub_start_answers_as_the_walk() {
        let db = hub_among_many(&["MATCH (a:Start {s: 1}), (m:Far) INSERT (a)-[:R]->(m)"]);
        for query in [
            "MATCH (a:Start)-[:R]-(m:Far {k: 7}) WHERE a.s = 1 RETURN a.s, m.k",
            "MATCH (a:Start)-[:R]-(m:Far {k: 7}) RETURN a.s, m.k",
            "MATCH (m:Far {k: 7})-[:R]-(a:Start) RETURN a.s, m.k",
        ] {
            same_rows(&db, query);
            assert_eq!(
                rows_of(&db.session(), Language::Gql, query, &no_params()).len(),
                1,
                "{query}"
            );
        }
        // One far hub, undirected: its walk back is all the edges, and the source stays.
        let db = hub_among_many(&[
            "MATCH (a:Start {s: 1}), (m:Far {k: 1}) UNWIND range(1, 5000) AS i INSERT (a)-[:R]->(m)",
        ]);
        let query = "MATCH (a:Start)-[:R]-(m:Far {k: 1}) WHERE a.s = 1 RETURN a.s, m.k";
        same_rows(&db, query);
        let plan = plan(&db, query);
        assert!(!starts_at(&plan, "m"), "{plan}");
    }

    /// A far end of a union has no one label to sample, so the far side reads nothing and the source
    /// stays. The rows are the walk's.
    #[test]
    fn a_far_union_keeps_the_source() {
        let db = hub_among_many(&[
            "CREATE INDEX on_near FOR (n:Near) ON (n.k)",
            "UNWIND range(1, 100) AS i INSERT (:Near {k: i})",
            "MATCH (a:Start {s: 1}), (m) WHERE m:Far OR m:Near INSERT (a)-[:R]->(m)",
        ]);
        for query in [
            "MATCH (a:Start)-[:R]->(m) WHERE (m:Far OR m:Near) AND m.k = 7 RETURN a.s, m.k",
            "MATCH (a:Start)-[:R]->(m:Far|Near {k: 7}) RETURN a.s, m.k",
        ] {
            same_rows(&db, query);
            assert_eq!(
                rows_of(&db.session(), Language::Gql, query, &no_params()).len(),
                2,
                "{query}"
            );
            let plan = plan(&db, query);
            assert!(!starts_at(&plan, "m"), "{query}\n{plan}");
        }
    }

    /// A source with no label and no check has nothing to compare: it reads every node's list, which
    /// holds all the far end's, so it turns, onto a far hub too.
    #[test]
    fn no_label_and_no_check_always_turns() {
        let db = hub_among_many(&[
            "MATCH (a:Start {s: 1}), (m:Far {k: 1}) UNWIND range(1, 5000) AS i INSERT (a)-[:R]->(m)",
        ]);
        let query = "MATCH (a)-[:R]->(m:Far {k: 1}) RETURN a.s, m.k";
        same_rows(&db, query);
        let plan = plan(&db, query);
        assert!(starts_at(&plan, "m"), "{plan}");
    }

    /// With a check on such a source it turns too, as 0.6.1 did: on the far hub and on a value
    /// of one far node, the walk back reads fewer lists than every node's.
    #[test]
    fn no_label_and_a_check_turns_as_released() {
        let db = hub_among_many(&[
            "MATCH (a:Start {s: 1}), (m:Far {k: 1}) UNWIND range(1, 5000) AS i INSERT (a)-[:R]->(m)",
        ]);
        for query in [
            "MATCH (a)-[:R]->(m:Far {k: 1}) WHERE a.s = 1 RETURN a.s, m.k",
            "MATCH (a)-[:R]->(m:Far {k: 2}) WHERE a.s = 1 RETURN a.s, m.k",
            "MATCH (a {s: 1})-[:R]->(m:Far {k: 1}) RETURN a.s, m.k",
        ] {
            same_rows(&db, query);
            let plan = plan(&db, query);
            assert!(starts_at(&plan, "m"), "{query}\n{plan}");
        }
    }

    #[test]
    fn ten_thousand_starts_turn() {
        let db = GrafeoDB::new_in_memory();
        run_all(
            &db.session(),
            &[
                "UNWIND range(1, 10000) AS i INSERT (:Start {s: i})-[:R]->(:Far {k: i})",
                "CREATE INDEX by_far FOR (n:Far) ON (n.k)",
            ],
        );
        let query = "MATCH (a:Start)-[:R]->(m:Far {k: 7}) RETURN a.s, m.k";
        same_rows(&db, query);
        let plan = plan(&db, query);
        assert!(starts_at(&plan, "m"), "{plan}");
    }

    /// Ten sources, and a key thousands of far nodes hold: the source is the smaller side.
    #[test]
    fn a_common_far_key_and_ten_sources_stay() {
        let db = GrafeoDB::new_in_memory();
        run_all(
            &db.session(),
            &[
                "UNWIND range(1, 10) AS i INSERT (:Start {s: i})",
                "UNWIND range(1, 5000) AS i INSERT (:Far {k: 1, n: i})",
                "MATCH (a:Start), (m:Far) WHERE m.n % 10 = a.s % 10 INSERT (a)-[:R]->(m)",
                "CREATE INDEX by_far FOR (n:Far) ON (n.k)",
            ],
        );
        let query = "MATCH (a:Start)-[:R]->(m:Far {k: 1}) RETURN a.s, m.n";
        same_rows(&db, query);
        let plan = plan(&db, query);
        assert!(!starts_at(&plan, "m"), "{plan}");
    }

    /// `$v` may be the key of a node with thousands of edges into it, which the plan cannot know: one
    /// start stays at the source, as the edge-weighted sample of the backward degree reads high.
    #[test]
    fn one_start_and_a_hub_key_stay() {
        let db = GrafeoDB::new_in_memory();
        run_all(
            &db.session(),
            &[
                "INSERT (:Start {s: 1})",
                "UNWIND range(1, 200) AS i INSERT (:Far {k: i})",
                "UNWIND range(1, 5000) AS i INSERT (:Other {s: i})",
                "MATCH (a:Other), (m:Far {k: 1}) INSERT (a)-[:R]->(m)",
                "MATCH (a:Start), (m:Far) WHERE m.k <= 2 INSERT (a)-[:R]->(m)",
                "CREATE INDEX by_far FOR (n:Far) ON (n.k)",
            ],
        );
        let query = "MATCH (a:Start)-[:R]->(m:Far {k: $v}) RETURN a.s, m.k";
        for value in [1, 2, 3] {
            let params = HashMap::from([("v".to_string(), Value::Int64(value))]);
            assert_eq!(
                rows_of(&db.session(), Language::Gql, query, &params),
                rows_of(&plain(&db), Language::Gql, query, &params),
                "{value}"
            );
            let plan = plan_of(&db.session(), Language::Gql, query, &params);
            assert!(!starts_at(&plan, "m"), "{value}: {plan}");
        }
    }

    /// A key updated many times: the index holds old and pending values until a collection, so its
    /// length reads distinct values high and leans towards a turn. A common value and a small source
    /// label keep the source all the same, and the rows are right before and after the collection.
    #[test]
    fn a_churned_key_with_a_common_value_and_a_small_source_stays() {
        let db = GrafeoDB::new_in_memory();
        run_all(
            &db.session(),
            &[
                "UNWIND range(1, 10) AS i INSERT (:Start {s: i})",
                "UNWIND range(1, 2000) AS i INSERT (:Far {k: 1, n: i})",
                "MATCH (a:Start), (m:Far) WHERE m.n % 10 = a.s % 10 INSERT (a)-[:R]->(m)",
                "CREATE INDEX by_far FOR (n:Far) ON (n.k)",
            ],
        );
        let session = db.session();
        for round in 2..12 {
            session
                .execute(&format!("MATCH (m:Far) SET m.k = {round}"))
                .unwrap();
        }
        let query = "MATCH (a:Start)-[:R]->(m:Far {k: 11}) RETURN a.s, m.n";
        for stage in ["before a collection", "after a collection"] {
            same_rows(&db, query);
            assert_eq!(
                rows_of(&db.session(), Language::Gql, query, &no_params()).len(),
                2_000,
                "{stage}"
            );
            let plan = plan(&db, query);
            assert!(!starts_at(&plan, "m"), "{stage}: {plan}");
            db.gc();
        }
        let old = "MATCH (a:Start)-[:R]->(m:Far {k: 1}) RETURN a.s, m.n";
        assert_eq!(
            rows_of(&db.session(), Language::Gql, old, &no_params()).len(),
            0
        );
    }

    /// A key-wide index counts all nodes as candidates, which reads high and may keep the source. The
    /// rows hold whichever it chooses.
    #[test]
    fn a_key_wide_index_answers_the_same_rows() {
        let db = GrafeoDB::new_in_memory();
        run_all(&db.session(), &GRAPH[..2]);
        db.create_property_index("k").unwrap();
        same_rows(&db, "MATCH (a:Start)-[:R]->(m:Far {k: 7}) RETURN a.name");
        same_rows(&db, "MATCH (a:Start)-[:R]->(m:Far {k: 9}) RETURN a.name");
    }

    /// Today's rule for an unlabelled source stands: it turns whatever the numbers.
    #[test]
    fn an_unlabelled_source_still_turns() {
        let db = in_memory();
        let query = "MATCH (a)-[:R]->(m:Far {k: 7}) RETURN a.name";
        same_rows(&db, query);
        let plan = plan(&db, query);
        assert!(starts_at(&plan, "m"), "{plan}");
    }
}

// ---------------------------------------------------------------------------
// A check on the source lowers what the source costs
// ---------------------------------------------------------------------------

/// 512 starts, `keyed` far nodes of each with `k: 1`, and a long list on the first.
fn pinned_source(keyed: i64, degree: i64) -> GrafeoDB {
    let db = GrafeoDB::new_in_memory();
    for s in 0..512 {
        let start = db
            .create_node_with_props(&["Start"], [("s", Value::Int64(s))])
            .unwrap();
        for d in 0..degree.min(if s == 0 { degree } else { 64 }) {
            let key = if d < keyed { 1 } else { 1_000 + d };
            let far = db
                .create_node_with_props(&["Far"], [("k", Value::Int64(key))])
                .unwrap();
            db.create_edge(start, far, "R").unwrap();
        }
    }
    db.session()
        .execute("CREATE INDEX far_k FOR (n:Far) ON (n.k)")
        .unwrap();
    db
}

/// A source an equality pins reads few rows, with no index on its key too: the walk back through
/// every far node holding the key loses, whatever the source's list.
#[test]
fn a_source_an_equality_pins_stays_at_the_source() {
    for keyed in [1, 8] {
        for degree in [128, 256, 512, 1_024, 4_096] {
            let db = pinned_source(keyed, degree);
            for (language, query) in [
                (
                    Language::Gql,
                    "MATCH (a:Start {s: 0})-[:R]->(m:Far {k: 1}) RETURN m.k",
                ),
                (
                    Language::Gql,
                    "MATCH (a:Start)-[:R]->(m:Far {k: 1}) WHERE a.s = 0 RETURN m.k",
                ),
                (
                    Language::Cypher,
                    "MATCH (a:Start {s: 0})-[:R]->(m:Far {k: 1}) RETURN m.k",
                ),
            ] {
                stays_at_the_source(&db, language, query);
                assert_eq!(
                    rows_of(&db.session(), language, query, &no_params()).len(),
                    usize::try_from(keyed).unwrap(),
                    "keyed {keyed}, degree {degree}: {query}"
                );
            }
        }
    }
}

/// A label of many starts, one edge each, and a far key one node holds: the source's plan reads the
/// whole label, for an equality with no index too, and the walk back one list. It still turns, so the
/// price of a check does not keep every source that has one.
#[test]
fn a_rare_far_value_on_a_large_label_still_turns() {
    let db = GrafeoDB::new_in_memory();
    for s in 0..100_000 {
        let start = db
            .create_node_with_props(&["Start"], [("s", Value::Int64(s))])
            .unwrap();
        let far = db
            .create_node_with_props(&["Far"], [("k", Value::Int64(s))])
            .unwrap();
        db.create_edge(start, far, "R").unwrap();
    }
    db.session()
        .execute("CREATE INDEX far_k FOR (n:Far) ON (n.k)")
        .unwrap();
    for (query, rows) in [
        ("MATCH (a:Start)-[:R]->(m:Far {k: 77}) RETURN a.s", 1),
        (
            "MATCH (a:Start {s: 77})-[:R]->(m:Far {k: 77}) RETURN a.s",
            1,
        ),
        (
            "MATCH (a:Start {s: 78})-[:R]->(m:Far {k: 77}) RETURN a.s",
            0,
        ),
        (
            "MATCH (a:Start)-[:R]->(m:Far {k: 77}) WHERE a.s > 5 RETURN a.s",
            1,
        ),
        (
            "MATCH (a:Start)-[:R]->(m:Far {k: 77}) WHERE a.s > 500 RETURN a.s",
            0,
        ),
        (
            "MATCH (a)-[:R]->(m:Far {k: 77}) WHERE a.s = 77 RETURN a.s",
            1,
        ),
    ] {
        let found = rows_of(&db.session(), Language::Gql, query, &no_params());
        assert_eq!(found.len(), rows, "{query}");
        assert_eq!(
            found,
            rows_of(&unturned(&db), Language::Gql, query, &no_params()),
            "{query}"
        );
        let plan = plan_of(&db.session(), Language::Gql, query, &no_params());
        assert!(starts_at(&plan, "m"), "{query}\n{plan}");
    }
}

/// A check of the source and a conjunct that reads both ends filter the rows the turned expand
/// makes, the source's first, as they ran before; the target's own go with the seek.
#[test]
fn a_check_of_the_source_filters_after_the_turn() {
    let db = GrafeoDB::new_in_memory();
    for s in 0..20_000 {
        let start = db
            .create_node_with_props(&["Start"], [("s", Value::Int64(s))])
            .unwrap();
        let far = db
            .create_node_with_props(&["Far"], [("k", Value::Int64(s)), ("n", Value::Int64(100))])
            .unwrap();
        db.create_edge(start, far, "R").unwrap();
    }
    db.session()
        .execute("CREATE INDEX far_k FOR (n:Far) ON (n.k)")
        .unwrap();
    let query =
        "MATCH (a:Start)-[:R]->(m:Far {k: 77}) WHERE m.n > a.s AND a.s > 5 AND m.n > 0 RETURN a.s";
    let wanted = [vec![Value::Int64(77)]];
    let found = rows_of(&db.session(), Language::Gql, query, &no_params());
    assert_eq!(found, expected(&wanted));
    assert_eq!(
        found,
        rows_of(&plain(&db), Language::Gql, query, &no_params())
    );
    let plan = plan_of(&db.session(), Language::Gql, query, &no_params());
    assert!(starts_at(&plan, "m"), "{plan}");
    let at = |text: &str| {
        plan.find(text)
            .unwrap_or_else(|| panic!("{text} is not in the plan\n{plan}"))
    };
    assert!(
        at("a.s Gt 5") < at("m.n Gt a.s") && at("m.n Gt a.s") < at("Expand (m)"),
        "the source's check first, both above the expand\n{plan}"
    );
    assert!(
        at("Expand (m)") < at("m.n Gt 0"),
        "the target's own under it, with the seek\n{plan}"
    );
}

/// A source pinned by an equality, and a far value most far nodes hold: the far end is priced by a
/// probe of the value in its index, so the common value keeps the source, and a value one far node
/// holds still turns where the source's label is large.
fn pinned_against_a_common_value(starts: i64, common: i64) -> GrafeoDB {
    let db = GrafeoDB::new_in_memory();
    let ids: Vec<_> = (0..starts)
        .map(|s| {
            db.create_node_with_props(&["Start"], [("s", Value::Int64(s))])
                .unwrap()
        })
        .collect();
    for n in 0..common + 4_096 {
        let key = if n < common { 1 } else { n + 1_000 };
        let far = db
            .create_node_with_props(&["Far"], [("k", Value::Int64(key))])
            .unwrap();
        db.create_edge(ids[usize::try_from(n % starts).unwrap()], far, "R")
            .unwrap();
    }
    db.session()
        .execute("CREATE INDEX far_k FOR (n:Far) ON (n.k)")
        .unwrap();
    db
}

#[test]
fn a_common_far_value_keeps_a_pinned_source() {
    for (starts, common) in [(512, 4_096), (4_096, 32_768)] {
        let db = pinned_against_a_common_value(starts, common);
        for (language, query) in [
            (
                Language::Gql,
                "MATCH (a:Start {s: 3})-[:R]->(m:Far {k: 1}) RETURN m.k",
            ),
            (
                Language::Gql,
                "MATCH (a:Start)-[:R]->(m:Far {k: 1}) WHERE a.s = 3 RETURN m.k",
            ),
            (
                Language::Gql,
                "MATCH (a)-[:R]->(m:Far {k: 1}) WHERE a.s = 3 RETURN m.k",
            ),
            (
                Language::Cypher,
                "MATCH (a:Start {s: 3})-[:R]->(m:Far {k: 1}) RETURN m.k",
            ),
        ] {
            stays_at_the_source(&db, language, query);
        }
    }
}

/// The same graph, a value one far node holds: the probe finds one, so the walk back wins.
#[test]
fn a_rare_far_value_still_turns_a_large_pinned_source() {
    let db = GrafeoDB::new_in_memory();
    for s in 0..100_000 {
        let start = db
            .create_node_with_props(&["Start"], [("s", Value::Int64(s))])
            .unwrap();
        let far = db
            .create_node_with_props(
                &["Far"],
                [("k", Value::Int64(if s < 50_000 { 1 } else { s }))],
            )
            .unwrap();
        db.create_edge(start, far, "R").unwrap();
    }
    db.session()
        .execute("CREATE INDEX far_k FOR (n:Far) ON (n.k)")
        .unwrap();
    for query in [
        "MATCH (a:Start {s: 77777})-[:R]->(m:Far {k: 77777}) RETURN a.s",
        "MATCH (a:Start)-[:R]->(m:Far {k: 77777}) WHERE a.s = 77777 RETURN a.s",
    ] {
        let found = rows_of(&db.session(), Language::Gql, query, &no_params());
        assert_eq!(found.len(), 1, "{query}");
        assert_eq!(
            found,
            rows_of(&unturned(&db), Language::Gql, query, &no_params()),
            "{query}"
        );
        let plan = plan_of(&db.session(), Language::Gql, query, &no_params());
        assert!(starts_at(&plan, "m"), "{query}\n{plan}");
    }
    // The common value on the same graph keeps the source.
    stays_at_the_source(
        &db,
        Language::Gql,
        "MATCH (a:Start {s: 3})-[:R]->(m:Far {k: 1}) RETURN m.k",
    );
}

/// A far key most far nodes hold, and a label of few starts. The plan is made for any value, which
/// is the limit a kept plan has, so a common value may read more than it would have: the rows hold
/// for a common and a rare value, whichever comes first and through the same kept plan.
#[test]
fn a_common_far_value_on_a_small_label_gives_the_same_rows() {
    let db = GrafeoDB::new_in_memory();
    let starts: Vec<_> = (0..64)
        .map(|s| {
            db.create_node_with_props(&["Start"], [("s", Value::Int64(s))])
                .unwrap()
        })
        .collect();
    for n in 0..8_192_i64 {
        let key = if n < 4_096 { 1 } else { n };
        let far = db
            .create_node_with_props(&["Far"], [("k", Value::Int64(key))])
            .unwrap();
        let start = starts[usize::try_from(n % 64).unwrap()];
        db.create_edge(start, far, "R").unwrap();
    }
    db.session()
        .execute("CREATE INDEX far_k FOR (n:Far) ON (n.k)")
        .unwrap();
    for query in [
        "MATCH (a:Start)-[:R]->(m:Far {k: $v}) RETURN a.s, m.k",
        "MATCH (a:Start {s: 3})-[:R]->(m:Far {k: $v}) RETURN a.s, m.k",
        "MATCH (a:Start)-[:R]->(m:Far {k: $v}) WHERE a.s > 60 RETURN a.s, m.k",
    ] {
        for order in [[1, 5_000], [5_000, 1]] {
            let kept = db.session();
            for value in order {
                let params = HashMap::from([("v".to_string(), Value::Int64(value))]);
                assert_eq!(
                    rows_of(&kept, Language::Gql, query, &params),
                    rows_of(&plain(&db), Language::Gql, query, &params),
                    "{query} with {value}"
                );
            }
        }
    }
}

/// No label on the source and a check few rows pass: the source's plan reads every node and few
/// edges, the walk back every edge into the far node.
#[test]
fn a_selective_check_on_a_source_with_no_label_stays_at_the_source() {
    let db = GrafeoDB::new_in_memory();
    let hub = db
        .create_node_with_props(&["Far"], [("id", Value::Int64(1))])
        .unwrap();
    for i in 0..40_000 {
        let age = if i % 2_000 == 0 { 95 } else { 30 };
        let a = db
            .create_node_with_props(&["Person"], [("age", Value::Int64(age))])
            .unwrap();
        if i % 2 == 0 {
            db.create_edge(a, hub, "R").unwrap();
        }
    }
    db.create_property_index("id").unwrap();
    stays_at_the_source(
        &db,
        Language::Gql,
        "MATCH (a)-[:R]->(m {id: 1}) WHERE a.age > 90 RETURN a.age",
    );
}
