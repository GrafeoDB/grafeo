//! A SQL/PGQ equality answers alike, its 22G04 included, whatever structure reads it (GA04 step b, the
//! ruling of 2026-11-06): a query's answer never depends on the plan. SQL raises "values not
//! comparable" when a comparison meets a value of another kind (ISO/IEC 9075-2 8.2), so a key column
//! or a property index may answer an equality only when it knows it holds no value of another kind;
//! otherwise the plan meets the values and raises. Each case runs without an index and with one, each
//! with the planner's choice, with the key-column read turned off, and with the plain scan, hot and
//! reopened. GQL and Cypher answer false across kinds and keep the index.
//!
//! ```bash
//! cargo test -p grafeo-engine --all-features --test main sql_pgq_lookups_across_kinds::
//! ```

use std::collections::HashMap;

use grafeo_common::types::Value;
use grafeo_common::utils::gqlstatus::GqlStatus;
use grafeo_engine::GrafeoDB;
use grafeo_engine::database::QueryResult;
use grafeo_engine::query::optimizer::{PlannerRule, PlannerRules};
use grafeo_engine::session::Session;

const NOT_COMPARABLE: &str = "22G04";

/// Rows of a row group: a column past this spans groups.
const GROUP: i64 = 65_536;

/// Numbers of two numeric kinds, a node without `k`, and a string `k` on another label, which no
/// `L` query reads.
const NUMBERS: &str = "INSERT (:L {row: 0, k: 1}), (:L {row: 1, k: 5}), (:L {row: 2, k: 5}), \
     (:L {row: 3, k: 2.5}), (:L {row: 4}), (:Other {row: -1, k: 'x'})";

fn answer(result: grafeo_common::utils::error::Result<QueryResult>) -> Result<Vec<i64>, String> {
    match result {
        Ok(result) => {
            let mut rows: Vec<i64> = result
                .rows()
                .iter()
                .map(|row| row[0].as_int64().expect("a row number"))
                .collect();
            rows.sort_unstable();
            Ok(rows)
        }
        Err(error) => Err(GqlStatus::from(&error).as_str().to_string()),
    }
}

fn sql_query(condition: &str) -> String {
    format!("SELECT * FROM GRAPH_TABLE (MATCH (n:L) WHERE {condition} COLUMNS (n.row AS row))")
}

fn sql(
    session: &Session,
    condition: &str,
    params: &HashMap<String, Value>,
) -> Result<Vec<i64>, String> {
    answer(session.execute_sql_with_params(&sql_query(condition), params.clone()))
}

fn session_with(db: &GrafeoDB, rules: PlannerRules) -> Session {
    let mut session = db.session();
    session.set_planner_rules(rules);
    session
}

/// The three plans a database offers: the planner's choice, no key-column read, the plain scan.
fn sessions(db: &GrafeoDB) -> [(&'static str, Session); 3] {
    [
        ("planner", db.session()),
        (
            "no key column",
            session_with(
                db,
                PlannerRules::all().with(PlannerRule::KeyColumnLookup, false),
            ),
        ),
        ("plain", session_with(db, PlannerRules::plain())),
    ]
}

fn run_all(db: &GrafeoDB, statements: &[&str]) {
    for statement in statements {
        db.execute(statement)
            .unwrap_or_else(|error| panic!("{statement}: {error}"));
    }
}

/// `setup` on a database without and with an index on `L.k`, hot and reopened: every route gives
/// `want` for each condition.
fn every_route_answers(
    name: &str,
    setup: &[&str],
    conditions: &[(&str, Result<Vec<i64>, String>)],
    params: &HashMap<String, Value>,
) -> Vec<String> {
    let mut wrong = Vec::new();
    for indexed in [false, true] {
        let fill = |db: &GrafeoDB| {
            run_all(db, setup);
            if indexed {
                db.execute("CREATE INDEX on_k FOR (n:L) ON (n.k)").unwrap();
            }
        };
        let hot = GrafeoDB::new_in_memory();
        fill(&hot);
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("lookups.grafeo");
        {
            let db = GrafeoDB::open(&path).unwrap();
            fill(&db);
            db.close().unwrap();
        }
        let reopened = GrafeoDB::open(&path).unwrap();
        for (stage, db) in [("hot", &hot), ("reopened", &reopened)] {
            for (plan, session) in sessions(db) {
                for (condition, want) in conditions {
                    let found = sql(&session, condition, params);
                    if &found != want {
                        wrong.push(format!(
                            "{name}, index {indexed}, {stage}, {plan}: `{condition}` gave {found:?}, want {want:?}"
                        ));
                    }
                }
            }
        }
        reopened.close().unwrap();
    }
    wrong
}

fn rows(found: &[i64]) -> Result<Vec<i64>, String> {
    Ok(found.to_vec())
}

fn raises() -> Result<Vec<i64>, String> {
    Err(NOT_COMPARABLE.to_string())
}

/// Name, setup, then each condition with its answer.
type Case = (
    &'static str,
    &'static [&'static str],
    &'static [(&'static str, &'static [i64], bool)],
);

/// `(condition, rows, raises)`: `raises` true means 22G04 whatever the rows say.
const CASES: &[Case] = &[
    // One kind of value (numbers of two numeric kinds are one kind to SQL, 4.5.1): rows, no error.
    (
        "numbers_only",
        &[NUMBERS],
        &[
            ("n.k = 5", &[1, 2], false),
            ("5 = n.k", &[1, 2], false),
            ("n.k = 5.0", &[1, 2], false),
            ("n.k = 2.5", &[3], false),
            ("n.k = 7", &[], false),
            ("n.k = 5 AND n.row > 1", &[2], false),
            // A probe of a kind the column does not hold meets its numbers: 22G04, though no row
            // could match.
            ("n.k = 'x'", &[], true),
            ("n.k = '5'", &[], true),
            ("n.k = TRUE", &[], true),
            ("n.k = CAST('2024-01-01' AS DATE)", &[], true),
        ],
    ),
    // A string among the numbers: every equality meets both kinds.
    (
        "a_string_among_numbers",
        &[NUMBERS, "INSERT (:L {row: 5, k: 'x'})"],
        &[
            ("n.k = 5", &[], true),
            ("n.k = 2.5", &[], true),
            ("n.k = 7", &[], true),
            ("n.k = 'x'", &[], true),
            ("n.k = 'y'", &[], true),
        ],
    ),
    (
        "a_date_among_numbers",
        &[NUMBERS, "INSERT (:L {row: 5, k: DATE '2024-01-01'})"],
        &[("n.k = 5", &[], true)],
    ),
    (
        "a_boolean_among_numbers",
        &[NUMBERS, "INSERT (:L {row: 5, k: true})"],
        &[("n.k = 5", &[], true)],
    ),
    // Strings only: the string probe finds its rows, a number probe raises.
    (
        "strings_only",
        &[
            "INSERT (:L {row: 0, k: 'a'}), (:L {row: 1, k: 'b'}), (:L {row: 2, k: 'a'}), (:L {row: 3})",
        ],
        &[
            ("n.k = 'a'", &[0, 2], false),
            ("n.k = 'c'", &[], false),
            ("n.k = 5", &[], true),
        ],
    ),
    // The other kind deleted, or written over, or removed: what is left is one kind, so rows; a
    // structure that still remembers the kind may take the scan, which raises nothing now.
    (
        "the_string_deleted",
        &[
            NUMBERS,
            "INSERT (:L {row: 5, k: 'x'})",
            "MATCH (n:L {row: 5}) DELETE n",
        ],
        &[("n.k = 5", &[1, 2], false), ("n.k = 'x'", &[], true)],
    ),
    (
        "the_string_written_over",
        &[
            NUMBERS,
            "INSERT (:L {row: 5, k: 'x'})",
            "MATCH (n:L {row: 5}) SET n.k = 5",
        ],
        &[("n.k = 5", &[1, 2, 5], false)],
    ),
    (
        "the_string_removed",
        &[
            NUMBERS,
            "INSERT (:L {row: 5, k: 'x'})",
            "MATCH (n:L {row: 5}) REMOVE n.k",
        ],
        &[("n.k = 5", &[1, 2], false)],
    ),
    (
        "a_number_written_over_with_a_string",
        &[NUMBERS, "MATCH (n:L {row: 0}) SET n.k = 'one'"],
        &[("n.k = 5", &[], true), ("n.k = 'one'", &[], true)],
    ),
    // Of two strings one is deleted: the other still meets the probe.
    (
        "one_of_two_strings_deleted",
        &[
            NUMBERS,
            "INSERT (:L {row: 5, k: 'x'}), (:L {row: 6, k: 'y'})",
            "MATCH (n:L {row: 5}) DELETE n",
        ],
        &[("n.k = 5", &[], true)],
    ),
];

/// Every case, without and with an index, hot and reopened, through each plan: one answer.
#[test]
fn an_equality_answers_alike_through_every_structure() {
    let mut wrong = Vec::new();
    let mut checked = 0;
    for (name, setup, conditions) in CASES {
        let conditions: Vec<(&str, Result<Vec<i64>, String>)> = conditions
            .iter()
            .map(|(condition, found, error)| {
                (*condition, if *error { raises() } else { rows(found) })
            })
            .collect();
        checked += conditions.len() * 2 * 2 * 3;
        wrong.extend(every_route_answers(
            name,
            setup,
            &conditions,
            &HashMap::new(),
        ));
    }
    assert!(
        wrong.is_empty(),
        "{} of {checked} wrong:\n{}",
        wrong.len(),
        wrong.join("\n")
    );
}

/// A parameter is a probe like a literal: its kind decides as the literal's would.
#[test]
fn a_parameter_probe_answers_as_its_literal() {
    let mut wrong = Vec::new();
    for (name, setup, value, want) in [
        (
            "numbers_only",
            &[NUMBERS][..],
            Value::Int64(5),
            rows(&[1, 2]),
        ),
        (
            "numbers_only",
            &[NUMBERS][..],
            Value::Float64(2.5),
            rows(&[3]),
        ),
        ("numbers_only", &[NUMBERS][..], Value::from("5"), raises()),
        (
            "a_string_among_numbers",
            &[NUMBERS, "INSERT (:L {row: 5, k: 'x'})"][..],
            Value::Int64(5),
            raises(),
        ),
        // A null probe is unknown against every value: no row and no error.
        ("numbers_only", &[NUMBERS][..], Value::Null, rows(&[])),
        (
            "a_string_among_numbers",
            &[NUMBERS, "INSERT (:L {row: 5, k: 'x'})"][..],
            Value::Null,
            rows(&[]),
        ),
    ] {
        let params = HashMap::from([("p".to_string(), value.clone())]);
        wrong.extend(
            every_route_answers(name, setup, &[("n.k = $p", want)], &params)
                .into_iter()
                .map(|line| format!("{line} (with {value:?})")),
        );
    }
    assert!(
        wrong.is_empty(),
        "{} wrong:\n{}",
        wrong.len(),
        wrong.join("\n")
    );
}

/// A column of more than one row group with the other kind in one group only: the equality raises
/// whichever group holds its match, and finds the match where all is one kind.
#[test]
fn another_kind_in_any_row_group_raises() {
    let numbered = |db: &GrafeoDB, other_at: Option<i64>| {
        for row in 0..GROUP + 2 {
            let k = if Some(row) == other_at {
                Value::from("x")
            } else {
                Value::Int64(row)
            };
            db.create_node_with_props(&["L"], [("row", Value::Int64(row)), ("k", k)])
                .unwrap();
        }
    };
    for (other_at, probe, want) in [
        (None, GROUP + 1, rows(&[GROUP + 1])),
        (None, 5, rows(&[5])),
        // The string in the last group, the match in the first.
        (Some(GROUP + 1), 5, raises()),
        // The string in the first group, the match in the last.
        (Some(3), GROUP, raises()),
    ] {
        for indexed in [false, true] {
            let db = GrafeoDB::new_in_memory();
            numbered(&db, other_at);
            if indexed {
                db.execute("CREATE INDEX on_k FOR (n:L) ON (n.k)").unwrap();
            }
            for (plan, session) in sessions(&db) {
                let condition = format!("n.k = {probe}");
                assert_eq!(
                    sql(&session, &condition, &HashMap::new()),
                    want,
                    "string at {other_at:?}, index {indexed}, {plan}: `{condition}`"
                );
            }
        }
    }
}

/// A transaction that writes a string meets it in its own equality; another session, which does not
/// see the write, finds its rows; after a rollback the rows come back.
#[test]
fn a_transaction_meets_the_kind_it_wrote() {
    for indexed in [false, true] {
        let db = GrafeoDB::new_in_memory();
        run_all(&db, &[NUMBERS]);
        if indexed {
            db.execute("CREATE INDEX on_k FOR (n:L) ON (n.k)").unwrap();
        }
        let mut writer = db.session();
        writer.begin_transaction().unwrap();
        writer.execute("INSERT (:L {row: 5, k: 'x'})").unwrap();
        assert_eq!(
            sql(&writer, "n.k = 5", &HashMap::new()),
            raises(),
            "index {indexed}: its own write"
        );
        for (plan, other) in sessions(&db) {
            assert_eq!(
                sql(&other, "n.k = 5", &HashMap::new()),
                rows(&[1, 2]),
                "index {indexed}, {plan}: another session"
            );
        }
        writer.rollback().unwrap();
        for (plan, session) in sessions(&db) {
            assert_eq!(
                sql(&session, "n.k = 5", &HashMap::new()),
                rows(&[1, 2]),
                "index {indexed}, {plan}: after the rollback"
            );
        }
    }
}

/// GQL and Cypher answer false across kinds, so the same graph gives them rows and no error, and
/// GQL keeps the index.
#[test]
fn gql_and_cypher_keep_their_rows_and_the_index() {
    let db = GrafeoDB::new_in_memory();
    run_all(&db, &[NUMBERS, "INSERT (:L {row: 5, k: 'x'})"]);
    db.execute("CREATE INDEX on_k FOR (n:L) ON (n.k)").unwrap();
    let session = db.session();
    let query = "MATCH (n:L) WHERE n.k = 5 RETURN n.row";
    assert_eq!(answer(session.execute(query)), rows(&[1, 2]));
    assert_eq!(answer(session.execute_cypher(query)), rows(&[1, 2]));
    assert_eq!(
        answer(session.execute("MATCH (n:L) WHERE n.k = 'x' RETURN n.row")),
        rows(&[5])
    );
    let plan = session.execute(&format!("EXPLAIN {query}")).unwrap().rows()[0][0]
        .as_str()
        .unwrap()
        .to_string();
    assert!(plan.contains("[index: k]"), "{plan}");
    // And SQL/PGQ on the same graph raises.
    assert_eq!(sql(&session, "n.k = 5", &HashMap::new()), raises());
}
