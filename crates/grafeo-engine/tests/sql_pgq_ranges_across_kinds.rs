//! A SQL/PGQ range (`<`, `<=`, `>`, `>=`, and their `AND`, the `BETWEEN` shape) answers alike, its
//! 22G04 included, whatever structure reads it: a query's answer never depends on the plan
//! (GA04 step b, the ruling of 2026-11-06). SQL raises "values not comparable" when a comparison
//! meets a value of another kind (ISO/IEC 9075-2 8.2), so a range scan or a zone map may read the
//! column only when every value it holds is of the bound's kind or a null; otherwise the plan
//! meets the values and raises. A column of one kind is read by the range scan, which the plan
//! shows, and a mixed column is not. Each case runs without an index and with one, hot and
//! reopened, each with the planner's choice and with the plain scan. GQL and Cypher keep their
//! plans.
//!
//! ```bash
//! cargo test -p grafeo-engine --all-features --test main sql_pgq_ranges_across_kinds::
//! ```

#![cfg(all(feature = "gql", feature = "cypher", feature = "sql-pgq"))]

use grafeo_common::utils::gqlstatus::GqlStatus;
use grafeo_engine::GrafeoDB;
use grafeo_engine::database::QueryResult;
use grafeo_engine::query::optimizer::{PlannerRule, PlannerRules};
use grafeo_engine::session::Session;

const NOT_COMPARABLE: &str = "22G04";

/// Numbers of two numeric kinds (one kind to SQL, 4.5.1), and a node without `k`.
const NUMBERS: &str = "INSERT (:L {row: 0, k: 1}), (:L {row: 1, k: 5}), (:L {row: 2, k: 5}), \
     (:L {row: 3, k: 2.5}), (:L {row: 4})";

/// Strings only.
const STRINGS: &str = "INSERT (:L {row: 0, k: 'a'}), (:L {row: 1, k: 'b'}), \
     (:L {row: 2, k: 'a'}), (:L {row: 3})";

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

fn sql(session: &Session, condition: &str) -> Result<Vec<i64>, String> {
    answer(session.execute_sql(&sql_query(condition)))
}

/// The physical plan `EXPLAIN ANALYZE` runs the query for, or why there is none.
fn plan(session: &Session, condition: &str) -> Result<String, String> {
    let query = sql_query(condition);
    let result = session
        .execute_sql(&format!("EXPLAIN ANALYZE {query}"))
        .map_err(|error| error.to_string())?;
    if result.columns != ["profile"] {
        return Err(format!(
            "EXPLAIN ANALYZE ran the statement (columns {:?})",
            result.columns
        ));
    }
    Ok(result.rows()[0][0].as_str().unwrap_or_default().to_string())
}

fn session_with(db: &GrafeoDB, rules: PlannerRules) -> Session {
    let mut session = db.session();
    session.set_planner_rules(rules);
    session
}

/// The two plans a database offers: the planner's choice and the plain scan.
fn sessions(db: &GrafeoDB) -> [(&'static str, Session); 2] {
    [
        ("planner", db.session()),
        ("plain", session_with(db, PlannerRules::plain())),
    ]
}

fn run_all(db: &GrafeoDB, statements: &[&str]) {
    for statement in statements {
        db.execute(statement)
            .unwrap_or_else(|error| panic!("{statement}: {error}"));
    }
}

/// `setup` on a database without and with an index on `L.k`, hot and reopened: every route
/// gives `want` for each condition, and the planner's plan reads the range with a range scan
/// when `range_scan` holds.
fn every_route_answers(name: &str, setup: &[&str], conditions: &[Condition]) -> Vec<String> {
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
        let path = dir.path().join("ranges.grafeo");
        {
            let db = GrafeoDB::open(&path).unwrap();
            fill(&db);
            db.close().unwrap();
        }
        let reopened = GrafeoDB::open(&path).unwrap();
        for (stage, db) in [("hot", &hot), ("reopened", &reopened)] {
            for (plan_name, session) in sessions(db) {
                for (condition, want_rows, want_raises, range_scan) in conditions {
                    let want = if *want_raises {
                        raises()
                    } else {
                        rows(want_rows)
                    };
                    let found = sql(&session, condition);
                    if found != want {
                        wrong.push(format!(
                            "{name}, index {indexed}, {stage}, {plan_name}: `{condition}` gave {found:?}, want {want:?}"
                        ));
                    }
                    if plan_name == "planner" && !*want_raises {
                        let explained = plan(&session, condition);
                        match &explained {
                            Ok(text) if *range_scan == text.contains("RangeScan") => {}
                            Ok(text) => wrong.push(format!(
                                "{name}, index {indexed}, {stage}: `{condition}` planned {text:?}, range scan {}",
                                if *range_scan { "expected" } else { "not expected" }
                            )),
                            Err(error) => wrong.push(format!(
                                "{name}, index {indexed}, {stage}: `{condition}` EXPLAIN: {error}"
                            )),
                        }
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

/// (condition, rows, raises, whether the planner's plan reads a range scan).
type Condition = (&'static str, &'static [i64], bool, bool);

/// Name, setup, conditions.
type Case = (&'static str, &'static [&'static str], &'static [Condition]);

const CASES: &[Case] = &[
    // One kind of value: the range scan reads the column, and answers.
    (
        "numbers_only",
        &[NUMBERS],
        &[
            ("n.k >= 5 AND n.k < 7", &[1, 2], false, true),
            ("n.k > 2.5", &[1, 2], false, true),
            ("n.k >= 1", &[0, 1, 2, 3], false, true),
            ("n.k < 2", &[0], false, true),
            // No row in range: the zone map proves it, the plan reads nothing.
            ("n.k > 100", &[], false, false),
            // A bound of a kind the column does not hold meets its numbers: 22G04, though no
            // row could match.
            ("n.k >= 'x'", &[], true, false),
            ("n.k < TRUE", &[], true, false),
        ],
    ),
    // A string among the numbers: every range meets both kinds, so 22G04 under every plan, and
    // no range scan reads the column.
    (
        "a_string_among_numbers",
        &[NUMBERS, "INSERT (:L {row: 5, k: 'x'})"],
        &[
            ("n.k >= 5 AND n.k < 7", &[], true, false),
            ("n.k > 100", &[], true, false),
            ("n.k >= 'x'", &[], true, false),
        ],
    ),
    // Strings only: the string range reads the column, a number bound raises.
    (
        "strings_only",
        &[STRINGS],
        &[
            ("n.k >= 'a' AND n.k < 'c'", &[0, 1, 2], false, true),
            ("n.k > 'z'", &[], false, false),
            ("n.k >= 5", &[], true, false),
        ],
    ),
    // A null bound never raises and matches nothing.
    (
        "null_bound",
        &[NUMBERS],
        &[("n.k >= null", &[], false, true)],
    ),
];

#[test]
fn every_range_answers_as_the_plain_plan_and_reads_one_kind_by_range_scan() {
    let mut wrong = Vec::new();
    for (name, setup, conditions) in CASES {
        wrong.extend(every_route_answers(name, setup, conditions));
    }
    assert!(wrong.is_empty(), "{}", wrong.join("\n"));
}

/// A range over an edge property, whose column is checked the same way.
#[test]
fn edge_ranges_across_kinds_answer_alike() {
    let mut wrong = Vec::new();
    let setup = &[
        "INSERT (:L {row: 0})-[:R {k: 1}]->(:L {row: 1})",
        "INSERT (:L {row: 2})-[:R {k: 5}]->(:L {row: 3})",
        "INSERT (:L {row: 4})-[:R {k: 'x'}]->(:L {row: 5})",
    ];
    let db = GrafeoDB::new_in_memory();
    run_all(&db, setup);
    for (plan_name, session) in sessions(&db) {
        let query = "SELECT * FROM GRAPH_TABLE (MATCH (:L)-[r:R]->(:L) WHERE r.k >= 2 AND r.k < 9 \
                     COLUMNS (r.k AS k))";
        let found = answer(session.execute_sql(query));
        // The plain scan compares the string edge and raises; so does every plan.
        if found != raises() {
            wrong.push(format!("{plan_name}: gave {found:?}, want 22G04"));
        }
    }
    assert!(wrong.is_empty(), "{}", wrong.join("\n"));
}

/// The rule-off control: with `IndexRange` off no plan reads a range scan, and the answers are
/// the same.
#[test]
fn the_rule_off_control_keeps_the_answers() {
    let db = GrafeoDB::new_in_memory();
    run_all(&db, &[NUMBERS]);
    let mut off = db.session();
    off.set_planner_rules(PlannerRules::all().with(PlannerRule::IndexRange, false));
    let plain = session_with(&db, PlannerRules::plain());
    for condition in ["n.k >= 5 AND n.k < 7", "n.k > 100"] {
        let a = sql(&off, condition);
        let b = sql(&plain, condition);
        assert_eq!(a, b, "`{condition}`");
        let explained = plan(&off, condition).unwrap();
        assert!(
            !explained.contains("RangeScan"),
            "`{condition}`: {explained}"
        );
    }
}

/// GQL and Cypher keep their plans on the same column: the range scan fires for them as before.
#[test]
fn gql_and_cypher_keep_their_range_scans() {
    let db = GrafeoDB::new_in_memory();
    run_all(&db, &[NUMBERS, "INSERT (:L {row: 5, k: 'x'})"]);
    let session = db.session();
    for language in ["GQL", "Cypher"] {
        let text = "MATCH (n:L) WHERE n.k >= 5 AND n.k < 7 RETURN n.row AS row";
        let result = if language == "GQL" {
            session.execute(text)
        } else {
            session.execute_cypher(text)
        }
        .unwrap();
        let mut rows: Vec<i64> = result
            .rows()
            .iter()
            .map(|row| row[0].as_int64().unwrap())
            .collect();
        rows.sort_unstable();
        // GQL orders the string before the numbers, so 'x' is out of [5, 7): no raise, two rows.
        // Cypher answers null for the cross-kind comparison: the row is filtered, two rows.
        assert_eq!(rows, vec![1, 2], "{language}");
        let explained = if language == "GQL" {
            session.execute(&format!("PROFILE {text}"))
        } else {
            session.execute_cypher(&format!("PROFILE {text}"))
        }
        .unwrap();
        let profile = explained.rows()[0][0].as_str().unwrap_or_default();
        assert!(profile.contains("RangeScan"), "{language}: {profile}");
    }
}
