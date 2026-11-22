//! A comparison of two values of two kinds raises in SQL/PGQ (22G04, ISO/IEC 9075-2 8.2 Syntax
//! Rule 3) and does not in GQL or Cypher. A rewrite that moves a check (predicate push-down, the
//! checks lifted over an intersection, the early checks of the join search) may move only what
//! cannot raise under the statement's rule, so under SQL/PGQ a comparison counts as raising unless
//! both sides' kinds are known at plan time (two literals of one kind).
//!
//! Errors may disappear, never appear: with a rewrite on and off (or optimized against written),
//! a statement gives the same rows, or rows where the other raises; never an error where the
//! other gives rows. GQL and Cypher keep their moves.
//!
//! ```bash
//! cargo test -p grafeo-engine --all-features --test main sql_pgq_comparisons_in_rewrites::
//! ```

#![cfg(all(feature = "gql", feature = "cypher", feature = "sql-pgq"))]
#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    reason = "test graphs of a few hundred nodes"
)]

use std::collections::HashMap;
use std::time::Duration;

use grafeo_common::types::{NodeId, Value};
use grafeo_common::utils::gqlstatus::GqlStatus;
use grafeo_engine::database::QueryResult;
use grafeo_engine::query::optimizer::{PlannerRule, PlannerRules};
use grafeo_engine::query::{QueryLanguage, QueryProcessor};
use grafeo_engine::session::Session;
use grafeo_engine::{Config, GrafeoDB};

/// The rows are what these tests read, not the query timeout.
const SLOW_BUILD: Duration = Duration::from_secs(600);

const NOT_COMPARABLE: &str = "22G04";

const MARK: &str = "[IntersectExpand into:";

/// A square from `a` over its few friends `b` and `d`, closed over the many people they follow
/// (`early_checks.rs`'s, anchored on `n`, which the odd person lacks).
const SQUARE: &str = "(a:Person {n: 3})-[:KNOWS]-(b:Person)-[:FOLLOWS]-(c:Person)-[:FOLLOWS]-(d:Person)-[:KNOWS]-(a)";

/// Rows, sorted, or the error's GQLSTATUS.
type Answer = Result<Vec<String>, String>;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Language {
    Gql,
    Cypher,
    Sql,
}

const LANGUAGES: [Language; 3] = [Language::Gql, Language::Cypher, Language::Sql];
const NOT_SQL: [Language; 2] = [Language::Gql, Language::Cypher];

/// `pattern` with `condition`, returning `columns`, in `language`.
fn statement(language: Language, pattern: &str, condition: &str, columns: &[&str]) -> String {
    match language {
        Language::Gql | Language::Cypher => {
            let items: Vec<String> = columns.iter().map(|c| format!("{c}.id")).collect();
            format!(
                "MATCH {pattern} WHERE {condition} RETURN {}",
                items.join(", ")
            )
        }
        Language::Sql => {
            let items: Vec<String> = columns.iter().map(|c| format!("{c}.id AS {c}")).collect();
            format!(
                "SELECT * FROM GRAPH_TABLE (MATCH {pattern} WHERE {condition} COLUMNS ({}))",
                items.join(", ")
            )
        }
    }
}

fn square(language: Language, condition: &str) -> String {
    statement(language, SQUARE, condition, &["b", "c", "d"])
}

fn answer(result: grafeo_common::utils::error::Result<QueryResult>) -> Answer {
    match result {
        Ok(result) => {
            let mut rows: Vec<String> =
                result.rows().iter().map(|row| format!("{row:?}")).collect();
            rows.sort();
            Ok(rows)
        }
        Err(error) => Err(GqlStatus::from(&error).as_str().to_string()),
    }
}

fn run(session: &Session, language: Language, query: &str) -> Answer {
    answer(match language {
        Language::Gql => session.execute(query),
        Language::Cypher => session.execute_cypher(query),
        Language::Sql => session.execute_sql(query),
    })
}

fn explain(session: &Session, language: Language, query: &str) -> String {
    let result = match language {
        Language::Gql => session.execute(&format!("EXPLAIN {query}")),
        Language::Cypher => session.execute_cypher(&format!("EXPLAIN {query}")),
        Language::Sql => session.execute_sql(&format!("EXPLAIN {query}")),
    }
    .unwrap_or_else(|error| panic!("EXPLAIN {query}: {error}"));
    plan_text(&result)
}

fn plan_text(result: &QueryResult) -> String {
    result
        .rows()
        .iter()
        .map(|row| match &row[0] {
            Value::String(line) => line.to_string(),
            other => format!("{other:?}"),
        })
        .collect::<Vec<_>>()
        .join("\n")
}

fn with_rules(db: &GrafeoDB, rules: PlannerRules) -> Session {
    let mut session = db.session();
    session.set_planner_rules(rules);
    session
}

fn without(db: &GrafeoDB, rule: PlannerRule) -> Session {
    with_rules(db, PlannerRules::all().with(rule, false))
}

/// The line of `plan` that holds every one of `words`, the first from the top.
fn line_of(plan: &str, words: &[&str]) -> Option<usize> {
    plan.lines()
        .position(|line| words.iter().all(|word| line.contains(word)))
}

/// Whether the filter holding `check` runs below an intersection in `plan`.
fn below_an_intersection(plan: &str, check: &str) -> bool {
    match (line_of(plan, &[MARK]), line_of(plan, &["Filter", check])) {
        (Some(mark), Some(line)) => line > mark,
        _ => false,
    }
}

/// Errors may disappear, never appear: `on` gives the rows of `off`, or rows where `off` raises,
/// or the error `off` raises; never an error where `off` gives rows.
fn no_error_appears(on: &Answer, off: &Answer) -> bool {
    match (on, off) {
        (Ok(on), Ok(off)) => on == off,
        (Ok(_), Err(_)) => true,
        (Err(on), Err(off)) => on == off,
        (Err(_), Ok(_)) => false,
    }
}

/// `size` people (`id` and `n` their number, a name and a vector), each who knows `knows`
/// others and follows `follows` others at random, and the odd person: a string `id`, an integer
/// `name` and a string `v`, no `n`, who knows person 3 and follows no one, so lies on no square.
fn people(size: u64, knows: u64, follows: u64) -> GrafeoDB {
    let db = GrafeoDB::with_config(Config::in_memory().with_query_timeout(SLOW_BUILD)).unwrap();
    let ids: Vec<NodeId> = (0..size)
        .map(|id| {
            db.create_node_with_props(
                &["Person"],
                [
                    ("id", Value::Int64(id as i64)),
                    ("n", Value::Int64(id as i64)),
                    ("name", Value::from(format!("p{id}").as_str())),
                    ("v", Value::Vector(vec![id as f32 + 1.0, 1.0].into())),
                ],
            )
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
    let odd = db
        .create_node_with_props(
            &["Person"],
            [
                ("id", Value::from("x")),
                ("name", Value::Int64(7)),
                ("v", Value::from("x")),
            ],
        )
        .unwrap();
    db.create_edge(odd, ids[3], "KNOWS").unwrap();
    db
}

fn squares() -> GrafeoDB {
    people(200, 4, 12)
}

/// The rows GQL gives the square: the odd person is on none of them, so every comparison the
/// written statement makes is of two integers, in every language.
fn written_rows(db: &GrafeoDB, condition: &str) -> Vec<String> {
    let rows = run(
        &without(db, PlannerRule::EarlyChecks),
        Language::Gql,
        &square(Language::Gql, condition),
    )
    .unwrap();
    assert!(!rows.is_empty(), "the test needs rows: {condition}");
    rows
}

// 1. The early checks of the join search.

/// The square's `b.id < d.id` moves under the intersection, where it meets the
/// odd person, a friend of `a` on no square, whom the written statement never compares. SQL/PGQ
/// gives the written rows with the rule on and off: no 22G04 appears.
#[test]
fn the_graph_table_square_gives_its_rows_with_the_early_checks_on_and_off() {
    let db = squares();
    let condition = "b.id < d.id AND c.n <> 3";
    let want = Ok(written_rows(&db, condition));
    let query = square(Language::Sql, condition);
    let on = run(&db.session(), Language::Sql, &query);
    let off = run(
        &without(&db, PlannerRule::EarlyChecks),
        Language::Sql,
        &query,
    );
    assert_eq!(off, want, "early checks off");
    assert_eq!(on, want, "early checks on");
}

// 2. Predicate push-down and the checks lifted over an intersection.

/// Alix (1), Gus (2) and Jules (7) know each other's way; Mia ('m') knows Jules and Vincent
/// ('v') knows no one. Alix likes Gus.
fn few() -> GrafeoDB {
    let db = GrafeoDB::new_in_memory();
    db.execute(
        "INSERT (alix:Person {id: 1}), (gus:Person {id: 2}), (jules:Person {id: 7}), \
                (mia:Person {id: 'm'}), (:Person {id: 'v'}), \
                (alix)-[:KNOWS]->(gus), (gus)-[:KNOWS]->(alix), (alix)-[:KNOWS]->(jules), \
                (mia)-[:KNOWS]->(jules), (alix)-[:LIKES]->(gus)",
    )
    .unwrap();
    db
}

/// The written rows: the pairs whose ids are `pairs`.
fn pairs(pairs: &[(i64, i64)]) -> Answer {
    let mut rows: Vec<String> = pairs
        .iter()
        .map(|&(a, b)| format!("{:?}", [Value::Int64(a), Value::Int64(b)]))
        .collect();
    rows.sort();
    Ok(rows)
}

/// A comparison of the near end alone may move below the expand onto Vincent and Mia, who like
/// no one: push-down may put a filter that can raise below an expand when no check is written
/// before it (decided 2026-11-02, "written checks only"), so the statement either answers Alix's
/// row or raises the comparison's error (22G04). It never answers other rows; neither outcome is
/// asserted over the other.
#[test]
fn a_comparison_of_the_near_end_alone_answers_its_row_or_raises() {
    let db = few();
    let query = statement(
        Language::Sql,
        "(a:Person)-[:LIKES]->(b:Person)",
        "a.id < 5",
        &["a", "b"],
    );
    let answer = run(&db.session(), Language::Sql, &query);
    assert!(
        answer == pairs(&[(1, 2)]) || answer == Err("22G04".to_string()),
        "Alix's row or 22G04: {answer:?}"
    );
}

/// A check between the two hops a triangle closes over is lifted above the intersection, where
/// it sees only the rows that close: errors may disappear, never appear. Vincent, whom Gus
/// knows, closes no triangle: with the intersection off, the push-down leaves `c.id < 5` between
/// the hops, where SQL/PGQ meets him and raises.
#[test]
fn the_checks_lifted_over_an_intersection_raise_no_new_error() {
    let db = GrafeoDB::new_in_memory();
    db.execute(
        "INSERT (alix:Person {id: 1, n: 1}), (gus:Person {id: 2, n: 2}), \
                (jules:Person {id: 4, n: 4}), (vincent:Person {id: 'v'}), \
                (alix)-[:KNOWS]->(gus), (gus)-[:KNOWS]->(jules), (jules)-[:KNOWS]->(alix), \
                (gus)-[:KNOWS]->(vincent)",
    )
    .unwrap();
    let pattern = "(a:Person {n: 1})-[:KNOWS]-(b:Person)-[:KNOWS]-(c:Person)-[:KNOWS]-(a)";
    for language in LANGUAGES {
        let query = statement(language, pattern, "c.id < 5", &["b", "c"]);
        let on = run(&db.session(), language, &query);
        let off = run(
            &without(&db, PlannerRule::IntersectExpand),
            language,
            &query,
        );
        assert!(
            no_error_appears(&on, &off),
            "{language:?}: on {on:?}, off {off:?}"
        );
    }
}

// 3. GQL and Cypher keep the move.

/// With the odd person in the graph, the GQL and Cypher squares still check `b` against `d`
/// below the intersection, and give the rows of the rule off.
#[test]
fn the_gql_and_cypher_squares_still_check_below_the_intersection() {
    let db = squares();
    let condition = "b.id < d.id AND c.n <> 3";
    for language in [Language::Gql, Language::Cypher] {
        let query = square(language, condition);
        let on = explain(&db.session(), language, &query);
        assert!(
            below_an_intersection(&on, "b.id Lt d.id"),
            "{language:?}:\n{on}"
        );
        let off = explain(&without(&db, PlannerRule::EarlyChecks), language, &query);
        assert!(
            !below_an_intersection(&off, "b.id Lt d.id"),
            "{language:?}:\n{off}"
        );
        let rows = run(&db.session(), language, &query);
        assert!(rows.as_ref().is_ok_and(|rows| !rows.is_empty()), "{rows:?}");
        assert_eq!(
            rows,
            run(&without(&db, PlannerRule::EarlyChecks), language, &query),
            "{language:?}"
        );
    }
}

// 4. Every operator `cannot_raise` lists, in every language.

/// The checks over `b` and `d` the early checks move under the intersection, where the odd
/// person meets each with a value of another kind (a string `id`, an integer `name`, a string
/// `v`). SQL/PGQ's `WHERE` has no label test and no `STARTS WITH`, `ENDS WITH` or `CONTAINS`
/// (its parser refuses them, 42001). A check whose rule-off run raises tells nothing, so every
/// rule-off run gives rows.
const CHECKS: &[(&str, &str, &[Language])] = &[
    ("=", "b.id = d.id", &LANGUAGES),
    ("<>", "b.id <> d.id", &LANGUAGES),
    ("<", "b.id < d.id", &LANGUAGES),
    ("<=", "b.id <= d.id", &LANGUAGES),
    (">", "b.id > d.id", &LANGUAGES),
    (">=", "b.id >= d.id", &LANGUAGES),
    ("STARTS WITH", "b.name STARTS WITH d.name", &NOT_SQL),
    ("ENDS WITH", "b.name ENDS WITH d.name", &NOT_SQL),
    ("CONTAINS", "b.name CONTAINS d.name", &NOT_SQL),
    ("IS NULL", "NOT (b.name IS NULL OR d.v IS NULL)", &LANGUAGES),
    (
        "IS NOT NULL",
        "b.name IS NOT NULL AND d.v IS NOT NULL",
        &LANGUAGES,
    ),
    ("hasLabel", "(b:Person OR d:Robot)", &NOT_SQL),
    (
        "cosine_similarity",
        "cosine_similarity(b.v, d.v) > 0.5",
        &LANGUAGES,
    ),
    (
        "cosine_distance",
        "cosine_distance(b.v, d.v) < 0.5",
        &LANGUAGES,
    ),
    (
        "euclidean_distance",
        "euclidean_distance(b.v, d.v) > 1.0",
        &LANGUAGES,
    ),
    ("dot_product", "dot_product(b.v, d.v) > 1.0", &LANGUAGES),
    (
        "manhattan_distance",
        "manhattan_distance(b.v, d.v) > 1.0",
        &LANGUAGES,
    ),
];

/// Each check, in each language, with the early checks on and off: errors may disappear, never
/// appear. Under SQL/PGQ the six comparisons raise on the odd person, so they must not move.
#[test]
fn every_listed_operator_raises_no_new_error_in_any_language() {
    let db = squares();
    let mut wrong = Vec::new();
    for (operator, check, languages) in CHECKS {
        for language in languages.iter().copied() {
            let query = square(language, check);
            let on = run(&db.session(), language, &query);
            let off = run(&without(&db, PlannerRule::EarlyChecks), language, &query);
            assert!(off.is_ok(), "{operator} in {language:?}, rule off: {off:?}");
            if !no_error_appears(&on, &off) {
                wrong.push(format!(
                    "{operator} in {language:?}: on {on:?}, off {off:?}"
                ));
            }
        }
    }
    assert!(wrong.is_empty(), "{}", wrong.join("\n"));
}

// 5. Every path that optimizes a SQL/PGQ statement plans it under SQL/PGQ's rule.

const PATH_CHECK: &str = "b.id < d.id";

fn sql_square() -> String {
    square(Language::Sql, PATH_CHECK)
}

/// `Session::execute_sql`, twice: the optimized plan and the cached one.
#[test]
fn execute_plans_the_square_under_the_sql_rule() {
    let db = squares();
    let want = Ok(written_rows(&db, PATH_CHECK));
    let session = db.session();
    for call in ["first", "cached"] {
        assert_eq!(run(&session, Language::Sql, &sql_square()), want, "{call}");
    }
    assert_eq!(
        answer(db.execute_language(&sql_square(), "sql", None)),
        want,
        "execute_language"
    );
}

/// `Session::execute_sql_with_params`: the prepared plan, on its first and later calls.
#[test]
fn a_prepared_statement_plans_the_square_under_the_sql_rule() {
    let db = squares();
    let want = Ok(written_rows(&db, PATH_CHECK));
    let query = sql_square().replace("{n: 3}", "{n: $anchor}");
    let session = db.session();
    for call in ["first", "prepared"] {
        let params = HashMap::from([("anchor".to_string(), Value::Int64(3))]);
        assert_eq!(
            answer(session.execute_sql_with_params(&query, params)),
            want,
            "{call}"
        );
    }
}

/// `EXPLAIN SELECT ...`: the comparison is not below the intersection.
#[test]
fn explain_plans_the_square_under_the_sql_rule() {
    let db = squares();
    let plan = explain(&db.session(), Language::Sql, &sql_square());
    assert!(!below_an_intersection(&plan, "b.id Lt d.id"), "{plan}");
}

/// `QueryProcessor::process` with `QueryLanguage::SqlPgq`.
#[test]
fn a_query_processor_plans_the_square_under_the_sql_rule() {
    let db = squares();
    let want = Ok(written_rows(&db, PATH_CHECK));
    let processor = QueryProcessor::for_lpg(db.store().unwrap());
    assert_eq!(
        answer(processor.process(&sql_square(), QueryLanguage::SqlPgq, None)),
        want
    );
}

/// The planner's key-column read (an equality on a key no index serves) answers as the scan's
/// filter does: the odd person's string `id` raises.
#[test]
fn a_key_column_read_raises_as_the_scan_does() {
    let db = squares();
    let query = "SELECT * FROM GRAPH_TABLE (MATCH (p:Person) WHERE p.id = 3 COLUMNS (p.n AS n))";
    for (route, session) in [
        ("planner", db.session()),
        ("plain", with_rules(&db, PlannerRules::plain())),
    ] {
        assert_eq!(
            answer(session.execute_sql(query)),
            Err(NOT_COMPARABLE.to_string()),
            "{route}"
        );
    }
}

/// The planner's per-row key seek through an index answers as the scan's filter does: the odd
/// person's string `id`, in the index, raises. (With the seek off, the hash join on `p.id = b.n`
/// gives rows: an error that disappears, which the rule allows.)
#[test]
fn a_key_seek_raises_as_the_scan_does() {
    let db = squares();
    db.execute("CREATE INDEX person_id FOR (p:Person) ON (p.id)")
        .unwrap();
    let query = "SELECT * FROM GRAPH_TABLE (MATCH (a:Person {n: 3})-[:KNOWS]-(b:Person), (p:Person) \
         WHERE p.id = b.n COLUMNS (p.n AS n))";
    let seek = |session: &Session| explain(session, Language::Sql, query).contains("[index: id]");
    assert!(seek(&db.session()), "the route under test");
    assert!(!seek(&without(&db, PlannerRule::KeySeek)), "the route off");
    for (route, session) in [
        ("planner", db.session()),
        ("plain", with_rules(&db, PlannerRules::plain())),
    ] {
        assert_eq!(
            answer(session.execute_sql(query)),
            Err(NOT_COMPARABLE.to_string()),
            "{route}"
        );
    }
}
