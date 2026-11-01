//! An EXISTS that is not a top-level conjunct of its filter (in an OR, under
//! NOT, in a CASE, in a function's argument) runs as a mark join: the
//! subquery runs once over the distinct values of the variables it shares
//! with the rows, and each row gets a true or false column that the filter
//! reads in place of the EXISTS. The rows are those of evaluating the EXISTS
//! per row, which each query here checks twice: against rows written out by
//! hand, and against the same query with each `EXISTS {...}` written as
//! `COUNT {...} > 0` (a COUNT subquery runs per row).
//!
//! - EXISTS in OR, NOT EXISTS in OR, EXISTS in CASE and in a function's
//!   argument, two marks in one predicate, a mark beside a top-level EXISTS
//!   (a semi join) in the same filter;
//! - a shared variable that is null (from an OPTIONAL MATCH);
//! - a guard that reads only the shared variable starts no run for the rows
//!   it rejects (the subquery would divide by zero for them), and a null
//!   guard does not reject;
//! - a guard over another variable runs per row (no MarkJoin);
//! - CALL and OPTIONAL MATCH shapes;
//! - an EXISTS the node's edges answer stays the edge check (no MarkJoin);
//! - at a past epoch and with a transaction's own writes;
//! - `RETURN *` and `RETURN n` show no mark column;
//! - the label branches of an OR over every node still narrow the scan.
//!
//! In GQL and Cypher.
//!
//! ```bash
//! cargo test -p grafeo-engine --all-features --test exists_mark_join
//! ```

#![cfg(all(feature = "gql", feature = "cypher"))]

use grafeo_common::types::Value;
use grafeo_engine::{Config, GrafeoDB, HistoryRetention};

const LANGUAGES: [&str; 2] = ["gql", "cypher"];

/// `(n)` knows an active Person.
const KNOWS_ACTIVE: &str = "EXISTS { MATCH (n)-[:KNOWS]->(m) WHERE m.active }";

/// Eight Persons, `i` 1 to 8: Alix, Gus, Vincent, Jules, Mia, Butch, Django,
/// Shosanna; the odd ones active. `zero` is 1, except 0 on Vincent and Jules
/// and missing on Butch. KNOWS: Alix to Gus and Vincent, Gus to Vincent,
/// Vincent to Jules, Jules to Alix, Mia to herself, Butch to Mia, Django to
/// Gus. So Alix, Gus, Jules, Mia and Butch know an active Person. LIVES_IN:
/// Alix Amsterdam (`pop` 900), Gus and Butch Berlin (3700), Vincent and
/// Jules Paris (2100), Mia Prague (1300), Django Barcelona (1600); Shosanna
/// lives nowhere. And 100 Noise nodes. History is kept, for reads at a past
/// epoch.
fn database() -> GrafeoDB {
    let db = GrafeoDB::with_config(
        Config::in_memory().with_history_retention(HistoryRetention::Forever),
    )
    .unwrap();
    for (i, name) in [
        "Alix", "Gus", "Vincent", "Jules", "Mia", "Butch", "Django", "Shosanna",
    ]
    .iter()
    .enumerate()
    {
        let i = i + 1;
        let zero = match name {
            &"Vincent" | &"Jules" => ", zero: 0",
            &"Butch" => "",
            _ => ", zero: 1",
        };
        db.execute(&format!(
            "INSERT (:Person {{i: {i}, name: '{name}', active: {}{zero}}})",
            i % 2 == 1
        ))
        .unwrap();
    }
    for (name, pop) in [
        ("Amsterdam", 900),
        ("Berlin", 3700),
        ("Paris", 2100),
        ("Prague", 1300),
        ("Barcelona", 1600),
    ] {
        db.execute(&format!("INSERT (:City {{name: '{name}', pop: {pop}}})"))
            .unwrap();
    }
    for (from, to) in [
        ("Alix", "Gus"),
        ("Alix", "Vincent"),
        ("Gus", "Vincent"),
        ("Vincent", "Jules"),
        ("Jules", "Alix"),
        ("Mia", "Mia"),
        ("Butch", "Mia"),
        ("Django", "Gus"),
    ] {
        knows(&db, from, to);
    }
    for (person, city) in [
        ("Alix", "Amsterdam"),
        ("Gus", "Berlin"),
        ("Vincent", "Paris"),
        ("Jules", "Paris"),
        ("Mia", "Prague"),
        ("Butch", "Berlin"),
        ("Django", "Barcelona"),
    ] {
        db.execute(&format!(
            "MATCH (p:Person {{name: '{person}'}}), (c:City {{name: '{city}'}}) \
             INSERT (p)-[:LIVES_IN]->(c)"
        ))
        .unwrap();
    }
    db.execute("UNWIND range(1, 100) AS k INSERT (:Noise {k: k})")
        .unwrap();
    db
}

fn knows(db: &GrafeoDB, from: &str, to: &str) {
    db.execute(&format!(
        "MATCH (a:Person {{name: '{from}'}}), (b:Person {{name: '{to}'}}) INSERT (a)-[:KNOWS]->(b)"
    ))
    .unwrap();
}

fn rows(db: &GrafeoDB, language: &str, query: &str) -> Vec<Vec<Value>> {
    db.execute_language(query, language, None)
        .unwrap_or_else(|error| panic!("{language} `{query}`: {error}"))
        .rows()
        .to_vec()
}

fn ints(values: &[i64]) -> Vec<Vec<Value>> {
    values.iter().map(|v| vec![Value::Int64(*v)]).collect()
}

fn text(db: &GrafeoDB, language: &str, query: &str) -> String {
    match &rows(db, language, query)[0][0] {
        Value::String(text) => text.to_string(),
        other => panic!("{language} `{query}`: {other:?}"),
    }
}

fn explain(db: &GrafeoDB, language: &str, query: &str) -> String {
    text(db, language, &format!("EXPLAIN {query}"))
}

/// `query` with each `EXISTS {...}` written as `COUNT {...} > 0`, which runs
/// per row: the reference for the rows of the mark join.
fn per_row(query: &str) -> String {
    let mut out = String::new();
    let mut rest = query;
    while let Some(start) = rest.find("EXISTS {") {
        out.push_str(&rest[..start]);
        out.push_str("(COUNT {");
        rest = &rest[start + "EXISTS {".len()..];
        let mut depth = 1;
        let end = rest
            .char_indices()
            .find_map(|(index, c)| {
                match c {
                    '{' => depth += 1,
                    '}' => depth -= 1,
                    _ => {}
                }
                (depth == 0).then_some(index)
            })
            .expect("a closed EXISTS");
        out.push_str(&per_row(&rest[..end]));
        out.push_str("} > 0)");
        rest = &rest[end + 1..];
    }
    out.push_str(rest);
    out
}

/// The rows of `query` are `expected`, and those of [`per_row`] of it, in
/// both languages; its plan holds a MarkJoin when `marked`, none when not.
fn check(db: &GrafeoDB, query: &str, expected: &[Vec<Value>], marked: bool) {
    for language in LANGUAGES {
        assert_eq!(rows(db, language, query), expected, "{language} `{query}`");
        let reference = per_row(query);
        assert_ne!(reference, query, "the reference has no EXISTS");
        assert_eq!(
            rows(db, language, &reference),
            expected,
            "{language} per row `{reference}`"
        );
        let plan = explain(db, language, query);
        assert_eq!(
            plan.contains("MarkJoin"),
            marked,
            "{language} `{query}`: {plan}"
        );
    }
}

fn persons(predicate: &str) -> String {
    format!("MATCH (n:Person) WHERE {predicate} RETURN n.i AS i ORDER BY i")
}

#[test]
fn exists_in_or() {
    let db = database();
    let query = persons(&format!("n.i = 8 OR {KNOWS_ACTIVE}"));
    check(&db, &query, &ints(&[1, 2, 4, 5, 6, 8]), true);
}

#[test]
fn not_exists_in_or() {
    let db = database();
    let query = persons(&format!("n.i = 1 OR NOT {KNOWS_ACTIVE}"));
    check(&db, &query, &ints(&[1, 3, 7, 8]), true);
}

#[test]
fn exists_in_case() {
    let db = database();
    let query = persons(&format!(
        "CASE WHEN {KNOWS_ACTIVE} THEN n.i > 3 ELSE n.i = 8 END"
    ));
    check(&db, &query, &ints(&[4, 5, 6, 8]), true);
}

#[test]
fn exists_in_a_function_argument() {
    let db = database();
    let query = persons(&format!("coalesce({KNOWS_ACTIVE}, false) AND n.i > 1"));
    check(&db, &query, &ints(&[2, 4, 5, 6]), true);
}

/// Two marks in one predicate: each EXISTS gets a column of its own.
#[test]
fn two_marks_in_one_predicate() {
    let db = database();
    let query = persons(&format!(
        "{KNOWS_ACTIVE} OR EXISTS {{ MATCH (n)-[:LIVES_IN]->(c) WHERE c.pop > 2000 }}"
    ));
    check(&db, &query, &ints(&[1, 2, 3, 4, 5, 6]), true);
    for language in LANGUAGES {
        let plan = explain(&db, language, &query);
        assert_eq!(plan.matches("MarkJoin").count(), 2, "{language}: {plan}");
    }
}

/// A top-level EXISTS (a semi join) and an EXISTS in an OR, in one filter.
#[test]
fn a_mark_beside_a_semi_join() {
    let db = database();
    let query = persons(&format!(
        "EXISTS {{ MATCH (n)-[:LIVES_IN]->(c) WHERE c.pop > 1000 }} AND (n.i = 1 OR {KNOWS_ACTIVE})"
    ));
    check(&db, &query, &ints(&[2, 4, 5, 6]), true);
    for language in LANGUAGES {
        let plan = explain(&db, language, &query);
        assert!(plan.contains("Join (Semi)"), "{language}: {plan}");
    }
}

/// `c` is null for Shosanna, who lives nowhere: the EXISTS over a null `c`
/// is false, as per row, so only `n.i = 8` keeps her, and NOT EXISTS keeps
/// her.
#[test]
fn a_null_shared_value() {
    let db = database();
    let active_resident = "EXISTS { MATCH (c)<-[:LIVES_IN]-(o) WHERE o.active }";
    let shape = |predicate: &str| {
        format!(
            "MATCH (n:Person) OPTIONAL MATCH (n)-[:LIVES_IN]->(c) WITH n, c \
             WHERE {predicate} RETURN n.i AS i ORDER BY i"
        )
    };
    check(
        &db,
        &shape(&format!("n.i = 8 OR {active_resident}")),
        &ints(&[1, 3, 4, 5, 7, 8]),
        true,
    );
    check(
        &db,
        &shape(&format!("n.i = 1 OR NOT {active_resident}")),
        &ints(&[1, 2, 6, 8]),
        true,
    );
}

/// The subquery divides by `n.zero`, 0 on Vincent and Jules, whose rows the
/// guard `n.zero <> 0` rejects: it starts no run for them. Butch has no
/// `zero`: a null guard does not reject, and `10 / null` matches nothing.
#[test]
fn a_guard_on_the_shared_variable_starts_no_run_for_the_rows_it_rejects() {
    let db = database();
    let query = persons(
        "(n.zero <> 0 AND EXISTS { MATCH (n)-[:KNOWS]->(m) WHERE 10 / n.zero > 0 AND m.active }) \
         OR n.i = 8",
    );
    check(&db, &query, &ints(&[1, 2, 5, 8]), true);
    for language in LANGUAGES {
        let plan = explain(&db, language, &query);
        let delim = plan
            .lines()
            .position(|line| line.contains("DelimScan"))
            .unwrap_or_else(|| panic!("{language}: {plan}"));
        let above = plan.lines().nth(delim - 1).unwrap_or_default();
        assert!(
            above.contains("Filter") && above.contains("zero"),
            "{language}: the guard filters the values: {plan}"
        );
    }
}

/// `NOT (guard AND EXISTS)`: a null guard (Butch, who knows active Mia)
/// gives `NOT (null AND true)`, null, and drops the row; a rejected row
/// (Vincent, Jules) gives `NOT false`, kept.
#[test]
fn a_null_guard_does_not_reject() {
    let db = database();
    let query = persons(&format!("NOT (n.zero <> 0 AND {KNOWS_ACTIVE})"));
    check(&db, &query, &ints(&[3, 4, 7, 8]), true);
}

/// A guard over `c`, which the subquery does not share: per row.
#[test]
fn a_guard_over_another_variable_runs_per_row() {
    let db = database();
    for (city, expected) in [("Paris", vec![8]), ("Berlin", vec![1, 2, 4, 5, 6, 8])] {
        let query = format!(
            "MATCH (n:Person), (c:City {{name: '{city}'}}) \
             WHERE (c.pop > 3000 AND {KNOWS_ACTIVE}) OR n.i = 8 RETURN n.i AS i ORDER BY i"
        );
        check(&db, &query, &ints(&expected), false);
    }
}

/// Cities with more than 3000 people or an active resident past `i` 3
/// (Berlin, Prague, Barcelona), per Person, inside CALL.
#[test]
fn exists_in_or_inside_call() {
    let db = database();
    let query = "MATCH (n:Person) CALL { WITH n MATCH (n)-[:LIVES_IN]->(c) \
                 WHERE c.pop > 3000 OR EXISTS { MATCH (c)<-[:LIVES_IN]-(o) WHERE o.active AND o.i > 3 } \
                 RETURN c.name AS city } RETURN n.i AS i, city ORDER BY i";
    let expected: Vec<Vec<Value>> = [
        (2, "Berlin"),
        (5, "Prague"),
        (6, "Berlin"),
        (7, "Barcelona"),
    ]
    .iter()
    .map(|(i, city)| vec![Value::Int64(*i), Value::from(*city)])
    .collect();
    for language in LANGUAGES {
        assert_eq!(rows(&db, language, query), expected, "{language}");
        assert_eq!(
            rows(&db, language, &per_row(query)),
            expected,
            "{language} per row"
        );
    }
}

/// The WHERE of an OPTIONAL MATCH filters the optional part, never the
/// Persons: every Person stays, with a null city where it does not hold.
#[test]
fn exists_in_the_where_of_an_optional_match_keeps_every_row() {
    let db = database();
    let query = "MATCH (n:Person) OPTIONAL MATCH (n)-[:LIVES_IN]->(c) \
                 WHERE c.pop > 3000 OR EXISTS { MATCH (c)<-[:LIVES_IN]-(o) WHERE o.active AND o.i > 3 } \
                 RETURN n.i AS i, c.name AS city ORDER BY i";
    let expected: Vec<Vec<Value>> = [
        (1, None),
        (2, Some("Berlin")),
        (3, None),
        (4, None),
        (5, Some("Prague")),
        (6, Some("Berlin")),
        (7, Some("Barcelona")),
        (8, None),
    ]
    .iter()
    .map(|(i, city)| vec![Value::Int64(*i), city.map_or(Value::Null, Value::from)])
    .collect();
    for language in LANGUAGES {
        assert_eq!(rows(&db, language, query), expected, "{language}");
        assert_eq!(
            rows(&db, language, &per_row(query)),
            expected,
            "{language} per row"
        );
    }
}

/// The same with a WHERE that reads only the outer `n`: it still filters
/// the optional part, so every Person stays.
#[test]
fn an_optional_match_where_over_the_outer_variable_keeps_every_row() {
    let db = database();
    let query = format!(
        "MATCH (n:Person) OPTIONAL MATCH (n)-[:LIVES_IN]->(c) WHERE n.i = 8 OR {KNOWS_ACTIVE} \
         RETURN n.i AS i, c.name AS city ORDER BY i"
    );
    let expected: Vec<Vec<Value>> = [
        (1, Some("Amsterdam")),
        (2, Some("Berlin")),
        (3, None),
        (4, Some("Paris")),
        (5, Some("Prague")),
        (6, Some("Berlin")),
        (7, None),
        (8, None),
    ]
    .iter()
    .map(|(i, city)| vec![Value::Int64(*i), city.map_or(Value::Null, Value::from)])
    .collect();
    for language in LANGUAGES {
        assert_eq!(rows(&db, language, &query), expected, "{language}");
        assert_eq!(
            rows(&db, language, &per_row(&query)),
            expected,
            "{language} per row"
        );
    }
}

/// An EXISTS the node's own edges answer stays the edge check.
#[test]
fn an_exists_the_edges_answer_stays_the_edge_check() {
    let db = database();
    let query = persons("n.i = 1 OR NOT EXISTS { MATCH (n)-[:KNOWS]->() }");
    check(&db, &query, &ints(&[1, 8]), false);
}

/// At a past epoch, before Django knew Alix, and now.
#[test]
fn at_a_past_epoch() {
    let db = database();
    let query = persons(&format!("n.i = 8 OR {KNOWS_ACTIVE}"));
    let before = db.current_epoch();
    knows(&db, "Django", "Alix");
    assert_eq!(rows(&db, "gql", &query), ints(&[1, 2, 4, 5, 6, 7, 8]));
    assert_eq!(
        db.execute_at_epoch(&query, before).unwrap().rows(),
        ints(&[1, 2, 4, 5, 6, 8]),
        "{query}"
    );
}

/// A transaction sees its own new edge; after its rollback nothing does.
#[test]
fn with_a_transactions_own_writes() {
    let db = database();
    for language in LANGUAGES {
        let query = persons(&format!("n.i = 8 OR {KNOWS_ACTIVE}"));
        let mut session = db.session();
        session.begin_transaction().unwrap();
        session
            .execute(
                "MATCH (a:Person {name: 'Django'}), (b:Person {name: 'Alix'}) \
                 INSERT (a)-[:KNOWS]->(b)",
            )
            .unwrap();
        let inside = session.execute_language(&query, language, None).unwrap();
        assert_eq!(inside.rows(), ints(&[1, 2, 4, 5, 6, 7, 8]), "{language}");
        session.rollback().unwrap();
        assert_eq!(
            rows(&db, language, &query),
            ints(&[1, 2, 4, 5, 6, 8]),
            "{language}"
        );
    }
}

/// `RETURN *` and `RETURN n` give `n` alone, with its own properties.
#[test]
fn the_mark_column_never_shows() {
    let db = database();
    for language in LANGUAGES {
        for query in [
            format!("MATCH (n:Person) WHERE n.i = 8 OR {KNOWS_ACTIVE} RETURN *"),
            format!("MATCH (n:Person) WHERE n.i = 8 OR {KNOWS_ACTIVE} RETURN n"),
        ] {
            assert!(
                explain(&db, language, &query).contains("MarkJoin"),
                "{language} `{query}`"
            );
            let result = db.execute_language(&query, language, None).unwrap();
            assert_eq!(result.columns, ["n"], "{language} `{query}`");
            assert_eq!(result.rows().len(), 6, "{language} `{query}`");
            for row in result.rows() {
                let shown = format!("{:?}", row[0]);
                assert!(!shown.contains("__mark"), "{language} `{query}`: {shown}");
            }
        }
    }
}

/// `(n:Person AND EXISTS) OR (n:City AND ...)` over every node reads the 8
/// Persons and 5 Cities, not the 100 Noise nodes too.
#[test]
fn the_label_branches_still_narrow_the_scan() {
    let db = database();
    let query = format!(
        "MATCH (n) WHERE (n:Person AND {KNOWS_ACTIVE}) OR (n:City AND n.pop > 2000) \
         RETURN n.name AS name ORDER BY name"
    );
    let expected: Vec<Vec<Value>> = ["Alix", "Berlin", "Butch", "Gus", "Jules", "Mia", "Paris"]
        .iter()
        .map(|name| vec![Value::from(*name)])
        .collect();
    check(&db, &query, &expected, true);
    for language in LANGUAGES {
        let profile = text(&db, language, &format!("PROFILE {query}"));
        let scanned: u64 = profile
            .lines()
            .filter(|line| line.trim_start().starts_with("Scan (n"))
            .filter_map(|line| {
                line.split("rows=")
                    .nth(1)?
                    .split_whitespace()
                    .next()?
                    .parse::<u64>()
                    .ok()
            })
            .sum();
        assert!(
            (1..=13).contains(&scanned),
            "{language}: the scan reads {scanned} rows: {profile}"
        );
    }
}
