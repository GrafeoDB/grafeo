//! A `WHERE` conjunct moves down the plan only into an input that binds every
//! variable it reads, and not below anything that changes the input's rows
//! or writes once per row (#455): one side of a join, the input of a `CALL`
//! subquery, the input of a later `MATCH`, an earlier `MATCH`'s filter. The
//! rows are those of the filter where it is written, in GQL and Cypher.
//!
//! ```bash
//! cargo test -p grafeo-engine --all-features --test filter_pushdown_sides
//! ```

#![cfg(all(feature = "lpg", feature = "gql", feature = "cypher"))]

use grafeo_common::types::Value;
use grafeo_engine::{GrafeoDB, Session};

#[derive(Debug, Clone, Copy)]
enum Language {
    Gql,
    Cypher,
}

const LANGUAGES: [Language; 2] = [Language::Gql, Language::Cypher];

/// `A` nodes Alix (k 1), Gus (2) and Vincent (3); `B` nodes Jules (1), Mia (2)
/// and Butch (4); `C` nodes Django (2) and Hans (3); `R` edges Alix to Django,
/// Gus to Hans and Gus to Django.
fn graph() -> GrafeoDB {
    let db = GrafeoDB::new_in_memory();
    db.execute(
        "INSERT (:A {name: 'Alix', k: 1}), (:A {name: 'Gus', k: 2}), (:A {name: 'Vincent', k: 3}), \
                (:B {name: 'Jules', k: 1}), (:B {name: 'Mia', k: 2}), (:B {name: 'Butch', k: 4}), \
                (:C {name: 'Django', k: 2}), (:C {name: 'Hans', k: 3})",
    )
    .unwrap();
    db.execute(
        "UNWIND [['Alix', 'Django'], ['Gus', 'Hans'], ['Gus', 'Django']] AS pair \
         MATCH (a:A {name: pair[0]}), (c:C {name: pair[1]}) INSERT (a)-[:R]->(c)",
    )
    .unwrap();
    db
}

fn run(session: &Session, language: Language, query: &str) -> Vec<Vec<Value>> {
    let result = match language {
        Language::Gql => session.execute(query),
        Language::Cypher => session.execute_cypher(query),
    };
    result
        .unwrap_or_else(|error| panic!("{language:?} `{query}` failed: {error}"))
        .rows()
        .to_vec()
}

/// The rows of `query` in a stable order (the query does not order them).
fn sorted(session: &Session, language: Language, query: &str) -> Vec<Vec<Value>> {
    let mut rows = run(session, language, query);
    rows.sort_by_key(|row| format!("{row:?}"));
    rows
}

/// The text of the plan that `EXPLAIN` returns for `query`.
fn plan(session: &Session, language: Language, query: &str) -> String {
    run(session, language, &format!("EXPLAIN {query}"))
        .iter()
        .map(|row| match &row[0] {
            Value::String(text) => text.to_string(),
            other => format!("{other:?}"),
        })
        .collect::<Vec<_>>()
        .join("\n")
}

/// A plan line without its indentation and its pushdown hint (`[...]`).
fn operator(line: &str) -> &str {
    let line = line.trim();
    match line.rfind(" [") {
        Some(hint) if line.ends_with(']') => &line[..hint],
        _ => line,
    }
}

/// Whether the plan has `upper` with `lower` as its input, hints aside.
fn sits_on(plan: &str, upper: &str, lower: &str) -> bool {
    let lines: Vec<&str> = plan.lines().collect();
    let indent = |line: &str| line.len() - line.trim_start().len();
    lines.windows(2).any(|pair| {
        operator(pair[0]) == upper
            && operator(pair[1]) == lower
            && indent(pair[1]) == indent(pair[0]) + 2
    })
}

/// A row of strings, integers (`#3`) and nulls (`null`).
fn row(items: &[&str]) -> Vec<Value> {
    items
        .iter()
        .map(|item| match *item {
            "null" => Value::Null,
            number if number.starts_with('#') => Value::Int64(number[1..].parse().unwrap()),
            text => Value::from(text),
        })
        .collect()
}

fn rows(items: &[&[&str]]) -> Vec<Vec<Value>> {
    items.iter().map(|items| row(items)).collect()
}

/// The queries with the rows they must return (sorted), in the languages
/// given.
type Cases = Vec<(&'static [Language], &'static str, Vec<Vec<Value>>)>;

fn assert_cases(cases: Cases) {
    let db = graph();
    let session = db.session();
    for (languages, query, expected) in cases {
        for language in languages {
            assert_eq!(
                sorted(&session, *language, query),
                expected,
                "{language:?} `{query}`"
            );
        }
    }
}

const BOTH: &[Language] = &[Language::Gql, Language::Cypher];
const GQL: &[Language] = &[Language::Gql];
const CYPHER: &[Language] = &[Language::Cypher];

/// A predicate that reads a column only one side of a join has (the length
/// of a named path, a value a subquery returns, an `UNWIND` variable) stays
/// above the join: on the other side that column is missing and every row
/// would fail.
#[test]
fn a_predicate_on_both_sides_stays_above_the_join() {
    let jules = rows(&[
        &["Jules", "Alix", "Django"],
        &["Jules", "Gus", "Django"],
        &["Jules", "Gus", "Hans"],
    ]);
    assert_cases(vec![
        (
            BOTH,
            "MATCH (x:B), (a:A), p = (a)-[:R*1..2]->(c) WHERE length(p) >= x.k \
             RETURN x.name, a.name, c.name",
            jules.clone(),
        ),
        (
            BOTH,
            "MATCH (x:B), (a:A), p = (a)-[:R]->(c) WHERE length(p) >= x.k \
             RETURN x.name, a.name, c.name",
            jules.clone(),
        ),
        (
            BOTH,
            "MATCH (a:A), (x:B), p = (a)-[:R]->(c) WHERE length(p) >= x.k \
             RETURN x.name, a.name, c.name",
            jules.clone(),
        ),
        (
            CYPHER,
            "MATCH (x:B), (a:A) OPTIONAL MATCH p = (a)-[:R*1..2]->(c) \
             WITH * WHERE length(p) >= x.k RETURN x.name, a.name, c.name",
            jules,
        ),
        (
            BOTH,
            "UNWIND [2] AS u MATCH (x:B), (a:A), (a)-[:R]->(c) WHERE c.k = u \
             RETURN x.name, a.name, c.name",
            rows(&[
                &["Butch", "Alix", "Django"],
                &["Butch", "Gus", "Django"],
                &["Jules", "Alix", "Django"],
                &["Jules", "Gus", "Django"],
                &["Mia", "Alix", "Django"],
                &["Mia", "Gus", "Django"],
            ]),
        ),
    ]);
}

/// `length(p)` reads a column of the expand that binds `p`: a filter on it
/// stays above that expand.
#[test]
fn a_filter_on_a_path_length_stays_above_its_expand() {
    let paths = rows(&[&["Alix", "Django"], &["Gus", "Django"], &["Gus", "Hans"]]);
    assert_cases(vec![
        (
            BOTH,
            "MATCH p = (a:A)-[:R]->(c) WHERE length(p) >= 1 RETURN a.name, c.name",
            paths.clone(),
        ),
        (
            BOTH,
            "MATCH p = (a:A)-[:R*1..2]->(c) WHERE length(p) = 1 RETURN a.name, c.name",
            paths.clone(),
        ),
        (
            BOTH,
            "MATCH p = (a:A)-[:R*1..2]->(c) WHERE length(p) >= 1 AND a.k = 2 \
             RETURN a.name, c.name",
            rows(&[&["Gus", "Django"], &["Gus", "Hans"]]),
        ),
        (
            BOTH,
            "MATCH p = (a:A)-[:R]->(c) WHERE length(p) > 1 RETURN a.name, c.name",
            rows(&[]),
        ),
    ]);
    let db = graph();
    let session = db.session();
    for language in LANGUAGES {
        let plan = plan(
            &session,
            language,
            "MATCH p = (a:A)-[:R*1..2]->(c) WHERE length(p) >= 1 AND a.k = 2 RETURN a.name",
        );
        assert!(
            sits_on(
                &plan,
                "Filter (_path_length_p Ge 1)",
                "Expand (a)->[:R*1..2]->(c)"
            ) && sits_on(&plan, "Filter (a.k Eq 2)", "NodeScan (a:A)"),
            "{language:?}:\n{plan}"
        );
    }
}

/// A predicate on a value a `CALL` subquery returns stays above the call.
#[test]
fn a_predicate_on_a_returned_value_stays_above_the_call() {
    let equal_keys = rows(&[&["Alix", "Jules", "#1"], &["Gus", "Mia", "#2"]]);
    assert_cases(vec![
        (
            CYPHER,
            "MATCH (a:A) MATCH (b:B) CALL { WITH b RETURN b.k AS w } \
             WITH * WHERE a.k = w RETURN a.name, b.name, w",
            equal_keys.clone(),
        ),
        (
            GQL,
            "MATCH (a:A) MATCH (b:B) CALL (b) { RETURN b.k AS w } \
             FILTER a.k = w RETURN a.name, b.name, w",
            equal_keys.clone(),
        ),
        (
            GQL,
            "MATCH (a:A) MATCH (b:B) CALL (b) { RETURN b.k AS w } \
             WITH a, b, w WHERE a.k = w RETURN a.name, b.name, w",
            equal_keys,
        ),
        (
            CYPHER,
            "MATCH (x:B), (a:A) CALL { WITH a RETURN a.k * 2 AS w } \
             WITH * WHERE x.k = w RETURN x.name, a.name, w",
            rows(&[&["Butch", "Gus", "#4"], &["Mia", "Alix", "#2"]]),
        ),
        (
            CYPHER,
            "MATCH (b:B) CALL { WITH b RETURN b.k AS w } WITH * WHERE b.k = w RETURN b.name, w",
            rows(&[&["Butch", "#4"], &["Jules", "#1"], &["Mia", "#2"]]),
        ),
        (
            CYPHER,
            "MATCH (x:B), (a:A) CALL { WITH a RETURN a.k * 2 AS w } \
             WITH * WHERE a.k = w - 1 RETURN x.name, a.name, w",
            rows(&[
                &["Butch", "Alix", "#2"],
                &["Jules", "Alix", "#2"],
                &["Mia", "Alix", "#2"],
            ]),
        ),
        // A subquery without imports: its variables are its own.
        (
            CYPHER,
            "MATCH (f:A) CALL { MATCH (x:B), (y:C) RETURN x, y } WITH * WHERE f.k = x.k \
             RETURN f.name, x.name, y.name",
            rows(&[
                &["Alix", "Jules", "Django"],
                &["Alix", "Jules", "Hans"],
                &["Gus", "Mia", "Django"],
                &["Gus", "Mia", "Hans"],
            ]),
        ),
    ]);
}

/// A `CALL` subquery that writes runs once per row of its input: a filter
/// written after it stays after it, so it writes for every row.
#[test]
fn a_filter_stays_after_a_call_that_writes() {
    for (language, query, expected_rows, created) in [
        (
            Language::Cypher,
            "MATCH (a:A) MATCH (b:B) CALL { WITH b CREATE (:T {p: b.name}) RETURN 1 AS one } \
             WITH * WHERE a.k = 1 RETURN a.name, b.name",
            rows(&[&["Alix", "Butch"], &["Alix", "Jules"], &["Alix", "Mia"]]),
            9,
        ),
        (
            Language::Cypher,
            "MATCH (a:A) CALL { WITH a CREATE (:T {p: a.name}) RETURN 1 AS one } \
             WITH * WHERE a.k = 1 RETURN a.name",
            rows(&[&["Alix"]]),
            3,
        ),
        (
            Language::Gql,
            "MATCH (a:A) MATCH (b:B) CALL (b) { INSERT (:T {p: b.name}) RETURN 1 AS one } \
             FILTER a.k = 1 RETURN a.name, b.name",
            rows(&[&["Alix", "Butch"], &["Alix", "Jules"], &["Alix", "Mia"]]),
            9,
        ),
        (
            Language::Gql,
            "MATCH (a:A) CALL (a) { INSERT (:T {p: a.name}) RETURN 1 AS one } \
             FILTER a.k = 1 RETURN a.name",
            rows(&[&["Alix"]]),
            3,
        ),
    ] {
        let db = graph();
        let session = db.session();
        assert_eq!(
            sorted(&session, language, query),
            expected_rows,
            "{language:?} `{query}`"
        );
        assert_eq!(
            run(&session, Language::Gql, "MATCH (t:T) RETURN count(t)"),
            [vec![Value::Int64(created)]],
            "{language:?} `{query}` writes once per row of its input"
        );
    }
}

/// A predicate on a variable an inner join equates filters both sides, each
/// right above the scan of that variable, before the join.
#[test]
fn a_predicate_on_a_join_key_filters_both_sides() {
    let db = graph();
    let session = db.session();
    let query = "MATCH (a:A), (b:B), (a)-[:R]->(c) WHERE a.k = 1 RETURN a.name, b.name, c.name";
    for language in LANGUAGES {
        let plan = plan(&session, language, query);
        let lines: Vec<&str> = plan.lines().map(operator).collect();
        assert_eq!(
            lines
                .iter()
                .filter(|line| **line == "Filter (a.k Eq 1)")
                .count(),
            2,
            "{language:?}:\n{plan}"
        );
        // One on each side, right above the scan of `a` there, and none
        // between the join and the `RETURN`.
        assert!(
            sits_on(&plan, "Filter (a.k Eq 1)", "NodeScan (a:A)")
                && sits_on(&plan, "Filter (a.k Eq 1)", "NodeScan (a:*)"),
            "{language:?}:\n{plan}"
        );
        assert!(
            sits_on(&plan, "Return (a.name, b.name, c.name)", "Join (Inner)"),
            "{language:?}:\n{plan}"
        );
        assert_eq!(
            sorted(&session, language, query),
            rows(&[
                &["Alix", "Butch", "Django"],
                &["Alix", "Jules", "Django"],
                &["Alix", "Mia", "Django"],
            ]),
            "{language:?}"
        );
    }
}

/// A conjunct that moves into the input of a later `MATCH` stays above
/// anything in that input that decides which rows there are: `LIMIT`, `SKIP`,
/// `DISTINCT`, an aggregate, and the order a `SORT` gives them.
#[test]
fn a_moved_conjunct_stays_above_what_shapes_the_input_rows() {
    let db = graph();
    let session = db.session();
    for (query, expected) in [
        (
            "MATCH (a:A) WITH a ORDER BY a.k DESC LIMIT 2 MATCH (b:B) WHERE a.k < 3 \
             RETURN a.name, b.name",
            rows(&[&["Gus", "Jules"], &["Gus", "Mia"], &["Gus", "Butch"]]),
        ),
        (
            "MATCH (a:A) WITH a ORDER BY a.k LIMIT 1 MATCH (b:B) WHERE a.k > 1 \
             RETURN a.name, b.name",
            rows(&[]),
        ),
        (
            "MATCH (a:A) WITH a ORDER BY a.k SKIP 1 MATCH (b:B) WHERE a.k > 1 \
             RETURN a.name, b.name",
            rows(&[
                &["Gus", "Jules"],
                &["Gus", "Mia"],
                &["Gus", "Butch"],
                &["Vincent", "Jules"],
                &["Vincent", "Mia"],
                &["Vincent", "Butch"],
            ]),
        ),
        (
            "MATCH (a:A) WITH a ORDER BY a.k DESC MATCH (b:B) WHERE a.k > 1 \
             RETURN a.name, b.name",
            rows(&[
                &["Vincent", "Jules"],
                &["Vincent", "Mia"],
                &["Vincent", "Butch"],
                &["Gus", "Jules"],
                &["Gus", "Mia"],
                &["Gus", "Butch"],
            ]),
        ),
        (
            "MATCH (a:A) WITH DISTINCT a.k % 2 AS k MATCH (b:B) WHERE k = 1 AND b.k = k \
             RETURN k, b.name",
            rows(&[&["#1", "Jules"]]),
        ),
        (
            "MATCH (a:A) WITH a.k % 2 AS k, count(*) AS c MATCH (b:B) WHERE c >= 2 AND b.k = k \
             RETURN k, c, b.name",
            rows(&[&["#1", "#2", "Jules"]]),
        ),
    ] {
        assert_eq!(
            run(&session, Language::Cypher, query),
            expected,
            "`{query}`"
        );
    }
}

/// The `WHERE` of the first `MATCH` is pushed down too when a later `MATCH`
/// runs on top of it: each conjunct right above the node it reads (#455, in
/// Cypher as written there).
#[test]
fn the_first_match_filter_is_pushed_down_below_a_later_scan() {
    let db = graph();
    let session = db.session();
    let query = "MATCH (a)-[:R]->(c) WHERE a.k = 2 AND c.k = 3 \
                 MATCH (b:B) WHERE b.k = a.k RETURN a.name, c.name, b.name";
    let plan = plan(&session, Language::Cypher, query);
    assert!(
        sits_on(&plan, "Filter (a.k Eq 2)", "NodeScan (a:*)")
            && sits_on(&plan, "Filter (c.k Eq 3)", "Expand (a)->[:R]->(c)"),
        "{plan}"
    );
    assert_eq!(
        sorted(&session, Language::Cypher, query),
        rows(&[&["Gus", "Hans", "Mia"]])
    );
}

/// `OPTIONAL MATCH` with a comma between its patterns: the rows are those
/// of the pattern as written, with nulls for a row without a match.
#[test]
fn an_optional_match_with_a_comma_keeps_its_rows() {
    let gus_and_alix = rows(&[
        &["Alix", "Django", "Butch"],
        &["Alix", "Django", "Jules"],
        &["Alix", "Django", "Mia"],
        &["Gus", "Django", "Butch"],
        &["Gus", "Django", "Jules"],
        &["Gus", "Django", "Mia"],
        &["Gus", "Hans", "Butch"],
        &["Gus", "Hans", "Jules"],
        &["Gus", "Hans", "Mia"],
    ]);
    assert_cases(vec![
        (
            CYPHER,
            "MATCH (a:A) WHERE a.k <= 2 OPTIONAL MATCH (a)-[:R]->(c), (b:B) \
             RETURN a.name, c.name, b.name",
            gus_and_alix.clone(),
        ),
        (
            CYPHER,
            "MATCH (a:A) WHERE a.k <= 2 OPTIONAL MATCH (b:B), (a)-[:R]->(c) \
             RETURN a.name, c.name, b.name",
            gus_and_alix.clone(),
        ),
        (
            GQL,
            "MATCH (a:A WHERE a.k <= 2) OPTIONAL MATCH (a)-[:R]->(c), (b:B) \
             RETURN a.name, c.name, b.name",
            gus_and_alix.clone(),
        ),
        (
            GQL,
            "MATCH (a:A WHERE a.k <= 2) OPTIONAL MATCH (b:B), (a)-[:R]->(c) \
             RETURN a.name, c.name, b.name",
            gus_and_alix,
        ),
        (
            BOTH,
            "MATCH (a:A) OPTIONAL MATCH (a)-[:R]->(c), (b:B) WHERE b.k = c.k \
             RETURN a.name, c.name, b.name",
            rows(&[
                &["Alix", "Django", "Mia"],
                &["Gus", "Django", "Mia"],
                &["Vincent", "null", "null"],
            ]),
        ),
        (
            CYPHER,
            "MATCH (a:A) OPTIONAL MATCH (a)-[:R]->(c), (b:B) WITH * WHERE c.k = a.k + 1 \
             RETURN a.name, c.name, b.name",
            rows(&[
                &["Alix", "Django", "Butch"],
                &["Alix", "Django", "Jules"],
                &["Alix", "Django", "Mia"],
                &["Gus", "Hans", "Butch"],
                &["Gus", "Hans", "Jules"],
                &["Gus", "Hans", "Mia"],
            ]),
        ),
    ]);
}

/// A variable a `WITH` drops or renames is not the variable of that name a
/// later `OPTIONAL MATCH` binds: a filter on the first does not filter the
/// second.
#[test]
fn a_filter_on_a_variable_a_with_drops_stays_with_it() {
    let reachable = |first: &str| {
        let mut all = Vec::new();
        for b in ["Butch", "Jules", "Mia"] {
            for (c, y) in [("Django", "Alix"), ("Django", "Gus"), ("Hans", "Gus")] {
                all.push(row(&[if first == "b" { b } else { first }, c, y]));
            }
        }
        all.sort_by_key(|row| format!("{row:?}"));
        all
    };
    assert_cases(vec![
        (
            BOTH,
            "MATCH (a:A) MATCH (b:B) WHERE a.k = 1 WITH b \
             OPTIONAL MATCH (a:C)<-[:R]-(y) RETURN b.name, a.name, y.name",
            reachable("b"),
        ),
        (
            BOTH,
            "MATCH (a:A) MATCH (b:B) WHERE a.k = 1 WITH a AS x, b \
             OPTIONAL MATCH (a:C)<-[:R]-(y) RETURN b.name, a.name, y.name",
            reachable("b"),
        ),
        (
            BOTH,
            "MATCH (a:A) WHERE a.k = 1 WITH a AS x \
             OPTIONAL MATCH (a:C)<-[:R]-(y) RETURN x.name, a.name, y.name",
            rows(&[
                &["Alix", "Django", "Alix"],
                &["Alix", "Django", "Gus"],
                &["Alix", "Hans", "Gus"],
            ]),
        ),
        // Renamed: the `a` after the `WITH` is the `C` node.
        (
            BOTH,
            "MATCH (a:A), (c:C) WHERE a.k = 1 WITH c AS a \
             OPTIONAL MATCH (a)<-[:R]-(y) RETURN a.name, y.name",
            rows(&[&["Django", "Alix"], &["Django", "Gus"], &["Hans", "Gus"]]),
        ),
    ]);
}

/// A condition that reads no variable filters every row where it is
/// written; moved down to the one empty row a query starts from, it filters
/// that row: with `$p = 1` the rows of the query, with `$p = 2` none.
#[test]
fn a_condition_without_variables_filters_the_row_a_query_starts_from() {
    let a_names = rows(&[&["Alix"], &["Gus"], &["Vincent"]]);
    let b_names = rows(&[&["#1", "Butch"], &["#1", "Jules"], &["#1", "Mia"]]);
    let cases: Vec<(&[Language], &str, Vec<Vec<Value>>)> = vec![
        (
            BOTH,
            "OPTIONAL MATCH (a:A) WITH * WHERE $p = 1 RETURN a.name",
            a_names.clone(),
        ),
        (
            GQL,
            "OPTIONAL MATCH (a:A) FILTER $p = 1 RETURN a.name",
            a_names.clone(),
        ),
        (
            BOTH,
            "OPTIONAL MATCH (a:Nothing) WITH * WHERE $p = 1 RETURN a.name",
            rows(&[&["null"]]),
        ),
        (
            GQL,
            "OPTIONAL MATCH (a:Nothing) FILTER $p = 1 RETURN a.name",
            rows(&[&["null"]]),
        ),
        (
            BOTH,
            "CALL { RETURN 1 AS x } WITH * WHERE $p = 1 RETURN x",
            rows(&[&["#1"]]),
        ),
        (
            GQL,
            "CALL { RETURN 1 AS x } FILTER $p = 1 RETURN x",
            rows(&[&["#1"]]),
        ),
        (
            BOTH,
            "CALL { MATCH (a:A) RETURN a } WITH * WHERE $p = 1 RETURN a.name",
            a_names.clone(),
        ),
        (
            GQL,
            "CALL { MATCH (a:A) RETURN a } FILTER $p = 1 RETURN a.name",
            a_names.clone(),
        ),
        (
            CYPHER,
            "WITH 1 AS x MATCH (b:B) WHERE $p = 1 RETURN x, b.name",
            b_names.clone(),
        ),
        (
            BOTH,
            "CALL { RETURN 1 AS x } MATCH (b:B) WHERE $p = 1 RETURN x, b.name",
            b_names,
        ),
        (
            BOTH,
            "OPTIONAL MATCH (a:A) MATCH (b:B) WHERE $p = 1 AND b.k = 1 RETURN a.name, b.name",
            rows(&[&["Alix", "Jules"], &["Gus", "Jules"], &["Vincent", "Jules"]]),
        ),
        (
            BOTH,
            "OPTIONAL MATCH (a:A) WITH a WHERE $p = 1 RETURN a.name",
            a_names,
        ),
        (
            CYPHER,
            "WITH 1 AS x WHERE $p = 1 RETURN x",
            rows(&[&["#1"]]),
        ),
    ];
    let db = graph();
    let session = db.session();
    for (languages, query, expected) in cases {
        for language in languages {
            for (p, expected) in [(1, expected.clone()), (2, Vec::new())] {
                let params = [("p".to_string(), Value::Int64(p))].into_iter().collect();
                let result = match language {
                    Language::Gql => session.execute_with_params(query, params),
                    Language::Cypher => session.execute_cypher_with_params(query, params),
                };
                let mut actual = result
                    .unwrap_or_else(|error| panic!("{language:?} `{query}` (p = {p}): {error}"))
                    .rows()
                    .to_vec();
                actual.sort_by_key(|row| format!("{row:?}"));
                assert_eq!(actual, expected, "{language:?} `{query}` (p = {p})");
            }
        }
    }
    // Conditions that always hold.
    for (language, query, expected) in [
        (
            Language::Cypher,
            "CALL { RETURN 1 AS x } WITH * WHERE true RETURN x",
            rows(&[&["#1"]]),
        ),
        (
            Language::Gql,
            "CALL { RETURN 1 AS x } WITH * WHERE true RETURN x",
            rows(&[&["#1"]]),
        ),
        (
            Language::Cypher,
            "CALL { RETURN 1 AS x } WITH * WHERE 1 = 1 RETURN x",
            rows(&[&["#1"]]),
        ),
        (
            Language::Gql,
            "CALL { RETURN 1 AS x } FILTER true RETURN x",
            rows(&[&["#1"]]),
        ),
        (
            Language::Gql,
            "CALL { RETURN 1 AS x } FILTER 1 = 1 RETURN x",
            rows(&[&["#1"]]),
        ),
        (
            Language::Cypher,
            "WITH 1 AS x MATCH (b:B) WHERE true AND b.k = 1 RETURN x, b.name",
            rows(&[&["#1", "Jules"]]),
        ),
    ] {
        assert_eq!(
            sorted(&session, language, query),
            expected,
            "{language:?} `{query}`"
        );
    }
}
