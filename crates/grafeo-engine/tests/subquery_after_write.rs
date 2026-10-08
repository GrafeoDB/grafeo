//! A subquery after a write in the outer query sees everything the outer
//! query wrote before it: a `CALL`, `EXISTS`, `COUNT` or `VALUE` subquery
//! that runs per row of a writing input runs only after that whole input is
//! read, as a later `MATCH` does. Here the input writes the nodes `k: 1` to
//! `k: n`, one per row, and the row `i` looks for the node `k: n + 1 - i`,
//! which a later row writes for the first half of the rows. In GQL and
//! Cypher, with a scan and with a property index (a seek) inside the
//! subquery, and with more rows than a chunk holds. These queries write, so
//! they are no cases of the differential test corpus (which only reads).
//!
//! ```bash
//! cargo test -p grafeo-engine --all-features --test subquery_after_write
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

/// The rows the input writes and returns, one node `(:N:M {k: i})` each.
const ROWS: i64 = 300;

/// More rows than a chunk holds (2048).
const MANY_ROWS: i64 = 2100;

/// The writing input of `rows` rows followed by `rest`, in which `LAST`
/// stands for `rows + 1`. Cypher needs a `WITH` between the write and a
/// reading clause; GQL's `CALL` follows the `INSERT` directly (GQL takes no
/// `CALL` after a `WITH`).
fn after_write(language: Language, rows: i64, rest: &str) -> String {
    let rest = rest.replace("LAST", &(rows + 1).to_string());
    match language {
        Language::Gql => format!("FOR i IN range(1, {rows}) INSERT (n:N:M {{k: i}}) {rest}"),
        Language::Cypher if rest.starts_with("WITH ") => {
            format!("UNWIND range(1, {rows}) AS i CREATE (n:N:M {{k: i}}) {rest}")
        }
        Language::Cypher => {
            format!("UNWIND range(1, {rows}) AS i CREATE (n:N:M {{k: i}}) WITH i {rest}")
        }
    }
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

/// The text of the plan `EXPLAIN query` returns.
fn explain(session: &Session, language: Language, query: &str) -> String {
    run(session, language, &format!("EXPLAIN {query}"))
        .iter()
        .map(|row| match &row[0] {
            Value::String(text) => text.to_string(),
            other => format!("{other:?}"),
        })
        .collect::<Vec<_>>()
        .join("\n")
}

/// Runs the writing input of `rows` rows followed by `rest` in `language` on
/// a new database, with or without a property index on `k`, and checks the
/// rows. With the index, the subquery of a `CALL` looks the key up (`EXPLAIN`
/// shows the index; it prints an `EXISTS` or `COUNT` subquery without its
/// plan), so both plans are covered.
fn check(
    language: Language,
    rows: i64,
    indexed: bool,
    rest: &str,
    expected: &dyn Fn(i64) -> Vec<Vec<Value>>,
) {
    let db = GrafeoDB::new_in_memory();
    if indexed {
        db.create_property_index("k").unwrap();
    }
    let session = db.session();
    let query = after_write(language, rows, rest);
    if indexed && rest.contains("CALL {") && rest.contains("LAST - i") {
        let plan = explain(&session, language, &query);
        assert!(
            plan.contains("[index"),
            "{language:?} `{query}` looks the key up:\n{plan}"
        );
    }
    assert_eq!(
        run(&session, language, &query),
        expected(rows),
        "{language:?} `{query}` (index: {indexed})"
    );
}

/// [`check`] in `languages`, with and without the index.
fn assert_after_write_in(
    languages: &[Language],
    rest: &str,
    expected: impl Fn(i64) -> Vec<Vec<Value>>,
) {
    for &language in languages {
        for indexed in [false, true] {
            check(language, ROWS, indexed, rest, &expected);
        }
    }
}

/// [`assert_after_write_in`] for GQL and Cypher.
fn assert_after_write(rest: &str, expected: impl Fn(i64) -> Vec<Vec<Value>>) {
    assert_after_write_in(&LANGUAGES, rest, expected);
}

fn int(value: i64) -> Value {
    Value::Int64(value)
}

/// One row with the number of rows the input wrote.
fn all_rows(rows: i64) -> Vec<Vec<Value>> {
    vec![vec![int(rows)]]
}

/// The first three rows, each with the key of the node it found, which a
/// later row wrote.
fn first_three_keys(rows: i64) -> Vec<Vec<Value>> {
    (1..=3).map(|i| vec![int(i), int(rows + 1 - i)]).collect()
}

/// The first three rows, each with `found`.
fn first_three(found: Value) -> impl Fn(i64) -> Vec<Vec<Value>> {
    move |_| (1..=3).map(|i| vec![int(i), found.clone()]).collect()
}

const CORRELATED_CALL: &str =
    "CALL { WITH i MATCH (t:N:M {k: LAST - i}) RETURN t } RETURN count(*) AS found";

const EXISTS_FILTER: &str =
    "WITH i WHERE EXISTS { MATCH (t:N:M {k: LAST - i}) } RETURN count(*) AS found";

const COUNT_PROJECTION: &str =
    "WITH i, COUNT { MATCH (t:N:M {k: LAST - i}) } AS found RETURN sum(found) AS found";

#[test]
fn a_correlated_call_sees_the_whole_write() {
    assert_after_write(CORRELATED_CALL, all_rows);
    assert_after_write(
        "CALL { WITH i MATCH (t:N:M {k: LAST - i}) RETURN t.k AS k } \
         RETURN i, k ORDER BY i LIMIT 3",
        first_three_keys,
    );
}

/// GQL has `OPTIONAL CALL`; openCypher has no such clause.
#[test]
fn an_optional_call_sees_the_whole_write() {
    assert_after_write_in(
        &[Language::Gql],
        "OPTIONAL CALL { WITH i MATCH (t:N:M {k: LAST - i}) RETURN t.k AS k } \
         RETURN i, k ORDER BY i LIMIT 3",
        first_three_keys,
    );
}

#[test]
fn an_uncorrelated_call_sees_the_whole_write() {
    assert_after_write(
        "CALL { MATCH (t:N:M) RETURN count(t) AS c } RETURN min(c) AS fewest",
        all_rows,
    );
}

#[test]
fn an_exists_filter_sees_the_whole_write() {
    assert_after_write(EXISTS_FILTER, all_rows);
    assert_after_write(
        "WITH i WHERE NOT EXISTS { MATCH (t:N:M {k: LAST - i}) } RETURN count(*) AS missing",
        |_| vec![vec![int(0)]],
    );
}

#[test]
fn an_exists_value_sees_the_whole_write() {
    assert_after_write(
        "RETURN i, EXISTS { MATCH (t:N:M {k: LAST - i}) } AS found ORDER BY i LIMIT 3",
        first_three(Value::Bool(true)),
    );
}

#[test]
fn a_count_subquery_sees_the_whole_write() {
    assert_after_write(COUNT_PROJECTION, all_rows);
    assert_after_write(
        "RETURN i, COUNT { MATCH (t:N:M {k: LAST - i}) } AS found ORDER BY i LIMIT 3",
        first_three(int(1)),
    );
}

/// GQL's `VALUE` subquery takes the same path as `COUNT`.
#[test]
fn a_value_subquery_sees_the_whole_write() {
    assert_after_write_in(
        &[Language::Gql],
        "RETURN i, VALUE { MATCH (t:N:M {k: LAST - i}) RETURN t.k } AS k ORDER BY i LIMIT 3",
        first_three_keys,
    );
}

/// An input of more rows than a chunk holds is read whole too (with the
/// index: a scan per row of so many rows would make the test slow).
#[test]
fn a_write_of_more_rows_than_a_chunk_is_seen_whole() {
    for language in LANGUAGES {
        for rest in [CORRELATED_CALL, EXISTS_FILTER, COUNT_PROJECTION] {
            check(language, MANY_ROWS, true, rest, &all_rows);
        }
    }
}

/// An `EXISTS` tied to the row by a node of its pattern alone, which runs as
/// a semi-join without a write, sees the whole write too: here each row's
/// own node and path.
#[test]
fn an_exists_on_a_written_node_sees_the_whole_write() {
    let path = "(a)-[:R]->(:Q)-[:S]->(:T)";
    for (language, write) in [
        (
            Language::Gql,
            format!("FOR i IN range(1, {ROWS}) INSERT (a:P {{k: i}})-[:R]->(:Q)-[:S]->(:T) WITH a"),
        ),
        (
            Language::Cypher,
            format!(
                "UNWIND range(1, {ROWS}) AS i CREATE (a:P {{k: i}})-[:R]->(:Q)-[:S]->(:T) WITH a"
            ),
        ),
    ] {
        for (condition, expected) in [("EXISTS", ROWS), ("NOT EXISTS", 0)] {
            let db = GrafeoDB::new_in_memory();
            let session = db.session();
            let query =
                format!("{write} WHERE {condition} {{ MATCH {path} }} RETURN count(*) AS found");
            assert_eq!(
                run(&session, language, &query),
                [[int(expected)]],
                "{language:?} `{query}`"
            );
        }
    }
}

/// An `EXISTS` or `COUNT` of one edge from a node of the row, which the edge
/// check answers as each row arrives when nothing writes, sees the whole
/// write too: the row `i` merges the hubs `i` and `i - 1` and links them, so
/// each hub but the last gets its incoming edge from the row after its own.
#[test]
fn an_edge_check_sees_the_whole_write() {
    for (language, write) in [
        (
            Language::Gql,
            "FOR i IN range(1, 4) MERGE (a:Hub {k: i}) MERGE (b:Hub {k: i - 1}) \
             INSERT (a)-[:R]->(b) WITH a, i",
        ),
        (
            Language::Cypher,
            "UNWIND range(1, 4) AS i MERGE (a:Hub {k: i}) MERGE (b:Hub {k: i - 1}) \
             CREATE (a)-[:R]->(b) WITH a, i",
        ),
    ] {
        let cases: [(&str, Vec<Vec<Value>>); 2] = [
            (
                "WHERE EXISTS { (a)<-[:R]-() } RETURN count(*) AS linked",
                vec![vec![int(3)]],
            ),
            (
                "RETURN i, COUNT { (a)<-[:R]-() } AS linked ORDER BY i",
                (1..=4)
                    .map(|i| vec![int(i), int(i64::from(i < 4))])
                    .collect(),
            ),
        ];
        for (rest, expected) in cases {
            let db = GrafeoDB::new_in_memory();
            let session = db.session();
            let query = format!("{write} {rest}");
            assert_eq!(
                run(&session, language, &query),
                expected,
                "{language:?} `{query}`"
            );
        }
    }
}

/// A Cypher pattern comprehension runs as a subquery per row too, and sees
/// the whole write (the hubs of [`an_edge_check_sees_the_whole_write`]).
#[test]
fn a_pattern_comprehension_sees_the_whole_write() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    let query = "UNWIND range(1, 4) AS i MERGE (a:Hub {k: i}) MERGE (b:Hub {k: i - 1}) \
                 CREATE (a)-[:R]->(b) WITH a, i \
                 RETURN i, size([(a)<-[:R]-(c) | c.k]) AS linked ORDER BY i";
    assert_eq!(
        run(&session, Language::Cypher, query),
        (1..=4)
            .map(|i| vec![int(i), int(i64::from(i < 4))])
            .collect::<Vec<_>>(),
        "`{query}`"
    );
}
