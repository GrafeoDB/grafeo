//! A `CALL` subquery without a final `RETURN` (a unit subquery) runs for its
//! writes and passes each row on once, as it came in (openCypher; GQL's
//! inline procedure call without a result), as `FOREACH` does: a body that
//! makes two rows of one writes twice and keeps the one row, a body that
//! makes none writes nothing and keeps every row, and nothing the body binds
//! is a variable after it. A body with a `RETURN` still joins its rows to
//! the row. In GQL and Cypher. These queries write, so they are no cases of
//! the differential test corpus (which only reads).
//!
//! ```bash
//! cargo test -p grafeo-engine --all-features --test unit_call_rows
//! ```

#![cfg(all(feature = "lpg", feature = "gql", feature = "cypher"))]

use grafeo_common::types::Value;
use grafeo_engine::{Config, GrafeoDB, Session};

#[derive(Debug, Clone, Copy)]
enum Language {
    Gql,
    Cypher,
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

fn int(value: i64) -> Value {
    Value::Int64(value)
}

/// The number of nodes with the label `W`.
fn written(session: &Session) -> Value {
    run(session, Language::Gql, "MATCH (w:W) RETURN count(w)")[0][0].clone()
}

/// Each query runs on a new database: the rows it returns and the `W` nodes
/// it writes, whatever number of rows the body makes.
#[test]
fn a_unit_call_keeps_one_row_per_input_row() {
    for (language, query, rows, nodes) in [
        (
            Language::Cypher,
            "UNWIND [1] AS i CALL { WITH i UNWIND [1, 2] AS x CREATE (:W) } RETURN count(*) AS rows",
            1,
            2,
        ),
        (
            Language::Cypher,
            "UNWIND [1, 2, 3] AS i CALL { WITH i UNWIND [] AS x CREATE (:W) } \
             RETURN count(*) AS rows",
            3,
            0,
        ),
        (
            Language::Cypher,
            "UNWIND [3, 19] AS i CALL { WITH i UNWIND range(1, i) AS x CREATE (:W {i: i, x: x}) } \
             RETURN count(*) AS rows",
            2,
            22,
        ),
        // Without an importing WITH, the body runs without the row.
        (
            Language::Cypher,
            "UNWIND [1, 2] AS i CALL { UNWIND [3, 19, 88] AS x CREATE (:W {x: x}) } \
             RETURN count(*) AS rows",
            2,
            6,
        ),
        // A unit subquery in a unit subquery.
        (
            Language::Cypher,
            "UNWIND [1] AS i CALL { WITH i UNWIND [1, 2] AS x \
             CALL { WITH x UNWIND [1, 2, 3] AS y CREATE (:W {x: x, y: y}) } } \
             RETURN count(*) AS rows",
            1,
            6,
        ),
        (
            Language::Gql,
            "FOR i IN [1] CALL { UNWIND [1, 2] AS x INSERT (:W) } RETURN count(*) AS rows",
            1,
            2,
        ),
        (
            Language::Gql,
            "FOR i IN [1] CALL { FOR x IN [1, 2] INSERT (:W) } RETURN count(*) AS rows",
            1,
            2,
        ),
        (
            Language::Gql,
            "FOR i IN [1, 2] CALL (i) { FOR x IN [] INSERT (:W) } RETURN count(*) AS rows",
            2,
            0,
        ),
        (
            Language::Gql,
            "FOR i IN [3, 19] CALL (i) { FOR x IN range(1, i) INSERT (:W {i: i, x: x}) } \
             RETURN count(*) AS rows",
            2,
            22,
        ),
    ] {
        let session = GrafeoDB::new_in_memory().session();
        assert_eq!(
            run(&session, language, query),
            [[int(rows)]],
            "{language:?} `{query}`"
        );
        assert_eq!(
            written(&session),
            int(nodes),
            "{language:?} the nodes `{query}` writes"
        );
    }
}

/// The rows after a unit subquery are those before it, each once with its
/// values, and only their variables are there: `RETURN *` returns `i`
/// alone. Without `ORDER BY` the row order is unspecified, so the database
/// shuffles the rows (`shuffle_unordered`) and they are compared sorted.
#[test]
fn a_unit_call_passes_each_row_on_unchanged() {
    for (language, query) in [
        (
            Language::Cypher,
            "UNWIND [19, 3, 88] AS i CALL { WITH i UNWIND range(1, 2) AS x CREATE (:W {i: i}) } \
             RETURN i",
        ),
        (
            Language::Gql,
            "FOR i IN [19, 3, 88] CALL (i) { FOR x IN range(1, 2) INSERT (:W {i: i}) } RETURN i",
        ),
    ] {
        let db = GrafeoDB::with_config(Config::in_memory().with_shuffle_unordered(true)).unwrap();
        let session = db.session();
        let mut rows = run(&session, language, query);
        rows.sort_by_key(|row| row[0].as_int64());
        assert_eq!(
            rows,
            [[int(3)], [int(19)], [int(88)]],
            "{language:?} `{query}`"
        );
    }
    for (language, query) in [
        (
            Language::Cypher,
            "UNWIND [3] AS i CALL { WITH i UNWIND [1, 2] AS x CREATE (w:W) } RETURN *",
        ),
        (
            Language::Gql,
            "FOR i IN [3] CALL (i) { FOR x IN [1, 2] INSERT (w:W) } RETURN *",
        ),
    ] {
        let session = GrafeoDB::new_in_memory().session();
        let result = match language {
            Language::Gql => session.execute(query),
            Language::Cypher => session.execute_cypher(query),
        }
        .unwrap_or_else(|error| panic!("{language:?} `{query}` failed: {error}"));
        assert_eq!(
            result.columns,
            ["i"],
            "{language:?} the columns of `{query}`"
        );
        assert_eq!(result.rows(), [[int(3)]], "{language:?} `{query}`");
    }
}

/// What the body binds is not a variable after it.
#[test]
fn a_unit_call_binds_nothing_after_it() {
    for (language, query, variable) in [
        (
            Language::Cypher,
            "UNWIND [3] AS i CALL { WITH i UNWIND [1, 2] AS x CREATE (w:W) } RETURN x",
            "x",
        ),
        (
            Language::Cypher,
            "UNWIND [3] AS i CALL { WITH i UNWIND [1, 2] AS x CREATE (w:W) } RETURN w",
            "w",
        ),
        (
            Language::Gql,
            "FOR i IN [3] CALL (i) { FOR x IN [1, 2] INSERT (w:W) } RETURN w",
            "w",
        ),
    ] {
        let session = GrafeoDB::new_in_memory().session();
        let error = match language {
            Language::Gql => session.execute(query),
            Language::Cypher => session.execute_cypher(query),
        }
        .expect_err(&format!(
            "{language:?} `{query}` reads a variable of the body"
        ));
        assert!(
            error
                .to_string()
                .contains(&format!("Undefined variable '{variable}'")),
            "{language:?} `{query}`: {error}"
        );
        assert_eq!(
            written(&session),
            int(0),
            "a query that fails writes nothing"
        );
    }
}

/// A Cypher query may end with a unit subquery: it runs the body for each
/// row.
#[test]
fn a_query_can_end_with_a_unit_call() {
    let session = GrafeoDB::new_in_memory().session();
    run(
        &session,
        Language::Cypher,
        "UNWIND [1, 2] AS i CALL { WITH i UNWIND [3, 19] AS x CREATE (:W {i: i, x: x}) }",
    );
    assert_eq!(written(&session), int(4));
}

/// A body with a final `RETURN` joins its rows to the row as before: one row
/// per row the body returns, none for a body that returns none.
#[test]
fn a_call_with_a_return_joins_its_rows() {
    for (language, query, rows) in [
        (
            Language::Cypher,
            "UNWIND [1] AS i CALL { WITH i UNWIND [1, 2] AS x CREATE (:W) RETURN x } \
             RETURN count(*) AS rows",
            2,
        ),
        (
            Language::Cypher,
            "UNWIND [1, 2] AS i CALL { WITH i UNWIND [] AS x CREATE (:W) RETURN x } \
             RETURN count(*) AS rows",
            0,
        ),
        (
            Language::Gql,
            "FOR i IN [1] CALL { FOR x IN [1, 2] INSERT (:W) RETURN x } RETURN count(*) AS rows",
            2,
        ),
    ] {
        let session = GrafeoDB::new_in_memory().session();
        assert_eq!(
            run(&session, language, query),
            [[int(rows)]],
            "{language:?} `{query}`"
        );
    }
}

/// `PROFILE` runs the plan too: its top line counts the one row, and it
/// writes as the query does.
#[test]
fn a_unit_call_profiles() {
    for (language, query) in [
        (
            Language::Cypher,
            "PROFILE UNWIND [1] AS i CALL { WITH i UNWIND [1, 2] AS x CREATE (:W) } RETURN i",
        ),
        (
            Language::Gql,
            "PROFILE FOR i IN [1] CALL { FOR x IN [1, 2] INSERT (:W) } RETURN i",
        ),
    ] {
        let session = GrafeoDB::new_in_memory().session();
        let rows = run(&session, language, query);
        let Value::String(text) = &rows[0][0] else {
            panic!("{language:?} `{query}` returned {rows:?}");
        };
        let top = text.lines().next().unwrap_or_default();
        assert!(
            top.contains("rows=1"),
            "{language:?} the top line of\n{text}"
        );
        assert_eq!(written(&session), int(2), "{language:?} `{query}` writes");
    }
}
