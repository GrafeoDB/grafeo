//! A `LIMIT 0` (and GQL's `FINISH`) after a write cuts the rows, not the
//! write: the write runs to its end and the clause passes no row on. A
//! `LIMIT 0` that never read its input wrote nothing: in a `CALL` body
//! (`FOR i IN [1, 2] CALL { INSERT (:W) FINISH } RETURN count(*) AS rows`
//! wrote no `W`, where two are right), and in a
//! statement that goes on after it (`... CREATE (:W) WITH j LIMIT 0 RETURN
//! j`). Only a `LIMIT 0` at the very end of a statement wrote, since the
//! statement reads its last clause the way that reads its input once. A
//! `LIMIT 1` there kept every write already. In GQL and Cypher. These queries
//! write, so they are no cases of the differential test corpus (which only
//! reads).
//!
//! ```bash
//! cargo test -p grafeo-engine --all-features --test limit_zero_after_a_write
//! ```

#![cfg(all(feature = "lpg", feature = "gql", feature = "cypher"))]

use grafeo_common::types::Value;
use grafeo_engine::{GrafeoDB, Session};

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

/// The number of nodes with `label`.
fn count(session: &Session, label: &str) -> Value {
    run(
        session,
        Language::Gql,
        &format!("MATCH (n:{label}) RETURN count(n)"),
    )[0][0]
        .clone()
}

/// Each query runs on a new database: the one row it returns (the rows
/// after the `CALL`) and the `W` nodes it writes, every row of the body's
/// write, however few rows the body passes on.
#[test]
fn a_limit_in_a_call_body_keeps_every_write() {
    for (language, query, rows, nodes) in [
        // The case of the bug report, with the RETURN GQL needs after a CALL.
        (
            Language::Gql,
            "FOR i IN [1, 2] CALL { INSERT (:W) FINISH } RETURN count(*) AS rows",
            2,
            2,
        ),
        (
            Language::Gql,
            "FOR i IN [1, 2] CALL (i) { FOR j IN [1, 2, 3] INSERT (:W {i: i, j: j}) FINISH } \
             RETURN count(*) AS rows",
            2,
            6,
        ),
        // A body that returns no row drops the row it ran for.
        (
            Language::Gql,
            "FOR i IN [1, 2] CALL { FOR j IN [1, 2, 3] INSERT (:W) RETURN j LIMIT 0 } \
             RETURN count(*) AS rows",
            0,
            6,
        ),
        (
            Language::Cypher,
            "UNWIND [1, 2] AS i CALL { UNWIND [1, 2, 3] AS j CREATE (:W) RETURN j LIMIT 0 } \
             RETURN count(*) AS rows",
            0,
            6,
        ),
        (
            Language::Cypher,
            "UNWIND [1, 2] AS i CALL { UNWIND [1, 2, 3] AS j CREATE (:W) RETURN j ORDER BY j LIMIT 0 } \
             RETURN count(*) AS rows",
            0,
            6,
        ),
        (
            Language::Cypher,
            "UNWIND [1, 2] AS i CALL { CREATE (w:W) RETURN count(w) AS c LIMIT 0 } \
             RETURN count(*) AS rows",
            0,
            2,
        ),
        // A unit body: the LIMIT 0 in its middle ends its rows, not its write.
        (
            Language::Cypher,
            "UNWIND [1, 2] AS i CALL { WITH i UNWIND [1, 2, 3] AS j CREATE (:W) \
             WITH j LIMIT 0 CREATE (:V) } RETURN count(*) AS rows",
            2,
            6,
        ),
        // A LIMIT 1 kept every write already.
        (
            Language::Gql,
            "FOR i IN [1, 2] CALL { FOR j IN [1, 2, 3] INSERT (:W) RETURN j LIMIT 1 } \
             RETURN count(*) AS rows",
            2,
            6,
        ),
        (
            Language::Cypher,
            "UNWIND [1, 2] AS i CALL { UNWIND [1, 2, 3] AS j CREATE (:W) RETURN j LIMIT 1 } \
             RETURN count(*) AS rows",
            2,
            6,
        ),
    ] {
        let session = GrafeoDB::new_in_memory().session();
        assert_eq!(
            run(&session, language, query),
            [[int(rows)]],
            "{language:?} `{query}`"
        );
        assert_eq!(
            count(&session, "W"),
            int(nodes),
            "{language:?} the W nodes `{query}` writes"
        );
        assert_eq!(
            count(&session, "V"),
            int(0),
            "{language:?} `{query}` writes no V after its LIMIT 0"
        );
    }
}

/// A `LIMIT 0` in the middle of a statement: the write before it runs to its
/// end, and nothing after it runs.
#[test]
fn a_limit_zero_in_the_middle_of_a_statement_keeps_every_write() {
    for (language, query) in [
        (
            Language::Cypher,
            "UNWIND [1, 2, 3] AS j CREATE (:W) WITH j LIMIT 0 RETURN j",
        ),
        (
            Language::Cypher,
            "UNWIND [1, 2, 3] AS j CREATE (:W) WITH j LIMIT 0 CREATE (:V) RETURN j",
        ),
        (
            Language::Cypher,
            "UNWIND [1, 2, 3] AS j CREATE (w:W) WITH w LIMIT 0 SET w.k = 3 RETURN w.k",
        ),
    ] {
        let session = GrafeoDB::new_in_memory().session();
        assert_eq!(
            run(&session, language, query),
            Vec::<Vec<Value>>::new(),
            "{language:?} `{query}` passes no row on"
        );
        assert_eq!(
            count(&session, "W"),
            int(3),
            "{language:?} `{query}` writes every W"
        );
        assert_eq!(
            count(&session, "V"),
            int(0),
            "{language:?} `{query}` writes nothing after its LIMIT 0"
        );
        assert_eq!(
            run(
                &session,
                Language::Gql,
                "MATCH (w:W) WHERE w.k IS NOT NULL RETURN count(w)"
            ),
            [[int(0)]],
            "{language:?} `{query}` sets nothing after its LIMIT 0"
        );
    }
}
