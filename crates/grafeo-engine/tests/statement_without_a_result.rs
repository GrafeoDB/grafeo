//! A statement that ends with a write and no `RETURN` has no result: no
//! columns and no rows (openCypher; ISO GQL's omitted result, which `FINISH`
//! asks for too). Its write counters still say what it changed. A Cypher
//! query ending with `CREATE`, `MERGE`, `SET`, `REMOVE`, `DELETE`, `FOREACH`
//! or a unit `CALL` returned the planner's internal columns (`__list__`, `i`,
//! `_anon_0`) and a row of raw IDs for each row it wrote; a GQL one (`FOR ...
//! INSERT`, `MATCH ... SET`) returned a row without columns for each, and
//! `FINISH` returned no rows but the internal columns. A statement that ends
//! with a `RETURN` still returns it, and a standalone procedure call its
//! output (see `call_procedures`). These queries write, so they are no cases
//! of the differential test corpus (which only reads).
//!
//! ```bash
//! cargo test -p grafeo-engine --all-features --test statement_without_a_result
//! ```

#![cfg(all(feature = "lpg", feature = "gql", feature = "cypher"))]

use grafeo_common::types::Value;
use grafeo_engine::database::QueryResult;
use grafeo_engine::{GrafeoDB, Session};

#[derive(Debug, Clone, Copy)]
enum Language {
    Gql,
    Cypher,
}

fn execute(session: &Session, language: Language, query: &str) -> QueryResult {
    let result = match language {
        Language::Gql => session.execute(query),
        Language::Cypher => session.execute_cypher(query),
    };
    result.unwrap_or_else(|error| panic!("{language:?} `{query}` failed: {error}"))
}

/// Two `W` nodes with `k` 3 and 19.
const SETUP: &str = "INSERT (:W {k: 3}), (:W {k: 19})";

/// What a write changed, in the order of `WriteCounters`: nodes created and
/// deleted, edges created and deleted, properties set, labels added and
/// removed.
type Changes = [u64; 7];

fn changes(result: &QueryResult) -> Changes {
    let c = &result.counters;
    [
        c.nodes_created,
        c.nodes_deleted,
        c.edges_created,
        c.edges_deleted,
        c.properties_set,
        c.labels_added,
        c.labels_removed,
    ]
}

/// Each statement runs on a new database with the two `W` nodes: it returns
/// no columns and no rows, and its counters say what it wrote.
#[test]
fn a_write_without_return_has_no_result() {
    for (language, query, expected) in [
        // The case of the bug report.
        (
            Language::Cypher,
            "UNWIND [1, 2] AS i CREATE (:V)",
            [2, 0, 0, 0, 0, 2, 0],
        ),
        (Language::Cypher, "CREATE (:V)", [1, 0, 0, 0, 0, 1, 0]),
        (
            Language::Cypher,
            "CREATE (:V {k: 3})-[:R]->(:V)",
            [2, 0, 1, 0, 1, 2, 0],
        ),
        (
            Language::Cypher,
            "MERGE (:V {k: 88})",
            [1, 0, 0, 0, 1, 1, 0],
        ),
        (
            Language::Cypher,
            "UNWIND [3, 88] AS i MERGE (:W {k: i})",
            [1, 0, 0, 0, 1, 1, 0],
        ),
        (
            Language::Cypher,
            "MATCH (a:W {k: 3}), (b:W {k: 19}) MERGE (a)-[:R]->(b)",
            [0, 0, 1, 0, 0, 0, 0],
        ),
        (
            Language::Cypher,
            "MATCH (w:W) SET w.k = 88",
            [0, 0, 0, 0, 2, 0, 0],
        ),
        (
            Language::Cypher,
            "MATCH (w:W) SET w:V",
            [0, 0, 0, 0, 0, 2, 0],
        ),
        (
            Language::Cypher,
            "MATCH (w:W) REMOVE w.k",
            [0, 0, 0, 0, 2, 0, 0],
        ),
        (
            Language::Cypher,
            "MATCH (w:W) REMOVE w:W",
            [0, 0, 0, 0, 0, 0, 2],
        ),
        (
            Language::Cypher,
            "MATCH (w:W) DELETE w",
            [0, 2, 0, 0, 0, 0, 0],
        ),
        (
            Language::Cypher,
            "MATCH (w:W) DETACH DELETE w",
            [0, 2, 0, 0, 0, 0, 0],
        ),
        (
            Language::Cypher,
            "UNWIND [1, 2] AS i FOREACH (x IN [i] | CREATE (:V))",
            [2, 0, 0, 0, 0, 2, 0],
        ),
        (
            Language::Cypher,
            "UNWIND [1, 2] AS i CALL { CREATE (:V) }",
            [2, 0, 0, 0, 0, 2, 0],
        ),
        (
            Language::Gql,
            "FOR i IN [1, 2] INSERT (:V)",
            [2, 0, 0, 0, 0, 2, 0],
        ),
        (
            Language::Gql,
            "MATCH (w:W) SET w.k = 88",
            [0, 0, 0, 0, 2, 0, 0],
        ),
        (
            Language::Gql,
            "MATCH (w:W) REMOVE w.k",
            [0, 0, 0, 0, 2, 0, 0],
        ),
        (
            Language::Gql,
            "MATCH (w:W) DETACH DELETE w",
            [0, 2, 0, 0, 0, 0, 0],
        ),
        (
            Language::Gql,
            "FOR i IN [1, 2] INSERT (:V) FINISH",
            [2, 0, 0, 0, 0, 2, 0],
        ),
        (Language::Gql, "MATCH (w:W) FINISH", [0; 7]),
        // A GQL statement may end with a CALL that writes (it used to be
        // refused for want of a RETURN), also one whose body returns rows.
        (
            Language::Gql,
            "FOR i IN [1, 2] CALL (i) { INSERT (:V {i: i}) }",
            [2, 0, 0, 0, 2, 2, 0],
        ),
        (
            Language::Gql,
            "MATCH (w:W) CALL { INSERT (:V) }",
            [2, 0, 0, 0, 0, 2, 0],
        ),
        (
            Language::Gql,
            "MATCH (w:W) CALL (w) { SET w.k = 88 }",
            [0, 0, 0, 0, 2, 0, 0],
        ),
        (
            Language::Gql,
            "MATCH (w:W) CALL (w) { INSERT (v:V) RETURN v }",
            [2, 0, 0, 0, 0, 2, 0],
        ),
    ] {
        let session = GrafeoDB::new_in_memory().session();
        execute(&session, Language::Gql, SETUP);
        let result = execute(&session, language, query);
        assert_eq!(
            result.columns,
            Vec::<String>::new(),
            "{language:?} `{query}` has no columns"
        );
        assert_eq!(
            result.rows(),
            &[] as &[Vec<Value>],
            "{language:?} `{query}` has no rows"
        );
        assert_eq!(
            changes(&result),
            expected,
            "{language:?} `{query}` counts its writes"
        );
    }
}

/// The write runs to its end: every row of a long input writes, also past
/// the first chunk of rows, and in a transaction.
#[test]
fn a_write_without_return_writes_every_row() {
    for (language, query) in [
        (
            Language::Cypher,
            "UNWIND range(1, 3000) AS i CREATE (:V {i: i})",
        ),
        (Language::Gql, "FOR i IN range(1, 3000) INSERT (:V {i: i})"),
        (
            Language::Gql,
            "FOR i IN range(1, 3000) INSERT (:V {i: i}) FINISH",
        ),
    ] {
        let db = GrafeoDB::new_in_memory();
        let mut session = db.session();
        session.begin_transaction().unwrap();
        let result = execute(&session, language, query);
        assert!(
            result.columns.is_empty() && result.rows().is_empty(),
            "{language:?} `{query}` has no result"
        );
        assert_eq!(
            result.counters.nodes_created, 3000,
            "{language:?} `{query}`"
        );
        session.commit().unwrap();
        assert_eq!(
            execute(
                &session,
                Language::Gql,
                "MATCH (v:V) RETURN count(v), sum(v.i)"
            )
            .rows(),
            [[Value::Int64(3000), Value::Int64(4_501_500)]],
            "{language:?} `{query}` writes every row"
        );
    }
}

/// A statement that ends with a `RETURN` returns it, after a write too.
#[test]
fn a_write_with_return_keeps_its_result() {
    for (language, query) in [
        (
            Language::Cypher,
            "UNWIND [3, 19] AS i CREATE (v:V {k: i}) RETURN v.k AS k",
        ),
        (
            Language::Gql,
            "FOR i IN [3, 19] INSERT (v:V {k: i}) RETURN v.k AS k",
        ),
    ] {
        let session = GrafeoDB::new_in_memory().session();
        let result = execute(&session, language, query);
        assert_eq!(result.columns, ["k"], "{language:?} `{query}`");
        assert_eq!(
            result.rows(),
            [[Value::Int64(3)], [Value::Int64(19)]],
            "{language:?} `{query}`"
        );
        assert_eq!(result.counters.nodes_created, 2, "{language:?} `{query}`");
    }
}
