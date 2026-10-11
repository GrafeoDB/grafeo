//! A write that takes a value from a node or edge the same statement (or an
//! earlier statement of its transaction) created writes that value: `CREATE
//! (a:N {k: 3}) CREATE (:P {k: a.k})` gives the `P` node `k: 3`. The value
//! came from a store read that did not see what the statement's transaction
//! wrote, so it wrote nothing: in a `MERGE` (`... MATCH (a:N {k: 5 - i})
//! MERGE (b:P {k: a.k})` merged every row into one `P` without `k`), a
//! `CREATE` of a node or edge, a `SET`, a `MERGE` of a relationship and its
//! `ON CREATE` and `ON MATCH`. In GQL and Cypher, in an auto-commit statement
//! and in a transaction (where the statement reads its own writes again
//! before the commit). These queries write, so they are no cases of the
//! differential test corpus (which only reads).
//!
//! ```bash
//! cargo test -p grafeo-engine --all-features --test write_values_from_new_entities
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

/// The values of the one column `read` returns.
fn values(session: &Session, read: &str) -> Vec<Value> {
    run(session, Language::Gql, read)
        .into_iter()
        .map(|mut row| row.remove(0))
        .collect()
}

fn ints(values: &[i64]) -> Vec<Value> {
    values.iter().copied().map(Value::Int64).collect()
}

/// Runs `write` on a new database, in an auto-commit statement and in a
/// transaction, and checks that `read` returns `expected` after it (in the
/// transaction before and after the commit). `setup` runs first, committed.
fn check(language: Language, setup: &str, write: &str, read: &str, expected: &[i64]) {
    let expected = ints(expected);
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    if !setup.is_empty() {
        run(&session, Language::Gql, setup);
    }
    run(&session, language, write);
    assert_eq!(
        values(&session, read),
        expected,
        "{language:?} `{write}` in an auto-commit statement"
    );

    let db = GrafeoDB::new_in_memory();
    let mut session = db.session();
    if !setup.is_empty() {
        run(&session, Language::Gql, setup);
    }
    session.begin_transaction().unwrap();
    run(&session, language, write);
    assert_eq!(
        values(&session, read),
        expected,
        "{language:?} `{write}` in a transaction, before the commit"
    );
    session.commit().unwrap();
    assert_eq!(
        values(&session, read),
        expected,
        "{language:?} `{write}` in a transaction, after the commit"
    );
}

const P_VALUES: &str = "MATCH (b:P) RETURN b.k ORDER BY b.k";
const R_VALUES: &str = "MATCH ()-[r:R]->() RETURN r.w ORDER BY r.w";

/// The case of the bug report: the `MATCH` after the write finds the node
/// the row `5 - i` created, and the `MERGE` reads its `k`. Four rows merge
/// four `P` nodes, not one without `k`.
#[test]
fn a_merge_reads_a_node_created_earlier_in_the_statement() {
    for (language, write) in [
        (
            Language::Gql,
            "FOR i IN range(1, 4) INSERT (:N {k: i}) WITH i \
             MATCH (a:N {k: 5 - i}) MERGE (b:P {k: a.k})",
        ),
        (
            Language::Cypher,
            "UNWIND range(1, 4) AS i CREATE (:N {k: i}) WITH i \
             MATCH (a:N {k: 5 - i}) MERGE (b:P {k: a.k})",
        ),
    ] {
        check(language, "", write, P_VALUES, &[1, 2, 3, 4]);
    }
}

/// A new node takes a property of a node, or of an edge, created before it
/// in the statement.
#[test]
fn a_create_reads_a_node_or_edge_created_earlier_in_the_statement() {
    for (language, write) in [
        (
            Language::Gql,
            "FOR i IN [3, 19] INSERT (a:N {k: i}) INSERT (:P {k: a.k})",
        ),
        (
            Language::Cypher,
            "UNWIND [3, 19] AS i CREATE (a:N {k: i}) CREATE (:P {k: a.k})",
        ),
        (
            Language::Cypher,
            "CREATE (a:N {k: 3}) CREATE (:P {k: a.k}) \
             WITH a CREATE (b:N {k: 19}) CREATE (:P {k: b.k})",
        ),
        (
            Language::Gql,
            "FOR i IN [3, 19] INSERT (:N)-[r:R {w: i}]->(:Q) INSERT (:P {k: r.w})",
        ),
        (
            Language::Cypher,
            "UNWIND [3, 19] AS i CREATE (:N)-[r:R {w: i}]->(:Q) CREATE (:P {k: r.w})",
        ),
    ] {
        check(language, "", write, P_VALUES, &[3, 19]);
    }
}

/// A new edge, created or merged, takes a property of a node created before
/// it in the statement.
#[test]
fn an_edge_write_reads_a_node_created_earlier_in_the_statement() {
    for (language, write) in [
        (
            Language::Gql,
            "FOR i IN [3, 19] INSERT (a:N {k: i}) INSERT (a)-[:R {w: a.k}]->(:Q)",
        ),
        (
            Language::Cypher,
            "UNWIND [3, 19] AS i CREATE (a:N {k: i}) CREATE (a)-[:R {w: a.k}]->(:Q)",
        ),
        (
            Language::Gql,
            "FOR i IN [3, 19] INSERT (a:N {k: i}), (c:Q) MERGE (a)-[:R {w: a.k}]->(c)",
        ),
        (
            Language::Cypher,
            "UNWIND [3, 19] AS i CREATE (a:N {k: i}), (c:Q) MERGE (a)-[:R {w: a.k}]->(c)",
        ),
        (
            Language::Cypher,
            "UNWIND [3, 19] AS i CREATE (a:N {k: i}), (c:Q) \
             MERGE (a)-[r:R]->(c) ON CREATE SET r.w = a.k",
        ),
    ] {
        check(language, "", write, R_VALUES, &[3, 19]);
    }
}

/// A `SET` on a node or edge takes a property of a node created before it in
/// the statement, and so do a `MERGE`'s `ON CREATE` and `ON MATCH`.
#[test]
fn a_set_reads_a_node_created_earlier_in_the_statement() {
    for (language, setup, write, read, expected) in [
        (
            Language::Gql,
            "",
            "FOR i IN [3, 19] INSERT (a:N {k: i}), (b:P) SET b.k = a.k",
            P_VALUES,
            &[3, 19][..],
        ),
        (
            Language::Cypher,
            "",
            "UNWIND [3, 19] AS i CREATE (a:N {k: i}), (b:P) SET b.k = a.k",
            P_VALUES,
            &[3, 19],
        ),
        (
            Language::Cypher,
            "",
            "UNWIND [3, 19] AS i CREATE (a:N {k: i})-[r:R]->(:Q) SET r.w = a.k",
            R_VALUES,
            &[3, 19],
        ),
        (
            Language::Cypher,
            "",
            "UNWIND [3, 19] AS i CREATE (a:N {k: i}) MERGE (b:P {id: i}) ON CREATE SET b.k = a.k",
            P_VALUES,
            &[3, 19],
        ),
        (
            Language::Gql,
            "",
            "FOR i IN [3, 19] INSERT (a:N {k: i}) MERGE (b:P {id: i}) ON CREATE SET b.k = a.k",
            P_VALUES,
            &[3, 19],
        ),
        (
            Language::Cypher,
            "INSERT (:P {id: 3}), (:P {id: 19})",
            "UNWIND [3, 19] AS i CREATE (a:N {k: i + 88}) MERGE (b:P {id: i}) ON MATCH SET b.k = a.k",
            P_VALUES,
            &[91, 107],
        ),
    ] {
        check(language, setup, write, read, expected);
    }
}

/// A computed value (`a.k + 0`) read the statement's own writes already;
/// it stays so.
#[test]
fn a_computed_value_reads_a_node_created_earlier_in_the_statement() {
    for (language, write) in [
        (
            Language::Gql,
            "FOR i IN [3, 19] INSERT (a:N {k: i}) INSERT (:P {k: a.k + 0})",
        ),
        (
            Language::Cypher,
            "UNWIND [3, 19] AS i CREATE (a:N {k: i}) MERGE (:P {k: a.k + 0})",
        ),
    ] {
        check(language, "", write, P_VALUES, &[3, 19]);
    }
}

/// In a transaction, a write takes a value from a node or edge an earlier
/// statement of the transaction created, before the commit.
#[test]
fn a_write_reads_a_node_created_earlier_in_the_transaction() {
    for (language, create, write) in [
        (
            Language::Gql,
            "INSERT (:N {k: 3})-[:R {w: 19}]->(:Q)",
            "MATCH (a:N)-[r:R]->() INSERT (:P {k: a.k}), (:P {k: r.w})",
        ),
        (
            Language::Cypher,
            "CREATE (:N {k: 3})-[:R {w: 19}]->(:Q)",
            "MATCH (a:N)-[r:R]->() CREATE (:P {k: a.k}), (:P {k: r.w})",
        ),
        (
            Language::Cypher,
            "CREATE (:N {k: 3})-[:R {w: 19}]->(:Q)",
            "MATCH (a:N)-[r:R]->() MERGE (:P {k: a.k}) MERGE (b:P {k: r.w})",
        ),
        (
            Language::Cypher,
            "CREATE (:N {k: 3})-[:R {w: 19}]->(:Q), (:P), (:P {id: 1})",
            "MATCH (a:N)-[r:R]->(), (b:P), (c:P {id: 1}) WHERE b.id IS NULL \
             SET b.k = a.k, c.k = r.w",
        ),
    ] {
        let db = GrafeoDB::new_in_memory();
        let mut session = db.session();
        session.begin_transaction().unwrap();
        run(&session, language, create);
        run(&session, language, write);
        assert_eq!(
            values(&session, P_VALUES),
            ints(&[3, 19]),
            "{language:?} `{write}` before the commit"
        );
        session.commit().unwrap();
        assert_eq!(
            values(&session, P_VALUES),
            ints(&[3, 19]),
            "{language:?} `{write}` after the commit"
        );
    }
}
