//! Two user-reported write queries, rolled back and reopened.
//!
//! #317: a `MERGE` whose `ON MATCH SET` reads the merged node
//! (`n.cnt = n.cnt + 1`). #181: Cypher `MATCH (a) MATCH (b) CREATE
//! (a)-[:REL]->(b)`, which created two unlabeled phantom nodes. The bugs
//! were in planning; these tests check that what the fixed queries write is
//! undone by a rollback, and kept by a close and a crash.
//!
//! ```bash
//! cargo test -p grafeo-engine --all-features --test write_queries_rollback_and_reopen
//! ```

#![cfg(all(feature = "lpg", feature = "gql", feature = "cypher"))]

use grafeo_common::types::Value;
use grafeo_engine::GrafeoDB;

#[cfg(all(feature = "wal", feature = "grafeo-file"))]
mod common;

/// The #317 statement: counts how often `MERGE` saw the counter `id`.
fn merge_counter(id: i64) -> String {
    format!("MERGE (n:Counter {{id: {id}}}) ON CREATE SET n.cnt = 1 ON MATCH SET n.cnt = n.cnt + 1")
}

/// Every counter as `[id, cnt]`, by id.
fn counters(db: &GrafeoDB) -> Vec<Vec<Value>> {
    db.execute("MATCH (n:Counter) RETURN n.id AS id, n.cnt ORDER BY id")
        .unwrap()
        .rows()
        .to_vec()
}

fn counter(id: i64, cnt: i64) -> Vec<Value> {
    vec![Value::Int64(id), Value::Int64(cnt)]
}

/// The #181 setup, as the reporter wrote it.
const TWO_NODES: [&str; 2] = ["CREATE (:A {id: '1'})", "CREATE (:A {id: '2'})"];

/// The #181 statement.
const MATCH_MATCH_CREATE: &str =
    "MATCH (a:A {id: '1'}) MATCH (b:A {id: '2'}) CREATE (a)-[:REL]->(b)";

/// `[nodes, unlabeled nodes]`, then every edge as `[source id, type,
/// target id]`.
fn graph(db: &GrafeoDB) -> (Vec<Value>, Vec<Vec<Value>>) {
    let run = |query: &str| db.execute_cypher(query).unwrap().rows().to_vec();
    let nodes = vec![
        run("MATCH (n) RETURN count(n)")[0][0].clone(),
        run("MATCH (n) WHERE size(labels(n)) = 0 RETURN count(n)")[0][0].clone(),
    ];
    let edges = run("MATCH (a)-[r]->(b) RETURN a.id AS a, type(r), b.id ORDER BY a");
    (nodes, edges)
}

/// Two `:A` nodes and nothing unlabeled, with `edges` `REL` edges from 1 to 2.
fn expected_graph(edges: usize) -> (Vec<Value>, Vec<Vec<Value>>) {
    (
        vec![Value::Int64(2), Value::Int64(0)],
        vec![vec![Value::from("1"), Value::from("REL"), Value::from("2")]; edges],
    )
}

/// A rolled-back `ON MATCH SET` leaves the count the last commit set, and a
/// rolled-back `ON CREATE` leaves no node. Inside the transaction each
/// statement sees the one before.
#[test]
fn merge_self_reference_rolls_back() {
    let db = GrafeoDB::new_in_memory();
    db.execute(&merge_counter(1)).unwrap();

    let mut session = db.session();
    session.begin_transaction().unwrap();
    session.execute(&merge_counter(1)).unwrap();
    session.execute(&merge_counter(1)).unwrap();
    session.execute(&merge_counter(2)).unwrap();
    assert_eq!(
        session
            .execute("MATCH (n:Counter) RETURN n.id AS id, n.cnt ORDER BY id")
            .unwrap()
            .rows(),
        [counter(1, 3), counter(2, 1)],
        "inside the transaction"
    );
    session.rollback().unwrap();
    assert_eq!(counters(&db), [counter(1, 1)], "after the rollback");

    db.execute(&merge_counter(1)).unwrap();
    assert_eq!(counters(&db), [counter(1, 2)], "after the next commit");
}

/// A rolled-back `MATCH MATCH CREATE` leaves no edge and no phantom node.
#[test]
fn match_match_create_rolls_back() {
    let db = GrafeoDB::new_in_memory();
    for statement in TWO_NODES {
        db.execute_cypher(statement).unwrap();
    }
    let mut session = db.session();
    session.begin_transaction().unwrap();
    session.execute_cypher(MATCH_MATCH_CREATE).unwrap();
    session.rollback().unwrap();
    assert_eq!(graph(&db), expected_graph(0), "after the rollback");

    db.execute_cypher(MATCH_MATCH_CREATE).unwrap();
    assert_eq!(graph(&db), expected_graph(1), "after a commit");
}

#[cfg(all(feature = "wal", feature = "grafeo-file"))]
mod persistent {
    use std::path::Path;

    use super::*;
    use crate::common::replay::reopened_after_crash;

    fn open(path: &Path) -> GrafeoDB {
        GrafeoDB::open(path).unwrap()
    }

    /// The counts `MERGE` wrote are there after a close, and the reopened
    /// database counts on from them.
    #[test]
    fn merge_self_reference_survives_close_and_reopen() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("db.grafeo");
        let db = open(&path);
        for id in [1, 1, 2] {
            db.execute(&merge_counter(id)).unwrap();
        }
        db.close().unwrap();
        drop(db);

        let db = open(&path);
        assert_eq!(counters(&db), [counter(1, 2), counter(2, 1)], "reopened");
        db.execute(&merge_counter(2)).unwrap();
        db.close().unwrap();
        drop(db);

        let db = open(&path);
        assert_eq!(
            counters(&db),
            [counter(1, 2), counter(2, 2)],
            "second reopen"
        );
        db.close().unwrap();
    }

    /// The counts are there after a crash, from the WAL, and a rolled-back
    /// `MERGE` after them stays rolled back.
    #[test]
    fn merge_self_reference_survives_a_crash() {
        let (_dir, db) = reopened_after_crash(
            "persistent::merge_self_reference_survives_a_crash",
            open,
            |db| {
                for id in [1, 1, 2] {
                    db.execute(&merge_counter(id)).unwrap();
                }
                let mut session = db.session();
                session.begin_transaction().unwrap();
                session.execute(&merge_counter(1)).unwrap();
                session.rollback().unwrap();
            },
        );
        assert_eq!(counters(&db), [counter(1, 2), counter(2, 1)]);
        db.execute(&merge_counter(1)).unwrap();
        assert_eq!(counters(&db), [counter(1, 3), counter(2, 1)], "counts on");
    }

    /// The edge `MATCH MATCH CREATE` made, and no phantom node, are there
    /// after a close.
    #[test]
    fn match_match_create_survives_close_and_reopen() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("db.grafeo");
        let db = open(&path);
        for statement in TWO_NODES {
            db.execute_cypher(statement).unwrap();
        }
        db.execute_cypher(MATCH_MATCH_CREATE).unwrap();
        db.close().unwrap();
        drop(db);

        let db = open(&path);
        assert_eq!(graph(&db), expected_graph(1));
        db.close().unwrap();
    }

    /// The same after a crash, with a second, rolled-back `MATCH MATCH
    /// CREATE` before it.
    #[test]
    fn match_match_create_survives_a_crash() {
        let (_dir, db) = reopened_after_crash(
            "persistent::match_match_create_survives_a_crash",
            open,
            |db| {
                for statement in TWO_NODES {
                    db.execute_cypher(statement).unwrap();
                }
                db.execute_cypher(MATCH_MATCH_CREATE).unwrap();
                let mut session = db.session();
                session.begin_transaction().unwrap();
                session.execute_cypher(MATCH_MATCH_CREATE).unwrap();
                session.rollback().unwrap();
            },
        );
        assert_eq!(graph(&db), expected_graph(1));
    }
}
