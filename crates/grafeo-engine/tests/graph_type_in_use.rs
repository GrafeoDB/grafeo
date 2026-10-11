//! A graph type that a graph has as its type is not dropped: ISO/IEC
//! 39075:2024 12.7 `<drop graph type statement>`, Syntax Rule 6 ("the graph
//! type identified by CGTPN shall not be referenced by any existing graph in
//! the GQL-catalog"), with or without `IF EXISTS`, which only spares a graph
//! type that does not exist (General Rule 1). `CREATE OR REPLACE GRAPH TYPE`
//! drops the graph type it replaces first (12.6, General Rule 2), so it is
//! refused too. A dropped graph types nothing and keeps nothing.
//!
//! ```bash
//! cargo test -p grafeo-engine --all-features --test graph_type_in_use
//! ```

#![cfg(all(feature = "gql", feature = "grafeo-file", feature = "wal"))]

mod common;

use std::path::Path;

use grafeo_common::types::Value;
use grafeo_engine::{Config, GrafeoDB};

fn open(path: &Path) -> GrafeoDB {
    GrafeoDB::with_config(Config::persistent(path)).unwrap()
}

fn rows(db: &GrafeoDB, query: &str) -> Vec<Vec<Value>> {
    db.execute(query)
        .unwrap_or_else(|error| panic!("{query}: {error}"))
        .rows()
        .to_vec()
}

/// The names of the graph types `SHOW GRAPH TYPES` lists.
fn graph_types(db: &GrafeoDB) -> Vec<Value> {
    let mut names: Vec<Value> = rows(db, "SHOW GRAPH TYPES")
        .into_iter()
        .map(|row| row[0].clone())
        .collect();
    names.sort_by_key(|name| format!("{name:?}"));
    names
}

/// A graph type and a graph typed by it.
const TYPED_GRAPH: &[&str] = &[
    "CREATE NODE TYPE City (name STRING)",
    "CREATE GRAPH TYPE atlas (NODE TYPE City)",
    "CREATE GRAPH europe TYPED atlas",
];

/// The statements that would drop `atlas`.
const DROPS: &[&str] = &[
    "DROP GRAPH TYPE atlas",
    "DROP GRAPH TYPE IF EXISTS atlas",
    "CREATE OR REPLACE GRAPH TYPE atlas (NODE TYPE Town (name STRING))",
];

#[test]
fn a_graph_type_a_graph_has_is_not_dropped() {
    let db = GrafeoDB::new_in_memory();
    for statement in TYPED_GRAPH {
        rows(&db, statement);
    }
    for statement in DROPS {
        let error = db
            .execute(statement)
            .map(|_| ())
            .expect_err(statement)
            .to_string();
        assert!(
            error.contains("'atlas'") && error.contains("'europe'"),
            "{statement}: {error}"
        );
    }
    assert_eq!(graph_types(&db), [Value::from("atlas")]);
    assert!(
        rows(&db, "SHOW NODE TYPES")
            .iter()
            .all(|row| row[0] != Value::from("Town")),
        "the refused CREATE OR REPLACE declared its element types"
    );

    // Once the graph is gone, so may the graph type be.
    rows(&db, "DROP GRAPH europe");
    rows(&db, "DROP GRAPH TYPE atlas");
    assert_eq!(graph_types(&db), Vec::<Value>::new());
}

/// In a schema too: the graph and the graph type are named in it.
#[test]
fn a_graph_type_a_graph_in_a_schema_has_is_not_dropped() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    session.execute("CREATE SCHEMA travel").unwrap();
    session.execute("SESSION SET SCHEMA travel").unwrap();
    for statement in TYPED_GRAPH {
        session.execute(statement).unwrap();
    }
    let error = session
        .execute("DROP GRAPH TYPE atlas")
        .map(|_| ())
        .unwrap_err()
        .to_string();
    assert!(error.contains("europe"), "{error}");
    session.execute("DROP GRAPH europe").unwrap();
    session.execute("DROP GRAPH TYPE atlas").unwrap();
}

/// A refused drop is not logged: after a WAL replay the graph type is still
/// there.
#[test]
fn a_refused_drop_leaves_the_graph_type_after_a_replay() {
    let (_dir, db) = common::replay::reopened_after_crash(
        "a_refused_drop_leaves_the_graph_type_after_a_replay",
        open,
        |db| {
            for statement in TYPED_GRAPH {
                rows(db, statement);
            }
            for statement in DROPS {
                assert!(db.execute(statement).is_err(), "{statement}");
            }
        },
    );
    assert_eq!(graph_types(&db), [Value::from("atlas")]);
}
