//! `GrafeoDB::update_nodes(labels, key, rows)`: each row's properties merge
//! into the existing nodes with all of `labels` whose `key` equals the row's
//! key; no node is created. With a property index on the key and without:
//!
//! - a float key equal to an integer one matches it (2.0 matches 2);
//! - a node with only some of the labels is not matched;
//! - a row without the key, and a key no node has, are skipped and listed;
//! - a key two nodes share updates both and counts as one row;
//! - the properties merge: the others stay;
//! - one write: a row that breaks a UNIQUE constraint writes nothing;
//! - in a session's transaction, a rollback undoes it;
//! - the text "1" matches the integer 1, as `=` does today (the 0.6.1
//!   string-against-number decision changes this; update the test then).
//!
//! ```bash
//! cargo test -p grafeo-engine --all-features --test update_nodes_by_key
//! ```

#![cfg(feature = "gql")]

use std::collections::HashMap;

use grafeo_common::types::{PropertyKey, Value};
use grafeo_engine::GrafeoDB;

/// Methods 1 to 5 (Graph:Method, `name` m1 to m5, `tier` 0), method 6 is
/// only a Graph (no Method label), and two Graph:Method nodes share id 7.
fn database(index: bool) -> GrafeoDB {
    let db = GrafeoDB::new_in_memory();
    if index {
        db.create_property_index("id").unwrap();
    }
    db.execute(
        "UNWIND range(1, 5) AS i INSERT (:Graph:Method {id: i, name: 'm' + toString(i), tier: 0})",
    )
    .unwrap();
    db.execute("INSERT (:Graph {id: 6, name: 'm6', tier: 0})")
        .unwrap();
    db.execute("INSERT (:Graph:Method {id: 7, name: 'a7', tier: 0}), (:Graph:Method {id: 7, name: 'b7', tier: 0})")
        .unwrap();
    db
}

fn row(pairs: &[(&str, Value)]) -> HashMap<PropertyKey, Value> {
    pairs
        .iter()
        .map(|(k, v)| (PropertyKey::new(*k), v.clone()))
        .collect()
}

/// `(name, tier)` of every node, by name.
fn state(db: &GrafeoDB) -> Vec<(String, i64)> {
    db.execute("MATCH (n:Graph) RETURN n.name, n.tier ORDER BY n.name")
        .unwrap()
        .rows()
        .iter()
        .map(|r| match (&r[0], &r[1]) {
            (Value::String(name), Value::Int64(tier)) => (name.to_string(), *tier),
            other => panic!("{other:?}"),
        })
        .collect()
}

fn tiers(db: &GrafeoDB) -> HashMap<String, i64> {
    state(db).into_iter().collect()
}

#[test]
fn the_key_matches_by_value_and_the_labels_must_all_be_there() {
    for index in [false, true] {
        let db = database(index);
        let summary = db
            .update_nodes(
                &["Graph", "Method"],
                "id",
                vec![
                    row(&[("id", Value::Float64(2.0)), ("tier", Value::Int64(1))]),
                    row(&[("id", Value::Int64(6)), ("tier", Value::Int64(1))]),
                    row(&[("tier", Value::Int64(1))]),
                    row(&[("id", Value::Int64(99)), ("tier", Value::Int64(1))]),
                    row(&[("id", Value::Int64(7)), ("tier", Value::Int64(1))]),
                ],
            )
            .unwrap();
        assert_eq!(summary.created, 0, "index {index}");
        assert_eq!(summary.updated, 2, "2.0 and the shared 7; index {index}");
        assert_eq!(summary.skipped, 3, "index {index}");
        assert_eq!(
            summary.skipped_rows,
            [1, 2, 3],
            "the Graph-only 6, no key, 99; index {index}"
        );
        let t = tiers(&db);
        assert_eq!(t["m2"], 1, "2.0 matches 2; index {index}");
        assert_eq!(t["m6"], 0, "a Graph without Method; index {index}");
        assert_eq!((t["a7"], t["b7"]), (1, 1), "both nodes of 7; index {index}");
        assert_eq!((t["m1"], t["m3"]), (0, 0), "index {index}");
        assert_eq!(
            db.execute("MATCH (n) RETURN count(n)").unwrap().rows()[0][0],
            Value::Int64(8),
            "no node created; index {index}"
        );
    }
}

/// A row's properties merge into the node: `name` stays when only `tier`
/// is set, and a new property is added.
#[test]
fn the_properties_merge() {
    let db = database(false);
    db.update_nodes(
        &["Graph", "Method"],
        "id",
        vec![row(&[
            ("id", Value::Int64(3)),
            ("tier", Value::Int64(9)),
            ("owner", Value::from("Gus")),
        ])],
    )
    .unwrap();
    assert_eq!(
        db.execute("MATCH (n:Method {id: 3}) RETURN n.name, n.tier, n.owner")
            .unwrap()
            .rows(),
        [[Value::from("m3"), Value::Int64(9), Value::from("Gus")]]
    );
}

/// A row that breaks a UNIQUE constraint fails the call, and no row of it
/// is written.
#[test]
fn a_constraint_violation_writes_nothing() {
    let db = database(false);
    db.execute_cypher("CREATE CONSTRAINT FOR (m:Method) REQUIRE m.name IS UNIQUE")
        .unwrap_or_else(|error| panic!("{error}"));
    let before = state(&db);
    let result = db.update_nodes(
        &["Graph", "Method"],
        "id",
        vec![
            row(&[("id", Value::Int64(1)), ("tier", Value::Int64(5))]),
            row(&[("id", Value::Int64(2)), ("name", Value::from("m3"))]),
        ],
    );
    assert!(result.is_err(), "name m3 is taken");
    assert_eq!(state(&db), before);
}

/// In a session's transaction the update is seen inside and undone by the
/// rollback.
#[test]
fn a_rollback_undoes_it() {
    let db = database(false);
    let mut session = db.session();
    session.begin_transaction().unwrap();
    let summary = session
        .update_nodes(
            &["Graph", "Method"],
            "id",
            vec![row(&[("id", Value::Int64(4)), ("tier", Value::Int64(3))])],
        )
        .unwrap();
    assert_eq!(summary.updated, 1);
    assert_eq!(
        session
            .execute("MATCH (n:Method {id: 4}) RETURN n.tier")
            .unwrap()
            .rows()[0][0],
        Value::Int64(3)
    );
    session.rollback().unwrap();
    assert_eq!(tiers(&db)["m4"], 0);
}

/// Today the text "1" matches the integer 1, as `=` does. The 0.6.1
/// decision on comparing text with numbers changes this: update the test
/// with that change.
#[test]
fn text_one_matches_integer_one_until_the_comparison_change() {
    let db = database(false);
    let summary = db
        .update_nodes(
            &["Graph", "Method"],
            "id",
            vec![row(&[("id", Value::from("1")), ("tier", Value::Int64(2))])],
        )
        .unwrap();
    assert_eq!(summary.updated, 1);
    assert_eq!(tiers(&db)["m1"], 2);
}
