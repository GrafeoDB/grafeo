//! A parameterized statement is checked, tracked and planned exactly like the
//! same statement with literal values (#526): schema and constraints, write
//! conflicts per graph, EXPLAIN and PROFILE (#460), and a cached plan never
//! keeps one call's values.

#![cfg(all(feature = "lpg", feature = "gql"))]

use std::collections::HashMap;

use grafeo_common::types::Value;
use grafeo_engine::{Config, GrafeoDB};

fn params(pairs: &[(&str, Value)]) -> HashMap<String, Value> {
    pairs
        .iter()
        .map(|(name, value)| ((*name).to_string(), value.clone()))
        .collect()
}

fn count(db: &GrafeoDB, label: &str) -> Value {
    db.execute(&format!("MATCH (n:{label}) RETURN count(n)"))
        .unwrap()
        .rows()[0][0]
        .clone()
}

#[test]
fn constraints_hold_for_parameterized_writes() {
    let db = GrafeoDB::new_in_memory();
    db.execute("CREATE CONSTRAINT person_email FOR (n:Person) ON (n.email) UNIQUE")
        .unwrap();
    db.execute("CREATE CONSTRAINT person_name FOR (n:Person) ON (n.name) NOT NULL")
        .unwrap();
    db.execute("INSERT (:Person {name: 'Alix', email: 'alix@example.org'})")
        .unwrap();
    let email = |value: &str| params(&[("email", Value::from(value))]);

    let err = db
        .execute_with_params(
            "INSERT (:Person {name: 'Gus', email: $email})",
            email("alix@example.org"),
        )
        .unwrap_err();
    assert!(err.to_string().contains("UNIQUE"), "{err}");
    let err = db
        .execute_with_params("INSERT (:Person {email: $email})", email("gus@example.org"))
        .unwrap_err();
    assert!(err.to_string().contains("name"), "{err}");
    db.execute_with_params(
        "INSERT (:Person {name: 'Gus', email: $email})",
        email("gus@example.org"),
    )
    .unwrap();
    let err = db
        .execute_with_params(
            "MATCH (n:Person {name: 'Gus'}) SET n.email = $email",
            email("alix@example.org"),
        )
        .unwrap_err();
    assert!(err.to_string().contains("UNIQUE"), "{err}");
    assert_eq!(count(&db, "Person"), Value::Int64(2));
}

#[test]
fn the_size_limit_holds_for_parameterized_writes() {
    let db = GrafeoDB::with_config(Config::in_memory().with_max_property_size(64)).unwrap();
    db.execute("INSERT (:Person {bio: 'short'})").unwrap();
    let err = db
        .execute_with_params(
            "MATCH (n:Person) SET n.bio = $bio",
            params(&[("bio", Value::from("x".repeat(1000)))]),
        )
        .unwrap_err();
    assert!(err.to_string().contains("exceeds maximum size"), "{err}");
}

#[cfg(feature = "cypher")]
#[test]
fn cypher_parameters_are_checked_too() {
    let db = GrafeoDB::new_in_memory();
    db.execute("CREATE CONSTRAINT person_email FOR (n:Person) ON (n.email) UNIQUE")
        .unwrap();
    db.execute("INSERT (:Person {email: 'alix@example.org'})")
        .unwrap();
    let err = db
        .execute_cypher_with_params(
            "CREATE (:Person {email: $email})",
            params(&[("email", Value::from("alix@example.org"))]),
        )
        .unwrap_err();
    assert!(err.to_string().contains("UNIQUE"), "{err}");
    assert_eq!(count(&db, "Person"), Value::Int64(1));
}

/// A parameterized write in a named graph is recorded as a write to that
/// graph: it conflicts with another transaction's write to the same node, and
/// not with the node of the same number in another graph.
#[test]
fn parameterized_writes_are_tracked_per_graph() {
    let db = GrafeoDB::new_in_memory();
    db.execute("CREATE GRAPH extraction").unwrap();
    db.execute("CREATE GRAPH model").unwrap();
    for graph in ["extraction", "model"] {
        db.graph(graph)
            .unwrap()
            .execute("INSERT (:Doc {id: 'd1'})")
            .unwrap();
    }
    let set = "MATCH (n:Doc {id: $id}) SET n.v = $v";
    let args = |v: i64| params(&[("id", Value::from("d1")), ("v", Value::Int64(v))]);

    let mut first = db.graph("model").unwrap().session().unwrap();
    first.begin_transaction().unwrap();
    first
        .execute("MATCH (n:Doc {id: 'd1'}) SET n.v = 1")
        .unwrap();

    let mut same_graph = db.graph("model").unwrap().session().unwrap();
    same_graph.begin_transaction().unwrap();
    assert!(
        same_graph.execute_with_params(set, args(2)).is_err(),
        "the same node in the same graph conflicts"
    );
    same_graph.rollback().unwrap();

    let mut other_graph = db.graph("extraction").unwrap().session().unwrap();
    other_graph.begin_transaction().unwrap();
    other_graph.execute_with_params(set, args(3)).unwrap();
    other_graph.commit().unwrap();
    first.commit().unwrap();
}

#[test]
fn a_cached_plan_does_not_keep_the_values() {
    let db = GrafeoDB::new_in_memory();
    db.create_property_index("id");
    db.execute("UNWIND range(0, 9) AS i INSERT (:Doc {id: 'd' + toString(i), n: i})")
        .unwrap();
    let query = "MATCH (n:Doc {id: $id}) RETURN n.n";
    for (id, n) in [("d3", 3), ("d7", 7), ("d3", 3), ("missing", -1)] {
        let rows = db
            .execute_with_params(query, params(&[("id", Value::from(id))]))
            .unwrap()
            .rows()
            .to_vec();
        if n < 0 {
            assert!(rows.is_empty(), "{id}: {rows:?}");
        } else {
            assert_eq!(rows, [vec![Value::Int64(n)]], "{id}");
        }
    }
}

#[test]
fn explain_and_profile_take_parameters() {
    let db = GrafeoDB::new_in_memory();
    db.execute("INSERT (:Person {name: 'Alix'})").unwrap();
    let name = || params(&[("name", Value::from("Alix"))]);

    let profile = db
        .execute_with_params(
            "PROFILE MATCH (p:Person) WHERE p.name = $name RETURN p.name",
            name(),
        )
        .unwrap();
    assert_eq!(profile.columns, ["profile"]);
    let explain = db
        .execute_with_params(
            "EXPLAIN MATCH (p:Person) WHERE p.name = $name RETURN p.name",
            name(),
        )
        .unwrap();
    assert_ne!(explain.columns, ["p.name"], "{:?}", explain.columns);
    #[cfg(feature = "cypher")]
    {
        let profile = db
            .execute_cypher_with_params(
                "PROFILE MATCH (p:Person) WHERE p.name = $name RETURN p.name",
                name(),
            )
            .unwrap();
        assert_eq!(profile.columns, ["profile"]);
    }
}
