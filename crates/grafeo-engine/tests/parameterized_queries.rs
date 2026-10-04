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

/// A write that uses a parameter nobody supplied fails before it writes:
/// it used to store the text "$e". Through `execute` (no parameter map at
/// all), an empty map, Cypher and GraphQL (a declared variable without a
/// default).
#[test]
fn an_unsupplied_parameter_fails_before_writing() {
    let db = GrafeoDB::new_in_memory();
    for result in [
        db.execute("INSERT (:P {e: $e})"),
        db.execute_with_params("INSERT (:P {e: $e})", HashMap::new()),
        #[cfg(feature = "cypher")]
        db.execute_cypher("CREATE (:P {e: $e})"),
        #[cfg(feature = "graphql")]
        db.execute_graphql("mutation ($e: String) { createP(e: $e) { e } }"),
    ] {
        let error = result.unwrap_err().to_string();
        assert!(error.contains("Missing parameter: $e"), "{error}");
    }
    assert_eq!(count(&db, "P"), Value::Int64(0));

    // EXPLAIN shows the plan without the values; PROFILE runs it, so it fails.
    db.execute("EXPLAIN INSERT (:P {e: $e})").unwrap();
    let error = db
        .execute("PROFILE INSERT (:P {e: $e})")
        .unwrap_err()
        .to_string();
    assert!(error.contains("Missing parameter: $e"), "{error}");
    assert_eq!(count(&db, "P"), Value::Int64(0));
}

/// Gremlin without a parameter map fails the same way where it reads a
/// parameter (`has`); it used to reach the planner with the parameter unset.
#[cfg(feature = "gremlin")]
#[test]
fn an_unsupplied_gremlin_parameter_is_missing() {
    let db = GrafeoDB::new_in_memory();
    db.execute("INSERT (:Person {name: 'Alix'})").unwrap();
    let error = db
        .execute_gremlin("g.V().has('name', $name).values('name')")
        .unwrap_err()
        .to_string();
    assert!(error.contains("Missing parameter: $name"), "{error}");
    let result = db
        .execute_gremlin_with_params(
            "g.V().has('name', $name).values('name')",
            params(&[("name", Value::from("Alix"))]),
        )
        .unwrap();
    assert_eq!(result.rows(), [vec![Value::from("Alix")]]);
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

/// An empty map fills in nothing: a statement without parameters reuses its
/// optimized plan like the same call without a map, and one that names a
/// parameter still fails as missing it.
#[test]
fn an_empty_parameter_map_uses_the_cached_plan() {
    let db = GrafeoDB::new_in_memory();
    db.execute("INSERT (:Person {name: 'Alix'})").unwrap();
    let query = "MATCH (p:Person) RETURN p.name";
    let hits = || db.query_cache().stats().optimized_hits;
    let before = hits();
    for _ in 0..3 {
        let rows = db
            .execute_with_params(query, HashMap::new())
            .unwrap()
            .rows()
            .to_vec();
        assert_eq!(rows, [vec![Value::from("Alix")]]);
    }
    assert_eq!(
        hits() - before,
        2,
        "the second and third call reuse the plan"
    );

    let error = db
        .execute_with_params(
            "MATCH (p:Person) WHERE p.name = $name RETURN p.name",
            HashMap::new(),
        )
        .unwrap_err();
    assert!(error.to_string().contains("$name"), "{error}");
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

/// Dotted access reads keys of a map parameter, in GQL and Cypher.
#[test]
fn dotted_access_reads_a_map_parameter() {
    use std::collections::BTreeMap;
    use std::sync::Arc;

    use grafeo_common::types::PropertyKey;

    let db = GrafeoDB::new_in_memory();
    let route = Value::Map(Arc::new(BTreeMap::from([(
        PropertyKey::new("to"),
        Value::from("Prague"),
    )])));
    let meta = Value::Map(Arc::new(BTreeMap::from([(
        PropertyKey::new("route"),
        route,
    )])));
    let result = db
        .execute_with_params(
            "RETURN $meta.route.to AS to",
            params(&[("meta", meta.clone())]),
        )
        .unwrap();
    assert_eq!(result.rows(), [[Value::from("Prague")]]);
    #[cfg(feature = "cypher")]
    {
        let result = db
            .session()
            .execute_cypher_with_params("RETURN $meta.route.to AS to", params(&[("meta", meta)]))
            .unwrap();
        assert_eq!(result.rows(), [[Value::from("Prague")]]);
    }
}

/// Dotted access on a node that a function returns is an error that says
/// what to do, never a silent null.
#[test]
fn dotted_access_on_a_node_expression_explains_itself() {
    let db = GrafeoDB::new_in_memory();
    db.execute("INSERT (:City {name: 'Amsterdam'})-[:ROAD]->(:City {name: 'Berlin'})")
        .unwrap();
    let err = db
        .execute("MATCH (:City)-[r:ROAD]->(:City) RETURN startNode(r).name")
        .unwrap_err()
        .to_string();
    assert!(err.contains("startNode(r) is not a map value"), "{err}");
    assert!(err.contains("bound to a variable in the pattern"), "{err}");
}
