//! Schema DDL whose effect is checked after the statement: a failed second
//! index of one name leaves the first; Cypher's `ALTER CURRENT GRAPH TYPE
//! SET` with an edge pattern declares its node types in the current graph's
//! type; a projection reports the labels its nodes carry. The statement forms
//! are in `tests/spec/lpg/gql/index_and_graph_type_ddl.gtest`.
//!
//! ```bash
//! cargo test -p grafeo-engine --all-features --test schema_ddl_followups
//! ```

#![cfg(feature = "gql")]

use grafeo_common::types::Value;
use grafeo_engine::{GrafeoDB, ProjectionSpec};

/// The rows of `query`, each as its cells' debug text.
fn rows(db: &GrafeoDB, query: &str) -> Vec<Vec<String>> {
    db.execute(query)
        .unwrap_or_else(|error| panic!("{query}: {error}"))
        .rows()
        .iter()
        .map(|row| row.iter().map(|value| format!("{value:?}")).collect())
        .collect()
}

/// A second `CREATE INDEX` of a name fails, naming it, and the first index
/// of the name stays, the only one.
#[test]
fn a_failed_second_index_of_one_name_leaves_the_first() {
    let db = GrafeoDB::new_in_memory();
    db.execute("CREATE INDEX person_idx FOR (p:Person) ON (p.name)")
        .unwrap();
    let error = db
        .execute("CREATE INDEX person_idx FOR (p:Person) ON (p.age)")
        .expect_err("a second index of the name")
        .to_string();
    assert!(error.contains("person_idx"), "{error}");
    assert_eq!(
        db.execute("SHOW INDEXES").unwrap().rows().to_vec(),
        vec![vec![
            Value::from("person_idx"),
            Value::from("Hash"),
            Value::from("Person"),
            Value::from("name"),
        ]]
    );
    // DROP INDEX of the name leaves no index behind.
    db.execute("DROP INDEX person_idx").unwrap();
    assert_eq!(
        db.execute("SHOW INDEXES").unwrap().rows().to_vec(),
        Vec::<Vec<Value>>::new()
    );
}

/// `ALTER CURRENT GRAPH TYPE SET { (:Stop {name STRING})-[:LEG]->(:Halt) }`
/// in Cypher declares Stop (with its property), Halt and LEG in the current
/// graph's type; it made a graph type named `__current__` instead.
#[cfg(feature = "cypher")]
#[test]
#[ignore = "Cypher ALTER CURRENT GRAPH TYPE SET with a pattern makes a graph type named __current__ instead of declaring the pattern's types"]
fn cypher_alter_current_graph_type_declares_the_pattern_types() {
    let db = GrafeoDB::new_in_memory();
    for statement in [
        "CREATE NODE TYPE Depot",
        "CREATE GRAPH TYPE transit (NODE TYPE Depot)",
        "CREATE GRAPH routes TYPED transit",
        "USE GRAPH routes",
    ] {
        db.execute(statement)
            .unwrap_or_else(|error| panic!("{statement}: {error}"));
    }
    db.execute_cypher("ALTER CURRENT GRAPH TYPE SET { (:Stop {name STRING})-[:LEG]->(:Halt) }")
        .unwrap();

    let mut failures = Vec::new();
    let graph_types = rows(&db, "SHOW GRAPH TYPES");
    if graph_types.iter().any(|row| row[0].contains("__current__")) {
        failures.push(format!("a graph type named __current__: {graph_types:?}"));
    }
    let node_types = rows(&db, "SHOW NODE TYPES");
    for (name, properties) in [("Stop", "name STRING"), ("Halt", "")] {
        if !node_types.iter().any(|row| {
            row[0] == format!("String({name:?})") && row[1] == format!("String({properties:?})")
        }) {
            failures.push(format!(
                "no node type {name} ({properties}): {node_types:?}"
            ));
        }
    }
    let edge_types = rows(&db, "SHOW EDGE TYPES");
    if !edge_types.iter().any(|row| row[0] == "String(\"LEG\")") {
        failures.push(format!("no edge type LEG: {edge_types:?}"));
    }
    // The current graph's type holds them: a Stop with a name of another
    // type is refused, a well-formed one is written.
    if db.execute("INSERT (:Stop {name: 7})").is_ok() {
        failures.push("a Stop with an integer name was written".to_string());
    }
    if let Err(error) = db.execute("INSERT (:Stop {name: 'Centraal'})-[:LEG]->(:Halt)") {
        failures.push(format!("a well-formed Stop and LEG were refused: {error}"));
    }
    assert!(failures.is_empty(), "\n{}", failures.join("\n"));
}

/// A projection's labels are the labels its nodes carry: an `Admin` label on
/// projected `Person` nodes is one, a spec label no node has is not.
#[test]
#[ignore = "a projection reports the labels of its spec (Person, Robot), not the labels its nodes carry (Admin, Person)"]
fn a_projection_reports_the_labels_of_its_nodes() {
    let db = GrafeoDB::new_in_memory();
    db.execute(
        "INSERT (:Person:Admin {name: 'Alix'}), (:Person {name: 'Gus'}), (:City {name: 'Paris'})",
    )
    .unwrap();
    db.create_projection(
        "people",
        ProjectionSpec::new().with_node_labels(["Person", "Robot"]),
    )
    .unwrap();
    let projection = db.projection("people").unwrap();
    let mut labels = projection.store().all_labels().unwrap();
    labels.sort();
    assert_eq!(labels, ["Admin", "Person"]);
}
