//! A query result reports what its writes changed: nodes and edges created
//! and deleted, properties set, labels added and removed.

#![cfg(all(feature = "lpg", feature = "gql"))]

use std::collections::HashMap;

use grafeo_common::types::Value;
use grafeo_engine::GrafeoDB;
use grafeo_engine::database::WriteCounters;

fn counters(db: &GrafeoDB, query: &str) -> WriteCounters {
    db.execute(query).unwrap().counters
}

#[test]
fn inserts_count_nodes_edges_labels_and_properties() {
    let db = GrafeoDB::new_in_memory();
    let c = counters(
        &db,
        "INSERT (:Person:Employee {name: 'Alix', age: 30})-[:KNOWS {since: 2020}]->(:Person {name: 'Gus'})",
    );
    assert_eq!(
        (
            c.nodes_created,
            c.nodes_deleted,
            c.edges_created,
            c.edges_deleted,
            c.properties_set,
            c.labels_added,
            c.labels_removed,
        ),
        (2, 0, 1, 0, 4, 3, 0),
        "nodes, edges, properties and labels created; nothing deleted or removed"
    );
}

#[test]
fn set_remove_and_delete_are_counted() {
    let db = GrafeoDB::new_in_memory();
    db.execute("INSERT (:Person {name: 'Alix'})-[:KNOWS]->(:Person {name: 'Gus'})")
        .unwrap();
    let alix = "MATCH (n:Person {name: 'Alix'})";

    let c = counters(&db, &format!("{alix} SET n.city = 'Paris', n.age = 31"));
    assert_eq!(c.properties_set, 2);
    assert_eq!(
        counters(&db, &format!("{alix} SET n:Manager")).labels_added,
        1
    );
    assert!(
        !counters(&db, &format!("{alix} SET n:Manager")).contains_updates(),
        "a label the node has is not added again"
    );
    assert_eq!(
        counters(&db, &format!("{alix} REMOVE n.city")).properties_set,
        1
    );
    assert!(
        !counters(&db, &format!("{alix} REMOVE n.missing")).contains_updates(),
        "removing an absent property changes nothing"
    );
    assert_eq!(
        counters(&db, &format!("{alix} REMOVE n:Manager")).labels_removed,
        1
    );
    let c = counters(&db, &format!("{alix} DETACH DELETE n"));
    assert_eq!((c.nodes_deleted, c.edges_deleted), (1, 1));
}

#[test]
fn merge_counts_only_what_it_creates() {
    let db = GrafeoDB::new_in_memory();
    let c = counters(&db, "MERGE (:City {name: 'Paris'})");
    assert_eq!(
        (c.nodes_created, c.labels_added, c.properties_set),
        (1, 1, 1)
    );
    assert!(!counters(&db, "MERGE (:City {name: 'Paris'})").contains_updates());

    // A key repeated within one statement creates one node; the second row
    // matches it and sets its property.
    let rows = Value::List(
        ["Berlin", "Prague", "Berlin"]
            .iter()
            .map(|name| {
                Value::Map(
                    [("name".into(), Value::from(*name))]
                        .into_iter()
                        .collect::<std::collections::BTreeMap<_, _>>()
                        .into(),
                )
            })
            .collect::<Vec<_>>()
            .into(),
    );
    let c = db
        .execute_with_params(
            "UNWIND $rows AS row MERGE (c:City {name: row.name}) SET c.seen = true",
            HashMap::from([("rows".to_string(), rows)]),
        )
        .unwrap()
        .counters;
    assert_eq!(c.nodes_created, 2);
    assert_eq!(
        db.execute("MATCH (c:City) RETURN count(c)").unwrap().rows()[0][0],
        Value::Int64(3)
    );
}

/// A label written twice is one label, counted once.
#[test]
fn a_repeated_label_counts_once() {
    let db = GrafeoDB::new_in_memory();
    let c = counters(&db, "INSERT (:City:City {name: 'Paris'})");
    assert_eq!((c.nodes_created, c.labels_added), (1, 1));
    assert_eq!(
        db.execute("MATCH (n:City) RETURN labels(n)")
            .unwrap()
            .rows()[0][0],
        Value::List(vec![Value::from("City")].into())
    );
}

/// Writes inside a stored procedure count for the statement that calls it.
#[cfg(feature = "algos")]
#[test]
fn a_procedure_counts_its_writes() {
    let db = GrafeoDB::new_in_memory();
    db.execute(
        "CREATE PROCEDURE add_city(name STRING) RETURNS (n INTEGER) AS { \
         INSERT (c:City {name: $name}) RETURN 1 AS n }",
    )
    .unwrap();
    let c = counters(&db, "CALL add_city('Paris') YIELD n RETURN n");
    assert_eq!(
        (c.nodes_created, c.labels_added, c.properties_set),
        (1, 1, 1)
    );
}

#[test]
fn reads_and_failed_checks_report_nothing() {
    let db = GrafeoDB::new_in_memory();
    db.execute("INSERT (:Person {name: 'Alix'})").unwrap();
    assert!(!counters(&db, "MATCH (n) RETURN n.name").contains_updates());
    assert_eq!(
        counters(&db, "INSERT (:Person {name: 'Gus'})").nodes_created,
        1,
        "counts are per statement, not cumulative"
    );
}

#[cfg(feature = "cypher")]
#[test]
fn cypher_reports_counters_too() {
    let db = GrafeoDB::new_in_memory();
    let c = db
        .execute_cypher("CREATE (a:Person {name: 'Alix'})-[:KNOWS]->(b:Person)")
        .unwrap()
        .counters;
    assert_eq!(
        (c.nodes_created, c.edges_created, c.labels_added),
        (2, 1, 2)
    );
}

/// A null write removes the property: it is gone from `keys` and
/// `properties`, and counts as a change only when the property was there.
#[test]
fn a_null_write_removes_the_property() {
    let db = GrafeoDB::new_in_memory();
    let c = counters(&db, "INSERT (:A {x: 1, y: 2, z: null})");
    assert_eq!(c.properties_set, 2, "the null is not written");
    assert_eq!(counters(&db, "MATCH (n:A) REMOVE n.x").properties_set, 1);
    assert_eq!(
        counters(&db, "MATCH (n:A) SET n.y = NULL").properties_set,
        1
    );
    assert!(!counters(&db, "MATCH (n:A) SET n.w = NULL").contains_updates());
    let row = db
        .execute("MATCH (n:A) RETURN keys(n), properties(n)")
        .unwrap()
        .rows()[0]
        .clone();
    assert_eq!(
        row,
        [
            Value::List(Vec::new().into()),
            Value::Map(std::collections::BTreeMap::new().into())
        ]
    );
}
