//! Labels and edge types in use (0.6.1): `label_count()`, `edge_type_count()`,
//! `detailed_stats()`, `schema()`, `CALL db.labels()` and
//! `CALL db.relationshipTypes()` count and list only the labels some node has
//! and the edge types some edge has, while the store's dictionary keeps every
//! name it gave an id. Each state is checked through all of them:
//!
//! - the last node of a label deleted, a label removed from its last node,
//!   the last edge of a type deleted, a label given again;
//! - a transaction: its uncommitted deletes are not in the counts (they read
//!   the committed state) but are in its own `CALL db.labels()`; a rollback
//!   and a commit;
//! - a past epoch, where the store reads instead of using its counts;
//! - a persistent database reopened;
//! - a projection (the trait's default methods).
//!
//! ```bash
//! cargo test -p grafeo-engine --all-features --test labels_in_use
//! ```

#![cfg(all(feature = "gql", feature = "cypher"))]

use grafeo_common::types::Value;
use grafeo_engine::admin::SchemaInfo;
use grafeo_engine::{Config, GrafeoDB, GraphStore, ProjectionSpec};

/// Alix, Gus (Person and Admin) and Vincent are Persons, Paris a City and
/// Mia a Robot. KNOWS: Alix to Gus, Gus to Vincent; LIVES_IN: Alix to Paris;
/// BUILT: Mia to Alix.
const GRAPH: &str = "INSERT (alix:Person {name: 'Alix'}), (gus:Person:Admin {name: 'Gus'}), \
     (vincent:Person {name: 'Vincent'}), (paris:City {name: 'Paris'}), \
     (mia:Robot {name: 'Mia'}), \
     (alix)-[:KNOWS]->(gus), (gus)-[:KNOWS]->(vincent), (alix)-[:LIVES_IN]->(paris), \
     (mia)-[:BUILT]->(alix)";

fn database() -> GrafeoDB {
    let db = GrafeoDB::new_in_memory();
    db.execute(GRAPH).unwrap();
    db
}

fn names(rows: &[Vec<Value>]) -> Vec<String> {
    let mut names: Vec<String> = rows
        .iter()
        .map(|row| match &row[0] {
            Value::String(name) => name.to_string(),
            other => panic!("{other:?}"),
        })
        .collect();
    names.sort();
    names
}

/// The labels and edge types in use, sorted, after checking that the
/// counts, the statistics, the schema and the procedures agree.
fn in_use(db: &GrafeoDB) -> (Vec<String>, Vec<String>) {
    let SchemaInfo::Lpg(schema) = db.schema().unwrap() else {
        panic!("an LPG schema");
    };
    let mut labels: Vec<String> = schema.labels.iter().map(|l| l.name.clone()).collect();
    labels.sort();
    let mut types: Vec<String> = schema.edge_types.iter().map(|t| t.name.clone()).collect();
    types.sort();
    assert_eq!(db.label_count().unwrap(), labels.len(), "label_count");
    assert_eq!(
        db.edge_type_count().unwrap(),
        types.len(),
        "edge_type_count"
    );
    let stats = db.detailed_stats().unwrap();
    assert_eq!(stats.label_count, labels.len(), "detailed_stats");
    assert_eq!(stats.edge_type_count, types.len(), "detailed_stats");
    assert_eq!(
        names(db.execute_cypher("CALL db.labels()").unwrap().rows()),
        labels,
        "db.labels()"
    );
    assert_eq!(
        names(
            db.execute_cypher("CALL db.relationshipTypes()")
                .unwrap()
                .rows()
        ),
        types,
        "db.relationshipTypes()"
    );
    (labels, types)
}

fn list(items: &[&str]) -> Vec<String> {
    items.iter().map(|s| (*s).to_string()).collect()
}

#[test]
fn deletes_and_removes_take_names_out_of_use() {
    let db = database();
    assert_eq!(
        in_use(&db),
        (
            list(&["Admin", "City", "Person", "Robot"]),
            list(&["BUILT", "KNOWS", "LIVES_IN"])
        )
    );

    db.execute("MATCH (n {name: 'Paris'}) DETACH DELETE n")
        .unwrap();
    assert_eq!(
        in_use(&db),
        (
            list(&["Admin", "Person", "Robot"]),
            list(&["BUILT", "KNOWS"])
        ),
        "the last City and with it the last LIVES_IN"
    );

    db.execute("MATCH (n {name: 'Gus'}) REMOVE n:Admin")
        .unwrap();
    assert_eq!(
        in_use(&db).0,
        list(&["Person", "Robot"]),
        "Admin removed from its last node"
    );

    db.execute("MATCH ()-[r:BUILT]->() DELETE r").unwrap();
    assert_eq!(
        in_use(&db),
        (list(&["Person", "Robot"]), list(&["KNOWS"])),
        "the last BUILT edge; Mia keeps Robot"
    );

    db.execute("MATCH (n {name: 'Vincent'}) SET n:Admin")
        .unwrap();
    assert_eq!(
        in_use(&db).0,
        list(&["Admin", "Person", "Robot"]),
        "Admin given again"
    );

    // The dictionary keeps every name it gave an id.
    let mut dictionary = db.store().unwrap().all_labels().unwrap();
    dictionary.sort();
    assert_eq!(dictionary, list(&["Admin", "City", "Person", "Robot"]));
    let mut types = db.store().unwrap().all_edge_types().unwrap();
    types.sort();
    assert_eq!(types, list(&["BUILT", "KNOWS", "LIVES_IN"]));
}

/// A transaction's deletes: the counts read the committed state, the
/// transaction's own `db.labels()` its view; a rollback keeps the names, a
/// commit takes them out.
#[test]
fn a_transactions_deletes() {
    let db = database();
    let before = in_use(&db);
    let mut session = db.session();
    session.begin_transaction().unwrap();
    session
        .execute("MATCH (n {name: 'Mia'}) DETACH DELETE n")
        .unwrap();
    assert_eq!(in_use(&db), before, "uncommitted: the counts are as before");
    assert_eq!(
        names(
            session
                .execute_language("CALL db.labels()", "cypher", None)
                .unwrap()
                .rows()
        ),
        list(&["Admin", "City", "Person"]),
        "the transaction's own view"
    );
    session.rollback().unwrap();
    assert_eq!(in_use(&db), before, "rolled back");

    session.begin_transaction().unwrap();
    session
        .execute("MATCH (n {name: 'Mia'}) DETACH DELETE n")
        .unwrap();
    session.commit().unwrap();
    assert_eq!(
        in_use(&db),
        (
            list(&["Admin", "City", "Person"]),
            list(&["KNOWS", "LIVES_IN"])
        ),
        "committed: Robot and BUILT went"
    );
}

/// At a past epoch the store reads the labels and edge types its counts no
/// longer answer for.
#[test]
fn a_past_epoch() {
    let db = GrafeoDB::with_config(
        Config::in_memory().with_history_retention(grafeo_engine::HistoryRetention::Forever),
    )
    .unwrap();
    db.execute(GRAPH).unwrap();
    let before = db.current_epoch();
    db.execute("MATCH (n {name: 'Paris'}) DETACH DELETE n")
        .unwrap();
    let then = db.as_of(before).unwrap().snapshot();
    let mut labels = then.store().labels_in_use(then.view()).unwrap();
    labels.sort();
    assert_eq!(labels, list(&["Admin", "City", "Person", "Robot"]));
    let mut types = then.store().edge_types_in_use(then.view()).unwrap();
    types.sort();
    assert_eq!(types, list(&["BUILT", "KNOWS", "LIVES_IN"]));
    assert_eq!(
        in_use(&db),
        (
            list(&["Admin", "Person", "Robot"]),
            list(&["BUILT", "KNOWS"])
        ),
        "now"
    );
}

#[cfg(feature = "grafeo-file")]
#[test]
fn a_persistent_database_reopened() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("people.grafeo");
    {
        let db = GrafeoDB::open(&path).unwrap();
        db.execute(GRAPH).unwrap();
        db.execute("MATCH (n {name: 'Paris'}) DETACH DELETE n")
            .unwrap();
        db.execute("MATCH (n {name: 'Gus'}) REMOVE n:Admin")
            .unwrap();
        db.close().unwrap();
    }
    let db = GrafeoDB::open(&path).unwrap();
    assert_eq!(
        in_use(&db),
        (list(&["Person", "Robot"]), list(&["BUILT", "KNOWS"]))
    );
    db.close().unwrap();
}

/// A projection of the Persons answers through the trait's default methods:
/// the labels its nodes have, the edge types between them.
#[test]
fn a_projection() {
    let db = database();
    db.create_projection("people", ProjectionSpec::new().with_node_labels(["Person"]))
        .unwrap();
    let view = GrafeoDB::with_read_store(
        db.projection("people").unwrap().shared(),
        Config::in_memory(),
    )
    .unwrap();
    assert_eq!(
        in_use(&view),
        (list(&["Admin", "Person"]), list(&["KNOWS"]))
    );
}
