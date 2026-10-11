//! A Cypher `MERGE ... ON CREATE SET n:Label` or `ON MATCH SET n:Label` adds
//! the label to the merged node (openCypher 9, "MERGE": `ON CREATE` and
//! `ON MATCH` take SET items, labels among them). The label items were dropped
//! while the property items beside them were written, so the MERGE reported
//! success and left the node without the label.
//!
//! An item the MERGE cannot apply (a label for a relationship MERGE, an item
//! on another variable than the merged one, or `n += <map>` with a map that is
//! not a literal) is an error, not a silent no-op or a write to the wrong
//! element.
//!
//! ```bash
//! cargo test -p grafeo-engine --all-features --test merge_set_labels
//! ```

#![cfg(all(feature = "lpg", feature = "gql", feature = "cypher"))]

use grafeo_common::types::Value;
use grafeo_engine::{GrafeoDB, Session};

fn cypher(session: &Session, query: &str) -> Vec<Vec<Value>> {
    session
        .execute_cypher(query)
        .unwrap_or_else(|error| panic!("`{query}` failed: {error}"))
        .rows()
        .to_vec()
}

/// Every node named `name`: its labels (sorted) and its property `p`, sorted.
fn labels_and(session: &Session, name: &str, property: &str) -> Vec<(Vec<String>, Value)> {
    let rows = cypher(
        session,
        &format!("MATCH (n {{name: '{name}'}}) RETURN labels(n), n.{property}"),
    );
    let mut nodes: Vec<(Vec<String>, Value)> = rows
        .into_iter()
        .map(|row| {
            let mut labels: Vec<String> = match &row[0] {
                Value::List(items) => items
                    .iter()
                    .map(|item| match item {
                        Value::String(label) => label.to_string(),
                        other => format!("{other:?}"),
                    })
                    .collect(),
                other => panic!("labels(n) is a list, got {other:?}"),
            };
            labels.sort();
            (labels, row[1].clone())
        })
        .collect();
    nodes.sort_by(|a, b| a.0.cmp(&b.0));
    nodes
}

fn labels(names: &[&str]) -> Vec<String> {
    names.iter().map(|label| (*label).to_string()).collect()
}

#[test]
fn on_create_set_adds_a_label_to_the_created_node() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    cypher(
        &session,
        "MERGE (p:Person {name: 'Alix'}) ON CREATE SET p:New",
    );
    assert_eq!(
        labels_and(&session, "Alix", "since"),
        [(labels(&["New", "Person"]), Value::Null)]
    );
    assert_eq!(
        cypher(&session, "MATCH (n:New) RETURN n.name"),
        [vec![Value::from("Alix")]],
        "a scan of the new label finds the node"
    );
}

#[test]
fn the_merge_returns_the_node_with_the_label_it_added() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    let rows = cypher(
        &session,
        "MERGE (p:Person {name: 'Alix'}) ON CREATE SET p:New RETURN p:New AS added",
    );
    assert_eq!(rows, [vec![Value::Bool(true)]]);
}

#[test]
fn on_create_set_adds_labels_and_properties_together() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    cypher(
        &session,
        "MERGE (p:Person {name: 'Alix'}) ON CREATE SET p:New:Fresh, p.since = 2019",
    );
    assert_eq!(
        labels_and(&session, "Alix", "since"),
        [(labels(&["Fresh", "New", "Person"]), Value::Int64(2019))]
    );
}

#[test]
fn on_match_set_adds_a_label_to_the_matched_node_and_on_create_does_not() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    cypher(&session, "CREATE (:Person {name: 'Alix'})");
    cypher(
        &session,
        "MERGE (p:Person {name: 'Alix'}) ON CREATE SET p:New, p.visits = 0 \
         ON MATCH SET p:Seen, p.visits = 3",
    );
    assert_eq!(
        labels_and(&session, "Alix", "visits"),
        [(labels(&["Person", "Seen"]), Value::Int64(3))]
    );
}

#[test]
fn on_create_set_label_leaves_a_matched_node_alone() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    cypher(&session, "CREATE (:Person {name: 'Alix'})");
    cypher(
        &session,
        "MERGE (p:Person {name: 'Alix'}) ON CREATE SET p:New",
    );
    assert_eq!(
        labels_and(&session, "Alix", "since"),
        [(labels(&["Person"]), Value::Null)]
    );
}

#[test]
fn on_match_set_labels_every_matching_node() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    cypher(
        &session,
        "CREATE (:Person {name: 'Alix', k: 3}), (:Person {name: 'Alix', k: 19})",
    );
    cypher(
        &session,
        "MERGE (p:Person {name: 'Alix'}) ON MATCH SET p:Seen",
    );
    assert_eq!(
        labels_and(&session, "Alix", "k")
            .into_iter()
            .map(|(labels, _)| labels)
            .collect::<Vec<_>>(),
        [labels(&["Person", "Seen"]), labels(&["Person", "Seen"])]
    );
}

#[test]
fn on_create_set_label_per_input_row() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    cypher(&session, "CREATE (:Person {name: 'Alix'})");
    cypher(
        &session,
        "UNWIND ['Alix', 'Gus'] AS name \
         MERGE (p:Person {name: name}) ON CREATE SET p:New ON MATCH SET p:Seen",
    );
    assert_eq!(
        labels_and(&session, "Alix", "k"),
        [(labels(&["Person", "Seen"]), Value::Null)]
    );
    assert_eq!(
        labels_and(&session, "Gus", "k"),
        [(labels(&["New", "Person"]), Value::Null)]
    );
}

#[test]
fn a_rolled_back_merge_leaves_no_label() {
    let db = GrafeoDB::new_in_memory();
    let mut session = db.session();
    cypher(&session, "CREATE (:Person {name: 'Alix'})");
    session.begin_transaction().unwrap();
    cypher(
        &session,
        "MERGE (p:Person {name: 'Alix'}) ON MATCH SET p:Seen",
    );
    session.rollback().unwrap();
    assert_eq!(
        labels_and(&session, "Alix", "k"),
        [(labels(&["Person"]), Value::Null)]
    );
}

#[test]
fn an_item_the_merge_cannot_apply_is_an_error_that_writes_nothing() {
    for query in [
        // A relationship has no labels, and the MERGE writes the relationship.
        "MERGE (a:Person {name: 'Alix'})-[r:KNOWS]->(b:Person {name: 'Gus'}) ON CREATE SET a:New",
        // An item on another variable than the merged one.
        "MERGE (a:Person {name: 'Alix'})-[r:KNOWS]->(b:Person {name: 'Gus'}) ON CREATE SET a.since = 3",
        "MATCH (g:Person {name: 'Gus'}) MERGE (p:Person {name: 'Alix'}) ON CREATE SET g:Seen",
        "MATCH (g:Person {name: 'Gus'}) MERGE (p:Person {name: 'Alix'}) ON MATCH SET g.seen = true",
        // A map that is not a literal (it was dropped).
        "MATCH (g:Person {name: 'Gus'}) MERGE (p:Person {name: 'Alix'}) ON CREATE SET p += properties(g)",
    ] {
        let db = GrafeoDB::new_in_memory();
        let session = db.session();
        cypher(&session, "CREATE (:Person {name: 'Gus'})");
        let error = session
            .execute_cypher(query)
            .expect_err("the item cannot be applied");
        assert!(
            error.to_string().contains("MERGE"),
            "{query}: the error names the MERGE: {error}"
        );
        assert_eq!(
            cypher(&session, "MATCH (n) RETURN count(n)"),
            [vec![Value::Int64(1)]],
            "{query}: nothing was written"
        );
    }
}
