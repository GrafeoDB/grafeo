//! A node keeps each of its labels through `compact()`.
//!
//! Every label of a node reads it: `labels(n)`, a label scan, a label count,
//! a pattern with several labels (whose scan reads the label with the fewest
//! nodes, which the planner counts per label), a traversal, and a write or
//! delete through any of them; after `recompact()` and after a close and
//! reopen too.
//!
//! ```bash
//! cargo test -p grafeo-engine --all-features --test compact_labels
//! ```

#![cfg(all(feature = "compact-store", feature = "lpg", feature = "gql"))]

use grafeo_common::types::Value;
use grafeo_engine::GrafeoDB;

/// Alix is a Person and an Employee, Gus a Person, Mia and Jules Employees,
/// and three Vincents (in Amsterdam, Berlin and Paris) are Persons and
/// Managers; Alix works with Mia. So a Person scan holds five nodes and an
/// Employee scan three, while without the nodes of several labels a Person
/// scan would hold one and an Employee scan two.
fn populate(db: &GrafeoDB) {
    db.execute(
        "INSERT (:Person:Employee {name: 'Alix'}), (:Person {name: 'Gus'}), \
         (:Employee {name: 'Mia'}), (:Employee {name: 'Jules'})",
    )
    .unwrap();
    db.execute(
        "UNWIND ['Amsterdam', 'Berlin', 'Paris'] AS city \
         INSERT (:Person:Manager {name: 'Vincent', city: city})",
    )
    .unwrap();
    db.execute(
        "MATCH (a {name: 'Alix'}), (m {name: 'Mia'}) INSERT (a)-[:WORKS_WITH {since: 2019}]->(m)",
    )
    .unwrap();
}

/// The first column of each row of `query`, as strings, sorted.
fn sorted_strings(db: &GrafeoDB, query: &str) -> Vec<String> {
    let mut values: Vec<String> = db
        .execute(query)
        .unwrap()
        .rows()
        .iter()
        .map(|row| match &row[0] {
            Value::String(text) => text.to_string(),
            other => format!("{other:?}"),
        })
        .collect();
    values.sort();
    values
}

/// The single value `query` returns.
fn scalar(db: &GrafeoDB, query: &str) -> Value {
    let rows = db.execute(query).unwrap().rows().to_vec();
    assert_eq!(rows.len(), 1, "{query}: {rows:?}");
    rows[0][0].clone()
}

/// The labels `labels(n)` returns for the node named `name`, sorted.
fn labels_of(db: &GrafeoDB, name: &str) -> Vec<String> {
    let Value::List(labels) = scalar(
        db,
        &format!("MATCH (n {{name: '{name}'}}) RETURN labels(n) AS labels"),
    ) else {
        panic!("labels(n) is a list");
    };
    let mut labels: Vec<String> = labels
        .iter()
        .map(|label| match label {
            Value::String(text) => text.to_string(),
            other => format!("{other:?}"),
        })
        .collect();
    labels.sort();
    labels
}

/// The node scans of the plan of `query`.
fn scans(db: &GrafeoDB, query: &str) -> Vec<String> {
    db.execute(&format!("EXPLAIN {query}"))
        .unwrap()
        .rows()
        .iter()
        .filter_map(|row| match &row[0] {
            Value::String(text) => Some(text.to_string()),
            _ => None,
        })
        .flat_map(|text| {
            text.lines()
                .map(str::trim)
                .filter(|line| line.starts_with("NodeScan"))
                .map(ToString::to_string)
                .collect::<Vec<_>>()
        })
        .collect()
}

/// Checks that every read finds each node under each of its labels.
fn assert_each_label_reads_its_nodes(db: &GrafeoDB, stage: &str) {
    assert_eq!(labels_of(db, "Alix"), ["Employee", "Person"], "{stage}");
    assert_eq!(
        sorted_strings(db, "MATCH (n:Person) RETURN n.name"),
        ["Alix", "Gus", "Vincent", "Vincent", "Vincent"],
        "{stage}: a Person scan"
    );
    assert_eq!(
        sorted_strings(db, "MATCH (n:Employee) RETURN n.name"),
        ["Alix", "Jules", "Mia"],
        "{stage}: an Employee scan"
    );
    assert_eq!(
        sorted_strings(db, "MATCH (n:Manager) RETURN n.city"),
        ["Amsterdam", "Berlin", "Paris"],
        "{stage}: a Manager scan"
    );
    assert_eq!(
        sorted_strings(db, "MATCH (n:Person:Employee) RETURN n.name"),
        ["Alix"],
        "{stage}: a pattern with two labels"
    );
    assert_eq!(
        scalar(db, "MATCH (n:Person) RETURN count(n)"),
        Value::Int64(5),
        "{stage}: a label count"
    );
    assert_eq!(
        sorted_strings(
            db,
            "MATCH (a:Person)-[:WORKS_WITH]->(b:Employee) RETURN a.name + ' ' + b.name"
        ),
        ["Alix Mia"],
        "{stage}: a traversal between labels"
    );
    // Employee has fewer nodes than Person, counted per label.
    assert_eq!(
        scans(db, "MATCH (n:Person:Employee) RETURN n.name"),
        ["NodeScan (n:Employee)"],
        "{stage}: the scan reads the label with the fewest nodes"
    );
}

#[test]
fn each_label_of_a_compacted_node_reads_it() {
    let mut db = GrafeoDB::new_in_memory();
    populate(&db);
    assert_each_label_reads_its_nodes(&db, "before compact()");
    db.compact().unwrap();
    assert_each_label_reads_its_nodes(&db, "after compact()");
    db.compact().unwrap();
    assert_each_label_reads_its_nodes(&db, "after recompact()");
}

/// A write or delete that matches a compacted node through one of its
/// labels finds it (a `DETACH DELETE` through such a match did nothing).
#[test]
fn a_write_through_one_label_reaches_a_compacted_node() {
    let mut db = GrafeoDB::new_in_memory();
    populate(&db);
    db.compact().unwrap();

    db.execute("MATCH (n:Manager {city: 'Berlin'}) SET n.city = 'Prague'")
        .unwrap();
    assert_eq!(
        sorted_strings(&db, "MATCH (n:Person:Manager) RETURN n.city"),
        ["Amsterdam", "Paris", "Prague"],
        "the Berlin Vincent moved"
    );

    db.execute("MATCH (n:Person {name: 'Alix'}) DETACH DELETE n")
        .unwrap();
    assert_eq!(
        sorted_strings(&db, "MATCH (n:Employee) RETURN n.name"),
        ["Jules", "Mia"],
        "Alix is gone from her other label too"
    );
    assert_eq!(
        sorted_strings(&db, "MATCH (n:Person) RETURN n.name"),
        ["Gus", "Vincent", "Vincent", "Vincent"]
    );
    assert_eq!(
        scalar(&db, "MATCH ()-[w:WORKS_WITH]->() RETURN count(w)"),
        Value::Int64(0),
        "her edge went with her"
    );
}

/// The direct API reads each label too.
#[test]
fn the_direct_api_reads_each_label_of_a_compacted_node() {
    let mut db = GrafeoDB::new_in_memory();
    let alix = db.create_node(&["Person", "Employee"]).unwrap();
    let gus = db.create_node(&["Person"]).unwrap();
    db.compact().unwrap();
    let mut labels: Vec<String> = db
        .get_node(alix)
        .expect("Alix")
        .labels
        .iter()
        .map(ToString::to_string)
        .collect();
    labels.sort();
    assert_eq!(labels, ["Employee", "Person"]);
    let mut people = db.graph_store().nodes_by_label("Person");
    people.sort_unstable();
    assert_eq!(people, [alix, gus]);
    assert_eq!(db.graph_store().nodes_by_label("Employee"), [alix]);
}

/// A file keeps each label of a node through `compact()`, a close and
/// reopen, `recompact()`, and the reopened database writing the file again.
#[cfg(all(feature = "grafeo-file", feature = "wal"))]
#[test]
fn a_reopened_compacted_file_keeps_each_label() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("labels.grafeo");
    {
        let mut db = GrafeoDB::open(&path).unwrap();
        populate(&db);
        db.compact().unwrap();
        db.close().unwrap();
    }
    {
        let mut db = GrafeoDB::open(&path).unwrap();
        assert_each_label_reads_its_nodes(&db, "after a reopen");
        db.compact().unwrap();
        assert_each_label_reads_its_nodes(&db, "after a reopen and a merge");
        db.close().unwrap();
    }
    let db = GrafeoDB::open(&path).unwrap();
    assert_each_label_reads_its_nodes(&db, "after a second reopen");
    db.close().unwrap();
}

/// The single count `query` returns in GQL, after checking that Cypher
/// returns the same.
#[cfg(all(feature = "cypher", feature = "grafeo-file", feature = "wal"))]
fn count_in_both(db: &GrafeoDB, query: &str) -> Value {
    let gql = scalar(db, query);
    let cypher = db
        .execute_cypher(query)
        .unwrap_or_else(|error| panic!("Cypher {query}: {error}"))
        .rows()
        .to_vec();
    assert_eq!(
        cypher,
        [vec![gql.clone()]],
        "{query}: GQL and Cypher return the same count"
    );
    gql
}

/// The queries of #595 on Alix (a Graph and a Repository, and `alix`
/// besides) and Gus (a Graph), in GQL and Cypher: each label, both labels
/// and a label no node has.
#[cfg(all(feature = "cypher", feature = "grafeo-file", feature = "wal"))]
fn assert_the_queries_of_the_issue(db: &GrafeoDB, stage: &str, alix: &[&str]) {
    for (query, expected) in [
        ("MATCH (n:Graph) RETURN count(n)", 2),
        ("MATCH (n:Repository) RETURN count(n)", 1),
        ("MATCH (n:Graph:Repository) RETURN count(n)", 1),
        ("MATCH (n:Missing) RETURN count(n)", 0),
        ("MATCH (n:Graph:Missing) RETURN count(n)", 0),
    ] {
        assert_eq!(
            count_in_both(db, query),
            Value::Int64(expected),
            "{stage}: {query}"
        );
    }
    let query = "MATCH (n {name: 'Alix'}) RETURN labels(n) AS labels";
    let cypher = db.execute_cypher(query).unwrap().rows().to_vec();
    assert_eq!(
        cypher,
        db.execute(query).unwrap().rows(),
        "{stage}: GQL and Cypher return the same labels"
    );
    assert_eq!(labels_of(db, "Alix"), alix, "{stage}: labels(n)");
}

/// The statements of #595 on a file this build compacts: Alix matches each
/// of her labels and both, also after a write through one of them, another
/// `compact()`, and a close and reopen.
#[cfg(all(feature = "cypher", feature = "grafeo-file", feature = "wal"))]
#[test]
fn the_queries_of_the_issue_hold_in_gql_and_cypher() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("issue.grafeo");
    {
        let mut db = GrafeoDB::open(&path).unwrap();
        db.execute("INSERT (:Graph:Repository {name: 'Alix'})")
            .unwrap();
        db.execute("INSERT (:Graph {name: 'Gus'})").unwrap();
        db.compact().unwrap();
        assert_the_queries_of_the_issue(&db, "after compact()", &["Graph", "Repository"]);

        db.execute_cypher("MATCH (n:Repository) SET n:Starred")
            .unwrap();
        db.compact().unwrap();
        assert_the_queries_of_the_issue(
            &db,
            "after a write and compact()",
            &["Graph", "Repository", "Starred"],
        );
        db.close().unwrap();
    }
    let db = GrafeoDB::open(&path).unwrap();
    assert_the_queries_of_the_issue(&db, "after a reopen", &["Graph", "Repository", "Starred"]);
    db.close().unwrap();
}
