//! Nodes with several labels in a database file compacted by 0.5.44 (#595).
//!
//! Up to 0.5.44, `compact()` stored a node with several labels under one
//! label, its labels joined with `|`: the node matched none of its labels,
//! and `labels(n)` returned the joined name. A write after `compact()` that
//! found such a node without a label copied it to the overlay with that name
//! as one of its labels. `fixtures/compacted-labels/0.5.44/labels.grafeo`
//! holds both (`write_fixture.py` next to it wrote it with the released
//! wheel). Its base holds Alix and Vincent (`Graph`, `Repository`), Mia
//! (`Archive`, `Graph`, `Repository`), Gus (`Graph`) and Butch (`Graph`,
//! `Repository`), with an edge from Alix to Vincent. After `compact()`, Alix
//! got a city and the label `Starred` (the overlay copied her), Jules
//! (`Graph`, `Repository`) was created with an edge to Mia (the overlay
//! copied Mia, the edge's end) and Butch was deleted.
//!
//! Opening the file folds the base into the one store. Each node then has
//! each of its labels, as before `compact()`: in GQL and Cypher, on a
//! read-write open (which migrates the file), a read-only open and
//! `open_in_memory`, after writes through the labels, and after a reopen and
//! a checkpoint. The database lists those labels and not the joined names
//! (`CALL db.labels()`, `label_count()`, `schema()`).
//!
//! ```bash
//! cargo test -p grafeo-engine --all-features --test compacted_file_labels
//! ```

#![cfg(all(
    feature = "grafeo-file",
    feature = "wal",
    feature = "lpg",
    feature = "gql",
    feature = "cypher"
))]

use std::path::{Path, PathBuf};

use grafeo_common::types::Value;
use grafeo_engine::GrafeoDB;
use grafeo_engine::admin::SchemaInfo;

/// The fixture's file name.
const FILE: &str = "labels.grafeo";

/// The nodes the fixture holds once folded, by name, with their labels.
const FOLDED: [(&str, &[&str]); 5] = [
    ("Alix", &["Graph", "Repository", "Starred"]),
    ("Gus", &["Graph"]),
    ("Jules", &["Graph", "Repository"]),
    ("Mia", &["Archive", "Graph", "Repository"]),
    ("Vincent", &["Graph", "Repository"]),
];

/// The nodes after [`write_through_labels`]: Vincent is starred, Mia no
/// longer archived.
const WRITTEN: [(&str, &[&str]); 5] = [
    ("Alix", &["Graph", "Repository", "Starred"]),
    ("Gus", &["Graph"]),
    ("Jules", &["Graph", "Repository"]),
    ("Mia", &["Graph", "Repository"]),
    ("Vincent", &["Graph", "Repository", "Starred"]),
];

/// Every label a check scans for: those of the fixture, and one no node has.
const LABELS: [&str; 5] = ["Archive", "Graph", "Missing", "Repository", "Starred"];

/// The labels the database lists: each label of the folded nodes (`Archive`
/// also once a write removed it from Mia, as for any label whose nodes lost
/// it), and none of the joined names the fold split (`Graph|Repository`,
/// `Archive|Graph|Repository`).
const LISTED: [&str; 4] = ["Archive", "Graph", "Repository", "Starred"];

/// A copy of the fixture in a new directory, so the committed file stays as
/// 0.5.44 wrote it.
fn copy_fixture() -> (tempfile::TempDir, PathBuf) {
    let source = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/compacted-labels/0.5.44")
        .join(FILE);
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join(FILE);
    std::fs::copy(&source, &path).unwrap();
    (dir, path)
}

/// The rows of `query` in GQL, after checking that Cypher returns the same.
fn rows(db: &GrafeoDB, query: &str) -> Vec<Vec<Value>> {
    let gql = db
        .execute(query)
        .unwrap_or_else(|error| panic!("GQL {query}: {error}"))
        .rows()
        .to_vec();
    let cypher = db
        .execute_cypher(query)
        .unwrap_or_else(|error| panic!("Cypher {query}: {error}"))
        .rows()
        .to_vec();
    assert_eq!(gql, cypher, "{query}: GQL and Cypher return the same rows");
    gql
}

/// The single count `query` returns, in GQL and Cypher.
fn count(db: &GrafeoDB, query: &str) -> i64 {
    match rows(db, query).as_slice() {
        [row] => match row.as_slice() {
            [Value::Int64(count)] => *count,
            other => panic!("{query}: expected a count, got {other:?}"),
        },
        other => panic!("{query}: expected one row, got {other:?}"),
    }
}

fn text(value: &Value) -> String {
    match value {
        Value::String(text) => text.to_string(),
        other => panic!("expected a string, got {other:?}"),
    }
}

/// A `labels(n)` value, sorted.
fn sorted_labels(value: &Value) -> Vec<String> {
    let Value::List(labels) = value else {
        panic!("labels(n) is a list, got {value:?}");
    };
    let mut labels: Vec<String> = labels.iter().map(text).collect();
    labels.sort();
    labels
}

/// The names of the `nodes` that have every label of `labels`, sorted.
fn named_with(nodes: &[(&str, &[&str])], labels: &[&str]) -> Vec<String> {
    let mut names: Vec<String> = nodes
        .iter()
        .filter(|(_, node_labels)| labels.iter().all(|label| node_labels.contains(label)))
        .map(|(name, _)| (*name).to_string())
        .collect();
    names.sort();
    names
}

/// Checks that `db` holds `nodes` and that each label reads each of its
/// nodes: the queries of #595, a scan and a count per label (one no node has
/// included), patterns of two labels, a traversal between labels, and the
/// store's label index and counts.
fn assert_each_label_reads_its_nodes(db: &GrafeoDB, stage: &str, nodes: &[(&str, &[&str])]) {
    // The queries of the issue.
    assert_eq!(
        count(db, "MATCH (n:Graph) RETURN count(n)"),
        5,
        "{stage}: every node is a Graph"
    );
    assert_eq!(
        count(db, "MATCH (n:Repository) RETURN count(n)"),
        4,
        "{stage}: every node but Gus is a Repository"
    );
    assert_eq!(
        count(db, "MATCH (n:Graph:Repository) RETURN count(n)"),
        4,
        "{stage}: a pattern with both labels"
    );
    assert_eq!(
        rows(db, "MATCH (n {name: 'Alix'}) RETURN labels(n)")
            .iter()
            .map(|row| sorted_labels(&row[0]))
            .collect::<Vec<_>>(),
        [["Graph", "Repository", "Starred"]],
        "{stage}: labels(n) lists each label, none of them joined"
    );

    // Every node with its labels.
    let all: Vec<(String, Vec<String>)> =
        rows(db, "MATCH (n) RETURN n.name, labels(n) ORDER BY n.name")
            .iter()
            .map(|row| (text(&row[0]), sorted_labels(&row[1])))
            .collect();
    let expected: Vec<(String, Vec<String>)> = nodes
        .iter()
        .map(|(name, labels)| {
            (
                (*name).to_string(),
                labels.iter().map(|label| (*label).to_string()).collect(),
            )
        })
        .collect();
    assert_eq!(all, expected, "{stage}: the nodes and their labels");

    // A scan and a count per label, through the query and the store.
    let store = db.graph_store();
    for label in LABELS {
        let names = named_with(nodes, &[label]);
        assert_eq!(
            rows(
                db,
                &format!("MATCH (n:{label}) RETURN n.name ORDER BY n.name")
            )
            .iter()
            .map(|row| text(&row[0]))
            .collect::<Vec<_>>(),
            names,
            "{stage}: a {label} scan"
        );
        let expected = i64::try_from(names.len()).unwrap();
        assert_eq!(
            count(db, &format!("MATCH (n:{label}) RETURN count(n)")),
            expected,
            "{stage}: a {label} count"
        );
        assert_eq!(
            store.nodes_by_label(label).len(),
            names.len(),
            "{stage}: the store's {label} index"
        );
        assert_eq!(
            store.nodes_by_label_count(label),
            names.len(),
            "{stage}: the store's {label} count"
        );
    }
    for labels in [
        ["Archive", "Repository"],
        ["Repository", "Starred"],
        ["Graph", "Missing"],
    ] {
        let [first, second] = labels;
        assert_eq!(
            rows(
                db,
                &format!("MATCH (n:{first}:{second}) RETURN n.name ORDER BY n.name")
            )
            .iter()
            .map(|row| text(&row[0]))
            .collect::<Vec<_>>(),
            named_with(nodes, &labels),
            "{stage}: a pattern with the labels {first} and {second}"
        );
    }

    // The base's edge, between two folded nodes, and the overlay's.
    assert_eq!(
        rows(
            db,
            "MATCH (a:Repository)-[r:FORKED_FROM]->(b:Graph:Repository) \
             RETURN a.name, b.name, r.since ORDER BY a.name"
        ),
        [
            vec![
                Value::from("Alix"),
                Value::from("Vincent"),
                Value::Int64(2019)
            ],
            vec![Value::from("Jules"), Value::from("Mia"), Value::Int64(2088)],
        ],
        "{stage}: a traversal between labels"
    );
    assert_eq!(
        rows(db, "MATCH (n:Starred {name: 'Alix'}) RETURN n.city"),
        [vec![Value::from("Amsterdam")]],
        "{stage}: the overlay's copy of Alix keeps her city"
    );
    assert_the_listed_labels(db, stage, nodes);
}

/// Checks that `CALL db.labels()`, `label_count()` and `schema()` list
/// exactly [`LISTED`], `schema()` with the number of `nodes` of each.
fn assert_the_listed_labels(db: &GrafeoDB, stage: &str, nodes: &[(&str, &[&str])]) {
    let mut listed: Vec<String> = rows(db, "CALL db.labels()")
        .iter()
        .map(|row| text(&row[0]))
        .collect();
    listed.sort();
    assert_eq!(listed, LISTED, "{stage}: CALL db.labels()");
    assert_eq!(db.label_count(), LISTED.len(), "{stage}: label_count()");
    let SchemaInfo::Lpg(schema) = db.schema() else {
        panic!("{stage}: an LPG schema");
    };
    let mut counts: Vec<(String, usize)> = schema
        .labels
        .iter()
        .map(|label| (label.name.clone(), label.count))
        .collect();
    counts.sort();
    let expected: Vec<(String, usize)> = LISTED
        .iter()
        .map(|label| ((*label).to_string(), named_with(nodes, &[label]).len()))
        .collect();
    assert_eq!(counts, expected, "{stage}: schema()");
}

/// Writes through the labels of folded nodes: stars Vincent (a base node
/// the overlay did not copy) through `Repository`, and removes `Archive`
/// from Mia (one it copied) through `Archive`.
fn write_through_labels(db: &GrafeoDB) {
    db.execute("MATCH (n:Repository {name: 'Vincent'}) SET n:Starred")
        .unwrap();
    db.execute_cypher("MATCH (n:Archive:Graph) REMOVE n:Archive")
        .unwrap();
}

/// A read-write open migrates the file and folds its base: each node has
/// each of its labels, writes through them reach the node, and the migrated
/// file keeps them, without the joined names, through a close and a reopen,
/// and a checkpoint and another reopen.
#[test]
fn each_label_of_a_folded_node_reads_it_and_survives_a_reopen() {
    let (_dir, path) = copy_fixture();
    {
        let db = GrafeoDB::open(&path).unwrap();
        assert_each_label_reads_its_nodes(&db, "after the fold", &FOLDED);
        write_through_labels(&db);
        assert_each_label_reads_its_nodes(&db, "after writes through the labels", &WRITTEN);
        db.close().unwrap();
    }
    {
        let db = GrafeoDB::open(&path).unwrap();
        assert_each_label_reads_its_nodes(&db, "after a reopen", &WRITTEN);
        db.wal_checkpoint().unwrap();
        assert_each_label_reads_its_nodes(&db, "after a checkpoint", &WRITTEN);
        db.close().unwrap();
    }
    let db = GrafeoDB::open(&path).unwrap();
    assert_each_label_reads_its_nodes(&db, "after a checkpoint and a reopen", &WRITTEN);
    db.close().unwrap();
}

/// A read-only open and `open_in_memory` fold the base the same way.
#[test]
fn a_read_only_open_and_an_in_memory_copy_read_each_label() {
    let (_dir, path) = copy_fixture();
    let db = GrafeoDB::open_read_only(&path).unwrap();
    assert_each_label_reads_its_nodes(&db, "read-only", &FOLDED);
    db.close().unwrap();
    drop(db);

    let db = GrafeoDB::open_in_memory(&path).unwrap();
    assert_each_label_reads_its_nodes(&db, "in memory", &FOLDED);
    write_through_labels(&db);
    assert_each_label_reads_its_nodes(&db, "in memory, after writes", &WRITTEN);
}
