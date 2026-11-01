//! The S fixes for Deriva's selector workloads (0.6.1 selector-gap plan),
//! each by its rows and, where it applies, its plan:
//!
//! - 2a: `labels(n)` and `type(r)` give the same groups without building the
//!   node or the edge: one group per label set, whatever order the labels
//!   were written in, and what a transaction or a past epoch sees.
//! - 2b: `RETURN n` gives the same node from a cheaper build: every property
//!   of a node with many, the same on every read, after updates.
//! - 3a: an edge check reads only the edges of its type: it finds a visible
//!   edge among many of other types, and skips a deleted one, one another
//!   transaction has not committed, and one a past epoch did not have.
//! - 1a0: `MATCH (n:Graph) WHERE (n:File OR n:TypeDefinition) AND ...` scans
//!   only File and TypeDefinition, with the rows of the full scan.
//! - 2d: `MATCH (n:Graph) RETURN count(*)` is answered from the counts, and
//!   still counts right in a transaction with its own writes and at a past
//!   epoch.
//!
//! ```bash
//! cargo test -p grafeo-engine --all-features --test selector_paths_s_group
//! ```

#![cfg(feature = "gql")]

use std::collections::BTreeMap;

use grafeo_common::types::Value;
use grafeo_engine::{Config, GrafeoDB, HistoryRetention};

fn rows(db: &GrafeoDB, query: &str) -> Vec<Vec<Value>> {
    db.execute(query)
        .unwrap_or_else(|error| panic!("{query}: {error}"))
        .rows()
        .to_vec()
}

fn plan(db: &GrafeoDB, keyword: &str, query: &str) -> String {
    match &rows(db, &format!("{keyword} {query}"))[0][0] {
        Value::String(text) => text.to_string(),
        other => panic!("{keyword} {query}: {other:?}"),
    }
}

/// A label list as a sorted list of names.
fn label_set(value: &Value) -> Vec<String> {
    let Value::List(items) = value else {
        panic!("labels(n) gave {value:?}");
    };
    let mut names: Vec<String> = items
        .iter()
        .map(|item| match item {
            Value::String(name) => name.to_string(),
            other => panic!("a label: {other:?}"),
        })
        .collect();
    names.sort();
    names
}

/// `labels(n), count(*)` as a map from the sorted label set to its count,
/// failing if one label set comes back as two groups.
fn label_groups(db: &GrafeoDB, query: &str) -> BTreeMap<Vec<String>, i64> {
    let mut groups = BTreeMap::new();
    for row in rows(db, query) {
        let Value::Int64(count) = row[1] else {
            panic!("a count: {:?}", row[1]);
        };
        let set = label_set(&row[0]);
        assert!(
            groups.insert(set.clone(), count).is_none(),
            "{set:?} came back as two groups"
        );
    }
    groups
}

fn history() -> GrafeoDB {
    GrafeoDB::with_config(Config::in_memory().with_history_retention(HistoryRetention::Forever))
        .unwrap()
}

fn set(names: &[&str]) -> Vec<String> {
    names.iter().map(|name| (*name).to_string()).collect()
}

// ---------------------------------------------------------------------------
// 2a: labels(n) and type(r)
// ---------------------------------------------------------------------------

/// One group per label set, also when the labels were written in another
/// order or added later.
#[test]
fn labels_group_by_label_set_in_any_written_order() {
    let db = GrafeoDB::new_in_memory();
    db.execute("INSERT (:Graph:File {id: 'f1'}), (:File:Graph {id: 'f2'}), (:Graph {id: 'f3'})")
        .unwrap();
    db.execute("MATCH (n {id: 'f3'}) SET n:File").unwrap();
    db.execute(
        "INSERT (:Graph:Method {id: 'm1'}), (:Method:Graph {id: 'm2'}), (:Graph {id: 'g1'})",
    )
    .unwrap();
    let groups = label_groups(&db, "MATCH (n:Graph) RETURN labels(n) AS l, count(*) AS c");
    assert_eq!(
        groups,
        BTreeMap::from([
            (set(&["File", "Graph"]), 3),
            (set(&["Graph"]), 1),
            (set(&["Graph", "Method"]), 2),
        ])
    );
}

/// labels(n) reads the labels a transaction and a past epoch see.
#[test]
fn labels_follow_the_transaction_and_the_epoch() {
    let db = history();
    db.execute("INSERT (:Graph:File {id: 'f1'}), (:Graph:Method {id: 'm1'})")
        .unwrap();
    let before = db.current_epoch();
    db.execute("MATCH (n {id: 'm1'}) REMOVE n:Method SET n:Test")
        .unwrap();
    let now = label_groups(&db, "MATCH (n:Graph) RETURN labels(n) AS l, count(*) AS c");
    assert_eq!(
        now,
        BTreeMap::from([(set(&["File", "Graph"]), 1), (set(&["Graph", "Test"]), 1)])
    );
    let then: BTreeMap<Vec<String>, i64> = db
        .execute_at_epoch(
            "MATCH (n:Graph) RETURN labels(n) AS l, count(*) AS c",
            before,
        )
        .unwrap()
        .rows()
        .iter()
        .map(|row| {
            (
                label_set(&row[0]),
                if let Value::Int64(c) = row[1] { c } else { -1 },
            )
        })
        .collect();
    assert_eq!(
        then,
        BTreeMap::from([(set(&["File", "Graph"]), 1), (set(&["Graph", "Method"]), 1)])
    );

    let mut session = db.session();
    session.begin_transaction().unwrap();
    session
        .execute("MATCH (n {id: 'f1'}) SET n:Archive")
        .unwrap();
    let inside = session
        .execute("MATCH (n {id: 'f1'}) RETURN labels(n) AS l")
        .unwrap();
    assert_eq!(
        label_set(&inside.rows()[0][0]),
        set(&["Archive", "File", "Graph"])
    );
    let outside = rows(&db, "MATCH (n {id: 'f1'}) RETURN labels(n) AS l");
    assert_eq!(label_set(&outside[0][0]), set(&["File", "Graph"]));
    session.rollback().unwrap();
}

/// type(r) groups by edge type, and reads what the transaction sees.
#[test]
fn edge_types_group_by_type() {
    let db = GrafeoDB::new_in_memory();
    db.execute(
        "INSERT (a:Graph {id: 'a'})-[:USES]->(b:Graph {id: 'b'}), (a)-[:USES]->(c:Graph {id: 'c'}), (b)-[:CONTAINS]->(c)",
    )
    .unwrap();
    let query = "MATCH ()-[r]->() RETURN type(r) AS t, count(*) AS c ORDER BY t";
    assert_eq!(
        rows(&db, query),
        [
            vec![Value::from("CONTAINS"), Value::Int64(1)],
            vec![Value::from("USES"), Value::Int64(2)],
        ]
    );
    let mut session = db.session();
    session.begin_transaction().unwrap();
    session
        .execute("MATCH ()-[r:CONTAINS]->() DELETE r")
        .unwrap();
    assert_eq!(
        session.execute(query).unwrap().rows(),
        [vec![Value::from("USES"), Value::Int64(2)]]
    );
    session.rollback().unwrap();
    assert_eq!(rows(&db, query).len(), 2);
}

// ---------------------------------------------------------------------------
// 2b: RETURN n
// ---------------------------------------------------------------------------

/// A node with many properties of every kind comes back whole, the same on
/// every read, and after an update with the new values only.
#[test]
fn return_n_gives_every_property_on_every_read() {
    let db = GrafeoDB::new_in_memory();
    let mut props: Vec<String> = (0..40).map(|i| format!("p{i}: {i}")).collect();
    props.extend([
        "name: 'Alix'".to_string(),
        "score: 2.5e0".to_string(),
        "flag: true".to_string(),
        "tags: ['a', 'b']".to_string(),
        "born: date('1990-01-02')".to_string(),
    ]);
    db.execute(&format!("INSERT (:Graph:Method {{{}}})", props.join(", ")))
        .unwrap();
    let read = || rows(&db, "MATCH (n:Method) RETURN n")[0][0].clone();
    let first = read();
    let Value::Map(map) = &first else {
        panic!("RETURN n gave {first:?}");
    };
    assert_eq!(map.len(), 47, "_id, _labels and 45 properties: {map:?}");
    for _ in 0..5 {
        assert_eq!(read(), first);
    }
    let properties = rows(&db, "MATCH (n:Method) RETURN properties(n) AS p")[0][0].clone();
    let Value::Map(properties) = properties else {
        panic!("properties(n)");
    };
    for (key, value) in properties.iter() {
        assert_eq!(map.get(key), Some(value), "{key:?}");
    }
    db.execute("MATCH (n:Method) SET n.p0 = 100 REMOVE n.p1")
        .unwrap();
    let Value::Map(after) = read() else {
        panic!("RETURN n after the update");
    };
    assert_eq!(after.len(), 46);
    assert_eq!(
        after.get(&grafeo_common::types::PropertyKey::new("p0")),
        Some(&Value::Int64(100))
    );
    assert_eq!(
        after.get(&grafeo_common::types::PropertyKey::new("p1")),
        None
    );
}

// ---------------------------------------------------------------------------
// 3a: typed edge checks
// ---------------------------------------------------------------------------

/// A node with 500 edges of other types and one IN_ENV edge: the typed check
/// finds it, and a node whose IN_ENV edge was deleted has none.
#[test]
fn a_typed_edge_check_finds_its_edge_among_many_others() {
    let db = GrafeoDB::new_in_memory();
    db.execute(
        "INSERT (:Graph:Module {id: 'busy'}), (:Graph:Module {id: 'gone'}), (:Env {id: 'prod'})",
    )
    .unwrap();
    db.execute(
        "MATCH (m:Module) UNWIND range(1, 500) AS i INSERT (m)-[:USES]->(:Graph {id: m.id + toString(i)})",
    )
    .unwrap();
    db.execute("MATCH (m:Module), (e:Env) INSERT (m)-[:IN_ENV]->(e)")
        .unwrap();
    db.execute("MATCH (:Module {id: 'gone'})-[r:IN_ENV]->() DELETE r")
        .unwrap();
    let query = "MATCH (m:Module) WHERE EXISTS { MATCH (m)-[:IN_ENV]->() } RETURN m.id AS id";
    assert_eq!(rows(&db, query), [[Value::from("busy")]]);
    let query = "MATCH (m:Module) WHERE NOT EXISTS { MATCH (m)-[:IN_ENV]->() } RETURN m.id AS id";
    assert_eq!(rows(&db, query), [[Value::from("gone")]]);
}

/// The check sees what the transaction sees: not another transaction's
/// uncommitted edge, its own new edge, not its own deleted edge, and at a
/// past epoch the edge that was there then.
#[test]
fn a_typed_edge_check_follows_visibility() {
    let db = history();
    db.execute("INSERT (:Graph:Module {id: 'a'})-[:IN_ENV]->(:Env), (:Graph:Module {id: 'b'})")
        .unwrap();
    let query =
        "MATCH (m:Module) WHERE EXISTS { MATCH (m)-[:IN_ENV]->() } RETURN m.id AS id ORDER BY id";
    let before = db.current_epoch();

    let mut other = db.session();
    other.begin_transaction().unwrap();
    other
        .execute("MATCH (m:Module {id: 'b'}), (e:Env) INSERT (m)-[:IN_ENV]->(e)")
        .unwrap();
    assert_eq!(
        rows(&db, query),
        [[Value::from("a")]],
        "another transaction's edge"
    );
    assert_eq!(
        other.execute(query).unwrap().rows(),
        [[Value::from("a")], [Value::from("b")]],
        "its own new edge"
    );
    other
        .execute("MATCH (:Module {id: 'a'})-[r:IN_ENV]->() DELETE r")
        .unwrap();
    assert_eq!(
        other.execute(query).unwrap().rows(),
        [[Value::from("b")]],
        "its own deleted edge"
    );
    other.commit().unwrap();
    assert_eq!(rows(&db, query), [[Value::from("b")]]);
    let then = db.execute_at_epoch(query, before).unwrap();
    assert_eq!(then.rows(), [[Value::from("a")]], "the past epoch's edge");
}

// ---------------------------------------------------------------------------
// 1a0: an OR of labels under a scan with a label
// ---------------------------------------------------------------------------

fn code_graph() -> GrafeoDB {
    let db = GrafeoDB::new_in_memory();
    for (label, count) in [
        ("File", 40),
        ("TypeDefinition", 30),
        ("Method", 300),
        ("Test", 100),
    ] {
        db.execute(&format!(
            "UNWIND range(1, {count}) AS i INSERT (:Graph:{label} {{id: '{label}' + toString(i), x: (i * 37) % 100}})"
        ))
        .unwrap();
    }
    db
}

#[test]
fn an_or_of_labels_under_a_labelled_scan_scans_those_labels() {
    let db = code_graph();
    let query = "MATCH (n:Graph) WHERE (n:File OR n:TypeDefinition) AND n.x >= 90 RETURN n.id AS id ORDER BY id";
    let reference = "MATCH (n) WHERE n:Graph AND (n:File OR n:TypeDefinition) AND n.x >= 90 RETURN n.id AS id ORDER BY id";
    let expected = rows(&db, reference);
    assert_ne!(expected.len(), 0, "the data selects nothing");
    assert_eq!(rows(&db, query), expected);
    let text = plan(&db, "EXPLAIN", query);
    assert!(
        text.contains("[labels: File|TypeDefinition]") && !text.contains("NodeScan (n:Graph)"),
        "{text}"
    );
    let profile = plan(&db, "PROFILE", query);
    assert!(
        profile.contains("Scan (n:File|TypeDefinition)  rows=70"),
        "the union's 40 + 30 nodes: {profile}"
    );
}

// ---------------------------------------------------------------------------
// 2d: count(*) from the counts
// ---------------------------------------------------------------------------

/// A bare count of a label reads no node: PROFILE has no scan row count.
#[test]
fn a_bare_label_count_reads_no_node() {
    let db = code_graph();
    assert_eq!(
        rows(&db, "MATCH (n:Graph) RETURN count(*) AS c"),
        [[Value::Int64(470)]]
    );
    let text = plan(&db, "PROFILE", "MATCH (n:Graph) RETURN count(*) AS c");
    assert!(!text.contains("Scan (n:Graph)  rows=470"), "{text}");
}

/// The count sees a transaction's own uncommitted writes, not another's, and
/// a past epoch's nodes; a label added or removed counts.
#[test]
fn a_bare_label_count_follows_visibility() {
    let db = history();
    db.execute("UNWIND range(1, 10) AS i INSERT (:Graph {i: i})")
        .unwrap();
    let before = db.current_epoch();
    db.execute("MATCH (n:Graph) WHERE n.i <= 2 DELETE n")
        .unwrap();
    db.execute("INSERT (:Other {i: 99})").unwrap();
    db.execute("MATCH (n:Other) SET n:Graph").unwrap();
    let count = "MATCH (n:Graph) RETURN count(*) AS c";
    assert_eq!(rows(&db, count), [[Value::Int64(9)]]);
    assert_eq!(
        db.execute_at_epoch(count, before).unwrap().rows(),
        [[Value::Int64(10)]]
    );
    let mut session = db.session();
    session.begin_transaction().unwrap();
    session
        .execute("UNWIND range(1, 5) AS i INSERT (:Graph {i: 100 + i})")
        .unwrap();
    session
        .execute("MATCH (n:Graph {i: 99}) REMOVE n:Graph")
        .unwrap();
    assert_eq!(session.execute(count).unwrap().rows(), [[Value::Int64(13)]]);
    assert_eq!(rows(&db, count), [[Value::Int64(9)]], "another session");
    session.rollback().unwrap();
    assert_eq!(rows(&db, count), [[Value::Int64(9)]]);
}
