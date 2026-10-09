//! `compact()` keeps the nodes without labels.
//!
//! The nodes without labels stay with their properties and their edges (to
//! and from labeled nodes and between themselves), in a scan of all nodes,
//! the counts and the statistics: through `compact()`, writes after it,
//! `recompact()`, and a close and reopen.
//!
//! ```bash
//! cargo test -p grafeo-engine --all-features --test compact_unlabeled_nodes
//! ```

#![cfg(all(feature = "compact-store", feature = "lpg", feature = "gql"))]

use grafeo_common::types::Value;
use grafeo_engine::GrafeoDB;

/// The statistics as the planner reads them: refreshed from the store's
/// counters first.
fn fresh_statistics(db: &GrafeoDB) -> std::sync::Arc<grafeo_core::statistics::Statistics> {
    let store = db.store();
    store.ensure_statistics_fresh();
    store.statistics()
}

/// Every node's name, sorted.
const NAMES: &str = "MATCH (n) RETURN n.name AS name ORDER BY name";

/// The names of the nodes without labels, sorted.
const UNLABELED: &str = "MATCH (n) WHERE size(labels(n)) = 0 RETURN n.name AS name ORDER BY name";

/// The names of the people, sorted.
const PEOPLE: &str = "MATCH (p:Person) RETURN p.name AS name ORDER BY name";

/// Who knows whom since when, by year.
const KNOWS: &str = "MATCH (a)-[k:KNOWS]->(b) RETURN a.name, b.name, k.since ORDER BY k.since";

/// Who knows Mia, read from her side.
const KNOW_MIA: &str = "MATCH (m {name: 'Mia'})<-[:KNOWS]-(a) RETURN a.name AS name ORDER BY name";

/// Gus (19) and Mia have no labels, Alix is a Person. Gus knows Alix since
/// 3, Alix knows Mia since 19, and Gus knows Mia since 88.
fn populate(db: &GrafeoDB) {
    db.execute(
        "INSERT (g {name: 'Gus', age: 19})-[:KNOWS {since: 3}]->(a:Person {name: 'Alix'}), \
         (a)-[:KNOWS {since: 19}]->(m {name: 'Mia'}), (g)-[:KNOWS {since: 88}]->(m)",
    )
    .unwrap();
}

fn rows(db: &GrafeoDB, query: &str) -> Vec<Vec<Value>> {
    db.execute(query).unwrap().rows().to_vec()
}

/// One row per name.
fn names(names: &[&str]) -> Vec<Vec<Value>> {
    names.iter().map(|name| vec![Value::from(*name)]).collect()
}

/// One row per `(who, whom, since)`.
fn knows(edges: &[(&str, &str, i64)]) -> Vec<Vec<Value>> {
    edges
        .iter()
        .map(|&(who, whom, since)| vec![Value::from(who), Value::from(whom), Value::Int64(since)])
        .collect()
}

/// What `db` holds, as the queries above and the counts read it.
#[derive(Debug, PartialEq)]
struct Holds {
    names: Vec<Vec<Value>>,
    unlabeled: Vec<Vec<Value>>,
    people: Vec<Vec<Value>>,
    knows: Vec<Vec<Value>>,
    know_mia: Vec<Vec<Value>>,
    gus_age: Vec<Vec<Value>>,
    counted: Vec<Vec<Value>>,
    node_count: usize,
    edge_count: usize,
}

impl Holds {
    fn of(db: &GrafeoDB) -> Self {
        Self {
            names: rows(db, NAMES),
            unlabeled: rows(db, UNLABELED),
            people: rows(db, PEOPLE),
            knows: rows(db, KNOWS),
            know_mia: rows(db, KNOW_MIA),
            gus_age: rows(db, "MATCH (g {name: 'Gus'}) RETURN g.age"),
            counted: rows(db, "MATCH (n) RETURN count(n)"),
            node_count: db.node_count(),
            edge_count: db.edge_count(),
        }
    }

    /// What `populate` writes.
    fn populated() -> Self {
        Self {
            names: names(&["Alix", "Gus", "Mia"]),
            unlabeled: names(&["Gus", "Mia"]),
            people: names(&["Alix"]),
            knows: knows(&[("Gus", "Alix", 3), ("Alix", "Mia", 19), ("Gus", "Mia", 88)]),
            know_mia: names(&["Alix", "Gus"]),
            gus_age: vec![vec![Value::Int64(19)]],
            counted: vec![vec![Value::Int64(3)]],
            node_count: 3,
            edge_count: 3,
        }
    }

    /// The node and edge counts of the statistics, which the planner
    /// estimates with, agree with the counts of `self`.
    fn statistics_agree(&self, db: &GrafeoDB, stage: &str) {
        let statistics = fresh_statistics(db);
        assert_eq!(
            (statistics.total_nodes, statistics.total_edges),
            (
                u64::try_from(self.node_count).unwrap(),
                u64::try_from(self.edge_count).unwrap()
            ),
            "{stage}: the statistics"
        );
    }

    /// What `populate` and then `change` write.
    fn changed() -> Self {
        Self {
            names: names(&["Alix", "Gus", "Jules", "Mia", "Vincent"]),
            unlabeled: names(&["Gus", "Jules"]),
            people: names(&["Alix", "Mia", "Vincent"]),
            knows: knows(&[
                ("Gus", "Alix", 3),
                ("Alix", "Mia", 19),
                ("Gus", "Mia", 88),
                ("Vincent", "Jules", 1988),
            ]),
            know_mia: names(&["Alix", "Gus"]),
            gus_age: vec![vec![Value::Int64(88)]],
            counted: vec![vec![Value::Int64(5)]],
            node_count: 5,
            edge_count: 4,
        }
    }
}

/// Writes after `compact()`: Gus (a compacted node without labels) turns
/// 88, Mia (another one) becomes a Person, and Vincent, a Person, knows
/// Jules, who has no label, since 1988.
fn change(db: &GrafeoDB) {
    db.execute("MATCH (g {name: 'Gus'}) SET g.age = 88")
        .unwrap();
    db.execute("MATCH (m {name: 'Mia'}) SET m:Person").unwrap();
    db.execute("INSERT (:Person {name: 'Vincent'})-[:KNOWS {since: 1988}]->({name: 'Jules'})")
        .unwrap();
}

/// `compact()` keeps the nodes without labels, with their properties and
/// edges, in the scans, counts and statistics.
#[test]
fn compact_keeps_the_nodes_without_labels() {
    let mut db = GrafeoDB::new_in_memory();
    populate(&db);
    assert_eq!(Holds::of(&db), Holds::populated(), "before compact()");
    db.compact().unwrap();
    assert_eq!(Holds::of(&db), Holds::populated(), "after compact()");
    Holds::populated().statistics_agree(&db, "after compact()");
}

/// A database whose nodes have no labels at all compacts whole, not to an
/// empty one.
#[test]
fn a_database_without_labels_compacts_whole() {
    let mut db = GrafeoDB::new_in_memory();
    db.execute("INSERT ({name: 'Vincent'})-[:KNOWS {since: 1988}]->({name: 'Jules'})")
        .unwrap();
    db.compact().unwrap();
    assert_eq!(rows(&db, NAMES), names(&["Jules", "Vincent"]));
    assert_eq!(
        rows(&db, KNOWS),
        knows(&[("Vincent", "Jules", 1988)]),
        "their edge"
    );
    assert_eq!(db.node_count(), 2);
    assert_eq!(fresh_statistics(&db).total_nodes, 2);
}

/// Writes reach the compacted nodes without labels, and `recompact()`
/// keeps them: also a node without labels created after `compact()`, and a
/// delete of one.
#[test]
fn nodes_without_labels_stay_through_writes_and_recompact() {
    let mut db = GrafeoDB::new_in_memory();
    populate(&db);
    db.compact().unwrap();
    change(&db);
    assert_eq!(Holds::of(&db), Holds::changed(), "after the writes");
    Holds::changed().statistics_agree(&db, "after the writes");
    db.compact().unwrap();
    assert_eq!(Holds::of(&db), Holds::changed(), "after recompact()");
    Holds::changed().statistics_agree(&db, "after recompact()");

    db.execute("MATCH (j {name: 'Jules'}) DETACH DELETE j")
        .unwrap();
    db.compact().unwrap();
    assert_eq!(
        rows(&db, UNLABELED),
        names(&["Gus"]),
        "Jules is deleted, Gus stays"
    );
    assert_eq!(db.node_count(), 4);
}

/// The nodes without labels are in the file: after `compact()` and a close
/// and reopen, after writes and `recompact()` of the reopened database, and
/// after the next reopen.
#[cfg(all(feature = "grafeo-file", feature = "wal"))]
#[test]
fn a_reopened_compacted_file_keeps_the_nodes_without_labels() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("unlabeled.grafeo");
    {
        let mut db = GrafeoDB::open(&path).unwrap();
        populate(&db);
        db.compact().unwrap();
        db.close().unwrap();
    }
    {
        let mut db = GrafeoDB::open(&path).unwrap();
        assert_eq!(Holds::of(&db), Holds::populated(), "after a reopen");
        Holds::populated().statistics_agree(&db, "after a reopen");
        change(&db);
        db.compact().unwrap();
        assert_eq!(
            Holds::of(&db),
            Holds::changed(),
            "after a reopen, writes and a merge"
        );
        db.close().unwrap();
    }
    let db = GrafeoDB::open(&path).unwrap();
    assert_eq!(Holds::of(&db), Holds::changed(), "after a second reopen");
    Holds::changed().statistics_agree(&db, "after a second reopen");
    db.close().unwrap();
}
