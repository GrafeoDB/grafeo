//! The calls that read or replace a database's whole state see all of it
//! after `compact()`.
//!
//! - `restore_snapshot()` replaces the whole state, and the database
//!   reopens with the snapshot, however the ids of the snapshot and of the
//!   data before it meet.
//! - `iter_nodes()` and `iter_edges()` read every node and edge, without
//!   those deleted since `compact()`.
//! - A snapshot taken after `compact()` imports whole.
//!
//! ```bash
//! cargo test -p grafeo-engine --all-features --test compact_whole_state
//! ```

#![cfg(all(feature = "compact-store", feature = "lpg", feature = "gql"))]

use grafeo_common::types::Value;
use grafeo_engine::GrafeoDB;

/// The names of the people, sorted.
const PEOPLE: &str = "MATCH (p:Person) RETURN p.name AS name ORDER BY name";

/// Who knows whom, since when.
const KNOWS: &str = "MATCH (a:Person)-[k:KNOWS]->(b:Person) \
                     RETURN a.name AS a, b.name AS b, k.since AS since ORDER BY a";

/// One row per name.
fn names(names: &[&str]) -> Vec<Vec<Value>> {
    names.iter().map(|name| vec![Value::from(*name)]).collect()
}

fn rows(db: &GrafeoDB, query: &str) -> Vec<Vec<Value>> {
    db.execute(query).unwrap().rows().to_vec()
}

/// Alix and Gus, who knows Alix since 2019, and Amsterdam, where Alix lives.
/// Created first, they have the ids 0 to 2 and the edges 0 and 1.
fn populate(db: &GrafeoDB) {
    db.execute(
        "INSERT (a:Person {name: 'Alix'})<-[:KNOWS {since: 2019}]-(:Person {name: 'Gus'}), \
         (a)-[:LIVES_IN]->(:City {name: 'Amsterdam'})",
    )
    .unwrap();
}

/// A snapshot of another database: Mia, who knows Jules since 1988. Its
/// nodes have the ids 0 and 1 and its edge the id 0, those of Alix, Gus and
/// the KNOWS edge of a compacted base.
fn snapshot_of_mia_and_jules() -> Vec<u8> {
    let source = GrafeoDB::new_in_memory();
    source
        .execute("INSERT (:Person {name: 'Mia'})-[:KNOWS {since: 1988}]->(:Person {name: 'Jules'})")
        .unwrap();
    source.export_snapshot().unwrap()
}

/// Checks that the database holds what the snapshot of Mia and Jules holds,
/// and nothing of the compacted base.
fn assert_holds_the_snapshot(db: &GrafeoDB, stage: &str) {
    assert_eq!(rows(db, PEOPLE), names(&["Jules", "Mia"]), "{stage}");
    assert_eq!(
        rows(db, KNOWS),
        vec![vec![
            Value::from("Mia"),
            Value::from("Jules"),
            Value::Int64(1988)
        ]],
        "{stage}"
    );
    assert_eq!(
        rows(db, "MATCH (n) RETURN count(n)"),
        vec![vec![Value::Int64(2)]],
        "{stage}: no node of the base"
    );
    assert_eq!(
        rows(db, "MATCH (c:City) RETURN c.name"),
        Vec::<Vec<Value>>::new(),
        "{stage}: Amsterdam was only in the base"
    );
    assert_eq!(db.node_count(), 2, "{stage}");
    assert_eq!(db.edge_count(), 1, "{stage}");
}

/// A compacted database with a base delete and an overlay node, restored
/// from the snapshot: the snapshot is all it holds. Writes after the restore
/// get ids of their own, and a new `compact()` compacts what it holds.
#[test]
fn a_restore_replaces_the_compacted_base() {
    let mut db = GrafeoDB::new_in_memory();
    populate(&db);
    db.compact().unwrap();
    db.execute("MATCH (c:City) DETACH DELETE c").unwrap();
    db.execute("INSERT (:Person {name: 'Vincent'})").unwrap();
    db.execute("MATCH (g:Person {name: 'Gus'}) SET g.age = 19")
        .unwrap();

    db.restore_snapshot(&snapshot_of_mia_and_jules()).unwrap();
    assert_holds_the_snapshot(&db, "after the restore");
    assert_eq!(
        db.iter_nodes().count(),
        2,
        "the direct iteration holds the snapshot too"
    );

    db.execute("INSERT (:Person {name: 'Vincent'})").unwrap();
    assert_eq!(rows(&db, PEOPLE), names(&["Jules", "Mia", "Vincent"]));
    db.compact().unwrap();
    assert_eq!(
        rows(&db, PEOPLE),
        names(&["Jules", "Mia", "Vincent"]),
        "compacted again"
    );
}

/// The restore of a compacted file: closed and opened again it holds the
/// snapshot, and takes writes. The same after a restore of a reopened
/// compacted file.
#[cfg(all(feature = "grafeo-file", feature = "wal"))]
#[test]
fn a_restored_compacted_file_reopens_as_a_plain_store() {
    let dir = tempfile::tempdir().unwrap();
    for (case, reopen_before_restore) in [
        ("restored at once", false),
        ("restored after a reopen", true),
    ] {
        let path = dir.path().join(format!("{reopen_before_restore}.grafeo"));
        {
            let mut db = GrafeoDB::open(&path).unwrap();
            populate(&db);
            db.compact().unwrap();
            db.execute("MATCH (c:City) DETACH DELETE c").unwrap();
            if reopen_before_restore {
                db.close().unwrap();
                db = GrafeoDB::open(&path).unwrap();
            }
            db.restore_snapshot(&snapshot_of_mia_and_jules()).unwrap();
            assert_holds_the_snapshot(&db, case);
            db.close().unwrap();
        }
        {
            let db = GrafeoDB::open(&path).unwrap();
            assert_holds_the_snapshot(&db, &format!("{case}, reopened"));
            db.execute("INSERT (:Person {name: 'Vincent'})").unwrap();
            db.close().unwrap();
        }
        let db = GrafeoDB::open(&path).unwrap();
        assert_eq!(
            rows(&db, PEOPLE),
            names(&["Jules", "Mia", "Vincent"]),
            "{case}, reopened twice"
        );
        db.close().unwrap();
    }
}

/// A snapshot of a compacted database holds the base and the overlay, and
/// imports whole: each label of a node with two, without what was deleted.
#[test]
fn a_snapshot_of_a_compacted_database_imports_whole() {
    let mut db = GrafeoDB::new_in_memory();
    populate(&db);
    db.execute("MATCH (a:Person {name: 'Alix'}) SET a:Employee")
        .unwrap();
    db.compact().unwrap();
    db.execute("MATCH (g:Person {name: 'Gus'}) DETACH DELETE g")
        .unwrap();
    db.execute("INSERT (:Person:Employee {name: 'Vincent'})")
        .unwrap();

    let imported = GrafeoDB::import_snapshot(&db.export_snapshot().unwrap()).unwrap();
    assert_eq!(rows(&imported, PEOPLE), names(&["Alix", "Vincent"]));
    assert_eq!(
        rows(
            &imported,
            "MATCH (e:Employee) RETURN e.name AS name ORDER BY name"
        ),
        names(&["Alix", "Vincent"])
    );
    assert_eq!(
        rows(
            &imported,
            "MATCH (:Person {name: 'Alix'})-[:LIVES_IN]->(c:City) RETURN c.name"
        ),
        names(&["Amsterdam"])
    );
    assert_eq!(rows(&imported, KNOWS), Vec::<Vec<Value>>::new());
}

/// The names of the nodes `iter_nodes()` returns, sorted.
fn iterated_names(db: &GrafeoDB) -> Vec<String> {
    let mut names: Vec<String> = db
        .iter_nodes()
        .map(|node| {
            match node
                .properties
                .get(&grafeo_common::types::PropertyKey::new("name"))
            {
                Some(Value::String(name)) => name.to_string(),
                other => format!("{other:?}"),
            }
        })
        .collect();
    names.sort();
    names
}

/// The types of the edges `iter_edges()` returns, sorted.
fn iterated_edge_types(db: &GrafeoDB) -> Vec<String> {
    let mut types: Vec<String> = db
        .iter_edges()
        .map(|edge| edge.edge_type.to_string())
        .collect();
    types.sort();
    types
}

/// `iter_nodes()` and `iter_edges()` read the compacted base and the
/// overlay: the nodes and edges of both, a base node changed since as it is
/// now, and not those deleted since.
#[test]
fn iteration_reads_the_compacted_base_and_the_overlay() {
    let mut db = GrafeoDB::new_in_memory();
    populate(&db);
    db.compact().unwrap();
    assert_eq!(iterated_names(&db), ["Alix", "Amsterdam", "Gus"]);
    assert_eq!(iterated_edge_types(&db), ["KNOWS", "LIVES_IN"]);

    db.execute("MATCH (a:Person {name: 'Alix'}) INSERT (a)-[:KNOWS {since: 3}]->(:Person {name: 'Vincent'})")
        .unwrap();
    db.execute("MATCH (g:Person {name: 'Gus'}) SET g.age = 88")
        .unwrap();
    db.execute("MATCH (c:City) DETACH DELETE c").unwrap();
    assert_eq!(iterated_names(&db), ["Alix", "Gus", "Vincent"]);
    assert_eq!(iterated_edge_types(&db), ["KNOWS", "KNOWS"]);
    let gus = db
        .iter_nodes()
        .find(|node| {
            node.properties
                .get(&grafeo_common::types::PropertyKey::new("name"))
                == Some(&Value::from("Gus"))
        })
        .expect("Gus");
    assert_eq!(
        gus.properties
            .get(&grafeo_common::types::PropertyKey::new("age")),
        Some(&Value::Int64(88)),
        "Gus as he is now"
    );
    let ids: Vec<_> = db.iter_nodes().map(|node| node.id).collect();
    let mut sorted = ids.clone();
    sorted.sort_unstable();
    assert_eq!(ids, sorted, "in id order");

    db.execute("MATCH (:Person {name: 'Gus'})-[k:KNOWS]->() DELETE k")
        .unwrap();
    assert_eq!(
        iterated_edge_types(&db),
        ["KNOWS"],
        "the base edge deleted since is gone"
    );
}
