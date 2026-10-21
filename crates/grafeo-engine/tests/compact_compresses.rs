//! `compact()` compresses the property values the graphs hold decoded (H5):
//! the memory they take drops, in memory and in a database file alike,
//! every read returns what it returned before (an open transaction's view
//! included), and a write after it reads back.
//!
//! ```bash
//! cargo test -p grafeo-engine --all-features --test compact_compresses
//! ```

#![cfg(all(feature = "lpg", feature = "gql"))]

use grafeo_common::types::Value;
use grafeo_engine::GrafeoDB;

/// People `0..20_000` with an age and a score each, then every age written
/// once more, so the columns are decoded in memory.
fn people(db: &GrafeoDB) {
    db.execute("UNWIND range(0, 19999) AS i INSERT (:Person {i: i, age: i % 90, score: i * 1.5})")
        .unwrap();
    db.execute("MATCH (p:Person) SET p.age = p.age + 1")
        .unwrap();
}

/// What the queries of these tests return.
fn answers(db: &GrafeoDB) -> Vec<Value> {
    [
        "MATCH (p:Person) RETURN sum(p.age)",
        "MATCH (p:Person {i: 777}) RETURN p.score",
        "MATCH (p:Person) WHERE p.age > 80 RETURN count(p)",
    ]
    .iter()
    .map(|query| db.execute(query).unwrap().rows()[0][0].clone())
    .collect()
}

/// The bytes the property columns take, nodes and edges.
fn property_bytes(db: &GrafeoDB) -> usize {
    let usage = db.memory_usage();
    usage.store.node_properties_bytes + usage.store.edge_properties_bytes
}

/// In memory: the columns' memory drops, every answer stays, an open
/// transaction keeps its view, and a write after it reads back.
#[test]
fn compact_compresses_an_in_memory_database() {
    let mut db = GrafeoDB::new_in_memory();
    people(&db);
    let before = answers(&db);
    let bytes = property_bytes(&db);
    let report = db.compact().unwrap();
    assert!(!report.checkpointed);
    assert_eq!(report.rows_compressed, 20_000, "{report:?}");
    let after = property_bytes(&db);
    assert!(after * 2 < bytes, "{bytes} bytes before, {after} after");
    assert_eq!(answers(&db), before);

    let mut session = db.session();
    session.begin_transaction().unwrap();
    let seen = session
        .execute("MATCH (p:Person {i: 5}) RETURN p.age")
        .unwrap()
        .rows()[0][0]
        .clone();
    db.execute("MATCH (p:Person {i: 5}) SET p.age = 41")
        .unwrap();
    let report = db.compact().unwrap();
    assert!(
        report.rows_compressed < 20_000,
        "the age column the transaction may still read stays decoded: {report:?}"
    );
    assert_eq!(
        session
            .execute("MATCH (p:Person {i: 5}) RETURN p.age")
            .unwrap()
            .rows()[0][0],
        seen,
        "the open transaction reads what it read"
    );
    session.commit().unwrap();
    assert_eq!(
        db.execute("MATCH (p:Person {i: 5}) RETURN p.age")
            .unwrap()
            .rows()[0][0],
        Value::Int64(41)
    );
}

/// In a database file: the same, and a reopen reads every answer.
#[cfg(feature = "grafeo-file")]
#[test]
fn compact_compresses_a_database_file() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("compact.grafeo");
    let mut db = GrafeoDB::open(&path).unwrap();
    people(&db);
    let before = answers(&db);
    let bytes = property_bytes(&db);
    let report = db.compact().unwrap();
    assert!(report.checkpointed);
    assert_eq!(report.rows_compressed, 20_000, "{report:?}");
    assert!(property_bytes(&db) * 2 < bytes);
    assert_eq!(answers(&db), before);
    db.execute("MATCH (p:Person {i: 7}) SET p.score = 0.5")
        .unwrap();
    let written = db
        .execute("MATCH (p:Person {i: 7}) RETURN p.score")
        .unwrap()
        .rows()[0][0]
        .clone();
    db.close().unwrap();

    let db = GrafeoDB::open(&path).unwrap();
    assert_eq!(answers(&db)[0], before[0]);
    assert_eq!(
        db.execute("MATCH (p:Person {i: 7}) RETURN p.score")
            .unwrap()
            .rows()[0][0],
        written
    );
    db.close().unwrap();
}
