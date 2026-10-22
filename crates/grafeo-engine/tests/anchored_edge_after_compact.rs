//! An edge between two compacted nodes, looked up from both anchored ends
//! (`MATCH (a:A {id: 1})-[:T]->(b:B {id: 2})`), stays found after writes that
//! touch one of its ends (#345): a property set on an end, an unrelated edge
//! from it after a reopen, the same inside a transaction that rolls back, and
//! with a property index on `id`. The user's own repro is
//! `post_compact_overlay_visibility.rs`.
//!
//! ```bash
//! cargo test -p grafeo-engine --all-features --test anchored_edge_after_compact
//! ```

#![cfg(all(feature = "gql", feature = "cypher"))]

use grafeo_common::types::Value;
use grafeo_engine::GrafeoDB;
use grafeo_engine::session::Session;

const ANCHORED: &str = "MATCH (a:A {id: 1})-[t:T]->(b:B {id: 2}) RETURN a.id, t.w, b.id";

/// `(:A {id: 1})-[:T {w: 7}]->(:B {id: 2})`, compacted.
fn compacted(db: &mut GrafeoDB) {
    db.execute("INSERT (:A {id: 1})-[:T {w: 7}]->(:B {id: 2})")
        .unwrap();
    db.compact().unwrap();
}

/// The edge, found from its anchored ends in GQL and in Cypher.
fn assert_found(session: &Session, stage: &str) {
    let edge = [[Value::Int64(1), Value::Int64(7), Value::Int64(2)]];
    assert_eq!(
        session.execute(ANCHORED).unwrap().rows(),
        edge,
        "{stage}: GQL"
    );
    assert_eq!(
        session.execute_cypher(ANCHORED).unwrap().rows(),
        edge,
        "{stage}: Cypher"
    );
}

#[test]
fn anchored_edge_survives_a_property_write_on_its_endpoint() {
    for (end, write) in [
        ("source", "MATCH (a:A {id: 1}) SET a.seen = true"),
        ("target", "MATCH (b:B {id: 2}) SET b.seen = true"),
    ] {
        let mut db = GrafeoDB::new_in_memory();
        compacted(&mut db);
        let session = db.session();
        assert_found(&session, "before the write");
        session.execute(write).unwrap();
        assert_found(&session, &format!("after a property write on its {end}"));
        session.execute_cypher(write).unwrap();
        assert_found(&session, &format!("after a Cypher write on its {end}"));
    }
}

#[cfg(all(feature = "grafeo-file", feature = "wal"))]
#[test]
fn anchored_edge_survives_an_unrelated_write_after_reopen() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("anchored.grafeo");
    {
        let mut db = GrafeoDB::open(&path).unwrap();
        compacted(&mut db);
        db.close().unwrap();
    }
    let db = GrafeoDB::open(&path).unwrap();
    let session = db.session();
    assert_found(&session, "after the reopen");
    session
        .execute("MATCH (a:A {id: 1}) INSERT (a)-[:UNRELATED]->(:C {id: 99})")
        .unwrap();
    assert_found(&session, "after an unrelated edge from its source");
    session
        .execute("MATCH (b:B {id: 2}) INSERT (:C {id: 98})-[:UNRELATED]->(b)")
        .unwrap();
    assert_found(&session, "after an unrelated edge to its target");
    db.close().unwrap();

    let db = GrafeoDB::open(&path).unwrap();
    assert_found(&db.session(), "after a second reopen");
    db.close().unwrap();
}

#[test]
fn anchored_edge_inside_a_transaction() {
    let mut db = GrafeoDB::new_in_memory();
    compacted(&mut db);
    let mut session = db.session();
    session.begin_transaction().unwrap();
    session
        .execute("MATCH (a:A {id: 1}) INSERT (a)-[:UNRELATED]->(:C {id: 99})")
        .unwrap();
    session
        .execute("MATCH (b:B {id: 2}) SET b.seen = true")
        .unwrap();
    assert_found(&session, "inside the transaction");
    session.rollback().unwrap();
    assert_found(&session, "after the rollback");
    assert_eq!(
        session
            .execute("MATCH ()-[r:UNRELATED]->() RETURN count(r)")
            .unwrap()
            .rows(),
        [[Value::Int64(0)]],
        "the rollback leaves out the unrelated edge"
    );
}

#[test]
fn anchored_edge_survives_unrelated_writes_with_an_index_on_its_ends() {
    for index_first in [true, false] {
        let mut db = GrafeoDB::new_in_memory();
        if index_first {
            db.create_property_index("id").unwrap();
        }
        compacted(&mut db);
        if !index_first {
            db.create_property_index("id").unwrap();
        }
        let session = db.session();
        let stage = if index_first {
            "index before compact"
        } else {
            "index after compact"
        };
        assert_found(&session, stage);
        session
            .execute("MATCH (a:A {id: 1}) INSERT (a)-[:UNRELATED]->(:C {id: 99})")
            .unwrap();
        session
            .execute("MATCH (b:B {id: 2}) SET b.seen = true")
            .unwrap();
        assert_found(&session, &format!("{stage}, after the writes"));
    }
}
