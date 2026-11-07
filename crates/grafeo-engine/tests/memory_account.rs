//! Everything the database holds takes its memory from the one buffer manager (#751, #434), and
//! `memory_usage()` reports what the grants hold. The walks over the store's parts
//! (`RowGroupStore::memory_breakdown`) stay as the oracle: after any workload each component the
//! report shows equals its walk, and the buffer manager's regions hold at least the counted
//! capacity (a growing part extends its grant a fixed step at a time, so the grant rounds up).
//! In 0.6.2 deleting does not lower the account: deleted rows are not reclaimed until 0.6.3.
//!
//! ```bash
//! cargo test -p grafeo-engine --all-features --test main memory_account::
//! ```

#![cfg(all(feature = "gql", feature = "grafeo-file"))]
#![allow(
    deprecated,
    reason = "the oracle walks the raw store, which no session shows"
)]

use grafeo_engine::GrafeoDB;

/// The account against the walk: each component's report equals its walk, and the grants of
/// the graph storage and index regions cover the four together.
fn assert_account_matches_walk(db: &GrafeoDB, what: &str) {
    let (store, indexes, mvcc, strings) = db.store().unwrap().memory_breakdown();
    let usage = db.memory_usage();
    assert_eq!(usage.store.total_bytes, store.total_bytes, "{what}: store");
    assert_eq!(
        usage.indexes.total_bytes, indexes.total_bytes,
        "{what}: indexes"
    );
    assert_eq!(usage.mvcc.total_bytes, mvcc.total_bytes, "{what}: versions");
    assert_eq!(
        usage.string_pool.total_bytes, strings.total_bytes,
        "{what}: string pool"
    );
    let walked = store.total_bytes + indexes.total_bytes + mvcc.total_bytes + strings.total_bytes;
    let granted =
        usage.buffer_manager.graph_storage_bytes + usage.buffer_manager.index_buffers_bytes;
    assert!(
        granted >= walked,
        "{what}: {granted} bytes granted for {walked} bytes held"
    );
}

/// The writes each case makes, on a fresh database.
fn workloads() -> Vec<(&'static str, fn(&GrafeoDB))> {
    vec![
        ("an empty database", |_| {}),
        ("strings", |db| {
            db.execute("UNWIND range(1, 2000) AS i INSERT (:Person {name: 'person ' || CAST(i AS STRING)})")
                .unwrap();
        }),
        ("long strings", |db| {
            db.execute(&format!(
                "UNWIND range(1, 200) AS i INSERT (:Note {{text: '{}'}})",
                "x".repeat(4_000)
            ))
            .unwrap();
        }),
        ("lists and maps", |db| {
            db.execute(
                "UNWIND range(1, 1000) AS i \
                 INSERT (:Bag {items: [i, i + 1, i + 2], meta: {k: i, s: 'v'}})",
            )
            .unwrap();
        }),
        ("vectors", |db| {
            db.execute(
                "UNWIND range(1, 500) AS i INSERT (:Doc {emb: vector([1.0, 0.5, 0.25, 0.125])})",
            )
            .unwrap();
        }),
        ("edges and labels", |db| {
            db.execute("UNWIND range(1, 1000) AS i INSERT (:A {i: i})-[:K {w: i}]->(:B {i: i})")
                .unwrap();
            db.execute("MATCH (a:A) WHERE a.i % 2 = 0 SET a:Even")
                .unwrap();
        }),
        ("updates kept for readers", |db| {
            db.execute("UNWIND range(1, 500) AS i INSERT (:C {v: i})")
                .unwrap();
            for round in 0..5 {
                db.execute(&format!("MATCH (c:C) SET c.v = c.v + {round}"))
                    .unwrap();
            }
        }),
        ("a merge", |db| {
            db.execute("UNWIND range(1, 500) AS i MERGE (:M {k: i % 50})")
                .unwrap();
        }),
        ("a property index over existing data", |db| {
            db.execute("UNWIND range(1, 2000) AS i INSERT (:Person {id: i})")
                .unwrap();
            db.create_property_index("id").unwrap();
        }),
    ]
}

#[test]
fn the_account_equals_the_walk_after_each_workload() {
    for (what, workload) in workloads() {
        let db = GrafeoDB::new_in_memory();
        workload(&db);
        assert_account_matches_walk(&db, what);
    }
}

/// All the workloads on one database, one after another: the account follows each step.
#[test]
fn the_account_follows_a_sequence_of_workloads() {
    let db = GrafeoDB::new_in_memory();
    for (what, workload) in workloads() {
        workload(&db);
        assert_account_matches_walk(&db, what);
    }
}

#[cfg(feature = "vector-index")]
#[test]
fn a_vector_index_is_counted() {
    let db = GrafeoDB::new_in_memory();
    db.execute("UNWIND range(1, 500) AS i INSERT (:Doc {emb: vector([1.0, 0.5, 0.25, 0.125])})")
        .unwrap();
    db.create_vector_index("Doc", "emb", None, None, None, None, None)
        .unwrap();
    assert_account_matches_walk(&db, "a vector index");
}

#[cfg(feature = "text-index")]
#[test]
fn a_text_index_is_counted() {
    let db = GrafeoDB::new_in_memory();
    db.execute(
        "UNWIND range(1, 500) AS i INSERT (:Doc {body: 'graph number ' || CAST(i AS STRING)})",
    )
    .unwrap();
    db.create_text_index("Doc", "body").unwrap();
    assert_account_matches_walk(&db, "a text index");
}

/// A reopened database holds its cold chunks through grants, and a read that thaws a column
/// takes a grant for the hot form.
#[test]
fn a_reopened_database_and_a_thaw_are_counted() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("amsterdam.grafeo");
    let db = GrafeoDB::open(&path).unwrap();
    db.execute(
        "UNWIND range(1, 5000) AS i INSERT (:Person {i: i, name: 'p' || CAST(i AS STRING)})",
    )
    .unwrap();
    db.close().unwrap();
    drop(db);

    let db = GrafeoDB::open(&path).unwrap();
    assert_account_matches_walk(&db, "reopened, cold");
    let found = db
        .execute("MATCH (p:Person) WHERE p.name = 'p77' RETURN p.i")
        .unwrap();
    assert_eq!(found.rows().len(), 1);
    assert_account_matches_walk(&db, "after a read");
    db.close().unwrap();
}

/// 2,000 people with their number `i`.
fn people() -> GrafeoDB {
    let db = GrafeoDB::new_in_memory();
    db.execute(
        "UNWIND range(1, 2000) AS i          INSERT (:Person {i: i, name: 'person ' || CAST(i AS STRING)})",
    )
    .unwrap();
    db
}

/// A delete of half the rows: the collect after it may free what the deleted rows held (their
/// values and version stamps), so the account may drop, but it equals the walk.
#[test]
fn deleting_half_the_rows_keeps_the_account_equal_to_the_walk() {
    let db = people();
    db.execute("MATCH (p:Person) WHERE p.i % 2 = 0 DETACH DELETE p")
        .unwrap();
    assert_account_matches_walk(&db, "after deleting half");
}

/// A rolled-back statement (an insert, a delete of every row) leaves the account no lower than
/// before, and equal to the walk.
#[test]
fn a_rolled_back_statement_does_not_lower_the_account() {
    let db = people();
    for statement in [
        "UNWIND range(1, 500) AS i INSERT (:Person {i: 5000 + i})",
        "MATCH (p:Person) DETACH DELETE p",
    ] {
        let before = db.memory_usage();
        let mut session = db.session();
        session.begin_transaction().unwrap();
        session.execute(statement).unwrap();
        session.rollback().unwrap();
        let after = db.memory_usage();
        assert!(
            after.store.total_bytes >= before.store.total_bytes,
            "{statement}: {} after the rollback, {} before",
            after.store.total_bytes,
            before.store.total_bytes
        );
        assert_account_matches_walk(&db, &format!("{statement}, rolled back"));
    }
}

/// A delete that empties every group lets the collect drop the groups whole: the account
/// follows the walk down.
#[test]
fn the_account_equals_the_walk_after_a_delete_that_empties_every_group() {
    let db = people();
    db.execute("MATCH (p:Person) DETACH DELETE p").unwrap();
    assert_account_matches_walk(&db, "after deleting every row");
}
