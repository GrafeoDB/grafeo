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
        "UNWIND range(1, 2000) AS i INSERT (:Person {i: i, name: 'person ' || CAST(i AS STRING)})",
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

// ---------------------------------------------------------------------------
// RDF, CDC and change sets (#751 patch 4)
// ---------------------------------------------------------------------------

/// The bytes the LPG store's walk finds, its four components together.
#[cfg(all(feature = "triple-store", feature = "sparql"))]
fn lpg_walked(db: &GrafeoDB) -> usize {
    let (store, indexes, mvcc, strings) = db.store().unwrap().memory_breakdown();
    store.total_bytes + indexes.total_bytes + mvcc.total_bytes + strings.total_bytes
}

/// The RDF store's account against its walk: no line differs, the report's `rdf` part equals
/// the walk, and the buffer manager's grants cover it with the LPG store's bytes.
#[cfg(all(feature = "triple-store", feature = "sparql"))]
fn assert_rdf_account_matches_walk(db: &GrafeoDB, what: &str) {
    let mismatches = db.rdf_store().account_mismatches();
    assert!(
        mismatches.is_empty(),
        "{what}: the RDF account's lines differ from the walk (line, counted, walked): {mismatches:?}"
    );
    let (_, triples_and_indexes, dictionary, ring, _) = db.rdf_store().heap_memory_bytes();
    let walked = triples_and_indexes + dictionary + ring;
    let usage = db.memory_usage();
    assert_eq!(
        usage.rdf.total_bytes, walked,
        "{what}: the report's rdf part"
    );
    let held = lpg_walked(db) + walked;
    assert!(
        usage.buffer_manager.allocated_bytes >= held,
        "{what}: {} bytes granted for {held} bytes held",
        usage.buffer_manager.allocated_bytes
    );
}

/// `count` N-Triples statements naming people of Lyon, from `from` on.
#[cfg(all(feature = "triple-store", feature = "sparql"))]
fn lyon_people(from: usize, count: usize) -> String {
    use std::fmt::Write;

    let mut text = String::new();
    for i in from..from + count {
        write!(
            text,
            "<http://example.org/person/{i}> <http://example.org/name> \"person {i} of Lyon\" . "
        )
        .unwrap();
    }
    text
}

/// The RDF account follows every way triples come and go: SPARQL `INSERT DATA` into the default
/// and a named graph, `batch_insert_rdf`, `load_rdf`, `DELETE DATA`, `DROP GRAPH`, and a reopen.
#[cfg(all(feature = "triple-store", feature = "sparql"))]
#[test]
fn the_rdf_account_equals_its_walk() {
    use grafeo_core::graph::rdf::{Term, Triple};

    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("lyon.grafeo");
    let db = GrafeoDB::open(&path).unwrap();
    assert_rdf_account_matches_walk(&db, "empty");

    db.execute_sparql(&format!("INSERT DATA {{ {} }}", lyon_people(0, 300)))
        .unwrap();
    assert_rdf_account_matches_walk(&db, "INSERT DATA");
    db.execute_sparql(&format!(
        "INSERT DATA {{ GRAPH <http://example.org/g> {{ {} }} }}",
        lyon_people(0, 200)
    ))
    .unwrap();
    assert_rdf_account_matches_walk(&db, "INSERT DATA into a named graph");

    let knows = |from: usize| {
        Triple::new(
            Term::iri(format!("http://example.org/person/{from}")),
            Term::iri("http://example.org/knows"),
            Term::iri(format!("http://example.org/person/{}", from + 1)),
        )
    };
    db.batch_insert_rdf((0..500).map(knows)).unwrap();
    assert_rdf_account_matches_walk(&db, "batch_insert_rdf");

    let file = dir.path().join("cities.nt");
    let mut lines = String::new();
    for i in 0..400 {
        use std::fmt::Write;

        writeln!(
            lines,
            "<http://example.org/city/{i}> <http://example.org/label> \"city {i}\" ."
        )
        .unwrap();
    }
    std::fs::write(&file, lines).unwrap();
    db.load_rdf(&file, None).unwrap();
    assert_rdf_account_matches_walk(&db, "load_rdf");

    db.execute_sparql(&format!("DELETE DATA {{ {} }}", lyon_people(0, 150)))
        .unwrap();
    assert_rdf_account_matches_walk(&db, "DELETE DATA");
    db.execute_sparql("DROP GRAPH <http://example.org/g>")
        .unwrap();
    assert_rdf_account_matches_walk(&db, "DROP GRAPH");
    db.close().unwrap();
    drop(db);

    let db = GrafeoDB::open(&path).unwrap();
    assert_rdf_account_matches_walk(&db, "reopened");
    db.close().unwrap();
}

/// With CDC on, the log's counted bytes equal its walk after commits and after a collect that
/// prunes it to its retention, and the report's `cdc` part equals the walk.
#[cfg(feature = "cdc")]
#[test]
fn the_cdc_account_equals_its_walk() {
    use grafeo_engine::Config;
    use grafeo_engine::cdc::CdcRetentionConfig;

    let check = |db: &GrafeoDB, what: &str| {
        let mismatches = db.cdc_log().account_mismatches();
        assert!(
            mismatches.is_empty(),
            "{what}: the CDC account differs from the walk: {mismatches:?}"
        );
        let (walked, _, events) = db.cdc_log().heap_memory_bytes();
        let usage = db.memory_usage();
        assert_eq!(
            usage.cdc.total_bytes, walked,
            "{what}: the report's cdc part"
        );
        events
    };
    let db = GrafeoDB::with_config(
        Config::in_memory()
            .with_cdc()
            .with_cdc_retention(CdcRetentionConfig::unlimited().with_max_events(200)),
    )
    .unwrap();
    check(&db, "empty");
    for round in 0..20 {
        db.execute(&format!(
            "UNWIND range(1, 50) AS i INSERT (:Person {{round: {round}, name: 'person ' || CAST(i AS STRING)}})"
        ))
        .unwrap();
        db.execute(&format!(
            "MATCH (p:Person {{round: {round}}}) SET p.city = 'Lyon'"
        ))
        .unwrap();
    }
    let before = check(&db, "after the commits");
    db.gc();
    let after = check(&db, "after the collect");
    assert!(
        after < before,
        "the collect pruned the log to its retention: {before} events before, {after} after"
    );
}

/// A transaction's change set holds a grant of the execution buffers while it is open (grants
/// move a step at a time, so the writes cross at least one), and gives it back at the commit and
/// at the rollback.
#[test]
fn a_change_set_holds_its_grant_until_the_transaction_ends() {
    use grafeo_common::memory::buffer::GRANT_STEP;

    let db = GrafeoDB::new_in_memory();
    let held = |db: &GrafeoDB| db.memory_usage().buffer_manager.execution_buffers_bytes;
    // 3,000 rows of 1,000-byte strings: past a grant step.
    let write = format!(
        "UNWIND range(1, 3000) AS i INSERT (:Person {{name: CAST(i AS STRING) || '{}'}})",
        "x".repeat(1_000)
    );
    for end in ["commit", "rollback"] {
        let before = held(&db);
        let mut session = db.session();
        session.begin_transaction().unwrap();
        session.execute(&write).unwrap();
        let during = held(&db);
        assert!(
            during >= before + GRANT_STEP,
            "{end}: the change set holds at least one grant step ({before} before, {during} with \
             the writes)"
        );
        if end == "commit" {
            session.commit().unwrap();
        } else {
            session.rollback().unwrap();
        }
        assert_eq!(held(&db), before, "{end}: the grant is given back");
    }
}

/// A database on a store of its own reports what it holds besides the store (its CDC log, RDF
/// store and ring) to the buffer manager, as one on the built-in store does, and its CDC log
/// keeps the configured retention.
#[cfg(feature = "cdc")]
#[test]
fn a_database_on_its_own_store_registers_its_consumers_and_keeps_the_cdc_retention() {
    use std::sync::Arc;

    use grafeo_core::graph::rowgroup::RowGroupStore;
    use grafeo_core::graph::{GraphBackend, GraphStore};
    use grafeo_engine::Config;
    use grafeo_engine::cdc::CdcRetentionConfig;

    let config = || {
        Config::in_memory()
            .with_cdc()
            .with_cdc_retention(CdcRetentionConfig::unlimited().with_max_events(3))
    };
    let names = |db: &GrafeoDB| -> Vec<String> {
        db.buffer_manager()
            .consumer_stats()
            .into_iter()
            .map(|consumer| consumer.name)
            .collect()
    };
    let db = GrafeoDB::with_store(
        Arc::new(RowGroupStore::new()) as Arc<dyn GraphBackend>,
        config(),
    )
    .unwrap();
    let registered = names(&db);
    assert!(
        registered.iter().any(|name| name == "cdc_log"),
        "{registered:?}"
    );
    for name in ["Vincent", "Jules", "Mia", "Butch", "Alix"] {
        db.execute(&format!("INSERT (:Person {{name: '{name}'}})"))
            .unwrap();
    }
    assert_eq!(db.cdc_log().event_count(), 5);
    db.gc();
    assert!(
        db.cdc_log().event_count() <= 3,
        "the configured retention: {} events",
        db.cdc_log().event_count()
    );

    let read = GrafeoDB::with_read_store(
        Arc::new(RowGroupStore::new()) as Arc<dyn GraphStore>,
        config(),
    )
    .unwrap();
    let registered = names(&read);
    assert!(
        registered.iter().any(|name| name == "cdc_log"),
        "{registered:?}"
    );
}
