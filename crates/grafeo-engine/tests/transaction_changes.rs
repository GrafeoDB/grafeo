//! A transaction's commit and rollback touch only what it changed (#410).
//!
//! Begin, commit and rollback used to scan every node and edge version chain,
//! so their cost grew with the size of the database. Each graph store now
//! keeps a per-transaction log of what the transaction created, deleted and
//! changed, and commit and rollback walk that log. These tests pin the
//! results (a rollback leaves counts and planner statistics as they were, a
//! savepoint undoes only its own transaction's later writes) and the cost (a
//! write in a large database costs about what it costs in a small one).

use std::time::Instant;

use grafeo_common::types::Value;
use grafeo_engine::GrafeoDB;

fn int(db: &GrafeoDB, query: &str) -> i64 {
    match db.execute(query).unwrap().rows()[0][0] {
        Value::Int64(n) => n,
        ref other => panic!("{query}: expected an integer, got {other:?}"),
    }
}

fn names(db: &GrafeoDB) -> Vec<String> {
    db.execute("MATCH (n:Person) RETURN n.name ORDER BY n.name")
        .unwrap()
        .rows()
        .iter()
        .map(|row| match &row[0] {
            Value::String(s) => s.to_string(),
            other => panic!("unexpected name {other:?}"),
        })
        .collect()
}

/// Planner statistics: total nodes and edges, `:Person` and `:Admin` nodes,
/// `:KNOWS` edges.
fn statistics(db: &GrafeoDB) -> (u64, u64, u64, u64, u64) {
    let store = db.store();
    store.ensure_statistics_fresh();
    let stats = store.statistics();
    let label = |name: &str| stats.get_label(name).map_or(0, |l| l.node_count);
    (
        stats.total_nodes,
        stats.total_edges,
        label("Person"),
        label("Admin"),
        stats.get_edge_type("KNOWS").map_or(0, |t| t.edge_count),
    )
}

fn alix_knows_gus() -> GrafeoDB {
    let db = GrafeoDB::new_in_memory();
    db.execute("INSERT (:Person {name: 'Alix'})-[:KNOWS]->(:Person {name: 'Gus'})")
        .unwrap();
    db.create_property_index("name");
    db
}

#[test]
fn rollback_restores_counts_and_statistics() {
    let db = alix_knows_gus();
    let before = statistics(&db);
    assert_eq!(before, (2, 1, 2, 0, 1));

    let mut session = db.session();
    session.begin_transaction().unwrap();
    session
        .execute("INSERT (:Person {name: 'Vincent'})")
        .unwrap();
    session
        .execute(
            "MATCH (a:Person {name: 'Alix'}), (v:Person {name: 'Vincent'}) \
             INSERT (a)-[:KNOWS]->(v)",
        )
        .unwrap();
    session
        .execute("MATCH (g:Person {name: 'Gus'}) DETACH DELETE g")
        .unwrap();
    session
        .execute("MATCH (a:Person {name: 'Alix'}) SET a:Admin, a.age = 30")
        .unwrap();
    session.rollback().unwrap();

    assert_eq!(statistics(&db), before);
    assert_eq!((db.node_count(), db.edge_count()), (2, 1));
    assert_eq!(names(&db), ["Alix", "Gus"]);
    assert_eq!(
        int(&db, "MATCH (:Person)-[k:KNOWS]->(:Person) RETURN count(k)"),
        1
    );
    assert_eq!(int(&db, "MATCH (n:Admin) RETURN count(n)"), 0);
    assert!(
        db.find_nodes_by_property("name", &Value::from("Vincent"))
            .is_empty()
    );
    assert_eq!(
        db.execute("MATCH (a:Person {name: 'Alix'}) RETURN a.age")
            .unwrap()
            .rows()[0][0],
        Value::Null
    );
}

#[test]
fn a_node_created_and_deleted_in_one_transaction_leaves_nothing() {
    for commit in [false, true] {
        let db = alix_knows_gus();
        let before = statistics(&db);
        let mut session = db.session();
        session.begin_transaction().unwrap();
        session
            .execute("INSERT (:Person {name: 'Vincent'})-[:KNOWS]->(:Person {name: 'Jules'})")
            .unwrap();
        session
            .execute("MATCH (n:Person) WHERE n.name IN ['Vincent', 'Jules'] DETACH DELETE n")
            .unwrap();
        if commit {
            session.commit().unwrap();
        } else {
            session.rollback().unwrap();
        }
        assert_eq!(statistics(&db), before, "commit: {commit}");
        assert_eq!(names(&db), ["Alix", "Gus"], "commit: {commit}");
    }
}

/// `DELETE` used to skip nodes and edges created earlier in the same
/// transaction, so they survived the commit.
#[test]
fn a_transaction_can_delete_what_it_created() {
    let db = GrafeoDB::new_in_memory();
    let mut session = db.session();
    session.begin_transaction().unwrap();
    session
        .execute("INSERT (:Person {name: 'Vincent'})-[:KNOWS]->(:Person {name: 'Jules'})")
        .unwrap();
    session
        .execute("MATCH (:Person)-[k:KNOWS]->(:Person) DELETE k")
        .unwrap();
    session
        .execute("MATCH (n:Person {name: 'Vincent'}) DELETE n")
        .unwrap();
    let inside = session.execute("MATCH (n:Person) RETURN n.name").unwrap();
    assert_eq!(inside.rows(), [vec![Value::from("Jules")]]);
    session.commit().unwrap();

    assert_eq!(names(&db), ["Jules"]);
    assert_eq!((db.node_count(), db.edge_count()), (1, 0));
}

#[test]
fn committed_writes_are_counted_once() {
    let db = alix_knows_gus();
    let mut session = db.session();
    session.begin_transaction().unwrap();
    session
        .execute("INSERT (:Person {name: 'Vincent'})")
        .unwrap();
    session
        .execute("MATCH (g:Person {name: 'Gus'}) DETACH DELETE g")
        .unwrap();
    session
        .execute("MATCH (a:Person {name: 'Alix'}) SET a:Admin")
        .unwrap();
    session.commit().unwrap();

    assert_eq!(statistics(&db), (2, 0, 2, 1, 0));
    assert_eq!(names(&db), ["Alix", "Vincent"]);
}

/// Ids are handed out by the store, so another transaction's nodes land
/// between the savepoint's ids: rolling back to the savepoint must keep them.
#[test]
fn savepoint_undoes_only_its_own_transactions_later_writes() {
    let db = GrafeoDB::new_in_memory();
    let mut first = db.session();
    let mut second = db.session();
    first.begin_transaction().unwrap();
    second.begin_transaction().unwrap();

    first.execute("INSERT (:Person {name: 'Alix'})").unwrap();
    first.savepoint("before_vincent").unwrap();
    second.execute("INSERT (:Person {name: 'Gus'})").unwrap();
    first.execute("INSERT (:Person {name: 'Vincent'})").unwrap();
    first
        .execute(
            "MATCH (a:Person {name: 'Alix'}), (v:Person {name: 'Vincent'}) \
             INSERT (a)-[:KNOWS]->(v)",
        )
        .unwrap();
    first
        .execute("MATCH (a:Person {name: 'Alix'}) SET a.age = 30")
        .unwrap();
    second.execute("INSERT (:Person {name: 'Jules'})").unwrap();
    first.rollback_to_savepoint("before_vincent").unwrap();
    first.commit().unwrap();
    second.commit().unwrap();

    assert_eq!(names(&db), ["Alix", "Gus", "Jules"]);
    assert_eq!(statistics(&db), (3, 0, 3, 0, 0));
    assert_eq!(
        db.execute("MATCH (a:Person {name: 'Alix'}) RETURN a.age")
            .unwrap()
            .rows()[0][0],
        Value::Null
    );
}

#[test]
fn prepared_commit_reports_the_entities_written() {
    let db = alix_knows_gus();
    let mut session = db.session();
    session.begin_transaction().unwrap();
    session
        .execute("INSERT (:Person {name: 'Vincent'})-[:KNOWS]->(:Person {name: 'Jules'})")
        .unwrap();
    session
        .execute("MATCH (a:Person {name: 'Alix'}) SET a.age = 30")
        .unwrap();

    let prepared = session.prepare_commit().unwrap();
    let info = prepared.info();
    // Vincent, Jules and Alix; the new KNOWS edge.
    assert_eq!((info.nodes_written, info.edges_written), (3, 1));
    prepared.commit().unwrap();
}

/// Fastest of `runs` timings of `write`, in microseconds.
fn fastest_micros(runs: usize, mut write: impl FnMut()) -> u128 {
    (0..runs)
        .map(|_| {
            let start = Instant::now();
            write();
            start.elapsed().as_micros()
        })
        .min()
        .unwrap()
}

/// One-node writes in a database of `size` nodes: auto-commit, explicit
/// commit and rollback, each the fastest of 20 runs, in microseconds.
fn write_costs(size: i64) -> [u128; 3] {
    let db = GrafeoDB::new_in_memory();
    // Batches of 10,000: with `tiered-storage`, one transaction's versions
    // must fit in one arena chunk.
    for first in (1..=size).step_by(10_000) {
        let last = (first + 9_999).min(size);
        db.execute(&format!(
            "UNWIND range({first}, {last}) AS i INSERT (:P {{i: i}})"
        ))
        .unwrap();
    }
    let mut session = db.session();
    let auto_commit = fastest_micros(20, || {
        session.execute("INSERT (:Y {v: 1})").unwrap();
    });
    let commit = fastest_micros(20, || {
        session.begin_transaction().unwrap();
        session.execute("INSERT (:Y {v: 1})").unwrap();
        session.commit().unwrap();
    });
    // The statement after a rollback is timed too: it pays for anything the
    // rollback left to recompute.
    let rollback = fastest_micros(20, || {
        session.begin_transaction().unwrap();
        session.execute("INSERT (:Y {v: 1})").unwrap();
        session.rollback().unwrap();
        session.execute("MATCH (y:Y) RETURN count(y)").unwrap();
    });
    [auto_commit, commit, rollback]
}

#[test]
fn write_cost_does_not_grow_with_the_database() {
    let small = write_costs(1_000);
    let large = write_costs(200_000);
    for (name, (small, large)) in ["auto-commit", "commit", "rollback"]
        .into_iter()
        .zip(small.into_iter().zip(large))
    {
        // A scan of every version chain makes the large case more than
        // 20 times slower; a write that touches only its own changes stays
        // within a small factor.
        assert!(
            large <= small.max(100) * 5,
            "{name}: {large} us at 200,000 nodes against {small} us at 1,000"
        );
    }
}
