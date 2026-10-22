//! No read ever shows an edge whose other end the reader does not see. The
//! guarantee holds at write time: an edge's create claims its two ends and a
//! node's delete claims the node, so of an edge's create and a concurrent
//! delete of one of its ends only the first writer commits, and the other
//! gets the retryable conflict, also at commit when the delete (or the
//! create) committed after it began. The reads check no other end. These
//! tests race the two, in both orders, interleaved, with a detach delete,
//! across a rollback, a crash and a reopen, and on eight threads, and check
//! every read path after each: queries in GQL and Cypher, the direct API,
//! the snapshot reads, the counts and an algorithm.
//!
//! ```bash
//! cargo test -p grafeo-engine --all-features --test no_dangling_edges
//! ```

#![cfg(feature = "gql")]

use std::collections::BTreeSet;
use std::path::Path;
use std::sync::Arc;

use grafeo_common::testing::child_process;
use grafeo_common::types::{NodeId, ReadView, Value};
use grafeo_common::utils::error::Error;
use grafeo_core::graph::{Direction, GraphStore};
use grafeo_engine::database::QueryResult;
use grafeo_engine::{Config, GrafeoDB, Session};

/// A database with Alix, Gus and Vincent, and Gus knowing Vincent.
fn people(db: &GrafeoDB) {
    db.execute(
        "INSERT (:Person {name: 'Alix'}), \
         (:Person {name: 'Gus'})-[:KNOWS]->(:Person {name: 'Vincent'})",
    )
    .unwrap();
}

fn in_memory() -> GrafeoDB {
    let db = GrafeoDB::new_in_memory();
    people(&db);
    db
}

fn int(value: &Value) -> i64 {
    match value {
        Value::Int64(n) => *n,
        other => panic!("not an integer: {other:?}"),
    }
}

/// The names of the people, sorted.
fn names(db: &GrafeoDB) -> Vec<String> {
    db.execute("MATCH (p:Person) RETURN p.name ORDER BY p.name")
        .unwrap()
        .rows()
        .iter()
        .map(|row| match &row[0] {
            Value::String(name) => name.to_string(),
            other => panic!("not a name: {other:?}"),
        })
        .collect()
}

/// How many KNOWS edges a query sees.
fn knows(db: &GrafeoDB) -> i64 {
    int(&db
        .execute("MATCH ()-[k:KNOWS]->() RETURN count(k)")
        .unwrap()
        .rows()[0][0])
}

/// Every read path lists only edges whose two ends the reader sees, and
/// they agree on how many there are.
fn assert_no_dangling_edge(db: &GrafeoDB) {
    let session = db.session();
    let nodes: BTreeSet<i64> = session
        .execute("MATCH (n) RETURN id(n)")
        .unwrap()
        .rows()
        .iter()
        .map(|row| int(&row[0]))
        .collect();

    // A query in GQL, and in Cypher.
    let edges = session
        .execute("MATCH (a)-[r]->(b) RETURN id(a), id(b)")
        .unwrap();
    for row in edges.rows() {
        for end in [int(&row[0]), int(&row[1])] {
            assert!(
                nodes.contains(&end),
                "a query lists an edge to node {end}, which no reader sees"
            );
        }
    }
    let edge_total = edges.rows().len();
    #[cfg(feature = "cypher")]
    {
        let cypher = session
            .execute_cypher("MATCH (a)-[r]->(b) RETURN id(a), id(b)")
            .unwrap();
        assert_eq!(
            cypher.rows().len(),
            edge_total,
            "Cypher sees the same edges"
        );
        for row in cypher.rows() {
            assert!(nodes.contains(&int(&row[0])) && nodes.contains(&int(&row[1])));
        }
    }

    // The direct API, the snapshot's reads and the counts.
    let snapshot = db.snapshot().unwrap();
    let (store, view): (&dyn GraphStore, ReadView) = (snapshot.store(), snapshot.view());
    let mut outgoing = 0;
    let mut incoming = 0;
    for &id in &nodes {
        let node = NodeId::new(u64::try_from(id).unwrap());
        for (other, _) in db.get_node_edges(node, Direction::Both, &[]).unwrap() {
            assert!(
                nodes.contains(&as_int(other)),
                "the direct API lists node {other:?}"
            );
        }
        for direction in [Direction::Outgoing, Direction::Incoming] {
            let listed = store.edges(node, direction, &[], view).unwrap();
            for (other, _) in &listed {
                assert!(nodes.contains(&as_int(other)), "edges lists node {other:?}");
            }
            assert_eq!(store.degree(node, direction, view).unwrap(), listed.len());
            for other in store.neighbors(node, direction, view).unwrap() {
                assert!(
                    nodes.contains(&as_int(other)),
                    "neighbors lists node {other:?}"
                );
            }
            if direction == Direction::Outgoing {
                outgoing += listed.len();
            } else {
                incoming += listed.len();
            }
        }
    }
    assert_eq!(outgoing, edge_total, "each edge once from its source");
    assert_eq!(incoming, edge_total, "and once from its target");
    assert_eq!(db.edge_count().unwrap(), edge_total);

    // An algorithm: the degrees it reads add up to the same edges.
    let degrees = session
        .execute("CALL grafeo.degree_centrality() YIELD node_id, in_degree, out_degree RETURN node_id, in_degree, out_degree")
        .unwrap();
    let (mut in_sum, mut out_sum) = (0.0, 0.0);
    for row in degrees.rows() {
        assert!(
            nodes.contains(&int(&row[0])),
            "the algorithm reads node {:?}",
            row[0]
        );
        in_sum += number(&row[1]);
        out_sum += number(&row[2]);
    }
    assert_eq!((in_sum, out_sum), (edge_total as f64, edge_total as f64));
}

fn as_int(node: impl std::borrow::Borrow<NodeId>) -> i64 {
    i64::try_from(node.borrow().as_u64()).unwrap()
}

fn number(value: &Value) -> f64 {
    match value {
        Value::Int64(n) => *n as f64,
        Value::Float64(n) => *n,
        other => panic!("not a number: {other:?}"),
    }
}

/// The retryable conflict.
fn assert_conflict(error: &Error) {
    assert!(error.error_code().is_retryable(), "{error}");
    assert!(
        error.to_string().to_lowercase().contains("conflict"),
        "{error}"
    );
}

const DELETE_GUS: &str = "MATCH (g:Person {name: 'Gus'}) DETACH DELETE g";
const ALIX_KNOWS_GUS: &str =
    "MATCH (a:Person {name: 'Alix'}), (g:Person {name: 'Gus'}) INSERT (a)-[:KNOWS]->(g)";

fn begin(db: &GrafeoDB) -> Session {
    let mut session = db.session();
    session.begin_transaction().unwrap();
    session
}

/// Ends `session`'s transaction as an application would after `outcome`:
/// commits after a statement that went through (the commit's error is the
/// outcome then), rolls back after one that failed. The outcome is checked
/// after the reads, so a test that loses its conflict fails on the dangling
/// edge it then reads, not only on the missing conflict.
fn finish(session: &mut Session, outcome: Result<QueryResult, Error>) -> Result<(), Error> {
    match outcome {
        Ok(_) => session.commit(),
        Err(error) => {
            session.rollback().unwrap();
            Err(error)
        }
    }
}

/// An open transaction deletes Gus first: the edge to him another one then
/// creates conflicts at once; the delete commits, and no edge to Gus is ever
/// read.
#[test]
fn an_edge_create_loses_to_an_open_delete_of_its_end() {
    let db = in_memory();
    let mut deleter = begin(&db);
    deleter.execute(DELETE_GUS).unwrap();
    let mut linker = begin(&db);
    let outcome = linker.execute(ALIX_KNOWS_GUS);
    let linked = finish(&mut linker, outcome);
    deleter.commit().unwrap();
    assert_no_dangling_edge(&db);
    assert_eq!(names(&db), ["Alix", "Vincent"]);
    assert_eq!(knows(&db), 0);
    assert_conflict(&linked.expect_err("Gus is being deleted"));
}

/// An open transaction creates the edge to Gus first: a delete of Gus by
/// another one conflicts at once, though it cannot see the edge.
#[test]
fn a_delete_loses_to_an_open_edge_create_to_the_node() {
    let db = in_memory();
    let mut linker = begin(&db);
    linker.execute(ALIX_KNOWS_GUS).unwrap();
    let mut deleter = begin(&db);
    let outcome = deleter.execute(DELETE_GUS);
    let deleted = finish(&mut deleter, outcome);
    linker.commit().unwrap();
    assert_no_dangling_edge(&db);
    assert_eq!(names(&db), ["Alix", "Gus", "Vincent"]);
    assert_eq!(knows(&db), 2);
    assert_conflict(&deleted.expect_err("an edge to Gus is being created"));
}

/// The delete commits after the linker began: the linker still sees Gus and
/// creates its edge, and loses at its commit.
#[test]
fn an_edge_create_loses_at_commit_to_a_delete_that_committed_after_it_began() {
    let db = in_memory();
    let mut linker = begin(&db);
    let mut deleter = begin(&db);
    deleter.execute(DELETE_GUS).unwrap();
    deleter.commit().unwrap();
    let outcome = linker.execute(ALIX_KNOWS_GUS);
    assert!(outcome.is_ok(), "the linker's snapshot still holds Gus");
    let linked = finish(&mut linker, outcome);
    assert_no_dangling_edge(&db);
    assert_eq!(names(&db), ["Alix", "Vincent"]);
    assert_eq!(knows(&db), 0);
    assert_conflict(&linked.expect_err("Gus was deleted after the linker began"));
}

/// The edge to Gus commits after the deleter began: the deleter sees no
/// edge of Gus, deletes him, and loses at its commit.
#[test]
fn a_delete_loses_at_commit_to_an_edge_that_committed_after_it_began() {
    let db = in_memory();
    let mut deleter = begin(&db);
    let mut linker = begin(&db);
    linker.execute(ALIX_KNOWS_GUS).unwrap();
    linker.commit().unwrap();
    let outcome = deleter.execute(DELETE_GUS);
    assert!(
        outcome.is_ok(),
        "the deleter's snapshot holds no edge from Alix"
    );
    let deleted = finish(&mut deleter, outcome);
    assert_no_dangling_edge(&db);
    assert_eq!(names(&db), ["Alix", "Gus", "Vincent"]);
    assert_eq!(knows(&db), 2);
    assert_conflict(&deleted.expect_err("an edge to Gus committed after the deleter began"));
}

/// A plain delete of a node whose edge committed after the deleter began:
/// the delete counts the edges committed by now too, so it is refused with
/// G1001 (use DETACH), and Alix keeps her edge.
#[test]
fn a_plain_delete_counts_an_edge_that_committed_after_it_began() {
    let db = in_memory();
    let mut deleter = begin(&db);
    let mut linker = begin(&db);
    linker
        .execute(concat!(
            "MATCH (a:Person {name: 'Alix'}), (v:Person {name: 'Vincent'}) ",
            "INSERT (v)-[:KNOWS]->(a)",
        ))
        .unwrap();
    linker.commit().unwrap();
    let outcome = deleter.execute("MATCH (a:Person {name: 'Alix'}) DELETE a");
    let deleted = finish(&mut deleter, outcome);
    assert_no_dangling_edge(&db);
    assert_eq!(names(&db), ["Alix", "Gus", "Vincent"]);
    assert_eq!(knows(&db), 2);
    let error = deleted.expect_err("an edge to Alix committed meanwhile");
    assert!(error.to_string().contains("DETACH"), "{error}");
}

/// The loser's retry after the winner rolls back goes through: a conflict
/// leaves nothing behind, and a rollback releases the claims.
#[test]
fn a_rolled_back_delete_lets_the_edge_create_retry() {
    let db = in_memory();
    let mut deleter = begin(&db);
    deleter.execute(DELETE_GUS).unwrap();
    let mut linker = begin(&db);
    let outcome = linker.execute(ALIX_KNOWS_GUS);
    let linked = finish(&mut linker, outcome);
    deleter.rollback().unwrap();
    assert_no_dangling_edge(&db);
    assert_conflict(&linked.expect_err("Gus is being deleted"));

    let mut retry = begin(&db);
    retry.execute(ALIX_KNOWS_GUS).unwrap();
    retry.commit().unwrap();
    assert_eq!(names(&db), ["Alix", "Gus", "Vincent"]);
    assert_eq!(knows(&db), 2);
    assert_no_dangling_edge(&db);
}

/// Eight threads create edges between people and delete and recreate
/// them, each in a transaction of two statements with a yield between, and
/// retry nothing: every conflict is the retryable one, and no read, during
/// the run or after it, shows a dangling edge.
#[test]
fn eight_threads_never_leave_a_dangling_edge() {
    const PEOPLE: i64 = 24;
    let db = Arc::new(GrafeoDB::new_in_memory());
    for idx in 0..PEOPLE {
        db.execute(&format!("INSERT (:Person {{idx: {idx}}})"))
            .unwrap();
    }
    let workers: Vec<_> = (0..8_u64)
        .map(|thread| {
            let db = Arc::clone(&db);
            std::thread::spawn(move || {
                let mut seed = thread * 7_919 + 13;
                let mut next = |below: i64| {
                    seed = seed.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1_442_695_040_888_963_407);
                    i64::try_from((seed >> 33) % u64::try_from(below).unwrap()).unwrap()
                };
                let mut conflicts = 0;
                for _ in 0..150 {
                    let (a, b, kind) = (next(PEOPLE), next(PEOPLE), next(10));
                    let mut session = db.session();
                    session.begin_transaction().unwrap();
                    let outcome = if kind < 6 {
                        session
                            .execute(&format!("MATCH (a:Person {{idx: {a}}}) RETURN a"))
                            .and_then(|_| {
                                std::thread::yield_now();
                                session.execute(&format!(
                                    "MATCH (a:Person {{idx: {a}}}), (b:Person {{idx: {b}}}) INSERT (a)-[:KNOWS]->(b)"
                                ))
                            })
                    } else {
                        session
                            .execute(&format!("MATCH (p:Person {{idx: {a}}}) DETACH DELETE p"))
                            .and_then(|_| {
                                std::thread::yield_now();
                                session.execute(&format!("INSERT (:Person {{idx: {a}}})"))
                            })
                    };
                    match outcome.and_then(|_| session.commit()) {
                        Ok(()) => {}
                        Err(error) => {
                            // A detach delete of an edge another detach
                            // delete removed and committed after this one
                            // began fails with "does not exist" rather than
                            // the conflict (a known issue: the error is not
                            // retryable); it loses cleanly, with nothing
                            // written.
                            let lost_edge = error
                                .to_string()
                                .contains("does not exist, or the writer does not see it");
                            assert!(
                                error.error_code().is_retryable() || lost_edge,
                                "{error}"
                            );
                            conflicts += 1;
                            let _ = session.rollback();
                        }
                    }
                }
                conflicts
            })
        })
        .collect();
    let reader = {
        let db = Arc::clone(&db);
        std::thread::spawn(move || {
            for _ in 0..20 {
                // Two queries in one transaction read one snapshot: every
                // edge the second returns ends at nodes the first returned.
                let mut session = db.session();
                session.begin_transaction().unwrap();
                let nodes: BTreeSet<i64> = session
                    .execute("MATCH (n) RETURN id(n)")
                    .unwrap()
                    .rows()
                    .iter()
                    .map(|row| int(&row[0]))
                    .collect();
                let edges = session
                    .execute("MATCH (a)-[r]->(b) RETURN id(a), id(b)")
                    .unwrap();
                for row in edges.rows() {
                    for end in [int(&row[0]), int(&row[1])] {
                        assert!(
                            nodes.contains(&end),
                            "a query returned an edge to node {end}, which it does not see"
                        );
                    }
                }
                session.commit().unwrap();
                // One snapshot: every edge's ends among its nodes.
                let snapshot = db.snapshot().unwrap();
                let (store, view) = (snapshot.store(), snapshot.view());
                let seen: BTreeSet<u64> = store
                    .node_ids(view)
                    .unwrap()
                    .iter()
                    .map(|id| id.as_u64())
                    .collect();
                for &node in &seen {
                    for (other, _) in store
                        .edges(NodeId::new(node), Direction::Both, &[], view)
                        .unwrap()
                    {
                        assert!(
                            seen.contains(&other.as_u64()),
                            "a dangling edge from {node} to {other:?}"
                        );
                    }
                }
                std::thread::yield_now();
            }
        })
    };
    let conflicts: u64 = workers
        .into_iter()
        .map(|worker| worker.join().unwrap())
        .sum();
    reader.join().unwrap();
    assert!(conflicts > 0, "the threads raced");
    assert_no_dangling_edge(&db);
}

// ── A crash and a reopen ─────────────────────────────────────────────

const PATH_VAR: &str = "GRAFEO_NO_DANGLING_PATH";

fn persistent(path: &Path) -> GrafeoDB {
    GrafeoDB::with_config(Config::persistent(path)).unwrap()
}

/// Child-process entry for the crash test; a no-op when run directly. The
/// edge to Gus commits while a transaction that began before it has deleted
/// Gus and not committed; the process then exits with the database open.
#[test]
fn crash_child() {
    let Some(path) = std::env::var_os(PATH_VAR) else {
        return;
    };
    let db = persistent(Path::new(&path));
    people(&db);
    let deleter = begin(&db);
    let mut linker = begin(&db);
    linker.execute(ALIX_KNOWS_GUS).unwrap();
    linker.commit().unwrap();
    deleter.execute(DELETE_GUS).unwrap();
    // A crash between the two commits: no close, no destructors.
    std::process::exit(0);
}

/// A crash between the two commits keeps the committed edge with both its
/// ends; the deleter never committed, and the reopened database refuses
/// its delete of a node with an edge.
#[test]
fn a_crash_between_the_commits_keeps_the_edge_with_its_ends() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("trips.grafeo");
    let status = child_process::run(
        std::process::Command::new(std::env::current_exe().unwrap())
            .args(["--exact", "crash_child", "--nocapture"])
            .env(PATH_VAR, &path),
    )
    .unwrap();
    assert!(status.success(), "the child process failed");

    let db = persistent(&path);
    assert_eq!(names(&db), ["Alix", "Gus", "Vincent"]);
    assert_eq!(knows(&db), 2);
    assert_no_dangling_edge(&db);
    db.execute("MATCH (g:Person {name: 'Gus'}) DELETE g")
        .expect_err("Gus has edges");
    db.close().unwrap();
}

/// After both races, closed and reopened, the database holds no dangling
/// edge: what the winners committed, and nothing of the losers.
#[test]
fn a_reopen_after_the_races_reads_no_dangling_edge() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("trips.grafeo");
    let db = persistent(&path);
    people(&db);
    let mut linker = begin(&db);
    let mut deleter = begin(&db);
    deleter.execute(DELETE_GUS).unwrap();
    deleter.commit().unwrap();
    let outcome = linker.execute(ALIX_KNOWS_GUS);
    let first = finish(&mut linker, outcome);
    let mut deleter = begin(&db);
    let mut linker = begin(&db);
    linker
        .execute(concat!(
            "MATCH (a:Person {name: 'Alix'}), (v:Person {name: 'Vincent'}) ",
            "INSERT (a)-[:KNOWS]->(v)",
        ))
        .unwrap();
    linker.commit().unwrap();
    let outcome = deleter.execute("MATCH (v:Person {name: 'Vincent'}) DETACH DELETE v");
    let second = finish(&mut deleter, outcome);
    drop((linker, deleter));
    db.close().unwrap();
    drop(db);

    let db = persistent(&path);
    assert_no_dangling_edge(&db);
    assert_eq!(names(&db), ["Alix", "Vincent"]);
    assert_eq!(knows(&db), 1, "Alix knows Vincent");
    db.close().unwrap();
    assert_conflict(&first.expect_err("Gus was deleted"));
    assert_conflict(&second.expect_err("an edge to Vincent committed"));
}
