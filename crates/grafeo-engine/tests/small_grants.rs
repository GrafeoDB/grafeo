//! Small holders of memory each hold a grant of about their own size: small named graphs, write
//! transactions open at once and queries running at once leave room under the limit for each
//! other and for the next write.
//!
//! ```bash
//! cargo test -p grafeo-engine --all-features --test main small_grants::
//! ```

#![cfg(feature = "gql")]

use grafeo_common::types::Value;
use grafeo_engine::{Config, GrafeoDB};

const MIB: usize = 1024 * 1024;

fn database() -> GrafeoDB {
    GrafeoDB::with_config(Config::in_memory().with_memory_limit(64 * MIB)).unwrap()
}

/// The bytes the database's buffer manager holds.
fn allocated(db: &GrafeoDB) -> usize {
    db.memory_usage().buffer_manager.allocated_bytes
}

/// A write and a read in the default graph are still admitted under the limit.
fn assert_admitted(db: &GrafeoDB) {
    db.execute("INSERT (:After {name: 'Butch'})")
        .expect("a write after the others is admitted");
    let result = db
        .execute("MATCH (a:After) RETURN count(a)")
        .expect("a read after the others is admitted");
    assert_eq!(result.rows(), [[Value::Int64(1)]], "the write landed");
}

#[test]
fn many_small_named_graphs_leave_room() {
    const GRAPHS: usize = 200;
    let db = database();
    let before = allocated(&db);
    let session = db.session();
    for k in 0..GRAPHS {
        session
            .execute(&format!("CREATE GRAPH g{k}"))
            .unwrap_or_else(|error| panic!("graph {k}: {error}"));
        session.execute(&format!("USE GRAPH g{k}")).unwrap();
        session
            .execute(&format!("INSERT (:Person {{name: 'Alix', k: {k}}})"))
            .unwrap_or_else(|error| panic!("node in graph {k}: {error}"));
    }
    let grown = allocated(&db).saturating_sub(before);
    assert!(
        grown < 16 * MIB,
        "{GRAPHS} one-node named graphs hold {grown} bytes"
    );
    assert_admitted(&db);
}

#[test]
fn many_open_write_transactions_leave_room() {
    const OPEN: usize = 64;
    let db = database();
    let before = allocated(&db);
    let mut sessions: Vec<_> = (0..OPEN).map(|_| db.session()).collect();
    for (k, session) in sessions.iter_mut().enumerate() {
        session.begin_transaction().unwrap();
        session
            .execute(&format!("INSERT (:Person {{name: 'Gus', k: {k}}})"))
            .unwrap_or_else(|error| panic!("transaction {k}: {error}"));
    }
    let grown = allocated(&db).saturating_sub(before);
    assert!(
        grown < 16 * MIB,
        "{OPEN} open one-node transactions hold {grown} bytes"
    );
    assert_admitted(&db);
    for session in &mut sessions {
        session.commit().unwrap();
    }
    let result = db.execute("MATCH (p:Person) RETURN count(p)").unwrap();
    assert_eq!(result.rows(), [[Value::Int64(64)]]);
}

#[test]
fn many_small_queries_at_once_each_answer() {
    const QUERIES: usize = 64;
    const PEOPLE: i64 = 20_000;
    const ROUNDS: usize = 8;
    let db = database();
    db.execute(&format!(
        "UNWIND range(0, {}) AS k INSERT (:Person {{name: 'Mia', k: k}})",
        PEOPLE - 1
    ))
    .unwrap();
    let start = std::sync::Barrier::new(QUERIES);
    std::thread::scope(|scope| {
        let handles: Vec<_> = (0..QUERIES)
            .map(|q| {
                let (db, start) = (&db, &start);
                scope.spawn(move || {
                    let session = db.session();
                    start.wait();
                    // Rounds of a sort and a grouping, so the queries' operator grants overlap.
                    let mut answers = Vec::new();
                    for round in 0..ROUNDS {
                        let sorted = session
                            .execute("MATCH (p:Person) RETURN p.k AS k ORDER BY k DESC LIMIT 3")
                            .unwrap_or_else(|error| panic!("sort {q}, round {round}: {error}"));
                        let grouped = session
                            .execute(
                                "MATCH (p:Person) RETURN p.k % 500 AS g, count(p) AS c ORDER BY g",
                            )
                            .unwrap_or_else(|error| panic!("grouping {q}, round {round}: {error}"));
                        answers.push((sorted.rows().to_vec(), grouped.rows().len()));
                    }
                    answers
                })
            })
            .collect();
        for handle in handles {
            for (sorted, groups) in handle.join().expect("a query thread") {
                assert_eq!(
                    sorted,
                    [
                        [Value::Int64(PEOPLE - 1)],
                        [Value::Int64(PEOPLE - 2)],
                        [Value::Int64(PEOPLE - 3)]
                    ]
                );
                assert_eq!(groups, 500);
            }
        }
    });
    assert_admitted(&db);
}
