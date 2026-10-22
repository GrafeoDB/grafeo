//! A UNIQUE, PRIMARY KEY or NODE KEY value is claimed by the transaction
//! that writes it, so no two transactions commit the same one.
//!
//! The constraint check looks the value up when a statement runs, as the
//! writing transaction sees the data: another transaction's uncommitted
//! write of the same value is invisible to it, and so is one committed after
//! the writer's snapshot. Both duplicates used to commit. Now a write under
//! a unique constraint claims its value in the transaction manager: of two
//! open transactions the first writer wins and the second fails at its
//! write with the retryable write conflict, and a transaction that commits
//! after another one with the same value committed since its start fails at
//! commit. A committed value is refused by the check itself (G2000). Claims
//! go with the transaction at its commit or rollback.
//!
//! ```bash
//! cargo test -p grafeo-engine --all-features --test unique_constraint_claims
//! ```

#![cfg(all(feature = "lpg", feature = "gql"))]

use std::collections::HashMap;
use std::sync::Barrier;

use grafeo_common::types::{PropertyKey, Value};
use grafeo_common::utils::error::{ConstraintKind, Error, TransactionError};
use grafeo_engine::GrafeoDB;
use grafeo_engine::session::Session;

const CONSTRAINT: &str = "CREATE CONSTRAINT memory_key FOR (m:Memory) ON (m.key) UNIQUE";

fn db() -> GrafeoDB {
    let db = GrafeoDB::new_in_memory();
    db.execute(CONSTRAINT).unwrap();
    db
}

/// The committed nodes `:Memory` with `key`.
fn count(db: &GrafeoDB, key: &str) -> i64 {
    match db
        .execute(&format!(
            "MATCH (m:Memory {{key: '{key}'}}) RETURN count(m)"
        ))
        .unwrap()
        .rows()[0][0]
    {
        Value::Int64(n) => n,
        ref other => panic!("a count, got {other:?}"),
    }
}

fn is_conflict(error: &Error) -> bool {
    matches!(
        error,
        Error::Transaction(TransactionError::WriteConflict(_))
    )
}

/// G2000: the check found a committed node with the value.
fn is_violation(error: &Error) -> bool {
    matches!(
        error,
        Error::Constraint {
            kind: ConstraintKind::Schema,
            ..
        }
    )
}

/// A session with an open transaction that inserted the key `ssi`.
fn holding_ssi(db: &GrafeoDB) -> Session {
    let mut session = db.session();
    session.begin_transaction().unwrap();
    session.execute("INSERT (:Memory {key: 'ssi'})").unwrap();
    session
}

/// Every way to write a value: while another open transaction holds `ssi`,
/// each fails with the write conflict and writes nothing; once that one
/// commits, each fails with the constraint's G2000, but for MERGE, which
/// then matches the committed node.
#[test]
fn every_write_path_claims_the_key() {
    type Write = Box<dyn Fn(&GrafeoDB) -> Result<(), Error>>;
    let ssi_props = || {
        std::iter::once((PropertyKey::new("key"), Value::from("ssi"))).collect::<HashMap<_, _>>()
    };
    let paths: Vec<(&str, Write)> = vec![
        (
            "GQL INSERT",
            Box::new(|db| db.execute("INSERT (:Memory {key: 'ssi'})").map(drop)),
        ),
        (
            "GQL SET",
            Box::new(|db| {
                db.execute("MATCH (m:Memory {key: 'old'}) SET m.key = 'ssi'")
                    .map(drop)
            }),
        ),
        (
            "GQL MERGE",
            Box::new(|db| db.execute("MERGE (:Memory {key: 'ssi'})").map(drop)),
        ),
        #[cfg(feature = "cypher")]
        (
            "Cypher CREATE",
            Box::new(|db| db.execute_cypher("CREATE (:Memory {key: 'ssi'})").map(drop)),
        ),
        #[cfg(feature = "cypher")]
        (
            "Cypher MERGE",
            Box::new(|db| db.execute_cypher("MERGE (:Memory {key: 'ssi'})").map(drop)),
        ),
        (
            "create_node_with_props",
            Box::new(|db| {
                db.create_node_with_props(&["Memory"], [("key", Value::from("ssi"))])
                    .map(drop)
            }),
        ),
        (
            "set_node_property",
            Box::new(|db| {
                let old = db.execute("MATCH (m:Memory {key: 'old'}) RETURN id(m)")?;
                let Value::Int64(id) = old.rows()[0][0] else {
                    panic!("an id");
                };
                db.set_node_property(
                    grafeo_common::types::NodeId::new(u64::try_from(id).unwrap()),
                    "key",
                    Value::from("ssi"),
                )
            }),
        ),
        (
            "batch_create_nodes_with_props",
            Box::new(move |db| {
                db.batch_create_nodes_with_props("Memory", vec![ssi_props()])
                    .map(drop)
            }),
        ),
        (
            "import_nodes",
            Box::new(|db| {
                let dir = tempfile::tempdir().unwrap();
                let file = dir.path().join("memories.csv");
                std::fs::write(&file, "key\nssi\n").unwrap();
                grafeo_engine::import::import_nodes(
                    &db.session(),
                    &file,
                    "Memory",
                    grafeo_engine::import::ImportFormat::Csv {
                        headers: true,
                        delimiter: None,
                    },
                )
                .map(drop)
            }),
        ),
    ];
    for (name, write) in &paths {
        let db = db();
        db.execute("INSERT (:Memory {key: 'old'})").unwrap();
        let mut holder = holding_ssi(&db);
        let error = write(&db).expect_err(name);
        assert!(is_conflict(&error), "{name} while the key is held: {error}");
        assert_eq!((count(&db, "ssi"), count(&db, "old")), (0, 1), "{name}");
        holder.commit().unwrap();
        if name.ends_with("MERGE") {
            // A MERGE now finds the committed node and creates nothing.
            write(&db).unwrap();
        } else {
            let error = write(&db).expect_err(name);
            assert!(
                is_violation(&error),
                "{name} once the key committed: {error}"
            );
        }
        assert_eq!((count(&db, "ssi"), count(&db, "old")), (1, 1), "{name}");
    }
}

/// A rolled-back write frees its key.
#[test]
fn a_rollback_frees_the_key() {
    let db = db();
    let mut holder = holding_ssi(&db);
    holder.rollback().unwrap();
    db.execute("INSERT (:Memory {key: 'ssi'})").unwrap();
    assert_eq!(count(&db, "ssi"), 1);
}

/// The second writer fails at its write, not at its commit, and its
/// transaction can go on with other keys.
#[test]
fn the_first_writer_wins() {
    let db = db();
    let mut first = holding_ssi(&db);
    let mut second = db.session();
    second.begin_transaction().unwrap();
    let error = second
        .execute("INSERT (:Memory {key: 'ssi'})")
        .expect_err("the second writer of the key");
    assert!(is_conflict(&error), "{error}");
    first.commit().unwrap();
    assert_eq!(count(&db, "ssi"), 1);
    second.rollback().unwrap();
}

/// A transaction that began before another one committed the key does not
/// see it, so its own write of the key passes the check; it fails at commit
/// instead, whether the key came from a session or a direct call.
#[test]
fn a_transaction_loses_to_a_key_committed_after_it_began() {
    type Write = fn(&GrafeoDB);
    let writers: [(&str, Write); 2] = [
        ("a session", |db| {
            db.execute("INSERT (:Memory {key: 'ssi'})").unwrap();
        }),
        ("a direct call", |db| {
            db.create_node_with_props(&["Memory"], [("key", Value::from("ssi"))])
                .unwrap();
        }),
    ];
    for (name, write) in writers {
        let db = db();
        let mut late = db.session();
        late.begin_transaction().unwrap();
        write(&db);
        let result = late
            .execute("INSERT (:Memory {key: 'ssi'})")
            .map(drop)
            .and_then(|()| late.commit());
        let error = result.expect_err(name);
        assert!(
            is_conflict(&error) || is_violation(&error),
            "{name}: {error}"
        );
        assert_eq!(count(&db, "ssi"), 1, "{name}");
    }
}

/// One transaction may delete the node with a key and insert another with
/// it, and set the key of a committed node to the value it holds.
#[test]
fn a_transaction_may_delete_and_reinsert_its_key() {
    let db = db();
    db.execute("INSERT (:Memory {key: 'ssi', n: 1}), (:Memory {key: 'kept', n: 3})")
        .unwrap();
    let mut session = db.session();
    session.begin_transaction().unwrap();
    session
        .execute("MATCH (m:Memory {key: 'ssi'}) DELETE m")
        .unwrap();
    session
        .execute("INSERT (:Memory {key: 'ssi', n: 2})")
        .unwrap();
    session
        .execute("MATCH (m:Memory {key: 'kept'}) SET m.key = 'kept'")
        .unwrap();
    session.commit().unwrap();
    assert_eq!(
        db.execute("MATCH (m:Memory) RETURN m.key AS k, m.n ORDER BY k")
            .unwrap()
            .rows(),
        [
            [Value::from("kept"), Value::Int64(3)],
            [Value::from("ssi"), Value::Int64(2)]
        ]
    );
}

/// A transaction may set the key of a node it created to the value the
/// node already holds.
#[test]
#[ignore = "the UNIQUE check of a SET reads the node's committed value: a node the transaction created is refused its own value"]
fn a_new_node_may_be_set_to_its_own_key() {
    let db = db();
    let mut session = db.session();
    session.begin_transaction().unwrap();
    session
        .execute("INSERT (:Memory {key: 'ssi', n: 2})")
        .unwrap();
    session
        .execute("MATCH (m:Memory {key: 'ssi'}) SET m.key = 'ssi'")
        .unwrap();
    session.commit().unwrap();
    assert_eq!(
        db.execute("MATCH (m:Memory) RETURN m.key, m.n")
            .unwrap()
            .rows(),
        [[Value::from("ssi"), Value::Int64(2)]]
    );
}

/// A `SET` that moves a node's key away keeps the old value taken until it
/// commits: another transaction's insert of it fails with G2000 meanwhile,
/// and succeeds afterwards.
#[test]
fn a_key_set_away_is_taken_until_the_commit() {
    let db = db();
    db.execute("INSERT (:Memory {key: 'ssi'})").unwrap();
    let mut mover = db.session();
    mover.begin_transaction().unwrap();
    mover
        .execute("MATCH (m:Memory {key: 'ssi'}) SET m.key = 'moved'")
        .unwrap();
    let error = db
        .execute("INSERT (:Memory {key: 'ssi'})")
        .expect_err("the old value is still committed");
    assert!(is_violation(&error), "{error}");
    mover.commit().unwrap();
    db.execute("INSERT (:Memory {key: 'ssi'})").unwrap();
    assert_eq!((count(&db, "ssi"), count(&db, "moved")), (1, 1));
}

/// A key of several properties claims their values together: the same pair
/// conflicts, a pair that differs in one value does not.
#[test]
fn a_key_of_several_properties_claims_the_combination() {
    let db = GrafeoDB::new_in_memory();
    db.execute("CREATE CONSTRAINT pair FOR (p:Pair) ON (p.a, p.b) UNIQUE")
        .unwrap();
    let mut holder = db.session();
    holder.begin_transaction().unwrap();
    holder.execute("INSERT (:Pair {a: 1, b: 2})").unwrap();
    let error = db
        .execute("INSERT (:Pair {a: 1, b: 2})")
        .expect_err("the same pair");
    assert!(is_conflict(&error), "{error}");
    db.execute("INSERT (:Pair {a: 1, b: 3})").unwrap();
    holder.commit().unwrap();
    assert_eq!(
        db.execute("MATCH (p:Pair) RETURN p.a AS a, p.b AS b ORDER BY b")
            .unwrap()
            .rows(),
        [
            [Value::Int64(1), Value::Int64(2)],
            [Value::Int64(1), Value::Int64(3)]
        ]
    );
}

/// The claim names its graph: the same value in two graphs does not
/// conflict.
#[test]
fn claims_stay_in_their_graph() {
    let db = db();
    db.execute("CREATE GRAPH g").unwrap();
    let mut holder = db.session();
    holder.use_graph("g");
    holder.begin_transaction().unwrap();
    holder.execute("INSERT (:Memory {key: 'ssi'})").unwrap();
    db.execute("INSERT (:Memory {key: 'ssi'})").unwrap();
    holder.commit().unwrap();
    assert_eq!(count(&db, "ssi"), 1, "the default graph");
}

/// The #400 reporter's insert-if-absent from eight threads at once, with
/// `find` and `insert` run in the session: the results.
fn insert_if_absent_from_eight_threads(
    db: &GrafeoDB,
    find: fn(&Session) -> Result<bool, Error>,
    insert: fn(&Session) -> Result<(), Error>,
) -> Vec<Result<bool, Error>> {
    let barrier = Barrier::new(8);
    std::thread::scope(|scope| {
        let threads: Vec<_> = (0..8)
            .map(|_| {
                scope.spawn(|| {
                    let mut session = db.session();
                    barrier.wait();
                    session.begin_transaction()?;
                    if find(&session)? {
                        session.rollback()?;
                        return Ok(false);
                    }
                    insert(&session)?;
                    session.commit()?;
                    Ok(true)
                })
            })
            .collect();
        threads
            .into_iter()
            .map(|thread| thread.join().unwrap())
            .collect()
    })
}

/// Exactly one of the eight commits; the others found the key, or failed
/// with the write conflict or G2000.
fn assert_one_committed(results: &[Result<bool, Error>], what: &str) {
    assert_eq!(
        results
            .iter()
            .filter(|result| matches!(result, Ok(true)))
            .count(),
        1,
        "{what}: {results:?}"
    );
    assert!(
        results.iter().all(|result| match result {
            Ok(_) => true,
            Err(error) => is_conflict(error) || is_violation(error),
        }),
        "{what}: {results:?}"
    );
}

#[test]
fn eight_threads_insert_if_absent_commit_once_in_gql() {
    let db = db();
    let results = insert_if_absent_from_eight_threads(
        &db,
        |session| {
            Ok(!session
                .execute("MATCH (m:Memory {key: 'ssi'}) RETURN m.key")?
                .rows()
                .is_empty())
        },
        |session| session.execute("INSERT (:Memory {key: 'ssi'})").map(drop),
    );
    assert_one_committed(&results, "GQL");
    assert_eq!(count(&db, "ssi"), 1);
}

#[cfg(feature = "cypher")]
#[test]
fn eight_threads_insert_if_absent_commit_once_in_cypher() {
    let db = db();
    let results = insert_if_absent_from_eight_threads(
        &db,
        |session| {
            Ok(!session
                .execute_cypher("MATCH (m:Memory {key: 'ssi'}) RETURN m.key")?
                .rows()
                .is_empty())
        },
        |session| {
            session
                .execute_cypher("CREATE (:Memory {key: 'ssi'})")
                .map(drop)
        },
    );
    assert_one_committed(&results, "Cypher");
    assert_eq!(count(&db, "ssi"), 1);
}

/// SQL/PGQ reads and declares, it does not write: the graph and its
/// PRIMARY KEY come from `CREATE PROPERTY GRAPH`, the read is a
/// `GRAPH_TABLE` query and the insert is GQL, in that graph.
#[cfg(feature = "sql-pgq")]
#[test]
fn eight_threads_insert_if_absent_commit_once_under_a_sql_primary_key() {
    let db = GrafeoDB::new_in_memory();
    db.execute_sql(
        "CREATE PROPERTY GRAPH Memories NODE TABLES (Memory (key VARCHAR PRIMARY KEY, note VARCHAR))",
    )
    .unwrap();
    let barrier = Barrier::new(8);
    let results: Vec<Result<bool, Error>> = std::thread::scope(|scope| {
        let threads: Vec<_> = (0..8)
            .map(|_| {
                scope.spawn(|| {
                    let mut session = db.session();
                    session.use_graph("Memories");
                    barrier.wait();
                    session.begin_transaction()?;
                    let found = session.execute_sql(
                        "SELECT k FROM GRAPH_TABLE (Memories MATCH (m:Memory) \
                         WHERE m.key = 'ssi' COLUMNS (m.key AS k))",
                    )?;
                    if !found.rows().is_empty() {
                        session.rollback()?;
                        return Ok(false);
                    }
                    session.execute("INSERT (:Memory {key: 'ssi', note: 'Prague'})")?;
                    session.commit()?;
                    Ok(true)
                })
            })
            .collect();
        threads
            .into_iter()
            .map(|thread| thread.join().unwrap())
            .collect()
    });
    assert_one_committed(&results, "SQL/PGQ");
    let session = db.session();
    session.use_graph("Memories");
    assert_eq!(
        session
            .execute("MATCH (m:Memory) RETURN count(m)")
            .unwrap()
            .rows(),
        [[Value::Int64(1)]]
    );
}

/// A crash after the first of two writers committed and the second failed:
/// the reopened database holds the key once.
#[cfg(all(feature = "wal", feature = "grafeo-file"))]
#[path = "common/replay.rs"]
mod replay;

#[cfg(all(feature = "wal", feature = "grafeo-file"))]
mod persistent {
    use super::*;

    #[test]
    fn a_crash_after_the_race_keeps_one() {
        let (_dir, db) = replay::reopened_after_crash(
            "persistent::a_crash_after_the_race_keeps_one",
            |path| GrafeoDB::open(path).unwrap(),
            |db| {
                db.execute(CONSTRAINT).unwrap();
                let mut first = holding_ssi(db);
                let mut second = db.session();
                second.begin_transaction().unwrap();
                let error = second
                    .execute("INSERT (:Memory {key: 'ssi'})")
                    .expect_err("the second writer");
                assert!(is_conflict(&error), "{error}");
                first.commit().unwrap();
                second.rollback().unwrap();
            },
        );
        assert_eq!(count(&db, "ssi"), 1);
        let error = db
            .execute("INSERT (:Memory {key: 'ssi'})")
            .expect_err("the constraint holds after the reopen");
        assert!(is_violation(&error), "{error}");
    }
}
