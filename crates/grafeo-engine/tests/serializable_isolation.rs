//! Serializable isolation prevents write skew; snapshot isolation does not
//! (#400).
//!
//! `IsolationLevel::Serializable` is documented as serializable snapshot
//! isolation: besides write-write conflicts it detects read-write ones, so
//! of two transactions that each read what the other then writes, one
//! fails with a serialization failure. The reporter's case is "insert if
//! absent": each transaction reads that no node has the key and inserts
//! one. Today query reads never enter the transaction's read set
//! (`TransactionManager::record_read` has no caller outside its unit
//! tests), so every such transaction commits; those tests are ignored
//! until #400 is fixed. Snapshot isolation allows write skew, as
//! documented. A UNIQUE constraint keeps a concurrent duplicate out at any
//! level: the second insert or its commit fails with the retryable write
//! conflict on the key (or the constraint violation, once the first one is
//! committed and seen), and one node is left.
//!
//! ```bash
//! cargo test -p grafeo-engine --all-features --test serializable_isolation -- --include-ignored
//! ```

#![cfg(feature = "gql")]

use std::sync::Barrier;

use grafeo_common::types::Value;
use grafeo_common::utils::error::{Error, TransactionError};
use grafeo_engine::GrafeoDB;
use grafeo_engine::session::Session;
use grafeo_engine::transaction::IsolationLevel;

/// The reporter's read: is there a node with the key?
const FIND_KEY: &str =
    "MATCH (m:Memory) WHERE m.idempotency_key = 'ssi' RETURN m.idempotency_key LIMIT 1";

const INSERT_KEY: &str = "INSERT (:Memory {idempotency_key: 'ssi'})";

fn count(db: &GrafeoDB, query: &str) -> Value {
    db.execute(query).unwrap().rows()[0][0].clone()
}

fn keys(db: &GrafeoDB) -> Value {
    count(
        db,
        "MATCH (m:Memory) WHERE m.idempotency_key = 'ssi' RETURN count(m)",
    )
}

/// Whether `error` is the serialization failure of a read-write conflict.
fn is_serialization_failure(error: &Error) -> bool {
    matches!(
        error,
        Error::Transaction(TransactionError::SerializationFailure(_))
    )
}

/// Whether `error` is what a duplicate of a UNIQUE key gets: the retryable
/// write conflict on the key, or the constraint violation.
fn is_duplicate_key(error: &Error) -> bool {
    match error {
        Error::Transaction(TransactionError::WriteConflict(message)) => {
            message.contains("on key :Memory(idempotency_key")
        }
        other => other.to_string().contains("UNIQUE constraint violation"),
    }
}

/// Insert if absent in `session`, at `level`: `Ok(true)` when it inserted
/// and committed, `Ok(false)` when it found the key and rolled back, the
/// error of the insert or commit otherwise (the transaction is then gone).
fn insert_if_absent(session: &mut Session, level: IsolationLevel) -> Result<bool, Error> {
    session.begin_transaction_with_isolation(level)?;
    if !session.execute(FIND_KEY)?.rows().is_empty() {
        session.rollback()?;
        return Ok(false);
    }
    session.execute(INSERT_KEY)?;
    session.commit()?;
    Ok(true)
}

/// Eight threads run insert if absent at `level` at once, released by a
/// barrier: the results, in thread order.
fn insert_if_absent_from_eight_threads(
    db: &GrafeoDB,
    level: IsolationLevel,
) -> Vec<Result<bool, Error>> {
    let barrier = Barrier::new(8);
    std::thread::scope(|scope| {
        let threads: Vec<_> = (0..8)
            .map(|_| {
                scope.spawn(|| {
                    let mut session = db.session();
                    barrier.wait();
                    insert_if_absent(&mut session, level)
                })
            })
            .collect();
        threads
            .into_iter()
            .map(|thread| thread.join().unwrap())
            .collect()
    })
}

/// Two sessions both read that the key is absent before either inserts.
/// Under serializable isolation the second commit is a serialization
/// failure and changes nothing: one node.
#[test]
#[ignore = "#400: query reads are not recorded in the SSI read set yet"]
fn serializable_insert_if_absent_interleaved_commits_once() {
    let db = GrafeoDB::new_in_memory();
    let mut alix = db.session();
    let mut gus = db.session();
    alix.begin_transaction_with_isolation(IsolationLevel::Serializable)
        .unwrap();
    gus.begin_transaction_with_isolation(IsolationLevel::Serializable)
        .unwrap();
    assert_eq!(
        alix.execute(FIND_KEY).unwrap().rows(),
        Vec::<Vec<Value>>::new()
    );
    assert_eq!(
        gus.execute(FIND_KEY).unwrap().rows(),
        Vec::<Vec<Value>>::new()
    );
    alix.execute(INSERT_KEY).unwrap();
    gus.execute(INSERT_KEY).unwrap();
    alix.commit().unwrap();
    let error = gus.commit().expect_err("the second commit must fail");
    assert!(is_serialization_failure(&error), "{error}");
    assert_eq!(keys(&db), Value::Int64(1));
}

/// The reporter's test: eight concurrent insert-if-absent transactions at
/// serializable isolation. Each one either inserts or fails with a
/// serialization failure, never sees the key (none had committed when it
/// read), and exactly one node is left.
#[test]
#[ignore = "#400: query reads are not recorded in the SSI read set yet"]
fn serializable_insert_if_absent_commits_once() {
    let db = GrafeoDB::new_in_memory();
    let results = insert_if_absent_from_eight_threads(&db, IsolationLevel::Serializable);
    let committed = results
        .iter()
        .filter(|result| matches!(result, Ok(true)))
        .count();
    let failed: Vec<&Error> = results
        .iter()
        .filter_map(|result| result.as_ref().err())
        .collect();
    assert_eq!(committed, 1, "{results:?}");
    assert_eq!(
        committed + failed.len() + results.iter().filter(|r| matches!(r, Ok(false))).count(),
        8
    );
    assert!(
        failed.iter().all(|error| is_serialization_failure(error)),
        "{failed:?}"
    );
    assert_eq!(keys(&db), Value::Int64(1));
}

/// Two doctors on call, and the rule that one must stay: each transaction
/// reads that both are on call and takes one off. Run interleaved at
/// `level`: the result of the second commit, and how many stay on call.
fn two_doctors_go_off_call(level: IsolationLevel) -> (Result<(), Error>, Value) {
    let db = GrafeoDB::new_in_memory();
    db.execute(
        "INSERT (:Doctor {name: 'Alix', on_call: true}), (:Doctor {name: 'Gus', on_call: true})",
    )
    .unwrap();
    let on_call = "MATCH (d:Doctor) WHERE d.on_call RETURN count(d)";
    let mut alix = db.session();
    let mut gus = db.session();
    alix.begin_transaction_with_isolation(level).unwrap();
    gus.begin_transaction_with_isolation(level).unwrap();
    for session in [&alix, &gus] {
        assert_eq!(
            session.execute(on_call).unwrap().rows()[0][0],
            Value::Int64(2),
            "both read two doctors on call"
        );
    }
    alix.execute("MATCH (d:Doctor {name: 'Alix'}) SET d.on_call = false")
        .unwrap();
    gus.execute("MATCH (d:Doctor {name: 'Gus'}) SET d.on_call = false")
        .unwrap();
    alix.commit().unwrap();
    let second = gus.commit();
    let left = count(&db, on_call);
    (second, left)
}

/// Snapshot isolation allows write skew, as documented: both commit, and
/// no doctor is left on call.
#[test]
fn snapshot_isolation_allows_write_skew() {
    let (second, left) = two_doctors_go_off_call(IsolationLevel::SnapshotIsolation);
    assert!(second.is_ok(), "{second:?}");
    assert_eq!(left, Value::Int64(0));
}

/// Serializable isolation does not: the second commit is a serialization
/// failure, and one doctor stays on call.
#[test]
#[ignore = "#400: query reads are not recorded in the SSI read set yet"]
fn serializable_prevents_two_row_write_skew() {
    let (second, left) = two_doctors_go_off_call(IsolationLevel::Serializable);
    let error = second.expect_err("the second commit must fail");
    assert!(is_serialization_failure(&error), "{error}");
    assert_eq!(left, Value::Int64(1));
}

/// What holds without serializable isolation: a UNIQUE constraint on the
/// key. Two open transactions insert the same key; whichever statement
/// comes second (its insert or its commit) fails, and one node is left.
#[test]
fn unique_constraint_rejects_an_interleaved_duplicate() {
    let db = GrafeoDB::new_in_memory();
    db.execute("CREATE CONSTRAINT memory_key FOR (m:Memory) ON (m.idempotency_key) UNIQUE")
        .unwrap();
    let mut alix = db.session();
    let mut gus = db.session();
    alix.begin_transaction().unwrap();
    gus.begin_transaction().unwrap();
    alix.execute(INSERT_KEY).unwrap();
    match gus.execute(INSERT_KEY) {
        // Refused at the insert: the first transaction still commits.
        Err(error) => {
            assert!(is_duplicate_key(&error), "{error}");
            alix.commit().unwrap();
        }
        // Refused at the commit, after the first one committed.
        Ok(_) => {
            alix.commit().unwrap();
            let error = gus.commit().expect_err("the duplicate's commit must fail");
            assert!(is_duplicate_key(&error), "{error}");
        }
    }
    assert_eq!(keys(&db), Value::Int64(1));
}

/// The reporter's eight threads, with a UNIQUE constraint on the key and
/// the default isolation: exactly one inserts, every other one either finds
/// the key or fails, and one node is left.
#[test]
fn unique_constraint_rejects_a_concurrent_duplicate() {
    let db = GrafeoDB::new_in_memory();
    db.execute("CREATE CONSTRAINT memory_key FOR (m:Memory) ON (m.idempotency_key) UNIQUE")
        .unwrap();
    let results = insert_if_absent_from_eight_threads(&db, IsolationLevel::SnapshotIsolation);
    let committed = results
        .iter()
        .filter(|result| matches!(result, Ok(true)))
        .count();
    assert_eq!(committed, 1, "{results:?}");
    // Every other thread found the key or lost on it.
    for result in &results {
        if let Err(error) = result {
            assert!(is_duplicate_key(error), "{results:?}");
        }
    }
    assert_eq!(keys(&db), Value::Int64(1));
}

/// Until queries record their reads (#400), asking for Serializable fails
/// with `GRAFEO-Q004` and names snapshot isolation, from the Rust API and
/// from GQL, and begins nothing: snapshot isolation still begins (#671).
#[test]
fn serializable_is_refused_until_reads_are_recorded() {
    use grafeo_common::utils::error::ErrorCode;

    let db = GrafeoDB::new_in_memory();
    let mut session = db.session();
    let refused = |error: Error| {
        assert_eq!(error.error_code(), ErrorCode::QueryUnsupported, "{error}");
        assert!(error.to_string().contains("snapshot isolation"), "{error}");
    };
    refused(
        session
            .begin_transaction_with_isolation(IsolationLevel::Serializable)
            .expect_err("Serializable is refused"),
    );
    assert!(!session.in_transaction());
    refused(
        session
            .execute("START TRANSACTION ISOLATION LEVEL SERIALIZABLE")
            .expect_err("Serializable is refused in GQL"),
    );
    assert!(!session.in_transaction());

    session
        .begin_transaction_with_isolation(IsolationLevel::SnapshotIsolation)
        .unwrap();
    assert!(session.in_transaction());
    session.rollback().unwrap();
}
