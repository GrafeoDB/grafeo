//! A commit that fails with a write-write conflict aborts the transaction
//! completely (#409): its entities are released, its versions discarded, an
//! abort is logged to the WAL and the session is left without a transaction.
//!
//! ```bash
//! cargo test -p grafeo-engine --features full --test failed_commit_aborts
//! ```

#![allow(missing_docs)]

use grafeo_common::types::Value;
use grafeo_engine::GrafeoDB;
use grafeo_engine::session::Session;

fn seed(db: &GrafeoDB) {
    db.session()
        .execute("INSERT (:Person {name: 'Alix', age: 30})")
        .unwrap();
}

fn age(db: &GrafeoDB) -> Value {
    let result = db
        .session()
        .execute("MATCH (n:Person {name: 'Alix'}) RETURN n.age")
        .unwrap();
    assert_eq!(result.row_count(), 1);
    result.rows()[0][0].clone()
}

fn temp_count(db: &GrafeoDB) -> Value {
    let result = db
        .session()
        .execute("MATCH (n:Temp) RETURN count(n)")
        .unwrap();
    result.rows()[0][0].clone()
}

fn set_age(session: &Session, age: i64) {
    session
        .execute(&format!(
            "MATCH (n:Person {{name: 'Alix'}}) SET n.age = {age}"
        ))
        .unwrap();
}

/// Runs the conflict from the issue: `loser` inserts a `:Temp` node and sets
/// Alix's age to 32 after `winner` committed age 31. Returns the loser's
/// commit error.
fn lose_conflict(db: &GrafeoDB) -> (Session, grafeo_common::utils::error::Error) {
    let mut loser = db.session();
    let mut winner = db.session();
    loser.begin_transaction().unwrap();
    winner.begin_transaction().unwrap();

    set_age(&winner, 31);
    winner.commit().unwrap();

    loser.execute("INSERT (:Temp {name: 'Vincent'})").unwrap();
    set_age(&loser, 32);
    let err = loser
        .commit()
        .expect_err("second writer of Alix must fail with a write-write conflict");
    (loser, err)
}

#[test]
fn failed_commit_releases_its_entities() {
    let db = GrafeoDB::new_in_memory();
    seed(&db);
    let (_loser, err) = lose_conflict(&db);
    assert!(
        err.to_string().to_lowercase().contains("conflict"),
        "unexpected error: {err}"
    );

    // A later writer of the same node must not conflict with the failed
    // transaction, which would happen if it were still active.
    let mut next = db.session();
    next.begin_transaction().unwrap();
    set_age(&next, 33);
    next.commit().unwrap();

    assert_eq!(age(&db), Value::Int64(33));
}

#[test]
fn failed_commit_leaves_session_without_transaction() {
    let db = GrafeoDB::new_in_memory();
    seed(&db);
    let (mut loser, _) = lose_conflict(&db);

    assert!(!loser.in_transaction());
    assert!(
        loser.rollback().is_err(),
        "there is no transaction left to roll back"
    );

    // The same session can start over and write the node it lost on.
    loser.begin_transaction().unwrap();
    set_age(&loser, 34);
    loser.commit().unwrap();
    assert_eq!(age(&db), Value::Int64(34));
}

#[test]
fn failed_commit_discards_its_writes() {
    let db = GrafeoDB::new_in_memory();
    seed(&db);
    let _ = lose_conflict(&db);

    assert_eq!(age(&db), Value::Int64(31), "the winner's value stays");
    assert_eq!(
        temp_count(&db),
        Value::Int64(0),
        "the loser's insert is gone"
    );

    // Later commits must not pick up the failed transaction's versions.
    let mut next = db.session();
    next.begin_transaction().unwrap();
    set_age(&next, 33);
    next.commit().unwrap();
    assert_eq!(temp_count(&db), Value::Int64(0));
}

#[cfg(feature = "wal")]
mod persistent {
    use super::*;
    use grafeo_engine::Config;
    use grafeo_engine::config::StorageFormat;
    use std::path::Path;

    fn configs(dir: &Path) -> Vec<(&'static str, Config)> {
        let mut configs = vec![(
            "wal directory",
            Config::persistent(dir.join("dir-db")).with_storage_format(StorageFormat::WalDirectory),
        )];
        #[cfg(feature = "grafeo-file")]
        configs.push((
            "single file",
            Config::persistent(dir.join("single.grafeo"))
                .with_storage_format(StorageFormat::SingleFile),
        ));
        configs
    }

    /// Without an abort marker, WAL recovery carries the failed transaction's
    /// records into the next commit and resurrects them on reopen.
    #[test]
    fn failed_commit_is_not_recovered_from_wal() {
        let dir = tempfile::tempdir().unwrap();
        for (name, config) in configs(dir.path()) {
            {
                let db = GrafeoDB::with_config(config.clone()).unwrap();
                seed(&db);
                let _ = lose_conflict(&db);

                let mut next = db.session();
                next.begin_transaction().unwrap();
                set_age(&next, 33);
                next.commit().unwrap();
                db.close().unwrap();
            }

            let db = GrafeoDB::with_config(config).unwrap();
            assert_eq!(age(&db), Value::Int64(33), "{name}: age after reopen");
            assert_eq!(
                temp_count(&db),
                Value::Int64(0),
                "{name}: failed insert resurrected on reopen"
            );
            db.close().unwrap();
        }
    }
}
