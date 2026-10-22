//! One clock per statement (#179): the clock functions without arguments
//! read one instant per statement, never one per transaction, session or
//! plan. A statement's rows all see it; the next statement, also in the same
//! transaction and also when its text is the same (a cached plan), reads the
//! clock again. The per-row cases run on every runner in
//! `tests/spec/rosetta/statement_clock.gtest`.
//!
//! ```bash
//! cargo test -p grafeo-engine --all-features --test statement_clock
//! ```

#![cfg(feature = "gql")]

use std::time::Duration;

use grafeo_common::types::Value;
use grafeo_engine::GrafeoDB;
use grafeo_engine::session::Session;

/// Longer than one tick of `timestamp()`'s milliseconds, and than a sleep's
/// granularity on Windows.
const TICK: Duration = Duration::from_millis(20);

/// The milliseconds `timestamp()` gives in its own statement.
fn millis(session: &Session) -> i64 {
    match session.execute("RETURN timestamp() AS ms").unwrap().rows() {
        [row] => match row[0] {
            Value::Int64(ms) => ms,
            ref other => panic!("timestamp() gave {other:?}"),
        },
        rows => panic!("one row, got {rows:?}"),
    }
}

/// The same query text twice: the second run reads the clock again, so a
/// cached plan never carries the first one's instant.
#[test]
fn consecutive_statements_read_the_clock_again() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    let first = millis(&session);
    std::thread::sleep(TICK);
    let second = millis(&session);
    assert!(second > first, "{first} then {second}");
}

/// Inside one transaction each statement reads its own instant (a
/// transaction is not one clock, unlike DuckDB's `now()`).
#[test]
fn each_statement_of_a_transaction_reads_its_own_instant() {
    let db = GrafeoDB::new_in_memory();
    let mut session = db.session();
    session.begin_transaction().unwrap();
    let first = millis(&session);
    std::thread::sleep(TICK);
    let second = millis(&session);
    session.commit().unwrap();
    assert!(second > first, "{first} then {second}");
}

/// A write over many rows, then a read in a later statement: the rows hold
/// one instant, and the later read is past it.
#[test]
fn a_write_over_many_rows_stores_one_instant() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    session
        .execute("FOR i IN range(1, 100000) INSERT (:Event {i: i, at: timestamp()})")
        .unwrap();
    let stamps = session
        .execute("MATCH (e:Event) RETURN count(DISTINCT e.at) AS stamps, max(e.at) AS at")
        .unwrap();
    let [row] = stamps.rows() else {
        panic!("one row, got {:?}", stamps.rows());
    };
    assert_eq!(row[0], Value::Int64(1), "one instant for every row");
    let Value::Int64(at) = row[1] else {
        panic!("max(e.at) gave {:?}", row[1]);
    };
    std::thread::sleep(TICK);
    assert!(
        millis(&session) > at,
        "a later statement reads the clock again"
    );
}
