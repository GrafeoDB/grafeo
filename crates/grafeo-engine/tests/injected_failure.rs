//! Rollback atomicity tests using the `testing-statement-injection` hook.
//!
//! These tests exercise the rollback contract independently of any specific
//! engine failure path (constraint violations, parse errors, etc.). The
//! injection hook forces `Session::execute` or `Session::commit` to return
//! `Err` at a chosen call count, letting us assert:
//!
//!   * a mid-tx statement failure leaves prior writes undone after ROLLBACK,
//!   * a commit-time failure leaves all writes undone,
//!   * both properties hold across schema boundaries.
//!
//! ```bash
//! cargo test -p grafeo-engine --all-features \
//!     --test main injected_failure::
//! ```

#![cfg(feature = "testing-statement-injection")]

use grafeo_common::testing::statement_failure::{
    with_commit_failure, with_statement_failure_after,
};
use grafeo_engine::GrafeoDB;

fn db() -> GrafeoDB {
    GrafeoDB::new_in_memory()
}

#[test]
fn injected_statement_failure_rolls_back_prior_writes_single_schema() {
    let db = db();
    let session = db.session();

    // The injection counter ticks for every `Session::execute` call, including
    // DDL and session commands, so arm it to fail *after* the three writes.
    // Counter layout: START TRANSACTION (1), INSERT (2), INSERT (3),
    // INSERT (4) <- should fail.
    with_statement_failure_after(4, || {
        session.execute("START TRANSACTION").unwrap();
        session.execute("INSERT (:Item {v: 1})").unwrap();
        session.execute("INSERT (:Item {v: 2})").unwrap();
        let failing = session.execute("INSERT (:Item {v: 3})");
        assert!(
            failing.is_err(),
            "4th execute should be injected as failure"
        );
    });

    // Transaction is still active after the injected failure; rollback explicitly.
    session.execute("ROLLBACK").unwrap();

    let remaining = session.execute("MATCH (n:Item) RETURN n").unwrap();
    assert_eq!(
        remaining.row_count(),
        0,
        "all in-transaction writes must be undone on ROLLBACK"
    );
}

#[test]
fn injected_statement_failure_rolls_back_cross_schema_writes() {
    let db = db();
    let session = db.session();

    session.execute("CREATE SCHEMA alpha").unwrap();
    session.execute("CREATE SCHEMA beta").unwrap();

    // Counter: START TX (1), SET SCHEMA alpha (2), INSERT alpha (3),
    // SET SCHEMA beta (4), INSERT beta (5), SET SCHEMA (6) -> fail.
    with_statement_failure_after(6, || {
        session.execute("START TRANSACTION").unwrap();
        session.execute("SESSION SET SCHEMA alpha").unwrap();
        session.execute("INSERT (:Row {owner: 'alpha'})").unwrap();
        session.execute("SESSION SET SCHEMA beta").unwrap();
        session.execute("INSERT (:Row {owner: 'beta'})").unwrap();
        let failing = session.execute("SESSION RESET SCHEMA");
        assert!(
            failing.is_err(),
            "6th execute should be injected as failure"
        );
    });

    session.execute("ROLLBACK").unwrap();

    session.execute("SESSION SET SCHEMA alpha").unwrap();
    let alpha = session.execute("MATCH (n:Row) RETURN n").unwrap();
    assert_eq!(
        alpha.row_count(),
        0,
        "alpha writes must be undone by cross-schema rollback after injected failure"
    );

    session.execute("SESSION SET SCHEMA beta").unwrap();
    let beta = session.execute("MATCH (n:Row) RETURN n").unwrap();
    assert_eq!(
        beta.row_count(),
        0,
        "beta writes must be undone by cross-schema rollback after injected failure"
    );
}

#[test]
fn injected_commit_failure_rolls_back_prior_writes() {
    let db = db();
    let session = db.session();

    with_commit_failure(|| {
        session.execute("START TRANSACTION").unwrap();
        session.execute("INSERT (:Item {v: 1})").unwrap();
        session.execute("INSERT (:Item {v: 2})").unwrap();
        let commit_result = session.execute("COMMIT");
        assert!(commit_result.is_err(), "injected commit failure expected");
    });

    // After the injected commit failure, the transaction's writes must not be
    // visible. A follow-up query should see zero `Item` nodes.
    let remaining = session.execute("MATCH (n:Item) RETURN n").unwrap();
    assert_eq!(
        remaining.row_count(),
        0,
        "injected commit failure must leave no writes visible"
    );
}
