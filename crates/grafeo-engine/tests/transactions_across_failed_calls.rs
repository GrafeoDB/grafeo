//! A transaction goes on across a statement that fails, whatever fails it (0.6.2, a sweep of
//! the new procedure errors and deadlines): a procedure argument out of its domain (22023), a
//! procedure past its deadline, a wrong `YIELD` column, a read or a write past the query timeout.
//! The failed statement is undone, earlier statements stay, the transaction stays active and takes
//! more statements, commit keeps what it wrote and rollback drops it, a savepoint before the failure
//! still rolls back to its point, and another transaction can write afterwards. Outside a
//! transaction, a statement that fails after a write leaves nothing (the 0.6.0 rule, CHANGELOG:
//! "a failed statement inside a transaction ... is now undone, and the transaction goes on").
//!
//! A procedure's deadline comes from the probe (`testing-deadline-probe`), so it passes at the first
//! poll on any machine; reads and writes pass a real 300 ms timeout, on statements that run far
//! longer than that in any build.
//!
//! ```bash
//! cargo test -p grafeo-engine --all-features --test main transactions_across_failed_calls::
//! ```

#![cfg(all(
    feature = "testing-deadline-probe",
    feature = "algos",
    feature = "cypher"
))]

use std::time::Duration;

use grafeo_common::types::Value;
use grafeo_common::utils::error::{Error, QueryErrorKind};
use grafeo_common::utils::gqlstatus::GqlStatus;
use grafeo_core::execution::deadline;
use grafeo_engine::session::Session;
use grafeo_engine::{Config, GrafeoDB};

/// What makes the statement fail.
#[derive(Clone, Copy, Debug)]
enum Failure {
    /// `pagerank` with a damping of 2.0: 22023.
    OutOfDomain,
    /// `betweenness_centrality` with its deadline passed at the first poll.
    CallPastItsDeadline,
    /// `YIELD` of a column the procedure has not.
    WrongYieldColumn,
    /// A read that runs past the query timeout.
    ReadPastTheTimeout,
    /// A write of many nodes that runs past the query timeout.
    WritePastTheTimeout,
}

const FAILURES: [Failure; 5] = [
    Failure::OutOfDomain,
    Failure::CallPastItsDeadline,
    Failure::WrongYieldColumn,
    Failure::ReadPastTheTimeout,
    Failure::WritePastTheTimeout,
];

impl Failure {
    fn statement(self) -> &'static str {
        match self {
            Self::OutOfDomain => "CALL grafeo.pagerank(2.0) YIELD score RETURN count(*) AS c",
            Self::CallPastItsDeadline => {
                "CALL grafeo.betweenness_centrality() YIELD centrality RETURN count(*) AS c"
            }
            Self::WrongYieldColumn => {
                "CALL grafeo.betweenness_centrality() YIELD nope RETURN count(*) AS c"
            }
            Self::ReadPastTheTimeout => {
                "UNWIND range(1, 200000000) AS i WITH i WHERE i % 7 = 3 RETURN count(*) AS c"
            }
            Self::WritePastTheTimeout => "UNWIND range(1, 200000000) AS i INSERT (:Junk {i: i})",
        }
    }

    /// The statement fails as this failure does: its status, or the timeout's kind.
    fn is_its_error(self, error: &Error) -> bool {
        let timeout = matches!(error, Error::Query(query) if query.kind == QueryErrorKind::Timeout);
        let status = GqlStatus::from(error).as_str().to_string();
        match self {
            Self::OutOfDomain => status == "22023",
            Self::WrongYieldColumn => status.starts_with("42"),
            Self::CallPastItsDeadline | Self::ReadPastTheTimeout | Self::WritePastTheTimeout => {
                timeout
            }
        }
    }
}

/// 60 nodes on a ring with chords, every statement past 300 ms timed out.
fn database() -> GrafeoDB {
    let db =
        GrafeoDB::with_config(Config::in_memory().with_query_timeout(Duration::from_millis(300)))
            .unwrap();
    let session = db.session();
    session
        .execute("UNWIND range(0, 59) AS i INSERT (:V {i: i})")
        .unwrap();
    session
        .execute(
            "MATCH (a:V), (b:V) WHERE b.i = (a.i + 1) % 60 OR b.i = (a.i * 7) % 60 \
             INSERT (a)-[:E]->(b)",
        )
        .unwrap();
    db
}

/// Runs `failure`'s statement: under the probe for a procedure past its deadline.
fn fail(session: &Session, failure: Failure) -> Result<(), Error> {
    let run = || session.execute(failure.statement()).map(|_| ());
    if matches!(failure, Failure::CallPastItsDeadline) {
        let _probe = deadline::expire_after_polls(0);
        run()
    } else {
        run()
    }
}

fn count(session: &Session, label: &str) -> i64 {
    let result = session
        .execute(&format!("MATCH (n:{label}) RETURN count(n) AS c"))
        .unwrap();
    match result.rows()[0][0] {
        Value::Int64(count) => count,
        ref other => panic!("{other:?}"),
    }
}

#[derive(Clone, Copy, Debug)]
enum End {
    Commit,
    Rollback,
}

/// A failed statement inside a transaction: undone, and the transaction goes on to its end.
#[test]
fn a_transaction_goes_on_across_a_failed_statement() {
    let mut wrong = Vec::new();
    for failure in FAILURES {
        for end in [End::Commit, End::Rollback] {
            let db = database();
            let mut session = db.session();
            session.begin_transaction().unwrap();
            session.execute("INSERT (:Mark {n: 1})").unwrap();
            let mut check = |what: &str, ok: bool, detail: String| {
                if !ok {
                    wrong.push(format!("{failure:?}, then {end:?}: {what}: {detail}"));
                }
            };
            match fail(&session, failure) {
                Err(error) => check(
                    "fails with its error",
                    failure.is_its_error(&error),
                    error.to_string(),
                ),
                Ok(()) => check("fails", false, "it succeeded".to_string()),
            }
            check(
                "the transaction is active",
                session.in_transaction(),
                String::new(),
            );
            let (marks, junk) = (count(&session, "Mark"), count(&session, "Junk"));
            check(
                "earlier writes stay, the failed one is undone",
                (marks, junk) == (1, 0),
                format!("{marks} marks, {junk} junk"),
            );
            session.execute("INSERT (:Mark {n: 2})").unwrap();
            match end {
                End::Commit => session.commit().unwrap(),
                End::Rollback => session.rollback().unwrap(),
            }
            let after = db.session();
            let (marks, junk) = (count(&after, "Mark"), count(&after, "Junk"));
            let want = match end {
                End::Commit => (2, 0),
                End::Rollback => (0, 0),
            };
            check(
                "the end keeps or drops the transaction's writes",
                (marks, junk) == want,
                format!("{marks} marks, {junk} junk, want {want:?}"),
            );
            let mut other = db.session();
            other.begin_transaction().unwrap();
            let written = other
                .execute("MATCH (m:Mark) SET m.seen = true")
                .and_then(|_| other.execute("INSERT (:Mark {n: 3})"))
                .and_then(|_| other.commit());
            check(
                "another transaction writes after",
                written.is_ok(),
                format!("{written:?}"),
            );
        }
    }
    assert!(wrong.is_empty(), "{}", wrong.join("\n"));
}

/// A savepoint taken before the failed statement still rolls back to its point.
#[test]
fn a_savepoint_before_a_failed_statement_still_holds() {
    for failure in FAILURES {
        let db = database();
        let mut session = db.session();
        session.begin_transaction().unwrap();
        session.execute("INSERT (:Mark {n: 1})").unwrap();
        session.savepoint("s").unwrap();
        session.execute("INSERT (:Mark {n: 2})").unwrap();
        assert!(fail(&session, failure).is_err(), "{failure:?}");
        session.rollback_to_savepoint("s").unwrap();
        assert_eq!(
            (count(&session, "Mark"), count(&session, "Junk")),
            (1, 0),
            "{failure:?}: back to the savepoint"
        );
        session.commit().unwrap();
        let after = db.session();
        assert_eq!(
            (count(&after, "Mark"), count(&after, "Junk")),
            (1, 0),
            "{failure:?}: after commit"
        );
    }
}

/// Outside a transaction, a statement that writes and then fails leaves nothing of its write.
#[test]
fn a_statement_that_writes_then_fails_leaves_nothing() {
    for (failure, cypher) in [
        (
            Failure::OutOfDomain,
            "CREATE (:Mark {n: 9}) WITH 1 AS x CALL grafeo.pagerank(2.0) YIELD score RETURN count(*) AS c",
        ),
        (
            Failure::CallPastItsDeadline,
            "CREATE (:Mark {n: 9}) WITH 1 AS x CALL grafeo.betweenness_centrality() YIELD centrality RETURN count(*) AS c",
        ),
        (
            Failure::ReadPastTheTimeout,
            "CREATE (:Mark {n: 9}) WITH 1 AS x UNWIND range(1, 200000000) AS i WITH i WHERE i % 7 = 3 RETURN count(*) AS c",
        ),
    ] {
        let db = database();
        let session = db.session();
        let outcome = if matches!(failure, Failure::CallPastItsDeadline) {
            let _probe = deadline::expire_after_polls(0);
            session.execute_cypher(cypher)
        } else {
            session.execute_cypher(cypher)
        };
        match outcome {
            Err(error) => assert!(failure.is_its_error(&error), "{failure:?}: {error}"),
            Ok(_) => panic!("{failure:?}: it succeeded"),
        }
        assert_eq!(
            count(&session, "Mark"),
            0,
            "{failure:?}: the write is undone"
        );
    }
}

/// A `YIELD` of a column the procedure has not fails when the statement is planned, before the
/// algorithm runs: under a deadline that passes at the first poll, the semantic error comes, not
/// the timeout. 0.6.1 ran the whole algorithm first (4.5 s of betweenness on 4,000 nodes) and
/// only then reported the column.
#[test]
fn a_wrong_yield_column_fails_before_the_algorithm_runs() {
    let db = database();
    let session = db.session();
    let _probe = deadline::expire_after_polls(0);
    for call in [
        "CALL grafeo.betweenness_centrality() YIELD nope RETURN count(*) AS c",
        "CALL grafeo.pagerank() YIELD nope RETURN count(*) AS c",
    ] {
        let error = session.execute(call).expect_err(call);
        assert!(
            Failure::WrongYieldColumn.is_its_error(&error),
            "{call}: {error}, want the semantic error of the column"
        );
    }
}
