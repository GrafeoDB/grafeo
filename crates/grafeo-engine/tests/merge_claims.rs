//! Two MERGEs of one pattern conflict, so concurrent MERGEs create it once.
//!
//! A MERGE that creates claims its pattern in the transaction manager before
//! it writes: the canonical key of the pattern in the graph written to. Of two
//! open transactions that claim one key, the first wins and the second fails
//! at its statement with the retryable write conflict (`GRAFEO-T001`); a
//! transaction that claims a key another one committed after it began fails
//! at its commit. Two MERGEs claim one key exactly when each would match what
//! the other creates, as a second run of the same MERGE matches the first's,
//! so every concurrent outcome here is checked against a serial run of the
//! same statements (the oracle). A claim never conflicts with its own
//! transaction and goes with the transaction at its commit or rollback, with
//! a failed statement and at a rollback to a savepoint.
//!
//! The races use a barrier and retry on `GRAFEO-T001`, as an application
//! does; they are small enough to run in CI.
//!
//! ```bash
//! cargo test -p grafeo-engine --all-features --test main merge_claims::
//! ```

#![cfg(feature = "gql")]

use std::collections::HashMap;
use std::sync::Barrier;

use grafeo_common::types::{PropertyKey, Value, ZonedDatetime};
use grafeo_common::utils::error::{ConstraintKind, Error, ErrorCode};
use grafeo_engine::GrafeoDB;
use grafeo_engine::session::Session;

/// How often a race retries one session's work before the test gives up.
const MAX_RETRIES: usize = 1_000;

/// The retryable write conflict, `GRAFEO-T001`.
fn is_conflict(error: &Error) -> bool {
    error.error_code() == ErrorCode::TransactionConflict && error.error_code().is_retryable()
}

/// G2000: a constraint check found a committed element with the value.
fn is_violation(error: &Error) -> bool {
    matches!(
        error,
        Error::Constraint {
            kind: ConstraintKind::Schema,
            ..
        }
    )
}

fn params(pairs: &[(&str, Value)]) -> HashMap<String, Value> {
    pairs
        .iter()
        .map(|(name, value)| ((*name).to_string(), value.clone()))
        .collect()
}

fn int(value: &Value) -> i64 {
    match value {
        Value::Int64(number) => *number,
        other => panic!("a count, got {other:?}"),
    }
}

/// The one integer `query` returns, in the session's graph.
fn count_in(session: &Session, query: &str) -> i64 {
    int(&session
        .execute(query)
        .unwrap_or_else(|error| panic!("{query}: {error}"))
        .rows()[0][0])
}

/// The one integer `query` returns, in the default graph.
fn count(db: &GrafeoDB, query: &str) -> i64 {
    count_in(&db.session(), query)
}

/// The nodes and edges of the default graph and of the graph `g`, when the
/// database has one.
fn elements(db: &GrafeoDB) -> [i64; 4] {
    let default = db.session();
    let nodes = count_in(&default, "MATCH (n) RETURN count(n)");
    let edges = count_in(&default, "MATCH ()-[e]->() RETURN count(e)");
    let in_g = db.session();
    in_g.use_graph("g");
    match in_g.execute("MATCH (n) RETURN count(n)") {
        Ok(result) => [
            nodes,
            edges,
            int(&result.rows()[0][0]),
            count_in(&in_g, "MATCH ()-[e]->() RETURN count(e)"),
        ],
        Err(_) => [nodes, edges, 0, 0],
    }
}

/// One statement a session runs.
type Run = Box<dyn Fn(&Session) -> Result<(), Error> + Send + Sync>;

fn gql(query: &'static str) -> Run {
    Box::new(move |session: &Session| session.execute(query).map(drop))
}

fn gql_with(query: &'static str, pairs: &[(&str, Value)]) -> Run {
    let bound = params(pairs);
    Box::new(move |session: &Session| session.execute_with_params(query, bound.clone()).map(drop))
}

#[cfg(feature = "cypher")]
fn cypher(query: &'static str) -> Run {
    Box::new(move |session: &Session| session.execute_cypher(query).map(drop))
}

#[cfg(feature = "cypher")]
fn cypher_with(query: &'static str, pairs: &[(&str, Value)]) -> Run {
    let bound = params(pairs);
    Box::new(move |session: &Session| {
        session
            .execute_cypher_with_params(query, bound.clone())
            .map(drop)
    })
}

/// Runs `work` in an explicit transaction of `session` and commits it; on an
/// error the transaction is rolled back, so the work can be retried.
fn in_transaction(
    session: &mut Session,
    work: impl FnOnce(&Session) -> Result<(), Error>,
) -> Result<(), Error> {
    session.begin_transaction()?;
    let result = work(session).and_then(|()| session.commit());
    if result.is_err() && session.in_transaction() {
        session.rollback()?;
    }
    result
}

/// The pause before retry number `retry` (1 for the first): a random time up
/// to a bound that doubles with each retry, from 40 microseconds to about
/// 2.5 milliseconds, as an application's retry loop takes. The races check
/// that each key is created once and that every session finishes; progress
/// itself is the deterministic cycle rows' to prove, since under load a
/// descheduled older holder makes a younger session fail for as long as it
/// stalls, and immediate retries would spend a budget counted in tries on
/// that.
fn backoff(retry: usize) -> std::time::Duration {
    use std::hash::BuildHasher;
    let bound = 20_u64 << retry.min(7);
    let random = std::collections::hash_map::RandomState::new().hash_one(retry);
    std::time::Duration::from_micros(random % bound + 1)
}

/// `sessions` sessions start `work` at once (a barrier) and retry it on the
/// write conflict after a random pause (see [`backoff`]), as an application
/// does. Each result is the number of retries, or the error that was not a
/// write conflict.
fn race(
    db: &GrafeoDB,
    sessions: usize,
    work: &(dyn Fn(&mut Session, usize) -> Result<(), Error> + Sync),
) -> Vec<Result<usize, Error>> {
    let barrier = Barrier::new(sessions);
    std::thread::scope(|scope| {
        let threads: Vec<_> = (0..sessions)
            .map(|index| {
                let barrier = &barrier;
                scope.spawn(move || {
                    let mut session = db.session();
                    barrier.wait();
                    let mut retries = 0;
                    loop {
                        match work(&mut session, index) {
                            Ok(()) => return Ok(retries),
                            Err(error) if is_conflict(&error) && retries < MAX_RETRIES => {
                                retries += 1;
                                std::thread::sleep(backoff(retries));
                            }
                            Err(error) => return Err(error),
                        }
                    }
                })
            })
            .collect();
        threads
            .into_iter()
            .map(|thread| thread.join().unwrap())
            .collect()
    })
}

/// Every session of a race ended Ok after its retries.
fn assert_all_ok(results: &[Result<usize, Error>], what: &str) {
    assert!(
        results.iter().all(Result::is_ok),
        "{what}: a session failed with something other than the write conflict: {results:?}"
    );
}

// Row 1: auto-commit sessions MERGE one node.

/// Ten sessions MERGE one node at once, in auto-commit, retrying on the
/// write conflict: one node, and every session ends Ok (the ones that lost
/// matched it on their retry). Twenty rounds, since one round may not race.
#[test]
fn ten_sessions_merge_one_node_once() {
    for round in 0..20 {
        let db = GrafeoDB::new_in_memory();
        let results = race(&db, 10, &|session, _| {
            session
                .execute("MERGE (:Singleton {key: 'only_one'})")
                .map(drop)
        });
        assert_all_ok(&results, &format!("round {round}"));
        assert_eq!(
            count(&db, "MATCH (n:Singleton) RETURN count(n)"),
            1,
            "round {round}: {results:?}"
        );
    }
}

/// The same in Cypher, and in explicit transactions retried as a whole.
#[test]
fn ten_transactions_merge_one_node_once() {
    for round in 0..20 {
        let db = GrafeoDB::new_in_memory();
        let results = race(&db, 10, &|session, _| {
            in_transaction(session, |session| {
                session
                    .execute("MERGE (:Singleton {key: 'only_one'})")
                    .map(drop)
            })
        });
        assert_all_ok(&results, &format!("round {round}"));
        assert_eq!(
            count(&db, "MATCH (n:Singleton) RETURN count(n)"),
            1,
            "round {round}: {results:?}"
        );
    }
}

#[cfg(feature = "cypher")]
#[test]
fn ten_sessions_merge_one_node_once_in_cypher() {
    for round in 0..20 {
        let db = GrafeoDB::new_in_memory();
        let results = race(&db, 10, &|session, _| {
            session
                .execute_cypher("MERGE (:Singleton {key: 'only_one'})")
                .map(drop)
        });
        assert_all_ok(&results, &format!("round {round}"));
        assert_eq!(
            count(&db, "MATCH (n:Singleton) RETURN count(n)"),
            1,
            "round {round}: {results:?}"
        );
    }
}

// Rows 2 to 6: explicit transactions and the claim's lifetime.

/// A session with an open transaction that MERGEd `(:Memory {key: 'ssi'})`.
fn merging_ssi(db: &GrafeoDB) -> Session {
    let mut session = db.session();
    session.begin_transaction().unwrap();
    session.execute("MERGE (:Memory {key: 'ssi'})").unwrap();
    session
}

fn ssi_count(db: &GrafeoDB) -> i64 {
    count(db, "MATCH (m:Memory {key: 'ssi'}) RETURN count(m)")
}

/// While one transaction holds the claim, another transaction's MERGE of the
/// pattern fails at its statement with the retryable `GRAFEO-T001`; once the
/// first commits, the second's retried transaction matches.
#[test]
fn a_merge_while_another_transaction_merges_fails_at_its_statement() {
    let db = GrafeoDB::new_in_memory();
    let mut first = merging_ssi(&db);
    let mut second = db.session();
    second.begin_transaction().unwrap();
    let error = second
        .execute("MERGE (:Memory {key: 'ssi'})")
        .expect_err("the second MERGE, while the first transaction is open");
    assert_eq!(
        error.error_code(),
        ErrorCode::TransactionConflict,
        "{error}"
    );
    assert!(error.error_code().is_retryable(), "{error}");
    assert!(error.to_string().contains("GRAFEO-T001"), "{error}");
    first.commit().unwrap();
    second.rollback().unwrap();
    in_transaction(&mut second, |session| {
        session.execute("MERGE (:Memory {key: 'ssi'})").map(drop)
    })
    .unwrap();
    assert_eq!(ssi_count(&db), 1);
}

/// A transaction that began before another one MERGEd and committed the
/// pattern does not see the node, so its MERGE creates; its commit fails with
/// the same `GRAFEO-T001`, and its retried transaction matches.
#[test]
fn a_merge_of_a_pattern_committed_after_the_transaction_began_fails_at_commit() {
    let db = GrafeoDB::new_in_memory();
    let mut late = db.session();
    late.begin_transaction().unwrap();
    db.execute("MERGE (:Memory {key: 'ssi'})").unwrap();
    late.execute("MERGE (:Memory {key: 'ssi'})")
        .expect("the late MERGE sees no node, so it creates");
    let error = late.commit().expect_err("the late commit");
    assert_eq!(
        error.error_code(),
        ErrorCode::TransactionConflict,
        "{error}"
    );
    assert!(error.error_code().is_retryable(), "{error}");
    assert_eq!(ssi_count(&db), 1, "the failed commit wrote nothing");
    in_transaction(&mut late, |session| {
        session.execute("MERGE (:Memory {key: 'ssi'})").map(drop)
    })
    .unwrap();
    assert_eq!(ssi_count(&db), 1);
}

/// A rollback releases the claim: a later MERGE creates, without a conflict.
#[test]
fn a_rollback_releases_the_claim() {
    let db = GrafeoDB::new_in_memory();
    let mut first = merging_ssi(&db);
    first.rollback().unwrap();
    db.execute("MERGE (:Memory {key: 'ssi'})").unwrap();
    assert_eq!(ssi_count(&db), 1);
}

/// A statement that fails after its MERGE created releases the claim, and the
/// transaction goes on: another transaction's MERGE creates without a
/// conflict. The division reads the created node, so it runs after the MERGE.
#[test]
fn a_failed_statement_releases_its_claim() {
    let db = GrafeoDB::new_in_memory();
    let mut first = db.session();
    first.begin_transaction().unwrap();
    let error = first
        .execute("MERGE (n:Memory {key: 'ssi', zero: 0}) RETURN 1 / n.zero AS x")
        .expect_err("a division by zero after the MERGE");
    assert!(!is_conflict(&error), "{error}");
    db.execute("MERGE (:Memory {key: 'ssi', zero: 0})")
        .expect("the failed statement's claim is gone");
    if first.in_transaction() {
        first.commit().unwrap();
    }
    assert_eq!(ssi_count(&db), 1);
}

/// A rollback to a savepoint releases the claims made since it, and keeps
/// the ones made before.
#[test]
fn a_rollback_to_a_savepoint_releases_the_claims_since() {
    let db = GrafeoDB::new_in_memory();
    let mut first = db.session();
    first.begin_transaction().unwrap();
    first.execute("MERGE (:Memory {key: 'kept'})").unwrap();
    first.savepoint("before_ssi").unwrap();
    first.execute("MERGE (:Memory {key: 'ssi'})").unwrap();
    first.rollback_to_savepoint("before_ssi").unwrap();
    db.execute("MERGE (:Memory {key: 'ssi'})")
        .expect("the claim since the savepoint is gone");
    let error = db
        .execute("MERGE (:Memory {key: 'kept'})")
        .expect_err("the claim before the savepoint is kept");
    assert!(is_conflict(&error), "{error}");
    first.commit().unwrap();
    assert_eq!(ssi_count(&db), 1);
    assert_eq!(
        count(&db, "MATCH (m:Memory {key: 'kept'}) RETURN count(m)"),
        1
    );
}

/// A transaction whose commit fails on another conflict (here a UNIQUE key
/// committed after it began) loses its claims with it: a later MERGE of its
/// pattern creates without a conflict.
#[test]
fn a_failed_commit_releases_the_claims() {
    let db = GrafeoDB::new_in_memory();
    db.execute("CREATE CONSTRAINT city_name FOR (c:City) ON (c.name) UNIQUE")
        .unwrap();
    let mut failing = db.session();
    failing.begin_transaction().unwrap();
    db.execute("INSERT (:City {name: 'Amsterdam'})").unwrap();
    failing.execute("MERGE (:Memory {key: 'ssi'})").unwrap();
    failing
        .execute("INSERT (:City {name: 'Amsterdam'})")
        .expect("the check does not see the later commit");
    let error = failing.commit().expect_err("the UNIQUE key at commit");
    assert!(is_conflict(&error) || is_violation(&error), "{error}");
    db.execute("MERGE (:Memory {key: 'ssi'})")
        .expect("the failed commit's claim is gone");
    assert_eq!(ssi_count(&db), 1);
    assert_eq!(
        count(&db, "MATCH (c:City {name: 'Amsterdam'}) RETURN count(c)"),
        1
    );
}

/// One transaction may claim one pattern any number of times: its second
/// MERGE matches its own node.
#[test]
fn a_transaction_never_conflicts_with_its_own_claim() {
    let db = GrafeoDB::new_in_memory();
    let mut session = merging_ssi(&db);
    session.execute("MERGE (:Memory {key: 'ssi'})").unwrap();
    session
        .execute("MATCH (m:Memory {key: 'ssi'}) DELETE m")
        .unwrap();
    session
        .execute("MERGE (:Memory {key: 'ssi'})")
        .expect("a MERGE after its own delete creates again");
    session.commit().unwrap();
    assert_eq!(ssi_count(&db), 1);
}

// Rows 7 and 8: which pairs of patterns conflict, against the serial run.

/// Two MERGEs: `first` runs in an open transaction (in the graph `g` when
/// `first_in_g`), `second` in auto-commit in the default graph while it is
/// open.
struct Pair {
    name: &'static str,
    first: Run,
    second: Run,
    first_in_g: bool,
}

fn pair(name: &'static str, first: Run, second: Run) -> Pair {
    Pair {
        name,
        first,
        second,
        first_in_g: false,
    }
}

fn database() -> GrafeoDB {
    let db = GrafeoDB::new_in_memory();
    db.execute("CREATE GRAPH g").unwrap();
    db
}

fn first_session(db: &GrafeoDB, pair: &Pair) -> Session {
    let session = db.session();
    if pair.first_in_g {
        session.use_graph("g");
    }
    session
}

/// The serial run: the elements after `first`, then after `second`.
fn serial(pair: &Pair) -> ([i64; 4], [i64; 4]) {
    let db = database();
    (pair.first)(&first_session(&db, pair))
        .unwrap_or_else(|error| panic!("{}: the first MERGE alone: {error}", pair.name));
    let after_first = elements(&db);
    (pair.second)(&db.session())
        .unwrap_or_else(|error| panic!("{}: the second MERGE alone: {error}", pair.name));
    (after_first, elements(&db))
}

/// What the second MERGE gave while the first transaction was open, and the
/// elements once the first committed and the second was retried after a
/// conflict.
fn concurrent(pair: &Pair) -> (Result<(), Error>, [i64; 4]) {
    let db = database();
    let mut first = first_session(&db, pair);
    first.begin_transaction().unwrap();
    (pair.first)(&first).unwrap_or_else(|error| panic!("{}: the first MERGE: {error}", pair.name));
    let second = db.session();
    let outcome = (pair.second)(&second);
    first.commit().unwrap();
    if matches!(&outcome, Err(error) if is_conflict(error)) {
        (pair.second)(&second)
            .unwrap_or_else(|error| panic!("{}: the retried second MERGE: {error}", pair.name));
    }
    (outcome, elements(&db))
}

fn amsterdam_at(offset: &str) -> Value {
    Value::ZonedDatetime(ZonedDatetime::parse(&format!("2024-01-01T{offset}")).unwrap())
}

/// Pairs that each match what the other creates (a serial second run
/// matches), so they claim one key: the second fails with `GRAFEO-T001`
/// while the first is open, and its retry matches.
fn conflicting_pairs() -> Vec<Pair> {
    let mut pairs = vec![
        pair(
            "labels in either order",
            gql("MERGE (:A:B {k: 1})"),
            gql("MERGE (:B:A {k: 1})"),
        ),
        pair(
            "1 and 1.0",
            gql("MERGE (:A {k: 1})"),
            gql("MERGE (:A {k: 1.0})"),
        ),
        pair(
            "[1] and [1.0]",
            gql("MERGE (:A {k: [1]})"),
            gql("MERGE (:A {k: [1.0]})"),
        ),
        pair(
            "a null entry and no entry",
            gql("MERGE (:A {name: 'Alix', k: null})"),
            gql("MERGE (:A {name: 'Alix'})"),
        ),
        pair(
            "properties in either order",
            gql("MERGE (:A {a: 1, b: 2})"),
            gql("MERGE (:A {b: 2, a: 1})"),
        ),
        pair(
            "one instant at two offsets",
            gql_with(
                "MERGE (:A {t: $t})",
                &[("t", amsterdam_at("10:00:00+01:00"))],
            ),
            gql_with("MERGE (:A {t: $t})", &[("t", amsterdam_at("09:00:00Z"))]),
        ),
        pair(
            "a map with its items in either order",
            gql("MERGE (:A {m: {x: 1, y: 'Gus'}})"),
            gql("MERGE (:A {m: {y: 'Gus', x: 1.0}})"),
        ),
        pair(
            "a path and its reverse",
            gql("MERGE (:P {n: 'Vincent'})-[:KNOWS]->(:P {n: 'Jules'})"),
            gql("MERGE (:P {n: 'Jules'})<-[:KNOWS]-(:P {n: 'Vincent'})"),
        ),
        pair(
            "a path with edge properties",
            gql("MERGE (:P {n: 'Mia'})-[:KNOWS {since: 1994}]->(:P {n: 'Butch'})"),
            gql("MERGE (:P {n: 'Mia'})-[:KNOWS {since: 1994.0}]->(:P {n: 'Butch'})"),
        ),
        pair(
            "a two-hop path",
            gql(
                "MERGE (:Q {n: 'Amsterdam'})-[:ROAD]->(:Q {n: 'Utrecht'})-[:ROAD]->(:Q {n: 'Arnhem'})",
            ),
            gql(
                "MERGE (:Q {n: 'Arnhem'})<-[:ROAD]-(:Q {n: 'Utrecht'})<-[:ROAD]-(:Q {n: 'Amsterdam'})",
            ),
        ),
        pair("an empty pattern", gql("MERGE (n)"), gql("MERGE (m)")),
    ];
    #[cfg(feature = "cypher")]
    pairs.extend([
        pair(
            "GQL and Cypher",
            gql("MERGE (:A {k: 7})"),
            cypher("MERGE (:A {k: 7})"),
        ),
        pair(
            "a node MERGE and the Cypher MERGE that sets as a clause",
            gql("MERGE (:A {k: 8})"),
            cypher("MERGE (n:A {k: 8}) ON CREATE SET n += {made: true}"),
        ),
        pair(
            "the Cypher MERGE that sets a map and a node MERGE",
            cypher_with(
                "MERGE (n:A {k: 9}) ON CREATE SET n += $props",
                &[(
                    "props",
                    Value::Map(std::sync::Arc::new(
                        [(PropertyKey::new("city"), Value::from("Prague"))].into(),
                    )),
                )],
            ),
            cypher("MERGE (:A {k: 9})"),
        ),
        pair(
            "two undirected relationships in opposite orders",
            cypher("MERGE (:P {n: 'Django'})-[:KNOWS]-(:P {n: 'Hans'})"),
            cypher("MERGE (:P {n: 'Hans'})-[:KNOWS]-(:P {n: 'Django'})"),
        ),
        pair(
            "an undirected relationship and the direction it creates",
            cypher("MERGE (:P {n: 'Shosanna'})-[:KNOWS]-(:P {n: 'Beatrix'})"),
            cypher("MERGE (:P {n: 'Shosanna'})-[:KNOWS]->(:P {n: 'Beatrix'})"),
        ),
    ]);
    pairs
}

#[test]
fn merges_that_match_what_the_other_creates_conflict() {
    let mut wrong = Vec::new();
    for pair in conflicting_pairs() {
        let (after_first, after_both) = serial(&pair);
        if after_first != after_both {
            wrong.push(format!(
                "{}: the serial second run created ({after_first:?} to {after_both:?}), \
                 so this pair is not one pattern",
                pair.name
            ));
            continue;
        }
        let (outcome, elements) = concurrent(&pair);
        match &outcome {
            Err(error) if is_conflict(error) => {}
            other => wrong.push(format!(
                "{}: the second MERGE gave {other:?}, not GRAFEO-T001",
                pair.name
            )),
        }
        if elements != after_both {
            wrong.push(format!(
                "{}: {elements:?}, the serial run {after_both:?}",
                pair.name
            ));
        }
    }
    assert!(wrong.is_empty(), "{}", wrong.join("\n"));
}

/// Pairs that do not match what the other creates (a serial second run
/// creates): no conflict, and both create.
fn independent_pairs() -> Vec<Pair> {
    let mut pairs = vec![
        pair(
            "'1' and 1",
            gql("MERGE (:A {k: '1'})"),
            gql("MERGE (:A {k: 1})"),
        ),
        pair(
            ":A and :A:B",
            gql("MERGE (:A {k: 1})"),
            gql("MERGE (:A:B {k: 1})"),
        ),
        pair(
            "different values",
            gql("MERGE (:A {k: 1})"),
            gql("MERGE (:A {k: 2})"),
        ),
        pair(
            "different labels",
            gql("MERGE (:A {k: 1})"),
            gql("MERGE (:B {k: 1})"),
        ),
        pair(
            "a property more",
            gql("MERGE (:A {k: 1})"),
            gql("MERGE (:A {k: 1, city: 'Berlin'})"),
        ),
        pair(
            "NaN, which matches nothing",
            gql_with("MERGE (:A {k: $k})", &[("k", Value::Float64(f64::NAN))]),
            gql_with("MERGE (:A {k: $k})", &[("k", Value::Float64(f64::NAN))]),
        ),
        pair(
            "a list with a null, which matches nothing",
            gql("MERGE (:A {k: [1, null]})"),
            gql("MERGE (:A {k: [1, null]})"),
        ),
        pair(
            "a node and a path that holds it",
            gql("MERGE (:P {n: 'Vincent'})"),
            gql("MERGE (:P {n: 'Vincent'})-[:KNOWS]->(:P {n: 'Jules'})"),
        ),
        pair(
            "different relationship types",
            gql("MERGE (:P {n: 'Mia'})-[:KNOWS]->(:P {n: 'Butch'})"),
            gql("MERGE (:P {n: 'Mia'})-[:LIKES]->(:P {n: 'Butch'})"),
        ),
        Pair {
            name: "the same pattern in two graphs",
            first: gql("MERGE (:A {k: 1})"),
            second: gql("MERGE (:A {k: 1})"),
            first_in_g: true,
        },
    ];
    #[cfg(feature = "cypher")]
    pairs.push(pair(
        "a local and a zoned datetime",
        cypher("MERGE (:A {t: localdatetime('2024-01-01T10:00:00')})"),
        cypher("MERGE (:A {t: datetime('2024-01-01T10:00:00Z')})"),
    ));
    pairs
}

#[test]
fn merges_that_do_not_match_what_the_other_creates_do_not_conflict() {
    let mut wrong = Vec::new();
    for pair in independent_pairs() {
        let (after_first, after_both) = serial(&pair);
        if after_first == after_both {
            wrong.push(format!(
                "{}: the serial second run matched ({after_both:?}), so this pair is one pattern",
                pair.name
            ));
            continue;
        }
        let (outcome, elements) = concurrent(&pair);
        if let Err(error) = &outcome {
            wrong.push(format!("{}: the second MERGE failed: {error}", pair.name));
        }
        if elements != after_both {
            wrong.push(format!(
                "{}: {elements:?}, the serial run {after_both:?}",
                pair.name
            ));
        }
    }
    assert!(wrong.is_empty(), "{}", wrong.join("\n"));
}

// Row 10: a constraint and the claim together.

/// Under UNIQUE and under NODE KEY: two transactions MERGE one key, and a
/// MERGE races an INSERT of it. The second fails once with `GRAFEO-T001`
/// while the first is open; the retried MERGE matches and the retried INSERT
/// meets the constraint (G2000), as without the claim. One node each time.
#[test]
fn a_constraint_and_the_claim_report_one_conflict() {
    for constraint in [
        "CREATE CONSTRAINT memory_key FOR (m:Memory) ON (m.key) UNIQUE",
        "CREATE CONSTRAINT memory_key FOR (m:Memory) ON (m.key) NODE KEY",
    ] {
        for (second, retried_is_a_violation) in [
            ("MERGE (:Memory {key: 'ssi'})", false),
            ("INSERT (:Memory {key: 'ssi'})", true),
        ] {
            let what = format!("{constraint}, then {second}");
            let db = GrafeoDB::new_in_memory();
            db.execute(constraint).unwrap();
            let mut first = merging_ssi(&db);
            let error = db.execute(second).expect_err(&what);
            assert!(is_conflict(&error), "{what}: {error}");
            first.commit().unwrap();
            let retried = db.execute(second);
            if retried_is_a_violation {
                let error = retried.expect_err(&what);
                assert!(is_violation(&error), "{what}: {error}");
            } else {
                retried.unwrap_or_else(|error| panic!("{what}: {error}"));
            }
            assert_eq!(ssi_count(&db), 1, "{what}");
        }
    }
}

/// Ten sessions MERGE one constrained node at once: one node, all Ok.
#[test]
fn ten_sessions_merge_one_constrained_node_once() {
    for round in 0..10 {
        let db = GrafeoDB::new_in_memory();
        db.execute("CREATE CONSTRAINT memory_key FOR (m:Memory) ON (m.key) UNIQUE")
            .unwrap();
        let results = race(&db, 10, &|session, _| {
            session.execute("MERGE (:Memory {key: 'ssi'})").map(drop)
        });
        assert_all_ok(&results, &format!("round {round}"));
        assert_eq!(ssi_count(&db), 1, "round {round}");
    }
}

// Rows 11 and 12: many claims in one statement.

/// A statement claims once per row that creates; a row that matches what an
/// earlier row of the statement created claims nothing and does not conflict.
#[test]
fn a_statement_does_not_conflict_with_its_own_rows() {
    let db = GrafeoDB::new_in_memory();
    db.execute("UNWIND [1, 2, 1] AS x MERGE (:A {k: x})")
        .unwrap();
    assert_eq!(count(&db, "MATCH (n:A) RETURN count(n)"), 2);
    let mut session = db.session();
    in_transaction(&mut session, |session| {
        session.execute("UNWIND [3, 3, 2] AS x MERGE (:A {k: x})")?;
        session.execute("UNWIND [3, 4] AS x MERGE (:A {k: x})")?;
        Ok(())
    })
    .unwrap();
    assert_eq!(count(&db, "MATCH (n:A) RETURN count(n)"), 4);
}

/// The keys of each node `label` holds, counted: every key `0..keys` once.
fn assert_each_key_once(db: &GrafeoDB, label: &str, keys: i64, what: &str) {
    let result = db
        .execute(&format!(
            "MATCH (n:{label}) RETURN n.k AS k, count(n) AS c ORDER BY k"
        ))
        .unwrap();
    let got: Vec<(Value, i64)> = result
        .rows()
        .iter()
        .map(|row| (row[0].clone(), int(&row[1])))
        .collect();
    let want: Vec<(Value, i64)> = (0..keys).map(|k| (Value::Int64(k), 1)).collect();
    assert_eq!(got, want, "{what}");
}

/// The keys session `index` of a race MERGEs: overlapping lists, in an order
/// of its own.
fn overlapping_keys(index: usize) -> Value {
    let keys: Vec<Value> = (0..6_i64)
        .map(|k| Value::Int64((k + i64::try_from(index).unwrap()) % 6))
        .collect();
    Value::List(keys.into())
}

#[test]
fn eight_sessions_unwind_overlapping_merges_create_each_key_once() {
    for round in 0..10 {
        let db = GrafeoDB::new_in_memory();
        let results = race(&db, 8, &|session, index| {
            session
                .execute_with_params(
                    "UNWIND $keys AS x MERGE (:A {k: x})",
                    params(&[("keys", overlapping_keys(index))]),
                )
                .map(drop)
        });
        assert_all_ok(&results, &format!("round {round}"));
        assert_each_key_once(&db, "A", 6, &format!("round {round}: {results:?}"));
    }
}

#[cfg(feature = "cypher")]
#[test]
fn eight_sessions_foreach_overlapping_merges_create_each_key_once() {
    let db = GrafeoDB::new_in_memory();
    db.execute_cypher("FOREACH (x IN [1, 2, 1] | MERGE (:F {k: x}))")
        .unwrap();
    assert_eq!(count(&db, "MATCH (n:F) RETURN count(n)"), 2, "alone");
    for round in 0..10 {
        let db = GrafeoDB::new_in_memory();
        let results = race(&db, 8, &|session, index| {
            session
                .execute_cypher_with_params(
                    "FOREACH (x IN $keys | MERGE (:F {k: x}))",
                    params(&[("keys", overlapping_keys(index))]),
                )
                .map(drop)
        });
        assert_all_ok(&results, &format!("round {round}"));
        assert_each_key_once(&db, "F", 6, &format!("round {round}: {results:?}"));
    }
}

fn row(pairs: &[(&str, Value)]) -> HashMap<PropertyKey, Value> {
    pairs
        .iter()
        .map(|(name, value)| (PropertyKey::new(*name), value.clone()))
        .collect()
}

/// The keys every session of an upsert race writes, in one order: 0 to 5.
///
/// An upsert also updates the rows it matches (`SET n += row`), so sessions
/// that upsert overlapping rows in different orders update committed nodes
/// in a cycle. A write cycle on existing elements can still fail every
/// member on every retry until wait-die covers write claims, planned for
/// 0.6.4; the rotated create cycle is covered by the UNWIND, FOREACH and
/// UNIQUE races. Rotate these keys again once that lands.
fn ordered_keys() -> Value {
    Value::List((0..6_i64).map(Value::Int64).collect::<Vec<_>>().into())
}

/// Eight sessions upsert the same node rows at once: each key once.
#[test]
fn eight_sessions_upsert_nodes_create_each_key_once() {
    for round in 0..10 {
        let db = GrafeoDB::new_in_memory();
        let results = race(&db, 8, &|session, _| {
            let Value::List(keys) = ordered_keys() else {
                unreachable!()
            };
            let rows = keys
                .iter()
                .map(|k| row(&[("k", k.clone()), ("city", Value::from("Paris"))]))
                .collect();
            session.upsert_nodes(&["A"], "k", rows, false).map(drop)
        });
        assert_all_ok(&results, &format!("round {round}"));
        assert_each_key_once(&db, "A", 6, &format!("round {round}: {results:?}"));
    }
}

/// Eight sessions upsert the same edge rows between existing nodes at once:
/// each edge key once. The rows come in one order, as for the node upsert
/// (see [`ordered_keys`]): an edge upsert updates the edges it matches.
#[test]
fn eight_sessions_upsert_edges_create_each_key_once() {
    let options = grafeo_engine::database::EdgeUpsertOptions::new();
    for round in 0..10 {
        let db = GrafeoDB::new_in_memory();
        db.execute("UNWIND range(0, 6) AS i INSERT (:Stop {id: i})")
            .unwrap();
        let results = race(&db, 8, &|session, _| {
            let Value::List(keys) = ordered_keys() else {
                unreachable!()
            };
            let rows = keys
                .iter()
                .map(|k| {
                    let from = int(k);
                    row(&[
                        ("id", Value::Int64(from)),
                        ("src", Value::Int64(from)),
                        ("dst", Value::Int64(from + 1)),
                    ])
                })
                .collect();
            session.upsert_edges("ROAD", rows, &options).map(drop)
        });
        assert_all_ok(&results, &format!("round {round}"));
        let result = db
            .execute("MATCH ()-[e:ROAD]->() RETURN e.id AS k, count(e) AS c ORDER BY k")
            .unwrap();
        let got: Vec<(Value, i64)> = result
            .rows()
            .iter()
            .map(|row| (row[0].clone(), int(&row[1])))
            .collect();
        let want: Vec<(Value, i64)> = (0..6).map(|k| (Value::Int64(k), 1)).collect();
        assert_eq!(got, want, "round {round}: {results:?}");
    }
}

// Row 13: relationships and paths.

/// Ten sessions MERGE one relationship between two bound nodes: one edge.
#[test]
fn ten_sessions_merge_one_relationship_once() {
    for round in 0..20 {
        let db = GrafeoDB::new_in_memory();
        db.execute("INSERT (:P {n: 'Vincent'}), (:P {n: 'Jules'})")
            .unwrap();
        let results = race(&db, 10, &|session, _| {
            session
                .execute(
                    "MATCH (a:P {n: 'Vincent'}), (b:P {n: 'Jules'}) \
                     MERGE (a)-[:KNOWS {since: 1994}]->(b)",
                )
                .map(drop)
        });
        assert_all_ok(&results, &format!("round {round}"));
        assert_eq!(
            count(&db, "MATCH ()-[e:KNOWS]->() RETURN count(e)"),
            1,
            "round {round}: {results:?}"
        );
    }
}

/// Ten sessions MERGE one two-hop path: three nodes and two edges.
#[test]
fn ten_sessions_merge_one_path_once() {
    for round in 0..20 {
        let db = GrafeoDB::new_in_memory();
        let results = race(&db, 10, &|session, _| {
            session
                .execute(
                    "MERGE (:Q {n: 'Amsterdam'})-[:ROAD]->(:Q {n: 'Utrecht'})\
                     -[:ROAD]->(:Q {n: 'Arnhem'})",
                )
                .map(drop)
        });
        assert_all_ok(&results, &format!("round {round}"));
        assert_eq!(
            (
                count(&db, "MATCH (n:Q) RETURN count(n)"),
                count(&db, "MATCH ()-[e:ROAD]->() RETURN count(e)"),
            ),
            (3, 2),
            "round {round}: {results:?}"
        );
    }
}

/// Ten sessions run Cypher's `MERGE ... ON CREATE SET n += $props` of one
/// node: one node.
#[cfg(feature = "cypher")]
#[test]
fn ten_sessions_merge_and_set_a_map_once_in_cypher() {
    for round in 0..20 {
        let db = GrafeoDB::new_in_memory();
        let results = race(&db, 10, &|session, index| {
            let props = Value::Map(std::sync::Arc::new(
                [(
                    PropertyKey::new("by"),
                    Value::Int64(i64::try_from(index).unwrap()),
                )]
                .into(),
            ));
            session
                .execute_cypher_with_params(
                    "MERGE (n:Memory {key: $key}) ON CREATE SET n += $props",
                    params(&[("key", Value::from("ssi")), ("props", props)]),
                )
                .map(drop)
        });
        assert_all_ok(&results, &format!("round {round}"));
        assert_eq!(ssi_count(&db), 1, "round {round}: {results:?}");
    }
}

/// Ten sessions MERGE one undirected relationship, five in each order: one
/// edge.
#[cfg(feature = "cypher")]
#[test]
fn ten_sessions_merge_one_undirected_relationship_once_in_cypher() {
    for round in 0..20 {
        let db = GrafeoDB::new_in_memory();
        db.execute("INSERT (:P {n: 'Mia'}), (:P {n: 'Butch'})")
            .unwrap();
        let results = race(&db, 10, &|session, index| {
            let query = if index % 2 == 0 {
                "MATCH (a:P {n: 'Mia'}), (b:P {n: 'Butch'}) MERGE (a)-[:KNOWS]-(b)"
            } else {
                "MATCH (a:P {n: 'Mia'}), (b:P {n: 'Butch'}) MERGE (b)-[:KNOWS]-(a)"
            };
            session.execute_cypher(query).map(drop)
        });
        assert_all_ok(&results, &format!("round {round}"));
        assert_eq!(
            count(&db, "MATCH ()-[e:KNOWS]->() RETURN count(e)"),
            1,
            "round {round}: {results:?}"
        );
    }
}

// Row 17: the plan says what it guards.

/// The lines of the optimized plan of `statement` that create and claim a
/// MERGE pattern (`Create... claims MERGE pattern`), and the plan.
fn claiming_create_lines(result: &grafeo_engine::database::QueryResult) -> (usize, String) {
    let Value::String(plan) = &result.rows()[0][0] else {
        panic!("EXPLAIN returns its plan as text: {:?}", result.rows());
    };
    let lines = plan
        .lines()
        .filter(|line| {
            line.trim_start().starts_with("Create") && line.contains("claims MERGE pattern")
        })
        .count();
    (lines, plan.to_string())
}

/// The create branch of a MERGE that creates through a plain create (a path
/// MERGE, Cypher's MERGE that sets a map as a clause) carries the claim
/// through every optimizer rewrite: its optimized plan shows it on a Create
/// line. A plain INSERT or CREATE claims nothing.
#[test]
fn explain_shows_the_claim_on_the_create_line() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    let path = session
        .execute(
            "EXPLAIN MERGE (:Q {n: 'Amsterdam'})-[:ROAD]->(:Q {n: 'Utrecht'})\
             -[:ROAD]->(:Q {n: 'Arnhem'})",
        )
        .unwrap();
    let (lines, plan) = claiming_create_lines(&path);
    assert!(lines >= 1, "a path MERGE: {plan}");
    let insert = session
        .execute("EXPLAIN INSERT (:Q {n: 'Amsterdam'})-[:ROAD]->(:Q {n: 'Utrecht'})")
        .unwrap();
    let (lines, plan) = claiming_create_lines(&insert);
    assert_eq!(lines, 0, "a plain INSERT: {plan}");
    #[cfg(feature = "cypher")]
    {
        let set_a_map = session
            .execute_cypher_with_params(
                "EXPLAIN MERGE (n:A {k: $k}) ON CREATE SET n += $props",
                params(&[
                    ("k", Value::Int64(1)),
                    (
                        "props",
                        Value::Map(std::sync::Arc::new(
                            [(PropertyKey::new("city"), Value::from("Prague"))].into(),
                        )),
                    ),
                ]),
            )
            .unwrap();
        let (lines, plan) = claiming_create_lines(&set_a_map);
        assert!(lines >= 1, "Cypher's MERGE that sets a map: {plan}");
        let create = session
            .execute_cypher("EXPLAIN CREATE (n:A {k: 1})")
            .unwrap();
        let (lines, plan) = claiming_create_lines(&create);
        assert_eq!(lines, 0, "a plain CREATE: {plan}");
    }
}

// The oldest claim wins: progress when claims form a cycle.
//
// Of two open transactions that claim one MERGE pattern or one key under
// UNIQUE, PRIMARY KEY or NODE KEY, the one that began first wins. A younger
// requester fails at once with `GRAFEO-T001`. An older requester goes on,
// and the younger holder is doomed: its claims stop counting, its reads go
// on, and it fails with `GRAFEO-T001` at its next claim of any kind and at
// its commit. A holder that is committing beats any requester. So in a cycle
// of claims the oldest member always commits.

/// `count` sessions, each with an open transaction, begun in order: the
/// first is the oldest.
fn open_in_order(db: &GrafeoDB, count: usize) -> Vec<Session> {
    (0..count)
        .map(|_| {
            let mut session = db.session();
            session.begin_transaction().unwrap();
            session
        })
        .collect()
}

/// Asserts that `outcome` is the retryable `GRAFEO-T001`.
fn assert_conflict<T: std::fmt::Debug>(outcome: Result<T, Error>, what: &str) {
    match outcome {
        Err(error) if is_conflict(&error) => {}
        other => panic!("{what}: {other:?}, not GRAFEO-T001"),
    }
}

/// A three-transaction cycle, with `claim(session, k)` writing the key `k`:
/// each holds its own key, then the oldest claims the middle one's, the
/// middle one the youngest's, and the youngest the oldest's. The oldest goes
/// on and commits; the middle one is doomed by it and fails at its next
/// claim and at its commit; the youngest fails at once against the oldest.
/// Their retries then finish the work: each key once.
fn a_cycle_of_three_lets_the_oldest_commit(
    db: &GrafeoDB,
    claim: fn(&Session, &str) -> Result<(), Error>,
    keys: fn(&GrafeoDB) -> Vec<(String, i64)>,
) {
    let mut sessions = open_in_order(db, 3);
    for (session, key) in sessions.iter().zip(["k1", "k2", "k3"]) {
        claim(session, key).unwrap_or_else(|error| panic!("{key} alone: {error}"));
    }
    claim(&sessions[0], "k2").expect("the oldest claims the middle one's key and goes on");
    assert_conflict(
        claim(&sessions[1], "k3"),
        "the doomed middle one's next claim",
    );
    assert_conflict(claim(&sessions[2], "k1"), "the youngest against the oldest");
    sessions[0].commit().expect("the oldest commits");
    assert_conflict(sessions[1].commit(), "the doomed middle one's commit");
    if sessions[2].in_transaction() {
        sessions[2].rollback().unwrap();
    }
    assert_eq!(
        keys(db),
        [("k1".to_string(), 1), ("k2".to_string(), 1)],
        "only the oldest's work is committed"
    );
    // The retries, as an application runs them.
    in_transaction(&mut sessions[1], |session| {
        claim(session, "k2")?;
        claim(session, "k3")
    })
    .unwrap();
    in_transaction(&mut sessions[2], |session| {
        claim(session, "k3")?;
        claim(session, "k1")
    })
    .unwrap();
    assert_eq!(
        keys(db),
        [
            ("k1".to_string(), 1),
            ("k2".to_string(), 1),
            ("k3".to_string(), 1)
        ]
    );
}

/// The `:Memory` keys and how many nodes hold each, in key order.
fn memory_keys(db: &GrafeoDB) -> Vec<(String, i64)> {
    db.execute("MATCH (m:Memory) RETURN m.key AS k, count(m) AS c ORDER BY k")
        .unwrap()
        .rows()
        .iter()
        .map(|row| match &row[0] {
            Value::String(key) => (key.to_string(), int(&row[1])),
            other => panic!("a key, got {other:?}"),
        })
        .collect()
}

fn merge_memory(session: &Session, key: &str) -> Result<(), Error> {
    session
        .execute_with_params(
            "MERGE (:Memory {key: $key})",
            params(&[("key", Value::from(key))]),
        )
        .map(drop)
}

fn insert_memory(session: &Session, key: &str) -> Result<(), Error> {
    session
        .execute_with_params(
            "INSERT (:Memory {key: $key})",
            params(&[("key", Value::from(key))]),
        )
        .map(drop)
}

/// Row 1: a cycle of MERGE patterns.
#[test]
fn a_cycle_of_merge_patterns_lets_the_oldest_commit() {
    let db = GrafeoDB::new_in_memory();
    a_cycle_of_three_lets_the_oldest_commit(&db, merge_memory, memory_keys);
}

/// Row 2: the same cycle with UNIQUE keys, written by INSERT. The retries
/// MERGE, since an INSERT of a key the oldest committed meets the
/// constraint.
#[test]
fn a_cycle_of_unique_keys_lets_the_oldest_commit() {
    let db = GrafeoDB::new_in_memory();
    db.execute("CREATE CONSTRAINT memory_key FOR (m:Memory) ON (m.key) UNIQUE")
        .unwrap();
    let mut sessions = open_in_order(&db, 3);
    for (session, key) in sessions.iter().zip(["k1", "k2", "k3"]) {
        insert_memory(session, key).unwrap_or_else(|error| panic!("{key} alone: {error}"));
    }
    insert_memory(&sessions[0], "k2").expect("the oldest claims the middle one's key and goes on");
    assert_conflict(
        insert_memory(&sessions[1], "k3"),
        "the doomed middle one's next claim",
    );
    assert_conflict(
        insert_memory(&sessions[2], "k1"),
        "the youngest against the oldest",
    );
    sessions[0].commit().expect("the oldest commits");
    assert_conflict(sessions[1].commit(), "the doomed middle one's commit");
    if sessions[2].in_transaction() {
        sessions[2].rollback().unwrap();
    }
    assert_eq!(
        memory_keys(&db),
        [("k1".to_string(), 1), ("k2".to_string(), 1)]
    );
    in_transaction(&mut sessions[1], |session| {
        merge_memory(session, "k2")?;
        merge_memory(session, "k3")
    })
    .unwrap();
    assert_eq!(
        memory_keys(&db),
        [
            ("k1".to_string(), 1),
            ("k2".to_string(), 1),
            ("k3".to_string(), 1)
        ]
    );
}

/// Row 3: a doomed transaction's reads go on, and it fails at its commit
/// even when it claims nothing more; a write fails at once.
#[test]
fn a_doomed_transaction_reads_and_fails_at_its_next_write_and_commit() {
    for writes in [false, true] {
        let db = GrafeoDB::new_in_memory();
        let mut sessions = open_in_order(&db, 2);
        merge_memory(&sessions[1], "ssi").unwrap();
        merge_memory(&sessions[0], "ssi").expect("the older claims the pattern and goes on");
        let read = sessions[1]
            .execute("MATCH (m:Memory) RETURN count(m) AS c")
            .expect("a doomed transaction still reads");
        assert_eq!(int(&read.rows()[0][0]), 1, "its own node");
        if writes {
            assert_conflict(
                sessions[1].execute("MATCH (m:Memory {key: 'ssi'}) SET m.city = 'Paris'"),
                "a doomed transaction's write",
            );
        }
        assert_conflict(sessions[1].commit(), "a doomed transaction's commit");
        sessions[0].commit().unwrap();
        assert_eq!(ssi_count(&db), 1, "writes: {writes}");
    }
}

/// Row 4: a younger requester against an older open holder fails at once,
/// for a MERGE pattern and for a UNIQUE key, and the holder is not harmed.
#[test]
fn a_younger_requester_fails_at_once() {
    for (constraint, claim) in [
        (
            None,
            merge_memory as fn(&Session, &str) -> Result<(), Error>,
        ),
        (
            Some("CREATE CONSTRAINT memory_key FOR (m:Memory) ON (m.key) UNIQUE"),
            insert_memory,
        ),
    ] {
        let db = GrafeoDB::new_in_memory();
        if let Some(constraint) = constraint {
            db.execute(constraint).unwrap();
        }
        let mut sessions = open_in_order(&db, 2);
        claim(&sessions[0], "ssi").unwrap();
        assert_conflict(claim(&sessions[1], "ssi"), "the younger requester");
        sessions[0].commit().expect("the older holder commits");
        sessions[1].rollback().unwrap();
        assert_eq!(ssi_count(&db), 1, "{constraint:?}");
    }
}

/// Row 5: a holder that is committing beats any requester, even an older
/// one: the requester fails, and the holder's commit goes through. The
/// requester's MERGE runs from inside the holder's commit, right after its
/// epoch is assigned, on its own thread.
#[cfg(feature = "testing-statement-injection")]
#[test]
fn a_requester_against_a_committing_holder_fails() {
    use std::sync::{Arc, Mutex};
    for (constraint, claim) in [
        (
            None,
            merge_memory as fn(&Session, &str) -> Result<(), Error>,
        ),
        (
            Some("CREATE CONSTRAINT memory_key FOR (m:Memory) ON (m.key) UNIQUE"),
            insert_memory,
        ),
    ] {
        let db = GrafeoDB::new_in_memory();
        if let Some(constraint) = constraint {
            db.execute(constraint).unwrap();
        }
        let mut older = db.session();
        older.begin_transaction().unwrap();
        let mut younger = db.session();
        younger.begin_transaction().unwrap();
        claim(&younger, "ssi").unwrap();
        let outcome: Arc<Mutex<Option<(Session, Result<(), Error>)>>> = Arc::default();
        let slot = Arc::clone(&outcome);
        grafeo_common::testing::commit_hook::after_next_commit_epoch(move || {
            let (sender, receiver) = std::sync::mpsc::channel();
            std::thread::spawn(move || {
                let result = claim(&older, "ssi");
                sender.send((older, result)).unwrap();
            });
            let done = receiver
                .recv_timeout(std::time::Duration::from_secs(30))
                .expect("the requester does not wait for the commit");
            *slot.lock().unwrap() = Some(done);
        });
        younger
            .commit()
            .expect("the committing holder's commit goes through");
        let (mut older, result) = outcome
            .lock()
            .unwrap()
            .take()
            .expect("the hook ran inside the commit");
        assert_conflict(result, "an older requester against a committing holder");
        older.rollback().unwrap();
        assert_eq!(ssi_count(&db), 1, "{constraint:?}");
    }
}

/// Row 6: a doomed holder's claims do not block a third transaction. The
/// middle one holds two patterns; the oldest dooms it on one; the youngest
/// then claims the other without a conflict, and commits.
#[test]
fn a_doomed_holder_does_not_block_a_third_transaction() {
    let db = GrafeoDB::new_in_memory();
    let mut sessions = open_in_order(&db, 3);
    merge_memory(&sessions[1], "k1").unwrap();
    merge_memory(&sessions[1], "k2").unwrap();
    merge_memory(&sessions[0], "k1").expect("the oldest dooms the middle one");
    merge_memory(&sessions[2], "k2").expect("the doomed holder's claim does not count");
    sessions[0].commit().unwrap();
    sessions[2].commit().unwrap();
    assert_conflict(sessions[1].commit(), "the doomed holder's commit");
    assert_eq!(
        memory_keys(&db),
        [("k1".to_string(), 1), ("k2".to_string(), 1)]
    );
}

/// Row 7: a rollback to a savepoint undoes the claims since it, but not the
/// doom: the transaction fails at its commit all the same.
#[test]
fn a_rollback_to_a_savepoint_keeps_a_transaction_doomed() {
    let db = GrafeoDB::new_in_memory();
    let mut sessions = open_in_order(&db, 2);
    sessions[1].savepoint("before").unwrap();
    merge_memory(&sessions[1], "ssi").unwrap();
    merge_memory(&sessions[0], "ssi").expect("the older dooms the younger");
    sessions[1].rollback_to_savepoint("before").unwrap();
    sessions[1]
        .execute("MATCH (m:Memory) RETURN count(m) AS c")
        .expect("its reads go on");
    assert_conflict(sessions[1].commit(), "the doomed transaction's commit");
    sessions[0].commit().unwrap();
    assert_eq!(ssi_count(&db), 1);
}

/// Eight sessions MERGE overlapping keys in rotated orders under a UNIQUE
/// constraint, so key claims and pattern claims meet: each key once, and
/// every session finishes.
#[test]
fn eight_sessions_unwind_merges_under_unique_create_each_key_once() {
    for round in 0..10 {
        let db = GrafeoDB::new_in_memory();
        db.execute("CREATE CONSTRAINT a_key FOR (n:A) ON (n.k) UNIQUE")
            .unwrap();
        let results = race(&db, 8, &|session, index| {
            session
                .execute_with_params(
                    "UNWIND $keys AS x MERGE (:A {k: x})",
                    params(&[("keys", overlapping_keys(index))]),
                )
                .map(drop)
        });
        assert_all_ok(&results, &format!("round {round}"));
        assert_each_key_once(&db, "A", 6, &format!("round {round}: {results:?}"));
    }
}

// A MERGE that only matches writes nothing, so it claims nothing.

/// `statement` (with `pairs` as its parameters) runs in two open
/// transactions over the committed state that `setup` wrote; both must
/// commit. Returns what failed, for the message.
fn both_commit(setup: &str, statement: &str, pairs: &[(&str, Value)]) -> Result<(), String> {
    let db = GrafeoDB::new_in_memory();
    db.execute(setup).unwrap();
    let mut first = db.session();
    let mut second = db.session();
    first.begin_transaction().unwrap();
    second.begin_transaction().unwrap();
    first
        .execute_with_params(statement, params(pairs))
        .map_err(|error| format!("the first transaction's statement: {error}"))?;
    second
        .execute_with_params(statement, params(pairs))
        .map_err(|error| format!("the second transaction's statement: {error}"))?;
    first
        .commit()
        .map_err(|error| format!("the first commit: {error}"))?;
    second
        .commit()
        .map_err(|error| format!("the second commit: {error}"))
}

const MEMORY: &str = "INSERT (:Memory {key: 'ssi', city: 'Prague'})";

/// Two open transactions MERGE the same committed node without ON MATCH:
/// each only matches, writes nothing and claims nothing, so both commit and
/// the node is unchanged. Before, each matched node was set with an empty
/// list, which claimed a write, so the second failed with `GRAFEO-T001`.
#[test]
fn two_merges_that_only_match_both_commit() {
    let db = GrafeoDB::new_in_memory();
    db.execute(MEMORY).unwrap();
    let mut first = db.session();
    let mut second = db.session();
    first.begin_transaction().unwrap();
    second.begin_transaction().unwrap();
    first.execute("MERGE (:Memory {key: 'ssi'})").unwrap();
    second
        .execute("MERGE (:Memory {key: 'ssi'})")
        .expect("a MERGE that only matches claims no write");
    first.commit().unwrap();
    second.commit().expect("nothing of it conflicts at commit");
    assert_eq!(ssi_count(&db), 1);
    assert_eq!(
        count(
            &db,
            "MATCH (m:Memory {key: 'ssi', city: 'Prague'}) RETURN count(m)"
        ),
        1,
        "the node is as it was"
    );
}

/// The write cycle of the release races: two transactions MERGE two
/// committed nodes in opposite orders, without ON MATCH. Nothing is written,
/// so both commit.
#[test]
fn merges_that_only_match_in_opposite_orders_both_commit() {
    let db = GrafeoDB::new_in_memory();
    db.execute("INSERT (:A {k: 0}), (:A {k: 1})").unwrap();
    let mut first = db.session();
    let mut second = db.session();
    first.begin_transaction().unwrap();
    second.begin_transaction().unwrap();
    first.execute("MERGE (:A {k: 0})").unwrap();
    second.execute("MERGE (:A {k: 1})").unwrap();
    first
        .execute("MERGE (:A {k: 1})")
        .expect("the first matches the second's node");
    second
        .execute("MERGE (:A {k: 0})")
        .expect("the second matches the first's node");
    first.commit().unwrap();
    second.commit().unwrap();
    assert_eq!(count(&db, "MATCH (n:A) RETURN count(n)"), 2);
}

/// The other statements that only match or can change no property: each
/// runs in two open transactions and both commit. A set that can change no
/// property claims nothing, on a node or an edge.
/// - `SET n += {}` and `SET n += $empty` (an empty map parameter) on a
///   committed node, `SET n += $null` and `SET n = $null` (a null
///   parameter), and `SET r += {}` on a committed edge.
/// - A Cypher node MERGE without ON MATCH, and with `ON MATCH SET n += {}`.
/// - A relationship MERGE between two bound nodes, and a path MERGE, that
///   match: their match branch never wrote, and must not start to.
#[test]
fn statements_that_change_nothing_both_commit() {
    let path = "INSERT (:P {n: 'Vincent'})-[:KNOWS {since: 1994}]->(:P {n: 'Jules'})";
    let empty = Value::Map(std::sync::Arc::new(std::collections::BTreeMap::new()));
    let cases: Vec<(&str, &str, &str, Vec<(&str, Value)>)> = vec![
        (
            "SET n += {}",
            MEMORY,
            "MATCH (m:Memory {key: 'ssi'}) SET m += {}",
            vec![],
        ),
        (
            "SET n += $empty",
            MEMORY,
            "MATCH (m:Memory {key: 'ssi'}) SET m += $empty",
            vec![("empty", empty)],
        ),
        (
            "SET n += $null",
            MEMORY,
            "MATCH (m:Memory {key: 'ssi'}) SET m += $nothing",
            vec![("nothing", Value::Null)],
        ),
        (
            "SET n = $null",
            MEMORY,
            "MATCH (m:Memory {key: 'ssi'}) SET m = $nothing",
            vec![("nothing", Value::Null)],
        ),
        (
            "SET r += {} on an edge",
            path,
            "MATCH ()-[r:KNOWS]->() SET r += {}",
            vec![],
        ),
        (
            "a relationship MERGE that matches",
            path,
            "MATCH (a:P {n: 'Vincent'}), (b:P {n: 'Jules'}) MERGE (a)-[:KNOWS]->(b)",
            vec![],
        ),
        (
            "a path MERGE that matches",
            path,
            "MERGE (:P {n: 'Vincent'})-[:KNOWS]->(:P {n: 'Jules'})",
            vec![],
        ),
    ];
    let mut wrong = Vec::new();
    for (name, setup, statement, pairs) in &cases {
        if let Err(failure) = both_commit(setup, statement, pairs) {
            wrong.push(format!("{name}: {failure}"));
        }
    }
    #[cfg(feature = "cypher")]
    for (name, statement) in [
        (
            "a Cypher node MERGE without ON MATCH",
            "MERGE (:Memory {key: 'ssi'})",
        ),
        (
            "a Cypher node MERGE with ON MATCH SET n += {}",
            "MERGE (m:Memory {key: 'ssi'}) ON MATCH SET m += {}",
        ),
    ] {
        let db = GrafeoDB::new_in_memory();
        db.execute(MEMORY).unwrap();
        let mut first = db.session();
        let mut second = db.session();
        first.begin_transaction().unwrap();
        second.begin_transaction().unwrap();
        let outcome = first
            .execute_cypher(statement)
            .and_then(|_| second.execute_cypher(statement))
            .and_then(|_| first.commit())
            .and_then(|()| second.commit());
        if let Err(error) = outcome {
            wrong.push(format!("{name}: {error}"));
        }
    }
    assert!(wrong.is_empty(), "{}", wrong.join("\n"));
}

/// `SET n = {}` and `SET r = {}` replace the properties, so they write the
/// element and claim it: the second transaction's SET fails with
/// `GRAFEO-T001`, as any write of one element by two open transactions
/// does.
#[test]
fn set_to_an_empty_map_still_conflicts() {
    for (setup, statement) in [
        (MEMORY, "MATCH (m:Memory {key: 'ssi'}) SET m = {}"),
        (
            "INSERT (:P {n: 'Vincent'})-[:KNOWS {since: 1994}]->(:P {n: 'Jules'})",
            "MATCH ()-[r:KNOWS]->() SET r = {}",
        ),
    ] {
        let db = GrafeoDB::new_in_memory();
        db.execute(setup).unwrap();
        let mut first = db.session();
        let mut second = db.session();
        first.begin_transaction().unwrap();
        second.begin_transaction().unwrap();
        first.execute(statement).unwrap();
        let error = second
            .execute(statement)
            .expect_err("a replace writes the element");
        assert!(is_conflict(&error), "{statement}: {error}");
        first.commit().unwrap();
        second.rollback().unwrap();
    }
}

// Rows 14 and 15: claims are not persisted.

#[cfg(all(feature = "wal", feature = "grafeo-file"))]
use crate::common::replay;

#[cfg(all(feature = "wal", feature = "grafeo-file"))]
mod persistent {
    use super::*;

    /// A crash while a transaction holds a claim: the reopened database has
    /// no claim (the WAL replay brings none back), so a MERGE creates one.
    #[test]
    fn a_crash_with_an_open_claim_leaves_none() {
        let (_dir, db) = replay::reopened_after_crash(
            &grafeo_common::test_path!("a_crash_with_an_open_claim_leaves_none"),
            |path| GrafeoDB::open(path).unwrap(),
            |db| {
                db.execute("MERGE (:Memory {key: 'committed'})").unwrap();
                let _open = merging_ssi(db);
                // Crash with the transaction open: no rollback, no close.
                std::process::exit(0);
            },
        );
        assert_eq!(
            count(&db, "MATCH (m:Memory {key: 'committed'}) RETURN count(m)"),
            1
        );
        assert_eq!(ssi_count(&db), 0, "the open transaction's node is gone");
        db.execute("MERGE (:Memory {key: 'ssi'})")
            .expect("no claim survives the crash");
        assert_eq!(ssi_count(&db), 1);
    }

    /// A close with a transaction open, then a reopen: a MERGE creates one,
    /// without a conflict.
    #[test]
    fn a_close_with_an_open_claim_leaves_none() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("claims.grafeo");
        let db = GrafeoDB::open(&path).unwrap();
        let open = merging_ssi(&db);
        let _ = db.close();
        drop(open);
        drop(db);
        let db = GrafeoDB::open(&path).unwrap();
        db.execute("MERGE (:Memory {key: 'ssi'})")
            .expect("no claim survives the close");
        assert_eq!(ssi_count(&db), 1);
        db.close().unwrap();
    }
}
