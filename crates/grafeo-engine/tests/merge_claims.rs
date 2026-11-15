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

/// `sessions` sessions start `work` at once (a barrier) and retry it on the
/// write conflict, as an application does. Each result is the number of
/// retries, or the error that was not a write conflict.
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
                                std::thread::yield_now();
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

/// Eight sessions upsert overlapping node rows at once: each key once.
#[test]
fn eight_sessions_upsert_nodes_create_each_key_once() {
    for round in 0..10 {
        let db = GrafeoDB::new_in_memory();
        let results = race(&db, 8, &|session, index| {
            let Value::List(keys) = overlapping_keys(index) else {
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

/// Eight sessions upsert overlapping edge rows between existing nodes at
/// once: each edge key once.
#[test]
fn eight_sessions_upsert_edges_create_each_key_once() {
    let options = grafeo_engine::database::EdgeUpsertOptions::new();
    for round in 0..10 {
        let db = GrafeoDB::new_in_memory();
        db.execute("UNWIND range(0, 6) AS i INSERT (:Stop {id: i})")
            .unwrap();
        let results = race(&db, 8, &|session, index| {
            let Value::List(keys) = overlapping_keys(index) else {
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
