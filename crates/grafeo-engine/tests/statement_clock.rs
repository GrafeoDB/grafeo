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

/// SPARQL's `NOW()` is one instant for the whole query (17.4.5.1), over
/// enough rows that a clock read per row would tick.
#[cfg(all(feature = "sparql", feature = "triple-store"))]
#[test]
fn sparql_now_is_one_instant_per_query() {
    use grafeo_engine::{Config, GraphModel};
    use std::fmt::Write as _;
    let db = GrafeoDB::with_config(Config::in_memory().with_graph_model(GraphModel::Rdf)).unwrap();
    let mut data = String::from("INSERT DATA {\n");
    for i in 0..50_000 {
        writeln!(
            data,
            "<http://example.org/e{i}> <http://example.org/n> {i} ."
        )
        .unwrap();
    }
    data.push('}');
    db.execute_sparql(&data).unwrap();
    let result = db
        .execute_sparql(
            "SELECT (COUNT(DISTINCT ?t) AS ?instants) (SAMPLE(?same) AS ?agree) WHERE { \
             ?s <http://example.org/n> ?n BIND(NOW() AS ?t) BIND(NOW() = NOW() AS ?same) }",
        )
        .unwrap();
    let row = &result.rows()[0];
    assert_eq!(
        row[0].to_string(),
        "\"1\"^^<http://www.w3.org/2001/XMLSchema#integer>"
    );
    assert!(
        row[1].to_string().starts_with("\"true\""),
        "two NOW() in a row agree: {:?}",
        row[1]
    );
}

/// A procedure's body runs in the statement that calls it: every call reads
/// the statement's instant, not its own. A local datetime has nanosecond
/// resolution, so two calls of a per-call clock never agree.
#[cfg(feature = "cypher")]
#[test]
fn a_procedure_body_reads_the_calling_statements_clock() {
    let db = GrafeoDB::new_in_memory();
    db.execute("CREATE PROCEDURE stamp() RETURNS (t ANY) AS { RETURN localdatetime() AS t }")
        .unwrap();
    let result = db
        .execute_cypher(
            "UNWIND range(1, 5000) AS i CALL stamp() YIELD t              RETURN count(DISTINCT t) AS instants, max(t) = localdatetime() AS same",
        )
        .unwrap();
    assert_eq!(
        result.rows()[0],
        [Value::Int64(1), Value::Bool(true)],
        "every call sees the statement's instant"
    );
}
