//! A label scan reads its node ids a page at a time and counts the page
//! against the query memory limit (0.6.4): under a limit below the label's
//! whole list, `MATCH (n:Person) ... LIMIT 10` and a count answer; below one
//! page the scan fails with `GRAFEO-S001` naming the query memory limit; and a
//! statement that writes per scanned node writes once per node there was,
//! over many pages, in auto-commit and in a transaction.
//!
//! ```bash
//! cargo test -p grafeo-engine --all-features --test main scan_grants::
//! ```

#![cfg(feature = "gql")]

use grafeo_common::memory::buffer::MemoryRegion;
use grafeo_common::types::Value;
use grafeo_common::utils::error::{Error, LimitBound, LimitRefusal};
use grafeo_engine::{Config, GrafeoDB};

const KIB: usize = 1024;

/// Persons: a list of ids of 160 KB, about ten pages.
const PERSONS: usize = 20_000;

/// A count over a scan of every Person: `count(n)` alone reads the label's
/// count without a scan, so the count is of a property.
const COUNT: &str = "MATCH (n:Person) RETURN count(n.name)";

/// A database of `persons` Persons, each with a name.
fn persons(config: Config, persons: usize) -> GrafeoDB {
    let db = GrafeoDB::with_config(config).unwrap();
    db.execute(&format!(
        "FOR i IN range(1, {persons}) INSERT (:Person {{name: 'Alix'}})"
    ))
    .unwrap();
    db
}

/// The bytes the statements of `db` hold.
fn execution_bytes(db: &GrafeoDB) -> usize {
    db.buffer_manager().stats().region_allocated[MemoryRegion::ExecutionBuffers.index()]
}

/// The bound `error` names: `GRAFEO-S001` of a memory bound.
fn refused_by(error: &Error) -> LimitBound {
    assert_eq!(error.error_code().as_str(), "GRAFEO-S001", "{error}");
    match error.limit_refusal() {
        Some(LimitRefusal::Memory { bound, .. }) => *bound,
        other => panic!("a memory bound's refusal, got {other:?}: {error}"),
    }
}

/// The one value of a one-row, one-column result.
fn single(db: &GrafeoDB, query: &str) -> Value {
    let result = db.execute(query).unwrap();
    assert_eq!(result.rows().len(), 1, "{query}");
    result.rows()[0][0].clone()
}

/// `count` as the integer a query returns.
fn int(count: usize) -> Value {
    Value::Int64(i64::try_from(count).unwrap())
}

/// How many nodes have `label`.
fn count(db: &GrafeoDB, label: &str) -> Value {
    single(db, &format!("MATCH (n:{label}) RETURN count(n)"))
}

#[test]
fn a_label_scan_under_a_limit_below_its_list_answers() {
    let db = persons(
        Config::in_memory().with_query_memory_limit(64 * KIB),
        PERSONS,
    );
    let result = db.execute("MATCH (n:Person) RETURN n LIMIT 10").unwrap();
    assert_eq!(result.rows().len(), 10);
    assert_eq!(single(&db, COUNT), int(PERSONS));
}

#[test]
fn a_label_scan_below_one_page_refuses() {
    let db = persons(
        Config::in_memory().with_query_memory_limit(8 * KIB),
        PERSONS,
    );
    let before = execution_bytes(&db);
    let error = db.execute(COUNT).expect_err("a page passes the limit");
    assert_eq!(refused_by(&error), LimitBound::QueryMemoryLimit);
    assert!(
        error
            .to_string()
            .contains(LimitBound::QueryMemoryLimit.name()),
        "{error}"
    );
    assert_eq!(execution_bytes(&db), before, "given back");
    assert_eq!(db.node_count().unwrap(), PERSONS, "changes nothing");
}

#[test]
fn a_label_scan_cut_short_by_a_limit_gives_its_grant_back() {
    let db = persons(Config::in_memory(), PERSONS);
    let before = execution_bytes(&db);
    let result = db.execute("MATCH (n:Person) RETURN n LIMIT 1").unwrap();
    assert_eq!(result.rows().len(), 1);
    assert_eq!(execution_bytes(&db), before, "given back");
}

/// A create per scanned Person, over about three pages, creates one per
/// Person there was: the scan never reads the Persons it creates.
#[test]
fn a_create_per_person_over_several_pages_creates_one_each() {
    let people = 5_000;
    let db = persons(Config::in_memory(), people);
    db.execute("MATCH (p:Person) CREATE (:Person)").unwrap();
    assert_eq!(count(&db, "Person"), int(2 * people), "auto-commit");

    let db = persons(Config::in_memory(), people);
    let mut session = db.session();
    session.begin_transaction().unwrap();
    session
        .execute("MATCH (p:Person) CREATE (:Person)")
        .unwrap();
    let seen = session.execute("MATCH (n:Person) RETURN count(n)").unwrap();
    assert_eq!(seen.rows()[0][0], int(2 * people), "in the transaction");
    session.commit().unwrap();
    assert_eq!(count(&db, "Person"), int(2 * people), "committed");
}

/// A label set per scanned Person, over about three pages, labels each.
#[test]
fn a_label_set_per_person_over_several_pages_labels_each() {
    let people = 5_000;
    let db = persons(Config::in_memory(), people);
    db.execute("MATCH (p:Person) SET p:Seen").unwrap();
    assert_eq!(count(&db, "Seen"), int(people), "auto-commit");

    let db = persons(Config::in_memory(), people);
    let mut session = db.session();
    session.begin_transaction().unwrap();
    session.execute("MATCH (p:Person) SET p:Seen").unwrap();
    session.commit().unwrap();
    assert_eq!(count(&db, "Seen"), int(people), "in a transaction");
    assert_eq!(count(&db, "Person"), int(people));
}
