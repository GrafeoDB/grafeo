//! Signed integers, negative decimals and negative floats keep their type and
//! value through a compact and a reopen, and filters, ranges and sums over
//! them agree with and without a property index (#160 and #301 reported the
//! two halves: negative literals in a property map, and signed integers that
//! became floats or strings after a compact).
//!
//! ```bash
//! cargo test -p grafeo-engine --all-features --test signed_values_after_reopen
//! ```

#![cfg(all(feature = "gql", feature = "grafeo-file", feature = "wal"))]

use std::path::Path;

use grafeo_common::types::Value;
use grafeo_engine::{Config, GrafeoDB};

fn open(path: &Path) -> GrafeoDB {
    GrafeoDB::with_config(Config::persistent(path)).unwrap()
}

/// The first row of a GQL query.
fn row(db: &GrafeoDB, query: &str) -> Vec<Value> {
    db.execute(query)
        .unwrap_or_else(|error| panic!("{query}: {error}"))
        .rows()[0]
        .clone()
}

/// Every check of the signed values, as one row each.
fn check(db: &GrafeoDB, stage: &str) {
    // -10000 to 9999: an integer sum of -10000, half the values negative
    assert_eq!(
        row(db, "MATCH (n:N) RETURN sum(n.v) AS s, count(n) AS c"),
        [Value::Int64(-10_000), Value::Int64(20_000)],
        "{stage}: sum and count"
    );
    assert_eq!(
        row(
            db,
            "MATCH (n:N) WHERE n.v < 0 RETURN count(n) AS c, min(n.v) AS lo, max(n.v) AS hi"
        ),
        [
            Value::Int64(10_000),
            Value::Int64(-10_000),
            Value::Int64(-1)
        ],
        "{stage}: the negative half"
    );
    assert_eq!(
        row(
            db,
            "MATCH (n:N) WHERE n.v = -7 RETURN count(n) AS c, n.v IS TYPED INT64 AS t"
        ),
        [Value::Int64(1), Value::Bool(true)],
        "{stage}: an exact negative lookup"
    );
    assert_eq!(
        row(
            db,
            "MATCH (n:N) WHERE n.v >= -3 AND n.v <= 3 RETURN count(n) AS c"
        ),
        [Value::Int64(7)],
        "{stage}: a range across zero"
    );
    // The negative literals of #160, stored by GQL
    assert_eq!(
        row(
            db,
            "MATCH (l:Location) RETURN l.a, l.b, l.c, l.d, l.e, l.a IS TYPED DECIMAL AS dec, \
             l.d IS TYPED FLOAT64 AS float"
        ),
        [
            Value::Decimal(grafeo_common::types::Decimal::parse("-6.248").unwrap()),
            Value::Decimal(grafeo_common::types::Decimal::parse("-6.248").unwrap()),
            Value::Int64(-6),
            Value::Float64(-33.8688),
            Value::Int64(i64::MIN),
            Value::Bool(true),
            Value::Bool(true),
        ],
        "{stage}: negative literals"
    );
    assert_eq!(
        row(
            db,
            "MATCH (l:Location) WHERE l.a = -6.248 AND l.d < -33 RETURN count(l) AS c"
        ),
        [Value::Int64(1)],
        "{stage}: filters on negative literals"
    );
}

fn write(db: &GrafeoDB) {
    db.execute("FOR i IN range(-10000, 9999) INSERT (:N {v: i})")
        .unwrap();
    db.execute(
        "INSERT (:Location {a: (-6.248), b: 0 - 6.248, c: -6, d: -33.8688e0, \
         e: -9223372036854775807 - 1})",
    )
    .unwrap();
}

#[test]
fn signed_values_survive_compact_reopen_and_an_index() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("signed.grafeo");
    {
        let mut db = open(&path);
        write(&db);
        check(&db, "as written");
        db.compact().unwrap();
        check(&db, "after compact");
        db.close().unwrap();
    }
    let db = open(&path);
    check(&db, "after the reopen");
    db.execute("CREATE INDEX n_v FOR (n:N) ON (n.v)").unwrap();
    db.execute("CREATE INDEX location_a FOR (l:Location) ON (l.a)")
        .unwrap();
    check(&db, "through the indexes");
    db.close().unwrap();
    let db = open(&path);
    check(&db, "with the indexes after a second reopen");
}

/// An equality on `n.v` looks up the property index, and finds the signed
/// value it asks for.
fn check_lookup(db: &GrafeoDB, stage: &str) {
    let query = "MATCH (n:N {v: -7}) RETURN n.v";
    let Value::String(plan) = &row(db, &format!("EXPLAIN {query}"))[0] else {
        panic!("{stage}: EXPLAIN returns its plan as text");
    };
    assert!(plan.contains("[index: v]"), "{stage}: {plan}");
    assert_eq!(row(db, query), [Value::Int64(-7)], "{stage}: {query}");
}

/// The indexes made before the compact: the compact keeps them serving the
/// signed values, and so do a reopen and a read-only open.
#[test]
fn indexes_made_before_compact_keep_serving_signed_values() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("signed_indexed.grafeo");
    {
        let mut db = open(&path);
        write(&db);
        db.execute("CREATE INDEX n_v FOR (n:N) ON (n.v)").unwrap();
        db.execute("CREATE INDEX location_a FOR (l:Location) ON (l.a)")
            .unwrap();
        check_lookup(&db, "before compact");
        db.compact().unwrap();
        check(&db, "indexed, after compact");
        check_lookup(&db, "after compact");
        db.close().unwrap();
    }
    let db = open(&path);
    check(&db, "indexed, after the reopen");
    check_lookup(&db, "after the reopen");
    db.close().unwrap();
    let db = GrafeoDB::open_read_only(&path).unwrap();
    check(&db, "indexed, read-only");
    check_lookup(&db, "read-only");
}
