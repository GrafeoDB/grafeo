//! After `compact()`, a property that a node or edge does not have stays
//! missing (#542): it reads as null, `keys()` leaves it out, `IS NULL`
//! matches it, and no filter matches it through the column's empty value
//! (`''`, `0`, `0.0`, `false`).

#![cfg(all(feature = "lpg", feature = "gql"))]

use grafeo_common::types::Value;
use grafeo_engine::GrafeoDB;

/// Two `:P` nodes and two `:R` edges of the same label and type: the first
/// of each has a value in every column type, the second only `n`.
fn populate(db: &GrafeoDB) {
    db.execute(
        "INSERT (:P {n: 1, s: 'Amsterdam', u: 19, i: -3, f: 1.5, b: true, v: vector([3.0, 19.0])}), \
                (:P {n: 2})",
    )
    .unwrap();
    db.execute(
        "MATCH (a:P {n: 1}), (b:P {n: 2}) \
         INSERT (a)-[:R {n: 1, s: 'Berlin', u: 19, i: -3, f: 2.5, b: true}]->(b), \
                (b)-[:R {n: 2}]->(a)",
    )
    .unwrap();
}

fn rows(db: &GrafeoDB, query: &str) -> Vec<Vec<Value>> {
    db.execute(query).unwrap().rows().to_vec()
}

fn keys(names: &[&str]) -> Value {
    Value::List(
        names
            .iter()
            .map(|n| Value::from(*n))
            .collect::<Vec<_>>()
            .into(),
    )
}

/// Checks every way a missing property could show up as a value.
fn assert_missing_stays_missing(db: &GrafeoDB, when: &str) {
    assert_eq!(
        rows(
            db,
            "MATCH (p:P {n: 2}) RETURN p.s, p.u, p.i, p.f, p.b, p.v, keys(p)"
        ),
        [vec![
            Value::Null,
            Value::Null,
            Value::Null,
            Value::Null,
            Value::Null,
            Value::Null,
            keys(&["n"])
        ]],
        "{when}: node values"
    );
    assert_eq!(
        rows(
            db,
            "MATCH ()-[r:R {n: 2}]->() RETURN r.s, r.u, r.i, r.f, r.b, keys(r)"
        ),
        [vec![
            Value::Null,
            Value::Null,
            Value::Null,
            Value::Null,
            Value::Null,
            keys(&["n"])
        ]],
        "{when}: edge values"
    );
    for column in ["s", "u", "i", "f", "b", "v"] {
        assert_eq!(
            rows(
                db,
                &format!("MATCH (p:P) WHERE p.{column} IS NULL RETURN p.n")
            ),
            [vec![Value::Int64(2)]],
            "{when}: p.{column} IS NULL"
        );
    }
    for (filter, expected) in [
        ("p.s = ''", 0),
        ("p.u = 0", 0),
        ("p.u < 1", 0),
        ("p.i = 0", 0),
        ("p.i < 0", 1),
        ("p.f = 0.0", 0),
        ("p.b = false", 0),
    ] {
        assert_eq!(
            rows(db, &format!("MATCH (p:P) WHERE {filter} RETURN count(p)")),
            [vec![Value::Int64(expected)]],
            "{when}: {filter}"
        );
    }
    for (filter, expected) in [("r.u = 0", 0), ("r.b = false", 0), ("r.s = ''", 0)] {
        assert_eq!(
            rows(
                db,
                &format!("MATCH ()-[r:R]->() WHERE {filter} RETURN count(r)")
            ),
            [vec![Value::Int64(expected)]],
            "{when}: {filter}"
        );
    }
    // The values that are there are untouched.
    assert_eq!(
        rows(db, "MATCH (p:P {n: 1}) RETURN p.s, p.u, p.i, p.b"),
        [vec![
            Value::from("Amsterdam"),
            Value::Int64(19),
            Value::Int64(-3),
            Value::Bool(true)
        ]],
        "{when}: present values"
    );
}

#[test]
fn missing_properties_stay_missing_after_compact() {
    let mut db = GrafeoDB::new_in_memory();
    populate(&db);
    db.compact().unwrap();
    assert_missing_stays_missing(&db, "after compact()");
    db.compact().unwrap();
    assert_missing_stays_missing(&db, "after a second compact()");
}

#[cfg(feature = "grafeo-file")]
#[test]
fn missing_properties_stay_missing_after_a_reopen() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("compact.grafeo");
    {
        let mut db = GrafeoDB::open(&path).unwrap();
        populate(&db);
        db.compact().unwrap();
        db.close().unwrap();
    }
    let db = GrafeoDB::open(&path).unwrap();
    assert_missing_stays_missing(&db, "after a reopen");
}
