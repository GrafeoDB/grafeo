//! Calls of stored procedures, as openCypher's CALL has them (TCK 2024.3 Call1
//! to Call6):
//!
//! - a procedure is called by its whole name: `CALL test.doNothing()` calls the
//!   procedure `test.doNothing`; the call looked it up by the last part of the
//!   name, so it found `doNothing` and never `test.doNothing`;
//! - an argument of another type than its parameter's is a semantic error
//!   (42); an integer binds to a float parameter as the float;
//! - a procedure that returns nothing, called in a query, keeps each row once;
//! - `YIELD` may not give two outputs one name, and `YIELD *` gives them all.
//!
//! ```bash
//! cargo test -p grafeo-engine --all-features --test main stored_procedure_calls::
//! ```

#![cfg(all(feature = "gql", feature = "cypher"))]

use grafeo_common::types::Value;
use grafeo_engine::GrafeoDB;

fn database() -> GrafeoDB {
    let db = GrafeoDB::new_in_memory();
    db.execute("CREATE PROCEDURE `test.one`() RETURNS (x INT64) AS { RETURN 1 AS x }")
        .unwrap();
    db.execute("CREATE PROCEDURE `one`() RETURNS (x INT64) AS { RETURN 2 AS x }")
        .unwrap();
    db
}

fn single(db: &GrafeoDB, language: &str, query: &str) -> Value {
    let result = match language {
        "gql" => db.execute(query),
        _ => db.execute_cypher(query),
    }
    .unwrap_or_else(|error| panic!("{language}: {query}: {error}"));
    let rows = result.rows();
    assert_eq!(rows.len(), 1, "{language}: {query}: {rows:?}");
    rows[0][0].clone()
}

/// The dotted name calls the procedure of that name, the plain name the
/// procedure of the plain name, standalone and in a query.
#[test]
fn a_dotted_name_calls_the_procedure_of_that_name() {
    let db = database();
    for language in ["gql", "cypher"] {
        for (query, expected) in [
            ("CALL test.one() YIELD x RETURN x", 1),
            ("CALL one() YIELD x RETURN x", 2),
            (
                "MATCH (n) WITH count(n) AS c CALL test.one() YIELD x RETURN x",
                1,
            ),
        ] {
            assert_eq!(
                single(&db, language, query),
                Value::Int64(expected),
                "{language}: {query}"
            );
        }
    }
    assert_eq!(single(&db, "cypher", "CALL test.one()"), Value::Int64(1));
}

/// A name with a part in front calls no procedure of the last part's name.
#[test]
fn a_dotted_name_does_not_call_the_procedure_of_its_last_part() {
    let db = GrafeoDB::new_in_memory();
    db.execute("CREATE PROCEDURE `one`() RETURNS (x INT64) AS { RETURN 2 AS x }")
        .unwrap();
    for language in ["gql", "cypher"] {
        let query = "CALL other.one() YIELD x RETURN x";
        let result = match language {
            "gql" => db.execute(query),
            _ => db.execute_cypher(query),
        };
        assert!(result.is_err(), "{language}: {query}: {result:?}");
    }
}

/// A built-in procedure is called as before.
#[test]
fn a_built_in_procedure_is_called_as_before() {
    let db = database();
    let result = db.execute_cypher("CALL grafeo.procedures() YIELD name RETURN count(name) AS c");
    assert!(result.is_ok(), "{result:?}");
}

fn error_status(db: &GrafeoDB, language: &str, query: &str) -> String {
    let result = match language {
        "gql" => db.execute(query),
        _ => db.execute_cypher(query),
    };
    let error = result.expect_err(&format!("{language}: {query} should fail"));
    grafeo_common::utils::gqlstatus::GqlStatus::from(&error)
        .as_str()
        .to_string()
}

/// An argument of another type than the parameter's is refused before the
/// procedure runs; a value the type stores is taken as a property of that type
/// stores it: null, an integer for a float when the float is exact, an integer
/// for a decimal, a zoned datetime for a `DATETIME` as its instant.
#[test]
fn an_argument_must_have_its_parameters_type() {
    let db = GrafeoDB::new_in_memory();
    db.execute(
        "CREATE PROCEDURE `test.int`(in INT64) RETURNS (out INT64) AS { RETURN $in AS out }",
    )
    .unwrap();
    db.execute(
        "CREATE PROCEDURE `test.float`(in FLOAT64) RETURNS (out FLOAT64) AS { RETURN $in AS out }",
    )
    .unwrap();
    db.execute(
        "CREATE PROCEDURE `test.decimal`(in DECIMAL) RETURNS (out DECIMAL) AS { RETURN $in AS out }",
    )
    .unwrap();
    db.execute(
        "CREATE PROCEDURE `test.instant`(in DATETIME) RETURNS (out DATETIME) AS { RETURN $in AS out }",
    )
    .unwrap();
    for language in ["gql", "cypher"] {
        for query in [
            "CALL test.int(true) YIELD out RETURN out",
            "CALL test.int('one') YIELD out RETURN out",
            "CALL test.float('one') YIELD out RETURN out",
            // 2^53 + 1 has no exact float.
            "CALL test.float(9007199254740993) YIELD out RETURN out",
            "CALL test.decimal('7') YIELD out RETURN out",
        ] {
            let status = error_status(&db, language, query);
            assert!(status.starts_with("42"), "{language}: {query}: {status}");
        }
        for (query, expected) in [
            ("CALL test.int(null) YIELD out RETURN out", Value::Null),
            ("CALL test.int(7) YIELD out RETURN out", Value::Int64(7)),
            (
                "CALL test.float(7) YIELD out RETURN out",
                Value::Float64(7.0),
            ),
            (
                "CALL test.float(7.5) YIELD out RETURN out",
                Value::Float64(7.5),
            ),
            (
                "CALL test.decimal(7) YIELD out RETURN out",
                Value::Decimal(grafeo_common::types::Decimal::from_i64(7)),
            ),
        ] {
            assert_eq!(
                single(&db, language, query),
                expected,
                "{language}: {query}"
            );
        }
    }
    // 2024-01-01T10:00+01:00 is 09:00 UTC.
    assert_eq!(
        single(
            &db,
            "cypher",
            "CALL test.instant(datetime('2024-01-01T10:00:00+01:00')) YIELD out RETURN out"
        ),
        Value::Timestamp(grafeo_common::types::Timestamp::from_micros(
            1_704_099_600_000_000
        ))
    );
}

/// A procedure that returns nothing, called in a query, keeps each row once
/// (TCK Call1 [4]).
#[test]
fn a_void_procedure_in_a_query_keeps_its_rows() {
    let db = GrafeoDB::new_in_memory();
    db.execute("CREATE PROCEDURE `test.doNothing`() RETURNS () AS { UNWIND [] AS r FINISH }")
        .unwrap();
    db.execute_cypher("CREATE (:A {name: 'a'}), (:B {name: 'b'}), (:C {name: 'c'})")
        .unwrap();
    let result = db
        .execute_cypher("MATCH (n) CALL test.doNothing() RETURN n.name AS name ORDER BY name")
        .unwrap();
    assert_eq!(
        result.rows(),
        &[
            vec![Value::from("a")],
            vec![Value::from("b")],
            vec![Value::from("c")],
        ]
    );
}

/// `YIELD` that gives two outputs one name, or one output the name of a
/// variable already bound, is a semantic error (TCK Call5 [5] [6]); `YIELD *`
/// gives every output (TCK Call5 [8]).
#[test]
fn yield_names_are_distinct_and_star_gives_all() {
    let db = database();
    db.execute(
        "CREATE PROCEDURE `test.two`() RETURNS (a INT64, b INT64) AS { RETURN 1 AS a, 2 AS b }",
    )
    .unwrap();
    for query in [
        "CALL test.two() YIELD a, b AS a RETURN a",
        "CALL test.two() YIELD a AS c, b AS c RETURN c",
        "WITH 1 AS a CALL test.two() YIELD a RETURN a",
    ] {
        let status = error_status(&db, "cypher", query);
        assert!(status.starts_with("42"), "{query}: {status}");
    }
    for query in [
        "CALL test.two() YIELD a, b AS a",
        "CALL test.two() YIELD a AS c, b AS c",
    ] {
        let status = error_status(&db, "gql", query);
        assert!(status.starts_with("42"), "gql: {query}: {status}");
    }
    let result = db.execute_cypher("CALL test.two() YIELD *").unwrap();
    assert_eq!(result.columns, ["a", "b"]);
    assert_eq!(result.rows(), &[vec![Value::Int64(1), Value::Int64(2)]]);
}
