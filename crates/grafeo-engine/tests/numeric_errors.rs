//! Numeric errors in queries (#639): an integer or decimal result that does
//! not fit fails the statement with GQLSTATUS 22003 (numeric value out of
//! range), an integer or decimal division by zero with 22012 (division by
//! zero), both with code GRAFEO-Q006, in every clause and in GQL, Cypher and
//! SQL/PGQ. Float arithmetic keeps its IEEE results, and a null operand gives
//! null without an error.
//!
//! ```bash
//! cargo test -p grafeo-engine --all-features --test numeric_errors
//! ```

#![cfg(all(feature = "lpg", feature = "gql", feature = "cypher"))]

use std::collections::HashMap;

use grafeo_common::types::{Decimal, Value};
use grafeo_common::utils::{Error, GqlStatus};
use grafeo_engine::GrafeoDB;

const LANGUAGES: [&str; 2] = ["gql", "cypher"];

fn d(text: &str) -> Value {
    Value::Decimal(Decimal::parse(text).unwrap())
}

/// The parameters every query of this file may use.
fn params() -> HashMap<String, Value> {
    [
        ("max", Value::Int64(i64::MAX)),
        ("min", Value::Int64(i64::MIN)),
        ("one", d("1.5")),
        ("zero", d("0")),
        ("half", d("0.5")),
        // 38 digits, the most a decimal holds.
        ("big", d("90000000000000000000000000000000000000")),
        ("wide", d("10000000000000000000000000")),
    ]
    .into_iter()
    .map(|(name, value)| (name.to_string(), value))
    .collect()
}

fn run(db: &GrafeoDB, language: &str, query: &str) -> Result<Vec<Vec<Value>>, Error> {
    db.execute_language(query, language, Some(params()))
        .map(|result| result.rows().to_vec())
}

/// The error `query` fails with.
fn error(db: &GrafeoDB, language: &str, query: &str) -> Error {
    match run(db, language, query) {
        Ok(rows) => panic!("{language} `{query}` should fail, gave {rows:?}"),
        Err(error) => error,
    }
}

/// Checks that `query` fails with `status` and a message naming `what`.
fn assert_fails(db: &GrafeoDB, language: &str, query: &str, status: GqlStatus, what: &str) {
    let error = error(db, language, query);
    assert_eq!(
        GqlStatus::from(&error),
        status,
        "{language} `{query}`: {error}"
    );
    assert_eq!(
        error.error_code().as_str(),
        "GRAFEO-Q006",
        "{language} `{query}`: {error}"
    );
    let message = error.to_string();
    assert!(
        message.contains(what),
        "{language} `{query}`: `{message}` should name `{what}`"
    );
}

fn out_of_range(db: &GrafeoDB, language: &str, query: &str, what: &str) {
    assert_fails(
        db,
        language,
        query,
        GqlStatus::DATA_NUMERIC_OUT_OF_RANGE,
        what,
    );
}

fn by_zero(db: &GrafeoDB, language: &str, query: &str) {
    assert_fails(
        db,
        language,
        query,
        GqlStatus::DATA_DIVISION_BY_ZERO,
        "division by zero",
    );
}

/// The one value of `query`.
fn value(db: &GrafeoDB, language: &str, query: &str) -> Value {
    let rows = run(db, language, query).unwrap_or_else(|e| panic!("{language} `{query}`: {e}"));
    assert_eq!(rows.len(), 1, "{language} `{query}`: {rows:?}");
    assert_eq!(rows[0].len(), 1, "{language} `{query}`: {rows:?}");
    rows[0][0].clone()
}

/// A database with `:N` nodes whose `v` is 0 and 2.
fn numbers() -> GrafeoDB {
    let db = GrafeoDB::new_in_memory();
    db.execute("INSERT (:N {v: 0}), (:N {v: 2})").unwrap();
    db
}

fn count(db: &GrafeoDB, query: &str) -> i64 {
    match db.execute(query).unwrap().rows()[0][0] {
        Value::Int64(n) => n,
        ref other => panic!("`{query}` counted {other:?}"),
    }
}

// ISO: GV17
#[test]
fn integer_overflow_is_out_of_range() {
    let db = GrafeoDB::new_in_memory();
    for language in LANGUAGES {
        for (query, what) in [
            ("RETURN $max + 1 AS x", "9223372036854775807 + 1"),
            ("RETURN $min - 1 AS x", "-9223372036854775808 - 1"),
            ("RETURN $max * 2 AS x", "9223372036854775807 * 2"),
            ("RETURN -$min AS x", "-(-9223372036854775808)"),
            ("RETURN $min / -1 AS x", "-9223372036854775808 / -1"),
            // Each step is checked, not only the result.
            (
                "RETURN 100 * 1000000000 * 100000000 AS x",
                "100000000000 * 100000000",
            ),
        ] {
            out_of_range(&db, language, query, what);
        }
    }
}

// ISO: GV17
#[test]
fn decimal_overflow_is_out_of_range() {
    let db = GrafeoDB::new_in_memory();
    for language in LANGUAGES {
        for query in [
            "RETURN $big + $big AS x",
            "RETURN $big - -$big AS x",
            "RETURN $wide * $wide AS x",
            "RETURN $big * 2 AS x",
            "RETURN $big / $half AS x",
        ] {
            out_of_range(&db, language, query, "numeric value out of range");
        }
    }
}

#[test]
fn integer_and_decimal_division_by_zero_fails() {
    let db = GrafeoDB::new_in_memory();
    for language in LANGUAGES {
        for query in [
            "RETURN 1 / 0 AS x",
            "RETURN 1 % 0 AS x",
            "RETURN $one / 0 AS x",
            "RETURN $one % 0 AS x",
            "RETURN 1 / $zero AS x",
            "RETURN 1 % $zero AS x",
            "RETURN $one / $zero AS x",
        ] {
            by_zero(&db, language, query);
        }
    }
}

#[test]
fn results_that_fit_do_not_fail() {
    let db = GrafeoDB::new_in_memory();
    for language in LANGUAGES {
        for (query, expected) in [
            ("RETURN $max + 0 AS x", Value::Int64(i64::MAX)),
            ("RETURN $min + 0 AS x", Value::Int64(i64::MIN)),
            ("RETURN -$max AS x", Value::Int64(-i64::MAX)),
            // The remainder of the one quotient that does not fit is 0.
            ("RETURN $min % -1 AS x", Value::Int64(0)),
            (
                "RETURN $big + 0 AS x",
                d("90000000000000000000000000000000000000"),
            ),
            // A null operand gives null, also by zero.
            ("RETURN null / 0 AS x", Value::Null),
            ("RETURN null + $max AS x", Value::Null),
        ] {
            assert_eq!(
                value(&db, language, query),
                expected,
                "{language} `{query}`"
            );
        }
    }
}

#[test]
fn float_arithmetic_keeps_its_ieee_results() {
    let db = GrafeoDB::new_in_memory();
    for language in LANGUAGES {
        assert_eq!(
            value(&db, language, "RETURN toFloat(1) / 0 AS x"),
            Value::Float64(f64::INFINITY),
            "{language}"
        );
        assert_eq!(
            value(&db, language, "RETURN toFloat($max) * 4 AS x"),
            Value::Float64(i64::MAX as f64 * 4.0),
            "{language}"
        );
        match value(&db, language, "RETURN toFloat(0) / 0 AS x") {
            Value::Float64(f) => assert!(f.is_nan(), "{language}: {f}"),
            other => panic!("{language}: {other:?}"),
        }
    }
}

/// An operand that is not evaluated cannot fail: the branches `CASE` does not
/// take, and the right side of an `AND` whose left side is false or an `OR`
/// whose left side is true.
#[test]
fn an_operand_that_is_not_evaluated_does_not_fail() {
    let db = GrafeoDB::new_in_memory();
    for language in LANGUAGES {
        for (query, expected) in [
            (
                "RETURN CASE WHEN false THEN 1 / 0 ELSE 1 END AS x",
                Value::Int64(1),
            ),
            (
                "RETURN CASE 1 WHEN 2 THEN $max + 1 ELSE 3 END AS x",
                Value::Int64(3),
            ),
            ("RETURN false AND 1 / 0 = 1 AS x", Value::Bool(false)),
            ("RETURN true OR 1 / 0 = 1 AS x", Value::Bool(true)),
            ("RETURN coalesce(1, 1 / 0) AS x", Value::Int64(1)),
        ] {
            assert_eq!(
                value(&db, language, query),
                expected,
                "{language} `{query}`"
            );
        }
    }
}

#[test]
fn every_clause_fails_on_a_division_by_zero() {
    let db = numbers();
    for language in LANGUAGES {
        for query in [
            "MATCH (n:N) WHERE 10 / n.v > 1 RETURN n.v AS x",
            "MATCH (n:N) RETURN 10 / n.v AS x",
            "MATCH (n:N) RETURN n.v AS x ORDER BY 10 / n.v",
            "MATCH (n:N) RETURN sum(10 / n.v) AS x",
            "MATCH (n:N) WITH 10 / n.v AS q RETURN q AS x",
            "MATCH (n:N) RETURN count(n) AS c, max(n.v) % min(n.v) AS x",
            "UNWIND [1, 0] AS i RETURN 10 / i AS x",
            "UNWIND [1, 2] AS i RETURN [j IN [1, 0] | i / j] AS x",
            "MATCH (n:N) RETURN 10 / n.v AS x LIMIT 5",
        ] {
            by_zero(&db, language, query);
        }
    }
}

/// A guard before the division in the same `AND` keeps the rows it guards
/// away from the division.
#[test]
fn a_guard_before_a_division_keeps_its_rows_out() {
    let db = numbers();
    for language in LANGUAGES {
        assert_eq!(
            value(
                &db,
                language,
                "MATCH (n:N) WHERE n.v <> 0 AND 10 / n.v > 1 RETURN n.v AS x"
            ),
            Value::Int64(2),
            "{language}"
        );
        assert_eq!(
            value(
                &db,
                language,
                "MATCH (n:N) WHERE n.v = 0 OR 10 / n.v > 1 RETURN count(n) AS x"
            ),
            Value::Int64(2),
            "{language}"
        );
    }
}

/// A write that fails on its data changes nothing: the rows written before
/// the failing row are rolled back with the statement.
#[test]
fn a_write_that_fails_changes_nothing() {
    for language in LANGUAGES {
        let db = numbers();
        let writes: &[&str] = if language == "gql" {
            &[
                "MATCH (n:N) SET n.w = 10 / n.v",
                "MATCH (n:N) SET n.w = n.v * $max",
                "INSERT (:M {v: 1 / 0})",
                "MATCH (n:N) INSERT (:M {v: 10 / n.v})",
                "MERGE (:M {v: 1 % 0})",
            ]
        } else {
            &[
                "MATCH (n:N) SET n.w = 10 / n.v",
                "MATCH (n:N) SET n.w = n.v * $max",
                "CREATE (:M {v: 1 / 0})",
                "MATCH (n:N) CREATE (:M {v: 10 / n.v})",
                "MERGE (:M {v: 1 % 0})",
            ]
        };
        for write in writes {
            let error = error(&db, language, write);
            let status = GqlStatus::from(&error);
            assert!(
                status == GqlStatus::DATA_DIVISION_BY_ZERO
                    || status == GqlStatus::DATA_NUMERIC_OUT_OF_RANGE,
                "{language} `{write}`: {error}"
            );
            assert_eq!(
                count(&db, "MATCH (n:N) WHERE n.w IS NOT NULL RETURN count(n)"),
                0,
                "{language} `{write}` set a property"
            );
            assert_eq!(
                count(&db, "MATCH (m:M) RETURN count(m)"),
                0,
                "{language} `{write}` inserted a node"
            );
        }
    }
}

/// A failing statement in an explicit transaction fails that statement; the
/// transaction's earlier writes are still there to commit.
#[test]
fn a_numeric_error_in_a_transaction_fails_the_statement() {
    let db = numbers();
    let mut session = db.session();
    session.begin_transaction().unwrap();
    session.execute("INSERT (:M {v: 1})").unwrap();
    let error = session.execute("RETURN 1 / 0 AS x").unwrap_err();
    assert_eq!(
        GqlStatus::from(&error),
        GqlStatus::DATA_DIVISION_BY_ZERO,
        "{error}"
    );
    session.commit().unwrap();
    assert_eq!(count(&db, "MATCH (m:M) RETURN count(m)"), 1);
}

#[cfg(feature = "sql-pgq")]
#[test]
fn sql_pgq_fails_on_numeric_errors() {
    let db = numbers();
    let query = |filter: &str, column: &str| {
        format!("SELECT * FROM GRAPH_TABLE (MATCH (n:N) {filter} COLUMNS ({column} AS x))")
    };
    by_zero(&db, "sql", &query("", "10 / n.v"));
    by_zero(&db, "sql", &query("WHERE 10 / n.v > 1", "n.v"));
    by_zero(&db, "sql", &query("", "n.v % 0"));
    out_of_range(
        &db,
        "sql",
        &query("WHERE n.v = 2", "n.v * 9223372036854775807"),
        "2 * 9223372036854775807",
    );
    let rows = run(&db, "sql", &query("WHERE n.v <> 0 AND 10 / n.v > 1", "n.v")).unwrap();
    assert_eq!(rows, vec![vec![Value::Int64(2)]]);
}

/// SPARQL keeps its own rule: an expression error leaves the variable
/// unbound (SPARQL 1.1, section 17.3) instead of failing the query.
#[cfg(feature = "sparql")]
#[test]
fn sparql_leaves_an_error_unbound() {
    let db = GrafeoDB::new_in_memory();
    let rows = db
        .execute_sparql("SELECT ?x WHERE { BIND(1 / 0 AS ?x) }")
        .unwrap();
    assert_eq!(rows.rows().len(), 1);
    assert_eq!(rows.rows()[0][0], Value::Null);
}
