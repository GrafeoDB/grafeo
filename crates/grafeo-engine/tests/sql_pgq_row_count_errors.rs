//! A bad `LIMIT $n` or `OFFSET $n` count in SQL/PGQ fails with an error that
//! names the clause the query wrote (OFFSET, LIMIT), where GQL names SKIP and
//! LIMIT; the GQLSTATUS and the error code are the same in both.
//!
//! Run with:
//! ```bash
//! cargo test -p grafeo-engine --all-features --test main sql_pgq_row_count_errors::
//! ```

#![cfg(all(feature = "sql-pgq", feature = "gql"))]

use std::collections::HashMap;

use grafeo_common::types::Value;
use grafeo_common::utils::GqlStatus;
use grafeo_common::utils::error::Error;
use grafeo_engine::GrafeoDB;

/// Alix and Gus, in Amsterdam and Berlin.
fn people() -> GrafeoDB {
    let db = GrafeoDB::new_in_memory();
    db.session()
        .execute(
            "INSERT (:Person {name: 'Alix', city: 'Amsterdam'}), \
                    (:Person {name: 'Gus', city: 'Berlin'})",
        )
        .unwrap();
    db
}

const SQL: &str = "SELECT * FROM GRAPH_TABLE (MATCH (p:Person) COLUMNS (p.name AS name))";

/// The error of the SQL/PGQ query `clause` appended to [`SQL`], with the
/// parameter `n` set to `value`.
fn sql_error(db: &GrafeoDB, clause: &str, value: Value) -> Error {
    let query = format!("{SQL} {clause} $n");
    let params = HashMap::from([("n".to_string(), value)]);
    db.session()
        .execute_sql_with_params(&query, params)
        .map(|_| ())
        .expect_err(&query)
}

/// The error of the GQL query with `clause $n`, `n` set to `value`.
fn gql_error(db: &GrafeoDB, clause: &str, value: Value) -> Error {
    let query = format!("MATCH (p:Person) RETURN p.name {clause} $n");
    let params = HashMap::from([("n".to_string(), value)]);
    db.session()
        .execute_with_params(&query, params)
        .map(|_| ())
        .expect_err(&query)
}

#[test]
fn a_negative_offset_names_offset() {
    let db = people();
    let error = sql_error(&db, "OFFSET", Value::Int64(-1));
    assert_eq!(
        error.to_string(),
        "GRAFEO-Q006: negative limit value: OFFSET must be non-negative, got -1"
    );
    assert_eq!(GqlStatus::from(&error), GqlStatus::DATA_NEGATIVE_LIMIT);
}

#[test]
fn a_negative_limit_names_limit() {
    let db = people();
    let error = sql_error(&db, "LIMIT", Value::Int64(-2));
    assert_eq!(
        error.to_string(),
        "GRAFEO-Q006: negative limit value: LIMIT must be non-negative, got -2"
    );
    assert_eq!(GqlStatus::from(&error), GqlStatus::DATA_NEGATIVE_LIMIT);
}

#[test]
fn a_fractional_or_text_count_names_its_clause() {
    let db = people();
    for (clause, value, shown) in [
        ("OFFSET", Value::Float64(1.5), "1.5"),
        ("OFFSET", Value::from("two"), r#""two""#),
        ("LIMIT", Value::Float64(0.5), "0.5"),
    ] {
        let error = sql_error(&db, clause, value);
        assert_eq!(
            error.to_string(),
            format!("GRAFEO-Q006: invalid value type: {clause} must be an integer, got {shown}"),
        );
        assert_eq!(GqlStatus::from(&error), GqlStatus::DATA_INVALID_VALUE_TYPE);
    }
}

#[test]
fn a_missing_count_parameter_names_its_clause() {
    let db = people();
    for clause in ["OFFSET", "LIMIT"] {
        let query = format!("{SQL} {clause} $n");
        let error = db
            .session()
            .execute_sql_with_params(&query, HashMap::new())
            .map(|_| ())
            .expect_err(&query);
        assert!(
            error
                .to_string()
                .ends_with(&format!("Missing parameter for {clause}: $n")),
            "{error}"
        );
    }
}

/// GQL keeps its wording, and both languages give one status and code.
#[test]
fn gql_keeps_skip_and_limit_with_the_same_status_and_code() {
    let db = people();
    for (gql_clause, sql_clause) in [("SKIP", "OFFSET"), ("LIMIT", "LIMIT")] {
        let gql = gql_error(&db, gql_clause, Value::Int64(-1));
        assert_eq!(
            gql.to_string(),
            "GRAFEO-Q006: negative limit value: SKIP/LIMIT must be non-negative, got -1"
        );
        let sql = sql_error(&db, sql_clause, Value::Int64(-1));
        assert_eq!(GqlStatus::from(&gql), GqlStatus::from(&sql));
        assert_eq!(gql.error_code(), sql.error_code());

        let gql = gql_error(&db, gql_clause, Value::Float64(1.5));
        assert_eq!(
            gql.to_string(),
            "GRAFEO-Q006: invalid value type: SKIP/LIMIT must be an integer, got 1.5"
        );
    }
}
