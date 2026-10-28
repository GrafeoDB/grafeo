//! Stored procedure rows agree with their declared or yielded columns.
//! Existing name-first projection and same-width positional results stay valid.

#![cfg(all(feature = "gql", feature = "algos"))]

use grafeo_common::types::Value;
use grafeo_common::utils::error::{Error, QueryErrorKind, Result};
use grafeo_engine::GrafeoDB;
use grafeo_engine::database::QueryResult;

fn assert_semantic_error(result: Result<QueryResult>, query: &str) {
    match result {
        Err(Error::Query(error)) => {
            assert_eq!(error.kind, QueryErrorKind::Semantic, "{query}: {error}");
        }
        other => panic!("{query}: expected a semantic error, got {other:?}"),
    }
}

#[test]
fn an_unresolved_wide_body_is_a_semantic_error() {
    let db = GrafeoDB::new_in_memory();
    db.execute(
        "CREATE PROCEDURE wide_result() RETURNS (declared INTEGER) AS { \
         RETURN 1 AS first, 2 AS second }",
    )
    .unwrap();

    // Ordinary named CALL is valid on its own. The body must not produce
    // [1, 2] under the one-column header [declared].
    let query = "CALL wide_result()";
    assert_semantic_error(db.execute(query), query);
}

#[test]
fn a_same_width_renamed_body_keeps_its_positional_result() {
    let db = GrafeoDB::new_in_memory();
    db.execute(
        "CREATE PROCEDURE renamed_result() RETURNS (declared INTEGER) AS { \
         RETURN 7 AS different_name }",
    )
    .unwrap();

    let result = db.execute("CALL renamed_result()").unwrap();
    assert_eq!(result.columns, ["declared"]);
    assert_eq!(result.rows(), &[vec![Value::Int64(7)]]);
}

#[test]
fn declared_names_select_only_their_values_in_declared_order() {
    let db = GrafeoDB::new_in_memory();
    db.execute(
        "CREATE PROCEDURE named_result() RETURNS (second INTEGER, first INTEGER) AS { \
         UNWIND [2, 1, 2] AS v \
         RETURN v AS first, 99 AS unrelated, v + 10 AS second ORDER BY first }",
    )
    .unwrap();

    let result = db.execute("CALL named_result()").unwrap();
    assert_eq!(result.columns, ["second", "first"]);
    assert_eq!(
        result.rows(),
        &[
            vec![Value::Int64(11), Value::Int64(1)],
            vec![Value::Int64(12), Value::Int64(2)],
            vec![Value::Int64(12), Value::Int64(2)],
        ]
    );
}

#[test]
fn partially_missing_declared_names_are_a_semantic_error() {
    let db = GrafeoDB::new_in_memory();
    db.execute(
        "CREATE PROCEDURE partial_result() RETURNS (present INTEGER, missing INTEGER) AS { \
         RETURN 1 AS present, 2 AS unrelated }",
    )
    .unwrap();

    let query = "CALL partial_result()";
    assert_semantic_error(db.execute(query), query);
}

#[test]
fn an_unresolved_narrow_body_is_a_semantic_error() {
    let db = GrafeoDB::new_in_memory();
    db.execute(
        "CREATE PROCEDURE narrow_result() RETURNS (first INTEGER, second INTEGER) AS { \
         RETURN 7 AS different_name }",
    )
    .unwrap();

    let query = "CALL narrow_result()";
    assert_semantic_error(db.execute(query), query);
}

#[test]
fn an_empty_body_still_checks_its_result_column_width() {
    let db = GrafeoDB::new_in_memory();
    db.execute(
        "CREATE PROCEDURE empty_wide_result() RETURNS (declared INTEGER) AS { \
         MATCH (:Absent) RETURN 1 AS first, 2 AS second }",
    )
    .unwrap();

    // No data row exists to expose the mismatch, but the body still declares
    // two output columns. Zero rows cannot make an incompatible header valid.
    let query = "CALL empty_wide_result()";
    assert_semantic_error(db.execute(query), query);
}

#[test]
fn yield_order_and_aliases_keep_exact_values_and_duplicate_rows() {
    let db = GrafeoDB::new_in_memory();
    db.execute(
        "CREATE PROCEDURE yielded_result() RETURNS (first INTEGER, second INTEGER) AS { \
         UNWIND [2, 1, 2] AS v \
         RETURN v + 100 AS second, v AS first, 999 AS unrelated ORDER BY first }",
    )
    .unwrap();

    let result = db
        .execute("CALL yielded_result() YIELD second AS rhs, first AS lhs")
        .unwrap();
    assert_eq!(result.columns, ["rhs", "lhs"]);
    assert_eq!(
        result.rows(),
        &[
            vec![Value::Int64(101), Value::Int64(1)],
            vec![Value::Int64(102), Value::Int64(2)],
            vec![Value::Int64(102), Value::Int64(2)],
        ]
    );
}

#[test]
fn a_result_width_error_leaves_none_of_the_procedure_bodys_writes() {
    let db = GrafeoDB::new_in_memory();
    db.execute("INSERT (:Kept {value: 5})").unwrap();
    let session = db.session();
    session
        .execute(
            "CREATE PROCEDURE writing_wide_result() RETURNS (declared INTEGER) AS { \
             INSERT (:WidthWrite {value: 99}) RETURN 1 AS first, 2 AS second }",
        )
        .unwrap();

    let query = "CALL writing_wide_result()";
    assert_semantic_error(session.execute(query), query);
    assert!(
        !session.in_transaction(),
        "the failed call leaves no transaction"
    );

    let fresh = db.session();
    let written = fresh
        .execute("MATCH (n:WidthWrite) RETURN n.value AS value")
        .unwrap();
    assert_eq!(written.columns, ["value"]);
    assert_eq!(written.rows(), &[] as &[Vec<Value>]);
    let kept = fresh
        .execute("MATCH (n:Kept) RETURN n.value AS value")
        .unwrap();
    assert_eq!(kept.columns, ["value"]);
    assert_eq!(kept.rows(), &[vec![Value::Int64(5)]]);
}

#[test]
fn a_valid_empty_body_keeps_its_declared_columns() {
    let db = GrafeoDB::new_in_memory();
    db.execute(
        "CREATE PROCEDURE empty_valid_result() RETURNS (declared INTEGER) AS { \
         MATCH (:Absent) RETURN 1 AS declared }",
    )
    .unwrap();

    let result = db.execute("CALL empty_valid_result()").unwrap();
    assert_eq!(result.columns, ["declared"]);
    assert_eq!(result.rows(), &[] as &[Vec<Value>]);
}
