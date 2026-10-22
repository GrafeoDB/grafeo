//! A JSON integer past 64 bits in `LOAD DATA ... FORMAT JSONL` is an exact
//! DECIMAL up to 38 digits (it used to be a float that lost its last digits):
//! an integer that fits 64 bits stays INT64, any other number is a FLOAT64
//! with its value, and an integer past 38 digits is an error naming the line
//! and the value (GQLSTATUS 22003). The bindings and the CLI follow the same
//! rule.
//!
//! ```bash
//! cargo test -p grafeo-engine --all-features --test jsonl_wide_integers
//! ```

#![cfg(feature = "jsonl-import")]

use grafeo_common::types::{Decimal, Value};
use grafeo_common::utils::GqlStatus;
use grafeo_engine::GrafeoDB;

fn d(text: &str) -> Value {
    Value::Decimal(Decimal::parse(text).unwrap())
}

/// A JSONL file of `lines` in a fresh directory, its path as LOAD DATA takes it.
fn jsonl(lines: &[String]) -> (tempfile::TempDir, String) {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("rows.jsonl");
    std::fs::write(&path, lines.join("\n")).unwrap();
    let path = path.to_string_lossy().replace('\\', "/");
    (dir, path)
}

#[test]
fn json_integers_follow_the_rule_and_floats_keep_their_value() {
    let largest = "9".repeat(38);
    let values = [
        ("5", Value::Int64(5), "INT64"),
        ("9223372036854775807", Value::Int64(i64::MAX), "INT64"),
        ("-9223372036854775808", Value::Int64(i64::MIN), "INT64"),
        ("9223372036854775808", d("9223372036854775808"), "DECIMAL"),
        ("18446744073709551616", d("18446744073709551616"), "DECIMAL"),
        ("-9223372036854775809", d("-9223372036854775809"), "DECIMAL"),
        (largest.as_str(), d(&largest), "DECIMAL"),
        ("1.50", Value::Float64(1.5), "FLOAT64"),
        ("1e300", Value::Float64(1e300), "FLOAT64"),
        ("9.2e18", Value::Float64(9.2e18), "FLOAT64"),
    ];
    let lines: Vec<String> = values
        .iter()
        .enumerate()
        .map(|(i, (text, _, _))| format!(r#"{{"i": {i}, "v": {text}, "l": [{text}]}}"#))
        .collect();
    let (_dir, path) = jsonl(&lines);
    let db = GrafeoDB::new_in_memory();
    let result = db
        .session()
        .execute(&format!(
            "LOAD DATA FROM '{path}' FORMAT JSONL AS row \
             RETURN row.v AS v, row.l[0] AS first, row.v IS TYPED INT64 AS int, \
             row.v IS TYPED DECIMAL AS decimal, row.v IS TYPED FLOAT64 AS float \
             ORDER BY row.i"
        ))
        .unwrap();
    assert_eq!(result.rows().len(), values.len());
    for (row, (text, value, kind)) in result.rows().iter().zip(&values) {
        let flags = [kind == &"INT64", kind == &"DECIMAL", kind == &"FLOAT64"].map(Value::Bool);
        assert_eq!(row[0], *value, "{text}");
        assert_eq!(row[1], *value, "{text} in a list");
        assert_eq!(row[2..5], flags, "{text} is {kind}");
    }
}

#[test]
fn arithmetic_over_int64_and_decimal_rows_stays_exact() {
    let lines = ["5", "18446744073709551616", "-9223372036854775808"]
        .map(|v| format!(r#"{{"v": {v}}}"#))
        .to_vec();
    let (_dir, path) = jsonl(&lines);
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    session
        .execute(&format!(
            "LOAD DATA FROM '{path}' FORMAT JSONL AS row INSERT (:N {{v: row.v}})"
        ))
        .unwrap();
    let next = session
        .execute("MATCH (n:N) RETURN n.v + 1 AS next ORDER BY next")
        .unwrap();
    let next: Vec<Value> = next.rows().iter().map(|row| row[0].clone()).collect();
    assert_eq!(
        next,
        [
            Value::Int64(i64::MIN + 1),
            Value::Int64(6),
            d("18446744073709551617")
        ]
    );
    let total = session
        .execute("MATCH (n:N) RETURN sum(n.v) AS total")
        .unwrap();
    assert_eq!(total.rows()[0][0], d("9223372036854775813"));
}

#[test]
fn an_integer_past_38_digits_is_an_error_naming_the_line() {
    let too_wide = format!("1{}", "0".repeat(38));
    let lines = vec![
        r#"{"v": 1}"#.to_string(),
        String::new(),
        format!(r#"{{"v": [2, {too_wide}]}}"#),
    ];
    let (_dir, path) = jsonl(&lines);
    let db = GrafeoDB::new_in_memory();
    let error = db
        .session()
        .execute(&format!(
            "LOAD DATA FROM '{path}' FORMAT JSONL AS row RETURN row.v AS v"
        ))
        .unwrap_err();
    let message = error.to_string();
    assert!(
        message.contains("line 3") && message.contains(&too_wide) && message.contains("38 digits"),
        "{message}"
    );
    assert_eq!(
        GqlStatus::from(&error),
        GqlStatus::DATA_NUMERIC_OUT_OF_RANGE
    );
}
