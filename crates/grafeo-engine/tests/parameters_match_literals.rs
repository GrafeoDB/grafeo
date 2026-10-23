//! A statement written with a parameter answers as the statement with the
//! parameter's value written as a literal: the same rows, or the same error,
//! in GQL and in Cypher, value after value, through the plan prepared on the
//! first call (a differential, from grafeo-c6's verification of prepared
//! plans).
//!
//! A value is written as the literal of its own type: in GQL a float takes
//! an exponent (`0.0e0`), since `0.0` is an exact numeric (a decimal); a
//! decimal is written with its point, in GQL only (Cypher has no decimal
//! literal).
//!
//! cargo test -p grafeo-engine --all-features --test parameters_match_literals

#![cfg(all(feature = "gql", feature = "cypher"))]

use std::collections::HashMap;

use grafeo_common::types::{Decimal, Value};
use grafeo_engine::GrafeoDB;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Language {
    Gql,
    Cypher,
}

/// `value` as a literal of `language`, or `None` when the language has no
/// literal of its type.
fn literal(value: &Value, language: Language) -> Option<String> {
    Some(match value {
        Value::Null => "NULL".into(),
        Value::Int64(n) => n.to_string(),
        Value::Float64(f) => match language {
            Language::Gql => format!("{f:?}e0"),
            Language::Cypher => format!("{f:?}"),
        },
        Value::Decimal(d) => match language {
            Language::Gql => d.to_string(),
            Language::Cypher => return None,
        },
        Value::String(s) => format!("'{s}'"),
        Value::Bool(b) => b.to_string(),
        Value::List(items) => format!(
            "[{}]",
            items
                .iter()
                .map(|item| literal(item, language))
                .collect::<Option<Vec<_>>>()?
                .join(", ")
        ),
        other => panic!("no literal for {other:?}"),
    })
}

/// A call's rows, or its error.
fn outcome(
    result: grafeo_common::utils::error::Result<grafeo_engine::database::QueryResult>,
) -> String {
    match result {
        Ok(result) => format!("{:?}", result.rows()),
        Err(error) => format!("ERROR {error}"),
    }
}

fn decimal(text: &str) -> Value {
    Value::Decimal(Decimal::parse(text).expect("a decimal"))
}

fn list(items: &[i64]) -> Value {
    Value::List(
        items
            .iter()
            .copied()
            .map(Value::Int64)
            .collect::<Vec<_>>()
            .into(),
    )
}

/// The statements, each with the values of `$x` it runs with.
fn cases() -> Vec<(&'static str, Vec<Value>)> {
    vec![
        (
            "RETURN 1 / $x AS r",
            vec![
                Value::Int64(2),
                Value::Int64(0),
                Value::Float64(0.0),
                Value::Float64(4.0),
                decimal("0.0"),
                decimal("2.5"),
                Value::Null,
            ],
        ),
        (
            "RETURN $x + 1 AS r",
            vec![
                Value::Int64(1),
                Value::Int64(i64::MAX),
                Value::Float64(1.5),
                decimal("1.25"),
                Value::from("a"),
                Value::Null,
            ],
        ),
        (
            "MATCH (n:P) RETURN n.v AS v ORDER BY v LIMIT $x",
            vec![
                Value::Int64(2),
                Value::Int64(0),
                Value::Int64(-1),
                Value::from("a"),
                Value::Float64(1.5),
                decimal("1.5"),
                Value::Bool(true),
            ],
        ),
        (
            "MATCH (n:P) RETURN n.v AS v ORDER BY v SKIP $x",
            vec![
                Value::Int64(4),
                Value::Int64(0),
                Value::Int64(-1),
                Value::Float64(2.0),
                Value::from("b"),
            ],
        ),
        (
            "MATCH (n:P) RETURN n.v AS v ORDER BY v DESC LIMIT $x",
            vec![Value::Int64(1), Value::Int64(3)],
        ),
        (
            "MATCH (n:P) WHERE n.v > $x RETURN count(n) AS c",
            vec![
                Value::Int64(3),
                Value::Float64(2.5),
                decimal("2.5"),
                Value::from("3"),
                Value::Null,
            ],
        ),
        (
            "MATCH (n:P {v: $x}) RETURN n.v AS v",
            vec![
                Value::Int64(2),
                Value::Float64(2.0),
                Value::from("2"),
                Value::Null,
            ],
        ),
        (
            "MATCH (n:P) WHERE n.v IN $x RETURN count(n) AS c",
            vec![list(&[1, 9]), list(&[]), Value::Int64(1)],
        ),
        (
            "UNWIND $x AS i RETURN sum(i) AS s",
            vec![list(&[1, 2]), list(&[]), Value::Int64(3)],
        ),
        (
            "MATCH (n:P) WHERE n.s STARTS WITH $x RETURN count(n) AS c",
            vec![Value::from("1"), Value::from(""), Value::Int64(1)],
        ),
        (
            "MATCH (n:P) WHERE n.v = $x OR n.v = $x + 1 RETURN count(n) AS c",
            vec![Value::Int64(2), Value::Int64(6)],
        ),
        (
            "RETURN $x AS r",
            vec![Value::Int64(1), Value::from("a"), Value::Float64(0.5)],
        ),
        (
            "MATCH (n:P) WITH n, $x AS k WHERE n.v < k RETURN count(n) AS c",
            vec![Value::Int64(3), Value::from("3")],
        ),
        (
            "MATCH (n:P) RETURN count(n) AS c, $x AS x",
            vec![Value::Int64(3), Value::from("x"), Value::Null],
        ),
        (
            "MATCH (n:P) RETURN $x AS x, count(n) AS c",
            vec![Value::Int64(3), Value::from("x")],
        ),
        (
            "MATCH (n:P) RETURN n.v % 2 AS parity, $x AS x, count(n) AS c ORDER BY parity",
            vec![Value::Int64(7), Value::from("y")],
        ),
    ]
}

#[test]
fn parameterized_calls_answer_as_their_literal_statements() {
    let db = GrafeoDB::new_in_memory();
    db.execute("FOR i IN range(1, 6) INSERT (:P {v: i, s: toString(i)})")
        .unwrap();
    db.create_property_index("v").unwrap();
    let mut mismatches = Vec::new();
    let mut compared = 0;
    for (query, values) in cases() {
        for language in [Language::Gql, Language::Cypher] {
            // Twice round: the second time each call takes the prepared plan.
            for _ in 0..2 {
                for value in &values {
                    let Some(text) = literal(value, language) else {
                        continue;
                    };
                    let params: HashMap<String, Value> = [("x".to_string(), value.clone())].into();
                    let written = query.replace("$x", &text);
                    let session = db.session();
                    let (with_parameter, with_literal) = match language {
                        Language::Gql => (
                            outcome(session.execute_with_params(query, params)),
                            outcome(session.execute(&written)),
                        ),
                        Language::Cypher => (
                            outcome(session.execute_cypher_with_params(query, params)),
                            outcome(session.execute_cypher(&written)),
                        ),
                    };
                    compared += 1;
                    if with_parameter != with_literal {
                        mismatches.push(format!(
                            "{language:?} `{query}` with x = {value:?}\n  parameter: {with_parameter}\n  literal:   {with_literal}"
                        ));
                    }
                }
            }
        }
    }
    assert!(compared > 200, "{compared} comparisons");
    assert!(
        mismatches.is_empty(),
        "{} of {compared} calls differ from their literal statements:\n{}",
        mismatches.len(),
        mismatches.join("\n")
    );
}
