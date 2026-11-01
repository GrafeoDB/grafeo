//! `all_different(list)` compares the items as values, by the equality `=`
//! uses. Byte strings come only as parameters, so they are tested here; the
//! other kinds are in `tests/spec/rosetta/all_different_values.gtest`. It
//! compared a short text of each item, which shows a byte string's first
//! byte and length only, so two byte strings that differ after their first
//! byte were the same.
//!
//! ```bash
//! cargo test -p grafeo-engine --all-features --test all_different_values
//! ```

#![cfg(all(feature = "gql", feature = "cypher"))]

use std::collections::HashMap;

use grafeo_common::types::Value;
use grafeo_engine::GrafeoDB;

fn bytes(data: &[u8]) -> Value {
    Value::Bytes(data.to_vec().into())
}

/// `query` in GQL and in Cypher with the parameters `params`: the first
/// column of the first row, per language.
fn both(query: &str, params: &[(&str, Value)]) -> Vec<(&'static str, Value)> {
    let db = GrafeoDB::new_in_memory();
    let params: HashMap<String, Value> = params
        .iter()
        .map(|(name, value)| ((*name).to_string(), value.clone()))
        .collect();
    let gql = db.execute_with_params(query, params.clone()).unwrap();
    let cypher = db.execute_cypher_with_params(query, params).unwrap();
    vec![
        ("gql", gql.rows()[0][0].clone()),
        ("cypher", cypher.rows()[0][0].clone()),
    ]
}

/// Runs each `(name, query, params, expected)` case in both languages and
/// lists the ones that differ.
fn check(cases: &[(&str, &str, Vec<(&str, Value)>, Value)]) {
    let failures: Vec<String> = cases
        .iter()
        .flat_map(|(name, query, params, expected)| {
            both(query, params)
                .into_iter()
                .filter(|(_, got)| got != expected)
                .map(|(language, got)| format!("{language}, {name}: {got:?}, want {expected:?}"))
                .collect::<Vec<_>>()
        })
        .collect();
    assert!(failures.is_empty(), "\n{}", failures.join("\n"));
}

/// Two byte strings that differ only after their first byte, of one length,
/// are different; so are two of different lengths.
#[test]
fn byte_strings_that_differ_are_different() {
    check(&[
        (
            "same first byte and length",
            "RETURN all_different([$a, $b]) AS d",
            vec![("a", bytes(b"ab")), ("b", bytes(b"ac"))],
            Value::Bool(true),
        ),
        (
            "three of one length",
            "RETURN all_different([$a, $b, $c]) AS d",
            vec![
                ("a", bytes(&[1, 2, 3])),
                ("b", bytes(&[1, 2, 4])),
                ("c", bytes(&[1, 9, 3])),
            ],
            Value::Bool(true),
        ),
        (
            "a list parameter",
            "RETURN all_different($l) AS d",
            vec![(
                "l",
                Value::List(vec![bytes(b"Alix"), bytes(b"Amsterdam")].into()),
            )],
            Value::Bool(true),
        ),
    ]);
}

/// `same(list)` of byte strings: true for equal ones, false for ones that
/// differ only after their first byte.
#[test]
fn same_compares_byte_strings_as_values() {
    check(&[
        (
            "equal bytes",
            "RETURN same([$a, $b]) AS s",
            vec![("a", bytes(b"ab")), ("b", bytes(b"ab"))],
            Value::Bool(true),
        ),
        (
            "same first byte and length",
            "RETURN same([$a, $b]) AS s",
            vec![("a", bytes(b"ab")), ("b", bytes(b"ac"))],
            Value::Bool(false),
        ),
    ]);
}

/// Equal byte strings are not different, and `=` agrees with
/// `all_different` on each pair.
#[test]
fn equal_byte_strings_are_not_different() {
    check(&[
        (
            "the same bytes twice",
            "RETURN all_different([$a, $b]) AS d",
            vec![("a", bytes(b"ab")), ("b", bytes(b"ab"))],
            Value::Bool(false),
        ),
        (
            "a repeat among others",
            "RETURN all_different([$a, $b, $c]) AS d",
            vec![
                ("a", bytes(&[1, 2, 3])),
                ("b", bytes(&[4])),
                ("c", bytes(&[1, 2, 3])),
            ],
            Value::Bool(false),
        ),
        (
            "= of different bytes",
            "RETURN $a = $b AS eq",
            vec![("a", bytes(b"ab")), ("b", bytes(b"ac"))],
            Value::Bool(false),
        ),
        (
            "= of equal bytes",
            "RETURN $a = $b AS eq",
            vec![("a", bytes(b"ab")), ("b", bytes(b"ab"))],
            Value::Bool(true),
        ),
    ]);
}
