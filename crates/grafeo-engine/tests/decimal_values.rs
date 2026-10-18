//! Exact decimals in queries (#610): arithmetic with decimals, integers and
//! floats, comparisons, order, grouping, aggregates and the numeric functions.
//! The decimals come in as parameters, so GQL and Cypher read the same values.
//!
//! ```bash
//! cargo test -p grafeo-engine --all-features --test decimal_values
//! ```

#![cfg(all(feature = "lpg", feature = "gql", feature = "cypher"))]

use std::collections::HashMap;

use grafeo_common::types::{Decimal, Value};
use grafeo_engine::GrafeoDB;

const LANGUAGES: [&str; 2] = ["gql", "cypher"];

fn d(text: &str) -> Value {
    Value::Decimal(Decimal::parse(text).unwrap())
}

/// The rows of `query` with `params`, each value as `Debug` prints it (a
/// decimal as `Decimal(1.50)`, so the scale shows).
fn rows(db: &GrafeoDB, language: &str, query: &str, params: &[(&str, Value)]) -> Vec<String> {
    let params: HashMap<String, Value> = params
        .iter()
        .map(|(name, value)| ((*name).to_string(), value.clone()))
        .collect();
    db.execute_language(query, language, Some(params))
        .unwrap_or_else(|e| panic!("{language} `{query}`: {e}"))
        .rows()
        .iter()
        .map(|row| {
            row.iter()
                .map(|value| format!("{value:?}"))
                .collect::<Vec<_>>()
                .join(", ")
        })
        .collect()
}

fn one(db: &GrafeoDB, language: &str, query: &str, params: &[(&str, Value)]) -> String {
    let rows = rows(db, language, query, params);
    assert_eq!(rows.len(), 1, "{language} `{query}`: {rows:?}");
    rows[0].clone()
}

#[test]
fn decimal_arithmetic_is_exact() {
    let db = GrafeoDB::new_in_memory();
    for language in LANGUAGES {
        for (query, expected) in [
            ("RETURN $a + $b AS x", "Decimal(0.3)"),
            ("RETURN $b - $a AS x", "Decimal(0.1)"),
            ("RETURN $a * $b AS x", "Decimal(0.02)"),
            ("RETURN $one / $three AS x", "Decimal(0.3333333333333333)"),
            ("RETURN -$a AS x", "Decimal(-0.1)"),
            // With an integer: a decimal.
            ("RETURN $a + 1 AS x", "Decimal(1.1)"),
            ("RETURN 2 * $price AS x", "Decimal(3.00)"),
            ("RETURN $price / 2 AS x", "Decimal(0.7500000000000000)"),
            ("RETURN $price % 1 AS x", "Decimal(0.50)"),
            // With a float: a float.
            ("RETURN $a + 0.5e0 AS x", "Float64(0.6)"),
            ("RETURN 2.0e0 * $price AS x", "Float64(3)"),
            // Division by zero and overflow give null, as for integers.
            ("RETURN $one / 0 AS x", "Null"),
            ("RETURN $max + 1 AS x", "Null"),
            ("RETURN $a + null AS x", "Null"),
        ] {
            let params = [
                ("a", d("0.1")),
                ("b", d("0.2")),
                ("one", d("1")),
                ("three", d("3")),
                ("price", d("1.50")),
                ("max", d("99999999999999999999999999999999999999")),
            ];
            assert_eq!(
                one(&db, language, query, &params),
                expected,
                "{language} `{query}`"
            );
        }
    }
}

#[test]
fn decimals_compare_with_every_number() {
    let db = GrafeoDB::new_in_memory();
    let params = [("a", d("1.50")), ("b", d("1.5")), ("c", d("3.0"))];
    for language in LANGUAGES {
        for (query, expected) in [
            ("RETURN $a = $b AS x", "Bool(true)"),
            ("RETURN $a <> $b AS x", "Bool(false)"),
            ("RETURN $c = 3 AS x", "Bool(true)"),
            ("RETURN $a = 1.5e0 AS x", "Bool(true)"),
            ("RETURN $a < 2 AS x", "Bool(true)"),
            ("RETURN $a > 1.49e0 AS x", "Bool(true)"),
            ("RETURN $a >= $b AS x", "Bool(true)"),
            ("RETURN $a IN [1, $b] AS x", "Bool(true)"),
            ("RETURN $a = null AS x", "Null"),
            // A numeric string meets a decimal as it meets a float.
            ("RETURN '1.5' = $a AS x", "Bool(true)"),
            ("RETURN $c = '3' AS x", "Bool(true)"),
            ("RETURN $c = '03.00' AS x", "Bool(true)"),
            ("RETURN $a = '15e-1' AS x", "Bool(true)"),
            ("RETURN '1.4' < $a AS x", "Bool(true)"),
            ("RETURN $a = 'one' AS x", "Bool(false)"),
            ("RETURN 1.5e0 = 'one' AS x", "Bool(false)"),
        ] {
            assert_eq!(
                one(&db, language, query, &params),
                expected,
                "{language} `{query}`"
            );
        }
    }
}

#[test]
fn decimals_order_among_the_other_numbers() {
    let db = GrafeoDB::new_in_memory();
    let params = [("a", d("1.25")), ("b", d("-2.5")), ("c", d("10"))];
    for language in LANGUAGES {
        assert_eq!(
            rows(
                &db,
                language,
                "UNWIND [$a, 2, 1.0e0, $b, $c, -3] AS v RETURN v ORDER BY v",
                &params
            ),
            [
                "Int64(-3)",
                "Decimal(-2.5)",
                "Float64(1)",
                "Decimal(1.25)",
                "Int64(2)",
                "Decimal(10)"
            ],
            "{language}"
        );
    }
}

/// `1.50` and `1.5` are one value: one group, one distinct row.
#[test]
fn equal_decimals_group_together() {
    let db = GrafeoDB::new_in_memory();
    let params = [("a", d("1.50")), ("b", d("1.5")), ("c", d("2"))];
    for language in LANGUAGES {
        let mut groups = rows(
            &db,
            language,
            "UNWIND [$a, $b, $c] AS v RETURN v, count(*) AS n",
            &params,
        );
        groups.sort();
        assert_eq!(groups.len(), 2, "{language}: {groups:?}");
        assert!(groups[0].ends_with("Int64(2)"), "{language}: {groups:?}");
        assert_eq!(
            rows(
                &db,
                language,
                "UNWIND [$a, $b, $c] AS v RETURN count(DISTINCT v) AS n",
                &params
            ),
            ["Int64(2)"],
            "{language}"
        );
    }
}

#[test]
fn aggregates_over_decimals_are_exact() {
    let db = GrafeoDB::new_in_memory();
    let tenth = d("0.1");
    let params = [
        ("t", tenth.clone()),
        ("x", d("1.25")),
        ("y", d("-3")),
        ("z", d("2.5")),
    ];
    for language in LANGUAGES {
        for (query, expected) in [
            // Ten tenths are exactly one; as floats they are 0.9999999999999999.
            (
                "UNWIND [$t, $t, $t, $t, $t, $t, $t, $t, $t, $t] AS v RETURN sum(v) AS s",
                "Decimal(1.0)",
            ),
            (
                "UNWIND [$x, $y, $z] AS v RETURN sum(v) AS s",
                "Decimal(0.75)",
            ),
            ("UNWIND [$x, 2] AS v RETURN sum(v) AS s", "Decimal(3.25)"),
            (
                "UNWIND [$x, $y, $z] AS v RETURN avg(v) AS s",
                "Decimal(0.2500000000000000)",
            ),
            ("UNWIND [$x, $y, $z] AS v RETURN min(v) AS s", "Decimal(-3)"),
            (
                "UNWIND [$x, $y, $z] AS v RETURN max(v) AS s",
                "Decimal(2.5)",
            ),
        ] {
            assert_eq!(
                one(&db, language, query, &params),
                expected,
                "{language} `{query}`"
            );
        }
    }
}

#[test]
fn numeric_functions_take_decimals() {
    let db = GrafeoDB::new_in_memory();
    let params = [("n", d("-2.345")), ("p", d("1.50")), ("h", d("2.5"))];
    for language in LANGUAGES {
        for (query, expected) in [
            ("RETURN abs($n) AS x", "Decimal(2.345)"),
            ("RETURN round($h) AS x", "Decimal(3)"),
            ("RETURN round($n) AS x", "Decimal(-2)"),
            ("RETURN ceil($n) AS x", "Decimal(-2)"),
            ("RETURN floor($n) AS x", "Decimal(-3)"),
            ("RETURN sign($n) AS x", "Int64(-1)"),
            ("RETURN toString($p) AS x", "String(\"1.50\")"),
            ("RETURN toFloat($p) AS x", "Float64(1.5)"),
            ("RETURN toInteger($n) AS x", "Int64(-2)"),
        ] {
            assert_eq!(
                one(&db, language, query, &params),
                expected,
                "{language} `{query}`"
            );
        }
    }
}

/// A stored decimal is found by an equal decimal at another scale, with and
/// without a property index.
#[test]
fn a_stored_decimal_is_found_by_an_equal_value() {
    for indexed in [false, true] {
        let db = GrafeoDB::new_in_memory();
        if indexed {
            db.create_property_index("price").unwrap();
        }
        let price = HashMap::from([("price".to_string(), d("1.50"))]);
        db.execute_with_params("INSERT (:Item {name: 'Mia', price: $price})", price)
            .unwrap();
        for language in LANGUAGES {
            for query in [
                "MATCH (n:Item {price: $p}) RETURN n.name",
                "MATCH (n:Item) WHERE n.price = $p RETURN n.name",
            ] {
                assert_eq!(
                    rows(&db, language, query, &[("p", d("1.5"))]),
                    ["String(\"Mia\")"],
                    "{language} `{query}`, indexed: {indexed}"
                );
            }
            assert_eq!(
                rows(
                    &db,
                    language,
                    "MATCH (n:Item) WHERE n.price > $p RETURN n.name",
                    &[("p", d("1.4"))]
                ),
                ["String(\"Mia\")"],
                "{language}, indexed: {indexed}"
            );
        }
    }
}

/// ISO GQL: `1.5` and `1.5M` are exact, `1.5e0`, `1.5F` and `1.5D` are
/// floats. openCypher has no decimal type: there `1.5` stays a float.
#[test]
fn gql_number_literals_are_exact_or_approximate_as_written() {
    let db = GrafeoDB::new_in_memory();
    for (query, expected) in [
        ("RETURN 0.1 + 0.2 AS x", "Decimal(0.3)"),
        ("RETURN 1.50 AS x", "Decimal(1.50)"),
        ("RETURN -1.5 AS x", "Decimal(-1.5)"),
        ("RETURN 1.5M AS x", "Decimal(1.5)"),
        ("RETURN 2M AS x", "Decimal(2)"),
        ("RETURN 1.5e0 AS x", "Float64(1.5)"),
        ("RETURN 1.5F AS x", "Float64(1.5)"),
        ("RETURN 1.5D AS x", "Float64(1.5)"),
        ("RETURN -2.5D AS x", "Float64(-2.5)"),
        ("RETURN 1.5 * 2 AS x", "Decimal(3.0)"),
        ("RETURN 1.5 + 0.5e0 AS x", "Float64(2)"),
    ] {
        assert_eq!(one(&db, "gql", query, &[]), expected, "`{query}`");
    }
    assert_eq!(
        one(&db, "cypher", "RETURN 0.1 + 0.2 AS x", &[]),
        "Float64(0.30000000000000004)"
    );
    let error = db
        .execute("RETURN 1234567890123456789012345678901234567890.5 AS x")
        .expect_err("39 digits is no decimal");
    assert!(error.to_string().contains("38 digits"), "{error}");
}
