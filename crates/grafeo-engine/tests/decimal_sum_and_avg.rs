//! `sum` and `avg` of decimals are exact and do not depend on the order of
//! the rows (#669). A sum adds without a limit and fails (22003) only when
//! the final value's integer part needs more than 38 digits; a fraction that
//! does not fit next to it is rounded half away from zero, as multiplication
//! rounds. An average is never beyond the largest row, so it never fails; it
//! follows the decimal division rule (at least 16 significant digits, at
//! least the largest input scale, at most 38 digits).
//!
//! The expected values were computed exactly (Python `fractions.Fraction`)
//! with these rules.

use std::collections::HashMap;

use grafeo_common::types::{Decimal, Value};
use grafeo_common::utils::GqlStatus;
use grafeo_engine::GrafeoDB;

const LANGUAGES: [&str; 2] = ["gql", "cypher"];

fn d(text: &str) -> Value {
    Value::Decimal(Decimal::parse(text).unwrap_or_else(|e| panic!("{text}: {e}")))
}

/// 37 nines and a zero decimal: twice it needs 39 digits.
const A: &str = "9999999999999999999999999999999999999.0";
/// 38 nines, the largest decimal integer.
const M: &str = "99999999999999999999999999999999999999";

fn negated(text: &str) -> String {
    text.strip_prefix('-')
        .map_or_else(|| format!("-{text}"), ToString::to_string)
}

/// The single row of `query` over the list `$xs`, in `language`.
fn row(db: &GrafeoDB, language: &str, query: &str, xs: &[Value]) -> Vec<Value> {
    let params = HashMap::from([("xs".to_string(), Value::List(xs.to_vec().into()))]);
    let result = db
        .execute_language(query, language, Some(params))
        .unwrap_or_else(|e| panic!("{language} `{query}` over {xs:?}: {e}"));
    let rows = result.rows();
    assert_eq!(rows.len(), 1, "{language} `{query}`: {rows:?}");
    rows[0].clone()
}

const SUM_AND_AVG: &str = "UNWIND $xs AS x RETURN sum(x) AS s, avg(x) AS a";

/// Every order of `values` (Heap's algorithm).
fn permutations(values: &[Value]) -> Vec<Vec<Value>> {
    fn heap(k: usize, items: &mut Vec<Value>, out: &mut Vec<Vec<Value>>) {
        if k <= 1 {
            out.push(items.clone());
            return;
        }
        heap(k - 1, items, out);
        for i in 0..k - 1 {
            if k.is_multiple_of(2) {
                items.swap(i, k - 1);
            } else {
                items.swap(0, k - 1);
            }
            heap(k - 1, items, out);
        }
    }
    let mut items = values.to_vec();
    let mut out = Vec::new();
    heap(items.len(), &mut items, &mut out);
    out
}

/// The cases of the issue: a running total past 38 digits that the next row
/// brings back is fine, in any order.
#[test]
fn a_sum_does_not_depend_on_the_order_of_the_rows() {
    let db = GrafeoDB::new_in_memory();
    let (a, minus_a) = (d(A), d(&negated(A)));
    for language in LANGUAGES {
        for xs in [
            [a.clone(), a.clone(), minus_a.clone()],
            [a.clone(), minus_a.clone(), a.clone()],
            [minus_a.clone(), a.clone(), a.clone()],
        ] {
            assert_eq!(
                row(&db, language, "UNWIND $xs AS x RETURN sum(x) AS s", &xs)[0],
                a,
                "{language} sum {xs:?}"
            );
        }
    }
}

/// The same per group: every group holds a, a and -a in another order.
#[test]
fn a_grouped_sum_does_not_depend_on_the_order_of_the_rows() {
    let db = GrafeoDB::new_in_memory();
    let (a, minus_a) = (d(A), d(&negated(A)));
    let orders = [
        [a.clone(), a.clone(), minus_a.clone()],
        [a.clone(), minus_a.clone(), a.clone()],
        [minus_a.clone(), a.clone(), a.clone()],
    ];
    let rows: Vec<Value> = orders
        .iter()
        .enumerate()
        .flat_map(|(group, order)| {
            order.iter().map(move |x| {
                Value::List(vec![Value::Int64(i64::try_from(group).unwrap()), x.clone()].into())
            })
        })
        .collect();
    for language in LANGUAGES {
        let params = HashMap::from([("rows".to_string(), Value::List(rows.clone().into()))]);
        let result = db
            .execute_language(
                "UNWIND $rows AS r RETURN r[0] AS g, sum(r[1]) AS s ORDER BY g",
                language,
                Some(params),
            )
            .unwrap_or_else(|e| panic!("{language}: {e}"));
        assert_eq!(
            result.rows(),
            [0, 1, 2].map(|g| vec![Value::Int64(g), a.clone()]),
            "{language}"
        );
    }
}

/// Three sets of six decimals of scales 0 to 38 and both signs; every one of
/// the 720 orders of each gives the exact sum and average.
#[test]
fn every_order_of_a_mixed_set_gives_the_exact_sum_and_average() {
    let sets = [
        // A fraction that fits: scale 35 for the sum, 36 for the average.
        (
            vec![
                "123.45",
                "-0.000000000000000000000000000000000001",
                "99999999999999999999.5",
                "-7",
                "0.33333333333333333333333333333333333333",
                "-99999999999999999999.4",
            ],
            "116.88333333333333333333333333333333333",
            "19.480555555555555555555555555555555555",
        ),
        // Two of the 38-digit values in a row need 39 digits; the total
        // fits, its fraction rounded to scale 0.
        (
            vec![
                "90000000000000000000000000000000000000",
                "90000000000000000000000000000000000000",
                "-90000000000000000000000000000000000000",
                "0.00000000000000000000000000000000000001",
                "-0.5",
                "12.25",
            ],
            "90000000000000000000000000000000000012",
            "15000000000000000000000000000000000002",
        ),
        // A 38-digit integer part: the sum keeps no fraction.
        (
            vec![
                "12345678901234567890123456789012345678",
                "-0.12345678901234567890123456789012345678",
                "98765432109876543210.123456789012345678",
                "-1234567890123456789012345678901234567.5",
                "0.000000000000000000000000000000000001",
                "-42",
            ],
            "11111111011111111199876543219987654278",
            "1851851835185185199979423869997942379.7",
        ),
    ];
    let db = GrafeoDB::new_in_memory();
    for (values, sum, average) in sets {
        let values: Vec<Value> = values.into_iter().map(d).collect();
        let expected = vec![d(sum), d(average)];
        for language in LANGUAGES {
            for order in permutations(&values) {
                let actual = row(&db, language, SUM_AND_AVG, &order);
                assert_eq!(actual, expected, "{language} {order:?}");
                // The value and its scale: the decimal prints as expected.
                assert_eq!(
                    format!("{actual:?}"),
                    format!("{expected:?}"),
                    "{language} {order:?}: the scale"
                );
            }
        }
    }
}

/// An average is never beyond its largest row: of two 38-digit values it is
/// that value, where the sum of the two does not fit.
#[test]
fn an_average_never_fails() {
    let db = GrafeoDB::new_in_memory();
    let m = d(M);
    let minus_m = d(&negated(M));
    for language in LANGUAGES {
        for (xs, expected) in [
            (vec![m.clone(), m.clone()], m.clone()),
            (vec![m.clone(), m.clone(), m.clone()], m.clone()),
            (vec![minus_m.clone(), minus_m.clone()], minus_m.clone()),
            (vec![m.clone(), minus_m.clone()], d("0")),
            (vec![d(A), d(A)], d(A)),
        ] {
            assert_eq!(
                row(&db, language, "UNWIND $xs AS x RETURN avg(x) AS a", &xs)[0],
                expected,
                "{language} avg {xs:?}"
            );
        }
        // The same values summed do not fit.
        let error = db
            .execute_language(
                "UNWIND $xs AS x RETURN sum(x) AS s",
                language,
                Some(HashMap::from([(
                    "xs".to_string(),
                    Value::List(vec![m.clone(), m.clone()].into()),
                )])),
            )
            .expect_err("a sum of two 38-digit values does not fit");
        assert_eq!(
            GqlStatus::from(&error),
            GqlStatus::DATA_NUMERIC_OUT_OF_RANGE,
            "{error}"
        );
        assert!(
            error
                .to_string()
                .contains("numeric value out of range: sum"),
            "{language}: {error}"
        );
    }
}

/// Where the sum fits, the average is what it was: the division rule on the
/// sum, e.g. 1 / 3 with 16 significant digits, and the largest input scale.
#[test]
fn an_average_follows_the_division_rule() {
    let db = GrafeoDB::new_in_memory();
    for language in LANGUAGES {
        for (xs, expected) in [
            (vec!["1", "0", "0"], "0.3333333333333333"),
            (vec!["1.50", "2.25"], "1.875000000000000"),
            (vec!["0.000001", "0"], "0.0000005000000000000000"),
            (vec!["10.000", "20.000"], "15.00000000000000"),
            (vec!["2", "3"], "2.500000000000000"),
            // The largest input scale wins over 16 significant digits.
            (
                vec!["1.00000000000000000001", "2"],
                "1.50000000000000000001",
            ),
        ] {
            let xs: Vec<Value> = xs.into_iter().map(d).collect();
            let actual = row(&db, language, "UNWIND $xs AS x RETURN avg(x) AS a", &xs)[0].clone();
            assert_eq!(
                format!("{actual:?}"),
                format!("{:?}", d(expected)),
                "{language} avg {xs:?}"
            );
        }
    }
}

/// What does not change: a float row makes the result a float, no rows give
/// NULL, DISTINCT follows the same rules, and integers before, between and
/// after decimals join the exact sum in any order.
#[test]
fn floats_empty_input_distinct_and_integers_keep_their_rules() {
    let db = GrafeoDB::new_in_memory();
    let (a, minus_a) = (d(A), d(&negated(A)));
    for language in LANGUAGES {
        let floats = row(
            &db,
            language,
            SUM_AND_AVG,
            &[d("1.5"), Value::Float64(2.0), d("0.5")],
        );
        assert_eq!(
            floats,
            [Value::Float64(4.0), Value::Float64(4.0 / 3.0)],
            "{language}"
        );

        let empty = row(
            &db,
            language,
            "UNWIND $xs AS x WITH x WHERE x > 100 RETURN sum(x) AS s, avg(x) AS a",
            &[d("1.5"), d("2.5")],
        );
        assert_eq!(empty, [Value::Null, Value::Null], "{language} no rows");

        for xs in [
            [a.clone(), a.clone(), minus_a.clone(), a.clone()],
            [minus_a.clone(), a.clone(), a.clone(), minus_a.clone()],
        ] {
            assert_eq!(
                row(
                    &db,
                    language,
                    "UNWIND $xs AS x RETURN sum(DISTINCT x) AS s, avg(DISTINCT x) AS a",
                    &xs
                ),
                [d("0.0"), d("0.0")],
                "{language} DISTINCT {xs:?}"
            );
        }
        assert_eq!(
            row(
                &db,
                language,
                "UNWIND $xs AS x RETURN avg(DISTINCT x) AS a",
                &[d(M), d(M), d(M)]
            )[0],
            d(M),
            "{language} avg DISTINCT"
        );

        let mixed = [
            Value::Int64(i64::MAX),
            Value::Int64(i64::MAX),
            d("0.5"),
            Value::Int64(-i64::MAX),
        ];
        for order in permutations(&mixed) {
            assert_eq!(
                row(&db, language, "UNWIND $xs AS x RETURN sum(x) AS s", &order)[0],
                d("9223372036854775807.5"),
                "{language} {order:?}"
            );
        }
    }
}

/// Grouped sums and averages that spill under memory pressure give the same
/// results. Each of 512 groups gets a twelve times, then -a ten times, in
/// rounds far apart: its running total passes 38 digits in a spilled and
/// reloaded state and comes back to 2a.
#[cfg(feature = "spill")]
#[test]
fn spilled_groups_keep_exact_decimal_sums_and_averages() {
    use grafeo_common::memory::buffer::{MemoryRegion, PressureLevel};
    use grafeo_engine::Config;

    let spill_root = tempfile::tempdir().unwrap();
    let db = GrafeoDB::with_config(
        Config::in_memory()
            .with_memory_limit(64 * 1024 * 1024)
            .with_spill_path(spill_root.path())
            .without_query_timeout(),
    )
    .unwrap();
    // Eleven rounds of 1024 rows, two per group and round.
    let rows: Vec<Value> = (0..11)
        .flat_map(|round| {
            (0..1024_i64).map(move |i| {
                let x = if round < 6 { d(A) } else { d(&negated(A)) };
                Value::List(vec![Value::Int64(i % 512), x].into())
            })
        })
        .collect();
    let params = HashMap::from([("rows".to_string(), Value::List(rows.into()))]);
    let manager = db.buffer_manager();
    let pressure = manager
        .try_allocate(
            manager.config().budget * 9 / 10,
            MemoryRegion::ExecutionBuffers,
        )
        .unwrap();
    assert_eq!(manager.pressure_level(), PressureLevel::High);
    // 2a does not fit at scale 1 next to 38 integer digits: scale 0.
    let sum = d("19999999999999999999999999999999999998");
    // 2a / 22, by the division rule: 36 integer digits, scale 1.
    let average = d("909090909090909090909090909090909090.8");
    for language in LANGUAGES {
        let result = db
            .execute_language(
                "UNWIND $rows AS r RETURN r[0] AS g, sum(r[1]) AS s, avg(r[1]) AS a",
                language,
                Some(params.clone()),
            )
            .unwrap_or_else(|e| panic!("{language}: {e}"));
        let mut rows = result.rows().to_vec();
        rows.sort_by_key(|row| row[0].as_int64().unwrap());
        assert_eq!(rows.len(), 512, "{language}");
        for (g, row) in rows.iter().enumerate() {
            assert_eq!(
                row,
                &vec![
                    Value::Int64(i64::try_from(g).unwrap()),
                    sum.clone(),
                    average.clone()
                ],
                "{language} group {g}"
            );
            assert_eq!(
                format!("{:?}", row[2]),
                format!("{average:?}"),
                "{language} group {g}: the scale"
            );
        }
    }
    drop(pressure);
}
