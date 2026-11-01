//! `UNWIND range(...)` makes its integers as it goes (it never builds the
//! list), and must give exactly what `range(...)` as a value gives: the same
//! integers in the same order, nothing for an empty range, and the same error
//! for a step of 0 or a bound that is no integer. Near `i64::MAX` and
//! `i64::MIN` the range ends at its bound instead of wrapping.
//!
//! Each case runs in GQL and Cypher, as the first clause (a constant range)
//! and after a MATCH (a range per row, with bounds from the row).
//!
//! ```bash
//! cargo test -p grafeo-engine --all-features --test range_unwind_matches_range
//! ```

#![cfg(all(feature = "gql", feature = "cypher"))]

use grafeo_common::types::Value;
use grafeo_common::utils::GqlStatus;
use grafeo_engine::GrafeoDB;

/// The arguments of `range(...)` in each case.
const CASES: &[&str] = &[
    // Ordinary ranges, inclusive of the end.
    "1, 5",
    "0, 0",
    "-3, 3",
    "1, 10, 3",
    "10, 1, -3",
    "5, 1, -1",
    "-1, -5, -2",
    // Empty ranges: the end on the wrong side of the start for the step.
    "5, 1",
    "1, 5, -1",
    "0, -1",
    "3, 2, 1",
    // A step past the end gives the start alone.
    "1, 2, 100",
    "2, 1, -100",
    // Step 0.
    "1, 5, 0",
    "5, 5, 0",
    // Bounds that are no integer.
    "1.5, 4",
    "1, 4.5",
    "1, 4, 1.5",
    "'1', 4",
    "1, '4'",
    "true, 4",
    // Null bounds.
    "null, 4",
    "1, null",
    "1, 4, null",
    // Near the ends of the integers: no wrap, no overflow.
    "9223372036854775805, 9223372036854775807",
    "9223372036854775806, 9223372036854775807, 5",
    "9223372036854775807, 9223372036854775807",
    "-9223372036854775806, -9223372036854775808, -1",
    "-9223372036854775807, -9223372036854775808, -9223372036854775807",
    "9223372036854775807, 9223372036854775800, -3",
    "-9223372036854775808, -9223372036854775806",
];

/// A query's answer as the collected list, or the GQLSTATUS of its error.
#[derive(Debug, PartialEq)]
enum Answer {
    List(Value),
    Error(String),
}

fn answer(db: &GrafeoDB, language: &str, query: &str) -> Answer {
    match db.execute_language(query, language, None) {
        Ok(result) => {
            let rows = result.rows();
            assert_eq!(rows.len(), 1, "{language} `{query}`: {rows:?}");
            Answer::List(rows[0][0].clone())
        }
        Err(error) => Answer::Error(GqlStatus::from(&error).as_str().to_string()),
    }
}

/// The value `range(...)` gives, and what `UNWIND range(...)` collects, as
/// the first clause.
fn eager_and_lazy(db: &GrafeoDB, language: &str, args: &str) -> (Answer, Answer) {
    let eager = format!("RETURN range({args}) AS r");
    let lazy = format!("UNWIND range({args}) AS x RETURN collect(x) AS r");
    (answer(db, language, &eager), answer(db, language, &lazy))
}

/// An empty list and a null are the same collected answer: `collect` of no
/// rows is an empty list.
fn same(eager: &Answer, lazy: &Answer) -> bool {
    match (eager, lazy) {
        (Answer::List(Value::Null), Answer::List(Value::List(items))) => items.is_empty(),
        _ => eager == lazy,
    }
}

#[test]
fn a_constant_unwind_of_range_gives_what_range_gives() {
    let db = GrafeoDB::new_in_memory();
    let mut wrong = Vec::new();
    for language in ["gql", "cypher"] {
        for args in CASES {
            let (eager, lazy) = eager_and_lazy(&db, language, args);
            if !same(&eager, &lazy) {
                wrong.push(format!(
                    "{language} range({args}): range() {eager:?}, UNWIND {lazy:?}"
                ));
            }
        }
    }
    assert!(wrong.is_empty(), "\n{}", wrong.join("\n"));
}

/// The same with the bounds read from a matched node, one range per row:
/// the planner path for an UNWIND after another clause.
#[test]
fn a_per_row_unwind_of_range_gives_what_range_gives() {
    let db = GrafeoDB::new_in_memory();
    let mut wrong = Vec::new();
    for (index, args) in CASES.iter().enumerate() {
        let parts: Vec<&str> = args.split(", ").collect();
        let (start, end) = (parts[0], parts[1]);
        let step = parts.get(2).copied().unwrap_or("1");
        db.execute(&format!(
            "INSERT (:Case {{i: {index}, s: {start}, e: {end}, st: {step}}})"
        ))
        .unwrap();
        for language in ["gql", "cypher"] {
            let node = if language == "gql" {
                format!("MATCH (c:Case {{i: {index}}})")
            } else {
                format!("MATCH (c:Case) WHERE c.i = {index}")
            };
            let eager = answer(
                &db,
                language,
                &format!("{node} RETURN range(c.s, c.e, c.st) AS r"),
            );
            let lazy = answer(
                &db,
                language,
                &format!("{node} UNWIND range(c.s, c.e, c.st) AS x RETURN collect(x) AS r"),
            );
            if !same(&eager, &lazy) {
                wrong.push(format!(
                    "{language} range({args}) per row: range() {eager:?}, UNWIND {lazy:?}"
                ));
            }
        }
    }
    assert!(wrong.is_empty(), "\n{}", wrong.join("\n"));
}

/// The ranges near the ends of the integers hold exactly their in-bounds
/// values (a check of the values, not only of the agreement).
#[test]
fn a_range_at_the_ends_of_the_integers_stops_at_its_bound() {
    let db = GrafeoDB::new_in_memory();
    let ints = |values: &[i64]| Value::List(values.iter().map(|v| Value::Int64(*v)).collect());
    for (args, expected) in [
        (
            "9223372036854775805, 9223372036854775807",
            ints(&[i64::MAX - 2, i64::MAX - 1, i64::MAX]),
        ),
        (
            "9223372036854775806, 9223372036854775807, 5",
            ints(&[i64::MAX - 1]),
        ),
        (
            "-9223372036854775806, -9223372036854775808, -1",
            ints(&[i64::MIN + 2, i64::MIN + 1, i64::MIN]),
        ),
        (
            "-9223372036854775807, -9223372036854775808, -9223372036854775807",
            ints(&[i64::MIN + 1]),
        ),
    ] {
        let query = format!("UNWIND range({args}) AS x RETURN collect(x) AS r");
        assert_eq!(
            answer(&db, "gql", &query),
            Answer::List(expected),
            "{query}"
        );
    }
}

/// A range far longer than memory holds is counted without building it,
/// and a LIMIT stops it early.
#[test]
fn a_huge_range_unwinds_without_a_list() {
    let db = GrafeoDB::new_in_memory();
    let result = db
        .execute("UNWIND range(1, 1000000000000) AS x RETURN x LIMIT 3")
        .unwrap();
    assert_eq!(
        result.rows(),
        [[Value::Int64(1)], [Value::Int64(2)], [Value::Int64(3)]]
    );
    let result = db
        .execute("UNWIND range(1, 2000000) AS x RETURN count(x) AS c, sum(x) AS s")
        .unwrap();
    assert_eq!(
        result.rows(),
        [[Value::Int64(2_000_000), Value::Int64(2_000_001_000_000)]]
    );
}
