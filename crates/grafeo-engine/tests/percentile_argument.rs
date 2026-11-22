//! The percentile of `percentileDisc` and `percentileCont` (GQL and SQL/PGQ
//! `percentile_disc`, `percentile_cont`) is a literal, a parameter or an
//! expression of them, the same for every row, and a number from 0 to 1
//! (openCypher TCK 2024.3 Aggregation6). Anything but a number literal used
//! to become the median, and a literal outside 0 to 1 was moved to its end.
//!
//! ```bash
//! cargo test -p grafeo-engine --all-features --test main percentile_argument::
//! ```

#![cfg(all(feature = "cypher", feature = "gql"))]

use std::collections::HashMap;

use grafeo_common::types::Value;
use grafeo_engine::GrafeoDB;

/// Three nodes with `price` 10.0, 20.0 and 30.0.
fn database() -> GrafeoDB {
    let db = GrafeoDB::new_in_memory();
    db.execute_cypher("CREATE (:P {price: 10.0}), (:P {price: 20.0}), (:P {price: 30.0})")
        .unwrap();
    db
}

fn run(
    db: &GrafeoDB,
    language: &str,
    query: &str,
    params: &[(&str, Value)],
) -> grafeo_common::utils::error::Result<Vec<Vec<Value>>> {
    let params: HashMap<String, Value> = params
        .iter()
        .map(|(name, value)| ((*name).to_string(), value.clone()))
        .collect();
    let result = match language {
        "gql" => db.execute_with_params(query, params),
        #[cfg(feature = "sql-pgq")]
        "sql" => db.execute_sql_with_params(query, params),
        _ => db.execute_cypher_with_params(query, params),
    }?;
    Ok(result.rows().to_vec())
}

fn value(db: &GrafeoDB, language: &str, query: &str, params: &[(&str, Value)]) -> Value {
    let rows = run(db, language, query, params)
        .unwrap_or_else(|error| panic!("{language}: {query} {params:?}: {error}"));
    assert_eq!(rows.len(), 1, "{language}: {query} {params:?}: {rows:?}");
    rows[0][0].clone()
}

fn status(db: &GrafeoDB, language: &str, query: &str, params: &[(&str, Value)]) -> String {
    let error = run(db, language, query, params)
        .expect_err(&format!("{language}: {query} {params:?} should fail"));
    grafeo_common::utils::gqlstatus::GqlStatus::from(&error)
        .as_str()
        .to_string()
}

const CYPHER_DISC: &str = "MATCH (n:P) RETURN percentileDisc(n.price, $p) AS p";
const CYPHER_CONT: &str = "MATCH (n:P) RETURN percentileCont(n.price, $p) AS p";
const GQL_DISC: &str = "MATCH (n:P) RETURN percentile_disc(n.price, $p) AS p";
const GQL_CONT: &str = "MATCH (n:P) RETURN percentile_cont(n.price, $p) AS p";

/// A parameter is the percentile, in Cypher and GQL, at both ends and inside
/// (TCK Aggregation6 [1] and [2]: 0.0 and 1.0 gave the median).
#[test]
fn a_parameter_is_the_percentile() {
    let db = database();
    for (language, query) in [
        ("cypher", CYPHER_DISC),
        ("cypher", CYPHER_CONT),
        ("gql", GQL_DISC),
        ("gql", GQL_CONT),
    ] {
        for (percentile, expected) in [
            (Value::Float64(0.0), 10.0),
            (Value::Float64(0.5), 20.0),
            (Value::Float64(1.0), 30.0),
            (Value::Int64(0), 10.0),
            (Value::Int64(1), 30.0),
            (
                Value::Float64(0.75),
                if query.contains("Cont") || query.contains("cont") {
                    25.0
                } else {
                    20.0
                },
            ),
        ] {
            assert_eq!(
                value(&db, language, query, &[("p", percentile.clone())]),
                Value::Float64(expected),
                "{language}: {query} with {percentile:?}"
            );
        }
    }
}

/// A literal at either end, and an expression of literals and parameters.
#[test]
fn a_literal_or_an_expression_of_constants_is_the_percentile() {
    let db = database();
    for (language, query, params, expected) in [
        (
            "cypher",
            "MATCH (n:P) RETURN percentileDisc(n.price, 0) AS p",
            vec![],
            10.0,
        ),
        (
            "cypher",
            "MATCH (n:P) RETURN percentileDisc(n.price, 1.0) AS p",
            vec![],
            30.0,
        ),
        (
            "gql",
            "MATCH (n:P) RETURN percentile_cont(n.price, 1) AS p",
            vec![],
            30.0,
        ),
        (
            "gql",
            "MATCH (n:P) RETURN percentile_cont(n.price, 0.0) AS p",
            vec![],
            10.0,
        ),
        (
            "cypher",
            "MATCH (n:P) RETURN percentileCont(n.price, $p / 4.0) AS p",
            vec![("p", Value::Int64(1))],
            15.0,
        ),
        (
            "gql",
            "MATCH (n:P) RETURN percentile_disc(n.price, 1 - $p) AS p",
            vec![("p", Value::Float64(0.0))],
            30.0,
        ),
    ] {
        assert_eq!(
            value(&db, language, query, &params),
            Value::Float64(expected),
            "{language}: {query}"
        );
    }
}

/// A percentile that reads the rows is refused before the statement runs
/// (42000), even where every row would give the same number.
#[test]
fn a_percentile_that_reads_the_rows_is_refused() {
    let db = database();
    for (language, query) in [
        (
            "cypher",
            "MATCH (n:P) RETURN percentileDisc(n.price, n.price / 100) AS p",
        ),
        (
            "cypher",
            "MATCH (n:P) WITH n, 0.5 AS half RETURN percentileCont(n.price, half) AS p",
        ),
        (
            "gql",
            "MATCH (n:P) RETURN percentile_disc(n.price, n.price / 100) AS p",
        ),
    ] {
        assert_eq!(
            status(&db, language, query, &[]),
            "42000",
            "{language}: {query}"
        );
    }
}

/// A percentile outside 0 to 1 is 22003, a null or no number 22G03, as a
/// literal and as a parameter (TCK Aggregation6 [3] and [4]); none is moved
/// to the nearest end.
#[test]
fn a_percentile_outside_zero_to_one_or_no_number_is_refused() {
    let db = database();
    for (language, query) in [
        ("cypher", CYPHER_DISC),
        ("cypher", CYPHER_CONT),
        ("gql", GQL_DISC),
        ("gql", GQL_CONT),
    ] {
        for (percentile, expected) in [
            (Value::Float64(1.5), "22003"),
            (Value::Float64(-0.1), "22003"),
            (Value::Float64(1.1), "22003"),
            (Value::Int64(1000), "22003"),
            (Value::Int64(-1), "22003"),
            (Value::Float64(f64::NAN), "22003"),
            (Value::Null, "22G03"),
            (Value::from("half"), "22G03"),
        ] {
            assert_eq!(
                status(&db, language, query, &[("p", percentile.clone())]),
                expected,
                "{language}: {query} with {percentile:?}"
            );
        }
    }
    for (language, query, expected) in [
        (
            "cypher",
            "MATCH (n:P) RETURN percentileDisc(n.price, 1.5) AS p",
            "22003",
        ),
        (
            "cypher",
            "MATCH (n:P) RETURN percentileCont(n.price, -0.1) AS p",
            "22003",
        ),
        (
            "gql",
            "MATCH (n:P) RETURN percentile_disc(n.price, 1.5) AS p",
            "22003",
        ),
        (
            "gql",
            "MATCH (n:P) RETURN percentile_cont(n.price, -0.1) AS p",
            "22003",
        ),
        (
            "cypher",
            "MATCH (n:P) RETURN percentileDisc(n.price, null) AS p",
            "22G03",
        ),
        (
            "gql",
            "MATCH (n:P) RETURN percentile_cont(n.price, NULL) AS p",
            "22G03",
        ),
    ] {
        assert_eq!(
            status(&db, language, query, &[]),
            expected,
            "{language}: {query}"
        );
    }
}

/// SQL/PGQ reads a parameter as the percentile too, and refuses one
/// outside 0 to 1.
#[cfg(feature = "sql-pgq")]
#[test]
fn sql_pgq_takes_a_parameter_as_the_percentile() {
    let db = database();
    let query = "SELECT PERCENTILE_DISC(price, $p) AS p \
                 FROM GRAPH_TABLE (MATCH (n:P) COLUMNS (n.price AS price))";
    assert_eq!(
        value(&db, "sql", query, &[("p", Value::Float64(0.0))]),
        Value::Float64(10.0)
    );
    assert_eq!(
        value(&db, "sql", query, &[("p", Value::Float64(1.0))]),
        Value::Float64(30.0)
    );
    assert_eq!(
        status(&db, "sql", query, &[("p", Value::Float64(1.5))]),
        "22003"
    );
}
