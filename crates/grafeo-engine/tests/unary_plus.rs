//! Unary plus is the number under it, exactly: a decimal keeps its scale and
//! a negative zero its sign; like unary minus it reads numbers only, and
//! gives null for anything else. SQL/PGQ did not parse `+x`, and GQL and
//! Cypher passed any value through it.

#![cfg(feature = "gql")]

use grafeo_common::types::Value;
use grafeo_engine::GrafeoDB;

fn row(db: &GrafeoDB, language: &str, query: &str) -> Vec<Value> {
    db.execute_language(query, language, None)
        .unwrap_or_else(|error| panic!("{language} `{query}`: {error}"))
        .rows()[0]
        .clone()
}

#[test]
fn a_number_under_plus_is_exactly_that_number() {
    let db = GrafeoDB::new_in_memory();
    db.execute("INSERT (:P {age: 30, price: 1.50, zero: -0.0e0})")
        .unwrap();
    let values = row(
        &db,
        "gql",
        "MATCH (n:P) RETURN +n.age AS a, +n.price AS p, +n.zero AS z",
    );
    assert_eq!(values[0], Value::Int64(30));
    assert_eq!(values[1].to_string(), "1.50", "a decimal keeps its scale");
    match values[2] {
        Value::Float64(zero) => assert!(zero == 0.0 && zero.is_sign_negative(), "{zero}"),
        ref other => panic!("{other:?}"),
    }
}

#[test]
fn plus_reads_numbers_only_as_minus_does() {
    let db = GrafeoDB::new_in_memory();
    db.execute("INSERT (:P {name: 'Alix', at: DATE '2024-03-01', ok: true})")
        .unwrap();
    let mut languages = vec!["gql"];
    #[cfg(feature = "cypher")]
    languages.push("cypher");
    for language in languages {
        let values = row(
            &db,
            language,
            "MATCH (n:P) RETURN +n.name AS a, +n.at AS b, +n.ok AS c, -n.name AS d",
        );
        assert_eq!(
            values,
            [Value::Null, Value::Null, Value::Null, Value::Null],
            "{language}"
        );
    }
}

/// A constant under plus folds as one under minus does, and a property value
/// written through it is the number.
#[test]
fn plus_of_a_constant_is_the_constant() {
    let db = GrafeoDB::new_in_memory();
    db.execute("INSERT (:P {age: +30, neg: +(-3)})").unwrap();
    let values = row(
        &db,
        "gql",
        "MATCH (n:P) RETURN n.age AS a, n.neg AS n, +'x' AS s",
    );
    assert_eq!(values, [Value::Int64(30), Value::Int64(-3), Value::Null]);
}

#[cfg(feature = "sql-pgq")]
#[test]
fn sql_pgq_parses_plus_everywhere_minus_goes() {
    let db = GrafeoDB::new_in_memory();
    db.execute("INSERT (:P {age: 30}), (:P {age: 2})").unwrap();
    for (query, expected) in [
        (
            "SELECT * FROM GRAPH_TABLE (MATCH (n:P) WHERE +n.age > 3 COLUMNS (+n.age AS a))",
            vec![Value::Int64(30)],
        ),
        (
            "SELECT +g.a AS a FROM GRAPH_TABLE (MATCH (n:P) COLUMNS (n.age AS a)) AS g WHERE +g.a > 3",
            vec![Value::Int64(30)],
        ),
        (
            "SELECT count(*) AS c FROM GRAPH_TABLE (MATCH (n:P) COLUMNS (n.age AS a)) AS g HAVING +count(*) > 1",
            vec![Value::Int64(2)],
        ),
    ] {
        assert_eq!(row(&db, "sql", query), expected, "{query}");
    }
}
