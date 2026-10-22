//! Exact decimals in queries (#610): arithmetic with decimals, integers and
//! floats, comparisons, order, grouping, aggregates and the numeric functions.
//! The decimals come in as parameters, so GQL and Cypher read the same values.
//!
//! ```bash
//! cargo test -p grafeo-engine --all-features --test decimal_values
//! ```

#![cfg(all(feature = "gql", feature = "cypher"))]

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

// ISO: GV17
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
            // Division by zero and overflow are errors (#639, in
            // numeric_errors.rs); a null operand gives null.
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
// ISO: GL04, GL05, GL07, GL08, GL09, GL10
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
        ("RETURN 2F AS x", "Float64(2)"),
        ("RETURN 1.5e0F AS x", "Float64(1.5)"),
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

/// `toDecimal(x)` makes a decimal of a number or a numeric string, a float by
/// its shortest text; `toDecimal(x, precision, scale)` rounds it to the scale
/// and gives null when it has more integer digits than the type holds.
#[test]
fn to_decimal_makes_decimals() {
    let db = GrafeoDB::new_in_memory();
    for language in LANGUAGES {
        for (query, expected) in [
            ("RETURN toDecimal(1.5e0) AS x", "Decimal(1.5)"),
            ("RETURN toDecimal(3) AS x", "Decimal(3)"),
            ("RETURN toDecimal($a) AS x", "Decimal(1.50)"),
            ("RETURN toDecimal('2.50') AS x", "Decimal(2.50)"),
            ("RETURN toDecimal('1e3') AS x", "Decimal(1000)"),
            (
                "RETURN toDecimal(0.1e0 + 0.2e0) AS x",
                "Decimal(0.30000000000000004)",
            ),
            ("RETURN toDecimal('abc') AS x", "Null"),
            ("RETURN toDecimal(true) AS x", "Null"),
            ("RETURN toDecimal(null) AS x", "Null"),
            // 1.005 is the shortest text of its float, so it rounds up.
            ("RETURN toDecimal(1.005e0, 5, 2) AS x", "Decimal(1.01)"),
            ("RETURN toDecimal($a, 5, 0) AS x", "Decimal(2)"),
            ("RETURN toDecimal(1234.5e0, 5, 2) AS x", "Null"),
            ("RETURN toDecimal(1, 0, 0) AS x", "Null"),
            ("RETURN toDecimal(1, 5, 6) AS x", "Null"),
        ] {
            assert_eq!(
                one(&db, language, query, &[("a", d("1.50"))]),
                expected,
                "{language} `{query}`"
            );
        }
    }
}

/// GQL `CAST` to `DECIMAL` or `DEC`: bare it keeps the value's own scale,
/// with a precision and scale it rounds as `toDecimal` does.
#[test]
fn gql_casts_to_decimal() {
    let db = GrafeoDB::new_in_memory();
    for (query, expected) in [
        ("RETURN CAST('1.50' AS DECIMAL) AS x", "Decimal(1.50)"),
        ("RETURN CAST(1.5e0 AS DEC) AS x", "Decimal(1.5)"),
        ("RETURN CAST(2.345 AS DECIMAL(10, 2)) AS x", "Decimal(2.35)"),
        ("RETURN CAST(2.5 AS dec(3)) AS x", "Decimal(3)"),
        ("RETURN CAST(123.4 AS DECIMAL(4, 2)) AS x", "Null"),
        (
            "RETURN CAST([1, '2.5', 0.5e0] AS LIST<DECIMAL>) AS x",
            "List([Decimal(1), Decimal(2.5), Decimal(0.5)])",
        ),
    ] {
        assert_eq!(one(&db, "gql", query, &[]), expected, "`{query}`");
    }
    for query in [
        "RETURN CAST(1 AS DECIMAL(0)) AS x",
        "RETURN CAST(1 AS DECIMAL(39, 2)) AS x",
        "RETURN CAST(1 AS DECIMAL(5, 6)) AS x",
    ] {
        let error = db.execute(query).expect_err(query);
        assert!(
            error.to_string().contains("precision"),
            "`{query}`: {error}"
        );
    }
}

/// SQL/PGQ number literals and `CAST` (#610): a number with a point is an
/// exact decimal and one with an exponent a float, as SQL specifies; `CAST`
/// converts to every SQL type, `DECIMAL(p, s)` and `NUMERIC(p, s)` as
/// `toDecimal` does and `VARCHAR(n)` to the first `n` characters.
#[cfg(feature = "sql-pgq")]
#[test]
fn sql_pgq_literals_and_casts() {
    let db = GrafeoDB::new_in_memory();
    db.execute("INSERT (:Person {name: 'Alix', age: 30})")
        .unwrap();
    let query = |expression: &str| {
        format!("SELECT * FROM GRAPH_TABLE (MATCH (n:Person) COLUMNS ({expression} AS x))")
    };
    for (expression, expected) in [
        ("0.1 + 0.2", "Decimal(0.3)"),
        ("1.50", "Decimal(1.50)"),
        ("-1.5", "Decimal(-1.5)"),
        ("1.5e0", "Float64(1.5)"),
        ("CAST('2.345' AS DECIMAL(10, 2))", "Decimal(2.35)"),
        ("CAST(2.5 AS NUMERIC)", "Decimal(2.5)"),
        ("CAST(2.5 AS DEC(3))", "Decimal(3)"),
        ("CAST(123.4 AS NUMERIC(4, 2))", "Null"),
        ("CAST(1.5 AS DOUBLE)", "Float64(1.5)"),
        ("CAST(1.5 AS DOUBLE PRECISION)", "Float64(1.5)"),
        ("CAST(1.5 AS REAL)", "Float64(1.5)"),
        ("CAST('42' AS INTEGER)", "Int64(42)"),
        ("CAST(n.age AS BIGINT)", "Int64(30)"),
        ("CAST(n.age AS VARCHAR)", "String(\"30\")"),
        ("CAST(n.name AS VARCHAR(2))", "String(\"Al\")"),
        ("CAST('true' AS BOOLEAN)", "Bool(true)"),
        ("CAST('2024-03-15' AS DATE)", "Date(2024-03-15)"),
    ] {
        assert_eq!(
            one(&db, "sql", &query(expression), &[]),
            expected,
            "`{expression}`"
        );
    }
    for (expression, word) in [
        ("CAST(1 AS DECIMAL(39))", "precision"),
        ("CAST(1 AS WIDGET)", "WIDGET"),
        ("CAST(1 DECIMAL)", "AS"),
    ] {
        let error = db
            .execute_language(&query(expression), "sql", None)
            .expect_err(expression)
            .to_string();
        assert!(error.contains(word), "`{expression}`: {error}");
    }
}

/// SPARQL numbers follow the XPath numeric promotion of SPARQL 1.1 (#610):
/// `1.5` is an `xsd:decimal`, `1.5e0` an `xsd:double`; an integer with an
/// integer is an integer except for division, which gives a decimal; a decimal
/// with an integer or a decimal is a decimal; a double makes a double.
#[cfg(all(feature = "sparql", feature = "triple-store"))]
#[test]
fn sparql_numbers_follow_xsd_promotion() {
    let db = GrafeoDB::new_in_memory();
    for (expression, expected) in [
        ("0.1 + 0.2", "Decimal(0.3)"),
        ("1.50", "Decimal(1.50)"),
        ("1.5e0", "Float64(1.5)"),
        ("1.5 + 1", "Decimal(2.5)"),
        ("1.5 * 2.0e0", "Float64(3)"),
        ("7 - 2", "Int64(5)"),
        ("5 / 2", "Decimal(2.500000000000000)"),
        ("1 / 3", "Decimal(0.3333333333333333)"),
        ("\"2.50\"^^xsd:decimal", "Decimal(2.50)"),
        ("xsd:decimal(\"1.25\")", "Decimal(1.25)"),
        ("xsd:decimal(1.5e0)", "Decimal(1.5)"),
        ("xsd:decimal(3)", "Decimal(3)"),
        ("STRDT(\"1.25\", xsd:decimal)", "Decimal(1.25)"),
        ("xsd:double(1.5)", "Float64(1.5)"),
        ("xsd:integer(2.5)", "Int64(2)"),
        (
            "DATATYPE(1.5)",
            "String(\"http://www.w3.org/2001/XMLSchema#decimal\")",
        ),
        (
            "DATATYPE(5 / 2)",
            "String(\"http://www.w3.org/2001/XMLSchema#decimal\")",
        ),
        ("ISNUMERIC(1.5)", "Bool(true)"),
        ("ABS(-1.5)", "Decimal(1.5)"),
        ("ROUND(2.5)", "Decimal(3)"),
        ("CEIL(2.1)", "Decimal(3)"),
        ("FLOOR(-2.1)", "Decimal(-3)"),
        ("STR(1.50)", "String(\"1.50\")"),
        ("1.5 = 1.50", "Bool(true)"),
        ("1.5 < 2", "Bool(true)"),
        ("0.1 + 0.2 = 0.3", "Bool(true)"),
        // An integer past the i64 range goes on as a decimal.
        ("9223372036854775807 + 1", "Decimal(9223372036854775808)"),
        ("-9223372036854775807 * 2", "Decimal(-18446744073709551614)"),
        // A value that does not cast, or a lexical form of another type.
        ("xsd:integer(\"abc\")", "Null"),
        ("xsd:decimal(\"1e3\")", "Null"),
        ("xsd:decimal(\"1e3\"^^xsd:double)", "Decimal(1000)"),
        ("xsd:boolean(\"1\")", "Bool(true)"),
        ("xsd:string(1.50)", "String(\"1.50\")"),
    ] {
        let query = format!(
            "PREFIX xsd: <http://www.w3.org/2001/XMLSchema#> \
             SELECT ({expression} AS ?x) WHERE {{}}"
        );
        assert_eq!(one(&db, "sparql", &query, &[]), expected, "`{expression}`");
    }
}

/// A stored `xsd:decimal` compares with a decimal of the query by value, also
/// across scales (#610).
#[cfg(all(feature = "sparql", feature = "triple-store"))]
#[test]
fn sparql_filters_stored_decimals_by_value() {
    let db = GrafeoDB::new_in_memory();
    db.execute_sparql(
        "PREFIX xsd: <http://www.w3.org/2001/XMLSchema#> INSERT DATA { \
         <http://ex.org/tea> <http://ex.org/price> \"2.50\"^^xsd:decimal . \
         <http://ex.org/cake> <http://ex.org/price> \"19.99\"^^xsd:decimal . \
         <http://ex.org/water> <http://ex.org/price> \"0.5\"^^xsd:decimal . }",
    )
    .unwrap();
    let names = |filter: &str| {
        rows(
            &db,
            "sparql",
            &format!(
                "SELECT ?item WHERE {{ ?item <http://ex.org/price> ?p FILTER({filter}) }} \
                 ORDER BY ?item"
            ),
            &[],
        )
    };
    assert_eq!(
        names("?p > 1.5"),
        [
            "String(\"http://ex.org/cake\")",
            "String(\"http://ex.org/tea\")"
        ]
    );
    assert_eq!(names("?p = 2.5"), ["String(\"http://ex.org/tea\")"]);
    assert_eq!(names("?p <= 0.50"), ["String(\"http://ex.org/water\")"]);
}

/// A decimal written into the store is an `xsd:decimal` literal with its
/// scale (#610).
#[cfg(all(feature = "sparql", feature = "triple-store"))]
#[test]
fn sparql_writes_decimals_as_xsd_decimal_literals() {
    let db = GrafeoDB::new_in_memory();
    db.execute_sparql(
        "INSERT { <http://ex.org/fee> <http://ex.org/amount> ?x } WHERE { BIND(1.50 AS ?x) }",
    )
    .unwrap();
    assert_eq!(
        one(
            &db,
            "sparql",
            "SELECT (DATATYPE(?o) AS ?d) (STR(?o) AS ?s) WHERE { ?f <http://ex.org/amount> ?o }",
            &[]
        ),
        "String(\"http://www.w3.org/2001/XMLSchema#decimal\"), String(\"1.50\")"
    );
}

/// `DELETE DATA` and `DELETE ... WHERE` remove the typed literals they are
/// given: a date, a duration and a decimal, each through the one conversion
/// from a value to its RDF literal.
#[cfg(all(feature = "sparql", feature = "triple-store"))]
#[test]
fn sparql_deletes_the_typed_literals_it_wrote() {
    let count = |db: &GrafeoDB| {
        one(
            db,
            "sparql",
            "SELECT (COUNT(*) AS ?n) WHERE { ?s ?p ?o }",
            &[],
        )
    };
    let data = "PREFIX xsd: <http://www.w3.org/2001/XMLSchema#> \
                { <http://ex.org/trip> <http://ex.org/on> \"2024-03-15\"^^xsd:date . \
                  <http://ex.org/trip> <http://ex.org/lasts> \"P1D\"^^xsd:duration . \
                  <http://ex.org/trip> <http://ex.org/costs> 19.50 . }";
    let db = GrafeoDB::new_in_memory();
    db.execute_sparql(&data.replacen('{', "INSERT DATA {", 1))
        .unwrap();
    assert_eq!(count(&db), "Int64(3)");
    db.execute_sparql(&data.replacen('{', "DELETE DATA {", 1))
        .unwrap();
    assert_eq!(count(&db), "Int64(0)", "DELETE DATA");

    db.execute_sparql(&data.replacen('{', "INSERT DATA {", 1))
        .unwrap();
    db.execute_sparql(
        "PREFIX xsd: <http://www.w3.org/2001/XMLSchema#> \
         DELETE { ?s ?p ?o } WHERE { ?s ?p ?o }",
    )
    .unwrap();
    assert_eq!(count(&db), "Int64(0)", "DELETE WHERE");
}

/// A triple scan gives a numeric literal as a number of its XSD type (#610):
/// `xsd:integer` and the integer types derived from it as integers, an
/// `xsd:decimal` as an exact decimal with its scale, `xsd:double` and
/// `xsd:float` as floats. A plain literal and an ill-typed one stay strings.
#[cfg(all(feature = "sparql", feature = "triple-store"))]
#[test]
fn sparql_scans_give_numbers_of_their_type() {
    let db = GrafeoDB::new_in_memory();
    db.execute_sparql(
        r#"PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>
        INSERT DATA {
            <http://ex.org/a> <http://ex.org/v> "30"^^xsd:integer .
            <http://ex.org/b> <http://ex.org/v> "7"^^xsd:int .
            <http://ex.org/c> <http://ex.org/v> "2.50"^^xsd:decimal .
            <http://ex.org/d> <http://ex.org/v> "1.5E0"^^xsd:double .
            <http://ex.org/e> <http://ex.org/v> "0.25"^^xsd:float .
            <http://ex.org/f> <http://ex.org/v> "30" .
            <http://ex.org/g> <http://ex.org/v> "abc"^^xsd:integer .
        }"#,
    )
    .unwrap();
    // The subject orders the rows; it is projected too, as an ORDER BY of a
    // variable the SELECT leaves out fails (inbox), and dropped here.
    let values = |projection: &str| {
        rows(
            &db,
            "sparql",
            &format!("SELECT ?s {projection} WHERE {{ ?s <http://ex.org/v> ?v }} ORDER BY ?s"),
            &[],
        )
        .into_iter()
        .map(|row| {
            row.split_once(", ")
                .map_or(row.clone(), |(_, rest)| rest.to_string())
        })
        .collect::<Vec<_>>()
    };
    assert_eq!(
        values("?v"),
        [
            "Int64(30)",
            "Int64(7)",
            "Decimal(2.50)",
            "Float64(1.5)",
            "Float64(0.25)",
            "String(\"30\")",
            "String(\"abc\")",
        ]
    );
    // Arithmetic on stored numbers follows their types.
    assert_eq!(
        values("(?v * 2 AS ?x)")[..5],
        [
            "Int64(60)",
            "Int64(14)",
            "Decimal(5.00)",
            "Float64(3)",
            "Float64(0.5)",
        ]
    );
    // STR gives the lexical form, DATATYPE the type.
    assert_eq!(
        values("(STR(?v) AS ?t) (DATATYPE(?v) AS ?d)")[2],
        "String(\"2.50\"), String(\"http://www.w3.org/2001/XMLSchema#decimal\")"
    );
    // A sum of stored decimals is exact.
    assert_eq!(
        one(
            &db,
            "sparql",
            "PREFIX xsd: <http://www.w3.org/2001/XMLSchema#> \
             SELECT (SUM(?v) AS ?total) WHERE { ?s <http://ex.org/v> ?v \
             FILTER(DATATYPE(?v) = xsd:decimal) }",
            &[]
        ),
        "Decimal(2.50)"
    );
}
