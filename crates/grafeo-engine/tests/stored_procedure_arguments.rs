//! A stored procedure's arguments reach its body as values, as parameters do
//! in a statement. The call put each argument into the body's text in place
//! of `$name`, in hash-map order: `$a` replaced the start of `$ab`, a value
//! holding query text became part of the query, and a list, a map or a date
//! arrived as its display string.
//!
//! ```bash
//! cargo test -p grafeo-engine --all-features --test stored_procedure_arguments
//! ```

#![cfg(all(feature = "lpg", feature = "gql"))]

use std::collections::{BTreeMap, HashMap};
use std::sync::Arc;

use grafeo_common::types::{PropertyKey, Value};
use grafeo_engine::GrafeoDB;

/// The rows `query` returns with `params`.
fn rows(db: &GrafeoDB, query: &str, params: &[(&str, Value)]) -> Vec<Vec<Value>> {
    let params: HashMap<String, Value> = params
        .iter()
        .map(|(name, value)| ((*name).to_string(), value.clone()))
        .collect();
    db.execute_with_params(query, params)
        .unwrap_or_else(|e| panic!("`{query}`: {e}"))
        .rows()
        .to_vec()
}

/// One parameter's name is a prefix of another's: each gets its own value,
/// whatever order the arguments are substituted in.
#[test]
fn a_parameter_that_prefixes_another_keeps_its_value() {
    let db = GrafeoDB::new_in_memory();
    db.execute(
        "CREATE PROCEDURE pair(a STRING, ab STRING, abc STRING) RETURNS (a STRING, ab STRING, \
         abc STRING) AS { RETURN $abc AS abc, $a AS a, $ab AS ab }",
    )
    .unwrap();
    assert_eq!(
        rows(
            &db,
            "CALL pair('Alix', 'Gus', 'Mia') YIELD a, ab, abc RETURN a, ab, abc",
            &[]
        ),
        [[Value::from("Alix"), Value::from("Gus"), Value::from("Mia")]]
    );
}

/// A value that holds a quote, a keyword and a comment is a value: it comes
/// back as it went in, and nothing it says runs.
#[test]
fn a_value_holding_query_text_stays_a_value() {
    let db = GrafeoDB::new_in_memory();
    db.execute("INSERT (:Person {name: 'Alix'}), (:Person {name: 'Gus'})")
        .unwrap();
    db.execute("CREATE PROCEDURE echo(text STRING) RETURNS (v STRING) AS { RETURN $text AS v }")
        .unwrap();
    let text = "it's' AS v NEXT MATCH (n) DETACH DELETE n RETURN 'x' -- $text";
    assert_eq!(
        rows(
            &db,
            "CALL echo($t) YIELD v RETURN v",
            &[("t", Value::from(text))]
        ),
        [[Value::from(text)]]
    );
    assert_eq!(
        rows(&db, "MATCH (p:Person) RETURN count(p) AS c", &[]),
        [[Value::Int64(2)]],
        "the value deleted nothing"
    );
}

/// Every kind of value arrives as itself, not as its display string.
#[test]
fn every_value_kind_arrives_as_itself() {
    let db = GrafeoDB::new_in_memory();
    db.execute("CREATE PROCEDURE same(x ANY) RETURNS (v ANY) AS { RETURN $x AS v }")
        .unwrap();
    let map: BTreeMap<PropertyKey, Value> = [(PropertyKey::new("city"), Value::from("Prague"))]
        .into_iter()
        .collect();
    for value in [
        Value::Null,
        Value::Bool(true),
        Value::Int64(-19),
        Value::Float64(2.5),
        Value::from("Amsterdam"),
        Value::List(Arc::from(vec![Value::Int64(3), Value::from("Berlin")])),
        Value::Map(Arc::new(map)),
        db.execute("RETURN date('2026-10-20') AS d").unwrap().rows()[0][0].clone(),
        Value::Bytes(Arc::from(vec![0_u8, 255])),
    ] {
        assert_eq!(
            rows(
                &db,
                "CALL same($x) YIELD v RETURN v",
                &[("x", value.clone())]
            ),
            [[value.clone()]],
            "{value:?}"
        );
    }
}

/// A literal argument works as before, and the body's own parameter that the
/// procedure does not declare is an error naming it, not a text left in place.
#[test]
fn literals_work_and_an_undeclared_parameter_is_an_error() {
    let db = GrafeoDB::new_in_memory();
    db.execute(
        "CREATE PROCEDURE add(a INTEGER, b INTEGER) RETURNS (s INTEGER) AS { RETURN $a + $b AS s }",
    )
    .unwrap();
    assert_eq!(
        rows(&db, "CALL add(3, 19) YIELD s RETURN s", &[]),
        [[Value::Int64(22)]]
    );
    db.execute("CREATE PROCEDURE stray(a INTEGER) RETURNS (s INTEGER) AS { RETURN $b AS s }")
        .unwrap();
    let error = db
        .execute("CALL stray(1) YIELD s RETURN s")
        .expect_err("the body reads an undeclared parameter");
    assert!(
        error.to_string().contains("$b") || error.to_string().contains("'b'"),
        "{error}"
    );
}
