//! A chain of comparisons in Cypher, `a < b <= c`, is the conjunction of its
//! comparisons, `a < b AND b <= c` (openCypher 9, "Comparison operators":
//! comparisons can be chained arbitrarily, and `a op1 b op2 c` does not compare
//! `a` with `c`). It was read as `(a < b) <= c`, which compares a boolean with
//! a number and is null, so a WHERE with a chain dropped every row. GQL has no
//! chained comparison: it rejects one instead of returning null.
//!
//! ```bash
//! cargo test -p grafeo-engine --all-features --test chained_comparisons
//! ```

#![cfg(all(feature = "lpg", feature = "gql", feature = "cypher"))]

use grafeo_common::types::Value;
use grafeo_engine::{GrafeoDB, Session};

fn cypher(session: &Session, query: &str) -> Vec<Vec<Value>> {
    session
        .execute_cypher(query)
        .unwrap_or_else(|error| panic!("`{query}` failed: {error}"))
        .rows()
        .to_vec()
}

/// The value of `expression` returned on its own.
fn value_of(session: &Session, expression: &str) -> Value {
    let rows = cypher(session, &format!("RETURN {expression} AS v"));
    assert_eq!(rows.len(), 1, "{expression}: one row");
    rows[0][0].clone()
}

fn people(session: &Session) {
    cypher(
        session,
        "CREATE (:Person {name: 'Alix', age: 19}), (:Person {name: 'Gus', age: 25}), \
         (:Person {name: 'Vincent', age: 30}), (:Person {name: 'Mia', age: 88}), \
         (:Person {name: 'Jules'})",
    );
}

fn names(rows: &[Vec<Value>]) -> Vec<String> {
    let mut names: Vec<String> = rows
        .iter()
        .map(|row| match &row[0] {
            Value::String(name) => name.to_string(),
            other => format!("{other:?}"),
        })
        .collect();
    names.sort();
    names
}

#[test]
fn a_chain_is_the_conjunction_of_its_comparisons() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    for (expression, expected) in [
        ("1 <= 2 < 3", true),
        ("3 > 2 >= 1", true),
        ("1 <= 5 < 3", false),
        ("5 < 3 <= 19", false),
        ("1 < 2 < 3 < 4", true),
        ("1 < 3 < 2 < 4", false),
        ("1 < 2 < 4 < 3", false),
        ("3 = 3 = 3", true),
        ("3 = 3 = 19", false),
        ("3 = 3 <> 19", true),
        // `a <> b <> c` does not compare a with c.
        ("3 <> 19 <> 3", true),
        ("19 > 3 < 88", true),
        ("1 < 1 + 1 < 3", true),
        ("'a' < 'b' <= 'b'", true),
    ] {
        assert_eq!(
            value_of(&session, expression),
            Value::Bool(expected),
            "{expression}"
        );
    }
}

#[test]
fn a_chain_with_a_null_operand_follows_three_valued_and() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    for (expression, expected) in [
        ("1 < null < 3", Value::Null),
        ("null < 3 < 19", Value::Null),
        ("3 < 19 < null", Value::Null),
        // One false comparison makes the chain false, null or not.
        ("19 < 3 < null", Value::Bool(false)),
        ("null < 19 < 3", Value::Bool(false)),
    ] {
        assert_eq!(value_of(&session, expression), expected, "{expression}");
    }
}

#[test]
fn a_chain_binds_tighter_than_not_and_or() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    for (expression, expected) in [
        ("NOT 1 < 2 < 3", false),
        ("NOT 1 < 3 < 2", true),
        ("1 < 2 < 3 AND 3 < 2", false),
        ("1 < 3 < 2 OR 2 < 3", true),
        ("1 < 3 < 2 XOR 3 < 19 < 88", true),
    ] {
        assert_eq!(
            value_of(&session, expression),
            Value::Bool(expected),
            "{expression}"
        );
    }
}

#[test]
fn a_where_with_a_chain_on_a_property_keeps_the_rows_in_range() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    people(&session);
    assert_eq!(
        names(&cypher(
            &session,
            "MATCH (p:Person) WHERE 19 <= p.age < 30 RETURN p.name"
        )),
        ["Alix", "Gus"],
        "a node without the property is not in range"
    );
    assert_eq!(
        names(&cypher(
            &session,
            "MATCH (p:Person) WHERE 88 > p.age >= 25 RETURN p.name"
        )),
        ["Gus", "Vincent"]
    );
}

#[test]
fn a_chain_reads_parameters() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    people(&session);
    let params = std::collections::HashMap::from([
        ("low".to_string(), Value::Int64(25)),
        ("high".to_string(), Value::Int64(88)),
    ]);
    let rows = session
        .execute_cypher_with_params(
            "MATCH (p:Person) WHERE $high > p.age >= $low RETURN p.name",
            params,
        )
        .expect("the chain with parameters runs")
        .rows()
        .to_vec();
    assert_eq!(names(&rows), ["Gus", "Vincent"]);
}

#[test]
fn a_chain_in_a_case_counts_the_rows_in_range() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    people(&session);
    assert_eq!(
        cypher(
            &session,
            "MATCH (p:Person) \
             RETURN sum(CASE WHEN 19 <= p.age < 30 THEN 1 ELSE 0 END) AS inside, \
                    sum(CASE WHEN p.age < 19 OR p.age >= 30 THEN 1 ELSE 0 END) AS outside"
        ),
        [vec![Value::Int64(2), Value::Int64(2)]]
    );
}

#[test]
fn gql_rejects_a_chained_comparison() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    assert!(
        session.execute("RETURN 1 <= 2 < 3 AS v").is_err(),
        "GQL has no chained comparison, so the query is an error and not null"
    );
}
