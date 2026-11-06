//! List concatenation. GQL joins two lists with `||` (ISO/IEC 39075, list value concatenation): a null
//! operand gives null, null elements are kept, and nested lists stay elements. Cypher joins with `+`, and
//! `+` of a list and one value appends or prepends that value, as openCypher defines it. Each form reads
//! the same from literals, parameters and stored properties. Released 0.6.1 gave null for every one of
//! them but GQL `[1] + [2]`, which still works in 0.6.2.
//!
//! ```bash
//! cargo test -p grafeo-engine --all-features --test main list_concatenation::
//! ```

use std::collections::HashMap;

use grafeo_common::types::Value;
use grafeo_engine::GrafeoDB;
use grafeo_engine::session::Session;

fn list(items: Vec<Value>) -> Value {
    Value::List(items.into())
}

fn ints(items: &[i64]) -> Value {
    list(items.iter().map(|i| Value::Int64(*i)).collect())
}

#[derive(Clone, Copy, Debug)]
enum Language {
    Gql,
    Cypher,
}

/// The one value of a one-row, one-column answer.
fn one(session: &Session, language: Language, query: &str, params: &[(&str, Value)]) -> Value {
    let params: HashMap<String, Value> = params
        .iter()
        .map(|(name, value)| ((*name).to_string(), value.clone()))
        .collect();
    let result = match language {
        Language::Gql => session.execute_with_params(query, params),
        Language::Cypher => session.execute_cypher_with_params(query, params),
    }
    .unwrap_or_else(|error| panic!("{language:?} `{query}`: {error}"));
    let rows = result.rows();
    assert_eq!(rows.len(), 1, "{language:?} `{query}`: {rows:?}");
    rows[0][0].clone()
}

fn stored() -> GrafeoDB {
    let db = GrafeoDB::new_in_memory();
    db.execute("INSERT (:N {name: 'Mia', a: [1, 2], b: [3], k: 3, s: ['Alix']})")
        .unwrap();
    db
}

#[test]
fn gql_joins_two_lists_with_the_concatenation_operator() {
    let db = stored();
    let session = db.session();
    let gql = Language::Gql;
    for (query, params, want) in [
        ("RETURN [1, 2] || [3] AS r", vec![], ints(&[1, 2, 3])),
        ("RETURN [] || [1] AS r", vec![], ints(&[1])),
        ("RETURN [1] || [] AS r", vec![], ints(&[1])),
        (
            "RETURN [1] || ['a'] AS r",
            vec![],
            list(vec![Value::Int64(1), Value::from("a")]),
        ),
        (
            "RETURN [1] || [null] AS r",
            vec![],
            list(vec![Value::Int64(1), Value::Null]),
        ),
        (
            "RETURN [[1]] || [[2, 3]] AS r",
            vec![],
            list(vec![ints(&[1]), ints(&[2, 3])]),
        ),
        (
            "RETURN $a || $b AS r",
            vec![("a", ints(&[1, 2])), ("b", ints(&[3]))],
            ints(&[1, 2, 3]),
        ),
        (
            "MATCH (n:N) RETURN n.a || n.b AS r",
            vec![],
            ints(&[1, 2, 3]),
        ),
        (
            "MATCH (n:N) RETURN n.s || ['Gus'] AS r",
            vec![],
            list(vec![Value::from("Alix"), Value::from("Gus")]),
        ),
        (
            "MATCH (n:N) WHERE n.a || n.b = [1, 2, 3] RETURN n.name AS r",
            vec![],
            Value::from("Mia"),
        ),
        (
            "LET a = [1, 2], b = [3] RETURN a || b AS r",
            vec![],
            ints(&[1, 2, 3]),
        ),
    ] {
        assert_eq!(one(&session, gql, query, &params), want, "`{query}`");
    }
}

#[test]
fn a_null_operand_gives_null() {
    let db = stored();
    let session = db.session();
    for (query, params) in [
        ("RETURN [1] || null AS r", vec![]),
        ("RETURN null || [1] AS r", vec![]),
        ("RETURN [1, 2] || null AS r", vec![]),
        ("RETURN null || [1, 2] AS r", vec![]),
        (
            "RETURN $a || $n AS r",
            vec![("a", ints(&[1])), ("n", Value::Null)],
        ),
        ("MATCH (n:N) RETURN n.a || n.missing AS r", vec![]),
    ] {
        assert_eq!(
            one(&session, Language::Gql, query, &params),
            Value::Null,
            "`{query}`"
        );
    }
}

#[test]
fn cypher_appends_or_prepends_one_value() {
    let db = stored();
    let session = db.session();
    let cypher = Language::Cypher;
    for (query, params, want) in [
        ("RETURN [1, 2] + 3 AS r", vec![], ints(&[1, 2, 3])),
        ("RETURN 0 + [1, 2] AS r", vec![], ints(&[0, 1, 2])),
        ("RETURN [] + 1 AS r", vec![], ints(&[1])),
        (
            "RETURN ['Alix'] + 'Gus' AS r",
            vec![],
            list(vec![Value::from("Alix"), Value::from("Gus")]),
        ),
        (
            "RETURN [1] + [[2]] AS r",
            vec![],
            list(vec![Value::Int64(1), ints(&[2])]),
        ),
        ("RETURN [1, 2] + [3] AS r", vec![], ints(&[1, 2, 3])),
        (
            "RETURN $a + $k AS r",
            vec![("a", ints(&[1, 2])), ("k", Value::Int64(3))],
            ints(&[1, 2, 3]),
        ),
        (
            "RETURN $k + $a AS r",
            vec![("a", ints(&[1, 2])), ("k", Value::Int64(3))],
            ints(&[3, 1, 2]),
        ),
        (
            "MATCH (n:N) RETURN n.a + n.k AS r",
            vec![],
            ints(&[1, 2, 3]),
        ),
        (
            "MATCH (n:N) RETURN n.k + n.a AS r",
            vec![],
            ints(&[3, 1, 2]),
        ),
        (
            "MATCH (n:N) WHERE n.a + n.k = [1, 2, 3] RETURN n.name AS r",
            vec![],
            Value::from("Mia"),
        ),
    ] {
        assert_eq!(one(&session, cypher, query, &params), want, "`{query}`");
    }
}

/// Cypher's `+` takes a null as one more element of the list, before or after it, as any other value
/// (openCypher List4; ruled 2026-11-05). The TCK has no case of a null operand today:
/// follow it if a later TCK adds one. A null with a number, or with a null, stays null.
#[test]
fn cypher_takes_a_null_as_an_element() {
    let db = stored();
    let session = db.session();
    let cypher = Language::Cypher;
    let with_null_after = list(vec![Value::Int64(1), Value::Int64(2), Value::Null]);
    let with_null_before = list(vec![Value::Null, Value::Int64(1), Value::Int64(2)]);
    for (query, params, want) in [
        ("RETURN [1, 2] + null AS r", vec![], with_null_after.clone()),
        (
            "RETURN null + [1, 2] AS r",
            vec![],
            with_null_before.clone(),
        ),
        (
            "RETURN $a + $n AS r",
            vec![("a", ints(&[1, 2])), ("n", Value::Null)],
            with_null_after.clone(),
        ),
        (
            "RETURN $n + $a AS r",
            vec![("a", ints(&[1, 2])), ("n", Value::Null)],
            with_null_before,
        ),
        (
            "MATCH (n:N) RETURN n.a + n.missing AS r",
            vec![],
            with_null_after,
        ),
        (
            "RETURN [1] + [null] AS r",
            vec![],
            list(vec![Value::Int64(1), Value::Null]),
        ),
        ("RETURN null + 1 AS r", vec![], Value::Null),
        ("RETURN null + null AS r", vec![], Value::Null),
    ] {
        assert_eq!(one(&session, cypher, query, &params), want, "`{query}`");
    }
}

/// What worked before stays: GQL `+` of two lists (refused from 0.7.0, with a warning naming `||` in
/// 0.6.2) and `||` of two strings.
#[test]
fn what_worked_before_still_works() {
    let db = stored();
    let session = db.session();
    assert_eq!(
        one(&session, Language::Gql, "RETURN [1, 2] + [3] AS r", &[]),
        ints(&[1, 2, 3])
    );
    assert_eq!(
        one(&session, Language::Gql, "RETURN 'Ali' || 'x' AS r", &[]),
        Value::from("Alix")
    );
}
