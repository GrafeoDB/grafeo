//! Null in expressions, as openCypher defines it and GQL's three-valued logic
//! gives it (openCypher TCK 2024.3: Quantifier1 to Quantifier4 [10], List2
//! [9], Null3 [4], Path1 [1], Path2 [3]):
//!
//! - `all`, `any`, `none` and `single` over a list with nulls are null where a
//!   null item could decide them, and true or false where no item could;
//! - a list slice with a null bound is null;
//! - `x IN []` is false, also for a null `x` or one with no value (an absent
//!   property);
//! - `nodes(null)` and `relationships(null)` are null.
//!
//! ```bash
//! cargo test -p grafeo-engine --all-features --test main null_semantics::
//! ```

#![cfg(all(feature = "gql", feature = "cypher"))]

use grafeo_common::types::Value;
use grafeo_engine::GrafeoDB;

fn run(db: &GrafeoDB, language: &str, query: &str) -> Vec<Vec<Value>> {
    match language {
        "gql" => db.execute(query),
        "cypher" => db.execute_cypher(query),
        other => panic!("unknown language {other}"),
    }
    .unwrap_or_else(|error| panic!("{language}: {query}: {error}"))
    .rows()
    .to_vec()
}

fn single_value(db: &GrafeoDB, language: &str, query: &str) -> Value {
    let rows = run(db, language, query);
    assert_eq!(rows.len(), 1, "{language}: {query}: {rows:?}");
    rows[0][0].clone()
}

fn three_valued(value: Option<bool>) -> Value {
    value.map_or(Value::Null, Value::Bool)
}

/// The quantifiers by openCypher's rule: count the items the predicate makes
/// true, false and null; an item that is null leaves open what it is.
fn quantifier(kind: &str, items: &[Option<bool>]) -> Option<bool> {
    let trues = items.iter().filter(|item| **item == Some(true)).count();
    let nulls = items.iter().filter(|item| item.is_none()).count();
    let falses = items.len() - trues - nulls;
    match kind {
        "all" => {
            if falses > 0 {
                Some(false)
            } else if nulls > 0 {
                None
            } else {
                Some(true)
            }
        }
        "any" => {
            if trues > 0 {
                Some(true)
            } else if nulls > 0 {
                None
            } else {
                Some(false)
            }
        }
        "none" => {
            if trues > 0 {
                Some(false)
            } else if nulls > 0 {
                None
            } else {
                Some(true)
            }
        }
        "single" => {
            if trues > 1 {
                Some(false)
            } else if nulls > 0 {
                None
            } else {
                Some(trues == 1)
            }
        }
        other => panic!("unknown quantifier {other}"),
    }
}

/// Every list of `true`, `false` and `null` up to three items.
fn lists() -> Vec<Vec<Option<bool>>> {
    let mut lists = vec![Vec::new()];
    let mut last = vec![Vec::new()];
    for _ in 0..3 {
        let mut next = Vec::new();
        for list in &last {
            for item in [Some(true), Some(false), None] {
                let mut longer: Vec<Option<bool>> = list.clone();
                longer.push(item);
                next.push(longer);
            }
        }
        lists.extend(next.iter().cloned());
        last = next;
    }
    lists
}

fn literal(item: Option<bool>) -> &'static str {
    match item {
        Some(true) => "true",
        Some(false) => "false",
        None => "null",
    }
}

/// Each quantifier over each list of true, false and null, against the rule.
#[test]
fn quantifiers_over_lists_with_nulls() {
    let db = GrafeoDB::new_in_memory();
    for language in ["gql", "cypher"] {
        for kind in ["all", "any", "none", "single"] {
            for list in lists() {
                let items: Vec<&str> = list.iter().map(|item| literal(*item)).collect();
                let query = format!("RETURN {kind}(x IN [{}] WHERE x) AS r", items.join(", "));
                assert_eq!(
                    single_value(&db, language, &query),
                    three_valued(quantifier(kind, &list)),
                    "{language}: {query}"
                );
            }
        }
    }
}

/// The TCK's form: a comparison that is null for a null item.
#[test]
fn quantifiers_over_a_comparison_with_null_items() {
    let db = GrafeoDB::new_in_memory();
    for language in ["gql", "cypher"] {
        for (query, expected) in [
            (
                "RETURN all(x IN [1, null, 3] WHERE x > 0) AS r",
                Value::Null,
            ),
            (
                "RETURN all(x IN [1, null, -3] WHERE x > 0) AS r",
                Value::Bool(false),
            ),
            ("RETURN any(x IN [-1, null] WHERE x > 0) AS r", Value::Null),
            (
                "RETURN any(x IN [1, null] WHERE x > 0) AS r",
                Value::Bool(true),
            ),
            ("RETURN none(x IN [-1, null] WHERE x > 0) AS r", Value::Null),
            (
                "RETURN single(x IN [1, null] WHERE x > 0) AS r",
                Value::Null,
            ),
            (
                "RETURN single(x IN [1, 2, null] WHERE x > 0) AS r",
                Value::Bool(false),
            ),
        ] {
            assert_eq!(
                single_value(&db, language, query),
                expected,
                "{language}: {query}"
            );
        }
    }
}

/// A slice with a null bound is null; a bound left out is still the start or
/// the end.
#[test]
fn a_slice_with_a_null_bound_is_null() {
    let db = GrafeoDB::new_in_memory();
    for language in ["gql", "cypher"] {
        for bounds in ["null..null", "1..null", "null..3", "..null", "null.."] {
            let query = format!("RETURN [1, 2, 3][{bounds}] AS r");
            assert_eq!(
                single_value(&db, language, &query),
                Value::Null,
                "{language}: {query}"
            );
        }
        let query = "RETURN [1, 2, 3][1..] AS r";
        assert_eq!(
            single_value(&db, language, query),
            Value::List(vec![Value::Int64(2), Value::Int64(3)].into()),
            "{language}: {query}"
        );
    }
}

/// `x IN list` by the rule: null for a null list; false for an empty one; true
/// when an item equals `x`; otherwise null when `x` or an item is null.
#[test]
fn in_with_nulls() {
    let db = GrafeoDB::new_in_memory();
    db.execute("INSERT (:N {v: 1})").unwrap();
    let lists: [(&str, Option<&[Option<i64>]>); 5] = [
        ("null", None),
        ("[]", Some(&[])),
        ("[1, 2, 3]", Some(&[Some(1), Some(2), Some(3)])),
        ("[1, 2, 3, null]", Some(&[Some(1), Some(2), Some(3), None])),
        ("[null, 1]", Some(&[None, Some(1)])),
    ];
    for language in ["gql", "cypher"] {
        // A null needle written as a literal, and two with no value: an
        // absent property and a computation on null.
        for (element_text, element) in [
            ("null", None),
            ("n.missing", None),
            ("null + 1", None),
            ("1", Some(1)),
            ("n.v", Some(1)),
            ("5", Some(5)),
        ] {
            for (list_text, list) in lists {
                let expected = match list {
                    None => Value::Null,
                    Some([]) => Value::Bool(false),
                    Some(items) if element.is_some() && items.contains(&element) => {
                        Value::Bool(true)
                    }
                    Some(items) if element.is_none() || items.contains(&None) => Value::Null,
                    Some(_) => Value::Bool(false),
                };
                // The list as a constant, and as a value bound before.
                for query in [
                    format!("MATCH (n:N) RETURN {element_text} IN {list_text} AS r"),
                    if language == "gql" {
                        format!("MATCH (n:N) LET l = {list_text} RETURN {element_text} IN l AS r")
                    } else {
                        format!(
                            "MATCH (n:N) WITH n, {list_text} AS l RETURN {element_text} IN l AS r"
                        )
                    },
                ] {
                    assert_eq!(
                        single_value(&db, language, &query),
                        expected,
                        "{language}: {query}"
                    );
                }
            }
        }
    }
}

/// A row whose needle has no value is kept by `NOT x IN []`: the IN is false.
#[test]
fn not_in_an_empty_list_keeps_a_needle_with_no_value() {
    let db = GrafeoDB::new_in_memory();
    db.execute("INSERT (:N {v: 1})").unwrap();
    for (language, query) in [
        (
            "cypher",
            "MATCH (n:N) WHERE NOT n.missing IN [] RETURN n.v AS v",
        ),
        (
            "gql",
            "MATCH (n:N) WHERE NOT n.missing IN [] RETURN n.v AS v",
        ),
        (
            "cypher",
            "MATCH (n:N) WITH n, [] AS l WHERE NOT n.missing IN l RETURN n.v AS v",
        ),
        (
            "cypher",
            "MATCH (n:N) WHERE NOT (null + 1) IN [] RETURN n.v AS v",
        ),
    ] {
        assert_eq!(
            run(&db, language, query),
            vec![vec![Value::Int64(1)]],
            "{language}: {query}"
        );
    }
}

/// `nodes` and `relationships` of null, and of a path an OPTIONAL MATCH left
/// null, are null.
#[test]
fn the_nodes_and_relationships_of_a_null_path_are_null() {
    let db = GrafeoDB::new_in_memory();
    for query in [
        "RETURN nodes(null) AS r",
        "RETURN relationships(null) AS r",
        "OPTIONAL MATCH p = (a:Missing)-[r]->() RETURN nodes(p) AS r",
        "OPTIONAL MATCH p = (a:Missing)-[r]->() RETURN relationships(p) AS r",
    ] {
        assert_eq!(single_value(&db, "cypher", query), Value::Null, "{query}");
    }
}
