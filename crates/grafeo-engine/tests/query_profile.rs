//! The structured form of `PROFILE` (#461): `QueryResult::profile` holds the
//! operator tree the text cell shows, with rows and times per operator.
//!
//! ```bash
//! cargo test -p grafeo-engine --all-features --test query_profile
//! ```

use grafeo_common::types::Value;
use grafeo_engine::GrafeoDB;
use grafeo_engine::database::{OperatorProfile, QueryProfile, QueryResult};

/// Alix (30), Gus (25), Vincent (35), Jules (40), all `:Person`.
fn people() -> GrafeoDB {
    let db = GrafeoDB::new_in_memory();
    db.execute(
        "INSERT (:Person {name: 'Alix', age: 30}), (:Person {name: 'Gus', age: 25}), \
         (:Person {name: 'Vincent', age: 35}), (:Person {name: 'Jules', age: 40})",
    )
    .unwrap();
    db
}

fn profile_of(result: &QueryResult) -> &QueryProfile {
    result.profile().expect("PROFILE carries a profile")
}

fn text_of(result: &QueryResult) -> String {
    match &result.rows()[0][0] {
        Value::String(text) => text.to_string(),
        other => panic!("the profile cell is text, got {other:?}"),
    }
}

/// The operators in pre-order, each with its depth.
fn pre_order(node: &OperatorProfile, depth: usize, out: &mut Vec<(usize, OperatorProfile)>) {
    out.push((depth, node.clone()));
    for child in &node.children {
        pre_order(child, depth + 1, out);
    }
}

#[test]
fn the_tree_matches_the_text_line_for_line() {
    let db = people();
    let result = db
        .execute("PROFILE MATCH (p:Person) WHERE p.age > 28 RETURN p.name ORDER BY p.name")
        .unwrap();
    let profile = profile_of(&result);
    let text = text_of(&result);

    let mut operators = Vec::new();
    pre_order(&profile.root, 0, &mut operators);
    let lines: Vec<&str> = text.lines().take_while(|line| !line.is_empty()).collect();
    assert_eq!(lines.len(), operators.len(), "{text}");
    for ((depth, operator), line) in operators.iter().zip(&lines) {
        let start = format!(
            "{}{} ({})  rows={}  ",
            "  ".repeat(*depth),
            operator.operator,
            operator.detail,
            operator.rows
        );
        assert!(
            line.starts_with(&start),
            "{line:?} does not start with {start:?}"
        );
    }
}

#[test]
fn the_root_produces_the_result_rows_and_times_add_up() {
    let db = people();
    let result = db
        .execute("PROFILE MATCH (p:Person) WHERE p.age > 28 RETURN p.name")
        .unwrap();
    let profile = profile_of(&result);
    assert_eq!(profile.root.rows, 3, "Alix, Vincent and Jules");
    // The profiled run is part of the whole call the result is timed for.
    assert!(profile.total_time_ms <= result.execution_time_ms.unwrap());

    let mut operators = Vec::new();
    pre_order(&profile.root, 0, &mut operators);
    for (_, operator) in &operators {
        assert!(!operator.operator.is_empty(), "every operator has a name");
        assert!(operator.self_time_ms >= 0.0);
        assert!(
            operator.time_ms >= operator.self_time_ms,
            "{}: {} < {}",
            operator.operator,
            operator.time_ms,
            operator.self_time_ms
        );
    }
}

#[cfg(feature = "cypher")]
#[test]
fn cypher_profile_carries_the_tree_too() {
    let db = people();
    let result = db
        .session()
        .execute_cypher("PROFILE MATCH (p:Person) RETURN count(p) AS c")
        .unwrap();
    assert_eq!(profile_of(&result).root.rows, 1);
}

#[test]
fn other_statements_carry_no_profile() {
    let db = people();
    for query in [
        "MATCH (p:Person) RETURN p.name",
        "EXPLAIN MATCH (p:Person) RETURN p.name",
        "INSERT (:Person {name: 'Mia'})",
    ] {
        assert!(db.execute(query).unwrap().profile().is_none(), "{query}");
    }
}

#[test]
fn the_profile_serializes_with_fixed_field_names() {
    let db = people();
    let result = db
        .execute("PROFILE MATCH (p:Person) RETURN p.name")
        .unwrap();
    let json = serde_json::to_value(profile_of(&result)).unwrap();
    let mut keys: Vec<_> = json.as_object().unwrap().keys().cloned().collect();
    keys.sort();
    assert_eq!(keys, ["root", "total_time_ms"]);
    let mut keys: Vec<_> = json["root"].as_object().unwrap().keys().cloned().collect();
    keys.sort();
    assert_eq!(
        keys,
        [
            "calls",
            "children",
            "detail",
            "operator",
            "rows",
            "self_time_ms",
            "time_ms"
        ]
    );
}
