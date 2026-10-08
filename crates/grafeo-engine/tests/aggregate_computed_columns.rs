//! A grouped or global aggregate that needs a computed column (a literal or
//! expression group key, or an expression operand) keeps every other column
//! it reads as it is, in GQL and Cypher. The planner used to copy those
//! columns through typed as nodes, so strings, floats, booleans, lists and
//! maps became `0` before the aggregate saw them (integers survived as IDs):
//! `collect(name)` gave `[0, 0]`, a string group key merged every group into
//! one, and `sum(score)` summed zeros.
//!
//! ```bash
//! cargo test -p grafeo-engine --all-features --test aggregate_computed_columns
//! ```

#![cfg(all(feature = "lpg", feature = "gql"))]

use grafeo_common::types::{PropertyKey, Value};
use grafeo_engine::GrafeoDB;

#[cfg(feature = "cypher")]
const LANGUAGES: &[&str] = &["gql", "cypher"];
#[cfg(not(feature = "cypher"))]
const LANGUAGES: &[&str] = &["gql"];

/// Alix (age 3, score 3.5, active, tags [Amsterdam], nick Al), Gus (age 19,
/// score 19.25, not active, tags [Berlin, Paris], no nick) and Mia (age 88,
/// score 88.5, active, no tags, no nick); Alix knows Gus (w 3) and Gus knows
/// Mia (w 19).
fn people() -> GrafeoDB {
    let db = GrafeoDB::new_in_memory();
    db.execute(
        "INSERT (alix:Person {name: 'Alix', age: 3, score: 3.5, active: true, \
         tags: ['Amsterdam'], nick: 'Al'}), \
         (gus:Person {name: 'Gus', age: 19, score: 19.25, active: false, \
         tags: ['Berlin', 'Paris']}), \
         (mia:Person {name: 'Mia', age: 88, score: 88.5, active: true, tags: []}), \
         (alix)-[:KNOWS {w: 3}]->(gus), (gus)-[:KNOWS {w: 19}]->(mia)",
    )
    .unwrap();
    db
}

/// A short form of a value that keeps its kind visible: strings quoted,
/// floats with a decimal point, a node as `('name')`, an edge as `[TYPE w]`,
/// a path as the nodes and edges it holds, in order, between `<` and `>`.
fn describe(value: &Value) -> String {
    match value {
        Value::String(text) => format!("'{text}'"),
        Value::Int64(number) => number.to_string(),
        Value::Float64(number) => format!("{number:?}"),
        Value::Bool(flag) => flag.to_string(),
        Value::Null => "null".to_string(),
        Value::List(items) => {
            let items: Vec<String> = items.iter().map(describe).collect();
            format!("[{}]", items.join(", "))
        }
        Value::Path { nodes, edges } => {
            let mut steps = Vec::new();
            for (index, node) in nodes.iter().enumerate() {
                steps.push(describe(node));
                if let Some(edge) = edges.get(index) {
                    steps.push(describe(edge));
                }
            }
            format!("<{}>", steps.join(" "))
        }
        Value::Map(map) => {
            let get = |key: &str| map.get(&PropertyKey::new(key));
            match (get("_type"), get("_labels")) {
                (Some(Value::String(edge_type)), _) => {
                    format!(
                        "[{edge_type} {}]",
                        describe(get("w").unwrap_or(&Value::Null))
                    )
                }
                (None, Some(_)) => {
                    format!("({})", describe(get("name").unwrap_or(&Value::Null)))
                }
                _ => {
                    let mut entries: Vec<String> = map
                        .iter()
                        .map(|(key, item)| format!("{}: {}", key.as_str(), describe(item)))
                        .collect();
                    entries.sort();
                    format!("{{{}}}", entries.join(", "))
                }
            }
        }
        other => format!("{other:?}"),
    }
}

/// The rows of `query` in `language`, each value described.
fn rows(db: &GrafeoDB, language: &str, query: &str) -> Vec<Vec<String>> {
    let result = match language {
        "gql" => db.execute(query),
        #[cfg(feature = "cypher")]
        "cypher" => db.execute_cypher(query),
        other => panic!("unknown language {other}"),
    }
    .unwrap_or_else(|error| panic!("{language}: {query}: {error}"));
    result
        .rows()
        .iter()
        .map(|row| row.iter().map(describe).collect())
        .collect()
}

/// Checks `query` in every language against `expected`, one row per slice.
fn check(db: &GrafeoDB, query: &str, expected: &[&[&str]]) {
    for language in LANGUAGES {
        assert_eq!(
            rows(db, language, query),
            expected
                .iter()
                .map(|row| row.iter().map(|cell| (*cell).to_string()).collect())
                .collect::<Vec<Vec<String>>>(),
            "{language}: {query}"
        );
    }
}

#[test]
fn a_computed_group_key_keeps_the_strings_it_groups() {
    check(
        &people(),
        "MATCH (n:Person) WITH n.name AS name, n.age AS age \
         RETURN age % 2 AS odd, collect(name) AS names ORDER BY odd",
        &[&["0", "['Mia']"], &["1", "['Alix', 'Gus']"]],
    );
}

#[test]
fn a_literal_group_key_keeps_floats_booleans_lists_and_maps() {
    check(
        &people(),
        "MATCH (n:Person) WITH n.name AS name, n.score AS score, n.active AS active, \
         n.tags AS tags, {age: n.age} AS info \
         RETURN 'all' AS g, collect(name) AS names, collect(score) AS scores, \
         collect(active) AS flags, collect(tags) AS tag_lists, collect(info) AS infos",
        &[&[
            "'all'",
            "['Alix', 'Gus', 'Mia']",
            "[3.5, 19.25, 88.5]",
            "[true, false, true]",
            "[['Amsterdam'], ['Berlin', 'Paris'], []]",
            "[{age: 3}, {age: 19}, {age: 88}]",
        ]],
    );
}

#[test]
fn a_constant_group_key_over_unwound_floats_keeps_the_floats() {
    check(
        &people(),
        "UNWIND [2.5, 3.5] AS x RETURN 0 AS g, collect(x) AS xs, sum(x) AS total",
        &[&["0", "[2.5, 3.5]", "6.0"]],
    );
}

#[test]
fn a_string_group_key_beside_a_computed_operand_keeps_its_groups() {
    check(
        &people(),
        "MATCH (n:Person) WITH n.name AS name, n.age AS age \
         RETURN name, sum(age * 2) AS doubled ORDER BY name",
        &[&["'Alix'", "6"], &["'Gus'", "38"], &["'Mia'", "176"]],
    );
}

#[test]
fn a_computed_operand_without_grouping_keeps_the_other_operands() {
    check(
        &people(),
        "MATCH (n:Person) WITH n.name AS name, n.score AS score \
         RETURN count(DISTINCT name) AS names, sum(score * 2) AS doubled, \
         max(name) AS last, min(score) AS low",
        &[&["3", "222.5", "'Mia'", "3.5"]],
    );
}

#[test]
fn several_aggregates_over_a_computed_group_key() {
    check(
        &people(),
        "MATCH (n:Person) WITH n.name AS name, n.age AS age, n.score AS score, \
         n.active AS active \
         RETURN age % 2 AS odd, count(*) AS c, count(DISTINCT name) AS names, \
         min(name) AS first, sum(score) AS total, collect(active) AS flags ORDER BY odd",
        &[
            &["0", "1", "1", "'Mia'", "88.5", "[true]"],
            &["1", "2", "2", "'Alix'", "22.75", "[true, false]"],
        ],
    );
}

#[test]
fn a_computed_group_key_keeps_nulls_and_the_values_beside_them() {
    check(
        &people(),
        "MATCH (n:Person) WITH n.name AS name, n.nick AS nick \
         RETURN size(name) AS len, count(nick) AS nicks, max(nick) AS nick, \
         collect(name) AS names ORDER BY len",
        &[
            &["3", "0", "null", "['Gus', 'Mia']"],
            &["4", "1", "'Al'", "['Alix']"],
        ],
    );
}

#[test]
fn an_aggregating_with_over_a_computed_key_keeps_the_strings() {
    check(
        &people(),
        "MATCH (n:Person) WITH n.name AS name, n.age AS age \
         WITH age % 2 AS odd, collect(name) AS names RETURN odd, names ORDER BY odd",
        &[&["0", "['Mia']"], &["1", "['Alix', 'Gus']"]],
    );
}

#[test]
fn nodes_and_edges_beside_a_computed_group_key_stay_nodes_and_edges() {
    let db = people();
    check(
        &db,
        "MATCH (a:Person)-[k:KNOWS]->(b:Person) \
         RETURN a.age % 2 AS odd, collect(k) AS ks, collect(b) AS bs",
        &[&["1", "[[KNOWS 3], [KNOWS 19]]", "[('Gus'), ('Mia')]"]],
    );
    check(
        &db,
        "MATCH (a:Person)-[k:KNOWS]->(b:Person) \
         RETURN b, count(DISTINCT k) AS c, sum(k.w * 2) AS doubled ORDER BY doubled",
        &[&["('Gus')", "1", "6"], &["('Mia')", "1", "38"]],
    );
    check(
        &db,
        "MATCH (a:Person)-[k:KNOWS]->(b:Person) \
         RETURN k, sum(b.age + 1) AS older ORDER BY older",
        &[&["[KNOWS 3]", "20"], &["[KNOWS 19]", "89"]],
    );
}

#[test]
fn unwound_nodes_beside_a_computed_group_key_stay_nodes() {
    check(
        &people(),
        "MATCH (a:Person) WITH collect(a) AS everyone UNWIND everyone AS p \
         RETURN p.age % 2 AS odd, collect(p) AS ps ORDER BY odd",
        &[&["0", "[('Mia')]"], &["1", "[('Alix'), ('Gus')]"]],
    );
}

#[test]
fn paths_beside_a_computed_operand_stay_paths() {
    check(
        &people(),
        "MATCH p = (a:Person)-[:KNOWS]->(b:Person) \
         RETURN a.age % 2 AS odd, collect(p) AS ps, collect(length(p)) AS lengths",
        // A path holds node and edge IDs: Alix (node 0) KNOWS (edge 0) Gus
        // (node 1), and Gus (node 1) KNOWS (edge 1) Mia (node 2).
        &[&["1", "[<0 0 1>, <1 1 2>]", "[1, 1]"]],
    );
}

#[test]
fn values_from_an_optional_match_beside_a_computed_group_key_stay_values() {
    check(
        &people(),
        "MATCH (a:Person) OPTIONAL MATCH (a)-[:KNOWS]->(b:Person) \
         WITH a, b.name AS friend, [a.name, a.age] AS pair \
         RETURN a.age % 2 AS odd, count(friend) AS friends, min(friend) AS first, \
         max(friend) AS last, collect(pair) AS pairs ORDER BY odd",
        &[
            &["0", "0", "null", "null", "[['Mia', 88]]"],
            &["1", "2", "'Gus'", "'Mia'", "[['Alix', 3], ['Gus', 19]]"],
        ],
    );
}

/// `CALL` procedures need the `algos` feature.
#[cfg(feature = "algos")]
#[test]
fn procedure_columns_beside_a_computed_group_key_stay_values() {
    check(
        &people(),
        "CALL db.labels() YIELD label RETURN size(label) AS len, collect(label) AS labels",
        &[&["6", "['Person']"]],
    );
}
