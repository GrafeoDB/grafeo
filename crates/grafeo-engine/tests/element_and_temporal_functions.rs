//! `id`, `elementId` and `properties` read the node or edge any expression
//! yields (`l[0]`, `m.x`, `startNode(r)`), as `labels` does; they read only a
//! variable and returned null for the rest. `properties` of a map is the map
//! (openCypher; it was null). `datetime`, `date` and `time` read a zoned
//! datetime (they returned null): its instant, and its date and time of day
//! where its offset puts them.
//!
//! ```bash
//! cargo test -p grafeo-engine --all-features --test element_and_temporal_functions
//! ```

#![cfg(all(feature = "lpg", feature = "gql", feature = "cypher"))]

use std::collections::BTreeMap;
use std::sync::Arc;

use grafeo_common::types::{Date, PropertyKey, Time, Value};
use grafeo_engine::GrafeoDB;

const LANGUAGES: [&str; 2] = ["gql", "cypher"];

fn rows(db: &GrafeoDB, language: &str, query: &str) -> Vec<Vec<Value>> {
    db.execute_language(query, language, None)
        .unwrap_or_else(|e| panic!("{language} `{query}`: {e}"))
        .rows()
        .to_vec()
}

fn map(entries: &[(&str, Value)]) -> Value {
    let map: BTreeMap<PropertyKey, Value> = entries
        .iter()
        .map(|(key, value)| (PropertyKey::from(*key), value.clone()))
        .collect();
    Value::Map(Arc::new(map))
}

/// Alix KNOWS Gus, since 2019.
fn pair() -> GrafeoDB {
    let db = GrafeoDB::new_in_memory();
    db.execute("INSERT (:P {name: 'Alix'})-[:KNOWS {since: 2019}]->(:P {name: 'Gus'})")
        .unwrap();
    db
}

/// A node taken from a list or a map has its id, element id and properties.
#[test]
fn a_node_an_expression_yields() {
    let db = pair();
    for language in LANGUAGES {
        let found = rows(
            &db,
            language,
            "MATCH (n:P {name: 'Alix'}) WITH n, [n] AS l, {x: n} AS m \
             RETURN id(l[0]) = id(n) AS same_id, elementId(l[0]) = elementId(n) AS same_element, \
             properties(l[0]) AS from_list, properties(m.x) AS from_map, id(m.x) = id(n) AS map_id",
        );
        let alix = map(&[("name", Value::from("Alix"))]);
        assert_eq!(
            found,
            [[
                Value::Bool(true),
                Value::Bool(true),
                alix.clone(),
                alix,
                Value::Bool(true)
            ]],
            "{language}"
        );
    }
}

/// So does an edge, and the nodes of `startNode` and `endNode`.
#[test]
fn an_edge_an_expression_yields() {
    let db = pair();
    for language in LANGUAGES {
        let found = rows(
            &db,
            language,
            "MATCH ()-[r]->() WITH r, [r] AS l \
             RETURN id(l[0]) = id(r) AS same_id, elementId(l[0]) = elementId(r) AS same_element, \
             properties(l[0]) AS edge, properties(startNode(r)) AS start, \
             elementId(endNode(r)) STARTS WITH 'n:' AS end_is_node",
        );
        assert_eq!(
            found,
            [[
                Value::Bool(true),
                Value::Bool(true),
                map(&[("since", Value::Int64(2019))]),
                map(&[("name", Value::from("Alix"))]),
                Value::Bool(true)
            ]],
            "{language}"
        );
    }
}

/// `properties` of a map is the map; of null, null.
#[test]
fn properties_of_a_map_is_the_map() {
    let db = GrafeoDB::new_in_memory();
    for language in LANGUAGES {
        assert_eq!(
            rows(
                &db,
                language,
                "RETURN properties({a: 1, b: 'x'}) AS p, properties(null) AS n"
            ),
            [[
                map(&[("a", Value::Int64(1)), ("b", Value::from("x"))]),
                Value::Null
            ]],
            "{language}"
        );
    }
    assert_eq!(
        rows(
            &GrafeoDB::new_in_memory(),
            "cypher",
            "WITH {a: 1} AS m RETURN properties(m) AS p"
        ),
        [[map(&[("a", Value::Int64(1))])]]
    );
}

/// Half past midnight at +01:00 is 23:30 the day before in UTC: `date` and
/// `time` take the local date and time of day (with the offset), `datetime`
/// the instant.
#[test]
fn temporal_functions_read_a_zoned_datetime() {
    let db = GrafeoDB::new_in_memory();
    let zoned = "zoned_datetime('2024-03-01T00:30:00+01:00')";
    for language in LANGUAGES {
        let found = rows(
            &db,
            language,
            &format!(
                "RETURN date({zoned}) AS d, time({zoned}) AS t, \
                 datetime({zoned}) = datetime('2024-02-29T23:30:00Z') AS instant, \
                 localdatetime({zoned}) IS NOT NULL AS local"
            ),
        );
        assert_eq!(
            found,
            [[
                Value::Date(Date::from_ymd(2024, 3, 1).unwrap()),
                Value::Time(Time::from_hms_nano(0, 30, 0, 0).unwrap().with_offset(3600)),
                Value::Bool(true),
                Value::Bool(true)
            ]],
            "{language}"
        );
    }
}
