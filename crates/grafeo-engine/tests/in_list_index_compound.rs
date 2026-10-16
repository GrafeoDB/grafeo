//! `var.prop IN list` uses the property index also inside an `AND` chain
//! (#456): the indexed `IN` conjunct finds the candidates and the whole
//! predicate decides each, so the rows are those of a scan.
//!
//! ```bash
//! cargo test -p grafeo-engine --all-features --test in_list_index_compound
//! ```

#![cfg(all(feature = "lpg", feature = "gql"))]

use std::collections::HashMap;

use grafeo_common::types::Value;
use grafeo_engine::GrafeoDB;

/// A thousand `Doc` nodes (ids 0..999, every other one active, a city by
/// `id % 5`) and a few `Other` nodes with the same ids; with and without a
/// property index on `id`.
fn docs(indexed: bool) -> GrafeoDB {
    let db = GrafeoDB::new_in_memory();
    if indexed {
        db.create_property_index("id").unwrap();
    }
    db.execute(
        "UNWIND range(0, 999) AS i INSERT (:Doc {id: i, active: i % 2 = 0, \
         city: ['Amsterdam', 'Berlin', 'Paris', 'Prague', 'Barcelona'][i % 5]})",
    )
    .unwrap();
    db.execute("UNWIND [3, 4, 19] AS i INSERT (:Other {id: i, active: true})")
        .unwrap();
    db
}

fn params() -> HashMap<String, Value> {
    let ids = [3, 4, 19, 88, 500, 5000]
        .into_iter()
        .map(Value::Int64)
        .chain([Value::Null])
        .collect::<Vec<_>>();
    HashMap::from([("ids".to_string(), Value::List(ids.into()))])
}

/// The rows of `query`, as text in text order.
fn rows(db: &GrafeoDB, query: &str) -> Vec<String> {
    let mut rows: Vec<String> = db
        .execute_with_params(query, params())
        .unwrap()
        .rows()
        .iter()
        .map(|row| format!("{row:?}"))
        .collect();
    rows.sort();
    rows
}

fn profile(db: &GrafeoDB, query: &str) -> String {
    db.execute_with_params(&format!("PROFILE {query}"), params())
        .unwrap()
        .rows()
        .iter()
        .map(|row| format!("{:?}", row[0]))
        .collect::<Vec<_>>()
        .join("\n")
}

/// Compound predicates with an `IN` list on the indexed `id`.
const COMPOUND: [&str; 9] = [
    "MATCH (n) WHERE n.id IN $ids AND n.active = true RETURN n.id",
    "MATCH (n:Doc) WHERE n.id IN $ids AND n.active = true RETURN n.id",
    "MATCH (n:Doc) WHERE n.active = true AND n.id IN $ids RETURN n.id",
    "MATCH (n:Doc) WHERE n.id IN $ids AND n.id > 4 RETURN n.id",
    "MATCH (n:Doc) WHERE n.id IN $ids AND (n.city = 'Prague' OR n.city = 'Amsterdam') RETURN n.id, n.city",
    "MATCH (n:Doc) WHERE n.id IN $ids AND NOT n.active RETURN n.id",
    "MATCH (n:Doc) WHERE n.id IN [3, 4, 19, 500] AND n.city STARTS WITH 'P' RETURN n.id",
    "MATCH (n:Doc) WHERE n.id IN $ids AND n.id IN [4, 19, 88] AND n.active = true RETURN n.id",
    "MATCH (n:Doc) WITH n WHERE n.id IN $ids AND n.active = true RETURN n.id",
];

#[test]
fn compound_in_lists_find_the_rows_of_a_scan() {
    let plain = docs(false);
    let indexed = docs(true);
    for query in COMPOUND {
        assert_eq!(rows(&indexed, query), rows(&plain, query), "{query}");
    }
    assert_eq!(
        rows(&indexed, COMPOUND[1]),
        vec!["[Int64(4)]", "[Int64(500)]", "[Int64(88)]"],
    );
    // Without a label the `Other` nodes count too.
    assert_eq!(
        rows(&indexed, COMPOUND[0]),
        vec![
            "[Int64(19)]",
            "[Int64(3)]",
            "[Int64(4)]",
            "[Int64(4)]",
            "[Int64(500)]",
            "[Int64(88)]"
        ],
    );
}

#[test]
fn compound_in_lists_take_the_index_instead_of_a_scan() {
    let indexed = docs(true);
    for query in COMPOUND {
        let plan = profile(&indexed, query);
        assert!(plan.contains("NodeList"), "{query}:\n{plan}");
        assert!(!plan.contains("rows=1000"), "{query}:\n{plan}");
    }
}

/// An `IN` under `OR` or `NOT` does not narrow the rows, so it is no lookup.
#[test]
fn an_in_list_that_does_not_narrow_the_rows_scans() {
    let plain = docs(false);
    let indexed = docs(true);
    for query in [
        "MATCH (n:Doc) WHERE n.id IN $ids OR n.city = 'Prague' RETURN n.id",
        "MATCH (n:Doc) WHERE NOT n.id IN $ids AND n.id < 6 RETURN n.id",
    ] {
        assert_eq!(rows(&indexed, query), rows(&plain, query), "{query}");
        let plan = profile(&indexed, query);
        assert!(!plan.contains("NodeList"), "{query}:\n{plan}");
    }
}

#[test]
fn explain_names_the_index_of_a_compound_in_list() {
    let indexed = docs(true);
    for query in [COMPOUND[1], COMPOUND[2]] {
        let explain = indexed
            .execute_with_params(&format!("EXPLAIN {query}"), params())
            .unwrap()
            .rows()
            .iter()
            .map(|row| format!("{:?}", row[0]))
            .collect::<Vec<_>>()
            .join("\n");
        assert!(explain.contains("[index"), "{query}:\n{explain}");
        assert!(!explain.contains("[label-first]"), "{query}:\n{explain}");
    }
}
