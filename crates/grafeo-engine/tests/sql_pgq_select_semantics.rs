//! SQL/PGQ (SQL:2023 GRAPH_TABLE) semantics of a SELECT: the property map of
//! every element of a pattern filters it, and LIMIT and OFFSET cut the rows of
//! the query after GROUP BY, HAVING, DISTINCT and ORDER BY.
//!
//! Run with:
//! ```bash
//! cargo test -p grafeo-engine --features sql-pgq --test sql_pgq_select_semantics
//! ```

#![cfg(feature = "sql-pgq")]

use grafeo_common::types::Value;
use grafeo_engine::GrafeoDB;

/// Alix knows Gus (since 3) and Vincent (since 19); Gus knows Mia (since 3);
/// Vincent knows Jules (since 88); Mia knows Jules (since 3).
fn friends() -> GrafeoDB {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    session
        .execute(
            "INSERT (alix:Person {name: 'Alix', city: 'Amsterdam'}), \
                    (gus:Person {name: 'Gus', city: 'Berlin'}), \
                    (vincent:Person {name: 'Vincent', city: 'Paris'}), \
                    (mia:Person {name: 'Mia', city: 'Prague'}), \
                    (jules:Person {name: 'Jules', city: 'Amsterdam'}), \
                    (alix)-[:KNOWS {since: 3}]->(gus), \
                    (alix)-[:KNOWS {since: 19}]->(vincent), \
                    (gus)-[:KNOWS {since: 3}]->(mia), \
                    (vincent)-[:KNOWS {since: 88}]->(jules), \
                    (mia)-[:KNOWS {since: 3}]->(jules)",
        )
        .unwrap();
    db
}

/// Four people: one in Amsterdam, two in Berlin, one in Paris.
fn cities() -> GrafeoDB {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    session
        .execute(
            "INSERT (:Person {name: 'Alix', city: 'Amsterdam'}), \
                    (:Person {name: 'Gus', city: 'Berlin'}), \
                    (:Person {name: 'Vincent', city: 'Berlin'}), \
                    (:Person {name: 'Mia', city: 'Paris'})",
        )
        .unwrap();
    db
}

fn rows(db: &GrafeoDB, query: &str) -> Vec<Vec<Value>> {
    db.session()
        .execute_sql(query)
        .unwrap_or_else(|error| panic!("{query}: {error}"))
        .rows()
        .to_vec()
}

/// The first column of every row, sorted.
fn sorted_names(db: &GrafeoDB, query: &str) -> Vec<String> {
    let mut names: Vec<String> = rows(db, query)
        .into_iter()
        .map(|row| match &row[0] {
            Value::String(name) => name.to_string(),
            Value::Null => "null".to_string(),
            other => panic!("{query}: expected a name, got {other:?}"),
        })
        .collect();
    names.sort();
    names
}

fn text(value: &str) -> Value {
    Value::String(value.into())
}

// ============================================================================
// Property maps on every element of a pattern
// ============================================================================
//
// SQL/PGQ (ISO/IEC 9075-16:2023) takes its element patterns from GQL: a
// property map `{k: v}` on an element pattern is a predicate on that element
// (ISO/IEC 39075:2024 16.7 <path pattern expression>, the <element property
// specification> of an <element pattern>: `x.k = v` for each pair), whichever
// element of the path it stands on.

#[test]
fn a_property_map_on_the_second_node_filters_it() {
    let db = friends();
    assert_eq!(
        sorted_names(
            &db,
            "SELECT * FROM GRAPH_TABLE (
                MATCH (a:Person {name: 'Alix'})-[:KNOWS]->(b:Person {name: 'Gus'})
                COLUMNS (b.name AS name))"
        ),
        ["Gus"],
        "the map on b keeps only Gus, as `WHERE b.name = 'Gus'` does"
    );
    assert_eq!(
        sorted_names(
            &db,
            "SELECT * FROM GRAPH_TABLE (
                MATCH (a:Person {name: 'Alix'})-[:KNOWS]->({city: 'Paris'})
                COLUMNS (a.name AS name))"
        ),
        ["Alix"],
        "an anonymous node with a map filters too: Alix knows one person in Paris"
    );
}

#[test]
fn a_property_map_on_a_third_node_filters_it() {
    let db = friends();
    assert_eq!(
        sorted_names(
            &db,
            "SELECT * FROM GRAPH_TABLE (
                MATCH (a:Person {name: 'Alix'})-[:KNOWS]->(b)-[:KNOWS]->(c:Person {name: 'Jules'})
                COLUMNS (b.name AS via))"
        ),
        ["Vincent"],
        "only the walk through Vincent ends at Jules in two hops"
    );
}

#[test]
fn a_property_map_on_an_edge_filters_it() {
    let db = friends();
    assert_eq!(
        sorted_names(
            &db,
            "SELECT * FROM GRAPH_TABLE (
                MATCH (a:Person {name: 'Alix'})-[e:KNOWS {since: 19}]->(b)
                COLUMNS (b.name AS name))"
        ),
        ["Vincent"],
        "a named edge with a map"
    );
    assert_eq!(
        sorted_names(
            &db,
            "SELECT * FROM GRAPH_TABLE (
                MATCH (a:Person {name: 'Alix'})-[:KNOWS {since: 3}]->(b)
                COLUMNS (b.name AS name))"
        ),
        ["Gus"],
        "an anonymous edge with a map"
    );
}

#[test]
fn a_property_map_on_the_end_of_a_variable_length_edge_filters_it() {
    let db = friends();
    assert_eq!(
        sorted_names(
            &db,
            "SELECT * FROM GRAPH_TABLE (
                MATCH (a:Person {name: 'Alix'})-[:KNOWS*1..3]->(b:Person {name: 'Jules'})
                COLUMNS (b.name AS name))"
        ),
        ["Jules", "Jules"],
        "two walks reach Jules (through Vincent, and through Gus and Mia); \
         the other people Alix reaches are left out"
    );
}

#[test]
fn a_property_map_on_a_variable_length_edge_holds_for_every_edge_of_the_walk() {
    let db = friends();
    for edge in ["e:KNOWS*1..3 {since: 3}", ":KNOWS*1..3 {since: 3}"] {
        let query = format!(
            "SELECT * FROM GRAPH_TABLE (
                MATCH (a:Person {{name: 'Alix'}})-[{edge}]->(b)
                COLUMNS (b.name AS name))"
        );
        assert_eq!(
            sorted_names(&db, &query),
            ["Gus", "Jules", "Mia"],
            "{edge}: the walks of edges since 3 only (Alix, Gus, Mia, Jules); \
             the edges since 19 and 88 break the walks through Vincent"
        );
    }
}

#[test]
fn a_property_map_on_a_later_node_takes_a_parameter() {
    let db = friends();
    let params = std::collections::HashMap::from([("city".to_string(), text("Prague"))]);
    let result = db
        .session()
        .execute_sql_with_params(
            "SELECT * FROM GRAPH_TABLE (
                MATCH (a:Person {name: 'Alix'})-[:KNOWS*1..2]->(b:Person {city: $city})
                COLUMNS (b.name AS name))",
            params,
        )
        .unwrap();
    assert_eq!(result.rows(), [vec![text("Mia")]]);
}

#[test]
fn a_property_map_in_an_optional_match_filters_only_the_optional_part() {
    let db = friends();
    assert_eq!(
        sorted_names(
            &db,
            "SELECT * FROM GRAPH_TABLE (
                MATCH (a:Person {name: 'Alix'})
                OPTIONAL MATCH (a)-[:KNOWS]->(b:Person {city: 'Berlin'})
                COLUMNS (b.name AS name))"
        ),
        ["Gus"],
        "Alix knows one person in Berlin"
    );
    assert_eq!(
        sorted_names(
            &db,
            "SELECT * FROM GRAPH_TABLE (
                MATCH (a:Person {name: 'Alix'})
                OPTIONAL MATCH (a)-[:KNOWS]->(b:Person {city: 'Prague'})
                COLUMNS (b.name AS name))"
        ),
        ["null"],
        "nobody Alix knows lives in Prague: the optional part is null and Alix's row stays"
    );
}

// ============================================================================
// LIMIT and OFFSET after GROUP BY, HAVING, DISTINCT and ORDER BY
// ============================================================================
//
// ISO/IEC 9075-2 <query expression>: the <order by clause>, <result offset
// clause> and <fetch first clause> (LIMIT) apply to the rows of the <query
// specification>, which are the rows after its FROM, WHERE, GROUP BY, HAVING
// and SELECT (with DISTINCT). Cutting the GRAPH_TABLE rows first splits
// groups and drops rows that DISTINCT would have kept.

#[test]
fn limit_cuts_the_groups_not_the_rows_they_are_built_from() {
    let db = cities();
    assert_eq!(
        rows(
            &db,
            "SELECT city, COUNT(*) AS n FROM GRAPH_TABLE (MATCH (p:Person) COLUMNS (p.city AS city))
             GROUP BY city ORDER BY city LIMIT 3"
        ),
        [
            vec![text("Amsterdam"), Value::Int64(1)],
            vec![text("Berlin"), Value::Int64(2)],
            vec![text("Paris"), Value::Int64(1)],
        ]
    );
    assert_eq!(
        rows(
            &db,
            "SELECT city, COUNT(*) AS n FROM GRAPH_TABLE (MATCH (p:Person) COLUMNS (p.city AS city))
             GROUP BY city ORDER BY n DESC LIMIT 1"
        ),
        [vec![text("Berlin"), Value::Int64(2)]],
        "the largest group, counted over every row"
    );
}

#[test]
fn offset_skips_groups_not_the_rows_they_are_built_from() {
    let db = cities();
    assert_eq!(
        rows(
            &db,
            "SELECT city, COUNT(*) AS n FROM GRAPH_TABLE (MATCH (p:Person) COLUMNS (p.city AS city))
             GROUP BY city ORDER BY city LIMIT 2 OFFSET 1"
        ),
        [
            vec![text("Berlin"), Value::Int64(2)],
            vec![text("Paris"), Value::Int64(1)],
        ]
    );
}

#[test]
fn limit_after_having_keeps_the_groups_that_pass_it() {
    let db = cities();
    assert_eq!(
        rows(
            &db,
            "SELECT city, COUNT(*) AS n FROM GRAPH_TABLE (MATCH (p:Person) COLUMNS (p.city AS city))
             GROUP BY city HAVING COUNT(*) > 1 ORDER BY city LIMIT 1"
        ),
        [vec![text("Berlin"), Value::Int64(2)]]
    );
}

#[test]
fn limit_on_an_aggregate_without_group_by_counts_every_row() {
    let db = cities();
    assert_eq!(
        rows(
            &db,
            "SELECT COUNT(*) AS n FROM GRAPH_TABLE (MATCH (p:Person) COLUMNS (p.city AS city)) LIMIT 1"
        ),
        [vec![Value::Int64(4)]],
        "the one row of the aggregate counts all four people"
    );
}

#[test]
fn limit_cuts_the_distinct_rows() {
    let db = cities();
    assert_eq!(
        rows(
            &db,
            "SELECT DISTINCT city FROM GRAPH_TABLE (MATCH (p:Person) COLUMNS (p.city AS city))
             ORDER BY city LIMIT 3"
        ),
        [
            vec![text("Amsterdam")],
            vec![text("Berlin")],
            vec![text("Paris")]
        ]
    );
    assert_eq!(
        rows(
            &db,
            "SELECT DISTINCT city FROM GRAPH_TABLE (MATCH (p:Person) COLUMNS (p.city AS city))
             ORDER BY city DESC LIMIT 2 OFFSET 1"
        ),
        [vec![text("Berlin")], vec![text("Amsterdam")]],
        "OFFSET skips distinct rows too"
    );
}

#[test]
fn limit_and_offset_without_grouping_keep_their_order() {
    let db = cities();
    assert_eq!(
        rows(
            &db,
            "SELECT name FROM GRAPH_TABLE (MATCH (p:Person) COLUMNS (p.name AS name))
             ORDER BY name LIMIT 2 OFFSET 1"
        ),
        [vec![text("Gus")], vec![text("Mia")]]
    );
}

#[test]
fn limit_counts_distinct_rows_of_walks_that_reach_a_node_twice() {
    // Two walks reach Jules: DISTINCT keeps one row for him, and LIMIT
    // counts that row once.
    let db = friends();
    assert_eq!(
        rows(
            &db,
            "SELECT DISTINCT name FROM GRAPH_TABLE (
                MATCH (a:Person {name: 'Alix'})-[:KNOWS*1..3]->(b)
                COLUMNS (b.name AS name))
             ORDER BY name LIMIT 4"
        ),
        [
            vec![text("Gus")],
            vec![text("Jules")],
            vec![text("Mia")],
            vec![text("Vincent")],
        ]
    );
}
