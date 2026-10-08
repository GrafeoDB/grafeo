//! A later MATCH whose comma-separated pattern parts reuse a variable an
//! earlier clause bound joins on that variable: the part that reuses it
//! expands from the bound node or edge, in whatever order the parts come, so
//! `MATCH (a:Person) MATCH (b:City), (a)-[:VISITED]->(b)` returns the visits
//! of each person, not every person with every visit.
//!
//! ```bash
//! cargo test -p grafeo-engine --all-features --test later_match_comma_patterns
//! ```

use grafeo_common::types::Value;
use grafeo_engine::GrafeoDB;

/// People Alix (`k` 1), Gus (3), Vincent (19) and Mia (88), and the cities
/// Amsterdam, Berlin and Paris. VISITED: Alix to Amsterdam (`w` 3), Gus to
/// Berlin (19) and to Amsterdam (88), Vincent to Paris (33) and to Mia (38),
/// who is no city. Mia visited nothing.
fn graph() -> GrafeoDB {
    let db = GrafeoDB::new_in_memory();
    db.execute(
        "INSERT (alix:Person {name: 'Alix', k: 1}), (gus:Person {name: 'Gus', k: 3}), \
         (vincent:Person {name: 'Vincent', k: 19}), (mia:Person {name: 'Mia', k: 88}), \
         (ams:City {name: 'Amsterdam'}), (ber:City {name: 'Berlin'}), \
         (par:City {name: 'Paris'}), \
         (alix)-[:VISITED {w: 3}]->(ams), (gus)-[:VISITED {w: 19}]->(ber), \
         (gus)-[:VISITED {w: 88}]->(ams), (vincent)-[:VISITED {w: 33}]->(par), \
         (vincent)-[:VISITED {w: 38}]->(mia)",
    )
    .unwrap();
    db
}

/// The rows of `result` as text, sorted.
fn rows(
    result: grafeo_common::utils::error::Result<grafeo_engine::database::QueryResult>,
) -> Vec<Vec<String>> {
    let mut rows: Vec<Vec<String>> = result
        .unwrap()
        .rows()
        .iter()
        .map(|row| {
            row.iter()
                .map(|value| match value {
                    Value::String(text) => text.to_string(),
                    Value::Int64(number) => number.to_string(),
                    Value::Null => "null".to_string(),
                    other => panic!("unexpected value {other:?}"),
                })
                .collect()
        })
        .collect();
    rows.sort();
    rows
}

/// `rows` for literal text cells.
fn expected(rows: &[&[&str]]) -> Vec<Vec<String>> {
    let mut rows: Vec<Vec<String>> = rows
        .iter()
        .map(|row| row.iter().map(|cell| (*cell).to_string()).collect())
        .collect();
    rows.sort();
    rows
}

/// Runs `query` as GQL and, with the `cypher` feature, as Cypher, and checks
/// both against `want`.
fn assert_both(db: &GrafeoDB, query: &str, want: &[&[&str]]) {
    assert_eq!(rows(db.execute(query)), expected(want), "GQL: {query}");
    #[cfg(feature = "cypher")]
    assert_eq!(
        rows(db.execute_cypher(query)),
        expected(want),
        "Cypher: {query}"
    );
}

/// The visits to cities, one row per VISITED edge to a city.
const VISITS: &[&[&str]] = &[
    &["Alix", "Amsterdam"],
    &["Gus", "Amsterdam"],
    &["Gus", "Berlin"],
    &["Vincent", "Paris"],
];

#[test]
fn a_later_pattern_part_expands_from_the_bound_node() {
    let db = graph();
    assert_both(
        &db,
        "MATCH (a:Person) MATCH (b:City), (a)-[:VISITED]->(b) RETURN a.name, b.name",
        VISITS,
    );
    assert_both(
        &db,
        "MATCH (a:Person) MATCH (b:City), (a)-[:VISITED]->(b) RETURN count(*)",
        &[&["4"]],
    );
}

/// The `EXPLAIN` plans of `query` as text: in GQL and, with the `cypher`
/// feature, in Cypher, each with the name of its language.
fn explain(db: &GrafeoDB, query: &str) -> Vec<(&'static str, String)> {
    let query = format!("EXPLAIN {query}");
    let languages: &[&'static str] = if cfg!(feature = "cypher") {
        &["GQL", "Cypher"]
    } else {
        &["GQL"]
    };
    languages
        .iter()
        .map(|&language| {
            let result = match language {
                #[cfg(feature = "cypher")]
                "Cypher" => db.execute_cypher(&query),
                _ => db.execute(&query),
            };
            let plan = result
                .unwrap()
                .rows()
                .iter()
                .map(|row| match &row[0] {
                    Value::String(text) => text.to_string(),
                    other => format!("{other:?}"),
                })
                .collect::<Vec<_>>()
                .join(
                    "
",
                );
            (language, plan)
        })
        .collect()
}

/// The part that starts from the bound node expands from it: no second scan
/// of people joined to the first, but an expand checked against the bound
/// city. A part that starts elsewhere is joined on every variable it shares.
#[test]
fn the_plan_expands_from_the_bound_node() {
    let db = graph();
    let query = "MATCH (a:Person) MATCH (b:City), (a)-[:VISITED]->(b) RETURN a.name, b.name";
    for (language, plan) in explain(&db, query) {
        assert!(
            plan.contains("Expand (a)->[:VISITED]->") && !plan.contains("Join"),
            "{language}: the part does not expand from the bound a:
{plan}"
        );
    }
    let query = "MATCH (a:Person) MATCH (b:City), (b)<-[:VISITED]-(a) RETURN a.name, b.name";
    for (language, plan) in explain(&db, query) {
        assert!(
            plan.contains("Join (Inner)"),
            "{language}: the part from b is not joined:
{plan}"
        );
    }
}

/// The parts may come in any order, and the bound node may be at either end
/// of the part that reuses it.
#[test]
fn the_order_of_the_parts_does_not_matter() {
    let db = graph();
    for query in [
        "MATCH (a:Person) MATCH (a)-[:VISITED]->(b), (b:City) RETURN a.name, b.name",
        "MATCH (a:Person) MATCH (b:City), (b)<-[:VISITED]-(a) RETURN a.name, b.name",
        "MATCH (a:Person) MATCH (b)<-[:VISITED]-(a), (b:City) RETURN a.name, b.name",
        "MATCH (a:Person) MATCH (c:City {name: 'Paris'}), (b:City), (a)-[:VISITED]->(b) \
         RETURN a.name, b.name",
        "MATCH (a:Person) MATCH (b:City), (c:City {name: 'Paris'}), (b)<-[:VISITED]-(a) \
         RETURN a.name, b.name",
    ] {
        assert_both(&db, query, VISITS);
    }
}

/// Both ends bound by earlier clauses: the part matches the edges between
/// them only.
#[test]
fn a_part_between_two_nodes_bound_before_matches_their_edges() {
    let db = graph();
    assert_both(
        &db,
        "MATCH (a:Person) MATCH (b:City) MATCH (c:City {name: 'Paris'}), (a)-[:VISITED]->(b) \
         RETURN a.name, b.name",
        VISITS,
    );
    assert_both(
        &db,
        "MATCH (a:Person) MATCH (b:City) MATCH (c:City {name: 'Paris'}), (b)<-[:VISITED]-(a) \
         RETURN a.name, b.name",
        VISITS,
    );
}

/// A WHERE on the bound node, after the later MATCH or on the first one (in
/// GQL inside its pattern: a WHERE between two MATCH clauses is Cypher).
#[test]
fn a_where_on_the_bound_node_keeps_its_visits() {
    let db = graph();
    assert_both(
        &db,
        "MATCH (a:Person) MATCH (b:City), (a)-[:VISITED]->(b) WHERE a.k = 1 \
         RETURN a.name, b.name",
        &[&["Alix", "Amsterdam"]],
    );
    let gus = expected(&[&["Gus", "Amsterdam"], &["Gus", "Berlin"]]);
    let query = "MATCH (a:Person WHERE a.k = 3) MATCH (b:City), (a)-[:VISITED]->(b) \
                 RETURN a.name, b.name";
    assert_eq!(rows(db.execute(query)), gus, "GQL: {query}");
    #[cfg(feature = "cypher")]
    {
        let query = "MATCH (a:Person) WHERE a.k = 3 MATCH (b:City), (a)-[:VISITED]->(b) \
                     RETURN a.name, b.name";
        assert_eq!(rows(db.execute_cypher(query)), gus, "Cypher: {query}");
    }
    assert_both(
        &db,
        "MATCH (a:Person) MATCH (b:City), (a)-[:VISITED]->(b) WHERE b.name = 'Amsterdam' \
         RETURN a.name",
        &[&["Alix"], &["Gus"]],
    );
}

/// OPTIONAL MATCH: a person without a visit to a city gets one row of nulls.
#[test]
fn an_optional_later_match_joins_on_the_bound_node() {
    let db = graph();
    let mut want = VISITS.to_vec();
    want.push(&["Mia", "null"]);
    assert_both(
        &db,
        "MATCH (a:Person) OPTIONAL MATCH (b:City), (a)-[:VISITED]->(b) RETURN a.name, b.name",
        &want,
    );
    assert_both(
        &db,
        "MATCH (a:Person) OPTIONAL MATCH (b:City), (b)<-[:VISITED]-(a) RETURN a.name, b.name",
        &want,
    );
}

/// An edge bound before: the part through it matches that edge only.
#[test]
fn a_later_part_through_a_bound_edge_matches_that_edge() {
    let db = graph();
    assert_both(
        &db,
        "MATCH (a:Person)-[r:VISITED]->(c:City) MATCH (b:City), (b)<-[r]-(x) \
         RETURN x.name, b.name",
        VISITS,
    );
    assert_both(
        &db,
        "MATCH (a:Person)-[r:VISITED]->(c:City) MATCH (b:City), (x)-[r]->(b) \
         RETURN x.name, b.name",
        VISITS,
    );
}

/// Values of the input rows stay what they were.
#[test]
fn the_input_rows_keep_their_values() {
    let db = graph();
    assert_both(
        &db,
        "UNWIND [1, 3] AS k MATCH (a:Person {k: k}) MATCH (b:City), (a)-[:VISITED]->(b) \
         RETURN k, a.name, b.name",
        &[
            &["1", "Alix", "Amsterdam"],
            &["3", "Gus", "Amsterdam"],
            &["3", "Gus", "Berlin"],
        ],
    );
    // The part reads an unwound value: it goes on from the row that holds it
    assert_both(
        &db,
        "UNWIND [3, 19] AS w MATCH (a:Person) MATCH (b:City), (a)-[:VISITED {w: w}]->(b) \
         RETURN w, a.name, b.name",
        &[&["19", "Gus", "Berlin"], &["3", "Alix", "Amsterdam"]],
    );
    assert_both(
        &db,
        "MATCH (a:Person) MATCH (b:City), (a)-[r:VISITED]->(b) RETURN a.name, r.w",
        &[
            &["Alix", "3"],
            &["Gus", "19"],
            &["Gus", "88"],
            &["Vincent", "33"],
        ],
    );
}

/// The MATCH of a CALL subquery goes on from the node the subquery imports.
#[test]
fn a_subquery_pattern_part_expands_from_the_imported_node() {
    let db = graph();
    assert_both(
        &db,
        "MATCH (a:Person) CALL { WITH a MATCH (b:City), (a)-[:VISITED]->(b) \
         RETURN b.name AS city } RETURN a.name, city",
        VISITS,
    );
    assert_both(
        &db,
        "MATCH (a:Person) CALL { WITH * MATCH (b:City), (a)-[:VISITED]->(b) \
         RETURN b.name AS city } RETURN a.name, city",
        VISITS,
    );
}
