//! A pattern that names an edge bound before matches that edge only: the
//! rows of the same pattern with a fresh edge filtered to equal it.
//!
//! ```bash
//! cargo test -p grafeo-engine --all-features --test bound_edges
//! ```

use grafeo_common::types::Value;
use grafeo_engine::GrafeoDB;

/// Alix, Gus, Vincent and Mia, and Amsterdam. Edges with `w` 1 to 6: KNOWS
/// Alix->Gus (1), Gus->Vincent (2), Vincent->Alix (3), a second KNOWS
/// Alix->Gus (4), LIVES_IN Alix->Amsterdam (5) and a LIKES loop on Mia (6).
/// The parallel edges tell a check on the edge from one on its endpoints.
fn graph() -> GrafeoDB {
    let db = GrafeoDB::new_in_memory();
    db.execute(
        "INSERT (alix:Person {name: 'Alix'}), (gus:Person {name: 'Gus'}), \
         (vincent:Person {name: 'Vincent'}), (mia:Person {name: 'Mia'}), \
         (ams:City {name: 'Amsterdam'}), \
         (alix)-[:KNOWS {w: 1}]->(gus), (gus)-[:KNOWS {w: 2}]->(vincent), \
         (vincent)-[:KNOWS {w: 3}]->(alix), (alix)-[:KNOWS {w: 4}]->(gus), \
         (alix)-[:LIVES_IN {w: 5}]->(ams), (mia)-[:LIKES {w: 6}]->(mia)",
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

#[test]
fn a_later_match_through_a_bound_edge_matches_that_edge() {
    let db = graph();
    assert_both(
        &db,
        "MATCH ()-[r]->() MATCH (x)-[r]->(y) RETURN r.w, x.name, y.name",
        &[
            &["1", "Alix", "Gus"],
            &["2", "Gus", "Vincent"],
            &["3", "Vincent", "Alix"],
            &["4", "Alix", "Gus"],
            &["5", "Alix", "Amsterdam"],
            &["6", "Mia", "Mia"],
        ],
    );
    assert_both(
        &db,
        "MATCH ()-[r]->() MATCH ()-[r]->() RETURN count(*)",
        &[&["6"]],
    );
}

/// With both ends bound too, the parallel KNOWS edges Alix->Gus each match
/// once, not twice.
#[test]
fn a_bound_edge_between_bound_nodes_matches_once() {
    let db = graph();
    assert_both(
        &db,
        "MATCH (a)-[r]->(b) MATCH (a)-[r]->(b) RETURN r.w",
        &[&["1"], &["2"], &["3"], &["4"], &["5"], &["6"]],
    );
    assert_both(
        &db,
        "MATCH (a)-[r]->(b) MATCH (a)-[r]->(c) RETURN r.w, c.name",
        &[
            &["1", "Gus"],
            &["2", "Vincent"],
            &["3", "Alix"],
            &["4", "Gus"],
            &["5", "Amsterdam"],
            &["6", "Mia"],
        ],
    );
}

/// The bound edge keeps its direction: read backwards its ends swap, and
/// from its own source backwards only the loop matches.
#[test]
fn a_bound_edge_keeps_its_direction() {
    let db = graph();
    assert_both(
        &db,
        "MATCH (a)-[r]->(b) MATCH (x)<-[r]-(y) RETURN r.w, x.name, y.name",
        &[
            &["1", "Gus", "Alix"],
            &["2", "Vincent", "Gus"],
            &["3", "Alix", "Vincent"],
            &["4", "Gus", "Alix"],
            &["5", "Amsterdam", "Alix"],
            &["6", "Mia", "Mia"],
        ],
    );
    assert_both(
        &db,
        "MATCH (a)-[r]->(b) MATCH (a)<-[r]-(b) RETURN r.w",
        &[&["6"]],
    );
    assert_both(
        &db,
        "MATCH (a)-[r:KNOWS {w: 2}]->(b) MATCH (x)-[r]-(y) RETURN x.name, y.name",
        &[&["Gus", "Vincent"], &["Vincent", "Gus"]],
    );
}

/// A type or a property map on the later pattern applies to the bound edge.
#[test]
fn a_type_or_property_on_the_later_pattern_checks_the_bound_edge() {
    let db = graph();
    assert_both(
        &db,
        "MATCH ()-[r:KNOWS]->() MATCH ()-[r:LIVES_IN]->() RETURN count(*)",
        &[&["0"]],
    );
    assert_both(
        &db,
        "MATCH ()-[r]->() MATCH ()-[r:KNOWS]->() RETURN r.w",
        &[&["1"], &["2"], &["3"], &["4"]],
    );
    assert_both(
        &db,
        "MATCH ()-[r]->() MATCH (x)-[r {w: 3}]->(y) RETURN x.name, y.name",
        &[&["Vincent", "Alix"]],
    );
}

/// The same edge twice in one path can only be a loop.
#[test]
fn the_same_edge_twice_in_a_path_is_a_loop() {
    let db = graph();
    assert_eq!(
        rows(db.execute("MATCH (a)-[r]->(b)-[r]->(c) RETURN a.name, b.name, c.name")),
        expected(&[&["Mia", "Mia", "Mia"]])
    );
}

/// A named path through the bound edge has that edge only.
#[test]
fn a_named_path_through_a_bound_edge() {
    let db = graph();
    assert_both(
        &db,
        "MATCH ()-[r:KNOWS]->() MATCH p = (x)-[r]->(y) RETURN r.w, length(p)",
        &[&["1", "1"], &["2", "1"], &["3", "1"], &["4", "1"]],
    );
}

/// A variable-length pattern that names the edge list of an earlier one
/// matches that list: each two-hop KNOWS walk once.
#[test]
fn a_bound_edge_list_matches_the_same_walk() {
    let db = graph();
    assert_both(&db, "MATCH ()-[r:KNOWS*2]->() RETURN count(*)", &[&["5"]]);
    assert_both(
        &db,
        "MATCH ()-[r:KNOWS*2]->() MATCH (x)-[r:KNOWS*2]->(y) RETURN count(*)",
        &[&["5"]],
    );
}

/// The edge stays bound through WITH, ORDER BY and LIMIT.
#[cfg(feature = "cypher")]
#[test]
fn a_bound_edge_after_an_ordered_cut() {
    let db = graph();
    assert_eq!(
        rows(db.execute_cypher(
            "MATCH ()-[r]->() WITH r ORDER BY r.w LIMIT 2 MATCH (x)-[r]->(y) \
             RETURN r.w, x.name, y.name"
        )),
        expected(&[&["1", "Alix", "Gus"], &["2", "Gus", "Vincent"]])
    );
}

/// A subquery that imports the edge, by name or with `WITH *`, matches it.
#[test]
fn a_subquery_matches_the_edge_it_imports() {
    let db = graph();
    let want: &[&[&str]] = &[
        &["1", "Gus"],
        &["2", "Vincent"],
        &["3", "Alix"],
        &["4", "Gus"],
        &["5", "Amsterdam"],
        &["6", "Mia"],
    ];
    assert_both(
        &db,
        "MATCH ()-[r]->() CALL { WITH r MATCH (x)-[r]->(y) RETURN y.name AS target } \
         RETURN r.w, target",
        want,
    );
    assert_both(
        &db,
        "MATCH ()-[r]->() CALL { WITH * MATCH (x)-[r]->(y) RETURN y.name AS target } \
         RETURN r.w, target",
        want,
    );
}

/// The edge stays bound across a subquery that returns other columns.
#[test]
fn a_bound_edge_after_a_subquery() {
    let db = graph();
    assert_both(
        &db,
        "MATCH ()-[r]->() CALL { RETURN 1 AS one } MATCH (x)-[r]->(y) RETURN count(*)",
        &[&["6"]],
    );
}

/// OPTIONAL MATCH joins on the bound edge: the KNOWS edges find themselves,
/// the others none.
#[test]
fn an_optional_match_through_a_bound_edge() {
    let db = graph();
    assert_both(
        &db,
        "MATCH ()-[r]->() OPTIONAL MATCH (x)-[r:KNOWS]->(y) RETURN r.w, y.name",
        &[
            &["1", "Gus"],
            &["2", "Vincent"],
            &["3", "Alix"],
            &["4", "Gus"],
            &["5", "null"],
            &["6", "null"],
        ],
    );
}
