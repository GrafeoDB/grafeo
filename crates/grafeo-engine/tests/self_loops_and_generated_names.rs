//! An undirected pattern takes a self-loop once, and the names the engine
//! makes up never take the place of a user's variable.
//!
//! A self-loop is in both adjacency lists of its node, and a walk in both
//! directions went through it twice: `MATCH (n)-[r]-(n)` returned Mia twice,
//! in GQL and Cypher (openCypher TCK Match3 [11] and [12] expect one row).
//! GQL numbered its aggregate columns `_agg_0`, `_agg_1`, ... whatever the
//! statement's own variables were called, and the planner named the columns
//! of anonymous edges `_anon_edge_0`, ..., so a user variable with such a name
//! read the engine's column.
//!
//! ```bash
//! cargo test -p grafeo-engine --all-features --test self_loops_and_generated_names
//! ```

#![cfg(all(feature = "lpg", feature = "gql", feature = "cypher"))]

use grafeo_common::types::Value;
use grafeo_engine::GrafeoDB;

/// Mia knows herself and Gus; Gus knows Alix.
fn self_loop() -> GrafeoDB {
    let db = GrafeoDB::new_in_memory();
    db.execute(
        "INSERT (m:Person {name: 'Mia'})-[:KNOWS {since: 1}]->(m), \
         (m)-[:KNOWS {since: 2}]->(g:Person {name: 'Gus'}), \
         (g)-[:KNOWS {since: 3}]->(:Person {name: 'Alix'})",
    )
    .unwrap();
    db
}

fn rows(db: &GrafeoDB, language: &str, query: &str) -> Vec<Vec<Value>> {
    db.execute_language(query, language, None)
        .unwrap_or_else(|e| panic!("{language} `{query}`: {e}"))
        .rows()
        .to_vec()
}

#[test]
fn an_undirected_pattern_takes_a_self_loop_once() {
    let db = self_loop();
    for language in ["gql", "cypher"] {
        assert_eq!(
            rows(&db, language, "MATCH (n)-[r]-(n) RETURN n.name AS name"),
            [[Value::from("Mia")]],
            "{language}: the loop, once"
        );
        // Mia's edges seen from her: the loop once, and the edge to Gus.
        assert_eq!(
            rows(
                &db,
                language,
                "MATCH (m:Person {name: 'Mia'})-[r]-(x) RETURN r.since AS since ORDER BY since"
            ),
            [[Value::Int64(1)], [Value::Int64(2)]],
            "{language}: each of Mia's edges once"
        );
        // Every edge from both of its ends, the loop from its one end.
        assert_eq!(
            rows(&db, language, "MATCH (a)-[r]-(b) RETURN count(r) AS c"),
            [[Value::Int64(5)]],
            "{language}: 2 + 2 + 1 directed walks"
        );
    }
}

/// The variable-length expand, the shortest path and the EXISTS and COUNT
/// edge checks walk it once too.
#[test]
fn every_walk_takes_a_self_loop_once() {
    let db = self_loop();
    for language in ["gql", "cypher"] {
        assert_eq!(
            rows(
                &db,
                language,
                "MATCH (m:Person {name: 'Mia'})-[*1..1]-(x) RETURN x.name AS name ORDER BY name"
            ),
            [[Value::from("Gus")], [Value::from("Mia")]],
            "{language}: one hop, the loop once"
        );
        assert_eq!(
            rows(
                &db,
                language,
                "MATCH (m:Person {name: 'Mia'}) RETURN COUNT { MATCH (m)-[]-(m) } AS c"
            ),
            [[Value::Int64(1)]],
            "{language}: COUNT of the loop"
        );
    }
}

/// A user variable named like an aggregate column keeps its values.
#[test]
fn a_variable_named_like_an_aggregate_column_keeps_its_values() {
    let db = GrafeoDB::new_in_memory();
    for language in ["gql", "cypher"] {
        assert_eq!(
            rows(
                &db,
                language,
                "UNWIND [3, 19] AS _agg_0 RETURN _agg_0, count(*) > 0 AS any ORDER BY _agg_0"
            ),
            [
                [Value::Int64(3), Value::Bool(true)],
                [Value::Int64(19), Value::Bool(true)]
            ],
            "{language}"
        );
        assert_eq!(
            rows(
                &db,
                language,
                "UNWIND [3, 19] AS _agg_1 RETURN _agg_1, count(*) AS c, sum(_agg_1) AS s \
                 ORDER BY _agg_1"
            ),
            [
                [Value::Int64(3), Value::Int64(1), Value::Int64(3)],
                [Value::Int64(19), Value::Int64(1), Value::Int64(19)]
            ],
            "{language}"
        );
    }
}

/// A user variable named like an anonymous edge's column keeps its node.
#[test]
fn a_variable_named_like_an_anonymous_edge_keeps_its_node() {
    let db = self_loop();
    for language in ["gql", "cypher"] {
        assert_eq!(
            rows(
                &db,
                language,
                "MATCH (_anon_edge_0:Person {name: 'Gus'}), (a)-[]->(b) \
                 RETURN _anon_edge_0.name AS n, count(*) AS c"
            ),
            [[Value::from("Gus"), Value::Int64(3)]],
            "{language}"
        );
    }
}
