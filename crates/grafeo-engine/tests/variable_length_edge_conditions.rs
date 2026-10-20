//! A condition on a variable-length edge holds for each edge of the path: an
//! inline `WHERE` (Cypher, as Neo4j 5 reads it, and GQL) and a property map
//! (both) check every hop, and may read the row's other variables. Cypher
//! checked its inline `WHERE` against the list of the path's edges, so a
//! variable-length edge with one returned no rows. GQL's ISO name of
//! `collect`, `collect_list`, was not an aggregate.
//!
//! ```bash
//! cargo test -p grafeo-engine --all-features --test variable_length_edge_conditions
//! ```

#![cfg(all(feature = "lpg", feature = "gql", feature = "cypher"))]

use grafeo_common::types::Value;
use grafeo_engine::GrafeoDB;

const LANGUAGES: [&str; 2] = ["gql", "cypher"];

/// A chain 0 -> 1 -> 2 -> 3 of KNOWS edges weighing 20, 30 and 5.
fn chain() -> GrafeoDB {
    let db = GrafeoDB::new_in_memory();
    db.execute(
        "INSERT (:P {i: 0})-[:KNOWS {w: 20}]->(:P {i: 1})-[:KNOWS {w: 30}]->(:P {i: 2})\
         -[:KNOWS {w: 5}]->(:P {i: 3})",
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

fn ints(values: &[i64]) -> Vec<Vec<Value>> {
    values.iter().map(|&v| vec![Value::Int64(v)]).collect()
}

/// Every edge of the path passes the inline `WHERE`: 1 and 2 are reached
/// over edges heavier than 10, 3 is not.
#[test]
fn an_inline_where_holds_for_every_edge() {
    let db = chain();
    for language in LANGUAGES {
        assert_eq!(
            rows(
                &db,
                language,
                "MATCH (a:P {i: 0})-[e:KNOWS*1..3 WHERE e.w > 10]->(b) RETURN b.i AS i ORDER BY i"
            ),
            ints(&[1, 2]),
            "{language}"
        );
    }
}

/// The condition may read a variable of the row.
#[test]
fn an_inline_where_reads_the_row() {
    let db = chain();
    for language in LANGUAGES {
        assert_eq!(
            rows(
                &db,
                language,
                "UNWIND [15, 25] AS t MATCH (a:P {i: 0})-[e:KNOWS*1..3 WHERE e.w > t]->(b) \
                 RETURN t, b.i AS i ORDER BY t, i"
            ),
            [
                [Value::Int64(15), Value::Int64(1)],
                [Value::Int64(15), Value::Int64(2)],
            ],
            "{language}: no edge from 0 weighs more than 25"
        );
    }
}

/// After the pattern the edge variable is the list of the path's edges.
#[test]
fn the_edge_variable_is_the_list_after_the_pattern() {
    let db = chain();
    for language in LANGUAGES {
        assert_eq!(
            rows(
                &db,
                language,
                "MATCH (a:P {i: 0})-[e:KNOWS*1..3 WHERE e.w > 10]->(b) \
                 RETURN size(e) AS hops ORDER BY hops"
            ),
            ints(&[1, 2]),
            "{language}"
        );
    }
}

/// A property map checks each edge too, beside an inline `WHERE`.
#[test]
fn a_property_map_and_an_inline_where_together() {
    let db = chain();
    db.execute("MATCH (a:P {i: 0})-[r:KNOWS]->() SET r.kind = 'x'")
        .unwrap();
    for language in LANGUAGES {
        assert_eq!(
            rows(
                &db,
                language,
                "MATCH (a:P {i: 0})-[e:KNOWS*1..3 {kind: 'x'} WHERE e.w > 10]->(b) \
                 RETURN b.i AS i"
            ),
            ints(&[1]),
            "{language}: only the first edge has kind x"
        );
    }
}

/// A single-hop inline `WHERE` reads the one edge, as before.
#[test]
fn a_single_hop_inline_where_reads_the_edge() {
    let db = chain();
    for language in LANGUAGES {
        assert_eq!(
            rows(
                &db,
                language,
                "MATCH (a)-[e:KNOWS WHERE e.w < 25]->(b) RETURN a.i AS i ORDER BY i"
            ),
            ints(&[0, 2]),
            "{language}"
        );
    }
}

/// `collect_list` is `collect` under its ISO GQL name.
#[test]
fn collect_list_is_collect() {
    let db = GrafeoDB::new_in_memory();
    assert_eq!(
        rows(
            &db,
            "gql",
            "UNWIND [3, 19] AS x RETURN collect_list(x) AS l"
        ),
        [[Value::List(vec![Value::Int64(3), Value::Int64(19)].into())]]
    );
}

/// Ten nodes, every pair joined both ways by a light edge (`w` 1), and a
/// chain 0 -> 1 -> ... -> 9 of heavy ones (`w` 10).
fn dense() -> GrafeoDB {
    let db = GrafeoDB::new_in_memory();
    db.execute("UNWIND range(0, 9) AS i INSERT (:D {i: i})")
        .unwrap();
    db.execute("MATCH (a:D), (b:D) WHERE a.i <> b.i INSERT (a)-[:R {w: 1}]->(b)")
        .unwrap();
    db.execute("MATCH (a:D), (b:D) WHERE b.i = a.i + 1 INSERT (a)-[:R {w: 10}]->(b)")
        .unwrap();
    db
}

/// The condition is checked during the search: of the 10^8 or so walks of up
/// to eight light or heavy edges from 0, only the heavy chain is walked, so
/// the search stays far inside its budget (enumerating every walk first and
/// filtering them after ran out of it).
#[test]
fn the_condition_prunes_the_search() {
    let db = dense();
    for language in LANGUAGES {
        for query in [
            "MATCH (a:D {i: 0})-[e:R*1..8 WHERE e.w > 5]->(b) RETURN b.i AS i ORDER BY i",
            "MATCH (a:D {i: 0})-[e:R*1..8 {w: 10}]->(b) RETURN b.i AS i ORDER BY i",
        ] {
            assert_eq!(
                rows(&db, language, query),
                ints(&[1, 2, 3, 4, 5, 6, 7, 8]),
                "{language} `{query}`"
            );
        }
    }
}

/// A condition that reads more than the edge and the row (here the path's
/// end node) cannot be checked during the search: it holds for the paths
/// found, as before.
#[test]
fn a_condition_on_the_end_node_still_filters() {
    let db = chain();
    for language in LANGUAGES {
        assert_eq!(
            rows(
                &db,
                language,
                "MATCH p = (a:P {i: 0})-[:KNOWS*1..3]->(b) \
                 WHERE all(r IN edges(p) WHERE r.w > b.i * 2) RETURN b.i AS i ORDER BY i"
            ),
            ints(&[1, 2]),
            "{language}"
        );
    }
}

/// A pattern from a node an OPTIONAL MATCH did not find matches nothing (it
/// was an internal error).
#[test]
fn a_variable_length_pattern_from_a_null_node_matches_nothing() {
    let db = chain();
    for language in LANGUAGES {
        assert_eq!(
            rows(
                &db,
                language,
                "OPTIONAL MATCH (n:Missing) MATCH (n)-[:KNOWS*1..3]->(m) RETURN count(m) AS c"
            ),
            ints(&[0]),
            "{language}"
        );
    }
}
