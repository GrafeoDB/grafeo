//! Nodes and edges a statement returns stay records through everything that
//! only orders, cuts, deduplicates or combines rows: `ORDER BY`, `LIMIT`,
//! `SKIP`, `DISTINCT`, `UNION`, `OTHERWISE`, `EXCEPT` and `INTERSECT`, in GQL
//! and Cypher. Edges used to come back as `0` after `ORDER BY`, `SKIP` or a
//! cut `LIMIT` (#482), and a later `UNION` or `OTHERWISE` branch returned raw
//! IDs, so `EXCEPT` and `INTERSECT` compared records with IDs.
//!
//! ```bash
//! cargo test -p grafeo-engine --all-features --test returned_entities
//! ```

#![cfg(all(feature = "lpg", feature = "gql"))]

use grafeo_common::types::{PropertyKey, Value};
use grafeo_engine::{Config, GrafeoDB};

/// `(:A {n: 1})-[:K {w: 1}]->(:B {n: 2})`, and the same with n 3, 4 and w 2,
/// and n 5, 6 and w 3.
fn three_edges(config: Config) -> GrafeoDB {
    let db = GrafeoDB::with_config(config).unwrap();
    for (a, w, b) in [(1, 1, 2), (3, 2, 4), (5, 3, 6)] {
        db.execute(&format!(
            "INSERT (:A {{n: {a}}})-[:K {{w: {w}}}]->(:B {{n: {b}}})"
        ))
        .unwrap();
    }
    db
}

/// A short form of a returned value: `K w=1` for an edge record, `n=1` for
/// a node record, the debug form for anything else.
fn describe(value: &Value) -> String {
    match value {
        Value::Map(map) => {
            let get = |key: &str| map.get(&PropertyKey::new(key));
            match (get("_type"), get("_labels")) {
                (Some(Value::String(edge_type)), _) => {
                    format!(
                        "{edge_type} w={:?}",
                        get("w").cloned().unwrap_or(Value::Null)
                    )
                }
                (None, Some(_)) => format!("n={:?}", get("n").cloned().unwrap_or(Value::Null)),
                _ => format!("{value:?}"),
            }
        }
        Value::List(items) => {
            let items: Vec<String> = items.iter().map(describe).collect();
            format!("[{}]", items.join(", "))
        }
        other => format!("{other:?}"),
    }
}

/// The rows of `query` in `language`, each value described.
fn rows(db: &GrafeoDB, language: &str, query: &str) -> Vec<Vec<String>> {
    let result = match language {
        "gql" => db.execute(query),
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

fn edges(ws: &[i64]) -> Vec<Vec<String>> {
    ws.iter().map(|w| vec![format!("K w=Int64({w})")]).collect()
}

fn nodes(ns: &[i64]) -> Vec<Vec<String>> {
    ns.iter().map(|n| vec![format!("n=Int64({n})")]).collect()
}

fn sorted(mut rows: Vec<Vec<String>>) -> Vec<Vec<String>> {
    rows.sort();
    rows
}

const LANGUAGES: [&str; 2] = ["gql", "cypher"];

#[test]
fn edges_through_order_by() {
    let db = three_edges(Config::in_memory());
    for language in LANGUAGES {
        for (query, expected) in [
            (
                "MATCH (a)-[r]->(b) RETURN r ORDER BY r.w",
                edges(&[1, 2, 3]),
            ),
            (
                "MATCH (a)-[r]->(b) RETURN r ORDER BY a.n DESC",
                edges(&[3, 2, 1]),
            ),
            (
                "MATCH (a)-[r]->(b) RETURN r ORDER BY r.w DESC LIMIT 2",
                edges(&[3, 2]),
            ),
            (
                "MATCH (a)-[r]->(b) RETURN r ORDER BY r.w SKIP 1",
                edges(&[2, 3]),
            ),
            (
                "MATCH (a)-[r]->(b) RETURN DISTINCT r ORDER BY r.w",
                edges(&[1, 2, 3]),
            ),
        ] {
            assert_eq!(rows(&db, language, query), expected, "{language}: {query}");
        }
        assert_eq!(
            rows(&db, language, "MATCH (a)-[r]->(b) RETURN a, r ORDER BY a.n"),
            [(1, 1), (3, 2), (5, 3)]
                .map(|(n, w)| vec![format!("n=Int64({n})"), format!("K w=Int64({w})")])
        );
    }
}

/// A chunk cut by `LIMIT` or `SKIP` keeps its records.
#[test]
fn edges_through_limit_and_skip() {
    let db = three_edges(Config::in_memory());
    for language in LANGUAGES {
        let limited = rows(&db, language, "MATCH (a)-[r]->(b) RETURN r LIMIT 2");
        assert_eq!(limited.len(), 2, "{language}");
        assert!(
            limited.iter().all(|row| row[0].starts_with("K w=")),
            "{language}: {limited:?}"
        );
        let skipped = rows(&db, language, "MATCH (a)-[r]->(b) RETURN r SKIP 1");
        assert_eq!(skipped.len(), 2, "{language}");
        assert!(
            skipped.iter().all(|row| row[0].starts_with("K w=")),
            "{language}: {skipped:?}"
        );
    }
}

/// A sort key on a RETURN alias is not a result column.
#[test]
fn a_sort_key_on_an_alias_is_not_returned() {
    let db = three_edges(Config::in_memory());
    for language in LANGUAGES {
        let query = "MATCH (a)-[r]->(b) RETURN r AS e ORDER BY e.w";
        let result = match language {
            "gql" => db.execute(query),
            _ => db.execute_cypher(query),
        }
        .unwrap();
        assert_eq!(result.columns, ["e"], "{language}");
        assert_eq!(rows(&db, language, query), edges(&[1, 2, 3]), "{language}");
    }
}

/// `RETURN *` with `ORDER BY` returns the pattern's variables as records, in
/// order, and no column for the sort key: a property of a returned variable,
/// the variable itself, and a cut or deduplicated result.
#[test]
fn return_star_with_order_by() {
    let db = three_edges(Config::in_memory());
    let record = |a: i64, w: i64, b: i64| {
        vec![
            format!("n=Int64({a})"),
            format!("K w=Int64({w})"),
            format!("n=Int64({b})"),
        ]
    };
    for language in LANGUAGES {
        for (query, expected) in [
            (
                "MATCH (a)-[r]->(b) RETURN * ORDER BY r.w DESC",
                vec![record(5, 3, 6), record(3, 2, 4), record(1, 1, 2)],
            ),
            (
                "MATCH (a)-[r]->(b) RETURN * ORDER BY a.n LIMIT 1",
                vec![record(1, 1, 2)],
            ),
            (
                "MATCH (a)-[r]->(b) RETURN * ORDER BY b.n DESC SKIP 1",
                vec![record(3, 2, 4), record(1, 1, 2)],
            ),
            (
                "MATCH (a)-[r]->(b) RETURN DISTINCT * ORDER BY r.w",
                vec![record(1, 1, 2), record(3, 2, 4), record(5, 3, 6)],
            ),
            (
                "MATCH (a)-[r]->(b) RETURN * ORDER BY a DESC",
                vec![record(5, 3, 6), record(3, 2, 4), record(1, 1, 2)],
            ),
        ] {
            let result = match language {
                "gql" => db.execute(query),
                _ => db.execute_cypher(query),
            }
            .unwrap_or_else(|error| panic!("{language}: {query}: {error}"));
            assert_eq!(result.columns, ["a", "r", "b"], "{language}: {query}");
            assert_eq!(rows(&db, language, query), expected, "{language}: {query}");
        }
    }
}

/// Every branch of a UNION returns records, so the branches' rows compare.
#[test]
fn every_union_branch_returns_records() {
    let db = three_edges(Config::in_memory());
    for language in LANGUAGES {
        for (query, expected) in [
            (
                "MATCH (a)-[r]->(b) WHERE a.n = 1 RETURN r \
                 UNION MATCH (a)-[r]->(b) WHERE a.n = 3 RETURN r",
                edges(&[1, 2]),
            ),
            (
                "MATCH (a)-[r]->(b) RETURN r \
                 UNION MATCH (a)-[r]->(b) WHERE a.n = 3 RETURN r",
                edges(&[1, 2, 3]),
            ),
            (
                "MATCH (a)-[r]->(b) WHERE a.n = 1 RETURN r \
                 UNION ALL MATCH (a)-[r]->(b) WHERE a.n = 1 RETURN r",
                edges(&[1, 1]),
            ),
            (
                "MATCH (a:A) WHERE a.n = 1 RETURN a UNION MATCH (a:A) WHERE a.n = 3 RETURN a",
                nodes(&[1, 3]),
            ),
        ] {
            assert_eq!(
                sorted(rows(&db, language, query)),
                expected,
                "{language}: {query}"
            );
        }
    }
}

/// GQL's other set operations compare and return records.
#[test]
fn gql_set_operations_on_records() {
    let db = three_edges(Config::in_memory());
    for (query, expected) in [
        (
            "MATCH (a)-[r]->(b) RETURN r EXCEPT MATCH (a)-[r]->(b) WHERE a.n = 3 RETURN r",
            edges(&[1, 3]),
        ),
        (
            "MATCH (a)-[r]->(b) RETURN r INTERSECT MATCH (a)-[r]->(b) WHERE a.n = 3 RETURN r",
            edges(&[2]),
        ),
        (
            "MATCH (a)-[r]->(b) WHERE a.n = 99 RETURN r \
             OTHERWISE MATCH (a)-[r]->(b) WHERE a.n = 3 RETURN r",
            edges(&[2]),
        ),
        (
            "MATCH (a:A) RETURN a EXCEPT MATCH (a:A) WHERE a.n = 3 RETURN a",
            nodes(&[1, 5]),
        ),
        (
            "MATCH (a:A) RETURN a INTERSECT MATCH (a:A) WHERE a.n = 3 RETURN a",
            nodes(&[3]),
        ),
    ] {
        assert_eq!(sorted(rows(&db, "gql", query)), expected, "{query}");
    }
}

/// The `shuffle_unordered` test option reorders records without changing them.
#[test]
fn shuffled_results_keep_their_records() {
    let db = three_edges(Config::in_memory().with_shuffle_unordered(true));
    for language in LANGUAGES {
        assert_eq!(
            sorted(rows(&db, language, "MATCH (a)-[r]->(b) RETURN r")),
            edges(&[1, 2, 3]),
            "{language}"
        );
    }
}

/// A name bound to an edge in one branch and to a node in the other is each
/// in its own branch, in both orders.
#[test]
fn a_name_is_a_node_or_an_edge_per_branch() {
    let db = three_edges(Config::in_memory());
    let expected = vec![
        vec!["K w=Int64(1)".to_string()],
        vec!["n=Int64(3)".to_string()],
    ];
    for language in LANGUAGES {
        for query in [
            "MATCH ()-[x]->() WHERE x.w = 1 RETURN x UNION ALL MATCH (x:A) WHERE x.n = 3 RETURN x",
            "MATCH (x:A) WHERE x.n = 3 RETURN x UNION ALL MATCH ()-[x]->() WHERE x.w = 1 RETURN x",
        ] {
            assert_eq!(
                sorted(rows(&db, language, query)),
                expected,
                "{language}: {query}"
            );
        }
    }
}

/// Edges ordered, cut or skipped before RETURN are still edges: their
/// properties and type read right after.
#[test]
fn edges_ordered_before_return_stay_edges() {
    let db = three_edges(Config::in_memory());
    for query in [
        "MATCH (a)-[r]->(b) WITH r ORDER BY r.w DESC LIMIT 2 RETURN r.w AS w, type(r) AS t",
        "MATCH (a)-[r]->(b) WITH r, a ORDER BY a.n SKIP 1 RETURN r.w AS w, type(r) AS t",
        "MATCH (a)-[r]->(b) WITH DISTINCT r ORDER BY r.w LIMIT 2 RETURN r.w AS w, type(r) AS t",
    ] {
        let result = rows(&db, "cypher", query);
        assert_eq!(result.len(), 2, "{query}");
        for row in &result {
            assert!(row[0].starts_with("Int64("), "{query}: {result:?}");
            assert_eq!(row[1], "String(\"K\")", "{query}: {result:?}");
        }
    }
    let gql = "MATCH (a)-[r]->(b) LET x = r.w RETURN r.w AS w, type(r) AS t, x ORDER BY w";
    assert_eq!(
        rows(&db, "gql", gql),
        [1, 2, 3].map(|w| vec![
            format!("Int64({w})"),
            "String(\"K\")".to_string(),
            format!("Int64({w})")
        ])
    );
}

/// An edge a subquery returns keeps its properties through the outer
/// query's ORDER BY and LIMIT.
#[cfg(feature = "cypher")]
#[test]
fn an_edge_from_a_subquery_keeps_its_properties() {
    let db = three_edges(Config::in_memory());
    let query = "MATCH (a:A) CALL { WITH a MATCH (a)-[r]->(b) RETURN r }                  RETURN a.n AS n, r.w AS w ORDER BY n DESC LIMIT 2";
    assert_eq!(
        rows(&db, "cypher", query),
        [(5, 3), (3, 2)].map(|(n, w)| vec![format!("Int64({n})"), format!("Int64({w})")])
    );
}

/// Nodes and edges collected into a list, kept as a group key or unwound from
/// a collected list are returned as records: `collect` and grouping used to
/// return their raw IDs, which overlap (edge 0 and node 0 both exist here).
#[test]
fn collected_grouped_and_unwound_entities_are_records() {
    let db = three_edges(Config::in_memory());
    for language in LANGUAGES {
        let rows = |query: &str| sorted(rows(&db, language, query));
        assert_eq!(
            rows("MATCH (a)-[r]->(b) WHERE r.w = 2 RETURN collect(r) AS rs"),
            [["[K w=Int64(2)]"]],
            "{language}"
        );
        assert_eq!(
            rows("MATCH (a:A) WHERE a.n = 3 RETURN collect(a) AS ns"),
            [["[n=Int64(3)]"]],
            "{language}"
        );
        assert_eq!(
            rows("MATCH (a)-[r]->(b) WITH r, count(*) AS c RETURN r"),
            edges(&[1, 2, 3]),
            "{language}"
        );
        assert_eq!(
            rows("MATCH (a:A)-[r]->(b) WITH a, count(r) AS c RETURN a"),
            nodes(&[1, 3, 5]),
            "{language}"
        );
        assert_eq!(
            rows("MATCH (a)-[r]->(b) WITH collect(r) AS rs UNWIND rs AS e RETURN e"),
            edges(&[1, 2, 3]),
            "{language}"
        );
    }
}
