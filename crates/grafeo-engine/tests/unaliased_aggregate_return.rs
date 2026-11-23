//! A Cypher `RETURN` that aggregates with no alias anywhere (`RETURN n,
//! count(n)`, `RETURN collect(n)`) returns its nodes and edges as nodes and
//! edges, with the columns named as written. The plan ended at the
//! aggregation, with no projection to read the ids back as nodes, so `n` came
//! back as its id and `collect(n)` as a list of ids (0.6.3 too); one alias
//! anywhere in the clause gave the nodes. openCypher TCK 2024.3: Return6 [10],
//! Aggregation5 [1].
//!
//! ```bash
//! cargo test -p grafeo-engine --all-features --test main unaliased_aggregate_return::
//! ```

#![cfg(all(feature = "gql", feature = "cypher"))]

use grafeo_common::types::{LogicalType, PropertyKey, Value};
use grafeo_engine::GrafeoDB;
use grafeo_engine::database::QueryResult;

/// `(:A {k: 1})-[:R {w: 7}]->(:B {k: 2})`.
fn database() -> GrafeoDB {
    let db = GrafeoDB::new_in_memory();
    db.execute_cypher("CREATE (:A {k: 1})-[:R {w: 7}]->(:B {k: 2})")
        .unwrap();
    db
}

fn run(db: &GrafeoDB, language: &str, query: &str) -> QueryResult {
    match language {
        "gql" => db.execute(query),
        "cypher" => db.execute_cypher(query),
        other => panic!("unknown language {other}"),
    }
    .unwrap_or_else(|error| panic!("{language}: {query}: {error}"))
}

/// The `k` of a node map, the `w` of an edge map; panics on anything else.
fn key_of(value: &Value) -> i64 {
    let Value::Map(map) = value else {
        panic!("not a node or edge: {value:?}");
    };
    let key = map
        .get(&PropertyKey::new("k"))
        .or_else(|| map.get(&PropertyKey::new("w")));
    match key {
        Some(Value::Int64(key)) => *key,
        other => panic!("no k or w in {map:?}: {other:?}"),
    }
}

/// A node or edge group key beside an unaliased aggregate is the node or
/// edge, typed as one, and the columns keep the names they are written with.
#[test]
fn a_group_key_beside_an_unaliased_aggregate_is_the_entity() {
    let db = database();
    for (language, query, columns, kind) in [
        (
            "cypher",
            "MATCH (n:A) RETURN n, count(n)",
            ["n", "count(n)"],
            LogicalType::Node,
        ),
        (
            "gql",
            "MATCH (n:A) RETURN n, count(n)",
            ["n", "count(n)"],
            LogicalType::Node,
        ),
        (
            "cypher",
            "MATCH ()-[r]->() RETURN r, count(*)",
            ["r", "count(*)"],
            LogicalType::Edge,
        ),
        (
            "gql",
            "MATCH ()-[r]->() RETURN r, count(*)",
            ["r", "count(*)"],
            LogicalType::Edge,
        ),
    ] {
        let result = run(&db, language, query);
        assert_eq!(result.columns, columns, "{language}: {query}");
        assert_eq!(result.rows().len(), 1, "{language}: {query}");
        let row = &result.rows()[0];
        assert_eq!(
            key_of(&row[0]),
            if kind == LogicalType::Node { 1 } else { 7 },
            "{language}: {query}"
        );
        assert_eq!(row[1], Value::Int64(1), "{language}: {query}");
        assert_eq!(result.cell_type(0, 0), &kind, "{language}: {query}");
    }
}

/// `collect()` of nodes with no alias is a list of the nodes (TCK Return6
/// [10]: `RETURN count(n), collect(n)`).
#[test]
fn an_unaliased_collect_of_nodes_is_a_list_of_nodes() {
    let db = database();
    for language in ["gql", "cypher"] {
        for query in [
            "MATCH (n) RETURN collect(n)",
            "MATCH (n) RETURN count(n), collect(n)",
        ] {
            let result = run(&db, language, query);
            let row = &result.rows()[0];
            let Value::List(nodes) = row.last().unwrap() else {
                panic!("{language}: {query}: {row:?}");
            };
            let mut keys: Vec<i64> = nodes.iter().map(key_of).collect();
            keys.sort_unstable();
            assert_eq!(keys, [1, 2], "{language}: {query}");
        }
    }
}

/// TCK Aggregation5 [1]: a node grouped beside `collect()` of a variable an
/// OPTIONAL MATCH left null is the node, and the list is empty.
#[test]
fn a_group_key_beside_collect_of_an_optional_variable_is_the_node() {
    let db = GrafeoDB::new_in_memory();
    db.execute_cypher("CREATE ()").unwrap();
    let result = run(
        &db,
        "cypher",
        "MATCH (n) OPTIONAL MATCH (n)-[:NOT_EXIST]->(x) RETURN n, collect(x)",
    );
    assert_eq!(result.columns, ["n", "collect(x)"]);
    let row = &result.rows()[0];
    let Value::Map(node) = &row[0] else {
        panic!("the group key is not the node: {row:?}");
    };
    assert!(node.contains_key(&PropertyKey::new("_labels")), "{row:?}");
    assert_eq!(row[1], Value::List(Vec::new().into()));
}

/// Plain group keys and ORDER BY over the unaliased columns are as before.
#[test]
fn plain_group_keys_and_order_by_are_unchanged() {
    let db = database();
    for language in ["gql", "cypher"] {
        let result = run(
            &db,
            language,
            "MATCH (n) RETURN n.k, count(*) ORDER BY n.k DESC",
        );
        assert_eq!(result.columns, ["n.k", "count(*)"], "{language}");
        assert_eq!(
            result.rows(),
            &[
                vec![Value::Int64(2), Value::Int64(1)],
                vec![Value::Int64(1), Value::Int64(1)],
            ],
            "{language}"
        );
    }
    let result = run(
        &db,
        "cypher",
        "MATCH (n) RETURN labels(n)[0] AS l, count(*) ORDER BY count(*) DESC, l",
    );
    assert_eq!(
        result.rows(),
        &[
            vec![Value::from("A"), Value::Int64(1)],
            vec![Value::from("B"), Value::Int64(1)],
        ]
    );
}
