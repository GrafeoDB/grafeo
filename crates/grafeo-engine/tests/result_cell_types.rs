//! Every cell of a result has the type it was built with, and that type says
//! where its nodes and edges are, inside lists, maps and paths too: a value is
//! a node exactly where its type says `Node`. A map that only looks like a
//! node (`{_id: 1, _labels: []}`) is a map, and an edge with a property named
//! `_labels` is an edge. Readers used to guess from the keys of the map.
//!
//! ```bash
//! cargo test -p grafeo-engine --all-features --test main result_cell_types::
//! ```

#![cfg(all(feature = "gql", feature = "cypher"))]

use grafeo_common::types::{LogicalType, PropertyKey, Value};
use grafeo_engine::database::QueryResult;
use grafeo_engine::{Config, GrafeoDB};

/// `(:A {n: 1})-[:K {w: 1}]->(:B {n: 2})`, and the same with n 3, 4 and w 2,
/// and n 5, 6 and w 3.
fn three_edges() -> GrafeoDB {
    let db = GrafeoDB::with_config(Config::in_memory()).unwrap();
    for (a, w, b) in [(1, 1, 2), (3, 2, 4), (5, 3, 6)] {
        db.execute(&format!(
            "INSERT (:A {{n: {a}}})-[:K {{w: {w}}}]->(:B {{n: {b}}})"
        ))
        .unwrap();
    }
    db
}

fn key<'a>(map: &'a std::collections::BTreeMap<PropertyKey, Value>, name: &str) -> &'a Value {
    map.get(&PropertyKey::new(name)).unwrap_or(&Value::Null)
}

/// The nodes and edges in `value` where its type `ty` says they are (the
/// reading rule), as `n=1` for a node and `w=1` for an edge.
fn by_type(value: &Value, ty: &LogicalType, found: &mut Vec<String>) {
    match (ty, value) {
        (LogicalType::Node, Value::Map(map)) => found.push(format!("n={:?}", key(map, "n"))),
        (LogicalType::Edge, Value::Map(map)) => found.push(format!("w={:?}", key(map, "w"))),
        (LogicalType::List(item), Value::List(items)) => {
            for value in items.iter() {
                by_type(value, item, found);
            }
        }
        (LogicalType::Tuple(types), Value::List(items)) => {
            for (value, ty) in items.iter().zip(types) {
                by_type(value, ty, found);
            }
        }
        (LogicalType::Struct(fields), Value::Map(map)) => {
            for (name, ty) in fields {
                by_type(key(map, name), ty, found);
            }
        }
        (LogicalType::Map { value: item, .. }, Value::Map(map)) => {
            for value in map.values() {
                by_type(value, item, found);
            }
        }
        (LogicalType::Path, Value::Path { nodes, edges }) => {
            for node in nodes.iter() {
                by_type(node, &LogicalType::Node, found);
            }
            for edge in edges.iter() {
                by_type(edge, &LogicalType::Edge, found);
            }
        }
        _ => {}
    }
}

/// The nodes and edges in `value` by the shape of its maps (the old rule):
/// a map with `_id` and `_labels` is a node, one with `_id` and `_type` an
/// edge, and the values of any other map are searched.
fn by_shape(value: &Value, found: &mut Vec<String>) {
    match value {
        Value::Map(map) => match (key(map, "_id"), key(map, "_labels"), key(map, "_type")) {
            (Value::Int64(_), Value::List(_), _) => found.push(format!("n={:?}", key(map, "n"))),
            (Value::Int64(_), _, Value::String(_)) => {
                found.push(format!("w={:?}", key(map, "w")));
            }
            _ => map.values().for_each(|value| by_shape(value, found)),
        },
        Value::List(items) => items.iter().for_each(|value| by_shape(value, found)),
        Value::Path { nodes, edges } => {
            nodes
                .iter()
                .chain(edges.iter())
                .for_each(|value| by_shape(value, found));
        }
        _ => {}
    }
}

/// Per row, the nodes and edges every cell holds by its type.
fn typed(result: &QueryResult) -> Vec<Vec<String>> {
    result
        .rows()
        .iter()
        .enumerate()
        .map(|(row, values)| {
            let mut found = Vec::new();
            for (column, value) in values.iter().enumerate() {
                by_type(value, result.cell_type(row, column), &mut found);
            }
            found.sort();
            found
        })
        .collect()
}

/// Per row, the nodes and edges every cell holds by the shape of its maps.
fn shaped(result: &QueryResult) -> Vec<Vec<String>> {
    result
        .rows()
        .iter()
        .map(|values| {
            let mut found = Vec::new();
            for value in values {
                by_shape(value, &mut found);
            }
            found.sort();
            found
        })
        .collect()
}

fn run(db: &GrafeoDB, language: &str, query: &str) -> QueryResult {
    match language {
        "gql" => db.execute(query),
        "cypher" => db.execute_cypher(query),
        other => panic!("unknown language {other}"),
    }
    .unwrap_or_else(|error| panic!("{language}: {query}: {error}"))
}

/// Queries that return real nodes and edges, alone and inside lists, maps and
/// paths, through ordering, cuts, deduplication and set operations: the typed
/// rule finds what the shape rule finds, and finds something in every row.
#[test]
fn the_types_find_every_returned_node_and_edge() {
    let db = three_edges();
    let queries = [
        "MATCH (a:A) RETURN a",
        "MATCH (a)-[r]->(b) RETURN r",
        "MATCH (a)-[r]->(b) RETURN a, r, b",
        "MATCH (a)-[r]->(b) RETURN [a, 1, r] AS l",
        "MATCH (a)-[r]->(b) RETURN {k: a, e: r, x: 1} AS m",
        "MATCH (a)-[r]->(b) RETURN [{k: a}] AS l",
        "MATCH p = (a)-[r]->(b) RETURN p",
        "MATCH p = (a)-[r]->(b) RETURN nodes(p) AS ns, relationships(p) AS rs",
        "MATCH (a:A) RETURN collect(a) AS all",
        "MATCH (a)-[r]->(b) RETURN collect(r) AS all",
        "MATCH (a)-[r*1..1]->(b) RETURN r",
        "MATCH (a)-[r]->(b) RETURN r ORDER BY r.w DESC LIMIT 2",
        "MATCH (a)-[r]->(b) RETURN DISTINCT a ORDER BY a.n SKIP 1",
        "MATCH (a:A) RETURN a AS x UNION ALL MATCH (b:B) RETURN b AS x",
    ];
    for language in ["gql", "cypher"] {
        for query in queries {
            let result = run(&db, language, query);
            let typed = typed(&result);
            assert!(
                !typed.is_empty() && typed.iter().all(|row| !row.is_empty()),
                "{language}: {query}: {typed:?}"
            );
            assert_eq!(typed, shaped(&result), "{language}: {query}");
        }
    }
}

/// A map that looks like a node or an edge is a map: only the real node next
/// to it is one.
#[test]
fn a_map_shaped_like_a_node_is_a_map() {
    let db = three_edges();
    for (query, expected) in [
        ("RETURN {_id: 1, _labels: []} AS m", vec![]),
        (
            "RETURN {_id: 1, _type: 'K', _source: 1, _target: 2} AS m",
            vec![],
        ),
        (
            "MATCH (a:A {n: 1}) RETURN [a, {_id: 9, _labels: ['A'], n: 9}] AS l",
            vec!["n=Int64(1)".to_string()],
        ),
        (
            "MATCH (a:A {n: 1}) RETURN {real: a, fake: {_id: 9, _labels: [], n: 9}} AS m",
            vec!["n=Int64(1)".to_string()],
        ),
    ] {
        for language in ["gql", "cypher"] {
            let result = run(&db, language, query);
            assert_eq!(
                typed(&result),
                std::slice::from_ref(&expected),
                "{language}: {query}"
            );
        }
    }
}

/// An edge with a property named `_labels` is an edge (the shape rule took it
/// for a node), and a node with a property named `_type` is a node; both keep
/// the property.
#[test]
fn structural_names_of_the_other_kind_stay_properties() {
    let db = GrafeoDB::with_config(Config::in_memory()).unwrap();
    db.execute("INSERT (:A {n: 1, _type: 'x'})-[:K {w: 7, _labels: ['L']}]->(:B {n: 2})")
        .unwrap();
    let result = run(&db, "gql", "MATCH (a:A)-[r]->(b) RETURN a, r");
    assert_eq!(typed(&result), [vec!["n=Int64(1)", "w=Int64(7)"]]);
    let row = &result.rows()[0];
    let (Value::Map(node), Value::Map(edge)) = (&row[0], &row[1]) else {
        panic!("{row:?}");
    };
    assert_eq!(key(node, "_type"), &Value::from("x"));
    assert_eq!(
        key(edge, "_labels"),
        &Value::List(vec![Value::from("L")].into())
    );
}

/// A set operation whose branches return nodes in one and edges in the other
/// gives each row the type of its own branch.
#[test]
fn each_branch_of_a_union_keeps_its_types() {
    let db = three_edges();
    for language in ["gql", "cypher"] {
        let result = run(
            &db,
            language,
            "MATCH (a:A) RETURN a AS x UNION ALL MATCH ()-[r]->() RETURN r AS x",
        );
        let kinds: Vec<&LogicalType> = (0..result.row_count())
            .map(|row| result.cell_type(row, 0))
            .collect();
        assert_eq!(kinds.len(), 6, "{language}");
        assert!(
            kinds[..3].iter().all(|ty| **ty == LogicalType::Node),
            "{language}: {kinds:?}"
        );
        assert!(
            kinds[3..].iter().all(|ty| **ty == LogicalType::Edge),
            "{language}: {kinds:?}"
        );
    }
}

/// The nodes of a path read in an expression carry the node's id under `_id`
/// and its labels under `_labels`, also when a property has that name: the
/// structural keys win everywhere, as for a returned node (#254). A property
/// read (`a._id`) still reads the property.
#[test]
fn path_nodes_keep_their_id_over_a_property_of_that_name() {
    let db = GrafeoDB::with_config(Config::in_memory()).unwrap();
    db.execute("INSERT (:A {n: 1, _id: 'prop'})-[:K]->(:B {n: 2})")
        .unwrap();
    let id = run(&db, "gql", "MATCH (a:A) RETURN id(a) AS id").rows()[0][0].clone();
    for (language, query) in [
        ("gql", "MATCH p = (a:A)-[]->() RETURN nodes(p)[0] AS first"),
        ("cypher", "MATCH p = (a:A)-->() RETURN nodes(p)[0] AS first"),
    ] {
        let result = run(&db, language, query);
        let Value::Map(first) = &result.rows()[0][0] else {
            panic!("{language}: {query}: {:?}", result.rows());
        };
        assert_eq!(key(first, "_id"), &id, "{language}: {query}");
        assert_eq!(typed(&result), [vec!["n=Int64(1)"]], "{language}: {query}");
    }
    let property = run(&db, "gql", "MATCH (a:A) RETURN a._id AS v");
    assert_eq!(property.rows()[0][0], Value::from("prop"));
}

/// An expression that builds a value from typed parts has the type it builds:
/// a list comprehension the list of what its projection gives (with its
/// variable an item of its list), `CASE` and `coalesce` what all their
/// results give (a null result aside), `+` of lists the list both sides make,
/// and `reduce` what its accumulator keeps through every step. Each gave
/// `Any`, so its nodes and edges came back as maps (or, from a list
/// comprehension over `nodes(p)`, as their ids).
#[test]
fn expressions_built_from_typed_parts_keep_their_types() {
    let db = three_edges();
    for query in [
        "MATCH (a:A) WITH collect(a) AS l RETURN [x IN l | x] AS m",
        "MATCH (a:A) RETURN [x IN collect(a) | x] AS m",
        "MATCH p = (a)-[r]->(b) WITH collect(p) AS ps RETURN [x IN ps | head(nodes(x))] AS m",
        "MATCH p = (a)-[r]->(b) RETURN [x IN nodes(p) | x][0] AS m",
        "MATCH p = (a)-[r]->(b) RETURN [x IN relationships(p) WHERE x.w > 0] AS m",
        "MATCH (a)-[r]->(b) RETURN [x IN range(1, 2) | a] AS m",
        "MATCH (a)-[r]->(b) RETURN [x IN [a, b] | {node: x}] AS m",
        "MATCH (a)-[r]->(b) RETURN last([x IN [a, b] | x]) AS m",
        "MATCH (a)-[r]->(b) RETURN [x IN [a, b] | x][0..1] AS m",
        "MATCH (a)-[r]->(b) RETURN CASE WHEN a.n > 2 THEN a ELSE b END AS m",
        "MATCH (a)-[r]->(b) RETURN CASE a.n WHEN 100 THEN null ELSE r END AS m",
        "MATCH (a)-[r]->(b) RETURN coalesce(null, a) AS m",
        "MATCH p = (a)-[r]->(b) RETURN nodes(p) + [a] AS m",
        "MATCH p = (a)-[r]->(b) RETURN [] + relationships(p) AS m",
        "MATCH p = (a)-[r]->(b) RETURN reduce(acc = [], x IN nodes(p) | acc + [x]) AS m",
        "MATCH (a)-[r]->(b) RETURN reduce(acc = a, x IN [1] | acc) AS m",
    ] {
        for language in ["gql", "cypher"] {
            let result = run(&db, language, query);
            let typed = typed(&result);
            assert!(
                !typed.is_empty() && typed.iter().all(|row| !row.is_empty()),
                "{language}: {query}: {typed:?} {:?}",
                result.rows()
            );
            assert_eq!(typed, shaped(&result), "{language}: {query}");
        }
    }
}

/// A list comprehension's variable is what its list holds, also when it has
/// the name of a node column, and a projection that reads no node gives no
/// node.
#[test]
fn a_comprehension_over_plain_values_holds_no_node() {
    let db = three_edges();
    for query in [
        "MATCH (a:A) RETURN [a IN [1, 2] | a] AS m",
        "MATCH (a:A) RETURN [x IN [a] | x.n] AS m",
        "UNWIND [1, 2] AS i RETURN [x IN [i] | x] AS m",
        "MATCH (a:A) RETURN reduce(s = 0, x IN [a] | s + x.n) AS m",
        // The inner `x` hides the outer one, which is a node.
        "MATCH (a:A) RETURN [x IN [a] | [x IN [1] | x]] AS m",
    ] {
        for language in ["gql", "cypher"] {
            let result = run(&db, language, query);
            for row in 0..result.row_count() {
                for column in 0..result.columns.len() {
                    let ty = result.cell_type(row, column);
                    assert!(!holds_node_or_edge(ty), "{language}: {query}: {ty:?}");
                }
            }
        }
    }
}

/// Whether a cell of type `ty` holds a node or an edge somewhere inside.
fn holds_node_or_edge(ty: &LogicalType) -> bool {
    match ty {
        LogicalType::Node | LogicalType::Edge | LogicalType::Path => true,
        LogicalType::List(item) | LogicalType::Map { value: item, .. } => holds_node_or_edge(item),
        LogicalType::Tuple(items) => items.iter().any(holds_node_or_edge),
        LogicalType::Struct(fields) => fields.iter().any(|(_, field)| holds_node_or_edge(field)),
        _ => false,
    }
}

/// A node or edge taken from a list comprehension, by index or by `UNWIND`,
/// is the node or edge, not its id.
#[test]
fn an_entity_taken_from_a_list_comprehension_is_the_entity() {
    let db = three_edges();
    for language in ["gql", "cypher"] {
        for query in [
            "MATCH p = (a)-[r]->(b) RETURN [x IN nodes(p) | x][0] AS m",
            "MATCH p = (a)-[r]->(b) RETURN [x IN relationships(p) | x][0] AS m",
            "MATCH p = (a)-[r]->(b) UNWIND [x IN nodes(p) | x] AS y RETURN y",
            "MATCH p = (a)-[r]->(b) UNWIND [x IN relationships(p) | x] AS y RETURN y",
        ] {
            let result = run(&db, language, query);
            let typed = typed(&result);
            assert!(
                !typed.is_empty() && typed.iter().all(|row| !row.is_empty()),
                "{language}: {query}: {:?}",
                result.rows()
            );
            assert_eq!(typed, shaped(&result), "{language}: {query}");
        }
    }
}

/// What a subquery gives keeps the types of its nodes and edges: a pattern
/// comprehension, a `VALUE` subquery and a `CALL` subquery, whose columns
/// join the outer rows.
#[test]
fn a_subquery_output_keeps_its_types() {
    let db = three_edges();
    for (language, query) in [
        (
            "cypher",
            "MATCH (a)-[r]->(b) RETURN [(a)-[s]->(c) | c] AS m",
        ),
        (
            "gql",
            "MATCH (a:A) RETURN VALUE { MATCH (b:B) RETURN b } AS m",
        ),
        (
            "cypher",
            "MATCH (a:A) CALL { WITH a MATCH (a)-->(b) RETURN b } RETURN b",
        ),
    ] {
        let result = run(&db, language, query);
        let typed = typed(&result);
        assert!(
            !typed.is_empty() && typed.iter().all(|row| !row.is_empty()),
            "{language}: {query}: {typed:?}"
        );
        assert_eq!(typed, shaped(&result), "{language}: {query}");
    }
}
