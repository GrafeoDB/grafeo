//! Lists of nodes and edges come back as nodes and edges: `relationships(p)`,
//! `nodes(p)`, the variable of a variable-length edge pattern (a list of the
//! path's edges, one per hop) and what `reverse`, `tail`, `head`, `last` and
//! indexing make of them.
//!
//! ```bash
//! cargo test -p grafeo-engine --all-features --test entity_lists
//! ```

use grafeo_common::types::{PropertyKey, Value};
use grafeo_engine::GrafeoDB;

/// Dir a -[CONTAINS {w: 1}]-> Dir b -[CONTAINS {w: 2}]-> File f.
fn chain() -> GrafeoDB {
    let db = GrafeoDB::new_in_memory();
    db.execute(
        "INSERT (:Dir {name: 'a'})-[:CONTAINS {w: 1}]->(:Dir {name: 'b'})\
         -[:CONTAINS {w: 2}]->(:File {name: 'f'})",
    )
    .unwrap();
    db
}

fn field(map: &Value, key: &str) -> Value {
    match map {
        Value::Map(map) => map
            .get(&PropertyKey::new(key))
            .cloned()
            .unwrap_or(Value::Null),
        other => panic!("expected a map, got {other:?}"),
    }
}

/// `(type, w)` of each relationship map in a list.
fn relationships(list: &Value) -> Vec<(Value, Value)> {
    match list {
        Value::List(items) => items
            .iter()
            .map(|edge| (field(edge, "_type"), field(edge, "w")))
            .collect(),
        other => panic!("expected a list, got {other:?}"),
    }
}

fn contains(w: i64) -> (Value, Value) {
    (Value::from("CONTAINS"), Value::Int64(w))
}

/// The one row of `query`.
fn row(
    result: grafeo_common::utils::error::Result<grafeo_engine::database::QueryResult>,
) -> Vec<Value> {
    let result = result.unwrap();
    assert_eq!(result.rows().len(), 1, "{:?}", result.rows());
    result.rows()[0].clone()
}

#[cfg(feature = "cypher")]
#[test]
fn a_variable_length_edge_variable_returns_its_relationships() {
    let db = chain();
    let row = row(db.execute_cypher(
        "MATCH p = (d:Dir {name: 'a'})-[r*1..3]->(x:File) \
         RETURN r, relationships(p), last(r), r[0], reverse(r), tail(r)",
    ));

    assert_eq!(relationships(&row[0]), vec![contains(1), contains(2)]);
    assert_eq!(relationships(&row[1]), vec![contains(1), contains(2)]);
    assert_eq!(field(&row[2], "w"), Value::Int64(2));
    assert_eq!(field(&row[3], "w"), Value::Int64(1));
    assert_eq!(relationships(&row[4]), vec![contains(2), contains(1)]);
    assert_eq!(relationships(&row[5]), vec![contains(2)]);
}

#[test]
fn a_gql_quantified_edge_variable_returns_its_relationships() {
    let db = chain();
    let row =
        row(db.execute("MATCH p = (d:Dir {name: 'a'})-[r]->{1,3}(x:File) RETURN r, nodes(p)"));

    assert_eq!(relationships(&row[0]), vec![contains(1), contains(2)]);
    let names: Vec<Value> = match &row[1] {
        Value::List(nodes) => nodes.iter().map(|node| field(node, "name")).collect(),
        other => panic!("expected a list, got {other:?}"),
    };
    assert_eq!(
        names,
        vec![Value::from("a"), Value::from("b"), Value::from("f")]
    );
}

/// An edge list keeps its items through WITH, under its own name or an alias.
#[test]
fn with_keeps_an_edge_list() {
    let db = chain();
    let row = row(db.execute(
        "MATCH p = (d:Dir {name: 'a'})-[r]->{1,3}(x:File) \
         WITH r AS hops, relationships(p) AS rels \
         RETURN [e IN hops | type(e)], [e IN rels | e.w], hops",
    ));

    assert_eq!(
        row[0],
        Value::List(vec![Value::from("CONTAINS"), Value::from("CONTAINS")].into())
    );
    assert_eq!(
        row[1],
        Value::List(vec![Value::Int64(1), Value::Int64(2)].into())
    );
    assert_eq!(relationships(&row[2]), vec![contains(1), contains(2)]);
}

/// A zero-length match binds the edge variable to an empty list.
#[test]
fn a_zero_length_match_binds_an_empty_list() {
    let db = chain();
    let result = db
        .execute("MATCH (x:File)-[r]->{0,1}(y) RETURN size(r), y.name")
        .unwrap();
    let rows: Vec<Vec<Value>> = result.rows().to_vec();
    assert_eq!(rows, vec![vec![Value::Int64(0), Value::from("f")]]);
}
