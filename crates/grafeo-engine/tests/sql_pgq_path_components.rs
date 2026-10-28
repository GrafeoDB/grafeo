//! SQL/PGQ path components retain node and edge values, order and identity.

#![cfg(feature = "sql-pgq")]

use std::collections::{BTreeMap, HashMap};
use std::sync::Arc;

use grafeo_common::types::{EdgeId, NodeId, PropertyKey, Value};
use grafeo_engine::GrafeoDB;

#[cfg(feature = "gql")]
#[path = "support/adversarial_paths.rs"]
mod adversarial_paths;

fn text(value: &str) -> Value {
    Value::String(value.into())
}

fn list(values: impl IntoIterator<Item = Value>) -> Value {
    Value::List(values.into_iter().collect::<Vec<_>>().into())
}

fn map(values: impl IntoIterator<Item = (&'static str, Value)>) -> Value {
    Value::Map(Arc::new(
        values
            .into_iter()
            .map(|(key, value)| (PropertyKey::new(key), value))
            .collect::<BTreeMap<_, _>>(),
    ))
}

fn id(value: u64) -> Value {
    Value::Int64(i64::try_from(value).unwrap())
}

struct Chain {
    db: GrafeoDB,
    nodes: [NodeId; 3],
    edges: [EdgeId; 2],
}

impl Chain {
    fn new() -> Self {
        let db = GrafeoDB::new_in_memory();
        let session = db.session();
        let nodes = [("Ana", 30), ("Ben", 40), ("Cy", 50)].map(|(name, age)| {
            session
                .create_node_with_props(
                    &["Person"],
                    [("name", text(name)), ("age", Value::Int64(age))],
                )
                .unwrap()
        });
        let first = session
            .create_edge_with_props(nodes[0], nodes[1], "KNOWS", [("weight", Value::Int64(3))])
            .unwrap();
        let second = session
            .create_edge_with_props(nodes[1], nodes[2], "KNOWS", [("weight", Value::Int64(7))])
            .unwrap();
        Self {
            db,
            nodes,
            edges: [first, second],
        }
    }

    fn node(&self, index: usize) -> Value {
        let (name, age) = [("Ana", 30), ("Ben", 40), ("Cy", 50)][index];
        map([
            ("_id", id(self.nodes[index].as_u64())),
            ("_labels", list([text("Person")])),
            ("name", text(name)),
            ("age", Value::Int64(age)),
        ])
    }

    fn edge(&self, index: usize) -> Value {
        map([
            ("_id", id(self.edges[index].as_u64())),
            ("_type", text("KNOWS")),
            ("_source", id(self.nodes[index].as_u64())),
            ("_target", id(self.nodes[index + 1].as_u64())),
            ("weight", Value::Int64([3, 7][index])),
        ])
    }

    fn forward(&self, hops: usize) -> Vec<Value> {
        vec![
            list((0..=hops).map(|index| self.node(index))),
            list((0..hops).map(|index| self.edge(index))),
            Value::Int64(i64::try_from(hops).unwrap()),
        ]
    }
}

#[test]
fn columns_return_complete_node_and_edge_values_in_walk_order() {
    let chain = Chain::new();
    let result = chain
        .db
        .execute_sql(
            "SELECT * FROM GRAPH_TABLE (
                MATCH (a:Person {name: 'Ana'})-[p:KNOWS*1..2]->(b:Person)
                COLUMNS (NODES(p) AS ns, EDGES(p) AS es, LENGTH(p) AS hops)
            ) ORDER BY hops",
        )
        .unwrap();
    assert_eq!(result.columns, ["ns", "es", "hops"]);
    assert_eq!(result.rows(), [chain.forward(1), chain.forward(2)]);
}

#[test]
fn qualified_columns_and_nested_head_last_preserve_entity_values() {
    let chain = Chain::new();
    let direct = chain
        .db
        .execute_sql(
            "SELECT NODES(p) AS ns, EDGES(p) AS es, LENGTH(p) AS hops
             FROM GRAPH_TABLE (
                MATCH (a:Person {name: 'Ana'})-[p:KNOWS*1..2]->(b:Person)
                COLUMNS (LENGTH(p) AS hops)
             ) ORDER BY hops",
        )
        .unwrap();
    assert_eq!(direct.rows(), [chain.forward(1), chain.forward(2)]);
    let result = chain
        .db
        .execute_sql(
            "SELECT HEAD(g.ns) AS first_node, LAST(g.ns) AS last_node,
                    HEAD(g.es) AS first_edge, LAST(g.es) AS last_edge
             FROM GRAPH_TABLE (
                MATCH (a:Person {name: 'Ana'})-[p:KNOWS*2..2]->(b:Person)
                COLUMNS (NODES(p) AS ns, EDGES(p) AS es)
             ) AS g",
        )
        .unwrap();
    assert_eq!(
        result.rows(),
        [vec![
            chain.node(0),
            chain.node(2),
            chain.edge(0),
            chain.edge(1)
        ]]
    );
}

#[test]
fn zero_hops_return_one_node_no_edges_and_zero_length() {
    let chain = Chain::new();
    let result = chain
        .db
        .execute_sql(
            "SELECT * FROM GRAPH_TABLE (
                MATCH (a:Person {name: 'Ana'})-[p:KNOWS*0..0]->(b:Person)
                COLUMNS (NODES(p) AS ns, EDGES(p) AS es, LENGTH(p) AS hops)
            )",
        )
        .unwrap();
    assert_eq!(result.rows(), [chain.forward(0)]);
}

#[test]
fn incoming_walk_reverses_list_order_without_reversing_edge_identity() {
    let chain = Chain::new();
    let result = chain
        .db
        .execute_sql(
            "SELECT * FROM GRAPH_TABLE (
                MATCH (a:Person {name: 'Cy'})<-[p:KNOWS*2..2]-(b:Person)
                COLUMNS (NODES(p) AS ns, EDGES(p) AS es, LENGTH(p) AS hops)
            )",
        )
        .unwrap();
    assert_eq!(
        result.rows(),
        [vec![
            list([chain.node(2), chain.node(1), chain.node(0)]),
            list([chain.edge(1), chain.edge(0)]),
            Value::Int64(2),
        ]]
    );
}

#[test]
fn parameters_and_cached_reexecution_keep_complete_components() {
    let chain = Chain::new();
    let session = chain.db.session();
    let query = "SELECT * FROM GRAPH_TABLE (
        MATCH (a:Person {name: $who})-[p:KNOWS*1..1]->(b:Person)
        COLUMNS (NODES(p) AS ns, EDGES(p) AS es))";
    for (name, source) in [("Ana", 0), ("Ben", 1), ("Ana", 0)] {
        let result = session
            .execute_sql_with_params(query, HashMap::from([("who".to_owned(), text(name))]))
            .unwrap();
        assert_eq!(
            result.rows(),
            [vec![
                list([chain.node(source), chain.node(source + 1)]),
                list([chain.edge(source)]),
            ]],
            "parameter {name}"
        );
    }
    let query = "SELECT * FROM GRAPH_TABLE (
        MATCH (a:Person {name: 'Ana'})-[p:KNOWS*1..2]->(b:Person)
        COLUMNS (NODES(p) AS ns, EDGES(p) AS es, LENGTH(p) AS hops)) ORDER BY hops";
    for _ in 0..2 {
        let result = session.execute_sql(query).unwrap();
        assert_eq!(result.rows(), [chain.forward(1), chain.forward(2)]);
    }
}

#[test]
fn snapshot_components_keep_their_node_and_edge_versions() {
    let chain = Chain::new();
    let query = "SELECT * FROM GRAPH_TABLE (
        MATCH (a:Person {name: 'Ana'})-[p:KNOWS*1..1]->(b:Person)
        COLUMNS (NODES(p) AS ns, EDGES(p) AS es, LENGTH(p) AS hops))";
    let mut snapshot = chain.db.session();
    snapshot.begin_transaction().unwrap();
    assert_eq!(
        snapshot.execute_sql(query).unwrap().rows(),
        [chain.forward(1)]
    );

    let mut writer = chain.db.session();
    writer.begin_transaction().unwrap();
    writer
        .set_node_property(chain.nodes[0], "age", Value::Int64(31))
        .unwrap();
    writer
        .set_edge_property(chain.edges[0], "weight", Value::Int64(11))
        .unwrap();
    writer.commit().unwrap();

    assert_eq!(
        snapshot.execute_sql(query).unwrap().rows(),
        [chain.forward(1)],
        "the open Snapshot transaction resolves both components at its read epoch"
    );
    let current = vec![
        list([
            map([
                ("_id", id(chain.nodes[0].as_u64())),
                ("_labels", list([text("Person")])),
                ("name", text("Ana")),
                ("age", Value::Int64(31)),
            ]),
            chain.node(1),
        ]),
        list([map([
            ("_id", id(chain.edges[0].as_u64())),
            ("_type", text("KNOWS")),
            ("_source", id(chain.nodes[0].as_u64())),
            ("_target", id(chain.nodes[1].as_u64())),
            ("weight", Value::Int64(11)),
        ])]),
        Value::Int64(1),
    ];
    assert_eq!(
        chain.db.session().execute_sql(query).unwrap().rows(),
        std::slice::from_ref(&current),
        "a fresh session resolves the committed versions"
    );
    snapshot.rollback().unwrap();
    assert_eq!(
        snapshot.execute_sql(query).unwrap().rows(),
        [current],
        "ending the Snapshot transaction releases its old component versions"
    );
}

#[test]
fn missing_endpoint_keeps_an_empty_result_with_component_columns() {
    let chain = Chain::new();
    let result = chain
        .db
        .execute_sql(
            "SELECT * FROM GRAPH_TABLE (
                MATCH (a:Person {name: 'Ana'})-[p:KNOWS*1..2]->(b:Person {name: 'Absent'})
                COLUMNS (NODES(p) AS ns, EDGES(p) AS es))",
        )
        .unwrap();
    assert_eq!(result.columns, ["ns", "es"]);
    assert_eq!(result.rows(), Vec::<Vec<Value>>::new());
}

#[cfg(feature = "gql")]
mod shared {
    use super::*;
    use grafeo_core::graph::Direction;

    fn expected(db: &GrafeoDB, names: &[&str], slots: &[i64], kind: &str) -> Vec<Value> {
        let session = db.session();
        let nodes: Vec<NodeId> = names
            .iter()
            .map(|name| {
                let ids = session.find_nodes_by_property("id", &text(name)).unwrap();
                assert_eq!(ids.len(), 1, "shared node {name}");
                ids[0]
            })
            .collect();
        let node_values = names.iter().zip(&nodes).map(|(name, node)| {
            map([
                ("_id", id(node.as_u64())),
                ("_labels", list([text("Node")])),
                ("id", text(name)),
            ])
        });
        let edge_values = slots.iter().enumerate().map(|(index, slot)| {
            let matches: Vec<EdgeId> = session
                .get_node_edges(nodes[index], Direction::Outgoing, &[kind])
                .unwrap()
                .into_iter()
                .filter_map(|(target, edge_id)| {
                    let edge = session.get_edge(edge_id).unwrap().unwrap();
                    (target == nodes[index + 1]
                        && edge.get_property("slot") == Some(&Value::Int64(*slot)))
                    .then_some(edge_id)
                })
                .collect();
            assert_eq!(matches.len(), 1, "shared edge slot {slot}");
            map([
                ("_id", id(matches[0].as_u64())),
                ("_type", text(kind)),
                ("_source", id(nodes[index].as_u64())),
                ("_target", id(nodes[index + 1].as_u64())),
                ("cost", Value::Int64(1)),
                ("slot", Value::Int64(*slot)),
            ])
        });
        vec![list(node_values), list(edge_values)]
    }

    fn sorted_rows(db: &GrafeoDB, query: &str) -> Vec<Vec<Value>> {
        let mut rows = db.execute_sql(query).unwrap().rows().to_vec();
        rows.sort_by_key(|row| format!("{row:?}"));
        rows
    }

    fn sorted(mut rows: Vec<Vec<Value>>) -> Vec<Vec<Value>> {
        rows.sort_by_key(|row| format!("{row:?}"));
        rows
    }

    #[test]
    fn free_endpoint_keeps_all_six_then_nine_parallel_walk_values() {
        let db = adversarial_paths::adversarial_path_graph(true);
        let query = "SELECT * FROM GRAPH_TABLE (
            MATCH (s:Node {id: 's'})-[p:REL*1..2]->(dst:Node)
            COLUMNS (NODES(p) AS ns, EDGES(p) AS es))";
        let mut expected_rows = vec![
            expected(&db, &["s", "a"], &[0], "REL"),
            expected(&db, &["s", "b"], &[1], "REL"),
            expected(&db, &["s", "a", "d"], &[0, 2], "REL"),
            expected(&db, &["s", "a", "h"], &[0, 9], "REL"),
            expected(&db, &["s", "b", "d"], &[1, 3], "REL"),
            expected(&db, &["s", "b", "g"], &[1, 7], "REL"),
        ];
        assert_eq!(sorted_rows(&db, query), sorted(expected_rows.clone()));
        adversarial_paths::add_parallel_rel(&db);
        expected_rows.extend([
            expected(&db, &["s", "a"], &[14], "REL"),
            expected(&db, &["s", "a", "d"], &[14, 2], "REL"),
            expected(&db, &["s", "a", "h"], &[14, 9], "REL"),
        ]);
        assert_eq!(sorted_rows(&db, query), sorted(expected_rows));
    }

    #[test]
    fn pinned_parallel_paths_keep_distinct_edges_and_identical_nodes() {
        let db = adversarial_paths::adversarial_path_graph(false);
        assert_eq!(
            sorted_rows(
                &db,
                "SELECT * FROM GRAPH_TABLE (
                    MATCH (s:Node {id: 's'})-[p:OTHER*1..1]->(x:Node {id: 'x'})
                    COLUMNS (NODES(p) AS ns, EDGES(p) AS es))",
            ),
            sorted(vec![
                expected(&db, &["s", "x"], &[11], "OTHER"),
                expected(&db, &["s", "x"], &[12], "OTHER"),
            ])
        );
    }
}
