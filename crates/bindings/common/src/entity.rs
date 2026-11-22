//! Language-agnostic entity extraction from query results.
//!
//! Query results encode nodes and edges as `Value::Map` records with the
//! structural keys `_id` and `_labels` (a node) or `_id`, `_type`, `_source`
//! and `_target` (an edge). Which maps are records is said by the type each
//! cell was built with ([`QueryResult::cell_type`]), never by the keys of a
//! map: `{_id: 1, _labels: []}` is a map. This module walks result rows by
//! their types and extracts deduplicated [`RawNode`] and [`RawEdge`] structs
//! that each binding can cheaply convert to its language-specific wrapper.

use std::collections::{BTreeMap, HashMap, HashSet};

use grafeo_common::types::{EdgeId, LogicalType, NodeId, PropertyKey, Value};
use grafeo_engine::database::{CellTypes, QueryResult};

/// A node extracted from query results (language-agnostic).
#[derive(Debug, Clone)]
pub struct RawNode {
    /// The node's internal ID.
    pub id: NodeId,
    /// Labels attached to this node.
    pub labels: Vec<String>,
    /// User-visible properties (metadata keys starting with `_` are stripped).
    ///
    /// Keys are interned [`PropertyKey`]s (cheap to clone via `ArcStr`).
    pub properties: HashMap<PropertyKey, Value>,
}

/// An edge extracted from query results (language-agnostic).
#[derive(Debug, Clone)]
pub struct RawEdge {
    /// The edge's internal ID.
    pub id: EdgeId,
    /// The relationship type.
    pub edge_type: String,
    /// Source node ID.
    pub source_id: NodeId,
    /// Target node ID.
    pub target_id: NodeId,
    /// User-visible properties (metadata keys starting with `_` are stripped).
    ///
    /// Keys are interned [`PropertyKey`]s (cheap to clone via `ArcStr`).
    pub properties: HashMap<PropertyKey, Value>,
}

/// Extracts entities from a [`QueryResult`] and maps them to binding-specific
/// types in one pass.
///
/// This is a convenience wrapper around [`extract_entities`] that applies
/// `map_node` and `map_edge` to each extracted [`RawNode`] and [`RawEdge`],
/// avoiding repetitive boilerplate in every language binding.
pub fn extract_and_map<N, E>(
    result: &QueryResult,
    map_node: impl Fn(RawNode) -> N,
    map_edge: impl Fn(RawEdge) -> E,
) -> (Vec<N>, Vec<E>) {
    extract_and_map_rows(result.rows(), &result.cell_types(), map_node, map_edge)
}

/// The nodes and edges of a [`QueryResult`], each once (by ID).
///
/// A value is a node where its cell type ([`QueryResult::cell_type`]) says
/// `Node`, and an edge where it says `Edge`: a returned node or edge, and the
/// ones inside a returned list, map or path (`RETURN [a, r]`, `RETURN {k: a}`,
/// `RETURN p`). A map whose type does not say so is a map, whatever its keys,
/// and the property values of a node or edge are not searched.
///
/// The structural keys (`_id` and `_labels` of a node; `_id`, `_type`, `_source`
/// and `_target` of an edge) are left out of the returned property maps; every
/// other key is a property, one that starts with `_` too (#254).
pub fn extract_entities(result: &QueryResult) -> (Vec<RawNode>, Vec<RawEdge>) {
    extract_entities_from_rows(result.rows(), &result.cell_types())
}

/// [`extract_entities`] of result rows a binding keeps, with their `types`
/// ([`QueryResult::cell_types`]).
///
/// Every node and edge of the rows is copied out of them, so a binding that
/// also hands out the rows can call this when its caller first asks for the
/// nodes or edges instead of with every result: a result read row by row
/// then never pays for the copies.
pub fn extract_entities_from_rows(
    rows: &[Vec<Value>],
    types: &CellTypes,
) -> (Vec<RawNode>, Vec<RawEdge>) {
    let mut found = Found::default();
    for (row, values) in rows.iter().enumerate() {
        for (column, value) in values.iter().enumerate() {
            found.visit(value, types.get(row, column));
        }
    }
    (found.nodes, found.edges)
}

/// [`extract_and_map`] of result rows a binding keeps (see
/// [`extract_entities_from_rows`]).
pub fn extract_and_map_rows<N, E>(
    rows: &[Vec<Value>],
    types: &CellTypes,
    map_node: impl Fn(RawNode) -> N,
    map_edge: impl Fn(RawEdge) -> E,
) -> (Vec<N>, Vec<E>) {
    let (raw_nodes, raw_edges) = extract_entities_from_rows(rows, types);
    let nodes = raw_nodes.into_iter().map(map_node).collect();
    let edges = raw_edges.into_iter().map(map_edge).collect();
    (nodes, edges)
}

/// The nodes and edges found so far, each once.
#[derive(Default)]
struct Found {
    nodes: Vec<RawNode>,
    edges: Vec<RawEdge>,
    seen_node_ids: HashSet<NodeId>,
    seen_edge_ids: HashSet<EdgeId>,
}

impl Found {
    /// Takes the node or edge `value` is where its type `ty` says so, or the
    /// ones in the list, map or path it is, by the types of their values.
    fn visit(&mut self, value: &Value, ty: &LogicalType) {
        match (ty, value) {
            (LogicalType::Node, Value::Map(map)) => self.node(map),
            (LogicalType::Edge, Value::Map(map)) => self.edge(map),
            (LogicalType::List(item), Value::List(items)) => {
                for value in items.iter() {
                    self.visit(value, item);
                }
            }
            (LogicalType::Tuple(types), Value::List(items)) => {
                for (value, ty) in items.iter().zip(types) {
                    self.visit(value, ty);
                }
            }
            (LogicalType::Struct(fields), Value::Map(map)) => {
                for (name, ty) in fields {
                    if let Some(value) = map.get(name.as_str()) {
                        self.visit(value, ty);
                    }
                }
            }
            (LogicalType::Map { value: item, .. }, Value::Map(map)) => {
                for value in map.values() {
                    self.visit(value, item);
                }
            }
            (LogicalType::Path, Value::Path { nodes, edges }) => {
                for node in nodes.iter() {
                    self.visit(node, &LogicalType::Node);
                }
                for edge in edges.iter() {
                    self.visit(edge, &LogicalType::Edge);
                }
            }
            _ => {}
        }
    }

    /// Takes the node record `map`, unless it was taken already.
    fn node(&mut self, map: &BTreeMap<PropertyKey, Value>) {
        let (Some(Value::Int64(id)), Some(Value::List(labels))) =
            (map.get("_id"), map.get("_labels"))
        else {
            return;
        };
        #[allow(
            clippy::cast_sign_loss,
            reason = "the engine stores a u64 id as i64 (`node_to_map`, `edge_to_map`): the cast restores its bits"
        )]
        let node_id = NodeId(*id as u64);
        if self.seen_node_ids.insert(node_id) {
            let labels = labels
                .iter()
                .filter_map(|label| match label {
                    Value::String(label) => Some(label.to_string()),
                    _ => None,
                })
                .collect();
            self.nodes.push(RawNode {
                id: node_id,
                labels,
                properties: user_properties(map, NODE_KEYS),
            });
        }
    }

    /// Takes the edge record `map`, unless it was taken already.
    fn edge(&mut self, map: &BTreeMap<PropertyKey, Value>) {
        let (
            Some(Value::Int64(id)),
            Some(Value::String(edge_type)),
            Some(Value::Int64(src)),
            Some(Value::Int64(dst)),
        ) = (
            map.get("_id"),
            map.get("_type"),
            map.get("_source"),
            map.get("_target"),
        )
        else {
            return;
        };
        #[allow(
            clippy::cast_sign_loss,
            reason = "the engine stores a u64 id as i64 (`node_to_map`, `edge_to_map`): the cast restores its bits"
        )]
        let edge_id = EdgeId(*id as u64);
        if self.seen_edge_ids.insert(edge_id) {
            #[allow(
                clippy::cast_sign_loss,
                reason = "the engine stores a u64 id as i64 (`node_to_map`, `edge_to_map`): the cast restores its bits"
            )]
            self.edges.push(RawEdge {
                id: edge_id,
                edge_type: edge_type.to_string(),
                source_id: NodeId(*src as u64),
                target_id: NodeId(*dst as u64),
                properties: user_properties(map, EDGE_KEYS),
            });
        }
    }
}

/// The cell types of a result as JSON (`grafeo_result_types_json` in C, the
/// `cellTypes` of a WASM result): `{"columns": [T, ...], "runs": [{"from":
/// row, "columns": [T, ...]}, ...]}`, each `T` as [`type_to_json`] writes it.
/// The rows before the first run have `columns`; a run's rows, from its
/// `from` row up to the next run, have its own.
#[must_use]
pub fn cell_types_to_json(types: &CellTypes) -> serde_json::Value {
    let columns = |types: &[LogicalType]| {
        types
            .iter()
            .map(type_to_json)
            .collect::<Vec<serde_json::Value>>()
    };
    serde_json::json!({
        "columns": columns(types.columns()),
        "runs": types
            .runs()
            .iter()
            .map(|(from, types)| serde_json::json!({ "from": from, "columns": columns(types) }))
            .collect::<Vec<_>>(),
    })
}

/// A cell type as JSON, for a binding's cell type accessor: `"node"`,
/// `"edge"` and `"path"`; `{"list": T}`, `{"tuple": [T, ...]}`,
/// `{"struct": {"key": T}}` and `{"map": T}` for the values a list or map
/// holds; `"any"`; the lower-case name of any other type (`"int64"`).
#[must_use]
pub fn type_to_json(ty: &LogicalType) -> serde_json::Value {
    use serde_json::json;
    match ty {
        LogicalType::Node => json!("node"),
        LogicalType::Edge => json!("edge"),
        LogicalType::Path => json!("path"),
        LogicalType::Any => json!("any"),
        LogicalType::List(item) => json!({ "list": type_to_json(item) }),
        LogicalType::Tuple(items) => {
            json!({ "tuple": items.iter().map(type_to_json).collect::<Vec<_>>() })
        }
        LogicalType::Struct(fields) => {
            let fields: serde_json::Map<String, serde_json::Value> = fields
                .iter()
                .map(|(name, ty)| (name.clone(), type_to_json(ty)))
                .collect();
            json!({ "struct": fields })
        }
        LogicalType::Map { value, .. } => json!({ "map": type_to_json(value) }),
        LogicalType::Records(shape) => type_to_json(shape),
        other => json!(other.to_string().to_lowercase()),
    }
}

/// The structural keys of a node map: its id and labels.
const NODE_KEYS: &[&str] = &["_id", "_labels"];

/// The structural keys of an edge map: its id, type and ends.
const EDGE_KEYS: &[&str] = &["_id", "_type", "_source", "_target"];

/// The properties of a node or edge map: every key but its `structural` ones.
fn user_properties(
    map: &BTreeMap<PropertyKey, Value>,
    structural: &[&str],
) -> HashMap<PropertyKey, Value> {
    let mut properties = HashMap::with_capacity(map.len());
    properties.extend(
        map.iter()
            .filter(|(k, _)| !structural.contains(&k.as_str()))
            .map(|(k, v)| (k.clone(), v.clone())),
    );
    properties
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;
    use std::sync::Arc;

    use grafeo_common::types::{PropertyKey, Value};
    use grafeo_engine::database::QueryResult;

    use super::*;

    fn node_map(id: i64, labels: &[&str], props: &[(&str, Value)]) -> Value {
        let mut map = BTreeMap::new();
        map.insert(PropertyKey::new("_id"), Value::Int64(id));
        let label_vals: Vec<Value> = labels.iter().map(|l| Value::String((*l).into())).collect();
        map.insert(PropertyKey::new("_labels"), Value::List(label_vals.into()));
        for (k, v) in props {
            map.insert(PropertyKey::new(*k), v.clone());
        }
        Value::Map(Arc::new(map))
    }

    /// An empty result with `columns` of the cell types `types`.
    fn typed(columns: &[&str], types: Vec<LogicalType>) -> QueryResult {
        let columns = columns.iter().map(|column| (*column).to_string()).collect();
        QueryResult::with_types(columns, types).unwrap()
    }

    fn edge_map(id: i64, edge_type: &str, src: i64, dst: i64, props: &[(&str, Value)]) -> Value {
        let mut map = BTreeMap::new();
        map.insert(PropertyKey::new("_id"), Value::Int64(id));
        map.insert(PropertyKey::new("_type"), Value::String(edge_type.into()));
        map.insert(PropertyKey::new("_source"), Value::Int64(src));
        map.insert(PropertyKey::new("_target"), Value::Int64(dst));
        for (k, v) in props {
            map.insert(PropertyKey::new(*k), v.clone());
        }
        Value::Map(Arc::new(map))
    }

    #[test]
    fn extracts_nodes_and_edges() {
        let mut result = typed(&["n", "e"], vec![LogicalType::Node, LogicalType::Edge]);
        result.push_row(vec![
            node_map(1, &["Person"], &[("name", Value::String("Alix".into()))]),
            edge_map(10, "KNOWS", 1, 2, &[("since", Value::Int64(2020))]),
        ]);
        result.push_row(vec![
            node_map(2, &["Person"], &[("name", Value::String("Gus".into()))]),
            Value::Null,
        ]);

        let (nodes, edges) = extract_entities(&result);

        assert_eq!(nodes.len(), 2);
        assert_eq!(edges.len(), 1);
        assert_eq!(nodes[0].id, NodeId(1));
        assert_eq!(nodes[0].labels, vec!["Person"]);
        assert_eq!(
            nodes[0].properties.get("name"),
            Some(&Value::String("Alix".into()))
        );
        assert!(!nodes[0].properties.contains_key("_id"));
        assert_eq!(edges[0].edge_type, "KNOWS");
        assert_eq!(edges[0].source_id, NodeId(1));
        assert_eq!(edges[0].target_id, NodeId(2));
    }

    /// A property whose name starts with `_` is a property (#254): only the
    /// structural keys of its kind are left out.
    #[test]
    fn keeps_underscore_properties_and_drops_the_structural_keys() {
        let mut result = typed(&["n", "e"], vec![LogicalType::Node, LogicalType::Edge]);
        result.push_row(vec![
            node_map(
                1,
                &["Person"],
                &[
                    ("_created", Value::Int64(2020)),
                    ("_source", Value::String("import".into())),
                ],
            ),
            edge_map(
                10,
                "KNOWS",
                1,
                2,
                &[("_note", Value::String("Paris".into()))],
            ),
        ]);

        let (nodes, edges) = extract_entities(&result);

        let mut node_keys: Vec<&str> = nodes[0]
            .properties
            .keys()
            .map(PropertyKey::as_str)
            .collect();
        node_keys.sort_unstable();
        assert_eq!(node_keys, ["_created", "_source"]);
        assert_eq!(
            nodes[0].properties.get("_source"),
            Some(&Value::String("import".into()))
        );
        let edge_keys: Vec<&str> = edges[0]
            .properties
            .keys()
            .map(PropertyKey::as_str)
            .collect();
        assert_eq!(edge_keys, ["_note"]);
        assert_eq!(
            (edges[0].id, edges[0].source_id, edges[0].target_id),
            (EdgeId(10), NodeId(1), NodeId(2))
        );
    }

    #[test]
    fn deduplicates_by_id() {
        let mut result = typed(&["n"], vec![LogicalType::Node]);
        result.push_row(vec![node_map(1, &["Person"], &[])]);
        result.push_row(vec![node_map(1, &["Person"], &[])]);

        let (nodes, _) = extract_entities(&result);
        assert_eq!(nodes.len(), 1);
    }

    /// Nodes and edges inside a returned list, map or path are the result's
    /// nodes and edges too (`RETURN [a, r]`, `RETURN {k: a}`, `RETURN p`);
    /// the properties of a node are not searched for more.
    #[test]
    fn extracts_nodes_and_edges_inside_lists_maps_and_paths() {
        let alix = node_map(1, &["Person"], &[("name", Value::String("Alix".into()))]);
        let gus = node_map(2, &["Person"], &[("name", Value::String("Gus".into()))]);
        let vincent = node_map(
            3,
            &["Person"],
            &[(
                "friend",
                node_map(19, &["Person"], &[("name", Value::String("Mia".into()))]),
            )],
        );
        let knows = edge_map(10, "KNOWS", 1, 2, &[("since", Value::Int64(2019))]);
        let mut wrapper = BTreeMap::new();
        wrapper.insert(PropertyKey::new("k"), vincent);
        let mut result = typed(
            &["l", "m", "p"],
            vec![
                LogicalType::Tuple(vec![LogicalType::Node, LogicalType::Int64]),
                LogicalType::Struct(vec![("k".into(), LogicalType::Node)]),
                LogicalType::Path,
            ],
        );
        result.push_row(vec![
            Value::List(vec![alix.clone(), Value::Int64(88)].into()),
            Value::Map(Arc::new(wrapper)),
            Value::Path {
                nodes: vec![alix, gus].into(),
                edges: vec![knows].into(),
            },
        ]);

        let (nodes, edges) = extract_entities(&result);

        let mut node_ids: Vec<u64> = nodes.iter().map(|node| node.id.0).collect();
        node_ids.sort_unstable();
        assert_eq!(node_ids, [1, 2, 3], "Mia is a property value, not a node");
        assert_eq!(edges.len(), 1);
        assert_eq!(edges[0].edge_type, "KNOWS");
        assert_eq!(edges[0].source_id, NodeId(1));
    }

    /// A map with the keys of a node or an edge is a map where its type does
    /// not say node or edge: `RETURN {_id: 1, _labels: []}` returns no node.
    #[test]
    fn a_map_with_the_keys_of_a_node_is_a_map() {
        let mut result = typed(
            &["any", "list", "map"],
            vec![
                LogicalType::Any,
                LogicalType::List(Box::new(LogicalType::Any)),
                LogicalType::Struct(vec![("k".into(), LogicalType::Any)]),
            ],
        );
        let fake_node = node_map(1, &[], &[]);
        let fake_edge = edge_map(2, "KNOWS", 1, 3, &[]);
        let mut wrapper = BTreeMap::new();
        wrapper.insert(PropertyKey::new("k"), fake_node.clone());
        result.push_row(vec![
            fake_node.clone(),
            Value::List(vec![fake_node, fake_edge].into()),
            Value::Map(Arc::new(wrapper)),
        ]);

        let (nodes, edges) = extract_entities(&result);

        assert!(nodes.is_empty(), "{nodes:?}");
        assert!(edges.is_empty(), "{edges:?}");
    }

    /// An edge with a property named `_labels` is an edge that keeps the
    /// property (the keys of the map made it a node), and a node with a
    /// property named `_type` is a node that keeps it.
    #[test]
    fn structural_names_of_the_other_kind_are_properties() {
        let mut result = typed(&["n", "e"], vec![LogicalType::Node, LogicalType::Edge]);
        let labels = Value::List(vec![Value::from("Paris")].into());
        result.push_row(vec![
            node_map(1, &["Person"], &[("_type", Value::from("import"))]),
            edge_map(10, "KNOWS", 1, 2, &[("_labels", labels.clone())]),
        ]);

        let (nodes, edges) = extract_entities(&result);

        assert_eq!(nodes.len(), 1);
        assert_eq!(
            nodes[0].properties.get("_type"),
            Some(&Value::from("import"))
        );
        assert_eq!(edges.len(), 1);
        assert_eq!(edges[0].id, EdgeId(10));
        assert_eq!(edges[0].properties.get("_labels"), Some(&labels));
    }

    #[test]
    fn cell_types_as_json() {
        use serde_json::json;
        let node = || Box::new(LogicalType::Node);
        for (ty, expected) in [
            (LogicalType::Node, json!("node")),
            (LogicalType::Records(node()), json!("node")),
            (LogicalType::Edge, json!("edge")),
            (LogicalType::Path, json!("path")),
            (LogicalType::Any, json!("any")),
            (LogicalType::Int64, json!("int64")),
            (LogicalType::List(node()), json!({"list": "node"})),
            (
                LogicalType::Tuple(vec![LogicalType::Node, LogicalType::Int64]),
                json!({"tuple": ["node", "int64"]}),
            ),
            (
                LogicalType::Struct(vec![("k".into(), LogicalType::Edge)]),
                json!({"struct": {"k": "edge"}}),
            ),
            (
                LogicalType::Map {
                    key: Box::new(LogicalType::String),
                    value: node(),
                },
                json!({"map": "node"}),
            ),
        ] {
            assert_eq!(type_to_json(&ty), expected, "{ty:?}");
        }
    }

    #[test]
    fn handles_empty_result() {
        let result = QueryResult::new(vec![]).unwrap();
        let (nodes, edges) = extract_entities(&result);
        assert!(nodes.is_empty());
        assert!(edges.is_empty());
    }
}
