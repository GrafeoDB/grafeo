//! Language-agnostic entity extraction from query results.
//!
//! Query results encode nodes and edges as `Value::Map` with metadata markers
//! (`_id`, `_labels`, `_type`, `_source`, `_target`). This module scans result
//! rows and extracts deduplicated [`RawNode`] and [`RawEdge`] structs that each
//! binding can cheaply convert to its language-specific wrapper.

use std::collections::{HashMap, HashSet};

use grafeo_common::types::{EdgeId, NodeId, PropertyKey, Value};
use grafeo_engine::database::QueryResult;

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
    extract_and_map_rows(result.rows(), map_node, map_edge)
}

/// Scans all values in a [`QueryResult`] for maps that look like resolved nodes
/// or edges, deduplicates by ID, and returns the extracted entities.
///
/// A map is treated as a **node** when it contains `_id` (Int64) and `_labels`
/// (List). It is treated as an **edge** when it contains `_id`, `_type`
/// (String), `_source` (Int64), and `_target` (Int64). Nodes and edges inside
/// a returned list, map or path count too (`RETURN [a, r]`, `RETURN {k: a}`,
/// `RETURN p`); the property values of a node or edge are not searched.
///
/// The structural keys (`_id` and `_labels` of a node; `_id`, `_type`, `_source`
/// and `_target` of an edge) are left out of the returned property maps; every
/// other key is a property, one that starts with `_` too (#254).
pub fn extract_entities(result: &QueryResult) -> (Vec<RawNode>, Vec<RawEdge>) {
    extract_entities_from_rows(result.rows())
}

/// [`extract_entities`] of result rows a binding keeps.
///
/// Every node and edge of the rows is copied out of them, so a binding that
/// also hands out the rows can call this when its caller first asks for the
/// nodes or edges instead of with every result: a result read row by row
/// then never pays for the copies.
pub fn extract_entities_from_rows(rows: &[Vec<Value>]) -> (Vec<RawNode>, Vec<RawEdge>) {
    let mut found = Found::default();
    for row in rows {
        for value in row {
            found.visit(value);
        }
    }
    (found.nodes, found.edges)
}

/// [`extract_and_map`] of result rows a binding keeps (see
/// [`extract_entities_from_rows`]).
pub fn extract_and_map_rows<N, E>(
    rows: &[Vec<Value>],
    map_node: impl Fn(RawNode) -> N,
    map_edge: impl Fn(RawEdge) -> E,
) -> (Vec<N>, Vec<E>) {
    let (raw_nodes, raw_edges) = extract_entities_from_rows(rows);
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
    /// Takes the node or edge `value` is, or the ones in the list, map or
    /// path it is.
    fn visit(&mut self, value: &Value) {
        match value {
            Value::Map(map) => {
                // Check for node: has _id and _labels
                if let (Some(Value::Int64(id)), Some(Value::List(labels))) =
                    (map.get("_id"), map.get("_labels"))
                {
                    // reason: ID encoding: i64 <-> u64 round-trip
                    #[allow(clippy::cast_sign_loss)]
                    let node_id = NodeId(*id as u64);
                    if self.seen_node_ids.insert(node_id) {
                        let label_strings: Vec<String> = labels
                            .iter()
                            .filter_map(|v| {
                                if let Value::String(s) = v {
                                    Some(s.to_string())
                                } else {
                                    None
                                }
                            })
                            .collect();
                        self.nodes.push(RawNode {
                            id: node_id,
                            labels: label_strings,
                            properties: user_properties(map, NODE_KEYS),
                        });
                    }
                }
                // Check for edge: has _id, _type, _source, _target
                else if let (
                    Some(Value::Int64(id)),
                    Some(Value::String(edge_type)),
                    Some(Value::Int64(src)),
                    Some(Value::Int64(dst)),
                ) = (
                    map.get("_id"),
                    map.get("_type"),
                    map.get("_source"),
                    map.get("_target"),
                ) {
                    // reason: IDs originate as u64 counters stored in i64; roundtrip is lossless
                    #[allow(clippy::cast_sign_loss)]
                    let edge_id = EdgeId(*id as u64);
                    if self.seen_edge_ids.insert(edge_id) {
                        // reason: value is non-negative by preceding validation
                        #[allow(clippy::cast_sign_loss)]
                        self.edges.push(RawEdge {
                            id: edge_id,
                            edge_type: edge_type.to_string(),
                            source_id: NodeId(*src as u64),
                            target_id: NodeId(*dst as u64),
                            properties: user_properties(map, EDGE_KEYS),
                        });
                    }
                }
                // Any other map: the nodes and edges among its values
                else {
                    for value in map.values() {
                        self.visit(value);
                    }
                }
            }
            Value::List(items) => {
                for item in items.iter() {
                    self.visit(item);
                }
            }
            Value::Path { nodes, edges } => {
                for item in nodes.iter().chain(edges.iter()) {
                    self.visit(item);
                }
            }
            _ => {}
        }
    }
}

/// The structural keys of a node map: its id and labels.
const NODE_KEYS: &[&str] = &["_id", "_labels"];

/// The structural keys of an edge map: its id, type and ends.
const EDGE_KEYS: &[&str] = &["_id", "_type", "_source", "_target"];

/// The properties of a node or edge map: every key but its `structural` ones.
fn user_properties(
    map: &std::collections::BTreeMap<PropertyKey, Value>,
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
        let mut result = QueryResult::new(vec!["n".into(), "e".into()]).unwrap();
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
        let mut result = QueryResult::new(vec!["n".into(), "e".into()]).unwrap();
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
        let mut result = QueryResult::new(vec!["n".into()]).unwrap();
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
        let mut result = QueryResult::new(vec!["l".into(), "m".into(), "p".into()]).unwrap();
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

    #[test]
    fn handles_empty_result() {
        let result = QueryResult::new(vec![]).unwrap();
        let (nodes, edges) = extract_entities(&result);
        assert!(nodes.is_empty());
        assert!(edges.is_empty());
    }
}
