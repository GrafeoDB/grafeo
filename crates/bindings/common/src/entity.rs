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
    let (raw_nodes, raw_edges) = extract_entities(result);
    let nodes = raw_nodes.into_iter().map(map_node).collect();
    let edges = raw_edges.into_iter().map(map_edge).collect();
    (nodes, edges)
}

/// Scans all values in a [`QueryResult`] for maps that look like resolved nodes
/// or edges, deduplicates by ID, and returns the extracted entities.
///
/// A map is treated as a **node** when it contains `_id` (Int64) and `_labels`
/// (List). It is treated as an **edge** when it contains `_id`, `_type`
/// (String), `_source` (Int64), and `_target` (Int64).
///
/// Properties whose key starts with `_` are considered internal metadata and are
/// excluded from the returned property maps.
pub fn extract_entities(result: &QueryResult) -> (Vec<RawNode>, Vec<RawEdge>) {
    let mut nodes = Vec::new();
    let mut edges = Vec::new();
    let mut seen_node_ids = HashSet::new();
    let mut seen_edge_ids = HashSet::new();

    for row in result.rows() {
        for value in row {
            if let Value::Map(map) = value {
                // Check for node: has _id and _labels
                if let (Some(Value::Int64(id)), Some(Value::List(labels))) =
                    (map.get("_id"), map.get("_labels"))
                {
                    // reason: ID encoding: i64 <-> u64 round-trip
                    #[allow(clippy::cast_sign_loss)]
                    let node_id = NodeId(*id as u64);
                    if seen_node_ids.insert(node_id) {
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
                        let properties: HashMap<PropertyKey, Value> = map
                            .iter()
                            .filter(|(k, _)| !k.as_str().starts_with('_'))
                            .map(|(k, v)| (k.clone(), v.clone()))
                            .collect();
                        nodes.push(RawNode {
                            id: node_id,
                            labels: label_strings,
                            properties,
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
                    if seen_edge_ids.insert(edge_id) {
                        let properties: HashMap<PropertyKey, Value> = map
                            .iter()
                            .filter(|(k, _)| !k.as_str().starts_with('_'))
                            .map(|(k, v)| (k.clone(), v.clone()))
                            .collect();
                        // reason: value is non-negative by preceding validation
                        #[allow(clippy::cast_sign_loss)]
                        edges.push(RawEdge {
                            id: edge_id,
                            edge_type: edge_type.to_string(),
                            source_id: NodeId(*src as u64),
                            target_id: NodeId(*dst as u64),
                            properties,
                        });
                    }
                }
            }
        }
    }

    (nodes, edges)
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

    #[test]
    fn deduplicates_by_id() {
        let mut result = QueryResult::new(vec!["n".into()]).unwrap();
        result.push_row(vec![node_map(1, &["Person"], &[])]);
        result.push_row(vec![node_map(1, &["Person"], &[])]);

        let (nodes, _) = extract_entities(&result);
        assert_eq!(nodes.len(), 1);
    }

    #[test]
    fn handles_empty_result() {
        let result = QueryResult::new(vec![]).unwrap();
        let (nodes, edges) = extract_entities(&result);
        assert!(nodes.is_empty());
        assert!(edges.is_empty());
    }
}
