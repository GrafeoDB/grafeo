//! Graph export serializers (GEXF, GraphML).
//!
//! Provides streaming serializers that write graph data directly to a [`std::io::Write`] sink.
//! No external XML library is needed: both formats are simple enough for `write!()` macros
//! with proper escaping.

pub mod gexf;
pub mod graphml;

use std::collections::BTreeMap;
use std::io;

use grafeo_common::PropertyKey;
use grafeo_common::types::Value;
use grafeo_core::graph::lpg::{Edge, Node};

/// Errors from graph export operations.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum ExportError {
    /// I/O error while writing output.
    #[error("I/O error: {0}")]
    Io(#[from] io::Error),
}

/// Escapes XML special characters in text content and attribute values.
#[must_use]
pub fn escape_xml(s: &str) -> String {
    let mut result = String::with_capacity(s.len());
    for ch in s.chars() {
        match ch {
            '&' => result.push_str("&amp;"),
            '<' => result.push_str("&lt;"),
            '>' => result.push_str("&gt;"),
            '"' => result.push_str("&quot;"),
            '\'' => result.push_str("&apos;"),
            _ => result.push(ch),
        }
    }
    result
}

/// Maps a grafeo [`Value`] to a GEXF attribute type string.
#[must_use]
pub fn value_to_gexf_type(value: &Value) -> &'static str {
    match value {
        Value::Int64(_) => "integer",
        Value::Float64(_) => "float",
        Value::Bool(_) => "boolean",
        Value::String(_) => "string",
        Value::Date(_) => "date",
        _ => "string",
    }
}

/// Maps a grafeo [`Value`] to a GraphML attribute type string.
#[must_use]
pub fn value_to_graphml_type(value: &Value) -> &'static str {
    match value {
        Value::Int64(_) => "long",
        Value::Float64(_) => "double",
        Value::Bool(_) => "boolean",
        Value::String(_) => "string",
        _ => "string",
    }
}

/// Converts a [`Value`] to an XML-safe string representation.
///
/// Returns `None` for `Value::Null` (callers should omit the element).
#[must_use]
pub fn value_to_xml_string(value: &Value) -> Option<String> {
    match value {
        Value::Null => None,
        Value::Bool(b) => Some(b.to_string()),
        Value::Int64(i) => Some(i.to_string()),
        Value::Float64(f) => Some(f.to_string()),
        Value::String(s) => Some(escape_xml(s.as_str())),
        Value::Date(d) => Some(d.to_string()),
        Value::Time(t) => Some(t.to_string()),
        Value::Timestamp(ts) => Some(ts.to_string()),
        Value::Duration(d) => Some(d.to_string()),
        Value::ZonedDatetime(zdt) => Some(zdt.to_string()),
        Value::Bytes(b) => {
            // Hex-encode binary data
            use std::fmt::Write;
            let hex = b.iter().fold(String::new(), |mut acc, byte| {
                let _ = write!(acc, "{byte:02x}");
                acc
            });
            Some(hex)
        }
        Value::Vector(v) => {
            let parts: Vec<String> = v.iter().map(|f| f.to_string()).collect();
            Some(parts.join(","))
        }
        Value::List(items) => {
            let parts: Vec<String> = items.iter().filter_map(value_to_xml_string).collect();
            Some(parts.join(","))
        }
        Value::Map(m) => {
            // Serialize as key=value pairs
            let parts: Vec<String> = m
                .iter()
                .map(|(k, v)| {
                    let val_str = value_to_xml_string(v).unwrap_or_default();
                    format!("{}={}", escape_xml(k.as_str()), val_str)
                })
                .collect();
            Some(parts.join(";"))
        }
        Value::Path { .. } | Value::GCounter(_) | Value::OnCounter { .. } => {
            Some(escape_xml(&value.to_string()))
        }
        _ => Some(escape_xml(&value.to_string())),
    }
}

/// Discovered property schema: maps property key to (attribute ID, GEXF/GraphML type).
pub(crate) type PropertySchema = BTreeMap<PropertyKey, (usize, &'static str)>;

/// Discovers the property schema for nodes by scanning all property keys and their types.
pub(crate) fn discover_node_schema<F>(nodes: &[Node], type_fn: F) -> PropertySchema
where
    F: Fn(&Value) -> &'static str,
{
    let mut schema: BTreeMap<PropertyKey, Option<&'static str>> = BTreeMap::new();
    for node in nodes {
        for (key, value) in node.properties.iter() {
            schema
                .entry(key.clone())
                .and_modify(|existing| {
                    if existing.is_none() && !value.is_null() {
                        *existing = Some(type_fn(value));
                    }
                })
                .or_insert_with(|| {
                    if value.is_null() {
                        None
                    } else {
                        Some(type_fn(value))
                    }
                });
        }
    }
    schema
        .into_iter()
        .enumerate()
        .map(|(idx, (key, type_str))| (key, (idx, type_str.unwrap_or("string"))))
        .collect()
}

/// Discovers the property schema for edges by scanning all property keys and their types.
pub(crate) fn discover_edge_schema<F>(edges: &[Edge], type_fn: F) -> PropertySchema
where
    F: Fn(&Value) -> &'static str,
{
    let mut schema: BTreeMap<PropertyKey, Option<&'static str>> = BTreeMap::new();
    for edge in edges {
        for (key, value) in edge.properties.iter() {
            schema
                .entry(key.clone())
                .and_modify(|existing| {
                    if existing.is_none() && !value.is_null() {
                        *existing = Some(type_fn(value));
                    }
                })
                .or_insert_with(|| {
                    if value.is_null() {
                        None
                    } else {
                        Some(type_fn(value))
                    }
                });
        }
    }
    schema
        .into_iter()
        .enumerate()
        .map(|(idx, (key, type_str))| (key, (idx, type_str.unwrap_or("string"))))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_escape_xml_basic() {
        assert_eq!(escape_xml("hello"), "hello");
        assert_eq!(escape_xml("a & b"), "a &amp; b");
        assert_eq!(escape_xml("<tag>"), "&lt;tag&gt;");
        assert_eq!(escape_xml("she said \"hi\""), "she said &quot;hi&quot;");
        assert_eq!(escape_xml("it's"), "it&apos;s");
    }

    #[test]
    fn test_escape_xml_combined() {
        assert_eq!(
            escape_xml("<a href=\"x&y\">"),
            "&lt;a href=&quot;x&amp;y&quot;&gt;"
        );
    }

    #[test]
    fn test_value_to_xml_string_null() {
        assert!(value_to_xml_string(&Value::Null).is_none());
    }

    #[test]
    fn test_value_to_xml_string_primitives() {
        assert_eq!(value_to_xml_string(&Value::Bool(true)).unwrap(), "true");
        assert_eq!(value_to_xml_string(&Value::Int64(42)).unwrap(), "42");
        assert_eq!(
            value_to_xml_string(&Value::Float64(3.125)).unwrap(),
            "3.125"
        );
        assert_eq!(
            value_to_xml_string(&Value::String("Alix & Gus".into())).unwrap(),
            "Alix &amp; Gus"
        );
    }

    #[test]
    fn test_value_to_xml_string_vector() {
        let v = Value::Vector(std::sync::Arc::from(vec![1.0f32, 2.0, 3.0].as_slice()));
        assert_eq!(value_to_xml_string(&v).unwrap(), "1,2,3");
    }

    #[test]
    fn test_gexf_type_mapping() {
        assert_eq!(value_to_gexf_type(&Value::Int64(0)), "integer");
        assert_eq!(value_to_gexf_type(&Value::Float64(0.0)), "float");
        assert_eq!(value_to_gexf_type(&Value::Bool(true)), "boolean");
        assert_eq!(value_to_gexf_type(&Value::String("".into())), "string");
    }

    #[test]
    fn test_graphml_type_mapping() {
        assert_eq!(value_to_graphml_type(&Value::Int64(0)), "long");
        assert_eq!(value_to_graphml_type(&Value::Float64(0.0)), "double");
        assert_eq!(value_to_graphml_type(&Value::Bool(true)), "boolean");
        assert_eq!(value_to_graphml_type(&Value::String("".into())), "string");
    }

    #[test]
    fn test_value_to_xml_string_bytes() {
        let v = Value::Bytes(std::sync::Arc::from(
            vec![0xDE, 0xAD, 0xBE, 0xEF].as_slice(),
        ));
        assert_eq!(value_to_xml_string(&v).unwrap(), "deadbeef");
    }

    #[test]
    fn test_value_to_xml_string_bytes_empty() {
        let v = Value::Bytes(std::sync::Arc::from(Vec::<u8>::new().as_slice()));
        assert_eq!(value_to_xml_string(&v).unwrap(), "");
    }

    #[test]
    fn test_value_to_xml_string_date() {
        use grafeo_common::types::Date;
        let date = Date::from_ymd(2025, 6, 15).unwrap();
        let v = Value::Date(date);
        let result = value_to_xml_string(&v).unwrap();
        assert!(
            result.contains("2025"),
            "date should contain the year: {result}"
        );
    }

    #[test]
    fn test_value_to_xml_string_time() {
        use grafeo_common::types::Time;
        let time = Time::from_hms(14, 30, 0).unwrap();
        let v = Value::Time(time);
        let result = value_to_xml_string(&v).unwrap();
        assert!(
            result.contains("14"),
            "time should contain the hour: {result}"
        );
    }

    #[test]
    fn test_value_to_xml_string_timestamp() {
        use grafeo_common::types::Timestamp;
        let ts = Timestamp::from_micros(1_000_000);
        let v = Value::Timestamp(ts);
        let result = value_to_xml_string(&v);
        assert!(result.is_some());
    }

    #[test]
    fn test_value_to_xml_string_duration() {
        use grafeo_common::types::Duration;
        let dur = Duration::new(2, 5, 0);
        let v = Value::Duration(dur);
        let result = value_to_xml_string(&v).unwrap();
        assert!(!result.is_empty(), "result is empty");
    }

    #[test]
    fn test_value_to_xml_string_zoned_datetime() {
        use grafeo_common::types::{Timestamp, ZonedDatetime};
        let zdt = ZonedDatetime::from_timestamp_offset(Timestamp::from_micros(0), 3600);
        let v = Value::ZonedDatetime(zdt);
        let result = value_to_xml_string(&v).unwrap();
        assert!(!result.is_empty(), "result is empty");
    }

    #[test]
    fn test_value_to_xml_string_list() {
        let items = vec![
            Value::Int64(1),
            Value::Int64(2),
            Value::Null,
            Value::Int64(3),
        ];
        let v = Value::List(std::sync::Arc::from(items.as_slice()));
        // Null is filtered out by value_to_xml_string returning None
        assert_eq!(value_to_xml_string(&v).unwrap(), "1,2,3");
    }

    #[test]
    fn test_value_to_xml_string_list_empty() {
        let v = Value::List(std::sync::Arc::from(Vec::<Value>::new().as_slice()));
        assert_eq!(value_to_xml_string(&v).unwrap(), "");
    }

    #[test]
    fn test_value_to_xml_string_map() {
        let mut map = BTreeMap::new();
        map.insert(PropertyKey::from("city"), Value::String("Amsterdam".into()));
        map.insert(PropertyKey::from("pop"), Value::Int64(900_000));
        let v = Value::Map(std::sync::Arc::new(map));
        let result = value_to_xml_string(&v).unwrap();
        // BTreeMap is sorted, so "city" comes before "pop"
        assert!(result.contains("city=Amsterdam"));
        assert!(result.contains("pop=900000"));
        assert!(result.contains(';'));
    }

    #[test]
    fn test_value_to_xml_string_map_with_null_value() {
        let mut map = BTreeMap::new();
        map.insert(PropertyKey::from("key"), Value::Null);
        let v = Value::Map(std::sync::Arc::new(map));
        let result = value_to_xml_string(&v).unwrap();
        // Null -> unwrap_or_default -> empty string
        assert_eq!(result, "key=");
    }

    #[test]
    fn test_value_to_xml_string_map_with_special_chars() {
        let mut map = BTreeMap::new();
        map.insert(PropertyKey::from("k&ey"), Value::String("<val>".into()));
        let v = Value::Map(std::sync::Arc::new(map));
        let result = value_to_xml_string(&v).unwrap();
        assert!(result.contains("k&amp;ey=&lt;val&gt;"));
    }

    #[test]
    fn test_gexf_type_date_variant() {
        use grafeo_common::types::Date;
        let date = Date::from_ymd(2025, 1, 1).unwrap();
        assert_eq!(value_to_gexf_type(&Value::Date(date)), "date");
    }

    #[test]
    fn test_gexf_type_fallback_to_string() {
        let v = Value::Bytes(std::sync::Arc::from(vec![1u8].as_slice()));
        assert_eq!(value_to_gexf_type(&v), "string");
    }

    #[test]
    fn test_graphml_type_fallback_to_string() {
        use grafeo_common::types::Duration;
        let dur = Duration::new(0, 0, 0);
        assert_eq!(value_to_graphml_type(&Value::Duration(dur)), "string");
    }

    #[test]
    fn test_discover_node_schema_multiple_nodes() {
        use grafeo_common::types::NodeId;
        use grafeo_core::graph::lpg::Node;

        let mut n1 = Node::new(NodeId(1));
        n1.set_property("name", Value::String("Alix".into()));
        n1.set_property("age", Value::Int64(30));

        let mut n2 = Node::new(NodeId(2));
        n2.set_property("name", Value::String("Gus".into()));
        n2.set_property("score", Value::Float64(9.5));

        let schema = discover_node_schema(&[n1, n2], value_to_gexf_type);
        assert_eq!(schema.len(), 3); // name, age, score
        assert_eq!(schema[&PropertyKey::from("name")].1, "string");
        assert_eq!(schema[&PropertyKey::from("age")].1, "integer");
        assert_eq!(schema[&PropertyKey::from("score")].1, "float");
    }

    #[test]
    fn test_discover_node_schema_null_then_typed() {
        use grafeo_common::types::NodeId;
        use grafeo_core::graph::lpg::Node;

        // First node has null for "age", second node has a typed value
        let mut n1 = Node::new(NodeId(1));
        n1.set_property("age", Value::Null);

        let mut n2 = Node::new(NodeId(2));
        n2.set_property("age", Value::Int64(25));

        let schema = discover_node_schema(&[n1, n2], value_to_gexf_type);
        // The type should be resolved from the second node
        assert_eq!(schema[&PropertyKey::from("age")].1, "integer");
    }

    #[test]
    fn test_discover_node_schema_all_null_falls_back_to_string() {
        use grafeo_common::types::NodeId;
        use grafeo_core::graph::lpg::Node;

        let mut n1 = Node::new(NodeId(1));
        n1.set_property("unknown", Value::Null);

        let schema = discover_node_schema(&[n1], value_to_gexf_type);
        // All null, should fall back to "string"
        assert_eq!(schema[&PropertyKey::from("unknown")].1, "string");
    }

    #[test]
    fn test_discover_node_schema_empty() {
        let schema = discover_node_schema(&[], value_to_gexf_type);
        assert!(schema.is_empty());
    }

    #[test]
    fn test_discover_edge_schema_multiple_edges() {
        use grafeo_common::types::{EdgeId, NodeId};
        use grafeo_core::graph::lpg::Edge;

        let mut e1 = Edge::new(EdgeId(1), NodeId(1), NodeId(2), "KNOWS");
        e1.set_property("since", Value::Int64(2020));

        let mut e2 = Edge::new(EdgeId(2), NodeId(2), NodeId(3), "FOLLOWS");
        e2.set_property("weight", Value::Float64(0.8));

        let schema = discover_edge_schema(&[e1, e2], value_to_graphml_type);
        assert_eq!(schema.len(), 2);
        assert_eq!(schema[&PropertyKey::from("since")].1, "long");
        assert_eq!(schema[&PropertyKey::from("weight")].1, "double");
    }

    #[test]
    fn test_discover_edge_schema_null_then_typed() {
        use grafeo_common::types::{EdgeId, NodeId};
        use grafeo_core::graph::lpg::Edge;

        let mut e1 = Edge::new(EdgeId(1), NodeId(1), NodeId(2), "KNOWS");
        e1.set_property("weight", Value::Null);

        let mut e2 = Edge::new(EdgeId(2), NodeId(2), NodeId(3), "KNOWS");
        e2.set_property("weight", Value::Float64(1.5));

        let schema = discover_edge_schema(&[e1, e2], value_to_graphml_type);
        assert_eq!(schema[&PropertyKey::from("weight")].1, "double");
    }

    #[test]
    fn test_discover_edge_schema_empty() {
        let schema = discover_edge_schema(&[], value_to_graphml_type);
        assert!(schema.is_empty());
    }

    #[test]
    fn test_discover_node_schema_ids_are_sequential() {
        use grafeo_common::types::NodeId;
        use grafeo_core::graph::lpg::Node;

        let mut n1 = Node::new(NodeId(1));
        n1.set_property("a", Value::Int64(1));
        n1.set_property("b", Value::Bool(true));
        n1.set_property("c", Value::Float64(1.0));

        let schema = discover_node_schema(&[n1], value_to_gexf_type);
        // IDs should be assigned sequentially from BTreeMap iteration (alphabetical)
        let ids: Vec<usize> = schema.values().map(|(id, _)| *id).collect();
        assert_eq!(ids, vec![0, 1, 2]);
    }
}
