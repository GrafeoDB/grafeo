//! The edge of the LPG model, as a read returns it.

use arcstr::ArcStr;
use grafeo_common::types::{EdgeId, NodeId, PropertyKey, PropertyMap, Value};

/// A relationship between two nodes, with a type and optional properties.
///
/// Think of edges as the "verbs" in your graph - KNOWS, WORKS_AT, PURCHASED.
/// Each edge connects exactly one source node to one destination node.
///
/// # Example
///
/// ```
/// use grafeo_core::graph::lpg::Edge;
/// use grafeo_common::types::{EdgeId, NodeId};
///
/// let mut works_at = Edge::new(
///     EdgeId::new(1),
///     NodeId::new(10),  // Alix
///     NodeId::new(20),  // Acme Corp
///     "WORKS_AT"
/// );
/// works_at.set_property("since", 2020i64);
/// works_at.set_property("role", "Engineer");
/// ```
#[derive(Debug, Clone)]
pub struct Edge {
    /// Unique identifier.
    pub id: EdgeId,
    /// Source node ID.
    pub src: NodeId,
    /// Destination node ID.
    pub dst: NodeId,
    /// Edge type/label.
    pub edge_type: ArcStr,
    /// Properties stored on this edge.
    pub properties: PropertyMap,
}

impl Edge {
    /// Creates a new edge.
    #[must_use]
    pub fn new(id: EdgeId, src: NodeId, dst: NodeId, edge_type: impl Into<ArcStr>) -> Self {
        Self {
            id,
            src,
            dst,
            edge_type: edge_type.into(),
            properties: PropertyMap::new(),
        }
    }

    /// Sets a property on this edge.
    pub fn set_property(&mut self, key: impl Into<PropertyKey>, value: impl Into<Value>) {
        self.properties.insert(key.into(), value.into());
    }

    /// Gets a property from this edge.
    #[must_use]
    pub fn get_property(&self, key: &str) -> Option<&Value> {
        self.properties.get(&PropertyKey::new(key))
    }

    /// Removes a property from this edge.
    pub fn remove_property(&mut self, key: &str) -> Option<Value> {
        self.properties.remove(&PropertyKey::new(key))
    }

    /// Returns the properties as a `BTreeMap` (for serialization compatibility).
    #[must_use]
    pub fn properties_as_btree(&self) -> std::collections::BTreeMap<PropertyKey, Value> {
        self.properties.to_btree_map()
    }

    /// Given one endpoint, returns the other end of this edge.
    ///
    /// Handy in traversals when you have a node and edge but need the neighbor.
    /// Returns `None` if `node` isn't connected to this edge.
    #[must_use]
    pub fn other_endpoint(&self, node: NodeId) -> Option<NodeId> {
        if node == self.src {
            Some(self.dst)
        } else if node == self.dst {
            Some(self.src)
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_edge_creation() {
        let edge = Edge::new(EdgeId::new(1), NodeId::new(10), NodeId::new(20), "KNOWS");

        assert_eq!(edge.id, EdgeId::new(1));
        assert_eq!(edge.src, NodeId::new(10));
        assert_eq!(edge.dst, NodeId::new(20));
        assert_eq!(edge.edge_type.as_str(), "KNOWS");
    }

    #[test]
    fn test_edge_properties() {
        let mut edge = Edge::new(EdgeId::new(1), NodeId::new(10), NodeId::new(20), "KNOWS");

        edge.set_property("since", 2020i64);
        edge.set_property("weight", 1.5f64);

        assert_eq!(
            edge.get_property("since").and_then(|v| v.as_int64()),
            Some(2020)
        );
        assert_eq!(
            edge.get_property("weight").and_then(|v| v.as_float64()),
            Some(1.5)
        );
    }

    #[test]
    fn test_edge_other_endpoint() {
        let edge = Edge::new(EdgeId::new(1), NodeId::new(10), NodeId::new(20), "KNOWS");

        assert_eq!(edge.other_endpoint(NodeId::new(10)), Some(NodeId::new(20)));
        assert_eq!(edge.other_endpoint(NodeId::new(20)), Some(NodeId::new(10)));
        assert_eq!(edge.other_endpoint(NodeId::new(30)), None);
    }
}
