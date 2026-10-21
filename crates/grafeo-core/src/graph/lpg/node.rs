//! The node of the LPG model, as a read returns it.

use arcstr::ArcStr;
use grafeo_common::types::{NodeId, PropertyKey, PropertyMap, Value};
use smallvec::SmallVec;

/// A node with its labels and properties fully loaded.
///
/// This is what you get back from
/// [`GraphStore::get_node`](crate::graph::GraphStore::get_node).
///
/// # Example
///
/// ```
/// use grafeo_core::graph::lpg::Node;
/// use grafeo_common::types::NodeId;
///
/// let mut person = Node::new(NodeId::new(1));
/// person.add_label("Person");
/// person.set_property("name", "Alix");
/// person.set_property("age", 30i64);
///
/// assert!(person.has_label("Person"));
/// ```
#[derive(Debug, Clone)]
pub struct Node {
    /// Unique identifier.
    pub id: NodeId,
    /// Labels attached to this node (inline storage for 1-2 labels).
    pub labels: SmallVec<[ArcStr; 2]>,
    /// Properties stored on this node.
    pub properties: PropertyMap,
}

impl Node {
    /// Creates a new node with the given ID.
    #[must_use]
    pub fn new(id: NodeId) -> Self {
        Self {
            id,
            labels: SmallVec::new(),
            properties: PropertyMap::new(),
        }
    }

    /// Creates a new node with labels.
    #[must_use]
    pub fn with_labels(id: NodeId, labels: impl IntoIterator<Item = impl Into<ArcStr>>) -> Self {
        Self {
            id,
            labels: labels.into_iter().map(Into::into).collect(),
            properties: PropertyMap::new(),
        }
    }

    /// Adds a label to this node.
    pub fn add_label(&mut self, label: impl Into<ArcStr>) {
        let label = label.into();
        if !self.labels.iter().any(|l| l.as_str() == label.as_str()) {
            self.labels.push(label);
        }
    }

    /// Removes a label from this node.
    pub fn remove_label(&mut self, label: &str) -> bool {
        if let Some(pos) = self.labels.iter().position(|l| l.as_str() == label) {
            self.labels.remove(pos);
            true
        } else {
            false
        }
    }

    /// Checks if this node has the given label.
    #[must_use]
    pub fn has_label(&self, label: &str) -> bool {
        self.labels.iter().any(|l| l.as_str() == label)
    }

    /// Sets a property on this node.
    pub fn set_property(&mut self, key: impl Into<PropertyKey>, value: impl Into<Value>) {
        self.properties.insert(key.into(), value.into());
    }

    /// Gets a property from this node.
    #[must_use]
    pub fn get_property(&self, key: &str) -> Option<&Value> {
        self.properties.get(&PropertyKey::new(key))
    }

    /// Removes a property from this node.
    pub fn remove_property(&mut self, key: &str) -> Option<Value> {
        self.properties.remove(&PropertyKey::new(key))
    }

    /// Returns the properties as a `BTreeMap` (for serialization compatibility).
    #[must_use]
    pub fn properties_as_btree(&self) -> std::collections::BTreeMap<PropertyKey, Value> {
        self.properties.to_btree_map()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_node_labels() {
        let mut node = Node::new(NodeId::new(1));

        node.add_label("Person");
        assert!(node.has_label("Person"));
        assert!(!node.has_label("Animal"));

        node.add_label("Employee");
        assert!(node.has_label("Employee"));

        // Adding same label again should be idempotent
        node.add_label("Person");
        assert_eq!(node.labels.len(), 2);

        // Remove label
        assert!(node.remove_label("Person"));
        assert!(!node.has_label("Person"));
        assert!(!node.remove_label("NotExists"));
    }

    #[test]
    fn test_node_properties() {
        let mut node = Node::new(NodeId::new(1));

        node.set_property("name", "Alix");
        node.set_property("age", 30i64);

        assert_eq!(
            node.get_property("name").and_then(|v| v.as_str()),
            Some("Alix")
        );
        assert_eq!(
            node.get_property("age").and_then(|v| v.as_int64()),
            Some(30)
        );
        assert!(node.get_property("missing").is_none());

        let removed = node.remove_property("name");
        assert!(removed.is_some());
        assert!(node.get_property("name").is_none());
    }
}
