//! Node types for the LPG model.
//!
//! Two representations here: [`Node`] is the friendly one with all the data,
//! [`NodeRecord`] is the compact 32-byte struct for storage.

use std::collections::BTreeMap;

use arcstr::ArcStr;
use grafeo_common::types::{EpochId, NodeId, PropertyKey, Value};
use serde::{Deserialize, Serialize};
use smallvec::SmallVec;

/// A node with its labels and properties fully loaded.
///
/// This is what you get back from [`LpgStore::get_node()`](super::LpgStore::get_node).
/// For bulk operations, the store works with [`NodeRecord`] internally.
///
/// # Example
///
/// ```
/// use grafeo_core::graph::lpg::Node;
/// use grafeo_common::types::NodeId;
///
/// let mut person = Node::new(NodeId::new(1));
/// person.add_label("Person");
/// person.set_property("name", "Alice");
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
    pub properties: BTreeMap<PropertyKey, Value>,
}

impl Node {
    /// Creates a new node with the given ID.
    #[must_use]
    pub fn new(id: NodeId) -> Self {
        Self {
            id,
            labels: SmallVec::new(),
            properties: BTreeMap::new(),
        }
    }

    /// Creates a new node with labels.
    #[must_use]
    pub fn with_labels(id: NodeId, labels: impl IntoIterator<Item = impl Into<ArcStr>>) -> Self {
        Self {
            id,
            labels: labels.into_iter().map(Into::into).collect(),
            properties: BTreeMap::new(),
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
}

/// The compact storage format for a node - exactly 32 bytes.
///
/// You won't interact with this directly most of the time. It's what lives
/// in memory for each node, with properties and labels stored separately.
/// The 32-byte size means two records fit in a cache line.
///
/// Fields are ordered to minimize padding: u64s first, then u32, then u16s.
#[repr(C)]
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct NodeRecord {
    /// Unique node identifier.
    pub id: NodeId,
    /// Epoch this record was created in.
    pub epoch: EpochId,
    /// Offset into the property arena.
    pub props_offset: u32,
    /// Number of labels on this node (labels stored externally).
    pub label_count: u16,
    /// Reserved for future use / alignment.
    pub _reserved: u16,
    /// Number of properties.
    pub props_count: u16,
    /// Flags (deleted, has_version, etc.).
    pub flags: NodeFlags,
    /// Padding to maintain 32-byte size.
    pub _padding: u32,
}

impl NodeRecord {
    /// Flag indicating the node is deleted.
    pub const FLAG_DELETED: u16 = 1 << 0;
    /// Flag indicating the node has version history.
    pub const FLAG_HAS_VERSION: u16 = 1 << 1;

    /// Creates a new node record.
    #[must_use]
    pub const fn new(id: NodeId, epoch: EpochId) -> Self {
        Self {
            id,
            label_count: 0,
            _reserved: 0,
            props_offset: 0,
            props_count: 0,
            flags: NodeFlags(0),
            epoch,
            _padding: 0,
        }
    }

    /// Checks if this node is deleted.
    #[must_use]
    pub const fn is_deleted(&self) -> bool {
        self.flags.contains(Self::FLAG_DELETED)
    }

    /// Marks this node as deleted.
    pub fn set_deleted(&mut self, deleted: bool) {
        if deleted {
            self.flags.set(Self::FLAG_DELETED);
        } else {
            self.flags.clear(Self::FLAG_DELETED);
        }
    }

    /// Returns the number of labels on this node.
    #[must_use]
    pub const fn label_count(&self) -> u16 {
        self.label_count
    }

    /// Sets the label count.
    pub fn set_label_count(&mut self, count: u16) {
        self.label_count = count;
    }
}

/// Bit flags packed into a node record.
///
/// Check flags with [`contains()`](Self::contains), set with [`set()`](Self::set).
#[repr(transparent)]
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize)]
pub struct NodeFlags(pub u16);

impl NodeFlags {
    /// Checks if a flag is set.
    #[must_use]
    pub const fn contains(&self, flag: u16) -> bool {
        (self.0 & flag) != 0
    }

    /// Sets a flag.
    pub fn set(&mut self, flag: u16) {
        self.0 |= flag;
    }

    /// Clears a flag.
    pub fn clear(&mut self, flag: u16) {
        self.0 &= !flag;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_node_record_size() {
        // Ensure NodeRecord is exactly 32 bytes
        assert_eq!(std::mem::size_of::<NodeRecord>(), 32);
    }

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

        node.set_property("name", "Alice");
        node.set_property("age", 30i64);

        assert_eq!(
            node.get_property("name").and_then(|v| v.as_str()),
            Some("Alice")
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

    #[test]
    fn test_node_record_flags() {
        let mut record = NodeRecord::new(NodeId::new(1), EpochId::INITIAL);

        assert!(!record.is_deleted());
        record.set_deleted(true);
        assert!(record.is_deleted());
        record.set_deleted(false);
        assert!(!record.is_deleted());
    }

    #[test]
    fn test_node_record_label_count() {
        let mut record = NodeRecord::new(NodeId::new(1), EpochId::INITIAL);

        assert_eq!(record.label_count(), 0);
        record.set_label_count(5);
        assert_eq!(record.label_count(), 5);

        // Can handle large label counts (no 64 limit)
        record.set_label_count(1000);
        assert_eq!(record.label_count(), 1000);
    }
}
