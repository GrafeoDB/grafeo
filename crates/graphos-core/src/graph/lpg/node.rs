//! Node types for the LPG model.

use std::collections::BTreeMap;
use std::sync::Arc;

use graphos_common::types::{EpochId, NodeId, PropertyKey, Value};

/// A node in the labeled property graph.
///
/// This is the high-level representation of a node with all its data.
#[derive(Debug, Clone)]
pub struct Node {
    /// Unique identifier.
    pub id: NodeId,
    /// Labels attached to this node.
    pub labels: Vec<Arc<str>>,
    /// Properties stored on this node.
    pub properties: BTreeMap<PropertyKey, Value>,
}

impl Node {
    /// Creates a new node with the given ID.
    #[must_use]
    pub fn new(id: NodeId) -> Self {
        Self {
            id,
            labels: Vec::new(),
            properties: BTreeMap::new(),
        }
    }

    /// Creates a new node with labels.
    #[must_use]
    pub fn with_labels(id: NodeId, labels: impl IntoIterator<Item = impl Into<Arc<str>>>) -> Self {
        Self {
            id,
            labels: labels.into_iter().map(Into::into).collect(),
            properties: BTreeMap::new(),
        }
    }

    /// Adds a label to this node.
    pub fn add_label(&mut self, label: impl Into<Arc<str>>) {
        let label = label.into();
        if !self.labels.iter().any(|l| l.as_ref() == label.as_ref()) {
            self.labels.push(label);
        }
    }

    /// Removes a label from this node.
    pub fn remove_label(&mut self, label: &str) -> bool {
        if let Some(pos) = self.labels.iter().position(|l| l.as_ref() == label) {
            self.labels.remove(pos);
            true
        } else {
            false
        }
    }

    /// Checks if this node has the given label.
    #[must_use]
    pub fn has_label(&self, label: &str) -> bool {
        self.labels.iter().any(|l| l.as_ref() == label)
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

/// The compact, cache-line friendly representation of a node.
///
/// This struct is exactly 32 bytes and is used for the primary node storage.
/// Properties are stored separately in columnar format.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct NodeRecord {
    /// Unique node identifier.
    pub id: NodeId,
    /// Bitmap of label IDs (supports up to 64 labels).
    pub label_bits: u64,
    /// Offset into the property arena.
    pub props_offset: u32,
    /// Number of properties.
    pub props_count: u16,
    /// Flags (deleted, has_version, etc.).
    pub flags: NodeFlags,
    /// Epoch this record was created in.
    pub epoch: EpochId,
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
            label_bits: 0,
            props_offset: 0,
            props_count: 0,
            flags: NodeFlags(0),
            epoch,
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

    /// Checks if this node has a label by its bit index.
    #[must_use]
    pub const fn has_label_bit(&self, bit: u8) -> bool {
        if bit >= 64 {
            return false;
        }
        (self.label_bits & (1 << bit)) != 0
    }

    /// Sets a label bit.
    pub fn set_label_bit(&mut self, bit: u8) {
        if bit < 64 {
            self.label_bits |= 1 << bit;
        }
    }

    /// Clears a label bit.
    pub fn clear_label_bit(&mut self, bit: u8) {
        if bit < 64 {
            self.label_bits &= !(1 << bit);
        }
    }

    /// Returns an iterator over the set label bits.
    pub fn label_bits_iter(&self) -> impl Iterator<Item = u8> + '_ {
        (0..64).filter(|&bit| self.has_label_bit(bit))
    }
}

/// Flags for a node record.
#[repr(transparent)]
#[derive(Debug, Clone, Copy, Default)]
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
    fn test_node_record_label_bits() {
        let mut record = NodeRecord::new(NodeId::new(1), EpochId::INITIAL);

        assert!(!record.has_label_bit(0));
        record.set_label_bit(0);
        assert!(record.has_label_bit(0));

        record.set_label_bit(5);
        record.set_label_bit(63);
        assert!(record.has_label_bit(5));
        assert!(record.has_label_bit(63));

        let bits: Vec<_> = record.label_bits_iter().collect();
        assert_eq!(bits, vec![0, 5, 63]);

        record.clear_label_bit(5);
        assert!(!record.has_label_bit(5));
    }
}
