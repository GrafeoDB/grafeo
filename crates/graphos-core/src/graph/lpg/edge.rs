//! Edge types for the LPG model.

use graphos_common::types::{EdgeId, EpochId, NodeId, PropertyKey, Value};
use std::collections::BTreeMap;
use std::sync::Arc;

/// An edge in the labeled property graph.
///
/// This is the high-level representation of an edge with all its data.
#[derive(Debug, Clone)]
pub struct Edge {
    /// Unique identifier.
    pub id: EdgeId,
    /// Source node ID.
    pub src: NodeId,
    /// Destination node ID.
    pub dst: NodeId,
    /// Edge type/label.
    pub edge_type: Arc<str>,
    /// Properties stored on this edge.
    pub properties: BTreeMap<PropertyKey, Value>,
}

impl Edge {
    /// Creates a new edge.
    #[must_use]
    pub fn new(id: EdgeId, src: NodeId, dst: NodeId, edge_type: impl Into<Arc<str>>) -> Self {
        Self {
            id,
            src,
            dst,
            edge_type: edge_type.into(),
            properties: BTreeMap::new(),
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

    /// Returns the other endpoint of this edge given one endpoint.
    ///
    /// Returns `None` if `node` is neither the source nor destination.
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

/// The compact representation of an edge.
///
/// This struct is used for edge storage with minimal overhead.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct EdgeRecord {
    /// Unique edge identifier.
    pub id: EdgeId,
    /// Source node ID.
    pub src: NodeId,
    /// Destination node ID.
    pub dst: NodeId,
    /// Edge type ID (index into type table).
    pub type_id: u32,
    /// Offset into the property arena.
    pub props_offset: u32,
    /// Number of properties.
    pub props_count: u16,
    /// Flags (deleted, has_version, etc.).
    pub flags: EdgeFlags,
    /// Epoch this record was created in.
    pub epoch: EpochId,
}

impl EdgeRecord {
    /// Flag indicating the edge is deleted.
    pub const FLAG_DELETED: u16 = 1 << 0;
    /// Flag indicating the edge has version history.
    pub const FLAG_HAS_VERSION: u16 = 1 << 1;

    /// Creates a new edge record.
    #[must_use]
    pub const fn new(id: EdgeId, src: NodeId, dst: NodeId, type_id: u32, epoch: EpochId) -> Self {
        Self {
            id,
            src,
            dst,
            type_id,
            props_offset: 0,
            props_count: 0,
            flags: EdgeFlags(0),
            epoch,
        }
    }

    /// Checks if this edge is deleted.
    #[must_use]
    pub const fn is_deleted(&self) -> bool {
        self.flags.contains(Self::FLAG_DELETED)
    }

    /// Marks this edge as deleted.
    pub fn set_deleted(&mut self, deleted: bool) {
        if deleted {
            self.flags.set(Self::FLAG_DELETED);
        } else {
            self.flags.clear(Self::FLAG_DELETED);
        }
    }
}

/// Flags for an edge record.
#[repr(transparent)]
#[derive(Debug, Clone, Copy, Default)]
pub struct EdgeFlags(pub u16);

impl EdgeFlags {
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
    fn test_edge_creation() {
        let edge = Edge::new(
            EdgeId::new(1),
            NodeId::new(10),
            NodeId::new(20),
            "KNOWS",
        );

        assert_eq!(edge.id, EdgeId::new(1));
        assert_eq!(edge.src, NodeId::new(10));
        assert_eq!(edge.dst, NodeId::new(20));
        assert_eq!(edge.edge_type.as_ref(), "KNOWS");
    }

    #[test]
    fn test_edge_properties() {
        let mut edge = Edge::new(
            EdgeId::new(1),
            NodeId::new(10),
            NodeId::new(20),
            "KNOWS",
        );

        edge.set_property("since", 2020i64);
        edge.set_property("weight", 1.5f64);

        assert_eq!(edge.get_property("since").and_then(|v| v.as_int64()), Some(2020));
        assert_eq!(edge.get_property("weight").and_then(|v| v.as_float64()), Some(1.5));
    }

    #[test]
    fn test_edge_other_endpoint() {
        let edge = Edge::new(
            EdgeId::new(1),
            NodeId::new(10),
            NodeId::new(20),
            "KNOWS",
        );

        assert_eq!(edge.other_endpoint(NodeId::new(10)), Some(NodeId::new(20)));
        assert_eq!(edge.other_endpoint(NodeId::new(20)), Some(NodeId::new(10)));
        assert_eq!(edge.other_endpoint(NodeId::new(30)), None);
    }

    #[test]
    fn test_edge_record_flags() {
        let mut record = EdgeRecord::new(
            EdgeId::new(1),
            NodeId::new(10),
            NodeId::new(20),
            0,
            EpochId::INITIAL,
        );

        assert!(!record.is_deleted());
        record.set_deleted(true);
        assert!(record.is_deleted());
    }

    #[test]
    fn test_edge_record_size() {
        // EdgeRecord should be a reasonable size for cache efficiency
        let size = std::mem::size_of::<EdgeRecord>();
        // Should be <= 64 bytes (one cache line)
        assert!(size <= 64, "EdgeRecord is {} bytes", size);
    }
}
