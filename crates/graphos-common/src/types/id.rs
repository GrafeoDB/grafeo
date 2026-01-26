//! Identifier types for graph elements and transactions.

use serde::{Deserialize, Serialize};
use std::fmt;

/// Unique identifier for a node in the graph.
///
/// Internally represented as a `u64`, allowing for up to 2^64 nodes.
/// NodeIds are assigned sequentially and are never reused within an epoch.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize, Default)]
#[repr(transparent)]
pub struct NodeId(pub u64);

impl NodeId {
    /// The invalid/null node ID.
    pub const INVALID: Self = Self(u64::MAX);

    /// Creates a new NodeId from a raw u64 value.
    #[inline]
    #[must_use]
    pub const fn new(id: u64) -> Self {
        Self(id)
    }

    /// Returns the raw u64 value.
    #[inline]
    #[must_use]
    pub const fn as_u64(self) -> u64 {
        self.0
    }

    /// Checks if this is a valid node ID.
    #[inline]
    #[must_use]
    pub const fn is_valid(self) -> bool {
        self.0 != u64::MAX
    }
}

impl fmt::Debug for NodeId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_valid() {
            write!(f, "NodeId({})", self.0)
        } else {
            write!(f, "NodeId(INVALID)")
        }
    }
}

impl fmt::Display for NodeId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl From<u64> for NodeId {
    fn from(id: u64) -> Self {
        Self(id)
    }
}

impl From<NodeId> for u64 {
    fn from(id: NodeId) -> Self {
        id.0
    }
}

/// Unique identifier for an edge in the graph.
///
/// Internally represented as a `u64`, allowing for up to 2^64 edges.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize, Default)]
#[repr(transparent)]
pub struct EdgeId(pub u64);

impl EdgeId {
    /// The invalid/null edge ID.
    pub const INVALID: Self = Self(u64::MAX);

    /// Creates a new EdgeId from a raw u64 value.
    #[inline]
    #[must_use]
    pub const fn new(id: u64) -> Self {
        Self(id)
    }

    /// Returns the raw u64 value.
    #[inline]
    #[must_use]
    pub const fn as_u64(self) -> u64 {
        self.0
    }

    /// Checks if this is a valid edge ID.
    #[inline]
    #[must_use]
    pub const fn is_valid(self) -> bool {
        self.0 != u64::MAX
    }
}

impl fmt::Debug for EdgeId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_valid() {
            write!(f, "EdgeId({})", self.0)
        } else {
            write!(f, "EdgeId(INVALID)")
        }
    }
}

impl fmt::Display for EdgeId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl From<u64> for EdgeId {
    fn from(id: u64) -> Self {
        Self(id)
    }
}

impl From<EdgeId> for u64 {
    fn from(id: EdgeId) -> Self {
        id.0
    }
}

/// Unique identifier for a transaction.
///
/// Transaction IDs are monotonically increasing and used for MVCC versioning.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize, Default)]
#[repr(transparent)]
pub struct TxId(pub u64);

impl TxId {
    /// The invalid/null transaction ID.
    pub const INVALID: Self = Self(0);

    /// Creates a new TxId from a raw u64 value.
    #[inline]
    #[must_use]
    pub const fn new(id: u64) -> Self {
        Self(id)
    }

    /// Returns the raw u64 value.
    #[inline]
    #[must_use]
    pub const fn as_u64(self) -> u64 {
        self.0
    }

    /// Returns the next transaction ID.
    #[inline]
    #[must_use]
    pub const fn next(self) -> Self {
        Self(self.0 + 1)
    }

    /// Checks if this is a valid transaction ID.
    #[inline]
    #[must_use]
    pub const fn is_valid(self) -> bool {
        self.0 != 0
    }
}

impl fmt::Debug for TxId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_valid() {
            write!(f, "TxId({})", self.0)
        } else {
            write!(f, "TxId(INVALID)")
        }
    }
}

impl fmt::Display for TxId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl From<u64> for TxId {
    fn from(id: u64) -> Self {
        Self(id)
    }
}

impl From<TxId> for u64 {
    fn from(id: TxId) -> Self {
        id.0
    }
}

/// Unique identifier for an epoch in the arena allocator.
///
/// Epochs are used for structural sharing in MVCC. Each write transaction
/// creates a new epoch, and old epochs can be garbage collected when no
/// readers reference them.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize, Default)]
#[repr(transparent)]
pub struct EpochId(pub u64);

impl EpochId {
    /// The initial epoch (epoch 0).
    pub const INITIAL: Self = Self(0);

    /// Creates a new EpochId from a raw u64 value.
    #[inline]
    #[must_use]
    pub const fn new(id: u64) -> Self {
        Self(id)
    }

    /// Returns the raw u64 value.
    #[inline]
    #[must_use]
    pub const fn as_u64(self) -> u64 {
        self.0
    }

    /// Returns the next epoch ID.
    #[inline]
    #[must_use]
    pub const fn next(self) -> Self {
        Self(self.0 + 1)
    }

    /// Checks if this epoch is visible at the given epoch.
    ///
    /// An epoch is visible if it was created before or at the viewing epoch.
    #[inline]
    #[must_use]
    pub const fn is_visible_at(self, viewing_epoch: Self) -> bool {
        self.0 <= viewing_epoch.0
    }
}

impl fmt::Debug for EpochId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "EpochId({})", self.0)
    }
}

impl fmt::Display for EpochId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl From<u64> for EpochId {
    fn from(id: u64) -> Self {
        Self(id)
    }
}

impl From<EpochId> for u64 {
    fn from(id: EpochId) -> Self {
        id.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_node_id_basic() {
        let id = NodeId::new(42);
        assert_eq!(id.as_u64(), 42);
        assert!(id.is_valid());
        assert!(!NodeId::INVALID.is_valid());
    }

    #[test]
    fn test_node_id_ordering() {
        let id1 = NodeId::new(1);
        let id2 = NodeId::new(2);
        assert!(id1 < id2);
    }

    #[test]
    fn test_edge_id_basic() {
        let id = EdgeId::new(100);
        assert_eq!(id.as_u64(), 100);
        assert!(id.is_valid());
        assert!(!EdgeId::INVALID.is_valid());
    }

    #[test]
    fn test_tx_id_basic() {
        let id = TxId::new(1);
        assert!(id.is_valid());
        assert!(!TxId::INVALID.is_valid());
        assert_eq!(id.next(), TxId::new(2));
    }

    #[test]
    fn test_epoch_visibility() {
        let e1 = EpochId::new(1);
        let e2 = EpochId::new(2);
        let e3 = EpochId::new(3);

        // e1 is visible at e2 and e3
        assert!(e1.is_visible_at(e2));
        assert!(e1.is_visible_at(e3));

        // e2 is visible at e2 and e3, but not e1
        assert!(!e2.is_visible_at(e1));
        assert!(e2.is_visible_at(e2));
        assert!(e2.is_visible_at(e3));

        // e3 is only visible at e3
        assert!(!e3.is_visible_at(e1));
        assert!(!e3.is_visible_at(e2));
        assert!(e3.is_visible_at(e3));
    }

    #[test]
    fn test_epoch_next() {
        let e = EpochId::INITIAL;
        assert_eq!(e.next(), EpochId::new(1));
        assert_eq!(e.next().next(), EpochId::new(2));
    }

    #[test]
    fn test_conversions() {
        // NodeId
        let node_id: NodeId = 42u64.into();
        let raw: u64 = node_id.into();
        assert_eq!(raw, 42);

        // EdgeId
        let edge_id: EdgeId = 100u64.into();
        let raw: u64 = edge_id.into();
        assert_eq!(raw, 100);

        // TxId
        let tx_id: TxId = 1u64.into();
        let raw: u64 = tx_id.into();
        assert_eq!(raw, 1);

        // EpochId
        let epoch_id: EpochId = 5u64.into();
        let raw: u64 = epoch_id.into();
        assert_eq!(raw, 5);
    }
}
